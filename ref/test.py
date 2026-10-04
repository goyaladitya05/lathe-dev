# Correctness of lathe against the reference.
# Per-kernel: each lathe kernel output vs fp32 math on lathe's own dumped inputs (bf16-rounded weights).
# End-to-end: lathe logits/loss vs the fp32 reference, measured against PyTorch bf16 autocast's own error.
#   python ref/test.py [B T]   (default 2 512)
import re
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from model import DATA, ROOT, Config, Model, get_batch, init_weights, lathe_layout, load_lathe, rope, rope_table

B, T = (int(v) for v in sys.argv[1:3]) if len(sys.argv) == 3 else (2, 512)
KERNEL_TOL = 2 ** -8  # rel L2 per kernel: one bf16 unit roundoff
E2E_RATIO = 2.0       # lathe error vs fp32 may be at most this times torch-bf16 error vs fp32
BIAS_BATCHES = 8
DT = {0: torch.bfloat16, 1: torch.float32, 2: torch.int32}
failures = []


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def check(name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<28} {err:.2e}  (tol {tol:.2e})")
    if not ok:
        failures.append(name)


def read_dump(path):
    raw, out, off = path.read_bytes(), {}, 0
    while off < len(raw):
        name = raw[off:off + 48].split(b"\0")[0].decode()
        dt, _, rows, cols = struct.unpack("<iiqq", raw[off + 48:off + 72])
        n = rows * cols * DT[dt].itemsize
        out[name] = torch.frombuffer(bytearray(raw[off + 72:off + 72 + n]), dtype=DT[dt]).view(rows, cols)
        off += 72 + n
    return out


def check_header(cfg):
    src = (ROOT / "src/models/smol135.h").read_text()
    h = {k: float(v) for k, v in re.findall(r"constexpr \w+ (\w+) = ([\d.e+-]+)f?;", src)}
    want = {"VOCAB": cfg.vocab, "DIM": cfg.dim, "LAYERS": cfg.layers, "HEADS": cfg.heads, "KV_HEADS": cfg.kv_heads,
            "FFN": cfg.ffn, "ROPE_THETA": cfg.rope_theta, "RMS_EPS": cfg.rms_eps}
    bad = {k: (h.get(k), v) for k, v in want.items() if h.get(k) != v}
    check("smol135.h == config.json", len(bad), 0)
    if bad:
        print("   ", bad)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def attention(qkv, cfg):
    hd = cfg.head_dim
    q, k, v = qkv.view(B, T, -1, hd).transpose(1, 2).split([cfg.heads, cfg.kv_heads, cfg.kv_heads], 1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).transpose(1, 2).reshape(B * T, -1)


def kernel_checks(d, model, x, y):
    cfg = model.cfg
    W = [t.to(torch.bfloat16).float().cuda() for t in lathe_layout(model.state_dict(), cfg)]  # lathe's bf16 weights
    cos, sin = rope_table(cfg, T, "cuda")
    rot = (cfg.heads + cfg.kv_heads) * cfg.head_dim
    for l in [0, cfg.layers - 1]:
        f = lambda name: d[f"L{l}.{name}"].float().cuda()
        wqkv, wo, wgu, wdown, ln1, ln2 = W[1 + 6 * l:7 + 6 * l]
        print(f" layer {l}")
        if l == 0:
            check("embed", rel(f("res"), W[0][x.view(-1).cuda()]), 0)
        check("rmsnorm 1", rel(f("ln1"), rmsnorm(f("res"), ln1, cfg.rms_eps)), KERNEL_TOL)
        check("matmul qkv", rel(f("qkv_pre"), f("ln1") @ wqkv.T), KERNEL_TOL)
        qk = f("qkv_pre")[:, :rot].view(B, T, -1, cfg.head_dim).transpose(1, 2)
        check("rope", rel(f("qkv")[:, :rot], rope(qk, cos, sin).transpose(1, 2).reshape(B * T, rot)), KERNEL_TOL)
        check("rope leaves v untouched", rel(f("qkv")[:, rot:], f("qkv_pre")[:, rot:]), 0)
        check("attention", rel(f("att"), attention(f("qkv"), cfg)), KERNEL_TOL)
        check("matmul o + residual", rel(f("res_mid"), f("res") + f("att") @ wo.T), KERNEL_TOL)
        check("rmsnorm 2", rel(f("ln2"), rmsnorm(f("res_mid"), ln2, cfg.rms_eps)), KERNEL_TOL)
        check("matmul gate|up", rel(f("gu"), f("ln2") @ wgu.T), KERNEL_TOL)
        g, u = f("gu").chunk(2, -1)
        check("swiglu", rel(f("act"), F.silu(g) * u), KERNEL_TOL)
        check("matmul down + residual", rel(f("res_out"), f("res_mid") + f("act") @ wdown.T), KERNEL_TOL)
    print(" output")
    lnf = d["lnf"].float().cuda()
    check("rmsnorm final", rel(lnf, rmsnorm(d[f"L{cfg.layers - 1}.res_out"].float().cuda(), W[-1], cfg.rms_eps)),
          KERNEL_TOL)
    check("matmul lm_head", rel(d["logits"].float().cuda(), lnf @ W[0].T), KERNEL_TOL)
    ce = F.cross_entropy(d["logits"].float().cuda(), y.view(-1).cuda(), reduction="none")
    check("cross entropy", rel(d["losses"].view(-1).cuda(), ce), 1e-6)


def e2e_checks(d, lathe_losses, model, tokens):
    # Accuracy on batch 0 (dumped logits); bias on mean loss over BIAS_BATCHES batches, tol = 3 standard errors.
    x, y = (t.cuda() for t in get_batch(tokens, 0, B, T))
    ce = lambda lg: F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1), reduction="none").double()
    with torch.no_grad():
        lg32 = model(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg16 = model(x)
        ref = [model(*(t.cuda() for t in get_batch(tokens, i, B, T)))[1].item() for i in range(BIAS_BATCHES)]
    lg = d["logits"].cuda().view_as(lg32)
    dl, dl16 = ce(lg) - ce(lg32), ce(lg16) - ce(lg32)
    print(f" end to end   batch 0 loss: fp32 {ce(lg32).mean():.6f}  torch-bf16 {ce(lg16).mean():.6f}  "
          f"lathe {ce(lg).mean():.6f}")
    print(f"   mean loss over {BIAS_BATCHES} batches: fp32 {np.mean(ref):.6f}  lathe {np.mean(lathe_losses):.6f}")
    check("logits rel err vs fp32", rel(lg, lg32), E2E_RATIO * rel(lg16, lg32))
    rms = lambda t: t.pow(2).mean().sqrt().item()
    check("per-token loss rms vs fp32", rms(dl), E2E_RATIO * rms(dl16))
    check("mean loss bias", abs(np.mean(lathe_losses) - np.mean(ref)), 3 * dl.std().item() / (BIAS_BATCHES * B * T) ** 0.5)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.load()
    print("setup")
    check_header(cfg)
    for name in ["init", "pretrained"]:
        if not (DATA / f"smol135_{name}.bin").exists():
            subprocess.run([sys.executable, ROOT / "ref/model.py", f"--{name}", DATA / f"smol135_{name}.bin"], check=True)
    a, b = Model(cfg), Model(cfg)
    init_weights(a)
    load_lathe(b, DATA / "smol135_init.bin")
    check("export/load round trip", max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters())), 0)
    subprocess.run(["make", "-s", f"BATCH={B}", f"SEQ={T}"], cwd=ROOT, check=True)

    tokens = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    x, y = get_batch(tokens, 0, B, T)
    for name in ["init", "pretrained"]:
        print(f"\n{name} weights")
        wpath, dpath = DATA / f"smol135_{name}.bin", ROOT / f"build/dump_{name}.bin"
        out = subprocess.run([ROOT / f"build/train_b{B}_t{T}", "--weights", wpath, "--data", DATA / "val.bin",
                              "--dump", dpath, "--batches", str(BIAS_BATCHES)], check=True, capture_output=True, text=True)
        lathe_losses = [float(v) for v in re.findall(r"batch \d+ loss ([\d.]+)", out.stdout)]
        d = read_dump(dpath)
        check("batch matches get_batch", float((d["tokens"] != torch.stack([x.view(-1), y.view(-1)])).sum()), 0)
        model = Model(cfg)
        load_lathe(model, wpath)
        model.cuda()
        kernel_checks(d, model, x, y)
        e2e_checks(d, lathe_losses, model, tokens)

    print("\nreference sanity: pretrained SmolLM2-135M val loss over 8 batches")
    model.eval()
    with torch.no_grad():
        loss = np.mean([model(*(t.cuda() for t in get_batch(tokens, i, B, T)))[1].item() for i in range(8)])
    check("pretrained val loss", loss, 3.5)

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all passed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
