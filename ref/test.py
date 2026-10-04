# Correctness of lathe against the reference.
#   python ref/test.py [B T]              kernels fwd/bwd, end to end, every parameter grad, AdamW (default 2 512)
#   python ref/test.py curve [steps]      loss curve vs reference training at 1x2048 (default 1000 steps)
# Per-kernel checks feed each lathe kernel's dumped inputs to fp32 math (bf16-rounded matmul weights).
# End-to-end checks measure lathe against an fp32 reference, with PyTorch bf16 autocast's own error as the
# yardstick: lathe may be at most E2E_RATIO times worse.
import re
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from model import (DATA, ROOT, Config, Model, from_lathe, get_batch, init_weights, lathe_layout, load_lathe,
                   make_optimizer, rope, rope_table)

KERNEL_TOL = 2 ** -8  # rel L2 for kernels with bf16 outputs: one bf16 unit roundoff
F32_TOL = 1e-4        # rel L2 for kernels with fp32 outputs: summation-order noise (K*eps = 1.2e-4 at K = 2048)
E2E_RATIO = 2.0
MEDIAN_RATIO = 1.25
BIAS_BATCHES = 8
CURVE_EARLY, CURVE_WINDOW_TOL, CURVE_FINAL_TOL = 50, 0.06, 0.03  # see curve()
failures = []


def rel(a, b):
    # ||a - b|| / ||b|| in fp64, chunked so 100M-element tensors fit in memory.
    num = den = 0.0
    for x, y in zip(a.reshape(-1).split(1 << 24), b.reshape(-1).split(1 << 24)):
        x, y = x.double(), y.double()
        num += (x - y).pow(2).sum().item()
        den += y.pow(2).sum().item()
    return (num / den) ** 0.5


def check(name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<34} {err:.2e}  (tol {tol:.2e})")
    if not ok:
        failures.append(name)


def read_dump(path):
    raw = np.memmap(path, dtype=np.uint8, mode="c")
    out, off = {}, 0
    while off < len(raw):
        name = bytes(raw[off:off + 48]).split(b"\0")[0].decode()
        dt, _, rows, cols = struct.unpack("<iiqq", bytes(raw[off + 48:off + 72]))
        npdt = {0: np.int16, 1: np.float32, 2: np.int32}[dt]
        n = rows * cols * np.dtype(npdt).itemsize
        t = torch.from_numpy(raw[off + 72:off + 72 + n].view(npdt).reshape(rows, cols))
        out[name] = t.view(torch.bfloat16) if dt == 0 else t
        off += 72 + n
    return out


def chunks(d, name):
    # lathe dumps the lm_head logits / dlogits one vocab chunk at a time.
    n = sum(k.startswith(name + ".") for k in d)
    return torch.cat([d[f"{name}.{c}"] for c in range(n)], 1)


def run_lathe(B, T, *args):
    out = subprocess.run([ROOT / f"build/train_b{B}_t{T}", *map(str, args)], capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"lathe failed: {' '.join(map(str, args))}\n{out.stderr}")
    return out.stdout


def losses_from(stdout, kind):
    return [float(v) for v in re.findall(rf"{kind} \d+ loss ([\d.]+)", stdout)]


def check_header(cfg):
    src = (ROOT / "src/models/smol135.h").read_text()
    h = {k: float(v) for k, v in re.findall(r"constexpr \w+ (\w+) = ([\d.e+-]+)f?;", src)}
    want = {"VOCAB": cfg.vocab, "DIM": cfg.dim, "LAYERS": cfg.layers, "HEADS": cfg.heads, "KV_HEADS": cfg.kv_heads,
            "FFN": cfg.ffn, "ROPE_THETA": cfg.rope_theta, "RMS_EPS": cfg.rms_eps}
    bad = {k: (h.get(k), v) for k, v in want.items() if h.get(k) != v}
    check("smol135.h == config.json", len(bad), 0)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rmsnorm_grads(x, w, dy, eps):
    # fp32 autograd of the reference RMSNorm: (dx, dw).
    x, w = x.clone().requires_grad_(), w.clone().requires_grad_()
    rmsnorm(x, w, eps).backward(dy)
    return x.grad, w.grad


def attention(qkv, cfg, B, T):
    q, k, v = qkv.view(B, T, -1, cfg.head_dim).transpose(1, 2).split([cfg.heads, cfg.kv_heads, cfg.kv_heads], 1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).transpose(1, 2).reshape(B * T, -1)


def attention_bwd(qkv, o, do, cfg, B, T):
    # fp32 attention backward from the kernel's inputs; D = rowsum(dO * O) uses the stored O, as flash kernels do.
    hd, G = cfg.head_dim, cfg.heads // cfg.kv_heads
    q, k, v = qkv.view(B, T, -1, hd).transpose(1, 2).split([cfg.heads, cfg.kv_heads, cfg.kv_heads], 1)
    k, v = k.repeat_interleave(G, 1), v.repeat_interleave(G, 1)
    o, do = (t.view(B, T, cfg.heads, hd).transpose(1, 2) for t in (o, do))
    mask = torch.ones(T, T, dtype=torch.bool, device=qkv.device).triu(1)
    p = (q @ k.transpose(-1, -2) / hd ** 0.5).masked_fill(mask, float("-inf")).softmax(-1)
    ds = p * (do @ v.transpose(-1, -2) - (do * o).sum(-1, keepdim=True))
    fold = lambda t: t.view(B, cfg.kv_heads, G, T, hd).sum(2)  # sum over the query heads sharing a kv head
    flat = lambda t: t.transpose(1, 2).reshape(B * T, -1)
    return flat(ds @ k / hd ** 0.5), flat(fold(ds.transpose(-1, -2) @ q / hd ** 0.5)), flat(fold(p.transpose(-1, -2) @ do))


def bf(t):
    return t.to(torch.bfloat16).float()


class Weights:
    # Lathe-order parameters: fp32 (norms, embedding lookup) and bf16-rounded (matmul inputs), per layer.
    def __init__(self, model):
        self.f = [t.float().cuda() for t in lathe_layout(model.state_dict(), model.cfg)]
        self.b = [bf(t) for t in self.f]

    def layer(self, l, kind):  # kind "f" or "b": [qkv, o, gu, down, ln1, ln2]
        return getattr(self, kind)[1 + 6 * l:7 + 6 * l]


def forward_kernel_checks(d, W, x, y, cfg, B, T):
    cos, sin = rope_table(cfg, T, "cuda")
    rot = (cfg.heads + cfg.kv_heads) * cfg.head_dim
    for l in [0, cfg.layers - 1]:
        f = lambda name: d[f"L{l}.{name}"].float().cuda()
        wqkv, wo, wgu, wdown, _, _ = W.layer(l, "b")
        *_, ln1, ln2 = W.layer(l, "f")
        print(f" forward layer {l}")
        if l == 0:
            check("embed", rel(f("res"), W.f[0][x.view(-1).cuda()]), 0)
        check("rmsnorm 1", rel(f("ln1"), rmsnorm(f("res"), ln1, cfg.rms_eps)), KERNEL_TOL)
        check("matmul qkv", rel(f("qkv_pre"), f("ln1") @ wqkv.T), KERNEL_TOL)
        qk = f("qkv_pre")[:, :rot].view(B, T, -1, cfg.head_dim).transpose(1, 2)
        check("rope", rel(f("qkv")[:, :rot], rope(qk, cos, sin).transpose(1, 2).reshape(B * T, rot)), KERNEL_TOL)
        check("rope leaves v untouched", rel(f("qkv")[:, rot:], f("qkv_pre")[:, rot:]), 0)
        check("attention", rel(f("att"), attention(f("qkv"), cfg, B, T)), KERNEL_TOL)
        check("matmul o", rel(f("ao"), f("att") @ wo.T), KERNEL_TOL)
        check("residual add 1 (fp32)", rel(f("res_mid"), f("res") + f("ao")), 0)
        check("rmsnorm 2", rel(f("ln2"), rmsnorm(f("res_mid"), ln2, cfg.rms_eps)), KERNEL_TOL)
        check("matmul gate|up", rel(f("gu"), f("ln2") @ wgu.T), KERNEL_TOL)
        g, u = d[f"L{l}.gu"].cuda().chunk(2, -1)
        check("swiglu (vs torch bf16 ops)", rel(f("act"), (F.silu(g) * u).float()), KERNEL_TOL)
        check("matmul down", rel(f("mo"), f("act") @ wdown.T), KERNEL_TOL)
        check("residual add 2 (fp32)", rel(f("res_out"), f("res_mid") + f("mo")), 0)
    print(" forward output")
    lnf = d["lnf"].float().cuda()
    res_l = d[f"L{cfg.layers - 1}.res_out"].float().cuda()
    check("rmsnorm final", rel(lnf, rmsnorm(res_l, W.f[-1], cfg.rms_eps)), KERNEL_TOL)
    check("matmul lm_head", rel(chunks(d, "logits").float().cuda(), lnf @ W.b[0].T), KERNEL_TOL)
    ce = F.cross_entropy(chunks(d, "logits").float().cuda(), y.view(-1).cuda(), reduction="none")
    check("cross entropy", rel(d["losses"].view(-1).cuda(), ce), F32_TOL)


def backward_kernel_checks(d, grads, W, x, y, cfg, B, T):
    cos, sin = rope_table(cfg, T, "cuda")
    rot = (cfg.heads + cfg.kv_heads) * cfg.head_dim
    eps = cfg.rms_eps
    logits, lnf = chunks(d, "logits").float().cuda(), d["lnf"].float().cuda()
    dlogits = chunks(d, "dlogits").float().cuda()
    onehot = F.one_hot(y.view(-1).cuda(), cfg.vocab).float()
    print(" backward output")
    check("cross entropy bwd", rel(dlogits, (logits.softmax(-1) - onehot) / (B * T)), KERNEL_TOL)
    check("matmul lm_head dX (fp32)", rel(d["dlnf"].float().cuda(), dlogits @ W.b[0]), F32_TOL)
    L = cfg.layers - 1
    dx, dw = rmsnorm_grads(d[f"L{L}.res_out"].float().cuda(), W.f[-1], d["dlnf"].float().cuda(), eps)
    check("rmsnorm final bwd dx", rel(d[f"L{L}.dres_out"].cuda(), dx), F32_TOL)
    check("rmsnorm final bwd dw", rel(grads["norm.weight"].cuda(), dw), F32_TOL)
    for l in [L, 0]:
        f = lambda name: d[f"L{l}.{name}"].float().cuda()
        p = f"layers.{l}."
        G = lambda name: grads[p + name].cuda()
        wqkv, wo, wgu, wdown, _, _ = W.layer(l, "b")
        *_, ln1, ln2 = W.layer(l, "f")
        print(f" backward layer {l}")
        dmo = bf(f("dres_out"))
        check("matmul down dX", rel(f("dact"), dmo @ wdown), KERNEL_TOL)
        check("matmul down dW (fp32)", rel(G("mlp.down_proj.weight"), dmo.T @ f("act")), F32_TOL)
        g, u = (t.clone().requires_grad_() for t in d[f"L{l}.gu"].cuda().chunk(2, -1))
        (F.silu(g) * u).backward(d[f"L{l}.dact"].cuda())
        check("swiglu bwd (vs torch bf16 ops)", rel(f("dgu"), torch.cat([g.grad, u.grad], -1).float()), KERNEL_TOL)
        check("matmul gate|up dX", rel(f("dln2"), f("dgu") @ wgu), KERNEL_TOL)
        gate, up = (f("dgu").T @ f("ln2")).chunk(2)
        check("matmul gate|up dW (fp32)", rel(torch.cat([G("mlp.gate_proj.weight"), G("mlp.up_proj.weight")]),
                                               torch.cat([gate, up])), F32_TOL)
        dx, dw = rmsnorm_grads(f("res_mid"), ln2, f("dln2"), eps)
        check("rmsnorm 2 bwd dx + residual", rel(f("dres_mid"), f("dres_out") + dx), F32_TOL)
        check("rmsnorm 2 bwd dw", rel(G("post_attention_layernorm.weight"), dw), F32_TOL)
        dao = bf(f("dres_mid"))
        check("matmul o dX", rel(f("datt"), dao @ wo), KERNEL_TOL)
        check("matmul o dW (fp32)", rel(G("self_attn.o_proj.weight"), dao.T @ f("att")), F32_TOL)
        hq, hk = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
        dq, dk, dv = attention_bwd(f("qkv"), f("att"), f("datt"), cfg, B, T)
        unrope = lambda t: rope(t.view(B, T, -1, cfg.head_dim).transpose(1, 2), cos, -sin).transpose(1, 2).reshape(B * T, -1)
        for name, got, w in zip(["dq", "dk", "dv"], f("dqkv_pre").split([hq, hk, hk], -1), [unrope(dq), unrope(dk), dv]):
            check(f"attention + rope bwd {name}", rel(got, w), KERNEL_TOL)
        check("matmul qkv dX", rel(f("dln1"), f("dqkv_pre") @ wqkv), KERNEL_TOL)
        q, k, v = (f("dqkv_pre").T @ f("ln1")).split([hq, hk, hk])
        check("matmul qkv dW (fp32)", rel(torch.cat([G("self_attn.q_proj.weight"), G("self_attn.k_proj.weight"),
                                                     G("self_attn.v_proj.weight")]), torch.cat([q, k, v])), F32_TOL)
        dx, dw = rmsnorm_grads(f("res"), ln1, f("dln1"), eps)
        check("rmsnorm 1 bwd dx + residual", rel(f("dres_in"), f("dres_mid") + dx), F32_TOL)
        check("rmsnorm 1 bwd dw", rel(G("input_layernorm.weight"), dw), F32_TOL)
    print(" backward embedding")
    gemb = dlogits.T @ lnf
    gemb.index_add_(0, x.view(-1).cuda(), d["L0.dres_in"].cuda())
    check("embedding grad (lm_head dW + lookup)", rel(grads["embed_tokens.weight"].cuda(), gemb), F32_TOL)


def e2e_checks(d, lathe_losses, model, tokens, B, T):
    # Accuracy on batch 0 (dumped logits); bias on mean loss over BIAS_BATCHES batches, tol = 3 standard errors.
    x, y = (t.cuda() for t in get_batch(tokens, 0, B, T))
    ce = lambda lg: F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1), reduction="none").double()
    with torch.no_grad():
        lg32 = model(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg16 = model(x)
        ref = [model(*(t.cuda() for t in get_batch(tokens, i, B, T)))[1].item() for i in range(BIAS_BATCHES)]
    lg = chunks(d, "logits").cuda().view_as(lg32)
    dl, dl16 = ce(lg) - ce(lg32), ce(lg16) - ce(lg32)
    print(f" end to end   batch 0 loss: fp32 {ce(lg32).mean():.6f}  torch-bf16 {ce(lg16).mean():.6f}  "
          f"lathe {ce(lg).mean():.6f}")
    print(f"   mean loss over {BIAS_BATCHES} batches: fp32 {np.mean(ref):.6f}  lathe {np.mean(lathe_losses):.6f}")
    check("logits rel err vs fp32", rel(lg, lg32), E2E_RATIO * rel(lg16, lg32))
    rms = lambda t: t.pow(2).mean().sqrt().item()
    check("per-token loss rms vs fp32", rms(dl), E2E_RATIO * rms(dl16))
    check("mean loss bias", abs(np.mean(lathe_losses) - np.mean(ref)), 3 * dl.std().item() / (BIAS_BATCHES * B * T) ** 0.5)


def ref_grads(model, x, y, bf16):
    model.zero_grad(set_to_none=True)
    model.ckpt = not bf16
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        _, loss = model(x, y)
    loss.backward()
    model.ckpt = False
    return {k: p.grad.cpu() for k, p in model.named_parameters()}


def grad_checks(grads, model, x, y, per_tensor):
    # Every parameter: lathe vs fp32 reference, measured against torch-bf16's error vs fp32.
    # per_tensor: each tensor within E2E_RATIO (random init: errors are stable, ratios ~1).
    # Otherwise the per-kind median over layers within MEDIAN_RATIO: with pretrained weights, single tensors
    # near massive activations swing several x between equally valid bf16 implementations (rounding luck after
    # catastrophic cancellation), while the median over layers is stable.
    x, y = x.cuda(), y.cuda()
    g32, g16 = ref_grads(model, x, y, False), ref_grads(model, x, y, True)
    model.zero_grad(set_to_none=True)
    groups = {}
    for k in g32:
        kind = re.sub(r"layers\.\d+\.", "", k).removesuffix(".weight")
        groups.setdefault(kind, []).append((rel(grads[k], g32[k]), rel(g16[k], g32[k]), k))
    print(f" every parameter grad ({len(g32)} tensors) vs fp32: lathe / torch-bf16 error ratio by kind")
    for kind, rows in groups.items():
        ratios = [e_l / e_t for e_l, e_t, _ in rows]
        if per_tensor:
            e_l, e_t, k = rows[int(np.argmax(ratios))]
            check(f"{kind} ({len(rows)}) worst", e_l, E2E_RATIO * e_t)
        else:
            check(f"{kind} ({len(rows)}) median ratio", float(np.median(ratios)), MEDIAN_RATIO)
        print(f"         ratio median {np.median(ratios):.2f}  max {max(ratios):.2f}")


def adamw_check(d, grads, wpath, cfg):
    # One step of PyTorch fused AdamW on lathe's grads must reproduce lathe's updated weights.
    model = Model(cfg)
    load_lathe(model, wpath)
    model.cuda()
    opt = make_optimizer(model)
    for k, p in model.named_parameters():
        p.grad = grads[k].cuda()
    opt.step()
    want = torch.cat([t.reshape(-1).cpu() for t in lathe_layout(model.state_dict(), cfg)])
    master = d["master"].view(-1)
    print(" optimizer")
    check("AdamW step vs torch fused AdamW", rel(master, want), 1e-6)
    check("bf16 copy == bf16(master)", float((d["params"].view(-1) != master.to(torch.bfloat16)).sum()), 0)


def unit(B, T):
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

    # Run lathe before this process touches the GPU, so its arena never competes with torch for memory.
    val, evals = DATA / "val.bin", {}
    for name in ["init", "pretrained"]:
        w = DATA / f"smol135_{name}.bin"
        run_lathe(B, T, "--weights", w, "--data", val, "--steps", 1, "--dump", ROOT / f"build/dump_{name}.bin")
        evals[name] = losses_from(run_lathe(B, T, "--weights", w, "--data", val, "--eval", BIAS_BATCHES), "batch")

    tokens = np.memmap(val, dtype=np.uint16, mode="r")
    x, y = get_batch(tokens, 0, B, T)
    for name in ["init", "pretrained"]:
        print(f"\n{name} weights")
        wpath = DATA / f"smol135_{name}.bin"
        d = read_dump(ROOT / f"build/dump_{name}.bin")
        check("batch matches get_batch", float((d["tokens"] != torch.stack([x.view(-1), y.view(-1)])).sum()), 0)
        model = Model(cfg)
        load_lathe(model, wpath)
        model.cuda()
        W = Weights(model)
        grads = from_lathe(d["grads"].view(-1), cfg)
        forward_kernel_checks(d, W, x, y, cfg, B, T)
        e2e_checks(d, evals[name], model, tokens, B, T)
        backward_kernel_checks(d, grads, W, x, y, cfg, B, T)
        grad_checks(grads, model, x, y, per_tensor=name == "init")
        del W
        adamw_check(d, grads, wpath, cfg)

    print("\nreference sanity: pretrained SmolLM2-135M val loss over 8 batches")
    model.eval()
    with torch.no_grad():
        loss = np.mean([model(*(t.cuda() for t in get_batch(tokens, i, B, T)))[1].item() for i in range(8)])
    check("pretrained val loss", loss, 3.5)


def curve(steps, B=1, T=2048, window=100):
    # lathe, torch-bf16 and fp32 (checkpointed) train from the same init on the same batches.
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = Config.load()
    subprocess.run(["make", "-s", f"BATCH={B}", f"SEQ={T}"], cwd=ROOT, check=True)
    train = DATA / "train.bin"
    out = run_lathe(B, T, "--weights", DATA / "smol135_init.bin", "--data", train, "--steps", steps)
    curves = {"lathe": np.array(losses_from(out, "step"))}
    tokens = np.memmap(train, dtype=np.uint16, mode="r")
    for name, bf16 in [("torch-bf16", True), ("fp32", False)]:
        model = Model(cfg)
        init_weights(model)
        model.cuda()
        model.ckpt = not bf16
        opt = make_optimizer(model)
        losses = []
        for i in range(steps):
            x, y = (t.cuda() for t in get_batch(tokens, i, B, T))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
        curves[name] = torch.stack(losses).cpu().numpy()
        del model, opt
        torch.cuda.empty_cache()
    f32, t16, la = curves["fp32"], curves["torch-bf16"], curves["lathe"]
    np.savetxt(ROOT / f"build/curve_b{B}_t{T}.csv", np.stack([np.arange(steps), f32, t16, la], 1), delimiter=",",
               header="step,fp32,torch_bf16,lathe", fmt="%.6f")
    print(f"{'steps':<10} {'fp32':>8} {'torch-bf16':>10} {'lathe':>8} | mean |torch-fp32| |lathe-fp32| |lathe-torch|")
    for s0 in range(0, steps, window):
        w = slice(s0, s0 + window)
        print(f"{s0:4d}-{min(s0 + window, steps) - 1:<5d} {f32[w].mean():8.4f} {t16[w].mean():10.4f} {la[w].mean():8.4f} | "
              f"{np.abs(t16[w] - f32[w]).mean():17.4f} {np.abs(la[w] - f32[w]).mean():12.4f} {np.abs(la[w] - t16[w]).mean():12.4f}")
    # bf16 runs track fp32 closely until the loss transition (steps ~50-100). After it, equally valid bf16
    # implementations follow nearby trajectories that slowly reconverge: torch-bf16 with flash, cuDNN or
    # mem-efficient attention, over init seeds 0 and 1, reached window offsets of 0.036 and late offsets of
    # 0.018, with the sign set by the seed. So: a sharp check before the transition, coarse ones (1.7x) after.
    early = slice(0, CURVE_EARLY)
    check(f"steps 0-{CURVE_EARLY - 1} mean |loss - fp32|", np.abs(la[early] - f32[early]).mean(),
          E2E_RATIO * np.abs(t16[early] - f32[early]).mean())
    offsets = [abs(la[a:a + window].mean() - f32[a:a + window].mean()) for a in range(0, steps, window)]
    check(f"worst {window}-step window |mean - fp32|", max(offsets), CURVE_WINDOW_TOL)
    check("last 200 steps |mean - fp32|", abs(la[-200:].mean() - f32[-200:].mean()), CURVE_FINAL_TOL)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "curve":
        curve(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    else:
        unit(*((int(v) for v in sys.argv[1:3]) if len(sys.argv) == 3 else (2, 512)))
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all passed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
