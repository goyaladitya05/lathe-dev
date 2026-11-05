# Where a training step's GPU time goes, for lathe and torch.compile on the same shapes, plus the floors to compare with.
#   python ref/breakdown.py lathe   [--model smol135] --B 1 --T 2048 [--tuned] [--dw-lt] [--adam8]   nsys graph node trace
#   python ref/breakdown.py compile [--model smol135] --B 1 --T 2048                                  the same, torch.compile
#   python ref/breakdown.py attn    [--model smol135] --B 1 --T 2048     torch SDPA backends at lathe's attention shape
#   python ref/breakdown.py bw                                           DRAM copy bandwidth
#   python ref/breakdown.py analyse FILE.sqlite [--model smol135] --B 1 --T 2048 [--stack lathe]
# A trace is cut into steps at the one host-to-device batch copy each step starts with; the last --steps steps count.
# Per bucket: kernel ms per step. busy: ms per step with at least one kernel running (the union over streams).
import argparse
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from baseline import flops_per_token, sol_tflops
from model import DATA, ROOT, TARGET, TARGET_SUFFIX, Config, Model, get_batch, init_weights, make_optimizer

NSYS = os.environ.get("NSYS", "nsys")
# First match wins: lathe's own kernels by name, then library kernels by pattern.
BUCKETS = [
    ("attention", r"^attention_|flash|fmha|cudnn|sdpa|_attention|efficient_attention|attn"),
    ("fused gemm (qkv+rope, down dX+swiglu)", r"^(qkv_rope|down_dx_swiglu)\b"),
    ("own dW (+AdamW)", r"^(dw_bf16|head_dw_adamw|gemm_tt_)"),
    ("AdamW passes", r"^(adamw_|embed_bwd_adamw)|multi_tensor|fused_adam|_adam|adamw"),
    ("gemm (cuBLASLt / library)", r"gemm|xmma|cutlass|nvjet|^sm\d+_|^ampere_|^ada_|^Kernel2|splitK|cublas"),
    ("memory-bound passes", r"^(rmsnorm|rmsnorm_bwd|colsum|swiglu|swiglu_bwd|rope_qkv|embed|cross_entropy|mean_loss|"
                            r"f32_to_bf16|next_step|attention_dsum)\b"),
    ("triton fused (torch)", r"^triton_"),
]


def bucket(name):
    for b, pat in BUCKETS:
        if re.search(pat, name):
            return b
    return "other"


def analyse(path, model, B, T, steps, stack):
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    k = np.array(db.execute("SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall(),
                 dtype=np.int64).reshape(-1, 4)
    h2d = sorted(s for (s,) in db.execute("SELECT start FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1"))
    if len(h2d) < steps + 1:
        sys.exit(f"{path}: {len(h2d)} host-to-device copies, need {steps + 1}")
    lo, hi = h2d[-steps - 1], h2d[-1]
    print(f"# {path}: {len(h2d)} host-to-device copies, counting the last {steps} steps")
    k = k[(k[:, 0] >= lo) & (k[:, 0] < hi)]
    k = k[np.argsort(k[:, 0])]
    busy, end = 0, lo  # union of kernel intervals
    for s, e in k[:, :2]:
        if e > end:
            busy += e - max(s, end)
            end = e
    wall = (hi - lo) / steps / 1e6
    dur = (k[:, 1] - k[:, 0]) / steps / 1e6
    per_b, per_k, per_s = {}, {}, {}
    for d, sid, n in zip(dur, k[:, 2], k[:, 3]):
        name = names[int(n)]
        b = bucket(name)
        per_b[b] = per_b.get(b, 0) + d
        c = per_k.setdefault(name, [0.0, 0, b])
        c[0] += d
        c[1] += 1
        per_s[int(sid)] = per_s.get(int(sid), 0) + d
    tok = B * T / wall * 1e3
    mfu = tok * flops_per_token(Config.load(model), T) / (sol_tflops() * 1e12)
    print(f"profile | {stack} | {model} {B}x{T} | {steps} steps | wall {wall:.3f} ms/step ({tok:,.0f} tok/s, MFU "
          f"{mfu:.1%}) | kernel sum {dur.sum():.3f} | busy {busy / steps / 1e6:.3f} ({busy / steps / 1e6 / wall:.1%}) "
          f"| idle {wall - busy / steps / 1e6:.3f}")
    for b, ms in sorted(per_b.items(), key=lambda x: -x[1]):
        print(f"profile_bucket | {b} | {ms:.3f} | {ms / wall:.1%}")
    for sid, ms in sorted(per_s.items(), key=lambda x: -x[1]):
        print(f"profile_stream | {sid} | {ms:.3f}")
    for name, (ms, n, b) in sorted(per_k.items(), key=lambda x: -x[1][0])[:25]:
        print(f"profile_kernel | {ms:.3f} | {n / steps:g}/step | {b} | {name[:110]}")


def nsys(cmd, out, extra=()):
    subprocess.run([NSYS, "profile", "-t", "cuda", "-s", "none", "--cpuctxsw=none", "--cuda-graph-trace=node",
                    "--export=sqlite", "-f", "true", "-o", str(out), *extra, *map(str, cmd)], check=True,
                   stdout=subprocess.DEVNULL)
    return Path(f"{out}.sqlite")


def lathe(args, tmp):
    flags = (["ADAM8=1"] if args.adam8 else []) + (["DW_LT=1"] if args.dw_lt else [])
    subprocess.run(["make", "-s", f"MODEL={args.model}", f"BATCH={args.B}", f"SEQ={args.T}", *flags], cwd=ROOT, check=True)
    exe = ROOT / (f"build/train_{args.model}_b{args.B}_t{args.T}{'_adam8' if args.adam8 else ''}"
                  f"{'_dwlt' if args.dw_lt else ''}{TARGET_SUFFIX}")
    cmd = [exe, "--data", DATA / "train.bin", "--steps", args.warmup + args.steps, *(["--tuned", 1] if args.tuned else [])]
    tags = [t for t, on in (("8-bit state", args.adam8), ("tuned gemms", args.tuned), ("cuBLASLt dW", args.dw_lt)) if on]
    analyse(nsys(cmd, tmp / "lathe"), args.model, args.B, args.T, args.steps,
            "lathe" + (f" ({', '.join(tags)})" if tags else ""))


def compile_inner(args):  # runs under nsys: warm up, then trace only the counted steps
    model = Model(Config.load(args.model))
    init_weights(model)
    model.cuda()
    opt = make_optimizer(model)
    fwd = torch.compile(model, mode="max-autotune")
    tokens = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")

    def step(i):
        xy = torch.stack(get_batch(tokens, i, args.B, args.T)).pin_memory().cuda(non_blocking=True)  # one copy per step
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(xy[0], xy[1])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for i in range(args.warmup):
        step(i)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for i in range(args.steps + 1):  # one extra step closes the last counted one
        step(args.warmup + i)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def compile_(args, tmp):
    cmd = [sys.executable, __file__, "compile-inner", "--model", args.model, "--B", args.B, "--T", args.T,
           "--warmup", args.warmup, "--steps", args.steps]
    path = nsys(cmd, tmp / "compile", ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"])
    analyse(path, args.model, args.B, args.T, args.steps, f"torch compile ({torch.__version__})")


def timed(fn, iters=20):
    # GPU kernel ms per call (torch.profiler), so launch gaps do not count, as in the step traces.
    from torch.profiler import ProfilerActivity, profile

    import time

    t = time.perf_counter()
    while time.perf_counter() - t < 1:  # about a second of warm-up, so clocks have ramped
        fn()
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages() if e.device_type.name == "CUDA") / iters / 1e3


def attn(args):
    # One layer's attention at lathe's shape: q [B, H, T, d], k and v [B, KV, T, d], causal, bf16.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F

    cfg = Config.load(args.model)
    B, H, KV, T, d = args.B, cfg.heads, cfg.kv_heads, args.T, cfg.head_dim
    q = torch.randn(B, H, T, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = (torch.randn(B, KV, T, d, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2))
    dout = torch.randn(B, H, T, d, device="cuda", dtype=torch.bfloat16)
    fwd_flops = 2 * B * H * d * T * (T + 1)  # QK^T and PV over the causal half, as in flops_per_token
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION),
                     ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        for gqa in (True, False):  # native GQA, else K and V repeated to H heads (the copy is not timed)
            kk, vv = (k, v) if gqa else (x.detach().repeat_interleave(H // KV, 1).requires_grad_() for x in (k, v))
            run = lambda: F.scaled_dot_product_attention(q, kk, vv, is_causal=True, enable_gqa=gqa)
            try:
                with sdpa_kernel(be):
                    f = timed(lambda: run())
                    fb = timed(lambda: torch.autograd.grad(run(), (q, kk, vv), dout))
            except RuntimeError as e:
                print(f"# attn {name} gqa={gqa}: {str(e).splitlines()[0][:120]}")
                continue
            print(f"attn | {name}{'' if gqa else ' (kv repeated)'} | {args.model} {B}x{T} H={H} KV={KV} d={d} | fwd "
                  f"{f:.4f} ms ({fwd_flops / f / 1e9:.1f} TFLOPS) | fwd+bwd {fb:.4f} ms ({3 * fwd_flops / fb / 1e9:.1f} "
                  f"TFLOPS) | x{cfg.layers} layers {fb * cfg.layers:.3f} ms/step")
            break


def bw(args):
    x = torch.empty(1 << 29, dtype=torch.uint8, device="cuda")  # 512 MiB, far past L2
    y = torch.empty_like(x)
    ms = timed(lambda: y.copy_(x))
    print(f"bw | {TARGET} | copy 512 MiB | {2 * x.numel() / ms / 1e6:.1f} GB/s (read + write)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lathe", "compile", "compile-inner", "attn", "bw", "analyse"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--model", default="smol135")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20, help="counted steps")
    ap.add_argument("--stack", default="lathe", help="analyse mode: label")
    ap.add_argument("--adam8", action="store_true")
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--dw-lt", action="store_true")
    args = ap.parse_args()
    if args.mode == "analyse":
        return analyse(args.file, args.model, args.B, args.T, args.steps, args.stack)
    if args.mode in ("compile-inner", "attn", "bw"):
        return {"compile-inner": compile_inner, "attn": attn, "bw": bw}[args.mode](args)
    with tempfile.TemporaryDirectory() as tmp:
        {"lathe": lathe, "compile": compile_}[args.mode](args, Path(tmp))


if __name__ == "__main__":
    main()
