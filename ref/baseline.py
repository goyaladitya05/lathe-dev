# Benchmarks with shared timing and clock logging. Prints BENCH.md rows ("# <stack> OOM" when it does not fit).
#   python ref/baseline.py --mode eager|compile|lathe|tinygrad [--model smol135] --B 1 --T 2048   training step
#   python ref/baseline.py --mode sol                                  bf16 cuBLASLt GEMM speed of light
#   python ref/baseline.py --mode session --shapes 1x2048,4x512        sol, then every stack per shape
import argparse
import datetime
import math
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from model import DATA, ROOT, TARGET, TARGET_SUFFIX, Config, Model, get_batch, init_weights, make_optimizer

SMI_QUERY = "clocks.sm,clocks.mem,temperature.gpu,power.draw"


def sol_tflops():
    return float(re.search(r"SOL_BF16_TFLOPS = ([\d.]+)", (ROOT / f"src/targets/{TARGET}.h").read_text()).group(1))


def flops_per_token(cfg, T):
    # 6N over matmul params (tied lm_head counted once) + causal attention, fwd+bwd: 6 * L * (H*hd) * (T+1).
    hd = cfg.head_dim
    layer = cfg.dim * (cfg.heads + 2 * cfg.kv_heads) * hd + cfg.heads * hd * cfg.dim + 3 * cfg.dim * cfg.ffn
    n = cfg.layers * layer + cfg.vocab * cfg.dim
    return 6 * n + 6 * cfg.layers * cfg.heads * hd * (T + 1)


def free_mib():  # free GPU memory before a run allocates anything (what bounds the largest shape that fits)
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def proc_mem_mib(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        p, mem = line.split(",")
        if int(p) == pid:
            return int(mem)
    return -1


class Smi:
    # nvidia-smi sampled every 250 ms while the block runs.
    def __enter__(self):
        self.p = subprocess.Popen(["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits",
                                   "-lms", "250"], stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        rows = []
        for line in self.p.communicate()[0].splitlines():
            try:
                rows.append([float(v) for v in line.split(",")])
            except ValueError:
                pass
        sm, mem, temp, power = np.array(rows).T
        self.stats = (f"{np.median(sm):.0f} ({sm.min():.0f}-{sm.max():.0f}) | {np.median(mem):.0f} | "
                      f"{temp.max():.0f} | {power.mean():.1f}")


def commit():
    return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def train_row(stack, args, rates, mem, clocks, versions):
    tps, B, T = statistics.median(rates), args.B, args.T
    mfu = tps * flops_per_token(Config.load(args.model), T) / (sol_tflops() * 1e12)
    stack += "" if args.model == "smol135" else f" ({args.model})"
    print(f"| {datetime.date.today()} | {commit()} | {stack} | {B}x{T} | {tps:,.0f} | "
          f"{min(rates):,.0f}-{max(rates):,.0f} | {100 * mfu:.1f}% | {mem} | {clocks} | {versions} |")


def torch_train(args):
    free = free_mib()
    model = Model(Config.load(args.model))
    init_weights(model)
    model.cuda()
    opt = make_optimizer(model)
    fwd = torch.compile(model, mode="max-autotune") if args.mode == "compile" else model
    tokens = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    it = iter(range(1 << 62))

    def step():
        x, y = (t.pin_memory().cuda(non_blocking=True) for t in get_batch(tokens, next(it), args.B, args.T))
        if args.mode == "compile":
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    t0 = time.perf_counter()
    try:
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        print(f"# torch {args.mode} OOM at {args.model} {args.B}x{args.T}, free at start {free:,} MiB")
        sys.exit(3)
    except Exception as e:  # e.g. the first compile of a shape dying in CUDA graph capture (BENCH.md): rerun once
        print(f"# torch {args.mode} failed at {args.model} {args.B}x{args.T}: {type(e).__name__}: {str(e)[:200]}")
        sys.exit(3)
    warm = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()

    rates = []
    with Smi() as smi:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.steps):
                loss = step()
            torch.cuda.synchronize()
            rates.append(args.B * args.T * args.steps / (time.perf_counter() - t))
        pmem = proc_mem_mib(os.getpid())
    peak, reserved = torch.cuda.max_memory_allocated() >> 20, torch.cuda.max_memory_reserved() >> 20
    retries = torch.cuda.memory_stats()["num_alloc_retries"]
    print(f"# free at start {free:,} MiB, warmup {warm:.1f}s, last loss {loss.item():.4f}, alloc retries {retries}, runs {[round(r) for r in rates]}")
    train_row(f"torch {args.mode}", args, rates, f"{peak:,} | {reserved:,} | {pmem:,}", smi.stats,
              f"torch {torch.__version__}")


def sol(args):
    # Large square bf16 GEMMs, fp32 accumulate (no reduced-precision split-K), cuBLASLt backend.
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    for n in (4096, 8192, 16384):
        a, b = (torch.randn(n, n, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        c = torch.empty_like(a)
        iters = max(5, math.ceil(20e12 / (2 * n ** 3)))  # about 1 s per run
        rates = []
        with Smi() as smi:
            for r in range(args.runs + 2):  # first 2 runs are warmup
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    torch.mm(a, b, out=c)
                torch.cuda.synchronize()
                if r >= 2:
                    rates.append(2 * n ** 3 * iters / (time.perf_counter() - t) / 1e12)
        print(f"| {datetime.date.today()} | {commit()} | cuBLASLt bf16 GEMM | {n}^3 | {statistics.median(rates):.2f} | "
              f"{min(rates):.2f}-{max(rates):.2f} | {smi.stats} | torch {torch.__version__} |")
        del a, b, c


def engine_train(args, stack, cmd, env, versions):
    # A subprocess engine that prints lathe-style "group g steps a-b ms X tok/s Y" lines every --steps steps:
    # same warmup / runs / steps as torch, timed by the engine with one sync per group.
    assert args.warmup % args.steps == 0
    total = args.warmup + args.runs * args.steps
    free = free_mib()
    p = subprocess.Popen([*map(str, cmd), "--steps", str(total), "--group", str(args.steps)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, cwd=ROOT, env={**os.environ, **env})
    rates, smi, pmem, arena, loss, tail = [], Smi(), -1, "-", "?", []
    for line in p.stdout:
        tail = (tail + [line.strip()])[-20:]
        if m := re.search(r"arena=([\d.]+)", line):
            arena = f"{float(m.group(1)):,.0f}"
        elif line.startswith("step "):
            loss = line.split()[-1]
        elif m := re.match(r"group \d+ steps (\d+)-\d+ ms [\d.]+ tok/s (\d+)", line):
            if int(m.group(1)) >= args.warmup:
                rates.append(float(m.group(2)))
            elif int(m.group(1)) + args.steps == args.warmup:  # warmup done: start sampling
                smi.__enter__()
                pmem = proc_mem_mib(p.pid)
    if rates:
        smi.__exit__()
    if p.wait():
        oom = any(k in l.lower() for l in tail for k in ("out of memory", "outofmemory", "memoryerror"))
        print(f"# {stack} {'OOM' if oom else 'failed'} at {args.model} {args.B}x{args.T}, free at start {free:,} MiB: "
              f"{tail[-1] if tail else ''}")
        sys.exit(3)
    print(f"# free at start {free:,} MiB, last loss {loss}, runs {[round(r) for r in rates]}")
    train_row(stack, args, rates, f"{arena} | - | {pmem:,}", smi.stats, versions)


def lathe_train(args):
    flags = (["ADAM8=1"] if args.adam8 else []) + (["DW_LT=1"] if args.dw_lt else [])
    subprocess.run(["make", "-s", f"MODEL={args.model}", f"BATCH={args.B}", f"SEQ={args.T}", *flags], cwd=ROOT, check=True)
    exe = ROOT / f"build/train_{args.model}_b{args.B}_t{args.T}{'_adam8' if args.adam8 else ''}{'_dwlt' if args.dw_lt else ''}{TARGET_SUFFIX}"
    tags = [t for t, on in (("8-bit state", args.adam8), ("tuned gemms", args.tuned), ("cuBLASLt dW", args.dw_lt)) if on]
    engine_train(args, "lathe" + (f" ({', '.join(tags)})" if tags else ""),
                 [exe, "--data", DATA / "train.bin", *(["--tuned", 1] if args.tuned else [])], {}, "CUDA 12.8")


def tinygrad_train(args):
    # Own venv; CUDA backend with BEAM search (tinygrad's NV backend fails BEAM here, see BENCH.md).
    env = {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "DEV": "CUDA", "BEAM": "2"}
    engine_train(args, "tinygrad", [ROOT / ".venv-tinygrad/bin/python", ROOT / "ref/tinygrad_baseline.py", "--model",
                                    args.model, "--B", args.B, "--T", args.T], env, "tinygrad 0.14.0 BEAM=2")


def session(args):
    # One sitting, so lathe is always compared against baselines measured under the same clocks and thermals.
    run = lambda mode, *extra: subprocess.run([sys.executable, __file__, "--mode", mode, "--model", args.model, *extra])
    run("sol")
    for shape in args.shapes.split(","):
        B, T = shape.split("x")
        for mode in args.stacks.split(","):
            if run(mode, "--B", B, "--T", T).returncode and mode == "compile":
                run(mode, "--B", B, "--T", T)  # first compile of a shape can die in CUDA graph capture (BENCH.md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "compile", "lathe", "tinygrad", "sol", "session"], default="eager")
    ap.add_argument("--model", default="smol135")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20, help="steps per timed run")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--shapes", default="1x2048", help="session mode: comma-separated BxT")
    ap.add_argument("--stacks", default="eager,compile,lathe,tinygrad", help="session mode: stacks to run")
    ap.add_argument("--adam8", action="store_true", help="lathe mode: the 8-bit AdamW state build (make ADAM8=1)")
    ap.add_argument("--tuned", action="store_true", help="lathe mode: measured cuBLASLt algorithms (build/lt_cache.txt)")
    ap.add_argument("--dw-lt", action="store_true", help="lathe mode: cuBLASLt layer weight grads (make DW_LT=1)")
    args = ap.parse_args()
    {"sol": sol, "lathe": lathe_train, "tinygrad": tinygrad_train, "session": session}.get(args.mode, torch_train)(args)


if __name__ == "__main__":
    main()
