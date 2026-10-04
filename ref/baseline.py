# Benchmarks with shared timing and clock logging. Prints BENCH.md rows.
#   python ref/baseline.py --mode eager|compile --B 1 --T 2048   training step: bf16 autocast, fp32 master, fused AdamW
#   python ref/baseline.py --mode sol                            bf16 cuBLASLt GEMM speed of light
import argparse
import datetime
import math
import os
import re
import statistics
import subprocess
import time

import numpy as np
import torch

from model import DATA, ROOT, Config, Model, get_batch, init_weights, make_optimizer

SMI_QUERY = "clocks.sm,clocks.mem,temperature.gpu,power.draw"


def sol_tflops():
    return float(re.search(r"SOL_BF16_TFLOPS = ([\d.]+)", (ROOT / "src/targets/sm89.h").read_text()).group(1))


def flops_per_token(cfg, T):
    # 6N over matmul params (tied lm_head counted once) + causal attention, fwd+bwd: 6 * L * (H*hd) * (T+1).
    hd = cfg.head_dim
    layer = cfg.dim * (cfg.heads + 2 * cfg.kv_heads) * hd + cfg.heads * hd * cfg.dim + 3 * cfg.dim * cfg.ffn
    n = cfg.layers * layer + cfg.vocab * cfg.dim
    return 6 * n + 6 * cfg.layers * cfg.heads * hd * (T + 1)


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


def train_row(stack, B, T, rates, mem, clocks, versions):
    tps = statistics.median(rates)
    mfu = tps * flops_per_token(Config.load(), T) / (sol_tflops() * 1e12)
    print(f"| {datetime.date.today()} | {commit()} | {stack} | {B}x{T} | {tps:,.0f} | "
          f"{min(rates):,.0f}-{max(rates):,.0f} | {100 * mfu:.1f}% | {mem} | {clocks} | {versions} |")


def torch_train(args):
    model = Model(Config.load())
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
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
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
    print(f"# warmup {warm:.1f}s, last loss {loss.item():.4f}, alloc retries {retries}, runs {[round(r) for r in rates]}")
    train_row(f"torch {args.mode}", args.B, args.T, rates, f"{peak:,} | {reserved:,} | {pmem:,}", smi.stats,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "compile", "sol"], default="eager")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20, help="steps per timed run")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    sol(args) if args.mode == "sol" else torch_train(args)


if __name__ == "__main__":
    main()
