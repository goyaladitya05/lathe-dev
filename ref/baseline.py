# Training-step throughput of the reference model: PyTorch eager or torch.compile(max-autotune).
# bf16 autocast, fp32 master weights, fused AdamW. Prints one BENCH.md row.
#   python ref/baseline.py --mode compile --B 8 --T 1024
import argparse
import datetime
import os
import statistics
import subprocess
import time

import numpy as np
import torch

from model import DATA, ROOT, Config, Model, get_batch, init_weights

SMI_QUERY = "clocks.sm,clocks.mem,temperature.gpu,power.draw"


def proc_mem_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, mem = line.split(",")
        if int(pid) == os.getpid():
            return int(mem)
    return -1


def smi_stats(out):
    rows = []
    for line in out.splitlines():
        try:
            rows.append([float(v) for v in line.split(",")])
        except ValueError:
            pass
    sm, mem, temp, power = np.array(rows).T
    return f"{np.median(sm):.0f} ({sm.min():.0f}-{sm.max():.0f}) | {np.median(mem):.0f} | {temp.max():.0f} | {power.mean():.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "compile"], default="eager")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20, help="steps per timed run")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    model = Model(Config.load())
    init_weights(model)
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
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

    smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits", "-lms", "250"],
                           stdout=subprocess.PIPE, text=True)
    rates = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.steps):
            loss = step()
        torch.cuda.synchronize()
        rates.append(args.B * args.T * args.steps / (time.perf_counter() - t))
    pmem = proc_mem_mib()
    smi.terminate()
    clocks = smi_stats(smi.communicate()[0])

    commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    peak = torch.cuda.max_memory_allocated() >> 20
    print(f"# warmup {warm:.1f}s, last loss {loss.item():.4f}, runs {[round(r) for r in rates]}")
    print(f"| {datetime.date.today()} | {commit} | torch {args.mode} | {args.B}x{args.T} | "
          f"{statistics.median(rates):,.0f} | {min(rates):,.0f}-{max(rates):,.0f} | {peak:,} | {pmem:,} | {clocks} | "
          f"torch {torch.__version__} |")


if __name__ == "__main__":
    main()
