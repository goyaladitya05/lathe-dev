# The strongest PyTorch training setup for SmolLM2 on this GPU, each memory/speed lever behind a switch:
#   --compile   torch.compile(mode="max-autotune")
#   --ckpt      activation checkpointing (recompute each block in backward)
#   --adam8     torchao AdamW8bit (block-wise 8-bit moments) instead of fused fp32 AdamW
#   --fused     optimizer step fused into backward: a per-parameter optimizer steps in a post-accumulate-grad hook
#               and frees the grad, so no full set of gradients ever exists
# Same model, init, data, bf16 autocast and AdamW hyperparameters as ref/baseline.py.
#   python ref/torch_strong.py --model smol360 --B 1 --T 1024 --ckpt --adam8 --fused [--compile]   one run: tok/s, peak memory
#   python ref/torch_strong.py --model smol360 --front                                           largest tokens per step per switch combination
import argparse
import itertools
import subprocess
import sys
import time

import numpy as np
import torch

from model import BETAS, DATA, EPS, LR, WD, Config, Model, get_batch, init_weights, make_optimizer


def run(args):
    model = Model(Config.load(args.model))
    init_weights(model)
    model.cuda()
    model.ckpt = args.ckpt
    params = list(model.parameters())
    if args.adam8:
        from torchao.optim import AdamW8bit
        make = lambda ps: AdamW8bit(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    else:
        make = lambda ps: torch.optim.AdamW(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, fused=True)
    if args.fused:  # one optimizer per parameter, stepped as soon as that parameter's grad is complete
        opts = {p: make([p]) for p in params}
        def hook(q):  # must return None
            opts[q].step()
            opts[q].zero_grad(set_to_none=True)
        for p in params:
            p.register_post_accumulate_grad_hook(hook)
        opt = None
    else:
        opt = make(params) if args.adam8 else make_optimizer(model)
    fwd = torch.compile(model, mode="max-autotune") if args.compile else model
    tokens = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    step_i = iter(range(1 << 62))

    def step():
        x, y = (t.cuda(non_blocking=True) for t in get_batch(tokens, next(step_i), args.B, args.T))
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(x, y)
        loss.backward()
        if opt is not None:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return loss

    try:
        for _ in range(args.warmup):
            loss = step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rates = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            for _ in range(args.steps):
                loss = step()
            torch.cuda.synchronize()
            rates.append(args.B * args.T * args.steps / (time.perf_counter() - t0))
    except torch.OutOfMemoryError:
        print(f"# OOM {args.model} {args.B}x{args.T} {switches(args)}")
        sys.exit(3)
    peak, reserved = torch.cuda.max_memory_allocated() >> 20, torch.cuda.max_memory_reserved() >> 20
    print(f"# ok {args.model} {args.B}x{args.T} {switches(args)} loss {loss.item():.4f} peak {peak} MiB reserved "
          f"{reserved} MiB tok/s {np.median(rates):.0f} spread {100 * (max(rates) - min(rates)) / np.median(rates):.2f}%")


def switches(args):
    return "+".join(n for n in ("compile", "ckpt", "adam8", "fused") if getattr(args, n)) or "plain"


def front(args):
    # Largest tokens per step (1 x T, T a multiple of 64 up to 4096) per switch combination, each probe in a fresh
    # process; a combination that does not fit 1 x 64 is skipped. Memory only: few steps per probe.
    for combo in itertools.product([False, True], repeat=4):
        flags = [f"--{n}" for n, on in zip(("compile", "ckpt", "adam8", "fused"), combo) if on]
        if not all(f"--{n}" in flags for n in args.only):
            continue
        probe = lambda T: subprocess.run(["flock", "/tmp/gpu-bench.lock",  # shared GPU: each probe holds the lock
                                          sys.executable, __file__, "--model", args.model, "--B", "1", "--T", str(T),
                                          "--warmup", "2", "--steps", "2", *flags], capture_output=True, text=True)
        lo, hi = 0, 4096 // 64 + 1  # in units of 64 tokens: lo fits (0 = none yet), hi does not
        last = None
        while hi - lo > 1:
            mid = lo + 1 if lo == 0 else (lo + hi) // 2
            r = probe(64 * mid)
            ok = r.returncode == 0 and "# ok" in r.stdout
            if ok:
                lo, last = mid, [l for l in r.stdout.splitlines() if l.startswith("# ok")][-1]
            else:
                hi = mid
                if lo == 0:
                    break
        name = "+".join(f[2:] for f in flags) or "plain"
        print(f"front | {args.model} | {name} | {64 * lo if lo else 'does not fit 1x64'} | {last or ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smol360")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--runs", type=int, default=1, help="timed segments of --steps steps: median and spread")
    ap.add_argument("--only", nargs="*", default=[], help="front: only combinations containing these switches")
    for n in ("compile", "ckpt", "adam8", "fused"):
        ap.add_argument(f"--{n}", action="store_true")
    ap.add_argument("--front", action="store_true", help="largest tokens per step for every switch combination")
    args = ap.parse_args()
    front(args) if args.front else run(args)


if __name__ == "__main__":
    main()
