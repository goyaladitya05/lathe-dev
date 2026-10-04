# tinygrad training-step baseline: same config, seed-0 init (lathe weight file), batches and precision recipe as the
# PyTorch baselines (bf16 compute with fp32 accumulation, fp32 master weights, AdamW). Runs in .venv-tinygrad.
# Prints "group g steps a-b ms X tok/s Y" lines like lathe; ref/baseline.py --mode tinygrad times it the same way.
#   LD_LIBRARY_PATH=/usr/local/cuda/lib64 DEV=CUDA BEAM=2 .venv-tinygrad/bin/python ref/tinygrad_baseline.py --B 4 --T 512
import argparse
import json
import time
from pathlib import Path

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit, dtypes
from tinygrad.nn.optim import AdamW

DATA = Path(__file__).resolve().parent.parent / "data"
BF = dtypes.bfloat16


def load_params(cfg, path):
    # lathe layout: embed, per layer [q|k|v, o, gate|up, down, ln1, ln2], final norm (fp32 after a 256-byte header).
    flat, off, hd = np.fromfile(path, dtype=np.float32, offset=256), 0, cfg["hidden_size"] // cfg["num_attention_heads"]
    D, F, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    QKV = (cfg["num_attention_heads"] + 2 * cfg["num_key_value_heads"]) * hd

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        off += n
        return Tensor(flat[off - n:off].reshape(shape).copy())

    p = {"embed": take(V, D), "layers": []}
    for _ in range(cfg["num_hidden_layers"]):
        p["layers"].append({"qkv": take(QKV, D), "o": take(D, D), "gu": take(2 * F, D), "down": take(D, F),
                            "ln1": take(D), "ln2": take(D)})
    p["norm"] = take(D)
    assert off == flat.size
    return p


def lin(x, w):  # bf16 inputs and weights, fp32 accumulation, bf16 output (PyTorch autocast linear)
    return x.cast(BF).matmul(w.cast(BF).T, dtype=dtypes.float).cast(BF)


def rmsnorm(x, w, eps):
    x = x.float()
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * w


def rope(x, cos, sin):  # x [B, H, T, hd] bf16, fp32 math, pairs (i, i + hd/2)
    x1, x2 = x.float().chunk(2, -1)
    return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1).cast(BF)


def forward(p, cfg, idx, tgt, cos, sin, mask):
    B, T = idx.shape
    H, KV, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    hd, eps = D // H, cfg["rms_norm_eps"]
    x = p["embed"][idx]  # fp32 residual stream, as under autocast
    for L in p["layers"]:
        qkv = lin(rmsnorm(x, L["ln1"], eps), L["qkv"])
        q, k, v = qkv.split([H * hd, KV * hd, KV * hd], dim=-1)
        q = rope(q.reshape(B, T, H, hd).transpose(1, 2), cos, sin)
        k = rope(k.reshape(B, T, KV, hd).transpose(1, 2), cos, sin)
        v = v.reshape(B, T, KV, hd).transpose(1, 2)
        k, v = (t.reshape(B, KV, 1, T, hd).expand(B, KV, H // KV, T, hd).reshape(B, H, T, hd) for t in (k, v))
        s = q.matmul(k.transpose(-1, -2), dtype=dtypes.float) * (hd ** -0.5) + mask  # fp32 scores
        att = s.softmax(-1).cast(BF).matmul(v, dtype=dtypes.float).cast(BF)
        x = x + lin(att.transpose(1, 2).reshape(B, T, D), L["o"]).float()
        g, u = lin(rmsnorm(x, L["ln2"], eps), L["gu"]).chunk(2, -1)
        x = x + lin(g.silu() * u, L["down"]).float()
    logits = lin(rmsnorm(x, p["norm"], eps), p["embed"])
    return logits.float().sparse_categorical_crossentropy(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smol135")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--group", type=int, default=20)
    args = ap.parse_args()
    cfg = json.loads((DATA / args.model / "config.json").read_text())
    B, T, N = args.B, args.T, args.B * args.T
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]

    p = load_params(cfg, DATA / f"{args.model}_init.bin")
    params = [p["embed"], p["norm"]] + [t for L in p["layers"] for t in L.values()]
    opt = AdamW(params, lr=3e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
    inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
    ang = np.outer(np.arange(T, dtype=np.float32), inv)
    cos, sin = Tensor(np.cos(ang)).is_param_(False).realize(), Tensor(np.sin(ang)).is_param_(False).realize()
    mask = Tensor(np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)).is_param_(False).realize()
    tokens = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    nb = (len(tokens) - 1) // N

    @TinyJit
    def step(idx, tgt):
        with Context(TRAINING=1):
            opt.zero_grad()
            loss = forward(p, cfg, idx, tgt, cos, sin, mask)
            loss.backward()
            opt.step()
            return loss.realize()

    print(f"tinygrad {Device.DEFAULT} B={B} T={T}", flush=True)
    t0, first = time.perf_counter(), 0
    for i in range(args.steps):
        s = (i % nb) * N  # same batch order as ref/model.py get_batch and lathe
        buf = tokens[s:s + N + 1].astype(np.int32)
        loss = step(Tensor(buf[:-1].reshape(B, T)).is_param_(False).realize(), Tensor(buf[1:].reshape(B, T)).is_param_(False).realize())
        if (i + 1) % args.group == 0 or i + 1 == args.steps:
            Device[Device.DEFAULT].synchronize()
            now = time.perf_counter()
            print(f"step {i} loss {loss.item():.6f}")  # TinyJit reuses output buffers: only the latest is valid
            ms = (now - t0) * 1e3
            print(f"group {i // args.group} steps {first}-{i} ms {ms:.1f} tok/s {(i + 1 - first) * N / ms * 1e3:.0f}", flush=True)
            first, t0 = i + 1, time.perf_counter()


if __name__ == "__main__":
    main()
