# Fetch SmolLM2-135M files and a FineWeb-Edu shard, tokenize to uint16 .bin.
# Output: data/train.bin, data/val.bin (docs prefixed with <|endoftext|>), data/smol135/*.
import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parent
MODEL_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATASET_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
MODEL_URL = f"https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/{MODEL_REV}/"
SHARD_URL = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{DATASET_REV}/sample/10BT/013_00000.parquet"


def fetch(url, path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r{path.name}: {done >> 20}/{total >> 20} MiB", end="", file=sys.stderr)
    print(file=sys.stderr)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--train_tokens", type=int, default=100_000_000)
    args = ap.parse_args()

    for name in ["config.json", "tokenizer.json", "model.safetensors"]:
        fetch(MODEL_URL + name, DATA / "smol135" / name)
    shard = fetch(SHARD_URL, DATA / "fineweb_edu_013.parquet")

    tok = Tokenizer.from_file(str(DATA / "smol135" / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    assert eot == 0

    # val first, then train, from consecutive docs, split at exact token counts.
    targets = [("val.bin", args.val_tokens), ("train.bin", args.train_tokens)]
    buf, n = [], 0
    for batch in pq.ParquetFile(shard).iter_batches(columns=["text"], batch_size=4096):
        for enc in tok.encode_batch_fast(batch.column(0).to_pylist(), add_special_tokens=False):
            ids = enc.ids
            buf.append(np.array([eot] + ids, dtype=np.uint16))
            n += len(ids) + 1
            if n >= targets[0][1]:
                name, want = targets.pop(0)
                arr = np.concatenate(buf)
                arr[:want].tofile(DATA / name)
                print(f"{name}: {want:,} tokens", file=sys.stderr)
                buf, n = [arr[want:]], len(arr) - want
                if not targets:
                    return
    sys.exit(f"shard exhausted before {targets[0][0]} was full")


if __name__ == "__main__":
    main()
