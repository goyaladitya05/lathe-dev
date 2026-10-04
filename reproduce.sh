#!/usr/bin/env bash
# Reproduce lathe's results from a fresh clone: data, weights, correctness tests, then one same-session benchmark.
# Needs: CUDA 12.8 at /usr/local/cuda, a python3 that can import torch with CUDA, internet for the data (~1.3 GB).
#   ./reproduce.sh                      everything (the tinygrad BEAM search alone takes hours the first time)
#   STACKS=eager,compile,lathe ./reproduce.sh       skip tinygrad
set -euo pipefail
cd "$(dirname "$0")"
STACKS=${STACKS:-eager,compile,lathe,tinygrad}
SHAPES=${SHAPES:-1x2048,2x1024,4x512}

[ -d .venv ] || { python3 -m venv --system-site-packages .venv && .venv/bin/pip install -q tokenizers pyarrow; }
if [[ $STACKS == *tinygrad* && ! -d .venv-tinygrad ]]; then
    python3 -m venv .venv-tinygrad && .venv-tinygrad/bin/pip install -q tinygrad==0.14.0 numpy
fi

.venv/bin/python data/prep.py
for w in init pretrained; do
    [ -f data/smol135_$w.bin ] || .venv/bin/python ref/model.py --$w data/smol135_$w.bin
done
.venv/bin/python ref/test.py
.venv/bin/python ref/baseline.py --mode session --shapes "$SHAPES" --stacks "$STACKS"
