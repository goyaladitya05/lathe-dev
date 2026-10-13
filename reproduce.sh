#!/usr/bin/env bash
# Reproduce lathe's results from a fresh clone: data, weights, correctness tests, then the benchmarks in one sitting.
# Needs: an sm_89 GPU (RTX 40 series / L4 / L40) with 6 GB or more, CUDA 12.8 at /usr/local/cuda, a python3 that can
# import torch with CUDA, internet for the data (~2 GB), ~10 GB of free host RAM for the 360M tests.
#   ./reproduce.sh                                  everything (tinygrad's first BEAM search takes about 20 minutes per shape)
#   STACKS=eager,compile,lathe ./reproduce.sh       skip tinygrad
#   SMOL360=0 ./reproduce.sh                        SmolLM2-135M only
set -euo pipefail
cd "$(dirname "$0")"
STACKS=${STACKS:-eager,compile,lathe,tinygrad}
SHAPES=${SHAPES:-1x2048,2x1024,4x512}
SMOL360=${SMOL360:-1}
MODELS=smol135; [ "$SMOL360" = 1 ] && MODELS=smol135,smol360

[ -d .venv ] || { python3 -m venv --system-site-packages .venv && .venv/bin/pip install -q tokenizers pyarrow torchao==0.18.0; }
if [[ $STACKS == *tinygrad* && ! -d .venv-tinygrad ]]; then
    python3 -m venv .venv-tinygrad && .venv-tinygrad/bin/pip install -q tinygrad==0.14.0 numpy
fi

.venv/bin/python data/prep.py --models $MODELS
for m in ${MODELS//,/ }; do for w in init pretrained; do
    [ -f data/${m}_$w.bin ] || .venv/bin/python ref/model.py --model $m --$w data/${m}_$w.bin
done; done

# lathe_tune MODEL B T FLAGS...: build, then measure every cuBLASLt GEMM of that binary into build/lt_cache.txt
lathe_tune() {
    local m=$1 b=$2 t=$3 suf=""; shift 3
    for f in "$@"; do case $f in ADAM8=1) suf+=_adam8;; DW_LT=1) suf+=_dwlt;; esac; done
    make -s MODEL=$m BATCH=$b SEQ=$t "$@"
    build/train_${m}_b${b}_t${t}${suf} --tune 1 --steps 1 --data data/train.bin | grep gemm_tune
}

# Correctness: default build, cuBLASLt-tuned build, 8-bit state build
.venv/bin/python ref/test.py
lathe_tune smol135 2 512 DW_LT=1
.venv/bin/python ref/test.py --tuned --dw-lt
.venv/bin/python ref/test.py --adam8
if [ "$SMOL360" = 1 ]; then
    lathe_tune smol360 1 256 ADAM8=1 DW_LT=1
    .venv/bin/python ref/test.py --adam8 --tuned --dw-lt --model smol360 --B 1 --T 256
fi

# SmolLM2-135M: every stack per shape, then lathe tuned
.venv/bin/python ref/baseline.py --mode session --shapes "$SHAPES" --stacks "$STACKS"
for s in ${SHAPES//,/ }; do
    lathe_tune smol135 ${s%x*} ${s#*x} DW_LT=1
    .venv/bin/python ref/baseline.py --mode lathe --B ${s%x*} --T ${s#*x} --tuned --dw-lt
done

# SmolLM2-360M: lathe 8-bit state tuned against the strongest PyTorch that fits (torchao AdamW8bit)
if [ "$SMOL360" = 1 ]; then
    for s in 1x1024 2x1152 9x256; do
        lathe_tune smol360 ${s%x*} ${s#*x} ADAM8=1 DW_LT=1
        .venv/bin/python ref/baseline.py --model smol360 --mode lathe --adam8 --B ${s%x*} --T ${s#*x} --tuned --dw-lt
    done
    .venv/bin/python ref/torch_strong.py --model smol360 --B 1 --T 1024 --compile --adam8 --runs 5 --steps 30
    .venv/bin/python ref/torch_strong.py --model smol360 --B 2 --T 1152 --compile --ckpt --adam8 --runs 5 --steps 30
fi
