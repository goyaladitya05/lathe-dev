# lathe

A from-scratch C++/CUDA training engine for one model on one GPU: SmolLM2-135M and SmolLM2-360M, trained with AdamW on an RTX 4050 Laptop GPU (6 GB). Same math as PyTorch (same initial weights, same batches, gradients within bf16 tolerance, matching loss curves), more tokens per second, less memory.

- **SmolLM2-135M:** {CMP_COMPILE} faster than torch.compile (max-autotune) and {CMP_EAGER} faster than PyTorch eager, at {CMP_MFU} MFU.
- **SmolLM2-360M:** {H360_TOKS} tokens/s at a 2 x 1152 batch ({H360_MFU} MFU). PyTorch needs an 8-bit optimizer just to fit the model in 6 GB; its fastest configuration that fits runs {T360_BEST} tokens/s.

Shapes, the model and the GPU are compile-time constants. There is no op library, no autograd and no allocator: one memory arena, a hand-written forward and backward, one CUDA graph per training step. The engine is about 2,100 lines of C++/CUDA, plus prebuilt SASS for some weight-gradient GEMMs.

## Results

RTX 4050 Laptop GPU (6 GB, sm_89, laptop in its performance power profile), CUDA 12.8, PyTorch 2.14.1, torchao 0.18.0. All rows from one sitting at the release commit; each number is the median of 5 timed runs, accepted when two attempts agree within 0.5%, on an idle machine with the GPU below 55 C before each run. MFU = tokens/s x model FLOPs per token (6N + causal attention) / 25.61 TFLOPS, the best bf16 GEMM rate measured on this GPU.

**SmolLM2-135M against PyTorch.** Same model, init, data, batch shape and precision recipe (bf16 compute, fp32 accumulation, fp32 master weights, AdamW). Tokens/s (MFU):

| batch x context | PyTorch eager | torch.compile (max-autotune) | lathe | lathe, tuned |
|---|---|---|---|---|
| 1 x 2048 | {E1} | {C1} | {L1} | **{LT1}** |
| 2 x 1024 | {E2} | {C2} | {L2} | **{LT2}** |
| 4 x 512 | {E3} | {C3} | {L3} | **{LT3}** |

"lathe, tuned" is the same engine with cuBLASLt algorithms picked by measurement and the layer weight gradients through cuBLASLt (`--tuned 1` on a `DW_LT=1` build; see [Tuning](#tuning)).

**SmolLM2-360M.** PyTorch's fp32 weights, gradients and two AdamW moments alone take 5.4 GiB, so plain PyTorch does not fit even a 1 x 64 batch. With torchao's 8-bit AdamW it fits, and activation checkpointing lets it reach long contexts. lathe here uses its own 8-bit optimizer state (below). Tokens/s (MFU):

| batch x context | torch.compile + torchao AdamW8bit | torch.compile + checkpointing + AdamW8bit | lathe, 8-bit state, tuned |
|---|---|---|---|
{T360_ROWS}

**Also on an NVIDIA L4** (24 GB, sm_89, datacenter card power-capped at 72 W; Modal, CUDA 12.8, PyTorch 2.14.1). lathe was not retuned for the L4's 58 SMs (its pass schedules and SASS tiles are tuned for the laptop's 20). Each shape ran on one card, all stacks back to back, twice (the two sittings agree within 2%); the numbers are their mean. MFU against that card's measured bf16 peak (48.7-54.6 TFLOPS). Under load lathe holds the L4 at 100% utilization against its power cap, so on this card it is bound by energy per step. Tokens/s (vs torch.compile):

| model | batch x context | PyTorch eager | torch.compile | lathe | lathe, tuned | lathe, 8-bit state, tuned |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 1 x 2048 | 15,793 | 20,285 | 23,358 (+15%) | **24,486 (+21%)** | |
| SmolLM2-135M | 2 x 1024 | 16,205 | 21,750 | 24,974 (+15%) | **25,935 (+19%)** | |
| SmolLM2-135M | 4 x 512 | 16,349 | 22,191 | 25,843 (+16%) | **27,479 (+24%)** | |
| SmolLM2-360M | 1 x 2048 | 7,880 | 9,501 | 10,255 (+8%) | 11,309 (+19%) | **11,433 (+20%)** |
| SmolLM2-360M | 2 x 1024 | 7,843 | 9,412 | 10,338 (+10%) | 11,381 (+21%) | **11,381 (+21%)** |
| SmolLM2-360M | 4 x 512 | 7,968 | 9,766 | 10,528 (+8%) | 11,697 (+20%) | **12,027 (+23%)** |

GPU memory at 360M: lathe 7.0 GiB (5.3 GiB with 8-bit state), torch.compile 7.9 GiB reserved, eager 9.9 GiB reserved.

**Largest batch that fits in 6 GB** (tokens per step):

| | PyTorch eager | torch.compile | lathe | lathe, 8-bit state |
|---|---|---|---|---|
| SmolLM2-135M | 2,048 | 3,584 | 5,632 | 6,656 |
| SmolLM2-360M | does not fit (fp32 AdamW); {F360_EAGER} with AdamW8bit | does not fit (fp32 AdamW); {F360_COMPILE} with AdamW8bit | 704 | 2,304 |

lathe keeps no fp32 gradients: each weight gradient goes from its GEMM to the optimizer as bf16, and AdamW runs inside the backward pass. Its fixed memory is 14 bytes per parameter with fp32 optimizer state and 8.1 with 8-bit state, against PyTorch's 16 (10 with AdamW8bit) plus its autocast copies.

**Fastest configurations** (lathe, tuned; measured the night before the release sitting, same engine code):

| model | | optimizer state | batch x context | tokens/s | MFU |
|---|---|---|---|---|---|
| SmolLM2-135M | context >= 1024 | fp32 | 5 x 1024 | 19,480 | 69.5% |
| SmolLM2-135M | any context | fp32 | 44 x 128 | 21,536 | 69.0% |
| SmolLM2-360M | context >= 1024 | 8-bit | 2 x 1152 | 8,171 | 76.0% |
| SmolLM2-360M | any context | 8-bit | 36 x 64 | 8,903 | 75.9% |

At a fixed number of tokens per step, shorter contexts are faster, since causal attention costs less per token.

## How it works

- **Everything static.** Model, batch and context are compile-time constants (`make MODEL=smol360 BATCH=2 SEQ=1152`), so every buffer lives at a fixed offset in one arena and every kernel launch has fixed dimensions. A whole training step (forward, backward, optimizer) is captured once and replayed as one CUDA graph.
- **No gradient tensors.** Each layer's weight gradients go into a small ring buffer, and an AdamW pass for that layer runs on a low-priority side stream while the next layer's backward runs. The embedding gradient keeps one fp32 row per distinct token in the batch.
- **Hand-written kernels where they pay.** The qkv projection with RoPE in its epilogue; the down projection's input gradient with SwiGLU's backward; cross-entropy writing the logit gradients in place; the head and embedding gradients fused with their AdamW update. Flash-style causal GQA attention, forward and a deterministic backward. The other matmuls are cuBLASLt, with fp32 accumulation like PyTorch's bf16 autocast.
- **8-bit optimizer state** (`make ADAM8=1`, a separate build). The AdamW moments of the layer matrices are stored as FP8 E4M3 in blocks of 64 with one fp32 scale per block (the second moment as its square root). The update runs in fp32 and the fp32 master weights are unchanged. This cuts optimizer memory from 8 to 2.1 bytes per parameter.
- **Measured GEMM algorithms** (`--tune 1`, then `--tuned 1`). For every GEMM, lathe times up to 32 cuBLASLt candidates and keeps the fastest one that is run-to-run deterministic and numerically close to the default pick. Gains per GEMM range from 0% to 19%, and 2.5-3.5% per step at 360M.
- **Prebuilt SASS** (`src/sass/`) for the layer weight-gradient GEMMs at a few token counts: hand-scheduled sm_89 cubins, bit-identical to the CUDA version they replace. In the tuned build (`DW_LT=1`) cuBLASLt takes over these GEMMs.

## Correctness

`ref/test.py` checks both models and every build:

- every kernel against fp32 math on its own inputs;
- logits and loss against an fp32 reference, at most 2x PyTorch bf16's own error;
- every parameter gradient (272 tensors for 135M, 290 for 360M);
- one AdamW step, bit-exact against PyTorch's fused AdamW (and the 8-bit state byte for byte);
- CUDA graph replay, bit-identical to eager execution;
- loss curves against PyTorch bf16 and fp32: 1,000 steps for 135M, 300 to 500 for 360M, each judged against PyTorch bf16's own distance from fp32.

## Quick start

Needs an sm_89 GPU (RTX 40 series, L4, L40) with 6 GB or more, Linux, CUDA 12.8 at `/usr/local/cuda`, and a Python with CUDA PyTorch (for data prep, the reference and the tests).

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install tokenizers pyarrow torchao==0.18.0
.venv/bin/python data/prep.py --models smol135,smol360           # SmolLM2 files + 100M FineWeb-Edu tokens
.venv/bin/python ref/model.py --model smol135 --init data/smol135_init.bin

make MODEL=smol135 BATCH=4 SEQ=512                                # build/train_smol135_b4_t512
build/train_smol135_b4_t512 --data data/train.bin --weights data/smol135_init.bin --steps 200

.venv/bin/python ref/test.py                                      # unit checks (135M, 2 x 512)
.venv/bin/python ref/test.py curve --steps 1000                   # loss curve against PyTorch
```

<a id="tuning"></a>**Tuning:** build with `DW_LT=1`, run once with `--tune 1` (writes `build/lt_cache.txt`, keyed by GEMM shape and cuBLASLt version), then pass `--tuned 1`:

```bash
make MODEL=smol360 BATCH=2 SEQ=1152 ADAM8=1 DW_LT=1
build/train_smol360_b2_t1152_adam8_dwlt --tune 1 --steps 1 --data data/train.bin
build/train_smol360_b2_t1152_adam8_dwlt --tuned 1 --data data/train.bin --steps 200
```

`./reproduce.sh` runs everything above for both models, then the benchmarks behind the tables. `ref/baseline.py` runs one benchmark row (`--mode eager|compile|lathe|tinygrad`); `ref/torch_strong.py` runs the strongest PyTorch setups (torch.compile, activation checkpointing, torchao AdamW8bit, optimizer fused into backward).

lathe is a training engine for measuring and checking, not a training framework: one GPU, two models, the batch shape fixed at build time, no checkpoint saving.

## What was tried and dropped

| idea | result |
|---|---|
| cross entropy that never materializes the logits (vocab-chunked) | slower: needs a second lm_head forward, which is compute-bound here |
| single-pass attention backward with atomic dq | +0.1 to +1.8%, but non-deterministic; deterministic backward kept |
| fused gate/up GEMM + SwiGLU | no faster than cuBLASLt plus a separate SwiGLU kernel |
| AdamW on a second stream as one big pass | no gain while the backward already filled all SMs; it paid off later as small per-layer passes beside compute-bound kernels |
| one attention-backward pass over more heads | slower (-0.3 to -0.9%) |
| attention dq as hand-scheduled SASS | no faster (-0.04 to -0.3%) |
| one dS buffer per attention group | slower (-0.9 to -1.4%) |
| PyTorch fp8 (torchao) as a baseline | rowwise: 12-22% slower than bf16 torch.compile on this GPU; tensorwise: faster, but diverges to NaN within 120 steps |
| llm.c as a baseline | its CUDA trainer is GPT-2 only (no RoPE, RMSNorm, SwiGLU or GQA) |

In progress: FP8 (E4M3) GEMMs for the layer linears, +5.5-6% at 360M so far, not in this release.

## License

MIT
