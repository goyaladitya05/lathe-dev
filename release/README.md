# lathe

A from-scratch C++/CUDA training engine for one model on one GPU: SmolLM2-135M and SmolLM2-360M, trained with AdamW on an RTX 4050 Laptop GPU (6 GB). Same math as PyTorch (same initial weights, same batches, gradients within bf16 tolerance, matching loss curves), more tokens per second, less memory.

- **SmolLM2-135M:** 42.5-49.8% faster than torch.compile (max-autotune) and 79.2-93.0% faster than PyTorch eager, at 68.4-68.7% MFU.
- **SmolLM2-360M:** 8,156 tokens/s at a 2 x 1152 batch (75.9% MFU), 1.52-1.75x the speed of the fastest PyTorch setup that fits in 6 GB, shape for shape. PyTorch fits the model only with an 8-bit optimizer, or with fp32 AdamW run inside the backward at up to 1 x 192 tokens; its fastest configuration that fits runs 5,023 tokens/s (torch.compile + checkpointing + AdamW8bit at 9 x 256).

Shapes, the model and the GPU are compile-time constants. There is no op library, no autograd and no allocator: one memory arena, a hand-written forward and backward, one CUDA graph per training step. The engine is about 2,100 lines of C++/CUDA, plus prebuilt SASS for some weight-gradient GEMMs.

## Results

RTX 4050 Laptop GPU (6 GB, sm_89, laptop in its performance power profile), CUDA 12.8, PyTorch 2.14.1, torchao 0.18.0. All rows from one sitting (2026-11-04, commit 5479e35; the engine code is unchanged since); each number is the median of 5 timed runs, accepted when two attempts agree within 0.5%, on an idle machine with the GPU below 55 C before each run. MFU = tokens/s x model FLOPs per token (6N + causal attention) / 25.61 TFLOPS, the best bf16 GEMM rate measured on this GPU.

**SmolLM2-135M against PyTorch.** Same model, init, data, batch shape and precision recipe (bf16 compute, fp32 accumulation, fp32 master weights, AdamW). Tokens/s (MFU):

| batch x context | PyTorch eager | torch.compile (max-autotune) | lathe | lathe, tuned |
|---|---|---|---|---|
| 1 x 2048 | 9,631 (38.3%) | 12,113 (48.2%) | 16,751 (66.7%) | **17,255 (68.7%)** |
| 2 x 1024 | 10,219 (36.4%) | 13,037 (46.5%) | 18,582 (66.3%) | **19,239 (68.6%)** |
| 4 x 512 | 10,546 (35.4%) | 13,593 (45.6%) | 19,507 (65.5%) | **20,356 (68.4%)** |

"lathe, tuned" is the same engine with cuBLASLt algorithms picked by measurement and the layer weight gradients through cuBLASLt (`--tuned 1` on a `DW_LT=1` build; see [Tuning](#tuning)).

**SmolLM2-360M.** PyTorch's fp32 weights, gradients and two AdamW moments alone take 5.4 GiB, so a standard PyTorch training loop does not fit even a 1 x 64 batch. Running fp32 AdamW inside the backward (a per-parameter hook, so the gradients never all exist at once) fits up to 1 x 192 in eager mode. With torchao's 8-bit AdamW it fits more easily, and activation checkpointing lets it reach long contexts. lathe here uses its own 8-bit optimizer state (below). Tokens/s (MFU):

| batch x context | torch.compile + torchao AdamW8bit | torch.compile + checkpointing + AdamW8bit | lathe, 8-bit state, tuned |
|---|---|---|---|
| 1 x 1024 | 4,414 | 3,829 | **7,532 (69.4%)** |
| 1 x 1536 | 4,997 | 4,286 | **7,597 (72.8%)** |
| 2 x 1152 | - | 4,748 | **8,156 (75.9%)** |
| 1 x 2048 | - | 4,431 | **7,479 (74.4%)** |
| 9 x 256 | - | 5,023 | **8,798 (76.2%)** |

torch.compile + AdamW8bit fits at most 1,536 tokens per step (one sequence), so it has no entry at the larger shapes.

**Also measured on other GPUs** (Colab; one card of each type; all stacks back to back on that card; each GPU with its own target header, `make TARGET=...`). Tokens/s, with MFU where it was recorded (against that card's measured bf16 peak); the last column compares the faster lathe build with torch.compile.

*NVIDIA L4* (24 GB, sm_89, power-capped; `TARGET=l4`), measured 2026-11-04:

| model | batch x context | PyTorch eager | torch.compile | lathe | lathe, tuned | lathe, 8-bit state, tuned | faster lathe vs torch.compile |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 1 x 2048 | 16,174 | 20,495 | 22,779 | 24,651 |  | +20.3% |
| SmolLM2-135M | 2 x 1024 | 16,756 | 21,642 | 24,999 | 27,020 |  | +24.9% |
| SmolLM2-135M | 4 x 512 | 17,014 | 22,304 | 26,601 | 28,699 |  | +28.7% |
| SmolLM2-360M | 1 x 2048 | 7,762 | 9,306 | 10,115 | 10,882 | 11,554 | +24.2% |
| SmolLM2-360M | 2 x 1024 | 7,942 | 9,631 | 11,157 | 11,737 | 12,453 | +29.3% |
| SmolLM2-360M | 4 x 512 | 8,089 | 9,828 | 11,253 | 12,188 | 13,010 | +32.4% |

*NVIDIA A100-SXM4-40GB* (sm_80; `TARGET=a100`), measured 2026-10-23:

| model | batch x context | PyTorch eager | torch.compile | lathe | lathe, tuned | lathe, 8-bit state, tuned | faster lathe vs torch.compile |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 1 x 2048 | 17,249 | 66,003 | 80,350 (30.4%) | 85,956 (32.6%) |  | +30.2% |
| SmolLM2-135M | 2 x 1024 | 17,516 | 72,992 | 91,987 (31.2%) | 99,521 (33.8%) |  | +36.3% |
| SmolLM2-135M | 4 x 512 | 17,157 | 76,047 | 99,395 (31.8%) | 106,559 (34.1%) |  | +40.1% |
| SmolLM2-360M | 1 x 2048 | 16,110 | 34,662 | 40,528 (38.4%) | 43,790 (41.5%) | 42,003 | +26.3% |
| SmolLM2-360M | 2 x 1024 | 16,157 | 37,235 | 44,863 (39.3%) | 48,875 (42.9%) | 46,223 | +31.3% |
| SmolLM2-360M | 4 x 512 | 15,970 | 38,770 | 47,144 (39.7%) | 51,324 (43.2%) | 48,446 | +32.4% |

*NVIDIA RTX PRO 6000 Blackwell Server Edition* (sm_120; `TARGET=g4`), measured 2026-10-23:

| model | batch x context | PyTorch eager | torch.compile | lathe | lathe, tuned | lathe, 8-bit state, tuned | faster lathe vs torch.compile |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 1 x 2048 | 54,617 | 97,810 | 113,531 (27.6%) | 112,242 |  | +16.1% |
| SmolLM2-135M | 2 x 1024 | 58,202 | 108,252 | 125,252 (27.3%) | 123,340 |  | +15.7% |
| SmolLM2-135M | 4 x 512 | 56,487 | 113,849 | 133,806 (27.5%) | 131,352 |  | +17.5% |
| SmolLM2-360M | 1 x 2048 | 36,469 | 52,729 | 60,758 (37.0%) | 59,082 | 54,926 | +15.2% |
| SmolLM2-360M | 2 x 1024 | 38,388 | 56,595 | 64,766 (36.5%) | 63,183 | 57,292 | +14.4% |
| SmolLM2-360M | 4 x 512 | 39,375 | 58,759 | 68,124 (36.9%) | 65,872 | 59,794 | +15.9% |

The diagnostics planned for the A100 and the G4 were never run: there is no breakdown of where a training step's GPU time goes on either card, for lathe or for torch.compile, so why lathe reaches a lower MFU there than on the laptop is not measured. On the G4, lathe's own weight-gradient kernel was faster than cuBLASLt's, so its default build is the faster one there.

**Largest batch that fits in 6 GB** (tokens per step; memory fronts measured from 2026-10-05 to 2026-10-13, the fused-optimizer PyTorch fronts on 2026-11-04):

| | PyTorch eager | torch.compile | lathe | lathe, 8-bit state |
|---|---|---|---|---|
| SmolLM2-135M | 2,048 | 3,584 | 5,632 | 6,656 |
| SmolLM2-360M | 192 with fp32 AdamW in the backward; 1,152 with AdamW8bit | does not fit with fp32 AdamW; 1,536 with AdamW8bit | 704 | 2,304 |

lathe keeps no full fp32 gradients: each layer's weight gradients go from their GEMM to the optimizer as bf16 (only the embedding gradient keeps fp32 rows, one per distinct token), and AdamW runs beside the backward pass. Its fixed memory is 14 bytes per parameter with fp32 optimizer state and 8.1 with 8-bit state, against PyTorch's 16 (10 with AdamW8bit) plus its autocast copies.

**Fastest configurations** (lathe, tuned; measured 2026-10-13, before the release sitting, on the same engine code):

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
- **Measured GEMM algorithms** (`--tune 1`, then `--tuned 1`). For every GEMM, lathe times up to 32 cuBLASLt candidates and keeps the fastest one that is run-to-run deterministic and numerically close to the default pick. Gains per GEMM range from 0% to 19.2%, and 2.5-3.5% per step at 360M.
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

Needs an NVIDIA GPU with 6 GB or more and a target header in `src/targets/`: any sm_89 card with the default `rtx4050` target (schedules tuned for its 20 SMs), or `make TARGET=l4`, `a100` or `g4` (RTX PRO 6000 Blackwell). Linux, CUDA 12.8 at `/usr/local/cuda`, and a Python with CUDA PyTorch (for data prep, the reference and the tests).

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

lathe is a training engine for measuring and checking, not a training framework: one GPU, two models, the batch shape fixed at build time, a constant learning rate, no checkpoint saving.

## What was tried and dropped

| idea | result |
|---|---|
| cross entropy that never materializes the logits (vocab-chunked) | slower: needs a second lm_head forward, which is compute-bound here |
| single-pass attention backward with atomic dq | +0.1 to +1.8%, but non-deterministic; deterministic backward kept |
| fused gate/up GEMM + SwiGLU | no faster than cuBLASLt plus a separate SwiGLU kernel |
| AdamW on a second stream as one big pass | no gain while the backward already filled all SMs; it paid off later as small per-layer passes beside compute-bound kernels |
| one attention-backward pass over more heads | slower (-0.33 to -0.85%) |
| attention dq as hand-scheduled SASS | no faster (-0.01 to -0.31%) |
| one dS buffer per attention group | slower (-0.94 to -1.37%) |
| PyTorch fp8 (torchao) as a baseline | rowwise: 12-22% slower than bf16 torch.compile on this GPU; tensorwise: faster, but diverges to NaN within 120 steps |
| llm.c as a baseline | its CUDA trainer is GPT-2 only (no RoPE, RMSNorm, SwiGLU or GQA) |

In progress: FP8 (E4M3) GEMMs for the layer linears, +5.5-6% at 360M so far, not in this release.

## License

Apache-2.0 (see LICENSE)
