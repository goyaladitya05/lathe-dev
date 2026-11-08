# lathe

A training engine for one model on one GPU, written from scratch in C++/CUDA: SmolLM2-135M and SmolLM2-360M on an RTX 4050 Laptop GPU (6 GB). Same math as PyTorch (same initial weights, same batches, gradients within bf16 tolerance, matching loss curves), more training tokens per second, less memory.

**Results:** on SmolLM2-135M, 42.5-49.8% faster than torch.compile and 79.2-93.0% faster than PyTorch eager at 68.4-68.7% MFU (lathe tuned, depending on the batch shape); SmolLM2-360M trains at 8,156 tokens/s (75.9% MFU) with 8-bit optimizer state, 1.52-1.75x the speed of the fastest PyTorch setup that fits in 6 GB, shape for shape.

Shapes, the model and the GPU are compile-time constants. There is no op library, no autograd, no allocator: one memory arena, hand-written forward and backward, one CUDA graph per training step. The engine is about 2,100 lines of C++/CUDA (`src/train.cu`, `src/kernels.cuh`), plus chisel's generated SASS for some weight-gradient GEMMs.

## Results

Measured on an RTX 4050 Laptop GPU (6 GB, laptop in its performance power profile), all stacks in one sitting (2026-11-04, commit 5479e35); each number is the median of 5 timed runs on an idle machine, GPU below 55 C before each run. Same model, init, data, batch shape and precision recipe (bf16 compute with fp32 accumulation, fp32 master weights, AdamW). Tokens/s (MFU):

| SmolLM2-135M | PyTorch eager | torch.compile (max-autotune) | lathe | lathe, tuned |
|---|---|---|---|---|
| 1x2048 | 9,631 (38.3%) | 12,113 (48.2%) | 16,751 (66.7%) | **17,255 (68.7%)** |
| 2x1024 | 10,219 (36.4%) | 13,037 (46.5%) | 18,582 (66.3%) | **19,239 (68.6%)** |
| 4x512 | 10,546 (35.4%) | 13,593 (45.6%) | 19,507 (65.5%) | **20,356 (68.4%)** |

"lathe, tuned" is the same engine with cuBLASLt algorithms picked by measurement (`--tuned 1`) and the layer weight gradients through cuBLASLt (`DW_LT=1`).

Per-process GPU memory at these shapes (nvidia-smi): lathe 3,294 MiB, torch.compile 4,216-4,236 MiB, eager 5,392 MiB.

MFU = tokens/s x model FLOPs per token (6N + causal attention) / 25.61 TFLOPS, the best bf16 GEMM rate measured on this GPU with cuBLASLt.

**SmolLM2-360M.** PyTorch's fp32 weights, gradients and AdamW moments alone take 5.4 GiB, so a standard PyTorch training loop does not fit even 1x64. It fits with torchao's 8-bit AdamW, or with fp32 AdamW run inside the backward (eager, up to 1x192). lathe here uses its own 8-bit optimizer state. Tokens/s against the PyTorch setups that fit:

| SmolLM2-360M | torch.compile + AdamW8bit | torch.compile + checkpointing + AdamW8bit | lathe, 8-bit state, tuned |
|---|---|---|---|
| 1 x 1024 | 4,414 | 3,829 | **7,532 (69.4%)** |
| 1 x 1536 | 4,997 | 4,286 | **7,597 (72.8%)** |
| 2 x 1152 | - | 4,748 | **8,156 (75.9%)** |
| 1 x 2048 | - | 4,431 | **7,479 (74.4%)** |
| 9 x 256 | - | 5,023 | **8,798 (76.2%)** |

torch.compile + AdamW8bit fits at most 1,536 tokens per step (one sequence), so it has no entry at the larger shapes.

**Largest batch that fits in 6 GB** (tokens per step):

| | PyTorch eager | torch.compile | lathe | lathe, 8-bit state |
|---|---|---|---|---|
| SmolLM2-135M | 2,048 | 3,584 | 5,632 | 6,656 |
| SmolLM2-360M | 192 with fp32 AdamW in the backward; 1,152 with AdamW8bit | does not fit with fp32 AdamW; 1,536 with AdamW8bit | 704 | 2,304 |

lathe keeps no fp32 weight gradients: each layer's weight gradients go from the GEMM into a small ring of bf16 buffers that an AdamW pass on a side stream consumes during the backward, and the embedding gradient keeps one fp32 row per distinct token. Its fixed memory is 14 bytes per parameter (fp32 master, bf16 copy, two AdamW moments) against PyTorch's 16 (fp32 weight, gradient, two moments) plus its autocast copies.

Correctness, checked by `ref/test.py` for both models: every kernel against fp32 math on its own inputs, logits and loss against an fp32 reference (at most 2x PyTorch bf16's own error), every one of the 272 (135M) / 290 (360M) parameter gradients, one AdamW step bit-exact against PyTorch's fused AdamW (the 8-bit state byte for byte), CUDA graph replay bit-identical to eager, and loss curves against PyTorch bf16 and fp32 (1,000 steps for 135M, 300 to 500 for 360M).

The public README is `release/README.md`; `release/export.py` builds the public tree.
