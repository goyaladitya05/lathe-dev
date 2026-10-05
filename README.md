# lathe

A training engine for one model on one GPU, written from scratch in C++/CUDA: SmolLM2-135M (and SmolLM2-360M) on an RTX 4050 Laptop. Same math as PyTorch (same initial weights, same batches, gradients within bf16 tolerance, matching loss curves), more training tokens per second, less memory.

**Results:** on SmolLM2-135M, 32-41% faster than torch.compile and 66-82% faster than PyTorch eager at 63-64% MFU (depending on the batch size), with room for 1.6x torch.compile's largest batch; SmolLM2-360M, which PyTorch cannot fit in 6 GB, trains at 5,257 tokens/s.

Shapes, the model and the GPU are compile-time constants. There is no op library, no autograd, no allocator: one memory arena, hand-written forward and backward, one CUDA graph per training step. The whole engine is 1,462 lines of C++/CUDA.

## Results

Measured on an RTX 4050 Laptop GPU (6 GB, laptop in its performance power profile). Training throughput (tokens/s, higher is better) and MFU, all stacks measured in one sitting (commit 6d5bf8a, 2026-10-05), same model, init, data, batch shape and precision recipe (bf16 compute with fp32 accumulation, fp32 master weights, AdamW):

| SmolLM2-135M | PyTorch eager | torch.compile (max-autotune) | lathe |
|---|---|---|---|
| 1x2048 | 9,555 (38.0%) | 12,034 (47.9%) | **15,853 (63.1%)** |
| 2x1024 | 10,107 (36.0%) | 12,974 (46.3%) | **17,874 (63.7%)** |
| 4x512 | 10,472 (35.2%) | 13,438 (45.1%) | **19,013 (63.9%)** |

lathe is 32-41% faster than torch.compile and 66-82% faster than eager at every shape.

Per-process GPU memory at these shapes (nvidia-smi): lathe 3.2 GiB, torch.compile 4.1 GiB, eager 5.3 GiB.

MFU = tokens/s x model FLOPs per token (6N + causal attention) / 25.61 TFLOPS, the best bf16 GEMM rate measured on this GPU with cuBLASLt (99% of its theoretical peak at 2.52 GHz).

**Largest batch that fits in 6 GB** (SmolLM2-135M, tokens per step):

| | max tokens per step | tokens/s, largest batch at T=512 | at T=1024 | at T=2048 |
|---|---|---|---|---|
| PyTorch eager | 2,048 | 10,472 (4x512) | 10,107 (2x1024) | 9,555 (1x2048) |
| torch.compile | 3,584 | 14,810 (7x512) | 13,952 (3x1024) | 12,034 (1x2048) |
| lathe | 5,632 | **19,927 (11x512, 66.9% MFU)** | **18,663 (5x1024)** | **16,531 (2x2048)** |

At each engine's own limit, lathe fits 1.6x the tokens per step of torch.compile and 2.75x those of eager, and trains 1.35x and 1.90x as many tokens per second.

lathe stores no matrix gradients (each one goes from the dW GEMM's registers straight into AdamW) and only one fp32 row per distinct token for the embedding gradient, so its fixed memory is 14 bytes per parameter (fp32 master, bf16 copy, two AdamW moments) against PyTorch's 16 (fp32 weight, gradient, two moments) plus its autocast copies.

**SmolLM2-360M** (fp32 weights, gradients and AdamW moments alone take 5.4 GiB in PyTorch):

| | fits in 6 GB | largest step | tokens/s | MFU |
|---|---|---|---|---|
| PyTorch eager | no (out of memory at 1x64) | - | - | - |
| torch.compile | no (out of memory at 1x64) | - | - | - |
| lathe | yes | 1x704 | **5,257** | **47.2%** |

At 1x512 and 1x256 lathe runs 4,461 and 2,738 tokens/s (39.5% and 23.7% MFU). MFU is lower than for 135M because AdamW still moves about 9.4 GB per step however few tokens the step trains on.

Correctness, checked by `ref/test.py` for both models: every kernel against fp32 math on its own inputs, logits and loss against an fp32 reference (at most 2x PyTorch bf16's own error), every one of the 272 (135M) / 290 (360M) parameter gradients, one AdamW step bit-exact against PyTorch's fused AdamW, CUDA graph replay bit-identical to eager, and loss curves (1000 steps against PyTorch bf16 and fp32 for 135M; 100 steps against an fp32 CPU reference for 360M).
