# Release posts (drafts; numbers from the release sitting)

Repo: https://github.com/goyaladitya05/lathe

## Hacker News

Title: Show HN: Lathe, a from-scratch CUDA trainer that beats torch.compile on a 6 GB laptop GPU

I wanted to know how far a single-purpose training engine could get past PyTorch on cheap hardware, so I wrote one for exactly two models (SmolLM2-135M and 360M) and one GPU (an RTX 4050 Laptop, 6 GB).

Everything is a compile-time constant: model, batch shape, GPU. That removes the op library, autograd and the allocator. There is one memory arena, a hand-written forward and backward, and the whole training step (forward, backward, AdamW) replays as one CUDA graph. Weight gradients never exist as fp32 tensors: each layer's gradients go through a small ring buffer to an AdamW pass that runs on a side stream while the next layer's backward is still running.

Results, same model, init, data and precision recipe as PyTorch:

- SmolLM2-135M: {CMP_COMPILE} faster than torch.compile (max-autotune), {CMP_EAGER} faster than eager, {CMP_MFU} MFU.
- SmolLM2-360M: plain PyTorch can't fit it in 6 GB at all (fp32 weights + grads + AdamW moments = 5.4 GiB). With torchao's 8-bit AdamW it fits, and its best setup runs {T360_BEST} tok/s. lathe, with its own FP8-coded optimizer state, runs {H360_TOKS} tok/s at {H360_MFU} MFU.

Correctness is the part I spent the most time on. Every kernel is checked against fp32 math, every parameter gradient against PyTorch, one AdamW step bit-exact, graph replay bit-identical to eager, and loss curves against PyTorch bf16 and fp32, each judged against PyTorch bf16's own error.

The README also lists what didn't work: a vocab-chunked cross entropy, atomic attention backward, PyTorch fp8 as a baseline (slower here, or NaN).

## Reddit (r/MachineLearning, [P])

Title: [P] lathe: a from-scratch C++/CUDA training engine for SmolLM2, {CMP_COMPILE} faster than torch.compile on a 6 GB laptop GPU, and it trains the 360M model that plain PyTorch can't fit

Body: same as the HN text, plus the 135M table from the README:

| batch x context | PyTorch eager | torch.compile | lathe, tuned |
|---|---|---|---|
| 1 x 2048 | {E1} | {C1} | {LT1} |
| 2 x 1024 | {E2} | {C2} | {LT2} |
| 4 x 512 | {E3} | {C3} | {LT3} |

Happy to answer questions about the kernels, the memory layout, or the test methodology.

## LinkedIn

I've released lathe, a training engine I wrote from scratch in C++/CUDA for one job: training SmolLM2 language models as fast as possible on a single 6 GB laptop GPU.

On SmolLM2-135M it trains {CMP_COMPILE} faster than torch.compile, at {CMP_MFU} of the GPU's measured peak. On SmolLM2-360M, a model plain PyTorch can't even fit in 6 GB, it trains at {H360_TOKS} tokens per second, against {T360_BEST} for the best PyTorch setup I could make fit (torch.compile with an 8-bit optimizer).

What made the difference: fixing every shape at compile time, one memory arena, a hand-written backward pass, the optimizer running inside the backward so the gradients never pile up in memory, an 8-bit optimizer state, and GEMM algorithms picked by measurement.

Every number is checked against PyTorch: gradients, the optimizer step bit for bit, and loss curves over hundreds of steps.

Code, results and the list of ideas that didn't work: https://github.com/goyaladitya05/lathe

## X thread

1/ I wrote a training engine from scratch in C++/CUDA for exactly two models (SmolLM2-135M, 360M) and one GPU (RTX 4050 Laptop, 6 GB).

It trains SmolLM2-135M {CMP_COMPILE} faster than torch.compile, at {CMP_MFU} MFU.

github.com/goyaladitya05/lathe

2/ SmolLM2-360M doesn't fit in plain PyTorch on 6 GB: fp32 weights, grads and AdamW moments alone are 5.4 GiB.

With torchao's 8-bit AdamW it fits: {T360_BEST} tok/s at best.
lathe: {H360_TOKS} tok/s, {H360_MFU} MFU.

3/ The trick is giving up generality. Model, batch and context are compile-time constants, so there is no op library, no autograd, no allocator. One arena, a hand-written backward, and the whole step replays as one CUDA graph.

4/ Weight gradients never exist as fp32 tensors. Each layer's dW goes through a small ring buffer to an AdamW pass on a side stream, which runs while the next layer's backward is still going. That leaves room for a much bigger batch.

5/ Two more levers: AdamW moments stored as FP8 E4M3 in blocks of 64 (2.1 bytes/param of optimizer state instead of 8), and cuBLASLt algorithms chosen by timing each GEMM rather than trusting the heuristic: up to +19% on single GEMMs.

6/ Correctness first: every kernel vs fp32 math, all 290 parameter grads vs PyTorch, one AdamW step bit-exact, graph replay bit-identical to eager, loss curves vs PyTorch bf16 and fp32.

7/ The README also lists what didn't work, with numbers. Code is MIT.
