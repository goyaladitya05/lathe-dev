// RTX 4050 Laptop: Ada sm_89, 20 SMs, 6 GB, 24 MB L2, 99 KB shared memory per block (opt-in).
#pragma once
#include <cstddef>

constexpr int TARGET_CC = 89, TARGET_SMS = 20;  // train.cu checks the device against these
constexpr int CE_THREADS = 512;                 // threads per cross-entropy row
constexpr size_t CUBLAS_WORKSPACE = 32u << 20;  // cuBLASLt workspace bytes
constexpr double SOL_BF16_TFLOPS = 25.61;       // best measured: ref/baseline.py --mode sol (16384^3, fp32 accumulate)
constexpr int GBM = 64, GWM = 16, GST = 3;       // fused-GEMM tiles (kernels.cuh), swept on sm89 (BENCH.md profiles)
constexpr bool FUSED_QKV_ROPE = true, FUSED_DOWN_DX = true;  // fused qkv + RoPE and down dX + SwiGLU GEMMs (else cuBLASLt + a pass)
constexpr int ATT_SPLIT = 1;  // query-tile ranges per attention dK/dV block (kernels.cuh attention_bwd_dkv)

// AdamW pass schedule (train.cu backward). Layer l + 1's pass starts with kernel ADAMW_AT of layer l's backward:
// 0 down dX + SwiGLU, 1 down dW, 2 gate_up dX, 3 gate_up dW, 4 o dX, 5 o dW, 6 attention, 7 qkv dX, 8 qkv dW; it runs
// on ADAMW_BLOCKS persistent blocks. ADAMW_SPLIT_LAST: layer 0's o, gate_up and down go beside its own attention, so
// only its qkv waits for the end of backward. ADAMW_AT8, ADAMW_BLOCKS8: the same for ADAM8=1 builds.
// ATT_HEAD_FASTEST: attention grids with the head index fastest (kernels.cuh att_grid).
#if defined(MODEL_smol135)
// Tuned at 1x2048 (chisel e049, e051).
constexpr int ADAMW_AT = 2;
constexpr int ADAMW_BLOCKS = 7;
constexpr bool ADAMW_SPLIT_LAST = false;
// 8-bit moments: the same schedule until tuned.
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = ADAMW_BLOCKS;
// +0.74% at 1x2048, +0.17% at 4x512 (chisel e059).
constexpr bool ATT_HEAD_FASTEST = true;
#elif defined(MODEL_smol360)
// Tuned at 1x704 (chisel e054: 14 blocks from the next layer's down dX, +8.3%).
constexpr int ADAMW_AT = 0;
constexpr int ADAMW_BLOCKS = 14;
constexpr bool ADAMW_SPLIT_LAST = false;
// 8-bit moments (16.25 instead of 28 bytes per parameter per pass), width by tokens per step (chisel e066, 8-bit at
// 1xT): 16 blocks +1.4% over 14 at 704, even at 1536, 14 +0.7-0.9% over 16 at 2304.
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = BATCH * SEQ <= 1536 ? 16 : 14;
// -0.22% at 1x704 (chisel e059).
constexpr bool ATT_HEAD_FASTEST = false;
#else
#error "no schedule for this model on rtx4050"
#endif
