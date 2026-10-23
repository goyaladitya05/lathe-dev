// NVIDIA A100: Ampere sm_80, 108 SMs, 40 or 80 GB HBM2e, 40 MB L2, 163 KB shared memory per block (opt-in).
#pragma once
#include <cstddef>

constexpr int TARGET_CC = 80, TARGET_SMS = 108;  // train.cu checks the device against these
constexpr int CE_THREADS = 512;                 // threads per cross-entropy row
constexpr size_t CUBLAS_WORKSPACE = 32u << 20;  // cuBLASLt workspace bytes
constexpr double SOL_BF16_TFLOPS = 269.03;      // best measured: ref/baseline.py --mode sol (8192^3, A100-SXM4-40GB)
constexpr int GBM = 64, GWM = 16, GST = 4;       // fused-GEMM tiles (kernels.cuh): 4 stages +0.9% / +0.7% (135M / 360M 1x2048)
constexpr bool FUSED_QKV_ROPE = false, FUSED_DOWN_DX = false;  // cuBLASLt + RoPE / SwiGLU-bwd passes: +2.3-3.1% (q62)
constexpr int ATT_SPLIT = 2;  // query-tile ranges per attention dK/dV block: +6.3% at 135M 1x2048, +5.0% at 360M (q63)
constexpr int ATT_SPLIT_FWD = 1;  // key-tile ranges per attention forward block (kernels.cuh attention_fwd)

// AdamW pass schedule and attention grids: see targets/rtx4050.h. Blocks without a note: rtx4050's values, untuned.
#if defined(MODEL_smol135)
// Pass beside the o dW GEMM on 54 blocks: with the unfused GEMMs and the attention split, +0.7% over 4 (q64), which
// was best before them (+4.7% at 1x2048, +14.5% at 4x512 over rtx4050's 2 / 14).
constexpr int ADAMW_AT = 5;
constexpr int ADAMW_BLOCKS = 54;
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = ADAMW_BLOCKS;
constexpr bool ATT_HEAD_FASTEST = true;
#elif defined(MODEL_smol360)
// At 1x2048: fp32 state beside the gate_up dX GEMM on 36 blocks (+26% over rtx4050's 0 / 14); 8-bit state from the
// same kernel on 108 blocks. Head-fastest grids +2-3% at every schedule.
constexpr int ADAMW_AT = 2;
constexpr int ADAMW_BLOCKS = 54;  // 36 before the unfused GEMMs and the attention split, 54 +0.25% after (q64)
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = 2;
constexpr int ADAMW_BLOCKS8 = 108;
constexpr bool ATT_HEAD_FASTEST = true;
#else
#error "no schedule for this model on a100"
#endif
