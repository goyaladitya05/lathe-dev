// NVIDIA L4: Ada sm_89, 58 SMs, 24 GB, 48 MB L2, 72 W power cap.
#pragma once
#include <cstddef>

constexpr int TARGET_CC = 89, TARGET_SMS = 58;  // train.cu checks the device against these
constexpr int CE_THREADS = 512;                 // threads per cross-entropy row
constexpr size_t CUBLAS_WORKSPACE = 32u << 20;  // cuBLASLt workspace bytes
constexpr double SOL_BF16_TFLOPS = 55.94;       // best measured: ref/baseline.py --mode sol (8192^3)
constexpr int GBM = 64, GWM = 16, GST = 3;       // fused-GEMM tiles (kernels.cuh), rtx4050's until tuned
constexpr bool FUSED_QKV_ROPE = true, FUSED_DOWN_DX = false;  // cuBLASLt down dX + SwiGLU-bwd pass: +3.5% / +0.9% (j60)
constexpr int ATT_SPLIT = 1;  // query-tile ranges per attention dK/dV block: every split is slower here (j40)

// AdamW pass schedule and attention grids: see targets/rtx4050.h. Blocks without a note: rtx4050's values, untuned.
#if defined(MODEL_smol135)
// At 1x2048: beside the gate_up dW GEMM on 29 blocks (j50, cuBLASLt dW; 4 / 29 with own dW, +2.5% over rtx4050's 2 / 14).
constexpr int ADAMW_AT = 3;
constexpr int ADAMW_BLOCKS = 29;
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = ADAMW_BLOCKS;
constexpr bool ATT_HEAD_FASTEST = true;
#elif defined(MODEL_smol360)
// At 1x2048: fp32 state as 135M (j50); 8-bit state from the o dX GEMM on 116 blocks (j30, own dW).
constexpr int ADAMW_AT = 3;
constexpr int ADAMW_BLOCKS = 29;
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = 4;
constexpr int ADAMW_BLOCKS8 = 116;
constexpr bool ATT_HEAD_FASTEST = true;
#else
#error "no schedule for this model on l4"
#endif
