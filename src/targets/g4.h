// NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab G4): sm_120, 188 SMs, 96 GB GDDR7, 600 W.
#pragma once
#include <cstddef>

constexpr int TARGET_CC = 120, TARGET_SMS = 188;  // train.cu checks the device against these
constexpr int CE_THREADS = 512;                 // threads per cross-entropy row
constexpr size_t CUBLAS_WORKSPACE = 32u << 20;  // cuBLASLt workspace bytes
constexpr double SOL_BF16_TFLOPS = 418.57;      // best measured: ref/baseline.py --mode sol (16384^3)
constexpr int GBM = 64, GWM = 16, GST = 3;       // fused-GEMM tiles (kernels.cuh): 3 and 4 stages measure the same
constexpr bool FUSED_QKV_ROPE = true, FUSED_DOWN_DX = false;  // fused qkv + RoPE, cuBLASLt down dX + SwiGLU-bwd pass
constexpr int ATT_SPLIT = 6;  // query-tile ranges per attention dK/dV block: +18-21% over 1, a plateau from 4 on

// AdamW pass schedule and attention grids: see targets/rtx4050.h. Measured at 1x2048 on the default build (own dW,
// which beats cuBLASLt dW here).
#if defined(MODEL_smol135)
constexpr int ADAMW_AT = 2;
constexpr int ADAMW_BLOCKS = 47;
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = ADAMW_BLOCKS;
constexpr bool ATT_HEAD_FASTEST = true;
#elif defined(MODEL_smol360)
constexpr int ADAMW_AT = 1;
constexpr int ADAMW_BLOCKS = 47;
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = 1;
constexpr int ADAMW_BLOCKS8 = 94;
constexpr bool ATT_HEAD_FASTEST = true;
#else
#error "no schedule for this model on g4"
#endif
