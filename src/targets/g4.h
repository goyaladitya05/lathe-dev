// NVIDIA RTX PRO 6000 Blackwell Server Edition (Colab G4): sm_120, 188 SMs, 96 GB GDDR7, 128 MB L2.
#pragma once
#include <cstddef>

constexpr int TARGET_CC = 120, TARGET_SMS = 188;  // train.cu checks the device against these
constexpr int CE_THREADS = 512;                 // threads per cross-entropy row
constexpr size_t CUBLAS_WORKSPACE = 32u << 20;  // cuBLASLt workspace bytes
constexpr double SOL_BF16_TFLOPS = 0;           // not measured yet: ref/baseline.py --mode sol on this GPU
constexpr int GBM = 64, GWM = 16, GST = 3;       // fused-GEMM tiles (kernels.cuh), rtx4050's until tuned

// AdamW pass schedule and attention grids: see targets/rtx4050.h. Untuned here: rtx4050's values until measured.
#if defined(MODEL_smol135)
constexpr int ADAMW_AT = 2;
constexpr int ADAMW_BLOCKS = 7;
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = ADAMW_BLOCKS;
constexpr bool ATT_HEAD_FASTEST = true;
#elif defined(MODEL_smol360)
constexpr int ADAMW_AT = 0;
constexpr int ADAMW_BLOCKS = 14;
constexpr bool ADAMW_SPLIT_LAST = false;
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = BATCH * SEQ <= 1536 ? 16 : 14;
constexpr bool ATT_HEAD_FASTEST = false;
#else
#error "no schedule for this model on g4"
#endif
