// RTX 4050 Laptop: Ada sm_89, 20 SMs, 6 GB, 24 MB L2, 99 KB shared memory per block (opt-in).
#pragma once
#include <cstddef>

constexpr int ATT_TILE = 64;                    // query rows per block and key rows per smem tile
constexpr int CE_THREADS = 512;                 // threads per cross-entropy row
constexpr size_t CUBLAS_WORKSPACE = 32u << 20;  // cuBLASLt workspace bytes
