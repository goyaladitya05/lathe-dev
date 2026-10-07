// SmolLM2-360M, from HuggingFaceTB/SmolLM2-360M config.json @ f8027fd.
// ref/test.py checks these against config.json; train.cu checks them against the weight file header.
#pragma once

constexpr int VOCAB = 49152;             // vocab_size
constexpr int DIM = 960;                 // hidden_size
constexpr int LAYERS = 32;               // num_hidden_layers
constexpr int HEADS = 15;                // num_attention_heads
constexpr int KV_HEADS = 5;              // num_key_value_heads
constexpr int FFN = 2560;                // intermediate_size
constexpr float ROPE_THETA = 100000.0f;  // rope_theta
constexpr float RMS_EPS = 1e-5f;         // rms_norm_eps

constexpr int HEAD_DIM = DIM / HEADS;                       // 64
constexpr int QKV_DIM = (HEADS + 2 * KV_HEADS) * HEAD_DIM;  // 1600, fused q|k|v projection

// AdamW pass schedule (train.cu backward). Layer l + 1's pass starts with kernel ADAMW_AT of layer l's backward:
// 0 down dX + SwiGLU, 1 down dW, 2 gate_up dX, 3 gate_up dW, 4 o dX, 5 o dW, 6 attention, 7 qkv dX, 8 qkv dW; it runs
// on ADAMW_BLOCKS persistent blocks. ADAMW_SPLIT_LAST: layer 0's o, gate_up and down go beside its own attention, so
// only its qkv waits for the end of backward. Tuned at 1x704 (chisel e054: 14 blocks from the next layer's down dX, +8.3%).
constexpr int ADAMW_AT = 0;
constexpr int ADAMW_BLOCKS = 14;
constexpr bool ADAMW_SPLIT_LAST = false;
// ADAM8=1 builds (8-bit moments, 16.25 instead of 28 bytes per parameter per pass): start and width of their pass.
constexpr int ADAMW_AT8 = ADAMW_AT;
// 8-bit pass width by tokens per step (chisel e066, 8-bit at 1xT): 16 blocks +1.4% over 14 at 704, even at 1536, 14 +0.7-0.9%
// over 16 at 2304.
constexpr int ADAMW_BLOCKS8 = BATCH * SEQ <= 1536 ? 16 : 14;
// FP8=1 builds: AdamW pass start (beside the FP8 gate_up dX GEMM); width as ADAMW_BLOCKS8 / ADAMW_BLOCKS.
constexpr int ADAMW_AT_FP8 = 2;
// Attention grids with the head index fastest (kernels.cuh att_grid): -0.22% at 1x704 (chisel e059).
constexpr bool ATT_HEAD_FASTEST = false;
