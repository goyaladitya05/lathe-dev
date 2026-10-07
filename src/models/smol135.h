// SmolLM2-135M, from HuggingFaceTB/SmolLM2-135M config.json @ 93efa2f.
// ref/test.py checks these against config.json; train.cu checks them against the weight file header.
#pragma once

constexpr int VOCAB = 49152;             // vocab_size
constexpr int DIM = 576;                 // hidden_size
constexpr int LAYERS = 30;               // num_hidden_layers
constexpr int HEADS = 9;                 // num_attention_heads
constexpr int KV_HEADS = 3;              // num_key_value_heads
constexpr int FFN = 1536;                // intermediate_size
constexpr float ROPE_THETA = 100000.0f;  // rope_theta
constexpr float RMS_EPS = 1e-5f;         // rms_norm_eps

constexpr int HEAD_DIM = DIM / HEADS;                       // 64
constexpr int QKV_DIM = (HEADS + 2 * KV_HEADS) * HEAD_DIM;  // 960, fused q|k|v projection

// AdamW pass schedule (train.cu backward). Layer l + 1's pass starts with kernel ADAMW_AT of layer l's backward:
// 0 down dX + SwiGLU, 1 down dW, 2 gate_up dX, 3 gate_up dW, 4 o dX, 5 o dW, 6 attention, 7 qkv dX, 8 qkv dW; it runs
// on ADAMW_BLOCKS persistent blocks. ADAMW_SPLIT_LAST: layer 0's o, gate_up and down go beside its own attention, so
// only its qkv waits for the end of backward. Tuned at 1x2048 (chisel e049, e051).
constexpr int ADAMW_AT = 2;
constexpr int ADAMW_BLOCKS = 7;
constexpr bool ADAMW_SPLIT_LAST = false;
// ADAM8=1 builds (8-bit moments): the same schedule until tuned.
constexpr int ADAMW_AT8 = ADAMW_AT;
constexpr int ADAMW_BLOCKS8 = ADAMW_BLOCKS;
// FP8=1 builds: AdamW pass start (untuned: the fp32 schedule's start); width as ADAMW_BLOCKS8 / ADAMW_BLOCKS.
constexpr int ADAMW_AT_FP8 = 2;
// Attention grids with the head index fastest (kernels.cuh att_grid): +0.74% at 1x2048, +0.17% at 4x512 (chisel e059).
constexpr bool ATT_HEAD_FASTEST = true;
