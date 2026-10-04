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
