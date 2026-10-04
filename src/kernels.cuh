// All lathe kernels. Shapes from the model header, tiles from the target header, B/T/N from train.cu.
#pragma once
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

static_assert(DIM % 64 == 0 && FFN % 8 == 0 && HEAD_DIM % 8 == 0, "vector widths");
static_assert(T % ATT_TILE == 0 && VOCAB % (CE_THREADS * 8) == 0, "no tail handling");

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// 8 bf16 <-> 8 floats via one 16-byte access.
__device__ __forceinline__ void load8(float* f, const bf16* p) {
    uint4 u = *(const uint4*)p;
    const bf162* h = (const bf162*)&u;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ void store8(bf16* p, const float* f) {
    uint4 u;
    bf162* h = (bf162*)&u;
#pragma unroll
    for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    *(uint4*)p = u;
}

__global__ void f32_to_bf16(bf16* out, const float* in, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

// out[n] = wte[tok[n]], one 16-byte chunk per thread.
__global__ void embed(bf16* out, const int* tok, const bf16* wte) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * DIM / 8) return;
    int n = i / (DIM / 8), c = i % (DIM / 8) * 8;
    *(uint4*)(out + (size_t)n * DIM + c) = *(const uint4*)(wte + (size_t)tok[n] * DIM + c);
}

// One warp per row: out = in * rsqrt(mean(in^2) + eps) * w, fp32 math, one rounding.
__global__ void rmsnorm(bf16* out, const bf16* in, const bf16* w) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const bf162* x = (const bf162*)(in + (size_t)row * DIM);
    float2 v[DIM / 64];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) {
        v[i] = __bfloat1622float2(x[lane + 32 * i]);
        ss += v[i].x * v[i].x + v[i].y * v[i].y;
    }
    float r = rsqrtf(warp_sum(ss) / DIM + RMS_EPS);
    bf162* y = (bf162*)(out + (size_t)row * DIM);
    const bf162* w2 = (const bf162*)w;
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) {
        float2 g = __bfloat1622float2(w2[lane + 32 * i]);
        y[lane + 32 * i] = __floats2bfloat162_rn(v[i].x * r * g.x, v[i].y * r * g.y);
    }
}

// In place on the q and k heads of qkv [N][QKV_DIM]. Non-interleaved: rotates pairs (d, d + HEAD_DIM/2).
// cs = [T][HEAD_DIM/2] of (cos, sin).
__global__ void rope(bf16* qkv, const float2* cs) {
    constexpr int HALF = HEAD_DIM / 2, ROT = HEADS + KV_HEADS;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * ROT * HALF) return;
    int d = i % HALF, n = i / (ROT * HALF);
    bf16* x = qkv + (size_t)n * QKV_DIM + i / HALF % ROT * HEAD_DIM;
    float2 c = cs[n % T * HALF + d];
    float x1 = __bfloat162float(x[d]), x2 = __bfloat162float(x[d + HALF]);
    x[d] = __float2bfloat16(x1 * c.x - x2 * c.y);
    x[d + HALF] = __float2bfloat16(x2 * c.x + x1 * c.y);
}

// Causal GQA attention, online softmax in fp32, one thread per query row.
// Block = (query tile, head, batch); each K/V tile is staged in shared memory and read by broadcast.
// qkv [N][QKV_DIM] post-RoPE, out [N][DIM].
__global__ void __launch_bounds__(ATT_TILE) attention_fwd(bf16* out, const bf16* qkv) {
    constexpr int D = HEAD_DIM, CH = 16;  // CH keys per softmax rescale
    static_assert(ATT_TILE % CH == 0, "chunking");
    const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const int kvh = h / (HEADS / KV_HEADS);
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    __shared__ __align__(16) bf16 ks[ATT_TILE][D], vs[ATT_TILE][D];

    const float scale = 1.4426950408889634f / sqrtf((float)D);  // log2(e)/sqrt(D), softmax via exp2
    float q[D], o[D];
    const bf16* qrow = seq + (size_t)(qt * ATT_TILE + t) * QKV_DIM + h * D;
#pragma unroll
    for (int c = 0; c < D; c += 8) load8(q + c, qrow + c);
#pragma unroll
    for (int d = 0; d < D; d++) {
        q[d] *= scale;
        o[d] = 0.f;
    }
    float m = -INFINITY, l = 0.f;

    for (int kt = 0; kt <= qt; kt++) {
        __syncthreads();
        const bf16* krow = seq + (size_t)(kt * ATT_TILE + t) * QKV_DIM + (HEADS + kvh) * D;
        const bf16* vrow = krow + KV_HEADS * D;
#pragma unroll
        for (int c = 0; c < D; c += 8) {
            *(uint4*)&ks[t][c] = *(const uint4*)(krow + c);
            *(uint4*)&vs[t][c] = *(const uint4*)(vrow + c);
        }
        __syncthreads();
        const int nk = kt == qt ? t + 1 : ATT_TILE;  // causal: keys <= query
        for (int j0 = 0; j0 < nk; j0 += CH) {
            float s[CH], mc = m;
#pragma unroll
            for (int j = 0; j < CH; j++) {
                float a = 0.f;
#pragma unroll
                for (int c = 0; c < D; c += 8) {
                    float k8[8];
                    load8(k8, &ks[j0 + j][c]);
#pragma unroll
                    for (int e = 0; e < 8; e++) a += q[c + e] * k8[e];
                }
                s[j] = j0 + j < nk ? a : -INFINITY;
                mc = fmaxf(mc, s[j]);
            }
            const float corr = exp2f(m - mc);
            l *= corr;
#pragma unroll
            for (int d = 0; d < D; d++) o[d] *= corr;
#pragma unroll
            for (int j = 0; j < CH; j++) {
                const float p = exp2f(s[j] - mc);
                l += p;
#pragma unroll
                for (int c = 0; c < D; c += 8) {
                    float v8[8];
                    load8(v8, &vs[j0 + j][c]);
#pragma unroll
                    for (int e = 0; e < 8; e++) o[c + e] += p * v8[e];
                }
            }
            m = mc;
        }
    }
    const float inv = 1.f / l;
    bf16* orow = out + (size_t)(b * T + qt * ATT_TILE + t) * DIM + h * D;
#pragma unroll
    for (int c = 0; c < D; c += 8) {
        float r[8];
#pragma unroll
        for (int e = 0; e < 8; e++) r[e] = o[c + e] * inv;
        store8(orow + c, r);
    }
}

// act = silu(gate) * up, with gu = [N][gate FFN | up FFN].
__global__ void swiglu(bf16* act, const bf16* gu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * FFN / 8) return;
    int n = i / (FFN / 8), c = i % (FFN / 8) * 8;
    float g[8], u[8];
    load8(g, gu + (size_t)n * 2 * FFN + c);
    load8(u, gu + (size_t)n * 2 * FFN + FFN + c);
#pragma unroll
    for (int k = 0; k < 8; k++) g[k] = g[k] / (1.f + expf(-g[k])) * u[k];
    store8(act + (size_t)n * FFN + c, g);
}

// One block per row: losses[n] = logsumexp(logits[n]) - logits[n][targets[n]].
__global__ void __launch_bounds__(CE_THREADS) cross_entropy_fwd(float* losses, const bf16* logits, const int* targets) {
    const bf16* x = logits + (size_t)blockIdx.x * VOCAB;
    float m = -INFINITY, s = 0.f;
    for (int c = threadIdx.x * 8; c < VOCAB; c += CE_THREADS * 8) {
        float v[8], mc = m;
        load8(v, x + c);
#pragma unroll
        for (int k = 0; k < 8; k++) mc = fmaxf(mc, v[k]);
        s *= expf(m - mc);
#pragma unroll
        for (int k = 0; k < 8; k++) s += expf(v[k] - mc);
        m = mc;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffff, m, o), s2 = __shfl_xor_sync(0xffffffff, s, o);
        float mn = fmaxf(m, m2);
        s = s * expf(m - mn) + s2 * expf(m2 - mn);
        m = mn;
    }
    __shared__ float wm[CE_THREADS / 32], ws[CE_THREADS / 32];
    if (threadIdx.x % 32 == 0) {
        wm[threadIdx.x / 32] = m;
        ws[threadIdx.x / 32] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < CE_THREADS / 32; i++) {
            float mn = fmaxf(m, wm[i]);
            s = s * expf(m - mn) + ws[i] * expf(wm[i] - mn);
            m = mn;
        }
        losses[blockIdx.x] = m + logf(s) - __bfloat162float(x[targets[blockIdx.x]]);
    }
}
