// All lathe kernels. Shapes from the model header, tiles from the target header,
// B/T/N and AdamW hyperparameters from train.cu. Precision follows PyTorch bf16 autocast op by op (BENCH.md).
#pragma once
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

static_assert(DIM % 64 == 0 && FFN % 8 == 0 && HEAD_DIM % 16 == 0, "vector widths");
static_assert(T % ATT_TILE == 0 && VOCAB % (CE_THREADS * 8) == 0, "no tail handling");

constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

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

// res[n] = wte[tok[n]] from the fp32 master (PyTorch's embedding output is fp32).
__global__ void embed(float* res, const int* tok, const float* wte) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * DIM / 4) return;
    int n = i / (DIM / 4), c = i % (DIM / 4) * 4;
    *(float4*)(res + (size_t)n * DIM + c) = *(const float4*)(wte + (size_t)tok[n] * DIM + c);
}

// rsqrt(mean(x^2) + eps) for a row held lane-strided as float2. Shared by fwd and bwd so both get the same r.
__device__ __forceinline__ float row_rstd(const float2 (&v)[DIM / 64]) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) ss += v[i].x * v[i].x + v[i].y * v[i].y;
    return rsqrtf(warp_sum(ss) / DIM + RMS_EPS);
}

// One warp per row. x = res (+ branch, written to res_out in fp32), out = bf16(x * r * w). fp32 weight.
__global__ void rmsnorm(bf16* out, float* res_out, const float* res, const bf16* branch, const float* w) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const float2* x = (const float2*)(res + (size_t)row * DIM);
    float2 v[DIM / 64];
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) {
        v[i] = x[lane + 32 * i];
        if (branch) {
            float2 b = __bfloat1622float2(((const bf162*)(branch + (size_t)row * DIM))[lane + 32 * i]);
            v[i].x += b.x;
            v[i].y += b.y;
            ((float2*)(res_out + (size_t)row * DIM))[lane + 32 * i] = v[i];
        }
    }
    float r = row_rstd(v);
    bf162* y = (bf162*)(out + (size_t)row * DIM);
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) {
        float2 g = ((const float2*)w)[lane + 32 * i];
        y[lane + 32 * i] = __floats2bfloat162_rn(v[i].x * r * g.x, v[i].y * r * g.y);
    }
}

// One warp per row. dx of out = x * r * w given dy; dres = dres_in + dx (fp32, in place allowed),
// dres_bf = bf16(dres) for the branch matmul, dw_rows[row] = dy * (x * r) for the weight grad.
__global__ void rmsnorm_bwd(float* dres, bf16* dres_bf, float* dw_rows, const float* dres_in, const bf16* dy,
                            const float* x, const float* w) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const size_t o = (size_t)row * DIM;
    float2 v[DIM / 64], d[DIM / 64], g[DIM / 64];
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) {
        v[i] = ((const float2*)(x + o))[lane + 32 * i];
        d[i] = __bfloat1622float2(((const bf162*)(dy + o))[lane + 32 * i]);
        float2 ww = ((const float2*)w)[lane + 32 * i];
        g[i] = {d[i].x * ww.x, d[i].y * ww.y};
        dot += g[i].x * v[i].x + g[i].y * v[i].y;
    }
    const float r = row_rstd(v), c = r * r * warp_sum(dot) / DIM;
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) {
        float2 acc = dres_in ? ((const float2*)(dres_in + o))[lane + 32 * i] : float2{0.f, 0.f};
        acc = {acc.x + r * (g[i].x - v[i].x * c), acc.y + r * (g[i].y - v[i].y * c)};
        ((float2*)(dres + o))[lane + 32 * i] = acc;
        if (dres_bf) ((bf162*)(dres_bf + o))[lane + 32 * i] = __floats2bfloat162_rn(acc.x, acc.y);
        ((float2*)(dw_rows + o))[lane + 32 * i] = {d[i].x * (v[i].x * r), d[i].y * (v[i].y * r)};
    }
}

// out[c] = sum over rows of rows[n][c]. Grid DIM/32, block (32, 8); fixed order, no atomics.
__global__ void colsum(float* out, const float* rows) {
    const int c = blockIdx.x * 32 + threadIdx.x, y = threadIdx.y;
    float acc = 0.f;
    for (int n = y; n < N; n += 8) acc += rows[(size_t)n * DIM + c];
    __shared__ float part[8][32];
    part[y][threadIdx.x] = acc;
    __syncthreads();
    if (y == 0) {
        for (int k = 1; k < 8; k++) acc += part[k][threadIdx.x];
        out[c] = acc;
    }
}

// In place on the q and k heads of qkv [N][QKV_DIM]. Non-interleaved: rotates pairs (d, d + HEAD_DIM/2).
// cs = [T][HEAD_DIM/2] of (cos, sin). BWD applies the inverse rotation to gradients.
template <bool BWD>
__global__ void rope(bf16* qkv, const float2* cs) {
    constexpr int HALF = HEAD_DIM / 2, ROT = HEADS + KV_HEADS;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * ROT * HALF) return;
    int d = i % HALF, n = i / (ROT * HALF);
    bf16* x = qkv + (size_t)n * QKV_DIM + i / HALF % ROT * HEAD_DIM;
    float2 c = cs[n % T * HALF + d];
    float s = BWD ? -c.y : c.y;
    float x1 = __bfloat162float(x[d]), x2 = __bfloat162float(x[d + HALF]);
    x[d] = __float2bfloat16(x1 * c.x - x2 * s);
    x[d + HALF] = __float2bfloat16(x2 * c.x + x1 * s);
}

// Causal GQA attention, online softmax in fp32, one thread per query row.
// Block = (query tile, head, batch); each K/V tile is staged in shared memory and read by broadcast.
// qkv [N][QKV_DIM] post-RoPE, out [N][DIM], lse [B][HEADS][T] in log2 units of the scaled scores.
__global__ void __launch_bounds__(ATT_TILE) attention_fwd(bf16* out, float* lse, const bf16* qkv) {
    constexpr int D = HEAD_DIM, CH = 16;  // CH keys per softmax rescale
    static_assert(ATT_TILE % CH == 0, "chunking");
    const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const int kvh = h / (HEADS / KV_HEADS);
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    __shared__ __align__(16) bf16 ks[ATT_TILE][D], vs[ATT_TILE][D];

    const float scale = LOG2E / sqrtf((float)D);  // softmax via exp2
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
    const int i = qt * ATT_TILE + t;
    lse[((size_t)b * HEADS + h) * T + i] = m + log2f(l);
    bf16* orow = out + (size_t)(b * T + i) * DIM + h * D;
#pragma unroll
    for (int c = 0; c < D; c += 8) {
        float r[8];
#pragma unroll
        for (int e = 0; e < 8; e++) r[e] = o[c + e] * inv;
        store8(orow + c, r);
    }
}

// Attention backward, fp32 math. Two threads per row, each owning half of the head dims; partial dot
// products are combined with one shuffle. P = exp2(s - lse) is recomputed from the forward lse.

// dq for one query tile of one head. Also writes Dsum = rowsum(dO * O), which attention_bwd_dkv reads.
__global__ void __launch_bounds__(2 * ATT_TILE) attention_bwd_dq(bf16* dqkv, float* Dsum, const bf16* qkv,
                                                                 const bf16* out, const bf16* dout,
                                                                 const float* lse) {
    constexpr int D = HEAD_DIM, HD = D / 2;
    const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int r = threadIdx.x / 2, off = threadIdx.x % 2 * HD, i = qt * ATT_TILE + r;
    const int kvh = h / (HEADS / KV_HEADS);
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    const size_t n = (size_t)b * T + i, row = ((size_t)b * HEADS + h) * T + i;
    __shared__ __align__(16) bf16 ks[ATT_TILE][D], vs[ATT_TILE][D];

    const float scale = LOG2E / sqrtf((float)D);
    float q[HD], dO[HD], dq[HD], dd = 0.f;
#pragma unroll
    for (int c = 0; c < HD; c += 8) {
        float o8[8];
        load8(q + c, seq + (size_t)i * QKV_DIM + h * D + off + c);
        load8(dO + c, dout + n * DIM + h * D + off + c);
        load8(o8, out + n * DIM + h * D + off + c);
#pragma unroll
        for (int e = 0; e < 8; e++) dd += dO[c + e] * o8[e];
    }
    dd += __shfl_xor_sync(0xffffffff, dd, 1);
    if (off == 0) Dsum[row] = dd;
#pragma unroll
    for (int d = 0; d < HD; d++) {
        q[d] *= scale;
        dq[d] = 0.f;
    }
    const float l2 = lse[row];

    for (int kt = 0; kt <= qt; kt++) {
        __syncthreads();
        const bf16* krow = seq + (size_t)(kt * ATT_TILE + r) * QKV_DIM + (HEADS + kvh) * D + off;
#pragma unroll
        for (int c = 0; c < HD; c += 8) {
            *(uint4*)&ks[r][off + c] = *(const uint4*)(krow + c);
            *(uint4*)&vs[r][off + c] = *(const uint4*)(krow + KV_HEADS * D + c);
        }
        __syncthreads();
        for (int j = 0; j < ATT_TILE; j++) {
            float s = 0.f, dp = 0.f, k8[8], v8[8];
#pragma unroll
            for (int c = 0; c < HD; c += 8) {
                load8(k8, &ks[j][off + c]);
                load8(v8, &vs[j][off + c]);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    s += q[c + e] * k8[e];
                    dp += dO[c + e] * v8[e];
                }
            }
            s += __shfl_xor_sync(0xffffffff, s, 1);
            dp += __shfl_xor_sync(0xffffffff, dp, 1);
            const float p = kt < qt || j <= r ? exp2f(s - l2) : 0.f, ds = p * (dp - dd);
#pragma unroll
            for (int c = 0; c < HD; c += 8) {
                load8(k8, &ks[j][off + c]);
#pragma unroll
                for (int e = 0; e < 8; e++) dq[c + e] += ds * k8[e];
            }
        }
    }
    float w[8];
#pragma unroll
    for (int c = 0; c < HD; c += 8) {
#pragma unroll
        for (int e = 0; e < 8; e++) w[e] = dq[c + e] * (scale / LOG2E);
        store8(dqkv + n * QKV_DIM + h * D + off + c, w);
    }
}

// dk, dv for one key tile of one kv head, summed over the query heads sharing it.
__global__ void __launch_bounds__(2 * ATT_TILE) attention_bwd_dkv(bf16* dqkv, const bf16* qkv, const bf16* dout,
                                                                  const float* lse, const float* Dsum) {
    constexpr int D = HEAD_DIM, HD = D / 2, G = HEADS / KV_HEADS;
    const int kt = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
    const int r = threadIdx.x / 2, off = threadIdx.x % 2 * HD, j = kt * ATT_TILE + r;
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    __shared__ __align__(16) bf16 qs[ATT_TILE][D], dos[ATT_TILE][D];
    __shared__ float ls[ATT_TILE], dds[ATT_TILE];

    const float scale = LOG2E / sqrtf((float)D);
    float k[HD], v[HD], dk[HD], dv[HD];
#pragma unroll
    for (int c = 0; c < HD; c += 8) {
        load8(k + c, seq + (size_t)j * QKV_DIM + (HEADS + g) * D + off + c);
        load8(v + c, seq + (size_t)j * QKV_DIM + (HEADS + KV_HEADS + g) * D + off + c);
    }
#pragma unroll
    for (int d = 0; d < HD; d++) dk[d] = dv[d] = 0.f;

    for (int h = g * G; h < (g + 1) * G; h++) {
        for (int qt = kt; qt < T / ATT_TILE; qt++) {
            __syncthreads();
            const int i = qt * ATT_TILE + r;
            const size_t n = (size_t)b * T + i;
#pragma unroll
            for (int c = 0; c < HD; c += 8) {
                *(uint4*)&qs[r][off + c] = *(const uint4*)(seq + (size_t)i * QKV_DIM + h * D + off + c);
                *(uint4*)&dos[r][off + c] = *(const uint4*)(dout + n * DIM + h * D + off + c);
            }
            if (off == 0) {
                ls[r] = lse[((size_t)b * HEADS + h) * T + i];
                dds[r] = Dsum[((size_t)b * HEADS + h) * T + i];
            }
            __syncthreads();
            for (int ii = 0; ii < ATT_TILE; ii++) {
                float s = 0.f, dp = 0.f, q8[8], o8[8];
#pragma unroll
                for (int c = 0; c < HD; c += 8) {
                    load8(q8, &qs[ii][off + c]);
                    load8(o8, &dos[ii][off + c]);
#pragma unroll
                    for (int e = 0; e < 8; e++) {
                        s += q8[e] * k[c + e];
                        dp += o8[e] * v[c + e];
                    }
                }
                s += __shfl_xor_sync(0xffffffff, s, 1);
                dp += __shfl_xor_sync(0xffffffff, dp, 1);
                const float p = qt > kt || ii >= r ? exp2f(s * scale - ls[ii]) : 0.f, ds = p * (dp - dds[ii]);
#pragma unroll
                for (int c = 0; c < HD; c += 8) {
                    load8(q8, &qs[ii][off + c]);
                    load8(o8, &dos[ii][off + c]);
#pragma unroll
                    for (int e = 0; e < 8; e++) {
                        dv[c + e] += p * o8[e];
                        dk[c + e] += ds * q8[e];
                    }
                }
            }
        }
    }
    const size_t n = (size_t)b * T + j;
    float w[8];
#pragma unroll
    for (int c = 0; c < HD; c += 8) {
#pragma unroll
        for (int e = 0; e < 8; e++) w[e] = dk[c + e] * (scale / LOG2E);
        store8(dqkv + n * QKV_DIM + (HEADS + g) * D + off + c, w);
        store8(dqkv + n * QKV_DIM + (HEADS + KV_HEADS + g) * D + off + c, dv + c);
    }
}

// act = bf16(bf16(silu(g)) * u), with gu = [N][gate FFN | up FFN]. Two roundings, as PyTorch silu then mul.
__global__ void swiglu(bf16* act, const bf16* gu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * FFN / 8) return;
    int n = i / (FFN / 8), c = i % (FFN / 8) * 8;
    float g[8], u[8];
    load8(g, gu + (size_t)n * 2 * FFN + c);
    load8(u, gu + (size_t)n * 2 * FFN + FFN + c);
#pragma unroll
    for (int k = 0; k < 8; k++) g[k] = round_bf16(g[k] / (1.f + expf(-g[k]))) * u[k];
    store8(act + (size_t)n * FFN + c, g);
}

// PyTorch mul and silu backward: ds = bf16(dact * u), du = bf16(dact * s), dg = bf16(ds * silu'(g)).
__global__ void swiglu_bwd(bf16* dgu, const bf16* dact, const bf16* gu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * FFN / 8) return;
    int n = i / (FFN / 8), c = i % (FFN / 8) * 8;
    float g[8], u[8], da[8], dg[8], du[8];
    load8(g, gu + (size_t)n * 2 * FFN + c);
    load8(u, gu + (size_t)n * 2 * FFN + FFN + c);
    load8(da, dact + (size_t)n * FFN + c);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const float sig = 1.f / (1.f + expf(-g[k])), s = round_bf16(g[k] / (1.f + expf(-g[k])));
        du[k] = da[k] * s;
        dg[k] = round_bf16(da[k] * u[k]) * sig * (1.f + g[k] * (1.f - sig));
    }
    store8(dgu + (size_t)n * 2 * FFN + c, dg);
    store8(dgu + (size_t)n * 2 * FFN + FFN + c, du);
}

// One block per row. losses[n] = logsumexp(x) - x[target] in fp32, then x is overwritten in place
// with dlogits = (softmax(x) - onehot) / N in bf16 (mean-reduced cross entropy).
__global__ void __launch_bounds__(CE_THREADS) cross_entropy(float* losses, bf16* logits, const int* targets) {
    bf16* x = logits + (size_t)blockIdx.x * VOCAB;
    const int tgt = targets[blockIdx.x];
    const float xt = __bfloat162float(x[tgt]);
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
    __shared__ float wm[CE_THREADS / 32], ws[CE_THREADS / 32], lse_s;
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
        lse_s = m + logf(s);
        losses[blockIdx.x] = lse_s - xt;
    }
    __syncthreads();
    const float lse = lse_s, inv_n = 1.f / N;
    for (int c = threadIdx.x * 8; c < VOCAB; c += CE_THREADS * 8) {
        float v[8];
        load8(v, x + c);
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] = (expf(v[k] - lse) - (c + k == tgt ? 1.f : 0.f)) * inv_n;
        store8(x + c, v);
    }
}

// loss = mean(losses), one block, fixed reduction order.
__global__ void __launch_bounds__(1024) mean_loss(float* loss, const float* losses) {
    __shared__ float part[1024];
    float acc = 0.f;
    for (int i = threadIdx.x; i < N; i += 1024) acc += losses[i];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (int w = 512; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) *loss = part[0] / N;
}

// gemb[tok] += sum of dres rows for that token. seg = [perm N][starts N+1][count 1] built on the host:
// positions sorted by token, so each block sums one token's rows in a fixed order (no atomics).
__global__ void __launch_bounds__(DIM / 4) embed_bwd(float* gemb, const float* dres, const int* tok, const int* seg) {
    const int* perm = seg;
    const int* starts = seg + N;
    if ((int)blockIdx.x >= starts[N + 1]) return;
    const int s0 = starts[blockIdx.x], s1 = starts[blockIdx.x + 1], c = threadIdx.x * 4;
    float4 acc = {0.f, 0.f, 0.f, 0.f};
    for (int k = s0; k < s1; k++) {
        float4 d = *(const float4*)(dres + (size_t)perm[k] * DIM + c);
        acc.x += d.x;
        acc.y += d.y;
        acc.z += d.z;
        acc.w += d.w;
    }
    float4* g = (float4*)(gemb + (size_t)tok[perm[s0]] * DIM + c);
    float4 cur = *g;
    *g = {cur.x + acc.x, cur.y + acc.y, cur.z + acc.z, cur.w + acc.w};
}

// Fused AdamW, same math as PyTorch's fused kernel (fp32), then refreshes the bf16 working copy.
__global__ void adamw(float* p, bf16* pb, const float* g, float* m, float* v, size_t n, float step) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    float pi = p[i], gi = g[i], mi = m[i], vi = v[i];
    pi -= LR * WEIGHT_DECAY * pi;
    mi = fmaf(BETA1, mi, fmaf(-BETA1, gi, gi));
    vi = fmaf(BETA2, vi, fmaf(-BETA2, gi * gi, gi * gi));
    pi -= (LR / bc1) * mi / (sqrtf(vi) / bc2_sqrt + ADAM_EPS);
    p[i] = pi;
    m[i] = mi;
    v[i] = vi;
    pb[i] = __float2bfloat16(pi);
}
