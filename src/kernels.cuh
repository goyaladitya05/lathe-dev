// All lathe kernels. Shapes from the model header, tiles from the target header; B/T/N, AdamW
// hyperparameters and the parameter layout from train.cu. Precision follows PyTorch bf16 autocast (BENCH.md).
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

// dx of out = x * r * w given dy; dres = dres_in + dx (fp32, in place allowed), dres_bf = bf16(dres) for the
// branch matmul. One warp per row, NORM_ROWS rows per block; the block's weight-grad sum of dy * (x * r) goes
// to dw_part[block], reduced by colsum. Fixed order everywhere.
constexpr int NORM_ROWS = 16;
static_assert(N % NORM_ROWS == 0, "whole norm blocks");
__global__ void __launch_bounds__(256) rmsnorm_bwd(float* dres, bf16* dres_bf, float* dw_part, const float* dres_in,
                                                   const bf16* dy, const float* x, const float* w) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float2 acc[DIM / 64] = {};
    for (int k = 0; k < NORM_ROWS / 8; k++) {
        const size_t o = (size_t)(blockIdx.x * NORM_ROWS + warp * (NORM_ROWS / 8) + k) * DIM;
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
            float2 a = dres_in ? ((const float2*)(dres_in + o))[lane + 32 * i] : float2{0.f, 0.f};
            a = {a.x + r * (g[i].x - v[i].x * c), a.y + r * (g[i].y - v[i].y * c)};
            ((float2*)(dres + o))[lane + 32 * i] = a;
            if (dres_bf) ((bf162*)(dres_bf + o))[lane + 32 * i] = __floats2bfloat162_rn(a.x, a.y);
            acc[i].x += d[i].x * (v[i].x * r);
            acc[i].y += d[i].y * (v[i].y * r);
        }
    }
    __shared__ float2 part[8][DIM / 2];
#pragma unroll
    for (int i = 0; i < DIM / 64; i++) part[warp][lane + 32 * i] = acc[i];
    __syncthreads();
    for (int c = threadIdx.x; c < DIM / 2; c += 256) {
        float2 sum = part[0][c];
        for (int k = 1; k < 8; k++) sum = {sum.x + part[k][c].x, sum.y + part[k][c].y};
        ((float2*)(dw_part + (size_t)blockIdx.x * DIM))[c] = sum;
    }
}

// out[c] = sum over the N / NORM_ROWS partial rows of rows[n][c]. Grid DIM/32, block (32, 8); fixed order.
__global__ void colsum(float* out, const float* rows) {
    const int c = blockIdx.x * 32 + threadIdx.x, y = threadIdx.y;
    float acc = 0.f;
    for (int n = y; n < N / NORM_ROWS; n += 8) acc += rows[(size_t)n * DIM + c];
    __shared__ float part[8][32];
    part[y][threadIdx.x] = acc;
    __syncthreads();
    if (y == 0) {
        for (int k = 1; k < 8; k++) acc += part[k][threadIdx.x];
        out[c] = acc;
    }
}

// ---- Tensor cores: mma.sync m16n8k16, bf16 inputs, fp32 accumulate ----
// Fragments (PTX ISA), lane = 4g + t. A 16x16: {row g, col 2t..}, {g+8, 2t}, {g, 8+2t}, {g+8, 8+2t}.
// B 16x8 (k x n): {k 2t.., n g}, {k 8+2t.., n g}. C 16x8: {row g, col 2t, 2t+1}, {row g+8, col 2t, 2t+1}.
static_assert(HEAD_DIM == 64 && ATT_TILE == 64, "attention kernels are written for 64x64 tiles");

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const bf16* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const bf16* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    bf162 h = __floats2bfloat162_rn(lo, hi);
    return *(uint32_t*)&h;
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int K>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(K));
}

// 64x64 bf16 tile in shared memory: row r, 16-byte chunk c lives at r*64 + 8*(c ^ (r & 7)), so the 8 rows
// an ldmatrix reads hit distinct banks.
constexpr int TILE = 64 * 64;
__device__ __forceinline__ int swz(int r, int c) { return r * 64 + ((c ^ (r & 7)) << 3); }

// Async copy of a 64x64 tile whose rows are `stride` elements apart (128 threads).
__device__ __forceinline__ void load_tile(bf16* dst, const bf16* src, size_t stride) {
    for (int i = threadIdx.x; i < 512; i += 128) cp16(dst + swz(i >> 3, i & 7), src + (i >> 3) * stride + (i & 7) * 8);
}

// A fragments for 16 tile rows from r0, all 64 columns (k = columns).
__device__ __forceinline__ void frag_a(uint32_t (&a)[4][4], const bf16* tile, int r0) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) ldsm4(a[kk], tile + swz(r0 + (lane & 15), kk * 2 + (lane >> 4)));
}

// acc[16 x 64] += A[16 x 64] * tile^T: tile rows become output columns, k = tile columns.
__device__ __forceinline__ void mma_abt(float (&acc)[8][4], const uint32_t (&a)[4][4], const bf16* tile) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
#pragma unroll
        for (int np = 0; np < 4; np++) {
            uint32_t r[4];
            ldsm4(r, tile + swz(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)));
            mma(acc[2 * np], a[kk], r[0], r[1]);
            mma(acc[2 * np + 1], a[kk], r[2], r[3]);
        }
}

// acc[16 x 64] += A[16 x 64] * tile: k = tile rows, output columns = tile columns.
__device__ __forceinline__ void mma_ab(float (&acc)[8][4], const uint32_t (&a)[4][4], const bf16* tile) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
#pragma unroll
        for (int np = 0; np < 4; np++) {
            uint32_t r[4];
            ldsm4t(r, tile + swz(kk * 16 + (lane & 15), np * 2 + (lane >> 4)));
            mma(acc[2 * np], a[kk], r[0], r[1]);
            mma(acc[2 * np + 1], a[kk], r[2], r[3]);
        }
}

// C block [16 x 64] fp32 -> bf16 A fragments with k = its 64 columns.
__device__ __forceinline__ void c_to_a(uint32_t (&a)[4][4], const float (&c)[8][4]) {
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        a[kk][0] = pack2(c[2 * kk][0], c[2 * kk][1]);
        a[kk][1] = pack2(c[2 * kk][2], c[2 * kk][3]);
        a[kk][2] = pack2(c[2 * kk + 1][0], c[2 * kk + 1][1]);
        a[kk][3] = pack2(c[2 * kk + 1][2], c[2 * kk + 1][3]);
    }
}

// ---- Layer GEMMs with fused epilogues (cuBLASLt cannot fuse these). A block computes BM rows x 64 columns with
// BM / WM warps of WM x 64, so each thread holds columns c and c + 32 of its 64 (fragments j and j + 4), e.g. a
// RoPE pair. BK = 32, cp.async ring. Used only where the fusion beats cuBLASLt plus a separate kernel.
constexpr int GBK = 32;
__device__ __forceinline__ int sw32(int r, int c) { return r * 32 + ((c ^ ((r >> 1) & 3)) << 3); }  // 64-byte rows

// acc += A[bm.., :K] * B^T where B tile row r is row brow(r) of B [.][K] (TN, both row-major with K columns).
template <int BM, int WM, int K, int STAGES, class BRow>
__device__ __forceinline__ void gemm_tn_loop(float (&acc)[WM / 16][8][4], const bf16* A, int bm, const bf16* B, BRow brow,
                                             bf16* sm) {
    constexpr int TH = BM / WM * 32, KT = K / GBK;
    static_assert(K % GBK == 0 && KT >= STAGES - 1, "pipeline depth");
    bf16 *as = sm, *bs = sm + STAGES * BM * GBK;
    const int lane = threadIdx.x & 31, wm0 = (threadIdx.x >> 5) * WM;
    auto load = [&](int kt, int st) {
        for (int i = threadIdx.x; i < BM * 4; i += TH)
            cp16(as + st * BM * GBK + sw32(i >> 2, i & 3), A + (size_t)(bm + (i >> 2)) * K + kt * GBK + (i & 3) * 8);
        for (int i = threadIdx.x; i < 64 * 4; i += TH)
            cp16(bs + st * 64 * GBK + sw32(i >> 2, i & 3), B + (size_t)brow(i >> 2) * K + kt * GBK + (i & 3) * 8);
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        load(st, st);
        cp_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        cp_commit();
        const bf16 *a = as + (kt % STAGES) * BM * GBK, *b = bs + (kt % STAGES) * 64 * GBK;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            uint32_t af[WM / 16][4], bfr[4][4];
#pragma unroll
            for (int i = 0; i < WM / 16; i++) ldsm4(af[i], a + sw32(wm0 + 16 * i + (lane & 15), 2 * kk + (lane >> 4)));
#pragma unroll
            for (int p = 0; p < 4; p++)
                ldsm4(bfr[p], b + sw32(16 * p + (lane & 7) + ((lane >> 4) << 3), 2 * kk + ((lane >> 3) & 1)));
#pragma unroll
            for (int i = 0; i < WM / 16; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) mma(acc[i][j], af[i], bfr[j / 2][(j & 1) * 2], bfr[j / 2][(j & 1) * 2 + 1]);
        }
    }
}

// acc += A[bm.., :K] * B[:K, bn..bn+63], B row-major [K][LDB] (NN); B points at column bn of row 0.
template <int BM, int WM, int K, int LDB, int STAGES>
__device__ __forceinline__ void gemm_nn_loop(float (&acc)[WM / 16][8][4], const bf16* A, int bm, const bf16* B, bf16* sm) {
    constexpr int TH = BM / WM * 32, KT = K / GBK;
    static_assert(K % GBK == 0 && KT >= STAGES - 1, "pipeline depth");
    bf16 *as = sm, *bs = sm + STAGES * BM * GBK;
    const int lane = threadIdx.x & 31, wm0 = (threadIdx.x >> 5) * WM;
    auto load = [&](int kt, int st) {
        for (int i = threadIdx.x; i < BM * 4; i += TH)
            cp16(as + st * BM * GBK + sw32(i >> 2, i & 3), A + (size_t)(bm + (i >> 2)) * K + kt * GBK + (i & 3) * 8);
        for (int i = threadIdx.x; i < GBK * 8; i += TH)
            cp16(bs + st * GBK * 64 + swz(i >> 3, i & 7), B + (size_t)(kt * GBK + (i >> 3)) * LDB + (i & 7) * 8);
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        load(st, st);
        cp_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        cp_commit();
        const bf16 *a = as + (kt % STAGES) * BM * GBK, *b = bs + (kt % STAGES) * GBK * 64;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            uint32_t af[WM / 16][4];
#pragma unroll
            for (int i = 0; i < WM / 16; i++) ldsm4(af[i], a + sw32(wm0 + 16 * i + (lane & 15), 2 * kk + (lane >> 4)));
#pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t r[4];
                ldsm4t(r, b + swz(16 * kk + (lane & 15), 2 * p + (lane >> 4)));
#pragma unroll
                for (int i = 0; i < WM / 16; i++) {
                    mma(acc[i][2 * p], af[i], r[0], r[1]);
                    mma(acc[i][2 * p + 1], af[i], r[2], r[3]);
                }
            }
        }
    }
}

constexpr int GBM = 64, GWM = 16, GST = 3;  // fused-GEMM tiles, swept on sm89 (BENCH.md profiles)
static_assert(N % GBM == 0 && QKV_DIM % 64 == 0 && FFN % 64 == 0, "whole GEMM tiles");

// qkv = ln1 . Wqkv^T, then RoPE on the q and k heads, rounding to bf16 before and after the rotation (PyTorch's
// projection then rope). Block = GBM tokens x one 64-column head unit (QKV_DIM / 64 of them).
__global__ void __launch_bounds__(GBM / GWM * 32) qkv_rope(bf16* qkv, const bf16* x, const bf16* w, const float2* cs) {
    __shared__ __align__(128) bf16 sm[GST * (GBM + 64) * GBK];
    const int bm = blockIdx.y * GBM, u = blockIdx.x;
    float acc[GWM / 16][8][4] = {};
    gemm_tn_loop<GBM, GWM, DIM, GST>(acc, x, bm, w, [&](int r) { return u * 64 + r; }, sm);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, wm0 = (threadIdx.x >> 5) * GWM;
    const bool rot = u < HEADS + KV_HEADS;
#pragma unroll
    for (int i = 0; i < GWM / 16; i++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int n = bm + wm0 + 16 * i + g + 8 * r;
            bf16* y = qkv + (size_t)n * QKV_DIM + u * 64 + 2 * t;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float a[2], b[2];
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const float x1 = round_bf16(acc[i][j][2 * r + e]), x2 = round_bf16(acc[i][j + 4][2 * r + e]);
                    if (rot) {
                        const float2 c = cs[n % T * 32 + 8 * j + 2 * t + e];
                        a[e] = x1 * c.x - x2 * c.y;
                        b[e] = x2 * c.x + x1 * c.y;
                    } else {
                        a[e] = x1;
                        b[e] = x2;
                    }
                }
                *(bf162*)(y + 8 * j) = __floats2bfloat162_rn(a[0], a[1]);
                *(bf162*)(y + 8 * j + 32) = __floats2bfloat162_rn(b[0], b[1]);
            }
        }
}

// dact = bf16(dmo . Wd) (PyTorch's matmul output), then PyTorch's mul and silu backward:
// ds = bf16(dact * u), du = bf16(dact * s), dg = bf16(ds * silu'(g)), s = bf16(silu(g)), into dgu.
// Block = GBM tokens x 64 FFN columns.
__global__ void __launch_bounds__(GBM / GWM * 32) down_dx_swiglu(bf16* dgu, const bf16* dmo, const bf16* wd, const bf16* gu) {
    __shared__ __align__(128) bf16 sm[GST * (GBM + 64) * GBK];
    const int bm = blockIdx.y * GBM, bn = blockIdx.x * 64;
    float acc[GWM / 16][8][4] = {};
    gemm_nn_loop<GBM, GWM, DIM, FFN, GST>(acc, dmo, bm, wd + bn, sm);
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, wm0 = (threadIdx.x >> 5) * GWM;
#pragma unroll
    for (int i = 0; i < GWM / 16; i++)
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const size_t n = bm + wm0 + 16 * i + g + 8 * r;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int c = bn + 8 * j + 2 * t;
                const float2 gf = __bfloat1622float2(*(const bf162*)(gu + n * 2 * FFN + c));
                const float2 uf = __bfloat1622float2(*(const bf162*)(gu + n * 2 * FFN + FFN + c));
                float dg[2], du[2];
                const float gv[2] = {gf.x, gf.y}, uv[2] = {uf.x, uf.y};
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const float da = round_bf16(acc[i][j][2 * r + e]), sig = 1.f / (1.f + expf(-gv[e]));
                    du[e] = da * round_bf16(gv[e] / (1.f + expf(-gv[e])));
                    dg[e] = round_bf16(da * uv[e]) * sig * (1.f + gv[e] * (1.f - sig));
                }
                *(bf162*)(dgu + n * 2 * FFN + c) = __floats2bfloat162_rn(dg[0], dg[1]);
                *(bf162*)(dgu + n * 2 * FFN + FFN + c) = __floats2bfloat162_rn(du[0], du[1]);
            }
        }
}

// ---- Attention (FlashAttention-2 style). Scores are scaled into log2 units so softmax uses exp2; lse is kept
// in those units. P (and dS in backward) are rounded to bf16 for the tensor-core products, as flash attention does.
// Each thread owns rows g and g+8 of its warp's 16 rows and columns 8j + 2t + {0,1} of the 64-column block.
constexpr int NT = T / 64;

// Causal GQA forward. Block = (64 query rows, head, batch), 4 warps x 16 rows; K/V tiles double-buffered.
// qkv [N][QKV_DIM] post-RoPE, out [N][DIM], lse [B][HEADS][T].
__global__ void __launch_bounds__(128) attention_fwd(bf16* out, float* lse, const bf16* qkv) {
    const int qt = NT - 1 - blockIdx.x, h = blockIdx.y, b = blockIdx.z;  // longest blocks first
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    const bf16* kg = seq + (HEADS + h / (HEADS / KV_HEADS)) * HEAD_DIM;
    const bf16* vg = kg + KV_HEADS * HEAD_DIM;
    __shared__ __align__(128) bf16 ks[2][TILE], vs[2][TILE];

    load_tile(ks[1], seq + (size_t)qt * 64 * QKV_DIM + h * HEAD_DIM, QKV_DIM);  // Q, staged in buffer 1
    load_tile(ks[0], kg, QKV_DIM);
    load_tile(vs[0], vg, QKV_DIM);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t qf[4][4];
    frag_a(qf, ks[1], warp * 16);
    __syncthreads();

    const float scale = LOG2E / sqrtf((float)HEAD_DIM);
    const int q0 = qt * 64 + warp * 16 + g;
    float o[8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    for (int kt = 0; kt <= qt; kt++) {
        if (kt < qt) {
            load_tile(ks[(kt + 1) & 1], kg + (size_t)(kt + 1) * 64 * QKV_DIM, QKV_DIM);
            load_tile(vs[(kt + 1) & 1], vg + (size_t)(kt + 1) * 64 * QKV_DIM, QKV_DIM);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        float s[8][4] = {};
        mma_abt(s, qf, ks[kt & 1]);
        float mx[2] = {m[0], m[1]};
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[j][e] * scale;
                if (kt == qt && kt * 64 + 8 * j + 2 * t + (e & 1) > q0 + 8 * (e >> 1)) v = -INFINITY;
                s[j][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            corr[r] = exp2f(m[r] - mx[r]);
            m[r] = mx[r];
            l[r] *= corr[r];
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                o[j][e] *= corr[e >> 1];
                s[j][e] = exp2f(s[j][e] - m[e >> 1]);
                l[e >> 1] += s[j][e];
            }
        uint32_t pf[4][4];
        c_to_a(pf, s);
        mma_ab(o, pf, vs[kt & 1]);
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
        const int q = q0 + 8 * r;
        bf16* orow = out + ((size_t)b * T + q) * DIM + h * HEAD_DIM + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++)
            *(bf162*)(orow + 8 * j) = __floats2bfloat162_rn(o[j][2 * r] / l[r], o[j][2 * r + 1] / l[r]);
        if (t == 0) lse[((size_t)b * HEADS + h) * T + q] = m[r] + log2f(l[r]);
    }
}

// dq (before RoPE backward) for 64 query rows of one head, written into dqkv's q columns, and
// Dsum = rowsum(dO * O), which attention_bwd_dkv reads.
__global__ void __launch_bounds__(128) attention_bwd_dq(bf16* dqkv, float* Dsum, const bf16* qkv, const bf16* out,
                                                        const bf16* dout, const float* lse) {
    const int qt = NT - 1 - blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    const bf16* kg = seq + (HEADS + h / (HEADS / KV_HEADS)) * HEAD_DIM;
    const bf16* vg = kg + KV_HEADS * HEAD_DIM;
    const size_t row0 = (size_t)b * T + qt * 64, lrow = ((size_t)b * HEADS + h) * T + qt * 64;
    __shared__ __align__(128) bf16 ks[2][TILE], vs[2][TILE];
    __shared__ float dsm[64];

    load_tile(ks[1], seq + (size_t)qt * 64 * QKV_DIM + h * HEAD_DIM, QKV_DIM);  // Q and dO staged in buffer 1
    load_tile(vs[1], dout + row0 * DIM + h * HEAD_DIM, DIM);
    load_tile(ks[0], kg, QKV_DIM);
    load_tile(vs[0], vg, QKV_DIM);
    cp_commit();
    {  // D, two threads per row
        const int r = threadIdx.x >> 1, c0 = (threadIdx.x & 1) * 32;
        const bf16 *dp = dout + (row0 + r) * DIM + h * HEAD_DIM + c0, *op = out + (row0 + r) * DIM + h * HEAD_DIM + c0;
        float acc = 0.f;
#pragma unroll
        for (int c = 0; c < 32; c += 8) {
            float a8[8], b8[8];
            load8(a8, dp + c);
            load8(b8, op + c);
#pragma unroll
            for (int e = 0; e < 8; e++) acc += a8[e] * b8[e];
        }
        acc += __shfl_xor_sync(0xffffffff, acc, 1);
        if (!(threadIdx.x & 1)) {
            dsm[r] = acc;
            Dsum[lrow + r] = acc;
        }
    }
    cp_wait<0>();
    __syncthreads();
    uint32_t qf[4][4], df[4][4];
    frag_a(qf, ks[1], warp * 16);
    frag_a(df, vs[1], warp * 16);
    const int r0 = warp * 16 + g, q0 = qt * 64 + r0;
    const float dd[2] = {dsm[r0], dsm[r0 + 8]}, l2[2] = {lse[lrow + r0], lse[lrow + r0 + 8]};
    __syncthreads();

    const float scale = LOG2E / sqrtf((float)HEAD_DIM);
    float dq[8][4] = {};
    for (int kt = 0; kt <= qt; kt++) {
        if (kt < qt) {
            load_tile(ks[(kt + 1) & 1], kg + (size_t)(kt + 1) * 64 * QKV_DIM, QKV_DIM);
            load_tile(vs[(kt + 1) & 1], vg + (size_t)(kt + 1) * 64 * QKV_DIM, QKV_DIM);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        float s[8][4] = {}, dp[8][4] = {};
        mma_abt(s, qf, ks[kt & 1]);
        mma_abt(dp, df, vs[kt & 1]);
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int r = e >> 1;
                const bool masked = kt == qt && kt * 64 + 8 * j + 2 * t + (e & 1) > q0 + 8 * r;
                const float p = masked ? 0.f : exp2f(s[j][e] * scale - l2[r]);
                s[j][e] = p * (dp[j][e] - dd[r]);
            }
        uint32_t sf[4][4];
        c_to_a(sf, s);
        mma_ab(dq, sf, ks[kt & 1]);
        __syncthreads();
    }
    const float k = scale / LOG2E;  // dS is w.r.t. scores scaled by 1/sqrt(HEAD_DIM)
#pragma unroll
    for (int r = 0; r < 2; r++) {
        bf16* row = dqkv + (row0 + r0 + 8 * r) * QKV_DIM + h * HEAD_DIM + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++) *(bf162*)(row + 8 * j) = __floats2bfloat162_rn(dq[j][2 * r] * k, dq[j][2 * r + 1] * k);
    }
}

// dk, dv contributions of one query head for 64 key rows (4 warps x 16 keys), over all query tiles that see
// them. fp32 partials [N][HEADS][dk 64 | dv 64]; attention_bwd_finish sums the heads of each kv group.
__global__ void __launch_bounds__(128) attention_bwd_dkv(float* part, const bf16* qkv, const bf16* dout,
                                                         const float* lse, const float* Dsum) {
    const int kt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;  // kt = 0 has the most work
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    const bf16* kg = seq + (HEADS + h / (HEADS / KV_HEADS)) * HEAD_DIM;
    const size_t lrow = ((size_t)b * HEADS + h) * T;
    __shared__ __align__(128) bf16 qs[2][TILE], dos[2][TILE];
    __shared__ __align__(16) float ls[2][64], dds[2][64];

    auto load_q = [&](int qt, int buf) {  // Q, dO, lse and D of query tile qt
        load_tile(qs[buf], seq + (size_t)qt * 64 * QKV_DIM + h * HEAD_DIM, QKV_DIM);
        load_tile(dos[buf], dout + ((size_t)b * T + qt * 64) * DIM + h * HEAD_DIM, DIM);
        if (threadIdx.x < 16) cp16(&ls[buf][threadIdx.x * 4], lse + lrow + qt * 64 + threadIdx.x * 4);
        else if (threadIdx.x < 32) cp16(&dds[buf][(threadIdx.x - 16) * 4], Dsum + lrow + qt * 64 + (threadIdx.x - 16) * 4);
    };
    load_tile(qs[1], kg + (size_t)kt * 64 * QKV_DIM, QKV_DIM);  // K and V staged in buffer 1
    load_tile(dos[1], kg + KV_HEADS * HEAD_DIM + (size_t)kt * 64 * QKV_DIM, QKV_DIM);
    load_q(kt, 0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t kf[4][4], vf[4][4];
    frag_a(kf, qs[1], warp * 16);
    frag_a(vf, dos[1], warp * 16);
    __syncthreads();

    const float scale = LOG2E / sqrtf((float)HEAD_DIM);
    const int k0 = kt * 64 + warp * 16 + g;
    float dk[8][4] = {}, dv[8][4] = {};
    for (int qt = kt; qt < NT; qt++) {
        const int buf = (qt - kt) & 1;
        if (qt + 1 < NT) load_q(qt + 1, buf ^ 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        float st[8][4] = {}, dpt[8][4] = {};
        mma_abt(st, kf, qs[buf]);   // S^T = K Q^T  [16 keys x 64 queries]
        mma_abt(dpt, vf, dos[buf]);  // dP^T = V dO^T
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int c = 8 * j + 2 * t + (e & 1);
                const bool masked = qt == kt && qt * 64 + c < k0 + 8 * (e >> 1);
                const float p = masked ? 0.f : exp2f(st[j][e] * scale - ls[buf][c]);
                st[j][e] = p;
                dpt[j][e] = p * (dpt[j][e] - dds[buf][c]);
            }
        uint32_t pf[4][4], sf[4][4];
        c_to_a(pf, st);
        c_to_a(sf, dpt);
        mma_ab(dv, pf, dos[buf]);  // dV += P^T dO
        mma_ab(dk, sf, qs[buf]);   // dK += dS^T Q
        __syncthreads();
    }
    const float k = scale / LOG2E;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        float* row = part + (((size_t)b * T + k0 + 8 * r) * HEADS + h) * 2 * HEAD_DIM + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            *(float2*)(row + 8 * j) = {dk[j][2 * r] * k, dk[j][2 * r + 1] * k};
            *(float2*)(row + HEAD_DIM + 8 * j) = {dv[j][2 * r], dv[j][2 * r + 1]};
        }
    }
}

// dqkv <- [RoPE^-1(dq) | RoPE^-1(sum over the group's heads of dk) | sum of dv], dq read in place from dqkv.
__global__ void attention_bwd_finish(bf16* dqkv, const float* part, const float2* cs) {
    constexpr int HALF = HEAD_DIM / 2, U = HEADS + 2 * KV_HEADS, G = HEADS / KV_HEADS;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * U * HALF) return;
    const int d = i % HALF, u = i / HALF % U, n = i / (HALF * U);
    bf16* y = dqkv + (size_t)n * QKV_DIM + u * HEAD_DIM;
    float x1 = 0.f, x2 = 0.f;
    if (u < HEADS) {
        x1 = __bfloat162float(y[d]);
        x2 = __bfloat162float(y[d + HALF]);
    } else {
        const int kv = (u - HEADS) % KV_HEADS, off = u < HEADS + KV_HEADS ? 0 : HEAD_DIM;
        for (int hh = kv * G; hh < (kv + 1) * G; hh++) {
            const float* p = part + ((size_t)n * HEADS + hh) * 2 * HEAD_DIM + off;
            x1 += p[d];
            x2 += p[d + HALF];
        }
    }
    if (u < HEADS + KV_HEADS) {  // inverse rotation
        const float2 c = cs[n % T * HALF + d];
        y[d] = __float2bfloat16(x1 * c.x + x2 * c.y);
        y[d + HALF] = __float2bfloat16(x2 * c.x - x1 * c.y);
    } else {
        y[d] = __float2bfloat16(x1);
        y[d + HALF] = __float2bfloat16(x2);
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

// AdamW, same math as PyTorch's fused kernel (fp32). bc1 = 1 - beta1^step, bc2_sqrt = sqrt(1 - beta2^step).
__device__ __forceinline__ void adamw_update(float& p, float& m, float& v, float g, float bc1, float bc2_sqrt) {
    p -= LR * WEIGHT_DECAY * p;
    m = fmaf(BETA1, m, fmaf(-BETA1, g, g));
    v = fmaf(BETA2, v, fmaf(-BETA2, g * g, g * g));
    p -= (LR / bc1) * m / (sqrtf(v) / bc2_sqrt + ADAM_EPS);
}

__global__ void next_step(float* step) { *step += 1.f; }

// Embedding rows of the batch's tokens: grad = (bf16-rounded head grad, already in gemb) + sum of dres rows for that
// token, then AdamW on the row. seg = [perm N][starts N+1][count 1] from the host: positions sorted by token, one
// block per distinct token, rows summed in a fixed order (no atomics).
__global__ void __launch_bounds__(DIM / 4) embed_bwd_adamw(float* p, bf16* pb, float* m, float* v, float* gemb,
                                                           const float* dres, const int* tok, const int* seg,
                                                           const float* step_count) {
    const int* perm = seg;
    const int* starts = seg + N;
    if ((int)blockIdx.x >= starts[N + 1]) return;
    const int s0 = starts[blockIdx.x], s1 = starts[blockIdx.x + 1];
    const size_t o = (size_t)tok[perm[s0]] * DIM + threadIdx.x * 4;
    float4 G = *(float4*)(gemb + o);
    for (int k = s0; k < s1; k++) {
        float4 d = *(const float4*)(dres + (size_t)perm[k] * DIM + threadIdx.x * 4);
        G.x += d.x;
        G.y += d.y;
        G.z += d.z;
        G.w += d.w;
    }
    *(float4*)(gemb + o) = G;
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    float4 P = *(float4*)(p + o), M = *(float4*)(m + o), V = *(float4*)(v + o);
    adamw_update(P.x, M.x, V.x, G.x, bc1, bc2_sqrt);
    adamw_update(P.y, M.y, V.y, G.y, bc1, bc2_sqrt);
    adamw_update(P.z, M.z, V.z, G.z, bc1, bc2_sqrt);
    adamw_update(P.w, M.w, V.w, G.w, bc1, bc2_sqrt);
    *(float4*)(p + o) = P;
    *(float4*)(m + o) = M;
    *(float4*)(v + o) = V;
    *(uint2*)(pb + o) = {pack2(P.x, P.y), pack2(P.z, P.w)};
}

// AdamW for the norm weights (gvec past the embedding), 4 per thread; also refreshes the bf16 copy. Weight matrices
// and the embedding are updated inside their dW GEMMs (dw_adamw) and embed_bwd_adamw.
__global__ void adamw_norms(float* p, bf16* pb, const float* gvec, float* m, float* v, const float* step_count) {
    const size_t j = P_EMBED + 4 * (blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (j >= G_VEC) return;
    const size_t o = j < G_VEC - DIM ? P_EMBED + (j - P_EMBED) / (2 * DIM) * P_LAYER + P_MAT + (j - P_EMBED) % (2 * DIM)
                                     : P_TOTAL - DIM + (j - (G_VEC - DIM));  // parameter offset of gvec[j]
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    float4 P = *(float4*)(p + o), G = *(const float4*)(gvec + j), M = *(float4*)(m + o), V = *(float4*)(v + o);
    adamw_update(P.x, M.x, V.x, G.x, bc1, bc2_sqrt);
    adamw_update(P.y, M.y, V.y, G.y, bc1, bc2_sqrt);
    adamw_update(P.z, M.z, V.z, G.z, bc1, bc2_sqrt);
    adamw_update(P.w, M.w, V.w, G.w, bc1, bc2_sqrt);
    *(float4*)(p + o) = P;
    *(float4*)(m + o) = M;
    *(float4*)(v + o) = V;
    *(uint2*)(pb + o) = {pack2(P.x, P.y), pack2(P.z, P.w)};
}

// dW = dY^T X for one weight matrix W [OUT][IN]: reduction over the N tokens with both operands token-major, so both
// come through ldmatrix.trans. The AdamW update is fused into the epilogue: the grad, rounded to bf16 like PyTorch's
// dW, updates master, moments and the bf16 copy of W in place while other blocks are still multiplying.
// gout (optional, tests) also receives the bf16 grad. Block = 64 x 64 of W, 4 warps x 16 rows; 4-stage cp.async.
// Tied embedding (lm_head dW): rows whose token is in the batch (bit set in `touched`) also get the lookup grad
// later, so their bf16-rounded head grad goes to gemb (fp32) and embed_bwd_adamw finishes them; gemb receives every
// row's grad when gout is set (tests).
constexpr int DW_ST = 4;
template <int OUT, int IN>
__global__ void __launch_bounds__(128) dw_adamw(float* p, bf16* pb, float* m, float* v, bf16* gout, const bf16* dY,
                                                const bf16* X, const float* step_count, float* gemb = nullptr,
                                                const uint32_t* touched = nullptr) {
    static_assert(OUT % 64 == 0 && IN % 64 == 0 && N % GBK == 0, "whole dW tiles");
    constexpr int KT = N / GBK;
    __shared__ __align__(128) bf16 sm[DW_ST * 2 * GBK * 64];
    bf16 *as = sm, *bs = sm + DW_ST * GBK * 64;  // [GBK tokens][64] tiles of dY and X, 128-byte swizzled rows
    const int bm = blockIdx.y * 64, bn = blockIdx.x * 64;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, wm0 = (threadIdx.x >> 5) * 16;
    auto load = [&](int kt, int st) {
        for (int i = threadIdx.x; i < GBK * 8; i += 128) {
            const size_t row = kt * GBK + (i >> 3);
            cp16(as + st * GBK * 64 + swz(i >> 3, i & 7), dY + row * OUT + bm + (i & 7) * 8);
            cp16(bs + st * GBK * 64 + swz(i >> 3, i & 7), X + row * IN + bn + (i & 7) * 8);
        }
    };
    float acc[8][4] = {};
#pragma unroll
    for (int st = 0; st < DW_ST - 1; st++) {
        load(st, st);
        cp_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<DW_ST - 2>();
        __syncthreads();
        if (kt + DW_ST - 1 < KT) load(kt + DW_ST - 1, (kt + DW_ST - 1) % DW_ST);
        cp_commit();
        const bf16 *a = as + (kt % DW_ST) * GBK * 64, *b = bs + (kt % DW_ST) * GBK * 64;
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            uint32_t af[4];
            ldsm4t(af, a + swz(16 * kk + (lane & 7) + ((lane >> 4) << 3), wm0 / 8 + ((lane >> 3) & 1)));
#pragma unroll
            for (int q = 0; q < 4; q++) {
                uint32_t r[4];
                ldsm4t(r, b + swz(16 * kk + (lane & 15), 2 * q + (lane >> 4)));
                mma(acc[2 * q], af, r[0], r[1]);
                mma(acc[2 * q + 1], af, r[2], r[3]);
            }
        }
    }
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int row = bm + wm0 + g + 8 * r;
            const size_t o = (size_t)row * IN + bn + 8 * j + 2 * t;
            const bf162 gb = __floats2bfloat162_rn(acc[j][2 * r], acc[j][2 * r + 1]);
            const float2 gf = __bfloat1622float2(gb);
            if (gemb) {
                if (gout || touched[row >> 5] >> (row & 31) & 1) *(float2*)(gemb + o) = gf;
                if (touched[row >> 5] >> (row & 31) & 1) continue;
            } else if (gout) {
                *(bf162*)(gout + o) = gb;
            }
            float2 P = *(float2*)(p + o), M = *(float2*)(m + o), V = *(float2*)(v + o);
            adamw_update(P.x, M.x, V.x, gf.x, bc1, bc2_sqrt);
            adamw_update(P.y, M.y, V.y, gf.y, bc1, bc2_sqrt);
            *(float2*)(p + o) = P;
            *(float2*)(m + o) = M;
            *(float2*)(v + o) = V;
            *(bf162*)(pb + o) = __floats2bfloat162_rn(P.x, P.y);
        }
}
