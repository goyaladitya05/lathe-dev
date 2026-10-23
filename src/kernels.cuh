// All lathe kernels. Shapes from the model header, tiles from the target header; B/T/N, AdamW
// hyperparameters and the parameter layout from train.cu. Precision follows PyTorch bf16 autocast (BENCH.md).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

static_assert(DIM % 64 == 0 && FFN % 8 == 0 && HEAD_DIM % 16 == 0, "vector widths");
static_assert(T % 64 == 0 && VOCAB % (CE_THREADS * 8) == 0, "no tail handling");

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

// Deterministic pseudo-random bf16 in [-1, 1) (GEMM tuning inputs).
__global__ void fill_bf16(bf16* p, size_t n, uint32_t seed) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = (uint32_t)i * 2654435761u ^ seed * 0x9e3779b9u;
    x ^= x >> 13, x *= 0x5bd1e995u, x ^= x >> 15;
    p[i] = __float2bfloat16((x & 0xffff) / 32768.f - 1.f);
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
static_assert(HEAD_DIM == 64, "attention kernels are written for 64x64 tiles");

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

// acc += A[bm.., :K] * B[:64, :K]^T (TN, both row-major with K columns); B points at the block's first row of B.
template <int BM, int WM, int K, int STAGES>
__device__ __forceinline__ void gemm_tn_loop(float (&acc)[WM / 16][8][4], const bf16* A, int bm, const bf16* B, bf16* sm) {
    constexpr int TH = BM / WM * 32, KT = K / GBK;
    static_assert(K % GBK == 0 && KT >= STAGES - 1, "pipeline depth");
    bf16 *as = sm, *bs = sm + STAGES * BM * GBK;
    const int lane = threadIdx.x & 31, wm0 = (threadIdx.x >> 5) * WM;
    auto load = [&](int kt, int st) {
        for (int i = threadIdx.x; i < BM * 4; i += TH)
            cp16(as + st * BM * GBK + sw32(i >> 2, i & 3), A + (size_t)(bm + (i >> 2)) * K + kt * GBK + (i & 3) * 8);
        for (int i = threadIdx.x; i < 64 * 4; i += TH)
            cp16(bs + st * 64 * GBK + sw32(i >> 2, i & 3), B + (size_t)(i >> 2) * K + kt * GBK + (i & 3) * 8);
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

// GBM, GWM, GST: the fused GEMMs' block rows, warp rows and pipeline stages, per GPU in targets/*.h. The k order is the
// same for every tile shape, so the results are too.
static_assert(GBM % GWM == 0 && GWM % 16 == 0 && GST * (GBM + 64) * GBK * 2 <= 48 * 1024, "fused-GEMM tiles");
static_assert(N % GBM == 0 && QKV_DIM % 64 == 0 && FFN % 64 == 0, "whole GEMM tiles");

// qkv = ln1 . Wqkv^T, then RoPE on the q and k heads, rounding to bf16 before and after the rotation (PyTorch's
// projection then rope). Block = GBM tokens x one 64-column head unit (QKV_DIM / 64 of them).
__global__ void __launch_bounds__(GBM / GWM * 32) qkv_rope(bf16* qkv, const bf16* x, const bf16* w, const float2* cs) {
    __shared__ __align__(128) bf16 sm[GST * (GBM + 64) * GBK];
    const int bm = blockIdx.y * GBM, u = blockIdx.x;
    float acc[GWM / 16][8][4] = {};
    gemm_tn_loop<GBM, GWM, DIM, GST>(acc, x, bm, w + (size_t)u * 64 * DIM, sm);
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
// Block = GBM tokens x 64 FFN columns. The block's gate and up lines, the epilogue's only DRAM reads, are prefetched
// into L2 at the start so they arrive while the GEMM runs (chisel e052).
__global__ void __launch_bounds__(GBM / GWM * 32) down_dx_swiglu(bf16* dgu, const bf16* dmo, const bf16* wd, const bf16* gu) {
    __shared__ __align__(128) bf16 sm[GST * (GBM + 64) * GBK];
    const int bm = blockIdx.y * GBM, bn = blockIdx.x * 64;
    for (int i = threadIdx.x; i < 2 * GBM; i += GBM / GWM * 32)  // 128-byte lines: GBM rows x (64 gate + 64 up columns)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(gu + (size_t)(bm + (i >> 1)) * 2 * FFN + (i & 1) * FFN + bn));
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
// Attention grids: (NT x heads, 1, B) with the head index fastest when ATT_HEAD_FASTEST (targets/*.h), so every head's
// longest blocks dispatch first, else (NT, heads, B). att_tile: the block's tile in dispatch order; att_head: its head.
template <int H> __device__ __forceinline__ int att_tile() { return ATT_HEAD_FASTEST ? blockIdx.x / H : blockIdx.x; }
template <int H> __device__ __forceinline__ int att_head() { return ATT_HEAD_FASTEST ? blockIdx.x % H : blockIdx.y; }
inline dim3 att_grid(int heads) { return ATT_HEAD_FASTEST ? dim3(NT * heads, 1, B) : dim3(NT, heads, B); }

// Causal GQA forward. Block = (64 query rows, head, batch), 4 warps x 16 rows; K/V tiles double-buffered.
// qkv [N][QKV_DIM] post-RoPE, out [N][DIM], lse [B][HEADS][T]. With ATT_SPLIT_FWD > 1 (targets/*.h) the key tiles are
// cut into that many ranges, one block each (blockIdx.z = b * ATT_SPLIT_FWD + range), and each block writes its
// unnormalized output, row max and row sum to opart / mlpart for attention_fwd_combine instead of out and lse.
constexpr int FWD_NT = (NT + ATT_SPLIT_FWD - 1) / ATT_SPLIT_FWD;  // key tiles per range
__global__ void __launch_bounds__(128) attention_fwd(bf16* out, float* lse, const bf16* qkv, float* opart,
                                                     float2* mlpart) {
    const int qt = NT - 1 - att_tile<HEADS>(), h = att_head<HEADS>();  // longest blocks first
    const int b = blockIdx.z / ATT_SPLIT_FWD, sp = blockIdx.z % ATT_SPLIT_FWD;
    const int k_lo = sp * FWD_NT, k_hi = min(qt + 1, (sp + 1) * FWD_NT);
    if (k_lo >= k_hi) return;  // a range wholly after the query tile: attention_fwd_combine skips it
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    const bf16* kg = seq + (HEADS + h / (HEADS / KV_HEADS)) * HEAD_DIM;
    const bf16* vg = kg + KV_HEADS * HEAD_DIM;
    __shared__ __align__(128) bf16 ks[2][TILE], vs[2][TILE];

    load_tile(ks[1], seq + (size_t)qt * 64 * QKV_DIM + h * HEAD_DIM, QKV_DIM);  // Q, staged in buffer 1
    load_tile(ks[0], kg + (size_t)k_lo * 64 * QKV_DIM, QKV_DIM);
    load_tile(vs[0], vg + (size_t)k_lo * 64 * QKV_DIM, QKV_DIM);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    uint32_t qf[4][4];
    frag_a(qf, ks[1], warp * 16);
    __syncthreads();

    const float scale = LOG2E / sqrtf((float)HEAD_DIM);
    const int q0 = qt * 64 + warp * 16 + g;
    float o[8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    for (int kt = k_lo; kt < k_hi; kt++) {
        const int buf = (kt - k_lo) & 1;
        if (kt + 1 < k_hi) {
            load_tile(ks[buf ^ 1], kg + (size_t)(kt + 1) * 64 * QKV_DIM, QKV_DIM);
            load_tile(vs[buf ^ 1], vg + (size_t)(kt + 1) * 64 * QKV_DIM, QKV_DIM);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        float s[8][4] = {};
        mma_abt(s, qf, ks[buf]);
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
        mma_ab(o, pf, vs[buf]);
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
        const int q = q0 + 8 * r;
        if constexpr (ATT_SPLIT_FWD > 1) {
            const size_t row = ((size_t)sp * N + (size_t)b * T + q) * HEADS + h;
            float* orow = opart + row * HEAD_DIM + 2 * t;
#pragma unroll
            for (int j = 0; j < 8; j++) *(float2*)(orow + 8 * j) = {o[j][2 * r], o[j][2 * r + 1]};
            if (t == 0) mlpart[row] = {m[r], l[r]};
            continue;
        }
        bf16* orow = out + ((size_t)b * T + q) * DIM + h * HEAD_DIM + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++)
            *(bf162*)(orow + 8 * j) = __floats2bfloat162_rn(o[j][2 * r] / l[r], o[j][2 * r + 1] / l[r]);
        if (t == 0) lse[((size_t)b * HEADS + h) * T + q] = m[r] + log2f(l[r]);
    }
}

// Merge attention_fwd's key-tile ranges (ATT_SPLIT_FWD > 1): out = sum_r 2^(m_r - M) o_r / sum_r 2^(m_r - M) l_r and
// lse = M + log2 of that sum, M = max_r m_r, over the ranges that reach the row's query tile. Thread = 8 columns of a
// (token, head) row.
__global__ void attention_fwd_combine(bf16* out, float* lse, const float* opart, const float2* mlpart) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * HEADS * HEAD_DIM / 8) return;
    const int c = i % (HEAD_DIM / 8) * 8, h = i / (HEAD_DIM / 8) % HEADS, n = i / (HEAD_DIM / 8 * HEADS);
    const int ranges = n % T / 64 / FWD_NT + 1;
    float M = -INFINITY;
    for (int r = 0; r < ranges; r++) M = fmaxf(M, mlpart[((size_t)r * N + n) * HEADS + h].x);
    float L = 0.f, acc[8] = {};
    for (int r = 0; r < ranges; r++) {
        const size_t row = ((size_t)r * N + n) * HEADS + h;
        const float w = exp2f(mlpart[row].x - M);
        L += w * mlpart[row].y;
        const float* o = opart + row * HEAD_DIM + c;
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] += w * o[e];
    }
    bf16* orow = out + (size_t)n * DIM + h * HEAD_DIM + c;
#pragma unroll
    for (int e = 0; e < 8; e += 2) *(bf162*)(orow + e) = __floats2bfloat162_rn(acc[e] / L, acc[e + 1] / L);
    if (c == 0) lse[((size_t)(n / T) * HEADS + h) * T + n % T] = M + log2f(L);
}

// Attention backward runs per KV group (the HEADS / KV_HEADS query heads sharing one K, V head): attention_bwd_dkv
// computes S, P, dP, dS for every (query tile, key tile) pair of the group's heads and also stores dS (bf16), so
// attention_bwd_dq is a GEMM over it instead of recomputing S and dP (5 products per tile pair instead of 7). A group's
// dS (B x GROUP x DS_TILES tiles of 8 KB) stays in L2 between the two. The stored values are the bf16 dS that a dq kernel
// recomputing it would pack, bit for bit: a bf16 mma product is transpose-symmetric (chisel e050).
constexpr int GROUP = HEADS / KV_HEADS;
constexpr int DS_TILES = NT * (NT + 1) / 2;                       // causal (query tile, key tile) pairs per head
constexpr size_t DS_GROUP = (size_t)B * GROUP * DS_TILES * TILE;  // bf16 elements of one group's dS
__device__ __forceinline__ size_t ds_tile(int b, int hg, int qt, int kt) {
    return (((size_t)b * GROUP + hg) * DS_TILES + qt * (qt + 1) / 2 + kt) * TILE;
}

// Dsum = rowsum(dO * O) for 64 query rows of one head (two threads per row, 32 columns each), read by attention_bwd_dkv.
__global__ void __launch_bounds__(128) attention_dsum(float* Dsum, const bf16* out, const bf16* dout) {
    const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const size_t row0 = (size_t)b * T + qt * 64, lrow = ((size_t)b * HEADS + h) * T + qt * 64;
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
    if (!(threadIdx.x & 1)) Dsum[lrow + r] = acc;
}

// dk, dv contributions of query head h0 + att_head for 64 key rows (4 warps x 16 keys), over the query tiles that see
// them, and every pair's dS^T tile into ds ([64 keys][64 queries] bf16). The query tiles are cut into ATT_SPLIT
// (targets/*.h) equal ranges, one block each (blockIdx.z = b * ATT_SPLIT + range), so no block walks all NT of them.
// fp32 partials [ATT_SPLIT][N][HEADS][dk 64 | dv 64]; attention_bwd_finish sums the ranges and the group's heads.
constexpr int SPLIT_NT = (NT + ATT_SPLIT - 1) / ATT_SPLIT;  // query tiles per range
__global__ void __launch_bounds__(128) attention_bwd_dkv(float* part, bf16* ds, const bf16* qkv, const bf16* dout,
                                                         const float* lse, const float* Dsum, int h0) {
    const int kt = att_tile<GROUP>(), hg = att_head<GROUP>(), h = h0 + hg;  // kt = 0 has the most work
    const int b = blockIdx.z / ATT_SPLIT, sp = blockIdx.z % ATT_SPLIT;
    const int q_lo = max(kt, sp * SPLIT_NT), q_hi = min(NT, (sp + 1) * SPLIT_NT);
    if (q_lo >= q_hi) return;  // a range wholly before the key tile: attention_bwd_finish skips it
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    const bf16* kg = seq + (HEADS + h / GROUP) * HEAD_DIM;
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
    load_q(q_lo, 0);
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
    for (int qt = q_lo; qt < q_hi; qt++) {
        const int buf = (qt - q_lo) & 1;
        if (qt + 1 < q_hi) load_q(qt + 1, buf ^ 1);
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
        // dS^T tile: sf[kk] holds rows g and g + 8 of the warp's 16 keys, query columns 16 kk + 2 t (+1) and + 8
        uint32_t* d = (uint32_t*)(ds + ds_tile(b, hg, qt, kt) + (warp * 16 + g) * 64 + 2 * t);
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            d[8 * kk] = sf[kk][0];
            d[8 * kk + 4 * 64] = sf[kk][1];
            d[8 * kk + 4] = sf[kk][2];
            d[8 * kk + 4 + 4 * 64] = sf[kk][3];
        }
        mma_ab(dv, pf, dos[buf]);  // dV += P^T dO
        mma_ab(dk, sf, qs[buf]);   // dK += dS^T Q
        __syncthreads();
    }
    const float k = scale / LOG2E;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        float* row = part + (((size_t)sp * N + (size_t)b * T + k0 + 8 * r) * HEADS + h) * 2 * HEAD_DIM + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            *(float2*)(row + 8 * j) = {dk[j][2 * r] * k, dk[j][2 * r + 1] * k};
            *(float2*)(row + HEAD_DIM + 8 * j) = {dv[j][2 * r], dv[j][2 * r + 1]};
        }
    }
}

// dq (before RoPE backward) for 64 query rows of head h0 + att_head, written into dqkv's q columns: dq += dS K over
// the key tiles kt = 0..qt in order, from the dS^T tiles attention_bwd_dkv stored (A fragments through ldmatrix.trans).
__global__ void __launch_bounds__(128) attention_bwd_dq(bf16* dqkv, const bf16* ds, const bf16* qkv, int h0) {
    const int qt = NT - 1 - att_tile<GROUP>(), hg = att_head<GROUP>(), h = h0 + hg, b = blockIdx.z;  // longest first
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* kg = qkv + (size_t)b * T * QKV_DIM + (HEADS + h / GROUP) * HEAD_DIM;
    __shared__ __align__(128) bf16 dst[2][TILE], ks[2][TILE];
    auto load = [&](int kt, int buf) {
        load_tile(dst[buf], ds + ds_tile(b, hg, qt, kt), 64);
        load_tile(ks[buf], kg + (size_t)kt * 64 * QKV_DIM, QKV_DIM);
    };
    load(0, 0);
    cp_commit();
    float dq[8][4] = {};
    for (int kt = 0; kt <= qt; kt++) {
        if (kt < qt) load(kt + 1, (kt + 1) & 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        uint32_t sf[4][4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
            ldsm4t(sf[kk], dst[kt & 1] + swz(kk * 16 + ((lane >> 4) << 3) + (lane & 7), warp * 2 + ((lane >> 3) & 1)));
        mma_ab(dq, sf, ks[kt & 1]);
        __syncthreads();
    }
    const float k = LOG2E / sqrtf((float)HEAD_DIM) / LOG2E;  // dS is w.r.t. scores scaled by 1/sqrt(HEAD_DIM)
    const size_t row0 = (size_t)b * T + qt * 64;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        bf16* row = dqkv + (row0 + warp * 16 + g + 8 * r) * QKV_DIM + h * HEAD_DIM + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++) *(bf162*)(row + 8 * j) = __floats2bfloat162_rn(dq[j][2 * r] * k, dq[j][2 * r + 1] * k);
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
        for (int sp = n % T / 64 / SPLIT_NT; sp < ATT_SPLIT && sp * SPLIT_NT < NT; sp++)  // the ranges written for this key
            for (int hh = kv * G; hh < (kv + 1) * G; hh++) {
                const float* p = part + (((size_t)sp * N + n) * HEADS + hh) * 2 * HEAD_DIM + off;
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

// Unfused forms for targets where cuBLASLt plus a separate pass beats the fused GEMMs (FUSED_QKV_ROPE, FUSED_DOWN_DX
// in targets/*.h). RoPE in place on cuBLASLt's bf16 qkv: one thread per rotation pair (token, q or k head, d < 32).
__global__ void rope_qkv(bf16* qkv, const float2* cs) {
    constexpr int ROT = HEADS + KV_HEADS;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * ROT * 32) return;
    const int d = i % 32, u = i / 32 % ROT, n = i / (32 * ROT);
    bf16* y = qkv + (size_t)n * QKV_DIM + u * 64 + d;
    const float x1 = __bfloat162float(y[0]), x2 = __bfloat162float(y[32]);
    const float2 c = cs[n % T * 32 + d];
    y[0] = __float2bfloat16(x1 * c.x - x2 * c.y);
    y[32] = __float2bfloat16(x2 * c.x + x1 * c.y);
}

// down_dx_swiglu's epilogue on cuBLASLt's bf16 dact = dmo . Wd, 8 FFN columns per thread.
__global__ void swiglu_bwd(bf16* dgu, const bf16* dact, const bf16* gu) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)N * FFN) return;
    const size_t n = i / FFN, c = i % FFN;
    float da[8], gv[8], uv[8];
    load8(da, dact + i);
    load8(gv, gu + n * 2 * FFN + c);
    load8(uv, gu + n * 2 * FFN + FFN + c);
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
        float dg[2], du[2];
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const float g = gv[e + k], sig = 1.f / (1.f + expf(-g));
            du[k] = da[e + k] * round_bf16(g / (1.f + expf(-g)));
            dg[k] = round_bf16(da[e + k] * uv[e + k]) * sig * (1.f + g * (1.f - sig));
        }
        *(bf162*)(dgu + n * 2 * FFN + c + e) = __floats2bfloat162_rn(dg[0], dg[1]);
        *(bf162*)(dgu + n * 2 * FFN + FFN + c + e) = __floats2bfloat162_rn(du[0], du[1]);
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

// Embedding rows of the batch's tokens: grad = (bf16-rounded head grad, in gemb_rows[segment]) + sum of dres rows for
// that token, then AdamW on the row; gemb_full (dumps only) receives the final grad. seg = [perm N][starts N+1]
// [count 1] from the host: positions sorted by token, one block per distinct token, rows summed in a fixed order
// (no atomics).
__global__ void __launch_bounds__(DIM / 4) embed_bwd_adamw(float* p, bf16* pb, float* m, float* v, const float* gemb_rows,
                                                           float* gemb_full, const float* dres, const int* tok,
                                                           const int* seg, const float* step_count) {
    const int* perm = seg;
    const int* starts = seg + N;
    if ((int)blockIdx.x >= starts[N + 1]) return;
    const int s0 = starts[blockIdx.x], s1 = starts[blockIdx.x + 1];
    const size_t o = (size_t)tok[perm[s0]] * DIM + threadIdx.x * 4;
    float4 G = *(const float4*)(gemb_rows + (size_t)blockIdx.x * DIM + threadIdx.x * 4);
    for (int k = s0; k < s1; k++) {
        float4 d = *(const float4*)(dres + (size_t)perm[k] * DIM + threadIdx.x * 4);
        G.x += d.x;
        G.y += d.y;
        G.z += d.z;
        G.w += d.w;
    }
    if (gemb_full) *(float4*)(gemb_full + o) = G;
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

// AdamW for the norm weights (gnorm = [LAYERS x (ln1, ln2)][final norm]), 4 per thread; also refreshes the bf16 copy.
// Layer matrices are updated by adamw_stream, the embedding by head_dw_adamw and embed_bwd_adamw.
// COMPACT: m and v are [G_NORM] (8-bit mode, where fp32 moments cover only the embedding and the norms) instead of
// [P_TOTAL].
template <bool COMPACT>
__global__ void adamw_norms(float* p, bf16* pb, const float* gnorm, float* m, float* v, const float* step_count) {
    const size_t j = 4 * (blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (j >= G_NORM) return;
    const size_t o = j < G_NORM - DIM ? P_EMBED + j / (2 * DIM) * P_LAYER + P_MAT + j % (2 * DIM)
                                      : P_TOTAL - DIM + (j - (G_NORM - DIM));  // parameter offset of gnorm[j]
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    const size_t om = COMPACT ? j : o;
    float4 P = *(float4*)(p + o), G = *(const float4*)(gnorm + j), M = *(float4*)(m + om), V = *(float4*)(v + om);
    adamw_update(P.x, M.x, V.x, G.x, bc1, bc2_sqrt);
    adamw_update(P.y, M.y, V.y, G.y, bc1, bc2_sqrt);
    adamw_update(P.z, M.z, V.z, G.z, bc1, bc2_sqrt);
    adamw_update(P.w, M.w, V.w, G.w, bc1, bc2_sqrt);
    *(float4*)(p + o) = P;
    *(float4*)(m + om) = M;
    *(float4*)(v + om) = V;
    *(uint2*)(pb + o) = {pack2(P.x, P.y), pack2(P.z, P.w)};
}

// dW = dY^T X over the N tokens for a 64 x 64 block of W [OUT][IN] (rows bm.., columns bn..): both operands are
// token-major, so both come through ldmatrix.trans. 4 warps x 16 rows; DW_ST-stage cp.async. acc[j] holds rows g and
// g + 8 of the warp's 16, columns 8 j + 2 t (+1).
constexpr int DW_ST = 3;
template <int OUT, int IN>
__device__ __forceinline__ void dw_loop(float (&acc)[8][4], const bf16* dY, const bf16* X, int bm, int bn, bf16* sm) {
    static_assert(OUT % 64 == 0 && IN % 64 == 0 && N % GBK == 0, "whole dW tiles");
    constexpr int KT = N / GBK;
    bf16 *as = sm, *bs = sm + DW_ST * GBK * 64;  // [GBK tokens][64] tiles of dY and X, 128-byte swizzled rows
    const int lane = threadIdx.x & 31, wm0 = (threadIdx.x >> 5) * 16;
    auto load = [&](int kt, int st) {
        for (int i = threadIdx.x; i < GBK * 8; i += 128) {
            const size_t row = kt * GBK + (i >> 3);
            cp16(as + st * GBK * 64 + swz(i >> 3, i & 7), dY + row * OUT + bm + (i & 7) * 8);
            cp16(bs + st * GBK * 64 + swz(i >> 3, i & 7), X + row * IN + bn + (i & 7) * 8);
        }
    };
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
}

// Layer weight grads: dW rounded to bf16 like PyTorch's dW, into g [OUT][IN]; adamw_stream applies them. Shapes that
// chisel was built for run its hand-scheduled SASS core instead (src/chisel/, same bits: chisel e037).
template <int OUT, int IN>
__global__ void __launch_bounds__(128) dw_bf16(bf16* g, const bf16* dY, const bf16* X) {
    __shared__ __align__(128) bf16 sm[DW_ST * 2 * GBK * 64];
    const int bm = blockIdx.y * 64, bn = blockIdx.x * 64;
    float acc[8][4] = {};
    dw_loop<OUT, IN>(acc, dY, X, bm, bn, sm);
    const int lane = threadIdx.x & 31, gr = lane >> 2, t = lane & 3, wm0 = (threadIdx.x >> 5) * 16;
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            *(bf162*)(g + (size_t)(bm + wm0 + gr + 8 * r) * IN + bn + 8 * j + 2 * t) =
                __floats2bfloat162_rn(acc[j][2 * r], acc[j][2 * r + 1]);
}

// AdamW over n parameters from their bf16 grad (each layer's four matrices, contiguous): the same update as the head's
// fused epilogue, as one streaming pass on a low-priority stream beside the backward kernels. A few persistent blocks;
// loads and stores evict-first (ld/st.global.cs) so the pass keeps out of L2's way (chisel e047-e051).
__global__ void adamw_stream(float* p, float* m, float* v, bf16* pb, const bf16* g, const float* step_count, size_t n) {
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += (size_t)gridDim.x * blockDim.x) {
        float2 P = __ldcs((float2*)p + i), M = __ldcs((float2*)m + i), V = __ldcs((float2*)v + i);
        const float2 G = __bfloat1622float2(__ldcs((const bf162*)g + i));
        adamw_update(P.x, M.x, V.x, G.x, bc1, bc2_sqrt);
        adamw_update(P.y, M.y, V.y, G.y, bc1, bc2_sqrt);
        __stcs((float2*)p + i, P);
        __stcs((float2*)m + i, M);
        __stcs((float2*)v + i, V);
        __stcs((bf162*)pb + i, __floats2bfloat162_rn(P.x, P.y));
    }
}

// 8-bit AdamW state for the layer matrices (make ADAM8=1; format in BENCH.md "8-bit AdamW state"). Blocks of 64
// consecutive parameters, one warp of the pass. m: FP8 E4M3 codes q = e4m3_rn_satfinite(m / s), s = max|m| / 448;
// v: the same for u = sqrtf(v). Dequantized m = float(q) * s, v = (float(q_u) * s_u)^2. A step dequantizes, runs the
// fp32 AdamW update (the weight update uses the new fp32 m, v), then requantizes with the new block maxima.
constexpr int Q8_BLOCK = 64;
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float2 e4m3x2_to_float2(unsigned short q) {
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(q, __NV_E4M3)));
}
__device__ __forceinline__ unsigned short float2_to_e4m3x2(float2 x, float a) {  // a: block max of |x|
    if (a == 0.f) return 0;
    const float s = a / 448.f;
    return __nv_cvt_float2_to_fp8x2(make_float2(x.x / s, x.y / s), __NV_SATFINITE, __NV_E4M3);
}
// Same as adamw_stream over n parameters (n and the start a multiple of Q8_BLOCK), with the moments as codes qm, qu
// [n] and scales sm, su [n / Q8_BLOCK]. Every warp iteration is exactly one block (the grid stride is a multiple of 32).
__global__ void adamw_stream8(float* p, unsigned char* qm, unsigned char* qu, float* sm, float* su, bf16* pb,
                              const bf16* g, const float* step_count, size_t n) {
    static_assert(Q8_BLOCK == 64, "one warp, two parameters per lane");
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += (size_t)gridDim.x * blockDim.x) {
        const size_t blk = i / 32;
        float2 P = __ldcs((float2*)p + i), M = e4m3x2_to_float2(__ldcs((unsigned short*)qm + i)) ;
        const float2 U = e4m3x2_to_float2(__ldcs((unsigned short*)qu + i));
        const float s_m = sm[blk], s_u = su[blk];
        M = {M.x * s_m, M.y * s_m};
        float2 V = {U.x * s_u * (U.x * s_u), U.y * s_u * (U.y * s_u)};
        const float2 G = __bfloat1622float2(__ldcs((const bf162*)g + i));
        adamw_update(P.x, M.x, V.x, G.x, bc1, bc2_sqrt);
        adamw_update(P.y, M.y, V.y, G.y, bc1, bc2_sqrt);
        __stcs((float2*)p + i, P);
        __stcs((bf162*)pb + i, __floats2bfloat162_rn(P.x, P.y));
        const float2 Un = {sqrtf(V.x), sqrtf(V.y)};
        const float am = warp_max(fmaxf(fabsf(M.x), fabsf(M.y))), au = warp_max(fmaxf(Un.x, Un.y));
        __stcs((unsigned short*)qm + i, float2_to_e4m3x2(M, am));
        __stcs((unsigned short*)qu + i, float2_to_e4m3x2(Un, au));
        if ((threadIdx.x & 31) == 0) {
            sm[blk] = am / 448.f;
            su[blk] = au / 448.f;
        }
    }
}

// lm_head dW (tied embedding) with AdamW fused into the epilogue: the grad, rounded to bf16 like PyTorch's dW, updates
// master, moments and the bf16 copy in place while other blocks are still multiplying. Rows whose token is in the
// batch (segmap[row] = its segment, else -1) also get the lookup grad later, so their bf16-rounded head grad goes to
// gemb_rows[segment] (fp32) and embed_bwd_adamw finishes them. gemb_full (dumps only) receives every row's head grad.
__global__ void __launch_bounds__(128) head_dw_adamw(float* p, bf16* pb, float* m, float* v, const bf16* dY,
                                                     const bf16* X, const float* step_count, float* gemb_rows,
                                                     float* gemb_full, const int* segmap) {
    constexpr int OUT = VOCAB, IN = DIM;
    __shared__ __align__(128) bf16 sm[DW_ST * 2 * GBK * 64];
    const int bm = blockIdx.y * 64, bn = blockIdx.x * 64;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, wm0 = (threadIdx.x >> 5) * 16;
    float acc[8][4] = {};
    dw_loop<OUT, IN>(acc, dY, X, bm, bn, sm);
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int row = bm + wm0 + g + 8 * r;
            const size_t o = (size_t)row * IN + bn + 8 * j + 2 * t;
            const bf162 gb = __floats2bfloat162_rn(acc[j][2 * r], acc[j][2 * r + 1]);
            const float2 gf = __bfloat1622float2(gb);
            if (gemb_full) *(float2*)(gemb_full + o) = gf;
            if (const int sg = segmap[row]; sg >= 0) {
                *(float2*)(gemb_rows + (size_t)sg * IN + bn + 8 * j + 2 * t) = gf;
                continue;
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
