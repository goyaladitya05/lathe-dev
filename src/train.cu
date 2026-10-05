// lathe: SmolLM2 training engine, model and GPU fixed at compile time: forward, backward, fused AdamW.
//   make MODEL=smol135 BATCH=1 SEQ=2048
//   build/train_smol135_b1_t2048 --steps 1000 --data data/train.bin   train from --weights (default seed-0 init)
//   build/train_smol135_b1_t2048 --eval 8 --data data/val.bin         forward-only loss on 8 batches
//   --dump PATH writes step 0 tensors (layer 0, last layer, grads, updated weights) for ref/test.py.
//   Training steps replay one captured CUDA graph; --eager 1 launches kernels directly instead.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cublasLt.h>
#if defined(CHISEL_DOWN) || defined(CHISEL_DKV) || defined(PROBE_DW) || defined(CHISEL_DW) || defined(CHISEL_DWS)
#include <cuda.h>
#endif
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include MODEL_H  // models/<MODEL>.h, from the Makefile
#include "targets/sm89.h"

constexpr int B = BATCH, T = SEQ, N = B * T;  // kernels.cuh specializes on these
#ifdef STEP_TIMES
constexpr bool STEP_TIMED = true;
#else
constexpr bool STEP_TIMED = false;
#endif
// AdamW, same as ref/model.py make_optimizer.
constexpr float LR = 3e-4f, BETA1 = 0.9f, BETA2 = 0.95f, ADAM_EPS = 1e-8f, WEIGHT_DECAY = 0.1f;
// Parameter layout, mirrored by ref/model.py lathe_layout. Matrices lead each layer so they stay 256B-aligned.
constexpr size_t P_EMBED = (size_t)VOCAB * DIM;
constexpr size_t P_QKV = (size_t)QKV_DIM * DIM, P_O = (size_t)DIM * DIM;
constexpr size_t P_GU = 2ull * FFN * DIM, P_DOWN = (size_t)DIM * FFN;
constexpr size_t P_LAYER = P_QKV + P_O + P_GU + P_DOWN + 2 * DIM;
constexpr size_t P_TOTAL = P_EMBED + LAYERS * P_LAYER + DIM;
static_assert(P_EMBED * 2 % 256 == 0 && P_QKV * 2 % 256 == 0 && P_O * 2 % 256 == 0 && P_GU * 2 % 256 == 0 &&
                  P_DOWN * 2 % 256 == 0 && P_LAYER * 2 % 256 == 0,
              "matrices must stay 256B-aligned for cuBLASLt");

// Grads: matrix grads (bf16 like PyTorch's dW) and most embedding rows never leave the dW GEMMs, which apply AdamW
// directly. Norm weight grads are fp32 in gnorm = [LAYERS x (ln1, ln2)][final norm].
constexpr size_t P_MAT = P_QKV + P_O + P_GU + P_DOWN;
constexpr size_t G_NORM = LAYERS * 2 * DIM + DIM;

#include "kernels.cuh"
// Hand-scheduled SASS from chisel, loaded as cubins (src/chisel_*.h).
#ifdef CHISEL_DOWN
#include "chisel_gemm.h"
static_assert(N == chisel_down_m && DIM == chisel_down_n && FFN == chisel_down_k, "regenerate chisel_gemm.h for this shape");
#endif
#if defined(PROBE_DW) || defined(CHISEL_DWS)
#include "chisel_dw.h"
#endif
#ifdef CHISEL_DW
#include "chisel_dwadam.h"
static_assert(N == chisel_dwa_qkv_k && QKV_DIM == chisel_dwa_qkv_m && DIM == chisel_dwa_qkv_n && FFN == chisel_dwa_down_n &&
              VOCAB == chisel_dwa_head_m, "regenerate chisel_dwadam.h for this shape");
#endif
#ifdef CHISEL_DKV
#include "chisel_att.h"
static_assert(B == 1 && T == 2048 && HEADS == 9 && KV_HEADS == 3 && HEAD_DIM == 64, "regenerate chisel_att.h for this shape");
#endif

#define CK(x) ck_cuda((x), #x, __LINE__)
#define CKB(x) ck_blas((x), #x, __LINE__)
static void ck_cuda(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "train.cu:%d %s: %s\n", line, what, cudaGetErrorString(e));
        exit(1);
    }
}
static void ck_blas(cublasStatus_t s, const char* what, int line) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "train.cu:%d %s: cublas status %d\n", line, what, (int)s);
        exit(1);
    }
}
static void die(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}
constexpr unsigned cdiv(size_t a, size_t b) { return (unsigned)((a + b - 1) / b); }

template <class X>
struct LayerP {
    X *qkv, *o, *gu, *down, *ln1, *ln2;
};
template <class X>
static LayerP<X> layer(X* p, int l) {
    X* q = p + P_EMBED + l * P_LAYER;
    X* ln = q + P_QKV + P_O + P_GU + P_DOWN;
    return {q, q + P_QKV, q + P_QKV + P_O, q + P_QKV + P_O + P_GU, ln, ln + DIM};
}

// Per-step host data: tokens, targets, embedding segments [perm N][starts N+1][count], token -> segment (or -1).
constexpr int IO_SEG = 2 * N, IO_SEGMAP = 4 * N + 2, IO_INTS = IO_SEGMAP + VOCAB;

// Per-layer activations kept for backward.
struct Acts {
    float* res;       // [LAYERS+1][N][DIM] residual stream into each layer, res[LAYERS] feeds the final norm
    float* res_mid;   // [LAYERS][N][DIM] after the attention residual add
    bf16 *ln1, *ln2;  // [LAYERS][N][DIM]
    bf16* qkv;        // [LAYERS][N][QKV_DIM] post-RoPE
    bf16* att;        // [LAYERS][N][DIM]
    float* lse;       // [LAYERS][B][HEADS][T]
    bf16* gu;         // [LAYERS][N][2*FFN]
    bf16* act;        // [LAYERS][N][FFN]
    bf16* lnf;        // [N][DIM]
    bf16* branch;     // [N][DIM] o_proj / down_proj output before the residual add (scratch)
    bf16* logits;     // [N][VOCAB], overwritten with dlogits by cross_entropy
    float* losses;    // [N]
    float* loss;      // [1] mean
    int* io;          // [IO_INTS]
};

// Backward scratch, reused by every layer.
struct Bwd {
    float* dres;     // [N][DIM] grad of the residual stream
    bf16* dres_bf;   // [N][DIM] bf16 copy: grad into the branch outputs
    bf16* dln;       // [N][DIM] grad of a norm output
    bf16* datt;      // [N][DIM]
    bf16* dqkv;      // [N][QKV_DIM]
    bf16* dgu;       // [N][2*FFN]
    float* dsum;     // [B][HEADS][T] rowsum(dO * O)
    float* part;     // [N][HEADS][2*HEAD_DIM] per-head dk | dv, summed by attention_bwd_finish
    float* dw_part;  // [N / NORM_ROWS][DIM] per-block norm weight grads
    float* gemb_rows;  // [N][DIM] fp32 head grads of the batch's embedding rows, by segment
    bf16* gmat;      // [P_MAT] one layer's matrix grads, dumps only (nullptr otherwise)
};

struct State {
    float* master;  // [P_TOTAL] fp32 master weights
    bf16* params;   // [P_TOTAL] bf16 working copy (matmul weights)
    float* gnorm;   // [G_NORM] norm weight grads
    float* gemb;    // [P_EMBED] full fp32 embedding grad, dumps only (nullptr otherwise)
    float *m, *v;   // [P_TOTAL] AdamW moments
    float2* rope;   // [T][HEAD_DIM/2] (cos, sin)
    float* step;    // [1] AdamW step count, advanced on the device so a captured step graph can be replayed
    void* workspace;
    Acts a;
    Bwd g;
};

// One cudaMalloc. carve() runs twice: once on a null base to size the arena, once to assign pointers.
struct Arena {
    char* base;
    size_t used;
    template <class X>
    X* take(size_t n) {
        used = (used + 255) & ~size_t(255);
        X* p = (X*)(base + used);
        used += n * sizeof(X);
        return p;
    }
};

static void carve(Arena& ar, State& s, bool dumping) {  // test-only buffers exist only when dumping
    const size_t ND = (size_t)N * DIM;
    s.master = ar.take<float>(P_TOTAL);
    s.params = ar.take<bf16>(P_TOTAL);
    s.gnorm = ar.take<float>(G_NORM);
    s.gemb = dumping ? ar.take<float>(P_EMBED) : nullptr;
    s.m = ar.take<float>(P_TOTAL);
    s.v = ar.take<float>(P_TOTAL);
    s.rope = ar.take<float2>(T * HEAD_DIM / 2);
    s.step = ar.take<float>(1);
    s.workspace = ar.take<char>(CUBLAS_WORKSPACE);
    Acts& a = s.a;
    a.res = ar.take<float>((LAYERS + 1) * ND);
    a.res_mid = ar.take<float>(LAYERS * ND);
    a.ln1 = ar.take<bf16>(LAYERS * ND);
    a.ln2 = ar.take<bf16>(LAYERS * ND);
    a.qkv = ar.take<bf16>((size_t)LAYERS * N * QKV_DIM);
    a.att = ar.take<bf16>(LAYERS * ND);
    a.lse = ar.take<float>((size_t)LAYERS * N * HEADS);
    a.gu = ar.take<bf16>((size_t)LAYERS * N * 2 * FFN);
    a.act = ar.take<bf16>((size_t)LAYERS * N * FFN);
    a.lnf = ar.take<bf16>(ND);
    a.branch = ar.take<bf16>(ND);
    a.logits = ar.take<bf16>((size_t)N * VOCAB);
    a.losses = ar.take<float>(N);
    a.loss = ar.take<float>(1);
    a.io = ar.take<int>(IO_INTS);
    Bwd& g = s.g;
    g.dres = ar.take<float>(ND);
    g.dres_bf = ar.take<bf16>(ND);
    g.dln = ar.take<bf16>(ND);
    g.datt = ar.take<bf16>(ND);
    g.dqkv = ar.take<bf16>((size_t)N * QKV_DIM);
    g.dgu = ar.take<bf16>((size_t)N * 2 * FFN);
    g.dsum = ar.take<float>((size_t)N * HEADS);
    g.part = ar.take<float>((size_t)N * HEADS * 2 * HEAD_DIM);
    g.dw_part = ar.take<float>((size_t)N / NORM_ROWS * DIM);
    g.gemb_rows = ar.take<float>(ND);
    g.gmat = dumping ? ar.take<bf16>(P_MAT) : nullptr;
}

static cudaStream_t stream;
static cublasLtHandle_t lt;

#if defined(CHISEL_DOWN) || defined(CHISEL_DKV) || defined(PROBE_DW) || defined(CHISEL_DW) || defined(CHISEL_DWS)
#define CKD(x) ck_drv((x), #x, __LINE__)
static void ck_drv(CUresult r, const char* what, int line) {
    if (r != CUDA_SUCCESS) {
        const char* s = "?";
        cuGetErrorString(r, &s);
        fprintf(stderr, "train.cu:%d %s: %s\n", line, what, s);
        exit(1);
    }
}
static CUfunction chisel_fn(const unsigned char* image, const char* name) {  // after the runtime made the context current
    CUmodule m;
    CUfunction f;
    CKD(cuModuleLoadData(&m, image));
    CKD(cuModuleGetFunction(&f, m, name));
    return f;
}
#endif
#ifdef CHISEL_DOWN
static CUfunction chisel_down_fn;
// y[N][DIM] = x[N][FFN] W^T with W [DIM][FFN], same contract as gemm(L.down, ...) (bit-identical, chisel e029)
static void chisel_down_fwd(bf16* y, const bf16* x, const bf16* w) {
    void* args[] = {&y, &x, &w};
    CKD(cuLaunchKernel(chisel_down_fn, chisel_down_grid[0], chisel_down_grid[1], chisel_down_grid[2], chisel_down_block,
                       1, 1, 0, (CUstream)stream, args, nullptr));
}
#endif
#ifdef PROBE_DW
static CUfunction probe_fn[5];
static bf16* probe_out;
static const unsigned* probe_grid[5] = {chisel_dw_qkv_grid, chisel_dw_o_grid, chisel_dw_gu_grid, chisel_dw_down_grid,
                                        chisel_dw_head_grid};
static const unsigned probe_block[5] = {chisel_dw_qkv_block, chisel_dw_o_block, chisel_dw_gu_block, chisel_dw_down_block,
                                        chisel_dw_head_block};
// chisel's dW core (dW[out][in] = dy^T x, bf16) on the inputs of the dW + AdamW kernel that just ran
static void probe_dw(int which, const bf16* dy, const bf16* x) {
    void* args[] = {&probe_out, &dy, &x};
    CKD(cuLaunchKernel(probe_fn[which], probe_grid[which][0], probe_grid[which][1], probe_grid[which][2],
                       probe_block[which], 1, 1, 0, (CUstream)stream, args, nullptr));
}
#endif
#ifdef CHISEL_DW
static CUfunction cdw_fn[5];
static float* chisel_bc;  // [LR / bc1, bc2_sqrt] of this step
static const unsigned* cdw_grid[5] = {chisel_dwa_qkv_grid, chisel_dwa_o_grid, chisel_dwa_gu_grid, chisel_dwa_down_grid,
                                      chisel_dwa_head_grid};
static const unsigned cdw_block[5] = {chisel_dwa_qkv_block, chisel_dwa_o_block, chisel_dwa_gu_block, chisel_dwa_down_block,
                                      chisel_dwa_head_block};
// the bias corrections with dw_adamw's own expressions, once per step (chisel's kernels read them)
__global__ void chisel_bias(float* bc, const float* step_count) {
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    bc[0] = LR / bc1;
    bc[1] = bc2_sqrt;
}
static int cdw_which(int out, int in) {
    return out == QKV_DIM ? 0 : out == 2 * FFN ? 2 : in == FFN ? 3 : 1;
}
// dw_adamw<out, in>'s contract without gout/gemb_full (bit-identical, chisel e045); head: segmap and gemb_rows
static void chisel_dw(int which, float* p, bf16* pb, float* m, float* v, const bf16* dy, const bf16* x,
                      const int* segmap = nullptr, float* gemb = nullptr) {
    void* args[] = {&p, &pb, &m, &v, &dy, &x, &chisel_bc, &segmap, &gemb};
    CKD(cuLaunchKernel(cdw_fn[which], cdw_grid[which][0], cdw_grid[which][1], cdw_grid[which][2], cdw_block[which], 1, 1,
                       0, (CUstream)stream, args, nullptr));
}
#endif
#ifdef CHISEL_DWS
static cudaStream_t dws_side;  // lowest priority; the main stream gets the highest
static CUfunction dws_fn[4];
static bf16* dws_grad[3];      // bf16 dW of a layer's four matrices (lathe's parameter layout), a ring of 3 layers
static cudaEvent_t dws_dw[3], dws_ad[3], dws_join;
static const unsigned* dws_grid[4] = {chisel_dw_qkv_grid, chisel_dw_o_grid, chisel_dw_gu_grid, chisel_dw_down_grid};
static const unsigned dws_block[4] = {chisel_dw_qkv_block, chisel_dw_o_block, chisel_dw_gu_block, chisel_dw_down_block};
static int dws_which(int out, int in) { return out == QKV_DIM ? 0 : out == 2 * FFN ? 2 : in == FFN ? 3 : 1; }
// dW[out][in] = dy^T x rounded to bf16, as dw_adamw rounds it (chisel e037: bit-identical)
static void dws_core(int which, bf16* y, const bf16* dy, const bf16* x) {
    void* args[] = {&y, &dy, &x};
    CKD(cuLaunchKernel(dws_fn[which], dws_grid[which][0], dws_grid[which][1], dws_grid[which][2], dws_block[which], 1,
                       1, 0, (CUstream)stream, args, nullptr));
}
// dw_adamw's update from the bf16 gradient, grid-stride (10 persistent blocks: e047)
__global__ void dws_adamw(float* p, float* m, float* v, bf16* pb, const bf16* g, const float* step_count, size_t n) {
    const float step = *step_count, bc1 = 1.f - powf(BETA1, step), bc2_sqrt = sqrtf(1.f - powf(BETA2, step));
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += (size_t)gridDim.x * blockDim.x) {
#ifdef DWS_CS  // streaming cache hints (evict first): keep the pass out of L2's way
        float2 P = __ldcs((float2*)p + i), M = __ldcs((float2*)m + i), V = __ldcs((float2*)v + i);
        const bf162 Gb = __ldcs((const bf162*)g + i);
        const float2 G = __bfloat1622float2(Gb);
#else
        float2 P = ((float2*)p)[i], M = ((float2*)m)[i], V = ((float2*)v)[i];
        const float2 G = __bfloat1622float2(((const bf162*)g)[i]);
#endif
        adamw_update(P.x, M.x, V.x, G.x, bc1, bc2_sqrt);
        adamw_update(P.y, M.y, V.y, G.y, bc1, bc2_sqrt);
#ifdef DWS_CS
        __stcs((float2*)p + i, P);
        __stcs((float2*)m + i, M);
        __stcs((float2*)v + i, V);
        __stcs((bf162*)pb + i, __floats2bfloat162_rn(P.x, P.y));
#else
        ((float2*)p)[i] = P;
        ((float2*)m)[i] = M;
        ((float2*)v)[i] = V;
        ((bf162*)pb)[i] = __floats2bfloat162_rn(P.x, P.y);
#endif
    }
}
static_assert(P_MAT % 2 == 0, "dws_adamw works on pairs");
static cudaEvent_t dws_go;
#ifndef DWS_BLOCKS
#define DWS_BLOCKS 10
#endif
// AdamW over params [a, b) of layer l's matrices on the side stream, after everything issued so far on the main stream
template <class S>
static void dws_range(S& s, int l, size_t a, size_t b) {
    const size_t o0 = P_EMBED + l * P_LAYER + a;
    CK(cudaEventRecord(dws_go, stream));
    CK(cudaStreamWaitEvent(dws_side, dws_go));
    dws_adamw<<<DWS_BLOCKS, 256, 0, dws_side>>>(s.master + o0, s.m + o0, s.v + o0, s.params + o0, dws_grad[l % 3] + a,
                                                s.step, b - a);
}
template <class S>
static void dws_pass(S& s, int l) {  // all of layer l; its ring slot is free after this
    dws_range(s, l, 0, P_MAT);
    CK(cudaEventRecord(dws_ad[l % 3], dws_side));
}
static_assert(P_QKV % 2 == 0, "dws_adamw works on pairs");
#endif
#ifdef CHISEL_DSQ
static_assert(B == 1 && HEADS == 9 && KV_HEADS == 3, "dS reuse: one sequence, three KV groups of three heads");
constexpr int DSQ_TILES = NT * (NT + 1) / 2;  // causal (qt, kt) tiles per head
static cudaStream_t dsq_s[3];
static cudaEvent_t dsq_fork, dsq_g[3];
static bf16* dsq_buf;                         // dS^T tiles of two groups, bf16 [64 k][64 q]
__device__ __forceinline__ size_t dsq_tile(int hg, int qt, int kt) {
    return ((size_t)hg * DSQ_TILES + qt * (qt + 1) / 2 + kt) * TILE;
}
// Dsum = rowsum(dO * O) with attention_bwd_dq's arithmetic
__global__ void __launch_bounds__(128) dsq_dsum(float* Dsum, const bf16* out, const bf16* dout) {
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
// attention_bwd_dkv for heads h0 + blockIdx.y, also storing every tile's dS^T (the bf16 values attention_bwd_dq packs:
// a bf16 mma product is transpose-symmetric, chisel e050)
__global__ void __launch_bounds__(128) dsq_dkv(float* part, bf16* ds, const bf16* qkv, const bf16* dout, const float* lse,
                                               const float* Dsum, int h0) {
    const int kt = blockIdx.x, hg = blockIdx.y, h = h0 + hg, b = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* seq = qkv + (size_t)b * T * QKV_DIM;
    const bf16* kg = seq + (HEADS + h / (HEADS / KV_HEADS)) * HEAD_DIM;
    const size_t lrow = ((size_t)b * HEADS + h) * T;
    __shared__ __align__(128) bf16 qs[2][TILE], dos[2][TILE];
    __shared__ __align__(16) float ls[2][64], dds[2][64];
    auto load_q = [&](int qt, int buf) {
        load_tile(qs[buf], seq + (size_t)qt * 64 * QKV_DIM + h * HEAD_DIM, QKV_DIM);
        load_tile(dos[buf], dout + ((size_t)b * T + qt * 64) * DIM + h * HEAD_DIM, DIM);
        if (threadIdx.x < 16) cp16(&ls[buf][threadIdx.x * 4], lse + lrow + qt * 64 + threadIdx.x * 4);
        else if (threadIdx.x < 32) cp16(&dds[buf][(threadIdx.x - 16) * 4], Dsum + lrow + qt * 64 + (threadIdx.x - 16) * 4);
    };
    load_tile(qs[1], kg + (size_t)kt * 64 * QKV_DIM, QKV_DIM);
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
        mma_abt(st, kf, qs[buf]);
        mma_abt(dpt, vf, dos[buf]);
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
        uint32_t* d = (uint32_t*)(ds + dsq_tile(hg, qt, kt) + (warp * 16 + g) * 64 + 2 * t);
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            d[8 * kk] = sf[kk][0];
            d[8 * kk + 4 * 64] = sf[kk][1];
            d[8 * kk + 4] = sf[kk][2];
            d[8 * kk + 4 + 4 * 64] = sf[kk][3];
        }
        mma_ab(dv, pf, dos[buf]);
        mma_ab(dk, sf, qs[buf]);
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
// dq of 64 query rows of head h0 + blockIdx.y: dq += dS K over kt = 0..qt, attention_bwd_dq's order and epilogue
__global__ void __launch_bounds__(128) dsq_dq(bf16* dqkv, const bf16* ds, const bf16* qkv, int h0) {
    const int qt = NT - 1 - blockIdx.x, hg = blockIdx.y, h = h0 + hg, b = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const bf16* kg = qkv + (size_t)b * T * QKV_DIM + (HEADS + h / (HEADS / KV_HEADS)) * HEAD_DIM;
    __shared__ __align__(128) bf16 dst[2][TILE], ks[2][TILE];
    auto load = [&](int kt, int buf) {
        load_tile(dst[buf], ds + dsq_tile(hg, qt, kt), 64);
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
    const float k = LOG2E / sqrtf((float)HEAD_DIM) / LOG2E;
    const size_t row0 = (size_t)b * T + qt * 64;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        bf16* row = dqkv + (row0 + warp * 16 + g + 8 * r) * QKV_DIM + h * HEAD_DIM + 2 * t;
#pragma unroll
        for (int j = 0; j < 8; j++) *(bf162*)(row + 8 * j) = __floats2bfloat162_rn(dq[j][2 * r] * k, dq[j][2 * r + 1] * k);
    }
}
static void dsq_attention(float* part, bf16* dqkv, float* dsum, const bf16* qkv, const bf16* out, const bf16* dout,
                          const float* lse) {
    dsq_dsum<<<dim3(NT, HEADS, B), 128, 0, stream>>>(dsum, out, dout);
    CK(cudaEventRecord(dsq_fork, stream));
    for (int gi = 0; gi < 3; gi++) {  // group gi reuses buffer gi % 2 after group gi - 2's dq has read it
        bf16* buf = dsq_buf + (size_t)(gi % 2) * 3 * DSQ_TILES * TILE;
        CK(cudaStreamWaitEvent(dsq_s[gi], dsq_fork));
        if (gi >= 2) CK(cudaStreamWaitEvent(dsq_s[gi], dsq_g[gi - 2]));
        dsq_dkv<<<dim3(NT, 3, B), 128, 0, dsq_s[gi]>>>(part, buf, qkv, dout, lse, dsum, 3 * gi);
        dsq_dq<<<dim3(NT, 3, B), 128, 0, dsq_s[gi]>>>(dqkv, buf, qkv, 3 * gi);
        CK(cudaEventRecord(dsq_g[gi], dsq_s[gi]));
    }
    for (int gi = 0; gi < 3; gi++) CK(cudaStreamWaitEvent(stream, dsq_g[gi]));
}
#endif
#ifdef CHISEL_DXPF
// lathe's down_dx_swiglu, plus: the block's gate and up lines (its epilogue's only DRAM reads) prefetched into L2
// while the GEMM runs; 64 rows x 2 lines of 128 bytes, one per thread
__global__ void __launch_bounds__(GBM / GWM * 32) down_dx_swiglu_pf(bf16* dgu, const bf16* dmo, const bf16* wd, const bf16* gu) {
    static_assert(GBM / GWM * 32 == 2 * GBM, "one prefetch per thread");
    __shared__ __align__(128) bf16 sm[GST * (GBM + 64) * GBK];
    const int bm = blockIdx.y * GBM, bn = blockIdx.x * 64;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(gu + (size_t)(bm + (threadIdx.x >> 1)) * 2 * FFN + (threadIdx.x & 1) * FFN + bn));
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
#endif
#ifdef CHISEL_DKV
static CUfunction chisel_dkv_fn;
// attention_bwd_dkv's contract (bit-identical); grid (NT, HEADS, B) x 128
static void chisel_dkv_launch(float* part, const bf16* qkv, const bf16* dout, const float* lse, const float* dsum) {
    void* args[] = {&part, &qkv, &dout, &lse, &dsum};
    CKD(cuLaunchKernel(chisel_dkv_fn, NT, HEADS, B, 128, 1, 1, 0, (CUstream)stream, args, nullptr));
}
#endif

// Per-op GPU time (make ... STEP_TIMES=1): an event after every launch, eager launches, a table at the end.
#ifdef STEP_TIMES
struct Mark {
    const char* name;
    cudaEvent_t ev;
};
static std::vector<Mark> marks;
static size_t mark_i;
static void mark(const char* name) {
    if (mark_i == marks.size()) {
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        marks.push_back({name, e});
    }
    marks[mark_i].name = name;
    CK(cudaEventRecord(marks[mark_i++].ev, stream));
}
#else
static void mark(const char*) {}
#endif

// Column-major D[m x n] = op(A)[m x k] B[k x n], bf16 inputs and output, fp32 accumulate. Built once per shape.
struct Gemm {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a, b, d;
    cublasLtMatmulAlgo_t algo;
};

static Gemm make_gemm(bool ta, int m, int n, int k) {
    Gemm g;
    CKB(cublasLtMatmulDescCreate(&g.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t opa = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    CKB(cublasLtMatmulDescSetAttribute(g.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)));
    CKB(cublasLtMatrixLayoutCreate(&g.a, CUDA_R_16BF, ta ? k : m, ta ? m : k, ta ? k : m));
    CKB(cublasLtMatrixLayoutCreate(&g.b, CUDA_R_16BF, k, n, k));
    CKB(cublasLtMatrixLayoutCreate(&g.d, CUDA_R_16BF, m, n, m));
    cublasLtMatmulPreference_t pref;
    CKB(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = CUBLAS_WORKSPACE;
    CKB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    uint32_t red = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;  // split-K partials summed in fp32, never in bf16
    CKB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof(red)));
    cublasLtMatmulHeuristicResult_t h;
    int found = 0;
    CKB(cublasLtMatmulAlgoGetHeuristic(lt, g.desc, g.a, g.b, g.d, g.d, pref, 1, &h, &found));
    if (!found) die("no cuBLASLt algo");
    CKB(cublasLtMatmulPreferenceDestroy(pref));
    g.algo = h.algo;
    return g;
}

static void gemm(const State& s, const Gemm& g, void* d, const bf16* a, const bf16* b) {
    const float one = 1.f, zero = 0.f;
    CKB(cublasLtMatmul(lt, g.desc, &one, a, g.a, b, g.b, &zero, d, g.d, d, g.d, &g.algo, s.workspace,
                       CUBLAS_WORKSPACE, stream));
}

// The cuBLASLt GEMMs of linears y = x W^T, W [out][in] row-major, x/y [N][features] row-major. The qkv forward,
// the down dX and every dW are own kernels with fused epilogues (qkv_rope, down_dx_swiglu, dw_adamw).
// Forward y[N][out] = x W^T (A = W, B = x); input grad dx[N][in] = dy W (A = W, B = dy).
static Gemm make_fwd(int out, int in) { return make_gemm(true, out, N, in); }
static Gemm make_dx(int out, int in) { return make_gemm(false, in, N, out); }
struct Lins {
    Gemm o, gu, down, head;             // forward
    Gemm qkv_dx, o_dx, gu_dx, head_dx;  // input grads
};

// Test hook: [name[48], dtype i32 (0 bf16, 1 f32, 2 i32), pad i32, rows i64, cols i64] + data. Read by ref/test.py.
static void dump(FILE* f, const char* name, const void* p, int dtype, size_t rows, size_t cols) {
    if (!f) return;
    size_t bytes = rows * cols * (dtype == 0 ? 2 : 4);
    std::vector<char> h(bytes);
    CK(cudaStreamSynchronize(stream));
    CK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
    char hdr[72] = {};
    snprintf(hdr, 48, "%s", name);
    int64_t shape[2] = {(int64_t)rows, (int64_t)cols};
    memcpy(hdr + 48, &dtype, 4);
    memcpy(hdr + 56, shape, 16);
    fwrite(hdr, 1, sizeof(hdr), f);
    fwrite(h.data(), 1, bytes, f);
}

// Dumps tensors of layers 0 and LAYERS-1 only, named "L<l>.<name>".
struct LayerDump {
    FILE* f;
    int l;
    void operator()(const char* name, const void* p, int dtype, size_t cols) const {
        if (!f || (l != 0 && l != LAYERS - 1)) return;
        char nm[48];
        snprintf(nm, sizeof(nm), "L%d.%s", l, name);
        dump(f, nm, p, dtype, N, cols);
    }
};

static void forward(State& s, const Lins& L, FILE* df) {
    Acts& a = s.a;
    const size_t ND = (size_t)N * DIM;
    const int* tokens = a.io;
    embed<<<cdiv(N * DIM / 4, 256), 256, 0, stream>>>(a.res, tokens, s.master);
    mark("fwd embedding");
    rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(a.ln1, nullptr, a.res, nullptr, layer(s.master, 0).ln1);
    mark("fwd rmsnorm");
    for (int l = 0; l < LAYERS; l++) {
        const LayerP<bf16> w = layer(s.params, l);
        const LayerP<float> wf = layer(s.master, l);
        float *res = a.res + l * ND, *res_mid = a.res_mid + l * ND;
        bf16 *ln1 = a.ln1 + l * ND, *ln2 = a.ln2 + l * ND, *att = a.att + l * ND;
        bf16 *qkv = a.qkv + (size_t)l * N * QKV_DIM, *gu = a.gu + (size_t)l * N * 2 * FFN;
        bf16* act = a.act + (size_t)l * N * FFN;
        float* lse = a.lse + (size_t)l * N * HEADS;
        const bool last = l == LAYERS - 1;
        const LayerDump put{df, l};

        put("res", res, 1, DIM);
        put("ln1", ln1, 0, DIM);
        qkv_rope<<<dim3(QKV_DIM / 64, N / GBM), GBM / GWM * 32, 0, stream>>>(qkv, ln1, w.qkv, s.rope);
        mark("fwd qkv GEMM + RoPE (own)");
        put("qkv", qkv, 0, QKV_DIM);
        attention_fwd<<<dim3(NT, HEADS, B), 128, 0, stream>>>(att, lse, qkv);
        mark("fwd attention");
        put("att", att, 0, DIM);
        gemm(s, L.o, a.branch, w.o, att);
        mark("fwd o GEMM 2048x576x576");
        put("ao", a.branch, 0, DIM);
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(ln2, res_mid, res, a.branch, wf.ln2);
        mark("fwd rmsnorm");
        put("res_mid", res_mid, 1, DIM);
        put("ln2", ln2, 0, DIM);
        gemm(s, L.gu, gu, w.gu, ln2);
        mark("fwd gate_up GEMM 2048x3072x576");
        put("gu", gu, 0, 2 * FFN);
        swiglu<<<cdiv(N * FFN / 8, 256), 256, 0, stream>>>(act, gu);
        mark("fwd swiglu");
        put("act", act, 0, FFN);
#ifdef CHISEL_DOWN
        chisel_down_fwd(a.branch, act, w.down);
        mark("fwd down GEMM 2048x576x1536 (chisel)");
#else
        gemm(s, L.down, a.branch, w.down, act);
        mark("fwd down GEMM 2048x576x1536");
#endif
        put("mo", a.branch, 0, DIM);
        // Next layer's input norm, or the final norm.
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(last ? a.lnf : a.ln1 + (l + 1) * ND, res + ND, res_mid, a.branch,
                                                last ? s.master + P_TOTAL - DIM : layer(s.master, l + 1).ln1);
        mark("fwd rmsnorm");
        put("res_out", res + ND, 1, DIM);
    }
    gemm(s, L.head, a.logits, s.params, a.lnf);  // tied embedding
    dump(df, "lnf", a.lnf, 0, N, DIM);
    mark("fwd head GEMM 2048x49152x576");
    dump(df, "logits", a.logits, 0, N, VOCAB);
    cross_entropy<<<N, CE_THREADS, 0, stream>>>(a.losses, a.logits, tokens + N);
    mark("cross entropy");
    mean_loss<<<1, 1024, 0, stream>>>(a.loss, a.losses);
    mark("loss");
    CK(cudaGetLastError());
    dump(df, "dlogits", a.logits, 0, N, VOCAB);
    dump(df, "losses", a.losses, 1, N, 1);
    dump(df, "tokens", a.io, 2, 2, N);
}

static void backward(State& s, const Lins& L, FILE* df) {
    Acts& a = s.a;
    Bwd& g = s.g;
    const size_t ND = (size_t)N * DIM;
#ifdef CHISEL_DW
    chisel_bias<<<1, 1, 0, stream>>>(chisel_bc, s.step);
    mark("chisel AdamW bias corrections");
#endif
    gemm(s, L.head_dx, g.dln, s.params, a.logits);
    mark("bwd head dX GEMM 2048x576x49152");
    {  // tied embedding: head dW with AdamW fused for rows not in the batch; the rest finish in embed_bwd_adamw
        constexpr auto k = dw_adamw<VOCAB, DIM, true>;
#ifdef CHISEL_DW
        if (!df)
            chisel_dw(4, s.master, s.params, s.m, s.v, a.logits, a.lnf, a.io + IO_SEGMAP, g.gemb_rows);
        else
#endif
        k<<<dim3(DIM / 64, VOCAB / 64), 128, 0, stream>>>(s.master, s.params, s.m, s.v, nullptr, a.logits, a.lnf, s.step,
                                                          g.gemb_rows, df ? s.gemb : nullptr, a.io + IO_SEGMAP);
        mark("bwd head dW + AdamW (own) 49152x576x2048");
#ifdef PROBE_DW
        probe_dw(4, a.logits, a.lnf);
        mark("probe chisel dW core head");
#endif
    }
    dump(df, "dlnf", g.dln, 0, N, DIM);
    rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, g.dres_bf, g.dw_part, nullptr, g.dln, a.res + LAYERS * ND,
                                                s.master + P_TOTAL - DIM);
    mark("bwd rmsnorm");
    colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(s.gnorm + G_NORM - DIM, g.dw_part);
    mark("bwd norm weight colsum");
    for (int l = LAYERS - 1; l >= 0; l--) {
        const LayerP<bf16> w = layer(s.params, l);
        const LayerP<float> wf = layer(s.master, l);
        bf16* gm = g.gmat;  // matrix grads are only stored when dumping (tests)
        const LayerP<bf16> gr = df ? LayerP<bf16>{gm, gm + P_QKV, gm + P_QKV + P_O, gm + P_QKV + P_O + P_GU, nullptr, nullptr}
                                   : LayerP<bf16>{};
        // dW of one weight matrix of this layer (at layer offset off) with its AdamW update fused in. Each runs
        // after the matrix's dX GEMM, the last reader of the old weights in this step.
        auto dw = [&](auto kernel, int out, int in, size_t off, bf16* gout, const bf16* dy, const bf16* x) {
            const size_t o = P_EMBED + l * P_LAYER + off;
#ifdef CHISEL_DWS
            if (!df) {
                const int slot = l % 3;
                if (off == P_QKV + P_O + P_GU && l + 3 < LAYERS)  // the layer's first dW: its ring slot's AdamW done
                    CK(cudaStreamWaitEvent(stream, dws_ad[slot]));
                if (CHISEL_DWS == 5 && off == P_QKV + P_O + P_GU && l + 1 < LAYERS) dws_pass(s, l + 1);
                dws_core(dws_which(out, in), dws_grad[slot] + off, dy, x);
                if (off == 0 && (CHISEL_DWS == 1 || (CHISEL_DWS != 1 && CHISEL_DWS != 3 && l == 0)))
                    dws_pass(s, l);  // 2, 4: with the next layer's attention backward / o dX
                if (off == 0 && CHISEL_DWS == 3 && l == 0) {  // layer 0's qkv; the rest went with its attention
                    dws_range(s, 0, 0, P_QKV);
                    CK(cudaEventRecord(dws_ad[0], dws_side));
                }
                return;
            }
#endif
#ifdef CHISEL_DW
            if (!df) return chisel_dw(cdw_which(out, in), s.master + o, s.params + o, s.m + o, s.v + o, dy, x);
#endif
            kernel<<<dim3(in / 64, out / 64), 128, 0, stream>>>(s.master + o, s.params + o, s.m + o, s.v + o, gout, dy, x,
                                                               s.step, nullptr, nullptr, nullptr);
        };
        float* gln = s.gnorm + l * 2 * DIM;
        float *res = a.res + l * ND, *res_mid = a.res_mid + l * ND;
        bf16 *ln1 = a.ln1 + l * ND, *ln2 = a.ln2 + l * ND, *att = a.att + l * ND;
        bf16 *qkv = a.qkv + (size_t)l * N * QKV_DIM, *gu = a.gu + (size_t)l * N * 2 * FFN;
        bf16* act = a.act + (size_t)l * N * FFN;
        float* lse = a.lse + (size_t)l * N * HEADS;
        const LayerDump put{df, l};

        put("dres_out", g.dres, 1, DIM);
#ifdef CHISEL_DXPF
        down_dx_swiglu_pf<<<dim3(FFN / 64, N / GBM), GBM / GWM * 32, 0, stream>>>(g.dgu, g.dres_bf, w.down, gu);
        mark("bwd down dX GEMM + SwiGLU bwd (own) 2048x1536x576 (chisel)");
#else
        down_dx_swiglu<<<dim3(FFN / 64, N / GBM), GBM / GWM * 32, 0, stream>>>(g.dgu, g.dres_bf, w.down, gu);
        mark("bwd down dX GEMM + SwiGLU bwd (own) 2048x1536x576");
#endif
        dw(dw_adamw<DIM, FFN>, DIM, FFN, P_QKV + P_O + P_GU, gr.down, g.dres_bf, act);
        mark("bwd down dW + AdamW (own) 576x1536x2048");
#ifdef PROBE_DW
        probe_dw(3, g.dres_bf, act);
        mark("probe chisel dW core down");
#endif
        put("dgu", g.dgu, 0, 2 * FFN);
#if defined(CHISEL_DWS) && CHISEL_DWS == 6
        if (!df && l + 1 < LAYERS) dws_pass(s, l + 1);  // the previous layer's AdamW from this gate_up dX GEMM on
#endif
        gemm(s, L.gu_dx, g.dln, w.gu, g.dgu);
        mark("bwd gate_up dX GEMM 2048x576x3072");
        dw(dw_adamw<2 * FFN, DIM>, 2 * FFN, DIM, P_QKV + P_O, gr.gu, g.dgu, ln2);
        mark("bwd gate_up dW + AdamW (own) 3072x576x2048");
#ifdef PROBE_DW
        probe_dw(2, g.dgu, ln2);
        mark("probe chisel dW core gate_up");
#endif
        put("dln2", g.dln, 0, DIM);
        rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, g.dres_bf, g.dw_part, g.dres, g.dln, res_mid, wf.ln2);
        mark("bwd rmsnorm");
        colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(gln + DIM, g.dw_part);
        mark("bwd norm weight colsum");
        put("dres_mid", g.dres, 1, DIM);
#if defined(CHISEL_DWS) && CHISEL_DWS == 4
        if (!df && l + 1 < LAYERS) dws_pass(s, l + 1);  // the previous layer's AdamW from this o dX GEMM on
#endif
        gemm(s, L.o_dx, g.datt, w.o, g.dres_bf);
        mark("bwd o dX GEMM 2048x576x576");
        dw(dw_adamw<DIM, DIM>, DIM, DIM, P_QKV, gr.o, g.dres_bf, att);
        mark("bwd o dW + AdamW (own) 576x576x2048");
#ifdef PROBE_DW
        probe_dw(1, g.dres_bf, att);
        mark("probe chisel dW core o");
#endif
        put("datt", g.datt, 0, DIM);
#if defined(CHISEL_DWS) && CHISEL_DWS == 2
        if (!df && l + 1 < LAYERS) dws_pass(s, l + 1);  // the previous layer's AdamW beside this attention backward
#endif
#if defined(CHISEL_DWS) && CHISEL_DWS == 3
        if (!df) {  // beside this attention backward: the previous layer's qkv (its slot is then free), this layer's rest
            if (l + 1 < LAYERS) {
                dws_range(s, l + 1, 0, P_QKV);
                CK(cudaEventRecord(dws_ad[(l + 1) % 3], dws_side));
            }
            dws_range(s, l, P_QKV, P_MAT);
        }
#endif
#ifdef CHISEL_DSQ
        dsq_attention(g.part, g.dqkv, g.dsum, qkv, att, g.datt, lse);
        mark("bwd attention dq + dk dv (dS reuse)");
#else
        attention_bwd_dq<<<dim3(NT, HEADS, B), 128, 0, stream>>>(g.dqkv, g.dsum, qkv, att, g.datt, lse);
        mark("bwd attention dq");
#ifdef CHISEL_DKV
        chisel_dkv_launch(g.part, qkv, g.datt, lse, g.dsum);
        mark("bwd attention dk dv (chisel)");
#else
        attention_bwd_dkv<<<dim3(NT, HEADS, B), 128, 0, stream>>>(g.part, qkv, g.datt, lse, g.dsum);
        mark("bwd attention dk dv");
#endif
#endif
        attention_bwd_finish<<<cdiv(N * (HEADS + 2 * KV_HEADS) * HEAD_DIM / 2, 256), 256, 0, stream>>>(g.dqkv, g.part,
                                                                                                       s.rope);
        mark("bwd attention finish");
        put("dqkv_pre", g.dqkv, 0, QKV_DIM);
        gemm(s, L.qkv_dx, g.dln, w.qkv, g.dqkv);
        mark("bwd qkv dX GEMM 2048x576x960");
        dw(dw_adamw<QKV_DIM, DIM>, QKV_DIM, DIM, 0, gr.qkv, g.dqkv, ln1);
        mark("bwd qkv dW + AdamW (own) 960x576x2048");
#ifdef PROBE_DW
        probe_dw(0, g.dqkv, ln1);
        mark("probe chisel dW core qkv");
#endif
        char nm[48];
        snprintf(nm, sizeof(nm), "gmat.%d", l);
        dump(df, nm, g.gmat, 0, 1, P_MAT);
        put("dln1", g.dln, 0, DIM);
        rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, l ? g.dres_bf : nullptr, g.dw_part, g.dres, g.dln, res,
                                                    wf.ln1);
        mark("bwd rmsnorm");
        colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(gln, g.dw_part);
        mark("bwd norm weight colsum");
        put("dres_in", g.dres, 1, DIM);
    }
    embed_bwd_adamw<<<N, DIM / 4, 0, stream>>>(s.master, s.params, s.m, s.v, g.gemb_rows, df ? s.gemb : nullptr, g.dres,
                                               a.io, a.io + IO_SEG, s.step);
    mark("bwd embedding + AdamW (own)");
#ifdef CHISEL_DWS
    if (!df) {  // the side stream's AdamW passes finish inside the step
        CK(cudaEventRecord(dws_join, dws_side));
        CK(cudaStreamWaitEvent(stream, dws_join));
    }
#endif
    CK(cudaGetLastError());
}

// AdamW runs inside backward for the weight matrices and the embedding (dw_adamw, embed_bwd_adamw); norms after it.
static void train_step(State& s, const Lins& L, FILE* df) {
    next_step<<<1, 1, 0, stream>>>(s.step);
    mark("step counter");
    forward(s, L, df);
    backward(s, L, df);
    dump(df, "gemb", s.gemb, 1, 1, P_EMBED);
    dump(df, "gnorm", s.gnorm, 1, 1, G_NORM);
    static_assert(G_NORM % 4 == 0, "adamw_norms works on float4");
    adamw_norms<<<cdiv(G_NORM / 4, 256), 256, 0, stream>>>(s.master, s.params, s.gnorm, s.m, s.v, s.step);
    mark("AdamW norm weights");
    CK(cudaGetLastError());
    dump(df, "master", s.master, 1, 1, P_TOTAL);
    dump(df, "params", s.params, 0, 1, P_TOTAL);
}

static void load_weights(State& s, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open weights");
    int32_t h[64];
    float theta, eps;
    if (fread(h, 4, 64, f) != 64) die("short weights header");
    memcpy(&theta, &h[8], 4);
    memcpy(&eps, &h[9], 4);
    if (h[0] != 0x4854414C || h[1] != 1) die("bad weights magic/version");
    if (h[2] != VOCAB || h[3] != DIM || h[4] != LAYERS || h[5] != HEADS || h[6] != KV_HEADS || h[7] != FFN ||
        theta != ROPE_THETA || eps != RMS_EPS)
        die("weights config does not match models/" MODEL ".h");
    std::vector<float> w(P_TOTAL);
    if (fread(w.data(), 4, P_TOTAL, f) != P_TOTAL || fgetc(f) != EOF) die("weights size mismatch");
    fclose(f);
    CK(cudaMemcpy(s.master, w.data(), P_TOTAL * 4, cudaMemcpyHostToDevice));
    f32_to_bf16<<<cdiv(P_TOTAL, 256), 256, 0, stream>>>(s.params, s.master, P_TOTAL);
    CK(cudaMemsetAsync(s.m, 0, P_TOTAL * 4, stream));
    CK(cudaMemsetAsync(s.v, 0, P_TOTAL * 4, stream));
    CK(cudaMemsetAsync(s.step, 0, sizeof(float), stream));
}

static void init_rope(State& s) {
    // Same fp32 ops as ref/model.py rope_table.
    std::vector<float2> cs(T * HEAD_DIM / 2);
    for (int i = 0; i < HEAD_DIM / 2; i++) {
        float inv = 1.0f / powf(ROPE_THETA, (float)(2 * i) / HEAD_DIM);
        for (int t = 0; t < T; t++) {
            float a = (float)t * inv;
            cs[t * HEAD_DIM / 2 + i] = {cosf(a), sinf(a)};
        }
    }
    CK(cudaMemcpy(s.rope, cs.data(), cs.size() * sizeof(float2), cudaMemcpyHostToDevice));
}

struct Tokens {
    const uint16_t* data;
    size_t n;
};

static Tokens map_tokens(const char* path) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) die("cannot open data");
    void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) die("mmap failed");
    close(fd);
    return {(const uint16_t*)p, (size_t)st.st_size / 2};
}

// Batch i = contiguous span of N+1 tokens, wrapping over whole batches (same as ref/model.py get_batch).
// Also builds embed_bwd segments: positions sorted by (token, position), one segment per distinct token.
static void load_batch(const Tokens& d, long i, int* io, std::pair<int, int>* sorted) {
    const size_t nb = (d.n - 1) / N, s = (i % nb) * N;
    int *tok = io, *tgt = io + N, *perm = io + IO_SEG, *starts = io + IO_SEG + N;
    for (int j = 0; j < N; j++) {
        tok[j] = d.data[s + j];
        tgt[j] = d.data[s + j + 1];
        sorted[j] = {tok[j], j};
    }
    std::sort(sorted, sorted + N);
    int count = 0;
    for (int j = 0; j < N; j++) {
        perm[j] = sorted[j].second;
        if (j == 0 || sorted[j].first != sorted[j - 1].first) starts[count++] = j;
    }
    starts[count] = N;
    starts[N + 1] = count;
    int* segmap = io + IO_SEGMAP;
    memset(segmap, 0xff, VOCAB * sizeof(int));  // -1
    for (int j = 0; j < count; j++) segmap[sorted[starts[j]].first] = j;
}

int main(int argc, char** argv) {
    const char *weights = "data/" MODEL "_init.bin", *data = "data/train.bin", *dump_path = nullptr;
    int steps = 0, evals = 0, group = 20, eager = 0;
    for (int i = 1; i < argc; i++) {
        if (i + 1 == argc) die("missing value");
        if (!strcmp(argv[i], "--weights")) weights = argv[++i];
        else if (!strcmp(argv[i], "--data")) data = argv[++i];
        else if (!strcmp(argv[i], "--dump")) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--steps")) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eval")) evals = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--group")) group = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eager")) eager = atoi(argv[++i]);
        else die("usage: train [--weights f] [--data f] [--steps n | --eval n] [--group n] [--dump f] [--eager 1]");
    }
    if ((steps > 0) == (evals > 0)) die("pass exactly one of --steps or --eval");

    CK(cudaSetDevice(0));
#ifdef CHISEL_DWS
    {  // highest priority for the main stream, lowest for the AdamW side stream
        int least, greatest;
        CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest));
    }
#else
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif
    CKB(cublasLtCreate(&lt));
#ifdef CHISEL_DOWN
    chisel_down_fn = chisel_fn(chisel_down, chisel_down_kernel);
#endif
#ifdef CHISEL_DKV
    chisel_dkv_fn = chisel_fn(chisel_dkv, chisel_dkv_kernel);
#endif
#ifdef CHISEL_DWS
    dws_fn[0] = chisel_fn(chisel_dw_qkv, chisel_dw_qkv_kernel);
    dws_fn[1] = chisel_fn(chisel_dw_o, chisel_dw_o_kernel);
    dws_fn[2] = chisel_fn(chisel_dw_gu, chisel_dw_gu_kernel);
    dws_fn[3] = chisel_fn(chisel_dw_down, chisel_dw_down_kernel);
    {
        int least, greatest;
        CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CK(cudaStreamCreateWithPriority(&dws_side, cudaStreamNonBlocking, least));
    }
    for (int k = 0; k < 3; k++) {
        CK(cudaMalloc(&dws_grad[k], P_MAT * 2));
        CK(cudaEventCreateWithFlags(&dws_dw[k], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&dws_ad[k], cudaEventDisableTiming));
    }
    CK(cudaEventCreateWithFlags(&dws_join, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&dws_go, cudaEventDisableTiming));
#endif
#ifdef CHISEL_DSQ
    {
        int least, greatest;
        CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        for (int i = 0; i < 3; i++) {
            CK(cudaStreamCreateWithPriority(&dsq_s[i], cudaStreamNonBlocking, std::min(least, greatest + i)));
            CK(cudaEventCreateWithFlags(&dsq_g[i], cudaEventDisableTiming));
        }
        CK(cudaEventCreateWithFlags(&dsq_fork, cudaEventDisableTiming));
        CK(cudaMalloc(&dsq_buf, (size_t)2 * 3 * DSQ_TILES * TILE * 2));
    }
#endif
#ifdef CHISEL_DW
    cdw_fn[0] = chisel_fn(chisel_dwa_qkv, chisel_dwa_qkv_kernel);
    cdw_fn[1] = chisel_fn(chisel_dwa_o, chisel_dwa_o_kernel);
    cdw_fn[2] = chisel_fn(chisel_dwa_gu, chisel_dwa_gu_kernel);
    cdw_fn[3] = chisel_fn(chisel_dwa_down, chisel_dwa_down_kernel);
    cdw_fn[4] = chisel_fn(chisel_dwa_head, chisel_dwa_head_kernel);
    CK(cudaMalloc(&chisel_bc, 8));
#endif
#ifdef PROBE_DW
    probe_fn[0] = chisel_fn(chisel_dw_qkv, chisel_dw_qkv_kernel);
    probe_fn[1] = chisel_fn(chisel_dw_o, chisel_dw_o_kernel);
    probe_fn[2] = chisel_fn(chisel_dw_gu, chisel_dw_gu_kernel);
    probe_fn[3] = chisel_fn(chisel_dw_down, chisel_dw_down_kernel);
    probe_fn[4] = chisel_fn(chisel_dw_head, chisel_dw_head_kernel);
    CK(cudaMalloc(&probe_out, (size_t)VOCAB * DIM * 2));
#endif

    State s;
    Arena ar{nullptr, 0};
    carve(ar, s, dump_path != nullptr);
    const size_t arena_bytes = ar.used;
    CK(cudaMalloc(&ar.base, arena_bytes));
    ar.used = 0;
    carve(ar, s, dump_path != nullptr);
    printf("B=%d T=%d params=%zu arena=%.1f MiB\n", B, T, P_TOTAL, arena_bytes / 1048576.0);

    load_weights(s, weights);
    init_rope(s);
    const Lins L = {make_fwd(DIM, DIM),     make_fwd(2 * FFN, DIM), make_fwd(DIM, FFN),     make_fwd(VOCAB, DIM),
                    make_dx(QKV_DIM, DIM), make_dx(DIM, DIM),       make_dx(2 * FFN, DIM), make_dx(VOCAB, DIM)};
    const Tokens d = map_tokens(data);
    int* h_io;
    float* h_loss;
    CK(cudaMallocHost(&h_io, IO_INTS * sizeof(int)));
    CK(cudaMallocHost(&h_loss, group * sizeof(float)));
    std::vector<std::pair<int, int>> sorted(N);
    cudaEvent_t copied;
    CK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    CK(cudaStreamSynchronize(stream));

    const bool train = steps > 0;
#ifdef STEP_TIMES
    std::vector<std::pair<const char*, float>> op_ms;
    int timed_steps = 0;
#endif
    cudaGraphExec_t graph = nullptr;
    const int total = train ? steps : evals;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0, first = 0; i < total; i++) {
        CK(cudaEventSynchronize(copied));  // the previous upload has left the pinned buffer
        load_batch(d, i, h_io, sorted.data());
        CK(cudaMemcpyAsync(s.a.io, h_io, IO_INTS * sizeof(int), cudaMemcpyHostToDevice, stream));
        CK(cudaEventRecord(copied, stream));
        FILE* df = i == 0 && dump_path ? fopen(dump_path, "wb") : nullptr;
        if (!train) {
            forward(s, L, df);
        } else if (df || eager || STEP_TIMED) {  // dumps synchronize, so this step runs eagerly
#ifdef STEP_TIMES
            mark_i = 0;
            mark("start");
#endif
            train_step(s, L, df);
#ifdef STEP_TIMES
            if (i >= 3) {  // after warm-up
                CK(cudaEventSynchronize(marks[mark_i - 1].ev));
                for (size_t k = 1; k < mark_i; k++) {
                    float ms;
                    CK(cudaEventElapsedTime(&ms, marks[k - 1].ev, marks[k].ev));
                    op_ms.push_back({marks[k].name, ms});
                }
                timed_steps++;
            }
#endif
        } else {  // the whole step is one CUDA graph, captured on first use
            if (!graph) {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                train_step(s, L, nullptr);
                CK(cudaStreamEndCapture(stream, &g));
#if defined(CHISEL_DWS) || defined(CHISEL_DSQ)
                CK(cudaGraphInstantiate(&graph, g, cudaGraphInstantiateFlagUseNodePriority));
#else
                CK(cudaGraphInstantiate(&graph, g, 0));
#endif
                CK(cudaGraphDestroy(g));
            }
            CK(cudaGraphLaunch(graph, stream));
        }
        CK(cudaMemcpyAsync(h_loss + i % group, s.a.loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
        if (df) fclose(df);
        if ((i + 1) % group == 0 || i + 1 == total) {
            CK(cudaStreamSynchronize(stream));
            auto now = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - t0).count();
            for (int k = first; k <= i; k++) printf("%s %d loss %.6f\n", train ? "step" : "batch", k, h_loss[k % group]);
            printf("group %d steps %d-%d ms %.1f tok/s %.0f\n", i / group, first, i, ms, (i + 1 - first) * N / ms * 1e3);
            fflush(stdout);
            first = i + 1;
            t0 = now;
        }
    }
#ifdef STEP_TIMES
    {  // per-op GPU ms per step, summed over layers, largest first
        std::vector<std::pair<std::string, std::pair<double, int>>> tot;
        for (auto& op : op_ms) {
            auto it = std::find_if(tot.begin(), tot.end(), [&](auto& x) { return x.first == op.first; });
            if (it == tot.end()) tot.push_back({op.first, {op.second, 1}});
            else it->second.first += op.second, it->second.second++;
        }
        std::sort(tot.begin(), tot.end(), [](auto& x, auto& y) { return x.second.first > y.second.first; });
        double all = 0;
        for (auto& x : tot) all += x.second.first;
        printf("step_times steps %d total_ms_per_step %.3f\n", timed_steps, all / timed_steps);
        for (auto& x : tot)
            printf("step_time | %s | %.3f | %d\n", x.first.c_str(), x.second.first / timed_steps,
                   x.second.second / timed_steps);
    }
#endif
    return 0;
}
