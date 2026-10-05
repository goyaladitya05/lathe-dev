// lathe: SmolLM2 training engine, model and GPU fixed at compile time: forward, backward, fused AdamW.
//   make MODEL=smol135 BATCH=1 SEQ=2048
//   build/train_smol135_b1_t2048 --steps 1000 --data data/train.bin   train from --weights (default seed-0 init)
//   build/train_smol135_b1_t2048 --eval 8 --data data/val.bin         forward-only loss on 8 batches
//   --dump PATH writes step 0 tensors (layer 0, last layer, grads, updated weights) for ref/test.py.
//   Training steps replay one captured CUDA graph; --eager 1 launches kernels directly instead.
//   make ... STEP_TIMES=1 builds a profiling variant: eager steps, a CUDA event after every launch, ms per op at the end.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include MODEL_H  // models/<MODEL>.h, from the Makefile
#include "targets/sm89.h"

constexpr int B = BATCH, T = SEQ, N = B * T;  // kernels.cuh specializes on these
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

// Grads: layer matrix grads are bf16 like PyTorch's dW, in a ring of per-layer buffers until their AdamW pass (see
// backward); most embedding rows never leave the lm_head dW GEMM, which applies AdamW directly. Norm weight grads are
// fp32 in gnorm = [LAYERS x (ln1, ln2)][final norm].
constexpr size_t P_MAT = P_QKV + P_O + P_GU + P_DOWN;
#ifndef ADAM8
#define ADAM8 0  // 1 (make ADAM8=1): 8-bit AdamW moments for the layer matrices (adamw_stream8, BENCH.md)
#endif
constexpr size_t G_NORM = LAYERS * 2 * DIM + DIM;

#include "kernels.cuh"
static_assert(!ADAM8 || (P_MAT % Q8_BLOCK == 0 && P_QKV % Q8_BLOCK == 0), "whole 8-bit blocks per layer");

// Layer dW cores in hand-scheduled SASS from chisel (github goyaladitya05/chisel, gen/lathe_export.py): one header
// per model and token count (CHISEL_DW_H, from the Makefile) with the cubins of the four matrix shapes. Shapes without
// one use dw_bf16; the bits are the same either way (chisel e037).
#if __has_include(CHISEL_DW_H)
#include CHISEL_DW_H
#define HAVE_CHISEL_DW 1
static_assert(chisel_dw_qkv_m == QKV_DIM && chisel_dw_qkv_n == DIM && chisel_dw_o_m == DIM && chisel_dw_o_n == DIM &&
                  chisel_dw_gu_m == 2 * FFN && chisel_dw_gu_n == DIM && chisel_dw_down_m == DIM && chisel_dw_down_n == FFN &&
                  chisel_dw_qkv_k == N && chisel_dw_o_k == N && chisel_dw_gu_k == N && chisel_dw_down_k == N,
              "chisel header built for another shape");
#endif

#define CK(x) ck_cuda((x), #x, __LINE__)
#define CKB(x) ck_blas((x), #x, __LINE__)
#define CKD(x) ck_drv((x), #x, __LINE__)
static void ck_cuda(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "train.cu:%d %s: %s\n", line, what, cudaGetErrorString(e));
        exit(1);
    }
}
#ifdef HAVE_CHISEL_DW
static void ck_drv(CUresult r, const char* what, int line) {
    if (r != CUDA_SUCCESS) {
        const char* s = "?";
        cuGetErrorString(r, &s);
        fprintf(stderr, "train.cu:%d %s: %s\n", line, what, s);
        exit(1);
    }
}
#endif
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
    bf16* logits;     // [N][VOCAB], overwritten with dlogits by cross_entropy; holds gring and ds when dead (RING)
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
    bf16* gring;     // [RING][P_MAT] matrix grads of the last RING layers, until their AdamW pass (in a.logits)
    bf16* ds;        // [2][DS_GROUP] bf16 dS tiles of two attention groups, attention_backward (in a.logits)
};
constexpr int RING = 3;

// gring and ds live in the logits buffer when it is large enough (IN_LOGITS). Lifetime of logits in a step: written by
// the lm_head forward and overwritten in place with dlogits by cross_entropy (main stream), last read by head_dw_adamw
// (main), dead until the next step's lm_head forward. In that window: gring is written by the layer dW cores (main,
// issued after head_dw_adamw) and read by the AdamW passes (side stream, joined into main at the end of backward); ds
// is written and read by attention_backward (attention streams, forked from main and joined within it). alias_check
// enforces this issue order on the host whenever a step is issued, eager or captured.
constexpr size_t RING_N = RING * P_MAT, DS_N = std::min(KV_HEADS, 2) * DS_GROUP;
constexpr bool IN_LOGITS = RING_N + DS_N <= (size_t)N * VOCAB;
static_assert(RING_N * sizeof(bf16) % 256 == 0, "ds stays 256B-aligned behind the ring");
static bool logits_live, side_open;  // logits between lm_head forward and head_dw_adamw; AdamW passes not yet joined
static void alias_check(bool ok, const char* what) {
    if (IN_LOGITS && !ok) die(what);
}

struct State {
    float* master;  // [P_TOTAL] fp32 master weights
    bf16* params;   // [P_TOTAL] bf16 working copy (matmul weights)
    float* gnorm;   // [G_NORM] norm weight grads
    float* gemb;    // [P_EMBED] full fp32 embedding grad, dumps only (nullptr otherwise)
    float *m, *v;   // [P_TOTAL] AdamW moments; ADAM8: [P_EMBED], the embedding's only
    float *mn, *vn;  // ADAM8: [G_NORM] the norm weights' moments (fp32 mode: inside m, v)
    unsigned char *qm, *qu;  // ADAM8: [LAYERS][P_MAT] E4M3 codes of the layer matrices' m and sqrt(v)
    float *sm, *su;          // ADAM8: [LAYERS][P_MAT / Q8_BLOCK] their block scales
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
    s.m = ar.take<float>(ADAM8 ? P_EMBED : P_TOTAL);
    s.v = ar.take<float>(ADAM8 ? P_EMBED : P_TOTAL);
    if (ADAM8) {
        s.mn = ar.take<float>(G_NORM);
        s.vn = ar.take<float>(G_NORM);
        s.qm = ar.take<unsigned char>(LAYERS * P_MAT);
        s.qu = ar.take<unsigned char>(LAYERS * P_MAT);
        s.sm = ar.take<float>(LAYERS * P_MAT / Q8_BLOCK);
        s.su = ar.take<float>(LAYERS * P_MAT / Q8_BLOCK);
    }
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
    g.gring = IN_LOGITS ? a.logits : ar.take<bf16>(RING_N);
    g.ds = IN_LOGITS ? a.logits + RING_N : ar.take<bf16>(DS_N);
}

// The main stream runs at the highest priority. Beside it: one stream per attention group (attention_backward, earlier
// groups higher) and the AdamW side stream at the lowest priority (backward). All fork from and join back into the
// main stream inside a step, so a step is still one captured graph.
static cudaStream_t stream, side, att_stream[KV_HEADS];
static cudaEvent_t ev_fork, ev_join, ev_ring[RING], ev_att[KV_HEADS];
static cublasLtHandle_t lt;
#ifdef HAVE_CHISEL_DW
static CUfunction chisel_dw[4];
#endif

// Per-op GPU time (make ... STEP_TIMES=1): an event after every launch on the main stream, eager steps, a table at the
// end; the AdamW passes on the side stream overlap other ops and are timed on their own.
#ifdef STEP_TIMES
constexpr bool STEP_TIMED = true;
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
static std::vector<std::pair<cudaEvent_t, cudaEvent_t>> side_marks;
static size_t side_i;
static void side_mark(bool end) {
    if (!end && side_i == side_marks.size()) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        side_marks.push_back({a, b});
    }
    CK(cudaEventRecord(end ? side_marks[side_i++].second : side_marks[side_i].first, side));
}
#else
constexpr bool STEP_TIMED = false;
static void mark(const char*) {}
static void side_mark(bool) {}
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
// the down dX and every dW are own kernels (qkv_rope, down_dx_swiglu, dw_bf16 / chisel's, head_dw_adamw).
// Forward y[N][out] = x W^T (A = W, B = x); input grad dx[N][in] = dy W (A = W, B = dy).
static Gemm make_fwd(int out, int in) { return make_gemm(true, out, N, in); }

// dW[out][in] = dy^T x rounded to bf16 into g, on the main stream.
enum { DW_QKV, DW_O, DW_GU, DW_DOWN };
template <int W, int OUT, int IN>
static void dw_core(bf16* g, const bf16* dy, const bf16* x) {
    alias_check(!logits_live, "logits aliasing: a dW core writes gring while logits are live");
#ifdef HAVE_CHISEL_DW
    static const unsigned* grid[4] = {chisel_dw_qkv_grid, chisel_dw_o_grid, chisel_dw_gu_grid, chisel_dw_down_grid};
    static const unsigned block[4] = {chisel_dw_qkv_block, chisel_dw_o_block, chisel_dw_gu_block, chisel_dw_down_block};
    void* args[] = {&g, &dy, &x};
    CKD(cuLaunchKernel(chisel_dw[W], grid[W][0], grid[W][1], grid[W][2], block[W], 1, 1, 0, (CUstream)stream, args,
                       nullptr));
#else
    dw_bf16<OUT, IN><<<dim3(IN / 64, OUT / 64), 128, 0, stream>>>(g, dy, x);
#endif
}
static Gemm make_dx(int out, int in) { return make_gemm(false, in, N, out); }
struct Lins {
    Gemm o, gu, down, head;             // forward
    Gemm qkv_dx, o_dx, gu_dx, head_dx;  // input grads
};

// Test hook: [name[48], dtype i32 (0 bf16, 1 f32, 2 i32, 3 u8), pad i32, rows i64, cols i64] + data. Read by ref/test.py.
static void dump(FILE* f, const char* name, const void* p, int dtype, size_t rows, size_t cols) {
    if (!f) return;
    size_t bytes = rows * cols * (dtype == 0 ? 2 : dtype == 3 ? 1 : 4);
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
        mark("fwd qkv GEMM + RoPE");
        put("qkv", qkv, 0, QKV_DIM);
        attention_fwd<<<att_grid(HEADS), 128, 0, stream>>>(att, lse, qkv);
        mark("fwd attention");
        put("att", att, 0, DIM);
        gemm(s, L.o, a.branch, w.o, att);
        mark("fwd o GEMM");
        put("ao", a.branch, 0, DIM);
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(ln2, res_mid, res, a.branch, wf.ln2);
        mark("fwd rmsnorm");
        put("res_mid", res_mid, 1, DIM);
        put("ln2", ln2, 0, DIM);
        gemm(s, L.gu, gu, w.gu, ln2);
        mark("fwd gate_up GEMM");
        put("gu", gu, 0, 2 * FFN);
        swiglu<<<cdiv(N * FFN / 8, 256), 256, 0, stream>>>(act, gu);
        mark("fwd swiglu");
        put("act", act, 0, FFN);
        gemm(s, L.down, a.branch, w.down, act);
        mark("fwd down GEMM");
        put("mo", a.branch, 0, DIM);
        // Next layer's input norm, or the final norm.
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(last ? a.lnf : a.ln1 + (l + 1) * ND, res + ND, res_mid, a.branch,
                                                last ? s.master + P_TOTAL - DIM : layer(s.master, l + 1).ln1);
        mark("fwd rmsnorm");
        put("res_out", res + ND, 1, DIM);
    }
    alias_check(!side_open, "logits aliasing: lm_head forward before the AdamW passes joined");
    logits_live = true;
    gemm(s, L.head, a.logits, s.params, a.lnf);  // tied embedding
    mark("fwd head GEMM");
    dump(df, "lnf", a.lnf, 0, N, DIM);
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

// Attention backward of one layer: Dsum, then per attention group (on its own stream, earlier groups at higher
// priority so later groups fill their tails) dK/dV storing dS and dq as a GEMM over it. Groups alternate between two
// dS buffers; a group waits until the one two before it has read its buffer.
static void attention_backward(float* part, bf16* dqkv, float* dsum, bf16* ds, const bf16* qkv, const bf16* out,
                               const bf16* dout, const float* lse) {
    alias_check(!logits_live, "logits aliasing: attention backward writes ds while logits are live");
    attention_dsum<<<dim3(NT, HEADS, B), 128, 0, stream>>>(dsum, out, dout);
    CK(cudaEventRecord(ev_fork, stream));
    for (int k = 0; k < KV_HEADS; k++) {
        bf16* buf = ds + (size_t)(k % 2) * DS_GROUP;
        CK(cudaStreamWaitEvent(att_stream[k], ev_fork));
        if (k >= 2) CK(cudaStreamWaitEvent(att_stream[k], ev_att[k - 2]));
        attention_bwd_dkv<<<att_grid(GROUP), 128, 0, att_stream[k]>>>(part, buf, qkv, dout, lse, dsum, k * GROUP);
        attention_bwd_dq<<<att_grid(GROUP), 128, 0, att_stream[k]>>>(dqkv, buf, qkv, k * GROUP);
        CK(cudaEventRecord(ev_att[k], att_stream[k]));
    }
    for (int k = 0; k < KV_HEADS; k++) CK(cudaStreamWaitEvent(stream, ev_att[k]));
}

// Layer weight grads go to a ring of RING per-layer buffers (dW cores on the main stream); each layer's AdamW is
// adamw_stream on the side stream, started with a kernel of the next layer's backward (ADAMW_AT, models/*.h: a window
// of compute-bound kernels; wider passes slow what they run beside, narrower ones spill into the memory-bound kernels
// after it: chisel e048-e051, e054), and joined at the end of backward. The ring slot a layer writes was last read by
// the pass of the layer RING above.
static_assert(ADAMW_AT >= 0 && ADAMW_AT <= 8 && ADAMW_BLOCKS > 0, "ADAMW_AT: 0..8, see models/*.h");
// 8-bit state builds have their own start and width (the pass moves 16.25 instead of 28 bytes per parameter).
constexpr int PASS_AT = ADAM8 ? ADAMW_AT8 : ADAMW_AT, PASS_BLOCKS = ADAM8 ? ADAMW_BLOCKS8 : ADAMW_BLOCKS;
static_assert(PASS_AT >= 0 && PASS_AT <= 8 && PASS_BLOCKS > 0, "ADAMW_AT8: 0..8, see models/*.h");
static void backward(State& s, const Lins& L, FILE* df) {
    Acts& a = s.a;
    Bwd& g = s.g;
    const size_t ND = (size_t)N * DIM;
    // AdamW of parameters [b, e) of layer l's matrices, beside the kernels issued after this point
    auto adamw_pass = [&](int l, size_t b, size_t e) {
        const size_t o = P_EMBED + l * P_LAYER + b;
        side_open = true;
        CK(cudaEventRecord(ev_fork, stream));
        CK(cudaStreamWaitEvent(side, ev_fork));
        side_mark(false);
        const bf16* gr = g.gring + (size_t)(l % RING) * P_MAT + b;
        if constexpr (ADAM8) {
            const size_t q = (size_t)l * P_MAT + b;  // index into the 8-bit state
            adamw_stream8<<<PASS_BLOCKS, 256, 0, side>>>(s.master + o, s.qm + q, s.qu + q, s.sm + q / Q8_BLOCK,
                                                          s.su + q / Q8_BLOCK, s.params + o, gr, s.step, e - b);
        } else {
            adamw_stream<<<PASS_BLOCKS, 256, 0, side>>>(s.master + o, s.m + o, s.v + o, s.params + o, gr, s.step, e - b);
        }
        side_mark(true);
        CK(cudaEventRecord(ev_ring[l % RING], side));
    };
    gemm(s, L.head_dx, g.dln, s.params, a.logits);
    mark("bwd head dX GEMM");
    // tied embedding: head dW with AdamW fused for rows not in the batch; the rest finish in embed_bwd_adamw
    head_dw_adamw<<<dim3(DIM / 64, VOCAB / 64), 128, 0, stream>>>(s.master, s.params, s.m, s.v, a.logits, a.lnf, s.step,
                                                                  g.gemb_rows, df ? s.gemb : nullptr, a.io + IO_SEGMAP);
    mark("bwd head dW + AdamW");
    logits_live = false;  // head_dw_adamw was the last reader
    dump(df, "dlnf", g.dln, 0, N, DIM);
    rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, g.dres_bf, g.dw_part, nullptr, g.dln, a.res + LAYERS * ND,
                                                s.master + P_TOTAL - DIM);
    mark("bwd rmsnorm");
    colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(s.gnorm + G_NORM - DIM, g.dw_part);
    mark("bwd norm weight colsum");
    for (int l = LAYERS - 1; l >= 0; l--) {
        const LayerP<bf16> w = layer(s.params, l);
        const LayerP<float> wf = layer(s.master, l);
        bf16* gl = g.gring + (size_t)(l % RING) * P_MAT;  // this layer's matrix grads: qkv | o | gu | down
        float* gln = s.gnorm + l * 2 * DIM;
        float *res = a.res + l * ND, *res_mid = a.res_mid + l * ND;
        bf16 *ln1 = a.ln1 + l * ND, *ln2 = a.ln2 + l * ND, *att = a.att + l * ND;
        bf16 *qkv = a.qkv + (size_t)l * N * QKV_DIM, *gu = a.gu + (size_t)l * N * 2 * FFN;
        bf16* act = a.act + (size_t)l * N * FFN;
        float* lse = a.lse + (size_t)l * N * HEADS;
        const LayerDump put{df, l};
        auto at = [&](int k) {  // before kernel k of this layer's backward (numbering in models/*.h)
            if (k == PASS_AT && l + 1 < LAYERS) adamw_pass(l + 1, 0, P_MAT);
            if (ADAMW_SPLIT_LAST && k == 6 && l == 0) adamw_pass(0, P_QKV, P_MAT);
        };

        put("dres_out", g.dres, 1, DIM);
        at(0);
        down_dx_swiglu<<<dim3(FFN / 64, N / GBM), GBM / GWM * 32, 0, stream>>>(g.dgu, g.dres_bf, w.down, gu);
        mark("bwd down dX GEMM + SwiGLU bwd");
        if (l + RING < LAYERS) CK(cudaStreamWaitEvent(stream, ev_ring[l % RING]));  // the slot's previous pass is done
        at(1);
        dw_core<DW_DOWN, DIM, FFN>(gl + P_QKV + P_O + P_GU, g.dres_bf, act);
        mark("bwd down dW");
        put("dgu", g.dgu, 0, 2 * FFN);
        at(2);
        gemm(s, L.gu_dx, g.dln, w.gu, g.dgu);
        mark("bwd gate_up dX GEMM");
        at(3);
        dw_core<DW_GU, 2 * FFN, DIM>(gl + P_QKV + P_O, g.dgu, ln2);
        mark("bwd gate_up dW");
        put("dln2", g.dln, 0, DIM);
        rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, g.dres_bf, g.dw_part, g.dres, g.dln, res_mid, wf.ln2);
        mark("bwd rmsnorm");
        colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(gln + DIM, g.dw_part);
        mark("bwd norm weight colsum");
        put("dres_mid", g.dres, 1, DIM);
        at(4);
        gemm(s, L.o_dx, g.datt, w.o, g.dres_bf);
        mark("bwd o dX GEMM");
        at(5);
        dw_core<DW_O, DIM, DIM>(gl + P_QKV, g.dres_bf, att);
        mark("bwd o dW");
        put("datt", g.datt, 0, DIM);
        at(6);
        attention_backward(g.part, g.dqkv, g.dsum, g.ds, qkv, att, g.datt, lse);
        mark("bwd attention (Dsum, dK/dV + dS, dq from dS)");
        attention_bwd_finish<<<cdiv(N * (HEADS + 2 * KV_HEADS) * HEAD_DIM / 2, 256), 256, 0, stream>>>(g.dqkv, g.part,
                                                                                                       s.rope);
        mark("bwd attention finish");
        put("dqkv_pre", g.dqkv, 0, QKV_DIM);
        at(7);
        gemm(s, L.qkv_dx, g.dln, w.qkv, g.dqkv);
        mark("bwd qkv dX GEMM");
        at(8);
        dw_core<DW_QKV, QKV_DIM, DIM>(gl, g.dqkv, ln1);
        mark("bwd qkv dW");
        char nm[48];
        snprintf(nm, sizeof(nm), "gmat.%d", l);
        dump(df, nm, gl, 0, 1, P_MAT);
        if (l == 0) adamw_pass(0, 0, ADAMW_SPLIT_LAST ? P_QKV : P_MAT);
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
    mark("bwd embedding + AdamW");
    CK(cudaEventRecord(ev_join, side));  // the AdamW passes finish inside the step
    CK(cudaStreamWaitEvent(stream, ev_join));
    side_open = false;
    CK(cudaGetLastError());
}

// AdamW runs inside backward for the weight matrices and the embedding (adamw_stream, head_dw_adamw, embed_bwd_adamw);
// norms after it.
static void train_step(State& s, const Lins& L, FILE* df) {
    next_step<<<1, 1, 0, stream>>>(s.step);
    mark("step counter");
    forward(s, L, df);
    backward(s, L, df);
    dump(df, "gemb", s.gemb, 1, 1, P_EMBED);
    dump(df, "gnorm", s.gnorm, 1, 1, G_NORM);
    static_assert(G_NORM % 4 == 0, "adamw_norms works on float4");
    adamw_norms<ADAM8><<<cdiv(G_NORM / 4, 256), 256, 0, stream>>>(s.master, s.params, s.gnorm, ADAM8 ? s.mn : s.m,
                                                                  ADAM8 ? s.vn : s.v, s.step);
    mark("AdamW norm weights");
    CK(cudaGetLastError());
    dump(df, "master", s.master, 1, 1, P_TOTAL);
    dump(df, "params", s.params, 0, 1, P_TOTAL);
    if (ADAM8)
        for (int l : {0, LAYERS - 1}) {  // the 8-bit state after the step, checked byte for byte by ref/test.py
            char nm[48];
            const size_t q = (size_t)l * P_MAT;
            for (auto [name, qs, ss] : {std::tuple{"m", s.qm, s.sm}, std::tuple{"u", s.qu, s.su}}) {
                snprintf(nm, sizeof(nm), "q%s.%d", name, l);
                dump(df, nm, qs + q, 3, 1, P_MAT);
                snprintf(nm, sizeof(nm), "s%s.%d", name, l);
                dump(df, nm, ss + q / Q8_BLOCK, 1, 1, P_MAT / Q8_BLOCK);
            }
        }
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
    CK(cudaMemsetAsync(s.m, 0, (ADAM8 ? P_EMBED : P_TOTAL) * 4, stream));
    CK(cudaMemsetAsync(s.v, 0, (ADAM8 ? P_EMBED : P_TOTAL) * 4, stream));
    if (ADAM8) {  // all codes and scales 0: m = v = 0 exactly
        CK(cudaMemsetAsync(s.mn, 0, G_NORM * 4, stream));
        CK(cudaMemsetAsync(s.vn, 0, G_NORM * 4, stream));
        CK(cudaMemsetAsync(s.qm, 0, LAYERS * P_MAT, stream));
        CK(cudaMemsetAsync(s.qu, 0, LAYERS * P_MAT, stream));
        CK(cudaMemsetAsync(s.sm, 0, LAYERS * P_MAT / Q8_BLOCK * 4, stream));
        CK(cudaMemsetAsync(s.su, 0, LAYERS * P_MAT / Q8_BLOCK * 4, stream));
    }
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
    int least, greatest;
    CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest));
    CK(cudaStreamCreateWithPriority(&side, cudaStreamNonBlocking, least));
    for (int k = 0; k < KV_HEADS; k++) {
        CK(cudaStreamCreateWithPriority(&att_stream[k], cudaStreamNonBlocking, std::min(least, greatest + k)));
        CK(cudaEventCreateWithFlags(&ev_att[k], cudaEventDisableTiming));
    }
    for (auto& e : ev_ring) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
    CKB(cublasLtCreate(&lt));
#ifdef HAVE_CHISEL_DW
    {  // after the runtime made the context current
        const unsigned char* image[4] = {chisel_dw_qkv, chisel_dw_o, chisel_dw_gu, chisel_dw_down};
        const char* name[4] = {chisel_dw_qkv_kernel, chisel_dw_o_kernel, chisel_dw_gu_kernel, chisel_dw_down_kernel};
        for (int k = 0; k < 4; k++) {
            CUmodule m;
            CKD(cuModuleLoadData(&m, image[k]));
            CKD(cuModuleGetFunction(&chisel_dw[k], m, name[k]));
        }
    }
#endif

    State s;
    Arena ar{nullptr, 0};
    carve(ar, s, dump_path != nullptr);
    const size_t arena_bytes = ar.used;
    CK(cudaMalloc(&ar.base, arena_bytes));
    ar.used = 0;
    carve(ar, s, dump_path != nullptr);
    printf("B=%d T=%d params=%zu arena=%.1f MiB adamw_state=%s\n", B, T, P_TOTAL, arena_bytes / 1048576.0,
           ADAM8 ? "8bit" : "fp32");

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
    double side_ms = 0;
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
            mark_i = side_i = 0;
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
                for (size_t k = 0; k < side_i; k++) {
                    float ms;
                    CK(cudaEventElapsedTime(&ms, side_marks[k].first, side_marks[k].second));
                    side_ms += ms;
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
                CK(cudaGraphInstantiate(&graph, g, cudaGraphInstantiateFlagUseNodePriority));
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
    if (timed_steps) {  // per-op GPU ms per step, summed over layers, largest first
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
        printf("step_time_side | AdamW passes (side stream, overlapping the ops above) | %.3f\n", side_ms / timed_steps);
    }
#endif
    return 0;
}
