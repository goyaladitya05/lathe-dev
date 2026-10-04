// lathe: SmolLM2-135M training engine. M2: forward, backward, fused AdamW.
//   make BATCH=1 SEQ=2048
//   build/train_b1_t2048 --steps 1000 --data data/train.bin   train from --weights (default seed-0 init)
//   build/train_b1_t2048 --eval 8 --data data/val.bin         forward-only loss on 8 batches
//   --dump PATH writes step 0 tensors (layer 0, last layer, grads, updated weights) for ref/test.py.
//   Training steps replay one captured CUDA graph; --eager 1 launches kernels directly instead.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "models/smol135.h"
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

// Grads: matrices in bf16 (PyTorch also rounds dW to bf16 before its fp32 .grad), per layer [qkv | o | gu | down];
// embedding and norm weights in fp32 (PyTorch sums those in fp32): [embedding][LAYERS x (ln1, ln2)][final norm].
constexpr size_t P_MAT = P_QKV + P_O + P_GU + P_DOWN;
constexpr size_t G_VEC = P_EMBED + LAYERS * 2 * DIM + DIM;

#include "kernels.cuh"

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

// Per-step host data: tokens, targets, embedding segments [perm N][starts N+1][count], bitmask of batch tokens.
constexpr int IO_SEG = 2 * N, IO_TOUCHED = 4 * N + 2, IO_INTS = IO_TOUCHED + VOCAB / 32;

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
    bf16* gmat;      // [P_MAT] one layer's matrix grads, written only when dumping (they never leave registers)
};

struct State {
    float* master;  // [P_TOTAL] fp32 master weights
    bf16* params;   // [P_TOTAL] bf16 working copy (matmul weights)
    float* gvec;    // [G_VEC] embedding and norm grads
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

static void carve(Arena& ar, State& s) {
    const size_t ND = (size_t)N * DIM;
    s.master = ar.take<float>(P_TOTAL);
    s.params = ar.take<bf16>(P_TOTAL);
    s.gvec = ar.take<float>(G_VEC);
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
    g.gmat = ar.take<bf16>(P_MAT);
}

static cudaStream_t stream;
static cublasLtHandle_t lt;

// Column-major D[m x n] = op(A)[m x k] op(B)[k x n], bf16 inputs, fp32 accumulate. Built once per shape.
struct Gemm {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a, b, d;
    cublasLtMatmulAlgo_t algo;
};

static Gemm make_gemm(bool ta, bool tb, int m, int n, int k, cudaDataType_t dtype) {
    Gemm g;
    CKB(cublasLtMatmulDescCreate(&g.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opa = ta ? CUBLAS_OP_T : CUBLAS_OP_N, opb = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    CKB(cublasLtMatmulDescSetAttribute(g.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)));
    CKB(cublasLtMatmulDescSetAttribute(g.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)));
    CKB(cublasLtMatrixLayoutCreate(&g.a, CUDA_R_16BF, ta ? k : m, ta ? m : k, ta ? k : m));
    CKB(cublasLtMatrixLayoutCreate(&g.b, CUDA_R_16BF, tb ? n : k, tb ? k : n, tb ? n : k));
    CKB(cublasLtMatrixLayoutCreate(&g.d, dtype, m, n, m));
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

// Linear y = x W^T, W [out][in] row-major, x/y [N][features] row-major.
struct Lin {
    Gemm fwd, dx, dw;
};
static Lin make_lin(int out, int in, cudaDataType_t dw_type = CUDA_R_16BF) {
    return {make_gemm(true, false, out, N, in, CUDA_R_16BF),   // y[N][out] = x W^T      (A = W, B = x)
            make_gemm(false, false, in, N, out, CUDA_R_16BF),  // dx[N][in] = dy W       (A = W, B = dy)
            make_gemm(false, true, in, out, N, dw_type)};      // dW[out][in] = dy^T x   (A = x, B = dy)
}
struct Lins {
    Lin qkv, o, gu, down, head;
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
    rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(a.ln1, nullptr, a.res, nullptr, layer(s.master, 0).ln1);
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
        put("qkv", qkv, 0, QKV_DIM);
        attention_fwd<<<dim3(NT, HEADS, B), 128, 0, stream>>>(att, lse, qkv);
        put("att", att, 0, DIM);
        gemm(s, L.o.fwd, a.branch, w.o, att);
        put("ao", a.branch, 0, DIM);
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(ln2, res_mid, res, a.branch, wf.ln2);
        put("res_mid", res_mid, 1, DIM);
        put("ln2", ln2, 0, DIM);
        gemm(s, L.gu.fwd, gu, w.gu, ln2);
        put("gu", gu, 0, 2 * FFN);
        swiglu<<<cdiv(N * FFN / 8, 256), 256, 0, stream>>>(act, gu);
        put("act", act, 0, FFN);
        gemm(s, L.down.fwd, a.branch, w.down, act);
        put("mo", a.branch, 0, DIM);
        // Next layer's input norm, or the final norm.
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(last ? a.lnf : a.ln1 + (l + 1) * ND, res + ND, res_mid, a.branch,
                                                last ? s.master + P_TOTAL - DIM : layer(s.master, l + 1).ln1);
        put("res_out", res + ND, 1, DIM);
    }
    gemm(s, L.head.fwd, a.logits, s.params, a.lnf);  // tied embedding
    dump(df, "lnf", a.lnf, 0, N, DIM);
    dump(df, "logits", a.logits, 0, N, VOCAB);
    cross_entropy<<<N, CE_THREADS, 0, stream>>>(a.losses, a.logits, tokens + N);
    mean_loss<<<1, 1024, 0, stream>>>(a.loss, a.losses);
    CK(cudaGetLastError());
    dump(df, "dlogits", a.logits, 0, N, VOCAB);
    dump(df, "losses", a.losses, 1, N, 1);
    dump(df, "tokens", a.io, 2, 2, N);
}

static void backward(State& s, const Lins& L, FILE* df) {
    Acts& a = s.a;
    Bwd& g = s.g;
    const size_t ND = (size_t)N * DIM;
    gemm(s, L.head.dx, g.dln, s.params, a.logits);
    {  // tied embedding: head dW with AdamW fused for rows not in the batch; the rest finish in embed_bwd_adamw
        constexpr auto k = dw_adamw<VOCAB, DIM>;
        k<<<dim3(DIM / 64, VOCAB / 64), 128, 0, stream>>>(s.master, s.params, s.m, s.v, df ? g.gmat : nullptr, a.logits,
                                                          a.lnf, s.step, s.gvec, (const uint32_t*)(a.io + IO_TOUCHED));
    }
    dump(df, "dlnf", g.dln, 0, N, DIM);
    rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, g.dres_bf, g.dw_part, nullptr, g.dln, a.res + LAYERS * ND,
                                                s.master + P_TOTAL - DIM);
    colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(s.gvec + G_VEC - DIM, g.dw_part);
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
            kernel<<<dim3(in / 64, out / 64), 128, 0, stream>>>(s.master + o, s.params + o, s.m + o, s.v + o, gout, dy, x,
                                                               s.step, nullptr, nullptr);
        };
        float* gln = s.gvec + P_EMBED + l * 2 * DIM;
        float *res = a.res + l * ND, *res_mid = a.res_mid + l * ND;
        bf16 *ln1 = a.ln1 + l * ND, *ln2 = a.ln2 + l * ND, *att = a.att + l * ND;
        bf16 *qkv = a.qkv + (size_t)l * N * QKV_DIM, *gu = a.gu + (size_t)l * N * 2 * FFN;
        bf16* act = a.act + (size_t)l * N * FFN;
        float* lse = a.lse + (size_t)l * N * HEADS;
        const LayerDump put{df, l};

        put("dres_out", g.dres, 1, DIM);
        down_dx_swiglu<<<dim3(FFN / 64, N / GBM), GBM / GWM * 32, 0, stream>>>(g.dgu, g.dres_bf, w.down, gu);
        dw(dw_adamw<DIM, FFN>, DIM, FFN, P_QKV + P_O + P_GU, gr.down, g.dres_bf, act);
        put("dgu", g.dgu, 0, 2 * FFN);
        gemm(s, L.gu.dx, g.dln, w.gu, g.dgu);
        dw(dw_adamw<2 * FFN, DIM>, 2 * FFN, DIM, P_QKV + P_O, gr.gu, g.dgu, ln2);
        put("dln2", g.dln, 0, DIM);
        rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, g.dres_bf, g.dw_part, g.dres, g.dln, res_mid, wf.ln2);
        colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(gln + DIM, g.dw_part);
        put("dres_mid", g.dres, 1, DIM);
        gemm(s, L.o.dx, g.datt, w.o, g.dres_bf);
        dw(dw_adamw<DIM, DIM>, DIM, DIM, P_QKV, gr.o, g.dres_bf, att);
        put("datt", g.datt, 0, DIM);
        attention_bwd_dq<<<dim3(NT, HEADS, B), 128, 0, stream>>>(g.dqkv, g.dsum, qkv, att, g.datt, lse);
        attention_bwd_dkv<<<dim3(NT, HEADS, B), 128, 0, stream>>>(g.part, qkv, g.datt, lse, g.dsum);
        attention_bwd_finish<<<cdiv(N * (HEADS + 2 * KV_HEADS) * HEAD_DIM / 2, 256), 256, 0, stream>>>(g.dqkv, g.part,
                                                                                                       s.rope);
        put("dqkv_pre", g.dqkv, 0, QKV_DIM);
        gemm(s, L.qkv.dx, g.dln, w.qkv, g.dqkv);
        dw(dw_adamw<QKV_DIM, DIM>, QKV_DIM, DIM, 0, gr.qkv, g.dqkv, ln1);
        char nm[48];
        snprintf(nm, sizeof(nm), "gmat.%d", l);
        dump(df, nm, g.gmat, 0, 1, P_MAT);
        put("dln1", g.dln, 0, DIM);
        rmsnorm_bwd<<<N / NORM_ROWS, 256, 0, stream>>>(g.dres, l ? g.dres_bf : nullptr, g.dw_part, g.dres, g.dln, res,
                                                    wf.ln1);
        colsum<<<DIM / 32, dim3(32, 8), 0, stream>>>(gln, g.dw_part);
        put("dres_in", g.dres, 1, DIM);
    }
    embed_bwd_adamw<<<N, DIM / 4, 0, stream>>>(s.master, s.params, s.m, s.v, s.gvec, g.dres, a.io, a.io + IO_SEG, s.step);
    CK(cudaGetLastError());
}

// AdamW runs inside backward for the weight matrices and the embedding (dw_adamw, embed_bwd_adamw); norms after it.
static void train_step(State& s, const Lins& L, FILE* df) {
    next_step<<<1, 1, 0, stream>>>(s.step);
    forward(s, L, df);
    backward(s, L, df);
    dump(df, "gvec", s.gvec, 1, 1, G_VEC);
    static_assert(P_EMBED % 4 == 0 && G_VEC % 4 == 0, "adamw_norms works on float4");
    adamw_norms<<<cdiv((G_VEC - P_EMBED) / 4, 256), 256, 0, stream>>>(s.master, s.params, s.gvec, s.m, s.v, s.step);
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
        die("weights config does not match models/smol135.h");
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
    uint32_t* touched = (uint32_t*)(io + IO_TOUCHED);
    memset(touched, 0, VOCAB / 8);
    for (int j = 0; j < count; j++) touched[sorted[starts[j]].first >> 5] |= 1u << (sorted[starts[j]].first & 31);
}

int main(int argc, char** argv) {
    const char *weights = "data/smol135_init.bin", *data = "data/train.bin", *dump_path = nullptr;
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
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CKB(cublasLtCreate(&lt));

    State s;
    Arena ar{nullptr, 0};
    carve(ar, s);
    const size_t arena_bytes = ar.used;
    CK(cudaMalloc(&ar.base, arena_bytes));
    ar.used = 0;
    carve(ar, s);
    printf("B=%d T=%d params=%zu arena=%.1f MiB\n", B, T, P_TOTAL, arena_bytes / 1048576.0);

    load_weights(s, weights);
    init_rope(s);
    const Lins L = {make_lin(QKV_DIM, DIM), make_lin(DIM, DIM), make_lin(2 * FFN, DIM), make_lin(DIM, FFN),
                    make_lin(VOCAB, DIM, CUDA_R_32F)};
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
        } else if (df || eager) {  // dumps synchronize, so this step runs eagerly
            train_step(s, L, df);
        } else {  // the whole step is one CUDA graph, captured on first use
            if (!graph) {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                train_step(s, L, nullptr);
                CK(cudaStreamEndCapture(stream, &g));
                CK(cudaGraphInstantiate(&graph, g, 0));
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
    return 0;
}
