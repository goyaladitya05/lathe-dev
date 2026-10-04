// lathe: SmolLM2-135M training engine. M1: forward pass + loss.
//   make BATCH=1 SEQ=2048 && build/train_b1_t2048 --weights data/smol135_init.bin --data data/val.bin
//   --dump PATH writes layer 0 / last layer / output tensors of batch 0 for ref/test.py.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Parameter layout, mirrored by ref/model.py lathe_layout. Matrices lead each layer so they stay 256B-aligned.
constexpr size_t P_EMBED = (size_t)VOCAB * DIM;
constexpr size_t P_QKV = (size_t)QKV_DIM * DIM, P_O = (size_t)DIM * DIM;
constexpr size_t P_GU = 2ull * FFN * DIM, P_DOWN = (size_t)DIM * FFN;
constexpr size_t P_LAYER = P_QKV + P_O + P_GU + P_DOWN + 2 * DIM;
constexpr size_t P_TOTAL = P_EMBED + LAYERS * P_LAYER + DIM;
static_assert(P_EMBED * 2 % 256 == 0 && P_QKV * 2 % 256 == 0 && P_O * 2 % 256 == 0 && P_GU * 2 % 256 == 0 &&
                  P_DOWN * 2 % 256 == 0 && P_LAYER * 2 % 256 == 0,
              "bf16 matrices must stay 256B-aligned for cuBLASLt");

struct LayerW {
    bf16 *qkv, *o, *gu, *down, *ln1, *ln2;
};
static LayerW layer_w(bf16* p, int l) {
    bf16* q = p + P_EMBED + l * P_LAYER;
    bf16* ln = q + P_QKV + P_O + P_GU + P_DOWN;
    return {q, q + P_QKV, q + P_QKV + P_O, q + P_QKV + P_O + P_GU, ln, ln + DIM};
}

// Per-layer activations are all kept; backward (M2) reads them.
struct Acts {
    bf16* res;      // [LAYERS+1][N][DIM] residual stream into each layer, res[LAYERS] feeds the final norm
    bf16* res_mid;  // [LAYERS][N][DIM] after the attention residual add
    bf16 *ln1, *ln2;  // [LAYERS][N][DIM]
    bf16* qkv;      // [LAYERS][N][QKV_DIM] post-RoPE
    bf16* att;      // [LAYERS][N][DIM]
    bf16* gu;       // [LAYERS][N][2*FFN]
    bf16* act;      // [LAYERS][N][FFN]
    bf16* lnf;      // [N][DIM]
    bf16* logits;   // [N][VOCAB]
    float* losses;  // [N]
    int* tokens;    // [2][N] inputs then targets
};

struct State {
    float* master;  // [P_TOTAL] fp32 master weights
    bf16* params;   // [P_TOTAL] bf16 working weights
    float2* rope;   // [T][HEAD_DIM/2] (cos, sin)
    void* workspace;
    Acts a;
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
    s.rope = ar.take<float2>(T * HEAD_DIM / 2);
    s.workspace = ar.take<char>(CUBLAS_WORKSPACE);
    Acts& a = s.a;
    a.res = ar.take<bf16>((LAYERS + 1) * ND);
    a.res_mid = ar.take<bf16>(LAYERS * ND);
    a.ln1 = ar.take<bf16>(LAYERS * ND);
    a.ln2 = ar.take<bf16>(LAYERS * ND);
    a.qkv = ar.take<bf16>((size_t)LAYERS * N * QKV_DIM);
    a.att = ar.take<bf16>(LAYERS * ND);
    a.gu = ar.take<bf16>((size_t)LAYERS * N * 2 * FFN);
    a.act = ar.take<bf16>((size_t)LAYERS * N * FFN);
    a.lnf = ar.take<bf16>(ND);
    a.logits = ar.take<bf16>((size_t)N * VOCAB);
    a.losses = ar.take<float>(N);
    a.tokens = ar.take<int>(2 * N);
}

static cudaStream_t stream;
static cublasLtHandle_t lt;

// y[n][m] = x[n][k] . w[m][k]^T (+ res[n][m]); row-major bf16, fp32 accumulate. Built once per shape.
struct Gemm {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t w, x, y;
    cublasLtMatmulAlgo_t algo;
};

static Gemm make_gemm(int m, int n, int k) {
    Gemm g;
    CKB(cublasLtMatmulDescCreate(&g.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tr = CUBLAS_OP_T;
    CKB(cublasLtMatmulDescSetAttribute(g.desc, CUBLASLT_MATMUL_DESC_TRANSA, &tr, sizeof(tr)));
    // Column-major views of the row-major tensors.
    CKB(cublasLtMatrixLayoutCreate(&g.w, CUDA_R_16BF, k, m, k));
    CKB(cublasLtMatrixLayoutCreate(&g.x, CUDA_R_16BF, k, n, k));
    CKB(cublasLtMatrixLayoutCreate(&g.y, CUDA_R_16BF, m, n, m));
    cublasLtMatmulPreference_t pref;
    CKB(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = CUBLAS_WORKSPACE;
    CKB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t h;
    int found = 0;
    CKB(cublasLtMatmulAlgoGetHeuristic(lt, g.desc, g.w, g.x, g.y, g.y, pref, 1, &h, &found));
    if (!found) die("no cuBLASLt algo");
    CKB(cublasLtMatmulPreferenceDestroy(pref));
    g.algo = h.algo;
    return g;
}

static void gemm(const State& s, const Gemm& g, bf16* y, const bf16* x, const bf16* w, const bf16* res = nullptr) {
    const float one = 1.f, zero = 0.f;
    CKB(cublasLtMatmul(lt, g.desc, &one, w, g.w, x, g.x, res ? &one : &zero, res ? res : y, g.y, y, g.y, &g.algo,
                       s.workspace, CUBLAS_WORKSPACE, stream));
}

struct Gemms {
    Gemm qkv, o, gu, down, head;
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

static void forward(State& s, const Gemms& g, FILE* df) {
    Acts& a = s.a;
    const size_t ND = (size_t)N * DIM;
    embed<<<cdiv(N * DIM / 8, 256), 256, 0, stream>>>(a.res, a.tokens, s.params);
    for (int l = 0; l < LAYERS; l++) {
        const LayerW w = layer_w(s.params, l);
        bf16 *res = a.res + l * ND, *res_mid = a.res_mid + l * ND, *ln1 = a.ln1 + l * ND, *ln2 = a.ln2 + l * ND;
        bf16 *qkv = a.qkv + (size_t)l * N * QKV_DIM, *att = a.att + l * ND;
        bf16 *gu = a.gu + (size_t)l * N * 2 * FFN, *act = a.act + (size_t)l * N * FFN;
        FILE* d = l == 0 || l == LAYERS - 1 ? df : nullptr;
        char nm[48];
        auto put = [&](const char* name, const bf16* p, size_t cols) {
            snprintf(nm, sizeof(nm), "L%d.%s", l, name);
            dump(d, nm, p, 0, N, cols);
        };

        put("res", res, DIM);
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(ln1, res, w.ln1);
        put("ln1", ln1, DIM);
        gemm(s, g.qkv, qkv, ln1, w.qkv);
        put("qkv_pre", qkv, QKV_DIM);
        rope<<<cdiv(N * (HEADS + KV_HEADS) * HEAD_DIM / 2, 256), 256, 0, stream>>>(qkv, s.rope);
        put("qkv", qkv, QKV_DIM);
        attention_fwd<<<dim3(T / ATT_TILE, HEADS, B), ATT_TILE, 0, stream>>>(att, qkv);
        put("att", att, DIM);
        gemm(s, g.o, res_mid, att, w.o, res);
        put("res_mid", res_mid, DIM);
        rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(ln2, res_mid, w.ln2);
        put("ln2", ln2, DIM);
        gemm(s, g.gu, gu, ln2, w.gu);
        put("gu", gu, 2 * FFN);
        swiglu<<<cdiv(N * FFN / 8, 256), 256, 0, stream>>>(act, gu);
        put("act", act, FFN);
        gemm(s, g.down, res + ND, act, w.down, res_mid);
        put("res_out", res + ND, DIM);
    }
    rmsnorm<<<cdiv(N, 8), 256, 0, stream>>>(a.lnf, a.res + LAYERS * ND, s.params + P_TOTAL - DIM);
    gemm(s, g.head, a.logits, a.lnf, s.params);  // tied embedding
    cross_entropy_fwd<<<N, CE_THREADS, 0, stream>>>(a.losses, a.logits, a.tokens + N);
    CK(cudaGetLastError());
    dump(df, "lnf", a.lnf, 0, N, DIM);
    dump(df, "logits", a.logits, 0, N, VOCAB);
    dump(df, "losses", a.losses, 1, N, 1);
    dump(df, "tokens", a.tokens, 2, 2, N);
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
static void load_batch(const Tokens& d, long i, int* h) {
    size_t nb = (d.n - 1) / N, s = (i % nb) * N;
    for (int j = 0; j < N; j++) {
        h[j] = d.data[s + j];
        h[N + j] = d.data[s + j + 1];
    }
}

int main(int argc, char** argv) {
    const char *weights = "data/smol135_init.bin", *data = "data/val.bin", *dump_path = nullptr;
    int batches = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--weights")) weights = argv[i + 1];
        else if (!strcmp(argv[i], "--data")) data = argv[i + 1];
        else if (!strcmp(argv[i], "--dump")) dump_path = argv[i + 1];
        else if (!strcmp(argv[i], "--batches")) batches = atoi(argv[i + 1]);
        else die("usage: train [--weights f] [--data f] [--batches n] [--dump f]");
    }

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
    const Gemms g = {make_gemm(QKV_DIM, N, DIM), make_gemm(DIM, N, DIM), make_gemm(2 * FFN, N, DIM),
                     make_gemm(DIM, N, FFN), make_gemm(VOCAB, N, DIM)};
    const Tokens d = map_tokens(data);
    int* h_tok;
    CK(cudaMallocHost(&h_tok, 2 * N * sizeof(int)));
    std::vector<float> losses(N);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    for (int i = 0; i < batches; i++) {
        load_batch(d, i, h_tok);
        CK(cudaMemcpyAsync(s.a.tokens, h_tok, 2 * N * sizeof(int), cudaMemcpyHostToDevice, stream));
        FILE* df = i == 0 && dump_path ? fopen(dump_path, "wb") : nullptr;
        CK(cudaEventRecord(e0, stream));
        forward(s, g, df);
        CK(cudaEventRecord(e1, stream));
        CK(cudaMemcpyAsync(losses.data(), s.a.losses, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        if (df) fclose(df);
        double sum = 0;
        for (float x : losses) sum += x;
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("batch %d loss %.6f fwd %.2f ms\n", i, sum / N, ms);
    }
    return 0;
}
