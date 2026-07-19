#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// statewise-poc: split-expert MoE execution across GPU (hot cache + dummy slot) and CPU (full tensor + sentinel ids)
// validates numerics + measures scheduler boundary overhead at decode batch sizes

static const int64_t n_embd   = 2048;
static const int64_t n_ff     = 768;
static const int64_t n_expert = 128;
static const int64_t n_used   = 8;
static const int64_t K_cache  = 32;   // hot slots; slot K_cache = dummy zeros

struct graph_pack {
    ggml_context * ctx;
    ggml_cgraph  * gf;
    ggml_tensor  * x, * ids, * ids_hot, * ids_cold, * y;
};

static graph_pack build_graph(int mode, int64_t n_tok, ggml_tensor * Wc, ggml_tensor * Wf, ggml_tensor * Wg) {
    // mode 0 = cpu ref, 1 = split, 2 = gpu full
    ggml_init_params ip = { 16*1024*1024, NULL, true };
    graph_pack g = {};
    g.ctx = ggml_init(ip);
    g.gf  = ggml_new_graph(g.ctx);
    g.x   = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, n_embd, 1, n_tok);
    ggml_set_name(g.x, "x"); ggml_set_input(g.x);
    if (mode == 1) {
        g.ids_hot  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_used, n_tok);
        g.ids_cold = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_used, n_tok);
        ggml_set_name(g.ids_hot, "ids_hot");  ggml_set_input(g.ids_hot);
        ggml_set_name(g.ids_cold, "ids_cold"); ggml_set_input(g.ids_cold);
        ggml_tensor * yh = ggml_mul_mat_id(g.ctx, Wc, g.x, g.ids_hot);
        ggml_tensor * yc = ggml_mul_mat_id(g.ctx, Wf, g.x, g.ids_cold);
        g.y = ggml_add(g.ctx, yh, yc);
    } else {
        g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_used, n_tok);
        ggml_set_name(g.ids, "ids"); ggml_set_input(g.ids);
        g.y = ggml_mul_mat_id(g.ctx, mode == 0 ? Wf : Wg, g.x, g.ids);
    }
    ggml_set_name(g.y, "y"); ggml_set_output(g.y);
    ggml_build_forward_expand(g.gf, g.y);
    return g;
}

int main(int argc, char ** argv) {
    const int64_t n_tok    = argc > 1 ? atoll(argv[1]) : 1;
    const double  hit_frac = argc > 2 ? atof(argv[2])  : 0.75;
    const int     n_iter   = argc > 3 ? atoi(argv[3])  : 300;
    ggml_time_init();

    ggml_backend_dev_t dg = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_t gpu = dg ? ggml_backend_dev_init(dg, NULL) : NULL;
    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), NULL);
    if (!gpu) { printf("no GPU backend, aborting\n"); return 1; }
    printf("gpu: %s | cpu: %s | n_tok=%lld hit=%.2f iters=%d\n",
        ggml_backend_name(gpu), ggml_backend_name(cpu), (long long)n_tok, hit_frac, n_iter);

    ggml_init_params ipw = { 8*ggml_tensor_overhead(), NULL, true };
    ggml_context * cw_gpu = ggml_init(ipw);
    ggml_context * cw_cpu = ggml_init(ipw);
    ggml_tensor * Wc = ggml_new_tensor_3d(cw_gpu, GGML_TYPE_F16, n_embd, n_ff, K_cache + 1);
    ggml_tensor * Wg = ggml_new_tensor_3d(cw_gpu, GGML_TYPE_F16, n_embd, n_ff, n_expert);
    ggml_tensor * Wf = ggml_new_tensor_3d(cw_cpu, GGML_TYPE_F16, n_embd, n_ff, n_expert);
    ggml_set_name(Wc, "W_cache"); ggml_set_name(Wg, "W_gpu_full"); ggml_set_name(Wf, "W_cpu_full");
    ggml_backend_buffer_t bg = ggml_backend_alloc_ctx_tensors(cw_gpu, gpu);
    ggml_backend_buffer_t bc = ggml_backend_alloc_ctx_tensors(cw_cpu, cpu);
    if (!bg || !bc) { printf("weight alloc failed\n"); return 1; }

    // fill weights: random f16, cache slots = experts 0..K-1, dummy slot = zeros
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> ud(-0.05f, 0.05f);
        const size_t slab = (size_t)n_embd*n_ff;
        std::vector<ggml_fp16_t> h(slab);
        std::vector<ggml_fp16_t> zero(slab);
        memset(zero.data(), 0, slab*sizeof(ggml_fp16_t));
        for (int64_t e = 0; e < n_expert; e++) {
            for (size_t i = 0; i < slab; i++) h[i] = ggml_fp32_to_fp16(ud(rng));
            ggml_backend_tensor_set(Wf, h.data(), e*slab*sizeof(ggml_fp16_t), slab*sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(Wg, h.data(), e*slab*sizeof(ggml_fp16_t), slab*sizeof(ggml_fp16_t));
            if (e < K_cache) ggml_backend_tensor_set(Wc, h.data(), e*slab*sizeof(ggml_fp16_t), slab*sizeof(ggml_fp16_t));
        }
        ggml_backend_tensor_set(Wc, zero.data(), K_cache*slab*sizeof(ggml_fp16_t), slab*sizeof(ggml_fp16_t));
    }
    printf("weights filled (%.0f MB cpu, %.0f MB gpu)\n",
        (double)ggml_backend_buffer_get_size(bc)/1e6, (double)ggml_backend_buffer_get_size(bg)/1e6);

    // routing
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<int32_t> ids(n_used*n_tok), ids_hot(n_used*n_tok), ids_cold(n_used*n_tok);
    int hits = 0;
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t s = 0; s < n_used; s++) {
            int32_t id = (u01(rng) < hit_frac) ? (int32_t)(u01(rng)*K_cache) : (int32_t)(K_cache + u01(rng)*(n_expert - K_cache));
            if (id >= n_expert) id = n_expert - 1;
            ids[t*n_used+s] = id;
            if (id < K_cache) { ids_hot[t*n_used+s] = id;      ids_cold[t*n_used+s] = -1; hits++; }
            else              { ids_hot[t*n_used+s] = K_cache; ids_cold[t*n_used+s] = id; }
        }
    }
    printf("actual hit rate: %.1f%%\n", 100.0*hits/(n_used*n_tok));

    std::vector<float> hx(n_embd*n_tok);
    for (auto & v : hx) v = (float)(u01(rng)*2 - 1);

    ggml_backend_t backends[2] = { gpu, cpu };
    const char * names[3] = { "all-CPU (ref)", "SPLIT hot-gpu/cold-cpu", "all-GPU" };
    std::vector<float> y_ref;
    double us[3] = {0,0,0};

    for (int mode = 0; mode <= 2; mode++) {
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, getenv("POC_PARALLEL") != NULL, false);
        graph_pack g = build_graph(mode, n_tok, Wc, Wf, Wg);
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
            ggml_tensor * nd = ggml_graph_node(g.gf, i);
            if (nd->op == GGML_OP_ADD) { ggml_backend_sched_set_tensor_backend(sched, nd, gpu); }
            if (nd->op != GGML_OP_MUL_MAT_ID) continue;
            printf("  supports_op(gpu, %s src0=%s): %d\n", nd->name, nd->src[0]->name, (int)ggml_backend_supports_op(gpu, nd));
            ggml_backend_sched_set_tensor_backend(sched, nd, nd->src[0] == Wf && mode != 2 ? cpu : gpu);
        }
        if (!ggml_backend_sched_alloc_graph(sched, g.gf)) { printf("alloc_graph failed mode %d\n", mode); return 1; }
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) {
            ggml_tensor * nd = ggml_graph_node(g.gf, i);
            ggml_backend_t bk = ggml_backend_sched_get_tensor_backend(sched, nd);
            printf("  node %-14s op=%-12s -> %s\n", nd->name, ggml_op_name(nd->op), bk ? ggml_backend_name(bk) : "?");
        }
        ggml_backend_tensor_set(g.x, hx.data(), 0, hx.size()*sizeof(float));
        if (mode == 1) {
            ggml_backend_tensor_set(g.ids_hot,  ids_hot.data(),  0, ids_hot.size()*sizeof(int32_t));
            ggml_backend_tensor_set(g.ids_cold, ids_cold.data(), 0, ids_cold.size()*sizeof(int32_t));
        } else {
            ggml_backend_tensor_set(g.ids, ids.data(), 0, ids.size()*sizeof(int32_t));
        }
        for (int i = 0; i < 10; i++) ggml_backend_sched_graph_compute(sched, g.gf);
        ggml_backend_sched_synchronize(sched);
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < n_iter; i++) ggml_backend_sched_graph_compute(sched, g.gf);
        ggml_backend_sched_synchronize(sched);
        us[mode] = (double)(ggml_time_us() - t0)/n_iter;

        std::vector<float> y(n_ff*n_used*n_tok);
        ggml_backend_tensor_get(g.y, y.data(), 0, y.size()*sizeof(float));
        if (mode == 0) { y_ref = y; }
        else {
            double maxa = 0, maxd = 0;
            for (size_t i = 0; i < y.size(); i++) { maxa = fmax(maxa, fabs((double)y_ref[i])); maxd = fmax(maxd, fabs((double)y[i] - y_ref[i])); }
            printf("%-24s rel-err vs ref: %.2e %s\n", names[mode], maxd/(maxa > 0 ? maxa : 1), (maxd/(maxa>0?maxa:1)) < 5e-3 ? "OK" : "FAIL");
        }
        ggml_backend_sched_free(sched);
        ggml_free(g.ctx);
    }
    printf("\n%-24s %10s\n", "config", "us/iter");
    for (int m = 0; m <= 2; m++) printf("%-24s %10.1f\n", names[m], us[m]);
    printf("\nsplit boundary overhead vs ideal(gpu_hot+cpu_cold parts): split=%.1fus cpu=%.1fus gpu=%.1fus\n", us[1], us[0], us[2]);
    printf("naive scale to 48 layers x 3 proj: split %.2f ms/token\n", us[1]*48*3/1000.0);
    return 0;
}




