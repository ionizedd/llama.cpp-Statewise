#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

// expert-stats: per-layer MoE expert routing telemetry (statewise fork, patch #1)
// hooks the scheduler eval callback, captures ffn_moe_topk-<il> selections,
// reports per-layer coverage + writes expert_counts.csv

struct expert_stats {
    std::map<int, std::vector<uint64_t>> counts;
    std::vector<int32_t> buf;
};

static bool collect_topk(struct ggml_tensor * t, bool ask, void * user_data) {
    expert_stats * st = (expert_stats *) user_data;
    const bool is_topk = strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk || t->type != GGML_TYPE_I32) {
        return true;
    }
    int il = -1;
    const char * dash = strrchr(t->name, '-');
    if (dash) il = atoi(dash + 1);

    // ffn_moe_topk is typically a non-contiguous VIEW (top n_expert_used rows of the
    // [n_expert, n_tokens] argsort) -> must honor nb[] strides, not read linearly
    const int64_t ne0 = t->ne[0];                          // n_expert_used
    const int64_t ne1 = t->ne[1] * t->ne[2] * t->ne[3];    // n_tokens
    const size_t  row_stride = t->nb[1] / sizeof(int32_t); // underlying row length
    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "expert-stats: topk '%s' ne=[%lld,%lld] row_stride=%zu cont=%d\n",
            t->name, (long long) ne0, (long long) ne1, row_stride, (int) ggml_is_contiguous(t));
        printed = true;
    }
    const size_t span_bytes = (size_t)(ne1 - 1) * t->nb[1] + (size_t) ne0 * sizeof(int32_t);
    st->buf.resize((span_bytes + sizeof(int32_t) - 1) / sizeof(int32_t));
    ggml_backend_tensor_get(t, st->buf.data(), 0, span_bytes);

    auto & c = st->counts[il];
    for (int64_t j = 0; j < ne1; j++) {
        for (int64_t i = 0; i < ne0; i++) {
            const int32_t id = st->buf[j * row_stride + i];
            if (id < 0) continue;
            if ((size_t) id >= c.size()) c.resize(id + 1, 0);
            c[id]++;
        }
    }
    return true;
}

static void print_usage(int, char ** argv) {
    printf("\nusage: %s -m model.gguf -f text.txt [-ngl N] [--n-cpu-moe N] [--chunks N] [-c N] [-b N] [-t N]\n\n", argv[0]);
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 2048;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    expert_stats stats;
    params.cb_eval           = collect_topk;
    params.cb_eval_user_data = &stats;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "error: failed to load model\n");
        return 1;
    }
    char mdesc[128];
    llama_model_desc(model, mdesc, sizeof(mdesc));
    printf("model: %s\n", mdesc);

    std::ifstream fin(params.prompt_file);
    if (!fin) {
        fprintf(stderr, "error: failed to open %s\n", params.prompt_file.c_str());
        return 1;
    }
    std::stringstream ss;
    ss << fin.rdbuf();
    std::string text = ss.str();

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true);
    printf("tokenized: %zu tokens\n", tokens.size());

    const int n_ctx_t     = llama_n_ctx(ctx);
    const int n_batch     = params.n_batch;
    const int n_chunk_max = (int)(tokens.size() / n_ctx_t);
    const int n_chunk     = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    printf("processing %d chunks of %d tokens (batch %d)\n", n_chunk, n_ctx_t, n_batch);

    for (int i = 0; i < n_chunk; i++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int j = 0; j < n_ctx_t; j += n_batch) {
            const int n_eval = std::min(n_batch, n_ctx_t - j);
            llama_batch batch = llama_batch_get_one(tokens.data() + (size_t)i * n_ctx_t + j, n_eval);
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "error: decode failed at chunk %d\n", i);
                return 1;
            }
        }
        printf("chunk %d/%d done\n", i + 1, n_chunk);
        fflush(stdout);
    }

    const uint64_t toks = (uint64_t) n_chunk * n_ctx_t;
    printf("\n==== EXPERT ROUTING STATS (%llu tokens) ====\n", (unsigned long long) toks);
    const int Ks[6] = {8, 16, 24, 32, 48, 64};
    std::vector<double> cov_sum(6, 0.0);
    std::vector<int> k90s;
    printf("layer  n_exp  ent%%    cov8  cov16  cov24  cov32  cov48  cov64   K90\n");
    for (auto & kv : stats.counts) {
        std::vector<uint64_t> c = kv.second;
        uint64_t total = 0;
        for (auto v : c) total += v;
        if (total == 0) continue;
        std::sort(c.begin(), c.end(), std::greater<uint64_t>());
        double H = 0.0;
        for (auto v : c) if (v) { double p = (double) v / total; H -= p * std::log2(p); }
        const double Hu = std::log2((double) c.size());
        uint64_t run = 0; int k90 = 0;
        for (size_t k = 0; k < c.size(); k++) {
            run += c[k];
            if (!k90 && (double) run / total >= 0.90) k90 = (int) k + 1;
        }
        printf("%5d  %5zu  %5.1f  ", kv.first, c.size(), 100.0 * H / Hu);
        for (int ki = 0; ki < 6; ki++) {
            uint64_t s = 0;
            for (int k = 0; k < Ks[ki] && k < (int) c.size(); k++) s += c[k];
            const double cov = 100.0 * s / total;
            cov_sum[ki] += cov;
            printf("%5.1f  ", cov);
        }
        printf("%4d\n", k90);
        k90s.push_back(k90);
    }
    const int L = (int) stats.counts.size();
    if (L > 0) {
        printf("\nMEAN over %d layers: ", L);
        for (int ki = 0; ki < 6; ki++) printf("cov%d=%.1f%%  ", Ks[ki], cov_sum[ki] / L);
        int k90min = 999, k90max = 0; double k90avg = 0.0;
        for (int v : k90s) { k90min = std::min(k90min, v); k90max = std::max(k90max, v); k90avg += v; }
        printf("\nK90 (experts needed for 90%% of routing): avg %.1f  min %d  max %d\n", k90avg / L, k90min, k90max);
    }
    std::ofstream csv("expert_counts.csv");
    csv << "layer,expert,count\n";
    for (auto & kv : stats.counts)
        for (size_t e = 0; e < kv.second.size(); e++)
            csv << kv.first << "," << e << "," << kv.second[e] << "\n";
    printf("wrote expert_counts.csv\n");

    return 0;
}


