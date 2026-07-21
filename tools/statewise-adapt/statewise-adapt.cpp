// statewise-adapt: v2 online adaptation supervisor (statewise fork)
//
// teacher-forces a corpus through the model in fixed-size epochs (one llama_decode
// per epoch) and, between decodes, runs the hysteresis swap policy against the
// per-layer expert counters collected via the eval callback. logs per-epoch
// hit-rate so the v2 success metric (collapse -> recovery after a domain switch)
// falls straight out of the CSV.
//
// usage: llama-statewise-adapt -m model.gguf -f corpusA.txt [-t 8 ...]
// requires env LLAMA_STATEWISE_MAP (same file seeds the supervisor's shadow map).
// config env:
//   SW_ADAPT=1        enable swaps (0 = static v1 control run, same accounting)
//   SW_FILE_B=path    second corpus; feed switches to it after SW_SWITCH_TOK tokens
//   SW_SWITCH_TOK=4096  tokens per phase
//   SW_EPOCH=256      tokens per epoch (= decode batch = policy cadence)
//   SW_WINDOW=8       window size in epochs for the policy counters
//   SW_HYST=1.5       swap only if cand_count > HYST * victim_count
//   SW_MAX_SWAPS=4    max swaps per layer per refresh
//   SW_MIN_COUNT=16   min window count for a candidate
//   SW_CSV=sw_adapt_log.csv
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static int    env_i(const char * k, int d)    { const char * v = getenv(k); return v ? atoi(v) : d; }
static double env_d(const char * k, double d) { const char * v = getenv(k); return v ? atof(v) : d; }

struct topk_state {
    std::map<int, std::vector<uint32_t>> epoch_counts; // layer -> per-expert counts, this epoch
    std::vector<int32_t> buf;
};

static bool collect_topk(struct ggml_tensor * t, bool ask, void * ud) {
    topk_state * st = (topk_state *) ud;
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
    // non-contiguous view: honor nb[] strides (see expert-stats)
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1] * t->ne[2] * t->ne[3];
    const size_t  row_stride = t->nb[1] / sizeof(int32_t);
    const size_t  span_bytes = (size_t)(ne1 - 1) * t->nb[1] + (size_t) ne0 * sizeof(int32_t);
    st->buf.resize((span_bytes + sizeof(int32_t) - 1) / sizeof(int32_t));
    ggml_backend_tensor_get(t, st->buf.data(), 0, span_bytes);
    auto & c = st->epoch_counts[il];
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

struct layer_shadow {
    int32_t K = 0;
    std::vector<int32_t> slot_expert;             // slot -> expert id currently hot
    std::vector<uint8_t> is_hot;                  // expert -> hot flag
    std::deque<std::vector<uint32_t>> window;     // recent epoch count vectors
};

static void grow(layer_shadow & sh, size_t n) {
    if (sh.is_hot.size() < n) sh.is_hot.resize(n, 0);
}

static void print_usage(int, char ** argv) {
    printf("\nusage: %s -m model.gguf -f corpusA.txt [-t N ...]   (env: see header)\n\n", argv[0]);
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx   = 512;   // known-good telemetry geometry on this box
    params.n_batch = 256;
    params.escape  = false;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    const int    adapt      = env_i("SW_ADAPT", 1);
    const char * file_b     = getenv("SW_FILE_B");
    const int    switch_tok = env_i("SW_SWITCH_TOK", 4096);
    const int    epoch_tok  = env_i("SW_EPOCH", 256);
    const int    win_epochs = env_i("SW_WINDOW", 8);
    const double hyst       = env_d("SW_HYST", 1.5);
    const int    max_swaps  = env_i("SW_MAX_SWAPS", 4);
    const int    min_count  = env_i("SW_MIN_COUNT", 16);
    const char * csv_path   = getenv("SW_CSV") ? getenv("SW_CSV") : "sw_adapt_log.csv";

    const char * map_path = getenv("LLAMA_STATEWISE_MAP");
    if (!map_path) {
        fprintf(stderr, "error: LLAMA_STATEWISE_MAP not set - nothing to adapt\n");
        return 1;
    }

    // seed shadow state from the same map file statewise_init consumes
    std::map<int, layer_shadow> shadows;
    {
        std::ifstream fin(map_path);
        if (!fin) { fprintf(stderr, "error: cannot open map '%s'\n", map_path); return 1; }
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int il = -1, K = 0;
            if (!(ss >> il >> K) || K <= 0) continue;
            layer_shadow sh;
            sh.K = K;
            sh.slot_expert.resize(K);
            for (int i = 0; i < K; i++) {
                ss >> sh.slot_expert[i];
                grow(sh, (size_t) sh.slot_expert[i] + 1);
                sh.is_hot[sh.slot_expert[i]] = 1;
            }
            shadows[il] = std::move(sh);
        }
    }
    fprintf(stderr, "statewise-adapt: shadow map for %zu layers, adapt=%d epoch=%d window=%d hyst=%.2f\n",
        shadows.size(), adapt, epoch_tok, win_epochs, hyst);

    params.n_batch           = epoch_tok;
    topk_state st;
    params.cb_eval           = collect_topk;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "error: failed to load model\n"); return 1; }

    // cross-check the live cache against the shadow map
    for (auto & kv : shadows) {
        const int32_t K = llama_statewise_layer_k(model, kv.first);
        if (K != kv.second.K) {
            fprintf(stderr, "error: layer %d shadow K=%d but live K=%d (cache inactive?)\n", kv.first, kv.second.K, K);
            return 1;
        }
    }

    // build the feed: phase A then phase B
    auto slurp = [](const char * p) {
        std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
    };
    std::vector<llama_token> feed = common_tokenize(ctx, slurp(params.prompt_file.c_str()), true);
    if ((int) feed.size() > switch_tok) feed.resize(switch_tok);
    const int tok_a = (int) feed.size();
    if (file_b) {
        std::vector<llama_token> fb = common_tokenize(ctx, slurp(file_b), false);
        if ((int) fb.size() > switch_tok) fb.resize(switch_tok);
        feed.insert(feed.end(), fb.begin(), fb.end());
    }
    const int n_ctx_t = llama_n_ctx(ctx);
    feed.resize((feed.size() / n_ctx_t) * n_ctx_t); // whole chunks only
    fprintf(stderr, "statewise-adapt: feed %zu tokens (phase A ends at %d), chunks of %d, epochs of %d\n",
        feed.size(), tok_a, n_ctx_t, epoch_tok);
    if (feed.empty()) { fprintf(stderr, "error: empty feed\n"); return 1; }

    std::ofstream csv(csv_path);
    csv << "epoch,tok_end,phase,hit_pct,swaps\n";

    int tok_done = 0, epoch_idx = 0;
    std::vector<double> curve_hit; std::vector<int> curve_tok;
    for (size_t off = 0; off < feed.size(); off += n_ctx_t) {
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int j = 0; j < n_ctx_t; j += epoch_tok) {
            const int n_eval = std::min(epoch_tok, n_ctx_t - j);
            llama_batch batch = llama_batch_get_one(feed.data() + off + j, n_eval);
            if (llama_decode(ctx, batch)) { fprintf(stderr, "error: decode failed\n"); return 1; }
            tok_done += n_eval; epoch_idx++;

            // ---- epoch close: accounting vs the map that was LIVE during this epoch
            uint64_t hits = 0, total = 0;
            int swaps = 0;
            for (auto & kv : shadows) {
                layer_shadow & sh = kv.second;
                auto it = st.epoch_counts.find(kv.first);
                std::vector<uint32_t> epoch;
                if (it != st.epoch_counts.end()) epoch = it->second;
                grow(sh, epoch.size());
                for (size_t e = 0; e < epoch.size(); e++) {
                    total += epoch[e];
                    if (sh.is_hot[e]) hits += epoch[e];
                }
                sh.window.push_back(std::move(epoch));
                while ((int) sh.window.size() > win_epochs) sh.window.pop_front();
            }
            const double hit_pct = total ? 100.0 * hits / total : 0.0;

            // ---- policy: hysteresis swaps from window counts
            if (adapt) {
                for (auto & kv : shadows) {
                    layer_shadow & sh = kv.second;
                    size_t n = sh.is_hot.size();
                    for (auto & ep : sh.window) n = std::max(n, ep.size());
                    grow(sh, n);
                    std::vector<uint64_t> w(n, 0);
                    for (auto & ep : sh.window)
                        for (size_t e = 0; e < ep.size(); e++) w[e] += ep[e];
                    // candidates: hottest uncached, desc
                    std::vector<int32_t> cand;
                    for (size_t e = 0; e < n; e++) if (!sh.is_hot[e] && w[e] > 0) cand.push_back((int32_t) e);
                    std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return w[a] > w[b]; });
                    // victims: slots by hot-expert count, asc
                    std::vector<int32_t> vslot(sh.K);
                    for (int32_t s = 0; s < sh.K; s++) vslot[s] = s;
                    std::sort(vslot.begin(), vslot.end(), [&](int32_t a, int32_t b) {
                        return w[sh.slot_expert[a]] < w[sh.slot_expert[b]]; });
                    for (int i = 0; i < max_swaps && i < (int) cand.size() && i < sh.K; i++) {
                        const int32_t e = cand[i], s = vslot[i], v = sh.slot_expert[s];
                        if ((int64_t) w[e] < min_count || (double) w[e] <= hyst * (double) w[v]) break;
                        const int32_t r = llama_statewise_swap(model, kv.first, s, e);
                        if (r == 0) {
                            sh.is_hot[v] = 0; sh.is_hot[e] = 1; sh.slot_expert[s] = e; swaps++;
                        } else {
                            fprintf(stderr, "warn: swap(il=%d,slot=%d,e=%d) -> %d\n", kv.first, s, e, r);
                            break;
                        }
                    }
                }
            }

            const char * phase = tok_done <= tok_a ? "A" : "B";
            csv << epoch_idx << "," << tok_done << "," << phase << ","
                << (int) (hit_pct * 100) / 100.0 << "," << swaps << "\n";
            csv.flush();
            fprintf(stderr, "epoch %3d  tok %5d  [%s]  hit %5.1f%%  swaps %d\n",
                epoch_idx, tok_done, phase, hit_pct, swaps);
            curve_hit.push_back(hit_pct); curve_tok.push_back(tok_done);
            st.epoch_counts.clear();
        }
    }

    // summary: end of A, dip at switch, end of B
    auto mean_range = [&](int lo, int hi) { // epochs [lo,hi)
        double s = 0; int c = 0;
        for (int i = lo; i < hi && i < (int) curve_hit.size(); i++) { s += curve_hit[i]; c++; }
        return c ? s / c : 0.0;
    };
    const int ep_a = tok_a / epoch_tok;
    const int n_ep = (int) curve_hit.size();
    printf("\n==== STATEWISE V2 (%s) ====\n", adapt ? "adaptive" : "static control");
    printf("phase A end   (last 4 A epochs): %5.1f%%\n", mean_range(ep_a - 4, ep_a));
    if (n_ep > ep_a) {
        printf("switch dip    (first 2 B epochs): %5.1f%%\n", mean_range(ep_a, ep_a + 2));
        printf("phase B end   (last 4 B epochs): %5.1f%%\n", mean_range(n_ep - 4, n_ep));
    }
    printf("csv: %s\n", csv_path);

    // dump the adapted map in statewise_init format -> reusable by any tool via LLAMA_STATEWISE_MAP
    if (const char * dump = getenv("SW_DUMP_MAP")) {
        std::ofstream fout(dump);
        fout << "# adapted map dumped by statewise-adapt\n";
        for (auto & kv : shadows) {
            fout << kv.first << " " << kv.second.K;
            for (int32_t s2 = 0; s2 < kv.second.K; s2++) fout << " " << kv.second.slot_expert[s2];
            fout << "\n";
        }
        printf("dumped adapted map: %s\n", dump);
    }
    return 0;
}
