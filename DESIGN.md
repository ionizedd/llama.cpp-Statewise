# DESIGN: statewise expert cache
Goal: beat static-placement decode (33.9 t/s @ ncmoe 20) by caching HOT experts in VRAM at per-expert granularity, adaptively.

## Why (measured, see BASELINES.md)
- Skew: top-32/128 experts per layer = 70% (wiki) / 84% (code) of routing. Static layer split == uniform curve.
- Hot sets stable within domain (83.9%) but ANTI-correlated across domains (11.3% < 25% random) -> placement must adapt at runtime.

## Mechanism
Per MoE layer l, per projection (gate/up/down):
- W_full stays on CPU (authoritative, serves ALL misses -- no repacking, identity ids).
- W_cache[l]: extra GPU tensor [n_embd, n_ff, K_l] holding COPIES of the K_l currently-hot experts. K_l budgeted per layer from telemetry (K90 spans 26-83).
- map_cache[l]: i32 [n_expert], expert id -> cache slot, or -1. map_cold[l]: expert id -> id, or -1 (hit).
- Graph (decode): ids_hot = get_rows(map_cache, ids); ids_cold = get_rows(map_cold, ids);
  out = mul_mat_id(W_cache, x, ids_hot) + mul_mat_id(W_full, x, ids_cold)
- ggml change: mul_mat_id treats id -1 as SKIP (CPU: skip in grouping pass + zero dst slot; Vulkan: shader guard).
- pp path: large ubatch keeps today's dense flow (skip the split when n_tokens > threshold; graphs are rebuilt per ubatch so this is a build-time branch). Rationale: pp at ub2048/no-mmap already hits 707 t/s.

## v1 (static-informed): populate W_cache + maps at LOAD from a profile CSV (tools/expert-stats output). No GGUF changes, no arch registration.
## v2 (online): usage counters (reuse topk telemetry path, in-process) -> periodic refresh between decodes: swap map contents + upload changed experts (2.4MB each) via ReBAR staging. Hysteresis: refresh every N tokens, evict only below-threshold experts. Full domain re-warm ~3.6GB ~= 150ms one-time.

## Projected (from measured coverage, achieved BW 514 GPU / 43 CPU)
K=32 (3.6GB): 70-84% hit -> est 38-46 t/s decode. K=48 (5.4GB): 85-92% -> est 42-50 t/s. Baseline 33.9.

## Order of work
1. ggml CPU mul_mat_id sentinel skip (+ dst zeroing) -- zero impact on default paths
2. Vulkan mul_mat_id/mul_mm id guard
3. Graph split behind flag (LLAMA_STATEWISE_CACHE=profile.csv), W_cache alloc+fill at load
4. Decode-correctness A/B (logits match vs baseline within quant tolerance)
5. Bench: decode t/s vs ncmoe frontier; DOMAIN-SHIFT test (wiki->code mid-session): v1 should degrade, v2 recover
6. v2 refresh loop
Risks: Vulkan dispatch overhead at small K; sched splits from extra nodes; refresh flapping (hysteresis)

## Process note
Upstream llama.cpp does not accept undisclosed AI-generated PRs (see PR #21620 fallout). This fork develops in the open under ionizedd; any upstreaming happens with explicit disclosure and human review by the repo owner.
