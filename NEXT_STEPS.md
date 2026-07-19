# NEXT_STEPS ? statewise session handoff (2026-07-19)
State: mechanism proven exact (statewise-poc), costs measured, placement SOLVED -> statewise_placement.json (17 gpu / 31 cached / 0 cpu @ 9.5GB, predicted 44.2 t/s vs 33.9 baseline).

## Implement next (in order)
1. llama-graph build_moe_ffn: env LLAMA_STATEWISE_MAP=statewise_placement.json -> for "cache" layers build split path (mul_mat_id(W_cache, x, ids_hot) + mul_mat_id(W_full, x, ids_cold), add). ids_hot/ids_cold via ggml_get_rows on per-layer i32 map tensors (hot: expert->slot else K=dummy; cold: expert->expert else -1).
2. W_cache + map tensors: alloc at load in a separate GPU buffer (like POC), fill by tensor_get/set slab copies from the CPU exps tensors. K per layer from JSON.
3. Pinning: weights' buffers drive sched correctly in llama (unlike raw POC); verify with GGML_SCHED_DEBUG=2 that cold muls stay CPU. NEVER let a CPU-weight node land on GPU (BAR reads).
4. A/B: llama-bench decode vs 33.9; then domain-shift (run on code prompts w/ wiki-built cache -> hit collapses to ~11% -> v2 motivation measured).
5. v2 online refresh: usage counters + periodic map/cache rewrite between decodes (hysteresis).
6. After placement: attack the 16.4ms fixed cost -> speculative decoding (see scratch/statewise_speculative_decoding heritage; llama.cpp has draft infra).

## Cost model constants (measured, POC + benches)
boundary 145us | cpu expert Q4 ~55us | full-GPU expert layer ~50us | expert 2.36MB | parallel sched = worse | fixed/token @ncmoe20 = 16.4ms

## Environment crib sheet (Windows box quirks)
- PowerShell MCP transport caps ~45-60s: run long jobs detached (Invoke-CimMethod Win32_Process Create -> log file -> poll). Start-Process children can hang the session (stream inheritance) - use CIM.
- Builds: C:\dev\es_pipeline.bat pattern (vcvars64 + VULKAN_SDK=C:\VulkanSDK\1.4.341.1 + winget Links PATH for ninja).
- Bench/run MoE: --no-mmap ALWAYS (mmap thrash), -dev Vulkan0 or GGML_VK_VISIBLE_DEVICES=0 (device 1 = iGPU), -t 8, -fa on; pp wants -ub 2048.
- Known bug: expert-stats crashes w/ -c 2048 -b 2048 (assert dies unflushed, 0xc0000409 in ucrtbase); use -c 512 -b 512. Guard TODO.
- git push: remote "fork" = https://github.com/ionizedd/llama.cpp-Statewise.git via C:\dev\push_fork.bat (creds cached).
- Models: G:\models. Corpora: G:\models\corpus. CSVs: C:\dev\expert_counts.csv (wiki), C:\dev\es_code\ (code), es_wA/es_wB (stability halves).

## Numbers that define the project
tg: GPU-8B 109.8 | MoE static frontier 16.6->33.9 | pp fixed: 707-2280 | skew: cov32 70%(wiki)/84%(code) | hot-set overlap: 83.9% within / 11.3% cross | target: 44+ t/s then spec-decode on the 16.4ms.

## Session addendum — 2026-07-19 "all out" research pass (2 sonnet agents + spec A/B)
CORRECTION: the spec-decode heritage folder is OUTSIDE this repo at C:\Users\PC\.gemini\antigravity-ide\scratch\statewise_speculative_decoding (local machine only).

### Speculative decoding (upstream infra verified: docs/speculative.md, tools/server/bench/speed-bench)
Measured A/B, novel-codegen prompt, 256 tok, ncmoe20: baseline 32.9 t/s | ngram-simple 30.7 (WORSE) | ngram-map-k 33.1 (neutral). Draft acceptance only 6.25% - fresh generation does not repeat history. ngram pays off on LONG REPETITIVE contexts (editing loops, agent transcripts, RAG re-quotes); re-test there post-placement. Ceiling for novel gen = EAGLE-3 draft (verify HF checkpoints for Qwen3-30B-A3B exist; costs VRAM vs the 9.5GB placement budget - solve jointly). Old seed/codebook draft approach: honorably retired - superseded by upstream ngram-map-k (agent audit: PoC numbers were simulated w/ train/test leakage).

### Agent verdicts
- GFNI: KILLED with prejudice. ggml uses AND+SHIFT for byte-aligned nibble extraction (already optimal, more ports, 1cy vs GFNI 3cy); TQ1_0 is base-3 arithmetic (not GF(2)-linear - structurally impossible); decode is bandwidth-bound so ALU wins round to zero; ngram hash is a 3-line LCG, not hot. GFNI appears NOWHERE in upstream (0 code-search hits) - correctly so.
- Successor lead from the GFNI autopsy: profile the missing CPU bandwidth (43 of ~65GB/s) - prefetch/access-pattern in the MoE expert gather. Memory-subsystem problem, not instruction selection.
- NEW #1 boundary-tax lead (from Vern's gpu_step_acceleration.hip pattern): profile ggml-vulkan submission granularity per decode token (one vkQueueSubmit or many?). If many -> batch into fewer submissions. Cheap to profile; decides itself.
- HARD CONSTRAINT (measured on THIS 7800X3D, bitshredder results_v2): sustained CPU read BW ~77GB/s up to ~90MB working set, then V-CACHE CLIFF -> ~44GB/s (-43%). Keep per-token CPU-touched expert bytes well under 90MB. Current worst case (miss set) is fine; matters if cold tier grows.
- Not worth pursuing (agent audit): hdc_bvh (accuracy collapse), crystal_brain/seedpack (different research program), weight_distillation/seed proofs (Vern already self-disproved via seed_entropy_proof - respect).
