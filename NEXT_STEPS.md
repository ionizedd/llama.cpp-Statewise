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
