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

## V2 BLUEPRINT — online adaptation (the core deliverable, post-v1 validation)
Goal: make the OOD row of the v1 table (25.8) climb toward the in-distribution row (31.2+) by letting the hot set follow the workload.
Architecture: supervisor OUTSIDE the graph (common/ or server layer), core stays thin:
1. Counters: cb_eval hook on ffn_moe_topk (proven in tools/expert-stats) accumulates per-layer expert counts in a ring window (~2048 tokens).
2. Small C API on llama_model: llama_statewise_swap(model, layer, slot, new_expert_id) -> copies one expert slab (2.4MB) into the cache slot via tensor_set + updates both map tensors (2 floats). All between decodes; no graph changes.
3. Policy (hysteresis): every 256 tokens, per layer: candidate = hottest uncached expert, victim = coldest cached slot; swap only if count_cand > 1.5x count_victim; cap ~4 swaps/layer/refresh. Full domain shift re-warms in ~2-3k tokens (~150ms total copy cost, amortized).
4. Instrument: log hit-rate per window (counts vs map) -> the v2 success metric IS the live hit-rate curve recovering after a domain switch.
5. Test: the wiki->code mid-session switch. v1 static: hit rate collapses and stays down. v2: collapses then recovers. Chart it.
Also queued: solver to charge pp compute-buffer headroom (C config pp dip 538->380); correctness note: split-path pp still guarded off (n_tokens<=8).

## HANDOFF MAP — for relay sessions (Fable does core work, Sonnet dispatches for builds/benches/polling)
DIRECTORIES:
- C:\dev\llama.cpp        THE FORK. branch statewise -> github.com/ionizedd/llama.cpp-Statewise. Docs: README-STATEWISE.md, DESIGN.md, BASELINES.md, NEXT_STEPS.md (this file). Artifacts: statewise_map.txt, statewise_placement.json. Tools: tools/expert-stats, tools/statewise-poc (+solve_placement.py).
- C:\dev\                 scripts (build_sw.bat, smoke_sw.bat, triage.bat, wiki_ab.bat, push_fork.bat), logs (*.log), python patch scripts (patch_*.py - the reliable way to edit source on this box), telemetry CSVs (expert_counts.csv wiki; es_code\, es_wA\, es_wB\).
- G:\models\              GGUFs (Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf = the model) + corpus\ (wikitext, code.txt).
- C:\Users\PC\.gemini\antigravity-ide\scratch\   Vern's prior-ideas archive (audited; see session addendum above).
- C:\Users\PC\byte-llm-fork\                     fleagle's ByteSieve-GLA: ~90M byte-level LM, O(1) recurrent state, no KV cache, tokenizer-free, has specdec\ subdir. INTEGRATION ANGLES: (a) draft model for speculative decoding (near-zero bandwidth; needs byte->token vocab bridge - nontrivial, research-grade); (b) port the arch into this fork as a GGUF-runnable model (llama.cpp already runs GLA-family archs). Scope with Vern/fleagle first.
BOOT SEQUENCE FOR A FRESH FABLE: (1) nyzkh-memory recall "statewise"; (2) read this file top to bottom; (3) git log --oneline -10 in the fork; (4) current work = V2 BLUEPRINT section above. Environment quirks are in the crib sheet section. Editing rule: use python patch scripts with exact-match+count asserts (see C:\dev\patch_*.py), never PS here-string source edits.
STATE AT HANDOFF (2026-07-20): v1 cache VALIDATED (+10.6% in-distribution, token-identical, OOD-neutral as designed). v2 online adaptation is next: blueprint above, all steps concrete. Same-session baselines only (box variance ~±3 t/s day to day).


## V2 VALIDATED - 2026-07-21 (Fable session, new account)
RESULT: online adaptation works. wiki->code switch @4096 tok: static hit 90.3% -> 17.6% dip -> stays 16.3%; adaptive 95.6% -> 19.2% dip -> >93% by +1.8k tok -> 97.3% steady. Adaptive beats static IN-distribution too (95.6 vs 90.3 - the policy keeps tuning the solver's wiki map). Swaps ~120/epoch during re-warm (cap 4x31=124), 10-40 steady churn, 0 errors in ~3000 swap calls (shadow map never diverged from live). Same-session pair, epoch=256 window=8 hyst=1.5. Evidence at repo root: sw_v2_static.csv, sw_v2_adapt.csv, statewise_v2_recovery.png.
IMPLEMENTED: llama_statewise_swap + llama_statewise_layer_k (public C API in llama.h; impl llama-model.cpp: readback map_hot -> victim scan -> 3 slab copies -> 4 map float writes; slab-first ordering keeps the map consistent at every instant, between-decodes only). tools/statewise-adapt supervisor: shadow map seeded from LLAMA_STATEWISE_MAP, per-epoch counters via cb_eval on ffn_moe_topk, hysteresis policy, per-epoch hit-rate CSV. Config via SW_* env (see file header).
EXACTNESS NOTE: correctness rests on the v1 invariant (split path numerically exact for ANY consistent map) + swap keeping maps consistent; a direct token-identity smoke during generation is deferred to cli integration.
NEXT: (1) wire the supervisor into llama-cli/llama-server (adaptation during real generation -> live t/s recovery on domain shift; counters cheap, swap sites = between decodes in the main loop). (2) steady-state churn 10-40 swaps/epoch: hyst 2.0 or per-slot cooldown could halve it - measure, don't guess. (3) then the 16.4ms fixed cost: spec-decode (ngram-map-k on long repetitive ctx; verify EAGLE-3 Qwen3-30B-A3B checkpoints; ByteSieve draft angle - HANDOFF MAP above).
SESSION QUIRKS (nyzkh pwsh7 MCP transport - additions to crib sheet): native-exe stdout is INVISIBLE (PS cmdlets print fine). Side-effect commands: invoke direct (& 'C:\path\exe' args) and VERIFY via Select-String/Get-Content, don't trust output. Long jobs: CIM Win32_Process Create 'cmd /c script.bat > log 2>&1' -> poll log (proven for builds + model runs). cmd /c with BOTH a quoted exe path AND a redirect in one string silently no-ops - don't. git log --output=FILE bypasses the capture hole. Patch scripts MUST carry an idempotency guard: replacements that contain their own anchors double-apply on re-run (happened this session; patch_v2_core.py has the guard pattern).


## V2 ADDENDUM - adapted-map t/s + identity scope (2026-07-21, same session)
SW_DUMP_MAP added to statewise-adapt: dumps the adapted shadow map in statewise_init format -> any tool can load it via LLAMA_STATEWISE_MAP. Workflow proven: re-warm on code (4096 tok, 97.2% hit) -> dump -> llama-cli bench with dumped map. RESULT (code prompt, tg256): none 29.0 / stale-wiki 28.1 / code-adapted 30.2 t/s. Stale cache = net loss, adapted = net win; +7.5% swing. See BASELINES.md table + identity note (v1 token-identity is length-scoped: greedy ties can flip after ~60 tok from CPU/GPU expert numerics - quality-neutral, not a bug).
NEXT for t/s: supervisor-in-cli/server = live adaptation during generation (map-dump workflow is the static approximation of it); then the 16.4ms fixed-cost attack (spec-decode leads in HANDOFF MAP).


## SPEC-DECODE ATTEMPT - status + retry recipe (2026-07-21, late session)
Qwen3-0.6B-Q8 draft downloaded (G:\models, 610MB, same Qwen3 vocab) + --spec-type draft-simple. Flags in this build: --spec-draft-n-max/-n-min, -md, -ngld.
FIRST run 20.6 t/s vs 30.2 no-draft: INVALID - RAM over-commit (pagefile peak 25.8GB, 80k pages-in/s DURING decode; Vern spotted sustained 500MB/s on G:). Machine truth: 32GB minus ~13GB baseline cannot hold 17.7GB no-mmap + draft overhead. mmap retry pair equally disk-bound at 99% RAM (base 21.9 / spec 17.7 - both streaming weights off SATA G:). VERDICT: spec on this rig is UNMEASURED, not dead. Blocked purely on RAM headroom.
CORE FIX SHIPPED: statewise_init now no-ops for models without MoE experts (sw_any_experts guard) - previously LLAMA_STATEWISE_MAP killed ANY dense draft/aux model load in the same process.
RETRY RECIPE: confirm >=24GB free idle, add --no-mmap back to C:\dev\spec_fair.bat, run - valid pair in ~4 min. THESIS unchanged and still worth the retry: verify batches <=8 tok stay on the split path so ~97% of verify expert reads hit the GPU cache -> MoE-hybrid spec finally viable; n-max 3 variant in C:\dev\spec_probe.bat.
HARDWARE (ranked for this roadmap): (1) 64GB RAM - biggest lever, unlocks spec + kills the whole paging failure class. (2) move GGUFs to the idle NVMe (Disk 2, 0% active in Task Manager) - SATA G: caps ~550MB/s, this makes every load and any mmap eviction ~6x cheaper. (3) cloud GPU: NOT for inference (local-first is the whole point); revisit only to train our own draft (ByteSieve distill / acceptance-tuned tiny model, few hours rental).


## TORCH PASSED (2026-07-21, end of day)
Project handed to fleagle - this work is bandwidth-bound and his hardware fits it (more GPUs, much more RAM/VRAM). Read FLEAGLE_HANDOFF.md at repo root first; it maps everything including the open spec-decode experiment and where bigger VRAM changes the placement math. Vern's local roadmap moves to the zeri-snn specialized-brain line (separate tree, out of this repo). Future Fable/Sonnet sessions on Vern's box: recall "statewise" for history; active development continues with fleagle.
