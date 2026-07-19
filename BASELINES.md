# BASELINES — 2026-07-18 · upstream 571d0d5 · MSVC 19.44 · Vulkan + AVX512 flags
HW: Ryzen 7 7800X3D (8C/16T) · DDR5-6000 2ch (~96 GB/s theor.) · RX 9070 XT 16GB (~645 GB/s) · Win11 26200 · Adrenalin 32.0.31021.5001 · models on G: (SN770)

## Qwen3-8B Q4_K_M (4.68 GiB)
| config        | pp512 | tg128 | eff. BW |
|---------------|-------|-------|---------|
| GPU ngl99     | 3080  | 109.8 | ~514 GB/s (80% peak) |
| CPU ngl0 t8   | 1358* | 9.2   | ~43 GB/s (*pp GPU-assisted) |

## Qwen3-30B-A3B-Instruct-2507 Q4_K_M (17.28 GiB, 48 layers) · ngl99 · fa on · t8 · dev Vulkan0
| ncmoe (expert layers on CPU) | pp512 | tg128 |
|------|--------|-------|
| 48   | 74.6±37  | 16.61 |
| 40   | 93.6±22  | 20.49 |
| 32   | 122±17   | 25.64 |
| 26   | 183±78   | 29.04 |
| 20   | 227±103  | 33.85 |

Notes:
- tg variance tiny, pp variance huge (page-cache pressure: 17GB model + host buffers vs 32GB RAM). Rerun pp after reboot for clean numbers.
- Static frontier: 16.6 -> 33.9 t/s. VRAM edge not yet found (try ncmoe 16/12, or --fit-target).
- Hybrid pp512 (75-227) is disproportionately bad vs dense GPU pp (3080) -> investigate -ub/-b batching of CPU expert matmuls.
- FORK TARGETS: (1) per-expert routing telemetry -> measure hot/cold skew; (2) dynamic hot-expert VRAM cache to beat static frontier; (3) zero-bandwidth draft spec decode (statewise_speculative_decoding heritage); (4) thread/kernel tuning to lift CPU 43 GB/s eff toward ~65; (5) DirectStorage tier later.

## Expert routing telemetry — 2026-07-18, Qwen3-30B-A3B-2507, wikitext-2, 49,152 tokens (96x512 chunks)
Tool: tools/expert-stats (cb_eval hook on ffn_moe_topk, stride-corrected view read)
MEAN coverage if top-K experts/layer were VRAM-cached:
| K cached | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|
| hit %  | 31.6 | 48.1 | 60.4 | 70.4 | 85.2 | 93.9 |
| uniform| 6.25 | 12.5 | 18.8 | 25.0 | 37.5 | 50.0 |
| VRAM   | 0.9GB| 1.8GB| 2.7GB| 3.6GB| 5.4GB| 7.3GB|
K90: avg 55.6 (44-83 across layers) vs 116 uniform. Skew is real: ~2.8x uniform at K=32.
KEY: static layer-split (ncmoe) hit-rate == uniform curve. Dynamic per-expert cache at 8.5GB ~= 96% vs static 58%.
Projection at ncmoe-20 VRAM budget: expert-read eff. BW ~92 -> ~357 GB/s => est. 40-50 t/s vs 33.9 static (pre spec-decode).
Caveats: wikitext-only (validate code/chat corpora), 512-tok contexts, known assert w/ -b 2048 (guard TODO).
Full per-layer data: expert_counts.csv (layer,expert,count).

## Hot-set stability — 2026-07-19 (the design-deciding numbers)
Top-32/layer overlap: WITHIN-domain (wikitext A/B halves, 24k tok each) = 83.9%. CROSS-domain (wikitext vs C++ code) = 11.3% (random = 25% -> ANTI-correlated: true domain specialists).
Code corpus is even skewier than prose: cov32 = 84.4% (vs 70.4), K90 avg 42.2 (vs 55.6).
VERDICT: static/offline expert placement cannot survive workload shifts; the VRAM expert cache must be ONLINE-ADAPTIVE (usage counters + per-layer budgets + hysteresis). Domain switch re-warm cost ~3.6GB over ReBAR ~= 150ms one-time -> amortized in dozens of tokens. This is the statewise thesis, now with data.
