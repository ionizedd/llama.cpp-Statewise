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
