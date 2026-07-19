# llama.cpp-Statewise

A research fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) exploring **stateful, bandwidth-aware execution** of MoE models on a single desktop (AMD RX 9070 XT 16GB + Ryzen 7800X3D, Windows, Vulkan).

## Thesis
Decode is memory-bound: tokens/sec = bytes touched / bandwidth of wherever those bytes live. This fork treats *placement* as a first-class, runtime-adaptive decision ? which experts live in VRAM, which stay in RAM, and when they move ? driven by measured routing state rather than static configuration.

## What is measured and proven so far
- Expert routing in Qwen3-30B-A3B is heavily skewed: top-32/128 experts per layer serve 70% (prose) to 84% (code) of routing ? but hot sets are domain-conditional (83.9% stable within-domain, 11.3% overlap across domains). Static placement cannot survive a workload shift; adaptive placement can. (`tools/expert-stats`, BASELINES.md)
- Split-expert execution (hot experts on GPU via a cache tensor with a dummy zero-slot; misses on CPU via sentinel ids in `mul_mat_id`) is numerically exact. (`tools/statewise-poc`, ggml CPU sentinel patch)
- The costs are known: ~145us per CPU<->GPU boundary, ~55us per CPU expert (Q4), ~50us per full-GPU expert layer, and a V-cache bandwidth cliff at ~90MB CPU working set on this CPU.
- A knapsack solver over the measured routing turns those constants into a per-layer placement map: predicted 44.2 t/s vs the 33.9 t/s static baseline at a 9.5GB expert budget. (`tools/statewise-poc/solve_placement.py`, `statewise_placement.json`)

## Status
Groundwork merged on branch `statewise`; the graph-level cache implementation is next (see NEXT_STEPS.md for the ordered plan, cost model, and environment notes). Baseline configs and all benchmark history live in BASELINES.md; the architecture and its amendments in DESIGN.md.

## Provenance
Built by Vern ([ionizedd](https://github.com/ionizedd)) working with Claude (Anthropic) in an extended human-AI collaboration; every number in the docs was measured on the machine described above, and design decisions trace to those measurements. This fork is a personal research lab ? nothing here is upstream-bound without explicit disclosure and human review.
