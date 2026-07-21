# FLEAGLE HANDOFF - the statewise fork is yours (2026-07-21)

From Vern + the Fable relay sessions. This work is memory-bandwidth-bound by
nature and has outgrown its origin box (RX 9070 XT 16GB + 32GB DDR5-6000).
Vern's local roadmap moves to the zeri-snn specialized-brain line; this repo
moves to you and your hardware.

## What this is

llama.cpp fork (branch `statewise`) adding an adaptive MoE expert cache for
hybrid CPU/GPU inference: hot experts live in a GPU-resident cache with a
dummy zero-slot, cold experts stay on CPU, and a split mul_mat_id path routes
each token's experts to whichever side holds them. v2 adds ONLINE adaptation:
a supervisor watches routing counters and swaps experts between decodes
through a tiny C API - the hot set follows the workload.

## Proven, all measured (same-session A/Bs; evidence in repo root + BASELINES.md)

- Split-expert execution numerically exact; v1 static cache +10.6% decode
  in-distribution (31.2 vs 28.2 t/s), OOD-neutral exactly as hot-set overlap
  predicted.
- v2 online adaptation: after a wiki->code domain switch, static hit-rate
  dies at ~16%; adaptive collapses then RECOVERS to 97% in ~1.8k tokens
  (sw_v2_*.csv + statewise_v2_recovery.png). Zero errors in ~3000 swaps.
- Adapted-map t/s (SW_DUMP_MAP workflow): stale cache is a net LOSS (28.1 vs
  29.0 none), adapted map a net win (30.2). Adaptation swings OOD -3% -> +4%.
- Identity honesty: exact-for-any-consistent-map invariant holds; greedy ties
  can still flip after ~60 tok from CPU-vs-GPU expert numerics
  (quality-neutral, same class as -ngl variation; scoped in BASELINES.md).

## Where your hardware changes the game

- If the whole model fits in VRAM, the CPU-offload premise vanishes - but the
  cache generalizes: hot experts in the fast tier, cold on a second GPU or
  host. The placement solver (tools/statewise-poc/solve_placement.py) is the
  same math with your constants. Re-measure the cost model first (BASELINES
  documents how each number was obtained).
- Big RAM unblocks the OPEN experiment we could not finish here: spec-decode
  x adapted cache. Thesis: verify batches <=8 tok stay on the split path, so
  ~97% of verify expert reads hit GPU bandwidth - MoE-hybrid spec finally
  viable. Retry recipe + flag names in NEXT_STEPS.md (our two attempts were
  invalidated by RAM paging, not by the idea).
- ByteSieve as the draft model is the research-grade angle (O(1) state =
  near-zero bandwidth scout; byte->token vocab bridge is the hard part).
  Documented in the HANDOFF MAP section of NEXT_STEPS.md. It being YOUR
  model makes this poetic.

## Boot sequence

1. README-STATEWISE.md -> DESIGN.md -> BASELINES.md -> NEXT_STEPS.md top to
   bottom (v2 blueprint, session addenda, retry recipes, quirks).
2. Code: src/llama-model.cpp (statewise_init / statewise_swap),
   src/llama-graph.cpp (split path), tools/statewise-adapt (v2 supervisor),
   tools/expert-stats (telemetry), tools/statewise-poc (+ placement solver).
3. For your rig: run expert-stats on your corpus -> solve_placement.py with
   your measured constants -> new map -> statewise-adapt validates recovery.

Windows-box-specific scripts (C:\dev) and the transport crib sheet in
NEXT_STEPS don't travel; the repo itself is self-contained.

Make it fly. - the lizard, on Vern's behalf


## P.S. - ByteSieve read-through (Fable, closing the loop)

Read bytesieve.py properly before signing off. The multi-aspect embedding
(value + UTF-8 role + intra-char phase) is the right kind of prior - roles
are spec ground truth, not heuristics - and O(1) recurrent state makes it
the cheapest possible scout. Two concrete observations for the draft angle:

1. THE VOCAB BRIDGE MAY BE ~50 LINES, NOT RESEARCH-GRADE, for greedy
   verification: llama.cpp's draft path needs only draft TOKEN IDS, not
   distributions. So: generate N bytes -> encode with the target's
   tokenizer -> submit the longest prefix that tokenizes STABLY (drop the
   possibly-mid-token tail; stability check = encode(bytes[0:k]) is a
   prefix of encode(bytes[0:k+1])'s ids). No probability alignment needed
   at temp 0.
2. For SAMPLED verify you need p_draft(token) - and it's exact and cheap:
   the chain rule over ByteSieve's per-byte distributions across the
   token's bytes, computable during the same forward pass, O(1) state
   unbothered. The "hard" bridge is bookkeeping, not math.

Rate math to check first: drafting must outrun the target several-fold in
TOKENS (so bytes/s divided by ~3-4 bytes/token). Measure that before
anything else. And the statewise synergy carries over: verify batches
<= 8 tokens ride the split path, so on an adapted cache the verify cost
stays on GPU bandwidth. Your two projects want to be one pipeline.
