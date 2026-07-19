import csv, json, sys
from collections import defaultdict

# statewise placement solver: per-layer tier choice from measured routing + POC cost model
CSV     = sys.argv[1] if len(sys.argv) > 1 else r"C:\dev\expert_counts.csv"
E_MB    = 2.36          # MB per expert (all 3 projections, Q4_K_M)
T_BOUND = 145.0         # us per CPU<->GPU boundary (POC)
T_MISS  = 55.0          # us per CPU expert (Q4 ~2.4MB @ 43GB/s)
T_GPUL  = 50.0          # us full-GPU expert layer
N_USED  = 8
BASE_MS = 29.5          # measured token time @ ncmoe20 (33.9 t/s)

counts = defaultdict(dict)
for row in csv.DictReader(open(CSV)):
    counts[int(row["layer"])][int(row["expert"])] = int(row["count"])
LAYERS = sorted(counts)

cov_cache = {}
def coverage(l, K):
    if (l, K) not in cov_cache:
        c = sorted(counts[l].values(), reverse=True)
        t = sum(c)
        cov_cache[(l, K)] = (sum(c[:K]) / t) if t else 0.0
    return cov_cache[(l, K)]

def options(l):
    opts = [(T_BOUND + N_USED * T_MISS, 0.0, "cpu", 0), (T_GPUL, 128 * E_MB, "gpu", 128)]
    for K in (8, 16, 24, 32, 48, 64, 96):
        opts.append((T_BOUND + N_USED * (1 - coverage(l, K)) * T_MISS, (K + 1) * E_MB, "cache", K))
    return opts

# calibration: fixed cost not in the layer model (attention, sampling, sync stacking)
est20 = 20 * (T_BOUND + N_USED * T_MISS) + 28 * T_GPUL
F_US  = BASE_MS * 1000 - est20

def solve(budget_mb):
    lo, hi, best = 0.0, 50.0, None
    for _ in range(80):
        lam  = (lo + hi) / 2
        pick = [min(options(l), key=lambda o: o[0] + lam * o[1]) for l in LAYERS]
        vram = sum(p[1] for p in pick)
        if vram > budget_mb: lo = lam
        else: hi, best = lam, pick
    return best

print(f"calibrated fixed cost: {F_US/1000:.1f} ms/token  (corpus: {CSV})")
print(f"{'budget':>8} {'vram':>8} {'layer-us':>9} {'est-ms':>7} {'est-t/s':>8}  tiers (gpu/cache/cpu)  boundaries")
for budget in (6000, 8000, 9500, 12000):
    pick = solve(budget)
    if pick is None: continue
    t_us  = sum(p[0] for p in pick)
    vram  = sum(p[1] for p in pick)
    tiers = [p[2] for p in pick]
    nb    = sum(1 for p in pick if p[2] != "gpu")
    est   = (t_us + F_US) / 1000
    print(f"{budget:>8} {vram:>7.0f}M {t_us:>9.0f} {est:>7.1f} {1000/est:>8.1f}  "
          f"{tiers.count('gpu')}/{tiers.count('cache')}/{tiers.count('cpu')}  {nb}")

pick = solve(9500)
out = { str(l): {"tier": p[2], "K": p[3], "hit": round(coverage(l, p[3]) if p[2] == "cache" else (1.0 if p[2] == "gpu" else 0.0), 3)} for l, p in zip(LAYERS, pick) }
json.dump(out, open(r"C:\dev\llama.cpp\statewise_placement.json", "w"), indent=1)
print("wrote statewise_placement.json (9500MB budget)")
