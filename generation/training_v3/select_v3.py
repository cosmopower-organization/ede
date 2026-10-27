"""Pick the best full-data run per observable (median over modes of the 95th pct; max as tie-break within 3%)."""
import glob, json, os
V3 = os.path.expanduser("~/rds/ede-v3"); R = f"{V3}/runs"
CMB = {"TT", "EE", "PP", "BB", "uTT", "uEE", "uBB", "rTE", "ruTE"}
best = {}
for f in glob.glob(f"{R}/*_eval.json"):
    name = os.path.basename(f)[:-10]
    if name.startswith("v2"): continue
    r = json.load(open(f))
    if "median_over_modes" not in r or ("Nall" not in name and "N103413" not in name and "N396" not in name): continue
    obs = r["obs"]
    if obs in CMB and not name.startswith("cs_"): continue      # CMB: common split only
    if obs in ("TE", "uTE"): continue                            # replaced by r_l
    m, x = r["median_over_modes"]["95"], r["max_over_modes"]["95"]
    b = best.get(obs)
    if b is None or m < 0.97 * b[1] or (m < 1.03 * b[1] and x < b[2]):
        best[obs] = (name, m, x)
lin = lambda v, o: v if o in ("rTE", "ruTE") else 100 * (10 ** v - 1)
for o in sorted(best):
    n, m, x = best[o]
    print(f"{o:7s} {n:60s} median {lin(m, o):.4f}  max {lin(x, o):.4f}")
json.dump({o: v[0] for o, v in best.items()}, open(f"{V3}/v3_selection.json", "w"), indent=1)
