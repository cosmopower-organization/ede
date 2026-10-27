"""TT scaling plot: test error vs training-set size, and per-ell error curves; v2 reference on a 20k subsample."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OBS = sys.argv[1] if len(sys.argv) > 1 else "TT"
V3 = "/home/cmbagent/rds/ede-v3"; R = f"{V3}/runs"
exec(open(f"{V3}/eval_v2.py").read().split("for obs in sys.argv[1:]:")[0])   # unflatten, predict

def frac(x): return 100 * (10 ** np.asarray(x) - 1)

# v2 reference, 20k random samples of its own (original) test set
v2f = f"{R}/v2sub_{OBS}_eval.json"
if not os.path.exists(v2f):
    d = unflatten(np.load(f"{SHIP}/{OBS}_v2_plain.npz", allow_pickle=False))
    P = np.load(f"{DATA}/{OBS}_testing_params.npz"); F = np.load(f"{DATA}/{OBS}_testing_processed_data.npz", mmap_mode="r")["features"]
    idx = np.sort(np.random.default_rng(0).choice(len(F), 20000, replace=False))
    X = np.stack([P[n] for n in [str(n) for n in d["parameters"]]], 1)[idx].astype(np.float64)
    err = np.abs(np.concatenate([predict(d, X[i:i + 2000]) for i in range(0, len(X), 2000)]) - F[idx])
    json.dump({str(q): np.percentile(err, q, axis=0).tolist() for q in (68, 95)}, open(v2f, "w"))
v2 = json.load(open(v2f))

runs = {}
for n in ("25000", "50000", "100000", "200000", "all"):
    f = f"{R}/cs_{OBS}_N{n}_h512x512x512x512_pca0_eval.json"
    if os.path.exists(f): runs[n] = json.load(open(f))
wide = json.load(open(f"{R}/cs_{OBS}_Nall_h1024x1024x1024x1024_pca0_eval.json"))
ells = np.arange(2, 2 + len(v2["95"]))

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2))
Ns = np.array([runs[n]["ntrain"] for n in runs])
for q, ls in (("95", "-"), ("68", "--")):
    a.loglog(Ns, [frac(np.median(runs[n]["per_mode"][q])) for n in runs], "o" + ls, c="C0", label=f"4x512, {q}th pct, median over ell")
    a.loglog(Ns, [frac(np.max(runs[n]["per_mode"][q])) for n in runs], "s" + ls, c="C1", label=f"4x512, {q}th pct, max over ell")
    a.plot(wide["ntrain"], frac(np.median(wide["per_mode"][q])), "*", ms=14, c="C0", mec="k")
    a.plot(wide["ntrain"], frac(np.max(wide["per_mode"][q])), "*", ms=14, c="C1", mec="k")
    a.axhline(frac(np.median(v2[q])), c="C0", ls=":" if q == "95" else "-.", lw=1)
    a.axhline(frac(np.max(v2[q])), c="C1", ls=":" if q == "95" else "-.", lw=1)
a.plot([], [], "*", ms=12, c="grey", mec="k", label="4x1024, all data")
a.plot([], [], ":", c="grey", label="v2 (250k samples, 95th: dotted, 68th: dash-dot)")
a.set_xlabel("training samples"); a.set_ylabel(r"$|C_\ell^{\rm emu}/C_\ell^{\rm CLASS}-1|$ [%]")
a.set_title(f"{OBS}: test error vs training-set size (common-split test set, 99k spectra)")
a.grid(which="both", alpha=0.2); a.legend(fontsize=7.5, loc="upper right")
for i, n in enumerate(runs):
    b.loglog(ells, frac(runs[n]["per_mode"]["95"]), lw=0.8, c=plt.cm.viridis(i / max(1, len(runs) - 1)), label=f"4x512, N={runs[n]['ntrain']}")
b.loglog(ells, frac(wide["per_mode"]["95"]), lw=0.9, c="C3", label=f"4x1024, N={wide['ntrain']}")
b.loglog(ells, frac(v2["95"]), lw=0.9, c="k", ls="--", label="v2")
b.set_xlabel(r"$\ell$"); b.set_ylabel("95th percentile [%]"); b.set_title(f"{OBS}: 95th percentile of the error vs $\\ell$")
b.grid(which="both", alpha=0.2); b.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{V3}/scaling_{OBS}.png", dpi=110)
print("saved; v2 median/max 95th:", frac(np.median(v2["95"])), frac(np.max(v2["95"])))
