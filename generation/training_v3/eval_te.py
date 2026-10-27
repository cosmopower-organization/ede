"""TE strategies compared in units of the cosmic-variance error sigma_l = sqrt((C_TT C_EE + C_TE^2)/(2l+1)).

Test samples: common-split test runs that are also in the original TE test set (so v2 TE is not
evaluated on its training samples), random 20k of them. r_l emulators: TE = tanh(r) sqrt(TT EE) with the
v3 TT and EE emulators (and, separately, with the CLASS TT and EE to isolate the r error).
"""
import json, os
import numpy as np
V3 = "/home/cmbagent/rds/ede-v3"; CS = f"{V3}/data_cs"; R = f"{V3}/runs"
exec(open(f"{V3}/eval_v2.py").read().split("for obs in sys.argv[1:]:")[0])   # unflatten, predict (v2 plain, with PCA)

def load_v3(name):
    return np.load(f"{R}/{name}.npz", allow_pickle=True)["arr_0"].flatten()[0]

cp = np.load(f"{CS}/common_testing_params.npz"); keys = sorted(cp.files)
cand = {r.tobytes(): i for i, r in enumerate(np.stack([cp[k] for k in keys], 1).astype(np.float64))}
ot = np.load(f"{DATA}/TE_testing_params.npz")
in_both = sorted(i for r in np.stack([ot[k] for k in keys], 1).astype(np.float64) if (i := cand.get(r.tobytes())) is not None)
rows = np.sort(np.random.default_rng(0).choice(in_both, min(20000, len(in_both)), replace=False))
print(f"common-split test runs also in the original TE test set: {len(in_both)}; using {len(rows)}", flush=True)

def feat(o): return np.load(f"{CS}/{o}_testing_processed_data.npz", mmap_mode="r")["features"][rows].astype(np.float64)
TE, TT, EE = feat("TE"), feat("TT"), feat("EE")
n = TE.shape[1]; ells = np.arange(2, 2 + n)
sig = np.sqrt((10 ** (TT + EE) + TE ** 2) / (2 * ells + 1))
X_all = {k: cp[k][rows] for k in keys}

def pred(d):
    X = np.stack([X_all[str(p)] for p in d["parameters"]], 1).astype(np.float64)
    return np.concatenate([predict(d, X[i:i + 2000]) for i in range(0, len(X), 2000)])

TT_hat = pred(load_v3("cs_TT_Nall_h1024x1024x1024x1024_pca0"))
EE_hat = pred(load_v3("cs_EE_Nall_h1024x1024x1024x1024_pca0"))
cands = {"v2 (PCA 64, 4x512)": ("v2", None),
         "PCA 64, 4x1024": ("TE", "cs_TE_Nall_h1024x1024x1024x1024_pca64"),
         "PCA 128, 4x1024": ("TE", "cs_TE_Nall_h1024x1024x1024x1024_pca128"),
         "PCA 256, 4x1024": ("TE", "cs_TE_Nall_h1024x1024x1024x1024_pca256"),
         "r_l direct, 4x512": ("r", "cs_rTE_Nall_h512x512x512x512_pca0"),
         "r_l direct, 4x1024": ("r", "cs_rTE_Nall_h1024x1024x1024x1024_pca0"),
         "r_l PCA 128, 4x1024": ("r", "cs_rTE_Nall_h1024x1024x1024x1024_pca128")}
ranges = [(2, 30), (30, 500), (500, 3000), (3000, n + 2)]
out = {}
for label, (kind, name) in cands.items():
    if kind == "v2":
        d = unflatten(np.load(f"{SHIP}/TE_v2_plain.npz", allow_pickle=False)); te = pred(d)
    elif kind == "TE":
        te = pred(load_v3(name))
    else:
        r = np.tanh(pred(load_v3(name)))
        te_true_amp = r * np.sqrt(10 ** (TT + EE))
        e_r = np.abs(te_true_amp - TE) / sig
        out[label + " [CLASS TT,EE]"] = {"p95": np.percentile(e_r, 95, axis=0).tolist(), "p68": np.percentile(e_r, 68, axis=0).tolist()}
        te = r * np.sqrt(10 ** (TT_hat + EE_hat))
    e = np.abs(te - TE) / sig
    out[label] = {"p95": np.percentile(e, 95, axis=0).tolist(), "p68": np.percentile(e, 68, axis=0).tolist()}
json.dump(out, open(f"{V3}/te_comparison.json", "w"))
print(f"\n|dTE| / sigma_l (cosmic variance per l), 95th percentile: max over l in ranges {ranges}; [68th pct median over all l]")
for label, v in out.items():
    p95 = np.array(v["p95"])
    s = "  ".join(f"{p95[(ells >= lo) & (ells < hi)].max():.4f}" for lo, hi in ranges)
    print(f"  {label:42s} {s}   [{np.median(v['p68']):.5f}]")
