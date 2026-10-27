"""Evaluate the shipped v2 emulators (_v2_plain.npz) on the ede-final test sets; same json format as train_v3.py."""
import json, os, sys
import numpy as np

V3 = os.path.expanduser("~/rds/ede-v3"); DATA = f"{V3}/data"; CLEAN = os.path.expanduser("~/pk_ede/retrain")
SHIP = os.path.expanduser("~/pk_ede/v2_plain")   # copies of cosmopower-organization/ede *_v2_plain.npz (2024 weights for PKNL)

def unflatten(saved):
    out, groups = {}, {}
    for kk in saved.files:
        if "." in kk:
            base, idx = kk.rsplit(".", 1)
            if idx == "n": continue
            groups.setdefault(base, {})[int(idx)] = saved[kk]
        else:
            v = saved[kk]; out[kk] = v.item() if v.ndim == 0 and v.dtype.kind in "iuf" else v
    for base, m in groups.items(): out[base] = [m[i] for i in sorted(m)]
    return out

def predict(d, X):
    x = (X - d["param_train_mean"]) / d["param_train_std"]
    W, b, a, be = d["weights_"], d["biases_"], d["alphas_"], d["betas_"]
    for i in range(len(W) - 1):
        x = x @ W[i] + b[i]
        x = (be[i] + (1 - be[i]) / (1 + np.exp(-a[i] * x))) * x
    o = x @ W[-1] + b[-1]
    if "pca_transform_matrix" in d:
        o = (o * d["pca_std"] + d["pca_mean"]) @ d["pca_transform_matrix"]
    return o * d["feature_train_std"] + d["feature_train_mean"]

for obs in sys.argv[1:]:
    out = f"{V3}/runs/v2_{obs}_eval.json"
    if os.path.exists(out):
        continue
    d = unflatten(np.load(f"{SHIP}/{obs}_v2_plain.npz", allow_pickle=False))
    names = [str(n) for n in d["parameters"]]
    if obs in ("PKNL", "PKNLCB"):
        tp, tf = f"{CLEAN}/{obs}_testing_params_clean.npz", f"{CLEAN}/{obs}_testing_processed_data_clean.npz"
    else:
        tp, tf = f"{DATA}/{obs}_testing_params.npz", f"{DATA}/{obs}_testing_processed_data.npz"
    P = np.load(tp)
    X = np.stack([P[n] for n in names], 1).astype(np.float64)
    Y = np.load(tf)["features"]
    pred = np.concatenate([predict(d, X[i:i + 5000]) for i in range(0, len(X), 5000)])
    err = np.abs(pred - Y)
    pct = {q: np.percentile(err, q, axis=0) for q in (68, 95, 99)}
    res = {"obs": obs, "ntrain": "v2", "n_test": int(len(Y)),
           "median_over_modes": {str(q): float(np.median(v)) for q, v in pct.items()},
           "max_over_modes": {str(q): float(np.max(v)) for q, v in pct.items()},
           "per_mode": {str(q): v.astype(float).round(7).tolist() for q, v in pct.items()},
           "feature_train_std": np.asarray(d["feature_train_std"], dtype=float).round(7).tolist()}
    json.dump(res, open(out, "w"))
    print(f"v2 {obs}: median-over-modes 68/95/99 = " + " / ".join(f"{res['median_over_modes'][q]:.2e}" for q in ("68", "95", "99")), flush=True)
