"""Export the selected ede-v3 networks in the three formats of cosmopower-organization/ede:
<name>_v3.npz (dict in arr_0), <name>_v3.pkl (cosmopower list), <name>_v3_plain.npz (pickle-free)."""
import json, os, pickle, sys
import numpy as np

V3 = os.path.expanduser("~/rds/ede-v3"); R = f"{V3}/runs"; OUT = sys.argv[1]
FOLDER = {"TT": "TTTEEE", "EE": "TTTEEE", "rTE": "TTTEEE", "uTT": "TTTEEE", "uEE": "TTTEEE", "ruTE": "TTTEEE",
          "BB": "BB", "uBB": "BB", "PP": "PP", "PKL": "PK", "PKLCB": "PK", "PKNL": "PK", "PKNLCB": "PK",
          "DER": "derived-parameters", "DAZ": "growth-and-distances", "HZ": "growth-and-distances", "S8Z": "growth-and-distances"}
sel = json.load(open(f"{V3}/v3_selection.json"))
for obs, run in sorted(sel.items()):
    d = np.load(f"{R}/{run}.npz", allow_pickle=True)["arr_0"].flatten()[0]
    f32 = lambda a: np.asarray(a, np.float32)
    W, b, a, be = ([f32(x) for x in d[k]] for k in ("weights_", "biases_", "alphas_", "betas_"))
    pm, ps, fm, fs = (f32(d[k]) for k in ("param_train_mean", "param_train_std", "feature_train_mean", "feature_train_std"))
    params = [str(p) for p in d["parameters"]]; modes = np.asarray(d["modes"], np.int64)
    n_hidden = [int(n) for n in d["n_hidden"]]; arch = [int(n) for n in d["architecture"]]
    n_layers, n_par, n_modes = int(d["n_layers"]), int(d["n_parameters"]), int(d["n_modes"])
    full = {"weights_": W, "biases_": b, "alphas_": a, "betas_": be, "param_train_mean": pm, "param_train_std": ps,
            "feature_train_mean": fm, "feature_train_std": fs, "n_parameters": n_par, "parameters": params,
            "n_modes": n_modes, "modes": modes, "n_hidden": n_hidden, "n_layers": n_layers, "architecture": arch}
    os.makedirs(f"{OUT}/{FOLDER[obs]}", exist_ok=True)
    base = f"{OUT}/{FOLDER[obs]}/{obs}_v3"
    np.savez(base + ".npz", full)
    with open(base + ".pkl", "wb") as fh:
        pickle.dump([W, b, a, be, pm, ps, fm, fs, n_par, params, n_modes, modes, n_hidden, n_layers, arch], fh)
    flat = {}
    for field, lst in (("weights_", W), ("biases_", b), ("alphas_", a), ("betas_", be)):
        flat[f"{field}.n"] = np.int64(len(lst))
        for i, x in enumerate(lst):
            flat[f"{field}.{i}"] = x
    flat.update(param_train_mean=pm, param_train_std=ps, feature_train_mean=fm, feature_train_std=fs,
                n_parameters=np.int64(n_par), parameters=np.array(params), n_modes=np.int64(n_modes), modes=modes,
                n_hidden=np.array(n_hidden, np.int64), n_layers=np.int64(n_layers), architecture=np.array(arch, np.int64))
    np.savez_compressed(base + "_plain.npz", **flat)
    print(f"{obs:7s} <- {run}  arch {arch[1:-1]}  -> {FOLDER[obs]}/{obs}_v3", flush=True)
