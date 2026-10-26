"""Export retrained networks in the three on-disk formats of cosmopower-organization/ede:
<name>_v2.npz (pickled dict in arr_0), <name>_v2.pkl (cosmopower 15-item list), <name>_v2_plain.npz (pickle-free)."""
import pickle, sys
import numpy as np

src_dir, out_dir = sys.argv[1], sys.argv[2]
for name in ("PKNL", "PKNLCB"):
    d = np.load(f"{src_dir}/{name}_v2_retrained.npz", allow_pickle=True)["arr_0"].flatten()[0]
    W = [np.asarray(w, np.float32) for w in d["weights_"]]
    b = [np.asarray(x, np.float32) for x in d["biases_"]]
    a = [np.asarray(x, np.float32) for x in d["alphas_"]]
    be = [np.asarray(x, np.float32) for x in d["betas_"]]
    pm, ps = np.asarray(d["param_train_mean"], np.float32), np.asarray(d["param_train_std"], np.float32)
    fm, fs = np.asarray(d["feature_train_mean"], np.float32), np.asarray(d["feature_train_std"], np.float32)
    params = [str(p) for p in d["parameters"]]
    modes = np.asarray(d["modes"], np.int64)
    n_hidden = [int(n) for n in d["n_hidden"]]
    arch = [int(n) for n in d["architecture"]]
    n_layers, n_par, n_modes = int(d["n_layers"]), int(d["n_parameters"]), int(d["n_modes"])

    full = {"weights_": W, "biases_": b, "alphas_": a, "betas_": be,
            "param_train_mean": pm, "param_train_std": ps, "feature_train_mean": fm, "feature_train_std": fs,
            "n_parameters": n_par, "parameters": params, "n_modes": n_modes, "modes": modes,
            "n_hidden": n_hidden, "n_layers": n_layers, "architecture": arch}
    np.savez(f"{out_dir}/{name}_v2.npz", full)

    with open(f"{out_dir}/{name}_v2.pkl", "wb") as f:
        pickle.dump([W, b, a, be, pm, ps, fm, fs, n_par, params, n_modes, modes, n_hidden, n_layers, arch], f)

    flat = {}
    for field, lst in (("weights_", W), ("biases_", b), ("alphas_", a), ("betas_", be)):
        flat[f"{field}.n"] = np.int64(len(lst))
        for i, x in enumerate(lst):
            flat[f"{field}.{i}"] = x
    flat.update(param_train_mean=pm, param_train_std=ps, feature_train_mean=fm, feature_train_std=fs,
                n_parameters=np.int64(n_par), parameters=np.array(params), n_modes=np.int64(n_modes), modes=modes,
                n_hidden=np.array(n_hidden, np.int64), n_layers=np.int64(n_layers), architecture=np.array(arch, np.int64))
    np.savez_compressed(f"{out_dir}/{name}_v2_plain.npz", **flat)
    print("exported", name)
