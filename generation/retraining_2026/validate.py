"""Validation of PKNL / PKNLCB emulators on the cleaned held-out test sets: shipped v2 vs retrained."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/cmbagent/pk_ede/retrain"
SHIPPED = "/home/bbolliet/class_sz_data_directory/ede/PK/{}_v2.npz"
NEW = D + "/{}_v2_retrained.npz"
k = np.geomspace(5e-4, 10., 1000)

def load_net(fn):
    d = np.load(fn, allow_pickle=True)["arr_0"].flatten()[0]
    return d

def predict(d, P):
    x = (np.stack([P[n] for n in d["parameters"]], 1) - np.asarray(d["param_train_mean"])) / np.asarray(d["param_train_std"])
    W, b, a, be = d["weights_"], d["biases_"], d["alphas_"], d["betas_"]
    out = []
    for s in range(0, len(x), 20000):
        h = x[s:s + 20000]
        for i in range(len(W) - 1):
            h = h @ np.asarray(W[i]) + np.asarray(b[i])
            h = (np.asarray(be[i]) + (1. - np.asarray(be[i])) / (1. + np.exp(-np.asarray(a[i]) * h))) * h
        h = h @ np.asarray(W[-1]) + np.asarray(b[-1])
        out.append(h * np.asarray(d["feature_train_std"]) + np.asarray(d["feature_train_mean"]))
    return np.concatenate(out)

zbins = [(0, 1), (1, 3), (3, 5), (5, 20.5)]
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
summary = []
for row, var in enumerate(["PKNL", "PKNLCB"]):
    P = dict(np.load(f"{D}/{var}_testing_params_clean.npz"))
    F = np.load(f"{D}/{var}_testing_processed_data_clean.npz")["features"]
    z = P["z_pk_save_nonclass"]
    errs = {}
    for lab, fn in [("shipped v2", SHIPPED.format(var)), ("retrained", NEW.format(var))]:
        pred = predict(load_net(fn), P)
        errs[lab] = 100 * np.abs(10 ** (pred - F) - 1)   # |emu/true - 1| in %
    # percentile bands vs k
    for col, lab in enumerate(["shipped v2", "retrained"]):
        ax = axes[row, col]
        e = errs[lab]
        for q, c, al in [(99, "salmon", .8), (95, "red", .7), (68, "darkred", 1.)]:
            ax.fill_between(k, 0, np.percentile(e, q, axis=0), color=c, alpha=al, label=f"{q}%")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 3e3)
        ax.grid(which="both", alpha=.2)
        ax.set_title(f"{var} {lab}  (N_test={len(z)})")
        ax.set_xlabel(r"$k\ [\mathrm{Mpc}^{-1}]$"); ax.set_ylabel(r"$|P_{\rm emu}/P_{\rm CLASS}-1|$ [%]")
        ax.legend(loc="upper left", frameon=False)
    # 95th percentile vs z bin at k=1 and k=10
    ax = axes[row, 2]
    for lab, ls in [("shipped v2", "--"), ("retrained", "-")]:
        e = errs[lab]
        for kk, c in [(0.1, "C0"), (1., "C1"), (10., "C2")]:
            j = np.argmin(abs(k - kk))
            v = [np.percentile(e[(z >= lo) & (z < hi), j], 95) for lo, hi in zbins]
            ax.plot(range(len(zbins)), v, ls=ls, c=c, marker="o", label=f"{lab}, k={kk:g}")
    ax.set_xticks(range(len(zbins)), [f"{lo}-{hi:g}" for lo, hi in zbins]); ax.set_xlabel("z bin")
    ax.set_yscale("log"); ax.set_ylabel("95th percentile [%]"); ax.grid(which="both", alpha=.2)
    ax.set_title(f"{var}: 95th percentile by redshift"); ax.legend(fontsize=7, ncol=2, frameon=False)
    for lab in ["shipped v2", "retrained"]:
        e = errs[lab]
        for kk in (0.01, 0.1, 1., 10.):
            j = np.argmin(abs(k - kk))
            summary.append((var, lab, kk, np.percentile(e[:, j], 68), np.percentile(e[:, j], 95), np.percentile(e[:, j], 99),
                            np.percentile(e[z < 3, j], 95)))
plt.tight_layout()
plt.savefig(f"{D}/pknl_validation.png", dpi=120)
print(f"{'emulator':8s} {'version':11s} {'k':>5s} {'68%':>8s} {'95%':>8s} {'99%':>8s} {'95% z<3':>8s}")
for s in summary:
    print(f"{s[0]:8s} {s[1]:11s} {s[2]:5g} {s[3]:8.3f} {s[4]:8.3f} {s[5]:8.3f} {s[6]:8.3f}")
