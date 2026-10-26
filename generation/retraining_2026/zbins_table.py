"""Test-set accuracy of the retrained non-linear emulators by redshift bin (markdown table)."""
import numpy as np, sys
nets = {"PKNL": sys.argv[1], "PKNLCB": sys.argv[2]}
sys.argv = ["x"]
exec(open("validate.py").read().split("zbins =")[0])   # load_net, predict, k, D
ks = (0.01, 0.1, 1., 10.)
idx = [np.argmin(abs(k - kk)) for kk in ks]
zb = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20)]
for var, fn in nets.items():
    P = dict(np.load(f"{D}/{var}_testing_params_clean.npz"))
    F = np.load(f"{D}/{var}_testing_processed_data_clean.npz")["features"]
    z = P["z_pk_save_nonclass"]
    e = 100 * np.abs(10 ** (predict(load_net(fn), P) - F) - 1)
    print(f"\n{var} ({fn}), N_test = {len(z)}, z from {z.min():.2f} to {z.max():.2f}")
    print("| z | N_test | 68% (k=0.01 / 0.1 / 1 / 10) | 95% (k=0.01 / 0.1 / 1 / 10) |")
    print("|---|---:|---|---|")
    for lo, hi in zb + [(0, 20.5)]:
        s = (z >= lo) & (z < (hi if hi < 20 else 20.5))
        f = lambda q: " / ".join(f"{np.percentile(e[s, j], q):.2f}" for j in idx)
        lab = "all" if (lo, hi) == (0, 20.5) else f"{lo}-{hi}"
        print(f"| {lab} | {s.sum()} | {f(68)} | {f(95)} |")
