"""r_l target for the unlensed spectra: atanh(uTE / sqrt(uTT uEE)) on the common split."""
import os, numpy as np
CS = os.path.expanduser("~/rds/ede-v3/data_cs")
for split in ("training", "testing"):
    te = np.load(f"{CS}/uTE_{split}_processed_data.npz"); modes = te["modes"]; TE = te["features"]
    TT = np.load(f"{CS}/uTT_{split}_processed_data.npz")["features"]; EE = np.load(f"{CS}/uEE_{split}_processed_data.npz")["features"]
    n = min(TE.shape[1], TT.shape[1], EE.shape[1]); R = np.empty((len(TE), n), np.float32)
    for i in range(0, len(TE), 20000):
        s = slice(i, i + 20000)
        R[s] = np.arctanh(np.clip(TE[s, :n] / np.sqrt(10.0 ** (TT[s, :n].astype(np.float64) + EE[s, :n])), -1 + 1e-7, 1 - 1e-7))
    np.savez(f"{CS}/ruTE_{split}_processed_data.npz", features=R, modes=modes[:n])
    if not os.path.lexists(f"{CS}/ruTE_{split}_params.npz"):
        os.symlink(f"common_{split}_params.npz", f"{CS}/ruTE_{split}_params.npz")
    print(split, "max |r|", float(np.abs(np.tanh(R)).max()), flush=True)
open(f"{CS}/built.log", "a").write("ok ruTE\n")
