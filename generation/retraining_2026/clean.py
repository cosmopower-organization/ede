"""Remove PKNL/PKNLCB samples where CLASS silently returned the linear spectrum.

Each non-linear sample is matched to its linear counterpart (PKL/PKLCB) through the
input parameters; a sample is flagged when log10(k^3 P_nl) equals log10(k^3 P_lin)
at all k. This also removes the z>5 rows that the reprocessing deliberately replaced
by P_lin, so the network only sees genuine HMcode output.
"""
import os, sys
import numpy as np
import pandas as pd

DATA = "/home/cmbagent/pk_ede/ede-final"
OUT = "/home/cmbagent/pk_ede/retrain"
TOL = 1e-4  # dex; float32 rounding of identical spectra is ~1e-6
Z_LIN = float(os.environ.get("Z_LIN", "inf"))  # linear-identical rows above this redshift are kept as P_lin targets (default: none kept, as for the released weights)
OUT = os.environ.get("CLEAN_OUT", OUT)

def load(var, split):
    p = np.load(f"{DATA}/{var}_{split}_params.npz")
    f = np.load(f"{DATA}/{var}_{split}_processed_data.npz")
    return {k: p[k] for k in p.files}, f["features"], f["modes"]

def key_frame(params):
    # exact match on the LHS inputs (same float64 values in both files)
    return pd.DataFrame({k: params[k] for k in params})

def clean(nl_var, l_var):
    print(f"==== {nl_var} (linear reference {l_var})")
    lin_p, lin_f = [], []
    for split in ("training", "testing"):
        p, f, _ = load(l_var, split)
        lin_p.append(key_frame(p)); lin_f.append(f)
    lin_p = pd.concat(lin_p, ignore_index=True); lin_f = np.concatenate(lin_f)
    cols = list(lin_p.columns)
    lin_p["_row"] = np.arange(len(lin_p))
    lin_p = lin_p.drop_duplicates(subset=cols)

    for split in ("training", "testing"):
        p, f, modes = load(nl_var, split)
        df = key_frame(p)
        df["_i"] = np.arange(len(df))
        m = df.merge(lin_p, on=cols, how="left")
        m = m.sort_values("_i")
        matched = m["_row"].notna().to_numpy()
        diff = np.full(len(df), np.inf)
        rows = m["_row"].to_numpy()
        diff[matched] = np.max(np.abs(f[matched] - lin_f[rows[matched].astype(int)]), axis=1)
        is_lin = diff < TOL
        z = p["z_pk_save_nonclass"]
        print(f"  {split}: {len(df)} samples, matched to linear: {matched.sum()}, "
              f"identical to linear: {is_lin.sum()} ({100*is_lin.mean():.1f}%)")
        for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20.5)]:
            sel = (z >= lo) & (z < hi)
            print(f"     z in [{lo},{hi}): {sel.sum():6d} samples, {100*is_lin[sel].mean():5.1f}% identical to linear")
        # physical sanity of the non-linear boost r = log10(P_nl / P_lin)
        k = np.geomspace(5e-4, 10., f.shape[1])
        r = f - lin_f[rows.astype(int)]
        bad_large_scales = np.max(np.abs(r[:, k < 0.01]), axis=1) > 0.02   # P_nl must equal P_lin on large scales
        bad_suppression = np.min(r, axis=1) < -0.3                          # HMcode suppresses by < a few tens of %
        bad_boost = np.max(r, axis=1) > 3.                                   # boost < 1000 for k <= 10/Mpc
        bad_spike = np.max(np.abs(np.diff(f, n=2, axis=1)), axis=1) > 5e-3   # spikes along k
        glitch = (bad_large_scales | bad_suppression | bad_boost | bad_spike) & ~is_lin
        print(f"     glitches among non-linear samples: {glitch.sum()} (large-scale {np.sum(bad_large_scales & ~is_lin)}, "
              f"suppression {np.sum(bad_suppression & ~is_lin)}, boost {np.sum(bad_boost & ~is_lin)}, spike {np.sum(bad_spike & ~is_lin)})")
        # rows identical to P_lin are dropped at z <= Z_LIN
        keep = ~glitch & ~(is_lin & (z <= Z_LIN))
        print(f"     kept {keep.sum()} (linear rows kept at z > {Z_LIN}: {np.sum(is_lin & (z > Z_LIN))})")
        np.savez(f"{OUT}/{nl_var}_{split}_params_clean.npz", **{k: v[keep] for k, v in p.items()})
        np.savez(f"{OUT}/{nl_var}_{split}_processed_data_clean.npz", features=f[keep], modes=modes)
        np.save(f"{OUT}/{nl_var}_{split}_is_linear.npy", is_lin)

if __name__ == "__main__":
    for nl_var, l_var in [("PKNL", "PKL"), ("PKNLCB", "PKLCB")]:
        if len(sys.argv) > 1 and nl_var not in sys.argv[1:]:
            continue
        clean(nl_var, l_var)
