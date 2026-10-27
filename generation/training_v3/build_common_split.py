"""Common train/test split for the CMB observables (and the r_l = TE/sqrt(TT EE) target).

The 2024 reprocessing split each observable independently. Here all train+test samples of an
observable are pooled, matched to a canonical ordering of the 496,081 CLASS runs through their
input parameters, and split with one fixed 20% test selection shared by all observables.
Rows are aligned across observables: row i is the same CLASS run in every *_cs file.
"""
import os, time
import numpy as np

V3 = os.path.expanduser("~/rds/ede-v3")
SRC, DST = f"{V3}/data", f"{V3}/data_cs"
os.makedirs(DST, exist_ok=True)
OBS = ["TT", "EE", "TE", "PP", "BB", "uBB", "uTT", "uEE", "uTE"]
TEST_FRAC, SEED = 0.2, 2026

def ok(name):
    return f"ok {name}" in open(f"{V3}/download.log").read().split("\n")

def load_params(o):
    ps = [np.load(f"{SRC}/{o}_{s}_params.npz") for s in ("training", "testing")]
    keys = sorted(ps[0].files)
    return keys, {k: np.concatenate([p[k] for p in ps]) for k in keys}

def row_keys(P, keys):
    a = np.ascontiguousarray(np.stack([P[k] for k in keys], 1).astype(np.float64))
    return [r.tobytes() for r in a]

# canonical ordering and split, from TT parameters
keys, P0 = load_params("TT")
rk = row_keys(P0, keys)
order = np.argsort(np.array(rk, dtype=object))
canon = {rk[i]: c for c, i in enumerate(order)}
N = len(canon)
is_test = np.zeros(N, bool)
is_test[np.random.default_rng(SEED).permutation(N)[: int(TEST_FRAC * N)]] = True
pos = np.empty(N, np.int64)
pos[~is_test] = np.arange((~is_test).sum()); pos[is_test] = np.arange(is_test.sum())
canon_params = {k: P0[k][order] for k in keys}
for split, sel in (("training", ~is_test), ("testing", is_test)):
    np.savez(f"{DST}/common_{split}_params.npz", **{k: v[sel] for k, v in canon_params.items()})
print(f"canonical samples {N}: training {(~is_test).sum()}, testing {is_test.sum()}", flush=True)

def build(o):
    k_o, P = load_params(o)
    assert k_o == keys, (o, k_o)
    idx = np.array([canon[r] for r in row_keys(P, keys)])
    assert len(np.unique(idx)) == len(idx), o
    present = np.zeros(N, bool); present[idx] = True        # an observable may lack a few runs
    opos = np.full(N, -1, np.int64)
    for t in (False, True):
        sel = present & (is_test == t)
        opos[sel] = np.arange(sel.sum())
    tr_src = np.load(f"{SRC}/{o}_training_processed_data.npz")
    modes = tr_src["modes"]
    F = tr_src["features"]; n_tr_src = len(F); nm = F.shape[1]
    out = {"training": np.empty(((present & ~is_test).sum(), nm), np.float32),
           "testing": np.empty(((present & is_test).sum(), nm), np.float32)}
    for src_rows, offset in ((F, 0), (None, n_tr_src)):
        if src_rows is None:
            src_rows = np.load(f"{SRC}/{o}_testing_processed_data.npz")["features"]
        c = idx[offset: offset + len(src_rows)]
        t = is_test[c]
        out["testing"][opos[c[t]]] = src_rows[t]
        out["training"][opos[c[~t]]] = src_rows[~t]
        del src_rows
    del F
    for split, tsel in (("training", False), ("testing", True)):
        np.savez(f"{DST}/{o}_{split}_processed_data.npz", features=out[split], modes=modes)
        pf = f"{DST}/{o}_{split}_params.npz"
        if os.path.lexists(pf):
            os.remove(pf)
        if present.all():
            os.symlink(f"common_{split}_params.npz", pf)
        else:
            sel = present & (is_test == tsel)
            np.savez(pf, **{k: v[sel] for k, v in canon_params.items()})
    print(f"[{time.strftime('%H:%M')}] built {o}: {nm} modes, {present.sum()} of {N} runs", flush=True)

def build_r():
    # r_l = TE / sqrt(TT EE); TE feature = l^2 C^TE, TT/EE features = log10(l^2 C)
    for split in ("training", "testing"):
        te = np.load(f"{DST}/TE_{split}_processed_data.npz")
        modes = te["modes"]; TE = te["features"]
        TT = np.load(f"{DST}/TT_{split}_processed_data.npz")["features"]
        EE = np.load(f"{DST}/EE_{split}_processed_data.npz")["features"]
        n = min(TE.shape[1], TT.shape[1], EE.shape[1])
        R = np.empty((len(TE), n), np.float32)
        for i in range(0, len(TE), 20000):
            s = slice(i, i + 20000)
            r = TE[s, :n] / np.sqrt(10.0 ** (TT[s, :n].astype(np.float64) + EE[s, :n]))
            R[s] = np.arctanh(np.clip(r, -1 + 1e-7, 1 - 1e-7))
        np.savez(f"{DST}/rTE_{split}_processed_data.npz", features=R, modes=modes[:n])
        if not os.path.exists(f"{DST}/rTE_{split}_params.npz"):
            os.symlink(f"common_{split}_params.npz", f"{DST}/rTE_{split}_params.npz")
        print(f"[{time.strftime('%H:%M')}] built rTE {split}: |r| max {np.abs(np.tanh(R)).max():.6f}", flush=True)

done = set()
if os.path.exists(f"{DST}/built.log"):
    done = {l.split()[1] for l in open(f"{DST}/built.log") if l.startswith("ok ")}
while len(done) < len(OBS) + 1:
    for o in OBS:
        if o not in done and all(ok(f"{o}_{s}_{t}.npz") for s in ("training", "testing") for t in ("params", "processed_data")):
            build(o); done.add(o)
            open(f"{DST}/built.log", "a").write(f"ok {o}\n")
    if "rTE" not in done and {"TT", "EE", "TE"} <= done:
        build_r(); done.add("rTE")
        open(f"{DST}/built.log", "a").write("ok rTE\n")
    time.sleep(60)
print("ALL BUILT", flush=True)
