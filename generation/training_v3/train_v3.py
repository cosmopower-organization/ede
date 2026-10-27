"""ede-v3 trainer: cosmopower architecture (alpha-beta activations), JAX, any observable.

Plain mode: network output = normalised features.
PCA mode (--pca N): fixed PCA basis of the normalised training features; the network outputs
normalised PCA coefficients, the loss is computed on the reconstructed normalised features.
The saved file has the cosmopower/class_sz format (Restore_NN or Restore_PCAplusNN).

--ntrain N uses the first N samples of a fixed random permutation of the training set,
so that subsets of different sizes are nested. 10% of the subset is used for validation
(early stopping). The full test set is evaluated at the end (eval json next to the npz).
"""
import argparse, json, os, time
import numpy as np
import jax, jax.numpy as jnp
import optax

p = argparse.ArgumentParser()
p.add_argument("--obs", required=True)
p.add_argument("--data", default=os.path.expanduser("~/rds/ede-v3/data"))
p.add_argument("--params-file", default=None)     # override, e.g. cleaned PKNL sets
p.add_argument("--features-file", default=None)
p.add_argument("--test-params-file", default=None)
p.add_argument("--test-features-file", default=None)
p.add_argument("--ntrain", type=int, default=0)   # 0 = all
p.add_argument("--hidden", default="512,512,512,512")
p.add_argument("--pca", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lrs", default="1e-3,1e-4,1e-5,1e-6")
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--patience", type=int, default=100)
p.add_argument("--max-epochs", type=int, default=1000)
p.add_argument("--inputs", default="")            # comma-separated input names, in order (as in the v2 file)
p.add_argument("--out", required=True)
args = p.parse_args()

D = args.data
pf = args.params_file or f"{D}/{args.obs}_training_params.npz"
ff = args.features_file or f"{D}/{args.obs}_training_processed_data.npz"
prm = np.load(pf)
names = args.inputs.split(",") if args.inputs else list(prm.files)
assert all(n in prm.files for n in names), (names, prm.files)
X = np.stack([prm[k] for k in names], 1).astype(np.float32)
fz = np.load(ff)
Y = fz["features"].astype(np.float32)
modes = fz["modes"]
assert len(X) == len(Y)

perm = np.random.default_rng(12345).permutation(len(X))      # fixed: nested subsets across runs
if args.ntrain:
    perm = perm[: args.ntrain]
X, Y = X[perm], Y[perm]
print(f"{args.obs}: {len(X)} training samples, inputs {names}, {Y.shape[1]} modes, pca={args.pca}", flush=True)

pm, ps = X.mean(0), X.std(0)
fm, fs = Y.mean(0, dtype=np.float64).astype(np.float32), Y.std(0, dtype=np.float64).astype(np.float32)
fs = np.where(fs > 0, fs, 1.).astype(np.float32)
Xn = (X - pm) / ps
Y -= fm; Y /= fs; Yn = Y            # in place: the C_ell training sets are 15 GB

if args.pca:
    # PCA basis of the normalised features (randomised SVD on a subsample)
    sub = Yn[np.random.default_rng(1).choice(len(Yn), min(len(Yn), 100000), replace=False)]
    _, _, Vt = np.linalg.svd(sub - 0., full_matrices=False)
    T = Vt[: args.pca].astype(np.float32)                       # (npca, nmodes)
    C = Yn @ T.T                                                 # PCA coefficients
    cm, cs = C.mean(0), C.std(0)
    out_dim = args.pca
else:
    out_dim = Yn.shape[1]

rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(Xn)); nval = int(0.1 * len(Xn))
iv, it = idx[:nval], idx[nval:]
Xt, Yt = jax.device_put(Xn[it]), jax.device_put(Yn[it])
Xv, Yv = jax.device_put(Xn[iv]), jax.device_put(Yn[iv])
if args.pca:
    Tj, cmj, csj = jnp.asarray(T), jnp.asarray(cm), jnp.asarray(cs)

hidden = [int(h) for h in args.hidden.split(",")]
arch = [Xn.shape[1]] + hidden + [out_dim]
key = jax.random.PRNGKey(args.seed)
params = {"W": [], "b": [], "a": [], "be": []}
for i in range(len(arch) - 1):
    key, k1 = jax.random.split(key)
    params["W"].append(1e-3 * jax.random.normal(k1, (arch[i], arch[i + 1])))
    params["b"].append(jnp.zeros(arch[i + 1]))
for i in range(len(arch) - 2):
    key, k1, k2 = jax.random.split(key, 3)
    params["a"].append(jax.random.normal(k1, (arch[i + 1],)))
    params["be"].append(jax.random.normal(k2, (arch[i + 1],)))

def net(pr, x):
    for i in range(len(pr["W"]) - 1):
        x = x @ pr["W"][i] + pr["b"][i]
        x = (pr["be"][i] + jax.nn.sigmoid(pr["a"][i] * x) * (1. - pr["be"][i])) * x
    return x @ pr["W"][-1] + pr["b"][-1]

def forward(pr, x):            # normalised features
    o = net(pr, x)
    return (o * csj + cmj) @ Tj if args.pca else o

def loss_fn(pr, x, y):
    return jnp.sqrt(jnp.mean((forward(pr, x) - y) ** 2))

# data arrays are passed as arguments (not closed over), so they are not embedded as constants
@jax.jit
def _val_loss(pr, xv, yv):
    nb = 16
    n = xv.shape[0] // nb * nb
    xs, ys = xv[:n].reshape(nb, -1, xv.shape[1]), yv[:n].reshape(nb, -1, yv.shape[1])
    return jnp.sqrt(jnp.mean(jax.lax.map(lambda ab: jnp.mean((forward(pr, ab[0]) - ab[1]) ** 2), (xs, ys))))

def val_loss(pr):
    return _val_loss(pr, Xv, Yv)

def make_epoch(opt, bs):
    nb = Xt.shape[0] // bs
    @jax.jit
    def _epoch(pr, st, k, xt, yt):
        ib_all = jax.random.permutation(k, xt.shape[0])[: nb * bs].reshape(nb, bs)
        def step(carry, ib):
            pr, st = carry
            l, g = jax.value_and_grad(loss_fn)(pr, xt[ib], yt[ib])
            up, st = opt.update(g, st, pr)
            return (optax.apply_updates(pr, up), st), l
        (pr, st), ls = jax.lax.scan(step, (pr, st), ib_all)
        return pr, st, ls.mean()
    return lambda pr, st, k: _epoch(pr, st, k, Xt, Yt)

lrs = [float(x) for x in args.lrs.split(",")]
best, best_val = params, float(val_loss(params))
t0 = time.time(); n_epochs = 0
for lr in lrs:
    opt = optax.chain(optax.clip_by_global_norm(args.clip), optax.adam(lr)) if args.clip > 0 else optax.adam(lr)
    st, pr, since = opt.init(best), best, 0
    epoch = make_epoch(opt, args.batch)
    for e in range(args.max_epochs):
        key, k1 = jax.random.split(key)
        pr, st, tl = epoch(pr, st, k1); n_epochs += 1
        vl = float(val_loss(pr))
        if vl < best_val:
            best, best_val, since = pr, vl, 0
        else:
            since += 1
        if e % 50 == 0:
            print(f"lr={lr:.0e} epoch {e:4d} train {float(tl):.6f} val {vl:.6f} best {best_val:.6f} [{time.time()-t0:.0f}s]", flush=True)
        if since >= args.patience:
            break
    print(f"lr={lr:.0e} done after {e+1} epochs, best val {best_val:.6f} [{time.time()-t0:.0f}s]", flush=True)
train_time = time.time() - t0

f32 = lambda a: np.asarray(a, dtype=np.float32)
out = {"weights_": [f32(w) for w in best["W"]], "biases_": [f32(b) for b in best["b"]],
       "alphas_": [f32(a) for a in best["a"]], "betas_": [f32(b) for b in best["be"]],
       "param_train_mean": f32(pm), "param_train_std": f32(ps),
       "feature_train_mean": f32(fm), "feature_train_std": f32(fs),
       "n_parameters": len(names), "parameters": names, "n_modes": int(Y.shape[1]), "modes": np.asarray(modes),
       "n_hidden": hidden, "n_layers": len(arch) - 1, "architecture": arch}
if args.pca:
    out.update(pca_mean=f32(cm), pca_std=f32(cs), n_pcas=args.pca, pca_transform_matrix=f32(T))
np.savez(args.out + ".npz", out)

# test-set evaluation (normalised-feature space -> errors in feature units)
tpf = args.test_params_file or f"{D}/{args.obs}_testing_params.npz"
tff = args.test_features_file or f"{D}/{args.obs}_testing_processed_data.npz"
tp = np.load(tpf)
Xs = (np.stack([tp[k] for k in names], 1).astype(np.float32) - pm) / ps
Ys = np.load(tff)["features"].astype(np.float32)
fwd = jax.jit(lambda x: forward(best, x))
pred = np.concatenate([np.asarray(fwd(jnp.asarray(Xs[i:i + 20000]))) for i in range(0, len(Xs), 20000)]) * fs + fm
err = np.abs(pred - Ys)                                   # |delta feature| (dex for log features)
pct = {q: np.percentile(err, q, axis=0) for q in (68, 95, 99)}
res = {"obs": args.obs, "ntrain": int(len(X)), "hidden": hidden, "pca": args.pca, "seed": args.seed,
       "best_val_loss": best_val, "epochs": n_epochs, "train_time_s": train_time, "n_test": int(len(Ys)),
       "median_over_modes": {str(q): float(np.median(v)) for q, v in pct.items()},
       "max_over_modes": {str(q): float(np.max(v)) for q, v in pct.items()},
       "per_mode": {str(q): v.astype(float).round(7).tolist() for q, v in pct.items()}}
json.dump(res, open(args.out + "_eval.json", "w"))
print(f"saved {args.out}.npz; val {best_val:.6f}; test median-over-modes 68/95/99 = "
      + " / ".join(f"{res['median_over_modes'][q]:.2e}" for q in ("68", "95", "99")) + f"; {train_time:.0f}s", flush=True)
