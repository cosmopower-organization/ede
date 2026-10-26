"""JAX re-implementation of cosmopower_NN training, same settings as 4_gpu_train_ede_final.py.

Network: 4 x 512 hidden layers, activation (beta + sigmoid(alpha x)(1 - beta)) x,
loss = RMS of normalised features, Adam with cooling schedule
lr = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], batch 1024, patience 100, max 1000 epochs per step,
validation_split 0.1. Saves a cosmopower-format .npz readable by classy_szfast Restore_NN.
"""
import argparse, time
import numpy as np
import jax, jax.numpy as jnp
import optax

p = argparse.ArgumentParser()
p.add_argument("--var", required=True)              # PKNL or PKNLCB
p.add_argument("--data", default="/home/cmbagent/pk_ede/retrain")
p.add_argument("--out", required=True)               # output filename without .npz
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lrs", default="1e-2,1e-3,1e-4,1e-5,1e-6")  # cooling schedule
p.add_argument("--clip", type=float, default=0.)             # global-norm gradient clipping (0 = off)
args = p.parse_args()

prm = np.load(f"{args.data}/{args.var}_training_params_clean.npz")
fea = np.load(f"{args.data}/{args.var}_training_processed_data_clean.npz")
names = list(prm.files)
X = np.stack([prm[k] for k in names], axis=1).astype(np.float32)
Y = fea["features"].astype(np.float32)
modes = fea["modes"]
print(f"{args.var}: {X.shape[0]} training samples, {X.shape[1]} parameters {names}, {Y.shape[1]} modes", flush=True)

# normalisation (computed on the full training set, as cosmopower does)
pm, ps = X.mean(0), X.std(0)
fm, fs = Y.mean(0), Y.std(0)
Xn, Yn = (X - pm) / ps, (Y - fm) / fs

# validation split 0.1
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(Xn)); nval = int(0.1 * len(Xn))
iv, it = perm[:nval], perm[nval:]
Xt, Yt = jax.device_put(Xn[it]), jax.device_put(Yn[it])
Xv, Yv = jax.device_put(Xn[iv]), jax.device_put(Yn[iv])

arch = [X.shape[1], 512, 512, 512, 512, Y.shape[1]]
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

def forward(pr, x):
    for i in range(len(pr["W"]) - 1):
        x = x @ pr["W"][i] + pr["b"][i]
        x = (pr["be"][i] + jax.nn.sigmoid(pr["a"][i] * x) * (1. - pr["be"][i])) * x
    return x @ pr["W"][-1] + pr["b"][-1]

def loss_fn(pr, x, y):
    return jnp.sqrt(jnp.mean((forward(pr, x) - y) ** 2))

@jax.jit
def val_loss(pr):
    # chunked to bound memory
    n = Xv.shape[0] // 8 * 8
    xs, ys = Xv[:n].reshape(8, -1, Xv.shape[1]), Yv[:n].reshape(8, -1, Yv.shape[1])
    return jnp.sqrt(jnp.mean(jax.vmap(lambda a, b: jnp.mean((forward(pr, a) - b) ** 2))(xs, ys)))

def make_epoch(opt, bs):
    nb = Xt.shape[0] // bs
    @jax.jit
    def epoch(pr, st, k):
        idx = jax.random.permutation(k, Xt.shape[0])[: nb * bs].reshape(nb, bs)
        def step(carry, ib):
            pr, st = carry
            l, g = jax.value_and_grad(loss_fn)(pr, Xt[ib], Yt[ib])
            up, st = opt.update(g, st, pr)
            return (optax.apply_updates(pr, up), st), l
        (pr, st), ls = jax.lax.scan(step, (pr, st), idx)
        return pr, st, ls.mean()
    return epoch

lrs = [float(x) for x in args.lrs.split(",")]
bss, pats, maxe = [1024] * len(lrs), [100] * len(lrs), [1000] * len(lrs)
best, best_val = params, float(val_loss(params))
t0 = time.time()
for lr, bs, pat, me in zip(lrs, bss, pats, maxe):
    opt = optax.chain(optax.clip_by_global_norm(args.clip), optax.adam(lr)) if args.clip > 0 else optax.adam(lr)
    st = opt.init(best)
    pr = best
    epoch = make_epoch(opt, bs)
    since = 0
    for e in range(me):
        key, k1 = jax.random.split(key)
        pr, st, tl = epoch(pr, st, k1)
        vl = float(val_loss(pr))
        if vl < best_val:
            best, best_val, since = pr, vl, 0
        else:
            since += 1
        if e % 25 == 0:
            print(f"lr={lr:.0e} epoch {e:4d} train {float(tl):.5f} val {vl:.5f} best {best_val:.5f} [{time.time()-t0:.0f}s]", flush=True)
        if since >= pat:
            print(f"lr={lr:.0e}: early stop at epoch {e}, best val {best_val:.5f}", flush=True)
            break

nl = len(arch) - 1
out = {
    "weights_": [np.asarray(w, dtype=np.float32) for w in best["W"]],
    "biases_": [np.asarray(b, dtype=np.float32) for b in best["b"]],
    "alphas_": [np.asarray(a, dtype=np.float32) for a in best["a"]],
    "betas_": [np.asarray(b, dtype=np.float32) for b in best["be"]],
    "param_train_mean": pm, "param_train_std": ps,
    "feature_train_mean": fm, "feature_train_std": fs,
    "n_parameters": X.shape[1], "parameters": names,
    "n_modes": Y.shape[1], "modes": modes,
    "n_hidden": arch[1:-1], "n_layers": nl, "architecture": arch,
}
np.savez(args.out + ".npz", out)
print(f"saved {args.out}.npz, best val loss {best_val:.5f}, {time.time()-t0:.0f}s", flush=True)
