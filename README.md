# ede

The `v2` emulators are the emulators used in [The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models](https://arxiv.org/abs/2503.14454) (2025).

These emulators were then used in [Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble tension](https://arxiv.org/abs/2505.08051) by Poulin et al (2025).

They are the highest accuracy class-based emulators available on this organization and match the camb-based [Jense et al. (2024) Emulators](https://github.com/cosmopower-organization/jense_2024_emulators) in LCDM well under 0.1-$\sigma$ where $\sigma$ is the typical 68\%CL on parameter constraints using ACT DR6 data. 

The emulators can be used in LCDM, mnu-LCDM, w-CDM and Neff-LCDM and combinations of these. 

They emulate distances, TT, TE, EE, BB and P(k).

## File formats: `_v2.npz` and `_v2_plain.npz`

Each emulator is provided in **two on-disk formats** with identical weights:

- **`<name>_v2.npz`** — the original CosmoPower output. A single
  `arr_0` object array holding a pickled `cosmopower_NN` /
  `cosmopower_PCAplusNN` instance. Loading requires `cosmopower` to be
  importable (which transitively imports `tensorflow` /
  `keras`), since pickle has to reconstruct
  `tensorflow.python.trackable.data_structures.ListWrapper`
  references. Drop-in compatible with the CosmoPower loader API.

- **`<name>_v2_plain.npz`** — the **same weights**, re-saved with a flat
  `np.savez_compressed(weights_.0=…, weights_.1=…, …, weights_.n=N, biases_.0=…, …)`
  layout. Loads with `allow_pickle=False` and zero TensorFlow /
  CosmoPower dependencies; pure numpy. The flat-key convention uses the
  format `<field>.<index>` for list-valued fields (`weights_`, `biases_`,
  `alphas_`, `betas_`) plus a `<field>.n` sentinel for the count, and
  preserves scalars / 1-D arrays as-is.

The plain variant is ~8 % smaller (due to per-array compression of all
the numeric blocks rather than the pickle protocol's blob compression).

### When to prefer which

- **Pipelines that already use CosmoPower** — keep using `_v2.npz`; no
  change.
- **Pipelines that load weights into a different framework** (e.g.
  JAX, PyTorch, raw numpy) and don't want to install TensorFlow just
  to deserialise — use `_v2_plain.npz`. Example loader (~20 lines):

  ```python
  import numpy as np
  npz = np.load("TT_v2_plain.npz", allow_pickle=False)
  def unflatten(saved):
      out = {}; groups = {}
      for k in saved.files:
          if "." in k:
              base, idx = k.rsplit(".", 1)
              if idx == "n": continue
              groups.setdefault(base, {})[int(idx)] = saved[k]
          else:
              v = saved[k]
              out[k] = v.item() if v.ndim == 0 and v.dtype.kind in "iuf" else v
      for base, idx_map in groups.items():
          out[base] = [idx_map[i] for i in sorted(idx_map)]
      return out
  d = unflatten(npz)
  # d["weights_"], d["biases_"], d["alphas_"], d["betas_"], d["param_train_mean"], ...
  ```

