# ede

The `v2` emulators are the emulators used in [The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models](https://arxiv.org/abs/2503.14454) (2025).

These emulators were then used in [Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble tension](https://arxiv.org/abs/2505.08051) by Poulin et al (2025).

They are the highest accuracy class-based emulators available on this organization and match the camb-based [Jense et al. (2024) Emulators](https://github.com/cosmopower-organization/jense_2024_emulators) in LCDM well under 0.1-$\sigma$ where $\sigma$ is the typical 68\%CL on parameter constraints using ACT DR6 data. 

The emulators can be used in LCDM, mnu-LCDM, w-CDM and Neff-LCDM and combinations of these. 

They emulate distances, TT, TE, EE, BB and P(k).

## ede-v3 (October 2026)

`*_v3` files: all emulators retrained on the same CLASS-EDE training set, with all training samples, a common train/test split for the CMB spectra, and a TE emulator for the correlation coefficient `r_l = C_l^TE / sqrt(C_l^TT C_l^EE)` (`TTTEEE/rTE_v3`, `TTTEEE/ruTE_v3`; output `atanh(r_l)`, TE = `tanh(r_l) sqrt(TT EE)`). Inputs, outputs and grids of the other files are those of v2. Training details in [generation/training_v3](generation/training_v3/README.md).

Test-set accuracy, 95th percentile of |emulator / CLASS - 1| (median over l, k or z / maximum), v2 evaluated on its own test set:

| emulator | v2 | v3 | v3 network |
|---|---|---|---|
| TT | 0.164% / 0.51% | 0.078% / 0.23% | 4x1024 |
| EE | 0.181% / 0.56% | 0.077% / 0.40% | 4x1024 |
| BB | 0.152% / 0.81% | 0.099% / 0.43% | 4x1024 |
| PP | 0.192% / 0.34% | 0.142% / 0.27% | 4x512 |
| uTT | 0.115% / 0.45% | 0.041% / 0.13% | 4x1024 |
| uEE | 0.286% / 1.21% | 0.214% / 0.56% | 4x1024 |
| uBB | 0.648% / 1.64% | 0.383% / 1.55% | 4x1024 |
| PKL | 0.156% / 0.32% | 0.025% / 0.28% | 4x1024 |
| PKLCB | 0.157% / 0.37% | 0.022% / 0.28% | 4x1024 |
| PKNL | 0.146% / 0.47% | 0.122% / 0.45% | 4x1024 |
| PKNLCB | 0.137% / 0.47% | 0.117% / 0.46% | 4x512 |
| DAZ | 0.010% / 0.044% | 0.0005% / 0.0010% | 4x1024 |
| HZ | 0.0051% / 0.019% | 0.0015% / 0.0044% | 4x1024 |
| S8Z | 0.015% / 0.032% | 0.0045% / 0.0079% | 4x1024 |
| DER | 0.013% / 0.067% | 0.0027% / 0.012% | 4x512 |

TE in units of the cosmic-variance error per multipole, sigma_l = sqrt((C_TT C_EE + C_TE^2)/(2l+1)), 19,841 test spectra (v3: TE reconstructed with the v3 TT and EE emulators; uTE with uTT and uEE):

| TE | 2 <= l < 30 | 30-500 | 500-3000 | 3000-9500 |
|---|---|---|---|---|
| TE v2, 95th percentile | 0.047 | 0.025 | 0.066 | 0.23 |
| TE v3, 95th percentile | 0.0074 | 0.015 | 0.039 | 0.038 |
| TE v2, 99.9th percentile | 0.22 | 0.086 | 0.30 | 2.86 |
| TE v3, 99.9th percentile | 0.028 | 0.070 | 0.24 | 0.18 |
| uTE v2, 95th percentile | 0.100 | 0.051 | 0.168 | 0.566 |
| uTE v3, 95th percentile | 0.0069 | 0.0115 | 0.041 | 0.101 |
| uTE v2, 99.9th percentile | 0.41 | 0.21 | 0.85 | 4.23 |
| uTE v3, 99.9th percentile | 0.025 | 0.046 | 0.27 | 0.71 |

PKNL v2: the retrained version of October 2026 (see below). For z > 5 class_sz and classy_szlite return the linear P(k) in place of PKNL.

## Usage

The emulators are used through [class_sz](https://github.com/CLASS-SZ/class_sz) with `cosmo_model: 7` (`ede-v3`, classy_szfast >= 0.0.25.post41 and a classy_sz release after 0.1.70.post40) or `cosmo_model: 6` (`ede-v2`), and through [classy_szlite](https://github.com/CLASS-SZ/classy_szlite) (ede-v3 by default, `classy_szlite.set_emulator_version("v2")` for v2). Example with the matter power spectrum, compared with CLASS-EDE from z = 0 to 15: [documentation page](https://class-sz.readthedocs.io/en/latest/notebooks/classy_szfast_matter_pk_ede.html) ([notebook](https://github.com/CLASS-SZ/class_sz/blob/master/docs/notebooks/classy_szfast_matter_pk_ede.ipynb)).

```python
from classy_sz import Class as Class_sz

params = {'omega_b': 0.02242, 'omega_cdm': 0.11933, 'H0': 67.66, 'tau_reio': 0.0561,
          'ln10^{10}A_s': 3.047, 'n_s': 0.9665,
          'fEDE': 0.1, 'log10z_c': 3.562, 'thetai_scf': 2.83}
csz = Class_sz()
csz.set(params)
csz.set({'output': 'mPk', 'cosmo_model': 6, 'skip_input': 1})
csz.initialize_classy_szfast()
pk_nl, k = csz.get_pknl_at_z(1., params_values_dict=params)   # k in 1/Mpc, P in Mpc^3
cls = csz.get_cmb_cls(params_values_dict=params)              # lensed TT, TE, EE, BB, PP
```

## Training set

The Boltzmann code (CLASS v3.2.2 with the EDE module), the CLASS settings, the parameter ranges and the generation, reprocessing and training scripts are in [generation/](generation/README.md). The training and test data are on [Google Drive](https://drive.google.com/drive/folders/1Sv_kHq2SuRcUhiN8EVGLhwlcGitvgDfd).

## Update: non-linear P(k) emulators retrained (October 2026)

`PK/PKNL_v2` and `PK/PKNLCB_v2` (all three formats) were retrained after removing corrupted spectra from the training set: CLASS-EDE had returned the linear spectrum in place of the HMcode result for about 16-22% of samples at z < 5. The previous non-linear emulators had errors of tens of percent at k > 0.3/Mpc; the retrained ones are accurate to 0.5% (95th percentile) up to k = 10/Mpc for 0 <= z <= 10, and to 1.3% for 10 < z <= 20. All other emulators, including the linear `PKL_v2` and the CMB spectra, are unchanged. Details and accuracy table in [PK/README.md](PK/README.md).

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

