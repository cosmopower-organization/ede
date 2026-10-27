# ede

CosmoPower emulators of CLASS with axion-like early dark energy (EDE), for the lensed and unlensed CMB power spectra (TT, TE, EE, BB, lensing potential PP), the linear and non-linear matter power spectrum, H(z), D_A(z), sigma_8(z) and derived parameters.

The emulators can be used in LCDM, mnu-LCDM, w-CDM, Neff-LCDM, EDE and combinations of these. The default `fEDE = 0.001` is the LCDM-equivalent point.

Two versions are provided, trained on the same CLASS-EDE spectra:

- **`v3`** (October 2026, recommended): all emulators retrained, test errors 1.2 to 20 times lower than v2.
- **`v2`**: the emulators used in [The Atacama Cosmology Telescope: DR6 Constraints on Extended Cosmological Models](https://arxiv.org/abs/2503.14454) (2025) and in [Impact of ACT DR6 and DESI DR2 for Early Dark Energy and the Hubble tension](https://arxiv.org/abs/2505.08051) by Poulin et al. (2025). They match the CAMB-based [Jense et al. (2024) emulators](https://github.com/cosmopower-organization/jense_2024_emulators) in LCDM well under 0.1 sigma, where sigma is the typical 68% CL on parameter constraints from ACT DR6 data.

## ede-v3

Files `*_v3` in each folder. Inputs, outputs and k, l, z grids are those of v2, except for TE (below). Changes with respect to v2:

- all training samples (396,865 per CMB spectrum; v2 used 250,000 for TT, EE, BB, PP, uTT, uEE, uBB);
- one train/test split common to all CMB spectra, so that the TT, TE and EE of a test sample come from the same CLASS run;
- wider networks (4x1024) where they reduce the test error, selected per observable;
- full float32 matrix products (`JAX_DEFAULT_MATMUL_PRECISION=highest`; the default TF32 products of recent NVIDIA GPUs increase the test errors by 20-35%);
- non-linear P(k) trained on the cleaned set of the October 2026 retraining (see Non-linear P(k) below);
- TE emulated through the correlation coefficient.

Training scripts and details: [generation/training_v3](generation/training_v3/README.md).

### TE through the correlation coefficient

C_l^TE changes sign, so it cannot be emulated in log10 like the other spectra; v2 used a PCA of TE followed by a network, and its relative error is not defined near the zeros of TE. v3 emulates

```
r_l = C_l^TE / sqrt(C_l^TT C_l^EE),
```

which is smooth and bounded, |r_l| < 1. The network outputs `atanh(r_l)` (`TTTEEE/rTE_v3`, `TTTEEE/ruTE_v3` for the unlensed spectra), and TE is reconstructed with the TT and EE emulators:

```
C_l^TE = tanh(rTE) sqrt(C_l^TT C_l^EE).
```

Any output then satisfies (C_l^TE)^2 < C_l^TT C_l^EE, so the TT/TE/EE covariance stays positive definite. class_sz (`cosmo_model: 7`) and classy_szlite do this reconstruction internally.

### Accuracy

Test set: 99,216 CLASS-EDE runs for the CMB spectra. 95th percentile of |emulator / CLASS - 1|, median over l, k or z / maximum; v2 evaluated on its own test set:

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

PKNL and PKNLCB v2 are the retrained versions of October 2026.

TE in units of the cosmic-variance error per multipole, sigma_l = sqrt((C_TT C_EE + C_TE^2)/(2l+1)), on 19,841 runs in both test sets (v3 TE reconstructed with the v3 TT and EE emulators, uTE with uTT and uEE):

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

### Scaling with the number of training spectra

TT test error for networks trained on 25,000 to 396,865 spectra (same test set):

![TT scaling](generation/training_v3/scaling_TT.png)

Between 100,000 and 396,865 training spectra the error decreases as about N^-0.4 for 30 < l < 3000, with no sign of saturation: more CLASS-EDE spectra would reduce it further. Above l ~ 5000 it stays at 0.2-0.25% independently of N and of the network size. With all training spectra, the 4x1024 network is more accurate than the 4x512 one (orange against dark blue).

## Usage

[class_sz](https://github.com/CLASS-SZ/class_sz): `cosmo_model: 7` for v3 (classy_szfast >= 0.0.25.post41 and a classy_sz release after 0.1.70.post40), `cosmo_model: 6` for v2.

```python
from classy_sz import Class as Class_sz

params = {'omega_b': 0.02242, 'omega_cdm': 0.11933, 'H0': 67.66, 'tau_reio': 0.0561,
          'ln10^{10}A_s': 3.047, 'n_s': 0.9665,
          'fEDE': 0.1, 'log10z_c': 3.562, 'thetai_scf': 2.83}
csz = Class_sz()
csz.set(params)
csz.set({'output': 'mPk', 'cosmo_model': 7, 'skip_input': 1})
csz.initialize_classy_szfast()
pk_nl, k = csz.get_pknl_at_z(1., params_values_dict=params)   # k in 1/Mpc, P in Mpc^3
cls = csz.get_cmb_cls(params_values_dict=params)              # lensed TT, TE, EE, BB, PP
```

Example with the matter power spectrum, compared with CLASS-EDE from z = 0 to 15: [documentation page](https://class-sz.readthedocs.io/en/latest/notebooks/classy_szfast_matter_pk_ede.html) ([notebook](https://github.com/CLASS-SZ/class_sz/blob/master/docs/notebooks/classy_szfast_matter_pk_ede.ipynb)).

[classy_szlite](https://github.com/CLASS-SZ/classy_szlite) (pure JAX, `pip install classy-szlite`): v3 from version 0.2.15; v2 with `classy_szlite <= 0.2.14` and `set_emulator_version("v2")`.

For z > 5, class_sz and classy_szlite return the linear P(k) in place of the non-linear one.

## Training set

The Boltzmann code (CLASS v3.2.2 with the EDE module and HMcode 2020), the CLASS settings, the parameter ranges, the sample counts and the generation, reprocessing and training scripts are in [generation/](generation/README.md). The training and test data are on [Google Drive](https://drive.google.com/drive/folders/1Sv_kHq2SuRcUhiN8EVGLhwlcGitvgDfd).

## Non-linear P(k): retraining of October 2026

CLASS-EDE returned the linear spectrum in place of the HMcode result, without an error, for about 16-22% of the training samples at z < 5. The original `PK/PKNL_v2` and `PK/PKNLCB_v2` had errors of tens of percent at k > 0.3/Mpc. They were replaced (same names, all three formats) by networks retrained after removing these samples: 0.5% (95th percentile) up to k = 10/Mpc for 0 <= z <= 10, and 1.3% for 10 < z <= 20. The v3 non-linear emulators use the same cleaned set. Details in [PK/README.md](PK/README.md).

## File formats

Each emulator is provided in three formats with identical weights (`<v>` = `v2` or `v3`):

- **`<name>_<v>.npz`**: the CosmoPower output, a single `arr_0` object array holding a pickled `cosmopower_NN` / `cosmopower_PCAplusNN` instance. Loading requires `cosmopower` (and TensorFlow).
- **`<name>_<v>.pkl`**: the same content as a pickle, as read by classy_szfast.
- **`<name>_<v>_plain.npz`**: the same weights in a flat `np.savez_compressed` layout, loadable with `allow_pickle=False` and plain numpy. List-valued fields (`weights_`, `biases_`, `alphas_`, `betas_`) are stored as `<field>.<index>` with a `<field>.n` count; scalars and 1-D arrays as they are. Used by classy_szlite.

Loader for the plain format:

```python
import numpy as np
npz = np.load("TT_v3_plain.npz", allow_pickle=False)
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
