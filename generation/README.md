# Training-set generation for the ede emulators

Code, settings and scripts used to compute the CLASS-EDE spectra on which the `v2` and `v3` emulators of this repository were trained, to retrain the non-linear matter power spectrum emulators of v2 and to train the `v3` emulators (October 2026). Both versions use the same CLASS-EDE spectra.

| folder | content |
|---|---|
| `class_ede/` | Boltzmann code used for all training spectra |
| `settings/` | CLASS input (`class_settings.yaml`), parameter ranges (`parameter_ranges.yaml`), original cobaya yaml, BBN table |
| `scripts_2024/` | original pipeline: Latin hypercube, spectra, reprocessing, training (2024) |
| `retraining_2026/` | cleaning and retraining of `PKNL_v2` and `PKNLCB_v2` (October 2026) |
| `training_v3/` | common train/test split of the CMB spectra, r_l target for TE, training, scaling runs and export of the `v3` emulators (October 2026) |

Example of use and comparison with CLASS-EDE: [class_sz documentation](https://class-sz.readthedocs.io/en/latest/notebooks/classy_szfast_matter_pk_ede.html) ([notebook](https://github.com/CLASS-SZ/class_sz/blob/master/docs/notebooks/classy_szfast_matter_pk_ede.ipynb)).

## Training data

[Google Drive folder `ede-final`](https://drive.google.com/drive/folders/1Sv_kHq2SuRcUhiN8EVGLhwlcGitvgDfd):

- `<obs>_training_params.npz`, `<obs>_testing_params.npz`: emulator inputs, one array per parameter;
- `<obs>_training_processed_data.npz`, `<obs>_testing_processed_data.npz`: `features` (float32, one row per sample) and `modes`;
- `LHS_parameter_file.npz`: the full Latin hypercube (500,000 samples);
- `group_N/`: raw per-sample outputs (`LHS_parameter_file_i.npz`, `<obs>/data_i.npy`) for 30 of the 1000 groups;
- `ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final.yaml`: settings file read by the generation scripts.

`<obs>` is one of TT, TE, EE, BB, PP, uTT, uTE, uEE, uBB (C_ell), S8Z, HZ, DAZ (functions of z), PKL, PKLCB, PKNL, PKNLCB (P(k)), DER (derived parameters). The PKNL and PKNLCB files are the 2024 versions, before cleaning; `retraining_2026/clean.py` produces the cleaned sets.

Number of samples (80/20 training/test split). The `v3` CMB emulators use the same 496,081 runs with one 80/20 split common to all CMB observables (`training_v3/build_common_split.py`: 396,865 training, 99,216 test), and all training samples; v2 used 250,000 training samples for TT, EE, BB, PP, uTT, uEE, uBB. The `v3` PKNL and PKNLCB emulators are trained on the `clean.py` sets.

| data | training | test | total | fraction of the 500,000 hypercube points |
|---|---:|---:|---:|---:|
| Latin hypercube (`LHS_parameter_file.npz`) | | | 500,000 | 100% |
| C_ell, H(z), D_A(z), sigma_8(z), derived parameters, PKL, PKLCB | 396,864 | 99,217 | 496,081 | 99.2% |
| PKNL, PKNLCB (2024 sets) | 383,536 | 95,885 | 479,421 | 95.9% |
| PKNL after `clean.py` (HMcode spectra only) | 103,413 | 25,755 | 129,168 | 25.8% |

- 3,919 hypercube points (0.8%) have no spectra: the CLASS-EDE run failed or produced no output.
- 16,660 further PKNL/PKNLCB samples were dropped in the 2024 reprocessing: CLASS raised an error for the non-linear P(k) at z <= 5. At z > 5 such samples were kept with P_lin in place of P_nl.
- Of the 479,421 PKNL samples, 350,222 are identical to P_lin: HMcode is not computed above a cosmology-dependent redshift (half of the sampled redshifts are above z = 10), and some runs returned the linear spectrum without error (16-22% of samples at z < 5). A further 31 are HMcode glitches. Counts for PKNLCB differ by fewer than 100 samples.

## Boltzmann code

`class_ede/` is CLASS v3.2.2 (J. Lesgourgues, T. Tram, N. Schoeneberg et al., https://github.com/lesgourg/class_public) with

- the axion-like early dark energy scalar field and the `fEDE`, `log10z_c` shooting of `class_ede` (Hill et al. 2020, https://github.com/mwt5345/class_ede),
- HMcode 2020 (`hmcode_version: '2020'`) as implemented in CLASS by J. Lesgourgues.

It is the git-tracked content of `codes/class_v3_nonlinear_ede/class` at commit `db8fd15` (2024-10-07) of the BoltzmannBenchmarks repository. Spectra recomputed with this code reproduce the stored training spectra: linear P(k) to better than 0.01% for k <= 0.1/Mpc and 0.1% at k = 10/Mpc; non-linear P(k) to better than 0.04% whenever CLASS applies HMcode (see Known issues).

Build:

```bash
cd class_ede
make clean; make -j
cd python && pip install .   # installs classy
```

## Settings

`settings/class_settings.yaml` lists the input passed to `classy` for each training sample: the base settings of the generation script (`output: tCl pCl lCl mPk`, `modes: s,t`, `non_linear: hmcode`, `z_max_pk: 20`) and the precision and model settings (`theory.classy.extra_args` of `ACTPol_lite_DR4_baseLCDM_taup_hip_bao_pks_ede-final.yaml`). Main choices: HMcode 2020 dark-matter only, three degenerate massive neutrinos (`N_ncdm: 3`, `m_ncdm` per species), HyRec, BBN helium from the PRIMAT 2021 table (`PRIMAT_Yp_DH_ErrorMC_2021_CLASS.dat`, converted from the CAMB table with `reformat_primat_file_from_camb_to_class.py`), `P_k_max_h/Mpc: 100`.

Outputs per sample: lensed and unlensed TT, TE, EE, BB, lensing potential PP up to l = 9500; H(z), D_A(z), sigma_8(z) on 5000 points in 0 <= z <= 20; derived parameters; linear and non-linear P(k) (total matter and cold dark matter + baryons) at k = geomspace(5e-4, 10, 1000) 1/Mpc and at one redshift `z_pk_save_nonclass` per sample.

## Parameter ranges

Latin hypercube of 500,000 points (`settings/parameter_ranges.yaml`, from `scripts_2024/1_generate_lhc-ede-final-splitdir.py`; see Training data for the number of computed spectra):

| parameter | min | max | |
|---|---|---|---|
| `fEDE` | 0.001 | 0.5 | |
| `log10z_c` | 3.0 | 4.3 | |
| `thetai_scf` | 0.1 | 3.1 | |
| `ln10^{10}A_s` | 2.5 | 3.5 | |
| `n_s` | 0.8 | 1.2 | |
| `H0` | 39.99 | 100.01 | km/s/Mpc |
| `omega_b` | 0.01933 | 0.02533 | |
| `omega_cdm` | 0.08 | 0.2 | |
| `tau_reio` | 0.02 | 0.12 | not an input of the P(k), H(z), D_A(z), sigma_8(z) emulators |
| `m_ncdm` | 0 | 0.33333 | eV per species, 3 degenerate species |
| `N_ur` | -2.0 | 2.5 | `N_ur = 0.00441` gives N_eff = 3.044 |
| `r` | 0 | 0.3 | |
| `z_pk_save_nonclass` | 0 | 20 | redshift of the saved P(k) |

Defaults used by class_sz for parameters not passed: `fEDE = 0.001`, `log10z_c = 3.562`, `thetai_scf = 2.83`, `r = 0`, `m_ncdm = 0.02`, `N_ur = 0.00441`.

## Pipeline (2024)

The scripts in `scripts_2024/` ran on the Flatiron cluster (disBatch); paths are those of that system.

1. `1_generate_lhc-ede-final-splitdir.py`: Latin hypercube, saved as `group_N/LHS_parameter_file_i.npz` (500 samples per group).
2. `2_generate_spectra_disbatch_ede-final-splitdir-1k-by_groupid.py`: one CLASS-EDE run per sample, all observables saved as `group_N/<obs>/data_i.npy`.
3. `3_reprocess_spectra_disbatch_ede-final-splitdir-1k-by_obs.py`: assembly into training and test sets (80/20), features log10(l^2 C_l), log10(k^3 P(k)), etc.; non-linear P(k) samples where CLASS raised an error were replaced by the linear P(k) when z > 5 and dropped otherwise.
4. `4_gpu_train_ede_final.py`: cosmopower networks, 4 hidden layers of 512 nodes, learning rates 1e-2 to 1e-6, batch 1024, patience 100.

## Retraining of PKNL_v2 and PKNLCB_v2 (October 2026)

Two defects of the non-linear training data, found by recomputing training samples with `class_ede/`:

- In some CLASS-EDE runs, `pk()` returns the linear spectrum at every redshift without raising an error (3 of 8 identical runs in one test). About 16-22% of the non-linear training spectra at z < 5 were such linear spectra.
- About 30 samples per emulator, all near fEDE ~ 0.45, n_s ~ 0.8, omega_cdm ~ 0.08, were HMcode outputs off by several orders of magnitude.

`retraining_2026/`:

- `clean.py`: matches each non-linear sample to its linear one through the input parameters, drops samples whose non-linear spectrum equals the linear one, and drops glitches (P_nl/P_lin != 1 within 0.02 dex at k < 0.01/Mpc, boost < 10^-0.3 or > 10^3, spikes along k). About 103,000 training samples remain per emulator.
- `train.py`: JAX implementation of the cosmopower network and loss (same architecture and activation), learning rates 1e-3 to 1e-6, global-norm gradient clipping 1.0, batch 1024, patience 100. Run with `JAX_DEFAULT_MATMUL_PRECISION=highest`; the default TF32 matrix products on recent NVIDIA GPUs increase the errors by 20-35%. Released weights: seed 32 (PKNL), seed 34 (PKNLCB).
- `validate.py`, `zbins_table.py`: accuracy on the test set, overall and by redshift (results in `../PK/README.md`).
- `export.py`: writes `_v2.npz`, `_v2.pkl` and `_v2_plain.npz`.

Commands used:

```bash
python clean.py
JAX_DEFAULT_MATMUL_PRECISION=highest python train.py --var PKNL   --seed 32 --lrs 1e-3,1e-4,1e-5,1e-6 --clip 1.0 --out PKNL_v2_retrained
JAX_DEFAULT_MATMUL_PRECISION=highest python train.py --var PKNLCB --seed 34 --lrs 1e-3,1e-4,1e-5,1e-6 --clip 1.0 --out PKNLCB_v2_retrained
python export.py <dir with *_v2_retrained.npz> <output dir>
```

Paths to the training data are set at the top of each script.

Since the retrained networks contain no linear spectra, class_sz (classy_szfast >= 0.0.25.post40) returns the linear P(k) for z > 5 (`pknl_zmax` in `emulators_meta_data.py`). Beyond a cosmology-dependent redshift between about 5 and 10, CLASS-EDE itself does not apply HMcode and returns the linear spectrum.

## Training of the v3 emulators (October 2026)

`training_v3/` (details in [training_v3/README.md](training_v3/README.md)):

- `build_common_split.py`, `build_ruTE.py`: one train/test split shared by TT, EE, BB, PP, uTT, uEE, uBB, TE and uTE, so that row i is the same CLASS-EDE run in every file; TE and uTE targets `atanh(r_l)`, `r_l = C_l^TE / sqrt(C_l^TT C_l^EE)`. TE is reconstructed as `tanh(r_l) sqrt(TT EE)` with the TT and EE emulators.
- `train_v3.py`: JAX cosmopower network, learning rates 1e-2 to 1e-6, global-norm gradient clipping 1.0, batch 1024, patience 100, `JAX_DEFAULT_MATMUL_PRECISION=highest`, all training samples.
- `runner.py`, `make_jobs*.py`: scaling runs (25,000 to 396,865 training samples) and capacity runs (4x512, 4x1024, 6x1024 hidden layers; PCA and r_l for TE). `select_v3.py` keeps, per observable, the run with the lowest median over modes of the 95th-percentile test error; `export_v3.py` writes `_v3.npz`, `_v3.pkl` and `_v3_plain.npz`.
- `eval_v2.py`, `eval_te.py`, `summarize.py`: test-set accuracy of v2 and v3 (tables in [../README.md](../README.md#ede-v3-october-2026)).

v3 test errors (95th percentile, median over modes) are 1.2 to 20 times lower than v2, for example TT 0.078% against 0.164%, PKL 0.025% against 0.156%, DAZ 0.0005% against 0.010%. TT errors decrease as about N^-0.4 up to the full training set:

![TT scaling](training_v3/scaling_TT.png)

## Known issues with class_ede

- Non-linear P(k) intermittently equal to the linear one at all redshifts, with no error. Check P_nl/P_lin at z = 0, k = 1/Mpc (about 9 in LCDM-like cosmologies) and recompute if equal to 1.
- `pk()` at redshifts beyond the HMcode range returns the linear spectrum or, with `lCl` in the output, raises an error.
