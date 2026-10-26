Matter power spectrum
linear and non-linear

non-linear = HMcode 2020, dark-matter only

k-range (1/Mpc, 1000 log-spaced points):
kmin = 5e-4
kmax = 10

redshift range:
zmin = 0
zmax = 20

outputs: log10(k^3 P(k)), P in Mpc^3

PKL / PKLCB: total matter / cold dark matter + baryons, linear
PKNL / PKNLCB: total matter / cold dark matter + baryons, non-linear

## PKNL_v2 and PKNLCB_v2 retrained (October 2026)

The non-linear v2 emulators were retrained on the same training set after removing
corrupted spectra. In about 16-22% of training samples at z < 5, CLASS-EDE returned the
linear spectrum in place of the HMcode result without raising an error; a few dozen more
samples were HMcode glitches. These were removed (103k samples remain) and the networks
retrained with the same architecture (4 x 512, same activation and input parameters).

Validation on the held-out test set (about 26k CLASS-EDE spectra, 0 <= z <= 20, drawn over
the full parameter ranges), |P_emu / P_CLASS - 1|, k in 1/Mpc:

| emulator         | percentile | k=0.01 | k=0.1 | k=1   | k=10  |
|------------------|-----------:|-------:|------:|------:|------:|
| PKNL, previous   | 95%        | 3.2%   | 4.4%  | 46%   | 95%   |
| PKNL, retrained  | 95%        | 0.09%  | 0.17% | 0.22% | 0.47% |
| PKNLCB, previous | 95%        | 3.9%   | 5.0%  | 48%   | 95%   |
| PKNLCB, retrained| 95%        | 0.08%  | 0.16% | 0.23% | 0.47% |

By redshift (PKNL retrained; PKNLCB within 0.1% of these), percentiles in %:

| z     | N_test | 68% (k=0.01 / 0.1 / 1 / 10) | 95% (k=0.01 / 0.1 / 1 / 10) |
|-------|-------:|-----------------------------|-----------------------------|
| 0-1   | 4161   | 0.05 / 0.07 / 0.07 / 0.17   | 0.13 / 0.20 / 0.18 / 0.41   |
| 1-3   | 7682   | 0.04 / 0.06 / 0.05 / 0.11   | 0.08 / 0.17 / 0.14 / 0.25   |
| 3-5   | 5728   | 0.03 / 0.05 / 0.05 / 0.10   | 0.07 / 0.15 / 0.21 / 0.32   |
| 5-10  | 6414   | 0.03 / 0.05 / 0.07 / 0.14   | 0.07 / 0.15 / 0.28 / 0.66   |
| 10-20 | 1770   | 0.04 / 0.06 / 0.12 / 0.35   | 0.10 / 0.17 / 0.39 / 1.28   |

At z > 10 the training set has fewer non-linear samples (HMcode output is often unavailable
there), hence the larger errors at k > 1/Mpc.

The previous weights are in the git history of this repository. File names and input
parameters are unchanged; class_sz picks up the new weights after `git pull` in
`class_sz_data_directory/ede`.
