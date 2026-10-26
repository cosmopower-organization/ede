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

Accuracy on the held-out test set, |P_emu / P_CLASS - 1|:

| emulator         | percentile | k=0.01 | k=0.1 | k=1   | k=10  |
|------------------|-----------:|-------:|------:|------:|------:|
| PKNL, previous   | 95%        | 3.2%   | 4.4%  | 46%   | 95%   |
| PKNL, retrained  | 95%        | 0.10%  | 0.20% | 0.24% | 0.48% |
| PKNLCB, previous | 95%        | 3.9%   | 5.0%  | 48%   | 95%   |
| PKNLCB, retrained| 95%        | 0.10%  | 0.19% | 0.24% | 0.47% |

The previous weights are in the git history of this repository. File names and input
parameters are unchanged; class_sz picks up the new weights after `git pull` in
`class_sz_data_directory/ede`.
