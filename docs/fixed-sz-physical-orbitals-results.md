# Fixed-Sz physical-orbital experiment: results

Branch: `codex/fixed-sz-physical-orbitals`. Sampled 3.15°/19Q ZrS2 native TAPW experiment; no old exports changed. Both runs exited successfully on test001. These are projection prototypes, not final continuum fits or production HF/ED results.

## Findings

All four explicitly defined orbital spaces have a k/Q-independent bare Pauli Sz and close under saved TR, C3 and C2 source actions. Closure residuals are below 3e-16; the largest projected unitarity/spin-covariance residual is about 7.1e-11. Fixed spin does not imply spin conservation: source spin-flip terms are retained.

Direct parent projections do not reproduce the selected native top six bands: errors are hundreds of meV and the subspace overlaps are poor. Therefore none of these tested direct parent Hamiltonians is an accurate compact replacement.

High-space reconstruction recovers the sampled native spectra and wavefunctions accurately, but changes the physical spin operator. This tradeoff remains after the test was extended beyond Gamma/K. A constant bare Sz is an exact definition of the chosen parent theory; it is not an exact replacement for the dressed microscopic observable.

## Sampled energy errors

Maximum absolute top-six error in meV, with one-to-one overlap matching and no energy alignment. Spin difference is ||Sz_dressed−Sz_bare||_2; it measures matrix difference, not just eigenvalue moduli.

| k index | States/Q | Direct | Fixed Schur | Normalized reconstructed | Min squared subspace overlap, reconstructed | Spin difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 254.6106 | 2.5791 | 0.034295 | 0.99986909 | 0.100522 |
| 0 | 16 | 224.2212 | 2.5735 | 0.037113 | 0.99985400 | 0.111409 |
| 0 | 32 | 213.8310 | 2.5485 | 0.011119 | 0.99994711 | 0.095547 |
| 0 | 48 | 349.3287 | 2.2666 | 0.015675 | 0.99998951 | 0.026625 |
| 10 | 8 | 344.5100 | 2.3685 | 0.018390 | 0.99990138 | 0.112751 |
| 10 | 48 | 303.8646 | 2.0905 | 0.016832 | 0.99998820 | 0.026509 |
| 20 | 8 | 311.0942 | 1.5360 | 0.004452 | 0.99996253 | 0.123128 |
| 20 | 48 | 263.2707 | 1.3336 | 0.007791 | 0.99999435 | 0.026231 |
| 40 | 8 | 333.3176 | 1.6356 | 0.006114 | 0.99997228 | 0.123836 |
| 40 | 16 | 313.7146 | 1.6345 | 0.005966 | 0.99996989 | 0.112890 |
| 40 | 32 | 299.2353 | 1.6164 | 0.008060 | 0.99998613 | 0.072619 |
| 40 | 48 | 284.8019 | 1.5644 | 0.010332 | 0.99999266 | 0.026120 |

Indices 0, 20 and 40 are Gamma, M and K; index 10 is an intermediate Gamma–M point. The 16/32-state candidates were screened at Gamma/K only.

Fixed-Schur overlaps/residuals/spins use bare F lifts, so they must not be called reconstructed physical-wavefunction checks. Only the normalized-reconstructed column uses Z=(F+GY)(I+Y†Y)^−1/2.

## What to retain from this attempt

- The new constant physical parent basis is usable as an opt-in starting point, including at the original 8 states/Q dimension. No rewrite of TAPW, symmetry generation, or the Schur formulas was needed.
- The 8-state normalized reconstruction is compact and accurate on the four checked points, but its physical Sz remains dressed. It is not a solution satisfying all three of compactness, strict physical Pauli Sz, and microscopic accuracy.
- The 48-state version reduces the bare-versus-dressed spin discrepancy but also retains far more orbitals. A small eigenvalue-modulus defect alone must not be used as evidence that its whole spin matrix is constant.
- Further work should optimize spin-independent molecular orbitals (including metal/hybridization tails) or enlarge the parent space with physically specified orbitals, then repeat independent momentum validation. Success is not guaranteed at eight states/Q.
- No continuum harmonic refit, source embedding/form-factor correction, full-BZ topology, or HF/ED run was performed. Existing results cannot be relabeled as validated by this prototype.

## Reproducibility

Environment `/data/home/zy/mambaforge`, Python 3.12.7, test001, one process with eight BLAS/OpenMP threads. Standing test001 direct-SSH authorization; no Slurm job was submitted. Source HEAD during execution: `23d4c0d6df75813ea94afbabdfec191c467efdf8`; exact executed module/runner copies and SHA256 values are retained with each run because development continued on this branch. User-owned pre-existing working-tree edits are not part of this prototype commit.

```bash
/data/home/zy/mambaforge/bin/python /data/work/zy/software/1.tapw_code/moirekp-release/scripts/zrs2_fixed_spin_trial.py --indices 0,40 --threads 8 --output /data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/screen_v1
```

Results: `/data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/screen_v1`; log: `/data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/screen_v1.log`. Final state: completed, exit 0.

```bash
/data/home/zy/mambaforge/bin/python /data/work/zy/software/1.tapw_code/moirekp-release/scripts/zrs2_fixed_spin_trial.py --indices 10,20 --candidates 8,48 --threads 8 --output /data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/path_check_v1
```

Results: `/data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/path_check_v1`; log: `/data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/path_check_v1.log`. Final state: completed, exit 0.

Verification: `/data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/verification.json`. Combined metrics: `/data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/zrs2_fixed_spin_20260927/combined_metrics.json`. The runner validates native reference eigenpair residuals; the final audit checks all twelve saved matrix archives for finite Hermitian matrices, exact bare spin square, and matching source-code snapshots.

Tests: 17 passed (8 new fixed-spin invariant tests plus 9 existing downfold tests). Read-only review found no core mathematical blocker; its Schur-lift labeling and provenance recommendations were incorporated in the follow-up runner and this report.
