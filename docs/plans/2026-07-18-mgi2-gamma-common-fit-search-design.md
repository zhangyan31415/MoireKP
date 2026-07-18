# MgI2 Gamma 19Q/31Q Common Fit Search Design

## Objective

Find one continuum-model fitting method that works unchanged for the MgI2
Gamma 19Q and 31Q projected Hamiltonians and Pareto-dominates their historical
baselines in top-10, top-12, and full-spectrum RMS and maximum band error.

The historical original-Heff baselines, using global top alignment, are:

| data set | top-10 RMS/Max | top-12 RMS/Max | full RMS/Max |
|---|---:|---:|---:|
| 19Q | 0.748/1.827 meV | 0.809/2.417 meV | 3.138/11.717 meV |
| 31Q | 1.497/4.263 meV | 1.527/4.970 meV | 3.655/10.281 meV |

Acceptance requires every listed metric to be no worse than the corresponding
baseline for both data sets, with at least one strict improvement.

## Fixed model space

Use the symmetry-aware Gamma 2x2 reduced fundamental-domain templates with
explicit-reduced term-space policy, harmonics 4/4, and kinetic/intralayer/
interlayer polynomial orders 10/6/10. This preserves the high-order terms used
by the historical 31Q model while keeping the compiled response space at the
observed 2198 channels, below the 3000-channel limit.

Do not use complete-all-pairs 10/6/10 because it produces 7072 candidates. Do
not use complete-all-pairs 6/4/4 as the primary search space because controlled
tests degraded the full spectrum strongly.

## Parallel candidates

1. Balanced linear objective: full supported Hamiltonian at unit weight plus a
   fixed top-12 two-sided projector with weight 10, representative path points
   0, 20, 40, and 60, and ridge 1e-10.
2. Guarded nonlinear refinement: start from the historical unweighted reduced
   fit and refine kinetic, onsite, and intralayer real/imaginary components on
   seven path points. Use top-12 band residual, a global matrix penalty,
   coefficient drift regularization, analytic-Jacobian Gauss-Newton, and a
   full-spectrum acceptance guard.
3. Hybrid: start from candidate 1 and apply the guarded nonlinear refinement
   from candidate 2. Reject or line-search any step that degrades top-10,
   top-12, or full-spectrum validation windows.

Each candidate is run for 19Q and 31Q with identical method-level parameters.
Only dimension-dependent band slices may differ.

## Parameter-count interpretation

Nonlinear refinement changes selected coefficients; it does not itself reduce
the number of active terms. Report separately: generated terms, compiled
candidate channels, retained response channels, nonlinear variables, and final
nonzero coefficients. Parameter reduction, if attempted, is a later prune and
refit stage protected by the same validation gate.

## Execution and artifacts

Run the three candidates independently on bigmem001, bigmem002, and bigmem003.
Write configs, logs, exit codes, metric tables, and rendered plots only below a
timestamped `validation_runs/` directory. Do not edit release-facing examples
or package code during the search.

If none of the initial candidates passes, use evidence from their losses and
acceptance reports to perform bounded searches over two-sided weight, matrix
weight, refinement variable families, and fit-point density. Keep the fixed
model space and the 3000-channel ceiling unless evidence shows that one must
change.
