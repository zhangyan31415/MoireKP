# Public Nonlinear Vectorized Jacobian Design

## Problem

The public `fit.method: nonlinear` path evaluates the complete-response band
Jacobian by looping over every retained response channel, every fit k-point,
and every selected band in Python.  For the PtSe2 19Q all-band case this means
456 channels, three band-loss k-points, and 38 bands for every loss evaluation.
The response basis and the k-points do not change during optimization, so most
of this work is repeated unnecessarily.

The legacy PtSe2 all-band refinement materialized a response tensor and used a
vectorized Hellmann--Feynman contraction.  Its nonlinear stage completed in
about 17 seconds, while the new public complete-response path takes roughly two
minutes.  The public loss definition and YAML interface are not responsible
for that difference; the repeated sparse Python traversal is.

## Constraints

- Do not change the public YAML interface or loss formulas.
- Keep the full-Hamiltonian, one-sided, two-sided, and band-energy losses exact.
- Start nonlinear optimization from the exact public linear solution.
- Do not regenerate semantic continuum operators during optimization.
- Keep a bounded-memory fallback for larger models.
- Preserve the current SciPy optimizer for the first optimization step so that
  backend speed can be measured without also changing optimizer behavior.

## Chosen Design

Before nonlinear optimization, compile the selected complete-response channels
at the configured `band_kpoints` onto their union of nonzero matrix entries.
The resulting array has shape

```
(n_band_kpoints, n_union_support_entries, n_solver_channels)
```

and is independent of the fit coefficients.

For a coefficient displacement from the exact linear solution, construct the
model Hamiltonian by a vectorized contraction on this cached design followed
by one scatter onto the dense Hamiltonian rows.  Compute all selected-band
Hellmann--Feynman derivatives at once with an `einsum` contraction between the
cached response design and the current eigenvectors.  This replaces the nested
channel/monomial/band Python loops.

The cached representation is preferred to a full
`(Nk, Nchannel, dim, dim)` response tensor because it omits structurally zero
matrix entries.  The numerical object is still the exact frozen compiled
response basis; no operator is re-derived.

## Memory Policy

The fast backend is automatic and has no YAML option.  Its estimated cache size
is compared with an internal 768 MiB limit.  If the union-support design fits,
the vectorized backend is used.  If it does not fit, the code falls back to the
existing low-memory sparse traversal and reports that decision.

For the current PtSe2 case the expected cache is about 25 MiB for three
band-loss k-points.  A 61-point PtSe2 path is expected to remain below the
limit at about 500 MiB.

## Optimizer

The first implementation keeps `scipy.optimize.least_squares` and the existing
`max_steps` meaning.  This isolates the Jacobian/backend change and should
produce the same objective and a numerically equivalent optimizer trajectory.
Switching to the legacy custom Gauss--Newton solver is a separate follow-up only
if the vectorized backend does not meet the performance target.

## Reporting

The terminal and nonlinear result metadata report:

- backend name;
- union-support entry count;
- cached byte count;
- whether the memory fallback was used;
- one-time cache construction time.

Loss-evaluation progress remains `1/max_steps` through `max_steps/max_steps`.

## Verification

1. A dense-oracle unit test compares cached Hamiltonians, band residuals, and
   analytic Jacobians with the existing sparse implementation for top and
   bottom windows.
2. A reuse test proves that response compilation occurs once for multiple loss
   and Jacobian evaluations.
3. A memory-policy test proves that oversized estimates select the sparse
   fallback without changing numerical results.
4. Existing public linear/nonlinear configuration and export tests remain
   green.
5. The PtSe2 19Q, three-k-point, 38-band configuration is rerun on gpuh204.
   Residuals and final band errors must remain numerically equivalent, and the
   nonlinear stage must improve by at least 3x, with a target of 17--30 seconds.

