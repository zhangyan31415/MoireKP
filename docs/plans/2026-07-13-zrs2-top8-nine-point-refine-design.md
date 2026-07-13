# ZrS2 Gamma q04 Top-Eight At-Most-Ten-Point Refine Design

## Objective

Refine the symmetry-exactified ZrS2 AB Gamma q04 continuum candidate so that
all top-eight bands and their joint low-energy subspace agree with the
fixed-Schur Heff target along the complete 61-point path.  Refinement may use at
most ten path points; all other points are held out from optimization.

The acceptance gates on the full path are:

- top-eight raw energy RMS no larger than 3 meV;
- top-eight raw maximum error no larger than 8 meV;
- top-eight projector overlap minimum at least 0.95;
- top-four and top-two projector overlap minima at least 0.95;
- the reference top-eight/top-nine boundary gap does not close.

## Starting point

Use the current order-eight linear polynomial model plus the fixed,
symmetry-exactified active-sampling correction as the initial Hamiltonian.  The
correction is not re-interpolated during coefficient refinement.  Rebuild the
order-eight model object so its active real and imaginary polynomial term
responses can be used as symmetry-preserving refine directions.

## Refine and holdout points

An explicit top-eight audit of the starting candidate finds its worst top-eight
projector overlap at index 12.  Use the full allowed budget of ten path indices:

```text
0, 10, 12, 20, 21, 30, 40, 46, 47, 50
```

They contain the independent high-symmetry endpoints, segment midpoints, the
current worst top-two, top-four, and top-eight overlap points, and the current
worst top-four energy-error point.  Index 60 duplicates Gamma and is left in
the holdout set.  The remaining 51 rows are never used to solve for coefficients.

## Refine objective

At each refine point, diagonalize the fixed-Schur target and form its top-eight
projector `P`.  Let `Q` contain the neighboring eight lower target bands.  For
each exact polynomial term response, construct two fixed-reference blocks:

```text
PP = P^dagger (H_target - H_model) P
PQ = P^dagger (H_target - H_model) Q
```

The PP rows control the top-eight energies and internal mixing.  The PQ rows
control leakage of the top-eight subspace into nearby states.  Add a
response-scaled trust penalty around the initial polynomial coefficients.
Sweep only a small fixed grid of PQ and trust weights, solve each candidate by
rank-revealing linear least squares, and rank candidates using held-out metrics.

If the best linear candidate is close to but does not satisfy the gates, apply
at most a few damped nonlinear/Gauss-Newton steps on the same ten points.  The
polynomial term-response basis remains fixed, and a line search rejects any
step that worsens the combined held-out score.

## Validation and outputs

Write scripts, Hamiltonians, metrics, and plots only under the existing
`validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/` directory.
Record train, holdout, and full-path top-eight energy errors; top-two, top-four,
and top-eight projector overlaps; boundary gaps; linear-system rank and
conditioning; selected weights; runtime; and host identity.

Do not modify production package code in this experiment.  Production
integration is considered only if the full-path gates pass and the result is
reproducible on bigmem002.
