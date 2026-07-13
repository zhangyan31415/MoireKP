# Faithful Response Fit Design

## Objective

Improve the unrefined ZrS2 Gamma q04 continuum model before any sparse-model
selection.  The faithful model must be selected on held-out k points and must
control both low-energy energies and subspace leakage.

Acceptance targets are:

- held-out top-four raw band RMS no larger than 2 meV;
- held-out top-four maximum error no larger than 5 meV;
- top-two projector overlap mean at least 0.99;
- top-two projector overlap minimum at least 0.95;
- no closing of the reference top-two/top-three boundary gap.

## Core observation

The current `linear_low_subspace` solver fits only the reference-subspace block

\[
P\,\Delta H\,P.
\]

This controls the low-energy matrix inside the target space but not the
off-block response responsible for eigenvector leakage.  The faithful linear
objective must also include

\[
P\,\Delta H\,Q,\qquad Q=1-P,
\]

with optional inverse-gap weighting.  Both blocks remain linear in the term
coefficients because the projector is taken from the fixed reference Heff.

## Prototype architecture

1. Build symmetry-exactified candidate models with interlayer order 8 and 10.
2. Use seven path points for the initial full-matrix coefficient fit.
3. Precompute real and imaginary term responses with the existing direct
   response routine.
4. Project each response into the reference top-four block and its coupling to
   the neighboring eight lower bands.
5. Solve a response-scaled linear least-squares problem containing `PP`, `PQ`,
   and a trust term around the initial full-matrix solution.
6. Evaluate every candidate on all 61 path points, separating training and
   held-out metrics.
7. Write all prototype artifacts under `validation_runs/`; do not modify or
   reference them from release-facing files.

## Later production integration

If the prototype meets the acceptance targets, add the cross-subspace block to
the existing `linear_low_subspace` solver.  Only after a faithful model passes
the guards should response-aware backward group ablation generate an
interpretable Pareto path.  The faithful model remains the default production
and topology model.

For a linear weighted least-squares optimum, the exact delete-and-refit cost of
a coefficient group can be computed from its Gram block,

\[
\Delta L_g=c_g^T\left[(A^TWA)^{-1}_{gg}\right]^{-1}c_g,
\]

with pivoted QR/SVD handling rank-deficient directions.  This replaces pruning
by raw coefficient magnitude.

