# MgI2 Gamma Linear Sparse Model Selection Design

## Objective

Select the smallest useful MgI2 Gamma q05 and q06 continuum models while
keeping the fitting problem linear in the model coefficients. Fix the
harmonic support at intra/inter `3/3` and use the fixed-Schur projected Heff as
the common reference.

## Baseline and comparison contract

For each q shell, first build an unpruned `3/3` model with maximum family
orders `10/6/10`, fit indices `[0, 40]`, and no target weighting. Every sparse
candidate is compared with that shell's baseline and the same projected Heff;
q05 and q06 are never compared through different downfolding methods.

## Linear sparse path

Keep the Hamiltonian linear in its coefficients. Candidate generation may
reduce family order and remove Hermiticity- and symmetry-closed term groups.
After each reduction, refit all retained coefficients by linear least squares.
An optional frozen-target low-energy projector supplies larger weights for the
top bands; it must not invoke a nonlinear eigenspectrum optimizer.

Evaluate a ladder of retained group fractions rather than trusting coefficient
magnitude alone. Include the full `3/3` vocabulary as a mandatory guard
candidate.

## Selection

Use held-out k points, top-band weighted RMS and maximum error, all-band error,
and subspace overlap. The high profile targets overlap at least `0.95` and
selects the smallest candidate on the accuracy plateau relative to the full
`3/3` baseline. The low profile may use a wider error plateau while preserving
the top-band subspace. Report active Kinect, onsite, intra, and inter terms as
well as independent real parameters.

## Validation outputs

Store configs, logs, candidate tables, eigvalue arrays, and plots under
`validation_runs/`. Do not publish machine paths or validation products in the
release package. Verify that no remote process remains after each run.
