# KP Automatic Model Selection Design

## Goal

Replace manually authored polynomial-order choices in `kp model` with an
auditable procedure that minimizes the number of independent real fit
parameters while preserving the weighted low-energy bands.  The selection
order is `Kinect`, then `intra`, then `inter`, followed by symmetry-closed term
ablation and a joint correction sweep.

The primary quality target is the low-energy weighted band error.  Mean
subspace overlap of at least `0.95` is the default quality target.  Full-band
and matrix residuals are guards against a locally good but globally distorted
model.

## Production invariants

- Every candidate uses the same production Hamiltonian responses used by band
  assembly and export.
- `complete_linear_v2` candidates are recompiled for each vocabulary.  A
  highest-order compiled basis is not sliced by nominal degree because finite
  harmonics, center conversion, adjoint closure, and rank pivots can mix or
  change the retained response channels.
- Selection groups are closed under the exactified symmetry group,
  Hermitian/adjoint closure, and any real/imaginary channel relation.
- Complexity is the number of independent fitted real response coefficients,
  not the number of YAML term records.
- Candidate comparison uses one common target-only weighting policy.  A
  residual-dependent reweighting round may refine the selected model, but it
  cannot define the metric used to compare different candidates.
- Diagnostic or approximate symmetry matrices never enter production model
  selection.

## Why a staged search needs guarded targets

A `Kinect`-only Hamiltonian cannot reproduce the complete bands when intra- or
inter-sector couplings are physically present.  Selecting its order from the
complete band error would reward high-order kinetic terms for absorbing
omitted interactions.

For kinetic selection, use the centered target

```text
Delta H(k) = H_eff(k) - H_eff(k_ref)
```

and the centered candidate response on kinetic-identifiable support.  The
reference removes constant onsite contributions.  Target eigenvectors from
the full `H_eff` define a degeneracy-safe low-energy spectral weight, so the
linear score emphasizes matrix directions that affect the requested bands.

Before staged fitting, compute principal correlations between the `Kinect`,
`intra`, and `inter` response subspaces on the selection grid.  A response
component that overlaps a later family above the identifiability tolerance is
not assigned in isolation; it is deferred to the first joint stage containing
all overlapping families.

## Candidate search

### 1. Kinetic order

Build nested kinetic vocabularies for configured candidate orders.  Fit the
centered target on blocked training folds and evaluate on held-out blocks.
Choose the lowest order whose mean validation loss is no greater than the best
mean loss plus one standard error.

The selected kinetic order is frozen for the next stage, but its coefficients
are not.  Coefficients are refitted jointly whenever another family is added.

### 2. Intralayer order

For every intralayer order candidate, rebuild and jointly fit the selected
kinetic vocabulary plus the candidate intralayer vocabulary.  Use the union of
their identifiable response support and the same fixed target spectral
weights.  Select the lowest near-best order by the same one-standard-error
rule.

### 3. Interlayer order

Add interlayer candidates and jointly refit all selected families.  At this
stage the complete weighted band loss is meaningful.  Candidate records
include weighted RMS and maximum errors, subspace overlap, all-band guards,
matrix guards, condition diagnostics, and parameter counts.

### 4. Symmetry-closed term ablation

Use normalized response amplitude `abs(c_j) * response_scale_j` only to screen
obviously negligible groups.  For every remaining group, remove the complete
group, refit all other coefficients, and measure the held-out loss increase.
Drop the group when the reduced model remains on the best one-standard-error
plateau and passes the overlap/global guards.

The reported importance of a group is

```text
validation loss after group removal and refit - full validation loss.
```

This is preferred to a bare coefficient magnitude because correlated terms
can exchange coefficient weight without changing the Hamiltonian.

### 5. Joint correction sweep

After the forward family search, test lowering each selected family order by
one step and, when the selected order lies on the search boundary, raising it
by one step.  Jointly refit all families for every correction candidate.  Stop
after a complete sweep makes no change or after a small configured iteration
limit.

## Validation splits and metrics

Adjacent points on a k path are correlated, so random point-wise folds are not
used.  K-path folds hold out contiguous pieces of each path segment while
keeping exact duplicate endpoints and symmetry-related points in the same
fold.  A two-dimensional selection mesh instead holds out radial/angular or
symmetry-orbit blocks.

For fixed target-derived band weights `w_kn`, the primary validation metric is

```text
sqrt(sum_kn w_kn * (epsilon_model_kn - epsilon_target_kn)^2 / sum_kn w_kn).
```

Weights are normalized to mean one.  The band window is completed across
degenerate boundaries.  Candidate overlap is evaluated for the whole selected
subspace using principal angles, not state-by-state overlaps that can exchange
inside a degenerate multiplet.

Selection status is:

- `PASS`: overlap is at least `0.95`, guards pass, and the candidate lies on
  the near-best validation plateau.
- `WARN_BEST_AVAILABLE`: no candidate reaches the quality target; select the
  simplest candidate statistically indistinguishable from the best observed
  quality and report every unmet target.
- `FAIL`: numerical certification fails or the best candidate violates a
  configurable safety floor.  Do not hide projector, gauge, band-window, or
  target-data failures by increasing model complexity.

## Configuration and outputs

The initial interface is an opt-in `fit.model_selection` mapping.  Explicit
`model.max_order` remains supported as an override and as the search ceiling.
The selection report records:

- candidate family orders and vocabulary hashes;
- fold definitions and fixed weight hashes;
- independent real parameter and active-group counts;
- per-fold weighted errors, overlap, and guards;
- response-space identifiability diagnostics;
- selected/rejected candidates with reasons;
- ablation importance and selection frequency;
- final status and unmet quality targets.

Normal release outputs contain a compact selection summary.  Candidate logs,
intermediate models, and external-data plots remain under `validation_runs/`.

## Initial validation scope

1. Synthetic nested response models with a known minimal order and redundant
   symmetry-related terms.
2. Bilayer MoTe2 K-valley examples under `examples/mote2_3.89/`.
3. MgI2 M-valley examples under `examples/mgi2_3.89/`.
4. Focused configured-model tests followed by the default release test suite.
