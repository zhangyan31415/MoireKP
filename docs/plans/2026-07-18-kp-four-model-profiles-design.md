# KP Four-Profile Automatic Model Design

## Goal

One configured `kp model` run produces four auditable continuum models:

- `linear/high`: the most accurate certified linear fit;
- `linear/low`: the smallest linear vocabulary on the primary low-energy quality plateau;
- `nonlinear/high`: the most accurate certified nonlinear band refinement;
- `nonlinear/low`: the smallest nonlinear-refined vocabulary that preserves the requested top or bottom bands.

The four profiles share the same target bands, fixed target-derived weights, path folds,
symmetry certification, and principal-angle overlap definitions.

## Candidate construction

The linear search remains the common vocabulary generator. It selects kinetic,
intralayer, and interlayer orders in stages, jointly refits all retained coefficients,
and prunes only symmetry- and Hermiticity-closed parameter groups. Harmonic support,
family orders, active groups, and independent real parameters are all recorded as
model-complexity coordinates.

Nonlinear candidates are refinements of several points on the linear complexity
frontier, not only the largest linear model. Each candidate keeps its vocabulary fixed
while the nonlinear solver optimizes its coefficients against the same primary band
target. This makes it possible for nonlinear fitting to improve accuracy without
silently inheriting the maximum linear vocabulary.

## Profile selection

Both high profiles require numerical and production-symmetry certification and the
configured subspace-overlap target. They minimize primary weighted validation error;
expanded-window error, maximum error, and complexity are deterministic tie breakers.
High may use more harmonics, higher orders, and more independent parameters.

Both low profiles use only the requested top or bottom band window, cluster-completed
at near-degenerate boundaries, as an accuracy gate. Expanded-window, all-band, and raw
matrix errors remain diagnostics and cannot reject a low candidate. Among candidates
on the adaptive primary-error plateau and above the overlap safety floor, low minimizes
complexity lexicographically:

1. independent real parameters;
2. active closed parameter groups;
3. harmonic support size;
4. sum and maximum of kinetic/intralayer/interlayer orders;
5. primary weighted maximum error.

The default low plateau is the maximum of two blocked-fold standard errors, the
configured relative tolerance, and a small absolute tolerance. If no candidate reaches
the overlap target but one remains above the safety floor, the best available compact
candidate is exported with `WARN_BEST_AVAILABLE`.

## Output and diagnostics

The model output contains independently runnable directories:

```text
model/
  linear/high/
  linear/low/
  nonlinear/high/
  nonlinear/low/
  auto_model_selection.json
  auto_model_selection.md
  candidate_metrics.csv
  model_complexity_frontier.pdf
```

The report compares primary RMS/max error, cluster-completed overlap, symmetry
residual, harmonic counts, family orders, active terms, and independent real parameters.
The legacy top-level result points to `nonlinear/high` when nonlinear refinement is
enabled and otherwise to `linear/high`.

## MgI2 Gamma acceptance check

The first external-data validation uses the existing 19Q and 31Q projected Hamiltonians.
Every profile is evaluated on the full 61-point path. Primary quality uses top-10 bands;
validation expands a boundary that cuts a near-degenerate cluster. A low profile is
accepted only if it materially reduces model complexity while preserving the primary
band error and principal-angle overlap. Results remain under `validation_runs/`.
