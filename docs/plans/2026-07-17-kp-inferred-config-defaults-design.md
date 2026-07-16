# KP Inferred Configuration Defaults Design

**Date:** 2026-07-17

## Goal

Remove four redundant declarations from ordinary MoireKP case files while
preserving fail-closed release behavior and expert overrides:

- infer `model.n_orb` from the projection contract;
- select omitted `model.harmonics` with the existing low-cost harmonic scan;
- default an omitted project gauge to automatic SCDM anchors;
- default an omitted downfolding method to `linearized_lowdin`.

Automatic resolutions are printed to the terminal.  This change does not add
`selected_model_config.yaml`, candidate JSON files, or recommendation plots.
Normal model artifacts continue to record the actual active term support.

## User-Facing Contract

An ordinary project/model case keeps the physical selections and may omit the
four derived settings:

```yaml
project:
  nlow_state_list:
    - [54, 55]
    - [54, 55]
  e_ref: -4.75

model:
  target_bands: top
  max_order:
    Kinect: 6
    intra: 2
    inter: 2
```

Explicit expert settings remain accepted.  The common rule is: omission uses
the release inference/default, an explicit expert setting is preserved, and a
structural contradiction fails.

## Orbital Counts

`kp model` treats the projection artifact as the authoritative source of the
continuum row layout.  It first reads `projection/basis.npz:nlow_state_list`
and resolves the physical-layer rows through `material.num_layer_list` into
the two model qset/sector counts.  If the projection artifact is unavailable,
the source case's `project.nlow_state_list` is the compatibility source.

An explicit `model.n_orb` is an assertion when projection metadata is
available.  A mismatch fails.  It is a legacy input only when neither
projection source is available.  The old inference
`heff_dim / (n_qset1 + n_qset2)` must not be used for unequal qset counts or
multilayer layouts because it cannot recover the row partition uniquely.

The terminal reports the resolved counts and their source.

## Harmonic Selection

If `model.harmonics` is omitted, `kp model` runs the existing harmonic
ablation scan before compiling the model basis.  It selects the smallest
low-cost candidate satisfying both

```text
plot RMS <= 1 meV
mean target-subspace overlap > 0.9
```

Candidates are ordered by total shell count, maximum per-family shell count,
intra/inter imbalance, and then the band residual.  If no candidate satisfies
the low-cost acceptance rule, model construction stops and asks for a wider
scan or an explicit harmonic specification.  It must not fall back to the
smallest unaccepted support.

The terminal prints the selected intra/inter counts, plot RMS and maximum
errors, mean subspace overlap, and the high-accuracy reference candidate.  The
selected counts feed the existing support-orbit harmonic generator.  No new
selection file or diagnostic plot is written.  Explicit `model.harmonics`
bypasses automatic selection.

## Gauge Default

If both `project.gauge` and a manual `project.norb_fix_list` are absent,
`kp project` behaves as if `project.gauge: auto` had been declared.  The
production path retains symmetry validation and fails when the source
symmetry information cannot validate the selected anchors.  A manual
`norb_fix_list` remains the expert override and is never combined with the
automatic gauge.

The terminal labels the resolved `auto_scdm` gauge as a default and reports
whether symmetry validation succeeded.

## Downfolding Default

If `project.downfold_method` and its legacy alias are absent, the default is
`linearized_lowdin`.  The reference energy `project.e_ref` remains an explicit
physical input and is not inferred from `material.efermi`.  If no high-energy
complement exists, the operation resolves to `first_order` because there is no
space to downfold.  Explicit downfolding methods remain supported.

The terminal prints the resolved method, whether it was defaulted, and the
reference energy when applicable.

## Compatibility And Failure Policy

- Existing explicit release cases keep their current behavior.
- Legacy model cases without `basis.npz` may use explicit `model.n_orb`.
- An explicit `model.n_orb` inconsistent with the projection artifact fails.
- Missing symmetry validation for the default automatic gauge fails.
- Missing `e_ref` for a nontrivial `linearized_lowdin` downfold fails.
- Failure to find an accepted low-cost harmonic candidate fails.
- Automatic values are visible in terminal output and normal model metadata;
  no new standalone recommendation artifacts are created.

## Verification

Targeted tests cover bilayer and 1+2 layer orbital inference, explicit-count
mismatch, automatic and manual gauge behavior, the default and explicit
downfold methods, accepted and rejected low-cost harmonic selection, explicit
harmonic preservation, terminal output, and absence of new recommendation
files.  The full release test suite remains the final gate.
