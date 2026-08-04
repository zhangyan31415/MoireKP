# Release Examples And Automatic Harmonics Design

**Date:** 2026-08-04

## Goal

Publish a compact, audited examples tree in which low-energy projection and
harmonic counts are automatic, polynomial orders remain explicit physical
inputs, PtSe2 uses nonlinear fitting, and every other released model uses
linear fitting.

## Release Scope

The release contains ten KP cases:

- bilayer MoTe2 K: spinful and spinless;
- bilayer MgI2 Gamma: spinful;
- bilayer MgI2 M: spinful and spinless;
- 1+2 A-AB MoTe2: Gamma, K1-A, and K1-B;
- bilayer PtSe2 Gamma;
- bilayer ZrS2 Gamma.

The corresponding seven material TAPW configurations remain. Legacy MoTe2
and MgI2 KP configurations, BiTeI, and the material-specific 9.43-degree TAPW
examples leave the release-facing tree. The synthetic KP smoke fixture moves
under `tests/fixtures`. Generic TAPW templates and K-path inputs remain.
Ignored local outputs, validation runs, and `examples/toZY` are not deleted.

## Automatic Projection Contract

Every released KP case declares `project.selection: auto` and omits an active
`project.nlow_state_list`. The audited automatic result remains immediately
below it as commented YAML. The comment explains that uncommenting the list
turns the case into an explicit selection and bypasses the automatic search.

Release tests extract the commented list and compare it with the automatic
selection artifact for the same case. Equality means the same ordered low-band
indices and the same low-energy span dimension; ordinary gauge-equivalent
frame coordinates are not compared column by column.

## Automatic Harmonic Contract

Released cases omit `model.harmonics`. `kp model` derives the candidate
envelope from the actual Q sets and active model sectors:

1. enumerate the distinct intra- and inter-sector Q-difference shells that
   actually occur;
2. generate all physically supported intra/inter count pairs;
3. build their Hermitian support masks;
4. remove empty, out-of-range, and duplicate masks;
5. evaluate the remaining masks with the existing Heff harmonic-ablation
   metrics on a small representative k sample;
6. select the smallest accepted support on the existing quality plateau;
7. compile and fit the continuum model only once for that selected support.

No release config exposes `ladder`, `grid`, `max_shell`, or candidate pairs.
An expert may still write `model.harmonics` explicitly to bypass automatic
selection. The selected shell counts, candidate count, RMS/maximum errors,
subspace overlap, and timing are recorded in terminal output and model
metadata.

The geometry-derived candidate set may contain roughly 100--200 masks for the
larger Q shells. These are inexpensive masked-Heff eigensolves, not complete
model fits. Geometry and shell maps are reused, masks are deduplicated, and
evaluation may be chunked to bound memory.

## Model Inputs

`model.max_order` remains explicit in every case. PtSe2 Gamma uses
`fit.method: nonlinear`; the other nine cases use `fit.method: linear`.
The 1+2 MoTe2 Gamma case uses the validated kinetic/intralayer/interlayer
orders 8/6/8.

## Documentation And Manifest

The examples README and each material README list only the released cases,
explain automatic projection and harmonics, show the commented explicit
selection override, and distinguish tracked inputs from external datasets.
The data manifest lists all five material families and their exact TAPW/KP
commands. License, DOI, and public data URL fields remain explicit release
blockers until real values are supplied; the implementation must not invent
them.

## Verification

Unit and contract tests cover dynamic candidate generation, duplicate-mask
removal, inactive-family handling, configuration inventory, automatic
projection comments, omitted harmonics, explicit orders, and fit methods.

Initial material validation runs two representative cases:

- MoTe2 K spinful q06, covering the ordinary two-sector K workflow;
- MgI2 M spinless q07, covering asymmetric Q-shell support.

Each run records the immutable input config, output directory, command, host,
return code, project/model timing, selected low bands, selected harmonic
counts, and model RMS/maximum band error. After those pass, the same protocol
expands to all ten release cases.
