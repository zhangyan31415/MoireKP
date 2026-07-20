# Autoproject Full-Pipeline Example Design

**Date:** 2026-07-20

## Goal

Publish and validate ten material-organized KP examples below
`examples/autoproject/`.  Every case must run the canonical sequence

```text
kp project -c <config>
kp model   -c <config>
```

from one complete, directly editable YAML file.

## Published layout

```text
examples/autoproject/
├── README.md
├── expected_selections.yaml
├── run_all.py
├── mote2_3.89/
│   ├── configs/{k1_spinless_q06,k1_spinful_q06}.yaml
│   └── runs/
├── mgi2_3.89/
│   ├── configs/{gamma_spinful_q04,m1_spinless_q07,m1_spinful_q07}.yaml
│   └── runs/
├── zrs2_3.15/
│   ├── configs/gamma_spinful_q04.yaml
│   └── runs/
├── mote2_aab_5.09/
│   ├── configs/{gamma_spinful_q04,k1_a_q04,k1_b_q04}.yaml
│   └── runs/
└── ptse2_7.34/
    ├── configs/gamma_spinful_q04.yaml
    └── runs/
```

The legacy shared `examples/autoproject/configs/` directory is removed after
all manifest references and tests use the material-local paths.

## Case inventory

The collection contains exactly ten cases:

1. MoTe2 3.89 K1 spinless q06;
2. MoTe2 3.89 K1 spinful q06;
3. MgI2 3.89 Gamma spinful q04;
4. MgI2 3.89 M1 spinless q07;
5. MgI2 3.89 M1 spinful q07;
6. ZrS2 3.15 Gamma spinful q04;
7. 1+2 A-AB MoTe2 5.09 Gamma spinful q04;
8. 1+2 A-AB MoTe2 5.09 K1 A q04;
9. 1+2 A-AB MoTe2 5.09 K1 B q04;
10. PtSe2 7.34 Gamma spinful q04.

## Configuration contract

Each YAML is standalone.  It contains the complete `system`, `project`,
`symmetry`, and `model` sections and does not depend on hidden low/high
profiles.  All user-tunable model decisions are explicit:

- harmonic counts;
- kinetic, intralayer, and interlayer polynomial orders;
- linear or nonlinear fit method;
- Hamiltonian fit rows;
- target band count;
- one-sided and two-sided Hamiltonian weights;
- nonlinear band-loss rows, weight, and step limit;
- comparison band slice and plot settings.

Validated material-specific harmonic support, polynomial orders, and fit rows
are copied from the existing release configs.  No automatic term pruning or
model-selection profile is enabled.

MoTe2, MgI2, and 1+2 A-AB MoTe2 use the public linear method with
`one_sided_weight: 0.0` and `two_sided_weight: 0.0`.  ZrS2 uses the public
linear method with `one_sided_weight: 0.0` and
`two_sided_weight: 300.0`.  PtSe2 remains unchanged from the reviewed
autoproject configuration: nonlinear fitting, 38 target bands, Hamiltonian
rows and band-loss rows `[0, 20, 40]`, one-sided and two-sided weights `1.0`,
band-loss weight `1.0`, and its current harmonics/orders/plot settings.

Projection selection remains case-specific.  Existing automatic-selection
cases continue to use `project.selection`; the reviewed MgI2 Gamma
second-run example may retain its explicit resolved `nlow_state_list`.

## Strict compiler behavior

The canonical `kp project -> kp model` route must not change algorithms after
an error.  Production automatic fallbacks from factorized actions, p=0 fixed
space, or finite-p symbolic atoms to full CSR Reynolds are removed.  Missing,
incompatible, or invalid certified symmetry data terminate `kp model` with the
operation, artifact path, compiler stage, and reason.  The full CSR Reynolds
implementation remains available only as a test/development correctness
oracle.

## Runner behavior

`run_all.py` treats `project` and `model` as two ordered stages for each case.
It checks the resolved low-energy selection after `project` and only then runs
`model`.  A completed case must contain non-empty:

- projection `heff.npy`, `basis.npz`, `wavefunctions.npz`, Q-block and band
  comparison PDFs;
- the KP symmetry package;
- compiled/fitted model data and coefficient registry;
- model-vs-Heff band comparison output;
- a model artifact identifying the executed response compiler and fit method.

`--check-only` validates both stages without running commands.  An optional
stage selector permits project-only or model-only diagnosis, while the default
always runs the full ordered pipeline.

## Parallel validation

Different cases use distinct output directories and may run on different
nodes concurrently.  Within one case, `kp model` never starts before
`kp project` completes successfully.  Node logs and run summaries stay under
`validation_runs/`; generated example outputs stay below each material's
ignored `runs/` directory.

## Acceptance criteria

- all ten YAML files load through the public config parser;
- every project selection matches `expected_selections.yaml`, including the
  two intentional A-AB K-sector warnings;
- all production symmetry stages pass;
- all ten `kp model` commands finish without any fallback diagnostic;
- ZrS2 records two-sided weight 300 and all other linear cases record zero
  one-/two-sided weight;
- PtSe2 records the unchanged nonlinear 1/1/1 weighting contract;
- every case writes the required projection/model artifacts;
- targeted autoproject/config/response tests pass;
- the default non-slow, non-external-data release suite is rerun and any
  unrelated dirty-worktree failures are reported exactly.
