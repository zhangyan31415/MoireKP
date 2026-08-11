# MoireKP

<p align="center">
  <b>TAPW workflows and continuum <i>k·p</i> model construction for twisted layered materials</b>
</p>

<p align="center">
  <a href="README.zh.md">中文</a>
  ·
  <a href="examples/README.md">Examples</a>
  ·
  <a href="#kp-workflow">KP workflow</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Package" src="https://img.shields.io/badge/package-tapw%20%7C%20kp-4c6ef5">
</p>

MoireKP is a Python package for moiré electronic-structure workflows. It uses
real-space Hamiltonian and optional overlap matrices together with a standard
structure file to build truncated atomic plane-wave (TAPW) source models and
fit low-energy continuum models with symmetry diagnostics.

| Module | Role | Main outputs |
| --- | --- | --- |
| `tapw` | Build TAPW Hamiltonians and compute bands, source-space symmetry representations, band representations, and topology data. | `band/`, `symmetry/`, `symm_rep/`, `topology/` |
| `kp` | Inspect TAPW spectra, project low-energy spaces and symmetry actions, analyze band representations, and export standalone continuum models. | `inspect/`, `projection/`, `symmetry/`, `symm_rep/`, `model/` |

## Installation

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate moirekp
```

`environment.yml` installs the package in editable mode. Verify that the
commands come from the active environment:

```bash
which python
which tapw
which kp
tapw --help
kp --help
```

## Quick checks

These checks do not require the external example data:

```bash
python -c "import tapw, kp"
tapw --help
tapw init --help
kp --help
```

The numerical examples require external matrices and arrays that are not
included in the source repository.

## TAPW workflow

Generate a starter directory:

```bash
tapw init -o workdir
```

Edit `workdir/config.yaml`, then run the required workflows with the same
config:

```bash
tapw run      -c workdir/config.yaml
tapw symm     -c workdir/config.yaml
tapw symm-rep -c workdir/config.yaml  # optional band representations
tapw topo     -c workdir/config.yaml  # optional topology
```

A TAPW config has this structure:

```yaml
system:
  output: outputs
  structure: POSCAR
  hamiltonian: H.npz
  overlap: S.npz
  orbitals: {Mo: s3p2d1, Te: s3p2d2}
  twist_index: 8
  layers: [1, 1]
  spin: true

bands:
  valley: K1
  q_shell: 6
  efermi: -4.10
  save_hamiltonian: true
  kpath:
    labels: [G, M, K, G]
    points_per_segment: 20
    coordinates:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]

symmetry:
  valley: K1
  q_shell: 6
  efermi: -4.10
  representation:
    points:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]

topology:
  valley: K1
  q_shell: 6
  mesh:
    n_b1: 21
    n_b2: 21
    range_b1: [-0.5, 0.5]
    range_b2: [-0.5, 0.5]
  bands:
    vbm2: {sector: valence, indices: [-1, -2]}
  berry_curvature: [{bands: vbm2}]
  quantum_geometry: [{bands: vbm2}]
  wcc: [{bands: vbm2, loop: b2}]
```

`system.structure` may be a POSCAR, CIF, or another format supported by ASE.
No OpenMX input file is required: the structure and orbital definitions come
from `system.structure` and `system.orbitals`. `system.overlap` may be omitted
for an orthogonal basis. The in-plane Bravais family is inferred from the
structure. Hexagonal and square lattices are supported; other in-plane metrics
are rejected. `layers` is explicit, including for ordinary `[1, 1]` bilayers.

The config does not use workflow `enable` switches. The command selects the
section to execute.

## KP workflow

The same KP YAML drives low-energy projection, symmetry exactification,
band-representation analysis, and continuum-model fitting. For example:

```bash
CFG=examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml
kp inspect  -c "$CFG"  # optional source-spectrum diagnostic
kp project  -c "$CFG"
kp symm     -c "$CFG"
kp model    -c "$CFG"
kp symm-rep -c "$CFG"  # optional band representations
```

`kp project` selects the smallest acceptable low-energy subspace and writes
the projected Heff. `kp symm` is the production symmetry step: it projects the
TAPW raw-H actions into that basis and writes the exactified symmetry package
used by `kp model`. `kp model` fits and exports the standalone continuum model.
The optional `kp symm-rep` command uses the persisted projected k-points to
identify little-group operations and reports both raw and polar-unitary block
diagnostics.

A current KP config has this structure:

```yaml
system:
  name: MoTe2
  output: ../outputs/k1_spinless
  tapw_output: ../../tapw/outputs
  layers: [1, 1]
  spin: up
  orbital_order: Te-s3p2d2,Mo-s3p2d1,Te-s3p2d2
  cell:
    - [52.4951, 0.0, 0.0]
    - [-26.2476, 45.4621, 0.0]
    - [0.0, 0.0, 27.0]

project:
  valley: K1
  q_shell: 6
  efermi: -4.10
  target: valence
  selection: auto
  e_ref: -4.60

symmetry: {}

bands:
  kpath:
    labels: [G, M, K, G]
    points_per_segment: 20
    coordinates:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]
  compare_to_heff: true
  band_slice: [46, 54]

model:
  target_bands: top
  max_order: {kinetic: 10, intralayer: 4, interlayer: 4}
  fit:
    method: linear
    kpoints: [0, 2]
    bands: 8
    one_sided_weight: 0.0
    two_sided_weight: 0.0
```

Conventions:

- Omitting `project.nlow_state_list` enables automatic low-energy selection.
  An explicit list reuses a previously selected subspace and skips the search.
- `symmetry: {}` means that operations are inferred from the TAPW symmetry
  package; it does not make the `kp symm` command optional.
- `bands` is a top-level section shared by projection and model comparison.
- If `model.harmonics` is omitted, `kp model` selects harmonic counts
  automatically. An explicit value bypasses that selection.
- `model.fit` controls fit rows, band count, method, and optional low-energy
  weights. Weight choices are case-specific and are recorded in each config.
- `kp model` writes the standalone evaluator directly; there is no separate
  `kp export` step.

After `kp model`, enter the generated `model/` directory, edit the user
parameters near the top of `evaluate.py`, and run:

```bash
python evaluate.py
```

## Output layout

TAPW writes `band/`, `symmetry/`, `symm_rep/`, and `topology/`; KP writes
`inspect/`, `projection/`, `symmetry/`, `symm_rep/`, and `model/`. The complete
file layouts are listed in [tapw/README.md](tapw/README.md) and
[kp/README.md](kp/README.md).

## Examples

The repository includes five material directories and ten KP case configs:

| Directory | Tracked KP cases |
| --- | --- |
| `examples/mote2_3.89` | K1 spinful and spinless |
| `examples/mgi2_3.89` | Gamma spinful; M1 spinful and spinless |
| `examples/mote2_aab_5.09` | Gamma spinful; K1-A and K1-B single-spin sectors |
| `examples/zrs2_3.15` | Gamma spinful |
| `examples/ptse2_7.34` | Gamma spinful configuration; full workflow not yet tested |

Configs, structure files, and small templates are tracked. Large OpenMX
matrices, TAPW arrays, symmetry exports, and generated KP outputs remain
external and are not committed to the source repository.

## License

The MoireKP software and repository-authored documentation, configuration
files, and small examples are licensed under `LGPL-3.0-or-later`. See
[COPYRIGHT](COPYRIGHT), [COPYING.LESSER](COPYING.LESSER), and
[COPYING](COPYING) for the copyright notice and complete license terms.

This software license does not automatically cover external OpenMX/TAPW/KP
inputs, large example datasets, or generated artifacts.
