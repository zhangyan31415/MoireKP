# MoireKP

<p align="center">
  <b>TAPW workflows and continuum <i>k·p</i> model construction for twisted bilayer materials</b>
</p>

<p align="center">
  <a href="README.zh.md">中文</a>
  ·
  <a href="examples/README.md">Examples</a>
  ·
  <a href="#kp-workflow">KP workflow</a>
  ·
  <a href="examples/data-manifest.yaml">Data manifest</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Package" src="https://img.shields.io/badge/package-tapw%20%7C%20kp-4c6ef5">
  <img alt="Status" src="https://img.shields.io/badge/status-CPC%20release%20candidate-orange">
  <img alt="Data" src="https://img.shields.io/badge/data-external%20datasets-lightgrey">
</p>

MoireKP is a Python package for moire electronic-structure workflows. It starts from OpenMX-derived real-space Hamiltonian and overlap matrices, builds truncated atomic plane-wave (TAPW) source models, and fits low-energy continuum models with symmetry diagnostics.

| Module | Role | Main outputs |
| --- | --- | --- |
| `tapw` | Build TAPW Hamiltonians and compute bands, source-space symmetry representations, and topology data. | `band/`, `symmetry/`, `topology/` |
| `kp` | Inspect TAPW spectra, project low-energy spaces, project symmetry, and export standalone continuum models. | `inspect/`, `projection/`, `symmetry/`, `model/` |

The release interface uses one config per case, short commands, and canonical output directories. Legacy split configs, old CLI aliases, and old output layouts are not part of the release-facing API.

## Contents

- [What It Does](#what-it-does)
- [Installation](#installation)
- [Quick Checks](#quick-checks)
- [TAPW Workflow](#tapw-workflow)
- [KP Workflow](#kp-workflow)
- [Output Layout](#output-layout)
- [Examples And Data](#examples-and-data)
- [Release Status](#release-status)

## What It Does

MoireKP is designed for workflows that need both a TAPW source Hamiltonian and a fitted low-energy continuum model:

- Generate TAPW band data from OpenMX `H/S` matrices and structure inputs.
- Export TAPW raw-H source symmetry representation matrices.
- Compute Berry curvature, quantum geometry, and Wilson-loop/WCC data.
- Inspect TAPW band/Q-block spectra before choosing low-energy states.
- Build projected/downfolded effective Hamiltonians.
- Project TAPW source symmetry into the KP continuum basis.
- Fit and export standalone `model/evaluate.py` packages for band, topology, and figure-data generation.

## Installation

Create the release environment from the repository root:

```bash
conda env create -f environment.yml
conda activate moirekp
```

`environment.yml` installs the package in editable mode. After installation, verify that the commands come from the active environment:

```bash
which python
which tapw
which kp
tapw --help
kp --help
```

For manual environment rebuilds, use MKL BLAS/LAPACK. Avoid accidentally running stale `tapw` or `kp` scripts from `~/.local/bin`.

## Quick Checks

These checks run from a clean source clone. They do not require external OpenMX matrices, TAPW Q-shell arrays, or precomputed KP outputs.

```bash
python -m pytest tests/test_release_contract.py tests/kp/test_example_dependency_contract.py -q
tapw --help
tapw init --help
kp --help
```

Before a release tag, run the fast baseline:

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

## TAPW Workflow

Generate a starter config:

```bash
tapw init -o workdir
```

Edit `workdir/config.yaml`, then run the requested TAPW workflows with the same config:

```bash
tapw run  -c workdir/config.yaml
tapw symm -c workdir/config.yaml
tapw topo -c workdir/config.yaml
```

A typical release-style TAPW config has this shape:

```yaml
case:
  output_root: outputs

bands:
  valley: K1
  q_shell: 6
  efermi: -4.10
  kpath:
    labels: [G, M, K, G]
    points_per_segment: 40
    coordinates:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]

symmetry:
  valley: K1
  q_shell: 6

topology:
  valley: Gamma
  q_shell: 3
  mesh:
    n_b1: 21
    n_b2: 21
    range_b1: [-0.5, 0.5]
    range_b2: [-0.5, 0.5]
  bands:
    vbm2:
      sector: valence
      indices: [-1, -2]
  berry_curvature:
    - bands: vbm2
  quantum_geometry:
    - bands: vbm2
  wcc:
    - bands: vbm2
      loop: b2
```

The config does not use workflow `enable` switches. The command selects which section is executed.

## KP Workflow

KP uses one YAML file for automatic low-energy projection, symmetry
exactification, and continuum-model fitting:

```bash
kp project -c kp/configs/K1_q06.yaml
kp model   -c kp/configs/K1_q06.yaml
```

`kp project` selects the smallest acceptable symmetry-compatible low-energy
subspace, reports its layer, spin, and orbital content, writes the TAPW-band
and reference-Q-block comparison, and prepares the exactified continuum
symmetry package. `kp inspect` and standalone `kp symm` remain optional
diagnostic commands.

A typical KP config has this shape:

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
  e_ref: -4.60

symmetry: {}

model:
  target_bands: top
  harmonics: {intralayer: 3, interlayer: 3}
  max_order: {kinetic: 10, intralayer: 4, interlayer: 4}
  fit:
    method: linear
    kpoints: [0, 2]
    bands: 8
    one_sided_weight: 0.0
    two_sided_weight: 0.0
  bands:
    compare_to_heff: true
    band_slice: [46, 54]
```

Conventions:

- `system` identifies the material, source TAPW output, layer/spin convention,
  orbital ordering, cell, and output directory.
- `project` identifies the valley, Q shell, target band edge, and projection
  reference energy.
- Omitting `project.nlow_state_list` enables automatic low-energy selection.
  Experts may provide it only to reuse an already reviewed subspace and skip
  the search.
- The usual `symmetry: {}` is intentional. `kp project` reads TAPW raw-H
  symmetry, constructs the continuum-basis actions, and exactifies them;
  users do not write operation matrices.
- `model` contains the user-controlled harmonic support, polynomial orders,
  fit points, band count, and optional low-energy weights.
- `kp model` directly writes the standalone model. There is no separate release-facing `kp export` step.

Model topology is handled inside the exported standalone model. After `kp model`, edit the user parameters at the top of `model/evaluate.py` and run:

```bash
python evaluate.py
```

## Output Layout

TAPW canonical layout:

```text
outputs/
  K1/
    q06/
      band/
        energies_vbm.txt
        energies_cbm.txt
        wavefunctions_vbm.npy
        wavefunctions_cbm.npy
        hamiltonian_k.npy
        g_vectors_group1.npy
        g_vectors_group2.npy
        kpoints.npy
      symmetry/
        representations.npz
        residuals.csv
        summary.md
  Gamma/
    q03/
      topology/
        grid21x21_b1_m0p5_0p5_b2_m0p5_0p5/
          chern_summary.json
          berry_curvature_vbm2.txt
          berry_curvature_vbm2.pdf
          quantum_geometry_vbm2.txt
          quantum_geometry_vbm2.pdf
          quantum_geometry_vbm2_trace_condition.txt
          wcc_vbm2_loop_b2.txt
          wcc_vbm2_loop_b2.pdf
```

KP canonical layout:

```text
outputs/
  K1/
    q06/
      inspect/
        spectrum.txt
        scatter.pdf
        wavefunctions.npz
      projection/
        heff.npy
        eigvals.txt
        wavefunctions.npz
        basis.npz
        basis.md
        scatter.pdf
      symmetry/
        representations.npz
        residuals.csv
        summary.md
      model/
        README.md
        MODEL.md
        evaluate.py
        model_data.npz
        eigvals.npy
        band_comparison.pdf
        band_comparison_all.pdf
        q_lattice_harmonics.pdf
```

Valley and q-shell are directory levels. Non-default topology ranges are encoded in the grid id, for example:

```text
grid21x41_b1_0p0_0p5_b2_m0p5_0p5
grid21x41_b1_m0p5_0p0_b2_m0p5_0p5
```

## Examples And Data

The repository tracks configs and small template inputs. Large OpenMX matrices, TAPW arrays, symmetry exports, and precomputed KP outputs are external data and are not committed to the source repository.

Common entry points:

```text
examples/tapw/mote2_9.43/
examples/tapw/mgi2_9.43/
examples/mote2_3.89/
examples/mgi2_3.89/
```

Dataset provenance and release status are tracked in:

```text
examples/data-manifest.yaml
```

Once the external data are available, run TAPW and KP workflows using the configs in each example directory.

## License

The MoireKP software and repository-authored documentation, configuration files, and small examples are licensed under `LGPL-3.0-or-later`. See [COPYRIGHT](COPYRIGHT), [COPYING.LESSER](COPYING.LESSER), and [COPYING](COPYING) for the copyright notice and complete license terms.

This software license does not automatically cover external OpenMX/TAPW/KP inputs, large example datasets, or generated artifacts. Their provenance and license metadata are tracked separately in `examples/data-manifest.yaml` and must be confirmed before an archival data release.

## Release Status

The software license is resolved. External dataset licenses, the DOI, public data URLs, and checksums still need final confirmation. Before an archival release, review:

```text
RELEASE_BLOCKERS.md
RELEASE_VALIDATION.md
examples/data-manifest.yaml
```

Before tagging a formal CPC archive, run:

```bash
scripts/release_gate.sh
```

This enables final release checks and should fail until dataset licenses, DOI, public data URLs, checksums, and release blockers are resolved.
