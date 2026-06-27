# TAPW

TAPW provides truncated atomic plane-wave workflows for twisted-material band calculations, orbital analysis, and topology post-processing.

## Features

- Band-structure calculations for twisted bilayer systems.
- Valley workflows for K, K prime, Gamma, and M points.
- C3-aware configuration and analysis utilities.
- MPI, PETSc, and SLEPc based solver support through the release conda environment.
- Orbital analysis and fatband plotting tools.

## Installation

Install from the repository root:

```bash
conda env create -f environment.yml
conda activate moirekp
```

The root environment includes `mpi4py`, `petsc4py`, `slepc4py`, and editable installation of this package.

## Basic Usage

Create a starter TAPW working directory:

```bash
tapw init -o output_dir
```

Edit `output_dir/config.yaml` and `output_dir/bands.yaml`, especially the paths to `H.dat`, `S.dat`, and `openmx.dat`.

Run a band calculation and optional symmetry analysis with the same config:

```bash
cd output_dir
tapw run -c config.yaml
tapw symm -c config.yaml
```

New configs use the canonical output layout by default:

```yaml
output_layout:
  style: canonical_v1
  root: outputs
  q_shell: q04
```

For the default spinful profile, TAPW writes `outputs/K1/q04` for a K1 valley
run. Non-default spin profiles should be explicit, for example
`profile: K1_up` or `profile: K1_spinless`.

The canonical band workflow writes user-facing names:

```text
outputs/K1/q04/
  manifest.json
  band/
    manifest.json
    energies_vbm.txt
    energies_cbm.txt
    wavefunctions_vbm.npy
    wavefunctions_cbm.npy
    hamiltonian_k.npy
    g_vectors_group1.npy
    g_vectors_group2.npy
    kpoints.npy
```

Plot the generated bands:

```bash
cd output_dir/outputs/K1/q04/band
tapw plot -c ../../../../bands.yaml
```

Legacy configs without `output_layout.style: canonical_v1` keep the existing
`Q_shell_<n_g>` names and may use `band_data/` instead of `band/`.

## Chern Post-Processing

```bash
cd output_dir
tapw chern -c config.yaml --n_g 4 --num_processes 100 --num_chern 20
cd outputs/K1/q04
tapw topo -c ../../../config.yaml -b -1 -2 -v 1 > tapw_chern.log
```

## Orbital Analysis

```bash
Q_SHELL_DIR=Q_shell_4
cd "output_dir/${Q_SHELL_DIR}"
tapw orbital . --config ../config.yaml --valley Gamma --band CBM
tapw fatband . --valley Gamma --band CBM --orbital-dir orbital_analysis --output-dir orbital_plots
```

Short aliases are the recommended release-facing commands. Longer compatibility forms remain supported: `tapw run --config ... --mode symmetry` is equivalent to `tapw symm -c ...`, and `tapw postprocess-memmap` is equivalent to `tapw final`. Legacy script entry points remain supported for existing scripts: `tapw-calc`, `tapw-config`, `tapw-plot`, `tapw-chernpost`, `tapw-orbital`, and `tapw-plot-orbital`.

## Examples

Release-facing TAPW examples live under `examples/tapw/`. Dataset provenance and unresolved release metadata are tracked in `examples/data-manifest.yaml` and `RELEASE_BLOCKERS.md`.
