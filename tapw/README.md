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
tapw config -o output_dir
```

Edit `output_dir/config.yaml` and `output_dir/bands.yaml`, especially the paths to `H.dat`, `S.dat`, and `openmx.dat`.

Run a band calculation:

```bash
cd output_dir
tapw calc --config config.yaml
```

Plot the generated bands:

```bash
cd output_dir/Q_shell_{n_g}/band
tapw plot --config ../../bands.yaml
```

Legacy outputs may use `band_data/` instead of `band/`.

## Chern Post-Processing

```bash
cd output_dir
tapw calc --config config.yaml --mode chern --n_g 4 --num_processes 100 --num_chern 20
cd output_dir/Q_shell_{n_g}
tapw chern-post --config config.yaml -b -1 -2 -v 1 > tapw_chern.log
```

## Orbital Analysis

```bash
cd output_dir/Q_shell_{n_g}
tapw orbital analyze . --config ../config.yaml --valley Gamma --band CBM
tapw orbital plot . --valley Gamma --band CBM --orbital-dir orbital_analysis --output-dir orbital_plots
```

Legacy aliases remain supported for existing scripts: `tapw-calc`, `tapw-config`, `tapw-plot`, `tapw-chernpost`, `tapw-orbital`, and `tapw-plot-orbital`.

## Examples

Release-facing TAPW examples live under `examples/tapw/`. Dataset provenance and unresolved release metadata are tracked in `examples/data-manifest.yaml` and `RELEASE_BLOCKERS.md`.
