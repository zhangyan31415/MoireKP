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

New configs use `case.output_root` and per-workflow sections. Canonical output
is the default for this release-facing format:

```yaml
case:
  name: K1_q04
  output_root: outputs

bands:
  enable: true
  valley: K1
  q_shell: 4
  efermi: -4.6

symmetry:
  enable: true
  valley: K1
  q_shell: 4

topology:
  enable: false
  valley: K1
  q_shell: 4
  mesh:
    n_b1: 31
    n_b2: 31
    range_b1: [-0.5, 0.5]
    range_b2: [-0.5, 0.5]
```

For the default spinful profile, TAPW writes `outputs/K1/q04` for a K1 valley
run. Non-default spin profiles should be explicit in the case/profile naming
used by downstream examples, for example `K1_up` or `K1_spinless`.

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
  symmetry/
    manifest.json
    summary.json
    summary.md
    details.csv
    representations.npz
    representations/
      manifest.json
      raw_h/
        C3z.npz
        C2.npz
        C2T.npz
      diagnostics/
        C3z_source.npz
        C3z_pin.npz
        C3z_pg.npz
  topology/
    grid31x31_b1_m0p5_0p5_b2_m0p5_0p5/
      manifest.json
      chern_summary.json
      berry_curvature_vbm2.txt
      berry_curvature_vbm2.pdf
      quantum_geometry_vbm2.txt
      quantum_geometry_vbm2.pdf
      quantum_geometry_vbm2_trace_condition.txt
      wcc_vbm2_loop_b2.txt
      wcc_vbm2_loop_b2.pdf
```

`symmetry/representations.npz` packs the TAPW raw-H sparse symmetry matrices as
CSR components, with keys such as `C2T_data`, `C2T_indices`, `C2T_indptr`, and
`C2T_shape`. The `representations/raw_h/*.npz` files remain as compatibility
aliases for older scripts. Topology grid directories always include the sampled
`b1` and `b2` ranges so partial grids do not overwrite each other, for example
`grid21x41_b1_0p0_0p5_b2_m0p5_0p5`.

Compatibility outputs that should be migrated away from in new workflows:

- `Q_shell_<n_g>/`: legacy root output used by old `compute` configs. New
  configs should use `outputs/<valley>/qNN/<workflow>/`.
- `symmetry/representations/raw_h/*.npz`: legacy per-operation raw-H matrix
  files. New readers should use `symmetry/representations.npz` plus
  `symmetry/representations/manifest.json`.
- `band_data/`: legacy band output directory name. New configs write `band/`.
- Old topology filenames such as `bc_bands_*`, `qgt_bands_*`, and
  `wcc_*_VBM_*`: new configs write bandset-labeled files such as
  `berry_curvature_vbm2.*`, `quantum_geometry_vbm2.*`, and
  `wcc_vbm2_loop_b2.*` under a range-aware `grid.../` directory.
- Long compatibility commands such as `tapw run --mode symmetry` and
  `tapw-chernpost`: release-facing examples should use `tapw symm` and
  `tapw chern`.

Plot the generated bands:

```bash
cd output_dir/outputs/K1/q04/band
tapw plot -c ../../../../bands.yaml
```

Legacy configs that only use `compute` keep the existing `Q_shell_<n_g>` names
and may use `band_data/` instead of `band/`.

## Chern Post-Processing

```bash
cd output_dir
tapw chern -c config.yaml
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
