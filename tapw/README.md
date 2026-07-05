# TAPW

TAPW provides truncated atomic plane-wave workflows for twisted-material band calculations, source-symmetry analysis, and topology post-processing.

## Features

- Band-structure calculations for twisted bilayer systems.
- Valley workflows for K, K prime, Gamma, and M points.
- C3-aware configuration and analysis utilities.
- MPI, PETSc, and SLEPc based solver support through the release conda environment.

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

Edit `output_dir/config.yaml`, especially the paths to `H.dat`, `S.dat`, and
`openmx.dat`.

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
  valley: K1
  q_shell: 4
  efermi: -4.6
  kpath:
    labels: [G, M, K, G]
    points_per_segment: 40
    coordinates:
      G: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]

symmetry:
  valley: K1
  q_shell: 4

topology:
  valley: K1
  q_shell: 4
  mesh:
    n_b1: 31
    n_b2: 31
    range_b1: [-0.5, 0.5]
    range_b2: [-0.5, 0.5]
```

The command selects the workflow. Release-style configs do not use per-section
`enable` switches: `tapw run` reads `bands`, `tapw symm` reads `symmetry`, and
`tapw topo` reads `topology`.

For the default spinful profile, TAPW writes `outputs/K1/q04` for a K1 valley
run. Non-default spin profiles should be explicit in the case/profile naming
used by downstream examples, for example `K1_up` or `K1_spinless`.

The canonical band workflow writes user-facing names:

```text
outputs/K1/q04/
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
  topology/
    grid31x31_b1_m0p5_0p5_b2_m0p5_0p5/
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
`C2T_shape`. TAPW symmetry writes only three release files:
`representations.npz`, `residuals.csv`, and `summary.md`. Topology grid
directories always include the sampled `b1` and `b2` ranges so partial grids do
not overwrite each other, for example `grid21x41_b1_0p0_0p5_b2_m0p5_0p5`.

## Symmetry Representation Post-Processing

After running source-symmetry analysis, compute band-subspace representation
matrices from the same config:

```bash
tapw symm -c config.yaml
tapw symm-rep -c config.yaml
```

Configure the reported points as fractional reciprocal coordinates under the
`symmetry` section:

```yaml
symmetry:
  valley: Gamma
  q_shell: 4
  efermi: -4.055365
  tolerance: 2.0e-2
  spglib_symprec: 5.0e-2
  representation:
    points:
      Gamma: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]
    valence_count: 20
    conduction_count: 20
    degeneracy_tol: 2.0e-3
```

`tapw symm-rep -c` requires the raw-H `representations.npz` from `tapw symm`.
If it is missing, the command fails and asks you to run `tapw symm -c` first.
The command solves the configured high-symmetry points directly, groups selected
states by energy degeneracy, projects unitary and antiunitary raw-H actions into
each band block, and writes `summary.md`, `bands.csv`, `characters.csv`,
`high_symmetry_wavefunctions.npz`, and `band_representations.npz` under
`outputs/<valley>/<q_shell>/symm_rep/`.

## Chern Post-Processing

```bash
cd output_dir
tapw topo -c config.yaml
```

## Examples

Release-facing TAPW examples live under `examples/tapw/`. Dataset provenance and unresolved release metadata are tracked in `examples/data-manifest.yaml` and `RELEASE_BLOCKERS.md`.
