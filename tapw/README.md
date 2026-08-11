# TAPW

TAPW provides truncated atomic plane-wave workflows for twisted layered
materials, including band calculations, source-symmetry analysis,
band-representation post-processing, and topology.

## Features

- Hexagonal and square in-plane lattice workflows.
- K, K-prime, Gamma, and M valley calculations where supported.
- Structure input through POSCAR, CIF, and other ASE-readable formats.
- Orbital definitions in YAML rather than an OpenMX structure input file.
- MPI, PETSc, and SLEPc solver support through the provided environment.

## Installation

Install from the repository root:

```bash
conda env create -f environment.yml
conda activate moirekp
```

The environment includes `mpi4py`, `petsc4py`, `slepc4py`, and an editable
installation of MoireKP.

## Quick check

```bash
python -c "import tapw"
tapw --help
tapw init --help
```

## Basic usage

Create a starter working directory:

```bash
tapw init -o workdir
```

The generated `workdir/config.yaml` uses the current `system` input:

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
    n_b1: 31
    n_b2: 31
    range_b1: [-0.5, 0.5]
    range_b2: [-0.5, 0.5]
```

`system.structure` and `system.orbitals` replace the old use of an OpenMX
input file for structure and orbital metadata. The Hamiltonian and optional
overlap matrices remain explicit inputs. Omit `system.overlap` only for an
orthogonal basis. The in-plane Bravais family is inferred from the structure;
unsupported metrics are rejected.

The command selects the workflow. Configs do not use per-section `enable`
switches:

```bash
tapw run      -c workdir/config.yaml
tapw symm     -c workdir/config.yaml
tapw symm-rep -c workdir/config.yaml  # optional
tapw topo     -c workdir/config.yaml  # optional
```

## Outputs

```text
outputs/<valley>/<q-shell>/
  band/
    energies_vbm.txt
    hamiltonian_k.npy
    g_vectors_group1.npy
    g_vectors_group2.npy
    kpoints.npy
  symmetry/
    representations.npz
    summary.md
  symm_rep/
    summary.md
    bands.csv
    characters.csv
    high_symmetry_wavefunctions.npz
    band_representations.npz
  topology/<grid-id>/
    chern_summary.json
    berry_curvature_<bands>.txt
    quantum_geometry_<bands>.txt
    wcc_<bands>_<loop>.txt
```

`symmetry/representations.npz` is the machine-readable TAPW raw-H symmetry
package. `summary.md` is the human-readable report.

Topology grid directory names include the sampled `b1` and `b2` ranges so
partial grids do not overwrite one another, for example
`grid21x41_b1_0p0_0p5_b2_m0p5_0p5`.

## Band-representation post-processing

`tapw symm-rep` requires `tapw symm` first. It solves the configured
high-symmetry points, groups the selected states by energy degeneracy, and
projects unitary and antiunitary raw-H actions into each band block.

Configure the reported points under `symmetry.representation`:

```yaml
symmetry:
  valley: Gamma
  q_shell: 4
  efermi: -4.055365
  representation:
    points:
      Gamma: [0.0, 0.0]
      M: [0.5, 0.0]
      K: [0.3333333333, 0.3333333333]
    valence_count: 20
    conduction_count: 20
    degeneracy_tol: 2.0e-3
```

## Examples

Tracked TAPW case configs live under
`examples/<material>/tapw/configs/`. Their large matrices and generated arrays
are external and are not committed to the source repository.
