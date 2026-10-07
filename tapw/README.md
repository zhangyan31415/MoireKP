# TAPW

TAPW provides truncated atomic plane-wave workflows for twisted layered
materials, including band calculations, source-symmetry analysis,
band-representation post-processing, and topology.

## Features

- Hexagonal and square in-plane lattice workflows.
- K, K-prime, Gamma, and M valley calculations where supported.
- Structure input through POSCAR, CIF, and other ASE-readable formats.
- Orbital definitions in YAML rather than an OpenMX structure input file.
- SIESTA HSX v1/v2 import through optional sisl, including orbital phases and spin ordering.
- OpenMX/ABACUS/SIESTA preparation with common raw H/S and optional separate symmetry outputs.
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

Use `tapw prepare-hs BACKEND --help` for the common DFT input interface, with
`BACKEND` set to `openmx`, `abacus` or `siesta`. See
[preparation tools](../scripts/README.md) and the
[example configurations](../examples/README.md).

For SIESTA input, install `moirekp[siesta]` and run
`tapw import-siesta siesta.HSX --output siesta_import`.
The generated YAML is an input fragment; supply the actual moire geometry and
workflow settings before running TAPW. See [SIESTA input](../scripts/siesta/README.md).

Create a starter working directory:

```bash
tapw init -o workdir
```

The generated `workdir/config.yaml` contains the input and workflow settings.


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

Set the reported points in the configuration under `symmetry.representation`.


## Examples

Tracked TAPW case configs live under
`examples/<material>/tapw/configs/`. Their large matrices and generated arrays
are external and are not committed to the source repository.
