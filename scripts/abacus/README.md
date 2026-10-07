# ABACUS H/S preparation

## Install and run ABACUS

Install the DFT executable separately, following the
[ABACUS installation guide](https://abacus.deepmodeling.com/en/latest/quick_start/easy_install.html).
The verified examples used an ABACUS 3.11.0-beta6 build with real-space H/S
export. Confirm that your build supports the tested `out_mat_hs2` form before
starting a large calculation.

The official conda-forge route for a separate installation is:

```bash
mamba create -p /absolute/abacus-env -c conda-forge abacus
```

This installs the DFT program, not MoireKP. Version-specific H/S export
should be checked on a small case before reproducing the examples.

In `INPUT`, include:

```text
basis_type lcao
gamma_only 0
out_mat_hs2 1 12
```

Keep the original `INPUT`, `STRU`, numerical orbital files, and
`OUT.<suffix>/hrs1_nao.csr` and `srs1_nao.csr` together. Run the calculation
from its case directory:

```bash
cd /absolute/abacus-case
OMP_NUM_THREADS=1 mpirun -np N /absolute/abacus
```

## Prepare H/S

The same command pattern used for OpenMX and SIESTA imports the completed case:

```bash
tapw prepare-hs abacus --input /absolute/abacus-case \
  --output /absolute/new/prepared --format both
```

The importer reads the original orbital file metadata, converts H once from
Ry to eV, keeps S dimensionless, and records the source Fermi energy and
basis ordering. Output includes H/S in NPZ and DAT, `structure.extxyz`,
`system.fragment.yaml`, `abacus_import.json` and `preparation.json`.
The destination must not exist. `prepare_hs.py` is an equivalent wrapper;
`--symmetrize` writes separate symmetry results without changing raw H/S.
See the [common guide](../README.md) and
[basis/spin conventions](../../docs/abacus_input.md).

The `_vendor` directory contains the separately licensed GPL-3.0 CSR
converter, its license, and a notice documenting the local sparse-support
correction. It is included in the installed wheel.
