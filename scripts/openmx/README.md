# OpenMX H/S preparation

## Install and run OpenMX

The tested DFT output uses OpenMX 3.9 and its matching pseudoatomic data.
Follow the [OpenMX 3.9 installation manual](https://www.openmx-square.org/openmx_man3.9/)
to build the MPI executable and `read_scfout.o`. Enable `HS.fileout on` in the
OpenMX input, then run that calculation with its own executable:

```bash
mpirun -np N /absolute/openmx3.9/source/openmx \
  /absolute/case/openmx.dat -nt 1
```

This is separate from the MoireKP package installation. The reader below is
compiled against that **same** OpenMX source and oneAPI/MKL runtime.

## Prepare H/S

`prepare_hs.py` imports a completed `.scfout` using its matching OpenMX input
file. The native reader checks embedded geometry, site counts and spin mode,
converts Hartree to eV, and preserves all raw H/S entries.

The exported versioned H.npz/S.npz mark their entries as `basis_order: source`.
TAPW then applies the same atom-type permutation it applies when first reading
H.dat/S.dat. DAT-generated NPZ caches are already permuted. Reimport OpenMX
data made before this marker was added; its unmarked H.npz/S.npz may give
incorrect TAPW bands even though direct source eigenvalues match.

```bash
bash scripts/openmx/build_openmx_symm_hs.sh /path/to/openmx/source build/openmx_import_hs
tapw prepare-hs openmx --input openmx.scfout --structure openmx.dat \
  --binary build/openmx_import_hs/analysis_symm_hs --output prepared --format both
```

Load the same oneAPI/MKL runtime used at build time. Add `--symmetrize` for
separate symmetry outputs. The historical top-level
`scripts/openmx_symm_hs_scfout.py` and `scripts/openmx_symm_hs_python.py`
retain their original symmetrized-output behavior; implementations are in
`scripts/legacy/openmx/`.
The per-backend `prepare_hs.py` accepts the same options as the installed CLI.
See [common options and TAPW continuation](../README.md).

## Paper examples

The [bilayer MoTe2](../../examples/mote2_3.89/README.md),
[bilayer ZrS2](../../examples/zrs2_3.89/README.md) and
[trilayer MoTe2](../../examples/mote2_aab_5.09/README.md) directories provide
paired NSOC/SOC inputs and `POSCAR_relaxed` / `POSCAR_rigid`. Run NSOC to
convergence on the relaxed structure, then one SOC iteration reading its
restart files (`scf.restart c2n`). Both inputs run in the same directory: NSOC
writes `nsoc_rst/`; SOC reads that directory and writes `soc.scfout`. The same
relaxed coordinates are embedded in both OpenMX inputs; TAPW uses the rigid
POSCAR reference.

After installing OpenMX with its DFT_DATA19 database, build the reader from
the repository root and set the executable/database paths:

```bash
bash scripts/openmx/build_openmx_symm_hs.sh /absolute/openmx3.9/source build/openmx_import_hs
export OPENMX=/absolute/openmx3.9/source/openmx
export OPENMX_DATA=/absolute/openmx3.9/DFT_DATA19
export OPENMX_READER="$PWD/build/openmx_import_hs/analysis_symm_hs"
export NPROCS=61
```

Use the MPI rank count assigned to your calculation. Each material README
connects NSOC, one-step SOC, relaxed-coordinate H/S symmetry averaging,
TAPW and KP.
The database must contain the PAO and PBE19 VPS files named by the supplied
input. See the [OpenMX input specification](https://www.openmx-square.org/openmx_man3.9/node21.html)
and [restart documentation](https://www.openmx-square.org/openmx_man3.9/node44.html).
