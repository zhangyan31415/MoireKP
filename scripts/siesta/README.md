# SIESTA H/S preparation

## Install and run SIESTA

SIESTA is a separate DFT executable. The verified examples used SIESTA 5.4.2.
The [official conda instructions](https://docs.siesta-project.org/projects/siesta/en/stable/installation/conda.html)
cover serial and MPI builds; choose an MPI variant if one calculation should
use multiple ranks.

```bash
mamba create -p /absolute/siesta-env -c conda-forge 'siesta=*=*openmpi*'
```

This is an external DFT environment. MoireKP with `[siesta]` is installed
separately; `sisl` alone can read a finished HSX without a SIESTA executable.

In the `.fdf` input, enable:

```text
SaveHS true
ForceAuxCell true
```

`ForceAuxCell` retains the translated H(R)/S(R) images needed away from Γ.
Run the DFT executable separately, for example:

```bash
siesta < /absolute/case/RUN.fdf > /absolute/case/siesta.out
```

## Prepare H/S

`prepare_hs.py` reads a completed self-describing HSX v1/v2 and emits the
canonical H/S, structure and basis used by TAPW. Install `moirekp[siesta]`.

```bash
tapw prepare-hs siesta --input /absolute/case/system.HSX \
  --output /absolute/new/prepared --format both
```

Add `--symmetrize` for separate symmetry outputs. Details, common options,
and backend comparisons are in [the script guide](../README.md).
`prepare_hs.py` accepts the same options as the installed CLI.
The earlier `tapw import-siesta` command remains available.
