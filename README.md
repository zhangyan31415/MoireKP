# MoireKP

MoireKP constructs symmetry-constrained moiré continuum models from localized-orbital Hamiltonians. `tapw` prepares H/S inputs and computes TAPW bands, symmetry representations and topology. `kp` selects a low-energy basis, constructs the effective Hamiltonian and exports a standalone continuum model.

[中文](README.zh.md) · [Examples](examples/README.md) · [TAPW reference](tapw/README.md) · [KP reference](kp/README.md)

## Install

Use a Python 3.11 or newer environment:

```bash
python -m pip install .
tapw --help
kp --help
```

For development, use `python -m pip install -e .`. SIESTA input requires `python -m pip install '.[siesta]'`. SLEPc is optional; `environment.yml` provides the complex PETSc/SLEPc environment for large sparse calculations.

## Prepare inputs

The common import command reads completed DFT outputs:

```bash
tapw prepare-hs --help
tapw prepare-hs openmx --help
tapw prepare-hs abacus --help
tapw prepare-hs siesta --help
```

Preparation writes H(R) in eV, dimensionless S(R), structure and orbital/spin metadata. See [the input guide](scripts/README.md) for the reader build and backend-specific commands.

## Run a model

The [paper examples](examples/README.md) include relaxed OpenMX SCF inputs, rigid TAPW references and commands to regenerate H/S and build the six models. Large matrices are generated locally. From the repository root:

```bash
TAPW_CFG=examples/zrs2_3.89/tapw/configs/zrs2_3.89_Gamma_spinful_q04.yaml
KP_CFG=examples/zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

`kp project` selects the low-energy space under the configured policy and applies energy-linearized Löwdin reduction by default. `kp symm` supplies the symmetry package consumed by `kp model`. The model fit and optional nonlinear refinement are separate from the reduction.

TAPW writes `band/`, `symmetry/`, `symm_rep/` and `topology/`. KP writes `inspect/`, `projection/`, `symmetry/`, `symm_rep/` and `model/`. Run the generated `model/evaluate.py` to evaluate an exported model without the MoireKP package.

Optional calculations use `tapw topo`, `tapw symm-rep`, `kp inspect` and `kp symm-rep`; each command has `--help`. Generate a starter configuration with `tapw init -o workdir`.

## Repository layout

| Path | Contents |
|---|---|
| `tapw/tapw/`, `kp/kp/` | Installable package code |
| `scripts/openmx/`, `abacus/`, `siesta/`, `common/` | Input preparation tools |
| `scripts/release/` | Software distribution builder |
| `examples/` | Six material families, current configs and input requirements |

The [example index](examples/README.md) identifies the six models used in the paper. The software distribution contains the source, first-principles inputs and model configurations.

## Build

```bash
python scripts/release/build_distribution.py --output dist
```

The builder checks the wheel against the source and excludes local research files from the source archive.

## License

Software, repository-authored documentation and small configurations use `LGPL-3.0-or-later`; see [COPYRIGHT](COPYRIGHT), [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING). This license does not automatically cover external datasets or DFT inputs. The separately licensed ABACUS CSR utility includes its own notice.
