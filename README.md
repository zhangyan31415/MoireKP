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

The [paper examples](examples/README.md) include NSOC/SOC OpenMX inputs, relaxed and rigid POSCAR files and commands to regenerate H/S and build the six models. Large matrices are generated locally. From the repository root:

```bash
TAPW_CFG=examples/zrs2_3.89/tapw/tapw_Gamma.yaml
KP_CFG=examples/zrs2_3.89/kp/kp_Gamma.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

`kp project` selects the low-energy space under the configured policy and applies energy-linearized Löwdin reduction by default. `kp symm` supplies the symmetry matrices used by `kp model` to determine the continuum-model coefficients.

TAPW writes `band/`, `symmetry/`, `symm_rep/` and `topology/`. KP writes `inspect/`, `projection/`, `symmetry/`, `symm_rep/` and `model/`. Run the generated `model/evaluate.py` to evaluate an exported model without the MoireKP package.

Optional calculations use `tapw topo`, `tapw symm-rep`, `kp inspect` and `kp symm-rep`; each command has `--help`. Generate a starter configuration with `tapw init -o workdir`.

## Repository layout

| Path | Contents |
|---|---|
| `tapw/tapw/`, `kp/kp/` | Installable package code |
| `scripts/openmx/`, `abacus/`, `siesta/`, `common/` | Input preparation tools |
| `scripts/release/` | Software distribution builder |
| `examples/` | Three paper examples with inputs and run instructions |

The [example index](examples/README.md) identifies the six models used in the paper. The software distribution contains the source, first-principles inputs and model configurations.

## Build

```bash
python scripts/release/build_distribution.py --output dist
```

## Citation

If you use MoireKP in your research, please consider citing the software paper and the underlying model-building method:

- Yan Zhang et al., [MoireKP: An Automated Workflow for Symmetry-Constrained Moiré Continuum Models](https://arxiv.org/abs/2610.37760), arXiv:2610.37760 (2026).
- Yan Zhang et al., [Universal Moiré-Model-Building Method without Fitting: Application to Twisted MoTe₂ and WSe₂](https://arxiv.org/abs/2411.08108), arXiv:2411.08108 (2024).

```bibtex
@misc{zhang_MoireKP_2026,
  title = {{MoireKP}: An Automated Workflow for Symmetry-Constrained Moir{\'e} Continuum Models},
  author = {Zhang, Yan and Pi, Hanqi and Miao, Wangqian and Fang, Zhong and Weng, Hongming and Dai, Xi and Bernevig, B. Andrei and Wu, Quansheng},
  year = {2026},
  month = oct,
  eprint = {2610.37760},
  archivePrefix = {arXiv},
  primaryClass = {cond-mat.mtrl-sci},
  doi = {10.48550/arXiv.2610.37760},
  url = {https://arxiv.org/abs/2610.37760}
}

@misc{zhang_Universal_2024,
  title = {Universal Moir{\'e}-Model-Building Method without Fitting: Application to Twisted {MoTe$_2$} and {WSe$_2$}},
  author = {Zhang, Yan and Pi, Hanqi and Liu, Jiaxuan and Miao, Wangqian and Qi, Ziyue and Regnault, Nicolas and Weng, Hongming and Dai, Xi and Bernevig, B. Andrei and Wu, Quansheng and Yu, Jiabin},
  year = {2024},
  month = nov,
  eprint = {2411.08108},
  archivePrefix = {arXiv},
  primaryClass = {cond-mat.mes-hall},
  doi = {10.48550/arXiv.2411.08108},
  url = {https://arxiv.org/abs/2411.08108}
}
```

## License

Software, repository-authored documentation and small configurations use `LGPL-3.0-or-later`; see [COPYRIGHT](COPYRIGHT), [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING).
