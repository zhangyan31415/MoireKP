# kp package

Python package for projecting TAPW Hamiltonians and fitting moire k.p continuum models.

## Install

Install from the repository root:

```bash
pip install -e .
```

## Clean-Clone Smoke

These commands do not require external TAPW arrays:

```bash
kp --help  # clean-clone
python -m pytest examples/minimal_synthetic -q  # clean-clone
```

## External-Data Examples

Release examples are under the repository-level `examples/` directory. Start with:

```bash
kp inspect -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml
kp symm    -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml
```

These commands require external TAPW arrays listed in `examples/data-manifest.yaml`.
See `examples/README.md` for the full input/output workflow and the meaning of
`nlow_state_list`.

The release CLI exposes only `kp inspect`, `kp project`, `kp symm`, and
`kp model`. `kp model` writes the standalone evaluator directly into the
canonical `model/` output directory.

## Package Layout

- `kp/io`: load TAPW Hamiltonians and Q sets
- `kp/blocks`: assemble Q blocks and project low-energy effective Hamiltonians
- `kp/symmetry`: symmetry operators and symmetry projection helpers
- `kp/model`: continuum-model building blocks
- `kp/viz`: plotting utilities
