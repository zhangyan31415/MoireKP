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
kp fit -c examples/mote2_3.89/kp/configs/model/mote2_3.89_K1.yaml  # precomputed
kp fit -c examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1.yaml  # precomputed
```

These model commands require precomputed `kp proj` and `kp symm` outputs
listed in `examples/data-manifest.yaml`. See `examples/README.md` for the full
input/output workflow and the meaning of `nlow_state_list` and `norb_fix_list`.

Longer compatibility forms remain supported: `kp plot` is equivalent to
`kp show`, `kp project` is equivalent to `kp proj`, and `kp model --config ...`
is equivalent to `kp fit -c ...`.

## Package Layout

- `kp/io`: load TAPW Hamiltonians and Q sets
- `kp/blocks`: assemble Q blocks and project low-energy effective Hamiltonians
- `kp/symmetry`: symmetry operators and symmetry projection helpers
- `kp/model`: continuum-model building blocks
- `kp/viz`: plotting utilities
