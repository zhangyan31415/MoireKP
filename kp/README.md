# kp package

Python package for projecting TAPW Hamiltonians and fitting moire k.p continuum models.

## Install

Install from the repository root:

```bash
pip install -e .
```

## Examples

Release examples are under the repository-level `examples/` directory. Start with:

```bash
kp model --config examples/mote2_3.89/kp/configs/model/mote2_3.89_K1.yaml
kp model --config examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1.yaml
```

See `examples/README.md` for the full input/output workflow and the meaning of `nlow_state_list` and `norb_fix_list`.

## Package Layout

- `kp/io`: load TAPW Hamiltonians and Q sets
- `kp/blocks`: assemble Q blocks and project low-energy effective Hamiltonians
- `kp/symmetry`: symmetry operators and symmetry projection helpers
- `kp/model`: continuum-model building blocks
- `kp/viz`: plotting utilities
