# kp

Utilities for building moire k.p models from TAPW Hamiltonians.

## Install

From this directory:

```bash
pip install -e .
```

If the `kp` command is not installed, run the CLI as:

```bash
PYTHONPATH=. python -m kp.cli
```

## Examples

Runnable examples are under `examples/`. Start with:

```bash
kp plot -c examples/mgi2/mgi2_8_gamma.yaml
kp project -c examples/mgi2/mgi2_8_gamma.yaml

kp plot -c examples/mote2/mote2_8_K.yaml
kp project -c examples/mote2/mote2_8_K.yaml
```

After `kp project` generates `heff_list.npy` and `heff_eig.npy`, run the matching notebook:

```text
examples/mgi2/model_mgi2_G.ipynb
examples/mote2/model_mote2_K.ipynb
```

See `examples/README.md` for the full input/output workflow and the meaning of `nlow_state_list` and `norb_fix_list`.

## Local Scratch Configs

`configs/` is treated as a local scratch area for experiments and generated plots. It is ignored by git and should not be uploaded to GitHub. Share reproducible examples through `examples/` instead.

## Package Layout

- `kp/io`: load TAPW Hamiltonians and Q sets
- `kp/blocks`: assemble Q blocks and project low-energy effective Hamiltonians
- `kp/symmetry`: symmetry operators and symmetry projection helpers
- `kp/model`: continuum-model building blocks
- `kp/viz`: plotting utilities
