# Bilayer MoTe2

This directory provides the K-valley model used in the paper.

Supply the matching inputs in `examples/mote2_3.89/openmx/selected_20261001/`:

```text
H_symm.npz
S_symm.npz
rigid.extxyz
```

These inputs are not included in the source repository.

Run from the repository root after supplying the inputs:

```bash
TAPW_CFG=examples/mote2_3.89/tapw/configs/mote2_3.89_K1_paper_20261001_q06.yaml
KP_CFG=examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_tuned_q06.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

Other spinful and spinless configurations use the inputs in `openmx/soc/`.
Run settings are defined in the corresponding YAML files.
