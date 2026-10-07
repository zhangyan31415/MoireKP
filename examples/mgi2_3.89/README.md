# Bilayer MgI2

This directory provides additional Gamma- and M-valley examples.

Supply the matching `H_symm.npz` and `S_symm.npz` in
`examples/mgi2_3.89/openmx/soc/`. The matching structure is included.
Large matrices and generated TAPW arrays are not included in the source repository.

Run from the repository root after supplying the inputs:

```bash
TAPW_CFG=examples/mgi2_3.89/tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml
KP_CFG=examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_spinful_q04.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

For the M valley, use the M1 configuration under `tapw/configs/` and the
matching spinful or spinless configuration under `kp/configs/`.
