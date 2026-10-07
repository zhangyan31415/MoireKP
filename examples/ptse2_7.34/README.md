# Bilayer PtSe2

This directory provides an additional Gamma-valley configuration. Its full
numerical workflow has not been validated.

Supply the matching `H_symm.npz` and `S_symm.npz` in
`examples/ptse2_7.34/openmx/soc/`. The matching structure is included.
Large matrices and generated TAPW arrays are not included in the source repository.

Run from the repository root after supplying the inputs:

```bash
TAPW_CFG=examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml
KP_CFG=examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```
