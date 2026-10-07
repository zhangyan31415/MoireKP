# AB bilayer ZrS2

This directory provides the Gamma- and M-valley models used in the paper.

Supply the matching `H_symm.npz` and `S_symm.npz` in
`examples/zrs2_3.89/openmx/soc/`. The matching structure is included.
Large matrices and generated TAPW arrays are not included in the source repository.

Use the matching TAPW and KP configuration:

| Valley | TAPW config under `tapw/configs/` | KP config under `kp/configs/` |
|---|---|---|
| Gamma | `zrs2_3.89_Gamma_spinful_q04.yaml` | `zrs2_3.89_Gamma_spinful_q04.yaml` |
| M | `zrs2_3.89_M1_spinful_q07.yaml` | `zrs2_3.89_M1_spinful_linearized_q07.yaml` |

Run from the repository root after supplying the inputs:

```bash
TAPW_CFG=examples/zrs2_3.89/tapw/configs/zrs2_3.89_Gamma_spinful_q04.yaml
KP_CFG=examples/zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

An additional spinless M configuration is supplied under `kp/configs/`.
