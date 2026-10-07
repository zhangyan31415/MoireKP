# A–AB trilayer MoTe2

This directory provides the KA, KB and Gamma models used in the paper.

Supply the matching `H_symm.npz` and `S_symm.npz` in
`examples/mote2_aab_5.09/openmx/soc/`. The matching structure is included.
Large matrices and generated TAPW arrays are not included in the source repository.

Use the matching TAPW and KP configuration:

| Model | TAPW config under `tapw/configs/` | KP config under `kp/configs/` |
|---|---|---|
| KA | `mote2_aab_5.09_K1_spinful_q04.yaml` | `mote2_aab_5.09_K1_A_q04.yaml` |
| KB | `mote2_aab_5.09_K1_spinful_q04.yaml` | `mote2_aab_5.09_K1_B_q04.yaml` |
| Gamma | `mote2_aab_5.09_Gamma_spinful_q04.yaml` | `mote2_aab_5.09_Gamma_spinful_q04.yaml` |

Run from the repository root after supplying the inputs:

```bash
TAPW_CFG=examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml
KP_CFG=examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```
