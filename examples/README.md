# Examples

Configs and small structural inputs are supplied for six material families. Numerical runs require the external H/S matrices and TAPW arrays listed in [data-manifest.yaml](data-manifest.yaml).

## Paper models

| Material | Valley | KP configuration |
|---|---|---|
| AA MoTe2, 3.89° | K | `mote2_3.89/kp/configs/mote2_3.89_K1_spinless_tuned_q06.yaml` |
| AB ZrS2, 3.89° | M1 | `zrs2_3.89/kp/configs/zrs2_3.89_M1_spinful_linearized_q07.yaml` |
| AB ZrS2, 3.89° | Gamma | `zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml` |
| A–AB MoTe2, 5.09° | KA | `mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml` |
| A–AB MoTe2, 5.09° | KB | `mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_B_q04.yaml` |
| A–AB MoTe2, 5.09° | Gamma | `mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml` |

The six models use linear coefficient fits and energy-linearized Löwdin reduction. ZrS2 Gamma uses a closed Gamma–M–K–Gamma path, rows `[5,20,40,55]`, orders 8/4/4 and weights 2.25/22.5 on the highest 20 states. The other five fits are unweighted. Material READMEs specify the matching TAPW configuration.

## Other examples

[MoTe2](mote2_3.89/README.md), [A–AB MoTe2](mote2_aab_5.09/README.md), [ZrS2 3.89°](zrs2_3.89/README.md), [ZrS2 3.15°](zrs2_3.15/README.md), [MgI2](mgi2_3.89/README.md) and [PtSe2](ptse2_7.34/README.md) provide additional spin/valley variants. PtSe2 is a supplied configuration whose full workflow has not been validated. Historical outputs and searches are excluded from the source distribution.

## Run

From the repository root, after supplying the data (external-data):

```bash
# external-data
TAPW_CFG=examples/zrs2_3.89/tapw/configs/zrs2_3.89_Gamma_spinful_q04.yaml
KP_CFG=examples/zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

`kp symm` is required before `kp model`. Configured `project.selection`, reference vectors, harmonic support, fit rows and weights define the selected case; an explicit choice overrides its automatic search.

Interface checks require no example matrices (clean-clone):

```bash
# clean-clone
tapw --help
kp --help
```
