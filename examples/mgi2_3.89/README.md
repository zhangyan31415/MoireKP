## MgI2 3.89

This release example covers MgI2 at 3.89 degrees. The release-facing KP entry
points use one config per case:

```text
kp/configs/mgi2_3.89_Gamma_q05.yaml
kp/configs/mgi2_3.89_M1_q07.yaml
kp/configs/mgi2_3.89_M1_spinless_q07.yaml
```

Each config drives inspection, projection, symmetry projection, and model
fitting. Release examples do not use split source/model config directories.

### Run KP

From the repository root:

```bash
kp inspect -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_q05.yaml       # external-data
kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_q05.yaml       # external-data
kp symm    -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_q05.yaml       # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_q05.yaml       # external-data

kp inspect -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_q07.yaml          # external-data
kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_q07.yaml          # external-data
kp symm    -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_q07.yaml          # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_q07.yaml          # external-data

kp inspect -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml # external-data
kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml # external-data
kp symm    -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml # external-data
```

Run `inspect` first and check `kp/outputs/<profile>/<q_shell>/inspect/` before
changing `project.nlow_state_list`. Each case uses `project.gauge: auto`; users
do not write `norb_fix_list`.

### Current Metrics

These GPU validation numbers were generated with the active release configs:

| case | gauge | model dim | active terms | all-band RMS / max | plotted-band RMS / max |
| --- | --- | ---: | ---: | ---: | ---: |
| Gamma Q5 | auto + linear low-subspace refinement | 124 | 634 | 3.350 / 11.455 meV before refinement | top 10 plot: 0.836 / 3.180 meV |
| M1 spinless Q7 | auto | 8 | 266 | 1.074 / 2.969 meV | bottom 8: 1.114 / 3.263 meV |
| M1 spinful Q7 | auto | 16 | 401 | 1.886 / 4.631 meV | bottom 8: 0.695 / 2.784 meV |

### Data Notes

The heavy TAPW arrays and symmetry-analysis outputs are external data. In this
local checkout they are available under `tapw/` for validation, but they are
not part of a light source release.
