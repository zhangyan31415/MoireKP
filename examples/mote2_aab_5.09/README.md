# 1+2 A-AB MoTe2 5.09 degree example

This release example promotes the current canonical 1+2 A-AB inputs formerly
kept in the local `mote2_aab_5.09_new` tree. It contains one Gamma and one K1
spinful TAPW calculation and three KP models:

```text
tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml
tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml
kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml
kp/configs/mote2_aab_5.09_K1_A_q04.yaml
kp/configs/mote2_aab_5.09_K1_B_q04.yaml
```

The K1 A and B files are distinct single-spin-sector continuum models selected
from the same spinful TAPW K1 calculation. They must not be combined into a
single frozen-model validation baseline: the historical A-sector and B-sector
reference artifacts have different provenance.

## Run TAPW

```bash
tapw run  -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml  # external-data
tapw run  -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml     # external-data
tapw symm -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml     # external-data
```

## Run KP

```bash
kp project -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml  # external-data
kp model   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml  # external-data

kp project -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml           # external-data
kp model   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml           # external-data

kp project -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_B_q04.yaml           # external-data
kp model   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_B_q04.yaml           # external-data
```

The Gamma projector uses the current multilayer row-order-aware low-state
selection. `kp project` also writes the Q-block eigenspectrum and exactified
symmetry package; standalone `kp inspect` and `kp symm` are optional
diagnostics. Large matrices and generated outputs remain external data, and
the complete local validation checkout exposes them through relative links.
