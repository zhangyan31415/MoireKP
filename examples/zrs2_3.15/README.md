# Bilayer ZrS2 3.15 Degree Gamma Valley

This spinful Gamma q04 release case uses automatic low-energy and harmonic
selection, explicit polynomial orders 8/2/3, and a target-weighted linear fit.

```bash
tapw run  -c examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
kp project -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
kp model   -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
```

The commented 4+4 low-state list is the reviewed explicit equivalent of the
automatic result. External data provenance is recorded in the shared manifest.
