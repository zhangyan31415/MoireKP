# Bilayer PtSe2 7.34 Degree Gamma Valley

This spinful Gamma q04 config uses automatic low-energy and harmonic selection
followed by one nonlinear model fit. Its polynomial orders are 10/8/8.

```bash
tapw run  -c examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml
tapw symm -c examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml
kp project -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml
kp symm   -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml
kp model   -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml
kp symm-rep -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml # optional
```

The config uses equal one-sided, two-sided, and band-loss weights. The
commented `[[54], [55]]` list records its intended explicit low-state
selection. The full numerical workflow has not yet been tested; large OpenMX
and generated TAPW/KP data are external.
