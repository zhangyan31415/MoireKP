# Bilayer PtSe2 7.34 Degree Gamma Valley

This release case is spinful Gamma q04 and uses automatic low-energy and
harmonic selection followed by one nonlinear model fit. Its polynomial orders
remain explicit at 10/8/8.

```bash
tapw run  -c examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
kp project -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
kp model   -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
```

The commented `[[54], [55]]` low-state list is the audited explicit equivalent
of the default automatic projection. Large OpenMX and generated TAPW/KP data
are supplied by the external release archive.
