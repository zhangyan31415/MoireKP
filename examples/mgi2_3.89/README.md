# Bilayer MgI2 3.89 Degree Gamma And M Valleys

The release contains Gamma spinful q04, M1 spinful q07, and M1 spinless q07 KP
models. The M1 spinless model selects one spin sector of the spinful TAPW
calculation.

```bash
tapw run  -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data
tapw run  -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data
tapw symm -c examples/mgi2_3.89/tapw/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data
kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_spinful_q04.yaml  # external-data
kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinful_q07.yaml     # external-data
kp project -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml    # external-data
kp model   -c examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml    # external-data
```

All three KP models use automatic low-energy and harmonic selection, explicit
polynomial orders, and one final linear fit. Commented low-state lists are the
equivalent reviewed explicit selections.
