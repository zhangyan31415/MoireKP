# Bilayer ZrS2 3.15 Degree Gamma Valley

The structure is AA-centered: the two Zr sites coincide at the twist center,
while the full moiré cell contains local AA, AB, and BA regions.

The spinful Gamma q04 case uses automatic low-energy and harmonic selection,
explicit polynomial orders 8/2/3, and a target-weighted linear fit with
`two_sided_weight: 300.0`.

```bash
tapw run  -c examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml
tapw symm -c examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml
kp project -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml
kp symm   -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml
kp model   -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml
kp symm-rep -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml # optional
```

The commented 4+4 low-state list records the explicit equivalent of the
automatic result.
