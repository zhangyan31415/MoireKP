# Bilayer MoTe2 3.89 Degree K Valley

One spinful TAPW K1 q06 calculation supports two automatic KP models:

- `kp/configs/mote2_3.89_K1_spinful_q06.yaml`;
- `kp/configs/mote2_3.89_K1_spinless_q06.yaml`.

The spinless case selects one spin sector from the spinful TAPW Hamiltonian;
it is not a separate non-SOC calculation.

```bash
tapw run  -c examples/mote2_3.89/tapw/configs/mote2_3.89_K1_spinful_q06.yaml
tapw symm -c examples/mote2_3.89/tapw/configs/mote2_3.89_K1_spinful_q06.yaml
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinful_q06.yaml
kp symm   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinful_q06.yaml
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinful_q06.yaml
kp symm-rep -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinful_q06.yaml # optional
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml
kp symm   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml
kp symm-rep -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml # optional
```

Both KP files automatically select the low-energy bands and harmonic counts.
Their commented `nlow_state_list` values record the equivalent fixed selection;
polynomial orders and linear fit inputs remain explicit.
