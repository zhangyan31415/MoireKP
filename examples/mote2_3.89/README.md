# Bilayer MoTe2 3.89 degree K-valley example

This release example contains one spinful TAPW K1 calculation and two KP
models built from it:

```text
tapw/configs/mote2_3.89_K1_spinful_q06.yaml
kp/configs/mote2_3.89_K1_spinful_q06.yaml
kp/configs/mote2_3.89_K1_spinless_q06.yaml
```

The `spinless` KP case selects the spin-up sector of the spinful TAPW
Hamiltonian. It is not a separate non-SOC TAPW calculation.

## Run TAPW

From the repository root:

```bash
tapw run  -c examples/mote2_3.89/tapw/configs/mote2_3.89_K1_spinful_q06.yaml  # external-data
tapw symm -c examples/mote2_3.89/tapw/configs/mote2_3.89_K1_spinful_q06.yaml  # external-data
```

## Run KP

```bash
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinful_q06.yaml   # external-data
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinful_q06.yaml   # external-data

kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml  # external-data
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml  # external-data
```

`kp project` writes the Q-block eigenspectrum, selected layer/spin/orbital
content, projected bands, and the exactified symmetry package. Standalone
`kp inspect` and `kp symm` remain optional diagnostics. Both KP cases keep the
reviewed low-energy basis and explicit model settings. Large OpenMX matrices
and generated TAPW/KP arrays are external data declared in
`examples/data-manifest.yaml`; the complete local validation checkout exposes
them through relative links.
