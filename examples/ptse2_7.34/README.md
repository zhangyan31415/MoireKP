# Bilayer PtSe2 7.34 degree Gamma-valley example

This material-local example keeps the complete `openmx/`, `tapw/`, and `kp/`
workflow under one directory. It contains one spinful Gamma q04 TAPW
calculation and its nonlinear KP model:

```text
openmx/soc/POSCAR_rigid
tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml
kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml
```

The TAPW structure input is always `openmx/soc/POSCAR_rigid`; it preserves the
cell, file-order species sequence, and Cartesian coordinates of the audited
rigid OpenMX input. The local OpenMX layout also exposes the SOC/NSOC calculation
files and the symmetrized SOC Hamiltonian and overlap through relative links to
the retained `examples/toZY/` data.

The KP case keeps the current automatic low-energy selection, which resolves
one state per layer (`[[54], [55]]`), and the current nonlinear fit: 38 bands
at k-point indices 0, 20, and 40, fifth-shell intra/interlayer harmonics, and
kinetic/intralayer/interlayer polynomial orders 10/8/8.

## Run TAPW

```bash
tapw run  -c examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
```

## Run KP

```bash
kp project -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
kp model   -c examples/ptse2_7.34/kp/configs/ptse2_7.34_Gamma_spinful_q04.yaml  # external-data
```

The canonical local result directory is:

```text
kp/outputs/gamma_spinful/Gamma/q04/
```

Large OpenMX matrices and generated TAPW/KP arrays remain external data. The
relative links are for this complete local validation workspace; a clean clone
still needs the external archive declared by `examples/data-manifest.yaml`.
