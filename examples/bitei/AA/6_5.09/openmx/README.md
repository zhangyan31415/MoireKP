# OpenMX inputs

`soc/POSCAR_rigid` is a materialized VASP-format copy of the geometry in
`soc/openmx.dat_rigid`. It contains 762 atoms in the original OpenMX order:
254 Bi, 254 Te, and 254 I atoms.

The SOC Hamiltonian and overlap consumed by TAPW are:

```text
soc/H_symm.npz
soc/S_symm.npz
```

The remaining OpenMX files are large external inputs and outputs. A complete
local validation tree may use relative links to the archived calculation.
Those links are intentionally not required by clean-checkout tests or included
in source archives.
