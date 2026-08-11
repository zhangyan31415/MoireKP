# PtSe2 structure and external matrices

The repository includes the small rigid structure used by the PtSe2 TAPW
config:

```text
soc/POSCAR_rigid
```

The corresponding Hamiltonian and overlap archives are external scientific
data:

```text
soc/H_symm.npz
soc/S_symm.npz
```

They are not committed to the source repository.

The TAPW YAML defines the orbital content directly, so this example does not
require an OpenMX input file for structure or orbital metadata.
