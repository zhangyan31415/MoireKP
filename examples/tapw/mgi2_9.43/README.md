# MgI2 9.43 Degree TAPW Example

This example uses the rigid OpenMX structure `openmx/soc/openmx.dat_rigid`.
Large OpenMX matrix files (`H.npz`, `S.npz`) are external-data files and are
not intended for source distribution.

Run from this directory:

```bash
tapw run --config configs/mgi2_m_gamma_tapw.yaml  # external-data
tapw run --config configs/mgi2_direct.yaml  # external-data
tapw plot --config configs/mgi2_m_gamma_tapw_bands.yaml  # generated-output
tapw plot --config configs/mgi2_direct_bands.yaml  # generated-output
```

The TAPW config evaluates M1 and Gamma valleys. The direct config runs the
non-TAPW generalized diagonalization path with SLEPc. The direct path requires
an environment with `petsc4py` and `slepc4py`; it is intentionally separate
from the TAPW projection path. Both commands require pending external OpenMX
matrices listed in `examples/data-manifest.yaml`.
