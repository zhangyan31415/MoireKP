# MgI2 9.43 Degree TAPW Example

This example uses the rigid OpenMX structure `openmx/soc/openmx.dat_rigid`.
Large OpenMX matrix files (`H.npz`, `S.npz`) are external-data files and are
not intended for source distribution.

Run from this directory:

```bash
tapw run -c configs/M1_q03.yaml  # external-data
tapw run -c configs/Gamma_q03.yaml  # external-data
tapw plot -c configs/M1_q03.plot.yaml  # generated-output
tapw plot -c configs/Gamma_q03.plot.yaml  # generated-output
```

The two TAPW configs evaluate M1 and Gamma separately with q-shell 3 and write
canonical outputs under `outputs/M1/q03/` and `outputs/Gamma/q03/`. The commands
require external OpenMX matrices listed in `examples/data-manifest.yaml`.
