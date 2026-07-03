# MoTe2 9.43 Degree TAPW Example

This example uses the rigid OpenMX structure `openmx/soc/openmx.dat_rigid`.
Large OpenMX matrix files (`H.npz`, `S.npz`) are external-data files and are
not intended for source distribution.

Run from this directory:

```bash
tapw run -c configs/K1_q03.yaml  # external-data
tapw plot -c configs/K1_q03.plot.yaml  # generated-output
```

The TAPW config evaluates the K1 valley with q-shell 3 and writes canonical
outputs under `outputs/K1/q03/`. The command requires external OpenMX matrices
listed in `examples/data-manifest.yaml`.
