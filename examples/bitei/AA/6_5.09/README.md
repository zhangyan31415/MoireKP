# BiTeI TeTe-AA 5.09

This example keeps the three workflow stages together:

```text
openmx/  ->  tapw/  ->  kp/
```

The maintained inputs are:

- `openmx/soc/POSCAR_rigid`, generated from the matching rigid OpenMX
  geometry;
- `tapw/configs/bitei_AA_6_5.09_Gamma_spinful_q05.yaml`;
- `kp/configs/bitei_AA_6_5.09_Gamma_spinful_q05.yaml`.

Run TAPW from the repository root when the OpenMX archive is present. These
are `external-data` commands:

<!-- external-data -->
```bash
tapw run -c examples/bitei/AA/6_5.09/tapw/configs/bitei_AA_6_5.09_Gamma_spinful_q05.yaml
# external-data
tapw symm -c examples/bitei/AA/6_5.09/tapw/configs/bitei_AA_6_5.09_Gamma_spinful_q05.yaml
```

The existing q05 TAPW source data can be projected and fitted directly. These
are also `external-data` commands:

<!-- external-data -->
```bash
kp project -c examples/bitei/AA/6_5.09/kp/configs/bitei_AA_6_5.09_Gamma_spinful_q05.yaml
# external-data
kp model -c examples/bitei/AA/6_5.09/kp/configs/bitei_AA_6_5.09_Gamma_spinful_q05.yaml
```

The maintained linear model uses three Hamiltonian rows (`0`, `20`, and `40`)
with zero low-energy weighting. Its reported and plotted validation window is
the top 14 bands.

The KP outputs are written below `kp/outputs/gamma_spinful/Gamma/q05/`.
The older `kp/configs/bitei_AA_6_5.09_Gamma_Q5.yaml` and its output trees are
retained for historical comparison; the spinful q05 file above is the
maintained release configuration.

Large OpenMX matrices, SCF files, and generated TAPW/KP arrays are external
example data. A local validation checkout may expose them through relative
links, but they are not part of the source distribution.
