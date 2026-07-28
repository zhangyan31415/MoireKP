# Bilayer ZrS2 3.15 degree Gamma-valley example

This release example contains one spinful Gamma q04 TAPW calculation and one
4+4-orbital KP model:

```text
tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml
kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml
```

This ZrS2 case enables a target-weighted linear model fit with projector weight
300. The objective uses the current-Heff support mask, a degeneracy-safe top
window, and a two-sided target projector. PtSe2 uses a separately validated
weight. Experimental development fit variants and generated-output files from
the source worktree are not part of this release example.

## Run TAPW

```bash
tapw run  -c examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
```

## Run KP

```bash
kp project -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
kp model   -c examples/zrs2_3.15/kp/configs/zrs2_3.15_Gamma_spinful_q04.yaml  # external-data
```

`kp project` writes the Q-block eigenspectrum and exactified symmetry package;
standalone `kp inspect` and `kp symm` are optional diagnostics. Large OpenMX
matrices and generated TAPW/KP arrays are external data declared in
`examples/data-manifest.yaml`; the complete local validation checkout exposes
them through relative links.
