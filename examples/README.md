# Examples

The source repository provides configurations and small structural inputs.
The three paper-system directories include relaxed OpenMX SCF inputs and
rigid TAPW references. Their READMEs give the commands to regenerate H/S and
construct the six continuum models. Large matrices and generated arrays are
produced locally.

## Paper examples

Inputs and workflows: commit [4d12c65](https://github.com/zhangyan31415/MoireKP/commit/4d12c6507d726a010c25568142c5fcbded34d070).

| Material | Valleys | Instructions |
|---|---|---|
| AA bilayer MoTe2 | K | [MoTe2](mote2_3.89/README.md) |
| AB bilayer ZrS2 | Gamma, M | [ZrS2](zrs2_3.89/README.md) |
| A–AB trilayer MoTe2 | KA, KB, Gamma | [Trilayer MoTe2](mote2_aab_5.09/README.md) |

## Additional examples

These extra configurations require their matching external inputs.

- [MgI2](mgi2_3.89/README.md)
- [ZrS2](zrs2_3.15/README.md)
- [PtSe2](ptse2_7.34/README.md), whose full numerical workflow has not been validated

## Workflow

Run the commands from the repository root, using the matching files listed
in the material README:

```bash
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

The YAML files contain the run settings. The output location is defined there.
