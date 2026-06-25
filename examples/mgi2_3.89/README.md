## MgI2 3.89

This release example covers MgI2 at 3.89 degrees. The active KP entry points
are one YAML file per case:

```text
kp/configs/mgi2_3.89_Gamma_Q5.yaml
kp/configs/mgi2_3.89_M1_spinless_Q7.yaml
kp/configs/mgi2_3.89_M1_spinful_Q7.yaml
```

The YAML files keep the user-facing parameters in one place: `valley`, `spin`,
`project.nlow_state_list`, `project.gauge`, `model.n_orb`, fitting settings,
and plotting settings. `symm.operations`, `source_config`, `symmetry_source`,
`sectors`, and `term_templates` are inferred by the program.

### Run KP

From the repository root:

```bash
kp project --config examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_Q5.yaml  # external-data
kp symm    --config examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_Q5.yaml  # external-data
kp model   --config examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_Q5.yaml  # external-data

kp project --config examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_Q7.yaml  # external-data
kp symm    --config examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_Q7.yaml  # external-data
kp model   --config examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_Q7.yaml  # external-data

kp project --config examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinful_Q7.yaml  # external-data
kp symm    --config examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinful_Q7.yaml  # external-data
kp model   --config examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinful_Q7.yaml  # external-data
```

Each case uses `project.gauge: auto`; users do not write `norb_fix_list`.

### Current Metrics

These GPU validation numbers were generated with the single-file YAMLs:

| case | gauge | model dim | active terms | all-band RMS / max | plotted-band RMS / max |
| --- | --- | ---: | ---: | ---: | ---: |
| Gamma Q5 | auto + linear low-subspace refinement | 124 | 634 | 3.350 / 11.455 meV before refinement | top 10 plot: 0.836 / 3.180 meV |
| M1 spinless Q7 | auto | 8 | 266 | 1.074 / 2.969 meV | bottom 8: 1.114 / 3.263 meV |
| M1 spinful Q7 | auto | 16 | 401 | 1.886 / 4.631 meV | bottom 8: 0.695 / 2.784 meV |

### Data Notes

The heavy TAPW arrays and symmetry-analysis outputs are external data. In this
local checkout they are available under `tapw/` for validation, but they are
not part of a light source release.
