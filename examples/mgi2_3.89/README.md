## MgI2 3.89

This release example covers MgI2 at 3.89 degrees. The active KP entry points
use split source/model configs:

```text
kp/configs/source/mgi2_3.89_Gamma.yaml
kp/configs/model/mgi2_3.89_Gamma.yaml
kp/configs/source/mgi2_3.89_M1.yaml
kp/configs/model/mgi2_3.89_M1.yaml
kp/configs/source/mgi2_3.89_M1_spinful.yaml
kp/configs/model/mgi2_3.89_M1_spinful.yaml
```

Source configs drive inspection, projection, and symmetry projection. Model
configs consume the projected/exactified outputs and define fitting settings,
sectors, and term templates.

### Run KP

From the repository root:

```bash
kp proj -c examples/mgi2_3.89/kp/configs/source/mgi2_3.89_Gamma.yaml  # external-data
kp symm -c examples/mgi2_3.89/kp/configs/source/mgi2_3.89_Gamma.yaml  # external-data
kp fit  -c examples/mgi2_3.89/kp/configs/model/mgi2_3.89_Gamma.yaml  # precomputed/external-data
kp export -c examples/mgi2_3.89/kp/configs/model/mgi2_3.89_Gamma.yaml -o exported/mgi2_3.89_Gamma  # precomputed

kp proj -c examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1.yaml  # external-data
kp symm -c examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1.yaml  # external-data
kp fit  -c examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1.yaml  # precomputed/external-data
kp export -c examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1.yaml -o exported/mgi2_3.89_M1  # precomputed

kp proj -c examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1_spinful.yaml  # external-data
kp symm -c examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1_spinful.yaml  # external-data
kp fit  -c examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1_spinful.yaml  # precomputed/external-data
kp export -c examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1_spinful.yaml -o exported/mgi2_3.89_M1_spinful  # precomputed
```

Each case uses `project.gauge: auto`; users do not write `norb_fix_list`.

### Current Metrics

These GPU validation numbers were generated with the active release configs:

| case | gauge | model dim | active terms | all-band RMS / max | plotted-band RMS / max |
| --- | --- | ---: | ---: | ---: | ---: |
| Gamma Q5 | auto + linear low-subspace refinement | 124 | 634 | 3.350 / 11.455 meV before refinement | top 10 plot: 0.836 / 3.180 meV |
| M1 spinless Q7 | auto | 8 | 266 | 1.074 / 2.969 meV | bottom 8: 1.114 / 3.263 meV |
| M1 spinful Q7 | auto | 16 | 401 | 1.886 / 4.631 meV | bottom 8: 0.695 / 2.784 meV |

### Data Notes

The heavy TAPW arrays and symmetry-analysis outputs are external data. In this
local checkout they are available under `tapw/` for validation, but they are
not part of a light source release.
