## MoTe2 3.89

This release example covers the MoTe2 K1 valley at 3.89 degrees. The active KP
entry points are:

```text
kp/configs/source/mote2_3.89_K1.yaml
kp/configs/model/mote2_3.89_K1.yaml
kp/configs/source/mote2_3.89_K1_spinful.yaml
kp/configs/model/mote2_3.89_K1_spinful.yaml
```

Both cases use automatic gauge anchors. The spinful case exercises the
spinful C3z exactification path: the projected raw-H C3z action is treated as a
block action first, then cleaned to a monomial continuum representation when
the block support is clearly cleaner than the raw monomial support. This avoids
using a single monomial phase branch for the two spinful C3z branches.

### Run KP

From the repository root:

```bash
kp proj -c examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml  # external-data
kp symm -c examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml  # external-data
kp fit  -c examples/mote2_3.89/kp/configs/model/mote2_3.89_K1.yaml  # precomputed/external-data

kp proj -c examples/mote2_3.89/kp/configs/source/mote2_3.89_K1_spinful.yaml  # external-data
kp symm -c examples/mote2_3.89/kp/configs/source/mote2_3.89_K1_spinful.yaml  # external-data
kp fit  -c examples/mote2_3.89/kp/configs/model/mote2_3.89_K1_spinful.yaml  # precomputed/external-data
```

### Current Metrics

These GPU validation numbers were generated with the active release configs:

| case | gauge | model dim | active terms | all-band RMS / max | plotted-band RMS / max |
| --- | --- | ---: | ---: | ---: | ---: |
| K1 spin up | auto | 8 | 94 | 1.215 / 4.907 meV | top 8: 1.226 / 4.942 meV |
| K1 spinful Q6 | auto | 16 | 386 | 1.470 / 4.740 meV | top 8: 1.235 / 4.841 meV |

### Data Notes

The heavy TAPW arrays and symmetry-analysis outputs are external data. In this
local checkout they are available under `tapw/` for validation, but they are
not part of a light source release.
