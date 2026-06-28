## MoTe2 3.89

This release example covers the MoTe2 K1 valley at 3.89 degrees. The
release-facing KP entry points use one config per case:

```text
kp/configs/mote2_3.89_K1_q06.yaml
kp/configs/mote2_3.89_K1_up_q06.yaml
```

Release examples do not use split source/model config directories. Both cases
use automatic gauge anchors. The spinful case exercises the
spinful C3z exactification path: the projected raw-H C3z action is treated as a
block action first, then cleaned to a monomial continuum representation when
the block support is clearly cleaner than the raw monomial support. This avoids
using a single monomial phase branch for the two spinful C3z branches.

### Run KP

From the repository root:

```bash
kp inspect -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml
kp symm    -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml

kp inspect -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_up_q06.yaml
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_up_q06.yaml
kp symm    -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_up_q06.yaml
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_up_q06.yaml
```

Run `inspect` first and check `kp/outputs/<profile>/<q_shell>/inspect/` before
changing `project.nlow_state_list`.

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
