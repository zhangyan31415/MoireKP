# MoireKP Release Examples

The release-facing examples contain five audited material families and ten KP
cases:

| Directory | KP cases |
|---|---|
| `mote2_3.89` | K1 spinful and spinless |
| `mgi2_3.89` | Gamma spinful; M1 spinful and spinless |
| `mote2_aab_5.09` | Gamma spinful; K1-A and K1-B single-spin sectors |
| `ptse2_7.34` | Gamma spinful |
| `zrs2_3.15` | Gamma spinful |

Each material directory contains small tracked OpenMX/TAPW templates and one
unified KP YAML per model. Large OpenMX matrices, TAPW arrays, symmetry
exports, and generated KP outputs are external data described by
`examples/data-manifest.yaml`.

## Workflow

Run the material's TAPW calculation and symmetry export first, then use the
same KP YAML for projection and model fitting:

```bash
tapw run  -c examples/<material>/tapw/configs/<case>.yaml  # external-data
tapw symm -c examples/<material>/tapw/configs/<case>.yaml  # external-data
kp project -c examples/<material>/kp/configs/<case>.yaml    # external-data
kp symm   -c examples/<material>/kp/configs/<case>.yaml    # external-data
kp model   -c examples/<material>/kp/configs/<case>.yaml    # external-data
```

`kp project` writes the selected spectrum, `kp symm` exactifies the symmetry
package used by `kp model`, and `kp inspect` remains an optional diagnostic.

## Automatic Low-Energy Selection

Every release KP config uses:

```yaml
project:
  selection: auto
  # Audited automatic result. Uncommenting this list makes selection explicit.
  # nlow_state_list:
  #   - [...]
```

The commented list records the reviewed result. If it is uncommented,
`nlow_state_list` takes precedence and bypasses the search, selecting the same
ordered low-energy bands as the audited automatic run.

## Automatic Harmonic Counts

Release configs intentionally omit `model.harmonics`. `kp model` enumerates
the actual intra/inter Q-difference support, removes duplicate masks, performs
a low-cost Heff harmonic ablation, and fits only the selected model once.
Polynomial cutoffs remain explicit under `model.max_order`. Experts may add
`model.harmonics` to bypass automatic selection.

PtSe2 uses nonlinear fitting. The other nine KP cases use linear fitting.

## Inline K-Path

TAPW and KP release configs define the plotted path directly under
`bands.kpath` with labels, points per segment, and fractional coordinates.
The release examples do not require `KPATH.in`; generated band plots read the
inline YAML and label the standard path as Gamma-M-K-G.

## Clean-Clone Checks

Scientific runs require external data. A clean clone can still run:

```bash
python -m pytest tests/test_release_contract.py tests/kp/test_example_dependency_contract.py -q  # clean-clone
tapw --help  # clean-clone
kp --help    # clean-clone
```

Dataset license, DOI, and public data URL values remain release blockers until
the external archive is finalized.
