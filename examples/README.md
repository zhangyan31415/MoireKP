# MoireKP examples

The repository includes five material directories and ten KP case configs:

| Directory | KP cases | Full workflow |
| --- | --- | --- |
| `mote2_3.89` | K1 spinful and spinless | tested |
| `mgi2_3.89` | Gamma spinful; M1 spinful and spinless | tested |
| `mote2_aab_5.09` | Gamma spinful; K1-A and K1-B single-spin sectors | tested |
| `zrs2_3.15` | Gamma spinful | tested |
| `ptse2_7.34` | Gamma spinful | not yet tested |

Configs, structure files, and small templates are tracked. Large OpenMX
matrices, TAPW arrays, symmetry exports, and generated KP outputs are external
and are not committed to the source repository.

## Workflow

Run the material's TAPW calculation and source-symmetry export first. Then use
the same KP YAML for projection, symmetry exactification, optional
band-representation analysis, and model fitting:

```bash
tapw run      -c examples/<material>/tapw/configs/<case>.yaml
tapw symm     -c examples/<material>/tapw/configs/<case>.yaml
tapw symm-rep -c examples/<material>/tapw/configs/<case>.yaml  # optional

kp project    -c examples/<material>/kp/configs/<case>.yaml
kp symm       -c examples/<material>/kp/configs/<case>.yaml
kp model      -c examples/<material>/kp/configs/<case>.yaml
kp symm-rep   -c examples/<material>/kp/configs/<case>.yaml    # optional
```

`kp project` writes the selected low-energy space and Heff. `kp symm`
exactifies the symmetry package required by `kp model`. `kp inspect` remains
an optional source-spectrum diagnostic.

## Automatic low-energy selection

KP configs use:

```yaml
project:
  selection: auto
  # Uncommenting this list makes the saved selection explicit.
  # nlow_state_list:
  #   - [...]
```

The commented list records the selected result where available. An explicit
`nlow_state_list` takes precedence and bypasses the search.

## Automatic harmonic counts

The configs omit `model.harmonics`. `kp model` enumerates the available
intra/inter Q-difference support, removes symmetry-equivalent duplicates,
performs low-cost Heff harmonic ablation, and fits the selected model once.
Polynomial cutoffs remain explicit under `model.max_order`. Experts may add
`model.harmonics` to bypass automatic selection.

The eight MoTe2/MgI2/A-AB configs use unweighted linear fits. The ZrS2 Gamma
config uses a linear fit with `two_sided_weight: 300.0`. The PtSe2 config
specifies a weighted nonlinear fit but has not yet been tested.

## Inline k path

TAPW and KP configs define the plotted path under `bands.kpath` with labels,
points per segment, and fractional coordinates. These examples do not require
`KPATH.in` for their standard Gamma-M-K-G plots.

## Installation check

Scientific runs require external data. The installed interfaces can be checked
without those files:

```bash
python -c "import tapw, kp"
tapw --help
kp --help
```
