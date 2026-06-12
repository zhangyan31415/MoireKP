## MgI2 3.89

- `openmx/`: OpenMX input snapshots for this material/angle.
- `tapw/`: TAPW inputs, Q-shell data, and symmetry-analysis outputs used by KP.
- `kp/configs/source/`: canonical `plot / project / symm` inputs.
- `kp/configs/model/`: canonical continuum-model configs.
- `kp/notebooks/`: notebooks wired to this directory layout.

Main commands:

```bash
PYTHONPATH=kp python -m kp.cli plot --config kp/examples/mgi2/3.89/kp/configs/source/mgi2_3.89_Gamma.yaml
PYTHONPATH=kp python -m kp.cli project --config kp/examples/mgi2/3.89/kp/configs/source/mgi2_3.89_M1.yaml
PYTHONPATH=kp python -m kp.cli project --config kp/examples/mgi2/3.89/kp/configs/source/mgi2_3.89_M1_spinful.yaml
PYTHONPATH=kp python -m kp.cli model --config kp/examples/mgi2/3.89/kp/configs/model/mgi2_3.89_Gamma.yaml
PYTHONPATH=kp python -m kp.cli model --config kp/examples/mgi2/3.89/kp/configs/model/mgi2_3.89_M1.yaml
PYTHONPATH=kp python -m kp.cli model --config kp/examples/mgi2/3.89/kp/configs/model/mgi2_3.89_M1_spinful.yaml
```

Production status on 2026-06-03:

- `Gamma`:
  - active config: `kp/configs/model/mgi2_3.89_Gamma.yaml`
  - user-facing symmetry names: `C3z`, `TR`, `C2`
  - aligned `top4` external-data comparison: RMS about `0.670 meV`, max about `2.210 meV`

- `M1`, `M2`, `M3`:
  - raw TAPW `hamk_*_valley.npy` and `g_vec_list_*` inputs live under `tapw/Q_shell_7/`
  - `kp project` writes `44 x 44` Heff for all three valleys
  - active config currently covers `M1`
  - user-facing symmetry names: `TR`, `C2`
  - `M1` aligned `bottom8` comparison is: RMS about `1.179 meV`, max about `3.987 meV`
  - `M1 spinful`: active config is `kp/configs/model/mgi2_3.89_M1_spinful.yaml`; source/project/symm input is `kp/configs/source/mgi2_3.89_M1_spinful.yaml`
  - symmetry action metadata and exactified continuum matrices are read from the `kp symm` manifest

## Historical Artifacts

The active configs above write to `kp/outputs/...`. Older saved runs may still exist under `kp/runs/...` and `kp/outputs/model/production/...`; keep them as historical comparison artifacts only, not as active command targets.
