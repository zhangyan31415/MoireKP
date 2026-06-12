## MoTe2 3.89

- `openmx/`: OpenMX input snapshots for this material/angle.
- `tapw/`: TAPW inputs and Q-shell outputs used by KP.
- `kp/configs/source/`: canonical `plot / project / symm` inputs.
- `kp/configs/model/`: canonical continuum-model configs.
- `kp/notebooks/`: notebooks wired to this directory layout.

Main commands:

```bash
kp plot --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml
kp project --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml
kp project --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1_spinful.yaml
kp model --config examples/mote2_3.89/kp/configs/model/mote2_3.89_K1.yaml
kp model --config examples/mote2_3.89/kp/configs/model/mote2_3.89_K1_spinful.yaml
```

Production status on 2026-06-03:

- `K1`: active config is `kp/configs/model/mote2_3.89_K1.yaml`
- `K1`: aligned `top8` external-data comparison is: RMS about `1.226 meV`, max about `4.942 meV`
- `K1 spinful`: active config is `kp/configs/model/mote2_3.89_K1_spinful.yaml`; source/project/symm input is `kp/configs/source/mote2_3.89_K1_spinful.yaml`
- `K2`: TAPW-projected symmetry data exists, but the canonical published production example is currently `K1`
- `Gamma`: TAPW full-space symmetry export exists, but the current published 10-band KP example remains `K1`

## Historical Artifacts

The active configs above write to `kp/outputs/...`. Older saved runs may still exist under `kp/runs/...` and `kp/outputs/model/production/...`; keep them as historical comparison artifacts only, not as active command targets.
