# A–AB trilayer MoTe2

This directory supplies the relaxed first-principles input and the separate
rigid TAPW reference for the paper example.

- `openmx/scf/openmx.dat`: fixed-geometry, from-scratch OpenMX SCF input.
- `openmx/scf/relaxed.extxyz`: the relaxed structure in the SCF input.
- `openmx/soc/POSCAR_rigid`: the reference used by TAPW.

## Generate H/S

Install OpenMX and build its matching matrix reader following the
[OpenMX guide](../../scripts/openmx/README.md#paper-examples).
Set `OPENMX`, `OPENMX_DATA`, `OPENMX_READER` and `NPROCS` as shown there.
Run from the repository root:

```bash
CASE=examples/mote2_aab_5.09/openmx
ln -s "$OPENMX_DATA" "$CASE/scf/DFT_DATA19"
(cd "$CASE/scf" && mpirun -np "$NPROCS" "$OPENMX" openmx.dat -nt 1 > openmx.stdout.log)
```

Check that the SCF reached convergence before continuing. If it reaches its
iteration limit, continue the SCF calculation to convergence before exporting H/S.
Use the relaxed SCF input for importing and symmetry averaging:

```bash
tapw prepare-hs openmx --input "$CASE/scf/openmx.scfout" \
  --structure "$CASE/scf/openmx.dat" --binary "$OPENMX_READER" \
  --output "$CASE/prepared" --format npz --symmetrize \
  --symprec 0.001 --threads 1 --assume-nonmagnetic
cp "$CASE/prepared/symmetrized/H_symm.npz" "$CASE/soc/H_symm.npz"
cp "$CASE/prepared/symmetrized/S_symm.npz" "$CASE/soc/S_symm.npz"
```

`prepared` must be a new output directory. The matrices are regenerated locally;
they are not included in the source repository. Use the supplied rigid reference
for the following TAPW steps.

## Construct the models

| Model | TAPW config under `tapw/configs/` | KP config under `kp/configs/` |
|---|---|---|
| KA | `mote2_aab_5.09_K1_spinful_q04.yaml` | `mote2_aab_5.09_K1_A_q04.yaml` |
| KB | `mote2_aab_5.09_K1_spinful_q04.yaml` | `mote2_aab_5.09_K1_B_q04.yaml` |
| Gamma | `mote2_aab_5.09_Gamma_spinful_q04.yaml` | `mote2_aab_5.09_Gamma_spinful_q04.yaml` |

Run the commands below for each matching configuration pair.

```bash
TAPW_CFG=examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml
KP_CFG=examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

The YAML files contain the model settings and output locations.
For geometry and topology, use `tapw topo --help` or the generated
`model/evaluate.py` topology settings.
