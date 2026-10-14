# A–AB trilayer MoTe2

`openmx/` contains the first-principles inputs; `tapw/` and `kp/` contain
the corresponding model configurations.

The OpenMX inputs are:

- `openmx_nsoc.dat`: NSOC self-consistent calculation on the relaxed structure.
- `openmx_soc.dat`: one SOC iteration using the converged NSOC restart files.
- `POSCAR_relaxed`: the structure used by both OpenMX calculations.
- `POSCAR_rigid`: the separate reference used by TAPW.

## Generate H/S

Install OpenMX and build its matching matrix reader following the
[OpenMX guide](../../scripts/openmx/README.md#paper-examples).
Set `OPENMX`, `OPENMX_DATA`, `OPENMX_READER` and `NPROCS` as shown there.
Run from the repository root:

```bash
CASE=examples/mote2_aab_5.09/openmx
ln -s "$OPENMX_DATA" "$CASE/DFT_DATA19"
(cd "$CASE" && mpirun -np "$NPROCS" "$OPENMX" openmx_nsoc.dat -nt 1 > nsoc.stdout.log)
```

Check that NSOC reached convergence before continuing. If it reaches its
iteration limit, continue NSOC to convergence first. Keep `nsoc_rst/` in this
directory, then run the one-step SOC calculation with the same MPI rank count:

```bash
test -f "$CASE/nsoc_rst/nsoc.crst_check"
(cd "$CASE" && mpirun -np "$NPROCS" "$OPENMX" openmx_soc.dat -nt 1 > soc.stdout.log)
```

The SOC input reads `nsoc_rst/` with `scf.restart c2n` and writes `soc.scfout`.
Check the SOC log for successful restart loading before importing H/S.
This is one-step SOC on the converged NSOC density.

```bash
tapw prepare-hs openmx --input "$CASE/soc.scfout" \
  --structure "$CASE/openmx_soc.dat" --binary "$OPENMX_READER" \
  --output "$CASE/prepared" --format npz --symmetrize \
  --symprec 0.001 --threads 1 --assume-nonmagnetic
cp "$CASE/prepared/symmetrized/H_symm.npz" "$CASE/H_symm.npz"
cp "$CASE/prepared/symmetrized/S_symm.npz" "$CASE/S_symm.npz"
```

`prepared` must be a new output directory. The matrices are generated locally.
Import and symmetry averaging use the relaxed OpenMX geometry; TAPW uses
`POSCAR_rigid` with the same atom order and orbital basis.

## Construct the models

| Model | TAPW input | KP input |
|---|---|---|
| KA | `tapw/tapw_K.yaml` | `kp/kp_KA.yaml` |
| KB | `tapw/tapw_K.yaml` | `kp/kp_KB.yaml` |
| Gamma | `tapw/tapw_Gamma.yaml` | `kp/kp_Gamma.yaml` |

Run the commands for each matching pair.

```bash
TAPW_CFG=examples/mote2_aab_5.09/tapw/tapw_K.yaml
KP_CFG=examples/mote2_aab_5.09/kp/kp_KA.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

Results are written under `output/tapw/` and `output/kp/`.
For geometry and topology, use `tapw topo --help` or the generated
`model/evaluate.py` topology settings.
