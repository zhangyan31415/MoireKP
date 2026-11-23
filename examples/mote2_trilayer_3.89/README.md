# Alternating trilayer MoTe2 at 3.89 degrees

| Valley | TAPW configuration | Model configuration | Model and validation data |
|---|---|---|---|
| K1 | [tapw_K1.yaml](tapw/tapw_K1.yaml) | [kp_K1.yaml](kp/kp_K1.yaml) | [model/K1/](model/K1/) |

This example uses the (K_1) valence valley of the A–A1–A trilayer with
six plane-wave shells. It retains 27 Q vectors per orientation and one
local valence state per physical layer and Q vector. The outer layers
share an orientation, giving an 81-dimensional continuum Hamiltonian.
The selected model uses kinetic, intralayer and interlayer orders of
6, 4 and 4 and harmonic counts of 4 and 4. It uses coefficient extraction
at zero-based path indices 10 and 50.

- `openmx/POSCAR_relaxed` contains the geometry used for SCF. The archived
  NSOC and one-step SOC input decks are in the same directory.
- `openmx/POSCAR_rigid` contains the geometry used in the TAPW projector
  with the original layer and atom order.
- `tapw/tapw_K1.yaml` records the upstream band settings in the public
  YAML schema. The companion `run_corrected_centers.py` assigns the
  physical K centers to the outer and middle orientation groups.
- `kp/` contains the configuration, an archived 81-dimensional reduced
  Hamiltonian along the 61-point path, and the corresponding Q vectors,
  TAPW valence bands and threefold rotation used as KP inputs.
- `model/K1/` contains the exported continuum coefficients and evaluator.

## Rebuild the continuum model from the archived input

Run these commands from the repository root with MoireKP installed:

```bash
CASE=examples/mote2_trilayer_3.89
unzip -n "$CASE/kp/input_K1.zip" -d "$CASE"
kp project -c "$CASE/kp/kp_K1.yaml"
kp symm -c "$CASE/kp/kp_K1.yaml"
kp model -c "$CASE/kp/kp_K1.yaml"
```

The archive contains the saved spin-up reduced Hamiltonian, with empty
spin-down loader slots, and the corresponding rotation matrix. Generated
files go under `output/`. The corrected TAPW reference was calculated from
the nonorthogonal projected H/S matrices. The three G vectors per layer
that change under the center reassignment were reprojected against the
entire corrected projector. The reduced input was then built by a
layer-resolved generalized Schur reduction at -4.59737978976 eV.
The archived input bypasses the need to rerun that large-matrix stage.

## OpenMX and TAPW inputs

Set `OPENMX`, `OPENMX_DATA`, `OPENMX_READER` and `NPROCS` using the
[OpenMX guide](../../scripts/openmx/README.md). The decks use the same
`System.Name` so the SOC step can read the converged NSOC restart. Retain
a copy of the converged NSOC outputs before the SOC run.

```bash
CASE=examples/mote2_trilayer_3.89
ln -s "$OPENMX_DATA" "$CASE/openmx/DFT_DATA19"
(cd "$CASE/openmx" && mpirun -np "$NPROCS" "$OPENMX" openmx_nsoc.dat -nt 1 > nsoc.stdout.log)
```

After checking NSOC convergence and keeping `openmx_rst/`, run:

```bash
(cd "$CASE/openmx" && mpirun -np "$NPROCS" "$OPENMX" openmx_soc.dat -nt 1 > soc.stdout.log)
tapw prepare-hs openmx --input "$CASE/openmx/openmx.scfout" \
  --structure "$CASE/openmx/openmx_soc.dat" --binary "$OPENMX_READER" \
  --output "$CASE/openmx/prepared" --format npz
cp "$CASE/openmx/prepared/H.npz" "$CASE/openmx/H.npz"
cp "$CASE/openmx/prepared/S.npz" "$CASE/openmx/S.npz"
python "$CASE/tapw/run_corrected_centers.py" run -c "$CASE/tapw/tapw_K1.yaml"
```

The published configuration and reduced matrices use the archived
generalized H/S workflow. The TAPW command above gives the upstream route
in the current package and does not carry out the generalized Schur
reduction or replace the archived KP input automatically.
