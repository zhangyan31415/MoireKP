# Alternating trilayer WSe2 at 3.89 degrees

This example uses the (K_1) valence valley with six plane-wave shells.
There are 27 Q vectors per orientation and one low-energy valence state
per physical layer and Q vector. The outer layers share an orientation,
giving an 81-dimensional continuum Hamiltonian. The selected model uses
kinetic, intralayer and interlayer orders of 6, 4 and 4 and harmonic
counts of 4 and 3. It uses coefficient extraction at zero-based path
indices 10, 40 and 50.

- `openmx/POSCAR_relaxed` contains the geometry used for SCF. The archived
  NSOC and one-step SOC input decks are in the same directory.
- `openmx/POSCAR_rigid` contains the geometry used to project H/S in
  the corrected TAPW calculation, in the relaxed-cell hexagonal frame.
  `POSCAR_rigid_input` contains the supplied rigid geometry before this
  cell adjustment.
- `tapw/tapw_K.yaml` records the TAPW band settings. The companion
  `run_corrected_centers.py` assigns the correct physical K center to the
  outer and middle orientation groups.
- `kp/` contains the configuration, an archived 81-dimensional reduced
  Hamiltonian along the 61-point path, and the corresponding Q vectors,
  TAPW valence bands and threefold rotation used as KP inputs.
- `model/K/` contains the exported continuum coefficients and evaluator.

## Rebuild the continuum model from the archived input

Run these commands from the repository root with MoireKP installed:

```bash
CASE=examples/wse2_trilayer_3.89
python "$CASE/kp/prepare_input.py"
kp project -c "$CASE/kp/kp_K.yaml"
kp symm -c "$CASE/kp/kp_K.yaml"
kp model -c "$CASE/kp/kp_K.yaml"
```

The input script restores the saved spin-up reduced Hamiltonian, with empty
spin-down loader slots, and the corresponding rotation matrix. Generated
files go under `output/`. The reduced input uses the same orientation and
reciprocal frame as the selected KP model.

## OpenMX and TAPW inputs

Set `OPENMX`, `OPENMX_DATA`, `OPENMX_READER` and `NPROCS` using the
[OpenMX guide](../../scripts/openmx/README.md). The decks use the same
`System.Name` so the SOC step can read the converged NSOC restart. Retain
a copy of the converged NSOC outputs before the SOC run.

```bash
CASE=examples/wse2_trilayer_3.89
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
python "$CASE/tapw/run_corrected_centers.py" run -c "$CASE/tapw/tapw_K.yaml"
```

The archived KP input was obtained by a layer-resolved downfolding of
the corrected spinful TAPW matrices. The reduction keeps the W
(d_{x^2-y^2}+i d_{xy}) valence branch on each physical layer and uses
energy-linearized Löwdin reduction at -4.73009299107 eV. The KP commands
above start at the stored result of that reduction. The OpenMX and TAPW
commands regenerate the upstream inputs and do not replace the archived
reduced matrices automatically.
