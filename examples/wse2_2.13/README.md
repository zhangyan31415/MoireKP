# AA bilayer WSe2 at 2.13 degrees

The spin-up K1 valence model has dimension 54, with one local state per
layer and 27 plane-wave channels per layer (TAPW Q shell 6). The six upper
valence bands have a TAPW RMS and maximum error of **0.547 / 1.377 meV** on the
31-point Gamma–M–K–Gamma path after Gamma-edge alignment. The model uses the
SOC OpenMX Hamiltonian of a 4326-atom structure relaxed with a
machine-learned force field at fixed cell.

The linear extraction uses equal one-sided and two-sided weights of 1, both over a six-state window. The extraction rows are 0, 2, 5, 10, 12, 15, 20, 22, 25, and 29 (zero-based). The two intralayer and three interlayer shells are the counts selected in the reported run.

The [standalone model](model/K1/) evaluates the extracted coefficients with
NumPy. The [TAPW](tapw/tapw_K1.yaml) and [KP](kp/kp_K1.yaml) YAML files below
specify how to regenerate the model.

## Reproduce the coefficient extraction

The [projected input archive](kp/projected_input.zip) contains the saved
effective Hamiltonian, projection and symmetry artifacts, and the small TAPW
reference tables and Q sets. Unpack it in a fresh example directory and run
the same KP configuration used for the full build.

```bash
CASE=examples/wse2_2.13
unzip -n "$CASE/kp/projected_input.zip" -d "$CASE"
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  kp model -c "$CASE/kp/kp_K1.yaml"
```

The regenerated continuum Hamiltonian matches the archived model on all 31
path points to 1.6e-14 eV in the largest matrix element difference.

## OpenMX inputs

- [openmx_nsoc.dat](openmx/openmx_nsoc.dat) is the NSOC self-consistent input.
- [openmx_soc.dat](openmx/openmx_soc.dat) reads the NSOC restart and runs one
  SOC iteration on the same relaxed coordinates.
- [POSCAR_relaxed](openmx/POSCAR_relaxed) records the geometry in the
  NSOC and SOC decks.
- [POSCAR_rigid](openmx/POSCAR_rigid) records the TAPW reference geometry.
  [openmx_rigid.dat](openmx/openmx_rigid.dat) is the original rigid geometry
  deck in OpenMX format.

Install OpenMX with DFT_DATA19 and build the matching reader following
the [OpenMX guide](../../scripts/openmx/README.md#paper-examples).
Set `OPENMX`, `OPENMX_DATA`, `OPENMX_READER`, and `NPROCS` as described there.

Run from the repository root.

```bash
CASE=examples/wse2_2.13
ln -s "$OPENMX_DATA" "$CASE/openmx/DFT_DATA19"
(cd "$CASE/openmx" && mpirun -np "$NPROCS" "$OPENMX" openmx_nsoc.dat -nt 1 > nsoc.stdout.log)
```

Check the NSOC log for convergence and retain the `openmx_rst/` files.
The original decks share the same OpenMX system name. The SOC step uses
the NSOC restart and writes `openmx.scfout` in the same directory.

```bash
test -f "$CASE/openmx/openmx_rst/openmx.crst_check"
(cd "$CASE/openmx" && mpirun -np "$NPROCS" "$OPENMX" openmx_soc.dat -nt 1 > soc.stdout.log)
tapw prepare-hs openmx --input "$CASE/openmx/openmx.scfout" \
  --structure "$CASE/openmx/openmx_soc.dat" --binary "$OPENMX_READER" \
  --output "$CASE/openmx/prepared" --format npz --threads 1
```

The workflow uses raw SOC H/S. The importer reads the relaxed geometry and
retains the OpenMX source orbital order. TAPW uses the rigid geometry with the
same atom order to construct its basis and permute the source matrices.
Large H/S matrices and TAPW results are generated locally.

## Construct the continuum model

```bash
TAPW_CFG="$CASE/tapw/tapw_K1.yaml"
KP_CFG="$CASE/kp/kp_K1.yaml"
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

These commands write to `output/tapw/` and `output/kp/` under the example.
