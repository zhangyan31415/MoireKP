# AA bilayer MoTe2 at 2.13°

The (K_1) model has dimension 54 and 246 real coefficient entries.
The 4,326-atom OpenMX supercell uses the Mo7.0-s3p2d2 and Te7.0-s3p2d2f1
orbital basis. Both OpenMX input decks use the relaxed coordinates in
`openmx/POSCAR_relaxed`. The TAPW calculation uses
`openmx/POSCAR_rigid`: the reference in-plane sites were mapped to the
relaxed cell with the original site order and out-of-plane positions.

## Evaluate the continuum model

From this example directory, run the standalone export with NumPy:

```bash
python model/K1/evaluate.py --model-root model/K1 --out bands_K1.npy
```

The export uses the archived momenta when `--kpoints` is omitted. Energies
are in eV.

## Generate H/S

Follow the [OpenMX guide](../../scripts/openmx/README.md#paper-examples) to
set `OPENMX`, `OPENMX_DATA`, `OPENMX_READER`, and `NPROCS`. Run from the
repository root:

```bash
CASE=examples/mote2_2.13/openmx
ln -s "$OPENMX_DATA" "$CASE/DFT_DATA19"
(cd "$CASE" && mpirun -np "$NPROCS" "$OPENMX" openmx_nsoc.dat -nt 1 > nsoc.stdout.log)
```

Check that NSOC reaches convergence and that `nsoc_rst/nsoc.crst_check`
exists. Then run one SOC iteration with the same rank count:

```bash
(cd "$CASE" && mpirun -np "$NPROCS" "$OPENMX" openmx_soc.dat -nt 1 > soc.stdout.log)
tapw prepare-hs openmx --input "$CASE/soc.scfout" \
  --structure "$CASE/openmx_soc.dat" --binary "$OPENMX_READER" \
  --output "$CASE/prepared" --format npz --symmetrize \
  --symprec 0.05 --threads 1 --assume-nonmagnetic
cp "$CASE/prepared/symmetrized/H_symm.npz" "$CASE/H_symm.npz"
cp "$CASE/prepared/symmetrized/S_symm.npz" "$CASE/S_symm.npz"
```

The SOC log should confirm it read the NSOC restart. The importer averages
the real-space H/S on the relaxed geometry over six nonmagnetic operations.
Use a new `prepared` directory. Large H/S arrays are generated locally.

## Construct the model

From the repository root:

```bash
TAPW_CFG=examples/mote2_2.13/tapw/tapw_K.yaml
KP_CFG=examples/mote2_2.13/kp/kp_K.yaml
tapw run  -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

The projected model retains one local state per layer at 27 Q points per
layer. The coefficient extraction uses linear matrix-element weighting
at path rows 0, 20, and 40. The top-eight-band RMS and maximum errors are
0.365 and 0.982 meV relative to TAPW after valence-edge alignment over
the 61-point path. Workers and BLAS threads in the YAMLs can be adjusted
to the compute allocation.

## 61-point TAPW validation

The frozen coefficients were evaluated on the full Gamma–M–K–Gamma path. A single constant aligns each model edge with TAPW at Gamma. The table uses all 61 points.

| Valley | RMS (meV) | Maximum (meV) |
| --- | ---: | ---: |
| K1 | 0.422 | 0.999 |

The `model/` directories contain the corresponding `reference_bands.npz` and `native_overlaps.npz`. Squared overlaps are computed in orthonormal TAPW space, with a common 3 meV grouping threshold for nearby matched bands. Run `python examples/additional_models/verify_models.py --case mote2_2.13` from the repository root to replay the archived comparisons.
