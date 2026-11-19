# AB bilayer ZrS2 at 2.13°

The 4,326-atom OpenMX supercell uses the Zr7.0-s3p2d1 and S7.0-s2p2d1
orbital basis. Both input decks use `openmx/POSCAR_relaxed`. TAPW uses
`openmx/POSCAR_rigid` with the same site and orbital order.

| Valley | Dimension | Real coefficient entries | Target bands | RMS error (meV) | Maximum error (meV) |
|---|---:|---:|---|---:|---:|
| Gamma | 152 | 1350 | Top 10 valence | 0.385 | 0.792 |
| M1 | 88 | 1039 | Bottom 8 conduction | 1.057 | 1.882 |

The errors are relative to TAPW after alignment at the relevant band edge
over the 61-point path. Gamma uses the eight explicitly selected local
states (40–47) and automatic gauge selection. M1 uses automatic state
selection. The coefficient extraction is linear and both models use
matrix-element weighting.

## Evaluate the continuum models

From this example directory, run each standalone export with NumPy:

```bash
python model/Gamma/evaluate.py --model-root model/Gamma --out bands_Gamma.npy
python model/M1/evaluate.py --model-root model/M1 --out bands_M1.npy
```

The exports use the archived momenta when `--kpoints` is omitted. Energies
are in eV.

## Generate H/S

Follow the [OpenMX guide](../../scripts/openmx/README.md#paper-examples) to
set `OPENMX`, `OPENMX_DATA`, `OPENMX_READER`, and `NPROCS`. Build the
reader against the same OpenMX 3.9 source used for the electronic
calculation. Run from the repository root:

```bash
CASE=examples/zrs2_2.13/openmx
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
  --symprec 0.001 --threads 1 --assume-nonmagnetic
cp "$CASE/prepared/symmetrized/H_symm.npz" "$CASE/H_symm.npz"
cp "$CASE/prepared/symmetrized/S_symm.npz" "$CASE/S_symm.npz"
```

The SOC log should confirm it read the NSOC restart. The importer averages
the real-space H/S on the relaxed geometry over six nonmagnetic operations.
Use a new `prepared` directory. Large H/S arrays are generated locally.

## Construct the models

From the repository root:

```bash
for VALLEY in Gamma M; do
  TAPW_CFG="examples/zrs2_2.13/tapw/tapw_${VALLEY}.yaml"
  KP_CFG="examples/zrs2_2.13/kp/kp_${VALLEY}.yaml"
  tapw run  -c "$TAPW_CFG"
  tapw symm -c "$TAPW_CFG"
  kp project -c "$KP_CFG"
  kp symm    -c "$KP_CFG"
  kp model   -c "$KP_CFG"
done
```

Workers and BLAS threads in the YAMLs can be adjusted to the compute allocation.
