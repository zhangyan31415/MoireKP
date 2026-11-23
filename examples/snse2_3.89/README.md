# AA bilayer SnSe2 at 3.89°

| Valley | TAPW configuration | Model configuration | Model and validation data |
|---|---|---|---|
| Gamma | [tapw_Gamma.yaml](tapw/tapw_Gamma.yaml) | [kp_Gamma.yaml](kp/kp_Gamma.yaml) + [refinement](kp/refine_Gamma.yaml) | [model/Gamma/](model/Gamma/) |
| M1 | [tapw_M1.yaml](tapw/tapw_M1.yaml) | [kp_M1.yaml](kp/kp_M1.yaml) | [model/M1/](model/M1/) |

These examples use the Sn-centered, machine-learning-force-field-relaxed
1302-atom bilayer at a fixed supercell size. The two valley models share
the same first-principles input and spatially averaged H/S pair.


The OpenMX inputs use the relaxed coordinates embedded in both decks.
[POSCAR_relaxed](openmx/POSCAR_relaxed) records those coordinates.
[POSCAR_rigid](openmx/POSCAR_rigid) records the rigid TAPW reference in the
same atom order and supercell. The basis is `s3p2d1` on both Sn and Se.

## Generate H/S

Follow the [OpenMX build guide](../../scripts/openmx/README.md) and set
`OPENMX`, `OPENMX_DATA`, `OPENMX_READER`, and `NPROCS`. The archived inputs
select ELPA2. From the repository root, run NSOC and then one SOC iteration.
Both OpenMX inputs use the system name `openmx`, so copy the converged NSOC
restart directory before starting SOC. Use the same MPI rank count in both
stages.

```bash
CASE=examples/snse2_3.89/openmx
mkdir -p "$CASE/nsoc" "$CASE/soc"
cp "$CASE/openmx_nsoc.dat" "$CASE/nsoc/"
cp "$CASE/openmx_soc.dat" "$CASE/soc/"
ln -s "$OPENMX_DATA" "$CASE/nsoc/DFT_DATA19"
ln -s "$OPENMX_DATA" "$CASE/soc/DFT_DATA19"
(cd "$CASE/nsoc" && mpirun -np "$NPROCS" "$OPENMX" openmx_nsoc.dat -nt 1 > nsoc.stdout.log)
```

Check NSOC convergence before running SOC. The SOC input uses
`scf.restart c2n`.

```bash
test -d "$CASE/nsoc/openmx_rst"
cp -a "$CASE/nsoc/openmx_rst" "$CASE/soc/"
(cd "$CASE/soc" && mpirun -np "$NPROCS" "$OPENMX" openmx_soc.dat -nt 1 > soc.stdout.log)
```

Check the SOC log for successful restart loading. The reference calculation
averages the spinful H and S over the six proper rotations of the relaxed
structure (space group P312, point group D3) at 0.01 Å site tolerance.
The native OpenMX reader applies that average with a cutoff of `1e-7`.

```bash
python "$CASE/make_transports.py" "$CASE/soc/openmx.dat#" "$CASE/transports.json"
"$OPENMX_READER" "$CASE/soc/openmx.scfout" \
  --symmetry-json "$CASE/transports.json" --output-dir "$CASE/symmetrized" \
  --output-format npz --threads 1 --cutoff 1e-7 --diagnostics full
cp "$CASE/symmetrized/H.npz" "$CASE/H_symm.npz"
cp "$CASE/symmetrized/S.npz" "$CASE/S_symm.npz"
```

## Run TAPW and KP

```bash
CASE=examples/snse2_3.89
for VALLEY in Gamma M1; do
  tapw run -c "$CASE/tapw/tapw_${VALLEY}.yaml"
  tapw symm -c "$CASE/tapw/tapw_${VALLEY}.yaml"
  kp project -c "$CASE/kp/kp_${VALLEY}.yaml"
  kp symm -c "$CASE/kp/kp_${VALLEY}.yaml"
  kp model -c "$CASE/kp/kp_${VALLEY}.yaml"
done
```

Generated files are written under `output/`. The above commands construct
the linear Gamma reference and the M1 model. The Gamma coefficients in the
comparison also use the refinement below.

## Reproduce the Gamma refinement

[refine_Gamma.yaml](kp/refine_Gamma.yaml) records the selected settings.
[input_Gamma.zip](kp/input_Gamma.zip) supplies the symmetrically adapted,
projected Hamiltonian, the frozen polynomial basis, the linear spectrum and
the saved starting coefficients for this refinement. The driver starts from
this checkpoint and saves new coefficients and an independent NumPy
evaluator to the named output directory.

```bash
CASE=examples/snse2_3.89
unzip -n "$CASE/kp/input_Gamma.zip" -d "$CASE"
python "$CASE/kp/refine_Gamma.py" "$CASE/kp/refine_Gamma.yaml" --check-inputs
python "$CASE/kp/refine_Gamma.py" "$CASE/kp/refine_Gamma.yaml" --output "$CASE/output/refined_Gamma"
```

The input check replays the linear spectrum. The refinement compiles a
cached response array of about 2.24 GB and needs additional working memory.
The selected reference run reached its 300-evaluation budget.

## 61-point TAPW validation

The frozen coefficients were evaluated on the full Gamma–M–K–Gamma path. A single constant aligns each model edge with TAPW at Gamma. The table uses all 61 points.

| Valley | RMS (meV) | Maximum (meV) |
| --- | ---: | ---: |
| Gamma | 0.744 | 1.363 |
| M1 | 0.710 | 1.733 |

The `model/` directories contain the corresponding `reference_bands.npz` validation archives. Squared overlaps are computed in orthonormal TAPW space, with a common 3 meV grouping threshold for nearby matched bands. Run `python examples/additional_models/verify_models.py --case snse2_3.89` from the repository root to replay the archived comparisons.
