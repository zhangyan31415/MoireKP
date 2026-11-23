# AB bilayer CaI2 at 3.89 degrees

The M1 conduction model uses a 1302-atom fixed-cell MLFF structure with
the ST OpenMX/SOC basis. Each layer retains one Kramers pair, giving dimension
72 with TAPW Q shell 6. The 932 real coefficients
use kinetic/intralayer/interlayer orders 6/5/5 and 3/5 harmonic counts.
The lowest six bands compared with TAPW give **0.032 meV RMS** and
**0.063 meV maximum** on 20 held-out momenta, after a single
Gamma-edge alignment. The two-sided and one-sided matrix weights, target
window, and sampling indices are in [the KP YAML](kp/kp_M1.yaml).

## Gamma valence model

The [fixed-angle Gamma model](model/Gamma/) retains six molecular spinors per
Q vector. Its **20 real coefficients** were refined at this angle on its own
TAPW training data. Its six-band RMS and maximum errors on 45 held-out
path points are **0.272 meV** and **0.634 meV**.
The [Gamma model configuration](model/Gamma/MODEL.json) lists the training
momenta. The model directory also contains the fixed basis, operator
templates, and sampled wave-function overlaps.

### Refit the 20 Gamma coefficients

The 3.89° Γ entry of Fig. R1 is a fixed-angle, nonlinear molecular model.
Its [model settings](model/Gamma/MODEL.json), [fixed operators](model/Gamma/FIXED_OPERATOR_TEMPLATES.npz)
and [training input](kp/input_Gamma.npz) reproduce the fit from the actual pre-refinement seed.
The archive contains eight TAPW target energies at 61 momenta (relative to the Γ valence edge),
six TAPW-to-molecular overlap matrices at the wave-function training points,
and the 20 initial real controls. Fifteen distinct momenta enter the energy loss.
The highest six bands are the R1 validation window. From the repository root:

```bash
CASE=examples/cai2_ab_3.89
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python "$CASE/kp/refine_Gamma.py" --workers 1 --output "$CASE/output/refitted_Gamma"
```

The output directory must be new. The script requires NumPy and SciPy and writes
`MATERIAL_CONTROLS.npz` and `COEFFICIENTS.npz` in the same format as the published model.

## Evaluate the continuum model

The [standalone M1 model](model/M1/) contains the original extracted
coefficients, numerical eigensolver and default Q-vector realization. From
the checkout root:

```bash
python examples/additional_models/verify_models.py --case cai2_ab_3.89
```

[Stored TAPW bands, continuum spectra, and rotation audit](reference/M1/)
include the M2 and M3 spectra after rotating the local wavevector of M1 by
the corresponding C3 operation. The three valleys use one common energy shift.

## Reproduce the M1 coefficient extraction

The [projected input archive](kp/projected_input_M1.zip) contains the saved
effective Hamiltonian, certified projection and symmetry artifacts, Q vectors,
and small native reference tables. Unpack it into a fresh case directory.

```bash
CASE=examples/cai2_ab_3.89
unzip -n "$CASE/kp/projected_input_M1.zip" -d "$CASE"
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  kp model -c "$CASE/kp/kp_M1.yaml"
```

## OpenMX inputs

- [openmx_nsoc.dat](openmx/openmx_nsoc.dat) is the original NSOC SCF deck.
- [openmx_soc.dat](openmx/openmx_soc.dat) reads the NSOC restart and runs one SOC iteration on the same relaxed coordinates.
- [POSCAR_relaxed](openmx/POSCAR_relaxed) records the relaxed coordinates.
- [POSCAR_rigid](openmx/POSCAR_rigid) and [openmx_rigid.dat](openmx/openmx_rigid.dat) record the TAPW reference geometry.

Follow the [OpenMX guide](../../scripts/openmx/README.md#paper-examples)
to install OpenMX and DFT_DATA19 and build the matching matrix reader.
Set the variables described there, then run from the repository root:

```bash
CASE=examples/cai2_ab_3.89
ln -s "$OPENMX_DATA" "$CASE/openmx/DFT_DATA19"
(cd "$CASE/openmx" && mpirun -np "$NPROCS" "$OPENMX" openmx_nsoc.dat -nt 1 > nsoc.stdout.log)
```

Verify NSOC convergence and retain the converged `openmx_rst/` restart.
The archived NSOC and SOC decks use the same system name and directory.
The SOC run writes `openmx.scfout`.

```bash
test -f "$CASE/openmx/openmx_rst/openmx.crst_check"
(cd "$CASE/openmx" && mpirun -np "$NPROCS" "$OPENMX" openmx_soc.dat -nt 1 > soc.stdout.log)
tapw prepare-hs openmx --input "$CASE/openmx/openmx.scfout" \
  --structure "$CASE/openmx/openmx_soc.dat" --binary "$OPENMX_READER" \
  --output "$CASE/openmx/prepared" --format npz --threads 1
```

Raw SOC H/S are imported in OpenMX source order. TAPW permutes them by the
rigid input atom order and then uses the existing threefold M-valley
averaging with the archived C3 convention. The M1 solver targets
-2.78658122 eV, 20 meV below the stored minimum conduction edge.
The [wrapper](tapw/run_m_band.py) selects the package's existing M-triplet
averaging routine.

```bash
TAPW_CFG="$CASE/tapw/tapw_M1.yaml"
KP_CFG="$CASE/kp/kp_M1.yaml"
python "$CASE/tapw/run_m_band.py" -c "$TAPW_CFG"
tapw symm -c "$TAPW_CFG"
kp project -c "$KP_CFG"
kp symm    -c "$KP_CFG"
kp model   -c "$KP_CFG"
```

Generated H/S, the large TAPW projected matrices, and the regenerated
model are written locally under `openmx/prepared/` and `output/`.

## Generate the Gamma TAPW inputs from OpenMX

After the SOC run described above, set `CASE=examples/cai2_ab_3.89`.
The Gamma calculation used **rigid-reference** space-group operations
(`symprec=0.001` Å, six unitary operations) to average the SOC Hamiltonian
and overlap matrices, with an element cutoff of `1e-7`. Use separate files
for these averaged matrices and the raw M-valley input.

```bash
CASE=examples/cai2_ab_3.89
mkdir -p "$CASE/output/gamma_symmetry" "$CASE/openmx/prepared"
cp "$CASE/openmx/openmx_rigid.dat" "$CASE/output/gamma_symmetry/openmx.dat_rigid"
python -m tapw.io.hs_symmetry --input-dir "$CASE/output/gamma_symmetry" \
  --output-dir "$CASE/output/gamma_symmetry/analysis" --dry-run --symprec 0.001 \
  --export-transports-json "$CASE/output/gamma_symmetry/transports.json"
"$OPENMX_READER" "$CASE/openmx/openmx.scfout" \
  --symmetry-json "$CASE/output/gamma_symmetry/transports.json" \
  --output-dir "$CASE/output/gamma_symmetry/averaged" --output-format npz \
  --threads 1 --cutoff 1e-7 --diagnostics full
cp "$CASE/output/gamma_symmetry/averaged/H.npz" "$CASE/openmx/prepared/H_symm.npz"
cp "$CASE/output/gamma_symmetry/averaged/S.npz" "$CASE/openmx/prepared/S_symm.npz"
tapw run -c "$CASE/tapw/tapw_Gamma.yaml"
```

The [Gamma TAPW YAML](tapw/tapw_Gamma.yaml) uses shell 6 (37 Q vectors) and a 61-point path.
To regenerate the training targets using the original molecular frame:

```bash
python "$CASE/kp/prepare_gamma_targets.py" "$CASE/output/tapw/Gamma/q06/band" "$CASE/output/targets_Gamma.npz"
python "$CASE/kp/refine_Gamma.py" --input "$CASE/output/targets_Gamma.npz" --output "$CASE/output/refitted_from_TAPW"
```

## 61-point TAPW validation

The frozen coefficients were evaluated on the full Gamma–M–K–Gamma path. A single constant aligns each model edge with TAPW at Gamma. The table uses all 61 points.

| Valley | RMS (meV) | Maximum (meV) |
| --- | ---: | ---: |
| Gamma | 0.269 | 0.634 |
| M1 | 0.034 | 0.083 |

The `model/` directories contain the corresponding `reference_bands.npz` and `native_overlaps.npz`. Squared overlaps are computed in orthonormal TAPW space, with a common 3 meV grouping threshold for nearby matched bands. Run `python examples/additional_models/verify_models.py --case cai2_ab_3.89` from the repository root to replay the archived comparisons.
