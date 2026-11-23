# AB bilayer CaI2 at 2.28 degrees

| Valley | TAPW configuration | Model configuration | Model and validation data |
|---|---|---|---|
| Gamma | [tapw_Gamma.yaml](tapw/tapw_Gamma.yaml) | [kp_Gamma.yaml](kp/kp_Gamma.yaml) | [model/Gamma/](model/Gamma/) |
| M1 | [tapw_M1.yaml](tapw/tapw_M1.yaml) | [kp_M1.yaml](kp/kp_M1.yaml) | [model/M1/](model/M1/) |

The M1 conduction model uses a 3786-atom fixed-cell MLFF structure with
the ST OpenMX/SOC basis. Each layer retains one Kramers pair, giving dimension
96 with TAPW Q shell 8. The 937 real coefficients
use kinetic/intralayer/interlayer orders 6/5/5 and 2/5 harmonic counts.
The lowest six bands compared with TAPW give **0.617 meV RMS** and
**1.265 meV maximum** on 20 held-out momenta, after a single
Gamma-edge alignment. The two-sided and one-sided matrix weights, target
window, and sampling indices are in [the KP YAML](kp/kp_M1.yaml).

## Gamma valence model

The [Gamma model](model/Gamma/) retains six spinful molecular states that span
both layers, at each of 31 Q vectors. The Hamiltonian has dimension 186. The
393 real coefficients are extracted linearly with kinetic order 4, coupling
order 1, and six harmonic shells including zero transfer. The one-sided and
two-sided matrix weights are 10 and 100 on the six highest valence states.
[The configuration](kp/kp_Gamma.yaml) specifies the 15 sampling momenta.
On the 61-point path, the six-band RMS / maximum energy errors relative to
TAPW are **0.039 / 0.123 meV**. The mean / minimum squared overlaps are
**0.9985 / 0.9950**, using the normalized molecular-state resolvent embedding.

To repeat the linear extraction, run from the repository root:

```bash
CASE=examples/cai2_ab_2.28
unzip -n "$CASE/kp/input_Gamma.zip" -d "$CASE"
kp project -c "$CASE/kp/kp_Gamma.yaml"
kp symm    -c "$CASE/kp/kp_Gamma.yaml"
kp model   -c "$CASE/kp/kp_Gamma.yaml"
```

The archive starts from the molecular effective Hamiltonian obtained from
shell-5 TAPW matrices with symmetrized real-space H and S. Its six physical
components are packed in the active source group. The second group contains
uncoupled padding at +10 eV for the two-group input interface. The model
retains only the six molecular components. The source representations for
C3z, C2 and time reversal are included in the archive.

The [TAPW configuration](tapw/tapw_Gamma.yaml) expects symmetrized
`openmx/prepared/H_symm.npz` and `S_symm.npz` and uses the supplied rigid
OpenMX cell. Given that TAPW band output and the unpacked archive, the
[molecular projection script](kp/prepare_gamma_input.py) reconstructs the
matrix and reference spectrum from the [fixed molecular frame](kp/molecular_frame_Gamma.npz):

```bash
python "$CASE/kp/prepare_gamma_input.py" "$CASE/output/tapw/Gamma/q05/band" "$CASE/input/rebuilt_Gamma"
```

## Evaluate the continuum model

The [standalone M1 model](model/M1/) contains the original extracted
coefficients, numerical eigensolver and default Q-vector realization. From
the checkout root:

```bash
python examples/additional_models/verify_models.py --case cai2_ab_2.28
```

[M-valley spectra, orbital weights, and rotation audit](reference/M1/)
include the M2 and M3 spectra after rotating the local wavevector of M1 by
the corresponding C3 operation. The three valleys use one common energy shift.

## Reproduce the M1 coefficient extraction

The [projected input archive](kp/input_M1.zip) contains the saved
effective Hamiltonian, certified projection and symmetry artifacts, Q vectors,
and small native reference tables. Unpack it into a fresh case directory.

```bash
CASE=examples/cai2_ab_2.28
unzip -n "$CASE/kp/input_M1.zip" -d "$CASE"
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
CASE=examples/cai2_ab_2.28
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
-2.79553427 eV, 20 meV below the stored minimum conduction edge.
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

After the SOC run described above, set `CASE=examples/cai2_ab_2.28`.
The Gamma calculation used **rigid-reference** space-group operations
(`symprec=0.001` Å, six unitary operations) to average the SOC Hamiltonian
and overlap matrices, with an element cutoff of `1e-7`. Use separate files
for these averaged matrices and the raw M-valley input.

```bash
CASE=examples/cai2_ab_2.28
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

The [Gamma TAPW YAML](tapw/tapw_Gamma.yaml) uses shell 5 (31 Q vectors) and a 61-point path.
The molecular projection command above regenerates `hamiltonian_k.npy` and
`energies_vbm.txt`. To use these regenerated files in a fresh unpacked example,
copy them from `input/rebuilt_Gamma/` into `input/Gamma/` before running
the three KP stages. The supplied symmetry representations and state order
belong to the fixed molecular frame.

## 61-point TAPW validation

The frozen coefficients were evaluated on the full Gamma–M–K–Gamma path. A single constant aligns each model edge with TAPW at Gamma. The table uses all 61 points.

| Valley | RMS (meV) | Maximum (meV) |
| --- | ---: | ---: |
| Gamma | 0.039 | 0.123 |
| M1 | 0.627 | 1.274 |

The `model/` directories contain the corresponding `reference_bands.npz` validation archives. Squared overlaps are computed in orthonormal TAPW space, with a common 3 meV grouping threshold for nearby matched bands. Run `python examples/additional_models/verify_models.py --case cai2_ab_2.28` from the repository root to replay the archived comparisons.
