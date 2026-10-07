# SIESTA input

`tapw import-siesta` reads self-describing SIESTA HSX files (versions 1 and 2)
and writes the real-space matrices and basis metadata used by TAPW. The reader
uses the optional `sisl` dependency. Existing OpenMX input is unchanged.

## Installation and import

In a project environment, install the optional reader:

```sh
python -m pip install '.[siesta]'
tapw import-siesta siesta.HSX --output siesta_import
```

SIESTA itself is only needed to generate the HSX, not to import an existing file.
A tested combination is SIESTA 5.4.2, sisl 0.16.4 and Python 3.11.16. SIESTA
provides conda-forge binaries; use a separate environment for the executable.

Generate an image-resolved HSX with a completed SIESTA calculation using:

```text
SaveHS true
ForceAuxCell true
```

`ForceAuxCell` matters for Gamma-only runs: a Gamma-folded matrix does not
provide the individual real-space translations needed away from Gamma.

The common preparation interface also offers identical options across DFT
backends: `tapw prepare-hs siesta --input siesta.HSX --output siesta_import
--format both`. Add `--symmetrize` for a separate `symmetrized/` result while
retaining raw H/S. The output directory must not exist. See
[preparation tools](../scripts/README.md). Outputs are:

| File | Contents |
|---|---|
| `H.npz`, `S.npz` | Versioned TAPW sparse real-space matrices, with exact basis dimension |
| `structure.extxyz` | Cell and atom positions in Angstrom, original site order |
| `siesta_import.json` | Source hash, HSX/sisl versions, energy zero, orbital permutation/phases, spin layout and validation residuals |
| `system.fragment.yaml` | Structure/matrix/orbital/spin fields for canonical input |
| `README.md` | Import conventions and remaining configuration requirements |

Add `--write-dat` to also write `H.dat` and `S.dat` in the existing four-header,
seven-column sparse text format. NPZ and DAT use the same units and ordering.

The YAML is an **input fragment**, not a runnable moire model. The user must
supply the actual `output`, `twist_index`, `layers` and workflow configuration.
Do not assign arbitrary twist/layer parameters to a primitive-cell test to make
it appear to be a moire benchmark. Use the orbital metadata and real geometry
of the intended system.

## Energy and basis conventions

- H is in eV; S is dimensionless. No additional Hartree factor is applied.
- The default `--energy-reference absolute` restores the original HSX energy
  reference. sisl first returns `H_file - Ef*S`; the importer adds `Ef*S` once.
- `--energy-reference fermi` keeps the sisl Fermi-relative matrix. The original
  Fermi level and chosen zero are always recorded. This is a matrix shift by
  `Ef*S(R)`, not subtraction from only the diagonal R=0 entries.
- Cell vectors are rows, in Angstrom. R is taken from sisl's actual supercell
  map, and the lattice-gauge Bloch sum is `sum_R M(R) exp(+2*pi*i*k.R)`.
- Scalar orbital shells are sorted by l, then n and zeta, preserving atom order.
  The importer maps SIESTA real harmonics into TAPW's positive Cartesian order
  and applies the necessary `(-1)^abs(m)` signs to BOTH H and S.
- Exported NPZ metadata explicitly records `basis_order: source`: canonical
  orbital harmonics still follow the source atom order. The TAPW loader applies
  its structure-dependent atom-type permutation, as it does for DAT input.
  Regenerate older unmarked imports from the original HSX before running TAPW;
  changing filenames does not establish the matrix ordering.
- Collinear channels are combined into a block-diagonal spin Hamiltonian.
  Noncollinear and spin-orbit blocks retain their complex spin mixing. Spinful
  output is `[all scalar orbitals up, all scalar orbitals down]`.
- No symmetry averaging, entry threshold, orbital truncation, fit, or coordinate
  wrapping is applied by the importer.

For non-element species labels recognized by sisl, explicitly confirm their
identity, for example:

```sh
tapw import-siesta FePt.HSX --output fept_import \
  --species Fe_fept_SOC=Fe --species Pt_fept_SOC=Pt
```

An override must agree with the element identified by the reader. Unknown,
ghost or floating species are rejected, as are incomplete angular shells,
unsupported l>3 shells, inconsistent per-element orbital specifications,
Nambu input, unreadable Fermi levels and legacy HSX v0. These cases must not be
silently converted using guessed orbital conventions. HSX does not contain
radial wavefunctions; this command does not reconstruct physical real-space
wavefunctions or certify Berry/quantum-metric observables.

Distinct SIESTA species labels cannot map to the same canonical element:
equal shell quantum numbers do not establish equal radial functions or
pseudopotentials, which are needed before symmetry may exchange sites.

Inputs must be stable completed files. The importer checks source identity and
hash while reading. Output publication reserves a new directory exclusively;
the JSON completion metadata is published last. Existing files/directories are
not replaced. A successful import checks paired-R Hermiticity; overlap positive
definiteness over the user's intended k region is a separate validation task.

## Validation

The small integration cases under `examples/siesta_validation_20260922/` cover:

- primitive Si, nonmagnetic;
- H atom, collinear polarization;
- tilted H atom, noncollinear spin;
- FePt, offsite spin-orbit coupling with relativistic pseudopotentials.

All final SCF runs converged. Source HSX matrices were compared with imported
H/S and the existing TAPW orbital-gauge matrix assembly at Gamma and five generic
or boundary k points. Complete H/S, overlap positivity, energy-reference shifts
and generalized eigenvalues were checked. Synthetic tests additionally cover
HSX v1/v2, p/d/f angular-function conventions, nontrivial 3D translations and
all four supported spin modes. These checks validate the input path; they are
not new moire continuum-model or physical quantum-geometry benchmarks.

See the dated validation report for commands, node/environment provenance,
measured errors and the retained records of unsuccessful input trials.

Primary format sources: [SIESTA documentation](https://docs.siesta-project.org/projects/siesta/en/5.4/reference/siesta.html)
and [sisl HSX reader](https://sisl.readthedocs.io/en/latest/api/io/generated/sisl.io.siesta.hsxSileSiesta.html).
