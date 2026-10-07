# ABACUS LCAO input

ABACUS LCAO calculations export both the Hamiltonian **H(R)** and the overlap
**S(R)**. Enable real-space CSR output (`out_mat_hs2 1`, with `gamma_only 0` in
the versions used by the examples). H is in Rydberg, S is dimensionless. See
[ABACUS matrix output documentation](https://abacus.deepmodeling.com/en/v3.11.0-beta8/advanced/elec_properties/hs_matrix.html).

## Prepare H/S

With the package installed, use the same entry point as the other backends:

```bash
tapw prepare-hs abacus \
  --input /absolute/path/to/abacus/case \
  --output /absolute/path/to/new/canonical \
  --format both
```

`python scripts/abacus/prepare_hs.py` accepts the same options for
compatibility.

The equivalent import-only command is:

```bash
python -m tapw.io.abacus /absolute/path/to/abacus/case \
  --output /absolute/path/to/new/canonical --write-dat
```

The source case must contain its original `INPUT` and `STRU`, its `OUT.<suffix>`
CSR files, and the numerical orbital files named in `STRU`. `orbital_dir` and
`stru_file` in `INPUT` are honored, including absolute paths. The destination
must not exist. The shared preparation command also exposes explicit symmetry
averaging; import alone does not average either matrix.

The output contains `H.npz`, `S.npz`, `structure.extxyz`,
`system.fragment.yaml`, and `abacus_import.json`; `--write-dat` also writes
seven-column `H.dat`/`S.dat`. NPZ files retain the exact basis dimension even
when edge orbitals have no nonzero Hamiltonian entries. The YAML is a system
input fragment. Actual moiré layer and twist information must be supplied
before a TAPW workflow can run.

## Basis and spin conventions

The orbital files' explicit `Element`, `Lmax`, and shell counts establish the
basis; filenames are not used to infer it. Site order is the original STRU
species/atom order. If a calculation log exists, its species, atom counts,
angular shell counts, and NLOCAL are checked against the supplied files.
Ambiguous species labels, missing metadata, inconsistent dimensions, and
angular momenta above f are rejected.

ABACUS source order is atom, increasing l, radial function (zeta), then real
harmonics in `m = 0, +1, -1, +2, -2, ...` order. Its real harmonics contain the
Condon–Shortley `(-1)^|m|` sign. Canonical conversion changes **both** H and S
by the same real unitary permutation and phase matrix.

| Shell | ABACUS source functions | TAPW output functions |
|---|---|---|
| s | s | s |
| p | pz, −px, −py | px, py, pz |
| d | dz², −dxz, −dyz, dx²−y², dxy | dz², dx²−y², dxy, dxz, dyz |
| f | fz³, −fxz², −fyz², fz(x²−y²), fxyz, −fx(x²−3y²), −fy(3x²−y²) | same order, positive Cartesian signs |

The f labels denote normalized real harmonic polynomials, including their
usual trace subtractions. Independent tests evaluate associated Legendre
functions and the normalized positive Cartesian polynomials at a general
direction for every s/p/d/f function.

The source audit used `Atom::set_index` in
[`source/source_cell/atom_spec.cpp`](https://github.com/deepmodeling/abacus-develop/blob/develop/source/source_cell/atom_spec.cpp)
and `YlmReal::rlylm` / `Ylm_Real` in
[`source/source_base/math_ylmreal.cpp`](https://github.com/deepmodeling/abacus-develop/blob/develop/source/source_base/math_ylmreal.cpp).
The audited local snapshots have SHA-256:

- `atom_spec.cpp`: `45792693556e290f29343230d63b4bd92e866e31a47b5cee49009b8b25574df3`
- `math_ylmreal.cpp`: `7a3200dca918a30693fd18b16f0bf0822de3709e55c3343528c8b7a6088613dd`

| ABACUS spin | Canonical representation |
|---|---|
| nspin = 1 | N × N scalar H and S |
| nspin = 2 | Two H channels combined into a 2N × 2N matrix; S duplicated; all ↑ orbitals followed by all ↓ orbitals |
| nspin = 4 | Full complex 2N × 2N H and S; source interleaved spinors permuted to all ↑ followed by all ↓ |

H converts once using `1 Ry = 13.605693122994 eV`. No Fermi subtraction or
energy alignment is applied. Absolute energy zeros and numerical basis sets
can differ between DFT codes; conversion accuracy does not mean their matrix
entries or unaligned eigenvalues should be identical.

## Supported files and validation

The separate CSR utility accepts the legacy
`data-HR-sparse_SPIN0.csr` / `data-SR-sparse_SPIN0.csr` family and the current
`hrs1_nao.csr` / `srs1_nao.csr` family, with the second H channel where required.
It selects a complete common ionic step, validates CSR indices and finite
values, and retains zero-cutoff sparse entries. H, S, and collinear channels
may have different sets of R vectors; omitted blocks are zero. In particular,
the supplied actual Si run has 55 H translations and 43 S translations.
Duplicate R blocks within one source remain errors.

The importer checks real-space Hermiticity, retains source hashes and exact
orbital maps, and publishes output only after successful conversion. It
supports fixed-geometry SCF/NSCF cases with explicit lattice vectors and
Direct, Cartesian, Cartesian_angstrom, or Cartesian_au coordinates. Original
input files must match the exported matrices. Relaxation/MD import, custom
species aliases, implicit lattice definitions, ABACUS binary/NPZ matrix
formats, and a standalone CSR directory without orbital/structure metadata
are currently unsupported. Matrix availability alone does not establish SCF
convergence; inspect each calculation's log and validation report.

`scripts/abacus/_vendor/analysis_abacus.py` is a separately licensed GPL-3.0-only
companion utility, executed as a subprocess. Its license and local modification
notice are kept beside it. The package does not import this utility as a Python
module. Its original snapshot SHA-256 is
`0a2b68bfd78505dcc81fc99ae9565448c4f534b9fb7cbcb4c935a8066a82fadb`.
The local change supports independently sparse translation sets, which was
necessary for the actual Si output above.

## Examples

The common-geometry examples are under `examples/dft_backends/si/abacus` and
`examples/dft_backends/mote2_9.43/abacus`. Their individual reports distinguish
native DFT convergence, matrix conversion, and the subsequent TAPW calculation.

For the default latest-step import, `source_fermi_ev` records the last
`E_Fermi` row from the calculation log after checking its Ry/eV columns. This
is metadata only: H is not shifted. The field is null if the log has no such
row or an explicit historical `--step` was requested, since the final SCF log
Fermi energy cannot safely be assigned to an earlier matrix snapshot.
Intermediate and final matrix text exports use 17 significant digits to avoid
rounding away small overlap or Hamiltonian entries.

The audited ABACUS version prints native energies using the historical
`Ry_to_eV = 13.605698` constant. Canonical H uses the stated
`13.605693122994` conversion. To keep `H - Ef*S` consistent,
`source_fermi_ev` is computed from the log's **Ry column with the canonical H
conversion**. The original values are retained as `source_fermi_ry` and
`source_fermi_reported_ev`, alongside `source_native_ry_to_ev = 13.605698`
and its source audit note. Native eV eigenvalue checks should report both the
as-printed difference and the difference after the known unit normalization
`13.605693122994 / 13.605698`. No fitted shift or scaling is applied.

## Source geometry and lattice gauge

STRU lengths use the audited **ABACUS-native** `BOHR_TO_A = 0.5291770`,
including `LATTICE_CONSTANT` and `Cartesian_au`. This intentionally reproduces
the actual geometry used by ABACUS. ASE's modern Bohr constant differs slightly
and must not be used to attach a different cell to the exported matrices. The
manifest records `source_native_bohr_to_angstrom`. To run exactly the same
Angstrom cell in multiple codes, prepare ABACUS's lattice constant with this
native conversion before the DFT calculation.

ABACUS wraps source fractional coordinates with `fmod(x + 10000, 1)` before
assembling real-space matrices. The importer preserves the original unwrapped
STRU geometry and computes each site's integer shift
`n_i = round(original_fractional_i - source_wrapped_fractional_i)`. This equals
`floor(original_fractional_i)` away from floating-point boundary cases. Each
matrix entry is reindexed as **R_canonical = R_ABACUS + n_i − n_j** in both H
and S, including all spin blocks. Site metadata retains both fractional
coordinates and `source_wrap_shift`; the manifest records the gauge convention.

The source operation is in
[`source/source_cell/read_stru.cpp`](https://github.com/deepmodeling/abacus-develop/blob/develop/source/source_cell/read_stru.cpp),
and Cartesian conversion is in
[`source/source_cell/read_atoms_helper.cpp`](https://github.com/deepmodeling/abacus-develop/blob/develop/source/source_cell/read_atoms_helper.cpp).
Independent regressions compare full orbital-position Bloch matrices at a
non-Gamma k point for negative positions, exact cell edges, positions beyond
one cell, and tiny negative roundoff values, in all three spin modes. A separate
Angstrom boundary regression checks the native Bohr constant. Gamma eigenvalues
alone cannot detect a missing site-gauge transformation.

## TAPW atom order metadata

New canonical H/S NPZ exports explicitly carry `basis_order: source`. This
means angular functions already use the canonical Cartesian convention while
atom order still follows STRU. TAPW applies its atom-type permutation when
loading both marked matrices. Previously unmarked NPZ exports retain their
historical loading behavior and should be reimported into a fresh directory
before using them as TAPW inputs. Full-precision DAT follows the same permutation
through the text loader; DAT cache files belong in a run-owned input directory.

The MoTe2 example's `run_basis_order_validation.py` creates a fresh marked
export, runs q_shell=3 from DAT and marked NPZ independently using the same
paired rigid geometry, and compares every projected eigenvalue at all 16
Γ–M–K–Γ points. Each run independently checks its saved Hamiltonians and
exported bands. See `basis_order_validation.json` for the comparison and native
Gamma gap. This is a finite-cutoff regression, not a cutoff-convergence claim.
