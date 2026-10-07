# SIESTA HSX import convention audit

Research date: 2026-09-22. This is a source audit, not an execution or
production-validation report. The upstream links below refer to sisl `main`;
the installed dependency version must be recorded and tested separately.

## Supported input boundary

Use `sisl.get_sile(path)` and its public `version`, `read_hamiltonian()`,
`read_fermi_level()` methods. Accept only explicitly supported HSX versions
1 and 2 initially. Version 0 does not reliably contain a complete cell,
geometry or Fermi energy. Do not infer these from an arbitrary bounding box.
Reject missing/invalid atomic quantum numbers, incomplete angular shells,
unsupported angular momentum (`l > 3`), unknown spin modes and unresolved
species identities. Do not infer chemical elements from an arbitrary species
label without checking against the user's structure/species declaration:
sisl's HSX reader falls back to parsing the first one or two label characters.

HSX stores `(n,l,zeta)` for each scalar orbital. The reader reconstructs
`m=-l,...,+l` within each shell. Polarization flags are not stored, and HSX
does not provide radial wavefunctions. These omissions do not prevent
matrix conversion, but they preclude claiming reconstructed real-space
wavefunctions or an independently determined radial basis.

Source: [HSX reader](https://github.com/zerothi/sisl/blob/main/src/sisl/io/siesta/binaries.py),
methods `read_basis`, `_r_geometry_v1`, `_r_hamiltonian_v1`.

## Energy, geometry and Fourier convention

For HSX 1/2, sisl converts Hamiltonian elements from Ry to eV and performs
`H.shift(-Ef)` during `read_hamiltonian()`. Thus the returned matrix is
`H_file[eV] - Ef[eV] S`; do not subtract the Fermi energy again. The public
`read_fermi_level()` returns the original Ef in eV. If absolute file energies
are requested, restore them with public `H.shift(+Ef)` before extraction.
Missing-Fermi warnings (including `H.Setup.Only` files) must not be silently
described as a valid Fermi-referenced calculation.

Geometry coordinates and row-vector cell are already in Angstrom. The
reader already converts Fortran/CSC orientation, SIESTA supercell order and
spin-component signs; do not repeat these conversions manually.

Use `H.Hk(k=(0,0,0), gauge='lattice', format='sc:csr')` and corresponding
`H.Sk(...)` to obtain the uncontracted real-space blocks through public APIs.
At Gamma all Bloch phases are unity. For scalar orbitals, column `c` has
orbital `c % N` and cell index `c // N`; use the actual
`H.geometry.lattice.sc_off[cell_index]`, never a guessed Cartesian-product
ordering. In lattice gauge, folding uses
`sum_R M(R) exp(+2 pi i k_frac dot R)`.

Sources: [HSX reader](https://github.com/zerothi/sisl/blob/main/src/sisl/io/siesta/binaries.py),
[sparse matrix API](https://github.com/zerothi/sisl/blob/main/src/sisl/physics/sparse.py),
[k-space dispatch](https://github.com/zerothi/sisl/blob/main/src/sisl/physics/_matrix_k.pyx).

## Angular and radial ordering into TAPW

`tapw/tapw/io/structure.py` counts complete `s/p/d/f` shells and preserves
the supplied atom/orbital sequence until its atom-type permutation.
`tapw/tapw/geometry/rotations.py` defines positive Cartesian harmonics and
applies `U_p` and `U_d` to obtain the actual OpenMX order below. The labels
in `tapw/tapw/orbital_analysis_tool.py` independently agree with that order.

For each atom, retain atom order and group radial shells by `l=0,1,2,3`.
Within each `l`, a deterministic `(n,zeta)` order is suitable; preserve an
explicit source-to-target table because TAPW's radial shell numbers only
enumerate shells and do not encode principal quantum number. Each shell
must contain exactly one of every `m=-l,...,+l`. Repeated ambiguous shell
identities require an explicit occurrence identity or rejection, not a
dictionary overwrite. The compact `sNpNdNfN` specification counts radial
shells, not the largest principal quantum number or zeta index.

| shell | TAPW order | source m order to gather | phases |
|---|---|---|---|
| s | s | 0 | + |
| p | px, py, pz | +1, -1, 0 | -, -, + |
| d | dz2, dx2-y2, dxy, dxz, dyz | 0, +2, -2, +1, -1 | +, +, +, -, - |
| f | fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2) | 0, +1, -1, +2, -2, +3, -3 | +, -, -, +, +, -, - |

**The phases are necessary.** sisl's `_rspherical_harm` deliberately uses
the SIESTA convention, including the associated-Legendre Condon–Shortley
phase. Its named px and py angular functions are negative multiples of x
and y. Relative to TAPW's polynomials the sign is `(-1)**abs(m)` for both
positive and negative m. A name-only permutation produces the wrong
representation of spatial symmetry, even though energy eigenvalues remain
unchanged. For target-to-source permutation `p` and signs `d`, convert each
block as `M_target[i,j] = d[i] M_source[p[i],p[j]] d[j]`, for both H and S.
No amplitude scaling is needed: the normalization within each l shell
differs from the TAPW polynomial representation by a common factor.

Source: [orbital implementation](https://github.com/zerothi/sisl/blob/main/src/sisl/_core/orbital.py),
functions `_rfact`, `_rspherical_harm`, and `AtomicOrbital.name`.

## Spin layout

* Unpolarized: public Hk/Sk return `N x (N*ncell)`.
* Collinear polarized: `Hk(spin=0,...)` and `Hk(spin=1,...)` each return
  scalar blocks; Sk is scalar. Assemble each cell's Hamiltonian as
  `diag(Hup(R),Hdown(R))` and overlap as `diag(S(R),S(R))` for TAPW's
  `[all up, all down]` order. Do not block-diagonalize the entire rectangular
  supercell arrays without repairing the cell-column order.
* Noncollinear/SOC: public Hk and Sk already return
  `2N x (2N*ncell)` with orbital-interleaved spins. Both row and column
  indexing is `2*orbital + spin`; cell index is `column // (2N)`.
  Extract each cell block and gather rows/columns with
  `concatenate([2*p, 2*p+1])`, then apply `concatenate([d,d])` signs.
* Nambu: reject unless a distinct particle/hole basis contract is added.

The HSX reader already corrects the SIESTA versus sisl imaginary sign of
the upper off-diagonal spin component. Its returned matrices must not be
conjugated again. Reading private `_csr._D` bypasses useful public spin
assembly and should be avoided by the importer.

Sources: [spin conversion](https://github.com/zerothi/sisl/blob/main/src/sisl/io/siesta/_help.py),
[spin dispatch and overlap](https://github.com/zerothi/sisl/blob/main/src/sisl/physics/sparse.py),
[explicit interleaved indices](https://github.com/zerothi/sisl/blob/main/src/sisl/physics/_matrix_phase_sc.pyx).

## Required verification before support claims

1. A synthetic asymmetric hopping model with nonzero R, nonorthogonal S,
   multiple atoms and a non-Gamma k: fold exported H/S and compare complete
   transformed matrices with sisl Hk/Sk. Check `M(-R)=M(R).conj().T`.
2. A nonzero-Ef HSX round trip: compare absolute and Fermi-relative spectra
   and demonstrate exactly one `Ef*S` shift.
3. Angular functions or a known spatial rotation: verify p/d/f phase and
   ordering against TAPW's Cartesian polynomials. Spectra alone cannot
   detect an incorrect basis convention.
4. Unpolarized, unequal polarized channels, NC imaginary mixing and SOC
   complex diagonal/off-diagonal spin terms: compare full matrices and
   generalized eigenvalues, including more than one supercell block.
5. Multiple radial shells and species; deliberately malformed/incomplete
   shells and unsupported l must fail with actionable errors.
6. An actual SIESTA-generated HSX case with a recorded SIESTA version,
   sisl version, input, pseudopotential provenance and executable command.
   Synthetic fixtures alone establish conversion behavior, not successful
   integration with a production SIESTA run.
