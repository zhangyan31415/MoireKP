"""Independent small-matrix and angular-function oracles using actual sisl.

These are synthetic interoperability tests, not production SIESTA calculations.
The optional dependency is skipped in the ordinary lightweight environment.
"""
import json

import numpy as np
import pytest
from scipy import linalg, sparse

sisl = pytest.importorskip("sisl")

from tapw.io.hr import HrSparseHandler, SPARSE_NPZ_METADATA_KEY
from tapw.io.siesta import convert_hamiltonian, import_siesta, orbital_layout


SPINS = ("unpolarized", "polarized", "non-collinear", "spin-orbit")
KPOINTS = ((0., 0., 0.), (.173, -.219, .071), (-.31, .127, -.083))


def _hamiltonian(spin, *, tag="Si"):
    # Deliberately noncanonical p,s shell order; two sites also detect an
    # accidental species/site sort and global-versus-local spin permutation.
    orbitals = [sisl.AtomicOrbital(n=3, l=1, m=m, zeta=1, R=2.)
                for m in (-1, 0, 1)]
    orbitals.append(sisl.AtomicOrbital(n=3, l=0, m=0, zeta=1, R=2.))
    atom = sisl.Atom(14, orbitals=orbitals, tag=tag)
    geometry = sisl.Geometry(
        [[.17, .29, .41], [1.37, .88, 1.22]], [atom, atom],
        lattice=sisl.Lattice([[4., .2, .1], [.3, 4.7, .4], [.1, .2, 5.3]],
                            nsc=[3, 3, 3]),
    )
    h = sisl.Hamiltonian(geometry, spin=spin, orthogonal=False, dtype=np.float64)
    rng = np.random.default_rng(19481)
    components = h.spin.size(np.float64)
    for r in ((0, 0, 0), (1, 0, 0), (0, -1, 0), (1, -1, 1)):
        for i in range(geometry.no):
            for j in range(geometry.no):
                data = rng.normal(scale=.13, size=components + 1)
                data[-1] *= .015
                if r == (0, 0, 0) and i == j:
                    data[-1] += 1.3
                h[i, j, r] = data
    # Public sisl conjugate transpose is the independent spin-component oracle.
    h = (h + h.transpose(conjugate=True, spin=True)) / 2
    h.finalize()
    return h


def _bloch(blocks, dimension, k):
    result = np.zeros((dimension, dimension), complex)
    for r, block in blocks.items():
        matrix = sparse.coo_matrix(
            (block["val"], (block["row"], block["col"])),
            shape=(dimension, dimension),
        ).toarray()
        result += matrix * np.exp(2j * np.pi * np.dot(k, r))
    return result


def _transform(matrix, spin):
    # Oracle explicitly independent of importer metadata/layout helpers.
    permutation = np.array([3, 2, 0, 1, 7, 6, 4, 5])
    phases = np.array([1., -1., -1., 1.] * 2)
    if spin in ("non-collinear", "spin-orbit"):
        permutation = np.r_[2 * permutation, 2 * permutation + 1]
        phases = np.tile(phases, 2)
    return matrix[np.ix_(permutation, permutation)] * phases[:, None] * phases[None, :]


def _reference(h, k, spin, fermi_shift):
    kwargs = dict(k=k, gauge="lattice", format="array", dtype=np.complex128)
    overlap = np.asarray(h.Sk(**kwargs))
    if spin == "polarized":
        hs = [_transform(np.asarray(h.Hk(spin=s, **kwargs)) + fermi_shift * overlap,
                         "unpolarized") for s in (0, 1)]
        ss = _transform(overlap, "unpolarized")
        return linalg.block_diag(*hs), linalg.block_diag(ss, ss)
    return (_transform(np.asarray(h.Hk(**kwargs)) + fermi_shift * overlap, spin),
            _transform(overlap, spin))


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("reference", ["absolute", "fermi"])
def test_actual_sisl_full_bloch_matrices_and_energy_reference(spin, reference):
    h = _hamiltonian(spin)
    before = [np.asarray(h.Hk(k=k, format="array", dtype=np.complex128)).copy()
              for k in KPOINTS]
    ef = 2.137
    hb, sb, metadata = convert_hamiltonian(h, fermi_ev=ef, energy_reference=reference)
    dimension = 8 if spin == "unpolarized" else 16
    assert metadata["basis_dimension"] == dimension
    assert metadata["system_orbitals"] == {"Si": "s1p1"}
    assert len(hb) >= 7
    for k, original in zip(KPOINTS, before):
        actual_h, actual_s = _bloch(hb, dimension, k), _bloch(sb, dimension, k)
        expect_h, expect_s = _reference(h, k, spin, ef if reference == "absolute" else 0.)
        np.testing.assert_allclose(actual_h, expect_h, atol=2e-13)
        np.testing.assert_allclose(actual_s, expect_s, atol=2e-13)
        np.testing.assert_allclose(actual_h, actual_h.conj().T, atol=2e-13)
        assert linalg.eigvalsh(actual_s).min() > 1.
        np.testing.assert_allclose(linalg.eigh(actual_h, actual_s, eigvals_only=True),
                                   linalg.eigh(expect_h, expect_s, eigvals_only=True), atol=2e-13)
        np.testing.assert_array_equal(h.Hk(k=k, format="array", dtype=np.complex128), original)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_sisl_angular_functions_match_positive_tapw_polynomials(l):
    from tapw.geometry import rotations

    polynomial_sets = {
        1: [rotations.px, rotations.py, rotations.pz],
        2: [rotations.dz2, rotations.dx2_y2, rotations.dxy, rotations.dxz, rotations.dyz],
        3: [rotations.fz3, rotations.fxz2, rotations.fyz2, rotations.fzx2_zy2,
            rotations.fxyz, rotations.fx3_3xy2, rotations.f3yx2_y3],
    }
    points = np.random.default_rng(1591).normal(size=(23, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    spherical = sisl.SphericalOrbital(l, lambda r: np.ones_like(r), R=4.)
    source = spherical.toAtomicOrbital(n=l + 1, zeta=1)
    quantum_numbers = [{key: getattr(orb, key) for key in ("n", "l", "m", "zeta")}
                      for orb in source]
    layout = orbital_layout(quantum_numbers)
    for index, polynomial in enumerate(polynomial_sets[l]):
        expected = np.array([float(polynomial(*point)) for point in points])
        actual = layout.phases[index] * source[layout.permutation[index]].psi(points)
        # Normalization varies with l; positive proportionality fixes signs and
        # full pointwise proportionality fixes the actual harmonic, not its name.
        factor = np.dot(expected, actual) / np.dot(expected, expected)
        assert factor > 0
        np.testing.assert_allclose(actual, factor * expected, atol=2e-13)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("spin", SPINS)
def test_actual_hsx_roundtrip_and_exported_npz(version, spin, tmp_path):
    h = _hamiltonian(spin)
    source = tmp_path / "fixture.HSX"
    h.write(str(source), version=version, dtype=np.float64)
    reader = sisl.get_sile(str(source))
    assert reader.version == version
    ef = reader.read_fermi_level()
    assert np.isfinite(ef)
    # sisl 0.16's public HSX writer stores Ef=0. Nonzero Ef is independently
    # checked above; do not claim this fixture tests a nonzero binary Ef field.
    output = tmp_path / "converted"
    metadata = import_siesta(source, output, write_dat=True)
    hb, sb = [], []
    for name, target in (("H", hb), ("S", sb)):
        # Canonical orbital harmonics retain source site order. TAPW must
        # apply its structure-dependent atom-type permutation on loading.
        with np.load(output / f"{name}.npz", allow_pickle=False) as archive:
            stored = json.loads(str(archive[SPARSE_NPZ_METADATA_KEY].item()))
            assert stored["basis_order"] == "source"
        handler = HrSparseHandler()
        handler.load_from_npz(str(output / f"{name}.npz"))
        target.append(handler.hr_sparse)
        assert handler.basis_dimension == metadata["basis_dimension"]
        assert (output / f"{name}.dat").is_file()
    for k in KPOINTS:
        expect_h, expect_s = _reference(h, k, spin, 0.)
        np.testing.assert_allclose(_bloch(hb[0], metadata["basis_dimension"], k), expect_h, atol=3e-12)
        np.testing.assert_allclose(_bloch(sb[0], metadata["basis_dimension"], k), expect_s, atol=3e-12)
    saved = json.loads((output / "siesta_import.json").read_text())
    assert saved["hsx_version"] == version
    assert saved["source_fermi_ev"] == ef
    with pytest.raises(FileExistsError):
        import_siesta(source, output)


def test_actual_custom_species_tag_requires_explicit_consistent_mapping():
    h = _hamiltonian("unpolarized", tag="Si_custom")
    with pytest.raises(ValueError, match="species"):
        convert_hamiltonian(h, fermi_ev=0.)
    _, _, metadata = convert_hamiltonian(h, fermi_ev=0., species_map={"Si_custom": "Si"})
    assert metadata["system_orbitals"] == {"Si": "s1p1"}
    with pytest.raises(ValueError):
        convert_hamiltonian(h, fermi_ev=0., species_map={"Si_custom": "C"})


def test_distinct_source_species_cannot_collapse_to_one_canonical_radial_basis():
    orbitals = [sisl.AtomicOrbital(n=3, l=0, m=0, zeta=1, R=2.)]
    atoms = [sisl.Atom(14, orbitals=orbitals, tag=tag) for tag in ('Si_a', 'Si_b')]
    geom = sisl.Geometry([[0., 0., 0.], [1., 1., 1.]], atoms,
                         lattice=sisl.Lattice([4., 4., 4.]))
    h = sisl.Hamiltonian(geom, orthogonal=False)
    h[0, 0] = [1., 1.]
    h[1, 1] = [2., 1.]
    with pytest.raises(ValueError, match='species|radial'):
        convert_hamiltonian(h, fermi_ev=0., species_map={'Si_a': 'Si', 'Si_b': 'Si'})


def test_actual_sisl_incomplete_angular_shell_is_rejected():
    orbitals = [sisl.AtomicOrbital(n=3, l=1, m=m, zeta=1, R=2.) for m in (-1, 0)]
    geometry = sisl.Geometry([[0., 0., 0.]], sisl.Atom(14, orbitals=orbitals),
                             lattice=sisl.Lattice([4., 4., 4.]))
    h = sisl.Hamiltonian(geometry, orthogonal=False)
    for i in range(geometry.no):
        h[i, i] = [1., 1.]
    with pytest.raises(ValueError, match="Incomplete angular shell"):
        convert_hamiltonian(h, fermi_ev=0.)


def test_valid_zero_fermi_referenced_hamiltonian_is_not_rejected():
    """A one-level Hamiltonian can vanish after shifting its level to Ef."""
    from tapw.io.siesta import convert_hamiltonian
    atom = sisl.Atom('H', [sisl.AtomicOrbital(n=1, l=0, m=0, zeta=1, R=1.)])
    geometry = sisl.Geometry([[0., 0., 0.]], atom, lattice=sisl.Lattice(5.))
    h = sisl.Hamiltonian(geometry, orthogonal=False)
    h[0, 0] = [0., 1.]
    h.finalize()
    hblocks, sblocks, meta = convert_hamiltonian(h, fermi_ev=-3., energy_reference='fermi')
    assert hblocks == {}
    assert meta['basis_dimension'] == 1
    assert len(sblocks[(0, 0, 0)]['val']) == 1
