import numpy as np
import scipy.linalg

from tapw.workflows.band import BandStructureCalculator


def _reference_transform(hamk: np.ndarray, samk: np.ndarray) -> np.ndarray:
    s_eig, s_vec = scipy.linalg.eigh(samk, check_finite=False)
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s_eig))
    uminvud = s_vec @ s_inv_sqrt @ s_vec.conj().T
    return uminvud @ hamk @ uminvud


def test_gen_h_new_cpu_matches_reference_for_complex_generalized_problem():
    rng = np.random.default_rng(20260315)

    raw_h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hamk = raw_h + raw_h.conj().T

    raw_s = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    samk = raw_s @ raw_s.conj().T
    samk += 2.0 * np.eye(4)

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    expected = np.linalg.eigvalsh(_reference_transform(hamk, samk))
    actual = np.linalg.eigvalsh(calculator.gen_H_new_cpu(hamk, samk))

    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_gen_h_new_cpu_matches_legacy_reference_matrix_for_complex_generalized_problem():
    rng = np.random.default_rng(20260315)

    raw_h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hamk = raw_h + raw_h.conj().T

    raw_s = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    samk = raw_s @ raw_s.conj().T
    samk += 2.0 * np.eye(4)

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    expected = _reference_transform(hamk, samk)
    actual = calculator.gen_H_new_cpu(hamk, samk)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_gen_h_new_cpu_does_not_force_extra_hermitian_symmetrization():
    rng = np.random.default_rng(1)

    raw_h = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    hamk = raw_h + raw_h.conj().T

    # Deliberately ill-conditioned SPD overlap so the legacy transform keeps a
    # measurable anti-Hermitian residue instead of being symmetrized away.
    q_mat, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    samk = q_mat @ np.diag(np.geomspace(1.0e-10, 1.0e2, 8)) @ q_mat.conj().T

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    actual = calculator.gen_H_new_cpu(hamk, samk)
    hermitian_part = (actual + actual.conj().T) / 2

    assert np.max(np.abs(actual - hermitian_part)) > 1.0e-8
