import numpy as np

import tapw.symmetry_analysis as symmetry_analysis


def test_frobenius_relative_residual_is_zero_for_identical_matrices():
    helper = getattr(symmetry_analysis, "frobenius_relative_residual", None)
    assert helper is not None

    matrix = np.array([[2.0, 1.0j], [-1.0j, 3.0]], dtype=np.complex128)

    residual = helper(matrix, matrix)

    assert residual == 0.0


def test_unitary_symmetrization_improves_a_broken_covariance_residual():
    helper = getattr(symmetry_analysis, "frobenius_relative_residual", None)
    symmetrize = getattr(symmetry_analysis, "symmetrize_unitary_orbit", None)
    assert helper is not None
    assert symmetrize is not None

    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    h0 = np.array([[2.0, 0.25], [0.25, 1.0]], dtype=np.complex128)
    h1_exact = u.conj().T @ h0 @ u
    h1_broken = h1_exact + np.array([[0.15, 0.0], [0.0, -0.05]], dtype=np.complex128)

    raw = helper(h0, u @ h1_broken @ u.conj().T)
    h_sym = symmetrize([h0, h1_broken], [np.eye(2, dtype=np.complex128), u])
    h_sym_partner = symmetrize([h1_broken, h0], [np.eye(2, dtype=np.complex128), u])
    improved = helper(h_sym, u @ h_sym_partner @ u.conj().T)

    assert raw > 0.0
    assert improved < raw


def test_lowdin_order_residual_is_small_for_exact_symmetric_orbit():
    helper = getattr(symmetry_analysis, "compute_lowdin_order_residual_unitary", None)
    assert helper is not None

    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    h0 = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.complex128)
    s0 = np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.complex128)
    h1 = u.conj().T @ h0 @ u
    s1 = u.conj().T @ s0 @ u

    result = helper(
        [h0, h1],
        [s0, s1],
        [np.eye(2, dtype=np.complex128), u],
    )

    assert result["supported"] is True
    assert result["not_supported_reason"] == ""
    assert result["residual"] < 1.0e-10


def test_lowdin_order_rejects_non_positive_definite_overlap_without_clipping():
    helper = getattr(symmetry_analysis, "compute_lowdin_order_residual_unitary", None)
    assert helper is not None

    u = np.eye(2, dtype=np.complex128)
    h0 = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.complex128)
    s0 = np.array([[1.0, 0.0], [0.0, -0.1]], dtype=np.complex128)

    result = helper(
        [h0],
        [s0],
        [u],
    )

    assert result["supported"] is False
    assert result["not_supported_reason"] == "S_not_positive_definite"
    assert result["residual"] is None
