from __future__ import annotations

import numpy as np
from scipy import sparse

from kp.model.response_basis import (
    CompiledResponseBasis,
    CompiledResponseRuntime,
    PolynomialCoordinateBasis,
    RawPolynomialSeed,
    compile_candidate_responses,
    identity_finite_group,
)


def _coordinate() -> PolynomialCoordinateBasis:
    return PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=[0.0, 0.0],
        reciprocal_basis=[[1.0, 0.0], [0.0, 1.0]],
        max_degree=1,
    )


def _identity_payload() -> dict[str, object]:
    return {
        "basis_layout": {"order": "toy", "dim": 1},
        "q_vectors": np.zeros((1, 2), dtype=np.float64),
        "basis_ordering": np.arange(1, dtype=np.int64),
        "n_orb": np.asarray([1, 0], dtype=np.int64),
        "exactified_matrices": {"identity": np.eye(1, dtype=np.complex128)},
        "k_pullbacks": {"identity": np.eye(2, dtype=np.float64)},
        "q_permutations": {"identity": np.arange(1, dtype=np.int64)},
        "sector_permutations": {"identity": np.asarray([0, 1], dtype=np.int64)},
        "dtype": "complex128",
        "compiler_version": "fit-frame-covariance-test-v1",
    }


def _basis(*, rotated_owner_frame: bool) -> CompiledResponseBasis:
    one = sparse.csr_matrix(np.ones((1, 1), dtype=np.complex128))
    constant = {(0, 0): one}
    linear = {(1, 0): one, (0, 1): one}
    if rotated_owner_frame:
        seeds = [
            RawPolynomialSeed(
                "constant-plus-linear",
                {**constant, **linear},
                "toy",
                {},
            ),
            RawPolynomialSeed(
                "constant-minus-linear",
                {
                    (0, 0): one,
                    (1, 0): -one,
                    (0, 1): -one,
                },
                "toy",
                {},
            ),
        ]
    else:
        seeds = [
            RawPolynomialSeed("constant", constant, "toy", {}),
            RawPolynomialSeed("linear", linear, "toy", {}),
        ]
    candidates = compile_candidate_responses(
        seeds,
        coordinate=_coordinate(),
        group=identity_finite_group(1),
    )
    return CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(),
        reduce=False,
    )


def test_zero_ridge_fit_is_covariant_under_response_owner_frame_change() -> None:
    """An underdetermined fit selects one physical polynomial, not one owner column."""

    direct = _basis(rotated_owner_frame=False)
    rotated = _basis(rotated_owner_frame=True)
    fit_points = np.zeros((1, 2), dtype=np.float64)
    target = np.ones((1, 1, 1), dtype=np.complex128)

    direct_fit = direct.fit(
        fit_points,
        target,
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 1],
    )
    rotated_fit = rotated.fit(
        fit_points,
        target,
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 1],
    )

    validation_points = np.asarray([[0.25, 0.0], [-0.25, 0.0]], dtype=np.float64)
    direct_h = CompiledResponseRuntime(direct, direct_fit).hamiltonians(validation_points)
    rotated_h = CompiledResponseRuntime(rotated, rotated_fit).hamiltonians(validation_points)
    np.testing.assert_allclose(direct_h, rotated_h, rtol=0.0, atol=5.0e-13)
    np.testing.assert_allclose(direct_h[:, 0, 0], 1.0, rtol=0.0, atol=5.0e-13)
    for fitted in (direct_fit, rotated_fit):
        assert (
            fitted.fit_solver_policy
            == "real_physical_coefficient_gram_whitened_minimum_norm_svd_v4"
        )
        assert fitted.fit_pivots == ()
        assert fitted.fit_response_gram_rank == 2
        assert fitted.fit_design_certified_rank == 1
        assert fitted.fit_response_gram_condition_number >= 1.0
        assert fitted.fit_response_gram_orthonormality_residual < 1.0e-12
        assert fitted.fit_selected_singular_value_min > fitted.fit_selected_error_bound
