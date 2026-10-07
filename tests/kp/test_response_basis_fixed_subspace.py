from __future__ import annotations

import importlib

import numpy as np
import pytest


def _api():
    return importlib.import_module("kp.model.response_basis_fixed_subspace")


def _rotation(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )


def _dense_reynolds_projector(actions: list[np.ndarray]) -> np.ndarray:
    """Independent dense finite-group average used only as a toy oracle."""

    return np.mean(np.stack(actions, axis=0), axis=0)


def test_identity_generator_retains_the_complete_real_vocabulary() -> None:
    api = _api()

    result = api.solve_generator_fixed_subspace(
        [api.RealGeneratorAction("identity", np.eye(4), antiunitary=False)]
    )

    assert result.rank == 4
    np.testing.assert_allclose(result.projector, np.eye(4), atol=1.0e-14)
    np.testing.assert_allclose(result.basis, np.eye(4), atol=1.0e-14)
    assert result.metadata["basis_convention"] == api.DETERMINISTIC_BASIS_CONVENTION_V1


def test_unitary_c3_fixed_space_matches_independent_dense_reynolds() -> None:
    api = _api()
    c3 = np.block(
        [
            [np.ones((1, 1)), np.zeros((1, 2))],
            [np.zeros((2, 1)), _rotation(2.0 * np.pi / 3.0)],
        ]
    )

    result = api.solve_generator_fixed_subspace(
        [api.RealGeneratorAction("C3", c3, antiunitary=False)]
    )
    oracle = _dense_reynolds_projector([np.eye(3), c3, c3 @ c3])

    assert result.rank == 1
    np.testing.assert_allclose(result.projector, oracle, atol=2.0e-14)
    np.testing.assert_allclose(result.basis[:, 0], [1.0, 0.0, 0.0], atol=2.0e-14)


def test_antiunitary_complex_action_is_compiled_as_a_real_linear_generator() -> None:
    api = _api()
    phase = np.exp(0.37j)

    antiunitary = api.real_linear_generator_from_complex(
        "T_phase",
        np.asarray([[phase]], dtype=np.complex128),
        antiunitary=True,
    )
    result = api.solve_generator_fixed_subspace([antiunitary])
    oracle = _dense_reynolds_projector([np.eye(2), antiunitary.matrix])

    assert antiunitary.antiunitary is True
    assert result.rank == 1
    np.testing.assert_allclose(result.projector, oracle, atol=2.0e-14)
    np.testing.assert_allclose(
        antiunitary.matrix @ result.basis,
        result.basis,
        atol=2.0e-14,
    )


def test_common_fixed_space_of_unitary_and_antiunitary_generators_matches_group_average() -> None:
    api = _api()
    c3 = np.block(
        [
            [np.ones((1, 1)), np.zeros((1, 2))],
            [np.zeros((2, 1)), _rotation(2.0 * np.pi / 3.0)],
        ]
    )
    reflection = np.diag([1.0, 1.0, -1.0])
    generators = [
        api.RealGeneratorAction("C3", c3, antiunitary=False),
        api.RealGeneratorAction("T", reflection, antiunitary=True),
    ]

    result = api.solve_generator_fixed_subspace(generators)
    oracle = _dense_reynolds_projector(
        [
            np.eye(3),
            c3,
            c3 @ c3,
            reflection,
            reflection @ c3,
            reflection @ c3 @ c3,
        ]
    )

    assert result.rank == 1
    np.testing.assert_allclose(result.projector, oracle, atol=3.0e-14)
    assert result.max_generator_residual < 3.0e-14


def test_nominal_degree_partition_is_rejected_when_generator_mixes_degrees() -> None:
    api = _api()
    mixes_degrees = api.RealGeneratorAction(
        "finite_center_adjoint",
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        antiunitary=False,
    )
    nominal_degree_blocks = [
        api.InvariantComponent("degree_0", (0,)),
        api.InvariantComponent("degree_1", (1,)),
    ]

    with pytest.raises(api.NonInvariantComponentError, match="degree_0.*not invariant"):
        api.solve_generator_fixed_subspace(
            [mixes_degrees],
            invariant_components=nominal_degree_blocks,
        )

    unpartitioned = api.solve_generator_fixed_subspace([mixes_degrees])
    assert unpartitioned.rank == 1
    np.testing.assert_allclose(
        unpartitioned.projector,
        np.full((2, 2), 0.5),
        atol=2.0e-14,
    )


def test_deterministic_basis_is_independent_of_generator_order_and_has_canonical_signs() -> None:
    api = _api()
    normal = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    reflection = np.eye(3) - 2.0 * np.outer(normal, normal)
    identity = np.eye(3)

    first = api.solve_generator_fixed_subspace(
        [
            api.RealGeneratorAction("reflection", reflection, antiunitary=False),
            api.RealGeneratorAction("identity", identity, antiunitary=False),
        ]
    )
    second = api.solve_generator_fixed_subspace(
        [
            api.RealGeneratorAction("identity", identity, antiunitary=False),
            api.RealGeneratorAction("reflection", reflection, antiunitary=False),
        ]
    )

    assert first.rank == 2
    np.testing.assert_allclose(first.projector, second.projector, atol=2.0e-14)
    np.testing.assert_allclose(first.basis, second.basis, atol=2.0e-14)
    for column in first.basis.T:
        pivot = int(np.flatnonzero(np.abs(column) > 1.0e-12)[0])
        assert column[pivot] > 0.0


def test_complex_real_linear_conventions_cover_linear_and_antilinear_maps() -> None:
    api = _api()
    matrix = np.asarray([[1.0 + 2.0j, -0.5j], [0.25, -1.0j]])
    vector = np.asarray([0.3 - 0.7j, 1.2 + 0.2j])
    real_vector = np.concatenate([vector.real, vector.imag])

    linear = api.real_linear_generator_from_complex("U", matrix, antiunitary=False)
    antilinear = api.real_linear_generator_from_complex("A", matrix, antiunitary=True)

    expected_linear = matrix @ vector
    expected_antilinear = matrix @ vector.conj()
    np.testing.assert_allclose(
        linear.matrix @ real_vector,
        np.concatenate([expected_linear.real, expected_linear.imag]),
    )
    np.testing.assert_allclose(
        antilinear.matrix @ real_vector,
        np.concatenate([expected_antilinear.real, expected_antilinear.imag]),
    )


def test_fixed_space_svd_uses_economy_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _api()
    original_svd = np.linalg.svd
    observed_full_matrices: list[bool] = []

    def recording_svd(*args, **kwargs):
        observed_full_matrices.append(bool(kwargs.get("full_matrices", True)))
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recording_svd)
    result = api.solve_generator_fixed_subspace(
        [api.RealGeneratorAction("identity", np.eye(32), antiunitary=False)]
    )

    assert result.rank == 32
    assert observed_full_matrices == [False]


def test_fixed_space_result_arrays_are_immutable_and_certification_is_auditable() -> None:
    api = _api()
    nearly_fixed = np.diag([1.0, 1.0 + 5.0e-11, -1.0])

    with pytest.raises(api.FixedSubspaceCertificationError, match="gray zone"):
        api.solve_generator_fixed_subspace(
            [api.RealGeneratorAction("nearly_fixed", nearly_fixed, antiunitary=False)],
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        )

    result = api.solve_generator_fixed_subspace(
        [api.RealGeneratorAction("well_separated", np.diag([1.0, 1.0, -1.0]), antiunitary=False)],
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )

    with pytest.raises(ValueError, match="read-only"):
        result.basis[0, 0] = 3.0
    with pytest.raises(ValueError, match="read-only"):
        result.projector[0, 0] = 3.0
    with pytest.raises(ValueError, match="read-only"):
        result.singular_values[0] = 3.0

    certification = result.metadata["fixed_subspace_certification"]
    assert certification["policy"] == api.FIXED_SUBSPACE_CERTIFICATION_POLICY_V1
    assert certification["gray_zone_factor"] == 100.0
    component = certification["components"][0]
    assert component["singular_values"] == pytest.approx(
        sorted(result.singular_values.tolist(), reverse=True)
    )
    assert component["gray_zone_singular_values"] == []
    assert component["spectral_gap_absolute"] > 0.0
    assert component["spectral_gap_ratio"] > 1.0


def test_generator_residual_above_certification_bound_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()

    def corrupt_basis(projector, rank, **kwargs):
        del projector, kwargs
        assert rank == 1
        return np.asarray([[0.0], [1.0]], dtype=np.float64)

    monkeypatch.setattr(api, "_deterministic_basis_from_projector", corrupt_basis)
    reflection = np.diag([1.0, -1.0])

    with pytest.raises(api.FixedSubspaceCertificationError, match="generator.*residual"):
        api.solve_generator_fixed_subspace(
            [api.RealGeneratorAction("reflection", reflection, antiunitary=False)]
        )


def test_canonical_frame_avoids_amplifying_low_leverage_projector_columns() -> None:
    api = _api()
    epsilon = 1.0e-10
    projector = np.zeros((4, 4), dtype=np.float64)
    projector[0, 0] = 1.0
    projector[2, 2] = 1.0
    projector[1, 1] = epsilon**2
    projector[1, 2] = projector[2, 1] = epsilon
    # A coefficient-space roundoff tail in the low-leverage column must not be
    # amplified by choosing that column before the unit-leverage column.
    projector[1, 3] = projector[3, 1] = 1.0e-20
    action = np.diag([1.0, 1.0, 1.0, -1.0])

    basis = api._deterministic_basis_from_projector(
        projector,
        2,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )

    assert np.linalg.norm(action @ basis - basis) < 1.0e-14


def test_action_error_bound_enters_fixed_rank_threshold_conservatively() -> None:
    api = _api()
    action = api.RealGeneratorAction(
        "uncertain",
        np.diag([1.0, 1.0 + 5.0e-8, -1.0]),
        antiunitary=False,
    )

    result = api.solve_generator_fixed_subspace(
        [action],
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
        action_absolute_error_bound=1.0e-7,
    )

    assert result.rank == 2
    assert result.metadata["action_absolute_error_bound"] == pytest.approx(1.0e-7)
