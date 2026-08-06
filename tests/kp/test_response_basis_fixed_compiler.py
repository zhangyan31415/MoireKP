from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy import sparse


def _api():
    return importlib.import_module("kp.model.response_basis_fixed_compiler")


def _rotation(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )


def test_nonorthogonal_vocabulary_fixed_space_matches_dense_reynolds_oracle() -> None:
    api = _api()
    physical_c3 = np.block(
        [
            [np.ones((1, 1)), np.zeros((1, 2))],
            [np.zeros((2, 1)), _rotation(2.0 * np.pi / 3.0)],
        ]
    )
    vocabulary = np.asarray(
        [
            [1.0, 0.2, -0.1],
            [0.0, 1.1, 0.3],
            [0.0, 0.0, 0.8],
        ]
    )

    result = api.compile_generator_fixed_vocabulary(
        sparse.csc_matrix(vocabulary),
        generator_images={"C3": sparse.csc_matrix(physical_c3 @ vocabulary)},
        antiunitary_parities={"C3": False},
        logical_channel_ids=("a", "b", "c"),
    )
    dense_reynolds = (np.eye(3) + physical_c3 + physical_c3 @ physical_c3) / 3.0

    assert result.rank == 1
    np.testing.assert_allclose(
        result.fixed_vectors @ result.fixed_vectors.T,
        dense_reynolds,
        atol=5.0e-13,
    )
    np.testing.assert_allclose(
        result.fixed_vectors @ result.logical_projection_coordinates,
        dense_reynolds @ vocabulary,
        atol=5.0e-13,
    )
    assert result.metadata["target_independent"] is True
    assert result.metadata["sample_grid_used"] is False


def test_common_unitary_antiunitary_fixed_space_matches_dense_group_average() -> None:
    api = _api()
    physical_c3 = np.block(
        [
            [np.ones((1, 1)), np.zeros((1, 2))],
            [np.zeros((2, 1)), _rotation(2.0 * np.pi / 3.0)],
        ]
    )
    physical_t = np.diag([1.0, 1.0, -1.0])
    vocabulary = np.asarray(
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.3]],
        dtype=np.float64,
    )

    result = api.compile_generator_fixed_vocabulary(
        vocabulary,
        generator_images={
            "C3": physical_c3 @ vocabulary,
            "T": physical_t @ vocabulary,
        },
        antiunitary_parities={"C3": False, "T": True},
        logical_channel_ids=("x", "y", "z"),
    )
    dense_group = [
        np.eye(3),
        physical_c3,
        physical_c3 @ physical_c3,
        physical_t,
        physical_t @ physical_c3,
        physical_t @ physical_c3 @ physical_c3,
    ]
    dense_reynolds = np.mean(np.stack(dense_group), axis=0)

    assert result.rank == 1
    np.testing.assert_allclose(
        result.fixed_vectors @ result.fixed_vectors.T,
        dense_reynolds,
        atol=8.0e-13,
    )
    assert result.metadata["antiunitary_parities"] == {"C3": False, "T": True}


def test_coordinate_only_compilation_does_not_materialize_ambient_fixed_vectors() -> None:
    api = _api()
    physical_c2 = np.diag([1.0, -1.0, 1.0])
    vocabulary = sparse.csc_matrix(
        np.asarray(
            [[1.0, 0.2], [0.0, 1.1], [0.0, 0.0]],
            dtype=np.float64,
        )
    )

    result = api.compile_generator_fixed_vocabulary(
        vocabulary,
        generator_images={"C2": sparse.csc_matrix(physical_c2 @ vocabulary)},
        antiunitary_parities={"C2": False},
        logical_channel_ids=("a", "b"),
        materialize_ambient_vectors=False,
    )

    assert result.fixed_vectors is None
    assert result.metadata["ambient_fixed_vectors_materialized"] is False
    ambient_fixed = vocabulary @ result.fixed_vocabulary_coordinates
    np.testing.assert_allclose(
        ambient_fixed.T @ ambient_fixed,
        np.eye(result.rank),
        atol=5.0e-13,
    )
    np.testing.assert_allclose(
        ambient_fixed @ result.logical_projection_coordinates,
        np.asarray([[1.0, 0.2], [0.0, 0.0], [0.0, 0.0]]),
        atol=5.0e-13,
    )


def test_coordinate_only_fixed_action_residual_avoids_gram_cancellation() -> None:
    api = _api()
    rng = np.random.default_rng(0)
    signs = np.diag([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    generator = rotation @ signs @ rotation.T
    vocabulary = rng.normal(size=(8, 8)) + 3.0 * np.eye(8)

    result = api.compile_generator_fixed_vocabulary(
        sparse.csc_matrix(vocabulary),
        generator_images={"g": sparse.csc_matrix(generator @ vocabulary)},
        antiunitary_parities={"g": False},
        logical_channel_ids=tuple(f"c{index}" for index in range(8)),
        materialize_ambient_vectors=False,
    )

    certification = result.metadata["fixed_physical_action_certification"][0]
    assert certification["residual_operator_norm"] < 1.0e-12


def test_generator_support_graph_is_split_into_certified_invariant_components() -> None:
    api = _api()
    vocabulary = sparse.eye(6, format="csc", dtype=np.float64)
    generator = sparse.diags(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        format="csc",
    )

    result = api.compile_generator_fixed_vocabulary(
        vocabulary,
        generator_images={"C2": generator},
        antiunitary_parities={"C2": False},
        logical_channel_ids=tuple(f"c{index}" for index in range(6)),
        materialize_ambient_vectors=False,
    )

    certification = result.metadata["invariant_component_certification"]
    assert certification["policy"] == "actual_generator_support_graph_certified_v1"
    assert certification["component_sizes"] == (1, 1, 1, 1, 1, 1)
    assert result.metadata["fixed_subspace"]["component_policy"] == (
        "actual_generator_invariance_certified_v1"
    )


def test_exact_joint_support_components_are_factored_before_dense_cholesky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    vocabulary = sparse.eye(8, format="csc", dtype=np.float64)
    generator = sparse.diags(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        format="csc",
    )
    observed_orders: list[int] = []
    original = api.scipy.linalg.cholesky

    def recording_cholesky(matrix: np.ndarray, *args: object, **kwargs: object):
        observed_orders.append(int(np.asarray(matrix).shape[0]))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(api.scipy.linalg, "cholesky", recording_cholesky)

    observed_gram_orders: list[int] = []
    original_dense_gram = api._dense_gram

    def recording_dense_gram(left: object, right: object) -> np.ndarray:
        observed_gram_orders.append(max(int(left.shape[1]), int(right.shape[1])))
        return original_dense_gram(left, right)

    monkeypatch.setattr(api, "_dense_gram", recording_dense_gram)

    result = api.compile_generator_fixed_vocabulary(
        vocabulary,
        generator_images={"C2": generator},
        antiunitary_parities={"C2": False},
        logical_channel_ids=tuple(f"c{index}" for index in range(8)),
        materialize_ambient_vectors=False,
    )

    assert max(observed_orders) == 1
    assert max(observed_gram_orders) == 1
    certification = result.metadata["early_direct_sum_certification"]
    assert certification["policy"] == "exact_coefficient_support_overlap_v1"
    assert certification["component_sizes"] == (1,) * 8


def test_generator_image_outside_authored_span_fails_closed() -> None:
    api = _api()
    vocabulary = np.asarray([[1.0], [0.0]], dtype=np.float64)
    outside = np.asarray([[0.0], [1.0]], dtype=np.float64)

    with pytest.raises(api.GeneratorSpanClosureError, match="outside authored span"):
        api.compile_generator_fixed_vocabulary(
            vocabulary,
            generator_images={"bad": outside},
            antiunitary_parities={"bad": False},
            logical_channel_ids=("only",),
        )


def test_rank_ambiguous_vocabulary_fails_closed_before_normalization() -> None:
    api = _api()
    vocabulary = np.asarray(
        [[1.0, 1.0], [0.0, 1.0e-14]],
        dtype=np.float64,
    )

    with pytest.raises(api.VocabularyRankCertificationError, match="rank is not certified"):
        api.compile_generator_fixed_vocabulary(
            vocabulary,
            generator_images={"identity": vocabulary},
            antiunitary_parities={"identity": False},
            logical_channel_ids=("first", "nearly-dependent"),
        )


def test_direct_actions_certify_rank_after_independent_column_scaling() -> None:
    api = _api()
    vocabulary = np.asarray(
        [[1.0, 1.0e-10], [0.0, 1.0e-11]],
        dtype=np.float64,
    )

    result = api.compile_generator_fixed_vocabulary_from_actions(
        vocabulary,
        generator_actions={"identity": np.eye(2, dtype=np.float64)},
        antiunitary_parities={"identity": False},
        logical_channel_ids=("order-zero", "high-order"),
    )

    assert result.rank == 2
    ambient = vocabulary @ result.fixed_vocabulary_coordinates
    np.testing.assert_allclose(ambient.T @ ambient, np.eye(2), atol=5.0e-13)
    np.testing.assert_allclose(
        ambient @ result.logical_projection_coordinates,
        vocabulary,
        atol=5.0e-13,
    )
    assert result.metadata["vocabulary_scaling_policy"] == (
        "independent_positive_column_norm_v1"
    )


def test_direct_actions_assign_zero_residual_to_empty_fixed_subspace() -> None:
    api = _api()

    result = api.compile_generator_fixed_vocabulary_from_actions(
        np.eye(2, dtype=np.float64),
        generator_actions={"sign": -np.eye(2, dtype=np.float64)},
        antiunitary_parities={"sign": False},
        logical_channel_ids=("x", "y"),
    )

    assert result.rank == 0
    assert result.fixed_vectors.shape == (2, 0)
    certification = result.metadata["fixed_physical_action_certification"][0]
    assert certification["residual_operator_norm"] == 0.0


def test_direct_action_bounds_are_propagated_through_metric_whitening() -> None:
    api = _api()
    dimension = 4
    correlation = 0.99
    gram = (
        (1.0 - correlation) * np.eye(dimension)
        + correlation * np.ones((dimension, dimension))
    )
    lower = np.linalg.cholesky(gram)
    vocabulary = lower.T

    eigenvalues, eigenvectors = np.linalg.eigh(lower.T @ lower)
    assert eigenvalues[:3] == pytest.approx((0.01, 0.01, 0.01))
    ill_conditioned_projector = eigenvectors[:, :3] @ eigenvectors[:, :3].T
    physical_residual_scale = 2.2e-12
    physical_action = (
        np.eye(dimension)
        + physical_residual_scale * ill_conditioned_projector
    )
    coefficient_action = np.linalg.solve(
        lower.T,
        physical_action @ lower.T,
    )
    ambient_residual = vocabulary @ (
        coefficient_action - np.eye(dimension)
    )
    column_bounds = np.linalg.norm(ambient_residual, axis=0)

    result = api.compile_generator_fixed_vocabulary_from_actions(
        vocabulary,
        generator_actions={"g": coefficient_action},
        antiunitary_parities={"g": False},
        logical_channel_ids=tuple(f"c{index}" for index in range(dimension)),
        generator_column_absolute_error_bounds={"g": column_bounds},
    )

    assert result.rank == dimension
    fixed_metadata = result.metadata["fixed_subspace"]["components"][0][
        "fixed_subspace"
    ]
    unwhitened_bound = float(np.linalg.norm(column_bounds))
    whitening_gain = float(np.linalg.norm(np.linalg.inv(lower.T), ord=2))
    assert fixed_metadata["action_absolute_error_bound"] == pytest.approx(
        unwhitened_bound * whitening_gain
    )


def test_direct_actions_still_reject_normalized_rank_ambiguity() -> None:
    api = _api()
    vocabulary = np.asarray(
        [[1.0, 1.0], [0.0, 1.0e-14]],
        dtype=np.float64,
    )

    with pytest.raises(api.VocabularyRankCertificationError, match="rank is not certified"):
        api.compile_generator_fixed_vocabulary_from_actions(
            vocabulary,
            generator_actions={"identity": np.eye(2, dtype=np.float64)},
            antiunitary_parities={"identity": False},
            logical_channel_ids=("first", "nearly-dependent"),
        )


def test_mismatched_generator_metadata_fails_closed() -> None:
    api = _api()
    vocabulary = np.eye(2)

    with pytest.raises(ValueError, match="generator metadata keys"):
        api.compile_generator_fixed_vocabulary(
            vocabulary,
            generator_images={"C2": np.diag([1.0, -1.0])},
            antiunitary_parities={"wrong": False},
            logical_channel_ids=("a", "b"),
        )


def test_result_is_immutable_and_records_propagated_bounds() -> None:
    api = _api()
    vocabulary = np.eye(2)
    result = api.compile_generator_fixed_vocabulary(
        vocabulary,
        generator_images={"C2": np.diag([1.0, -1.0])},
        antiunitary_parities={"C2": False},
        logical_channel_ids=("a", "b"),
        generator_absolute_error_bounds={"C2": 2.0e-13},
    )

    with pytest.raises(ValueError, match="read-only"):
        result.fixed_vectors[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.logical_projection_coordinates[0, 0] = 0.0
    with pytest.raises(TypeError):
        result.metadata["target_independent"] = False
    assert result.metadata["generator_certification"][0]["input_absolute_error_bound"] == pytest.approx(
        2.0e-13
    )


def test_generator_span_certification_uses_each_columns_own_error_bound() -> None:
    api = _api()
    vocabulary = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    image = vocabulary.copy()
    image[2, 1] = 1.0e-6

    result = api.compile_generator_fixed_vocabulary(
        vocabulary,
        generator_images={"g": image},
        antiunitary_parities={"g": False},
        logical_channel_ids=("clean", "noisy"),
        generator_column_absolute_error_bounds={"g": (1.0e-12, 2.0e-6)},
    )

    columns = result.metadata["generator_certification"][0]["columns"]
    assert columns[0]["logical_channel_id"] == "clean"
    assert columns[0]["outside_span_residual"] < columns[0]["span_certification_bound"]
    assert columns[0]["input_absolute_error_bound"] == pytest.approx(1.0e-12)
    assert columns[1]["outside_span_residual"] == pytest.approx(1.0e-6)
    assert columns[1]["input_absolute_error_bound"] == pytest.approx(2.0e-6)


def test_no_fit_target_or_sample_grid_can_enter_compiler_signature() -> None:
    api = _api()
    import inspect

    parameters = set(inspect.signature(api.compile_generator_fixed_vocabulary).parameters)
    forbidden = {"heff", "fit_indices", "regularization", "band_window", "kpoints"}
    assert not parameters.intersection(forbidden)
