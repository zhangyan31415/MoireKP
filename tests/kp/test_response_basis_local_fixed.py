from __future__ import annotations

import importlib
from types import MappingProxyType

import numpy as np
import pytest
from scipy import sparse


def _api():
    return importlib.import_module("kp.model.response_basis_local_fixed")


def _component(**overrides: object):
    api = _api()
    values: dict[str, object] = {
        "component_index": 0,
        "nominal_degrees": (0, 1),
        "logical_owner_indices": (2, 5),
        "provenance_by_owner": {
            2: {"source": "constant", "path": ["seed", 0]},
            5: {"source": "linear"},
        },
        "ambient_columns": sparse.csr_matrix(
            np.asarray([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        ),
        "generator_actions": {
            "C3": np.eye(2),
            "T": np.diag([1.0, -1.0]),
        },
        "generator_error_bounds": {"C3": 1.0e-13, "T": 2.0e-13},
        "antiunitary_parities": {"C3": False, "T": True},
        "hermitian_action": np.eye(2),
        "exact_support_rows": (0, 1),
    }
    values.update(overrides)
    return api.LocalVocabularyComponent(**values)


def _compilation(**overrides: object):
    api = _api()
    values: dict[str, object] = {
        "component_index": 0,
        "component_owner_indices": (2, 5),
        "fixed_vocabulary_coordinates": np.asarray([[1.0], [0.0]]),
        "logical_projection_coordinates": np.asarray([[1.0, 0.0]]),
        "fixed_rank": 1,
        "selected_global_owner_indices": (2,),
        "singular_values": np.asarray([0.0, 2.0]),
        "gray_zone_certificate": {"threshold": 1.0e-12, "values": []},
        "generator_residuals": {"C3": 1.0e-14, "T": 2.0e-14},
        "hermitian_residual": 3.0e-14,
        "propagated_error_bounds": {"C3": 4.0e-13, "T": 5.0e-13},
        "certification_metadata": {"policy": "local_test_v1", "trace": [0, 1]},
    }
    values.update(overrides)
    return api.LocalFixedCompilation(**values)


def test_numeric_buffers_are_byte_backed_and_cannot_be_reenabled() -> None:
    component = _component()
    result = _compilation()

    for array in (
        component.generator_actions["C3"],
        component.hermitian_action,
        component.ambient_columns.data,
        component.ambient_columns.indices,
        component.ambient_columns.indptr,
        result.fixed_vocabulary_coordinates,
        result.logical_projection_coordinates,
        result.singular_values,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)


def test_valid_local_component_normalizes_storage_and_deep_freezes_metadata() -> None:
    provenance = {
        2: {"source": "constant", "path": ["seed", 0]},
        5: {"source": "linear"},
    }
    actions = {
        "C3": np.eye(2, dtype=np.float32),
        "T": np.diag([1, -1]),
    }
    ambient = sparse.csr_matrix(np.asarray([[1, 0], [0, 2], [0, 0]]))

    component = _component(
        provenance_by_owner=provenance,
        generator_actions=actions,
        ambient_columns=ambient,
    )
    provenance[2]["path"].append("mutated")
    actions["C3"][0, 0] = 9.0

    assert sparse.isspmatrix_csc(component.ambient_columns)
    assert component.ambient_columns.dtype == np.float64
    assert isinstance(component.provenance_by_owner, MappingProxyType)
    assert isinstance(component.provenance_by_owner[2], MappingProxyType)
    assert component.provenance_by_owner[2]["path"] == ("seed", 0)
    assert component.generator_actions["C3"].dtype == np.float64
    assert component.generator_actions["C3"][0, 0] == 1.0
    assert not component.generator_actions["C3"].flags.writeable
    with pytest.raises(TypeError):
        component.provenance_by_owner[2]["source"] = "changed"


def test_valid_local_compilation_normalizes_arrays_and_deep_freezes_metadata() -> None:
    metadata = {"policy": "local_test_v1", "trace": [0, {"rank": 1}]}
    result = _compilation(certification_metadata=metadata)
    metadata["trace"][1]["rank"] = 9

    assert result.fixed_vocabulary_coordinates.dtype == np.float64
    assert result.logical_projection_coordinates.dtype == np.float64
    assert result.singular_values.dtype == np.float64
    assert not result.fixed_vocabulary_coordinates.flags.writeable
    assert result.certification_metadata["trace"][1]["rank"] == 1
    with pytest.raises(TypeError):
        result.gray_zone_certificate["threshold"] = 0.0


def test_typed_unavailability_preserves_stable_reason_and_certificate() -> None:
    api = _api()
    certificate = {"component_index": 3, "details": ["rank gray zone"]}

    error = api.LocalGeneratorNullspaceUnavailable(
        "gray_zone",
        certificate=certificate,
    )
    certificate["details"].append("mutated")

    assert isinstance(error, RuntimeError)
    assert str(error) == "gray_zone"
    assert error.reason == "gray_zone"
    assert error.certificate == {
        "component_index": 3,
        "details": ("rank gray zone",),
    }
    with pytest.raises(TypeError):
        error.certificate["component_index"] = 4


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"component_index": -1}, "component_index"),
        ({"nominal_degrees": (0, -1)}, "nominal degree"),
        ({"nominal_degrees": (0, 1.5)}, "nominal degree"),
        ({"logical_owner_indices": (2, 2)}, "owner indices"),
        ({"logical_owner_indices": (5, 2)}, "canonical"),
        ({"logical_owner_indices": (2,)}, "column count"),
        ({"provenance_by_owner": {2: {}, 7: {}}}, "provenance"),
        ({"exact_support_rows": (1, 1)}, "duplicate"),
        ({"exact_support_rows": (1, 0)}, "canonical"),
        ({"exact_support_rows": (0, 3)}, "ambient row"),
        ({"ambient_columns": sparse.eye(3, format="csc")}, "column count"),
        ({"hermitian_action": np.eye(3)}, "hermitian_action"),
        ({"hermitian_action": np.asarray([[1.0, np.nan], [0.0, 1.0]])}, "non-finite"),
        (
            {"generator_actions": {"C3": np.eye(3), "T": np.eye(2)}},
            "generator action.*shape",
        ),
        (
            {
                "generator_actions": {
                    "C3": np.asarray([[1.0, 1.0j], [0.0, 1.0]]),
                    "T": np.eye(2),
                }
            },
            "real",
        ),
        (
            {
                "generator_actions": {
                    "C3": np.asarray([[1.0, np.inf], [0.0, 1.0]]),
                    "T": np.eye(2),
                }
            },
            "non-finite",
        ),
        (
            {"generator_actions": {"T": np.eye(2), "C3": np.eye(2)}},
            "canonical generator ordering",
        ),
        ({"antiunitary_parities": {"C3": False}}, "parity keys"),
        (
            {"antiunitary_parities": {"C3": False, "T": True, "extra": False}},
            "parity keys",
        ),
        ({"generator_error_bounds": {"C3": 0.0}}, "error-bound keys"),
        ({"generator_error_bounds": {"C3": -1.0, "T": 0.0}}, "error bound"),
        ({"generator_error_bounds": {"C3": np.nan, "T": 0.0}}, "error bound"),
    ],
)
def test_local_component_rejects_invalid_contract(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _component(**overrides)


def test_local_component_rejects_complex_sparse_ambient_columns() -> None:
    ambient = sparse.csc_matrix(
        np.asarray([[1.0, 0.0j], [0.0, 1.0j], [0.0, 0.0]])
    )
    with pytest.raises(ValueError, match="ambient_columns.*real"):
        _component(ambient_columns=ambient)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"component_index": -1}, "component_index"),
        ({"component_owner_indices": (5, 2)}, "canonical"),
        ({"component_owner_indices": (2, 2)}, "owner indices"),
        ({"fixed_rank": -1}, "fixed_rank"),
        ({"fixed_rank": 2}, "fixed_vocabulary_coordinates"),
        (
            {"fixed_vocabulary_coordinates": np.ones((3, 1))},
            "fixed_vocabulary_coordinates",
        ),
        (
            {"logical_projection_coordinates": np.ones((1, 3))},
            "logical_projection_coordinates",
        ),
        ({"selected_global_owner_indices": (5, 2)}, "canonical"),
        ({"selected_global_owner_indices": (2, 2)}, "selected owner indices"),
        ({"selected_global_owner_indices": (7,)}, "does not belong"),
        ({"selected_global_owner_indices": ()}, "fixed_rank"),
        ({"singular_values": np.asarray([0.0])}, "singular_values.*shape"),
        ({"singular_values": np.asarray([0.0, np.nan])}, "singular_values"),
        ({"generator_residuals": {"C3": np.nan, "T": 0.0}}, "generator residual"),
        (
            {"generator_residuals": {"T": 0.0, "C3": 0.0}},
            "canonical generator ordering",
        ),
        ({"hermitian_residual": np.nan}, "hermitian_residual"),
        ({"propagated_error_bounds": {"C3": -1.0, "T": 0.0}}, "propagated error bound"),
        (
            {"propagated_error_bounds": {"C3": 0.0, "T": np.inf}},
            "propagated error bound",
        ),
        (
            {"propagated_error_bounds": {"T": 0.0, "C3": 0.0}},
            "canonical generator ordering",
        ),
        ({"fixed_vocabulary_coordinates": np.asarray([[1.0j], [0.0]])}, "real"),
    ],
)
def test_local_compilation_rejects_invalid_contract(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _compilation(**overrides)


def test_zero_rank_local_compilation_has_canonical_empty_shapes() -> None:
    result = _compilation(
        fixed_vocabulary_coordinates=np.zeros((2, 0)),
        logical_projection_coordinates=np.zeros((0, 2)),
        fixed_rank=0,
        selected_global_owner_indices=(),
    )

    assert result.fixed_vocabulary_coordinates.shape == (2, 0)
    assert result.logical_projection_coordinates.shape == (0, 2)


def _build_components(
    ambient: np.ndarray,
    *,
    actions: dict[str, np.ndarray] | None = None,
    hermitian: np.ndarray | None = None,
    owners: tuple[int, ...] | None = None,
):
    api = _api()
    dimension = int(ambient.shape[1])
    owner_indices = owners or tuple(range(dimension))
    generator_actions = actions or {"identity": np.eye(dimension)}
    return api.build_exact_joint_components(
        sparse.csc_matrix(ambient, dtype=np.float64),
        nominal_degrees=tuple(0 for _ in owner_indices),
        logical_owner_indices=owner_indices,
        provenance_by_owner={owner: {"owner": owner} for owner in owner_indices},
        generator_actions=generator_actions,
        generator_error_bounds={name: 0.0 for name in generator_actions},
        antiunitary_parities={name: False for name in generator_actions},
        hermitian_action=(
            np.eye(dimension) if hermitian is None else hermitian
        ),
        column_absolute_error_bounds=np.zeros(dimension),
    )


def test_exact_joint_components_split_disjoint_action_adjoint_and_support() -> None:
    components = _build_components(np.eye(3))

    assert tuple(component.logical_owner_indices for component in components) == (
        (0,),
        (1,),
        (2,),
    )


def test_exact_joint_components_merge_shared_ambient_lower_tail() -> None:
    ambient = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, -3.0],
        ]
    )

    components = _build_components(ambient)

    assert tuple(component.logical_owner_indices for component in components) == (
        (0, 1),
    )


def test_exact_joint_components_merge_hermitian_partner() -> None:
    ambient = np.eye(2)
    hermitian = np.asarray([[0.0, 1.0], [1.0, 0.0]])

    components = _build_components(ambient, hermitian=hermitian)

    assert tuple(component.logical_owner_indices for component in components) == (
        (0, 1),
    )


def test_local_fixed_space_matches_independent_dense_reynolds_projector() -> None:
    from kp.model.response_basis_fixed_subspace import (
        real_linear_generator_from_complex,
    )

    unitary = real_linear_generator_from_complex(
        "C2",
        np.diag([1.0, -1.0]),
        antiunitary=False,
    ).matrix
    antiunitary = real_linear_generator_from_complex(
        "T",
        np.eye(2),
        antiunitary=True,
    ).matrix
    components = _build_components(
        np.eye(4),
        actions={"C2": unitary, "T": antiunitary},
    )
    api = _api()

    result = api.solve_local_fixed_component(components[0])
    physical = np.asarray(
        components[0].ambient_columns @ result.fixed_vocabulary_coordinates
    )
    fast_projector = physical @ physical.T
    dense_reynolds = 0.25 * (
        np.eye(4)
        + unitary
        + antiunitary
        + unitary @ antiunitary
    )

    assert result.fixed_rank == 1
    np.testing.assert_allclose(fast_projector, dense_reynolds, atol=1.0e-12)


def test_local_reynolds_block_solver_matches_dense_group_average() -> None:
    from kp.model.response_basis_fixed_subspace import (
        real_linear_generator_from_complex,
    )

    unitary = real_linear_generator_from_complex(
        "C2",
        np.diag([1.0, -1.0]),
        antiunitary=False,
    ).matrix
    antiunitary = real_linear_generator_from_complex(
        "T",
        np.eye(2),
        antiunitary=True,
    ).matrix
    component = _build_components(
        np.eye(4),
        actions={"C2": unitary, "T": antiunitary},
    )[0]
    api = _api()

    result = api.solve_local_reynolds_component_blocked(
        component,
        group_words=((), ("C2",), ("T",), ("C2", "T")),
    )
    physical = np.asarray(
        component.ambient_columns @ result.fixed_vocabulary_coordinates
    )
    physical_q = np.linalg.qr(physical)[0]
    dense_reynolds = 0.25 * (
        np.eye(4)
        + unitary
        + antiunitary
        + antiunitary @ unitary
    )

    assert result.fixed_rank == 1
    assert result.certification_metadata["algorithm"] == (
        "exact_action_block__small_matrix_reynolds_v1"
    )
    np.testing.assert_allclose(
        physical_q @ physical_q.T,
        dense_reynolds,
        atol=1.0e-12,
    )


def test_local_fixed_solver_filters_duplicate_and_nonorthogonal_vocabulary() -> None:
    ambient = np.asarray(
        [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    component = _build_components(ambient)[0]
    api = _api()

    result = api.solve_local_fixed_component(component)
    physical = np.asarray(
        component.ambient_columns @ result.fixed_vocabulary_coordinates
    )

    assert result.fixed_rank == 2
    assert result.selected_global_owner_indices == (0, 2)
    np.testing.assert_allclose(physical, ambient[:, [0, 2]], atol=1.0e-12)
    physical_q = np.linalg.qr(physical)[0]
    ambient_q = np.linalg.qr(ambient[:, [0, 2]])[0]
    np.testing.assert_allclose(
        physical_q @ physical_q.T,
        ambient_q @ ambient_q.T,
        atol=1.0e-12,
    )


def test_local_fixed_solver_accepts_dense_two_by_two_complex_route_block() -> None:
    from kp.model.response_basis_fixed_subspace import (
        real_linear_generator_from_complex,
    )

    dense_c3 = np.asarray(
        [[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128
    )
    c3_real = real_linear_generator_from_complex(
        "C3",
        dense_c3,
        antiunitary=False,
    ).matrix
    component = _build_components(
        np.eye(4),
        actions={"C3": c3_real},
    )[0]
    api = _api()

    result = api.solve_local_fixed_component(component)

    assert result.fixed_rank == 0
    assert result.certification_metadata["generator_action_nnz"]["C3"] == 4


def test_local_fixed_solver_fails_closed_on_fixed_rank_gray_zone() -> None:
    action = np.diag([1.0, 1.0 + 5.0e-11])
    component = _build_components(
        np.eye(2),
        actions={"near": action},
    )[-1]
    api = _api()

    with pytest.raises(
        api.LocalGeneratorNullspaceUnavailable,
        match="fixed_rank_gray_zone",
    ):
        api.solve_local_fixed_component(component)


def test_local_fixed_solver_uses_typed_fallback_for_oversize_component() -> None:
    dimension = 513
    component = _build_components(
        np.ones((1, dimension)),
    )[0]
    api = _api()

    with pytest.raises(
        api.LocalGeneratorNullspaceUnavailable,
        match="oversize_component",
    ):
        api.solve_local_fixed_component(component, dense_component_cutoff=512)


def test_block_backend_solves_oversized_full_rank_component_by_action_blocks() -> None:
    dimension = 5
    ambient = np.vstack(
        [
            np.ones((1, dimension), dtype=np.float64),
            np.eye(dimension, dtype=np.float64),
        ]
    )
    component = _build_components(ambient)[0]
    api = _api()

    blocked = api.solve_local_fixed_component_blocked(
        component,
        dense_component_cutoff=4,
    )
    dense = api.solve_local_fixed_component(
        component,
        dense_component_cutoff=8,
    )
    blocked_physical = np.asarray(
        component.ambient_columns @ blocked.fixed_vocabulary_coordinates
    )
    dense_physical = np.asarray(
        component.ambient_columns @ dense.fixed_vocabulary_coordinates
    )
    blocked_q = np.linalg.qr(blocked_physical)[0]
    dense_q = np.linalg.qr(dense_physical)[0]

    assert blocked.fixed_rank == dense.fixed_rank == dimension
    assert blocked.certification_metadata["action_block_count"] == dimension
    assert blocked.certification_metadata["maximum_action_block_dimension"] == 1
    assert blocked.certification_metadata[
        "global_vocabulary_full_rank_certified"
    ]
    np.testing.assert_allclose(
        blocked_q @ blocked_q.T,
        dense_q @ dense_q.T,
        atol=1.0e-11,
    )


def test_block_backend_fails_closed_on_cross_block_vocabulary_dependency() -> None:
    dimension = 5
    ambient = np.vstack(
        [
            np.ones((1, dimension), dtype=np.float64),
            np.eye(dimension, dtype=np.float64),
        ]
    )
    ambient[:, -1] = ambient[:, 0]
    component = _build_components(ambient)[0]
    api = _api()

    with pytest.raises(
        api.LocalGeneratorNullspaceUnavailable,
        match="oversize_vocabulary_rank_deficient",
    ):
        api.solve_local_fixed_component_blocked(
            component,
            dense_component_cutoff=4,
        )


def test_block_backend_uses_typed_fallback_for_oversized_action_block() -> None:
    dimension = 5
    cycle = np.zeros((dimension, dimension), dtype=np.float64)
    for column in range(dimension):
        cycle[(column + 1) % dimension, column] = 1.0
    component = _build_components(
        np.eye(dimension, dtype=np.float64),
        actions={"cycle": cycle},
    )[0]
    api = _api()

    with pytest.raises(
        api.LocalGeneratorNullspaceUnavailable,
        match="oversize_irreducible_action_block",
    ):
        api.solve_local_fixed_component_blocked(
            component,
            dense_component_cutoff=4,
        )


def test_generator_order_is_canonicalized_without_changing_components() -> None:
    ambient = np.eye(2)
    canonical = _build_components(
        ambient,
        actions={"C3": np.eye(2), "T": np.eye(2)},
    )
    shuffled = _build_components(
        ambient,
        actions={"T": np.eye(2), "C3": np.eye(2)},
    )

    assert tuple(component.logical_owner_indices for component in canonical) == tuple(
        component.logical_owner_indices for component in shuffled
    )
    assert tuple(canonical[0].generator_actions) == ("C3", "T")
    assert tuple(shuffled[0].generator_actions) == ("C3", "T")


def test_component_order_and_owner_selection_ignore_input_shuffle() -> None:
    ambient = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    canonical = _build_components(ambient, owners=(2, 5, 8))
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    shuffled = _build_components(
        ambient[:, permutation],
        owners=(8, 2, 5),
    )
    api = _api()

    canonical_result = tuple(
        api.solve_local_fixed_component(component) for component in canonical
    )
    shuffled_result = tuple(
        api.solve_local_fixed_component(component) for component in shuffled
    )

    assert tuple(c.logical_owner_indices for c in canonical) == tuple(
        c.logical_owner_indices for c in shuffled
    )
    assert tuple(r.selected_global_owner_indices for r in canonical_result) == tuple(
        r.selected_global_owner_indices for r in shuffled_result
    )
