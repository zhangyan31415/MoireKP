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
