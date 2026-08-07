from __future__ import annotations

import importlib
from types import MappingProxyType

import numpy as np
import pytest
from scipy import sparse


def _api():
    return importlib.import_module("kp.model.response_basis_local_fixed")


def _block(**overrides: object):
    api = _api()
    values: dict[str, object] = {
        "block_index": 0,
        "nominal_degrees": (0, 1),
        "owner_indices": (2, 5),
        "generator_actions": {
            "C2": np.diag([1.0, -1.0]),
            "T": np.eye(2),
        },
        "generator_error_bounds": {"C2": 1.0e-13, "T": 2.0e-13},
        "antiunitary_parities": {"C2": False, "T": True},
        "hermitian_action": np.eye(2),
    }
    values.update(overrides)
    return api.LocalActionBlock(**values)


def test_action_block_is_small_immutable_and_has_no_ambient_state() -> None:
    actions = {"C2": np.diag([1.0, -1.0]), "T": np.eye(2)}
    block = _block(generator_actions=actions)
    actions["C2"][0, 0] = 9.0

    assert isinstance(block.generator_actions, MappingProxyType)
    assert block.generator_actions["C2"][0, 0] == 1.0
    assert not hasattr(block, "ambient_columns")
    assert not hasattr(block, "exact_support_rows")
    for array in (*block.generator_actions.values(), block.hermitian_action):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)


def test_typed_error_freezes_its_certificate() -> None:
    api = _api()
    certificate = {"block_index": 3, "details": ["rank gray zone"]}
    error = api.LocalResponseCompilationError(
        "gray_zone",
        certificate=certificate,
    )
    certificate["details"].append("mutated")

    assert str(error) == "gray_zone"
    assert error.reason == "gray_zone"
    assert error.certificate == {
        "block_index": 3,
        "details": ("rank gray zone",),
    }
    with pytest.raises(TypeError):
        error.certificate["block_index"] = 4


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"block_index": -1}, "block_index"),
        ({"nominal_degrees": (0, -1)}, "nominal degree"),
        ({"nominal_degrees": (0,)}, "nominal degree count"),
        ({"owner_indices": (5, 2)}, "canonical"),
        ({"owner_indices": (2, 2)}, "duplicate"),
        ({"generator_actions": {"C2": np.eye(3), "T": np.eye(2)}}, "shape"),
        ({"generator_actions": {"T": np.eye(2), "C2": np.eye(2)}}, "ordering"),
        ({"generator_error_bounds": {"C2": 0.0}}, "error-bound keys"),
        ({"antiunitary_parities": {"C2": False}}, "parity keys"),
        ({"hermitian_action": np.eye(3)}, "hermitian_action"),
    ],
)
def test_action_block_rejects_invalid_contract(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _block(**overrides)


def test_action_blocks_depend_only_on_generator_and_hermitian_graphs() -> None:
    api = _api()
    blocks = api.build_local_action_blocks(
        nominal_degrees=(0, 0, 0),
        logical_owner_indices=(0, 1, 2),
        generator_actions={"identity": sparse.eye(3, format="csc")},
        generator_error_bounds={"identity": 0.0},
        antiunitary_parities={"identity": False},
        hermitian_action=sparse.csc_matrix(
            np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
    )

    assert tuple(block.owner_indices for block in blocks) == ((0, 1), (2,))
    assert all(not hasattr(block, "ambient_columns") for block in blocks)


def test_action_block_construction_is_invariant_to_owner_input_order() -> None:
    api = _api()
    action = sparse.csc_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    canonical = api.build_local_action_blocks(
        nominal_degrees=(0, 1, 2),
        logical_owner_indices=(2, 5, 9),
        generator_actions={"C2": action},
        generator_error_bounds={"C2": 0.0},
        antiunitary_parities={"C2": False},
        hermitian_action=sparse.eye(3, format="csc"),
    )
    permutation = np.asarray([2, 0, 1])
    shuffled = api.build_local_action_blocks(
        nominal_degrees=(2, 0, 1),
        logical_owner_indices=(9, 2, 5),
        generator_actions={"C2": action[permutation, :][:, permutation]},
        generator_error_bounds={"C2": 0.0},
        antiunitary_parities={"C2": False},
        hermitian_action=sparse.eye(3, format="csc"),
    )

    assert tuple(block.owner_indices for block in canonical) == tuple(
        block.owner_indices for block in shuffled
    )


def test_action_block_diagnostic_reports_bmax_sparsity_and_degree_tails() -> None:
    api = _api()
    generator = sparse.csc_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    blocks = api.build_local_action_blocks(
        nominal_degrees=(2, 1, 0),
        logical_owner_indices=(0, 1, 2),
        generator_actions={"affine": generator},
        generator_error_bounds={"affine": 0.0},
        antiunitary_parities={"affine": False},
        hermitian_action=sparse.eye(3, format="csc"),
    )

    diagnostic = api.summarize_local_action_blocks(blocks)

    assert diagnostic == {
        "schema_version": "local-action-block-diagnostic-v1",
        "closure_real_owner_count": 3,
        "action_block_count": 2,
        "action_block_dimensions": [2, 1],
        "maximum_action_block_dimension": 2,
        "generator_action_nonzero_count": {"affine": 4},
        "hermitian_action_nonzero_count": 3,
        "generator_lower_degree_edge_count": {"affine": 1},
        "hermitian_lower_degree_edge_count": 0,
        "total_lower_degree_edge_count": 1,
    }


def test_action_block_reynolds_uses_only_small_owner_coordinates() -> None:
    api = _api()
    block = api.LocalActionBlock(
        block_index=0,
        nominal_degrees=(0, 0),
        owner_indices=(4, 7),
        generator_actions={"C2": np.diag([1.0, -1.0])},
        generator_error_bounds={"C2": 0.0},
        antiunitary_parities={"C2": False},
        hermitian_action=np.eye(2),
    )

    result = api.solve_local_reynolds_block(
        block,
        group_words=((), ("C2",)),
    )

    assert result.fixed_rank == 1
    assert result.selected_global_owner_indices == (4,)
    np.testing.assert_array_equal(
        result.fixed_vocabulary_coordinates,
        np.asarray([[1.0], [0.0]]),
    )
    assert result.certification_metadata["raw_owner_injectivity_required"]


def test_small_reynolds_matches_independent_dense_group_average() -> None:
    from kp.model.response_basis_fixed_subspace import (
        real_linear_generator_from_complex,
    )

    api = _api()
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
    block = api.LocalActionBlock(
        block_index=0,
        nominal_degrees=(0, 0, 0, 0),
        owner_indices=(0, 1, 2, 3),
        generator_actions={"C2": unitary, "T": antiunitary},
        generator_error_bounds={"C2": 0.0, "T": 0.0},
        antiunitary_parities={"C2": False, "T": True},
        hermitian_action=np.eye(4),
    )
    result = api.solve_local_reynolds_block(
        block,
        group_words=((), ("C2",), ("T",), ("C2", "T")),
    )
    coordinates = result.fixed_vocabulary_coordinates
    basis = np.linalg.qr(coordinates)[0]
    dense_reynolds = 0.25 * (
        np.eye(4) + unitary + antiunitary + antiunitary @ unitary
    )

    np.testing.assert_allclose(basis @ basis.T, dense_reynolds, atol=1.0e-12)


def test_small_reynolds_fails_closed_when_hermitian_is_not_an_involution() -> None:
    api = _api()
    block = _block(hermitian_action=np.diag([1.0, 2.0]))

    with pytest.raises(
        api.LocalResponseCompilationError,
        match="local_reynolds_projector_certification_failed",
    ):
        api.solve_local_reynolds_block(
            block,
            group_words=((), ("C2",), ("T",), ("C2", "T")),
        )
