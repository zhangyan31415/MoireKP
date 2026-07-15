from __future__ import annotations

import numpy as np
import pytest

from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    JointExactificationError,
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
    SemilinearBlock,
    compile_continuum_magnetic_presentation,
    compose_semilinear,
    extract_block_route_action,
    inverse_semilinear,
    materialize_block_route_action,
    validate_presentation_action_relations,
)


def test_semilinear_composition_conjugates_only_the_inner_block() -> None:
    outer = SemilinearBlock(np.array([[1.0j]]), antiunitary=True)
    inner = SemilinearBlock(
        np.array([[np.exp(0.3j)]]),
        antiunitary=False,
    )

    product = compose_semilinear(outer, inner)

    np.testing.assert_allclose(product.matrix, [[1.0j * np.exp(-0.3j)]])
    assert product.antiunitary is True


def test_semilinear_inverse_is_a_two_sided_inverse() -> None:
    unitary = np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128)
    value = SemilinearBlock(unitary, antiunitary=True)

    inverse = inverse_semilinear(value)

    identity = np.eye(2, dtype=np.complex128)
    np.testing.assert_allclose(compose_semilinear(value, inverse).matrix, identity)
    np.testing.assert_allclose(compose_semilinear(inverse, value).matrix, identity)


def test_semilinear_block_owns_a_readonly_complex_copy() -> None:
    source = np.eye(2, dtype=np.float64)

    value = SemilinearBlock(source, antiunitary=False)
    source[0, 0] = 0.0

    assert value.matrix.dtype == np.dtype(np.complex128)
    assert value.matrix.flags.c_contiguous
    assert value.matrix.flags.writeable is False
    np.testing.assert_array_equal(value.matrix, np.eye(2, dtype=np.complex128))


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.zeros((2, 3), dtype=np.complex128), "square"),
        (np.array([[np.nan]], dtype=np.complex128), "finite"),
        (np.diag([1.0, 2.0]).astype(np.complex128), "unitarity"),
    ],
)
def test_semilinear_block_rejects_invalid_matrices(
    matrix: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(JointExactificationError, match=message):
        SemilinearBlock(
            matrix,
            antiunitary=False,
            unitarity_certification_bound=1.0e-14,
        )


def test_semilinear_block_rejects_invalid_certification_bound() -> None:
    with pytest.raises(JointExactificationError, match="certification bound"):
        SemilinearBlock(
            np.eye(1, dtype=np.complex128),
            antiunitary=False,
            unitarity_certification_bound=-1.0,
        )


def test_block_route_action_validates_and_owns_route_blocks() -> None:
    first = np.eye(2, dtype=np.complex128)
    second = np.diag([1.0j, -1.0j]).astype(np.complex128)

    action = BlockRouteAction(
        name="g",
        antiunitary=False,
        fiber_permutation=(1, 0),
        fiber_dimensions=(2, 2),
        route_blocks=(first, second),
    )
    first[0, 0] = 0.0

    assert action.route_blocks[0].flags.writeable is False
    assert action.route_blocks[0].flags.c_contiguous
    np.testing.assert_array_equal(action.route_blocks[0], np.eye(2))


@pytest.mark.parametrize(
    ("permutation", "dimensions", "blocks", "message"),
    [
        ((0, 0), (1, 1), (np.eye(1), np.eye(1)), "permutation"),
        ((1, 0), (1, 2), (np.ones((2, 1)), np.ones((1, 2))), "dimension"),
        ((1, 0), (2, 2), (np.eye(2),), "one route block"),
        (
            (1, 0),
            (2, 2),
            (np.ones((1, 2)), np.ones((2, 1))),
            "shape",
        ),
        (
            (0,),
            (2,),
            (np.diag([1.0, 2.0]),),
            "unitarity",
        ),
    ],
)
def test_block_route_action_rejects_invalid_structure(
    permutation: tuple[int, ...],
    dimensions: tuple[int, ...],
    blocks: tuple[np.ndarray, ...],
    message: str,
) -> None:
    with pytest.raises(JointExactificationError, match=message):
        BlockRouteAction(
            name="g",
            antiunitary=False,
            fiber_permutation=permutation,
            fiber_dimensions=dimensions,
            route_blocks=blocks,
        )


def test_route_round_trip_preserves_small_internal_entries() -> None:
    eps = 5.0e-5
    block = np.array(
        [[np.sqrt(1.0 - eps**2), eps], [-eps, np.sqrt(1.0 - eps**2)]],
        dtype=np.complex128,
    )
    action = BlockRouteAction(
        name="g",
        antiunitary=False,
        fiber_permutation=(1, 0),
        fiber_dimensions=(2, 2),
        route_blocks=(block, block.conj().T),
    )

    dense = materialize_block_route_action(action)
    restored = extract_block_route_action(
        dense,
        name="g",
        antiunitary=False,
        fiber_indices=((0, 1), (2, 3)),
        fiber_permutation=(1, 0),
        off_route_bound=0.0,
    )

    assert restored.route_blocks[0][0, 1] == eps
    np.testing.assert_array_equal(materialize_block_route_action(restored), dense)
    route_support = dense != 0.0
    assert np.count_nonzero(route_support) == 8
    assert np.count_nonzero(dense[~route_support]) == 0


def test_route_round_trip_preserves_noncontiguous_fiber_layout() -> None:
    action = BlockRouteAction(
        name="g",
        antiunitary=True,
        fiber_permutation=(1, 0),
        fiber_dimensions=(2, 2),
        route_blocks=(
            np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128),
            np.array([[0.0, 1.0], [1.0j, 0.0]], dtype=np.complex128),
        ),
        fiber_indices=((0, 2), (1, 3)),
    )

    dense = materialize_block_route_action(action)
    restored = extract_block_route_action(
        dense,
        name="g",
        antiunitary=True,
        fiber_indices=((0, 2), (1, 3)),
        fiber_permutation=(1, 0),
        off_route_bound=0.0,
    )

    assert restored.fiber_indices == ((0, 2), (1, 3))
    np.testing.assert_array_equal(materialize_block_route_action(restored), dense)


def test_route_extraction_rejects_off_route_pollution_above_bound() -> None:
    dense = np.eye(2, dtype=np.complex128)
    dense[1, 0] = 2.0e-8

    with pytest.raises(JointExactificationError, match="off-route"):
        extract_block_route_action(
            dense,
            name="g",
            antiunitary=False,
            fiber_indices=((0,), (1,)),
            fiber_permutation=(0, 1),
            off_route_bound=1.0e-9,
        )


def test_route_extraction_rejects_incompatible_fiber_dimensions() -> None:
    with pytest.raises(JointExactificationError, match="dimension"):
        extract_block_route_action(
            np.eye(3, dtype=np.complex128),
            name="g",
            antiunitary=False,
            fiber_indices=((0,), (1, 2)),
            fiber_permutation=(1, 0),
            off_route_bound=0.0,
        )


@pytest.mark.parametrize(
    "fiber_indices",
    [
        ((0,), (0,)),
        ((0,), (2,)),
        ((0,), (1, 2)),
    ],
)
def test_route_extraction_requires_a_complete_disjoint_fiber_partition(
    fiber_indices: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(JointExactificationError, match="fiber indices"):
        extract_block_route_action(
            np.eye(2, dtype=np.complex128),
            name="g",
            antiunitary=False,
            fiber_indices=fiber_indices,
            fiber_permutation=tuple(range(len(fiber_indices))),
            off_route_bound=0.0,
        )


def _manifest_operation(
    name: str,
    *,
    antiunitary: bool,
    power: int,
    phase: int,
    action_type: str,
    action_value: float | None = None,
    sector_map: str = "identity",
) -> dict[str, object]:
    action_map: dict[str, object] = {
        "type": action_type,
        "in_model_frame": True,
    }
    if action_type == "rotation":
        action_map["angle_deg"] = action_value
    elif action_type == "reflection":
        action_map["axis_deg"] = action_value
    action = {
        "antiunitary": antiunitary,
        "k_map": dict(action_map),
        "q_map": dict(action_map),
        "sector_map": sector_map,
        "valley_map": "identity",
    }
    return {
        "name": name,
        "operation": name,
        "antiunitary": antiunitary,
        "declared_model_action": action,
        "group_relations": [
            {
                "type": "power",
                "name": f"{name}^{power}",
                "operation": name,
                "power": power,
                "phase": phase,
                "source": "kp_symm_manifest",
            }
        ],
    }


def _tr(*, phase: int) -> dict[str, object]:
    return _manifest_operation(
        "TR",
        antiunitary=True,
        power=2,
        phase=phase,
        action_type="negation",
    )


def _c3z() -> dict[str, object]:
    return _manifest_operation(
        "C3z",
        antiunitary=False,
        power=3,
        phase=-1,
        action_type="rotation",
        action_value=120.0,
    )


def _c2(*, phase: int) -> dict[str, object]:
    return _manifest_operation(
        "C2",
        antiunitary=False,
        power=2,
        phase=phase,
        action_type="reflection",
        action_value=360.0,
        sector_map="layer_exchange",
    )


def _c2t() -> dict[str, object]:
    return _manifest_operation(
        "C2T",
        antiunitary=True,
        power=2,
        phase=1,
        action_type="reflection",
        action_value=360.0,
        sector_map="layer_exchange",
    )


def _relation_signature(
    relation: MagneticRelation,
) -> tuple[str, tuple[str, ...], tuple[str, ...], complex]:
    return relation.name, relation.lhs, relation.rhs, relation.central_phase


def test_compiles_mgi2_m_magnetic_presentation() -> None:
    presentation = compile_continuum_magnetic_presentation(
        [_tr(phase=1), _c2(phase=1)]
    )

    assert presentation.generators == (
        MagneticGenerator("TR", antiunitary=True),
        MagneticGenerator("C2", antiunitary=False),
    )
    assert tuple(map(_relation_signature, presentation.relations)) == (
        ("TR^2", ("TR", "TR"), (), 1.0 + 0.0j),
        ("C2^2", ("C2", "C2"), (), 1.0 + 0.0j),
        ("TR_C2_commute", ("TR", "C2"), ("C2", "TR"), 1.0 + 0.0j),
    )
    assert presentation.central_phases == (1.0 + 0.0j,)


def test_compiles_mote2_k_magnetic_presentation_with_explicit_minus_identity() -> None:
    presentation = compile_continuum_magnetic_presentation([_c2t(), _c3z()])

    assert presentation.generators == (
        MagneticGenerator("C3z", antiunitary=False),
        MagneticGenerator("C2T", antiunitary=True),
    )
    assert tuple(map(_relation_signature, presentation.relations)) == (
        ("C3z^3", ("C3z", "C3z", "C3z"), (), -1.0 + 0.0j),
        ("C2T^2", ("C2T", "C2T"), (), 1.0 + 0.0j),
        (
            "C2T_C3z_dihedral",
            ("C2T", "C3z", "C2T"),
            ("C3z", "C3z"),
            -1.0 + 0.0j,
        ),
    )
    assert presentation.central_phases == (1.0 + 0.0j, -1.0 + 0.0j)
    assert presentation.central_phases[0] != presentation.central_phases[1]


def test_compiles_spinful_gamma_magnetic_presentation() -> None:
    presentation = compile_continuum_magnetic_presentation(
        [_c2(phase=-1), _tr(phase=-1), _c3z()]
    )

    assert tuple(generator.name for generator in presentation.generators) == (
        "TR",
        "C3z",
        "C2",
    )
    assert tuple(map(_relation_signature, presentation.relations)) == (
        ("TR^2", ("TR", "TR"), (), -1.0 + 0.0j),
        ("C3z^3", ("C3z", "C3z", "C3z"), (), -1.0 + 0.0j),
        ("C2^2", ("C2", "C2"), (), -1.0 + 0.0j),
        ("TR_C3z_commute", ("TR", "C3z"), ("C3z", "TR"), 1.0 + 0.0j),
        ("TR_C2_commute", ("TR", "C2"), ("C2", "TR"), 1.0 + 0.0j),
        (
            "C2_C3z_dihedral",
            ("C2", "C3z", "C2"),
            ("C3z", "C3z"),
            1.0 + 0.0j,
        ),
    )


def test_compiles_gamma_tr_c3z_subset() -> None:
    presentation = compile_continuum_magnetic_presentation([_c3z(), _tr(phase=-1)])

    assert tuple(generator.name for generator in presentation.generators) == (
        "TR",
        "C3z",
    )
    assert presentation.relations[-1] == MagneticRelation(
        "TR_C3z_commute",
        lhs=("TR", "C3z"),
        rhs=("C3z", "TR"),
        central_phase=1.0 + 0.0j,
    )


@pytest.mark.parametrize(
    "operations",
    [
        [_manifest_operation(
            "C4z",
            antiunitary=False,
            power=4,
            phase=-1,
            action_type="rotation",
            action_value=90.0,
        )],
        [_tr(phase=1), _tr(phase=1)],
        [_tr(phase=1)],
    ],
)
def test_presentation_compiler_fails_closed_for_unsupported_or_ambiguous_sets(
    operations: list[dict[str, object]],
) -> None:
    with pytest.raises(JointExactificationError, match="unsupported|duplicate"):
        compile_continuum_magnetic_presentation(operations)


def test_presentation_compiler_requires_matching_explicit_k_and_q_actions() -> None:
    c3z = _c3z()
    c3z["declared_model_action"]["q_map"] = {
        "type": "reflection",
        "axis_deg": 0.0,
        "in_model_frame": True,
    }

    with pytest.raises(JointExactificationError, match="k/Q actions"):
        compile_continuum_magnetic_presentation([_tr(phase=-1), c3z])


def test_discrete_relation_validation_accepts_v4_and_d3_actions() -> None:
    unit_blocks4 = tuple(np.eye(1, dtype=np.complex128) for _ in range(4))
    mgi2_actions = {
        "TR": BlockRouteAction("TR", True, (1, 0, 3, 2), (1, 1, 1, 1), unit_blocks4),
        "C2": BlockRouteAction("C2", False, (2, 3, 0, 1), (1, 1, 1, 1), unit_blocks4),
    }
    validate_presentation_action_relations(
        mgi2_actions,
        compile_continuum_magnetic_presentation([_tr(phase=1), _c2(phase=1)]),
    )

    unit_blocks6 = tuple(np.eye(1, dtype=np.complex128) for _ in range(6))
    mote2_actions = {
        "C3z": BlockRouteAction(
            "C3z",
            False,
            (1, 2, 0, 4, 5, 3),
            (1, 1, 1, 1, 1, 1),
            unit_blocks6,
        ),
        "C2T": BlockRouteAction(
            "C2T",
            True,
            (0, 2, 1, 3, 5, 4),
            (1, 1, 1, 1, 1, 1),
            unit_blocks6,
        ),
    }
    validate_presentation_action_relations(
        mote2_actions,
        compile_continuum_magnetic_presentation([_c3z(), _c2t()]),
    )


def test_discrete_relation_validation_rejects_nonclosing_permutations() -> None:
    blocks = tuple(np.eye(1, dtype=np.complex128) for _ in range(4))
    actions = {
        "TR": BlockRouteAction("TR", True, (1, 0, 2, 3), (1, 1, 1, 1), blocks),
        "C2": BlockRouteAction("C2", False, (0, 2, 1, 3), (1, 1, 1, 1), blocks),
    }

    with pytest.raises(JointExactificationError, match="does not close"):
        validate_presentation_action_relations(
            actions,
            compile_continuum_magnetic_presentation([_tr(phase=1), _c2(phase=1)]),
        )
