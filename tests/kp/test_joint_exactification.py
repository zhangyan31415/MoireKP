from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import least_squares

from kp.symmetry.joint_exactification import (
    ActionOrbit,
    BlockRouteAction,
    FreeOrbitReport,
    JointExactificationError,
    JointExactificationConfig,
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
    QuotientGroupElement,
    SemilinearBlock,
    compile_action_orbits,
    compile_continuum_magnetic_presentation,
    compile_quotient_group_elements,
    compose_semilinear,
    extract_block_route_action,
    inverse_semilinear,
    materialize_block_route_action,
    project_u1_relations,
    synchronize_free_orbit,
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


def _repeated_permutation(
    local_permutation: tuple[int, ...],
    repeats: int,
) -> tuple[int, ...]:
    size = len(local_permutation)
    return tuple(
        repeat * size + local_permutation[index]
        for repeat in range(repeats)
        for index in range(size)
    )


def _unit_route_action(
    name: str,
    antiunitary: bool,
    permutation: tuple[int, ...],
) -> BlockRouteAction:
    return BlockRouteAction(
        name,
        antiunitary,
        permutation,
        tuple(1 for _ in permutation),
        tuple(np.eye(1, dtype=np.complex128) for _ in permutation),
    )


def _mgi2_v4_actions(repeats: int = 11) -> dict[str, BlockRouteAction]:
    return {
        "TR": _unit_route_action(
            "TR",
            True,
            _repeated_permutation((1, 0, 3, 2), repeats),
        ),
        "C2": _unit_route_action(
            "C2",
            False,
            _repeated_permutation((2, 3, 0, 1), repeats),
        ),
    }


def _mote2_regular_d3_actions(repeats: int = 9) -> dict[str, BlockRouteAction]:
    return {
        "C3z": _unit_route_action(
            "C3z",
            False,
            _repeated_permutation((1, 2, 0, 4, 5, 3), repeats),
        ),
        "C2T": _unit_route_action(
            "C2T",
            True,
            _repeated_permutation((3, 5, 4, 0, 2, 1), repeats),
        ),
    }


def test_v4_quotient_group_has_deterministic_canonical_words() -> None:
    presentation = compile_continuum_magnetic_presentation(
        [_tr(phase=1), _c2(phase=1)]
    )

    elements = compile_quotient_group_elements(
        _mgi2_v4_actions(repeats=1),
        presentation,
    )

    assert all(isinstance(element, QuotientGroupElement) for element in elements)
    assert tuple(element.canonical_word for element in elements) == (
        (),
        ("TR",),
        ("C2",),
        ("TR", "C2"),
    )
    assert tuple(element.antiunitary for element in elements) == (
        False,
        True,
        False,
        True,
    )


def test_mgi2_v4_compiles_eleven_free_four_fiber_orbits() -> None:
    orbits = compile_action_orbits(
        _mgi2_v4_actions(),
        compile_continuum_magnetic_presentation([_tr(phase=1), _c2(phase=1)]),
    )

    assert len(orbits) == 11
    assert all(isinstance(orbit, ActionOrbit) for orbit in orbits)
    assert tuple(orbit.root for orbit in orbits) == tuple(range(0, 44, 4))
    assert all(len(orbit.fibers) == 4 for orbit in orbits)
    assert all(orbit.stabilizer_words == ((),) for orbit in orbits)
    assert orbits[0].fibers == (0, 1, 2, 3)
    assert orbits[0].transporter_words == (
        (),
        ("TR",),
        ("C2",),
        ("TR", "C2"),
    )


def test_mote2_d3_compiles_nine_free_six_fiber_orbits() -> None:
    orbits = compile_action_orbits(
        _mote2_regular_d3_actions(),
        compile_continuum_magnetic_presentation([_c3z(), _c2t()]),
    )

    assert len(orbits) == 9
    assert tuple(orbit.root for orbit in orbits) == tuple(range(0, 54, 6))
    assert all(len(orbit.fibers) == 6 for orbit in orbits)
    assert all(orbit.stabilizer_words == ((),) for orbit in orbits)
    assert orbits[0].transporter_words == (
        (),
        ("C3z",),
        ("C3z", "C3z"),
        ("C2T",),
        ("C3z", "C2T"),
        ("C2T", "C3z"),
    )


def _gamma_d3_times_tr_actions() -> dict[str, BlockRouteAction]:
    tr_local = tuple((index + 6) % 12 for index in range(12))
    c3_local: list[int] = []
    c2_local: list[int] = []
    for time_sector in range(2):
        for reflection_sector in range(2):
            for rotation_sector in range(3):
                c3_local.append(
                    time_sector * 6
                    + reflection_sector * 3
                    + (rotation_sector + 1) % 3
                )
                c2_local.append(
                    time_sector * 6
                    + (1 - reflection_sector) * 3
                    + (-rotation_sector) % 3
                )
    tr = (*_repeated_permutation(tr_local, 3), 37, 36)
    c3 = (*_repeated_permutation(tuple(c3_local), 3), 36, 37)
    c2 = (*_repeated_permutation(tuple(c2_local), 3), 36, 37)
    return {
        "TR": _unit_route_action("TR", True, tr),
        "C3z": _unit_route_action("C3z", False, c3),
        "C2": _unit_route_action("C2", False, c2),
    }


def test_gamma_d3_times_tr_has_three_free_orbits_and_one_stabilized_orbit() -> None:
    orbits = compile_action_orbits(
        _gamma_d3_times_tr_actions(),
        compile_continuum_magnetic_presentation(
            [_tr(phase=-1), _c3z(), _c2(phase=-1)]
        ),
    )

    assert tuple(orbit.root for orbit in orbits) == (0, 12, 24, 36)
    assert tuple(len(orbit.fibers) for orbit in orbits) == (12, 12, 12, 2)
    assert tuple(len(orbit.stabilizer_words) for orbit in orbits) == (1, 1, 1, 6)
    assert orbits[-1].fibers == (36, 37)
    assert orbits[-1].transporter_words == ((), ("TR",))
    assert all("TR" not in word for word in orbits[-1].stabilizer_words)


def test_quotient_group_compiler_enforces_explicit_limits() -> None:
    actions = _mgi2_v4_actions(repeats=1)
    presentation = compile_continuum_magnetic_presentation(
        [_tr(phase=1), _c2(phase=1)]
    )

    with pytest.raises(JointExactificationError, match="group-size limit"):
        compile_quotient_group_elements(
            actions,
            presentation,
            max_group_size=3,
        )
    with pytest.raises(JointExactificationError, match="word-length limit"):
        compile_quotient_group_elements(
            actions,
            presentation,
            max_word_length=1,
        )


def _mgi2_measured_u1_defect_actions() -> dict[str, BlockRouteAction]:
    tr_permutation = (
        14, 8, 12, 18, 15, 11, 19, 16, 1, 17, 21, 5, 2, 20, 0, 4, 7,
        9, 3, 6, 13, 10, 23, 22, 34, 40, 38, 33, 37, 43, 41, 39, 36,
        27, 24, 42, 32, 28, 26, 31, 25, 30, 35, 29,
    )
    c2_permutation = (
        23, 26, 30, 24, 27, 31, 28, 32, 38, 40, 42, 39, 41, 43, 22, 33,
        36, 25, 34, 37, 29, 35, 14, 0, 3, 17, 1, 4, 6, 20, 2, 5, 7,
        15, 18, 21, 16, 19, 8, 11, 9, 12, 10, 13,
    )
    positive_pi_defects = (
        8.573551341761743e-06,
        8.715867566788660e-06,
        8.794147711821410e-06,
        8.111774012053985e-06,
        8.317126971135735e-06,
        8.482813055987748e-06,
        8.260437259099973e-06,
        8.563159882957194e-06,
        8.727936607222375e-06,
        8.331784509429951e-06,
        7.875880483787512e-06,
        8.755444839803062e-06,
        8.435707563148043e-06,
        8.325201826586692e-06,
        8.941317854116448e-06,
        9.025485210756301e-06,
        9.244602727331852e-06,
        9.043688192544863e-06,
        9.211630187522246e-06,
        9.492111233466716e-06,
        9.396686003082522e-06,
        9.310424010422480e-06,
    )
    c2_angles = np.empty(44, dtype=np.float64)
    for source, defect in enumerate(positive_pi_defects):
        target = c2_permutation[source]
        c2_angles[source] = -np.pi + defect
        c2_angles[target] = np.pi - defect
    dimensions = tuple(1 for _ in range(44))
    return {
        "TR": BlockRouteAction(
            "TR",
            True,
            tr_permutation,
            dimensions,
            tuple(np.ones((1, 1), dtype=np.complex128) for _ in range(44)),
        ),
        "C2": BlockRouteAction(
            "C2",
            False,
            c2_permutation,
            dimensions,
            tuple(
                np.asarray([[np.exp(1.0j * angle)]], dtype=np.complex128)
                for angle in c2_angles
            ),
        ),
    }


def test_u1_projection_reproduces_measured_mgi2_joint_correction() -> None:
    projected, report = project_u1_relations(
        _mgi2_measured_u1_defect_actions(),
        compile_continuum_magnetic_presentation([_tr(phase=1), _c2(phase=1)]),
    )

    assert report["constraint_shape"] == (132, 88)
    assert report["rank"] == 55
    assert report["nullity"] == 33
    assert report["correction_rms_by_operation"]["TR"] == pytest.approx(
        4.3625209915e-6,
        rel=2.0e-10,
    )
    assert report["correction_rms_by_operation"]["C2"] == pytest.approx(
        4.3625209915e-6,
        rel=2.0e-10,
    )
    assert report["maximum_phase_correction"] == pytest.approx(
        4.4519406526e-6,
        rel=2.0e-10,
    )
    assert report["pre_relation_residuals"]["TR_C2_commute"]["rms"] == pytest.approx(
        1.7450083966e-5,
        rel=2.0e-10,
    )
    assert report["post_relation_residual_max"] < 5.0e-15
    assert set(projected) == {"TR", "C2"}


def test_u1_projection_is_idempotent_at_the_floating_floor() -> None:
    presentation = compile_continuum_magnetic_presentation(
        [_tr(phase=1), _c2(phase=1)]
    )
    projected, _ = project_u1_relations(
        _mgi2_measured_u1_defect_actions(),
        presentation,
    )

    repeated, report = project_u1_relations(projected, presentation)

    assert report["maximum_phase_correction"] < 5.0e-15
    assert report["post_relation_residual_max"] < 5.0e-15
    for name in projected:
        for left, right in zip(
            repeated[name].route_blocks,
            projected[name].route_blocks,
        ):
            np.testing.assert_allclose(left, right, atol=5.0e-15, rtol=0.0)


def test_u1_projection_rejects_non_scalar_route_blocks() -> None:
    blocks = tuple(np.eye(2, dtype=np.complex128) for _ in range(4))
    actions = {
        "TR": BlockRouteAction("TR", True, (1, 0, 3, 2), (2, 2, 2, 2), blocks),
        "C2": BlockRouteAction("C2", False, (2, 3, 0, 1), (2, 2, 2, 2), blocks),
    }

    with pytest.raises(JointExactificationError, match=r"U\(1\)"):
        project_u1_relations(
            actions,
            compile_continuum_magnetic_presentation([_tr(phase=1), _c2(phase=1)]),
        )


def _random_unitary(rng: np.random.Generator, dimension: int) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return np.asarray(q @ np.diag(phases.conj()), dtype=np.complex128)


def _normalized_random_skew(
    rng: np.random.Generator,
    dimension: int,
    amplitude: float,
) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(
        size=(dimension, dimension)
    )
    skew = 0.5 * (raw - raw.conj().T)
    norm = np.linalg.norm(skew)
    return np.asarray(amplitude * skew / norm, dtype=np.complex128)


def _mote2_free_ud_actions(
    dimension: int,
    *,
    perturbation: float,
    seed: int,
) -> dict[str, BlockRouteAction]:
    rng = np.random.default_rng(seed)
    permutations = {
        "C3z": (1, 2, 0, 4, 5, 3),
        "C2T": (3, 5, 4, 0, 2, 1),
    }
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    c3_two = np.diag(np.exp(1.0j * np.asarray([-np.pi / 3.0, np.pi / 3.0])))
    c2t_two = 1.0j * sigma_z
    copy_count = dimension // 2
    generator_blocks = {
        "C3z": np.kron(np.eye(copy_count), c3_two),
        "C2T": np.kron(np.eye(copy_count), c2t_two),
    }
    fiber_gauge = tuple(_random_unitary(rng, dimension) for _ in range(6))
    actions: dict[str, BlockRouteAction] = {}
    for name, antiunitary in (("C3z", False), ("C2T", True)):
        route_blocks: list[np.ndarray] = []
        for source, target in enumerate(permutations[name]):
            source_gauge = fiber_gauge[source].conj() if antiunitary else fiber_gauge[source]
            block = (
                fiber_gauge[target].conj().T
                @ generator_blocks[name]
                @ source_gauge
            )
            if perturbation:
                block = (
                    expm(
                        _normalized_random_skew(
                            rng,
                            dimension,
                            perturbation,
                        )
                    )
                    @ block
                )
            route_blocks.append(np.asarray(block, dtype=np.complex128))
        actions[name] = BlockRouteAction(
            name,
            antiunitary,
            permutations[name],
            tuple(dimension for _ in range(6)),
            tuple(route_blocks),
        )
    return actions


def _replace_route_blocks(
    actions: dict[str, BlockRouteAction],
    blocks: dict[str, tuple[np.ndarray, ...]],
) -> dict[str, BlockRouteAction]:
    return {
        name: BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=blocks[name],
            fiber_indices=action.fiber_indices,
        )
        for name, action in actions.items()
    }


def _test_word_block(
    actions: dict[str, BlockRouteAction],
    word: tuple[str, ...],
    source: int,
) -> np.ndarray:
    dimension = actions[next(iter(actions))].fiber_dimensions[source]
    block = np.eye(dimension, dtype=np.complex128)
    current = source
    for name in reversed(word):
        action = actions[name]
        inner = block.conj() if action.antiunitary else block
        block = action.route_blocks[current] @ inner
        current = action.fiber_permutation[current]
    return block


def _maximum_matrix_relation_residual(
    actions: dict[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> float:
    maximum = 0.0
    fiber_count = len(next(iter(actions.values())).fiber_dimensions)
    for relation in presentation.relations:
        for source in range(fiber_count):
            lhs = _test_word_block(actions, relation.lhs, source)
            rhs = _test_word_block(actions, relation.rhs, source)
            residual = np.linalg.norm(lhs - relation.central_phase * rhs) / np.sqrt(
                lhs.shape[0]
            )
            maximum = max(maximum, float(residual))
    return maximum


@pytest.mark.parametrize("dimension", [2, 4])
def test_free_orbit_ud_synchronization_restores_all_magnetic_relations(
    dimension: int,
) -> None:
    presentation = compile_continuum_magnetic_presentation([_c3z(), _c2t()])
    actions = _mote2_free_ud_actions(
        dimension,
        perturbation=8.0e-7,
        seed=9000 + dimension,
    )
    orbit = compile_action_orbits(actions, presentation)[0]

    blocks, report = synchronize_free_orbit(
        actions,
        presentation,
        orbit,
        config=JointExactificationConfig(max_iterations=30),
    )
    synchronized = _replace_route_blocks(actions, blocks)

    assert isinstance(report, FreeOrbitReport)
    assert report.converged is True
    assert report.block_dimension == dimension
    assert report.objective_final <= report.objective_initial
    assert report.route_correction_rms < 5.0e-6
    assert report.route_correction_max < 5.0e-6
    assert report.relation_residual_max < 5.0e-12
    assert _maximum_matrix_relation_residual(synchronized, presentation) < 5.0e-12
    assert report.central_transition_counts["-1"] > 0
    for action in synchronized.values():
        for block in action.route_blocks:
            np.testing.assert_allclose(
                block.conj().T @ block,
                np.eye(dimension),
                atol=5.0e-13,
                rtol=0.0,
            )


def test_free_orbit_ud_synchronization_is_idempotent_for_exact_input() -> None:
    presentation = compile_continuum_magnetic_presentation([_c3z(), _c2t()])
    actions = _mote2_free_ud_actions(4, perturbation=0.0, seed=9104)
    orbit = compile_action_orbits(actions, presentation)[0]

    blocks, report = synchronize_free_orbit(
        actions,
        presentation,
        orbit,
        config=JointExactificationConfig(),
    )

    assert report.route_correction_rms < 5.0e-13
    assert report.route_correction_max < 5.0e-13
    for name, action in actions.items():
        for actual, expected in zip(blocks[name], action.route_blocks):
            np.testing.assert_allclose(actual, expected, atol=5.0e-13, rtol=0.0)


def _gauge_transform_actions(
    actions: dict[str, BlockRouteAction],
    gauge: tuple[np.ndarray, ...],
) -> dict[str, BlockRouteAction]:
    transformed: dict[str, BlockRouteAction] = {}
    for name, action in actions.items():
        blocks = []
        for source, target in enumerate(action.fiber_permutation):
            source_gauge = gauge[source].conj() if action.antiunitary else gauge[source]
            blocks.append(
                gauge[target].conj().T @ action.route_blocks[source] @ source_gauge
            )
        transformed[name] = BlockRouteAction(
            name,
            action.antiunitary,
            action.fiber_permutation,
            action.fiber_dimensions,
            tuple(blocks),
        )
    return transformed


def test_free_orbit_ud_synchronization_is_fiber_gauge_covariant() -> None:
    presentation = compile_continuum_magnetic_presentation([_c3z(), _c2t()])
    actions = _mote2_free_ud_actions(2, perturbation=7.0e-7, seed=9202)
    orbit = compile_action_orbits(actions, presentation)[0]
    blocks, _ = synchronize_free_orbit(
        actions,
        presentation,
        orbit,
        config=JointExactificationConfig(max_iterations=30),
    )
    synchronized = _replace_route_blocks(actions, blocks)
    rng = np.random.default_rng(9203)
    gauge = tuple(_random_unitary(rng, 2) for _ in range(6))
    gauged_actions = _gauge_transform_actions(actions, gauge)
    gauged_orbit = compile_action_orbits(gauged_actions, presentation)[0]

    gauged_blocks, _ = synchronize_free_orbit(
        gauged_actions,
        presentation,
        gauged_orbit,
        config=JointExactificationConfig(max_iterations=30),
    )
    expected = _gauge_transform_actions(synchronized, gauge)

    for name in expected:
        for actual, target in zip(gauged_blocks[name], expected[name].route_blocks):
            np.testing.assert_allclose(actual, target, atol=5.0e-10, rtol=0.0)


def test_free_orbit_u1_path_matches_direct_phase_oracle() -> None:
    actions = _mgi2_measured_u1_defect_actions()
    presentation = compile_continuum_magnetic_presentation(
        [_tr(phase=1), _c2(phase=1)]
    )
    orbit = compile_action_orbits(actions, presentation)[0]
    direct, _ = project_u1_relations(actions, presentation)

    blocks, report = synchronize_free_orbit(
        actions,
        presentation,
        orbit,
        config=JointExactificationConfig(),
    )

    assert report.solver == "u1_relation_projection"
    for name in actions:
        for source in orbit.fibers:
            np.testing.assert_allclose(
                blocks[name][source],
                direct[name].route_blocks[source],
                atol=2.0e-15,
                rtol=0.0,
            )


def test_free_orbit_central_transition_requires_configured_separation() -> None:
    presentation = compile_continuum_magnetic_presentation([_c3z(), _c2t()])
    actions = _mote2_free_ud_actions(2, perturbation=1.0e-7, seed=9302)
    orbit = compile_action_orbits(actions, presentation)[0]

    with pytest.raises(JointExactificationError, match="central.*separation"):
        synchronize_free_orbit(
            actions,
            presentation,
            orbit,
            config=JointExactificationConfig(central_branch_margin=3.0),
        )


def _test_skew_basis(dimension: int) -> tuple[np.ndarray, ...]:
    basis: list[np.ndarray] = []
    for row in range(dimension):
        value = np.zeros((dimension, dimension), dtype=np.complex128)
        value[row, row] = 1.0j
        basis.append(value)
    for row in range(dimension):
        for column in range(row + 1, dimension):
            real = np.zeros((dimension, dimension), dtype=np.complex128)
            real[row, column] = 1.0 / np.sqrt(2.0)
            real[column, row] = -1.0 / np.sqrt(2.0)
            basis.append(real)
            imaginary = np.zeros((dimension, dimension), dtype=np.complex128)
            imaginary[row, column] = 1.0j / np.sqrt(2.0)
            imaginary[column, row] = 1.0j / np.sqrt(2.0)
            basis.append(imaginary)
    return tuple(basis)


def _independent_free_orbit_objective(
    actions: dict[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
) -> float:
    dimension = actions["C3z"].fiber_dimensions[orbit.root]
    frames = {
        fiber: _test_word_block(actions, word, orbit.root)
        for fiber, word in zip(orbit.fibers, orbit.transporter_words)
    }
    central: dict[tuple[str, int], complex] = {}
    identity = np.eye(dimension, dtype=np.complex128)
    for generator in presentation.generators:
        action = actions[generator.name]
        for source in orbit.fibers:
            target = action.fiber_permutation[source]
            source_frame = frames[source].conj() if action.antiunitary else frames[source]
            holonomy = (
                frames[target].conj().T
                @ action.route_blocks[source]
                @ source_frame
            )
            central[(generator.name, source)] = min(
                presentation.central_phases,
                key=lambda phase: np.linalg.norm(holonomy - phase * identity),
            )
    variables = tuple(fiber for fiber in orbit.fibers if fiber != orbit.root)
    basis = _test_skew_basis(dimension)
    parameter_count = len(variables) * len(basis)

    def residual(parameters: np.ndarray) -> np.ndarray:
        trial = {orbit.root: identity}
        for fiber_index, fiber in enumerate(variables):
            start = fiber_index * len(basis)
            generator = sum(
                parameters[start + basis_index] * value
                for basis_index, value in enumerate(basis)
            )
            trial[fiber] = expm(generator) @ frames[fiber]
        pieces = []
        for magnetic_generator in presentation.generators:
            action = actions[magnetic_generator.name]
            for source in orbit.fibers:
                target = action.fiber_permutation[source]
                source_inverse = trial[source].conj().T
                if action.antiunitary:
                    source_inverse = source_inverse.conj()
                predicted = (
                    central[(magnetic_generator.name, source)]
                    * trial[target]
                    @ source_inverse
                )
                difference = predicted - action.route_blocks[source]
                pieces.extend((difference.real.ravel(), difference.imag.ravel()))
        return np.concatenate(pieces)

    result = least_squares(
        residual,
        np.zeros(parameter_count, dtype=np.float64),
        jac="2-point",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=300,
    )
    assert result.success
    return float(np.dot(result.fun, result.fun))


def test_free_orbit_analytic_solver_matches_independent_numerical_oracle() -> None:
    presentation = compile_continuum_magnetic_presentation([_c3z(), _c2t()])
    actions = _mote2_free_ud_actions(2, perturbation=8.0e-7, seed=9402)
    orbit = compile_action_orbits(actions, presentation)[0]
    reference_objective = _independent_free_orbit_objective(
        actions,
        presentation,
        orbit,
    )

    _, report = synchronize_free_orbit(
        actions,
        presentation,
        orbit,
        config=JointExactificationConfig(max_iterations=30),
    )

    comparison_bound = 1.0e-18 + 1.0e-7 * reference_objective
    assert report.objective_final <= reference_objective + comparison_bound
