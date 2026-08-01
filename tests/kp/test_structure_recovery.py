from __future__ import annotations

import json
import numpy as np

from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    JointExactificationConfig,
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
    certify_joint_block_actions,
    compile_action_orbits,
    joint_exactify_block_actions,
    materialize_block_route_action,
)
from kp.symmetry.projection import _derive_projection_basis_frame
from kp.symmetry.structure_certificate import (
    CertificationStatus,
    FiberSectorLayout,
    StructureGrade,
    certify_common_fiber_gauge,
    derive_target_basis_hash,
)
from kp.symmetry.structure_recovery import (
    _recover_for_anchor_section,
    recover_uniform_internal_action,
)


def _mgi2_gamma_fixture() -> tuple[
    dict[str, BlockRouteAction], MagneticPresentation, FiberSectorLayout
]:
    q_count = 7
    c3 = (0, 3, 4, 5, 6, 1, 2)
    tr = (0, 4, 5, 6, 1, 2, 3)
    c2 = (0, 1, 6, 5, 4, 3, 2)

    def within_sector(permutation: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            sector * q_count + permutation[q]
            for sector in range(2)
            for q in range(q_count)
        )

    c2_permutation = tuple(
        (1 - sector) * q_count + c2[q]
        for sector in range(2)
        for q in range(q_count)
    )
    dimensions = (2,) * 14
    j = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    minus_identity = -np.eye(2, dtype=np.complex128)
    i_sigma_x = np.asarray([[0.0, 1.0j], [1.0j, 0.0]], dtype=np.complex128)
    actions = {
        "TR": BlockRouteAction(
            "TR", True, within_sector(tr), dimensions, tuple(j for _ in dimensions)
        ),
        "C3z": BlockRouteAction(
            "C3z",
            False,
            within_sector(c3),
            dimensions,
            tuple(minus_identity for _ in dimensions),
        ),
        "C2": BlockRouteAction(
            "C2", False, c2_permutation, dimensions, tuple(i_sigma_x for _ in dimensions)
        ),
    }
    presentation = MagneticPresentation(
        generators=(
            MagneticGenerator("TR", True),
            MagneticGenerator("C3z", False),
            MagneticGenerator("C2", False),
        ),
        relations=(
            MagneticRelation("TR^2", ("TR",) * 2, (), -1.0),
            MagneticRelation("C3z^3", ("C3z",) * 3, (), -1.0),
            MagneticRelation("C2^2", ("C2",) * 2, (), -1.0),
            MagneticRelation(
                "TR_C3z_commute", ("TR", "C3z"), ("C3z", "TR"), 1.0
            ),
            MagneticRelation(
                "TR_C2_commute", ("TR", "C2"), ("C2", "TR"), 1.0
            ),
            MagneticRelation(
                "C2_C3z_dihedral",
                ("C2", "C3z", "C2"),
                ("C3z", "C3z"),
                1.0,
            ),
        ),
        central_phases=(1.0, -1.0),
        source="structure-recovery-test",
    )
    layout = FiberSectorLayout(
        sector_names=("L1", "L2"),
        fiber_sectors=(0,) * 7 + (1,) * 7,
        basis_frame_hash="scattered-frame",
    )
    return actions, presentation, layout


def _gauge_actions(
    actions: dict[str, BlockRouteAction], gauge: tuple[np.ndarray, ...]
) -> dict[str, BlockRouteAction]:
    output: dict[str, BlockRouteAction] = {}
    for name, action in actions.items():
        blocks = []
        for source, target in enumerate(action.fiber_permutation):
            right = gauge[source].conjugate() if action.antiunitary else gauge[source]
            blocks.append(gauge[target].conjugate().T @ action.route_blocks[source] @ right)
        output[name] = BlockRouteAction(
            name,
            action.antiunitary,
            action.fiber_permutation,
            action.fiber_dimensions,
            tuple(blocks),
            fiber_indices=action.fiber_indices,
        )
    return output


def _deterministic_u2_gauges(count: int) -> tuple[np.ndarray, ...]:
    gauges = []
    for index in range(count):
        theta = 0.19 * index - 0.43
        phi = 0.11 * index + 0.07
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.complex128,
        )
        gauges.append(rotation @ np.diag([np.exp(1.0j * phi), np.exp(-1.0j * phi)]))
    return tuple(gauges)


def test_recovers_mgi2_uniform_structure_from_scattered_u2_gauge() -> None:
    uniform, presentation, layout = _mgi2_gamma_fixture()
    scattered = _gauge_actions(uniform, _deterministic_u2_gauges(14))

    result = recover_uniform_internal_action(
        scattered,
        presentation,
        layout=layout,
    )

    assert result.status is CertificationStatus.CERTIFIED
    assert result.grade is StructureGrade.UNIFORM_INTERNAL
    assert result.actions is not None
    assert result.fiber_gauge is not None
    assert result.common_gauge_witness is not None
    assert result.common_gauge_witness.status is CertificationStatus.CERTIFIED
    assert result.certificate is not None
    for operation in result.certificate.operations.values():
        assert operation.grade is StructureGrade.UNIFORM_INTERNAL
    for name, action in result.actions.items():
        for block in action.route_blocks:
            np.testing.assert_array_equal(block, uniform[name].route_blocks[0])
    assert result.report["selection_policy"] == "equivariant_anchor_schreier_intertwiner"
    assert np.isfinite(result.report["closest_to_source_objective"])
    standardization = result.report["sector_standardization"]
    assert standardization["status"] == "certified"
    assert standardization["cyclotomic_snap_order"] == 24
    assert set(standardization["algebraic_route_encoding"]) == {"TR", "C3z", "C2"}
    assert any(
        np.linalg.norm(result.fiber_gauge[fiber] - np.eye(2)) > 1.0e-3
        for fiber in range(14)
    )

    repeated = recover_uniform_internal_action(
        scattered,
        presentation,
        layout=layout,
    )
    assert repeated.report == result.report
    assert repeated.certificate is not None
    assert repeated.certificate.certificate_hash == result.certificate.certificate_hash
    for first, second in zip(result.fiber_gauge, repeated.fiber_gauge):
        np.testing.assert_array_equal(first, second)


def test_projection_transaction_accepts_certified_uniform_recovery() -> None:
    uniform, presentation, _layout = _mgi2_gamma_fixture()
    scattered = _gauge_actions(uniform, _deterministic_u2_gauges(14))
    joint = joint_exactify_block_actions(
        scattered,
        presentation,
        config=JointExactificationConfig(),
    )
    matrices = {
        name: materialize_block_route_action(action)
        for name, action in joint.actions.items()
    }
    reports = {
        "__joint_exactification__": {
            "metadata": dict(joint.artifact_metadata),
            "arrays": dict(joint.artifact_arrays),
            "report": dict(joint.report),
        }
    }

    frame, canonical = _derive_projection_basis_frame(
        matrices,
        reports,
        operations=[
            {"name": "TR", "antiunitary": True},
            {"name": "C3z", "antiunitary": False},
            {"name": "C2", "antiunitary": False},
        ],
        sectors=[
            {"name": "L1", "n_orb": 2, "n_q": 7},
            {"name": "L2", "n_orb": 2, "n_q": 7},
        ],
        tolerance=1.0e-10,
    )

    assert canonical is not None
    transaction = frame.components["joint_structure_transaction"]
    assert transaction["decision"] == "uniform_recovery_accepted"
    assert transaction["transition"]["is_monotone"] is True
    assert transaction["common_gauge_witness"]["status"] == "certified"
    assert frame.components["uniform_structure_recovery"]["status"] == "certified"
    for operation in transaction["post_certificate"]["operations"].values():
        assert operation["grade"] == "uniform_internal"
    json.dumps(frame.artifact(), sort_keys=True, allow_nan=False)


def test_near_algebraic_uniform_action_is_exactified_not_identity_preserved() -> None:
    uniform, presentation, _layout = _mgi2_gamma_fixture()
    theta = 1.0e-13
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.complex128,
    )
    perturbed = _gauge_actions(uniform, (rotation,) * 14)
    assert np.count_nonzero(perturbed["C2"].route_blocks[0]) == 4
    assert 0.0 < abs(perturbed["C2"].route_blocks[0][0, 0]) < 1.0e-12
    joint = joint_exactify_block_actions(
        perturbed,
        presentation,
        config=JointExactificationConfig(),
    )
    matrices = {
        name: materialize_block_route_action(action)
        for name, action in joint.actions.items()
    }
    reports = {
        "__joint_exactification__": {
            "metadata": dict(joint.artifact_metadata),
            "arrays": dict(joint.artifact_arrays),
            "report": dict(joint.report),
        }
    }

    frame, canonical = _derive_projection_basis_frame(
        matrices,
        reports,
        operations=[
            {"name": "TR", "antiunitary": True},
            {"name": "C3z", "antiunitary": False},
            {"name": "C2", "antiunitary": False},
        ],
        sectors=[
            {"name": "L1", "n_orb": 2, "n_q": 7},
            {"name": "L2", "n_orb": 2, "n_q": 7},
        ],
        tolerance=1.0e-10,
    )

    assert canonical is not None
    transaction = frame.components["joint_structure_transaction"]
    assert transaction["decision"] != "identity_preserved"
    assert transaction["common_gauge_witness"]["status"] == "certified"
    assert transaction["transition"]["is_monotone"] is True
    for name, action in uniform.items():
        np.testing.assert_array_equal(
            canonical[name],
            materialize_block_route_action(action),
        )


def test_already_uniform_recovery_is_closest_for_its_selected_algebraic_target() -> None:
    uniform, presentation, layout = _mgi2_gamma_fixture()
    sector_gauges = (
        np.asarray(
            [
                [
                    -0.9805049727803485 + 0.1508955134670114j,
                    -0.07157977509269041 - 0.10352235587647325j,
                ],
                [
                    -0.1247336798572222 - 0.01679438828333192j,
                    0.3105814740538941 + 0.9421775870853516j,
                ],
            ],
            dtype=np.complex128,
        ),
        np.asarray(
            [
                [
                    0.3794542377874721 + 0.5442726571637145j,
                    0.3499686602069566 + 0.6612894169439173j,
                ],
                [
                    -0.2533426968127031 - 0.7039880922719824j,
                    0.14696825384152565 + 0.6470074004783055j,
                ],
            ],
            dtype=np.complex128,
        ),
    )
    source = _gauge_actions(
        uniform,
        (sector_gauges[0],) * 7 + (sector_gauges[1],) * 7,
    )

    result = recover_uniform_internal_action(
        source,
        presentation,
        layout=layout,
    )

    assert result.status is CertificationStatus.CERTIFIED
    assert result.actions is not None
    assert result.fiber_gauge is not None
    assert result.report["selection_policy"] == "uniform_sector_reduction_standardization"
    assert result.report["closest_scope"] == "fixed_selected_algebraic_target"

    closest_gauge, closest_objective, _reports = _recover_for_anchor_section(
        source,
        result.actions,
        presentation,
        compile_action_orbits(source, presentation),
    )
    for fibers in (range(0, 7), range(7, 14)):
        reference = closest_gauge[fibers.start]
        for fiber in fibers:
            np.testing.assert_allclose(
                closest_gauge[fiber],
                reference,
                atol=5.0e-14,
                rtol=0.0,
            )

    target_basis_hash = derive_target_basis_hash(
        str(layout.basis_frame_hash),
        closest_gauge,
    )
    closest_witness = certify_common_fiber_gauge(
        source,
        result.actions,
        closest_gauge,
        presentation,
        source_basis_hash=str(layout.basis_frame_hash),
        target_basis_hash=target_basis_hash,
    )
    assert closest_witness.status is CertificationStatus.CERTIFIED
    assert certify_joint_block_actions(result.actions, presentation)["status"] == "certified"

    reported_objective = float(result.report["closest_to_source_objective"])
    recomputed_objective = float(
        sum(
            np.linalg.norm(value - np.eye(value.shape[0]), ord="fro") ** 2
            for value in result.fiber_gauge
        )
    )
    np.testing.assert_allclose(
        reported_objective,
        recomputed_objective,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert reported_objective <= closest_objective + 1.0e-10


def test_uniform_recovery_without_equivariant_anchor_is_unknown() -> None:
    identity = np.eye(2, dtype=np.complex128)
    action = BlockRouteAction(
        "g",
        False,
        (1, 2, 0),
        (2, 2, 2),
        (identity, identity, identity),
    )
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("g", False),),
        relations=(MagneticRelation("g^3", ("g",) * 3, (), 1.0),),
        central_phases=(1.0,),
        source="no-anchor-test",
    )
    layout = FiberSectorLayout(
        sector_names=("only",),
        fiber_sectors=(0, 0, 0),
        basis_frame_hash="no-anchor-frame",
    )

    scattered = _gauge_actions(
        {"g": action},
        _deterministic_u2_gauges(3),
    )

    result = recover_uniform_internal_action(
        scattered,
        presentation,
        layout=layout,
    )

    assert result.status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED
    assert result.actions is None
    assert result.fiber_gauge is None
    assert result.certificate is None
    assert result.common_gauge_witness is None


def test_recovers_zrs2_style_u4_uniform_action() -> None:
    c3_internal = np.diag(
        [
            np.exp(-1.0j * np.pi / 3.0),
            np.exp(1.0j * np.pi / 3.0),
            -1.0,
            -1.0,
        ]
    ).astype(np.complex128)
    uniform = {
        "C3z": BlockRouteAction(
            "C3z",
            False,
            (0, 2, 3, 1),
            (4, 4, 4, 4),
            (c3_internal,) * 4,
        )
    }
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("C3z", False),),
        relations=(MagneticRelation("C3z^3", ("C3z",) * 3, (), -1.0),),
        central_phases=(1.0, -1.0),
        source="zrs2-style-recovery-test",
    )
    layout = FiberSectorLayout(
        sector_names=("only",),
        fiber_sectors=(0, 0, 0, 0),
        basis_frame_hash="zrs2-scattered-frame",
    )
    rng = np.random.default_rng(20260717)
    gauges = []
    for _ in range(4):
        raw = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
        unitary, triangular = np.linalg.qr(raw)
        diagonal = np.diag(triangular)
        unitary = unitary @ np.diag(np.conjugate(diagonal / np.abs(diagonal)))
        gauges.append(unitary)
    scattered = _gauge_actions(uniform, tuple(gauges))

    result = recover_uniform_internal_action(
        scattered,
        presentation,
        layout=layout,
    )

    assert result.status is CertificationStatus.CERTIFIED
    assert result.certificate is not None
    assert result.certificate.operations["C3z"].grade is StructureGrade.UNIFORM_INTERNAL
    assert result.common_gauge_witness is not None
    assert result.common_gauge_witness.status is CertificationStatus.CERTIFIED
