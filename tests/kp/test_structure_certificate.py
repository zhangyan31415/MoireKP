from __future__ import annotations

import json

import numpy as np
import pytest

from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    MagneticGenerator,
    MagneticPresentation,
)
from kp.symmetry.structure_certificate import (
    CertificationStatus,
    FiberSectorLayout,
    StructureGrade,
    certify_common_fiber_gauge,
    certify_joint_structure,
    compare_joint_structure,
    derive_target_basis_hash,
)


def _presentation(*generators: tuple[str, bool]) -> MagneticPresentation:
    return MagneticPresentation(
        generators=tuple(
            MagneticGenerator(name=name, antiunitary=antiunitary)
            for name, antiunitary in generators
        ),
        relations=(),
        central_phases=(1.0,),
        source="structure-certificate-test",
    )


def _action(
    name: str,
    blocks: tuple[np.ndarray, ...],
    *,
    antiunitary: bool = False,
    permutation: tuple[int, ...] | None = None,
) -> BlockRouteAction:
    fiber_count = len(blocks)
    return BlockRouteAction(
        name=name,
        antiunitary=antiunitary,
        fiber_permutation=(
            tuple(range(fiber_count)) if permutation is None else permutation
        ),
        fiber_dimensions=(int(np.asarray(blocks[0]).shape[0]),) * fiber_count,
        route_blocks=blocks,
    )


def _single_sector_layout(
    fiber_count: int,
    *,
    basis_frame_hash: str = "source-frame",
) -> FiberSectorLayout:
    return FiberSectorLayout(
        sector_names=("only",),
        fiber_sectors=(0,) * fiber_count,
        basis_frame_hash=basis_frame_hash,
    )


def _sector(certificate, operation: str, source_sector: str):
    return certificate.operations[operation].source_sectors[source_sector]


def _strict_json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def test_structure_grades_uniform_projective_and_general_routes() -> None:
    identity = np.eye(2, dtype=np.complex128)
    skew = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    presentation = _presentation(
        ("uniform", False),
        ("projective", False),
        ("general", False),
    )
    actions = {
        "uniform": _action("uniform", (identity, identity, identity)),
        "projective": _action(
            "projective",
            (identity, 1.0j * identity, -identity),
        ),
        "general": _action("general", (identity, skew, identity)),
    }

    certificate = certify_joint_structure(
        actions,
        presentation,
        layout=_single_sector_layout(3),
    )

    assert certificate.status is CertificationStatus.CERTIFIED
    assert _sector(certificate, "uniform", "only").grade is StructureGrade.UNIFORM_INTERNAL
    assert (
        _sector(certificate, "projective", "only").grade
        is StructureGrade.PROJECTIVELY_UNIFORM
    )
    assert _sector(certificate, "general", "only").grade is StructureGrade.GENERAL_UD_ROUTE
    assert _sector(certificate, "uniform", "only").uniform_residual_max == 0.0
    assert _sector(certificate, "projective", "only").projective_residual_max == 0.0
    general = _sector(certificate, "general", "only")
    assert general.minimum_overlap_magnitude == 0.0
    np.testing.assert_allclose(general.projective_residual_max, np.sqrt(2.0))
    assert general.relative_phases[1] is None
    _strict_json(certificate.artifact())


def test_common_gauge_witness_uses_conjugated_source_frame_for_antiunitary() -> None:
    skew = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    permutation = (1, 0)
    before = {
        "TR": _action(
            "TR",
            (skew, skew),
            antiunitary=True,
            permutation=permutation,
        )
    }
    gauges = (
        np.diag([np.exp(0.31j), np.exp(-0.17j)]),
        np.asarray(
            [[np.cos(0.23), 1.0j * np.sin(0.23)],
             [1.0j * np.sin(0.23), np.cos(0.23)]],
            dtype=np.complex128,
        ),
    )
    after_blocks = tuple(
        gauges[target].conjugate().T @ skew @ gauges[source].conjugate()
        for source, target in enumerate(permutation)
    )
    after = {
        "TR": _action(
            "TR",
            after_blocks,
            antiunitary=True,
            permutation=permutation,
        )
    }
    presentation = _presentation(("TR", True))

    target_basis_hash = derive_target_basis_hash("before", gauges)
    witness = certify_common_fiber_gauge(
        before,
        after,
        gauges,
        presentation,
        source_basis_hash="before",
        target_basis_hash=target_basis_hash,
    )

    assert witness.status is CertificationStatus.CERTIFIED
    assert witness.source_basis_hash == "before"
    assert witness.target_basis_hash == target_basis_hash
    assert witness.max_residual < witness.certification_bound
    _strict_json(witness.artifact())


def test_monotone_transition_rejects_uniform_to_projective_or_general() -> None:
    identity = np.eye(2, dtype=np.complex128)
    presentation = _presentation(("g", False))
    layout = _single_sector_layout(3)
    permutation = (1, 2, 0)
    before_actions = {
        "g": _action(
            "g",
            (identity, identity, identity),
            permutation=permutation,
        )
    }
    projective_gauge = tuple(
        phase * identity for phase in (1.0, -1.0j, -1.0)
    )
    general_gauge = (
        identity,
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.diag([1.0j, -1.0j]).astype(np.complex128),
    )

    for gauges, expected_grade in (
        (projective_gauge, StructureGrade.PROJECTIVELY_UNIFORM),
        (general_gauge, StructureGrade.GENERAL_UD_ROUTE),
    ):
        after_blocks = tuple(
            gauges[target].conjugate().T @ gauges[source]
            for source, target in enumerate(permutation)
        )
        after_actions = {
            "g": _action("g", after_blocks, permutation=permutation)
        }
        before = certify_joint_structure(
            before_actions,
            presentation,
            layout=_single_sector_layout(3, basis_frame_hash="before"),
        )
        target_basis_hash = derive_target_basis_hash("before", gauges)
        after = certify_joint_structure(
            after_actions,
            presentation,
            layout=_single_sector_layout(
                3,
                basis_frame_hash=target_basis_hash,
            ),
        )
        witness = certify_common_fiber_gauge(
            before_actions,
            after_actions,
            gauges,
            presentation,
            source_basis_hash="before",
            target_basis_hash=target_basis_hash,
        )

        transition = compare_joint_structure(before, after, gauge_witness=witness)

        assert _sector(after, "g", "only").grade is expected_grade
        assert witness.status is CertificationStatus.CERTIFIED
        assert transition.status is CertificationStatus.CERTIFIED_INFEASIBLE
        assert not transition.is_monotone
        assert transition.regressions == (("g", "only"),)
        _strict_json(transition.artifact())


def test_sector_wide_common_conjugation_preserves_structure_grade() -> None:
    diagonal = np.diag([1.0, -1.0]).astype(np.complex128)
    angle = 0.37
    common = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    conjugated = common.conjugate().T @ diagonal @ common
    before_actions = {"C2": _action("C2", (diagonal,) * 3)}
    after_actions = {"C2": _action("C2", (conjugated,) * 3)}
    presentation = _presentation(("C2", False))
    before = certify_joint_structure(
        before_actions,
        presentation,
        layout=_single_sector_layout(3, basis_frame_hash="before"),
    )
    target_basis_hash = derive_target_basis_hash("before", (common,) * 3)
    after = certify_joint_structure(
        after_actions,
        presentation,
        layout=_single_sector_layout(3, basis_frame_hash=target_basis_hash),
    )
    witness = certify_common_fiber_gauge(
        before_actions,
        after_actions,
        (common,) * 3,
        presentation,
        source_basis_hash="before",
        target_basis_hash=target_basis_hash,
    )

    transition = compare_joint_structure(before, after, gauge_witness=witness)

    assert witness.status is CertificationStatus.CERTIFIED
    assert transition.status is CertificationStatus.CERTIFIED
    assert transition.is_monotone
    assert transition.regressions == ()
    assert (
        _sector(before, "C2", "only").grade
        is _sector(after, "C2", "only").grade
        is StructureGrade.UNIFORM_INTERNAL
    )


def test_common_gauge_witness_rejects_nonunitary_fiber_gauge() -> None:
    identity = np.eye(2, dtype=np.complex128)
    actions = {"g": _action("g", (identity, identity))}
    presentation = _presentation(("g", False))
    nonunitary = np.diag([1.0, 0.5]).astype(np.complex128)

    with pytest.raises(ValueError, match="unitar"):
        certify_common_fiber_gauge(
            actions,
            actions,
            (identity, nonunitary),
            presentation,
            source_basis_hash="before-frame",
            target_basis_hash="after-frame",
        )


@pytest.mark.parametrize(
    ("before_basis_hash", "after_basis_hash"),
    (
        ("wrong-before-frame", "witness-frame"),
        ("witness-frame", "wrong-after-frame"),
    ),
)
def test_structure_transition_binds_witness_to_certificate_basis_hashes(
    before_basis_hash: str,
    after_basis_hash: str,
) -> None:
    identity = np.eye(2, dtype=np.complex128)
    actions = {"g": _action("g", (identity, identity))}
    presentation = _presentation(("g", False))
    before = certify_joint_structure(
        actions,
        presentation,
        layout=FiberSectorLayout(
            sector_names=("only",),
            fiber_sectors=(0, 0),
            basis_frame_hash=before_basis_hash,
        ),
    )
    after = certify_joint_structure(
        actions,
        presentation,
        layout=FiberSectorLayout(
            sector_names=("only",),
            fiber_sectors=(0, 0),
            basis_frame_hash=after_basis_hash,
        ),
    )
    witness = certify_common_fiber_gauge(
        actions,
        actions,
        (identity, identity),
        presentation,
        source_basis_hash="witness-frame",
        target_basis_hash="witness-frame",
    )

    transition = compare_joint_structure(before, after, gauge_witness=witness)

    assert witness.status is CertificationStatus.CERTIFIED
    assert transition.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert not transition.is_monotone
    assert "basis" in (transition.reason or "")
    _strict_json(transition.artifact())


def test_common_gauge_witness_rejects_caller_declared_unrelated_target_hash() -> None:
    identity = np.eye(2, dtype=np.complex128)
    actions = {"g": _action("g", (identity, identity))}
    presentation = _presentation(("g", False))

    with pytest.raises(ValueError, match="target basis hash"):
        certify_common_fiber_gauge(
            actions,
            actions,
            (identity, identity),
            presentation,
            source_basis_hash="source-frame",
            target_basis_hash="unrelated-frame",
        )


def test_common_gauge_witness_checks_each_fiber_against_its_own_bound() -> None:
    identity = np.eye(2, dtype=np.complex128)
    before = {"g": _action("g", (identity, identity))}
    after = {"g": _action("g", (-identity, identity))}
    presentation = _presentation(("g", False))

    witness = certify_common_fiber_gauge(
        before,
        after,
        (identity, identity),
        presentation,
        source_basis_hash="same-frame",
        target_basis_hash="same-frame",
        absolute_error_bounds={"g": (0.0, 10.0)},
    )

    assert witness.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert witness.residual_by_operation_fiber["g"][0] > (
        witness.certification_bound_by_operation_fiber["g"][0]
    )
    assert witness.residual_by_operation_fiber["g"][1] <= (
        witness.certification_bound_by_operation_fiber["g"][1]
    )
    _strict_json(witness.artifact())


def test_certificate_hash_changes_when_fiber_permutation_changes() -> None:
    identity = np.eye(2, dtype=np.complex128)
    presentation = _presentation(("g", False))
    layout = _single_sector_layout(3)
    fixed_routes = certify_joint_structure(
        {"g": _action("g", (identity,) * 3, permutation=(0, 1, 2))},
        presentation,
        layout=layout,
    )
    permuted_routes = certify_joint_structure(
        {"g": _action("g", (identity,) * 3, permutation=(1, 2, 0))},
        presentation,
        layout=layout,
    )

    assert fixed_routes.certificate_hash != permuted_routes.certificate_hash
    _strict_json(fixed_routes.artifact())
    _strict_json(permuted_routes.artifact())


def test_multiple_target_sector_is_unknown_and_strictly_json_serializable() -> None:
    scalar_identity = np.ones((1, 1), dtype=np.complex128)
    presentation = _presentation(("g", False))
    layout = FiberSectorLayout(
        sector_names=("A", "B"),
        fiber_sectors=(0, 0, 1, 1),
        basis_frame_hash="source-frame",
    )
    certificate = certify_joint_structure(
        {
            "g": _action(
                "g",
                (scalar_identity,) * 4,
                permutation=(0, 2, 1, 3),
            )
        },
        presentation,
        layout=layout,
    )

    assert certificate.status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED
    for sector_name in layout.sector_names:
        row = _sector(certificate, "g", sector_name)
        assert row.status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED
        assert row.grade is None
        assert row.target_sector is None
        assert row.obstruction_witness == {"kind": "multiple_target_sectors"}
    encoded = _strict_json(certificate.artifact())
    assert "Infinity" not in encoded
    assert "NaN" not in encoded
    assert len(certificate.certificate_hash) == 64


def test_certificate_and_witness_strictly_validate_action_identity() -> None:
    identity = np.eye(2, dtype=np.complex128)
    presentation = _presentation(("g", False))
    layout = _single_sector_layout(2)
    wrongly_named = {"g": _action("not-g", (identity, identity))}

    with pytest.raises(ValueError, match="name"):
        certify_joint_structure(wrongly_named, presentation, layout=layout)

    with pytest.raises(ValueError, match="name"):
        certify_common_fiber_gauge(
            wrongly_named,
            wrongly_named,
            (identity, identity),
            presentation,
            source_basis_hash="before-frame",
            target_basis_hash="after-frame",
        )


def test_common_gauge_witness_validates_presentation_antiunitary_parity() -> None:
    identity = np.eye(2, dtype=np.complex128)
    unitary_actions = {"TR": _action("TR", (identity, identity))}
    antiunitary_presentation = _presentation(("TR", True))

    with pytest.raises(ValueError, match="antiunitary"):
        certify_common_fiber_gauge(
            unitary_actions,
            unitary_actions,
            (identity, identity),
            antiunitary_presentation,
            source_basis_hash="before-frame",
            target_basis_hash="after-frame",
        )


def test_uniform_projective_boundary_is_reported_as_unknown() -> None:
    scalar_identity = np.ones((1, 1), dtype=np.complex128)
    absolute_error = 1.0e-6
    # The observed uniform residual lies only a few ulps above its propagated
    # bound.  It is exactly projective, but the data do not decisively rule out
    # a uniform representative, so the stronger-vs-weaker grade is ambiguous.
    boundary_phase = np.exp(1.0j * (2.0 * absolute_error + 5.0e-14))
    presentation = _presentation(("g", False))
    certificate = certify_joint_structure(
        {
            "g": _action(
                "g",
                (scalar_identity, boundary_phase * scalar_identity),
            )
        },
        presentation,
        layout=_single_sector_layout(2),
        block_absolute_error_bounds={"g": (absolute_error, absolute_error)},
    )
    row = _sector(certificate, "g", "only")

    assert row.status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED
    assert row.grade is None
    assert certificate.status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED
    _strict_json(certificate.artifact())


def test_source_sectors_are_classified_independently() -> None:
    identity = np.eye(2, dtype=np.complex128)
    presentation = _presentation(("g", False))
    layout = FiberSectorLayout(
        sector_names=("A", "B"),
        fiber_sectors=(0, 0, 1, 1),
        basis_frame_hash="source-frame",
    )
    certificate = certify_joint_structure(
        {
            "g": _action(
                "g",
                (identity, identity, identity, 1.0j * identity),
            )
        },
        presentation,
        layout=layout,
    )

    sector_a = _sector(certificate, "g", "A")
    sector_b = _sector(certificate, "g", "B")
    assert sector_a.source_fibers == (0, 1)
    assert sector_a.reference_fiber == 0
    assert sector_a.grade is StructureGrade.UNIFORM_INTERNAL
    assert sector_b.source_fibers == (2, 3)
    assert sector_b.reference_fiber == 2
    assert sector_b.grade is StructureGrade.PROJECTIVELY_UNIFORM
    assert certificate.operations["g"].grade is StructureGrade.PROJECTIVELY_UNIFORM
    _strict_json(certificate.artifact())
