from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from kp.symmetry.canonical_target import (
    CANONICAL_TARGET_ACTIONS_V1,
    LITERAL_ROUTE_SOURCE_ACTIONS_V1,
    CanonicalTargetError,
    certify_fixed_target_exactification,
    decode_canonical_target_actions,
    encode_canonical_target_actions,
    encode_literal_route_source_actions,
    load_literal_route_source_artifact,
)
from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    MagneticGenerator,
    MagneticPresentation,
    materialize_block_route_action,
)


def _certificate() -> dict[str, str]:
    return {
        "status": "certified",
        "basis_frame_hash": "basis-frame",
        "layout_hash": "layout",
        "presentation_hash": "presentation",
        "declared_cocycle_hash": "cocycle",
        "actions_hash": "actions",
    }


def _encoding_fixture() -> tuple[
    dict[str, BlockRouteAction], MagneticPresentation
]:
    half_sqrt_three = float(np.sqrt(3.0) / 2.0)
    root_4 = complex(0.5, half_sqrt_three)
    root_8 = complex(-0.5, half_sqrt_three)
    indices = ((0, 2), (1, 3))
    dimensions = (2, 2)
    actions = {
        "C3z": BlockRouteAction(
            "C3z",
            False,
            (1, 0),
            dimensions,
            (
                np.diag([root_4, complex(0.0, -1.0)]),
                np.asarray([[0.0, root_8], [-1.0, 0.0]], dtype=np.complex128),
            ),
            fiber_indices=indices,
        ),
        "TR": BlockRouteAction(
            "TR",
            True,
            (0, 1),
            dimensions,
            (
                np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128),
                np.diag([complex(0.0, 1.0), complex(0.0, -1.0)]),
            ),
            fiber_indices=indices,
        ),
    }
    presentation = MagneticPresentation(
        generators=(
            MagneticGenerator("C3z", False),
            MagneticGenerator("TR", True),
        ),
        relations=(),
        central_phases=(1.0,),
        source="canonical-target-test",
    )
    return actions, presentation


def test_cyclotomic_monomial_target_json_roundtrip_is_bitwise_exact() -> None:
    actions, presentation = _encoding_fixture()

    encoded = encode_canonical_target_actions(
        actions,
        presentation,
        certificate=_certificate(),
        root_order=24,
    )
    transported = json.loads(json.dumps(encoded, sort_keys=True, allow_nan=False))
    decoded = decode_canonical_target_actions(
        transported,
        actions,
        presentation,
    )

    assert transported["version"] == CANONICAL_TARGET_ACTIONS_V1
    assert transported["numeric_entries_modified"] == 0
    assert transported["root_order"] == 24
    assert set(decoded) == set(actions)
    for name, expected in actions.items():
        actual = decoded[name]
        assert actual.antiunitary is expected.antiunitary
        assert actual.fiber_permutation == expected.fiber_permutation
        assert actual.fiber_dimensions == expected.fiber_dimensions
        assert actual.fiber_indices == expected.fiber_indices
        for actual_block, expected_block in zip(
            actual.route_blocks,
            expected.route_blocks,
        ):
            np.testing.assert_array_equal(actual_block, expected_block)
        np.testing.assert_array_equal(
            materialize_block_route_action(actual),
            materialize_block_route_action(expected),
        )


def _dense_su2() -> np.ndarray:
    theta = 0.417
    alpha = -0.231
    beta = 0.683
    a = np.cos(theta) * np.exp(1.0j * alpha)
    b = np.sin(theta) * np.exp(1.0j * beta)
    value = np.asarray(
        [[a, b], [-np.conjugate(b), np.conjugate(a)]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(np.linalg.det(value), 1.0, atol=2.0e-15)
    return value


def _dense_u4() -> np.ndarray:
    left = _dense_su2()
    theta = -0.293
    alpha = 0.517
    beta = -0.809
    a = np.cos(theta) * np.exp(1.0j * alpha)
    b = np.sin(theta) * np.exp(1.0j * beta)
    right = np.asarray(
        [[a, b], [-np.conjugate(b), np.conjugate(a)]],
        dtype=np.complex128,
    )
    return np.asarray(
        np.exp(0.137j) * np.kron(left, right),
        dtype=np.complex128,
        order="C",
    )


def _single_action_fixture(
    block: np.ndarray,
    *,
    antiunitary: bool,
) -> tuple[dict[str, BlockRouteAction], MagneticPresentation]:
    dimension = int(block.shape[0])
    actions = {
        "A": BlockRouteAction(
            "A",
            antiunitary,
            (0,),
            (dimension,),
            (block,),
        )
    }
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("A", antiunitary),),
        relations=(),
        central_phases=(1.0,),
        source="general-dense-canonical-target-test",
    )
    return actions, presentation


@pytest.mark.parametrize(
    ("block", "antiunitary"),
    [
        (_dense_su2(), False),
        (_dense_su2(), True),
        (_dense_u4(), False),
        (_dense_u4(), True),
    ],
    ids=("su2-unitary", "su2-antiunitary", "u4-unitary", "u4-antiunitary"),
)
def test_general_dense_ud_target_roundtrips_every_complex128_bit(
    block: np.ndarray,
    antiunitary: bool,
) -> None:
    actions, presentation = _single_action_fixture(
        block,
        antiunitary=antiunitary,
    )

    artifact = encode_canonical_target_actions(
        actions,
        presentation,
        certificate=_certificate(),
        root_order=24,
    )
    transported = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    decoded = decode_canonical_target_actions(
        transported,
        actions,
        presentation,
    )["A"]

    assert transported["numeric_entries_modified"] == 0
    assert decoded.antiunitary is antiunitary
    assert decoded.fiber_permutation == (0,)
    assert decoded.fiber_dimensions == (block.shape[0],)
    actual = np.asarray(decoded.route_blocks[0], dtype=np.complex128, order="C")
    expected = np.asarray(block, dtype=np.complex128, order="C")
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def test_signed_zero_payload_roundtrips_bitwise_instead_of_being_resnapped() -> None:
    block = np.asarray(
        [
            [complex(-0.0, -1.0), complex(-0.0, 0.0)],
            [complex(0.0, -0.0), complex(0.0, 1.0)],
        ],
        dtype=np.complex128,
        order="C",
    )
    actions, presentation = _single_action_fixture(block, antiunitary=True)

    artifact = encode_canonical_target_actions(
        actions,
        presentation,
        certificate=_certificate(),
        root_order=24,
    )
    transported = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    decoded = decode_canonical_target_actions(
        transported,
        actions,
        presentation,
    )["A"]

    actual = np.asarray(decoded.route_blocks[0], dtype=np.complex128, order="C")
    assert actual.tobytes(order="C") == block.tobytes(order="C")
    np.testing.assert_array_equal(
        actual.view(np.uint64),
        block.view(np.uint64),
    )


@pytest.mark.parametrize(
    "block",
    [
        np.asarray(
            [
                [complex(-0.0, -1.0), complex(-0.0, 0.0)],
                [complex(0.0, -0.0), complex(0.0, 1.0)],
            ],
            dtype=np.complex128,
            order="C",
        ),
        _dense_u4(),
    ],
    ids=("signed-zero-u2", "general-dense-u4"),
)
def test_literal_route_source_roundtrip_allows_ambiguous_structure_classifier(
    block: np.ndarray,
) -> None:
    actions, presentation = _single_action_fixture(block, antiunitary=False)

    artifact = encode_literal_route_source_actions(
        actions,
        presentation,
        structure_identity={
            "status": "unknown_or_unsupported",
            "reason": "uniform/projective boundary is numerically ambiguous",
        },
        source_role="test_phase_source",
    )
    transported = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    restored = load_literal_route_source_artifact(transported)

    assert transported["version"] == LITERAL_ROUTE_SOURCE_ACTIONS_V1
    assert transported["status"] == "relation_certified"
    assert transported["threshold_cleanup_performed"] is False
    assert restored.structure_identity["status"] == "unknown_or_unsupported"
    actual = np.asarray(
        restored.actions["A"].route_blocks[0],
        dtype=np.complex128,
        order="C",
    )
    expected = np.asarray(block, dtype=np.complex128, order="C")
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def test_canonical_target_still_rejects_ambiguous_structure_identity() -> None:
    actions, presentation = _single_action_fixture(
        _dense_su2(),
        antiunitary=False,
    )

    with pytest.raises(CanonicalTargetError, match="complete certified"):
        encode_canonical_target_actions(
            actions,
            presentation,
            certificate={"status": "unknown_or_unsupported"},
        )


def test_canonical_target_decode_rejects_changed_antiunitary_or_route_layout() -> None:
    actions, presentation = _encoding_fixture()
    encoded = encode_canonical_target_actions(
        actions,
        presentation,
        certificate=_certificate(),
        root_order=24,
    )

    changed_parity = copy.deepcopy(encoded)
    changed_parity["actions"][1]["antiunitary"] = False
    with pytest.raises(CanonicalTargetError, match="artifact hash mismatch"):
        decode_canonical_target_actions(changed_parity, actions, presentation)

    changed_indices = copy.deepcopy(encoded)
    changed_indices["actions"][0]["fiber_indices"] = [[0, 1], [2, 3]]
    with pytest.raises(CanonicalTargetError, match="artifact hash mismatch"):
        decode_canonical_target_actions(changed_indices, actions, presentation)


def _identity_fixed_target() -> dict[str, BlockRouteAction]:
    return {
        "C3z": BlockRouteAction(
            "C3z",
            False,
            (0,),
            (2,),
            (np.eye(2, dtype=np.complex128),),
        )
    }


def test_fixed_target_accepts_unique_closest_branch_within_limits() -> None:
    target = _identity_fixed_target()
    raw = np.diag([np.exp(2.0e-6j), np.exp(-3.0e-6j)]).astype(np.complex128)
    raw[0, 1] = 4.0e-8j

    certificate = certify_fixed_target_exactification(
        {"C3z": raw},
        target,
        root_order=4,
        max_rms_correction=1.0e-4,
        max_route_correction=1.0e-4,
    )

    assert certificate["status"] == "certified"
    assert certificate["selection_policy"] == (
        "persisted_project_owned_target_with_direct_correction_budget"
    )
    assert certificate["branch_check_by_operation"]["C3z"] == (
        "diagnostic_local_target_nearest"
    )
    assert certificate["numeric_entries_modified"] == 0
    assert 0.0 < certificate["correction_rms"] < 1.0e-4
    assert 0.0 < certificate["correction_max"] < 1.0e-4
    assert certificate["minimum_branch_margin"] > 1.0


def test_fixed_target_distance_certificate_accepts_general_dense_ud_target() -> None:
    block = _dense_u4()
    target, _presentation = _single_action_fixture(block, antiunitary=True)
    perturbation = np.asarray(
        np.diag(np.exp(1.0j * np.asarray([1.0, -2.0, 3.0, -4.0]) * 1.0e-7)),
        dtype=np.complex128,
    )
    raw = block @ perturbation

    certificate = certify_fixed_target_exactification(
        {"A": raw},
        target,
        root_order=24,
        max_rms_correction=1.0e-5,
        max_route_correction=1.0e-5,
    )

    assert certificate["status"] == "certified"
    assert certificate["numeric_entries_modified"] == 0
    assert 0.0 < certificate["correction_rms"] < 1.0e-5
    assert 0.0 < certificate["correction_max"] < 1.0e-5


@pytest.mark.parametrize("angle", [np.pi / 4.0, np.deg2rad(50.0)])
def test_fixed_target_reports_local_branch_tie_or_crossing_without_reselecting(
    angle: float,
) -> None:
    target = _identity_fixed_target()
    raw = np.diag([np.exp(1.0j * angle), 1.0]).astype(np.complex128)

    certificate = certify_fixed_target_exactification(
        {"C3z": raw},
        target,
        root_order=4,
        max_rms_correction=2.0,
        max_route_correction=2.0,
    )

    assert certificate["status"] == "certified"
    assert certificate["minimum_branch_margin"] <= 1.0e-15
    assert certificate["branch_check_by_operation"]["C3z"] in {
        "diagnostic_local_target_nearest",
        "diagnostic_local_alternative_closer_non_authoritative",
    }


@pytest.mark.parametrize(
    ("max_rms", "max_route", "message"),
    [
        (1.0e-3, 1.0, "RMS correction"),
        (1.0, 1.0e-3, "route correction"),
    ],
)
def test_fixed_target_rejects_configured_correction_limit(
    max_rms: float,
    max_route: float,
    message: str,
) -> None:
    target = _identity_fixed_target()
    raw = np.diag([np.exp(0.2j), np.exp(-0.2j)]).astype(np.complex128)

    with pytest.raises(CanonicalTargetError, match=message):
        certify_fixed_target_exactification(
            {"C3z": raw},
            target,
            root_order=4,
            max_rms_correction=max_rms,
            max_route_correction=max_route,
        )
