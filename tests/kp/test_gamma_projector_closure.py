from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

import kp.blocks as public_blocks
import kp.blocks.gamma_layout as gamma_layout_module

from kp.blocks.gamma_layout import (
    GammaLayoutError,
    GammaRowLayout,
    GammaRoutingThresholds,
    GammaRoutingError,
    assemble_gamma_routed_projectors,
    build_gamma_routed_frames,
    certify_gamma_raw_action,
    certify_routed_covariance,
    close_gamma_projector_clusters,
    cluster_gamma_eigensystem,
    infer_gamma_raw_action_q_permutations,
)
from kp.symmetry.candidate_certificate import (
    CandidateOperationInput,
    CandidateProjectionState,
    CandidateSymmetryStatus,
    CandidateSymmetryThresholds,
    certify_candidate_symmetries,
)
from kp.symmetry.joint_exactification import (
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
)
from kp.projection_selection import CandidateRejectionReason


def _layout(*, q1: np.ndarray | None = None, q2: np.ndarray | None = None) -> GammaRowLayout:
    first = np.asarray([[0.0, 0.0], [1.0, 0.0]]) if q1 is None else np.asarray(q1)
    second = np.asarray([[0.0, 0.0], [0.0, 1.0]]) if q2 is None else np.asarray(q2)
    return GammaRowLayout.build(
        qsets=(first, second),
        num_layer_list=(1, 2),
        num_orb_per_layer_list=((2,), (2, 2)),
        spin_convention="all",
        source_basis_hash="tapw-source-basis-a",
    )


def _thresholds() -> GammaRoutingThresholds:
    return GammaRoutingThresholds(
        energy_same_ev=1.0e-8,
        energy_different_ev=1.0e-6,
        capture_zero_fraction=1.0e-10,
        capture_loss_max=1.0e-9,
        local_action_isometry=1.0e-10,
        off_route_leakage=1.0e-10,
        closure_residual=1.0e-9,
        route_zero_gap=1.0e-6,
        route_covariance=1.0e-9,
        projector_residual=1.0e-10,
        anchor_sigma_min=1.0e-8,
        max_rank=12,
        max_iterations=16,
    )


def _full_action_from_local_routes(
    layout: GammaRowLayout,
    local_actions: tuple[np.ndarray, ...],
    q_permutation: tuple[int, ...],
) -> np.ndarray:
    full = np.zeros(
        (layout.full_dimension, layout.full_dimension), dtype=np.complex128
    )
    for source_q, target_q in enumerate(q_permutation):
        source_rows = layout.same_q_full_rows(source_q)
        target_rows = layout.same_q_full_rows(target_q)
        full[np.ix_(target_rows, source_rows)] = local_actions[source_q]
    return full


def _identity_action(
    layout: GammaRowLayout,
    *,
    thresholds: GammaRoutingThresholds | None = None,
):
    gates = _thresholds() if thresholds is None else thresholds
    return certify_gamma_raw_action(
        name="identity",
        full_action=np.eye(layout.full_dimension, dtype=np.complex128),
        layout=layout,
        q_permutations=(tuple(range(layout.q_count)),) * 2,
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=gates,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )


def _equal_group_layout() -> GammaRowLayout:
    return GammaRowLayout.build(
        qsets=(np.zeros((2, 2)), np.ones((2, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((2,), (2,)),
        spin_convention="all",
        source_basis_hash="equal-groups",
    )


def _route_inference_layout(q_count: int = 3) -> GammaRowLayout:
    return GammaRowLayout.build(
        qsets=(np.zeros((q_count, 2)), np.ones((q_count, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash="route-inference",
    )


def _sector_routed_local_action(
    layout: GammaRowLayout, sector_map: tuple[int, int]
) -> np.ndarray:
    local = np.zeros(
        (layout.same_q_dimension, layout.same_q_dimension), dtype=np.complex128
    )
    for source_group, target_group in enumerate(sector_map):
        source_rows = layout.source_group_local_rows(source_group)
        target_rows = layout.source_group_local_rows(target_group)
        local[np.ix_(target_rows, source_rows)] = np.eye(source_rows.size)
    return local


def test_infer_raw_action_q_routes_finds_nontrivial_three_q_permutation() -> None:
    layout = _route_inference_layout()
    thresholds = replace(_thresholds(), max_rank=layout.same_q_dimension)
    route = (2, 0, 1)
    local = _sector_routed_local_action(layout, (0, 1))
    full_action = _full_action_from_local_routes(
        layout, (local,) * layout.q_count, route
    )

    inferred = infer_gamma_raw_action_q_permutations(
        full_action=full_action,
        layout=layout,
        sector_map=(0, 1),
        thresholds=thresholds,
    )

    assert inferred == (route, route)
    certified = certify_gamma_raw_action(
        name="inferred-three-q-route",
        full_action=full_action,
        layout=layout,
        q_permutations=inferred,
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=thresholds,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    assert certified.q_permutation == route


def test_infer_raw_action_q_routes_supports_manifest_sector_swap() -> None:
    layout = _route_inference_layout(q_count=2)
    thresholds = replace(_thresholds(), max_rank=layout.same_q_dimension)
    route = (1, 0)
    local = _sector_routed_local_action(layout, (1, 0))
    full_action = _full_action_from_local_routes(
        layout, (local,) * layout.q_count, route
    )

    inferred = infer_gamma_raw_action_q_permutations(
        full_action=full_action,
        layout=layout,
        sector_map=(1, 0),
        thresholds=thresholds,
    )

    assert inferred == (route, route)
    certified = certify_gamma_raw_action(
        name="inferred-sector-swap",
        full_action=full_action,
        layout=layout,
        q_permutations=inferred,
        sector_map=(1, 0),
        antiunitary=False,
        thresholds=thresholds,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    assert certified.sector_map == (1, 0)


def test_infer_raw_action_q_routes_rejects_ambiguous_off_route_support() -> None:
    layout = _route_inference_layout(q_count=2)
    thresholds = replace(_thresholds(), max_rank=layout.same_q_dimension)
    local = _sector_routed_local_action(layout, (0, 1)) / np.sqrt(2.0)
    ambiguous = np.zeros(
        (layout.full_dimension, layout.full_dimension), dtype=np.complex128
    )
    for source_q in range(layout.q_count):
        source_rows = layout.same_q_full_rows(source_q)
        for target_q in range(layout.q_count):
            target_rows = layout.same_q_full_rows(target_q)
            ambiguous[np.ix_(target_rows, source_rows)] = local

    with pytest.raises(GammaRoutingError, match="unique mapped-sector Q route") as error:
        infer_gamma_raw_action_q_permutations(
            full_action=ambiguous,
            layout=layout,
            sector_map=(0, 1),
            thresholds=thresholds,
        )
    assert error.value.reason is CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH


def test_infer_raw_action_q_routes_rejects_nonbijective_support() -> None:
    layout = _route_inference_layout(q_count=3)
    thresholds = replace(_thresholds(), max_rank=layout.same_q_dimension)
    local = _sector_routed_local_action(layout, (0, 1))
    nonbijective = _full_action_from_local_routes(
        layout, (local,) * layout.q_count, (0, 0, 2)
    )

    with pytest.raises(GammaRoutingError, match="not a permutation") as error:
        infer_gamma_raw_action_q_permutations(
            full_action=nonbijective,
            layout=layout,
            sector_map=(0, 1),
            thresholds=thresholds,
        )
    assert error.value.reason is CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH


def test_infer_raw_action_q_routes_rejects_source_group_route_disagreement() -> None:
    layout = _route_inference_layout(q_count=2)
    thresholds = replace(_thresholds(), max_rank=layout.same_q_dimension)
    full_action = np.zeros(
        (layout.full_dimension, layout.full_dimension), dtype=np.complex128
    )
    group_routes = ((0, 1), (1, 0))
    for source_group, route in enumerate(group_routes):
        local_rows = layout.source_group_local_rows(source_group)
        for source_q, target_q in enumerate(route):
            source_rows = np.asarray(
                [
                    layout.rows_by_q[source_q][row].full_row
                    for row in local_rows
                ],
                dtype=np.intp,
            )
            target_rows = np.asarray(
                [
                    layout.rows_by_q[target_q][row].full_row
                    for row in local_rows
                ],
                dtype=np.intp,
            )
            full_action[np.ix_(target_rows, source_rows)] = np.eye(source_rows.size)

    with pytest.raises(GammaRoutingError, match="disagree between source groups"):
        infer_gamma_raw_action_q_permutations(
            full_action=full_action,
            layout=layout,
            sector_map=(0, 1),
            thresholds=thresholds,
        )


def test_infer_raw_action_q_routes_rejects_wrong_manifest_sector_map() -> None:
    layout = _route_inference_layout(q_count=2)
    thresholds = replace(_thresholds(), max_rank=layout.same_q_dimension)
    identity = np.eye(layout.full_dimension, dtype=np.complex128)

    with pytest.raises(GammaRoutingError, match="unique mapped-sector Q route"):
        infer_gamma_raw_action_q_permutations(
            full_action=identity,
            layout=layout,
            sector_map=(1, 0),
            thresholds=thresholds,
        )


def test_infer_raw_action_q_routes_dense_and_sparse_are_canonically_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _route_inference_layout(q_count=3)
    thresholds = replace(_thresholds(), max_rank=layout.same_q_dimension)
    route = (1, 2, 0)
    local = _sector_routed_local_action(layout, (0, 1))
    dense = _full_action_from_local_routes(
        layout, (local,) * layout.q_count, route
    )

    original_toarray = sparse.csr_matrix.toarray

    def guarded_toarray(matrix, *args, **kwargs):
        if matrix.shape == (layout.full_dimension, layout.full_dimension):
            raise AssertionError("full raw-H action was densified")
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", guarded_toarray)
    sparse_action = sparse.csr_matrix(dense)
    common = dict(
        layout=layout,
        sector_map=(0, 1),
        thresholds=thresholds,
    )

    dense_route = infer_gamma_raw_action_q_permutations(
        full_action=dense, **common
    )
    sparse_route = infer_gamma_raw_action_q_permutations(
        full_action=sparse_action, **common
    )

    assert dense_route == sparse_route == (route, route)


def test_gamma_row_layout_has_real_1_plus_2_spinful_rows_and_bijection() -> None:
    layout = _layout()

    assert layout.same_q_dimension == 12
    assert layout.full_dimension == 24
    np.testing.assert_array_equal(
        layout.same_q_full_rows(0),
        [0, 1, 4, 5, 6, 7, 12, 13, 16, 17, 18, 19],
    )
    np.testing.assert_array_equal(
        layout.same_q_full_rows(1),
        [2, 3, 8, 9, 10, 11, 14, 15, 20, 21, 22, 23],
    )
    addresses = [layout.address_for_full_row(row) for row in range(layout.full_dimension)]
    assert sorted(address.full_row for address in addresses) == list(range(24))
    assert len({(address.q_index, address.same_q_local_row) for address in addresses}) == 24
    q0_down_group1_layer1 = layout.address_for_full_row(18)
    assert (
        q0_down_group1_layer1.source_group,
        q0_down_group1_layer1.layer_in_group,
        q0_down_group1_layer1.spin_index,
        q0_down_group1_layer1.orbital,
    ) == (1, 1, 1, 0)


def test_gamma_layout_hash_binds_q_order_source_basis_and_spin() -> None:
    layout = _layout()
    swapped_q = _layout(q1=np.asarray([[1.0, 0.0], [0.0, 0.0]]))
    changed_source = GammaRowLayout.build(
        qsets=(np.asarray([[0.0, 0.0], [1.0, 0.0]]), np.asarray([[0.0, 0.0], [0.0, 1.0]])),
        num_layer_list=(1, 2),
        num_orb_per_layer_list=((2,), (2, 2)),
        spin_convention="all",
        source_basis_hash="tapw-source-basis-b",
    )
    assert len(layout.layout_hash) == 64
    assert swapped_q.layout_hash != layout.layout_hash
    assert changed_source.layout_hash != layout.layout_hash

    with pytest.raises(GammaLayoutError) as exc_info:
        GammaRowLayout.build(
            qsets=(np.asarray([[0.0, 0.0], [1.0, 0.0]]), np.asarray([[0.0, 0.0], [0.0, 1.0]])),
            num_layer_list=(1, 2),
            num_orb_per_layer_list=((2,), (2, 2)),
            spin_convention="up",
            source_basis_hash="tapw-source-basis-a",
        )
    assert exc_info.value.reason.value == "UNSUPPORTED_SPIN_ROUTE"


@pytest.mark.parametrize(
    ("qsets", "layers", "orbitals", "message"),
    [
        ((np.zeros((2, 2)),), (1,), ((2,),), "exactly two source groups"),
        ((np.zeros((2, 2)), np.zeros((3, 2))), (1, 2), ((2,), (2, 2)), "equal ordered Q counts"),
        ((np.zeros((2, 2)), np.zeros((2, 3))), (1, 2), ((2,), (2, 2)), "coordinate dimensions"),
        ((np.zeros((2, 2)), np.zeros((2, 2))), (1, 2), ((2,), (2, 3)), "equal per-physical-layer orbital width"),
    ],
)
def test_gamma_layout_rejects_unsupported_v1_domain(qsets, layers, orbitals, message) -> None:
    with pytest.raises(GammaLayoutError, match=message):
        GammaRowLayout.build(
            qsets=qsets,
            num_layer_list=layers,
            num_orb_per_layer_list=orbitals,
            spin_convention="all",
            source_basis_hash="tapw-source-basis-a",
        )


@pytest.mark.parametrize(
    ("layers", "orbitals"),
    [
        ((True, 2), ((2,), (2, 2))),
        ((1.5, 2), ((2,), (2, 2))),
        ((1, 2), ((True,), (2, 2))),
        ((1, 2), ((2,), (2, 2.5))),
    ],
)
def test_gamma_layout_rejects_bool_and_fractional_integer_metadata(layers, orbitals) -> None:
    with pytest.raises(GammaLayoutError) as exc_info:
        GammaRowLayout.build(
            qsets=(np.zeros((2, 2)), np.zeros((2, 2))),
            num_layer_list=layers,
            num_orb_per_layer_list=orbitals,
            spin_convention="all",
            source_basis_hash="tapw-source-basis-a",
        )
    assert exc_info.value.reason is CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT


@pytest.mark.parametrize("source_hash", ["", 17, None])
def test_gamma_layout_requires_a_literal_nonempty_source_basis_hash(source_hash) -> None:
    with pytest.raises(GammaLayoutError) as exc_info:
        GammaRowLayout.build(
            qsets=(np.zeros((2, 2)), np.zeros((2, 2))),
            num_layer_list=(1, 2),
            num_orb_per_layer_list=((2,), (2, 2)),
            spin_convention="all",
            source_basis_hash=source_hash,
        )
    assert exc_info.value.reason is CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT


def test_public_gamma_exports_and_stable_rejection_values_are_complete() -> None:
    exported = {
        "GammaCertifiedRawAction",
        "GammaClusterClosure",
        "GammaEnergyCluster",
        "GammaLayoutError",
        "GammaRoutedFrames",
        "GammaRoutingError",
        "GammaRoutingThresholds",
        "GammaRowAddress",
        "GammaRowLayout",
        "assemble_gamma_routed_projectors",
        "build_gamma_routed_frames",
        "certify_gamma_raw_action",
        "certify_routed_covariance",
        "close_gamma_projector_clusters",
        "cluster_gamma_eigensystem",
    }
    assert exported <= set(public_blocks.__all__)
    assert all(hasattr(public_blocks, name) for name in exported)
    expected_reasons = {
        "UNSUPPORTED_GAMMA_LAYOUT",
        "UNSUPPORTED_SPIN_ROUTE",
        "INVALID_GAMMA_ROW_LAYOUT",
        "GAMMA_Q_ROUTE_MISMATCH",
        "LOCAL_ACTION_ISOMETRY",
        "RAW_ACTION_ROUTE_LEAKAGE",
        "AMBIGUOUS_ENERGY_CLUSTER",
        "AMBIGUOUS_CLUSTER_CAPTURE",
        "SYMMETRY_CLOSURE_FAILURE",
        "AMBIGUOUS_SOURCE_GROUP_ROUTING",
        "EMPTY_SOURCE_GROUP",
        "SOURCE_GROUP_RANK_CHANGE",
        "SOURCE_GROUP_ROUTE_COVARIANCE",
        "PROJECTOR_FRAME_RANK",
        "HANDOFF_K_COVERAGE",
        "HANDOFF_IDENTITY",
        "CANDIDATE_SYMMETRY_FAILED",
    }
    assert {
        reason.value
        for reason in CandidateRejectionReason
        if reason.value.isupper()
    } == expected_reasons


def test_energy_clusters_use_two_thresholds_and_reject_gray_zone() -> None:
    vectors = np.eye(4, dtype=np.complex128)
    clusters = cluster_gamma_eigensystem(
        np.asarray([0.0, 0.5e-8, 2.0e-6, 3.0e-6]),
        vectors,
        thresholds=_thresholds(),
    )
    assert [cluster.band_indices for cluster in clusters] == [(0, 1), (2,), (3,)]

    with pytest.raises(GammaRoutingError, match="energy-cluster gray zone"):
        cluster_gamma_eigensystem(
            np.asarray([0.0, 2.0e-7]),
            np.eye(2, dtype=np.complex128),
            thresholds=_thresholds(),
        )


def test_routing_thresholds_are_explicit_identity_bound_and_domain_checked() -> None:
    payload = _thresholds().to_payload()
    payload.pop("schema")
    restored = GammaRoutingThresholds.from_normalized_config(payload)
    assert restored == _thresholds()
    assert len(restored.identity_hash) == 64

    changed = dict(payload)
    changed["capture_loss_max"] = 2.0e-9
    assert GammaRoutingThresholds.from_normalized_config(changed).identity_hash != restored.identity_hash
    missing = dict(payload)
    missing.pop("route_covariance")
    with pytest.raises(ValueError, match="missing.*route_covariance"):
        GammaRoutingThresholds.from_normalized_config(missing)
    invalid = dict(payload)
    invalid["anchor_sigma_min"] = 0.0
    with pytest.raises(ValueError, match="anchor_sigma_min"):
        GammaRoutingThresholds.from_normalized_config(invalid)
    noninteger = dict(payload)
    noninteger["max_rank"] = 3.5
    with pytest.raises(ValueError, match="positive integers"):
        GammaRoutingThresholds.from_normalized_config(noninteger)


def test_routing_threshold_scalars_reject_bool_string_and_overflow() -> None:
    for field, invalid in (
        ("capture_loss_max", "0.1"),
        ("local_action_isometry", True),
        ("route_covariance", 10**10000),
    ):
        with pytest.raises(ValueError, match="strict finite numeric"):
            replace(_thresholds(), **{field: invalid})


def test_raw_action_rejects_group_dependent_q_route_before_projection() -> None:
    layout = _layout()
    with pytest.raises(GammaRoutingError, match="q route mismatch"):
        certify_gamma_raw_action(
            name="bad-route",
            full_action=np.eye(layout.full_dimension, dtype=np.complex128),
            layout=layout,
            q_permutations=((1, 0), (0, 1)),
            sector_map=(0, 1),
            antiunitary=False,
            thresholds=_thresholds(),
            tapw_source_basis_hash="tapw-source-basis-a",
        )


def test_raw_action_rejects_off_route_leakage_and_nonisometry() -> None:
    layout = _layout()
    leaked = np.eye(layout.full_dimension, dtype=np.complex128)
    leaked[layout.same_q_full_rows(1)[0], layout.same_q_full_rows(0)[0]] = 0.1
    with pytest.raises(GammaRoutingError, match="off-route leakage"):
        certify_gamma_raw_action(
            name="leaked",
            full_action=leaked,
            layout=layout,
            q_permutations=((0, 1), (0, 1)),
            sector_map=(0, 1),
            antiunitary=False,
            thresholds=_thresholds(),
            tapw_source_basis_hash="tapw-source-basis-a",
        )


def test_raw_action_rejects_same_q_wrong_sector_illegal_routes_and_source_identity() -> None:
    layout = _layout()
    identity = np.eye(layout.full_dimension, dtype=np.complex128)
    with pytest.raises(GammaRoutingError) as wrong_sector:
        certify_gamma_raw_action(
            name="wrong-sector",
            full_action=identity,
            layout=layout,
            q_permutations=((0, 1), (0, 1)),
            sector_map=(1, 0),
            antiunitary=False,
            thresholds=_thresholds(),
            tapw_source_basis_hash="tapw-source-basis-a",
        )
    assert wrong_sector.value.reason is CandidateRejectionReason.RAW_ACTION_ROUTE_LEAKAGE

    with pytest.raises(GammaRoutingError) as illegal_sector:
        certify_gamma_raw_action(
            name="illegal-sector",
            full_action=identity,
            layout=layout,
            q_permutations=((0, 1), (0, 1)),
            sector_map=(0, 0),
            antiunitary=False,
            thresholds=_thresholds(),
            tapw_source_basis_hash="tapw-source-basis-a",
        )
    assert illegal_sector.value.reason is CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH

    with pytest.raises(GammaRoutingError) as wrong_basis:
        certify_gamma_raw_action(
            name="wrong-basis",
            full_action=identity,
            layout=layout,
            q_permutations=((0, 1), (0, 1)),
            sector_map=(0, 1),
            antiunitary=False,
            thresholds=_thresholds(),
            tapw_source_basis_hash="tapw-source-basis-b",
        )
    assert wrong_basis.value.reason is CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT


@pytest.mark.parametrize(
    ("q_routes", "sector_map"),
    [
        (((False, 1), (0, 1)), (0, 1)),
        (((0.0, 1), (0, 1)), (0, 1)),
        (((0, 1), (0, 1)), (False, 1)),
        (((0, 1), (0, 1)), (0.0, 1)),
    ],
)
def test_raw_action_route_metadata_requires_strict_integral_values(
    q_routes, sector_map
) -> None:
    layout = _layout()
    with pytest.raises(GammaRoutingError) as exc_info:
        certify_gamma_raw_action(
            name="non-integral-route",
            full_action=np.eye(layout.full_dimension, dtype=np.complex128),
            layout=layout,
            q_permutations=q_routes,
            sector_map=sector_map,
            antiunitary=False,
            thresholds=_thresholds(),
            tapw_source_basis_hash=layout.tapw_source_basis_hash,
        )
    assert exc_info.value.reason is CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH


def test_raw_action_rejects_non_bool_antiunitary_instead_of_coercion() -> None:
    layout = _layout()
    for invalid in ("false", 1):
        with pytest.raises(GammaRoutingError) as exc_info:
            certify_gamma_raw_action(
                name="invalid-antiunitary",
                full_action=np.eye(layout.full_dimension, dtype=np.complex128),
                layout=layout,
                q_permutations=((0, 1), (0, 1)),
                sector_map=(0, 1),
                antiunitary=invalid,
                thresholds=_thresholds(),
                tapw_source_basis_hash=layout.tapw_source_basis_hash,
            )
        assert exc_info.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_raw_action_rejects_threshold_rank_above_layout_local_dimension() -> None:
    layout = _layout()
    oversized = replace(_thresholds(), max_rank=13)
    with pytest.raises(GammaRoutingError) as exc_info:
        certify_gamma_raw_action(
            name="oversized-threshold-rank",
            full_action=np.eye(layout.full_dimension, dtype=np.complex128),
            layout=layout,
            q_permutations=((0, 1), (0, 1)),
            sector_map=(0, 1),
            antiunitary=False,
            thresholds=oversized,
            tapw_source_basis_hash=layout.tapw_source_basis_hash,
        )
    assert exc_info.value.reason is CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE


def test_dense_and_csr_raw_actions_have_equal_identity_without_full_densification() -> None:
    layout = _layout()
    dense = np.eye(layout.full_dimension, dtype=np.complex128)

    class GuardedCSR(sparse.csr_matrix):
        def toarray(self, *args, **kwargs):
            if self.shape == (layout.full_dimension, layout.full_dimension):
                raise AssertionError("full raw-H action was densified")
            return super().toarray(*args, **kwargs)

        def __array__(self, *args, **kwargs):
            if self.shape == (layout.full_dimension, layout.full_dimension):
                raise AssertionError("full raw-H action was densified")
            return super().__array__(*args, **kwargs)

    csr = GuardedCSR(sparse.csr_matrix(dense))
    common = dict(
        name="identity",
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=_thresholds(),
        tapw_source_basis_hash="tapw-source-basis-a",
    )
    dense_result = certify_gamma_raw_action(full_action=dense, **common)
    sparse_result = certify_gamma_raw_action(full_action=csr, **common)

    assert sparse_result.action_hash == dense_result.action_hash
    assert sparse_result.layout_hash == layout.layout_hash
    assert sparse_result.thresholds_hash == _thresholds().identity_hash
    for left, right in zip(
        dense_result.local_actions_by_source_q,
        sparse_result.local_actions_by_source_q,
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)

    poisoned = sparse.csr_matrix(dense)
    poisoned.data[0] = np.nan
    with pytest.raises(GammaRoutingError, match="finite"):
        certify_gamma_raw_action(full_action=poisoned, **common)

    scaled = 0.5 * np.eye(layout.full_dimension, dtype=np.complex128)
    with pytest.raises(GammaRoutingError, match="isometry"):
        certify_gamma_raw_action(
            name="scaled",
            full_action=scaled,
            layout=layout,
            q_permutations=((0, 1), (0, 1)),
            sector_map=(0, 1),
            antiunitary=False,
            thresholds=_thresholds(),
            tapw_source_basis_hash="tapw-source-basis-a",
        )


def test_noncanonical_sparse_formats_equal_dense_without_format_errors() -> None:
    layout = _layout()
    dimension = layout.full_dimension
    indptr = [0]
    indices = []
    data = []
    for row in range(dimension):
        indices.extend((row, row))
        data.extend((0.75, 0.25))
        indptr.append(len(indices))
    noncanonical_csr = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(dimension, dimension),
    )
    assert not noncanonical_csr.has_canonical_format
    coo = sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            (
                np.repeat(np.arange(dimension), 2),
                np.asarray(indices),
            ),
        ),
        shape=(dimension, dimension),
    )
    dia = sparse.dia_matrix(
        (np.ones((1, dimension), dtype=np.complex128), np.asarray([0])),
        shape=(dimension, dimension),
    )
    common = dict(
        name="identity-all-sparse-formats",
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=_thresholds(),
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    dense = certify_gamma_raw_action(
        full_action=np.eye(dimension, dtype=np.complex128), **common
    )
    for candidate in (noncanonical_csr, coo, dia):
        certified = certify_gamma_raw_action(full_action=candidate, **common)
        assert certified.action_hash == dense.action_hash
        for actual, expected in zip(
            certified.local_actions_by_source_q,
            dense.local_actions_by_source_q,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


def test_noncanonical_csr_route_leakage_uses_mathematical_duplicate_sum() -> None:
    layout = _layout()
    dimension = layout.full_dimension
    leak_row = int(layout.same_q_full_rows(1)[0])
    leak_column = int(layout.same_q_full_rows(0)[0])
    rows = []
    for row in range(dimension):
        entries = [(row, 1.0)]
        if row == leak_row:
            entries.extend(((leak_column, 0.2), (leak_column, 0.2)))
        rows.append(entries)
    indptr = [0]
    indices = []
    data = []
    for entries in rows:
        for column, value in reversed(entries):
            indices.append(column)
            data.append(value)
        indptr.append(len(indices))
    noncanonical = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.complex128),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(dimension, dimension),
    )
    assert not noncanonical.has_canonical_format
    gates = replace(_thresholds(), off_route_leakage=0.1)
    common = dict(
        name="duplicate-leakage",
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=gates,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    dense = np.eye(dimension, dtype=np.complex128)
    dense[leak_row, leak_column] = 0.4
    for candidate in (dense, noncanonical):
        with pytest.raises(GammaRoutingError) as exc_info:
            certify_gamma_raw_action(full_action=candidate, **common)
        assert exc_info.value.reason is CandidateRejectionReason.RAW_ACTION_ROUTE_LEAKAGE


def test_cluster_closure_adds_complete_kramers_clusters_and_is_gauge_invariant() -> None:
    layout = _layout()
    local_dim = layout.same_q_dimension
    tr_local = np.zeros((local_dim, local_dim), dtype=np.complex128)
    half = local_dim // 2
    tr_local[:half, half:] = np.eye(half)
    tr_local[half:, :half] = -np.eye(half)
    tr_full = np.zeros((layout.full_dimension, layout.full_dimension), dtype=np.complex128)
    for q in range(layout.q_count):
        rows = layout.same_q_full_rows(q)
        tr_full[np.ix_(rows, rows)] = tr_local
    tr = certify_gamma_raw_action(
        name="TR",
        full_action=tr_full,
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=True,
        thresholds=_thresholds(),
        tapw_source_basis_hash="tapw-source-basis-a",
    )

    eigenvalues = tuple(np.r_[np.full(4, -2.0), np.arange(8, dtype=float)] for _ in range(2))
    ordered_columns = (0, 6, 2, 8, 1, 3, 4, 5, 7, 9, 10, 11)
    eigenvectors = tuple(
        np.eye(local_dim, dtype=np.complex128)[:, ordered_columns]
        for _ in range(2)
    )
    closed = close_gamma_projector_clusters(
        eigenvalues,
        eigenvectors,
        layout=layout,
        seed_band_indices=((0, 1, 2, 3), (0, 1, 2, 3)),
        actions=(tr,),
        thresholds=_thresholds(),
    )
    assert closed.band_indices_by_q == ((0, 1, 2, 3), (0, 1, 2, 3))
    assert closed.rank_by_q == (4, 4)

    rng = np.random.default_rng(7)
    haar, _ = np.linalg.qr(
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )
    rotated = [matrix.copy() for matrix in eigenvectors]
    rotated[0][:, :4] = rotated[0][:, :4] @ haar
    rotated[1][:, :4] = rotated[1][:, :4] @ haar
    closed_rotated = close_gamma_projector_clusters(
        eigenvalues,
        tuple(rotated),
        layout=layout,
        seed_band_indices=((0, 1, 2, 3), (0, 1, 2, 3)),
        actions=(tr,),
        thresholds=_thresholds(),
    )
    assert closed_rotated.band_indices_by_q == closed.band_indices_by_q
    for left, right in zip(closed.projectors_by_q, closed_rotated.projectors_by_q, strict=True):
        np.testing.assert_allclose(left, right, atol=1.0e-12)


def test_cluster_capture_gray_zone_rejects_instead_of_adding_nonzero_overlap() -> None:
    layout = _layout()
    local = np.eye(layout.same_q_dimension, dtype=np.complex128)
    local[np.ix_([0, 1], [0, 1])] = np.asarray(
        [[1.0, -1.0], [1.0, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    full = np.zeros((layout.full_dimension, layout.full_dimension), dtype=np.complex128)
    for q in range(layout.q_count):
        rows = layout.same_q_full_rows(q)
        full[np.ix_(rows, rows)] = local
    action = certify_gamma_raw_action(
        name="mixed",
        full_action=full,
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=_thresholds(),
        tapw_source_basis_hash="tapw-source-basis-a",
    )
    values = tuple(np.arange(layout.same_q_dimension, dtype=float) for _ in range(2))
    vectors = tuple(np.eye(layout.same_q_dimension, dtype=np.complex128) for _ in range(2))

    with pytest.raises(GammaRoutingError) as exc_info:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((0,), (0,)),
            actions=(action,),
            thresholds=_thresholds(),
        )
    assert exc_info.value.reason.value == "AMBIGUOUS_CLUSTER_CAPTURE"


def test_cluster_closure_freezes_one_joint_band_set_across_every_q() -> None:
    layout = _layout()
    values = tuple(
        np.arange(layout.same_q_dimension, dtype=float)
        for _ in range(layout.q_count)
    )
    vectors = tuple(
        np.eye(layout.same_q_dimension, dtype=np.complex128)
        for _ in range(layout.q_count)
    )
    closed = close_gamma_projector_clusters(
        values,
        vectors,
        layout=layout,
        seed_band_indices=((0,), (1,)),
        actions=(_identity_action(layout),),
        thresholds=_thresholds(),
    )
    assert closed.band_indices_by_q == ((0, 1), (0, 1))
    assert closed.rank_by_q == (2, 2)

    strict_rank = replace(_thresholds(), max_rank=1)
    strict_action = _identity_action(layout, thresholds=strict_rank)
    with pytest.raises(GammaRoutingError) as exc_info:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((0,), (1,)),
            actions=(strict_action,),
            thresholds=strict_rank,
        )
    assert exc_info.value.reason is CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE


def test_cluster_closure_rejects_split_production_clusters() -> None:
    layout = _layout()
    values = tuple(
        np.r_[np.zeros(2), np.arange(1, layout.same_q_dimension - 1)]
        for _ in range(layout.q_count)
    )
    vectors = tuple(
        np.eye(layout.same_q_dimension, dtype=np.complex128)
        for _ in range(layout.q_count)
    )
    with pytest.raises(GammaRoutingError) as exc_info:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((0,), (0,)),
            actions=(_identity_action(layout),),
            thresholds=_thresholds(),
        )
    assert exc_info.value.reason is CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER


def test_action_identity_and_frozen_local_blocks_cannot_be_laundered() -> None:
    layout = _layout()
    source = np.eye(layout.full_dimension, dtype=np.complex128)
    action = certify_gamma_raw_action(
        name="identity",
        full_action=source,
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=_thresholds(),
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    source[0, 0] = 0.0
    assert action.local_actions_by_source_q[0][0, 0] == 1.0
    with pytest.raises(ValueError):
        action.local_actions_by_source_q[0][0, 0] = 0.0

    values = tuple(
        np.arange(layout.same_q_dimension, dtype=float)
        for _ in range(layout.q_count)
    )
    vectors = tuple(
        np.eye(layout.same_q_dimension, dtype=np.complex128)
        for _ in range(layout.q_count)
    )
    changed_thresholds = replace(_thresholds(), closure_residual=2.0e-9)
    with pytest.raises(GammaRoutingError) as threshold_error:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((0,), (0,)),
            actions=(action,),
            thresholds=changed_thresholds,
        )
    assert threshold_error.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY

    poisoned = action.local_actions_by_source_q[0].copy()
    poisoned[0, 0] = 0.5
    tampered = replace(
        action,
        local_actions_by_source_q=(poisoned, action.local_actions_by_source_q[1]),
    )
    with pytest.raises(GammaRoutingError) as tamper_error:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((0,), (0,)),
            actions=(tampered,),
            thresholds=_thresholds(),
        )
    assert tamper_error.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_capture_checks_total_other_weight_and_does_not_clip_eta() -> None:
    layout = _layout()
    relaxed = replace(
        _thresholds(),
        capture_zero_fraction=0.06,
        capture_loss_max=0.10,
        local_action_isometry=0.10,
    )
    first = np.asarray([np.sqrt(0.94), np.sqrt(0.055), np.sqrt(0.055)])
    first /= np.linalg.norm(first)
    seed = np.column_stack((first, np.eye(3)[:, 1:]))
    unitary, _ = np.linalg.qr(seed.astype(np.complex128))
    phase = np.vdot(unitary[:, 0], first)
    unitary[:, 0] *= np.exp(-1j * np.angle(phase))
    local = np.eye(layout.same_q_dimension, dtype=np.complex128)
    local[2:5, 2:5] = np.sqrt(1.05) * unitary
    full = _full_action_from_local_routes(layout, (local, local), (0, 1))
    action = certify_gamma_raw_action(
        name="cumulative-capture",
        full_action=full,
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=relaxed,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    values = tuple(
        np.arange(layout.same_q_dimension, dtype=float)
        for _ in range(layout.q_count)
    )
    vectors = tuple(
        np.eye(layout.same_q_dimension, dtype=np.complex128)
        for _ in range(layout.q_count)
    )
    with pytest.raises(GammaRoutingError, match="captures") as cumulative:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((2,), (2,)),
            actions=(action,),
            thresholds=relaxed,
        )
    assert cumulative.value.reason is CandidateRejectionReason.AMBIGUOUS_CLUSTER_CAPTURE

    scaled = np.eye(layout.same_q_dimension, dtype=np.complex128)
    scaled[2, 2] = np.sqrt(1.02)
    scaled_action = certify_gamma_raw_action(
        name="eta-above-one",
        full_action=_full_action_from_local_routes(layout, (scaled, scaled), (0, 1)),
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=relaxed,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    with pytest.raises(GammaRoutingError, match="out-of-range captures") as out_of_range:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((2,), (2,)),
            actions=(scaled_action,),
            thresholds=relaxed,
        )
    assert out_of_range.value.reason is CandidateRejectionReason.AMBIGUOUS_CLUSTER_CAPTURE


def _routed_eigensystems(layout: GammaRowLayout):
    dim = layout.same_q_dimension
    evals = tuple(np.arange(dim, dtype=float) for _ in range(layout.q_count))
    vecs = tuple(np.eye(dim, dtype=np.complex128) for _ in range(layout.q_count))
    # One Kramers pair per source group: up/down orbital 0 in group 0 and group 1.
    joint = (0, 6, 2, 8)
    return evals, vecs, joint


def test_routed_production_rejects_cluster_split_route_zero_empty_and_rank_change() -> None:
    layout = _layout()
    dim = layout.same_q_dimension
    identity = np.eye(dim, dtype=np.complex128)

    split_values = tuple(np.r_[np.zeros(2), np.arange(1, dim - 1)] for _ in range(2))
    with pytest.raises(GammaRoutingError) as split:
        build_gamma_routed_frames(
            split_values,
            (identity, identity),
            joint_band_indices=(0, 2, 6, 8),
            layout=layout,
            thresholds=_thresholds(),
        )
    assert split.value.reason is CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER

    mixed = identity.copy()
    mixed[:, 0] = (identity[:, 0] + identity[:, 2]) / np.sqrt(2.0)
    mixed[:, 2] = (identity[:, 0] - identity[:, 2]) / np.sqrt(2.0)
    values = tuple(np.arange(dim, dtype=float) for _ in range(2))
    with pytest.raises(GammaRoutingError) as route_zero:
        build_gamma_routed_frames(
            values,
            (mixed, mixed),
            joint_band_indices=(0,),
            layout=layout,
            thresholds=_thresholds(),
            require_complete_clusters=False,
        )
    assert route_zero.value.reason is CandidateRejectionReason.AMBIGUOUS_SOURCE_GROUP_ROUTING

    with pytest.raises(GammaRoutingError) as empty:
        build_gamma_routed_frames(
            values,
            (identity, identity),
            joint_band_indices=(0, 1),
            layout=layout,
            thresholds=_thresholds(),
            require_complete_clusters=False,
        )
    assert empty.value.reason is CandidateRejectionReason.EMPTY_SOURCE_GROUP

    selected_columns = (0, 1, 2, 6)

    def completed_basis(desired):
        remaining = tuple(index for index in range(dim) if index not in desired)
        permutation = [None] * dim
        for column, basis_row in zip(selected_columns, desired, strict=True):
            permutation[column] = basis_row
        for column, basis_row in zip(
            (index for index in range(dim) if index not in selected_columns),
            remaining,
            strict=True,
        ):
            permutation[column] = basis_row
        return identity[:, permutation]

    q0 = completed_basis((0, 1, 6, 2))
    q1 = completed_basis((0, 6, 2, 8))
    with pytest.raises(GammaRoutingError) as rank_change:
        build_gamma_routed_frames(
            values,
            (q0, q1),
            joint_band_indices=(0, 1, 2, 6),
            layout=layout,
            thresholds=_thresholds(),
            require_complete_clusters=False,
        )
    assert rank_change.value.reason is CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE


def test_routed_frames_validate_finite_eigensystems_and_anchor_certificate() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    nan_values = list(evals)
    nan_values[0] = nan_values[0].copy()
    nan_values[0][0] = np.nan
    with pytest.raises(GammaRoutingError):
        build_gamma_routed_frames(
            nan_values,
            vecs,
            joint_band_indices=joint,
            layout=layout,
            thresholds=_thresholds(),
            require_complete_clusters=False,
        )

    anchor = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    replacement_columns = (1, 7, 3, 9)
    remaining = tuple(
        index for index in range(layout.same_q_dimension)
        if index not in replacement_columns
    )
    permutation = [None] * layout.same_q_dimension
    for column, basis_row in zip(joint, replacement_columns, strict=True):
        permutation[column] = basis_row
    for column, basis_row in zip(
        (index for index in range(layout.same_q_dimension) if index not in joint),
        remaining,
        strict=True,
    ):
        permutation[column] = basis_row
    orthogonal = tuple(
        np.eye(layout.same_q_dimension, dtype=np.complex128)[:, permutation]
        for _ in range(layout.q_count)
    )
    with pytest.raises(GammaRoutingError) as rank_loss:
        build_gamma_routed_frames(
            evals,
            orthogonal,
            joint_band_indices=joint,
            layout=layout,
            thresholds=_thresholds(),
            anchor_frames=anchor.local_frames_by_q,
            require_complete_clusters=False,
        )
    assert rank_loss.value.reason is CandidateRejectionReason.PROJECTOR_FRAME_RANK

    invalid_anchors = []
    nan_anchor = anchor.local_frames_by_q[0].copy()
    nan_anchor[0, 0] = np.nan
    invalid_anchors.append((nan_anchor, anchor.local_frames_by_q[1]))
    duplicate = anchor.local_frames_by_q[0].copy()
    duplicate[:, 1] = duplicate[:, 0]
    invalid_anchors.append((duplicate, anchor.local_frames_by_q[1]))
    crossed = anchor.local_frames_by_q[0].copy()
    crossed[:, 2] = crossed[:, 0]
    invalid_anchors.append((crossed, anchor.local_frames_by_q[1]))
    for anchors in invalid_anchors:
        with pytest.raises(GammaRoutingError) as invalid:
            build_gamma_routed_frames(
                evals,
                vecs,
                joint_band_indices=joint,
                layout=layout,
                thresholds=_thresholds(),
                anchor_frames=anchors,
                require_complete_clusters=False,
            )
        assert invalid.value.reason is CandidateRejectionReason.PROJECTOR_FRAME_RANK


def test_routed_frames_and_real_assembler_are_orthonormal_and_complete() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )

    assert routed.group_dimensions == (2, 2)
    assert min(routed.route_gaps) > 0.0
    u_low, u_high = assemble_gamma_routed_projectors(
        routed,
        layout=layout,
        thresholds=_thresholds(),
        include_high=True,
    )
    np.testing.assert_allclose(u_low.conj().T @ u_low, np.eye(8), atol=1.0e-12)
    assert u_high is not None
    np.testing.assert_allclose(u_high.conj().T @ u_high, np.eye(16), atol=1.0e-12)
    np.testing.assert_allclose(u_low.conj().T @ u_high, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        u_low @ u_low.conj().T + u_high @ u_high.conj().T,
        np.eye(layout.full_dimension),
        atol=1.0e-12,
    )

    changed_thresholds = replace(_thresholds(), projector_residual=2.0e-10)
    with pytest.raises(GammaRoutingError) as exc_info:
        assemble_gamma_routed_projectors(
            routed,
            layout=layout,
            thresholds=changed_thresholds,
            include_high=False,
        )
    assert exc_info.value.reason.value == "HANDOFF_IDENTITY"


def test_assembler_uses_exact_c_g_alpha_q_order_and_revalidates_frame_hash() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    u_low, _ = assemble_gamma_routed_projectors(
        routed,
        layout=layout,
        thresholds=_thresholds(),
        include_high=False,
    )
    for group in range(2):
        start, _stop = routed.group_offsets[group : group + 2]
        for alpha in range(routed.group_dimensions[group]):
            for q in range(layout.q_count):
                column = (
                    layout.q_count * sum(routed.group_dimensions[:group])
                    + alpha * layout.q_count
                    + q
                )
                expected = np.zeros(layout.full_dimension, dtype=np.complex128)
                expected[layout.same_q_full_rows(q)] = routed.local_frames_by_q[q][
                    :, start + alpha
                ]
                np.testing.assert_array_equal(u_low[:, column], expected)

    assert not routed.local_frames_by_q[0].flags.writeable
    assert not routed.routed_projectors_by_q[0][0].flags.writeable
    tampered_frames = [frame.copy() for frame in routed.local_frames_by_q]
    tampered_frames[0][0, 0] += 0.25
    tampered = replace(routed, local_frames_by_q=tuple(tampered_frames))
    with pytest.raises(GammaRoutingError) as stale:
        assemble_gamma_routed_projectors(
            tampered,
            layout=layout,
            thresholds=_thresholds(),
            include_high=False,
        )
    assert stale.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY

    duplicate_joint = replace(
        routed,
        joint_band_indices=(0, 0, 2, 8),
    )
    with pytest.raises(GammaRoutingError) as duplicate:
        assemble_gamma_routed_projectors(
            duplicate_joint,
            layout=layout,
            thresholds=_thresholds(),
            include_high=False,
        )
    assert duplicate.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_assembler_rejects_hash_consistent_positive_negative_group_swap() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    assert routed.group_dimensions == (2, 2)
    swapped_frames = tuple(
        np.column_stack((frame[:, 2:], frame[:, :2]))
        for frame in routed.local_frames_by_q
    )
    swapped_references = tuple(
        (groups[1], groups[0])
        for groups in routed.reference_frames_by_q_group
    )
    swapped_projectors = tuple(
        (groups[1], groups[0])
        for groups in routed.routed_projectors_by_q
    )
    frame_tensor = np.stack(swapped_frames, axis=0)
    reference_tensor = np.stack(
        [np.column_stack(groups) for groups in swapped_references], axis=0
    )
    swapped = replace(
        routed,
        local_frames_by_q=swapped_frames,
        reference_frames_by_q_group=swapped_references,
        routed_projectors_by_q=swapped_projectors,
        frame_hash=gamma_layout_module._hash_array(frame_tensor),
        reference_frame_hash=gamma_layout_module._hash_array(reference_tensor),
    )
    with pytest.raises(GammaRoutingError) as exc_info:
        assemble_gamma_routed_projectors(
            swapped,
            layout=layout,
            thresholds=_thresholds(),
            include_high=False,
        )
    assert exc_info.value.reason is CandidateRejectionReason.PROJECTOR_FRAME_RANK


def test_routed_frame_route_gaps_require_strict_finite_numeric_values() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    with pytest.raises(ValueError, match="strict finite numeric"):
        replace(routed, route_gaps=("1.0", 1.0))
    with pytest.raises(ValueError, match="strict finite numeric"):
        replace(routed, route_gaps=())
    for invalid in (np.nan, True):
        with pytest.raises(ValueError, match="strict finite numeric"):
            replace(routed, route_gaps=(invalid, 1.0))
    with pytest.raises(ValueError, match="strict finite numeric"):
        replace(routed, route_gaps=(10**10000, 1.0))
    normalized = replace(routed, route_gaps=(np.float32(1.0), np.int64(1)))
    assert normalized.route_gaps == (1.0, 1.0)
    assert all(type(gap) is float for gap in normalized.route_gaps)


def test_assembler_rejects_float_group_offsets_with_typed_error() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    invalid = replace(routed, group_offsets=(0.0, 2.0, 4.0))
    with pytest.raises(GammaRoutingError) as exc_info:
        assemble_gamma_routed_projectors(
            invalid,
            layout=layout,
            thresholds=_thresholds(),
            include_high=False,
        )
    assert exc_info.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_assembler_rejects_missing_group_projector_with_typed_error() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    invalid_projectors = (
        (routed.routed_projectors_by_q[0][0],),
        routed.routed_projectors_by_q[1],
    )
    invalid = replace(routed, routed_projectors_by_q=invalid_projectors)
    with pytest.raises(GammaRoutingError) as exc_info:
        assemble_gamma_routed_projectors(
            invalid,
            layout=layout,
            thresholds=_thresholds(),
            include_high=False,
        )
    assert exc_info.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_mixed_type_certified_q_permutation_fails_closed() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    action = replace(_identity_action(layout), q_permutation=(0, "1"))
    with pytest.raises(GammaRoutingError) as exc_info:
        certify_routed_covariance(
            routed.routed_projectors_by_q,
            layout=layout,
            actions=(action,),
            thresholds=_thresholds(),
        )
    assert exc_info.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_assembler_certifies_high_frame_orthogonality_and_completeness(monkeypatch) -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    high_rank = layout.same_q_dimension - sum(routed.group_dimensions)
    monkeypatch.setattr(
        "kp.blocks.gamma_layout.scipy.linalg.null_space",
        lambda _matrix: np.zeros(
            (layout.same_q_dimension, high_rank), dtype=np.complex128
        ),
    )
    with pytest.raises(GammaRoutingError) as exc_info:
        assemble_gamma_routed_projectors(
            routed,
            layout=layout,
            thresholds=_thresholds(),
            include_high=True,
        )
    assert exc_info.value.reason is CandidateRejectionReason.PROJECTOR_FRAME_RANK


def test_routing_is_covariant_under_l2_orbital_swap() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    local_swap = np.eye(layout.same_q_dimension, dtype=np.complex128)
    for spin in range(2):
        start = spin * 6 + 2
        local_swap[[start, start + 1], :] = local_swap[[start + 1, start], :]
    swapped_vecs = tuple(local_swap.conj().T @ vectors for vectors in vecs)
    swapped = build_gamma_routed_frames(
        evals,
        swapped_vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )

    assert swapped.group_dimensions == routed.group_dimensions
    for old_projectors, new_projectors in zip(
        routed.routed_projectors_by_q,
        swapped.routed_projectors_by_q,
        strict=True,
    ):
        for old, new in zip(old_projectors, new_projectors, strict=True):
            np.testing.assert_allclose(new, local_swap.conj().T @ old @ local_swap, atol=1.0e-12)


def test_source_group_routing_ignores_physical_layer_composition_changes() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    rotated = [vectors.copy() for vectors in vecs]
    layer_rotation = np.eye(layout.same_q_dimension, dtype=np.complex128)
    theta = 0.37
    for spin in range(2):
        indices = [spin * 6 + 2, spin * 6 + 4]
        layer_rotation[np.ix_(indices, indices)] = [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    rotated = tuple(layer_rotation @ vectors for vectors in rotated)

    routed = build_gamma_routed_frames(
        evals,
        rotated,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    assert routed.group_dimensions == (2, 2)
    group1 = routed.routed_projectors_by_q[0][1]
    np.testing.assert_allclose(np.trace(layout.source_group_local_projector(0) @ group1), 0.0, atol=1.0e-12)
    assert float(np.trace(layout.physical_layer_local_projector(1) @ group1).real) < 2.0
    assert float(np.trace(layout.physical_layer_local_projector(2) @ group1).real) > 0.0


def test_whole_frame_mix_is_rejected_while_groupwise_routing_preserves_projectors() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    anchor = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    angle = 0.31
    mix = np.eye(4, dtype=np.complex128)
    mix[np.ix_([0, 2], [0, 2])] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    whole_mixed = tuple(frame @ mix for frame in anchor.local_frames_by_q)
    for original, mixed in zip(anchor.local_frames_by_q, whole_mixed, strict=True):
        np.testing.assert_allclose(
            mixed @ mixed.conj().T,
            original @ original.conj().T,
            atol=1.0e-12,
        )
        assert not np.allclose(
            mixed[:, :2] @ mixed[:, :2].conj().T,
            anchor.routed_projectors_by_q[0][0],
        )

    tampered = replace(anchor, local_frames_by_q=whole_mixed)
    with pytest.raises(GammaRoutingError) as stale:
        assemble_gamma_routed_projectors(
            tampered,
            layout=layout,
            thresholds=_thresholds(),
            include_high=False,
        )
    assert stale.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY

    with pytest.raises(GammaRoutingError) as cross_group_anchor:
        build_gamma_routed_frames(
            evals,
            vecs,
            joint_band_indices=joint,
            layout=layout,
            thresholds=_thresholds(),
            anchor_frames=whole_mixed,
            require_complete_clusters=False,
        )
    assert (
        cross_group_anchor.value.reason
        is CandidateRejectionReason.PROJECTOR_FRAME_RANK
    )


def test_routed_covariance_checks_every_required_route() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    identity = certify_gamma_raw_action(
        name="identity",
        full_action=np.eye(layout.full_dimension, dtype=np.complex128),
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=_thresholds(),
        tapw_source_basis_hash="tapw-source-basis-a",
    )
    residuals = certify_routed_covariance(
        routed.routed_projectors_by_q,
        layout=layout,
        actions=(identity,),
        thresholds=_thresholds(),
    )
    assert residuals == (("identity", 0, 0.0), ("identity", 1, 0.0))


def test_routed_covariance_rejects_wrong_q_route_nan_and_threshold_laundering() -> None:
    layout = _layout()
    identity_local = np.eye(layout.same_q_dimension, dtype=np.complex128)
    swap = certify_gamma_raw_action(
        name="q-swap",
        full_action=_full_action_from_local_routes(
            layout, (identity_local, identity_local), (1, 0)
        ),
        layout=layout,
        q_permutations=((1, 0), (1, 0)),
        sector_map=(0, 1),
        antiunitary=False,
        thresholds=_thresholds(),
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    projectors = []
    for q in range(layout.q_count):
        groups = []
        for group in range(2):
            frame = np.zeros((layout.same_q_dimension, 1), dtype=np.complex128)
            row = layout.source_group_local_rows(group)[q]
            frame[row, 0] = 1.0
            groups.append(frame @ frame.conj().T)
        projectors.append(tuple(groups))
    with pytest.raises(GammaRoutingError) as wrong_route:
        certify_routed_covariance(
            projectors,
            layout=layout,
            actions=(swap,),
            thresholds=_thresholds(),
        )
    assert wrong_route.value.reason is CandidateRejectionReason.SOURCE_GROUP_ROUTE_COVARIANCE

    poisoned = [[matrix.copy() for matrix in groups] for groups in projectors]
    poisoned[0][0][0, 0] = np.nan
    with pytest.raises(GammaRoutingError):
        certify_routed_covariance(
            poisoned,
            layout=layout,
            actions=(swap,),
            thresholds=_thresholds(),
        )

    changed = replace(_thresholds(), route_covariance=2.0e-9)
    with pytest.raises(GammaRoutingError) as laundering:
        certify_routed_covariance(
            projectors,
            layout=layout,
            actions=(swap,),
            thresholds=changed,
        )
    assert laundering.value.reason is CandidateRejectionReason.HANDOFF_IDENTITY


def test_exchange_covariance_requires_equal_routed_group_rank() -> None:
    layout = _equal_group_layout()
    local = np.zeros(
        (layout.same_q_dimension, layout.same_q_dimension), dtype=np.complex128
    )
    group0 = layout.source_group_local_rows(0)
    group1 = layout.source_group_local_rows(1)
    local[np.ix_(group1, group0)] = np.eye(group0.size)
    local[np.ix_(group0, group1)] = np.eye(group1.size)
    routed = []
    for _q in range(layout.q_count):
        p0 = np.zeros((layout.same_q_dimension, layout.same_q_dimension), dtype=np.complex128)
        p1 = p0.copy()
        p0[group0[0], group0[0]] = 1.0
        p1[group1[:2], group1[:2]] = 1.0
        routed.append((p0, p1))
    loose = replace(_thresholds(), route_covariance=10.0, max_rank=8)
    loose_exchange = certify_gamma_raw_action(
        name="exchange",
        full_action=_full_action_from_local_routes(layout, (local, local), (0, 1)),
        layout=layout,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(1, 0),
        antiunitary=False,
        thresholds=loose,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
    )
    with pytest.raises(GammaRoutingError) as rank_change:
        certify_routed_covariance(
            routed,
            layout=layout,
            actions=(loose_exchange,),
            thresholds=loose,
        )
    assert rank_change.value.reason is CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE


def test_real_routed_assembler_preserves_kramers_spectrum_and_wrong_rows_fail_certificate() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        require_complete_clusters=False,
    )
    u_low, _ = assemble_gamma_routed_projectors(
        routed,
        layout=layout,
        thresholds=_thresholds(),
        include_high=False,
    )
    local_dim = layout.same_q_dimension
    tr_local = np.zeros((local_dim, local_dim), dtype=np.complex128)
    half = local_dim // 2
    tr_local[:half, half:] = np.eye(half)
    tr_local[half:, :half] = -np.eye(half)
    tr_full = np.zeros((layout.full_dimension, layout.full_dimension), dtype=np.complex128)
    for q in range(layout.q_count):
        rows = layout.same_q_full_rows(q)
        tr_full[np.ix_(rows, rows)] = tr_local
    exact = u_low.conj().T @ tr_full @ u_low.conj()
    heff = np.zeros((u_low.shape[1], u_low.shape[1]), dtype=np.complex128)
    heff[np.diag_indices_from(heff)] = np.repeat([0.2, 0.7, 1.1, 1.6], 2)
    # Enforce the same antiunitary covariance used by the certificate.
    heff = 0.5 * (heff + exact @ heff.conj() @ exact.conj().T)
    eigenvalues = np.linalg.eigvalsh(heff)
    np.testing.assert_allclose(eigenvalues[0::2], eigenvalues[1::2], atol=1.0e-12)

    presentation = MagneticPresentation(
        generators=(MagneticGenerator("TR", True),),
        relations=(
            MagneticRelation("TR^2", lhs=("TR", "TR"), rhs=(), central_phase=-1.0),
        ),
        central_phases=(1.0, -1.0),
        source="synthetic_gamma_routed",
    )

    def certificate(frame: np.ndarray):
        return certify_candidate_symmetries(
            candidate_id="gamma-routed",
            states={0: CandidateProjectionState(u_low=frame, heff=heff)},
            operations={
                "TR": CandidateOperationInput(
                    name="TR",
                    antiunitary=True,
                    d_full=tr_full,
                    pairs=((0, 0),),
                )
            },
            exactified_actions={"TR": exact},
            presentation=presentation,
            required_pairs={"TR": ((0, 0),)},
            thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
        )

    good = certificate(u_low)
    assert good.status is CandidateSymmetryStatus.CERTIFIED
    wrong_permutation = np.arange(layout.full_dimension)
    wrong_permutation[[0, 1]] = wrong_permutation[[1, 0]]
    bad = certificate(u_low[wrong_permutation])
    assert bad.status is CandidateSymmetryStatus.FAILED
    assert (
        bad.operations[0].pairs[0].raw_h_leakage > 1.0e-10
        or bad.operations[0].pairs[0].heff_covariance_residual > 1.0e-10
    )


def test_independent_tr_symmetric_hamiltonian_and_legacy_gamma_rows() -> None:
    layout = _layout()
    local_dim = layout.same_q_dimension
    paired_energies = np.asarray([0.2, 2.0, 0.7, 3.0, 4.0, 5.0])
    local_h = np.diag(np.r_[paired_energies, paired_energies]).astype(np.complex128)
    tr_local = np.zeros((local_dim, local_dim), dtype=np.complex128)
    half = local_dim // 2
    tr_local[:half, half:] = np.eye(half)
    tr_local[half:, :half] = -np.eye(half)
    np.testing.assert_allclose(tr_local @ local_h.conj() @ tr_local.conj().T, local_h)

    h_full = np.zeros((layout.full_dimension, layout.full_dimension), dtype=np.complex128)
    tr_full = np.zeros_like(h_full)
    for q in range(layout.q_count):
        rows = layout.same_q_full_rows(q)
        h_full[np.ix_(rows, rows)] = local_h
        tr_full[np.ix_(rows, rows)] = tr_local
    np.testing.assert_allclose(tr_full @ h_full.conj() @ tr_full.conj().T, h_full)

    eigenvalues_by_q = []
    eigenvectors_by_q = []
    for _q in range(layout.q_count):
        values, vectors = np.linalg.eigh(local_h)
        eigenvalues_by_q.append(values)
        eigenvectors_by_q.append(vectors)
    joint = (0, 1, 2, 3)
    anchors = tuple(
        np.eye(local_dim, dtype=np.complex128)[:, (0, 6, 2, 8)]
        for _ in range(layout.q_count)
    )
    routed = build_gamma_routed_frames(
        eigenvalues_by_q,
        eigenvectors_by_q,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        anchor_frames=anchors,
    )
    u_low, _ = assemble_gamma_routed_projectors(
        routed,
        layout=layout,
        thresholds=_thresholds(),
        include_high=False,
    )
    heff = u_low.conj().T @ h_full @ u_low

    exact = np.zeros((u_low.shape[1], u_low.shape[1]), dtype=np.complex128)
    for group in range(2):
        base = layout.q_count * sum(routed.group_dimensions[:group])
        for q in range(layout.q_count):
            up = base + q
            down = base + layout.q_count + q
            exact[down, up] = -1.0
            exact[up, down] = 1.0
    np.testing.assert_allclose(exact @ heff.conj() @ exact.conj().T, heff)
    spectrum = np.linalg.eigvalsh(heff)
    np.testing.assert_allclose(spectrum[0::2], spectrum[1::2], atol=1.0e-12)

    presentation = MagneticPresentation(
        generators=(MagneticGenerator("TR", True),),
        relations=(
            MagneticRelation("TR^2", lhs=("TR", "TR"), rhs=(), central_phase=-1.0),
        ),
        central_phases=(1.0, -1.0),
        source="independent_gamma_hamiltonian",
    )

    def certificate(frame: np.ndarray):
        return certify_candidate_symmetries(
            candidate_id="gamma-routed-independent",
            states={0: CandidateProjectionState(u_low=frame, heff=heff)},
            operations={
                "TR": CandidateOperationInput(
                    name="TR", antiunitary=True, d_full=tr_full, pairs=((0, 0),)
                )
            },
            exactified_actions={"TR": exact},
            presentation=presentation,
            required_pairs={"TR": ((0, 0),)},
            thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
        )

    assert certificate(u_low).status is CandidateSymmetryStatus.CERTIFIED
    legacy_q_major = np.concatenate(
        [layout.same_q_full_rows(q) for q in range(layout.q_count)]
    )
    wrong_rows = u_low[legacy_q_major, :]
    failed = certificate(wrong_rows)
    assert failed.status is CandidateSymmetryStatus.FAILED
    pair = failed.operations[0].pairs[0]
    assert (
        pair.raw_h_leakage > 1.0e-10
        or pair.exactification_distance > 1.0e-10
        or pair.intertwining_residual > 1.0e-10
    )
