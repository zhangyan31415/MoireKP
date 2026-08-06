from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

import kp.blocks as public_blocks

from kp.blocks.gamma_layout import (
    GammaLayoutError,
    GammaRowLayout,
    GammaRoutingThresholds,
    GammaRoutingError,
    certify_gamma_raw_action,
    close_gamma_projector_clusters,
    cluster_gamma_eigensystem,
    infer_gamma_raw_action_q_permutations,
)
from kp.projection_selection import CandidateRejectionReason


def _layout(*, q1: np.ndarray | None = None, q2: np.ndarray | None = None) -> GammaRowLayout:
    first = np.asarray([[0.0, 0.0], [1.0, 0.0]]) if q1 is None else np.asarray(q1)
    second = first.copy() if q2 is None else np.asarray(q2)
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
    qset = np.zeros((2, 2))
    return GammaRowLayout.build(
        qsets=(qset, qset.copy()),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((2,), (2,)),
        spin_convention="all",
        source_basis_hash="equal-groups",
    )


def _route_inference_layout(q_count: int = 3) -> GammaRowLayout:
    qset = np.zeros((q_count, 2))
    return GammaRowLayout.build(
        qsets=(qset, qset.copy()),
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


def _mgi2_like_gray_zone_case() -> tuple[
    GammaRowLayout,
    GammaRoutingThresholds,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    layout = GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((12,), (12,)),
        spin_convention="all",
        source_basis_hash="mgi2-like-gray-zone",
    )
    thresholds = replace(_thresholds(), energy_different_ev=1.0e-4)
    values = np.r_[
        0.0,
        2.85251e-5,
        np.arange(1, layout.same_q_dimension - 1),
    ]
    permutation = np.arange(layout.same_q_dimension)
    candidate_rows = tuple(layout.source_group_local_rows(0)[:2]) + tuple(
        layout.source_group_local_rows(1)[:2]
    )
    for candidate_row, candidate_band in zip(
        candidate_rows,
        (40, 41, 42, 43),
        strict=True,
    ):
        permutation[candidate_row], permutation[candidate_band] = (
            permutation[candidate_band],
            permutation[candidate_row],
        )
    vectors = np.eye(layout.same_q_dimension, dtype=np.complex128)[:, permutation]
    return layout, thresholds, (values,), (vectors,)


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


def test_gamma_layout_hash_binds_joint_q_order_source_basis_and_spin() -> None:
    layout = _layout()
    jointly_reordered = _layout(q1=np.asarray([[1.0, 0.0], [0.0, 0.0]]))
    qset = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    changed_source = GammaRowLayout.build(
        qsets=(qset, qset.copy()),
        num_layer_list=(1, 2),
        num_orb_per_layer_list=((2,), (2, 2)),
        spin_convention="all",
        source_basis_hash="tapw-source-basis-b",
    )
    assert len(layout.layout_hash) == 64
    assert jointly_reordered.layout_hash != layout.layout_hash
    assert changed_source.layout_hash != layout.layout_hash

    with pytest.raises(GammaLayoutError) as exc_info:
        GammaRowLayout.build(
            qsets=(qset, qset.copy()),
            num_layer_list=(1, 2),
            num_orb_per_layer_list=((2,), (2, 2)),
            spin_convention="up",
            source_basis_hash="tapw-source-basis-a",
        )
    assert exc_info.value.reason.value == "UNSUPPORTED_SPIN_ROUTE"


@pytest.mark.parametrize(
    "second_qset",
    (
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        np.asarray([[0.0, 0.0], [1.0 + 2.0e-12, 0.0]]),
    ),
    ids=("one_sided_reorder", "one_sided_coordinate_drift"),
)
def test_gamma_layout_rejects_one_sided_q_changes_with_typed_error(
    second_qset: np.ndarray,
) -> None:
    first_qset = np.asarray([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(GammaLayoutError) as exc_info:
        _layout(q1=first_qset, q2=second_qset)

    assert exc_info.value.reason is CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH


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


def test_public_gamma_raw_layout_exports_and_rejection_values_are_present() -> None:
    exported = {
        "GammaCertifiedRawAction",
        "GammaEnergyCluster",
        "GammaLayoutError",
        "GammaRoutingError",
        "GammaRoutingThresholds",
        "GammaRowAddress",
        "GammaRowLayout",
        "certify_gamma_raw_action",
        "cluster_gamma_eigensystem",
    }
    assert exported <= set(public_blocks.__all__)
    assert all(hasattr(public_blocks, name) for name in exported)
    required_reasons = {
        "UNSUPPORTED_GAMMA_LAYOUT",
        "UNSUPPORTED_SPIN_ROUTE",
        "INVALID_GAMMA_ROW_LAYOUT",
        "GAMMA_Q_ROUTE_MISMATCH",
        "LOCAL_ACTION_ISOMETRY",
        "RAW_ACTION_ROUTE_LEAKAGE",
        "AMBIGUOUS_ENERGY_CLUSTER",
        "SYMMETRY_CLOSURE_FAILURE",
        "HANDOFF_IDENTITY",
        "CANDIDATE_SYMMETRY_FAILED",
    }
    assert required_reasons <= {
        reason.value
        for reason in CandidateRejectionReason
        if reason.value.isupper()
    }


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


def test_cluster_closure_ignores_gray_zone_outside_candidate_support() -> None:
    layout, thresholds, values, vectors = _mgi2_like_gray_zone_case()
    candidate = (40, 41, 42, 43)

    closed = close_gamma_projector_clusters(
        values,
        vectors,
        layout=layout,
        seed_band_indices=(candidate,),
        actions=(_identity_action(layout, thresholds=thresholds),),
        thresholds=thresholds,
    )

    assert closed.band_indices_by_q == (candidate,)


def test_cluster_closure_rejects_gray_zone_touching_candidate_support() -> None:
    layout, thresholds, values, vectors = _mgi2_like_gray_zone_case()

    with pytest.raises(GammaRoutingError, match="energy-cluster gray zone") as rejected:
        close_gamma_projector_clusters(
            values,
            vectors,
            layout=layout,
            seed_band_indices=((1,),),
            actions=(_identity_action(layout, thresholds=thresholds),),
            thresholds=thresholds,
        )

    assert rejected.value.reason is CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER


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


def test_certified_raw_action_freezes_source_and_local_blocks() -> None:
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
