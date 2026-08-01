from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

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
            seed_band_indices=((0,), (0,)),
            actions=(action,),
            thresholds=_thresholds(),
        )
    assert exc_info.value.reason.value == "AMBIGUOUS_CLUSTER_CAPTURE"


def _routed_eigensystems(layout: GammaRowLayout):
    dim = layout.same_q_dimension
    evals = tuple(np.arange(dim, dtype=float) for _ in range(layout.q_count))
    vecs = tuple(np.eye(dim, dtype=np.complex128) for _ in range(layout.q_count))
    # One Kramers pair per source group: up/down orbital 0 in group 0 and group 1.
    joint = (0, 6, 2, 8)
    return evals, vecs, joint


def test_routed_frames_and_real_assembler_are_orthonormal_and_complete() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        reference_q_index=0,
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


def test_routing_is_covariant_under_l2_orbital_swap() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        reference_q_index=0,
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
        reference_q_index=0,
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
        reference_q_index=0,
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
        reference_q_index=0,
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
    with pytest.raises(GammaRoutingError, match="group slice"):
        assemble_gamma_routed_projectors(
            tampered,
            layout=layout,
            thresholds=_thresholds(),
            include_high=False,
        )

    # Re-routing a joint frame before groupwise alignment recovers the certified slices.
    rebuilt = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        reference_q_index=0,
        anchor_frames=anchor.local_frames_by_q,
        require_complete_clusters=False,
    )
    for expected, actual in zip(
        anchor.routed_projectors_by_q,
        rebuilt.routed_projectors_by_q,
        strict=True,
    ):
        np.testing.assert_allclose(actual[0], expected[0], atol=1.0e-12)
        np.testing.assert_allclose(actual[1], expected[1], atol=1.0e-12)


def test_routed_covariance_checks_every_required_route() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        reference_q_index=0,
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
        actions=(identity,),
        thresholds=_thresholds(),
    )
    assert residuals == (("identity", 0, 0.0), ("identity", 1, 0.0))


def test_real_routed_assembler_preserves_kramers_spectrum_and_wrong_rows_fail_certificate() -> None:
    layout = _layout()
    evals, vecs, joint = _routed_eigensystems(layout)
    routed = build_gamma_routed_frames(
        evals,
        vecs,
        joint_band_indices=joint,
        layout=layout,
        thresholds=_thresholds(),
        reference_q_index=0,
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
