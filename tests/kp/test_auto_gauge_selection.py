from __future__ import annotations

import numpy as np
import pytest

from kp.basis.selection import (
    build_reference_projectors_from_rows,
    compute_leverage_scores,
    GaugeCandidateSymmetryMetrics,
    polar_align_low_subspace,
    select_anchor_rows_qrcp,
    select_gauge_candidate_by_symmetry,
)
from kp.blocks.blocks import (
    _assign_anchor_references_to_bands,
    _canonical_resolved_anchor_key,
    _complete_gamma_spinful_reference_terms,
    _reference_overlap_singular_values,
    resolve_project_gauge_anchor_candidates,
)


def test_one_dimensional_state_selects_largest_leverage_row() -> None:
    u_low = np.array([[0.1], [0.8], [0.6]], dtype=np.complex128)
    u_low = u_low / np.linalg.norm(u_low)

    scores = compute_leverage_scores(u_low)
    selection = select_anchor_rows_qrcp(u_low, n_anchors=1)

    assert int(np.argmax(scores)) == 1
    assert selection.selected_rows == [1]
    assert selection.sigma_min > 0.0
    assert selection.condition_number == pytest.approx(1.0)


def test_qrcp_selects_full_rank_rows_for_degenerate_subspace() -> None:
    u_low = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    u_low /= np.sqrt(2.0)

    selection = select_anchor_rows_qrcp(u_low, n_anchors=2)

    assert selection.selected_rows == [0, 1]
    assert selection.rank == 2
    assert selection.sigma_min > 0.0
    assert np.isfinite(selection.condition_number)


def test_metric_leverage_matches_lowdin_transformed_rows() -> None:
    u_low = np.array([[1.0], [0.0]], dtype=np.complex128)
    overlap = np.diag([4.0, 1.0]).astype(np.complex128)
    u_metric = u_low / 2.0

    metric_scores = compute_leverage_scores(
        u_metric,
        overlap=overlap,
        basis_is_orthonormal=False,
    )
    lowdin_scores = compute_leverage_scores(
        np.diag(np.sqrt(np.diag(overlap))) @ u_metric,
    )

    np.testing.assert_allclose(metric_scores, lowdin_scores)
    np.testing.assert_allclose(metric_scores, [1.0, 0.0])


def test_candidate_tie_breaker_is_deterministic() -> None:
    u_low = np.eye(3, 2, dtype=np.complex128)

    first = select_anchor_rows_qrcp(u_low, n_anchors=2, candidate_rows=[1, 0, 2])
    second = select_anchor_rows_qrcp(u_low, n_anchors=2, candidate_rows=[2, 0, 1])

    assert first.selected_rows == [0, 1]
    assert second.selected_rows == [0, 1]


def test_rank_deficient_candidates_fail_clearly() -> None:
    u_low = np.eye(3, 2, dtype=np.complex128)

    with pytest.raises(ValueError, match="rank deficient"):
        select_anchor_rows_qrcp(u_low, n_anchors=2, candidate_rows=[0])


def test_raw_delta_projectors_and_polar_alignment() -> None:
    u_low = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0j],
            [0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    phi = build_reference_projectors_from_rows([0, 1], basis_size=3)
    aligned, rotation, singular_values = polar_align_low_subspace(u_low, phi)

    np.testing.assert_allclose(phi, np.eye(3, 2, dtype=np.complex128))
    np.testing.assert_allclose(aligned, phi)
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(2))
    np.testing.assert_allclose(singular_values, [1.0, 1.0])


def test_gamma_spinful_auto_gauge_completes_adjacent_chiral_pairs() -> None:
    u_low = np.zeros((4, 2), dtype=np.complex128)
    u_low[1, 0] = 1.0 / np.sqrt(2.0)
    u_low[0, 0] = -1.0j / np.sqrt(2.0)
    u_low[3, 1] = 1.0 / np.sqrt(2.0)
    u_low[2, 1] = 1.0j / np.sqrt(2.0)

    references, details, warnings = _complete_gamma_spinful_reference_terms(
        u_low,
        [1, 3],
        segments=[(0, 2), (2, 4)],
    )
    assigned, scores = _assign_anchor_references_to_bands(u_low, references)
    singular_values = _reference_overlap_singular_values(u_low, assigned)

    assert warnings == []
    assert [detail["partner_row"] for detail in details] == [0, 2]
    assert [len(ref) for ref in assigned] == [2, 2]
    assert min(scores) > 0.99
    np.testing.assert_allclose(singular_values, [1.0, 1.0], atol=1.0e-12)


def test_gamma_spinful_auto_gauge_falls_back_to_same_spin_layer_partner() -> None:
    u_low = np.zeros((8, 2), dtype=np.complex128)
    u_low[1, 0] = 0.5
    u_low[3, 0] = 0.5
    u_low[5, 1] = 0.5j
    u_low[7, 1] = 0.5j

    references, details, warnings = _complete_gamma_spinful_reference_terms(
        u_low,
        [1, 5],
        segments=[(0, 2), (2, 4), (4, 6), (6, 8)],
    )
    ordered = sorted(references, key=lambda ref: min(row for row, _coef in ref))
    singular_values = _reference_overlap_singular_values(u_low, ordered)

    assert warnings == []
    assert [detail["partner_scope"] for detail in details] == [
        "same_spin_layer_exchange",
        "same_spin_layer_exchange",
    ]
    assert ordered == [
        [(1, 1.0 + 0.0j), (3, 1.0 + 0.0j)],
        [(5, 1.0 + 0.0j), (7, 1.0 + 0.0j)],
    ]
    np.testing.assert_allclose(singular_values, [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])


def test_symmetry_scored_candidate_loop_selects_valid_low_complexity_candidate() -> None:
    decision = select_gauge_candidate_by_symmetry(
        [
            GaugeCandidateSymmetryMetrics(
                candidate_id="bad_branch",
                exactification_distance_by_op={"TR": 1.414, "C3z": 0.0},
                active_term_count=200,
            ),
            GaugeCandidateSymmetryMetrics(
                candidate_id="good",
                exactification_distance_by_op={"TR": 1.0e-11, "C3z": 8.0e-7},
                active_term_count=216,
            ),
            GaugeCandidateSymmetryMetrics(
                candidate_id="bloated",
                exactification_distance_by_op={"TR": 2.0e-11, "C3z": 7.0e-7},
                active_term_count=300,
            ),
        ],
        max_exactification_distance=1.0e-3,
    )

    assert decision.selected.candidate_id == "good"
    assert decision.rankings[0]["candidate_id"] == "good"
    assert decision.rankings[-1]["candidate_id"] == "bad_branch"
    assert decision.rankings[-1]["status"] == "rejected"


def test_symmetry_scored_candidate_loop_fails_when_all_candidates_have_bad_residuals() -> None:
    with pytest.raises(ValueError, match="No gauge candidate passed symmetry validation"):
        select_gauge_candidate_by_symmetry(
            [
                GaugeCandidateSymmetryMetrics(
                    candidate_id="bad_tr",
                    exactification_distance_by_op={"TR": 1.414},
                    active_term_count=100,
                ),
                GaugeCandidateSymmetryMetrics(
                    candidate_id="bad_c2",
                    exactification_distance_by_op={"C2": 1.0e-2},
                    active_term_count=90,
                ),
            ],
            max_exactification_distance=1.0e-3,
        )


def test_symmetry_scored_candidate_loop_fails_on_ambiguous_candidates() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        select_gauge_candidate_by_symmetry(
            [
                GaugeCandidateSymmetryMetrics(
                    candidate_id="candidate_a",
                    exactification_distance_by_op={"TR": 1.0e-10, "C3z": 5.0e-7},
                    active_term_count=216,
                ),
                GaugeCandidateSymmetryMetrics(
                    candidate_id="candidate_b",
                    exactification_distance_by_op={"TR": 2.0e-10, "C3z": 5.1e-7},
                    active_term_count=216,
                ),
            ],
            max_exactification_distance=1.0e-3,
        )


def test_symmetry_scored_candidate_loop_uses_phase_branch_stability() -> None:
    decision = select_gauge_candidate_by_symmetry(
        [
            GaugeCandidateSymmetryMetrics(
                candidate_id="phase_unstable",
                exactification_distance_by_op={"T": 1.0e-10},
                active_term_count=216,
                phase_branch_distance_by_op={"T": 4.0},
            ),
            GaugeCandidateSymmetryMetrics(
                candidate_id="phase_stable",
                exactification_distance_by_op={"T": 1.1e-10},
                active_term_count=216,
                phase_branch_distance_by_op={"T": 0.05},
            ),
        ],
        max_exactification_distance=1.0e-3,
    )

    assert decision.selected.candidate_id == "phase_stable"
    assert decision.rankings[0]["max_phase_branch_distance"] == pytest.approx(0.05)


def test_symmetry_scored_candidate_loop_uses_layout_priority_when_metrics_tie() -> None:
    decision = select_gauge_candidate_by_symmetry(
        [
            GaugeCandidateSymmetryMetrics(
                candidate_id="layout_swapped",
                exactification_distance_by_op={"TR": 1.0e-10},
                phase_branch_distance_by_op={"TR": 0.0},
                support_off_by_op={"TR": 1.0e-12},
                metadata={"candidate_priority": 1},
            ),
            GaugeCandidateSymmetryMetrics(
                candidate_id="model_frame",
                exactification_distance_by_op={"TR": 1.0e-10},
                phase_branch_distance_by_op={"TR": 0.0},
                support_off_by_op={"TR": 1.0e-12},
                metadata={"candidate_priority": 0},
            ),
        ],
        max_exactification_distance=1.0e-3,
    )

    assert decision.selected.candidate_id == "model_frame"
    assert decision.rankings[0]["candidate_priority"] == 0


def test_gamma_spinful_auto_gauge_exposes_multiple_generic_candidates() -> None:
    trial = np.eye(8, dtype=np.complex128)
    trial[:, 0] = 0.0
    trial[1, 0] = 1.0 / np.sqrt(2.0)
    trial[3, 0] = 1.0 / np.sqrt(2.0)
    trial[:, 1] = 0.0
    trial[5, 1] = 1.0 / np.sqrt(2.0)
    trial[7, 1] = 1.0 / np.sqrt(2.0)
    unitary, _ = np.linalg.qr(trial)
    ham = unitary @ np.diag(np.arange(8, dtype=float)) @ unitary.conj().T

    candidates = resolve_project_gauge_anchor_candidates(
        ham,
        q_count=1,
        orb_per_layer0=2,
        num_layer_list=[1, 1],
        spin="all",
        Qlayer_list=[[np.zeros((1, 2), dtype=float)], [np.zeros((1, 2), dtype=float)]],
        num_orb_per_layer_list=[[2], [2]],
        nlow_state_list=[[0], [1]],
        norb_fix_list="auto",
        gauge_config="auto",
        mode="gamma",
    )

    candidate_ids = [candidate.candidate_id for candidate in candidates]

    assert candidate_ids[0] == "gamma_model_frame"
    assert "gamma_completed_overlap_assignment" in candidate_ids
    assert "qrcp_delta_overlap_assignment" in candidate_ids
    assert len({tuple(map(str, candidate.resolved_norb_fix_list)) for candidate in candidates}) >= 2
    for candidate in candidates:
        payload = candidate.report.to_dict()
        assert payload["gauge_mode"] == "auto_scdm"
        assert payload["symmetry_closure_quality"]["status"] == "not_available"
        assert "MoTe2" not in str(payload)
        assert "MgI2" not in str(payload)


def test_auto_gauge_candidate_key_ignores_reference_term_order() -> None:
    left = [[[[52, 1.0], [38, 1.0]]]]
    right = [[[[38, 1.0], [52, 1.0]]]]

    assert _canonical_resolved_anchor_key(left) == _canonical_resolved_anchor_key(right)
