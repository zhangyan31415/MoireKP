from __future__ import annotations

import numpy as np
import pytest

from kp import blocks as blocks_mod
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
from kp.blocks.gamma_layout import GammaRowLayout
from kp.identity import hash_array


def _gamma_model_mutability_records():
    layout = GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(4, dtype=np.int64)),
    )
    eigenvalues = np.arange(4, dtype=float)
    eigenvectors = np.eye(4, dtype=np.complex128)
    anchor = blocks_mod.build_gamma_model_anchor_spec(
        reference_eigenvalues_by_q=(eigenvalues,),
        reference_eigenvectors_by_q=(eigenvectors,),
        joint_band_indices=(0, 1, 2, 3),
        group_ranks=(2, 2),
        layout=layout,
    )
    frames = blocks_mod.build_gamma_model_frames(
        eigenvalues_by_q=(eigenvalues,),
        eigenvectors_by_q=(eigenvectors,),
        layout=layout,
        anchor_spec=anchor,
    )
    return anchor, frames


def test_gamma_model_frame_aligns_complete_joint_space_with_full_u4() -> None:
    layout = GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(4, dtype=np.int64)),
    )
    eigenvalues = (np.asarray([-4.0, -3.0, -2.0, -1.0]),)
    eigenframe = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.complex128,
    ) / 2.0

    anchor_spec = blocks_mod.build_gamma_model_anchor_spec(
        reference_eigenvalues_by_q=eigenvalues,
        reference_eigenvectors_by_q=(eigenframe,),
        joint_band_indices=(0, 1, 2, 3),
        group_ranks=(2, 2),
        layout=layout,
    )
    first = blocks_mod.build_gamma_model_frames(
        eigenvalues_by_q=eigenvalues,
        eigenvectors_by_q=(eigenframe,),
        layout=layout,
        anchor_spec=anchor_spec,
    )
    second = blocks_mod.build_gamma_model_frames(
        eigenvalues_by_q=eigenvalues,
        eigenvectors_by_q=(eigenframe,),
        layout=layout,
        anchor_spec=anchor_spec,
    )

    model_frame = first.local_frames_by_q[0]
    expected = np.eye(4, dtype=np.complex128)[:, [3, 1, 2, 0]]
    assert not model_frame.flags.writeable
    assert not first.alignment_unitaries_by_q[0].flags.writeable
    np.testing.assert_allclose(model_frame, expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        model_frame,
        second.local_frames_by_q[0],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        model_frame.conj().T @ model_frame,
        np.eye(4),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert anchor_spec.model_column_order == ((0, 0), (0, 1), (1, 0), (1, 1))

    routing_frame = np.eye(4, dtype=np.complex128)[:, [0, 2, 1, 3]]
    routing_to_model = routing_frame.conj().T @ model_frame
    assert np.linalg.norm(routing_to_model[:2, 2:]) > 0.9
    assert np.linalg.norm(routing_to_model[2:, :2]) > 0.9


def test_gamma_model_anchor_reference_validation_is_explicit() -> None:
    layout = GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((2,), (2,)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(8, dtype=np.int64)),
    )
    eigenvalues = np.arange(8, dtype=float)
    reference_vectors = np.eye(8, dtype=np.complex128)
    anchor_spec = blocks_mod.build_gamma_model_anchor_spec(
        reference_eigenvalues_by_q=(eigenvalues,),
        reference_eigenvectors_by_q=(reference_vectors,),
        joint_band_indices=(0, 1, 2, 3),
        group_ranks=(2, 2),
        layout=layout,
    )

    blocks_mod.validate_gamma_model_anchor_reference(
        reference_eigenvalues_by_q=(eigenvalues,),
        reference_eigenvectors_by_q=(reference_vectors,),
        layout=layout,
        anchor_spec=anchor_spec,
    )

    changed_energies = eigenvalues.copy()
    changed_energies[0] += 0.25
    with pytest.raises(ValueError, match="reference eigenvalues"):
        blocks_mod.validate_gamma_model_anchor_reference(
            reference_eigenvalues_by_q=(changed_energies,),
            reference_eigenvectors_by_q=(reference_vectors,),
            layout=layout,
            anchor_spec=anchor_spec,
        )

    angle = 0.2
    identity = np.eye(4, dtype=np.complex128)
    changed_vectors = np.block(
        [
            [np.cos(angle) * identity, -np.sin(angle) * identity],
            [np.sin(angle) * identity, np.cos(angle) * identity],
        ]
    )
    with pytest.raises(ValueError, match="reference projector"):
        blocks_mod.validate_gamma_model_anchor_reference(
            reference_eigenvalues_by_q=(eigenvalues,),
            reference_eigenvectors_by_q=(changed_vectors,),
            layout=layout,
            anchor_spec=anchor_spec,
        )

    other_k = blocks_mod.build_gamma_model_frames(
        eigenvalues_by_q=(eigenvalues,),
        eigenvectors_by_q=(changed_vectors,),
        layout=layout,
        anchor_spec=anchor_spec,
    )
    np.testing.assert_allclose(
        other_k.local_frames_by_q[0] @ other_k.local_frames_by_q[0].conj().T,
        changed_vectors[:, :4] @ changed_vectors[:, :4].conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_gamma_model_anchor_recursively_freezes_mutable_sequences() -> None:
    anchor, _ = _gamma_model_mutability_records()

    anchor_kwargs = {name: getattr(anchor, name) for name in anchor.__dataclass_fields__}
    mutable_bands = list(anchor.joint_band_indices)
    mutable_references = [
        [list(term) for term in reference]
        for reference in anchor.resolved_reference_terms
    ]
    anchor_kwargs.update(
        joint_band_indices=mutable_bands,
        group_ranks=list(anchor.group_ranks),
        model_column_order=[list(value) for value in anchor.model_column_order],
        resolved_reference_terms=mutable_references,
        selected_rows=list(anchor.selected_rows),
        reference_singular_values=list(anchor.reference_singular_values),
        reference_eigenvalues=list(anchor.reference_eigenvalues),
        warnings=list(anchor.warnings),
    )
    frozen_anchor = blocks_mod.GammaModelAnchorSpec(**anchor_kwargs)
    mutable_bands.append(99)
    mutable_references[0].append([3, 1.0 + 0.0j])

    assert frozen_anchor.joint_band_indices == anchor.joint_band_indices
    assert frozen_anchor.resolved_reference_terms == anchor.resolved_reference_terms
    assert isinstance(frozen_anchor.joint_band_indices, tuple)
    assert isinstance(frozen_anchor.resolved_reference_terms[0], tuple)
    assert not hasattr(frozen_anchor.joint_band_indices, "append")

    invalid_anchor_kwargs = {
        name: getattr(anchor, name) for name in anchor.__dataclass_fields__
    }
    invalid_references = [
        [list(term) for term in reference]
        for reference in anchor.resolved_reference_terms
    ]
    invalid_references[0][0][0] = float(invalid_references[0][0][0])
    invalid_anchor_kwargs["resolved_reference_terms"] = invalid_references
    with pytest.raises(ValueError, match="reference row.*strict integer"):
        blocks_mod.GammaModelAnchorSpec(**invalid_anchor_kwargs)


def test_gamma_model_frames_recursively_freeze_mutable_sequences() -> None:
    _, frames = _gamma_model_mutability_records()

    frame_kwargs = {name: getattr(frames, name) for name in frames.__dataclass_fields__}
    mutable_frames = [np.array(value, copy=True) for value in frames.local_frames_by_q]
    mutable_quality = [list(value) for value in frames.alignment_singular_values_by_q]
    frame_kwargs.update(
        joint_band_indices=list(frames.joint_band_indices),
        group_ranks=list(frames.group_ranks),
        local_frames_by_q=mutable_frames,
        alignment_unitaries_by_q=[
            np.array(value, copy=True) for value in frames.alignment_unitaries_by_q
        ],
        alignment_singular_values_by_q=mutable_quality,
        orthonormality_residuals_by_q=list(frames.orthonormality_residuals_by_q),
        projector_residuals_by_q=list(frames.projector_residuals_by_q),
        frame_hashes_by_q=list(frames.frame_hashes_by_q),
        alignment_hashes_by_q=list(frames.alignment_hashes_by_q),
    )
    frozen_frames = blocks_mod.GammaModelFrames(**frame_kwargs)
    mutable_frames.append(np.eye(4, dtype=np.complex128))
    mutable_quality[0].append(0.0)

    assert len(frozen_frames.local_frames_by_q) == 1
    assert frozen_frames.alignment_singular_values_by_q == frames.alignment_singular_values_by_q
    assert isinstance(frozen_frames.alignment_singular_values_by_q, tuple)
    assert isinstance(frozen_frames.alignment_singular_values_by_q[0], tuple)
    assert not hasattr(frozen_frames.frame_hashes_by_q, "append")


def test_gamma_model_unsorted_joint_bands_use_true_maximum_for_bounds() -> None:
    layout = GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(4, dtype=np.int64)),
    )
    eigenvalues = np.arange(4, dtype=float)
    eigenvectors = np.eye(4, dtype=np.complex128)

    anchor = blocks_mod.build_gamma_model_anchor_spec(
        reference_eigenvalues_by_q=(eigenvalues,),
        reference_eigenvectors_by_q=(eigenvectors,),
        joint_band_indices=(3, 0, 1, 2),
        group_ranks=(2, 2),
        layout=layout,
    )
    assert anchor.joint_band_indices == (3, 0, 1, 2)

    with pytest.raises(
        IndexError,
        match="Gamma model joint band 4 outside Q 0 eigensystem size 4",
    ):
        blocks_mod.build_gamma_model_anchor_spec(
            reference_eigenvalues_by_q=(eigenvalues,),
            reference_eigenvectors_by_q=(eigenvectors,),
            joint_band_indices=(4, 0, 1, 2),
            group_ranks=(2, 2),
            layout=layout,
        )


def test_gamma_model_anchor_spec_preserves_legacy_resolved_references() -> None:
    layout = GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(4, dtype=np.int64)),
    )
    eigenframe = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.complex128,
    ) / 2.0
    hamiltonian = eigenframe @ np.diag([-4.0, -3.0, -2.0, -1.0]) @ eigenframe.conj().T
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    legacy, legacy_report = blocks_mod.resolve_project_gauge_anchors(
        hamiltonian,
        q_count=1,
        orb_per_layer0=1,
        num_layer_list=[1, 1],
        spin="all",
        Qlayer_list=[[np.zeros((1, 2))], [np.zeros((1, 2))]],
        num_orb_per_layer_list=[[1], [1]],
        nlow_state_list=[[0, 1], [2, 3]],
        norb_fix_list="auto",
        gauge_config="auto",
        mode="gamma",
    )
    anchor_spec = blocks_mod.build_gamma_model_anchor_spec(
        reference_eigenvalues_by_q=(eigenvalues,),
        reference_eigenvectors_by_q=(eigenvectors,),
        joint_band_indices=(0, 1, 2, 3),
        group_ranks=(2, 2),
        layout=layout,
    )

    assert anchor_spec.resolved_norb_fix_list == legacy
    legacy_payload = legacy_report.to_dict()
    assert legacy_payload["resolved_norb_fix_list"] == legacy
    assert legacy_payload["selections"][0]["reference_ordering"] == "gamma_model_frame"
    np.testing.assert_allclose(
        legacy_payload["selections"][0]["reference_singular_values"],
        np.ones(4),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert anchor_spec.reference_q == 0
    assert anchor_spec.layout_hash == layout.layout_hash

    payload = anchor_spec.to_payload()
    restored = blocks_mod.GammaModelAnchorSpec.from_payload(payload)
    assert restored == anchor_spec
    tampered = dict(payload)
    tampered["joint_band_indices"] = [0, 1, 2, 5]
    with pytest.raises(ValueError, match="identity hash mismatch"):
        blocks_mod.GammaModelAnchorSpec.from_payload(tampered)
    with pytest.raises(ValueError, match="strict integers"):
        blocks_mod.build_gamma_model_anchor_spec(
            reference_eigenvalues_by_q=(eigenvalues,),
            reference_eigenvectors_by_q=(eigenvectors,),
            joint_band_indices=(0, 1, 2, 3.0),
            group_ranks=(2, 2),
            layout=layout,
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
