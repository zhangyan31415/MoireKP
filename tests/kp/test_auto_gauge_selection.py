from __future__ import annotations

import numpy as np
import pytest

from kp.basis.selection import (
    build_reference_projectors_from_rows,
    compute_leverage_scores,
    polar_align_low_subspace,
    select_anchor_rows_qrcp,
)
from kp.blocks.blocks import (
    _assign_anchor_references_to_bands,
    _complete_gamma_spinful_reference_terms,
    _reference_overlap_singular_values,
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
