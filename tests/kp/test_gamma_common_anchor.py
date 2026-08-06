from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from kp import blocks as blocks_mod
from kp.basis.selection import AutoGaugeConfig
from kp.blocks import blocks as blocks_impl
from kp.blocks.gamma_layout import GammaLayoutError, GammaRoutingError, GammaRowLayout
from kp.identity import hash_array
from kp.projection_selection import CandidateRejectionReason


def _layout(q_vectors: np.ndarray) -> GammaRowLayout:
    qset = np.asarray(q_vectors, dtype=np.float64)
    return GammaRowLayout.build(
        qsets=(qset, qset),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((2,), (2,)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(8, dtype=np.int64)),
    )


def _eigenframe(*, first_angle: float, second_angle: float) -> np.ndarray:
    """Return an 8D eigensystem with two layer-resolved low states."""

    first_cos, first_sin = np.cos(first_angle), np.sin(first_angle)
    second_cos, second_sin = np.cos(second_angle), np.sin(second_angle)
    frame = np.zeros((8, 8), dtype=np.complex128)
    frame[:, 0] = first_cos * np.eye(8)[:, 0] + first_sin * np.eye(8)[:, 4]
    frame[:, 1] = second_cos * np.eye(8)[:, 2] + second_sin * np.eye(8)[:, 6]
    frame[:, 2] = np.eye(8)[:, 1]
    frame[:, 3] = np.eye(8)[:, 3]
    frame[:, 4] = -first_sin * np.eye(8)[:, 0] + first_cos * np.eye(8)[:, 4]
    frame[:, 5] = np.eye(8)[:, 5]
    frame[:, 6] = -second_sin * np.eye(8)[:, 2] + second_cos * np.eye(8)[:, 6]
    frame[:, 7] = np.eye(8)[:, 7]
    np.testing.assert_allclose(frame.conj().T @ frame, np.eye(8), atol=1.0e-14)
    return frame


def _eigenvalues() -> np.ndarray:
    return np.asarray([-2.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def _build_anchor(
    *,
    layout: GammaRowLayout,
    eigenvectors_by_q: tuple[np.ndarray, ...],
    eigenvalues_by_q: tuple[np.ndarray, ...] | None = None,
    joint_band_indices: tuple[int, ...] = (0, 1),
    reference_q_index: int = 0,
    source_group_orbits: tuple[tuple[int, ...], ...] | None = None,
):
    values_by_q = (
        tuple(_eigenvalues() for _ in eigenvectors_by_q)
        if eigenvalues_by_q is None
        else eigenvalues_by_q
    )
    # Common-anchor ownership is derived from the selected reference orbitals.
    # Routed source-group ranks are deliberately not an input to this API.
    return blocks_mod.build_gamma_common_anchor_spec(
        reference_eigenvalues_by_q=values_by_q,
        reference_eigenvectors_by_q=eigenvectors_by_q,
        joint_band_indices=joint_band_indices,
        layout=layout,
        gauge_config=AutoGaugeConfig(reference_q_index=reference_q_index),
        source_group_orbits=source_group_orbits,
    )


def _build_frames(
    *,
    layout: GammaRowLayout,
    eigenvectors_by_q: tuple[np.ndarray, ...],
    anchor_spec,
    eigenvalues_by_q: tuple[np.ndarray, ...] | None = None,
):
    values_by_q = (
        tuple(_eigenvalues() for _ in eigenvectors_by_q)
        if eigenvalues_by_q is None
        else eigenvalues_by_q
    )
    return blocks_mod.build_gamma_common_anchor_frames(
        eigenvalues_by_q=values_by_q,
        eigenvectors_by_q=eigenvectors_by_q,
        layout=layout,
        anchor_spec=anchor_spec,
    )


def _materialize_references(anchor_spec) -> np.ndarray:
    references = np.zeros(
        (anchor_spec.same_q_dimension, len(anchor_spec.resolved_reference_terms)),
        dtype=np.complex128,
    )
    for column, terms in enumerate(anchor_spec.resolved_reference_terms):
        for row, coefficient in terms:
            references[int(row), column] += complex(coefficient)
        references[:, column] /= np.linalg.norm(references[:, column])
    return references


def _rotate_selected_columns(frame: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    rotated = np.array(frame, copy=True)
    rotated[:, :2] = frame[:, :2] @ np.asarray(unitary, dtype=np.complex128)
    return rotated


def test_common_anchor_public_apis_do_not_accept_routed_group_ranks() -> None:
    for api_name in (
        "build_gamma_common_anchor_spec",
        "build_gamma_common_anchor_frames",
    ):
        parameters = inspect.signature(getattr(blocks_mod, api_name)).parameters
        assert "group_ranks" not in parameters


def test_common_anchor_mixed_references_reject_separate_owner_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(np.asarray([[0.0, 0.0]]))
    eigenvectors = _eigenframe(first_angle=0.18, second_angle=0.31)
    real_resolve = blocks_impl._resolve_gamma_reference_core

    def ambiguous_owner(*args, **kwargs):
        resolution = real_resolve(*args, **kwargs)
        references = list(resolution.references_by_band)
        references[0] = ((0, 1.0 + 0.0j), (2, 1.0 + 0.0j))
        return replace(resolution, references_by_band=tuple(references))

    monkeypatch.setattr(
        blocks_impl,
        "_resolve_gamma_reference_core",
        ambiguous_owner,
    )

    with pytest.raises(GammaRoutingError, match="multiple physical layers") as error:
        _build_anchor(
            layout=layout,
            eigenvectors_by_q=(eigenvectors,),
            source_group_orbits=((0,), (1,)),
        )

    assert error.value.reason is CandidateRejectionReason.PROJECTOR_FRAME_RANK


def test_common_anchor_mixed_references_use_symmetry_owner_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed anchors may mix layers without defining their model owners."""

    layout = _layout(np.asarray([[0.0, 0.0]]))
    eigenvectors = _eigenframe(first_angle=0.18, second_angle=0.31)
    real_resolve = blocks_impl._resolve_gamma_reference_core

    def mixed_references(*args, **kwargs):
        resolution = real_resolve(*args, **kwargs)
        references = list(resolution.references_by_band)
        references[0] = ((0, 1.0 + 0.0j), (2, 1.0 + 0.0j))
        references[1] = ((4, 1.0 + 0.0j), (6, 1.0 + 0.0j))
        return replace(resolution, references_by_band=tuple(references))

    monkeypatch.setattr(
        blocks_impl,
        "_resolve_gamma_reference_core",
        mixed_references,
    )

    anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(eigenvectors,),
    )

    assert anchor.resolved_reference_terms == (
        ((0, 1.0 + 0.0j), (2, 1.0 + 0.0j)),
        ((4, 1.0 + 0.0j), (6, 1.0 + 0.0j)),
    )
    assert anchor.model_group_ranks == (1, 1)
    assert anchor.model_group_qset_indices == (0, 1)
    assert anchor.reference_source_groups == (0, 1)
    assert anchor.reference_physical_layers == (0, 1)


def test_common_anchor_quality_gate_is_a_typed_candidate_rejection() -> None:
    layout = _layout(np.asarray([[0.0, 0.0], [1.0, 0.0]]))
    reference = _eigenframe(first_angle=0.18, second_angle=0.31)
    unsupported = np.eye(8, dtype=np.complex128)[:, (1, 3, 0, 2, 4, 5, 6, 7)]
    anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(reference, unsupported),
    )

    with pytest.raises(GammaRoutingError, match="coverage failed") as error:
        _build_frames(
            layout=layout,
            eigenvectors_by_q=(reference, unsupported),
            anchor_spec=anchor,
        )

    assert error.value.reason is CandidateRejectionReason.PROJECTOR_FRAME_RANK


@pytest.mark.parametrize(
    "second_qset_rows",
    (
        ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
        ((0.0, 0.0), (1.25, 0.0), (0.0, 1.0)),
    ),
    ids=("order_mismatch", "coordinate_mismatch"),
)
def test_gamma_common_layout_requires_identical_qset_coordinates_and_order(
    second_qset_rows,
) -> None:
    qset = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    layout_kwargs = {
        "num_layer_list": (1, 1),
        "num_orb_per_layer_list": ((2,), (2,)),
        "spin_convention": "all",
        "source_basis_hash": hash_array(np.arange(24, dtype=np.int64)),
    }
    matched = GammaRowLayout.build(qsets=(qset, qset.copy()), **layout_kwargs)
    permutation = np.asarray((2, 0, 1), dtype=np.intp)
    jointly_permuted = GammaRowLayout.build(
        qsets=(qset[permutation], qset[permutation]),
        **layout_kwargs,
    )

    assert matched.q_count == jointly_permuted.q_count == 3
    with pytest.raises(GammaLayoutError):
        GammaRowLayout.build(
            qsets=(qset, np.asarray(second_qset_rows, dtype=np.float64)),
            **layout_kwargs,
        )


def test_common_anchor_selection_uses_only_the_representative_q(monkeypatch) -> None:
    layout = _layout(np.asarray([[0.0, 0.0], [1.0, 0.0]]))
    reference = _eigenframe(first_angle=0.18, second_angle=0.31)
    nonreference = _eigenframe(first_angle=0.63, second_angle=0.49)
    qrcp_inputs: list[np.ndarray] = []
    real_select_anchor_rows_qrcp = blocks_impl.select_anchor_rows_qrcp

    def record_qrcp_input(u_low, *args, **kwargs):
        qrcp_inputs.append(np.array(u_low, dtype=np.complex128, copy=True))
        return real_select_anchor_rows_qrcp(u_low, *args, **kwargs)

    monkeypatch.setattr(blocks_impl, "select_anchor_rows_qrcp", record_qrcp_input)
    _build_anchor(
        layout=layout,
        eigenvectors_by_q=(reference, nonreference),
    )

    assert len(qrcp_inputs) == 1
    np.testing.assert_allclose(
        qrcp_inputs[0],
        reference[:, :2],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_common_anchor_completes_spinful_chiral_reference_orbitals() -> None:
    """QRCP coordinate pivots must become the closed reference-orbital fibre."""

    layout = _layout(np.asarray([[0.0, 0.0]]))
    low_columns: list[np.ndarray] = []
    for first, second, phase in (
        (0, 1, 1.0j),
        (4, 5, -1.0j),
        (2, 3, 1.0j),
        (6, 7, -1.0j),
    ):
        column = np.zeros(8, dtype=np.complex128)
        column[first] = 1.0 / np.sqrt(2.0)
        column[second] = phase / np.sqrt(2.0)
        low_columns.append(column)
    low_frame = np.stack(low_columns, axis=1)
    complement = np.linalg.svd(low_frame, full_matrices=True)[0][:, 4:]
    eigenvectors = np.column_stack((low_frame, complement))
    eigenvalues = np.arange(8, dtype=np.float64)

    anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(eigenvectors,),
        eigenvalues_by_q=(eigenvalues,),
        joint_band_indices=(0, 1, 2, 3),
    )
    references = _materialize_references(anchor)

    # The source action is identical in every (layer, spin) two-orbital
    # segment.  A raw QRCP delta pivot is not closed under it, whereas the
    # automatically completed p_x +/- i p_y references are exact eigenvectors.
    local_action = np.zeros((8, 8), dtype=np.complex128)
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    for rows in ((0, 1), (2, 3), (4, 5), (6, 7)):
        local_action[np.ix_(rows, rows)] = rotation
    projected = references @ references.conj().T
    closure_residual = np.linalg.norm(
        (np.eye(8, dtype=np.complex128) - projected)
        @ local_action
        @ references,
        ord="fro",
    )

    assert closure_residual < 1.0e-12
    assert all(len(terms) == 2 for terms in anchor.resolved_reference_terms)


def test_common_anchor_reselection_is_invariant_to_representative_eigenspace_gauge() -> None:
    layout = _layout(np.asarray([[0.0, 0.0], [1.0, 0.0]]))
    frames_by_q = (
        _eigenframe(first_angle=0.18, second_angle=0.31),
        _eigenframe(first_angle=0.42, second_angle=0.27),
    )
    baseline_anchor = _build_anchor(layout=layout, eigenvectors_by_q=frames_by_q)
    baseline = _build_frames(
        eigenvectors_by_q=frames_by_q,
        layout=layout,
        anchor_spec=baseline_anchor,
    )
    representative_unitary = np.asarray(
        [
            [np.cos(0.37), -1.0j * np.sin(0.37)],
            [-1.0j * np.sin(0.37), np.cos(0.37)],
        ],
        dtype=np.complex128,
    )
    rotated_frames_by_q = (
        _rotate_selected_columns(frames_by_q[0], representative_unitary),
        frames_by_q[1],
    )
    rotated_anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=rotated_frames_by_q,
    )
    rotated = _build_frames(
        eigenvectors_by_q=rotated_frames_by_q,
        layout=layout,
        anchor_spec=rotated_anchor,
    )

    baseline_references = _materialize_references(baseline_anchor)
    rotated_references = _materialize_references(rotated_anchor)
    np.testing.assert_allclose(
        baseline_references @ baseline_references.conj().T,
        rotated_references @ rotated_references.conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )

    for q_index, (left, right) in enumerate(
        zip(baseline.local_frames_by_q, rotated.local_frames_by_q)
    ):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            left @ left.conj().T,
            frames_by_q[q_index][:, :2] @ frames_by_q[q_index][:, :2].conj().T,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_common_anchor_is_covariant_to_q_order_permutation() -> None:
    q_vectors = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    original_layout = _layout(q_vectors)
    permuted_layout = _layout(q_vectors[[1, 0]])
    original_eigenvectors = (
        _eigenframe(first_angle=0.18, second_angle=0.31),
        _eigenframe(first_angle=0.42, second_angle=0.27),
    )
    permuted_eigenvectors = original_eigenvectors[::-1]

    original_anchor = _build_anchor(
        layout=original_layout,
        eigenvectors_by_q=original_eigenvectors,
        reference_q_index=0,
    )
    permuted_anchor = _build_anchor(
        layout=permuted_layout,
        eigenvectors_by_q=permuted_eigenvectors,
        reference_q_index=1,
    )
    original = _build_frames(
        eigenvectors_by_q=original_eigenvectors,
        layout=original_layout,
        anchor_spec=original_anchor,
    )
    permuted = _build_frames(
        eigenvectors_by_q=permuted_eigenvectors,
        layout=permuted_layout,
        anchor_spec=permuted_anchor,
    )

    assert original_anchor.selected_rows == permuted_anchor.selected_rows
    assert original_anchor.resolved_reference_terms == permuted_anchor.resolved_reference_terms
    np.testing.assert_allclose(
        original.local_frames_by_q[0],
        permuted.local_frames_by_q[1],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        original.local_frames_by_q[1],
        permuted.local_frames_by_q[0],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_common_anchor_span_is_covariant_to_l2_orbital_permutation() -> None:
    layout = _layout(np.asarray([[0.0, 0.0]]))
    eigenvectors = _eigenframe(first_angle=0.18, second_angle=0.31)
    permutation = np.eye(8, dtype=np.complex128)
    permutation[:, [2, 3]] = permutation[:, [3, 2]]
    permutation[:, [6, 7]] = permutation[:, [7, 6]]
    swapped_eigenvectors = permutation @ eigenvectors

    original_anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(eigenvectors,),
    )
    swapped_anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(swapped_eigenvectors,),
    )
    original_references = _materialize_references(original_anchor)
    swapped_references = _materialize_references(swapped_anchor)
    np.testing.assert_allclose(
        swapped_references @ swapped_references.conj().T,
        permutation
        @ original_references
        @ original_references.conj().T
        @ permutation.conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )

    original = _build_frames(
        eigenvectors_by_q=(eigenvectors,),
        layout=layout,
        anchor_spec=original_anchor,
    )
    swapped = _build_frames(
        eigenvectors_by_q=(swapped_eigenvectors,),
        layout=layout,
        anchor_spec=swapped_anchor,
    )
    original_span = original.local_frames_by_q[0] @ original.local_frames_by_q[0].conj().T
    swapped_span = swapped.local_frames_by_q[0] @ swapped.local_frames_by_q[0].conj().T
    np.testing.assert_allclose(
        swapped_span,
        permutation @ original_span @ permutation.conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_bitei_shaped_mixed_owner_fibre_is_gauge_covariant() -> None:
    """Mixed anchors keep their owner fibres under U(2) and an L2 swap."""

    layout = _layout(np.asarray([[0.0, 0.0]]))
    low_frame = np.zeros((8, 2), dtype=np.complex128)
    low_frame[(4, 7), 0] = 1.0 / np.sqrt(2.0)
    low_frame[(0, 3), 1] = 1.0 / np.sqrt(2.0)
    complement = np.linalg.svd(low_frame, full_matrices=True)[0][:, 2:]
    eigenvectors = np.column_stack((low_frame, complement))

    baseline = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(eigenvectors,),
    )
    low_unitary = np.asarray(
        [
            [np.cos(0.37), -1.0j * np.sin(0.37)],
            [-1.0j * np.sin(0.37), np.cos(0.37)],
        ],
        dtype=np.complex128,
    )
    gauge_rotated = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(
            _rotate_selected_columns(eigenvectors, low_unitary),
        ),
    )
    column_swap = np.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    column_reordered = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(
            _rotate_selected_columns(eigenvectors, column_swap),
        ),
    )
    l2_swap = np.eye(8, dtype=np.complex128)
    l2_swap[:, [2, 3]] = l2_swap[:, [3, 2]]
    l2_swap[:, [6, 7]] = l2_swap[:, [7, 6]]
    orbital_swapped = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(l2_swap @ eigenvectors,),
    )
    baseline_frames = _build_frames(
        layout=layout,
        eigenvectors_by_q=(eigenvectors,),
        anchor_spec=baseline,
    )
    rotated_eigenvectors = _rotate_selected_columns(eigenvectors, low_unitary)
    gauge_rotated_frames = _build_frames(
        layout=layout,
        eigenvectors_by_q=(rotated_eigenvectors,),
        anchor_spec=gauge_rotated,
    )
    reordered_eigenvectors = _rotate_selected_columns(eigenvectors, column_swap)
    column_reordered_frames = _build_frames(
        layout=layout,
        eigenvectors_by_q=(reordered_eigenvectors,),
        anchor_spec=column_reordered,
    )
    orbital_swapped_frames = _build_frames(
        layout=layout,
        eigenvectors_by_q=(l2_swap @ eigenvectors,),
        anchor_spec=orbital_swapped,
    )

    def owner_projectors(anchor_spec) -> tuple[np.ndarray, ...]:
        references = _materialize_references(anchor_spec)
        return tuple(
            references[:, owner.reference_columns]
            @ references[:, owner.reference_columns].conj().T
            for owner in anchor_spec.owner_specs
        )

    baseline_references = _materialize_references(baseline)
    baseline_span = baseline_references @ baseline_references.conj().T
    rotated_references = _materialize_references(gauge_rotated)
    reordered_references = _materialize_references(column_reordered)
    swapped_references = _materialize_references(orbital_swapped)
    np.testing.assert_allclose(
        rotated_references @ rotated_references.conj().T,
        baseline_span,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        reordered_references @ reordered_references.conj().T,
        baseline_span,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        swapped_references @ swapped_references.conj().T,
        l2_swap @ baseline_span @ l2_swap.conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )
    baseline_frame = baseline_frames.local_frames_by_q[0]
    rotated_frame = gauge_rotated_frames.local_frames_by_q[0]
    reordered_frame = column_reordered_frames.local_frames_by_q[0]
    swapped_frame = orbital_swapped_frames.local_frames_by_q[0]
    np.testing.assert_allclose(
        rotated_frame @ rotated_frame.conj().T,
        baseline_frame @ baseline_frame.conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        reordered_frame @ reordered_frame.conj().T,
        baseline_frame @ baseline_frame.conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        swapped_frame @ swapped_frame.conj().T,
        l2_swap @ baseline_frame @ baseline_frame.conj().T @ l2_swap.conj().T,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert baseline.model_group_qset_indices == (0, 1)
    assert gauge_rotated.model_group_qset_indices == (0, 1)
    assert column_reordered.model_group_qset_indices == (0, 1)
    assert orbital_swapped.model_group_qset_indices == (0, 1)
    for baseline_owner, rotated_owner, reordered_owner, swapped_owner in zip(
        owner_projectors(baseline),
        owner_projectors(gauge_rotated),
        owner_projectors(column_reordered),
        owner_projectors(orbital_swapped),
    ):
        np.testing.assert_allclose(
            rotated_owner,
            baseline_owner,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            reordered_owner,
            baseline_owner,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            swapped_owner,
            l2_swap @ baseline_owner @ l2_swap.conj().T,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_common_anchor_derives_aab_model_owners_from_physical_layers() -> None:
    qset = np.asarray([[0.0, 0.0]], dtype=np.float64)
    layout = GammaRowLayout.build(
        qsets=(qset, qset),
        num_layer_list=(1, 2),
        num_orb_per_layer_list=((2,), (2, 2)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(12, dtype=np.int64)),
    )
    low_rows = (2, 8, 4, 10)
    remaining_rows = tuple(row for row in range(12) if row not in low_rows)
    eigenvectors = np.eye(12, dtype=np.complex128)[:, low_rows + remaining_rows]
    eigenvalues = np.arange(-4.0, 8.0, dtype=np.float64)

    # The fixture itself has no weight on physical layer 0.  Its four low
    # columns are two anchors on layer 1 and two anchors on layer 2, while both
    # model layers live in source Q-set 1.
    low_addresses = tuple(layout.rows_by_q[0][row] for row in low_rows)
    assert tuple(address.physical_layer for address in low_addresses) == (1, 1, 2, 2)
    assert tuple(address.source_group for address in low_addresses) == (1, 1, 1, 1)

    anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(eigenvectors,),
        eigenvalues_by_q=(eigenvalues,),
        joint_band_indices=(0, 1, 2, 3),
    )

    assert tuple(
        (
            owner.lambda_index,
            owner.rank,
            owner.source_group,
            owner.qset_index,
            tuple(owner.physical_layers),
            tuple(owner.reference_columns),
        )
        for owner in anchor.owner_specs
    ) == (
        (0, 2, 1, 1, (1,), (0, 1)),
        (1, 2, 1, 1, (2,), (2, 3)),
    )
    assert anchor.physical_layer_anchor_counts == (0, 2, 2)
    assert anchor.source_qset_anchor_counts == (0, 4)
    assert anchor.active_model_layers == (1, 2)
    assert anchor.model_group_ranks == (2, 2)
    assert anchor.model_group_qset_indices == (1, 1)
    assert anchor.continuum_sector_ranks == (2, 2)


def test_common_anchor_places_one_active_aab_owner_in_its_source_qset_slot() -> None:
    qset = np.asarray([[0.0, 0.0]], dtype=np.float64)
    layout = GammaRowLayout.build(
        qsets=(qset, qset),
        num_layer_list=(1, 2),
        num_orb_per_layer_list=((2,), (2, 2)),
        spin_convention="all",
        source_basis_hash=hash_array(np.arange(12, dtype=np.int64)),
    )
    low_rows = (2, 8)
    remaining_rows = tuple(row for row in range(12) if row not in low_rows)
    eigenvectors = np.eye(12, dtype=np.complex128)[:, low_rows + remaining_rows]
    eigenvalues = np.arange(-2.0, 10.0, dtype=np.float64)

    anchor = _build_anchor(
        layout=layout,
        eigenvectors_by_q=(eigenvectors,),
        eigenvalues_by_q=(eigenvalues,),
        joint_band_indices=(0, 1),
    )

    assert tuple(owner.to_payload() for owner in anchor.owner_specs) == (
        {
            "lambda_index": 0,
            "rank": 2,
            "source_group": 1,
            "qset_index": 1,
            "physical_layers": [1],
            "reference_columns": [0, 1],
        },
    )
    assert anchor.model_group_ranks == (2,)
    assert anchor.model_group_qset_indices == (1,)
    assert anchor.continuum_sector_ranks == (0, 2)
