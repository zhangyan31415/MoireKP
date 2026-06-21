from __future__ import annotations

from contextlib import nullcontext
from typing import Any, List, Tuple, Literal

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize

from .downfold import (
    DownfoldingOptions,
    downfold_from_projector_groups,
    downfold_from_projectors,
    projector_groups_from_block_columns,
    set_projector_blas_threads as _set_downfold_projector_blas_threads,
)
from kp.basis.selection import AutoGaugeConfig, GaugeAnchorReport, select_anchor_rows_qrcp

PROJECTOR_BLAS_THREADS = 8


def _bounded_blas_threads():
    try:
        from threadpoolctl import threadpool_limits
    except Exception:  # pragma: no cover - optional runtime dependency
        return nullcontext()
    return threadpool_limits(limits=PROJECTOR_BLAS_THREADS, user_api="blas")


def set_projector_blas_threads(threads: int) -> None:
    global PROJECTOR_BLAS_THREADS
    PROJECTOR_BLAS_THREADS = max(1, int(threads))
    _set_downfold_projector_blas_threads(PROJECTOR_BLAS_THREADS)


def _extract_square_block(matrix: np.ndarray, index: np.ndarray) -> np.ndarray:
    idx = np.asarray(index, dtype=np.intp)
    if idx.ndim != 1:
        raise ValueError("block index must be one-dimensional")
    if idx.size == 0:
        return np.zeros((0, 0), dtype=np.asarray(matrix).dtype)
    start = int(idx[0])
    stop = start + int(idx.size)
    if np.array_equal(idx, np.arange(start, stop, dtype=np.intp)):
        return matrix[start:stop, start:stop]
    return matrix[np.ix_(idx, idx)]


def _complement_indices(size: int, selected: list[int] | np.ndarray) -> np.ndarray:
    selected_arr = np.asarray(selected, dtype=np.intp)
    if selected_arr.size == 0:
        return np.arange(size, dtype=np.intp)
    mask = np.ones(size, dtype=bool)
    mask[selected_arr] = False
    return np.nonzero(mask)[0]


def _hermitian_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with _bounded_blas_threads():
        return scipy.linalg.eigh(
            np.asarray(matrix, dtype=np.complex128),
            check_finite=False,
            overwrite_a=True,
            driver="evr",
        )


def _hermitian_eigh_columns(matrix: np.ndarray, columns: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    if columns is None:
        return _hermitian_eigh(matrix)
    if not columns:
        size = int(np.asarray(matrix).shape[0])
        return np.zeros(size, dtype=float), np.zeros((size, size), dtype=np.complex128)
    size = int(np.asarray(matrix).shape[0])
    unique_cols = sorted({int(col) for col in columns})
    if unique_cols[0] < 0 or unique_cols[-1] >= size:
        raise IndexError(f"eigenvector column request {unique_cols} outside block size {size}")
    lo, hi = unique_cols[0], unique_cols[-1]
    with _bounded_blas_threads():
        eig_window, vec_window = scipy.linalg.eigh(
            np.asarray(matrix, dtype=np.complex128),
            subset_by_index=(lo, hi),
            check_finite=False,
            overwrite_a=True,
            driver="evr",
        )
    eig = np.zeros(size, dtype=float)
    vec = np.zeros((size, size), dtype=np.complex128)
    if len(unique_cols) == hi - lo + 1:
        eig[lo : hi + 1] = eig_window
        vec[:, lo : hi + 1] = vec_window
    else:
        cols = np.asarray(unique_cols, dtype=np.intp)
        window_cols = cols - lo
        eig[cols] = eig_window[window_cols]
        vec[:, cols] = vec_window[:, window_cols]
    return eig, vec


def align_eigenstates(U_low: np.ndarray, Phi_ref: np.ndarray) -> np.ndarray:
    """Align low-energy eigenstates to a reference basis using Procrustes via SVD.

    Parameters
    - U_low: (M, N) eigenvector matrix (columns are states to align)
    - Phi_ref: (M, N) reference vectors (columns)

    Returns
    - U_aligned: (M, N) aligned eigenvectors
    - V: (N, N) unitary rotation applied in the low-energy subspace
    """

    # Overlap O = Phi_ref^† U_low
    O = Phi_ref.conj().T @ U_low
    # SVD: O = X Σ Y^†
    X, s, Yh = np.linalg.svd(O, full_matrices=False)
    # Optimal unitary: V = Y X^†
    V = Yh.conj().T @ X.conj().T
    # Apply rotation in subspace
    U_aligned = U_low @ V
    return U_aligned, V


def _is_reference_pair(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and not isinstance(value[0], (list, tuple))
    )


def _layer_reference_entries(layer_refs: Any, n_bands: int) -> list[Any]:
    if n_bands == 1 and _is_reference_pair(layer_refs):
        return [layer_refs]
    if isinstance(layer_refs, (list, tuple)):
        return list(layer_refs)
    return [layer_refs]


def _parse_reference_terms(reference: Any, *, context: str) -> list[tuple[int, complex]]:
    raw_items = [reference] if _is_reference_pair(reference) else reference
    if not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]

    terms: list[tuple[int, complex]] = []
    for item in raw_items:
        if _is_reference_pair(item):
            idxc, coef = item
            terms.append((int(idxc), complex(coef)))
        else:
            terms.append((int(item), complex(1.0)))
    if not terms:
        raise ValueError(f"{context}: empty reference in norb_fix_list")
    return terms


def _resolve_reference_index(
    idx: int,
    *,
    block_dim: int,
    context: str,
    allow_layer_global: bool = False,
    layer: int | None = None,
    layer_block_dim: int | None = None,
    total_layers: int | None = None,
) -> int:
    if 0 <= idx < block_dim:
        return idx

    if allow_layer_global:
        if layer is None or layer_block_dim is None or total_layers is None:
            raise ValueError(f"{context}: layer-global reference resolution is missing layer metadata")
        total_dim = int(layer_block_dim) * int(total_layers)
        if 0 <= idx < total_dim:
            owner_layer = int(idx) // int(layer_block_dim)
            if owner_layer != int(layer):
                raise ValueError(
                    f"{context}: reference index {idx} belongs to layer {owner_layer}, "
                    f"not layer {layer}"
                )
            local_idx = int(idx) - owner_layer * int(layer_block_dim)
            if 0 <= local_idx < block_dim:
                return local_idx
        raise ValueError(
            f"{context}: reference index {idx} is outside local block dimension {block_dim} "
            f"and combined same-Q dimension {total_dim}"
        )

    raise ValueError(f"{context}: reference index {idx} is outside block dimension {block_dim}")


def _reference_terms_for_layer(
    *,
    nlow_state_list: Any,
    norb_fix_list: Any,
    layer: int,
    block_dim: int,
    context: str,
    allow_layer_global: bool = False,
    layer_for_global: int | None = None,
    layer_block_dim: int | None = None,
    total_layers: int | None = None,
) -> tuple[list[int], list[list[tuple[int, complex]]]]:
    if layer >= len(nlow_state_list):
        raise IndexError(f"{context}: missing nlow_state_list entry for layer {layer}")

    bands = [int(band) for band in nlow_state_list[layer]]
    if not bands:
        return [], []
    if layer >= len(norb_fix_list):
        raise IndexError(f"{context}: missing norb_fix_list entry for layer {layer}")

    ref_entries = _layer_reference_entries(norb_fix_list[layer], len(bands))
    if len(ref_entries) != len(bands):
        raise ValueError(
            f"{context}: nlow_state_list layer {layer} has {len(bands)} bands but "
            f"norb_fix_list layer {layer} has {len(ref_entries)} references"
        )

    resolved: list[list[tuple[int, complex]]] = []
    for ref_idx, reference in enumerate(ref_entries):
        ref_context = f"{context} reference {ref_idx}"
        terms = []
        for raw_idx, coef in _parse_reference_terms(reference, context=ref_context):
            terms.append(
                (
                    _resolve_reference_index(
                        int(raw_idx),
                        block_dim=block_dim,
                        context=ref_context,
                        allow_layer_global=allow_layer_global,
                        layer=layer_for_global,
                        layer_block_dim=layer_block_dim,
                        total_layers=total_layers,
                    ),
                    coef,
                )
            )
        resolved.append(terms)
    return bands, resolved


def _align_selected_eigenstates(
    vec: np.ndarray,
    bands: list[int],
    references: list[list[tuple[int, complex]]],
    *,
    context: str,
) -> None:
    if len(bands) != len(references):
        raise ValueError(f"{context}: band/reference length mismatch")
    if not bands:
        return

    for band in bands:
        if band < -vec.shape[1] or band >= vec.shape[1]:
            raise IndexError(f"{context}: low-state band index {band} outside block dimension {vec.shape[1]}")

    phi_ref = np.zeros((vec.shape[0], len(bands)), dtype=complex)
    for col_idx, terms in enumerate(references):
        col = np.zeros(phi_ref.shape[0], dtype=complex)
        for idxc, coef in terms:
            col[idxc] += coef
        norm = np.linalg.norm(col)
        if norm <= 0.0:
            raise ValueError(f"{context}: reference {col_idx} has zero norm after resolving norb_fix_list")
        phi_ref[:, col_idx] = col / norm

    u_low = vec[:, np.array(bands, dtype=int)]
    u_aligned, _ = align_eigenstates(u_low, phi_ref)
    vec[:, bands] = u_aligned


def _is_auto_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"auto", "auto_scdm"}


def _gauge_requests_auto(gauge_config: Any) -> bool:
    if _is_auto_token(gauge_config):
        return True
    if isinstance(gauge_config, dict):
        method = gauge_config.get("method", gauge_config.get("mode"))
        return _is_auto_token(method)
    return False


def _auto_gauge_config(gauge_config: Any) -> AutoGaugeConfig:
    if gauge_config is None or _is_auto_token(gauge_config):
        return AutoGaugeConfig()
    if not isinstance(gauge_config, dict):
        raise ValueError(f"project.gauge must be 'auto' or a mapping, got {gauge_config!r}")
    method = str(gauge_config.get("method", "auto_scdm")).lower()
    if method != "auto_scdm":
        raise ValueError(f"Unsupported project.gauge.method={method!r}; expected 'auto_scdm'")
    anchor_scope = str(gauge_config.get("anchor_scope", "per_sector")).lower()
    if anchor_scope != "per_sector":
        raise ValueError("Phase 1 auto gauge only supports project.gauge.anchor_scope='per_sector'")
    candidate_pool = str(gauge_config.get("candidate_pool", "all")).lower()
    if candidate_pool != "all":
        raise ValueError("Phase 1 auto gauge only supports project.gauge.candidate_pool='all'")
    projected_anchors = bool(gauge_config.get("projected_anchors", False))
    if projected_anchors:
        raise ValueError("project.gauge.projected_anchors=true is not supported in Phase 1")
    return AutoGaugeConfig(
        method=method,
        anchor_scope=anchor_scope,
        candidate_pool=candidate_pool,
        projected_anchors=projected_anchors,
        min_sigma=float(gauge_config.get("min_sigma", 1.0e-6)),
        max_condition=float(gauge_config.get("max_condition", 1.0e6)),
        reference_q_index=int(gauge_config.get("reference_q_index", gauge_config.get("ref_q_index", 0))),
        basis_is_orthonormal=bool(gauge_config.get("basis_is_orthonormal", True)),
    )


def _manual_gauge_report(norb_fix_list: Any) -> GaugeAnchorReport:
    return GaugeAnchorReport(
        gauge_mode="manual_norb_fix_list",
        resolved_norb_fix_list=norb_fix_list,
        selections=[],
        metric={"type": "orthonormal", "basis_is_orthonormal": True},
        state_selection_quality={
            "status": "not_evaluated",
            "reason": "manual gauge anchors were provided",
        },
        gauge_anchor_quality={
            "status": "manual",
            "sigma_min": None,
            "condition_number": None,
        },
        symmetry_closure_quality={
            "status": "not_available",
            "subspace_leakage": None,
            "reason": "symmetry validation is not available during gauge resolution",
        },
    )


def _auto_reference_terms(row: int) -> list[tuple[int, complex]]:
    return [(int(row), 1.0 + 0.0j)]


def _format_auto_reference_terms(terms: list[tuple[int, complex]]) -> list[list[Any]]:
    out: list[list[Any]] = []
    for row, coef in terms:
        value = complex(coef)
        if abs(value.imag) < 1.0e-14:
            coef_out: Any = float(value.real)
        elif abs(value.real) < 1.0e-14 and abs(value.imag - 1.0) < 1.0e-14:
            coef_out = "1j"
        elif abs(value.real) < 1.0e-14 and abs(value.imag + 1.0) < 1.0e-14:
            coef_out = "-1j"
        else:
            coef_out = {"real": float(value.real), "imag": float(value.imag)}
        out.append([int(row), coef_out])
    return out


def _format_complex_for_report(value: complex) -> Any:
    return _format_auto_reference_terms([(0, complex(value))])[0][1]


def _reference_overlap_scores(u_low: np.ndarray, references: list[list[tuple[int, complex]]]) -> np.ndarray:
    u = np.asarray(u_low, dtype=np.complex128)
    scores = np.zeros((len(references), u.shape[1]), dtype=float)
    for ref_pos, terms in enumerate(references):
        overlap = np.zeros(u.shape[1], dtype=np.complex128)
        norm_sq = 0.0
        for row, coef in terms:
            overlap += np.conjugate(complex(coef)) * u[int(row), :]
            norm_sq += abs(complex(coef)) ** 2
        if norm_sq <= 0.0:
            raise ValueError("auto gauge produced a zero-norm reference")
        scores[ref_pos, :] = np.abs(overlap) / np.sqrt(norm_sq)
    return scores


def _reference_overlap_singular_values(u_low: np.ndarray, references_by_band: list[list[tuple[int, complex]]]) -> np.ndarray:
    u = np.asarray(u_low, dtype=np.complex128)
    overlap = np.zeros((len(references_by_band), u.shape[1]), dtype=np.complex128)
    for ref_pos, terms in enumerate(references_by_band):
        norm_sq = 0.0
        for row, coef in terms:
            overlap[ref_pos, :] += np.conjugate(complex(coef)) * u[int(row), :]
            norm_sq += abs(complex(coef)) ** 2
        if norm_sq <= 0.0:
            raise ValueError("auto gauge produced a zero-norm reference")
        overlap[ref_pos, :] /= np.sqrt(norm_sq)
    return np.linalg.svd(overlap, compute_uv=False)


def _same_segment(row: int, partner: int, segments: list[tuple[int, int]]) -> bool:
    for start, stop in segments:
        if start <= int(row) < stop:
            return start <= int(partner) < stop
    return False


def _segment_index_for_row(row: int, segments: list[tuple[int, int]]) -> int | None:
    for index, (start, stop) in enumerate(segments):
        if int(start) <= int(row) < int(stop):
            return int(index)
    return None


def _gamma_reference_sort_key(
    terms: list[tuple[int, complex]],
    *,
    segments: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Order Gamma references in the continuum model frame.

    Gamma same-Q blocks are laid out as spin blocks, with physical layers inside
    each spin block.  The model frame expects physical layer sectors first, and
    spin partners inside each sector.  QRCP row order is not stable enough for
    that semantic ordering, especially in nearly degenerate Gamma subspaces.
    """

    if not terms:
        return (0, 0, 0, 0)
    total_segments = max(len(segments), 1)
    spin_count = 2 if total_segments % 2 == 0 else 1
    layer_count = max(total_segments // spin_count, 1)
    layer_indices: list[int] = []
    spin_indices: list[int] = []
    rows: list[int] = []
    for row, _coef in terms:
        row_int = int(row)
        rows.append(row_int)
        segment_index = _segment_index_for_row(row_int, segments)
        if segment_index is None:
            continue
        spin_indices.append(int(segment_index // layer_count))
        layer_indices.append(int(segment_index % layer_count))
    layer_key = -max(layer_indices) if layer_indices else 0
    spin_key = -max(spin_indices) if spin_indices else 0
    span_key = len(set(layer_indices)) if layer_indices else 0
    row_key = min(rows)
    return (layer_key, spin_key, span_key, row_key)


def _same_spin_layer_exchange_candidates(
    row: int,
    *,
    segments: list[tuple[int, int]],
) -> list[int]:
    segment_index = _segment_index_for_row(row, segments)
    if segment_index is None or len(segments) < 2 or len(segments) % 2 != 0:
        return []
    layer_count = len(segments) // 2
    spin_index = int(segment_index) // layer_count
    layer_index = int(segment_index) % layer_count
    candidates: list[int] = []
    for other_layer in range(layer_count):
        if other_layer == layer_index:
            continue
        other_segment = spin_index * layer_count + other_layer
        start, stop = segments[other_segment]
        candidates.extend(range(int(start), int(stop)))
    return candidates


def _order_gamma_references_for_model_basis(
    references: list[list[tuple[int, complex]]],
    *,
    segments: list[tuple[int, int]],
) -> list[list[tuple[int, complex]]]:
    return [
        list(ref)
        for ref in sorted(
            references,
            key=lambda ref: _gamma_reference_sort_key(ref, segments=segments),
        )
    ]


def _gamma_same_q_row_segments(
    block_dim: int,
    num_layer_list: list[int],
    num_orb_per_layer_list: list[list[int]],
    *,
    spin: Literal["up", "down", "all"],
) -> list[tuple[int, int]]:
    widths = [int(width) for group in num_orb_per_layer_list for width in group]
    total_layers = int(sum(int(n) for n in num_layer_list))
    if len(widths) != total_layers:
        if total_layers <= 0:
            return [(0, int(block_dim))]
        spin_count = 2 if spin == "all" else 1
        if int(block_dim) % (spin_count * total_layers) != 0:
            return [(0, int(block_dim))]
        widths = [int(block_dim) // (spin_count * total_layers)] * total_layers
    per_spin_dim = int(sum(widths))
    spin_count = 2 if spin == "all" else 1
    if int(block_dim) != spin_count * per_spin_dim:
        return [(0, int(block_dim))]
    segments: list[tuple[int, int]] = []
    for spin_index in range(spin_count):
        offset = spin_index * per_spin_dim
        cursor = offset
        for width in widths:
            segments.append((cursor, cursor + int(width)))
            cursor += int(width)
    return segments


def _complete_gamma_spinful_reference_terms(
    u_low: np.ndarray,
    selected_rows: list[int],
    *,
    segments: list[tuple[int, int]],
    leverage_rtol: float = 1.0e-2,
    magnitude_rtol: float = 5.0e-2,
) -> tuple[list[list[tuple[int, complex]]], list[dict[str, Any]], list[str]]:
    u = np.asarray(u_low, dtype=np.complex128)
    leverage = np.real(np.sum(np.abs(u) ** 2, axis=1))
    selected = {int(row) for row in selected_rows}
    references: list[list[tuple[int, complex]]] = []
    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in selected_rows:
        row = int(row)
        row_mag = np.abs(u[row, :])
        row_norm = max(float(np.linalg.norm(row_mag)), 1.0e-15)
        candidates: list[tuple[float, int, complex, float, float]] = []
        adjacent_rows = [row - 1, row + 1]
        segment_index = _segment_index_for_row(row, segments)
        if segment_index is None:
            partner_groups = [adjacent_rows]
        else:
            start, stop = segments[segment_index]
            same_segment_rows = [
                candidate
                for candidate in range(int(start), int(stop))
                if candidate not in adjacent_rows and candidate != row
            ]
            partner_groups = [adjacent_rows + same_segment_rows, _same_spin_layer_exchange_candidates(row, segments=segments)]
        partner_scope = "same_segment"
        for scope_index, partner_rows in enumerate(partner_groups):
            scope_candidates: list[tuple[float, int, complex, float, float]] = []
            for partner in partner_rows:
                if partner < 0 or partner >= u.shape[0] or partner in selected:
                    continue
                if scope_index == 0 and not _same_segment(row, partner, segments):
                    continue
                lev_ref = max(float(abs(leverage[row])), float(abs(leverage[partner])), 1.0e-15)
                leverage_rel = float(abs(leverage[row] - leverage[partner]) / lev_ref)
                if leverage_rel > float(leverage_rtol):
                    continue
                partner_mag = np.abs(u[partner, :])
                magnitude_rel = float(np.linalg.norm(row_mag - partner_mag) / row_norm)
                if magnitude_rel > float(magnitude_rtol):
                    continue
                for phase in (1.0 + 0.0j, -1.0 + 0.0j, 1.0j, -1.0j):
                    overlap = (u[row, :] + np.conjugate(phase) * u[partner, :]) / np.sqrt(2.0)
                    score = float(np.linalg.norm(overlap))
                    scope_candidates.append((score, int(partner), complex(phase), leverage_rel, magnitude_rel))
            if scope_candidates:
                candidates = scope_candidates
                partner_scope = "same_segment" if scope_index == 0 else "same_spin_layer_exchange"
                break
        if not candidates:
            references.append(_auto_reference_terms(row))
            warnings.append(f"auto gauge gamma spinful row {row} did not find a safe adjacent chiral partner")
            details.append({"row": int(row), "partner_row": None, "partner_phase": None, "pair_score": None})
            continue
        score, partner, phase, leverage_rel, magnitude_rel = sorted(
            candidates,
            key=lambda item: (
                -float(item[0]),
                0 if abs(int(item[1]) - row) == 1 else 1,
                abs(int(item[1]) - row),
                int(item[1]),
                float(np.real(item[2])),
                float(np.imag(item[2])),
            ),
        )[0]
        references.append([(int(row), 1.0 + 0.0j), (int(partner), phase)])
        details.append(
            {
                "row": int(row),
                "partner_row": int(partner),
                "partner_phase": _format_complex_for_report(phase),
                "pair_score": float(score),
                "partner_scope": partner_scope,
                "leverage_relative_difference": float(leverage_rel),
                "row_magnitude_relative_difference": float(magnitude_rel),
            }
        )
    return references, details, warnings


def _assign_anchor_references_to_bands(
    u_low: np.ndarray,
    references: list[list[tuple[int, complex]]],
) -> tuple[list[list[tuple[int, complex]]], list[float]]:
    if not references:
        return [], []
    u = np.asarray(u_low, dtype=np.complex128)
    if len(references) != u.shape[1]:
        raise ValueError(
            f"anchor assignment requires one reference per low-state column, "
            f"got {len(references)} references for {u.shape[1]} columns"
        )
    scores = _reference_overlap_scores(u, references)
    tie_break = np.asarray([min(int(row) for row, _coef in ref) for ref in references], dtype=float)[:, np.newaxis]
    scale = max(float(np.max(np.abs(scores))), 1.0)
    cost = -scores + 1.0e-14 * scale * tie_break / max(float(u.shape[0]), 1.0)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    assigned: list[list[tuple[int, complex]] | None] = [None] * u.shape[1]
    assigned_scores: list[float | None] = [None] * u.shape[1]
    for ref_pos, band_pos in zip(row_ind, col_ind):
        assigned[int(band_pos)] = references[int(ref_pos)]
        assigned_scores[int(band_pos)] = float(scores[int(ref_pos), int(band_pos)])
    if any(ref is None for ref in assigned):
        raise ValueError("anchor assignment failed to cover all low-state columns")
    return (
        [list(ref) for ref in assigned if ref is not None],
        [float(score) for score in assigned_scores if score is not None],
    )


def _selection_dict(
    *,
    scope: str,
    bands: list[int],
    selection,
    layer: int | None = None,
    q_index: int | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "layer": None if layer is None else int(layer),
        "q_index": None if q_index is None else int(q_index),
        "bands": [int(band) for band in bands],
        "selected_rows": [int(row) for row in selection.selected_rows],
        "method": selection.method,
        "rank": int(selection.rank),
        "singular_values": [float(value) for value in selection.singular_values],
        "sigma_min": float(selection.sigma_min),
        "condition_number": float(selection.condition_number),
        "selected_leverage_scores": [float(value) for value in selection.leverage_scores],
        "top_leverage_rows": [
            {
                "row": int(candidate.row),
                "leverage": float(candidate.leverage),
                "selected": bool(candidate.selected),
                "label": candidate.label,
            }
            for candidate in selection.candidates
        ],
        "warnings": list(selection.warnings),
    }


def resolve_project_gauge_anchors(
    hamk_reference: np.ndarray,
    q_count: int,
    orb_per_layer0: int,
    num_layer_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    Qlayer_list: List[List[np.ndarray]] | None = None,
    num_orb_per_layer_list: List[List[int]] | None = None,
    nlow_state_list: List[List[int]] | None = None,
    norb_fix_list: Any = None,
    gauge_config: Any = None,
    mode: str = "gamma",
) -> tuple[list[Any], GaugeAnchorReport]:
    """Resolve manual or automatic gauge anchors to legacy norb_fix_list format."""

    if nlow_state_list is None:
        raise ValueError("project.nlow_state_list must be provided before resolving gauge anchors")
    total_layers = int(sum(int(n) for n in num_layer_list))
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"project.nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}"
        )

    auto_from_norb = _is_auto_token(norb_fix_list)
    auto_from_gauge = _gauge_requests_auto(gauge_config)
    has_manual = norb_fix_list is not None and not auto_from_norb
    if has_manual and auto_from_gauge:
        raise ValueError("manual norb_fix_list cannot be combined with project.gauge auto")
    if has_manual:
        if len(norb_fix_list) != total_layers:
            raise ValueError(
                f"project.norb_fix_list must have {total_layers} physical-layer rows "
                f"(sum(num_layer_list)); got {len(norb_fix_list)}"
            )
        return norb_fix_list, _manual_gauge_report(norb_fix_list)
    if not (auto_from_norb or auto_from_gauge):
        raise ValueError(
            "project.norb_fix_list is missing; write project.gauge: auto or provide manual norb_fix_list anchors"
        )

    config = _auto_gauge_config(gauge_config)
    mode_lower = str(mode).lower()
    ref_q = int(config.reference_q_index)
    if ref_q < 0 or ref_q >= int(q_count):
        raise IndexError(f"project.gauge.reference_q_index={ref_q} outside available Q range 0..{int(q_count) - 1}")
    if Qlayer_list is None:
        Qlayer_list = [[np.arange(q_count) for _ in range(n)] for n in num_layer_list]
    if num_orb_per_layer_list is None:
        num_orb_per_layer_list = [[int(orb_per_layer0) for _ in range(n)] for n in num_layer_list]

    _, h_vec_blk, _, _ = get_H_block(
        np.asarray(hamk_reference, dtype=np.complex128),
        Qlayer_list,
        num_layer_list,
        num_orb_per_layer_list,
        nlow_state_list,
        [],
        spin=spin,
        mode=mode_lower,
        selected_bands_by_layer=nlow_state_list,
    )
    resolved: list[Any] = [[] for _ in range(total_layers)]
    selections: list[dict[str, Any]] = []
    warnings: list[str] = []

    if mode_lower == "gamma":
        vec = np.asarray(h_vec_blk[ref_q], dtype=np.complex128)
        bands_flat: list[int] = []
        owners: list[tuple[int, int]] = []
        for layer, bands in enumerate(nlow_state_list):
            for band_pos, band in enumerate(bands):
                bands_flat.append(int(band))
                owners.append((int(layer), int(band_pos)))
        if bands_flat:
            if len(set(bands_flat)) != len(bands_flat):
                raise ValueError("project.gauge auto cannot resolve duplicate gamma low-state band indices")
            u_low = vec[:, np.asarray(bands_flat, dtype=np.intp)]
            selection = select_anchor_rows_qrcp(
                u_low,
                n_anchors=len(bands_flat),
                basis_is_orthonormal=config.basis_is_orthonormal,
            )
            if selection.sigma_min < config.min_sigma:
                raise ValueError(
                    f"auto gauge sigma_min={selection.sigma_min:.3e} below min_sigma={config.min_sigma:.3e}"
                )
            if selection.condition_number > config.max_condition:
                raise ValueError(
                    f"auto gauge condition_number={selection.condition_number:.3e} exceeds "
                    f"max_condition={config.max_condition:.3e}"
                )
            references = [_auto_reference_terms(int(row)) for row in selection.selected_rows]
            completion_details: list[dict[str, Any]] = []
            if spin == "all":
                segments = _gamma_same_q_row_segments(
                    u_low.shape[0],
                    num_layer_list,
                    num_orb_per_layer_list,
                    spin=spin,
                )
                references, completion_details, completion_warnings = _complete_gamma_spinful_reference_terms(
                    u_low,
                    selection.selected_rows,
                    segments=segments,
                )
                warnings.extend(completion_warnings)
                references_by_band = _order_gamma_references_for_model_basis(references, segments=segments)
                scores = _reference_overlap_scores(u_low, references_by_band)
                assigned_scores = [float(np.max(scores[index, :])) for index in range(scores.shape[0])]
            else:
                references_by_band, assigned_scores = _assign_anchor_references_to_bands(u_low, references)
            reference_singular_values = _reference_overlap_singular_values(u_low, references_by_band)
            reference_sigma_min = float(np.min(reference_singular_values)) if reference_singular_values.size else 0.0
            reference_sigma_max = float(np.max(reference_singular_values)) if reference_singular_values.size else 0.0
            reference_condition = (
                float("inf") if reference_sigma_min <= 0.0 else float(reference_sigma_max / reference_sigma_min)
            )
            if reference_sigma_min < config.min_sigma:
                raise ValueError(
                    f"auto gauge reference sigma_min={reference_sigma_min:.3e} "
                    f"below min_sigma={config.min_sigma:.3e}"
                )
            if reference_condition > config.max_condition:
                raise ValueError(
                    f"auto gauge reference condition_number={reference_condition:.3e} exceeds "
                    f"max_condition={config.max_condition:.3e}"
                )
            for (layer, _band_pos), terms in zip(owners, references_by_band):
                resolved[layer].append(_format_auto_reference_terms(terms))
            selections.append(
                _selection_dict(
                    scope="gamma_same_q",
                    bands=bands_flat,
                    selection=selection,
                    q_index=ref_q,
                )
            )
            selections[-1]["resolved_references_by_band"] = [
                _format_auto_reference_terms(terms) for terms in references_by_band
            ]
            selections[-1]["assigned_reference_scores"] = [float(score) for score in assigned_scores]
            selections[-1]["reference_ordering"] = "gamma_model_frame" if spin == "all" else "overlap_assignment"
            selections[-1]["reference_score_mode"] = "row_max_overlap" if spin == "all" else "assigned_overlap"
            selections[-1]["reference_singular_values"] = [float(value) for value in reference_singular_values.tolist()]
            selections[-1]["reference_sigma_min"] = float(reference_sigma_min)
            selections[-1]["reference_condition_number"] = float(reference_condition)
            selections[-1]["anchor_completion"] = completion_details
            warnings.extend(selection.warnings)
    else:
        for layer, bands in enumerate(nlow_state_list):
            layer_bands = [int(band) for band in bands]
            if not layer_bands:
                continue
            block_index = int(layer) * int(q_count) + ref_q
            vec = np.asarray(h_vec_blk[block_index], dtype=np.complex128)
            u_low = vec[:, np.asarray(layer_bands, dtype=np.intp)]
            selection = select_anchor_rows_qrcp(
                u_low,
                n_anchors=len(layer_bands),
                basis_is_orthonormal=config.basis_is_orthonormal,
            )
            if selection.sigma_min < config.min_sigma:
                raise ValueError(
                    f"auto gauge layer {layer} sigma_min={selection.sigma_min:.3e} "
                    f"below min_sigma={config.min_sigma:.3e}"
                )
            if selection.condition_number > config.max_condition:
                raise ValueError(
                    f"auto gauge layer {layer} condition_number={selection.condition_number:.3e} "
                    f"exceeds max_condition={config.max_condition:.3e}"
                )
            references = [_auto_reference_terms(int(row)) for row in selection.selected_rows]
            references_by_band, assigned_scores = _assign_anchor_references_to_bands(u_low, references)
            reference_singular_values = _reference_overlap_singular_values(u_low, references_by_band)
            reference_sigma_min = float(np.min(reference_singular_values)) if reference_singular_values.size else 0.0
            reference_sigma_max = float(np.max(reference_singular_values)) if reference_singular_values.size else 0.0
            reference_condition = (
                float("inf") if reference_sigma_min <= 0.0 else float(reference_sigma_max / reference_sigma_min)
            )
            if reference_sigma_min < config.min_sigma:
                raise ValueError(
                    f"auto gauge layer {layer} reference sigma_min={reference_sigma_min:.3e} "
                    f"below min_sigma={config.min_sigma:.3e}"
                )
            if reference_condition > config.max_condition:
                raise ValueError(
                    f"auto gauge layer {layer} reference condition_number={reference_condition:.3e} "
                    f"exceeds max_condition={config.max_condition:.3e}"
                )
            resolved[layer] = [_format_auto_reference_terms(terms) for terms in references_by_band]
            selections.append(
                _selection_dict(
                    scope="physical_layer",
                    layer=layer,
                    bands=layer_bands,
                    selection=selection,
                    q_index=ref_q,
                )
            )
            selections[-1]["resolved_references_by_band"] = [
                _format_auto_reference_terms(terms) for terms in references_by_band
            ]
            selections[-1]["assigned_reference_scores"] = [float(score) for score in assigned_scores]
            selections[-1]["reference_singular_values"] = [float(value) for value in reference_singular_values.tolist()]
            selections[-1]["reference_sigma_min"] = float(reference_sigma_min)
            selections[-1]["reference_condition_number"] = float(reference_condition)
            warnings.extend(selection.warnings)

    sigma_values = [
        float(row.get("reference_sigma_min", row["sigma_min"]))
        for row in selections
        if row.get("reference_sigma_min", row.get("sigma_min")) is not None
    ]
    cond_values = [
        float(row.get("reference_condition_number", row["condition_number"]))
        for row in selections
        if row.get("reference_condition_number", row.get("condition_number")) is not None
    ]
    sigma_min = min(sigma_values) if sigma_values else None
    condition_number = max(cond_values) if cond_values else None
    report = GaugeAnchorReport(
        gauge_mode="auto_scdm",
        resolved_norb_fix_list=resolved,
        selections=selections,
        metric={
            "type": "orthonormal" if config.basis_is_orthonormal else "overlap_metric",
            "basis_is_orthonormal": bool(config.basis_is_orthonormal),
            "formula": "diag(U U^dagger)" if config.basis_is_orthonormal else "diag(S^1/2 U U^dagger S^1/2)",
        },
        state_selection_quality={
            "status": "not_evaluated",
            "reason": "auto gauge fixes anchors for the configured nlow_state_list only",
        },
        gauge_anchor_quality={
            "status": "ok",
            "sigma_min": sigma_min,
            "condition_number": condition_number,
            "min_sigma": float(config.min_sigma),
            "max_condition": float(config.max_condition),
        },
        symmetry_closure_quality={
            "status": "not_available",
            "subspace_leakage": None,
            "reason": "symmetry validation is not available during gauge resolution",
        },
        warnings=warnings,
    )
    return resolved, report


def _physical_layer_entry_index(nlow_state_list: Any, num_layer_arr: np.ndarray, group_index: int, layer_in_group: int) -> int:
    total_layers = int(np.sum(num_layer_arr))
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}. "
            "Use [] for layers that do not contribute."
        )
    return int(np.sum(num_layer_arr[:group_index]) + layer_in_group)


def _source_group_band_lists(nlow_state_list: Any, num_layer_list: List[int]) -> list[list[int]]:
    rows = [[int(band) for band in row] for row in nlow_state_list]
    total_layers = int(sum(num_layer_list))
    if len(rows) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(rows)}. "
            "Use [] for layers that do not contribute."
        )
    grouped: list[list[int]] = []
    offset = 0
    for n_layers in num_layer_list:
        bands: list[int] = []
        for local in range(int(n_layers)):
            bands.extend(rows[offset + local])
        grouped.append(bands)
        offset += int(n_layers)
    return grouped


def get_H_block(
    Hamk_list: np.ndarray,
    Qlayer_list: List[np.ndarray],
    num_layer_list: List[int],
    num_orb_per_layer_list: List[List[int]],
    nlow_state_list: List[int],
    norb_fix_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    mode: Literal["gamma", "K1", "K2"] = "gamma",
    selected_bands_by_layer: list[list[int]] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble Hamiltonian sub-blocks and diagonalize per-Q selection.

    This ports the core indexing logic from the notebook's get_H_block
    to extract same-Q blocks across layers, handle spins, and compute
    eigenvalues/eigenvectors per block.

    Returns
    - H_GM_diag_eig: array of eigenvalues per block (list -> array)
    - H_GM_diag_eig_vec: array of eigenvectors per block (list -> array)
    - H_diag_block: list/array of extracted Hamiltonian blocks
    - U_new_block: reserved compatibility output for unitary transforms (empty)
    """

    # Inputs can be large dense arrays
    hamk = Hamk_list

    # Normalize list-like shapes
    num_layer_arr = np.array(num_layer_list)
    num_orb_per_layer = [np.array(x) for x in num_orb_per_layer_list]
    total_layers = int(num_layer_arr.sum())
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}. "
            "Use [] for layers that do not contribute."
        )
    if norb_fix_list and len(norb_fix_list) != total_layers:
        raise ValueError(
            f"norb_fix_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(norb_fix_list)}."
        )
    if selected_bands_by_layer is not None and len(selected_bands_by_layer) != total_layers:
        raise ValueError(
            f"selected_bands_by_layer must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(selected_bands_by_layer)}."
        )
    # Number of Q points per group
    num_q_list = [np.array([len(q_layer) for q_layer in q_group]) for q_group in Qlayer_list]

    H_diag_block: List[np.ndarray] = []
    H_GM_diag_eig: List[np.ndarray] = []
    H_GM_diag_eig_vec: List[np.ndarray] = []
    U_new_block: List[np.ndarray] = []

    # helper for spin handling
    def apply_spin(idx: np.ndarray) -> np.ndarray:
        if spin == "all":
            return np.concatenate((idx, idx + hamk.shape[1] // 2))
        if spin == "down":
            return idx + hamk.shape[1] // 2
        return idx

    # Expected orbitals per layer (assuming group 0 orbitals define block width)
    orb_per_layer_0 = int(num_orb_per_layer[0][0])
    q_count_0 = int(num_q_list[0][0])

    if mode == "gamma":
        # Same-Q across all layers grouped together
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover
            tqdm = lambda x, total=None, miniters=None: x  # fallback

        # for iqx in tqdm(range(q_count_0), total=q_count_0, miniters=1):
        for iqx in range(q_count_0):
            same_q_index_parts: List[np.ndarray] = []
            for ilx in range(len(num_layer_arr)):
                for j in range(num_layer_arr[ilx]):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer_0, (iqx_ + 1) * orb_per_layer_0)
                    # index2 = np.array([15, 18, 20, 16, 19, 21, 24, 26, 25, 27, 14, 22, 17, 23])-14
                    # iqx2 = index2[iqx] * num_layer_arr[ilx] + j
                    # base_2 = np.arange(iqx2 * orb_per_layer_0, (iqx2 + 1) * orb_per_layer_0)
                    # if ilx != 0:
                    #     print("base_2 = ",base_2)
                    #     print("base = ",base)
                    shift = q_count_0 * orb_per_layer_0
                    if ilx == 0:
                        same_q_index_parts.append(base + shift * num_layer_arr[:ilx].sum())
                    else:
                        same_q_index_parts.append(base + shift * num_layer_arr[:ilx].sum())
            # print(same_q_index_parts)
            same_q_index = np.concatenate(same_q_index_parts)

            expected = int(num_layer_arr.sum()) * orb_per_layer_0
            if same_q_index.shape[0] != expected:
                raise ValueError(
                    f"Mismatch in block size: got {same_q_index.shape[0]} vs expected {expected}"
                )

            same_q_index = apply_spin(same_q_index)
            block = _extract_square_block(hamk, same_q_index)
            # print(block[10,20])
            selected_bands: list[int] | None = None
            if selected_bands_by_layer is not None:
                selected_bands = sorted(
                    {
                        int(band)
                        for layer_bands in selected_bands_by_layer
                        for band in layer_bands
                    }
                )
            eig, vec = _hermitian_eigh_columns(block, selected_bands)
            # print(np.sort(eig)[:5],np.linalg.norm(block))
            if nlow_state_list and norb_fix_list:
                bands_flat: list[int] = []
                ref_flat: list[list[tuple[int, complex]]] = []
                for layer in range(len(nlow_state_list)):
                    layer_context = f"mode gamma q {iqx} layer {layer}"
                    bands_layer, refs_layer = _reference_terms_for_layer(
                        nlow_state_list=nlow_state_list,
                        norb_fix_list=norb_fix_list,
                        layer=layer,
                        block_dim=vec.shape[0],
                        context=layer_context,
                    )
                    bands_flat.extend(bands_layer)
                    ref_flat.extend(refs_layer)
                if bands_flat:
                    if len(set(bands_flat)) != len(bands_flat):
                        raise ValueError(f"mode gamma q {iqx}: duplicate low-state band indices are ambiguous")
                    _align_selected_eigenstates(
                        vec,
                        bands_flat,
                        ref_flat,
                        context=f"mode gamma q {iqx}",
                    )

            H_diag_block.append(block)
            H_GM_diag_eig.append(eig)

            H_GM_diag_eig_vec.append(vec)

        H_GM_diag_eig = np.array(H_GM_diag_eig, dtype=object)
        H_GM_diag_eig_vec = np.array(H_GM_diag_eig_vec, dtype=object)
        H_diag_block = np.array(H_diag_block, dtype=object)

    else:
        # Per-layer, per-Q blocks
        for ilx in range(len(num_layer_arr)):
            for jj in range(num_layer_arr[ilx]):
                for iq in range(int(num_q_list[ilx][0])):
                    iqx = iq * num_layer_arr[ilx] + jj
                    base = np.arange(iqx * orb_per_layer_0, (iqx + 1) * orb_per_layer_0)
                    shift = q_count_0 * orb_per_layer_0
                    same_q_index = base + shift * num_layer_arr[:ilx].sum()
                    same_q_index = apply_spin(same_q_index)
                    # print(f"ilx = {ilx}, j = {jj}, iq = {iq}, iqx = {iqx}, base = {base}, shift = {shift}, same_q_index = {same_q_index}")
                    block = _extract_square_block(hamk, same_q_index)
                    entry_layer = _physical_layer_entry_index(nlow_state_list, num_layer_arr, int(ilx), int(jj))
                    selected_bands = None if selected_bands_by_layer is None else [int(band) for band in selected_bands_by_layer[entry_layer]]
                    eig, vec = _hermitian_eigh_columns(block, selected_bands)

                    if nlow_state_list and norb_fix_list:
                        layer_for_global = int(num_layer_arr[:ilx].sum() + jj)
                        context = f"mode {mode} layer {entry_layer} q {iq}"
                        bands_flat, ref_flat = _reference_terms_for_layer(
                            nlow_state_list=nlow_state_list,
                            norb_fix_list=norb_fix_list,
                            layer=entry_layer,
                            block_dim=vec.shape[0],
                            context=context,
                            allow_layer_global=spin != "all" and vec.shape[0] == orb_per_layer_0,
                            layer_for_global=layer_for_global,
                            layer_block_dim=orb_per_layer_0,
                            total_layers=int(num_layer_arr.sum()),
                        )
                        _align_selected_eigenstates(
                            vec,
                            bands_flat,
                            ref_flat,
                            context=context,
                        )



                    H_diag_block.append(block)
                    H_GM_diag_eig.append(eig)
                    H_GM_diag_eig_vec.append(vec)
    return (
        np.array(H_GM_diag_eig, dtype=object),
        np.array(H_GM_diag_eig_vec, dtype=object),
        np.array(H_diag_block, dtype=object),
        np.array(U_new_block, dtype=object),
    )


def calculate_energy_lists(
    H_GM_diag_eig_vec,
    nlow_state_list,
    norb_fix_list,
    Qlayer_list,
    num_orb_per_layer_list,
    mode="gamma",
    *,
    include_high: bool = True,
):
    """
    Calculate low energy and high energy lists from the given eigenvector matrix.

    Parameters:
    H_GM_diag_eig_vec (numpy.ndarray): Eigenvector matrix.
    nlow_state_list: List of low energy state indices per layer.
    norb_fix_list: List of fixed orbital indices per layer.
    Qlayer_list: List of Q layers.
    num_orb_per_layer_list: Number of orbitals per layer.
    mode: "gamma" or "K1"/"K2" - determines indexing structure.

    Returns:
    tuple: Low energy list (numpy.ndarray), High energy list (numpy.ndarray).
    """
    # Normalize mode to lowercase
    mode_lower = mode.lower() if isinstance(mode, str) else mode

    theta_rot = 0
    U_low_proj = None
    layer_low_blocks = []
    layer_row_dims = []
    layer_low_col_dims = []
    Qlayer1, Qlayer2 = Qlayer_list
    Qlayer1 = Qlayer1[0]
    Qlayer2 = Qlayer2[0]
    phase = 1

    # Convert H_GM_diag_eig_vec to list if it's an array
    if isinstance(H_GM_diag_eig_vec, np.ndarray):
        H_vec_list = H_GM_diag_eig_vec.tolist()
    else:
        H_vec_list = H_GM_diag_eig_vec

    # Ensure all elements are numpy arrays
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_vec_list]

    for ilayer in range(len(nlow_state_list)):
        nlow_state = nlow_state_list[ilayer]
        Qlayer = Qlayer1 if ilayer == 0 else Qlayer2
        q_count_layer = len(Qlayer)
        vec_probe_idx = 0 if mode_lower == "gamma" else ilayer * len(Qlayer1)
        if vec_probe_idx >= len(H_vec_list):
            raise IndexError(f"vec_probe_idx {vec_probe_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}")
        vec_probe = np.asarray(H_vec_list[vec_probe_idx], dtype=np.complex128)
        row_dim = vec_probe.shape[0] * q_count_layer
        layer_row_dims.append(row_dim)
        layer_low_col_dims.append(len(nlow_state) * q_count_layer)
        if nlow_state == []:
            layer_low_blocks.append(np.zeros((row_dim, 0), dtype=np.complex128))
            continue
        layer_low = np.zeros((row_dim, len(nlow_state) * q_count_layer), dtype=np.complex128)

        bands_arr = np.asarray(nlow_state, dtype=np.intp)
        col_base = np.arange(len(bands_arr), dtype=np.intp) * q_count_layer
        for i in range(q_count_layer):
            # Index calculation depends on mode
            if mode_lower == "gamma":
                # gamma mode: structure is [Q0_all_layers, Q1_all_layers, ...]
                # But get_H_block returns [Q0, Q1, ...] for gamma mode
                vec_idx = i
            else:
                # non-gamma mode: structure is [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, ...]
                # Index = layer_offset + Q_index
                # Assuming Qlayer1 and Qlayer2 have same length for simplicity
                layer_offset = ilayer * len(Qlayer1)
                vec_idx = layer_offset + i

            if vec_idx >= len(H_vec_list):
                raise IndexError(f"vec_idx {vec_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}, i={i}")

            vec = np.asarray(H_vec_list[vec_idx], dtype=np.complex128)
            row_start = i * vec.shape[0]
            row_stop = row_start + vec.shape[0]
            layer_low[row_start:row_stop, col_base + i] = vec[:, bands_arr]
        layer_low_blocks.append(layer_low)

    total_low_cols = sum(layer_low_col_dims)
    U_low_proj = np.zeros((sum(layer_row_dims), total_low_cols), dtype=np.complex128)
    row_offset = 0
    col_offset = 0
    for block, row_dim, col_dim in zip(layer_low_blocks, layer_row_dims, layer_low_col_dims):
        if col_dim:
            U_low_proj[row_offset:row_offset + row_dim, col_offset:col_offset + col_dim] = block
        row_offset += row_dim
        col_offset += col_dim

    def gamma_full_row_order(block_dim: int, q_count: int, orb0: int) -> np.ndarray:
        """Map q-major same-Q block rows back to the full Hamiltonian row order."""
        group_layer_counts = [len(group) for group in num_orb_per_layer_list]
        total_layers = sum(group_layer_counts)
        per_spin_dim = total_layers * orb0
        if block_dim == 2 * per_spin_dim:
            spin_count = 2
        elif block_dim == per_spin_dim:
            spin_count = 1
        else:
            raise ValueError(
                f"Cannot infer gamma row order from block_dim={block_dim}, "
                f"orb0={orb0}, layers={group_layer_counts}"
            )

        order = []
        spin_full_offset = q_count * per_spin_dim
        for spin in range(spin_count):
            q_block_spin_offset = spin * per_spin_dim
            for group_index, layer_count in enumerate(group_layer_counts):
                layer_offset = sum(group_layer_counts[:group_index])
                for local in range(q_count * layer_count):
                    q_index = local // layer_count
                    layer_in_group = local % layer_count
                    row_base = (
                        q_index * block_dim
                        + q_block_spin_offset
                        + (layer_offset + layer_in_group) * orb0
                    )
                    order.extend(range(row_base, row_base + orb0))
        if spin_count == 2:
            down_start = q_count * per_spin_dim
            if order[down_start:down_start + orb0] == order[:orb0]:
                raise AssertionError("gamma row-order construction duplicated spin sectors")
        return np.asarray(order, dtype=int)

    # The following reordering is only for gamma mode. get_H_block returns rows
    # as q-major same-Q blocks; full Hamk rows are grouped by source sector.
    gamma_row_order = None
    if mode_lower == "gamma":
        n_g = len(Qlayer1)
        orb0 = num_orb_per_layer_list[0][0]
        block_dim0 = np.asarray(H_vec_list[0], dtype=np.complex128).shape[0]
        gamma_row_order = gamma_full_row_order(block_dim0, n_g, int(orb0))
        U_low_proj = U_low_proj[gamma_row_order]

    # print("shape of Uproj = ",np.shape(U_low_proj))
    if not include_high:
        return U_low_proj, None

    U_high_proj = None
    layer_high_blocks = []
    layer_high_col_dims = []
    for ilayer in range(len(nlow_state_list)):
        nlow_state = nlow_state_list[ilayer]
        Qlayer = Qlayer1 if ilayer == 0 else Qlayer2
        q_count_layer = len(Qlayer)
        vec_probe_idx = 0 if mode_lower == "gamma" else ilayer * len(Qlayer1)
        vec_probe = np.asarray(H_vec_list[vec_probe_idx], dtype=np.complex128)
        high_col_count = vec_probe.shape[1] - len(nlow_state)
        high_block = np.zeros(
            (vec_probe.shape[0] * q_count_layer, high_col_count * q_count_layer),
            dtype=np.complex128,
        )
        high_bands = _complement_indices(vec_probe.shape[1], nlow_state)
        for i in range(len(Qlayer)):
            # Index calculation depends on mode
            if mode_lower == "gamma":
                vec_idx = i
            else:
                # non-gamma mode: structure is [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, ...]
                layer_offset = ilayer * len(Qlayer1)
                vec_idx = layer_offset + i

            if vec_idx >= len(H_vec_list):
                raise IndexError(f"vec_idx {vec_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}, i={i}")

            # Ensure vec is a numpy array
            vec = np.asarray(H_vec_list[vec_idx], dtype=np.complex128)
            vecpart = vec[:, high_bands]
            # for j in range(vecpart.shape[1]):
            #     vecpart[:,j] = vecpart[:,j] * ((vecpart[norb_fix,j] / np.abs(vecpart[norb_fix,j])) ** (-1)) * np.exp(-1j * 2 * theta_rot * np.pi / 180)
            row_start = i * vec.shape[0]
            row_stop = row_start + vec.shape[0]
            col_start = i * vecpart.shape[1]
            col_stop = col_start + vecpart.shape[1]
            high_block[row_start:row_stop, col_start:col_stop] = vecpart
        layer_high_blocks.append(high_block)
        layer_high_col_dims.append(high_block.shape[1])
    total_high_cols = sum(layer_high_col_dims)
    U_high_proj = np.zeros((sum(layer_row_dims), total_high_cols), dtype=np.complex128)
    row_offset = 0
    col_offset = 0
    for block, row_dim, col_dim in zip(layer_high_blocks, layer_row_dims, layer_high_col_dims):
        if col_dim:
            U_high_proj[row_offset:row_offset + row_dim, col_offset:col_offset + col_dim] = block
        row_offset += row_dim
        col_offset += col_dim

    # The following reordering is only for gamma mode
    if mode_lower == "gamma":
        U_high_proj = U_high_proj[gamma_row_order]

    # print("shape of U_high_proj = ",np.shape(U_high_proj))
    # ULowEnergyList = np.array(ULowEnergyList)
    # UHighEnergyList = np.array(UHighEnergyList)

    # return ULowEnergyList, UHighEnergyList
    return U_low_proj, U_high_proj


def _assemble_projectors_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Assemble block eigenvectors in the original full-Hamiltonian row order."""
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("projector assembly requires at least one index block")

    full_dim = max(int(np.max(idx)) for idx in idx_list) + 1
    q_count = len(idx_list) // len(nlow_state_list)
    if q_count * len(nlow_state_list) != len(idx_list):
        raise ValueError("index blocks must be ordered by layer then Q")

    bands_by_layer = [[int(band) for band in layer_bands] for layer_bands in nlow_state_list]
    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_layer:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count
    U_low = np.zeros((full_dim, low_dim), dtype=np.complex128)
    U_high = None
    if include_high:
        high_dim = sum(vec.shape[1] - len(nlow_state_list[i // q_count]) for i, vec in enumerate(H_vec_list))
        U_high = np.zeros((full_dim, high_dim), dtype=np.complex128)

    high_col = 0
    for block_idx, (vec, idx) in enumerate(zip(H_vec_list, idx_list)):
        layer = block_idx // q_count
        q_index = block_idx % q_count
        bands = bands_by_layer[layer]
        if bands:
            for band_slot, band in enumerate(bands):
                col = low_offsets[layer] + band_slot * q_count + q_index
                U_low[idx, col] = vec[:, band]
        if U_high is not None:
            high_bands = np.delete(np.arange(vec.shape[1]), bands)
            U_high[idx, high_col : high_col + len(high_bands)] = vec[:, high_bands]
            high_col += len(high_bands)

    return U_low, U_high


def _assemble_projector_groups_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Assemble block-sparse projector groups in the projected basis order."""
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("projector assembly requires at least one index block")

    bands_by_layer = [[int(band) for band in layer_bands] for layer_bands in nlow_state_list]
    q_count = len(idx_list) // len(bands_by_layer)
    if q_count * len(bands_by_layer) != len(idx_list):
        raise ValueError("index blocks must be ordered by layer then Q")

    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_layer:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count

    low_locals: list[np.ndarray] = []
    low_cols: list[np.ndarray] = []
    high_locals: list[np.ndarray] = []
    high_cols: list[np.ndarray] = []
    high_dim = 0

    for block_idx, vec in enumerate(H_vec_list):
        layer = block_idx // q_count
        q_index = block_idx % q_count
        bands = bands_by_layer[layer]
        if bands:
            bands_arr = np.asarray(bands, dtype=np.intp)
            cols = np.asarray(
                [low_offsets[layer] + band_slot * q_count + q_index for band_slot in range(len(bands))],
                dtype=np.intp,
            )
            low_locals.append(np.ascontiguousarray(vec[:, bands_arr], dtype=np.complex128))
            low_cols.append(cols)
        else:
            low_locals.append(np.zeros((vec.shape[0], 0), dtype=np.complex128))
            low_cols.append(np.zeros(0, dtype=np.intp))

        if include_high:
            high_bands = _complement_indices(vec.shape[1], bands)
            high_locals.append(np.ascontiguousarray(vec[:, high_bands], dtype=np.complex128))
            high_cols.append(np.arange(high_dim, high_dim + high_bands.size, dtype=np.intp))
            high_dim += int(high_bands.size)

    low_groups = projector_groups_from_block_columns(
        idx_list,
        low_locals,
        low_cols,
        n_columns=low_dim,
    )
    if not include_high:
        return low_groups, None
    high_groups = projector_groups_from_block_columns(
        idx_list,
        high_locals,
        high_cols,
        n_columns=high_dim,
    )
    return low_groups, high_groups


def _flatten_band_groups(nlow_state_list: Any) -> tuple[list[list[int]], list[int]]:
    bands_by_group = [[int(band) for band in group] for group in nlow_state_list]
    flat = [band for group in bands_by_group for band in group]
    if len(flat) != len(set(flat)):
        raise ValueError("Gamma same-Q projector bands must be unique across sector groups")
    return bands_by_group, flat


def _assemble_gamma_projectors_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Assemble Gamma same-Q projectors without duplicating the full row space.

    Gamma blocks contain all active sectors/layers for a given Q. A nested
    nlow_state_list such as [[0, 1], [2, 3]] therefore describes sector labels
    inside the same block, not independent block row spaces.
    """
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"Gamma projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("Gamma projector assembly requires at least one index block")

    bands_by_group, flat_bands = _flatten_band_groups(nlow_state_list)
    q_count = len(idx_list)
    full_dim = max(int(np.max(idx)) for idx in idx_list) + 1

    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_group:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count

    U_low = np.zeros((full_dim, low_dim), dtype=np.complex128)
    high_dim = 0
    high_bands_by_q: list[np.ndarray] = []
    if include_high:
        for vec in H_vec_list:
            high_bands = _complement_indices(vec.shape[1], flat_bands)
            high_bands_by_q.append(high_bands)
            high_dim += int(high_bands.size)
        U_high = np.zeros((full_dim, high_dim), dtype=np.complex128)
    else:
        U_high = None

    high_col = 0
    for q_index, (vec, idx) in enumerate(zip(H_vec_list, idx_list)):
        idx = np.asarray(idx, dtype=np.intp)
        for group_index, bands in enumerate(bands_by_group):
            for band_slot, band in enumerate(bands):
                col = low_offsets[group_index] + band_slot * q_count + q_index
                U_low[idx, col] = vec[:, band]
        if U_high is not None:
            high_bands = high_bands_by_q[q_index]
            U_high[idx, high_col : high_col + high_bands.size] = vec[:, high_bands]
            high_col += int(high_bands.size)

    return U_low, U_high


def _assemble_gamma_projector_groups_from_block_eigenvectors(
    H_GM_diag_eig_vec,
    idx_list: list[np.ndarray],
    nlow_state_list,
    *,
    include_high: bool,
):
    """Block-sparse variant of _assemble_gamma_projectors_from_block_eigenvectors."""
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_GM_diag_eig_vec.tolist()]
    if len(H_vec_list) != len(idx_list):
        raise ValueError(f"Gamma projector block count mismatch: {len(H_vec_list)} vectors vs {len(idx_list)} index blocks")
    if not idx_list:
        raise ValueError("Gamma projector assembly requires at least one index block")

    bands_by_group, flat_bands = _flatten_band_groups(nlow_state_list)
    q_count = len(idx_list)

    low_offsets: list[int] = []
    low_dim = 0
    for bands in bands_by_group:
        low_offsets.append(low_dim)
        low_dim += len(bands) * q_count

    low_locals: list[np.ndarray] = []
    low_cols: list[np.ndarray] = []
    high_locals: list[np.ndarray] = []
    high_cols: list[np.ndarray] = []
    high_dim = 0

    for q_index, vec in enumerate(H_vec_list):
        cols: list[int] = []
        local_band_order: list[int] = []
        for group_index, bands in enumerate(bands_by_group):
            for band_slot, band in enumerate(bands):
                cols.append(low_offsets[group_index] + band_slot * q_count + q_index)
                local_band_order.append(int(band))
        low_cols.append(np.asarray(cols, dtype=np.intp))
        low_locals.append(np.ascontiguousarray(vec[:, np.asarray(local_band_order, dtype=np.intp)], dtype=np.complex128))

        if include_high:
            high_bands = _complement_indices(vec.shape[1], flat_bands)
            high_locals.append(np.ascontiguousarray(vec[:, high_bands], dtype=np.complex128))
            high_cols.append(np.arange(high_dim, high_dim + high_bands.size, dtype=np.intp))
            high_dim += int(high_bands.size)

    low_groups = projector_groups_from_block_columns(
        idx_list,
        low_locals,
        low_cols,
        n_columns=low_dim,
    )
    if not include_high:
        return low_groups, None
    high_groups = projector_groups_from_block_columns(
        idx_list,
        high_locals,
        high_cols,
        n_columns=high_dim,
    )
    return low_groups, high_groups


def project_heff_full(
    hamk_full: np.ndarray,
    q_count: int,
    orb_per_layer0: int,
    num_layer_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    bands: List[int] | List[List[int]] = None,
    comps: List[int] | List[List[int]] = None,
    # Optional path: call get_H_block directly to reuse alignment logic.
    Qlayer_list: List[List[np.ndarray]] | None = None,
    num_orb_per_layer_list: List[List[int]] | None = None,
    nlow_state_list: List[List[int]] | None = None,
    norb_fix_list: List[List[List[Tuple[int, complex]]]] | None = None,
    second_order: bool = True,
    E_ref: float | None = None,
    mode = "Gamma",
    downfold_method: str | None = None,
    pole_warning_mev: float = 10.0,
    pole_danger_mev: float = 1.0,
    fail_on_near_pole: bool = False,
    compute_pole_diagnostics: bool = False,
    compute_condition_number: bool = False,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project full H(k) to Heff(k).

    一阶：Heff = U_low^† H U_low
    二阶（Löwdin/SW 下折）：Heff = H00 + H01 (E_ref I - H11)^{-1} H10

    参数
    - hamk_full: (N, N) 该 k 点的全空间厄米矩阵
    - q_count: Q 的个数（如 len(Q_set1)）
    - orb_per_layer0: 单自旋、每层、每 Q 的轨道数（用于索引）
    - num_layer_list: 每个“侧/谷”的层数，如 [1,1]
    - spin: 'up'|'down'|'all'（影响索引映射）
    - bands: 每个 Q 选取的“低能带”索引（可平铺或嵌套）；长度 = m_bands
    - comps: 每个低能带希望对齐的“参考全局轨道”索引（与 bands 对应；可选）
    - second_order: False 则一阶；True 则按二阶下折
    - E_ref: 二阶中的参考能量，若 None 则取 H00 本征值均值

    返回
    - Heff(k) (M, M)；其本征值 heig (M,) 与本征矢 hvec (M, M)
    """
    # if bands is None:
    #     raise ValueError("bands (nlow_state_list) must be provided")

    # 将 bands 摊平（comps 另行处理，因其可能含复系数组合）
    # def _flatten(x):
    #     if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
    #         out = []
    #         for s in x:
    #             out.extend(list(s))
    #         return list(out)
    #     return list(x)

    # bands_flat = np.array(_flatten(bands), dtype=int)

    # m_bands = bands_flat.size
    # Normalize mode to lowercase for consistent comparison
    mode_lower = mode.lower() if isinstance(mode, str) else mode
    total_layers = int(sum(int(n) for n in num_layer_list))
    if nlow_state_list is None:
        raise ValueError("nlow_state_list must be provided")
    if len(nlow_state_list) != total_layers:
        raise ValueError(
            f"nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(nlow_state_list)}. "
            "Use [] for layers that do not contribute."
        )
    has_anchor_list = bool(norb_fix_list)
    if has_anchor_list and len(norb_fix_list) != total_layers:
        raise ValueError(
            f"norb_fix_list must have {total_layers} physical-layer rows "
            f"(sum(num_layer_list)); got {len(norb_fix_list)}."
        )
    if mode_lower == "gamma":
        m_bands = sum(len(layer_bands) for layer_bands in nlow_state_list)
    else:
        nonempty_bands = [layer_bands for layer_bands in nlow_state_list if len(layer_bands) > 0]
        if nonempty_bands:
            m_bands = sum(len(layer_bands) for layer_bands in nlow_state_list)
        else:
            raise ValueError("For non-gamma mode, nlow_state_list must contain at least one active band")
    if m_bands == 0:
        raise ValueError("Empty bands list for projection")

    # 构建每个 Q 的全局索引
    num_layer_arr = np.array(num_layer_list)
    shift = q_count * orb_per_layer0
    idx_list: List[np.ndarray] = []

    if mode_lower == "gamma":
        # gamma mode: each Q combines all layers
        for iqx in range(q_count):
            parts = []
            for ilx in range(len(num_layer_arr)):
                for j in range(num_layer_arr[ilx]):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
                    parts.append(base + shift * num_layer_arr[:ilx].sum())
            same_q_index = np.concatenate(parts)
            if spin == "all":
                same_q_index = np.concatenate((same_q_index, same_q_index + hamk_full.shape[0] // 2))
            elif spin == "down":
                same_q_index = same_q_index + hamk_full.shape[0] // 2
            idx_list.append(same_q_index)
    else:
        # non-gamma mode: each (layer, Q) pair is a separate block
        # Structure: [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, layer1_Q1, ..., layer1_Qn]
        for ilx in range(len(num_layer_arr)):
            for j in range(num_layer_arr[ilx]):
                for iqx in range(q_count):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
                    same_q_index = base + shift * num_layer_arr[:ilx].sum()
                    if spin == "all":
                        same_q_index = np.concatenate((same_q_index, same_q_index + hamk_full.shape[0] // 2))
                    elif spin == "down":
                        same_q_index = same_q_index + hamk_full.shape[0] // 2
                    idx_list.append(same_q_index)

    method = (downfold_method or ("fixed_schur" if second_order else "first_order")).lower()
    include_high = method != "first_order"

    if Qlayer_list is None:
        Qlayer_list = [[np.arange(q_count) for _ in range(n)] for n in num_layer_list]
    if num_orb_per_layer_list is None:
        num_orb_per_layer_list = [[int(orb_per_layer0) for _ in range(n)] for n in num_layer_list]

    H_eig_blk, H_vec_blk, _, _ = get_H_block(
        hamk_full,
        Qlayer_list,
        num_layer_list,
        num_orb_per_layer_list,
        nlow_state_list,
        norb_fix_list if has_anchor_list else [],
        spin=spin,
        mode=mode_lower,
        selected_bands_by_layer=None if include_high else nlow_state_list,
    )

    options = DownfoldingOptions(
        method=method,
        e_ref=E_ref,
        pole_warning_mev=float(pole_warning_mev),
        pole_danger_mev=float(pole_danger_mev),
        fail_on_near_pole=bool(fail_on_near_pole),
        compute_pole_diagnostics=bool(compute_pole_diagnostics) or bool(fail_on_near_pole),
        compute_condition_number=bool(compute_condition_number),
    )
    if mode_lower == "gamma":
        low_groups, high_groups = _assemble_gamma_projector_groups_from_block_eigenvectors(
            H_vec_blk,
            idx_list,
            _source_group_band_lists(nlow_state_list, num_layer_list),
            include_high=include_high,
        )
        result = downfold_from_projector_groups(hamk_full, low_groups, high_groups, options)
    else:
        low_groups, high_groups = _assemble_projector_groups_from_block_eigenvectors(
            H_vec_blk,
            idx_list,
            nlow_state_list,
            include_high=include_high,
        )
        result = downfold_from_projector_groups(hamk_full, low_groups, high_groups, options)

    Heff = result.heff
    heig, hvec = _hermitian_eigh(Heff)
    if return_diagnostics:
        return Heff, heig, hvec, result
    return Heff, heig, hvec
