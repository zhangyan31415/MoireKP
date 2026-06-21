from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
import yaml


@dataclass(frozen=True)
class AutoGaugeConfig:
    method: str = "auto_scdm"
    anchor_scope: str = "per_sector"
    candidate_pool: str = "all"
    projected_anchors: bool = False
    min_sigma: float = 1.0e-6
    max_condition: float = 1.0e6
    reference_q_index: int = 0
    basis_is_orthonormal: bool = True


@dataclass(frozen=True)
class AnchorCandidate:
    row: int
    leverage: float
    label: str | None = None
    selected: bool = False


@dataclass(frozen=True)
class AutoGaugeSelection:
    selected_rows: list[int]
    method: str
    singular_values: list[float]
    sigma_min: float
    condition_number: float
    rank: int
    leverage_scores: list[float]
    candidates: list[AnchorCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GaugeCandidateSymmetryMetrics:
    candidate_id: str
    exactification_distance_by_op: Mapping[str, float]
    active_term_count: int | None = None
    phase_branch_distance_by_op: Mapping[str, float] = field(default_factory=dict)
    support_off_by_op: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GaugeCandidateSymmetryDecision:
    selected: GaugeCandidateSymmetryMetrics
    rankings: list[Mapping[str, Any]]


@dataclass(frozen=True)
class GaugeAnchorReport:
    gauge_mode: str
    resolved_norb_fix_list: list[Any]
    selections: list[Mapping[str, Any]]
    metric: Mapping[str, Any]
    state_selection_quality: Mapping[str, Any]
    gauge_anchor_quality: Mapping[str, Any]
    symmetry_closure_quality: Mapping[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        if abs(value.imag) < 1.0e-15:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _validate_u_low(U_low: np.ndarray) -> np.ndarray:
    u = np.asarray(U_low, dtype=np.complex128)
    if u.ndim != 2:
        raise ValueError(f"U_low must be a 2D matrix, got shape={u.shape}")
    if u.shape[0] == 0 or u.shape[1] == 0:
        raise ValueError(f"U_low must be non-empty, got shape={u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("U_low contains NaN or Inf")
    return u


def _validate_overlap(overlap: np.ndarray | None, dim: int) -> np.ndarray | None:
    if overlap is None:
        return None
    s = np.asarray(overlap, dtype=np.complex128)
    if s.shape != (dim, dim):
        raise ValueError(f"overlap must have shape {(dim, dim)}, got {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("overlap contains NaN or Inf")
    return s


def _candidate_rows(candidate_rows: Sequence[int] | None, basis_size: int) -> list[int]:
    if candidate_rows is None:
        rows = list(range(basis_size))
    else:
        rows = sorted({int(row) for row in candidate_rows})
    for row in rows:
        if row < 0 or row >= basis_size:
            raise IndexError(f"candidate row {row} outside basis size {basis_size}")
    if not rows:
        raise ValueError("candidate_rows must not be empty")
    return rows


def _metric_row_matrix(
    U_low: np.ndarray,
    *,
    overlap: np.ndarray | None,
    basis_is_orthonormal: bool,
) -> np.ndarray:
    u = _validate_u_low(U_low)
    s = _validate_overlap(overlap, u.shape[0])
    if basis_is_orthonormal:
        if s is not None:
            raise ValueError("overlap was provided but basis_is_orthonormal=True")
        return u
    if s is None:
        raise ValueError("overlap is required when basis_is_orthonormal=False")
    evals, evecs = np.linalg.eigh(0.5 * (s + s.conj().T))
    if np.any(evals <= 0.0):
        raise ValueError("overlap must be positive definite for metric-aware leverage")
    overlap_sqrt = (evecs * np.sqrt(evals)) @ evecs.conj().T
    return overlap_sqrt @ u


def compute_leverage_scores(
    U_low: np.ndarray,
    overlap: np.ndarray | None = None,
    basis_is_orthonormal: bool = True,
    candidate_rows: Sequence[int] | None = None,
) -> np.ndarray:
    """Return finite-basis leverage scores for rows of a low subspace."""

    row_matrix = _metric_row_matrix(
        U_low,
        overlap=overlap,
        basis_is_orthonormal=basis_is_orthonormal,
    )
    rows = _candidate_rows(candidate_rows, row_matrix.shape[0])
    scores = np.real(np.sum(np.abs(row_matrix) ** 2, axis=1))
    scores = np.maximum(scores, 0.0)
    if not np.all(np.isfinite(scores[rows])):
        raise ValueError("leverage scores contain NaN or Inf")
    return scores


def _weighted_rows(row_matrix: np.ndarray, rows: Sequence[int], weights: Sequence[float] | None) -> np.ndarray:
    matrix = np.asarray(row_matrix[np.asarray(rows, dtype=np.intp), :], dtype=np.complex128)
    if weights is None:
        return matrix
    weights_arr = np.asarray([float(weights[int(row)]) for row in rows], dtype=float)
    if weights_arr.shape != (len(rows),):
        raise ValueError("weights must provide one scalar per basis row")
    if np.any(weights_arr < 0.0) or not np.all(np.isfinite(weights_arr)):
        raise ValueError("weights must be finite non-negative values")
    return matrix * np.sqrt(weights_arr)[:, np.newaxis]


def select_anchor_rows_qrcp(
    U_low: np.ndarray,
    n_anchors: int,
    overlap: np.ndarray | None = None,
    candidate_rows: Sequence[int] | None = None,
    labels: Mapping[int, str] | Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    method: str = "qrcp",
    basis_is_orthonormal: bool = True,
    rank_tol: float = 1.0e-10,
) -> AutoGaugeSelection:
    """Select anchor rows whose trial-projector overlap is full rank."""

    if method != "qrcp":
        raise ValueError(f"Unsupported auto gauge selection method {method!r}")
    if int(n_anchors) <= 0:
        raise ValueError(f"n_anchors must be positive, got {n_anchors}")
    n_anchors = int(n_anchors)

    row_matrix = _metric_row_matrix(
        U_low,
        overlap=overlap,
        basis_is_orthonormal=basis_is_orthonormal,
    )
    basis_size, subspace_dim = row_matrix.shape
    if n_anchors > subspace_dim:
        raise ValueError(f"n_anchors={n_anchors} exceeds low-subspace dimension {subspace_dim}")
    rows = _candidate_rows(candidate_rows, basis_size)
    if len(rows) < n_anchors:
        raise ValueError(
            f"candidate row set is rank deficient: need {n_anchors} anchors but only {len(rows)} candidates"
        )

    leverage = compute_leverage_scores(
        U_low,
        overlap=overlap,
        basis_is_orthonormal=basis_is_orthonormal,
        candidate_rows=rows,
    )
    candidate_matrix = _weighted_rows(row_matrix, rows, weights)
    warnings: list[str] = []
    try:
        _q, _r, pivots = scipy.linalg.qr(candidate_matrix.T, mode="economic", pivoting=True)
        pivot_positions = [int(pos) for pos in pivots[:n_anchors]]
        selected_rows = [int(rows[pos]) for pos in pivot_positions]
        actual_method = "qrcp"
    except Exception as exc:  # pragma: no cover - SciPy pivoted QR is expected.
        warnings.append(f"QRCP failed ({exc}); used leverage fallback")
        ordered = sorted(rows, key=lambda row: (-float(leverage[row]), int(row)))
        selected_rows = ordered[:n_anchors]
        actual_method = "leverage_fallback"

    overlap_matrix = row_matrix[np.asarray(selected_rows, dtype=np.intp), :]
    singular_values = np.linalg.svd(overlap_matrix, compute_uv=False)
    rank = int(np.sum(singular_values > float(rank_tol)))
    required_rank = min(n_anchors, subspace_dim)
    if rank < required_rank:
        raise ValueError(
            f"selected anchor rows are rank deficient: rank={rank}, required={required_rank}, "
            f"rows={selected_rows}"
        )
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    sigma_max = float(np.max(singular_values)) if singular_values.size else 0.0
    condition = float("inf") if sigma_min <= 0.0 else float(sigma_max / sigma_min)

    def label_for(row: int) -> str | None:
        if labels is None:
            return None
        if isinstance(labels, Mapping):
            return labels.get(int(row))
        if 0 <= int(row) < len(labels):
            return str(labels[int(row)])
        return None

    top_rows = sorted(rows, key=lambda row: (-float(leverage[row]), int(row)))[: min(len(rows), 16)]
    candidates = [
        AnchorCandidate(
            row=int(row),
            leverage=float(leverage[row]),
            label=label_for(int(row)),
            selected=int(row) in set(selected_rows),
        )
        for row in top_rows
    ]
    return AutoGaugeSelection(
        selected_rows=selected_rows,
        method=actual_method,
        singular_values=[float(x) for x in singular_values.tolist()],
        sigma_min=sigma_min,
        condition_number=condition,
        rank=rank,
        leverage_scores=[float(leverage[row]) for row in selected_rows],
        candidates=candidates,
        warnings=warnings,
    )


def select_gauge_candidate_by_symmetry(
    candidates: Sequence[GaugeCandidateSymmetryMetrics],
    *,
    max_exactification_distance: float = 1.0e-3,
    ambiguity_exactification_tolerance: float = 1.0e-6,
) -> GaugeCandidateSymmetryDecision:
    """Choose an auto-gauge candidate from symmetry validation metrics.

    This is the small policy layer used after candidate anchors have been
    projected through raw-H symmetry and exactified.  It deliberately treats
    missing metrics as invalid instead of filling them with zero.
    """

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("No gauge candidates were provided for symmetry validation")
    if max_exactification_distance <= 0.0:
        raise ValueError("max_exactification_distance must be positive")
    if ambiguity_exactification_tolerance < 0.0:
        raise ValueError("ambiguity_exactification_tolerance must be non-negative")

    ranked: list[dict[str, Any]] = []
    valid: list[tuple[tuple[int, float, float, int, int, float, str], GaugeCandidateSymmetryMetrics, dict[str, Any]]] = []
    for candidate in candidate_list:
        if not candidate.candidate_id:
            raise ValueError("gauge candidate id must be non-empty")
        distances = {
            str(op): float(value)
            for op, value in dict(candidate.exactification_distance_by_op).items()
        }
        phase_distances = {
            str(op): float(value)
            for op, value in dict(candidate.phase_branch_distance_by_op).items()
        }
        support_off = {
            str(op): float(value)
            for op, value in dict(candidate.support_off_by_op).items()
        }
        finite_distances = [
            value for value in distances.values() if np.isfinite(value) and value >= 0.0
        ]
        finite_phase = [
            value for value in phase_distances.values() if np.isfinite(value) and value >= 0.0
        ]
        finite_support = [
            value for value in support_off.values() if np.isfinite(value) and value >= 0.0
        ]
        reasons: list[str] = []
        if len(finite_distances) != len(distances) or not finite_distances:
            reasons.append("missing_or_invalid_exactification_distance")
        if len(finite_phase) != len(phase_distances):
            reasons.append("invalid_phase_branch_distance")
        if len(finite_support) != len(support_off):
            reasons.append("invalid_support_off_metric")
        max_distance = max(finite_distances) if finite_distances else None
        max_phase = max(finite_phase) if finite_phase else None
        max_support = max(finite_support) if finite_support else None
        if max_distance is not None and max_distance > float(max_exactification_distance):
            reasons.append("exactification_distance_exceeds_threshold")
        active_term_count = candidate.active_term_count
        if active_term_count is not None:
            active_term_count = int(active_term_count)
            if active_term_count < 0:
                reasons.append("active_term_count_is_negative")
        metadata = dict(candidate.metadata)
        candidate_priority = int(metadata.get("candidate_priority", 0))
        if candidate_priority < 0:
            reasons.append("candidate_priority_is_negative")
        status = "ok" if not reasons else "rejected"
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "status": status,
            "reasons": reasons,
            "max_exactification_distance": max_distance,
            "max_phase_branch_distance": max_phase,
            "max_support_off": max_support,
            "exactification_distance_by_op": distances,
            "active_term_count": active_term_count,
            "candidate_priority": candidate_priority,
            "phase_branch_distance_by_op": phase_distances,
            "support_off_by_op": support_off,
            "metadata": metadata,
        }
        active_sort = active_term_count if active_term_count is not None else 10**18
        distance_sort = max_distance if max_distance is not None else float("inf")
        phase_sort = max_phase if max_phase is not None else float("inf")
        support_sort = max_support if max_support is not None else float("inf")
        if np.isfinite(distance_sort) and ambiguity_exactification_tolerance > 0.0:
            distance_bucket = int(np.floor(float(distance_sort) / float(ambiguity_exactification_tolerance)))
        else:
            distance_bucket = 10**18
        sort_key = (
            int(distance_bucket),
            float(phase_sort),
            float(support_sort),
            int(active_sort),
            int(candidate_priority),
            float(distance_sort),
            str(candidate.candidate_id),
        )
        row["sort_key"] = [
            sort_key[0],
            sort_key[1],
            sort_key[2],
            sort_key[3],
            sort_key[4],
            sort_key[5],
            sort_key[6],
        ]
        ranked.append(row)
        if status == "ok":
            valid.append((sort_key, candidate, row))

    if not valid:
        ranked.sort(key=lambda item: (item["status"] != "ok", *item["sort_key"]))
        raise ValueError(f"No gauge candidate passed symmetry validation: {ranked}")

    valid.sort(key=lambda item: item[0])
    best_key, best_candidate, _best_row = valid[0]
    if len(valid) > 1:
        next_key, next_candidate, _next_row = valid[1]
        same_residual_bucket = best_key[0] == next_key[0]
        close_residual = abs(float(best_key[5]) - float(next_key[5])) <= float(ambiguity_exactification_tolerance)
        same_phase = (
            (not np.isfinite(best_key[1]) and not np.isfinite(next_key[1]))
            or abs(float(best_key[1]) - float(next_key[1])) <= 1.0e-12
        )
        same_support = (
            (not np.isfinite(best_key[2]) and not np.isfinite(next_key[2]))
            or abs(float(best_key[2]) - float(next_key[2])) <= 1.0e-12
        )
        same_complexity = best_key[3] == next_key[3]
        same_priority = best_key[4] == next_key[4]
        if same_residual_bucket and close_residual and same_phase and same_support and same_complexity and same_priority:
            raise ValueError(
                "ambiguous symmetry-validated gauge candidates: "
                f"{best_candidate.candidate_id!r} and {next_candidate.candidate_id!r}"
            )

    ranked.sort(
        key=lambda item: (
            item["status"] != "ok",
            *item["sort_key"],
        )
    )
    return GaugeCandidateSymmetryDecision(selected=best_candidate, rankings=ranked)


def build_reference_projectors_from_rows(
    selected_rows: Sequence[int],
    *,
    basis_size: int,
    phases: Sequence[complex] | complex | None = None,
) -> np.ndarray:
    rows = [int(row) for row in selected_rows]
    if basis_size <= 0:
        raise ValueError(f"basis_size must be positive, got {basis_size}")
    if phases is None:
        phase_values = [1.0 + 0.0j for _ in rows]
    elif isinstance(phases, (complex, float, int)):
        phase_values = [complex(phases) for _ in rows]
    else:
        phase_values = [complex(value) for value in phases]
    if len(phase_values) != len(rows):
        raise ValueError("phases must have one value per selected row")
    phi = np.zeros((int(basis_size), len(rows)), dtype=np.complex128)
    for col, (row, phase) in enumerate(zip(rows, phase_values)):
        if row < 0 or row >= basis_size:
            raise IndexError(f"selected row {row} outside basis size {basis_size}")
        phi[row, col] = phase
    return phi


def polar_align_low_subspace(
    U_low: np.ndarray,
    Phi_ref: np.ndarray,
    *,
    overlap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = _validate_u_low(U_low)
    phi = np.asarray(Phi_ref, dtype=np.complex128)
    if phi.shape != u.shape:
        raise ValueError(f"Phi_ref shape {phi.shape} must match U_low shape {u.shape}")
    if overlap is None:
        trial_overlap = phi.conj().T @ u
    else:
        s = _validate_overlap(overlap, u.shape[0])
        trial_overlap = phi.conj().T @ s @ u
    x, singular_values, yh = np.linalg.svd(trial_overlap, full_matrices=False)
    rotation = yh.conj().T @ x.conj().T
    return u @ rotation, rotation, singular_values


def write_basis_selection_report(
    output_dir: str | Path,
    report: GaugeAnchorReport | Mapping[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if isinstance(report, GaugeAnchorReport) else _json_ready(dict(report))
    (out / "basis_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if payload.get("gauge_mode") == "auto_scdm" and "resolved_norb_fix_list" in payload:
        (out / "auto_norb_fix_list.yaml").write_text(
            yaml.safe_dump(
                {"norb_fix_list": payload["resolved_norb_fix_list"]},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    (out / "basis_selection.md").write_text(_format_report_markdown(payload), encoding="utf-8")


def _format_report_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# KP Basis Selection",
        "",
        f"- gauge_mode: `{payload.get('gauge_mode')}`",
    ]
    metric = payload.get("metric", {})
    if isinstance(metric, Mapping):
        lines.append(f"- metric: `{metric.get('type', 'unknown')}`")
    quality = payload.get("gauge_anchor_quality", {})
    if isinstance(quality, Mapping):
        lines.append(f"- gauge_anchor_quality: `{quality.get('status', 'unknown')}`")
        if quality.get("sigma_min") is not None:
            lines.append(f"- sigma_min: {float(quality['sigma_min']):.6e}")
        if quality.get("condition_number") is not None:
            lines.append(f"- condition_number: {float(quality['condition_number']):.6e}")
    symmetry = payload.get("symmetry_closure_quality", {})
    if isinstance(symmetry, Mapping):
        lines.append(f"- symmetry_closure_quality: `{symmetry.get('status', 'unknown')}`")
    lines.append("")
    lines.append("## Selections")
    for idx, selection in enumerate(payload.get("selections", []) or []):
        if not isinstance(selection, Mapping):
            continue
        lines.append("")
        lines.append(f"### Selection {idx}")
        lines.append(f"- scope: `{selection.get('scope')}`")
        if selection.get("layer") is not None:
            lines.append(f"- layer: {selection.get('layer')}")
        lines.append(f"- bands: `{selection.get('bands')}`")
        lines.append(f"- selected_rows: `{selection.get('selected_rows')}`")
        if selection.get("sigma_min") is not None:
            lines.append(f"- sigma_min: {float(selection['sigma_min']):.6e}")
        if selection.get("condition_number") is not None:
            lines.append(f"- condition_number: {float(selection['condition_number']):.6e}")
    lines.append("")
    return "\n".join(lines)
