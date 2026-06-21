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

    selected_rows = sorted(selected_rows)
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
