from __future__ import annotations
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
from ..plot_style import (
    KP_BAND_BOX_ASPECT,
    KP_DPI,
    KP_LEGEND_KWARGS,
    apply_kp_axis_style,
    kp_plot_rc_context,
    kp_font_family,
    relative_energy_ylabel,
)
from ..reporting import KpReporter
from .core import load_Q_sets_from_gvec_files
from . import configuration


def _model_row_metadata_for_harmonic_scan(
    qset1: np.ndarray,
    qset2: np.ndarray,
    n_orb: tuple[int, int],
) -> list[dict[str, Any]]:
    q1 = np.asarray(qset1, dtype=float)
    q2 = np.asarray(qset2, dtype=float)
    rows: list[dict[str, Any]] = []
    for layer, qset, norb in ((1, q1, int(n_orb[0])), (2, q2, int(n_orb[1]))):
        for orbital in range(norb):
            for q_index, qvec in enumerate(qset):
                rows.append(
                    {
                        "layer": int(layer),
                        "q_index": int(q_index),
                        "orbital": int(orbital),
                        "q": np.asarray(qvec, dtype=float),
                    }
                )
    return rows



def _unique_nonnegative_norms(values: Sequence[float], *, tol: float) -> list[float]:
    norms: list[float] = []
    for value in sorted(float(item) for item in values):
        if value < -tol:
            continue
        candidate = 0.0 if abs(value) <= tol else value
        if not norms or abs(candidate - norms[-1]) > tol * max(1.0, abs(candidate), abs(norms[-1])):
            norms.append(float(candidate))
    return norms



def _shell_subspace_band_count(dim: int, *, fraction: float = 0.5) -> int:
    if dim <= 0:
        raise ValueError("Q-shell subspace requires positive shell dimension")
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"Q-shell subspace fraction must be in (0, 1], got {fraction}")
    return max(1, min(int(dim), int(np.floor(float(dim) * float(fraction)))))



def _q_shell_row_indices(
    qset1: np.ndarray,
    qset2: np.ndarray,
    n_orb: tuple[int, int],
    *,
    max_shells: int | None = None,
    subspace_fraction: float = 0.5,
    tol: float = 1.0e-6,
) -> list[dict[str, Any]]:
    rows = _model_row_metadata_for_harmonic_scan(qset1, qset2, n_orb)
    if not rows:
        return []
    norms = _unique_nonnegative_norms([float(np.linalg.norm(np.asarray(row["q"], dtype=float))) for row in rows], tol=tol)
    if max_shells is not None:
        if int(max_shells) <= 0:
            raise ValueError(f"max_shells must be positive when provided, got {max_shells}")
        norms = norms[: int(max_shells)]
    shells: list[dict[str, Any]] = []
    row_norms = [float(np.linalg.norm(np.asarray(row["q"], dtype=float))) for row in rows]
    for shell_index, norm_bound in enumerate(norms):
        selected = [idx for idx, norm in enumerate(row_norms) if norm <= float(norm_bound) + tol]
        dim = int(len(selected))
        shells.append(
            {
                "shell_index": int(shell_index),
                "q_norm_max": float(norm_bound),
                "rows": selected,
                "dimension": dim,
                "subspace_bands": _shell_subspace_band_count(dim, fraction=subspace_fraction),
            }
        )
    return shells



def _shell_subspace_overlap_report(
    model_h: np.ndarray,
    target_h: np.ndarray,
    shells: Sequence[Mapping[str, Any]],
    *,
    target_bands: str,
    subspace_fraction: float = 0.5,
    window_config: Mapping[str, Any] | None = None,
    align: str | None = None,
) -> dict[str, Any]:
    from . import fit_sampling
    from . import refinement

    model_arr = np.asarray(model_h, dtype=np.complex128)
    target_arr = np.asarray(target_h, dtype=np.complex128)
    if model_arr.shape != target_arr.shape or model_arr.ndim != 3:
        raise ValueError(f"shell subspace report expects matching (Nk,dim,dim) arrays, got {model_arr.shape}, {target_arr.shape}")
    target_key = str(target_bands).strip().lower()
    align_key = str(align if align is not None else target_key).strip().lower()
    shell_reports: list[dict[str, Any]] = []
    for shell in shells:
        rows = np.asarray(shell.get("rows", []), dtype=int)
        if rows.size == 0:
            continue
        model_shell = model_arr[:, rows[:, None], rows]
        target_shell = target_arr[:, rows[:, None], rows]
        target_eig, target_vec = np.linalg.eigh(target_shell)
        model_eig, model_vec = np.linalg.eigh(model_shell)
        shell_dim = int(rows.size)
        effective_window = dict(window_config or {})
        if "center_fraction" not in effective_window and "subspace_fraction" not in effective_window:
            effective_window["center_fraction"] = float(subspace_fraction)
        if "mode" not in effective_window and shell.get("subspace_bands") is not None:
            effective_window["mode"] = "fixed_fraction"
            effective_window["center_fraction"] = float(shell.get("subspace_bands")) / float(shell_dim)
        window = fit_sampling._shell_band_window_from_target_eig(
            target_eig,
            target_bands=target_key,
            window_config=effective_window,
        )
        n_bands = int(window["n_bands"])
        band_slice = tuple(int(value) for value in window["band_slice"])
        band_metrics = refinement._band_refinement_metrics(
            model_shell,
            target_eig,
            target_shell,
            band_slice=(int(band_slice[0]), int(band_slice[1])),
            align=align_key,
        )
        subspace = refinement._subspace_overlap_metrics(
            model_vec[:, :, band_slice[0] : band_slice[1]],
            target_vec[:, :, band_slice[0] : band_slice[1]],
        )
        shell_reports.append(
            {
                "shell_index": int(shell.get("shell_index", len(shell_reports))),
                "q_norm_max": float(shell.get("q_norm_max", 0.0)),
                "dimension": shell_dim,
                "subspace_bands": n_bands,
                "band_slice": [int(band_slice[0]), int(band_slice[1])],
                "window": window,
                "band": band_metrics,
                "subspace": subspace,
            }
        )
    mean_overlap = (
        float(np.mean([float(item["subspace"]["mean_overlap"]) for item in shell_reports])) if shell_reports else None
    )
    max_leakage = (
        float(np.max([float(item["subspace"]["max_leakage"]) for item in shell_reports])) if shell_reports else None
    )
    return {
        "enabled": True,
        "mode": "q_shell_principal_angles",
        "target_bands": target_key,
        "subspace_fraction": float(subspace_fraction),
        "window": dict(window_config or {"mode": "fixed_fraction", "center_fraction": float(subspace_fraction)}),
        "shell_count": int(len(shell_reports)),
        "mean_overlap": mean_overlap,
        "max_leakage": max_leakage,
        "shells": shell_reports,
    }



def _unique_positive_norms(values: Sequence[float], *, tol: float) -> list[float]:
    norms: list[float] = []
    for value in sorted(float(item) for item in values if float(item) > tol):
        if not norms or abs(value - norms[-1]) > tol * max(1.0, abs(value), abs(norms[-1])):
            norms.append(value)
    return norms



def _harmonic_shell_norms_from_qsets(qset1: np.ndarray, qset2: np.ndarray, *, tol: float = 1.0e-6) -> dict[str, list[float]]:
    q1 = np.asarray(qset1, dtype=float)
    q2 = np.asarray(qset2, dtype=float)
    intra_norms = [
        float(np.linalg.norm(qset[i] - qset[j]))
        for qset in (q1, q2)
        for i in range(len(qset))
        for j in range(len(qset))
    ]
    inter_norms = [float(np.linalg.norm(qi - qj)) for qi in q1 for qj in q2]
    return {
        "intra": _unique_positive_norms(intra_norms, tol=tol),
        "inter": _unique_positive_norms(inter_norms, tol=tol),
    }



def _harmonic_shell_index(norm: float, shell_norms: Sequence[float], *, tol: float) -> int:
    value = float(norm)
    if value <= tol:
        return 0
    shells = [float(item) for item in shell_norms]
    if not shells:
        return 10**9
    return int(np.argmin([abs(value - shell) for shell in shells])) + 1



def _shell_index_matrix(norms: np.ndarray, shells: Sequence[float]) -> np.ndarray:
    values = np.asarray(norms, dtype=float)
    shell_values = np.asarray([float(item) for item in shells], dtype=float)
    if shell_values.size == 0:
        return np.full(values.shape, 10**9, dtype=np.int32)
    return (np.argmin(np.abs(values[..., None] - shell_values[None, None, :]), axis=-1) + 1).astype(np.int32)



def _harmonic_ablation_shell_maps(
    rows: Sequence[Mapping[str, Any]],
    shell_norms: Mapping[str, Sequence[float]],
    *,
    tol: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    q = np.asarray([np.asarray(row["q"], dtype=float) for row in rows], dtype=float)
    layers = np.asarray([int(row["layer"]) for row in rows], dtype=np.int16)
    delta = q[:, None, :] - q[None, :, :]
    norms = np.linalg.norm(delta, axis=-1)
    zero = norms <= float(tol)
    same_layer = layers[:, None] == layers[None, :]
    intra = np.logical_and(~zero, same_layer)
    inter = np.logical_and(~zero, ~same_layer)
    intra_shell = _shell_index_matrix(norms, shell_norms.get("intra", []))
    inter_shell = _shell_index_matrix(norms, shell_norms.get("inter", []))
    return {
        "zero": zero,
        "intra": intra,
        "inter": inter,
        "intra_shell": intra_shell,
        "inter_shell": inter_shell,
    }



def _harmonic_ablation_mask_from_shell_maps(
    shell_maps: Mapping[str, np.ndarray],
    *,
    intra_shells: int,
    inter_shells: int,
) -> tuple[np.ndarray, dict[str, int]]:
    zero = np.asarray(shell_maps["zero"], dtype=bool)
    intra = np.asarray(shell_maps["intra"], dtype=bool)
    inter = np.asarray(shell_maps["inter"], dtype=bool)
    intra_shell = np.asarray(shell_maps["intra_shell"], dtype=np.int32)
    inter_shell = np.asarray(shell_maps["inter_shell"], dtype=np.int32)
    intra_keep = np.logical_and(intra, intra_shell <= int(intra_shells))
    inter_keep = np.logical_and(inter, inter_shell <= int(inter_shells))
    mask = np.logical_or(zero, np.logical_or(intra_keep, inter_keep))
    intra_shell_values = intra_shell[intra]
    inter_shell_values = inter_shell[inter]
    stats = {
        "onsite_or_zero": int(np.count_nonzero(zero)),
        "intra_kept": int(np.count_nonzero(intra_keep)),
        "intra_dropped": int(np.count_nonzero(intra) - np.count_nonzero(intra_keep)),
        "inter_kept": int(np.count_nonzero(inter_keep)),
        "inter_dropped": int(np.count_nonzero(inter) - np.count_nonzero(inter_keep)),
        "max_intra_shell_seen": int(np.max(intra_shell_values)) if intra_shell_values.size else 0,
        "max_inter_shell_seen": int(np.max(inter_shell_values)) if inter_shell_values.size else 0,
    }
    return mask, stats



def _harmonic_ablation_mask(
    rows: Sequence[Mapping[str, Any]],
    shell_norms: Mapping[str, Sequence[float]],
    *,
    intra_shells: int,
    inter_shells: int,
    tol: float = 1.0e-6,
) -> tuple[np.ndarray, dict[str, int]]:
    return _harmonic_ablation_mask_from_shell_maps(
        _harmonic_ablation_shell_maps(rows, shell_norms, tol=tol),
        intra_shells=intra_shells,
        inter_shells=inter_shells,
    )



def _physical_harmonic_ablation_candidates(
    qset1: np.ndarray,
    qset2: np.ndarray,
    *,
    n_orb: tuple[int, int],
    tol: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Enumerate unique harmonic support masks available in the actual Q rows."""
    from . import validation


    q1 = np.asarray(qset1, dtype=float)
    q2 = np.asarray(qset2, dtype=float)
    rows = _model_row_metadata_for_harmonic_scan(q1, q2, n_orb)
    if not rows:
        raise ValueError("physical harmonic candidates require at least one active model row")
    shell_norms = _harmonic_shell_norms_from_qsets(q1, q2, tol=tol)
    shell_maps = _harmonic_ablation_shell_maps(rows, shell_norms, tol=tol)
    intra_shell = np.asarray(shell_maps["intra_shell"], dtype=np.int32)
    inter_shell = np.asarray(shell_maps["inter_shell"], dtype=np.int32)
    intra_support = np.asarray(shell_maps["intra"], dtype=bool)
    inter_support = np.asarray(shell_maps["inter"], dtype=bool)
    active_intra_shells = intra_shell[intra_support]
    active_inter_shells = inter_shell[inter_support]
    max_intra_positive = int(np.max(active_intra_shells)) if active_intra_shells.size else 0
    max_inter_positive = int(np.max(active_inter_shells)) if active_inter_shells.size else 0

    layers = np.asarray([int(row["layer"]) for row in rows], dtype=np.int16)
    cross_sector = layers[:, None] != layers[None, :]
    has_inter_zero = bool(np.any(np.logical_and(cross_sector, np.asarray(shell_maps["zero"], dtype=bool))))
    max_intra_count = max_intra_positive + 1 if max_intra_positive else 0
    max_inter_count = max_inter_positive + int(has_inter_zero)

    unique: dict[str, dict[str, Any]] = {}
    for intra_count in range(max_intra_count + 1):
        for inter_count in range(max_inter_count + 1):
            mask = validation._selected_harmonic_support_mask(
                q1,
                q2,
                n_orb=(int(n_orb[0]), int(n_orb[1])),
                current_counts={"intra": int(intra_count), "inter": int(inter_count)},
                tol=tol,
            )
            packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
            fingerprint = hashlib.sha256(
                np.asarray(mask.shape, dtype=np.int64).tobytes() + packed.tobytes()
            ).hexdigest()
            record = {
                "intra_shells": int(intra_count),
                "inter_shells": int(inter_count),
                "support_entries": int(np.count_nonzero(mask)),
                "mask_fingerprint": fingerprint,
                "mask": np.asarray(mask, dtype=bool),
            }
            previous = unique.get(fingerprint)
            key = (
                int(intra_count) + int(inter_count),
                max(int(intra_count), int(inter_count)),
                abs(int(intra_count) - int(inter_count)),
                int(intra_count),
                int(inter_count),
            )
            if previous is None:
                unique[fingerprint] = record
                continue
            previous_key = (
                int(previous["intra_shells"]) + int(previous["inter_shells"]),
                max(int(previous["intra_shells"]), int(previous["inter_shells"])),
                abs(int(previous["intra_shells"]) - int(previous["inter_shells"])),
                int(previous["intra_shells"]),
                int(previous["inter_shells"]),
            )
            if key < previous_key:
                unique[fingerprint] = record

    return sorted(
        unique.values(),
        key=lambda item: (
            int(item["support_entries"]),
            int(item["intra_shells"]) + int(item["inter_shells"]),
            int(item["intra_shells"]),
            int(item["inter_shells"]),
        ),
    )



def _evaluate_harmonic_ablation_candidate(
    heff: np.ndarray,
    mask: np.ndarray,
    *,
    target_bands: str,
    primary_bands: int,
    plot_bands: int,
    target_eig: np.ndarray | None = None,
    target_vec: np.ndarray | None = None,
) -> dict[str, Any]:
    from . import fit_sampling
    from . import refinement

    target = np.asarray(heff)
    candidate = target * np.asarray(mask, dtype=bool)[None, :, :]
    candidate = 0.5 * (candidate + np.swapaxes(candidate.conj(), -1, -2))
    if target_eig is None or target_vec is None:
        target_eig, target_vec = np.linalg.eigh(target)
    candidate_eig, candidate_vec = np.linalg.eigh(candidate)
    dim = int(target.shape[-1])
    primary_slice = tuple(fit_sampling._auto_low_energy_band_slice(dim, int(primary_bands), target_bands))
    plot_slice = tuple(fit_sampling._auto_low_energy_band_slice(dim, int(plot_bands), target_bands))
    primary_band = refinement._band_refinement_metrics(
        candidate,
        target_eig,
        target,
        band_slice=(int(primary_slice[0]), int(primary_slice[1])),
        align=target_bands,
        eigenvalues=candidate_eig,
    )
    plot_band = refinement._band_refinement_metrics(
        candidate,
        target_eig,
        target,
        band_slice=(int(plot_slice[0]), int(plot_slice[1])),
        align=target_bands,
        eigenvalues=candidate_eig,
    )
    target_basis = target_vec[:, :, primary_slice[0] : primary_slice[1]]
    candidate_basis = candidate_vec[:, :, primary_slice[0] : primary_slice[1]]
    subspace = refinement._subspace_overlap_metrics(candidate_basis, target_basis)
    _low_vec, low_matrix = refinement._low_subspace_matrix_residual(
        candidate - target,
        target_basis,
        weight=1.0,
        sigma_mev=1.0,
        normalize=False,
    )
    raw_delta = candidate - target
    return {
        "primary_rms_mev": float(primary_band["top_band_rms_mev"]),
        "primary_max_mev": float(primary_band["top_band_max_mev"]),
        "plot_rms_mev": float(plot_band["top_band_rms_mev"]),
        "plot_max_mev": float(plot_band["top_band_max_mev"]),
        "subspace_mean_overlap": float(subspace["mean_overlap"]),
        "subspace_max_leakage": float(subspace["max_leakage"]),
        "subspace_min_singular_value": float(subspace["min_singular_value"]),
        "low_matrix_rms_mev": float(low_matrix["rms_mev"] or 0.0),
        "low_matrix_max_mev": float(low_matrix["max_mev"] or 0.0),
        "raw_matrix_rms_mev": float(np.sqrt(np.mean(np.abs(raw_delta) ** 2)) * 1000.0),
        "raw_matrix_max_mev": float(np.max(np.abs(raw_delta)) * 1000.0),
    }



def _harmonic_ablation_candidate_is_accepted(candidate: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    if float(candidate["plot_rms_mev"]) > float(thresholds.get("plot_rms_mev", 1.0)):
        return False
    if float(candidate["plot_max_mev"]) > float(thresholds.get("plot_max_mev", 3.0)):
        return False
    if float(candidate["subspace_mean_overlap"]) < float(thresholds.get("min_overlap", 0.98)):
        return False
    if thresholds.get("max_leakage") is not None and float(candidate["subspace_max_leakage"]) > float(thresholds["max_leakage"]):
        return False
    if thresholds.get("low_matrix_rms_mev") is not None and float(candidate["low_matrix_rms_mev"]) > float(thresholds["low_matrix_rms_mev"]):
        return False
    return True



def _harmonic_selection_threshold_values(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    values = {
        "plot_rms_mev": 0.2,
        "plot_max_mev": 0.6,
        "min_overlap": 0.9995,
        "low_matrix_rms_mev": 0.1,
    }
    values.update(dict(raw or {}))
    return values



def _select_harmonic_ablation_candidate(candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    rows = [dict(item) for item in candidates]
    accepted = [item for item in rows if bool(item.get("accepted", False))]

    def support_key(item: Mapping[str, Any]) -> tuple[int, int, int, float, float, float]:
        intra = int(item["intra_shells"])
        inter = int(item["inter_shells"])
        return (
            intra + inter,
            max(intra, inter),
            abs(intra - inter),
            float(item["plot_rms_mev"]),
            float(item["low_matrix_rms_mev"]),
            float(item["plot_max_mev"]),
        )

    def quality_key(item: Mapping[str, Any]) -> tuple[float, float, float, int, int]:
        return (
            float(item["plot_rms_mev"]),
            float(item["low_matrix_rms_mev"]),
            float(item["plot_max_mev"]),
            int(item["intra_shells"]),
            int(item["inter_shells"]),
        )

    if accepted:
        best = min(accepted, key=quality_key)
        plot_rms_limit = float(best["plot_rms_mev"]) + 0.05
        plot_max_limit = float(best["plot_max_mev"]) + 0.35
        low_matrix_limit = float(best["low_matrix_rms_mev"]) + 0.05
        plateau = [
            item
            for item in accepted
            if float(item["plot_rms_mev"]) <= plot_rms_limit
            and float(item["plot_max_mev"]) <= plot_max_limit
            and float(item["low_matrix_rms_mev"]) <= low_matrix_limit
        ]
        if plateau:
            selected = min(plateau, key=support_key)
            status = "accepted_quality_plateau_candidate"
        else:
            selected = best
            status = "accepted_best_quality_candidate"
        return dict(selected), status
    if not rows:
        raise ValueError("harmonic ablation selection requires at least one candidate")
    return dict(min(rows, key=quality_key)), "no_candidate_met_thresholds"



def _default_harmonic_candidate_pairs(max_shell: int) -> list[tuple[int, int]]:
    seeds = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 0),
        (0, 3),
        (3, 1),
        (1, 3),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
        (5, 5),
    ]
    out: list[tuple[int, int]] = []
    limit = int(max_shell)
    for intra, inter in seeds:
        if intra <= limit and inter <= limit and (intra, inter) not in out:
            out.append((int(intra), int(inter)))
    if not out:
        out.append((0, 0))
    return out



def _harmonic_candidate_pairs_from_config(raw: Any, *, max_shell: int, search: str) -> list[tuple[int, int]] | None:
    if raw is None:
        if str(search).strip().lower() == "grid":
            return None
        return _default_harmonic_candidate_pairs(max_shell)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("fit.harmonic_selection.candidate_pairs must be a list of [intra, inter] pairs")
    pairs: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError("fit.harmonic_selection.candidate_pairs entries must be [intra, inter]")
        intra, inter = int(item[0]), int(item[1])
        if intra < 0 or inter < 0:
            raise ValueError("fit.harmonic_selection.candidate_pairs entries must be non-negative")
        if intra > int(max_shell) or inter > int(max_shell):
            continue
        pair = (intra, inter)
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("fit.harmonic_selection.candidate_pairs did not contain any pair within max_shell")
    return pairs



def _run_harmonic_ablation_selection(
    heff: np.ndarray,
    qset1: np.ndarray,
    qset2: np.ndarray,
    *,
    n_orb: tuple[int, int],
    target_bands: str,
    primary_bands: int,
    plot_bands: int,
    max_shell: int | None = None,
    thresholds: Mapping[str, Any] | None = None,
    candidate_pairs: Sequence[tuple[int, int]] | None = None,
    tol: float = 1.0e-6,
) -> dict[str, Any]:
    from . import validation

    target = np.asarray(heff)
    if target.ndim != 3 or target.shape[-1] != target.shape[-2]:
        raise ValueError(f"harmonic ablation expects heff shape (Nk,dim,dim), got {target.shape}")
    if target_bands not in {"top", "bottom"}:
        raise ValueError("target_bands must be 'top' or 'bottom'")
    rows = _model_row_metadata_for_harmonic_scan(qset1, qset2, n_orb)
    if len(rows) != int(target.shape[-1]):
        raise ValueError(f"harmonic ablation row count {len(rows)} does not match heff dimension {target.shape[-1]}")
    shell_norms = _harmonic_shell_norms_from_qsets(qset1, qset2, tol=tol)
    threshold_values = dict(thresholds or {})
    records: list[dict[str, Any]] = []
    full_norm = float(np.linalg.norm(target))
    target_eig, target_vec = np.linalg.eigh(target)
    generation_started = time.perf_counter()
    physical_candidates: list[dict[str, Any]] | None = None
    if candidate_pairs is None and max_shell is None:
        physical_candidates = _physical_harmonic_ablation_candidates(
            qset1,
            qset2,
            n_orb=n_orb,
            tol=tol,
        )
        pair_list = [
            (int(item["intra_shells"]), int(item["inter_shells"]))
            for item in physical_candidates
        ]
        max_intra_count = max((pair[0] for pair in pair_list), default=0)
        max_inter_count = max((pair[1] for pair in pair_list), default=0)
        candidate_generation = {
            "mode": "physical_q_support",
            "max_intra_count": int(max_intra_count),
            "max_inter_count": int(max_inter_count),
            "raw_pair_count": int((max_intra_count + 1) * (max_inter_count + 1)),
            "unique_mask_count": int(len(physical_candidates)),
            "duplicate_mask_count": int(
                (max_intra_count + 1) * (max_inter_count + 1) - len(physical_candidates)
            ),
        }
    else:
        limit = int(max_shell if max_shell is not None else 0)
        pair_list = (
            [(int(intra), int(inter)) for intra, inter in candidate_pairs]
            if candidate_pairs is not None
            else [
                (int(intra_shells), int(inter_shells))
                for intra_shells in range(limit + 1)
                for inter_shells in range(limit + 1)
            ]
        )
        candidate_generation = {
            "mode": "configured_pairs" if candidate_pairs is not None else "configured_grid",
            "max_intra_count": max((pair[0] for pair in pair_list), default=0),
            "max_inter_count": max((pair[1] for pair in pair_list), default=0),
            "raw_pair_count": int(len(pair_list)),
            "unique_mask_count": int(len(pair_list)),
            "duplicate_mask_count": 0,
        }
    candidate_generation["elapsed_seconds"] = float(time.perf_counter() - generation_started)

    for candidate_index, (intra_shells, inter_shells) in enumerate(pair_list):
        if physical_candidates is None:
            mask = validation._selected_harmonic_support_mask(
                qset1,
                qset2,
                n_orb=n_orb,
                current_counts={"intra": int(intra_shells), "inter": int(inter_shells)},
                tol=tol,
            )
        else:
            mask = np.asarray(physical_candidates[candidate_index]["mask"], dtype=bool)
        metrics = _evaluate_harmonic_ablation_candidate(
            target,
            mask,
            target_bands=target_bands,
            primary_bands=int(primary_bands),
            plot_bands=int(plot_bands),
            target_eig=target_eig,
            target_vec=target_vec,
        )
        kept_norm = float(np.linalg.norm(target * mask[None, :, :]))
        candidate = {
            "intra_shells": int(intra_shells),
            "inter_shells": int(inter_shells),
            "complexity": int(intra_shells) + int(inter_shells),
            "kept_matrix_fraction": float(np.count_nonzero(mask) / mask.size),
            "kept_frobenius_fraction": float(kept_norm / full_norm) if full_norm > 0 else None,
            "support_entries": int(np.count_nonzero(mask)),
            "total_entries": int(mask.size),
            **metrics,
        }
        candidate["accepted"] = _harmonic_ablation_candidate_is_accepted(candidate, threshold_values)
        records.append(candidate)
    selected, status = _select_harmonic_ablation_candidate(records)
    return {
        "enabled": True,
        "method": "heff_harmonic_ablation",
        "target_bands": target_bands,
        "primary_bands": int(primary_bands),
        "plot_bands": int(plot_bands),
        "max_shell": None if max_shell is None else int(max_shell),
        "candidate_pairs": [[int(intra), int(inter)] for intra, inter in pair_list],
        "candidate_generation": candidate_generation,
        "thresholds": threshold_values,
        "shell_norms": {key: [float(item) for item in value] for key, value in shell_norms.items()},
        "selection_status": status,
        "selected": selected,
        "candidates": records,
    }



def _evenly_spaced_sample_indices(n_rows: int, count: int | None) -> list[int] | None:
    if count is None or int(count) <= 0 or int(count) >= int(n_rows):
        return None
    requested = int(count)
    if requested == 1:
        return [0]
    selected = sorted({int(round(i * (int(n_rows) - 1) / (requested - 1))) for i in range(requested)})
    if len(selected) < requested:
        for idx in range(int(n_rows)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) == requested:
                break
        selected.sort()
    return [int(idx) for idx in selected]



def _harmonic_selection_options(fit: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    raw = fit.get("harmonic_selection")
    if raw is None:
        return False, {}
    if raw is True:
        return True, {}
    if raw is False:
        return False, {}
    if not isinstance(raw, Mapping):
        raise ValueError("fit.harmonic_selection must be a mapping or boolean when provided")
    options = dict(raw)
    return bool(options.get("enabled", False)), options



def _harmonic_recommendation_options(
    fit: Mapping[str, Any],
    *,
    inherit_from: Mapping[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    raw = fit.get("harmonic_recommendation", False)
    if raw is False:
        return False, {}
    if raw in (True, None):
        options: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        options = dict(raw)
    else:
        raise ValueError("fit.harmonic_recommendation must be a mapping or boolean when provided")
    for key, value in dict(inherit_from or {}).items():
        options.setdefault(key, value)
    return bool(options.get("enabled", True)), options



def _format_harmonic_recommendation_candidate(candidate: Mapping[str, Any]) -> str:
    return (
        f"intra={int(candidate['intra_shells'])} inter={int(candidate['inter_shells'])} | "
        f"plot RMS={float(candidate['plot_rms_mev']):.3f} meV "
        f"Max={float(candidate['plot_max_mev']):.3f} meV | "
        f"primary RMS={float(candidate['primary_rms_mev']):.3f} meV "
        f"Max={float(candidate['primary_max_mev']):.3f} meV | "
        f"mean subspace overlap={float(candidate['subspace_mean_overlap']):.6f}"
    )



def _format_harmonic_recommendation_metrics(candidate: Mapping[str, Any]) -> str:
    return (
        f"plot {float(candidate['plot_rms_mev']):.3f}/{float(candidate['plot_max_mev']):.3f} meV | "
        f"primary {float(candidate['primary_rms_mev']):.3f}/{float(candidate['primary_max_mev']):.3f} meV | "
        f"mean subspace overlap {float(candidate['subspace_mean_overlap']):.6f}"
    )



def _format_harmonic_recommendation_support(candidate: Mapping[str, Any]) -> str:
    return f"intra={int(candidate['intra_shells'])} inter={int(candidate['inter_shells'])}"



def _harmonic_recommendation_quality_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, int, int]:
    return (
        float(candidate["plot_rms_mev"]),
        float(candidate["low_matrix_rms_mev"]),
        float(candidate["plot_max_mev"]),
        int(candidate["intra_shells"]),
        int(candidate["inter_shells"]),
    )



def _harmonic_recommendation_support_key(candidate: Mapping[str, Any]) -> tuple[int, int, int, float, float]:
    intra = int(candidate["intra_shells"])
    inter = int(candidate["inter_shells"])
    return (
        intra + inter,
        max(intra, inter),
        abs(intra - inter),
        float(candidate["plot_rms_mev"]),
        float(candidate["plot_max_mev"]),
    )



def _low_cost_harmonic_recommendation_candidate(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = [item for item in report.get("candidates", []) if isinstance(item, Mapping)]
    if not candidates:
        return None
    relaxed = [
        item
        for item in candidates
        if float(item["plot_rms_mev"]) <= 1.0
        and float(item["subspace_mean_overlap"]) > 0.9
    ]
    if not relaxed:
        return None
    return min(relaxed, key=_harmonic_recommendation_support_key)



def _high_accuracy_harmonic_recommendation_candidate(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = [item for item in report.get("candidates", []) if isinstance(item, Mapping)]
    if not candidates:
        return None
    high_accuracy = [
        item
        for item in candidates
        if float(item["plot_rms_mev"]) < 1.0
        and float(item["plot_max_mev"]) < 3.0
        and float(item["subspace_mean_overlap"]) > 0.95
    ]
    if high_accuracy:
        return min(high_accuracy, key=_harmonic_recommendation_support_key)
    selected = report.get("selected")
    if isinstance(selected, Mapping):
        return selected
    return min(candidates, key=_harmonic_recommendation_quality_key)



def _harmonic_recommendation_named_candidates(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    high_accuracy = _high_accuracy_harmonic_recommendation_candidate(report)
    if high_accuracy is not None:
        out["high"] = high_accuracy
    low_cost = _low_cost_harmonic_recommendation_candidate(report)
    if low_cost is not None:
        out["low"] = low_cost
    return out



def _harmonic_candidate_for_counts(report: Mapping[str, Any], counts: Mapping[str, int]) -> Mapping[str, Any] | None:
    pair = (int(counts.get("intra", 0)), int(counts.get("inter", 0)))
    for candidate in report.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        candidate_pair = (int(candidate.get("intra_shells", -1)), int(candidate.get("inter_shells", -1)))
        if candidate_pair == pair:
            return candidate
    return None



def _harmonic_recommendation_plot_candidates(
    report: Mapping[str, Any],
    *,
    current_counts: Mapping[str, int],
) -> list[tuple[str, Mapping[str, Any]]]:
    named = _harmonic_recommendation_named_candidates(report)
    rows: list[tuple[str, Mapping[str, Any]]] = []
    current = _harmonic_candidate_for_counts(report, current_counts)
    if current is not None:
        rows.append(("current", current))
    low = named.get("low")
    if low is not None:
        rows.append(("low", low))
    high = named.get("high")
    if high is not None:
        rows.append(("high", high))
    return rows



def _terminal_color_enabled(stream: Any | None = None) -> bool:
    out = stream if stream is not None else sys.stdout
    return bool(getattr(out, "isatty", lambda: False)()) and os.environ.get("NO_COLOR") is None



def _terminal_color(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"



def _resolved_harmonic_count_limits(harmonics: Mapping[str, Any]) -> dict[str, int]:
    limits = configuration._harmonic_count_limits(harmonics)
    for kind in ("intra", "inter"):
        if kind in limits:
            continue
        raw = harmonics.get(kind)
        if isinstance(raw, (int, np.integer)):
            limits[kind] = int(raw)
        elif isinstance(raw, Mapping) and "count" in raw:
            limits[kind] = int(raw["count"])
    return limits



def _print_harmonic_recommendation_report(
    report: Mapping[str, Any],
    *,
    current_counts: Mapping[str, int],
    plot_path: Path | None = None,
    color: bool | None = None,
) -> None:
    named = _harmonic_recommendation_named_candidates(report)
    selected = named.get("high")
    if selected is None:
        return
    use_color = _terminal_color_enabled() if color is None else bool(color)
    prefix = _terminal_color("[kp model]", "36", enabled=use_color)
    current_pair = (int(current_counts.get("intra", 0)), int(current_counts.get("inter", 0)))
    current = None
    for candidate in report.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        pair = (int(candidate.get("intra_shells", -1)), int(candidate.get("inter_shells", -1)))
        if pair == current_pair:
            current = candidate
            break
    low_cost = named.get("low")
    print(
        f"{prefix} "
        + _terminal_color("--- Harmonic recommendation (diagnostic only) ---", "1;36", enabled=use_color),
        flush=True,
    )
    current_label = _terminal_color("actual model.harmonics", "2", enabled=use_color)
    if current is not None:
        line = (
            f"  {current_label}: "
            + _format_harmonic_recommendation_support(current)
            + " (unchanged) | "
            + _format_harmonic_recommendation_metrics(current)
        )
        print(f"{prefix} " + _terminal_color(line, "2", enabled=use_color), flush=True)
    else:
        line = (
            f"  {current_label}: "
            f"intra={current_pair[0]} inter={current_pair[1]} | not scanned",
            " (unchanged)",
        )
        print(f"{prefix} " + _terminal_color("".join(line), "2", enabled=use_color), flush=True)
    if low_cost is not None:
        label = _terminal_color("low-cost", "1;33", enabled=use_color)
        line = (
            f"  {label}: "
            + _format_harmonic_recommendation_support(low_cost)
            + " | "
            + _format_harmonic_recommendation_metrics(low_cost)
            + " | relaxed accuracy"
        )
        print(f"{prefix} " + _terminal_color(line, "33", enabled=use_color), flush=True)
    label = _terminal_color("high-acc", "1;32", enabled=use_color)
    line = (
        f"  {label}: "
        + _format_harmonic_recommendation_support(selected)
        + " | "
        + _format_harmonic_recommendation_metrics(selected)
        + " | balanced high accuracy"
    )
    print(f"{prefix} " + _terminal_color(line, "32", enabled=use_color), flush=True)
    if plot_path is not None:
        print(
            f"{prefix} "
            + _terminal_color(f"  band plot: {plot_path.resolve()}", "35", enabled=use_color),
            flush=True,
        )



def _print_automatic_harmonic_selection_report(
    report: Mapping[str, Any],
    *,
    reporter: KpReporter | None = None,
) -> None:
    selected = report.get("selected")
    if not isinstance(selected, Mapping):
        return
    output = reporter if reporter is not None else KpReporter("kp model")

    def candidate_row(label: str, candidate: Mapping[str, Any]) -> tuple[object, ...]:
        return (
            label,
            int(candidate["intra_shells"]),
            int(candidate["inter_shells"]),
            f"{float(candidate['plot_rms_mev']):.3f} / {float(candidate['plot_max_mev']):.3f}",
            f"{float(candidate['primary_rms_mev']):.3f} / {float(candidate['primary_max_mev']):.3f}",
            f"{float(candidate['subspace_mean_overlap']):.6f}",
        )

    output.status("Automatic harmonic selection", "PASS")
    rows = [candidate_row("selected", selected)]
    quality = report.get("quality_selected")
    if isinstance(quality, Mapping):
        rows.append(candidate_row("high", quality))
    output.table(
        ("Choice", "Intra", "Inter", "Plot RMS/Max (meV)", "Primary RMS/Max (meV)", "Overlap"),
        rows,
        right_align=(1, 2, 5),
        column_styles={0: "1"},
    )
    quality_label = "low-cost choice"
    if isinstance(quality, Mapping):
        quality_label += " · high-accuracy reference"
    output.fields([("quality", quality_label)])



def _aligned_harmonic_plot_bands(eig: np.ndarray, target_slice: np.ndarray, *, band_slice: tuple[int, int], target_bands: str) -> np.ndarray:
    selected = np.asarray(eig, dtype=float)
    if target_bands == "top":
        selected = selected + (float(target_slice[0, -1]) - float(selected[0, -1]))
        return selected - float(target_slice[0, -1])
    if target_bands == "bottom":
        selected = selected + (float(target_slice[0, 0]) - float(selected[0, 0]))
        return selected - float(target_slice[0, 0])
    return selected



def _eigvalsh_harmonic_plot_window(hamiltonians: np.ndarray, band_slice: tuple[int, int]) -> np.ndarray:
    start, stop = int(band_slice[0]), int(band_slice[1])
    return np.asarray(
        [
            scipy.linalg.eigh(
                np.asarray(matrix, dtype=np.complex128),
                eigvals_only=True,
                subset_by_index=[start, stop - 1],
                driver="evr",
            )
            for matrix in np.asarray(hamiltonians)
        ],
        dtype=float,
    )



def _save_harmonic_recommendation_band_plot(
    *,
    heff: np.ndarray,
    qset1: np.ndarray,
    qset2: np.ndarray,
    n_orb: tuple[int, int],
    target_bands: str,
    plot_bands: int,
    report: Mapping[str, Any],
    current_counts: Mapping[str, int],
    path: Path,
    tol: float,
) -> Path | None:
    from . import fit_sampling
    from . import validation

    plot_candidates = _harmonic_recommendation_plot_candidates(report, current_counts=current_counts)
    if len(plot_candidates) < 1:
        return None
    target = np.asarray(heff, dtype=np.complex128)
    if target.ndim != 3 or target.shape[-1] != target.shape[-2]:
        raise ValueError(f"harmonic recommendation band plot expects Heff shape (Nk,dim,dim), got {target.shape}")
    dim = int(target.shape[-1])
    band_slice = tuple(fit_sampling._auto_low_energy_band_slice(dim, int(plot_bands), target_bands))
    target_slice = _eigvalsh_harmonic_plot_window(target, band_slice)
    if target_bands == "top":
        target_plot = target_slice - float(target_slice[0, -1])
        ylabel = relative_energy_ylabel("top")
    elif target_bands == "bottom":
        target_plot = target_slice - float(target_slice[0, 0])
        ylabel = relative_energy_ylabel("bottom")
    else:
        target_plot = target_slice
        ylabel = "Energy (eV)"

    def candidate_plot(candidate: Mapping[str, Any]) -> np.ndarray:
        mask = validation._selected_harmonic_support_mask(
            qset1,
            qset2,
            n_orb=n_orb,
            current_counts={
                "intra": int(candidate["intra_shells"]),
                "inter": int(candidate["inter_shells"]),
            },
            tol=tol,
        )
        masked = target * mask[None, :, :]
        masked = 0.5 * (masked + np.swapaxes(masked.conj(), -1, -2))
        eig = _eigvalsh_harmonic_plot_window(masked, band_slice)
        return _aligned_harmonic_plot_bands(eig, target_slice, band_slice=band_slice, target_bands=target_bands)

    candidate_bands = [(name, candidate, candidate_plot(candidate)) for name, candidate in plot_candidates]

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(target_plot.shape[0], dtype=float)
    styles = {
        "current": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.15, "label": "current Heff support mask"},
        "low": {"color": "#E69F00", "linestyle": "--", "linewidth": 1.05, "label": "low-cost recommendation"},
        "high": {"color": "#009E73", "linestyle": "-.", "linewidth": 1.05, "label": "high-accuracy recommendation"},
    }
    legend_kwargs = dict(KP_LEGEND_KWARGS)
    legend_kwargs.update(
        {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.18),
            "ncol": 2,
            "frameon": False,
        }
    )
    with kp_plot_rc_context():
        fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=KP_DPI)
        for ib in range(target_plot.shape[1]):
            ax.plot(x, target_plot[:, ib], color="black", linewidth=1.2, alpha=0.85)
            for name, _candidate, bands in candidate_bands:
                style = styles.get(name, {"color": "0.4", "linestyle": ":", "linewidth": 1.0, "label": name})
                ax.plot(
                    x,
                    bands[:, ib],
                    color=str(style["color"]),
                    linewidth=float(style["linewidth"]),
                    linestyle=str(style["linestyle"]),
                    alpha=0.86,
                )
        ax.set_xlabel("k-point index")
        ax.set_ylabel(ylabel)
        ax.set_title("Harmonic support comparison")
        ax.grid(True, axis="y", color="0.88", linewidth=0.65)
        ax.axhline(0.0, color="0.88", linewidth=0.7, zorder=0)
        ax.legend(
            handles=[
                Line2D([0], [0], color="black", linewidth=1.2, label="original Heff"),
                *[
                    Line2D(
                        [0],
                        [0],
                        color=str(styles.get(name, {}).get("color", "0.4")),
                        linewidth=float(styles.get(name, {}).get("linewidth", 1.0)),
                        linestyle=str(styles.get(name, {}).get("linestyle", ":")),
                        label=str(styles.get(name, {}).get("label", name)),
                    )
                    for name, _candidate, _bands in candidate_bands
                ],
            ],
            **legend_kwargs,
        )
        apply_kp_axis_style(ax, box_aspect=KP_BAND_BOX_ASPECT, font_family=kp_font_family())
        fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
        fig.savefig(out, dpi=KP_DPI, bbox_inches="tight")
        plt.close(fig)
    return out



def _run_and_print_harmonic_recommendation(
    *,
    heff_file: Path,
    qset1_file: Path,
    qset2_file: Path,
    output_dir: Path,
    rotation_deg: float,
    band_indices: Sequence[int] | None,
    n_orb_values: Sequence[int],
    target_bands: str,
    fit: Mapping[str, Any],
    current_counts: Mapping[str, int],
    options: Mapping[str, Any],
) -> None:
    thresholds_raw = options.get("thresholds", {})
    if thresholds_raw is None:
        thresholds_raw = {}
    if not isinstance(thresholds_raw, Mapping):
        raise ValueError("fit.harmonic_recommendation.thresholds must be a mapping when provided")
    threshold_values = _harmonic_selection_threshold_values(thresholds_raw)
    Q_set1_for_selection, Q_set2_for_selection = load_Q_sets_from_gvec_files(
        qset1_file,
        qset2_file,
        rotation_deg=rotation_deg,
    )
    heff_for_selection = configuration._select_rows(np.load(heff_file, mmap_mode="r"), band_indices)
    sample_indices = _evenly_spaced_sample_indices(
        int(heff_for_selection.shape[0]),
        int(options.get("sample_kpoints", 7)),
    )
    if sample_indices is None:
        heff_scan = np.asarray(heff_for_selection)
    else:
        heff_scan = np.asarray(heff_for_selection)[np.asarray(sample_indices, dtype=int)]
    n_primary_for_selection = int(sum(int(value) for value in n_orb_values))
    dim_for_selection = int(heff_for_selection.shape[-1])
    plot_bands_for_selection = int(
        options.get(
            "plot_bands",
            fit.get("weighted_fit_bands", min(dim_for_selection, max(n_primary_for_selection, 10))),
        )
    )
    max_shell_for_selection = max(
        int(options.get("max_shell", 5)),
        int(current_counts.get("intra", 0)),
        int(current_counts.get("inter", 0)),
    )
    candidate_pairs_for_selection = _harmonic_candidate_pairs_from_config(
        options.get("candidate_pairs"),
        max_shell=max_shell_for_selection,
        search=str(options.get("search", "ladder")),
    )
    current_pair = (int(current_counts.get("intra", 0)), int(current_counts.get("inter", 0)))
    if candidate_pairs_for_selection is not None and current_pair not in candidate_pairs_for_selection:
        candidate_pairs_for_selection = [*candidate_pairs_for_selection, current_pair]
    report = _run_harmonic_ablation_selection(
        heff_scan,
        Q_set1_for_selection,
        Q_set2_for_selection,
        n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
        target_bands=target_bands,
        primary_bands=n_primary_for_selection,
        plot_bands=plot_bands_for_selection,
        max_shell=max_shell_for_selection,
        thresholds=threshold_values,
        candidate_pairs=candidate_pairs_for_selection,
        tol=float(options.get("tol", 1.0e-6)),
    )
    report["sample_indices"] = sample_indices
    plot_path = _save_harmonic_recommendation_band_plot(
        heff=np.asarray(heff_for_selection),
        qset1=Q_set1_for_selection,
        qset2=Q_set2_for_selection,
        n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
        target_bands=target_bands,
        plot_bands=plot_bands_for_selection,
        report=report,
        current_counts=current_counts,
        path=Path(output_dir) / "harmonic_recommendation_bands.png",
        tol=float(options.get("tol", 1.0e-6)),
    )
    _print_harmonic_recommendation_report(report, current_counts=current_counts, plot_path=plot_path)



def _run_model_harmonic_recommendation_once(
    *,
    model_config: configuration.ConfiguredModel,
    output_dir: Path,
    reporter: KpReporter | None = None,
) -> None:
    from . import candidates

    fit = model_config.raw.get("fit", {})
    if not isinstance(fit, Mapping):
        return
    automatic_report = model_config.automatic_harmonic_selection
    if isinstance(automatic_report, Mapping) and automatic_report:
        _print_automatic_harmonic_selection_report(automatic_report, reporter=reporter)
        return
    _, harmonic_selection_cfg = _harmonic_selection_options(fit)
    recommendation_enabled, recommendation_cfg = _harmonic_recommendation_options(
        fit,
        inherit_from=harmonic_selection_cfg,
    )
    if not recommendation_enabled:
        return
    _run_and_print_harmonic_recommendation(
        heff_file=model_config.heff_file,
        qset1_file=model_config.qset1_file,
        qset2_file=model_config.qset2_file,
        output_dir=output_dir,
        rotation_deg=model_config.rotation_deg,
        band_indices=model_config.band_indices,
        n_orb_values=model_config.n_orb,
        target_bands=str(model_config.raw.get("model", {}).get("target_bands", "top")).strip().lower(),
        fit=fit,
        current_counts=_resolved_harmonic_count_limits(model_config.harmonics_config),
        options=recommendation_cfg,
    )
