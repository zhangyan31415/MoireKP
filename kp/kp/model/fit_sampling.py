from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from . import configuration


def _auto_low_energy_band_slice(dim: int, n_bands: int, target_bands: str) -> list[int]:
    if n_bands <= 0:
        raise ValueError(f"n_bands must be positive, got {n_bands}")
    if n_bands > dim:
        raise ValueError(f"n_bands={n_bands} exceeds Hamiltonian dimension {dim}")
    target = str(target_bands or "top").strip().lower()
    if target == "top":
        return [int(dim - n_bands), int(dim)]
    if target == "bottom":
        return [0, int(n_bands)]
    raise ValueError("model.target_bands must be 'top' or 'bottom'")



def _auto_low_energy_window_record(
    eigvals: np.ndarray,
    *,
    n_bands: int,
    target_bands: str,
    role: str,
    gap_tolerance_mev: float,
    use_for_loss: bool | None = None,
) -> dict[str, Any]:
    from . import refinement

    eig = np.asarray(eigvals, dtype=float)
    if eig.ndim != 2:
        raise ValueError(f"auto low-energy windows expect eigvals shape (Nk,dim), got {eig.shape}")
    band_slice = _auto_low_energy_band_slice(int(eig.shape[1]), int(n_bands), target_bands)
    gaps = refinement._boundary_gap_mev_by_k(eig, band_slice)
    finite = gaps[np.isfinite(gaps)]
    boundary_gap = float(np.min(finite)) if finite.size else None
    stable = boundary_gap is None or boundary_gap >= float(gap_tolerance_mev)
    return {
        "role": str(role),
        "target_bands": str(target_bands or "top").strip().lower(),
        "n_bands": int(n_bands),
        "band_slice": band_slice,
        "boundary_gap_mev": boundary_gap,
        "gap_tolerance_mev": float(gap_tolerance_mev),
        "state": "ok" if stable else "boundary_gap_small",
        "use_for_loss": bool(stable) if use_for_loss is None else bool(use_for_loss),
    }



def _auto_low_energy_windows(
    eigvals: np.ndarray,
    *,
    n_primary: int,
    target_bands: str,
    gap_tolerance_mev: float = 0.1,
    max_expanded: int = 2,
) -> dict[str, Any]:
    eig = np.asarray(eigvals, dtype=float)
    if eig.ndim != 2:
        raise ValueError(f"auto low-energy windows expect eigvals shape (Nk,dim), got {eig.shape}")
    dim = int(eig.shape[1])

    primary = _auto_low_energy_window_record(
        eig,
        n_bands=int(n_primary),
        target_bands=target_bands,
        role="primary",
        gap_tolerance_mev=gap_tolerance_mev,
    )
    expanded: list[dict[str, Any]] = []
    for n_bands in range(int(n_primary) + 1, min(dim, int(n_primary) + int(max_expanded)) + 1):
        expanded.append(
            _auto_low_energy_window_record(
                eig,
                n_bands=n_bands,
                target_bands=target_bands,
                role="expanded",
                gap_tolerance_mev=gap_tolerance_mev,
            )
        )
    return {"primary": primary, "expanded": expanded}



def _shell_band_window_from_target_eig(
    eigvals: np.ndarray,
    *,
    target_bands: str,
    window_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from . import harmonic_selection
    from . import refinement

    eig = np.asarray(eigvals, dtype=float)
    if eig.ndim != 2:
        raise ValueError(f"shell band window expects eigvals shape (Nk,dim), got {eig.shape}")
    dim = int(eig.shape[1])
    if dim <= 0:
        raise ValueError("shell band window requires positive shell dimension")
    target = str(target_bands or "top").strip().lower()
    if target not in {"top", "bottom"}:
        raise ValueError("model.target_bands must be 'top' or 'bottom'")
    cfg = dict(window_config or {})
    mode = str(cfg.get("mode", "gap_aware")).strip().lower()
    center_fraction = float(cfg.get("center_fraction", cfg.get("subspace_fraction", 0.5)))
    if center_fraction <= 0.0 or center_fraction > 1.0:
        raise ValueError("shell projected matrix window center_fraction must be in (0, 1]")
    gap_tolerance_mev = float(cfg.get("gap_tolerance_mev", 0.1))
    if gap_tolerance_mev < 0.0:
        raise ValueError("shell projected matrix window gap_tolerance_mev must be non-negative")

    if mode in {"fraction", "fixed_fraction", "half"}:
        n_bands = harmonic_selection._shell_subspace_band_count(dim, fraction=center_fraction)
        band_slice = _auto_low_energy_band_slice(dim, n_bands, target)
        gap_by_k = refinement._boundary_gap_mev_by_k(eig, band_slice)
        finite = gap_by_k[np.isfinite(gap_by_k)]
        boundary_gap = float(np.min(finite)) if finite.size else None
        return {
            "mode": "fixed_fraction",
            "target_bands": target,
            "n_bands": int(n_bands),
            "band_slice": [int(band_slice[0]), int(band_slice[1])],
            "boundary_gap_mev": boundary_gap,
            "gap_tolerance_mev": gap_tolerance_mev,
            "window_quality": "ok"
            if boundary_gap is None or boundary_gap >= gap_tolerance_mev
            else "boundary_gap_small",
            "candidate_count": 1,
            "center_fraction": center_fraction,
        }
    if mode != "gap_aware":
        raise ValueError(f"Unsupported shell projected matrix window mode {mode!r}")

    raw_search = cfg.get("search_fraction", [0.35, 0.65])
    if not isinstance(raw_search, Sequence) or isinstance(raw_search, (str, bytes)) or len(raw_search) != 2:
        raise ValueError("shell projected matrix window search_fraction must be [lo, hi]")
    lo = float(raw_search[0])
    hi = float(raw_search[1])
    if lo <= 0.0 or hi <= 0.0 or lo > hi:
        raise ValueError("shell projected matrix window search_fraction must satisfy 0 < lo <= hi")
    lo_n = max(1, int(np.ceil(lo * dim)))
    hi_n = min(dim, int(np.floor(hi * dim)))
    if hi_n < lo_n:
        center_n = harmonic_selection._shell_subspace_band_count(dim, fraction=center_fraction)
        lo_n = hi_n = int(center_n)
    center_n_float = center_fraction * dim
    candidates: list[dict[str, Any]] = []
    for n_bands in range(int(lo_n), int(hi_n) + 1):
        band_slice = _auto_low_energy_band_slice(dim, int(n_bands), target)
        gap_by_k = refinement._boundary_gap_mev_by_k(eig, band_slice)
        finite = gap_by_k[np.isfinite(gap_by_k)]
        boundary_gap = float(np.min(finite)) if finite.size else None
        candidates.append(
            {
                "n_bands": int(n_bands),
                "band_slice": [int(band_slice[0]), int(band_slice[1])],
                "boundary_gap_mev": boundary_gap,
                "distance_to_center": float(abs(float(n_bands) - center_n_float)),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            -float(row["boundary_gap_mev"]) if row["boundary_gap_mev"] is not None else -np.inf,
            float(row["distance_to_center"]),
            int(row["n_bands"]),
        ),
    )
    boundary_gap = selected["boundary_gap_mev"]
    return {
        "mode": "gap_aware",
        "target_bands": target,
        "n_bands": int(selected["n_bands"]),
        "band_slice": [int(value) for value in selected["band_slice"]],
        "boundary_gap_mev": boundary_gap,
        "gap_tolerance_mev": gap_tolerance_mev,
        "window_quality": "ok" if boundary_gap is None or boundary_gap >= gap_tolerance_mev else "boundary_gap_small",
        "candidate_count": int(len(candidates)),
        "center_fraction": center_fraction,
        "search_fraction": [float(lo), float(hi)],
    }



def _select_adaptive_fit_indices(
    kpoints: np.ndarray,
    *,
    initial_points: int = 2,
    max_points: int = 15,
    initial_indices: Sequence[int] | None = None,
    residual_scores: np.ndarray | None = None,
) -> tuple[list[int], dict[str, Any]]:
    arr = configuration._validate_kpoints(kpoints)
    n_k = int(arr.shape[0])
    if n_k <= 0:
        raise ValueError("auto_low_energy fit selection requires at least one k-point")
    candidates = _auto_low_energy_fit_candidate_sets(
        arr,
        initial_points=initial_points,
        max_points=max_points,
        initial_indices=initial_indices,
        residual_scores=residual_scores,
    )
    if not candidates:
        raise ValueError("auto_low_energy fit selection did not produce any candidate k-point set")
    selected_list = [int(idx) for idx in candidates[0]["indices"]]
    metadata = {
        "mode": "auto_low_energy",
        "source": "candidate_scan_pending",
        "initial_points": int(initial_points),
        "initial_indices": [int(idx) for idx in initial_indices] if initial_indices is not None else None,
        "max_points": int(max_points),
        "candidate_count": int(n_k),
        "selected_indices": [int(idx) for idx in selected_list],
        "fit_candidate_sets": candidates,
        "residual_scores_used": residual_scores is not None,
    }
    return selected_list, metadata



def _auto_low_energy_refinement_indices(
    kpoints: np.ndarray,
    *,
    base_indices: Sequence[int],
    max_points: int = 8,
) -> list[int]:
    arr = configuration._validate_kpoints(kpoints)
    n_k = int(arr.shape[0])
    if n_k <= 0:
        raise ValueError("auto_low_energy refinement selection requires at least one k-point")
    limit = max(1, min(int(max_points), n_k))
    vertices = _kpath_turning_point_indices(arr)
    if n_k > 1 and (n_k - 1) not in vertices:
        vertices.append(n_k - 1)
    vertices = sorted({int(index) for index in vertices if 0 <= int(index) < n_k})

    candidates: list[tuple[int, int]] = []

    def add(priority: int, index: int) -> None:
        if 0 <= int(index) < n_k:
            candidates.append((int(priority), int(index)))

    for index in base_indices:
        add(0, int(index))
    for index in vertices:
        add(1, int(index))
    for left, right in zip(vertices, vertices[1:]):
        if int(right) > int(left):
            add(2, int(round((int(left) + int(right)) / 2.0)))
    if len(vertices) >= 2:
        left = int(vertices[0])
        right = int(vertices[1])
        if right > left + 1:
            near_edge_offset = max(1, int(round((right - left) * 0.1)))
            add(3, left + near_edge_offset)

    if n_k > 1:
        for index in _uniform_fit_indices(n_k, min(limit, max(2, limit))):
            add(4, int(index))

    selected: list[int] = []
    seen: set[int] = set()
    for _priority, index in sorted(candidates, key=lambda item: (item[0], item[1])):
        if index in seen:
            continue
        seen.add(index)
        selected.append(index)
        if len(selected) >= limit:
            break
    return sorted(selected)



def _kpath_turning_point_indices(kpoints: np.ndarray, *, atol: float = 1.0e-10) -> list[int]:
    arr = configuration._validate_kpoints(kpoints)
    n_k = int(arr.shape[0])
    if n_k <= 1:
        return [0]
    vertices = {0, n_k - 1}
    steps = np.diff(arr, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    unit = np.zeros_like(steps)
    nonzero = norms > atol
    unit[nonzero] = steps[nonzero] / norms[nonzero, None]
    for idx in range(1, n_k - 1):
        prev_candidates = np.where(nonzero[:idx])[0]
        next_candidates = np.where(nonzero[idx:])[0] + idx
        if prev_candidates.size == 0 or next_candidates.size == 0:
            continue
        prev_dir = unit[int(prev_candidates[-1])]
        next_dir = unit[int(next_candidates[0])]
        if np.linalg.norm(prev_dir - next_dir) > 1.0e-7:
            vertices.add(int(idx))
    return sorted(vertices)



def _dedupe_fit_indices(indices: Sequence[int], n_k: int, max_points: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in indices:
        idx = int(raw)
        if idx < 0 or idx >= int(n_k) or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        if len(out) >= int(max_points):
            break
    return out



def _uniform_fit_indices(n_k: int, count: int) -> list[int]:
    count = max(1, min(int(count), int(n_k)))
    if count == 1:
        return [0]
    return sorted({int(round(i * (int(n_k) - 1) / (count - 1))) for i in range(count)})



def _auto_low_energy_fit_candidate_sets(
    kpoints: np.ndarray,
    *,
    initial_points: int = 5,
    max_points: int = 7,
    initial_indices: Sequence[int] | None = None,
    residual_scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    arr = configuration._validate_kpoints(kpoints)
    n_k = int(arr.shape[0])
    max_count = max(1, min(int(max_points), n_k))
    initial = max(1, min(int(initial_points), max_count))
    vertices = _kpath_turning_point_indices(arr)
    internal_vertices = [idx for idx in vertices if idx not in {0, n_k - 1}]
    closes_path = n_k > 1 and bool(np.allclose(arr[0], arr[-1], atol=1.0e-10, rtol=0.0))
    vertex_ladder = [idx for idx in vertices if not (closes_path and idx == n_k - 1)]
    if not vertex_ladder:
        vertex_ladder = [0]

    if initial_indices is not None:
        minimal = _dedupe_fit_indices(initial_indices, n_k, max_count)
    elif internal_vertices:
        minimal = [0, int(internal_vertices[-1])]
    elif n_k > 1 and initial >= 2:
        minimal = [0, n_k - 1]
    else:
        minimal = [0]

    rows: list[tuple[str, list[int], str, int, bool]] = [
        ("minimal", minimal, "minimal low-energy fit anchors", max_count, False),
    ]
    if len(vertex_ladder) >= 3:
        rows.append(("junctions_3", vertex_ladder[:3], "minimal anchors plus first high-symmetry junction", max_count, False))
    if len(vertex_ladder) >= 4:
        rows.append(("junctions_4", vertex_ladder[:4], "all high-symmetry junctions", max_count, False))
    elif len(vertex_ladder) > 1:
        rows.append((f"junctions_{len(vertex_ladder)}", vertex_ladder, "available high-symmetry junctions", max_count, False))

    uniform_count = min(max_count, max(5, initial))
    if uniform_count > len(minimal):
        rows.append((f"uniform_{uniform_count}", _uniform_fit_indices(n_k, uniform_count), "uniform path anchors", max_count, False))

    if residual_scores is not None:
        scores = np.asarray(residual_scores, dtype=float)
        if scores.shape != (n_k,):
            raise ValueError(f"residual_scores must have shape ({n_k},), got {scores.shape}")
        residual_ranked = [int(idx) for idx in np.argsort(-scores, kind="mergesort")]
        residual_seed = vertex_ladder if len(vertex_ladder) > 1 else minimal
        for target_count in range(min(max_count, len(residual_seed) + 1), max_count + 1):
            rows.append(
                (
                    f"residual_augmented_{target_count}",
                    [*residual_seed, *residual_ranked],
                    f"high-symmetry anchors plus residual peaks up to {target_count} fit points",
                    target_count,
                    True,
                )
            )

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for name, raw_indices, reason, limit, allow_truncate in rows:
        full_indices = _dedupe_fit_indices(raw_indices, n_k, n_k)
        if not allow_truncate and len(full_indices) > int(limit):
            continue
        indices = _dedupe_fit_indices(raw_indices, n_k, int(limit))
        if not indices:
            continue
        key = tuple(indices)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "name": name,
                "indices": [int(idx) for idx in indices],
                "fit_count": int(len(indices)),
                "reason": reason,
            }
        )
    return candidates



def _choose_auto_fit_candidate_record(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not records:
        raise ValueError("auto_low_energy fit candidate selection requires at least one record")
    rows = [dict(row) for row in records]
    finite_rows = [
        row
        for row in rows
        if np.isfinite(float(row.get("plot_rms_mev", np.inf)))
        and np.isfinite(float(row.get("plot_max_mev", np.inf)))
        and np.isfinite(float(row.get("all_rms_mev", np.inf)))
        and np.isfinite(float(row.get("all_max_mev", np.inf)))
    ]
    if not finite_rows:
        selected = rows[0]
        return selected, {
            "selection_status": "selected_first_candidate_no_finite_metrics",
            "selected": selected,
            "rejected": rows[1:],
        }
    best_all_rms = min(float(row["all_rms_mev"]) for row in finite_rows)
    best_all_max = min(float(row["all_max_mev"]) for row in finite_rows)
    all_rms_limit = max(best_all_rms + 5.0, best_all_rms * 3.0)
    all_max_limit = max(best_all_max + 25.0, best_all_max * 5.0)
    sane: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        if row not in finite_rows:
            rejected.append({**row, "reason": "non_finite_metric"})
            continue
        if float(row["all_rms_mev"]) > all_rms_limit or float(row["all_max_mev"]) > all_max_limit:
            rejected.append({**row, "reason": "global_error_outlier"})
            continue
        sane.append(row)
    pool = sane if sane else finite_rows
    status = "selected_sane_low_energy_candidate" if sane else "selected_best_finite_candidate"
    best_plot_rms = min(float(row.get("plot_rms_mev", np.inf)) for row in pool)
    best_plot_max = min(float(row.get("plot_max_mev", np.inf)) for row in pool)
    best_pool_all_rms = min(float(row.get("all_rms_mev", np.inf)) for row in pool)
    best_pool_all_max = min(float(row.get("all_max_mev", np.inf)) for row in pool)
    near_best = [
        row
        for row in pool
        if float(row.get("plot_rms_mev", np.inf)) <= best_plot_rms + 0.05
        and float(row.get("plot_max_mev", np.inf)) <= best_plot_max + 0.25
        and float(row.get("all_rms_mev", np.inf)) <= best_pool_all_rms + 0.25
        and float(row.get("all_max_mev", np.inf)) <= best_pool_all_max + 1.0
    ]
    if near_best:
        selected = min(
            near_best,
            key=lambda row: (
                int(row.get("fit_count", len(row.get("indices", [])))),
                int(row.get("active_terms", 0) or 0),
                int(row.get("n_variables", row.get("variables", 0)) or 0),
                float(row.get("plot_rms_mev", np.inf)),
                float(row.get("plot_max_mev", np.inf)),
                str(row.get("name", "")),
            ),
        )
        if selected is not min(
            pool,
            key=lambda row: (
                float(row.get("plot_rms_mev", np.inf)),
                float(row.get("plot_max_mev", np.inf)),
                float(row.get("all_rms_mev", np.inf)),
                int(row.get("fit_count", len(row.get("indices", [])))),
                str(row.get("name", "")),
            ),
        ):
            status = "selected_simplest_near_best_candidate"
    else:
        selected = min(
            pool,
            key=lambda row: (
                float(row.get("plot_rms_mev", np.inf)),
                float(row.get("plot_max_mev", np.inf)),
                float(row.get("all_rms_mev", np.inf)),
                int(row.get("fit_count", len(row.get("indices", [])))),
                str(row.get("name", "")),
            ),
        )
    rejected_keys = {
        (str(row.get("name", "")), tuple(int(idx) for idx in row.get("indices", [])))
        for row in rejected
    }
    for row in rows:
        key = (str(row.get("name", "")), tuple(int(idx) for idx in row.get("indices", [])))
        if row is selected or key in rejected_keys:
            continue
        rejected.append({**row, "reason": "not_selected"})
        rejected_keys.add(key)
    return selected, {
        "selection_status": status,
        "selected": selected,
        "rejected": rejected,
        "global_sanity": {
            "best_all_rms_mev": float(best_all_rms),
            "best_all_max_mev": float(best_all_max),
            "all_rms_limit_mev": float(all_rms_limit),
            "all_max_limit_mev": float(all_max_limit),
        },
        "near_best_rule": {
            "plot_rms_margin_mev": 0.05,
            "plot_max_margin_mev": 0.25,
            "all_rms_margin_mev": 0.25,
            "all_max_margin_mev": 1.0,
            "candidate_count": int(len(near_best)),
        },
    }



_MATRIX_LOSS_BLOCK_ALIASES = {
    "all": "all",
    "dense": "all",
    "full": "all",
    "kinetic": "kinetic_diagonal",
    "kinetic_diagonal": "kinetic_diagonal",
    "kinetic_or_onsite_diagonal": "kinetic_diagonal",
    "onsite": "kinetic_diagonal",
    "diagonal": "kinetic_diagonal",
    "intra": "intralayer",
    "intralayer": "intralayer",
    "intralayer_offdiagonal": "intralayer",
    "inter": "interlayer",
    "interlayer": "interlayer",
    "tunneling": "interlayer",
}
