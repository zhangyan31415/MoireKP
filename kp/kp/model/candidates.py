from __future__ import annotations
import copy
import contextlib
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from ..reporting import KpReporter
from .core import (
    MoireConfig,
    _prepare_band_state,
    build_model,
    compute_bands,
    compute_coefficients,
    setup_logging,
)
from .model_selection import (
    BandWindowMetrics,
    CandidateScore,
    FamilyOrders,
    blocked_kpath_folds,
    build_fixed_band_weights,
    build_production_candidate_spec,
    clone_family_vocabulary,
    run_local_order_correction_sweep,
    run_staged_family_selection,
    select_four_model_profiles,
    select_high_low_profiles,
    subspace_overlap_metrics,
    weighted_band_error,
)
from . import artifacts
from . import configuration
from . import fit_sampling
from . import harmonic_selection
from . import refinement
from . import validation


class CandidateRejected(ValueError):
    """A candidate cannot satisfy its explicitly declared numerical constraints."""



@contextlib.contextmanager
def _redirect_model_output(log_path: Path, *, verbose: bool, mode: str):
    if verbose:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open(mode, encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            yield



def _progress_line(
    message: str,
    *,
    enabled: bool,
    style: str | None = None,
    stream: Any | None = None,
) -> None:
    if enabled:
        out = stream if stream is not None else sys.stdout
        use_color = harmonic_selection._terminal_color_enabled(out)
        style_code = {
            "start": "1",
            "done": "32",
            "section": "1",
            "fit_start": "1",
            "fit_done": "32",
            "path": "2",
            "warning": "33",
            "error": "31",
        }.get(str(style or ""))
        prefix = harmonic_selection._terminal_color("[kp model]", "2;36", enabled=use_color)
        body = harmonic_selection._terminal_color(str(message), style_code, enabled=use_color) if style_code else str(message)
        print(f"{prefix} {body}", file=out, flush=True)



def _term_progress_summary_rows(model: Any) -> list[tuple[str, int, int, int]]:
    rows: dict[str, dict[str, int]] = {}
    for term in getattr(model, "terms", {}).values():
        tag = str(getattr(term, "tag", "unknown") or "unknown")
        row = rows.setdefault(tag, {"terms": 0, "active_terms": 0, "variables": 0})
        row["terms"] += 1
        if bool(getattr(term, "active", True)):
            row["active_terms"] += 1
        row["variables"] += 2
    family_order = {
        "onsite": 0,
        "kinect": 1,
        "kinetic": 1,
        "intra": 2,
        "intralayer": 2,
        "inter": 3,
        "interlayer": 3,
    }
    return [
        (tag, int(values["terms"]), int(values["active_terms"]), int(values["variables"]))
        for tag, values in sorted(
            rows.items(),
            key=lambda item: (
                family_order.get(item[0].strip().lower(), len(family_order)),
                item[0].lower(),
            ),
        )
    ]



def _print_term_progress_summary(model: Any, *, enabled: bool, stream: Any | None = None) -> None:
    if not enabled:
        return
    out = stream if stream is not None else sys.stdout
    reporter = KpReporter("kp model", stream=out)
    rows = _term_progress_summary_rows(model)
    reporter.line()
    reporter.section("Continuum basis")
    if not rows:
        reporter.field("status", "no terms")
        return
    reporter.table(
        ("Family", "Terms", "Components"),
        ((tag, terms, variables) for tag, terms, _active_terms, variables in rows),
        right_align=(1, 2),
    )



def _print_fitted_term_progress_summary(
    model: Any,
    *,
    enabled: bool,
    stream: Any | None = None,
) -> None:
    if not enabled:
        return
    out = stream if stream is not None else sys.stdout
    reporter = KpReporter("kp model", stream=out)
    rows = _term_progress_summary_rows(model)
    reporter.line()
    reporter.section("Fitted continuum basis")
    if not rows:
        reporter.field("status", "no terms")
        return
    reporter.table(
        ("Family", "Seed terms", "Active terms"),
        ((tag, terms, active_terms) for tag, terms, active_terms, _variables in rows),
        right_align=(1, 2),
    )
    seed_terms = sum(row[1] for row in rows)
    active_terms = sum(row[2] for row in rows)
    fields: list[tuple[str, object]] = [
        ("active semantic terms", f"{active_terms} / {seed_terms}"),
    ]
    basis = getattr(model, "_compiled_response_basis", None)
    fitted = getattr(model, "_fitted_response_model", None)
    if basis is not None:
        fields.append(("independent coefficients", len(getattr(basis, "channels", ()))))
    if fitted is not None:
        coefficients = np.asarray(getattr(fitted, "coefficients", ()), dtype=float)
        fields.append(("nonzero coefficients", int(np.count_nonzero(coefficients))))
    reporter.fields(fields)



def _prune_small_coefficients(model: Any, threshold: float) -> dict[str, Any]:
    threshold = float(threshold)
    if threshold <= 0.0:
        return {"enabled": False, "threshold": threshold, "dropped": 0, "kept": None}
    basis = getattr(model, "_compiled_response_basis", None)
    fitted = getattr(model, "_fitted_response_model", None)
    if basis is not None or fitted is not None:
        if basis is None or fitted is None:
            raise ValueError("complete response model has an incomplete basis/fit runtime pair")
        coefficients = np.asarray(fitted.coefficients, dtype=float).copy()
        active = coefficients != 0.0
        amplitudes = np.abs(coefficients) * basis.response_scales
        drop = active & (amplitudes < threshold)
        keep = active & ~drop
        coefficients[drop] = 0.0
        refined_fit = fitted.with_coefficients(
            coefficients,
            channel_ids=basis.channel_ids,
            response_scales=basis.response_scales,
            provenance={
                "operation": "coefficient_pruning",
                "threshold": threshold,
                "normalization": basis.coefficient_normalization,
            },
        )
        setattr(model, "_fitted_response_model", refined_fit)
        refinement._sync_complete_response_coefficients_to_terms(model, coefficients)
        return {
            "enabled": True,
            "threshold": threshold,
            "normalization": basis.coefficient_normalization,
            "dropped": int(np.count_nonzero(drop)),
            "kept": int(np.count_nonzero(keep)),
            "dropped_channel_ids": [
                basis.channels[index].channel_id for index in np.flatnonzero(drop)
            ],
        }
    terms = list(getattr(model, "terms", {}).values())
    dropped = 0
    kept = 0
    for term in terms:
        if not bool(getattr(term, "active", True)):
            continue
        real = float(getattr(term, "r_value_real", 0.0) or 0.0)
        imag = float(getattr(term, "r_value_imag", 0.0) or 0.0)
        if float(np.hypot(real, imag)) < threshold:
            term.active = False
            term.r_value_real = 0.0
            term.r_value_imag = 0.0
            dropped += 1
        else:
            kept += 1
    return {"enabled": True, "threshold": threshold, "dropped": dropped, "kept": kept}



def _apply_model_selection_term_filter(
    model: Any,
    excluded_keys: Sequence[Any],
) -> dict[str, int]:
    """Remove symmetry/Hermiticity-closed ContinuumTerm groups before fitting."""

    excluded = set(excluded_keys)
    terms = getattr(model, "terms", {})
    initial = len(terms)
    if excluded:
        for key in list(terms):
            if key in excluded:
                terms.pop(key)
        candidate_terms = getattr(model, "candidate_terms", None)
        if candidate_terms is not None:
            model.candidate_terms = [
                term
                for term in candidate_terms
                if getattr(term, "key", None) not in excluded
            ]
    return {
        "requested": int(len(excluded)),
        "removed": int(initial - len(terms)),
        "remaining": int(len(terms)),
    }



def _run_model_pipeline(
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    log_path: Path,
    *,
    verbose: bool,
    progress: bool,
) -> dict[str, Any]:
    setup_logging(moire_config.log_level)
    first_log_write = True
    progress_stream = sys.stdout

    def run_stage(label: str, func):
        nonlocal first_log_write
        _progress_line(f"{label} ...", enabled=progress, style="start", stream=progress_stream)
        t0 = time.perf_counter()
        mode = "w" if first_log_write else "a"
        first_log_write = False
        with _redirect_model_output(log_path, verbose=verbose, mode=mode):
            result = func()
        _progress_line(f"{label} done in {time.perf_counter() - t0:.2f} s", enabled=progress, style="done", stream=progress_stream)
        return result

    model = run_stage("building continuum terms", lambda: build_model(moire_config))
    term_filter = _apply_model_selection_term_filter(
        model,
        getattr(model_config, "model_selection_excluded_term_keys", ()),
    )
    diagnostics = None
    null_filter = artifacts._null_channel_filter_summary(
        model,
        model_config.null_channel_abs_tol,
        model_config.null_channel_rel_tol,
    )
    pruning = {"enabled": False, "threshold": 0.0, "dropped": 0, "kept": None}
    refinement = {"enabled": False}
    if moire_config.heff is not None and moire_config.kpoints_fit is not None:
        _print_term_progress_summary(model, enabled=progress, stream=progress_stream)

        def fit_progress(message: str, *, state: str | None = None) -> None:
            style = "fit_done" if state == "done" else "fit_start" if state == "start" else None
            _progress_line(str(message), enabled=progress, style=style, stream=progress_stream)

        previous_output_dir = moire_config.output_dir
        if (
            str(moire_config.response_semantics)
            == "complete_linear"
            and previous_output_dir is None
        ):
            moire_config.output_dir = log_path.parent
        try:
            model, diagnostics = run_stage(
                "fitting coefficients",
                lambda: compute_coefficients(moire_config, model, progress_callback=fit_progress),
            )
        finally:
            moire_config.output_dir = previous_output_dir
        public_fit = dict(getattr(model_config, "fit_method_config", {}) or {})
        if public_fit:
            fitted_response = getattr(model, "_fitted_response_model", None)
            objective = getattr(fitted_response, "fit_objective", {}) or {}
            resolved = list(objective.get("resolved_band_counts", ()))
            if resolved:
                unique_counts = sorted({int(value) for value in resolved})
                _progress_line(
                    "low-energy window: "
                    f"requested={int(public_fit['requested_bands'])}, "
                    f"resolved={unique_counts}",
                    enabled=progress,
                    style="fit_done",
                )
        null_filter = artifacts._null_channel_filter_summary(
            model,
            model_config.null_channel_abs_tol,
            model_config.null_channel_rel_tol,
        )
        if null_filter.get("enabled") and int(null_filter.get("filtered_columns", 0)) > 0:
            _progress_line(
                "null-channel filter: abs_tol={abs_tol:g}, rel_tol={rel_tol:g}, "
                "filtered_columns={filtered_columns}".format(**null_filter),
                enabled=progress,
            )
        pruning = _prune_small_coefficients(model, model_config.coeff_prune_threshold)
        if pruning["enabled"]:
            _progress_line(
                "coefficient pruning: threshold={threshold:g}, kept={kept}, dropped={dropped}".format(**pruning),
                enabled=progress,
            )
        if bool(model_config.band_refinement_config.get("enabled", False)):
            refinement = run_stage(
                "refining band coefficients",
                lambda: refinement.refine_band_coefficients(
                    moire_config,
                    model_config,
                    model,
                    progress_callback=fit_progress,
                ),
            )
            refined = refinement.get("refined", {})
            if refined:
                _progress_line(
                    "band refinement: top RMS={top_band_rms_mev:.3f} meV, "
                    "top Max={top_band_max_mev:.3f} meV".format(**refined),
                    enabled=progress,
                )
            guard = refinement.get("acceptance_guard", {})
            if refinement.get("skipped"):
                _progress_line(
                    f"nonlinear refinement skipped: {refinement.get('reason', 'unspecified')}; input coefficients retained",
                    enabled=progress,
                )
            elif str(refinement.get("mode", "")) == "public_nonlinear":
                _progress_line(
                    "nonlinear band loss: "
                    f"{float(refinement['linear_initial_band_loss']):.6g} -> "
                    f"{float(refinement['nonlinear_final_band_loss']):.6g}; "
                    f"steps={int(refinement['nfev'])}/{int(refinement['max_steps'])}",
                    enabled=progress,
                    style="fit_done",
                )
            if guard.get("enabled") and not bool(guard.get("accepted", True)):
                failed = ", ".join(str(item) for item in guard.get("failed_windows", [])) or "validation"
                _progress_line(
                    f"band refinement rejected by acceptance guard; reverted=True; failed windows: {failed}",
                    enabled=progress,
                )
        _print_fitted_term_progress_summary(
            model,
            enabled=progress,
            stream=progress_stream,
        )
    if moire_config.kpoints is None:
        raise ValueError("config.kpoints must be provided for band computation.")
    want_band_overlap_vectors = bool(model_config.compare_to_heff)
    band_hamiltonians: list[np.ndarray] = []
    compiled_operator_runtime: list[Any | None] = []
    eigvals = run_stage(
        f"computing bands on {len(moire_config.kpoints)} k-points",
        lambda: compute_bands(
            moire_config,
            model,
            moire_config.kpoints,
            return_eigvecs=bool(moire_config.save_eigvecs or want_band_overlap_vectors),
            hamiltonians_out=band_hamiltonians,
            compiled_runtime_out=compiled_operator_runtime,
        ),
    )
    return {
        "model": model,
        "eigvals": eigvals,
        "band_hamiltonians": np.asarray(band_hamiltonians, dtype=complex),
        "compiled_operator_runtime": (
            compiled_operator_runtime[0]
            if compiled_operator_runtime
            else None
        ),
        "diagnostics": diagnostics,
        "null_channel_filter": null_filter,
        "coefficient_pruning": pruning,
        "band_refinement": refinement,
        "term_group_filter": term_filter,
        "model_log": str(log_path),
    }



def _model_config_for_fit_candidate(
    model_config: configuration.ConfiguredModel,
    candidate: Mapping[str, Any],
    *,
    scan_report: Mapping[str, Any] | None = None,
    disable_refinement: bool = False,
) -> configuration.ConfiguredModel:
    indices = [int(idx) for idx in candidate.get("indices", [])]
    metadata = {
        **dict(model_config.fit_selection_metadata),
        "source": "auto_low_energy_candidate_scan",
        "selected_candidate": str(candidate.get("name", "candidate")),
        "selected_indices": indices,
        "selected_candidate_reason": candidate.get("reason"),
    }
    if scan_report is not None:
        metadata["candidate_scan"] = artifacts._json_safe(scan_report)
    refinement_config = dict(model_config.band_refinement_config)
    if disable_refinement:
        refinement_config = {"enabled": False, "disabled_for": "auto_fit_candidate_prefit_scan"}
    return replace(
        model_config,
        fit_indices=indices,
        fit_selection_metadata=metadata,
        band_refinement_config=refinement_config,
    )



def _moire_config_for_fit_indices(
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    *,
    kpoints_all: np.ndarray,
    heff_list: np.ndarray,
    indices: Sequence[int],
) -> MoireConfig:
    out = copy.copy(moire_config)
    out.kpoints_fit = configuration._select_rows(kpoints_all, indices)
    out.heff = configuration._block_diag_heff(heff_list, indices)
    out.response_fit_indices = [int(index) for index in indices]
    out.output_dir = None
    return out



def _model_and_moire_for_order_candidate(
    *,
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    orders: FamilyOrders,
    active_families: Sequence[str],
) -> tuple[MoireConfig, configuration.ConfiguredModel, Any]:
    vocabulary = clone_family_vocabulary(
        max_order=model_config.max_order,
        term_templates=model_config.term_templates,
        orders=orders,
        active_families=active_families,
    )
    candidate_model = replace(
        model_config,
        max_order=dict(vocabulary.max_order),
        term_templates=[dict(item) for item in vocabulary.term_templates],
        band_refinement_config={
            "enabled": False,
            "disabled_for": "automatic_family_order_scan",
        },
    )
    candidate_moire = copy.copy(moire_config)
    candidate_moire.max_order = dict(vocabulary.max_order)
    candidate_moire.term_templates = [
        dict(item) for item in vocabulary.term_templates
    ]
    candidate_moire.output_dir = None
    specification = build_production_candidate_spec(vocabulary, orders)
    return candidate_moire, candidate_model, specification



def _active_independent_real_parameter_count(
    model: Any,
    *,
    tolerance: float,
) -> int:
    count = 0
    for term in getattr(model, "terms", {}).values():
        if not bool(getattr(term, "active", True)):
            continue
        for attribute in ("r_value_real", "r_value_imag"):
            value = float(getattr(term, attribute, 0.0))
            if abs(value) > float(tolerance):
                count += 1
    return int(count)



def _active_term_group_count(model: Any, *, tolerance: float) -> int:
    return int(
        sum(
            1
            for term in getattr(model, "terms", {}).values()
            if bool(getattr(term, "active", True))
            and (
                abs(float(getattr(term, "r_value_real", 0.0))) > float(tolerance)
                or abs(float(getattr(term, "r_value_imag", 0.0))) > float(tolerance)
            )
        )
    )



def _harmonic_support_size(
    model_config: configuration.ConfiguredModel,
    *,
    model: Any | None = None,
    tolerance: float = 0.0,
) -> int:
    if model is not None:
        active_harmonics: set[tuple[str, str]] = set()
        for term in getattr(model, "terms", {}).values():
            if not bool(getattr(term, "active", True)):
                continue
            if (
                abs(float(getattr(term, "r_value_real", 0.0))) <= float(tolerance)
                and abs(float(getattr(term, "r_value_imag", 0.0))) <= float(tolerance)
            ):
                continue
            tag = str(getattr(term, "tag", "")).strip().lower()
            if tag not in {"intra", "intralayer", "inter", "interlayer"}:
                continue
            metadata = getattr(term, "registry_metadata", {}) or {}
            harmonic_id = metadata.get("harmonic_id")
            if harmonic_id is None:
                harmonic_id = tuple(metadata.get("harmonic_vector", ()))
            family = "intra" if tag.startswith("intra") else "inter"
            active_harmonics.add((family, repr(harmonic_id)))
        return int(len(active_harmonics))
    counts = harmonic_selection._resolved_harmonic_count_limits(model_config.harmonics_config)
    return int(sum(max(0, int(counts.get(kind, 0))) for kind in ("intra", "inter")))



def _automatic_term_group_keep_sets(
    ranked_importance: Sequence[tuple[Any, float]],
    *,
    keep_fractions: Sequence[float],
) -> tuple[tuple[Any, ...], ...]:
    """Build a deterministic compact-model ladder from closed term groups."""

    ranked = list(ranked_importance)
    if not ranked:
        return ()
    ordered = sorted(
        ranked,
        key=lambda item: (float(item[1]), repr(item[0])),
    )
    count = len(ordered)
    keep_counts = sorted(
        {
            max(1, min(count - 1, int(math.ceil(float(fraction) * count))))
            for fraction in keep_fractions
            if count > 1
        }
    )
    return tuple(
        tuple(key for key, _importance in ordered[-keep_count:])
        for keep_count in keep_counts
    )



def _adjoint_closed_term_importance_groups(
    ranked_importance: Sequence[tuple[Any, float]],
) -> tuple[tuple[tuple[Any, ...], float], ...]:
    """Combine raw term keys into Hermitian-adjoint-closed pruning groups."""

    ranked = list(ranked_importance)
    importance_by_key = {key: float(importance) for key, importance in ranked}
    visited: set[Any] = set()
    groups: list[tuple[tuple[Any, ...], float]] = []
    for key, importance in ranked:
        if key in visited:
            continue
        required = (
            "Mz",
            "Mz_star",
            "layer_from",
            "layer_to",
            "orbital_from",
            "orbital_to",
            "p",
        )
        if all(hasattr(key, attribute) for attribute in required):
            partner = type(key)(
                int(key.Mz_star),
                int(key.Mz),
                int(key.layer_to),
                int(key.layer_from),
                int(key.orbital_to),
                int(key.orbital_from),
                tuple(-float(value) for value in key.p),
            )
        else:
            partner = key
        members = (key,)
        group_importance = float(importance)
        if partner != key and partner in importance_by_key:
            members = (key, partner)
            group_importance = max(
                group_importance,
                float(importance_by_key[partner]),
            )
        visited.update(members)
        groups.append((members, group_importance))
    return tuple(
        sorted(groups, key=lambda item: (float(item[1]), repr(item[0])))
    )



def _select_nonlinear_frontier_scores(
    candidates: Sequence[CandidateScore],
    *,
    profiles: Any,
    maximum_candidates: int,
    maximum_variables: int = 3000,
    maximum_orders: FamilyOrders | None = None,
) -> tuple[CandidateScore, ...]:
    """Choose a bounded linear complexity frontier for nonlinear refinement."""

    limit = max(1, int(maximum_candidates))
    variable_limit = max(1, int(maximum_variables))
    eligible = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.solver_family == "linear"
            and candidate.independent_real_parameters <= variable_limit
            and (
                maximum_orders is None
                or (
                    candidate.orders.kinetic <= maximum_orders.kinetic
                    and candidate.orders.intra <= maximum_orders.intra
                    and candidate.orders.inter <= maximum_orders.inter
                )
            )
            and candidate.certified
            and candidate.guards_passed
        ),
        key=lambda candidate: (
            candidate.independent_real_parameters,
            candidate.weighted_rms_mev,
            candidate.name,
        ),
    )
    frontier: list[CandidateScore] = []
    best_rms = float("inf")
    for candidate in eligible:
        if float(candidate.weighted_rms_mev) < best_rms:
            frontier.append(candidate)
            best_rms = float(candidate.weighted_rms_mev)

    required_names = {
        decision.selected.name
        for decision in (profiles.high, profiles.low)
        if decision.selected is not None
        and decision.selected.independent_real_parameters <= variable_limit
    }
    if maximum_orders is not None:
        endpoint = [
            candidate
            for candidate in eligible
            if candidate.orders == maximum_orders
        ]
        if endpoint:
            required_names.add(
                min(
                    endpoint,
                    key=lambda candidate: (
                        candidate.weighted_rms_mev,
                        candidate.weighted_max_mev,
                        candidate.independent_real_parameters,
                        candidate.name,
                    ),
                ).name
            )
    selected_by_name = {candidate.name: candidate for candidate in frontier}
    selected_by_name.update(
        {
            candidate.name: candidate
            for candidate in eligible
            if candidate.name in required_names
        }
    )
    selected = sorted(
        selected_by_name.values(),
        key=lambda candidate: (
            candidate.independent_real_parameters,
            candidate.weighted_rms_mev,
            candidate.name,
        ),
    )
    if len(selected) <= limit:
        return tuple(selected)

    required = [candidate for candidate in selected if candidate.name in required_names]
    optional = [candidate for candidate in selected if candidate.name not in required_names]
    kept = list(required[:limit])
    remaining = limit - len(kept)
    if remaining > 0:
        if remaining >= len(optional):
            kept.extend(optional)
        else:
            positions = np.linspace(0, len(optional) - 1, remaining, dtype=int)
            kept.extend(optional[int(index)] for index in positions)
    return tuple(
        sorted(
            {candidate.name: candidate for candidate in kept}.values(),
            key=lambda candidate: (
                candidate.independent_real_parameters,
                candidate.weighted_rms_mev,
                candidate.name,
            ),
        )
    )



def _complete_optional_profile_pair(profiles: Any) -> Any | None:
    """Return an optional solver profile only when both user profiles exist."""

    if profiles is None:
        return None
    high = getattr(profiles, "high", None)
    low = getattr(profiles, "low", None)
    if getattr(high, "selected", None) is None or getattr(low, "selected", None) is None:
        return None
    return profiles



@dataclass(frozen=True)
class AutomaticProfileBandWindows:
    """Resolved compact and broad edge-band windows."""

    low_band_count: int
    high_band_count: int
    layer_q_counts: tuple[int, int]



def _automatic_profile_band_windows(
    qset1: np.ndarray,
    qset2: np.ndarray,
    *,
    n_orb: tuple[int, int],
    total_bands: int,
    q_shell_count: int = 2,
    tol: float = 1.0e-8,
) -> AutomaticProfileBandWindows:
    """Resolve low=2x central manifold and high=complete radial Q shells."""

    dimension = int(total_bands)
    if dimension <= 0:
        raise ValueError("total_bands must be positive")
    if int(q_shell_count) <= 0:
        raise ValueError("q_shell_count must be positive")
    orbitals = tuple(int(value) for value in n_orb)
    if len(orbitals) != 2 or any(value < 0 for value in orbitals):
        raise ValueError("n_orb must contain two non-negative layer dimensions")

    layer_q_counts: list[int] = []
    for qset, norb in zip((qset1, qset2), orbitals):
        points = np.asarray(qset, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Q set must have shape (N, 2), got {points.shape}")
        if norb == 0 or len(points) == 0:
            layer_q_counts.append(0)
            continue
        norms = np.linalg.norm(points, axis=1)
        shells = harmonic_selection._unique_nonnegative_norms(norms, tol=float(tol))
        retained = shells[: int(q_shell_count)]
        if not retained:
            layer_q_counts.append(0)
            continue
        cutoff = float(retained[-1])
        layer_q_counts.append(
            int(
                np.count_nonzero(
                    norms <= cutoff + float(tol) * max(1.0, abs(cutoff))
                )
            )
        )

    low = min(dimension, max(1, 2 * sum(orbitals)))
    high = min(
        dimension,
        max(low, sum(norb * count for norb, count in zip(orbitals, layer_q_counts))),
    )
    return AutomaticProfileBandWindows(
        low_band_count=int(low),
        high_band_count=int(high),
        layer_q_counts=(int(layer_q_counts[0]), int(layer_q_counts[1])),
    )



def _linear_profile_fit_objective(
    *,
    target_bands: str,
    band_count: int,
    two_sided_projector_weight: float = 300.0,
    floor: float = 0.05,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Build the complete-response spectral objective for one profile window."""

    edge = str(target_bands).strip().lower()
    if edge not in {"top", "bottom"}:
        raise ValueError("target_bands must be 'top' or 'bottom'")
    count = int(band_count)
    if count <= 0:
        raise ValueError("band_count must be positive")
    projector_weight = float(two_sided_projector_weight)
    if not np.isfinite(projector_weight) or projector_weight <= 0.0:
        raise ValueError("two_sided_projector_weight must be positive and finite")
    return {
        "mode": "target_spectral_linear",
        "target_reference": "current_heff_support_mask",
        "edge": edge,
        "window": {
            "mode": "fixed_count_degeneracy_safe",
            "bands": count,
            "degeneracy_tol_mev": 0.1,
        },
        "floor": float(floor),
        "alpha": float(alpha),
        "normalization": "mean_trace_per_dimension",
        "two_sided_projector": {
            "enabled": True,
            "weight": projector_weight,
        },
    }



def _automatic_selection_primary_band_count(
    model_config: configuration.ConfiguredModel,
    *,
    total_bands: int,
) -> int:
    """Resolve the edge-band window used by automatic model selection.

    ``model.band_slice`` is a broad comparison/export window and must not be
    interpreted as the primary low-energy target.  Prefer the fitting
    objective, then an explicitly weighted refinement window, then the user
    facing plot window.  The projected orbital count is the final fallback.
    """

    count: int | None = None
    objective = getattr(model_config, "response_fit_objective", {}) or {}
    if (
        isinstance(objective, Mapping)
        and str(objective.get("mode", "")).strip().lower()
        == "target_spectral_linear"
    ):
        window = objective.get("window", {})
        if isinstance(window, Mapping) and str(window.get("mode", "")).strip().lower() == (
            "fixed_count_degeneracy_safe"
        ):
            count = int(window.get("bands", 0))

    refinement = getattr(model_config, "band_refinement_config", {}) or {}
    if count is None and isinstance(refinement, Mapping) and bool(
        refinement.get("enabled", False)
    ):
        weighted = refinement.get("weighted_band_loss", {})
        if isinstance(weighted, Mapping) and bool(weighted.get("enabled", False)):
            count = int(weighted.get("primary_bands", 0))

    plot = getattr(model_config, "band_plot_config", {}) or {}
    if count is None and isinstance(plot, Mapping) and "top_bands" in plot:
        count = int(plot["top_bands"])

    if count is None or count <= 0:
        count = max(1, int(sum(model_config.n_orb)))
    return min(int(total_bands), int(count))



def _automatic_low_profile_band_count(
    model_config: configuration.ConfiguredModel,
    *,
    total_bands: int,
) -> int:
    """Use twice the physical projected manifold for compact profiles."""

    return min(int(total_bands), max(1, 2 * int(sum(model_config.n_orb))))



def _balanced_family_order_candidates(
    order_candidates: Mapping[str, Sequence[int]],
) -> tuple[FamilyOrders, ...]:
    """Pair non-maximal family orders into a compact joint-model ladder."""

    values = {
        family: tuple(sorted({int(value) for value in order_candidates[family]}))
        for family in ("kinetic", "intra", "inter")
    }
    count = min(len(items) for items in values.values())
    if count <= 1:
        return ()
    return tuple(
        FamilyOrders(
            values["kinetic"][index],
            values["intra"][index],
            values["inter"][index],
        )
        for index in range(count - 1)
    )



def _rank_closed_term_groups_for_edge(
    *,
    model: Any,
    moire_config: MoireConfig,
    target_hamiltonians: np.ndarray,
    model_config: configuration.ConfiguredModel,
    kpoints: np.ndarray,
) -> tuple[tuple[tuple[Any, ...], float], ...]:
    """Rank Hermitian-adjoint-closed term groups by edge contribution."""

    terms = list(getattr(model, "terms", {}).values())
    if not terms:
        return ()

    def coefficient_importance(term: Any) -> float:
        return float(
            np.hypot(
                float(getattr(term, "r_value_real", 0.0)),
                float(getattr(term, "r_value_imag", 0.0)),
            )
        )

    state = _prepare_band_state(moire_config, model)
    _target_eigvals, target_eigvecs = np.linalg.eigh(
        np.asarray(target_hamiltonians, dtype=np.complex128)
    )
    target_bands = str(
        model_config.raw.get("model", {}).get("target_bands", "top")
    ).strip().lower()
    primary_count = _automatic_selection_primary_band_count(
        model_config,
        total_bands=target_eigvecs.shape[2],
    )
    primary_slice = fit_sampling._auto_low_energy_band_slice(
        target_eigvecs.shape[2],
        min(primary_count, target_eigvecs.shape[2]),
        target_bands,
    )
    target_basis = target_eigvecs[
        :,
        :,
        primary_slice[0] : primary_slice[1],
    ]
    ranked = []
    for term in terms:
        real_response, imag_response = (
            refinement._term_response_pair_hamiltonians_for_kpoints(
                state,
                term,
                kpoints,
            )
        )
        contribution = (
            float(getattr(term, "r_value_real", 0.0)) * real_response
            + float(getattr(term, "r_value_imag", 0.0)) * imag_response
        )
        projected = np.einsum(
            "kdi,kde,kej->kij",
            target_basis.conj(),
            contribution,
            target_basis,
            optimize=True,
        )
        importance = float(np.sqrt(np.mean(np.abs(projected) ** 2)))
        ranked.append((getattr(term, "key"), importance))
    ranked = tuple(sorted(ranked, key=lambda item: (item[1], repr(item[0]))))
    return _adjoint_closed_term_importance_groups(ranked)



def _blocked_standard_error(
    per_k_squared_error: np.ndarray,
    kpoints: np.ndarray,
    *,
    n_folds: int,
) -> float:
    squared = np.asarray(per_k_squared_error, dtype=float)
    points = np.asarray(kpoints, dtype=float)
    if squared.ndim != 1 or len(squared) != len(points) or len(points) < 2:
        return 0.0
    try:
        folds = blocked_kpath_folds(
            points,
            n_folds=min(int(n_folds), len(points)),
        )
    except ValueError:
        return 0.0
    fold_rms_mev = np.asarray(
        [
            np.sqrt(np.mean(squared[fold.validation_indices])) * 1000.0
            for fold in folds
        ],
        dtype=float,
    )
    if len(fold_rms_mev) < 2:
        return 0.0
    return float(np.std(fold_rms_mev, ddof=1) / np.sqrt(len(fold_rms_mev)))



def _stage_matrix_candidate_metrics(
    *,
    stage: str,
    model_hamiltonians: np.ndarray,
    target_hamiltonians: np.ndarray,
    qset1_dim: int,
    kpoints: np.ndarray,
    n_folds: int,
) -> tuple[float, float, float]:
    model = np.asarray(model_hamiltonians, dtype=np.complex128)
    target = np.asarray(target_hamiltonians, dtype=np.complex128)
    if model.shape != target.shape or model.ndim != 3:
        raise ValueError("candidate and target Hamiltonians must have matching 3D shape")
    residual = model - target
    if stage == "kinetic":
        reference = int(np.argmin(np.linalg.norm(np.asarray(kpoints), axis=1)))
        diagonal = np.diagonal(residual, axis1=1, axis2=2)
        selected = diagonal - diagonal[reference : reference + 1]
    elif stage == "intra":
        left = residual[:, :qset1_dim, :qset1_dim].reshape(len(residual), -1)
        right = residual[:, qset1_dim:, qset1_dim:].reshape(len(residual), -1)
        selected = np.concatenate((left, right), axis=1)
    else:
        raise ValueError(f"matrix stage metrics do not support stage {stage!r}")
    per_k_squared = np.mean(np.abs(selected) ** 2, axis=1)
    return (
        float(np.sqrt(np.mean(per_k_squared)) * 1000.0),
        _blocked_standard_error(per_k_squared, kpoints, n_folds=n_folds),
        float(np.max(np.abs(selected)) * 1000.0),
    )



def _align_model_eigenvalues_for_selection(
    model_eigenvalues: Any,
    target_eigenvalues: Any,
    *,
    align: str,
) -> np.ndarray:
    """Apply the same global edge alignment used by band-plot metrics."""

    model = np.asarray(model_eigenvalues, dtype=float)
    target = np.asarray(target_eigenvalues, dtype=float)
    if model.shape != target.shape or model.ndim != 2:
        raise ValueError("model and target eigenvalues must have matching 2D shape")
    mode = str(align).strip().lower()
    if mode in {"top", "top_band", "top-band"}:
        return model - float(np.max(model[:, -1])) + float(np.max(target[:, -1]))
    if mode in {"bottom", "bottom_band", "bottom-band"}:
        return model - float(np.min(model[:, 0])) + float(np.min(target[:, 0]))
    if mode in {"none", "false", "0", ""}:
        return model.copy()
    raise ValueError(f"unsupported automatic-selection alignment {align!r}")



def _band_refinement_result_is_selectable(result: Mapping[str, Any]) -> bool:
    guard = (result.get("band_refinement") or {}).get("acceptance_guard", {})
    return (
        not bool(guard.get("enabled"))
        or bool(guard.get("accepted", True))
        or bool(guard.get("reverted", False))
    )



def _complete_candidate_score(
    *,
    name: str,
    orders: FamilyOrders,
    result: Mapping[str, Any],
    target_hamiltonians: np.ndarray,
    model_config: configuration.ConfiguredModel,
    kpoints: np.ndarray,
    n_folds: int,
    solver_family: str = "linear",
    selection_scope: str = "both",
    primary_band_count: int | None = None,
    low_band_count: int | None = None,
    high_band_count: int | None = None,
) -> CandidateScore:
    model_hamiltonians = np.asarray(result["band_hamiltonians"], dtype=np.complex128)
    target_eigvals, target_eigvecs = np.linalg.eigh(target_hamiltonians)
    model_eigvals, model_eigvecs = np.linalg.eigh(model_hamiltonians)
    model_eigvals = _align_model_eigenvalues_for_selection(
        model_eigvals,
        target_eigvals,
        align=str(model_config.band_plot_config.get("align", "none")),
    )
    target_bands = str(
        model_config.raw.get("model", {}).get("target_bands", "top")
    ).strip().lower()

    def measure_window(count: int | None) -> BandWindowMetrics | None:
        if count is None:
            return None
        resolved = min(int(target_eigvals.shape[1]), max(1, int(count)))
        window_weights = build_fixed_band_weights(
            target_eigvals,
            band_edge=target_bands,
            n_bands=resolved,
            floor=0.0,
        )
        band_metrics = weighted_band_error(
            model_eigvals,
            target_eigvals,
            window_weights,
        )
        start, stop = fit_sampling._auto_low_energy_band_slice(
            int(target_eigvals.shape[1]),
            resolved,
            target_bands,
        )
        overlap_metrics = subspace_overlap_metrics(
            model_eigvecs[:, :, start:stop],
            target_eigvecs[:, :, start:stop],
        )
        return BandWindowMetrics(
            band_count=resolved,
            weighted_rms_mev=band_metrics.weighted_rms_mev,
            weighted_max_mev=band_metrics.maximum_abs_error_mev,
            mean_subspace_overlap=overlap_metrics.mean_overlap,
            minimum_singular_value=overlap_metrics.minimum_singular_value,
        )

    low_window_metrics = measure_window(low_band_count)
    high_window_metrics = measure_window(high_band_count)
    primary_count = (
        (
            int(low_window_metrics.band_count)
            if str(selection_scope).strip().lower() == "low"
            and low_window_metrics is not None
            else int(high_window_metrics.band_count)
            if high_window_metrics is not None
            else _automatic_selection_primary_band_count(
                model_config,
                total_bands=target_eigvals.shape[1],
            )
        )
        if primary_band_count is None
        else min(int(target_eigvals.shape[1]), max(1, int(primary_band_count)))
    )
    weights = build_fixed_band_weights(
        target_eigvals,
        band_edge=target_bands,
        n_bands=primary_count,
        floor=0.0,
    )
    metrics = weighted_band_error(model_eigvals, target_eigvals, weights)
    normalization_by_k = np.sum(weights, axis=1)
    squared_by_k = np.divide(
        np.sum(weights * (model_eigvals - target_eigvals) ** 2, axis=1),
        normalization_by_k,
        out=np.zeros_like(normalization_by_k),
        where=normalization_by_k > 0.0,
    )
    band_slice = fit_sampling._auto_low_energy_band_slice(
        target_eigvals.shape[1],
        primary_count,
        target_bands,
    )
    overlap = subspace_overlap_metrics(
        model_eigvecs[:, :, band_slice[0] : band_slice[1]],
        target_eigvecs[:, :, band_slice[0] : band_slice[1]],
    )
    expanded = validation.compare_bands(model_eigvals, target_eigvals, band_slice=None)
    model = result.get("model")
    return CandidateScore(
        name=name,
        orders=orders,
        independent_real_parameters=_active_independent_real_parameter_count(
            model,
            tolerance=model_config.coeff_tol,
        ),
        weighted_rms_mev=metrics.weighted_rms_mev,
        weighted_rms_se_mev=_blocked_standard_error(
            squared_by_k,
            kpoints,
            n_folds=n_folds,
        ),
        weighted_max_mev=metrics.maximum_abs_error_mev,
        mean_subspace_overlap=overlap.mean_overlap,
        expanded_weighted_rms_mev=float(expanded["rms_error_mev"]),
        certified=True,
        guards_passed=_band_refinement_result_is_selectable(result),
        solver_family=solver_family,
        active_group_count=_active_term_group_count(
            model,
            tolerance=model_config.coeff_tol,
        ),
        harmonic_support_size=_harmonic_support_size(
            model_config,
            model=model,
            tolerance=model_config.coeff_tol,
        ),
        selection_scope=selection_scope,
        primary_band_count=int(primary_count),
        low_window_metrics=low_window_metrics,
        high_window_metrics=high_window_metrics,
    )



def _complete_term_templates_for_profile_refit(
    model_config: configuration.ConfiguredModel,
    *,
    moire_config: MoireConfig | None = None,
) -> list[dict[str, Any]]:
    """Return a symmetry-closed vocabulary for complete-response profile fits."""

    current = [copy.deepcopy(row) for row in model_config.term_templates]
    if current and all(row.get("term_space_policy") == "complete" for row in current):
        return current

    valley_type = str(model_config.valley_model.get("valley_type", ""))
    if valley_type in configuration._CASE_DERIVED_GENERATORS:
        if moire_config is None:
            raise ValueError(
                f"{valley_type} complete-response profile refit requires resolved "
                "Q sets, sectors, and harmonics"
            )
        templates, _diagnostics = configuration._case_derived_term_templates(
            valley_type=valley_type,
            sectors=list(moire_config.sectors),
            n_orb=(int(moire_config.n_orb1), int(moire_config.n_orb2)),
            max_order=model_config.max_order,
            Q_set1=np.asarray(moire_config.Q_set1, dtype=float),
            Q_set2=np.asarray(moire_config.Q_set2, dtype=float),
            intra_harmonics_map=dict(moire_config.intra_harmonics_map),
            inter_harmonics_map=dict(moire_config.inter_harmonics_map),
        )
        return templates
    return current


def _refit_linear_profile_candidates(
    *,
    frontier_scores: Sequence[CandidateScore],
    linear_runs: Mapping[str, Mapping[str, Any]],
    target_hamiltonians: np.ndarray,
    kpoints: np.ndarray,
    n_folds: int,
    low_primary_count: int,
    high_primary_count: int,
    candidate_dir: Path,
    verbose: bool,
    two_sided_projector_weight: float = 300.0,
) -> dict[str, dict[str, Any]]:
    """Refit shortlisted linear vocabularies for their physical profile window."""

    candidate_dir.mkdir(parents=True, exist_ok=True)
    refitted: dict[str, dict[str, Any]] = {}
    for score in frontier_scores:
        if score.selection_scope not in {"high", "low"}:
            raise ValueError("linear profile refit requires a high or low selection scope")
        base_run = linear_runs[score.name]
        base_model_config = base_run["model_config"]
        count = (
            int(low_primary_count)
            if score.selection_scope == "low"
            else int(high_primary_count)
        )
        if base_model_config.fit_weighting == "uniform":
            base_result = base_run.get("result")
            if not isinstance(base_result, Mapping):
                raise ValueError(
                    "uniform linear profile selection requires the completed "
                    f"equal-matrix candidate result for {score.name!r}"
                )
            name = f"{score.name}__linear_{score.selection_scope}_uniform"
            candidate_model = replace(
                base_model_config,
                response_fit_objective={"mode": "equal_matrix"},
            )
            refitted_score = _complete_candidate_score(
                name=name,
                orders=score.orders,
                result=base_result,
                target_hamiltonians=target_hamiltonians,
                model_config=candidate_model,
                kpoints=kpoints,
                n_folds=n_folds,
                solver_family="linear",
                selection_scope=score.selection_scope,
                primary_band_count=count,
                low_band_count=int(low_primary_count),
                high_band_count=int(high_primary_count),
            )
            refitted[name] = {
                **base_run,
                "score": refitted_score,
                "result": base_result,
                "model_config": candidate_model,
                "stage": "linear_profile_uniform_reuse",
                "linear_source": score.name,
                "selection_scope": score.selection_scope,
                "fit_objective": {"mode": "equal_matrix"},
            }
            continue
        target_bands = str(
            base_model_config.raw.get("model", {}).get("target_bands", "top")
        ).strip().lower()
        objective = _linear_profile_fit_objective(
            target_bands=target_bands,
            band_count=count,
            two_sided_projector_weight=two_sided_projector_weight,
        )
        complete_templates = _complete_term_templates_for_profile_refit(
            base_model_config,
            moire_config=base_run["moire_config"],
        )
        candidate_model = replace(
            base_model_config,
            response_semantics="complete_linear",
            response_fit_objective=objective,
            term_templates=complete_templates,
            model_selection_excluded_term_keys=(),
            band_refinement_config={
                "enabled": False,
                "disabled_for": f"linear_{score.selection_scope}_weighted_refit",
            },
        )
        candidate_moire = copy.copy(base_run["moire_config"])
        candidate_moire.output_dir = None
        candidate_moire.response_semantics = "complete_linear"
        candidate_moire.response_fit_objective = copy.deepcopy(objective)
        candidate_moire.term_templates = [
            copy.deepcopy(row) for row in complete_templates
        ]
        name = f"{score.name}__linear_{score.selection_scope}_weighted"
        result = _run_model_pipeline(
            candidate_moire,
            candidate_model,
            candidate_dir / f"{name}.log",
            verbose=verbose,
            progress=False,
        )
        refitted_score = _complete_candidate_score(
            name=name,
            orders=score.orders,
            result=result,
            target_hamiltonians=target_hamiltonians,
            model_config=candidate_model,
            kpoints=kpoints,
            n_folds=n_folds,
            solver_family="linear",
            selection_scope=score.selection_scope,
            primary_band_count=count,
            low_band_count=int(low_primary_count),
            high_band_count=int(high_primary_count),
        )
        refitted[name] = {
            "score": refitted_score,
            "result": result,
            "moire_config": candidate_moire,
            "model_config": candidate_model,
            "specification": base_run.get("specification"),
            "stage": "linear_profile_weighted_refit",
            "linear_source": score.name,
            "selection_scope": score.selection_scope,
            "fit_objective": objective,
        }
    return refitted



def _refine_nonlinear_frontier_candidates(
    *,
    frontier_scores: Sequence[CandidateScore],
    linear_runs: Mapping[str, Mapping[str, Any]],
    model_config: configuration.ConfiguredModel,
    target_hamiltonians: np.ndarray,
    kpoints: np.ndarray,
    n_folds: int,
    low_primary_count: int,
    high_primary_count: int,
    candidate_dir: Path,
    verbose: bool,
) -> dict[str, dict[str, Any]]:
    """Nonlinearly refine each selected linear vocabulary independently."""

    refinement = dict(model_config.band_refinement_config or {})
    if not bool(refinement.get("enabled", False)):
        return {}
    candidate_dir.mkdir(parents=True, exist_ok=True)
    refined_runs: dict[str, dict[str, Any]] = {}
    target_bands = str(
        model_config.raw.get("model", {}).get("target_bands", "top")
    ).strip().lower()
    primary_slice = fit_sampling._auto_low_energy_band_slice(
        int(np.asarray(target_hamiltonians).shape[-1]),
        low_primary_count,
        target_bands,
    )
    high_slice = fit_sampling._auto_low_energy_band_slice(
        int(np.asarray(target_hamiltonians).shape[-1]),
        int(high_primary_count),
        target_bands,
    )
    high_refinement = copy.deepcopy(refinement)
    high_refinement["band_slice"] = list(high_slice)
    high_weighted = dict(high_refinement.get("weighted_band_loss", {}))
    high_weighted.update(
        {
            "enabled": model_config.fit_weighting != "uniform",
            "primary_bands": int(high_primary_count),
        }
    )
    high_refinement["weighted_band_loss"] = high_weighted
    high_guard = dict(high_refinement.get("acceptance_guard", {}))
    high_guard.update(
        {
            "enabled": True,
            "band_slice": list(high_slice),
            "align": target_bands,
            "guard_primary_window": True,
        }
    )
    high_refinement["acceptance_guard"] = high_guard
    for score in frontier_scores:
        base_run = linear_runs[score.name]
        low_refinement = copy.deepcopy(refinement)
        low_refinement["matrix_weight"] = 0.0
        low_refinement["optimizer"] = "auto"
        low_refinement["band_slice"] = list(primary_slice)
        low_weighted = dict(low_refinement.get("weighted_band_loss", {}))
        low_weighted.update(
            {
                "enabled": model_config.fit_weighting != "uniform",
                "primary_bands": int(low_primary_count),
                "floor": 0.0,
            }
        )
        low_refinement["weighted_band_loss"] = low_weighted
        low_guard = dict(low_refinement.get("acceptance_guard", {}))
        low_guard.update(
            {
                "enabled": True,
                "band_slice": list(primary_slice),
                "align": target_bands,
                "guard_primary_window": True,
                "guard_all_bands": False,
            }
        )
        low_refinement["acceptance_guard"] = low_guard
        scoped_refinements = {
            "high": high_refinement,
            "low": low_refinement,
        }
        requested_scopes = (
            ("high", "low")
            if score.selection_scope == "both"
            else (score.selection_scope,)
        )
        for selection_scope in requested_scopes:
            scoped_refinement = scoped_refinements[selection_scope]
            candidate_model = replace(
                base_run["model_config"],
                band_refinement_config=copy.deepcopy(scoped_refinement),
            )
            candidate_moire = copy.copy(base_run["moire_config"])
            candidate_moire.output_dir = None
            name = f"{score.name}__nonlinear_{selection_scope}"
            result = _run_model_pipeline(
                candidate_moire,
                candidate_model,
                candidate_dir / f"{name}.log",
                verbose=verbose,
                progress=False,
            )
            refined_score = _complete_candidate_score(
                name=name,
                orders=score.orders,
                result=result,
                target_hamiltonians=target_hamiltonians,
                model_config=candidate_model,
                kpoints=kpoints,
                n_folds=n_folds,
                solver_family="nonlinear",
                selection_scope=selection_scope,
                primary_band_count=(
                    int(low_primary_count)
                    if selection_scope == "low"
                    else int(high_primary_count)
                ),
                low_band_count=int(low_primary_count),
                high_band_count=int(high_primary_count),
            )
            refined_runs[name] = {
                "score": refined_score,
                "result": result,
                "moire_config": candidate_moire,
                "model_config": candidate_model,
                "specification": base_run.get("specification"),
                "stage": "nonlinear_frontier",
                "linear_source": score.name,
                "selection_scope": selection_scope,
            }
    return refined_runs



def _four_profile_candidate_pool(
    *,
    linear_high_scores: Sequence[CandidateScore],
    linear_low_scores: Sequence[CandidateScore],
    weighted_linear_scores: Sequence[CandidateScore],
    nonlinear_scores: Sequence[CandidateScore],
) -> tuple[CandidateScore, ...]:
    """Keep the complete linear scan when comparing four model profiles."""

    candidates_by_name: dict[str, CandidateScore] = {}
    for score in (
        *linear_high_scores,
        *linear_low_scores,
        *weighted_linear_scores,
        *nonlinear_scores,
    ):
        candidates_by_name.setdefault(score.name, score)
    return tuple(candidates_by_name.values())



def _preferred_automatic_primary_family(
    profile_families: Mapping[str, Any],
) -> str:
    """Prefer a passing high profile over a warning-only solver family."""

    linear_profiles = profile_families.get("linear")
    nonlinear_profiles = profile_families.get("nonlinear")
    if nonlinear_profiles is None:
        return "linear"
    if linear_profiles is None:
        return "nonlinear"

    linear_high = linear_profiles.high
    nonlinear_high = nonlinear_profiles.high
    if nonlinear_high.selected is None:
        return "linear"
    if linear_high.selected is None:
        return "nonlinear"

    status_priority = {"PASS": 2, "WARN_BEST_AVAILABLE": 1, "FAIL": 0}
    linear_priority = status_priority.get(str(linear_high.status), 0)
    nonlinear_priority = status_priority.get(str(nonlinear_high.status), 0)
    if linear_priority > nonlinear_priority:
        return "linear"
    if nonlinear_priority > linear_priority:
        return "nonlinear"
    linear_rms = float(linear_high.selected.weighted_rms_mev)
    nonlinear_rms = float(nonlinear_high.selected.weighted_rms_mev)
    if linear_rms < nonlinear_rms - 1.0e-12:
        return "linear"
    return "nonlinear"



def _run_automatic_family_order_scan(
    *,
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    output_dir: Path,
    verbose: bool,
    progress: bool,
) -> dict[str, Any]:
    selection_config = model_config.model_selection_config
    if selection_config is None or not bool(selection_config.enabled):
        raise ValueError("automatic family order scan requires enabled model selection")
    original_heff = configuration._load_heff_in_model_basis(model_config)
    target_all = validation._current_heff_support_hamiltonians(
        original_heff,
        qset1=np.asarray(moire_config.Q_set1, dtype=float),
        qset2=np.asarray(moire_config.Q_set2, dtype=float),
        n_orb=model_config.n_orb,
        harmonics_config=model_config.harmonics_config,
    )
    target = np.asarray(configuration._select_rows(target_all, model_config.band_indices))
    kpoints = np.asarray(moire_config.kpoints, dtype=float)
    if target.shape[0] != len(kpoints):
        raise ValueError(
            "automatic model selection target and band k-point counts differ"
        )
    profile_windows = _automatic_profile_band_windows(
        np.asarray(moire_config.Q_set1, dtype=float),
        np.asarray(moire_config.Q_set2, dtype=float),
        n_orb=tuple(int(value) for value in model_config.n_orb),
        total_bands=int(target.shape[-1]),
    )
    low_primary_count = int(profile_windows.low_band_count)
    high_primary_count = int(profile_windows.high_band_count)
    candidate_dir = output_dir / "candidate_logs"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, Any]] = {}

    def evaluator(
        stage: str,
        orders: FamilyOrders,
        active_families: tuple[str, ...],
    ) -> CandidateScore:
        candidate_moire, candidate_model, specification = (
            _model_and_moire_for_order_candidate(
                moire_config=moire_config,
                model_config=model_config,
                orders=orders,
                active_families=active_families,
            )
        )
        cached = runs.get(specification.name)
        if cached is not None:
            return cached["score"]
        result = _run_model_pipeline(
            candidate_moire,
            candidate_model,
            candidate_dir / f"{specification.name}.log",
            verbose=verbose,
            progress=False,
        )
        if stage == "inter":
            score = _complete_candidate_score(
                name=specification.name,
                orders=orders,
                result=result,
                target_hamiltonians=target,
                model_config=model_config,
                kpoints=kpoints,
                n_folds=selection_config.n_folds,
                low_band_count=low_primary_count,
                high_band_count=high_primary_count,
            )
        else:
            rms_mev, se_mev, maximum_mev = _stage_matrix_candidate_metrics(
                stage=stage,
                model_hamiltonians=np.asarray(result["band_hamiltonians"]),
                target_hamiltonians=target,
                qset1_dim=int(len(moire_config.Q_set1) * moire_config.n_orb1),
                kpoints=kpoints,
                n_folds=selection_config.n_folds,
            )
            score = CandidateScore(
                name=specification.name,
                orders=orders,
                independent_real_parameters=_active_independent_real_parameter_count(
                    result.get("model"),
                    tolerance=model_config.coeff_tol,
                ),
                weighted_rms_mev=rms_mev,
                weighted_rms_se_mev=se_mev,
                weighted_max_mev=maximum_mev,
                mean_subspace_overlap=1.0,
            )
        runs[specification.name] = {
            "score": score,
            "result": result,
            "moire_config": candidate_moire,
            "model_config": candidate_model,
            "specification": specification,
            "stage": stage,
        }
        return score

    staged = run_staged_family_selection(selection_config, evaluator)
    staged_complete_scores = tuple(
        stage.candidates
        for stage in staged.stages
        if stage.stage == "inter"
    )
    candidates_by_name = {
        candidate.name: candidate
        for candidates in staged_complete_scores
        for candidate in candidates
    }
    correction = None
    if staged.selected_orders is not None:
        correction = run_local_order_correction_sweep(
            staged.selected_orders,
            maximum_orders=FamilyOrders(
                kinetic=max(selection_config.order_candidates["kinetic"]),
                intra=max(selection_config.order_candidates["intra"]),
                inter=max(selection_config.order_candidates["inter"]),
            ),
            evaluator=lambda orders: evaluator(
                "inter",
                orders,
                ("kinetic", "intra", "inter"),
            ),
            overlap_target=selection_config.overlap_target,
            overlap_safety_floor=selection_config.overlap_safety_floor,
        )
        candidates_by_name.update(
            {candidate.name: candidate for candidate in correction.candidates}
        )
    balanced_candidates = tuple(
        evaluator("inter", orders, ("kinetic", "intra", "inter"))
        for orders in _balanced_family_order_candidates(
            selection_config.order_candidates
        )
    )
    candidates_by_name.update(
        {candidate.name: candidate for candidate in balanced_candidates}
    )
    candidates = tuple(candidates_by_name.values())
    profiles = select_high_low_profiles(
        candidates,
        overlap_target=selection_config.overlap_target,
        overlap_safety_floor=selection_config.overlap_safety_floor,
        low_se_multiplier=selection_config.low_se_multiplier,
        low_relative_rms_tolerance=selection_config.low_relative_rms_tolerance,
        low_minimum_tolerance_mev=selection_config.low_minimum_tolerance_mev,
    )
    term_group_pruning: dict[str, Any] = {
        "enabled": bool(selection_config.pruning_keep_fractions),
        "candidates": [],
    }
    if (
        selection_config.pruning_keep_fractions
        and profiles.high.selected is not None
    ):
        base_score = profiles.high.selected
        base_run = runs[base_score.name]
        ranked_groups = _rank_closed_term_groups_for_edge(
            model=base_run["result"].get("model"),
            moire_config=base_run["moire_config"],
            target_hamiltonians=target,
            model_config=base_run["model_config"],
            kpoints=kpoints,
        )
        all_groups = tuple(group for group, _importance in ranked_groups)
        all_keys = tuple(key for group in all_groups for key in group)
        keep_sets = _automatic_term_group_keep_sets(
            ranked_groups,
            keep_fractions=selection_config.pruning_keep_fractions,
        )
        for keep_groups in keep_sets:
            keep_key_set = {
                key for group in keep_groups for key in group
            }
            excluded_keys = tuple(
                key for key in all_keys if key not in keep_key_set
            )
            name = f"{base_score.name}__groups{len(keep_groups)}"
            pruned_model_config = replace(
                base_run["model_config"],
                model_selection_excluded_term_keys=excluded_keys,
            )
            pruned_moire_config = copy.copy(base_run["moire_config"])
            pruned_moire_config.output_dir = None
            result = _run_model_pipeline(
                pruned_moire_config,
                pruned_model_config,
                candidate_dir / f"{name}.log",
                verbose=verbose,
                progress=False,
            )
            score = _complete_candidate_score(
                name=name,
                orders=base_score.orders,
                result=result,
                target_hamiltonians=target,
                model_config=pruned_model_config,
                kpoints=kpoints,
                n_folds=selection_config.n_folds,
                low_band_count=low_primary_count,
                high_band_count=high_primary_count,
            )
            result["term_group_pruning"] = {
                "enabled": True,
                "base_candidate": base_score.name,
                "kept_groups": int(len(keep_groups)),
                "excluded_groups": int(len(all_groups) - len(keep_groups)),
                "total_groups": int(len(all_groups)),
            }
            runs[name] = {
                "score": score,
                "result": result,
                "moire_config": pruned_moire_config,
                "model_config": pruned_model_config,
                "specification": None,
                "stage": "term_group_pruning",
            }
            candidates_by_name[name] = score
            term_group_pruning["candidates"].append(
                {
                    "name": name,
                    "kept_groups": int(len(keep_groups)),
                    "excluded_groups": int(len(all_groups) - len(keep_groups)),
                    "score": artifacts._model_selection_candidate_record(score),
                }
            )
        candidates = tuple(candidates_by_name.values())
        profiles = select_high_low_profiles(
            candidates,
            overlap_target=selection_config.overlap_target,
            overlap_safety_floor=selection_config.overlap_safety_floor,
            low_se_multiplier=selection_config.low_se_multiplier,
            low_relative_rms_tolerance=selection_config.low_relative_rms_tolerance,
            low_minimum_tolerance_mev=selection_config.low_minimum_tolerance_mev,
        )
        term_group_pruning["base_candidate"] = base_score.name
        term_group_pruning["total_groups"] = int(len(all_groups))
    nonlinear_frontier: dict[str, Any] = {
        "enabled": False,
        "linear_sources": [],
        "candidates": [],
    }
    nonlinear_profiles = None
    refinement_config = dict(model_config.band_refinement_config or {})
    if bool(refinement_config.get("enabled", False)):
        linear_high_scores = tuple(
            replace(score, selection_scope="high") for score in candidates
        )
        linear_low_scores: list[CandidateScore] = []
        for score in candidates:
            source_run = runs[score.name]
            low_name = f"{score.name}__linear_low"
            low_score = _complete_candidate_score(
                name=low_name,
                orders=score.orders,
                result=source_run["result"],
                target_hamiltonians=target,
                model_config=source_run["model_config"],
                kpoints=kpoints,
                n_folds=selection_config.n_folds,
                solver_family="linear",
                selection_scope="low",
                primary_band_count=low_primary_count,
                low_band_count=low_primary_count,
                high_band_count=high_primary_count,
            )
            runs[low_name] = {
                **source_run,
                "score": low_score,
                "stage": "linear_low_profile",
                "linear_source": score.name,
                "selection_scope": "low",
            }
            linear_low_scores.append(low_score)
        linear_low_profiles = select_high_low_profiles(
            tuple(linear_low_scores),
            overlap_target=selection_config.overlap_target,
            overlap_safety_floor=selection_config.overlap_safety_floor,
            low_se_multiplier=selection_config.low_se_multiplier,
            low_relative_rms_tolerance=selection_config.low_relative_rms_tolerance,
            low_minimum_tolerance_mev=selection_config.low_minimum_tolerance_mev,
        )
        maximum_variables = min(
            3000,
            int(refinement_config.get("max_variables", 3000)),
        )
        high_frontier_scores = _select_nonlinear_frontier_scores(
            candidates,
            profiles=profiles,
            maximum_candidates=int(
                refinement_config.get("frontier_max_candidates", 4)
            ),
            maximum_variables=maximum_variables,
            maximum_orders=FamilyOrders(6, 4, 4),
        )
        high_frontier_scores = tuple(
            replace(score, selection_scope="high")
            for score in high_frontier_scores
        )
        low_frontier_scores = _select_nonlinear_frontier_scores(
            tuple(linear_low_scores),
            profiles=linear_low_profiles,
            maximum_candidates=int(
                refinement_config.get("frontier_max_candidates", 4)
            ),
            maximum_variables=maximum_variables,
            maximum_orders=FamilyOrders(4, 2, 2),
        )
        frontier_scores = tuple(
            {
                score.name: score
                for score in (*high_frontier_scores, *low_frontier_scores)
            }.values()
        )
        linear_profile_runs = _refit_linear_profile_candidates(
            frontier_scores=frontier_scores,
            linear_runs=runs,
            target_hamiltonians=target,
            kpoints=kpoints,
            n_folds=selection_config.n_folds,
            low_primary_count=low_primary_count,
            high_primary_count=high_primary_count,
            candidate_dir=candidate_dir,
            verbose=verbose,
        )
        runs.update(linear_profile_runs)
        linear_profile_scores = tuple(
            run["score"] for run in linear_profile_runs.values()
        )
        nonlinear_runs = _refine_nonlinear_frontier_candidates(
            frontier_scores=linear_profile_scores,
            linear_runs=runs,
            model_config=model_config,
            target_hamiltonians=target,
            kpoints=kpoints,
            n_folds=selection_config.n_folds,
            low_primary_count=low_primary_count,
            high_primary_count=high_primary_count,
            candidate_dir=candidate_dir,
            verbose=verbose,
        )
        runs.update(nonlinear_runs)
        nonlinear_score_list = [
            run["score"] for run in nonlinear_runs.values()
        ]
        for name, nonlinear_run in nonlinear_runs.items():
            score = nonlinear_run["score"]
            if score.selection_scope != "high":
                continue
            low_name = f"{name}__low_score"
            low_score = _complete_candidate_score(
                name=low_name,
                orders=score.orders,
                result=nonlinear_run["result"],
                target_hamiltonians=target,
                model_config=nonlinear_run["model_config"],
                kpoints=kpoints,
                n_folds=selection_config.n_folds,
                solver_family="nonlinear",
                selection_scope="low",
                primary_band_count=low_primary_count,
                low_band_count=low_primary_count,
                high_band_count=high_primary_count,
            )
            runs[low_name] = {
                **nonlinear_run,
                "score": low_score,
                "stage": "nonlinear_high_low_rescore",
                "nonlinear_source": name,
                "selection_scope": "low",
            }
            nonlinear_score_list.append(low_score)
        nonlinear_scores = tuple(nonlinear_score_list)
        if nonlinear_scores:
            four_profile_candidates = _four_profile_candidate_pool(
                linear_high_scores=linear_high_scores,
                linear_low_scores=tuple(linear_low_scores),
                weighted_linear_scores=linear_profile_scores,
                nonlinear_scores=nonlinear_scores,
            )
            four_profiles = select_four_model_profiles(
                four_profile_candidates,
                overlap_target=selection_config.overlap_target,
                overlap_safety_floor=selection_config.overlap_safety_floor,
                low_se_multiplier=selection_config.low_se_multiplier,
                low_relative_rms_tolerance=selection_config.low_relative_rms_tolerance,
                low_minimum_tolerance_mev=selection_config.low_minimum_tolerance_mev,
            )
            profiles = four_profiles.linear
            nonlinear_profiles = _complete_optional_profile_pair(
                four_profiles.nonlinear
            )
        nonlinear_frontier = {
            "enabled": True,
            "maximum_candidates": int(
                refinement_config.get("frontier_max_candidates", 4)
            ),
            "maximum_variables": maximum_variables,
            "linear_sources": [score.name for score in frontier_scores],
            "weighted_linear_candidates": [
                score.name for score in linear_profile_scores
            ],
            "linear_high_sources": [
                score.name for score in high_frontier_scores
            ],
            "linear_low_sources": [
                score.name for score in low_frontier_scores
            ],
            "candidates": [
                artifacts._model_selection_candidate_record(score)
                for score in nonlinear_scores
            ],
        }
    if progress:
        selected = staged.selected_orders
        _progress_line(
            f"automatic family-order scan selected staged orders: {selected}",
            enabled=True,
        )
    return {
        "staged_selection": staged,
        "profiles": profiles,
        "profile_families": {
            "linear": profiles,
            "nonlinear": nonlinear_profiles,
        },
        "nonlinear_frontier": nonlinear_frontier,
        "correction_sweep": correction,
        "term_group_pruning": term_group_pruning,
        "profile_band_windows": {
            "low_band_count": int(profile_windows.low_band_count),
            "high_band_count": int(profile_windows.high_band_count),
            "q_shell_count": 2,
            "layer_q_counts": [int(value) for value in profile_windows.layer_q_counts],
        },
        "runs": runs,
        "target_hamiltonians": target,
    }



def _execute_fit_candidate(
    *, candidate: Mapping[str, Any], name: str, indices: list[int],
    record: dict[str, Any], moire_config: MoireConfig, model_config: configuration.ConfiguredModel,
    kpoints_all: np.ndarray, heff_list: np.ndarray, heff_selected: np.ndarray,
    original_heff_selected: np.ndarray, candidate_log_dir: Path, verbose: bool,
) -> tuple[dict[str, Any], MoireConfig, configuration.ConfiguredModel, Path]:
    candidate_model = _model_config_for_fit_candidate(
        model_config,
        {**candidate, "indices": indices},
        disable_refinement=True,
    )
    candidate_moire = _moire_config_for_fit_indices(
        moire_config,
        candidate_model,
        kpoints_all=kpoints_all,
        heff_list=heff_list,
        indices=indices,
    )
    candidate_log = candidate_log_dir / f"{name}.log"
    result = _run_model_pipeline(
        candidate_moire,
        candidate_model,
        candidate_log,
        verbose=verbose,
        progress=False,
    )
    eigvals = result["eigvals"][0] if isinstance(result["eigvals"], tuple) else result["eigvals"]
    comparison = validation.compare_bands(eigvals, heff_selected, band_slice=model_config.band_slice)
    plot_comparison = validation.compare_bands_for_plot(
        eigvals,
        heff_selected,
        band_slice=model_config.band_slice,
        plot_config=model_config.band_plot_config,
    )
    secondary_plot_comparison = validation.compare_bands_for_plot(
        eigvals,
        original_heff_selected,
        band_slice=model_config.band_slice,
        plot_config=model_config.band_plot_config,
    )
    record.update(
        {
            "status": "ok",
            "reference": "current_heff_support_mask",
            "all_rms_mev": float(comparison["rms_error_mev"]),
            "all_max_mev": float(comparison["max_abs_error_mev"]),
            "plot_rms_mev": float(plot_comparison["rms_error_mev"]),
            "plot_max_mev": float(plot_comparison["max_abs_error_mev"]),
            "secondary_original_heff": {
                "reference": "original_heff",
                "plot_rms_mev": float(secondary_plot_comparison["rms_error_mev"]),
                "plot_max_mev": float(secondary_plot_comparison["max_abs_error_mev"]),
            },
            "log": str(candidate_log.resolve()),
        }
    )
    return (result, candidate_moire, candidate_model, candidate_log)



def _run_auto_low_energy_fit_candidate_scan(
    *,
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    output_dir: Path,
    log_path: Path,
    verbose: bool,
    progress: bool,
) -> tuple[dict[str, Any], MoireConfig, configuration.ConfiguredModel] | None:
    metadata = dict(model_config.fit_selection_metadata or {})
    if str(metadata.get("mode", "")).strip().lower() != "auto_low_energy":
        return None
    raw_candidates = metadata.get("fit_candidate_sets", [])
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)) or len(raw_candidates) <= 1:
        return None
    fit_cfg = model_config.raw.get("fit", {})
    if isinstance(fit_cfg, Mapping) and fit_cfg.get("candidate_scan") is False:
        return None

    candidates = [dict(item) for item in raw_candidates if isinstance(item, Mapping)]
    if len(candidates) <= 1:
        return None
    max_candidates = int(fit_cfg.get("max_fit_candidates", 4)) if isinstance(fit_cfg, Mapping) else 4
    candidates = candidates[: max(1, max_candidates)]
    if len(candidates) <= 1:
        return None

    kpoints_all = configuration._load_kpoints(model_config)
    original_heff_list = configuration._load_heff_in_model_basis(model_config)
    heff_list = validation._current_heff_support_hamiltonians(
        original_heff_list,
        qset1=np.asarray(moire_config.Q_set1, dtype=float),
        qset2=np.asarray(moire_config.Q_set2, dtype=float),
        n_orb=model_config.n_orb,
        harmonics_config=model_config.harmonics_config,
    )
    heff_selected = np.linalg.eigvalsh(configuration._select_rows(heff_list, model_config.band_indices))
    if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
        original_heff_eig = np.load(model_config.heff_eig_file)
        original_heff_selected = configuration._select_rows(original_heff_eig, model_config.band_indices)
    else:
        original_heff_selected = np.linalg.eigvalsh(
            configuration._select_rows(original_heff_list, model_config.band_indices)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_log_dir = output_dir / "candidate_logs"
    candidate_log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    runs: dict[str, tuple[dict[str, Any], MoireConfig, configuration.ConfiguredModel, Path]] = {}
    if progress:
        print(f"[kp model] scanning {len(candidates)} auto fit-k candidates ...", flush=True)

    for candidate in candidates:
        name = str(candidate.get("name", f"candidate_{len(records)}"))
        indices = [int(idx) for idx in candidate.get("indices", [])]
        record = {
            "name": name,
            "indices": indices,
            "fit_count": int(len(indices)),
            "reason": candidate.get("reason"),
            "status": "failed",
        }
        try:
            runs[name] = _execute_fit_candidate(
                candidate=candidate, name=name, indices=indices, record=record,
                moire_config=moire_config, model_config=model_config,
                kpoints_all=kpoints_all, heff_list=heff_list, heff_selected=heff_selected,
                original_heff_selected=original_heff_selected,
                candidate_log_dir=candidate_log_dir, verbose=verbose,
            )
        except CandidateRejected as exc:
            record.update({"status": "failed", "error": str(exc)})
        records.append(record)

    initial_ok_records = [row for row in records if row.get("status") == "ok"]
    if initial_ok_records and not bool(fit_cfg.get("disable_residual_augmented_fit", False)):
        preliminary_record, _preliminary_report = fit_sampling._choose_auto_fit_candidate_record(initial_ok_records)
        preliminary_run = runs.get(str(preliminary_record.get("name", "")))
        if preliminary_run is not None:
            preliminary_result = preliminary_run[0]
            preliminary_eigvals = (
                preliminary_result["eigvals"][0]
                if isinstance(preliminary_result["eigvals"], tuple)
                else preliminary_result["eigvals"]
            )
            residual_scores = validation._band_residual_scores_for_plot(
                preliminary_eigvals,
                heff_selected,
                band_slice=model_config.band_slice,
                plot_config=model_config.band_plot_config,
            )
            residual_candidates = fit_sampling._auto_low_energy_fit_candidate_sets(
                kpoints_all,
                initial_points=int(metadata.get("initial_points", fit_cfg.get("initial_points", 2))),
                max_points=int(fit_cfg.get("max_fit_points", fit_cfg.get("max_points", 7))),
                initial_indices=metadata.get("initial_indices"),
                residual_scores=residual_scores,
            )
            seen_candidate_keys = {
                (str(row.get("name", "")), tuple(int(idx) for idx in row.get("indices", [])))
                for row in records
            }
            max_residual_candidates = int(fit_cfg.get("max_residual_fit_candidates", 2))
            residual_added = 0
            for candidate in residual_candidates:
                if not str(candidate.get("name", "")).startswith("residual_augmented"):
                    continue
                key = (str(candidate.get("name", "")), tuple(int(idx) for idx in candidate.get("indices", [])))
                if key in seen_candidate_keys:
                    continue
                if residual_added >= max_residual_candidates:
                    break
                candidates.append(dict(candidate))
                seen_candidate_keys.add(key)
                residual_added += 1
                name = str(candidate.get("name", f"candidate_{len(records)}"))
                indices = [int(idx) for idx in candidate.get("indices", [])]
                record = {
                    "name": name,
                    "indices": indices,
                    "fit_count": int(len(indices)),
                    "reason": candidate.get("reason"),
                    "status": "failed",
                    "candidate_source": "residual_augmented",
                }
                try:
                    runs[name] = _execute_fit_candidate(
                        candidate=candidate, name=name, indices=indices, record=record,
                        moire_config=moire_config, model_config=model_config,
                        kpoints_all=kpoints_all, heff_list=heff_list, heff_selected=heff_selected,
                        original_heff_selected=original_heff_selected,
                        candidate_log_dir=candidate_log_dir, verbose=verbose,
                    )
                except CandidateRejected as exc:
                    record.update({"status": "failed", "error": str(exc)})
                records.append(record)

    ok_records = [row for row in records if row.get("status") == "ok"]
    if not ok_records:
        raise RuntimeError(f"auto_low_energy fit candidate scan failed for all candidates: {records!r}")
    selected_record, report = fit_sampling._choose_auto_fit_candidate_record(ok_records)
    report = {
        **report,
        "enabled": True,
        "mode": "prefit_full_model_scan",
        "reference": "current_heff_support_mask",
        "secondary_reference": "original_heff",
        "candidates": records,
    }
    selected_name = str(selected_record["name"])
    selected_candidate = next(item for item in candidates if str(item.get("name")) == selected_name)
    selected_model = _model_config_for_fit_candidate(
        model_config,
        selected_candidate,
        scan_report=report,
        disable_refinement=False,
    )
    selected_moire = _moire_config_for_fit_indices(
        moire_config,
        selected_model,
        kpoints_all=kpoints_all,
        heff_list=heff_list,
        indices=selected_record["indices"],
    )
    if progress:
        print(
            "[kp model] selected auto fit-k candidate: "
            f"{selected_name} indices={selected_record['indices']} "
            f"plot RMS={float(selected_record['plot_rms_mev']):.3f} meV",
            flush=True,
        )
    selected_results = _run_model_pipeline(selected_moire, selected_model, log_path, verbose=verbose, progress=progress)
    selected_results["auto_fit_candidate_scan"] = report
    return selected_results, selected_moire, selected_model



def _selected_profile_pipeline_result(
    *,
    selected_run: Mapping[str, Any],
    selected_moire: Any,
    selected_model: configuration.ConfiguredModel,
    profile_log: Path,
    verbose: bool,
    progress: bool,
) -> dict[str, Any]:
    cached = selected_run.get("result")
    if isinstance(cached, dict):
        return cached
    return _run_model_pipeline(
        selected_moire,
        selected_model,
        profile_log,
        verbose=verbose,
        progress=progress,
    )
