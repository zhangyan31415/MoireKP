from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from .core import (
    ContinuumModelBuilder,
    MoireConfig,
)
from . import configuration


def compare_bands(
    model_eigvals: np.ndarray,
    heff_eigvals: np.ndarray,
    *,
    band_slice: Sequence[int] | None = None,
) -> dict[str, float | int | list[int] | None]:
    model = np.asarray(model_eigvals, dtype=float)
    heff = np.asarray(heff_eigvals)
    if heff.ndim == 3:
        heff = np.linalg.eigvalsh(heff)
    heff = np.asarray(heff, dtype=float)
    if model.ndim != 2 or heff.ndim != 2:
        raise ValueError(f"model/heff eigvals must be 2D after eigensolve, got {model.shape} and {heff.shape}")
    if model.shape[0] != heff.shape[0]:
        raise ValueError(f"k-point counts differ: model={model.shape[0]}, heff={heff.shape[0]}")
    nbands = min(model.shape[1], heff.shape[1])
    start = 0
    stop = nbands
    if band_slice is not None:
        if len(band_slice) != 2:
            raise ValueError(f"band_slice must be [start, stop], got {band_slice}")
        start, stop = int(band_slice[0]), int(band_slice[1])
    model_sel = np.sort(model, axis=1)[:, start:stop]
    heff_sel = np.sort(heff, axis=1)[:, start:stop]
    if model_sel.shape != heff_sel.shape:
        raise ValueError(f"selected band shapes differ: model={model_sel.shape}, heff={heff_sel.shape}")
    delta = model_sel - heff_sel
    rms = float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    return {
        "num_kpoints": int(delta.shape[0]),
        "num_bands": int(delta.shape[1]),
        "band_slice": [start, stop],
        "rms_error": rms,
        "max_abs_error": max_abs,
        "rms_error_mev": 1000.0 * rms,
        "max_abs_error_mev": 1000.0 * max_abs,
    }



def matrix_residual(model_hamiltonians: np.ndarray, heff_hamiltonians: np.ndarray) -> dict[str, Any]:
    model = np.asarray(model_hamiltonians, dtype=complex)
    heff = np.asarray(heff_hamiltonians, dtype=complex)
    if model.ndim == 2:
        model = model[None, :, :]
    if heff.ndim == 2:
        heff = heff[None, :, :]
    if model.shape != heff.shape:
        raise ValueError(f"model/heff Hamiltonian arrays must have same shape, got {model.shape} and {heff.shape}")
    residuals = []
    for H_model, H_heff in zip(model, heff):
        denom = float(np.linalg.norm(H_heff))
        if denom == 0.0:
            denom = 1.0
        residuals.append(float(np.linalg.norm(H_model - H_heff) / denom))
    arr = np.asarray(residuals, dtype=float)
    return {
        "num_kpoints": int(arr.size),
        "rms_residual": float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0,
        "max_residual": float(np.max(arr)) if arr.size else 0.0,
        "residuals": arr.tolist(),
    }



def _selected_harmonic_support_mask(
    qset1: np.ndarray,
    qset2: np.ndarray,
    *,
    n_orb: tuple[int, int],
    current_counts: Mapping[str, int],
    tol: float = 1.0e-6,
) -> np.ndarray:
    from . import harmonic_selection

    rows = harmonic_selection._model_row_metadata_for_harmonic_scan(qset1, qset2, n_orb)
    shell_norms = harmonic_selection._harmonic_shell_norms_from_qsets(qset1, qset2, tol=tol)
    q = np.asarray([np.asarray(row["q"], dtype=float) for row in rows], dtype=float)
    layers = np.asarray([int(row["layer"]) for row in rows], dtype=np.int16)
    norms = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    zero = norms <= float(tol)
    same_layer = layers[:, None] == layers[None, :]
    intra_count = int(current_counts.get("intra", 0))
    inter_count = int(current_counts.get("inter", 0))
    intra_positive_shells = max(0, intra_count - 1)
    has_inter_zero = bool(np.any(np.logical_and(~same_layer, zero)))
    inter_positive_shells = max(0, inter_count - int(has_inter_zero))
    intra_shell = harmonic_selection._shell_index_matrix(norms, shell_norms.get("intra", []))
    inter_shell = harmonic_selection._shell_index_matrix(norms, shell_norms.get("inter", []))
    onsite_or_local = np.logical_and(same_layer, zero)
    intra = (
        np.logical_and.reduce((same_layer, ~zero, intra_shell <= intra_positive_shells))
        if intra_positive_shells > 0
        else np.zeros(norms.shape, dtype=bool)
    )
    inter = (
        np.logical_and.reduce((~same_layer, ~zero, inter_shell <= inter_positive_shells))
        if inter_positive_shells > 0
        else np.zeros(norms.shape, dtype=bool)
    )
    inter_zero = np.logical_and.reduce((~same_layer, zero)) if inter_count > 0 else np.zeros(norms.shape, dtype=bool)
    return np.logical_or.reduce((onsite_or_local, intra, inter_zero, inter))



def _current_heff_support_hamiltonians(
    heff_hamiltonians: np.ndarray,
    *,
    qset1: np.ndarray,
    qset2: np.ndarray,
    n_orb: tuple[int, int],
    harmonics_config: Mapping[str, Any],
) -> np.ndarray:
    """Project Heff onto the support represented by the configured harmonics."""
    from . import harmonic_selection

    heff = np.asarray(heff_hamiltonians, dtype=np.complex128)
    if heff.ndim != 3 or heff.shape[-1] != heff.shape[-2]:
        raise ValueError(f"Heff support projection expects (Nk,dim,dim), got {heff.shape}")
    mask = _selected_harmonic_support_mask(
        np.asarray(qset1, dtype=float),
        np.asarray(qset2, dtype=float),
        n_orb=(int(n_orb[0]), int(n_orb[1])),
        current_counts=harmonic_selection._resolved_harmonic_count_limits(harmonics_config),
    )
    if mask.shape != heff.shape[-2:]:
        raise ValueError(
            f"Heff support mask shape {mask.shape} does not match Hamiltonian shape {heff.shape[-2:]}"
        )
    supported = heff * mask[None, :, :]
    return 0.5 * (supported + np.swapaxes(supported.conj(), -1, -2))



def _retarget_auto_low_energy_windows_to_current_support(
    model_config: configuration.ConfiguredModel,
    support_heff: np.ndarray,
) -> None:
    """Finalize auto-low-energy windows after canonical Q/support is available."""
    from . import fit_sampling

    target_eigvals = np.linalg.eigvalsh(np.asarray(support_heff, dtype=np.complex128))
    fit_cfg = model_config.raw.get("fit", {})
    if not isinstance(fit_cfg, Mapping):
        fit_cfg = {}
    target_bands = str(
        model_config.raw.get("model", {}).get("target_bands", "top")
    ).strip().lower()
    n_primary = int(sum(model_config.n_orb))
    gap_tolerance_mev = float(fit_cfg.get("gap_tolerance_mev", 0.1))
    windows = fit_sampling._auto_low_energy_windows(
        target_eigvals,
        n_primary=n_primary,
        target_bands=target_bands,
        gap_tolerance_mev=gap_tolerance_mev,
        max_expanded=int(fit_cfg.get("max_expanded_windows", 2)),
    )
    dim = int(target_eigvals.shape[1])
    weighted_fit_bands = int(
        fit_cfg.get(
            "weighted_fit_bands",
            fit_cfg.get("fit_bands", min(dim, max(n_primary, 10))),
        )
    )
    weighted_fit_window = fit_sampling._auto_low_energy_window_record(
        target_eigvals,
        n_bands=weighted_fit_bands,
        target_bands=target_bands,
        role="weighted_fit",
        gap_tolerance_mev=gap_tolerance_mev,
        use_for_loss=True,
    )
    windows["weighted_fit"] = weighted_fit_window
    for key in ("primary", "weighted_fit"):
        if isinstance(windows.get(key), Mapping):
            windows[key] = {**dict(windows[key]), "reference": "current_heff_support_mask"}
    windows["expanded"] = [
        {**dict(window), "reference": "current_heff_support_mask"}
        for window in windows.get("expanded", [])
        if isinstance(window, Mapping)
    ]
    windows["reference"] = "current_heff_support_mask"

    refinement = dict(model_config.band_refinement_config or {})
    weighted_slice = list(windows["weighted_fit"]["band_slice"])
    refinement["auto_windows"] = windows
    refinement["band_slice"] = weighted_slice
    refinement["reference"] = "current_heff_support_mask"
    for loss_name in ("subspace_loss", "low_subspace_matrix_loss"):
        loss = refinement.get(loss_name)
        if isinstance(loss, Mapping):
            refinement[loss_name] = {**dict(loss), "band_slice": weighted_slice}
    model_config.band_refinement_config = refinement



def _refinement_target_hamiltonians(
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    raw_cfg: Mapping[str, Any],
    row_selector: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    original = configuration._select_rows(
        configuration._load_heff_in_model_basis(model_config),
        row_selector,
    )
    fit_mode = str(
        getattr(model_config, "fit_selection_metadata", {}).get("mode", "")
    ).strip().lower()
    requested_reference = str(raw_cfg.get("reference", "")).strip().lower()
    if requested_reference not in {"", "original_heff", "current_heff_support_mask"}:
        raise ValueError(
            "fit.refine_bands.reference must be 'original_heff' or "
            "'current_heff_support_mask'"
        )
    response_semantics = str(
        model_config.response_semantics
    ).strip().lower()
    use_current_support = requested_reference == "current_heff_support_mask" or (
        requested_reference != "original_heff"
        and (
            str(raw_cfg.get("mode", "")).strip().lower() == "auto_low_energy"
            or fit_mode == "auto_low_energy"
            or response_semantics == "complete_linear"
        )
    )
    if use_current_support:
        target = _current_heff_support_hamiltonians(
            original,
            qset1=np.asarray(moire_config.Q_set1, dtype=float),
            qset2=np.asarray(moire_config.Q_set2, dtype=float),
            n_orb=tuple(int(value) for value in model_config.n_orb),
            harmonics_config=getattr(model_config, "harmonics_config", {}) or {},
        )
        return target, np.linalg.eigvalsh(target), "current_heff_support_mask"

    heff_eig_file = getattr(model_config, "heff_eig_file", None)
    if heff_eig_file is not None and heff_eig_file.exists():
        eigvals = configuration._select_rows(np.load(heff_eig_file), row_selector)
    else:
        eigvals = np.linalg.eigvalsh(original)
    return np.asarray(original), np.asarray(eigvals), "original_heff"



def _resolve_plot_band_slice(
    *,
    nbands: int,
    metric_band_slice: Sequence[int] | None,
    plot_config: Mapping[str, Any],
) -> tuple[int, int]:
    if plot_config.get("band_slice") is not None:
        values = configuration._as_int_list(plot_config.get("band_slice"), name="bands.plot.band_slice")
        if len(values) != 2:
            raise ValueError(f"bands.plot.band_slice must be [start, stop], got {values}")
        start, stop = values
    elif plot_config.get("bottom_bands") is not None:
        count = int(plot_config["bottom_bands"])
        if count <= 0:
            raise ValueError(f"bands.plot.bottom_bands must be positive, got {count}")
        start, stop = 0, min(int(nbands), count)
    elif plot_config.get("top_bands") is not None:
        count = int(plot_config["top_bands"])
        if count <= 0:
            raise ValueError(f"bands.plot.top_bands must be positive, got {count}")
        start, stop = max(0, int(nbands) - count), int(nbands)
    elif metric_band_slice is not None:
        if len(metric_band_slice) != 2:
            raise ValueError(f"band_slice must be [start, stop], got {metric_band_slice}")
        start, stop = int(metric_band_slice[0]), int(metric_band_slice[1])
    else:
        start, stop = 0, int(nbands)

    if start < 0 or stop > nbands or start >= stop:
        raise ValueError(f"Invalid plot band slice [{start}, {stop}] for nbands={nbands}")
    return start, stop



def compare_bands_for_plot(
    model_eigvals: np.ndarray,
    heff_eigvals: np.ndarray,
    *,
    band_slice: Sequence[int] | None = None,
    plot_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = np.asarray(model_eigvals, dtype=float)
    heff = np.asarray(heff_eigvals)
    if heff.ndim == 3:
        heff = np.linalg.eigvalsh(heff)
    heff = np.asarray(heff, dtype=float)
    if model.ndim != 2 or heff.ndim != 2:
        raise ValueError(f"model/heff eigvals must be 2D after eigensolve, got {model.shape} and {heff.shape}")
    if model.shape[0] != heff.shape[0]:
        raise ValueError(f"k-point counts differ: model={model.shape[0]}, heff={heff.shape[0]}")

    nbands = min(model.shape[1], heff.shape[1])
    plot_options = dict(plot_config or {})
    start, stop = _resolve_plot_band_slice(nbands=nbands, metric_band_slice=band_slice, plot_config=plot_options)
    model_sel = np.sort(model, axis=1)[:, start:stop]
    heff_sel = np.sort(heff, axis=1)[:, start:stop]
    align = str(plot_options.get("align", "none")).lower()
    model_ref = 0.0
    heff_ref = 0.0
    if align in {"top", "top_band", "top-band"}:
        model_ref = float(np.max(model_sel[:, -1]))
        heff_ref = float(np.max(heff_sel[:, -1]))
        model_sel = model_sel - model_ref
        heff_sel = heff_sel - heff_ref
    elif align in {"bottom", "bottom_band", "bottom-band"}:
        model_ref = float(np.min(model_sel[:, 0]))
        heff_ref = float(np.min(heff_sel[:, 0]))
        model_sel = model_sel - model_ref
        heff_sel = heff_sel - heff_ref
    elif align not in {"none", "false", "0"}:
        raise ValueError(f"Unsupported bands.plot.align={plot_options.get('align')!r}; expected 'none', 'top', or 'bottom'")

    delta = model_sel - heff_sel
    rms = float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    payload: dict[str, Any] = {
        "num_kpoints": int(delta.shape[0]),
        "num_bands": int(delta.shape[1]),
        "band_slice": [start, stop],
        "align": align,
        "rms_error": rms,
        "max_abs_error": max_abs,
        "rms_error_mev": 1000.0 * rms,
        "max_abs_error_mev": 1000.0 * max_abs,
    }
    if align in {"top", "top_band", "top-band", "bottom", "bottom_band", "bottom-band"}:
        payload.update(
            {
                "reference_kind": "top" if align.startswith("top") else "bottom",
                "heff_reference_eV": heff_ref,
                "model_reference_eV": model_ref,
                "model_alignment_shift_eV": model_ref - heff_ref,
                "model_alignment_shift_meV": 1000.0 * (model_ref - heff_ref),
                "aligned_rms_error_meV": 1000.0 * rms,
                "aligned_max_abs_error_meV": 1000.0 * max_abs,
            }
        )
    return payload



def _band_residual_scores_for_plot(
    model_eigvals: np.ndarray,
    heff_eigvals: np.ndarray,
    *,
    band_slice: Sequence[int] | None = None,
    plot_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    model = np.asarray(model_eigvals, dtype=float)
    heff = np.asarray(heff_eigvals)
    if heff.ndim == 3:
        heff = np.linalg.eigvalsh(heff)
    heff = np.asarray(heff, dtype=float)
    if model.ndim != 2 or heff.ndim != 2:
        raise ValueError(f"model/heff eigvals must be 2D after eigensolve, got {model.shape} and {heff.shape}")
    if model.shape[0] != heff.shape[0]:
        raise ValueError(f"k-point counts differ: model={model.shape[0]}, heff={heff.shape[0]}")
    nbands = min(model.shape[1], heff.shape[1])
    plot_options = dict(plot_config or {})
    start, stop = _resolve_plot_band_slice(nbands=nbands, metric_band_slice=band_slice, plot_config=plot_options)
    model_sel = np.sort(model, axis=1)[:, start:stop]
    heff_sel = np.sort(heff, axis=1)[:, start:stop]
    align = str(plot_options.get("align", "none")).lower()
    if align in {"top", "top_band", "top-band"}:
        model_sel = model_sel - float(np.max(model_sel[:, -1]))
        heff_sel = heff_sel - float(np.max(heff_sel[:, -1]))
    elif align in {"bottom", "bottom_band", "bottom-band"}:
        model_sel = model_sel - float(np.min(model_sel[:, 0]))
        heff_sel = heff_sel - float(np.min(heff_sel[:, 0]))
    elif align not in {"none", "false", "0"}:
        raise ValueError(f"Unsupported bands.plot.align={plot_options.get('align')!r}; expected 'none', 'top', or 'bottom'")
    delta = model_sel - heff_sel
    return np.sqrt(np.mean(delta**2, axis=1)) if delta.size else np.zeros(model.shape[0], dtype=float)



def _all_band_plot_config(plot_config: Mapping[str, Any] | None) -> dict[str, Any]:
    options = dict(plot_config or {})
    for key in ("band_slice", "top_bands", "bottom_bands", "ylim"):
        options.pop(key, None)
    options["plot_all_bands"] = True
    options["show_metrics"] = False
    options["legend_outside"] = True
    return options



def _window_band_plot_config(
    plot_config: Mapping[str, Any] | None,
    *,
    target_bands: str = "top",
    default_bands: int = 10,
) -> dict[str, Any]:
    if default_bands <= 0:
        raise ValueError(f"default_bands must be positive, got {default_bands}")
    options = dict(plot_config or {})
    options.setdefault("show_metrics", False)
    options.setdefault("legend_outside", True)
    options.setdefault("plot_all_bands", True)
    if options.get("band_slice") is not None:
        return options
    target = str(target_bands or "top").strip().lower()
    if options.get("top_bands") is not None:
        options["top_bands"] = int(options["top_bands"])
        return options
    if options.get("bottom_bands") is not None:
        options["bottom_bands"] = int(options["bottom_bands"])
        return options
    if target == "bottom":
        options["bottom_bands"] = int(default_bands)
    else:
        options["top_bands"] = int(default_bands)
    return options



def _selected_index_positions(model_config: configuration.ConfiguredModel, count: int) -> tuple[list[int], list[int]]:
    selected = list(model_config.band_indices or list(range(count)))
    fit_positions = [selected.index(int(idx)) for idx in model_config.fit_indices if int(idx) in selected]
    all_positions = list(range(len(selected)))
    holdout_positions = [idx for idx in all_positions if idx not in fit_positions]
    return fit_positions, holdout_positions



def _compute_validation_outputs(
    *,
    results: Mapping[str, Any],
    model_config: configuration.ConfiguredModel,
    moire_config: MoireConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from . import artifacts

    validations: dict[str, Any] = {}
    summary: dict[str, Any] = {"validation_incomplete": False, "missing": []}
    strict = _validation_strict_enabled(model_config.validation_config)
    band_level_only = bool(model_config.validation_config.get("band_level_only", False))
    model = results.get("model")

    if model is None:
        validations["matrix_residuals"] = {"available": False, "reason": "model object missing from pipeline results"}
        validations["block_residuals"] = {"available": False, "reason": "block residual validation is not implemented in the current configured pipeline"}
        validations["symmetry_residuals"] = {"available": False, "reason": "model object missing from pipeline results"}
        if bool(model_config.validation_config.get("allow_missing_wavefunction_overlap", False)):
            validations["wavefunction_overlap"] = {
                "available": False,
                "reason": str(model_config.validation_config.get("reason", "explicitly waived by config")),
            }
        else:
            validations["wavefunction_overlap"] = {
                "available": False,
                "reason": "wavefunction/projector overlap source data is not available in current configured pipeline",
            }
        validations["orthogonalization"] = {"available": True, "diagnostics": artifacts._json_safe(results.get("diagnostics", {}))}
        for name, payload in validations.items():
            if not bool(payload.get("available", False)):
                summary["validation_incomplete"] = True
                summary["missing"].append({"name": name, "reason": payload.get("reason", "unknown")})
        if strict:
            for name, payload in validations.items():
                if not bool(payload.get("available", False)):
                    if band_level_only and name in {"block_residuals", "symmetry_residuals", "wavefunction_overlap"}:
                        continue
                    if name == "wavefunction_overlap" and bool(model_config.validation_config.get("allow_missing_wavefunction_overlap", False)):
                        continue
                    raise ValueError(
                        f"validation.strict=true requires {name}; current configured pipeline reported unavailable: "
                        f"{payload.get('reason', 'unknown')}"
                    )
        return validations, summary

    cached_band_hamiltonians = results.get("band_hamiltonians")
    validation_hamiltonians: np.ndarray | None = None
    hermiticity_residuals: list[float] = []

    def get_validation_hamiltonians() -> np.ndarray:
        nonlocal validation_hamiltonians, hermiticity_residuals
        if validation_hamiltonians is not None:
            return validation_hamiltonians
        if cached_band_hamiltonians is not None:
            cached = np.asarray(cached_band_hamiltonians, dtype=complex)
            expected_kpoints = len(np.asarray(moire_config.kpoints, dtype=float))
            if cached.ndim != 3 or cached.shape[0] != expected_kpoints or cached.shape[1] != cached.shape[2]:
                raise ValueError(
                    "band-stage Hamiltonians must have shape (Nk, dim, dim), "
                    f"got {cached.shape} for Nk={expected_kpoints}"
                )
            validation_hamiltonians = cached
            hermiticity_residuals = []
            for hamiltonian in validation_hamiltonians:
                denom = float(np.linalg.norm(hamiltonian)) or 1.0
                hermiticity_residuals.append(
                    float(np.linalg.norm(hamiltonian - hamiltonian.conj().T) / denom)
                )
            return validation_hamiltonians
        raise ValueError("validation requires Hamiltonians from the complete response band stage")

    if model_config.compare_to_heff:
        h_model = get_validation_hamiltonians()
        heff_all = configuration._load_heff_in_model_basis(model_config)
        heff_selected = configuration._select_rows(heff_all, model_config.band_indices)
        residual = matrix_residual(h_model, heff_selected)
        fit_positions, holdout_positions = _selected_index_positions(model_config, len(residual["residuals"]))
        residuals_arr = np.asarray(residual["residuals"], dtype=float)
        residual["fit_rms_residual"] = float(np.sqrt(np.mean(residuals_arr[fit_positions] ** 2))) if fit_positions else None
        residual["holdout_rms_residual"] = float(np.sqrt(np.mean(residuals_arr[holdout_positions] ** 2))) if holdout_positions else None
        residual["fit_positions"] = fit_positions
        residual["holdout_positions"] = holdout_positions
        residual["hermiticity_rms_residual"] = float(np.sqrt(np.mean(np.asarray(hermiticity_residuals, dtype=float) ** 2)))
        residual["hermiticity_max_residual"] = float(np.max(hermiticity_residuals)) if hermiticity_residuals else 0.0
        validations["matrix_residuals"] = {"available": True, **residual}
    else:
        validations["matrix_residuals"] = {"available": False, "reason": "compare_to_heff_disabled"}
    validations["block_residuals"] = {
        "available": False,
        "reason": "block residual validation is not implemented in the current configured pipeline",
    }

    operation_registry = artifacts._build_operation_registry(model_config)
    symmetry_gen = moire_config.symmetry_gen
    h_model = get_validation_hamiltonians()
    kpoints = np.asarray(moire_config.kpoints, dtype=float)
    rows = []
    for operation in operation_registry:
        k_map = operation.get("k_map", {})
        canonical = str(operation.get("canonical_operation", ""))
        antiunitary = bool(operation.get("antiunitary", False))
        D = symmetry_gen.get_operator(canonical, None)
        matched = []
        residuals = []
        for idx, kval in enumerate(kpoints):
            k_src = ContinuumModelBuilder._apply_k_map_to_vector(kval, {"name": canonical, "k_map": k_map})
            distances = np.linalg.norm(kpoints - np.asarray(k_src, dtype=float), axis=1)
            j = int(np.argmin(distances))
            if float(distances[j]) > 1.0e-8:
                continue
            source_h = h_model[j].conj() if antiunitary else h_model[j]
            cov = np.asarray(D) @ source_h @ np.asarray(D).conj().T
            denom = float(np.linalg.norm(h_model[idx])) or 1.0
            residuals.append(float(np.linalg.norm(h_model[idx] - cov) / denom))
            matched.append([int(idx), int(j)])
        if residuals:
            rows.append(
                {
                    **operation,
                    "available": True,
                    "matched_pairs": matched,
                    "covariance_rms_residual": float(np.sqrt(np.mean(np.asarray(residuals) ** 2))),
                    "covariance_max_residual": float(np.max(residuals)),
                }
            )
    if rows:
        validations["symmetry_residuals"] = {"available": True, "operations": rows}
    else:
        validations["symmetry_residuals"] = {"available": False, "reason": "no_k_map_pairs_matched_on_current_kpath"}

    wave_cfg = dict(model_config.validation_config)
    if bool(wave_cfg.get("allow_missing_wavefunction_overlap", False)):
        validations["wavefunction_overlap"] = {
            "available": False,
            "reason": str(wave_cfg.get("reason", "explicitly waived by config")),
        }
    else:
        validations["wavefunction_overlap"] = {
            "available": False,
            "reason": "wavefunction/projector overlap source data is not available in current configured pipeline",
        }

    diagnostics = artifacts._json_safe(results.get("diagnostics", {}))
    validations["orthogonalization"] = {"available": True, "diagnostics": diagnostics}

    for name, payload in validations.items():
        if not bool(payload.get("available", False)):
            summary["validation_incomplete"] = True
            summary["missing"].append({"name": name, "reason": payload.get("reason", "unknown")})

    if strict:
        for name, payload in validations.items():
            if not bool(payload.get("available", False)):
                if band_level_only and name in {"block_residuals", "symmetry_residuals", "wavefunction_overlap"}:
                    continue
                if name == "wavefunction_overlap" and bool(model_config.validation_config.get("allow_missing_wavefunction_overlap", False)):
                    continue
                raise ValueError(
                    f"validation.strict=true requires {name}; current configured pipeline reported unavailable: "
                    f"{payload.get('reason', 'unknown')}"
                )

    return validations, summary



def _validation_strict_enabled(validation_config: Mapping[str, Any]) -> bool:
    if bool(validation_config.get("strict", False)):
        return True
    for key in ("mode", "profile", "level"):
        value = validation_config.get(key)
        if value is not None and str(value).strip().lower() in {"strict", "production"}:
            return True
    return bool(validation_config.get("production", False))
