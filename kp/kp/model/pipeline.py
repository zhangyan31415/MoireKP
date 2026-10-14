from __future__ import annotations
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from ..reporting import KpReporter
from . import artifacts
from . import candidates
from . import configuration
from . import harmonic_selection
from ..viz import model_diagnostics
from . import validation


def run_configured_model(path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    reporter = KpReporter("kp model")
    candidates._progress_line("loading configuration ...", enabled=True, style="start")
    moire_config, model_config = configuration.build_moire_config_from_file(path)
    configuration._preflight_certified_model_selection(model_config)
    output_dir = model_config.output_dir
    output_profile = artifacts._model_output_profile(model_config)
    canonical_output = isinstance(model_config.raw.get("case"), Mapping) and bool(model_config.raw["case"].get("profile")) and bool(model_config.raw["case"].get("q_shell"))
    diagnostics_dir = artifacts._model_diagnostics_dir(output_dir) if output_profile == "debug" else None
    output_dir.mkdir(parents=True, exist_ok=True)
    n_orb_meta = model_config.orbital_count_metadata.get("n_orb", {})
    n_orb_source = str(n_orb_meta.get("source", "model_config")) if isinstance(n_orb_meta, Mapping) else "model_config"
    n_orb_source_label = {
        "projection_basis": "inferred from projection/basis.npz",
        "projection_gamma_routed": "inferred from routed projection group ranks",
        "source_project": "inferred from project.nlow_state_list",
        "model_config": "explicit model.n_orb assertion",
    }.get(n_orb_source, n_orb_source)
    public_fit = dict(model_config.fit_method_config or {})
    setup_rows: list[tuple[str, object]] = [
        ("basis", f"n_orb={list(model_config.n_orb)} · {n_orb_source_label}"),
    ]
    if public_fit:
        setup_rows.extend(
            [
                ("method", public_fit["method"]),
                (
                    "target",
                    f"{model_config.raw.get('model', {}).get('target_bands', 'top')} "
                    f"{public_fit['requested_bands']} bands",
                ),
                (
                    "fit k rows",
                    f"{public_fit['hamiltonian_kpoints']} · projected Heff row indices",
                ),
                (
                    "H weights",
                    f"full=1 · one-sided={public_fit['one_sided_weight']:g} · "
                    f"two-sided={public_fit['two_sided_weight']:g}",
                ),
            ]
        )
        if public_fit["method"] == "nonlinear":
            setup_rows.extend(
                [
                    ("band rows", public_fit["band_kpoints"]),
                    ("band loss", public_fit["band_loss_weight"]),
                    ("max steps", public_fit["max_steps"]),
                ]
            )
    setup_rows.append(("output", output_dir))
    reporter.section("Model setup")
    reporter.fields(setup_rows)
    artifacts._cleanup_stale_band_outputs(output_dir)
    artifacts._cleanup_stale_model_debug_outputs(output_dir)
    artifacts._cleanup_stale_hamiltonian_element_outputs(output_dir)
    harmonic_selection._run_model_harmonic_recommendation_once(
        model_config=model_config,
        output_dir=output_dir,
        reporter=reporter,
    )
    if diagnostics_dir is not None:
        model_diagnostics.save_bM_diagnostics(model_config=model_config, output_dir=diagnostics_dir)
        model_diagnostics.save_harmonics_diagnostics(moire_config=moire_config, model_config=model_config, output_dir=diagnostics_dir)
    verbose = bool(model_config.output_config.get("verbose", False))
    progress = bool(model_config.output_config.get("progress", True))
    log_name = str(model_config.output_config.get("log_file", "model_run.log"))
    log_path = output_dir / log_name
    moire_config.output_dir = None
    automatic_order_scan: dict[str, Any] | None = None
    profile_results: dict[str, dict[str, Any]] = {}
    try:
        fit_scan_result = candidates._run_auto_low_energy_fit_candidate_scan(
            moire_config=moire_config,
            model_config=model_config,
            output_dir=output_dir,
            log_path=log_path,
            verbose=verbose,
            progress=progress,
        )
        if fit_scan_result is None:
            prefitted_results = None
        else:
            prefitted_results, moire_config, model_config = fit_scan_result
        selection_config = model_config.model_selection_config
        if selection_config is not None and bool(selection_config.enabled):
            automatic_order_scan = candidates._run_automatic_family_order_scan(
                moire_config=moire_config,
                model_config=model_config,
                output_dir=output_dir,
                verbose=verbose,
                progress=progress,
            )
            profile_families = automatic_order_scan.get("profile_families") or {
                "linear": automatic_order_scan["profiles"],
                "nonlinear": None,
            }
            nonlinear_profiles = profile_families.get("nonlinear")
            requested_profiles = tuple(model_config.requested_profiles)
            if requested_profiles:
                automatic_order_scan["requested_profiles"] = list(requested_profiles)
                nested_results: dict[str, dict[str, dict[str, Any]]] = {}
                for profile_name in requested_profiles:
                    solver_family, quality = profile_name.split("/", maxsplit=1)
                    profiles = profile_families.get(solver_family)
                    if profiles is None:
                        raise RuntimeError(
                            f"requested model profile {profile_name!r} is unavailable"
                        )
                    decision = getattr(profiles, quality)
                    if decision.selected is None:
                        raise RuntimeError(
                            f"automatic {profile_name} model selection failed: "
                            f"{decision.unmet_targets}"
                        )
                    selected_run = automatic_order_scan["runs"][decision.selected.name]
                    selected_model = replace(
                        selected_run["model_config"],
                        output_dir=output_dir / solver_family / quality,
                    )
                    selected_moire = copy.copy(selected_run["moire_config"])
                    selected_moire.output_dir = None
                    profile_log = selected_model.output_dir / log_name
                    profile_log.parent.mkdir(parents=True, exist_ok=True)
                    selected_results = candidates._selected_profile_pipeline_result(
                        selected_run=selected_run,
                        selected_moire=selected_moire,
                        selected_model=selected_model,
                        profile_log=profile_log,
                        verbose=verbose,
                        progress=progress,
                    )
                    nested_results.setdefault(solver_family, {})[quality] = {
                        "results": selected_results,
                        "moire_config": selected_moire,
                        "model_config": selected_model,
                        "score": decision.selected,
                        "status": decision.status,
                    }
                profile_results = nested_results
                primary_name = next(
                    name
                    for name in (
                        "nonlinear/high",
                        "linear/high",
                        "nonlinear/low",
                        "linear/low",
                    )
                    if name in requested_profiles
                )
                primary_family, primary_quality = primary_name.split("/", maxsplit=1)
                primary_profile = profile_results[primary_family][primary_quality]
                automatic_order_scan["primary_profile"] = primary_name
            elif nonlinear_profiles is None:
                profiles = profile_families["linear"]
                for profile_name, decision in (
                    ("high", profiles.high),
                    ("low", profiles.low),
                ):
                    if decision.selected is None:
                        raise RuntimeError(
                            f"automatic {profile_name} model selection failed: "
                            f"{decision.unmet_targets}"
                        )
                    selected_run = automatic_order_scan["runs"][decision.selected.name]
                    selected_model = replace(
                        selected_run["model_config"],
                        output_dir=(
                            output_dir
                            if profile_name == "high"
                            else output_dir / "low"
                        ),
                    )
                    selected_moire = copy.copy(selected_run["moire_config"])
                    selected_moire.output_dir = None
                    profile_log = selected_model.output_dir / log_name
                    profile_log.parent.mkdir(parents=True, exist_ok=True)
                    selected_results = candidates._selected_profile_pipeline_result(
                        selected_run=selected_run,
                        selected_moire=selected_moire,
                        selected_model=selected_model,
                        profile_log=profile_log,
                        verbose=verbose,
                        progress=progress,
                    )
                    profile_results[profile_name] = {
                        "results": selected_results,
                        "moire_config": selected_moire,
                        "model_config": selected_model,
                        "score": decision.selected,
                        "status": decision.status,
                    }
                primary_profile = profile_results["high"]
                automatic_order_scan["primary_profile"] = "linear/high"
            else:
                nested_results: dict[str, dict[str, dict[str, Any]]] = {}
                for solver_family in ("linear", "nonlinear"):
                    profiles = profile_families[solver_family]
                    nested_results[solver_family] = {}
                    for quality, decision in (
                        ("high", profiles.high),
                        ("low", profiles.low),
                    ):
                        if decision.selected is None:
                            raise RuntimeError(
                                f"automatic {solver_family}/{quality} model selection failed: "
                                f"{decision.unmet_targets}"
                            )
                        selected_run = automatic_order_scan["runs"][decision.selected.name]
                        selected_model = replace(
                            selected_run["model_config"],
                            output_dir=output_dir / solver_family / quality,
                        )
                        selected_moire = copy.copy(selected_run["moire_config"])
                        selected_moire.output_dir = None
                        profile_log = selected_model.output_dir / log_name
                        profile_log.parent.mkdir(parents=True, exist_ok=True)
                        selected_results = candidates._selected_profile_pipeline_result(
                            selected_run=selected_run,
                            selected_moire=selected_moire,
                            selected_model=selected_model,
                            profile_log=profile_log,
                            verbose=verbose,
                            progress=progress,
                        )
                        nested_results[solver_family][quality] = {
                            "results": selected_results,
                            "moire_config": selected_moire,
                            "model_config": selected_model,
                            "score": decision.selected,
                            "status": decision.status,
                        }
                profile_results = nested_results
                primary_family = candidates._preferred_automatic_primary_family(
                    profile_families
                )
                primary_profile = profile_results[primary_family]["high"]
                automatic_order_scan["primary_profile"] = (
                    f"{primary_family}/high"
                )
            results = primary_profile["results"]
            moire_config = primary_profile["moire_config"]
            model_config = primary_profile["model_config"]
        elif prefitted_results is None:
            results = candidates._run_model_pipeline(
                moire_config,
                model_config,
                log_path,
                verbose=verbose,
                progress=progress,
            )
        else:
            results = prefitted_results
    finally:
        moire_config.output_dir = output_dir
    if automatic_order_scan is not None:
        results["automatic_family_order_scan"] = automatic_order_scan
        results["profile_results"] = profile_results
    eigvals = results["eigvals"]
    model_eigvecs = None
    if isinstance(eigvals, tuple):
        eigvals_array = eigvals[0]
        if len(eigvals) > 1:
            model_eigvecs = eigvals[1]
    else:
        eigvals_array = eigvals

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "eigvals.npy", np.asarray(eigvals_array))

    comparison = None
    plot_comparison = None
    all_band_plot_comparison = None
    comparison_vs_full_heff = None
    plot_comparison_vs_full_heff = None
    all_band_plot_comparison_vs_full_heff = None
    band_plot_path = None
    all_band_plot_path = None
    full_heff_band_plot_path = None
    hamiltonian_element_plot_path = None
    q_lattice_plot_path = model_diagnostics.save_q_lattice_harmonics_plot(
        Q_set1=np.asarray(moire_config.Q_set1, dtype=float),
        Q_set2=np.asarray(moire_config.Q_set2, dtype=float),
        bM1=np.asarray(moire_config.bM1, dtype=float),
        bM2=np.asarray(moire_config.bM2, dtype=float),
        intra_harmonics=getattr(moire_config, "intra_harmonics_map", {}) or {},
        inter_harmonics=getattr(moire_config, "inter_harmonics_map", {}) or {},
        path=output_dir / "q_lattice_harmonics.pdf",
        sectors=getattr(moire_config, "sectors", []),
        inter_sector_pairs=[
            list(pair)
            for row in getattr(moire_config, "term_templates", [])
            if str(row.get("source", "")) == "tunneling"
            for pair in row.get("sector_pairs", [])
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2
        ],
    )
    if model_config.compare_to_heff:
        heff_matrix_all = configuration._load_heff_in_model_basis(model_config)
        heff_matrix_selected = configuration._select_rows(heff_matrix_all, model_config.band_indices)
        if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
            heff_eig = np.load(model_config.heff_eig_file)
            heff_selected = configuration._select_rows(heff_eig, model_config.band_indices)
        else:
            heff_selected = np.linalg.eigvalsh(heff_matrix_selected)
        harmonic_mask = validation._selected_harmonic_support_mask(
            np.asarray(moire_config.Q_set1, dtype=float),
            np.asarray(moire_config.Q_set2, dtype=float),
            n_orb=(int(model_config.n_orb[0]), int(model_config.n_orb[1])),
            current_counts=harmonic_selection._resolved_harmonic_count_limits(model_config.harmonics_config),
        )
        current_support_heff = validation._current_heff_support_hamiltonians(
            heff_matrix_selected,
            qset1=np.asarray(moire_config.Q_set1, dtype=float),
            qset2=np.asarray(moire_config.Q_set2, dtype=float),
            n_orb=model_config.n_orb,
            harmonics_config=model_config.harmonics_config,
        )
        current_support_eigvals, current_support_eigvecs = np.linalg.eigh(current_support_heff)
        np.save(output_dir / "current_heff_support_eigvals.npy", current_support_eigvals)
        overlap_weights = None
        if model_eigvecs is not None:
            model_eigvecs_selected = np.asarray(model_eigvecs)
            if model_eigvecs_selected.shape != current_support_eigvecs.shape:
                raise ValueError(
                    "model/current-support eigenvector shape mismatch after band-path selection: "
                    f"{model_eigvecs_selected.shape} != {current_support_eigvecs.shape}"
                )
            overlap_weights = model_diagnostics._band_overlap_weights(
                model_eigvecs_selected,
                current_support_eigvecs,
                model_eigvals=np.asarray(eigvals_array, dtype=float),
                reference_eigvals=np.asarray(current_support_eigvals, dtype=float),
                degeneracy_tol=float(model_config.band_plot_config.get("overlap_degeneracy_tol", 3.0e-3)),
            )
        comparison = {
            **validation.compare_bands(eigvals_array, current_support_eigvals, band_slice=model_config.band_slice),
            "reference": "current_heff_support_mask",
        }
        comparison_vs_full_heff = {
            **validation.compare_bands(eigvals_array, heff_selected, band_slice=model_config.band_slice),
            "reference": "original_heff",
        }
        if diagnostics_dir is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            with (diagnostics_dir / "comparison.json").open("w", encoding="utf-8") as handle:
                json.dump(comparison, handle, indent=2)
            with (diagnostics_dir / "comparison_vs_full_heff.json").open("w", encoding="utf-8") as handle:
                json.dump(comparison_vs_full_heff, handle, indent=2)
        target_bands = str(model_config.raw.get("model", {}).get("target_bands", "top")).strip().lower()
        window_plot_config = validation._window_band_plot_config(model_config.band_plot_config, target_bands=target_bands)
        band_plot_config = dict(window_plot_config)
        if overlap_weights is not None:
            band_plot_config["plot_all_bands"] = True
        plot_comparison = validation.compare_bands_for_plot(
            eigvals_array,
            current_support_eigvals,
            band_slice=model_config.band_slice,
            plot_config=band_plot_config,
        )
        plot_comparison["reference"] = "current_heff_support_mask"
        plot_comparison_vs_full_heff = validation.compare_bands_for_plot(
            eigvals_array,
            heff_selected,
            band_slice=model_config.band_slice,
            plot_config=band_plot_config,
        )
        plot_comparison_vs_full_heff["reference"] = "original_heff"
        plot_comparison_name = model_diagnostics._plot_comparison_filename(band_plot_config)
        if diagnostics_dir is not None:
            with (diagnostics_dir / "comparison_plot.json").open("w", encoding="utf-8") as handle:
                json.dump(plot_comparison, handle, indent=2)
            with (diagnostics_dir / plot_comparison_name).open("w", encoding="utf-8") as handle:
                json.dump(plot_comparison, handle, indent=2)
            with (diagnostics_dir / "comparison_plot_vs_full_heff.json").open("w", encoding="utf-8") as handle:
                json.dump(plot_comparison_vs_full_heff, handle, indent=2)
        x_values, x_ticks, x_ticklabels = model_diagnostics._plot_axis_from_kpath(model_config, np.asarray(eigvals_array).shape[0])
        band_plot_path = model_diagnostics.save_band_comparison_plot(
            eigvals_array,
            current_support_eigvals,
            output_dir / "band_comparison.pdf",
            band_slice=model_config.band_slice,
            plot_config=band_plot_config,
            x=x_values,
            x_ticks=x_ticks,
            x_ticklabels=x_ticklabels,
            title=model_diagnostics._band_plot_title(model_config),
            overlap_weights=overlap_weights,
            reference_label="Current Heff support mask",
        )
        full_heff_band_plot_path = model_diagnostics.save_band_comparison_plot(
            eigvals_array,
            heff_selected,
            output_dir / "band_comparison_vs_full_heff.pdf",
            band_slice=model_config.band_slice,
            plot_config=band_plot_config,
            x=x_values,
            x_ticks=x_ticks,
            x_ticklabels=x_ticklabels,
            title=f"{model_diagnostics._band_plot_title(model_config)} (vs original Heff)",
            reference_label="Original Heff",
        )
        all_band_config = validation._all_band_plot_config(model_config.band_plot_config)
        all_band_plot_comparison = validation.compare_bands_for_plot(
            eigvals_array,
            current_support_eigvals,
            band_slice=None,
            plot_config=all_band_config,
        )
        all_band_plot_comparison["reference"] = "current_heff_support_mask"
        all_band_plot_comparison_vs_full_heff = validation.compare_bands_for_plot(
            eigvals_array,
            heff_selected,
            band_slice=None,
            plot_config=all_band_config,
        )
        all_band_plot_comparison_vs_full_heff["reference"] = "original_heff"
        if diagnostics_dir is not None:
            with (diagnostics_dir / "comparison_all_bands.json").open("w", encoding="utf-8") as handle:
                json.dump(all_band_plot_comparison, handle, indent=2)
            with (diagnostics_dir / "comparison_all_bands_vs_full_heff.json").open("w", encoding="utf-8") as handle:
                json.dump(all_band_plot_comparison_vs_full_heff, handle, indent=2)
        all_band_plot_path = model_diagnostics.save_band_comparison_plot(
            eigvals_array,
            current_support_eigvals,
            output_dir / "band_comparison_all.pdf",
            band_slice=None,
            plot_config=all_band_config,
            x=x_values,
            x_ticks=x_ticks,
            x_ticklabels=x_ticklabels,
            title=f"{model_diagnostics._band_plot_title(model_config)} (all bands)",
            reference_label="Current Heff support mask",
        )
        model_hamiltonians = np.asarray(results.get("band_hamiltonians"), dtype=np.complex128)
        fit_positions, _holdout_positions = validation._selected_index_positions(model_config, int(heff_matrix_selected.shape[0]))
        selected_indices = list(model_config.band_indices or list(range(int(heff_matrix_selected.shape[0]))))
        fit_k_indices = [int(selected_indices[pos]) for pos in fit_positions]
        if harmonic_mask is None:
            raise ValueError("current harmonic support mask is unavailable")
        qnorm_plot_dir = output_dir / "hamiltonian_element_comparisons"
        hamiltonian_element_plot_path = model_diagnostics.save_hamiltonian_element_comparison_plot(
            model_hamiltonians,
            heff_matrix_selected,
            qnorm_plot_dir / "native_full.png",
            harmonic_mask=harmonic_mask,
            positions=fit_positions,
            k_indices=fit_k_indices,
            title=f"{model_diagnostics._band_plot_title(model_config)} Hamiltonian elements",
            qset1=np.asarray(moire_config.Q_set1, dtype=float),
            qset2=np.asarray(moire_config.Q_set2, dtype=float),
            n_orb=(int(model_config.n_orb[0]), int(model_config.n_orb[1])),
            subtract_layer_diagonal_mean=True,
        )
        model_diagnostics.save_hamiltonian_element_comparison_plot(
            model_hamiltonians,
            heff_matrix_selected,
            qnorm_plot_dir / "native_selected_support.png",
            harmonic_mask=harmonic_mask,
            positions=fit_positions,
            k_indices=fit_k_indices,
            title=f"{model_diagnostics._band_plot_title(model_config)} Hamiltonian elements",
            qset1=np.asarray(moire_config.Q_set1, dtype=float),
            qset2=np.asarray(moire_config.Q_set2, dtype=float),
            n_orb=(int(model_config.n_orb[0]), int(model_config.n_orb[1])),
            subtract_layer_diagonal_mean=True,
            view="selected_support",
        )
        model_diagnostics.save_hamiltonian_element_comparison_plot(
            model_hamiltonians,
            heff_matrix_selected,
            qnorm_plot_dir / "qnorm_full.png",
            harmonic_mask=harmonic_mask,
            positions=fit_positions,
            k_indices=fit_k_indices,
            title=f"{model_diagnostics._band_plot_title(model_config)} Hamiltonian elements",
            qset1=np.asarray(moire_config.Q_set1, dtype=float),
            qset2=np.asarray(moire_config.Q_set2, dtype=float),
            n_orb=(int(model_config.n_orb[0]), int(model_config.n_orb[1])),
            subtract_layer_diagonal_mean=True,
            q_order="norm_shell",
        )
        model_diagnostics.save_hamiltonian_element_comparison_plot(
            model_hamiltonians,
            heff_matrix_selected,
            qnorm_plot_dir / "qnorm_selected_support.png",
            harmonic_mask=harmonic_mask,
            positions=fit_positions,
            k_indices=fit_k_indices,
            title=f"{model_diagnostics._band_plot_title(model_config)} Hamiltonian elements",
            qset1=np.asarray(moire_config.Q_set1, dtype=float),
            qset2=np.asarray(moire_config.Q_set2, dtype=float),
            n_orb=(int(model_config.n_orb[0]), int(model_config.n_orb[1])),
            subtract_layer_diagonal_mean=True,
            view="selected_support",
            q_order="norm_shell",
        )
    results["configured_model"] = model_config
    results["moire_config"] = moire_config
    results["comparison"] = comparison
    results["plot_comparison"] = plot_comparison
    results["all_band_plot_comparison"] = all_band_plot_comparison
    results["comparison_vs_full_heff"] = comparison_vs_full_heff
    results["plot_comparison_vs_full_heff"] = plot_comparison_vs_full_heff
    results["all_band_plot_comparison_vs_full_heff"] = all_band_plot_comparison_vs_full_heff
    results["q_lattice_plot"] = str(q_lattice_plot_path.resolve())
    if band_plot_path is not None:
        results["band_plot"] = str(band_plot_path.resolve())
    if all_band_plot_path is not None:
        results["all_band_plot"] = str(all_band_plot_path.resolve())
    if full_heff_band_plot_path is not None:
        results["full_heff_band_plot"] = str(full_heff_band_plot_path.resolve())
    if hamiltonian_element_plot_path is not None:
        results["hamiltonian_element_plot"] = str(hamiltonian_element_plot_path.resolve())
        results["hamiltonian_element_plot_pdf"] = str(hamiltonian_element_plot_path.with_suffix(".pdf").resolve())
    candidates._progress_line("validating fitted model ...", enabled=progress)
    validation_started = time.perf_counter()
    validations, validation_summary = validation._compute_validation_outputs(
        results=results,
        model_config=model_config,
        moire_config=moire_config,
    )
    candidates._progress_line(
        f"validating fitted model done in {time.perf_counter() - validation_started:.2f} s",
        enabled=progress,
    )
    results["validations"] = validations
    results["validation_summary"] = validation_summary
    if automatic_order_scan is None:
        auto_model_selection = artifacts._write_auto_model_selection_outputs(
            results=results,
            output_dir=output_dir,
            model_config=model_config,
            moire_config=moire_config,
        )
    else:
        auto_model_selection = artifacts._write_high_low_model_selection_outputs(
            scan=automatic_order_scan,
            output_dir=output_dir,
        )
    results["auto_model_selection"] = auto_model_selection
    artifacts._write_model_registry_outputs(
        results=results,
        output_dir=output_dir,
        model_config=model_config,
        validations=validations,
        validation_summary=validation_summary,
        diagnostics_dir=diagnostics_dir,
    )
    if automatic_order_scan is not None:
        if any(family in profile_results for family in ("linear", "nonlinear")):
            for solver_family, family_results in profile_results.items():
                for quality, profile in family_results.items():
                    profile["results"] = artifacts._materialize_automatic_profile_output(
                        profile_name=f"{solver_family}/{quality}",
                        profile=profile,
                        target_hamiltonians=np.asarray(
                            automatic_order_scan["target_hamiltonians"]
                        ),
                        root_output_dir=output_dir,
                    )
        else:
            for profile_name in ("high", "low"):
                profile_results[profile_name]["results"] = (
                    artifacts._materialize_automatic_profile_output(
                        profile_name=profile_name,
                        profile=profile_results[profile_name],
                        target_hamiltonians=np.asarray(
                            automatic_order_scan["target_hamiltonians"]
                        ),
                        root_output_dir=output_dir,
                    )
                )
        results["profile_results"] = profile_results
        results["auto_model_selection"] = auto_model_selection
    results["model_log"] = str(results.get("model_log") or log_path)
    results["runtime_s"] = float(time.perf_counter() - started)
    return results
