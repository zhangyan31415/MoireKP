from __future__ import annotations
import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import yaml
from ..plot_style import (
    KP_BAND_BOX_ASPECT,
    KP_BAND_FIGSIZE,
    KP_DPI,
    KP_LEGEND_KWARGS,
    KP_MARKER_STYLE,
    KP_MODEL_STYLE,
    KP_PRIMARY_STYLE,
    KP_REFERENCE_STYLE,
    apply_kp_axis_style,
    kp_font_family,
)
from .core import MoireConfig
from .model_selection import (
    BandWindowMetrics,
    CandidateScore,
    high_low_selection_record,
)
from . import configuration


_MODEL_DEBUG_OUTPUT_FILES = {
    "terms.json",
    "discarded_terms.json",
    "coefficients.json",
    "fit_diagnostics.json",
    "orthogonalization.json",
    "matrix_residuals.json",
    "block_residuals.json",
    "wavefunction_overlap.json",
    "symmetry_residuals.json",
    "operation_registry.json",
    "bM_diagnostic.json",
    "harmonics_diagnostic.json",
    "harmonics_diagnostic.pdf",
    "comparison.json",
    "comparison_plot.json",
}



def _cleanup_stale_band_outputs(output_dir: Path) -> None:
    for pattern in (
        "band_comparison.pdf",
        "band_comparison_vs_full_heff.pdf",
        "current_heff_support_eigvals.npy",
        "comparison_top*.json",
        "band_comparison_top*.pdf",
        "comparison_plot.json",
        "comparison_all_bands.json",
        "band_comparison_all.pdf",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()



def _model_output_profile(model_config: configuration.ConfiguredModel) -> str:
    profile = str(model_config.output_config.get("profile", "release")).strip().lower()
    if profile in {"", "default"}:
        return "release"
    if profile not in {"release", "debug"}:
        raise ValueError(f"output.profile must be 'release' or 'debug', got {profile!r}")
    return profile



def _model_diagnostics_dir(output_dir: Path) -> Path:
    return output_dir / "diagnostics"



def _cleanup_stale_model_debug_outputs(output_dir: Path) -> None:
    for filename in _MODEL_DEBUG_OUTPUT_FILES:
        path = output_dir / filename
        if path.exists() and path.is_file():
            path.unlink()
    for pattern in ("comparison_top*.json", "comparison_bottom*.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    diagnostics_dir = _model_diagnostics_dir(output_dir)
    if diagnostics_dir.exists():
        shutil.rmtree(diagnostics_dir)



def _cleanup_stale_hamiltonian_element_outputs(output_dir: Path) -> None:
    for filename in (
        "hamiltonian_element_comparison.png",
        "hamiltonian_element_comparison.pdf",
        "hamiltonian_element_comparison_selected_support.png",
        "hamiltonian_element_comparison_selected_support.pdf",
    ):
        path = output_dir / filename
        if path.exists() and path.is_file():
            path.unlink()
    grouped_dir = output_dir / "hamiltonian_element_comparisons"
    if grouped_dir.exists():
        shutil.rmtree(grouped_dir)



def _term_key_to_dict(key: Any) -> dict[str, Any]:
    attrs = ("Mz", "Mz_star", "layer_from", "layer_to", "orbital_from", "orbital_to", "p")
    return {attr: getattr(key, attr) for attr in attrs if hasattr(key, attr)}



def _term_to_dict(term: Any) -> dict[str, Any]:
    return {
        "key": _term_key_to_dict(getattr(term, "key", None)),
        "tag": getattr(term, "tag", None),
        "active": bool(getattr(term, "active", False)),
        "r_value_real": float(np.real(getattr(term, "r_value_real", 0.0))),
        "r_value_imag": float(np.real(getattr(term, "r_value_imag", 0.0))),
        "symmetry_ops": getattr(term, "symmetry_ops", []),
        "registry_metadata": _json_safe(getattr(term, "registry_metadata", {})),
    }



_RELEASE_SYMMETRY_OP_KEYS = (
    "name",
    "operation",
    "user_name",
    "params",
    "antiunitary",
    "k_map",
    "q_map",
    "sector_map",
    "internal_resolved_action",
)



def _compact_symmetry_op_for_release(operation: Any) -> dict[str, Any]:
    if not isinstance(operation, Mapping):
        return {}
    return {
        key: _json_safe(operation[key])
        for key in _RELEASE_SYMMETRY_OP_KEYS
        if key in operation and operation[key] is not None
    }



def _term_to_release_dict(term: Any) -> dict[str, Any]:
    payload = {
        "key": _term_key_to_dict(getattr(term, "key", None)),
        "tag": getattr(term, "tag", None),
        "active": bool(getattr(term, "active", False)),
        "r_value_real": float(np.real(getattr(term, "r_value_real", 0.0))),
        "r_value_imag": float(np.real(getattr(term, "r_value_imag", 0.0))),
        "symmetry_ops": [
            op
            for op in (_compact_symmetry_op_for_release(operation) for operation in getattr(term, "symmetry_ops", []))
            if op
        ],
        "registry_metadata": _json_safe(getattr(term, "registry_metadata", {})),
    }
    if hasattr(term, "_moire_needs_hermitize_real") and hasattr(term, "_moire_needs_hermitize_imag"):
        payload["hermitization"] = {
            "real": bool(getattr(term, "_moire_needs_hermitize_real")),
            "imag": bool(getattr(term, "_moire_needs_hermitize_imag")),
            "inconsistent": bool(getattr(term, "_moire_hermitize_flags_inconsistent", False)),
        }
    return payload



def _operation_physics_level(model_config: configuration.ConfiguredModel, operation: Mapping[str, Any]) -> str:
    representation_level = str(operation.get("representation_level", ""))
    if representation_level:
        return representation_level
    name = str(operation.get("name", ""))
    if model_config.symmetry_source_config.get("type") == "toy_generator":
        return "toy"
    return "physical"



def _symmetry_integrity(model_config: configuration.ConfiguredModel) -> str:
    if model_config.symmetry_source_config.get("type") == "toy_generator":
        return "toy_generator"
    if model_config.symmetry_source_metadata.get("exactification_owner") == "kp_symm":
        return "kp_symm_exactified"
    if model_config.symmetry_source_metadata.get("source_matrix_projection_reports"):
        return "source_matrix_projected"
    return "matrix_backed"



def _term_template_tag(template: Mapping[str, Any]) -> str:
    if "tag" in template:
        return str(template["tag"])
    source = str(template.get("source", ""))
    if source == "diagonal_kp":
        return "Kinect"
    if source == "onsite":
        return "Onsite"
    if source == "moire_potential":
        return "intra"
    if source == "tunneling":
        return "inter"
    return ""



def _term_tags_by_symmetry_tag(term_templates: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for template in term_templates:
        if not isinstance(template, Mapping):
            continue
        tag = _term_template_tag(template)
        if not tag:
            continue
        term_name = str(template.get("name", tag))
        out.setdefault(tag, [])
        if term_name not in out[tag]:
            out[tag].append(term_name)
    return out



def _build_operation_registry(model_config: configuration.ConfiguredModel) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    source_ops_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    source_ops_by_name: dict[str, Mapping[str, Any]] = {}
    for op in model_config.symmetry_source_config.get("operations", []):
        if not isinstance(op, Mapping) or op.get("name") is None:
            continue
        source_name = str(op.get("name"))
        source_operation = str(op.get("operation", source_name))
        source_ops_by_identity[(source_name, source_operation)] = op
        source_ops_by_name.setdefault(source_name, op)
    term_tags_by_tag = _term_tags_by_symmetry_tag(getattr(model_config, "term_templates", []))
    for tag, operations in model_config.symmetry_map.items():
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            user_name = str(operation.get("user_name", operation.get("name", "")))
            canonical_name = str(operation.get("name", ""))
            configuration._require_resolved_operation_action(operation)
            resolved_k_map = operation["k_map"]
            resolved_q_map = operation["q_map"]
            resolved_sector_map = operation["sector_map"]
            source_operation = str(operation.get("operation", canonical_name))
            source_record = source_ops_by_identity.get(
                (canonical_name, source_operation),
                source_ops_by_name.get(canonical_name, {}),
            )
            if isinstance(source_record, Mapping):
                source_operation = str(operation.get("operation", source_record.get("operation", canonical_name)))
            key = (
                user_name,
                canonical_name,
                json.dumps(_json_safe(resolved_k_map), sort_keys=True),
                json.dumps(_json_safe(resolved_q_map), sort_keys=True),
                json.dumps(_json_safe(resolved_sector_map), sort_keys=True),
                source_operation,
            )
            if key in rows_by_key:
                row = rows_by_key[key]
                if str(tag) not in row["used_by_tags"]:
                    row["used_by_tags"].append(str(tag))
                for term_tag in term_tags_by_tag.get(str(tag), []):
                    if term_tag not in row["term_tags"]:
                        row["term_tags"].append(term_tag)
                continue
            antiunitary = bool(operation["antiunitary"])
            rows_by_key[key] = {
                "user_operation": user_name,
                "canonical_operation": canonical_name,
                "operation_alias": operation.get("operation_alias", source_record.get("operation_alias")),
                "canonical_physical_operation": operation.get(
                    "canonical_physical_operation",
                    source_record.get("canonical_physical_operation", canonical_name),
                ),
                "representation_level": operation.get("representation_level", source_record.get("representation_level")),
                "effective_name": operation.get("effective_name", source_record.get("effective_name")),
                "physical_parent": operation.get("physical_parent", source_record.get("physical_parent")),
                "approximation": _json_safe(operation.get("approximation", source_record.get("approximation"))),
                "derived_from": _json_safe(operation.get("derived_from", source_record.get("derived_from"))),
                "valley_type": str(model_config.valley_model.get("valley_type", "")),
                "valley_mode": str(model_config.valley_model.get("mode", "")),
                "spin_convention": str(model_config.valley_model.get("spin_convention", "")),
                "operation_physics_level": _operation_physics_level(model_config, operation),
                "antiunitary": antiunitary,
                "k_map": _json_safe(resolved_k_map),
                "q_map": _json_safe(resolved_q_map),
                "sector_map": _json_safe(resolved_sector_map),
                "source_operation": source_operation,
                "used_by_tags": [str(tag)],
                "term_tags": list(term_tags_by_tag.get(str(tag), [])),
                "internal_resolved_action": _json_safe(operation.get("internal_resolved_action")),
                "matrix_kind": operation.get("matrix_kind", source_record.get("matrix_kind", model_config.symmetry_source_config.get("matrix_kind"))),
                **{
                    key: _json_safe(operation.get(key, source_record.get(key)))
                    for key in configuration.SOURCE_META_KEYS
                },
            }
    for row in rows_by_key.values():
        row["used_by_tags"] = sorted(row["used_by_tags"])
        row["term_tags"] = sorted(row["term_tags"])
    return list(rows_by_key.values())



def _build_run_summary(
    *,
    model_config: configuration.ConfiguredModel,
    results: Mapping[str, Any],
    operation_registry: Sequence[Mapping[str, Any]],
    validation_summary: Mapping[str, Any],
    active_terms_hash: str | None = None,
    output_profile: str = "release",
) -> dict[str, Any]:
    uses_toy = model_config.symmetry_source_config.get("type") == "toy_generator"
    fitted_response = getattr(results.get("model"), "_fitted_response_model", None)
    fit_method = dict(model_config.fit_method_config or {})
    if fit_method and fitted_response is not None:
        fit_method["resolved_objective"] = _json_safe(
            getattr(fitted_response, "fit_objective", {})
        )
        if fit_method.get("method") == "nonlinear":
            fit_method["nonlinear_result"] = _json_safe(
                results.get("band_refinement", {})
            )
    summary = {
        "output_profile": output_profile,
        "model_config_path": str(model_config.path),
        "symmetry_integrity": _symmetry_integrity(model_config),
        "symmetry_source_type": str(model_config.symmetry_source_config.get("type", "none")),
        "uses_toy_generator": bool(uses_toy),
        "validation_incomplete": bool(validation_summary.get("validation_incomplete", False)),
        "valley_type": str(model_config.valley_model.get("valley_type", "")),
        "valley_mode": str(model_config.valley_model.get("mode", "")),
        "spin_convention": str(model_config.valley_model.get("spin_convention", "")),
        "comparison": _json_safe(results.get("comparison")),
        "plot_comparison": _json_safe(results.get("plot_comparison")),
        "all_band_plot_comparison": _json_safe(results.get("all_band_plot_comparison")),
        "comparison_vs_full_heff": _json_safe(results.get("comparison_vs_full_heff")),
        "plot_comparison_vs_full_heff": _json_safe(results.get("plot_comparison_vs_full_heff")),
        "all_band_plot_comparison_vs_full_heff": _json_safe(
            results.get("all_band_plot_comparison_vs_full_heff")
        ),
        "all_band_plot": "band_comparison_all.pdf" if results.get("all_band_plot") else None,
        "full_heff_band_plot": (
            "band_comparison_vs_full_heff.pdf" if results.get("full_heff_band_plot") else None
        ),
        "q_lattice_plot": "q_lattice_harmonics.pdf" if results.get("q_lattice_plot") else None,
        "null_channel_filter": _json_safe(results.get("null_channel_filter")),
        "coefficient_pruning": _json_safe(results.get("coefficient_pruning")),
        "band_refinement": _json_safe(results.get("band_refinement", {"enabled": False})),
        "auto_model_selection": _json_safe(results.get("auto_model_selection", {"enabled": False})),
        "orbital_counts": _json_safe(model_config.orbital_count_metadata),
        "fit_selection": _json_safe(model_config.fit_selection_metadata),
        "fit_method": _json_safe(fit_method) if fit_method else None,
        "term_template_profile": _json_safe(model_config.term_template_metadata),
        "operations": list(operation_registry),
    }
    if output_profile == "debug":
        summary["operation_registry_file"] = "diagnostics/operation_registry.json"
    if active_terms_hash is not None:
        summary["active_terms_hash"] = active_terms_hash
        summary["active_terms_hash_file"] = "active_terms.sha256"
    return summary



def _model_selection_candidate_record(candidate: CandidateScore) -> dict[str, Any]:
    def window_record(metrics: BandWindowMetrics | None) -> dict[str, Any] | None:
        if metrics is None:
            return None
        return {
            "band_count": int(metrics.band_count),
            "weighted_rms_mev": float(metrics.weighted_rms_mev),
            "weighted_max_mev": float(metrics.weighted_max_mev),
            "mean_subspace_overlap": float(metrics.mean_subspace_overlap),
            "minimum_singular_value": float(metrics.minimum_singular_value),
        }

    return {
        "name": str(candidate.name),
        "solver_family": str(candidate.solver_family),
        "orders": {
            "kinetic": int(candidate.orders.kinetic),
            "intra": int(candidate.orders.intra),
            "inter": int(candidate.orders.inter),
        },
        "independent_real_parameters": int(candidate.independent_real_parameters),
        "active_group_count": int(candidate.active_group_count),
        "harmonic_support_size": int(candidate.harmonic_support_size),
        "selection_scope": str(candidate.selection_scope),
        "primary_band_count": int(candidate.primary_band_count),
        "primary_weighted_rms_mev": float(candidate.weighted_rms_mev),
        "primary_weighted_rms_se_mev": float(candidate.weighted_rms_se_mev),
        "primary_weighted_max_mev": float(candidate.weighted_max_mev),
        "expanded_all_band_rms_mev": (
            None
            if candidate.expanded_weighted_rms_mev is None
            else float(candidate.expanded_weighted_rms_mev)
        ),
        "mean_subspace_overlap": float(candidate.mean_subspace_overlap),
        "certified": bool(candidate.certified),
        "guards_passed": bool(candidate.guards_passed),
        "low_window": window_record(candidate.low_window_metrics),
        "high_window": window_record(candidate.high_window_metrics),
    }



def _write_high_low_model_selection_outputs(
    *,
    scan: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write the auditable high/low decision and its complexity frontier."""

    profiles = scan["profiles"]
    profile_families = scan.get("profile_families") or {
        "linear": profiles,
        "nonlinear": None,
    }
    requested_profiles = tuple(str(value) for value in scan.get("requested_profiles", ()))
    requested_profile_set = set(requested_profiles)
    nonlinear_profiles = profile_families.get("nonlinear")
    unique: dict[str, CandidateScore] = {}
    for run in scan.get("runs", {}).values():
        score = run.get("score") if isinstance(run, Mapping) else None
        if isinstance(score, CandidateScore):
            unique[score.name] = score
    candidates = sorted(
        unique.values(),
        key=lambda item: (
            item.independent_real_parameters,
            item.weighted_rms_mev,
            item.name,
        ),
    )
    summary = {
        "enabled": True,
        "mode": (
            "automatic_four_model_profiles"
            if nonlinear_profiles is not None
            else "automatic_high_low_family_orders"
        ),
        "reference": "current_heff_support_mask",
        "selection_rule": {
            "high": "minimum primary edge weighted validation RMS subject to overlap gate",
            "low": "fewest independent real parameters on the adaptive max(2SE, relative RMS, minimum tolerance) plateau subject to overlap gate",
            "expanded_all_band_error": "diagnostic_only",
        },
        "profile_band_windows": _json_safe(scan.get("profile_band_windows", {})),
        **high_low_selection_record(profiles),
        "term_group_pruning": _json_safe(
            scan.get("term_group_pruning", {"enabled": False})
        ),
        "nonlinear_frontier": _json_safe(
            scan.get("nonlinear_frontier", {"enabled": False})
        ),
        "candidates": [
            _model_selection_candidate_record(candidate) for candidate in candidates
        ],
    }
    if nonlinear_profiles is not None:
        summary["profile_families"] = {
            "linear": high_low_selection_record(profile_families["linear"])["profiles"],
            "nonlinear": high_low_selection_record(nonlinear_profiles)["profiles"],
        }
        for solver_family in ("linear", "nonlinear"):
            for quality in ("high", "low"):
                profile_name = f"{solver_family}/{quality}"
                if not requested_profiles or profile_name in requested_profile_set:
                    summary["profile_families"][solver_family][quality]["output_dir"] = (
                        profile_name
                    )
    if requested_profiles:
        summary["requested_profiles"] = list(requested_profiles)
        summary["published_profiles"] = {
            profile_name: {"output_dir": profile_name}
            for profile_name in requested_profiles
        }
    else:
        summary["profiles"]["high"]["output_dir"] = "high"
        summary["profiles"]["low"]["output_dir"] = "low"
    if scan.get("primary_profile") is not None:
        summary["primary_profile"] = str(scan["primary_profile"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "auto_model_selection.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    csv_lines = [
        "candidate,solver_family,selection_scope,primary_band_count,kinetic_order,intra_order,inter_order,independent_real_parameters,active_group_count,harmonic_support_size,primary_weighted_rms_mev,primary_weighted_rms_se_mev,primary_weighted_max_mev,mean_subspace_overlap,low_band_count,low_rms_mev,low_max_mev,low_mean_overlap,low_minimum_singular_value,high_band_count,high_rms_mev,high_max_mev,high_mean_overlap,high_minimum_singular_value,expanded_all_band_rms_mev,selected_high,selected_low"
    ]
    high_name = (
        None if profiles.high.selected is None else profiles.high.selected.name
    )
    low_name = None if profiles.low.selected is None else profiles.low.selected.name
    for candidate in candidates:
        low_window = candidate.low_window_metrics
        high_window = candidate.high_window_metrics
        csv_lines.append(
            ",".join(
                [
                    candidate.name,
                    candidate.solver_family,
                    candidate.selection_scope,
                    str(candidate.primary_band_count),
                    str(candidate.orders.kinetic),
                    str(candidate.orders.intra),
                    str(candidate.orders.inter),
                    str(candidate.independent_real_parameters),
                    str(candidate.active_group_count),
                    str(candidate.harmonic_support_size),
                    f"{candidate.weighted_rms_mev:.12g}",
                    f"{candidate.weighted_rms_se_mev:.12g}",
                    f"{candidate.weighted_max_mev:.12g}",
                    f"{candidate.mean_subspace_overlap:.12g}",
                    "" if low_window is None else str(low_window.band_count),
                    "" if low_window is None else f"{low_window.weighted_rms_mev:.12g}",
                    "" if low_window is None else f"{low_window.weighted_max_mev:.12g}",
                    "" if low_window is None else f"{low_window.mean_subspace_overlap:.12g}",
                    "" if low_window is None else f"{low_window.minimum_singular_value:.12g}",
                    "" if high_window is None else str(high_window.band_count),
                    "" if high_window is None else f"{high_window.weighted_rms_mev:.12g}",
                    "" if high_window is None else f"{high_window.weighted_max_mev:.12g}",
                    "" if high_window is None else f"{high_window.mean_subspace_overlap:.12g}",
                    "" if high_window is None else f"{high_window.minimum_singular_value:.12g}",
                    "" if candidate.expanded_weighted_rms_mev is None else f"{candidate.expanded_weighted_rms_mev:.12g}",
                    str(candidate.name == high_name).lower(),
                    str(candidate.name == low_name).lower(),
                ]
            )
        )
    (output_dir / "candidate_metrics.csv").write_text(
        "\n".join(csv_lines) + "\n",
        encoding="utf-8",
    )

    def profile_line(label: str, decision: Any) -> str:
        selected = decision.selected
        if selected is None:
            return f"- {label}: no acceptable candidate (`{decision.status}`)."
        low_rms = (
            "n/a"
            if selected.low_window_metrics is None
            else f"{selected.low_window_metrics.weighted_rms_mev:.3f}"
        )
        high_rms = (
            "n/a"
            if selected.high_window_metrics is None
            else f"{selected.high_window_metrics.weighted_rms_mev:.3f}"
        )
        return (
            f"- {label}: `{selected.name}`; orders "
            f"k/intra/inter=`{selected.orders.kinetic}/{selected.orders.intra}/{selected.orders.inter}`; "
            f"{selected.independent_real_parameters} independent real parameters; "
            f"primary RMS `{selected.weighted_rms_mev:.3f} meV`; "
            f"low/high RMS `{low_rms}/{high_rms} meV`; "
            f"mean overlap `{selected.mean_subspace_overlap:.6f}`; status `{decision.status}`."
        )

    if nonlinear_profiles is None:
        profile_lines = [
            profile_line("High Accuracy", profile_families["linear"].high),
            profile_line("Low Parameter", profile_families["linear"].low),
        ]
    else:
        profile_lines = [
            profile_line("Linear High", profile_families["linear"].high),
            profile_line("Linear Low", profile_families["linear"].low),
        ]
        profile_lines.extend(
            [
                profile_line("Nonlinear High", nonlinear_profiles.high),
                profile_line("Nonlinear Low", nonlinear_profiles.low),
            ]
        )
    md_lines = [
        "# Automatic High/Low KP Model Selection",
        "",
        *profile_lines,
        "",
        "Both profiles are selected using the fixed primary edge-band weights and the subspace-overlap gate. Low uses an adaptive statistical/material-difficulty tolerance. The expanded/all-band error is diagnostic only and never disqualifies the low profile.",
        "",
        "| candidate | solver | orders k/i/t | real parameters | groups | harmonics | primary RMS ± SE (meV) | max (meV) | overlap | all-band RMS (meV) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        selected_marks = []
        if candidate.name == high_name:
            selected_marks.append("high")
        if candidate.name == low_name:
            selected_marks.append("low")
        label = candidate.name
        if selected_marks:
            label += " (" + ", ".join(selected_marks) + ")"
        expanded = candidate.expanded_weighted_rms_mev
        md_lines.append(
            f"| {label} | {candidate.solver_family} | {candidate.orders.kinetic}/{candidate.orders.intra}/{candidate.orders.inter} | "
            f"{candidate.independent_real_parameters} | {candidate.active_group_count} | "
            f"{candidate.harmonic_support_size} | {candidate.weighted_rms_mev:.3f} ± "
            f"{candidate.weighted_rms_se_mev:.3f} | {candidate.weighted_max_mev:.3f} | "
            f"{candidate.mean_subspace_overlap:.6f} | "
            f"{'n/a' if expanded is None else f'{expanded:.3f}'} |"
        )
    (output_dir / "auto_model_selection.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=KP_DPI)
    if candidates:
        parameter_counts = np.asarray(
            [candidate.independent_real_parameters for candidate in candidates],
            dtype=float,
        )
        errors = np.asarray(
            [candidate.weighted_rms_mev for candidate in candidates],
            dtype=float,
        )
        error_bars = np.asarray(
            [candidate.weighted_rms_se_mev for candidate in candidates],
            dtype=float,
        )
        ax.errorbar(
            parameter_counts,
            errors,
            yerr=error_bars,
            fmt="o",
            color=KP_REFERENCE_STYLE["color"],
            alpha=0.75,
            capsize=2.5,
            label="candidates",
        )
        for label, selected, color in (
            ("high", profiles.high.selected, KP_PRIMARY_STYLE["color"]),
            ("low", profiles.low.selected, KP_MODEL_STYLE["color"]),
        ):
            if selected is not None:
                ax.scatter(
                    [selected.independent_real_parameters],
                    [selected.weighted_rms_mev],
                    s=85,
                    marker="*",
                    color=color,
                    label=label,
                    zorder=5,
                )
    ax.set_xlabel("independent real parameters")
    ax.set_ylabel("primary weighted validation RMS (meV)")
    ax.set_title("KP model complexity frontier")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(**KP_LEGEND_KWARGS)
    apply_kp_axis_style(ax, box_aspect=KP_BAND_BOX_ASPECT, font_family=kp_font_family())
    fig.tight_layout()
    fig.savefig(output_dir / "model_complexity_frontier.pdf", dpi=KP_DPI)
    plt.close(fig)
    return summary



def _materialize_automatic_profile_output(
    *,
    profile_name: str,
    profile: Mapping[str, Any],
    target_hamiltonians: np.ndarray,
    root_output_dir: Path,
) -> dict[str, Any]:
    """Write one selected profile as a complete configured-model output."""
    from ..viz import model_diagnostics
    from . import validation


    name = str(profile_name).strip().lower()
    parts = tuple(part for part in name.split("/") if part)
    if parts not in {
        ("high",),
        ("low",),
        ("linear", "high"),
        ("linear", "low"),
        ("nonlinear", "high"),
        ("nonlinear", "low"),
    }:
        raise ValueError(
            "automatic profile name must be high/low or "
            f"linear/nonlinear high/low, got {profile_name!r}"
        )
    output_dir = Path(root_output_dir).joinpath(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = dict(profile["results"])
    profile["results"] = results
    moire_config = profile["moire_config"]
    original_model_config = profile["model_config"]
    model_config = replace(original_model_config, output_dir=output_dir)
    profile["model_config"] = model_config

    eig_output = results["eigvals"]
    model_eigvecs = None
    if isinstance(eig_output, tuple):
        eigvals = np.asarray(eig_output[0], dtype=float)
        if len(eig_output) > 1:
            model_eigvecs = np.asarray(eig_output[1], dtype=np.complex128)
    else:
        eigvals = np.asarray(eig_output, dtype=float)
    target = np.asarray(target_hamiltonians, dtype=np.complex128)
    target_eigvals, target_eigvecs = np.linalg.eigh(target)
    if eigvals.shape != target_eigvals.shape:
        raise ValueError(
            f"{name} profile eigenvalue shape {eigvals.shape} does not match "
            f"selection target {target_eigvals.shape}"
        )
    np.save(output_dir / "eigvals.npy", eigvals)
    np.save(output_dir / "current_heff_support_eigvals.npy", target_eigvals)

    heff_matrix_all = configuration._load_heff_in_model_basis(model_config)
    heff_matrix_selected = np.asarray(
        configuration._select_rows(heff_matrix_all, model_config.band_indices)
    )
    original_heff_eigvals = np.linalg.eigvalsh(heff_matrix_selected)
    comparison = {
        **validation.compare_bands(
            eigvals,
            target_eigvals,
            band_slice=model_config.band_slice,
        ),
        "reference": "current_heff_support_mask",
    }
    comparison_vs_full_heff = {
        **validation.compare_bands(
            eigvals,
            original_heff_eigvals,
            band_slice=model_config.band_slice,
        ),
        "reference": "original_heff",
    }
    target_bands = str(
        model_config.raw.get("model", {}).get("target_bands", "top")
    ).strip().lower()
    band_plot_config = validation._window_band_plot_config(
        model_config.band_plot_config,
        target_bands=target_bands,
    )
    overlap_weights = None
    if model_eigvecs is not None:
        if model_eigvecs.shape != target_eigvecs.shape:
            raise ValueError(
                f"{name} profile eigenvector shape {model_eigvecs.shape} does not "
                f"match selection target {target_eigvecs.shape}"
            )
        overlap_weights = model_diagnostics._band_overlap_weights(
            model_eigvecs,
            target_eigvecs,
            model_eigvals=eigvals,
            reference_eigvals=target_eigvals,
            degeneracy_tol=float(
                model_config.band_plot_config.get(
                    "overlap_degeneracy_tol",
                    3.0e-3,
                )
            ),
        )
        band_plot_config["plot_all_bands"] = True
    plot_comparison = validation.compare_bands_for_plot(
        eigvals,
        target_eigvals,
        band_slice=model_config.band_slice,
        plot_config=band_plot_config,
    )
    plot_comparison["reference"] = "current_heff_support_mask"
    plot_comparison_vs_full_heff = validation.compare_bands_for_plot(
        eigvals,
        original_heff_eigvals,
        band_slice=model_config.band_slice,
        plot_config=band_plot_config,
    )
    plot_comparison_vs_full_heff["reference"] = "original_heff"
    all_band_config = validation._all_band_plot_config(model_config.band_plot_config)
    all_band_plot_comparison = validation.compare_bands_for_plot(
        eigvals,
        target_eigvals,
        band_slice=None,
        plot_config=all_band_config,
    )
    all_band_plot_comparison["reference"] = "current_heff_support_mask"
    all_band_plot_comparison_vs_full_heff = validation.compare_bands_for_plot(
        eigvals,
        original_heff_eigvals,
        band_slice=None,
        plot_config=all_band_config,
    )
    all_band_plot_comparison_vs_full_heff["reference"] = "original_heff"

    x_values, x_ticks, x_ticklabels = model_diagnostics._plot_axis_from_kpath(
        model_config,
        eigvals.shape[0],
    )
    band_plot_path = model_diagnostics.save_band_comparison_plot(
        eigvals,
        target_eigvals,
        output_dir / "band_comparison.pdf",
        band_slice=model_config.band_slice,
        plot_config=band_plot_config,
        x=x_values,
        x_ticks=x_ticks,
        x_ticklabels=x_ticklabels,
        title=f"{model_diagnostics._band_plot_title(model_config)} ({name})",
        overlap_weights=overlap_weights,
        reference_label="Current Heff support mask",
    )
    full_heff_plot_path = model_diagnostics.save_band_comparison_plot(
        eigvals,
        original_heff_eigvals,
        output_dir / "band_comparison_vs_full_heff.pdf",
        band_slice=model_config.band_slice,
        plot_config=band_plot_config,
        x=x_values,
        x_ticks=x_ticks,
        x_ticklabels=x_ticklabels,
        title=f"{model_diagnostics._band_plot_title(model_config)} ({name}, vs original Heff)",
        reference_label="Original Heff",
    )
    all_band_plot_path = model_diagnostics.save_band_comparison_plot(
        eigvals,
        target_eigvals,
        output_dir / "band_comparison_all.pdf",
        band_slice=None,
        plot_config=all_band_config,
        x=x_values,
        x_ticks=x_ticks,
        x_ticklabels=x_ticklabels,
        title=f"{model_diagnostics._band_plot_title(model_config)} ({name}, all bands)",
        reference_label="Current Heff support mask",
    )
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
            if isinstance(pair, Sequence)
            and not isinstance(pair, (str, bytes))
            and len(pair) == 2
        ],
    )

    score = profile.get("score")
    selected_record = (
        _model_selection_candidate_record(score)
        if isinstance(score, CandidateScore)
        else _json_safe(score)
    )
    results.update(
        {
            "configured_model": model_config,
            "moire_config": moire_config,
            "comparison": comparison,
            "plot_comparison": plot_comparison,
            "all_band_plot_comparison": all_band_plot_comparison,
            "comparison_vs_full_heff": comparison_vs_full_heff,
            "plot_comparison_vs_full_heff": plot_comparison_vs_full_heff,
            "all_band_plot_comparison_vs_full_heff": all_band_plot_comparison_vs_full_heff,
            "band_plot": str(Path(band_plot_path).resolve()),
            "full_heff_band_plot": str(Path(full_heff_plot_path).resolve()),
            "all_band_plot": str(Path(all_band_plot_path).resolve()),
            "q_lattice_plot": str(Path(q_lattice_plot_path).resolve()),
            "auto_model_selection": {
                "enabled": True,
                "mode": (
                    "automatic_four_model_profiles"
                    if len(parts) == 2
                    else "automatic_high_low_family_orders"
                ),
                "profile": name,
                "status": str(profile.get("status", "unknown")),
                "selected": selected_record,
                "reference": "current_heff_support_mask",
            },
        }
    )
    selected_raw = copy.deepcopy(model_config.raw)
    selected_model_section = selected_raw.setdefault("model", {})
    if isinstance(selected_model_section, dict):
        selected_model_section["max_order"] = dict(model_config.max_order)
        selected_model_section["response_semantics"] = str(
            model_config.response_semantics
        )
        selected_model_section["term_templates"] = [
            copy.deepcopy(row) for row in model_config.term_templates
        ]
        selected_fit = selected_model_section.setdefault("fit", {})
        if isinstance(selected_fit, dict):
            selected_fit["objective"] = copy.deepcopy(
                model_config.response_fit_objective
            )
            selected_fit["refine_bands"] = copy.deepcopy(
                model_config.band_refinement_config
            )
    (output_dir / "selected_model_config.yaml").write_text(
        yaml.safe_dump(_json_safe(selected_raw), sort_keys=False),
        encoding="utf-8",
    )
    validations, validation_summary = validation._compute_validation_outputs(
        results=results,
        model_config=model_config,
        moire_config=moire_config,
    )
    results["validations"] = validations
    results["validation_summary"] = validation_summary
    _write_model_registry_outputs(
        results=results,
        output_dir=output_dir,
        model_config=model_config,
        validations=validations,
        validation_summary=validation_summary,
        diagnostics_dir=None,
    )
    active_terms_path = output_dir / "active_terms.json"
    active_terms_payload = json.loads(active_terms_path.read_text(encoding="utf-8"))
    if isinstance(active_terms_payload, list) and active_terms_payload:
        from .export import export_standalone_model

        try:
            export_standalone_model(output_dir, output_dir, force=True)
        except (FileNotFoundError, ValueError) as exc:
            if len(parts) == 2:
                raise
            results["standalone_export"] = {
                "status": "skipped_unavailable_metadata",
                "reason": str(exc),
            }
        else:
            results["standalone_export"] = {"status": "exported"}
    else:
        results["standalone_export"] = {"status": "skipped_empty_model"}
    return results



def _null_channel_filter_summary(model: Any, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    reports = list(getattr(model, "_null_channel_filter_reports", []) or [])
    filtered = int(sum(int(row.get("filtered", 0)) for row in reports))
    summary: dict[str, Any] = {
        "enabled": bool(float(abs_tol or 0.0) > 0.0 or float(rel_tol or 0.0) > 0.0),
        "abs_tol": float(abs_tol or 0.0),
        "rel_tol": float(rel_tol or 0.0),
        "filtered_columns": filtered,
        "blocks_with_filtered_columns": len(reports),
    }
    if reports:
        min_norms = [float(row["min_filtered_norm"]) for row in reports if "min_filtered_norm" in row]
        max_norms = [float(row["max_filtered_norm"]) for row in reports if "max_filtered_norm" in row]
        min_ratios = [float(row["min_filtered_ratio"]) for row in reports if "min_filtered_ratio" in row]
        max_ratios = [float(row["max_filtered_ratio"]) for row in reports if "max_filtered_ratio" in row]
        summary.update(
            {
                "min_filtered_norm": float(min(min_norms)) if min_norms else None,
                "max_filtered_norm": float(max(max_norms)) if max_norms else None,
                "min_filtered_ratio": float(min(min_ratios)) if min_ratios else None,
                "max_filtered_ratio": float(max(max_ratios)) if max_ratios else None,
                "reports": reports,
            }
        )
    return summary



def _write_model_registry_outputs(
    *,
    results: Mapping[str, Any],
    output_dir: Path,
    model_config: configuration.ConfiguredModel,
    validations: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    diagnostics_dir: Path | None = None,
) -> None:
    model = results.get("model")
    terms = list(getattr(model, "terms", {}).values()) if model is not None else []
    active_terms = [term for term in terms if bool(getattr(term, "active", False))]
    active = [_term_to_release_dict(term) for term in active_terms]
    active_terms_text = json.dumps(_json_safe(active), indent=2)
    with (output_dir / "active_terms.json").open("w", encoding="utf-8") as handle:
        handle.write(active_terms_text)
    active_terms_hash = hashlib.sha256(active_terms_text.encode("utf-8")).hexdigest()
    with (output_dir / "fit_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(model_config.fit_selection_metadata), handle, indent=2)
    with (output_dir / "active_terms.sha256").open("w", encoding="utf-8") as handle:
        handle.write(f"{active_terms_hash}\n")
    operation_registry = _build_operation_registry(model_config)
    output_profile = "debug" if diagnostics_dir is not None else "release"
    if diagnostics_dir is not None:
        rows = [_term_to_dict(term) for term in terms]
        discarded = [row for row in rows if not row["active"]]
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        with (diagnostics_dir / "terms.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        with (diagnostics_dir / "discarded_terms.json").open("w", encoding="utf-8") as handle:
            json.dump(discarded, handle, indent=2)
        with (diagnostics_dir / "coefficients.json").open("w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "key": row["key"],
                        "tag": row["tag"],
                        "r_value_real": row["r_value_real"],
                        "r_value_imag": row["r_value_imag"],
                    }
                    for row in active
                ],
                handle,
                indent=2,
            )
        with (diagnostics_dir / "fit_diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(results.get("diagnostics", {})), handle, indent=2)
        for name in ("orthogonalization", "matrix_residuals", "block_residuals", "wavefunction_overlap", "symmetry_residuals"):
            with (diagnostics_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
                json.dump(_json_safe(validations.get(name, {"available": False, "reason": "missing"})), handle, indent=2)
        with (diagnostics_dir / "operation_registry.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(operation_registry), handle, indent=2)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(
                _build_run_summary(
                    model_config=model_config,
                    results=results,
                    operation_registry=operation_registry,
                    validation_summary=validation_summary,
                    active_terms_hash=active_terms_hash,
                    output_profile=output_profile,
                )
            ),
            handle,
            indent=2,
        )



def _subspace_leakage_curve(model_basis: np.ndarray, target_basis: np.ndarray) -> list[float]:
    model_u = np.asarray(model_basis, dtype=np.complex128)
    target_u = np.asarray(target_basis, dtype=np.complex128)
    if model_u.shape != target_u.shape or model_u.ndim != 3:
        raise ValueError(f"subspace leakage curve expects matching (Nk,dim,N) arrays, got {model_u.shape}, {target_u.shape}")
    n_state = int(model_u.shape[2])
    values: list[float] = []
    for idx in range(model_u.shape[0]):
        singular_values = np.linalg.svd(target_u[idx].conj().T @ model_u[idx], compute_uv=False)
        overlap = float(np.sum(np.clip(singular_values, 0.0, 1.0) ** 2) / n_state)
        values.append(float(max(0.0, 1.0 - overlap)))
    return values



def _write_auto_model_selection_outputs(
    *,
    results: Mapping[str, Any],
    output_dir: Path,
    model_config: configuration.ConfiguredModel,
    moire_config: MoireConfig,
) -> dict[str, Any]:
    from . import fit_sampling
    from ..viz import model_diagnostics
    from . import refinement
    from . import validation

    refine_cfg = dict(model_config.band_refinement_config or {})
    fit_mode = str(model_config.fit_selection_metadata.get("mode", "")).strip().lower()
    if fit_mode != "auto_low_energy" and str(refine_cfg.get("mode", "")).strip().lower() != "auto_low_energy":
        return {"enabled": False}
    if moire_config.kpoints is None:
        return {"enabled": False, "reason": "missing_kpoints"}
    model = results.get("model")
    if model is None:
        return {"enabled": False, "reason": "missing_model"}

    original_heff_all = configuration._select_rows(
        configuration._load_heff_in_model_basis(model_config),
        model_config.band_indices,
    )
    heff_all = validation._current_heff_support_hamiltonians(
        original_heff_all,
        qset1=np.asarray(moire_config.Q_set1, dtype=float),
        qset2=np.asarray(moire_config.Q_set2, dtype=float),
        n_orb=model_config.n_orb,
        harmonics_config=model_config.harmonics_config,
    )
    h_model = refinement._model_hamiltonians_for_kpoints(moire_config, model, np.asarray(moire_config.kpoints, dtype=float))
    heff_eig, heff_vec = np.linalg.eigh(heff_all)
    original_heff_eig = np.linalg.eigvalsh(original_heff_all)
    model_eig, model_vec = np.linalg.eigh(h_model)
    target_bands = str(refine_cfg.get("target_bands", model_config.raw.get("model", {}).get("target_bands", "top"))).strip().lower()
    auto_windows = refine_cfg.get("auto_windows")
    if not isinstance(auto_windows, Mapping):
        auto_windows = fit_sampling._auto_low_energy_windows(
            heff_eig,
            n_primary=max(1, int(sum(model_config.n_orb))),
            target_bands=target_bands,
            gap_tolerance_mev=float(refine_cfg.get("gap_tolerance_mev", 0.1)),
        )
    window_records = [dict(auto_windows.get("primary", {}))]
    weighted_fit_window = auto_windows.get("weighted_fit")
    if isinstance(weighted_fit_window, Mapping):
        primary_slice = tuple(window_records[0].get("band_slice", ())) if window_records else ()
        weighted_slice = tuple(weighted_fit_window.get("band_slice", ()))
        if weighted_slice and weighted_slice != primary_slice:
            window_records.append(dict(weighted_fit_window))
    expanded = auto_windows.get("expanded", [])
    if isinstance(expanded, Sequence) and not isinstance(expanded, (str, bytes)):
        window_records.extend(dict(item) for item in expanded if isinstance(item, Mapping))
    window_records = [item for item in window_records if item.get("band_slice") is not None]
    if not window_records:
        primary_slice = fit_sampling._auto_low_energy_band_slice(heff_all.shape[-1], max(1, int(sum(model_config.n_orb))), target_bands)
        window_records = [{"role": "primary", "n_bands": primary_slice[1] - primary_slice[0], "band_slice": primary_slice}]

    raw_delta = h_model - heff_all
    active_terms = [
        term for term in getattr(model, "terms", {}).values()
        if bool(getattr(term, "active", True))
    ]
    n_variables = int((results.get("band_refinement") or {}).get("n_variables", 0) or 0)
    candidate = {
        "name": "configured_auto_low_energy",
        "selected": True,
        "reference": "current_heff_support_mask",
        "active_terms": int(len(active_terms)),
        "n_variables": n_variables,
        "fit_selection": _json_safe(model_config.fit_selection_metadata),
        "windows": [],
        "raw_matrix": {
            "rms_mev": float(np.sqrt(np.mean(np.abs(raw_delta) ** 2)) * 1000.0),
            "max_mev": float(np.max(np.abs(raw_delta)) * 1000.0),
        },
    }
    matrix_report: dict[str, Any] = {"raw_matrix": candidate["raw_matrix"], "windows": []}
    primary_leakage_curve: list[float] | None = None
    for window in window_records:
        band_slice_values = list(window["band_slice"])
        band_slice = (int(band_slice_values[0]), int(band_slice_values[1]))
        band_metrics = refinement._band_refinement_metrics(
            h_model,
            heff_eig,
            heff_all,
            band_slice=band_slice,
            align=str(refine_cfg.get("align", target_bands)),
        )
        target_basis = heff_vec[:, :, band_slice[0] : band_slice[1]]
        model_basis = model_vec[:, :, band_slice[0] : band_slice[1]]
        subspace_metrics = refinement._subspace_overlap_metrics(model_basis, target_basis)
        _low_vec, low_matrix_metrics = refinement._low_subspace_matrix_residual(
            raw_delta,
            target_basis,
            weight=1.0,
            sigma_mev=1.0,
            normalize=False,
        )
        window_payload = {
            "role": str(window.get("role", "window")),
            "n_bands": int(window.get("n_bands", band_slice[1] - band_slice[0])),
            "band_slice": [int(band_slice[0]), int(band_slice[1])],
            "boundary_gap_mev": window.get("boundary_gap_mev"),
            "state": window.get("state", "unknown"),
            "use_for_loss": bool(window.get("use_for_loss", True)),
            "band": band_metrics,
            "subspace": subspace_metrics,
            "low_subspace_matrix": low_matrix_metrics,
        }
        candidate["windows"].append(window_payload)
        matrix_report["windows"].append(
            {
                "role": window_payload["role"],
                "n_bands": window_payload["n_bands"],
                "band_slice": window_payload["band_slice"],
                "low_subspace_matrix": low_matrix_metrics,
            }
        )
        if window_payload["role"] == "primary":
            primary_leakage_curve = _subspace_leakage_curve(model_basis, target_basis)

    primary = candidate["windows"][0]
    secondary_plot = validation.compare_bands_for_plot(
        model_eig,
        original_heff_eig,
        band_slice=primary["band_slice"],
        plot_config={**model_config.band_plot_config, "band_slice": primary["band_slice"]},
    )
    candidate["secondary_original_heff"] = {
        "reference": "original_heff",
        "plot_rms_mev": float(secondary_plot["rms_error_mev"]),
        "plot_max_mev": float(secondary_plot["max_abs_error_mev"]),
        "comparison": secondary_plot,
    }

    summary = {
        "enabled": True,
        "mode": "auto_low_energy",
        "selected_candidate": "configured_auto_low_energy",
        "selection_rule": "single_candidate_phase1_metrics",
        "fit_candidate_selection": _json_safe(model_config.fit_selection_metadata.get("candidate_scan", {"enabled": False})),
        "harmonic_selection": _json_safe(refine_cfg.get("auto_harmonic_selection", {"enabled": False})),
        "candidates": [candidate],
        "band_refinement": _json_safe(results.get("band_refinement", {})),
        "weighted_band_loss": _json_safe((results.get("band_refinement") or {}).get("weighted_band_loss", {})),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "auto_model_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2)
    with (output_dir / "matrix_residual_report.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(matrix_report), handle, indent=2)
    with (output_dir / "selected_model_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_json_safe(model_config.raw), handle, sort_keys=False)

    csv_lines = [
        "candidate,selected,window,n_bands,band_rms_mev,band_max_mev,mean_overlap,max_leakage,min_singular_value,low_matrix_rms_mev,low_matrix_max_mev,active_terms,n_variables"
    ]
    for window in candidate["windows"]:
        csv_lines.append(
            ",".join(
                [
                    candidate["name"],
                    "true",
                    str(window["role"]),
                    str(window["n_bands"]),
                    f"{float(window['band']['top_band_rms_mev']):.12g}",
                    f"{float(window['band']['top_band_max_mev']):.12g}",
                    f"{float(window['subspace']['mean_overlap']):.12g}",
                    f"{float(window['subspace']['max_leakage']):.12g}",
                    f"{float(window['subspace']['min_singular_value']):.12g}",
                    f"{float(window['low_subspace_matrix']['rms_mev'] or 0.0):.12g}",
                    f"{float(window['low_subspace_matrix']['max_mev'] or 0.0):.12g}",
                    str(candidate["active_terms"]),
                    str(candidate["n_variables"]),
                ]
            )
        )
    (output_dir / "candidate_metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# Auto Low-Energy Model Selection",
        "",
        f"Selected candidate: `{candidate['name']}`.",
        "",
        "Primary energy, matrix, and subspace metrics use the current Heff support mask. Original Heff is retained only as a labelled secondary diagnostic.",
        "",
    ]
    harmonic_selection = refine_cfg.get("auto_harmonic_selection", {"enabled": False})
    fit_candidate_selection = model_config.fit_selection_metadata.get("candidate_scan", {"enabled": False})
    if isinstance(fit_candidate_selection, Mapping) and bool(fit_candidate_selection.get("enabled", False)):
        selected_fit = fit_candidate_selection.get("selected", {})
        if isinstance(selected_fit, Mapping):
            md_lines.extend(
                [
                    "## Fit K-Point Selection",
                    "",
                    "Prefit candidate scan selected "
                    f"`{selected_fit.get('name', 'unknown')}` with indices "
                    f"`{selected_fit.get('indices', [])}`.",
                    "",
                ]
            )
    if isinstance(harmonic_selection, Mapping) and bool(harmonic_selection.get("enabled", False)):
        selected_harmonics = harmonic_selection.get("selected", {})
        if isinstance(selected_harmonics, Mapping):
            md_lines.extend(
                [
                    "## Harmonic Selection",
                    "",
                    "Heff harmonic ablation selected "
                    f"`intra={int(selected_harmonics.get('intra_shells', 0))}`, "
                    f"`inter={int(selected_harmonics.get('inter_shells', 0))}` "
                    f"with status `{harmonic_selection.get('selection_status', 'unknown')}`.",
                    "",
                ]
            )
    md_lines.extend(
        [
            "| window | N | band RMS/max (meV) | mean overlap | max leakage | low-matrix RMS/max (meV) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for window in candidate["windows"]:
        md_lines.append(
            "| {role} | {n} | {rms:.3f}/{maxv:.3f} | {overlap:.6f} | {leak:.3e} | {mrms:.3f}/{mmax:.3f} |".format(
                role=window["role"],
                n=window["n_bands"],
                rms=float(window["band"]["top_band_rms_mev"]),
                maxv=float(window["band"]["top_band_max_mev"]),
                overlap=float(window["subspace"]["mean_overlap"]),
                leak=float(window["subspace"]["max_leakage"]),
                mrms=float(window["low_subspace_matrix"]["rms_mev"] or 0.0),
                mmax=float(window["low_subspace_matrix"]["max_mev"] or 0.0),
            )
        )
    md_lines.extend(
        [
            "",
            f"Active terms: `{candidate['active_terms']}`; refined variables: `{candidate['n_variables']}`.",
            "",
            "Phase 1 uses the configured auto-low-energy model as the selected candidate and writes the same metric schema used by later multi-candidate scans.",
        ]
    )
    (output_dir / "auto_model_selection.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    model_diagnostics.save_band_comparison_plot(
        model_eig,
        heff_eig,
        output_dir / "band_comparison_top_primary.pdf",
        band_slice=primary["band_slice"],
        plot_config={**model_config.band_plot_config, "band_slice": primary["band_slice"]},
        title="Auto low-energy primary window",
        reference_label="Current Heff support mask",
    )
    model_diagnostics.save_band_comparison_plot(
        model_eig,
        original_heff_eig,
        output_dir / "band_comparison_top_vs_full_heff.pdf",
        band_slice=primary["band_slice"],
        plot_config={**model_config.band_plot_config, "band_slice": primary["band_slice"]},
        title="Auto low-energy primary window (vs original Heff)",
        reference_label="Original Heff",
    )
    weighted_window = next((item for item in candidate["windows"] if item.get("role") == "weighted_fit"), None)
    if weighted_window is not None:
        model_diagnostics.save_band_comparison_plot(
            model_eig,
            heff_eig,
            output_dir / "band_comparison_weighted_fit.pdf",
            band_slice=weighted_window["band_slice"],
            plot_config={**model_config.band_plot_config, "band_slice": weighted_window["band_slice"]},
            title="Auto low-energy weighted fit window",
            reference_label="Current Heff support mask",
        )
    if primary_leakage_curve is not None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=KP_BAND_FIGSIZE, dpi=KP_DPI)
        x = np.arange(len(primary_leakage_curve))
        ax.plot(x, 100.0 * np.asarray(primary_leakage_curve), label="Leakage", **KP_PRIMARY_STYLE, **KP_MARKER_STYLE)
        ax.set_xlabel("k-path index")
        ax.set_ylabel("subspace leakage (%)")
        ax.set_title("Primary low-energy subspace leakage")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(**KP_LEGEND_KWARGS)
        apply_kp_axis_style(ax, box_aspect=KP_BAND_BOX_ASPECT, font_family=kp_font_family())
        fig.tight_layout()
        fig.savefig(output_dir / "subspace_leakage.pdf", dpi=KP_DPI)
        plt.close(fig)
    return summary



def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    return value
