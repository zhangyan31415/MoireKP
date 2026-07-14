from __future__ import annotations

import ast
import copy
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import joblib
import scipy.linalg
import scipy.optimize
import scipy.sparse
import yaml

from .schema import (
    canonical_operation_name_for_valley,
    canonical_source_operation_name_for_valley,
    effective_operation_metadata_for_valley,
    validate_model_config,
)
from .symmetry import load_symmetry_source
from ..config.case import normalize_case_config
from ..identity import (
    PROJECTION_ARTIFACT_IDENTITY_FIELDS,
    exactified_operation_provenance_is_complete,
    hash_array,
    load_projection_artifact_identity,
    load_projection_k_indices,
    require_identity_fields,
    require_matching_identity,
)
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
    kp_plot_rc_context,
    kp_font_family,
    relative_energy_ylabel,
)
from ..symmetry.action_schema import SOURCE_MATRIX_SEMANTICS, allows_inferred_action_metadata
from ..symmetry.geometry import (
    bM_candidates_from_q_distances,
    canonical_bM_pair_from_candidates,
    sector_qset,
    sectors_with_q_offsets,
)
from .core import (
    ContinuumModelBuilder,
    MoireConfig,
    SymmetryGenerator,
    _compute_one_k,
    _prepare_band_state,
    build_model,
    compute_bands,
    compute_coefficients,
    generate_kpath_from_file,
    load_Q_sets_from_gvec_files,
    reciprocal_Tmat_from_Tmat,
    rot,
    setup_logging,
)

DEFAULT_MAX_ORDER = {"Kinect": 2, "intra": 0, "inter": 0}
HARMONICS_USER_ALIASES = {
    "intralayer": "intra",
    "interlayer": "inter",
}
MAX_ORDER_USER_ALIASES = {
    "kinetic": "Kinect",
    "intralayer": "intra",
    "interlayer": "inter",
}
SYMMETRY_MAP_USER_ALIASES = {
    "kinetic": "Kinect",
    "onsite": "Onsite",
    "intralayer": "intra",
    "interlayer": "inter",
}
MAX_DERIVATIVE_ORDER_USER_ALIASES = {
    "intralayer_zero": "moire_intra_zero",
    "intralayer_nonzero": "moire_intra_nonzero",
    "interlayer_zero": "tunneling_zero",
    "interlayer_nonzero": "tunneling_nonzero",
}
GAMMA_LEGACY_ORDER_ALIASES = {
    "tunneling_zero": "inter",
    "tunneling_nonzero": "inter",
    "moire_intra_zero": "intra",
    "moire_intra_nonzero": "intra",
    "kinetic": "Kinect",
}
SOURCE_META_KEYS = (
    "source_matrix_role",
    "source_gauge",
    "target_role",
    "gauge_correction",
    "antiunitary_convention",
    "spin_map",
    "valley_map",
)
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


@dataclass
class ConfiguredModel:
    path: Path
    raw: dict[str, Any]
    source_config: Path
    source_raw: dict[str, Any]
    qset1_file: Path
    qset2_file: Path
    kpoints_file: Path | None
    heff_file: Path
    heff_eig_file: Path | None
    output_dir: Path
    rotation_deg: float
    fit_indices: list[int]
    fit_selection_metadata: dict[str, Any]
    band_indices: list[int] | None
    n_orb: tuple[int, int]
    nlow_state: list[int]
    bM_config: dict[str, Any]
    harmonics_config: dict[str, Any]
    max_order: dict[str, int]
    symmetry_map: dict[str, list[dict[str, Any]]]
    coeff_tol: float
    coeff_prune_threshold: float
    compare_to_heff: bool
    kpath_config: dict[str, Any] = field(default_factory=dict)
    valley_model: dict[str, Any] = field(default_factory=dict)
    symmetry_source_config: dict[str, Any] = field(default_factory=dict)
    symmetry_source_metadata: dict[str, Any] = field(default_factory=dict)
    sectors_config: list[dict[str, Any]] = field(default_factory=list)
    term_templates: list[dict[str, Any]] = field(default_factory=list)
    term_template_metadata: dict[str, Any] = field(default_factory=dict)
    output_config: dict[str, Any] = field(default_factory=dict)
    band_slice: list[int] | None = None
    band_plot_config: dict[str, Any] = field(default_factory=dict)
    bM_diagnostics: dict[str, Any] = field(default_factory=dict)
    harmonics_diagnostics: dict[str, Any] = field(default_factory=dict)
    orbital_count_metadata: dict[str, Any] = field(default_factory=dict)
    validation_config: dict[str, Any] = field(default_factory=dict)
    band_refinement_config: dict[str, Any] = field(default_factory=dict)
    null_channel_abs_tol: float = 0.0
    null_channel_rel_tol: float = 0.0
    artifact_identity: dict[str, Any] = field(default_factory=dict)
    project_k_indices: list[int] = field(default_factory=list)
    kpoints_from_projection: bool = False


def _resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data

SOURCE_SEMANTICS: dict[str, Any] = {
    "source_matrix_role": "raw_h_sewing_action",
    "source_gauge": "raw_saved_TAPW",
    "target_role": "continuum_internal_rep",
    "gauge_correction": {"kind": "none"},
    "spin_map": "from_kp_symm_output",
    "valley_map": "identity",
}

def _as_int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of integers, got {value!r}")
    return [int(item) for item in value]


def _num_layer_list_from_material(material: Mapping[str, Any]) -> list[int]:
    raw = material.get("num_layer_list")
    if raw is not None:
        layers = _as_int_list(raw, name="source_config.material.num_layer_list")
        if not layers or any(value <= 0 for value in layers):
            raise ValueError(f"source_config material.num_layer_list must contain positive integers, got {raw!r}")
        return layers
    num_layers = int(material.get("num_layers", 2))
    if num_layers <= 0:
        raise ValueError(f"source_config material.num_layers must be positive, got {num_layers}")
    if num_layers == 1:
        return [1, 0]
    return [1, num_layers - 1]


def _layer_groups_metadata(num_layer_list: Sequence[int], values: Sequence[int]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    layer_start = 1
    for group_index, layer_count_raw in enumerate(num_layer_list):
        layer_count = int(layer_count_raw)
        layer_stop = layer_start + layer_count
        group_values = [int(item) for item in values[layer_start - 1 : layer_stop - 1]]
        groups.append(
            {
                "qset": f"qset{group_index + 1}",
                "source_group": group_index + 1,
                "layers": list(range(layer_start, layer_stop)),
                "values": group_values,
                "total": int(sum(group_values)),
            }
        )
        layer_start = layer_stop
    return groups


def _resolve_layerwise_counts(
    values: Sequence[int],
    *,
    name: str,
    num_layer_list: Sequence[int],
    default_from: str | None = None,
) -> tuple[list[int], dict[str, Any]]:
    raw_values = [int(value) for value in values]
    qset_count = 2
    total_layers = int(sum(int(value) for value in num_layer_list))
    metadata: dict[str, Any] = {
        "raw": list(raw_values),
        "num_layer_list": [int(value) for value in num_layer_list],
        "total_layers": total_layers,
    }
    if default_from is not None:
        metadata["default_from"] = default_from

    if len(raw_values) == qset_count and (total_layers == qset_count or default_from is not None):
        metadata.update(
            {
                "input_kind": "default_from_n_orb" if default_from is not None else "legacy_qset",
                "resolved_qset": list(raw_values),
            }
        )
        return raw_values, metadata
    if len(raw_values) != total_layers:
        raise ValueError(
            f"{name} must have {total_layers} physical-layer entries "
            f"(sum(source_config material.num_layer_list)); got {len(raw_values)} values: {raw_values}"
        )
    if len(num_layer_list) != qset_count:
        raise ValueError(
            f"{name} physical-layer counts require exactly two source groups for qset1/qset2 resolution, "
            f"got source_config material.num_layer_list={list(num_layer_list)}"
        )

    groups = _layer_groups_metadata(num_layer_list, raw_values)
    qset_values = [int(group["total"]) for group in groups]
    active_layer_values = [int(value) for value in raw_values if int(value) > 0]
    resolved_model_sectors = active_layer_values if len(active_layer_values) == qset_count else qset_values
    metadata.update(
        {
            "input_kind": "default_from_n_orb" if default_from is not None else "physical_layer",
            "resolved_qset": list(qset_values),
            "resolved_model_sectors": list(resolved_model_sectors),
            "groups": groups,
        }
    )
    return resolved_model_sectors, metadata


def _infer_k_sectors_from_layerwise_counts(
    n_orb_resolution: Mapping[str, Any],
    *,
    valley_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if str(valley_model.get("valley_type", "")) != "K":
        return []
    if str(n_orb_resolution.get("input_kind", "")) != "physical_layer":
        return []
    groups = n_orb_resolution.get("groups", [])
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        return []

    sectors: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        qset = str(group.get("qset", ""))
        layers = group.get("layers", [])
        values = group.get("values", [])
        if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for layer, value in zip(layers, values):
            n_orb = int(value)
            if n_orb <= 0:
                continue
            sectors.append({"name": f"L{int(layer)}", "qset": qset, "n_orb": n_orb})
    return sectors


def _model_section(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    section = raw.get("model", {})
    if not isinstance(section, Mapping):
        raise ValueError("model section must be a mapping")
    return section


def _get_path_value(raw: Mapping[str, Any], key: str) -> Any:
    if key in raw:
        return raw[key]
    input_section = raw.get("input", {})
    if isinstance(input_section, Mapping) and key in input_section:
        return input_section[key]
    return None


def _infer_n_orb_pair_from_arrays(qset1_file: Path, qset2_file: Path, heff_file: Path) -> list[int]:
    q1 = np.load(qset1_file, mmap_mode="r")
    q2 = np.load(qset2_file, mmap_mode="r")
    heff = np.load(heff_file, mmap_mode="r")
    if heff.ndim < 2:
        raise ValueError(f"Cannot infer model.n_orb from heff shape {heff.shape}")
    dim = int(heff.shape[-1])
    total_q = int(q1.shape[0]) + int(q2.shape[0])
    if total_q <= 0 or dim % total_q != 0:
        raise ValueError(f"Cannot infer model.n_orb: heff dim={dim}, total Q count={total_q}")
    n_orb = dim // total_q
    return [int(n_orb), int(n_orb)]


def _default_symmetry_map_from_valley_model(valley_model: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    names = valley_model.get("allowed_internal_symmetries", [])
    if names is None:
        names = []
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise ValueError("valley_model.allowed_internal_symmetries must be a list")
    ops = [{"name": str(name)} for name in names]
    return {"Kinect": list(ops), "Onsite": list(ops), "intra": list(ops), "inter": list(ops)}


def _operation_rows_from_symmetry_source(
    symmetry_source: Mapping[str, Any],
    valley_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    operations = symmetry_source.get("operations", [])
    if isinstance(operations, Mapping):
        operations = [
            {**(dict(value) if isinstance(value, Mapping) else {}), "name": str(key)}
            for key, value in operations.items()
        ]
    if operations is None:
        operations = []
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("symmetry_source.operations must be a list or mapping when provided")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for operation in operations:
        if isinstance(operation, Mapping):
            raw_name = operation.get("name", operation.get("operation"))
        else:
            raw_name = operation
        if raw_name is None:
            continue
        user_name = str(raw_name)
        canonical_name = canonical_operation_name_for_valley(user_name, valley_model)
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        row = {
            "name": canonical_name,
            **effective_operation_metadata_for_valley(user_name, valley_model),
        }
        if user_name != canonical_name:
            row["user_name"] = user_name
        rows.append(row)
    return rows


def _default_symmetry_map_from_source_or_valley(
    symmetry_source: Mapping[str, Any],
    valley_model: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    ops = _operation_rows_from_symmetry_source(symmetry_source, valley_model)
    if not ops:
        return _default_symmetry_map_from_valley_model(valley_model)
    return {
        "Kinect": [dict(op) for op in ops],
        "Onsite": [dict(op) for op in ops],
        "intra": [dict(op) for op in ops],
        "inter": [dict(op) for op in ops],
    }


def _valley_type_from_label(label: str) -> str:
    if label == "Gamma":
        return "Gamma"
    if label.startswith("K"):
        return "K"
    if label.startswith("M"):
        return "M"
    raise ValueError(f"Unsupported valley label {label!r}; expected Gamma, K*, or M*")


def _spin_convention_from_short(spin: str, valley_type: str) -> str:
    if spin in {"spinful", "all"}:
        return "spinful"
    if spin in {"spin_up_projected", "up"}:
        return "spin_up_projected"
    if spin in {"spin_down_projected", "down"}:
        return "spin_down_projected"
    if spin == "spinless_effective":
        return "spinless_effective"
    raise ValueError(
        f"Unsupported spin value {spin!r}; use spinful, spin_up_projected, "
        "spin_down_projected, or spinless_effective"
    )


def _default_internal_symmetries(valley_type: str, spin_convention: str) -> list[str]:
    if valley_type == "Gamma":
        return ["C3z", "TR", "C2"] if spin_convention == "spinful" else ["C3z", "C2"]
    if valley_type == "K":
        return ["C3z", "C2T"]
    if valley_type == "M":
        return ["TR", "C2"]
    raise ValueError(f"Unsupported valley_type {valley_type!r}")


def _normalize_short_model_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)

    short_valley = out.get("valley")
    valley_model = out.get("valley_model")
    if short_valley is not None:
        valley_label = str(short_valley)
        valley_type = _valley_type_from_label(valley_label)
        spin = str(out.get("spin", "spinful" if valley_type == "Gamma" else "spinless"))
        spin_convention = _spin_convention_from_short(spin, valley_type)
        out["valley_model"] = {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": valley_type,
            "mode": "single_valley",
            "active_valleys": [valley_label],
            "spin_convention": spin_convention,
            "allowed_internal_symmetries": _default_internal_symmetries(valley_type, spin_convention),
            "external_sewing_symmetries": [],
        }
    elif not isinstance(valley_model, Mapping):
        out["valley_model"] = {}

    return out


def _normalize_alias_mapping(
    raw: Any,
    *,
    aliases: Mapping[str, str],
    section_name: str,
) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{section_name} must be a mapping")
    out = dict(raw)
    for alias, canonical in aliases.items():
        if alias not in out:
            continue
        if canonical in out and out[canonical] != out[alias]:
            raise ValueError(
                f"{section_name} uses both {canonical!r} and alias {alias!r}; keep only one spelling"
            )
        out[canonical] = out.pop(alias)
    return out


def _normalize_model_user_aliases(model: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(model)
    if "harmonics" in out:
        out["harmonics"] = _normalize_alias_mapping(
            out.get("harmonics"),
            aliases=HARMONICS_USER_ALIASES,
            section_name="model.harmonics",
        )
    if "max_order" in out:
        out["max_order"] = _normalize_alias_mapping(
            out.get("max_order"),
            aliases=MAX_ORDER_USER_ALIASES,
            section_name="model.max_order",
        )
    if "max_derivative_order" in out:
        out["max_derivative_order"] = _normalize_alias_mapping(
            out.get("max_derivative_order"),
            aliases=MAX_DERIVATIVE_ORDER_USER_ALIASES,
            section_name="model.max_derivative_order",
        )
    if "symmetry_map" in out:
        out["symmetry_map"] = _normalize_alias_mapping(
            out.get("symmetry_map"),
            aliases=SYMMETRY_MAP_USER_ALIASES,
            section_name="model.symmetry_map",
        )
    return out


def _normalize_user_symmetry_names(raw: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    valley_model = out.get("valley_model", {})
    if not isinstance(valley_model, Mapping):
        return out
    valley_model_out = dict(valley_model)
    allowed = valley_model_out.get("allowed_internal_symmetries")
    if isinstance(allowed, Sequence) and not isinstance(allowed, (str, bytes)):
        valley_model_out["allowed_internal_symmetries"] = [
            canonical_operation_name_for_valley(str(name), valley_model_out) for name in allowed
        ]
        valley_model_out["allowed_internal_symmetries_user"] = [str(name) for name in allowed]
    out["valley_model"] = valley_model_out

    model = out.get("model", {})
    if not isinstance(model, Mapping):
        return out
    model_out = dict(model)
    if "symmetry_map" not in model_out:
        out["model"] = model_out
        return out
    symmetry_map = model_out.get("symmetry_map", {})
    if isinstance(symmetry_map, Mapping):
        normalized_map: dict[str, list[dict[str, Any]]] = {}
        for tag, operations in symmetry_map.items():
            rows: list[dict[str, Any]] = []
            if isinstance(operations, Sequence) and not isinstance(operations, (str, bytes)):
                for operation in operations:
                    if isinstance(operation, Mapping):
                        row = dict(operation)
                        if "name" in row:
                            user_name = str(row["name"])
                            canonical_name = canonical_operation_name_for_valley(user_name, valley_model_out)
                            row.setdefault("user_name", user_name)
                            row["name"] = canonical_name
                            row.update(effective_operation_metadata_for_valley(user_name, valley_model_out))
                        rows.append(row)
                    else:
                        user_name = str(operation)
                        canonical_name = canonical_operation_name_for_valley(user_name, valley_model_out)
                        rows.append(
                            {
                                "user_name": user_name,
                                "name": canonical_name,
                                **effective_operation_metadata_for_valley(user_name, valley_model_out),
                            }
                        )
            normalized_map[str(tag)] = rows
        model_out["symmetry_map"] = normalized_map
    out["model"] = model_out
    return out


def _default_symmetry_source(
    raw: Mapping[str, Any],
    source_raw: Mapping[str, Any],
    *,
    source_base: Path,
) -> dict[str, Any]:
    source = raw.get("symmetry_source", {})
    if source is None:
        source = {}
    if isinstance(source, (str, Path)):
        out: dict[str, Any] = {"path": str(source)}
    elif isinstance(source, Mapping):
        out = dict(source)
    else:
        raise ValueError("symmetry_source must be a path string or mapping when provided")

    if out.get("path") and not out.get("type"):
        out["type"] = "kp_symm_output"
    if not out.get("type") and out.get("source") == "toy":
        out["type"] = "toy_generator"
    if not out.get("type") and not out.get("path"):
        symm = source_raw.get("symm", {})
        if isinstance(symm, Mapping) and symm.get("output_dir"):
            out["type"] = "kp_symm_output"
            out["inferred_from_source_config"] = True
    if out.get("type") == "kp_symm_output" and not out.get("path"):
        symm = source_raw.get("symm", {})
        if isinstance(symm, Mapping) and symm.get("output_dir"):
            resolved = _resolve_path(symm.get("output_dir"), source_base)
            if resolved is not None:
                out["path"] = str(resolved)
                out["inferred_from_source_config"] = True
    if out.get("type") == "kp_symm_output":
        out.setdefault("use", "raw")
        if isinstance(raw.get("valley_model"), Mapping):
            out.setdefault("valley_model", dict(raw["valley_model"]))
        symm = source_raw.get("symm", {})
        if "operations" not in out and isinstance(symm, Mapping) and symm.get("operations"):
            out["operations"] = list(symm.get("operations", []))
    return out


def _source_matrix_file(source_name: str, use: str, matrix_kind: str) -> str:
    if matrix_kind != "action":
        raise ValueError("kp_symm_output production symmetry_source.matrix_kind must be 'action'")
    return f"{source_name}_low_{use}.npy"


def _default_source_operation_name(name: str) -> str:
    if name in {"C3z"}:
        return "C3"
    if name in {"TR"}:
        return "TR"
    return name


def _matrix_file_stem(source_operation: str) -> str:
    return source_operation


def _complete_kp_symm_operation_entry(
    operation: Any,
    valley_model: Mapping[str, Any],
    *,
    use: str,
    matrix_kind: str,
    allow_inferred: bool = False,
) -> dict[str, Any]:
    row = dict(operation) if isinstance(operation, Mapping) else {"name": str(operation)}
    user_name = str(row.get("name", row.get("operation", "")))
    name = canonical_source_operation_name_for_valley(user_name, valley_model)
    source_operation = str(row.get("operation", _default_source_operation_name(name)))
    if matrix_kind != "action":
        raise ValueError("kp_symm_output production symmetry_source.matrix_kind must be 'action'")
    required_fields = (
        "antiunitary",
        "k_map",
        "q_map",
        "sector_map",
        "source_matrix_role",
        "source_gauge",
        "target_role",
        "gauge_correction",
        "antiunitary_convention",
    )
    missing_action = [field for field in required_fields if field not in row]
    if allow_inferred and "q_map" in missing_action and "k_map" in row:
        missing_action.remove("q_map")
    if missing_action:
        raise ValueError(
            f"symmetry_source operation {user_name!r} requires explicit action metadata when matrix_kind is set; "
            f"missing {missing_action}"
        )
    completed = {
        "name": name,
        "operation": source_operation,
        "matrix_file": _source_matrix_file(_matrix_file_stem(source_operation), use, matrix_kind),
        **effective_operation_metadata_for_valley(user_name, valley_model),
    }
    completed.update(row)
    completed["name"] = name
    completed.setdefault("operation", source_operation)
    if allow_inferred and "q_map" not in completed and "k_map" in completed:
        completed["q_map"] = copy.deepcopy(completed["k_map"])
        completed["inferred_fields"] = sorted(set([*completed.get("inferred_fields", []), "q_map"]))
    return completed


def _normalize_kp_symm_source(raw: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    valley_model = out.get("valley_model", {})
    symmetry_source = out.get("symmetry_source", {})
    if not isinstance(valley_model, Mapping) or not isinstance(symmetry_source, Mapping):
        return out
    if str(symmetry_source.get("type", "none")) != "kp_symm_output":
        return out
    source_out = dict(symmetry_source)
    operations = source_out.get("operations")
    if operations is None:
        out["symmetry_source"] = source_out
        return out
    if isinstance(operations, Mapping):
        operations = [
            {**(dict(value) if isinstance(value, Mapping) else {}), "name": str(key)}
            for key, value in operations.items()
        ]
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("symmetry_source.operations must be a list or mapping when provided")
    use = str(source_out.get("use", "raw"))
    matrix_kind_raw = source_out.get("matrix_kind", source_out.get("kind"))
    allow_inferred = allows_inferred_action_metadata(source_out)
    operation_defaults = {
        key: copy.deepcopy(source_out[key])
        for key in (
            "source_matrix_role",
            "source_gauge",
            "target_role",
            "gauge_correction",
            "antiunitary_convention",
        )
        if key in source_out
    }
    if matrix_kind_raw is None:
        normalized_operations: list[Any] = []
        for operation in operations:
            if isinstance(operation, Mapping):
                row = dict(operation)
                user_name = str(row.get("name", row.get("operation", "")))
                if user_name:
                    row["name"] = canonical_source_operation_name_for_valley(user_name, valley_model)
                    row.update(effective_operation_metadata_for_valley(user_name, valley_model))
                normalized_operations.append(row)
            else:
                user_name = str(operation)
                normalized_operations.append(
                    {
                        "name": canonical_source_operation_name_for_valley(user_name, valley_model),
                        **effective_operation_metadata_for_valley(user_name, valley_model),
                    }
                )
        source_out["operations"] = normalized_operations
    else:
        matrix_kind = str(matrix_kind_raw)
        source_out["operations"] = [
            _complete_kp_symm_operation_entry(
                {**operation_defaults, **(dict(operation) if isinstance(operation, Mapping) else {"name": str(operation)})},
                valley_model,
                use=use,
                matrix_kind=matrix_kind,
                allow_inferred=allow_inferred,
            )
            for operation in operations
        ]
    out["symmetry_source"] = source_out
    return out


def _max_derivative_order_values(
    model: Mapping[str, Any],
    *,
    valley_model: Mapping[str, Any] | None = None,
    n_orb: Sequence[int] | None = None,
) -> dict[str, int]:
    legacy = {**DEFAULT_MAX_ORDER, **{str(key): int(value) for key, value in dict(model.get("max_order", {})).items()}}
    values = dict(legacy)
    for semantic_key, legacy_key in GAMMA_LEGACY_ORDER_ALIASES.items():
        values.setdefault(semantic_key, int(legacy.get(legacy_key, 0)))
    explicit = model.get("max_derivative_order", {})
    if explicit is None:
        explicit = {}
    if not isinstance(explicit, Mapping):
        raise ValueError("model.max_derivative_order must be a mapping when provided")
    for key, value in explicit.items():
        values[str(key)] = int(value)
    return values


def _default_auto_low_energy_order_config(valley_model: Mapping[str, Any] | None) -> dict[str, Any]:
    valley_type = str((valley_model or {}).get("valley_type", "")).strip()
    if valley_type == "Gamma":
        return {
            "profile": "gamma_compact_ladder_start",
            "max_order": {"Kinect": 6, "intra": 4, "inter": 6},
            "max_derivative_order": {
                "moire_intra_zero": 0,
                "moire_intra_nonzero": 4,
                "tunneling_zero": 6,
                "tunneling_nonzero": 6,
            },
        }
    if valley_type == "M":
        return {
            "profile": "m_release_baseline",
            "max_order": {"Kinect": 10, "intra": 4, "inter": 6},
            "max_derivative_order": {},
        }
    if valley_type == "K":
        return {
            "profile": "k_release_baseline",
            "max_order": {"Kinect": 6, "intra": 4, "inter": 4},
            "max_derivative_order": {},
        }
    return {
        "profile": "generic_release_baseline",
        "max_order": {"Kinect": 6, "intra": 4, "inter": 4},
        "max_derivative_order": {},
    }


def _reflect_vector(kvec: np.ndarray, axis_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(axis_deg))
    axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    return 2.0 * axis * float(np.dot(axis, kvec)) - np.asarray(kvec, dtype=float)


def _rotate_k_map_to_model_frame(k_map: Any, *, rotation_deg: float) -> Any:
    if not isinstance(k_map, Mapping):
        return k_map
    out = dict(k_map)
    if bool(out.get("in_model_frame", False)):
        return out
    map_type = str(out.get("type", "")).lower()
    if map_type in {"reflection", "mirror", "mirror_axis"} and "axis_deg" in out:
        out["axis_deg"] = float(out["axis_deg"]) + float(rotation_deg)
    return out


def _is_accepted_support_exactification_provenance(provenance: Any) -> bool:
    return (
        isinstance(provenance, Mapping)
        and provenance.get("source") == "support_exactification"
        and bool(provenance.get("accepted_by_user"))
    )


def _symmetry_operation_index(metadata: Mapping[str, Any], *, rotation_deg: float) -> dict[str, dict[str, Any]]:
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for record in operations:
        if not isinstance(record, Mapping):
            continue
        enriched = dict(record)
        resolved_action = record.get("internal_resolved_action")
        if isinstance(resolved_action, Mapping) and not _is_accepted_support_exactification_provenance(
            resolved_action.get("provenance")
        ):
            resolved_action = None
        if not isinstance(resolved_action, Mapping):
            resolved_action = record.get("model_action")
        if isinstance(resolved_action, Mapping):
            for key in ("antiunitary", "k_map", "q_map", "sector_map"):
                if key in resolved_action:
                    enriched[key] = copy.deepcopy(resolved_action[key])
        else:
            enriched["k_map"] = _rotate_k_map_to_model_frame(record.get("k_map"), rotation_deg=rotation_deg)
            if "q_map" in record:
                enriched["q_map"] = _rotate_k_map_to_model_frame(record.get("q_map"), rotation_deg=rotation_deg)
        keys = [record.get("name"), record.get("operation")]
        for key in keys:
            if key is None:
                continue
            index[str(key)] = enriched
    return index


def _enrich_symmetry_map(
    symmetry_map: Mapping[str, Sequence[Mapping[str, Any]]],
    symmetry_metadata: Mapping[str, Any],
    *,
    rotation_deg: float,
    require_action_metadata: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    op_index = _symmetry_operation_index(symmetry_metadata, rotation_deg=rotation_deg)
    out: dict[str, list[dict[str, Any]]] = {}
    for tag, operations in symmetry_map.items():
        rows: list[dict[str, Any]] = []
        for operation in operations:
            if not isinstance(operation, Mapping):
                rows.append({"name": str(operation)})
                continue
            row = dict(operation)
            name = str(row.get("name", ""))
            if name and name in op_index:
                source_row = dict(op_index[name])
                merged = dict(source_row)
                for key, value in row.items():
                    if key not in {"antiunitary", "k_map", "q_map", "sector_map"}:
                        merged[key] = value
                row = merged
            if require_action_metadata:
                _require_resolved_operation_action(row)
            rows.append(row)
        out[str(tag)] = rows
    return out


def _require_resolved_operation_action(operation: Mapping[str, Any]) -> None:
    name = str(operation.get("name", ""))
    missing = [field for field in ("antiunitary", "k_map", "q_map", "sector_map") if field not in operation]
    if missing:
        raise ValueError(f"Operation {name!r} requires explicit resolved action metadata; missing {missing}")


def _default_kpath_config(raw: Mapping[str, Any], *, source_base: Path) -> dict[str, Any]:
    kpath = raw.get("kpath", {})
    if kpath is None:
        kpath = {}
    if not isinstance(kpath, Mapping):
        raise ValueError("kpath section must be a mapping")
    out = dict(kpath)
    if "file" not in out:
        candidate = (source_base / "../../tapw/KPATH.in").resolve()
        if candidate.exists():
            out["file"] = str(candidate)
    return out


def _rotation_deg_from_symmetry_manifest(raw: Mapping[str, Any], *, base: Path) -> float | None:
    symmetry_source = raw.get("symmetry_source", {})
    if not isinstance(symmetry_source, Mapping) or str(symmetry_source.get("type", "")) != "kp_symm_output":
        return None
    path_raw = symmetry_source.get("path")
    if path_raw is None:
        return None
    path = Path(str(path_raw))
    if not path.is_absolute():
        path = (base / path).resolve()
    for filename in ("manifest.json", "summary.json"):
        manifest_path = path / filename
        if not manifest_path.exists():
            continue
        manifest = _load_yaml(manifest_path)
        frame = manifest.get("frame", {})
        if not isinstance(frame, Mapping):
            continue
        q_transform = frame.get("q_transform", {})
        if isinstance(q_transform, Mapping) and "rotation_deg" in q_transform:
            return float(q_transform["rotation_deg"])
        k_transform = frame.get("k_transform", {})
        if isinstance(k_transform, Mapping) and "rotation_deg" in k_transform:
            return float(k_transform["rotation_deg"])
    representations = path / "representations.npz"
    if representations.exists():
        with np.load(representations, allow_pickle=False) as payload:
            if "__metadata_json__" in payload.files:
                metadata = json.loads(str(payload["__metadata_json__"].item()))
                if isinstance(metadata, Mapping):
                    frame = metadata.get("frame", {})
                    if isinstance(frame, Mapping):
                        q_transform = frame.get("q_transform", {})
                        if isinstance(q_transform, Mapping) and "rotation_deg" in q_transform:
                            return float(q_transform["rotation_deg"])
                        k_transform = frame.get("k_transform", {})
                        if isinstance(k_transform, Mapping) and "rotation_deg" in k_transform:
                            return float(k_transform["rotation_deg"])
        return None
    return None


def _packed_symmetry_operation_metadata(name: str, valley_model: Mapping[str, Any]) -> dict[str, Any]:
    family = canonical_operation_name_for_valley(str(name), valley_model)
    if family == "C3z":
        action = {
            "antiunitary": False,
            "k_map": {"type": "rotation", "angle_deg": 120.0},
            "q_map": {"type": "rotation", "angle_deg": 120.0},
            "sector_map": "identity",
        }
    elif family == "C2T":
        action = {
            "antiunitary": True,
            "k_map": {"type": "reflection", "axis_deg": 180.0},
            "q_map": {"type": "reflection", "axis_deg": 180.0},
            "sector_map": "identity",
        }
    elif family == "TR":
        action = {
            "antiunitary": True,
            "k_map": {"type": "negation"},
            "q_map": {"type": "negation"},
            "sector_map": "identity",
        }
    elif family == "C2":
        action = {
            "antiunitary": False,
            "k_map": {"type": "reflection", "axis_deg": 0.0},
            "q_map": {"type": "reflection", "axis_deg": 0.0},
            "sector_map": "layer_exchange",
        }
    else:
        action = {}
    if action:
        action["antiunitary_convention"] = "U_K" if action["antiunitary"] else "none"
        action["matrix_kind"] = "continuum_internal_rep_exact"
        action["spin_map"] = "from_kp_symm_output"
        action["valley_map"] = "identity"
        action.update(copy.deepcopy(SOURCE_MATRIX_SEMANTICS))
    return action


def _load_symmetry_manifest(raw: Mapping[str, Any], *, base: Path) -> dict[str, Any]:
    symmetry_source = raw.get("symmetry_source", {})
    if not isinstance(symmetry_source, Mapping) or str(symmetry_source.get("type", "")) != "kp_symm_output":
        return {}
    path_raw = symmetry_source.get("path")
    if path_raw is None:
        return {}
    path = Path(str(path_raw))
    if not path.is_absolute():
        path = (base / path).resolve()
    for filename in ("manifest.json", "summary.json"):
        manifest_path = path / filename
        if manifest_path.exists():
            manifest = _load_yaml(manifest_path)
            return dict(manifest) if isinstance(manifest, Mapping) else {}
    representations = path / "representations.npz"
    if representations.exists():
        with np.load(representations, allow_pickle=False) as payload:
            if "__metadata_json__" in payload.files:
                metadata = json.loads(str(payload["__metadata_json__"].item()))
                return dict(metadata) if isinstance(metadata, Mapping) else {}
        valley_model = raw.get("valley_model", {})
        if not isinstance(valley_model, Mapping):
            valley_model = {}
        with np.load(representations, allow_pickle=False) as payload:
            return {
                "operations": [
                    {
                        "name": str(name),
                        "matrix_file": "representations.npz",
                        "matrix_array_key": str(name),
                        "matrix_source": "kp_symm_exactified_action",
                        **effective_operation_metadata_for_valley(str(name), valley_model),
                        **_packed_symmetry_operation_metadata(str(name), valley_model),
                    }
                    for name in payload.files
                    if not str(name).startswith("__")
                ],
                "frame": {
                    "q_transform": {"rotation_deg": 0.0},
                    "k_transform": {"rotation_deg": 0.0},
                },
            }
    return {}


def _load_model_artifact_identity(
    heff_file: str | Path,
    symmetry_source: Mapping[str, Any],
    *,
    base: str | Path | None = None,
) -> dict[str, Any]:
    if str(symmetry_source.get("type", "")) != "kp_symm_output":
        return {}
    source_path = symmetry_source.get("path")
    if source_path in (None, ""):
        raise ValueError("kp model requires symmetry_source.path for kp_symm_output")
    symmetry_dir = Path(str(source_path)).expanduser()
    if not symmetry_dir.is_absolute():
        if base is None:
            raise ValueError("Relative symmetry_source.path requires a config base directory")
        symmetry_dir = Path(base) / symmetry_dir
    symmetry_dir = symmetry_dir.resolve()
    representations = symmetry_dir / "representations.npz"
    if not representations.is_file():
        raise FileNotFoundError(
            f"kp model requires canonical kp symm output: {representations}"
        )
    with np.load(representations, allow_pickle=False) as payload:
        if "__metadata_json__" not in payload.files:
            raise ValueError(f"kp model symmetry pack lacks __metadata_json__: {representations}")
        metadata = json.loads(str(payload["__metadata_json__"].item()))
        matrix_keys = {str(name) for name in payload.files if not str(name).startswith("__")}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"kp model symmetry metadata must be a mapping: {representations}")
    symmetry_identity_raw = metadata.get("artifact_identity")
    if not isinstance(symmetry_identity_raw, Mapping):
        raise ValueError(
            f"kp model symmetry pack lacks artifact_identity; rerun kp symm: {representations}"
        )
    symmetry_identity = require_identity_fields(
        symmetry_identity_raw,
        PROJECTION_ARTIFACT_IDENTITY_FIELDS,
        "kp model symmetry pack",
    )
    heff_path = Path(heff_file).resolve()
    project_dir = heff_path.parent
    canonical_heff = (project_dir / "heff.npy").resolve()
    if heff_path != canonical_heff:
        raise ValueError(f"kp model requires canonical projection/heff.npy, got {heff_path}")
    project_identity = load_projection_artifact_identity(project_dir)
    require_matching_identity(
        project_identity,
        symmetry_identity,
        PROJECTION_ARTIFACT_IDENTITY_FIELDS,
        "kp model projection/symmetry",
    )
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError(f"kp model symmetry pack operations must be a list: {representations}")
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError(f"kp model symmetry pack operation metadata must be a mapping: {operation!r}")
        name = str(operation.get("name", operation.get("operation", "")))
        require_matching_identity(
            project_identity,
            operation,
            ("basis_hash", "k_indices_hash", "heff_hash"),
            f"kp model symmetry operation {name!r}",
        )
        if not _operation_matrix_is_exactified(operation):
            raise ValueError(
                f"kp model symmetry operation {name!r} lacks complete exactification provenance"
            )
        matrix_key = str(operation.get("matrix_array_key", name))
        if matrix_key not in matrix_keys:
            raise ValueError(
                f"kp model symmetry operation {name!r} references missing matrix array {matrix_key!r}"
            )
    return project_identity


def _apply_project_k_indices(kpoints: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    points = _validate_kpoints(kpoints)
    if not indices:
        return points[:0]
    max_index = max(int(index) for index in indices)
    if max_index >= len(points):
        raise IndexError(
            f"projection k_indices contains {max_index}, but the source k-point path has {len(points)} rows"
        )
    return np.asarray(points[[int(index) for index in indices]], dtype=float)


def _symmetry_operations_for_default_templates(raw: Mapping[str, Any], *, base: Path) -> list[Mapping[str, Any]]:
    operations: list[Mapping[str, Any]] = []
    symmetry_source = raw.get("symmetry_source", {})
    if isinstance(symmetry_source, Mapping):
        raw_operations = symmetry_source.get("operations", [])
        if isinstance(raw_operations, Mapping):
            raw_operations = [
                {**(dict(value) if isinstance(value, Mapping) else {}), "name": str(key)}
                for key, value in raw_operations.items()
            ]
        if isinstance(raw_operations, Sequence) and not isinstance(raw_operations, (str, bytes)):
            operations.extend(dict(item) for item in raw_operations if isinstance(item, Mapping))
    manifest_operations = _load_symmetry_manifest(raw, base=base).get("operations", [])
    if isinstance(manifest_operations, Sequence) and not isinstance(manifest_operations, (str, bytes)):
        operations.extend(dict(item) for item in manifest_operations if isinstance(item, Mapping))
    return operations


def _rotation_deg_from_config(raw: Mapping[str, Any], *, base: Path) -> float:
    if "coordinate_frame" in raw:
        raise ValueError("coordinate_frame is not a supported model input; use the kp_symm manifest frame")
    legacy_labels: list[str] = []
    if "rotation_deg" in raw:
        legacy_labels.append("rotation_deg")
    model = raw.get("model", {})
    if isinstance(model, Mapping):
        if "rotation_deg" in model:
            legacy_labels.append("model.rotation_deg")
        if "q_rotation_deg" in model:
            legacy_labels.append("model.q_rotation_deg")
    kpath = raw.get("kpath", {})
    if isinstance(kpath, Mapping) and "phase_deg" in kpath:
        legacy_labels.append("kpath.phase_deg")
    if legacy_labels:
        labels = ", ".join(legacy_labels)
        raise ValueError(f"Model frame rotation must come from the kp_symm manifest; remove {labels}")
    manifest_rotation = _rotation_deg_from_symmetry_manifest(raw, base=base)
    if manifest_rotation is not None:
        return manifest_rotation
    symmetry_source = raw.get("symmetry_source", {})
    if isinstance(symmetry_source, Mapping) and str(symmetry_source.get("type", "")) == "toy_generator":
        return 0.0
    raise ValueError(
        "Configured model YAML requires a kp_symm symmetry manifest frame rotation. "
        "Rerun `kp symm` with automatic frame inference; do not hand-write model frame rotations."
    )


def load_model_config(path: str | Path) -> ConfiguredModel:
    cfg_path = Path(path).resolve()
    raw = normalize_case_config(_load_yaml(cfg_path), config_path=cfg_path)
    base = cfg_path.parent
    raw = dict(raw)
    validation = raw.get("validation", {})
    if validation is None:
        validation = {}
    if not isinstance(validation, Mapping):
        raise ValueError("validation section must be a mapping when provided")
    raw["validation"] = dict(validation)

    if raw.get("source_config") not in (None, ""):
        raise ValueError("source_config is not supported in release-only KP configs; use one case YAML.")
    if all(isinstance(raw.get(section), Mapping) for section in ("material", "project", "plot")):
        source_config = cfg_path
        source_raw = dict(raw)
    else:
        raise ValueError("Configured model YAML requires inline material/project/plot sections")
    source_base = source_config.parent
    material = source_raw.get("material", {})
    project = source_raw.get("project", {})
    plot = source_raw.get("plot", {})
    if not isinstance(material, Mapping) or not isinstance(project, Mapping) or not isinstance(plot, Mapping):
        raise ValueError("source_config must contain mapping sections: material, project, plot")
    num_layer_list = _num_layer_list_from_material(material)

    qset1_file = _resolve_path(material.get("qset1_file"), source_base)
    qset2_file = _resolve_path(material.get("qset2_file"), source_base)
    if qset1_file is None or qset2_file is None:
        raise ValueError("source_config material must provide qset1_file and qset2_file")

    raw = _normalize_short_model_config(raw)

    heff_file = _resolve_path(_get_path_value(raw, "heff_file"), base)
    if heff_file is None:
        project_out = _resolve_path(project.get("out_dir"), source_base)
        if project_out is None:
            raise ValueError("heff_file is missing and source_config project.out_dir is unavailable")
        heff_file = project_out / "heff.npy"
    heff_eig_file = _resolve_path(_get_path_value(raw, "heff_eig_file"), base)
    if heff_eig_file is not None:
        raise ValueError("heff_eig_file is not supported in release-only KP configs; use projection/heff.npy.")
    try:
        project_k_indices = load_projection_k_indices(heff_file)
    except FileNotFoundError:
        heff_rows = int(np.load(heff_file, mmap_mode="r", allow_pickle=False).shape[0])
        project_k_indices = list(range(heff_rows))

    kpoints_file = _resolve_path(_get_path_value(raw, "kpoints_file"), base)
    kpoints_from_projection = False
    if kpoints_file is None:
        projection_kpoints = heff_file.parent / "kpoints.npy"
        if projection_kpoints.is_file():
            kpoints_file = projection_kpoints
            kpoints_from_projection = True

    output_section = raw.get("output", {})
    if output_section is None:
        output_section = {}
    if not isinstance(output_section, Mapping):
        raise ValueError("output section must be a mapping")
    output_dir = _resolve_path(output_section.get("dir", "model"), base)
    if output_dir is None:
        raise ValueError("Failed to resolve output directory")

    model_raw = _normalize_model_user_aliases(_model_section(raw))
    valley_model_raw = raw.get("valley_model", {})
    if not isinstance(valley_model_raw, Mapping):
        valley_model_raw = {}
    symmetry_source = _default_symmetry_source(raw, source_raw, source_base=source_base)
    raw["symmetry_source"] = symmetry_source
    kpath_config = _default_kpath_config(raw, source_base=source_base)
    raw["kpath"] = kpath_config
    fit_for_defaults = raw.get("fit", {})
    if not isinstance(fit_for_defaults, Mapping):
        fit_for_defaults = {}
    auto_low_energy_mode = str(fit_for_defaults.get("mode", "")).strip().lower() == "auto_low_energy"
    if auto_low_energy_mode:
        defaults_meta: dict[str, bool] = {}
        if "harmonics" not in model_raw:
            model_raw["harmonics"] = {"intra": {"count": 2}, "inter": {"count": 2}}
            defaults_meta["harmonics"] = True
        if "max_order" not in model_raw and "max_derivative_order" not in model_raw:
            order_defaults = _default_auto_low_energy_order_config(
                raw.get("valley_model", {}) if isinstance(raw.get("valley_model", {}), Mapping) else {}
            )
            model_raw["max_order"] = dict(order_defaults["max_order"])
            if order_defaults.get("max_derivative_order"):
                model_raw["max_derivative_order"] = dict(order_defaults["max_derivative_order"])
            model_raw["auto_low_energy_order_profile"] = {
                "profile": str(order_defaults["profile"]),
                "source": "internal_default",
            }
            defaults_meta["max_order"] = True
        if defaults_meta:
            model_raw["auto_low_energy_defaults"] = defaults_meta
    if "max_order" not in model_raw:
        model_raw["max_order"] = dict(DEFAULT_MAX_ORDER)
    raw["model"] = model_raw
    raw = _normalize_user_symmetry_names(raw)
    raw = _normalize_kp_symm_source(raw)
    model_raw = _normalize_model_user_aliases(_model_section(raw))
    if "symmetry_map" not in model_raw:
        valley_model_normalized = raw.get("valley_model", {})
        symmetry_source_normalized = raw.get("symmetry_source", {})
        if not _operation_rows_from_symmetry_source(
            symmetry_source_normalized if isinstance(symmetry_source_normalized, Mapping) else {},
            valley_model_normalized if isinstance(valley_model_normalized, Mapping) else {},
        ):
            manifest_operations = _symmetry_operations_for_default_templates(raw, base=base)
            if manifest_operations:
                symmetry_source_normalized = {
                    **(dict(symmetry_source_normalized) if isinstance(symmetry_source_normalized, Mapping) else {}),
                    "operations": [dict(item) for item in manifest_operations if isinstance(item, Mapping)],
                }
        model_raw["symmetry_map"] = _default_symmetry_map_from_source_or_valley(
            symmetry_source_normalized if isinstance(symmetry_source_normalized, Mapping) else {},
            valley_model_normalized if isinstance(valley_model_normalized, Mapping) else {},
        )
        raw["model"] = model_raw
        raw = _normalize_user_symmetry_names(raw)
    rotation_deg = _rotation_deg_from_config(raw, base=base)

    model = _model_section(raw)
    model_raw = _normalize_model_user_aliases(model)
    if "n_orb" in model:
        n_orb_raw = model["n_orb"]
    elif "n_orb1" in model or "n_orb2" in model:
        n_orb_raw = [model.get("n_orb1", 2), model.get("n_orb2", 2)]
    else:
        n_orb_raw = _infer_n_orb_pair_from_arrays(qset1_file, qset2_file, heff_file)
    n_orb_input = _as_int_list(n_orb_raw, name="model.n_orb")
    n_orb_values, n_orb_resolution = _resolve_layerwise_counts(
        n_orb_input,
        name="model.n_orb",
        num_layer_list=num_layer_list,
    )
    nlow_state_default_from_n_orb = "nlow_state" not in model
    nlow_state_input = _as_int_list(model.get("nlow_state", n_orb_input), name="model.nlow_state")
    nlow_state, nlow_state_resolution = _resolve_layerwise_counts(
        nlow_state_input,
        name="model.nlow_state",
        num_layer_list=num_layer_list,
        default_from="model.n_orb" if nlow_state_default_from_n_orb else None,
    )
    if len(n_orb_values) != 2:
        raise ValueError(f"model.n_orb must resolve to two qset counts, got {n_orb_values}")
    if len(nlow_state) != 2:
        raise ValueError(f"model.nlow_state must resolve to two qset counts, got {nlow_state}")
    model_raw["n_orb"] = list(n_orb_values)
    model_raw["nlow_state"] = list(nlow_state)
    model_raw["n_orb_resolution"] = n_orb_resolution
    model_raw["nlow_state_resolution"] = nlow_state_resolution
    if "groups" in n_orb_resolution:
        model_raw["n_orb_layerwise"] = n_orb_resolution["raw"]
        model_raw["n_orb_resolved_qset"] = n_orb_resolution["resolved_qset"]
    if "groups" in nlow_state_resolution:
        model_raw["nlow_state_layerwise"] = nlow_state_resolution["raw"]
        model_raw["nlow_state_resolved_qset"] = nlow_state_resolution["resolved_qset"]
    raw["model"] = model_raw
    model = _model_section(raw)
    orbital_count_metadata = {
        "num_layer_list": [int(value) for value in num_layer_list],
        "total_layers": int(sum(int(value) for value in num_layer_list)),
        "n_orb": n_orb_resolution,
        "nlow_state": nlow_state_resolution,
    }
    max_order_values = _max_derivative_order_values(
        model,
        valley_model=raw.get("valley_model", {}) if isinstance(raw.get("valley_model", {}), Mapping) else {},
        n_orb=n_orb_values,
    )

    fit = raw.get("fit", {})
    if not isinstance(fit, Mapping):
        raise ValueError("fit section must be a mapping")
    target_bands = str(model.get("target_bands", raw.get("target_bands", "top"))).strip().lower()
    if target_bands not in {"top", "bottom"}:
        raise ValueError("model.target_bands must be 'top' or 'bottom'")
    model_raw["target_bands"] = target_bands
    raw["model"] = model_raw

    auto_low_energy_refine_indices: list[int] | None = None
    if fit.get("indices") is not None:
        fit_indices = _as_int_list(fit.get("indices", []), name="fit.indices")
        fit_selection_metadata = {
            "mode": "manual",
            "source": "fit.indices",
            "selected_indices": [int(index) for index in fit_indices],
        }
    elif fit.get("mode") is not None and str(fit.get("mode")).strip().lower() in {"auto", "auto_compact"}:
        fit_kpoints_all = _load_kpoints_from_inputs(
            kpoints_file=kpoints_file,
            kpath_config=kpath_config,
            base=base,
            rotation_deg=rotation_deg,
            kpoints_from_projection=kpoints_from_projection,
        )
        if not kpoints_from_projection:
            fit_kpoints_all = _apply_project_k_indices(fit_kpoints_all, project_k_indices)
        fit_indices, fit_selection_metadata = _select_auto_fit_indices(fit_kpoints_all, fit)
    elif fit.get("mode") is not None and str(fit.get("mode")).strip().lower() == "auto_low_energy":
        fit_kpoints_all = _load_kpoints_from_inputs(
            kpoints_file=kpoints_file,
            kpath_config=kpath_config,
            base=base,
            rotation_deg=rotation_deg,
            kpoints_from_projection=kpoints_from_projection,
        )
        if not kpoints_from_projection:
            fit_kpoints_all = _apply_project_k_indices(fit_kpoints_all, project_k_indices)
        max_points = int(fit.get("max_points", 2))
        initial_points = int(fit.get("initial_points", 2))
        initial_fit_indices = (
            _as_int_list(fit.get("initial_fit_indices"), name="fit.initial_fit_indices")
            if fit.get("initial_fit_indices") is not None
            else None
        )
        fit_indices, fit_selection_metadata = _select_adaptive_fit_indices(
            fit_kpoints_all,
            initial_points=initial_points,
            max_points=max_points,
            initial_indices=initial_fit_indices,
        )
        explicit_refine_indices = fit.get("refine_indices", fit.get("refinement_indices"))
        if explicit_refine_indices is not None:
            auto_low_energy_refine_indices = _as_int_list(
                explicit_refine_indices,
                name="fit.refine_indices",
            )
        else:
            auto_low_energy_refine_indices = _auto_low_energy_refinement_indices(
                fit_kpoints_all,
                base_indices=fit_indices,
                max_points=int(fit.get("max_refine_points", fit.get("refine_max_points", 8))),
            )
        fit_selection_metadata["refine_indices"] = [int(index) for index in auto_low_energy_refine_indices]
    else:
        fit_indices = []
        fit_selection_metadata = {
            "mode": "manual",
            "source": "missing_fit_indices",
            "selected_indices": [],
        }

    bands = raw.get("bands", {})
    if bands is None:
        bands = {}
    if not isinstance(bands, Mapping):
        raise ValueError("bands section must be a mapping")
    band_indices_raw = bands.get("indices")
    band_indices = None if band_indices_raw is None else _as_int_list(band_indices_raw, name="bands.indices")
    band_slice = None if bands.get("band_slice") is None else _as_int_list(bands.get("band_slice"), name="bands.band_slice")
    band_plot = bands.get("plot", {})
    if band_plot is None:
        band_plot = {}
    if not isinstance(band_plot, Mapping):
        raise ValueError("bands.plot section must be a mapping when provided")

    harmonic_selection_report: dict[str, Any] | None = None
    harmonic_selection_enabled, harmonic_selection_cfg = _harmonic_selection_options(fit)
    harmonics_defaulted = bool(model.get("auto_low_energy_defaults", {}).get("harmonics", False))
    if (
        str(fit.get("mode", "")).strip().lower() == "auto_low_energy"
        and harmonics_defaulted
        and harmonic_selection_enabled
    ):
        thresholds_raw = harmonic_selection_cfg.get("thresholds", {})
        if thresholds_raw is None:
            thresholds_raw = {}
        if not isinstance(thresholds_raw, Mapping):
            raise ValueError("fit.harmonic_selection.thresholds must be a mapping when provided")
        threshold_values = _harmonic_selection_threshold_values(thresholds_raw)
        Q_set1_for_selection, Q_set2_for_selection = load_Q_sets_from_gvec_files(
            qset1_file,
            qset2_file,
            rotation_deg=rotation_deg,
        )
        heff_for_selection = _select_rows(np.load(heff_file, mmap_mode="r"), band_indices)
        sample_indices = _evenly_spaced_sample_indices(
            int(heff_for_selection.shape[0]),
            int(harmonic_selection_cfg.get("sample_kpoints", 7)),
        )
        if sample_indices is None:
            heff_scan = np.asarray(heff_for_selection)
        else:
            heff_scan = np.asarray(heff_for_selection)[np.asarray(sample_indices, dtype=int)]
        n_primary_for_selection = int(sum(n_orb_values))
        dim_for_selection = int(heff_for_selection.shape[-1])
        plot_bands_for_selection = int(
            harmonic_selection_cfg.get(
                "plot_bands",
                fit.get("weighted_fit_bands", min(dim_for_selection, max(n_primary_for_selection, 10))),
            )
        )
        max_shell_for_selection = int(harmonic_selection_cfg.get("max_shell", 5))
        candidate_pairs_for_selection = _harmonic_candidate_pairs_from_config(
            harmonic_selection_cfg.get("candidate_pairs"),
            max_shell=max_shell_for_selection,
            search=str(harmonic_selection_cfg.get("search", "ladder")),
        )
        harmonic_selection_report = _run_harmonic_ablation_selection(
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
            tol=float(harmonic_selection_cfg.get("tol", 1.0e-6)),
        )
        harmonic_selection_report["sample_indices"] = sample_indices
        selected_harmonics = harmonic_selection_report["selected"]
        if bool(harmonic_selection_cfg.get("require_accepted", False)) and not bool(selected_harmonics.get("accepted", False)):
            raise ValueError(
                "auto_low_energy harmonic selection did not find an accepted support; "
                f"best status={harmonic_selection_report['selection_status']}"
            )
        model_raw["harmonics"] = {
            "intra": {"count": int(selected_harmonics["intra_shells"])},
            "inter": {"count": int(selected_harmonics["inter_shells"])},
        }
        model_raw["auto_low_energy_harmonic_selection"] = harmonic_selection_report
        raw["model"] = model_raw
        model = _model_section(raw)

    harmonics = model.get("harmonics", {})
    if not isinstance(harmonics, Mapping):
        raise ValueError("model.harmonics must be a mapping")
    harmonic_count_limits = _harmonic_count_limits(harmonics)
    symmetry_map = model.get("symmetry_map", {})
    if not isinstance(symmetry_map, Mapping):
        raise ValueError("model.symmetry_map must be a mapping")

    validate_model_config(raw, nlow_state=nlow_state)

    valley_model = raw.get("valley_model", {})
    symmetry_source = raw.get("symmetry_source", {})
    sectors = raw.get("sectors", [])
    if sectors is None:
        sectors = []
    if not isinstance(sectors, Sequence) or isinstance(sectors, (str, bytes)):
        raise ValueError("sectors must be a list when provided")
    if not sectors:
        sectors = _infer_k_sectors_from_layerwise_counts(
            n_orb_resolution,
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
        )
    _validate_sector_orbital_counts(sectors, n_orb_values)
    term_templates = model.get("term_templates", raw.get("term_templates"))
    template_symmetry_operations = _symmetry_operations_for_default_templates(raw, base=base)
    term_template_metadata: dict[str, Any] = {"input_kind": "explicit"}
    if term_templates is None:
        term_templates = _default_term_templates_for_model(
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
            n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
            max_order=max_order_values,
            harmonic_counts=harmonic_count_limits,
            symmetry_operations=template_symmetry_operations,
            sectors=sectors,
        )
        term_template_metadata = _default_term_template_profile_metadata(
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
            n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
            symmetry_operations=template_symmetry_operations,
            sectors=sectors,
        )
    if not isinstance(term_templates, Sequence) or isinstance(term_templates, (str, bytes)):
        raise ValueError("term_templates must be a list when provided")
    if not term_templates:
        term_templates = _default_term_templates_for_model(
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
            n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
            max_order=max_order_values,
            harmonic_counts=harmonic_count_limits,
            symmetry_operations=template_symmetry_operations,
            sectors=sectors,
        )
        term_template_metadata = _default_term_template_profile_metadata(
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
            n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
            symmetry_operations=template_symmetry_operations,
            sectors=sectors,
        )

    symmetry_source_metadata = {}
    if isinstance(symmetry_source, Mapping) and bool(symmetry_source.get("inferred_from_source_config", False)):
        symmetry_source_metadata["inferred_from_source_config"] = True

    fit_mode = str(fit.get("mode", "")).strip().lower()
    coeff_prune_default = 1.0e-4 if fit_mode == "auto_low_energy" else 0.0
    coeff_prune_threshold = float(fit.get("coeff_prune_threshold", coeff_prune_default))
    if coeff_prune_threshold < 0.0:
        raise ValueError(f"fit.coeff_prune_threshold must be non-negative, got {coeff_prune_threshold}")
    null_channel_abs_tol = float(fit.get("null_channel_abs_tol", 0.0))
    if null_channel_abs_tol < 0.0:
        raise ValueError(f"fit.null_channel_abs_tol must be non-negative, got {null_channel_abs_tol}")
    null_channel_rel_tol = float(fit.get("null_channel_rel_tol", 0.0))
    if null_channel_rel_tol < 0.0:
        raise ValueError(f"fit.null_channel_rel_tol must be non-negative, got {null_channel_rel_tol}")
    output_profile = str(output_section.get("profile", "release")).strip().lower()
    if output_profile in {"", "default"}:
        output_profile = "release"
    if output_profile not in {"release", "debug"}:
        raise ValueError(f"output.profile must be 'release' or 'debug', got {output_profile!r}")
    output_section = dict(output_section)
    output_section["profile"] = output_profile

    refine_raw = fit.get("refine_bands", {})
    if refine_raw is True:
        refine_config = {"enabled": True}
    elif refine_raw in (False, None):
        refine_config = {"enabled": False}
    elif isinstance(refine_raw, Mapping):
        refine_config = dict(refine_raw)
    else:
        raise ValueError("fit.refine_bands must be a mapping or boolean when provided")
    if str(fit.get("mode", "")).strip().lower() == "auto_low_energy":
        heff_eig_for_windows = (
            np.load(heff_eig_file)
            if heff_eig_file is not None and heff_eig_file.exists()
            else np.linalg.eigvalsh(np.load(heff_file, mmap_mode="r"))
        )
        heff_eig_for_windows = _select_rows(heff_eig_for_windows, band_indices)
        n_primary = int(sum(n_orb_values))
        gap_tolerance_mev = float(fit.get("gap_tolerance_mev", 0.1))
        windows = _auto_low_energy_windows(
            heff_eig_for_windows,
            n_primary=n_primary,
            target_bands=target_bands,
            gap_tolerance_mev=gap_tolerance_mev,
            max_expanded=int(fit.get("max_expanded_windows", 2)),
        )
        dim_for_windows = int(np.asarray(heff_eig_for_windows).shape[1])
        weighted_fit_bands = int(fit.get("weighted_fit_bands", fit.get("fit_bands", min(dim_for_windows, max(n_primary, 10)))))
        if weighted_fit_bands < n_primary:
            raise ValueError(
                "fit.weighted_fit_bands must be at least the primary low-energy dimension "
                f"sum(model.n_orb)={n_primary}, got {weighted_fit_bands}"
            )
        if weighted_fit_bands > dim_for_windows:
            raise ValueError(
                f"fit.weighted_fit_bands={weighted_fit_bands} exceeds Hamiltonian dimension {dim_for_windows}"
            )
        weighted_fit_window = _auto_low_energy_window_record(
            heff_eig_for_windows,
            n_bands=weighted_fit_bands,
            target_bands=target_bands,
            role="weighted_fit",
            gap_tolerance_mev=gap_tolerance_mev,
            use_for_loss=True,
        )
        windows["weighted_fit"] = weighted_fit_window
        primary_band_slice = list(windows["primary"]["band_slice"])
        weighted_band_slice = list(weighted_fit_window["band_slice"])
        auto_refine_defaults = {
            "enabled": True,
            "mode": "auto_low_energy",
            "target_bands": target_bands,
            "auto_windows": windows,
            "auto_harmonic_selection": _json_safe(harmonic_selection_report)
            if harmonic_selection_report is not None
            else {"enabled": False},
            "band_slice": list(weighted_fit_window["band_slice"]),
            "align": target_bands,
            "solver": str(fit.get("refinement_solver", "linear_low_subspace")),
            "indices": [int(index) for index in (auto_low_energy_refine_indices or fit_indices)],
            "variable_tags": ["Kinect", "Onsite", "intra"],
            "components": ["real"],
            "refinement_candidates": [
                {
                    "name": "A_intra_real",
                    "variable_tags": ["Kinect", "Onsite", "intra"],
                    "components": ["real"],
                    "role": "default_compact",
                },
                {
                    "name": "B_add_inter_real",
                    "variable_tags": ["Kinect", "Onsite", "intra", "inter"],
                    "components": ["real"],
                    "role": "accept_only_if_pareto_better",
                },
                {
                    "name": "C_add_imag",
                    "variable_tags": ["Kinect", "Onsite", "intra", "inter"],
                    "components": ["real", "imag"],
                    "role": "accept_only_if_pareto_better",
                },
            ],
            "regularization": float(fit.get("linear_regularization", fit.get("regularization", 0.3))),
            "band_sigma_mev": float(fit.get("band_sigma_mev", 1.0)),
            "normalize_band_loss": True,
            "weighted_band_loss": {
                "enabled": True,
                "primary_bands": int(n_primary),
                "fit_bands": int(weighted_fit_bands),
                "decay": float(fit.get("band_weight_decay", 0.75)),
                "floor": float(fit.get("band_weight_floor", 0.45)),
                "normalize_mean": True,
            },
            "coefficient_weight": float(fit.get("coefficient_weight", 0.02)),
            "max_nfev": int(fit.get("max_nfev", 8)),
            "use_fit_kpoints": bool(fit.get("use_fit_kpoints", False)),
            "max_variables": int(fit.get("max_variables", 900)),
            "subspace_loss": {
                "enabled": bool(fit.get("subspace_loss_enabled", False)),
                "mode": "principal_angles",
                "weight": float(fit.get("subspace_weight", 2.0)),
                "band_slice": weighted_band_slice,
                "normalize": True,
                "gap_tolerance_mev": gap_tolerance_mev,
            },
            "shell_projected_matrix_loss": {
                "enabled": True,
                "weight": float(fit.get("shell_projected_matrix_weight", fit.get("shell_subspace_weight", 1.0))),
                "max_shells": int(fit.get("shell_projected_matrix_max_shells", fit.get("shell_subspace_max_shells", 3))),
                "window": {
                    "mode": "fixed_fraction",
                    "center_fraction": float(
                        fit.get("shell_projected_matrix_fraction", fit.get("shell_subspace_fraction", 0.5))
                    ),
                    "gap_tolerance_mev": float(fit.get("shell_projected_matrix_gap_tolerance_mev", gap_tolerance_mev)),
                },
                "shell_decay": float(fit.get("shell_projected_matrix_shell_decay", fit.get("shell_subspace_decay", 0.75))),
                "legacy_alias": "shell_subspace_loss",
            },
            "low_subspace_matrix_loss": {
                "enabled": bool(fit.get("low_subspace_matrix_loss_enabled", False)),
                "weight": float(fit.get("low_subspace_matrix_weight", 0.25)),
                "sigma_mev": float(fit.get("low_subspace_matrix_sigma_mev", 10.0)),
                "band_slice": weighted_band_slice,
                "normalize": True,
                "gap_tolerance_mev": gap_tolerance_mev,
            },
            "matrix_loss": {
                "enabled": True,
                "mode": "block_normalized",
                "weight": float(fit.get("raw_matrix_weight", 0.0)),
                "sigma_mev": float(fit.get("raw_matrix_sigma_mev", 10.0)),
                "blocks": {"kinetic_diagonal": 1.0, "intralayer": 1.0, "interlayer": 0.5},
            },
            "acceptance_guard": {
                "enabled": True,
                "profile": "low_energy",
                "selection": "best_validation_window",
                "line_search_alphas": list(fit.get("acceptance_line_search_alphas", [0.5])),
                "guard_all_bands": bool(fit.get("acceptance_guard_all_bands", False)),
                "max_rms_increase_mev": float(fit.get("acceptance_max_rms_increase_mev", 999.0)),
                "max_max_increase_mev": float(fit.get("acceptance_max_max_increase_mev", 999.0)),
                "max_all_band_rms_increase_mev": float(fit.get("acceptance_max_all_band_rms_increase_mev", 3.0)),
                "max_all_band_max_increase_mev": float(fit.get("acceptance_max_all_band_max_increase_mev", 5.0)),
            },
        }
        if refine_config.get("enabled", False):
            merged = dict(auto_refine_defaults)
            merged.update(refine_config)
            refine_config = merged
        else:
            refine_config = auto_refine_defaults

    return ConfiguredModel(
        path=cfg_path,
        raw=dict(raw),
        source_config=source_config,
        source_raw=dict(source_raw),
        qset1_file=qset1_file,
        qset2_file=qset2_file,
        kpoints_file=kpoints_file,
        kpoints_from_projection=kpoints_from_projection,
        heff_file=heff_file,
        heff_eig_file=heff_eig_file,
        project_k_indices=project_k_indices,
        output_dir=output_dir,
        rotation_deg=rotation_deg,
        fit_indices=fit_indices,
        fit_selection_metadata=fit_selection_metadata,
        band_indices=band_indices,
        n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
        nlow_state=nlow_state,
        bM_config=dict(model.get("bM", {})),
        harmonics_config=dict(harmonics),
        max_order=max_order_values,
        symmetry_map={str(key): list(value) for key, value in dict(symmetry_map).items()},
        coeff_tol=float(fit.get("coeff_tol", 1.0e-6)),
        null_channel_abs_tol=null_channel_abs_tol,
        null_channel_rel_tol=null_channel_rel_tol,
        coeff_prune_threshold=coeff_prune_threshold,
        compare_to_heff=bool(bands.get("compare_to_heff", True)),
        kpath_config=dict(kpath_config),
        valley_model=dict(valley_model),
        symmetry_source_config=dict(symmetry_source) if isinstance(symmetry_source, Mapping) else {},
        symmetry_source_metadata=symmetry_source_metadata,
        sectors_config=[dict(item) for item in sectors],
        term_templates=[dict(item) for item in term_templates],
        term_template_metadata=term_template_metadata,
        output_config=dict(output_section),
        band_slice=band_slice,
        band_plot_config=dict(band_plot),
        orbital_count_metadata=orbital_count_metadata,
        validation_config=dict(raw.get("validation", {})),
        band_refinement_config=refine_config,
    )


def evaluate_vector_expression(value: Any, variables: Mapping[str, Any]) -> np.ndarray:
    """Evaluate a small vector expression such as ``bM1 + bM2`` or ``rot(q1, 240)``."""
    if isinstance(value, np.ndarray):
        return _as_vector(value)
    if isinstance(value, (list, tuple)):
        return _as_vector(np.array(value, dtype=float))
    if isinstance(value, Mapping):
        if "expr" not in value:
            raise ValueError(f"Vector mapping must contain expr, got {value!r}")
        value = value["expr"]
    if not isinstance(value, str):
        raise ValueError(f"Unsupported vector expression {value!r}")
    expr = ast.parse(value, mode="eval")
    result = _eval_ast(expr.body, variables)
    return _as_vector(result)


def _as_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"Expected a 2-vector, got shape {arr.shape}: {value!r}")
    return arr


def _as_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.shape != ():
        raise ValueError(f"Expected scalar, got shape {arr.shape}: {value!r}")
    return float(arr)


def _eval_ast(node: ast.AST, variables: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown vector expression symbol: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant in vector expression: {node.value!r}")
    if isinstance(node, ast.List):
        return np.array([_as_scalar(_eval_ast(elt, variables)) for elt in node.elts], dtype=float)
    if isinstance(node, ast.Tuple):
        return np.array([_as_scalar(_eval_ast(elt, variables)) for elt in node.elts], dtype=float)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand, variables)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return np.asarray(left) + np.asarray(right)
        if isinstance(node.op, ast.Sub):
            return np.asarray(left) - np.asarray(right)
        if isinstance(node.op, ast.Mult):
            return np.asarray(left) * np.asarray(right)
        if isinstance(node.op, ast.Div):
            return np.asarray(left) / np.asarray(right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "rot" and len(node.args) == 2:
            vec = evaluate_vector_expression_ast(node.args[0], variables)
            angle = _as_scalar(_eval_ast(node.args[1], variables))
            return rot(vec, angle)
        if node.func.id == "norm" and len(node.args) == 1:
            vec = evaluate_vector_expression_ast(node.args[0], variables)
            return float(np.linalg.norm(vec))
    raise ValueError(f"Unsupported vector expression syntax: {ast.dump(node)}")


def evaluate_vector_expression_ast(node: ast.AST, variables: Mapping[str, Any]) -> np.ndarray:
    return _as_vector(_eval_ast(node, variables))


def _harmonic_filter(kind: str, indices: Sequence[int], *, sign: float = 1.0) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind, "indices": [int(index) for index in indices]}
    if sign != 1.0:
        out["sign"] = float(sign)
    return out


def _harmonic_count_limits(harmonics: Mapping[str, Any]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for kind in ("intra", "inter"):
        raw = harmonics.get(kind)
        if _is_auto_harmonic_spec(raw):
            limits[kind] = _auto_harmonic_count(raw, name=f"model.harmonics.{kind}")
        elif isinstance(raw, Mapping):
            limits[kind] = max((int(key) for key in raw.keys()), default=0)
    return limits


@dataclass(frozen=True)
class _TermTemplateProfile:
    valley_type: str
    templates: tuple[Mapping[str, Any], ...]
    n_orb: tuple[int, int] | None = None


def _term_template_row(
    name: str,
    source: str,
    sector_pairs: Sequence[Sequence[int]],
    orbital_pairs: Any,
    *,
    max_order: int | None = None,
    max_order_from: str | None = None,
    harmonics: Any | None = None,
    harmonic_filter: Mapping[str, Any] | None = None,
    monomial_constraints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if max_order is not None and max_order_from is not None:
        raise ValueError(f"{name}: use either max_order or max_order_from, not both")
    if harmonics is not None and harmonic_filter is not None:
        raise ValueError(f"{name}: use either harmonics or harmonic_filter, not both")

    def _sector_ref(value: Any) -> int | str:
        if isinstance(value, (int, np.integer)):
            return int(value)
        return str(value)

    row: dict[str, Any] = {
        "name": name,
        "source": source,
        "sector_pairs": [[_sector_ref(i), _sector_ref(j)] for i, j in sector_pairs],
        "orbital_pairs": copy.deepcopy(orbital_pairs),
    }
    if max_order is not None:
        row["max_order"] = int(max_order)
    if max_order_from is not None:
        row["max_order_from"] = str(max_order_from)
    if harmonics is not None:
        row["harmonics"] = copy.deepcopy(harmonics)
    if harmonic_filter is not None:
        row["harmonic_filter"] = copy.deepcopy(dict(harmonic_filter))
    if monomial_constraints is not None:
        row["monomial_constraints"] = copy.deepcopy(dict(monomial_constraints))
    return row


_K_C3_MONOMIAL_CONSTRAINTS: dict[str, Any] = {
    "exclude_m_sum_zero": True,
    "difference_mod": 3,
    "difference_residue": 0,
    "require_mz_ge_mz_star": True,
}


_TERM_TEMPLATE_PROFILES: tuple[_TermTemplateProfile, ...] = (
    _TermTemplateProfile(
        valley_type="K",
        templates=(
            _term_template_row(
                "kinetic_layer1",
                "diagonal_kp",
                [[1, 1]],
                "diagonal",
                max_order_from="Kinect",
                monomial_constraints={
                    "exclude_m_sum_zero": True,
                    "difference_mod": 3,
                    "difference_residue": 0,
                    "require_mz_ge_mz_star": True,
                },
            ),
            _term_template_row("onsite_layer1", "onsite", [[1, 1]], "diagonal", max_order=0),
            _term_template_row(
                "intra_layer1",
                "moire_potential",
                [[1, 1]],
                "all",
                harmonic_filter=_harmonic_filter("intra", [2, 3, 4]),
                max_order_from="intra",
            ),
            _term_template_row(
                "inter_21_positive",
                "tunneling",
                [[2, 1]],
                "all",
                harmonic_filter=_harmonic_filter("inter", [1, 2, 3, 4]),
                max_order_from="inter",
            ),
            _term_template_row(
                "inter_21_negative",
                "tunneling",
                [[2, 1]],
                "all",
                harmonic_filter=_harmonic_filter("inter", [2, 3, 4], sign=-1.0),
                max_order_from="inter",
            ),
        ),
    ),
    _TermTemplateProfile(
        valley_type="Gamma",
        n_orb=(1, 1),
        templates=(
            _term_template_row("gamma_1x1_kinetic", "diagonal_kp", [[1, 1], [2, 2]], [[1, 1]], max_order_from="Kinect"),
            _term_template_row("gamma_1x1_onsite", "onsite", [[1, 1], [2, 2]], [[1, 1]], max_order=0),
            _term_template_row(
                "gamma_1x1_intra_nonzero",
                "moire_potential",
                [[1, 1], [2, 2]],
                [[1, 1]],
                harmonic_filter={"kind": "intra", "indices": "nonzero", "start": 2, "fallback_count": 4},
                max_order_from="moire_intra_nonzero",
            ),
            _term_template_row(
                "gamma_1x1_inter_zero",
                "tunneling",
                [[2, 1], [1, 2]],
                [[1, 1]],
                harmonic_filter=_harmonic_filter("inter", [1]),
                max_order_from="tunneling_zero",
            ),
            _term_template_row(
                "gamma_1x1_inter_nonzero",
                "tunneling",
                [[2, 1], [1, 2]],
                [[1, 1]],
                harmonic_filter={"kind": "inter", "indices": "nonzero", "start": 2, "fallback_count": 4},
                max_order_from="tunneling_nonzero",
            ),
        ),
    ),
    _TermTemplateProfile(
        valley_type="Gamma",
        n_orb=(2, 2),
        templates=(
            _term_template_row("gamma_kinetic", "diagonal_kp", [[1, 1]], [[1, 1]], max_order_from="Kinect"),
            _term_template_row("gamma_onsite", "onsite", [[1, 1]], [[1, 1]], max_order=0),
            _term_template_row(
                "gamma_intra_zero",
                "moire_potential",
                [[1, 1]],
                [[2, 1]],
                harmonic_filter=_harmonic_filter("intra", [1]),
                max_order_from="moire_intra_zero",
            ),
            _term_template_row(
                "gamma_intra_zero_kdependent",
                "moire_potential",
                [[1, 1]],
                [[1, 2], [2, 1]],
                harmonic_filter=_harmonic_filter("intra", [1]),
                max_order_from="moire_intra_nonzero",
                monomial_constraints={"exclude_m_sum_zero": True},
            ),
            _term_template_row(
                "gamma_intra_nonzero",
                "moire_potential",
                [[1, 1]],
                [[1, 1], [1, 2], [2, 1]],
                harmonic_filter={"kind": "intra", "indices": "nonzero", "start": 2, "fallback_count": 4},
                max_order_from="moire_intra_nonzero",
            ),
            _term_template_row(
                "gamma_inter_zero",
                "tunneling",
                [[2, 1]],
                [[1, 1], [2, 1]],
                harmonic_filter=_harmonic_filter("inter", [1]),
                max_order_from="tunneling_zero",
            ),
            _term_template_row(
                "gamma_inter_nonzero",
                "tunneling",
                [[2, 1]],
                [[1, 1], [2, 1]],
                harmonic_filter={"kind": "inter", "indices": "nonzero", "start": 2, "fallback_count": 4},
                max_order_from="tunneling_nonzero",
            ),
            _term_template_row(
                "gamma_inter_nonzero_negative",
                "tunneling",
                [[2, 1]],
                [[1, 1]],
                harmonic_filter={"kind": "inter", "indices": "nonzero", "start": 2, "fallback_count": 4, "sign": -1.0},
                max_order_from="tunneling_nonzero",
            ),
        ),
    ),
    _TermTemplateProfile(
        valley_type="Gamma",
        templates=(
            _term_template_row(
                "gamma_generic_kinetic",
                "diagonal_kp",
                [[1, 1], [2, 2]],
                "diagonal",
                max_order_from="Kinect",
            ),
            _term_template_row(
                "gamma_generic_onsite",
                "onsite",
                [[1, 1], [2, 2]],
                "diagonal",
                max_order=0,
            ),
            _term_template_row(
                "gamma_generic_intra",
                "moire_potential",
                [[1, 1], [2, 2]],
                "all",
                harmonics="intra",
                max_order_from="intra",
            ),
            _term_template_row(
                "gamma_generic_inter",
                "tunneling",
                [[2, 1], [1, 2]],
                "all",
                harmonics="inter",
                max_order_from="inter",
            ),
        ),
    ),
    _TermTemplateProfile(
        valley_type="M",
        templates=(
            _term_template_row("m1_kinetic_bottom", "diagonal_kp", [[1, 1]], "diagonal", max_order_from="Kinect"),
            _term_template_row("m1_onsite_bottom", "onsite", [[1, 1]], "diagonal", max_order=0),
            _term_template_row(
                "m1_intra_bottom",
                "moire_potential",
                [[1, 1]],
                "all",
                harmonics="intra",
                max_order_from="intra",
            ),
            _term_template_row(
                "m1_inter_top_to_bottom",
                "tunneling",
                [[2, 1]],
                "all",
                harmonics="inter",
                max_order_from="inter",
            ),
        ),
    ),
)


def _clamp_harmonic_filter(filter_spec: Mapping[str, Any], harmonic_counts: Mapping[str, int] | None) -> dict[str, Any]:
    out = copy.deepcopy(dict(filter_spec))
    kind = str(out["kind"])
    if out.get("indices") == "nonzero":
        start = int(out.pop("start", 2))
        fallback_count = int(out.pop("fallback_count", 4))
        count = int(harmonic_counts.get(kind, fallback_count)) if harmonic_counts and kind in harmonic_counts else fallback_count
        out["indices"] = list(range(start, count + 1))
    elif harmonic_counts and kind in harmonic_counts:
        count = int(harmonic_counts[kind])
        out["indices"] = [int(index) for index in out.get("indices", []) if int(index) <= count]
    return out


def _expand_term_template_profile(
    template: Mapping[str, Any],
    max_order: Mapping[str, int],
    *,
    harmonic_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    row = copy.deepcopy(dict(template))
    max_order_key = row.pop("max_order_from", None)
    if max_order_key is not None:
        row["max_order"] = int(max_order.get(str(max_order_key), 0))
    harmonic_filter = row.pop("harmonic_filter", None)
    if harmonic_filter is not None:
        if not isinstance(harmonic_filter, Mapping):
            raise ValueError(f"harmonic_filter profile must be a mapping, got {harmonic_filter!r}")
        harmonic_filter = _clamp_harmonic_filter(harmonic_filter, harmonic_counts)
        row["harmonics"] = _harmonic_filter(
            str(harmonic_filter["kind"]),
            harmonic_filter.get("indices", []),
            sign=float(harmonic_filter.get("sign", 1.0)),
        )
    return row


def _term_template_profile_for(valley_type: str, n_orb: tuple[int, int]) -> tuple[_TermTemplateProfile, ...]:
    exact_matches: list[_TermTemplateProfile] = []
    generic_matches: list[_TermTemplateProfile] = []
    for profile in _TERM_TEMPLATE_PROFILES:
        if profile.valley_type != str(valley_type):
            continue
        if profile.n_orb == tuple(n_orb):
            exact_matches.append(profile)
        elif profile.n_orb is None:
            generic_matches.append(profile)
    return tuple(exact_matches or generic_matches)


def _term_name_fragment(value: Any) -> str:
    text = str(value).strip()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    out = "_".join(part for part in out.split("_") if part)
    return out or "sector"


def _sector_qset_slot(sector: Mapping[str, Any]) -> int:
    qset = str(sector.get("qset", ""))
    if qset == "qset1":
        return 1
    if qset == "qset2":
        return 2
    raise ValueError(f"Unsupported sector qset {qset!r}; expected qset1 or qset2")


def _active_template_sectors(
    sectors: Sequence[Mapping[str, Any]] | None,
    n_orb: tuple[int, int],
) -> list[dict[str, Any]]:
    if not sectors:
        return []
    active: list[dict[str, Any]] = []
    seen_qsets: set[str] = set()
    for raw in sectors:
        if not isinstance(raw, Mapping):
            continue
        sector = dict(raw)
        qset = str(sector.get("qset", ""))
        slot = _sector_qset_slot(sector)
        if int(n_orb[slot - 1]) <= 0:
            continue
        if qset in seen_qsets:
            raise ValueError(
                "Default sector-aware term generation currently expects at most one active sector per qset; "
                "use explicit term_templates for multiple sectors sharing a qset."
            )
        seen_qsets.add(qset)
        sector.setdefault("name", f"L{slot}")
        active.append(sector)
    return active


def _default_k_sector_term_templates(
    *,
    sectors: Sequence[Mapping[str, Any]],
    n_orb: tuple[int, int],
    max_order: Mapping[str, int],
    harmonic_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    active = _active_template_sectors(sectors, n_orb)
    if not active:
        return []
    intra_count = int((harmonic_counts or {}).get("intra", 3))
    inter_count = int((harmonic_counts or {}).get("inter", 0))
    templates: list[dict[str, Any]] = []
    for sector in active:
        name = str(sector["name"])
        label = _term_name_fragment(name)
        pair = [[name, name]]
        templates.extend(
            [
                _term_template_row(
                    f"kinetic_{label}",
                    "diagonal_kp",
                    pair,
                    "diagonal",
                    max_order=int(max_order.get("Kinect", 0)),
                    monomial_constraints=_K_C3_MONOMIAL_CONSTRAINTS,
                ),
                _term_template_row(f"onsite_{label}", "onsite", pair, "diagonal", max_order=0),
            ]
        )
        if intra_count >= 2:
            templates.append(
                _term_template_row(
                    f"intra_{label}_first_shell",
                    "moire_potential",
                    pair,
                    "all",
                    harmonics=_harmonic_filter("intra", [2], sign=-1.0),
                    max_order=int(max_order.get("intra", 0)),
                )
            )
        if intra_count >= 3:
            templates.append(
                _term_template_row(
                    f"intra_{label}_second_shell",
                    "moire_potential",
                    pair,
                    "all",
                    harmonics=_harmonic_filter("intra", range(3, intra_count + 1)),
                    max_order=int(max_order.get("intra", 0)),
                )
            )

    from_sectors = [sector for sector in active if str(sector.get("qset")) == "qset2"]
    to_sectors = [sector for sector in active if str(sector.get("qset")) == "qset1"]
    if inter_count > 0:
        for from_sector in from_sectors:
            for to_sector in to_sectors:
                from_name = str(from_sector["name"])
                to_name = str(to_sector["name"])
                templates.append(
                    _term_template_row(
                        f"inter_{_term_name_fragment(from_name)}_to_{_term_name_fragment(to_name)}",
                        "tunneling",
                        [[from_name, to_name]],
                        "all",
                        harmonics=_harmonic_filter("inter", range(1, inter_count + 1)),
                        max_order=int(max_order.get("inter", 0)),
                    )
                )
    return templates


def _template_inter_harmonic_vectors(
    template: Mapping[str, Any],
    inter_harmonics_map: Mapping[int, np.ndarray],
) -> list[np.ndarray]:
    raw = template.get("harmonics", "inter")
    sign = float(template.get("harmonic_sign", 1.0))
    indices = None
    if isinstance(raw, Mapping):
        sign = float(raw.get("sign", sign))
        indices = raw.get("indices")
        raw = raw.get("kind", "inter")
    if str(raw) != "inter":
        return []
    items = sorted(inter_harmonics_map.items()) if indices is None else [
        (int(index), inter_harmonics_map[int(index)])
        for index in indices
        if int(index) in inter_harmonics_map
    ]
    return [sign * np.asarray(vector, dtype=float) for _index, vector in items]


def _raw_q_support_count_for_sector_pair(
    *,
    sector_from: Mapping[str, Any],
    sector_to: Mapping[str, Any],
    harmonic_vectors: Sequence[np.ndarray],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    tol: float = 1.0e-5,
) -> int:
    q_from = sector_qset(sector_from, Q_set1, Q_set2)
    q_to = sector_qset(sector_to, Q_set1, Q_set2)
    count = 0
    for p_vector in harmonic_vectors:
        diff = q_from[:, None, :] - np.asarray(p_vector, dtype=float) - q_to[None, :, :]
        count += int(np.count_nonzero(np.linalg.norm(diff, axis=2) < tol))
    return count


def _orient_k_sector_inter_templates_by_raw_support(
    templates: Sequence[Mapping[str, Any]],
    *,
    valley_model: Mapping[str, Any],
    n_orb: tuple[int, int],
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    inter_harmonics_map: Mapping[int, np.ndarray],
    term_template_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = [dict(row) for row in templates]
    if str(valley_model.get("valley_type", "")) != "K":
        return out, []
    if str(term_template_metadata.get("input_kind", "")) != "default":
        return out, []
    if "k_sector_aware" not in list(term_template_metadata.get("profiles", [])):
        return out, []
    active = _active_template_sectors(sectors, n_orb)
    candidate_pairs = [
        (sector_from, sector_to)
        for sector_from in active
        for sector_to in active
        if str(sector_from.get("qset")) != str(sector_to.get("qset"))
    ]
    if not candidate_pairs:
        return out, []

    by_name = {str(sector.get("name")): sector for sector in active}
    diagnostics: list[dict[str, Any]] = []
    for row in out:
        if str(row.get("source", "")) != "tunneling":
            continue
        harmonic_vectors = _template_inter_harmonic_vectors(row, inter_harmonics_map)
        if not harmonic_vectors:
            continue
        scores = [
            (
                _raw_q_support_count_for_sector_pair(
                    sector_from=sector_from,
                    sector_to=sector_to,
                    harmonic_vectors=harmonic_vectors,
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                ),
                str(sector_from["name"]),
                str(sector_to["name"]),
            )
            for sector_from, sector_to in candidate_pairs
        ]
        current_pairs = row.get("sector_pairs", [])
        if not isinstance(current_pairs, Sequence) or isinstance(current_pairs, (str, bytes)):
            current_pairs = []
        current_support = 0
        for pair in current_pairs:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                continue
            sector_from = by_name.get(str(pair[0]))
            sector_to = by_name.get(str(pair[1]))
            if sector_from is None or sector_to is None:
                continue
            current_support += _raw_q_support_count_for_sector_pair(
                sector_from=sector_from,
                sector_to=sector_to,
                harmonic_vectors=harmonic_vectors,
                Q_set1=Q_set1,
                Q_set2=Q_set2,
            )
        best_support = max((score[0] for score in scores), default=0)
        best_pairs = [[from_name, to_name] for support, from_name, to_name in scores if support == best_support and support > 0]
        changed = bool(best_pairs) and best_support > current_support
        if changed:
            row["sector_pairs"] = best_pairs
            if len(best_pairs) == 1:
                row["name"] = (
                    f"inter_{_term_name_fragment(best_pairs[0][0])}"
                    f"_to_{_term_name_fragment(best_pairs[0][1])}"
                )
        diagnostics.append(
            {
                "template": str(row.get("name", "inter")),
                "current_support": int(current_support),
                "best_support": int(best_support),
                "scores": [
                    {"sector_pair": [from_name, to_name], "support": int(support)}
                    for support, from_name, to_name in scores
                ],
                "selected_sector_pairs": copy.deepcopy(row.get("sector_pairs", [])),
                "changed": changed,
            }
        )
    return out, diagnostics


def _default_term_template_profile_metadata(
    *,
    valley_model: Mapping[str, Any],
    n_orb: tuple[int, int],
    symmetry_operations: Sequence[Mapping[str, Any]] | None,
    sectors: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    valley_type = str(valley_model.get("valley_type", ""))
    sector_profile = valley_type == "K" and bool(_active_template_sectors(sectors, n_orb))
    profiles = () if sector_profile else _term_template_profile_for(valley_type, n_orb)
    profile_names: list[str] = []
    if sector_profile:
        profile_names.append("k_sector_aware")
    else:
        for profile in profiles:
            if profile.valley_type == "Gamma" and profile.n_orb == (2, 2):
                if _gamma_2x2_sector_diagonal_pairs(symmetry_operations) == [[1, 1]]:
                    profile_names.append("gamma_2x2_symmetry_aware")
                else:
                    profile_names.append("gamma_2x2_independent_sectors")
            elif profile.n_orb is None:
                profile_names.append(f"{profile.valley_type.lower()}_default")
            else:
                profile_names.append(
                    f"{profile.valley_type.lower()}_{'x'.join(str(value) for value in profile.n_orb)}"
                )
    metadata: dict[str, Any] = {
        "input_kind": "default",
        "valley_type": valley_type,
        "n_orb": [int(n_orb[0]), int(n_orb[1])],
        "profiles": profile_names,
    }
    if valley_type == "Gamma" and n_orb == (2, 2):
        metadata["gamma_2x2_sector_diagonal_pairs"] = _gamma_2x2_sector_diagonal_pairs(symmetry_operations)
    return metadata


_GAMMA_2X2_SECTOR_DIAGONAL_TEMPLATE_NAMES = {
    "gamma_kinetic",
    "gamma_onsite",
    "gamma_intra_zero",
    "gamma_intra_zero_kdependent",
    "gamma_intra_nonzero",
}
_GAMMA_1X1_SECTOR_DIAGONAL_TEMPLATE_NAMES = {
    "gamma_1x1_kinetic",
    "gamma_1x1_onsite",
    "gamma_1x1_intra_nonzero",
}
_GAMMA_SECTOR_DIAGONAL_TEMPLATE_NAMES = (
    _GAMMA_2X2_SECTOR_DIAGONAL_TEMPLATE_NAMES | _GAMMA_1X1_SECTOR_DIAGONAL_TEMPLATE_NAMES
)


def _sector_map_exchanges_model_sectors(sector_map: Any) -> bool:
    if isinstance(sector_map, str):
        return sector_map.lower() in {"layer_exchange", "sector_exchange", "exchange", "swap"}
    if isinstance(sector_map, Mapping):
        normalized = {str(key): str(value) for key, value in sector_map.items()}
        if len(normalized) == 2 and all(
            key != value and normalized.get(value) == key for key, value in normalized.items()
        ):
            return True
        return (
            normalized.get("1") == "2"
            and normalized.get("2") == "1"
        ) or (
            normalized.get("L1") == "L2"
            and normalized.get("L2") == "L1"
        )
    return False


def _operation_exchanges_model_sectors(operation: Mapping[str, Any]) -> bool:
    candidates: list[Any] = [operation]
    for key in ("model_action", "internal_resolved_action", "declared_model_action"):
        value = operation.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    return any(
        isinstance(candidate, Mapping) and _sector_map_exchanges_model_sectors(candidate.get("sector_map"))
        for candidate in candidates
    )


def _gamma_2x2_sector_diagonal_pairs(symmetry_operations: Sequence[Mapping[str, Any]] | None) -> list[list[int]]:
    if symmetry_operations and any(_operation_exchanges_model_sectors(operation) for operation in symmetry_operations):
        return [[1, 1]]
    return [[1, 1], [2, 2]]


def _apply_gamma_2x2_sector_policy(
    templates: list[dict[str, Any]],
    *,
    valley_type: str,
    n_orb: tuple[int, int],
    symmetry_operations: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if str(valley_type) != "Gamma" or tuple(n_orb) != (2, 2):
        return templates
    sector_pairs = _gamma_2x2_sector_diagonal_pairs(symmetry_operations)
    for row in templates:
        if row.get("name") in _GAMMA_SECTOR_DIAGONAL_TEMPLATE_NAMES:
            row["sector_pairs"] = copy.deepcopy(sector_pairs)
    return templates


def _default_term_templates_for_model(
    *,
    valley_model: Mapping[str, Any],
    n_orb: tuple[int, int],
    max_order: Mapping[str, int],
    harmonic_counts: Mapping[str, int] | None = None,
    symmetry_operations: Sequence[Mapping[str, Any]] | None = None,
    sectors: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    valley_type = str(valley_model.get("valley_type", ""))
    if valley_type == "K" and sectors:
        templates = _default_k_sector_term_templates(
            sectors=sectors,
            n_orb=n_orb,
            max_order=max_order,
            harmonic_counts=harmonic_counts,
        )
        if templates:
            return templates
    profiles = _term_template_profile_for(valley_type, n_orb)
    if profiles:
        templates = [
            _expand_term_template_profile(template, max_order, harmonic_counts=harmonic_counts)
            for profile in profiles
            for template in profile.templates
        ]
        return _apply_gamma_2x2_sector_policy(
            templates,
            valley_type=valley_type,
            n_orb=n_orb,
            symmetry_operations=symmetry_operations,
        )
    return []


def _build_variables(Q_set1: np.ndarray, bM1: np.ndarray, bM2: np.ndarray, model: Mapping[str, Any]) -> dict[str, Any]:
    variables: dict[str, Any] = {
        "zero": np.zeros(2, dtype=float),
        "bM1": np.asarray(bM1, dtype=float),
        "bM2": np.asarray(bM2, dtype=float),
        "sqrt3": float(np.sqrt(3.0)),
        "pi": float(np.pi),
        "bM_norm": float(np.linalg.norm(bM1)),
    }
    variables["q1"] = variables["bM_norm"] * np.array([0.0, 1.0 / np.sqrt(3.0)], dtype=float)
    variables["q3"] = rot(variables["q1"], 240.0)

    custom = model.get("vectors", {})
    if custom is None:
        return variables
    if not isinstance(custom, Mapping):
        raise ValueError("model.vectors must be a mapping")
    for key, expr in custom.items():
        variables[str(key)] = evaluate_vector_expression(expr, variables)
    return variables


def _harmonics_map(raw: Mapping[str, Any], variables: Mapping[str, Any]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key, value in raw.items():
        out[int(key)] = evaluate_vector_expression(value, variables)
    return out


def _is_auto_harmonic_spec(raw: Any) -> bool:
    if isinstance(raw, int):
        return True
    if isinstance(raw, Mapping):
        return bool(raw.get("auto", False)) or "count" in raw
    return False


def _auto_harmonic_count(raw: Any, *, name: str) -> int:
    if isinstance(raw, int):
        count = raw
    elif isinstance(raw, Mapping):
        if "count" not in raw:
            raise ValueError(f"{name} auto harmonics requires count")
        count = int(raw["count"])
    else:
        raise ValueError(f"{name} auto harmonics must be an integer or mapping, got {raw!r}")
    if count < 0:
        raise ValueError(f"{name} harmonic count must be non-negative, got {count}")
    return count


def _c3_orbit_vectors(vector: np.ndarray) -> list[np.ndarray]:
    base = [np.asarray(vector, dtype=float), rot(vector, 120.0), rot(vector, 240.0)]
    return base + [-v for v in base]


def _vector_key(vector: np.ndarray, tol: float) -> tuple[int, int]:
    return tuple(np.rint(np.asarray(vector, dtype=float) / tol).astype(int).tolist())


def _unique_vector_records(vectors: list[np.ndarray], *, tol: float, bM1: np.ndarray, bM2: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for vector in vectors:
        arr = np.asarray(vector, dtype=float)
        key = _vector_key(arr, tol)
        entry = grouped.setdefault(key, {"sum": np.zeros(2, dtype=float), "pair_count": 0})
        entry["sum"] += arr
        entry["pair_count"] += 1

    basis = np.column_stack([bM1, bM2])
    records: list[dict[str, Any]] = []
    for entry in grouped.values():
        vector = entry["sum"] / float(entry["pair_count"])
        lattice_index = None
        try:
            coeff = np.linalg.solve(basis, vector)
            rounded = np.rint(coeff).astype(int)
            if np.linalg.norm(vector - basis @ rounded) <= 10.0 * tol:
                lattice_index = [int(rounded[0]), int(rounded[1])]
        except np.linalg.LinAlgError:
            lattice_index = None
        records.append(
            {
                "vector": vector,
                "norm": float(np.linalg.norm(vector)),
                "pair_count": int(entry["pair_count"]),
                "lattice_index": lattice_index,
            }
        )
    return records


def _default_sectors(config: ConfiguredModel) -> list[dict[str, Any]]:
    if config.sectors_config:
        return [dict(item) for item in config.sectors_config]
    return [
        {"name": "L1", "qset": "qset1", "n_orb": config.n_orb[0]},
        {"name": "L2", "qset": "qset2", "n_orb": config.n_orb[1]},
    ]


def _validate_sector_orbital_counts(
    sectors: Sequence[Mapping[str, Any]],
    n_orb_values: Sequence[int],
) -> None:
    expected_by_qset = {"qset1": int(n_orb_values[0]), "qset2": int(n_orb_values[1])}
    for sector in sectors:
        qset_name = str(sector.get("qset", "qset1"))
        if qset_name not in expected_by_qset or "n_orb" not in sector:
            continue
        actual = int(sector["n_orb"])
        expected = expected_by_qset[qset_name]
        if actual != expected:
            name = str(sector.get("name", qset_name))
            raise ValueError(
                f"sectors.{name}.n_orb={actual} is inconsistent with model.n_orb for {qset_name} "
                f"({expected})"
            )


def _sector_offset(sector: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(sector.get("q_offset", [0.0, 0.0]), dtype=float)


_HERMITIAN_HARMONIC_ACTION: dict[str, Any] = {"name": "HermitianPair", "type": "negation"}


def _orbit_member_indices(
    records: list[dict[str, Any]],
    seed: np.ndarray,
    *,
    tol: float,
    actions: Sequence[Mapping[str, Any]] = (_HERMITIAN_HARMONIC_ACTION,),
) -> list[int]:
    targets = _harmonic_orbit_vectors(seed, actions, tol=20.0 * tol)
    indices: list[int] = []
    for idx, record in enumerate(records):
        vector = record["vector"]
        if any(np.linalg.norm(vector - target) <= 20.0 * tol for target in targets):
            indices.append(idx)
    return indices


def _representative_score(vector: np.ndarray) -> tuple[float, float, float]:
    theta = float(np.mod(np.arctan2(vector[1], vector[0]), 2.0 * np.pi))
    sector = np.pi / 3.0
    reduced = float(np.mod(theta, sector))
    wedge_distance = min(reduced, sector - reduced)
    circular_angle = min(theta, 2.0 * np.pi - theta)
    if wedge_distance < 1.0e-8:
        wedge_distance = 0.0
    if circular_angle < 1.0e-8:
        circular_angle = 0.0
    return (
        round(wedge_distance, 12),
        round(circular_angle, 12),
        round(theta, 12),
        round(float(np.linalg.norm(vector)), 12),
    )


def _select_star_representative(records: list[dict[str, Any]], indices: list[int]) -> int:
    return min(indices, key=lambda idx: _representative_score(records[idx]["vector"]))


def _candidate_vectors(kind: str, Q_set1: np.ndarray, Q_set2: np.ndarray) -> list[np.ndarray]:
    if kind == "intra":
        return [Q[i] - Q[j] for Q in (Q_set1, Q_set2) for i in range(len(Q)) for j in range(len(Q))]
    if kind == "inter":
        return [Q_set2[j] - Q_set1[i] for i in range(len(Q_set1)) for j in range(len(Q_set2))]
    raise ValueError(f"Unsupported harmonic kind: {kind}")


def _support_count_for_sector_pair(
    *,
    sector_from: Mapping[str, Any],
    sector_to: Mapping[str, Any],
    p_vector: np.ndarray,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    tol: float,
) -> int:
    q_from = sector_qset(sector_from, Q_set1, Q_set2)
    q_to = sector_qset(sector_to, Q_set1, Q_set2)
    off_from = _sector_offset(sector_from)
    off_to = _sector_offset(sector_to)
    count = 0
    shifted_to = q_to + off_to
    for q in q_from + off_from:
        delta = np.linalg.norm(q - p_vector - shifted_to, axis=1)
        count += int(np.count_nonzero(delta <= tol))
    return count


def _harmonic_action_key(action: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(action.get("type", "")),
        round(float(action.get("angle_deg", 0.0)), 12),
        round(float(action.get("axis_deg", 0.0)), 12),
        json.dumps(_json_safe(action.get("sector_map")), sort_keys=True),
    )


def _named_operation_harmonic_action(name: Any) -> dict[str, Any] | None:
    operation_name = str(name)
    if operation_name in {"C3z", "C3"}:
        return {"name": "C3z", "type": "rotation", "angle_deg": 120.0}
    if operation_name in {"TR", "C2", "C2z"}:
        return {"name": operation_name, "type": "negation"}
    return None


def _operation_to_harmonic_action(operation: Mapping[str, Any]) -> dict[str, Any] | None:
    q_map = operation.get("q_map", operation.get("k_map"))
    if not isinstance(q_map, Mapping):
        for key in ("name", "operation"):
            if key in operation:
                action = _named_operation_harmonic_action(operation[key])
                if action is not None:
                    return action
        return None
    map_type = str(q_map.get("type", "")).lower()
    if map_type not in {"identity", "negation", "rotation", "reflection", "mirror"}:
        return None
    action = {"name": str(operation.get("name", map_type)), "type": map_type}
    for key in ("angle_deg", "axis_deg"):
        if key in q_map:
            action[key] = float(q_map[key])
    if "sector_map" in operation:
        action["sector_map"] = copy.deepcopy(operation["sector_map"])
    return action


def _harmonic_actions_from_operations(operations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [dict(_HERMITIAN_HARMONIC_ACTION)]
    seen = {_harmonic_action_key(actions[0])}
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        action = _operation_to_harmonic_action(operation)
        if action is None:
            continue
        key = _harmonic_action_key(action)
        if key not in seen:
            actions.append(action)
            seen.add(key)
    return actions


def _sector_pair_specs(raw: Any, kind: str, sectors: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    by_name = {str(sector["name"]): sector for sector in sectors}
    raw_pairs = raw.get("sector_pairs") if isinstance(raw, Mapping) else None
    if raw_pairs is not None:
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for pair in raw_pairs:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError(f"{kind} sector_pairs entries must be [from, to], got {pair!r}")
            pairs.append((by_name[str(pair[0])], by_name[str(pair[1])]))
        return pairs
    if kind == "intra":
        return [(sector, sector) for sector in sectors]
    if kind == "inter":
        if len(sectors) < 2:
            raise ValueError("inter auto harmonics require at least two sectors")
        return [(sectors[1], sectors[0])]
    raise ValueError(f"Unsupported harmonic kind: {kind}")


def _support_harmonic_records(
    *,
    raw: Any,
    kind: str,
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    tol: float,
) -> list[dict[str, Any]]:
    basis = np.column_stack([bM1, bM2])
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for sector_from, sector_to in _sector_pair_specs(raw, kind, sectors):
        q_from = sector_qset(sector_from, Q_set1, Q_set2) + _sector_offset(sector_from)
        q_to = sector_qset(sector_to, Q_set1, Q_set2) + _sector_offset(sector_to)
        pair_label = [str(sector_from["name"]), str(sector_to["name"])]
        for qf in q_from:
            for qt in q_to:
                vector = np.asarray(qf - qt, dtype=float)
                key = _vector_key(vector, tol)
                entry = grouped.setdefault(
                    key,
                    {"sum": np.zeros(2, dtype=float), "pair_count": 0, "support_count": 0, "sector_pairs": []},
                )
                entry["sum"] += vector
                entry["pair_count"] += 1
                entry["support_count"] += _support_count_for_sector_pair(
                    sector_from=sector_from,
                    sector_to=sector_to,
                    p_vector=vector,
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                    tol=tol,
                )
                if pair_label not in entry["sector_pairs"]:
                    entry["sector_pairs"].append(pair_label)

    records: list[dict[str, Any]] = []
    for entry in grouped.values():
        vector = entry["sum"] / float(entry["pair_count"])
        lattice_index = None
        shell = None
        try:
            coeff = np.linalg.solve(basis, vector)
            rounded = np.rint(coeff).astype(int)
            if np.linalg.norm(vector - basis @ rounded) <= 10.0 * tol:
                lattice_index = [int(rounded[0]), int(rounded[1])]
                shell = int(max(abs(rounded[0]), abs(rounded[1]), abs(rounded[0] + rounded[1])))
        except np.linalg.LinAlgError:
            lattice_index = None
            shell = None
        records.append(
            {
                "vector": vector,
                "norm": float(np.linalg.norm(vector)),
                "pair_count": int(entry["pair_count"]),
                "support_count": int(entry["support_count"]),
                "lattice_index": lattice_index,
                "shell": shell,
                "sector_pairs": entry["sector_pairs"],
            }
        )
    records.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))
    return records


def _harmonic_orbit_record_indices(
    records: Sequence[Mapping[str, Any]],
    seed: np.ndarray,
    *,
    actions: Sequence[Mapping[str, Any]],
    tol: float,
) -> list[int]:
    targets = _harmonic_orbit_vectors(seed, actions, tol=20.0 * tol)
    return [
        idx
        for idx, record in enumerate(records)
        if any(np.linalg.norm(np.asarray(record["vector"], dtype=float) - target) <= 20.0 * tol for target in targets)
    ]


def _auto_harmonics_from_support(
    *,
    raw: Any,
    kind: str,
    count: int,
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    symmetry_operations: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if count == 0:
        return {}, {"kind": kind, "selected": [], "count": 0}
    b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    tol = max(1.0e-8, b_norm * 1.0e-6)
    records = _support_harmonic_records(
        raw=raw,
        kind=kind,
        sectors=sectors,
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        bM1=bM1,
        bM2=bM2,
        tol=tol,
    )
    actions = _harmonic_actions_from_operations(symmetry_operations)
    assigned: set[int] = set()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if idx in assigned:
            continue
        member_indices = _harmonic_orbit_record_indices(records, record["vector"], actions=actions, tol=tol)
        assigned.update(member_indices)
        rep_idx = _select_star_representative(records, member_indices)
        rep = dict(records[rep_idx])
        rep["member_count"] = int(len(member_indices))
        rep["orbit_vectors"] = [np.asarray(records[i]["vector"], dtype=float) for i in member_indices]
        rep["pair_count"] = int(sum(records[i]["pair_count"] for i in member_indices))
        rep["support_count"] = int(sum(records[i]["support_count"] for i in member_indices))
        rep["orbit_generators"] = [str(action["name"]) for action in actions]
        candidates.append(rep)
        for member_idx in member_indices:
            if member_idx != rep_idx:
                rejected.append(
                    {
                        "vector": np.asarray(records[member_idx]["vector"], dtype=float).tolist(),
                        "reason": "duplicate_symmetry_orbit",
                        "representative": np.asarray(rep["vector"], dtype=float).tolist(),
                        "orbit_generators": list(rep["orbit_generators"]),
                    }
                )
    candidates.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))
    if count > len(candidates):
        raise ValueError(f"Requested {count} {kind} harmonics but only found {len(candidates)} support orbits")

    selected = candidates[:count]
    harmonic_map = {idx + 1: np.asarray(item["vector"], dtype=float) for idx, item in enumerate(selected)}
    selected_rows = []
    for idx, item in enumerate(selected, start=1):
        selected_rows.append(
            {
                "index": idx,
                "vector": np.asarray(item["vector"], dtype=float).tolist(),
                "norm": float(item["norm"]),
                "pair_count": int(item["pair_count"]),
                "member_count": int(item["member_count"]),
                "orbit_size": int(item["member_count"]),
                "orbit_generators": list(item["orbit_generators"]),
                "orbit_vectors": [np.asarray(vector, dtype=float).tolist() for vector in item["orbit_vectors"]],
                "lattice_index": item["lattice_index"],
                "shell": item["shell"],
                "sector_pair": item["sector_pairs"][0] if len(item["sector_pairs"]) == 1 else None,
                "sector_pairs": item["sector_pairs"],
                "support_count": int(item["support_count"]),
            }
        )
    return harmonic_map, {
        "kind": kind,
        "count": count,
        "generation": "qset_support_symmetry_orbit",
        "selection_rule": "qset_support_plus_operation_action_orbits",
        "orbit_generators": [str(action["name"]) for action in actions],
        "tolerance": tol,
        "selected": selected_rows,
        "rejected": rejected,
    }


def _auto_harmonics_from_q_sets(
    *,
    kind: str,
    count: int,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    tol: float | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if count == 0:
        return {}, {"kind": kind, "selected": [], "count": 0}
    b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    tol = float(tol if tol is not None else max(1.0e-8, b_norm * 1.0e-6))
    records = _unique_vector_records(_candidate_vectors(kind, Q_set1, Q_set2), tol=tol, bM1=bM1, bM2=bM2)
    records.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))

    assigned: set[int] = set()
    stars: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if idx in assigned:
            continue
        member_indices = _orbit_member_indices(records, record["vector"], tol=tol)
        assigned.update(member_indices)
        rep_idx = _select_star_representative(records, member_indices)
        rep = records[rep_idx]
        stars.append(
            {
                "vector": np.asarray(rep["vector"], dtype=float),
                "norm": float(rep["norm"]),
                "pair_count": int(sum(records[i]["pair_count"] for i in member_indices)),
                "member_count": int(len(member_indices)),
                "lattice_index": rep["lattice_index"],
                "orbit_generators": [str(_HERMITIAN_HARMONIC_ACTION["name"])],
            }
        )

    stars.sort(key=lambda item: (round(float(item["norm"]) / tol), _representative_score(item["vector"])))
    selected = stars[:count]
    if len(selected) < count:
        raise ValueError(f"Requested {count} {kind} harmonics but only found {len(selected)} Q-difference stars")

    harmonic_map = {idx + 1: np.asarray(item["vector"], dtype=float) for idx, item in enumerate(selected)}
    selected_rows = []
    for idx, item in enumerate(selected, start=1):
        selected_rows.append(
            {
                "index": idx,
                "vector": np.asarray(item["vector"], dtype=float).tolist(),
                "norm": float(item["norm"]),
                "pair_count": int(item["pair_count"]),
                "member_count": int(item["member_count"]),
                "orbit_size": int(item["member_count"]),
                "orbit_generators": list(item["orbit_generators"]),
                "lattice_index": item["lattice_index"],
            }
        )
    return harmonic_map, {
        "kind": kind,
        "count": count,
        "generation": "symmetry_orbit",
        "orbit_generators": [str(_HERMITIAN_HARMONIC_ACTION["name"])],
        "tolerance": tol,
        "selected": selected_rows,
    }


def _apply_harmonic_action(vector: np.ndarray, action: Mapping[str, Any]) -> np.ndarray:
    action_type = str(action.get("type", "")).lower()
    if action_type == "identity":
        return np.asarray(vector, dtype=float)
    if action_type == "negation":
        return -np.asarray(vector, dtype=float)
    if action_type == "rotation":
        return rot(np.asarray(vector, dtype=float), float(action.get("angle_deg", 0.0)))
    if action_type in {"reflection", "mirror"}:
        return _reflect_vector(np.asarray(vector, dtype=float), float(action.get("axis_deg", 0.0)))
    raise ValueError(f"Unsupported harmonic orbit action: {action!r}")


def _harmonic_orbit_vectors(seed: np.ndarray, actions: Sequence[Mapping[str, Any]], *, tol: float) -> list[np.ndarray]:
    orbit: list[np.ndarray] = []
    pending = [np.asarray(seed, dtype=float)]
    while pending:
        vector = pending.pop()
        if any(np.linalg.norm(vector - existing) <= tol for existing in orbit):
            continue
        orbit.append(vector)
        for action in actions:
            pending.append(_apply_harmonic_action(vector, action))
    orbit.sort(key=_representative_score)
    return orbit


_AUTO_VALLEY_HARMONIC_PROFILES: dict[tuple[str, str], dict[str, Any]] = {
    ("K", "inter"): {
        "selection_rule": "K_inter_geometry",
        "orbit_generators": [{"name": "C3z", "type": "rotation", "angle_deg": 120.0}],
        "seeds": ["q1", "-2.0 * q1", "q1 + bM2", "2.0 * bM2 + q3"],
    },
    ("M", "inter"): {
        "selection_rule": "M_inter_geometry",
        "variables": {"m_q1": "0.5 * norm(bM1) * [1.0, 0.0]"},
        "orbit_generators": [{"name": "C2", "type": "reflection", "axis_deg": 0.0}],
        "seeds": ["m_q1", "bM2 - m_q1", "bM2 + m_q1", "bM1 + m_q1", "bM2 - 3.0 * m_q1"],
    },
    ("K", "intra"): {
        "selection_rule": "K_intra_geometry",
        "orbit_generators": [{"name": "C3z", "type": "rotation", "angle_deg": 120.0}],
        "seeds": ["[0.0, 0.0]", "-bM1", "bM1 + bM2", "2.0 * bM1"],
    },
    ("M", "intra"): {
        "selection_rule": "M_intra_geometry",
        "orbit_generators": [{"name": "C2", "type": "reflection", "axis_deg": 0.0}],
        "seeds": ["bM1", "bM2", "-bM1 + bM2", "bM1 + bM2", "-2.0 * bM1 + bM2", "2.0 * bM1"],
    },
}


def _variables_for_valley_harmonic_profile(profile: Mapping[str, Any], variables: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(variables)
    extra_variables = profile.get("variables", {})
    if extra_variables is None:
        return out
    if not isinstance(extra_variables, Mapping):
        raise ValueError(f"valley harmonic profile variables must be a mapping, got {extra_variables!r}")
    for key, expr in extra_variables.items():
        out[str(key)] = evaluate_vector_expression(expr, out)
    return out


def _candidate_orbits_from_valley_profile(
    profile: Mapping[str, Any],
    variables: Mapping[str, Any],
    *,
    tol: float,
) -> list[dict[str, Any]]:
    seed_exprs = profile.get("seeds", [])
    if not isinstance(seed_exprs, Sequence) or isinstance(seed_exprs, (str, bytes)):
        raise ValueError(f"valley harmonic profile seeds must be a list, got {seed_exprs!r}")
    raw_actions = profile.get("orbit_generators", [])
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise ValueError(f"valley harmonic profile orbit_generators must be a list, got {raw_actions!r}")
    actions = [dict(action) for action in raw_actions if isinstance(action, Mapping)]
    if len(actions) != len(raw_actions):
        raise ValueError(f"valley harmonic profile orbit_generators must contain mappings, got {raw_actions!r}")
    scoped_variables = _variables_for_valley_harmonic_profile(profile, variables)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for expr in seed_exprs:
        seed = evaluate_vector_expression(expr, scoped_variables)
        orbit = _harmonic_orbit_vectors(seed, actions, tol=tol)
        orbit_key = tuple(sorted(_vector_key(np.asarray(vector, dtype=float), tol) for vector in orbit))
        duplicate = orbit_key in seen
        if not duplicate:
            seen.add(orbit_key)
        rows.append(
            {
                "seed_expr": expr,
                "representative": np.asarray(seed, dtype=float),
                "orbit": orbit,
                "duplicate": duplicate,
                "orbit_generators": [str(action.get("name", action.get("type", ""))) for action in actions],
            }
        )
    return rows


def _auto_valley_harmonics(
    *,
    kind: str,
    count: int,
    valley_model: Mapping[str, Any] | None,
    bM1: np.ndarray,
    bM2: np.ndarray,
    variables: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any]] | None:
    valley_type = str((valley_model or {}).get("valley_type", ""))
    profile = _AUTO_VALLEY_HARMONIC_PROFILES.get((valley_type, kind))
    if profile is None:
        return None
    profile_variables = {**dict(variables), "bM1": np.asarray(bM1, dtype=float), "bM2": np.asarray(bM2, dtype=float)}
    b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    tol = max(1.0e-8, b_norm * 1.0e-8)
    candidate_rows = _candidate_orbits_from_valley_profile(profile, profile_variables, tol=tol)
    candidates = [row for row in candidate_rows if not bool(row["duplicate"])]
    if count > len(candidates):
        raise ValueError(f"{valley_type} {kind} auto harmonics found only {len(candidates)} unique geometry orbits, got {count}")
    selected = candidates[:count]
    harmonic_map = {idx + 1: np.asarray(row["representative"], dtype=float) for idx, row in enumerate(selected)}
    return harmonic_map, {
        "kind": kind,
        "count": count,
        "generation": "valley_geometry_orbit",
        "selection_rule": str(profile.get("selection_rule", f"{valley_type}_{kind}_geometry")),
        "selected": [
            {
                "index": idx,
                "vector": np.asarray(vector, dtype=float).tolist(),
                "norm": float(np.linalg.norm(vector)),
                "pair_count": None,
                "member_count": int(len(selected[idx - 1]["orbit"])),
                "orbit_size": int(len(selected[idx - 1]["orbit"])),
                "orbit_generators": list(selected[idx - 1]["orbit_generators"]),
                "orbit_vectors": [np.asarray(item, dtype=float).tolist() for item in selected[idx - 1]["orbit"]],
                "seed": str(selected[idx - 1]["seed_expr"]),
                "lattice_index": None,
            }
            for idx, vector in harmonic_map.items()
        ],
    }


def _resolve_harmonics_maps(
    harmonics: Mapping[str, Any],
    variables: Mapping[str, Any],
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    sectors: list[dict[str, Any]] | None = None,
    valley_model: Mapping[str, Any] | None = None,
    symmetry_map: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    diagnostics: dict[str, Any] = {"auto_used": False}
    maps: dict[str, dict[int, np.ndarray]] = {}
    for kind in ("intra", "inter"):
        raw = harmonics.get(kind, {})
        if _is_auto_harmonic_spec(raw):
            count = _auto_harmonic_count(raw, name=f"harmonics.{kind}")
            use_explicit_sector_pairs = (
                kind == "inter"
                and bool(sectors)
                and isinstance(raw, Mapping)
                and raw.get("sector_pairs") is not None
            )
            if use_explicit_sector_pairs:
                maps[kind], diagnostics[kind] = _auto_harmonics_from_support(
                    raw=raw,
                    kind=kind,
                    count=count,
                    sectors=sectors,
                    Q_set1=Q_set1,
                    Q_set2=Q_set2,
                    bM1=bM1,
                    bM2=bM2,
                    symmetry_operations=list((symmetry_map or {}).get(kind, [])),
                )
            else:
                valley_harmonics = _auto_valley_harmonics(
                    kind=kind,
                    count=count,
                    valley_model=valley_model,
                    bM1=bM1,
                    bM2=bM2,
                    variables=variables,
                )
                if valley_harmonics is not None:
                    maps[kind], diagnostics[kind] = valley_harmonics
                elif sectors:
                    maps[kind], diagnostics[kind] = _auto_harmonics_from_support(
                        raw=raw,
                        kind=kind,
                        count=count,
                        sectors=sectors,
                        Q_set1=Q_set1,
                        Q_set2=Q_set2,
                        bM1=bM1,
                        bM2=bM2,
                        symmetry_operations=list((symmetry_map or {}).get(kind, [])),
                    )
                else:
                    maps[kind], diagnostics[kind] = _auto_harmonics_from_q_sets(
                        kind=kind,
                        count=count,
                        Q_set1=Q_set1,
                        Q_set2=Q_set2,
                        bM1=bM1,
                        bM2=bM2,
                    )
            diagnostics["auto_used"] = True
        else:
            maps[kind] = _harmonics_map(dict(raw), variables)
            diagnostics[kind] = {
                "kind": kind,
                "count": len(maps[kind]),
                "harmonics_source": "explicit_config",
                "selected": [
                    {
                        "index": int(index),
                        "vector": np.asarray(vector, dtype=float).tolist(),
                        "norm": float(np.linalg.norm(vector)),
                        "pair_count": None,
                        "member_count": None,
                        "lattice_index": None,
                        "p_label": None,
                        "support_count": None,
                        "support_weight_norm": None,
                        "chosen": True,
                        "rejected_reason": None,
                    }
                    for index, vector in sorted(maps[kind].items())
                ],
            }
    return maps["intra"], maps["inter"], diagnostics


def _load_kpoints(config: ConfiguredModel) -> np.ndarray:
    if config.kpoints_file is not None:
        kpoints = _validate_kpoints(np.load(config.kpoints_file, allow_pickle=False))
        if bool(getattr(config, "kpoints_from_projection", False)):
            heff_rows = int(np.load(config.heff_file, mmap_mode="r", allow_pickle=False).shape[0])
            if len(kpoints) != heff_rows:
                raise ValueError(
                    "projection kpoints/Heff row mismatch: "
                    f"{len(kpoints)} != {heff_rows} ({config.kpoints_file})"
                )
            identity = getattr(config, "artifact_identity", {})
            stored_hash = require_identity_fields(
                identity,
                ("kpoints_hash",),
                f"KP projection k-points {config.kpoints_file}",
            )["kpoints_hash"]
            actual_hash = hash_array(kpoints)
            if actual_hash != stored_hash:
                raise ValueError(
                    f"KP projection k-points {config.kpoints_file} kpoints_hash mismatch: "
                    f"{stored_hash} != {actual_hash}"
                )
            return np.asarray([rot(point, float(config.rotation_deg)) for point in kpoints], dtype=float)
        return _apply_project_k_indices(kpoints, config.project_k_indices)

    if not config.kpath_config:
        raise ValueError("Configured model requires kpoints_file or kpath section")
    kpath = config.kpath_config
    file_path = _resolve_path(kpath.get("file"), config.path.parent)
    if file_path is None:
        raise ValueError("kpath.file is required when kpoints_file is absent")
    tmat = np.asarray(kpath.get("tmat"), dtype=float)
    phase_deg = float(config.rotation_deg)
    segment_points = kpath.get("segment_points")
    output_file = _resolve_path(kpath.get("output_file"), config.path.parent)
    generated = generate_kpath_from_file(
        Tmat=tmat,
        file_path=file_path,
        phase_deg=phase_deg,
        segment_points=None if segment_points is None else int(segment_points),
        output_file_path=output_file,
    )
    return _apply_project_k_indices(generated.kpoints_2d, config.project_k_indices)


def _load_kpoints_from_inputs(
    *,
    kpoints_file: Path | None,
    kpath_config: Mapping[str, Any],
    base: Path,
    rotation_deg: float,
    kpoints_from_projection: bool = False,
) -> np.ndarray:
    if kpoints_file is not None:
        points = _validate_kpoints(np.load(kpoints_file, allow_pickle=False))
        if kpoints_from_projection:
            return np.asarray([rot(point, float(rotation_deg)) for point in points], dtype=float)
        return points
    if not kpath_config:
        raise ValueError("fit.mode=auto_compact requires kpoints_file or kpath section")
    file_path = _resolve_path(kpath_config.get("file"), base)
    if file_path is None:
        raise ValueError("kpath.file is required when fit.mode=auto_compact uses generated k-points")
    tmat = np.asarray(kpath_config.get("tmat"), dtype=float)
    segment_points = kpath_config.get("segment_points")
    output_file = _resolve_path(kpath_config.get("output_file"), base)
    generated = generate_kpath_from_file(
        Tmat=tmat,
        file_path=file_path,
        phase_deg=float(rotation_deg),
        segment_points=None if segment_points is None else int(segment_points),
        output_file_path=output_file,
    )
    return _validate_kpoints(generated.kpoints_2d)


def _select_auto_fit_indices(
    kpoints: np.ndarray,
    fit_config: Mapping[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    mode = str(fit_config.get("mode", "auto_compact")).strip().lower()
    if mode == "auto":
        mode = "auto_compact"
    if mode != "auto_compact":
        raise ValueError(f"Unsupported fit.mode {fit_config.get('mode')!r}; expected 'auto_compact'")
    arr = _validate_kpoints(kpoints)
    candidate_count = int(arr.shape[0])
    if candidate_count <= 0:
        raise ValueError("fit.mode=auto_compact requires at least one k-point")
    max_points = int(fit_config.get("max_points", 4))
    if max_points <= 0:
        raise ValueError(f"fit.max_points must be positive for auto_compact, got {max_points}")
    count = min(max_points, candidate_count)
    if count == 1:
        selected = [0]
    else:
        selected = sorted({int(round(i * (candidate_count - 1) / (count - 1))) for i in range(count)})
        if len(selected) < count:
            for idx in range(candidate_count):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) == count:
                    break
            selected.sort()
    metadata = {
        "mode": "auto_compact",
        "source": "auto_compact_path_spacing",
        "requested_max_points": max_points,
        "candidate_count": candidate_count,
        "selected_indices": [int(idx) for idx in selected],
    }
    return [int(idx) for idx in selected], metadata


def _validate_kpoints(kpoints: Any) -> np.ndarray:
    arr = np.asarray(kpoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk,2), got {arr.shape}")
    return arr


def _select_rows(arr: np.ndarray, indices: Sequence[int] | None) -> np.ndarray:
    if indices is None:
        return np.asarray(arr)
    idx = np.asarray(list(indices), dtype=int)
    return np.asarray(arr)[idx]


def _block_diag_heff(heff_list: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if not indices:
        raise ValueError("fit.indices must contain at least one k-point index")
    blocks = [np.asarray(heff_list[int(i)]) for i in indices]
    return scipy.linalg.block_diag(*blocks)


def _symmetry_artifact_path(config: ConfiguredModel) -> Path | None:
    if config.symmetry_source_config.get("type") != "kp_symm_output":
        return None
    raw_path = config.symmetry_source_config.get("path")
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else (config.path.parent / path).resolve()


def _load_symmetry_artifact_manifest(path: Path) -> dict[str, Any]:
    for name in ("manifest.json", "summary.json"):
        candidate = path / name
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            return data if isinstance(data, dict) else {}
    return {}


def _load_q_sets_from_symmetry_artifact(config: ConfiguredModel) -> tuple[np.ndarray, np.ndarray] | None:
    artifact_path = _symmetry_artifact_path(config)
    if artifact_path is None:
        return None
    representations = artifact_path / "representations.npz"
    canonical_keys = (
        "__q_model_canonical_layer1__",
        "__q_model_canonical_layer2__",
    )
    raw_keys = (
        "__q_model_raw_layer1__",
        "__q_model_raw_layer2__",
    )
    if representations.is_file():
        with np.load(representations, allow_pickle=False) as payload:
            present_canonical = tuple(key in payload.files for key in canonical_keys)
            present_raw = tuple(key in payload.files for key in raw_keys)
            if any(present_canonical) and not all(present_canonical):
                raise ValueError(
                    f"kp symm package contains an incomplete canonical Q pair: {representations}"
                )
            if any(present_raw) and not all(present_raw):
                raise ValueError(f"kp symm package contains an incomplete raw Q pair: {representations}")
            if all(present_canonical):
                if not all(present_raw) or "__metadata_json__" not in payload.files:
                    raise ValueError(
                        f"kp symm canonical Q package lacks raw Q or metadata: {representations}"
                    )
                q1 = np.asarray(payload[canonical_keys[0]], dtype=np.float64)
                q2 = np.asarray(payload[canonical_keys[1]], dtype=np.float64)
                raw_q1 = np.asarray(payload[raw_keys[0]], dtype=np.float64)
                raw_q2 = np.asarray(payload[raw_keys[1]], dtype=np.float64)
                metadata = json.loads(str(payload["__metadata_json__"].item()))
                q_model = metadata.get("q_model", {}) if isinstance(metadata, Mapping) else {}
                canonicalization = (
                    q_model.get("canonicalization", {}) if isinstance(q_model, Mapping) else {}
                )
                if not isinstance(canonicalization, Mapping) or canonicalization.get("status") not in {
                    "certified",
                    "not_needed",
                }:
                    raise ValueError(
                        f"kp symm canonical Q package is not certified: {representations}"
                    )
                sector_order = tuple(canonicalization.get("sector_order", ()))
                if sector_order != ("L1", "L2"):
                    raise ValueError(
                        "kp symm canonical Q package has unsupported sector ordering: "
                        f"{sector_order!r}"
                    )
                from ..symmetry.q_canonicalization import q_geometry_hash

                expected_raw_hash = str(canonicalization.get("raw_q_hash", ""))
                expected_canonical_hash = str(canonicalization.get("canonical_q_hash", ""))
                actual_raw_hash = q_geometry_hash(
                    {"L1": raw_q1, "L2": raw_q2}, sector_order
                )
                actual_canonical_hash = q_geometry_hash(
                    {"L1": q1, "L2": q2}, sector_order
                )
                if actual_raw_hash != expected_raw_hash:
                    raise ValueError("kp symm raw Q hash does not match the packed arrays")
                if actual_canonical_hash != expected_canonical_hash:
                    raise ValueError("kp symm canonical Q hash does not match the packed arrays")
                for name, array in (("layer1", q1), ("layer2", q2)):
                    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
                        raise ValueError(
                            f"kp symm canonical Q {name} must be a finite (N, 2) array"
                        )
                return q1, q2
    manifest = _load_symmetry_artifact_manifest(artifact_path)
    q_model = manifest.get("q_model", {})
    if not isinstance(q_model, Mapping):
        return None
    files = q_model.get("files", {})
    if not isinstance(files, Mapping):
        return None
    layer1 = files.get("layer1")
    layer2 = files.get("layer2")
    if layer1 is None or layer2 is None:
        return None
    q1_path = Path(str(layer1))
    q2_path = Path(str(layer2))
    if not q1_path.is_absolute():
        q1_path = artifact_path / q1_path
    if not q2_path.is_absolute():
        q2_path = artifact_path / q2_path
    if not q1_path.exists() or not q2_path.exists():
        return None
    return np.load(q1_path), np.load(q2_path)


def _load_model_q_sets(config: ConfiguredModel) -> tuple[np.ndarray, np.ndarray]:
    artifact_q_sets = _load_q_sets_from_symmetry_artifact(config)
    if artifact_q_sets is not None:
        return artifact_q_sets
    if (
        str(config.symmetry_source_config.get("type", "")) == "kp_symm_output"
        and str(getattr(config, "response_semantics", "legacy_frozen_v1")) == "complete_linear_v2"
    ):
        raise ValueError(
            "complete_linear_v2 requires canonical Q arrays from the exactified symmetry package; "
            "rerun `kp symm` with the current code"
        )
    return load_Q_sets_from_gvec_files(
        config.qset1_file,
        config.qset2_file,
        rotation_deg=config.rotation_deg,
    )


def _operation_matrix_is_exactified(record: Mapping[str, Any]) -> bool:
    return exactified_operation_provenance_is_complete(record)


def _requires_model_side_exactification(metadata: Mapping[str, Any]) -> bool:
    if bool(metadata.get("requires_model_exactification", False)):
        return True
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)) or not operations:
        return False
    return not all(isinstance(record, Mapping) and _operation_matrix_is_exactified(record) for record in operations)


def build_moire_config_from_file(path: str | Path) -> tuple[MoireConfig, ConfiguredModel]:
    config = load_model_config(path)
    config.artifact_identity = _load_model_artifact_identity(
        config.heff_file,
        config.symmetry_source_config,
        base=config.path.parent,
    )
    heff_list = np.load(config.heff_file, mmap_mode="r")
    Q_set1, Q_set2 = _load_model_q_sets(config)

    bM1, bM2, bM_diagnostics = _build_bM_vectors(Q_set1, Q_set2, config)
    config.bM_diagnostics = bM_diagnostics
    model_section = _model_section(config.raw)
    variables = _build_variables(Q_set1, bM1, bM2, model_section)
    harmonics = config.harmonics_config
    sectors = sectors_with_q_offsets(
        _default_sectors(config),
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        bM1=bM1,
        bM2=bM2,
    )
    preliminary_symmetry_map = _enrich_symmetry_map(
        config.symmetry_map,
        {},
        rotation_deg=config.rotation_deg,
        require_action_metadata=False,
    )
    intra, inter, harmonics_diagnostics = _resolve_harmonics_maps(
        harmonics,
        variables,
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        bM1=bM1,
        bM2=bM2,
        sectors=sectors,
        valley_model=config.valley_model,
        symmetry_map=preliminary_symmetry_map,
    )
    config.harmonics_diagnostics = harmonics_diagnostics
    term_templates, inter_direction_diagnostics = _orient_k_sector_inter_templates_by_raw_support(
        config.term_templates,
        valley_model=config.valley_model,
        n_orb=config.n_orb,
        sectors=sectors,
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        inter_harmonics_map=inter,
        term_template_metadata=config.term_template_metadata,
    )
    config.term_templates = term_templates
    if inter_direction_diagnostics:
        config.term_template_metadata = dict(config.term_template_metadata)
        config.term_template_metadata["inter_sector_pair_resolution"] = inter_direction_diagnostics
    kpoints_all = _load_kpoints(config)
    band_kpoints = _select_rows(kpoints_all, config.band_indices)
    fit_kpoints = _select_rows(kpoints_all, config.fit_indices)
    expected_dim = len(Q_set1) * config.n_orb[0] + len(Q_set2) * config.n_orb[1]
    loaded_symmetry = load_symmetry_source(config.symmetry_source_config, base=config.path.parent, expected_dim=expected_dim)
    source_type = str(config.symmetry_source_config.get("type", "none"))
    if source_type == "kp_symm_output" and _requires_model_side_exactification(loaded_symmetry.metadata):
        raise ValueError(
            "kp_symm_output must provide exactified continuum matrices. "
            "Rerun `kp symm` so the manifest contains matrix_kind='continuum_internal_rep_exact' "
            "and matrix_source='kp_symm_exactified_action'."
        )
    if config.symmetry_source_metadata:
        loaded_symmetry.metadata = {**config.symmetry_source_metadata, **loaded_symmetry.metadata}
        if hasattr(loaded_symmetry.generator, "metadata") and isinstance(loaded_symmetry.generator.metadata, dict):
            loaded_symmetry.generator.metadata.update(config.symmetry_source_metadata)
    config.symmetry_source_metadata = loaded_symmetry.metadata
    enriched_symmetry_map = _enrich_symmetry_map(
        config.symmetry_map,
        loaded_symmetry.metadata,
        rotation_deg=config.rotation_deg,
    )
    config.symmetry_map = enriched_symmetry_map

    moire_config = MoireConfig(
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        n_orb1=config.n_orb[0],
        n_orb2=config.n_orb[1],
        nlow_state=list(config.nlow_state),
        bM1=bM1,
        bM2=bM2,
        intra_harmonics_map=intra,
        inter_harmonics_map=inter,
        max_order=dict(config.max_order),
        symmetry_map=dict(enriched_symmetry_map),
        kpoints=band_kpoints,
        kpoints_fit=fit_kpoints,
        heff=_block_diag_heff(heff_list, config.fit_indices),
        coeff_tol=config.coeff_tol,
        null_channel_abs_tol=config.null_channel_abs_tol,
        null_channel_rel_tol=config.null_channel_rel_tol,
        output_dir=config.output_dir,
    )
    if loaded_symmetry.generator is not None:
        moire_config.symmetry_gen = loaded_symmetry.generator
    moire_config.term_templates = [dict(item) for item in config.term_templates]
    moire_config.sectors = sectors
    moire_config.symmetry_source_metadata = loaded_symmetry.metadata
    moire_config.bM_diagnostics = bM_diagnostics
    moire_config.orbital_count_metadata = dict(config.orbital_count_metadata)
    setattr(moire_config, "_harmonics_diagnostics", harmonics_diagnostics)
    return moire_config, config


def _bM_candidate_rows(candidates: list[np.ndarray], *, max_rows: int = 24) -> list[dict[str, Any]]:
    rows = []
    for vector in candidates:
        arr = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm <= 1.0e-12:
            continue
        rows.append({"vector": arr.tolist(), "norm": norm, "angle_deg": float(np.degrees(np.arctan2(arr[1], arr[0])))})
    rows.sort(key=lambda item: (item["norm"], item["angle_deg"]))
    return rows[:max_rows]


def _reconstruction_diagnostics(
    *,
    vectors: list[np.ndarray],
    bM1: np.ndarray,
    bM2: np.ndarray,
    tol: float,
) -> dict[str, Any]:
    basis = np.column_stack([bM1, bM2])
    errors: list[float] = []
    failures = 0
    checked = 0
    for vector in vectors:
        arr = np.asarray(vector, dtype=float)
        if np.linalg.norm(arr) <= 1.0e-12:
            continue
        checked += 1
        coeff = np.linalg.solve(basis, arr)
        rounded = np.rint(coeff)
        error = float(np.linalg.norm(arr - basis @ rounded))
        errors.append(error)
        if error > tol:
            failures += 1
    if errors:
        max_error = float(np.max(errors))
        median_error = float(np.median(errors))
    else:
        max_error = 0.0
        median_error = 0.0
    return {
        "checked_count": int(checked),
        "max_error": max_error,
        "median_error": median_error,
        "failure_count": int(failures),
        "failure_fraction": float(failures / checked) if checked else 0.0,
        "tolerance": float(tol),
    }


def _bM_geometry(bM1: np.ndarray, bM2: np.ndarray) -> dict[str, Any]:
    n1 = float(np.linalg.norm(bM1))
    n2 = float(np.linalg.norm(bM2))
    dot = float(np.dot(bM1, bM2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 1.0
    return {
        "b1": np.asarray(bM1, dtype=float).tolist(),
        "b2": np.asarray(bM2, dtype=float).tolist(),
        "norms": [n1, n2],
        "angle_deg": float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))),
    }


def _bM_candidates_from_tmat(Tmat: Any, *, rotation_deg: float) -> list[np.ndarray]:
    reciprocal = reciprocal_Tmat_from_Tmat(np.asarray(Tmat, dtype=float))
    g1 = rot(reciprocal[0, :2], rotation_deg)
    g2 = rot(reciprocal[1, :2], rotation_deg)
    candidates: list[np.ndarray] = []
    for m in range(-2, 3):
        for n in range(-2, 3):
            if m == 0 and n == 0:
                continue
            candidates.append(m * g1 + n * g2)
    return candidates


def _build_bM_vectors(Q_set1: np.ndarray, Q_set2: np.ndarray, config: ConfiguredModel) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = config.bM_config
    angle = float(raw.get("angle_deg", 60.0))
    source = str(raw.get("source", "auto")).lower()
    if raw.get("infer_from_q", False):
        source = "q_distance"
    if source == "auto" and "bM1" in raw:
        source = "explicit"
    if not raw:
        source = "q_distance"
    elif source == "auto":
        source = "tmat" if config.kpath_config.get("tmat") is not None else "q_distance"
    if source in {"q", "q_distance", "q_distances"}:
        candidates = bM_candidates_from_q_distances(Q_set1, Q_set2)
        bM1, bM2 = canonical_bM_pair_from_candidates(candidates, angle_deg=angle)
        b_norm = max(float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
        validation = raw.get("validation", {})
        if validation is None:
            validation = {}
        if not isinstance(validation, Mapping):
            raise ValueError("model.bM.validation must be a mapping")
        tol = float(validation.get("max_error", max(1.0e-8, b_norm * 1.0e-6)))
        max_failure_fraction = float(validation.get("max_failure_fraction", 0.0))
        reconstruction = _reconstruction_diagnostics(vectors=candidates, bM1=bM1, bM2=bM2, tol=tol)
        diagnostics = {
            "source": "q_distance",
            **_bM_geometry(bM1, bM2),
            "candidates": _bM_candidate_rows(candidates),
            "reconstruction": reconstruction,
        }
        if reconstruction["failure_fraction"] > max_failure_fraction:
            raise ValueError(
                "q_distance bM reconstruction validation failed: "
                f"failure_fraction={reconstruction['failure_fraction']:.3g} > {max_failure_fraction:.3g}, "
                f"max_error={reconstruction['max_error']:.3g}"
            )
        if config.kpath_config.get("tmat") is not None:
            tmat_candidates = _bM_candidates_from_tmat(config.kpath_config["tmat"], rotation_deg=config.rotation_deg)
            t_bM1, t_bM2 = canonical_bM_pair_from_candidates(tmat_candidates, angle_deg=angle)
            diagnostics["comparison_with_tmat"] = {
                "b1_delta_norm": float(np.linalg.norm(bM1 - t_bM1)),
                "b2_delta_norm": float(np.linalg.norm(bM2 - t_bM2)),
                "tmat_b1": t_bM1.tolist(),
                "tmat_b2": t_bM2.tolist(),
            }
        return bM1, bM2, diagnostics
    if source == "tmat":
        if config.kpath_config.get("tmat") is None:
            raise ValueError("model.bM.source=tmat requires kpath.tmat")
        candidates = _bM_candidates_from_tmat(config.kpath_config["tmat"], rotation_deg=config.rotation_deg)
        bM1, bM2 = canonical_bM_pair_from_candidates(candidates, angle_deg=angle)
        diagnostics = {
            "source": "tmat",
            **_bM_geometry(bM1, bM2),
            "candidates": _bM_candidate_rows(candidates),
            "comparison_with_tmat": {"b1_delta_norm": 0.0, "b2_delta_norm": 0.0, "tmat_b1": bM1.tolist(), "tmat_b2": bM2.tolist()},
        }
        return bM1, bM2, diagnostics
    variables = {"zero": np.zeros(2), "pi": float(np.pi), "sqrt3": float(np.sqrt(3.0))}
    if "bM1" not in raw:
        raise ValueError("model.bM must provide bM1 or source q_distance/tmat")
    bM1 = evaluate_vector_expression(raw["bM1"], variables)
    variables["bM1"] = bM1
    if "bM2" in raw:
        bM2 = evaluate_vector_expression(raw["bM2"], variables)
    else:
        bM2 = rot(bM1, float(raw.get("angle_deg", 60.0)))
    return bM1, bM2, {"source": "explicit", **_bM_geometry(bM1, bM2), "candidates": []}


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
    rows = _model_row_metadata_for_harmonic_scan(qset1, qset2, n_orb)
    shell_norms = _harmonic_shell_norms_from_qsets(qset1, qset2, tol=tol)
    q = np.asarray([np.asarray(row["q"], dtype=float) for row in rows], dtype=float)
    layers = np.asarray([int(row["layer"]) for row in rows], dtype=np.int16)
    norms = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    zero = norms <= float(tol)
    same_layer = layers[:, None] == layers[None, :]
    intra_count = int(current_counts.get("intra", 0))
    inter_count = int(current_counts.get("inter", 0))
    intra_positive_shells = max(0, intra_count - 1)
    inter_positive_shells = max(0, inter_count - 1)
    intra_shell = _shell_index_matrix(norms, shell_norms.get("intra", []))
    inter_shell = _shell_index_matrix(norms, shell_norms.get("inter", []))
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


def _hamiltonian_element_comparison_arrays(
    model_hamiltonians: np.ndarray,
    heff_hamiltonians: np.ndarray,
    harmonic_mask: np.ndarray,
    *,
    positions: Sequence[int] | None = None,
    qset1: np.ndarray | None = None,
    qset2: np.ndarray | None = None,
    n_orb: tuple[int, int] | None = None,
    subtract_layer_diagonal_mean: bool = False,
) -> dict[str, Any]:
    model = np.asarray(model_hamiltonians, dtype=np.complex128)
    heff = np.asarray(heff_hamiltonians, dtype=np.complex128)
    mask = np.asarray(harmonic_mask, dtype=bool)
    if model.ndim == 2:
        model = model[None, :, :]
    if heff.ndim == 2:
        heff = heff[None, :, :]
    if model.shape != heff.shape:
        raise ValueError(f"model/heff Hamiltonian arrays must have same shape, got {model.shape} and {heff.shape}")
    if model.ndim != 3 or model.shape[-1] != model.shape[-2]:
        raise ValueError(f"Hamiltonian arrays must have shape (Nk, dim, dim), got {model.shape}")
    if mask.shape != model.shape[-2:]:
        raise ValueError(f"harmonic mask shape {mask.shape} does not match Hamiltonian dimension {model.shape[-2:]}")
    if positions is None:
        selected = list(range(model.shape[0]))
    else:
        selected = [int(item) for item in positions]
    if not selected:
        raise ValueError("Hamiltonian element comparison requires at least one k-point position")
    for pos in selected:
        if pos < 0 or pos >= model.shape[0]:
            raise IndexError(f"k-point position {pos} is outside 0..{model.shape[0] - 1}")
    model_selected = model[np.asarray(selected, dtype=int)]
    heff_selected = heff[np.asarray(selected, dtype=int)]

    display_order = list(range(model.shape[-1]))
    q_block_boundaries: list[int] = []
    layer_boundaries: list[int] = []
    display_layers: np.ndarray | None = None
    if qset1 is not None and qset2 is not None and n_orb is not None:
        display = _q_block_display_metadata(qset1, qset2, n_orb, dim=model.shape[-1])
        display_order = display["order"]
        q_block_boundaries = display["q_block_boundaries"]
        layer_boundaries = display["layer_boundaries"]
        display_layers = display["layers"]
    else:
        display_layers = np.ones(model.shape[-1], dtype=np.int16)

    heff_offsets: list[list[float]] = []
    model_offsets: list[list[float]] = []
    if subtract_layer_diagonal_mean:
        heff_selected, heff_offsets = _subtract_layer_diagonal_means(heff_selected, display_layers)
        model_selected, model_offsets = _subtract_layer_diagonal_means(model_selected, display_layers)

    order = np.asarray(display_order, dtype=int)
    model_selected = model_selected[:, order][:, :, order]
    heff_selected = heff_selected[:, order][:, :, order]
    mask = mask[np.ix_(order, order)]
    diff_selected = model_selected - heff_selected
    mask3 = mask[None, :, :]
    return {
        "positions": selected,
        "heff": np.where(mask3, np.abs(heff_selected), np.nan),
        "model": np.where(mask3, np.abs(model_selected), np.nan),
        "diff": np.where(mask3, np.abs(diff_selected), np.nan),
        "mask": mask,
        "display_order": display_order,
        "q_block_boundaries": q_block_boundaries,
        "layer_boundaries": layer_boundaries,
        "diag_offsets": {
            "heff": heff_offsets,
            "model": model_offsets,
        },
    }


def _q_block_display_metadata(
    qset1: np.ndarray,
    qset2: np.ndarray,
    n_orb: tuple[int, int],
    *,
    dim: int,
) -> dict[str, Any]:
    rows = _model_row_metadata_for_harmonic_scan(qset1, qset2, n_orb)
    if len(rows) != int(dim):
        raise ValueError(f"basis metadata dimension {len(rows)} does not match Hamiltonian dimension {dim}")
    order = sorted(
        range(len(rows)),
        key=lambda idx: (
            int(rows[idx]["layer"]),
            int(rows[idx]["q_index"]),
            int(rows[idx]["orbital"]),
        ),
    )
    ordered_rows = [rows[idx] for idx in order]
    layers = np.asarray([int(row["layer"]) for row in ordered_rows], dtype=np.int16)
    q_block_boundaries: list[int] = []
    layer_boundaries: list[int] = []
    for idx in range(1, len(ordered_rows)):
        prev = ordered_rows[idx - 1]
        curr = ordered_rows[idx]
        if int(prev["layer"]) != int(curr["layer"]):
            layer_boundaries.append(idx)
            q_block_boundaries.append(idx)
        elif int(prev["q_index"]) != int(curr["q_index"]):
            q_block_boundaries.append(idx)
    return {
        "order": [int(idx) for idx in order],
        "layers": layers,
        "q_block_boundaries": q_block_boundaries,
        "layer_boundaries": layer_boundaries,
    }


def _subtract_layer_diagonal_means(
    hamiltonians: np.ndarray,
    layers: np.ndarray,
) -> tuple[np.ndarray, list[list[float]]]:
    adjusted = np.asarray(hamiltonians, dtype=np.complex128).copy()
    layer_arr = np.asarray(layers, dtype=int)
    unique_layers = [int(item) for item in np.unique(layer_arr)]
    offsets: list[list[float]] = []
    for ik in range(adjusted.shape[0]):
        row_offsets: list[float] = []
        diag = np.diagonal(adjusted[ik]).copy()
        for layer in unique_layers:
            indices = np.flatnonzero(layer_arr == layer)
            if indices.size == 0:
                row_offsets.append(0.0)
                continue
            offset = complex(np.mean(diag[indices]))
            adjusted[ik, indices, indices] -= offset
            row_offsets.append(float(np.real_if_close(offset).real))
        offsets.append(row_offsets)
    return adjusted, offsets


def _finite_heatmap_vmax(*arrays: np.ndarray) -> float:
    values = []
    for arr in arrays:
        data = np.asarray(arr, dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size:
            values.append(float(np.max(finite)))
    vmax = max(values) if values else 0.0
    return vmax if vmax > 0.0 else 1.0


def _matrix_index_ticks(dim: int, *, max_ticks: int = 8) -> list[int]:
    size = int(dim)
    if size <= 0:
        return []
    if size <= max_ticks:
        return list(range(size))
    step = max(1, int(math.ceil((size - 1) / float(max_ticks - 1))))
    ticks = list(range(0, size, step))
    if ticks[-1] != size - 1:
        ticks.append(size - 1)
    return ticks


def save_hamiltonian_element_comparison_plot(
    model_hamiltonians: np.ndarray,
    heff_hamiltonians: np.ndarray,
    path: str | Path,
    *,
    harmonic_mask: np.ndarray,
    positions: Sequence[int] | None = None,
    k_indices: Sequence[int] | None = None,
    title: str | None = None,
    qset1: np.ndarray | None = None,
    qset2: np.ndarray | None = None,
    n_orb: tuple[int, int] | None = None,
    subtract_layer_diagonal_mean: bool = True,
) -> Path:
    panels = _hamiltonian_element_comparison_arrays(
        model_hamiltonians,
        heff_hamiltonians,
        harmonic_mask,
        positions=positions,
        qset1=qset1,
        qset2=qset2,
        n_orb=n_orb,
        subtract_layer_diagonal_mean=subtract_layer_diagonal_mean,
    )
    heff = np.asarray(panels["heff"], dtype=float)
    model = np.asarray(panels["model"], dtype=float)
    diff = np.asarray(panels["diff"], dtype=float)
    nrows = int(heff.shape[0])
    if k_indices is None:
        labels = [f"k position {idx}" for idx in panels["positions"]]
    else:
        labels = [f"k index {int(idx)}" for idx in k_indices]
        if len(labels) != nrows:
            raise ValueError(f"k_indices length {len(labels)} does not match plotted row count {nrows}")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out = out.with_suffix(".pdf")
    raster_dpi = max(int(KP_DPI), 360)
    main_vmax = _finite_heatmap_vmax(heff, model)
    diff_vmax = _finite_heatmap_vmax(diff)
    dim = int(heff.shape[-1])
    index_ticks = _matrix_index_ticks(dim)
    q_boundaries = [int(item) for item in panels.get("q_block_boundaries", [])]
    layer_boundaries = [int(item) for item in panels.get("layer_boundaries", [])]
    diag_offsets = panels.get("diag_offsets", {})

    def format_offsets(kind: str, row_index: int) -> str:
        rows = diag_offsets.get(kind, []) if isinstance(diag_offsets, Mapping) else []
        if not rows or row_index >= len(rows):
            return ""
        values = rows[row_index]
        if not values:
            return ""
        parts = [f"L{idx + 1}={float(value):+.4g}" for idx, value in enumerate(values)]
        return "diag mean subtracted\n" + ", ".join(parts) + " eV"

    def draw_boundaries(ax: Any) -> None:
        for boundary in q_boundaries:
            pos = float(boundary) - 0.5
            ax.axhline(pos, color="0.65", linewidth=0.3, alpha=0.85)
            ax.axvline(pos, color="0.65", linewidth=0.3, alpha=0.85)
        for boundary in layer_boundaries:
            pos = float(boundary) - 0.5
            ax.axhline(pos, color="black", linewidth=1.0, alpha=0.9)
            ax.axvline(pos, color="black", linewidth=1.0, alpha=0.9)

    with kp_plot_rc_context():
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=3,
            figsize=(8.4, max(2.6, 2.35 * nrows)),
            dpi=KP_DPI,
            squeeze=False,
            constrained_layout=True,
        )
        main_cmap = plt.get_cmap("viridis").copy()
        diff_cmap = plt.get_cmap("magma").copy()
        main_cmap.set_bad(color="white")
        diff_cmap.set_bad(color="white")
        main_norm = Normalize(vmin=0.0, vmax=main_vmax)
        diff_norm = Normalize(vmin=0.0, vmax=diff_vmax)
        columns = [
            ("Heff |H|", heff, main_cmap, main_norm),
            ("Model |H|", model, main_cmap, main_norm),
            ("|Model - Heff|", diff, diff_cmap, diff_norm),
        ]
        for irow in range(nrows):
            for icol, (col_title, data, cmap, norm) in enumerate(columns):
                ax = axes[irow, icol]
                ax.imshow(np.ma.masked_invalid(data[irow]), origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
                draw_boundaries(ax)
                ax.set_xticks(index_ticks)
                ax.set_yticks(index_ticks)
                ax.tick_params(axis="both", labelsize=6, length=2)
                if irow != nrows - 1:
                    ax.set_xticklabels([])
                if icol != 0:
                    ax.set_yticklabels([])
                ax.set_box_aspect(1.0)
                if irow == 0:
                    ax.set_title(col_title)
                if icol == 0:
                    ax.set_ylabel(labels[irow], rotation=90, labelpad=14)
                    text = format_offsets("heff", irow)
                    if text:
                        ax.text(
                            0.02,
                            0.98,
                            text,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=5.5,
                            color="white",
                            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"},
                        )
                elif icol == 1:
                    text = format_offsets("model", irow)
                    if text:
                        ax.text(
                            0.02,
                            0.98,
                            text,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=5.5,
                            color="white",
                            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"},
                        )
                if irow == nrows - 1:
                    ax.set_xlabel("column: layer/Q/orbital")
        if title:
            fig.suptitle(title)
        main_sm = ScalarMappable(norm=main_norm, cmap=main_cmap)
        diff_sm = ScalarMappable(norm=diff_norm, cmap=diff_cmap)
        fig.colorbar(main_sm, ax=axes[:, :2].ravel().tolist(), fraction=0.025, pad=0.02, label="|H| (eV)")
        fig.colorbar(diff_sm, ax=axes[:, 2].ravel().tolist(), fraction=0.025, pad=0.02, label="|ΔH| (eV)")
        fig.savefig(out, dpi=raster_dpi, bbox_inches="tight")
        fig.savefig(pdf_out, bbox_inches="tight")
        plt.close(fig)
    return out


def _normalise_kpath_label(label: Any) -> str:
    text = str(label).strip()
    if text.lower() in {"gamma", "g", "γ"}:
        return r"$\Gamma$"
    return text


def _resolve_plot_band_slice(
    *,
    nbands: int,
    metric_band_slice: Sequence[int] | None,
    plot_config: Mapping[str, Any],
) -> tuple[int, int]:
    if plot_config.get("band_slice") is not None:
        values = _as_int_list(plot_config.get("band_slice"), name="bands.plot.band_slice")
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


def _plot_comparison_filename(plot_config: Mapping[str, Any] | None) -> str:
    plot_options = dict(plot_config or {})
    align = str(plot_options.get("align", "none")).lower()
    suffix = "_aligned" if align in {"top", "top_band", "top-band", "bottom", "bottom_band", "bottom-band"} else ""
    if plot_options.get("bottom_bands") is not None:
        return f"comparison_bottom{int(plot_options['bottom_bands'])}{suffix}.json"
    if plot_options.get("top_bands") is not None:
        return f"comparison_top{int(plot_options['top_bands'])}{suffix}.json"
    return "comparison_plot.json"


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


def _cleanup_stale_band_outputs(output_dir: Path) -> None:
    for pattern in (
        "comparison_top*.json",
        "band_comparison_top*.pdf",
        "comparison_plot.json",
        "comparison_all_bands.json",
        "band_comparison_all.pdf",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()


def _model_output_profile(model_config: ConfiguredModel) -> str:
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


def _plot_axis_from_kpath(config: ConfiguredModel, npoints: int) -> tuple[np.ndarray, list[float] | None, list[str] | None]:
    if not config.kpath_config:
        return np.arange(npoints, dtype=float), None, None
    if config.band_indices is not None:
        return np.arange(npoints, dtype=float), None, None

    file_path = _resolve_path(config.kpath_config.get("file"), config.path.parent)
    if file_path is None or not file_path.exists() or config.kpath_config.get("tmat") is None:
        return np.arange(npoints, dtype=float), None, None

    generated = generate_kpath_from_file(
        Tmat=np.asarray(config.kpath_config.get("tmat"), dtype=float),
        file_path=file_path,
        phase_deg=float(config.rotation_deg),
        segment_points=None
        if config.kpath_config.get("segment_points") is None
        else int(config.kpath_config.get("segment_points")),
    )
    x = np.asarray(generated.x, dtype=float)
    if x.shape[0] != npoints:
        return np.arange(npoints, dtype=float), None, None
    return x, list(generated.x_ticks), [_normalise_kpath_label(label) for label in generated.labels_ticks]


def _band_plot_title(config: ConfiguredModel) -> str:
    if config.raw.get("title"):
        return str(config.raw["title"])
    source_plot = config.source_raw.get("plot", {})
    if isinstance(source_plot, Mapping) and source_plot.get("title"):
        return str(source_plot["title"])
    return config.path.stem


def _band_overlap_weights(
    model_eigvecs: np.ndarray,
    reference_eigvecs: np.ndarray,
    *,
    model_eigvals: np.ndarray | None = None,
    reference_eigvals: np.ndarray | None = None,
    degeneracy_tol: float = 0.0,
) -> np.ndarray:
    model_vec = np.asarray(model_eigvecs)
    ref_vec = np.asarray(reference_eigvecs)
    if model_vec.ndim != 3 or ref_vec.ndim != 3:
        raise ValueError(
            "band overlap expects eigenvectors with shape (Nk,dim,N), "
            f"got {model_vec.shape} and {ref_vec.shape}"
        )
    if model_vec.shape != ref_vec.shape:
        raise ValueError(f"band overlap eigenvector shapes differ: model={model_vec.shape}, reference={ref_vec.shape}")
    weights = np.abs(np.einsum("kdn,kdn->kn", np.conjugate(model_vec), ref_vec)) ** 2
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    tol = float(degeneracy_tol)
    if tol <= 0.0:
        return weights
    if model_eigvals is None or reference_eigvals is None:
        return weights
    model_val = np.asarray(model_eigvals, dtype=float)
    ref_val = np.asarray(reference_eigvals, dtype=float)
    if model_val.shape != weights.shape or ref_val.shape != weights.shape:
        raise ValueError(
            "band overlap degeneracy eigvals must match overlap shape, "
            f"got model={model_val.shape}, reference={ref_val.shape}, overlap={weights.shape}"
        )

    nk, nbands = weights.shape
    clustered = weights.copy()
    for ik in range(nk):
        start = 0
        for ib in range(nbands - 1):
            model_gap = abs(float(model_val[ik, ib + 1] - model_val[ik, ib]))
            ref_gap = abs(float(ref_val[ik, ib + 1] - ref_val[ik, ib]))
            if model_gap <= tol or ref_gap <= tol:
                continue
            stop = ib + 1
            if stop - start > 1:
                model_subspace = model_vec[ik, :, start:stop]
                ref_subspace = ref_vec[ik, :, start:stop]
                value = float(np.linalg.norm(model_subspace.conj().T @ ref_subspace, ord="fro") ** 2 / (stop - start))
                clustered[ik, start:stop] = np.clip(value, 0.0, 1.0)
            start = stop
        stop = nbands
        if stop - start > 1:
            model_subspace = model_vec[ik, :, start:stop]
            ref_subspace = ref_vec[ik, :, start:stop]
            value = float(np.linalg.norm(model_subspace.conj().T @ ref_subspace, ord="fro") ** 2 / (stop - start))
            clustered[ik, start:stop] = np.clip(value, 0.0, 1.0)
    return clustered


def save_band_comparison_plot(
    model_eigvals: np.ndarray,
    heff_eigvals: np.ndarray,
    path: str | Path,
    *,
    band_slice: Sequence[int] | None = None,
    plot_config: Mapping[str, Any] | None = None,
    x: Sequence[float] | None = None,
    x_ticks: Sequence[float] | None = None,
    x_ticklabels: Sequence[str] | None = None,
    title: str | None = None,
    overlap_weights: np.ndarray | None = None,
) -> Path:
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
    model_sorted = np.sort(model, axis=1)[:, :nbands]
    heff_sorted = np.sort(heff, axis=1)[:, :nbands]
    overlap_sorted = None
    if overlap_weights is not None:
        overlap = np.asarray(overlap_weights, dtype=float)
        if overlap.ndim != 2:
            raise ValueError(f"overlap_weights must be 2D, got {overlap.shape}")
        if overlap.shape[0] != model.shape[0] or overlap.shape[1] < nbands:
            raise ValueError(
                "overlap_weights must have one row per k-point and at least the plotted band count, "
                f"got overlap={overlap.shape}, model={model.shape}, heff={heff.shape}"
            )
        overlap_sorted = np.clip(overlap[:, :nbands], 0.0, 1.0)
    model_sel = model_sorted[:, start:stop]
    heff_sel = heff_sorted[:, start:stop]
    overlap_sel = overlap_sorted[:, start:stop] if overlap_sorted is not None else None
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
    plot_all_bands = bool(plot_options.get("plot_all_bands", plot_options.get("show_all_bands", True)))
    if plot_all_bands:
        model_plot = model_sorted - model_ref
        heff_plot = heff_sorted - heff_ref
        overlap_plot = overlap_sorted
    else:
        model_plot = model_sel
        heff_plot = heff_sel
        overlap_plot = overlap_sel

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray(x, dtype=float) if x is not None else np.arange(model_plot.shape[0], dtype=float)
    if x_values.shape[0] != model_plot.shape[0]:
        raise ValueError(f"plot x-axis length {x_values.shape[0]} does not match k-point count {model_sel.shape[0]}")
    figsize_raw = plot_options.get("figsize", list(KP_BAND_FIGSIZE))
    if not isinstance(figsize_raw, Sequence) or isinstance(figsize_raw, (str, bytes)) or len(figsize_raw) != 2:
        raise ValueError(f"bands.plot.figsize must be [width, height], got {figsize_raw!r}")
    dpi = int(plot_options.get("dpi", KP_DPI))
    fig, ax = plt.subplots(figsize=(float(figsize_raw[0]), float(figsize_raw[1])), dpi=dpi)
    overlap_scalar = None
    overlap_label = str(plot_options.get("overlap_label", "Wavefunction overlap"))
    for ib in range(heff_plot.shape[1]):
        ax.plot(x_values, heff_plot[:, ib], **KP_REFERENCE_STYLE)
    if overlap_plot is not None:
        cmap = plt.get_cmap(str(plot_options.get("overlap_cmap", "Blues")))
        colorbar_values = overlap_sel if overlap_sel is not None and overlap_sel.size else overlap_plot
        default_vmin = float(np.nanmin(colorbar_values))
        default_vmax = float(np.nanmax(colorbar_values))
        vmin = float(plot_options.get("overlap_vmin", default_vmin))
        vmax = float(plot_options.get("overlap_vmax", default_vmax))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            center = 0.5 * (vmin + vmax)
            pad = max(1.0e-6, 0.01 * max(abs(center), 1.0))
            vmin = center - pad
            vmax = center + pad
        norm = Normalize(
            vmin=vmin,
            vmax=vmax,
        )
        linewidth = float(plot_options.get("overlap_linewidth", KP_MODEL_STYLE.get("linewidth", 1.0)))
        alpha = float(plot_options.get("overlap_alpha", KP_MODEL_STYLE.get("alpha", 0.95)))
        linestyle = str(plot_options.get("overlap_linestyle", KP_MODEL_STYLE.get("linestyle", "-")))
        for ib in range(model_plot.shape[1]):
            points = np.column_stack([x_values, model_plot[:, ib]])
            if points.shape[0] < 2:
                ax.plot(
                    x_values,
                    model_plot[:, ib],
                    color=cmap(norm(float(overlap_plot[0, ib]))),
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle=linestyle,
                )
                continue
            segments = np.stack([points[:-1], points[1:]], axis=1)
            values = 0.5 * (overlap_plot[:-1, ib] + overlap_plot[1:, ib])
            collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=linewidth,
                alpha=alpha,
                linestyles=linestyle,
            )
            collection.set_array(np.asarray(values, dtype=float))
            ax.add_collection(collection)
        overlap_scalar = ScalarMappable(norm=norm, cmap=cmap)
        overlap_scalar.set_array([])
    else:
        for ib in range(model_plot.shape[1]):
            ax.plot(x_values, model_plot[:, ib], **KP_MODEL_STYLE)
    if x_ticks is not None and x_ticklabels is not None and len(x_ticks) == len(x_ticklabels):
        ax.set_xticks([float(item) for item in x_ticks])
        ax.set_xticklabels(list(x_ticklabels))
        for tick in x_ticks:
            ax.axvline(float(tick), color="0.86", linewidth=0.75, zorder=0)
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    else:
        ax.set_xlabel("k-point index")
    if model_ref or heff_ref:
        ref_label = "top" if align in {"top", "top_band", "top-band"} else "bottom"
        ax.set_ylabel(relative_energy_ylabel(ref_label))
    else:
        ax.set_ylabel("Energy (eV)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", color="0.88", linewidth=0.65)
    ax.axhline(0.0, color="0.88", linewidth=0.7, zorder=0)
    if model_sel.size and heff_sel.size:
        ymin = float(min(np.min(model_sel), np.min(heff_sel)))
        ymax = float(max(np.max(model_sel), np.max(heff_sel)))
        pad = max(0.004, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
        ylim_cfg = plot_options.get("ylim")
        if ylim_cfg is not None:
            if not isinstance(ylim_cfg, Sequence) or isinstance(ylim_cfg, (str, bytes)) or len(ylim_cfg) != 2:
                raise ValueError(f"bands.plot.ylim must be [ymin, ymax], got {ylim_cfg!r}")
            values = [float(ylim_cfg[0]), float(ylim_cfg[1])]
            if len(values) != 2:
                raise ValueError(f"bands.plot.ylim must be [ymin, ymax], got {ylim_cfg!r}")
            ax.set_ylim(float(values[0]), float(values[1]))
        else:
            ax.set_ylim(ymin - pad, ymax + pad)
    show_metrics = bool(plot_options.get("show_metrics", plot_options.get("annotate_metrics", True)))
    if show_metrics and align in {"top", "top_band", "top-band", "bottom", "bottom_band", "bottom-band"}:
        delta = model_sel - heff_sel
        rms_mev = 1000.0 * float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
        shift_mev = 1000.0 * (model_ref - heff_ref)
        band_label = "top" if align in {"top", "top_band", "top-band"} else "bottom"
        ax.text(
            0.06,
            0.08,
            f"{band_label} {stop - start} bands\naligned RMS {rms_mev:.2f} meV\nshift {shift_mev:.2f} meV",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
    legend_handles = [
        Line2D([0], [0], label="Reference", **KP_REFERENCE_STYLE),
        Line2D([0], [0], label="KP model", **KP_MODEL_STYLE),
    ]
    if bool(plot_options.get("legend_outside", False)):
        ax.legend(
            handles=legend_handles,
            loc=str(plot_options.get("legend_loc", "upper center")),
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=KP_LEGEND_KWARGS["frameon"],
            fontsize=KP_LEGEND_KWARGS["fontsize"],
        )
    else:
        ax.legend(
            handles=legend_handles,
            loc=str(plot_options.get("legend_loc", KP_LEGEND_KWARGS["loc"])),
            frameon=KP_LEGEND_KWARGS["frameon"],
            fontsize=KP_LEGEND_KWARGS["fontsize"],
        )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    box_aspect = plot_options.get("box_aspect", KP_BAND_BOX_ASPECT)
    if box_aspect in {None, False, "none", "None"}:
        resolved_box_aspect = None
    else:
        resolved_box_aspect = float(box_aspect)
    font_family = str(plot_options.get("font_family") or kp_font_family())
    apply_kp_axis_style(
        ax,
        box_aspect=resolved_box_aspect,
        font_family=font_family,
    )
    fig.tight_layout()
    if overlap_scalar is not None:
        colorbar_ax = inset_axes(
            ax,
            width=str(plot_options.get("overlap_colorbar_size", "4%")),
            height="100%",
            loc="lower left",
            bbox_to_anchor=(float(plot_options.get("overlap_colorbar_x", 1.07)), 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        vmin = float(overlap_scalar.norm.vmin)
        vmax = float(overlap_scalar.norm.vmax)
        gradient = np.linspace(vmin, vmax, 256, dtype=float).reshape(-1, 1)
        colorbar_ax.imshow(
            gradient,
            aspect="auto",
            origin="lower",
            cmap=overlap_scalar.cmap,
            norm=overlap_scalar.norm,
            extent=(0.0, 1.0, vmin, vmax),
        )
        colorbar_ax.set_xticks([])
        colorbar_ax.set_yticks(np.linspace(vmin, vmax, 6))
        colorbar_ax.yaxis.tick_right()
        colorbar_ax.yaxis.set_label_position("right")
        colorbar_ax.set_ylabel(overlap_label)
        apply_kp_axis_style(colorbar_ax, box_aspect=None, font_family=font_family)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _first_bz_vertices(bM1: np.ndarray, bM2: np.ndarray) -> np.ndarray:
    from scipy.spatial import Voronoi

    points = []
    origin_index = 0
    for m in range(-2, 3):
        for n in range(-2, 3):
            if m == 0 and n == 0:
                origin_index = len(points)
            points.append(m * np.asarray(bM1, dtype=float) + n * np.asarray(bM2, dtype=float))
    points_arr = np.asarray(points, dtype=float)
    vor = Voronoi(points_arr)
    region_index = vor.point_region[origin_index]
    region = vor.regions[region_index]
    if not region or -1 in region:
        raise ValueError("Failed to construct a bounded first Brillouin-zone polygon")
    vertices = np.asarray(vor.vertices[region], dtype=float)
    center = np.mean(vertices, axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]


def save_harmonics_diagnostic_plot(
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    diagnostics: Mapping[str, Any],
    path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=KP_DPI)
    ax.scatter(Q_set1[:, 0], Q_set1[:, 1], s=28, marker="o", color="#1f77b4", alpha=0.75, label="layer 1 Q")
    ax.scatter(Q_set2[:, 0], Q_set2[:, 1], s=34, marker="^", color="#ff7f0e", alpha=0.75, label="layer 2 Q")

    try:
        bz = _first_bz_vertices(bM1, bM2)
        ax.add_patch(Polygon(bz, closed=True, fill=False, edgecolor="0.25", linewidth=1.2, linestyle="-", label="1st BZ"))
    except Exception:
        bz = np.empty((0, 2), dtype=float)

    selected_vectors: list[tuple[str, int, np.ndarray]] = []
    for kind in ("intra", "inter"):
        for row in diagnostics.get(kind, {}).get("selected", []):
            vector = np.asarray(row["vector"], dtype=float)
            selected_vectors.append((kind, int(row["index"]), vector))

    radii = sorted({round(float(np.linalg.norm(vector)), 12) for _kind, _index, vector in selected_vectors if np.linalg.norm(vector) > 1.0e-10})
    for radius in radii:
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor="0.75", linewidth=0.8, linestyle="--"))

    colors = {"intra": "#6a3d9a", "inter": "#d62728"}
    linestyles = {"intra": "-", "inter": "--"}
    for kind, index, vector in selected_vectors:
        if np.linalg.norm(vector) <= 1.0e-10:
            ax.scatter([0.0], [0.0], s=48, marker="x", color=colors[kind], linewidths=1.2)
            ax.text(0.0, 0.0, f" {kind[0]}{index}", color=colors[kind], fontsize=8, va="bottom")
            continue
        ax.annotate(
            "",
            xy=vector,
            xytext=(0.0, 0.0),
            arrowprops={
                "arrowstyle": "->",
                "color": colors[kind],
                "linewidth": 1.4,
                "linestyle": linestyles[kind],
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        ax.text(vector[0], vector[1], f" {kind[0]}{index}", color=colors[kind], fontsize=8, va="center")

    all_points = [np.asarray(Q_set1), np.asarray(Q_set2), np.zeros((1, 2)), np.asarray([bM1, bM2])]
    if bz.size:
        all_points.append(bz)
    if selected_vectors:
        all_points.append(np.asarray([vector for _kind, _index, vector in selected_vectors], dtype=float))
    stacked = np.vstack(all_points)
    span = float(max(np.ptp(stacked[:, 0]), np.ptp(stacked[:, 1]), np.linalg.norm(bM1), np.linalg.norm(bM2), 1.0))
    center = np.mean(stacked, axis=0)
    pad = 0.12 * span
    ax.set_xlim(center[0] - 0.55 * span - pad, center[0] + 0.55 * span + pad)
    ax.set_ylim(center[1] - 0.55 * span - pad, center[1] + 0.55 * span + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="0.88", linewidth=0.7)
    ax.axvline(0.0, color="0.88", linewidth=0.7)
    ax.set_xlabel("Qx")
    ax.set_ylabel("Qy")
    ax.set_title("Auto-selected harmonic vectors")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color=colors["intra"], lw=1.5, linestyle=linestyles["intra"]),
            Line2D([0], [0], color=colors["inter"], lw=1.5, linestyle=linestyles["inter"]),
        ]
    )
    labels.extend(["intra selected", "inter selected"])
    ax.legend(handles, labels, **KP_LEGEND_KWARGS)
    apply_kp_axis_style(ax, box_aspect=None, font_family=kp_font_family())
    fig.tight_layout()
    fig.savefig(out, dpi=KP_DPI)
    plt.close(fig)
    return out


def save_harmonics_diagnostics(
    *,
    moire_config: MoireConfig,
    model_config: ConfiguredModel,
    output_dir: Path,
) -> None:
    diagnostics = model_config.harmonics_diagnostics
    if not diagnostics:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "harmonics_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    save_harmonics_diagnostic_plot(
        Q_set1=np.asarray(moire_config.Q_set1, dtype=float),
        Q_set2=np.asarray(moire_config.Q_set2, dtype=float),
        bM1=np.asarray(moire_config.bM1, dtype=float),
        bM2=np.asarray(moire_config.bM2, dtype=float),
        diagnostics=diagnostics,
        path=output_dir / "harmonics_diagnostic.pdf",
    )


_Q_LATTICE_PALETTE = {
    "red": "#d1495b",
    "blue": "#3b6fb6",
    "gold": "#d99000",
    "green": "#2a9d64",
    "edge": "#737b86",
    "frame": "#737b86",
    "dark": "#20242a",
    "paper": "white",
}


def _nearest_point(points: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(points, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if arr.size == 0:
        return tgt, float("inf")
    distances = np.linalg.norm(arr - tgt, axis=1)
    idx = int(np.argmin(distances))
    return arr[idx], float(distances[idx])


def _harmonic_vectors_from_map(mapping: Mapping[int, np.ndarray]) -> list[tuple[int, np.ndarray]]:
    return [(int(index), np.asarray(vector, dtype=float)) for index, vector in sorted(mapping.items(), key=lambda item: int(item[0]))]


def _q_lattice_match_tolerance(qset1: np.ndarray, qset2: np.ndarray, bM1: np.ndarray, bM2: np.ndarray) -> float:
    stacked = np.vstack([np.asarray(qset1, dtype=float), np.asarray(qset2, dtype=float)])
    span = max(float(np.ptp(stacked[:, 0])), float(np.ptp(stacked[:, 1])), float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    return max(1.0e-7, 1.0e-5 * span)


def _choose_q_lattice_anchor(
    qset1: np.ndarray,
    qset2: np.ndarray,
    intra_vectors: Sequence[tuple[int, np.ndarray]],
    inter_vectors: Sequence[tuple[int, np.ndarray]],
    *,
    tol: float,
) -> np.ndarray:
    q1 = np.asarray(qset1, dtype=float)
    q2 = np.asarray(qset2, dtype=float)
    best_score: tuple[int, float, float] | None = None
    best = q1[0]
    for point in q1:
        miss_count = 0
        residual = 0.0
        for _index, vector in intra_vectors:
            _nearest, distance = _nearest_point(q1, point - vector)
            residual += distance
            miss_count += int(distance > tol)
        for _index, vector in inter_vectors:
            _nearest, distance = _nearest_point(q2, point + vector)
            residual += distance
            miss_count += int(distance > tol)
        score = (miss_count, round(residual, 12), round(float(np.linalg.norm(point)), 12))
        if best_score is None or score < best_score:
            best_score = score
            best = point
    return np.asarray(best, dtype=float)


def _draw_q_lattice_basis_arrows(ax: Any, bM1: np.ndarray, bM2: np.ndarray) -> None:
    origin = np.zeros(2, dtype=float)
    for vector, label, offset in (
        (np.asarray(bM1, dtype=float), r"$\mathbf{b}_{M1}$", np.array([0.04, -0.08])),
        (np.asarray(bM2, dtype=float), r"$\mathbf{b}_{M2}$", np.array([0.04, 0.06])),
    ):
        ax.annotate(
            "",
            xy=vector,
            xytext=origin,
            arrowprops=dict(arrowstyle="-|>", lw=1.15, color=_Q_LATTICE_PALETTE["dark"], mutation_scale=8.5),
            zorder=6,
        )
        text_xy = vector + offset * max(float(np.linalg.norm(vector)), 1.0)
        ax.text(text_xy[0], text_xy[1], label, fontsize=8.4, color=_Q_LATTICE_PALETTE["dark"], zorder=7)


def _draw_q_lattice_segments(ax: Any, points: np.ndarray, *, alpha: float = 1.0, linewidth: float = 1.05) -> None:
    from matplotlib.collections import LineCollection

    arr = np.asarray(points, dtype=float)
    distances = [
        float(np.linalg.norm(arr[i] - arr[j]))
        for i in range(len(arr))
        for j in range(i + 1, len(arr))
        if np.linalg.norm(arr[i] - arr[j]) > 1.0e-10
    ]
    if not distances:
        return
    nearest = min(distances)
    threshold = 1.04 * nearest
    segments = [(arr[i], arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr)) if np.linalg.norm(arr[i] - arr[j]) <= threshold]
    ax.add_collection(LineCollection(segments, colors=_Q_LATTICE_PALETTE["edge"], linewidths=linewidth, alpha=alpha, zorder=1))


def _q_lattice_hex_shell_from_radius(points: np.ndarray, bM1: np.ndarray, bM2: np.ndarray) -> int:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return 0
    b1 = np.asarray(bM1, dtype=float)
    b2 = np.asarray(bM2, dtype=float)
    spacing = max(float(np.linalg.norm(b1)), float(np.linalg.norm(b2)), 1.0e-12)
    radius = float(np.max(np.linalg.norm(arr, axis=1)))
    outer_vertex_allowance = 1.0 / math.sqrt(3.0)
    return max(0, int(math.ceil(max(0.0, radius / spacing - outer_vertex_allowance) - 1.0e-10)))


def _q_lattice_hex_centers(bM1: np.ndarray, bM2: np.ndarray, shell: int) -> np.ndarray:
    b1 = np.asarray(bM1, dtype=float)
    b2 = np.asarray(bM2, dtype=float)
    centers = []
    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            if max(abs(i), abs(j), abs(i + j)) <= shell:
                centers.append(i * b1 + j * b2)
    return np.asarray(centers, dtype=float)


def _draw_q_lattice_hex_cells(ax: Any, *, bM1: np.ndarray, bM2: np.ndarray, points: np.ndarray, linewidth: float = 1.0) -> np.ndarray:
    from matplotlib.collections import LineCollection

    shell = _q_lattice_hex_shell_from_radius(points, bM1, bM2)
    centers = _q_lattice_hex_centers(bM1, bM2, shell)
    try:
        base_hex = _first_bz_vertices(np.asarray(bM1, dtype=float), np.asarray(bM2, dtype=float))
    except Exception:
        return np.empty((0, 2), dtype=float)
    if centers.size == 0 or base_hex.size == 0:
        return base_hex
    segments = []
    vertices = []
    for center in centers:
        poly = base_hex + center
        vertices.append(poly)
        for idx in range(len(poly)):
            segments.append((poly[idx], poly[(idx + 1) % len(poly)]))
    ax.add_collection(
        LineCollection(
            segments,
            colors=_Q_LATTICE_PALETTE["edge"],
            linewidths=linewidth,
            alpha=0.92,
            zorder=1,
        )
    )
    return np.vstack(vertices)


def _q_lattice_nearest_spacing(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=float)
    distances = [
        float(np.linalg.norm(arr[i] - arr[j]))
        for i in range(len(arr))
        for j in range(i + 1, len(arr))
        if np.linalg.norm(arr[i] - arr[j]) > 1.0e-10
    ]
    return min(distances) if distances else 1.0


def _draw_q_lattice_harmonic_arrow(
    ax: Any,
    start: np.ndarray,
    stop: np.ndarray,
    *,
    color: str,
    spacing: float,
    curvature: float = 0.045,
) -> None:
    from matplotlib.patches import FancyArrow

    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(stop, dtype=float)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        return

    unit = direction / length
    normal = np.array([-unit[1], unit[0]], dtype=float)
    head_length = min(0.30 * spacing, 0.34 * length)
    head_base = p1 - head_length * unit
    shaft_end = p1 - 0.62 * head_length * unit
    control = 0.5 * (p0 + shaft_end) + curvature * length * normal
    t = np.linspace(0.0, 1.0, 40)[:, None]
    curve = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * control + t**2 * shaft_end
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.35, solid_capstyle="round", zorder=6)
    arrow_head = FancyArrow(
        head_base[0],
        head_base[1],
        p1[0] - head_base[0],
        p1[1] - head_base[1],
        width=0.001 * spacing,
        head_width=0.28 * spacing,
        head_length=head_length,
        length_includes_head=True,
        overhang=0.30,
        color=color,
        linewidth=0.0,
        zorder=7,
    )
    ax.add_patch(arrow_head)


def _q_lattice_inter_arrow_endpoints(
    *,
    qset1: np.ndarray,
    qset2: np.ndarray,
    sectors: Sequence[Mapping[str, Any]] | None,
    inter_sector_pairs: Sequence[Sequence[str]] | None,
    vector: np.ndarray,
    fallback_anchor: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sectors and inter_sector_pairs:
        by_name = {str(sector.get("name")): sector for sector in sectors}
        for pair in inter_sector_pairs:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                continue
            sector_from = by_name.get(str(pair[0]))
            sector_to = by_name.get(str(pair[1]))
            if sector_from is None or sector_to is None:
                continue
            q_from = sector_qset(sector_from, qset1, qset2)
            q_to = sector_qset(sector_to, qset1, qset2)
            for point in q_from:
                target, distance = _nearest_point(q_to, np.asarray(point, dtype=float) - np.asarray(vector, dtype=float))
                if distance <= tol:
                    return np.asarray(point, dtype=float), np.asarray(target, dtype=float)
    return np.asarray(fallback_anchor, dtype=float), np.asarray(fallback_anchor, dtype=float) + np.asarray(vector, dtype=float)


def save_q_lattice_harmonics_plot(
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    intra_harmonics: Mapping[int, np.ndarray],
    inter_harmonics: Mapping[int, np.ndarray],
    path: str | Path,
    title: str | None = None,
    sectors: Sequence[Mapping[str, Any]] | None = None,
    inter_sector_pairs: Sequence[Sequence[str]] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    q1 = np.asarray(Q_set1, dtype=float)
    q2 = np.asarray(Q_set2, dtype=float)
    b1 = np.asarray(bM1, dtype=float)
    b2 = np.asarray(bM2, dtype=float)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    intra_vectors = _harmonic_vectors_from_map(intra_harmonics)
    inter_vectors = _harmonic_vectors_from_map(inter_harmonics)
    tol = _q_lattice_match_tolerance(q1, q2, b1, b2)
    anchor = _choose_q_lattice_anchor(q1, q2, intra_vectors, inter_vectors, tol=tol)
    spacing = _q_lattice_nearest_spacing(np.vstack([q1, q2]))

    with kp_plot_rc_context():
        fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=KP_DPI, constrained_layout=True)
        hex_vertices = _draw_q_lattice_hex_cells(ax, bM1=b1, bM2=b2, points=np.vstack([q1, q2]), linewidth=1.0)
        ax.scatter(q1[:, 0], q1[:, 1], s=40, color=_Q_LATTICE_PALETTE["red"], edgecolor="none", linewidth=0.0, zorder=3)
        ax.scatter(q2[:, 0], q2[:, 1], s=40, color=_Q_LATTICE_PALETTE["blue"], edgecolor="none", linewidth=0.0, zorder=3)

        arrow_targets: list[np.ndarray] = [anchor]
        styles = {
            "intra": {"color": _Q_LATTICE_PALETTE["gold"], "label": "intra"},
            "inter": {"color": _Q_LATTICE_PALETTE["green"], "label": "inter"},
        }
        for _index, vector in intra_vectors:
            ideal_target = anchor - vector
            target, distance = _nearest_point(q1, ideal_target)
            if distance > tol:
                target = ideal_target
            arrow_targets.append(np.asarray(target, dtype=float))
            _draw_q_lattice_harmonic_arrow(
                ax,
                anchor,
                target,
                color=styles["intra"]["color"],
                spacing=spacing,
            )
        for _index, vector in inter_vectors:
            start, target = _q_lattice_inter_arrow_endpoints(
                qset1=q1,
                qset2=q2,
                sectors=sectors,
                inter_sector_pairs=inter_sector_pairs,
                vector=vector,
                fallback_anchor=anchor,
                tol=tol,
            )
            arrow_targets.extend([np.asarray(start, dtype=float), np.asarray(target, dtype=float)])
            _draw_q_lattice_harmonic_arrow(
                ax,
                start,
                target,
                color=styles["inter"]["color"],
                spacing=spacing,
            )

        all_points = [q1, q2, np.asarray([anchor]), np.asarray(arrow_targets, dtype=float)]
        if hex_vertices.size:
            all_points.append(hex_vertices)
        stacked = np.vstack(all_points)
        span = max(float(np.ptp(stacked[:, 0])), float(np.ptp(stacked[:, 1])), float(np.linalg.norm(b1)), float(np.linalg.norm(b2)), 1.0)
        center = np.mean(stacked, axis=0)
        half = 0.58 * span
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if title:
            ax.set_title(title, fontsize=13, pad=7, weight="semibold")
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor=_Q_LATTICE_PALETTE["red"], markeredgecolor="none", markersize=6, label=r"$Q_1$"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=_Q_LATTICE_PALETTE["blue"], markeredgecolor="none", markersize=6, label=r"$Q_2$"),
                Line2D([0], [0], color=_Q_LATTICE_PALETTE["gold"], lw=1.45, linestyle="-", label="intra"),
                Line2D([0], [0], color=_Q_LATTICE_PALETTE["green"], lw=1.45, linestyle="-", label="inter"),
            ],
            **KP_LEGEND_KWARGS,
        )
        apply_kp_axis_style(ax, box_aspect=None, font_family=kp_font_family())
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def save_bM_diagnostics(*, model_config: ConfiguredModel, output_dir: Path) -> None:
    if not model_config.bM_diagnostics:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bM_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(model_config.bM_diagnostics, handle, indent=2)


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


def _operation_physics_level(model_config: ConfiguredModel, operation: Mapping[str, Any]) -> str:
    representation_level = str(operation.get("representation_level", ""))
    if representation_level:
        return representation_level
    name = str(operation.get("name", ""))
    if model_config.symmetry_source_config.get("type") == "toy_generator":
        return "toy"
    return "physical"


def _symmetry_integrity(model_config: ConfiguredModel) -> str:
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


def _build_operation_registry(model_config: ConfiguredModel) -> list[dict[str, Any]]:
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
            _require_resolved_operation_action(operation)
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
                    for key in SOURCE_META_KEYS
                },
            }
    for row in rows_by_key.values():
        row["used_by_tags"] = sorted(row["used_by_tags"])
        row["term_tags"] = sorted(row["term_tags"])
    return list(rows_by_key.values())


def _selected_index_positions(model_config: ConfiguredModel, count: int) -> tuple[list[int], list[int]]:
    selected = list(model_config.band_indices or list(range(count)))
    fit_positions = [selected.index(int(idx)) for idx in model_config.fit_indices if int(idx) in selected]
    all_positions = list(range(len(selected)))
    holdout_positions = [idx for idx in all_positions if idx not in fit_positions]
    return fit_positions, holdout_positions


def _compute_validation_outputs(
    *,
    results: Mapping[str, Any],
    model_config: ConfiguredModel,
    moire_config: MoireConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        validations["orthogonalization"] = {"available": True, "diagnostics": _json_safe(results.get("diagnostics", {}))}
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
        state = _prepare_band_state(moire_config, model)
        h_model_list = []
        residuals = []
        for i, k in enumerate(np.asarray(moire_config.kpoints, dtype=float)):
            H_model, _w, _v, _counts, _prof = _compute_one_k(i, k, state, solve_eig=False)
            h_model_list.append(H_model)
            denom = float(np.linalg.norm(H_model)) or 1.0
            residuals.append(float(np.linalg.norm(H_model - H_model.conj().T) / denom))
        validation_hamiltonians = np.asarray(h_model_list)
        hermiticity_residuals = residuals
        return validation_hamiltonians

    if model_config.compare_to_heff:
        try:
            h_model = get_validation_hamiltonians()
            heff_all = np.load(model_config.heff_file, mmap_mode="r")
            heff_selected = _select_rows(heff_all, model_config.band_indices)
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
        except Exception as exc:
            validations["matrix_residuals"] = {"available": False, "reason": str(exc)}
    else:
        validations["matrix_residuals"] = {"available": False, "reason": "compare_to_heff_disabled"}
    validations["block_residuals"] = {
        "available": False,
        "reason": "block residual validation is not implemented in the current configured pipeline",
    }

    operation_registry = _build_operation_registry(model_config)
    try:
        state = _prepare_band_state(moire_config, model)
        h_model = get_validation_hamiltonians()
        kpoints = np.asarray(moire_config.kpoints, dtype=float)
        rows = []
        for operation in operation_registry:
            k_map = operation.get("k_map", {})
            canonical = str(operation.get("canonical_operation", ""))
            antiunitary = bool(operation.get("antiunitary", False))
            try:
                D = state.symmetry_gen.get_operator(canonical, None)
            except Exception:
                continue
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
    except Exception as exc:
        validations["symmetry_residuals"] = {"available": False, "reason": str(exc)}

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

    diagnostics = _json_safe(results.get("diagnostics", {}))
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


def _build_run_summary(
    *,
    model_config: ConfiguredModel,
    results: Mapping[str, Any],
    operation_registry: Sequence[Mapping[str, Any]],
    validation_summary: Mapping[str, Any],
    active_terms_hash: str | None = None,
    output_profile: str = "release",
) -> dict[str, Any]:
    uses_toy = model_config.symmetry_source_config.get("type") == "toy_generator"
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
        "all_band_plot": "band_comparison_all.pdf" if results.get("all_band_plot") else None,
        "q_lattice_plot": "q_lattice_harmonics.pdf" if results.get("q_lattice_plot") else None,
        "null_channel_filter": _json_safe(results.get("null_channel_filter")),
        "coefficient_pruning": _json_safe(results.get("coefficient_pruning")),
        "band_refinement": _json_safe(results.get("band_refinement", {"enabled": False})),
        "auto_model_selection": _json_safe(results.get("auto_model_selection", {"enabled": False})),
        "orbital_counts": _json_safe(model_config.orbital_count_metadata),
        "fit_selection": _json_safe(model_config.fit_selection_metadata),
        "term_template_profile": _json_safe(model_config.term_template_metadata),
        "operations": list(operation_registry),
    }
    if output_profile == "debug":
        summary["operation_registry_file"] = "diagnostics/operation_registry.json"
    if active_terms_hash is not None:
        summary["active_terms_hash"] = active_terms_hash
        summary["active_terms_hash_file"] = "active_terms.sha256"
    return summary


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
    model_config: ConfiguredModel,
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
    model_config: ConfiguredModel,
    moire_config: MoireConfig,
) -> dict[str, Any]:
    refine_cfg = dict(model_config.band_refinement_config or {})
    fit_mode = str(model_config.fit_selection_metadata.get("mode", "")).strip().lower()
    if fit_mode != "auto_low_energy" and str(refine_cfg.get("mode", "")).strip().lower() != "auto_low_energy":
        return {"enabled": False}
    if moire_config.kpoints is None:
        return {"enabled": False, "reason": "missing_kpoints"}
    model = results.get("model")
    if model is None:
        return {"enabled": False, "reason": "missing_model"}

    heff_all = _select_rows(np.load(model_config.heff_file, mmap_mode="r"), model_config.band_indices)
    h_model = _model_hamiltonians_for_kpoints(moire_config, model, np.asarray(moire_config.kpoints, dtype=float))
    heff_eig, heff_vec = np.linalg.eigh(heff_all)
    model_eig, model_vec = np.linalg.eigh(h_model)
    target_bands = str(refine_cfg.get("target_bands", model_config.raw.get("model", {}).get("target_bands", "top"))).strip().lower()
    auto_windows = refine_cfg.get("auto_windows")
    if not isinstance(auto_windows, Mapping):
        auto_windows = _auto_low_energy_windows(
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
        primary_slice = _auto_low_energy_band_slice(heff_all.shape[-1], max(1, int(sum(model_config.n_orb))), target_bands)
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
        band_metrics = _band_refinement_metrics(
            h_model,
            heff_eig,
            heff_all,
            band_slice=band_slice,
            align=str(refine_cfg.get("align", target_bands)),
        )
        target_basis = heff_vec[:, :, band_slice[0] : band_slice[1]]
        model_basis = model_vec[:, :, band_slice[0] : band_slice[1]]
        subspace_metrics = _subspace_overlap_metrics(model_basis, target_basis)
        _low_vec, low_matrix_metrics = _low_subspace_matrix_residual(
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

    primary = candidate["windows"][0]
    save_band_comparison_plot(
        model_eig,
        heff_eig,
        output_dir / "band_comparison_top_primary.pdf",
        band_slice=primary["band_slice"],
        plot_config={**model_config.band_plot_config, "band_slice": primary["band_slice"]},
        title="Auto low-energy primary window",
    )
    weighted_window = next((item for item in candidate["windows"] if item.get("role") == "weighted_fit"), None)
    if weighted_window is not None:
        save_band_comparison_plot(
            model_eig,
            heff_eig,
            output_dir / "band_comparison_weighted_fit.pdf",
            band_slice=weighted_window["band_slice"],
            plot_config={**model_config.band_plot_config, "band_slice": weighted_window["band_slice"]},
            title="Auto low-energy weighted fit window",
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
        use_color = _terminal_color_enabled(out)
        style_code = {
            "start": "36",
            "done": "1;32",
            "section": "1;35",
            "fit_start": "1;33",
            "fit_done": "1;32",
            "path": "1;35",
            "warning": "1;33",
            "error": "1;31",
        }.get(str(style or ""))
        prefix = _terminal_color("[kp model]", "2;36", enabled=use_color)
        body = _terminal_color(str(message), style_code, enabled=use_color) if style_code else str(message)
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
    return [
        (tag, int(values["terms"]), int(values["active_terms"]), int(values["variables"]))
        for tag, values in sorted(rows.items(), key=lambda item: item[0].lower())
    ]


def _print_term_progress_summary(model: Any, *, enabled: bool, stream: Any | None = None) -> None:
    rows = _term_progress_summary_rows(model)
    if not rows:
        _progress_line("term summary before fitting: no terms", enabled=enabled, style="section", stream=stream)
        return
    _progress_line("term summary before fitting:", enabled=enabled, style="section", stream=stream)
    for tag, terms, active_terms, variables in rows:
        _progress_line(
            f"  {tag}: terms={terms}, fit_components={variables}",
            enabled=enabled,
            style="fit_start",
            stream=stream,
        )


def _prune_small_coefficients(model: Any, threshold: float) -> dict[str, Any]:
    threshold = float(threshold)
    if threshold <= 0.0:
        return {"enabled": False, "threshold": threshold, "dropped": 0, "kept": None}
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


def _refinement_tag_alias(tag: str) -> str:
    aliases = {
        "kinetic": "Kinect",
        "onsite": "Onsite",
        "intralayer": "intra",
        "interlayer": "inter",
        "tunneling": "inter",
        "moire": "intra",
    }
    return aliases.get(str(tag), str(tag))


def _refinement_band_slice(config: Mapping[str, Any], model_config: ConfiguredModel, dim: int) -> tuple[int, int]:
    if config.get("band_slice") is not None:
        raw = list(config["band_slice"])
        if len(raw) != 2:
            raise ValueError(f"fit.refine_bands.band_slice must be [start, stop], got {raw!r}")
        start, stop = int(raw[0]), int(raw[1])
    elif config.get("top_bands") is not None:
        count = int(config["top_bands"])
        start, stop = max(0, int(dim) - count), int(dim)
    elif model_config.band_plot_config.get("top_bands") is not None:
        count = int(model_config.band_plot_config["top_bands"])
        start, stop = max(0, int(dim) - count), int(dim)
    elif model_config.band_slice is not None:
        start, stop = int(model_config.band_slice[0]), int(model_config.band_slice[1])
    else:
        start, stop = 0, int(dim)
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid band refinement slice [{start}, {stop}] for dimension {dim}")
    return start, stop


def _model_hamiltonians_for_kpoints(moire_config: MoireConfig, model: Any, kpoints: np.ndarray) -> np.ndarray:
    state = _prepare_band_state(moire_config, model)
    hamiltonians = []
    for index, kval in enumerate(np.asarray(kpoints, dtype=float)):
        h_model, _w, _v, _counts, _prof = _compute_one_k(index, kval, state, solve_eig=False)
        hamiltonians.append(0.5 * (h_model + h_model.conj().T))
    return np.asarray(hamiltonians)


def _term_component_hamiltonians_for_kpoints(
    moire_config: MoireConfig,
    state: Any,
    term: Any,
    component: str,
    kpoints: np.ndarray,
) -> np.ndarray:
    """Return the linear Hamiltonian response for one coefficient component."""
    remove = np.asarray(getattr(state, "remove", np.array([], dtype=int)), dtype=int)
    if remove.size:
        raise NotImplementedError("direct term response is only linear when no Schur remove subspace is active")
    keep = np.asarray(getattr(state, "keep", np.array([], dtype=int)), dtype=int)
    dim_full = int(getattr(state, "dim_full"))
    real = 1.0 if component == "real" else 0.0
    imag = 1.0 if component == "imag" else 0.0
    if component not in {"real", "imag"}:
        raise ValueError(f"unknown coefficient component {component!r}")
    hamiltonians = []
    for kval in np.asarray(kpoints, dtype=float):
        h_full = np.zeros((dim_full, dim_full), dtype=np.complex128)
        ContinuumModelBuilder.add_symmetrized_term_to_matrix_static(
            h_full,
            getattr(term, "Y_basis"),
            kval,
            getattr(term, "symmetry_ops", []),
            real,
            imag,
            symmetry_gen=getattr(state, "symmetry_gen", None),
            term=term,
        )
        h = h_full[np.ix_(keep, keep)] if keep.size != dim_full else h_full
        hamiltonians.append(0.5 * (h + h.conj().T))
    return np.asarray(hamiltonians)


def _term_response_pair_hamiltonians_for_kpoints(
    state: Any,
    term: Any,
    kpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real/imag coefficient responses while symmetrizing each term once."""
    remove = np.asarray(getattr(state, "remove", np.array([], dtype=int)), dtype=int)
    if remove.size:
        raise NotImplementedError("direct term response is only linear when no Schur remove subspace is active")
    keep = np.asarray(getattr(state, "keep", np.array([], dtype=int)), dtype=int)
    dim_full = int(getattr(state, "dim_full"))
    real_list: list[np.ndarray] = []
    imag_list: list[np.ndarray] = []
    y_basis = getattr(term, "Y_basis")
    sym_ops = getattr(term, "symmetry_ops", [])
    symmetry_gen = getattr(state, "symmetry_gen", None)
    sparse_available = (
        ContinuumModelBuilder._SYMM_USE_SPARSE_BASIS
        and hasattr(y_basis, "eval_sparse")
        and symmetry_gen is not None
    )
    k_array = np.asarray(kpoints, dtype=float)
    if sparse_available and k_array.size and (
        not hasattr(term, "_moire_needs_hermitize_real")
        or not hasattr(term, "_moire_needs_hermitize_imag")
    ):
        ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
            y_basis,
            k_array[0],
            sym_ops,
            symmetry_gen=symmetry_gen,
            term=term,
            use_cache=False,
        )
    needs_herm_real = bool(getattr(term, "_moire_needs_hermitize_real", False))
    needs_herm_imag = bool(getattr(term, "_moire_needs_hermitize_imag", False))
    has_herm_flags = hasattr(term, "_moire_needs_hermitize_real") and hasattr(term, "_moire_needs_hermitize_imag")
    inconsistent = bool(getattr(term, "_moire_hermitize_flags_inconsistent", False))
    use_sparse_direct = (
        sparse_available
        and has_herm_flags
        and not inconsistent
    )
    for kval in k_array:
        if use_sparse_direct:
            h_real = np.zeros((dim_full, dim_full), dtype=np.complex128)
            h_imag = np.zeros((dim_full, dim_full), dtype=np.complex128)
            orbit_actions = (
                ContinuumModelBuilder._get_symmetry_orbit_actions_cached(kval, sym_ops, symmetry_gen)
                if sym_ops
                else [(kval, None, tuple(), False)]
            )
            for kk, action, op_seq_applied, is_anti in orbit_actions:
                if op_seq_applied != tuple():
                    use_sparse_direct = False
                    break
                use_add_at = not getattr(y_basis, "_moire_sparse_unique", True)
                rows, cols, vals0 = y_basis.eval_sparse(kk)
                if action is not None:
                    _perm, inv_perm, vals, inv_vals, is_anti_total = action
                    rr = inv_perm[rows]
                    cc = inv_perm[cols]
                    vv = np.conjugate(vals0) if is_anti_total else vals0
                    vv = vv * vals[rr] * inv_vals[cc]
                else:
                    rr, cc, vv = rows, cols, vals0
                    is_anti_total = bool(is_anti)
                support = ContinuumModelBuilder.term_output_support_mask(term, rr, cc, dim_full)
                if not np.all(support):
                    rr = rr[support]
                    cc = cc[support]
                    vv = vv[support]
                sign = -1.0 if is_anti_total else 1.0
                if use_add_at:
                    np.add.at(h_real, (rr, cc), vv)
                    np.add.at(h_imag, (rr, cc), 1j * sign * vv)
                else:
                    h_real[rr, cc] += vv
                    h_imag[rr, cc] += 1j * sign * vv
                if needs_herm_real:
                    vv_h = np.conjugate(vv)
                    if use_add_at:
                        np.add.at(h_real, (cc, rr), vv_h)
                    else:
                        h_real[cc, rr] += vv_h
                if needs_herm_imag:
                    vv_h = np.conjugate(vv)
                    if use_add_at:
                        np.add.at(h_imag, (cc, rr), -1j * sign * vv_h)
                    else:
                        h_imag[cc, rr] += -1j * sign * vv_h
            else:
                if keep.size != dim_full:
                    h_real = h_real[np.ix_(keep, keep)]
                    h_imag = h_imag[np.ix_(keep, keep)]
                real_list.append(0.5 * (h_real + h_real.conj().T))
                imag_list.append(0.5 * (h_imag + h_imag.conj().T))
                continue
        h_real, h_imag = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static(
            y_basis,
            kval,
            sym_ops,
            symmetry_gen=symmetry_gen,
            term=term,
            use_cache=False,
        )
        if keep.size != dim_full:
            h_real = h_real[np.ix_(keep, keep)]
            h_imag = h_imag[np.ix_(keep, keep)]
        real_list.append(0.5 * (h_real + h_real.conj().T))
        imag_list.append(0.5 * (h_imag + h_imag.conj().T))
    return np.asarray(real_list), np.asarray(imag_list)


def _select_band_refinement_variables(model: Any, config: Mapping[str, Any]) -> list[tuple[Any, str]]:
    raw_tags = config.get("variable_tags", config.get("tags", ["Kinect", "Onsite", "inter"]))
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags = {_refinement_tag_alias(str(tag)) for tag in raw_tags}
    raw_components = config.get("components", ["real"])
    if isinstance(raw_components, str):
        raw_components = [raw_components]
    components = {str(item).strip().lower() for item in raw_components}
    if not components <= {"real", "imag"}:
        raise ValueError(f"fit.refine_bands.components must contain only 'real'/'imag', got {sorted(components)!r}")
    variables: list[tuple[Any, str]] = []
    for term in getattr(model, "terms", {}).values():
        if not getattr(term, "active", True):
            continue
        if str(getattr(term, "tag", "")) not in tags:
            continue
        if "real" in components:
            variables.append((term, "real"))
        if "imag" in components:
            variables.append((term, "imag"))
    return variables


def _term_component_value(term: Any, component: str) -> float:
    if component == "real":
        return float(getattr(term, "r_value_real", 0.0))
    if component == "imag":
        return float(getattr(term, "r_value_imag", 0.0))
    raise ValueError(f"unknown coefficient component {component!r}")


def _set_term_component_value(term: Any, component: str, value: float) -> None:
    if component == "real":
        term.r_value_real = float(value)
        return
    if component == "imag":
        term.r_value_imag = float(value)
        return
    raise ValueError(f"unknown coefficient component {component!r}")


def _band_refinement_metrics(
    hamiltonians: np.ndarray,
    target_eig: np.ndarray,
    target_h: np.ndarray,
    *,
    band_slice: tuple[int, int],
    align: str,
) -> dict[str, float]:
    eig = np.linalg.eigvalsh(hamiltonians)
    start, stop = band_slice
    model_sel = eig[:, start:stop]
    target_sel = target_eig[:, start:stop]
    shift = 0.0
    if align == "top":
        shift = float(target_sel[0, -1] - model_sel[0, -1])
        model_sel = model_sel + shift
    elif align == "bottom":
        shift = float(target_sel[0, 0] - model_sel[0, 0])
        model_sel = model_sel + shift
    elif align not in {"none", ""}:
        raise ValueError(f"fit.refine_bands.align currently supports 'top', 'bottom', or 'none', got {align!r}")
    diff = model_sel - target_sel
    all_diff = eig - target_eig
    matrix_diff = hamiltonians - target_h
    return {
        "top_band_rms_mev": float(np.sqrt(np.mean(diff**2)) * 1000.0),
        "top_band_max_mev": float(np.max(np.abs(diff)) * 1000.0),
        "all_band_rms_mev": float(np.sqrt(np.mean(all_diff**2)) * 1000.0),
        "all_band_max_mev": float(np.max(np.abs(all_diff)) * 1000.0),
        "matrix_element_rms_mev": float(np.sqrt(np.mean(np.abs(matrix_diff) ** 2)) * 1000.0),
        "matrix_element_max_mev": float(np.max(np.abs(matrix_diff)) * 1000.0),
        "alignment_shift_mev": float(shift * 1000.0),
    }


def _band_refinement_band_residual(
    model_selected: np.ndarray,
    target_selected: np.ndarray,
    *,
    band_sigma: float,
    normalize: bool,
) -> np.ndarray:
    residual = (np.asarray(model_selected, dtype=float) - np.asarray(target_selected, dtype=float)) / float(band_sigma)
    if normalize and residual.size:
        residual = residual / np.sqrt(float(residual.size))
    return residual


def _band_refinement_eigenvalue_jacobian(
    eigvecs: np.ndarray,
    basis: np.ndarray,
    *,
    band_slice: tuple[int, int],
    align: str,
    band_sigma: float,
    normalize: bool,
    edge_band_weights: np.ndarray | None = None,
    band_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic Jacobian of the selected eigenvalue residual.

    For a real coefficient y_j and Hermitian response B_j(k),
    d lambda_n(k) / d y_j = <u_n(k)|B_j(k)|u_n(k)>.
    """
    vec = np.asarray(eigvecs, dtype=np.complex128)
    responses = np.asarray(basis, dtype=np.complex128)
    if vec.ndim != 3:
        raise ValueError(f"eigvecs must have shape (n_k, dim, dim), got {vec.shape}")
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    if responses.shape[1:] != (vec.shape[0], vec.shape[1], vec.shape[1]):
        raise ValueError(
            "basis/eigvec shape mismatch: "
            f"basis={responses.shape}, eigvecs={vec.shape}"
        )
    start, stop = band_slice
    selected_vec = vec[:, :, start:stop]
    selected = np.einsum("kdn,jkde,ken->knj", selected_vec.conj(), responses, selected_vec, optimize=True).real
    if align == "top":
        selected = selected - selected[0:1, -1:, :]
    elif align == "bottom":
        selected = selected - selected[0:1, 0:1, :]
    elif align not in {"none", ""}:
        raise ValueError(f"fit.refine_bands.align currently supports 'top', 'bottom', or 'none', got {align!r}")
    selected = selected / float(band_sigma)
    if normalize and selected.size:
        selected = selected / np.sqrt(float(selected.shape[0] * selected.shape[1]))
    if edge_band_weights is not None:
        weights = np.asarray(edge_band_weights, dtype=float)
        if weights.shape != (selected.shape[1],):
            raise ValueError(f"edge_band_weights must have shape ({selected.shape[1]},), got {weights.shape}")
        selected = selected * np.sqrt(weights.reshape(1, -1, 1))
    if band_weights is not None:
        weights = np.asarray(band_weights, dtype=float)
        if weights.shape != selected.shape[:2]:
            raise ValueError(f"band_weights must have shape {selected.shape[:2]}, got {weights.shape}")
        selected = selected * np.sqrt(weights[:, :, None])
    return selected.reshape(selected.shape[0] * selected.shape[1], selected.shape[2])


@dataclass
class _BandRefinementOptimizeResult:
    x: np.ndarray
    cost: float
    nfev: int
    njev: int
    status: int
    message: str


def _solve_band_refinement_gauss_newton(
    residual_fn,
    jacobian_fn,
    y0: np.ndarray,
    *,
    max_nfev: int,
    xtol: float,
    ftol: float,
    gtol: float,
    initial_damping: float = 1.0e-6,
    max_line_search_steps: int = 8,
) -> _BandRefinementOptimizeResult:
    """Small dense damped Gauss-Newton solver for refinement coefficients."""
    y = np.asarray(y0, dtype=float).copy()
    max_eval = max(1, int(max_nfev))
    damping = max(float(initial_damping), 0.0)
    line_steps = max(1, int(max_line_search_steps))
    residual = np.asarray(residual_fn(y), dtype=float)
    cost = 0.5 * float(residual @ residual)
    nfev = 1
    njev = 0
    message = "max_nfev reached"
    status = 0
    for _iteration in range(max_eval):
        jac = np.asarray(jacobian_fn(y), dtype=float)
        njev += 1
        gradient = jac.T @ residual
        grad_norm = float(np.linalg.norm(gradient, ord=np.inf)) if gradient.size else 0.0
        if grad_norm <= float(gtol):
            status = 1
            message = "gtol satisfied"
            break
        normal = jac.T @ jac
        diag = np.maximum(np.diag(normal), 1.0)
        local_damping = damping
        accepted = False
        step = np.zeros_like(y)
        trial_cost = cost
        trial_residual = residual
        for _damping_round in range(8):
            lhs = normal + local_damping * np.diag(diag)
            try:
                step = scipy.linalg.solve(lhs, -gradient, assume_a="pos", check_finite=False)
            except (scipy.linalg.LinAlgError, ValueError):
                step = scipy.linalg.lstsq(lhs, -gradient, check_finite=False)[0]
            if not np.all(np.isfinite(step)):
                local_damping = max(local_damping * 10.0, 1.0e-12)
                continue
            for line_index in range(line_steps):
                alpha = 0.5**line_index
                trial_y = y + alpha * step
                trial_residual = np.asarray(residual_fn(trial_y), dtype=float)
                nfev += 1
                trial_cost = 0.5 * float(trial_residual @ trial_residual)
                if trial_cost < cost:
                    y = trial_y
                    residual = trial_residual
                    accepted = True
                    break
                if nfev >= max_eval:
                    break
            if accepted or nfev >= max_eval:
                break
            local_damping = max(local_damping * 10.0, 1.0e-12)
        improvement = cost - trial_cost
        if accepted:
            step_norm = float(np.linalg.norm(y - (y - alpha * step)))
            scale_norm = float(np.linalg.norm(y) + float(xtol))
            prev_cost = cost
            cost = trial_cost
            damping = max(local_damping * 0.3, 1.0e-14)
            if improvement <= float(ftol) * max(prev_cost, 1.0):
                status = 2
                message = "ftol satisfied"
                break
            if step_norm <= float(xtol) * max(scale_norm, 1.0):
                status = 3
                message = "xtol satisfied"
                break
        else:
            damping = max(local_damping * 10.0, 1.0e-12)
            if nfev >= max_eval:
                break
            status = 4
            message = "line search failed to reduce cost"
            break
        if nfev >= max_eval:
            break
    return _BandRefinementOptimizeResult(
        x=y,
        cost=cost,
        nfev=int(nfev),
        njev=int(njev),
        status=int(status),
        message=message,
    )


def _band_refinement_global_matrix_jacobian(
    basis: np.ndarray,
    *,
    matrix_weight: float,
    matrix_sigma: float,
) -> np.ndarray:
    responses = np.asarray(basis, dtype=np.complex128)
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    scale = np.sqrt(float(matrix_weight)) / float(matrix_sigma)
    flat = responses.reshape(responses.shape[0], -1).T
    return scale * np.vstack([flat.real, flat.imag])


def _band_refinement_global_matrix_jacobian_sparse(
    basis: np.ndarray,
    *,
    matrix_weight: float,
    matrix_sigma: float,
) -> scipy.sparse.csr_matrix:
    responses = np.asarray(basis, dtype=np.complex128)
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    scale = np.sqrt(float(matrix_weight)) / float(matrix_sigma)
    flat = responses.reshape(responses.shape[0], -1).T
    return scale * scipy.sparse.vstack(
        [
            scipy.sparse.csr_matrix(flat.real),
            scipy.sparse.csr_matrix(flat.imag),
        ],
        format="csr",
    )


def _band_refinement_reduced_global_matrix_loss(
    basis: np.ndarray,
    target_delta: np.ndarray,
    *,
    matrix_weight: float,
    matrix_sigma: float,
    rcond: float = 1.0e-12,
) -> dict[str, Any]:
    """Compress the linear full-matrix loss to coefficient space.

    The legacy matrix residual is A d + c, where d = y - y0.  Since A is
    linear, ||A d + c||^2 differs from ||L (d - d*)||^2 only by a constant
    when L.T @ L = A.T @ A and d* is a least-squares minimizer.  The optimizer
    sees the same gradient/curvature but with at most n_variables rows.
    """
    responses = np.asarray(basis, dtype=np.complex128)
    delta = np.asarray(target_delta, dtype=np.complex128)
    if responses.ndim != 4:
        raise ValueError(f"basis must have shape (n_variables, n_k, dim, dim), got {responses.shape}")
    if delta.shape != responses.shape[1:]:
        raise ValueError(f"target_delta must have shape {responses.shape[1:]}, got {delta.shape}")
    if matrix_weight <= 0.0:
        raise ValueError(f"matrix_weight must be positive, got {matrix_weight}")
    if matrix_sigma <= 0.0:
        raise ValueError(f"matrix_sigma must be positive, got {matrix_sigma}")
    if rcond < 0.0:
        raise ValueError(f"rcond must be non-negative, got {rcond}")

    flat = responses.reshape(responses.shape[0], -1)
    target = delta.reshape(-1)
    scale_sq = float(matrix_weight) / (float(matrix_sigma) ** 2)
    gram = scale_sq * np.real(flat.conj() @ flat.T)
    gram = 0.5 * (gram + gram.T)
    linear = scale_sq * np.real(flat.conj() @ target)
    evals, evecs = scipy.linalg.eigh(gram)
    if evals.size == 0:
        return {
            "jacobian": np.zeros((0, responses.shape[0]), dtype=float),
            "center_delta": np.zeros(responses.shape[0], dtype=float),
            "rank": 0,
            "condition_number": None,
            "source_rows": int(2 * target.size),
            "compressed_rows": 0,
            "rcond": float(rcond),
        }
    max_eval = float(np.max(evals))
    cutoff = max(float(rcond) * max(max_eval, 1.0), np.finfo(float).eps * max(responses.shape[0], 1) * max_eval)
    keep = evals > cutoff
    if not np.any(keep):
        return {
            "jacobian": np.zeros((0, responses.shape[0]), dtype=float),
            "center_delta": np.zeros(responses.shape[0], dtype=float),
            "rank": 0,
            "condition_number": None,
            "source_rows": int(2 * target.size),
            "compressed_rows": 0,
            "rcond": float(rcond),
        }
    kept_evals = np.asarray(evals[keep], dtype=float)
    kept_vecs = np.asarray(evecs[:, keep], dtype=float)
    jac = np.sqrt(kept_evals)[:, None] * kept_vecs.T
    center_delta = -kept_vecs @ ((kept_vecs.T @ linear) / kept_evals)
    return {
        "jacobian": jac,
        "center_delta": center_delta,
        "rank": int(kept_evals.size),
        "condition_number": float(np.sqrt(float(kept_evals[-1] / kept_evals[0]))) if kept_evals[0] > 0.0 else None,
        "source_rows": int(2 * target.size),
        "compressed_rows": int(kept_evals.size),
        "rcond": float(rcond),
    }


def _edge_weighted_band_weights(
    n_bands: int,
    *,
    primary_bands: int,
    target_bands: str,
    decay: float = 0.65,
    floor: float = 0.15,
    normalize_mean: bool = True,
) -> np.ndarray:
    """Return deterministic edge-decaying weights for a top/bottom low-energy band window."""
    n = int(n_bands)
    primary = int(primary_bands)
    if n <= 0:
        raise ValueError(f"n_bands must be positive, got {n_bands}")
    if primary <= 0:
        raise ValueError(f"primary_bands must be positive, got {primary_bands}")
    if primary > n:
        raise ValueError(f"primary_bands={primary} exceeds n_bands={n}")
    decay_value = float(decay)
    floor_value = float(floor)
    if not (0.0 < decay_value <= 1.0):
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    if not (0.0 <= floor_value <= 1.0):
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    target = str(target_bands or "top").strip().lower()
    if target not in {"top", "bottom"}:
        raise ValueError("model.target_bands must be 'top' or 'bottom'")
    indices = np.arange(n, dtype=int)
    if target == "top":
        distance = np.maximum(0, (n - 1 - indices) - (primary - 1))
    else:
        distance = np.maximum(0, indices - (primary - 1))
    weights = floor_value + (1.0 - floor_value) * np.power(decay_value, distance.astype(float))
    if normalize_mean:
        mean = float(np.mean(weights))
        if mean <= 0.0 or not np.isfinite(mean):
            raise ValueError("weighted band loss produced invalid zero/NaN mean weight")
        weights = weights / mean
    return weights.astype(float)


def _boundary_gap_mev_by_k(eigvals: np.ndarray, band_slice: Sequence[int]) -> np.ndarray:
    eig = np.asarray(eigvals, dtype=float)
    if eig.ndim != 2:
        raise ValueError(f"boundary gap expects eigvals shape (Nk,dim), got {eig.shape}")
    if len(band_slice) != 2:
        raise ValueError(f"band_slice must be [start, stop], got {band_slice!r}")
    start, stop = int(band_slice[0]), int(band_slice[1])
    dim = eig.shape[1]
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid band_slice {band_slice!r} for dim={dim}")
    gaps: list[np.ndarray] = []
    if start > 0:
        gaps.append(np.abs(eig[:, start] - eig[:, start - 1]))
    if stop < dim:
        gaps.append(np.abs(eig[:, stop] - eig[:, stop - 1]))
    if not gaps:
        return np.full(eig.shape[0], np.inf, dtype=float)
    return np.min(np.stack(gaps, axis=0), axis=0) * 1000.0


def _gap_weights_from_eigvals(
    eigvals: np.ndarray,
    band_slice: Sequence[int],
    *,
    gap_tolerance_mev: float,
) -> np.ndarray:
    gap_mev = _boundary_gap_mev_by_k(eigvals, band_slice)
    if gap_tolerance_mev <= 0.0:
        return np.ones_like(gap_mev, dtype=float)
    weights = np.clip(gap_mev / float(gap_tolerance_mev), 0.0, 1.0)
    weights[~np.isfinite(weights)] = 1.0
    return weights


def _subspace_overlap_metrics(
    model_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    gap_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    model_u = np.asarray(model_basis, dtype=np.complex128)
    target_u = np.asarray(target_basis, dtype=np.complex128)
    if model_u.shape != target_u.shape or model_u.ndim != 3:
        raise ValueError(
            "subspace overlap expects model/target basis arrays with matching "
            f"shape (Nk,dim,N), got {model_u.shape} and {target_u.shape}"
        )
    n_k, _dim, n_state = model_u.shape
    if n_state <= 0:
        raise ValueError("subspace overlap requires at least one state")
    overlaps: list[float] = []
    leakages: list[float] = []
    min_sv: list[float] = []
    for idx in range(n_k):
        singular_values = np.linalg.svd(target_u[idx].conj().T @ model_u[idx], compute_uv=False)
        clipped = np.clip(singular_values, 0.0, 1.0)
        overlap = float(np.sum(clipped**2) / n_state)
        leakage = float(max(0.0, 1.0 - overlap))
        overlaps.append(overlap)
        leakages.append(leakage)
        min_sv.append(float(np.min(singular_values)))
    overlap_arr = np.asarray(overlaps, dtype=float)
    leakage_arr = np.asarray(leakages, dtype=float)
    min_sv_arr = np.asarray(min_sv, dtype=float)
    report = {
        "n_k": int(n_k),
        "n_state": int(n_state),
        "mean_overlap": float(np.mean(overlap_arr)),
        "min_overlap": float(np.min(overlap_arr)),
        "p05_overlap": float(np.percentile(overlap_arr, 5)),
        "mean_leakage": float(np.mean(leakage_arr)),
        "max_leakage": float(np.max(leakage_arr)),
        "p95_leakage": float(np.percentile(leakage_arr, 95)),
        "mean_min_singular_value": float(np.mean(min_sv_arr)),
        "min_singular_value": float(np.min(min_sv_arr)),
    }
    if gap_weights is not None:
        weights = np.asarray(gap_weights, dtype=float)
        if weights.shape != (n_k,):
            raise ValueError(f"gap_weights must have shape ({n_k},), got {weights.shape}")
        report["gap_weight_min"] = float(np.min(weights))
        report["gap_weight_mean"] = float(np.mean(weights))
    return report


def _principal_angle_subspace_residual(
    model_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    weight: float,
    normalize: bool,
    gap_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if weight < 0.0:
        raise ValueError("fit.refine_bands.subspace_loss.weight must be non-negative")
    model_u = np.asarray(model_basis, dtype=np.complex128)
    target_u = np.asarray(target_basis, dtype=np.complex128)
    if model_u.shape != target_u.shape or model_u.ndim != 3:
        raise ValueError(
            "subspace loss expects model/target basis arrays with matching "
            f"shape (Nk,dim,N), got {model_u.shape} and {target_u.shape}"
        )
    n_k = int(model_u.shape[0])
    if gap_weights is None:
        weights = np.ones(n_k, dtype=float)
    else:
        weights = np.asarray(gap_weights, dtype=float)
        if weights.shape != (n_k,):
            raise ValueError(f"gap_weights must have shape ({n_k},), got {weights.shape}")
    parts: list[np.ndarray] = []
    for idx in range(n_k):
        singular_values = np.linalg.svd(target_u[idx].conj().T @ model_u[idx], compute_uv=False)
        principal = np.sqrt(np.maximum(0.0, 1.0 - np.clip(singular_values, 0.0, 1.0) ** 2))
        parts.append(np.sqrt(weight * max(float(weights[idx]), 0.0)) * principal)
    residual = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
    if normalize and residual.size:
        residual = residual / np.sqrt(float(residual.size))
    report = _subspace_overlap_metrics(model_u, target_u, gap_weights=weights)
    report.update(
        {
            "enabled": True,
            "mode": "principal_angles",
            "weight": float(weight),
            "normalize": bool(normalize),
            "loss_norm_sq": float(np.sum(residual**2)),
        }
    )
    return residual.astype(float), report


def _low_subspace_matrix_residual(
    matrix_delta: np.ndarray,
    target_basis: np.ndarray,
    *,
    weight: float,
    sigma_mev: float,
    normalize: bool,
    gap_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta = np.asarray(matrix_delta, dtype=np.complex128)
    basis = np.asarray(target_basis, dtype=np.complex128)
    if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
        raise ValueError(f"low-subspace matrix loss expects delta shape (Nk,dim,dim), got {delta.shape}")
    if basis.ndim != 3 or basis.shape[0] != delta.shape[0] or basis.shape[1] != delta.shape[1]:
        raise ValueError(
            "low-subspace matrix loss expects target basis shape (Nk,dim,N), "
            f"got delta={delta.shape}, basis={basis.shape}"
        )
    if weight < 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.sigma_mev must be positive")
    n_k = int(delta.shape[0])
    if gap_weights is None:
        weights = np.ones(n_k, dtype=float)
    else:
        weights = np.asarray(gap_weights, dtype=float)
        if weights.shape != (n_k,):
            raise ValueError(f"gap_weights must have shape ({n_k},), got {weights.shape}")
    sigma = float(sigma_mev) * 1.0e-3
    values: list[np.ndarray] = []
    scaled: list[np.ndarray] = []
    for idx in range(n_k):
        projected = basis[idx].conj().T @ delta[idx] @ basis[idx]
        values.append(projected.reshape(-1))
        scaled.append(np.sqrt(weight * max(float(weights[idx]), 0.0)) * projected.reshape(-1) / sigma)
    flat_values = np.concatenate(values) if values else np.zeros(0, dtype=np.complex128)
    flat_scaled = np.concatenate(scaled) if scaled else np.zeros(0, dtype=np.complex128)
    residual = np.concatenate([flat_scaled.real, flat_scaled.imag])
    if normalize and residual.size:
        residual = residual / np.sqrt(float(flat_values.size))
    abs_values = np.abs(flat_values)
    report = {
        "enabled": True,
        "mode": "target_subspace_projected",
        "weight": float(weight),
        "sigma_mev": float(sigma_mev),
        "normalize": bool(normalize),
        "n_k": int(n_k),
        "n_state": int(basis.shape[2]),
        "rms_mev": float(np.sqrt(np.mean(abs_values**2)) * 1000.0) if abs_values.size else None,
        "max_mev": float(np.max(abs_values) * 1000.0) if abs_values.size else None,
        "loss_norm_sq": float(np.sum(residual**2)),
    }
    return residual.astype(float), report


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
        window = _shell_band_window_from_target_eig(
            target_eig,
            target_bands=target_key,
            window_config=effective_window,
        )
        n_bands = int(window["n_bands"])
        band_slice = tuple(int(value) for value in window["band_slice"])
        band_metrics = _band_refinement_metrics(
            model_shell,
            target_eig,
            target_shell,
            band_slice=(int(band_slice[0]), int(band_slice[1])),
            align=align_key,
        )
        subspace = _subspace_overlap_metrics(
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
    target = np.asarray(heff)
    candidate = target * np.asarray(mask, dtype=bool)[None, :, :]
    candidate = 0.5 * (candidate + np.swapaxes(candidate.conj(), -1, -2))
    if target_eig is None or target_vec is None:
        target_eig, target_vec = np.linalg.eigh(target)
    candidate_eig, candidate_vec = np.linalg.eigh(candidate)
    dim = int(target.shape[-1])
    primary_slice = tuple(_auto_low_energy_band_slice(dim, int(primary_bands), target_bands))
    plot_slice = tuple(_auto_low_energy_band_slice(dim, int(plot_bands), target_bands))
    primary_band = _band_refinement_metrics(
        candidate,
        target_eig,
        target,
        band_slice=(int(primary_slice[0]), int(primary_slice[1])),
        align=target_bands,
    )
    plot_band = _band_refinement_metrics(
        candidate,
        target_eig,
        target,
        band_slice=(int(plot_slice[0]), int(plot_slice[1])),
        align=target_bands,
    )
    target_basis = target_vec[:, :, primary_slice[0] : primary_slice[1]]
    candidate_basis = candidate_vec[:, :, primary_slice[0] : primary_slice[1]]
    subspace = _subspace_overlap_metrics(candidate_basis, target_basis)
    _low_vec, low_matrix = _low_subspace_matrix_residual(
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
    max_shell: int,
    thresholds: Mapping[str, Any] | None = None,
    candidate_pairs: Sequence[tuple[int, int]] | None = None,
    tol: float = 1.0e-6,
) -> dict[str, Any]:
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
    pair_list = (
        [(int(intra), int(inter)) for intra, inter in candidate_pairs]
        if candidate_pairs is not None
        else [
            (int(intra_shells), int(inter_shells))
            for intra_shells in range(int(max_shell) + 1)
            for inter_shells in range(int(max_shell) + 1)
        ]
    )
    for intra_shells, inter_shells in pair_list:
        mask = _selected_harmonic_support_mask(
            qset1,
            qset2,
            n_orb=n_orb,
            current_counts={"intra": int(intra_shells), "inter": int(inter_shells)},
            tol=tol,
        )
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
        "max_shell": int(max_shell),
        "candidate_pairs": [[int(intra), int(inter)] for intra, inter in pair_list],
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
    raw = fit.get("harmonic_recommendation", True)
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
        f"overlap={float(candidate['subspace_mean_overlap']):.6f}"
    )


def _format_harmonic_recommendation_metrics(candidate: Mapping[str, Any]) -> str:
    return (
        f"plot {float(candidate['plot_rms_mev']):.3f}/{float(candidate['plot_max_mev']):.3f} meV | "
        f"primary {float(candidate['primary_rms_mev']):.3f}/{float(candidate['primary_max_mev']):.3f} meV | "
        f"overlap {float(candidate['subspace_mean_overlap']):.6f}"
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
    relaxed_thresholds = {
        "plot_rms_mev": 1.0,
        "plot_max_mev": 3.0,
        "min_overlap": 0.98,
    }
    relaxed = [
        item
        for item in candidates
        if _harmonic_ablation_candidate_is_accepted(item, relaxed_thresholds)
    ]
    rows = relaxed or candidates
    return min(rows, key=_harmonic_recommendation_support_key)


def _harmonic_recommendation_named_candidates(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    selected = report.get("selected", {})
    out: dict[str, Mapping[str, Any]] = {}
    if isinstance(selected, Mapping):
        out["high"] = selected
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
    limits = _harmonic_count_limits(harmonics)
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
        + f" | {report.get('selection_status', 'unknown')}"
    )
    print(f"{prefix} " + _terminal_color(line, "32", enabled=use_color), flush=True)
    if plot_path is not None:
        print(
            f"{prefix} "
            + _terminal_color(f"  band plot: {plot_path.resolve()}", "35", enabled=use_color),
            flush=True,
        )


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
    plot_candidates = _harmonic_recommendation_plot_candidates(report, current_counts=current_counts)
    if len(plot_candidates) < 1:
        return None
    target = np.asarray(heff, dtype=np.complex128)
    if target.ndim != 3 or target.shape[-1] != target.shape[-2]:
        raise ValueError(f"harmonic recommendation band plot expects Heff shape (Nk,dim,dim), got {target.shape}")
    dim = int(target.shape[-1])
    band_slice = tuple(_auto_low_energy_band_slice(dim, int(plot_bands), target_bands))
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
        mask = _selected_harmonic_support_mask(
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
    heff_for_selection = _select_rows(np.load(heff_file, mmap_mode="r"), band_indices)
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
    model_config: ConfiguredModel,
    output_dir: Path,
) -> None:
    fit = model_config.raw.get("fit", {})
    if not isinstance(fit, Mapping):
        return
    harmonic_selection_enabled, harmonic_selection_cfg = _harmonic_selection_options(fit)
    if harmonic_selection_enabled and isinstance(model_config.raw.get("model", {}).get("auto_low_energy_harmonic_selection"), Mapping):
        return
    recommendation_enabled, recommendation_cfg = _harmonic_recommendation_options(
        fit,
        inherit_from=harmonic_selection_cfg,
    )
    if not recommendation_enabled:
        return
    try:
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
    except Exception as exc:
        _progress_line(f"harmonic recommendation skipped: {exc}", enabled=True, style="warning")


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
    eig = np.asarray(eigvals, dtype=float)
    if eig.ndim != 2:
        raise ValueError(f"auto low-energy windows expect eigvals shape (Nk,dim), got {eig.shape}")
    band_slice = _auto_low_energy_band_slice(int(eig.shape[1]), int(n_bands), target_bands)
    gaps = _boundary_gap_mev_by_k(eig, band_slice)
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
        n_bands = _shell_subspace_band_count(dim, fraction=center_fraction)
        band_slice = _auto_low_energy_band_slice(dim, n_bands, target)
        gap_by_k = _boundary_gap_mev_by_k(eig, band_slice)
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
        center_n = _shell_subspace_band_count(dim, fraction=center_fraction)
        lo_n = hi_n = int(center_n)
    center_n_float = center_fraction * dim
    candidates: list[dict[str, Any]] = []
    for n_bands in range(int(lo_n), int(hi_n) + 1):
        band_slice = _auto_low_energy_band_slice(dim, int(n_bands), target)
        gap_by_k = _boundary_gap_mev_by_k(eig, band_slice)
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
    arr = _validate_kpoints(kpoints)
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
    arr = _validate_kpoints(kpoints)
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
    arr = _validate_kpoints(kpoints)
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
    arr = _validate_kpoints(kpoints)
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


def _matrix_loss_block_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _MATRIX_LOSS_BLOCK_ALIASES:
        raise ValueError(
            "fit.refine_bands.matrix_loss.blocks contains unknown block "
            f"{name!r}; expected one of {sorted(_MATRIX_LOSS_BLOCK_ALIASES)}"
        )
    return _MATRIX_LOSS_BLOCK_ALIASES[key]


def _matrix_loss_block_masks(moire_config: MoireConfig, dim: int) -> dict[str, np.ndarray]:
    q1_count = int(np.asarray(moire_config.Q_set1).shape[0])
    q2_count = int(np.asarray(moire_config.Q_set2).shape[0])
    n_orb1 = int(moire_config.n_orb1)
    n_orb2 = int(moire_config.n_orb2)
    expected = q1_count * n_orb1 + q2_count * n_orb2
    if expected != int(dim):
        raise ValueError(
            "fit.refine_bands.matrix_loss cannot build qset_orbital blocks: "
            f"model dimension is {dim}, but qset/orbital layout implies {expected}"
        )

    sectors: list[int] = []
    q_indices: list[int] = []
    orbital_indices: list[int] = []
    for sector, q_count, n_orb in ((1, q1_count, n_orb1), (2, q2_count, n_orb2)):
        for orbital_index in range(n_orb):
            for q_index in range(q_count):
                sectors.append(sector)
                q_indices.append(q_index)
                orbital_indices.append(orbital_index)

    sector_arr = np.asarray(sectors, dtype=int)
    q_arr = np.asarray(q_indices, dtype=int)
    orbital_arr = np.asarray(orbital_indices, dtype=int)
    same_sector = sector_arr[:, None] == sector_arr[None, :]
    same_q = q_arr[:, None] == q_arr[None, :]
    same_orbital = orbital_arr[:, None] == orbital_arr[None, :]
    kinetic_diagonal = same_sector & same_q & same_orbital
    intralayer = same_sector & ~kinetic_diagonal
    interlayer = ~same_sector
    return {
        "all": np.ones((dim, dim), dtype=bool),
        "kinetic_diagonal": kinetic_diagonal,
        "intralayer": intralayer,
        "interlayer": interlayer,
    }


def _matrix_loss_config(raw_cfg: Mapping[str, Any], *, matrix_weight: float, matrix_sigma_mev: float) -> dict[str, Any] | None:
    raw = raw_cfg.get("matrix_loss")
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError("fit.refine_bands.matrix_loss must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    mode = str(cfg.get("mode", "block_normalized")).strip().lower()
    if mode in {"block", "blocks"}:
        mode = "block_normalized"
    if mode not in {"global", "dense", "block_normalized"}:
        raise ValueError("fit.refine_bands.matrix_loss.mode must be 'global' or 'block_normalized'")
    weight = float(cfg.get("weight", matrix_weight if matrix_weight > 0.0 else 1.0))
    sigma_mev = float(cfg.get("sigma_mev", matrix_sigma_mev))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.sigma_mev must be positive")
    blocks_raw = cfg.get("blocks")
    if blocks_raw is None:
        blocks = {"kinetic_diagonal": 1.0, "intralayer": 1.0, "interlayer": 1.0}
    elif not isinstance(blocks_raw, Mapping):
        raise ValueError("fit.refine_bands.matrix_loss.blocks must be a mapping")
    else:
        blocks = {}
        for block_name, block_weight in blocks_raw.items():
            canonical = _matrix_loss_block_name(str(block_name))
            value = float(block_weight)
            if value < 0.0:
                raise ValueError("fit.refine_bands.matrix_loss block weights must be non-negative")
            blocks[canonical] = value
    return {
        "enabled": True,
        "mode": "global" if mode == "dense" else mode,
        "weight": weight,
        "sigma_mev": sigma_mev,
        "normalize_band_loss": bool(cfg.get("normalize_band_loss", True)),
        "blocks": blocks,
    }


def _matrix_loss_residual(
    matrix_delta: np.ndarray,
    masks: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    report: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta = np.asarray(matrix_delta, dtype=np.complex128)
    if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
        raise ValueError(f"matrix loss expects delta shape (Nk,dim,dim), got {delta.shape}")
    mode = str(config.get("mode", "block_normalized")).strip().lower()
    weight = float(config.get("weight", 1.0))
    sigma_mev = float(config.get("sigma_mev", 10.0))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.matrix_loss.sigma_mev must be positive")
    sigma = sigma_mev * 1.0e-3
    if mode == "global":
        scaled = np.sqrt(weight) * delta.reshape(-1) / sigma
        vector = np.concatenate([scaled.real, scaled.imag])
        if not report:
            return vector, {"enabled": True, "mode": "global"}
        abs_delta = np.abs(delta)
        report = {
            "enabled": True,
            "mode": "global",
            "weight": weight,
            "sigma_mev": sigma_mev,
            "rms_mev": float(np.sqrt(np.mean(abs_delta**2)) * 1000.0),
            "max_mev": float(np.max(abs_delta) * 1000.0),
            "loss_norm_sq": float(np.sum(vector**2)),
        }
        return vector, report
    if mode != "block_normalized":
        raise ValueError("fit.refine_bands.matrix_loss.mode must be 'global' or 'block_normalized'")

    blocks_raw = config.get("blocks", {"kinetic_diagonal": 1.0, "intralayer": 1.0, "interlayer": 1.0})
    if not isinstance(blocks_raw, Mapping):
        raise ValueError("fit.refine_bands.matrix_loss.blocks must be a mapping")
    parts: list[np.ndarray] = []
    block_reports: dict[str, Any] = {}
    for raw_name, raw_weight in blocks_raw.items():
        name = _matrix_loss_block_name(str(raw_name))
        block_weight = float(raw_weight)
        if block_weight < 0.0:
            raise ValueError("fit.refine_bands.matrix_loss block weights must be non-negative")
        mask = np.asarray(masks.get(name), dtype=bool)
        if mask.shape != delta.shape[1:]:
            raise ValueError(f"fit.refine_bands.matrix_loss block {name!r} has shape {mask.shape}, expected {delta.shape[1:]}")
        count = int(np.count_nonzero(mask))
        if count == 0 or block_weight == 0.0 or weight == 0.0:
            block_reports[name] = {
                "weight": block_weight,
                "count": count,
                "rms_mev": None,
                "max_mev": None,
                "loss_norm_sq": 0.0,
            }
            continue
        values = delta[:, mask].reshape(-1)
        normalizer = float(np.sqrt(values.size))
        scaled = np.sqrt(weight * block_weight) * values / (sigma * normalizer)
        vector = np.concatenate([scaled.real, scaled.imag])
        parts.append(vector)
        if not report:
            continue
        abs_values = np.abs(values)
        block_reports[name] = {
            "weight": block_weight,
            "count": count,
            "rms_mev": float(np.sqrt(np.mean(abs_values**2)) * 1000.0),
            "max_mev": float(np.max(abs_values) * 1000.0),
            "loss_norm_sq": float(np.sum(vector**2)),
        }
    residual = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
    if not report:
        return residual, {"enabled": True, "mode": "block_normalized"}
    return residual, {
        "enabled": True,
        "mode": "block_normalized",
        "weight": weight,
        "sigma_mev": sigma_mev,
        "normalization": "per_block_rms",
        "blocks": block_reports,
    }


def _refinement_loss_band_slice(
    raw: Mapping[str, Any],
    *,
    default_slice: tuple[int, int],
    dim: int,
    target_bands: str,
) -> tuple[int, int]:
    if raw.get("band_slice") is not None:
        values = _as_int_list(raw.get("band_slice"), name="fit.refine_bands loss band_slice")
        if len(values) != 2:
            raise ValueError(f"fit.refine_bands loss band_slice must be [start, stop], got {values!r}")
        start, stop = int(values[0]), int(values[1])
    elif raw.get("top_bands") is not None:
        start, stop = _auto_low_energy_band_slice(dim, int(raw["top_bands"]), "top")
    elif raw.get("bottom_bands") is not None:
        start, stop = _auto_low_energy_band_slice(dim, int(raw["bottom_bands"]), "bottom")
    elif raw.get("n_bands") is not None:
        start, stop = _auto_low_energy_band_slice(dim, int(raw["n_bands"]), target_bands)
    else:
        start, stop = int(default_slice[0]), int(default_slice[1])
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid refinement loss slice [{start}, {stop}] for dimension {dim}")
    return int(start), int(stop)


def _subspace_loss_config(
    raw_cfg: Mapping[str, Any],
    *,
    default_slice: tuple[int, int],
    dim: int,
    target_bands: str,
) -> dict[str, Any] | None:
    raw = raw_cfg.get("subspace_loss")
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError("fit.refine_bands.subspace_loss must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    mode = str(cfg.get("mode", "principal_angles")).strip().lower()
    if mode not in {"principal_angles", "projector"}:
        raise ValueError("fit.refine_bands.subspace_loss.mode must be 'principal_angles' or 'projector'")
    weight = float(cfg.get("weight", 1.0))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.subspace_loss.weight must be non-negative")
    band_slice = _refinement_loss_band_slice(cfg, default_slice=default_slice, dim=dim, target_bands=target_bands)
    return {
        "enabled": True,
        "mode": mode,
        "weight": weight,
        "normalize": bool(cfg.get("normalize", True)),
        "gap_tolerance_mev": float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))),
        "band_slice": [int(band_slice[0]), int(band_slice[1])],
    }


def _low_subspace_matrix_loss_config(
    raw_cfg: Mapping[str, Any],
    *,
    default_slice: tuple[int, int],
    dim: int,
    target_bands: str,
) -> dict[str, Any] | None:
    raw = raw_cfg.get("low_subspace_matrix_loss")
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    weight = float(cfg.get("weight", 1.0))
    sigma_mev = float(cfg.get("sigma_mev", cfg.get("matrix_sigma_mev", 10.0)))
    if weight < 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.weight must be non-negative")
    if sigma_mev <= 0.0:
        raise ValueError("fit.refine_bands.low_subspace_matrix_loss.sigma_mev must be positive")
    band_slice = _refinement_loss_band_slice(cfg, default_slice=default_slice, dim=dim, target_bands=target_bands)
    return {
        "enabled": True,
        "weight": weight,
        "sigma_mev": sigma_mev,
        "normalize": bool(cfg.get("normalize", True)),
        "gap_tolerance_mev": float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))),
        "band_slice": [int(band_slice[0]), int(band_slice[1])],
    }


def _shell_projected_matrix_loss_config(raw_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = raw_cfg.get("shell_projected_matrix_loss")
    source_key = "shell_projected_matrix_loss"
    if raw is None and "shell_subspace_loss" in raw_cfg:
        raw = raw_cfg.get("shell_subspace_loss")
        source_key = "shell_subspace_loss"
    elif raw is not None and "shell_subspace_loss" in raw_cfg:
        raise ValueError(
            "fit.refine_bands.shell_projected_matrix_loss conflicts with legacy "
            "fit.refine_bands.shell_subspace_loss; use only shell_projected_matrix_loss"
        )
    if raw in (None, False):
        return None
    if raw is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
    else:
        raise ValueError(f"fit.refine_bands.{source_key} must be a mapping or boolean when provided")
    if not bool(cfg.get("enabled", True)):
        return None
    weight = float(cfg.get("weight", 1.0))
    if weight < 0.0:
        raise ValueError(f"fit.refine_bands.{source_key}.weight must be non-negative")
    max_shells_raw = cfg.get("max_shells", 3)
    max_shells = int(max_shells_raw) if max_shells_raw is not None else None
    if max_shells is not None and max_shells <= 0:
        raise ValueError(f"fit.refine_bands.{source_key}.max_shells must be positive")
    raw_window = cfg.get("window")
    if raw_window is None:
        if source_key == "shell_subspace_loss":
            window = {
                "mode": "fixed_fraction",
                "center_fraction": float(cfg.get("subspace_fraction", 0.5)),
            }
        else:
            window = {
                "mode": "gap_aware",
                "center_fraction": float(cfg.get("subspace_fraction", 0.5)),
                "search_fraction": [0.35, 0.65],
                "gap_tolerance_mev": float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))),
            }
    elif isinstance(raw_window, Mapping):
        window = dict(raw_window)
        if "center_fraction" not in window and "subspace_fraction" in cfg:
            window["center_fraction"] = float(cfg["subspace_fraction"])
        window.setdefault("center_fraction", 0.5)
        window.setdefault("mode", "gap_aware" if source_key != "shell_subspace_loss" else "fixed_fraction")
        if str(window.get("mode", "")).strip().lower() == "gap_aware":
            window.setdefault("search_fraction", [0.35, 0.65])
            window.setdefault("gap_tolerance_mev", float(cfg.get("gap_tolerance_mev", raw_cfg.get("gap_tolerance_mev", 0.1))))
    else:
        raise ValueError(f"fit.refine_bands.{source_key}.window must be a mapping when provided")
    center_fraction = float(window.get("center_fraction", window.get("subspace_fraction", 0.5)))
    if center_fraction <= 0.0 or center_fraction > 1.0:
        raise ValueError(f"fit.refine_bands.{source_key}.window.center_fraction must be in (0, 1]")
    window["center_fraction"] = center_fraction
    shell_decay = float(cfg.get("shell_decay", 1.0))
    if shell_decay < 0.0:
        raise ValueError(f"fit.refine_bands.{source_key}.shell_decay must be non-negative")
    return {
        "enabled": True,
        "weight": weight,
        "max_shells": max_shells,
        "subspace_fraction": center_fraction,
        "window": window,
        "shell_decay": shell_decay,
        "tol": float(cfg.get("tol", 1.0e-6)),
        "source_key": source_key,
        "legacy_alias": "shell_subspace_loss",
    }


def _refinement_acceptance_guard_config(
    raw_cfg: Mapping[str, Any],
    model_config: ConfiguredModel,
    *,
    dim: int,
) -> dict[str, Any] | None:
    raw = raw_cfg.get("acceptance_guard")
    if raw is False:
        return None
    if raw is None:
        enabled = bool(raw_cfg.get("enabled", False))
        cfg: dict[str, Any] = {}
    elif raw is True:
        enabled = True
        cfg = {}
    elif isinstance(raw, Mapping):
        cfg = dict(raw)
        enabled = bool(cfg.get("enabled", True))
    else:
        raise ValueError("fit.refine_bands.acceptance_guard must be a mapping or boolean when provided")
    if not enabled:
        return None

    plot_config = dict(model_config.band_plot_config or {})
    if "band_slice" in cfg:
        plot_config["band_slice"] = cfg["band_slice"]
    if "top_bands" in cfg:
        plot_config["top_bands"] = cfg["top_bands"]
        plot_config.pop("bottom_bands", None)
    if "bottom_bands" in cfg:
        plot_config["bottom_bands"] = cfg["bottom_bands"]
        plot_config.pop("top_bands", None)
    if "align" in cfg:
        plot_config["align"] = cfg["align"]
    start, stop = _resolve_plot_band_slice(
        nbands=int(dim),
        metric_band_slice=model_config.band_slice,
        plot_config=plot_config,
    )
    align = str(plot_config.get("align", raw_cfg.get("align", "none"))).strip().lower()
    max_rms_increase = float(cfg.get("max_rms_increase_mev", 0.05))
    max_max_increase = float(cfg.get("max_max_increase_mev", 0.25))
    target_bands = str(raw_cfg.get("target_bands", model_config.raw.get("model", {}).get("target_bands", "top"))).strip().lower()
    windows: list[dict[str, Any]] = [
        {
            "name": "validation_window",
            "band_slice": [int(start), int(stop)],
            "align": align,
            "max_rms_increase_mev": max_rms_increase,
            "max_max_increase_mev": max_max_increase,
        }
    ]
    if bool(cfg.get("guard_primary_window", True)):
        primary_count = max(1, min(int(sum(model_config.n_orb)), int(dim)))
        primary_slice = _auto_low_energy_band_slice(int(dim), primary_count, target_bands)
        windows.append(
            {
                "name": "primary_window",
                "band_slice": primary_slice,
                "align": str(cfg.get("primary_align", align)),
                "max_rms_increase_mev": float(cfg.get("max_primary_rms_increase_mev", max_rms_increase)),
                "max_max_increase_mev": float(cfg.get("max_primary_max_increase_mev", max_max_increase)),
            }
        )
    if bool(cfg.get("guard_all_bands", True)):
        windows.append(
            {
                "name": "all_bands",
                "band_slice": [0, int(dim)],
                "align": str(cfg.get("all_band_align", align)),
                "max_rms_increase_mev": float(cfg.get("max_all_band_rms_increase_mev", max_rms_increase)),
                "max_max_increase_mev": float(cfg.get("max_all_band_max_increase_mev", max_max_increase)),
            }
        )

    raw_alphas = cfg.get("line_search_alphas", [1.0, 0.75, 0.5, 0.25, 0.1, 0.05])
    if not isinstance(raw_alphas, Sequence) or isinstance(raw_alphas, (str, bytes)):
        raise ValueError("fit.refine_bands.acceptance_guard.line_search_alphas must be a list")
    alphas = sorted({float(value) for value in raw_alphas if float(value) > 0.0}, reverse=True)
    if not alphas:
        alphas = [1.0]
    selection = str(cfg.get("selection", "first_accepted")).strip().lower()
    if selection not in {"first_accepted", "best_validation_window"}:
        raise ValueError(
            "fit.refine_bands.acceptance_guard.selection must be "
            "'first_accepted' or 'best_validation_window'"
        )

    return {
        "enabled": True,
        "band_slice": [int(start), int(stop)],
        "align": align,
        "max_rms_increase_mev": max_rms_increase,
        "max_max_increase_mev": max_max_increase,
        "windows": windows,
        "line_search_alphas": alphas,
        "selection": selection,
    }


def _guard_window_acceptance(
    *,
    base_h: np.ndarray,
    candidate_h: np.ndarray,
    heff_eig: np.ndarray,
    heff_all: np.ndarray,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    band_slice = tuple(int(value) for value in window["band_slice"])
    align = str(window.get("align", "none"))
    initial = _band_refinement_metrics(base_h, heff_eig, heff_all, band_slice=band_slice, align=align)
    candidate = _band_refinement_metrics(candidate_h, heff_eig, heff_all, band_slice=band_slice, align=align)
    rms_limit = float(initial["top_band_rms_mev"]) + float(window["max_rms_increase_mev"])
    max_limit = float(initial["top_band_max_mev"]) + float(window["max_max_increase_mev"])
    accepted = (
        float(candidate["top_band_rms_mev"]) <= rms_limit
        and float(candidate["top_band_max_mev"]) <= max_limit
    )
    return {
        **dict(window),
        "initial": initial,
        "candidate": candidate,
        "rms_limit_mev": float(rms_limit),
        "max_limit_mev": float(max_limit),
        "accepted": bool(accepted),
    }


def _apply_refinement_acceptance_guard(
    *,
    raw_cfg: Mapping[str, Any],
    model_config: ConfiguredModel,
    base_h: np.ndarray,
    heff_eig: np.ndarray,
    heff_all: np.ndarray,
    y0: np.ndarray,
    candidate_y: np.ndarray,
    h_from_y: Any,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    guard_cfg = _refinement_acceptance_guard_config(raw_cfg, model_config, dim=base_h.shape[-1])
    if guard_cfg is None:
        return np.asarray(candidate_y, dtype=float), None

    y0_arr = np.asarray(y0, dtype=float)
    candidate_arr = np.asarray(candidate_y, dtype=float)
    trials: list[dict[str, Any]] = []
    accepted_trials: list[tuple[np.ndarray, dict[str, Any]]] = []
    for alpha in guard_cfg["line_search_alphas"]:
        trial_y = y0_arr + float(alpha) * (candidate_arr - y0_arr)
        trial_h = h_from_y(trial_y)
        windows = [
            _guard_window_acceptance(
                base_h=base_h,
                candidate_h=trial_h,
                heff_eig=heff_eig,
                heff_all=heff_all,
                window=window,
            )
            for window in guard_cfg["windows"]
        ]
        failed = [str(window["name"]) for window in windows if not bool(window["accepted"])]
        trial_report = {
            "alpha": float(alpha),
            "accepted": not failed,
            "failed_windows": failed,
            "windows": windows,
        }
        trials.append(trial_report)
        if not failed:
            accepted_trials.append((trial_y, trial_report))
        if not failed and str(guard_cfg.get("selection", "first_accepted")) == "first_accepted":
            return trial_y, {
                **guard_cfg,
                "selected_alpha": float(alpha),
                "accepted": True,
                "reverted": False,
                "scaled": float(alpha) < 1.0,
                "failed_windows": [],
                "trials": trials,
                "windows": windows,
                "reason": None,
            }

    if accepted_trials:
        def trial_key(item: tuple[np.ndarray, dict[str, Any]]) -> tuple[float, float, float]:
            _trial_y, trial = item
            windows = list(trial.get("windows", []))
            validation = next((window for window in windows if window.get("name") == "validation_window"), windows[0])
            candidate = validation["candidate"]
            return (
                float(candidate["top_band_rms_mev"]),
                float(candidate["top_band_max_mev"]),
                -float(trial.get("alpha", 0.0)),
            )

        selected_y, selected_trial = min(accepted_trials, key=trial_key)
        return selected_y, {
            **guard_cfg,
            "selected_alpha": float(selected_trial["alpha"]),
            "accepted": True,
            "reverted": False,
            "scaled": float(selected_trial["alpha"]) < 1.0,
            "failed_windows": [],
            "trials": trials,
            "windows": list(selected_trial.get("windows", [])),
            "reason": None,
        }

    last = trials[-1] if trials else {"failed_windows": [], "windows": []}
    return y0_arr.copy(), {
        **guard_cfg,
        "selected_alpha": 0.0,
        "accepted": False,
        "reverted": True,
        "scaled": False,
        "failed_windows": list(last.get("failed_windows", [])),
        "trials": trials,
        "windows": list(last.get("windows", [])),
        "reason": "validation_window_worsened",
    }


def refine_band_coefficients(moire_config: MoireConfig, model_config: ConfiguredModel, model: Any) -> dict[str, Any]:
    raw_cfg = dict(model_config.band_refinement_config or {})
    if not bool(raw_cfg.get("enabled", False)):
        return {"enabled": False}
    if moire_config.kpoints is None:
        raise ValueError("fit.refine_bands requires model kpoints")
    use_fit_kpoints = bool(raw_cfg.get("use_fit_kpoints", str(raw_cfg.get("mode", "")).strip().lower() == "auto_low_energy"))
    refine_row_selector: Sequence[int] | None = model_config.band_indices
    kpoints = np.asarray(moire_config.kpoints, dtype=float)
    fit_kpoints_report = {
        "source": "band_indices",
        "count": int(len(kpoints)),
        "selected_indices": [int(value) for value in (model_config.band_indices or [])],
    }
    if raw_cfg.get("indices") is not None:
        explicit_indices = _as_int_list(raw_cfg.get("indices"), name="fit.refine_bands.indices")
        for index in explicit_indices:
            if index < 0 or index >= len(kpoints):
                raise IndexError(
                    f"fit.refine_bands.indices contains {index}, but available k indices are 0..{len(kpoints) - 1}"
                )
        kpoints = _select_rows(kpoints, explicit_indices)
        refine_row_selector = [int(value) for value in explicit_indices]
        fit_kpoints_report = {
            "source": "fit.refine_bands.indices",
            "count": int(len(kpoints)),
            "selected_indices": [int(value) for value in refine_row_selector],
        }
    elif use_fit_kpoints and moire_config.kpoints_fit is not None and model_config.fit_indices:
        kpoints = np.asarray(moire_config.kpoints_fit, dtype=float)
        refine_row_selector = [int(value) for value in model_config.fit_indices]
        fit_kpoints_report = {
            "source": "fit.indices",
            "count": int(len(kpoints)),
            "selected_indices": [int(value) for value in refine_row_selector],
        }
    if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
        heff_eig = np.load(model_config.heff_eig_file)
    else:
        heff_eig = np.linalg.eigvalsh(np.load(model_config.heff_file, mmap_mode="r"))
    heff_eig = _select_rows(heff_eig, refine_row_selector)
    heff_all = _select_rows(np.load(model_config.heff_file, mmap_mode="r"), refine_row_selector)
    if int(heff_all.shape[0]) != int(len(kpoints)):
        raise ValueError(
            "fit.refine_bands k-point/Heff row mismatch: "
            f"kpoints={len(kpoints)}, heff_rows={heff_all.shape[0]}, source={fit_kpoints_report['source']}"
        )

    variables = _select_band_refinement_variables(model, raw_cfg)
    if not variables:
        raise ValueError("fit.refine_bands selected no active variables")

    base_h_start = time.perf_counter()
    print("[kp model] refinement base Hamiltonian ...", flush=True)
    base_h = _model_hamiltonians_for_kpoints(moire_config, model, kpoints)
    print(f"[kp model] refinement base Hamiltonian done in {time.perf_counter() - base_h_start:.2f} s", flush=True)
    band_slice = _refinement_band_slice(raw_cfg, model_config, base_h.shape[-1])
    align = str(raw_cfg.get("align", model_config.band_plot_config.get("align", "top"))).strip().lower()
    initial_metrics_start = time.perf_counter()
    print("[kp model] refinement initial metrics ...", flush=True)
    initial_metrics = _band_refinement_metrics(base_h, heff_eig, heff_all, band_slice=band_slice, align=align)
    print(f"[kp model] refinement initial metrics done in {time.perf_counter() - initial_metrics_start:.2f} s", flush=True)
    solver = str(raw_cfg.get("solver", "")).strip().lower()
    direct_linear_response = solver in {"linear_low_subspace", "low_subspace_linear", "linear_projected_matrix"}
    direct_response_state = None
    try:
        direct_response_state = _prepare_band_state(moire_config, model)
    except AttributeError:
        direct_response_state = None

    y0_values: list[float] = []
    basis_items: list[np.ndarray] = []
    kept_variables: list[tuple[Any, str]] = []
    norm_tol = float(raw_cfg.get("variable_norm_tol", 1.0e-12))
    response_build_start = time.perf_counter()
    response_build_report = {
        "mode": "direct_pair_parallel" if direct_response_state is not None else "full_model_fallback",
        "direct_pair_terms": 0,
        "fallback_components": 0,
        "zero_norm_components": 0,
    }

    def append_response(term: Any, component: str, delta: np.ndarray) -> None:
        original = _term_component_value(term, component)
        if float(np.linalg.norm(delta)) <= norm_tol:
            response_build_report["zero_norm_components"] += 1
            return
        kept_variables.append((term, component))
        y0_values.append(original)
        basis_items.append(delta)

    if direct_response_state is not None:
        term_order: list[Any] = []
        components_by_term: dict[int, list[str]] = {}
        terms_by_id: dict[int, Any] = {}
        for term, component in variables:
            term_id = id(term)
            if term_id not in components_by_term:
                term_order.append(term)
                terms_by_id[term_id] = term
                components_by_term[term_id] = []
            components_by_term[term_id].append(component)
        response_n_jobs_raw = raw_cfg.get("response_n_jobs")
        if response_n_jobs_raw is None:
            response_n_jobs = 1
        else:
            response_n_jobs = int(response_n_jobs_raw)
        if response_n_jobs == 0:
            response_n_jobs = 1
        response_build_report["parallel_n_jobs"] = int(response_n_jobs)

        trace_response_terms = bool(raw_cfg.get("response_trace_terms", False))

        def compute_term_pair(index: int, term: Any) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
            if trace_response_terms:
                print(
                    "[kp model] refinement response term "
                    f"{index + 1}/{len(term_order)} tag={getattr(term, 'tag', '')} "
                    f"component_count={len(components_by_term[id(term)])} start",
                    flush=True,
                )
            term_start = time.perf_counter()
            pair = _term_response_pair_hamiltonians_for_kpoints(direct_response_state, term, kpoints)
            if trace_response_terms:
                print(
                    "[kp model] refinement response term "
                    f"{index + 1}/{len(term_order)} done in {time.perf_counter() - term_start:.2f} s",
                    flush=True,
                )
            return id(term), pair

        try:
            if abs(response_n_jobs) == 1 or len(term_order) <= 1:
                term_pairs = [compute_term_pair(index, term) for index, term in enumerate(term_order)]
            else:
                response_backend = str(raw_cfg.get("response_backend", "threading")).strip().lower()
                parallel_kwargs: dict[str, Any] = {
                    "n_jobs": response_n_jobs,
                    "verbose": int(raw_cfg.get("response_verbose", 0) or 0),
                }
                if response_backend in {"thread", "threads", "threading"}:
                    parallel_kwargs.update({"prefer": "threads", "require": "sharedmem"})
                elif response_backend in {"process", "processes", "loky"}:
                    parallel_kwargs.update({"backend": "loky"})
                else:
                    raise ValueError(
                        "fit.refine_bands.response_backend must be 'threading' or 'loky', "
                        f"got {raw_cfg.get('response_backend')!r}"
                    )
                response_build_report["parallel_backend"] = response_backend
                term_pairs = joblib.Parallel(
                    **parallel_kwargs,
                )(joblib.delayed(compute_term_pair)(index, term) for index, term in enumerate(term_order))
            response_pair_cache = {term_id: pair for term_id, pair in term_pairs}
            response_build_report["direct_pair_terms"] = int(len(response_pair_cache))
            for term, component in variables:
                pair = response_pair_cache[id(term)]
                append_response(term, component, pair[0] if component == "real" else pair[1])
        except NotImplementedError:
            direct_response_state = None
            y0_values.clear()
            basis_items.clear()
            kept_variables.clear()
            response_build_report["mode"] = "full_model_fallback"
            response_build_report["direct_pair_terms"] = 0
            response_build_report["zero_norm_components"] = 0

    if direct_response_state is None:
        for term, component in variables:
            original = _term_component_value(term, component)
            _set_term_component_value(term, component, original + 1.0)
            try:
                delta = _model_hamiltonians_for_kpoints(moire_config, model, kpoints) - base_h
            finally:
                _set_term_component_value(term, component, original)
            response_build_report["fallback_components"] += 1
            append_response(term, component, delta)
    response_build_report["wall_seconds"] = float(time.perf_counter() - response_build_start)
    response_build_report["requested_components"] = int(len(variables))
    response_build_report["kept_components"] = int(len(kept_variables))
    if not kept_variables:
        raise ValueError("fit.refine_bands selected variables but all had zero model response")

    y0 = np.asarray(y0_values, dtype=float)
    basis = np.asarray(basis_items, dtype=np.complex128)
    print(
        "[kp model] refinement response basis: "
        f"{response_build_report['kept_components']}/{response_build_report['requested_components']} components, "
        f"{response_build_report['direct_pair_terms']} direct terms, "
        f"{response_build_report.get('parallel_n_jobs', 1)} jobs, "
        f"{response_build_report['wall_seconds']:.2f} s",
        flush=True,
    )
    max_variables_raw = raw_cfg.get("max_variables")
    max_variables = int(max_variables_raw) if max_variables_raw is not None else None
    if max_variables is not None and max_variables > 0 and len(kept_variables) > max_variables:
        skipped_metrics = _band_refinement_metrics(base_h, heff_eig, heff_all, band_slice=band_slice, align=align)
        return {
            "enabled": True,
            "skipped": True,
            "reason": "variable_count_exceeds_max_variables",
            "fit_kpoints": fit_kpoints_report,
            "band_slice": [int(band_slice[0]), int(band_slice[1])],
            "align": align,
            "n_variables": int(len(kept_variables)),
            "max_variables": int(max_variables),
            "response_basis": response_build_report,
            "initial": skipped_metrics,
            "refined": skipped_metrics,
            "weighted_band_loss": {"enabled": False, "reason": "skipped"},
            "matrix_loss": {"enabled": False, "reason": "skipped"},
            "subspace_loss": {"enabled": False, "reason": "skipped"},
            "low_subspace_matrix_loss": {"enabled": False, "reason": "skipped"},
            "max_nfev": int(raw_cfg.get("max_nfev", 250)),
            "nfev": 0,
            "primary_nfev": 0,
        }
    scale = np.maximum(np.abs(y0), 1.0)
    band_sigma = float(raw_cfg.get("band_sigma_mev", 1.0)) * 1.0e-3
    matrix_sigma = float(raw_cfg.get("matrix_sigma_mev", 10.0)) * 1.0e-3
    if band_sigma <= 0.0 or matrix_sigma <= 0.0:
        raise ValueError("fit.refine_bands band_sigma_mev and matrix_sigma_mev must be positive")
    matrix_weight = float(raw_cfg.get("matrix_weight", 0.0))
    coeff_weight = float(raw_cfg.get("coefficient_weight", 0.0))
    if matrix_weight < 0.0 or coeff_weight < 0.0:
        raise ValueError("fit.refine_bands matrix_weight and coefficient_weight must be non-negative")
    matrix_loss_cfg = _matrix_loss_config(raw_cfg, matrix_weight=matrix_weight, matrix_sigma_mev=matrix_sigma * 1000.0)
    matrix_loss_masks = None
    if matrix_loss_cfg is not None:
        matrix_loss_masks = _matrix_loss_block_masks(moire_config, dim=base_h.shape[-1])
    target_bands = str(raw_cfg.get("target_bands", model_config.raw.get("model", {}).get("target_bands", "top"))).strip().lower()
    subspace_loss_cfg = _subspace_loss_config(
        raw_cfg,
        default_slice=band_slice,
        dim=base_h.shape[-1],
        target_bands=target_bands,
    )
    low_matrix_loss_cfg = _low_subspace_matrix_loss_config(
        raw_cfg,
        default_slice=band_slice,
        dim=base_h.shape[-1],
        target_bands=target_bands,
    )
    shell_projected_matrix_loss_cfg = _shell_projected_matrix_loss_config(raw_cfg)
    shell_specs: list[dict[str, Any]] = []
    if shell_projected_matrix_loss_cfg is not None:
        qset1 = np.asarray(moire_config.Q_set1, dtype=float)
        qset2 = np.asarray(moire_config.Q_set2, dtype=float)
        shell_specs = _q_shell_row_indices(
            qset1,
            qset2,
            (int(moire_config.n_orb1), int(moire_config.n_orb2)),
            max_shells=shell_projected_matrix_loss_cfg.get("max_shells"),
            subspace_fraction=float(shell_projected_matrix_loss_cfg["subspace_fraction"]),
            tol=float(shell_projected_matrix_loss_cfg["tol"]),
        )
        if not shell_specs:
            raise ValueError("fit.refine_bands.shell_projected_matrix_loss found no Q-shell rows")
    target_eig_full = None
    target_vec_full = None
    subspace_target_basis = None
    subspace_gap_weights = None
    low_matrix_target_basis = None
    low_matrix_gap_weights = None
    if subspace_loss_cfg is not None or low_matrix_loss_cfg is not None:
        target_eig_full, target_vec_full = np.linalg.eigh(heff_all)
    if subspace_loss_cfg is not None:
        s0, s1 = (int(value) for value in subspace_loss_cfg["band_slice"])
        assert target_vec_full is not None and target_eig_full is not None
        subspace_target_basis = target_vec_full[:, :, s0:s1]
        subspace_gap_weights = _gap_weights_from_eigvals(
            target_eig_full,
            (s0, s1),
            gap_tolerance_mev=float(subspace_loss_cfg["gap_tolerance_mev"]),
        )
    if low_matrix_loss_cfg is not None:
        m0, m1 = (int(value) for value in low_matrix_loss_cfg["band_slice"])
        assert target_vec_full is not None and target_eig_full is not None
        low_matrix_target_basis = target_vec_full[:, :, m0:m1]
        low_matrix_gap_weights = _gap_weights_from_eigvals(
            target_eig_full,
            (m0, m1),
            gap_tolerance_mev=float(low_matrix_loss_cfg["gap_tolerance_mev"]),
        )
    if "normalize_band_loss" in raw_cfg:
        normalize_band_loss = bool(raw_cfg.get("normalize_band_loss"))
    elif matrix_loss_cfg is not None:
        normalize_band_loss = bool(matrix_loss_cfg.get("normalize_band_loss", True))
    else:
        normalize_band_loss = False
    max_nfev = int(raw_cfg.get("max_nfev", 250))
    target_slice = heff_eig[:, band_slice[0] : band_slice[1]]
    weighted_band_raw = raw_cfg.get("weighted_band_loss", {})
    if weighted_band_raw is True:
        weighted_band_cfg: Mapping[str, Any] = {"enabled": True}
    elif weighted_band_raw in (False, None):
        weighted_band_cfg = {}
    elif isinstance(weighted_band_raw, Mapping):
        weighted_band_cfg = weighted_band_raw
    else:
        raise ValueError("fit.refine_bands.weighted_band_loss must be a mapping or boolean when provided")
    edge_band_weights = None
    weighted_band_report: dict[str, Any] = {"enabled": False}
    if bool(weighted_band_cfg.get("enabled", False)):
        primary_bands = int(weighted_band_cfg.get("primary_bands", max(1, min(int(sum(model_config.n_orb)), target_slice.shape[1]))))
        decay = float(weighted_band_cfg.get("decay", 0.65))
        floor = float(weighted_band_cfg.get("floor", 0.15))
        normalize_mean = bool(weighted_band_cfg.get("normalize_mean", True))
        edge_band_weights = _edge_weighted_band_weights(
            int(target_slice.shape[1]),
            primary_bands=primary_bands,
            target_bands=target_bands,
            decay=decay,
            floor=floor,
            normalize_mean=normalize_mean,
        )
        weighted_band_report = {
            "enabled": True,
            "primary_bands": int(primary_bands),
            "fit_bands": int(target_slice.shape[1]),
            "target_bands": target_bands,
            "decay": float(decay),
            "floor": float(floor),
            "normalize_mean": bool(normalize_mean),
            "weights": [float(value) for value in edge_band_weights],
            "weight_min": float(np.min(edge_band_weights)),
            "weight_max": float(np.max(edge_band_weights)),
            "weight_mean": float(np.mean(edge_band_weights)),
        }

    def h_from_y(y: np.ndarray) -> np.ndarray:
        return base_h + np.tensordot(np.asarray(y, dtype=float) - y0, basis, axes=(0, 0))

    def aligned_selected_eigs(eig: np.ndarray) -> np.ndarray:
        selected = eig[:, band_slice[0] : band_slice[1]]
        if align == "top":
            selected = selected + (target_slice[0, -1] - selected[0, -1])
        elif align == "bottom":
            selected = selected + (target_slice[0, 0] - selected[0, 0])
        elif align not in {"none", ""}:
            raise ValueError(f"fit.refine_bands.align currently supports 'top', 'bottom', or 'none', got {align!r}")
        return selected

    def selected_eigs_from_y(y: np.ndarray) -> np.ndarray:
        return aligned_selected_eigs(np.linalg.eigvalsh(h_from_y(y)))

    if solver in {"linear_low_subspace", "low_subspace_linear", "linear_projected_matrix"}:
        target_eig_full, target_vec_full = np.linalg.eigh(heff_all)
        s0, s1 = (int(value) for value in band_slice)
        subspace_dim = int(s1 - s0)
        if subspace_dim <= 0:
            raise ValueError(f"linear_low_subspace solver requires a non-empty band slice, got {band_slice}")
        if edge_band_weights is None:
            subspace_weights = np.ones(subspace_dim, dtype=float)
        else:
            subspace_weights = np.asarray(edge_band_weights, dtype=float)
            if subspace_weights.shape != (subspace_dim,):
                raise ValueError(
                    "linear_low_subspace weighted band shape mismatch: "
                    f"weights={subspace_weights.shape}, band_slice={band_slice}"
                )
        weight_matrix = np.sqrt(np.outer(subspace_weights, subspace_weights))
        n_complex = int(len(kpoints) * subspace_dim * subspace_dim)
        design_complex = np.empty((n_complex, len(kept_variables)), dtype=np.complex128)
        target_complex = np.empty(n_complex, dtype=np.complex128)
        row0 = 0
        delta_target = heff_all - base_h
        for k_index in range(len(kpoints)):
            target_basis = target_vec_full[k_index, :, s0:s1]
            target_block = target_basis.conj().T @ delta_target[k_index] @ target_basis
            target_complex[row0 : row0 + subspace_dim * subspace_dim] = (target_block * weight_matrix).reshape(-1)
            for var_index in range(len(kept_variables)):
                projected = target_basis.conj().T @ basis[var_index, k_index] @ target_basis
                design_complex[row0 : row0 + subspace_dim * subspace_dim, var_index] = (
                    projected * weight_matrix
                ).reshape(-1)
            row0 += subspace_dim * subspace_dim
        design_chunks = [design_complex]
        target_chunks = [target_complex]
        shell_linear_report: dict[str, Any] = {"enabled": False}
        if shell_projected_matrix_loss_cfg is not None:
            shell_design_chunks: list[np.ndarray] = []
            shell_target_chunks: list[np.ndarray] = []
            shell_rows_report: list[dict[str, Any]] = []
            shell_weight = float(shell_projected_matrix_loss_cfg["weight"])
            shell_decay = float(shell_projected_matrix_loss_cfg["shell_decay"])
            shell_window_cfg = dict(shell_projected_matrix_loss_cfg["window"])
            for shell in shell_specs:
                rows = np.asarray(shell["rows"], dtype=int)
                shell_dim = int(rows.size)
                target_shell_all = heff_all[:, rows[:, None], rows]
                shell_eig_all, shell_vec_all = np.linalg.eigh(target_shell_all)
                shell_window = _shell_band_window_from_target_eig(
                    shell_eig_all,
                    target_bands=target_bands,
                    window_config=shell_window_cfg,
                )
                n_shell_bands = int(shell_window["n_bands"])
                shell_slice = tuple(int(value) for value in shell_window["band_slice"])
                n_shell_complex = int(len(kpoints) * n_shell_bands * n_shell_bands)
                shell_design = np.empty((n_shell_complex, len(kept_variables)), dtype=np.complex128)
                shell_target = np.empty(n_shell_complex, dtype=np.complex128)
                scale_shell = np.sqrt(shell_weight * (shell_decay ** int(shell["shell_index"])))
                shell_row0 = 0
                for k_index in range(len(kpoints)):
                    target_shell = target_shell_all[k_index]
                    base_shell = base_h[k_index][rows[:, None], rows]
                    target_basis = shell_vec_all[k_index, :, shell_slice[0] : shell_slice[1]]
                    target_block = target_basis.conj().T @ (target_shell - base_shell) @ target_basis
                    block_size = n_shell_bands * n_shell_bands
                    shell_target[shell_row0 : shell_row0 + block_size] = (scale_shell * target_block).reshape(-1)
                    for var_index in range(len(kept_variables)):
                        basis_shell = basis[var_index, k_index][rows[:, None], rows]
                        projected = target_basis.conj().T @ basis_shell @ target_basis
                        shell_design[shell_row0 : shell_row0 + block_size, var_index] = (
                            scale_shell * projected
                        ).reshape(-1)
                    shell_row0 += block_size
                shell_design_chunks.append(shell_design)
                shell_target_chunks.append(shell_target)
                shell_rows_report.append(
                    {
                        "shell_index": int(shell["shell_index"]),
                        "dimension": shell_dim,
                        "subspace_bands": n_shell_bands,
                        "band_slice": [int(shell_slice[0]), int(shell_slice[1])],
                        "window": shell_window,
                        "complex_rows": int(n_shell_complex),
                        "weight": float(scale_shell**2),
                    }
                )
            if shell_design_chunks:
                design_chunks.extend(shell_design_chunks)
                target_chunks.extend(shell_target_chunks)
            shell_linear_report = {
                "enabled": True,
                "mode": "q_shell_projected_linear",
                "target_bands": target_bands,
                "weight": float(shell_projected_matrix_loss_cfg["weight"]),
                "shell_decay": float(shell_projected_matrix_loss_cfg["shell_decay"]),
                "subspace_fraction": float(shell_projected_matrix_loss_cfg["subspace_fraction"]),
                "window": shell_window_cfg,
                "source_key": str(shell_projected_matrix_loss_cfg.get("source_key", "shell_projected_matrix_loss")),
                "shells": shell_rows_report,
            }
        design_complex = np.vstack(design_chunks)
        target_complex = np.concatenate(target_chunks)
        design = np.vstack([design_complex.real, design_complex.imag])
        target = np.concatenate([target_complex.real, target_complex.imag])
        regularization = float(raw_cfg.get("regularization", raw_cfg.get("linear_regularization", 0.0)))
        if regularization < 0.0:
            raise ValueError("fit.refine_bands.regularization must be non-negative")
        if regularization > 0.0:
            design = np.vstack([design, np.sqrt(regularization) * np.diag(1.0 / scale)])
            target = np.concatenate([target, np.zeros(len(kept_variables), dtype=float)])
        delta, residuals, rank, singular_values = np.linalg.lstsq(
            design,
            target,
            rcond=float(raw_cfg.get("rcond", 1.0e-10)),
        )
        candidate_y = y0 + np.asarray(delta, dtype=float)
        current_y, acceptance_guard = _apply_refinement_acceptance_guard(
            raw_cfg=raw_cfg,
            model_config=model_config,
            base_h=base_h,
            heff_eig=heff_eig,
            heff_all=heff_all,
            y0=y0,
            candidate_y=candidate_y,
            h_from_y=h_from_y,
        )
        for (term, component), value in zip(kept_variables, current_y):
            _set_term_component_value(term, component, float(value))
        refined_h = h_from_y(current_y)
        refined_metrics = _band_refinement_metrics(refined_h, heff_eig, heff_all, band_slice=band_slice, align=align)
        shell_projected_matrix_report = {"enabled": False}
        if shell_projected_matrix_loss_cfg is not None:
            shell_projected_matrix_report = {
                **shell_linear_report,
                "initial": _shell_subspace_overlap_report(
                    base_h,
                    heff_all,
                    shell_specs,
                    target_bands=target_bands,
                    subspace_fraction=float(shell_projected_matrix_loss_cfg["subspace_fraction"]),
                    window_config=dict(shell_projected_matrix_loss_cfg["window"]),
                    align=align,
                ),
                "refined": _shell_subspace_overlap_report(
                    refined_h,
                    heff_all,
                    shell_specs,
                    target_bands=target_bands,
                    subspace_fraction=float(shell_projected_matrix_loss_cfg["subspace_fraction"]),
                    window_config=dict(shell_projected_matrix_loss_cfg["window"]),
                    align=align,
                ),
            }
        drift = np.abs((current_y - y0) / scale)
        variable_report = [
            {
                "tag": str(getattr(term, "tag", "")),
                "component": component,
                "term": _term_key_to_dict(getattr(term, "key", None)),
                "initial": float(initial),
                "refined": float(refined),
            }
            for (term, component), initial, refined in zip(kept_variables, y0, current_y)
        ]
        raw_tags = raw_cfg.get("variable_tags", ["Kinect", "Onsite", "inter"])
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        raw_components = raw_cfg.get("components", ["real"])
        if isinstance(raw_components, str):
            raw_components = [raw_components]
        return {
            "enabled": True,
            "solver": "linear_low_subspace",
            "fit_kpoints": fit_kpoints_report,
            "band_slice": [int(band_slice[0]), int(band_slice[1])],
            "align": align,
            "variable_tags": sorted({_refinement_tag_alias(str(tag)) for tag in raw_tags}),
            "components": [str(component).strip().lower() for component in raw_components],
            "n_variables": int(len(kept_variables)),
            "max_variables": int(max_variables) if max_variables is not None else None,
            "response_basis": response_build_report,
            "linear_system": {
                "rows": int(design.shape[0]),
                "cols": int(design.shape[1]),
                "rank": int(rank),
                "regularization": float(regularization),
                "residual_norm": float(np.sqrt(float(residuals[0]))) if np.size(residuals) else None,
                "condition_number": (
                    float(singular_values[0] / singular_values[-1])
                    if np.size(singular_values) and float(singular_values[-1]) > 0.0
                    else None
                ),
            },
            "weighted_band_loss": weighted_band_report,
            "matrix_loss": {"enabled": False},
            "subspace_loss": {"enabled": False},
            "shell_projected_matrix_loss": shell_projected_matrix_report,
            "shell_subspace_loss": {
                "enabled": bool(shell_projected_matrix_report.get("enabled", False)),
                "legacy_alias_of": "shell_projected_matrix_loss",
            },
            "low_subspace_matrix_loss": {
                "enabled": True,
                "band_slice": [int(s0), int(s1)],
                "mode": "projected_heff_subspace_linear",
            },
            "acceptance_guard": acceptance_guard if acceptance_guard is not None else {"enabled": False},
            "coefficient_weight": 0.0,
            "regularization": float(regularization),
            "max_nfev": 0,
            "nfev": 0,
            "primary_nfev": 0,
            "cost": None,
            "reweight": {"enabled": False, "rounds": []},
            "initial": initial_metrics,
            "refined": refined_metrics,
            "max_scaled_coefficient_drift": float(np.max(drift)) if drift.size else 0.0,
            "p95_scaled_coefficient_drift": float(np.quantile(drift, 0.95)) if drift.size else 0.0,
            "variables": variable_report,
        }

    jacobian_mode = str(raw_cfg.get("jacobian", "auto")).strip().lower()
    if jacobian_mode in {"", "true"}:
        jacobian_mode = "auto"
    if jacobian_mode in {"false", "off", "none", "finite_difference", "finite-difference", "fd"}:
        jacobian_mode = "finite_difference"
    if jacobian_mode not in {"auto", "analytic", "finite_difference"}:
        raise ValueError(
            "fit.refine_bands.jacobian must be 'auto', 'analytic', or 'finite_difference', "
            f"got {raw_cfg.get('jacobian')!r}"
        )
    analytic_jacobian_reasons: list[str] = []
    if subspace_loss_cfg is not None:
        analytic_jacobian_reasons.append("subspace_loss")
    if low_matrix_loss_cfg is not None:
        analytic_jacobian_reasons.append("low_subspace_matrix_loss")
    if matrix_loss_cfg is not None:
        analytic_jacobian_reasons.append("matrix_loss")
    if shell_projected_matrix_loss_cfg is not None:
        analytic_jacobian_reasons.append("shell_projected_matrix_loss")
    analytic_jacobian_enabled = jacobian_mode in {"auto", "analytic"} and not analytic_jacobian_reasons
    if jacobian_mode == "analytic" and analytic_jacobian_reasons:
        raise ValueError(
            "fit.refine_bands.jacobian='analytic' is not available with "
            + ", ".join(analytic_jacobian_reasons)
        )
    matrix_jacobian = None
    matrix_loss_reduced = None
    matrix_compression_mode = str(raw_cfg.get("matrix_compression", "auto")).strip().lower()
    if matrix_compression_mode in {"", "true"}:
        matrix_compression_mode = "auto"
    if matrix_compression_mode in {"false", "off", "none", "disabled"}:
        matrix_compression_mode = "disabled"
    if matrix_compression_mode not in {"auto", "reduced", "disabled"}:
        raise ValueError(
            "fit.refine_bands.matrix_compression must be 'auto', 'reduced', or 'disabled', "
            f"got {raw_cfg.get('matrix_compression')!r}"
        )
    if analytic_jacobian_enabled and matrix_weight > 0.0:
        if matrix_compression_mode in {"auto", "reduced"}:
            matrix_compression_start = time.perf_counter()
            matrix_loss_reduced = _band_refinement_reduced_global_matrix_loss(
                basis,
                base_h - heff_all,
                matrix_weight=matrix_weight,
                matrix_sigma=matrix_sigma,
                rcond=float(raw_cfg.get("matrix_compression_rcond", 1.0e-12)),
            )
            print(
                "[kp model] refinement matrix compression: "
                f"{matrix_loss_reduced['source_rows']} -> {matrix_loss_reduced['compressed_rows']} rows, "
                f"rank={matrix_loss_reduced['rank']}, "
                f"{time.perf_counter() - matrix_compression_start:.2f} s",
                flush=True,
            )
        else:
            matrix_jacobian = _band_refinement_global_matrix_jacobian_sparse(
                basis,
                matrix_weight=matrix_weight,
                matrix_sigma=matrix_sigma,
            )
    coeff_jacobian = None
    if analytic_jacobian_enabled and coeff_weight > 0.0:
        if matrix_jacobian is not None:
            coeff_jacobian = scipy.sparse.diags(float(coeff_weight) * (1.0 / scale), format="csr")
        else:
            coeff_jacobian = float(coeff_weight) * np.diag(1.0 / scale)
    jacobian_storage = None
    if analytic_jacobian_enabled:
        jacobian_storage = "sparse_csr" if matrix_jacobian is not None else "dense_reduced"
    jacobian_report = {
        "enabled": bool(analytic_jacobian_enabled),
        "mode": "analytic_hellmann_feynman" if analytic_jacobian_enabled else "finite_difference",
        "storage": jacobian_storage,
        "source": jacobian_mode,
        "fallback_reasons": analytic_jacobian_reasons,
        "matrix_loss": (
            {
                "compressed": True,
                "mode": "coefficient_space_quadratic",
                "source_rows": int(matrix_loss_reduced["source_rows"]),
                "compressed_rows": int(matrix_loss_reduced["compressed_rows"]),
                "rank": int(matrix_loss_reduced["rank"]),
                "condition_number": matrix_loss_reduced["condition_number"],
                "rcond": float(matrix_loss_reduced["rcond"]),
            }
            if matrix_loss_reduced is not None
            else {"compressed": False, "mode": "explicit_sparse" if matrix_jacobian is not None else "not_used"}
        ),
    }

    def residual(y: np.ndarray, band_weights: np.ndarray | None = None) -> np.ndarray:
        need_eigenvectors = subspace_loss_cfg is not None or low_matrix_loss_cfg is not None
        h_current = None
        eig = None
        vec = None
        if need_eigenvectors:
            h_current = h_from_y(y)
            eig, vec = np.linalg.eigh(h_current)
            model_selected = aligned_selected_eigs(eig)
        else:
            model_selected = selected_eigs_from_y(y)
        band_resid = _band_refinement_band_residual(
            model_selected,
            target_slice,
            band_sigma=band_sigma,
            normalize=normalize_band_loss,
        )
        if edge_band_weights is not None:
            band_resid = band_resid * np.sqrt(edge_band_weights.reshape(1, -1))
        if band_weights is not None:
            band_resid = band_resid * np.sqrt(np.asarray(band_weights, dtype=float))
        parts = [band_resid.ravel()]
        if subspace_loss_cfg is not None:
            assert vec is not None
            s0, s1 = (int(value) for value in subspace_loss_cfg["band_slice"])
            assert subspace_target_basis is not None and subspace_gap_weights is not None
            subspace_resid, _subspace_report = _principal_angle_subspace_residual(
                vec[:, :, s0:s1],
                subspace_target_basis,
                weight=float(subspace_loss_cfg["weight"]),
                normalize=bool(subspace_loss_cfg["normalize"]),
                gap_weights=subspace_gap_weights,
            )
            parts.append(subspace_resid)
        if low_matrix_loss_cfg is not None:
            assert h_current is not None
            assert low_matrix_target_basis is not None and low_matrix_gap_weights is not None
            low_matrix_resid, _low_matrix_report = _low_subspace_matrix_residual(
                h_current - heff_all,
                low_matrix_target_basis,
                weight=float(low_matrix_loss_cfg["weight"]),
                sigma_mev=float(low_matrix_loss_cfg["sigma_mev"]),
                normalize=bool(low_matrix_loss_cfg["normalize"]),
                gap_weights=low_matrix_gap_weights,
            )
            parts.append(low_matrix_resid)
        if matrix_weight > 0.0:
            if matrix_loss_cfg is not None:
                matrix_delta = (h_current if h_current is not None else h_from_y(y)) - heff_all
                assert matrix_loss_masks is not None
                matrix_resid, _matrix_report = _matrix_loss_residual(
                    matrix_delta,
                    matrix_loss_masks,
                    matrix_loss_cfg,
                    report=False,
                )
                parts.append(matrix_resid)
            elif matrix_loss_reduced is not None:
                delta_y = (np.asarray(y, dtype=float) - y0) - np.asarray(matrix_loss_reduced["center_delta"], dtype=float)
                parts.append(np.asarray(matrix_loss_reduced["jacobian"], dtype=float) @ delta_y)
            else:
                matrix_delta = (h_current if h_current is not None else h_from_y(y)) - heff_all
                matrix_delta = matrix_delta.reshape(-1) / matrix_sigma
                parts.append(np.sqrt(matrix_weight) * np.concatenate([matrix_delta.real, matrix_delta.imag]))
        elif matrix_loss_cfg is not None:
            matrix_delta = (h_current if h_current is not None else h_from_y(y)) - heff_all
            assert matrix_loss_masks is not None
            matrix_resid, _matrix_report = _matrix_loss_residual(
                matrix_delta,
                matrix_loss_masks,
                matrix_loss_cfg,
                report=False,
            )
            parts.append(matrix_resid)
        if coeff_weight > 0.0:
            parts.append(float(coeff_weight) * ((np.asarray(y, dtype=float) - y0) / scale))
        return np.concatenate(parts)

    def jacobian(y: np.ndarray, band_weights: np.ndarray | None = None) -> np.ndarray:
        if not analytic_jacobian_enabled:
            raise RuntimeError("analytic band-refinement Jacobian is not enabled")
        h_current = h_from_y(y)
        _eig, vec = np.linalg.eigh(h_current)
        band_jacobian = _band_refinement_eigenvalue_jacobian(
            vec,
            basis,
            band_slice=band_slice,
            align=align,
            band_sigma=band_sigma,
            normalize=normalize_band_loss,
            edge_band_weights=edge_band_weights,
            band_weights=band_weights,
        )
        if matrix_jacobian is not None:
            parts = [scipy.sparse.csr_matrix(band_jacobian), matrix_jacobian]
            if coeff_jacobian is not None:
                parts.append(coeff_jacobian)
            return scipy.sparse.vstack(parts, format="csr")
        parts = [band_jacobian]
        if matrix_loss_reduced is not None:
            parts.append(np.asarray(matrix_loss_reduced["jacobian"], dtype=float))
        if coeff_jacobian is not None:
            parts.append(np.asarray(coeff_jacobian, dtype=float))
        return np.vstack(parts)

    least_squares_kwargs: dict[str, Any] = {
        "method": str(raw_cfg.get("method", "trf")),
        "max_nfev": max_nfev,
        "xtol": float(raw_cfg.get("xtol", 1.0e-10)),
        "ftol": float(raw_cfg.get("ftol", 1.0e-10)),
        "gtol": float(raw_cfg.get("gtol", 1.0e-10)),
    }
    optimizer_mode = str(raw_cfg.get("optimizer", "auto")).strip().lower()
    if optimizer_mode in {"", "true"}:
        optimizer_mode = "auto"
    if optimizer_mode in {"least_squares", "scipy", "scipy-least-squares"}:
        optimizer_mode = "scipy_least_squares"
    if optimizer_mode not in {"auto", "gauss_newton", "scipy_least_squares"}:
        raise ValueError(
            "fit.refine_bands.optimizer must be 'auto', 'gauss_newton', or 'scipy_least_squares', "
            f"got {raw_cfg.get('optimizer')!r}"
        )
    use_gauss_newton = (
        optimizer_mode in {"auto", "gauss_newton"}
        and analytic_jacobian_enabled
        and matrix_loss_reduced is not None
        and matrix_jacobian is None
    )
    if optimizer_mode == "gauss_newton" and not use_gauss_newton:
        raise ValueError("fit.refine_bands.optimizer='gauss_newton' requires analytic reduced Jacobian support")
    if analytic_jacobian_enabled:
        least_squares_kwargs["jac"] = lambda y: jacobian(y)
        if matrix_jacobian is not None and least_squares_kwargs["method"] != "lm":
            least_squares_kwargs["tr_solver"] = str(raw_cfg.get("tr_solver", "lsmr"))
    if use_gauss_newton:
        optimizer_start = time.perf_counter()
        result = _solve_band_refinement_gauss_newton(
            lambda y: residual(y),
            lambda y: jacobian(y),
            y0,
            max_nfev=max_nfev,
            xtol=float(raw_cfg.get("xtol", 1.0e-10)),
            ftol=float(raw_cfg.get("ftol", 1.0e-10)),
            gtol=float(raw_cfg.get("gtol", 1.0e-10)),
            initial_damping=float(raw_cfg.get("gauss_newton_damping", 1.0e-6)),
            max_line_search_steps=int(raw_cfg.get("gauss_newton_line_search_steps", 8)),
        )
        print(
            "[kp model] refinement optimizer: "
            f"gauss_newton nfev={result.nfev} njev={result.njev} "
            f"cost={result.cost:.6g}, {time.perf_counter() - optimizer_start:.2f} s",
            flush=True,
        )
    else:
        optimizer_start = time.perf_counter()
        result = scipy.optimize.least_squares(
            lambda y: residual(y),
            y0,
            **least_squares_kwargs,
        )
        print(
            "[kp model] refinement optimizer: "
            f"scipy_least_squares nfev={int(result.nfev)} "
            f"cost={float(result.cost):.6g}, {time.perf_counter() - optimizer_start:.2f} s",
            flush=True,
        )
    current_y = np.asarray(result.x, dtype=float)
    total_nfev = int(result.nfev)
    total_njev = int(getattr(result, "njev", 0) or 0)
    reweight_raw = raw_cfg.get("reweight", {})
    if reweight_raw is True:
        reweight_cfg: Mapping[str, Any] = {"rounds": 1}
    elif reweight_raw in (False, None):
        reweight_cfg = {}
    elif isinstance(reweight_raw, Mapping):
        reweight_cfg = reweight_raw
    else:
        raise ValueError("fit.refine_bands.reweight must be a mapping or boolean when provided")
    reweight_rounds = int(reweight_cfg.get("rounds", 0) or 0)
    reweight_reports: list[dict[str, Any]] = []
    if reweight_rounds < 0:
        raise ValueError("fit.refine_bands.reweight.rounds must be non-negative")
    if reweight_rounds:
        threshold = float(reweight_cfg.get("threshold_mev", reweight_cfg.get("threshold", 1.0)))
        alpha = float(reweight_cfg.get("alpha", 2.0))
        power = float(reweight_cfg.get("power", 2.0))
        cap = float(reweight_cfg.get("cap", 10.0))
        reweight_max_nfev = int(reweight_cfg.get("max_nfev", max_nfev))
        if threshold <= 0.0:
            raise ValueError("fit.refine_bands.reweight.threshold_mev must be positive")
        if alpha < 0.0 or power <= 0.0 or cap < 1.0:
            raise ValueError("fit.refine_bands.reweight requires alpha>=0, power>0, cap>=1")
        for round_index in range(reweight_rounds):
            eig_current = selected_eigs_from_y(current_y)
            diff_mev = (eig_current - target_slice) * 1000.0
            weights = 1.0 + alpha * np.power(np.abs(diff_mev) / threshold, power)
            weights = np.clip(weights, 1.0, cap)
            before = _band_refinement_metrics(h_from_y(current_y), heff_eig, heff_all, band_slice=band_slice, align=align)
            reweight_kwargs: dict[str, Any] = {
                "method": str(reweight_cfg.get("method", raw_cfg.get("method", "trf"))),
                "max_nfev": reweight_max_nfev,
                "xtol": float(reweight_cfg.get("xtol", raw_cfg.get("xtol", 1.0e-10))),
                "ftol": float(reweight_cfg.get("ftol", raw_cfg.get("ftol", 1.0e-10))),
                "gtol": float(reweight_cfg.get("gtol", raw_cfg.get("gtol", 1.0e-10))),
            }
            if analytic_jacobian_enabled:
                reweight_kwargs["jac"] = lambda y, w=weights: jacobian(y, w)
                if matrix_jacobian is not None and reweight_kwargs["method"] != "lm":
                    reweight_kwargs["tr_solver"] = str(reweight_cfg.get("tr_solver", raw_cfg.get("tr_solver", "lsmr")))
            if use_gauss_newton:
                result = _solve_band_refinement_gauss_newton(
                    lambda y, w=weights: residual(y, w),
                    lambda y, w=weights: jacobian(y, w),
                    current_y,
                    max_nfev=reweight_max_nfev,
                    xtol=float(reweight_cfg.get("xtol", raw_cfg.get("xtol", 1.0e-10))),
                    ftol=float(reweight_cfg.get("ftol", raw_cfg.get("ftol", 1.0e-10))),
                    gtol=float(reweight_cfg.get("gtol", raw_cfg.get("gtol", 1.0e-10))),
                    initial_damping=float(reweight_cfg.get("gauss_newton_damping", raw_cfg.get("gauss_newton_damping", 1.0e-6))),
                    max_line_search_steps=int(
                        reweight_cfg.get("gauss_newton_line_search_steps", raw_cfg.get("gauss_newton_line_search_steps", 8))
                    ),
                )
            else:
                result = scipy.optimize.least_squares(
                    lambda y, w=weights: residual(y, w),
                    current_y,
                    **reweight_kwargs,
                )
            current_y = np.asarray(result.x, dtype=float)
            total_nfev += int(result.nfev)
            total_njev += int(getattr(result, "njev", 0) or 0)
            after = _band_refinement_metrics(h_from_y(current_y), heff_eig, heff_all, band_slice=band_slice, align=align)
            reweight_reports.append(
                {
                    "round": int(round_index + 1),
                    "nfev": int(result.nfev),
                    "cost": float(result.cost),
                    "before": before,
                    "after": after,
                    "weight_stats": {
                        "max": float(np.max(weights)),
                        "p99": float(np.quantile(weights, 0.99)),
                        "p95": float(np.quantile(weights, 0.95)),
                        "mean": float(np.mean(weights)),
                    },
                }
            )
    current_y, acceptance_guard = _apply_refinement_acceptance_guard(
        raw_cfg=raw_cfg,
        model_config=model_config,
        base_h=base_h,
        heff_eig=heff_eig,
        heff_all=heff_all,
        y0=y0,
        candidate_y=current_y,
        h_from_y=h_from_y,
    )
    for (term, component), value in zip(kept_variables, current_y):
        _set_term_component_value(term, component, float(value))

    refined_metrics = _band_refinement_metrics(h_from_y(current_y), heff_eig, heff_all, band_slice=band_slice, align=align)
    matrix_loss_report = None
    subspace_loss_report = None
    low_matrix_loss_report = None
    refined_h = h_from_y(current_y)
    if matrix_loss_cfg is not None:
        assert matrix_loss_masks is not None
        _initial_vec, initial_matrix_loss = _matrix_loss_residual(base_h - heff_all, matrix_loss_masks, matrix_loss_cfg)
        _refined_vec, refined_matrix_loss = _matrix_loss_residual(refined_h - heff_all, matrix_loss_masks, matrix_loss_cfg)
        matrix_loss_report = {
            "enabled": True,
            "initial": initial_matrix_loss,
            "refined": refined_matrix_loss,
        }
    if subspace_loss_cfg is not None:
        assert subspace_target_basis is not None and subspace_gap_weights is not None
        s0, s1 = (int(value) for value in subspace_loss_cfg["band_slice"])
        _base_w, base_v = np.linalg.eigh(base_h)
        _ref_w, refined_v = np.linalg.eigh(refined_h)
        _initial_vec, initial_subspace = _principal_angle_subspace_residual(
            base_v[:, :, s0:s1],
            subspace_target_basis,
            weight=float(subspace_loss_cfg["weight"]),
            normalize=bool(subspace_loss_cfg["normalize"]),
            gap_weights=subspace_gap_weights,
        )
        _refined_vec, refined_subspace = _principal_angle_subspace_residual(
            refined_v[:, :, s0:s1],
            subspace_target_basis,
            weight=float(subspace_loss_cfg["weight"]),
            normalize=bool(subspace_loss_cfg["normalize"]),
            gap_weights=subspace_gap_weights,
        )
        subspace_loss_report = {
            "enabled": True,
            "band_slice": [int(s0), int(s1)],
            "initial": initial_subspace,
            "refined": refined_subspace,
        }
    if low_matrix_loss_cfg is not None:
        assert low_matrix_target_basis is not None and low_matrix_gap_weights is not None
        _initial_vec, initial_low_matrix = _low_subspace_matrix_residual(
            base_h - heff_all,
            low_matrix_target_basis,
            weight=float(low_matrix_loss_cfg["weight"]),
            sigma_mev=float(low_matrix_loss_cfg["sigma_mev"]),
            normalize=bool(low_matrix_loss_cfg["normalize"]),
            gap_weights=low_matrix_gap_weights,
        )
        _refined_vec, refined_low_matrix = _low_subspace_matrix_residual(
            refined_h - heff_all,
            low_matrix_target_basis,
            weight=float(low_matrix_loss_cfg["weight"]),
            sigma_mev=float(low_matrix_loss_cfg["sigma_mev"]),
            normalize=bool(low_matrix_loss_cfg["normalize"]),
            gap_weights=low_matrix_gap_weights,
        )
        low_matrix_loss_report = {
            "enabled": True,
            "band_slice": list(low_matrix_loss_cfg["band_slice"]),
            "initial": initial_low_matrix,
            "refined": refined_low_matrix,
        }
    drift = np.abs((current_y - y0) / scale)
    variable_report = [
        {
            "tag": str(getattr(term, "tag", "")),
            "component": component,
            "term": _term_key_to_dict(getattr(term, "key", None)),
            "initial": float(initial),
            "refined": float(refined),
        }
        for (term, component), initial, refined in zip(kept_variables, y0, current_y)
    ]
    raw_tags = raw_cfg.get("variable_tags", ["Kinect", "Onsite", "inter"])
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    raw_components = raw_cfg.get("components", ["real"])
    if isinstance(raw_components, str):
        raw_components = [raw_components]
    return {
        "enabled": True,
        "fit_kpoints": fit_kpoints_report,
        "band_slice": [int(band_slice[0]), int(band_slice[1])],
        "align": align,
        "variable_tags": sorted({_refinement_tag_alias(str(tag)) for tag in raw_tags}),
        "components": [str(component).strip().lower() for component in raw_components],
        "n_variables": int(len(kept_variables)),
        "max_variables": int(max_variables) if max_variables is not None else None,
        "response_basis": response_build_report,
        "band_sigma_mev": float(band_sigma * 1000.0),
        "normalize_band_loss": bool(normalize_band_loss),
        "weighted_band_loss": weighted_band_report,
        "matrix_weight": float(matrix_weight),
        "matrix_sigma_mev": float(matrix_sigma * 1000.0),
        "matrix_loss": matrix_loss_report if matrix_loss_report is not None else {"enabled": False},
        "subspace_loss": subspace_loss_report if subspace_loss_report is not None else {"enabled": False},
        "low_subspace_matrix_loss": low_matrix_loss_report if low_matrix_loss_report is not None else {"enabled": False},
        "jacobian": jacobian_report,
        "acceptance_guard": acceptance_guard if acceptance_guard is not None else {"enabled": False},
        "coefficient_weight": float(coeff_weight),
        "max_nfev": int(max_nfev),
        "nfev": int(total_nfev),
        "njev": int(total_njev),
        "primary_nfev": int(total_nfev - sum(int(item["nfev"]) for item in reweight_reports)),
        "cost": float(result.cost),
        "optimizer": {
            "mode": "gauss_newton" if use_gauss_newton else "scipy_least_squares",
            "requested": optimizer_mode,
            "status": int(getattr(result, "status", 0) or 0),
            "message": str(getattr(result, "message", "")),
        },
        "reweight": {
            "enabled": bool(reweight_rounds),
            "rounds": reweight_reports,
        },
        "initial": initial_metrics,
        "refined": refined_metrics,
        "max_scaled_coefficient_drift": float(np.max(drift)) if drift.size else 0.0,
        "p95_scaled_coefficient_drift": float(np.quantile(drift, 0.95)) if drift.size else 0.0,
        "variables": variable_report,
    }


def _run_model_pipeline(
    moire_config: MoireConfig,
    model_config: ConfiguredModel,
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
    diagnostics = None
    null_filter = _null_channel_filter_summary(
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

        model, diagnostics = run_stage(
            "fitting coefficients",
            lambda: compute_coefficients(moire_config, model, progress_callback=fit_progress),
        )
        null_filter = _null_channel_filter_summary(
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
                lambda: refine_band_coefficients(moire_config, model_config, model),
            )
            refined = refinement.get("refined", {})
            if refined:
                _progress_line(
                    "band refinement: top RMS={top_band_rms_mev:.3f} meV, "
                    "top Max={top_band_max_mev:.3f} meV".format(**refined),
                    enabled=progress,
                )
            guard = refinement.get("acceptance_guard", {})
            if guard.get("enabled") and not bool(guard.get("accepted", True)):
                failed = ", ".join(str(item) for item in guard.get("failed_windows", [])) or "validation"
                _progress_line(
                    f"band refinement rejected by acceptance guard; reverted=True; failed windows: {failed}",
                    enabled=progress,
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
    }


def _model_config_for_fit_candidate(
    model_config: ConfiguredModel,
    candidate: Mapping[str, Any],
    *,
    scan_report: Mapping[str, Any] | None = None,
    disable_refinement: bool = False,
) -> ConfiguredModel:
    indices = [int(idx) for idx in candidate.get("indices", [])]
    metadata = {
        **dict(model_config.fit_selection_metadata),
        "source": "auto_low_energy_candidate_scan",
        "selected_candidate": str(candidate.get("name", "candidate")),
        "selected_indices": indices,
        "selected_candidate_reason": candidate.get("reason"),
    }
    if scan_report is not None:
        metadata["candidate_scan"] = _json_safe(scan_report)
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
    model_config: ConfiguredModel,
    *,
    kpoints_all: np.ndarray,
    heff_list: np.ndarray,
    indices: Sequence[int],
) -> MoireConfig:
    out = copy.copy(moire_config)
    out.kpoints_fit = _select_rows(kpoints_all, indices)
    out.heff = _block_diag_heff(heff_list, indices)
    out.output_dir = None
    return out


def _run_auto_low_energy_fit_candidate_scan(
    *,
    moire_config: MoireConfig,
    model_config: ConfiguredModel,
    output_dir: Path,
    log_path: Path,
    verbose: bool,
    progress: bool,
) -> tuple[dict[str, Any], MoireConfig, ConfiguredModel] | None:
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

    kpoints_all = _load_kpoints(model_config)
    heff_list = np.load(model_config.heff_file, mmap_mode="r")
    if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
        heff_eig = np.load(model_config.heff_eig_file)
    else:
        heff_eig = np.linalg.eigvalsh(heff_list)
    heff_selected = _select_rows(heff_eig, model_config.band_indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_log_dir = output_dir / "candidate_logs"
    candidate_log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    runs: dict[str, tuple[dict[str, Any], MoireConfig, ConfiguredModel, Path]] = {}
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
            comparison = compare_bands(eigvals, heff_selected, band_slice=model_config.band_slice)
            plot_comparison = compare_bands_for_plot(
                eigvals,
                heff_selected,
                band_slice=model_config.band_slice,
                plot_config=model_config.band_plot_config,
            )
            record.update(
                {
                    "status": "ok",
                    "all_rms_mev": float(comparison["rms_error_mev"]),
                    "all_max_mev": float(comparison["max_abs_error_mev"]),
                    "plot_rms_mev": float(plot_comparison["rms_error_mev"]),
                    "plot_max_mev": float(plot_comparison["max_abs_error_mev"]),
                    "log": str(candidate_log.resolve()),
                }
            )
            runs[name] = (result, candidate_moire, candidate_model, candidate_log)
        except Exception as exc:  # pragma: no cover - exercised by external validation paths
            record.update({"status": "failed", "error": str(exc)})
        records.append(record)

    initial_ok_records = [row for row in records if row.get("status") == "ok"]
    if initial_ok_records and not bool(fit_cfg.get("disable_residual_augmented_fit", False)):
        preliminary_record, _preliminary_report = _choose_auto_fit_candidate_record(initial_ok_records)
        preliminary_run = runs.get(str(preliminary_record.get("name", "")))
        if preliminary_run is not None:
            preliminary_result = preliminary_run[0]
            preliminary_eigvals = (
                preliminary_result["eigvals"][0]
                if isinstance(preliminary_result["eigvals"], tuple)
                else preliminary_result["eigvals"]
            )
            residual_scores = _band_residual_scores_for_plot(
                preliminary_eigvals,
                heff_selected,
                band_slice=model_config.band_slice,
                plot_config=model_config.band_plot_config,
            )
            residual_candidates = _auto_low_energy_fit_candidate_sets(
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
                    comparison = compare_bands(eigvals, heff_selected, band_slice=model_config.band_slice)
                    plot_comparison = compare_bands_for_plot(
                        eigvals,
                        heff_selected,
                        band_slice=model_config.band_slice,
                        plot_config=model_config.band_plot_config,
                    )
                    record.update(
                        {
                            "status": "ok",
                            "all_rms_mev": float(comparison["rms_error_mev"]),
                            "all_max_mev": float(comparison["max_abs_error_mev"]),
                            "plot_rms_mev": float(plot_comparison["rms_error_mev"]),
                            "plot_max_mev": float(plot_comparison["max_abs_error_mev"]),
                            "log": str(candidate_log.resolve()),
                        }
                    )
                    runs[name] = (result, candidate_moire, candidate_model, candidate_log)
                except Exception as exc:  # pragma: no cover - exercised by external validation paths
                    record.update({"status": "failed", "error": str(exc)})
                records.append(record)

    ok_records = [row for row in records if row.get("status") == "ok"]
    if not ok_records:
        raise RuntimeError(f"auto_low_energy fit candidate scan failed for all candidates: {records!r}")
    selected_record, report = _choose_auto_fit_candidate_record(ok_records)
    report = {
        **report,
        "enabled": True,
        "mode": "prefit_full_model_scan",
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


def run_configured_model(path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    _progress_line("loading configuration ...", enabled=True, style="start")
    moire_config, model_config = build_moire_config_from_file(path)
    output_dir = model_config.output_dir
    output_profile = _model_output_profile(model_config)
    canonical_output = isinstance(model_config.raw.get("case"), Mapping) and bool(model_config.raw["case"].get("profile")) and bool(model_config.raw["case"].get("q_shell"))
    diagnostics_dir = _model_diagnostics_dir(output_dir) if output_profile == "debug" else None
    output_dir.mkdir(parents=True, exist_ok=True)
    _progress_line(f"output directory: {output_dir}", enabled=True, style="path")
    _cleanup_stale_band_outputs(output_dir)
    _cleanup_stale_model_debug_outputs(output_dir)
    _run_model_harmonic_recommendation_once(model_config=model_config, output_dir=output_dir)
    if diagnostics_dir is not None:
        save_bM_diagnostics(model_config=model_config, output_dir=diagnostics_dir)
        save_harmonics_diagnostics(moire_config=moire_config, model_config=model_config, output_dir=diagnostics_dir)
    verbose = bool(model_config.output_config.get("verbose", False))
    progress = bool(model_config.output_config.get("progress", True))
    log_name = str(model_config.output_config.get("log_file", "model_run.log"))
    log_path = output_dir / log_name
    moire_config.output_dir = None
    try:
        scan_result = _run_auto_low_energy_fit_candidate_scan(
            moire_config=moire_config,
            model_config=model_config,
            output_dir=output_dir,
            log_path=log_path,
            verbose=verbose,
            progress=progress,
        )
        if scan_result is None:
            results = _run_model_pipeline(moire_config, model_config, log_path, verbose=verbose, progress=progress)
        else:
            results, moire_config, model_config = scan_result
    finally:
        moire_config.output_dir = output_dir
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
    band_plot_path = None
    all_band_plot_path = None
    hamiltonian_element_plot_path = None
    q_lattice_plot_path = save_q_lattice_harmonics_plot(
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
        heff_eigvecs = None
        heff_matrix_all = np.load(model_config.heff_file, mmap_mode="r")
        if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
            heff_eig = np.load(model_config.heff_eig_file)
        else:
            heff_eig = np.linalg.eigvalsh(heff_matrix_all)
        if model_eigvecs is not None and model_config.heff_file is not None and model_config.heff_file.exists():
            _, heff_eigvecs = np.linalg.eigh(heff_matrix_all)
        heff_selected = _select_rows(heff_eig, model_config.band_indices)
        heff_matrix_selected = _select_rows(heff_matrix_all, model_config.band_indices)
        overlap_weights = None
        if model_eigvecs is not None and heff_eigvecs is not None:
            model_eigvecs_selected = _select_rows(np.asarray(model_eigvecs), model_config.band_indices)
            heff_eigvecs_selected = _select_rows(heff_eigvecs, model_config.band_indices)
            if model_eigvecs_selected.shape == heff_eigvecs_selected.shape:
                overlap_weights = _band_overlap_weights(
                    model_eigvecs_selected,
                    heff_eigvecs_selected,
                    model_eigvals=np.asarray(eigvals_array, dtype=float),
                    reference_eigvals=np.asarray(heff_selected, dtype=float),
                    degeneracy_tol=float(model_config.band_plot_config.get("overlap_degeneracy_tol", 3.0e-3)),
                )
        comparison = compare_bands(eigvals_array, heff_selected, band_slice=model_config.band_slice)
        if diagnostics_dir is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            with (diagnostics_dir / "comparison.json").open("w", encoding="utf-8") as handle:
                json.dump(comparison, handle, indent=2)
        target_bands = str(model_config.raw.get("model", {}).get("target_bands", "top")).strip().lower()
        window_plot_config = _window_band_plot_config(model_config.band_plot_config, target_bands=target_bands)
        band_plot_config = dict(window_plot_config)
        if overlap_weights is not None:
            band_plot_config["top_bands"] = 6
            band_plot_config["plot_all_bands"] = True
        plot_comparison = compare_bands_for_plot(
            eigvals_array,
            heff_selected,
            band_slice=model_config.band_slice,
            plot_config=band_plot_config,
        )
        plot_comparison_name = _plot_comparison_filename(band_plot_config)
        if diagnostics_dir is not None:
            with (diagnostics_dir / "comparison_plot.json").open("w", encoding="utf-8") as handle:
                json.dump(plot_comparison, handle, indent=2)
            with (diagnostics_dir / plot_comparison_name).open("w", encoding="utf-8") as handle:
                json.dump(plot_comparison, handle, indent=2)
        x_values, x_ticks, x_ticklabels = _plot_axis_from_kpath(model_config, np.asarray(eigvals_array).shape[0])
        band_plot_path = save_band_comparison_plot(
            eigvals_array,
            heff_selected,
            output_dir / "band_comparison.pdf",
            band_slice=model_config.band_slice,
            plot_config=band_plot_config,
            x=x_values,
            x_ticks=x_ticks,
            x_ticklabels=x_ticklabels,
            title=_band_plot_title(model_config),
            overlap_weights=overlap_weights,
        )
        all_band_config = _all_band_plot_config(model_config.band_plot_config)
        all_band_plot_comparison = compare_bands_for_plot(
            eigvals_array,
            heff_selected,
            band_slice=None,
            plot_config=all_band_config,
        )
        if diagnostics_dir is not None:
            with (diagnostics_dir / "comparison_all_bands.json").open("w", encoding="utf-8") as handle:
                json.dump(all_band_plot_comparison, handle, indent=2)
        all_band_plot_path = save_band_comparison_plot(
            eigvals_array,
            heff_selected,
            output_dir / "band_comparison_all.pdf",
            band_slice=None,
            plot_config=all_band_config,
            x=x_values,
            x_ticks=x_ticks,
            x_ticklabels=x_ticklabels,
            title=f"{_band_plot_title(model_config)} (all bands)",
        )
        try:
            model_hamiltonians = np.asarray(results.get("band_hamiltonians"), dtype=np.complex128)
            fit_positions, _holdout_positions = _selected_index_positions(model_config, int(heff_matrix_selected.shape[0]))
            selected_indices = list(model_config.band_indices or list(range(int(heff_matrix_selected.shape[0]))))
            fit_k_indices = [int(selected_indices[pos]) for pos in fit_positions]
            harmonic_mask = _selected_harmonic_support_mask(
                np.asarray(moire_config.Q_set1, dtype=float),
                np.asarray(moire_config.Q_set2, dtype=float),
                n_orb=(int(model_config.n_orb[0]), int(model_config.n_orb[1])),
                current_counts=_resolved_harmonic_count_limits(model_config.harmonics_config),
            )
            hamiltonian_element_plot_path = save_hamiltonian_element_comparison_plot(
                model_hamiltonians,
                heff_matrix_selected,
                output_dir / "hamiltonian_element_comparison.png",
                harmonic_mask=harmonic_mask,
                positions=fit_positions,
                k_indices=fit_k_indices,
                title=f"{_band_plot_title(model_config)} Hamiltonian elements",
                qset1=np.asarray(moire_config.Q_set1, dtype=float),
                qset2=np.asarray(moire_config.Q_set2, dtype=float),
                n_orb=(int(model_config.n_orb[0]), int(model_config.n_orb[1])),
                subtract_layer_diagonal_mean=True,
            )
        except Exception as exc:
            _progress_line(f"Hamiltonian element comparison skipped: {exc}", enabled=progress, style="warning")
    results["configured_model"] = model_config
    results["moire_config"] = moire_config
    results["comparison"] = comparison
    results["plot_comparison"] = plot_comparison
    results["all_band_plot_comparison"] = all_band_plot_comparison
    results["q_lattice_plot"] = str(q_lattice_plot_path.resolve())
    if band_plot_path is not None:
        results["band_plot"] = str(band_plot_path.resolve())
    if all_band_plot_path is not None:
        results["all_band_plot"] = str(all_band_plot_path.resolve())
    if hamiltonian_element_plot_path is not None:
        results["hamiltonian_element_plot"] = str(hamiltonian_element_plot_path.resolve())
        results["hamiltonian_element_plot_pdf"] = str(hamiltonian_element_plot_path.with_suffix(".pdf").resolve())
    _progress_line("validating fitted model ...", enabled=progress)
    validation_started = time.perf_counter()
    validations, validation_summary = _compute_validation_outputs(
        results=results,
        model_config=model_config,
        moire_config=moire_config,
    )
    _progress_line(
        f"validating fitted model done in {time.perf_counter() - validation_started:.2f} s",
        enabled=progress,
    )
    results["validations"] = validations
    results["validation_summary"] = validation_summary
    results["auto_model_selection"] = _write_auto_model_selection_outputs(
        results=results,
        output_dir=output_dir,
        model_config=model_config,
        moire_config=moire_config,
    )
    _write_model_registry_outputs(
        results=results,
        output_dir=output_dir,
        model_config=model_config,
        validations=validations,
        validation_summary=validation_summary,
        diagnostics_dir=diagnostics_dir,
    )
    results["model_log"] = str(log_path)
    results["runtime_s"] = float(time.perf_counter() - started)
    return results
