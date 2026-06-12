from __future__ import annotations

import ast
import copy
import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
import yaml

from .schema import (
    canonical_operation_name_for_valley,
    canonical_source_operation_name_for_valley,
    effective_operation_metadata_for_valley,
    validate_model_config,
)
from .symmetry import load_symmetry_source
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
GAMMA_LEGACY_ORDER_ALIASES = {
    "tunneling_zero": "inter",
    "tunneling_nonzero": "intra",
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
    band_indices: list[int] | None
    n_orb: tuple[int, int]
    nlow_state: list[int]
    bM_config: dict[str, Any]
    harmonics_config: dict[str, Any]
    max_order: dict[str, int]
    symmetry_map: dict[str, list[dict[str, Any]]]
    coeff_tol: float
    compare_to_heff: bool
    kpath_config: dict[str, Any] = field(default_factory=dict)
    valley_model: dict[str, Any] = field(default_factory=dict)
    symmetry_source_config: dict[str, Any] = field(default_factory=dict)
    symmetry_source_metadata: dict[str, Any] = field(default_factory=dict)
    sectors_config: list[dict[str, Any]] = field(default_factory=list)
    term_templates: list[dict[str, Any]] = field(default_factory=list)
    output_config: dict[str, Any] = field(default_factory=dict)
    band_slice: list[int] | None = None
    band_plot_config: dict[str, Any] = field(default_factory=dict)
    bM_diagnostics: dict[str, Any] = field(default_factory=dict)
    harmonics_diagnostics: dict[str, Any] = field(default_factory=dict)
    validation_config: dict[str, Any] = field(default_factory=dict)


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
    if spin == "spin_down_projected":
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

    model = out.get("model", {})
    if isinstance(model, Mapping) and "symmetry_map" not in model:
        model_out = dict(model)
        model_out["symmetry_map"] = _default_symmetry_map_from_valley_model(out.get("valley_model", {}))
        out["model"] = model_out
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
        valley_model = raw.get("valley_model", {})
        out.setdefault("use", "raw")
        out.setdefault("operations", list(valley_model.get("allowed_internal_symmetries", [])))
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
) -> dict[str, Any]:
    row = dict(operation) if isinstance(operation, Mapping) else {"name": str(operation)}
    user_name = str(row.get("name", row.get("operation", "")))
    name = canonical_source_operation_name_for_valley(user_name, valley_model)
    source_operation = str(row.get("operation", _default_source_operation_name(name)))
    if matrix_kind != "action":
        raise ValueError("kp_symm_output production symmetry_source.matrix_kind must be 'action'")
    semantics = SOURCE_SEMANTICS
    missing_action = [field for field in ("antiunitary", "k_map", "sector_map") if field not in row]
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
        **copy.deepcopy(dict(semantics)),
    }
    completed.update(row)
    completed["name"] = name
    completed.setdefault("operation", source_operation)
    completed["antiunitary_convention"] = "U_K" if completed["antiunitary"] else "none"
    if "q_map" not in completed and "k_map" in completed:
        completed["q_map"] = copy.deepcopy(completed["k_map"])
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
        operations = valley_model.get("allowed_internal_symmetries", [])
    if isinstance(operations, Mapping):
        operations = [
            {**(dict(value) if isinstance(value, Mapping) else {}), "name": str(key)}
            for key, value in operations.items()
        ]
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("symmetry_source.operations must be a list or mapping when provided")
    use = str(source_out.get("use", "raw"))
    matrix_kind_raw = source_out.get("matrix_kind", source_out.get("kind"))
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
            _complete_kp_symm_operation_entry(operation, valley_model, use=use, matrix_kind=matrix_kind)
            for operation in operations
        ]
    out["symmetry_source"] = source_out
    return out


def _max_derivative_order_values(model: Mapping[str, Any]) -> dict[str, int]:
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
            elif "k_map" in enriched:
                enriched["q_map"] = copy.deepcopy(enriched["k_map"])
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
                if "q_map" not in merged and "k_map" in merged:
                    merged["q_map"] = dict(merged["k_map"]) if isinstance(merged["k_map"], Mapping) else merged["k_map"]
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
    return None


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
    raise ValueError(
        "Configured model YAML requires a kp_symm symmetry manifest frame rotation. "
        "Rerun `kp symm` with automatic frame inference; do not hand-write model frame rotations."
    )


def load_model_config(path: str | Path) -> ConfiguredModel:
    cfg_path = Path(path).resolve()
    raw = _load_yaml(cfg_path)
    base = cfg_path.parent
    raw = dict(raw)
    validation = raw.get("validation", {})
    if validation is None:
        validation = {}
    if not isinstance(validation, Mapping):
        raise ValueError("validation section must be a mapping when provided")
    raw["validation"] = dict(validation)

    source_config = _resolve_path(raw.get("source_config"), base)
    if source_config is None:
        raise ValueError("Configured model YAML requires source_config")
    source_raw = _load_yaml(source_config)
    source_base = source_config.parent
    material = source_raw.get("material", {})
    project = source_raw.get("project", {})
    plot = source_raw.get("plot", {})
    if not isinstance(material, Mapping) or not isinstance(project, Mapping) or not isinstance(plot, Mapping):
        raise ValueError("source_config must contain mapping sections: material, project, plot")

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
        heff_file = project_out / "heff_list.npy"
    heff_eig_file = _resolve_path(_get_path_value(raw, "heff_eig_file"), base)
    if heff_eig_file is None:
        candidate = heff_file.with_name("heff_eig.npy")
        heff_eig_file = candidate if candidate.exists() else None

    kpoints_file = _resolve_path(_get_path_value(raw, "kpoints_file"), base)

    output_section = raw.get("output", {})
    if output_section is None:
        output_section = {}
    if not isinstance(output_section, Mapping):
        raise ValueError("output section must be a mapping")
    output_dir = _resolve_path(output_section.get("dir", "model"), base)
    if output_dir is None:
        raise ValueError("Failed to resolve output directory")

    model_raw = dict(_model_section(raw))
    valley_model_raw = raw.get("valley_model", {})
    if not isinstance(valley_model_raw, Mapping):
        valley_model_raw = {}
    symmetry_source = _default_symmetry_source(raw, source_raw, source_base=source_base)
    raw["symmetry_source"] = symmetry_source
    kpath_config = _default_kpath_config(raw, source_base=source_base)
    raw["kpath"] = kpath_config
    if "symmetry_map" not in model_raw:
        model_raw["symmetry_map"] = _default_symmetry_map_from_valley_model(valley_model_raw)
    if "max_order" not in model_raw:
        model_raw["max_order"] = dict(DEFAULT_MAX_ORDER)
    raw["model"] = model_raw
    raw = _normalize_user_symmetry_names(raw)
    raw = _normalize_kp_symm_source(raw)

    model = _model_section(raw)
    if "n_orb" in model:
        n_orb_raw = model["n_orb"]
    elif "n_orb1" in model or "n_orb2" in model:
        n_orb_raw = [model.get("n_orb1", 2), model.get("n_orb2", 2)]
    else:
        n_orb_raw = _infer_n_orb_pair_from_arrays(qset1_file, qset2_file, heff_file)
    n_orb_values = _as_int_list(n_orb_raw, name="model.n_orb")
    if len(n_orb_values) != 2:
        raise ValueError(f"model.n_orb must have length 2, got {n_orb_values}")
    nlow_state = _as_int_list(model.get("nlow_state", n_orb_values), name="model.nlow_state")
    if len(nlow_state) != 2:
        raise ValueError(f"model.nlow_state must have length 2, got {nlow_state}")
    max_order_values = _max_derivative_order_values(model)

    fit = raw.get("fit", {})
    if not isinstance(fit, Mapping):
        raise ValueError("fit section must be a mapping")
    fit_indices = _as_int_list(fit.get("indices", []), name="fit.indices")

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
    _validate_sector_orbital_counts(sectors, n_orb_values)
    term_templates = model.get("term_templates", raw.get("term_templates"))
    if term_templates is None:
        term_templates = _default_term_templates_for_model(
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
            n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
            max_order=max_order_values,
            harmonic_counts=harmonic_count_limits,
        )
    if not isinstance(term_templates, Sequence) or isinstance(term_templates, (str, bytes)):
        raise ValueError("term_templates must be a list when provided")
    if not term_templates:
        term_templates = _default_term_templates_for_model(
            valley_model=valley_model if isinstance(valley_model, Mapping) else {},
            n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
            max_order=max_order_values,
            harmonic_counts=harmonic_count_limits,
        )

    symmetry_source_metadata = {}
    if isinstance(symmetry_source, Mapping) and bool(symmetry_source.get("inferred_from_source_config", False)):
        symmetry_source_metadata["inferred_from_source_config"] = True

    return ConfiguredModel(
        path=cfg_path,
        raw=dict(raw),
        source_config=source_config,
        source_raw=dict(source_raw),
        qset1_file=qset1_file,
        qset2_file=qset2_file,
        kpoints_file=kpoints_file,
        heff_file=heff_file,
        heff_eig_file=heff_eig_file,
        output_dir=output_dir,
        rotation_deg=_rotation_deg_from_config(raw, base=base),
        fit_indices=fit_indices,
        band_indices=band_indices,
        n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
        nlow_state=nlow_state,
        bM_config=dict(model.get("bM", {})),
        harmonics_config=dict(harmonics),
        max_order=max_order_values,
        symmetry_map={str(key): list(value) for key, value in dict(symmetry_map).items()},
        coeff_tol=float(fit.get("coeff_tol", 1.0e-6)),
        compare_to_heff=bool(bands.get("compare_to_heff", True)),
        kpath_config=dict(kpath_config),
        valley_model=dict(valley_model),
        symmetry_source_config=dict(symmetry_source) if isinstance(symmetry_source, Mapping) else {},
        symmetry_source_metadata=symmetry_source_metadata,
        sectors_config=[dict(item) for item in sectors],
        term_templates=[dict(item) for item in term_templates],
        output_config=dict(output_section),
        band_slice=band_slice,
        band_plot_config=dict(band_plot),
        validation_config=dict(raw.get("validation", {})),
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
    row: dict[str, Any] = {
        "name": name,
        "source": source,
        "sector_pairs": [[int(i), int(j)] for i, j in sector_pairs],
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
                "gamma_intra_nonzero",
                "moire_potential",
                [[1, 1]],
                [[1, 1], [1, 2], [2, 1]],
                harmonic_filter=_harmonic_filter("intra", [2, 3, 4]),
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
                harmonic_filter=_harmonic_filter("inter", [2, 3, 4]),
                max_order_from="tunneling_nonzero",
            ),
            _term_template_row(
                "gamma_inter_nonzero_negative",
                "tunneling",
                [[2, 1]],
                [[1, 1]],
                harmonic_filter=_harmonic_filter("inter", [2, 3, 4], sign=-1.0),
                max_order_from="tunneling_nonzero",
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
    if harmonic_counts and kind in harmonic_counts:
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
    matches: list[_TermTemplateProfile] = []
    for profile in _TERM_TEMPLATE_PROFILES:
        if profile.valley_type != str(valley_type):
            continue
        if profile.n_orb is not None and profile.n_orb != tuple(n_orb):
            continue
        matches.append(profile)
    return tuple(matches)


def _default_term_templates_for_model(
    *,
    valley_model: Mapping[str, Any],
    n_orb: tuple[int, int],
    max_order: Mapping[str, int],
    harmonic_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    valley_type = str(valley_model.get("valley_type", ""))
    profiles = _term_template_profile_for(valley_type, n_orb)
    if profiles:
        return [
            _expand_term_template_profile(template, max_order, harmonic_counts=harmonic_counts)
            for profile in profiles
            for template in profile.templates
        ]
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
    return (wedge_distance, circular_angle, theta, float(np.linalg.norm(vector)))


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


def _operation_to_harmonic_action(operation: Mapping[str, Any]) -> dict[str, Any] | None:
    q_map = operation.get("q_map", operation.get("k_map"))
    if not isinstance(q_map, Mapping):
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
            pass
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
        kpoints = np.load(config.kpoints_file)
        return _validate_kpoints(kpoints)

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
    return _validate_kpoints(generated.kpoints_2d)


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
    return load_Q_sets_from_gvec_files(
        config.qset1_file,
        config.qset2_file,
        rotation_deg=config.rotation_deg,
    )


def _operation_matrix_is_exactified(record: Mapping[str, Any]) -> bool:
    matrix_kind = str(record.get("matrix_kind", ""))
    matrix_source = str(record.get("matrix_source", record.get("matrix_file_role", "")))
    return matrix_kind == "continuum_internal_rep_exact" or matrix_source in {
        "kp_symm_exactified_action",
        "exactified_raw_h_action_projection",
    }


def _requires_model_side_exactification(metadata: Mapping[str, Any]) -> bool:
    if bool(metadata.get("requires_model_exactification", False)):
        return True
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)) or not operations:
        return False
    return not all(isinstance(record, Mapping) and _operation_matrix_is_exactified(record) for record in operations)


def build_moire_config_from_file(path: str | Path) -> tuple[MoireConfig, ConfiguredModel]:
    config = load_model_config(path)
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
        output_dir=config.output_dir,
    )
    if loaded_symmetry.generator is not None:
        moire_config.symmetry_gen = loaded_symmetry.generator
    moire_config.term_templates = [dict(item) for item in config.term_templates]
    moire_config.sectors = sectors
    moire_config.symmetry_source_metadata = loaded_symmetry.metadata
    moire_config.bM_diagnostics = bM_diagnostics
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
    if not raw or source == "auto":
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


def _plot_comparison_filename(plot_config: Mapping[str, Any] | None) -> str:
    plot_options = dict(plot_config or {})
    align = str(plot_options.get("align", "none")).lower()
    suffix = "_aligned" if align in {"top", "top_band", "top-band", "bottom", "bottom_band", "bottom-band"} else ""
    if plot_options.get("bottom_bands") is not None:
        return f"comparison_bottom{int(plot_options['bottom_bands'])}{suffix}.json"
    if plot_options.get("top_bands") is not None:
        return f"comparison_top{int(plot_options['top_bands'])}{suffix}.json"
    return "comparison_plot.json"


def _cleanup_stale_band_outputs(output_dir: Path) -> None:
    for pattern in ("comparison_top*.json", "band_comparison_top*.png", "comparison_plot.json"):
        for path in output_dir.glob(pattern):
            path.unlink()


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

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray(x, dtype=float) if x is not None else np.arange(model_sel.shape[0], dtype=float)
    if x_values.shape[0] != model_sel.shape[0]:
        raise ValueError(f"plot x-axis length {x_values.shape[0]} does not match k-point count {model_sel.shape[0]}")
    figsize_raw = plot_options.get("figsize", [3.2, 5.8])
    if not isinstance(figsize_raw, Sequence) or isinstance(figsize_raw, (str, bytes)) or len(figsize_raw) != 2:
        raise ValueError(f"bands.plot.figsize must be [width, height], got {figsize_raw!r}")
    dpi = int(plot_options.get("dpi", 220))
    fig, ax = plt.subplots(figsize=(float(figsize_raw[0]), float(figsize_raw[1])), dpi=dpi)
    for ib in range(heff_sel.shape[1]):
        ax.plot(x_values, heff_sel[:, ib], color="0.20", linewidth=0.9, alpha=0.9)
    for ib in range(model_sel.shape[1]):
        ax.plot(x_values, model_sel[:, ib], color="#d7263d", linewidth=1.0, alpha=0.92)
    if x_ticks is not None and x_ticklabels is not None and len(x_ticks) == len(x_ticklabels):
        ax.set_xticks([float(item) for item in x_ticks])
        ax.set_xticklabels(list(x_ticklabels))
        for tick in x_ticks:
            ax.axvline(float(tick), color="0.86", linewidth=0.75, zorder=0)
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    else:
        ax.set_xlabel("k-point index")
    if model_ref or heff_ref:
        ref_label = r"top" if align in {"top", "top_band", "top-band"} else r"bottom"
        ax.set_ylabel(rf"$E - E_{{\mathrm{{{ref_label}}}}}$ (eV)")
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
    if align in {"top", "top_band", "top-band", "bottom", "bottom_band", "bottom-band"}:
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
    ax.legend(
        handles=[
            Line2D([0], [0], color="0.20", lw=1.3, label="Heff"),
            Line2D([0], [0], color="#d7263d", lw=1.3, label="model"),
        ],
        loc=str(plot_options.get("legend_loc", "lower right")),
        frameon=False,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=180)
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
    ax.set_xlabel(r"$Q_x$")
    ax.set_ylabel(r"$Q_y$")
    ax.set_title("Auto-selected harmonic vectors")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color=colors["intra"], lw=1.5, linestyle=linestyles["intra"]),
            Line2D([0], [0], color=colors["inter"], lw=1.5, linestyle=linestyles["inter"]),
        ]
    )
    labels.extend(["intra selected", "inter selected"])
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out)
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
        path=output_dir / "harmonics_diagnostic.png",
    )


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
    strict = bool(model_config.validation_config.get("strict", False))
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
                    if name == "wavefunction_overlap" and bool(model_config.validation_config.get("allow_missing_wavefunction_overlap", False)):
                        continue
                    raise ValueError(
                        f"validation.strict=true requires {name}; current configured pipeline reported unavailable: "
                        f"{payload.get('reason', 'unknown')}"
                    )
        return validations, summary

    validation_hamiltonians: np.ndarray | None = None
    hermiticity_residuals: list[float] = []

    def get_validation_hamiltonians() -> np.ndarray:
        nonlocal validation_hamiltonians, hermiticity_residuals
        if validation_hamiltonians is not None:
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
                if name == "wavefunction_overlap" and bool(model_config.validation_config.get("allow_missing_wavefunction_overlap", False)):
                    continue
                raise ValueError(
                    f"validation.strict=true requires {name}; current configured pipeline reported unavailable: "
                    f"{payload.get('reason', 'unknown')}"
                )

    return validations, summary


def _build_run_summary(
    *,
    model_config: ConfiguredModel,
    results: Mapping[str, Any],
    operation_registry: Sequence[Mapping[str, Any]],
    validation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    uses_toy = model_config.symmetry_source_config.get("type") == "toy_generator"
    return {
        "symmetry_integrity": _symmetry_integrity(model_config),
        "symmetry_source_type": str(model_config.symmetry_source_config.get("type", "none")),
        "uses_toy_generator": bool(uses_toy),
        "validation_incomplete": bool(validation_summary.get("validation_incomplete", False)),
        "valley_type": str(model_config.valley_model.get("valley_type", "")),
        "valley_mode": str(model_config.valley_model.get("mode", "")),
        "spin_convention": str(model_config.valley_model.get("spin_convention", "")),
        "operation_registry_file": "operation_registry.json",
        "comparison": _json_safe(results.get("comparison")),
        "plot_comparison": _json_safe(results.get("plot_comparison")),
        "operations": list(operation_registry),
    }


def _write_model_registry_outputs(
    *,
    results: Mapping[str, Any],
    output_dir: Path,
    model_config: ConfiguredModel,
    validations: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
) -> None:
    model = results.get("model")
    terms = list(getattr(model, "terms", {}).values()) if model is not None else []
    rows = [_term_to_dict(term) for term in terms]
    active = [row for row in rows if row["active"]]
    discarded = [row for row in rows if not row["active"]]
    with (output_dir / "terms.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (output_dir / "active_terms.json").open("w", encoding="utf-8") as handle:
        json.dump(active, handle, indent=2)
    with (output_dir / "discarded_terms.json").open("w", encoding="utf-8") as handle:
        json.dump(discarded, handle, indent=2)
    with (output_dir / "coefficients.json").open("w", encoding="utf-8") as handle:
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
    with (output_dir / "fit_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results.get("diagnostics", {})), handle, indent=2)
    for name in ("orthogonalization", "matrix_residuals", "block_residuals", "wavefunction_overlap", "symmetry_residuals"):
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(validations.get(name, {"available": False, "reason": "missing"})), handle, indent=2)
    operation_registry = _build_operation_registry(model_config)
    with (output_dir / "operation_registry.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(operation_registry), handle, indent=2)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(
                _build_run_summary(
                    model_config=model_config,
                    results=results,
                    operation_registry=operation_registry,
                    validation_summary=validation_summary,
                )
            ),
            handle,
            indent=2,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
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


def _progress_line(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[kp model] {message}", flush=True)


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

    def run_stage(label: str, func):
        nonlocal first_log_write
        _progress_line(f"{label} ...", enabled=progress)
        t0 = time.perf_counter()
        mode = "w" if first_log_write else "a"
        first_log_write = False
        with _redirect_model_output(log_path, verbose=verbose, mode=mode):
            result = func()
        _progress_line(f"{label} done in {time.perf_counter() - t0:.2f} s", enabled=progress)
        return result

    model = run_stage("building continuum terms", lambda: build_model(moire_config))
    diagnostics = None
    if moire_config.heff is not None and moire_config.kpoints_fit is not None:
        model, diagnostics = run_stage("fitting coefficients", lambda: compute_coefficients(moire_config, model))
    if moire_config.kpoints is None:
        raise ValueError("config.kpoints must be provided for band computation.")
    eigvals = run_stage(
        f"computing bands on {len(moire_config.kpoints)} k-points",
        lambda: compute_bands(moire_config, model, moire_config.kpoints, return_eigvecs=moire_config.save_eigvecs),
    )
    return {"model": model, "eigvals": eigvals, "diagnostics": diagnostics}


def run_configured_model(path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    print("[kp model] loading configuration ...", flush=True)
    moire_config, model_config = build_moire_config_from_file(path)
    output_dir = model_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kp model] output directory: {output_dir}", flush=True)
    _cleanup_stale_band_outputs(output_dir)
    save_bM_diagnostics(model_config=model_config, output_dir=output_dir)
    save_harmonics_diagnostics(moire_config=moire_config, model_config=model_config, output_dir=output_dir)
    verbose = bool(model_config.output_config.get("verbose", False))
    progress = bool(model_config.output_config.get("progress", True))
    log_name = str(model_config.output_config.get("log_file", "model_run.log"))
    log_path = output_dir / log_name
    moire_config.output_dir = None
    try:
        results = _run_model_pipeline(moire_config, model_config, log_path, verbose=verbose, progress=progress)
    finally:
        moire_config.output_dir = output_dir
    eigvals = results["eigvals"]
    if isinstance(eigvals, tuple):
        eigvals_array = eigvals[0]
    else:
        eigvals_array = eigvals

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "eigvals.npy", np.asarray(eigvals_array))

    comparison = None
    plot_comparison = None
    band_plot_path = None
    if model_config.compare_to_heff:
        if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
            heff_eig = np.load(model_config.heff_eig_file)
        else:
            heff_eig = np.linalg.eigvalsh(np.load(model_config.heff_file, mmap_mode="r"))
        heff_selected = _select_rows(heff_eig, model_config.band_indices)
        comparison = compare_bands(eigvals_array, heff_selected, band_slice=model_config.band_slice)
        with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2)
        plot_comparison = compare_bands_for_plot(
            eigvals_array,
            heff_selected,
            band_slice=model_config.band_slice,
            plot_config=model_config.band_plot_config,
        )
        plot_comparison_name = _plot_comparison_filename(model_config.band_plot_config)
        with (output_dir / "comparison_plot.json").open("w", encoding="utf-8") as handle:
            json.dump(plot_comparison, handle, indent=2)
        with (output_dir / plot_comparison_name).open("w", encoding="utf-8") as handle:
            json.dump(plot_comparison, handle, indent=2)
        x_values, x_ticks, x_ticklabels = _plot_axis_from_kpath(model_config, np.asarray(eigvals_array).shape[0])
        band_plot_path = save_band_comparison_plot(
            eigvals_array,
            heff_selected,
            output_dir / "band_comparison.png",
            band_slice=model_config.band_slice,
            plot_config=model_config.band_plot_config,
            x=x_values,
            x_ticks=x_ticks,
            x_ticklabels=x_ticklabels,
            title=_band_plot_title(model_config),
        )
    results["configured_model"] = model_config
    results["moire_config"] = moire_config
    results["comparison"] = comparison
    results["plot_comparison"] = plot_comparison
    if band_plot_path is not None:
        results["band_plot"] = str(band_plot_path.resolve())
    validations, validation_summary = _compute_validation_outputs(
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
    )
    results["model_log"] = str(log_path)
    results["runtime_s"] = float(time.perf_counter() - started)
    return results
