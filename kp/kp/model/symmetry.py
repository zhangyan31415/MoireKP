from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .schema import M_EFFECTIVE_OPERATION_ALIASES
from .schema import canonical_operation_name_for_valley
from ..symmetry.action_schema import (
    PRODUCTION_MATRIX_KINDS,
    SOURCE_MATRIX_SEMANTICS,
    allows_inferred_action_metadata,
    complete_action_operation_metadata,
    validate_action_map,
)


_CANONICAL_OPERATION_NAMES = {"C3z", "C2", "TR", "C2T"}
_ARTIFACT_METADATA_KEYS = (
    "strict_metadata",
    "requires_model_exactification",
    "exactification_owner",
    "kp_symm_exactification",
    "source_action_definition",
    "model_action_definition",
)
_MANIFEST_AUTHORED_OPERATION_FIELDS = {
    "antiunitary",
    "k_map",
    "q_map",
    "sector_map",
    "matrix_file",
    "matrix_kind",
    "kind",
    "source_matrix_role",
    "source_gauge",
    "target_role",
    "gauge_correction",
    "antiunitary_convention",
    "spin_map",
    "valley_map",
    "model_action",
    "declared_model_action",
    "model_basis_action",
    "support_resolution",
    "pairs",
    "representation_pairs",
    "combined_raw_h_residual",
}


def _canonical_manifest_operation_name(name: str) -> str:
    if name == "C3":
        return "C3z"
    if name == "T":
        return "TR"
    return name


@dataclass
class LoadedSymmetrySource:
    source_type: str
    generator: Any | None
    metadata: dict[str, Any]


class MatrixSymmetryGenerator:
    """Low-energy symmetry representation loaded from kp symm/TAPW projection output."""

    def __init__(self, matrices: Mapping[str, np.ndarray], metadata: Mapping[str, Any]):
        self.matrices = {str(key): np.asarray(value, dtype=complex) for key, value in matrices.items()}
        self.metadata = dict(metadata)
        self.cached_operators: dict[str, np.ndarray] = {}

    def get_operator(self, name: str, params: Any = None) -> np.ndarray:
        key = str(name)
        if key not in self.matrices:
            raise ValueError(f"Symmetry matrix for operation {name!r} is not loaded")
        matrix = self.matrices[key]
        if name == "C3z" and params in {2, "2"}:
            return matrix @ matrix
        if name == "C3z" and params in {-1, "-1"}:
            return np.linalg.inv(matrix)
        if name == "C3z" and params in {-2, "-2"}:
            inv = np.linalg.inv(matrix)
            return inv @ inv
        return matrix


def _load_manifest(path: Path) -> dict[str, Any]:
    for name in ("manifest.json", "summary.json"):
        candidate = path / name
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Symmetry manifest root must be a mapping: {candidate}")
            return data
    return {}


def _packed_operation_metadata(name: str, valley_model: Mapping[str, Any]) -> dict[str, Any]:
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
        action["matrix_source"] = "kp_symm_exactified_action"
        action["spin_map"] = "from_kp_symm_output"
        action["valley_map"] = "identity"
        action.update({key: value for key, value in SOURCE_MATRIX_SEMANTICS.items()})
    return action


def _packed_manifest(path: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    packed = path / "representations.npz"
    if not packed.exists():
        return {}
    with np.load(packed, allow_pickle=False) as payload:
        if "__metadata_json__" in payload.files:
            metadata = json.loads(str(payload["__metadata_json__"].item()))
            return dict(metadata) if isinstance(metadata, Mapping) else {}
    valley_model = raw.get("valley_model", {})
    if not isinstance(valley_model, Mapping):
        valley_model = {}
    with np.load(packed, allow_pickle=False) as payload:
        operations = [
            {
                "name": _canonical_manifest_operation_name(str(name)),
                "operation": _canonical_manifest_operation_name(str(name)),
                "matrix_file": "representations.npz",
                "matrix_array_key": str(name),
                **_packed_operation_metadata(str(name), valley_model),
            }
            for name in payload.files
            if not str(name).startswith("__")
        ]
    return {
        "operations": operations,
        "exactification_owner": "kp_symm",
        "kp_symm_exactification": {"status": "exactified", "matrix_source": "kp_symm_exactified_action"},
    }


def _load_matrix(path: Path) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = list(arr.files)
        if not keys:
            raise ValueError(f"Empty npz symmetry matrix file: {path}")
        return np.asarray(arr[keys[0]], dtype=complex)
    return np.asarray(arr, dtype=complex)


def _load_operation_matrix(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    if path.suffix == ".npz" and record.get("matrix_array_key") is not None:
        with np.load(path, allow_pickle=False) as payload:
            return np.asarray(payload[str(record["matrix_array_key"])], dtype=complex)
    return _load_matrix(path)


def _manifest_operations(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_ops = manifest.get("operations", manifest.get("matrices", []))
    if isinstance(manifest_ops, dict):
        flattened = []
        for valley_value in manifest_ops.values():
            if isinstance(valley_value, list):
                flattened.extend([item for item in valley_value if isinstance(item, Mapping)])
            elif isinstance(valley_value, dict):
                for op_name, op_value in valley_value.items():
                    if isinstance(op_value, Mapping):
                        row = dict(op_value)
                        row.setdefault("name", str(op_name))
                        flattened.append(row)
        manifest_ops = flattened
    elif not isinstance(manifest_ops, list):
        manifest_ops = []
    out: list[dict[str, Any]] = []
    for op in manifest_ops:
        if not isinstance(op, Mapping):
            continue
        row = dict(op)
        operation_name = row.get("operation", row.get("name"))
        if row.get("name") is None and operation_name is not None:
            row["name"] = _canonical_manifest_operation_name(str(operation_name))
        out.append(row)
    return out


def _operation_records(raw_operations: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_ops = _manifest_operations(manifest)
    manifest_index: dict[str, dict[str, Any]] = {}
    for op in manifest_ops:
        for key in (op.get("operation"), op.get("name")):
            if key is None:
                continue
            manifest_index[str(key)] = dict(op)
            manifest_index[_canonical_manifest_operation_name(str(key))] = dict(op)
    if raw_operations is None:
        raw_operations = []
    if isinstance(raw_operations, Mapping):
        mapped: list[dict[str, Any]] = []
        for key, value in raw_operations.items():
            row = dict(value) if isinstance(value, Mapping) else {}
            row.setdefault("name", str(key))
            mapped.append(row)
        raw_operations = mapped
    if not isinstance(raw_operations, list):
        raise ValueError("symmetry_source.operations must be a list")
    if not raw_operations:
        return [dict(op) for op in manifest_ops if isinstance(op, Mapping)]
    records: list[dict[str, Any]] = []
    for item in raw_operations:
        if isinstance(item, str):
            matched = manifest_index.get(item)
            if matched is None:
                raise ValueError(f"Operation {item!r} not found in kp_symm_output manifest")
            records.append(matched)
        elif isinstance(item, Mapping):
            row = dict(item)
            requested = row.get("name", row.get("operation"))
            if requested is None:
                records.append(row)
                continue
            source_key = str(row.get("operation", requested))
            matched = manifest_index.get(source_key)
            if matched is None:
                records.append(row)
                continue
            merged = dict(matched)
            merged.setdefault("operation", source_key)
            for key, value in row.items():
                if key in {"name", "user_name"}:
                    merged[key] = value
                elif key == "operation":
                    merged.setdefault("operation", value)
                elif key in _MANIFEST_AUTHORED_OPERATION_FIELDS and key in matched:
                    continue
                elif key not in matched:
                    merged[key] = value
            merged["name"] = str(requested)
            records.append(merged)
        else:
            raise ValueError(f"Unsupported symmetry operation metadata: {item!r}")
    return records


def _validate_map(raw: Any, *, field: str) -> Any:
    return validate_action_map(raw, field=field)


def _pairs_for_matrix_kind(record: Mapping[str, Any], matrix_kind: str) -> list[Any]:
    pairs = record.get("pairs", [])
    return pairs if isinstance(pairs, list) else []


def _quality_warnings_from_pairs(record: Mapping[str, Any], matrix_kind: str) -> list[str]:
    warnings_out: list[str] = []
    for pair in _pairs_for_matrix_kind(record, matrix_kind):
        if isinstance(pair, Mapping):
            warnings_raw = pair.get("quality_warnings", [])
            if isinstance(warnings_raw, list):
                warnings_out.extend(str(item) for item in warnings_raw)
    return warnings_out


def _metric_from_pairs(record: Mapping[str, Any], use: str, key: str, matrix_kind: str) -> float | None:
    pairs = _pairs_for_matrix_kind(record, matrix_kind)
    if not isinstance(pairs, list) or not pairs:
        return None
    first = pairs[0]
    if not isinstance(first, Mapping):
        return None
    use_block = first.get(use, {})
    if isinstance(use_block, Mapping) and key in use_block:
        return float(use_block[key])
    return None


def _manifest_matrix_kind(record: Mapping[str, Any]) -> str:
    raw = record.get("matrix_kind", record.get("kind"))
    if raw:
        return str(raw)
    raise ValueError("kp_symm_output operation requires explicit matrix_kind metadata")


def _manifest_default_matrix_kind(manifest: Mapping[str, Any]) -> str | None:
    raw = manifest.get("default_matrix_kind", manifest.get("matrix_kind"))
    if raw:
        return str(raw)
    return None


def _complete_operation_record(record: Mapping[str, Any], *, use: str, allow_inferred: bool = False) -> dict[str, Any]:
    out = dict(record)
    if not allow_inferred:
        inferred_markers = [
            key
            for key in ("inferred_fields", "defaulted_fields")
            if out.get(key)
        ]
        if inferred_markers:
            raise ValueError(
                f"kp_symm_output operation {out.get('name', out.get('operation', ''))!r} "
                f"cannot use inferred production action metadata: {inferred_markers}"
            )
    raw_name = str(out.get("name", ""))
    alias = M_EFFECTIVE_OPERATION_ALIASES.get(raw_name)
    if alias is not None:
        out.setdefault("operation_alias", raw_name)
        out.setdefault("effective_name", str(alias["effective_name"]))
        out.setdefault("representation_level", "effective_single_spin")
        out.setdefault("canonical_physical_operation", str(alias["canonical"]))
        out.setdefault("physical_parent", str(alias["canonical"]))
        out.setdefault("approximation", {"kind": "spin_SU2_effective_block"})
        if "derived_from" in alias:
            out.setdefault("derived_from", list(alias["derived_from"]))
        out["name"] = str(alias["canonical"])
    elif raw_name:
        out["name"] = _canonical_manifest_operation_name(raw_name)
    if out.get("operation") is not None:
        out["operation"] = _canonical_manifest_operation_name(str(out["operation"]))
    canonical_name = str(out.get("name", ""))
    if canonical_name not in _CANONICAL_OPERATION_NAMES:
        raise ValueError(f"Unsupported canonical operation {canonical_name!r}")
    out = complete_action_operation_metadata(
        out,
        allow_inferred=allow_inferred,
        context="kp_symm_output operation",
    )
    matrix_kind = str(out["matrix_kind"])
    support_resolution = out.get("support_resolution")
    if not isinstance(support_resolution, Mapping):
        model_basis_action = out.get("model_basis_action")
        if isinstance(model_basis_action, Mapping):
            support_resolution = model_basis_action.get("support_resolution")
    support_matrix_source = None
    if isinstance(support_resolution, Mapping):
        support_matrix_source = support_resolution.get("support_matrix_source")
    if support_matrix_source == "representation":
        out.setdefault(
            "representation_projection_diagnostic",
            {
                "status": "raw_action_exactification_problem",
                "support_matrix_source": "representation",
                "representation_support_cleaner_than_raw_action": True,
                "projection_warnings": _quality_warnings_from_pairs(out, "representation"),
            },
        )
    quality_warnings = _quality_warnings_from_pairs(out, matrix_kind)
    if quality_warnings:
        out["projection_quality_warnings"] = quality_warnings
    out["residual"] = out.get("residual", _metric_from_pairs(out, use, "heff_covariance_residual", matrix_kind))
    out["leakage"] = out.get("leakage", _metric_from_pairs(out, use, "subspace_leakage", matrix_kind))
    if out["residual"] is None:
        out["residual"] = out.get("combined_raw_h_residual", 0.0)
    if out["leakage"] is None:
        out["leakage"] = 0.0
    return out


def _validate_exactified_operation_record(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if str(record.get("matrix_kind", "")) != "continuum_internal_rep_exact":
        return
    name = str(record.get("name", record.get("operation", "")))
    if str(manifest.get("exactification_owner", "")) != "kp_symm":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires kp_symm exactification provenance"
        )
    if str(record.get("matrix_source", "")) != "kp_symm_exactified_action":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires matrix_source='kp_symm_exactified_action'"
        )
    if str(record.get("source_matrix_role", "")) != "raw_h_sewing_action":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires source_matrix_role='raw_h_sewing_action'"
        )
    if str(record.get("target_role", "")) != "continuum_internal_rep":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires target_role='continuum_internal_rep'"
        )
    report = record.get("source_matrix_projection_report")
    if not isinstance(report, Mapping):
        return
    if report.get("source_matrix_role") is not None and str(report.get("source_matrix_role")) != "raw_h_sewing_action":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires raw-H action exactification"
        )
    support_resolution = report.get("support_resolution")
    if isinstance(support_resolution, Mapping):
        if support_resolution.get("matrix_kind") is not None and str(support_resolution.get("matrix_kind")) != "action":
            raise ValueError(
                f"continuum_internal_rep_exact operation {name!r} requires source support matrix_kind='action'"
            )
        if support_resolution.get("matrix_source") is not None and str(support_resolution.get("matrix_source")) != "raw_h_action_projection":
            raise ValueError(
                f"continuum_internal_rep_exact operation {name!r} requires raw-H action exactification"
            )


def _add_inferred_field(record: dict[str, Any], field: str) -> None:
    raw_fields = record.get("inferred_fields", [])
    fields = [str(item) for item in raw_fields] if isinstance(raw_fields, list) else []
    record["inferred_fields"] = sorted({*fields, str(field)})


def load_symmetry_source(raw: Mapping[str, Any] | None, *, base: Path, expected_dim: int | None = None) -> LoadedSymmetrySource:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("symmetry_source must be a mapping")
    source_type = str(raw.get("type", "none"))
    if source_type == "none":
        return LoadedSymmetrySource(source_type="none", generator=None, metadata={"operations": []})
    if source_type == "toy_generator":
        if not bool(raw.get("allow", False)):
            raise ValueError("toy_generator requires explicit symmetry_source.allow: true")
        warnings.warn("toy_generator is not a validated TAPW low-energy representation", RuntimeWarning, stacklevel=2)
        return LoadedSymmetrySource(source_type="toy_generator", generator=None, metadata={"operations": [], "basis_template": raw.get("basis_template")})
    if source_type != "kp_symm_output":
        raise ValueError(f"Unsupported symmetry_source.type: {source_type}")

    path_raw = raw.get("path")
    if path_raw is None:
        raise ValueError("symmetry_source.type=kp_symm_output requires path")
    path = Path(path_raw)
    if not path.is_absolute():
        path = (base / path).resolve()
    manifest = _load_manifest(path)
    if not manifest:
        manifest = _packed_manifest(path, raw)
    records = _operation_records(raw.get("operations"), manifest)
    matrices: dict[str, np.ndarray] = {}
    metadata_records: list[dict[str, Any]] = []
    use = str(raw.get("use", "raw"))
    allow_inferred = allows_inferred_action_metadata(raw) or allows_inferred_action_metadata(manifest)
    root_matrix_kind = raw.get("matrix_kind", raw.get("kind"))
    manifest_matrix_kind = _manifest_default_matrix_kind(manifest) if allow_inferred else None
    for matrix_kind_raw in (root_matrix_kind, manifest_matrix_kind):
        if matrix_kind_raw is not None and str(matrix_kind_raw) not in PRODUCTION_MATRIX_KINDS:
            raise ValueError("production symmetry matrices must use raw-action or kp_symm exactified matrices")
    inferred_matrix_kind = root_matrix_kind if root_matrix_kind is not None else manifest_matrix_kind
    if not allow_inferred:
        inferred_matrix_kind = None
    for record in records:
        record = dict(record)
        used_inferred_matrix_kind = False
        if allow_inferred and inferred_matrix_kind is not None and ("matrix_kind" not in record and "kind" not in record):
            record["matrix_kind"] = str(inferred_matrix_kind)
            _add_inferred_field(record, "matrix_kind")
            used_inferred_matrix_kind = True
        elif "matrix_kind" in record or "kind" in record:
            record["matrix_kind"] = _manifest_matrix_kind(record)
        record = _complete_operation_record(record, use=use, allow_inferred=allow_inferred)
        if used_inferred_matrix_kind:
            _add_inferred_field(record, "matrix_kind")
        _validate_exactified_operation_record(record, manifest)
        name = str(record["name"])
        matrix_file = Path(str(record["matrix_file"]))
        if not matrix_file.is_absolute():
            matrix_file = path / matrix_file
        matrix = _load_operation_matrix(matrix_file, record)
        if expected_dim is not None and matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"Symmetry matrix {matrix_file} has shape {matrix.shape}, expected {(expected_dim, expected_dim)}")
        matrices[name] = matrix
        metadata_record = dict(record)
        metadata_record["matrix_file"] = str(matrix_file)
        metadata_record["use"] = use
        metadata_records.append(metadata_record)
    artifact_metadata = {
        key: manifest[key]
        for key in _ARTIFACT_METADATA_KEYS
        if key in manifest
    }
    metadata = {"operations": metadata_records, "path": str(path), "use": use, **artifact_metadata}
    return LoadedSymmetrySource(
        source_type="kp_symm_output",
        generator=MatrixSymmetryGenerator(matrices, metadata),
        metadata=metadata,
    )
