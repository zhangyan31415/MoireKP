from __future__ import annotations

import copy
from typing import Any, Mapping


ACTION_MAP_TYPES = {"rotation", "reflection", "identity", "negation"}
PRODUCTION_MATRIX_KINDS = {"action", "continuum_internal_rep_exact"}

SOURCE_MATRIX_SEMANTICS = {
    "source_matrix_role": "raw_h_sewing_action",
    "source_gauge": "raw_saved_TAPW",
    "target_role": "continuum_internal_rep",
    "gauge_correction": {"kind": "none"},
}

REQUIRED_OPERATION_METADATA = {
    "name",
    "matrix_file",
    "antiunitary",
    "k_map",
    "q_map",
    "sector_map",
    "spin_map",
    "valley_map",
    "matrix_kind",
    "source_matrix_role",
    "source_gauge",
    "target_role",
    "gauge_correction",
    "antiunitary_convention",
}


def allows_inferred_action_metadata(config: Mapping[str, Any] | None) -> bool:
    if not isinstance(config, Mapping):
        return False
    return str(config.get("mode", "")).lower() == "diagnostic" or bool(
        config.get("allow_inferred_action_metadata", False)
    )


def validate_action_map(raw: Any, *, field: str) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    out = dict(raw)
    map_type = str(out.get("type", "")).lower()
    if map_type not in ACTION_MAP_TYPES:
        raise ValueError(f"Unsupported {field}.type {out.get('type')!r}; use explicit standard action metadata")
    return out


def _default_antiunitary_convention(record: Mapping[str, Any]) -> str:
    return "U_K" if bool(record.get("antiunitary", False)) else "none"


def complete_action_operation_metadata(
    record: Mapping[str, Any],
    *,
    allow_inferred: bool = False,
    context: str = "kp_symm_output operation",
) -> dict[str, Any]:
    """Validate one production operation record.

    Production mode is intentionally strict: the manifest must carry the
    physical and matrix-role metadata that the model consumes. Diagnostic mode
    may infer/default fields, but the returned record records exactly what was
    inferred/defaulted.
    """

    out = dict(record)
    inferred: list[str] = []
    defaulted: list[str] = []

    if allow_inferred:
        if "q_map" not in out and "k_map" in out:
            out["q_map"] = copy.deepcopy(out["k_map"])
            inferred.append("q_map")
        if "matrix_kind" not in out and "kind" in out:
            out["matrix_kind"] = out["kind"]
            inferred.append("matrix_kind")
        elif "matrix_kind" not in out:
            out["matrix_kind"] = "action"
            defaulted.append("matrix_kind")
        for key, value in SOURCE_MATRIX_SEMANTICS.items():
            if key not in out:
                out[key] = copy.deepcopy(value)
                defaulted.append(key)
        if "antiunitary_convention" not in out and "antiunitary" in out:
            out["antiunitary_convention"] = _default_antiunitary_convention(out)
            defaulted.append("antiunitary_convention")
        if "spin_map" not in out:
            out["spin_map"] = "from_kp_symm_output"
            defaulted.append("spin_map")
        if "valley_map" not in out:
            out["valley_map"] = "identity"
            defaulted.append("valley_map")
    elif "kind" in out and "matrix_kind" not in out:
        raise ValueError(f"{context} must use explicit matrix_kind, not legacy kind")

    name = str(out.get("name", out.get("operation", "")))
    missing = sorted(REQUIRED_OPERATION_METADATA - set(out))
    if missing:
        raise ValueError(f"{context} {name!r} metadata is incomplete; missing {missing}")

    matrix_kind = str(out["matrix_kind"])
    if matrix_kind not in PRODUCTION_MATRIX_KINDS:
        raise ValueError("production symmetry matrices must use raw-action or kp_symm exactified matrices")
    out["matrix_kind"] = matrix_kind

    if out.get("sector_map") in {None, "auto"}:
        raise ValueError(f"{context} {name!r} requires explicit sector_map metadata")
    out["k_map"] = validate_action_map(out["k_map"], field="k_map")
    out["q_map"] = validate_action_map(out["q_map"], field="q_map")
    out["antiunitary"] = bool(out["antiunitary"])

    if inferred:
        out["inferred_fields"] = sorted(set(inferred))
    if defaulted:
        out["defaulted_fields"] = sorted(set(defaulted))
    return out
