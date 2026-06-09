from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..blocks.blocks import _assemble_projectors_from_block_eigenvectors, calculate_energy_lists, get_H_block
from ..blocks.downfold import DownfoldingOptions, downfold_from_projectors
from ..io.tapw_loader import load_Q_sets, load_hamk
from ..model.config_schema import M_EFFECTIVE_OPERATION_ALIASES

try:
    import scipy.sparse as _sparse
except Exception:  # pragma: no cover - scipy is optional for dense-only inputs
    _sparse = None


@dataclass
class ProjectionState:
    hamk: np.ndarray
    heff: np.ndarray
    u_low: np.ndarray


@dataclass
class RepresentationData:
    matrix: np.ndarray
    from_full_spinful: bool
    spin_leakage: float | None


@dataclass
class ActionRepresentation:
    matrix: Any
    representation: RepresentationData
    pg: RepresentationData | None
    raw_h: RepresentationData | None
    pg_filename: str | None
    raw_h_filename: str | None
    action_source: str
    combined_raw_h_residual: float | None


def _resolve(path: str | None, base_dir: str) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


_SUPPORTED_OPERATION_LABELS = frozenset({"C3", "C3z", "C2", "C2T", "TR"})


def _validate_operation_label(label: str) -> str:
    text = str(label)
    if text in M_EFFECTIVE_OPERATION_ALIASES:
        text = str(M_EFFECTIVE_OPERATION_ALIASES[text]["canonical"])
    if text not in _SUPPORTED_OPERATION_LABELS:
        raise ValueError(
            f"Unsupported symm operation {text!r}; use a standard operation family and explicit action metadata"
        )
    return text


def _normalize_nlow_state_list(project_cfg: dict[str, Any]) -> list[list[int]]:
    active = project_cfg.get("active_indices")
    if active is not None:
        if isinstance(active, str):
            return [[int(part.strip()) for part in active.split(",") if part.strip()]]
        return [[int(value) for value in active]]
    nlow_state_list = project_cfg.get("nlow_state_list", [])
    if nlow_state_list and not isinstance(nlow_state_list[0], (list, tuple)):
        return [[int(x) for x in nlow_state_list]]
    return [[int(x) for x in layer] for layer in nlow_state_list]


def _downfold_method(project_cfg: dict[str, Any]) -> str:
    return str(project_cfg.get("downfold_method", project_cfg.get("method", "fixed_schur"))).lower()


def _e_ref(project_cfg: dict[str, Any]) -> float | None:
    value = project_cfg.get("e_ref", project_cfg.get("E_ref"))
    return None if value is None else float(value)


def _infer_orbitals_per_layer(hamk2d: np.ndarray, q_count: int, num_layers: int) -> int:
    base = int(hamk2d.shape[0]) // 2
    divisor = int(q_count) * int(num_layers)
    if divisor <= 0 or base % divisor != 0:
        raise ValueError(f"Cannot infer orbitals: base={base}, q_count={q_count}, num_layers={num_layers}")
    return base // divisor


def _spin_slice_hamk(hamk2d: np.ndarray, spin: str) -> np.ndarray:
    spin_lower = str(spin).lower()
    if spin_lower == "all":
        return np.asarray(hamk2d, dtype=np.complex128)
    half = hamk2d.shape[0] // 2
    if spin_lower == "up":
        return np.asarray(hamk2d[:half, :half], dtype=np.complex128)
    if spin_lower == "down":
        return np.asarray(hamk2d[half:, half:], dtype=np.complex128)
    raise ValueError(f"Unsupported spin value: {spin!r}")


def _is_sparse(matrix: Any) -> bool:
    return _sparse is not None and _sparse.issparse(matrix)


def _matrix_norm(matrix: Any) -> float:
    if _is_sparse(matrix):
        return float(_sparse.linalg.norm(matrix))
    return float(np.linalg.norm(matrix))


def _as_dense(matrix: Any) -> np.ndarray:
    if _is_sparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.complex128)
    return np.asarray(matrix, dtype=np.complex128)


def _fro_relative(lhs: Any, rhs: Any, denominator: Any) -> float:
    denom = _matrix_norm(denominator)
    if denom == 0.0:
        denom = 1.0
    return float(_matrix_norm(lhs - rhs) / denom)


def _unitarity_error(matrix: np.ndarray) -> float:
    ident = np.eye(matrix.shape[1], dtype=np.complex128)
    return _fro_relative(matrix.conj().T @ matrix, ident, ident)


def _load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Representation file missing: {path}")
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.complex128)
    if path.suffix == ".npz":
        try:
            import scipy.sparse

            return scipy.sparse.load_npz(path).tocsr()
        except Exception:
            data = np.load(path)
            for key in ("matrix", "D", "representation", "arr_0"):
                if key in data.files:
                    return np.asarray(data[key], dtype=np.complex128)
            raise ValueError(f"Cannot find matrix key in {path}; keys={data.files}")
    raise ValueError(f"Unsupported representation file extension: {path}")


def _slice_representation_for_spin(matrix: np.ndarray, spin: str, target_dim: int) -> RepresentationData:
    if matrix.shape == (target_dim, target_dim):
        matrix_out = matrix.tocsr() if _is_sparse(matrix) else np.asarray(matrix, dtype=np.complex128)
        return RepresentationData(matrix=matrix_out, from_full_spinful=False, spin_leakage=None)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Representation matrix must be square, got shape={matrix.shape}")
    spin_lower = str(spin).lower()
    if spin_lower == "all":
        raise ValueError(f"D dimension {matrix.shape} does not match spin=all U dimension {target_dim}")
    if matrix.shape != (2 * target_dim, 2 * target_dim):
        raise ValueError(f"D dimension {matrix.shape} cannot be sliced to U dimension {target_dim}")
    if spin_lower == "up":
        row = slice(0, target_dim)
        other = slice(target_dim, 2 * target_dim)
    elif spin_lower == "down":
        row = slice(target_dim, 2 * target_dim)
        other = slice(0, target_dim)
    else:
        raise ValueError(f"Unsupported spin value: {spin!r}")
    leakage_blocks = [matrix[row, other], matrix[other, row]]
    leakage = float(np.sqrt(sum(float(_matrix_norm(block) ** 2) for block in leakage_blocks)) / np.sqrt(target_dim))
    matrix_out = matrix[row, row].tocsr() if _is_sparse(matrix) else np.asarray(matrix[row, row], dtype=np.complex128)
    return RepresentationData(matrix=matrix_out, from_full_spinful=True, spin_leakage=leakage)


def _operation_entry(manifest: dict[str, Any], valley: str, operation: str) -> dict[str, Any]:
    def _matches(entry: dict[str, Any]) -> bool:
        entry_operation = str(entry.get("operation", entry.get("name", "")))
        entry_valley = str(entry.get("valley_label", entry.get("valley", valley)))
        return entry_operation == operation and entry_valley == valley

    operations = manifest.get("operations")
    if isinstance(operations, dict):
        if valley in operations:
            valley_ops = operations[valley]
            if isinstance(valley_ops, dict):
                if operation in valley_ops:
                    entry = valley_ops[operation]
                    return dict(entry) if isinstance(entry, dict) else {"filename": entry}
                if isinstance(valley_ops.get("operations"), dict):
                    entry = valley_ops["operations"].get(operation)
                    if entry is not None:
                        return dict(entry) if isinstance(entry, dict) else {"filename": entry}
        if operation in operations:
            entry = operations[operation]
            return dict(entry) if isinstance(entry, dict) else {"filename": entry}
    if isinstance(operations, list):
        for entry in operations:
            if not isinstance(entry, dict):
                continue
            if _matches(entry):
                return dict(entry)
    matrices = manifest.get("matrices")
    if isinstance(matrices, list):
        for entry in matrices:
            if isinstance(entry, dict) and _matches(entry):
                return dict(entry)
    valleys = manifest.get("valleys")
    if isinstance(valleys, dict) and valley in valleys:
        valley_ops = valleys[valley].get("operations", {}) if isinstance(valleys[valley], dict) else {}
        if operation in valley_ops:
            entry = valley_ops[operation]
            return dict(entry) if isinstance(entry, dict) else {"filename": entry}
    raise KeyError(f"manifest missing operation {valley}/{operation}")


def _entry_filename(entry: dict[str, Any], valley: str, operation: str) -> str:
    for key in ("filename", "file", "path", "representation_filename"):
        if entry.get(key):
            return str(entry[key])
    return f"{valley}/{operation}.npz"


def _operation_action_metadata(entry: dict[str, Any], operation: str, antiunitary: bool, *, strict: bool = True) -> dict[str, Any]:
    if isinstance(entry.get("k_map"), dict):
        k_map = dict(entry["k_map"])
    elif entry.get("axis_deg") is not None:
        k_map = {"type": "reflection", "axis_deg": float(entry["axis_deg"])}
    elif operation in {"C3", "C3z"}:
        k_map = {"type": "rotation", "angle_deg": 120.0}
    elif operation == "TR":
        k_map = {"type": "negation"}
    else:
        raise ValueError(f"Operation {operation!r} requires explicit k_map metadata in the TAPW symmetry manifest")
    if strict and "q_map" not in entry:
        raise ValueError(f"Operation {operation!r} requires explicit q_map metadata in the TAPW symmetry manifest")
    q_map_inferred = "q_map" not in entry
    q_map_raw = entry.get("q_map", k_map)
    if strict and ("sector_map" not in entry or entry.get("sector_map") == "auto"):
        raise ValueError(f"Operation {operation!r} requires explicit sector_map metadata in the TAPW symmetry manifest")
    has_sector_map = "sector_map" in entry
    sector_map = entry["sector_map"] if has_sector_map else "auto"
    out = {
        "k_map": k_map,
        "q_map": dict(q_map_raw) if isinstance(q_map_raw, dict) else q_map_raw,
        "sector_map": sector_map,
        "sector_map_source": "manifest" if has_sector_map else "diagnostic_qset_closure",
        "spin_map": entry.get("spin_map", "from_kp_symm_output"),
        "valley_map": entry.get("valley_map", "identity"),
        "antiunitary": bool(antiunitary),
    }
    if q_map_inferred:
        out["q_map_inferred_from_k_map"] = True
    if sector_map == "auto":
        out["sector_map_candidates"] = ["identity", "layer_exchange"]
        out["candidate_source"] = "diagnostic_qset_closure"
        out["not_canonical"] = True
    return out


def _rotation_matrix_2d(angle_deg: float) -> list[list[float]]:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return [[c, -s], [s, c]]


def _frame_metadata(*, rotation_deg: float) -> dict[str, Any]:
    rotation = _rotation_matrix_2d(rotation_deg)
    return {
        "source": "tapw_q_lists",
        "model": "continuum_model_q_basis",
        "k_transform": {
            "formula": "k_model = R(rotation_deg) @ k_source",
            "rotation_deg": float(rotation_deg),
            "linear_matrix": rotation,
        },
        "q_transform": {
            "formula": "q_model = R(rotation_deg) @ (layer_mean - q_source)",
            "rotation_deg": float(rotation_deg),
            "center": "layer_mean",
            "linear_matrix": [[-value for value in row] for row in rotation],
        },
    }


def _model_q_sets(q1: np.ndarray, q2: np.ndarray, *, rotation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    q1_arr = np.asarray(q1, dtype=float)
    q2_arr = np.asarray(q2, dtype=float)
    theta = np.deg2rad(float(rotation_deg))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    center1 = np.mean(q1_arr, axis=0)
    center2 = np.mean(q2_arr, axis=0)
    return (center1 - q1_arr) @ rotation.T, (center2 - q2_arr) @ rotation.T


def _model_frame_map(raw: Any, *, rotation_deg: float) -> Any:
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    if bool(out.get("in_model_frame", False)):
        return out
    if str(out.get("type", "")).lower() == "reflection" and "axis_deg" in out:
        out["axis_deg"] = float(out["axis_deg"]) + float(rotation_deg)
    out["in_model_frame"] = True
    return out


def normalize_reflection_axis_deg(axis_deg: float) -> float:
    return float(axis_deg) % 180.0


def reflection_axis_equiv(lhs: float, rhs: float, *, tol: float = 1.0e-8) -> bool:
    delta = (normalize_reflection_axis_deg(lhs) - normalize_reflection_axis_deg(rhs) + 90.0) % 180.0 - 90.0
    return abs(delta) <= float(tol)


def _conjugate_action_to_model_frame(source_action: dict[str, Any], *, rotation_deg: float) -> dict[str, Any]:
    model_action = dict(source_action)
    model_action["k_map"] = _model_frame_map(source_action.get("k_map"), rotation_deg=rotation_deg)
    model_action["q_map"] = _model_frame_map(source_action.get("q_map", source_action.get("k_map")), rotation_deg=rotation_deg)
    model_action["action_source"] = "derived_by_frame_conjugation"
    model_action["derivation"] = {
        "formula": "A_model = R(rotation_deg) @ A_source @ R(-rotation_deg)",
        "rotation_deg": float(rotation_deg),
        "reflection_axis_convention": "mirror_axis_deg",
        "q_transform": "q_model = R(rotation_deg) @ (sector_center - q_source)",
        "q_affine_offset_ignored_for_linear_action": True,
    }
    return model_action


def _model_frame_action_metadata(source_action: dict[str, Any], *, rotation_deg: float) -> dict[str, Any]:
    return _conjugate_action_to_model_frame(source_action, rotation_deg=rotation_deg)


def _model_action_metadata(
    source_action: dict[str, Any],
    *,
    valley: str,
    operation: str,
    rotation_deg: float,
) -> dict[str, Any]:
    return _conjugate_action_to_model_frame(source_action, rotation_deg=rotation_deg)


def _rotation_from_action_map(action_map: Any) -> np.ndarray:
    if not isinstance(action_map, dict):
        return np.eye(2, dtype=float)
    map_type = str(action_map.get("type", "")).lower()
    if map_type == "rotation":
        theta = np.deg2rad(float(action_map.get("angle_deg", 0.0)))
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    if map_type == "reflection":
        theta = np.deg2rad(float(action_map.get("axis_deg", 0.0)))
        axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        return 2.0 * np.outer(axis, axis) - np.eye(2, dtype=float)
    if map_type == "negation":
        return -np.eye(2, dtype=float)
    return np.eye(2, dtype=float)


def _shift_reflection_axis(action_map: Any, shift_deg: float) -> Any:
    if not isinstance(action_map, dict):
        return action_map
    out = dict(action_map)
    if str(out.get("type", "")).lower() == "reflection" and "axis_deg" in out:
        out["axis_deg"] = float(out["axis_deg"]) + float(shift_deg)
    return out


def _sector_target(sector: str, sector_map: Any) -> str:
    if isinstance(sector_map, dict):
        return str(sector_map.get(sector, sector))
    if str(sector_map) == "layer_exchange":
        return {"L1": "L2", "L2": "L1"}.get(sector, sector)
    return sector


def _action_key_for_mismatch(action: dict[str, Any]) -> str:
    comparable = {
        key: value
        for key, value in action.items()
        if key not in {"sector_map_source", "action_source", "derivation"}
    }
    return json.dumps(comparable, sort_keys=True)


def _has_explicit_sector_map(action: dict[str, Any]) -> bool:
    return action.get("sector_map") not in {"auto", None}


def _action_candidates_from_model_action(model_action: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]]
    sector_map = model_action.get("sector_map", "identity")
    if sector_map in {"auto", None}:
        candidates = []
        for value in ("identity", "layer_exchange"):
            candidate = dict(model_action)
            candidate["sector_map"] = value
            candidate["sector_map_source"] = "inferred_from_q_support"
            candidates.append(candidate)
    else:
        candidates = [dict(model_action)]
        if sector_map == "identity":
            swapped = dict(model_action)
            swapped["sector_map"] = "layer_exchange"
            candidates.append(swapped)
    if (
        bool(model_action.get("antiunitary", False))
        and isinstance(model_action.get("q_map"), dict)
        and str(model_action["q_map"].get("type", "")).lower() == "reflection"
    ):
        for shift in (-90.0, 90.0):
            for base in list(candidates):
                shifted = dict(base)
                shifted["k_map"] = _shift_reflection_axis(base.get("k_map"), shift)
                shifted["q_map"] = _shift_reflection_axis(base.get("q_map"), shift)
                candidates.append(shifted)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _sector_orbital_counts(
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    *,
    low_dim: int | None,
) -> tuple[int, int]:
    if len(nlow_state_list) >= 2:
        return len(nlow_state_list[0]), len(nlow_state_list[1])
    if low_dim is not None:
        q_total = int(len(q_model1) + len(q_model2))
        if q_total > 0 and int(low_dim) % q_total == 0:
            per_sector = int(low_dim) // q_total
            return per_sector, per_sector
    if nlow_state_list:
        count = len(nlow_state_list[0])
        if count % 2 == 0 and len(q_model1) == len(q_model2):
            return count // 2, count // 2
        return count, 0
    return 0, 0


def _model_basis_labels(
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    *,
    low_dim: int | None = None,
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    n_orb1, n_orb2 = _sector_orbital_counts(q_model1, q_model2, nlow_state_list, low_dim=low_dim)
    for sector, qset, bands in (
        ("L1", np.asarray(q_model1, dtype=float), range(n_orb1)),
        ("L2", np.asarray(q_model2, dtype=float), range(n_orb2)),
    ):
        for orbital_slot, _band in enumerate(bands):
            for q_index, q_vector in enumerate(qset):
                labels.append(
                    {
                        "sector": sector,
                        "q_index": int(q_index),
                        "orbital": int(orbital_slot),
                        "q_vector": np.asarray(q_vector, dtype=float),
                    }
                )
    return labels


def _basis_action_for_candidate(
    *,
    action: dict[str, Any],
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    low_dim: int | None,
    tol: float,
) -> dict[str, Any]:
    qsets = {"L1": np.asarray(q_model1, dtype=float), "L2": np.asarray(q_model2, dtype=float)}
    labels = _model_basis_labels(q_model1, q_model2, nlow_state_list, low_dim=low_dim)
    target_index = {
        (str(label["sector"]), int(label["q_index"]), int(label["orbital"])): idx
        for idx, label in enumerate(labels)
    }
    q_action_items: list[dict[str, Any]] = []
    q_action_seen: set[tuple[str, int]] = set()
    perm = np.full(len(labels), -1, dtype=int)
    missing: list[dict[str, Any]] = []
    R = _rotation_from_action_map(action.get("q_map", action.get("k_map")))
    for src_idx, label in enumerate(labels):
        source_sector = str(label["sector"])
        target_sector = _sector_target(source_sector, action.get("sector_map", "identity"))
        target_qset = qsets.get(target_sector)
        if target_qset is None or target_qset.size == 0:
            missing.append({"source_index": int(src_idx), "reason": "missing_target_sector"})
            continue
        mapped = R @ np.asarray(label["q_vector"], dtype=float)
        distances = np.linalg.norm(target_qset - mapped, axis=1)
        target_q_index = int(np.argmin(distances))
        residual = float(distances[target_q_index])
        if residual > float(tol):
            missing.append(
                {
                    "source_index": int(src_idx),
                    "source_sector": source_sector,
                    "target_sector": target_sector,
                    "q_residual": residual,
                }
            )
            continue
        key = (target_sector, target_q_index, int(label["orbital"]))
        if key not in target_index:
            missing.append({"source_index": int(src_idx), "reason": "missing_target_orbital"})
            continue
        perm[src_idx] = int(target_index[key])
        q_key = (source_sector, int(label["q_index"]))
        if q_key not in q_action_seen:
            q_action_seen.add(q_key)
            q_action_items.append(
                {
                    "source_sector": source_sector,
                    "source_q_index": int(label["q_index"]),
                    "target_sector": target_sector,
                    "target_q_index": target_q_index,
                    "q_residual": residual,
                }
            )
    return {
        "complete": bool(np.all(perm >= 0)),
        "perm": perm,
        "items": q_action_items,
        "missing": missing,
        "sector_map": action.get("sector_map", "identity"),
    }


def _block_support_residual(D: np.ndarray, labels: list[dict[str, Any]], perm: np.ndarray) -> float:
    arr = np.asarray(D, dtype=np.complex128)
    if arr.ndim == 3:
        arr = arr[0]
    mask = np.zeros(arr.shape, dtype=bool)
    target_groups: dict[tuple[str, int], list[int]] = {}
    for idx, label in enumerate(labels):
        target_groups.setdefault((str(label["sector"]), int(label["q_index"])), []).append(idx)
    for src_idx, target_idx in enumerate(perm):
        if target_idx < 0:
            continue
        src_label = labels[src_idx]
        tgt_label = labels[int(target_idx)]
        src_cols = target_groups[(str(src_label["sector"]), int(src_label["q_index"]))]
        tgt_rows = target_groups[(str(tgt_label["sector"]), int(tgt_label["q_index"]))]
        mask[np.ix_(tgt_rows, src_cols)] = True
    off = arr.copy()
    off[mask] = 0.0
    denom = float(np.linalg.norm(arr))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(off) / denom)


def _resolve_projected_model_action(
    *,
    D_low: np.ndarray,
    support_matrices: Sequence[tuple[str, np.ndarray]] | None = None,
    model_action: dict[str, Any],
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    tol: float,
    discover_action_candidates: bool = False,
    accept_support_resolved_action: bool = False,
    strict: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix_options = list(support_matrices or [("raw", D_low)])
    low_dim = int(np.asarray(matrix_options[0][1]).shape[-1])
    labels = _model_basis_labels(q_model1, q_model2, nlow_state_list, low_dim=low_dim)
    declared_basis_action = _basis_action_for_candidate(
        action=model_action,
        q_model1=q_model1,
        q_model2=q_model2,
        nlow_state_list=nlow_state_list,
        low_dim=low_dim,
        tol=tol,
    )
    declared_residuals: list[dict[str, Any]] = []
    if declared_basis_action["complete"]:
        for matrix_label, matrix in matrix_options:
            residual = _block_support_residual(matrix, labels, np.asarray(declared_basis_action["perm"], dtype=int))
            declared_residuals.append({"matrix": matrix_label, "block_off_support_rel": residual})
    best: tuple[float, str, dict[str, Any], dict[str, Any]] | None = None
    diagnostics: list[dict[str, Any]] = []
    candidate_actions = _action_candidates_from_model_action(model_action) if discover_action_candidates else [dict(model_action)]
    for candidate in candidate_actions:
        basis_action = _basis_action_for_candidate(
            action=candidate,
            q_model1=q_model1,
            q_model2=q_model2,
            nlow_state_list=nlow_state_list,
            low_dim=low_dim,
            tol=tol,
        )
        residuals: list[dict[str, Any]] = []
        if basis_action["complete"]:
            for matrix_label, matrix in matrix_options:
                residual = _block_support_residual(matrix, labels, np.asarray(basis_action["perm"], dtype=int))
                residuals.append({"matrix": matrix_label, "block_off_support_rel": residual})
                if best is None or residual < best[0]:
                    best = (residual, matrix_label, candidate, basis_action)
        diagnostics.append(
            {
                "action": {
                    "k_map": candidate.get("k_map"),
                    "q_map": candidate.get("q_map"),
                    "sector_map": candidate.get("sector_map"),
                    "antiunitary": candidate.get("antiunitary"),
                },
                "complete": bool(basis_action["complete"]),
                "missing_count": int(len(basis_action["missing"])),
                "support_residuals": residuals,
                "block_off_support_rel": min((row["block_off_support_rel"] for row in residuals), default=None),
            }
        )
    selected_candidate = dict(model_action)
    selected_basis_action = declared_basis_action
    if best is None:
        support_matrix_source = None
        selected_residual = None
    else:
        selected_residual, support_matrix_source, selected_candidate, selected_basis_action = best
    action_mismatch = _action_key_for_mismatch(selected_candidate) != _action_key_for_mismatch(model_action)
    if strict and action_mismatch:
        raise ValueError("support discovery selected an action that differs from declared model_action in strict mode")
    use_selected = bool(accept_support_resolved_action and action_mismatch)
    resolved_action = dict(selected_candidate if use_selected else model_action)
    basis_action = selected_basis_action if use_selected else declared_basis_action
    residual = selected_residual if use_selected else min((row["block_off_support_rel"] for row in declared_residuals), default=None)
    selected_report = selected_candidate if discover_action_candidates else dict(model_action)
    selected_report_residual = selected_residual if discover_action_candidates else residual
    provenance = None
    if use_selected:
        provenance = {
            "source": "support_exactification",
            "accepted_by_user": True,
            "declared_model_action": model_action,
            "selected_action_candidate": selected_candidate,
            "declared_support_residual": min((row["block_off_support_rel"] for row in declared_residuals), default=None),
            "selected_support_residual": selected_residual,
        }
        resolved_action["provenance"] = provenance
    basis_action_out = {
        "complete": bool(basis_action["complete"]),
        "sector_map": basis_action["sector_map"],
        "items": basis_action["items"],
        "missing": basis_action["missing"],
        "support_resolution": {
            "block_off_support_rel": residual,
            "support_matrix_source": support_matrix_source,
            "matrix_kind": "action",
            "matrix_source": "raw_h_action_projection",
            "action_mismatch": bool(action_mismatch),
            "candidate_source": "support_discovery" if discover_action_candidates else "manifest_model_action",
            "declared_model_action": model_action,
            "selected_model_action": selected_report,
            "selected_action_candidate": selected_report,
            "declared_support_residual": min((row["block_off_support_rel"] for row in declared_residuals), default=None),
            "selected_support_residual": selected_report_residual,
            "candidates": diagnostics,
        },
    }
    if provenance is not None:
        basis_action_out["support_resolution"]["provenance"] = provenance
    return resolved_action, basis_action_out


def _has_projection_quality_warnings(pair_rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(bool(row.get("quality_warnings")) for row in pair_rows if isinstance(row, Mapping))


def _select_operation_matrix_kind(
    model_basis_action: Mapping[str, Any],
    *,
    representation_pair_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    support_source = (model_basis_action.get("support_resolution") or {}).get("support_matrix_source")
    representation_invalid = _has_projection_quality_warnings(representation_pair_rows)
    return "action", {
        "kind": "action",
        "matrix_source": "raw_h_action_projection",
        "reason": "representation_projection_diagnostic_only" if support_source == "representation" else "support_source_selected_action",
        "support_matrix_source": support_source,
        "representation_quality_warnings": representation_invalid,
        "representation_projection_diagnostic": {
            "status": "raw_action_exactification_problem" if support_source == "representation" else "not_selected",
            "support_matrix_source": support_source,
            "projection_warnings": [str(item) for row in representation_pair_rows for item in row.get("quality_warnings", [])]
            if representation_pair_rows
            else [],
            "representation_support_cleaner_than_raw_action": support_source == "representation",
        },
    }


def _optional_entry_filename(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if entry.get(key):
            return str(entry[key])
    return None


def _pairs_from_entry(entry: dict[str, Any], nk: int, *, default_k_index: int | None = None) -> list[tuple[int, int]]:
    raw_pairs = entry.get("k_pairs", entry.get("pairs"))
    if raw_pairs is not None:
        pairs: list[tuple[int, int]] = []
        for item in raw_pairs:
            if isinstance(item, dict):
                target = item.get("target", item.get("target_index", item.get("k_target")))
                source = item.get("source", item.get("source_index", item.get("k_source")))
            else:
                target, source = item
            pairs.append((int(target), int(source)))
        return pairs
    target_indices = entry.get("target_indices")
    source_indices = entry.get("source_indices")
    if target_indices is not None and source_indices is not None:
        if len(target_indices) != len(source_indices):
            raise ValueError("target_indices and source_indices have different lengths")
        return [(int(t), int(s)) for t, s in zip(target_indices, source_indices)]
    rule = str(entry.get("source_k_rule", entry.get("k_rule", ""))).lower()
    target_rule = str(entry.get("target_k_rule", "")).lower()
    if rule in {"same", "identity", "same_index"} or target_rule in {"same", "identity", "same_index"}:
        return [(idx, idx) for idx in range(nk)]
    if rule in {"gamma", "gamma_only"} or target_rule in {"gamma", "gamma_only"}:
        return [(0, 0)]
    if default_k_index is not None:
        return [(int(default_k_index), int(default_k_index))]
    raise ValueError("Manifest operation must provide k_pairs/source_indices or a supported k rule")


def _check_spin_leakage(operation: str, label: str, rep: RepresentationData, tolerance: float) -> None:
    if rep.spin_leakage is not None and rep.spin_leakage > tolerance:
        raise ValueError(f"{operation} {label} spin off-block leakage {rep.spin_leakage:.3e} exceeds tolerance")


def _load_spin_sliced_representation(
    *,
    path: Path,
    spin: str,
    full_dim: int,
    spin_sector_sewing: str | None = None,
) -> RepresentationData:
    matrix = _load_matrix(path)
    if spin_sector_sewing is None:
        return _slice_representation_for_spin(matrix, spin, full_dim)
    if str(spin_sector_sewing).lower() != "up_to_down":
        raise ValueError(f"Unsupported spin_sector_sewing mode: {spin_sector_sewing!r}")
    spin_lower = str(spin).lower()
    if spin_lower != "up":
        raise ValueError("spin_sector_sewing=up_to_down currently requires spin: up")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Representation matrix must be square, got shape={matrix.shape}")
    if matrix.shape != (2 * full_dim, 2 * full_dim):
        raise ValueError(f"D dimension {matrix.shape} cannot be sliced to up_to_down block with full_dim {full_dim}")
    row = slice(full_dim, 2 * full_dim)
    col = slice(0, full_dim)
    block = matrix[row, col].tocsr() if _is_sparse(matrix) else np.asarray(matrix[row, col], dtype=np.complex128)
    return RepresentationData(matrix=block, from_full_spinful=True, spin_leakage=None)


def _build_action_representation(
    *,
    operation: str,
    entry: dict[str, Any],
    rep_root: Path,
    filename: str,
    antiunitary: bool,
    spin: str,
    full_dim: int,
    tolerance: float,
    spin_sector_sewing: str | None = None,
) -> ActionRepresentation:
    rep = _load_spin_sliced_representation(
        path=rep_root / filename,
        spin=spin,
        full_dim=full_dim,
        spin_sector_sewing=spin_sector_sewing,
    )
    _check_spin_leakage(operation, "representation", rep, tolerance)
    if rep.matrix.shape != (full_dim, full_dim):
        raise ValueError(f"{operation} D shape {rep.matrix.shape} does not match U_low full dimension {full_dim}")

    pg_filename = _optional_entry_filename(entry, "pg_file", "periodic_gauge_file", "source_pg_file")
    raw_h_filename = _optional_entry_filename(
        entry,
        "raw_h_operator_file",
        "raw_operator_file",
        "full_space_action_file",
        "action_operator_file",
    )
    if raw_h_filename is None:
        raise ValueError(
            f"{operation} source manifest must provide raw_h_operator_file; "
            "bare representation_file or representation_file+pg_file is not a valid action source"
        )
    pg_rep: RepresentationData | None = None
    raw_h_rep: RepresentationData | None = None
    combined: np.ndarray | None = None
    combined_residual: float | None = None

    if pg_filename is not None:
        if spin_sector_sewing is not None and str(spin_sector_sewing).lower() == "up_to_down":
            pg_rep = _load_spin_sliced_representation(
                path=rep_root / pg_filename,
                spin=spin,
                full_dim=full_dim,
                spin_sector_sewing=None,
            )
        else:
            pg_rep = _load_spin_sliced_representation(
                path=rep_root / pg_filename,
                spin=spin,
                full_dim=full_dim,
                spin_sector_sewing=spin_sector_sewing,
            )
        _check_spin_leakage(operation, "periodic-gauge", pg_rep, tolerance)
        if pg_rep.matrix.shape != (full_dim, full_dim):
            raise ValueError(f"{operation} PG shape {pg_rep.matrix.shape} does not match U_low full dimension {full_dim}")
        pg_for_raw_h = pg_rep.matrix.conj() if antiunitary else pg_rep.matrix
        combined = rep.matrix @ pg_for_raw_h

    raw_h_rep = _load_spin_sliced_representation(
        path=rep_root / raw_h_filename,
        spin=spin,
        full_dim=full_dim,
        spin_sector_sewing=spin_sector_sewing,
    )
    _check_spin_leakage(operation, "raw-H operator", raw_h_rep, tolerance)
    if raw_h_rep.matrix.shape != (full_dim, full_dim):
        raise ValueError(f"{operation} raw-H operator shape {raw_h_rep.matrix.shape} does not match U_low full dimension {full_dim}")
    if combined is not None:
        combined_residual = _fro_relative(raw_h_rep.matrix, combined, raw_h_rep.matrix)
        if combined_residual > tolerance:
            raise ValueError(
                f"{operation} manifest raw-H operator disagrees with D_g^(0)+PG by "
                f"{combined_residual:.3e}"
            )
    return ActionRepresentation(
        matrix=raw_h_rep.matrix,
        representation=rep,
        pg=pg_rep,
        raw_h=raw_h_rep,
        pg_filename=pg_filename,
        raw_h_filename=raw_h_filename,
        action_source="raw_h_operator_file",
        combined_raw_h_residual=combined_residual,
    )


def _projectors_for_k(
    hamk_spin: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    *,
    orb0: int,
    spin: str,
    mode: str,
    nlow_state_list: list[list[int]],
    norb_fix_list: list[Any],
    method: str,
    e_ref: float | None,
    project_cfg: dict[str, Any],
) -> ProjectionState:
    q_layers = [[q1], [q2]]
    orb_layers = [[orb0], [orb0]]
    _, h_vec_blk, _, _ = get_H_block(
        hamk_spin,
        q_layers,
        [1, 1],
        orb_layers,
        nlow_state_list,
        norb_fix_list,
        spin=spin,
        mode=mode,
    )
    include_high = method != "first_order"
    if spin == "all" and mode.lower() != "gamma":
        q_count = int(len(q1))
        shift = q_count * int(orb0)
        idx_list: list[np.ndarray] = []
        for layer in range(2):
            for iq in range(q_count):
                base = np.arange(iq * int(orb0), (iq + 1) * int(orb0)) + shift * layer
                idx_list.append(np.concatenate((base, base + hamk_spin.shape[0] // 2)))
        u_low, u_high = _assemble_projectors_from_block_eigenvectors(
            h_vec_blk,
            idx_list,
            nlow_state_list,
            include_high=include_high,
        )
    else:
        u_low, u_high = calculate_energy_lists(
            h_vec_blk,
            nlow_state_list,
            norb_fix_list,
            q_layers,
            orb_layers,
            mode=mode,
            include_high=include_high,
        )
    u_low = np.asarray(u_low, dtype=np.complex128)
    u_high = None if u_high is None else np.asarray(u_high, dtype=np.complex128)
    result = downfold_from_projectors(
        hamk_spin,
        u_low,
        u_high,
        DownfoldingOptions(
            method=method,
            e_ref=e_ref,
            pole_warning_mev=float(project_cfg.get("pole_warning_mev", 10.0)),
            pole_danger_mev=float(project_cfg.get("pole_danger_mev", 1.0)),
            fail_on_near_pole=_as_bool(project_cfg.get("fail_on_near_pole", False)),
        ),
    )
    return ProjectionState(hamk=np.asarray(hamk_spin, dtype=np.complex128), heff=np.asarray(result.heff, dtype=np.complex128), u_low=u_low)


def _full_space_covariance_residual(
    *,
    d_full: np.ndarray,
    h_target: np.ndarray,
    h_source: np.ndarray,
    antiunitary: bool,
) -> float:
    h_source_work = h_source.conj() if antiunitary else h_source
    d_dag = d_full.conjugate().transpose() if _is_sparse(d_full) else d_full.conj().T
    h_cov = (d_full @ h_source_work) @ d_dag
    if _is_sparse(h_cov):
        h_cov = h_cov.toarray()
    return _fro_relative(h_target, h_cov, h_target)


def _project_operation(
    *,
    operation: str,
    antiunitary: bool,
    d_full: np.ndarray,
    states: dict[int, ProjectionState],
    pairs: list[tuple[int, int]],
    tolerance: float,
    enforce_heff_covariance: bool = True,
    raise_on_quality_failure: bool = True,
    target_states: dict[int, ProjectionState] | None = None,
    source_states: dict[int, ProjectionState] | None = None,
):
    raw_mats = []
    polar_mats = []
    pair_rows = []
    if target_states is None:
        target_states = states
    if source_states is None:
        source_states = states
    for target_idx, source_idx in pairs:
        target = target_states[target_idx]
        source = source_states[source_idx]
        u_source = source.u_low.conj() if antiunitary else source.u_low
        h_source = source.heff.conj() if antiunitary else source.heff
        image = d_full @ u_source
        if _is_sparse(image):
            image = image.toarray()
        image = np.asarray(image, dtype=np.complex128)
        d_raw = target.u_low.conj().T @ image
        x, singular_values, yh = np.linalg.svd(d_raw, full_matrices=False)
        d_polar = x @ yh

        def metrics(d_matrix: np.ndarray) -> dict[str, Any]:
            leakage = float(np.linalg.norm(image - target.u_low @ d_matrix) / np.sqrt(d_matrix.shape[0]))
            h_cov = d_matrix @ h_source @ d_matrix.conj().T
            return {
                "d_unitarity_error": _unitarity_error(d_matrix),
                "subspace_leakage": leakage,
                "heff_covariance_residual": _fro_relative(target.heff, h_cov, target.heff),
            }

        raw_metrics = metrics(d_raw)
        polar_metrics = metrics(d_polar)
        sv_max_dev = float(np.max(np.abs(singular_values - 1.0))) if singular_values.size else 0.0
        quality_warnings = []
        if sv_max_dev > tolerance:
            message = f"singular values deviate from 1 by {sv_max_dev:.3e}"
            if raise_on_quality_failure:
                raise ValueError(f"{operation} k=({target_idx},{source_idx}) {message}")
            quality_warnings.append(message)
        if raw_metrics["subspace_leakage"] > tolerance:
            message = f"subspace leakage {raw_metrics['subspace_leakage']:.3e} exceeds tolerance"
            if raise_on_quality_failure:
                raise ValueError(f"{operation} k=({target_idx},{source_idx}) {message}")
            quality_warnings.append(message)
        if enforce_heff_covariance and raw_metrics["heff_covariance_residual"] > tolerance:
            message = f"Heff covariance residual {raw_metrics['heff_covariance_residual']:.3e} exceeds tolerance"
            if raise_on_quality_failure:
                raise ValueError(f"{operation} k=({target_idx},{source_idx}) {message}")
            quality_warnings.append(message)
        raw_mats.append(d_raw)
        polar_mats.append(d_polar)
        pair_rows.append(
            {
                "target_k_index": int(target_idx),
                "source_k_index": int(source_idx),
                "d_shape": list(d_raw.shape),
                "singular_values": [float(value) for value in singular_values],
                "singular_value_max_deviation": sv_max_dev,
                "quality_warnings": quality_warnings,
                "raw": raw_metrics,
                "polar": polar_metrics,
            }
        )
    return raw_mats, polar_mats, pair_rows


def _save_matrix_stack(path: Path, matrices: list[np.ndarray]) -> None:
    if len(matrices) == 1:
        np.save(path, matrices[0])
    else:
        np.save(path, np.stack(matrices, axis=0))


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# KP Symmetry Projection Summary",
        "",
        f"Valley: {summary['valley']}",
        f"Spin: {summary['spin']}",
        f"Tolerance: {summary['tolerance']}",
        f"Full dimension: {summary['full_dim']}",
        f"Low dimension: {summary['low_dim']}",
        "",
        "## Operations",
        "",
    ]
    for op in summary["operations"]:
        lines.append(f"### {op['operation']}")
        lines.append("")
        lines.append(f"- antiunitary: {op['antiunitary']}")
        lines.append(f"- representation_file: `{op['representation_file']}`")
        lines.append(f"- action_source: {op.get('action_source', 'representation_file')}")
        if op.get("pg_file") is not None:
            lines.append(f"- pg_file: `{op['pg_file']}`")
        if op.get("raw_h_operator_file") is not None:
            lines.append(f"- raw_h_operator_file: `{op['raw_h_operator_file']}`")
        if op.get("combined_raw_h_residual") is not None:
            lines.append(f"- combined_raw_h_residual: {op['combined_raw_h_residual']:.6e}")
        if op.get("spin_leakage") is not None:
            lines.append(f"- spin_leakage: {op['spin_leakage']:.6e}")
        if op.get("pg_spin_leakage") is not None:
            lines.append(f"- pg_spin_leakage: {op['pg_spin_leakage']:.6e}")
        if op.get("raw_h_spin_leakage") is not None:
            lines.append(f"- raw_h_spin_leakage: {op['raw_h_spin_leakage']:.6e}")
        for pair in op["pairs"]:
            raw = pair["raw"]
            polar = pair["polar"]
            lines.append(
                f"- k target/source {pair['target_k_index']}/{pair['source_k_index']}: "
                f"full cov={pair.get('full_space_covariance_residual', float('nan')):.6e}, "
                f"raw cov={raw['heff_covariance_residual']:.6e}, "
                f"raw leakage={raw['subspace_leakage']:.6e}, "
                f"polar cov={polar['heff_covariance_residual']:.6e}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_symmetry_projection_from_config(cfg_path: str) -> dict[str, Any]:
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    material = cfg.get("material", {})
    plot_cfg = cfg.get("plot", {})
    project_cfg = cfg.get("project", {})
    symm_cfg = cfg.get("symm", {})
    if not _as_bool(symm_cfg.get("enable", True)):
        raise ValueError("symm.enable is false")

    valley = str(symm_cfg.get("valley", "K1"))
    spin = str(symm_cfg.get("spin", material.get("spin", "all"))).lower()
    spin_sector_sewing = symm_cfg.get("spin_sector_sewing")
    q_rotation_deg = float(plot_cfg.get("q_rotation_deg", symm_cfg.get("q_rotation_deg", 0.0)))
    source_operations = [str(op) for op in symm_cfg.get("operations", [])]
    if not source_operations:
        raise ValueError("symm.operations must not be empty")
    operation_requests = [{"source": _validate_operation_label(operation), "output": operation} for operation in source_operations]
    tolerance = float(symm_cfg.get("tolerance", 1.0e-2))

    tapw_symmetry_dir = _resolve(symm_cfg.get("tapw_symmetry_dir"), cfg_dir)
    if tapw_symmetry_dir is None:
        raise ValueError("symm.tapw_symmetry_dir is required")
    rep_root = Path(tapw_symmetry_dir) / "representations"
    manifest_path = rep_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Representation manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    hamk_file = _resolve(material.get("hamk_file"), cfg_dir)
    qset1_file = _resolve(material.get("qset1_file"), cfg_dir)
    qset2_file = _resolve(material.get("qset2_file"), cfg_dir)
    if hamk_file is None or qset1_file is None or qset2_file is None:
        raise ValueError("material.hamk_file/qset1_file/qset2_file are required")
    hamk = load_hamk(hamk_file, mmap_mode="r")
    q1, q2 = load_Q_sets(qset1_file, qset2_file)
    if q2 is None:
        raise ValueError("K-valley symmetry projection requires qset2_file")
    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    nk = int(hamk3d.shape[0])
    q_count = int(len(q1))
    num_layers = int(material.get("num_layers", 2))
    mode = str(project_cfg.get("mode", "K1")).lower()
    method = _downfold_method(project_cfg)
    e_ref = _e_ref(project_cfg)
    if method in {"fixed_schur", "linearized_lowdin"} and e_ref is None:
        raise ValueError(f"project.downfold_method={method!r} requires project.e_ref")
    nlow_state_list = _normalize_nlow_state_list(project_cfg)
    norb_fix_list = project_cfg.get("norb_fix_list", [])
    if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
        orb0 = int(material["num_orb_per_layer"][0])
    else:
        orb0 = _infer_orbitals_per_layer(np.asarray(hamk3d[0]), q_count, num_layers)

    operation_entries: dict[str, dict[str, Any]] = {}
    all_pairs: set[tuple[int, int]] = set()
    default_k_index = int(plot_cfg.get("hamk_index", 0))
    for request in operation_requests:
        operation = request["source"]
        output_operation = request["output"]
        entry = _operation_entry(manifest, valley, operation)
        pairs = _pairs_from_entry(entry, nk, default_k_index=default_k_index)
        for target_idx, source_idx in pairs:
            if target_idx < 0 or target_idx >= nk or source_idx < 0 or source_idx >= nk:
                raise IndexError(f"{output_operation} source/target k unavailable: target={target_idx}, source={source_idx}, nk={nk}")
            all_pairs.add((target_idx, source_idx))
        entry["_pairs"] = pairs
        operation_entries[output_operation] = entry

    required_k = sorted({idx for pair in all_pairs for idx in pair})
    if not required_k:
        raise ValueError("No source/target k points requested by symmetry operations")
    if spin_sector_sewing is None:
        hamk_source_by_k = {
            k_index: _spin_slice_hamk(np.asarray(hamk3d[k_index], dtype=np.complex128), spin)
            for k_index in required_k
        }
        hamk_target_by_k = hamk_source_by_k
    else:
        if str(spin_sector_sewing).lower() != "up_to_down":
            raise ValueError(f"Unsupported spin_sector_sewing mode: {spin_sector_sewing!r}")
        hamk_source_by_k = {
            k_index: _spin_slice_hamk(np.asarray(hamk3d[k_index], dtype=np.complex128), "up")
            for k_index in required_k
        }
        hamk_target_by_k = {
            k_index: _spin_slice_hamk(np.asarray(hamk3d[k_index], dtype=np.complex128), "down")
            for k_index in required_k
        }
    full_dim = int(hamk_source_by_k[required_k[0]].shape[0])

    operation_payloads: dict[str, dict[str, Any]] = {}
    for request in operation_requests:
        operation = request["source"]
        output_operation = request["output"]
        entry = operation_entries[output_operation]
        antiunitary = _as_bool(entry.get("antiunitary", False))
        filename = _entry_filename(entry, valley, operation)
        action = _build_action_representation(
            operation=output_operation,
            entry=entry,
            rep_root=rep_root,
            filename=filename,
            antiunitary=antiunitary,
            spin=spin,
            full_dim=full_dim,
            tolerance=tolerance,
            spin_sector_sewing=None if spin_sector_sewing is None else str(spin_sector_sewing),
        )
        full_pair_rows = []
        for target_idx, source_idx in entry["_pairs"]:
            full_residual = _full_space_covariance_residual(
                d_full=action.matrix,
                h_target=hamk_target_by_k[target_idx],
                h_source=hamk_source_by_k[source_idx],
                antiunitary=antiunitary,
            )
            if full_residual > tolerance:
                raise ValueError(
                    f"{output_operation} k=({target_idx},{source_idx}) full-space covariance residual "
                    f"{full_residual:.3e} exceeds tolerance. The TAPW representation is not "
                    "compatible with the KP input hamk/source-k rule."
                )
            full_pair_rows.append(
                {
                    "target_k_index": int(target_idx),
                    "source_k_index": int(source_idx),
                    "full_space_covariance_residual": full_residual,
                }
            )
        operation_payloads[output_operation] = {
            "entry": entry,
            "antiunitary": antiunitary,
            "filename": filename,
            "action": action,
            "full_pair_rows": full_pair_rows,
        }

    states: dict[int, ProjectionState] = {}
    source_states: dict[int, ProjectionState] = {}
    target_states: dict[int, ProjectionState] = {}
    if spin_sector_sewing is None:
        for k_index in required_k:
            states[k_index] = _projectors_for_k(
                hamk_source_by_k[k_index],
                q1,
                q2,
                orb0=orb0,
                spin=spin,
                mode=mode,
                nlow_state_list=nlow_state_list,
                norb_fix_list=norb_fix_list,
                method=method,
                e_ref=e_ref,
                project_cfg=project_cfg,
            )
        source_states = states
        target_states = states
        first_state = states[required_k[0]]
    else:
        if str(spin_sector_sewing).lower() != "up_to_down":
            raise ValueError(f"Unsupported spin_sector_sewing mode: {spin_sector_sewing!r}")
        for k_index in required_k:
            source_states[k_index] = _projectors_for_k(
                hamk_source_by_k[k_index],
                q1,
                q2,
                orb0=orb0,
                spin="up",
                mode=mode,
                nlow_state_list=nlow_state_list,
                norb_fix_list=norb_fix_list,
                method=method,
                e_ref=e_ref,
                project_cfg=project_cfg,
            )
            target_states[k_index] = _projectors_for_k(
                hamk_target_by_k[k_index],
                q1,
                q2,
                orb0=orb0,
                spin="up",
                mode=mode,
                nlow_state_list=nlow_state_list,
                norb_fix_list=norb_fix_list,
                method=method,
                e_ref=e_ref,
                project_cfg=project_cfg,
            )
        first_state = source_states[required_k[0]]
    low_dim = int(first_state.u_low.shape[1])
    output_dir = Path(_resolve(symm_cfg.get("output_dir", "symm_project"), cfg_dir) or "symm_project")
    output_dir.mkdir(parents=True, exist_ok=True)
    q_model1, q_model2 = _model_q_sets(q1, q2, rotation_deg=q_rotation_deg)
    np.save(output_dir / "q_model_layer1.npy", q_model1)
    np.save(output_dir / "q_model_layer2.npy", q_model2)

    summary = {
        "config": cfg_path,
        "valley": valley,
        "spin": spin,
        "mode": mode,
        "tolerance": tolerance,
        "q_count": q_count,
        "orbital_block_dim": orb0,
        "full_dim": full_dim,
        "low_dim": low_dim,
        "frame": _frame_metadata(rotation_deg=q_rotation_deg),
        "q_model": {
            "files": {"layer1": "q_model_layer1.npy", "layer2": "q_model_layer2.npy"},
            "source_files": {"layer1": str(qset1_file), "layer2": str(qset2_file)},
            "formula": "q_model = R(rotation_deg) @ (layer_mean - q_source)",
        },
        "operations": [],
    }

    for request in operation_requests:
        operation = request["source"]
        output_operation = request["output"]
        payload = operation_payloads[output_operation]
        entry = payload["entry"]
        antiunitary = bool(payload["antiunitary"])
        filename = str(payload["filename"])
        action = payload["action"]
        rep = action.representation
        source_action_metadata = _operation_action_metadata(entry, output_operation, antiunitary)
        model_action_metadata = _model_action_metadata(
            source_action_metadata,
            valley=valley,
            operation=output_operation,
            rotation_deg=q_rotation_deg,
        )

        raw_mats, polar_mats, pair_rows = _project_operation(
            operation=output_operation,
            antiunitary=antiunitary,
            d_full=action.matrix,
            states=source_states,
            pairs=entry["_pairs"],
            tolerance=tolerance,
            target_states=target_states,
            source_states=source_states,
        )
        rep_raw_mats, rep_polar_mats, rep_pair_rows = _project_operation(
            operation=output_operation,
            antiunitary=antiunitary,
            d_full=rep.matrix,
            states=source_states,
            pairs=entry["_pairs"],
            tolerance=tolerance,
            enforce_heff_covariance=False,
            raise_on_quality_failure=False,
            target_states=target_states,
            source_states=source_states,
        )
        for pair_row, full_pair_row in zip(pair_rows, payload["full_pair_rows"]):
            pair_row["full_space_covariance_residual"] = full_pair_row["full_space_covariance_residual"]
        resolved_model_action, model_basis_action = _resolve_projected_model_action(
            D_low=np.asarray(raw_mats[0], dtype=np.complex128),
            support_matrices=[
                ("raw_action", np.asarray(raw_mats[0], dtype=np.complex128)),
                ("representation", np.asarray(rep_raw_mats[0], dtype=np.complex128)),
            ],
            model_action=model_action_metadata,
            q_model1=q_model1,
            q_model2=q_model2,
            nlow_state_list=nlow_state_list,
            tol=max(float(tolerance), 1.0e-8),
        )
        _save_matrix_stack(output_dir / f"{output_operation}_low_raw.npy", raw_mats)
        _save_matrix_stack(output_dir / f"{output_operation}_low_polar.npy", polar_mats)
        _save_matrix_stack(output_dir / f"{output_operation}_low_representation_raw.npy", rep_raw_mats)
        _save_matrix_stack(output_dir / f"{output_operation}_low_representation_polar.npy", rep_polar_mats)
        matrix_kind, matrix_selection = _select_operation_matrix_kind(
            model_basis_action,
            representation_pair_rows=rep_pair_rows,
        )
        summary["operations"].append(
            {
                "operation": output_operation,
                "antiunitary": antiunitary,
                "matrix_file": f"{output_operation}_low_raw.npy",
                "representation_matrix_file": f"{output_operation}_low_representation_raw.npy",
                "matrix_kind": matrix_kind,
                "source_matrix_role": "bare_D0_internal_rep" if matrix_kind == "representation" else "raw_h_sewing_action",
                "source_gauge": "raw_saved_TAPW",
                "target_role": "continuum_internal_rep",
                "gauge_correction": {"kind": "none"},
                "antiunitary_convention": "U_K" if antiunitary else "none",
                **resolved_model_action,
                "source_action": source_action_metadata,
                "model_action": resolved_model_action,
                "declared_model_action": model_action_metadata,
                "model_basis_action": model_basis_action,
                "matrix_selection": matrix_selection,
                "axis_deg": entry.get("axis_deg"),
                "status": entry.get("status"),
                "square_residual": entry.get("square_residual"),
                "spglib_index": entry.get("spglib_index"),
                "ld_source_rule": entry.get("ld_source_rule"),
                "representation_file": str(rep_root / filename),
                "pg_file": None if action.pg_filename is None else str(rep_root / action.pg_filename),
                "raw_h_operator_file": None if action.raw_h_filename is None else str(rep_root / action.raw_h_filename),
                "action_source": action.action_source,
                "combined_raw_h_residual": action.combined_raw_h_residual,
                "from_full_spinful": rep.from_full_spinful,
                "spin_leakage": rep.spin_leakage,
                "pg_spin_leakage": None if action.pg is None else action.pg.spin_leakage,
                "raw_h_spin_leakage": None if action.raw_h is None else action.raw_h.spin_leakage,
                "spin_sector_sewing": spin_sector_sewing,
                "source_spin": spin if spin_sector_sewing is None else "up",
                "target_spin": spin if spin_sector_sewing is None else "down",
                "pairs": pair_rows,
                "representation_pairs": rep_pair_rows,
            }
        )

    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_dir / "manifest.json").write_text(payload, encoding="utf-8")
    (output_dir / "summary.json").write_text(payload, encoding="utf-8")
    _write_summary_md(output_dir / "summary.md", summary)
    return summary
