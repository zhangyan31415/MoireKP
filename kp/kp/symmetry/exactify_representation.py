from __future__ import annotations

import copy
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .joint_exactification import (
    JointExactificationConfig,
    JointExactificationBranchError,
    JointExactificationError,
    compile_continuum_magnetic_presentation,
    extract_block_route_action,
    joint_exactify_block_actions,
    materialize_block_route_action,
    phase_preserving_joint_source,
)

@dataclass(frozen=True)
class OperationAction:
    name: str
    canonical_name: str | None
    antiunitary: bool
    k_map: object
    R: np.ndarray
    sector_map: object
    q_map: object
    central_phase: complex
    group_relations: list[dict[str, Any]]
    source: str
    orbital_map: object = None


@dataclass(frozen=True)
class BasisLabel:
    index: int
    sector: str
    q_integer: tuple[int, int] | None
    q_vector: np.ndarray
    orbital: int
    internal_label: str | None


@dataclass
class LabelAction:
    operation: OperationAction
    perm: np.ndarray
    block_slices: list[tuple[int, int]]
    missing: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass
class ExactificationReport:
    status: str
    reason: str | None = None
    off_support_rel: float = 0.0
    off_support_max: float = 0.0
    support_mismatch_count: int = 0
    amplitude_min: float = 0.0
    amplitude_max: float = 0.0
    amplitude_mean: float = 0.0
    amplitude_std: float = 0.0
    phase_mean_deg: float = 0.0
    phase_std_deg: float = 0.0
    selected_roots: dict[str, list[float]] = field(default_factory=dict)
    distance_mod_global_phase: float = 0.0
    group_residuals: dict[str, float] = field(default_factory=dict)
    joint_group_residuals: dict[str, float] = field(default_factory=dict)
    algebraic_canonicalization: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["selected_roots"] = {key: value for key, value in self.selected_roots.items()}
        return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return [float(obj.real), float(obj.imag)]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def canonical_q_coordinates(
    Q: Sequence[float],
    sector: str,
    bM1: Sequence[float],
    bM2: Sequence[float],
    q_offset: Sequence[float],
    tol: float,
) -> tuple[str, int, int]:
    b1 = np.asarray(bM1, dtype=float)
    b2 = np.asarray(bM2, dtype=float)
    q = np.asarray(Q, dtype=float) - np.asarray(q_offset, dtype=float)
    basis = np.column_stack([b1, b2])
    coeff = np.linalg.solve(basis, q)
    rounded = np.rint(coeff).astype(int)
    residual = q - basis @ rounded
    if float(np.linalg.norm(residual)) > float(tol):
        raise ValueError(
            f"Q cannot be integerized for sector {sector!r}: residual={np.linalg.norm(residual):.3e}, "
            f"Q={np.asarray(Q, dtype=float).tolist()}, q_offset={np.asarray(q_offset, dtype=float).tolist()}"
        )
    return str(sector), int(rounded[0]), int(rounded[1])


def infer_q_offset_from_qset(qset: np.ndarray, bM1: Sequence[float], bM2: Sequence[float]) -> np.ndarray:
    arr = np.asarray(qset, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return np.zeros(2, dtype=float)
    basis = np.column_stack([np.asarray(bM1, dtype=float), np.asarray(bM2, dtype=float)])
    coeffs = np.linalg.solve(basis, arr.T).T
    frac = coeffs - np.rint(coeffs)
    # Map to a canonical cell around zero to avoid branch jumps near +/- 0.5.
    frac = ((frac + 0.5) % 1.0) - 0.5
    offset_frac = np.median(frac, axis=0)
    return basis @ offset_frac


def _normalize_sector_map(raw: object, sector_names: Sequence[str] | None = None) -> dict[str, str]:
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        if raw == "identity":
            return {}
        if raw == "layer_exchange":
            names = [str(name) for name in (sector_names or [])]
            if len(names) == 2:
                return {names[0]: names[1], names[1]: names[0]}
            return {"L1": "L2", "L2": "L1", "bottom": "top", "top": "bottom"}
    return {}


def _sector_names_from_basis_labels(basis_labels: Sequence[BasisLabel]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for label in basis_labels:
        name = str(label.sector)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _rotation_from_k_map(k_map: object) -> np.ndarray:
    if isinstance(k_map, Mapping):
        map_type = str(k_map.get("type", "")).lower()
        if map_type == "rotation":
            theta = np.deg2rad(float(k_map.get("angle_deg", 0.0)))
            return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
        if map_type == "reflection":
            theta = np.deg2rad(float(k_map.get("axis_deg", 0.0)))
            axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
            return 2.0 * np.outer(axis, axis) - np.eye(2, dtype=float)
        if map_type == "identity":
            return np.eye(2, dtype=float)
        if map_type == "negation":
            return -np.eye(2, dtype=float)
    return np.eye(2, dtype=float)


def _standardize_k_map(raw: object) -> object:
    return dict(raw) if isinstance(raw, Mapping) else raw


def _target_sector(
    label: BasisLabel,
    operation: OperationAction,
    sector_names: Sequence[str] | None = None,
) -> str:
    sector_map = _normalize_sector_map(operation.sector_map, sector_names=sector_names)
    return sector_map.get(label.sector, label.sector)


def _mapping_get_int(mapping: Mapping[Any, Any], key: int) -> int | None:
    for candidate in (key, str(key)):
        if candidate in mapping:
            return int(mapping[candidate])
    return None


def _mapping_int_keys(mapping: Mapping[Any, Any]) -> set[int]:
    keys: set[int] = set()
    for key in mapping:
        try:
            keys.add(int(key))
        except (TypeError, ValueError):
            continue
    return keys


def _mapping_get_exactification_orbital(mapping: Mapping[Any, Any], orbital: int) -> int | None:
    # Exactification BasisLabel.orbital is 1-based. A map containing slot 0 is
    # accepted as internal 0-based form and converted to the 1-based basis.
    if 0 in _mapping_int_keys(mapping):
        mapped = _mapping_get_int(mapping, int(orbital) - 1)
        return None if mapped is None else int(mapped) + 1
    return _mapping_get_int(mapping, int(orbital))


def _target_orbital(label: BasisLabel, target_sector: str, operation: OperationAction) -> int:
    orbital = int(label.orbital)
    raw = operation.orbital_map
    if not isinstance(raw, Mapping):
        return orbital
    source_sector = str(label.sector)
    candidates: list[Any] = []
    for key in (source_sector, f"{source_sector}->{target_sector}", f"{source_sector}:{target_sector}", "*"):
        if key in raw:
            candidates.append(raw[key])
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = candidate.get(target_sector)
            if isinstance(nested, Mapping):
                mapped = _mapping_get_exactification_orbital(nested, orbital)
                if mapped is not None:
                    return mapped
            mapped = _mapping_get_exactification_orbital(candidate, orbital)
            if mapped is not None:
                return mapped
    return orbital


def _build_operation_from_record(record: Mapping[str, Any]) -> OperationAction:
    name = str(record.get("name"))
    k_map = _standardize_k_map(record.get("k_map", {}))
    q_map = _standardize_k_map(record.get("q_map", k_map))
    return OperationAction(
        name=name,
        canonical_name=name,
        antiunitary=bool(record.get("antiunitary", False)),
        k_map=k_map,
        R=_rotation_from_k_map(q_map),
        sector_map=record.get("sector_map", "identity"),
        q_map=q_map,
        central_phase=complex(1.0, 0.0),
        group_relations=[],
        source="kp_symm_output",
        orbital_map=record.get("orbital_map"),
    )


def _rotate_k_map_to_model_frame(k_map: object, *, rotation_deg: float) -> object:
    if not isinstance(k_map, Mapping):
        return k_map
    out = dict(k_map)
    if bool(out.get("in_model_frame", False)):
        return out
    if str(out.get("type", "")).lower() == "reflection" and "axis_deg" in out:
        out["axis_deg"] = float(out["axis_deg"]) + float(rotation_deg)
    return out


def _model_frame_k_map(k_map: object) -> object:
    if not isinstance(k_map, Mapping):
        return k_map
    return {**dict(k_map), "in_model_frame": True}


def _action_candidate_summary(operation: OperationAction) -> dict[str, Any]:
    return {
        "k_map": _jsonable(operation.k_map),
        "q_map": _jsonable(operation.q_map),
        "sector_map": _jsonable(operation.sector_map),
        "orbital_map": _jsonable(operation.orbital_map),
        "antiunitary": bool(operation.antiunitary),
    }


def _action_summaries_equal(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    return json.dumps(_jsonable(lhs), sort_keys=True) == json.dumps(_jsonable(rhs), sort_keys=True)


def _candidate_operation_actions(
    record: Mapping[str, Any],
    sectors: Sequence[Mapping[str, Any]],
    *,
    rotation_deg: float,
    explicit_candidates: Sequence[Mapping[str, Any]] | None = None,
    require_explicit_action_candidates: bool = False,
    discover_action_candidates: bool = False,
) -> list[OperationAction]:
    if explicit_candidates:
        out: list[OperationAction] = []
        for candidate in explicit_candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError(f"exactification.action_candidates entries must be mappings, got {candidate!r}")
            merged = dict(record)
            merged.update(candidate)
            if "k_map" in candidate and "q_map" not in candidate:
                merged["q_map"] = candidate["k_map"]
            if "k_map" in merged:
                merged["k_map"] = _rotate_k_map_to_model_frame(merged.get("k_map", {}), rotation_deg=rotation_deg)
            if "q_map" in merged:
                merged["q_map"] = _rotate_k_map_to_model_frame(merged.get("q_map", {}), rotation_deg=rotation_deg)
            out.append(_build_operation_from_record(merged))
        return out
    if require_explicit_action_candidates:
        raise ValueError(
            f"exactification strict mode requires explicit action_candidates for operation {record.get('name')!r}"
        )
    model_record = dict(record)
    model_action = record.get("model_action")
    if isinstance(model_action, Mapping):
        for key in ("antiunitary", "k_map", "q_map", "sector_map", "orbital_map"):
            if key in model_action:
                model_record[key] = model_action[key]
        model_record["k_map"] = _model_frame_k_map(model_record.get("k_map", {}))
        if "q_map" in model_record:
            model_record["q_map"] = _model_frame_k_map(model_record.get("q_map", {}))
    else:
        model_record["k_map"] = _rotate_k_map_to_model_frame(model_record.get("k_map", {}), rotation_deg=rotation_deg)
        if "q_map" in model_record:
            model_record["q_map"] = _rotate_k_map_to_model_frame(model_record.get("q_map", {}), rotation_deg=rotation_deg)
    out = [_build_operation_from_record(model_record)]
    if discover_action_candidates:
        for sector_map in ("identity", "layer_exchange"):
            if sector_map == model_record.get("sector_map"):
                continue
            candidate_record = dict(model_record)
            candidate_record["sector_map"] = sector_map
            out.append(_build_operation_from_record(candidate_record))
    return out


def build_basis_labels(
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    sectors: Sequence[Mapping[str, Any]],
    n_orb1: int,
    n_orb2: int,
    bM1: np.ndarray,
    bM2: np.ndarray,
    tol: float,
    q_offsets: Mapping[str, np.ndarray] | None = None,
    forbid_inferred_q_offset: bool = False,
) -> list[BasisLabel]:
    q_lookup = {"qset1": np.asarray(Q_set1, dtype=float), "qset2": np.asarray(Q_set2, dtype=float)}
    labels: list[BasisLabel] = []
    idx = 0
    for sector in sectors:
        name = str(sector.get("name"))
        qset_name = str(sector.get("qset", "qset1"))
        n_orb = int(sector.get("n_orb", n_orb1 if qset_name == "qset1" else n_orb2))
        qset = q_lookup[qset_name]
        if q_offsets is not None and name in q_offsets:
            q_offset = np.asarray(q_offsets[name], dtype=float)
        elif "q_offset" in sector:
            q_offset = np.asarray(sector["q_offset"], dtype=float)
            try:
                canonical_q_coordinates(qset[0], name, bM1, bM2, q_offset, tol)
            except Exception:
                if forbid_inferred_q_offset:
                    raise ValueError(f"exactification strict mode forbids inferred q_offset for sector {name!r}")
                q_offset = infer_q_offset_from_qset(qset, bM1, bM2)
        else:
            if forbid_inferred_q_offset:
                raise ValueError(f"exactification strict mode forbids inferred q_offset for sector {name!r}")
            q_offset = infer_q_offset_from_qset(qset, bM1, bM2)
        for orbital in range(1, n_orb + 1):
            for qvec in qset:
                qint = canonical_q_coordinates(qvec, name, bM1, bM2, q_offset, tol)
                labels.append(
                    BasisLabel(
                        index=idx,
                        sector=name,
                        q_integer=(qint[1], qint[2]),
                        q_vector=np.asarray(qvec, dtype=float),
                        orbital=orbital,
                        internal_label=None,
                    )
                )
                idx += 1
    return labels


def build_label_action(
    basis_labels: Sequence[BasisLabel],
    operation: OperationAction,
    bM1: np.ndarray,
    bM2: np.ndarray,
    q_offsets: Mapping[str, np.ndarray],
    tol: float,
) -> LabelAction:
    lookup: dict[tuple[str, int, int, int], int] = {}
    by_sector_orbital: dict[tuple[str, int], list[BasisLabel]] = {}
    for label in basis_labels:
        if label.q_integer is None:
            raise ValueError(f"Basis label {label.index} is missing q_integer")
        lookup[(label.sector, label.q_integer[0], label.q_integer[1], label.orbital)] = label.index
        by_sector_orbital.setdefault((label.sector, int(label.orbital)), []).append(label)
    perm = np.full(len(basis_labels), -1, dtype=int)
    missing: list[dict[str, Any]] = []
    sector_names = _sector_names_from_basis_labels(basis_labels)
    for label in basis_labels:
        target_sector = _target_sector(label, operation, sector_names=sector_names)
        target_orbital = _target_orbital(label, target_sector, operation)
        q_target = operation.R @ np.asarray(label.q_vector, dtype=float)
        target_idx = None
        integerize_error: str | None = None
        try:
            _, n1, n2 = canonical_q_coordinates(q_target, target_sector, bM1, bM2, q_offsets[target_sector], tol)
            key = (target_sector, n1, n2, target_orbital)
            target_idx = lookup.get(key)
        except Exception as exc:
            integerize_error = str(exc)
        if target_idx is None:
            candidates = by_sector_orbital.get((target_sector, target_orbital), [])
            if candidates:
                distances = [float(np.linalg.norm(np.asarray(candidate.q_vector, dtype=float) - q_target)) for candidate in candidates]
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= float(tol):
                    target_idx = int(candidates[best_idx].index)
            if target_idx is None:
                payload = {
                    "index": int(label.index),
                    "sector": label.sector,
                    "target_sector": target_sector,
                    "source_orbital": int(label.orbital),
                    "target_orbital": int(target_orbital),
                }
                if integerize_error is not None:
                    payload["reason"] = integerize_error
                missing.append(payload)
                continue
        perm[label.index] = target_idx
    diagnostics = {
        "missing_count": int(len(missing)),
        "perm_complete": bool(np.all(perm >= 0)),
    }
    return LabelAction(operation=operation, perm=perm, block_slices=[], missing=missing, diagnostics=diagnostics)


def _choose_best_action_candidate(
    D_num: np.ndarray,
    candidates: Sequence[OperationAction],
    basis_labels: Sequence[BasisLabel],
    bM1: np.ndarray,
    bM2: np.ndarray,
    q_offsets: Mapping[str, np.ndarray],
    tol: float,
) -> dict[str, Any] | None:
    best: tuple[float, float, dict[str, Any]] | None = None
    for operation in candidates:
        label_action = build_label_action(basis_labels, operation, bM1, bM2, q_offsets, tol)
        if label_action.missing:
            continue
        mono_report = analyze_monomial_support(np.asarray(D_num, dtype=complex), label_action.perm)
        try:
            block_groups = build_block_groups(basis_labels, operation, bM1, bM2, q_offsets, tol)
            block_report = analyze_block_support(np.asarray(D_num, dtype=complex), block_groups)
        except Exception:
            block_groups = None
            block_report = None
        mono_amp = max(abs(mono_report.amplitude_min - 1.0), abs(mono_report.amplitude_max - 1.0))
        block_amp = float("inf") if block_report is None else max(abs(block_report.amplitude_min - 1.0), abs(block_report.amplitude_max - 1.0))
        mono_score = (mono_report.off_support_rel, mono_amp, "monomial")
        block_score = (float("inf"), float("inf"), "block") if block_report is None else (block_report.off_support_rel, block_amp, "block")
        score = min(mono_score, block_score)
        payload = {
            "operation": operation,
            "label_action": label_action,
            "monomial_report": mono_report,
            "block_groups": block_groups,
            "block_report": block_report,
            "preferred_mode": score[2],
        }
        if best is None or (score[0], score[1]) < (best[0], best[1]):
            best = (score[0], score[1], payload)
    return None if best is None else best[2]


def _candidate_support_residuals(
    D_num: np.ndarray,
    candidates: Sequence[OperationAction],
    basis_labels: Sequence[BasisLabel],
    bM1: np.ndarray,
    bM2: np.ndarray,
    q_offsets: Mapping[str, np.ndarray],
    tol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operation in candidates:
        row: dict[str, Any] = {"action": _action_candidate_summary(operation)}
        try:
            label_action = build_label_action(basis_labels, operation, bM1, bM2, q_offsets, tol)
            row["perm_complete"] = bool(label_action.diagnostics["perm_complete"])
            row["missing_count"] = int(label_action.diagnostics["missing_count"])
            if not label_action.missing:
                mono_report = analyze_monomial_support(np.asarray(D_num, dtype=complex), label_action.perm)
                row["monomial_off_support_rel"] = float(mono_report.off_support_rel)
                row["monomial_amplitude_deviation"] = float(
                    max(abs(mono_report.amplitude_min - 1.0), abs(mono_report.amplitude_max - 1.0))
                )
                try:
                    block_groups = build_block_groups(basis_labels, operation, bM1, bM2, q_offsets, tol)
                    block_report = analyze_block_support(np.asarray(D_num, dtype=complex), block_groups)
                    row["block_off_support_rel"] = float(block_report.off_support_rel)
                    row["block_amplitude_deviation"] = float(
                        max(abs(block_report.amplitude_min - 1.0), abs(block_report.amplitude_max - 1.0))
                    )
                except Exception as exc:
                    row["block_error"] = str(exc)
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def analyze_monomial_support(D_num: np.ndarray, perm: Sequence[int], block_dims: Any = None) -> ExactificationReport:
    arr = np.asarray(D_num, dtype=complex)
    perm_arr = np.asarray(perm, dtype=int)
    mask = np.zeros(arr.shape, dtype=bool)
    support_values: list[complex] = []
    support_mismatch = 0
    for src_idx, tgt_idx in enumerate(perm_arr):
        if tgt_idx < 0:
            support_mismatch += 1
            continue
        mask[tgt_idx, src_idx] = True
        support_values.append(arr[tgt_idx, src_idx])
    D_off = arr.copy()
    D_off[mask] = 0.0
    support = np.asarray(support_values, dtype=complex)
    denom = float(np.linalg.norm(arr))
    if denom == 0.0:
        denom = 1.0
    phases = np.angle(support) if support.size else np.array([0.0])
    amps = np.abs(support) if support.size else np.array([0.0])
    return ExactificationReport(
        status="diagnostic",
        off_support_rel=float(np.linalg.norm(D_off) / denom),
        off_support_max=float(np.max(np.abs(D_off))) if D_off.size else 0.0,
        support_mismatch_count=int(support_mismatch),
        amplitude_min=float(np.min(amps)),
        amplitude_max=float(np.max(amps)),
        amplitude_mean=float(np.mean(amps)),
        amplitude_std=float(np.std(amps)),
        phase_mean_deg=float(np.mean(phases) * 180.0 / np.pi),
        phase_std_deg=float(np.std(phases) * 180.0 / np.pi),
    )


def _group_key(label: BasisLabel) -> tuple[str, tuple[int, int]]:
    if label.q_integer is None:
        raise ValueError(f"Basis label {label.index} missing q_integer")
    return str(label.sector), (int(label.q_integer[0]), int(label.q_integer[1]))


def build_block_groups(
    basis_labels: Sequence[BasisLabel],
    operation: OperationAction,
    bM1: np.ndarray,
    bM2: np.ndarray,
    q_offsets: Mapping[str, np.ndarray],
    tol: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    by_group: dict[tuple[str, tuple[int, int]], list[BasisLabel]] = {}
    for label in basis_labels:
        by_group.setdefault(_group_key(label), []).append(label)
    for labels in by_group.values():
        labels.sort(key=lambda item: int(item.orbital))
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[str, tuple[int, int]]] = set()
    sector_names = _sector_names_from_basis_labels(basis_labels)
    for key, src_group in by_group.items():
        if key in seen:
            continue
        sample = src_group[0]
        target_sector = _target_sector(sample, operation, sector_names=sector_names)
        q_target = operation.R @ np.asarray(sample.q_vector, dtype=float)
        try:
            _, n1, n2 = canonical_q_coordinates(q_target, target_sector, bM1, bM2, q_offsets[target_sector], tol)
            target_key = (target_sector, (n1, n2))
        except Exception:
            # Fallback: nearest group center
            best_key = None
            best_dist = None
            for candidate_key, candidate_group in by_group.items():
                if candidate_key[0] != target_sector:
                    continue
                dist = float(np.linalg.norm(np.asarray(candidate_group[0].q_vector, dtype=float) - q_target))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_key = candidate_key
            if best_key is None or best_dist is None or best_dist > tol:
                raise ValueError(f"Cannot map group {key!r} under {operation.name}")
            target_key = best_key
        tgt_group = by_group.get(target_key)
        if tgt_group is None or len(tgt_group) != len(src_group):
            raise ValueError(f"Block support mismatch for {key!r} -> {target_key!r}")
        row_idx = np.asarray([label.index for label in tgt_group], dtype=int)
        col_idx = np.asarray([label.index for label in src_group], dtype=int)
        groups.append((row_idx, col_idx))
        seen.add(key)
    return groups


def analyze_block_support(D_num: np.ndarray, groups: Sequence[tuple[np.ndarray, np.ndarray]]) -> ExactificationReport:
    arr = np.asarray(D_num, dtype=complex)
    mask = np.zeros(arr.shape, dtype=bool)
    blocks: list[np.ndarray] = []
    for rows, cols in groups:
        mask[np.ix_(rows, cols)] = True
        blocks.append(arr[np.ix_(rows, cols)])
    D_off = arr.copy()
    D_off[mask] = 0.0
    singular_values = np.concatenate([np.linalg.svd(block, compute_uv=False) for block in blocks]) if blocks else np.array([0.0])
    return ExactificationReport(
        status="diagnostic",
        off_support_rel=float(np.linalg.norm(D_off) / max(np.linalg.norm(arr), 1.0)),
        off_support_max=float(np.max(np.abs(D_off))) if D_off.size else 0.0,
        support_mismatch_count=0,
        amplitude_min=float(np.min(singular_values)),
        amplitude_max=float(np.max(singular_values)),
        amplitude_mean=float(np.mean(singular_values)),
        amplitude_std=float(np.std(singular_values)),
        phase_mean_deg=0.0,
        phase_std_deg=0.0,
    )


def _block_support_is_substantially_cleaner(
    monomial_report: ExactificationReport,
    block_report: ExactificationReport | None,
    *,
    reject_if_off_support_rel_gt: float,
    reject_if_amplitude_deviation_gt: float,
) -> bool:
    if block_report is None or block_report.support_mismatch_count:
        return False
    block_amp_dev = max(abs(block_report.amplitude_min - 1.0), abs(block_report.amplitude_max - 1.0))
    if block_report.off_support_rel > reject_if_off_support_rel_gt or block_amp_dev > reject_if_amplitude_deviation_gt:
        return False
    mono_off = float(monomial_report.off_support_rel)
    block_off = float(block_report.off_support_rel)
    if mono_off <= 1.0e-12:
        return False
    return block_off < mono_off and block_off <= max(1.0e-12, 0.1 * mono_off)


def root_of_unity(order: int, power: int) -> complex:
    return complex(np.exp(2.0j * np.pi * int(power) / int(order)))


def nearest_root_of_unity(z: complex, allowed_roots: Sequence[complex]) -> tuple[complex, float]:
    if not allowed_roots:
        raise ValueError("allowed_roots must be non-empty")
    u = complex(z)
    if abs(u) == 0.0:
        raise ValueError("cannot snap zero to a root of unity")
    u /= abs(u)
    best = min(allowed_roots, key=lambda root: abs(np.angle(u / root)))
    return complex(best), float(np.angle(u / best))


def _roots_for_power_phase(power: int, phase: complex) -> list[complex]:
    phase_angle = float(np.angle(complex(phase)))
    return [complex(np.exp(1j * (phase_angle + 2.0 * np.pi * m) / int(power))) for m in range(int(power))]


def _phase_class_key(label: BasisLabel, phase_classes: object) -> str:
    if isinstance(phase_classes, str):
        if phase_classes == "global":
            return "global"
        if phase_classes == "sector":
            return f"sector:{label.sector}"
        if phase_classes == "orbital":
            return f"orbital:{label.orbital}"
        if phase_classes == "layer":
            return f"layer:{label.sector}"
    return "global"


def _phase_class_key_for_mapping(
    label: BasisLabel,
    phase_classes: object,
    *,
    src_idx: int,
    tgt_idx: int,
) -> str:
    if isinstance(phase_classes, str) and phase_classes in {"entry", "matrix_element"}:
        return f"entry:{int(tgt_idx)}:{int(src_idx)}"
    return _phase_class_key(label, phase_classes)


def _unit_phase(value: complex) -> complex:
    z = complex(value)
    if abs(z) == 0.0:
        return 1.0 + 0.0j
    return z / abs(z)


def _exactify_antiunitary_monomial_phases(
    arr: np.ndarray,
    perm_arr: np.ndarray,
    *,
    allowed_roots: Sequence[complex],
    central_phase: complex,
) -> tuple[np.ndarray, dict[str, list[float]], list[str]]:
    roots = [complex(root) for root in allowed_roots] or roots_of_unity_up_to(12)
    if not roots:
        roots = [1.0 + 0.0j]
    exact = np.zeros_like(arr, dtype=complex)
    selected_roots: dict[str, list[float]] = {}
    notes: list[str] = ["antiunitary_pair_phase_exactification"]
    visited: set[int] = set()
    central = complex(central_phase)
    for src_idx, tgt_idx_raw in enumerate(perm_arr):
        if src_idx in visited:
            continue
        tgt_idx = int(tgt_idx_raw)
        reverse_tgt = int(perm_arr[tgt_idx])
        if reverse_tgt != src_idx:
            raise ValueError("antiunitary monomial exactification requires two-cycle support")
        if src_idx == tgt_idx:
            if abs(central - 1.0) > 1.0e-10:
                raise ValueError("antiunitary fixed-point support cannot realize a nontrivial square phase")
            value = _unit_phase(arr[tgt_idx, src_idx])
            root, residual = nearest_root_of_unity(value, roots)
            exact[tgt_idx, src_idx] = root
            selected_roots[f"entry:{tgt_idx}:{src_idx}"] = [float(root.real), float(root.imag), float(residual)]
            visited.add(src_idx)
            continue

        forward = _unit_phase(arr[tgt_idx, src_idx])
        backward = _unit_phase(arr[src_idx, tgt_idx])
        best_alpha = roots[0]
        best_beta = np.conjugate(central / best_alpha)
        best_residual = float("inf")
        for alpha in roots:
            if abs(alpha) == 0.0:
                continue
            beta = np.conjugate(central / alpha)
            residual = float(abs(forward - alpha) + abs(backward - beta))
            if residual < best_residual:
                best_alpha = alpha
                best_beta = beta
                best_residual = residual
        exact[tgt_idx, src_idx] = best_alpha
        exact[src_idx, tgt_idx] = best_beta
        selected_roots[f"entry:{tgt_idx}:{src_idx}"] = [
            float(best_alpha.real),
            float(best_alpha.imag),
            float(abs(forward - best_alpha)),
        ]
        selected_roots[f"entry:{src_idx}:{tgt_idx}"] = [
            float(best_beta.real),
            float(best_beta.imag),
            float(abs(backward - best_beta)),
        ]
        visited.add(src_idx)
        visited.add(tgt_idx)
    return exact, selected_roots, notes


def _exactify_order2_monomial_pair_phases(
    arr: np.ndarray,
    perm_arr: np.ndarray,
    *,
    allowed_roots: Sequence[complex],
    central_phase: complex,
) -> tuple[np.ndarray, dict[str, list[float]], list[str]]:
    roots = [complex(root) for root in allowed_roots]
    exact = np.zeros_like(arr, dtype=complex)
    selected_roots: dict[str, list[float]] = {}
    notes: list[str] = ["order2_pair_phase_exactification"]
    visited: set[int] = set()
    central = complex(central_phase)
    for src_idx, tgt_idx_raw in enumerate(perm_arr):
        if src_idx in visited:
            continue
        tgt_idx = int(tgt_idx_raw)
        reverse_tgt = int(perm_arr[tgt_idx])
        if reverse_tgt != src_idx:
            raise ValueError("order-2 monomial exactification requires two-cycle support")
        if src_idx == tgt_idx:
            value = _unit_phase(arr[tgt_idx, src_idx])
            candidate_roots = roots or _roots_for_power_phase(2, central)
            root, residual = nearest_root_of_unity(value, candidate_roots)
            exact[tgt_idx, src_idx] = root
            selected_roots[f"entry:{tgt_idx}:{src_idx}"] = [float(root.real), float(root.imag), float(residual)]
            visited.add(src_idx)
            continue

        forward = _unit_phase(arr[tgt_idx, src_idx])
        backward = _unit_phase(arr[src_idx, tgt_idx])
        if roots:
            best_alpha = roots[0]
            best_beta = central / best_alpha
            best_residual = float("inf")
            for alpha in roots:
                if abs(alpha) == 0.0:
                    continue
                beta = central / alpha
                residual = float(abs(forward - alpha) + abs(backward - beta))
                if residual < best_residual:
                    best_alpha = alpha
                    best_beta = beta
                    best_residual = residual
        else:
            product = forward * backward
            correction_roots = _roots_for_power_phase(2, central / product)
            correction = min(correction_roots, key=lambda root: abs(root - 1.0))
            best_alpha = forward * correction
            best_beta = backward * correction
            best_residual = float(abs(forward - best_alpha) + abs(backward - best_beta))
        exact[tgt_idx, src_idx] = best_alpha
        exact[src_idx, tgt_idx] = best_beta
        selected_roots[f"entry:{tgt_idx}:{src_idx}"] = [
            float(best_alpha.real),
            float(best_alpha.imag),
            float(abs(forward - best_alpha)),
        ]
        selected_roots[f"entry:{src_idx}:{tgt_idx}"] = [
            float(best_beta.real),
            float(best_beta.imag),
            float(abs(backward - best_beta)),
        ]
        visited.add(src_idx)
        visited.add(tgt_idx)
    return exact, selected_roots, notes


def exactify_1d_monomial_phases(
    D_num: np.ndarray,
    perm: Sequence[int],
    *,
    labels: Sequence[BasisLabel],
    phase_classes: object,
    allowed_roots: Sequence[complex],
    operation_name: str,
    power: int,
    central_phase: complex,
    antiunitary: bool = False,
    reject_if_off_support_rel_gt: float = 1.0e-6,
    reject_if_amplitude_deviation_gt: float = 2.0e-2,
) -> tuple[np.ndarray, ExactificationReport]:
    arr = np.asarray(D_num, dtype=complex)
    perm_arr = np.asarray(perm, dtype=int)
    input_report = analyze_monomial_support(arr, perm_arr)
    if input_report.support_mismatch_count:
        raise ValueError(f"{operation_name} exactification failed: support mismatch count={input_report.support_mismatch_count}")
    if input_report.off_support_rel > reject_if_off_support_rel_gt:
        raise ValueError(
            f"{operation_name} exactification failed: off-support pollution {input_report.off_support_rel:.3e} exceeds "
            f"{reject_if_off_support_rel_gt:.3e}"
        )
    amp_dev = max(abs(input_report.amplitude_min - 1.0), abs(input_report.amplitude_max - 1.0))
    if amp_dev > reject_if_amplitude_deviation_gt:
        raise ValueError(
            f"{operation_name} exactification failed: amplitude deviation {amp_dev:.3e} exceeds "
            f"{reject_if_amplitude_deviation_gt:.3e}"
        )

    if antiunitary:
        exact, selected_roots, notes = _exactify_antiunitary_monomial_phases(
            arr,
            perm_arr,
            allowed_roots=allowed_roots,
            central_phase=central_phase,
        )
    elif int(power) == 2:
        exact, selected_roots, notes = _exactify_order2_monomial_pair_phases(
            arr,
            perm_arr,
            allowed_roots=allowed_roots,
            central_phase=central_phase,
        )
    else:
        by_class: dict[str, list[complex]] = {}
        class_members: dict[str, list[tuple[int, int]]] = {}
        for src_idx, tgt_idx in enumerate(perm_arr):
            key = _phase_class_key_for_mapping(labels[src_idx], phase_classes, src_idx=src_idx, tgt_idx=int(tgt_idx))
            by_class.setdefault(key, []).append(arr[tgt_idx, src_idx] / abs(arr[tgt_idx, src_idx]))
            class_members.setdefault(key, []).append((tgt_idx, src_idx))

        exact = np.zeros_like(arr, dtype=complex)
        selected_roots = {}
        notes = []
        split_phase_class_chord_tol = 0.25
        for key, values in by_class.items():
            class_values = np.asarray(values, dtype=complex)
            avg = np.sum(class_values)
            split_class = False
            if abs(avg) == 0.0:
                split_class = True
            else:
                avg = avg / abs(avg)
                root, residual = nearest_root_of_unity(avg, allowed_roots)
                max_chord_residual = float(np.max(np.abs(class_values - root)))
                split_class = bool(max_chord_residual > split_phase_class_chord_tol and len(class_values) > 1)

            if split_class:
                notes.append("split_phase_class")
                for tgt_idx, src_idx in class_members[key]:
                    value = arr[tgt_idx, src_idx] / abs(arr[tgt_idx, src_idx])
                    root, residual = nearest_root_of_unity(value, allowed_roots)
                    selected_roots[f"{key}|entry:{int(tgt_idx)}:{int(src_idx)}"] = [
                        float(root.real),
                        float(root.imag),
                        float(residual),
                    ]
                    exact[tgt_idx, src_idx] = root
            else:
                selected_roots[key] = [float(root.real), float(root.imag), float(residual)]
                for tgt_idx, src_idx in class_members[key]:
                    exact[tgt_idx, src_idx] = root

    ident = np.eye(exact.shape[0], dtype=complex)
    if antiunitary:
        power_residual = antiunitary_square_residual(exact, central_phase)
    else:
        power_residual = float(np.linalg.norm(np.linalg.matrix_power(exact, int(power)) - complex(central_phase) * ident) / np.sqrt(exact.shape[0]))
    if power_residual > 1.0e-8:
        raise ValueError(
            f"{operation_name} exactification failed: group power residual {power_residual:.3e} exceeds 1.000e-08"
        )
    alpha = np.vdot(exact, arr) / np.vdot(exact, exact)
    alpha = 1.0 + 0.0j if abs(alpha) == 0.0 else alpha / abs(alpha)
    distance = float(np.linalg.norm(arr - alpha * exact) / max(np.linalg.norm(exact), 1.0))
    final_report = analyze_monomial_support(exact, perm_arr)
    report = ExactificationReport(
        status="exactified",
        off_support_rel=final_report.off_support_rel,
        off_support_max=final_report.off_support_max,
        support_mismatch_count=final_report.support_mismatch_count,
        amplitude_min=final_report.amplitude_min,
        amplitude_max=final_report.amplitude_max,
        amplitude_mean=final_report.amplitude_mean,
        amplitude_std=final_report.amplitude_std,
        phase_mean_deg=final_report.phase_mean_deg,
        phase_std_deg=final_report.phase_std_deg,
        selected_roots=selected_roots,
        distance_mod_global_phase=distance,
        group_residuals={"power": power_residual},
        joint_group_residuals={
            "input_off_support_rel": float(input_report.off_support_rel),
            "input_off_support_max": float(input_report.off_support_max),
        },
        notes=sorted(set(notes)),
    )
    return exact, report


def roots_of_unity_up_to(max_order: int) -> list[complex]:
    roots: list[complex] = []
    seen: set[tuple[int, int]] = set()
    for order in range(1, int(max_order) + 1):
        for power in range(order):
            root = root_of_unity(order, power)
            key = (int(round(root.real * 10**12)), int(round(root.imag * 10**12)))
            if key not in seen:
                seen.add(key)
                roots.append(root)
    return roots


def infer_monomial_perm_from_blocks(
    D_num: np.ndarray,
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    operation_name: str,
    reject_if_off_support_rel_gt: float,
    reject_if_amplitude_deviation_gt: float,
) -> tuple[np.ndarray, ExactificationReport]:
    arr = np.asarray(D_num, dtype=complex)
    perm = np.full(arr.shape[1], -1, dtype=int)
    for rows, cols in groups:
        block = arr[np.ix_(rows, cols)]
        if block.shape[0] != block.shape[1]:
            raise ValueError(f"{operation_name} block_monomial cleanup requires square support blocks")
        local_rows = np.argmax(np.abs(block), axis=0)
        if len(set(int(row) for row in local_rows)) != len(local_rows):
            raise ValueError(f"{operation_name} block_monomial cleanup found non-bijective internal support")
        for local_col, local_row in enumerate(local_rows):
            perm[int(cols[local_col])] = int(rows[int(local_row)])

    report = analyze_monomial_support(arr, perm)
    if report.support_mismatch_count:
        raise ValueError(f"{operation_name} block_monomial cleanup failed: support mismatch count={report.support_mismatch_count}")
    if report.off_support_rel > reject_if_off_support_rel_gt:
        raise ValueError(
            f"{operation_name} block_monomial cleanup failed: inferred off-support {report.off_support_rel:.3e} exceeds "
            f"{reject_if_off_support_rel_gt:.3e}"
        )
    amp_dev = max(abs(report.amplitude_min - 1.0), abs(report.amplitude_max - 1.0))
    if amp_dev > reject_if_amplitude_deviation_gt:
        raise ValueError(
            f"{operation_name} block_monomial cleanup failed: amplitude deviation {amp_dev:.3e} exceeds "
            f"{reject_if_amplitude_deviation_gt:.3e}"
        )
    return perm, report


def _polar_unitary(block: np.ndarray) -> np.ndarray:
    u, _s, vh = np.linalg.svd(np.asarray(block, dtype=complex), full_matrices=False)
    return u @ vh


def _snap_block_unitary(U: np.ndarray, *, power: int, central_phase: complex) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(U)
    roots = _roots_for_power_phase(power, central_phase)
    snapped = np.array([nearest_root_of_unity(value, roots)[0] for value in eigvals], dtype=complex)
    return eigvecs @ np.diag(snapped) @ np.linalg.inv(eigvecs)


def _gamma_spinful_c3_four_state_template() -> np.ndarray:
    sqrt3 = np.sqrt(3.0)
    return np.array(
        [
            [-0.25 - 0.25j * sqrt3, -0.25 * sqrt3 - 0.75j, 0.0, 0.0],
            [0.25 * sqrt3 + 0.75j, -0.25 - 0.25j * sqrt3, 0.0, 0.0],
            [0.0, 0.0, -0.25 + 0.25j * sqrt3, -0.25 * sqrt3 + 0.75j],
            [0.0, 0.0, 0.25 * sqrt3 - 0.75j, -0.25 + 0.25j * sqrt3],
        ],
        dtype=np.complex128,
    )


def _canonicalize_gamma_spinful_c3_four_state(
    matrix: np.ndarray,
    *,
    mode: str,
    match_tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(matrix, dtype=np.complex128)
    required = mode == "gamma_spinful_c3_4"
    if arr.shape != (4, 4):
        if required:
            raise ValueError(
                "gamma_spinful_c3_4 algebraic exactification requires a 4x4 internal block"
            )
        return arr, {
            "status": "not_applicable",
            "template": "gamma_spinful_c3_4",
            "shape": list(arr.shape),
        }

    base = _gamma_spinful_c3_four_state_template()
    sqrt3 = np.sqrt(3.0)
    roots = [
        1.0 + 0.0j,
        -0.5 + 0.5j * sqrt3,
        -0.5 - 0.5j * sqrt3,
    ]
    best_matrix: np.ndarray | None = None
    best_distance = float("inf")
    best_branch: dict[str, Any] | None = None
    for conjugated in (False, True):
        branch_base = base.conj() if conjugated else base
        for permutation in itertools.permutations(range(4)):
            permuted = branch_base[np.ix_(permutation, permutation)]
            for root_index, root in enumerate(roots):
                candidate = complex(root) * permuted
                distance = float(np.linalg.norm(arr - candidate) / max(np.linalg.norm(candidate), 1.0))
                if distance < best_distance:
                    best_matrix = candidate
                    best_distance = distance
                    best_branch = {
                        "conjugated": bool(conjugated),
                        "permutation": [int(value) for value in permutation],
                        "global_root_power": int(root_index),
                    }

    assert best_matrix is not None and best_branch is not None
    report = {
        "status": "canonicalized" if best_distance <= float(match_tol) else "not_recognized",
        "template": "gamma_spinful_c3_4",
        "match_distance": float(best_distance),
        "match_tolerance": float(match_tol),
        "selected_branch": best_branch,
    }
    if best_distance <= float(match_tol):
        return np.asarray(best_matrix, dtype=np.complex128), report
    if required:
        raise ValueError(
            "C3z block does not match required gamma_spinful_c3_4 algebraic template: "
            f"distance={best_distance:.3e}, tolerance={float(match_tol):.3e}"
        )
    return arr, report


def exactify_block_monomial_representation(
    D_num: np.ndarray,
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    operation_name: str,
    power: int,
    central_phase: complex,
    antiunitary: bool,
    reject_if_off_support_rel_gt: float = 1.0e-6,
    reject_if_amplitude_deviation_gt: float = 2.0e-2,
    algebraic_template: str | None = None,
    algebraic_match_tol: float = 1.0e-5,
) -> tuple[np.ndarray, ExactificationReport]:
    arr = np.asarray(D_num, dtype=complex)
    input_report = analyze_block_support(arr, groups)
    if input_report.off_support_rel > reject_if_off_support_rel_gt:
        raise ValueError(
            f"{operation_name} block exactification failed: off-support pollution {input_report.off_support_rel:.3e} exceeds "
            f"{reject_if_off_support_rel_gt:.3e}"
        )
    amp_dev = max(abs(input_report.amplitude_min - 1.0), abs(input_report.amplitude_max - 1.0))
    if amp_dev > reject_if_amplitude_deviation_gt:
        raise ValueError(
            f"{operation_name} block exactification failed: amplitude deviation {amp_dev:.3e} exceeds "
            f"{reject_if_amplitude_deviation_gt:.3e}"
        )
    unitary_blocks = [_polar_unitary(arr[np.ix_(rows, cols)]) for rows, cols in groups]
    ref = unitary_blocks[0]
    aligned_blocks: list[np.ndarray] = []
    for block in unitary_blocks:
        alpha = np.vdot(block, ref)
        if abs(alpha) > 0.0:
            alpha = alpha / abs(alpha)
        else:
            alpha = 1.0 + 0.0j
        aligned_blocks.append(block * alpha)
    U_avg = _polar_unitary(sum(aligned_blocks))
    U_exact = _snap_block_unitary(U_avg, power=power, central_phase=central_phase)
    algebraic_report: dict[str, Any] = {}
    template_mode = str(algebraic_template or "none").strip().lower()
    if template_mode not in {"none", "auto", "gamma_spinful_c3_4"}:
        raise ValueError(f"Unsupported algebraic block template {algebraic_template!r}")
    if template_mode in {"auto", "gamma_spinful_c3_4"} and operation_name == "C3z":
        U_exact, algebraic_report = _canonicalize_gamma_spinful_c3_four_state(
            U_exact,
            mode=template_mode,
            match_tol=float(algebraic_match_tol),
        )
    exact = np.zeros_like(arr, dtype=complex)
    for rows, cols in groups:
        exact[np.ix_(rows, cols)] = U_exact
    if antiunitary:
        power_residual = antiunitary_square_residual(U_exact, central_phase)
    else:
        power_residual = float(
            np.linalg.norm(
                np.linalg.matrix_power(U_exact, int(power))
                - complex(central_phase) * np.eye(U_exact.shape[0])
            )
            / np.sqrt(U_exact.shape[0])
        )
    if power_residual > 1.0e-8:
        raise ValueError(
            f"{operation_name} block exactification failed: group power residual {power_residual:.3e} exceeds 1.000e-08"
        )
    alpha = np.vdot(exact, arr) / np.vdot(exact, exact)
    alpha = alpha / abs(alpha)
    distance = float(np.linalg.norm(arr - alpha * exact) / max(np.linalg.norm(exact), 1.0))
    final_report = analyze_block_support(exact, groups)
    exact_report = ExactificationReport(
        status="exactified",
        off_support_rel=final_report.off_support_rel,
        off_support_max=final_report.off_support_max,
        support_mismatch_count=0,
        amplitude_min=final_report.amplitude_min,
        amplitude_max=final_report.amplitude_max,
        amplitude_mean=final_report.amplitude_mean,
        amplitude_std=final_report.amplitude_std,
        distance_mod_global_phase=distance,
        group_residuals={"power": power_residual},
        joint_group_residuals={
            "input_off_support_rel": float(input_report.off_support_rel),
            "input_off_support_max": float(input_report.off_support_max),
        },
        algebraic_canonicalization=algebraic_report,
        notes=[
            "block_exactification",
            *(["algebraic_gamma_spinful_c3_4"] if algebraic_report.get("status") == "canonicalized" else []),
        ],
    )
    return exact, exact_report


def antiunitary_square_residual(D_a: np.ndarray, phase: complex) -> float:
    arr = np.asarray(D_a, dtype=complex)
    ident = np.eye(arr.shape[0], dtype=complex)
    return float(np.linalg.norm(arr @ arr.conj() - complex(phase) * ident) / np.sqrt(arr.shape[0]))


def compare_operation_convention(
    op1: OperationAction,
    D1: np.ndarray,
    op2: OperationAction,
    D2: np.ndarray,
) -> dict[str, Any]:
    arr1 = np.asarray(D1, dtype=complex)
    arr2 = np.asarray(D2, dtype=complex)
    alpha = np.vdot(arr1, arr2) / np.vdot(arr1, arr1)
    if abs(alpha) > 0.0:
        alpha /= abs(alpha)
    diff = arr1 * alpha - arr2
    matrix_distance = float(np.linalg.norm(diff) / max(np.linalg.norm(arr2), 1.0))
    k_map_equal = _jsonable(op1.k_map) == _jsonable(op2.k_map)
    q_map_equal = _jsonable(op1.q_map) == _jsonable(op2.q_map)
    sector_map_equal = _jsonable(op1.sector_map) == _jsonable(op2.sector_map)
    antiunitary_equal = bool(op1.antiunitary) == bool(op2.antiunitary)
    close = bool(matrix_distance < 1.0e-6)
    interchangeable = bool(close and k_map_equal and q_map_equal and sector_map_equal and antiunitary_equal)
    message = None
    if close and not interchangeable:
        message = "Matrices are close but operations are not interchangeable for term symmetrization."
    return {
        "matrix_distance_mod_global_phase": matrix_distance,
        "matrix_close_up_to_global_phase": close,
        "k_map_equal": k_map_equal,
        "q_map_equal": q_map_equal,
        "sector_map_equal": sector_map_equal,
        "antiunitary_equal": antiunitary_equal,
        "interchangeable_for_term_symmetrization": interchangeable,
        "message": message,
    }


def _parse_allowed_roots(spec: object) -> list[complex]:
    if isinstance(spec, str):
        if spec == "sixth_roots_cube_minus_one":
            return [complex(np.exp(1j * (np.pi + 2.0 * np.pi * m) / 3.0)) for m in range(3)]
        if spec == "third_roots":
            return [root_of_unity(3, m) for m in range(3)]
        if spec == "fourth_roots":
            return [root_of_unity(4, m) for m in range(4)]
        if spec == "sixth_roots":
            return [root_of_unity(6, m) for m in range(6)]
        if spec == "twelfth_roots":
            return [root_of_unity(12, m) for m in range(12)]
        if spec.startswith("roots_up_to_"):
            return roots_of_unity_up_to(int(spec.removeprefix("roots_up_to_")))
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        out: list[complex] = []
        for item in spec:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
                out.append(complex(float(item[0]), float(item[1])))
            else:
                out.append(complex(item))
        return out
    return []


def _operation_exactification_config(exact_cfg: Mapping[str, Any], operation_name: str) -> dict[str, Any]:
    op_cfgs = exact_cfg.get("operations", {})
    if not isinstance(op_cfgs, Mapping):
        return {}
    entry = op_cfgs.get(operation_name, {})
    if entry is None:
        return {}
    if not isinstance(entry, Mapping):
        raise ValueError(f"exactification.operations.{operation_name} must be a mapping")
    return dict(entry)


def _default_auto_monomial_cleanup(
    operation_name: str,
    op: OperationAction,
    record: Mapping[str, Any],
) -> bool:
    if op.antiunitary:
        return False
    family = op.canonical_name or operation_name
    if family == "C2" or str(operation_name).startswith("C2_"):
        return True
    if family == "C3z":
        return str(record.get("source_matrix_role", "")) == "raw_h_sewing_action"
    return False


def _central_phase_for_operation(operation_name: str, cfg: Mapping[str, Any], *, antiunitary: bool) -> tuple[int, complex]:
    central_cfg = cfg.get("central_phase", {})
    if isinstance(central_cfg, Mapping):
        for key, value in central_cfg.items():
            raw = str(key).replace(" ", "")
            if not raw.startswith(operation_name):
                continue
            if "^" in raw:
                _, power_str = raw.split("^", 1)
                return int(power_str), complex(value)
    if operation_name == "C3z":
        return 3, complex(-1.0, 0.0)
    if antiunitary:
        return 2, complex(1.0, 0.0)
    return 1, complex(1.0, 0.0)


def _complex_from_metadata(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            return complex(float(value[0]), float(value[1]))
    return complex(value)


def _relation_operation_matches(relation: Mapping[str, Any], operation_name: str) -> bool:
    raw_name = relation.get("name", relation.get("operation", ""))
    if raw_name in {None, ""}:
        return True
    text = str(raw_name).replace(" ", "")
    relation_op = text.split("^", 1)[0]
    names = {str(operation_name)}
    if operation_name == "C3z":
        names.add("C3")
    return relation_op in names


def _power_relation_from_record(record: Mapping[str, Any], operation_name: str) -> tuple[int, complex, dict[str, Any]] | None:
    relations = record.get("group_relations", [])
    if isinstance(relations, Mapping):
        items: list[Mapping[str, Any]] = []
        for key, phase in relations.items():
            raw = str(key).replace(" ", "")
            if "^" not in raw:
                continue
            op_name, power_text = raw.split("^", 1)
            items.append({"type": "power", "name": f"{op_name}^{power_text}", "power": int(power_text), "phase": phase})
    elif isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)):
        items = [item for item in relations if isinstance(item, Mapping)]
    else:
        items = []
    for relation in items:
        if str(relation.get("type", "power")) != "power":
            continue
        if "power" not in relation or "phase" not in relation:
            continue
        if not _relation_operation_matches(relation, operation_name):
            continue
        parsed = {
            "type": "power",
            "name": relation.get("name", f"{operation_name}^{relation['power']}"),
            "power": int(relation["power"]),
            "phase": _complex_from_metadata(relation["phase"]),
        }
        return int(parsed["power"]), complex(parsed["phase"]), parsed
    return None


def _power_relation_for_exactification(
    *,
    record: Mapping[str, Any],
    operation_name: str,
    op_cfg: Mapping[str, Any],
    exact_cfg: Mapping[str, Any],
    antiunitary: bool,
    require_group_relations: bool,
) -> tuple[int, complex, str, dict[str, Any] | None, bool]:
    relation = _power_relation_from_record(record, operation_name)
    if relation is not None:
        power, central_phase, relation_record = relation
        return power, central_phase, "manifest", relation_record, False
    if require_group_relations:
        raise ValueError(
            f"exactification strict mode requires explicit group relation metadata for operation {operation_name!r}"
        )
    if "power" in op_cfg or "central_phase" in op_cfg:
        power = int(op_cfg.get("power", 2 if antiunitary else 1))
        central_phase = _complex_from_metadata(op_cfg.get("central_phase", 1.0))
        return power, central_phase, "operation_config", {"type": "power", "power": power, "phase": central_phase}, False
    power, central_phase = _central_phase_for_operation(operation_name, exact_cfg, antiunitary=antiunitary)
    return power, central_phase, "legacy_config_or_name", {"type": "power", "power": power, "phase": central_phase}, True


def _joint_exactification_config(
    exact_cfg: Mapping[str, Any],
) -> tuple[JointExactificationConfig | None, bool]:
    raw = exact_cfg.get("joint_exactification", {})
    if isinstance(raw, bool):
        enabled = bool(raw)
        settings: Mapping[str, Any] = {}
    elif raw is None:
        enabled = True
        settings = {}
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", True))
        settings = raw
    else:
        raise ValueError("exactification.joint_exactification must be a mapping or boolean")
    required = bool(settings.get("required", False))
    if not enabled:
        if required:
            raise ValueError("joint exactification cannot be both disabled and required")
        return None, False
    allowed = {
        "enabled",
        "required",
        "max_rms_correction",
        "max_route_correction",
        "central_branch_margin",
        "max_iterations",
        "condition_limit",
    }
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"Unsupported joint exactification keys: {unknown}")
    kwargs = {
        key: settings[key]
        for key in (
            "max_rms_correction",
            "max_route_correction",
            "central_branch_margin",
            "max_iterations",
            "condition_limit",
        )
        if key in settings
    }
    return JointExactificationConfig(enabled=True, **kwargs), required


def _joint_fiber_indices(labels: Sequence[BasisLabel]) -> tuple[tuple[int, ...], ...]:
    fibers: dict[tuple[str, tuple[int, int]], list[BasisLabel]] = {}
    for label in labels:
        if label.q_integer is None:
            raise JointExactificationError(
                f"basis label {label.index} lacks exact integer Q coordinates"
            )
        key = (str(label.sector), tuple(int(value) for value in label.q_integer))
        fibers.setdefault(key, []).append(label)
    ordered: list[tuple[int, ...]] = []
    for members in fibers.values():
        members.sort(key=lambda item: (int(item.orbital), int(item.index)))
        ordered.append(tuple(int(item.index) for item in members))
    return tuple(ordered)


def _joint_fiber_permutation(
    label_action: LabelAction,
    fiber_indices: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    index_to_fiber = {
        int(index): int(fiber)
        for fiber, members in enumerate(fiber_indices)
        for index in members
    }
    permutation: list[int] = []
    for source, members in enumerate(fiber_indices):
        targets = {
            index_to_fiber[int(label_action.perm[int(index)])]
            for index in members
        }
        if len(targets) != 1:
            raise JointExactificationError(
                "resolved operation does not map a complete continuum fiber to one "
                f"target fiber: operation={label_action.operation.name!r}, "
                f"source={source}, targets={sorted(targets)}"
            )
        target = next(iter(targets))
        if len(tuple(fiber_indices[target])) != len(tuple(members)):
            raise JointExactificationError(
                "resolved operation changes a continuum fiber dimension: "
                f"operation={label_action.operation.name!r}, source={source}, "
                f"target={target}"
            )
        permutation.append(int(target))
    return tuple(permutation)


def _joint_presentation_record(
    *,
    name: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_action = report.get("resolved_action")
    relation = report.get("group_relation")
    if not isinstance(resolved_action, Mapping):
        raise JointExactificationError(
            f"operation {name!r} lacks a resolved model action"
        )
    if report.get("group_relation_source") != "manifest" or not isinstance(
        relation, Mapping
    ):
        raise JointExactificationError(
            f"operation {name!r} lacks an explicit manifest power relation"
        )
    power_relation = {
        "type": "power",
        "operation": str(name),
        "power": int(relation["power"]),
        "phase": relation["phase"],
    }
    return {
        "name": str(name),
        "operation": str(name),
        "antiunitary": bool(resolved_action.get("antiunitary", False)),
        "declared_model_action": copy.deepcopy(dict(resolved_action)),
        "group_relations": [power_relation],
    }


def exactify_loaded_symmetry_source(
    *,
    loaded_metadata: Mapping[str, Any],
    matrices: Mapping[str, np.ndarray],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    sectors: Sequence[Mapping[str, Any]],
    n_orb: tuple[int, int],
    bM1: np.ndarray,
    bM2: np.ndarray,
    raw_config: Mapping[str, Any],
    rotation_deg: float = 0.0,
    output_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    exact_cfg = raw_config.get("exactification", {})
    if not isinstance(exact_cfg, Mapping):
        raise ValueError("symmetry_source.exactification must be a mapping")
    tol = float(exact_cfg.get("q_tol", 1.0e-6))
    strict = bool(exact_cfg.get("strict", False))
    require_explicit_action_candidates = bool(
        exact_cfg.get("require_explicit_action_candidates", strict)
    )
    forbid_inferred_q_offset = bool(exact_cfg.get("forbid_inferred_q_offset", strict))
    require_group_relations = bool(exact_cfg.get("require_group_relations", strict))
    q_lookup = {"qset1": np.asarray(Q_set1, dtype=float), "qset2": np.asarray(Q_set2, dtype=float)}
    q_offsets: dict[str, np.ndarray] = {}
    inferred_q_offsets: dict[str, bool] = {}
    for sector in sectors:
        name = str(sector.get("name"))
        qset_name = str(sector.get("qset", "qset1"))
        qset = q_lookup[qset_name]
        if "q_offset" not in sector:
            if forbid_inferred_q_offset:
                raise ValueError(
                    f"exactification strict mode forbids inferred q_offset for sector {name!r}"
                )
            q_offset = infer_q_offset_from_qset(qset, bM1, bM2)
            inferred_q_offsets[name] = True
        else:
            q_offset = np.asarray(sector["q_offset"], dtype=float)
            try:
                canonical_q_coordinates(qset[0], name, bM1, bM2, q_offset, tol)
                inferred_q_offsets[name] = False
            except Exception:
                if forbid_inferred_q_offset:
                    raise ValueError(
                        f"exactification strict mode forbids inferred q_offset for sector {name!r}"
                    )
                q_offset = infer_q_offset_from_qset(qset, bM1, bM2)
                inferred_q_offsets[name] = True
        q_offsets[name] = q_offset
    labels = build_basis_labels(
        Q_set1=np.asarray(Q_set1, dtype=float),
        Q_set2=np.asarray(Q_set2, dtype=float),
        sectors=sectors,
        n_orb1=int(n_orb[0]),
        n_orb2=int(n_orb[1]),
        bM1=np.asarray(bM1, dtype=float),
        bM2=np.asarray(bM2, dtype=float),
        tol=tol,
        q_offsets=q_offsets,
        forbid_inferred_q_offset=forbid_inferred_q_offset,
    )
    out: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    joint_config, joint_required = _joint_exactification_config(exact_cfg)
    joint_fiber_indices = _joint_fiber_indices(labels)
    joint_fiber_permutations: dict[str, tuple[int, ...]] = {}
    joint_antiunitary: dict[str, bool] = {}
    op_records = loaded_metadata.get("operations", [])
    if not isinstance(op_records, Sequence) or isinstance(op_records, (str, bytes)):
        raise ValueError("loaded_metadata.operations must be a list")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("exactified_*.npy", "*_exactification_report.json"):
            for old_file in output_dir.glob(pattern):
                if old_file.is_file():
                    old_file.unlink()
    for record in op_records:
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("name"))
        op_cfg = _operation_exactification_config(exact_cfg, name)
        discover_action_candidates = bool(
            op_cfg.get("discover_action_candidates", exact_cfg.get("discover_action_candidates", False))
        )
        accept_support_resolved_action = bool(
            op_cfg.get("accept_support_resolved_action", exact_cfg.get("accept_support_resolved_action", False))
        )
        action_candidates_raw = op_cfg.get("action_candidates")
        explicit_action_candidates = (
            action_candidates_raw
            if isinstance(action_candidates_raw, Sequence) and not isinstance(action_candidates_raw, (str, bytes))
            else None
        )
        candidates = _candidate_operation_actions(
            record,
            sectors,
            rotation_deg=rotation_deg,
            explicit_candidates=explicit_action_candidates,
            require_explicit_action_candidates=require_explicit_action_candidates,
            discover_action_candidates=discover_action_candidates,
        )
        D_num = np.asarray(matrices[name], dtype=complex)
        candidate_support_residuals = _candidate_support_residuals(
            D_num,
            candidates,
            labels,
            np.asarray(bM1),
            np.asarray(bM2),
            q_offsets,
            max(tol, 1.0e-8),
        )
        candidate = _choose_best_action_candidate(
            D_num,
            candidates,
            labels,
            np.asarray(bM1),
            np.asarray(bM2),
            q_offsets,
            max(tol, 1.0e-8),
        )
        if candidate is None:
            allow_no_candidate = bool(
                op_cfg.get(
                    "allow_no_complete_support_candidate",
                    exact_cfg.get("allow_no_complete_support_candidate", False),
                )
            )
            if not allow_no_candidate:
                raise ValueError(
                    f"{name} exactification failed: No candidate produced complete geometry support"
                )
            manifest_op = candidates[0]
            manifest_label_action = build_label_action(
                labels,
                manifest_op,
                np.asarray(bM1),
                np.asarray(bM2),
                q_offsets,
                max(tol, 1.0e-8),
            )
            manifest_action = _action_candidate_summary(manifest_op)
            reports[name] = {
                "input_matrix_file": record.get("matrix_file"),
                "operation_alias": record.get("operation_alias"),
                "canonical_physical_operation": record.get("canonical_physical_operation", name),
                "physical_parent": record.get("physical_parent", record.get("canonical_physical_operation", name)),
                "representation_level": record.get("representation_level"),
                "effective_name": record.get("effective_name"),
                "approximation": copy.deepcopy(record.get("approximation")),
                "derived_from": copy.deepcopy(record.get("derived_from")),
                "source_matrix_role": record["source_matrix_role"],
                "source_gauge": record["source_gauge"],
                "target_role": record["target_role"],
                "resolved_action": manifest_action,
                "manifest_model_action": manifest_action,
                "selected_action_candidate": None,
                "support_resolution": {
                    "action_mismatch": False,
                    "candidate_source": "no_complete_support_candidate",
                    "declared_model_action": manifest_action,
                    "selected_action_candidate": None,
                    "candidate_support_residuals": candidate_support_residuals,
                },
                "preferred_mode": "skipped",
                "label_action": {
                    "perm_complete": manifest_label_action.diagnostics["perm_complete"],
                    "missing_count": manifest_label_action.diagnostics["missing_count"],
                },
                "report": {
                    "status": "skipped_no_complete_support_candidate",
                    "reason": "No candidate produced complete geometry support; keeping loaded matrix and manifest action.",
                },
                "q_offsets": {sector_name: value.tolist() for sector_name, value in q_offsets.items()},
                "inferred_q_offsets": dict(inferred_q_offsets),
                "inference_used": bool(any(inferred_q_offsets.values()) or not op_cfg.get("action_candidates")),
            }
            out[name] = D_num
            continue
        op = candidate["operation"]
        manifest_action = _action_candidate_summary(candidates[0])
        selected_action = _action_candidate_summary(op)
        action_mismatch = not _action_summaries_equal(manifest_action, selected_action)
        if not action_mismatch:
            candidate_source = "manifest_model_action"
        elif explicit_action_candidates is not None:
            candidate_source = "explicit_action_candidates_report"
        elif discover_action_candidates:
            candidate_source = "support_discovery"
        else:
            candidate_source = "support_resolution"
        if action_mismatch and not accept_support_resolved_action:
            raise ValueError(
                f"{name} exactification selected a support action that differs from manifest model_action; "
                "set accept_support_resolved_action=true only after auditing the mismatch"
            )
        resolved_action = copy.deepcopy(selected_action if (action_mismatch and accept_support_resolved_action) else manifest_action)
        provenance = None
        if action_mismatch and accept_support_resolved_action:
            declared_support_residual = next(
                (
                    row.get("monomial_off_support_rel", row.get("block_off_support_rel"))
                    for row in candidate_support_residuals
                    if _action_summaries_equal(row.get("action", {}), manifest_action)
                ),
                None,
            )
            selected_support_residual = next(
                (
                    row.get("monomial_off_support_rel", row.get("block_off_support_rel"))
                    for row in candidate_support_residuals
                    if _action_summaries_equal(row.get("action", {}), selected_action)
                ),
                None,
            )
            provenance = {
                "source": "support_exactification",
                "accepted_by": (
                    "explicit_action_candidates"
                    if explicit_action_candidates is not None
                    else "kp_projected_basis_inference"
                ),
                "accepted_by_user": explicit_action_candidates is not None,
                "declared_model_action": manifest_action,
                "selected_action_candidate": selected_action,
                "declared_support_residual": declared_support_residual,
                "selected_support_residual": selected_support_residual,
            }
            resolved_action["provenance"] = provenance
        if strict and action_mismatch:
            raise ValueError(
                f"{name} exactification selected a support action that differs from manifest model_action in strict mode"
            )
        label_action = candidate["label_action"]
        joint_fiber_permutations[name] = _joint_fiber_permutation(
            label_action,
            joint_fiber_indices,
        )
        joint_antiunitary[name] = bool(op.antiunitary)
        support_mode = str(op_cfg.get("support_mode", "auto")).lower()
        if support_mode not in {"auto", "monomial", "block", "block_monomial"}:
            raise ValueError(f"Unsupported support_mode for {name}: {support_mode!r}")
        preferred_mode = candidate["preferred_mode"]
        reject_off = float(op_cfg.get("reject_if_off_support_rel_gt", exact_cfg.get("reject_if_off_support_rel_gt", 1.0e-6)))
        reject_amp = float(op_cfg.get("reject_if_amplitude_deviation_gt", exact_cfg.get("reject_if_amplitude_deviation_gt", 2.0e-2)))
        monomial_report = candidate["monomial_report"]
        monomial_amp_dev = max(
            abs(float(monomial_report.amplitude_min) - 1.0),
            abs(float(monomial_report.amplitude_max) - 1.0),
        )
        if (
            support_mode == "auto"
            and not monomial_report.support_mismatch_count
            and float(monomial_report.off_support_rel) <= reject_off
            and monomial_amp_dev <= reject_amp
            and not _block_support_is_substantially_cleaner(
                monomial_report,
                candidate["block_report"],
                reject_if_off_support_rel_gt=reject_off,
                reject_if_amplitude_deviation_gt=reject_amp,
            )
        ):
            preferred_mode = "monomial"
        if support_mode == "monomial":
            preferred_mode = "monomial"
        elif support_mode == "block":
            preferred_mode = "block"
        elif support_mode == "block_monomial":
            preferred_mode = "block_monomial"
        support_report = candidate["monomial_report"] if preferred_mode == "monomial" else candidate["block_report"]
        power, central_phase, group_relation_source, group_relation, group_relation_inferred = (
            _power_relation_for_exactification(
                record=record,
                operation_name=name,
                op_cfg=op_cfg,
                exact_cfg=exact_cfg,
                antiunitary=op.antiunitary,
                require_group_relations=require_group_relations,
            )
        )
        allowed_roots_spec = op_cfg.get("allowed_roots")
        if allowed_roots_spec is None:
            allowed_roots_map = exact_cfg.get("allowed_roots", {})
            allowed_roots_spec = allowed_roots_map.get(name) if isinstance(allowed_roots_map, Mapping) else None
        allowed_roots = _parse_allowed_roots(allowed_roots_spec)
        if not allowed_roots and op.antiunitary:
            allowed_roots = [1.0 + 0.0j]
        phase_classes = op_cfg.get("phase_classes")
        if phase_classes is None:
            phase_classes = exact_cfg.get("phase_classes", "global")
            if isinstance(phase_classes, Mapping):
                phase_classes = phase_classes.get(name, "global")
        if preferred_mode == "monomial":
            if not allowed_roots and not (int(power) == 2 and not op.antiunitary):
                allowed_roots = _roots_for_power_phase(power, central_phase)
            D_exact, report = exactify_1d_monomial_phases(
                D_num,
                label_action.perm,
                labels=labels,
                phase_classes=phase_classes,
                allowed_roots=allowed_roots,
                operation_name=name,
                power=power,
                central_phase=central_phase,
                antiunitary=op.antiunitary,
                reject_if_off_support_rel_gt=reject_off,
                reject_if_amplitude_deviation_gt=reject_amp,
            )
        else:
            groups = candidate["block_groups"]
            if not groups:
                raise ValueError(f"{name} exactification could not construct block support groups")
            block_exact, block_report = exactify_block_monomial_representation(
                D_num,
                groups,
                operation_name=name,
                power=power,
                central_phase=central_phase,
                antiunitary=op.antiunitary,
                reject_if_off_support_rel_gt=reject_off,
                reject_if_amplitude_deviation_gt=reject_amp,
                algebraic_template=op_cfg.get("algebraic_template", exact_cfg.get("algebraic_template")),
                algebraic_match_tol=float(
                    op_cfg.get("algebraic_match_tol", exact_cfg.get("algebraic_match_tol", 1.0e-5))
                ),
            )
            D_exact, report = block_exact, block_report
            cleanup_required = preferred_mode == "block_monomial"
            cleanup_allowed = bool(op_cfg.get("auto_monomial_cleanup", _default_auto_monomial_cleanup(name, op, record)))
            if cleanup_required or cleanup_allowed:
                cleanup_roots = allowed_roots or roots_of_unity_up_to(
                    int(op_cfg.get("monomial_root_order_max", exact_cfg.get("monomial_root_order_max", 12)))
                )
                cleanup_tol = float(
                    op_cfg.get("monomial_cleanup_tol", exact_cfg.get("monomial_cleanup_tol", 1.0e-4))
                )
                try:
                    cleanup_perm, cleanup_support_report = infer_monomial_perm_from_blocks(
                        block_exact,
                        groups,
                        operation_name=name,
                        reject_if_off_support_rel_gt=cleanup_tol,
                        reject_if_amplitude_deviation_gt=reject_amp,
                    )
                    D_exact, cleanup_report = exactify_1d_monomial_phases(
                        block_exact,
                        cleanup_perm,
                        labels=labels,
                        phase_classes=op_cfg.get("monomial_phase_classes", "matrix_element"),
                        allowed_roots=cleanup_roots,
                        operation_name=name,
                        power=power,
                        central_phase=central_phase,
                        antiunitary=op.antiunitary,
                        reject_if_off_support_rel_gt=cleanup_tol,
                        reject_if_amplitude_deviation_gt=reject_amp,
                    )
                except ValueError:
                    if cleanup_required:
                        raise
                else:
                    preferred_mode = "block_monomial"
                    inferred_note = "inferred_monomial_support"
                    cleanup_report.notes = [*block_report.notes, inferred_note, "block_monomial_cleanup"]
                    cleanup_report.joint_group_residuals = {
                        "block_power": float(block_report.group_residuals.get("power", 0.0)),
                        "cleanup_power": float(cleanup_report.group_residuals.get("power", 0.0)),
                        "cleanup_input_off_support_rel": float(cleanup_support_report.off_support_rel),
                        "cleanup_input_off_support_max": float(cleanup_support_report.off_support_max),
                    }
                    report = cleanup_report
        out[name] = D_exact
        reports[name] = {
            "input_matrix_file": record.get("matrix_file"),
            "operation_alias": record.get("operation_alias"),
            "canonical_physical_operation": record.get("canonical_physical_operation", name),
            "physical_parent": record.get("physical_parent", record.get("canonical_physical_operation", name)),
            "representation_level": record.get("representation_level"),
            "effective_name": record.get("effective_name"),
            "approximation": copy.deepcopy(record.get("approximation")),
            "derived_from": copy.deepcopy(record.get("derived_from")),
            "source_matrix_role": record["source_matrix_role"],
            "source_gauge": record["source_gauge"],
            "target_role": record["target_role"],
            "resolved_action": resolved_action,
            "manifest_model_action": manifest_action,
            "selected_action_candidate": selected_action,
            "support_resolution": {
                "action_mismatch": bool(action_mismatch),
                "candidate_source": candidate_source,
                "declared_model_action": manifest_action,
                "selected_action_candidate": selected_action,
                "declared_support_residual": None if provenance is None else provenance["declared_support_residual"],
                "selected_support_residual": None if provenance is None else provenance["selected_support_residual"],
                "matrix_kind": "action",
                "matrix_source": "raw_h_action_projection",
                "candidate_support_residuals": candidate_support_residuals,
            },
            "preferred_mode": preferred_mode,
            "group_relation_source": group_relation_source,
            "group_relation": _jsonable(group_relation),
            "label_action": {
                "perm_complete": label_action.diagnostics["perm_complete"],
                "missing_count": label_action.diagnostics["missing_count"],
            },
            "support_diagnostics": support_report.to_dict(),
            "report": report.to_dict(),
            "q_offsets": {sector_name: value.tolist() for sector_name, value in q_offsets.items()},
            "inferred_q_offsets": dict(inferred_q_offsets),
            "inference_used": bool(any(inferred_q_offsets.values()) or group_relation_inferred or not op_cfg.get("action_candidates")),
        }
        if provenance is not None:
            reports[name]["support_resolution"]["provenance"] = provenance
        if output_dir is not None:
            np.save(output_dir / f"exactified_{name}.npy", D_exact)
            with (output_dir / f"{name.lower()}_exactification_report.json").open("w", encoding="utf-8") as handle:
                json.dump(_jsonable(reports[name]), handle, indent=2)

    if joint_config is not None and out:
        try:
            presentation_records = [
                _joint_presentation_record(name=name, report=reports[name])
                for name in out
            ]
            presentation = compile_continuum_magnetic_presentation(
                presentation_records
            )
        except JointExactificationError as exc:
            if joint_required:
                raise ValueError(
                    "joint exactification is required but the continuum magnetic "
                    f"presentation is incomplete or unsupported: {exc}"
                ) from exc
        else:
            reference_actions = {
                generator.name: extract_block_route_action(
                    out[generator.name],
                    name=generator.name,
                    antiunitary=joint_antiunitary[generator.name],
                    fiber_indices=joint_fiber_indices,
                    fiber_permutation=joint_fiber_permutations[generator.name],
                    off_route_bound=0.0,
                )
                for generator in presentation.generators
            }
            phase_source_report: dict[str, Any] | None = None
            try:
                result = joint_exactify_block_actions(
                    reference_actions,
                    presentation,
                    config=joint_config,
                )
            except JointExactificationBranchError as canonical_error:
                result, phase_source_report = phase_preserving_joint_source(
                    {
                        generator.name: matrices[generator.name]
                        for generator in presentation.generators
                    },
                    reference_actions,
                    presentation,
                    config=joint_config,
                )
                phase_source_report["canonical_source_rejection"] = str(
                    canonical_error
                )
            artifact_hash = str(result.artifact_metadata["artifact_hash"])
            for generator in presentation.generators:
                name = generator.name
                out[name] = materialize_block_route_action(result.actions[name])
                operation_summary = {
                    "status": "certified",
                    "artifact_hash": artifact_hash,
                    "presentation_source": result.presentation.source,
                    "relation_certification_bound": float(
                        result.report["relation_certification_bound"]
                    ),
                    "pre_relation_residual_max": float(
                        result.report["pre_relation_residual_max"]
                    ),
                    "post_relation_residual_max": float(
                        result.report["post_relation_residual_max"]
                    ),
                    "route_correction_rms": float(
                        result.report["route_correction_rms_by_operation"][name]
                    ),
                    "route_correction_max": float(
                        result.report["route_correction_max_by_operation"][name]
                    ),
                    "root_report": _jsonable(dict(result.report)),
                }
                reports[name]["joint_exactification"] = operation_summary
                reports[name]["report"].setdefault(
                    "joint_group_residuals", {}
                ).update(
                    {
                        relation_name: float(values["max"])
                        for relation_name, values in result.report[
                            "post_relation_residuals"
                        ].items()
                    }
                )
            reports["__joint_exactification__"] = {
                "metadata": dict(result.artifact_metadata),
                "arrays": dict(result.artifact_arrays),
                "report": dict(result.report),
            }
            if phase_source_report is not None:
                reports["__joint_exactification__"][
                    "phase_preserving_source"
                ] = phase_source_report
            if output_dir is not None:
                for generator in presentation.generators:
                    name = generator.name
                    np.save(output_dir / f"exactified_{name}.npy", out[name])
                    with (
                        output_dir / f"{name.lower()}_exactification_report.json"
                    ).open("w", encoding="utf-8") as handle:
                        json.dump(_jsonable(reports[name]), handle, indent=2)
    return out, reports
