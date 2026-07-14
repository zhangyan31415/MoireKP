"""Canonicalize model-Q coordinates against certified discrete symmetry actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CANONICAL_Q_VERSION = "constrained_group_action_projection_v1"


class QCanonicalizationError(ValueError):
    """The authored Q geometry cannot be certified against its discrete action."""


@dataclass(frozen=True)
class CanonicalQResult:
    raw_q: Mapping[str, np.ndarray]
    canonical_q: Mapping[str, np.ndarray]
    artifact: Mapping[str, Any]


def _readonly_q_mapping(source: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, values in source.items():
        array = np.array(values, dtype=np.float64, order="C", copy=True)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
            raise QCanonicalizationError(
                f"Q sector {name!r} must have shape (N, 2) with N > 0, got {array.shape}"
            )
        array.setflags(write=False)
        out[str(name)] = array
    if not out:
        raise QCanonicalizationError("Q canonicalization requires at least one sector")
    return MappingProxyType(out)


def _q_map_matrix(q_map: object) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(q_map, Mapping):
        raise QCanonicalizationError("a certified Q action requires explicit q_map metadata")
    if "linear_matrix" in q_map:
        matrix = np.asarray(q_map["linear_matrix"], dtype=np.float64)
    else:
        map_type = str(q_map.get("type", "")).lower()
        if map_type == "rotation":
            theta = np.deg2rad(float(q_map.get("angle_deg", 0.0)))
            matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                dtype=np.float64,
            )
        elif map_type == "reflection":
            theta = np.deg2rad(float(q_map.get("axis_deg", 0.0)))
            axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
            matrix = 2.0 * np.outer(axis, axis) - np.eye(2, dtype=np.float64)
        elif map_type == "negation":
            matrix = -np.eye(2, dtype=np.float64)
        elif map_type == "identity":
            matrix = np.eye(2, dtype=np.float64)
        else:
            raise QCanonicalizationError(
                f"unsupported explicit q_map type {q_map.get('type')!r}"
            )
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise QCanonicalizationError(f"q_map linear matrix must be finite (2, 2), got {matrix}")
    translation_raw = q_map.get(
        "translation",
        q_map.get("affine_offset", q_map.get("offset", [0.0, 0.0])),
    )
    translation = np.asarray(translation_raw, dtype=np.float64)
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise QCanonicalizationError(
            f"q_map translation must be a finite length-two vector, got {translation_raw!r}"
        )
    return matrix, translation


def _operation_q_map(operation: Mapping[str, Any]) -> object:
    for container_name in (
        "internal_resolved_action",
        "model_action",
        "declared_model_action",
    ):
        container = operation.get(container_name)
        if isinstance(container, Mapping) and "q_map" in container:
            return container["q_map"]
    if "q_map" in operation:
        return operation["q_map"]
    raise QCanonicalizationError(
        f"operation {operation.get('name', '<unnamed>')!r} lacks explicit q_map metadata"
    )


def q_geometry_hash(source: Mapping[str, np.ndarray], sector_order: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(CANONICAL_Q_VERSION.encode("utf-8"))
    for sector in sector_order:
        encoded = str(sector).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="little", signed=False))
        digest.update(encoded)
        array = np.asarray(source[sector], dtype="<f8", order="C")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonicalize_q_geometry(
    q_by_sector: Mapping[str, np.ndarray],
    operations: Sequence[Mapping[str, Any]],
    *,
    relative_cleanup_tolerance: float = 1.0e-8,
    absolute_cleanup_tolerance: float | None = None,
) -> CanonicalQResult:
    """Return the nearest Q coordinates satisfying all certified discrete actions."""

    if relative_cleanup_tolerance <= 0.0:
        raise QCanonicalizationError("relative_cleanup_tolerance must be positive")
    raw = _readonly_q_mapping(q_by_sector)
    sector_order = tuple(raw)
    offsets: dict[str, int] = {}
    point_count = 0
    for sector in sector_order:
        offsets[sector] = point_count
        point_count += int(raw[sector].shape[0])
    raw_matrix = np.concatenate([raw[sector] for sector in sector_order], axis=0)
    raw_vector = raw_matrix.reshape(-1)
    q_scale = max(float(np.max(np.linalg.norm(raw_matrix, axis=1))), np.finfo(float).tiny)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    operation_records: list[dict[str, Any]] = []
    parsed_operations: list[tuple[str, np.ndarray, np.ndarray, tuple[tuple[str, int, str, int], ...]]] = []
    for operation_index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise QCanonicalizationError(f"operation {operation_index} must be a mapping")
        name = str(operation.get("name", f"operation_{operation_index}"))
        basis_action = operation.get("model_basis_action")
        if not isinstance(basis_action, Mapping) or not bool(basis_action.get("complete", False)):
            raise QCanonicalizationError(
                f"operation {name!r} lacks a complete certified model_basis_action"
            )
        items_raw = basis_action.get("items")
        if not isinstance(items_raw, Sequence) or isinstance(items_raw, (str, bytes)):
            raise QCanonicalizationError(f"operation {name!r} has no certified Q permutation items")
        matrix, translation = _q_map_matrix(_operation_q_map(operation))
        mappings: list[tuple[str, int, str, int]] = []
        targets_by_source: dict[tuple[str, int], tuple[str, int]] = {}
        for item in items_raw:
            if not isinstance(item, Mapping):
                raise QCanonicalizationError(f"operation {name!r} contains a non-mapping Q item")
            source_sector = str(item["source_sector"])
            target_sector = str(item["target_sector"])
            source_index = int(item["source_q_index"])
            target_index = int(item["target_q_index"])
            if source_sector not in raw or target_sector not in raw:
                raise QCanonicalizationError(
                    f"operation {name!r} references an unknown Q sector"
                )
            if not 0 <= source_index < raw[source_sector].shape[0]:
                raise QCanonicalizationError(
                    f"operation {name!r} source Q index {source_index} is out of range"
                )
            if not 0 <= target_index < raw[target_sector].shape[0]:
                raise QCanonicalizationError(
                    f"operation {name!r} target Q index {target_index} is out of range"
                )
            source_key = (source_sector, source_index)
            target_key = (target_sector, target_index)
            previous = targets_by_source.get(source_key)
            if previous is not None and previous != target_key:
                raise QCanonicalizationError(
                    f"operation {name!r} assigns a conflicting target to {source_key}: "
                    f"{previous} and {target_key}"
                )
            targets_by_source[source_key] = target_key
        expected_sources = {
            (sector, q_index)
            for sector in sector_order
            for q_index in range(raw[sector].shape[0])
        }
        if set(targets_by_source) != expected_sources:
            missing = sorted(expected_sources - set(targets_by_source))
            raise QCanonicalizationError(
                f"operation {name!r} Q permutation is incomplete; missing {missing[:8]}"
            )
        for (source_sector, source_index), (target_sector, target_index) in sorted(
            targets_by_source.items()
        ):
            mappings.append((source_sector, source_index, target_sector, target_index))
            source_global = offsets[source_sector] + source_index
            target_global = offsets[target_sector] + target_index
            for component in range(2):
                row = np.zeros(2 * point_count, dtype=np.float64)
                row[2 * target_global + component] = 1.0
                row[2 * source_global : 2 * source_global + 2] -= matrix[component]
                rows.append(row)
                rhs.append(float(translation[component]))
        parsed_operations.append((name, matrix, translation, tuple(mappings)))

    if not rows:
        canonical = _readonly_q_mapping(raw)
        artifact = MappingProxyType(
            {
                "version": CANONICAL_Q_VERSION,
                "status": "not_needed",
                "sector_order": list(sector_order),
                "ordering_changed": False,
                "constraint_rank": 0,
                "constraint_count": 0,
                "raw_closure_max": 0.0,
                "canonical_closure_max": 0.0,
                "rms_correction": 0.0,
                "max_correction": 0.0,
                "raw_q_hash": q_geometry_hash(raw, sector_order),
                "canonical_q_hash": q_geometry_hash(canonical, sector_order),
                "operations": [],
            }
        )
        return CanonicalQResult(raw_q=raw, canonical_q=canonical, artifact=artifact)

    constraint = np.stack(rows, axis=0)
    target = np.asarray(rhs, dtype=np.float64)
    singular_values = np.linalg.svd(constraint, compute_uv=False)
    singular_max = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = float(
        max(constraint.shape) * np.finfo(float).eps * max(singular_max, np.finfo(float).tiny)
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    singular_min_retained = float(singular_values[rank - 1]) if rank else 0.0
    condition_number = (
        float(singular_max / singular_min_retained)
        if singular_min_retained > 0.0
        else 1.0
    )
    residual_target = target - constraint @ raw_vector
    correction, *_ = np.linalg.lstsq(
        constraint,
        residual_target,
        rcond=(rank_tolerance / singular_max if singular_max > 0.0 else None),
    )
    canonical_vector = raw_vector + correction
    constraint_residual = float(np.linalg.norm(constraint @ canonical_vector - target))
    eps = np.finfo(float).eps
    dot_length = int(constraint.shape[1])
    gamma = float(dot_length * eps / max(1.0 - dot_length * eps, np.finfo(float).tiny))
    evaluation_scale = float(
        np.linalg.norm(constraint, ord=2) * np.linalg.norm(canonical_vector)
        + np.linalg.norm(target)
    )
    solver_floor = float(
        condition_number * gamma * evaluation_scale
    )
    if constraint_residual > solver_floor:
        raise QCanonicalizationError(
            "Q symmetry constraints are numerically inconsistent: "
            f"residual={constraint_residual:.6e}, certified_floor={solver_floor:.6e}"
        )
    canonical_matrix = canonical_vector.reshape(point_count, 2)
    correction_matrix = correction.reshape(point_count, 2)
    correction_norms = np.linalg.norm(correction_matrix, axis=1)
    max_correction = float(np.max(correction_norms))
    rms_correction = float(np.sqrt(np.mean(correction_norms**2)))
    absolute_tolerance = (
        float(absolute_cleanup_tolerance)
        if absolute_cleanup_tolerance is not None
        else float(512.0 * np.finfo(float).eps * max(q_scale, np.finfo(float).tiny))
    )
    cleanup_tolerance = float(absolute_tolerance + relative_cleanup_tolerance * q_scale)
    if max_correction > cleanup_tolerance:
        raise QCanonicalizationError(
            "required Q correction exceeds cleanup tolerance: "
            f"correction={max_correction:.6e}, tolerance={cleanup_tolerance:.6e}"
        )

    canonical_mutable: dict[str, np.ndarray] = {}
    cursor = 0
    for sector in sector_order:
        count = int(raw[sector].shape[0])
        canonical_mutable[sector] = canonical_matrix[cursor : cursor + count]
        cursor += count
    canonical = _readonly_q_mapping(canonical_mutable)

    raw_closure_max = 0.0
    canonical_closure_max = 0.0
    for name, matrix, translation, mappings in parsed_operations:
        raw_max = 0.0
        canonical_max = 0.0
        for source_sector, source_index, target_sector, target_index in mappings:
            raw_residual = raw[target_sector][target_index] - (
                matrix @ raw[source_sector][source_index] + translation
            )
            canonical_residual = canonical[target_sector][target_index] - (
                matrix @ canonical[source_sector][source_index] + translation
            )
            raw_max = max(raw_max, float(np.linalg.norm(raw_residual)))
            canonical_max = max(canonical_max, float(np.linalg.norm(canonical_residual)))
        raw_closure_max = max(raw_closure_max, raw_max)
        canonical_closure_max = max(canonical_closure_max, canonical_max)
        operation_records.append(
            {
                "name": name,
                "raw_closure_max": raw_max,
                "canonical_closure_max": canonical_max,
                "mapping_count": len(mappings),
            }
        )
    if canonical_closure_max > solver_floor:
        raise QCanonicalizationError(
            "canonical Q closure exceeds the constrained-solver floor: "
            f"closure={canonical_closure_max:.6e}, floor={solver_floor:.6e}"
        )

    artifact = MappingProxyType(
        {
            "version": CANONICAL_Q_VERSION,
            "status": "certified",
            "sector_order": list(sector_order),
            "ordering_changed": False,
            "constraint_rank": rank,
            "constraint_count": int(constraint.shape[0]),
            "rank_tolerance": rank_tolerance,
            "constraint_condition_number": condition_number,
            "smallest_retained_singular_value": singular_min_retained,
            "constraint_residual": constraint_residual,
            "solver_backward_error_bound": solver_floor,
            "raw_closure_max": raw_closure_max,
            "canonical_closure_max": canonical_closure_max,
            "rms_correction": rms_correction,
            "max_correction": max_correction,
            "q_scale": q_scale,
            "relative_cleanup_tolerance": float(relative_cleanup_tolerance),
            "absolute_cleanup_tolerance": absolute_tolerance,
            "cleanup_tolerance": cleanup_tolerance,
            "raw_q_hash": q_geometry_hash(raw, sector_order),
            "canonical_q_hash": q_geometry_hash(canonical, sector_order),
            "operations": operation_records,
        }
    )
    return CanonicalQResult(raw_q=raw, canonical_q=canonical, artifact=artifact)
