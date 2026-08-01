"""Certified Q-route/orbital-block factorization of exactified continuum actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FACTORIZED_RESPONSE_ACTION_V1 = "factorized_response_action_v1"


class FactorizedActionCertificationError(ValueError):
    """An exactified continuum action has no certified factorized form."""


def explicit_linear_action_matrix(action_map: Mapping[str, Any]) -> np.ndarray:
    """Return the finite 2D Cartesian linear map from explicit action metadata."""

    if not isinstance(action_map, Mapping):
        raise FactorizedActionCertificationError("action map must be an explicit mapping")
    if "linear_matrix" in action_map:
        matrix = np.asarray(action_map["linear_matrix"], dtype=np.float64)
    else:
        map_type = str(action_map.get("type", "")).strip().lower()
        if map_type == "identity":
            matrix = np.eye(2, dtype=np.float64)
        elif map_type == "negation":
            matrix = -np.eye(2, dtype=np.float64)
        elif map_type == "rotation":
            angle = np.deg2rad(float(action_map.get("angle_deg", 0.0)))
            matrix = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=np.float64,
            )
        elif map_type == "reflection":
            angle = np.deg2rad(float(action_map.get("axis_deg", 0.0)))
            axis = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
            matrix = 2.0 * np.outer(axis, axis) - np.eye(2, dtype=np.float64)
        else:
            raise FactorizedActionCertificationError(
                f"unsupported explicit action map type {action_map.get('type')!r}"
            )
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise FactorizedActionCertificationError(
            f"action linear matrix must be finite 2x2, got {matrix}"
        )
    return matrix


def _readonly_complex(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.complex128, order="C", copy=True)
    array.setflags(write=False)
    return array


def _validate_permutation(values: Sequence[int], size: int, *, name: str) -> tuple[int, ...]:
    permutation = tuple(int(value) for value in values)
    if len(permutation) != int(size) or sorted(permutation) != list(range(int(size))):
        raise FactorizedActionCertificationError(
            f"{name} must be a permutation of range({int(size)}), got {permutation}"
        )
    return permutation


def _gamma(operation_count: int) -> float:
    product = float(max(1, int(operation_count))) * np.finfo(np.float64).eps
    if product >= 1.0:
        raise FactorizedActionCertificationError("floating-point certification bound overflow")
    return float(product / (1.0 - product))


def _canonical_metadata_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _array_digest_update(digest: Any, value: np.ndarray, *, dtype: str) -> None:
    array = np.asarray(value, dtype=np.dtype(dtype), order="C")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))


@dataclass(frozen=True)
class FactorizedSymmetryAction:
    name: str
    antiunitary: bool
    k_forward: tuple[tuple[float, float], tuple[float, float]]
    q_permutation: tuple[int, ...]
    sector_permutation: tuple[int, ...]
    q_counts: tuple[int, ...]
    n_orb: tuple[int, ...]
    orbital_blocks: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    q_phases: tuple[complex, ...]
    off_route_residual: float
    phase_alignment_residual: float
    q_covariance_residual: float
    unitarity_residual: float
    matrix_certification_bound: float
    phase_certification_bound: float
    q_certification_bound: float
    unitarity_certification_bound: float
    artifact_hash: str
    version: str = FACTORIZED_RESPONSE_ACTION_V1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "orbital_blocks",
            tuple(_readonly_complex(block) for block in self.orbital_blocks),
        )

    def metadata(self, *, array_keys: Sequence[str] | None = None) -> dict[str, Any]:
        keys = tuple(str(key) for key in (array_keys or ()))
        if keys and len(keys) != len(self.orbital_blocks):
            raise ValueError("factorized action array keys must label every orbital block")
        return {
            "version": self.version,
            "status": "certified",
            "name": self.name,
            "antiunitary": bool(self.antiunitary),
            "k_forward": [list(row) for row in self.k_forward],
            "q_permutation": list(self.q_permutation),
            "sector_permutation": list(self.sector_permutation),
            "q_counts": list(self.q_counts),
            "n_orb": list(self.n_orb),
            "orbital_block_array_keys": list(keys),
            "q_phases": [[float(value.real), float(value.imag)] for value in self.q_phases],
            "off_route_residual": float(self.off_route_residual),
            "phase_alignment_residual": float(self.phase_alignment_residual),
            "q_covariance_residual": float(self.q_covariance_residual),
            "unitarity_residual": float(self.unitarity_residual),
            "matrix_certification_bound": float(self.matrix_certification_bound),
            "phase_certification_bound": float(self.phase_certification_bound),
            "q_certification_bound": float(self.q_certification_bound),
            "unitarity_certification_bound": float(self.unitarity_certification_bound),
            "artifact_hash": self.artifact_hash,
        }


def materialize_factorized_matrix(action: FactorizedSymmetryAction) -> np.ndarray:
    """Reconstruct the certified full action in sector/orbital/Q basis order."""

    q_counts = tuple(int(value) for value in action.q_counts)
    n_orb = tuple(int(value) for value in action.n_orb)
    if len(q_counts) != len(n_orb):
        raise FactorizedActionCertificationError(
            "factorized action q_counts/n_orb layout is inconsistent"
        )
    sector_count = len(q_counts)
    q_permutation = _validate_permutation(
        action.q_permutation,
        sum(q_counts),
        name="q_permutation",
    )
    sector_permutation = _validate_permutation(
        action.sector_permutation,
        sector_count,
        name="sector_permutation",
    )
    if len(action.orbital_blocks) != sector_count:
        raise FactorizedActionCertificationError(
            "factorized action must contain one orbital block per source sector"
        )
    if len(action.q_phases) != sum(q_counts):
        raise FactorizedActionCertificationError(
            "factorized action must contain one phase per source Q"
        )
    q_offsets = np.cumsum((0, *q_counts[:-1])).astype(np.int64)
    basis_offsets = np.cumsum(
        (0, *(q_counts[index] * n_orb[index] for index in range(sector_count - 1)))
    ).astype(np.int64)

    def basis_index(sector: int, orbital: int, q_index: int) -> int:
        return int(basis_offsets[sector] + orbital * q_counts[sector] + q_index)

    dim = int(sum(q_counts[index] * n_orb[index] for index in range(sector_count)))
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    source_global_q = 0
    for source_sector, source_q_count in enumerate(q_counts):
        target_sector = int(sector_permutation[source_sector])
        block = np.asarray(action.orbital_blocks[source_sector], dtype=np.complex128)
        expected_shape = (n_orb[target_sector], n_orb[source_sector])
        if block.shape != expected_shape:
            raise FactorizedActionCertificationError(
                f"factorized orbital block {source_sector} has shape {block.shape}, "
                f"expected {expected_shape}"
            )
        for source_q in range(source_q_count):
            target_global_q = int(q_permutation[source_global_q])
            target_q = target_global_q - int(q_offsets[target_sector])
            if not 0 <= target_q < q_counts[target_sector]:
                raise FactorizedActionCertificationError(
                    "factorized q_permutation is inconsistent with sector_permutation"
                )
            rows = [
                basis_index(target_sector, orbital, target_q)
                for orbital in range(n_orb[target_sector])
            ]
            columns = [
                basis_index(source_sector, orbital, source_q)
                for orbital in range(n_orb[source_sector])
            ]
            matrix[np.ix_(rows, columns)] = (
                complex(action.q_phases[source_global_q]) * block
            )
            source_global_q += 1
    return matrix


def _factorized_action_hash(
    *,
    metadata: Mapping[str, Any],
    orbital_blocks: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(FACTORIZED_RESPONSE_ACTION_V1.encode("utf-8"))
    digest.update(_canonical_metadata_json(metadata))
    for block in orbital_blocks:
        _array_digest_update(digest, np.asarray(block), dtype="<c16")
    return digest.hexdigest()


def _action_hash_metadata_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "version",
        "name",
        "antiunitary",
        "k_forward",
        "q_permutation",
        "sector_permutation",
        "q_counts",
        "n_orb",
        "q_phases",
        "off_route_residual",
        "phase_alignment_residual",
        "q_covariance_residual",
        "unitarity_residual",
        "matrix_certification_bound",
        "phase_certification_bound",
        "q_certification_bound",
        "unitarity_certification_bound",
    )
    missing = [field for field in fields if field not in record]
    if missing:
        raise FactorizedActionCertificationError(
            f"factorized action metadata is missing fields: {missing}"
        )
    return {field: record[field] for field in fields}


def _package_hash(metadata_without_hash: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(FACTORIZED_RESPONSE_ACTION_V1.encode("utf-8"))
    digest.update(_canonical_metadata_json(metadata_without_hash))
    return digest.hexdigest()


def pack_factorized_actions(
    actions: Sequence[FactorizedSymmetryAction],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Freeze certified actions into JSON metadata plus canonical NPZ arrays."""

    ordered = sorted(actions, key=lambda action: action.name)
    if not ordered or len({action.name for action in ordered}) != len(ordered):
        raise ValueError("factorized action package requires unique nonempty actions")
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for action_index, action in enumerate(ordered):
        keys: list[str] = []
        for sector, block in enumerate(action.orbital_blocks):
            key = f"__factorized_response_action_{action_index}_sector_{sector}__"
            arrays[key] = np.asarray(block, dtype=np.complex128)
            keys.append(key)
        records.append(action.metadata(array_keys=keys))
    root = {
        "version": FACTORIZED_RESPONSE_ACTION_V1,
        "status": "certified",
        "basis_ordering": "sector__orbital__q_v1",
        "actions": records,
    }
    return {**root, "package_hash": _package_hash(root)}, arrays


def load_factorized_actions(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, FactorizedSymmetryAction]:
    """Load and hash-verify factorized actions from a canonical package."""

    if str(metadata.get("version", "")) != FACTORIZED_RESPONSE_ACTION_V1:
        raise FactorizedActionCertificationError("unsupported factorized action package version")
    if str(metadata.get("status", "")) != "certified":
        raise FactorizedActionCertificationError("factorized action package is not certified")
    records = metadata.get("actions")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise FactorizedActionCertificationError("factorized action package has no action records")
    root = {
        "version": metadata["version"],
        "status": metadata["status"],
        "basis_ordering": metadata.get("basis_ordering"),
        "actions": list(records),
    }
    expected_package_hash = str(metadata.get("package_hash", ""))
    actual_package_hash = _package_hash(root)
    if actual_package_hash != expected_package_hash:
        raise FactorizedActionCertificationError(
            "factorized action package hash mismatch: "
            f"stored={expected_package_hash}, computed={actual_package_hash}"
        )

    restored: dict[str, FactorizedSymmetryAction] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise FactorizedActionCertificationError("factorized action record must be a mapping")
        record = dict(raw_record)
        name = str(record.get("name", ""))
        if not name or name in restored:
            raise FactorizedActionCertificationError(
                f"factorized action name must be unique and nonempty, got {name!r}"
            )
        raw_keys = record.get("orbital_block_array_keys")
        if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, (str, bytes)):
            raise FactorizedActionCertificationError(
                f"factorized action {name!r} lacks orbital block array keys"
            )
        keys = tuple(str(key) for key in raw_keys)
        missing = [key for key in keys if key not in arrays]
        if missing:
            raise FactorizedActionCertificationError(
                f"factorized action {name!r} references missing arrays: {missing}"
            )
        blocks = tuple(np.asarray(arrays[key], dtype=np.complex128) for key in keys)
        expected_action_hash = str(record.get("artifact_hash", ""))
        actual_action_hash = _factorized_action_hash(
            metadata=_action_hash_metadata_from_record(record),
            orbital_blocks=blocks,
        )
        if actual_action_hash != expected_action_hash:
            raise FactorizedActionCertificationError(
                f"factorized action {name!r} hash mismatch: "
                f"stored={expected_action_hash}, computed={actual_action_hash}"
            )
        phases_raw = record["q_phases"]
        phases = tuple(complex(float(value[0]), float(value[1])) for value in phases_raw)
        restored[name] = FactorizedSymmetryAction(
            name=name,
            antiunitary=bool(record["antiunitary"]),
            k_forward=tuple(
                tuple(float(value) for value in row) for row in record["k_forward"]
            ),
            q_permutation=tuple(int(value) for value in record["q_permutation"]),
            sector_permutation=tuple(int(value) for value in record["sector_permutation"]),
            q_counts=tuple(int(value) for value in record["q_counts"]),
            n_orb=tuple(int(value) for value in record["n_orb"]),
            orbital_blocks=blocks,
            q_phases=phases,
            off_route_residual=float(record["off_route_residual"]),
            phase_alignment_residual=float(record["phase_alignment_residual"]),
            q_covariance_residual=float(record["q_covariance_residual"]),
            unitarity_residual=float(record["unitarity_residual"]),
            matrix_certification_bound=float(record["matrix_certification_bound"]),
            phase_certification_bound=float(record["phase_certification_bound"]),
            q_certification_bound=float(record["q_certification_bound"]),
            unitarity_certification_bound=float(record["unitarity_certification_bound"]),
            artifact_hash=expected_action_hash,
        )
    return restored


def load_factorized_actions_from_npz(
    path: str | Path,
) -> dict[str, FactorizedSymmetryAction]:
    """Load the optional certified fast-path artifact from representations.npz."""

    package_path = Path(path)
    with np.load(package_path, allow_pickle=False) as payload:
        if "__metadata_json__" not in payload.files:
            return {}
        metadata = json.loads(str(np.asarray(payload["__metadata_json__"]).item()))
        if not isinstance(metadata, Mapping):
            raise FactorizedActionCertificationError(
                f"factorized action package metadata is not a mapping: {package_path}"
            )
        exactification = metadata.get("kp_symm_exactification", {})
        root = (
            exactification.get("factorized_response_action")
            if isinstance(exactification, Mapping)
            else None
        )
        if not isinstance(root, Mapping) or str(root.get("status", "")) != "certified":
            return {}
        arrays = {
            str(key): np.asarray(payload[str(key)])
            for key in payload.files
            if str(key).startswith("__factorized_response_action_")
        }
    return load_factorized_actions(root, arrays)


def certify_factorized_action(
    *,
    name: str,
    matrix: np.ndarray,
    antiunitary: bool,
    k_forward: np.ndarray,
    q_permutation: Sequence[int],
    sector_permutation: Sequence[int],
    q_vectors: Sequence[np.ndarray],
    q_counts: Sequence[int],
    n_orb: Sequence[int],
    matrix_absolute_error_bound: float,
    q_absolute_error_bound: float,
) -> FactorizedSymmetryAction:
    """Certify ``U`` as Q-route phases times one orbital block per source sector."""

    q_sizes = tuple(int(value) for value in q_counts)
    orbital_sizes = tuple(int(value) for value in n_orb)
    if not q_sizes or len(q_sizes) != len(orbital_sizes):
        raise FactorizedActionCertificationError("q_counts and n_orb must have equal nonzero length")
    if any(value <= 0 for value in q_sizes):
        raise FactorizedActionCertificationError("q_counts entries must be positive")
    if any(value < 0 for value in orbital_sizes) or not any(
        value > 0 for value in orbital_sizes
    ):
        raise FactorizedActionCertificationError(
            "n_orb entries must be nonnegative with at least one active sector"
        )
    sector_count = len(q_sizes)
    total_q = int(sum(q_sizes))
    q_perm = _validate_permutation(q_permutation, total_q, name="q_permutation")
    sector_perm = _validate_permutation(
        sector_permutation, sector_count, name="sector_permutation"
    )
    for source_sector, target_sector in enumerate(sector_perm):
        if (orbital_sizes[source_sector] == 0) != (
            orbital_sizes[target_sector] == 0
        ):
            raise FactorizedActionCertificationError(
                "sector_permutation cannot exchange active and zero-orbital sectors"
            )
    q_arrays = tuple(np.asarray(value, dtype=np.float64) for value in q_vectors)
    if len(q_arrays) != sector_count:
        raise FactorizedActionCertificationError("q_vectors must contain every sector")
    for sector, (array, count) in enumerate(zip(q_arrays, q_sizes)):
        if array.shape != (count, 2) or not np.all(np.isfinite(array)):
            raise FactorizedActionCertificationError(
                f"q_vectors[{sector}] has shape {array.shape}, expected {(count, 2)}"
            )
    forward = np.asarray(k_forward, dtype=np.float64)
    if forward.shape != (2, 2) or not np.all(np.isfinite(forward)):
        raise FactorizedActionCertificationError("k_forward must be a finite 2x2 matrix")
    if abs(float(np.linalg.det(forward))) <= np.finfo(np.float64).eps:
        raise FactorizedActionCertificationError("k_forward must be invertible")
    unitary = np.asarray(matrix, dtype=np.complex128)
    dim = int(sum(q_sizes[index] * orbital_sizes[index] for index in range(sector_count)))
    if unitary.shape != (dim, dim) or not np.all(np.isfinite(unitary)):
        raise FactorizedActionCertificationError(
            f"exactified action matrix has shape {unitary.shape}, expected {(dim, dim)}"
        )
    matrix_input_bound = float(matrix_absolute_error_bound)
    q_input_bound = float(q_absolute_error_bound)
    if matrix_input_bound < 0.0 or q_input_bound < 0.0:
        raise FactorizedActionCertificationError("certification input bounds must be nonnegative")

    matrix_norm = float(np.linalg.norm(unitary, ord="fro"))
    matrix_bound = float(
        matrix_input_bound + _gamma(8 * dim) * max(matrix_norm, np.finfo(float).tiny)
    )
    phase_bound = float(4.0 * matrix_bound)
    unitarity_bound = float(
        2.0 * max(matrix_norm, 1.0) * matrix_bound
        + matrix_bound**2
        + _gamma(2 * dim) * max(matrix_norm**2, 1.0)
    )
    unitarity_residual = float(
        np.linalg.norm(unitary.conj().T @ unitary - np.eye(dim), ord="fro")
    )

    q_offsets = np.cumsum((0, *q_sizes[:-1])).astype(np.int64)
    basis_offsets = np.cumsum(
        (0, *(q_sizes[index] * orbital_sizes[index] for index in range(sector_count - 1)))
    ).astype(np.int64)

    def basis_index(sector: int, orbital: int, q_index: int) -> int:
        return int(basis_offsets[sector] + orbital * q_sizes[sector] + q_index)

    allowed = np.zeros((dim, dim), dtype=bool)
    blocks_by_sector: list[list[tuple[int, np.ndarray]]] = [
        [] for _ in range(sector_count)
    ]
    q_residuals: list[float] = []
    source_global_q = 0
    for source_sector, source_q_count in enumerate(q_sizes):
        target_sector = int(sector_perm[source_sector])
        target_start = int(q_offsets[target_sector])
        target_stop = target_start + q_sizes[target_sector]
        for source_q in range(source_q_count):
            target_global_q = int(q_perm[source_global_q])
            if not target_start <= target_global_q < target_stop:
                raise FactorizedActionCertificationError(
                    "q_permutation is inconsistent with sector_permutation: "
                    f"source_sector={source_sector}, source_q={source_q}, "
                    f"target_global_q={target_global_q}"
                )
            target_q = target_global_q - target_start
            rows = [
                basis_index(target_sector, orbital, target_q)
                for orbital in range(orbital_sizes[target_sector])
            ]
            columns = [
                basis_index(source_sector, orbital, source_q)
                for orbital in range(orbital_sizes[source_sector])
            ]
            allowed[np.ix_(rows, columns)] = True
            blocks_by_sector[source_sector].append(
                (source_global_q, np.array(unitary[np.ix_(rows, columns)], copy=True))
            )
            if orbital_sizes[source_sector] > 0:
                expected_target_q = forward @ q_arrays[source_sector][source_q]
                q_residuals.append(
                    float(
                        np.linalg.norm(
                            q_arrays[target_sector][target_q] - expected_target_q
                        )
                    )
                )
            source_global_q += 1

    off_route_residual = float(np.linalg.norm(unitary[~allowed]))
    if off_route_residual > matrix_bound:
        raise FactorizedActionCertificationError(
            f"exactified action has off-route weight: residual={off_route_residual:.6e}, "
            f"bound={matrix_bound:.6e}"
        )
    if unitarity_residual > unitarity_bound:
        raise FactorizedActionCertificationError(
            f"exactified action is not unitary within its propagated bound: "
            f"residual={unitarity_residual:.6e}, bound={unitarity_bound:.6e}"
        )

    q_scale = max(
        1.0,
        *(float(np.linalg.norm(vector)) for array in q_arrays for vector in array),
    )
    q_bound = float(
        q_input_bound
        + _gamma(8) * max(float(np.linalg.norm(forward, ord=2)), 1.0) * q_scale
    )
    q_covariance_residual = max(q_residuals, default=0.0)
    if q_covariance_residual > q_bound:
        raise FactorizedActionCertificationError(
            f"canonical Q action is inconsistent with k_forward: "
            f"residual={q_covariance_residual:.6e}, bound={q_bound:.6e}"
        )

    orbital_blocks: list[np.ndarray] = []
    q_phases = np.zeros(total_q, dtype=np.complex128)
    phase_alignment_residual = 0.0
    for source_sector, entries in enumerate(blocks_by_sector):
        if not entries:
            raise FactorizedActionCertificationError(
                f"source sector {source_sector} has no Q blocks"
            )
        target_sector = int(sector_perm[source_sector])
        if orbital_sizes[source_sector] == 0:
            orbital_blocks.append(
                np.zeros(
                    (orbital_sizes[target_sector], orbital_sizes[source_sector]),
                    dtype=np.complex128,
                )
            )
            for global_q, _block in entries:
                q_phases[global_q] = 1.0 + 0.0j
            continue
        _, reference = max(entries, key=lambda item: float(np.linalg.norm(item[1])))
        reference_norm = float(np.linalg.norm(reference, ord="fro"))
        if reference_norm <= matrix_bound:
            raise FactorizedActionCertificationError(
                f"source sector {source_sector} orbital block is numerical zero"
            )
        orbital_blocks.append(reference)
        denominator = complex(np.vdot(reference, reference))
        for global_q, block in entries:
            overlap = complex(np.vdot(reference, block)) / denominator
            if abs(overlap) <= np.finfo(np.float64).tiny:
                raise FactorizedActionCertificationError(
                    f"source sector {source_sector} Q blocks are not phase-equivalent"
                )
            phase = overlap / abs(overlap)
            residual = float(np.linalg.norm(block - phase * reference, ord="fro"))
            phase_alignment_residual = max(phase_alignment_residual, residual)
            q_phases[global_q] = phase
    if phase_alignment_residual > phase_bound:
        raise FactorizedActionCertificationError(
            "Q orbital blocks are not phase-equivalent: "
            f"residual={phase_alignment_residual:.6e}, bound={phase_bound:.6e}"
        )

    metadata_without_hash = {
        "version": FACTORIZED_RESPONSE_ACTION_V1,
        "name": str(name),
        "antiunitary": bool(antiunitary),
        "k_forward": forward.tolist(),
        "q_permutation": list(q_perm),
        "sector_permutation": list(sector_perm),
        "q_counts": list(q_sizes),
        "n_orb": list(orbital_sizes),
        "q_phases": [[float(value.real), float(value.imag)] for value in q_phases],
        "off_route_residual": off_route_residual,
        "phase_alignment_residual": phase_alignment_residual,
        "q_covariance_residual": q_covariance_residual,
        "unitarity_residual": unitarity_residual,
        "matrix_certification_bound": matrix_bound,
        "phase_certification_bound": phase_bound,
        "q_certification_bound": q_bound,
        "unitarity_certification_bound": unitarity_bound,
    }
    artifact_hash = _factorized_action_hash(
        metadata=metadata_without_hash,
        orbital_blocks=orbital_blocks,
    )
    return FactorizedSymmetryAction(
        name=str(name),
        antiunitary=bool(antiunitary),
        k_forward=tuple(tuple(float(value) for value in row) for row in forward),
        q_permutation=q_perm,
        sector_permutation=sector_perm,
        q_counts=q_sizes,
        n_orb=orbital_sizes,
        orbital_blocks=tuple(orbital_blocks),
        q_phases=tuple(complex(value) for value in q_phases),
        off_route_residual=off_route_residual,
        phase_alignment_residual=phase_alignment_residual,
        q_covariance_residual=q_covariance_residual,
        unitarity_residual=unitarity_residual,
        matrix_certification_bound=matrix_bound,
        phase_certification_bound=phase_bound,
        q_certification_bound=q_bound,
        unitarity_certification_bound=unitarity_bound,
        artifact_hash=artifact_hash,
    )


__all__ = [
    "FACTORIZED_RESPONSE_ACTION_V1",
    "FactorizedActionCertificationError",
    "FactorizedSymmetryAction",
    "certify_factorized_action",
    "explicit_linear_action_matrix",
    "load_factorized_actions",
    "load_factorized_actions_from_npz",
    "materialize_factorized_matrix",
    "pack_factorized_actions",
]
