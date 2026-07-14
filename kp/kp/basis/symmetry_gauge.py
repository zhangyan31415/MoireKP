"""Deterministic symmetry-adapted gauges inside a projected orbital block.

The stable projector subspace is selected elsewhere (currently by SCDM).  This
module only rotates inside that fixed subspace.  It deliberately uses operation
metadata and spectra rather than operation names, so valleys with different
little groups can use the same code and valleys without a resolving unitary
symmetry safely keep the input gauge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


SYMMETRY_ADAPTED_GAUGE_VERSION = "finite_unitary__antiunitary_pairing_v1"


@dataclass(frozen=True)
class SymmetryGaugeOperation:
    name: str
    matrix: np.ndarray
    antiunitary: bool
    power: int | None = None
    can_resolve: bool = True
    can_pair: bool = True
    can_anchor: bool = True


@dataclass(frozen=True)
class SymmetryAdaptedInternalFrame:
    status: str
    unitary: np.ndarray
    primary_operation: str | None
    eigenvalues: tuple[complex, ...]
    pairing_operation: str | None = None
    reason: str | None = None
    version: str = SYMMETRY_ADAPTED_GAUGE_VERSION

    def artifact(self) -> dict[str, Any]:
        matrix = np.asarray(self.unitary, dtype=np.complex128)
        return {
            "version": self.version,
            "status": self.status,
            "primary_operation": self.primary_operation,
            "pairing_operation": self.pairing_operation,
            "reason": self.reason,
            "dimension": int(matrix.shape[0]),
            "unitary_real": matrix.real.tolist(),
            "unitary_imag": matrix.imag.tolist(),
            "eigenvalues": [[float(value.real), float(value.imag)] for value in self.eigenvalues],
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "SymmetryAdaptedInternalFrame":
        real = np.asarray(artifact["unitary_real"], dtype=np.float64)
        imag = np.asarray(artifact["unitary_imag"], dtype=np.float64)
        return cls(
            status=str(artifact["status"]),
            unitary=np.asarray(real + 1.0j * imag, dtype=np.complex128),
            primary_operation=(
                None if artifact.get("primary_operation") is None else str(artifact["primary_operation"])
            ),
            pairing_operation=(
                None if artifact.get("pairing_operation") is None else str(artifact["pairing_operation"])
            ),
            reason=None if artifact.get("reason") is None else str(artifact["reason"]),
            eigenvalues=tuple(complex(*value) for value in artifact.get("eigenvalues", [])),
            version=str(artifact.get("version", SYMMETRY_ADAPTED_GAUGE_VERSION)),
        )


@dataclass(frozen=True)
class SymmetryAdaptedBasisFrame:
    status: str
    full_unitary: np.ndarray
    sector_frames: tuple[tuple[str, int, int, SymmetryAdaptedInternalFrame], ...]
    reason: str | None = None
    version: str = SYMMETRY_ADAPTED_GAUGE_VERSION

    def artifact(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "status": self.status,
            "reason": self.reason,
            "basis_ordering": "sector__orbital__q_v1",
            "dimension": int(self.full_unitary.shape[0]),
            "sectors": [
                {
                    "name": name,
                    "n_orb": int(n_orb),
                    "n_q": int(n_q),
                    "internal_frame": frame.artifact(),
                }
                for name, n_orb, n_q, frame in self.sector_frames
            ],
        }
        payload["frame_hash"] = symmetry_adapted_frame_hash(payload)
        return payload

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "SymmetryAdaptedBasisFrame":
        stored_hash = artifact.get("frame_hash")
        if stored_hash not in (None, ""):
            computed_hash = symmetry_adapted_frame_hash(artifact)
            if str(stored_hash) != computed_hash:
                raise ValueError(
                    "symmetry-adapted frame artifact hash mismatch: "
                    f"{stored_hash!r} != {computed_hash!r}"
                )
        sectors_raw = artifact.get("sectors", [])
        if not isinstance(sectors_raw, Sequence) or isinstance(sectors_raw, (str, bytes)):
            raise ValueError("symmetry-adapted frame artifact sectors must be a sequence")
        sectors: list[tuple[str, int, int, SymmetryAdaptedInternalFrame]] = []
        for row in sectors_raw:
            if not isinstance(row, Mapping) or not isinstance(row.get("internal_frame"), Mapping):
                raise ValueError("invalid symmetry-adapted sector-frame artifact")
            sectors.append(
                (
                    str(row["name"]),
                    int(row["n_orb"]),
                    int(row["n_q"]),
                    SymmetryAdaptedInternalFrame.from_artifact(row["internal_frame"]),
                )
            )
        full = _assemble_full_basis_unitary(sectors)
        return cls(
            status=str(artifact.get("status", "not_applicable")),
            full_unitary=full,
            sector_frames=tuple(sectors),
            reason=None if artifact.get("reason") is None else str(artifact["reason"]),
            version=str(artifact.get("version", SYMMETRY_ADAPTED_GAUGE_VERSION)),
        )


def symmetry_adapted_frame_hash(artifact: Mapping[str, Any]) -> str:
    payload = {str(key): value for key, value in artifact.items() if str(key) != "frame_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def transform_internal_operation(
    matrix: np.ndarray,
    unitary: np.ndarray,
    *,
    antiunitary: bool,
) -> np.ndarray:
    """Transform ``D`` for ``D`` or for an antiunitary sewing part ``D K``."""

    operation = np.asarray(matrix, dtype=np.complex128)
    frame = np.asarray(unitary, dtype=np.complex128)
    right = frame.conjugate() if antiunitary else frame
    return np.asarray(frame.conjugate().T @ operation @ right, dtype=np.complex128)


def _phase_fix(vector: np.ndarray, *, tolerance: float) -> np.ndarray:
    out = np.asarray(vector, dtype=np.complex128).copy()
    pivot = int(np.argmax(np.abs(out)))
    amplitude = out[pivot]
    if abs(amplitude) > tolerance:
        out *= np.conjugate(amplitude / abs(amplitude))
    if out[pivot].real < 0.0:
        out *= -1.0
    return out


def _orthonormalize_columns(columns: Sequence[np.ndarray], *, tolerance: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for raw in columns:
        vector = np.asarray(raw, dtype=np.complex128).copy()
        for previous in out:
            vector -= previous * np.vdot(previous, vector)
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        out.append(_phase_fix(vector / norm, tolerance=tolerance))
    return out


def _canonical_subspace_basis(projector: np.ndarray, rank: int, *, tolerance: float) -> list[np.ndarray]:
    candidates = [np.asarray(projector[:, index], dtype=np.complex128) for index in range(projector.shape[1])]
    norm_bucket = max(float(tolerance), 100.0 * np.finfo(np.float64).eps)
    candidates.sort(
        key=lambda vector: (
            -int(np.rint(float(np.linalg.norm(vector)) / norm_bucket)),
            int(np.argmax(np.abs(vector))),
        )
    )
    basis = _orthonormalize_columns(candidates, tolerance=tolerance)
    if len(basis) < int(rank):
        raise ValueError(
            f"could not construct a deterministic basis for a rank-{rank} symmetry eigenspace"
        )
    return basis[: int(rank)]


def _cluster_unitary_eigenspaces(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.complex128))
    normalized = np.asarray(
        [value / abs(value) if abs(value) > tolerance else value for value in values],
        dtype=np.complex128,
    )
    clusters: list[list[int]] = []
    for index, value in enumerate(normalized):
        for cluster in clusters:
            reference = normalized[cluster[0]]
            if abs(value - reference) <= 100.0 * tolerance:
                cluster.append(index)
                break
        else:
            clusters.append([index])
    out: list[dict[str, Any]] = []
    for indices in clusters:
        subspace, _ = np.linalg.qr(vectors[:, indices], mode="reduced")
        projector = np.asarray(subspace @ subspace.conjugate().T, dtype=np.complex128)
        eigenvalue = complex(np.mean(normalized[indices]))
        eigenvalue = eigenvalue / abs(eigenvalue)
        phase = float(np.angle(eigenvalue))
        if abs(phase + np.pi) <= 100.0 * tolerance:
            phase = float(np.pi)
        out.append(
            {
                "eigenvalue": eigenvalue,
                "phase": phase,
                "rank": len(indices),
                "projector": projector,
            }
        )
    return out


def _ordered_cluster_groups(
    clusters: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> list[tuple[str, tuple[int, ...]]]:
    unused = set(range(len(clusters)))
    groups: list[tuple[str, tuple[int, ...]]] = []
    complex_negative = sorted(
        (
            index
            for index in unused
            if float(clusters[index]["phase"]) < -100.0 * tolerance
            and abs(abs(float(clusters[index]["phase"])) - np.pi) > 100.0 * tolerance
        ),
        key=lambda index: abs(float(clusters[index]["phase"])),
    )
    for left in complex_negative:
        if left not in unused:
            continue
        target = np.conjugate(complex(clusters[left]["eigenvalue"]))
        matches = [
            index
            for index in unused
            if index != left and abs(complex(clusters[index]["eigenvalue"]) - target) <= 100.0 * tolerance
        ]
        if matches:
            right = min(matches, key=lambda index: abs(float(clusters[index]["phase"])))
            groups.append(("conjugate", (left, right)))
            unused.remove(left)
            unused.remove(right)
    self_conjugate = sorted(
        unused,
        key=lambda index: (
            0 if abs(float(clusters[index]["phase"])) <= 100.0 * tolerance else 1,
            abs(float(clusters[index]["phase"])),
            float(clusters[index]["phase"]),
        ),
    )
    for index in self_conjugate:
        groups.append(("single", (index,)))
    return groups


def _best_pairing_operation(
    operations: Sequence[SymmetryGaugeOperation],
    *,
    dimension: int,
    tolerance: float,
) -> SymmetryGaugeOperation | None:
    candidates: list[tuple[float, str, SymmetryGaugeOperation]] = []
    identity = np.eye(dimension, dtype=np.complex128)
    for operation in operations:
        matrix = np.asarray(operation.matrix, dtype=np.complex128)
        if not operation.antiunitary or not operation.can_pair or matrix.shape != (dimension, dimension):
            continue
        unitarity = float(np.linalg.norm(matrix.conjugate().T @ matrix - identity))
        if unitarity <= 100.0 * tolerance:
            candidates.append((unitarity, str(operation.name), operation))
    return min(candidates, default=(0.0, "", None), key=lambda item: (item[0], item[1]))[2]


def _frame_from_primary(
    primary: SymmetryGaugeOperation,
    operations: Sequence[SymmetryGaugeOperation],
    *,
    tolerance: float,
) -> tuple[np.ndarray, tuple[complex, ...], str | None]:
    matrix = np.asarray(primary.matrix, dtype=np.complex128)
    clusters = _cluster_unitary_eigenspaces(matrix, tolerance=tolerance)
    groups = _ordered_cluster_groups(clusters, tolerance=tolerance)
    pairing = _best_pairing_operation(operations, dimension=matrix.shape[0], tolerance=tolerance)
    columns: list[np.ndarray] = []
    ordered_values: list[complex] = []
    kramers_pairs: list[tuple[int, int]] = []
    for kind, indices in groups:
        left = clusters[indices[0]]
        left_basis = _canonical_subspace_basis(
            np.asarray(left["projector"]),
            int(left["rank"]),
            tolerance=tolerance,
        )
        if kind == "conjugate" and pairing is not None:
            right = clusters[indices[1]]
            right_projector = np.asarray(right["projector"], dtype=np.complex128)
            paired = [right_projector @ np.asarray(pairing.matrix) @ vector.conjugate() for vector in left_basis]
            right_basis = _orthonormalize_columns(paired, tolerance=tolerance)
            if len(right_basis) == len(left_basis):
                for left_vector, right_vector in zip(left_basis, right_basis):
                    pair_start = len(columns)
                    columns.extend((left_vector, right_vector))
                    kramers_pairs.append((pair_start, pair_start + 1))
                    ordered_values.extend((complex(left["eigenvalue"]), complex(right["eigenvalue"])))
                continue
        if kind == "single" and pairing is not None and int(left["rank"]) % 2 == 0:
            projector = np.asarray(left["projector"], dtype=np.complex128)
            used: list[np.ndarray] = []
            for seed in left_basis:
                candidate = seed.copy()
                for previous in used:
                    candidate -= previous * np.vdot(previous, candidate)
                norm = float(np.linalg.norm(candidate))
                if norm <= tolerance:
                    continue
                candidate = _phase_fix(candidate / norm, tolerance=tolerance)
                partner = projector @ np.asarray(pairing.matrix) @ candidate.conjugate()
                for previous in [*used, candidate]:
                    partner -= previous * np.vdot(previous, partner)
                partner_norm = float(np.linalg.norm(partner))
                if partner_norm <= tolerance:
                    continue
                partner /= partner_norm
                used.extend((candidate, partner))
                if len(used) == int(left["rank"]):
                    break
            if len(used) == int(left["rank"]):
                pair_start = len(columns)
                columns.extend(used)
                kramers_pairs.extend(
                    (pair_start + offset, pair_start + offset + 1)
                    for offset in range(0, len(used), 2)
                )
                ordered_values.extend([complex(left["eigenvalue"])] * len(used))
                continue
        columns.extend(left_basis)
        ordered_values.extend([complex(left["eigenvalue"])] * len(left_basis))
    frame = np.column_stack(columns)
    if frame.shape != matrix.shape:
        raise ValueError(f"symmetry-adapted frame has shape {frame.shape}, expected {matrix.shape}")
    unitarity = float(np.linalg.norm(frame.conjugate().T @ frame - np.eye(frame.shape[1])))
    if unitarity > 1000.0 * tolerance:
        raise ValueError(f"symmetry-adapted frame is not unitary: residual={unitarity:.3e}")
    # A canonical antiunitary pair still permits
    # (v, T v) -> (exp(i phi) v, exp(-i phi) T v).  Fix that residual
    # freedom with the first available secondary finite-order unitary.  This
    # is deliberately spectrum/metadata driven; operation family names have
    # no role other than deterministic tie breaking.
    secondary = sorted(
        (
            operation
            for operation in operations
            if not operation.antiunitary
            and operation.can_anchor
            and operation.name != primary.name
            and operation.power is not None
            and int(operation.power) > 1
            and np.asarray(operation.matrix).shape == matrix.shape
        ),
        key=lambda operation: (-int(operation.power or 0), str(operation.name)),
    )
    for left_index, right_index in kramers_pairs:
        for operation in secondary:
            transformed = transform_internal_operation(
                operation.matrix,
                frame,
                antiunitary=False,
            )
            amplitude = complex(transformed[left_index, right_index])
            if abs(amplitude) <= 100.0 * tolerance:
                continue
            phase = 0.5 * (float(np.angle(amplitude)) + 0.5 * float(np.pi))
            frame[:, left_index] *= np.exp(1.0j * phase)
            frame[:, right_index] *= np.exp(-1.0j * phase)
            break
    return np.asarray(frame, dtype=np.complex128), tuple(ordered_values), None if pairing is None else pairing.name


def derive_symmetry_adapted_internal_frame(
    operations: Sequence[SymmetryGaugeOperation],
    *,
    tolerance: float = 1.0e-10,
) -> SymmetryAdaptedInternalFrame:
    """Choose a deterministic internal basis from the available symmetries.

    A finite-order unitary with a nontrivial spectrum is used as the resolving
    generator.  Complex-conjugate eigenspaces are paired by an available
    antiunitary operation.  If no unitary resolves the basis, the identity is
    returned and no physical meaning is inferred from operation names.
    """

    rows = list(operations)
    if not rows:
        return SymmetryAdaptedInternalFrame(
            status="not_applicable",
            unitary=np.eye(0, dtype=np.complex128),
            primary_operation=None,
            eigenvalues=(),
            reason="no_symmetry_operations",
        )
    dimensions = {
        int(np.asarray(operation.matrix).shape[0])
        for operation in rows
        if np.asarray(operation.matrix).ndim == 2
        and np.asarray(operation.matrix).shape[0] == np.asarray(operation.matrix).shape[1]
    }
    if len(dimensions) != 1:
        raise ValueError(f"symmetry gauge operations must share one square dimension, got {sorted(dimensions)}")
    dimension = dimensions.pop()
    identity = np.eye(dimension, dtype=np.complex128)
    candidates: list[tuple[tuple[int, int, str], SymmetryGaugeOperation]] = []
    for operation in rows:
        matrix = np.asarray(operation.matrix, dtype=np.complex128)
        if (
            operation.antiunitary
            or not operation.can_resolve
            or operation.power is None
            or int(operation.power) <= 1
        ):
            continue
        if matrix.shape != (dimension, dimension):
            continue
        if float(np.linalg.norm(matrix.conjugate().T @ matrix - identity)) > 100.0 * tolerance:
            continue
        clusters = _cluster_unitary_eigenspaces(matrix, tolerance=tolerance)
        if len(clusters) <= 1:
            continue
        score = (-len(clusters), -int(operation.power), str(operation.name))
        candidates.append((score, operation))
    if not candidates:
        return SymmetryAdaptedInternalFrame(
            status="not_applicable",
            unitary=identity,
            primary_operation=None,
            eigenvalues=(),
            reason="no_resolving_finite_unitary",
        )
    _score, primary = min(candidates, key=lambda item: item[0])
    frame, eigenvalues, pairing_name = _frame_from_primary(primary, rows, tolerance=tolerance)
    transformed = transform_internal_operation(primary.matrix, frame, antiunitary=False)
    off_diagonal = transformed - np.diag(np.diag(transformed))
    if float(np.linalg.norm(off_diagonal)) > 1000.0 * tolerance:
        raise ValueError(
            f"derived frame did not diagonalize primary operation {primary.name!r}: "
            f"off_diagonal={np.linalg.norm(off_diagonal):.3e}"
        )
    return SymmetryAdaptedInternalFrame(
        status="applied",
        unitary=frame,
        primary_operation=primary.name,
        eigenvalues=eigenvalues,
        pairing_operation=pairing_name,
    )


def transform_basis_operation(
    matrix: np.ndarray,
    full_unitary: np.ndarray,
    *,
    antiunitary: bool,
) -> np.ndarray:
    """Transform a full model-basis operation in the adapted frame."""

    return transform_internal_operation(matrix, full_unitary, antiunitary=antiunitary)


def transform_basis_hamiltonian(matrix: np.ndarray, full_unitary: np.ndarray) -> np.ndarray:
    """Rotate one Hamiltonian or a stack of Hamiltonians into the frame."""

    hamiltonian = np.asarray(matrix, dtype=np.complex128)
    frame = np.asarray(full_unitary, dtype=np.complex128)
    return np.asarray(frame.conjugate().T @ hamiltonian @ frame, dtype=np.complex128)


def transform_basis_vectors(vectors: np.ndarray, full_unitary: np.ndarray) -> np.ndarray:
    """Rotate model-basis coordinates of column eigenvectors."""

    values = np.asarray(vectors, dtype=np.complex128)
    frame = np.asarray(full_unitary, dtype=np.complex128)
    return np.asarray(frame.conjugate().T @ values, dtype=np.complex128)


def _operation_power(record: Mapping[str, Any]) -> int | None:
    if record.get("power") is not None:
        return int(record["power"])
    relations = record.get("group_relations", [])
    if isinstance(relations, Mapping):
        relations = [relations]
    if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)):
        for relation in relations:
            if isinstance(relation, Mapping) and relation.get("power") is not None:
                return int(relation["power"])
    return None


def _sector_offsets(
    sectors: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, int, int, int]], int]:
    rows: list[tuple[str, int, int, int]] = []
    offset = 0
    for index, sector in enumerate(sectors):
        name = str(sector.get("name", f"sector{index}"))
        n_orb = int(sector["n_orb"])
        n_q = int(sector["n_q"])
        if n_orb <= 0 or n_q <= 0:
            raise ValueError(f"invalid sector dimensions for {name!r}: n_orb={n_orb}, n_q={n_q}")
        rows.append((name, n_orb, n_q, offset))
        offset += n_orb * n_q
    return rows, offset


def _q_block_indices(*, offset: int, n_orb: int, n_q: int, q_index: int) -> list[int]:
    return [int(offset + orbital * n_q + q_index) for orbital in range(n_orb)]


def _mapped_internal_block(
    matrix: np.ndarray,
    *,
    source: tuple[str, int, int, int],
    sectors: Sequence[tuple[str, int, int, int]],
    tolerance: float,
) -> tuple[np.ndarray, str] | None:
    source_name, source_n_orb, source_n_q, source_offset = source
    source_indices = _q_block_indices(
        offset=source_offset,
        n_orb=source_n_orb,
        n_q=source_n_q,
        q_index=0,
    )
    candidates: list[tuple[float, str, np.ndarray]] = []
    for target_name, target_n_orb, target_n_q, target_offset in sectors:
        if target_n_orb != source_n_orb:
            continue
        for target_q in range(target_n_q):
            target_indices = _q_block_indices(
                offset=target_offset,
                n_orb=target_n_orb,
                n_q=target_n_q,
                q_index=target_q,
            )
            block = np.asarray(matrix[np.ix_(target_indices, source_indices)], dtype=np.complex128)
            candidates.append((float(np.linalg.norm(block)), target_name, block))
    if not candidates:
        return None
    norm, target_name, block = max(candidates, key=lambda item: item[0])
    if norm <= 100.0 * tolerance:
        return None
    # Exactified actions are block monomial.  Fail closed if the selected
    # internal action is not unitary rather than inventing a dense gauge.
    identity = np.eye(source_n_orb, dtype=np.complex128)
    if float(np.linalg.norm(block.conjugate().T @ block - identity)) > 1000.0 * tolerance:
        return None
    return block, target_name


def _assemble_full_basis_unitary(
    sectors: Sequence[tuple[str, int, int, SymmetryAdaptedInternalFrame]],
) -> np.ndarray:
    dimension = sum(int(n_orb) * int(n_q) for _name, n_orb, n_q, _frame in sectors)
    full = np.zeros((dimension, dimension), dtype=np.complex128)
    offset = 0
    for _name, n_orb, n_q, frame in sectors:
        block = np.kron(np.asarray(frame.unitary, dtype=np.complex128), np.eye(n_q, dtype=np.complex128))
        width = int(n_orb) * int(n_q)
        full[offset : offset + width, offset : offset + width] = block
        offset += width
    return full


def _kramers_pair_indices(frame: SymmetryAdaptedInternalFrame) -> list[tuple[int, int]]:
    dimension = int(frame.unitary.shape[0])
    if frame.pairing_operation is None or dimension % 2 != 0:
        return []
    return [(index, index + 1) for index in range(0, dimension, 2)]


def _align_cross_sector_pair_phases(
    resolved: list[tuple[str, int, int, SymmetryAdaptedInternalFrame]],
    *,
    matrices: Mapping[str, np.ndarray],
    records: Mapping[str, Mapping[str, Any]],
    sector_rows: Sequence[tuple[str, int, int, int]],
    tolerance: float,
) -> list[tuple[str, int, int, SymmetryAdaptedInternalFrame]]:
    """Fix relative Kramers-pair phases along actual inter-sector actions."""

    if len(resolved) < 2:
        return resolved
    sector_index = {name: index for index, (name, _n_orb, _n_q, _frame) in enumerate(resolved)}
    aligned = {resolved[0][0]}
    pending = set(sector_index).difference(aligned)
    while pending:
        progress = False
        for source_name in sorted(aligned):
            source_position = sector_index[source_name]
            source_row = sector_rows[source_position]
            source_frame = resolved[source_position][3]
            source_pairs = _kramers_pair_indices(source_frame)
            if not source_pairs:
                continue
            for operation_name in sorted(set(matrices).intersection(records)):
                record = records[operation_name]
                if bool(record.get("antiunitary", False)):
                    continue
                mapped = _mapped_internal_block(
                    np.asarray(matrices[operation_name], dtype=np.complex128),
                    source=source_row,
                    sectors=sector_rows,
                    tolerance=tolerance,
                )
                if mapped is None:
                    continue
                internal, target_name = mapped
                if target_name not in pending:
                    continue
                target_position = sector_index[target_name]
                target_frame = resolved[target_position][3]
                target_pairs = _kramers_pair_indices(target_frame)
                if len(target_pairs) != len(source_pairs):
                    continue
                transformed = np.asarray(
                    target_frame.unitary.conjugate().T @ internal @ source_frame.unitary,
                    dtype=np.complex128,
                )
                target_unitary = np.asarray(target_frame.unitary, dtype=np.complex128).copy()
                used_targets: set[int] = set()
                for source_pair in source_pairs:
                    choices: list[tuple[float, int, np.ndarray]] = []
                    for target_pair_index, target_pair in enumerate(target_pairs):
                        if target_pair_index in used_targets:
                            continue
                        subblock = transformed[np.ix_(target_pair, source_pair)]
                        choices.append((float(np.linalg.norm(subblock)), target_pair_index, subblock))
                    if not choices:
                        continue
                    subblock_norm, target_pair_index, subblock = max(choices, key=lambda item: item[0])
                    if subblock_norm <= 100.0 * tolerance:
                        continue
                    target_pair = target_pairs[target_pair_index]
                    row_local, column_local = np.unravel_index(
                        int(np.argmax(np.abs(subblock))),
                        subblock.shape,
                    )
                    amplitude = complex(subblock[row_local, column_local])
                    if abs(amplitude) <= 100.0 * tolerance:
                        continue
                    target_sign = 1.0 if row_local == 0 else -1.0
                    correction = (float(np.angle(amplitude)) + 0.5 * float(np.pi)) / target_sign
                    target_unitary[:, target_pair[0]] *= np.exp(1.0j * correction)
                    target_unitary[:, target_pair[1]] *= np.exp(-1.0j * correction)
                    used_targets.add(target_pair_index)
                resolved[target_position] = (
                    target_name,
                    resolved[target_position][1],
                    resolved[target_position][2],
                    replace(target_frame, unitary=target_unitary),
                )
                aligned.add(target_name)
                pending.remove(target_name)
                progress = True
                break
            if progress:
                break
        if not progress:
            break
    return resolved


def derive_symmetry_adapted_basis_frame(
    matrices: Mapping[str, np.ndarray],
    *,
    operations: Sequence[Mapping[str, Any]],
    sectors: Sequence[Mapping[str, Any]],
    tolerance: float = 1.0e-10,
) -> SymmetryAdaptedBasisFrame:
    """Derive per-sector frames from the actual exactified little-group actions.

    The only structural assumption is the release model ordering
    ``sector -> orbital -> Q``.  Whether an operation preserves a sector is
    discovered from its exact block-monomial action, not from its family name.
    """

    sector_rows, dimension = _sector_offsets(sectors)
    if not sector_rows:
        return SymmetryAdaptedBasisFrame(
            status="not_applicable",
            full_unitary=np.eye(0, dtype=np.complex128),
            sector_frames=(),
            reason="no_sectors",
        )
    records = {str(record.get("name", "")): record for record in operations if record.get("name")}
    for name, matrix in matrices.items():
        if np.asarray(matrix).shape != (dimension, dimension):
            raise ValueError(
                f"symmetry matrix {name!r} has shape {np.asarray(matrix).shape}, expected {(dimension, dimension)}"
            )
    resolved: list[tuple[str, int, int, SymmetryAdaptedInternalFrame]] = []
    for source in sector_rows:
        source_name, n_orb, n_q, _offset = source
        internal_operations: list[SymmetryGaugeOperation] = []
        for name in sorted(set(matrices).intersection(records)):
            mapped = _mapped_internal_block(
                np.asarray(matrices[name], dtype=np.complex128),
                source=source,
                sectors=sector_rows,
                tolerance=tolerance,
            )
            if mapped is None:
                continue
            internal, target_name = mapped
            record = records[name]
            preserves_sector = target_name == source_name
            internal_operations.append(
                SymmetryGaugeOperation(
                    name=name,
                    matrix=internal,
                    antiunitary=bool(record.get("antiunitary", False)),
                    power=_operation_power(record),
                    can_resolve=preserves_sector,
                    can_pair=preserves_sector,
                    can_anchor=preserves_sector,
                )
            )
        if internal_operations:
            internal_frame = derive_symmetry_adapted_internal_frame(
                internal_operations,
                tolerance=tolerance,
            )
        else:
            internal_frame = SymmetryAdaptedInternalFrame(
                status="not_applicable",
                unitary=np.eye(n_orb, dtype=np.complex128),
                primary_operation=None,
                eigenvalues=(),
                reason="no_usable_internal_actions",
            )
        resolved.append((source_name, n_orb, n_q, internal_frame))
    resolved = _align_cross_sector_pair_phases(
        resolved,
        matrices=matrices,
        records=records,
        sector_rows=sector_rows,
        tolerance=tolerance,
    )
    full = _assemble_full_basis_unitary(resolved)
    applied = any(frame.status == "applied" for _name, _n_orb, _n_q, frame in resolved)
    return SymmetryAdaptedBasisFrame(
        status="applied" if applied else "not_applicable",
        full_unitary=full,
        sector_frames=tuple(resolved),
        reason=None if applied else "no_resolving_finite_unitary",
    )
