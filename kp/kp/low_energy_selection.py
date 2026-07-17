from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReferencePoint:
    k_index: int
    k_coordinate: tuple[float, ...]
    q_indices: tuple[int, ...]
    q_vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ReferenceBlock:
    key: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    physical_layer: int | None = None
    joint_layers: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReferenceState:
    block_key: str
    band_index: int
    energy: float
    vector: np.ndarray
    physical_layer: int | None
    joint_layers: tuple[int, ...]


@dataclass(frozen=True)
class EnergyCluster:
    states: tuple[ReferenceState, ...]


@dataclass(frozen=True)
class LowEnergySeed:
    states: tuple[ReferenceState, ...]


@dataclass(frozen=True)
class ReferenceAction:
    name: str
    source_block: str
    target_block: str
    matrix: np.ndarray
    antiunitary: bool


@dataclass(frozen=True)
class SymmetryAddition:
    operation: str
    source_block: str
    source_band: int
    target_block: str
    target_band: int
    overlap_weight: float


@dataclass(frozen=True)
class SymmetryClosure:
    states: tuple[ReferenceState, ...]
    additions: tuple[SymmetryAddition, ...]


class SymmetryClosureError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionThresholds:
    band_rms_mev: float
    band_max_mev: float
    subspace_overlap: float
    symmetry_residual: float
    symmetry_leakage: float


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    dimension: int
    band_rms_mev: float
    band_max_mev: float
    subspace_overlap: float
    symmetry_residual: float
    symmetry_leakage: float
    structural_failure: str | None = None


@dataclass(frozen=True)
class ThresholdViolation:
    metric: str
    value: float
    threshold: float
    normalized_violation: float


@dataclass(frozen=True)
class SelectionDecision:
    selected: CandidateMetrics
    status: str
    violations: tuple[ThresholdViolation, ...]
    structural_failures: tuple[tuple[str, str], ...]


class CandidateSelectionError(ValueError):
    pass


def _candidate_violations(
    candidate: CandidateMetrics,
    thresholds: SelectionThresholds,
) -> tuple[ThresholdViolation, ...]:
    violations: list[ThresholdViolation] = []
    upper_bounds = (
        ("band_rms_mev", candidate.band_rms_mev, thresholds.band_rms_mev),
        ("band_max_mev", candidate.band_max_mev, thresholds.band_max_mev),
        ("symmetry_residual", candidate.symmetry_residual, thresholds.symmetry_residual),
        ("symmetry_leakage", candidate.symmetry_leakage, thresholds.symmetry_leakage),
    )
    for metric, value, threshold in upper_bounds:
        if not np.isfinite(value) or value > threshold:
            normalized = np.inf if not np.isfinite(value) else (value - threshold) / threshold
            violations.append(
                ThresholdViolation(
                    metric=metric,
                    value=float(value),
                    threshold=float(threshold),
                    normalized_violation=float(normalized),
                )
            )

    if not np.isfinite(candidate.subspace_overlap) or candidate.subspace_overlap < thresholds.subspace_overlap:
        normalized = (
            np.inf
            if not np.isfinite(candidate.subspace_overlap)
            else (thresholds.subspace_overlap - candidate.subspace_overlap)
            / thresholds.subspace_overlap
        )
        violations.append(
            ThresholdViolation(
                metric="subspace_overlap",
                value=float(candidate.subspace_overlap),
                threshold=float(thresholds.subspace_overlap),
                normalized_violation=float(normalized),
            )
        )
    return tuple(violations)


def select_projection_candidate(
    candidates: Sequence[CandidateMetrics],
    thresholds: SelectionThresholds,
) -> SelectionDecision:
    if not candidates:
        raise CandidateSelectionError("no projection candidates were provided")
    threshold_values = (
        thresholds.band_rms_mev,
        thresholds.band_max_mev,
        thresholds.subspace_overlap,
        thresholds.symmetry_residual,
        thresholds.symmetry_leakage,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in threshold_values):
        raise ValueError("selection thresholds must be finite and strictly positive")

    structural_failures = tuple(
        (candidate.candidate_id, str(candidate.structural_failure))
        for candidate in candidates
        if candidate.structural_failure is not None
    )
    valid = tuple(candidate for candidate in candidates if candidate.structural_failure is None)
    if not valid:
        raise CandidateSelectionError("no structurally valid projection candidates remain")

    violations_by_id = {
        candidate.candidate_id: _candidate_violations(candidate, thresholds)
        for candidate in valid
    }
    passing = tuple(
        candidate for candidate in valid if not violations_by_id[candidate.candidate_id]
    )
    if passing:
        selected = min(
            passing,
            key=lambda candidate: (
                candidate.dimension,
                candidate.band_rms_mev,
                1.0 - candidate.subspace_overlap,
                candidate.symmetry_residual,
                candidate.candidate_id,
            ),
        )
        return SelectionDecision(
            selected=selected,
            status="PASS",
            violations=(),
            structural_failures=structural_failures,
        )

    def fallback_key(candidate: CandidateMetrics) -> tuple[float, float, float, int, str]:
        normalized = tuple(
            violation.normalized_violation
            for violation in violations_by_id[candidate.candidate_id]
        )
        return (
            max(normalized),
            sum(normalized),
            candidate.band_rms_mev,
            candidate.dimension,
            candidate.candidate_id,
        )

    selected = min(valid, key=fallback_key)
    return SelectionDecision(
        selected=selected,
        status="WARN",
        violations=violations_by_id[selected.candidate_id],
        structural_failures=structural_failures,
    )


def resolve_reference_point(
    kpoints: np.ndarray,
    qsets: Sequence[np.ndarray],
    *,
    origin_tolerance: float = 1.0e-10,
) -> ReferencePoint:
    k_array = np.asarray(kpoints, dtype=float)
    if k_array.ndim != 2 or k_array.shape[0] == 0:
        raise ValueError(f"kpoints must be a non-empty rank-2 array, got {k_array.shape}")

    k_norms = np.linalg.norm(k_array, axis=1)
    k_index = int(np.argmin(k_norms))
    if float(k_norms[k_index]) > float(origin_tolerance):
        raise ValueError(
            "TAPW kpoints do not contain the valley expansion origin "
            f"within tolerance {origin_tolerance:g}; nearest norm={k_norms[k_index]:.6g}"
        )

    q_indices: list[int] = []
    q_vectors: list[tuple[float, ...]] = []
    for group_index, qset in enumerate(qsets):
        q_array = np.asarray(qset, dtype=float)
        if q_array.ndim != 2 or q_array.shape[0] == 0:
            raise ValueError(
                f"qsets[{group_index}] must be a non-empty rank-2 array, got {q_array.shape}"
            )
        q_index = int(np.argmin(np.linalg.norm(q_array, axis=1)))
        q_indices.append(q_index)
        q_vectors.append(tuple(float(value) for value in q_array[q_index]))

    return ReferencePoint(
        k_index=k_index,
        k_coordinate=tuple(float(value) for value in k_array[k_index]),
        q_indices=tuple(q_indices),
        q_vectors=tuple(q_vectors),
    )


def build_reference_state_pool(blocks: Sequence[ReferenceBlock]) -> tuple[ReferenceState, ...]:
    states: list[ReferenceState] = []
    for block in blocks:
        eigenvalues = np.asarray(block.eigenvalues, dtype=float)
        eigenvectors = np.asarray(block.eigenvectors, dtype=np.complex128)
        if eigenvalues.ndim != 1:
            raise ValueError(f"reference block {block.key!r} eigenvalues must be rank 1")
        if eigenvectors.ndim != 2 or eigenvectors.shape[1] != eigenvalues.size:
            raise ValueError(
                f"reference block {block.key!r} eigenvectors must have one column per eigenvalue"
            )
        for band_index, energy in enumerate(eigenvalues):
            states.append(
                ReferenceState(
                    block_key=str(block.key),
                    band_index=int(band_index),
                    energy=float(energy),
                    vector=np.asarray(eigenvectors[:, band_index], dtype=np.complex128),
                    physical_layer=None if block.physical_layer is None else int(block.physical_layer),
                    joint_layers=tuple(int(layer) for layer in block.joint_layers),
                )
            )
    return tuple(states)


def cluster_edge_states(
    states: Sequence[ReferenceState],
    *,
    edge: str,
    efermi: float,
    degeneracy_tolerance: float = 1.0e-6,
) -> tuple[EnergyCluster, ...]:
    edge_name = str(edge).lower()
    if edge_name == "valence":
        ordered = sorted(
            (state for state in states if state.energy <= float(efermi)),
            key=lambda state: (-state.energy, state.block_key, state.band_index),
        )
    elif edge_name == "conduction":
        ordered = sorted(
            (state for state in states if state.energy >= float(efermi)),
            key=lambda state: (state.energy, state.block_key, state.band_index),
        )
    else:
        raise ValueError(f"edge must be 'valence' or 'conduction', got {edge!r}")

    clusters: list[EnergyCluster] = []
    current: list[ReferenceState] = []
    cluster_energy: float | None = None
    for state in ordered:
        if cluster_energy is None or abs(state.energy - cluster_energy) <= float(degeneracy_tolerance):
            current.append(state)
            if cluster_energy is None:
                cluster_energy = state.energy
            continue
        clusters.append(EnergyCluster(states=tuple(current)))
        current = [state]
        cluster_energy = state.energy
    if current:
        clusters.append(EnergyCluster(states=tuple(current)))
    return tuple(clusters)


def cumulative_energy_seeds(clusters: Sequence[EnergyCluster]) -> tuple[LowEnergySeed, ...]:
    seeds: list[LowEnergySeed] = []
    accumulated: list[ReferenceState] = []
    for cluster in clusters:
        accumulated.extend(cluster.states)
        seeds.append(LowEnergySeed(states=tuple(accumulated)))
    return tuple(seeds)


def close_seed_under_symmetry(
    seed: LowEnergySeed,
    states: Sequence[ReferenceState],
    actions: Sequence[ReferenceAction],
    *,
    leakage_tolerance: float = 1.0e-8,
) -> SymmetryClosure:
    by_block: dict[str, list[ReferenceState]] = {}
    by_key: dict[tuple[str, int], ReferenceState] = {}
    for state in states:
        by_block.setdefault(state.block_key, []).append(state)
        by_key[(state.block_key, state.band_index)] = state

    selected: dict[tuple[str, int], ReferenceState] = {
        (state.block_key, state.band_index): state for state in seed.states
    }
    additions: list[SymmetryAddition] = []
    tolerance_squared = float(leakage_tolerance) ** 2

    changed = True
    while changed:
        changed = False
        selected_snapshot = tuple(selected.values())
        for action in actions:
            target_states = by_block.get(str(action.target_block), [])
            if not target_states:
                raise SymmetryClosureError(
                    f"symmetry action {action.name!r} has no target block {action.target_block!r}"
                )
            matrix = np.asarray(action.matrix, dtype=np.complex128)
            for source_state in selected_snapshot:
                if source_state.block_key != str(action.source_block):
                    continue
                source_vector = np.asarray(source_state.vector, dtype=np.complex128)
                if matrix.ndim != 2 or matrix.shape[1] != source_vector.size:
                    raise SymmetryClosureError(
                        f"symmetry action {action.name!r} shape {matrix.shape} cannot act on "
                        f"block {source_state.block_key!r} vector dimension {source_vector.size}"
                    )
                image = matrix @ (source_vector.conjugate() if action.antiunitary else source_vector)
                image_norm = float(np.linalg.norm(image))
                if image_norm <= float(leakage_tolerance):
                    raise SymmetryClosureError(
                        f"symmetry action {action.name!r} annihilates source "
                        f"{source_state.block_key}:{source_state.band_index}"
                    )
                image = image / image_norm

                scored: list[tuple[float, ReferenceState]] = []
                for target_state in target_states:
                    target_vector = np.asarray(target_state.vector, dtype=np.complex128)
                    if target_vector.size != image.size:
                        raise SymmetryClosureError(
                            f"symmetry action {action.name!r} target vector dimension "
                            f"{target_vector.size} does not match image dimension {image.size}"
                        )
                    weight = float(abs(np.vdot(target_vector, image)) ** 2)
                    scored.append((weight, target_state))

                total_capture = float(sum(weight for weight, _state in scored))
                if total_capture < 1.0 - tolerance_squared:
                    raise SymmetryClosureError(
                        f"target block {action.target_block!r} cannot represent symmetry image "
                        f"of {source_state.block_key}:{source_state.band_index}; "
                        f"captured weight={total_capture:.6g}"
                    )

                captured = float(
                    sum(
                        weight
                        for weight, target_state in scored
                        if (target_state.block_key, target_state.band_index) in selected
                    )
                )
                if 1.0 - captured <= tolerance_squared:
                    continue

                unselected = sorted(
                    (
                        (weight, target_state)
                        for weight, target_state in scored
                        if (target_state.block_key, target_state.band_index) not in selected
                    ),
                    key=lambda item: (-item[0], item[1].band_index),
                )
                for weight, target_state in unselected:
                    if weight <= tolerance_squared:
                        continue
                    key = (target_state.block_key, target_state.band_index)
                    selected[key] = by_key[key]
                    additions.append(
                        SymmetryAddition(
                            operation=str(action.name),
                            source_block=source_state.block_key,
                            source_band=int(source_state.band_index),
                            target_block=target_state.block_key,
                            target_band=int(target_state.band_index),
                            overlap_weight=float(weight),
                        )
                    )
                    captured += float(weight)
                    changed = True
                    if 1.0 - captured <= tolerance_squared:
                        break
                if 1.0 - captured > tolerance_squared:
                    raise SymmetryClosureError(
                        f"target block {action.target_block!r} cannot represent symmetry image "
                        f"of {source_state.block_key}:{source_state.band_index} within leakage tolerance"
                    )

    return SymmetryClosure(states=tuple(selected.values()), additions=tuple(additions))
