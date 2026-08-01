from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

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


class CandidateSelectionFailureCode(str, Enum):
    """Stable reasons why automatic candidate selection failed closed."""

    NO_CANDIDATES = "NO_CANDIDATES"
    DUPLICATE_CANDIDATE_ID = "DUPLICATE_CANDIDATE_ID"
    STRUCTURAL_REJECTION = "STRUCTURAL_REJECTION"
    NONFINITE_METRIC = "NONFINITE_METRIC"
    HARD_METRIC_FAILED = "HARD_METRIC_FAILED"


class CandidateSelectionError(ValueError):
    """Typed failure from automatic projection candidate selection."""

    def __init__(
        self,
        failure_code: CandidateSelectionFailureCode,
        message: str,
        *,
        candidate_ids: Sequence[str] = (),
        structural_failures: Sequence[tuple[str, str]] = (),
    ) -> None:
        super().__init__(message)
        self.failure_code = CandidateSelectionFailureCode(failure_code)
        self.candidate_ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
        self.structural_failures = tuple(
            (str(candidate_id), str(reason))
            for candidate_id, reason in structural_failures
        )


@dataclass(frozen=True)
class BasisLabel:
    orbital: str
    physical_layer: int | str
    spin: str


@dataclass(frozen=True)
class StateComposition:
    block_key: str
    band_index: int
    physical_layer: int | None
    joint_layers: tuple[int, ...]
    orbital_weights: tuple[tuple[str, float], ...]
    layer_weights: tuple[tuple[str, float], ...]
    spin_weights: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FrozenBandCandidate:
    candidate_id: str
    nlow_state_list: tuple[tuple[int, ...], ...]
    states: tuple[ReferenceState, ...]

    @property
    def dimension(self) -> int:
        return sum(len(bands) for bands in self.nlow_state_list)


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
        raise CandidateSelectionError(
            CandidateSelectionFailureCode.NO_CANDIDATES,
            "no projection candidates were provided",
        )
    threshold_values = (
        thresholds.band_rms_mev,
        thresholds.band_max_mev,
        thresholds.subspace_overlap,
        thresholds.symmetry_residual,
        thresholds.symmetry_leakage,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in threshold_values):
        raise ValueError("selection thresholds must be finite and strictly positive")

    candidate_ids = tuple(str(candidate.candidate_id) for candidate in candidates)
    duplicate_ids = tuple(
        candidate_id
        for candidate_id in dict.fromkeys(candidate_ids)
        if candidate_ids.count(candidate_id) > 1
    )
    if duplicate_ids:
        raise CandidateSelectionError(
            CandidateSelectionFailureCode.DUPLICATE_CANDIDATE_ID,
            "projection candidate IDs must be unique",
            candidate_ids=duplicate_ids,
        )

    invalid_dimensions = tuple(
        candidate.candidate_id
        for candidate in candidates
        if not isinstance(candidate.dimension, (int, np.integer))
        or isinstance(candidate.dimension, (bool, np.bool_))
        or int(candidate.dimension) <= 0
    )
    if invalid_dimensions:
        raise CandidateSelectionError(
            CandidateSelectionFailureCode.STRUCTURAL_REJECTION,
            "projection candidate dimensions must be positive integers",
            candidate_ids=invalid_dimensions,
        )

    metric_names = (
        "band_rms_mev",
        "band_max_mev",
        "subspace_overlap",
        "symmetry_residual",
        "symmetry_leakage",
    )
    nonfinite_ids = tuple(
        candidate.candidate_id
        for candidate in candidates
        if any(
            not np.isfinite(float(getattr(candidate, metric_name)))
            for metric_name in metric_names
        )
    )
    if nonfinite_ids:
        raise CandidateSelectionError(
            CandidateSelectionFailureCode.NONFINITE_METRIC,
            "projection candidate metrics must be finite",
            candidate_ids=nonfinite_ids,
        )

    structural_failures = tuple(
        (candidate.candidate_id, str(candidate.structural_failure))
        for candidate in candidates
        if candidate.structural_failure is not None
    )
    valid = tuple(candidate for candidate in candidates if candidate.structural_failure is None)
    if not valid:
        raise CandidateSelectionError(
            CandidateSelectionFailureCode.STRUCTURAL_REJECTION,
            "no structurally valid projection candidates remain",
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            structural_failures=structural_failures,
        )

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

    raise CandidateSelectionError(
        CandidateSelectionFailureCode.HARD_METRIC_FAILED,
        "no projection candidate passed every hard metric",
        candidate_ids=tuple(candidate.candidate_id for candidate in valid),
        structural_failures=structural_failures,
    )


def compute_state_compositions(
    states: Sequence[ReferenceState],
    basis_labels: Mapping[str, Sequence[BasisLabel]],
) -> tuple[StateComposition, ...]:
    compositions: list[StateComposition] = []
    for state in states:
        labels = tuple(basis_labels.get(state.block_key, ()))
        vector = np.asarray(state.vector, dtype=np.complex128)
        if len(labels) != vector.size:
            raise ValueError(
                f"reference block {state.block_key!r} has {vector.size} rows but "
                f"{len(labels)} basis labels"
            )
        weights = np.abs(vector) ** 2
        norm = float(np.sum(weights))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                f"reference state {state.block_key}:{state.band_index} has zero or invalid norm"
            )
        weights = weights / norm

        def grouped(attribute: str) -> tuple[tuple[str, float], ...]:
            totals: dict[str, float] = {}
            for label, weight in zip(labels, weights, strict=True):
                key = str(getattr(label, attribute))
                totals[key] = totals.get(key, 0.0) + float(weight)
            return tuple((key, totals[key]) for key in sorted(totals))

        compositions.append(
            StateComposition(
                block_key=state.block_key,
                band_index=state.band_index,
                physical_layer=state.physical_layer,
                joint_layers=state.joint_layers,
                orbital_weights=grouped("orbital"),
                layer_weights=grouped("physical_layer"),
                spin_weights=grouped("spin"),
            )
        )
    return tuple(compositions)


def _gamma_state_layer(
    state: ReferenceState,
    labels: Sequence[BasisLabel],
    *,
    total_layers: int,
) -> int:
    vector = np.asarray(state.vector, dtype=np.complex128)
    if vector.size != len(labels):
        raise ValueError(
            f"reference block {state.block_key!r} has {vector.size} rows but "
            f"{len(labels)} basis labels"
        )
    weights = np.abs(vector) ** 2
    layer_weights = np.zeros(int(total_layers), dtype=float)
    for label, weight in zip(labels, weights, strict=True):
        try:
            layer = int(label.physical_layer)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Gamma basis label physical_layer must be an integer, got {label.physical_layer!r}"
            ) from exc
        if layer < 0 or layer >= int(total_layers):
            raise ValueError(
                f"Gamma basis label layer {layer} is outside 0..{int(total_layers) - 1}"
            )
        layer_weights[layer] += float(weight)
    return int(np.argmax(layer_weights))


def generate_frozen_band_candidates(
    blocks: Sequence[ReferenceBlock],
    *,
    edge: str,
    efermi: float,
    mode: str,
    total_layers: int,
    basis_labels: Mapping[str, Sequence[BasisLabel]] | None = None,
    degeneracy_tolerance: float = 1.0e-6,
    active_layer_window: float = 0.1,
    max_dimension: int = 16,
) -> tuple[FrozenBandCandidate, ...]:
    if int(total_layers) <= 0:
        raise ValueError("total_layers must be positive")
    if int(max_dimension) <= 0:
        raise ValueError("max_dimension must be positive")
    states = build_reference_state_pool(blocks)
    if str(mode).lower() == "gamma":
        if len(blocks) != 1:
            raise ValueError(f"Gamma selection requires one joint reference block, got {len(blocks)}")
        block = blocks[0]
        labels = tuple((basis_labels or {}).get(block.key, ()))
        if not labels:
            raise ValueError("Gamma selection requires physical-layer basis labels")
        clusters = cluster_edge_states(
            states,
            edge=edge,
            efermi=efermi,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        candidates: list[FrozenBandCandidate] = []
        accumulated: list[ReferenceState] = []
        for cluster in clusters:
            accumulated.extend(cluster.states)
            if len(accumulated) > int(max_dimension):
                break
            rows: list[list[int]] = [[] for _ in range(int(total_layers))]
            for state in accumulated:
                layer = _gamma_state_layer(state, labels, total_layers=int(total_layers))
                rows[layer].append(int(state.band_index))
            normalized = tuple(tuple(sorted(row)) for row in rows)
            candidates.append(
                FrozenBandCandidate(
                    candidate_id=f"dim-{len(accumulated):03d}",
                    nlow_state_list=normalized,
                    states=tuple(accumulated),
                )
            )
        return tuple(candidates)

    states_by_layer: dict[int, list[ReferenceState]] = {}
    for state in states:
        if state.physical_layer is None:
            raise ValueError("non-Gamma reference states require a physical_layer")
        states_by_layer.setdefault(int(state.physical_layer), []).append(state)
    if not states_by_layer:
        raise ValueError("no non-Gamma physical-layer reference blocks were provided")

    clusters_by_layer = {
        layer: cluster_edge_states(
            layer_states,
            edge=edge,
            efermi=efermi,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        for layer, layer_states in states_by_layer.items()
    }
    missing_edges = [layer for layer, clusters in clusters_by_layer.items() if not clusters]
    if missing_edges:
        raise ValueError(f"no {edge} edge states found for physical layer(s) {missing_edges}")
    first_energies = {
        layer: clusters[0].states[0].energy for layer, clusters in clusters_by_layer.items()
    }
    if str(edge).lower() == "valence":
        best_energy = max(first_energies.values())
    else:
        best_energy = min(first_energies.values())
    active_layers = tuple(
        layer
        for layer in sorted(first_energies)
        if abs(first_energies[layer] - best_energy) <= float(active_layer_window)
    )
    if not active_layers:
        raise ValueError("no active physical layer lies within the edge activation window")

    candidates = []
    max_depth = max(len(clusters_by_layer[layer]) for layer in active_layers)
    for depth in range(1, max_depth + 1):
        selected: list[ReferenceState] = []
        rows: list[list[int]] = [[] for _ in range(int(total_layers))]
        complete = True
        for layer in active_layers:
            clusters = clusters_by_layer[layer]
            if len(clusters) < depth:
                complete = False
                break
            for cluster in clusters[:depth]:
                for state in cluster.states:
                    selected.append(state)
                    rows[layer].append(int(state.band_index))
        if not complete:
            break
        dimension = sum(len(row) for row in rows)
        if dimension > int(max_dimension):
            break
        candidates.append(
            FrozenBandCandidate(
                candidate_id=f"dim-{dimension:03d}-depth-{depth:03d}",
                nlow_state_list=tuple(tuple(sorted(set(row))) for row in rows),
                states=tuple(selected),
            )
        )
    return tuple(candidates)


def _violation_payload(violation: ThresholdViolation) -> dict[str, Any]:
    return {
        "metric": violation.metric,
        "value": violation.value,
        "threshold": violation.threshold,
        "normalized_violation": violation.normalized_violation,
    }


def build_selection_report(
    *,
    reference: ReferencePoint,
    closure: SymmetryClosure,
    decision: SelectionDecision,
    compositions: Sequence[StateComposition],
    candidates: Sequence[CandidateMetrics],
    thresholds: SelectionThresholds,
) -> dict[str, Any]:
    selected_id = decision.selected.candidate_id
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.candidate_id == selected_id:
            continue
        candidate_violations = (
            ()
            if candidate.structural_failure is not None
            else _candidate_violations(candidate, thresholds)
        )
        rejected.append(
            {
                "candidate_id": candidate.candidate_id,
                "dimension": candidate.dimension,
                "structural_failure": candidate.structural_failure,
                "metrics": {
                    "band_rms_mev": candidate.band_rms_mev,
                    "band_max_mev": candidate.band_max_mev,
                    "subspace_overlap": candidate.subspace_overlap,
                    "symmetry_residual": candidate.symmetry_residual,
                    "symmetry_leakage": candidate.symmetry_leakage,
                },
                "violations": [_violation_payload(item) for item in candidate_violations],
            }
        )

    composition_payload = [
        {
            "block": composition.block_key,
            "band_index": composition.band_index,
            "physical_layer": composition.physical_layer,
            "joint_layers": list(composition.joint_layers),
            "orbital_weights": dict(composition.orbital_weights),
            "layer_weights": dict(composition.layer_weights),
            "spin_weights": dict(composition.spin_weights),
        }
        for composition in compositions
    ]
    return {
        "status": decision.status,
        "selected_candidate": selected_id,
        "selection_scope": "fixed band indices applied unchanged to every Q and every k",
        "reference": {
            "k_index": reference.k_index,
            "k_coordinate": list(reference.k_coordinate),
            "q_indices": list(reference.q_indices),
            "q_vectors": [list(vector) for vector in reference.q_vectors],
        },
        "selected_states": [
            {"block": state.block_key, "band_index": state.band_index}
            for state in closure.states
        ],
        "symmetry_relations": [
            {
                "operation": addition.operation,
                "source": f"{addition.source_block}:{addition.source_band}",
                "target": f"{addition.target_block}:{addition.target_band}",
                "overlap_weight": addition.overlap_weight,
            }
            for addition in closure.additions
        ],
        "compositions": composition_payload,
        "selected_violations": [_violation_payload(item) for item in decision.violations],
        "rejected_candidates": rejected,
        "structural_failures": [
            {"candidate_id": candidate_id, "reason": reason}
            for candidate_id, reason in decision.structural_failures
        ],
    }


def render_selection_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Automatic low-energy selection",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Selection scope: {report['selection_scope']}.",
        "",
        "## Selected states",
        "",
    ]
    for state in report["selected_states"]:
        lines.append(f"- {state['block']}:{state['band_index']}")
    if report["symmetry_relations"]:
        lines.extend(["", "## Symmetry partner relations", ""])
        for relation in report["symmetry_relations"]:
            lines.append(
                f"- {relation['operation']}: {relation['source']} -> {relation['target']} "
                f"(weight {relation['overlap_weight']:.6g})"
            )
    if report["compositions"]:
        lines.extend(["", "## Reference composition", ""])
        for composition in report["compositions"]:
            lines.append(
                f"- {composition['block']}:{composition['band_index']}: "
                f"orbital={composition['orbital_weights']}, "
                f"layer={composition['layer_weights']}, spin={composition['spin_weights']}"
            )
    if report["rejected_candidates"]:
        lines.extend(["", "## Rejected candidates", ""])
        for candidate in report["rejected_candidates"]:
            lines.append(f"- {candidate['candidate_id']} (dimension {candidate['dimension']})")
    return "\n".join(lines) + "\n"


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
