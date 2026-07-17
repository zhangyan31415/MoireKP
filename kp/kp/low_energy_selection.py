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
