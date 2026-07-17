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
