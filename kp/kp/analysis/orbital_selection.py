from __future__ import annotations

from typing import Dict, List, Tuple, Any, Sequence

import numpy as np


def compute_orbital_weights(
    eigvecs: np.ndarray,
    Q_set: np.ndarray,
    orbital_order: Sequence[Any],
    band_indices: Sequence[int],
    *,
    score: str = "max",
) -> np.ndarray:
    """Compute orbital weights across Q for selected bands.

    Returns array of shape (n_orbitals,) with aggregated weights.
    """

    vecs = np.asarray(eigvecs, dtype=np.complex128)
    if vecs.ndim < 2:
        raise ValueError(f"eigvecs must have at least basis and band axes, got shape {vecs.shape}")
    n_basis = vecs.shape[-2]
    n_bands = vecs.shape[-1]
    selected = [int(index) for index in band_indices]
    if any(index < 0 or index >= n_bands for index in selected):
        raise ValueError(f"band_indices out of range for {n_bands} bands: {selected}")
    n_orbitals = len(orbital_order)
    if n_orbitals <= 0:
        raise ValueError("orbital_order must contain at least one orbital")
    basis_weights = np.take(np.abs(vecs) ** 2, selected, axis=-1)
    reduce_axes = tuple(range(basis_weights.ndim - 2)) + (basis_weights.ndim - 1,)
    score_name = str(score).lower()
    if score_name == "sum":
        per_basis = np.sum(basis_weights, axis=reduce_axes)
    elif score_name == "max":
        per_basis = np.max(basis_weights, axis=reduce_axes)
    else:
        raise ValueError(f"Unsupported orbital weight score {score!r}; use 'max' or 'sum'")
    weights = np.zeros(n_orbitals, dtype=float)
    for basis_index in range(n_basis):
        weights[basis_index % n_orbitals] += float(per_basis[basis_index])
    return weights


def rank_orbitals(weights: np.ndarray, *, top_k: int | None = None) -> List[int]:
    """Return indices of orbitals ranked by descending weight."""

    values = np.asarray(weights, dtype=float).ravel()
    ranked = sorted(range(values.size), key=lambda index: (-values[index], index))
    if top_k is not None:
        ranked = ranked[: max(0, int(top_k))]
    return ranked


def select_orbit_set(
    weights: np.ndarray,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Select a set of orbitals based on weights and optional constraints."""

    metadata = dict(meta or {})
    indices = rank_orbitals(weights, top_k=metadata.get("top_k"))
    values = np.asarray(weights, dtype=float).ravel()
    return {"indices": indices, "weights": [float(values[index]) for index in indices]}
