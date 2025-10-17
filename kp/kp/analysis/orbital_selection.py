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
    """Compute orbital weights across Q for given bands (placeholder).

    Returns array of shape (n_orbitals,) with aggregated weights.
    """

    raise NotImplementedError


def rank_orbitals(weights: np.ndarray, *, top_k: int | None = None) -> List[int]:
    """Return indices of orbitals ranked by descending weight (placeholder)."""

    raise NotImplementedError


def select_orbit_set(
    weights: np.ndarray,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Select a set of orbitals based on weights and optional constraints (placeholder)."""

    raise NotImplementedError

