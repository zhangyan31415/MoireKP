from __future__ import annotations

from typing import Callable

import numpy as np


def check_symmetry_consistency(
    H_of_k: Callable[[np.ndarray], np.ndarray],
    D: np.ndarray,
    k_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    k_points: np.ndarray | None = None,
    tol: float = 1e-8,
) -> float:
    """Return max Frobenius norm of D H(k) D^† - H(g·k) over k_points (placeholder)."""

    raise NotImplementedError

