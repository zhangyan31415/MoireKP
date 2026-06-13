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
    """Return max Frobenius norm of D H(k) D^† - H(g·k) over k_points."""

    points = np.asarray(k_points if k_points is not None else np.zeros((1, 2)), dtype=float)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    rep = np.asarray(D, dtype=np.complex128)
    transform = k_transform if k_transform is not None else (lambda k: k)
    max_residual = 0.0
    for point in points:
        h_src = np.asarray(H_of_k(point), dtype=np.complex128)
        h_dst = np.asarray(H_of_k(np.asarray(transform(point), dtype=float)), dtype=np.complex128)
        covariant = rep @ h_src @ rep.conj().T
        residual = float(np.linalg.norm(covariant - h_dst, ord="fro"))
        max_residual = max(max_residual, residual)
    if max_residual <= float(tol):
        return 0.0
    return max_residual
