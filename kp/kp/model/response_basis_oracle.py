from __future__ import annotations

from typing import Sequence

import numpy as np

from .response_basis import FiniteGroup, PolynomialCoordinateBasis, RawPolynomialSeed


def _evaluate_seed_directly(
    seed: RawPolynomialSeed,
    point: np.ndarray,
    coordinate: PolynomialCoordinateBasis,
) -> np.ndarray:
    dimensionless = (np.asarray(point, dtype=np.float64) - np.asarray(coordinate.origin)) / float(
        coordinate.scale
    )
    w = complex(dimensionless[0], dimensionless[1])
    result = np.zeros((seed.dim, seed.dim), dtype=np.complex128)
    for (r, s), coefficient in seed.coefficients.items():
        result += coefficient.toarray() * (w**int(r)) * (np.conjugate(w) ** int(s))
    return result


def dense_direct_response(
    seed: RawPolynomialSeed,
    kpoints: Sequence[Sequence[float]],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: FiniteGroup,
    component: str,
) -> np.ndarray:
    """Independent dense implementation of the Reynolds plus Hermitian formula.

    This module intentionally evaluates raw polynomial matrices point by point.  It
    does not import or reuse sparse polynomial substitution, transformation, or
    coalescing helpers from the production compiler.
    """

    if component not in {"real", "imag"}:
        raise ValueError("component must be 'real' or 'imag'")
    phase = 1.0 + 0.0j if component == "real" else 0.0 + 1.0j
    points = np.asarray(kpoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk,2), got {points.shape}")
    output = []
    for point in points:
        reynolds = np.zeros((seed.dim, seed.dim), dtype=np.complex128)
        for element in group.elements:
            pulled_point = np.asarray(element.k_pullback, dtype=np.float64) @ point
            raw = phase * _evaluate_seed_directly(seed, pulled_point, coordinate)
            if element.antiunitary:
                raw = np.conjugate(raw)
            unitary = np.asarray(element.internal_u, dtype=np.complex128)
            reynolds += unitary @ raw @ unitary.conj().T
        reynolds /= float(len(group.elements))
        output.append(0.5 * (reynolds + reynolds.conj().T))
    return np.asarray(output, dtype=np.complex128)
