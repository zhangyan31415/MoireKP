"""Fixed-window band and anchor validation for non-Gamma projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from .identity import hash_array, hash_mapping
from .projection_selection import (
    CandidateRejected,
    FixedWindowBandMetrics,
    FrozenTargetWindow,
    TargetWindowSpec,
    evaluate_fixed_target_window,
    resolve_target_window,
)


@dataclass(frozen=True)
class NonGammaValidationTarget:
    """One fixed target-energy window at validation k points."""

    window: FrozenTargetWindow


@dataclass(frozen=True)
class NonGammaCandidateValidation:
    band_metrics: FixedWindowBandMetrics
    gauge_anchor_quality: float


def prepare_non_gamma_validation_target(
    *,
    reference_rows: Sequence[np.ndarray],
    validation_k_indices: Sequence[int],
    edge: str,
    band_count: int,
    energy_reference_ev: float,
    degeneracy_tolerance_mev: float,
) -> NonGammaValidationTarget:
    """Freeze the target bands directly from the authoritative band file."""

    indices = tuple(int(index) for index in validation_k_indices)
    try:
        target_values = np.stack(
            [np.asarray(row, dtype=float).ravel() for row in reference_rows],
            axis=0,
        )
    except ValueError as exc:
        raise ValueError(
            "fixed target-window reference rows must have one common band count"
        ) from exc
    window = resolve_target_window(
        target_values,
        TargetWindowSpec(
            edge=edge,
            band_count=int(band_count),
            validation_k_indices=indices,
            energy_reference_ev=float(energy_reference_ev),
            degeneracy_tolerance_mev=float(degeneracy_tolerance_mev),
        ),
    )

    return NonGammaValidationTarget(window=window)


def _compact_target_window(window: FrozenTargetWindow) -> FrozenTargetWindow:
    compact_indices = tuple(range(len(window.spec.validation_k_indices)))
    return FrozenTargetWindow(
        spec=TargetWindowSpec(
            edge=window.spec.edge,
            band_count=window.spec.band_count,
            validation_k_indices=compact_indices,
            energy_reference_ev=window.spec.energy_reference_ev,
            degeneracy_tolerance_mev=window.spec.degeneracy_tolerance_mev,
        ),
        target_band_ids=window.target_band_ids,
        target_energies_ev=window.target_energies_ev,
    )


def evaluate_non_gamma_validation_candidate(
    target: NonGammaValidationTarget,
    *,
    candidate_eigenvalues: Sequence[np.ndarray],
    gauge_anchor_quality: float,
) -> NonGammaCandidateValidation:
    """Score projected bands and carry the independent anchor quality."""

    count = len(target.window.spec.validation_k_indices)
    if len(candidate_eigenvalues) != count:
        raise ValueError("candidate validation rows do not match the fixed target")
    try:
        candidate_values = np.stack(
            [np.asarray(row, dtype=float).ravel() for row in candidate_eigenvalues],
            axis=0,
        )
    except ValueError as exc:
        raise ValueError("candidate spectra must have one fixed dimension") from exc
    compact_window = _compact_target_window(target.window)
    band_metrics = evaluate_fixed_target_window(compact_window, candidate_values)
    anchor_quality = float(gauge_anchor_quality)
    if not np.isfinite(anchor_quality) or not 0.0 <= anchor_quality <= 1.0:
        raise ValueError("gauge_anchor_quality must lie in [0, 1]")
    return NonGammaCandidateValidation(
        band_metrics=band_metrics,
        gauge_anchor_quality=anchor_quality,
    )


def canonical_action_matrix_hash(matrix: Any) -> str:
    """Hash dense or CSR actions without ever densifying sparse matrices."""

    if sparse.issparse(matrix):
        csr = matrix.tocsr(copy=True)
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        return hash_mapping(
            {
                "schema": "kp.canonical-csr-action.v1",
                "shape": [int(value) for value in csr.shape],
                "data_hash": hash_array(
                    np.ascontiguousarray(csr.data, dtype=np.complex128)
                ),
                "indices_hash": hash_array(
                    np.ascontiguousarray(csr.indices, dtype=np.int64)
                ),
                "indptr_hash": hash_array(
                    np.ascontiguousarray(csr.indptr, dtype=np.int64)
                ),
            }
        )
    return hash_array(np.asarray(matrix, dtype=np.complex128))


def candidate_rejection_diagnostic(error: Exception) -> str:
    if isinstance(error, CandidateRejected):
        return f"{error.reason.value}: {error}"
    return f"{type(error).__name__}: {error}"


__all__ = [
    "NonGammaCandidateValidation",
    "NonGammaValidationTarget",
    "candidate_rejection_diagnostic",
    "canonical_action_matrix_hash",
    "evaluate_non_gamma_validation_candidate",
    "prepare_non_gamma_validation_target",
]
