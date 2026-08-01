"""Gauge-invariant metrics for fixed-window projection selection.

This module contains only inexpensive, side-effect-free selection primitives.
Candidate construction and symmetry certification remain the responsibility of
their dedicated layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class CandidateRejectionReason(str, Enum):
    """Machine-readable reasons why one projection candidate is unusable."""

    INSUFFICIENT_TARGET_BANDS = "insufficient_target_bands"
    INSUFFICIENT_CANDIDATE_BANDS = "insufficient_candidate_bands"
    TARGET_BOUNDARY_DEGENERACY = "target_boundary_degeneracy"
    SUBSPACE_OVERLAP = "subspace_overlap"


class CandidateRejected(RuntimeError):
    """Typed, auditable rejection of one otherwise well-formed candidate."""

    def __init__(
        self,
        reason: CandidateRejectionReason,
        message: str,
        *,
        k_index: Optional[int] = None,
        required_band_count: Optional[int] = None,
        available_band_count: Optional[int] = None,
        boundary_gap_mev: Optional[float] = None,
        measured_value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.reason = CandidateRejectionReason(reason)
        self.k_index = None if k_index is None else int(k_index)
        self.required_band_count = (
            None if required_band_count is None else int(required_band_count)
        )
        self.available_band_count = (
            None if available_band_count is None else int(available_band_count)
        )
        self.boundary_gap_mev = (
            None if boundary_gap_mev is None else float(boundary_gap_mev)
        )
        self.measured_value = (
            None if measured_value is None else float(measured_value)
        )
        self.threshold = None if threshold is None else float(threshold)


@dataclass(frozen=True)
class TargetWindowSpec:
    """One target-band window shared by every projection candidate."""

    edge: str
    band_count: int
    validation_k_indices: Tuple[int, ...]
    energy_reference_ev: float
    degeneracy_tolerance_mev: float

    def __post_init__(self) -> None:
        edge = str(self.edge).strip().lower()
        if edge not in {"valence", "conduction"}:
            raise ValueError("edge must be 'valence' or 'conduction'")
        band_count = int(self.band_count)
        if band_count <= 0:
            raise ValueError("band_count must be positive")
        indices = tuple(int(index) for index in self.validation_k_indices)
        if not indices or any(index < 0 for index in indices):
            raise ValueError(
                "validation_k_indices must contain non-negative indices"
            )
        if len(set(indices)) != len(indices):
            raise ValueError("validation_k_indices must be unique")
        energy_reference = float(self.energy_reference_ev)
        if not np.isfinite(energy_reference):
            raise ValueError("energy_reference_ev must be finite")
        degeneracy_tolerance = float(self.degeneracy_tolerance_mev)
        if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0.0:
            raise ValueError(
                "degeneracy_tolerance_mev must be finite and non-negative"
            )
        object.__setattr__(self, "edge", edge)
        object.__setattr__(self, "band_count", band_count)
        object.__setattr__(self, "validation_k_indices", indices)
        object.__setattr__(self, "energy_reference_ev", energy_reference)
        object.__setattr__(
            self, "degeneracy_tolerance_mev", degeneracy_tolerance
        )


@dataclass(frozen=True)
class FrozenTargetWindow:
    """Resolved target-band identities and energies shared by all candidates."""

    spec: TargetWindowSpec
    target_band_ids: Tuple[Tuple[int, ...], ...]
    target_energies_ev: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        band_ids = tuple(
            tuple(int(band_id) for band_id in row)
            for row in self.target_band_ids
        )
        expected_shape = (len(self.spec.validation_k_indices), self.spec.band_count)
        if len(band_ids) != expected_shape[0] or any(
            len(row) != expected_shape[1] for row in band_ids
        ):
            raise ValueError("target_band_ids shape does not match TargetWindowSpec")
        if any(band_id < 0 for row in band_ids for band_id in row):
            raise ValueError("target_band_ids must be non-negative")
        try:
            energies = tuple(
                tuple(float(energy) for energy in row)
                for row in self.target_energies_ev
            )
        except TypeError as exc:
            raise ValueError("target_energies_ev must be two-dimensional") from exc
        if len(energies) != expected_shape[0] or any(
            len(row) != expected_shape[1] for row in energies
        ):
            raise ValueError(
                "target_energies_ev shape does not match TargetWindowSpec"
            )
        if not all(np.isfinite(energy) for row in energies for energy in row):
            raise ValueError("target_energies_ev must be finite")
        object.__setattr__(self, "target_band_ids", band_ids)
        object.__setattr__(self, "target_energies_ev", energies)


@dataclass(frozen=True)
class FixedWindowBandMetrics:
    """Energy errors for a target window whose size never changes by candidate."""

    band_count: int
    validation_k_indices: Tuple[int, ...]
    rms_error_mev: float
    maximum_abs_error_mev: float
    errors_mev: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        band_count = int(self.band_count)
        if band_count <= 0:
            raise ValueError("band_count must be positive")
        validation_indices = tuple(
            int(index) for index in self.validation_k_indices
        )
        if not validation_indices or any(index < 0 for index in validation_indices):
            raise ValueError(
                "validation_k_indices must contain non-negative indices"
            )
        if len(set(validation_indices)) != len(validation_indices):
            raise ValueError("validation_k_indices must be unique")
        try:
            errors = tuple(
                tuple(float(error) for error in row) for row in self.errors_mev
            )
        except TypeError as exc:
            raise ValueError("errors_mev must be two-dimensional") from exc
        if len(errors) != len(validation_indices) or any(
            len(row) != band_count for row in errors
        ):
            raise ValueError("errors_mev shape does not match the fixed target window")
        if not all(np.isfinite(error) for row in errors for error in row):
            raise ValueError("errors_mev must be finite")
        rms_error = float(self.rms_error_mev)
        maximum_error = float(self.maximum_abs_error_mev)
        if not np.isfinite(rms_error):
            raise ValueError("rms_error_mev must be finite")
        if not np.isfinite(maximum_error):
            raise ValueError("maximum_abs_error_mev must be finite")
        if rms_error < 0.0 or maximum_error < 0.0:
            raise ValueError("band errors must be non-negative")
        error_array = np.asarray(errors, dtype=float)
        expected_rms = float(np.sqrt(np.mean(error_array**2)))
        expected_maximum = float(np.max(np.abs(error_array)))
        if not np.isclose(rms_error, expected_rms, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("rms_error_mev is inconsistent with errors_mev")
        if not np.isclose(
            maximum_error,
            expected_maximum,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "maximum_abs_error_mev is inconsistent with errors_mev"
            )
        object.__setattr__(self, "band_count", band_count)
        object.__setattr__(self, "validation_k_indices", validation_indices)
        object.__setattr__(self, "rms_error_mev", rms_error)
        object.__setattr__(self, "maximum_abs_error_mev", maximum_error)
        object.__setattr__(self, "errors_mev", errors)


@dataclass(frozen=True)
class ProjectionOverlapMetrics:
    """Gauge-invariant projection metrics with gauge-anchor quality separated."""

    minimum_principal_overlap_squared: float
    mean_principal_overlap_squared: float
    target_capture: float
    gauge_anchor_quality: Optional[float]
    validation_k_indices: Tuple[int, ...]
    worst_k_index: int

    def __post_init__(self) -> None:
        values = {
            "minimum_principal_overlap_squared": float(
                self.minimum_principal_overlap_squared
            ),
            "mean_principal_overlap_squared": float(
                self.mean_principal_overlap_squared
            ),
            "target_capture": float(self.target_capture),
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if (
            values["minimum_principal_overlap_squared"]
            > values["mean_principal_overlap_squared"]
        ):
            raise ValueError(
                "minimum principal overlap squared cannot exceed mean overlap"
            )
        anchor = self.gauge_anchor_quality
        if anchor is not None:
            anchor = float(anchor)
            if not np.isfinite(anchor):
                raise ValueError("gauge_anchor_quality must be finite")
            if not 0.0 <= anchor <= 1.0:
                raise ValueError("gauge_anchor_quality must lie in [0, 1]")
        validation_indices = tuple(
            int(index) for index in self.validation_k_indices
        )
        if not validation_indices or any(index < 0 for index in validation_indices):
            raise ValueError(
                "validation_k_indices must contain non-negative indices"
            )
        if len(set(validation_indices)) != len(validation_indices):
            raise ValueError("validation_k_indices must be unique")
        worst_k_index = int(self.worst_k_index)
        if worst_k_index not in validation_indices:
            raise ValueError("worst_k_index must belong to validation_k_indices")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "gauge_anchor_quality", anchor)
        object.__setattr__(self, "validation_k_indices", validation_indices)
        object.__setattr__(self, "worst_k_index", worst_k_index)


def _validated_eigenvalues(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] <= 0:
        raise ValueError(f"{name} must have shape (Nk, Nband) with Nband > 0")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.diff(array, axis=1) < 0.0):
        raise ValueError(f"{name} must be sorted in non-decreasing energy order")
    return array


def _target_edge_window_indices(
    eigenvalues: np.ndarray,
    *,
    k_index: int,
    spec: TargetWindowSpec,
) -> np.ndarray:
    row = eigenvalues[k_index]
    if spec.edge == "valence":
        eligible = np.flatnonzero(row <= spec.energy_reference_ev)
        selected = eligible[-spec.band_count :]
    else:
        eligible = np.flatnonzero(row >= spec.energy_reference_ev)
        selected = eligible[: spec.band_count]
    available = int(eligible.size)
    if available < spec.band_count:
        raise CandidateRejected(
            CandidateRejectionReason.INSUFFICIENT_TARGET_BANDS,
            f"target has {available} {spec.edge} bands at k={k_index}; "
            f"the fixed target window requires {spec.band_count}",
            k_index=k_index,
            required_band_count=spec.band_count,
            available_band_count=available,
        )
    return selected


def _candidate_edge_window_indices(
    eigenvalues: np.ndarray,
    *,
    k_index: int,
    spec: TargetWindowSpec,
) -> np.ndarray:
    band_count = int(eigenvalues.shape[1])
    if band_count < spec.band_count:
        raise CandidateRejected(
            CandidateRejectionReason.INSUFFICIENT_CANDIDATE_BANDS,
            f"candidate has {band_count} total bands at k={k_index}; "
            f"the fixed target window requires {spec.band_count}",
            k_index=k_index,
            required_band_count=spec.band_count,
            available_band_count=band_count,
        )
    if spec.edge == "valence":
        return np.arange(band_count - spec.band_count, band_count)
    return np.arange(spec.band_count)


def _certify_target_boundary(
    row: np.ndarray,
    selected: np.ndarray,
    *,
    k_index: int,
    tolerance_mev: float,
) -> None:
    start = int(selected[0])
    stop = int(selected[-1]) + 1
    boundary_pairs = []
    if start > 0:
        boundary_pairs.append((start - 1, start))
    if stop < row.size:
        boundary_pairs.append((stop - 1, stop))
    for left, right in boundary_pairs:
        gap_mev = float(abs(row[right] - row[left]) * 1000.0)
        if gap_mev <= tolerance_mev:
            raise CandidateRejected(
                CandidateRejectionReason.TARGET_BOUNDARY_DEGENERACY,
                "fixed target window cuts a degenerate multiplet at "
                f"k={k_index}: boundary gap {gap_mev:.12g} meV is not above "
                f"the {tolerance_mev:.12g} meV tolerance",
                k_index=k_index,
                boundary_gap_mev=gap_mev,
            )


def resolve_target_window(
    target_eigenvalues_ev: np.ndarray,
    spec: TargetWindowSpec,
) -> FrozenTargetWindow:
    """Resolve and freeze target-band identities before candidates are scored."""

    target = _validated_eigenvalues("target_eigenvalues_ev", target_eigenvalues_ev)
    maximum_k_index = max(spec.validation_k_indices)
    if maximum_k_index >= target.shape[0]:
        raise ValueError("validation_k_indices exceed the target spectrum")

    target_band_ids = []
    target_energies = np.empty(
        (len(spec.validation_k_indices), spec.band_count), dtype=float
    )
    for output_index, k_index in enumerate(spec.validation_k_indices):
        target_indices = _target_edge_window_indices(
            target,
            k_index=k_index,
            spec=spec,
        )
        _certify_target_boundary(
            target[k_index],
            target_indices,
            k_index=k_index,
            tolerance_mev=spec.degeneracy_tolerance_mev,
        )
        target_band_ids.append(tuple(int(index) for index in target_indices))
        target_energies[output_index] = target[k_index, target_indices]

    return FrozenTargetWindow(
        spec=spec,
        target_band_ids=tuple(target_band_ids),
        target_energies_ev=target_energies,
    )


def evaluate_fixed_target_window(
    target_window: FrozenTargetWindow,
    candidate_eigenvalues_ev: np.ndarray,
) -> FixedWindowBandMetrics:
    """Score one candidate against an already resolved target-band window.

    Absolute frozen target energies are compared directly with the candidate
    edge energies.  No candidate classification or spectrum re-centering uses
    the target's global energy reference.
    """

    if not isinstance(target_window, FrozenTargetWindow):
        raise TypeError("target_window must be a FrozenTargetWindow")
    spec = target_window.spec
    candidate = _validated_eigenvalues(
        "candidate_eigenvalues_ev", candidate_eigenvalues_ev
    )
    maximum_k_index = max(spec.validation_k_indices)
    if maximum_k_index >= candidate.shape[0]:
        raise ValueError("validation_k_indices exceed the candidate spectrum")

    errors = np.empty(
        (len(spec.validation_k_indices), spec.band_count), dtype=float
    )
    for output_index, k_index in enumerate(spec.validation_k_indices):
        candidate_indices = _candidate_edge_window_indices(
            candidate,
            k_index=k_index,
            spec=spec,
        )
        frozen_target = np.asarray(
            target_window.target_energies_ev[output_index],
            dtype=float,
        )
        errors[output_index] = (
            candidate[k_index, candidate_indices] - frozen_target
        ) * 1000.0

    error_values = tuple(
        tuple(float(error) for error in row) for row in errors
    )
    return FixedWindowBandMetrics(
        band_count=spec.band_count,
        validation_k_indices=spec.validation_k_indices,
        rms_error_mev=float(np.sqrt(np.mean(errors**2))),
        maximum_abs_error_mev=float(np.max(np.abs(errors))),
        errors_mev=error_values,
    )


def _validated_basis(name: str, values: np.ndarray) -> np.ndarray:
    basis = np.asarray(values, dtype=np.complex128)
    if basis.ndim != 3 or basis.shape[2] <= 0:
        raise ValueError(f"{name} must have shape (Nk, ambient_dim, Nstate)")
    if not np.all(np.isfinite(basis)):
        raise ValueError(f"{name} must be finite")
    identity = np.eye(basis.shape[2], dtype=np.complex128)
    for k_index in range(basis.shape[0]):
        gram = basis[k_index].conj().T @ basis[k_index]
        if not np.allclose(gram, identity, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(f"{name} must have orthonormal columns at k={k_index}")
    return basis


def projection_overlap_metrics(
    model_basis: np.ndarray,
    target_basis: np.ndarray,
    *,
    validation_k_indices: Tuple[int, ...],
    projection_basis: np.ndarray,
    gauge_anchor_quality: Optional[float] = None,
) -> ProjectionOverlapMetrics:
    """Compare model and target subspaces without conflating anchor quality.

    Principal overlaps use the reconstructed model-band subspace.  Target
    capture instead measures how much of the target lies in the complete
    candidate projection space.
    """

    model = _validated_basis("model_basis", model_basis)
    target = _validated_basis("target_basis", target_basis)
    if model.shape != target.shape:
        raise ValueError("model_basis and target_basis must have matching shape")
    projection = _validated_basis("projection_basis", projection_basis)
    if projection.shape[:2] != target.shape[:2]:
        raise ValueError(
            "projection_basis must match target_basis in Nk and ambient dimension"
        )
    validation_indices = tuple(int(index) for index in validation_k_indices)
    if not validation_indices or any(index < 0 for index in validation_indices):
        raise ValueError(
            "validation_k_indices must contain non-negative indices"
        )
    if len(set(validation_indices)) != len(validation_indices):
        raise ValueError("validation_k_indices must be unique and non-negative")
    if max(validation_indices) >= target.shape[0]:
        raise ValueError("validation_k_indices exceed the basis k-point axis")
    if gauge_anchor_quality is not None:
        gauge_anchor_quality = float(gauge_anchor_quality)
        if not np.isfinite(gauge_anchor_quality) or not (
            0.0 <= gauge_anchor_quality <= 1.0
        ):
            raise ValueError("gauge_anchor_quality must be None or lie in [0, 1]")

    squared_overlaps = []
    captures = []
    per_k_minima = []
    for k_index in validation_indices:
        singular_values = np.linalg.svd(
            target[k_index].conj().T @ model[k_index], compute_uv=False
        )
        squared = np.clip(singular_values, 0.0, 1.0) ** 2
        squared_overlaps.extend(float(value) for value in squared)
        per_k_minima.append(float(np.min(squared)))
        projected_target = projection[k_index].conj().T @ target[k_index]
        captures.append(
            float(
                np.clip(
                    np.linalg.norm(projected_target, ord="fro") ** 2
                    / target.shape[2],
                    0.0,
                    1.0,
                )
            )
        )

    worst_local_index = int(np.argmin(per_k_minima))
    return ProjectionOverlapMetrics(
        minimum_principal_overlap_squared=float(np.min(per_k_minima)),
        mean_principal_overlap_squared=float(np.mean(squared_overlaps)),
        target_capture=float(np.mean(captures)),
        gauge_anchor_quality=gauge_anchor_quality,
        validation_k_indices=validation_indices,
        worst_k_index=validation_indices[worst_local_index],
    )


def certify_minimum_principal_overlap(
    metrics: ProjectionOverlapMetrics,
    *,
    threshold: float,
) -> ProjectionOverlapMetrics:
    """Reject a candidate if any validation k/state misses the overlap floor."""

    validated_metrics = ProjectionOverlapMetrics(
        minimum_principal_overlap_squared=(
            metrics.minimum_principal_overlap_squared
        ),
        mean_principal_overlap_squared=metrics.mean_principal_overlap_squared,
        target_capture=metrics.target_capture,
        gauge_anchor_quality=metrics.gauge_anchor_quality,
        validation_k_indices=metrics.validation_k_indices,
        worst_k_index=metrics.worst_k_index,
    )
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and lie in [0, 1]")
    if validated_metrics.minimum_principal_overlap_squared < threshold:
        raise CandidateRejected(
            CandidateRejectionReason.SUBSPACE_OVERLAP,
            "candidate minimum principal overlap squared "
            f"{validated_metrics.minimum_principal_overlap_squared:.12g} is below "
            f"the required {threshold:.12g}",
            k_index=validated_metrics.worst_k_index,
            measured_value=validated_metrics.minimum_principal_overlap_squared,
            threshold=threshold,
        )
    return validated_metrics


__all__ = [
    "CandidateRejected",
    "CandidateRejectionReason",
    "FixedWindowBandMetrics",
    "FrozenTargetWindow",
    "ProjectionOverlapMetrics",
    "TargetWindowSpec",
    "certify_minimum_principal_overlap",
    "evaluate_fixed_target_window",
    "projection_overlap_metrics",
    "resolve_target_window",
]
