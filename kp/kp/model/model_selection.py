"""Model-complexity selection primitives for configured KP fitting.

The helpers in this module are intentionally independent of the expensive
model builder.  Candidate generation and numerical fitting live in the
configured pipeline; this module defines how already measured candidates are
compared reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class FamilyOrders:
    """Maximum polynomial order selected for each model family."""

    kinetic: int
    intra: int
    inter: int

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(value < 0 for value in values):
            raise ValueError(f"family orders must be non-negative, got {values}")

    def as_tuple(self) -> tuple[int, int, int]:
        return int(self.kinetic), int(self.intra), int(self.inter)


@dataclass(frozen=True)
class CandidateScore:
    """Validation summary for one fitted candidate vocabulary."""

    name: str
    orders: FamilyOrders
    independent_real_parameters: int
    weighted_rms_mev: float
    weighted_rms_se_mev: float
    weighted_max_mev: float
    mean_subspace_overlap: float
    certified: bool = True
    guards_passed: bool = True

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("candidate name must be non-empty")
        if int(self.independent_real_parameters) < 0:
            raise ValueError(
                "independent_real_parameters must be non-negative, got "
                f"{self.independent_real_parameters}"
            )


@dataclass(frozen=True)
class SelectionDecision:
    """Auditable result of one-standard-error candidate selection."""

    status: str
    selected: CandidateScore | None
    best_weighted_rms_mev: float | None
    one_se_threshold_mev: float | None
    plateau_names: tuple[str, ...] = ()
    unmet_targets: tuple[str, ...] = ()
    rejected_reasons: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationFold:
    """One deterministic train/validation split of an ordered k path."""

    index: int
    train_indices: np.ndarray = field(repr=False, compare=False)
    validation_indices: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("train_indices", "validation_indices"):
            value = np.asarray(getattr(self, name), dtype=np.int64).copy()
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class WeightedBandMetrics:
    weighted_rms_mev: float
    maximum_abs_error_mev: float


@dataclass(frozen=True)
class SubspaceOverlapMetrics:
    mean_overlap: float
    minimum_overlap: float
    maximum_leakage: float
    minimum_singular_value: float


def _candidate_nonfinite(candidate: CandidateScore) -> bool:
    values = (
        candidate.weighted_rms_mev,
        candidate.weighted_rms_se_mev,
        candidate.weighted_max_mev,
        candidate.mean_subspace_overlap,
    )
    return not bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


def blocked_kpath_folds(
    kpoints: Sequence[Sequence[float]],
    *,
    n_folds: int,
    duplicate_tolerance: float = 1.0e-12,
) -> tuple[ValidationFold, ...]:
    """Split an ordered k path into contiguous held-out blocks.

    Consecutive duplicate coordinates are treated as one atomic group.  This
    prevents a repeated high-symmetry junction from appearing in both train
    and validation data.
    """

    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("kpoints must have non-empty shape (Nk, dimension)")
    tolerance = float(duplicate_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("duplicate_tolerance must be finite and non-negative")

    atomic_groups: list[np.ndarray] = []
    start = 0
    for index in range(1, len(points)):
        if np.allclose(points[index], points[index - 1], atol=tolerance, rtol=0.0):
            continue
        atomic_groups.append(np.arange(start, index, dtype=np.int64))
        start = index
    atomic_groups.append(np.arange(start, len(points), dtype=np.int64))

    count = int(n_folds)
    if count < 2 or count > len(atomic_groups):
        raise ValueError(
            "n_folds must be between 2 and the number of non-duplicate k blocks "
            f"({len(atomic_groups)}), got {count}"
        )
    all_indices = np.arange(len(points), dtype=np.int64)
    folds: list[ValidationFold] = []
    for fold_index, group_indices in enumerate(
        np.array_split(np.arange(len(atomic_groups), dtype=np.int64), count)
    ):
        validation = np.concatenate(
            [atomic_groups[int(group_index)] for group_index in group_indices]
        )
        train = np.setdiff1d(all_indices, validation, assume_unique=True)
        folds.append(
            ValidationFold(
                index=int(fold_index),
                train_indices=train,
                validation_indices=validation,
            )
        )
    return tuple(folds)


def build_fixed_band_weights(
    target_eigenvalues: np.ndarray,
    *,
    band_edge: str,
    n_bands: int,
    floor: float = 0.05,
    degeneracy_tolerance_mev: float = 0.10,
) -> np.ndarray:
    """Build target-only edge weights with degeneracy-safe boundaries."""

    values = np.asarray(target_eigenvalues, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("target_eigenvalues must have non-empty shape (Nk, Nband)")
    if not np.all(np.isfinite(values)):
        raise ValueError("target_eigenvalues must be finite")
    if np.any(np.diff(values, axis=1) < 0.0):
        raise ValueError("target_eigenvalues must be sorted in ascending order")
    edge = str(band_edge).strip().lower()
    if edge not in {"top", "bottom"}:
        raise ValueError("band_edge must be 'top' or 'bottom'")
    count = int(n_bands)
    if count <= 0 or count > values.shape[1]:
        raise ValueError(f"n_bands must be in 1..{values.shape[1]}, got {count}")
    floor_value = float(floor)
    if not np.isfinite(floor_value) or not 0.0 <= floor_value <= 1.0:
        raise ValueError("floor must be finite and in [0, 1]")
    tolerance_ev = float(degeneracy_tolerance_mev) * 1.0e-3
    if not np.isfinite(tolerance_ev) or tolerance_ev < 0.0:
        raise ValueError("degeneracy_tolerance_mev must be finite and non-negative")

    selected = np.zeros(values.shape, dtype=bool)
    for k_index, row in enumerate(values):
        selected_count = count
        if edge == "top":
            while selected_count < len(row):
                boundary = len(row) - selected_count
                if float(row[boundary] - row[boundary - 1]) > tolerance_ev:
                    break
                selected_count += 1
            selected[k_index, len(row) - selected_count :] = True
        else:
            while selected_count < len(row):
                boundary = selected_count
                if float(row[boundary] - row[boundary - 1]) > tolerance_ev:
                    break
                selected_count += 1
            selected[k_index, :selected_count] = True

    weights = np.full(values.shape, floor_value, dtype=float)
    weights[selected] = 1.0
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("band weights have zero total weight")
    weights *= float(weights.size) / total
    weights.setflags(write=False)
    return weights


def weighted_band_error(
    model_eigenvalues: np.ndarray,
    target_eigenvalues: np.ndarray,
    weights: np.ndarray,
) -> WeightedBandMetrics:
    """Evaluate a common fixed-weight band metric in meV."""

    model = np.asarray(model_eigenvalues, dtype=float)
    target = np.asarray(target_eigenvalues, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if model.shape != target.shape or model.shape != weight.shape or model.ndim != 2:
        raise ValueError("model, target, and weights must have matching shape (Nk, Nband)")
    if not np.all(np.isfinite(model)) or not np.all(np.isfinite(target)):
        raise ValueError("band eigenvalues must be finite")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("band weights must be finite and non-negative")
    normalization = float(np.sum(weight))
    if normalization <= 0.0:
        raise ValueError("band weights must have positive total weight")
    difference = model - target
    return WeightedBandMetrics(
        weighted_rms_mev=float(
            np.sqrt(np.sum(weight * difference**2) / normalization) * 1000.0
        ),
        maximum_abs_error_mev=float(np.max(np.abs(difference)) * 1000.0),
    )


def subspace_overlap_metrics(
    model_basis: np.ndarray,
    target_basis: np.ndarray,
) -> SubspaceOverlapMetrics:
    """Compare equal-dimensional subspaces through principal angles."""

    model = np.asarray(model_basis, dtype=np.complex128)
    target = np.asarray(target_basis, dtype=np.complex128)
    if model.shape != target.shape or model.ndim != 3:
        raise ValueError(
            "model_basis and target_basis must have matching shape (Nk, dim, Nstate)"
        )
    if model.shape[-1] <= 0:
        raise ValueError("subspace overlap requires at least one state")
    overlaps: list[float] = []
    minimum_singular_values: list[float] = []
    for k_index in range(model.shape[0]):
        singular_values = np.linalg.svd(
            target[k_index].conj().T @ model[k_index], compute_uv=False
        )
        clipped = np.clip(singular_values, 0.0, 1.0)
        overlaps.append(float(np.mean(clipped**2)))
        minimum_singular_values.append(float(np.min(clipped)))
    overlap = np.asarray(overlaps, dtype=float)
    return SubspaceOverlapMetrics(
        mean_overlap=float(np.mean(overlap)),
        minimum_overlap=float(np.min(overlap)),
        maximum_leakage=float(np.max(1.0 - overlap)),
        minimum_singular_value=float(np.min(minimum_singular_values)),
    )


def _selection_key(candidate: CandidateScore) -> tuple[object, ...]:
    return (
        int(candidate.independent_real_parameters),
        candidate.orders.as_tuple(),
        float(candidate.weighted_max_mev),
        float(candidate.weighted_rms_mev),
        str(candidate.name),
    )


def select_simplest_near_best(
    candidates: Sequence[CandidateScore],
    *,
    overlap_target: float = 0.95,
    overlap_safety_floor: float = 0.90,
) -> SelectionDecision:
    """Select the simplest candidate on the best candidate's 1-SE plateau.

    Candidates that fail numerical certification, global guards, or metric
    finiteness never define the quality plateau.  The overlap target controls a
    normal ``PASS``.  If it is unreachable, candidates above the explicit
    safety floor may still produce an auditable ``WARN_BEST_AVAILABLE``.
    """

    target = float(overlap_target)
    floor = float(overlap_safety_floor)
    if not np.isfinite(target) or not np.isfinite(floor):
        raise ValueError("overlap thresholds must be finite")
    if not 0.0 <= floor <= target <= 1.0:
        raise ValueError(
            "overlap thresholds must satisfy 0 <= safety_floor <= target <= 1"
        )

    valid: list[CandidateScore] = []
    rejected: dict[str, str] = {}
    for candidate in candidates:
        if not bool(candidate.certified):
            rejected[candidate.name] = "not_certified"
        elif not bool(candidate.guards_passed):
            rejected[candidate.name] = "guard_failure"
        elif _candidate_nonfinite(candidate):
            rejected[candidate.name] = "non_finite_metric"
        elif float(candidate.weighted_rms_se_mev) < 0.0:
            rejected[candidate.name] = "negative_standard_error"
        else:
            valid.append(candidate)

    pass_pool = [
        candidate
        for candidate in valid
        if float(candidate.mean_subspace_overlap) >= target
    ]
    if pass_pool:
        pool = pass_pool
        status = "PASS"
        unmet_targets: tuple[str, ...] = ()
    else:
        pool = [
            candidate
            for candidate in valid
            if float(candidate.mean_subspace_overlap) >= floor
        ]
        if not pool:
            return SelectionDecision(
                status="FAIL",
                selected=None,
                best_weighted_rms_mev=None,
                one_se_threshold_mev=None,
                unmet_targets=(f"mean_subspace_overlap>={floor:.2f}",),
                rejected_reasons=rejected,
            )
        status = "WARN_BEST_AVAILABLE"
        unmet_targets = (f"mean_subspace_overlap>={target:.2f}",)

    best = min(
        pool,
        key=lambda candidate: (
            float(candidate.weighted_rms_mev),
            float(candidate.weighted_max_mev),
            _selection_key(candidate),
        ),
    )
    threshold = float(best.weighted_rms_mev) + float(best.weighted_rms_se_mev)
    plateau = [
        candidate
        for candidate in pool
        if float(candidate.weighted_rms_mev) <= threshold
    ]
    selected = min(plateau, key=_selection_key)
    return SelectionDecision(
        status=status,
        selected=selected,
        best_weighted_rms_mev=float(best.weighted_rms_mev),
        one_se_threshold_mev=threshold,
        plateau_names=tuple(candidate.name for candidate in plateau),
        unmet_targets=unmet_targets,
        rejected_reasons=rejected,
    )


__all__ = [
    "CandidateScore",
    "FamilyOrders",
    "SelectionDecision",
    "SubspaceOverlapMetrics",
    "ValidationFold",
    "WeightedBandMetrics",
    "blocked_kpath_folds",
    "build_fixed_band_weights",
    "select_simplest_near_best",
    "subspace_overlap_metrics",
    "weighted_band_error",
]
