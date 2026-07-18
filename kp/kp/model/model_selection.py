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


def _candidate_nonfinite(candidate: CandidateScore) -> bool:
    values = (
        candidate.weighted_rms_mev,
        candidate.weighted_rms_se_mev,
        candidate.weighted_max_mev,
        candidate.mean_subspace_overlap,
    )
    return not bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


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
    "select_simplest_near_best",
]
