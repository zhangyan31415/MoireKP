"""Model-complexity selection primitives for configured KP fitting.

The helpers in this module are intentionally independent of the expensive
model builder.  Candidate generation and numerical fitting live in the
configured pipeline; this module defines how already measured candidates are
compared reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
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


@dataclass(frozen=True)
class ModelSelectionConfig:
    enabled: bool
    order_candidates: Mapping[str, tuple[int, ...]]
    n_folds: int = 5
    overlap_target: float = 0.95
    overlap_safety_floor: float = 0.90

    def __post_init__(self) -> None:
        candidates = {
            str(family): tuple(int(value) for value in values)
            for family, values in self.order_candidates.items()
        }
        object.__setattr__(self, "order_candidates", MappingProxyType(candidates))


@dataclass(frozen=True)
class FamilyVocabulary:
    max_order: Mapping[str, int]
    term_templates: tuple[Mapping[str, object], ...]
    active_families: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_order",
            MappingProxyType({str(key): int(value) for key, value in self.max_order.items()}),
        )
        object.__setattr__(
            self,
            "term_templates",
            tuple(MappingProxyType(dict(template)) for template in self.term_templates),
        )


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


_FAMILY_ORDER = ("kinetic", "intra", "inter")


def _canonical_family(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "kinect": "kinetic",
        "kinetic": "kinetic",
        "diagonal_kp": "kinetic",
        "intra": "intra",
        "intralayer": "intra",
        "moire": "intra",
        "moire_potential": "intra",
        "inter": "inter",
        "interlayer": "inter",
        "tunneling": "inter",
    }
    if text in aliases:
        return aliases[text]
    if text.startswith("moire_intra"):
        return "intra"
    if text.startswith("tunneling_"):
        return "inter"
    raise ValueError(f"unknown model-selection family {value!r}")


def _order_spec_values(
    raw: object,
    *,
    family: str,
    ceiling: int,
) -> tuple[int, ...]:
    if raw is None:
        start = 1 if family == "kinetic" and ceiling > 0 else 0
        values = list(range(start, ceiling + 1))
    elif isinstance(raw, Mapping):
        start = int(raw.get("min", 1 if family == "kinetic" and ceiling > 0 else 0))
        stop = int(raw.get("max", ceiling))
        step = int(raw.get("step", 1))
        if step <= 0:
            raise ValueError(f"{family} order step must be positive")
        values = list(range(start, stop + 1, step))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [int(value) for value in raw]
    else:
        raise ValueError(
            f"{family} order candidates must be a sequence or min/max mapping"
        )
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError(f"{family} order candidate list must not be empty")
    if normalized[0] < 0:
        raise ValueError(f"{family} order candidates must be non-negative")
    if normalized[-1] > int(ceiling):
        raise ValueError(
            f"{family} order candidate {normalized[-1]} exceeds configured ceiling {ceiling}"
        )
    return normalized


def parse_model_selection_config(
    raw: object,
    *,
    maximum_orders: FamilyOrders,
) -> ModelSelectionConfig:
    """Parse the opt-in fit.model_selection mapping without pipeline state."""

    if raw in (None, False):
        payload: Mapping[str, object] = {}
        enabled = False
    elif raw is True:
        payload = {}
        enabled = True
    elif isinstance(raw, Mapping):
        payload = raw
        enabled = bool(raw.get("enabled", True))
    else:
        raise ValueError("fit.model_selection must be a mapping or boolean")

    raw_orders = payload.get("orders", {})
    if raw_orders is None:
        raw_orders = {}
    if not isinstance(raw_orders, Mapping):
        raise ValueError("fit.model_selection.orders must be a mapping")
    canonical_specs: dict[str, object] = {}
    for key, value in raw_orders.items():
        family = _canonical_family(key)
        if family in canonical_specs:
            raise ValueError(f"duplicate order specification for family {family!r}")
        canonical_specs[family] = value
    ceilings = dict(zip(_FAMILY_ORDER, maximum_orders.as_tuple()))
    candidates = {
        family: _order_spec_values(
            canonical_specs.get(family),
            family=family,
            ceiling=int(ceilings[family]),
        )
        for family in _FAMILY_ORDER
    }

    quality = payload.get("quality", {})
    if quality is None:
        quality = {}
    if not isinstance(quality, Mapping):
        raise ValueError("fit.model_selection.quality must be a mapping")
    overlap_target = float(quality.get("overlap_target", 0.95))
    overlap_floor = float(quality.get("overlap_safety_floor", 0.90))
    if not 0.0 <= overlap_floor <= overlap_target <= 1.0:
        raise ValueError(
            "model-selection overlap thresholds must satisfy "
            "0 <= safety_floor <= target <= 1"
        )
    n_folds = int(payload.get("folds", 5))
    if n_folds < 2:
        raise ValueError("fit.model_selection.folds must be at least 2")
    return ModelSelectionConfig(
        enabled=enabled,
        order_candidates=candidates,
        n_folds=n_folds,
        overlap_target=overlap_target,
        overlap_safety_floor=overlap_floor,
    )


def _max_order_family(key: object) -> str | None:
    text = str(key).strip().lower()
    if text == "onsite":
        return None
    try:
        return _canonical_family(text)
    except ValueError:
        return None


def _template_family(template: Mapping[str, object]) -> str | None:
    source = str(template.get("source", "")).strip().lower()
    if source == "onsite":
        return None
    for value in (source, template.get("tag", ""), template.get("name", "")):
        if not str(value).strip():
            continue
        try:
            return _canonical_family(value)
        except ValueError:
            continue
    raise ValueError(
        "model selection cannot classify term template family: "
        f"{dict(template)!r}"
    )


def clone_family_vocabulary(
    *,
    max_order: Mapping[str, int],
    term_templates: Sequence[Mapping[str, object]],
    orders: FamilyOrders,
    active_families: Sequence[str],
) -> FamilyVocabulary:
    """Clone and restrict one staged family vocabulary.

    Onsite templates are retained at every stage.  A family order of zero is a
    valid constant vocabulary; omitting a family from ``active_families`` is
    the distinct operation that removes its templates.
    """

    active_set = {_canonical_family(value) for value in active_families}
    active = tuple(family for family in _FAMILY_ORDER if family in active_set)
    order_by_family = dict(zip(_FAMILY_ORDER, orders.as_tuple()))
    cloned_max = {str(key): int(value) for key, value in max_order.items()}
    for key in list(cloned_max):
        family = _max_order_family(key)
        if family is not None:
            cloned_max[key] = int(order_by_family[family])
    cloned_max.update(
        {
            "Kinect": int(orders.kinetic),
            "intra": int(orders.intra),
            "inter": int(orders.inter),
        }
    )

    cloned_templates: list[dict[str, object]] = []
    for template in term_templates:
        family = _template_family(template)
        if family is not None and family not in active_set:
            continue
        copied = dict(template)
        if family is not None:
            copied["max_order"] = int(order_by_family[family])
        cloned_templates.append(copied)
    return FamilyVocabulary(
        max_order=cloned_max,
        term_templates=tuple(cloned_templates),
        active_families=active,
    )


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
    "FamilyVocabulary",
    "FamilyOrders",
    "ModelSelectionConfig",
    "SelectionDecision",
    "SubspaceOverlapMetrics",
    "ValidationFold",
    "WeightedBandMetrics",
    "blocked_kpath_folds",
    "build_fixed_band_weights",
    "clone_family_vocabulary",
    "parse_model_selection_config",
    "select_simplest_near_best",
    "subspace_overlap_metrics",
    "weighted_band_error",
]
