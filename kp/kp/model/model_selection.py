"""Model-complexity selection primitives for configured KP fitting.

The helpers in this module are intentionally independent of the expensive
model builder.  Candidate generation and numerical fitting live in the
configured pipeline; this module defines how already measured candidates are
compared reproducibly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

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
    expanded_weighted_rms_mev: float | None = None
    certified: bool = True
    guards_passed: bool = True
    solver_family: str = "linear"
    active_group_count: int = 0
    harmonic_support_size: int = 0

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("candidate name must be non-empty")
        if int(self.independent_real_parameters) < 0:
            raise ValueError(
                "independent_real_parameters must be non-negative, got "
                f"{self.independent_real_parameters}"
            )
        solver_family = str(self.solver_family).strip().lower()
        if solver_family not in {"linear", "nonlinear"}:
            raise ValueError(
                "solver_family must be 'linear' or 'nonlinear', got "
                f"{self.solver_family!r}"
            )
        if int(self.active_group_count) < 0 or int(self.harmonic_support_size) < 0:
            raise ValueError("candidate complexity counts must be non-negative")
        object.__setattr__(self, "solver_family", solver_family)


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
class HighLowSelectionResult:
    """High-accuracy and compact edge-preserving profile decisions."""

    high: SelectionDecision
    low: SelectionDecision
    low_se_multiplier: float
    low_relative_rms_tolerance: float = 1.0
    low_minimum_tolerance_mev: float = 0.10


@dataclass(frozen=True)
class FourProfileSelectionResult:
    """Linear and nonlinear high/low profile decisions."""

    linear: HighLowSelectionResult
    nonlinear: HighLowSelectionResult


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
class CenteredLinearFit:
    """Least-squares result for a response measured relative to one k point."""

    coefficients: np.ndarray = field(repr=False, compare=False)
    validation_rms: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float).copy()
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be one-dimensional")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)


@dataclass(frozen=True)
class ResponseSubspaceOverlap:
    """Principal-correlation audit between two family response spaces."""

    maximum_correlation: float
    minimum_principal_angle_degrees: float
    left_rank: int
    right_rank: int
    defer_to_joint_stage: bool


@dataclass(frozen=True)
class StageSelectionResult:
    """Candidate table and decision for one family-order stage."""

    stage: str
    active_families: tuple[str, ...]
    candidates: tuple[CandidateScore, ...]
    decision: SelectionDecision


@dataclass(frozen=True)
class StagedSelectionResult:
    """Result of the kinetic -> intra -> inter order scan."""

    status: str
    selected_orders: FamilyOrders | None
    stages: tuple[StageSelectionResult, ...]


@dataclass(frozen=True)
class ParameterGroup:
    """Smallest symmetry/Hermiticity-closed unit allowed to be removed."""

    group_id: str
    parameter_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        group_id = str(self.group_id)
        indices = tuple(int(index) for index in self.parameter_indices)
        if not group_id:
            raise ValueError("parameter group id must be non-empty")
        if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError(
                "parameter group indices must be unique non-negative integers"
            )
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "parameter_indices", indices)


@dataclass(frozen=True)
class GroupAblationStep:
    removed_group_id: str
    remaining_group_ids: tuple[str, ...]
    candidate: CandidateScore
    validation_loss_increase_mev: float
    accepted: bool


@dataclass(frozen=True)
class GroupAblationResult:
    status: str
    selected: CandidateScore
    active_group_ids: tuple[str, ...]
    steps: tuple[GroupAblationStep, ...]


@dataclass(frozen=True)
class CorrectionSweepResult:
    candidates: tuple[CandidateScore, ...]
    decision: SelectionDecision


@dataclass(frozen=True)
class ModelSelectionConfig:
    enabled: bool
    order_candidates: Mapping[str, tuple[int, ...]]
    n_folds: int = 5
    overlap_target: float = 0.95
    overlap_safety_floor: float = 0.90
    low_se_multiplier: float = 2.0
    low_relative_rms_tolerance: float = 1.0
    low_minimum_tolerance_mev: float = 0.10
    pruning_keep_fractions: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85)

    def __post_init__(self) -> None:
        candidates = {
            str(family): tuple(int(value) for value in values)
            for family, values in self.order_candidates.items()
        }
        object.__setattr__(self, "order_candidates", MappingProxyType(candidates))
        fractions = tuple(float(value) for value in self.pruning_keep_fractions)
        if any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in fractions):
            raise ValueError("term pruning keep fractions must be finite and lie in (0, 1)")
        object.__setattr__(self, "pruning_keep_fractions", tuple(sorted(set(fractions))))


@dataclass(frozen=True)
class ProductionCandidateSpec:
    """Deterministic identity of one fully rebuilt production vocabulary."""

    name: str
    orders: FamilyOrders
    active_families: tuple[str, ...]
    vocabulary_hash: str


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
    if candidate.expanded_weighted_rms_mev is not None:
        values = (*values, candidate.expanded_weighted_rms_mev)
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

    if raw is None:
        payload: Mapping[str, object] = {}
        enabled = True
    elif raw is False:
        payload = {}
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
    profiles = payload.get("profiles", {})
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, Mapping):
        raise ValueError("fit.model_selection.profiles must be a mapping")
    low_profile = profiles.get("low", {})
    if low_profile is None:
        low_profile = {}
    if not isinstance(low_profile, Mapping):
        raise ValueError("fit.model_selection.profiles.low must be a mapping")
    low_se_multiplier = float(low_profile.get("standard_error_multiplier", 2.0))
    if not np.isfinite(low_se_multiplier) or low_se_multiplier < 0.0:
        raise ValueError(
            "low profile standard_error_multiplier must be finite and non-negative"
        )
    low_relative_rms_tolerance = float(
        low_profile.get("relative_rms_tolerance", 1.0)
    )
    low_minimum_tolerance_mev = float(
        low_profile.get("minimum_tolerance_mev", 0.10)
    )
    if (
        not np.isfinite(low_relative_rms_tolerance)
        or low_relative_rms_tolerance < 0.0
        or not np.isfinite(low_minimum_tolerance_mev)
        or low_minimum_tolerance_mev < 0.0
    ):
        raise ValueError("low profile adaptive RMS tolerances must be finite and non-negative")
    keep_fractions_raw = low_profile.get(
        "term_keep_fractions",
        (0.25, 0.40, 0.55, 0.70, 0.85),
    )
    if not isinstance(keep_fractions_raw, Sequence) or isinstance(
        keep_fractions_raw,
        (str, bytes),
    ):
        raise ValueError("low profile term_keep_fractions must be a sequence")
    return ModelSelectionConfig(
        enabled=enabled,
        order_candidates=candidates,
        n_folds=n_folds,
        overlap_target=overlap_target,
        overlap_safety_floor=overlap_floor,
        low_se_multiplier=low_se_multiplier,
        low_relative_rms_tolerance=low_relative_rms_tolerance,
        low_minimum_tolerance_mev=low_minimum_tolerance_mev,
        pruning_keep_fractions=tuple(float(value) for value in keep_fractions_raw),
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


def build_production_candidate_spec(
    vocabulary: FamilyVocabulary,
    orders: FamilyOrders,
) -> ProductionCandidateSpec:
    """Create a stable candidate name and exact-vocabulary content hash."""

    active = tuple(str(family) for family in vocabulary.active_families)
    payload = {
        "orders": list(orders.as_tuple()),
        "active_families": list(active),
        "max_order": dict(vocabulary.max_order),
        "term_templates": [dict(template) for template in vocabulary.term_templates],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    name = (
        f"k{orders.kinetic}_i{orders.intra}_t{orders.inter}__"
        + "-".join(active)
    )
    return ProductionCandidateSpec(
        name=name,
        orders=orders,
        active_families=active,
        vocabulary_hash=hashlib.sha256(encoded).hexdigest(),
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
    weighted_support = weight > 0.0
    return WeightedBandMetrics(
        weighted_rms_mev=float(
            np.sqrt(np.sum(weight * difference**2) / normalization) * 1000.0
        ),
        maximum_abs_error_mev=float(
            np.max(np.abs(difference[weighted_support])) * 1000.0
        ),
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


def _real_response_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("response matrix must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("response matrix must be finite")
    if np.iscomplexobj(array):
        return np.concatenate((array.real, array.imag), axis=0)
    return np.asarray(array, dtype=float)


def _orthonormal_response_basis(
    values: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    matrix = _real_response_matrix(values)
    if matrix.shape[1] == 0 or not np.any(matrix):
        return np.empty((matrix.shape[0], 0), dtype=float), 0
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    cutoff = float(tolerance) * max(float(singular_values[0]), 1.0)
    rank = int(np.count_nonzero(singular_values > cutoff))
    return left[:, :rank], rank


def response_subspace_overlap(
    left_response: np.ndarray,
    right_response: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
    defer_threshold: float = 0.999,
) -> ResponseSubspaceOverlap:
    """Audit whether two term families contain indistinguishable responses.

    Columns are independent real fit directions and rows are flattened target
    observations. Complex responses are represented by stacked real and
    imaginary parts, matching a fit with real-valued coefficients.
    """

    left = np.asarray(left_response)
    right = np.asarray(right_response)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("family responses must have shape (Nobservation, Nterm)")
    tol = float(tolerance)
    threshold = float(defer_threshold)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("defer_threshold must be finite and in [0, 1]")

    left_basis, left_rank = _orthonormal_response_basis(left, tolerance=tol)
    right_basis, right_rank = _orthonormal_response_basis(right, tolerance=tol)
    if left_rank == 0 or right_rank == 0:
        correlation = 0.0
    else:
        correlations = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
        correlation = float(np.clip(correlations[0], 0.0, 1.0))
    angle = float(np.degrees(np.arccos(np.clip(correlation, 0.0, 1.0))))
    return ResponseSubspaceOverlap(
        maximum_correlation=correlation,
        minimum_principal_angle_degrees=angle,
        left_rank=left_rank,
        right_rank=right_rank,
        defer_to_joint_stage=bool(correlation >= threshold),
    )


def fit_centered_linear_response(
    design: np.ndarray,
    target: np.ndarray,
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reference_index: int,
) -> CenteredLinearFit:
    """Fit real coefficients to ``response(k) - response(k_ref)``.

    Centering prevents a constant onsite or other omitted k-independent family
    from masquerading as a need for higher kinetic order during the first
    staged scan.
    """

    design_array = np.asarray(design, dtype=np.complex128)
    target_array = np.asarray(target, dtype=np.complex128)
    if design_array.ndim != 3 or target_array.ndim != 2:
        raise ValueError(
            "design and target must have shape (Nk, Nobservation, Nterm) "
            "and (Nk, Nobservation)"
        )
    if design_array.shape[:2] != target_array.shape:
        raise ValueError("design and target k/observation dimensions must match")
    if not np.all(np.isfinite(design_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("design and target must be finite")
    reference = int(reference_index)
    if reference < 0 or reference >= target_array.shape[0]:
        raise ValueError("reference_index is outside the k-point range")
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1 or train.size == 0 or validation.size == 0:
        raise ValueError("train_indices and validation_indices must be non-empty vectors")
    if np.any(train < 0) or np.any(train >= len(target_array)):
        raise ValueError("train_indices are outside the k-point range")
    if np.any(validation < 0) or np.any(validation >= len(target_array)):
        raise ValueError("validation_indices are outside the k-point range")

    centered_design = design_array - design_array[reference : reference + 1]
    centered_target = target_array - target_array[reference : reference + 1]
    train_design = centered_design[train].reshape(-1, design_array.shape[-1])
    train_target = centered_target[train].reshape(-1)
    real_design = np.concatenate((train_design.real, train_design.imag), axis=0)
    real_target = np.concatenate((train_target.real, train_target.imag), axis=0)
    coefficients, _, _, _ = np.linalg.lstsq(real_design, real_target, rcond=None)

    predicted = np.einsum(
        "kop,p->ko", centered_design[validation], coefficients, optimize=True
    )
    residual = predicted - centered_target[validation]
    validation_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    return CenteredLinearFit(
        coefficients=coefficients,
        validation_rms=validation_rms,
    )


def _selection_key(candidate: CandidateScore) -> tuple[object, ...]:
    return (
        int(candidate.independent_real_parameters),
        int(candidate.active_group_count),
        int(candidate.harmonic_support_size),
        sum(candidate.orders.as_tuple()),
        max(candidate.orders.as_tuple()),
        candidate.orders.as_tuple(),
        float(candidate.weighted_max_mev),
        float(candidate.weighted_rms_mev),
        str(candidate.name),
    )


def _quality_key(candidate: CandidateScore) -> tuple[object, ...]:
    expanded = candidate.expanded_weighted_rms_mev
    return (
        float(candidate.weighted_rms_mev),
        float(expanded) if expanded is not None else float("inf"),
        float(candidate.weighted_max_mev),
        int(candidate.independent_real_parameters),
        candidate.orders.as_tuple(),
        str(candidate.name),
    )


def select_simplest_near_best(
    candidates: Sequence[CandidateScore],
    *,
    overlap_target: float = 0.95,
    overlap_safety_floor: float = 0.90,
    enforce_overlap: bool = True,
    standard_error_multiplier: float = 1.0,
    prefer_simplest: bool = True,
    relative_rms_tolerance: float = 0.0,
    minimum_tolerance_mev: float = 0.0,
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
    se_multiplier = float(standard_error_multiplier)
    if not np.isfinite(se_multiplier) or se_multiplier < 0.0:
        raise ValueError("standard_error_multiplier must be finite and non-negative")
    relative_tolerance = float(relative_rms_tolerance)
    minimum_tolerance = float(minimum_tolerance_mev)
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
        or not np.isfinite(minimum_tolerance)
        or minimum_tolerance < 0.0
    ):
        raise ValueError("adaptive RMS tolerances must be finite and non-negative")

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

    if not bool(enforce_overlap):
        pool = valid
        status = "PASS"
        unmet_targets: tuple[str, ...] = ()
    else:
        pass_pool = [
            candidate
            for candidate in valid
            if float(candidate.mean_subspace_overlap) >= target
        ]
        if pass_pool:
            pool = pass_pool
            status = "PASS"
            unmet_targets = ()
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

    if not pool:
        return SelectionDecision(
            status="FAIL",
            selected=None,
            best_weighted_rms_mev=None,
            one_se_threshold_mev=None,
            unmet_targets=("valid_candidate",),
            rejected_reasons=rejected,
        )

    best = min(pool, key=_quality_key)
    best_rms = float(best.weighted_rms_mev)
    tolerance = max(
        se_multiplier * float(best.weighted_rms_se_mev),
        relative_tolerance * best_rms,
        minimum_tolerance,
    )
    threshold = best_rms + tolerance
    plateau = [
        candidate
        for candidate in pool
        if float(candidate.weighted_rms_mev) <= threshold
    ]
    selected = min(plateau, key=_selection_key) if prefer_simplest else best
    return SelectionDecision(
        status=status,
        selected=selected,
        best_weighted_rms_mev=float(best.weighted_rms_mev),
        one_se_threshold_mev=threshold,
        plateau_names=tuple(candidate.name for candidate in plateau),
        unmet_targets=unmet_targets,
        rejected_reasons=rejected,
    )


def select_high_low_profiles(
    candidates: Sequence[CandidateScore],
    *,
    overlap_target: float = 0.95,
    overlap_safety_floor: float = 0.90,
    low_se_multiplier: float = 2.0,
    low_relative_rms_tolerance: float = 1.0,
    low_minimum_tolerance_mev: float = 0.10,
) -> HighLowSelectionResult:
    """Select accurate ``high`` and compact edge-preserving ``low`` models.

    Both profiles use the primary edge-weighted loss stored in
    ``weighted_rms_mev``. Expanded-window error is a quality tie breaker for
    high and remains a diagnostic for low; it is never a low-profile gate.
    """

    multiplier = float(low_se_multiplier)
    high = select_simplest_near_best(
        candidates,
        overlap_target=overlap_target,
        overlap_safety_floor=overlap_safety_floor,
        standard_error_multiplier=0.0,
        prefer_simplest=False,
    )
    low = select_simplest_near_best(
        candidates,
        overlap_target=overlap_target,
        overlap_safety_floor=overlap_safety_floor,
        standard_error_multiplier=multiplier,
        prefer_simplest=True,
        relative_rms_tolerance=low_relative_rms_tolerance,
        minimum_tolerance_mev=low_minimum_tolerance_mev,
    )
    return HighLowSelectionResult(
        high=high,
        low=low,
        low_se_multiplier=multiplier,
        low_relative_rms_tolerance=float(low_relative_rms_tolerance),
        low_minimum_tolerance_mev=float(low_minimum_tolerance_mev),
    )


def select_four_model_profiles(
    candidates: Sequence[CandidateScore],
    *,
    overlap_target: float = 0.95,
    overlap_safety_floor: float = 0.90,
    low_se_multiplier: float = 2.0,
    low_relative_rms_tolerance: float = 1.0,
    low_minimum_tolerance_mev: float = 0.10,
) -> FourProfileSelectionResult:
    """Select high/low models independently for linear and nonlinear solvers."""

    groups = {
        family: tuple(
            candidate
            for candidate in candidates
            if candidate.solver_family == family
        )
        for family in ("linear", "nonlinear")
    }
    options = {
        "overlap_target": overlap_target,
        "overlap_safety_floor": overlap_safety_floor,
        "low_se_multiplier": low_se_multiplier,
        "low_relative_rms_tolerance": low_relative_rms_tolerance,
        "low_minimum_tolerance_mev": low_minimum_tolerance_mev,
    }
    return FourProfileSelectionResult(
        linear=select_high_low_profiles(groups["linear"], **options),
        nonlinear=select_high_low_profiles(groups["nonlinear"], **options),
    )


def _candidate_score_record(candidate: CandidateScore | None) -> dict[str, object] | None:
    if candidate is None:
        return None
    return {
        "name": str(candidate.name),
        "orders": list(candidate.orders.as_tuple()),
        "independent_real_parameters": int(candidate.independent_real_parameters),
        "primary_weighted_rms_mev": float(candidate.weighted_rms_mev),
        "primary_weighted_rms_se_mev": float(candidate.weighted_rms_se_mev),
        "primary_weighted_max_mev": float(candidate.weighted_max_mev),
        "expanded_weighted_rms_mev": (
            None
            if candidate.expanded_weighted_rms_mev is None
            else float(candidate.expanded_weighted_rms_mev)
        ),
        "mean_subspace_overlap": float(candidate.mean_subspace_overlap),
        "certified": bool(candidate.certified),
        "guards_passed": bool(candidate.guards_passed),
        "solver_family": str(candidate.solver_family),
        "active_group_count": int(candidate.active_group_count),
        "harmonic_support_size": int(candidate.harmonic_support_size),
    }


def _selection_decision_record(decision: SelectionDecision) -> dict[str, object]:
    return {
        "status": str(decision.status),
        "selected": _candidate_score_record(decision.selected),
        "best_primary_weighted_rms_mev": decision.best_weighted_rms_mev,
        "quality_threshold_mev": decision.one_se_threshold_mev,
        "plateau_names": list(decision.plateau_names),
        "unmet_targets": list(decision.unmet_targets),
        "rejected_reasons": dict(decision.rejected_reasons),
    }


def high_low_selection_record(
    result: HighLowSelectionResult,
) -> dict[str, object]:
    """Convert a dual-profile decision into a JSON-safe audit record."""

    high = _selection_decision_record(result.high)
    low = _selection_decision_record(result.low)
    high["standard_error_multiplier"] = 0.0
    low["standard_error_multiplier"] = float(result.low_se_multiplier)
    low["relative_rms_tolerance"] = float(result.low_relative_rms_tolerance)
    low["minimum_tolerance_mev"] = float(result.low_minimum_tolerance_mev)
    return {
        "profiles": {
            "high": high,
            "low": low,
        }
    }


def run_staged_family_selection(
    config: ModelSelectionConfig,
    evaluator: Callable[
        [str, FamilyOrders, tuple[str, ...]],
        CandidateScore,
    ],
) -> StagedSelectionResult:
    """Select family orders sequentially while jointly refitting active terms.

    ``evaluator`` is invoked afresh for every candidate with all families that
    are active at that stage. Only the previously selected *orders* are held
    fixed; an evaluator must therefore refit their coefficients together with
    the newly introduced family. Full-model subspace overlap is enforced only
    after the inter-family stage completes.
    """

    if not bool(config.enabled):
        raise ValueError("staged family selection requires an enabled config")
    current = FamilyOrders(0, 0, 0)
    stage_results: list[StageSelectionResult] = []
    for stage_index, stage in enumerate(_FAMILY_ORDER):
        active_families = _FAMILY_ORDER[: stage_index + 1]
        candidates: list[CandidateScore] = []
        for order in config.order_candidates[stage]:
            values = list(current.as_tuple())
            values[stage_index] = int(order)
            proposed = FamilyOrders(*values)
            candidate = evaluator(stage, proposed, active_families)
            if candidate.orders != proposed:
                raise ValueError(
                    f"evaluator returned orders {candidate.orders} for proposed {proposed}"
                )
            candidates.append(candidate)
        decision = select_simplest_near_best(
            candidates,
            overlap_target=config.overlap_target,
            overlap_safety_floor=config.overlap_safety_floor,
            enforce_overlap=stage == "inter",
        )
        stage_results.append(
            StageSelectionResult(
                stage=stage,
                active_families=active_families,
                candidates=tuple(candidates),
                decision=decision,
            )
        )
        if decision.selected is None:
            return StagedSelectionResult(
                status="FAIL",
                selected_orders=None,
                stages=tuple(stage_results),
            )
        current = decision.selected.orders
    return StagedSelectionResult(
        status=stage_results[-1].decision.status,
        selected_orders=current,
        stages=tuple(stage_results),
    )


def _validate_parameter_groups(groups: Sequence[ParameterGroup]) -> None:
    identifiers = [group.group_id for group in groups]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("parameter group ids must be unique")
    all_indices = [
        index for group in groups for index in group.parameter_indices
    ]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("parameter groups must be disjoint")


def run_group_ablation(
    full_candidate: CandidateScore,
    groups: Sequence[ParameterGroup],
    evaluator: Callable[[tuple[ParameterGroup, ...]], CandidateScore],
    *,
    overlap_target: float = 0.95,
    overlap_safety_floor: float = 0.90,
) -> GroupAblationResult:
    """Greedily remove closed parameter groups and refit after every removal.

    A removal is accepted only when the refitted reduced candidate is selected
    on the current model's one-standard-error plateau. Individual real/imaginary
    components or adjoint partners are never exposed to this routine separately;
    callers encode the required closure in each :class:`ParameterGroup`.
    """

    active = list(groups)
    _validate_parameter_groups(active)
    current = full_candidate
    steps: list[GroupAblationStep] = []
    status = "PASS"
    while active:
        attempted: list[
            tuple[ParameterGroup, tuple[ParameterGroup, ...], CandidateScore]
        ] = []
        names = {current.name}
        for removed in active:
            remaining = tuple(
                group for group in active if group.group_id != removed.group_id
            )
            candidate = evaluator(remaining)
            if candidate.name in names:
                raise ValueError(
                    "ablation evaluator candidate names must be unique per iteration"
                )
            names.add(candidate.name)
            attempted.append((removed, remaining, candidate))
        decision = select_simplest_near_best(
            [current, *(candidate for _, _, candidate in attempted)],
            overlap_target=overlap_target,
            overlap_safety_floor=overlap_safety_floor,
        )
        status = decision.status
        selected_name = decision.selected.name if decision.selected is not None else None
        chosen: (
            tuple[ParameterGroup, tuple[ParameterGroup, ...], CandidateScore] | None
        ) = None
        for removed, remaining, candidate in attempted:
            accepted = candidate.name == selected_name
            steps.append(
                GroupAblationStep(
                    removed_group_id=removed.group_id,
                    remaining_group_ids=tuple(group.group_id for group in remaining),
                    candidate=candidate,
                    validation_loss_increase_mev=float(
                        candidate.weighted_rms_mev - current.weighted_rms_mev
                    ),
                    accepted=accepted,
                )
            )
            if accepted:
                chosen = (removed, remaining, candidate)
        if chosen is None:
            break
        _, remaining, current = chosen
        active = list(remaining)
    return GroupAblationResult(
        status=status,
        selected=current,
        active_group_ids=tuple(group.group_id for group in active),
        steps=tuple(steps),
    )


def run_local_order_correction_sweep(
    selected_orders: FamilyOrders,
    *,
    maximum_orders: FamilyOrders,
    evaluator: Callable[[FamilyOrders], CandidateScore],
    overlap_target: float = 0.95,
    overlap_safety_floor: float = 0.90,
) -> CorrectionSweepResult:
    """Recheck the selected point and its single-family +/-1 neighbors."""

    selected = selected_orders.as_tuple()
    maximum = maximum_orders.as_tuple()
    if any(value > ceiling for value, ceiling in zip(selected, maximum)):
        raise ValueError("selected orders must not exceed maximum orders")
    order_points = {selected_orders}
    for family_index in range(len(_FAMILY_ORDER)):
        for delta in (-1, 1):
            values = list(selected)
            values[family_index] += delta
            if 0 <= values[family_index] <= maximum[family_index]:
                order_points.add(FamilyOrders(*values))
    candidates = tuple(evaluator(orders) for orders in sorted(order_points))
    for orders, candidate in zip(sorted(order_points), candidates):
        if candidate.orders != orders:
            raise ValueError(
                f"evaluator returned orders {candidate.orders} for proposed {orders}"
            )
    decision = select_simplest_near_best(
        candidates,
        overlap_target=overlap_target,
        overlap_safety_floor=overlap_safety_floor,
    )
    return CorrectionSweepResult(candidates=candidates, decision=decision)


__all__ = [
    "CandidateScore",
    "CenteredLinearFit",
    "CorrectionSweepResult",
    "FamilyVocabulary",
    "FamilyOrders",
    "FourProfileSelectionResult",
    "GroupAblationResult",
    "GroupAblationStep",
    "HighLowSelectionResult",
    "ModelSelectionConfig",
    "ParameterGroup",
    "ProductionCandidateSpec",
    "ResponseSubspaceOverlap",
    "SelectionDecision",
    "StageSelectionResult",
    "StagedSelectionResult",
    "SubspaceOverlapMetrics",
    "ValidationFold",
    "WeightedBandMetrics",
    "blocked_kpath_folds",
    "build_fixed_band_weights",
    "clone_family_vocabulary",
    "build_production_candidate_spec",
    "fit_centered_linear_response",
    "parse_model_selection_config",
    "high_low_selection_record",
    "response_subspace_overlap",
    "run_staged_family_selection",
    "run_group_ablation",
    "run_local_order_correction_sweep",
    "select_high_low_profiles",
    "select_four_model_profiles",
    "select_simplest_near_best",
    "subspace_overlap_metrics",
    "weighted_band_error",
]
