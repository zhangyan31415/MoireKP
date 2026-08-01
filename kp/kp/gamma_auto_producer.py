"""Pure, fail-closed automatic Gamma projection production.

The producer starts from already loaded physical arrays and explicit symmetry
metadata.  It has no CLI or filesystem side effects: every candidate is built
through the routed Gamma core, downfolded, symmetry-certified, and scored
before the smallest passing candidate is returned.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .blocks.downfold import DownfoldingOptions, downfold_from_projectors
from .blocks.gamma_layout import (
    GammaCertifiedRawAction,
    GammaRoutingError,
    GammaRoutingThresholds,
    GammaRowLayout,
    GammaRoutedFrames,
    assemble_gamma_routed_projectors,
    build_gamma_routed_frames,
    certify_gamma_raw_action,
    certify_routed_covariance,
    close_gamma_projector_clusters,
    gamma_certified_action_route_contract,
)
from .identity import build_projection_basis_identity, hash_array, hash_mapping
from .low_energy_selection import (
    CandidateMetrics,
    SelectionDecision,
    SelectionThresholds,
    select_projection_candidate,
)
from .projection_handoff import (
    GammaRoutedBasisSpec,
    certify_gamma_routed_basis_spec,
    gamma_sampled_k_route_contract,
)
from .projection_selection import (
    CandidateRejected,
    CandidateRejectionReason,
    FrozenTargetWindow,
    TargetWindowSpec,
    evaluate_fixed_target_window,
    projection_overlap_metrics,
    resolve_target_window,
)
from .selection_artifact import (
    SelectionInputIdentity,
    build_selection_policy_hash,
    gamma_routed_ordered_q_identity_hash,
    hash_frozen_target_window,
    hash_validation_k_indices,
)
from .symmetry.candidate_certificate import (
    CandidateOperationInput,
    CandidateProjectionState,
    CandidateSymmetryCertificate,
    CandidateSymmetryStatus,
    CandidateSymmetryThresholds,
    candidate_action_package_hash,
    candidate_raw_action_package_hash,
    certify_candidate_symmetries,
    evaluate_projected_pair,
)
from .symmetry.joint_exactification import (
    BlockRouteAction,
    JointExactificationConfig,
    JointExactificationError,
    MagneticPresentation,
    certify_joint_block_actions,
    joint_exactify_block_actions,
    materialize_block_route_action,
)


GAMMA_AUTO_PRODUCER_VERSION = "kp.gamma-auto-producer.v1"
GAMMA_AUTO_CANDIDATE_SCHEMA = "kp.gamma-auto-candidate.v1"
GAMMA_AUTO_METRIC_SCHEMA = "kp.candidate-metrics.v1"
GAMMA_AUTO_ORDERING_RULE = "dimension-error-overlap-symmetry-v1"


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    payload = dict(value)
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValueError(f"{context} field mismatch ({'; '.join(details)})")
    return payload


def _strict_real(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a strict finite numeric value")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a strict finite numeric value") from error
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{field} must be a strict finite numeric value")
    return result


def _strict_positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a strict positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a strict positive integer")
    return result


def _strict_indices(value: Any, *, field: str, nonempty: bool = True) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray, np.ndarray)):
        if isinstance(value, np.ndarray):
            raw = tuple(value.tolist())
        else:
            raise ValueError(f"{field} must be a sequence of strict integers")
    else:
        try:
            raw = tuple(value)
        except TypeError as error:
            raise ValueError(f"{field} must be a sequence of strict integers") from error
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
        for item in raw
    ):
        raise ValueError(f"{field} must contain strict integers")
    result = tuple(int(item) for item in raw)
    if (nonempty and not result) or any(item < 0 for item in result):
        raise ValueError(f"{field} must contain nonnegative indices")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique indices")
    return result


def _sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 identity")
    if text == "0" * 64:
        raise ValueError(f"{field} must not be a placeholder identity")
    return text


@dataclass(frozen=True)
class GammaDownfoldConfig:
    options: DownfoldingOptions

    @classmethod
    def from_normalized_config(cls, value: Any) -> "GammaDownfoldConfig":
        payload = _require_exact_keys(
            value,
            {
                "method",
                "e_ref",
                "pole_warning_mev",
                "pole_danger_mev",
                "fail_on_near_pole",
                "compute_pole_diagnostics",
                "compute_condition_number",
            },
            context="automatic Gamma downfold",
        )
        method = str(payload["method"]).strip().lower()
        if method not in {"first_order", "fixed_schur", "linearized_lowdin"}:
            raise ValueError("automatic Gamma downfold method is unsupported")
        e_ref_raw = payload["e_ref"]
        if method == "first_order":
            if e_ref_raw is not None:
                raise ValueError("first_order automatic Gamma downfold requires e_ref=null")
            e_ref = None
        else:
            e_ref = _strict_real(e_ref_raw, field="downfold.e_ref")
        for name in (
            "fail_on_near_pole",
            "compute_pole_diagnostics",
            "compute_condition_number",
        ):
            if type(payload[name]) is not bool:
                raise ValueError(f"downfold.{name} must be a strict bool")
        warning = _strict_real(
            payload["pole_warning_mev"],
            field="downfold.pole_warning_mev",
            nonnegative=True,
        )
        danger = _strict_real(
            payload["pole_danger_mev"],
            field="downfold.pole_danger_mev",
            nonnegative=True,
        )
        if danger > warning:
            raise ValueError("downfold pole_danger_mev cannot exceed pole_warning_mev")
        return cls(
            DownfoldingOptions(
                method=method,
                e_ref=e_ref,
                pole_warning_mev=warning,
                pole_danger_mev=danger,
                fail_on_near_pole=payload["fail_on_near_pole"],
                compute_pole_diagnostics=payload["compute_pole_diagnostics"],
                compute_condition_number=payload["compute_condition_number"],
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": self.options.method,
            "e_ref": self.options.e_ref,
            "pole_warning_mev": self.options.pole_warning_mev,
            "pole_danger_mev": self.options.pole_danger_mev,
            "fail_on_near_pole": self.options.fail_on_near_pole,
            "compute_pole_diagnostics": self.options.compute_pole_diagnostics,
            "compute_condition_number": self.options.compute_condition_number,
        }


@dataclass(frozen=True)
class GammaAutomaticSelectionConfig:
    routing_thresholds: GammaRoutingThresholds
    selection_thresholds: SelectionThresholds
    candidate_symmetry_thresholds: CandidateSymmetryThresholds
    exactification: JointExactificationConfig
    target_window_spec: TargetWindowSpec
    candidate_seed_band_indices: tuple[tuple[int, ...], ...]
    reference_k_index: int
    downfold: GammaDownfoldConfig

    @classmethod
    def from_normalized_config(
        cls,
        value: Any,
    ) -> "GammaAutomaticSelectionConfig":
        payload = _require_exact_keys(
            value,
            {
                "mode",
                "routing_thresholds",
                "selection_thresholds",
                "candidate_symmetry_thresholds",
                "exactification",
                "target_window",
                "candidate_seed_band_indices",
                "reference_k_index",
                "downfold",
            },
            context="automatic Gamma selection",
        )
        if payload["mode"] != "auto":
            raise ValueError("automatic Gamma selection mode must be exactly 'auto'")
        routing = GammaRoutingThresholds.from_normalized_config(
            payload["routing_thresholds"]
        )

        selection_payload = _require_exact_keys(
            payload["selection_thresholds"],
            {item.name for item in fields(SelectionThresholds)},
            context="automatic Gamma selection_thresholds",
        )
        selection_values = {
            name: _strict_real(value, field=f"selection_thresholds.{name}")
            for name, value in selection_payload.items()
        }
        selection = SelectionThresholds(**selection_values)
        # Reuse the public selector as the authoritative domain validator.
        select_projection_candidate(
            (
                CandidateMetrics(
                    candidate_id="config-domain-probe",
                    dimension=1,
                    band_rms_mev=0.0,
                    band_max_mev=0.0,
                    subspace_overlap=1.0,
                    symmetry_residual=0.0,
                    symmetry_leakage=0.0,
                ),
            ),
            selection,
        )

        symmetry_payload = _require_exact_keys(
            payload["candidate_symmetry_thresholds"],
            {item.name for item in fields(CandidateSymmetryThresholds)},
            context="automatic Gamma candidate_symmetry_thresholds",
        )
        candidate_symmetry = CandidateSymmetryThresholds(
            **{
                name: _strict_real(
                    raw,
                    field=f"candidate_symmetry_thresholds.{name}",
                    nonnegative=True,
                )
                for name, raw in symmetry_payload.items()
            }
        )

        exact_payload = _require_exact_keys(
            payload["exactification"],
            {
                "enabled",
                "max_rms_correction",
                "max_route_correction",
                "central_branch_margin",
                "max_iterations",
                "condition_limit",
            },
            context="automatic Gamma exactification",
        )
        if exact_payload["enabled"] is not True:
            raise ValueError("automatic Gamma exactification.enabled must be true")
        exactification = JointExactificationConfig(
            enabled=True,
            max_rms_correction=_strict_real(
                exact_payload["max_rms_correction"],
                field="exactification.max_rms_correction",
            ),
            max_route_correction=_strict_real(
                exact_payload["max_route_correction"],
                field="exactification.max_route_correction",
            ),
            central_branch_margin=_strict_real(
                exact_payload["central_branch_margin"],
                field="exactification.central_branch_margin",
            ),
            max_iterations=_strict_positive_integer(
                exact_payload["max_iterations"],
                field="exactification.max_iterations",
            ),
            condition_limit=_strict_real(
                exact_payload["condition_limit"],
                field="exactification.condition_limit",
            ),
        )

        target_payload = _require_exact_keys(
            payload["target_window"],
            {
                "edge",
                "band_count",
                "validation_k_indices",
                "energy_reference_ev",
                "degeneracy_tolerance_mev",
            },
            context="automatic Gamma target_window",
        )
        target_spec = TargetWindowSpec(
            edge=str(target_payload["edge"]),
            band_count=_strict_positive_integer(
                target_payload["band_count"], field="target_window.band_count"
            ),
            validation_k_indices=_strict_indices(
                target_payload["validation_k_indices"],
                field="target_window.validation_k_indices",
            ),
            energy_reference_ev=_strict_real(
                target_payload["energy_reference_ev"],
                field="target_window.energy_reference_ev",
            ),
            degeneracy_tolerance_mev=_strict_real(
                target_payload["degeneracy_tolerance_mev"],
                field="target_window.degeneracy_tolerance_mev",
                nonnegative=True,
            ),
        )

        try:
            raw_seeds = tuple(payload["candidate_seed_band_indices"])
        except TypeError as error:
            raise ValueError(
                "candidate_seed_band_indices must be a sequence"
            ) from error
        seeds = tuple(
            _strict_indices(seed, field=f"candidate_seed_band_indices[{index}]")
            for index, seed in enumerate(raw_seeds)
        )
        if not seeds:
            raise ValueError("candidate_seed_band_indices must be nonempty")
        if len(set(seeds)) != len(seeds):
            raise ValueError("candidate_seed_band_indices must be unique")
        reference = _strict_indices(
            (payload["reference_k_index"],),
            field="reference_k_index",
        )[0]
        return cls(
            routing_thresholds=routing,
            selection_thresholds=selection,
            candidate_symmetry_thresholds=candidate_symmetry,
            exactification=exactification,
            target_window_spec=target_spec,
            candidate_seed_band_indices=seeds,
            reference_k_index=reference,
            downfold=GammaDownfoldConfig.from_normalized_config(payload["downfold"]),
        )

    def policy_payload(self) -> dict[str, Any]:
        return {
            "schema": "kp.gamma-auto-policy-payload.v1",
            "routing_thresholds": self.routing_thresholds.to_payload(),
            "selection_thresholds": {
                item.name: getattr(self.selection_thresholds, item.name)
                for item in fields(SelectionThresholds)
            },
            "candidate_symmetry_thresholds": {
                item.name: getattr(self.candidate_symmetry_thresholds, item.name)
                for item in fields(CandidateSymmetryThresholds)
            },
            "exactification": {
                item.name: getattr(self.exactification, item.name)
                for item in fields(JointExactificationConfig)
            },
            "target_window": {
                "edge": self.target_window_spec.edge,
                "band_count": self.target_window_spec.band_count,
                "validation_k_indices": list(
                    self.target_window_spec.validation_k_indices
                ),
                "energy_reference_ev": self.target_window_spec.energy_reference_ev,
                "degeneracy_tolerance_mev": (
                    self.target_window_spec.degeneracy_tolerance_mev
                ),
            },
            "candidate_seed_band_indices": [
                list(seed) for seed in self.candidate_seed_band_indices
            ],
            "reference_k_index": self.reference_k_index,
            "downfold": self.downfold.to_payload(),
        }


@dataclass(frozen=True)
class GammaRawOperationSpec:
    name: str
    full_action: Any
    antiunitary: bool
    q_permutations: tuple[tuple[int, ...], tuple[int, ...]]
    sector_map: tuple[int, int]
    pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Gamma raw operation name must be nonempty")
        if type(self.antiunitary) is not bool:
            raise ValueError("Gamma raw operation antiunitary must be a strict bool")
        try:
            q_routes = tuple(
                _strict_indices(route, field=f"operation {name} q_permutations")
                for route in self.q_permutations
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"operation {name} has invalid q_permutations") from error
        if len(q_routes) != 2:
            raise ValueError(f"operation {name} requires two Q permutations")
        sector = _strict_indices(self.sector_map, field=f"operation {name} sector_map")
        if len(sector) != 2 or set(sector) != {0, 1}:
            raise ValueError(f"operation {name} sector_map must permute two groups")
        try:
            raw_pairs = tuple(self.pairs)
        except TypeError as error:
            raise ValueError(f"operation {name} pairs must be a sequence") from error
        pairs: list[tuple[int, int]] = []
        for pair in raw_pairs:
            if len(pair) != 2:
                raise ValueError(f"operation {name} pairs must contain k pairs")
            target, source = pair
            if any(
                isinstance(item, (bool, np.bool_))
                or not isinstance(item, Integral)
                or int(item) < 0
                for item in (target, source)
            ):
                raise ValueError(
                    f"operation {name} pairs must contain nonnegative strict integers"
                )
            pairs.append((int(target), int(source)))
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError(f"operation {name} pairs must be nonempty and unique")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "q_permutations", q_routes)
        object.__setattr__(self, "sector_map", sector)
        object.__setattr__(self, "pairs", tuple(pairs))


@dataclass(frozen=True)
class GammaAutomaticProducerInputs:
    source_hamiltonians: np.ndarray
    k_indices: tuple[int, ...]
    kpoints: np.ndarray
    qsets: tuple[np.ndarray, np.ndarray]
    num_layer_list: tuple[int, int]
    num_orb_per_layer_list: tuple[tuple[int, ...], tuple[int, ...]]
    tapw_source_basis_hash: str
    operations: tuple[GammaRawOperationSpec, ...]
    presentation: MagneticPresentation

    def __post_init__(self) -> None:
        hamiltonians = np.array(
            self.source_hamiltonians,
            dtype=np.complex128,
            copy=True,
            order="C",
        )
        if (
            hamiltonians.ndim != 3
            or hamiltonians.shape[0] <= 0
            or hamiltonians.shape[1] != hamiltonians.shape[2]
            or not np.all(np.isfinite(hamiltonians))
        ):
            raise ValueError("source_hamiltonians must be finite (Nk,N,N)")
        k_indices = _strict_indices(self.k_indices, field="production k_indices")
        if len(k_indices) != hamiltonians.shape[0]:
            raise ValueError("production k_indices do not match source_hamiltonians")
        kpoints = np.array(self.kpoints, dtype=np.float64, copy=True, order="C")
        if (
            kpoints.shape != (len(k_indices), 2)
            or not np.all(np.isfinite(kpoints))
        ):
            raise ValueError("production kpoints must be finite (Nk,2) in k_indices order")
        qsets = tuple(
            np.array(qset, dtype=np.float64, copy=True, order="C")
            for qset in self.qsets
        )
        if len(qsets) != 2 or any(
            qset.ndim != 2 or qset.shape[0] <= 0 or not np.all(np.isfinite(qset))
            for qset in qsets
        ):
            raise ValueError("Gamma qsets must contain two finite nonempty matrices")
        basis_hash = _sha256(
            self.tapw_source_basis_hash,
            field="tapw_source_basis_hash",
        )
        operations = tuple(self.operations)
        if not operations or any(
            not isinstance(operation, GammaRawOperationSpec)
            for operation in operations
        ):
            raise ValueError("Gamma automatic production requires raw operations")
        names = tuple(operation.name for operation in operations)
        if len(set(names)) != len(names):
            raise ValueError("Gamma raw operation names must be unique")
        presentation_names = tuple(
            generator.name for generator in self.presentation.generators
        )
        if set(names) != set(presentation_names):
            raise ValueError("Gamma raw operations must match the magnetic presentation")
        hamiltonians.setflags(write=False)
        kpoints.setflags(write=False)
        for qset in qsets:
            qset.setflags(write=False)
        object.__setattr__(self, "source_hamiltonians", hamiltonians)
        object.__setattr__(self, "k_indices", k_indices)
        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "qsets", qsets)
        object.__setattr__(self, "tapw_source_basis_hash", basis_hash)
        object.__setattr__(self, "operations", operations)


@dataclass(frozen=True)
class GammaCandidateRejection:
    seed_band_indices: tuple[int, ...]
    reason: CandidateRejectionReason
    diagnostic: str


@dataclass(frozen=True)
class GammaCandidateEvaluation:
    metrics: CandidateMetrics
    handoff: GammaRoutedBasisSpec
    symmetry_certificate: CandidateSymmetryCertificate


@dataclass(frozen=True)
class GammaAutomaticSelectionResult:
    selection_input: SelectionInputIdentity
    candidates: tuple[CandidateMetrics, ...]
    rejected_candidates: tuple[GammaCandidateRejection, ...]
    evaluations: tuple[GammaCandidateEvaluation, ...]
    decision: SelectionDecision
    handoff: GammaRoutedBasisSpec
    frozen_target_window: FrozenTargetWindow


@dataclass(frozen=True)
class GammaAutomaticSelectionPreparation:
    """Candidate-independent, identity-complete automatic Gamma inputs."""

    inputs: GammaAutomaticProducerInputs
    config: GammaAutomaticSelectionConfig
    layout: GammaRowLayout
    certified_actions: tuple[GammaCertifiedRawAction, ...]
    operation_inputs: Mapping[str, CandidateOperationInput]
    required_pairs: Mapping[str, tuple[tuple[int, int], ...]]
    raw_action_package_hash: str
    local_values: tuple[tuple[np.ndarray, ...], ...]
    local_vectors: tuple[tuple[np.ndarray, ...], ...]
    frozen_target_window: FrozenTargetWindow
    target_values: np.ndarray
    target_vectors: tuple[np.ndarray, ...]
    source_hamiltonian_hash: str
    selection_input: SelectionInputIdentity


def _local_eigensystems(
    hamiltonians: np.ndarray,
    layout: GammaRowLayout,
    *,
    hermiticity_tolerance: float,
) -> tuple[tuple[tuple[np.ndarray, ...], ...], tuple[tuple[np.ndarray, ...], ...]]:
    values_by_k: list[tuple[np.ndarray, ...]] = []
    vectors_by_k: list[tuple[np.ndarray, ...]] = []
    for k_position, hamiltonian in enumerate(hamiltonians):
        scale = max(1.0, float(np.linalg.norm(hamiltonian, ord="fro")))
        residual = float(
            np.linalg.norm(hamiltonian - hamiltonian.conj().T, ord="fro") / scale
        )
        if residual > hermiticity_tolerance:
            raise ValueError(
                f"source Hamiltonian k-position {k_position} is not Hermitian"
            )
        values_by_q: list[np.ndarray] = []
        vectors_by_q: list[np.ndarray] = []
        for q_index in range(layout.q_count):
            rows = layout.same_q_full_rows(q_index)
            block = np.asarray(
                hamiltonian[np.ix_(rows, rows)], dtype=np.complex128
            )
            values, vectors = np.linalg.eigh(block)
            values_by_q.append(values)
            vectors_by_q.append(vectors)
        values_by_k.append(tuple(values_by_q))
        vectors_by_k.append(tuple(vectors_by_q))
    return tuple(values_by_k), tuple(vectors_by_k)


def _target_eigensystems(
    hamiltonians: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    from threadpoolctl import threadpool_limits

    requested_workers = _strict_positive_integer(
        workers,
        field="target eigensystem workers",
    )
    effective_workers = min(requested_workers, len(hamiltonians))
    with threadpool_limits(limits=1, user_api="blas"):
        if effective_workers == 1:
            eigensystems = tuple(
                np.linalg.eigh(hamiltonian) for hamiltonian in hamiltonians
            )
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                eigensystems = tuple(executor.map(np.linalg.eigh, hamiltonians))
    values = [eigensystem[0] for eigensystem in eigensystems]
    vectors = [eigensystem[1] for eigensystem in eigensystems]
    return np.stack(values, axis=0), tuple(vectors)


def _certify_source_hamiltonian_covariance(
    inputs: GammaAutomaticProducerInputs,
    *,
    threshold: float,
) -> None:
    positions = {k_index: position for position, k_index in enumerate(inputs.k_indices)}
    for operation in inputs.operations:
        action = operation.full_action
        adjoint = action.conjugate().transpose()
        for target_k, source_k in operation.pairs:
            if target_k not in positions or source_k not in positions:
                raise ValueError(
                    f"operation {operation.name} references k outside production coverage"
                )
            target = inputs.source_hamiltonians[positions[target_k]]
            source = inputs.source_hamiltonians[positions[source_k]]
            transformed_source = source.conj() if operation.antiunitary else source
            transformed = action @ transformed_source @ adjoint
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            transformed = np.asarray(transformed, dtype=np.complex128)
            scale = max(1.0, float(np.linalg.norm(target, ord="fro")))
            residual = float(
                np.linalg.norm(target - transformed, ord="fro") / scale
            )
            if not np.isfinite(residual) or residual > threshold:
                raise GammaRoutingError(
                    CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
                    f"operation {operation.name} source Hamiltonian covariance "
                    f"residual {residual:.3e} exceeds gate {threshold:.3e}",
                )


def _edge_indices(values: np.ndarray, spec: TargetWindowSpec) -> np.ndarray:
    if spec.edge == "valence":
        eligible = np.flatnonzero(values <= spec.energy_reference_ev)
        selected = eligible[-spec.band_count :]
    else:
        eligible = np.flatnonzero(values >= spec.energy_reference_ev)
        selected = eligible[: spec.band_count]
    if selected.size != spec.band_count:
        raise CandidateRejected(
            CandidateRejectionReason.INSUFFICIENT_TARGET_BANDS,
            "spectrum does not contain the fixed target edge window",
            required_band_count=spec.band_count,
            available_band_count=int(eligible.size),
        )
    return selected


def _canonical_candidate_id(
    *,
    joint_band_indices: Sequence[int],
    closure_hashes: Sequence[str],
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
) -> str:
    identity = hash_mapping(
        {
            "schema": GAMMA_AUTO_CANDIDATE_SCHEMA,
            "joint_band_indices": list(joint_band_indices),
            "closure_hashes": list(closure_hashes),
            "layout_hash": layout.layout_hash,
            "routing_thresholds_hash": thresholds.identity_hash,
        }
    )
    return "gamma-routed-" + identity[:24]


def _reference_anchor_frames(routed: GammaRoutedFrames) -> tuple[np.ndarray, ...]:
    return tuple(
        np.ascontiguousarray(np.column_stack(groups), dtype=np.complex128)
        for groups in routed.reference_frames_by_q_group
    )


def _routing_certificate_hash(
    *,
    k_index: int,
    routed: GammaRoutedFrames,
    residuals: Sequence[tuple[str, int, float]],
    actions: Sequence[GammaCertifiedRawAction],
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
) -> str:
    return hash_mapping(
        {
            "schema": "kp.gamma-routing-covariance-certificate.v1",
            "k_index": int(k_index),
            "layout_hash": layout.layout_hash,
            "thresholds_hash": thresholds.identity_hash,
            "frame_hash": routed.frame_hash,
            "action_hashes": [action.action_hash for action in actions],
            "residuals": [
                [str(name), int(q_index), float(value)]
                for name, q_index, value in residuals
            ],
        }
    )


def _polar_unitary(matrix: np.ndarray, *, sigma_minimum: float) -> np.ndarray:
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or float(np.min(singular)) < sigma_minimum:
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            "candidate projected symmetry action has a singular polar anchor",
        )
    return np.ascontiguousarray(left @ right, dtype=np.complex128)


def _exactified_candidate_actions(
    *,
    states: Mapping[int, CandidateProjectionState],
    operations: Mapping[str, CandidateOperationInput],
    presentation: MagneticPresentation,
    config: JointExactificationConfig,
    sigma_minimum: float,
) -> dict[str, np.ndarray]:
    route_actions: dict[str, BlockRouteAction] = {}
    for generator in presentation.generators:
        operation = operations[generator.name]
        projected: list[np.ndarray] = []
        for target_k, source_k in operation.pairs:
            target = states[target_k]
            source = states[source_k]
            projected.append(
                evaluate_projected_pair(
                    d_full=operation.d_full,
                    target_u_low=target.u_low,
                    source_u_low=source.u_low,
                    target_heff=None,
                    source_heff=None,
                    antiunitary=operation.antiunitary,
                    compute_heff_covariance=False,
                ).projected_action
            )
        mean_action = np.mean(np.stack(projected, axis=0), axis=0)
        unitary = _polar_unitary(mean_action, sigma_minimum=sigma_minimum)
        dimension = int(unitary.shape[0])
        route_actions[generator.name] = BlockRouteAction(
            name=generator.name,
            antiunitary=generator.antiunitary,
            fiber_permutation=(0,),
            fiber_dimensions=(dimension,),
            route_blocks=(unitary,),
        )
    try:
        certify_joint_block_actions(route_actions, presentation)
        exact_actions = route_actions
    except JointExactificationError:
        exact_actions = dict(
            joint_exactify_block_actions(
                route_actions,
                presentation,
                config=config,
            ).actions
        )
    return {
        name: materialize_block_route_action(action)
        for name, action in exact_actions.items()
    }


def _certificate_metrics(
    certificate: CandidateSymmetryCertificate,
) -> tuple[float, float]:
    leakage_values: list[float] = []
    residual_values: list[float] = []
    for operation in certificate.operations:
        if operation.exact_action_unitarity_residual is not None:
            residual_values.append(operation.exact_action_unitarity_residual)
        if operation.antiunitary_square_residual is not None:
            residual_values.append(operation.antiunitary_square_residual)
        for pair in operation.pairs:
            if pair.raw_h_leakage is not None:
                leakage_values.append(pair.raw_h_leakage)
            for value in (
                pair.exactification_distance,
                pair.intertwining_residual,
                pair.heff_covariance_residual,
            ):
                if value is not None:
                    residual_values.append(value)
    for relation in certificate.relations:
        if relation.residual is not None:
            residual_values.append(relation.residual)
    for state in certificate.states:
        for value in (
            state.projection_orthonormality_residual,
            state.heff_hermiticity_residual,
        ):
            if value is not None:
                residual_values.append(value)
    return (
        max(residual_values, default=0.0),
        max(leakage_values, default=0.0),
    )


def _evaluate_candidate(
    *,
    seed: tuple[int, ...],
    inputs: GammaAutomaticProducerInputs,
    config: GammaAutomaticSelectionConfig,
    layout: GammaRowLayout,
    certified_actions: tuple[GammaCertifiedRawAction, ...],
    operation_inputs: Mapping[str, CandidateOperationInput],
    required_pairs: Mapping[str, tuple[tuple[int, int], ...]],
    raw_action_package_hash: str,
    local_values: tuple[tuple[np.ndarray, ...], ...],
    local_vectors: tuple[tuple[np.ndarray, ...], ...],
    frozen_target: FrozenTargetWindow,
    target_values: np.ndarray,
    target_vectors: tuple[np.ndarray, ...],
    source_hamiltonian_hash: str,
) -> GammaCandidateEvaluation:
    closures = tuple(
        close_gamma_projector_clusters(
            local_values[position],
            local_vectors[position],
            layout=layout,
            seed_band_indices=tuple(seed for _ in range(layout.q_count)),
            actions=certified_actions,
            thresholds=config.routing_thresholds,
        )
        for position in range(len(inputs.k_indices))
    )
    joint = closures[0].band_indices_by_q[0]
    if any(
        bands != joint
        for closure in closures
        for bands in closure.band_indices_by_q
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
            "Gamma candidate closure band set changes across production k/Q",
        )
    closure_hashes = tuple(closure.closure_hash for closure in closures)
    candidate_id = _canonical_candidate_id(
        joint_band_indices=joint,
        closure_hashes=closure_hashes,
        layout=layout,
        thresholds=config.routing_thresholds,
    )
    reference_position = inputs.k_indices.index(config.reference_k_index)
    reference = build_gamma_routed_frames(
        local_values[reference_position],
        local_vectors[reference_position],
        joint_band_indices=joint,
        layout=layout,
        thresholds=config.routing_thresholds,
        require_complete_clusters=True,
    )
    anchors = _reference_anchor_frames(reference)
    routed_by_position: list[GammaRoutedFrames] = []
    routing_hashes: list[str] = []
    for position, k_index in enumerate(inputs.k_indices):
        routed = build_gamma_routed_frames(
            local_values[position],
            local_vectors[position],
            joint_band_indices=joint,
            layout=layout,
            thresholds=config.routing_thresholds,
            anchor_frames=anchors,
            require_complete_clusters=True,
        )
        residuals = certify_routed_covariance(
            routed.routed_projectors_by_q,
            layout=layout,
            actions=certified_actions,
            thresholds=config.routing_thresholds,
        )
        routed_by_position.append(routed)
        routing_hashes.append(
            _routing_certificate_hash(
                k_index=k_index,
                routed=routed,
                residuals=residuals,
                actions=certified_actions,
                layout=layout,
                thresholds=config.routing_thresholds,
            )
        )

    u_low_by_position: list[np.ndarray] = []
    authoritative_heff: list[np.ndarray] = []
    for position, routed in enumerate(routed_by_position):
        include_high = config.downfold.options.method != "first_order"
        u_low, u_high = assemble_gamma_routed_projectors(
            routed,
            layout=layout,
            thresholds=config.routing_thresholds,
            include_high=include_high,
        )
        result = downfold_from_projectors(
            inputs.source_hamiltonians[position],
            u_low,
            u_high,
            config.downfold.options,
        )
        if not np.all(np.isfinite(result.heff)):
            raise ValueError("automatic Gamma downfold produced nonfinite Heff")
        if result.hermiticity_residual > config.candidate_symmetry_thresholds.heff_hermiticity_residual:
            raise ValueError("automatic Gamma downfold failed the Hermiticity gate")
        u_low_by_position.append(u_low)
        authoritative_heff.append(np.asarray(result.heff, dtype=np.complex128))
    heff_tensor = np.stack(authoritative_heff, axis=0)
    states = {
        k_index: CandidateProjectionState(
            u_low=u_low_by_position[position],
            heff=heff_tensor[position],
        )
        for position, k_index in enumerate(inputs.k_indices)
    }
    exactified_actions = _exactified_candidate_actions(
        states=states,
        operations=operation_inputs,
        presentation=inputs.presentation,
        config=config.exactification,
        sigma_minimum=config.routing_thresholds.anchor_sigma_min,
    )
    certificate = certify_candidate_symmetries(
        candidate_id=candidate_id,
        states=states,
        operations=operation_inputs,
        exactified_actions=exactified_actions,
        presentation=inputs.presentation,
        required_pairs=required_pairs,
        thresholds=config.candidate_symmetry_thresholds,
        required_state_k_indices=inputs.k_indices,
    )
    if certificate.status is not CandidateSymmetryStatus.CERTIFIED:
        raise GammaRoutingError(
            CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
            "Gamma automatic candidate symmetry failed: "
            + ", ".join(certificate.failures),
        )
    certified_raw_action_package_hash = candidate_action_package_hash(
        certificate.input_identity_payload
    )
    if certified_raw_action_package_hash != raw_action_package_hash:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "candidate certificate raw-action package differs from preselection input",
        )

    model_dim = int(heff_tensor.shape[-1])
    artifact_identity = build_projection_basis_identity(
        qset1=inputs.qsets[0],
        qset2=inputs.qsets[1],
        source_hamiltonian_hash=source_hamiltonian_hash,
        spin="all",
        mode="Gamma",
        energy_scale=1.0,
        nlow_state_list={
            "projection_basis_kind": "gamma_routed",
            "joint_band_indices": list(joint),
        },
        resolved_norb_fix_list=[],
        gauge_mode="gamma_routed",
        num_layer_list=list(layout.num_layer_list),
        num_orb_per_layer_list=[
            [layout.uniform_orbital_count] * count
            for count in layout.num_layer_list
        ],
        orbital_block_dim=layout.same_q_dimension,
        model_dim=model_dim,
        k_indices=inputs.k_indices,
        gauge_frame_hash=hash_array(
            np.stack(
                [
                    np.stack(routed.local_frames_by_q, axis=0)
                    for routed in routed_by_position
                ],
                axis=0,
            )
        ),
    )
    handoff = certify_gamma_routed_basis_spec(
        candidate_id=candidate_id,
        artifact_identity=artifact_identity,
        layout=layout,
        thresholds=config.routing_thresholds,
        k_indices=inputs.k_indices,
        kpoints=inputs.kpoints,
        routed_frames=tuple(routed_by_position),
        authoritative_heff=heff_tensor,
        heff_k_indices=inputs.k_indices,
        closure_certificate_hashes=closure_hashes,
        routing_certificate_hashes=tuple(routing_hashes),
        source_hamiltonian_hash=source_hamiltonian_hash,
        ordered_q_hashes=layout.ordered_qset_hashes,
        raw_action_package_hash=raw_action_package_hash,
        operations=operation_inputs,
        exactified_actions=exactified_actions,
        presentation=inputs.presentation,
        required_pairs=required_pairs,
        candidate_thresholds=config.candidate_symmetry_thresholds,
        certified_gamma_actions=certified_actions,
    )

    candidate_eigenvalues: list[np.ndarray] = []
    model_basis: list[np.ndarray] = []
    target_basis: list[np.ndarray] = []
    for position, heff in enumerate(heff_tensor):
        values, vectors = np.linalg.eigh(heff)
        candidate_eigenvalues.append(values)
        model_indices = (
            np.arange(model_dim - config.target_window_spec.band_count, model_dim)
            if config.target_window_spec.edge == "valence"
            else np.arange(config.target_window_spec.band_count)
        )
        model_basis.append(
            np.ascontiguousarray(
                u_low_by_position[position] @ vectors[:, model_indices]
            )
        )
        target_indices = _edge_indices(
            target_values[position], config.target_window_spec
        )
        target_basis.append(
            np.ascontiguousarray(target_vectors[position][:, target_indices])
        )
    band_metrics = evaluate_fixed_target_window(
        frozen_target,
        np.stack(candidate_eigenvalues, axis=0),
    )
    overlap = projection_overlap_metrics(
        np.stack(model_basis, axis=0),
        np.stack(target_basis, axis=0),
        validation_k_indices=config.target_window_spec.validation_k_indices,
        projection_basis=np.stack(u_low_by_position, axis=0),
    )
    symmetry_residual, symmetry_leakage = _certificate_metrics(certificate)
    metrics = CandidateMetrics(
        candidate_id=candidate_id,
        dimension=model_dim,
        band_rms_mev=band_metrics.rms_error_mev,
        band_max_mev=band_metrics.maximum_abs_error_mev,
        subspace_overlap=min(
            overlap.minimum_principal_overlap_squared,
            overlap.target_capture,
        ),
        symmetry_residual=symmetry_residual,
        symmetry_leakage=symmetry_leakage,
    )
    return GammaCandidateEvaluation(
        metrics=metrics,
        handoff=handoff,
        symmetry_certificate=certificate,
    )


def prepare_gamma_automatic_selection(
    inputs: GammaAutomaticProducerInputs,
    config: GammaAutomaticSelectionConfig,
    *,
    workers: int = 1,
) -> GammaAutomaticSelectionPreparation:
    """Freeze candidate-independent physics and the final PENDING identity."""

    if not isinstance(inputs, GammaAutomaticProducerInputs):
        raise TypeError("inputs must be GammaAutomaticProducerInputs")
    if not isinstance(config, GammaAutomaticSelectionConfig):
        raise TypeError("config must be GammaAutomaticSelectionConfig")
    target_workers = _strict_positive_integer(
        workers,
        field="target eigensystem workers",
    )
    expected_k_indices = tuple(range(len(inputs.k_indices)))
    if inputs.k_indices != expected_k_indices:
        raise ValueError(
            "automatic Gamma v1 requires contiguous positional k_indices "
            f"{expected_k_indices}; got {inputs.k_indices}"
        )
    if config.reference_k_index not in inputs.k_indices:
        raise ValueError("reference_k_index is not in production k_indices")
    if max(config.target_window_spec.validation_k_indices) >= len(inputs.k_indices):
        raise ValueError("target validation k indices exceed production k coverage")
    layout = GammaRowLayout.build(
        qsets=inputs.qsets,
        num_layer_list=inputs.num_layer_list,
        num_orb_per_layer_list=inputs.num_orb_per_layer_list,
        spin_convention="all",
        source_basis_hash=inputs.tapw_source_basis_hash,
    )
    if inputs.source_hamiltonians.shape[1:] != (
        layout.full_dimension,
        layout.full_dimension,
    ):
        raise ValueError("source Hamiltonian dimension does not match Gamma layout")
    if config.routing_thresholds.max_rank > layout.same_q_dimension:
        raise ValueError("routing max_rank exceeds Gamma local dimension")
    certified_actions = tuple(
        certify_gamma_raw_action(
            name=operation.name,
            full_action=operation.full_action,
            layout=layout,
            q_permutations=operation.q_permutations,
            sector_map=operation.sector_map,
            antiunitary=operation.antiunitary,
            thresholds=config.routing_thresholds,
            tapw_source_basis_hash=inputs.tapw_source_basis_hash,
        )
        for operation in inputs.operations
    )
    operation_inputs: dict[str, CandidateOperationInput] = {}
    for operation, certified_action in zip(
        inputs.operations, certified_actions, strict=True
    ):
        route_contract = gamma_certified_action_route_contract(
            certified_action,
            layout=layout,
            thresholds=config.routing_thresholds,
            full_action=operation.full_action,
        )
        route_contract["sampled_k_route"] = gamma_sampled_k_route_contract(
            inputs.kpoints,
            operation.pairs,
        )
        operation_inputs[operation.name] = CandidateOperationInput(
            name=operation.name,
            antiunitary=operation.antiunitary,
            d_full=operation.full_action,
            pairs=operation.pairs,
            route_contract=route_contract,
        )
    required_pairs = {
        operation.name: operation.pairs for operation in inputs.operations
    }
    raw_action_package_hash = candidate_raw_action_package_hash(
        operations=operation_inputs,
        presentation=inputs.presentation,
        required_pairs=required_pairs,
    )
    _certify_source_hamiltonian_covariance(
        inputs,
        threshold=config.candidate_symmetry_thresholds.heff_covariance_residual,
    )
    local_values, local_vectors = _local_eigensystems(
        inputs.source_hamiltonians,
        layout,
        hermiticity_tolerance=config.candidate_symmetry_thresholds.heff_hermiticity_residual,
    )
    target_values, target_vectors = _target_eigensystems(
        inputs.source_hamiltonians,
        workers=target_workers,
    )
    frozen_target = resolve_target_window(target_values, config.target_window_spec)
    source_hamiltonian_hash = hash_array(inputs.source_hamiltonians)

    hard_thresholds: dict[str, float] = {
        **{
            f"selection.{item.name}": float(
                getattr(config.selection_thresholds, item.name)
            )
            for item in fields(SelectionThresholds)
        },
        **{
            f"routing.{name}": float(value)
            for name, value in config.routing_thresholds.to_payload().items()
            if name != "schema"
        },
        **{
            f"candidate_symmetry.{item.name}": float(
                getattr(config.candidate_symmetry_thresholds, item.name)
            )
            for item in fields(CandidateSymmetryThresholds)
        },
    }
    policy_hash = build_selection_policy_hash(
        hard_thresholds=hard_thresholds,
        candidate_envelope_config=config.policy_payload(),
        candidate_generator_version=GAMMA_AUTO_PRODUCER_VERSION,
        candidate_schema_version=GAMMA_AUTO_CANDIDATE_SCHEMA,
        metric_schema_version=GAMMA_AUTO_METRIC_SCHEMA,
        ordering_rule_version=GAMMA_AUTO_ORDERING_RULE,
    )
    ordered_q_hash = hash_mapping(
        {
            "schema": "kp.gamma-routed-ordered-q-identity.v1",
            "ordered_q_hashes": list(layout.ordered_qset_hashes),
        }
    )
    # This identity is complete before the first candidate is materialized.
    selection_input = SelectionInputIdentity.create(
        selection_mode="auto",
        frozen_target_window_hash=hash_frozen_target_window(frozen_target),
        validation_k_indices_hash=hash_validation_k_indices(frozen_target),
        ordered_q_hash=ordered_q_hash,
        source_hamiltonian_hash=source_hamiltonian_hash,
        action_package_hash=raw_action_package_hash,
        row_layout_hash=layout.layout_hash,
        selection_policy_hash=policy_hash,
    )

    target_values.setflags(write=False)
    for by_k in (*local_values, *local_vectors):
        for value in by_k:
            value.setflags(write=False)
    for value in target_vectors:
        value.setflags(write=False)
    return GammaAutomaticSelectionPreparation(
        inputs=inputs,
        config=config,
        layout=layout,
        certified_actions=certified_actions,
        operation_inputs=MappingProxyType(dict(operation_inputs)),
        required_pairs=MappingProxyType(dict(required_pairs)),
        raw_action_package_hash=raw_action_package_hash,
        local_values=local_values,
        local_vectors=local_vectors,
        frozen_target_window=frozen_target,
        target_values=target_values,
        target_vectors=target_vectors,
        source_hamiltonian_hash=source_hamiltonian_hash,
        selection_input=selection_input,
    )


def evaluate_gamma_automatic_selection(
    preparation: GammaAutomaticSelectionPreparation,
) -> GammaAutomaticSelectionResult:
    """Materialize, certify, score, and select candidates after PENDING exists."""

    if not isinstance(preparation, GammaAutomaticSelectionPreparation):
        raise TypeError("preparation must be GammaAutomaticSelectionPreparation")
    inputs = preparation.inputs
    config = preparation.config

    evaluations: list[GammaCandidateEvaluation] = []
    rejections: list[GammaCandidateRejection] = []
    for seed in config.candidate_seed_band_indices:
        try:
            evaluations.append(
                _evaluate_candidate(
                    seed=seed,
                    inputs=inputs,
                    config=config,
                    layout=preparation.layout,
                    certified_actions=preparation.certified_actions,
                    operation_inputs=preparation.operation_inputs,
                    required_pairs=preparation.required_pairs,
                    raw_action_package_hash=preparation.raw_action_package_hash,
                    local_values=preparation.local_values,
                    local_vectors=preparation.local_vectors,
                    frozen_target=preparation.frozen_target_window,
                    target_values=preparation.target_values,
                    target_vectors=preparation.target_vectors,
                    source_hamiltonian_hash=preparation.source_hamiltonian_hash,
                )
            )
        except CandidateRejected as error:
            rejections.append(
                GammaCandidateRejection(
                    seed_band_indices=seed,
                    reason=error.reason,
                    diagnostic=str(error),
                )
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            rejections.append(
                GammaCandidateRejection(
                    seed_band_indices=seed,
                    reason=CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
                    diagnostic=str(error),
                )
            )
    if not evaluations:
        details = "; ".join(
            f"{item.seed_band_indices}: {item.reason.value}: {item.diagnostic}"
            for item in rejections
        )
        raise GammaRoutingError(
            CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
            "all automatic Gamma candidates were rejected" + (
                " (" + details + ")" if details else ""
            ),
        )
    metrics = tuple(evaluation.metrics for evaluation in evaluations)
    decision = select_projection_candidate(metrics, config.selection_thresholds)
    selected = next(
        evaluation
        for evaluation in evaluations
        if evaluation.metrics.candidate_id == decision.selected.candidate_id
    )
    if (
        gamma_routed_ordered_q_identity_hash(selected.handoff)
        != preparation.selection_input.ordered_q_hash
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "selected routed handoff Q identity differs from preselection input",
        )
    return GammaAutomaticSelectionResult(
        selection_input=preparation.selection_input,
        candidates=metrics,
        rejected_candidates=tuple(rejections),
        evaluations=tuple(evaluations),
        decision=decision,
        handoff=selected.handoff,
        frozen_target_window=preparation.frozen_target_window,
    )


def produce_gamma_automatic_selection(
    inputs: GammaAutomaticProducerInputs,
    config: GammaAutomaticSelectionConfig,
    *,
    workers: int = 1,
) -> GammaAutomaticSelectionResult:
    """Prepare and evaluate one pure automatic Gamma selection."""

    return evaluate_gamma_automatic_selection(
        prepare_gamma_automatic_selection(inputs, config, workers=workers)
    )


__all__ = [
    "GammaAutomaticProducerInputs",
    "GammaAutomaticSelectionConfig",
    "GammaAutomaticSelectionPreparation",
    "GammaAutomaticSelectionResult",
    "GammaCandidateEvaluation",
    "GammaCandidateRejection",
    "GammaDownfoldConfig",
    "GammaRawOperationSpec",
    "evaluate_gamma_automatic_selection",
    "prepare_gamma_automatic_selection",
    "produce_gamma_automatic_selection",
]
