"""Pure automatic Gamma projection production with one common anchor fibre.

The producer starts from already loaded physical arrays and explicit symmetry
metadata.  It has no CLI or filesystem side effects: each candidate selects
one reference-orbital pattern at the representative point, copies that pattern
to every Q fibre, downfolds once per k, and is symmetry-certified before the
smallest passing candidate is returned.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .basis.selection import AutoGaugeConfig
from .blocks import (
    GammaCommonAnchorFrames,
    GammaCommonAnchorSpec,
    build_gamma_common_anchor_frames,
    build_gamma_common_anchor_spec,
)
from .blocks.downfold import (
    DownfoldingOptions,
    downfold_from_projector_groups,
    downfold_from_projectors,
    projector_groups_from_block_columns,
)
from .blocks.gamma_layout import (
    GammaCertifiedRawAction,
    GammaRoutingError,
    GammaRoutingThresholds,
    GammaRowLayout,
    certify_gamma_raw_action,
    gamma_certified_action_route_contract,
)
from .identity import build_projection_basis_identity, hash_array, hash_mapping
from .low_energy_selection import (
    CandidateSelectionError,
    CandidateSelectionFailureCode,
    CandidateMetrics,
    ReferencePoint,
    SelectionDecision,
    SelectionThresholds,
    select_projection_candidate,
)
from .projection_handoff import (
    GammaCommonAnchorBasisSpec,
    gamma_sampled_k_route_contract,
)
from .projection_selection import (
    CandidateRejected,
    CandidateRejectionReason,
    FrozenTargetWindow,
    TargetWindowSpec,
    evaluate_fixed_target_window,
    resolve_target_window,
)
from .selection_artifact import (
    SelectionInputIdentity,
    build_selection_policy_hash,
    gamma_common_anchor_ordered_q_identity_hash,
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


GAMMA_AUTO_PRODUCER_VERSION = "kp.gamma-auto-common-anchor-producer.v2"
GAMMA_AUTO_CANDIDATE_SCHEMA = "kp.gamma-auto-common-anchor-candidate.v2"
GAMMA_AUTO_METRIC_SCHEMA = "kp.gamma-candidate-metrics-pre-symmetry.v2"
GAMMA_AUTO_ORDERING_RULE = "dimension-error-overlap-v2"
GAMMA_AUTO_PUBLIC_POLICY_SCHEMA = "kp.gamma-auto-common-anchor.v2"

_GAMMA_AUTO_DEFAULT_MAX_DIMENSION = 16
_GAMMA_AUTO_DEFAULT_DEGENERACY_TOLERANCE_EV = 1.0e-5
_GAMMA_AUTO_DEFAULT_VALIDATION_BANDS = 8


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
class GammaProducerIntegrityThresholds:
    source_hamiltonian_covariance_residual: float
    routed_model_heff_covariance_residual: float

    def __post_init__(self) -> None:
        source = _strict_real(
            self.source_hamiltonian_covariance_residual,
            field=(
                "producer integrity threshold "
                "source_hamiltonian_covariance_residual"
            ),
            nonnegative=True,
        )
        routed_model = _strict_real(
            self.routed_model_heff_covariance_residual,
            field=(
                "producer integrity threshold "
                "routed_model_heff_covariance_residual"
            ),
            nonnegative=True,
        )
        if routed_model <= 0.0:
            raise ValueError(
                "producer integrity threshold "
                "routed_model_heff_covariance_residual must be positive"
            )
        object.__setattr__(
            self,
            "source_hamiltonian_covariance_residual",
            source,
        )
        object.__setattr__(
            self,
            "routed_model_heff_covariance_residual",
            routed_model,
        )

    @classmethod
    def from_normalized_config(
        cls,
        value: Any,
    ) -> "GammaProducerIntegrityThresholds":
        payload = _require_exact_keys(
            value,
            {
                "source_hamiltonian_covariance_residual",
                "routed_model_heff_covariance_residual",
            },
            context="automatic Gamma producer_integrity_thresholds",
        )
        source = _strict_real(
            payload["source_hamiltonian_covariance_residual"],
            field=(
                "producer_integrity_thresholds."
                "source_hamiltonian_covariance_residual"
            ),
            nonnegative=True,
        )
        routed_model = _strict_real(
            payload["routed_model_heff_covariance_residual"],
            field=(
                "producer_integrity_thresholds."
                "routed_model_heff_covariance_residual"
            ),
            nonnegative=True,
        )
        if routed_model <= 0.0:
            raise ValueError(
                "producer_integrity_thresholds."
                "routed_model_heff_covariance_residual must be positive"
            )
        return cls(
            source_hamiltonian_covariance_residual=source,
            routed_model_heff_covariance_residual=routed_model,
        )

    def to_payload(self) -> dict[str, float]:
        return {
            "source_hamiltonian_covariance_residual": (
                self.source_hamiltonian_covariance_residual
            ),
            "routed_model_heff_covariance_residual": (
                self.routed_model_heff_covariance_residual
            ),
        }


@dataclass(frozen=True)
class GammaAutomaticSelectionConfig:
    producer_integrity_thresholds: GammaProducerIntegrityThresholds
    routing_thresholds: GammaRoutingThresholds
    selection_thresholds: SelectionThresholds
    candidate_symmetry_thresholds: CandidateSymmetryThresholds
    exactification: JointExactificationConfig
    target_window_spec: TargetWindowSpec
    candidate_seed_band_indices: tuple[tuple[int, ...], ...]
    reference_k_index: int
    downfold: GammaDownfoldConfig
    max_dimension: int = _GAMMA_AUTO_DEFAULT_MAX_DIMENSION
    degeneracy_tolerance: float = _GAMMA_AUTO_DEFAULT_DEGENERACY_TOLERANCE_EV
    generated_candidate_envelope: bool = False

    def __post_init__(self) -> None:
        max_dimension = _strict_positive_integer(
            self.max_dimension,
            field="automatic Gamma max_dimension",
        )
        degeneracy_tolerance = _strict_real(
            self.degeneracy_tolerance,
            field="automatic Gamma degeneracy_tolerance",
            nonnegative=True,
        )
        if type(self.generated_candidate_envelope) is not bool:
            raise ValueError(
                "automatic Gamma generated_candidate_envelope must be a strict bool"
            )
        if self.generated_candidate_envelope:
            if self.candidate_seed_band_indices:
                raise ValueError(
                    "public automatic Gamma policy cannot contain "
                    "candidate_seed_band_indices"
                )
            if int(self.reference_k_index) != -1:
                raise ValueError(
                    "public automatic Gamma policy cannot contain reference_k_index"
                )
        object.__setattr__(self, "max_dimension", max_dimension)
        object.__setattr__(self, "degeneracy_tolerance", degeneracy_tolerance)

    @classmethod
    def from_normalized_project_config(
        cls,
        value: Any,
    ) -> "GammaAutomaticSelectionConfig":
        """Materialize the versioned universal policy from public project input.

        Candidate bands and the reference point are deliberately absent here:
        both depend on the actual case arrays and are resolved by
        :func:`prepare_gamma_automatic_selection`.
        """

        if not isinstance(value, Mapping):
            raise ValueError("automatic Gamma project config must be a mapping")
        project = dict(value)
        selection_raw = project.get("selection")
        if not isinstance(selection_raw, Mapping):
            raise ValueError("automatic Gamma project.selection must be a mapping")
        selection_payload = dict(selection_raw)
        allowed_selection_fields = {
            "mode",
            "edge",
            "efermi",
            "max_dimension",
            "degeneracy_tolerance",
            "validation_indices",
            "validation_bands",
            "thresholds",
        }
        unknown = sorted(set(selection_payload) - allowed_selection_fields)
        if unknown:
            raise ValueError(
                "public automatic Gamma selection contains unsupported fields: "
                + ", ".join(unknown)
            )
        if selection_payload.get("mode") != "auto":
            raise ValueError("automatic Gamma selection mode must be exactly 'auto'")

        project_edge = project.get("target")
        selection_edge = selection_payload.get("edge")
        if (
            project_edge is not None
            and selection_edge is not None
            and str(project_edge).strip().lower()
            != str(selection_edge).strip().lower()
        ):
            raise ValueError("project.target conflicts with project.selection.edge")
        edge_raw = selection_edge if selection_edge is not None else project_edge
        if edge_raw is None:
            raise ValueError("automatic Gamma project.target is required")

        project_efermi = project.get("efermi")
        selection_efermi = selection_payload.get("efermi")
        if project_efermi is not None and selection_efermi is not None:
            project_efermi_value = _strict_real(
                project_efermi, field="project.efermi"
            )
            selection_efermi_value = _strict_real(
                selection_efermi, field="project.selection.efermi"
            )
            if project_efermi_value != selection_efermi_value:
                raise ValueError(
                    "project.efermi conflicts with project.selection.efermi"
                )
            energy_reference = project_efermi_value
        elif selection_efermi is not None:
            energy_reference = _strict_real(
                selection_efermi, field="project.selection.efermi"
            )
        elif project_efermi is not None:
            energy_reference = _strict_real(project_efermi, field="project.efermi")
        else:
            raise ValueError("automatic Gamma project.efermi is required")

        max_dimension = _strict_positive_integer(
            selection_payload.get(
                "max_dimension", _GAMMA_AUTO_DEFAULT_MAX_DIMENSION
            ),
            field="project.selection.max_dimension",
        )
        degeneracy_tolerance = _strict_real(
            selection_payload.get(
                "degeneracy_tolerance",
                _GAMMA_AUTO_DEFAULT_DEGENERACY_TOLERANCE_EV,
            ),
            field="project.selection.degeneracy_tolerance",
            nonnegative=True,
        )
        validation_indices = _strict_indices(
            selection_payload.get("validation_indices", (0,)),
            field="project.selection.validation_indices",
        )
        validation_bands = _strict_positive_integer(
            selection_payload.get(
                "validation_bands", _GAMMA_AUTO_DEFAULT_VALIDATION_BANDS
            ),
            field="project.selection.validation_bands",
        )

        threshold_defaults = {
            "band_rms_mev": 3.0,
            "band_max_mev": 3.0,
            "subspace_overlap": 0.05,
        }
        authored_thresholds = selection_payload.get("thresholds", {})
        if not isinstance(authored_thresholds, Mapping):
            raise ValueError("project.selection.thresholds must be a mapping")
        unknown_thresholds = sorted(
            set(authored_thresholds) - set(threshold_defaults)
        )
        if unknown_thresholds:
            raise ValueError(
                "project.selection.thresholds contains unsupported fields: "
                + ", ".join(unknown_thresholds)
            )
        selection_thresholds = SelectionThresholds(
            **{
                name: _strict_real(
                    authored_thresholds.get(name, default),
                    field=f"project.selection.thresholds.{name}",
                )
                for name, default in threshold_defaults.items()
            },
            # Symmetry is materialized after public candidate selection.  Keep
            # positive internal values only to satisfy the shared record type;
            # they are neither authored nor applied by the public selector.
            symmetry_residual=1.0,
            symmetry_leakage=1.0,
        )
        select_projection_candidate(
            (
                CandidateMetrics(
                    candidate_id="public-config-domain-probe",
                    dimension=1,
                    band_rms_mev=0.0,
                    band_max_mev=0.0,
                    subspace_overlap=1.0,
                    symmetry_residual=0.0,
                    symmetry_leakage=0.0,
                ),
            ),
            selection_thresholds,
        )

        method = str(
            project.get("downfold_method", project.get("method", "first_order"))
        ).strip().lower()
        e_ref = project.get("e_ref", project.get("E_ref"))
        downfold = GammaDownfoldConfig.from_normalized_config(
            {
                "method": method,
                "e_ref": e_ref,
                "pole_warning_mev": project.get("pole_warning_mev", 10.0),
                "pole_danger_mev": project.get("pole_danger_mev", 1.0),
                "fail_on_near_pole": project.get("fail_on_near_pole", False),
                "compute_pole_diagnostics": project.get(
                    "compute_pole_diagnostics", False
                ),
                "compute_condition_number": project.get(
                    "compute_condition_number", False
                ),
            }
        )

        routing = GammaRoutingThresholds(
            energy_same_ev=1.0e-9,
            energy_different_ev=1.0e-5,
            capture_zero_fraction=1.0e-9,
            capture_loss_max=1.0e-9,
            local_action_isometry=1.0e-8,
            off_route_leakage=1.0e-8,
            closure_residual=1.0e-8,
            route_zero_gap=1.0e-8,
            route_covariance=1.0e-8,
            projector_residual=1.0e-8,
            anchor_sigma_min=1.0e-8,
            max_rank=max_dimension,
            max_iterations=20,
        )
        candidate_symmetry = CandidateSymmetryThresholds(
            raw_h_leakage=1.0e-8,
            exactification_distance=5.0e-3,
            intertwining_residual=1.0e-8,
            heff_covariance_residual=1.0e-8,
            relation_residual=1.0e-8,
            antiunitary_square_residual=1.0e-8,
            exact_action_unitarity_residual=1.0e-8,
            projection_orthonormality_residual=1.0e-8,
            heff_hermiticity_residual=1.0e-8,
            raw_h_action_unitarity_residual=1.0e-8,
        )
        return cls(
            producer_integrity_thresholds=GammaProducerIntegrityThresholds(
                source_hamiltonian_covariance_residual=1.0e-8,
                # Retained solely because the legacy compatibility record still
                # has this field.  The common-anchor public policy does not use
                # or bind this former routed/model-frame gate.
                routed_model_heff_covariance_residual=1.0,
            ),
            routing_thresholds=routing,
            selection_thresholds=selection_thresholds,
            candidate_symmetry_thresholds=candidate_symmetry,
            exactification=JointExactificationConfig(),
            target_window_spec=TargetWindowSpec(
                edge=str(edge_raw),
                band_count=validation_bands,
                validation_k_indices=validation_indices,
                energy_reference_ev=energy_reference,
                degeneracy_tolerance_mev=degeneracy_tolerance * 1000.0,
            ),
            candidate_seed_band_indices=(),
            reference_k_index=-1,
            downfold=downfold,
            max_dimension=max_dimension,
            degeneracy_tolerance=degeneracy_tolerance,
            generated_candidate_envelope=True,
        )

    @classmethod
    def from_normalized_config(
        cls,
        value: Any,
    ) -> "GammaAutomaticSelectionConfig":
        payload = _require_exact_keys(
            value,
            {
                "mode",
                "producer_integrity_thresholds",
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
        producer_integrity = GammaProducerIntegrityThresholds.from_normalized_config(
            payload["producer_integrity_thresholds"]
        )
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
            producer_integrity_thresholds=producer_integrity,
            routing_thresholds=routing,
            selection_thresholds=selection,
            candidate_symmetry_thresholds=candidate_symmetry,
            exactification=exactification,
            target_window_spec=target_spec,
            candidate_seed_band_indices=seeds,
            reference_k_index=reference,
            downfold=GammaDownfoldConfig.from_normalized_config(payload["downfold"]),
            max_dimension=routing.max_rank,
            degeneracy_tolerance=(
                target_spec.degeneracy_tolerance_mev / 1000.0
            ),
            generated_candidate_envelope=False,
        )

    def policy_payload(self) -> dict[str, Any]:
        if self.generated_candidate_envelope:
            return {
                "schema": GAMMA_AUTO_PUBLIC_POLICY_SCHEMA,
                "candidate_envelope": {
                    "max_dimension": self.max_dimension,
                    "degeneracy_tolerance_ev": self.degeneracy_tolerance,
                    "construction": "representative-joint-spectrum-complete-clusters",
                },
                "selection_thresholds": {
                    name: getattr(self.selection_thresholds, name)
                    for name in (
                        "band_rms_mev",
                        "band_max_mev",
                        "subspace_overlap",
                    )
                },
                "symmetry": {"status": "post_selection_pending"},
                "target_window": {
                    "edge": self.target_window_spec.edge,
                    "band_count": self.target_window_spec.band_count,
                    "validation_k_indices": list(
                        self.target_window_spec.validation_k_indices
                    ),
                    "energy_reference_ev": (
                        self.target_window_spec.energy_reference_ev
                    ),
                    "degeneracy_tolerance_mev": (
                        self.target_window_spec.degeneracy_tolerance_mev
                    ),
                },
                "common_anchor_numerics": {
                    "local_action_isometry": (
                        self.routing_thresholds.local_action_isometry
                    ),
                    "off_route_leakage": self.routing_thresholds.off_route_leakage,
                    "projector_residual": self.routing_thresholds.projector_residual,
                    "anchor_sigma_min": self.routing_thresholds.anchor_sigma_min,
                },
                "exactification": {
                    item.name: getattr(self.exactification, item.name)
                    for item in fields(JointExactificationConfig)
                },
                "downfold": self.downfold.to_payload(),
            }
        return {
            "schema": "kp.gamma-auto-policy-payload.v2",
            "producer_integrity_thresholds": (
                self.producer_integrity_thresholds.to_payload()
            ),
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
    external_target_band_spectra: np.ndarray | None = None

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
        target_spectra: np.ndarray | None
        if self.external_target_band_spectra is None:
            target_spectra = None
        else:
            target_spectra = np.array(
                self.external_target_band_spectra,
                dtype=np.float64,
                copy=True,
                order="C",
            )
            if (
                target_spectra.ndim != 2
                or target_spectra.shape[0] != len(k_indices)
                or target_spectra.shape[1] <= 0
                or not np.all(np.isfinite(target_spectra))
            ):
                raise ValueError(
                    "external_target_band_spectra must be finite (Nk,Nband)"
                )
            if np.any(np.diff(target_spectra, axis=1) < 0.0):
                raise ValueError(
                    "external_target_band_spectra must be sorted in ascending energy"
                )
            target_spectra.setflags(write=False)
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
        object.__setattr__(self, "external_target_band_spectra", target_spectra)


@dataclass(frozen=True)
class GammaCandidateRejection:
    seed_band_indices: tuple[int, ...]
    reason: CandidateRejectionReason
    diagnostic: str


@dataclass(frozen=True)
class GammaCandidateEvaluation:
    metrics: CandidateMetrics
    handoff: GammaCommonAnchorBasisSpec
    symmetry_certificate: CandidateSymmetryCertificate | None


@dataclass(frozen=True)
class GammaAutomaticSelectionResult:
    selection_input: SelectionInputIdentity
    candidates: tuple[CandidateMetrics, ...]
    rejected_candidates: tuple[GammaCandidateRejection, ...]
    evaluations: tuple[GammaCandidateEvaluation, ...]
    decision: SelectionDecision
    handoff: GammaCommonAnchorBasisSpec
    frozen_target_window: FrozenTargetWindow


@dataclass(frozen=True)
class GammaAutomaticSelectionPreparation:
    """Candidate-independent, identity-complete automatic Gamma inputs."""

    inputs: GammaAutomaticProducerInputs
    config: GammaAutomaticSelectionConfig
    reference_point: ReferencePoint
    candidate_seed_band_indices: tuple[tuple[int, ...], ...]
    effective_max_dimension: int
    layout: GammaRowLayout
    common_anchor_gauge_config: AutoGaugeConfig
    certified_actions: tuple[GammaCertifiedRawAction, ...]
    operation_inputs: Mapping[str, CandidateOperationInput]
    required_pairs: Mapping[str, tuple[tuple[int, int], ...]]
    raw_action_package_hash: str
    local_values: tuple[tuple[np.ndarray, ...], ...]
    local_vectors: tuple[tuple[np.ndarray, ...], ...]
    frozen_target_window: FrozenTargetWindow
    target_values: np.ndarray
    source_hamiltonian_hash: str
    selection_input: SelectionInputIdentity


def _local_eigensystems(
    hamiltonians: np.ndarray,
    layout: GammaRowLayout,
    *,
    hermiticity_tolerance: float,
    workers: int = 1,
) -> tuple[tuple[tuple[np.ndarray, ...], ...], tuple[tuple[np.ndarray, ...], ...]]:
    worker_count = min(
        _strict_positive_integer(workers, field="local Gamma eigensystem workers"),
        len(hamiltonians),
    )

    def solve_k(
        item: tuple[int, np.ndarray],
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        k_position, hamiltonian = item
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
        return tuple(values_by_q), tuple(vectors_by_q)

    indexed = tuple(enumerate(hamiltonians))
    if worker_count == 1:
        solved = tuple(solve_k(item) for item in indexed)
    else:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                solved = tuple(executor.map(solve_k, indexed))
    return (
        tuple(values for values, _vectors in solved),
        tuple(vectors for _values, vectors in solved),
    )


def _geometric_reference_q_index(qset: np.ndarray) -> int:
    """Choose the unique ordered-Q entry nearest the Gamma origin."""

    coordinates = np.asarray(qset, dtype=np.float64)
    if coordinates.ndim != 2 or not np.all(np.isfinite(coordinates)):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma common-anchor reference Q coordinates must be finite",
        )
    squared_norms = np.einsum("qi,qi->q", coordinates, coordinates)
    minimum = float(np.min(squared_norms))
    scale = max(1.0, float(np.max(squared_norms)))
    matches = np.flatnonzero(
        np.isclose(squared_norms, minimum, rtol=1.0e-12, atol=1.0e-14 * scale)
    )
    if matches.size != 1:
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "automatic Gamma requires a unique geometric reference Q; "
            "the minimum-norm Q must be unique",
        )
    return int(matches[0])


def _resolved_reference_point(
    inputs: GammaAutomaticProducerInputs,
    config: GammaAutomaticSelectionConfig,
    *,
    reference_q_index: int,
) -> ReferencePoint:
    if config.generated_candidate_envelope:
        k_position = int(np.argmin(np.linalg.norm(inputs.kpoints, axis=1)))
        k_index = int(inputs.k_indices[k_position])
    else:
        if config.reference_k_index not in inputs.k_indices:
            raise ValueError("reference_k_index is not in production k_indices")
        k_index = int(config.reference_k_index)
        k_position = inputs.k_indices.index(k_index)
    q_index = int(reference_q_index)
    return ReferencePoint(
        k_index=k_index,
        k_coordinate=tuple(float(value) for value in inputs.kpoints[k_position]),
        q_indices=(q_index, q_index),
        q_vectors=tuple(
            tuple(float(value) for value in qset[q_index])
            for qset in inputs.qsets
        ),
    )


def _generated_candidate_seed_envelope(
    values: np.ndarray,
    config: GammaAutomaticSelectionConfig,
    *,
    max_dimension: int,
) -> tuple[tuple[int, ...], ...]:
    """Accumulate complete edge clusters at the actual representative fibre."""

    eigenvalues = np.asarray(values, dtype=np.float64)
    if eigenvalues.ndim != 1 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("representative Gamma spectrum must be one finite vector")
    edge = config.target_window_spec.edge
    energy_reference = float(config.target_window_spec.energy_reference_ev)
    if edge == "valence":
        eligible = np.flatnonzero(eigenvalues <= energy_reference)
        ordered = tuple(
            sorted(eligible.tolist(), key=lambda index: (-eigenvalues[index], index))
        )
    else:
        eligible = np.flatnonzero(eigenvalues >= energy_reference)
        ordered = tuple(
            sorted(eligible.tolist(), key=lambda index: (eigenvalues[index], index))
        )

    clusters: list[list[int]] = []
    for band_index in ordered:
        if not clusters:
            clusters.append([int(band_index)])
            continue
        previous = clusters[-1][-1]
        if (
            abs(float(eigenvalues[band_index] - eigenvalues[previous]))
            <= config.degeneracy_tolerance
        ):
            clusters[-1].append(int(band_index))
        else:
            clusters.append([int(band_index)])

    candidates: list[tuple[int, ...]] = []
    accumulated: list[int] = []
    for cluster in clusters:
        if len(accumulated) + len(cluster) > int(max_dimension):
            break
        accumulated.extend(cluster)
        candidates.append(tuple(sorted(accumulated)))
    if not candidates:
        raise CandidateSelectionError(
            CandidateSelectionFailureCode.NO_CANDIDATES,
            "automatic Gamma generated no complete candidate cluster within "
            f"max_dimension={int(max_dimension)}",
        )
    return tuple(candidates)


def _common_anchor_gauge_config(
    *,
    reference_q_index: int,
    thresholds: GammaRoutingThresholds,
) -> AutoGaugeConfig:
    min_sigma = float(thresholds.anchor_sigma_min)
    return AutoGaugeConfig(
        min_sigma=min_sigma,
        max_condition=max(1.0, 1.0 / min_sigma),
        reference_q_index=int(reference_q_index),
    )


def _certify_candidate_seed_envelope(
    *,
    seed: Sequence[int],
    local_values: Sequence[Sequence[np.ndarray]],
    max_rank: int,
    degeneracy_tolerance_mev: float,
    k_indices: Sequence[int],
) -> tuple[int, ...]:
    """Reject oversized seeds and cuts through local energy clusters."""

    joint = tuple(int(index) for index in seed)
    if len(joint) > int(max_rank):
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            f"Gamma candidate rank {len(joint)} exceeds max_rank {int(max_rank)}",
        )
    selected = set(joint)
    tolerance_ev = float(degeneracy_tolerance_mev) / 1000.0
    for position, values_by_q in enumerate(local_values):
        for q_index, raw_values in enumerate(values_by_q):
            values = np.asarray(raw_values, dtype=np.float64)
            if any(index < 0 or index >= values.size for index in joint):
                raise GammaRoutingError(
                    CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
                    "Gamma candidate seed lies outside a local eigensystem",
                )
            for boundary, gap in enumerate(np.diff(values), start=1):
                if float(gap) > tolerance_ev:
                    continue
                if ((boundary - 1) in selected) == (boundary in selected):
                    continue
                raise GammaRoutingError(
                    CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                    "Gamma candidate cuts a degenerate local energy cluster at "
                    f"k={int(k_indices[position])}, q={q_index}, bands "
                    f"{boundary - 1}/{boundary} (gap={float(gap) * 1000.0:.3e} meV)",
                )
    return joint


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


def _canonical_common_candidate_id(
    *,
    joint_band_indices: Sequence[int],
    anchor_spec: GammaCommonAnchorSpec,
    layout: GammaRowLayout,
) -> str:
    identity = hash_mapping(
        {
            "schema": GAMMA_AUTO_CANDIDATE_SCHEMA,
            "joint_band_indices": list(joint_band_indices),
            "anchor_spec_identity_hash": anchor_spec.identity_hash,
            "layout_hash": layout.layout_hash,
        }
    )
    return "gamma-common-anchor-" + identity[:24]


def _model_columns_for_q(
    *,
    q_index: int,
    q_count: int,
    model_group_ranks: Sequence[int],
) -> np.ndarray:
    """Return the established sector-orbital-Q model column order."""

    return np.asarray(
        [
            q_count * sum(model_group_ranks[:group]) + orbital * q_count + q_index
            for group, rank in enumerate(model_group_ranks)
            for orbital in range(int(rank))
        ],
        dtype=np.intp,
    )


def _assemble_common_anchor_frame(
    frames: GammaCommonAnchorFrames,
    *,
    layout: GammaRowLayout,
) -> np.ndarray:
    """Embed compact common-anchor fibres in the full TAPW row layout."""

    if frames.layout_hash != layout.layout_hash:
        raise ValueError("Gamma common-anchor frames do not match the row layout")
    ranks = tuple(int(rank) for rank in frames.model_group_ranks)
    local_rank = sum(ranks)
    if len(frames.local_frames_by_q) != layout.q_count:
        raise ValueError("Gamma common-anchor frames do not cover every ordered Q")
    assembled = np.zeros(
        (layout.full_dimension, layout.q_count * local_rank),
        dtype=np.complex128,
    )
    for q_index, raw_frame in enumerate(frames.local_frames_by_q):
        frame = np.asarray(raw_frame, dtype=np.complex128)
        if frame.shape != (layout.same_q_dimension, local_rank):
            raise ValueError("Gamma common-anchor local frame shape is invalid")
        rows = layout.same_q_full_rows(q_index)
        columns = _model_columns_for_q(
            q_index=q_index,
            q_count=layout.q_count,
            model_group_ranks=ranks,
        )
        assembled[np.ix_(rows, columns)] = frame
    return assembled


def _common_anchor_projector_groups(
    frames: GammaCommonAnchorFrames,
    *,
    layout: GammaRowLayout,
    eigenvectors_by_q: Sequence[np.ndarray],
) -> tuple[Any, Any]:
    """Build block-sparse low/high groups without a dense global U_high."""

    ranks = tuple(int(rank) for rank in frames.model_group_ranks)
    local_rank = len(frames.joint_band_indices)
    selected_indices = set(frames.joint_band_indices)
    high_indices = np.asarray(
        [
            index
            for index in range(layout.same_q_dimension)
            if index not in selected_indices
        ],
        dtype=np.intp,
    )
    high_rank = int(high_indices.size)
    block_rows: list[np.ndarray] = []
    low_columns: list[np.ndarray] = []
    high_columns: list[np.ndarray] = []
    low_global_columns: list[np.ndarray] = []
    high_global_columns: list[np.ndarray] = []
    for q_index, (raw_low, raw_vectors) in enumerate(
        zip(frames.local_frames_by_q, eigenvectors_by_q, strict=True)
    ):
        block_rows.append(layout.same_q_full_rows(q_index))
        low_columns.append(np.asarray(raw_low, dtype=np.complex128))
        high_columns.append(
            np.asarray(raw_vectors, dtype=np.complex128)[:, high_indices]
        )
        low_global_columns.append(
            _model_columns_for_q(
                q_index=q_index,
                q_count=layout.q_count,
                model_group_ranks=ranks,
            )
        )
        high_global_columns.append(
            np.arange(
                q_index * high_rank,
                (q_index + 1) * high_rank,
                dtype=np.intp,
            )
        )
    return (
        projector_groups_from_block_columns(
            block_rows,
            low_columns,
            low_global_columns,
            n_columns=layout.q_count * local_rank,
        ),
        projector_groups_from_block_columns(
            block_rows,
            high_columns,
            high_global_columns,
            n_columns=layout.q_count * high_rank,
        ),
    )


def _freeze_validation_target_window(
    target_values: np.ndarray,
    spec: TargetWindowSpec,
) -> FrozenTargetWindow:
    """Resolve compact validation spectra while retaining physical k indices."""

    compact_spec = TargetWindowSpec(
        edge=spec.edge,
        band_count=spec.band_count,
        validation_k_indices=tuple(range(len(spec.validation_k_indices))),
        energy_reference_ev=spec.energy_reference_ev,
        degeneracy_tolerance_mev=spec.degeneracy_tolerance_mev,
    )
    compact = resolve_target_window(target_values, compact_spec)
    return FrozenTargetWindow(
        spec=spec,
        target_band_ids=compact.target_band_ids,
        target_energies_ev=compact.target_energies_ev,
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
    certified_actions: Sequence[GammaCertifiedRawAction],
    layout: GammaRowLayout,
    model_group_ranks: Sequence[int],
    presentation: MagneticPresentation,
    config: JointExactificationConfig,
    sigma_minimum: float,
    off_route_tolerance: float,
    uniformity_tolerance: float,
) -> dict[str, np.ndarray]:
    certified_by_name = {action.name: action for action in certified_actions}
    generator_names = {generator.name for generator in presentation.generators}
    if set(certified_by_name) != generator_names or set(operations) != generator_names:
        raise GammaRoutingError(
            CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
            "Gamma common-anchor symmetry packages disagree",
        )
    local_rank = sum(int(rank) for rank in model_group_ranks)
    model_dimension = layout.q_count * local_rank
    all_model_rows = np.arange(model_dimension, dtype=np.intp)
    internal_actions: dict[str, BlockRouteAction] = {}
    for generator in presentation.generators:
        operation = operations[generator.name]
        certified = certified_by_name[generator.name]
        internal_blocks: list[np.ndarray] = []
        for target_k, source_k in operation.pairs:
            target = states[target_k]
            source = states[source_k]
            projected = evaluate_projected_pair(
                d_full=operation.d_full,
                target_u_low=target.u_low,
                source_u_low=source.u_low,
                target_heff=None,
                source_heff=None,
                antiunitary=operation.antiunitary,
                compute_heff_covariance=False,
            ).projected_action
            for source_q, target_q in enumerate(certified.q_permutation):
                source_columns = _model_columns_for_q(
                    q_index=source_q,
                    q_count=layout.q_count,
                    model_group_ranks=model_group_ranks,
                )
                target_columns = _model_columns_for_q(
                    q_index=target_q,
                    q_count=layout.q_count,
                    model_group_ranks=model_group_ranks,
                )
                outside_target = np.setdiff1d(
                    all_model_rows,
                    target_columns,
                    assume_unique=True,
                )
                leakage = float(
                    np.linalg.norm(
                        projected[np.ix_(outside_target, source_columns)],
                        ord="fro",
                    )
                    / np.sqrt(max(1, local_rank))
                )
                if not np.isfinite(leakage) or leakage > off_route_tolerance:
                    raise GammaRoutingError(
                        CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
                        f"Gamma common-anchor operation {generator.name} has "
                        f"off-route model-Q leakage {leakage:.3e}",
                    )
                internal_blocks.append(
                    np.asarray(
                        projected[np.ix_(target_columns, source_columns)],
                        dtype=np.complex128,
                    )
                )
        mean_internal = np.mean(np.stack(internal_blocks, axis=0), axis=0)
        uniformity_residual = max(
            float(
                np.linalg.norm(block - mean_internal, ord="fro")
                / np.sqrt(max(1, local_rank))
            )
            for block in internal_blocks
        )
        if uniformity_residual > uniformity_tolerance:
            raise GammaRoutingError(
                CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
                f"Gamma common-anchor operation {generator.name} lacks a "
                "Q-independent internal action: residual "
                f"{uniformity_residual:.3e} exceeds {uniformity_tolerance:.3e}",
            )
        unitary = _polar_unitary(
            mean_internal,
            sigma_minimum=sigma_minimum,
        )
        internal_actions[generator.name] = BlockRouteAction(
            name=generator.name,
            antiunitary=generator.antiunitary,
            fiber_permutation=(0,),
            fiber_dimensions=(local_rank,),
            route_blocks=(unitary,),
        )
    try:
        certify_joint_block_actions(internal_actions, presentation)
        exact_internal_actions = internal_actions
    except JointExactificationError:
        exact_internal_actions = dict(
            joint_exactify_block_actions(
                internal_actions,
                presentation,
                config=config,
            ).actions
        )
    materialized: dict[str, np.ndarray] = {}
    for name, internal_action in exact_internal_actions.items():
        rho = materialize_block_route_action(internal_action)
        full_action = np.zeros(
            (model_dimension, model_dimension),
            dtype=np.complex128,
        )
        for source_q, target_q in enumerate(
            certified_by_name[name].q_permutation
        ):
            source_columns = _model_columns_for_q(
                q_index=source_q,
                q_count=layout.q_count,
                model_group_ranks=model_group_ranks,
            )
            target_columns = _model_columns_for_q(
                q_index=target_q,
                q_count=layout.q_count,
                model_group_ranks=model_group_ranks,
            )
            full_action[np.ix_(target_columns, source_columns)] = rho
        materialized[name] = full_action
    return materialized


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


def _relative_frobenius_residual(lhs: Any, rhs: Any) -> float:
    left = np.asarray(lhs, dtype=np.complex128)
    right = np.asarray(rhs, dtype=np.complex128)
    denominator = float(np.linalg.norm(left, ord="fro"))
    if denominator == 0.0:
        denominator = 1.0
    return float(np.linalg.norm(left - right, ord="fro") / denominator)


def _require_residual_gate(
    residuals: Mapping[str, float],
    *,
    tolerance: float,
    context: str,
) -> None:
    if not residuals:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"{context}: no residual evidence was supplied",
        )
    nonfinite = next(
        (
            (name, float(value))
            for name, value in residuals.items()
            if not np.isfinite(value)
        ),
        None,
    )
    if nonfinite is not None:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"{context}: {nonfinite[0]}={nonfinite[1]:.3e} is nonfinite",
        )
    worst_name, worst_value = max(
        residuals.items(), key=lambda item: float(item[1])
    )
    if float(worst_value) > tolerance:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"{context}: {worst_name}={float(worst_value):.3e} "
            f"exceeds {tolerance:.3e}",
        )


def _reference_candidate_metrics(
    preparation: GammaAutomaticSelectionPreparation,
    seed: tuple[int, ...],
) -> CandidateMetrics:
    """Score one branch at the representative k row without a handoff build."""

    inputs = preparation.inputs
    config = preparation.config
    layout = preparation.layout
    reference_position = inputs.k_indices.index(preparation.reference_point.k_index)
    reference_q = preparation.common_anchor_gauge_config.reference_q_index
    reference_values = preparation.local_values[reference_position][reference_q]
    joint = _certify_candidate_seed_envelope(
        seed=seed,
        local_values=((reference_values,),),
        max_rank=preparation.effective_max_dimension,
        degeneracy_tolerance_mev=(
            config.target_window_spec.degeneracy_tolerance_mev
        ),
        k_indices=(preparation.reference_point.k_index,),
    )
    anchor_spec = build_gamma_common_anchor_spec(
        reference_eigenvalues_by_q=preparation.local_values[reference_position],
        reference_eigenvectors_by_q=preparation.local_vectors[reference_position],
        joint_band_indices=joint,
        layout=layout,
        gauge_config=preparation.common_anchor_gauge_config,
        projector_tolerance=config.routing_thresholds.projector_residual,
        orthonormality_tolerance=config.routing_thresholds.projector_residual,
    )
    frames = build_gamma_common_anchor_frames(
        eigenvalues_by_q=preparation.local_values[reference_position],
        eigenvectors_by_q=preparation.local_vectors[reference_position],
        layout=layout,
        anchor_spec=anchor_spec,
    )
    u_low = _assemble_common_anchor_frame(frames, layout=layout)
    if config.downfold.options.method == "first_order":
        downfolded = downfold_from_projectors(
            inputs.source_hamiltonians[reference_position],
            u_low,
            None,
            config.downfold.options,
        )
    else:
        low_groups, high_groups = _common_anchor_projector_groups(
            frames,
            layout=layout,
            eigenvectors_by_q=preparation.local_vectors[reference_position],
        )
        downfolded = downfold_from_projector_groups(
            inputs.source_hamiltonians[reference_position],
            low_groups,
            high_groups,
            config.downfold.options,
        )
    heff = np.asarray(downfolded.heff, dtype=np.complex128)
    _require_residual_gate(
        {
            "reference_heff": max(
                float(downfolded.hermiticity_residual),
                _relative_frobenius_residual(heff, heff.conj().T),
            )
        },
        tolerance=float(
            config.candidate_symmetry_thresholds.heff_hermiticity_residual
        ),
        context="Gamma representative-point downfold Hermiticity gate failed",
    )
    if inputs.external_target_band_spectra is None:
        target_row = np.sort(
            np.concatenate(preparation.local_values[reference_position])
        )
    else:
        target_row = inputs.external_target_band_spectra[reference_position]
    reference_target_spec = replace(
        config.target_window_spec,
        validation_k_indices=(0,),
    )
    reference_target = resolve_target_window(
        np.asarray(target_row, dtype=np.float64)[np.newaxis, :],
        reference_target_spec,
    )
    band_metrics = evaluate_fixed_target_window(
        reference_target,
        np.linalg.eigvalsh(heff)[np.newaxis, :],
    )
    coverage_sigma_min = float(
        min(frames.alignment_singular_values_by_q[reference_q])
    )
    return CandidateMetrics(
        candidate_id=_canonical_common_candidate_id(
            joint_band_indices=joint,
            anchor_spec=anchor_spec,
            layout=layout,
        ),
        dimension=int(heff.shape[-1]),
        band_rms_mev=band_metrics.rms_error_mev,
        band_max_mev=band_metrics.maximum_abs_error_mev,
        subspace_overlap=float(
            np.clip(coverage_sigma_min * coverage_sigma_min, 0.0, 1.0)
        ),
        symmetry_residual=None,
        symmetry_leakage=None,
    )


def _select_public_reference_seed(
    preparation: GammaAutomaticSelectionPreparation,
) -> tuple[int, ...]:
    """Return the first complete representative branch passing local metrics."""

    attempted_ids: list[str] = []
    structural_failures: list[tuple[str, str]] = []
    for seed in preparation.candidate_seed_band_indices:
        try:
            metrics = _reference_candidate_metrics(preparation, seed)
        except CandidateRejected as error:
            label = "bands-" + "-".join(str(index) for index in seed)
            structural_failures.append((label, str(error)))
            continue
        attempted_ids.append(metrics.candidate_id)
        try:
            select_projection_candidate(
                (metrics,),
                preparation.config.selection_thresholds,
                allow_pending_symmetry=True,
            )
        except CandidateSelectionError as error:
            if error.failure_code is CandidateSelectionFailureCode.HARD_METRIC_FAILED:
                continue
            raise
        return tuple(seed)
    raise CandidateSelectionError(
        CandidateSelectionFailureCode.HARD_METRIC_FAILED,
        "no representative-point Gamma branch passed the local projection metrics",
        candidate_ids=tuple(attempted_ids),
        structural_failures=tuple(structural_failures),
    )


def _evaluate_candidate(
    *,
    seed: tuple[int, ...],
    inputs: GammaAutomaticProducerInputs,
    config: GammaAutomaticSelectionConfig,
    layout: GammaRowLayout,
    common_anchor_gauge_config: AutoGaugeConfig,
    certified_actions: tuple[GammaCertifiedRawAction, ...],
    operation_inputs: Mapping[str, CandidateOperationInput],
    required_pairs: Mapping[str, tuple[tuple[int, int], ...]],
    raw_action_package_hash: str,
    local_values: tuple[tuple[np.ndarray, ...], ...],
    local_vectors: tuple[tuple[np.ndarray, ...], ...],
    frozen_target: FrozenTargetWindow,
    source_hamiltonian_hash: str,
    reference_k_index: int,
    candidate_max_rank: int,
) -> GammaCandidateEvaluation:
    """Materialize one common-anchor candidate without eigenstate routing."""

    reference_position = inputs.k_indices.index(int(reference_k_index))
    if config.generated_candidate_envelope:
        reference_q = common_anchor_gauge_config.reference_q_index
        certification_values = ((local_values[reference_position][reference_q],),)
        certification_k_indices = (int(reference_k_index),)
    else:
        certification_values = local_values
        certification_k_indices = inputs.k_indices
    joint = _certify_candidate_seed_envelope(
        seed=seed,
        local_values=certification_values,
        max_rank=candidate_max_rank,
        degeneracy_tolerance_mev=(
            config.target_window_spec.degeneracy_tolerance_mev
        ),
        k_indices=certification_k_indices,
    )
    anchor_spec = build_gamma_common_anchor_spec(
        reference_eigenvalues_by_q=local_values[reference_position],
        reference_eigenvectors_by_q=local_vectors[reference_position],
        joint_band_indices=joint,
        layout=layout,
        gauge_config=common_anchor_gauge_config,
        projector_tolerance=config.routing_thresholds.projector_residual,
        orthonormality_tolerance=config.routing_thresholds.projector_residual,
    )
    candidate_id = _canonical_common_candidate_id(
        joint_band_indices=joint,
        anchor_spec=anchor_spec,
        layout=layout,
    )
    common_frames = tuple(
        build_gamma_common_anchor_frames(
            eigenvalues_by_q=local_values[position],
            eigenvectors_by_q=local_vectors[position],
            layout=layout,
            anchor_spec=anchor_spec,
        )
        for position in range(len(inputs.k_indices))
    )

    u_low_by_position: list[np.ndarray] = []
    authoritative_heff: list[np.ndarray] = []
    include_high = config.downfold.options.method != "first_order"
    for position, frames in enumerate(common_frames):
        u_low = _assemble_common_anchor_frame(frames, layout=layout)
        if include_high:
            low_groups, high_groups = _common_anchor_projector_groups(
                frames,
                layout=layout,
                eigenvectors_by_q=local_vectors[position],
            )
            result = downfold_from_projector_groups(
                inputs.source_hamiltonians[position],
                low_groups,
                high_groups,
                config.downfold.options,
            )
        else:
            result = downfold_from_projectors(
                inputs.source_hamiltonians[position],
                u_low,
                None,
                config.downfold.options,
            )
        heff = np.asarray(result.heff, dtype=np.complex128)
        if not np.all(np.isfinite(heff)):
            raise ValueError("automatic Gamma common-anchor downfold produced nonfinite Heff")
        _require_residual_gate(
            {
                "common_anchor_heff": max(
                    float(result.hermiticity_residual),
                    _relative_frobenius_residual(heff, heff.conj().T),
                )
            },
            tolerance=float(
                config.candidate_symmetry_thresholds.heff_hermiticity_residual
            ),
            context=(
                "Gamma common-anchor downfold Hermiticity gate failed at "
                f"k={inputs.k_indices[position]}"
            ),
        )
        u_low_by_position.append(u_low)
        authoritative_heff.append(heff)

    heff_tensor = np.stack(authoritative_heff, axis=0)
    certificate: CandidateSymmetryCertificate | None = None
    if not config.generated_candidate_envelope:
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
            certified_actions=certified_actions,
            layout=layout,
            model_group_ranks=anchor_spec.model_group_ranks,
            presentation=inputs.presentation,
            config=config.exactification,
            sigma_minimum=config.routing_thresholds.anchor_sigma_min,
            off_route_tolerance=config.routing_thresholds.off_route_leakage,
            uniformity_tolerance=(
                config.candidate_symmetry_thresholds.exactification_distance
            ),
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
        if (
            candidate_action_package_hash(certificate.input_identity_payload)
            != raw_action_package_hash
        ):
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "candidate certificate raw-action package differs from preselection input",
            )

    if certificate is None:
        pending_symmetry_input_hash = hash_mapping(
            {
                "schema": "kp.gamma-post-selection-symmetry-input.v1",
                "candidate_id": candidate_id,
                "raw_action_package_hash": raw_action_package_hash,
            }
        )
        pending_symmetry_certificate_hash = hash_mapping(
            {
                "schema": "kp.gamma-post-selection-symmetry-pending.v1",
                "input_identity_hash": pending_symmetry_input_hash,
                "status": "pending",
            }
        )
    else:
        pending_symmetry_input_hash = certificate.input_identity_hash
        pending_symmetry_certificate_hash = certificate.certificate_hash

    local_frame_tensor = np.stack(
        [np.stack(frames.local_frames_by_q, axis=0) for frames in common_frames],
        axis=0,
    )
    model_dim = int(heff_tensor.shape[-1])
    anchor_payload = anchor_spec.to_payload()
    artifact_identity = build_projection_basis_identity(
        qset1=inputs.qsets[0],
        qset2=inputs.qsets[1],
        source_hamiltonian_hash=source_hamiltonian_hash,
        spin="all",
        mode="Gamma",
        energy_scale=1.0,
        nlow_state_list={
            "projection_basis_kind": "gamma_common_anchor",
            "joint_band_indices": list(joint),
            "owner_specs": anchor_payload["owner_specs"],
        },
        resolved_norb_fix_list=anchor_payload["resolved_reference_terms"],
        gauge_mode="auto_common_anchor",
        num_layer_list=list(layout.num_layer_list),
        num_orb_per_layer_list=[
            list(group) for group in inputs.num_orb_per_layer_list
        ],
        orbital_block_dim=layout.same_q_dimension,
        model_dim=model_dim,
        k_indices=inputs.k_indices,
        gauge_frame_hash=hash_array(local_frame_tensor),
    )
    handoff = GammaCommonAnchorBasisSpec.create(
        candidate_id=candidate_id,
        artifact_identity=artifact_identity,
        layout=layout,
        anchor_spec=anchor_spec,
        k_indices=inputs.k_indices,
        local_frames=local_frame_tensor,
        authoritative_heff=heff_tensor,
        kpoints=inputs.kpoints,
        source_hamiltonian_hash=source_hamiltonian_hash,
        action_package_hash=raw_action_package_hash,
        candidate_certificate_hash=pending_symmetry_certificate_hash,
        candidate_input_identity_hash=pending_symmetry_input_hash,
    )

    candidate_eigenvalues: list[np.ndarray] = []
    for heff in heff_tensor:
        values = np.linalg.eigvalsh(heff)
        candidate_eigenvalues.append(values)
    band_metrics = evaluate_fixed_target_window(
        frozen_target,
        np.stack(candidate_eigenvalues, axis=0),
    )
    # Candidate selection is defined at the same representative fibre that
    # fixes the common-anchor pattern.  Other (k, Q) fibres still pass through
    # ``build_gamma_common_anchor_frames``, whose ``min_sigma`` gate certifies
    # numerical full rank, but their conditioning is diagnostic rather than a
    # second physical candidate-selection rule.
    reference_frames = common_frames[reference_position]
    coverage_sigma_min = float(
        min(
            reference_frames.alignment_singular_values_by_q[
                common_anchor_gauge_config.reference_q_index
            ]
        )
    )
    common_anchor_coverage = float(
        np.clip(coverage_sigma_min * coverage_sigma_min, 0.0, 1.0)
    )
    symmetry_residual: float | None
    symmetry_leakage: float | None
    if certificate is None:
        symmetry_residual = None
        symmetry_leakage = None
    else:
        symmetry_residual, symmetry_leakage = _certificate_metrics(certificate)
    metrics = CandidateMetrics(
        candidate_id=candidate_id,
        dimension=model_dim,
        band_rms_mev=band_metrics.rms_error_mev,
        band_max_mev=band_metrics.maximum_abs_error_mev,
        subspace_overlap=common_anchor_coverage,
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
    _strict_positive_integer(
        workers,
        field="automatic Gamma producer workers",
    )
    expected_k_indices = tuple(range(len(inputs.k_indices)))
    if inputs.k_indices != expected_k_indices:
        raise ValueError(
            "automatic Gamma v1 requires contiguous positional k_indices "
            f"{expected_k_indices}; got {inputs.k_indices}"
        )
    if (
        not config.generated_candidate_envelope
        and config.reference_k_index not in inputs.k_indices
    ):
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
    if config.generated_candidate_envelope:
        effective_max_dimension = min(
            config.max_dimension,
            layout.same_q_dimension,
        )
        effective_routing_thresholds = replace(
            config.routing_thresholds,
            max_rank=effective_max_dimension,
        )
    else:
        effective_max_dimension = config.routing_thresholds.max_rank
        effective_routing_thresholds = config.routing_thresholds
    if not np.allclose(
        inputs.qsets[0],
        inputs.qsets[1],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "automatic Gamma common-anchor fibres require the two ordered Q sets "
            "to match coordinate-by-coordinate",
        )
    try:
        reference_q_index = _geometric_reference_q_index(inputs.qsets[0])
    except GammaRoutingError as error:
        if config.generated_candidate_envelope:
            raise ValueError(str(error)) from error
        raise
    reference_point = _resolved_reference_point(
        inputs,
        config,
        reference_q_index=reference_q_index,
    )
    common_anchor_gauge = _common_anchor_gauge_config(
        reference_q_index=reference_q_index,
        thresholds=effective_routing_thresholds,
    )
    if inputs.source_hamiltonians.shape[1:] != (
        layout.full_dimension,
        layout.full_dimension,
    ):
        raise ValueError("source Hamiltonian dimension does not match Gamma layout")
    if (
        not config.generated_candidate_envelope
        and effective_max_dimension > layout.same_q_dimension
    ):
        raise ValueError("routing max_rank exceeds Gamma local dimension")
    certified_actions = tuple(
        certify_gamma_raw_action(
            name=operation.name,
            full_action=operation.full_action,
            layout=layout,
            q_permutations=operation.q_permutations,
            sector_map=operation.sector_map,
            antiunitary=operation.antiunitary,
            thresholds=effective_routing_thresholds,
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
            thresholds=effective_routing_thresholds,
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
    # The public common-anchor path certifies the symmetry on each materialized
    # low-energy candidate below.  A full-source covariance gate is both
    # redundant and physically too strong: harmless high-energy TAPW noise must
    # not veto an otherwise covariant low-energy model.  Keep the legacy gate
    # only for callers that explicitly supplied the legacy producer policy.
    if not config.generated_candidate_envelope:
        _certify_source_hamiltonian_covariance(
            inputs,
            threshold=(
                config.producer_integrity_thresholds
                .source_hamiltonian_covariance_residual
            ),
        )
    local_values, local_vectors = _local_eigensystems(
        inputs.source_hamiltonians,
        layout,
        hermiticity_tolerance=config.candidate_symmetry_thresholds.heff_hermiticity_residual,
        workers=workers,
    )
    if config.generated_candidate_envelope:
        reference_position = inputs.k_indices.index(reference_point.k_index)
        candidate_seed_band_indices = _generated_candidate_seed_envelope(
            local_values[reference_position][reference_q_index],
            config,
            max_dimension=effective_max_dimension,
        )
    else:
        candidate_seed_band_indices = config.candidate_seed_band_indices
    validation_positions = config.target_window_spec.validation_k_indices
    if inputs.external_target_band_spectra is None:
        # Compatibility for small pure-array callers that predate the runtime
        # band-file handoff.  This combines the already-computed local spectra;
        # it never diagonalizes the full source Hamiltonian.
        target_values = np.stack(
            [
                np.sort(
                    np.concatenate(
                        [local_values[position][q] for q in range(layout.q_count)]
                    )
                )
                for position in validation_positions
            ],
            axis=0,
        )
    else:
        target_values = np.asarray(
            inputs.external_target_band_spectra[
                np.asarray(validation_positions, dtype=np.intp)
            ],
            dtype=np.float64,
        )
    frozen_target = _freeze_validation_target_window(
        target_values,
        config.target_window_spec,
    )
    source_hamiltonian_hash = hash_array(inputs.source_hamiltonians)

    producer_hard_thresholds = (
        {}
        if config.generated_candidate_envelope
        else {
            f"producer_integrity.{name}": float(value)
            for name, value in config.producer_integrity_thresholds.to_payload().items()
        }
    )
    routing_hard_thresholds = (
        {
            "common_anchor.local_action_isometry": float(
                effective_routing_thresholds.local_action_isometry
            ),
            "common_anchor.off_route_leakage": float(
                effective_routing_thresholds.off_route_leakage
            ),
            "common_anchor.projector_residual": float(
                effective_routing_thresholds.projector_residual
            ),
            "common_anchor.anchor_sigma_min": float(
                effective_routing_thresholds.anchor_sigma_min
            ),
            "candidate_envelope.effective_max_dimension": float(
                effective_max_dimension
            ),
        }
        if config.generated_candidate_envelope
        else {
            f"routing.{name}": float(value)
            for name, value in effective_routing_thresholds.to_payload().items()
            if name != "schema"
        }
    )
    hard_thresholds: dict[str, float] = {
        **producer_hard_thresholds,
        **{
            f"selection.{item.name}": float(
                getattr(config.selection_thresholds, item.name)
            )
            for item in fields(SelectionThresholds)
            if (
                not config.generated_candidate_envelope
                or item.name
                in {"band_rms_mev", "band_max_mev", "subspace_overlap"}
            )
        },
        **routing_hard_thresholds,
        **{
            f"candidate_symmetry.{item.name}": float(
                getattr(config.candidate_symmetry_thresholds, item.name)
            )
            for item in fields(CandidateSymmetryThresholds)
            if not config.generated_candidate_envelope
        },
    }
    policy_payload = config.policy_payload()
    policy_payload["common_anchor"] = {
        "reference_q_index": common_anchor_gauge.reference_q_index,
        "min_sigma": common_anchor_gauge.min_sigma,
        "max_condition": common_anchor_gauge.max_condition,
    }
    if config.generated_candidate_envelope:
        policy_payload["generated_candidate_envelope"] = {
            "schema": "kp.gamma-auto-generated-candidate-envelope.v1",
            "reference_k_index": reference_point.k_index,
            "reference_k_coordinate": list(reference_point.k_coordinate),
            "reference_q_indices": list(reference_point.q_indices),
            "reference_q_vectors": [
                list(vector) for vector in reference_point.q_vectors
            ],
            "candidate_seed_band_indices": [
                list(seed) for seed in candidate_seed_band_indices
            ],
            "effective_max_dimension": effective_max_dimension,
        }
    policy_hash = build_selection_policy_hash(
        hard_thresholds=hard_thresholds,
        candidate_envelope_config=policy_payload,
        candidate_generator_version=GAMMA_AUTO_PRODUCER_VERSION,
        candidate_schema_version=GAMMA_AUTO_CANDIDATE_SCHEMA,
        metric_schema_version=GAMMA_AUTO_METRIC_SCHEMA,
        ordering_rule_version=GAMMA_AUTO_ORDERING_RULE,
    )
    ordered_q_hash = hash_mapping(
        {
            "schema": "kp.gamma-common-anchor-ordered-q-identity.v1",
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
    return GammaAutomaticSelectionPreparation(
        inputs=inputs,
        config=config,
        reference_point=reference_point,
        candidate_seed_band_indices=candidate_seed_band_indices,
        effective_max_dimension=effective_max_dimension,
        layout=layout,
        common_anchor_gauge_config=common_anchor_gauge,
        certified_actions=certified_actions,
        operation_inputs=MappingProxyType(dict(operation_inputs)),
        required_pairs=MappingProxyType(dict(required_pairs)),
        raw_action_package_hash=raw_action_package_hash,
        local_values=local_values,
        local_vectors=local_vectors,
        frozen_target_window=frozen_target,
        target_values=target_values,
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
    seeds_to_materialize = (
        (_select_public_reference_seed(preparation),)
        if config.generated_candidate_envelope
        else preparation.candidate_seed_band_indices
    )
    for seed in seeds_to_materialize:
        try:
            evaluations.append(
                _evaluate_candidate(
                    seed=seed,
                    inputs=inputs,
                    config=config,
                    layout=preparation.layout,
                    common_anchor_gauge_config=(
                        preparation.common_anchor_gauge_config
                    ),
                    certified_actions=preparation.certified_actions,
                    operation_inputs=preparation.operation_inputs,
                    required_pairs=preparation.required_pairs,
                    raw_action_package_hash=preparation.raw_action_package_hash,
                    local_values=preparation.local_values,
                    local_vectors=preparation.local_vectors,
                    frozen_target=preparation.frozen_target_window,
                    source_hamiltonian_hash=preparation.source_hamiltonian_hash,
                    reference_k_index=preparation.reference_point.k_index,
                    candidate_max_rank=preparation.effective_max_dimension,
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
    if not evaluations:
        details = "; ".join(
            f"{item.seed_band_indices}: {item.reason.value}: {item.diagnostic}"
            for item in rejections
        )
        reason = (
            rejections[0].reason
            if len(rejections) == 1
            else CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED
        )
        raise GammaRoutingError(
            reason,
            "all automatic Gamma candidates were rejected" + (
                " (" + details + ")" if details else ""
            ),
        )
    metrics = tuple(evaluation.metrics for evaluation in evaluations)
    decision = select_projection_candidate(
        metrics,
        config.selection_thresholds,
        allow_pending_symmetry=config.generated_candidate_envelope,
    )
    selected = next(
        evaluation
        for evaluation in evaluations
        if evaluation.metrics.candidate_id == decision.selected.candidate_id
    )
    if (
        gamma_common_anchor_ordered_q_identity_hash(selected.handoff)
        != preparation.selection_input.ordered_q_hash
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "selected common-anchor handoff Q identity differs from preselection input",
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
    "GammaProducerIntegrityThresholds",
    "GammaRawOperationSpec",
    "evaluate_gamma_automatic_selection",
    "prepare_gamma_automatic_selection",
    "produce_gamma_automatic_selection",
]
