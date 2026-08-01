"""Immutable, fail-closed projection-selection identities and transactions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

import numpy as np

from .identity import hash_array, hash_mapping
from .low_energy_selection import CandidateMetrics
from .projection_selection import FrozenTargetWindow


SELECTION_ARTIFACT_SCHEMA_VERSION = "kp.selection-artifact.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRANSACTION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PAYLOAD_MANIFEST_SCHEMA_VERSION = "kp.selection-payload-manifest.v1"


class CertificationStatus(str, Enum):
    PENDING = "PENDING"
    CERTIFIED = "CERTIFIED"
    FAILED = "FAILED"
    UNVERIFIED_OVERRIDE = "UNVERIFIED_OVERRIDE"


class SelectionFailureCode(str, Enum):
    NO_CANDIDATES = "NO_CANDIDATES"
    DUPLICATE_CANDIDATE_ID = "DUPLICATE_CANDIDATE_ID"
    STRUCTURAL_REJECTION = "STRUCTURAL_REJECTION"
    NONFINITE_METRIC = "NONFINITE_METRIC"
    HARD_METRIC_FAILED = "HARD_METRIC_FAILED"
    CANDIDATE_CERTIFICATE_FAILED = "CANDIDATE_CERTIFICATE_FAILED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    PERSISTENCE_FAILURE = "PERSISTENCE_FAILURE"
    TRANSACTION_SUPERSEDED = "TRANSACTION_SUPERSEDED"
    EXPLICIT_UNVERIFIED = "EXPLICIT_UNVERIFIED"


class SelectionBindingError(ValueError):
    """Typed rejection for a selection/handoff identity mismatch."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: SelectionFailureCode = SelectionFailureCode.IDENTITY_MISMATCH,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: Sequence[str],
    context: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} must be a mapping")
    expected_set = set(expected)
    actual_set = set(payload)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{context} has invalid schema ({'; '.join(details)})")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")
    return value


def _require_hash(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _require_positive_integer(value: Any, field: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or int(value) <= 0
    ):
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _require_nonnegative_integer(value: Any, field: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or int(value) < 0
    ):
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _require_selection_mode(value: Any) -> str:
    mode = _require_nonempty_string(value, "selection_mode").lower()
    if mode not in {"auto", "explicit"}:
        raise ValueError("selection_mode must be 'auto' or 'explicit'")
    return mode


def _require_transaction_id(value: Any) -> str:
    transaction_id = _require_nonempty_string(value, "transaction_id")
    if _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None:
        raise ValueError("transaction_id must be a safe path component")
    return transaction_id


def _canonical_policy_value(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{field} must contain only finite numbers")
        return result
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} mapping keys must be nonempty strings")
            output[key] = _canonical_policy_value(item, f"{field}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            _canonical_policy_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field} contains unsupported value {type(value).__name__}")


def _normalized_hard_thresholds(values: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = _require_nonempty_string(raw_name, "hard threshold name")
        if not isinstance(raw_value, Real) or isinstance(raw_value, (bool, np.bool_)):
            raise ValueError(f"hard threshold {name} must be a real scalar")
        try:
            value = float(raw_value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"hard threshold {name} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"hard threshold {name} must be finite")
        output[name] = value
    return output


def _normalized_metrics(metrics: CandidateMetrics) -> CandidateMetrics:
    if not isinstance(metrics, CandidateMetrics):
        raise TypeError("metrics must be CandidateMetrics")
    candidate_id = _require_nonempty_string(metrics.candidate_id, "metrics.candidate_id")
    dimension = _require_positive_integer(metrics.dimension, "metrics.dimension")
    if metrics.structural_failure is not None:
        raise ValueError("certification metrics cannot contain structural_failure")
    names = (
        "band_rms_mev",
        "band_max_mev",
        "subspace_overlap",
        "symmetry_residual",
        "symmetry_leakage",
    )
    raw_values = tuple(getattr(metrics, name) for name in names)
    if any(
        not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
        for value in raw_values
    ):
        raise ValueError("candidate metrics must be real scalars")
    try:
        values = tuple(float(value) for value in raw_values)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("candidate metrics must be representable as finite floats") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("candidate metrics must be finite")
    if any(values[index] < 0.0 for index in (0, 1, 3, 4)):
        raise ValueError("candidate errors and symmetry residuals must be non-negative")
    if not 0.0 <= values[2] <= 1.0:
        raise ValueError("candidate subspace_overlap must lie in [0, 1]")
    return CandidateMetrics(
        candidate_id=candidate_id,
        dimension=dimension,
        band_rms_mev=values[0],
        band_max_mev=values[1],
        subspace_overlap=values[2],
        symmetry_residual=values[3],
        symmetry_leakage=values[4],
        structural_failure=None,
    )


def _metrics_to_dict(metrics: CandidateMetrics) -> dict[str, Any]:
    normalized = _normalized_metrics(metrics)
    return {
        "candidate_id": normalized.candidate_id,
        "dimension": normalized.dimension,
        "band_rms_mev": normalized.band_rms_mev,
        "band_max_mev": normalized.band_max_mev,
        "subspace_overlap": normalized.subspace_overlap,
        "symmetry_residual": normalized.symmetry_residual,
        "symmetry_leakage": normalized.symmetry_leakage,
        "structural_failure": None,
    }


def _metrics_from_dict(payload: Mapping[str, Any]) -> CandidateMetrics:
    fields = (
        "candidate_id",
        "dimension",
        "band_rms_mev",
        "band_max_mev",
        "subspace_overlap",
        "symmetry_residual",
        "symmetry_leakage",
        "structural_failure",
    )
    _require_exact_keys(payload, fields, "candidate metrics")
    return _normalized_metrics(CandidateMetrics(**{field: payload[field] for field in fields}))


def hash_validation_k_indices(window: FrozenTargetWindow) -> str:
    if not isinstance(window, FrozenTargetWindow):
        raise TypeError("window must be FrozenTargetWindow")
    return hash_mapping(
        {
            "schema": "kp.selection-validation-k.v1",
            "validation_k_indices": list(window.spec.validation_k_indices),
        }
    )


def hash_frozen_target_window(window: FrozenTargetWindow) -> str:
    if not isinstance(window, FrozenTargetWindow):
        raise TypeError("window must be FrozenTargetWindow")
    return hash_mapping(
        {
            "schema": "kp.frozen-target-window.v1",
            "spec": {
                "edge": window.spec.edge,
                "band_count": window.spec.band_count,
                "validation_k_indices": list(window.spec.validation_k_indices),
                "energy_reference_ev": window.spec.energy_reference_ev,
                "degeneracy_tolerance_mev": window.spec.degeneracy_tolerance_mev,
            },
            "target_band_ids": [list(row) for row in window.target_band_ids],
            "target_energies_ev": [list(row) for row in window.target_energies_ev],
        }
    )


def build_selection_policy_hash(
    *,
    hard_thresholds: Mapping[str, Any],
    candidate_envelope_config: Mapping[str, Any],
    candidate_generator_version: str,
    candidate_schema_version: str,
    metric_schema_version: str,
    ordering_rule_version: str,
) -> str:
    if not isinstance(hard_thresholds, Mapping):
        raise TypeError("hard_thresholds must be a mapping")
    if not isinstance(candidate_envelope_config, Mapping):
        raise TypeError("candidate_envelope_config must be a mapping")
    return hash_mapping(
        {
            "schema": "kp.selection-policy.v1",
            "hard_thresholds": _normalized_hard_thresholds(hard_thresholds),
            "candidate_envelope_config": _canonical_policy_value(
                candidate_envelope_config, "candidate_envelope_config"
            ),
            "candidate_generator_version": _require_nonempty_string(
                candidate_generator_version, "candidate_generator_version"
            ),
            "candidate_schema_version": _require_nonempty_string(
                candidate_schema_version, "candidate_schema_version"
            ),
            "metric_schema_version": _require_nonempty_string(
                metric_schema_version, "metric_schema_version"
            ),
            "ordering_rule_version": _require_nonempty_string(
                ordering_rule_version, "ordering_rule_version"
            ),
        }
    )


@dataclass(frozen=True)
class SelectionInputIdentity:
    selection_mode: str
    frozen_target_window_hash: str
    validation_k_indices_hash: str
    ordered_q_hash: str
    source_hamiltonian_hash: str
    action_package_hash: str
    row_layout_hash: str
    selection_policy_hash: str
    selection_input_identity_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_mode", _require_selection_mode(self.selection_mode))
        for field in (
            "frozen_target_window_hash",
            "validation_k_indices_hash",
            "ordered_q_hash",
            "source_hamiltonian_hash",
            "action_package_hash",
            "row_layout_hash",
            "selection_policy_hash",
        ):
            object.__setattr__(self, field, _require_hash(getattr(self, field), field))
        supplied = _require_hash(
            self.selection_input_identity_hash, "selection_input_identity_hash"
        )
        if supplied != self._expected_hash():
            raise ValueError("selection input identity hash mismatch")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": "kp.selection-input-identity.v1",
            "selection_mode": self.selection_mode,
            "frozen_target_window_hash": self.frozen_target_window_hash,
            "validation_k_indices_hash": self.validation_k_indices_hash,
            "ordered_q_hash": self.ordered_q_hash,
            "source_hamiltonian_hash": self.source_hamiltonian_hash,
            "action_package_hash": self.action_package_hash,
            "row_layout_hash": self.row_layout_hash,
            "selection_policy_hash": self.selection_policy_hash,
        }

    def _expected_hash(self) -> str:
        return hash_mapping(self._hash_payload())

    @classmethod
    def create(
        cls,
        *,
        selection_mode: str,
        frozen_target_window_hash: str,
        validation_k_indices_hash: str,
        ordered_q_hash: str,
        source_hamiltonian_hash: str,
        action_package_hash: str,
        row_layout_hash: str,
        selection_policy_hash: str,
    ) -> "SelectionInputIdentity":
        values = {
            "selection_mode": _require_selection_mode(selection_mode),
            "frozen_target_window_hash": _require_hash(
                frozen_target_window_hash, "frozen_target_window_hash"
            ),
            "validation_k_indices_hash": _require_hash(
                validation_k_indices_hash, "validation_k_indices_hash"
            ),
            "ordered_q_hash": _require_hash(ordered_q_hash, "ordered_q_hash"),
            "source_hamiltonian_hash": _require_hash(
                source_hamiltonian_hash, "source_hamiltonian_hash"
            ),
            "action_package_hash": _require_hash(
                action_package_hash, "action_package_hash"
            ),
            "row_layout_hash": _require_hash(row_layout_hash, "row_layout_hash"),
            "selection_policy_hash": _require_hash(
                selection_policy_hash, "selection_policy_hash"
            ),
        }
        payload = {"schema": "kp.selection-input-identity.v1", **values}
        return cls(**values, selection_input_identity_hash=hash_mapping(payload))

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload.pop("schema")
        return {
            **payload,
            "selection_input_identity_hash": self.selection_input_identity_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionInputIdentity":
        fields = (
            "selection_mode",
            "frozen_target_window_hash",
            "validation_k_indices_hash",
            "ordered_q_hash",
            "source_hamiltonian_hash",
            "action_package_hash",
            "row_layout_hash",
            "selection_policy_hash",
            "selection_input_identity_hash",
        )
        _require_exact_keys(payload, fields, "selection input identity")
        return cls(**{field: payload[field] for field in fields})


def build_selection_input_identity_hash(
    *,
    selection_mode: str,
    frozen_target_window_hash: str,
    validation_k_indices_hash: str,
    ordered_q_hash: str,
    source_hamiltonian_hash: str,
    action_package_hash: str,
    row_layout_hash: str,
    selection_policy_hash: str,
) -> str:
    return SelectionInputIdentity.create(
        selection_mode=selection_mode,
        frozen_target_window_hash=frozen_target_window_hash,
        validation_k_indices_hash=validation_k_indices_hash,
        ordered_q_hash=ordered_q_hash,
        source_hamiltonian_hash=source_hamiltonian_hash,
        action_package_hash=action_package_hash,
        row_layout_hash=row_layout_hash,
        selection_policy_hash=selection_policy_hash,
    ).selection_input_identity_hash


@dataclass(frozen=True)
class ResolvedCandidateIdentity:
    selection_mode: str
    candidate_id: str
    candidate_dimension: int
    projection_basis_kind: str
    basis_handoff_hash: str
    authoritative_heff_hash: str
    heff_k_indices_hash: str
    resolved_candidate_hash: str

    def __post_init__(self) -> None:
        mode = _require_selection_mode(self.selection_mode)
        object.__setattr__(self, "selection_mode", mode)
        object.__setattr__(
            self, "candidate_id", _require_nonempty_string(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self,
            "candidate_dimension",
            _require_positive_integer(self.candidate_dimension, "candidate_dimension"),
        )
        object.__setattr__(
            self,
            "projection_basis_kind",
            _require_nonempty_string(self.projection_basis_kind, "projection_basis_kind"),
        )
        for field in (
            "basis_handoff_hash",
            "authoritative_heff_hash",
            "heff_k_indices_hash",
        ):
            object.__setattr__(self, field, _require_hash(getattr(self, field), field))
        supplied = _require_hash(self.resolved_candidate_hash, "resolved_candidate_hash")
        if supplied != self._expected_hash():
            raise ValueError("resolved candidate hash mismatch")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": "kp.resolved-candidate-identity.v1",
            "selection_mode": self.selection_mode,
            "candidate_id": self.candidate_id,
            "candidate_dimension": self.candidate_dimension,
            "projection_basis_kind": self.projection_basis_kind,
            "basis_handoff_hash": self.basis_handoff_hash,
            "authoritative_heff_hash": self.authoritative_heff_hash,
            "heff_k_indices_hash": self.heff_k_indices_hash,
        }

    def _expected_hash(self) -> str:
        return hash_mapping(self._hash_payload())

    @classmethod
    def create(cls, **values: Any) -> "ResolvedCandidateIdentity":
        mode = _require_selection_mode(values["selection_mode"])
        payload = {
            "schema": "kp.resolved-candidate-identity.v1",
            "selection_mode": mode,
            "candidate_id": _require_nonempty_string(values["candidate_id"], "candidate_id"),
            "candidate_dimension": _require_positive_integer(
                values["candidate_dimension"], "candidate_dimension"
            ),
            "projection_basis_kind": _require_nonempty_string(
                values["projection_basis_kind"], "projection_basis_kind"
            ),
            "basis_handoff_hash": _require_hash(
                values["basis_handoff_hash"], "basis_handoff_hash"
            ),
            "authoritative_heff_hash": _require_hash(
                values["authoritative_heff_hash"], "authoritative_heff_hash"
            ),
            "heff_k_indices_hash": _require_hash(
                values["heff_k_indices_hash"], "heff_k_indices_hash"
            ),
        }
        return cls(
            **{key: value for key, value in payload.items() if key != "schema"},
            resolved_candidate_hash=hash_mapping(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload.pop("schema")
        return {**payload, "resolved_candidate_hash": self.resolved_candidate_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedCandidateIdentity":
        fields = (
            "selection_mode",
            "candidate_id",
            "candidate_dimension",
            "projection_basis_kind",
            "basis_handoff_hash",
            "authoritative_heff_hash",
            "heff_k_indices_hash",
            "resolved_candidate_hash",
        )
        _require_exact_keys(payload, fields, "resolved candidate identity")
        return cls(**{field: payload[field] for field in fields})


@dataclass(frozen=True)
class SelectionMetricEvidence:
    candidate_id: str
    frozen_target_window_hash: str
    validation_k_indices_hash: str
    basis_handoff_hash: str
    metrics: CandidateMetrics
    metrics_hash: str

    def __post_init__(self) -> None:
        candidate_id = _require_nonempty_string(self.candidate_id, "candidate_id")
        metrics = _normalized_metrics(self.metrics)
        if metrics.candidate_id != candidate_id:
            raise ValueError("metric evidence candidate_id does not match metrics")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "metrics", metrics)
        for field in (
            "frozen_target_window_hash",
            "validation_k_indices_hash",
            "basis_handoff_hash",
        ):
            object.__setattr__(self, field, _require_hash(getattr(self, field), field))
        supplied = _require_hash(self.metrics_hash, "metrics_hash")
        if supplied != self._expected_hash():
            raise ValueError("selection metric evidence hash mismatch")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": "kp.selection-metric-evidence.v1",
            "candidate_id": self.candidate_id,
            "frozen_target_window_hash": self.frozen_target_window_hash,
            "validation_k_indices_hash": self.validation_k_indices_hash,
            "basis_handoff_hash": self.basis_handoff_hash,
            "metrics": _metrics_to_dict(self.metrics),
        }

    def _expected_hash(self) -> str:
        return hash_mapping(self._hash_payload())

    @classmethod
    def create(cls, **values: Any) -> "SelectionMetricEvidence":
        candidate_id = _require_nonempty_string(values["candidate_id"], "candidate_id")
        metrics = _normalized_metrics(values["metrics"])
        if metrics.candidate_id != candidate_id:
            raise ValueError("metric evidence candidate_id does not match metrics")
        payload = {
            "schema": "kp.selection-metric-evidence.v1",
            "candidate_id": candidate_id,
            "frozen_target_window_hash": _require_hash(
                values["frozen_target_window_hash"], "frozen_target_window_hash"
            ),
            "validation_k_indices_hash": _require_hash(
                values["validation_k_indices_hash"], "validation_k_indices_hash"
            ),
            "basis_handoff_hash": _require_hash(
                values["basis_handoff_hash"], "basis_handoff_hash"
            ),
            "metrics": _metrics_to_dict(metrics),
        }
        return cls(
            candidate_id=candidate_id,
            frozen_target_window_hash=payload["frozen_target_window_hash"],
            validation_k_indices_hash=payload["validation_k_indices_hash"],
            basis_handoff_hash=payload["basis_handoff_hash"],
            metrics=metrics,
            metrics_hash=hash_mapping(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload.pop("schema")
        return {**payload, "metrics_hash": self.metrics_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionMetricEvidence":
        fields = (
            "candidate_id",
            "frozen_target_window_hash",
            "validation_k_indices_hash",
            "basis_handoff_hash",
            "metrics",
            "metrics_hash",
        )
        _require_exact_keys(payload, fields, "selection metric evidence")
        if not isinstance(payload["metrics"], Mapping):
            raise TypeError("selection metric evidence metrics must be a mapping")
        return cls(
            candidate_id=payload["candidate_id"],
            frozen_target_window_hash=payload["frozen_target_window_hash"],
            validation_k_indices_hash=payload["validation_k_indices_hash"],
            basis_handoff_hash=payload["basis_handoff_hash"],
            metrics=_metrics_from_dict(payload["metrics"]),
            metrics_hash=payload["metrics_hash"],
        )


@dataclass(frozen=True)
class CertificationEvidence:
    metric_evidence: SelectionMetricEvidence
    symmetry_certificate_hash: str
    symmetry_input_identity_hash: str
    evidence_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.metric_evidence, SelectionMetricEvidence):
            raise TypeError("metric_evidence must be SelectionMetricEvidence")
        for field in ("symmetry_certificate_hash", "symmetry_input_identity_hash"):
            object.__setattr__(self, field, _require_hash(getattr(self, field), field))
        supplied = _require_hash(self.evidence_hash, "evidence_hash")
        if supplied != self._expected_hash():
            raise ValueError("certification evidence hash mismatch")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": "kp.certification-evidence.v1",
            "metric_evidence": self.metric_evidence.to_dict(),
            "symmetry_certificate_hash": self.symmetry_certificate_hash,
            "symmetry_input_identity_hash": self.symmetry_input_identity_hash,
        }

    def _expected_hash(self) -> str:
        return hash_mapping(self._hash_payload())

    @classmethod
    def create(cls, **values: Any) -> "CertificationEvidence":
        metric_evidence = values["metric_evidence"]
        if not isinstance(metric_evidence, SelectionMetricEvidence):
            raise TypeError("metric_evidence must be SelectionMetricEvidence")
        payload = {
            "schema": "kp.certification-evidence.v1",
            "metric_evidence": metric_evidence.to_dict(),
            "symmetry_certificate_hash": _require_hash(
                values["symmetry_certificate_hash"], "symmetry_certificate_hash"
            ),
            "symmetry_input_identity_hash": _require_hash(
                values["symmetry_input_identity_hash"], "symmetry_input_identity_hash"
            ),
        }
        return cls(
            metric_evidence=metric_evidence,
            symmetry_certificate_hash=payload["symmetry_certificate_hash"],
            symmetry_input_identity_hash=payload["symmetry_input_identity_hash"],
            evidence_hash=hash_mapping(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload.pop("schema")
        return {**payload, "evidence_hash": self.evidence_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertificationEvidence":
        fields = (
            "metric_evidence",
            "symmetry_certificate_hash",
            "symmetry_input_identity_hash",
            "evidence_hash",
        )
        _require_exact_keys(payload, fields, "certification evidence")
        if not isinstance(payload["metric_evidence"], Mapping):
            raise TypeError("certification metric_evidence must be a mapping")
        return cls(
            metric_evidence=SelectionMetricEvidence.from_dict(payload["metric_evidence"]),
            symmetry_certificate_hash=payload["symmetry_certificate_hash"],
            symmetry_input_identity_hash=payload["symmetry_input_identity_hash"],
            evidence_hash=payload["evidence_hash"],
        )


@dataclass(frozen=True)
class SelectionIdentity:
    selection_input: SelectionInputIdentity
    selection_policy_hash: str
    resolved_candidate: ResolvedCandidateIdentity
    certification_evidence: CertificationEvidence | None
    selection_identity_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.selection_input, SelectionInputIdentity):
            raise TypeError("selection_input must be SelectionInputIdentity")
        object.__setattr__(
            self,
            "selection_policy_hash",
            _require_hash(self.selection_policy_hash, "selection_policy_hash"),
        )
        if not isinstance(self.resolved_candidate, ResolvedCandidateIdentity):
            raise TypeError("resolved_candidate must be ResolvedCandidateIdentity")
        if self.selection_input.selection_mode != self.resolved_candidate.selection_mode:
            raise ValueError("selection input mode does not match resolved candidate mode")
        if self.selection_input.selection_policy_hash != self.selection_policy_hash:
            raise ValueError("selection input policy does not match selection identity policy")
        evidence = self.certification_evidence
        if evidence is not None:
            if not isinstance(evidence, CertificationEvidence):
                raise TypeError("certification_evidence must be CertificationEvidence or None")
            if evidence.metric_evidence.candidate_id != self.resolved_candidate.candidate_id:
                raise ValueError("certification evidence candidate does not match resolved candidate")
            if (
                evidence.metric_evidence.metrics.dimension
                != self.resolved_candidate.candidate_dimension
            ):
                raise ValueError("certification evidence dimension does not match resolved candidate")
            if (
                evidence.metric_evidence.frozen_target_window_hash
                != self.selection_input.frozen_target_window_hash
            ):
                raise ValueError("certification evidence target does not match selection input")
            if (
                evidence.metric_evidence.validation_k_indices_hash
                != self.selection_input.validation_k_indices_hash
            ):
                raise ValueError(
                    "certification evidence validation-k identity does not match selection input"
                )
            if (
                evidence.metric_evidence.basis_handoff_hash
                != self.resolved_candidate.basis_handoff_hash
            ):
                raise ValueError("certification evidence basis does not match resolved candidate")
        supplied = _require_hash(self.selection_identity_hash, "selection_identity_hash")
        if supplied != self._expected_hash():
            raise ValueError("selection identity hash mismatch")

    @property
    def selection_input_identity_hash(self) -> str:
        return self.selection_input.selection_input_identity_hash

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": "kp.selection-identity.v1",
            "selection_input": self.selection_input.to_dict(),
            "selection_policy_hash": self.selection_policy_hash,
            "resolved_candidate": self.resolved_candidate.to_dict(),
            "certification_evidence": (
                None
                if self.certification_evidence is None
                else self.certification_evidence.to_dict()
            ),
        }

    def _expected_hash(self) -> str:
        return hash_mapping(self._hash_payload())

    @classmethod
    def create(
        cls,
        *,
        selection_input: SelectionInputIdentity,
        selection_policy_hash: str,
        resolved_candidate: ResolvedCandidateIdentity,
        certification_evidence: CertificationEvidence | None,
    ) -> "SelectionIdentity":
        if not isinstance(selection_input, SelectionInputIdentity):
            raise TypeError("selection_input must be SelectionInputIdentity")
        resolved = resolved_candidate
        evidence = certification_evidence
        if not isinstance(resolved, ResolvedCandidateIdentity):
            raise TypeError("resolved_candidate must be ResolvedCandidateIdentity")
        if evidence is not None and not isinstance(evidence, CertificationEvidence):
            raise TypeError("certification_evidence must be CertificationEvidence or None")
        payload = {
            "schema": "kp.selection-identity.v1",
            "selection_input": selection_input.to_dict(),
            "selection_policy_hash": _require_hash(
                selection_policy_hash, "selection_policy_hash"
            ),
            "resolved_candidate": resolved.to_dict(),
            "certification_evidence": None if evidence is None else evidence.to_dict(),
        }
        return cls(
            selection_input=selection_input,
            selection_policy_hash=payload["selection_policy_hash"],
            resolved_candidate=resolved,
            certification_evidence=evidence,
            selection_identity_hash=hash_mapping(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload.pop("schema")
        return {**payload, "selection_identity_hash": self.selection_identity_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionIdentity":
        fields = (
            "selection_input",
            "selection_policy_hash",
            "resolved_candidate",
            "certification_evidence",
            "selection_identity_hash",
        )
        _require_exact_keys(payload, fields, "selection identity")
        if not isinstance(payload["selection_input"], Mapping):
            raise TypeError("selection input identity must be a mapping")
        if not isinstance(payload["resolved_candidate"], Mapping):
            raise TypeError("selection resolved_candidate must be a mapping")
        evidence_payload = payload["certification_evidence"]
        if evidence_payload is not None and not isinstance(evidence_payload, Mapping):
            raise TypeError("selection certification_evidence must be a mapping or null")
        return cls(
            selection_input=SelectionInputIdentity.from_dict(payload["selection_input"]),
            selection_policy_hash=payload["selection_policy_hash"],
            resolved_candidate=ResolvedCandidateIdentity.from_dict(payload["resolved_candidate"]),
            certification_evidence=(
                None
                if evidence_payload is None
                else CertificationEvidence.from_dict(evidence_payload)
            ),
            selection_identity_hash=payload["selection_identity_hash"],
        )


def _require_gamma_routed_handoff(handoff: Any) -> Any:
    from .projection_handoff import GammaRoutedBasisSpec

    if not isinstance(handoff, GammaRoutedBasisSpec):
        raise SelectionBindingError(
            "certified Gamma selection requires a GammaRoutedBasisSpec handoff"
        )
    if not handoff.candidate_id or not handoff.candidate_certificate_envelope_json:
        raise SelectionBindingError(
            "certified Gamma selection requires a complete candidate certificate"
        )
    return handoff


def gamma_routed_ordered_q_identity_hash(handoff: Any) -> str:
    """Hash the ordered Q identities owned by a routed Gamma handoff."""

    routed = _require_gamma_routed_handoff(handoff)
    return hash_mapping(
        {
            "schema": "kp.gamma-routed-ordered-q-identity.v1",
            "ordered_q_hashes": list(routed.ordered_q_hashes),
        }
    )


def _verified_gamma_candidate_envelope(handoff: Any) -> Mapping[str, Any]:
    from .symmetry.candidate_certificate import (
        CandidateSymmetryStatus,
        candidate_action_package_hash,
        verify_candidate_certificate_envelope,
    )

    routed = _require_gamma_routed_handoff(handoff)
    try:
        payload = json.loads(routed.candidate_certificate_envelope_json)
        verified = verify_candidate_certificate_envelope(payload)
    except (TypeError, ValueError) as exc:
        raise SelectionBindingError(
            "routed Gamma candidate certificate envelope is invalid"
        ) from exc
    if (
        verified.get("status") != CandidateSymmetryStatus.CERTIFIED.value
        or verified.get("candidate_id") != routed.candidate_id
        or verified.get("certificate_hash") != routed.candidate_certificate_hash
        or verified.get("input_identity_hash")
        != routed.candidate_input_identity_hash
        or candidate_action_package_hash(verified["input_identity_payload"])
        != routed.raw_action_package_hash
    ):
        raise SelectionBindingError(
            "routed Gamma candidate certificate does not match its handoff"
        )
    return verified


def _gamma_handoff_identity_values(handoff: Any) -> dict[str, str | int]:
    routed = _require_gamma_routed_handoff(handoff)
    _verified_gamma_candidate_envelope(routed)
    return {
        "candidate_id": routed.candidate_id,
        "candidate_dimension": routed.model_dim,
        "projection_basis_kind": routed.projection_basis_kind,
        "basis_handoff_hash": routed.handoff_identity_hash,
        "authoritative_heff_hash": routed.heff_hash,
        "heff_k_indices_hash": hash_array(
            np.asarray(routed.heff_k_indices, dtype=np.int64)
        ),
    }


def _verify_gamma_selection_input(
    selection_input: SelectionInputIdentity,
    handoff: Any,
) -> None:
    routed = _require_gamma_routed_handoff(handoff)
    if not isinstance(selection_input, SelectionInputIdentity):
        raise SelectionBindingError(
            "certified Gamma selection requires a SelectionInputIdentity"
        )
    expected = {
        "selection_mode": "auto",
        "ordered_q_hash": gamma_routed_ordered_q_identity_hash(routed),
        "source_hamiltonian_hash": routed.source_hamiltonian_hash,
        "action_package_hash": routed.raw_action_package_hash,
        "row_layout_hash": routed.layout.layout_hash,
    }
    mismatched = tuple(
        field
        for field, value in expected.items()
        if getattr(selection_input, field) != value
    )
    if mismatched:
        raise SelectionBindingError(
            "selection input does not match routed Gamma handoff fields: "
            + ", ".join(mismatched)
        )


def build_certified_gamma_selection_identity(
    *,
    selection_input: SelectionInputIdentity,
    handoff: Any,
    metrics: CandidateMetrics,
) -> SelectionIdentity:
    """Build certification evidence only from a validated routed handoff."""

    routed = _require_gamma_routed_handoff(handoff)
    _verify_gamma_selection_input(selection_input, routed)
    values = _gamma_handoff_identity_values(routed)
    try:
        normalized_metrics = _normalized_metrics(metrics)
    except (TypeError, ValueError) as exc:
        raise SelectionBindingError("selection metrics are invalid") from exc
    if (
        normalized_metrics.candidate_id != values["candidate_id"]
        or normalized_metrics.dimension != values["candidate_dimension"]
    ):
        raise SelectionBindingError(
            "selection metrics do not match the routed Gamma candidate"
        )
    resolved = ResolvedCandidateIdentity.create(
        selection_mode="auto",
        **values,
    )
    metric_evidence = SelectionMetricEvidence.create(
        candidate_id=normalized_metrics.candidate_id,
        frozen_target_window_hash=selection_input.frozen_target_window_hash,
        validation_k_indices_hash=selection_input.validation_k_indices_hash,
        basis_handoff_hash=str(values["basis_handoff_hash"]),
        metrics=normalized_metrics,
    )
    evidence = CertificationEvidence.create(
        metric_evidence=metric_evidence,
        symmetry_certificate_hash=routed.candidate_certificate_hash,
        symmetry_input_identity_hash=routed.candidate_input_identity_hash,
    )
    return SelectionIdentity.create(
        selection_input=selection_input,
        selection_policy_hash=selection_input.selection_policy_hash,
        resolved_candidate=resolved,
        certification_evidence=evidence,
    )


def verify_certified_gamma_selection_identity(
    identity: SelectionIdentity,
    handoff: Any,
) -> SelectionIdentity:
    """Rebuild and compare every selection identity field to the handoff."""

    if not isinstance(identity, SelectionIdentity):
        raise SelectionBindingError("selection identity has the wrong type")
    evidence = identity.certification_evidence
    if evidence is None:
        raise SelectionBindingError(
            "certified Gamma selection requires certification evidence"
        )
    expected = build_certified_gamma_selection_identity(
        selection_input=identity.selection_input,
        handoff=handoff,
        metrics=evidence.metric_evidence.metrics,
    )
    if identity != expected:
        raise SelectionBindingError(
            "selection identity does not match the routed Gamma handoff"
        )
    return identity


def verify_certified_gamma_selection_artifact(
    artifact: "SelectionArtifact",
    handoff: Any,
) -> "SelectionArtifact":
    if (
        not isinstance(artifact, SelectionArtifact)
        or artifact.certification_status is not CertificationStatus.CERTIFIED
        or artifact.identity is None
    ):
        raise SelectionBindingError("selection artifact is not CERTIFIED")
    verify_certified_gamma_selection_identity(artifact.identity, handoff)
    return artifact


@dataclass(frozen=True)
class SelectionArtifact:
    schema_version: str
    transaction_id: str
    selection_input_identity_hash: str
    payload_manifest_hash: str | None
    identity: SelectionIdentity | None
    metrics: CandidateMetrics | None
    certification_status: CertificationStatus
    status_reason: SelectionFailureCode | None
    failure_codes: tuple[SelectionFailureCode, ...]
    diagnostic: str | None
    artifact_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SELECTION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported selection artifact schema {self.schema_version!r}")
        object.__setattr__(
            self, "transaction_id", _require_transaction_id(self.transaction_id)
        )
        object.__setattr__(
            self,
            "selection_input_identity_hash",
            _require_hash(self.selection_input_identity_hash, "selection_input_identity_hash"),
        )
        if self.payload_manifest_hash is not None:
            object.__setattr__(
                self,
                "payload_manifest_hash",
                _require_hash(self.payload_manifest_hash, "payload_manifest_hash"),
            )
        try:
            status = CertificationStatus(self.certification_status)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid certification_status") from exc
        object.__setattr__(self, "certification_status", status)
        if self.identity is not None and not isinstance(self.identity, SelectionIdentity):
            raise TypeError("identity must be SelectionIdentity or None")
        metrics = None if self.metrics is None else _normalized_metrics(self.metrics)
        object.__setattr__(self, "metrics", metrics)
        if self.status_reason is None:
            reason = None
        else:
            try:
                reason = SelectionFailureCode(self.status_reason)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid status_reason") from exc
        object.__setattr__(self, "status_reason", reason)
        try:
            failures = tuple(SelectionFailureCode(code) for code in self.failure_codes)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid selection failure code") from exc
        if len(set(failures)) != len(failures):
            raise ValueError("selection failure codes must be unique")
        object.__setattr__(self, "failure_codes", failures)
        if self.diagnostic is not None:
            object.__setattr__(
                self, "diagnostic", _require_nonempty_string(self.diagnostic, "diagnostic")
            )
        self._validate_state()
        supplied = _require_hash(self.artifact_hash, "artifact_hash")
        if supplied != self._expected_hash():
            raise ValueError("selection artifact hash mismatch")

    def _validate_state(self) -> None:
        status = self.certification_status
        if self.identity is not None and (
            self.identity.selection_input_identity_hash
            != self.selection_input_identity_hash
        ):
            raise ValueError("selection artifact input identity does not match final identity")
        if status is CertificationStatus.PENDING:
            if any(
                value is not None
                for value in (
                    self.payload_manifest_hash,
                    self.identity,
                    self.metrics,
                    self.status_reason,
                    self.diagnostic,
                )
            ) or self.failure_codes:
                raise ValueError("PENDING selection artifact cannot contain final identity or evidence")
            return
        if status is CertificationStatus.FAILED:
            if self.payload_manifest_hash is not None:
                raise ValueError("FAILED selection artifact cannot bind certified payloads")
            if self.identity is not None or self.metrics is not None:
                raise ValueError("FAILED selection artifact cannot claim a resolved candidate")
            if self.status_reason is None or not self.failure_codes:
                raise ValueError("FAILED selection artifact requires typed failure codes")
            if self.status_reason not in self.failure_codes:
                raise ValueError("FAILED status_reason must be present in failure_codes")
            if self.diagnostic is None:
                raise ValueError("FAILED selection artifact requires a diagnostic")
            return
        if status is CertificationStatus.UNVERIFIED_OVERRIDE:
            if self.payload_manifest_hash is not None:
                raise ValueError("UNVERIFIED_OVERRIDE cannot bind certified payloads")
            if self.identity is None:
                raise ValueError("UNVERIFIED_OVERRIDE requires a resolved candidate identity")
            if self.identity.resolved_candidate.selection_mode != "explicit":
                raise ValueError("UNVERIFIED_OVERRIDE requires explicit selection mode")
            if self.identity.certification_evidence is not None or self.metrics is not None:
                raise ValueError("UNVERIFIED_OVERRIDE cannot contain certification evidence")
            if self.status_reason is not SelectionFailureCode.EXPLICIT_UNVERIFIED:
                raise ValueError("UNVERIFIED_OVERRIDE requires EXPLICIT_UNVERIFIED reason")
            if self.failure_codes != (SelectionFailureCode.EXPLICIT_UNVERIFIED,):
                raise ValueError("UNVERIFIED_OVERRIDE requires exactly EXPLICIT_UNVERIFIED")
            if self.diagnostic is None:
                raise ValueError("UNVERIFIED_OVERRIDE requires a diagnostic")
            return
        assert status is CertificationStatus.CERTIFIED
        if self.payload_manifest_hash is None:
            raise ValueError("CERTIFIED selection artifact requires a payload manifest hash")
        if self.identity is None:
            raise ValueError("CERTIFIED selection artifact requires an identity")
        if self.identity.resolved_candidate.selection_mode != "auto":
            raise ValueError("CERTIFIED selection artifact requires automatic selection mode")
        if self.identity.certification_evidence is None:
            raise ValueError("CERTIFIED selection artifact requires certification evidence")
        evidence_metrics = self.identity.certification_evidence.metric_evidence.metrics
        if self.metrics is None or self.metrics != evidence_metrics:
            raise ValueError("top-level metrics must have certification evidence as single source of truth")
        if self.status_reason is not None or self.failure_codes:
            raise ValueError("CERTIFIED selection artifact cannot contain failure codes")
        if self.diagnostic is not None:
            raise ValueError("CERTIFIED selection artifact cannot contain a diagnostic")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "selection_input_identity_hash": self.selection_input_identity_hash,
            "payload_manifest_hash": self.payload_manifest_hash,
            "identity": None if self.identity is None else self.identity.to_dict(),
            "metrics": None if self.metrics is None else _metrics_to_dict(self.metrics),
            "certification_status": self.certification_status.value,
            "status_reason": None if self.status_reason is None else self.status_reason.value,
            "failure_codes": [code.value for code in self.failure_codes],
            "diagnostic": self.diagnostic,
        }

    def _expected_hash(self) -> str:
        return hash_mapping(self._hash_payload())

    @classmethod
    def _create(cls, **values: Any) -> "SelectionArtifact":
        provisional = object.__new__(cls)
        for field, value in values.items():
            object.__setattr__(provisional, field, value)
        payload = cls._hash_payload(provisional)
        return cls(**values, artifact_hash=hash_mapping(payload))

    @classmethod
    def pending(
        cls,
        *,
        transaction_id: str,
        selection_input_identity_hash: str,
    ) -> "SelectionArtifact":
        return cls._create(
            schema_version=SELECTION_ARTIFACT_SCHEMA_VERSION,
            transaction_id=transaction_id,
            selection_input_identity_hash=selection_input_identity_hash,
            payload_manifest_hash=None,
            identity=None,
            metrics=None,
            certification_status=CertificationStatus.PENDING,
            status_reason=None,
            failure_codes=(),
            diagnostic=None,
        )

    @classmethod
    def certified(
        cls,
        *,
        transaction_id: str,
        identity: SelectionIdentity,
        payload_manifest_hash: str,
        projection_handoff: Any | None = None,
    ) -> "SelectionArtifact":
        if not isinstance(identity, SelectionIdentity):
            raise TypeError("identity must be SelectionIdentity")
        if identity.certification_evidence is None:
            raise ValueError("CERTIFIED selection artifact requires certification evidence")
        from .projection_handoff import GAMMA_ROUTED_BASIS_KIND

        if (
            identity.resolved_candidate.projection_basis_kind
            == GAMMA_ROUTED_BASIS_KIND
        ):
            if projection_handoff is None:
                raise SelectionBindingError(
                    "CERTIFIED routed Gamma selection requires its projection handoff"
                )
            verify_certified_gamma_selection_identity(identity, projection_handoff)
        return cls._create(
            schema_version=SELECTION_ARTIFACT_SCHEMA_VERSION,
            transaction_id=transaction_id,
            selection_input_identity_hash=identity.selection_input_identity_hash,
            payload_manifest_hash=_require_hash(
                payload_manifest_hash, "payload_manifest_hash"
            ),
            identity=identity,
            metrics=identity.certification_evidence.metric_evidence.metrics,
            certification_status=CertificationStatus.CERTIFIED,
            status_reason=None,
            failure_codes=(),
            diagnostic=None,
        )

    @classmethod
    def failed(
        cls,
        *,
        transaction_id: str,
        selection_input_identity_hash: str,
        reason: SelectionFailureCode,
        failure_codes: Sequence[SelectionFailureCode],
        diagnostic: str,
    ) -> "SelectionArtifact":
        return cls._create(
            schema_version=SELECTION_ARTIFACT_SCHEMA_VERSION,
            transaction_id=transaction_id,
            selection_input_identity_hash=selection_input_identity_hash,
            payload_manifest_hash=None,
            identity=None,
            metrics=None,
            certification_status=CertificationStatus.FAILED,
            status_reason=reason,
            failure_codes=tuple(failure_codes),
            diagnostic=diagnostic,
        )

    @classmethod
    def unverified_override(
        cls,
        *,
        transaction_id: str,
        identity: SelectionIdentity,
        diagnostic: str,
    ) -> "SelectionArtifact":
        if not isinstance(identity, SelectionIdentity):
            raise TypeError("identity must be SelectionIdentity")
        return cls._create(
            schema_version=SELECTION_ARTIFACT_SCHEMA_VERSION,
            transaction_id=transaction_id,
            selection_input_identity_hash=identity.selection_input_identity_hash,
            payload_manifest_hash=None,
            identity=identity,
            metrics=None,
            certification_status=CertificationStatus.UNVERIFIED_OVERRIDE,
            status_reason=SelectionFailureCode.EXPLICIT_UNVERIFIED,
            failure_codes=(SelectionFailureCode.EXPLICIT_UNVERIFIED,),
            diagnostic=diagnostic,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "artifact_hash": self.artifact_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionArtifact":
        fields = (
            "schema_version",
            "transaction_id",
            "selection_input_identity_hash",
            "payload_manifest_hash",
            "identity",
            "metrics",
            "certification_status",
            "status_reason",
            "failure_codes",
            "diagnostic",
            "artifact_hash",
        )
        _require_exact_keys(payload, fields, "selection artifact")
        identity_payload = payload["identity"]
        metrics_payload = payload["metrics"]
        if identity_payload is not None and not isinstance(identity_payload, Mapping):
            raise TypeError("selection identity must be a mapping or null")
        if metrics_payload is not None and not isinstance(metrics_payload, Mapping):
            raise TypeError("selection metrics must be a mapping or null")
        failure_codes = payload["failure_codes"]
        if not isinstance(failure_codes, list):
            raise TypeError("selection failure_codes must be a list")
        return cls(
            schema_version=payload["schema_version"],
            transaction_id=payload["transaction_id"],
            selection_input_identity_hash=payload["selection_input_identity_hash"],
            payload_manifest_hash=payload["payload_manifest_hash"],
            identity=(
                None if identity_payload is None else SelectionIdentity.from_dict(identity_payload)
            ),
            metrics=(None if metrics_payload is None else _metrics_from_dict(metrics_payload)),
            certification_status=payload["certification_status"],
            status_reason=payload["status_reason"],
            failure_codes=tuple(failure_codes),
            diagnostic=payload["diagnostic"],
            artifact_hash=payload["artifact_hash"],
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"selection artifact JSON contains non-finite value {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"selection artifact JSON contains duplicate field {key!r}")
        output[key] = value
    return output


def load_selection_artifact(path: str | Path) -> SelectionArtifact:
    artifact_path = Path(path)
    try:
        encoded = _read_regular_file(
            artifact_path,
            "selection artifact marker",
        ).decode("utf-8")
        payload = json.loads(
            encoded,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"selection artifact marker is not valid UTF-8: {artifact_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid selection artifact JSON: {artifact_path}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("selection artifact JSON root must be a mapping")
    return SelectionArtifact.from_dict(payload)


class SelectionTransactionError(RuntimeError):
    def __init__(
        self,
        failure_code: SelectionFailureCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.failure_code = SelectionFailureCode(failure_code)


def _payload_content_hash(content: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"kp.selection-payload.v1\0")
    digest.update(content)
    return digest.hexdigest()


def _require_payload_name(value: Any) -> str:
    name = _require_nonempty_string(value, "payload name")
    path = Path(name)
    if (
        path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != name
        or name == "payload_manifest.json"
    ):
        raise ValueError(f"unsafe payload name {name!r}")
    return name


def _read_regular_file(path: Path, context: str) -> bytes:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ValueError(f"{context} is not readable") from exc
    if stat.S_ISLNK(status.st_mode):
        raise ValueError(f"{context} must not be a symlink")
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{context} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{context} must be a regular non-symlink file") from exc
    try:
        opened_status = os.fstat(descriptor)
        if not stat.S_ISREG(opened_status.st_mode):
            raise ValueError(f"{context} must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _regular_generation_files(generation: Path) -> set[str]:
    try:
        generation_status = generation.lstat()
    except OSError as exc:
        raise ValueError("selection generation directory is not readable") from exc
    if stat.S_ISLNK(generation_status.st_mode) or not stat.S_ISDIR(
        generation_status.st_mode
    ):
        raise ValueError("selection generation must be a real directory, not a symlink")
    output: set[str] = set()
    for root, directories, files in os.walk(generation, followlinks=False):
        root_path = Path(root)
        for directory in directories:
            directory_path = root_path / directory
            directory_status = directory_path.lstat()
            if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(
                directory_status.st_mode
            ):
                raise ValueError("selection payload path must not traverse a symlink")
        for filename in files:
            file_path = root_path / filename
            file_status = file_path.lstat()
            if stat.S_ISLNK(file_status.st_mode):
                raise ValueError("selection payload file must not be a symlink")
            if not stat.S_ISREG(file_status.st_mode):
                raise ValueError("selection payload file must be a regular file")
            output.add(file_path.relative_to(generation).as_posix())
    return output


class SelectionArtifactStore:
    """Crash-safe selection generation store with one exclusive directory lock."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.marker_path = self.output_directory / "selection_artifact.json"
        self.lock_path = self.output_directory / ".selection-artifact.lock"
        self.generations_directory = self.output_directory / "selection_generations"
        if fault_injector is not None and not callable(fault_injector):
            raise TypeError("fault_injector must be callable or None")
        self._fault_injector = fault_injector

    def begin(
        self,
        pending: SelectionArtifact,
        *,
        selection_mode: str = "auto",
    ) -> "SelectionArtifactTransaction":
        if not isinstance(pending, SelectionArtifact):
            raise TypeError("pending must be SelectionArtifact")
        if pending.certification_status is not CertificationStatus.PENDING:
            raise ValueError("a selection transaction must begin with PENDING")
        mode = _require_selection_mode(selection_mode)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        lock_handle = self.lock_path.open("a+b")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            generation = self._generation_directory(pending.transaction_id)
            if generation.exists():
                raise SelectionTransactionError(
                    SelectionFailureCode.TRANSACTION_SUPERSEDED,
                    "selection transaction ID already has a persisted generation",
                )
            self._atomic_write_artifact(pending)
            if self.load_current() != pending:
                raise SelectionTransactionError(
                    SelectionFailureCode.PERSISTENCE_FAILURE,
                    "PENDING selection marker failed strict reload",
                )
            self._checkpoint("after_pending_marker")
        except Exception:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            raise
        return SelectionArtifactTransaction(self, pending, mode, lock_handle)

    def load_current(self, *, require_certified: bool = False) -> SelectionArtifact:
        artifact = load_selection_artifact(self.marker_path)
        if require_certified and artifact.certification_status is not CertificationStatus.CERTIFIED:
            raise ValueError(
                f"current selection generation is not certified: "
                f"{artifact.certification_status.value}"
            )
        return artifact

    def load_current_payloads(
        self,
        *,
        require_certified: bool = True,
    ) -> dict[str, bytes]:
        lock_handle = self.lock_path.open("a+b")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            artifact = self.load_current(require_certified=require_certified)
            manifest_hash, payloads = self._load_generation(artifact.transaction_id)
            if (
                artifact.certification_status is CertificationStatus.CERTIFIED
                and artifact.payload_manifest_hash != manifest_hash
            ):
                raise ValueError(
                    "selection payload manifest does not match the certified artifact"
                )
            if self.load_current(require_certified=require_certified) != artifact:
                raise ValueError("current selection marker changed during payload snapshot")
            return payloads
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    def _generation_directory(self, transaction_id: str) -> Path:
        return self.generations_directory / _require_transaction_id(transaction_id)

    def _checkpoint(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)

    def _atomic_write_artifact(self, artifact: SelectionArtifact) -> None:
        if not isinstance(artifact, SelectionArtifact):
            raise TypeError("artifact must be SelectionArtifact")
        encoded = (
            json.dumps(
                artifact.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self._atomic_write_bytes(self.marker_path, encoded)

    def _atomic_write_bytes(self, target: Path, encoded: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_generation(self, transaction_id: str) -> tuple[str, dict[str, bytes]]:
        generation = self._generation_directory(transaction_id)
        manifest_path = generation / "payload_manifest.json"
        actual_files = _regular_generation_files(generation)
        if "payload_manifest.json" not in actual_files:
            raise ValueError("selection generation is missing its regular payload manifest")
        try:
            payload = json.loads(
                _read_regular_file(
                    manifest_path, "selection payload manifest"
                ).decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid selection payload manifest JSON") from exc
        fields = ("schema_version", "transaction_id", "payloads", "manifest_hash")
        _require_exact_keys(payload, fields, "selection payload manifest")
        if payload["schema_version"] != _PAYLOAD_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported selection payload manifest schema")
        if _require_transaction_id(payload["transaction_id"]) != transaction_id:
            raise ValueError("selection payload manifest transaction mismatch")
        records = payload["payloads"]
        if not isinstance(records, list):
            raise TypeError("selection payload manifest payloads must be a list")
        normalized_records: list[dict[str, Any]] = []
        loaded: dict[str, bytes] = {}
        for index, record in enumerate(records):
            _require_exact_keys(
                record,
                ("name", "content_hash", "size"),
                f"selection payload record {index}",
            )
            name = _require_payload_name(record["name"])
            if name in loaded:
                raise ValueError("selection payload names must be unique")
            content_hash = _require_hash(record["content_hash"], "content_hash")
            size = _require_nonnegative_integer(record["size"], "payload size")
            content = _read_regular_file(
                generation / name, f"selection payload {name}"
            )
            if len(content) != size or _payload_content_hash(content) != content_hash:
                raise ValueError(f"selection payload hash mismatch for {name}")
            loaded[name] = content
            normalized_records.append(
                {"name": name, "content_hash": content_hash, "size": size}
            )
        if [record["name"] for record in normalized_records] != sorted(loaded):
            raise ValueError("selection payload records must use canonical name order")
        if actual_files - {"payload_manifest.json"} != set(loaded):
            raise ValueError("selection generation contains unbound payload files")
        manifest_hash = _require_hash(payload["manifest_hash"], "manifest_hash")
        expected_hash = hash_mapping(
            {
                "schema_version": _PAYLOAD_MANIFEST_SCHEMA_VERSION,
                "transaction_id": transaction_id,
                "payloads": normalized_records,
            }
        )
        if manifest_hash != expected_hash:
            raise ValueError("selection payload manifest hash mismatch")
        return manifest_hash, loaded

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class SelectionArtifactTransaction:
    def __init__(
        self,
        store: SelectionArtifactStore,
        pending: SelectionArtifact,
        selection_mode: str,
        lock_handle: BinaryIO,
    ) -> None:
        self._store = store
        self._pending = pending
        self._selection_mode = selection_mode
        self._lock_handle = lock_handle
        self._active = True
        self._published = False
        self._payload_manifest_hash: str | None = None
        self._payload_count = 0

    @property
    def generation_directory(self) -> Path:
        return self._store._generation_directory(self._pending.transaction_id)

    @property
    def payload_manifest_hash(self) -> str | None:
        return self._payload_manifest_hash

    def __enter__(self) -> "SelectionArtifactTransaction":
        if not self._active:
            raise RuntimeError("selection transaction is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def stage_payloads(self, payloads: Mapping[str, bytes]) -> str:
        if not self._active:
            raise RuntimeError("selection transaction is closed")
        if self._payload_manifest_hash is not None:
            raise SelectionTransactionError(
                SelectionFailureCode.PERSISTENCE_FAILURE,
                "selection payloads have already been staged",
            )
        if not isinstance(payloads, Mapping):
            raise TypeError("payloads must be a mapping")
        normalized: dict[str, bytes] = {}
        for raw_name, raw_content in payloads.items():
            name = _require_payload_name(raw_name)
            if name in normalized:
                raise ValueError("selection payload names must be unique")
            if not isinstance(raw_content, bytes):
                raise TypeError("selection payload values must be bytes")
            normalized[name] = raw_content
        generation = self.generation_directory
        try:
            self._store.generations_directory.mkdir(parents=True, exist_ok=True)
            generation.mkdir()
            self._store._fsync_directory(self._store.generations_directory)
            records: list[dict[str, Any]] = []
            for name in sorted(normalized):
                content = normalized[name]
                self._store._atomic_write_bytes(generation / name, content)
                records.append(
                    {
                        "name": name,
                        "content_hash": _payload_content_hash(content),
                        "size": len(content),
                    }
                )
            self._store._checkpoint("after_payloads_replaced")
            manifest_body = {
                "schema_version": _PAYLOAD_MANIFEST_SCHEMA_VERSION,
                "transaction_id": self._pending.transaction_id,
                "payloads": records,
            }
            manifest_hash = hash_mapping(manifest_body)
            encoded = (
                json.dumps(
                    {**manifest_body, "manifest_hash": manifest_hash},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            self._store._atomic_write_bytes(
                generation / "payload_manifest.json", encoded
            )
            self._store._checkpoint("after_payload_manifest")
            reloaded_hash, reloaded_payloads = self._store._load_generation(
                self._pending.transaction_id
            )
            if reloaded_hash != manifest_hash or reloaded_payloads != normalized:
                raise ValueError("selection payload generation failed strict reload")
        except SelectionTransactionError:
            raise
        except Exception as exc:
            raise SelectionTransactionError(
                SelectionFailureCode.PERSISTENCE_FAILURE,
                "selection payload generation failed persistence validation",
            ) from exc
        self._payload_manifest_hash = manifest_hash
        self._payload_count = len(normalized)
        return manifest_hash

    def publish(self, artifact: SelectionArtifact) -> None:
        if not self._active:
            raise RuntimeError("selection transaction is closed")
        if self._published:
            raise SelectionTransactionError(
                SelectionFailureCode.PERSISTENCE_FAILURE,
                "selection transaction already published a final marker",
            )
        if not isinstance(artifact, SelectionArtifact):
            raise TypeError("artifact must be SelectionArtifact")
        if artifact.transaction_id != self._pending.transaction_id:
            raise SelectionTransactionError(
                SelectionFailureCode.IDENTITY_MISMATCH,
                "final selection transaction ID does not match PENDING",
            )
        allowed = (
            {CertificationStatus.CERTIFIED, CertificationStatus.FAILED}
            if self._selection_mode == "auto"
            else {CertificationStatus.UNVERIFIED_OVERRIDE}
        )
        if artifact.certification_status not in allowed:
            raise SelectionTransactionError(
                SelectionFailureCode.IDENTITY_MISMATCH,
                f"illegal PENDING transition for {self._selection_mode} selection",
            )
        if (
            artifact.selection_input_identity_hash
            != self._pending.selection_input_identity_hash
        ):
            raise SelectionTransactionError(
                SelectionFailureCode.IDENTITY_MISMATCH,
                "final selection input identity does not match PENDING",
            )
        if artifact.certification_status is CertificationStatus.CERTIFIED and (
            self._payload_manifest_hash is None or self._payload_count == 0
        ):
            raise SelectionTransactionError(
                SelectionFailureCode.PERSISTENCE_FAILURE,
                "CERTIFIED selection requires an explicit nonempty payload stage",
            )
        if self._payload_manifest_hash is None:
            self.stage_payloads({})
        current = self._store.load_current()
        if current != self._pending:
            raise SelectionTransactionError(
                SelectionFailureCode.TRANSACTION_SUPERSEDED,
                "selection transaction was superseded or its PENDING marker changed",
            )
        try:
            manifest_hash, _ = self._store._load_generation(
                self._pending.transaction_id
            )
        except Exception as exc:
            raise SelectionTransactionError(
                SelectionFailureCode.PERSISTENCE_FAILURE,
                "selection payload generation failed strict reload before publication",
            ) from exc
        if manifest_hash != self._payload_manifest_hash:
            raise SelectionTransactionError(
                SelectionFailureCode.PERSISTENCE_FAILURE,
                "selection payload manifest changed before publication",
            )
        if (
            artifact.certification_status is CertificationStatus.CERTIFIED
            and artifact.payload_manifest_hash != manifest_hash
        ):
            raise SelectionTransactionError(
                SelectionFailureCode.IDENTITY_MISMATCH,
                "CERTIFIED selection artifact does not bind the staged payload manifest",
            )
        self._store._checkpoint("before_final_marker")
        try:
            self._store._atomic_write_artifact(artifact)
            if self._store.load_current() != artifact:
                raise ValueError("final selection marker failed strict reload")
        except Exception as exc:
            try:
                self._store._atomic_write_artifact(self._pending)
            except Exception:
                pass
            raise SelectionTransactionError(
                SelectionFailureCode.PERSISTENCE_FAILURE,
                "final selection marker failed persistence validation",
            ) from exc
        self._published = True

    def close(self) -> None:
        if not self._active:
            return
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_handle.close()
            self._active = False


__all__ = [
    "CertificationEvidence",
    "CertificationStatus",
    "ResolvedCandidateIdentity",
    "SELECTION_ARTIFACT_SCHEMA_VERSION",
    "SelectionArtifact",
    "SelectionArtifactStore",
    "SelectionBindingError",
    "SelectionFailureCode",
    "SelectionIdentity",
    "SelectionInputIdentity",
    "SelectionMetricEvidence",
    "SelectionTransactionError",
    "build_selection_input_identity_hash",
    "build_selection_policy_hash",
    "build_certified_gamma_selection_identity",
    "gamma_routed_ordered_q_identity_hash",
    "hash_frozen_target_window",
    "hash_validation_k_indices",
    "load_selection_artifact",
    "verify_certified_gamma_selection_artifact",
    "verify_certified_gamma_selection_identity",
]
