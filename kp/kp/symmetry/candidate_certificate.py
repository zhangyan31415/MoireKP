"""Pure in-memory symmetry certification for projected candidate subspaces.

The evaluator consumes explicit source actions, projected states, exactified
continuum actions, and one magnetic presentation.  It deliberately performs
no artifact I/O and never searches nested diagnostic payloads for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ..identity import hash_array, hash_mapping

try:
    from scipy import sparse as _sparse
except Exception:  # pragma: no cover - dense inputs remain supported
    _sparse = None

from .joint_exactification import (
    BlockRouteAction,
    JointExactificationError,
    MagneticPresentation,
    certify_joint_block_actions,
)


class CandidateSymmetryStatus(str, Enum):
    """Fail-closed outcome of one candidate symmetry evaluation."""

    CERTIFIED = "certified"
    FAILED = "failed"


class CandidateJointCertificationStatus(str, Enum):
    """Whether strict joint exact-action certification ran and succeeded."""

    CERTIFIED = "certified"
    FAILED = "failed"
    NOT_RUN = "not_run"


class CandidateJointFailureCode(str, Enum):
    """Stable machine-readable reason for a failed/not-run joint certificate."""

    INCOMPLETE_ACTION_COVERAGE = "incomplete_action_coverage"
    ACTION_DIMENSION_MISMATCH = "action_dimension_mismatch"
    ACTION_NOT_UNITARY = "action_not_unitary"
    RELATION_CERTIFICATION_FAILED = "relation_certification_failed"
    JOINT_CERTIFICATION_FAILED = "joint_certification_failed"


@dataclass(frozen=True)
class CandidateProjectionState:
    """The in-memory projected basis and effective Hamiltonian at one k point."""

    u_low: np.ndarray
    heff: np.ndarray


@dataclass(frozen=True)
class CandidateOperationInput:
    """One raw-H source action and the k-pairs on which it was observed."""

    name: str
    antiunitary: bool
    d_full: object
    pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        pairs = tuple((int(target), int(source)) for target, source in self.pairs)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "antiunitary", bool(self.antiunitary))
        object.__setattr__(self, "pairs", pairs)


@dataclass(frozen=True)
class CandidateSymmetryThresholds:
    """Independent hard gates used by candidate certification."""

    raw_h_leakage: float
    exactification_distance: float
    intertwining_residual: float
    heff_covariance_residual: float
    relation_residual: float
    antiunitary_square_residual: float
    exact_action_unitarity_residual: float
    projection_orthonormality_residual: float
    heff_hermiticity_residual: float
    raw_h_action_unitarity_residual: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"candidate symmetry threshold {item.name} must be finite and nonnegative"
                )
            object.__setattr__(self, item.name, value)

    @classmethod
    def uniform(cls, value: float) -> "CandidateSymmetryThresholds":
        threshold = float(value)
        return cls(*(threshold for _ in fields(cls)))


@dataclass(frozen=True)
class CandidatePairCertificate:
    """Metrics for one operation route ``source k -> target k``."""

    operation: str
    target_k_index: int
    source_k_index: int
    raw_h_leakage: float | None
    raw_action_unitarity_residual: float | None
    exactification_distance: float | None
    intertwining_residual: float | None
    heff_covariance_residual: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class CandidateStateCertificate:
    """Validity and identity diagnostics for one used projected k-state."""

    role: str
    k_index: int
    projection_orthonormality_residual: float | None
    heff_hermiticity_residual: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class CandidateOperationCertificate:
    """Coverage and metrics for one unitary or antiunitary operation."""

    name: str
    antiunitary: bool
    required_pairs: tuple[tuple[int, int], ...]
    observed_pairs: tuple[tuple[int, int], ...]
    coverage_complete: bool
    raw_h_action_unitarity_residual: float | None
    exact_action_finite: bool
    exact_action_unitarity_residual: float | None
    pairs: tuple[CandidatePairCertificate, ...]
    antiunitary_square_phase: complex | None
    antiunitary_square_residual: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and all(pair.passed for pair in self.pairs)


@dataclass(frozen=True)
class CandidateRelationCertificate:
    """Residual for one declared magnetic group relation."""

    name: str
    residual: float | None
    maximum_entry: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class CandidateSymmetryCertificate:
    """Complete fail-closed certificate for a candidate projected subspace."""

    candidate_id: str
    status: CandidateSymmetryStatus
    required_operations: tuple[str, ...]
    observed_operations: tuple[str, ...]
    operation_coverage_complete: bool
    pair_coverage_complete: bool
    thresholds: CandidateSymmetryThresholds
    presentation_payload: Mapping[str, object]
    presentation_hash: str
    input_identity_hash: str
    states: tuple[CandidateStateCertificate, ...]
    operations: tuple[CandidateOperationCertificate, ...]
    relations: tuple[CandidateRelationCertificate, ...]
    relation_residual_max: float | None
    joint_certification_status: CandidateJointCertificationStatus
    joint_certification_failure: CandidateJointFailureCode | None
    joint_certification_diagnostic: str | None
    failures: tuple[str, ...]
    certificate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if hash_mapping(self.presentation_payload) != self.presentation_hash:
            raise ValueError("candidate symmetry presentation hash mismatch")
        object.__setattr__(
            self,
            "certificate_hash",
            _candidate_certificate_hash(self),
        )


def _canonical_metric(value: float | None) -> float | None:
    if value is None:
        return None
    metric = float(value)
    if not np.isfinite(metric):
        raise ValueError("candidate symmetry certificates cannot contain NaN or Inf")
    return 0.0 if metric == 0.0 else metric


def _finite_metric_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    metric = float(value)
    return metric if np.isfinite(metric) else None


def _canonical_phase(value: complex | None) -> list[float] | None:
    if value is None:
        return None
    phase = complex(value)
    return [
        _canonical_metric(float(phase.real)),
        _canonical_metric(float(phase.imag)),
    ]


def _candidate_certificate_payload(
    certificate: CandidateSymmetryCertificate,
) -> dict[str, object]:
    threshold_payload = {
        item.name: _canonical_metric(getattr(certificate.thresholds, item.name))
        for item in fields(certificate.thresholds)
    }
    operation_payloads: list[dict[str, object]] = []
    for operation in sorted(certificate.operations, key=lambda item: item.name):
        pair_payloads = [
            {
                "operation": pair.operation,
                "target_k_index": int(pair.target_k_index),
                "source_k_index": int(pair.source_k_index),
                "raw_h_leakage": _canonical_metric(pair.raw_h_leakage),
                "raw_action_unitarity_residual": _canonical_metric(
                    pair.raw_action_unitarity_residual
                ),
                "exactification_distance": _canonical_metric(
                    pair.exactification_distance
                ),
                "intertwining_residual": _canonical_metric(
                    pair.intertwining_residual
                ),
                "heff_covariance_residual": _canonical_metric(
                    pair.heff_covariance_residual
                ),
                "failures": sorted(pair.failures),
            }
            for pair in sorted(
                operation.pairs,
                key=lambda item: (
                    item.operation,
                    item.target_k_index,
                    item.source_k_index,
                ),
            )
        ]
        operation_payloads.append(
            {
                "name": operation.name,
                "antiunitary": bool(operation.antiunitary),
                "required_pairs": [list(pair) for pair in sorted(operation.required_pairs)],
                "observed_pairs": [list(pair) for pair in sorted(operation.observed_pairs)],
                "coverage_complete": bool(operation.coverage_complete),
                "raw_h_action_unitarity_residual": _canonical_metric(
                    operation.raw_h_action_unitarity_residual
                ),
                "exact_action_finite": bool(operation.exact_action_finite),
                "exact_action_unitarity_residual": _canonical_metric(
                    operation.exact_action_unitarity_residual
                ),
                "pairs": pair_payloads,
                "antiunitary_square_phase": _canonical_phase(
                    operation.antiunitary_square_phase
                ),
                "antiunitary_square_residual": _canonical_metric(
                    operation.antiunitary_square_residual
                ),
                "failures": sorted(operation.failures),
            }
        )
    relation_payloads = [
        {
            "name": relation.name,
            "residual": _canonical_metric(relation.residual),
            "maximum_entry": _canonical_metric(relation.maximum_entry),
            "failures": sorted(relation.failures),
        }
        for relation in sorted(certificate.relations, key=lambda item: item.name)
    ]
    state_payloads = [
        {
            "role": state.role,
            "k_index": int(state.k_index),
            "projection_orthonormality_residual": _canonical_metric(
                state.projection_orthonormality_residual
            ),
            "heff_hermiticity_residual": _canonical_metric(
                state.heff_hermiticity_residual
            ),
            "failures": sorted(state.failures),
        }
        for state in sorted(
            certificate.states,
            key=lambda item: (item.role, item.k_index),
        )
    ]
    return {
        "version": "candidate_symmetry_certificate_v2",
        "candidate_id": certificate.candidate_id,
        "status": certificate.status.value,
        "required_operations": sorted(certificate.required_operations),
        "observed_operations": sorted(certificate.observed_operations),
        "operation_coverage_complete": bool(
            certificate.operation_coverage_complete
        ),
        "pair_coverage_complete": bool(certificate.pair_coverage_complete),
        "thresholds": threshold_payload,
        "presentation_hash": certificate.presentation_hash,
        "input_identity_hash": certificate.input_identity_hash,
        "states": state_payloads,
        "operations": operation_payloads,
        "relations": relation_payloads,
        "relation_residual_max": _canonical_metric(
            certificate.relation_residual_max
        ),
        "joint_certification_status": certificate.joint_certification_status.value,
        "joint_certification_failure": (
            None
            if certificate.joint_certification_failure is None
            else certificate.joint_certification_failure.value
        ),
        "failures": sorted(certificate.failures),
    }


def _candidate_certificate_hash(
    certificate: CandidateSymmetryCertificate,
) -> str:
    payload = _candidate_certificate_payload(certificate)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_payload(value: complex) -> list[float]:
    phase = complex(value)
    return [float(phase.real), float(phase.imag)]


def _canonical_presentation_payload(
    presentation: MagneticPresentation,
) -> dict[str, object]:
    """Return the versioned magnetic contract; ``source`` is descriptive only."""

    return {
        "version": "kp.candidate-symmetry-presentation.v1",
        "source_identity_bound": False,
        "generators": [
            {
                "name": generator.name,
                "antiunitary": bool(generator.antiunitary),
            }
            for generator in sorted(
                presentation.generators,
                key=lambda item: item.name,
            )
        ],
        "relations": [
            {
                "name": relation.name,
                "lhs": list(relation.lhs),
                "rhs": list(relation.rhs),
                "central_phase": _phase_payload(relation.central_phase),
            }
            for relation in sorted(
                presentation.relations,
                key=lambda item: item.name,
            )
        ],
        "central_phases": [
            _phase_payload(phase)
            for phase in sorted(
                presentation.central_phases,
                key=lambda value: (complex(value).real, complex(value).imag),
            )
        ],
    }


def _complex_array_identity_hash(value: object) -> str:
    if value is None:
        return hash_mapping({"version": "kp.candidate-array-input.v1", "status": "missing"})
    try:
        array = np.asarray(value, dtype=np.dtype("<c16"))
    except (TypeError, ValueError):
        return hash_mapping(
            {
                "version": "kp.candidate-array-input.v1",
                "status": "unhashable",
                "type": type(value).__name__,
            }
        )
    return hash_array(array)


def _raw_action_identity_hash(value: object) -> str:
    """Hash dense/sparse raw actions through one canonical sparse coordinate stream."""

    if value is None:
        return hash_mapping({"version": "kp.raw-action-coordinate.v1", "status": "missing"})
    try:
        if _sparse is not None and _sparse.issparse(value):
            coordinate = value.tocoo(copy=True)
            coordinate.sum_duplicates()
            shape = tuple(int(item) for item in coordinate.shape)
            row = np.asarray(coordinate.row, dtype=np.dtype("<i8"))
            column = np.asarray(coordinate.col, dtype=np.dtype("<i8"))
            data = np.asarray(coordinate.data, dtype=np.dtype("<c16"))
            keep = data != 0.0
            row = row[keep]
            column = column[keep]
            data = data[keep]
            order = np.lexsort((column, row))
            row = row[order]
            column = column[order]
            data = data[order]
        else:
            dense = np.asarray(value, dtype=np.dtype("<c16"))
            if dense.ndim != 2:
                raise ValueError("raw action must be two-dimensional")
            shape = tuple(int(item) for item in dense.shape)
            row, column = np.nonzero(dense != 0.0)
            row = np.asarray(row, dtype=np.dtype("<i8"))
            column = np.asarray(column, dtype=np.dtype("<i8"))
            data = np.asarray(dense[row, column], dtype=np.dtype("<c16"))
    except (TypeError, ValueError):
        return hash_mapping(
            {
                "version": "kp.raw-action-coordinate.v1",
                "status": "unhashable",
                "type": type(value).__name__,
            }
        )
    return hash_mapping(
        {
            "version": "kp.raw-action-coordinate.v1",
            "shape": list(shape),
            "row_hash": hash_array(row),
            "column_hash": hash_array(column),
            "data_hash": hash_array(data),
        }
    )


def _candidate_input_identity_hash(
    *,
    presentation_hash: str,
    target_states: Mapping[int, CandidateProjectionState],
    source_states: Mapping[int, CandidateProjectionState],
    operations: Mapping[str, CandidateOperationInput],
    exactified_actions: Mapping[str, np.ndarray | None],
    required_pairs: Mapping[str, Sequence[tuple[int, int]]],
) -> str:
    operation_records: list[dict[str, object]] = []
    for key in sorted(set(operations) | set(exactified_actions)):
        operation = operations.get(key)
        operation_records.append(
            {
                "key": str(key),
                "name": None if operation is None else operation.name,
                "antiunitary": (
                    None if operation is None else bool(operation.antiunitary)
                ),
                "pairs": (
                    []
                    if operation is None
                    else [list(pair) for pair in sorted(operation.pairs)]
                ),
                "raw_action_hash": _raw_action_identity_hash(
                    None if operation is None else operation.d_full
                ),
                "exact_action_hash": _complex_array_identity_hash(
                    exactified_actions.get(key)
                ),
            }
        )
    state_references: set[tuple[str, int]] = set()
    for operation in operations.values():
        for target_index, source_index in operation.pairs:
            state_references.add(("target", int(target_index)))
            state_references.add(("source", int(source_index)))
    state_records: list[dict[str, object]] = []
    for role, index in sorted(state_references):
        mapping = target_states if role == "target" else source_states
        state = mapping.get(index)
        state_records.append(
            {
                "role": role,
                "k_index": index,
                "present": state is not None,
                "u_low_hash": _complex_array_identity_hash(
                    None if state is None else state.u_low
                ),
                "heff_hash": _complex_array_identity_hash(
                    None if state is None else state.heff
                ),
            }
        )
    required_pair_records = {
        str(name): [
            [int(target_index), int(source_index)]
            for target_index, source_index in sorted(pairs)
        ]
        for name, pairs in sorted(required_pairs.items())
    }
    return hash_mapping(
        {
            "version": "kp.candidate-symmetry-input.v1",
            "presentation_hash": presentation_hash,
            "required_pairs": required_pair_records,
            "operations": operation_records,
            "states": state_records,
        }
    )


@dataclass(frozen=True)
class ProjectedPairEvaluation:
    """Shared projection kernel used by production projection and certification."""

    projected_action: np.ndarray
    evaluated_action: np.ndarray
    source_image: np.ndarray
    subspace_residual: float
    action_unitarity_residual: float
    heff_covariance_residual: float | None


def _finite_complex_matrix(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "toarray"):
            value = value.toarray()
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _finite_raw_h_action(value: object) -> object | None:
    """Validate a raw-H action without materializing a sparse full matrix."""

    if value is None:
        return None
    if _sparse is not None and _sparse.issparse(value):
        if len(value.shape) != 2 or not np.all(np.isfinite(value.data)):
            return None
        return value
    return _finite_complex_matrix(value)


def _raw_action_unitarity_residual(value: object) -> float | None:
    if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
        return None
    dimension = int(value.shape[0])
    if dimension <= 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        if _sparse is not None and _sparse.issparse(value):
            product = value.conjugate().transpose() @ value
            defect = product - _sparse.identity(
                dimension,
                dtype=np.complex128,
                format="csr",
            )
            residual = float(
                np.sqrt(np.sum(np.square(np.abs(defect.data))))
                / np.sqrt(dimension)
            )
        else:
            residual = _unitarity_residual(np.asarray(value, dtype=np.complex128))
    return _finite_metric_or_none(residual)


def _fro_relative(lhs: np.ndarray, rhs: np.ndarray, denominator: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.linalg.norm(denominator, ord="fro"))
    if norm == 0.0:
        norm = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.linalg.norm(lhs - rhs, ord="fro") / norm)


def _unitarity_residual(matrix: np.ndarray) -> float:
    identity = np.eye(matrix.shape[1], dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        product = matrix.conj().T @ matrix
    return _fro_relative(product, identity, identity)


def _state_certificate(
    *,
    role: str,
    k_index: int,
    state: CandidateProjectionState | None,
    thresholds: CandidateSymmetryThresholds,
) -> CandidateStateCertificate:
    failures: list[str] = []
    projection_residual: float | None = None
    heff_residual: float | None = None
    if state is None:
        failures.append("missing_projection_state")
    else:
        u_low = _finite_complex_matrix(state.u_low)
        heff = _finite_complex_matrix(state.heff)
        if u_low is None:
            failures.append("nonfinite_u_low")
        elif (
            u_low.shape[0] < u_low.shape[1]
            or u_low.shape[1] <= 0
        ):
            failures.append("invalid_u_low_shape")
        else:
            projection_residual = _finite_metric_or_none(
                _unitarity_residual(u_low)
            )
            if projection_residual is None:
                failures.append("nonfinite_projection_orthonormality_residual")
            elif (
                projection_residual
                > thresholds.projection_orthonormality_residual
            ):
                failures.append("projection_orthonormality_residual")
        if heff is None:
            failures.append("nonfinite_heff")
        elif (
            heff.shape[0] != heff.shape[1]
            or u_low is None
            or heff.shape[0] != u_low.shape[1]
        ):
            failures.append("heff_dimension_mismatch")
        else:
            heff_residual = _finite_metric_or_none(
                _fro_relative(heff, heff.conj().T, heff)
            )
            if heff_residual is None:
                failures.append("nonfinite_heff_hermiticity_residual")
            elif heff_residual > thresholds.heff_hermiticity_residual:
                failures.append("heff_hermiticity_residual")
    return CandidateStateCertificate(
        role=str(role),
        k_index=int(k_index),
        projection_orthonormality_residual=projection_residual,
        heff_hermiticity_residual=heff_residual,
        failures=tuple(failures),
    )


def evaluate_projected_pair(
    *,
    d_full: object,
    target_u_low: np.ndarray,
    source_u_low: np.ndarray,
    target_heff: np.ndarray | None,
    source_heff: np.ndarray | None,
    antiunitary: bool,
    action: np.ndarray | None = None,
    compute_heff_covariance: bool = True,
) -> ProjectedPairEvaluation:
    """Project one source action and evaluate one chosen low-space action."""

    source_basis = source_u_low.conj() if antiunitary else source_u_low
    with np.errstate(over="ignore", invalid="ignore"):
        source_image_raw = d_full @ source_basis
    if _sparse is not None and _sparse.issparse(source_image_raw):
        source_image_raw = source_image_raw.toarray()
    source_image = np.asarray(source_image_raw, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):
        projected = np.asarray(
            target_u_low.conj().T @ source_image,
            dtype=np.complex128,
        )
    evaluated = projected if action is None else np.asarray(action, dtype=np.complex128)
    dimension = int(evaluated.shape[0])
    with np.errstate(over="ignore", invalid="ignore"):
        subspace_residual = float(
            np.linalg.norm(source_image - target_u_low @ evaluated, ord="fro")
            / np.sqrt(max(1, dimension))
        )
    heff_covariance: float | None = None
    if compute_heff_covariance:
        if target_heff is None or source_heff is None:
            raise ValueError("Heff covariance was requested without both effective Hamiltonians")
        source_hamiltonian = source_heff.conj() if antiunitary else source_heff
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = evaluated @ source_hamiltonian @ evaluated.conj().T
        heff_covariance = _fro_relative(target_heff, transformed, target_heff)
    return ProjectedPairEvaluation(
        projected_action=projected,
        evaluated_action=evaluated,
        source_image=source_image,
        subspace_residual=subspace_residual,
        action_unitarity_residual=_unitarity_residual(evaluated),
        heff_covariance_residual=heff_covariance,
    )


def _evaluate_word(
    word: Sequence[str],
    actions: Mapping[str, np.ndarray],
    antiunitary: Mapping[str, bool],
) -> tuple[np.ndarray, bool]:
    dimension = next(iter(actions.values())).shape[0]
    matrix = np.eye(dimension, dtype=np.complex128)
    parity = False
    for name in reversed(tuple(word)):
        matrix = actions[name] @ (matrix.conj() if antiunitary[name] else matrix)
        parity ^= antiunitary[name]
    return matrix, parity


def _relation_certificates(
    *,
    actions: Mapping[str, np.ndarray],
    parity: Mapping[str, bool],
    presentation: MagneticPresentation,
    threshold: float,
) -> tuple[tuple[CandidateRelationCertificate, ...], list[str]]:
    certificates: list[CandidateRelationCertificate] = []
    failures: list[str] = []
    if not actions:
        return (), failures
    for relation in presentation.relations:
        lhs, lhs_parity = _evaluate_word(relation.lhs, actions, parity)
        rhs, rhs_parity = _evaluate_word(relation.rhs, actions, parity)
        relation_failures: list[str] = []
        if lhs_parity != rhs_parity:
            relation_failures.append("semilinear_parity")
        difference = lhs - relation.central_phase * rhs
        residual_raw = float(
            np.linalg.norm(difference, ord="fro") / np.sqrt(lhs.shape[0])
        )
        maximum_entry_raw = float(np.max(np.abs(difference)))
        residual = _finite_metric_or_none(residual_raw)
        maximum_entry = _finite_metric_or_none(maximum_entry_raw)
        if residual is None or maximum_entry is None:
            relation_failures.append("nonfinite")
        elif residual > threshold:
            relation_failures.append("residual")
        if relation_failures:
            failures.append(f"relation:{relation.name}")
        certificates.append(
            CandidateRelationCertificate(
                name=relation.name,
                residual=residual,
                maximum_entry=maximum_entry,
                failures=tuple(relation_failures),
            )
        )
    return tuple(certificates), failures


def _antiunitary_square_phase(
    presentation: MagneticPresentation,
    operation: str,
) -> complex | None:
    matches = [
        relation.central_phase
        for relation in presentation.relations
        if relation.lhs == (operation, operation) and not relation.rhs
    ]
    if len(matches) != 1:
        return None
    return complex(matches[0])


def _empty_operation_certificate(
    *,
    operation: CandidateOperationInput,
    required_pairs: tuple[tuple[int, int], ...],
    failures: Sequence[str],
    raw_h_action_unitarity_residual: float | None = None,
) -> CandidateOperationCertificate:
    required_set = set(required_pairs)
    observed_set = set(operation.pairs)
    coverage_complete = bool(
        required_pairs
        and operation.pairs
        and required_set == observed_set
        and len(required_set) == len(required_pairs)
        and len(observed_set) == len(operation.pairs)
    )
    return CandidateOperationCertificate(
        name=operation.name,
        antiunitary=operation.antiunitary,
        required_pairs=required_pairs,
        observed_pairs=operation.pairs,
        coverage_complete=coverage_complete,
        raw_h_action_unitarity_residual=raw_h_action_unitarity_residual,
        exact_action_finite=False,
        exact_action_unitarity_residual=None,
        pairs=(),
        antiunitary_square_phase=None,
        antiunitary_square_residual=None,
        failures=tuple(failures),
    )


def certify_candidate_symmetries(
    *,
    candidate_id: str,
    states: Mapping[int, CandidateProjectionState],
    operations: Mapping[str, CandidateOperationInput],
    exactified_actions: Mapping[str, np.ndarray | None],
    presentation: MagneticPresentation,
    required_pairs: Mapping[str, Sequence[tuple[int, int]]],
    thresholds: CandidateSymmetryThresholds,
    target_states: Mapping[int, CandidateProjectionState] | None = None,
    source_states: Mapping[int, CandidateProjectionState] | None = None,
) -> CandidateSymmetryCertificate:
    """Certify one candidate directly from explicit in-memory physics inputs."""

    target = states if target_states is None else target_states
    source = states if source_states is None else source_states
    presentation_payload = _canonical_presentation_payload(presentation)
    presentation_hash = hash_mapping(presentation_payload)
    input_identity_hash = _candidate_input_identity_hash(
        presentation_hash=presentation_hash,
        target_states=target,
        source_states=source,
        operations=operations,
        exactified_actions=exactified_actions,
        required_pairs=required_pairs,
    )
    state_certificate_by_role_and_k: dict[
        tuple[str, int], CandidateStateCertificate
    ] = {}
    for operation in operations.values():
        for target_index, source_index in operation.pairs:
            for role, index, mapping in (
                ("target", int(target_index), target),
                ("source", int(source_index), source),
            ):
                key = (role, index)
                if key not in state_certificate_by_role_and_k:
                    state_certificate_by_role_and_k[key] = _state_certificate(
                        role=role,
                        k_index=index,
                        state=mapping.get(index),
                        thresholds=thresholds,
                    )
    required_names = tuple(generator.name for generator in presentation.generators)
    required_set = set(required_names)
    observed_names = tuple(sorted(str(name) for name in operations))
    operation_set = set(operations)
    exact_set = set(exactified_actions)
    failures: list[str] = []
    for name in sorted(required_set - operation_set):
        failures.append(f"missing_operation:{name}")
    for name in sorted(operation_set - required_set):
        failures.append(f"unexpected_operation:{name}")
    for name in sorted(required_set - exact_set):
        failures.append(f"missing_exactified_action:{name}")
    for name in sorted(exact_set - required_set):
        failures.append(f"unexpected_exactified_action:{name}")
    if set(required_pairs) != required_set:
        for name in sorted(required_set - set(required_pairs)):
            failures.append(f"missing_required_pair_coverage:{name}")
        for name in sorted(set(required_pairs) - required_set):
            failures.append(f"unexpected_required_pair_coverage:{name}")

    operation_coverage_complete = bool(
        operation_set == required_set
        and exact_set == required_set
        and set(required_pairs) == required_set
    )
    pair_coverage_complete = True
    operation_certificates: list[CandidateOperationCertificate] = []
    finite_exact_actions: dict[str, np.ndarray] = {}
    parity: dict[str, bool] = {}

    for generator in presentation.generators:
        name = generator.name
        operation = operations.get(name)
        if operation is None:
            pair_coverage_complete = False
            continue
        expected_pairs = tuple(
            (int(target_index), int(source_index))
            for target_index, source_index in required_pairs.get(name, ())
        )
        observed_pairs = tuple(operation.pairs)
        expected_set = set(expected_pairs)
        observed_set = set(observed_pairs)
        duplicate_expected = len(expected_set) != len(expected_pairs)
        duplicate_observed = len(observed_set) != len(observed_pairs)
        coverage_complete = bool(
            expected_set == observed_set
            and expected_pairs
            and observed_pairs
            and not duplicate_expected
            and not duplicate_observed
        )
        if not coverage_complete:
            pair_coverage_complete = False
            if not expected_pairs or not observed_pairs:
                failures.append(f"empty_pair_coverage:{name}")
            for target_index, source_index in sorted(expected_set - observed_set):
                failures.append(f"missing_pair:{name}:{target_index}<-{source_index}")
            for target_index, source_index in sorted(observed_set - expected_set):
                failures.append(f"unexpected_pair:{name}:{target_index}<-{source_index}")
            if duplicate_expected:
                failures.append(f"duplicate_required_pair:{name}")
            if duplicate_observed:
                for target_index, source_index in sorted(
                    pair
                    for pair in observed_set
                    if observed_pairs.count(pair) > 1
                ):
                    failures.append(
                        f"duplicate_pair:{name}:{target_index}<-{source_index}"
                    )

        operation_failures: list[str] = []
        if not expected_pairs or not observed_pairs:
            operation_failures.append("empty_pair_coverage")
        if operation.name != name:
            operation_failures.append("name_mismatch")
        if operation.antiunitary != generator.antiunitary:
            operation_failures.append("antiunitary_parity_mismatch")
        d_full = _finite_raw_h_action(operation.d_full)
        exact = _finite_complex_matrix(exactified_actions.get(name))
        raw_h_action_unitarity: float | None = None
        if d_full is None:
            operation_failures.append("nonfinite_raw_h_action")
            failures.append(f"nonfinite_raw_h_action:{name}")
        elif d_full.shape[0] != d_full.shape[1] or d_full.shape[0] <= 0:
            operation_failures.append("nonsquare_raw_h_action")
            failures.append(f"nonsquare_raw_h_action:{name}")
            d_full = None
        else:
            raw_h_action_unitarity = _raw_action_unitarity_residual(d_full)
            if raw_h_action_unitarity is None:
                operation_failures.append(
                    "nonfinite_raw_h_action_unitarity_residual"
                )
                failures.append(
                    f"nonfinite_raw_h_action_unitarity_residual:{name}"
                )
            elif (
                raw_h_action_unitarity
                > thresholds.raw_h_action_unitarity_residual
            ):
                operation_failures.append("raw_h_action_unitarity_residual")
        if exact is None:
            operation_failures.append("nonfinite_exactified_action")
            failures.append(f"nonfinite_exactified_action:{name}")
        elif exact.shape[0] != exact.shape[1]:
            operation_failures.append("nonsquare_exactified_action")
            failures.append(f"nonsquare_exactified_action:{name}")
            exact = None
        if d_full is None or exact is None:
            pair_coverage_complete = False
            for target_index, source_index in sorted(expected_set):
                failures.append(
                    f"unevaluated_pair:{name}:{target_index}<-{source_index}"
            )
            if expected_pairs:
                operation_failures.append("no_evaluated_required_pair")
            operation_certificates.append(
                _empty_operation_certificate(
                    operation=operation,
                    required_pairs=expected_pairs,
                    failures=operation_failures,
                    raw_h_action_unitarity_residual=raw_h_action_unitarity,
                )
            )
            continue

        finite_exact_actions[name] = exact
        parity[name] = bool(operation.antiunitary)
        exact_action_unitarity = _unitarity_residual(exact)
        if not np.isfinite(exact_action_unitarity):
            exact_action_unitarity = None
            operation_failures.append("nonfinite_exact_action_unitarity_residual")
            failures.append(f"nonfinite_exact_action_unitarity_residual:{name}")
        elif (
            exact_action_unitarity
            > thresholds.exact_action_unitarity_residual
        ):
            operation_failures.append("exact_action_unitarity_residual")
        pair_certificates: list[CandidatePairCertificate] = []
        evaluated_pairs: set[tuple[int, int]] = set()
        for target_index, source_index in observed_pairs:
            pair_failures: list[str] = []
            target_state = target.get(target_index)
            source_state = source.get(source_index)
            target_state_certificate = state_certificate_by_role_and_k[
                ("target", int(target_index))
            ]
            source_state_certificate = state_certificate_by_role_and_k[
                ("source", int(source_index))
            ]
            if (
                not target_state_certificate.passed
                or not source_state_certificate.passed
            ):
                pair_coverage_complete = False
                if not target_state_certificate.passed:
                    failures.append(f"invalid_state:target:{target_index}")
                if not source_state_certificate.passed:
                    failures.append(f"invalid_state:source:{source_index}")
                operation_failures.append("invalid_projection_state")
                continue
            assert target_state is not None
            assert source_state is not None
            target_u = _finite_complex_matrix(target_state.u_low)
            source_u = _finite_complex_matrix(source_state.u_low)
            target_heff = _finite_complex_matrix(target_state.heff)
            source_heff = _finite_complex_matrix(source_state.heff)
            if any(
                value is None
                for value in (target_u, source_u, target_heff, source_heff)
            ):
                failures.append(f"nonfinite_state:{name}:{target_index}<-{source_index}")
                operation_failures.append("nonfinite_projection_state")
                continue
            assert target_u is not None
            assert source_u is not None
            assert target_heff is not None
            assert source_heff is not None
            if (
                d_full.shape[1] != source_u.shape[0]
                or d_full.shape[0] != target_u.shape[0]
                or exact.shape != (target_u.shape[1], source_u.shape[1])
                or target_heff.shape != (target_u.shape[1], target_u.shape[1])
                or source_heff.shape != (source_u.shape[1], source_u.shape[1])
            ):
                failures.append(f"dimension_mismatch:{name}:{target_index}<-{source_index}")
                operation_failures.append("dimension_mismatch")
                continue
            raw = evaluate_projected_pair(
                d_full=d_full,
                target_u_low=target_u,
                source_u_low=source_u,
                target_heff=target_heff,
                source_heff=source_heff,
                antiunitary=operation.antiunitary,
            )
            exact_metrics = evaluate_projected_pair(
                d_full=d_full,
                target_u_low=target_u,
                source_u_low=source_u,
                target_heff=target_heff,
                source_heff=source_heff,
                antiunitary=operation.antiunitary,
                action=exact,
            )
            dimension = int(exact.shape[1])
            with np.errstate(over="ignore", invalid="ignore"):
                exactification_distance = float(
                    np.linalg.norm(raw.projected_action - exact, ord="fro")
                    / np.sqrt(max(1, dimension))
                )
            values = {
                "raw_h_leakage": raw.subspace_residual,
                "exactification_distance": exactification_distance,
                "intertwining_residual": exact_metrics.subspace_residual,
                "heff_covariance_residual": exact_metrics.heff_covariance_residual,
            }
            for metric_name, value in values.items():
                threshold = float(getattr(thresholds, metric_name))
                if value is None or not np.isfinite(float(value)) or float(value) > threshold:
                    pair_failures.append(metric_name)
            raw_action_unitarity = _finite_metric_or_none(
                raw.action_unitarity_residual
            )
            if raw_action_unitarity is None:
                pair_failures.append("nonfinite_raw_action_unitarity_residual")
            finite_values = {
                metric_name: _finite_metric_or_none(value)
                for metric_name, value in values.items()
            }
            pair_certificates.append(
                CandidatePairCertificate(
                    operation=name,
                    target_k_index=target_index,
                    source_k_index=source_index,
                    raw_h_leakage=finite_values["raw_h_leakage"],
                    raw_action_unitarity_residual=raw_action_unitarity,
                    exactification_distance=finite_values[
                        "exactification_distance"
                    ],
                    intertwining_residual=finite_values[
                        "intertwining_residual"
                    ],
                    heff_covariance_residual=finite_values[
                        "heff_covariance_residual"
                    ],
                    failures=tuple(dict.fromkeys(pair_failures)),
                )
            )
            evaluated_pairs.add((int(target_index), int(source_index)))

        square_phase: complex | None = None
        square_residual: float | None = None
        if operation.antiunitary:
            square_phase = _antiunitary_square_phase(presentation, name)
            if square_phase is None:
                operation_failures.append("missing_antiunitary_square_relation")
                failures.append(f"missing_antiunitary_square_relation:{name}")
            else:
                identity = np.eye(exact.shape[0], dtype=np.complex128)
                with np.errstate(over="ignore", invalid="ignore"):
                    square_residual_raw = float(
                        np.linalg.norm(
                            exact @ exact.conj() - square_phase * identity,
                            ord="fro",
                        )
                        / np.sqrt(exact.shape[0])
                    )
                square_residual = _finite_metric_or_none(square_residual_raw)
                if (
                    square_residual is None
                    or square_residual > thresholds.antiunitary_square_residual
                ):
                    operation_failures.append("antiunitary_square_residual")
                    failures.append(f"antiunitary_square:{name}")
        if not coverage_complete:
            operation_failures.append("pair_coverage")
        for target_index, source_index in sorted(expected_set - evaluated_pairs):
            pair_coverage_complete = False
            failures.append(f"unevaluated_pair:{name}:{target_index}<-{source_index}")
        if expected_pairs and not (expected_set & evaluated_pairs):
            operation_failures.append("no_evaluated_required_pair")
        if any(not pair.passed for pair in pair_certificates):
            operation_failures.append("pair_metrics")
        operation_certificates.append(
            CandidateOperationCertificate(
                name=name,
                antiunitary=operation.antiunitary,
                required_pairs=expected_pairs,
                observed_pairs=observed_pairs,
                coverage_complete=coverage_complete,
                raw_h_action_unitarity_residual=raw_h_action_unitarity,
                exact_action_finite=True,
                exact_action_unitarity_residual=exact_action_unitarity,
                pairs=tuple(pair_certificates),
                antiunitary_square_phase=square_phase,
                antiunitary_square_residual=square_residual,
                failures=tuple(dict.fromkeys(operation_failures)),
            )
        )

    relation_certificates: tuple[CandidateRelationCertificate, ...] = ()
    joint_certification_status = CandidateJointCertificationStatus.NOT_RUN
    joint_certification_failure: CandidateJointFailureCode | None = (
        CandidateJointFailureCode.INCOMPLETE_ACTION_COVERAGE
    )
    joint_certification_diagnostic: str | None = None
    if set(finite_exact_actions) == required_set and set(parity) == required_set:
        dimensions = {matrix.shape for matrix in finite_exact_actions.values()}
        if len(dimensions) != 1:
            failures.append("joint_action_dimension_mismatch")
            joint_certification_status = CandidateJointCertificationStatus.FAILED
            joint_certification_failure = (
                CandidateJointFailureCode.ACTION_DIMENSION_MISMATCH
            )
        else:
            relation_certificates, relation_failures = _relation_certificates(
                actions=finite_exact_actions,
                parity=parity,
                presentation=presentation,
                threshold=thresholds.relation_residual,
            )
            failures.extend(relation_failures)
            dimension = next(iter(dimensions))[0]
            try:
                joint_actions = {
                    name: BlockRouteAction(
                        name=name,
                        antiunitary=parity[name],
                        fiber_permutation=(0,),
                        fiber_dimensions=(dimension,),
                        route_blocks=(matrix,),
                    )
                    for name, matrix in finite_exact_actions.items()
                }
                certify_joint_block_actions(joint_actions, presentation)
                joint_certification_status = (
                    CandidateJointCertificationStatus.CERTIFIED
                )
                joint_certification_failure = None
            except (JointExactificationError, ValueError) as exc:
                joint_certification_status = CandidateJointCertificationStatus.FAILED
                if any(not relation.passed for relation in relation_certificates):
                    joint_certification_failure = (
                        CandidateJointFailureCode.RELATION_CERTIFICATION_FAILED
                    )
                elif any(
                    operation.exact_action_unitarity_residual is None
                    or operation.exact_action_unitarity_residual > 0.0
                    for operation in operation_certificates
                ):
                    joint_certification_failure = (
                        CandidateJointFailureCode.ACTION_NOT_UNITARY
                    )
                else:
                    joint_certification_failure = (
                        CandidateJointFailureCode.JOINT_CERTIFICATION_FAILED
                    )
                joint_certification_diagnostic = str(exc)

    if joint_certification_status is not CandidateJointCertificationStatus.CERTIFIED:
        assert joint_certification_failure is not None
        failures.append(
            f"joint_certification:{joint_certification_failure.value}"
        )

    failures.extend(
        f"operation:{operation.name}"
        for operation in operation_certificates
        if not operation.passed
    )
    failures = list(dict.fromkeys(failures))
    finite_relation_residuals = tuple(
        relation.residual
        for relation in relation_certificates
        if relation.residual is not None
    )
    relation_residual_max = (
        max(finite_relation_residuals) if finite_relation_residuals else None
    )
    status = (
        CandidateSymmetryStatus.CERTIFIED
        if not failures
        and operation_coverage_complete
        and pair_coverage_complete
        and joint_certification_status
        is CandidateJointCertificationStatus.CERTIFIED
        and len(operation_certificates) == len(required_names)
        and len(relation_certificates) == len(presentation.relations)
        else CandidateSymmetryStatus.FAILED
    )
    return CandidateSymmetryCertificate(
        candidate_id=str(candidate_id),
        status=status,
        required_operations=required_names,
        observed_operations=observed_names,
        operation_coverage_complete=operation_coverage_complete,
        pair_coverage_complete=pair_coverage_complete,
        thresholds=thresholds,
        presentation_payload=presentation_payload,
        presentation_hash=presentation_hash,
        input_identity_hash=input_identity_hash,
        states=tuple(
            state_certificate_by_role_and_k[key]
            for key in sorted(state_certificate_by_role_and_k)
        ),
        operations=tuple(operation_certificates),
        relations=relation_certificates,
        relation_residual_max=relation_residual_max,
        joint_certification_status=joint_certification_status,
        joint_certification_failure=joint_certification_failure,
        joint_certification_diagnostic=joint_certification_diagnostic,
        failures=tuple(failures),
    )


__all__ = [
    "CandidateJointCertificationStatus",
    "CandidateJointFailureCode",
    "CandidateOperationCertificate",
    "CandidateOperationInput",
    "CandidatePairCertificate",
    "CandidateProjectionState",
    "CandidateRelationCertificate",
    "CandidateStateCertificate",
    "CandidateSymmetryCertificate",
    "CandidateSymmetryStatus",
    "CandidateSymmetryThresholds",
    "ProjectedPairEvaluation",
    "certify_candidate_symmetries",
    "evaluate_projected_pair",
]
