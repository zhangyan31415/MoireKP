"""Pure in-memory symmetry certification for projected candidate subspaces.

The evaluator consumes explicit source actions, projected states, exactified
continuum actions, and one magnetic presentation.  It deliberately performs
no artifact I/O and never searches nested diagnostic payloads for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

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
    raw_h_leakage: float
    raw_action_unitarity_residual: float
    exactification_distance: float
    intertwining_residual: float
    heff_covariance_residual: float
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
    exact_action_finite: bool
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
    residual: float
    maximum_entry: float
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
    operations: tuple[CandidateOperationCertificate, ...]
    relations: tuple[CandidateRelationCertificate, ...]
    relation_residual_max: float | None
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ProjectedPairEvaluation:
    """Shared projection kernel used by production projection and certification."""

    action: np.ndarray
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


def _fro_relative(lhs: np.ndarray, rhs: np.ndarray, denominator: np.ndarray) -> float:
    norm = float(np.linalg.norm(denominator, ord="fro"))
    if norm == 0.0:
        norm = 1.0
    return float(np.linalg.norm(lhs - rhs, ord="fro") / norm)


def _unitarity_residual(matrix: np.ndarray) -> float:
    identity = np.eye(matrix.shape[1], dtype=np.complex128)
    return _fro_relative(matrix.conj().T @ matrix, identity, identity)


def evaluate_projected_pair(
    *,
    d_full: np.ndarray,
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
    source_image = np.asarray(d_full @ source_basis, dtype=np.complex128)
    projected = np.asarray(target_u_low.conj().T @ source_image, dtype=np.complex128)
    evaluated = projected if action is None else np.asarray(action, dtype=np.complex128)
    dimension = int(evaluated.shape[0])
    subspace_residual = float(
        np.linalg.norm(source_image - target_u_low @ evaluated, ord="fro")
        / np.sqrt(max(1, dimension))
    )
    heff_covariance: float | None = None
    if compute_heff_covariance:
        if target_heff is None or source_heff is None:
            raise ValueError("Heff covariance was requested without both effective Hamiltonians")
        source_hamiltonian = source_heff.conj() if antiunitary else source_heff
        transformed = evaluated @ source_hamiltonian @ evaluated.conj().T
        heff_covariance = _fro_relative(target_heff, transformed, target_heff)
    return ProjectedPairEvaluation(
        action=projected,
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
        residual = float(np.linalg.norm(difference, ord="fro") / np.sqrt(lhs.shape[0]))
        maximum_entry = float(np.max(np.abs(difference)))
        if not np.isfinite(residual) or not np.isfinite(maximum_entry):
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
) -> CandidateOperationCertificate:
    return CandidateOperationCertificate(
        name=operation.name,
        antiunitary=operation.antiunitary,
        required_pairs=required_pairs,
        observed_pairs=operation.pairs,
        coverage_complete=set(required_pairs) == set(operation.pairs),
        exact_action_finite=False,
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
            and not duplicate_expected
            and not duplicate_observed
        )
        if not coverage_complete:
            pair_coverage_complete = False
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
        if operation.name != name:
            operation_failures.append("name_mismatch")
        if operation.antiunitary != generator.antiunitary:
            operation_failures.append("antiunitary_parity_mismatch")
        d_full = _finite_complex_matrix(operation.d_full)
        exact = _finite_complex_matrix(exactified_actions.get(name))
        if d_full is None:
            operation_failures.append("nonfinite_raw_h_action")
            failures.append(f"nonfinite_raw_h_action:{name}")
        if exact is None:
            operation_failures.append("nonfinite_exactified_action")
            failures.append(f"nonfinite_exactified_action:{name}")
        elif exact.shape[0] != exact.shape[1]:
            operation_failures.append("nonsquare_exactified_action")
            failures.append(f"nonsquare_exactified_action:{name}")
            exact = None
        if d_full is None or exact is None:
            operation_certificates.append(
                _empty_operation_certificate(
                    operation=operation,
                    required_pairs=expected_pairs,
                    failures=operation_failures,
                )
            )
            continue

        finite_exact_actions[name] = exact
        parity[name] = bool(operation.antiunitary)
        pair_certificates: list[CandidatePairCertificate] = []
        for target_index, source_index in observed_pairs:
            pair_failures: list[str] = []
            target_state = target.get(target_index)
            source_state = source.get(source_index)
            if target_state is None or source_state is None:
                pair_coverage_complete = False
                failures.append(f"missing_state:{name}:{target_index}<-{source_index}")
                operation_failures.append("missing_projection_state")
                continue
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
            exactification_distance = float(
                np.linalg.norm(raw.action - exact, ord="fro")
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
            pair_certificates.append(
                CandidatePairCertificate(
                    operation=name,
                    target_k_index=target_index,
                    source_k_index=source_index,
                    raw_h_leakage=raw.subspace_residual,
                    raw_action_unitarity_residual=raw.action_unitarity_residual,
                    exactification_distance=exactification_distance,
                    intertwining_residual=exact_metrics.subspace_residual,
                    heff_covariance_residual=float(exact_metrics.heff_covariance_residual),
                    failures=tuple(pair_failures),
                )
            )

        square_phase: complex | None = None
        square_residual: float | None = None
        if operation.antiunitary:
            square_phase = _antiunitary_square_phase(presentation, name)
            if square_phase is None:
                operation_failures.append("missing_antiunitary_square_relation")
                failures.append(f"missing_antiunitary_square_relation:{name}")
            else:
                identity = np.eye(exact.shape[0], dtype=np.complex128)
                square_residual = float(
                    np.linalg.norm(exact @ exact.conj() - square_phase * identity, ord="fro")
                    / np.sqrt(exact.shape[0])
                )
                if (
                    not np.isfinite(square_residual)
                    or square_residual > thresholds.antiunitary_square_residual
                ):
                    operation_failures.append("antiunitary_square_residual")
                    failures.append(f"antiunitary_square:{name}")
        if not coverage_complete:
            operation_failures.append("pair_coverage")
        if any(not pair.passed for pair in pair_certificates):
            operation_failures.append("pair_metrics")
        operation_certificates.append(
            CandidateOperationCertificate(
                name=name,
                antiunitary=operation.antiunitary,
                required_pairs=expected_pairs,
                observed_pairs=observed_pairs,
                coverage_complete=coverage_complete,
                exact_action_finite=True,
                pairs=tuple(pair_certificates),
                antiunitary_square_phase=square_phase,
                antiunitary_square_residual=square_residual,
                failures=tuple(dict.fromkeys(operation_failures)),
            )
        )

    relation_certificates: tuple[CandidateRelationCertificate, ...] = ()
    if set(finite_exact_actions) == required_set and set(parity) == required_set:
        dimensions = {matrix.shape for matrix in finite_exact_actions.values()}
        if len(dimensions) != 1:
            failures.append("joint_action_dimension_mismatch")
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
            except (JointExactificationError, ValueError) as exc:
                failures.append(f"joint_relation_certification:{exc}")

    failures.extend(
        f"operation:{operation.name}"
        for operation in operation_certificates
        if not operation.passed
    )
    failures = list(dict.fromkeys(failures))
    relation_residual_max = (
        max(relation.residual for relation in relation_certificates)
        if relation_certificates
        else None
    )
    status = (
        CandidateSymmetryStatus.CERTIFIED
        if not failures
        and operation_coverage_complete
        and pair_coverage_complete
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
        operations=tuple(operation_certificates),
        relations=relation_certificates,
        relation_residual_max=relation_residual_max,
        failures=tuple(failures),
    )


__all__ = [
    "CandidateOperationCertificate",
    "CandidateOperationInput",
    "CandidatePairCertificate",
    "CandidateProjectionState",
    "CandidateRelationCertificate",
    "CandidateSymmetryCertificate",
    "CandidateSymmetryStatus",
    "CandidateSymmetryThresholds",
    "ProjectedPairEvaluation",
    "certify_candidate_symmetries",
    "evaluate_projected_pair",
]
