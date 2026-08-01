"""Joint certificates for continuum fiber-gauge structure.

This module deliberately classifies route blocks without choosing an eigenframe.
Degenerate operations can therefore protect an existing factorization even when
they cannot resolve an internal basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


JOINT_STRUCTURE_CERTIFICATE_V1 = "joint_structure_certificate_v1"
COMMON_FIBER_GAUGE_WITNESS_V1 = "common_fiber_gauge_witness_v1"
STRUCTURE_TRANSITION_V1 = "joint_structure_transition_v1"


class StructureGrade(IntEnum):
    """Ordered continuum-route complexity; larger values are simpler."""

    GENERAL_UD_ROUTE = 0
    PROJECTIVELY_UNIFORM = 1
    UNIFORM_INTERNAL = 2


class CertificationStatus(str, Enum):
    """Proof status; solver failure is intentionally distinct from infeasibility."""

    CERTIFIED = "certified"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNKNOWN_OR_UNSUPPORTED = "unknown_or_unsupported"


_GRADE_LABELS = {
    StructureGrade.GENERAL_UD_ROUTE: "general_Ud_route",
    StructureGrade.PROJECTIVELY_UNIFORM: "projectively_uniform",
    StructureGrade.UNIFORM_INTERNAL: "uniform_internal",
}


def _grade_label(value: StructureGrade | None) -> str | None:
    return None if value is None else _GRADE_LABELS[StructureGrade(value)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, Enum):
        return value.value
    return value


def _deep_freeze(value: Any) -> Any:
    """Detach nested certificate payloads from caller-owned mutable objects."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        return tuple(_deep_freeze(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matrix_hash(matrix: np.ndarray) -> str:
    value = np.asarray(matrix, dtype="<c16", order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes(order="C"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _gamma(operation_count: int) -> float:
    product = float(max(1, int(operation_count))) * np.finfo(np.float64).eps
    if product >= 1.0:
        raise ValueError("floating-point certification bound overflow")
    return float(product / (1.0 - product))


@dataclass(frozen=True)
class FiberSectorLayout:
    """Source-sector ownership for every continuum fiber."""

    sector_names: tuple[str, ...]
    fiber_sectors: tuple[int, ...]
    basis_frame_hash: str | None = None
    declared_phase_signature: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.sector_names)
        sectors = tuple(int(value) for value in self.fiber_sectors)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("fiber-sector layout requires unique nonempty sector names")
        if not sectors or any(value < 0 or value >= len(names) for value in sectors):
            raise ValueError("fiber-sector layout contains an invalid sector index")
        if set(sectors) != set(range(len(names))):
            raise ValueError("every declared fiber sector must own at least one fiber")
        object.__setattr__(self, "sector_names", names)
        object.__setattr__(self, "fiber_sectors", sectors)
        object.__setattr__(
            self,
            "basis_frame_hash",
            None if self.basis_frame_hash in (None, "") else str(self.basis_frame_hash),
        )
        signature = {} if self.declared_phase_signature is None else self.declared_phase_signature
        object.__setattr__(self, "declared_phase_signature", _deep_freeze(signature))

    def artifact(self) -> dict[str, Any]:
        return {
            "sector_names": list(self.sector_names),
            "fiber_sectors": list(self.fiber_sectors),
            "basis_frame_hash": self.basis_frame_hash,
            "declared_phase_signature": _jsonable(self.declared_phase_signature),
        }


@dataclass(frozen=True)
class SectorStructureCertificate:
    operation: str
    antiunitary: bool
    source_sector: str
    target_sector: str | None
    status: CertificationStatus
    grade: StructureGrade | None
    source_fibers: tuple[int, ...]
    reference_fiber: int | None
    reference_block_hash: str | None
    relative_phases: tuple[complex | None, ...]
    uniform_residual_max: float | None
    projective_residual_max: float | None
    minimum_overlap_magnitude: float | None
    certification_bound_max: float | None
    obstruction_witness: Mapping[str, Any] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.obstruction_witness is not None:
            object.__setattr__(
                self,
                "obstruction_witness",
                _deep_freeze(self.obstruction_witness),
            )

    def artifact(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "antiunitary": bool(self.antiunitary),
            "source_sector": self.source_sector,
            "target_sector": self.target_sector,
            "status": self.status.value,
            "grade": _grade_label(self.grade),
            "source_fibers": list(self.source_fibers),
            "reference_fiber": self.reference_fiber,
            "reference_block_hash": self.reference_block_hash,
            "relative_phases": [
                None if phase is None else [float(phase.real), float(phase.imag)]
                for phase in self.relative_phases
            ],
            "uniform_residual_max": (
                None
                if self.uniform_residual_max is None
                else float(self.uniform_residual_max)
            ),
            "projective_residual_max": (
                None
                if self.projective_residual_max is None
                else float(self.projective_residual_max)
            ),
            "minimum_overlap_magnitude": (
                None
                if self.minimum_overlap_magnitude is None
                else float(self.minimum_overlap_magnitude)
            ),
            "certification_bound_max": (
                None
                if self.certification_bound_max is None
                else float(self.certification_bound_max)
            ),
            "obstruction_witness": _jsonable(self.obstruction_witness),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OperationStructureCertificate:
    name: str
    antiunitary: bool
    status: CertificationStatus
    grade: StructureGrade | None
    fiber_permutation: tuple[int, ...]
    fiber_dimensions: tuple[int, ...]
    fiber_indices: tuple[tuple[int, ...], ...]
    source_sectors: Mapping[str, SectorStructureCertificate]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_sectors", MappingProxyType(dict(self.source_sectors)))

    def artifact(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "antiunitary": bool(self.antiunitary),
            "status": self.status.value,
            "grade": _grade_label(self.grade),
            "fiber_permutation": list(self.fiber_permutation),
            "fiber_dimensions": list(self.fiber_dimensions),
            "fiber_indices": [list(value) for value in self.fiber_indices],
            "sectors": [
                certificate.artifact()
                for certificate in self.source_sectors.values()
            ],
        }


@dataclass(frozen=True)
class JointStructureCertificate:
    status: CertificationStatus
    basis_frame_hash: str | None
    layout_hash: str
    presentation_hash: str
    declared_cocycle_hash: str
    actions_hash: str
    operations: Mapping[str, OperationStructureCertificate]
    relation_certificate: Mapping[str, Any]
    certificate_hash: str
    version: str = JOINT_STRUCTURE_CERTIFICATE_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", MappingProxyType(dict(self.operations)))
        object.__setattr__(
            self,
            "relation_certificate",
            _deep_freeze(self.relation_certificate),
        )

    def artifact(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "basis_frame_hash": self.basis_frame_hash,
            "layout_hash": self.layout_hash,
            "presentation_hash": self.presentation_hash,
            "declared_cocycle_hash": self.declared_cocycle_hash,
            "actions_hash": self.actions_hash,
            "operations": {
                name: certificate.artifact()
                for name, certificate in self.operations.items()
            },
            "relation_certificate": _jsonable(self.relation_certificate),
            "certificate_hash": self.certificate_hash,
        }


@dataclass(frozen=True)
class CommonGaugeWitness:
    status: CertificationStatus
    source_basis_hash: str
    target_basis_hash: str
    source_actions_hash: str
    target_actions_hash: str
    presentation_hash: str
    gauge_hash: str
    residual_by_operation: Mapping[str, float]
    certification_bound_by_operation: Mapping[str, float]
    residual_by_operation_fiber: Mapping[str, tuple[float, ...]]
    certification_bound_by_operation_fiber: Mapping[str, tuple[float, ...]]
    max_residual: float | None
    certification_bound: float | None
    gauge_unitarity_residual_max: float
    gauge_unitarity_certification_bound: float
    reason: str | None = None
    version: str = COMMON_FIBER_GAUGE_WITNESS_V1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residual_by_operation",
            MappingProxyType({str(key): float(value) for key, value in self.residual_by_operation.items()}),
        )
        object.__setattr__(
            self,
            "certification_bound_by_operation",
            MappingProxyType(
                {str(key): float(value) for key, value in self.certification_bound_by_operation.items()}
            ),
        )
        object.__setattr__(
            self,
            "residual_by_operation_fiber",
            MappingProxyType(
                {
                    str(key): tuple(float(item) for item in value)
                    for key, value in self.residual_by_operation_fiber.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "certification_bound_by_operation_fiber",
            MappingProxyType(
                {
                    str(key): tuple(float(item) for item in value)
                    for key, value in self.certification_bound_by_operation_fiber.items()
                }
            ),
        )

    def artifact(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "source_basis_hash": self.source_basis_hash,
            "target_basis_hash": self.target_basis_hash,
            "source_actions_hash": self.source_actions_hash,
            "target_actions_hash": self.target_actions_hash,
            "presentation_hash": self.presentation_hash,
            "gauge_hash": self.gauge_hash,
            "residual_by_operation": dict(self.residual_by_operation),
            "certification_bound_by_operation": dict(self.certification_bound_by_operation),
            "residual_by_operation_fiber": {
                key: list(value)
                for key, value in self.residual_by_operation_fiber.items()
            },
            "certification_bound_by_operation_fiber": {
                key: list(value)
                for key, value in self.certification_bound_by_operation_fiber.items()
            },
            "max_residual": (
                None if self.max_residual is None else float(self.max_residual)
            ),
            "certification_bound": (
                None
                if self.certification_bound is None
                else float(self.certification_bound)
            ),
            "gauge_unitarity_residual_max": float(
                self.gauge_unitarity_residual_max
            ),
            "gauge_unitarity_certification_bound": float(
                self.gauge_unitarity_certification_bound
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StructureTransition:
    status: CertificationStatus
    is_monotone: bool
    regressions: tuple[tuple[str, str], ...]
    reason: str | None = None
    version: str = STRUCTURE_TRANSITION_V1

    def artifact(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "is_monotone": bool(self.is_monotone),
            "regressions": [list(value) for value in self.regressions],
            "reason": self.reason,
        }


def _presentation_artifact(presentation: Any) -> dict[str, Any]:
    return {
        "generators": [
            {"name": generator.name, "antiunitary": bool(generator.antiunitary)}
            for generator in presentation.generators
        ],
        "relations": [
            {
                "name": relation.name,
                "lhs": list(relation.lhs),
                "rhs": list(relation.rhs),
                "central_phase": [
                    float(relation.central_phase.real),
                    float(relation.central_phase.imag),
                ],
            }
            for relation in presentation.relations
        ],
        "central_phases": [
            [float(value.real), float(value.imag)]
            for value in presentation.central_phases
        ],
    }


def _actions_artifact(
    actions: Mapping[str, Any],
    presentation: Any,
) -> dict[str, Any]:
    """Bind a certificate or witness to the exact route-action snapshot."""

    return {
        generator.name: {
            "name": actions[generator.name].name,
            "antiunitary": bool(actions[generator.name].antiunitary),
            "fiber_permutation": list(actions[generator.name].fiber_permutation),
            "fiber_dimensions": list(actions[generator.name].fiber_dimensions),
            "fiber_indices": [
                list(value) for value in actions[generator.name].fiber_indices
            ],
            "route_block_hashes": [
                _matrix_hash(value)
                for value in actions[generator.name].route_blocks
            ],
        }
        for generator in presentation.generators
    }


def _actions_hash(actions: Mapping[str, Any], presentation: Any) -> str:
    return _stable_hash(_actions_artifact(actions, presentation))


def _operation_error_bounds(
    values: Mapping[str, float | Sequence[float]] | None,
    name: str,
    fiber_count: int,
) -> tuple[float, ...]:
    if values is None or name not in values:
        return (0.0,) * int(fiber_count)
    raw = values[name]
    if np.isscalar(raw):
        bounds = (float(raw),) * int(fiber_count)
    else:
        bounds = tuple(float(value) for value in raw)
    if len(bounds) != int(fiber_count):
        raise ValueError(
            f"absolute error bounds for {name!r} must contain {fiber_count} values"
        )
    if any(not np.isfinite(value) or value < 0.0 for value in bounds):
        raise ValueError(f"absolute error bounds for {name!r} must be finite and nonnegative")
    return bounds


def _classify_sector(
    action: Any,
    *,
    source_sector_index: int,
    layout: FiberSectorLayout,
    error_bounds: Sequence[float],
) -> SectorStructureCertificate:
    sector_name = layout.sector_names[source_sector_index]
    source_fibers = tuple(
        index
        for index, owner in enumerate(layout.fiber_sectors)
        if owner == source_sector_index
    )
    target_indices = {
        layout.fiber_sectors[action.fiber_permutation[source]]
        for source in source_fibers
    }
    if len(target_indices) != 1:
        return SectorStructureCertificate(
            operation=action.name,
            antiunitary=action.antiunitary,
            source_sector=sector_name,
            target_sector=None,
            status=CertificationStatus.UNKNOWN_OR_UNSUPPORTED,
            grade=None,
            source_fibers=source_fibers,
            reference_fiber=None,
            reference_block_hash=None,
            relative_phases=tuple(None for _ in source_fibers),
            uniform_residual_max=None,
            projective_residual_max=None,
            minimum_overlap_magnitude=None,
            certification_bound_max=None,
            obstruction_witness={"kind": "multiple_target_sectors"},
            reason="one source sector routes to multiple target sectors",
        )
    target_index = next(iter(target_indices))
    dimensions = {int(action.fiber_dimensions[source]) for source in source_fibers}
    if len(dimensions) != 1:
        return SectorStructureCertificate(
            operation=action.name,
            antiunitary=action.antiunitary,
            source_sector=sector_name,
            target_sector=layout.sector_names[target_index],
            status=CertificationStatus.UNKNOWN_OR_UNSUPPORTED,
            grade=None,
            source_fibers=source_fibers,
            reference_fiber=None,
            reference_block_hash=None,
            relative_phases=tuple(None for _ in source_fibers),
            uniform_residual_max=None,
            projective_residual_max=None,
            minimum_overlap_magnitude=None,
            certification_bound_max=None,
            reason="mixed internal dimensions inside one source sector",
        )
    dimension = dimensions.pop()
    reference_fiber = min(source_fibers)
    reference = np.asarray(action.route_blocks[reference_fiber], dtype=np.complex128)
    sqrt_dimension = float(np.sqrt(dimension))
    uniform_residuals: list[float] = []
    projective_residuals: list[float] = []
    overlaps: list[float] = []
    phases: list[complex | None] = []
    bounds: list[float] = []
    ambiguity_margins: list[float] = []
    for source in source_fibers:
        block = np.asarray(action.route_blocks[source], dtype=np.complex128)
        overlap = complex(np.vdot(reference, block))
        overlap_magnitude = float(abs(overlap))
        uniform = float(np.linalg.norm(block - reference, ord="fro") / sqrt_dimension)
        floating = float(
            _gamma(8 * dimension * dimension + 16)
            * (
                np.linalg.norm(block, ord="fro")
                + np.linalg.norm(reference, ord="fro")
            )
            / sqrt_dimension
        )
        bound = float(
            (error_bounds[source] + error_bounds[reference_fiber]) / sqrt_dimension
            + floating
        )
        overlap_bound = float(
            (error_bounds[source] + error_bounds[reference_fiber]) * sqrt_dimension
            + floating * dimension
        )
        phase = overlap / overlap_magnitude if overlap_magnitude > overlap_bound else None
        if phase is None:
            projective_squared = float(
                (
                    np.linalg.norm(block, ord="fro") ** 2
                    + np.linalg.norm(reference, ord="fro") ** 2
                    - 2.0 * overlap_magnitude
                )
                / float(dimension)
            )
            projective = float(np.sqrt(max(0.0, projective_squared)))
        else:
            # The algebraically equivalent direct norm avoids catastrophic
            # cancellation when exact blocks differ only by a unit phase.
            projective = float(
                np.linalg.norm(block - phase * reference, ord="fro")
                / sqrt_dimension
            )
        uniform_residuals.append(uniform)
        projective_residuals.append(projective)
        overlaps.append(overlap_magnitude)
        phases.append(phase)
        bounds.append(bound)
        ambiguity_margins.append(max(8.0 * floating, 64.0 * np.finfo(np.float64).eps))
    uniform_max = max(uniform_residuals)
    projective_max = max(projective_residuals)
    bound_max = max(bounds)
    uniform_certified = all(
        residual <= bound
        for residual, bound in zip(uniform_residuals, bounds)
    )
    uniform_disproved = any(
        residual > bound + margin
        for residual, bound, margin in zip(
            uniform_residuals,
            bounds,
            ambiguity_margins,
        )
    )
    projective_certified = all(
        residual <= bound
        for residual, bound in zip(projective_residuals, bounds)
    )
    projective_disproved = any(
        residual > bound + margin
        for residual, bound, margin in zip(
            projective_residuals,
            bounds,
            ambiguity_margins,
        )
    )
    if uniform_certified:
        status = CertificationStatus.CERTIFIED
        grade: StructureGrade | None = StructureGrade.UNIFORM_INTERNAL
        reason = None
    elif uniform_disproved and projective_certified:
        status = CertificationStatus.CERTIFIED
        grade = StructureGrade.PROJECTIVELY_UNIFORM
        reason = None
    elif projective_disproved:
        status = CertificationStatus.CERTIFIED
        grade = StructureGrade.GENERAL_UD_ROUTE
        reason = None
    else:
        status = CertificationStatus.UNKNOWN_OR_UNSUPPORTED
        grade = None
        reason = "structure residual overlaps its propagated numerical bound"
    proven_offenders = tuple(
        index
        for index, (residual, bound, margin) in enumerate(
            zip(projective_residuals, bounds, ambiguity_margins)
        )
        if residual > bound + margin
    )
    offender = (
        max(
            proven_offenders,
            key=lambda index: (
                projective_residuals[index]
                - bounds[index]
                - ambiguity_margins[index],
                -source_fibers[index],
            ),
        )
        if proven_offenders
        else int(np.argmax(projective_residuals))
    )
    obstruction = None
    if grade is StructureGrade.GENERAL_UD_ROUTE:
        obstruction = {
            "reference_fiber": int(reference_fiber),
            "offending_fiber": int(source_fibers[offender]),
            "projective_residual": float(projective_residuals[offender]),
            "certification_bound": float(bounds[offender]),
        }
    return SectorStructureCertificate(
        operation=action.name,
        antiunitary=action.antiunitary,
        source_sector=sector_name,
        target_sector=layout.sector_names[target_index],
        status=status,
        grade=grade,
        source_fibers=source_fibers,
        reference_fiber=reference_fiber,
        reference_block_hash=_matrix_hash(reference),
        relative_phases=tuple(phases),
        uniform_residual_max=uniform_max,
        projective_residual_max=projective_max,
        minimum_overlap_magnitude=min(overlaps),
        certification_bound_max=bound_max,
        obstruction_witness=obstruction,
        reason=reason,
    )


def certify_joint_structure(
    actions: Mapping[str, Any],
    presentation: Any,
    *,
    layout: FiberSectorLayout,
    block_absolute_error_bounds: Mapping[str, float | Sequence[float]] | None = None,
) -> JointStructureCertificate:
    """Classify all exact route blocks in one shared continuum basis."""

    from .joint_exactification import validate_presentation_action_relations

    generator_names = tuple(generator.name for generator in presentation.generators)
    if set(actions) != set(generator_names):
        raise ValueError("joint structure action names must match the presentation")
    for generator in presentation.generators:
        action = actions[generator.name]
        if action.name != generator.name:
            raise ValueError(
                f"action name {action.name!r} does not match generator "
                f"{generator.name!r}"
            )
        if bool(action.antiunitary) != bool(generator.antiunitary):
            raise ValueError(
                f"antiunitary parity mismatch for {generator.name!r}"
            )
    try:
        validate_presentation_action_relations(actions, presentation)
    except Exception as exc:
        # Present one stable public exception type from this independent
        # certificate layer while preserving the detailed validation message.
        raise ValueError(str(exc)) from exc
    reference = actions[generator_names[0]]
    fiber_count = len(reference.fiber_dimensions)
    if len(layout.fiber_sectors) != fiber_count:
        raise ValueError("fiber-sector layout size does not match the joint actions")
    operation_certificates: dict[str, OperationStructureCertificate] = {}
    overall_status = CertificationStatus.CERTIFIED
    for generator in presentation.generators:
        action = actions[generator.name]
        if bool(action.antiunitary) != bool(generator.antiunitary):
            raise ValueError(f"antiunitary parity mismatch for {generator.name!r}")
        if (
            len(action.fiber_dimensions) != fiber_count
            or tuple(action.fiber_indices) != tuple(reference.fiber_indices)
        ):
            raise ValueError("joint structure actions do not share one fiber layout")
        error_bounds = _operation_error_bounds(
            block_absolute_error_bounds,
            generator.name,
            fiber_count,
        )
        rows = {
            layout.sector_names[index]: _classify_sector(
                action,
                source_sector_index=index,
                layout=layout,
                error_bounds=error_bounds,
            )
            for index in range(len(layout.sector_names))
        }
        statuses = {row.status for row in rows.values()}
        status = (
            CertificationStatus.UNKNOWN_OR_UNSUPPORTED
            if CertificationStatus.UNKNOWN_OR_UNSUPPORTED in statuses
            else CertificationStatus.CERTIFIED
        )
        grades = [row.grade for row in rows.values() if row.grade is not None]
        grade = min(grades) if len(grades) == len(rows) else None
        operation_certificates[generator.name] = OperationStructureCertificate(
            name=generator.name,
            antiunitary=generator.antiunitary,
            status=status,
            grade=grade,
            fiber_permutation=tuple(action.fiber_permutation),
            fiber_dimensions=tuple(action.fiber_dimensions),
            fiber_indices=tuple(tuple(value) for value in action.fiber_indices),
            source_sectors=rows,
        )
        if status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED:
            overall_status = status
    if presentation.relations:
        from .joint_exactification import certify_joint_block_actions

        relation_certificate = certify_joint_block_actions(actions, presentation)
    else:
        relation_certificate = {
            "status": "certified",
            "relation_residuals": {},
            "relation_residual_max": 0.0,
            "relation_certification_bound": 0.0,
            "reason": "presentation_has_no_relations",
        }
    layout_artifact = layout.artifact()
    layout_hash = _stable_hash(
        {
            "sector_names": layout_artifact["sector_names"],
            "fiber_sectors": layout_artifact["fiber_sectors"],
            "fiber_dimensions": list(reference.fiber_dimensions),
            "fiber_indices": [list(value) for value in reference.fiber_indices],
            "operation_fiber_permutations": {
                name: list(actions[name].fiber_permutation)
                for name in generator_names
            },
        }
    )
    presentation_artifact = _presentation_artifact(presentation)
    presentation_hash = _stable_hash(presentation_artifact)
    declared_cocycle_hash = _stable_hash(
        {
            "presentation": presentation_artifact,
            "declared_phase_signature": _jsonable(layout.declared_phase_signature),
        }
    )
    actions_hash = _actions_hash(actions, presentation)
    payload = {
        "version": JOINT_STRUCTURE_CERTIFICATE_V1,
        "status": overall_status.value,
        "basis_frame_hash": layout.basis_frame_hash,
        "layout_hash": layout_hash,
        "presentation_hash": presentation_hash,
        "declared_cocycle_hash": declared_cocycle_hash,
        "actions_hash": actions_hash,
        "operations": {
            name: certificate.artifact()
            for name, certificate in operation_certificates.items()
        },
        "relation_certificate": _jsonable(relation_certificate),
    }
    certificate_hash = _stable_hash(payload)
    return JointStructureCertificate(
        status=overall_status,
        basis_frame_hash=layout.basis_frame_hash,
        layout_hash=layout_hash,
        presentation_hash=presentation_hash,
        declared_cocycle_hash=declared_cocycle_hash,
        actions_hash=actions_hash,
        operations=operation_certificates,
        relation_certificate=relation_certificate,
        certificate_hash=certificate_hash,
    )


def _gauge_hash(gauge: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(COMMON_FIBER_GAUGE_WITNESS_V1.encode("utf-8"))
    for value in gauge:
        matrix = np.asarray(value, dtype="<c16", order="C")
        digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes(order="C"))
        digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def common_fiber_gauge_hash(gauge: Sequence[np.ndarray]) -> str:
    """Return the canonical content hash used by common-gauge witnesses.

    Artifact consumers use this public spelling to bind a persisted block
    gauge to its certified transaction without re-running a gauge solver.
    """

    return _gauge_hash(gauge)


def derive_target_basis_hash(
    source_basis_hash: str,
    fiber_gauge: Sequence[np.ndarray],
) -> str:
    """Derive the target basis identity from its source and exact gauge bytes.

    The literal identity gauge preserves the source frame identity.  Every
    non-identity gauge receives a domain-separated content identity, so a
    caller cannot attach a certified witness to an unrelated target frame.
    """

    matrices = tuple(np.asarray(value, dtype=np.complex128) for value in fiber_gauge)
    if not matrices:
        raise ValueError("target basis hash requires at least one fiber gauge")
    if all(
        value.shape[0] == value.shape[1]
        and np.array_equal(
            value,
            np.eye(value.shape[0], dtype=np.complex128),
        )
        for value in matrices
    ):
        return str(source_basis_hash)
    digest = hashlib.sha256()
    digest.update(b"kp:joint-structure-frame:v1\0")
    digest.update(str(source_basis_hash).encode("utf-8"))
    for value in matrices:
        matrix = np.asarray(value, dtype="<c16", order="C")
        digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes(order="C"))
        digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def certify_common_fiber_gauge(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    fiber_gauge: Sequence[np.ndarray],
    presentation: Any,
    *,
    source_basis_hash: str,
    target_basis_hash: str,
    absolute_error_bounds: Mapping[str, float | Sequence[float]] | None = None,
) -> CommonGaugeWitness:
    """Prove that one semilinear fiber gauge maps all actions to a candidate."""

    from .joint_exactification import validate_presentation_action_relations

    names = tuple(generator.name for generator in presentation.generators)
    for label, actions in (("source", before), ("target", after)):
        if set(actions) != set(names):
            raise ValueError(
                f"{label} action names must match the presentation"
            )
        for generator in presentation.generators:
            action = actions[generator.name]
            if action.name != generator.name:
                raise ValueError(
                    f"{label} action name {action.name!r} does not match "
                    f"generator {generator.name!r}"
                )
            if bool(action.antiunitary) != bool(generator.antiunitary):
                raise ValueError(
                    f"{label} antiunitary parity mismatch for "
                    f"{generator.name!r}"
                )
        try:
            validate_presentation_action_relations(actions, presentation)
        except Exception as exc:
            raise ValueError(f"{label} action validation failed: {exc}") from exc
    reference = before[names[0]]
    gauge = tuple(np.asarray(value, dtype=np.complex128) for value in fiber_gauge)
    if len(gauge) != len(reference.fiber_dimensions):
        raise ValueError("common fiber gauge must contain one matrix per fiber")
    unitarity_residuals: list[float] = []
    unitarity_bounds: list[float] = []
    for fiber, (value, dimension) in enumerate(
        zip(gauge, reference.fiber_dimensions)
    ):
        expected_shape = (int(dimension), int(dimension))
        if value.shape != expected_shape:
            raise ValueError(
                f"common fiber gauge {fiber} has shape {value.shape}, "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"common fiber gauge {fiber} must be finite")
        identity = np.eye(int(dimension), dtype=np.complex128)
        residual = float(
            np.linalg.norm(value.conjugate().T @ value - identity, ord="fro")
        )
        bound = float(
            _gamma(16 * int(dimension) * int(dimension) + 32)
            * max(1.0, np.linalg.norm(value, ord="fro") ** 2)
        )
        if residual > bound:
            raise ValueError(
                f"common fiber gauge {fiber} is not certified unitary: "
                f"residual={residual:.6e}, bound={bound:.6e}"
            )
        unitarity_residuals.append(residual)
        unitarity_bounds.append(bound)
    source_actions_hash = _actions_hash(before, presentation)
    target_actions_hash = _actions_hash(after, presentation)
    presentation_hash = _stable_hash(_presentation_artifact(presentation))
    gauge_hash = _gauge_hash(gauge)
    expected_target_basis_hash = derive_target_basis_hash(
        str(source_basis_hash),
        gauge,
    )
    if str(target_basis_hash) != expected_target_basis_hash:
        raise ValueError(
            "target basis hash is not derived from the source frame and common gauge"
        )
    unitarity_residual_max = max(unitarity_residuals, default=0.0)
    unitarity_bound_max = max(unitarity_bounds, default=0.0)
    residual_by_operation: dict[str, float] = {}
    bound_by_operation: dict[str, float] = {}
    residual_by_operation_fiber: dict[str, tuple[float, ...]] = {}
    bound_by_operation_fiber: dict[str, tuple[float, ...]] = {}
    witness_status = CertificationStatus.CERTIFIED
    reason = None
    for generator in presentation.generators:
        source_action = before[generator.name]
        target_action = after[generator.name]
        if (
            source_action.antiunitary != target_action.antiunitary
            or source_action.fiber_permutation != target_action.fiber_permutation
            or source_action.fiber_dimensions != target_action.fiber_dimensions
            or source_action.fiber_indices != target_action.fiber_indices
        ):
            return CommonGaugeWitness(
                status=CertificationStatus.CERTIFIED_INFEASIBLE,
                source_basis_hash=str(source_basis_hash),
                target_basis_hash=str(target_basis_hash),
                source_actions_hash=source_actions_hash,
                target_actions_hash=target_actions_hash,
                presentation_hash=presentation_hash,
                gauge_hash=gauge_hash,
                residual_by_operation={},
                certification_bound_by_operation={},
                residual_by_operation_fiber={},
                certification_bound_by_operation_fiber={},
                max_residual=None,
                certification_bound=None,
                gauge_unitarity_residual_max=unitarity_residual_max,
                gauge_unitarity_certification_bound=unitarity_bound_max,
                reason=f"route layout changed for {generator.name}",
            )
        errors = _operation_error_bounds(
            absolute_error_bounds,
            generator.name,
            len(gauge),
        )
        residuals: list[float] = []
        bounds: list[float] = []
        for source, target in enumerate(source_action.fiber_permutation):
            source_gauge = (
                gauge[source].conjugate()
                if source_action.antiunitary
                else gauge[source]
            )
            predicted = np.asarray(
                gauge[target].conjugate().T
                @ source_action.route_blocks[source]
                @ source_gauge,
                dtype=np.complex128,
            )
            actual = np.asarray(target_action.route_blocks[source], dtype=np.complex128)
            residual = float(np.linalg.norm(predicted - actual, ord="fro"))
            dimension = int(actual.shape[0])
            floating = float(
                _gamma(16 * dimension * dimension + 32)
                * max(
                    1.0,
                    np.linalg.norm(predicted, ord="fro"),
                    np.linalg.norm(actual, ord="fro"),
                )
            )
            bound = float(2.0 * errors[source] + floating)
            residuals.append(residual)
            bounds.append(bound)
        operation_residual = max(residuals)
        operation_bound = max(bounds)
        residual_by_operation[generator.name] = operation_residual
        bound_by_operation[generator.name] = operation_bound
        residual_by_operation_fiber[generator.name] = tuple(residuals)
        bound_by_operation_fiber[generator.name] = tuple(bounds)
        decisively_failed = any(
            residual
            > bound
            + max(
                8.0 * bound,
                64.0 * np.finfo(np.float64).eps,
            )
            for residual, bound in zip(residuals, bounds)
        )
        overlaps_bound = any(
            residual > bound
            for residual, bound in zip(residuals, bounds)
        )
        if decisively_failed:
            witness_status = CertificationStatus.CERTIFIED_INFEASIBLE
            reason = f"common gauge does not reproduce {generator.name}"
        elif overlaps_bound and witness_status is CertificationStatus.CERTIFIED:
            witness_status = CertificationStatus.UNKNOWN_OR_UNSUPPORTED
            reason = f"common-gauge residual for {generator.name} overlaps its bound"
    max_residual = max(residual_by_operation.values(), default=0.0)
    certification_bound = max(bound_by_operation.values(), default=0.0)
    return CommonGaugeWitness(
        status=witness_status,
        source_basis_hash=str(source_basis_hash),
        target_basis_hash=str(target_basis_hash),
        source_actions_hash=source_actions_hash,
        target_actions_hash=target_actions_hash,
        presentation_hash=presentation_hash,
        gauge_hash=gauge_hash,
        residual_by_operation=residual_by_operation,
        certification_bound_by_operation=bound_by_operation,
        residual_by_operation_fiber=residual_by_operation_fiber,
        certification_bound_by_operation_fiber=bound_by_operation_fiber,
        max_residual=max_residual,
        certification_bound=certification_bound,
        gauge_unitarity_residual_max=unitarity_residual_max,
        gauge_unitarity_certification_bound=unitarity_bound_max,
        reason=reason,
    )


def compare_joint_structure(
    before: JointStructureCertificate,
    after: JointStructureCertificate,
    *,
    gauge_witness: CommonGaugeWitness,
) -> StructureTransition:
    """Accept only common-gauge transitions that do not lower any structure grade."""

    if gauge_witness.status is CertificationStatus.CERTIFIED_INFEASIBLE:
        return StructureTransition(
            status=CertificationStatus.CERTIFIED_INFEASIBLE,
            is_monotone=False,
            regressions=(),
            reason=gauge_witness.reason or "candidate lacks a certified common gauge",
        )
    if (
        before.basis_frame_hash != gauge_witness.source_basis_hash
        or after.basis_frame_hash != gauge_witness.target_basis_hash
    ):
        return StructureTransition(
            status=CertificationStatus.CERTIFIED_INFEASIBLE,
            is_monotone=False,
            regressions=(),
            reason="common-gauge witness basis hashes do not match its certificates",
        )
    if (
        before.actions_hash != gauge_witness.source_actions_hash
        or after.actions_hash != gauge_witness.target_actions_hash
        or before.presentation_hash != gauge_witness.presentation_hash
        or after.presentation_hash != gauge_witness.presentation_hash
    ):
        return StructureTransition(
            status=CertificationStatus.CERTIFIED_INFEASIBLE,
            is_monotone=False,
            regressions=(),
            reason="common-gauge witness is not bound to these action certificates",
        )
    if (
        gauge_witness.status is not CertificationStatus.CERTIFIED
        or before.status is not CertificationStatus.CERTIFIED
        or after.status is not CertificationStatus.CERTIFIED
    ):
        return StructureTransition(
            status=CertificationStatus.UNKNOWN_OR_UNSUPPORTED,
            is_monotone=False,
            regressions=(),
            reason="certificate or common-gauge witness is not decisive",
        )
    if (
        before.layout_hash != after.layout_hash
        or before.presentation_hash != after.presentation_hash
        or before.declared_cocycle_hash != after.declared_cocycle_hash
        or tuple(before.operations) != tuple(after.operations)
    ):
        return StructureTransition(
            status=CertificationStatus.CERTIFIED_INFEASIBLE,
            is_monotone=False,
            regressions=(),
            reason="candidate changed the protected layout, presentation, or phase signature",
        )
    regressions: list[tuple[str, str]] = []
    for name, before_operation in before.operations.items():
        after_operation = after.operations[name]
        if tuple(before_operation.source_sectors) != tuple(after_operation.source_sectors):
            return StructureTransition(
                status=CertificationStatus.CERTIFIED_INFEASIBLE,
                is_monotone=False,
                regressions=(),
                reason=f"candidate changed source-sector labels for {name}",
            )
        for sector_name, before_sector in before_operation.source_sectors.items():
            after_sector = after_operation.source_sectors[sector_name]
            if before_sector.grade is None or after_sector.grade is None:
                return StructureTransition(
                    status=CertificationStatus.UNKNOWN_OR_UNSUPPORTED,
                    is_monotone=False,
                    regressions=tuple(regressions),
                    reason=f"structure grade is unknown for {name}/{sector_name}",
                )
            if after_sector.grade < before_sector.grade:
                regressions.append((name, sector_name))
    if regressions:
        return StructureTransition(
            status=CertificationStatus.CERTIFIED_INFEASIBLE,
            is_monotone=False,
            regressions=tuple(regressions),
            reason="candidate lowers at least one protected structure grade",
        )
    return StructureTransition(
        status=CertificationStatus.CERTIFIED,
        is_monotone=True,
        regressions=(),
        reason=None,
    )
