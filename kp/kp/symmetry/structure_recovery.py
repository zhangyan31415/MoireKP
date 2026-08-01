"""Certified recovery of Q-uniform continuum symmetry actions.

The solver in this module is intentionally bounded.  It proves recovery only
when the discrete fiber action admits an equivariant anchor section (normally
the central-Q fiber in every sector).  All other cases fail closed as
``unknown_or_unsupported``; they are not declared mathematically infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .joint_exactification import (
    ActionOrbit,
    BlockRouteAction,
    JointExactificationError,
    MagneticPresentation,
    SemilinearBlock,
    certify_joint_block_actions,
    compile_action_orbits,
    compose_semilinear,
    derive_standard_generator_fiber_gauge,
    inverse_semilinear,
    semilinear_kappa,
    validate_presentation_action_relations,
)
from .structure_certificate import (
    CertificationStatus,
    CommonGaugeWitness,
    FiberSectorLayout,
    JointStructureCertificate,
    StructureGrade,
    certify_common_fiber_gauge,
    certify_joint_structure,
    compare_joint_structure,
    derive_target_basis_hash,
)


STRUCTURE_RECOVERY_V1 = "equivariant_anchor_uniform_recovery_v1"
FIXED_TARGET_ALIGNMENT_VERSION = "fixed_target_common_gauge_alignment_v2"


class _RecoveryUnsupported(RuntimeError):
    """Internal control flow for a bounded solver that cannot prove recovery."""


@dataclass(frozen=True)
class StructureRecoveryResult:
    """A certified uniform action, or an explicit fail-closed result."""

    status: CertificationStatus
    grade: StructureGrade | None
    actions: Mapping[str, BlockRouteAction] | None
    fiber_gauge: tuple[np.ndarray, ...] | None
    certificate: JointStructureCertificate | None
    common_gauge_witness: CommonGaugeWitness | None
    report: Mapping[str, Any]
    version: str = STRUCTURE_RECOVERY_V1

    def __post_init__(self) -> None:
        actions = None if self.actions is None else MappingProxyType(dict(self.actions))
        gauge = (
            None
            if self.fiber_gauge is None
            else tuple(_readonly_complex(value) for value in self.fiber_gauge)
        )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "fiber_gauge", gauge)
        object.__setattr__(self, "report", _deep_freeze(self.report))

    def artifact(self) -> dict[str, Any]:
        """Return a detached, strict-JSON-ready copy of the immutable report."""

        return _jsonable(self.report)


def _readonly_complex(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.complex128, order="C", copy=True)
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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
    if isinstance(value, CertificationStatus):
        return value.value
    if isinstance(value, StructureGrade):
        return int(value)
    return value


def _unknown(reason: str, **details: Any) -> StructureRecoveryResult:
    return StructureRecoveryResult(
        status=CertificationStatus.UNKNOWN_OR_UNSUPPORTED,
        grade=None,
        actions=None,
        fiber_gauge=None,
        certificate=None,
        common_gauge_witness=None,
        report={
            "version": STRUCTURE_RECOVERY_V1,
            "status": CertificationStatus.UNKNOWN_OR_UNSUPPORTED.value,
            "selection_policy": "equivariant_anchor_schreier_intertwiner",
            "reason": str(reason),
            **details,
        },
    )


def _sector_permutations(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    layout: FiberSectorLayout,
) -> dict[str, tuple[int, ...]]:
    output: dict[str, tuple[int, ...]] = {}
    for generator in presentation.generators:
        action = actions[generator.name]
        targets: list[int] = []
        for sector in range(len(layout.sector_names)):
            source_fibers = (
                fiber
                for fiber, owner in enumerate(layout.fiber_sectors)
                if owner == sector
            )
            target_sectors = {
                layout.fiber_sectors[action.fiber_permutation[fiber]]
                for fiber in source_fibers
            }
            if len(target_sectors) != 1:
                raise _RecoveryUnsupported(
                    f"{generator.name}/{layout.sector_names[sector]} does not "
                    "induce one target sector"
                )
            targets.append(int(next(iter(target_sectors))))
        if sorted(targets) != list(range(len(layout.sector_names))):
            raise _RecoveryUnsupported(
                f"{generator.name} does not induce a sector permutation"
            )
        output[generator.name] = tuple(targets)
    return output


def _sector_components(
    sector_count: int,
    sector_permutations: Mapping[str, Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(sector_count))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        component = {root}
        frontier = [root]
        while frontier:
            source = frontier.pop()
            for permutation in sector_permutations.values():
                target = int(permutation[source])
                if target not in component:
                    component.add(target)
                    frontier.append(target)
                for inverse_source, inverse_target in enumerate(permutation):
                    if int(inverse_target) == source and inverse_source not in component:
                        component.add(inverse_source)
                        frontier.append(inverse_source)
        ordered = tuple(sorted(component))
        components.append(ordered)
        unseen.difference_update(component)
    return tuple(components)


def _anchor_sections(
    orbits: Sequence[ActionOrbit],
    layout: FiberSectorLayout,
    sector_permutations: Mapping[str, Sequence[int]],
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    max_candidates: int,
) -> tuple[tuple[int, ...], ...]:
    components = _sector_components(len(layout.sector_names), sector_permutations)
    choices: list[list[dict[int, int]]] = []
    for component in components:
        rows: list[dict[int, int]] = []
        for orbit in orbits:
            by_sector: dict[int, list[int]] = {}
            for fiber in orbit.fibers:
                by_sector.setdefault(layout.fiber_sectors[fiber], []).append(fiber)
            if set(by_sector) != set(component):
                continue
            if any(len(by_sector[sector]) != 1 for sector in component):
                continue
            rows.append({sector: by_sector[sector][0] for sector in component})
        rows.sort(key=lambda row: tuple(row[sector] for sector in component))
        if not rows:
            return ()
        choices.append(rows)
    count = 1
    for rows in choices:
        count *= len(rows)
        if count > int(max_candidates):
            raise _RecoveryUnsupported(
                "equivariant anchor-section candidate limit exceeded"
            )
    sections: list[tuple[int, ...]] = []
    for selected in product(*choices):
        anchors = [-1] * len(layout.sector_names)
        for row in selected:
            for sector, fiber in row.items():
                anchors[sector] = fiber
        if any(value < 0 for value in anchors):
            continue
        valid = True
        for generator in presentation.generators:
            action = actions[generator.name]
            sector_permutation = sector_permutations[generator.name]
            for sector, anchor in enumerate(anchors):
                if action.fiber_permutation[anchor] != anchors[sector_permutation[sector]]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            sections.append(tuple(anchors))
    return tuple(sorted(set(sections)))


def _uniform_target_actions(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    layout: FiberSectorLayout,
    anchors: Sequence[int],
) -> dict[str, BlockRouteAction]:
    targets: dict[str, BlockRouteAction] = {}
    for generator in presentation.generators:
        action = actions[generator.name]
        blocks = tuple(
            np.asarray(
                action.route_blocks[anchors[layout.fiber_sectors[source]]],
                dtype=np.complex128,
            )
            for source in range(len(layout.fiber_sectors))
        )
        targets[generator.name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=blocks,
            fiber_indices=action.fiber_indices,
        )
    certify_joint_block_actions(targets, presentation)
    return targets


def _gauge_transform_actions(
    actions: Mapping[str, BlockRouteAction],
    gauge: Sequence[np.ndarray],
) -> dict[str, BlockRouteAction]:
    transformed: dict[str, BlockRouteAction] = {}
    for name, action in actions.items():
        blocks = []
        for source, target in enumerate(action.fiber_permutation):
            source_gauge = (
                gauge[source].conjugate()
                if action.antiunitary
                else gauge[source]
            )
            blocks.append(
                np.asarray(
                    gauge[target].conjugate().T
                    @ action.route_blocks[source]
                    @ source_gauge,
                    dtype=np.complex128,
                )
            )
        transformed[name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=tuple(blocks),
            fiber_indices=action.fiber_indices,
        )
    return transformed


def _standardize_uniform_sector_action(
    target: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    layout: FiberSectorLayout,
    sector_permutations: Mapping[str, Sequence[int]],
    anchors: Sequence[int],
) -> tuple[
    dict[str, BlockRouteAction],
    tuple[np.ndarray, ...],
    dict[str, Any],
]:
    """Apply one standard algebraic frame per sector, never per Q fiber."""

    sector_dimensions = tuple(
        int(target[presentation.generators[0].name].fiber_dimensions[anchor])
        for anchor in anchors
    )
    reduced: dict[str, BlockRouteAction] = {}
    for generator in presentation.generators:
        action = target[generator.name]
        reduced[generator.name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=tuple(sector_permutations[generator.name]),
            fiber_dimensions=sector_dimensions,
            route_blocks=tuple(
                np.asarray(action.route_blocks[anchors[sector]], dtype=np.complex128)
                for sector in range(len(anchors))
            ),
        )
    standard = derive_standard_generator_fiber_gauge(reduced, presentation)
    if standard.report.get("status") != "certified":
        raise _RecoveryUnsupported(
            "reduced sector action has no certified algebraic standard frame: "
            f"{standard.report.get('reason', 'unsupported')}"
        )
    expanded = tuple(
        np.asarray(
            standard.fiber_gauge[layout.fiber_sectors[fiber]],
            dtype=np.complex128,
        )
        for fiber in range(len(layout.fiber_sectors))
    )
    standardized: dict[str, BlockRouteAction] = {}
    for generator in presentation.generators:
        full_action = target[generator.name]
        reduced_action = standard.actions[generator.name]
        standardized[generator.name] = BlockRouteAction(
            name=full_action.name,
            antiunitary=full_action.antiunitary,
            fiber_permutation=full_action.fiber_permutation,
            fiber_dimensions=full_action.fiber_dimensions,
            route_blocks=tuple(
                np.asarray(
                    reduced_action.route_blocks[layout.fiber_sectors[source]],
                    dtype=np.complex128,
                )
                for source in range(len(layout.fiber_sectors))
            ),
            fiber_indices=full_action.fiber_indices,
        )
    certify_joint_block_actions(standardized, presentation)
    return standardized, expanded, dict(standard.report)


def _route_word(
    actions: Mapping[str, BlockRouteAction],
    word: Sequence[str],
    source: int,
) -> tuple[SemilinearBlock, int]:
    reference = actions[next(iter(actions))]
    value = SemilinearBlock(
        np.eye(reference.fiber_dimensions[source], dtype=np.complex128),
        False,
    )
    current = int(source)
    for name in reversed(tuple(word)):
        action = actions[name]
        edge = SemilinearBlock(
            action.route_blocks[current],
            action.antiunitary,
            unitarity_certification_bound=action.unitarity_certification_bound,
        )
        value = compose_semilinear(edge, value)
        current = action.fiber_permutation[current]
    return value, current


def _orbit_transporters(
    actions: Mapping[str, BlockRouteAction],
    orbit: ActionOrbit,
) -> dict[int, SemilinearBlock]:
    output: dict[int, SemilinearBlock] = {}
    for fiber, word in zip(orbit.fibers, orbit.transporter_words):
        value, target = _route_word(actions, word, orbit.root)
        if target != fiber:
            raise _RecoveryUnsupported(
                f"transporter {word} reaches {target}, expected {fiber}"
            )
        output[fiber] = value
    return output


def _schreier_holonomies(
    before: Mapping[str, BlockRouteAction],
    target: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    before_transporters: Mapping[int, SemilinearBlock],
    target_transporters: Mapping[int, SemilinearBlock],
) -> tuple[tuple[SemilinearBlock, SemilinearBlock], ...]:
    rows: list[tuple[SemilinearBlock, SemilinearBlock]] = []
    for generator in presentation.generators:
        before_action = before[generator.name]
        target_action = target[generator.name]
        for source in orbit.fibers:
            target_fiber = before_action.fiber_permutation[source]
            before_edge = SemilinearBlock(
                before_action.route_blocks[source],
                before_action.antiunitary,
                unitarity_certification_bound=(
                    before_action.unitarity_certification_bound
                ),
            )
            target_edge = SemilinearBlock(
                target_action.route_blocks[source],
                target_action.antiunitary,
                unitarity_certification_bound=(
                    target_action.unitarity_certification_bound
                ),
            )
            before_loop = compose_semilinear(
                inverse_semilinear(before_transporters[target_fiber]),
                compose_semilinear(before_edge, before_transporters[source]),
            )
            target_loop = compose_semilinear(
                inverse_semilinear(target_transporters[target_fiber]),
                compose_semilinear(target_edge, target_transporters[source]),
            )
            if before_loop.antiunitary != target_loop.antiunitary:
                raise _RecoveryUnsupported("Schreier holonomy parity mismatch")
            rows.append((before_loop, target_loop))
    return tuple(rows)


def _real_vector(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    return np.concatenate((value.real.ravel(order="C"), value.imag.ravel(order="C")))


def _complex_matrix(vector: np.ndarray, dimension: int) -> np.ndarray:
    size = int(dimension) * int(dimension)
    value = np.asarray(vector, dtype=np.float64)
    return np.asarray(
        value[:size].reshape((dimension, dimension))
        + 1.0j * value[size:].reshape((dimension, dimension)),
        dtype=np.complex128,
    )


def _real_matrix_basis(dimension: int) -> tuple[np.ndarray, ...]:
    real: list[np.ndarray] = []
    imaginary: list[np.ndarray] = []
    for row in range(dimension):
        for column in range(dimension):
            value = np.zeros((dimension, dimension), dtype=np.complex128)
            value[row, column] = 1.0
            real.append(value)
            imaginary.append(1.0j * value)
    return tuple((*real, *imaginary))


def _root_intertwiner(
    holonomies: Sequence[tuple[SemilinearBlock, SemilinearBlock]],
    before_transporters: Mapping[int, SemilinearBlock],
    target_transporters: Mapping[int, SemilinearBlock],
    orbit: ActionOrbit,
    *,
    reference_fiber_gauge: Mapping[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    dimension = int(next(iter(before_transporters.values())).matrix.shape[0])
    basis = _real_matrix_basis(dimension)
    columns: list[np.ndarray] = []
    for seed in basis:
        residuals = []
        for before_loop, target_loop in holonomies:
            residuals.append(
                before_loop.matrix
                @ semilinear_kappa(seed, before_loop.antiunitary)
                - seed @ target_loop.matrix
            )
        columns.append(
            np.concatenate([_real_vector(value) for value in residuals])
        )
    system = np.column_stack(columns)
    _left, singular_values, right_h = np.linalg.svd(system, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    rank_bound = float(
        4096.0
        * np.finfo(np.float64).eps
        * max(1, *system.shape)
        * max(1.0, largest)
    )
    ambiguous = [
        float(value)
        for value in singular_values
        if rank_bound / 16.0 < float(value) < rank_bound * 16.0
    ]
    if ambiguous:
        raise _RecoveryUnsupported(
            "Schreier intertwiner rank overlaps its floating-point bound"
        )
    rank = int(np.count_nonzero(singular_values > rank_bound))
    nullspace = np.asarray(right_h[rank:, :].T, dtype=np.float64)
    if nullspace.shape[1] == 0:
        raise _RecoveryUnsupported("Schreier intertwiner null space is empty")

    # The closest-to-reference objective is linear in the root unitary:
    # W_f = T_f^B kappa_p(W_root) (T_f^A)^dagger.  Project its exact real
    # coefficient onto the intertwiner space, then take the polar factor.  If
    # that projection is nonsingular, the polar unitary is the global optimum
    # inside this certified intertwiner class.
    objective_coefficient = np.zeros(2 * dimension * dimension, dtype=np.float64)
    for fiber in orbit.fibers:
        before_transporter = before_transporters[fiber]
        target_transporter = target_transporters[fiber]
        transport_columns = []
        for seed in basis:
            transported = (
                before_transporter.matrix
                @ semilinear_kappa(seed, before_transporter.antiunitary)
                @ target_transporter.matrix.conjugate().T
            )
            transport_columns.append(_real_vector(transported))
        transport = np.column_stack(transport_columns)
        reference = (
            np.eye(dimension, dtype=np.complex128)
            if reference_fiber_gauge is None
            else np.asarray(reference_fiber_gauge[fiber], dtype=np.complex128)
        )
        objective_coefficient += transport.T @ _real_vector(reference)
    projected = nullspace @ (nullspace.T @ objective_coefficient)
    seed = _complex_matrix(projected, dimension)
    left, polar_singular_values, right_h_polar = np.linalg.svd(seed)
    polar_bound = float(
        4096.0
        * np.finfo(np.float64).eps
        * max(1, dimension)
        * max(1.0, float(polar_singular_values[0]))
    )
    if (
        polar_singular_values.size != dimension
        or float(polar_singular_values[-1]) <= polar_bound
    ):
        raise _RecoveryUnsupported(
            "closest projected Schreier intertwiner is singular"
        )
    root = np.asarray(left @ right_h_polar, dtype=np.complex128)
    residual_max = max(
        float(
            np.linalg.norm(
                before_loop.matrix
                @ semilinear_kappa(root, before_loop.antiunitary)
                - root @ target_loop.matrix,
                ord="fro",
            )
        )
        for before_loop, target_loop in holonomies
    )
    residual_bound = float(
        65536.0
        * np.finfo(np.float64).eps
        * max(1, len(holonomies), dimension)
    )
    if residual_max > residual_bound:
        raise _RecoveryUnsupported(
            "polar root does not satisfy every Schreier intertwiner equation"
        )
    return root, {
        "root": int(orbit.root),
        "fibers": [int(value) for value in orbit.fibers],
        "block_dimension": dimension,
        "linear_rank": rank,
        "linear_nullity": int(nullspace.shape[1]),
        "rank_certification_bound": rank_bound,
        "singular_values": [float(value) for value in singular_values],
        "closest_projection_singular_values": [
            float(value) for value in polar_singular_values
        ],
        "intertwiner_residual_max": residual_max,
        "intertwiner_certification_bound": residual_bound,
    }


def _recover_for_anchor_section(
    before: Mapping[str, BlockRouteAction],
    target: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbits: Sequence[ActionOrbit],
    *,
    reference_fiber_gauge: Sequence[np.ndarray] | None = None,
) -> tuple[tuple[np.ndarray, ...], float, list[dict[str, Any]]]:
    reference = before[presentation.generators[0].name]
    objective_reference = (
        tuple(
            np.eye(dimension, dtype=np.complex128)
            for dimension in reference.fiber_dimensions
        )
        if reference_fiber_gauge is None
        else tuple(
            np.asarray(value, dtype=np.complex128)
            for value in reference_fiber_gauge
        )
    )
    if len(objective_reference) != len(reference.fiber_dimensions):
        raise _RecoveryUnsupported(
            "reference gauge does not cover every protected fiber"
        )
    reference_by_fiber = {
        fiber: objective_reference[fiber]
        for fiber in range(len(objective_reference))
    }
    gauge: list[np.ndarray | None] = [None] * len(reference.fiber_dimensions)
    reports: list[dict[str, Any]] = []
    for orbit in orbits:
        before_transporters = _orbit_transporters(before, orbit)
        target_transporters = _orbit_transporters(target, orbit)
        holonomies = _schreier_holonomies(
            before,
            target,
            presentation,
            orbit,
            before_transporters,
            target_transporters,
        )
        root, report = _root_intertwiner(
            holonomies,
            before_transporters,
            target_transporters,
            orbit,
            reference_fiber_gauge=reference_by_fiber,
        )
        for fiber in orbit.fibers:
            before_transporter = before_transporters[fiber]
            target_transporter = target_transporters[fiber]
            gauge[fiber] = np.asarray(
                before_transporter.matrix
                @ semilinear_kappa(root, before_transporter.antiunitary)
                @ target_transporter.matrix.conjugate().T,
                dtype=np.complex128,
            )
        reports.append(report)
    if any(value is None for value in gauge):
        raise _RecoveryUnsupported("recovery did not assign every fiber gauge")
    completed = tuple(
        np.asarray(value, dtype=np.complex128)
        for value in gauge
        if value is not None
    )
    objective = float(
        sum(
            np.linalg.norm(
                value - objective_reference[fiber],
                ord="fro",
            )
            ** 2
            for fiber, value in enumerate(completed)
        )
    )
    return completed, objective, reports


def _finalize_uniform_recovery(
    source: Mapping[str, BlockRouteAction],
    target: Mapping[str, BlockRouteAction],
    gauge: Sequence[np.ndarray],
    presentation: MagneticPresentation,
    *,
    layout: FiberSectorLayout,
    pre_certificate: JointStructureCertificate,
    report_details: Mapping[str, Any],
) -> StructureRecoveryResult:
    completed_gauge = tuple(np.asarray(value, dtype=np.complex128) for value in gauge)
    target_basis_hash = derive_target_basis_hash(
        str(layout.basis_frame_hash),
        completed_gauge,
    )
    target_layout = FiberSectorLayout(
        sector_names=layout.sector_names,
        fiber_sectors=layout.fiber_sectors,
        basis_frame_hash=target_basis_hash,
        declared_phase_signature=layout.declared_phase_signature,
    )
    post_certificate = certify_joint_structure(
        target,
        presentation,
        layout=target_layout,
    )
    if (
        post_certificate.status is not CertificationStatus.CERTIFIED
        or any(
            sector.grade is not StructureGrade.UNIFORM_INTERNAL
            for operation in post_certificate.operations.values()
            for sector in operation.source_sectors.values()
        )
    ):
        raise _RecoveryUnsupported(
            "reconstructed action is not certified uniform in every source sector"
        )
    witness = certify_common_fiber_gauge(
        source,
        target,
        completed_gauge,
        presentation,
        source_basis_hash=str(layout.basis_frame_hash),
        target_basis_hash=target_basis_hash,
    )
    transition = compare_joint_structure(
        pre_certificate,
        post_certificate,
        gauge_witness=witness,
    )
    if (
        witness.status is not CertificationStatus.CERTIFIED
        or transition.status is not CertificationStatus.CERTIFIED
        or not transition.is_monotone
    ):
        raise _RecoveryUnsupported(
            "uniform reconstruction failed its common-gauge transaction"
        )
    relation_certificate = certify_joint_block_actions(target, presentation)
    report = {
        "version": STRUCTURE_RECOVERY_V1,
        "status": CertificationStatus.CERTIFIED.value,
        "grade": "uniform_internal",
        "canonicality": "canonical_up_to_continuous_stabilizer",
        "residual_stabilizer_dimension": None,
        **dict(report_details),
        "common_gauge_witness": witness.artifact(),
        "transition": transition.artifact(),
        "relation_certification": dict(relation_certificate),
    }
    return StructureRecoveryResult(
        status=CertificationStatus.CERTIFIED,
        grade=StructureGrade.UNIFORM_INTERNAL,
        actions=target,
        fiber_gauge=completed_gauge,
        certificate=post_certificate,
        common_gauge_witness=witness,
        report=report,
    )


def recover_fixed_target_common_gauge(
    source: Mapping[str, BlockRouteAction],
    target: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    layout: FiberSectorLayout,
    source_action_absolute_error_bounds: Mapping[
        str, float | Sequence[float]
    ] | None = None,
    reference_fiber_gauge: Sequence[np.ndarray] | None = None,
) -> StructureRecoveryResult:
    """Find a certified common gauge to one supplied exact target.

    The target is not reselected or modified.  This is the transaction used to
    retain raw projected phase information while adopting the algebraic target
    already chosen by ``kp project``.  Continuous stabilizer freedom is fixed
    by proximity to ``reference_fiber_gauge`` when supplied, or to the identity
    for backward-compatible standalone calls.
    """

    def unknown(reason: str) -> StructureRecoveryResult:
        return StructureRecoveryResult(
            status=CertificationStatus.UNKNOWN_OR_UNSUPPORTED,
            grade=None,
            actions=None,
            fiber_gauge=None,
            certificate=None,
            common_gauge_witness=None,
            report={
                "version": FIXED_TARGET_ALIGNMENT_VERSION,
                "status": CertificationStatus.UNKNOWN_OR_UNSUPPORTED.value,
                "selection_policy": "fixed_target_closest_common_intertwiner",
                "closest_scope": "fixed_supplied_target",
                "reason": str(reason),
            },
            version=FIXED_TARGET_ALIGNMENT_VERSION,
        )

    try:
        names = tuple(generator.name for generator in presentation.generators)
        if set(source) != set(names) or set(target) != set(names):
            raise _RecoveryUnsupported(
                "source/target generator names mismatch the presentation"
            )
        reference = source[names[0]]
        supplied_reference = reference_fiber_gauge is not None
        objective_reference = (
            tuple(
                np.eye(dimension, dtype=np.complex128)
                for dimension in reference.fiber_dimensions
            )
            if reference_fiber_gauge is None
            else tuple(
                np.asarray(value, dtype=np.complex128)
                for value in reference_fiber_gauge
            )
        )
        if len(objective_reference) != len(reference.fiber_dimensions):
            raise _RecoveryUnsupported(
                "reference gauge does not cover every protected fiber"
            )
        reference_unitarity_residuals: list[float] = []
        for fiber, (value, dimension) in enumerate(
            zip(objective_reference, reference.fiber_dimensions)
        ):
            if value.shape != (dimension, dimension) or not np.all(
                np.isfinite(value)
            ):
                raise _RecoveryUnsupported(
                    f"reference gauge fiber {fiber} has invalid shape or values"
                )
            residual = float(
                np.linalg.norm(
                    value.conjugate().T @ value
                    - np.eye(dimension, dtype=np.complex128),
                    ord="fro",
                )
            )
            bound = float(
                4096.0
                * np.finfo(np.float64).eps
                * max(1, dimension * dimension)
                * max(1.0, float(np.linalg.norm(value, ord="fro") ** 2))
            )
            if residual > bound:
                raise _RecoveryUnsupported(
                    f"reference gauge fiber {fiber} is not unitary"
                )
            reference_unitarity_residuals.append(residual)
        declared_error_names = set(source_action_absolute_error_bounds or {})
        if not declared_error_names.issubset(set(names)):
            raise _RecoveryUnsupported(
                "source action error bounds contain undeclared generators"
            )
        normalized_error_bounds: dict[str, float | tuple[float, ...]] = {}
        for name in names:
            raw_error = (source_action_absolute_error_bounds or {}).get(name, 0.0)
            if np.isscalar(raw_error):
                value = float(raw_error)
                if not np.isfinite(value) or value < 0.0:
                    raise _RecoveryUnsupported(
                        f"source action error bound for {name} must be finite and nonnegative"
                    )
                normalized_error_bounds[name] = value
            else:
                values = tuple(float(value) for value in raw_error)
                if len(values) != len(reference.fiber_dimensions):
                    raise _RecoveryUnsupported(
                        f"source action error bounds for {name} do not cover every fiber"
                    )
                if any(not np.isfinite(value) or value < 0.0 for value in values):
                    raise _RecoveryUnsupported(
                        f"source action error bounds for {name} must be finite and nonnegative"
                    )
                normalized_error_bounds[name] = values
        if len(layout.fiber_sectors) != len(reference.fiber_dimensions):
            raise _RecoveryUnsupported(
                "source layout does not cover every protected fiber"
            )
        for generator in presentation.generators:
            before = source[generator.name]
            after = target[generator.name]
            if before.antiunitary != after.antiunitary:
                raise _RecoveryUnsupported(
                    f"antiunitary parity mismatch for {generator.name}"
                )
            if (
                before.fiber_permutation != after.fiber_permutation
                or before.fiber_dimensions != after.fiber_dimensions
                or before.fiber_indices != after.fiber_indices
            ):
                raise _RecoveryUnsupported(
                    f"route layout mismatch for {generator.name}"
                )
        validate_presentation_action_relations(source, presentation)
        validate_presentation_action_relations(target, presentation)
        if not layout.basis_frame_hash:
            raise _RecoveryUnsupported(
                "source basis frame hash is required for fixed-target alignment"
            )
        pre_certificate = certify_joint_structure(
            source,
            presentation,
            layout=layout,
        )
        source_relations = certify_joint_block_actions(source, presentation)
        target_relations = certify_joint_block_actions(target, presentation)
        orbits = compile_action_orbits(source, presentation)
        gauge, objective, orbit_reports = _recover_for_anchor_section(
            source,
            target,
            presentation,
            orbits,
            reference_fiber_gauge=objective_reference,
        )
        target_basis_hash = derive_target_basis_hash(
            str(layout.basis_frame_hash),
            gauge,
        )
        target_layout = FiberSectorLayout(
            sector_names=layout.sector_names,
            fiber_sectors=layout.fiber_sectors,
            basis_frame_hash=target_basis_hash,
            declared_phase_signature=layout.declared_phase_signature,
        )
        post_certificate = certify_joint_structure(
            target,
            presentation,
            layout=target_layout,
        )
        witness = certify_common_fiber_gauge(
            source,
            target,
            gauge,
            presentation,
            source_basis_hash=str(layout.basis_frame_hash),
            target_basis_hash=target_basis_hash,
            absolute_error_bounds=normalized_error_bounds,
        )
        if (
            post_certificate.status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED
            and witness.status is CertificationStatus.CERTIFIED
        ):
            post_certificate = certify_joint_structure(
                target,
                presentation,
                layout=target_layout,
                block_absolute_error_bounds=dict(
                    witness.certification_bound_by_operation
                ),
            )
        transition = compare_joint_structure(
            pre_certificate,
            post_certificate,
            gauge_witness=witness,
        )
        if (
            post_certificate.status is not CertificationStatus.CERTIFIED
            or witness.status is not CertificationStatus.CERTIFIED
        ):
            raise _RecoveryUnsupported(
                "fixed target failed its common-gauge structure transaction"
            )
        if (
            pre_certificate.status is CertificationStatus.CERTIFIED
            and (
                transition.status is not CertificationStatus.CERTIFIED
                or not transition.is_monotone
            )
        ):
            raise _RecoveryUnsupported(
                "fixed target regressed a decidable source structure certificate"
            )
        grades = tuple(
            sector.grade
            for operation in post_certificate.operations.values()
            for sector in operation.source_sectors.values()
            if sector.grade is not None
        )
        grade = min(grades) if grades else None
        report = {
            "version": FIXED_TARGET_ALIGNMENT_VERSION,
            "status": CertificationStatus.CERTIFIED.value,
            "selection_policy": "fixed_target_closest_common_intertwiner",
            "closest_scope": (
                "fixed_supplied_target_and_certified_reference_gauge"
                if supplied_reference
                else "fixed_supplied_target"
            ),
            "target_was_modified": False,
            "reference_gauge_supplied": supplied_reference,
            "reference_gauge_unitarity_residual_max": max(
                reference_unitarity_residuals,
                default=0.0,
            ),
            "source_structure_classifier_status": pre_certificate.status.value,
            "certification_basis": (
                "certified_source_relations+exact_target_structure+"
                "bounded_common_gauge_witness"
                if any(
                    (
                        max(value, default=0.0)
                        if isinstance(value, tuple)
                        else value
                    )
                    > 0.0
                    for value in normalized_error_bounds.values()
                )
                else "exact_source_relations+exact_target_structure+common_gauge_witness"
            ),
            "source_action_absolute_error_bounds": {
                name: (
                    list(value) if isinstance(value, tuple) else float(value)
                )
                for name, value in normalized_error_bounds.items()
            },
            "closest_to_source_objective": objective,
            "closest_to_reference_objective": objective,
            "tie_break_policy": (
                "polar_projection_of_reference_gauge_objective_in_certified_"
                "schreier_intertwiner_space"
            ),
            "orbit_intertwiners": orbit_reports,
            "source_relation_certification": dict(source_relations),
            "target_relation_certification": dict(target_relations),
            "common_gauge_witness": witness.artifact(),
            "transition": transition.artifact(),
        }
        return StructureRecoveryResult(
            status=CertificationStatus.CERTIFIED,
            grade=grade,
            actions=target,
            fiber_gauge=gauge,
            certificate=post_certificate,
            common_gauge_witness=witness,
            report=report,
            version=FIXED_TARGET_ALIGNMENT_VERSION,
        )
    except (
        JointExactificationError,
        _RecoveryUnsupported,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return unknown(str(exc))


def recover_uniform_internal_action(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    layout: FiberSectorLayout,
    max_anchor_candidates: int = 4096,
    max_intertwiner_candidates: int = 4096,
) -> StructureRecoveryResult:
    """Recover an algebraic uniform action with a bounded structural proof.

    An already uniform action is reduced to one fiber per physical sector and
    standardized directly; this needs no central-Q anchor.  A genuinely
    scattered action still requires the equivariant-anchor Schreier proof.

    ``max_intertwiner_candidates`` is reserved as an explicit public bound for
    future singular-nullspace tie branches.  V1 accepts only a nonsingular
    closest projected intertwiner, so it never silently performs an unbounded
    search.
    """

    if int(max_anchor_candidates) <= 0 or int(max_intertwiner_candidates) <= 0:
        raise ValueError("structure-recovery candidate limits must be positive")
    if not layout.basis_frame_hash:
        return _unknown("source basis frame hash is required for recovery")
    try:
        validate_presentation_action_relations(actions, presentation)
        reference = actions[presentation.generators[0].name]
        if len(layout.fiber_sectors) != len(reference.fiber_dimensions):
            raise _RecoveryUnsupported(
                "fiber-sector layout does not cover the route actions"
            )
        pre_certificate = certify_joint_structure(
            actions,
            presentation,
            layout=layout,
        )
        if pre_certificate.status is not CertificationStatus.CERTIFIED:
            raise _RecoveryUnsupported(
                "source structure certificate is numerically undecidable"
            )
        sector_permutations = _sector_permutations(
            actions,
            presentation,
            layout,
        )
        already_uniform = all(
            sector.grade is StructureGrade.UNIFORM_INTERNAL
            for operation in pre_certificate.operations.values()
            for sector in operation.source_sectors.values()
        )
        if already_uniform:
            representatives = tuple(
                min(
                    fiber
                    for fiber, owner in enumerate(layout.fiber_sectors)
                    if owner == sector
                )
                for sector in range(len(layout.sector_names))
            )
            uniform_target = _uniform_target_actions(
                actions,
                presentation,
                layout,
                representatives,
            )
            standardized, standardizer_gauge, standard_report = (
                _standardize_uniform_sector_action(
                    uniform_target,
                    presentation,
                    layout,
                    sector_permutations,
                    representatives,
                )
            )
            standardizer_objective = float(
                sum(
                    np.linalg.norm(
                        value - np.eye(value.shape[0], dtype=np.complex128),
                        ord="fro",
                    )
                    ** 2
                    for value in standardizer_gauge
                )
            )
            orbits = compile_action_orbits(actions, presentation)
            gauge, objective, orbit_reports = _recover_for_anchor_section(
                actions,
                standardized,
                presentation,
                orbits,
            )
            objective_bound = float(
                4096.0
                * np.finfo(np.float64).eps
                * max(1.0, standardizer_objective, objective)
            )
            if objective > standardizer_objective + objective_bound:
                raise _RecoveryUnsupported(
                    "fixed-target closest intertwiner is worse than the "
                    "certified sector standardizer"
                )
            return _finalize_uniform_recovery(
                actions,
                standardized,
                gauge,
                presentation,
                layout=layout,
                pre_certificate=pre_certificate,
                report_details={
                    "selection_policy": "uniform_sector_reduction_standardization",
                    "tie_break_policy": (
                        "nearest_algebraic_sector_target_then_deterministic_standardizer"
                    ),
                    "closest_scope": "fixed_selected_algebraic_target",
                    "selected_sector_representatives": [
                        int(value) for value in representatives
                    ],
                    "closest_to_source_objective": objective,
                    "direct_sector_standardizer_objective": (
                        standardizer_objective
                    ),
                    "objective_certification_bound": objective_bound,
                    "sector_standardization": standard_report,
                    "orbit_intertwiners": orbit_reports,
                    "rejected_anchor_candidates": [],
                },
            )
        orbits = compile_action_orbits(actions, presentation)
        sections = _anchor_sections(
            orbits,
            layout,
            sector_permutations,
            actions,
            presentation,
            max_candidates=int(max_anchor_candidates),
        )
        if not sections:
            raise _RecoveryUnsupported("no equivariant anchor section exists")

        successful: list[
            tuple[
                float,
                tuple[int, ...],
                dict[str, BlockRouteAction],
                tuple[np.ndarray, ...],
                list[dict[str, Any]],
                dict[str, Any],
            ]
        ] = []
        failures: list[dict[str, Any]] = []
        for anchors in sections:
            try:
                target = _uniform_target_actions(
                    actions,
                    presentation,
                    layout,
                    anchors,
                )
                target, _sector_gauge, standard_report = (
                    _standardize_uniform_sector_action(
                        target,
                        presentation,
                        layout,
                        sector_permutations,
                        anchors,
                    )
                )
                gauge, objective, orbit_reports = _recover_for_anchor_section(
                    actions,
                    target,
                    presentation,
                    orbits,
                )
                successful.append(
                    (
                        objective,
                        anchors,
                        target,
                        gauge,
                        orbit_reports,
                        standard_report,
                    )
                )
            except (
                JointExactificationError,
                _RecoveryUnsupported,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                failures.append(
                    {
                        "anchors": [int(value) for value in anchors],
                        "reason": str(exc),
                    }
                )
        if not successful:
            raise _RecoveryUnsupported(
                "no anchor section produced a certified unitary intertwiner"
            )
        minimum_objective = min(row[0] for row in successful)
        objective_tie_bound = float(
            4096.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                minimum_objective,
                len(reference.fiber_dimensions)
                * max(reference.fiber_dimensions),
            )
        )
        indistinguishable = tuple(
            row
            for row in successful
            if row[0] <= minimum_objective + objective_tie_bound
        )
        selected = min(indistinguishable, key=lambda row: row[1])
        (
            objective,
            anchors,
            target,
            gauge,
            orbit_reports,
            standard_report,
        ) = selected
        target_basis_hash = derive_target_basis_hash(
            layout.basis_frame_hash,
            gauge,
        )
        target_layout = FiberSectorLayout(
            sector_names=layout.sector_names,
            fiber_sectors=layout.fiber_sectors,
            basis_frame_hash=target_basis_hash,
            declared_phase_signature=layout.declared_phase_signature,
        )
        post_certificate = certify_joint_structure(
            target,
            presentation,
            layout=target_layout,
        )
        if (
            post_certificate.status is not CertificationStatus.CERTIFIED
            or any(
                sector.grade is not StructureGrade.UNIFORM_INTERNAL
                for operation in post_certificate.operations.values()
                for sector in operation.source_sectors.values()
            )
        ):
            raise _RecoveryUnsupported(
                "reconstructed action is not certified uniform in every source sector"
            )
        witness = certify_common_fiber_gauge(
            actions,
            target,
            gauge,
            presentation,
            source_basis_hash=layout.basis_frame_hash,
            target_basis_hash=target_basis_hash,
        )
        transition = compare_joint_structure(
            pre_certificate,
            post_certificate,
            gauge_witness=witness,
        )
        if (
            witness.status is not CertificationStatus.CERTIFIED
            or transition.status is not CertificationStatus.CERTIFIED
            or not transition.is_monotone
        ):
            raise _RecoveryUnsupported(
                "uniform reconstruction failed its common-gauge transaction"
            )
        relation_certificate = certify_joint_block_actions(target, presentation)
        report = {
            "version": STRUCTURE_RECOVERY_V1,
            "status": CertificationStatus.CERTIFIED.value,
            "grade": "uniform_internal",
            "selection_policy": "equivariant_anchor_schreier_intertwiner",
            "tie_break_policy": (
                "selected_algebraic_target_then_minimum_full_fiber_gauge_"
                "objective_with_certified_ties_lexicographic"
            ),
            "closest_scope": "fixed_selected_algebraic_target",
            "anchor_objective_tie_bound": objective_tie_bound,
            "anchor_candidates_within_tie_bound": len(indistinguishable),
            "anchor_candidates_considered": len(sections),
            "successful_anchor_candidates": len(successful),
            "selected_anchors": [int(value) for value in anchors],
            "closest_to_source_objective": objective,
            "sector_standardization": standard_report,
            "canonicality": "canonical_up_to_continuous_stabilizer",
            "residual_stabilizer_dimension": None,
            "orbit_intertwiners": orbit_reports,
            "common_gauge_witness": witness.artifact(),
            "transition": transition.artifact(),
            "relation_certification": dict(relation_certificate),
            "rejected_anchor_candidates": failures,
        }
        return StructureRecoveryResult(
            status=CertificationStatus.CERTIFIED,
            grade=StructureGrade.UNIFORM_INTERNAL,
            actions=target,
            fiber_gauge=gauge,
            certificate=post_certificate,
            common_gauge_witness=witness,
            report=report,
        )
    except (
        JointExactificationError,
        _RecoveryUnsupported,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return _unknown(str(exc))
