"""Joint exactification primitives for semilinear block-route actions.

The objects in this module store only the structurally nonzero route blocks.
They never infer numerical zeros from entry magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import operator
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

from ..identity import hash_array


class JointExactificationError(ValueError):
    """Input data cannot define a certified joint symmetry action."""


def _readonly_complex(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.complex128, order="C", copy=True)
    array.setflags(write=False)
    return array


def _unitarity_residual(matrix: np.ndarray) -> float:
    identity = np.eye(matrix.shape[1], dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro"))


def _roundoff_unitarity_bound(matrix: np.ndarray) -> float:
    """Return a dimension/scale-derived bound, not an engineering tolerance."""

    dimension = int(matrix.shape[1])
    epsilon = float(np.finfo(np.float64).eps)
    product = float(dimension) * epsilon
    if product >= 1.0:
        raise JointExactificationError(
            "matrix dimension is too large for a finite floating certification bound"
        )
    gamma = product / (1.0 - product)
    absolute_product_scale = float(
        np.linalg.norm(np.abs(matrix).T @ np.abs(matrix), ord="fro")
    )
    identity_scale = float(np.sqrt(dimension))
    return float(8.0 * (gamma * absolute_product_scale + epsilon * identity_scale))


def _certification_bound(
    value: float | None,
    *,
    default: float,
) -> float:
    bound = float(default if value is None else value)
    if not np.isfinite(bound) or bound < 0.0:
        raise JointExactificationError(
            f"unitarity certification bound must be finite and nonnegative, got {bound}"
        )
    return bound


def _validate_unitarity(
    matrix: np.ndarray,
    *,
    certification_bound: float,
    label: str,
) -> None:
    residual = _unitarity_residual(matrix)
    if residual > certification_bound:
        raise JointExactificationError(
            f"{label} unitarity residual {residual:.6e} exceeds certification "
            f"bound {certification_bound:.6e}"
        )


@dataclass(frozen=True)
class SemilinearBlock:
    """A certified matrix block followed by optional complex conjugation."""

    matrix: np.ndarray = field(repr=False, compare=False)
    antiunitary: bool
    unitarity_certification_bound: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        matrix = _readonly_complex(self.matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise JointExactificationError(
                f"semilinear block matrix must be square, got shape {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise JointExactificationError("semilinear block matrix must be finite")
        bound = _certification_bound(
            self.unitarity_certification_bound,
            default=_roundoff_unitarity_bound(matrix),
        )
        _validate_unitarity(
            matrix,
            certification_bound=bound,
            label="semilinear block",
        )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "antiunitary", bool(self.antiunitary))
        object.__setattr__(self, "unitarity_certification_bound", bound)


def semilinear_kappa(matrix: np.ndarray, antiunitary: bool) -> np.ndarray:
    """Apply the scalar-field automorphism belonging to a semilinear action."""

    array = np.asarray(matrix, dtype=np.complex128)
    return array.conj() if antiunitary else array


def compose_semilinear(
    outer: SemilinearBlock,
    inner: SemilinearBlock,
) -> SemilinearBlock:
    """Compose ``outer`` after ``inner`` using ``B kappa_a(C)``."""

    if outer.matrix.shape != inner.matrix.shape:
        raise JointExactificationError(
            "semilinear blocks must have equal dimensions for composition"
        )
    transformed_inner = semilinear_kappa(inner.matrix, outer.antiunitary)
    matrix = outer.matrix @ transformed_inner
    propagated_bound = (
        float(np.linalg.norm(transformed_inner, ord=2)) ** 2
        * float(outer.unitarity_certification_bound)
        + float(inner.unitarity_certification_bound)
        + _roundoff_unitarity_bound(matrix)
    )
    return SemilinearBlock(
        matrix,
        outer.antiunitary ^ inner.antiunitary,
        unitarity_certification_bound=propagated_bound,
    )


def inverse_semilinear(value: SemilinearBlock) -> SemilinearBlock:
    """Return the inverse of ``B K^a`` in the semilinear product."""

    matrix = value.matrix.conj().T
    if value.antiunitary:
        matrix = matrix.conj()
    propagated_bound = float(
        float(value.unitarity_certification_bound)
        + _roundoff_unitarity_bound(matrix)
    )
    return SemilinearBlock(
        matrix,
        value.antiunitary,
        unitarity_certification_bound=propagated_bound,
    )


def _integer_tuple(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        try:
            result.append(int(operator.index(value)))
        except TypeError as exc:
            raise JointExactificationError(f"{label} entries must be integers") from exc
    return tuple(result)


def _canonical_fiber_indices(
    dimensions: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    offset = 0
    for dimension in dimensions:
        groups.append(tuple(range(offset, offset + int(dimension))))
        offset += int(dimension)
    return tuple(groups)


def _validated_fiber_indices(
    values: Sequence[Sequence[int]] | None,
    *,
    dimensions: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    if values is None:
        return _canonical_fiber_indices(dimensions)
    groups = tuple(
        _integer_tuple(group, label="fiber indices")
        for group in values
    )
    if len(groups) != len(dimensions):
        raise JointExactificationError(
            "fiber indices must contain one index group per fiber"
        )
    for fiber, (group, dimension) in enumerate(zip(groups, dimensions)):
        if len(group) != int(dimension):
            raise JointExactificationError(
                f"fiber indices group {fiber} has {len(group)} entries, "
                f"expected {int(dimension)}"
            )
    flat = tuple(index for group in groups for index in group)
    total_dimension = int(sum(int(value) for value in dimensions))
    if sorted(flat) != list(range(total_dimension)):
        raise JointExactificationError(
            "fiber indices must be a complete disjoint partition of "
            f"range({total_dimension}), got {groups}"
        )
    return groups


@dataclass(frozen=True)
class BlockRouteAction:
    """A block-monomial action represented by one U(d) block per source fiber."""

    name: str
    antiunitary: bool
    fiber_permutation: tuple[int, ...]
    fiber_dimensions: tuple[int, ...]
    route_blocks: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    fiber_indices: tuple[tuple[int, ...], ...] | None = field(
        default=None,
        kw_only=True,
    )
    unitarity_certification_bound: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise JointExactificationError("block-route action name must be nonempty")
        permutation = _integer_tuple(
            self.fiber_permutation,
            label="fiber permutation",
        )
        dimensions = _integer_tuple(
            self.fiber_dimensions,
            label="fiber dimensions",
        )
        fiber_count = len(dimensions)
        if fiber_count == 0 or any(value <= 0 for value in dimensions):
            raise JointExactificationError(
                "fiber dimensions must contain positive dimensions"
            )
        indices = _validated_fiber_indices(
            self.fiber_indices,
            dimensions=dimensions,
        )
        if len(permutation) != fiber_count or sorted(permutation) != list(
            range(fiber_count)
        ):
            raise JointExactificationError(
                f"fiber permutation must be a permutation of range({fiber_count}), "
                f"got {permutation}"
            )
        blocks = tuple(_readonly_complex(block) for block in self.route_blocks)
        if len(blocks) != fiber_count:
            raise JointExactificationError(
                f"block-route action requires one route block per source fiber; "
                f"got {len(blocks)} for {fiber_count} fibers"
            )
        for source, target in enumerate(permutation):
            if dimensions[target] != dimensions[source]:
                raise JointExactificationError(
                    "fiber dimension must be preserved along every permutation route; "
                    f"source {source} has {dimensions[source]}, target {target} has "
                    f"{dimensions[target]}"
                )
            expected_shape = (dimensions[target], dimensions[source])
            if blocks[source].shape != expected_shape:
                raise JointExactificationError(
                    f"route block {source} has shape {blocks[source].shape}, "
                    f"expected shape {expected_shape}"
                )
            if not np.all(np.isfinite(blocks[source])):
                raise JointExactificationError(f"route block {source} must be finite")
        default_bound = max(_roundoff_unitarity_bound(block) for block in blocks)
        bound = _certification_bound(
            self.unitarity_certification_bound,
            default=default_bound,
        )
        for source, block in enumerate(blocks):
            _validate_unitarity(
                block,
                certification_bound=bound,
                label=f"route block {source}",
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "antiunitary", bool(self.antiunitary))
        object.__setattr__(self, "fiber_permutation", permutation)
        object.__setattr__(self, "fiber_dimensions", dimensions)
        object.__setattr__(self, "route_blocks", blocks)
        object.__setattr__(self, "fiber_indices", indices)
        object.__setattr__(self, "unitarity_certification_bound", bound)


def materialize_block_route_action(action: BlockRouteAction) -> np.ndarray:
    """Materialize an action by assigning complete blocks on known routes."""

    dimension = int(sum(action.fiber_dimensions))
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for source, target in enumerate(action.fiber_permutation):
        rows = action.fiber_indices[target]
        columns = action.fiber_indices[source]
        matrix[np.ix_(rows, columns)] = action.route_blocks[source]
    return matrix


def extract_block_route_action(
    matrix: np.ndarray,
    *,
    name: str,
    antiunitary: bool,
    fiber_indices: Sequence[Sequence[int]],
    fiber_permutation: Sequence[int],
    off_route_bound: float,
    unitarity_certification_bound: float | None = None,
) -> BlockRouteAction:
    """Extract declared route blocks without modifying any block entry."""

    dense = np.asarray(matrix, dtype=np.complex128)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise JointExactificationError(
            f"dense block-route matrix must be square, got shape {dense.shape}"
        )
    if not np.all(np.isfinite(dense)):
        raise JointExactificationError("dense block-route matrix must be finite")
    dimensions = tuple(len(tuple(group)) for group in fiber_indices)
    indices = _validated_fiber_indices(
        fiber_indices,
        dimensions=dimensions,
    )
    if sum(dimensions) != dense.shape[0]:
        raise JointExactificationError(
            "fiber indices total dimension must equal the dense matrix dimension; "
            f"got {sum(dimensions)} and {dense.shape[0]}"
        )
    permutation = _integer_tuple(
        fiber_permutation,
        label="fiber permutation",
    )
    fiber_count = len(dimensions)
    if len(permutation) != fiber_count or sorted(permutation) != list(
        range(fiber_count)
    ):
        raise JointExactificationError(
            f"fiber permutation must be a permutation of range({fiber_count}), "
            f"got {permutation}"
        )
    for source, target in enumerate(permutation):
        if dimensions[target] != dimensions[source]:
            raise JointExactificationError(
                "fiber dimension must be preserved along every permutation route; "
                f"source {source} has {dimensions[source]}, target {target} has "
                f"{dimensions[target]}"
            )
    bound = float(off_route_bound)
    if not np.isfinite(bound) or bound < 0.0:
        raise JointExactificationError(
            f"off-route certification bound must be finite and nonnegative, got {bound}"
        )
    route_mask = np.zeros(dense.shape, dtype=bool)
    blocks: list[np.ndarray] = []
    for source, target in enumerate(permutation):
        rows = indices[target]
        columns = indices[source]
        route_mask[np.ix_(rows, columns)] = True
        blocks.append(np.array(dense[np.ix_(rows, columns)], copy=True, order="C"))
    off_route_residual = float(np.linalg.norm(dense[~route_mask]))
    if off_route_residual > bound:
        raise JointExactificationError(
            f"off-route residual {off_route_residual:.6e} exceeds certification "
            f"bound {bound:.6e}"
        )
    return BlockRouteAction(
        name=name,
        antiunitary=antiunitary,
        fiber_permutation=permutation,
        fiber_dimensions=dimensions,
        route_blocks=tuple(blocks),
        fiber_indices=indices,
        unitarity_certification_bound=unitarity_certification_bound,
    )


@dataclass(frozen=True)
class MagneticGenerator:
    """A named unitary or antiunitary generator."""

    name: str
    antiunitary: bool

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise JointExactificationError("magnetic generator name must be nonempty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "antiunitary", bool(self.antiunitary))


def _v1_central_phase(value: Any, *, label: str) -> complex:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = tuple(value)
        if len(parts) != 2:
            raise JointExactificationError(
                f"{label} must encode a scalar phase, got {value!r}"
            )
        phase = complex(float(parts[0]), float(parts[1]))
    else:
        phase = complex(value)
    if not np.isfinite(phase.real) or not np.isfinite(phase.imag):
        raise JointExactificationError(f"{label} must be finite")
    if phase not in {complex(1.0, 0.0), complex(-1.0, 0.0)}:
        raise JointExactificationError(
            f"{label} must be an explicitly declared V1 central phase +1 or -1, "
            f"got {phase}"
        )
    return phase


@dataclass(frozen=True)
class MagneticRelation:
    """A relation ``lhs = central_phase * rhs`` between generator words."""

    name: str
    lhs: tuple[str, ...]
    rhs: tuple[str, ...]
    central_phase: complex

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise JointExactificationError("magnetic relation name must be nonempty")
        lhs = tuple(str(value).strip() for value in self.lhs)
        rhs = tuple(str(value).strip() for value in self.rhs)
        if any(not value for value in (*lhs, *rhs)):
            raise JointExactificationError("magnetic relation words must name generators")
        phase = _v1_central_phase(
            self.central_phase,
            label=f"magnetic relation {name!r} central phase",
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "central_phase", phase)


@dataclass(frozen=True)
class MagneticPresentation:
    """A finite central magnetic presentation used by joint exactification."""

    generators: tuple[MagneticGenerator, ...]
    relations: tuple[MagneticRelation, ...]
    central_phases: tuple[complex, ...]
    source: str

    def __post_init__(self) -> None:
        generators = tuple(self.generators)
        relations = tuple(self.relations)
        names = tuple(generator.name for generator in generators)
        if not names or len(set(names)) != len(names):
            raise JointExactificationError(
                "magnetic presentation generators must be nonempty and unique"
            )
        relation_names = tuple(relation.name for relation in relations)
        if len(set(relation_names)) != len(relation_names):
            raise JointExactificationError(
                "magnetic presentation relation names must be unique"
            )
        generator_names = set(names)
        for relation in relations:
            unknown = set((*relation.lhs, *relation.rhs)) - generator_names
            if unknown:
                raise JointExactificationError(
                    f"magnetic relation {relation.name!r} references unknown "
                    f"generators {sorted(unknown)}"
                )
        central_phases = tuple(
            _v1_central_phase(value, label="presentation central phase")
            for value in self.central_phases
        )
        if (
            not central_phases
            or central_phases[0] != complex(1.0, 0.0)
            or len(set(central_phases)) != len(central_phases)
        ):
            raise JointExactificationError(
                "presentation central phases must be unique and start with +1"
            )
        if any(
            relation.central_phase not in central_phases
            for relation in relations
        ):
            raise JointExactificationError(
                "every relation central phase must belong to the declared central kernel"
            )
        source = str(self.source).strip()
        if not source:
            raise JointExactificationError("magnetic presentation source must be nonempty")
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "central_phases", central_phases)
        object.__setattr__(self, "source", source)


_SUPPORTED_PRESENTATION_ORDERS: Mapping[frozenset[str], tuple[str, ...]] = {
    frozenset(("TR", "C2")): ("TR", "C2"),
    frozenset(("C3z", "C2T")): ("C3z", "C2T"),
    frozenset(("TR", "C3z")): ("TR", "C3z"),
    frozenset(("TR", "C3z", "C2")): ("TR", "C3z", "C2"),
}

_EXPECTED_ANTIUNITARY: Mapping[str, bool] = {
    "TR": True,
    "C3z": False,
    "C2": False,
    "C2T": True,
}

_EXPECTED_ACTION_TYPE: Mapping[str, str] = {
    "TR": "negation",
    "C3z": "rotation",
    "C2": "reflection",
    "C2T": "reflection",
}

_EXPECTED_POWER: Mapping[str, int] = {
    "TR": 2,
    "C3z": 3,
    "C2": 2,
    "C2T": 2,
}


def _normalized_angle(value: Any, *, period: float, label: str) -> float:
    angle = float(value)
    if not np.isfinite(angle):
        raise JointExactificationError(f"{label} must be finite")
    normalized = float(angle % period)
    scale = max(1.0, abs(angle), period)
    floating_bound = float(64.0 * np.finfo(np.float64).eps * scale)
    if abs(normalized - period) <= floating_bound or abs(normalized) <= floating_bound:
        return 0.0
    return normalized


def _action_map_signature(
    value: Any,
    *,
    label: str,
) -> tuple[str, float | None]:
    if not isinstance(value, Mapping):
        raise JointExactificationError(f"{label} must be explicit action metadata")
    if value.get("in_model_frame") is False:
        raise JointExactificationError(f"{label} must be expressed in the model frame")
    action_type = str(value.get("type", "")).strip().lower()
    if action_type in {"identity", "negation"}:
        return action_type, None
    if action_type == "rotation":
        if "angle_deg" not in value:
            raise JointExactificationError(f"{label} rotation must declare angle_deg")
        return action_type, _normalized_angle(
            value["angle_deg"],
            period=360.0,
            label=f"{label} rotation angle",
        )
    if action_type == "reflection":
        if "axis_deg" not in value:
            raise JointExactificationError(f"{label} reflection must declare axis_deg")
        return action_type, _normalized_angle(
            value["axis_deg"],
            period=180.0,
            label=f"{label} reflection axis",
        )
    raise JointExactificationError(
        f"{label} has unsupported explicit action type {action_type!r}"
    )


def _angle_signatures_equal(
    left: tuple[str, float | None],
    right: tuple[str, float | None],
) -> bool:
    if left[0] != right[0]:
        return False
    if left[1] is None or right[1] is None:
        return left[1] is None and right[1] is None
    scale = max(1.0, abs(left[1]), abs(right[1]))
    bound = float(64.0 * np.finfo(np.float64).eps * scale)
    return bool(abs(left[1] - right[1]) <= bound)


def _declared_action_signature(
    record: Mapping[str, Any],
    *,
    name: str,
) -> tuple[tuple[str, float | None], str, str]:
    candidates: list[Mapping[str, Any]] = []
    for key in ("declared_model_action", "model_action"):
        value = record.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    if not candidates and isinstance(record.get("k_map"), Mapping):
        candidates.append(record)
    if not candidates:
        raise JointExactificationError(
            f"operation {name!r} lacks explicit declared model action metadata"
        )

    signatures: list[tuple[tuple[str, float | None], str, str]] = []
    for action in candidates:
        if "antiunitary" in action and bool(action["antiunitary"]) != bool(
            _EXPECTED_ANTIUNITARY[name]
        ):
            raise JointExactificationError(
                f"operation {name!r} action antiunitary parity is inconsistent"
            )
        k_signature = _action_map_signature(
            action.get("k_map"),
            label=f"operation {name!r} k action",
        )
        q_signature = _action_map_signature(
            action.get("q_map"),
            label=f"operation {name!r} Q action",
        )
        if not _angle_signatures_equal(k_signature, q_signature):
            raise JointExactificationError(
                f"operation {name!r} explicit k/Q actions do not match"
            )
        signatures.append(
            (
                k_signature,
                str(action.get("sector_map", "")).strip(),
                str(action.get("valley_map", "")).strip(),
            )
        )
    first = signatures[0]
    for candidate in signatures[1:]:
        if (
            not _angle_signatures_equal(first[0], candidate[0])
            or first[1:] != candidate[1:]
        ):
            raise JointExactificationError(
                f"operation {name!r} has ambiguous declared model actions"
            )
    expected_type = _EXPECTED_ACTION_TYPE[name]
    if first[0][0] != expected_type:
        raise JointExactificationError(
            f"operation {name!r} requires explicit {expected_type} k/Q actions, "
            f"got {first[0][0]!r}"
        )
    if name == "C3z":
        angle = float(first[0][1])
        allowed = (120.0, 240.0)
        if not any(
            _angle_signatures_equal(("rotation", angle), ("rotation", item))
            for item in allowed
        ):
            raise JointExactificationError(
                f"operation 'C3z' rotation angle must have order three, got {angle}"
            )
    return first


def _manifest_power_phase(record: Mapping[str, Any], *, name: str) -> complex:
    raw_relations = record.get("group_relations")
    if not isinstance(raw_relations, Sequence) or isinstance(
        raw_relations,
        (str, bytes),
    ):
        raise JointExactificationError(
            f"operation {name!r} must declare manifest group_relations"
        )
    matches: list[Mapping[str, Any]] = []
    for relation in raw_relations:
        if not isinstance(relation, Mapping):
            continue
        if str(relation.get("type", "power")) != "power":
            continue
        relation_operation = str(relation.get("operation", name)).strip()
        if relation_operation == name:
            matches.append(relation)
    if len(matches) != 1:
        raise JointExactificationError(
            f"operation {name!r} requires exactly one unambiguous power relation"
        )
    relation = matches[0]
    expected_power = _EXPECTED_POWER[name]
    if int(relation.get("power", 0)) != expected_power:
        raise JointExactificationError(
            f"operation {name!r} power relation must use power {expected_power}"
        )
    return _v1_central_phase(
        relation.get("phase"),
        label=f"operation {name!r} power phase",
    )


def compile_continuum_magnetic_presentation(
    operations: Sequence[Mapping[str, Any]],
) -> MagneticPresentation:
    """Compile a recognized magnetic presentation from explicit KP metadata."""

    records: dict[str, Mapping[str, Any]] = {}
    for raw_record in operations:
        if not isinstance(raw_record, Mapping):
            raise JointExactificationError("operation records must be mappings")
        name = str(raw_record.get("name", raw_record.get("operation", ""))).strip()
        operation = str(raw_record.get("operation", name)).strip()
        if not name or name != operation:
            raise JointExactificationError(
                f"operation record has ambiguous name/operation fields: {name!r}, "
                f"{operation!r}"
            )
        if name in records:
            raise JointExactificationError(f"duplicate operation record {name!r}")
        records[name] = raw_record

    operation_set = frozenset(records)
    generator_order = _SUPPORTED_PRESENTATION_ORDERS.get(operation_set)
    if generator_order is None:
        raise JointExactificationError(
            "unsupported or ambiguous continuum magnetic operation set "
            f"{sorted(operation_set)}"
        )

    power_phases: dict[str, complex] = {}
    generators: list[MagneticGenerator] = []
    relations: list[MagneticRelation] = []
    for name in generator_order:
        record = records[name]
        antiunitary = bool(record.get("antiunitary", False))
        expected_antiunitary = _EXPECTED_ANTIUNITARY[name]
        if antiunitary != expected_antiunitary:
            raise JointExactificationError(
                f"operation {name!r} antiunitary parity must be "
                f"{expected_antiunitary}"
            )
        _declared_action_signature(record, name=name)
        phase = _manifest_power_phase(record, name=name)
        power_phases[name] = phase
        generators.append(MagneticGenerator(name, antiunitary))
        power = _EXPECTED_POWER[name]
        relations.append(
            MagneticRelation(
                f"{name}^{power}",
                lhs=tuple(name for _ in range(power)),
                rhs=(),
                central_phase=phase,
            )
        )

    if operation_set == frozenset(("TR", "C2")):
        relations.append(
            MagneticRelation(
                "TR_C2_commute",
                lhs=("TR", "C2"),
                rhs=("C2", "TR"),
                central_phase=1.0,
            )
        )
    elif operation_set == frozenset(("C3z", "C2T")):
        relations.append(
            MagneticRelation(
                "C2T_C3z_dihedral",
                lhs=("C2T", "C3z", "C2T"),
                rhs=("C3z", "C3z"),
                central_phase=power_phases["C2T"] / power_phases["C3z"],
            )
        )
    elif operation_set == frozenset(("TR", "C3z")):
        relations.append(
            MagneticRelation(
                "TR_C3z_commute",
                lhs=("TR", "C3z"),
                rhs=("C3z", "TR"),
                central_phase=1.0,
            )
        )
    elif operation_set == frozenset(("TR", "C3z", "C2")):
        relations.extend(
            (
                MagneticRelation(
                    "TR_C3z_commute",
                    lhs=("TR", "C3z"),
                    rhs=("C3z", "TR"),
                    central_phase=1.0,
                ),
                MagneticRelation(
                    "TR_C2_commute",
                    lhs=("TR", "C2"),
                    rhs=("C2", "TR"),
                    central_phase=1.0,
                ),
                MagneticRelation(
                    "C2_C3z_dihedral",
                    lhs=("C2", "C3z", "C2"),
                    rhs=("C3z", "C3z"),
                    central_phase=power_phases["C2"] / power_phases["C3z"],
                ),
            )
        )

    central_phases = (complex(1.0, 0.0),)
    if any(
        relation.central_phase == complex(-1.0, 0.0)
        for relation in relations
    ):
        central_phases = (*central_phases, complex(-1.0, 0.0))
    return MagneticPresentation(
        generators=tuple(generators),
        relations=tuple(relations),
        central_phases=central_phases,
        source="kp_symm_manifest_explicit_actions_v1",
    )


def _evaluate_discrete_word(
    word: Sequence[str],
    *,
    actions: Mapping[str, BlockRouteAction],
    fiber_count: int,
) -> tuple[tuple[int, ...], bool]:
    permutation = tuple(range(fiber_count))
    antiunitary = False
    for name in reversed(tuple(word)):
        action = actions[name]
        permutation = tuple(
            action.fiber_permutation[permutation[source]]
            for source in range(fiber_count)
        )
        antiunitary ^= action.antiunitary
    return permutation, antiunitary


def validate_presentation_action_relations(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> None:
    """Require exact closure of every relation on permutation/parity data."""

    generator_names = tuple(generator.name for generator in presentation.generators)
    if set(actions) != set(generator_names):
        raise JointExactificationError(
            "block-route actions must match the presentation generators exactly"
        )
    reference: BlockRouteAction | None = None
    for generator in presentation.generators:
        action = actions[generator.name]
        if action.name != generator.name or action.antiunitary != generator.antiunitary:
            raise JointExactificationError(
                f"action {generator.name!r} is inconsistent with its generator"
            )
        if reference is None:
            reference = action
        elif (
            action.fiber_dimensions != reference.fiber_dimensions
            or action.fiber_indices != reference.fiber_indices
        ):
            raise JointExactificationError(
                "all presentation actions must use one common fiber layout"
            )
    if reference is None:
        raise JointExactificationError("presentation must contain generators")
    fiber_count = len(reference.fiber_dimensions)
    for relation in presentation.relations:
        lhs = _evaluate_discrete_word(
            relation.lhs,
            actions=actions,
            fiber_count=fiber_count,
        )
        rhs = _evaluate_discrete_word(
            relation.rhs,
            actions=actions,
            fiber_count=fiber_count,
        )
        if lhs != rhs:
            raise JointExactificationError(
                f"magnetic relation {relation.name!r} does not close on exact "
                f"fiber permutation/parity data: lhs={lhs}, rhs={rhs}"
            )


@dataclass(frozen=True)
class QuotientGroupElement:
    """One canonical element of the permutation/parity quotient group."""

    canonical_word: tuple[str, ...]
    antiunitary: bool
    fiber_permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        word = tuple(str(value).strip() for value in self.canonical_word)
        if any(not value for value in word):
            raise JointExactificationError("quotient-group words must name generators")
        permutation = _integer_tuple(
            self.fiber_permutation,
            label="quotient-group fiber permutation",
        )
        if sorted(permutation) != list(range(len(permutation))):
            raise JointExactificationError(
                "quotient-group fiber permutation must be bijective"
            )
        object.__setattr__(self, "canonical_word", word)
        object.__setattr__(self, "antiunitary", bool(self.antiunitary))
        object.__setattr__(self, "fiber_permutation", permutation)


@dataclass(frozen=True)
class ActionOrbit:
    """Canonical orbit, root transporters, and its full discrete stabilizer."""

    root: int
    fibers: tuple[int, ...]
    transporter_words: tuple[tuple[str, ...], ...]
    stabilizer_words: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        try:
            root = int(operator.index(self.root))
        except TypeError as exc:
            raise JointExactificationError("action-orbit root must be an integer") from exc
        fibers = _integer_tuple(self.fibers, label="action-orbit fibers")
        if not fibers or tuple(sorted(set(fibers))) != fibers or root != fibers[0]:
            raise JointExactificationError(
                "action-orbit fibers must be sorted, unique, and start with the root"
            )
        transporters = tuple(
            tuple(str(value).strip() for value in word)
            for word in self.transporter_words
        )
        stabilizers = tuple(
            tuple(str(value).strip() for value in word)
            for word in self.stabilizer_words
        )
        if len(transporters) != len(fibers) or transporters[0] != ():
            raise JointExactificationError(
                "action-orbit transporters must label every fiber and start with identity"
            )
        if not stabilizers or stabilizers[0] != ():
            raise JointExactificationError(
                "action-orbit stabilizer words must start with identity"
            )
        if any(
            not value
            for word in (*transporters, *stabilizers)
            for value in word
        ):
            raise JointExactificationError(
                "action-orbit words must contain nonempty generator names"
            )
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "fibers", fibers)
        object.__setattr__(self, "transporter_words", transporters)
        object.__setattr__(self, "stabilizer_words", stabilizers)


def _positive_limit(value: int, *, label: str) -> int:
    try:
        limit = int(operator.index(value))
    except TypeError as exc:
        raise JointExactificationError(f"{label} must be a positive integer") from exc
    if limit <= 0:
        raise JointExactificationError(f"{label} must be a positive integer")
    return limit


def _compose_discrete_actions(
    outer_permutation: Sequence[int],
    outer_antiunitary: bool,
    inner_permutation: Sequence[int],
    inner_antiunitary: bool,
) -> tuple[tuple[int, ...], bool]:
    if len(outer_permutation) != len(inner_permutation):
        raise JointExactificationError(
            "cannot compose discrete actions with different fiber counts"
        )
    return (
        tuple(
            int(outer_permutation[int(inner_permutation[source])])
            for source in range(len(outer_permutation))
        ),
        bool(outer_antiunitary) ^ bool(inner_antiunitary),
    )


def compile_quotient_group_elements(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    max_group_size: int = 4096,
    max_word_length: int = 128,
) -> tuple[QuotientGroupElement, ...]:
    """Enumerate shortest canonical words in the finite discrete quotient."""

    group_size_limit = _positive_limit(max_group_size, label="group-size limit")
    word_length_limit = _positive_limit(
        max_word_length,
        label="word-length limit",
    )
    validate_presentation_action_relations(actions, presentation)
    generator_names = tuple(
        generator.name for generator in presentation.generators
    )
    fiber_count = len(actions[generator_names[0]].fiber_permutation)
    identity = tuple(range(fiber_count))
    elements: list[QuotientGroupElement] = [
        QuotientGroupElement((), False, identity)
    ]
    known: dict[tuple[tuple[int, ...], bool], int] = {
        (identity, False): 0,
    }
    cursor = 0
    while cursor < len(elements):
        current = elements[cursor]
        cursor += 1
        for generator_name in generator_names:
            generator = actions[generator_name]
            permutation, antiunitary = _compose_discrete_actions(
                current.fiber_permutation,
                current.antiunitary,
                generator.fiber_permutation,
                generator.antiunitary,
            )
            key = (permutation, antiunitary)
            if key in known:
                continue
            word = (*current.canonical_word, generator_name)
            if len(word) > word_length_limit:
                raise JointExactificationError(
                    "quotient-group word-length limit exceeded while proving closure: "
                    f"limit={word_length_limit}, candidate={word}"
                )
            if len(elements) >= group_size_limit:
                raise JointExactificationError(
                    "quotient-group group-size limit exceeded while proving closure: "
                    f"limit={group_size_limit}"
                )
            known[key] = len(elements)
            elements.append(
                QuotientGroupElement(word, antiunitary, permutation)
            )
    return tuple(elements)


def compile_action_orbits(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    max_group_size: int = 4096,
    max_word_length: int = 128,
) -> tuple[ActionOrbit, ...]:
    """Decompose the exact fiber action into canonical orbit/stabilizer data."""

    elements = compile_quotient_group_elements(
        actions,
        presentation,
        max_group_size=max_group_size,
        max_word_length=max_word_length,
    )
    fiber_count = len(elements[0].fiber_permutation)
    unassigned = set(range(fiber_count))
    orbits: list[ActionOrbit] = []
    while unassigned:
        root = min(unassigned)
        transporter_by_fiber: dict[int, tuple[str, ...]] = {}
        stabilizer_words: list[tuple[str, ...]] = []
        for element in elements:
            target = int(element.fiber_permutation[root])
            transporter_by_fiber.setdefault(target, element.canonical_word)
            if target == root:
                stabilizer_words.append(element.canonical_word)
        fibers = tuple(sorted(transporter_by_fiber))
        if len(fibers) * len(stabilizer_words) != len(elements):
            raise JointExactificationError(
                "orbit-stabilizer identity failed for compiled discrete action: "
                f"root={root}, orbit={len(fibers)}, stabilizer="
                f"{len(stabilizer_words)}, group={len(elements)}"
            )
        orbit = ActionOrbit(
            root=root,
            fibers=fibers,
            transporter_words=tuple(
                transporter_by_fiber[fiber] for fiber in fibers
            ),
            stabilizer_words=tuple(stabilizer_words),
        )
        orbits.append(orbit)
        unassigned.difference_update(fibers)
    return tuple(orbits)


def _u1_word_coefficients(
    word: Sequence[str],
    source: int,
    *,
    actions: Mapping[str, BlockRouteAction],
    variable_indices: Mapping[tuple[str, int], int],
    variable_count: int,
) -> np.ndarray:
    """Return the signed route-phase linear form for one semilinear word."""

    names = tuple(word)
    route_sources = [0] * len(names)
    current_source = int(source)
    for position in range(len(names) - 1, -1, -1):
        name = names[position]
        route_sources[position] = current_source
        current_source = actions[name].fiber_permutation[current_source]
    coefficients = np.zeros(variable_count, dtype=np.float64)
    left_antiunitary = False
    for position, name in enumerate(names):
        sign = -1.0 if left_antiunitary else 1.0
        coefficients[variable_indices[(name, route_sources[position])]] += sign
        left_antiunitary ^= actions[name].antiunitary
    return coefficients


def _central_phase_angle(phase: complex) -> float:
    return 0.0 if complex(phase) == complex(1.0, 0.0) else float(np.pi)


def _u1_angles(
    actions: Mapping[str, BlockRouteAction],
    generator_names: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [
                np.angle(complex(block[0, 0]))
                for block in actions[name].route_blocks
            ],
            dtype=np.float64,
        )
        for name in generator_names
    }


def _u1_relation_residual_table(
    *,
    constraint_matrix: np.ndarray,
    angle_vector: np.ndarray,
    target_angles: np.ndarray,
    relation_names: Sequence[str],
) -> tuple[dict[str, dict[str, float]], float]:
    residuals = np.angle(
        np.exp(1.0j * (constraint_matrix @ angle_vector - target_angles))
    )
    table: dict[str, dict[str, float]] = {}
    for name in dict.fromkeys(str(value) for value in relation_names):
        values = residuals[
            np.asarray([value == name for value in relation_names], dtype=bool)
        ]
        table[name] = {
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "max": float(np.max(np.abs(values))),
        }
    maximum = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    return table, maximum


def project_u1_relations(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, BlockRouteAction], Mapping[str, Any]]:
    """Project scalar route phases onto every declared magnetic relation."""

    validate_presentation_action_relations(actions, presentation)
    generator_names = tuple(
        generator.name for generator in presentation.generators
    )
    reference = actions[generator_names[0]]
    fiber_count = len(reference.fiber_dimensions)
    if any(
        dimension != 1
        for action in actions.values()
        for dimension in action.fiber_dimensions
    ):
        raise JointExactificationError(
            "direct phase projection is defined only for U(1) route blocks"
        )

    raw_weights = {} if weights is None else dict(weights)
    unknown_weights = set(raw_weights) - set(generator_names)
    if unknown_weights:
        raise JointExactificationError(
            f"U(1) projection weights reference unknown generators "
            f"{sorted(unknown_weights)}"
        )
    effective_weights: dict[str, float] = {}
    for name in generator_names:
        weight = float(raw_weights.get(name, 1.0))
        if not np.isfinite(weight) or weight <= 0.0:
            raise JointExactificationError(
                f"U(1) projection weight for {name!r} must be finite and positive"
            )
        effective_weights[name] = weight

    variable_indices = {
        (name, source): generator_index * fiber_count + source
        for generator_index, name in enumerate(generator_names)
        for source in range(fiber_count)
    }
    variable_count = len(generator_names) * fiber_count
    angle_by_generator = _u1_angles(actions, generator_names)
    angle_vector = np.concatenate(
        [angle_by_generator[name] for name in generator_names]
    )
    rows: list[np.ndarray] = []
    target_angles: list[float] = []
    relation_names: list[str] = []
    for relation in presentation.relations:
        for source in range(fiber_count):
            lhs = _u1_word_coefficients(
                relation.lhs,
                source,
                actions=actions,
                variable_indices=variable_indices,
                variable_count=variable_count,
            )
            rhs = _u1_word_coefficients(
                relation.rhs,
                source,
                actions=actions,
                variable_indices=variable_indices,
                variable_count=variable_count,
            )
            rows.append(lhs - rhs)
            target_angles.append(_central_phase_angle(relation.central_phase))
            relation_names.append(relation.name)
    constraint_matrix = np.asarray(rows, dtype=np.float64)
    target_vector = np.asarray(target_angles, dtype=np.float64)
    raw_mismatch = constraint_matrix @ angle_vector - target_vector
    principal_residual = np.angle(np.exp(1.0j * raw_mismatch))
    branch_margin = float(np.min(np.pi - np.abs(principal_residual)))
    epsilon = float(np.finfo(np.float64).eps)
    branch_certification_bound = float(
        128.0
        * epsilon
        * max(
            1,
            constraint_matrix.shape[0],
            constraint_matrix.shape[1],
            max(
                (len(relation.lhs) + len(relation.rhs))
                for relation in presentation.relations
            ),
        )
    )
    if branch_margin <= branch_certification_bound:
        raise JointExactificationError(
            "U(1) relation phase is too close to the principal-branch boundary: "
            f"margin={branch_margin:.6e}, certification_bound="
            f"{branch_certification_bound:.6e}"
        )

    square_root_weights = np.concatenate(
        [
            np.full(fiber_count, np.sqrt(effective_weights[name]))
            for name in generator_names
        ]
    )
    weighted_constraints = constraint_matrix / square_root_weights[np.newaxis, :]
    left_vectors, singular_values, right_vectors_h = np.linalg.svd(
        weighted_constraints,
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] == 0.0:
        raise JointExactificationError("U(1) relation constraint matrix has zero rank")
    rank_tolerance = float(
        singular_values[0]
        * max(weighted_constraints.shape)
        * epsilon
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if rank == 0:
        raise JointExactificationError("U(1) relation constraint matrix has zero rank")
    right_vectors = right_vectors_h[:rank, :].T
    weighted_correction = right_vectors @ (
        (left_vectors[:, :rank].T @ (-principal_residual))
        / singular_values[:rank]
    )
    correction = weighted_correction / square_root_weights
    consistency_residual = float(
        np.linalg.norm(constraint_matrix @ correction + principal_residual)
    )
    consistency_scale = max(
        1.0,
        float(np.linalg.norm(principal_residual)),
        float(np.linalg.norm(constraint_matrix, ord=2) * np.linalg.norm(correction)),
    )
    consistency_bound = float(
        128.0
        * epsilon
        * max(constraint_matrix.shape)
        * consistency_scale
    )
    if consistency_residual > consistency_bound:
        raise JointExactificationError(
            "U(1) principal-branch relation constraints are inconsistent: "
            f"residual={consistency_residual:.6e}, bound="
            f"{consistency_bound:.6e}"
        )

    projected: dict[str, BlockRouteAction] = {}
    correction_rms: dict[str, float] = {}
    for generator_index, name in enumerate(generator_names):
        action = actions[name]
        start = generator_index * fiber_count
        stop = start + fiber_count
        generator_correction = correction[start:stop]
        correction_rms[name] = float(
            np.sqrt(np.mean(np.square(generator_correction)))
        )
        route_blocks = tuple(
            np.asarray(
                [[complex(block[0, 0]) * np.exp(1.0j * delta)]],
                dtype=np.complex128,
            )
            for block, delta in zip(action.route_blocks, generator_correction)
        )
        projected[name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=route_blocks,
            fiber_indices=action.fiber_indices,
        )

    projected_angle_by_generator = _u1_angles(projected, generator_names)
    projected_angles = np.concatenate(
        [projected_angle_by_generator[name] for name in generator_names]
    )
    pre_table, pre_maximum = _u1_relation_residual_table(
        constraint_matrix=constraint_matrix,
        angle_vector=angle_vector,
        target_angles=target_vector,
        relation_names=relation_names,
    )
    post_table, post_maximum = _u1_relation_residual_table(
        constraint_matrix=constraint_matrix,
        angle_vector=projected_angles,
        target_angles=target_vector,
        relation_names=relation_names,
    )
    smallest = float(singular_values[rank - 1])
    report: dict[str, Any] = {
        "constraint_shape": tuple(int(value) for value in constraint_matrix.shape),
        "rank": rank,
        "nullity": int(variable_count - rank),
        "rank_tolerance": rank_tolerance,
        "smallest_nonzero_singular_value": smallest,
        "condition_number": float(singular_values[0] / smallest),
        "consistency_residual": consistency_residual,
        "consistency_bound": consistency_bound,
        "branch_margin": branch_margin,
        "branch_certification_bound": branch_certification_bound,
        "weights": dict(effective_weights),
        "correction_rms_by_operation": correction_rms,
        "maximum_phase_correction": float(np.max(np.abs(correction))),
        "pre_relation_residuals": pre_table,
        "pre_relation_residual_max": pre_maximum,
        "post_relation_residuals": post_table,
        "post_relation_residual_max": post_maximum,
    }
    return projected, report


@dataclass(frozen=True)
class ClosestCyclotomicU1GaugeResult:
    """A certified scalar canonical representation and its common gauge."""

    actions: Mapping[str, BlockRouteAction]
    fiber_gauge: tuple[np.ndarray, ...]
    root_order: int
    root_exponents: Mapping[str, tuple[int, ...]]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(
            self,
            "fiber_gauge",
            tuple(_readonly_complex(value) for value in self.fiber_gauge),
        )
        object.__setattr__(self, "root_order", int(self.root_order))
        object.__setattr__(
            self,
            "root_exponents",
            MappingProxyType(
                {
                    str(name): tuple(int(value) for value in values)
                    for name, values in self.root_exponents.items()
                }
            ),
        )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))


@dataclass(frozen=True)
class StandardGeneratorFiberGaugeResult:
    """A certified scalar gauge with canonical unitary generator cycles."""

    actions: Mapping[str, BlockRouteAction]
    fiber_gauge: tuple[np.ndarray, ...]
    gauge_angles: tuple[float, ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(
            self,
            "fiber_gauge",
            tuple(_readonly_complex(value) for value in self.fiber_gauge),
        )
        object.__setattr__(
            self,
            "gauge_angles",
            tuple(float(value) for value in self.gauge_angles),
        )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))


def _cyclotomic_order_from_presentation(
    presentation: MagneticPresentation,
) -> int:
    order = 1
    power_generators: set[str] = set()
    for relation in presentation.relations:
        if relation.rhs or not relation.lhs:
            continue
        generator = relation.lhs[0]
        if any(value != generator for value in relation.lhs):
            continue
        central_order = 1 if relation.central_phase == complex(1.0, 0.0) else 2
        order = math.lcm(order, len(relation.lhs) * central_order)
        power_generators.add(generator)
    expected = {generator.name for generator in presentation.generators}
    if power_generators != expected:
        missing = sorted(expected - power_generators)
        raise JointExactificationError(
            "closest cyclotomic U(1) gauge requires one declared power "
            f"relation for every generator; missing={missing}"
        )
    if order <= 1 or order > 4096:
        raise JointExactificationError(
            f"unsupported cyclotomic root order {order}; expected 2..4096"
        )
    return order


def _reduced_cyclotomic_fraction(order: int, exponent: int) -> tuple[int, int]:
    order = int(order)
    if order <= 0:
        raise JointExactificationError(f"cyclotomic order must be positive, got {order}")
    exponent = int(exponent) % order
    divisor = math.gcd(order, exponent)
    return order // divisor, exponent // divisor


def _cyclotomic_root(order: int, exponent: int) -> complex:
    order, exponent = _reduced_cyclotomic_fraction(order, exponent)
    if 12 % order == 0:
        half_sqrt_three = float(np.sqrt(3.0) / 2.0)
        roots_12 = (
            complex(1.0, 0.0),
            complex(half_sqrt_three, 0.5),
            complex(0.5, half_sqrt_three),
            complex(0.0, 1.0),
            complex(-0.5, half_sqrt_three),
            complex(-half_sqrt_three, 0.5),
            complex(-1.0, 0.0),
            complex(-half_sqrt_three, -0.5),
            complex(-0.5, -half_sqrt_three),
            complex(0.0, -1.0),
            complex(0.5, -half_sqrt_three),
            complex(half_sqrt_three, -0.5),
        )
        return roots_12[(exponent * (12 // order)) % 12]
    angle = 2.0 * np.pi * float(exponent) / float(order)
    value = complex(np.cos(angle), np.sin(angle))
    quarter = 4 * exponent
    if quarter % order == 0:
        axis = (quarter // order) % 4
        return (
            complex(1.0, 0.0),
            complex(0.0, 1.0),
            complex(-1.0, 0.0),
            complex(0.0, -1.0),
        )[axis]
    return value


def _closest_cyclotomic_exponent(
    value: complex,
    *,
    order: int,
) -> tuple[int, float, float]:
    roots = np.asarray(
        [_cyclotomic_root(order, exponent) for exponent in range(order)],
        dtype=np.complex128,
    )
    distances = np.abs(roots - complex(value))
    indices = np.argsort(distances, kind="stable")
    best = int(indices[0])
    best_distance = float(distances[best])
    margin = float(distances[int(indices[1])] - best_distance)
    certification_bound = float(
        256.0 * np.finfo(np.float64).eps * max(1, order)
    )
    if margin <= certification_bound:
        raise JointExactificationError(
            "nearest cyclotomic root is not unique: "
            f"order={order}, margin={margin:.6e}, "
            f"certification_bound={certification_bound:.6e}"
        )
    return best, best_distance, margin


def _gauge_transform_fiber_actions(
    actions: Mapping[str, BlockRouteAction],
    gauge: Sequence[np.ndarray],
) -> dict[str, BlockRouteAction]:
    transformed: dict[str, BlockRouteAction] = {}
    for name, action in actions.items():
        blocks: list[np.ndarray] = []
        propagated_bounds: list[float] = []
        for source, target in enumerate(action.fiber_permutation):
            source_gauge = (
                gauge[source].conjugate() if action.antiunitary else gauge[source]
            )
            block = np.asarray(
                gauge[target].conjugate().T
                @ action.route_blocks[source]
                @ source_gauge,
                dtype=np.complex128,
            )
            blocks.append(block)
            # A product of three already-certified unitary factors carries the
            # input residuals in addition to the final GEMM roundoff.  The
            # single-matrix default bound intentionally does not include that
            # provenance, so propagate it explicitly instead of thresholding
            # the transformed entries.
            propagated_bounds.append(
                float(
                    action.unitarity_certification_bound
                    + _unitarity_residual(np.asarray(gauge[target]))
                    + _unitarity_residual(np.asarray(gauge[source]))
                    + 8.0 * _roundoff_unitarity_bound(block)
                )
            )
        transformed[name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=tuple(blocks),
            fiber_indices=action.fiber_indices,
            unitarity_certification_bound=max(propagated_bounds),
        )
    return transformed


def derive_closest_cyclotomic_u1_gauge(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    reference_actions: Mapping[str, BlockRouteAction] | None = None,
) -> ClosestCyclotomicU1GaugeResult:
    """Choose and certify the closest discrete scalar representation gauge.

    The route roots minimize the separable total Frobenius distance. They are
    accepted only when one common semilinear fiber gauge maps the supplied
    jointly closed representation to that discrete candidate.
    """

    validate_presentation_action_relations(actions, presentation)
    reference_actions = actions if reference_actions is None else reference_actions
    validate_presentation_action_relations(reference_actions, presentation)
    if any(
        dimension != 1
        for action in actions.values()
        for dimension in action.fiber_dimensions
    ):
        raise JointExactificationError(
            "closest cyclotomic gauge is defined only for U(1) route blocks"
        )
    names = tuple(generator.name for generator in presentation.generators)
    reference = actions[names[0]]
    fiber_count = len(reference.fiber_dimensions)
    root_order = _cyclotomic_order_from_presentation(presentation)
    root_exponents: dict[str, tuple[int, ...]] = {}
    nearest_distances: dict[str, tuple[float, ...]] = {}
    branch_margins: dict[str, tuple[float, ...]] = {}
    canonical: dict[str, BlockRouteAction] = {}
    for name in names:
        action = actions[name]
        reference_action = reference_actions[name]
        if (
            action.antiunitary != reference_action.antiunitary
            or action.fiber_permutation != reference_action.fiber_permutation
            or action.fiber_dimensions != reference_action.fiber_dimensions
            or action.fiber_indices != reference_action.fiber_indices
        ):
            raise JointExactificationError(
                f"cyclotomic reference action layout differs for {name!r}"
            )
        selected = tuple(
            _closest_cyclotomic_exponent(
                complex(block[0, 0]),
                order=root_order,
            )
            for block in reference_action.route_blocks
        )
        exponents = tuple(value[0] for value in selected)
        root_exponents[name] = exponents
        nearest_distances[name] = tuple(value[1] for value in selected)
        branch_margins[name] = tuple(value[2] for value in selected)
        canonical[name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=tuple(
                np.asarray(
                    [[_cyclotomic_root(root_order, exponent)]],
                    dtype=np.complex128,
                )
                for exponent in exponents
            ),
            fiber_indices=action.fiber_indices,
        )

    relation_certification = certify_joint_block_actions(canonical, presentation)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for generator in presentation.generators:
        name = generator.name
        action = actions[name]
        for source, target in enumerate(action.fiber_permutation):
            row = np.zeros(fiber_count, dtype=np.float64)
            row[source] += -1.0 if action.antiunitary else 1.0
            row[target] -= 1.0
            rows.append(row)
            ratio = complex(
                canonical[name].route_blocks[source][0, 0]
                / action.route_blocks[source][0, 0]
            )
            targets.append(float(np.angle(ratio)))
    gauge_matrix = np.asarray(rows, dtype=np.float64)
    gauge_target = np.asarray(targets, dtype=np.float64)
    gauge_angles, _residuals, rank, singular_values = np.linalg.lstsq(
        gauge_matrix,
        gauge_target,
        rcond=None,
    )
    linear_residual = gauge_matrix @ gauge_angles - gauge_target
    linear_residual_max = float(np.max(np.abs(linear_residual)))
    linear_bound = float(
        1024.0
        * np.finfo(np.float64).eps
        * max(1, gauge_matrix.shape[0], gauge_matrix.shape[1])
    )
    if linear_residual_max > linear_bound:
        raise JointExactificationError(
            "nearest cyclotomic representation is not connected by one common "
            "fiber gauge: "
            f"residual={linear_residual_max:.6e}, bound={linear_bound:.6e}"
        )
    fiber_gauge = tuple(
        np.asarray([[np.exp(1.0j * angle)]], dtype=np.complex128)
        for angle in gauge_angles
    )
    reframed = _gauge_transform_fiber_actions(actions, fiber_gauge)
    common_gauge_residual_max = max(
        float(np.linalg.norm(actual - expected, ord="fro"))
        for name in names
        for actual, expected in zip(
            reframed[name].route_blocks,
            canonical[name].route_blocks,
        )
    )
    matrix_bound = float(
        2048.0 * np.finfo(np.float64).eps * max(1, fiber_count, len(names))
    )
    if common_gauge_residual_max > matrix_bound:
        raise JointExactificationError(
            "common fiber gauge did not reproduce the selected cyclotomic "
            f"routes: residual={common_gauge_residual_max:.6e}, "
            f"bound={matrix_bound:.6e}"
        )
    distance_values = np.asarray(
        [
            distance
            for name in names
            for distance in nearest_distances[name]
        ],
        dtype=np.float64,
    )
    report: dict[str, Any] = {
        "status": "certified",
        "selection_policy": "nearest_total_frobenius",
        "selection_reference": (
            "joint_closed_routes"
            if reference_actions is actions
            else "stage1_projected_routes"
        ),
        "tie_break_policy": "minimum_norm_common_fiber_gauge",
        "root_order": root_order,
        "root_exponents": {
            name: list(root_exponents[name]) for name in names
        },
        "nearest_distance_rms": float(
            np.sqrt(np.mean(np.square(distance_values)))
        ),
        "nearest_distance_max": float(np.max(distance_values)),
        "minimum_root_branch_margin": float(
            min(min(branch_margins[name]) for name in names)
        ),
        "gauge_angles": [float(value) for value in gauge_angles],
        "gauge_angle_max": float(np.max(np.abs(gauge_angles))),
        "gauge_linear_rank": int(rank),
        "gauge_linear_singular_values": [
            float(value) for value in singular_values
        ],
        "gauge_linear_residual_max": linear_residual_max,
        "gauge_linear_certification_bound": linear_bound,
        "common_gauge_residual_max": common_gauge_residual_max,
        "common_gauge_certification_bound": matrix_bound,
        "relation_certification": {
            **dict(relation_certification),
            "post_relation_residual_max": float(
                relation_certification["relation_residual_max"]
            ),
        },
    }
    return ClosestCyclotomicU1GaugeResult(
        actions=canonical,
        fiber_gauge=fiber_gauge,
        root_order=root_order,
        root_exponents=root_exponents,
        report=report,
    )


def _permutation_cycles(permutation: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    for root in range(len(permutation)):
        if root in seen:
            continue
        cycle: list[int] = []
        current = int(root)
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = int(permutation[current])
        if current != root:
            raise JointExactificationError(
                "fiber permutation traversal did not close at its cycle root"
            )
        cycles.append(tuple(cycle))
    return tuple(cycles)


def _standard_root_label(value: complex) -> str:
    exact = complex(value)
    labels = {
        complex(1.0, 0.0): "+1",
        complex(-1.0, 0.0): "-1",
        complex(0.0, 1.0): "+i",
        complex(0.0, -1.0): "-i",
    }
    return labels.get(exact, f"{exact.real:+.16g}{exact.imag:+.16g}i")


def _identity_standard_generator_result(
    actions: Mapping[str, BlockRouteAction],
    *,
    reason: str,
) -> StandardGeneratorFiberGaugeResult:
    reference = next(iter(actions.values()))
    gauge = tuple(
        np.eye(int(dimension), dtype=np.complex128)
        for dimension in reference.fiber_dimensions
    )
    return StandardGeneratorFiberGaugeResult(
        actions=actions,
        fiber_gauge=gauge,
        gauge_angles=tuple(0.0 for _ in reference.fiber_dimensions),
        report={
            "status": "not_applicable",
            "reason": str(reason),
            "standardized_generators": [],
        },
    )


def derive_standard_generator_u1_gauge(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    reference_actions: Mapping[str, BlockRouteAction] | None = None,
) -> StandardGeneratorFiberGaugeResult:
    """Uniformize unitary ``C2`` cycles with one certified common U(1) gauge.

    Every other generator is held fixed.  The selected uniform phase is the
    cycle-holonomy root with minimum total chordal distance from the supplied
    exact route phases.
    """

    validate_presentation_action_relations(actions, presentation)
    selection_reference = actions if reference_actions is None else reference_actions
    names = tuple(generator.name for generator in presentation.generators)
    if not names:
        raise JointExactificationError(
            "standard generator gauge requires at least one declared generator"
        )
    reference = actions[names[0]]
    if any(
        dimension != 1
        for action in actions.values()
        for dimension in action.fiber_dimensions
    ):
        return _identity_standard_generator_result(
            actions,
            reason="non_scalar_fibers",
        )
    secondary_names = tuple(
        generator.name
        for generator in presentation.generators
        if generator.name == "C2" and not generator.antiunitary
    )
    if not secondary_names:
        return _identity_standard_generator_result(
            actions,
            reason="no_unitary_C2_generator",
        )

    root_order = _cyclotomic_order_from_presentation(presentation)
    target_actions = dict(actions)
    cycle_targets: dict[str, list[str]] = {}
    cycle_reports: dict[str, list[dict[str, Any]]] = {}
    for name in secondary_names:
        action = actions[name]
        reference_action = selection_reference[name]
        if (
            reference_action.antiunitary != action.antiunitary
            or reference_action.fiber_permutation != action.fiber_permutation
            or len(reference_action.route_blocks) != len(action.route_blocks)
        ):
            raise JointExactificationError(
                f"standard generator selection reference layout differs for {name!r}"
            )
        target_blocks = [np.asarray(block) for block in action.route_blocks]
        target_labels: list[str] = []
        reports: list[dict[str, Any]] = []
        for cycle in _permutation_cycles(action.fiber_permutation):
            holonomy = complex(1.0, 0.0)
            for source in cycle:
                value = complex(action.route_blocks[source][0, 0])
                holonomy *= value
            holonomy_exponent, holonomy_distance, _margin = (
                _closest_cyclotomic_exponent(holonomy, order=root_order)
            )
            holonomy_root = _cyclotomic_root(root_order, holonomy_exponent)
            holonomy_bound = float(
                2048.0
                * np.finfo(np.float64).eps
                * max(1, root_order, len(cycle))
            )
            if abs(holonomy - holonomy_root) > holonomy_bound:
                raise JointExactificationError(
                    f"{name} cycle holonomy is not a certified cyclotomic root: "
                    f"cycle={cycle}, distance={holonomy_distance:.6e}, "
                    f"bound={holonomy_bound:.6e}"
                )
            cycle_order = int(root_order * len(cycle))
            candidates = tuple(
                _cyclotomic_root(
                    cycle_order,
                    holonomy_exponent + branch * root_order,
                )
                for branch in range(len(cycle))
            )
            costs = np.asarray(
                [
                    sum(
                        float(
                            np.linalg.norm(
                                candidate
                                * np.eye(
                                    reference_action.route_blocks[source].shape[0],
                                    dtype=np.complex128,
                                )
                                - reference_action.route_blocks[source],
                                ord="fro",
                            )
                            ** 2
                        )
                        for source in cycle
                    )
                    for candidate in candidates
                ],
                dtype=np.float64,
            )
            order = np.argsort(costs, kind="stable")
            best_index = int(order[0])
            if len(order) > 1:
                cost_margin = float(costs[int(order[1])] - costs[best_index])
                cost_bound = float(
                    4096.0
                    * np.finfo(np.float64).eps
                    * max(1, len(cycle), root_order)
                )
                if cost_margin <= cost_bound:
                    raise JointExactificationError(
                        f"{name} standard cycle root is not uniquely nearest: "
                        f"cycle={cycle}, margin={cost_margin:.6e}, "
                        f"bound={cost_bound:.6e}"
                    )
            selected = candidates[best_index]
            for source in cycle:
                target_blocks[source] = np.asarray(
                    [[selected]],
                    dtype=np.complex128,
                )
            target_labels.append(_standard_root_label(selected))
            reports.append(
                {
                    "fibers": [int(value) for value in cycle],
                    "length": int(len(cycle)),
                    "holonomy_root_order": int(root_order),
                    "holonomy_root_exponent": int(holonomy_exponent),
                    "uniform_root_order": int(cycle_order),
                    "uniform_root_exponent": int(
                        holonomy_exponent + best_index * root_order
                    ),
                    "uniform_root": _standard_root_label(selected),
                    "distance_squared": float(costs[best_index]),
                }
            )
        target_actions[name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=tuple(target_blocks),
            fiber_indices=action.fiber_indices,
        )
        cycle_targets[name] = target_labels
        cycle_reports[name] = reports

    relation_certification = certify_joint_block_actions(
        target_actions,
        presentation,
    )
    fiber_count = len(reference.fiber_dimensions)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for generator in presentation.generators:
        name = generator.name
        action = actions[name]
        target_action = target_actions[name]
        for source, target in enumerate(action.fiber_permutation):
            row = np.zeros(fiber_count, dtype=np.float64)
            row[source] += -1.0 if action.antiunitary else 1.0
            row[target] -= 1.0
            rows.append(row)
            ratio = complex(
                target_action.route_blocks[source][0, 0]
                / action.route_blocks[source][0, 0]
            )
            targets.append(float(np.angle(ratio)))
    gauge_matrix = np.asarray(rows, dtype=np.float64)
    gauge_target = np.asarray(targets, dtype=np.float64)
    gauge_angles, _residuals, rank, singular_values = np.linalg.lstsq(
        gauge_matrix,
        gauge_target,
        rcond=None,
    )
    linear_residual = gauge_matrix @ gauge_angles - gauge_target
    linear_residual_max = float(np.max(np.abs(linear_residual)))
    linear_bound = float(
        2048.0
        * np.finfo(np.float64).eps
        * max(1, gauge_matrix.shape[0], gauge_matrix.shape[1])
    )
    if linear_residual_max > linear_bound:
        raise JointExactificationError(
            "standard generator targets are not connected by one common "
            "semilinear fiber gauge: "
            f"residual={linear_residual_max:.6e}, bound={linear_bound:.6e}"
        )
    fiber_gauge = tuple(
        np.asarray([[np.exp(1.0j * angle)]], dtype=np.complex128)
        for angle in gauge_angles
    )
    reframed = _gauge_transform_fiber_actions(actions, fiber_gauge)
    common_gauge_residual_max = max(
        float(np.linalg.norm(actual - expected, ord="fro"))
        for name in names
        for actual, expected in zip(
            reframed[name].route_blocks,
            target_actions[name].route_blocks,
        )
    )
    matrix_bound = float(
        4096.0 * np.finfo(np.float64).eps * max(1, fiber_count, len(names))
    )
    if common_gauge_residual_max > matrix_bound:
        raise JointExactificationError(
            "common fiber gauge did not reproduce the standard generator "
            f"targets: residual={common_gauge_residual_max:.6e}, "
            f"bound={matrix_bound:.6e}"
        )
    return StandardGeneratorFiberGaugeResult(
        actions=target_actions,
        fiber_gauge=fiber_gauge,
        gauge_angles=tuple(float(value) for value in gauge_angles),
        report={
            "status": "certified",
            "selection_policy": "nearest_uniform_cycle_root",
            "selection_reference": (
                "input_scalar_routes"
                if reference_actions is None
                else "supplied_route_blocks"
            ),
            "tie_break_policy": "minimum_norm_common_semilinear_fiber_gauge",
            "preserved_generators": [
                name for name in names if name not in secondary_names
            ],
            "standardized_generators": list(secondary_names),
            "cycle_targets": cycle_targets,
            "cycles": cycle_reports,
            "gauge_angles": [float(value) for value in gauge_angles],
            "gauge_angle_max": float(np.max(np.abs(gauge_angles))),
            "gauge_linear_rank": int(rank),
            "gauge_linear_singular_values": [
                float(value) for value in singular_values
            ],
            "gauge_linear_residual_max": linear_residual_max,
            "gauge_linear_certification_bound": linear_bound,
            "common_gauge_residual_max": common_gauge_residual_max,
            "common_gauge_certification_bound": matrix_bound,
            "relation_certification": dict(relation_certification),
        },
    )


def _transporter_antiunitary_parity(
    word: Sequence[str],
    presentation: MagneticPresentation,
) -> bool:
    parity_by_name = {
        generator.name: bool(generator.antiunitary)
        for generator in presentation.generators
    }
    parity = False
    for name in word:
        parity ^= parity_by_name[name]
    return parity


def _canonical_stabilizer_root_frame(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose a deterministic algebraic frame for one stabilized root fiber."""

    from kp.basis.symmetry_gauge import (
        SymmetryGaugeOperation,
        derive_symmetry_adapted_internal_frame,
    )

    dimension = int(actions[presentation.generators[0].name].fiber_dimensions[orbit.root])
    operations: list[SymmetryGaugeOperation] = []
    evaluated: list[tuple[tuple[str, ...], np.ndarray, bool]] = []
    for word in orbit.stabilizer_words:
        if not word:
            continue
        block, target, antiunitary = _evaluate_route_word(actions, word, orbit.root)
        if target != orbit.root:
            raise JointExactificationError(
                f"declared stabilizer word {word} leaves root {orbit.root}"
            )
        matrix = np.asarray(block, dtype=np.complex128)
        evaluated.append((word, matrix, antiunitary))
        operations.append(
            SymmetryGaugeOperation(
                name="*".join(word),
                matrix=matrix,
                antiunitary=antiunitary,
                power=12,
                can_resolve=not antiunitary,
                can_pair=antiunitary,
                can_anchor=True,
            )
        )
    adapted = derive_symmetry_adapted_internal_frame(
        operations,
        tolerance=1.0e-10,
    )
    if adapted.status == "applied":
        return np.asarray(adapted.unitary, dtype=np.complex128), {
            "status": "finite_unitary_adapted",
            "primary_operation": adapted.primary_operation,
            "pairing_operation": adapted.pairing_operation,
        }

    identity = np.eye(dimension, dtype=np.complex128)
    for word, matrix, antiunitary in evaluated:
        if not antiunitary:
            continue
        square_residual = float(
            np.linalg.norm(matrix @ matrix.conjugate() + identity, ord="fro")
        )
        bound = float(
            4096.0 * np.finfo(np.float64).eps * max(1, dimension)
        )
        if square_residual > bound or dimension % 2:
            continue
        columns: list[np.ndarray] = []
        for seed_index in range(dimension):
            candidate = identity[:, seed_index].copy()
            for previous in columns:
                candidate -= previous * np.vdot(previous, candidate)
            norm = float(np.linalg.norm(candidate))
            if norm <= bound:
                continue
            candidate /= norm
            pivot = int(np.argmax(np.abs(candidate)))
            amplitude = candidate[pivot]
            if abs(amplitude) > bound:
                candidate *= np.conjugate(amplitude / abs(amplitude))
            partner = matrix @ candidate.conjugate()
            for previous in [*columns, candidate]:
                partner -= previous * np.vdot(previous, partner)
            partner_norm = float(np.linalg.norm(partner))
            if partner_norm <= bound:
                continue
            partner /= partner_norm
            columns.extend((candidate, partner))
            if len(columns) == dimension:
                break
        if len(columns) == dimension:
            frame = np.column_stack(columns)
            canonical = frame.conjugate().T @ matrix @ frame.conjugate()
            target = np.zeros_like(canonical)
            for pair in range(0, dimension, 2):
                target[pair, pair + 1] = complex(-1.0, 0.0)
                target[pair + 1, pair] = complex(1.0, 0.0)
            residual = float(np.linalg.norm(canonical - target, ord="fro"))
            if residual <= 16.0 * bound:
                return np.asarray(frame, dtype=np.complex128), {
                    "status": "antiunitary_kramers_adapted",
                    "pairing_operation": "*".join(word),
                    "canonicalization_residual": residual,
                }
    return identity, {
        "status": "identity",
        "reason": "stabilizer_has_no_resolving_unitary_or_kramers_pairing",
    }


def _involutive_permutations(dimension: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate deterministic involutions without factorial brute force."""

    def build(remaining: tuple[int, ...]) -> list[dict[int, int]]:
        if not remaining:
            return [{}]
        first = remaining[0]
        output: list[dict[int, int]] = []
        for suffix in build(remaining[1:]):
            output.append({first: first, **suffix})
        for position, partner in enumerate(remaining[1:], start=1):
            rest = remaining[1:position] + remaining[position + 1 :]
            for suffix in build(rest):
                output.append({first: partner, partner: first, **suffix})
        return output

    return tuple(
        tuple(mapping[index] for index in range(dimension))
        for mapping in build(tuple(range(dimension)))
    )


def _involution_edge_costs(
    reference_blocks: Sequence[np.ndarray],
    phase: complex,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    dimension = int(reference_blocks[0].shape[0])
    edges = tuple(
        (left, right)
        for left in range(dimension)
        for right in range(left, dimension)
    )
    costs: list[float] = []
    for left, right in edges:
        columns = ((left, left),) if left == right else ((left, right), (right, left))
        cost = 0.0
        for block in reference_blocks:
            for source, target in columns:
                column = np.asarray(block[:, source], dtype=np.complex128)
                residual = column.copy()
                residual[target] -= phase
                cost += float(np.vdot(residual, residual).real)
        costs.append(cost)
    return edges, np.asarray(costs, dtype=np.float64)


def _milp_involution(
    reference_blocks: Sequence[np.ndarray],
    phase: complex,
    *,
    cost_bound: float | None = None,
    roundoff_bound: float = 0.0,
    allowed_edges: set[tuple[int, int]] | None = None,
    required_transposition_counts: Sequence[
        tuple[set[int], int]
    ] = (),
) -> tuple[tuple[int, ...], float]:
    """Solve nearest involution as a matching, with an algebraic lexicographic tie break."""

    dimension = int(reference_blocks[0].shape[0])
    edges, edge_costs = _involution_edge_costs(reference_blocks, phase)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    incidence = np.zeros((dimension, len(edges)), dtype=np.float64)
    for index, (left, right) in enumerate(edges):
        incidence[left, index] = 1.0
        if right != left:
            incidence[right, index] = 1.0
    constraint_rows = [incidence]
    constraint_lower = [np.ones(dimension, dtype=np.float64)]
    constraint_upper = [np.ones(dimension, dtype=np.float64)]
    for indices, required in required_transposition_counts:
        row = np.asarray(
            [
                1.0
                if left != right and left in indices and right in indices
                else 0.0
                for left, right in edges
            ],
            dtype=np.float64,
        )[None, :]
        constraint_rows.append(row)
        constraint_lower.append(np.asarray([float(required)]))
        constraint_upper.append(np.asarray([float(required)]))
    degree_constraint = LinearConstraint(
        np.vstack(constraint_rows),
        np.concatenate(constraint_lower),
        np.concatenate(constraint_upper),
    )
    integrality = np.ones(len(edges), dtype=np.int32)
    lower_bounds = np.zeros(len(edges), dtype=np.float64)
    upper_bounds = np.asarray(
        [
            1.0
            if allowed_edges is None or edge in allowed_edges
            else 0.0
            for edge in edges
        ],
        dtype=np.float64,
    )
    unit_bounds = Bounds(lower_bounds, upper_bounds)
    primary = milp(
        edge_costs,
        integrality=integrality,
        bounds=unit_bounds,
        constraints=degree_constraint,
        options={"presolve": True},
    )
    if not primary.success or primary.fun is None:
        raise JointExactificationError(
            f"nearest involutive monomial matching failed: status={primary.message}"
        )

    def decode(solution: np.ndarray) -> tuple[tuple[int, ...], float]:
        permutation = list(range(dimension))
        assigned: set[int] = set()
        for index in np.flatnonzero(np.asarray(solution) > 0.5):
            left, right = edges[int(index)]
            if left in assigned or right in assigned:
                raise JointExactificationError(
                    "MILP involution solution assigns a fiber more than once"
                )
            permutation[left] = right
            permutation[right] = left
            assigned.update((left, right))
        if assigned != set(range(dimension)):
            raise JointExactificationError(
                "MILP involution solution does not assign every fiber"
            )
        matrix = phase * _permutation_matrix(permutation)
        cost = sum(
            float(np.linalg.norm(matrix - block, ord="fro") ** 2)
            for block in reference_blocks
        )
        return tuple(permutation), cost

    if primary.x is None:
        raise JointExactificationError("nearest involutive monomial matching returned no solution")
    primary_permutation, primary_cost = decode(primary.x)
    if cost_bound is None:
        return primary_permutation, primary_cost

    maximum_cost = float(cost_bound + roundoff_bound)
    lower = lower_bounds.copy()
    upper = upper_bounds.copy()
    selected_edges: list[tuple[int, int]] = []
    assigned: set[int] = set()
    for source in range(dimension):
        if source in assigned:
            continue
        selected: tuple[int, int] | None = None
        for target in range(source, dimension):
            if target in assigned:
                continue
            edge = (source, target)
            if allowed_edges is not None and edge not in allowed_edges:
                continue
            trial_lower = lower.copy()
            trial_upper = upper.copy()
            index = edge_index[edge]
            trial_lower[index] = 1.0
            trial_upper[index] = 1.0
            feasible = milp(
                edge_costs,
                integrality=integrality,
                bounds=Bounds(trial_lower, trial_upper),
                constraints=degree_constraint,
                options={"presolve": True},
            )
            if not feasible.success or feasible.x is None:
                continue
            _trial_permutation, trial_cost = decode(feasible.x)
            if trial_cost <= maximum_cost:
                selected = edge
                lower = trial_lower
                upper = trial_upper
                break
        if selected is None:
            raise JointExactificationError(
                "nearest involutive monomial matching has no deterministic tie-break completion"
            )
        selected_edges.append(selected)
        assigned.update(selected)
    permutation = list(range(dimension))
    for left, right in selected_edges:
        permutation[left] = right
        permutation[right] = left
    matrix = phase * _permutation_matrix(permutation)
    cost = sum(
        float(np.linalg.norm(matrix - block, ord="fro") ** 2)
        for block in reference_blocks
    )
    if cost > maximum_cost:
        raise JointExactificationError(
            "lexicographic involution is outside the certified nearest set: "
            f"cost={cost:.16e}, maximum={maximum_cost:.16e}"
        )
    return tuple(permutation), cost


def _nearest_involutive_monomial(
    reference_blocks: Sequence[np.ndarray],
    phases: Sequence[tuple[int, complex]],
    *,
    roundoff_bound: float,
) -> tuple[tuple[int, ...], int, complex, float, dict[str, Any]]:
    dimension = int(reference_blocks[0].shape[0])
    rows: list[tuple[float, tuple[int, ...], int, complex]] = []
    if dimension <= 12:
        for permutation in _involutive_permutations(dimension):
            matrix = _permutation_matrix(permutation)
            for phase_exponent, phase in phases:
                candidate = phase * matrix
                cost = sum(
                    float(np.linalg.norm(candidate - block, ord="fro") ** 2)
                    for block in reference_blocks
                )
                rows.append((cost, permutation, phase_exponent, phase))
        best_cost = min(row[0] for row in rows)
        nearest = [row for row in rows if row[0] <= best_cost + roundoff_bound]
        selected = min(nearest, key=lambda row: (row[1], row[2]))
        solver = "enumerated_involutions"
        nearest_count: int | None = len(nearest)
    else:
        phase_optima: list[tuple[float, int, complex]] = []
        for phase_exponent, phase in phases:
            _permutation, cost = _milp_involution(reference_blocks, phase)
            phase_optima.append((cost, phase_exponent, phase))
        best_cost = min(row[0] for row in phase_optima)
        rows = []
        for cost, phase_exponent, phase in phase_optima:
            if cost > best_cost + roundoff_bound:
                continue
            permutation, lex_cost = _milp_involution(
                reference_blocks,
                phase,
                cost_bound=best_cost,
                roundoff_bound=roundoff_bound,
            )
            rows.append((lex_cost, permutation, phase_exponent, phase))
        selected = min(rows, key=lambda row: (row[1], row[2]))
        solver = "milp_lexicographic"
        nearest_count = None
    cost, permutation, phase_exponent, phase = selected
    return permutation, phase_exponent, phase, cost, {
        "matching_solver": solver,
        "nearest_solution_count": nearest_count,
        "nearest_cost_bound": float(roundoff_bound),
        "tie_break": "algebraic_lexicographic",
    }


def _permutation_matrix(permutation: Sequence[int]) -> np.ndarray:
    dimension = len(permutation)
    value = np.zeros((dimension, dimension), dtype=np.complex128)
    for source, target in enumerate(permutation):
        value[int(target), int(source)] = complex(1.0, 0.0)
    return value


def _standardize_ud_monomial_actions(
    central_actions: Mapping[str, BlockRouteAction],
    reference_actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> StandardGeneratorFiberGaugeResult:
    """Gauge exact U(d) routes to nearest algebraic monomial C2 blocks."""

    c2_names = tuple(
        generator.name
        for generator in presentation.generators
        if generator.name == "C2" and not generator.antiunitary
    )
    if len(c2_names) != 1:
        raise JointExactificationError(
            "standard free-orbit U(d) gauge requires exactly one unitary C2"
        )
    name = c2_names[0]
    action = central_actions[name]
    reference = reference_actions[name]
    root_order = _cyclotomic_order_from_presentation(presentation)
    fiber_count = len(action.fiber_dimensions)
    fiber_gauge: list[np.ndarray | None] = [None] * fiber_count
    labels: list[str] = []
    reports: list[dict[str, Any]] = []
    maximum_cycle_length = 1
    for cycle in _permutation_cycles(action.fiber_permutation):
        maximum_cycle_length = max(maximum_cycle_length, len(cycle))
        dimensions = {int(action.fiber_dimensions[source]) for source in cycle}
        if len(dimensions) != 1:
            raise JointExactificationError(
                f"C2 cycle {cycle} changes U(d) block dimension"
            )
        dimension = dimensions.pop()
        identity = np.eye(dimension, dtype=np.complex128)
        matching_report: dict[str, Any] = {
            "matching_solver": "fixed_cycle",
            "nearest_solution_count": 1,
            "nearest_cost_bound": 0.0,
            "tie_break": "not_applicable",
        }
        if len(cycle) == 1:
            selected = np.asarray(action.route_blocks[cycle[0]])
            fiber_gauge[cycle[0]] = identity
            label = "fixed_central_route"
            distance_squared = float(
                np.linalg.norm(
                    selected - reference.route_blocks[cycle[0]],
                    ord="fro",
                )
                ** 2
            )
            selected_permutation = tuple(range(dimension))
            selected_phase = complex(selected[0, 0])
        elif len(cycle) == 2:
            holonomy_matrix = np.asarray(
                action.route_blocks[cycle[1]] @ action.route_blocks[cycle[0]],
                dtype=np.complex128,
            )
            root_distances = [
                float(
                    np.linalg.norm(
                        holonomy_matrix
                        - _cyclotomic_root(root_order, candidate) * identity,
                        ord="fro",
                    )
                )
                for candidate in range(root_order)
            ]
            exponent = int(np.argmin(root_distances))
            distance = float(root_distances[exponent])
            root = _cyclotomic_root(root_order, exponent)
            holonomy_bound = float(
                2048.0
                * np.finfo(np.float64).eps
                * max(1, root_order, dimension)
            )
            if distance > holonomy_bound:
                raise JointExactificationError(
                    "free U(d) C2 holonomy is not a certified root: "
                    f"cycle={cycle}, distance={distance:.6e}"
                )
            cycle_order = int(root_order * len(cycle))
            phases = tuple(
                (
                    exponent + branch * root_order,
                    _cyclotomic_root(
                        cycle_order,
                        exponent + branch * root_order,
                    ),
                )
                for branch in range(len(cycle))
            )
            uniqueness_bound = float(
                4096.0
                * np.finfo(np.float64).eps
                * max(1, root_order, dimension, len(cycle))
            )
            (
                selected_permutation,
                _selected_phase_exponent,
                selected_phase,
                distance_squared,
                matching_report,
            ) = _nearest_involutive_monomial(
                tuple(reference.route_blocks[source] for source in cycle),
                phases,
                roundoff_bound=uniqueness_bound,
            )
            selected = np.asarray(
                selected_phase * _permutation_matrix(selected_permutation),
                dtype=np.complex128,
            )
            source = cycle[0]
            target = cycle[1]
            fiber_gauge[source] = identity
            fiber_gauge[target] = np.asarray(
                action.route_blocks[source] @ selected.conjugate().T,
                dtype=np.complex128,
            )
            label = (
                f"{_standard_root_label(selected_phase)}*"
                + ("I" if selected_permutation == tuple(range(dimension)) else f"P{selected_permutation}")
            )
        else:
            raise JointExactificationError(
                f"unitary C2 has unsupported fiber cycle length {len(cycle)}"
            )
        labels.append(label)
        reports.append(
            {
                "fibers": [int(value) for value in cycle],
                "length": int(len(cycle)),
                "uniform_block": label,
                "uniform_phase": _standard_root_label(selected_phase),
                "internal_permutation": [int(value) for value in selected_permutation],
                "distance_squared": float(distance_squared),
                **matching_report,
            }
        )
    if any(value is None for value in fiber_gauge):
        raise JointExactificationError(
            "standard C2 cycle construction did not assign every U(d) fiber"
        )
    preliminary_gauge = tuple(
        np.asarray(value, dtype=np.complex128)
        for value in fiber_gauge
        if value is not None
    )
    transformed = _gauge_transform_fiber_actions(
        central_actions,
        preliminary_gauge,
    )
    snap_order = int(root_order * maximum_cycle_length)
    structural_bound = float(
        8192.0
        * np.finfo(np.float64).eps
        * max(1, fiber_count, snap_order, max(action.fiber_dimensions))
    )
    snapped: dict[str, BlockRouteAction] = {}
    algebraic_route_encoding: dict[str, list[dict[str, Any]]] = {}
    snap_distance_max = 0.0
    for generator_name, transformed_action in transformed.items():
        blocks: list[np.ndarray] = []
        encoded_blocks: list[dict[str, Any]] = []
        for block in transformed_action.route_blocks:
            dimension = block.shape[0]
            targets = tuple(int(np.argmax(np.abs(block[:, source]))) for source in range(dimension))
            if sorted(targets) != list(range(dimension)):
                raise JointExactificationError(
                    f"{generator_name} transformed U(d) route lacks unique monomial support"
                )
            support = np.zeros_like(block)
            for source, target in enumerate(targets):
                support[target, source] = block[target, source]
            off_support = float(np.linalg.norm(block - support, ord="fro"))
            if off_support > structural_bound:
                raise JointExactificationError(
                    f"{generator_name} transformed U(d) route is not structurally monomial: "
                    f"residual={off_support:.6e}, bound={structural_bound:.6e}"
                )
            exact = np.zeros_like(block)
            root_fractions: list[list[int]] = []
            for source, target in enumerate(targets):
                exponent, distance, _margin = _closest_cyclotomic_exponent(
                    complex(block[target, source]),
                    order=snap_order,
                )
                snap_distance_max = max(snap_distance_max, distance)
                exact[target, source] = _cyclotomic_root(snap_order, exponent)
                reduced_order, reduced_exponent = _reduced_cyclotomic_fraction(
                    snap_order,
                    exponent,
                )
                root_fractions.append([reduced_exponent, reduced_order])
            blocks.append(exact)
            encoded_blocks.append(
                {
                    "support_target_by_source": [int(value) for value in targets],
                    "root_exponent_over_order_by_source": root_fractions,
                }
            )
        snapped[generator_name] = BlockRouteAction(
            name=transformed_action.name,
            antiunitary=transformed_action.antiunitary,
            fiber_permutation=transformed_action.fiber_permutation,
            fiber_dimensions=transformed_action.fiber_dimensions,
            route_blocks=tuple(blocks),
            fiber_indices=transformed_action.fiber_indices,
        )
        algebraic_route_encoding[generator_name] = encoded_blocks
    certification = certify_joint_block_actions(snapped, presentation)
    common_residual = max(
        float(np.linalg.norm(actual - expected, ord="fro"))
        for generator_name in snapped
        for actual, expected in zip(
            transformed[generator_name].route_blocks,
            snapped[generator_name].route_blocks,
        )
    )
    if common_residual > structural_bound:
        raise JointExactificationError(
            "standard U(d) monomial targets are not reproduced by their common gauge: "
            f"residual={common_residual:.6e}, bound={structural_bound:.6e}"
        )
    return StandardGeneratorFiberGaugeResult(
        actions=snapped,
        fiber_gauge=preliminary_gauge,
        gauge_angles=tuple(0.0 for _ in range(fiber_count)),
        report={
            "status": "certified",
            "standardized_generators": [name],
            "cycle_targets": {name: labels},
            "cycles": {name: reports},
            "selection_reference": "supplied_Ud_route_blocks",
            "selection_policy": "nearest_uniform_cyclotomic_monomial_C2_block",
            "cyclotomic_snap_order": snap_order,
            "algebraic_route_encoding": algebraic_route_encoding,
            "cyclotomic_snap_distance_max": snap_distance_max,
            "structural_monomial_certification_bound": structural_bound,
            "common_gauge_residual_max": common_residual,
            "common_gauge_certification_bound": structural_bound,
            "relation_certification": dict(certification),
        },
    )


def _canonicalize_ud_monomial_actions_without_c2(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> StandardGeneratorFiberGaugeResult:
    """Snap a certified orbit-frame representation to algebraic monomial blocks."""

    reference = actions[presentation.generators[0].name]
    fiber_count = len(reference.fiber_dimensions)
    root_order = _cyclotomic_order_from_presentation(presentation)
    structural_bound = float(
        8192.0
        * np.finfo(np.float64).eps
        * max(1, fiber_count, root_order, max(reference.fiber_dimensions))
    )
    snapped: dict[str, BlockRouteAction] = {}
    algebraic_route_encoding: dict[str, list[dict[str, Any]]] = {}
    snap_distance_max = 0.0
    for name, action in actions.items():
        blocks: list[np.ndarray] = []
        encoded_blocks: list[dict[str, Any]] = []
        for block in action.route_blocks:
            dimension = block.shape[0]
            targets = tuple(
                int(np.argmax(np.abs(block[:, source])))
                for source in range(dimension)
            )
            if sorted(targets) != list(range(dimension)):
                raise JointExactificationError(
                    f"{name} orbit-frame U(d) route lacks unique monomial support"
                )
            support = np.zeros_like(block)
            for source, target in enumerate(targets):
                support[target, source] = block[target, source]
            off_support = float(np.linalg.norm(block - support, ord="fro"))
            if off_support > structural_bound:
                raise JointExactificationError(
                    f"{name} orbit-frame U(d) route is not structurally monomial: "
                    f"residual={off_support:.6e}, bound={structural_bound:.6e}"
                )
            exact = np.zeros_like(block)
            root_fractions: list[list[int]] = []
            for source, target in enumerate(targets):
                exponent, distance, _margin = _closest_cyclotomic_exponent(
                    complex(block[target, source]),
                    order=root_order,
                )
                snap_distance_max = max(snap_distance_max, distance)
                exact[target, source] = _cyclotomic_root(root_order, exponent)
                reduced_order, reduced_exponent = _reduced_cyclotomic_fraction(
                    root_order,
                    exponent,
                )
                root_fractions.append([reduced_exponent, reduced_order])
            blocks.append(exact)
            encoded_blocks.append(
                {
                    "support_target_by_source": [int(value) for value in targets],
                    "root_exponent_over_order_by_source": root_fractions,
                }
            )
        snapped[name] = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=tuple(blocks),
            fiber_indices=action.fiber_indices,
        )
        algebraic_route_encoding[name] = encoded_blocks
    certification = certify_joint_block_actions(snapped, presentation)
    common_residual = max(
        float(np.linalg.norm(actual - expected, ord="fro"))
        for name in snapped
        for actual, expected in zip(
            actions[name].route_blocks,
            snapped[name].route_blocks,
        )
    )
    if common_residual > structural_bound:
        raise JointExactificationError(
            "orbit-frame U(d) actions do not reproduce their algebraic monomial targets: "
            f"residual={common_residual:.6e}, bound={structural_bound:.6e}"
        )
    gauge = tuple(
        np.eye(int(dimension), dtype=np.complex128)
        for dimension in reference.fiber_dimensions
    )
    return StandardGeneratorFiberGaugeResult(
        actions=snapped,
        fiber_gauge=gauge,
        gauge_angles=tuple(0.0 for _ in range(fiber_count)),
        report={
            "status": "certified",
            "standardized_generators": [],
            "cycle_targets": {},
            "selection_policy": "certified_algebraic_monomial_orbit_frame",
            "cyclotomic_snap_order": root_order,
            "algebraic_route_encoding": algebraic_route_encoding,
            "cyclotomic_snap_distance_max": snap_distance_max,
            "structural_monomial_certification_bound": structural_bound,
            "common_gauge_residual_max": common_residual,
            "common_gauge_certification_bound": structural_bound,
            "relation_certification": dict(certification),
        },
    )


def _canonical_projector_basis(
    projector: np.ndarray,
    rank: int,
    *,
    bound: float,
) -> np.ndarray:
    """Build a deterministic basis from projected coordinate vectors."""

    dimension = int(projector.shape[0])
    columns: list[np.ndarray] = []
    hermitian_projector = 0.5 * (projector + projector.conjugate().T)
    for coordinate in range(dimension):
        candidate = np.asarray(
            hermitian_projector[:, coordinate],
            dtype=np.complex128,
        ).copy()
        for previous in columns:
            candidate -= previous * np.vdot(previous, candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= bound:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        amplitude = candidate[pivot]
        if abs(amplitude) > bound:
            candidate *= np.conjugate(amplitude / abs(amplitude))
        columns.append(candidate)
        if len(columns) == rank:
            break
    if len(columns) != rank:
        raise JointExactificationError(
            "rank-deficient Procrustes projector lacks a certified coordinate basis: "
            f"expected={rank}, found={len(columns)}"
        )
    return np.column_stack(columns) if columns else np.zeros((dimension, 0), dtype=np.complex128)


def _deterministic_procrustes_unitary(
    coefficient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the polar Procrustes factor with a deterministic null-space extension."""

    dimension = int(coefficient.shape[0])
    left, singular_values, right_h = np.linalg.svd(
        coefficient,
        full_matrices=True,
    )
    scale = max(1.0, float(singular_values[0]) if singular_values.size else 0.0)
    rank_bound = float(
        4096.0 * np.finfo(np.float64).eps * max(1, dimension) * scale
    )
    rank = int(np.count_nonzero(singular_values > rank_bound))
    right = right_h.conjugate().T
    root_unitary = np.asarray(
        right[:, :rank] @ left[:, :rank].conjugate().T,
        dtype=np.complex128,
    )
    nullity = dimension - rank
    tie_break = "not_applicable"
    if nullity:
        identity = np.eye(dimension, dtype=np.complex128)
        left_projector = identity - left[:, :rank] @ left[:, :rank].conjugate().T
        right_projector = identity - right[:, :rank] @ right[:, :rank].conjugate().T
        left_null = _canonical_projector_basis(
            left_projector,
            nullity,
            bound=rank_bound,
        )
        right_null = _canonical_projector_basis(
            right_projector,
            nullity,
            bound=rank_bound,
        )
        root_unitary += right_null @ left_null.conjugate().T
        tie_break = "projected_coordinate_basis"
    unitarity = _unitarity_residual(root_unitary)
    unitarity_bound = _roundoff_unitarity_bound(root_unitary)
    if unitarity > 16.0 * unitarity_bound:
        raise JointExactificationError(
            "deterministic Procrustes completion is not unitary: "
            f"residual={unitarity:.6e}, bound={16.0 * unitarity_bound:.6e}"
        )
    return root_unitary, singular_values, {
        "rank": rank,
        "nullity": nullity,
        "rank_certification_bound": rank_bound,
        "tie_break": tie_break,
    }


def _declared_generator_projective_power(
    presentation: MagneticPresentation,
    name: str,
) -> int | None:
    for relation in presentation.relations:
        if relation.rhs or not relation.lhs:
            continue
        if any(value != name for value in relation.lhs):
            continue
        central_order = 1 if relation.central_phase == complex(1.0, 0.0) else 2
        return int(len(relation.lhs) * central_order)
    return None


def _uniform_route_block(
    action: BlockRouteAction,
    *,
    root_order: int,
) -> tuple[np.ndarray, float, float] | None:
    dimensions = {int(value) for value in action.fiber_dimensions}
    if len(dimensions) != 1 or not action.route_blocks:
        return None
    dimension = dimensions.pop()
    reference = np.asarray(action.route_blocks[0], dtype=np.complex128)
    residual = max(
        float(np.linalg.norm(np.asarray(block) - reference, ord="fro"))
        for block in action.route_blocks
    )
    bound = float(
        8192.0
        * np.finfo(np.float64).eps
        * max(1, len(action.route_blocks), dimension, root_order)
    )
    if residual > bound:
        return None
    return reference, residual, bound


def _closest_algebraic_uniform_stabilizer(
    frame: np.ndarray,
    diagonal_action: np.ndarray,
    *,
    root_order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fix a common eigenframe by the nearest cyclotomic monomial stabilizer."""

    dimension = int(frame.shape[0])
    diagonal = np.diag(diagonal_action)
    exponents = tuple(
        _closest_cyclotomic_exponent(complex(value), order=root_order)[0]
        for value in diagonal
    )
    groups: dict[int, list[int]] = {}
    for index, exponent in enumerate(exponents):
        groups.setdefault(int(exponent), []).append(index)
    roots = tuple(_cyclotomic_root(root_order, value) for value in range(root_order))
    stabilizer = np.zeros((dimension, dimension), dtype=np.complex128)
    assignments: list[dict[str, Any]] = []
    objective_bound = float(
        4096.0
        * np.finfo(np.float64).eps
        * max(1, dimension, root_order)
    )
    for eigen_exponent in sorted(groups):
        indices = groups[eigen_exponent]
        size = len(indices)
        weights = np.empty((size, size), dtype=np.float64)
        phase_exponents = np.empty((size, size), dtype=np.int64)
        for row, target_index in enumerate(indices):
            for column, source_index in enumerate(indices):
                values = np.asarray(
                    [
                        float(
                            np.real(
                                frame[target_index, source_index] * root
                            )
                        )
                        for root in roots
                    ],
                    dtype=np.float64,
                )
                maximum = float(np.max(values))
                tied = np.flatnonzero(values >= maximum - objective_bound)
                exponent = int(tied[0])
                weights[row, column] = float(values[exponent])
                phase_exponents[row, column] = exponent
        rows, columns = linear_sum_assignment(weights, maximize=True)
        optimum = float(np.sum(weights[rows, columns]))
        assigned_columns: set[int] = set()
        selected: list[tuple[int, int]] = []
        accumulated = 0.0
        for row in range(size):
            chosen: int | None = None
            for column in range(size):
                if column in assigned_columns:
                    continue
                remaining_rows = list(range(row + 1, size))
                remaining_columns = [
                    value
                    for value in range(size)
                    if value not in assigned_columns and value != column
                ]
                remaining = 0.0
                if remaining_rows:
                    sub_rows, sub_columns = linear_sum_assignment(
                        weights[np.ix_(remaining_rows, remaining_columns)],
                        maximize=True,
                    )
                    remaining = float(
                        np.sum(
                            weights[
                                np.asarray(remaining_rows)[sub_rows],
                                np.asarray(remaining_columns)[sub_columns],
                            ]
                        )
                    )
                total = accumulated + float(weights[row, column]) + remaining
                if total >= optimum - objective_bound:
                    chosen = column
                    break
            if chosen is None:
                raise JointExactificationError(
                    "common eigenframe stabilizer has no certified lexicographic assignment"
                )
            selected.append((row, chosen))
            assigned_columns.add(chosen)
            accumulated += float(weights[row, chosen])
        for row, column in selected:
            target_index = indices[row]
            source_index = indices[column]
            phase_exponent = int(phase_exponents[row, column])
            stabilizer[source_index, target_index] = roots[phase_exponent]
            assignments.append(
                {
                    "target_index": int(target_index),
                    "source_index": int(source_index),
                    "phase_exponent": phase_exponent,
                    "phase_order": int(root_order),
                }
            )
    identity = np.eye(dimension, dtype=np.complex128)
    unitarity = float(
        np.linalg.norm(stabilizer.conjugate().T @ stabilizer - identity, ord="fro")
    )
    preservation_bound = float(
        4096.0
        * np.finfo(np.float64).eps
        * max(1, dimension, root_order)
    )
    if unitarity > preservation_bound:
        raise JointExactificationError(
            "common eigenframe algebraic stabilizer is not certified unitary: "
            f"residual={unitarity:.6e}, bound={preservation_bound:.6e}"
        )
    preservation = float(
        np.linalg.norm(
            stabilizer.conjugate().T @ diagonal_action @ stabilizer
            - diagonal_action,
            ord="fro",
        )
    )
    if preservation > preservation_bound:
        raise JointExactificationError(
            "closest algebraic stabilizer does not preserve the uniform generator: "
            f"residual={preservation:.6e}, bound={preservation_bound:.6e}"
        )
    result = np.asarray(frame @ stabilizer, dtype=np.complex128)
    objective_before = float(np.linalg.norm(frame - identity, ord="fro") ** 2)
    objective_after = float(np.linalg.norm(result - identity, ord="fro") ** 2)
    if objective_after > objective_before + preservation_bound:
        raise JointExactificationError(
            "closest algebraic stabilizer increased common-frame distance: "
            f"before={objective_before:.16e}, after={objective_after:.16e}"
        )
    return result, {
        "status": "certified",
        "root_order": int(root_order),
        "eigenvalue_exponents": [int(value) for value in exponents],
        "assignments": assignments,
        "objective_before": objective_before,
        "objective_after": objective_after,
        "uniform_generator_preservation_residual": preservation,
        "uniform_generator_preservation_bound": preservation_bound,
        "tie_break": "lexicographic_assignment_then_smallest_root_exponent",
    }


def _diagonal_cyclotomic_groups(
    diagonal_action: np.ndarray,
    *,
    root_order: int,
) -> tuple[tuple[int, ...], ...]:
    groups: dict[int, list[int]] = {}
    for index, value in enumerate(np.diag(diagonal_action)):
        exponent = _closest_cyclotomic_exponent(
            complex(value),
            order=root_order,
        )[0]
        groups.setdefault(int(exponent), []).append(int(index))
    return tuple(tuple(groups[key]) for key in sorted(groups))


def _eigenspace_group_mapping(
    matrix: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    bound: float,
) -> tuple[int, ...]:
    mapping: list[int] = []
    for source_indices in groups:
        weights = np.asarray(
            [
                np.linalg.norm(
                    matrix[np.ix_(target_indices, source_indices)],
                    ord="fro",
                )
                ** 2
                for target_indices in groups
            ],
            dtype=np.float64,
        )
        target = int(np.argmax(weights))
        off_target = float(
            max(0.0, float(np.sum(weights) - weights[target])) ** 0.5
        )
        if off_target > bound:
            raise JointExactificationError(
                "C2 mixes distinct uniform-generator eigenspaces: "
                f"residual={off_target:.6e}, bound={bound:.6e}"
            )
        if len(groups[target]) != len(source_indices):
            raise JointExactificationError(
                "C2 changes a uniform-generator eigenspace dimension"
            )
        mapping.append(target)
    if sorted(mapping) != list(range(len(groups))):
        raise JointExactificationError(
            "C2 uniform-generator eigenspace action is not bijective"
        )
    return tuple(mapping)


def _allowed_involution_edges_for_mapping(
    groups: Sequence[Sequence[int]],
    mapping: Sequence[int],
) -> set[tuple[int, int]]:
    group_by_index = {
        int(index): int(group_index)
        for group_index, indices in enumerate(groups)
        for index in indices
    }
    dimension = sum(len(indices) for indices in groups)
    allowed: set[tuple[int, int]] = set()
    for left in range(dimension):
        for right in range(left, dimension):
            left_group = group_by_index[left]
            right_group = group_by_index[right]
            if (
                int(mapping[left_group]) == right_group
                and int(mapping[right_group]) == left_group
            ):
                allowed.add((left, right))
    return allowed


def _required_fixed_group_transpositions(
    matrix: np.ndarray,
    groups: Sequence[Sequence[int]],
    mapping: Sequence[int],
    *,
    phase: complex,
    bound: float,
) -> tuple[tuple[set[int], int], ...]:
    requirements: list[tuple[set[int], int]] = []
    for group_index, indices in enumerate(groups):
        if int(mapping[group_index]) != group_index:
            continue
        internal = np.asarray(
            matrix[np.ix_(indices, indices)],
            dtype=np.complex128,
        )
        values = np.linalg.eigvals(internal)
        negative_count = 0
        for value in values:
            positive_distance = abs(complex(value) - complex(phase))
            negative_distance = abs(complex(value) + complex(phase))
            if min(positive_distance, negative_distance) > bound:
                raise JointExactificationError(
                    "fixed C2 eigenspace has a non-projective-involution eigenvalue: "
                    f"value={value}, phase={phase}, bound={bound:.6e}"
                )
            if negative_distance < positive_distance:
                negative_count += 1
        requirements.append((set(int(value) for value in indices), negative_count))
    return tuple(requirements)


def _block_commutant_procrustes(
    coefficient: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    result = np.zeros_like(coefficient, dtype=np.complex128)
    for indices in groups:
        block = np.asarray(
            coefficient[np.ix_(indices, indices)],
            dtype=np.complex128,
        )
        unitary, _singular_values, _report = _deterministic_procrustes_unitary(
            block
        )
        result[np.ix_(indices, indices)] = unitary
    return result


def _cyclotomic_eigenspace_bases(
    matrix: np.ndarray,
    *,
    root_order: int,
    bound: float,
) -> dict[int, np.ndarray]:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.complex128))
    columns_by_exponent: dict[int, list[int]] = {}
    for index, value in enumerate(values):
        exponent = _closest_cyclotomic_exponent(
            complex(value),
            order=root_order,
        )[0]
        columns_by_exponent.setdefault(int(exponent), []).append(int(index))
    result: dict[int, np.ndarray] = {}
    for exponent in sorted(columns_by_exponent):
        columns = columns_by_exponent[exponent]
        basis, _ = np.linalg.qr(vectors[:, columns], mode="reduced")
        projector = np.asarray(basis @ basis.conjugate().T, dtype=np.complex128)
        result[exponent] = _canonical_projector_basis(
            projector,
            len(columns),
            bound=bound,
        )
    return result


def _closest_fixed_c2_commutant(
    source_block: np.ndarray,
    target_block: np.ndarray,
    diagonal_action: np.ndarray,
    common_frame: np.ndarray,
    *,
    root_order: int,
    bound: float,
) -> np.ndarray:
    """Solve S^dagger B S=M inside the commutant of diagonal_action."""

    groups = _diagonal_cyclotomic_groups(
        diagonal_action,
        root_order=root_order,
    )
    source_mapping = _eigenspace_group_mapping(
        source_block,
        groups,
        bound=bound,
    )
    target_mapping = _eigenspace_group_mapping(
        target_block,
        groups,
        bound=bound,
    )
    if source_mapping != target_mapping:
        raise JointExactificationError(
            "C2 monomial target has the wrong uniform-generator eigenspace action"
        )
    stabilizer = np.zeros_like(source_block, dtype=np.complex128)
    for group_cycle in _permutation_cycles(source_mapping):
        if len(group_cycle) == 2:
            left_group, right_group = group_cycle
            left = groups[left_group]
            right = groups[right_group]
            source_route = np.asarray(
                source_block[np.ix_(right, left)],
                dtype=np.complex128,
            )
            target_route = np.asarray(
                target_block[np.ix_(right, left)],
                dtype=np.complex128,
            )
            coefficient = np.asarray(
                common_frame[np.ix_(left, left)]
                + target_route.conjugate().T
                @ common_frame[np.ix_(right, right)]
                @ source_route,
                dtype=np.complex128,
            )
            left_stabilizer, _singular_values, _report = (
                _deterministic_procrustes_unitary(coefficient)
            )
            right_stabilizer = np.asarray(
                source_route
                @ left_stabilizer
                @ target_route.conjugate().T,
                dtype=np.complex128,
            )
            stabilizer[np.ix_(left, left)] = left_stabilizer
            stabilizer[np.ix_(right, right)] = right_stabilizer
        elif len(group_cycle) == 1:
            group = groups[group_cycle[0]]
            source_internal = np.asarray(
                source_block[np.ix_(group, group)],
                dtype=np.complex128,
            )
            target_internal = np.asarray(
                target_block[np.ix_(group, group)],
                dtype=np.complex128,
            )
            source_bases = _cyclotomic_eigenspace_bases(
                source_internal,
                root_order=2 * root_order,
                bound=bound,
            )
            target_bases = _cyclotomic_eigenspace_bases(
                target_internal,
                root_order=2 * root_order,
                bound=bound,
            )
            if source_bases.keys() != target_bases.keys():
                raise JointExactificationError(
                    "fixed C2 monomial target has the wrong internal spectrum"
                )
            frame_block = np.asarray(
                common_frame[np.ix_(group, group)],
                dtype=np.complex128,
            )
            internal_stabilizer = np.zeros_like(
                source_internal,
                dtype=np.complex128,
            )
            for exponent in source_bases:
                source_basis = source_bases[exponent]
                target_basis = target_bases[exponent]
                if source_basis.shape[1] != target_basis.shape[1]:
                    raise JointExactificationError(
                        "fixed C2 eigenspace multiplicities do not match target"
                    )
                coefficient = np.asarray(
                    target_basis.conjugate().T
                    @ frame_block
                    @ source_basis,
                    dtype=np.complex128,
                )
                residual_unitary, _singular_values, _report = (
                    _deterministic_procrustes_unitary(coefficient)
                )
                internal_stabilizer += (
                    source_basis
                    @ residual_unitary
                    @ target_basis.conjugate().T
                )
            stabilizer[np.ix_(group, group)] = internal_stabilizer
        else:
            raise JointExactificationError(
                "C2 has unsupported uniform-generator eigenspace cycle "
                f"{group_cycle}"
            )
    identity = np.eye(source_block.shape[0], dtype=np.complex128)
    checks = {
        "unitarity": np.linalg.norm(
            stabilizer.conjugate().T @ stabilizer - identity,
            ord="fro",
        ),
        "primary_preservation": np.linalg.norm(
            stabilizer.conjugate().T
            @ diagonal_action
            @ stabilizer
            - diagonal_action,
            ord="fro",
        ),
        "C2_intertwining": np.linalg.norm(
            stabilizer.conjugate().T
            @ source_block
            @ stabilizer
            - target_block,
            ord="fro",
        ),
    }
    failed = {name: float(value) for name, value in checks.items() if value > bound}
    if failed:
        raise JointExactificationError(
            f"fixed C2 commutant solver failed certification: {failed}, "
            f"bound={bound:.6e}"
        )
    return np.asarray(stabilizer, dtype=np.complex128)


def _q_uniform_c2_orbit_stabilizer_gauge(
    actions: Mapping[str, BlockRouteAction],
    *,
    primary_name: str,
    common_frame: np.ndarray,
    root_order: int,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Standardize C2 inside the orbit stabilizer of a uniform generator."""

    primary = actions[primary_name]
    c2 = actions["C2"]
    dimension = int(primary.fiber_dimensions[0])
    diagonal = np.asarray(primary.route_blocks[0], dtype=np.complex128)
    primary_cycles = tuple(_permutation_cycles(primary.fiber_permutation))
    cycle_by_fiber = {
        int(fiber): int(cycle_index)
        for cycle_index, cycle in enumerate(primary_cycles)
        for fiber in cycle
    }
    induced: list[int] = []
    for cycle in primary_cycles:
        target_cycles = {
            cycle_by_fiber[int(c2.fiber_permutation[source])]
            for source in cycle
        }
        if len(target_cycles) != 1:
            raise JointExactificationError(
                "C2 does not induce a well-defined action on Q-uniform "
                f"{primary_name} cycles: cycle={cycle}, targets={sorted(target_cycles)}"
            )
        induced.append(int(next(iter(target_cycles))))
    if sorted(induced) != list(range(len(primary_cycles))):
        raise JointExactificationError(
            f"C2 action on {primary_name} cycles is not bijective"
        )
    fiber_gauge = [
        np.eye(int(value), dtype=np.complex128)
        for value in primary.fiber_dimensions
    ]
    identity = np.eye(dimension, dtype=np.complex128)
    bound = float(
        16384.0
        * np.finfo(np.float64).eps
        * max(
            1,
            len(primary.fiber_dimensions),
            dimension,
            root_order,
        )
    )
    reports: list[dict[str, Any]] = []
    for quotient_cycle in _permutation_cycles(induced):
        if len(quotient_cycle) not in {1, 2}:
            raise JointExactificationError(
                "unitary C2 has unsupported Q-uniform-generator orbit cycle "
                f"{quotient_cycle}"
            )
        involved_fibers = tuple(
            source
            for cycle_index in quotient_cycle
            for source in primary_cycles[cycle_index]
        )
        reference_source = int(min(involved_fibers))
        reference_target = int(c2.fiber_permutation[reference_source])
        holonomy = np.asarray(
            c2.route_blocks[reference_target]
            @ c2.route_blocks[reference_source],
            dtype=np.complex128,
        )
        root_distances = [
            float(
                np.linalg.norm(
                    holonomy - _cyclotomic_root(root_order, exponent) * identity,
                    ord="fro",
                )
            )
            for exponent in range(root_order)
        ]
        holonomy_exponent = int(np.argmin(root_distances))
        if root_distances[holonomy_exponent] > bound:
            raise JointExactificationError(
                "Q-uniform C2 orbit holonomy is not a certified central root: "
                f"cycle={quotient_cycle}, residual="
                f"{root_distances[holonomy_exponent]:.6e}, bound={bound:.6e}"
            )
        phase_order = int(2 * root_order)
        phase_rows = tuple(
            (
                int(holonomy_exponent + branch * root_order),
                _cyclotomic_root(
                    phase_order,
                    holonomy_exponent + branch * root_order,
                ),
            )
            for branch in range(2)
        )
        candidate_rows: list[
            tuple[
                float,
                float,
                tuple[int, ...],
                int,
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                str,
            ]
        ] = []
        orientations = (
            ((quotient_cycle[0], quotient_cycle[1]),)
            if len(quotient_cycle) == 2
            else ((quotient_cycle[0], quotient_cycle[0]),)
        )
        if len(quotient_cycle) == 2:
            orientations = (
                (quotient_cycle[0], quotient_cycle[1]),
                (quotient_cycle[1], quotient_cycle[0]),
            )
        for source_cycle, target_cycle in orientations:
            source_fiber = int(min(primary_cycles[source_cycle]))
            if cycle_by_fiber[int(c2.fiber_permutation[source_fiber])] != target_cycle:
                continue
            source_block = np.asarray(
                c2.route_blocks[source_fiber],
                dtype=np.complex128,
            )
            diagonal_groups = _diagonal_cyclotomic_groups(
                diagonal,
                root_order=root_order,
            )
            source_mapping = _eigenspace_group_mapping(
                source_block,
                diagonal_groups,
                bound=bound,
            )
            allowed_edges = _allowed_involution_edges_for_mapping(
                diagonal_groups,
                source_mapping,
            )
            permutation_rows: list[
                tuple[tuple[int, ...], int, complex, str]
            ] = []
            if dimension <= 12:
                for permutation in _involutive_permutations(dimension):
                    selected_edges = {
                        (min(source, target), max(source, target))
                        for source, target in enumerate(permutation)
                    }
                    if not selected_edges.issubset(allowed_edges):
                        continue
                    for phase_exponent, phase in phase_rows:
                        permutation_rows.append(
                            (
                                tuple(int(value) for value in permutation),
                                int(phase_exponent),
                                complex(phase),
                                "enumerated_involutions",
                            )
                        )
            else:
                for phase_exponent, phase in phase_rows:
                    try:
                        transposition_requirements = (
                            _required_fixed_group_transpositions(
                                source_block,
                                diagonal_groups,
                                source_mapping,
                                phase=phase,
                                bound=bound,
                            )
                            if len(quotient_cycle) == 1
                            else ()
                        )
                        reference_blocks = tuple(
                            np.asarray(c2.route_blocks[source])
                            for source in involved_fibers
                        )
                        _permutation, optimal_cost = _milp_involution(
                            reference_blocks,
                            phase,
                            allowed_edges=allowed_edges,
                            required_transposition_counts=(
                                transposition_requirements
                            ),
                        )
                        permutation, _cost = _milp_involution(
                            reference_blocks,
                            phase,
                            cost_bound=optimal_cost,
                            roundoff_bound=bound,
                            allowed_edges=allowed_edges,
                            required_transposition_counts=(
                                transposition_requirements
                            ),
                        )
                    except JointExactificationError:
                        continue
                    permutation_rows.append(
                        (
                            tuple(int(value) for value in permutation),
                            int(phase_exponent),
                            complex(phase),
                            "milp_involution_with_eigenspace_constraints",
                        )
                    )
            for permutation, phase_exponent, phase, matching_solver in permutation_rows:
                permutation_matrix = _permutation_matrix(permutation)
                target_block = np.asarray(
                    phase * permutation_matrix,
                    dtype=np.complex128,
                )
                try:
                    if len(quotient_cycle) == 1:
                        source_stabilizer = _closest_fixed_c2_commutant(
                            source_block,
                            target_block,
                            diagonal,
                            common_frame,
                            root_order=root_order,
                            bound=bound,
                        )
                        target_stabilizer = source_stabilizer
                    else:
                        coefficient = np.asarray(
                            len(primary_cycles[source_cycle]) * common_frame
                            + len(primary_cycles[target_cycle])
                            * target_block.conjugate().T
                            @ common_frame
                            @ source_block,
                            dtype=np.complex128,
                        )
                        source_stabilizer = _block_commutant_procrustes(
                            coefficient,
                            diagonal_groups,
                        )
                        target_stabilizer = np.asarray(
                            source_block
                            @ source_stabilizer
                            @ target_block.conjugate().T,
                            dtype=np.complex128,
                        )
                except JointExactificationError:
                    continue
                trial_by_cycle = {
                    int(source_cycle): source_stabilizer,
                    int(target_cycle): target_stabilizer,
                }
                certification_failed = False
                for value in trial_by_cycle.values():
                    unitarity = float(
                        np.linalg.norm(
                            value.conjugate().T @ value - identity,
                            ord="fro",
                        )
                    )
                    preservation = float(
                        np.linalg.norm(
                            value.conjugate().T @ diagonal @ value - diagonal,
                            ord="fro",
                        )
                    )
                    if unitarity > bound or preservation > bound:
                        certification_failed = True
                        break
                if certification_failed:
                    continue
                route_residual = 0.0
                for source in involved_fibers:
                    source_stabilizer = trial_by_cycle[cycle_by_fiber[source]]
                    target_stabilizer = trial_by_cycle[
                        cycle_by_fiber[int(c2.fiber_permutation[source])]
                    ]
                    actual = np.asarray(
                        target_stabilizer.conjugate().T
                        @ c2.route_blocks[source]
                        @ source_stabilizer,
                        dtype=np.complex128,
                    )
                    route_residual = max(
                        route_residual,
                        float(
                            np.linalg.norm(
                                actual - target_block,
                                ord="fro",
                            )
                        ),
                    )
                if route_residual > bound:
                    continue
                objective = 0.0
                for cycle_index in quotient_cycle:
                    value = trial_by_cycle[int(cycle_index)]
                    objective += len(primary_cycles[int(cycle_index)]) * float(
                        np.linalg.norm(
                            common_frame @ value - identity,
                            ord="fro",
                        )
                        ** 2
                    )
                target_distance = sum(
                    float(
                        np.linalg.norm(
                            target_block - c2.route_blocks[source],
                            ord="fro",
                        )
                        ** 2
                    )
                    for source in involved_fibers
                )
                candidate_rows.append(
                    (
                        target_distance,
                        objective,
                        tuple(int(value) for value in permutation),
                        int(phase_exponent),
                        int(source_cycle),
                        source_stabilizer,
                        target_stabilizer,
                        target_block,
                        matching_solver,
                    )
                )
        if not candidate_rows:
            raise JointExactificationError(
                "no C2 monomial target is reachable inside the Q-uniform "
                f"{primary_name} orbit stabilizer for cycle {quotient_cycle}"
            )
        objective_bound = float(bound * max(1, len(involved_fibers)))
        best_target_distance = min(row[0] for row in candidate_rows)
        nearest_targets = [
            row
            for row in candidate_rows
            if row[0] <= best_target_distance + objective_bound
        ]
        best_objective = min(row[1] for row in nearest_targets)
        nearest = [
            row
            for row in nearest_targets
            if row[1] <= best_objective + objective_bound
        ]
        selected = min(
            nearest,
            key=lambda row: (row[2], row[3], row[4]),
        )
        (
            target_distance,
            objective,
            permutation,
            phase_exponent,
            source_cycle,
            source_stabilizer,
            target_stabilizer,
            _target_block,
            matching_solver,
        ) = selected
        target_cycle = int(induced[source_cycle])
        if source_cycle == target_cycle:
            assigned = {int(source_cycle): source_stabilizer}
        else:
            assigned = {
                int(source_cycle): source_stabilizer,
                int(target_cycle): target_stabilizer,
            }
        for cycle_index, value in assigned.items():
            for fiber in primary_cycles[cycle_index]:
                fiber_gauge[int(fiber)] = np.asarray(value, dtype=np.complex128)
        reports.append(
            {
                "primary_cycles": [int(value) for value in quotient_cycle],
                "source_cycle": int(source_cycle),
                "target_cycle": int(target_cycle),
                "internal_permutation": [int(value) for value in permutation],
                "phase_exponent": int(phase_exponent),
                "phase_order": int(phase_order),
                "target_distance_squared": float(target_distance),
                "proximity_objective": float(objective),
                "nearest_candidate_count": int(len(nearest)),
                "matching_solver": matching_solver,
                "selection": "closest_continuous_q_uniform_orbit_stabilizer",
            }
        )
    gauge = tuple(np.asarray(value, dtype=np.complex128) for value in fiber_gauge)
    primary_reframed = _gauge_transform_fiber_actions(
        {primary_name: primary},
        gauge,
    )[primary_name]
    reference_block = np.asarray(primary_reframed.route_blocks[0])
    primary_residual = max(
        float(np.linalg.norm(block - reference_block, ord="fro"))
        for block in primary_reframed.route_blocks
    )
    if primary_residual > bound:
        raise JointExactificationError(
            f"C2 orbit stabilizer broke Q-uniform {primary_name}: "
            f"residual={primary_residual:.6e}, bound={bound:.6e}"
        )
    return gauge, {
        "status": "certified",
        "primary_cycle_count": int(len(primary_cycles)),
        "C2_quotient_cycles": reports,
        "primary_route_residual_max": primary_residual,
        "certification_bound": bound,
        "tie_break": "minimum_Frobenius_then_algebraic_lexicographic",
    }


def _derive_q_uniform_generator_result(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> StandardGeneratorFiberGaugeResult | None:
    """Use one internal frame when a resolving finite generator is Q-uniform."""

    from kp.basis.symmetry_gauge import (
        SymmetryGaugeOperation,
        derive_symmetry_adapted_internal_frame,
    )

    names = tuple(generator.name for generator in presentation.generators)
    reference = actions[names[0]]
    if max(reference.fiber_dimensions) <= 1:
        return None
    root_order = _cyclotomic_order_from_presentation(presentation)
    uniform: dict[str, tuple[np.ndarray, float, float]] = {}
    operation_rows: list[SymmetryGaugeOperation] = []
    primary_candidates: set[str] = set()
    for generator in presentation.generators:
        action = actions[generator.name]
        row = _uniform_route_block(action, root_order=root_order)
        if row is None:
            continue
        block, _residual, _bound = row
        power = _declared_generator_projective_power(
            presentation,
            generator.name,
        )
        uniform[generator.name] = row
        operation_rows.append(
            SymmetryGaugeOperation(
                name=generator.name,
                matrix=block,
                antiunitary=generator.antiunitary,
                power=power,
                can_resolve=(
                    not generator.antiunitary
                    and generator.name != "C2"
                    and power is not None
                    and power >= 3
                ),
                can_pair=generator.antiunitary,
                can_anchor=not generator.antiunitary,
            )
        )
        if (
            not generator.antiunitary
            and generator.name != "C2"
            and power is not None
            and power >= 3
        ):
            primary_candidates.add(generator.name)
    if not primary_candidates:
        return None
    adapted = derive_symmetry_adapted_internal_frame(
        operation_rows,
        tolerance=1.0e-10,
    )
    if (
        adapted.status != "applied"
        or adapted.primary_operation not in primary_candidates
    ):
        return None
    primary_name = str(adapted.primary_operation)
    adapted_frame = np.asarray(adapted.unitary, dtype=np.complex128)
    primary_input = uniform[primary_name][0]
    adapted_diagonal = np.asarray(
        adapted_frame.conjugate().T @ primary_input @ adapted_frame,
        dtype=np.complex128,
    )
    internal_frame, closest_stabilizer = _closest_algebraic_uniform_stabilizer(
        adapted_frame,
        adapted_diagonal,
        root_order=root_order,
    )
    common_fiber_gauge = tuple(
        internal_frame.copy() for _ in reference.fiber_dimensions
    )
    transformed = _gauge_transform_fiber_actions(actions, common_fiber_gauge)
    primary_transformed = transformed[primary_name]
    common_diagonal = np.asarray(
        primary_transformed.route_blocks[0],
        dtype=np.complex128,
    )
    off_diagonal = float(
        np.linalg.norm(
            common_diagonal - np.diag(np.diag(common_diagonal)),
            ord="fro",
        )
    )
    primary_uniform_residual = max(
        float(np.linalg.norm(block - common_diagonal, ord="fro"))
        for block in primary_transformed.route_blocks
    )
    primary_bound = uniform[primary_name][2]
    if off_diagonal > primary_bound or primary_uniform_residual > primary_bound:
        raise JointExactificationError(
            f"Q-uniform {primary_name} frame did not produce one diagonal route: "
            f"off_diagonal={off_diagonal:.6e}, "
            f"uniform_residual={primary_uniform_residual:.6e}, "
            f"bound={primary_bound:.6e}"
        )
    has_unitary_c2 = any(
        generator.name == "C2" and not generator.antiunitary
        for generator in presentation.generators
    )
    try:
        if has_unitary_c2:
            stabilizer_gauge, orbit_stabilizer_report = (
                _q_uniform_c2_orbit_stabilizer_gauge(
                    transformed,
                    primary_name=primary_name,
                    common_frame=internal_frame,
                    root_order=root_order,
                )
            )
            stabilized = _gauge_transform_fiber_actions(
                transformed,
                stabilizer_gauge,
            )
        else:
            stabilizer_gauge = tuple(
                np.eye(int(value), dtype=np.complex128)
                for value in reference.fiber_dimensions
            )
            orbit_stabilizer_report = {
                "status": "not_applicable",
                "reason": "no_unitary_C2",
            }
            stabilized = transformed
        algebraic = _canonicalize_ud_monomial_actions_without_c2(
            stabilized,
            presentation,
        )
    except JointExactificationError as exc:
        raise JointExactificationError(
            f"Q-uniform {primary_name} frame cannot be algebraically exactified "
            f"without breaking its common route: {exc}"
        ) from exc
    target_actions = dict(algebraic.actions)
    fiber_gauge = tuple(
        np.asarray(common @ stabilizer, dtype=np.complex128)
        for common, stabilizer in zip(
            common_fiber_gauge,
            stabilizer_gauge,
        )
    )
    exact_primary = target_actions[primary_name]
    exact_reference = np.asarray(exact_primary.route_blocks[0])
    exact_uniform_residual = max(
        float(np.linalg.norm(block - exact_reference, ord="fro"))
        for block in exact_primary.route_blocks
    )
    if exact_uniform_residual != 0.0:
        raise JointExactificationError(
            f"algebraic {primary_name} routes are not bitwise Q-uniform: "
            f"residual={exact_uniform_residual:.6e}"
        )
    relation_certification = certify_joint_block_actions(
        target_actions,
        presentation,
    )
    reframed = _gauge_transform_fiber_actions(actions, fiber_gauge)
    common_gauge_residual = max(
        float(np.linalg.norm(actual - expected, ord="fro"))
        for name in names
        for actual, expected in zip(
            reframed[name].route_blocks,
            target_actions[name].route_blocks,
        )
    )
    matrix_bound = float(
        8192.0
        * np.finfo(np.float64).eps
        * max(
            1,
            len(reference.fiber_dimensions),
            len(names),
            max(reference.fiber_dimensions),
            root_order,
        )
    )
    if common_gauge_residual > matrix_bound:
        raise JointExactificationError(
            "Q-uniform common internal frame does not reproduce its algebraic "
            f"targets: residual={common_gauge_residual:.6e}, "
            f"bound={matrix_bound:.6e}"
        )
    algebraic_report = dict(algebraic.report)
    spectrum = [
        [float(value.real), float(value.imag)]
        for value in np.diag(exact_reference)
    ]
    return StandardGeneratorFiberGaugeResult(
        actions=target_actions,
        fiber_gauge=fiber_gauge,
        gauge_angles=tuple(0.0 for _ in reference.fiber_dimensions),
        report={
            **algebraic_report,
            "fiber_mode": "q_uniform_Ud",
            "selection_policy": (
                "one_common_internal_finite_generator_frame_then_"
                "algebraic_monomial_exactification"
            ),
            "tie_break_policy": "deterministic_common_internal_eigenframe",
            "uniform_generator": {
                "name": primary_name,
                "status": "certified",
                "input_route_residual_max": uniform[primary_name][1],
                "input_route_certification_bound": primary_bound,
                "post_frame_off_diagonal_residual": off_diagonal,
                "post_frame_route_residual_max": primary_uniform_residual,
                "exact_route_residual_max": exact_uniform_residual,
                "spectrum": spectrum,
                "internal_frame": adapted.artifact(),
                "closest_algebraic_stabilizer": closest_stabilizer,
                "C2_orbit_stabilizer": orbit_stabilizer_report,
            },
            "C2_standardization_status": (
                "certified_in_q_uniform_stabilizer_gauge"
                if has_unitary_c2
                else "not_applicable"
            ),
            "common_gauge_proximity_objective": float(
                sum(
                    np.linalg.norm(
                        value - np.eye(value.shape[0]),
                        ord="fro",
                    )
                    ** 2
                    for value in fiber_gauge
                )
            ),
            "common_gauge_residual_max": common_gauge_residual,
            "common_gauge_certification_bound": matrix_bound,
            "relation_certification": dict(relation_certification),
        },
    )


def derive_standard_generator_fiber_gauge(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> StandardGeneratorFiberGaugeResult:
    """Choose a certified standard generator gauge for scalar or free U(d) fibers.

    Non-scalar free orbits are first parallel-transported to their exact central
    cocycle.  The scalar cocycle then uses the same nearest uniform ``C2`` root
    rule as U(1), while the remaining root-frame freedom is fixed by the global
    minimum-Frobenius unitary Procrustes solution.
    """

    validate_presentation_action_relations(actions, presentation)
    if all(
        dimension == 1
        for action in actions.values()
        for dimension in action.fiber_dimensions
    ):
        scalar = derive_standard_generator_u1_gauge(actions, presentation)
        report = dict(scalar.report)
        if report.get("status") == "certified":
            report["fiber_mode"] = "U1"
        return StandardGeneratorFiberGaugeResult(
            actions=scalar.actions,
            fiber_gauge=scalar.fiber_gauge,
            gauge_angles=scalar.gauge_angles,
            report=report,
        )

    q_uniform = _derive_q_uniform_generator_result(actions, presentation)
    if q_uniform is not None:
        return q_uniform

    names = tuple(generator.name for generator in presentation.generators)
    has_unitary_c2 = any(
        generator.name == "C2" and not generator.antiunitary
        for generator in presentation.generators
    )
    reference = actions[names[0]]
    orbits = compile_action_orbits(actions, presentation)

    frames: dict[int, np.ndarray] = {}
    parities: dict[int, bool] = {}
    base_gauge: list[np.ndarray | None] = [None] * len(reference.fiber_dimensions)
    free_roots: set[int] = set()
    orbit_reports: list[dict[str, Any]] = []
    for orbit in orbits:
        orbit_frames = _canonical_transporter_frames(actions, orbit)
        if orbit.stabilizer_words == ((),):
            root_frame = np.eye(
                int(reference.fiber_dimensions[orbit.root]),
                dtype=np.complex128,
            )
            free_roots.add(orbit.root)
            root_report: dict[str, Any] = {
                "status": "free_orbit",
                "transporter_policy": "canonical_shortest_words",
            }
        else:
            root_frame, root_report = _canonical_stabilizer_root_frame(
                actions,
                presentation,
                orbit,
            )
        for fiber, word in zip(orbit.fibers, orbit.transporter_words):
            frames[fiber] = np.asarray(orbit_frames[fiber], dtype=np.complex128)
            parities[fiber] = _transporter_antiunitary_parity(word, presentation)
            base_gauge[fiber] = np.asarray(
                orbit_frames[fiber]
                @ semilinear_kappa(root_frame, parities[fiber]),
                dtype=np.complex128,
            )
        orbit_reports.append(
            {
                "root": int(orbit.root),
                "fibers": [int(value) for value in orbit.fibers],
                "block_dimension": int(reference.fiber_dimensions[orbit.root]),
                "stabilizer_size": int(len(orbit.stabilizer_words)),
                "root_frame": root_report,
            }
        )

    if any(value is None for value in base_gauge):
        raise JointExactificationError(
            "standard U(d) transporter construction did not assign every fiber"
        )
    certified_base_gauge = tuple(
        np.asarray(value, dtype=np.complex128)
        for value in base_gauge
        if value is not None
    )
    base_actions = _gauge_transform_fiber_actions(actions, certified_base_gauge)
    certify_joint_block_actions(base_actions, presentation)
    fiber_count = len(reference.fiber_dimensions)
    ud_standard = (
        _standardize_ud_monomial_actions(
            base_actions,
            actions,
            presentation,
        )
        if has_unitary_c2
        else _canonicalize_ud_monomial_actions_without_c2(
            base_actions,
            presentation,
        )
    )
    if ud_standard.report.get("status") != "certified":
        return _identity_standard_generator_result(
            actions,
            reason=str(ud_standard.report.get("reason", "Ud_standardization_failed")),
        )
    target_actions = dict(ud_standard.actions)
    relation_certification = certify_joint_block_actions(
        target_actions,
        presentation,
    )

    fiber_gauge: list[np.ndarray | None] = [None] * fiber_count
    proximity_objective = 0.0
    procrustes_reports: list[dict[str, Any]] = []
    for orbit in orbits:
        dimension = int(reference.fiber_dimensions[orbit.root])
        if orbit.root not in free_roots:
            for fiber in orbit.fibers:
                value = np.asarray(
                    certified_base_gauge[fiber]
                    @ ud_standard.fiber_gauge[fiber],
                    dtype=np.complex128,
                )
                fiber_gauge[fiber] = value
                proximity_objective += float(
                    np.linalg.norm(
                        value - np.eye(dimension, dtype=np.complex128),
                        ord="fro",
                    )
                    ** 2
                )
            procrustes_reports.append(
                {
                    "root": int(orbit.root),
                    "status": "fixed_by_stabilizer_frame",
                }
            )
            continue
        coefficient = np.zeros((dimension, dimension), dtype=np.complex128)
        for fiber in orbit.fibers:
            frame = frames[fiber]
            standard_gauge = ud_standard.fiber_gauge[fiber]
            procrustes_factor = standard_gauge @ frame
            coefficient += (
                procrustes_factor
                if not parities[fiber]
                else procrustes_factor.conjugate()
            )
        root_unitary, singular_values, procrustes_certification = (
            _deterministic_procrustes_unitary(coefficient)
        )
        for fiber in orbit.fibers:
            residual_root = semilinear_kappa(root_unitary, parities[fiber])
            value = np.asarray(
                frames[fiber]
                @ residual_root
                @ ud_standard.fiber_gauge[fiber],
                dtype=np.complex128,
            )
            fiber_gauge[fiber] = value
            proximity_objective += float(
                np.linalg.norm(
                    value - np.eye(dimension, dtype=np.complex128),
                    ord="fro",
                )
                ** 2
            )
        procrustes_reports.append(
            {
                "root": int(orbit.root),
                "status": "minimum_frobenius_root_frame",
                "singular_values": [float(value) for value in singular_values],
                **procrustes_certification,
            }
        )
    if any(value is None for value in fiber_gauge):
        raise JointExactificationError(
            "standard U(d) gauge did not assign every fiber"
        )
    certified_gauge = tuple(
        np.asarray(value, dtype=np.complex128)
        for value in fiber_gauge
        if value is not None
    )
    reframed = _gauge_transform_fiber_actions(actions, certified_gauge)
    common_gauge_residual_max = max(
        float(np.linalg.norm(actual - expected, ord="fro"))
        for name in names
        for actual, expected in zip(
            reframed[name].route_blocks,
            target_actions[name].route_blocks,
        )
    )
    matrix_bound = float(
        8192.0
        * np.finfo(np.float64).eps
        * max(
            1,
            fiber_count,
            len(names),
            max(reference.fiber_dimensions),
        )
    )
    if common_gauge_residual_max > matrix_bound:
        raise JointExactificationError(
            "common U(d) fiber gauge did not reproduce the standard "
            f"generator targets: residual={common_gauge_residual_max:.6e}, "
            f"bound={matrix_bound:.6e}"
        )
    standard_report = dict(ud_standard.report)
    return StandardGeneratorFiberGaugeResult(
        actions=target_actions,
        fiber_gauge=certified_gauge,
        gauge_angles=tuple(0.0 for _ in range(fiber_count)),
        report={
            **standard_report,
            "fiber_mode": (
                "free_orbit_Ud"
                if len(free_roots) == len(orbits)
                else "mixed_free_stabilized_Ud"
                if free_roots
                else "stabilized_orbit_Ud"
            ),
            "selection_policy": (
                "nearest_uniform_monomial_C2_then_closest_common_Ud_gauge"
                if has_unitary_c2
                else "algebraic_monomial_orbit_frame_then_closest_common_Ud_gauge"
            ),
            "tie_break_policy": "global_unitary_procrustes_per_free_orbit",
            "orbit_transport": orbit_reports,
            "procrustes": procrustes_reports,
            "common_gauge_proximity_objective": proximity_objective,
            "common_gauge_residual_max": common_gauge_residual_max,
            "common_gauge_certification_bound": matrix_bound,
            "relation_certification": dict(relation_certification),
        },
    )


@dataclass(frozen=True)
class JointExactificationConfig:
    """Numerical safety gates shared by joint exactification stages."""

    enabled: bool = True
    max_rms_correction: float = 1.0e-3
    max_route_correction: float = 5.0e-3
    central_branch_margin: float = 1.0e-3
    max_iterations: int = 20
    condition_limit: float = 1.0e12

    def __post_init__(self) -> None:
        for name in (
            "max_rms_correction",
            "max_route_correction",
            "central_branch_margin",
            "condition_limit",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise JointExactificationError(
                    f"joint exactification {name} must be finite and positive"
                )
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "max_iterations",
            _positive_limit(self.max_iterations, label="maximum iteration count"),
        )
        object.__setattr__(self, "enabled", bool(self.enabled))


def condition_limited_svd_rank(
    singular_values: Sequence[float],
    *,
    matrix_shape: Sequence[int],
    condition_limit: float,
) -> tuple[int, float, int, float]:
    """Select the stable SVD subspace used by a minimum-norm Newton step.

    Directions below the floating-point numerical-rank bound are null.  A
    direction that is numerically nonzero but would make the retained system
    more ill-conditioned than ``condition_limit`` is also left in the null
    space.  This is a truncated-SVD solve: convergence and the final relation
    certificate remain the authority, so a discarded direction can never
    silently hide a required correction.
    """

    values = np.asarray(tuple(singular_values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise JointExactificationError(
            "SVD singular values must be one nonempty finite vector"
        )
    largest = float(values[0])
    if largest <= 0.0 or np.any(values < 0.0):
        raise JointExactificationError("SVD system has zero rank")
    limit = float(condition_limit)
    if not np.isfinite(limit) or limit <= 1.0:
        raise JointExactificationError(
            "SVD condition limit must be finite and greater than one"
        )
    shape = tuple(int(value) for value in matrix_shape)
    if not shape or any(value <= 0 for value in shape):
        raise JointExactificationError("SVD matrix shape must be positive")
    numerical_cutoff = float(
        largest * max(shape) * np.finfo(np.float64).eps
    )
    numerical_rank = int(np.count_nonzero(values > numerical_cutoff))
    stable_cutoff = float(max(numerical_cutoff, largest / limit))
    rank = int(np.count_nonzero(values > stable_cutoff))
    if rank == 0:
        raise JointExactificationError(
            "SVD system has no direction inside the configured condition limit"
        )
    condition_number = float(largest / values[rank - 1])
    return rank, condition_number, numerical_rank, stable_cutoff


@dataclass(frozen=True)
class FreeOrbitReport:
    """Certification and optimization data for one free action orbit."""

    root: int
    fibers: tuple[int, ...]
    block_dimension: int
    solver: str
    iterations: int
    converged: bool
    objective_initial: float
    objective_final: float
    route_correction_rms: float
    route_correction_max: float
    minimum_central_separation: float
    central_transition_counts: Mapping[str, int]
    condition_number: float
    relation_residual_max: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", int(self.root))
        object.__setattr__(self, "fibers", tuple(int(value) for value in self.fibers))
        object.__setattr__(self, "block_dimension", int(self.block_dimension))
        object.__setattr__(self, "solver", str(self.solver))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(
            self,
            "central_transition_counts",
            MappingProxyType(
                {
                    str(key): int(value)
                    for key, value in self.central_transition_counts.items()
                }
            ),
        )


def pack_skew_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Pack a skew-Hermitian matrix into deterministic real ``u(d)`` coordinates."""

    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise JointExactificationError(
            f"skew-Hermitian matrix must be square, got shape {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise JointExactificationError("skew-Hermitian matrix must be finite")
    dimension = int(value.shape[0])
    defect = float(np.linalg.norm(value + value.conj().T, ord="fro"))
    scale = max(1.0, float(np.linalg.norm(value, ord="fro")))
    bound = float(64.0 * np.finfo(np.float64).eps * max(1, dimension) * scale)
    if defect > bound:
        raise JointExactificationError(
            f"matrix is not skew-Hermitian: residual={defect:.6e}, "
            f"bound={bound:.6e}"
        )
    coordinates: list[float] = [float(value[index, index].imag) for index in range(dimension)]
    root_two = float(np.sqrt(2.0))
    for row in range(dimension):
        for column in range(row + 1, dimension):
            coordinates.append(root_two * float(value[row, column].real))
            coordinates.append(root_two * float(value[row, column].imag))
    return np.asarray(coordinates, dtype=np.float64)


def unpack_skew_hermitian(
    coordinates: Sequence[float],
    dimension: int,
) -> np.ndarray:
    """Restore a skew-Hermitian matrix from ``pack_skew_hermitian`` coordinates."""

    size = _positive_limit(dimension, label="skew-Hermitian dimension")
    values = np.asarray(tuple(coordinates), dtype=np.float64)
    if values.shape != (size * size,) or not np.all(np.isfinite(values)):
        raise JointExactificationError(
            "skew-Hermitian coordinates must be a finite vector of length "
            f"{size * size}, got shape {values.shape}"
        )
    matrix = np.zeros((size, size), dtype=np.complex128)
    cursor = 0
    for index in range(size):
        matrix[index, index] = 1.0j * values[cursor]
        cursor += 1
    inverse_root_two = float(1.0 / np.sqrt(2.0))
    for row in range(size):
        for column in range(row + 1, size):
            upper = inverse_root_two * complex(
                values[cursor], values[cursor + 1]
            )
            cursor += 2
            matrix[row, column] = upper
            matrix[column, row] = -upper.conjugate()
    return matrix


@dataclass(frozen=True)
class StabilizedOrbitReport:
    """Diagnostics for the constrained projection on a non-free fiber orbit."""

    root: int
    fibers: tuple[int, ...]
    block_dimension: int
    solver: str
    iterations: int
    converged: bool
    rank: int
    numerical_rank: int
    nullity: int
    stable_svd_cutoff: float
    truncated_direction_count: int
    condition_number: float
    relation_objective_initial: float
    relation_objective_final: float
    route_correction_rms: float
    route_correction_max: float
    minimum_relation_log_branch_margin: float
    relation_residual_max: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", int(self.root))
        object.__setattr__(self, "fibers", tuple(int(value) for value in self.fibers))
        object.__setattr__(self, "block_dimension", int(self.block_dimension))
        object.__setattr__(self, "solver", str(self.solver))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "numerical_rank", int(self.numerical_rank))
        object.__setattr__(self, "nullity", int(self.nullity))
        object.__setattr__(
            self,
            "truncated_direction_count",
            int(self.truncated_direction_count),
        )


def _evaluate_route_word(
    actions: Mapping[str, BlockRouteAction],
    word: Sequence[str],
    source: int,
) -> tuple[np.ndarray, int, bool]:
    dimension = actions[next(iter(actions))].fiber_dimensions[source]
    block = np.eye(dimension, dtype=np.complex128)
    current = int(source)
    antiunitary = False
    for name in reversed(tuple(word)):
        action = actions[name]
        block = action.route_blocks[current] @ semilinear_kappa(
            block,
            action.antiunitary,
        )
        current = action.fiber_permutation[current]
        antiunitary ^= action.antiunitary
    return block, current, antiunitary


def _route_word_left_variation(
    actions: Mapping[str, BlockRouteAction],
    word: Sequence[str],
    source: int,
    *,
    variable: tuple[str, int],
    generator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a route word and its left-trivialized edge variation."""

    names = tuple(word)
    route_sources = [0] * len(names)
    current = int(source)
    for position in range(len(names) - 1, -1, -1):
        name = names[position]
        route_sources[position] = current
        current = actions[name].fiber_permutation[current]
    dimension = actions[next(iter(actions))].fiber_dimensions[source]
    prefix = np.eye(dimension, dtype=np.complex128)
    left_variation = np.zeros((dimension, dimension), dtype=np.complex128)
    prefix_antiunitary = False
    for position, name in enumerate(names):
        action = actions[name]
        edge_source = route_sources[position]
        if variable == (name, edge_source):
            transformed_generator = semilinear_kappa(
                generator,
                prefix_antiunitary,
            )
            left_variation += (
                prefix @ transformed_generator @ prefix.conj().T
            )
        factor = semilinear_kappa(
            action.route_blocks[edge_source],
            prefix_antiunitary,
        )
        prefix = prefix @ factor
        prefix_antiunitary ^= action.antiunitary
    return prefix, left_variation


def _skew_part(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (matrix - matrix.conj().T), dtype=np.complex128)


def _relation_mismatch(
    actions: Mapping[str, BlockRouteAction],
    relation: MagneticRelation,
    source: int,
) -> np.ndarray:
    lhs, lhs_target, lhs_antiunitary = _evaluate_route_word(
        actions,
        relation.lhs,
        source,
    )
    rhs, rhs_target, rhs_antiunitary = _evaluate_route_word(
        actions,
        relation.rhs,
        source,
    )
    if (lhs_target, lhs_antiunitary) != (rhs_target, rhs_antiunitary):
        raise JointExactificationError(
            f"relation {relation.name!r} has inconsistent discrete targets"
        )
    target = relation.central_phase * rhs
    return np.asarray(lhs @ target.conj().T, dtype=np.complex128)


def _minimum_principal_branch_margin(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(matrix)
    angles = np.angle(eigenvalues)
    return float(np.min(np.pi - np.abs(angles)))


def _stabilized_relation_system(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    variables: Sequence[tuple[str, int]],
    basis: Sequence[np.ndarray],
    *,
    with_jacobian: bool,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    dimension = actions[next(iter(actions))].fiber_dimensions[orbit.root]
    row_count = len(presentation.relations) * len(orbit.fibers) * dimension * dimension
    column_count = len(variables) * len(basis)
    residual = np.empty(row_count, dtype=np.float64)
    jacobian = (
        np.empty((row_count, column_count), dtype=np.float64)
        if with_jacobian
        else None
    )
    minimum_margin = float("inf")
    row_start = 0
    for relation in presentation.relations:
        for source in orbit.fibers:
            mismatch = _relation_mismatch(actions, relation, source)
            minimum_margin = min(
                minimum_margin,
                _minimum_principal_branch_margin(mismatch),
            )
            row_stop = row_start + dimension * dimension
            residual[row_start:row_stop] = pack_skew_hermitian(
                _skew_part(mismatch)
            )
            if jacobian is not None:
                lhs, _, _ = _evaluate_route_word(actions, relation.lhs, source)
                rhs, _, _ = _evaluate_route_word(actions, relation.rhs, source)
                target = relation.central_phase * rhs
                relation_mismatch = lhs @ target.conj().T
                for variable_index, variable in enumerate(variables):
                    for basis_index, generator in enumerate(basis):
                        _, lhs_variation = _route_word_left_variation(
                            actions,
                            relation.lhs,
                            source,
                            variable=variable,
                            generator=generator,
                        )
                        _, rhs_variation = _route_word_left_variation(
                            actions,
                            relation.rhs,
                            source,
                            variable=variable,
                            generator=generator,
                        )
                        mismatch_variation = (
                            lhs_variation
                            - relation_mismatch
                            @ rhs_variation
                            @ relation_mismatch.conj().T
                        )
                        delta = mismatch_variation @ relation_mismatch
                        column = variable_index * len(basis) + basis_index
                        jacobian[row_start:row_stop, column] = (
                            pack_skew_hermitian(_skew_part(delta))
                        )
            row_start = row_stop
    return residual, jacobian, minimum_margin


def _antiunitary_fixed_fiber_compatibility(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
) -> None:
    for relation in presentation.relations:
        if (
            len(relation.lhs) != 2
            or relation.lhs[0] != relation.lhs[1]
            or relation.rhs
            or relation.central_phase != complex(-1.0, 0.0)
        ):
            continue
        name = relation.lhs[0]
        action = actions[name]
        if not action.antiunitary:
            continue
        for source in orbit.fibers:
            if (
                action.fiber_permutation[source] == source
                and action.fiber_dimensions[source] % 2
            ):
                raise JointExactificationError(
                    "fixed-fiber Kramers antiunitary with square -I requires "
                    f"even dimension, got {action.fiber_dimensions[source]}"
                )


def _updated_stabilized_actions(
    actions: Mapping[str, BlockRouteAction],
    variables: Sequence[tuple[str, int]],
    basis: Sequence[np.ndarray],
    step: np.ndarray,
    scale: float,
) -> dict[str, BlockRouteAction]:
    blocks = {
        name: [np.asarray(block) for block in action.route_blocks]
        for name, action in actions.items()
    }
    for variable_index, (name, source) in enumerate(variables):
        offset = variable_index * len(basis)
        generator = np.zeros_like(blocks[name][source])
        for basis_index, value in enumerate(basis):
            generator += scale * step[offset + basis_index] * value
        blocks[name][source] = expm(generator) @ blocks[name][source]
    return _actions_with_route_blocks(
        actions,
        {name: tuple(values) for name, values in blocks.items()},
    )


def exactify_stabilized_orbit(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    *,
    config: JointExactificationConfig,
) -> tuple[dict[str, tuple[np.ndarray, ...]], StabilizedOrbitReport]:
    """Project one non-free orbit onto all declared semilinear relations."""

    if not config.enabled:
        raise JointExactificationError("joint exactification is disabled by configuration")
    validate_presentation_action_relations(actions, presentation)
    canonical_orbits = compile_action_orbits(actions, presentation)
    if orbit not in canonical_orbits:
        raise JointExactificationError("stabilized-orbit input is not canonical")
    if len(orbit.stabilizer_words) == 1:
        raise JointExactificationError("stabilized-orbit solver requires a nontrivial stabilizer")
    dimensions = {
        actions[generator.name].fiber_dimensions[source]
        for generator in presentation.generators
        for source in orbit.fibers
    }
    if len(dimensions) != 1:
        raise JointExactificationError(
            "stabilized-orbit exactification requires one constant block dimension"
        )
    dimension = dimensions.pop()
    _antiunitary_fixed_fiber_compatibility(actions, presentation, orbit)
    variables = tuple(
        (generator.name, source)
        for generator in presentation.generators
        for source in orbit.fibers
    )
    basis = _skew_hermitian_basis(dimension)
    working = dict(actions)
    residual, jacobian, minimum_margin = _stabilized_relation_system(
        working,
        presentation,
        orbit,
        variables,
        basis,
        with_jacobian=True,
    )
    if minimum_margin <= config.central_branch_margin:
        raise JointExactificationError(
            "stabilized relation-log branch is not safely separated from pi: "
            f"margin={minimum_margin:.6e}, required="
            f"{config.central_branch_margin:.6e}"
        )
    objective_initial = float(np.dot(residual, residual))
    epsilon = float(np.finfo(np.float64).eps)
    residual_bound = float(
        128.0
        * epsilon
        * max(
            1,
            dimension,
            len(variables) * len(basis),
            len(presentation.relations),
            max(
                len(relation.lhs) + len(relation.rhs)
                for relation in presentation.relations
            ),
        )
    )
    condition_number = 1.0
    rank = 0
    numerical_rank = 0
    stable_svd_cutoff = 0.0
    iterations = 0
    converged = False
    for _ in range(config.max_iterations + 1):
        maximum = float(np.linalg.norm(residual, ord=np.inf))
        if maximum <= residual_bound:
            converged = True
            break
        if jacobian is None:
            raise JointExactificationError("missing stabilized-orbit Jacobian")
        left, singular_values, right_h = np.linalg.svd(jacobian, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] == 0.0:
            raise JointExactificationError("stabilized-orbit Jacobian has zero rank")
        rank, condition_number, numerical_rank, stable_svd_cutoff = (
            condition_limited_svd_rank(
                singular_values,
                matrix_shape=jacobian.shape,
                condition_limit=config.condition_limit,
            )
        )
        step = right_h[:rank, :].T @ (
            (left[:, :rank].T @ (-residual)) / singular_values[:rank]
        )
        objective = float(np.dot(residual, residual))
        accepted = False
        for line_search_step in range(16):
            scale = float(0.5**line_search_step)
            trial = _updated_stabilized_actions(
                working,
                variables,
                basis,
                step,
                scale,
            )
            trial_residual, trial_jacobian, trial_margin = (
                _stabilized_relation_system(
                    trial,
                    presentation,
                    orbit,
                    variables,
                    basis,
                    with_jacobian=True,
                )
            )
            trial_objective = float(np.dot(trial_residual, trial_residual))
            if (
                trial_margin > config.central_branch_margin
                and trial_objective < objective
            ):
                working = trial
                residual = trial_residual
                jacobian = trial_jacobian
                minimum_margin = min(minimum_margin, trial_margin)
                iterations += 1
                accepted = True
                break
        if not accepted:
            raise JointExactificationError(
                "stabilized-orbit line search failed to reduce relation residual"
            )
    if not converged:
        raise JointExactificationError(
            "stabilized-orbit exactification did not converge within "
            f"{config.max_iterations} iterations"
        )
    route_blocks = {
        name: tuple(action.route_blocks) for name, action in working.items()
    }
    correction_rms, correction_max = _route_correction_metrics(
        actions,
        presentation,
        orbit,
        route_blocks,
    )
    if correction_rms > config.max_rms_correction:
        raise JointExactificationError(
            "stabilized-orbit RMS correction exceeds configured limit: "
            f"correction={correction_rms:.6e}, limit={config.max_rms_correction:.6e}"
        )
    if correction_max > config.max_route_correction:
        raise JointExactificationError(
            "stabilized-orbit route correction exceeds configured limit: "
            f"correction={correction_max:.6e}, limit={config.max_route_correction:.6e}"
        )
    relation_residual = _relation_residual_for_sources(
        working,
        presentation,
        orbit.fibers,
    )
    report = StabilizedOrbitReport(
        root=orbit.root,
        fibers=orbit.fibers,
        block_dimension=dimension,
        solver="minimum_norm_route_newton",
        iterations=iterations,
        converged=True,
        rank=rank,
        numerical_rank=numerical_rank,
        nullity=len(variables) * len(basis) - rank,
        stable_svd_cutoff=stable_svd_cutoff,
        truncated_direction_count=max(0, numerical_rank - rank),
        condition_number=condition_number,
        relation_objective_initial=objective_initial,
        relation_objective_final=float(np.dot(residual, residual)),
        route_correction_rms=correction_rms,
        route_correction_max=correction_max,
        minimum_relation_log_branch_margin=minimum_margin,
        relation_residual_max=relation_residual,
    )
    return route_blocks, report


def _canonical_transporter_frames(
    actions: Mapping[str, BlockRouteAction],
    orbit: ActionOrbit,
) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    for fiber, word in zip(orbit.fibers, orbit.transporter_words):
        block, target, _ = _evaluate_route_word(actions, word, orbit.root)
        if target != fiber:
            raise JointExactificationError(
                f"transporter word {word} maps root {orbit.root} to {target}, "
                f"not declared fiber {fiber}"
            )
        frames[fiber] = np.asarray(block, dtype=np.complex128)
    return frames


def _central_phase_label(phase: complex) -> str:
    return "+1" if complex(phase) == complex(1.0, 0.0) else "-1"


def _evaluate_central_word(
    actions: Mapping[str, BlockRouteAction],
    central_transitions: Mapping[tuple[str, int], complex],
    word: Sequence[str],
    source: int,
) -> complex:
    value = complex(1.0, 0.0)
    current = int(source)
    for name in reversed(tuple(word)):
        action = actions[name]
        value = central_transitions[(name, current)] * (
            value.conjugate() if action.antiunitary else value
        )
        current = action.fiber_permutation[current]
    return value


def _select_central_transitions(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    frames: Mapping[int, np.ndarray],
    *,
    minimum_separation: float,
) -> tuple[dict[tuple[str, int], complex], float, Mapping[str, int]]:
    dimension = actions[next(iter(actions))].fiber_dimensions[orbit.root]
    identity = np.eye(dimension, dtype=np.complex128)
    transitions: dict[tuple[str, int], complex] = {}
    separations: list[float] = []
    counts = {"+1": 0, "-1": 0}
    for generator in presentation.generators:
        action = actions[generator.name]
        for source in orbit.fibers:
            target = action.fiber_permutation[source]
            source_frame = semilinear_kappa(
                frames[source],
                action.antiunitary,
            )
            holonomy = (
                frames[target].conj().T
                @ action.route_blocks[source]
                @ source_frame
            )
            distances = [
                float(np.linalg.norm(holonomy - phase * identity) / np.sqrt(dimension))
                for phase in presentation.central_phases
            ]
            order = sorted(range(len(distances)), key=lambda index: (distances[index], index))
            selected = complex(presentation.central_phases[order[0]])
            if len(order) > 1:
                separation = float(distances[order[1]] - distances[order[0]])
                separations.append(separation)
                if separation <= minimum_separation:
                    raise JointExactificationError(
                        "central transition separation does not satisfy the configured "
                        f"margin: generator={generator.name!r}, source={source}, "
                        f"separation={separation:.6e}, required={minimum_separation:.6e}"
                    )
            transitions[(generator.name, source)] = selected
            counts[_central_phase_label(selected)] += 1

    for relation in presentation.relations:
        for source in orbit.fibers:
            lhs = _evaluate_central_word(
                actions,
                transitions,
                relation.lhs,
                source,
            )
            rhs = _evaluate_central_word(
                actions,
                transitions,
                relation.rhs,
                source,
            )
            if lhs != relation.central_phase * rhs:
                raise JointExactificationError(
                    "selected central transition table violates associativity/"
                    f"presentation relation {relation.name!r} at source {source}: "
                    f"lhs={lhs}, rhs={relation.central_phase * rhs}"
                )
    minimum = min(separations) if separations else float("inf")
    return transitions, float(minimum), MappingProxyType(counts)


def _predicted_free_orbit_blocks(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    frames: Mapping[int, np.ndarray],
    central_transitions: Mapping[tuple[str, int], complex],
) -> dict[tuple[str, int], np.ndarray]:
    predicted: dict[tuple[str, int], np.ndarray] = {}
    for generator in presentation.generators:
        action = actions[generator.name]
        for source in orbit.fibers:
            target = action.fiber_permutation[source]
            source_inverse = semilinear_kappa(
                frames[source].conj().T,
                action.antiunitary,
            )
            predicted[(generator.name, source)] = np.asarray(
                central_transitions[(generator.name, source)]
                * frames[target]
                @ source_inverse,
                dtype=np.complex128,
            )
    return predicted


def _complex_residual_vector(value: np.ndarray) -> np.ndarray:
    return np.concatenate((value.real.ravel(), value.imag.ravel()))


def _free_orbit_objective(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    predicted: Mapping[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, float]:
    pieces: list[np.ndarray] = []
    for generator in presentation.generators:
        action = actions[generator.name]
        for source in orbit.fibers:
            pieces.append(
                _complex_residual_vector(
                    predicted[(generator.name, source)]
                    - action.route_blocks[source]
                )
            )
    residual = np.concatenate(pieces)
    return residual, float(np.dot(residual, residual))


def _skew_hermitian_basis(dimension: int) -> tuple[np.ndarray, ...]:
    basis: list[np.ndarray] = []
    for row in range(dimension):
        value = np.zeros((dimension, dimension), dtype=np.complex128)
        value[row, row] = 1.0j
        basis.append(value)
    normalization = float(1.0 / np.sqrt(2.0))
    for row in range(dimension):
        for column in range(row + 1, dimension):
            real = np.zeros((dimension, dimension), dtype=np.complex128)
            real[row, column] = normalization
            real[column, row] = -normalization
            basis.append(real)
            imaginary = np.zeros((dimension, dimension), dtype=np.complex128)
            imaginary[row, column] = 1.0j * normalization
            imaginary[column, row] = 1.0j * normalization
            basis.append(imaginary)
    return tuple(basis)


def _free_orbit_residual_and_jacobian(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    frames: Mapping[int, np.ndarray],
    central_transitions: Mapping[tuple[str, int], complex],
    variable_fibers: Sequence[int],
    basis: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    predicted = _predicted_free_orbit_blocks(
        actions,
        presentation,
        orbit,
        frames,
        central_transitions,
    )
    residual, objective = _free_orbit_objective(
        actions,
        presentation,
        orbit,
        predicted,
    )
    variable_offsets = {
        fiber: index * len(basis)
        for index, fiber in enumerate(variable_fibers)
    }
    rows_per_edge = 2 * basis[0].size
    edge_count = len(presentation.generators) * len(orbit.fibers)
    jacobian = np.zeros(
        (edge_count * rows_per_edge, len(variable_fibers) * len(basis)),
        dtype=np.float64,
    )
    edge_index = 0
    for generator in presentation.generators:
        action = actions[generator.name]
        for source in orbit.fibers:
            target = action.fiber_permutation[source]
            block = predicted[(generator.name, source)]
            row_start = edge_index * rows_per_edge
            row_stop = row_start + rows_per_edge
            if target in variable_offsets:
                offset = variable_offsets[target]
                for basis_index, value in enumerate(basis):
                    jacobian[row_start:row_stop, offset + basis_index] += (
                        _complex_residual_vector(value @ block)
                    )
            if source in variable_offsets:
                offset = variable_offsets[source]
                for basis_index, value in enumerate(basis):
                    source_variation = semilinear_kappa(
                        value,
                        action.antiunitary,
                    )
                    jacobian[row_start:row_stop, offset + basis_index] += (
                        _complex_residual_vector(-block @ source_variation)
                    )
            edge_index += 1
    return residual, jacobian, objective


def _updated_frames(
    frames: Mapping[int, np.ndarray],
    variable_fibers: Sequence[int],
    basis: Sequence[np.ndarray],
    step: np.ndarray,
    scale: float,
) -> dict[int, np.ndarray]:
    updated = {fiber: np.asarray(value) for fiber, value in frames.items()}
    for fiber_index, fiber in enumerate(variable_fibers):
        offset = fiber_index * len(basis)
        generator = np.zeros_like(frames[fiber])
        for basis_index, value in enumerate(basis):
            generator += scale * step[offset + basis_index] * value
        updated[fiber] = expm(generator) @ frames[fiber]
    return updated


def _full_route_block_output(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    predicted: Mapping[tuple[str, int], np.ndarray],
) -> dict[str, tuple[np.ndarray, ...]]:
    output: dict[str, tuple[np.ndarray, ...]] = {}
    orbit_fibers = set(orbit.fibers)
    for generator in presentation.generators:
        action = actions[generator.name]
        output[generator.name] = tuple(
            np.asarray(predicted[(generator.name, source)], dtype=np.complex128)
            if source in orbit_fibers
            else np.asarray(action.route_blocks[source], dtype=np.complex128)
            for source in range(len(action.route_blocks))
        )
    return output


def _actions_with_route_blocks(
    actions: Mapping[str, BlockRouteAction],
    route_blocks: Mapping[str, tuple[np.ndarray, ...]],
) -> dict[str, BlockRouteAction]:
    return {
        name: BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            route_blocks=route_blocks[name],
            fiber_indices=action.fiber_indices,
        )
        for name, action in actions.items()
    }


def _relation_residual_for_sources(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    sources: Sequence[int],
) -> float:
    maximum = 0.0
    for relation in presentation.relations:
        for source in sources:
            lhs, lhs_target, lhs_antiunitary = _evaluate_route_word(
                actions,
                relation.lhs,
                source,
            )
            rhs, rhs_target, rhs_antiunitary = _evaluate_route_word(
                actions,
                relation.rhs,
                source,
            )
            if (lhs_target, lhs_antiunitary) != (rhs_target, rhs_antiunitary):
                raise JointExactificationError(
                    f"relation {relation.name!r} has inconsistent discrete targets"
                )
            residual = float(
                np.linalg.norm(lhs - relation.central_phase * rhs)
                / np.sqrt(lhs.shape[0])
            )
            maximum = max(maximum, residual)
    return maximum


def _route_correction_metrics(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    route_blocks: Mapping[str, tuple[np.ndarray, ...]],
) -> tuple[float, float]:
    distances: list[float] = []
    for generator in presentation.generators:
        action = actions[generator.name]
        for source in orbit.fibers:
            distances.append(
                float(
                    np.linalg.norm(
                        route_blocks[generator.name][source]
                        - action.route_blocks[source]
                    )
                    / np.sqrt(action.fiber_dimensions[source])
                )
            )
    values = np.asarray(distances, dtype=np.float64)
    return (
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.max(values)),
    )


def synchronize_free_orbit(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    *,
    config: JointExactificationConfig,
) -> tuple[dict[str, tuple[np.ndarray, ...]], FreeOrbitReport]:
    """Find the nearest exact representation on one free fiber orbit."""

    if not config.enabled:
        raise JointExactificationError("joint exactification is disabled by configuration")
    validate_presentation_action_relations(actions, presentation)
    canonical_orbits = compile_action_orbits(actions, presentation)
    if orbit not in canonical_orbits:
        raise JointExactificationError("free-orbit input is not canonical for these actions")
    if orbit.stabilizer_words != ((),):
        raise JointExactificationError(
            f"orbit rooted at {orbit.root} has a nontrivial stabilizer"
        )
    dimensions = {
        actions[generator.name].fiber_dimensions[source]
        for generator in presentation.generators
        for source in orbit.fibers
    }
    if len(dimensions) != 1:
        raise JointExactificationError(
            "free-orbit synchronization requires one constant block dimension"
        )
    dimension = dimensions.pop()
    frames = _canonical_transporter_frames(actions, orbit)
    central, minimum_separation, central_counts = _select_central_transitions(
        actions,
        presentation,
        orbit,
        frames,
        minimum_separation=config.central_branch_margin,
    )
    initial_predicted = _predicted_free_orbit_blocks(
        actions,
        presentation,
        orbit,
        frames,
        central,
    )
    _, objective_initial = _free_orbit_objective(
        actions,
        presentation,
        orbit,
        initial_predicted,
    )

    condition_number = 1.0
    iterations = 0
    solver = "analytic_frame_gauss_newton"
    if dimension == 1:
        direct, direct_report = project_u1_relations(actions, presentation)
        route_blocks = {
            name: tuple(
                direct[name].route_blocks[source]
                if source in set(orbit.fibers)
                else actions[name].route_blocks[source]
                for source in range(len(actions[name].route_blocks))
            )
            for name in actions
        }
        solver = "u1_relation_projection"
        condition_number = float(direct_report["condition_number"])
    else:
        variable_fibers = tuple(
            fiber for fiber in orbit.fibers if fiber != orbit.root
        )
        basis = _skew_hermitian_basis(dimension)
        epsilon = float(np.finfo(np.float64).eps)
        converged = False
        for _ in range(config.max_iterations):
            residual, jacobian, objective = _free_orbit_residual_and_jacobian(
                actions,
                presentation,
                orbit,
                frames,
                central,
                variable_fibers,
                basis,
            )
            left_vectors, singular_values, right_vectors_h = np.linalg.svd(
                jacobian,
                full_matrices=False,
            )
            if singular_values.size == 0 or singular_values[0] == 0.0:
                raise JointExactificationError(
                    "free-orbit synchronization Jacobian has zero rank"
                )
            rank, condition_number, _numerical_rank, _stable_cutoff = (
                condition_limited_svd_rank(
                    singular_values,
                    matrix_shape=jacobian.shape,
                    condition_limit=config.condition_limit,
                )
            )
            gradient = jacobian.T @ residual
            gradient_bound = float(
                256.0
                * epsilon
                * max(jacobian.shape)
                * max(
                    1.0,
                    float(singular_values[0] * np.linalg.norm(residual)),
                )
            )
            if float(np.linalg.norm(gradient, ord=np.inf)) <= gradient_bound:
                converged = True
                break
            step = right_vectors_h[:rank, :].T @ (
                (left_vectors[:, :rank].T @ (-residual))
                / singular_values[:rank]
            )
            step_bound = float(
                256.0 * epsilon * max(jacobian.shape) * max(1.0, np.linalg.norm(step))
            )
            if float(np.linalg.norm(step, ord=np.inf)) <= step_bound:
                converged = True
                break
            accepted = False
            for line_search_step in range(14):
                scale = float(0.5**line_search_step)
                trial_frames = _updated_frames(
                    frames,
                    variable_fibers,
                    basis,
                    step,
                    scale,
                )
                trial_predicted = _predicted_free_orbit_blocks(
                    actions,
                    presentation,
                    orbit,
                    trial_frames,
                    central,
                )
                _, trial_objective = _free_orbit_objective(
                    actions,
                    presentation,
                    orbit,
                    trial_predicted,
                )
                if trial_objective < objective:
                    frames = trial_frames
                    iterations += 1
                    accepted = True
                    break
            if not accepted:
                if float(np.linalg.norm(step, ord=np.inf)) <= np.sqrt(epsilon):
                    converged = True
                    break
                raise JointExactificationError(
                    "free-orbit synchronization line search failed to decrease objective"
                )
        if not converged:
            raise JointExactificationError(
                "free-orbit synchronization did not converge within "
                f"{config.max_iterations} iterations"
            )
        final_predicted = _predicted_free_orbit_blocks(
            actions,
            presentation,
            orbit,
            frames,
            central,
        )
        route_blocks = _full_route_block_output(
            actions,
            presentation,
            orbit,
            final_predicted,
        )

    synchronized_actions = _actions_with_route_blocks(actions, route_blocks)
    relation_residual = _relation_residual_for_sources(
        synchronized_actions,
        presentation,
        orbit.fibers,
    )
    correction_rms, correction_max = _route_correction_metrics(
        actions,
        presentation,
        orbit,
        route_blocks,
    )
    if correction_rms > config.max_rms_correction:
        raise JointExactificationError(
            "free-orbit RMS correction exceeds configured limit: "
            f"correction={correction_rms:.6e}, "
            f"limit={config.max_rms_correction:.6e}"
        )
    if correction_max > config.max_route_correction:
        raise JointExactificationError(
            "free-orbit route correction exceeds configured limit: "
            f"correction={correction_max:.6e}, "
            f"limit={config.max_route_correction:.6e}"
        )
    objective_final = float(
        sum(
            np.linalg.norm(
                route_blocks[generator.name][source]
                - actions[generator.name].route_blocks[source]
            )
            ** 2
            for generator in presentation.generators
            for source in orbit.fibers
        )
    )
    report = FreeOrbitReport(
        root=orbit.root,
        fibers=orbit.fibers,
        block_dimension=dimension,
        solver=solver,
        iterations=iterations,
        converged=True,
        objective_initial=objective_initial,
        objective_final=objective_final,
        route_correction_rms=correction_rms,
        route_correction_max=correction_max,
        minimum_central_separation=minimum_separation,
        central_transition_counts=central_counts,
        condition_number=condition_number,
        relation_residual_max=relation_residual,
    )
    return route_blocks, report


JOINT_BLOCK_REPRESENTATION_V1 = "joint_block_representation_v1"


@dataclass(frozen=True)
class JointExactificationResult:
    """Certified output of one complete joint generator exactification."""

    actions: Mapping[str, BlockRouteAction]
    presentation: MagneticPresentation
    report: Mapping[str, Any]
    artifact_metadata: Mapping[str, Any]
    artifact_arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))
        object.__setattr__(
            self,
            "artifact_metadata",
            MappingProxyType(dict(self.artifact_metadata)),
        )
        object.__setattr__(
            self,
            "artifact_arrays",
            MappingProxyType(
                {
                    str(key): _readonly_complex(value)
                    for key, value in self.artifact_arrays.items()
                }
            ),
        )


def _relation_residual_summary(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> tuple[dict[str, dict[str, float]], float]:
    reference = actions[presentation.generators[0].name]
    fiber_count = len(reference.fiber_dimensions)
    table: dict[str, dict[str, float]] = {}
    global_maximum = 0.0
    for relation in presentation.relations:
        values: list[float] = []
        maximum_entry = 0.0
        for source in range(fiber_count):
            lhs, lhs_target, lhs_antiunitary = _evaluate_route_word(
                actions,
                relation.lhs,
                source,
            )
            rhs, rhs_target, rhs_antiunitary = _evaluate_route_word(
                actions,
                relation.rhs,
                source,
            )
            if (lhs_target, lhs_antiunitary) != (
                rhs_target,
                rhs_antiunitary,
            ):
                raise JointExactificationError(
                    f"relation {relation.name!r} has inconsistent discrete targets"
                )
            difference = lhs - relation.central_phase * rhs
            normalized = float(
                np.linalg.norm(difference, ord="fro") / np.sqrt(lhs.shape[0])
            )
            values.append(normalized)
            maximum_entry = max(
                maximum_entry,
                float(np.max(np.abs(difference))),
            )
        array = np.asarray(values, dtype=np.float64)
        maximum = float(np.max(array))
        table[relation.name] = {
            "rms": float(np.sqrt(np.mean(np.square(array)))),
            "max": maximum,
            "maximum_entry": maximum_entry,
        }
        global_maximum = max(global_maximum, maximum)
    return table, global_maximum


def _joint_relation_certification_bound(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbits: Sequence[ActionOrbit],
) -> float:
    maximum_dimension = max(
        dimension
        for action in actions.values()
        for dimension in action.fiber_dimensions
    )
    maximum_word_length = max(
        (
            len(relation.lhs) + len(relation.rhs)
            for relation in presentation.relations
        ),
        default=1,
    )
    maximum_orbit_coordinate_count = max(
        len(presentation.generators)
        * len(orbit.fibers)
        * maximum_dimension
        * maximum_dimension
        for orbit in orbits
    )
    return float(
        128.0
        * np.finfo(np.float64).eps
        * max(
            1,
            maximum_dimension,
            maximum_word_length,
            maximum_orbit_coordinate_count,
        )
    )


def certify_joint_block_actions(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> dict[str, Any]:
    """Certify every declared relation without changing any route block."""

    validate_presentation_action_relations(actions, presentation)
    orbits = compile_action_orbits(actions, presentation)
    relation_table, relation_maximum = _relation_residual_summary(
        actions,
        presentation,
    )
    certification_bound = _joint_relation_certification_bound(
        actions,
        presentation,
        orbits,
    )
    if relation_maximum > certification_bound:
        raise JointExactificationError(
            "joint relation certification failed: "
            f"residual={relation_maximum:.6e}, bound={certification_bound:.6e}"
        )
    return {
        "status": "certified",
        "relation_residuals": relation_table,
        "relation_residual_max": relation_maximum,
        "relation_certification_bound": certification_bound,
    }


def _operation_correction_metrics(
    before: Mapping[str, BlockRouteAction],
    after: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    rms_by_operation: dict[str, float] = {}
    max_by_operation: dict[str, float] = {}
    all_values: list[float] = []
    for generator in presentation.generators:
        values = np.asarray(
            [
                float(
                    np.linalg.norm(target - source, ord="fro")
                    / np.sqrt(source.shape[0])
                )
                for source, target in zip(
                    before[generator.name].route_blocks,
                    after[generator.name].route_blocks,
                )
            ],
            dtype=np.float64,
        )
        rms_by_operation[generator.name] = float(
            np.sqrt(np.mean(np.square(values)))
        )
        max_by_operation[generator.name] = float(np.max(values))
        all_values.extend(float(value) for value in values)
    combined = np.asarray(all_values, dtype=np.float64)
    return (
        rms_by_operation,
        max_by_operation,
        float(np.sqrt(np.mean(np.square(combined)))),
        float(np.max(combined)),
    )


def _matrix_sha256(matrix: np.ndarray) -> str:
    value = np.asarray(matrix, dtype=np.dtype("<c16"), order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes(order="C"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _presentation_metadata(presentation: MagneticPresentation) -> dict[str, Any]:
    return {
        "source": presentation.source,
        "generators": [
            {
                "name": generator.name,
                "antiunitary": generator.antiunitary,
            }
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
            [float(phase.real), float(phase.imag)]
            for phase in presentation.central_phases
        ],
    }


def _presentation_from_metadata(metadata: Mapping[str, Any]) -> MagneticPresentation:
    return MagneticPresentation(
        generators=tuple(
            MagneticGenerator(
                str(record["name"]),
                bool(record["antiunitary"]),
            )
            for record in metadata["generators"]
        ),
        relations=tuple(
            MagneticRelation(
                str(record["name"]),
                tuple(str(value) for value in record["lhs"]),
                tuple(str(value) for value in record["rhs"]),
                complex(*record["central_phase"]),
            )
            for record in metadata["relations"]
        ),
        central_phases=tuple(
            complex(*value) for value in metadata["central_phases"]
        ),
        source=str(metadata["source"]),
    )


def magnetic_presentation_artifact(
    presentation: MagneticPresentation,
) -> dict[str, Any]:
    """Return the stable JSON representation of a magnetic presentation."""

    return _presentation_metadata(presentation)


def magnetic_presentation_from_artifact(
    metadata: Mapping[str, Any],
) -> MagneticPresentation:
    """Reconstruct a magnetic presentation from its stable JSON artifact."""

    return _presentation_from_metadata(metadata)


def _free_orbit_report_metadata(report: FreeOrbitReport) -> dict[str, Any]:
    return {
        "kind": "free",
        "root": report.root,
        "fibers": list(report.fibers),
        "block_dimension": report.block_dimension,
        "solver": report.solver,
        "iterations": report.iterations,
        "converged": report.converged,
        "objective_initial": report.objective_initial,
        "objective_final": report.objective_final,
        "route_correction_rms": report.route_correction_rms,
        "route_correction_max": report.route_correction_max,
        "minimum_central_separation": report.minimum_central_separation,
        "central_transition_counts": dict(report.central_transition_counts),
        "condition_number": report.condition_number,
        "relation_residual_max": report.relation_residual_max,
    }


def _stabilized_orbit_report_metadata(
    report: StabilizedOrbitReport,
) -> dict[str, Any]:
    return {
        "kind": "stabilized",
        "root": report.root,
        "fibers": list(report.fibers),
        "block_dimension": report.block_dimension,
        "solver": report.solver,
        "iterations": report.iterations,
        "converged": report.converged,
        "rank": report.rank,
        "numerical_rank": report.numerical_rank,
        "nullity": report.nullity,
        "stable_svd_cutoff": report.stable_svd_cutoff,
        "truncated_direction_count": report.truncated_direction_count,
        "condition_number": report.condition_number,
        "relation_objective_initial": report.relation_objective_initial,
        "relation_objective_final": report.relation_objective_final,
        "route_correction_rms": report.route_correction_rms,
        "route_correction_max": report.route_correction_max,
        "minimum_relation_log_branch_margin": (
            report.minimum_relation_log_branch_margin
        ),
        "relation_residual_max": report.relation_residual_max,
    }


def _joint_artifact_hash(
    metadata_without_hash: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata_without_hash,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for key in sorted(arrays):
        digest.update(str(key).encode("utf-8"))
        value = np.asarray(arrays[key], dtype=np.dtype("<c16"), order="C")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes(order="C"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _build_joint_artifact(
    before: Mapping[str, BlockRouteAction],
    after: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    report: Mapping[str, Any],
    config: JointExactificationConfig,
    certification_bound: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    action_records: list[dict[str, Any]] = []
    for generator_index, generator in enumerate(presentation.generators):
        action = after[generator.name]
        keys: list[str] = []
        for source, block in enumerate(action.route_blocks):
            key = (
                f"__joint_block_representation_{generator_index}_route_{source}__"
            )
            keys.append(key)
            arrays[key] = np.array(block, dtype=np.complex128, order="C", copy=True)
        action_records.append(
            {
                "name": action.name,
                "antiunitary": action.antiunitary,
                "fiber_permutation": list(action.fiber_permutation),
                "fiber_dimensions": list(action.fiber_dimensions),
                "fiber_indices": [list(group) for group in action.fiber_indices],
                "route_array_keys": keys,
                "unitarity_certification_bound": float(
                    action.unitarity_certification_bound
                ),
            }
        )
    metadata: dict[str, Any] = {
        "version": JOINT_BLOCK_REPRESENTATION_V1,
        "status": "certified",
        "presentation": _presentation_metadata(presentation),
        "actions": action_records,
        "config": {
            "enabled": config.enabled,
            "max_rms_correction": config.max_rms_correction,
            "max_route_correction": config.max_route_correction,
            "central_branch_margin": config.central_branch_margin,
            "max_iterations": config.max_iterations,
            "condition_limit": config.condition_limit,
        },
        "relation_certification_bound": certification_bound,
        "stage1_action_hashes": {
            generator.name: _matrix_sha256(
                materialize_block_route_action(before[generator.name])
            )
            for generator in presentation.generators
        },
        "joint_action_hashes": {
            generator.name: _matrix_sha256(
                materialize_block_route_action(after[generator.name])
            )
            for generator in presentation.generators
        },
        "report": dict(report),
    }
    metadata["artifact_hash"] = _joint_artifact_hash(metadata, arrays)
    return metadata, arrays


def joint_exactify_block_actions(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    config: JointExactificationConfig,
) -> JointExactificationResult:
    """Jointly exactify and certify one complete continuum generator set."""

    if not config.enabled:
        raise JointExactificationError("joint exactification is disabled by configuration")
    validate_presentation_action_relations(actions, presentation)
    before = dict(actions)
    orbits = compile_action_orbits(before, presentation)
    pre_table, pre_maximum = _relation_residual_summary(before, presentation)
    orbit_reports: list[dict[str, Any]] = []
    dimensions = {
        dimension
        for action in before.values()
        for dimension in action.fiber_dimensions
    }
    if dimensions == {1}:
        working, u1_report = project_u1_relations(before, presentation)
        orbit_reports.append(
            {
                "kind": "u1_global",
                "roots": [orbit.root for orbit in orbits],
                "solver": "u1_relation_projection",
                "report": dict(u1_report),
            }
        )
    else:
        working = dict(before)
        for orbit in orbits:
            if orbit.stabilizer_words == ((),):
                route_blocks, orbit_report = synchronize_free_orbit(
                    working,
                    presentation,
                    orbit,
                    config=config,
                )
                orbit_reports.append(_free_orbit_report_metadata(orbit_report))
            else:
                route_blocks, orbit_report = exactify_stabilized_orbit(
                    working,
                    presentation,
                    orbit,
                    config=config,
                )
                orbit_reports.append(
                    _stabilized_orbit_report_metadata(orbit_report)
                )
            working = _actions_with_route_blocks(working, route_blocks)
    post_table, post_maximum = _relation_residual_summary(working, presentation)
    certification_bound = _joint_relation_certification_bound(
        working,
        presentation,
        orbits,
    )
    if post_maximum > certification_bound:
        raise JointExactificationError(
            "joint relation certification failed: "
            f"residual={post_maximum:.6e}, bound={certification_bound:.6e}"
        )
    (
        rms_by_operation,
        max_by_operation,
        correction_rms,
        correction_max,
    ) = _operation_correction_metrics(before, working, presentation)
    if correction_rms > config.max_rms_correction:
        raise JointExactificationError(
            "joint RMS correction exceeds configured limit: "
            f"correction={correction_rms:.6e}, limit={config.max_rms_correction:.6e}"
        )
    if correction_max > config.max_route_correction:
        raise JointExactificationError(
            "joint route correction exceeds configured limit: "
            f"correction={correction_max:.6e}, limit={config.max_route_correction:.6e}"
        )
    report: dict[str, Any] = {
        "status": "certified",
        "pre_relation_residuals": pre_table,
        "pre_relation_residual_max": pre_maximum,
        "post_relation_residuals": post_table,
        "post_relation_residual_max": post_maximum,
        "relation_certification_bound": certification_bound,
        "route_correction_rms_by_operation": rms_by_operation,
        "route_correction_max_by_operation": max_by_operation,
        "route_correction_rms": correction_rms,
        "route_correction_max": correction_max,
        "orbit_reports": orbit_reports,
    }
    if dimensions == {1}:
        closest = derive_closest_cyclotomic_u1_gauge(
            working,
            presentation,
            reference_actions=before,
        )
        report["closest_cyclotomic_u1_gauge"] = dict(closest.report)
    metadata, arrays = _build_joint_artifact(
        before,
        working,
        presentation,
        report,
        config,
        certification_bound,
    )
    return JointExactificationResult(
        actions=working,
        presentation=presentation,
        report=report,
        artifact_metadata=metadata,
        artifact_arrays=arrays,
    )


def reframe_joint_exactification_result(
    result: JointExactificationResult,
    actions: Mapping[str, BlockRouteAction],
    *,
    provenance: Mapping[str, Any],
) -> JointExactificationResult:
    """Rebuild a certified artifact after one common continuum-basis gauge."""

    expected_names = tuple(generator.name for generator in result.presentation.generators)
    if set(actions) != set(expected_names):
        raise JointExactificationError(
            "reframed joint actions must contain exactly the declared generators"
        )
    for name in expected_names:
        before = result.actions[name]
        after = actions[name]
        if (
            before.antiunitary != after.antiunitary
            or before.fiber_permutation != after.fiber_permutation
            or before.fiber_dimensions != after.fiber_dimensions
            or before.fiber_indices != after.fiber_indices
        ):
            raise JointExactificationError(
                f"common gauge changed the discrete route layout for operation {name!r}"
            )
    certification = certify_joint_block_actions(actions, result.presentation)
    report = dict(result.report)
    report["post_gauge_certification"] = certification
    raw_config = result.artifact_metadata.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise JointExactificationError("joint artifact lacks its effective configuration")
    config = JointExactificationConfig(
        enabled=bool(raw_config.get("enabled", True)),
        max_rms_correction=float(raw_config["max_rms_correction"]),
        max_route_correction=float(raw_config["max_route_correction"]),
        central_branch_margin=float(raw_config["central_branch_margin"]),
        max_iterations=int(raw_config["max_iterations"]),
        condition_limit=float(raw_config["condition_limit"]),
    )
    metadata, arrays = _build_joint_artifact(
        result.actions,
        actions,
        result.presentation,
        report,
        config,
        float(certification["relation_certification_bound"]),
    )
    metadata["stage1_action_hashes"] = dict(
        result.artifact_metadata["stage1_action_hashes"]
    )
    metadata["pre_gauge_joint_action_hashes"] = dict(
        result.artifact_metadata["joint_action_hashes"]
    )
    metadata["pre_gauge_artifact_hash"] = str(
        result.artifact_metadata["artifact_hash"]
    )
    metadata["post_exactification_gauge"] = dict(provenance)
    metadata_without_hash = dict(metadata)
    metadata_without_hash.pop("artifact_hash", None)
    metadata["artifact_hash"] = _joint_artifact_hash(metadata_without_hash, arrays)
    return JointExactificationResult(
        actions=actions,
        presentation=result.presentation,
        report=report,
        artifact_metadata=metadata,
        artifact_arrays=arrays,
    )


def certify_fixed_target_joint_result(
    target_actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    config: JointExactificationConfig,
    provenance: Mapping[str, Any],
    fixed_target_certificate: Mapping[str, Any],
    diagnostic_result: JointExactificationResult | None = None,
) -> JointExactificationResult:
    """Build production output directly from the project-owned target.

    A free joint solver result is deliberately optional and diagnostic-only.
    It can neither select the final gauge nor influence the final route blocks.
    """

    expected_names = tuple(generator.name for generator in presentation.generators)
    if set(target_actions) != set(expected_names):
        raise JointExactificationError(
            "fixed target must contain exactly the declared generators"
        )
    for generator in presentation.generators:
        action = target_actions[generator.name]
        if bool(action.antiunitary) != bool(generator.antiunitary):
            raise JointExactificationError(
                f"fixed target antiunitary parity mismatch for {generator.name!r}"
            )

    expected_name_set = set(expected_names)
    certificate_hashes = fixed_target_certificate.get("target_matrix_hashes")
    if not isinstance(certificate_hashes, Mapping):
        raise JointExactificationError(
            "fixed target certificate lacks target matrix hashes"
        )
    if set(certificate_hashes) != expected_name_set:
        raise JointExactificationError(
            "fixed target certificate target matrix hash names do not match "
            "the supplied target actions"
        )
    for name in expected_names:
        actual_hash = hash_array(materialize_block_route_action(target_actions[name]))
        if certificate_hashes[name] != actual_hash:
            raise JointExactificationError(
                f"fixed target certificate target matrix hash mismatch for {name!r}"
            )

    certificate_parity = fixed_target_certificate.get("operation_antiunitary")
    if (
        not isinstance(certificate_parity, Mapping)
        or set(certificate_parity) != expected_name_set
    ):
        raise JointExactificationError(
            "fixed target certificate antiunitary parity names do not match "
            "the supplied target actions"
        )
    for name in expected_names:
        declared_parity = certificate_parity[name]
        if not isinstance(declared_parity, (bool, np.bool_)) or bool(
            declared_parity
        ) != bool(target_actions[name].antiunitary):
            raise JointExactificationError(
                f"fixed target certificate antiunitary parity mismatch for {name!r}"
            )

    certificate_layout = fixed_target_certificate.get("route_layout")
    if (
        not isinstance(certificate_layout, Mapping)
        or set(certificate_layout) != expected_name_set
    ):
        raise JointExactificationError(
            "fixed target certificate route layout names do not match "
            "the supplied target actions"
        )
    for name in expected_names:
        action = target_actions[name]
        expected_layout = {
            "fiber_permutation": list(action.fiber_permutation),
            "fiber_dimensions": list(action.fiber_dimensions),
            "fiber_indices": [list(group) for group in action.fiber_indices],
        }
        declared_layout = certificate_layout[name]
        if not isinstance(declared_layout, Mapping) or any(
            declared_layout.get(field) != value
            for field, value in expected_layout.items()
        ):
            raise JointExactificationError(
                f"fixed target certificate route layout mismatch for {name!r}"
            )

    cleanup = fixed_target_certificate.get(
        "post_exactification_storage_cleanup", {}
    )
    expected_cleanup_version = "post_exactification_storage_cleanup_v1"
    if (
        fixed_target_certificate.get("status") != "certified"
        or fixed_target_certificate.get("numeric_entries_modified") != 0
        or fixed_target_certificate.get("numeric_entries_modified_scope")
        != expected_cleanup_version
        or not isinstance(cleanup, Mapping)
        or cleanup.get("version") != expected_cleanup_version
        or cleanup.get("numeric_entries_modified") != 0
        or cleanup.get("threshold_cleanup_performed") is not False
    ):
        raise JointExactificationError(
            "fixed target lacks a zero-mutation post-exactification storage certificate"
        )
    if fixed_target_certificate.get("additional_basis_gauge_applied") is not False:
        raise JointExactificationError(
            "fixed target certificate must prove that no additional basis gauge was applied"
        )

    relation_certification = certify_joint_block_actions(
        target_actions,
        presentation,
    )
    diagnostic_record: dict[str, Any]
    if diagnostic_result is None:
        diagnostic_record = {
            "status": "not_run",
            "production_used": False,
            "reason": "project_owned_target_is_authoritative",
        }
    else:
        diagnostic_record = {
            "status": str(
                diagnostic_result.artifact_metadata.get("status", "unknown")
            ),
            "production_used": False,
            "artifact_hash": str(
                diagnostic_result.artifact_metadata.get("artifact_hash", "")
            ),
            "joint_action_hashes": dict(
                diagnostic_result.artifact_metadata.get("joint_action_hashes", {})
            ),
        }
    residuals = dict(relation_certification["relation_residuals"])
    residual_max = float(relation_certification["relation_residual_max"])
    report: dict[str, Any] = {
        "status": "certified",
        "production_path": "persisted_project_owned_fixed_target",
        "fixed_target_exactification": dict(fixed_target_certificate),
        "project_target_relation_certification": relation_certification,
        "relation_certification_bound": float(
            relation_certification["relation_certification_bound"]
        ),
        "pre_relation_residuals": residuals,
        "pre_relation_residual_max": residual_max,
        "post_relation_residuals": residuals,
        "post_relation_residual_max": residual_max,
        "route_correction_rms_by_operation": dict(
            fixed_target_certificate.get("correction_rms_by_operation", {})
        ),
        "route_correction_max_by_operation": dict(
            fixed_target_certificate.get("correction_max_by_operation", {})
        ),
        "route_correction_rms": float(
            fixed_target_certificate.get("correction_rms", 0.0)
        ),
        "route_correction_max": float(
            fixed_target_certificate.get("correction_max", 0.0)
        ),
        "diagnostic_joint_result": diagnostic_record,
    }
    metadata, arrays = _build_joint_artifact(
        target_actions,
        target_actions,
        presentation,
        report,
        config,
        float(relation_certification["relation_certification_bound"]),
    )
    metadata["production_action_source"] = "persisted_project_canonical_target"
    metadata["fixed_target_exactification"] = {
        "provenance": dict(provenance),
        "certificate": dict(fixed_target_certificate),
        "additional_basis_gauge_applied": False,
    }
    metadata["diagnostic_joint_result"] = diagnostic_record
    metadata_without_hash = dict(metadata)
    metadata_without_hash.pop("artifact_hash", None)
    metadata["artifact_hash"] = _joint_artifact_hash(metadata_without_hash, arrays)
    return JointExactificationResult(
        actions=target_actions,
        presentation=presentation,
        report=report,
        artifact_metadata=metadata,
        artifact_arrays=arrays,
    )


def load_joint_exactification_artifact(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> JointExactificationResult:
    """Load, hash-check, and re-certify an in-memory joint artifact."""

    root = dict(metadata)
    if root.get("version") != JOINT_BLOCK_REPRESENTATION_V1:
        raise JointExactificationError(
            f"unsupported joint artifact version {root.get('version')!r}"
        )
    expected_hash = str(root.pop("artifact_hash", ""))
    actual_hash = _joint_artifact_hash(root, arrays)
    if not expected_hash or actual_hash != expected_hash:
        raise JointExactificationError(
            "joint artifact hash mismatch: "
            f"expected={expected_hash!r}, actual={actual_hash!r}"
        )
    presentation = _presentation_from_metadata(root["presentation"])
    actions: dict[str, BlockRouteAction] = {}
    expected_keys: set[str] = set()
    for record in root["actions"]:
        keys = tuple(str(key) for key in record["route_array_keys"])
        expected_keys.update(keys)
        try:
            route_blocks = tuple(np.asarray(arrays[key]) for key in keys)
        except KeyError as error:
            raise JointExactificationError(
                f"joint artifact route array is missing: {error.args[0]}"
            ) from error
        action = BlockRouteAction(
            name=str(record["name"]),
            antiunitary=bool(record["antiunitary"]),
            fiber_permutation=tuple(int(value) for value in record["fiber_permutation"]),
            fiber_dimensions=tuple(int(value) for value in record["fiber_dimensions"]),
            route_blocks=route_blocks,
            fiber_indices=tuple(
                tuple(int(value) for value in group)
                for group in record["fiber_indices"]
            ),
            unitarity_certification_bound=float(
                record["unitarity_certification_bound"]
            ),
        )
        actions[action.name] = action
    if set(arrays) != expected_keys:
        raise JointExactificationError(
            "joint artifact arrays do not exactly match declared route keys"
        )
    validate_presentation_action_relations(actions, presentation)
    _, maximum = _relation_residual_summary(actions, presentation)
    bound = float(root["relation_certification_bound"])
    if maximum > bound:
        raise JointExactificationError(
            "loaded joint artifact relation certification failed: "
            f"residual={maximum:.6e}, bound={bound:.6e}"
        )
    restored_metadata = dict(root)
    restored_metadata["artifact_hash"] = expected_hash
    return JointExactificationResult(
        actions=actions,
        presentation=presentation,
        report=dict(root["report"]),
        artifact_metadata=restored_metadata,
        artifact_arrays=arrays,
    )


__all__ = [
    "ActionOrbit",
    "BlockRouteAction",
    "FreeOrbitReport",
    "JOINT_BLOCK_REPRESENTATION_V1",
    "JointExactificationError",
    "JointExactificationConfig",
    "JointExactificationResult",
    "MagneticGenerator",
    "MagneticPresentation",
    "MagneticRelation",
    "QuotientGroupElement",
    "SemilinearBlock",
    "StabilizedOrbitReport",
    "certify_joint_block_actions",
    "certify_fixed_target_joint_result",
    "compile_action_orbits",
    "compile_continuum_magnetic_presentation",
    "compile_quotient_group_elements",
    "compose_semilinear",
    "exactify_stabilized_orbit",
    "extract_block_route_action",
    "inverse_semilinear",
    "joint_exactify_block_actions",
    "load_joint_exactification_artifact",
    "magnetic_presentation_artifact",
    "magnetic_presentation_from_artifact",
    "materialize_block_route_action",
    "pack_skew_hermitian",
    "project_u1_relations",
    "reframe_joint_exactification_result",
    "semilinear_kappa",
    "synchronize_free_orbit",
    "unpack_skew_hermitian",
    "validate_presentation_action_relations",
]
