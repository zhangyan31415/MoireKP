"""Joint exactification primitives for semilinear block-route actions.

The objects in this module store only the structurally nonzero route blocks.
They never infer numerical zeros from entry magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Mapping, Sequence

import numpy as np


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


__all__ = [
    "ActionOrbit",
    "BlockRouteAction",
    "JointExactificationError",
    "MagneticGenerator",
    "MagneticPresentation",
    "MagneticRelation",
    "QuotientGroupElement",
    "SemilinearBlock",
    "compile_action_orbits",
    "compile_continuum_magnetic_presentation",
    "compile_quotient_group_elements",
    "compose_semilinear",
    "extract_block_route_action",
    "inverse_semilinear",
    "materialize_block_route_action",
    "semilinear_kappa",
    "validate_presentation_action_relations",
]
