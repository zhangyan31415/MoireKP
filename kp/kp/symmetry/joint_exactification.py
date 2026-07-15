"""Joint exactification primitives for semilinear block-route actions.

The objects in this module store only the structurally nonzero route blocks.
They never infer numerical zeros from entry magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import expm


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
    nullity: int
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
        object.__setattr__(self, "nullity", int(self.nullity))


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
        rank_tolerance = float(
            singular_values[0] * max(jacobian.shape) * epsilon
        )
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        if rank == 0:
            raise JointExactificationError("stabilized-orbit Jacobian has zero rank")
        condition_number = float(singular_values[0] / singular_values[rank - 1])
        if condition_number > config.condition_limit:
            raise JointExactificationError(
                "stabilized-orbit Jacobian exceeds condition limit: "
                f"condition={condition_number:.6e}, limit={config.condition_limit:.6e}"
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
        nullity=len(variables) * len(basis) - rank,
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
            rank_tolerance = float(
                singular_values[0] * max(jacobian.shape) * epsilon
            )
            rank = int(np.count_nonzero(singular_values > rank_tolerance))
            smallest = float(singular_values[rank - 1])
            condition_number = float(singular_values[0] / smallest)
            if condition_number > config.condition_limit:
                raise JointExactificationError(
                    "free-orbit synchronization Jacobian exceeds condition limit: "
                    f"condition={condition_number:.6e}, "
                    f"limit={config.condition_limit:.6e}"
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


__all__ = [
    "ActionOrbit",
    "BlockRouteAction",
    "FreeOrbitReport",
    "JointExactificationError",
    "JointExactificationConfig",
    "MagneticGenerator",
    "MagneticPresentation",
    "MagneticRelation",
    "QuotientGroupElement",
    "SemilinearBlock",
    "StabilizedOrbitReport",
    "compile_action_orbits",
    "compile_continuum_magnetic_presentation",
    "compile_quotient_group_elements",
    "compose_semilinear",
    "exactify_stabilized_orbit",
    "extract_block_route_action",
    "inverse_semilinear",
    "materialize_block_route_action",
    "pack_skew_hermitian",
    "project_u1_relations",
    "semilinear_kappa",
    "synchronize_free_orbit",
    "unpack_skew_hermitian",
    "validate_presentation_action_relations",
]
