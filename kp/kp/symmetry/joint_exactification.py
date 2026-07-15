"""Joint exactification primitives for semilinear block-route actions.

The objects in this module store only the structurally nonzero route blocks.
They never infer numerical zeros from entry magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Sequence

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


@dataclass(frozen=True)
class BlockRouteAction:
    """A block-monomial action represented by one U(d) block per source fiber."""

    name: str
    antiunitary: bool
    fiber_permutation: tuple[int, ...]
    fiber_dimensions: tuple[int, ...]
    route_blocks: tuple[np.ndarray, ...] = field(repr=False, compare=False)
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
        object.__setattr__(self, "unitarity_certification_bound", bound)


__all__ = [
    "BlockRouteAction",
    "JointExactificationError",
    "SemilinearBlock",
    "compose_semilinear",
    "inverse_semilinear",
    "semilinear_kappa",
]
