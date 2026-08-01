"""Experimental generator fixed-subspace primitives.

This module deliberately stops at target-independent real coefficient space.  It
does not consume Heff, a fit grid, band windows, or solver pivots, and it is not
wired into the public ``complete_linear_v2`` pipeline yet.

The intended integration point is after construction of the joint
Hermitian/adjoint coefficient vocabulary and before expansion into sparse
Hamiltonian response arrays.  A symmetry-invariant vocabulary is obtained from
the common fixed space of the finite-group *generators*, rather than by applying
the full Reynolds average to every seed and reducing the resulting dense set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg


REAL_LINEAR_ACTION_CONVENTION_V1 = "real_then_imag__linear_or_M_conjugate_v1"
DETERMINISTIC_BASIS_CONVENTION_V1 = (
    "projector_columns__pivoted_qr__first_significant_positive_v2"
)
FIXED_SUBSPACE_ALGORITHM_V1 = "stacked_generator_minus_identity__svd_v1"
FIXED_SUBSPACE_CERTIFICATION_POLICY_V1 = (
    "absolute_relative_threshold__100x_gray_zone__generator_residual_v1"
)
FIXED_SUBSPACE_GRAY_ZONE_FACTOR_V1 = 100.0


class NonInvariantComponentError(ValueError):
    """Raised when a proposed coefficient-space component is not invariant."""


class FixedSubspaceCertificationError(ValueError):
    """Raised when a computed fixed-space basis fails generator certification."""


@dataclass(frozen=True)
class RealGeneratorAction:
    """One real-linear group-generator action on a shared vocabulary."""

    name: str
    matrix: np.ndarray
    antiunitary: bool

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("generator name must be non-empty")
        raw = np.asarray(self.matrix)
        if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
            raise ValueError(
                f"real generator {self.name!r} contains complex entries; "
                "use real_linear_generator_from_complex"
            )
        matrix = np.asarray(np.real(raw), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"generator {self.name!r} must be a square matrix, got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"generator {self.name!r} contains non-finite values")
        matrix = np.array(matrix, dtype=np.float64, copy=True, order="C")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class InvariantComponent:
    """A proposed invariant direct-sum component in coefficient coordinates.

    ``label`` is diagnostic only.  Invariance is always checked from the actual
    generator matrices; labels such as nominal polynomial degree carry no proof.
    """

    label: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.indices)
        if not str(self.label):
            raise ValueError("component label must be non-empty")
        if not indices:
            raise ValueError(f"component {self.label!r} must not be empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"component {self.label!r} contains duplicate indices")
        object.__setattr__(self, "indices", tuple(sorted(indices)))


@dataclass(frozen=True)
class FixedSubspaceResult:
    """Deterministic orthonormal basis for a common generator fixed space."""

    basis: np.ndarray
    projector: np.ndarray
    rank: int
    singular_values: np.ndarray
    max_generator_residual: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected_shapes = {
            "basis": (int(np.asarray(self.projector).shape[0]), int(self.rank)),
            "projector": tuple(np.asarray(self.projector).shape),
        }
        if expected_shapes["projector"] != (
            expected_shapes["projector"][0],
            expected_shapes["projector"][0],
        ):
            raise ValueError("fixed-space projector must be square")
        for field_name in ("basis", "projector", "singular_values"):
            value = np.asarray(getattr(self, field_name), dtype=np.float64)
            if field_name == "basis" and value.shape != expected_shapes["basis"]:
                raise ValueError(
                    f"fixed-space basis has shape {value.shape}, expected "
                    f"{expected_shapes['basis']}"
                )
            value = np.array(value, dtype=np.float64, copy=True, order="C")
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)


def real_linear_generator_from_complex(
    name: str,
    matrix: np.ndarray,
    *,
    antiunitary: bool,
) -> RealGeneratorAction:
    """Compile ``z -> M z`` or ``z -> M conj(z)`` into ``[Re z; Im z]``.

    Antiunitarity is metadata after compilation: both cases are ordinary real
    matrices, which is essential when computing the real common fixed space.
    """

    complex_matrix = np.asarray(matrix, dtype=np.complex128)
    if (
        complex_matrix.ndim != 2
        or complex_matrix.shape[0] != complex_matrix.shape[1]
    ):
        raise ValueError(
            f"complex generator {name!r} must be square, got {complex_matrix.shape}"
        )
    real = np.real(complex_matrix)
    imag = np.imag(complex_matrix)
    if antiunitary:
        real_matrix = np.block([[real, imag], [imag, -real]])
    else:
        real_matrix = np.block([[real, -imag], [imag, real]])
    return RealGeneratorAction(name, real_matrix, antiunitary=bool(antiunitary))


def _validate_actions(
    actions: Sequence[RealGeneratorAction],
) -> tuple[RealGeneratorAction, ...]:
    ordered = tuple(sorted(actions, key=lambda action: action.name))
    if not ordered:
        raise ValueError("at least one generator action is required")
    names = [action.name for action in ordered]
    if len(set(names)) != len(names):
        raise ValueError("generator names must be unique")
    dimension = ordered[0].matrix.shape[0]
    for action in ordered[1:]:
        if action.matrix.shape != (dimension, dimension):
            raise ValueError(
                "all generator actions must use the same vocabulary dimension; "
                f"expected {(dimension, dimension)}, got {action.matrix.shape} "
                f"for {action.name!r}"
            )
    return ordered


def _validated_components(
    actions: Sequence[RealGeneratorAction],
    components: Sequence[InvariantComponent] | None,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[InvariantComponent, ...]:
    dimension = actions[0].matrix.shape[0]
    if components is None:
        return (InvariantComponent("complete_vocabulary", tuple(range(dimension))),)
    proposed = tuple(components)
    flattened = [index for component in proposed for index in component.indices]
    if sorted(flattened) != list(range(dimension)):
        raise ValueError(
            "invariant components must form an exact, non-overlapping partition "
            f"of coordinates 0..{dimension - 1}"
        )
    all_indices = np.arange(dimension, dtype=np.int64)
    for component in proposed:
        inside = np.asarray(component.indices, dtype=np.int64)
        outside = np.setdiff1d(all_indices, inside, assume_unique=True)
        for action in actions:
            leakage = action.matrix[np.ix_(outside, inside)]
            leakage_norm = float(np.linalg.norm(leakage, ord="fro"))
            action_scale = max(1.0, float(np.linalg.norm(action.matrix, ord="fro")))
            tolerance = absolute_tolerance + relative_tolerance * action_scale
            if leakage_norm > tolerance:
                raise NonInvariantComponentError(
                    f"component {component.label!r} is not invariant under generator "
                    f"{action.name!r}: outside-component leakage {leakage_norm:.6e} "
                    f"exceeds tolerance {tolerance:.6e}"
                )
    return proposed


def _svd_fixed_space(
    matrices: Sequence[np.ndarray],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dimension = matrices[0].shape[0]
    identity = np.eye(dimension, dtype=np.float64)
    constraints = np.vstack([matrix - identity for matrix in matrices])
    # The stacked constraint matrix always has at least ``dimension`` rows.
    # Economy SVD therefore retains the complete right singular-vector basis
    # while avoiding an unused O((n_generators * dimension)^2) left factor.
    _, singular_values, vh = np.linalg.svd(constraints, full_matrices=False)
    scale = max(1.0, float(singular_values[0]) if singular_values.size else 0.0)
    tolerance = absolute_tolerance + relative_tolerance * scale
    fixed = singular_values <= tolerance
    gray_upper = FIXED_SUBSPACE_GRAY_ZONE_FACTOR_V1 * tolerance
    gray = singular_values[(singular_values > tolerance) & (singular_values < gray_upper)]
    if gray.size:
        raise FixedSubspaceCertificationError(
            "fixed-subspace singular values enter the versioned gray zone; "
            "refusing to classify near-fixed directions: "
            f"values={[float(value) for value in gray]}, threshold={tolerance:.6e}, "
            f"gray_upper={gray_upper:.6e}"
        )
    fixed_values = singular_values[fixed]
    nonfixed_values = singular_values[~fixed]
    largest_fixed = float(np.max(fixed_values)) if fixed_values.size else 0.0
    smallest_nonfixed = float(np.min(nonfixed_values)) if nonfixed_values.size else None
    if smallest_nonfixed is None:
        spectral_gap_absolute = None
        spectral_gap_ratio = None
    else:
        spectral_gap_absolute = float(smallest_nonfixed - largest_fixed)
        spectral_gap_ratio = float(
            smallest_nonfixed / max(tolerance, largest_fixed, np.finfo(float).tiny)
        )
    artifact = {
        "singular_values": [float(value) for value in singular_values],
        "fixed_threshold": float(tolerance),
        "gray_zone_upper": float(gray_upper),
        "gray_zone_singular_values": [float(value) for value in gray],
        "fixed_rank": int(np.count_nonzero(fixed)),
        "largest_fixed_singular_value": float(largest_fixed),
        "smallest_nonfixed_singular_value": smallest_nonfixed,
        "spectral_gap_absolute": spectral_gap_absolute,
        "spectral_gap_ratio": spectral_gap_ratio,
    }
    return np.asarray(vh.T[:, fixed], dtype=np.float64), singular_values, artifact


def _deterministic_basis_from_projector(
    projector: np.ndarray,
    rank: int,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> np.ndarray:
    """Choose a deterministic orthonormal frame from a subspace projector."""

    dimension = projector.shape[0]
    if rank == 0:
        return np.zeros((dimension, 0), dtype=np.float64)
    threshold = absolute_tolerance + relative_tolerance * max(
        1.0, float(np.linalg.norm(projector, ord=2))
    )
    q_matrix, r_matrix, pivots = scipy.linalg.qr(
        np.asarray(projector, dtype=np.float64),
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(r_matrix))
    if diagonal.size < rank or float(diagonal[rank - 1]) <= threshold:
        raise RuntimeError(
            "failed to construct the certified number of deterministic fixed-space "
            f"vectors: expected {rank}, pivoted-QR diagonal={diagonal.tolist()}"
        )
    basis = np.asarray(q_matrix[:, :rank], dtype=np.float64).copy()
    for column_index in range(rank):
        column = basis[:, column_index]
        significant = np.flatnonzero(np.abs(column) > threshold)
        if significant.size and column[int(significant[0])] < 0.0:
            basis[:, column_index] = -column
    projector_residual = float(
        np.linalg.norm(
            projector - basis @ (basis.T @ projector),
            ord="fro",
        )
    )
    projector_bound = float(
        absolute_tolerance
        + relative_tolerance * max(1.0, float(np.linalg.norm(projector, ord="fro")))
        + np.finfo(float).eps * max(1, dimension) * max(1.0, float(np.linalg.norm(projector, ord="fro")))
    )
    if projector_residual > projector_bound:
        raise FixedSubspaceCertificationError(
            "pivoted-QR canonical frame does not span the certified projector: "
            f"residual {projector_residual:.6e} exceeds {projector_bound:.6e}; "
            f"pivots={pivots[:rank].tolist()}"
        )
    return basis


def solve_generator_fixed_subspace(
    generator_actions: Sequence[RealGeneratorAction],
    *,
    invariant_components: Sequence[InvariantComponent] | None = None,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-12,
    action_absolute_error_bound: float = 0.0,
) -> FixedSubspaceResult:
    """Solve the target-independent common fixed space of group generators.

    Optional components are an optimization only.  Every proposed component is
    certified against every actual generator before blockwise solution, so a
    nominal-degree partition that mixes under finite-center or adjoint actions is
    rejected rather than silently changing the fixed space.
    """

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0 or action_absolute_error_bound < 0.0:
        raise ValueError("fixed-subspace tolerances must be non-negative")
    actions = _validate_actions(generator_actions)
    components = _validated_components(
        actions,
        invariant_components,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    dimension = actions[0].matrix.shape[0]
    raw_blocks: list[np.ndarray] = []
    all_singular_values: list[np.ndarray] = []
    component_certifications: list[dict[str, Any]] = []
    component_ranks: dict[str, int] = {}
    for component in components:
        indices = np.asarray(component.indices, dtype=np.int64)
        restricted = [action.matrix[np.ix_(indices, indices)] for action in actions]
        local_basis, singular_values, certification = _svd_fixed_space(
            restricted,
            absolute_tolerance=absolute_tolerance + action_absolute_error_bound,
            relative_tolerance=relative_tolerance,
        )
        local_projector = local_basis @ local_basis.T
        local_projector = 0.5 * (local_projector + local_projector.T)
        local_basis = _deterministic_basis_from_projector(
            local_projector,
            int(local_basis.shape[1]),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        embedded = np.zeros((dimension, local_basis.shape[1]), dtype=np.float64)
        embedded[indices, :] = local_basis
        raw_blocks.append(embedded)
        all_singular_values.append(singular_values)
        component_ranks[component.label] = int(local_basis.shape[1])
        component_certifications.append(
            {"label": component.label, **certification}
        )
    raw_basis = (
        np.column_stack(raw_blocks)
        if raw_blocks
        else np.zeros((dimension, 0), dtype=np.float64)
    )
    rank = int(raw_basis.shape[1])
    # Every component has already received a deterministic local frame.  Since
    # the certified components have disjoint coordinate support, concatenation
    # is itself deterministic and orthonormal; a second global O(n^3) QR would
    # only repeat work and destroy the benefit of support-graph splitting.
    basis = raw_basis
    projector = basis @ basis.T
    residual_records: list[dict[str, float | str]] = []
    basis_scale = max(1.0, float(np.linalg.norm(basis, ord="fro")))
    for action in actions:
        residual = float(np.linalg.norm(action.matrix @ basis - basis, ord="fro"))
        action_scale = max(1.0, float(np.linalg.norm(action.matrix, ord=2)))
        backward_error = float(
            np.finfo(np.float64).eps * action_scale * basis_scale * max(1, dimension)
        )
        certification_bound = float(
            absolute_tolerance
            + action_absolute_error_bound
            + relative_tolerance * action_scale * basis_scale
            + backward_error
        )
        residual_records.append(
            {
                "generator": action.name,
                "residual": residual,
                "certification_bound": certification_bound,
                "backward_error_bound": backward_error,
            }
        )
        if residual > certification_bound:
            raise FixedSubspaceCertificationError(
                f"generator {action.name!r} residual {residual:.6e} exceeds "
                f"certification bound {certification_bound:.6e}"
            )
    singular_values = (
        np.concatenate(all_singular_values)
        if all_singular_values
        else np.zeros(0, dtype=np.float64)
    )
    metadata: dict[str, Any] = {
        "algorithm": FIXED_SUBSPACE_ALGORITHM_V1,
        "basis_convention": DETERMINISTIC_BASIS_CONVENTION_V1,
        "real_linear_action_convention": REAL_LINEAR_ACTION_CONVENTION_V1,
        "vocabulary_dimension": dimension,
        "generator_names": [action.name for action in actions],
        "antiunitary_parities": {
            action.name: bool(action.antiunitary) for action in actions
        },
        "absolute_tolerance": float(absolute_tolerance),
        "action_absolute_error_bound": float(action_absolute_error_bound),
        "relative_tolerance": float(relative_tolerance),
        "component_labels": [component.label for component in components],
        "component_ranks": component_ranks,
        "component_policy": "actual_generator_invariance_certified_v1",
        "fixed_subspace_certification": {
            "policy": FIXED_SUBSPACE_CERTIFICATION_POLICY_V1,
            "gray_zone_factor": FIXED_SUBSPACE_GRAY_ZONE_FACTOR_V1,
            "engineering_policy": True,
            "mathematical_constant": False,
            "components": component_certifications,
            "generator_residuals": residual_records,
        },
    }
    return FixedSubspaceResult(
        basis=basis,
        projector=projector,
        rank=rank,
        singular_values=singular_values,
        max_generator_residual=max(
            (float(record["residual"]) for record in residual_records),
            default=0.0,
        ),
        metadata=metadata,
    )


__all__ = [
    "DETERMINISTIC_BASIS_CONVENTION_V1",
    "FIXED_SUBSPACE_ALGORITHM_V1",
    "FIXED_SUBSPACE_CERTIFICATION_POLICY_V1",
    "FIXED_SUBSPACE_GRAY_ZONE_FACTOR_V1",
    "REAL_LINEAR_ACTION_CONVENTION_V1",
    "FixedSubspaceResult",
    "FixedSubspaceCertificationError",
    "InvariantComponent",
    "NonInvariantComponentError",
    "RealGeneratorAction",
    "real_linear_generator_from_complex",
    "solve_generator_fixed_subspace",
]
