"""Certified generator fixed-space compilation in global real coefficient space.

This experimental module contains no fit target or k-point sampling.  Its input is
an authored Hermitian vocabulary ``V`` and the image ``g(V)`` under each actual
finite-group generator, all represented in the same global polynomial
coefficient coordinates.  It certifies that the authored span is invariant,
constructs an orthonormal real representation of every generator, and solves
their common fixed space without applying a full-group Reynolds average to every
seed.

The dense Reynolds path remains an independent test oracle; it is not imported
or called here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
from scipy import sparse

from .response_basis_fixed_subspace import (
    InvariantComponent,
    RealGeneratorAction,
    solve_generator_fixed_subspace,
)


FIXED_VOCABULARY_COMPILER_V1 = (
    "global_real_coefficient_gram__generator_span_certification__fixed_space_v1"
)


class VocabularyRankCertificationError(ValueError):
    """Raised when the authored vocabulary does not have certified full rank."""


class GeneratorSpanClosureError(ValueError):
    """Raised when a generator image leaves the authored coefficient span."""


class GeneratorRepresentationError(ValueError):
    """Raised when a compiled generator is not orthogonal within its error bound."""


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


def _real_matrix(
    value: np.ndarray | sparse.spmatrix,
    *,
    name: str,
) -> np.ndarray | sparse.csc_matrix:
    raw = value
    if sparse.issparse(raw):
        matrix = sparse.csc_matrix(raw)
        if np.iscomplexobj(matrix.data) and np.any(np.imag(matrix.data) != 0.0):
            raise ValueError(f"{name} must use explicit global [Re; Im] coordinates")
        matrix = sparse.csc_matrix(np.real(matrix), dtype=np.float64)
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"{name} contains non-finite values")
        return matrix
    array = np.asarray(raw)
    if np.iscomplexobj(array) and np.any(np.imag(array) != 0.0):
        raise ValueError(f"{name} must use explicit global [Re; Im] coordinates")
    array = np.asarray(np.real(array), dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _dense_gram(left: np.ndarray | sparse.spmatrix, right: np.ndarray | sparse.spmatrix) -> np.ndarray:
    product = left.T @ right
    return np.asarray(product.toarray() if sparse.issparse(product) else product, dtype=np.float64)


def _left_multiply(
    matrix: np.ndarray | sparse.spmatrix,
    right: np.ndarray,
) -> np.ndarray:
    product = matrix @ right
    return np.asarray(product.toarray() if sparse.issparse(product) else product, dtype=np.float64)


def _certified_generator_support_components(
    actions: Sequence[RealGeneratorAction],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[InvariantComponent, ...], dict[str, Any]]:
    """Find and certify invariant components from actual generator support."""

    dimension = int(actions[0].matrix.shape[0])
    parent = list(range(dimension))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        low, high = sorted((left_root, right_root))
        parent[high] = low
        return True

    tolerances: dict[str, float] = {}
    for action in actions:
        action_scale = max(1.0, float(np.linalg.norm(action.matrix, ord="fro")))
        tolerance = float(absolute_tolerance + relative_tolerance * action_scale)
        tolerances[action.name] = tolerance
        rows, columns = np.nonzero(np.abs(action.matrix) > tolerance)
        for row, column in zip(rows, columns):
            union(int(row), int(column))

    # Tiny entries may be individually below the graph threshold while their
    # aggregate leakage is not.  Merge the largest leaking neighbor until every
    # proposed block passes the same Frobenius certification used by the solver.
    while True:
        components_by_root: dict[int, list[int]] = {}
        for index in range(dimension):
            components_by_root.setdefault(find(index), []).append(index)
        ordered_components = sorted(
            components_by_root.values(),
            key=lambda values: tuple(values),
        )
        root_by_index = {
            index: find(index)
            for index in range(dimension)
        }
        merged = False
        all_indices = np.arange(dimension, dtype=np.int64)
        for action in actions:
            tolerance = tolerances[action.name]
            for inside_values in ordered_components:
                inside = np.asarray(inside_values, dtype=np.int64)
                outside = np.setdiff1d(all_indices, inside, assume_unique=True)
                if outside.size == 0:
                    continue
                leakage = action.matrix[np.ix_(outside, inside)]
                if float(np.linalg.norm(leakage, ord="fro")) <= tolerance:
                    continue
                block_norms: list[tuple[float, int]] = []
                inside_root = root_by_index[int(inside[0])]
                for outside_values in ordered_components:
                    outside_root = root_by_index[int(outside_values[0])]
                    if outside_root == inside_root:
                        continue
                    block = action.matrix[
                        np.ix_(
                            np.asarray(outside_values, dtype=np.int64),
                            inside,
                        )
                    ]
                    block_norms.append(
                        (float(np.linalg.norm(block, ord="fro")), outside_root)
                    )
                _, neighbor_root = max(block_norms, key=lambda item: (item[0], -item[1]))
                union(inside_root, neighbor_root)
                merged = True
                break
            if merged:
                break
        if not merged:
            break

    final_by_root: dict[int, list[int]] = {}
    for index in range(dimension):
        final_by_root.setdefault(find(index), []).append(index)
    final_components = sorted(final_by_root.values(), key=lambda values: tuple(values))
    components = tuple(
        InvariantComponent(
            f"generator_support_component_{component_index}",
            tuple(values),
        )
        for component_index, values in enumerate(final_components)
    )
    max_leakage = 0.0
    all_indices = np.arange(dimension, dtype=np.int64)
    for action in actions:
        for values in final_components:
            inside = np.asarray(values, dtype=np.int64)
            outside = np.setdiff1d(all_indices, inside, assume_unique=True)
            if outside.size:
                max_leakage = max(
                    max_leakage,
                    float(
                        np.linalg.norm(
                            action.matrix[np.ix_(outside, inside)],
                            ord="fro",
                        )
                    ),
                )
    return components, {
        "policy": "actual_generator_support_graph_certified_v1",
        "component_count": int(len(components)),
        "component_sizes": tuple(len(values) for values in final_components),
        "generator_leakage_tolerances": tolerances,
        "max_certified_outside_component_leakage": float(max_leakage),
    }


@dataclass(frozen=True)
class FixedVocabularyCompilation:
    """A target-independent orthonormal basis of invariant responses."""

    fixed_vectors: np.ndarray | None
    fixed_vocabulary_coordinates: np.ndarray
    logical_projection_coordinates: np.ndarray
    rank: int
    logical_channel_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        fixed = (
            None
            if self.fixed_vectors is None
            else np.array(self.fixed_vectors, dtype=np.float64, copy=True, order="C")
        )
        vocabulary_coordinates = np.array(
            self.fixed_vocabulary_coordinates,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        mapping = np.array(
            self.logical_projection_coordinates,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        logical_ids = tuple(str(value) for value in self.logical_channel_ids)
        if fixed is not None and (fixed.ndim != 2 or fixed.shape[1] != int(self.rank)):
            raise ValueError("fixed response vectors have an inconsistent rank")
        if vocabulary_coordinates.shape != (len(logical_ids), int(self.rank)):
            raise ValueError("fixed vocabulary coordinates have an inconsistent shape")
        if mapping.shape != (int(self.rank), len(logical_ids)):
            raise ValueError("logical-to-fixed mapping has an inconsistent shape")
        if fixed is not None:
            fixed.setflags(write=False)
        vocabulary_coordinates.setflags(write=False)
        mapping.setflags(write=False)
        object.__setattr__(self, "fixed_vectors", fixed)
        object.__setattr__(
            self,
            "fixed_vocabulary_coordinates",
            vocabulary_coordinates,
        )
        object.__setattr__(self, "logical_projection_coordinates", mapping)
        object.__setattr__(self, "logical_channel_ids", logical_ids)
        object.__setattr__(self, "metadata", _immutable(self.metadata))


def _compile_generator_fixed_vocabulary_unsplit(
    vocabulary: np.ndarray | sparse.spmatrix,
    *,
    generator_images: Mapping[str, np.ndarray | sparse.spmatrix],
    antiunitary_parities: Mapping[str, bool],
    logical_channel_ids: Sequence[str],
    generator_absolute_error_bounds: Mapping[str, float] | None = None,
    generator_column_absolute_error_bounds: Mapping[str, Sequence[float]] | None = None,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-12,
    rank_gray_factor: float = 100.0,
    materialize_ambient_vectors: bool = True,
) -> FixedVocabularyCompilation:
    """Compile the common generator fixed space of an authored vocabulary.

    ``vocabulary[:, j]`` is logical Hermitian channel ``j`` in the fixed global
    two-dimensional polynomial coefficient basis.  ``generator_images[name]``
    must contain the corresponding columns after applying that actual generator.
    No nominal-degree partition is accepted or inferred.
    """

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("fixed-vocabulary tolerances must be non-negative")
    if rank_gray_factor <= 1.0:
        raise ValueError("rank_gray_factor must exceed one")
    matrix = _real_matrix(vocabulary, name="vocabulary")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vocabulary must be a non-empty two-dimensional matrix")
    logical_ids = tuple(str(value) for value in logical_channel_ids)
    if len(logical_ids) != matrix.shape[1] or len(set(logical_ids)) != len(logical_ids):
        raise ValueError("logical channel ids must uniquely label every vocabulary column")
    image_keys = {str(key) for key in generator_images}
    parity_keys = {str(key) for key in antiunitary_parities}
    if not image_keys or image_keys != parity_keys:
        raise ValueError("generator metadata keys must exactly match generator image keys")
    bounds = {
        str(key): float(value)
        for key, value in (generator_absolute_error_bounds or {}).items()
    }
    if not set(bounds).issubset(image_keys) or any(value < 0.0 for value in bounds.values()):
        raise ValueError("generator absolute error bounds must be non-negative and keyed by generator")
    column_bounds = {
        str(key): np.asarray(value, dtype=np.float64)
        for key, value in (generator_column_absolute_error_bounds or {}).items()
    }
    if not set(column_bounds).issubset(image_keys):
        raise ValueError("generator column error bounds must be keyed by generator")
    for name, values in column_bounds.items():
        if values.shape != (matrix.shape[1],) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"generator {name!r} column error bounds must be finite, non-negative, "
                f"and have shape {(matrix.shape[1],)}"
            )

    gram = _dense_gram(matrix, matrix)
    gram = 0.5 * (gram + gram.T)
    eigenvalues = scipy.linalg.eigvalsh(gram, check_finite=False)
    eigenvalues[eigenvalues < 0.0] = np.maximum(eigenvalues[eigenvalues < 0.0], 0.0)
    singular_values = np.sqrt(eigenvalues[::-1])
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    rank_backward_error = float(
        np.finfo(np.float64).eps * max(matrix.shape) * max(largest, 1.0)
    )
    rank_threshold = float(
        absolute_tolerance + relative_tolerance * max(largest, 1.0) + rank_backward_error
    )
    if smallest <= rank_gray_factor * rank_threshold:
        raise VocabularyRankCertificationError(
            "authored vocabulary rank is not certified before normalization: "
            f"smallest singular value {smallest:.6e}, gray-zone upper bound "
            f"{rank_gray_factor * rank_threshold:.6e}"
        )
    condition = float(largest / smallest)
    try:
        lower = scipy.linalg.cholesky(gram, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError as exc:
        raise VocabularyRankCertificationError(
            "authored vocabulary Gram matrix is not positive definite"
        ) from exc
    lower_inverse = scipy.linalg.solve_triangular(
        lower,
        np.eye(lower.shape[0], dtype=np.float64),
        lower=True,
        check_finite=False,
    )
    whitened_gram = lower_inverse @ gram @ lower_inverse.T
    orthogonality_residual = float(
        np.linalg.norm(whitened_gram - np.eye(matrix.shape[1]), ord="fro")
    )
    orthogonality_bound = float(
        absolute_tolerance
        + relative_tolerance * max(1.0, np.sqrt(float(matrix.shape[1])))
        + condition * np.finfo(np.float64).eps * max(matrix.shape)
    )
    if orthogonality_residual > orthogonality_bound:
        raise VocabularyRankCertificationError(
            "authored vocabulary whitening failed certification: "
            f"residual {orthogonality_residual:.6e}, bound {orthogonality_bound:.6e}"
        )

    actions: list[RealGeneratorAction] = []
    certification: list[dict[str, Any]] = []
    action_representation_error_bounds: list[float] = []
    image_grams: dict[str, np.ndarray] = {}
    image_cross_grams: dict[str, np.ndarray] = {}
    for name in sorted(image_keys):
        image = _real_matrix(generator_images[name], name=f"generator image {name!r}")
        if image.shape != matrix.shape:
            raise ValueError(
                f"generator image {name!r} has shape {image.shape}, expected {matrix.shape}"
            )
        image_gram = _dense_gram(image, image)
        image_gram = 0.5 * (image_gram + image_gram.T)
        cross_gram = _dense_gram(matrix, image)
        image_grams[name] = image_gram
        image_cross_grams[name] = cross_gram
        whitened_cross = scipy.linalg.solve_triangular(
            lower,
            cross_gram,
            lower=True,
            check_finite=False,
        )
        representation = scipy.linalg.solve_triangular(
            lower,
            whitened_cross.T,
            lower=True,
            check_finite=False,
        ).T
        whitened_image_gram = lower_inverse @ image_gram @ lower_inverse.T
        image_norm_squared = max(0.0, float(np.trace(whitened_image_gram)))
        image_norm = float(np.sqrt(image_norm_squared))
        projection_coordinates = scipy.linalg.cho_solve(
            (lower, True),
            cross_gram,
            check_finite=False,
        )
        raw_column_residuals: list[float] = []
        for column_index in range(matrix.shape[1]):
            if sparse.issparse(matrix) and sparse.issparse(image):
                coordinate_column = sparse.csc_matrix(
                    projection_coordinates[:, column_index].reshape(-1, 1)
                )
                residual_column = (
                    image.getcol(column_index) - matrix @ coordinate_column
                ).tocsc()
                residual_column.eliminate_zeros()
                residual_norm = float(np.linalg.norm(residual_column.data))
            else:
                image_column = np.asarray(image[:, column_index], dtype=np.float64).reshape(-1)
                projected_column = _left_multiply(
                    matrix,
                    projection_coordinates[:, column_index].reshape(-1, 1),
                ).reshape(-1)
                residual_norm = float(np.linalg.norm(image_column - projected_column))
            raw_column_residuals.append(residual_norm)
        per_column_input = column_bounds.get(
            name,
            np.zeros(matrix.shape[1], dtype=np.float64),
        )
        aggregate_raw_bound = float(np.linalg.norm(per_column_input))
        aggregate_input_bound = max(float(bounds.get(name, 0.0)), aggregate_raw_bound)
        whitening_gain = float(np.linalg.norm(lower_inverse.T, ord=2))
        outside_norm = float(
            whitening_gain * np.linalg.norm(np.asarray(raw_column_residuals))
        )
        input_bound = float(whitening_gain * aggregate_input_bound)
        condition_roundoff = float(
            condition
            * np.finfo(np.float64).eps
            * max(matrix.shape)
            * max(image_norm, 1.0)
        )
        span_bound = float(
            input_bound
            + absolute_tolerance
            + relative_tolerance * max(image_norm, 1.0)
            + condition_roundoff
        )
        if outside_norm > span_bound:
            raise GeneratorSpanClosureError(
                f"generator {name!r} image leaves the authored span: outside authored span "
                f"residual {outside_norm:.6e} exceeds {span_bound:.6e}"
            )
        column_records: list[dict[str, Any]] = []
        for column_index, channel_id in enumerate(logical_ids):
            column_norm_squared = max(0.0, float(image_gram[column_index, column_index]))
            column_norm = float(np.sqrt(column_norm_squared))
            column_residual = raw_column_residuals[column_index]
            column_roundoff = float(
                condition
                * np.finfo(np.float64).eps
                * max(matrix.shape)
                * max(column_norm, 1.0)
            )
            column_bound = float(
                per_column_input[column_index]
                + absolute_tolerance
                + relative_tolerance * max(column_norm, 1.0)
                + column_roundoff
            )
            if column_residual > column_bound:
                raise GeneratorSpanClosureError(
                    f"generator {name!r} column {channel_id!r} leaves the authored span: "
                    f"outside authored span residual {column_residual:.6e} exceeds "
                    f"its own bound {column_bound:.6e}"
                )
            column_records.append(
                {
                    "logical_channel_id": channel_id,
                    "input_absolute_error_bound": float(per_column_input[column_index]),
                    "outside_span_residual": column_residual,
                    "span_certification_bound": column_bound,
                    "condition_roundoff_bound": column_roundoff,
                }
            )
        representation_orthogonality = float(
            np.linalg.norm(
                representation.T @ representation - np.eye(representation.shape[0]),
                ord="fro",
            )
        )
        representation_bound = float(
            span_bound / max(image_norm, 1.0)
            + absolute_tolerance
            + relative_tolerance * max(1.0, np.sqrt(float(representation.shape[0])))
            + condition_roundoff / max(image_norm, 1.0)
        )
        if representation_orthogonality > representation_bound:
            raise GeneratorRepresentationError(
                f"generator {name!r} real representation is not orthogonal: residual "
                f"{representation_orthogonality:.6e} exceeds {representation_bound:.6e}"
            )
        actions.append(
            RealGeneratorAction(
                name,
                representation,
                antiunitary=bool(antiunitary_parities[name]),
            )
        )
        action_representation_error_bounds.append(representation_bound)
        certification.append(
            {
                "generator": name,
                "antiunitary": bool(antiunitary_parities[name]),
                "input_absolute_error_bound": input_bound,
                "outside_span_residual": outside_norm,
                "span_certification_bound": span_bound,
                "condition_roundoff_bound": condition_roundoff,
                "representation_orthogonality_residual": representation_orthogonality,
                "representation_orthogonality_bound": representation_bound,
                "whitening_gain": whitening_gain,
                "columns": tuple(column_records),
            }
        )

    invariant_components, component_certification = (
        _certified_generator_support_components(
            actions,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    )
    fixed = solve_generator_fixed_subspace(
        actions,
        invariant_components=invariant_components,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        action_absolute_error_bound=max(
            action_representation_error_bounds,
            default=0.0,
        ),
    )
    fixed_vocabulary_coordinates = lower_inverse.T @ fixed.basis
    logical_projection_coordinates = fixed.basis.T @ lower.T
    fixed_orthogonality_residual = float(
        np.linalg.norm(
            fixed_vocabulary_coordinates.T
            @ gram
            @ fixed_vocabulary_coordinates
            - np.eye(fixed.rank),
            ord="fro",
        )
    )
    if fixed_orthogonality_residual > orthogonality_bound:
        raise VocabularyRankCertificationError(
            "fixed response basis orthogonality failed certification: "
            f"residual {fixed_orthogonality_residual:.6e}, bound {orthogonality_bound:.6e}"
        )
    fixed_vectors = (
        _left_multiply(matrix, fixed_vocabulary_coordinates)
        if bool(materialize_ambient_vectors)
        else None
    )
    if sparse.issparse(matrix):
        sparse_fixed_coordinates = sparse.csc_matrix(fixed_vocabulary_coordinates)
        sparse_fixed_ambient = (matrix @ sparse_fixed_coordinates).tocsc()
    else:
        sparse_fixed_coordinates = None
        sparse_fixed_ambient = None
    fixed_action_certification: list[dict[str, float | str]] = []
    for name in sorted(image_grams):
        image = _real_matrix(generator_images[name], name=f"generator image {name!r}")
        if sparse.issparse(image):
            assert sparse_fixed_coordinates is not None
            assert sparse_fixed_ambient is not None
            transformed_fixed = (image @ sparse_fixed_coordinates).tocsc()
            residual_fixed = (transformed_fixed - sparse_fixed_ambient).tocsc()
            residual_fixed.eliminate_zeros()
            residual_product = residual_fixed.T @ residual_fixed
            transformed_product = transformed_fixed.T @ transformed_fixed
            residual_gram = np.asarray(residual_product.toarray(), dtype=np.float64)
            transformed_gram = np.asarray(
                transformed_product.toarray(),
                dtype=np.float64,
            )
        else:
            transformed_fixed = _left_multiply(
                image,
                fixed_vocabulary_coordinates,
            )
            fixed_ambient = _left_multiply(
                matrix,
                fixed_vocabulary_coordinates,
            )
            residual_fixed = transformed_fixed - fixed_ambient
            residual_gram = residual_fixed.T @ residual_fixed
            transformed_gram = transformed_fixed.T @ transformed_fixed
        residual_gram = 0.5 * (residual_gram + residual_gram.T)
        residual_eigenvalues = scipy.linalg.eigvalsh(
            residual_gram,
            check_finite=False,
        )
        residual_operator_norm = float(
            np.sqrt(max(0.0, float(residual_eigenvalues[-1])))
            if residual_eigenvalues.size
            else 0.0
        )
        transformed_eigenvalues = scipy.linalg.eigvalsh(
            0.5 * (transformed_gram + transformed_gram.T),
            check_finite=False,
        )
        transformed_norm = float(
            np.sqrt(max(0.0, float(transformed_eigenvalues[-1])))
            if transformed_eigenvalues.size
            else 0.0
        )
        subtraction_roundoff = float(
            np.finfo(np.float64).eps
            * max(1, fixed.rank)
            * (transformed_norm + 1.0)
        )
        fixed_action_certification.append(
            {
                "generator": name,
                "residual_operator_norm": residual_operator_norm,
                "certified_operator_bound": residual_operator_norm
                + subtraction_roundoff,
                "subtraction_roundoff_bound": subtraction_roundoff,
            }
        )
    metadata = {
        "compiler": FIXED_VOCABULARY_COMPILER_V1,
        "target_independent": True,
        "sample_grid_used": False,
        "logical_channel_count": int(matrix.shape[1]),
        "fixed_rank": int(fixed.rank),
        "logical_channel_ids": logical_ids,
        "antiunitary_parities": {
            name: bool(antiunitary_parities[name]) for name in sorted(image_keys)
        },
        "vocabulary_singular_values": tuple(float(value) for value in singular_values),
        "vocabulary_rank_threshold": rank_threshold,
        "rank_gray_factor": float(rank_gray_factor),
        "vocabulary_condition_estimate": condition,
        "vocabulary_orthogonality_residual": orthogonality_residual,
        "vocabulary_orthogonality_bound": orthogonality_bound,
        "fixed_orthogonality_residual": fixed_orthogonality_residual,
        "generator_certification": tuple(certification),
        "fixed_physical_action_certification": tuple(fixed_action_certification),
        "fixed_subspace": dict(fixed.metadata),
        "invariant_component_certification": component_certification,
        "ambient_fixed_vectors_materialized": bool(materialize_ambient_vectors),
        "block_rule": "complete_joint_group_adjoint_vocabulary__no_nominal_degree_blocks",
    }
    return FixedVocabularyCompilation(
        fixed_vectors=fixed_vectors,
        fixed_vocabulary_coordinates=fixed_vocabulary_coordinates,
        logical_projection_coordinates=logical_projection_coordinates,
        rank=int(fixed.rank),
        logical_channel_ids=logical_ids,
        metadata=metadata,
    )


def _exact_joint_direct_sum_components(
    vocabulary: np.ndarray | sparse.spmatrix,
    generator_images: Mapping[str, np.ndarray | sparse.spmatrix],
) -> tuple[tuple[int, ...], ...]:
    """Return conservative exact support-connected direct summands.

    Two vocabulary columns are joined whenever their exact coefficient support
    overlaps.  A vocabulary column and the logical column of ``g(V)`` are joined
    by the same rule.  Disjoint returned components therefore have exactly
    disjoint ambient-coordinate support and no generator image has support in a
    different component.  This is a sufficient structural direct-sum proof; it
    deliberately ignores numerical cancellation that could reveal a finer
    split.

    The graph is built directly from CSC column support.  Forming global
    ``V.T @ V`` and ``V.T @ g(V)`` matrices here is both unnecessary and very
    expensive for high polynomial orders because the ambient coefficient space
    is large.  No tolerance or nominal-degree partition is used.
    """

    dimension = int(vocabulary.shape[1])
    parent = list(range(dimension))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    def nonzero_rows_by_column(
        matrix: np.ndarray | sparse.spmatrix,
    ) -> tuple[np.ndarray, ...]:
        if sparse.issparse(matrix):
            csc = sparse.csc_matrix(matrix, copy=False)
            columns: list[np.ndarray] = []
            for column in range(csc.shape[1]):
                start = int(csc.indptr[column])
                stop = int(csc.indptr[column + 1])
                values = csc.data[start:stop]
                rows = csc.indices[start:stop]
                columns.append(
                    np.asarray(rows[values != 0.0], dtype=np.int64)
                )
            return tuple(columns)
        array = np.asarray(matrix)
        return tuple(
            np.flatnonzero(array[:, column] != 0.0).astype(np.int64, copy=False)
            for column in range(array.shape[1])
        )

    vocabulary_rows = nonzero_rows_by_column(vocabulary)
    first_vocabulary_column_by_row: dict[int, int] = {}
    for column, rows in enumerate(vocabulary_rows):
        for row_value in rows:
            row = int(row_value)
            first = first_vocabulary_column_by_row.setdefault(row, column)
            union(first, column)

    for name in sorted(generator_images):
        for image_column, rows in enumerate(
            nonzero_rows_by_column(generator_images[name])
        ):
            for row_value in rows:
                vocabulary_column = first_vocabulary_column_by_row.get(int(row_value))
                if vocabulary_column is not None:
                    union(vocabulary_column, image_column)

    by_root: dict[int, list[int]] = {}
    for index in range(dimension):
        by_root.setdefault(find(index), []).append(index)
    return tuple(
        tuple(indices)
        for indices in sorted(by_root.values(), key=lambda values: tuple(values))
    )


def _combine_direct_sum_compilations(
    parts: Sequence[tuple[tuple[int, ...], FixedVocabularyCompilation]],
    *,
    ambient_dimension: int,
    logical_channel_ids: tuple[str, ...],
    antiunitary_parities: Mapping[str, bool],
    materialize_ambient_vectors: bool,
) -> FixedVocabularyCompilation:
    total_rank = int(sum(part.rank for _, part in parts))
    vocabulary_coordinates = np.zeros(
        (len(logical_channel_ids), total_rank), dtype=np.float64
    )
    logical_projection = np.zeros(
        (total_rank, len(logical_channel_ids)), dtype=np.float64
    )
    fixed_blocks: list[np.ndarray] = []
    rank_offset = 0
    for indices, part in parts:
        next_offset = rank_offset + int(part.rank)
        rows = np.asarray(indices, dtype=np.int64)
        vocabulary_coordinates[rows, rank_offset:next_offset] = (
            part.fixed_vocabulary_coordinates
        )
        logical_projection[rank_offset:next_offset, rows] = (
            part.logical_projection_coordinates
        )
        if materialize_ambient_vectors:
            if part.fixed_vectors is None:
                raise RuntimeError("direct-sum part omitted requested ambient fixed vectors")
            fixed_blocks.append(np.asarray(part.fixed_vectors, dtype=np.float64))
        rank_offset = next_offset
    fixed_vectors = (
        np.column_stack(fixed_blocks)
        if materialize_ambient_vectors and fixed_blocks
        else (
            np.zeros((ambient_dimension, 0), dtype=np.float64)
            if materialize_ambient_vectors
            else None
        )
    )

    singular_values = sorted(
        (
            float(value)
            for _, part in parts
            for value in part.metadata["vocabulary_singular_values"]
        ),
        reverse=True,
    )
    smallest = min(singular_values) if singular_values else 1.0
    largest = max(singular_values) if singular_values else 1.0
    generator_certification: list[dict[str, Any]] = []
    fixed_action_certification: list[dict[str, Any]] = []
    for name in sorted(antiunitary_parities):
        generator_records = [
            record
            for _, part in parts
            for record in part.metadata["generator_certification"]
            if str(record["generator"]) == name
        ]
        fixed_records = [
            record
            for _, part in parts
            for record in part.metadata["fixed_physical_action_certification"]
            if str(record["generator"]) == name
        ]
        generator_certification.append(
            {
                "generator": name,
                "antiunitary": bool(antiunitary_parities[name]),
                "input_absolute_error_bound": float(
                    sum(float(record["input_absolute_error_bound"]) for record in generator_records)
                ),
                "outside_span_residual": float(
                    np.sqrt(
                        sum(float(record["outside_span_residual"]) ** 2 for record in generator_records)
                    )
                ),
                "span_certification_bound": float(
                    sum(float(record["span_certification_bound"]) for record in generator_records)
                ),
                "condition_roundoff_bound": float(
                    sum(float(record["condition_roundoff_bound"]) for record in generator_records)
                ),
                "representation_orthogonality_residual": float(
                    np.sqrt(
                        sum(
                            float(record["representation_orthogonality_residual"]) ** 2
                            for record in generator_records
                        )
                    )
                ),
                "representation_orthogonality_bound": float(
                    sum(
                        float(record["representation_orthogonality_bound"])
                        for record in generator_records
                    )
                ),
                "whitening_gain": float(
                    max((float(record["whitening_gain"]) for record in generator_records), default=0.0)
                ),
                "columns": tuple(
                    column
                    for record in generator_records
                    for column in record["columns"]
                ),
            }
        )
        fixed_action_certification.append(
            {
                "generator": name,
                "residual_operator_norm": float(
                    max((float(record["residual_operator_norm"]) for record in fixed_records), default=0.0)
                ),
                "certified_operator_bound": float(
                    max((float(record["certified_operator_bound"]) for record in fixed_records), default=0.0)
                ),
                "subtraction_roundoff_bound": float(
                    max((float(record["subtraction_roundoff_bound"]) for record in fixed_records), default=0.0)
                ),
            }
        )

    component_sizes = tuple(len(indices) for indices, _ in parts)
    metadata = {
        "compiler": FIXED_VOCABULARY_COMPILER_V1,
        "target_independent": True,
        "sample_grid_used": False,
        "logical_channel_count": int(len(logical_channel_ids)),
        "fixed_rank": total_rank,
        "logical_channel_ids": logical_channel_ids,
        "antiunitary_parities": {
            name: bool(antiunitary_parities[name])
            for name in sorted(antiunitary_parities)
        },
        "vocabulary_singular_values": tuple(singular_values),
        "vocabulary_rank_threshold": float(
            max(
                float(part.metadata["vocabulary_rank_threshold"])
                for _, part in parts
            )
        ),
        "rank_gray_factor": float(parts[0][1].metadata["rank_gray_factor"]),
        "vocabulary_condition_estimate": float(largest / smallest),
        "vocabulary_orthogonality_residual": float(
            np.sqrt(
                sum(
                    float(part.metadata["vocabulary_orthogonality_residual"]) ** 2
                    for _, part in parts
                )
            )
        ),
        "vocabulary_orthogonality_bound": float(
            sum(float(part.metadata["vocabulary_orthogonality_bound"]) for _, part in parts)
        ),
        "fixed_orthogonality_residual": float(
            np.sqrt(
                sum(
                    float(part.metadata["fixed_orthogonality_residual"]) ** 2
                    for _, part in parts
                )
            )
        ),
        "generator_certification": tuple(generator_certification),
        "fixed_physical_action_certification": tuple(fixed_action_certification),
        "fixed_subspace": {
            "component_policy": "actual_generator_invariance_certified_v1",
            "direct_sum_part_count": int(len(parts)),
            "direct_sum_part_ranks": tuple(int(part.rank) for _, part in parts),
        },
        "invariant_component_certification": {
            "policy": "actual_generator_support_graph_certified_v1",
            "component_count": int(len(parts)),
            "component_sizes": component_sizes,
            "generator_leakage_tolerances": {},
            "max_certified_outside_component_leakage": 0.0,
        },
        "early_direct_sum_certification": {
            "policy": "exact_coefficient_support_overlap_v1",
            "component_count": int(len(parts)),
            "component_sizes": component_sizes,
            "exact_zero_required": True,
            "nominal_degree_partition_used": False,
        },
        "ambient_fixed_vectors_materialized": bool(materialize_ambient_vectors),
        "block_rule": "exact_coefficient_support_overlap__then_actual_generator_support",
    }
    return FixedVocabularyCompilation(
        fixed_vectors=fixed_vectors,
        fixed_vocabulary_coordinates=vocabulary_coordinates,
        logical_projection_coordinates=logical_projection,
        rank=total_rank,
        logical_channel_ids=logical_channel_ids,
        metadata=metadata,
    )


def compile_generator_fixed_vocabulary(
    vocabulary: np.ndarray | sparse.spmatrix,
    *,
    generator_images: Mapping[str, np.ndarray | sparse.spmatrix],
    antiunitary_parities: Mapping[str, bool],
    logical_channel_ids: Sequence[str],
    generator_absolute_error_bounds: Mapping[str, float] | None = None,
    generator_column_absolute_error_bounds: Mapping[str, Sequence[float]] | None = None,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-12,
    rank_gray_factor: float = 100.0,
    materialize_ambient_vectors: bool = True,
) -> FixedVocabularyCompilation:
    """Compile exact joint direct summands before any dense rank factorization."""

    matrix = _real_matrix(vocabulary, name="vocabulary")
    logical_ids = tuple(str(value) for value in logical_channel_ids)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vocabulary must be a non-empty two-dimensional matrix")
    if len(logical_ids) != matrix.shape[1] or len(set(logical_ids)) != len(logical_ids):
        raise ValueError("logical channel ids must uniquely label every vocabulary column")
    image_keys = {str(key) for key in generator_images}
    parity_keys = {str(key) for key in antiunitary_parities}
    if not image_keys or image_keys != parity_keys:
        raise ValueError("generator metadata keys must exactly match generator image keys")
    images = {
        str(name): _real_matrix(value, name=f"generator image {name!r}")
        for name, value in generator_images.items()
    }
    for name, image in images.items():
        if image.shape != matrix.shape:
            raise ValueError(
                f"generator image {name!r} has shape {image.shape}, expected {matrix.shape}"
            )
    components = _exact_joint_direct_sum_components(matrix, images)
    if len(components) <= 1:
        return _compile_generator_fixed_vocabulary_unsplit(
            matrix,
            generator_images=images,
            antiunitary_parities=antiunitary_parities,
            logical_channel_ids=logical_ids,
            generator_absolute_error_bounds=generator_absolute_error_bounds,
            generator_column_absolute_error_bounds=generator_column_absolute_error_bounds,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            rank_gray_factor=rank_gray_factor,
            materialize_ambient_vectors=materialize_ambient_vectors,
        )

    aggregate_bounds = {
        str(name): float(value)
        for name, value in (generator_absolute_error_bounds or {}).items()
    }
    column_bounds = {
        str(name): np.asarray(value, dtype=np.float64)
        for name, value in (generator_column_absolute_error_bounds or {}).items()
    }
    parts: list[tuple[tuple[int, ...], FixedVocabularyCompilation]] = []
    for indices in components:
        columns = np.asarray(indices, dtype=np.int64)
        local_column_bounds = {
            name: values[columns]
            for name, values in column_bounds.items()
        }
        local_aggregate_bounds = {
            name: (
                float(np.linalg.norm(local_column_bounds[name]))
                if name in local_column_bounds
                else float(value)
            )
            for name, value in aggregate_bounds.items()
        }
        part = _compile_generator_fixed_vocabulary_unsplit(
            matrix[:, columns],
            generator_images={name: image[:, columns] for name, image in images.items()},
            antiunitary_parities=antiunitary_parities,
            logical_channel_ids=tuple(logical_ids[index] for index in indices),
            generator_absolute_error_bounds=local_aggregate_bounds,
            generator_column_absolute_error_bounds=local_column_bounds,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            rank_gray_factor=rank_gray_factor,
            materialize_ambient_vectors=materialize_ambient_vectors,
        )
        parts.append((indices, part))
    return _combine_direct_sum_compilations(
        parts,
        ambient_dimension=int(matrix.shape[0]),
        logical_channel_ids=logical_ids,
        antiunitary_parities=antiunitary_parities,
        materialize_ambient_vectors=materialize_ambient_vectors,
    )


def compile_generator_fixed_vocabulary_from_actions(
    vocabulary: np.ndarray | sparse.spmatrix,
    *,
    generator_actions: Mapping[str, np.ndarray | sparse.spmatrix],
    antiunitary_parities: Mapping[str, bool],
    logical_channel_ids: Sequence[str],
    generator_column_absolute_error_bounds: Mapping[str, Sequence[float]] | None = None,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-12,
    rank_gray_factor: float = 100.0,
    materialize_ambient_vectors: bool = True,
) -> FixedVocabularyCompilation:
    """Compile the fixed space directly from ``g(V) = V A_g`` actions.

    The common kernel of ``A_g - I`` is independent of the non-orthogonal
    vocabulary metric.  We therefore solve that kernel first and use the
    physical coefficient-space Gram matrix only to normalize the much smaller
    fixed space.  No fit target or k-point grid enters this construction.
    """

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("fixed-vocabulary tolerances must be non-negative")
    if rank_gray_factor <= 1.0:
        raise ValueError("rank_gray_factor must exceed one")
    matrix = _real_matrix(vocabulary, name="vocabulary")
    logical_ids = tuple(str(value) for value in logical_channel_ids)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vocabulary must be a non-empty two-dimensional matrix")
    if len(logical_ids) != matrix.shape[1] or len(set(logical_ids)) != len(logical_ids):
        raise ValueError("logical channel ids must uniquely label every vocabulary column")
    action_keys = {str(key) for key in generator_actions}
    if not action_keys or action_keys != {str(key) for key in antiunitary_parities}:
        raise ValueError("generator actions and antiunitary parities must have identical keys")
    actions = {
        str(name): sparse.csc_matrix(
            _real_matrix(value, name=f"generator action {name!r}")
        )
        for name, value in generator_actions.items()
    }
    for name, action in actions.items():
        if action.shape != (matrix.shape[1], matrix.shape[1]):
            raise ValueError(
                f"generator action {name!r} has shape {action.shape}, expected "
                f"{(matrix.shape[1], matrix.shape[1])}"
            )
    column_bounds = {
        str(name): np.asarray(values, dtype=np.float64)
        for name, values in (generator_column_absolute_error_bounds or {}).items()
    }
    if not set(column_bounds).issubset(action_keys):
        raise ValueError("generator column bounds must be keyed by generator")
    for name, values in column_bounds.items():
        if values.shape != (matrix.shape[1],) or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"generator {name!r} column bounds must be finite, non-negative, "
                f"and have shape {(matrix.shape[1],)}"
            )

    images = {
        name: (matrix @ action).tocsc() if sparse.issparse(matrix) else matrix @ action.toarray()
        for name, action in actions.items()
    }
    components = _exact_joint_direct_sum_components(matrix, images)
    fixed_coordinate_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    fixed_ambient_blocks: list[np.ndarray] = []
    singular_values_all: list[float] = []
    rank_thresholds: list[float] = []
    component_records: list[dict[str, Any]] = []
    fixed_action_residuals: dict[str, list[float]] = {
        name: [] for name in sorted(actions)
    }
    fixed_action_bounds: dict[str, list[float]] = {
        name: [] for name in sorted(actions)
    }
    total_rank = 0
    fixed_orthogonality_squared = 0.0

    for component_index, indices in enumerate(components):
        columns = np.asarray(indices, dtype=np.int64)
        local_vocabulary = matrix[:, columns]
        raw_gram = _dense_gram(local_vocabulary, local_vocabulary)
        raw_gram = 0.5 * (raw_gram + raw_gram.T)
        column_scales = np.sqrt(np.maximum(np.diag(raw_gram), 0.0))
        if np.any(~np.isfinite(column_scales)) or np.any(column_scales <= 0.0):
            raise VocabularyRankCertificationError(
                "authored vocabulary contains a non-finite or zero-norm column "
                f"in component {component_index}"
            )
        inverse_column_scales = 1.0 / column_scales
        gram = (
            inverse_column_scales[:, None]
            * raw_gram
            * inverse_column_scales[None, :]
        )
        gram = 0.5 * (gram + gram.T)
        eigenvalues = scipy.linalg.eigvalsh(gram, check_finite=False)
        eigenvalues[eigenvalues < 0.0] = np.maximum(eigenvalues[eigenvalues < 0.0], 0.0)
        singular_values = np.sqrt(eigenvalues[::-1])
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        rank_backward_error = float(
            np.finfo(np.float64).eps
            * max(local_vocabulary.shape)
            * max(largest, 1.0)
        )
        rank_threshold = float(
            absolute_tolerance
            + relative_tolerance * max(largest, 1.0)
            + rank_backward_error
        )
        if smallest <= rank_gray_factor * rank_threshold:
            raise VocabularyRankCertificationError(
                "authored vocabulary rank is not certified before normalization: "
                f"component={component_index}, smallest={smallest:.6e}, "
                f"gray-zone upper={rank_gray_factor * rank_threshold:.6e}"
            )
        singular_values_all.extend(float(value) for value in singular_values)
        rank_thresholds.append(rank_threshold)

        lower = scipy.linalg.cholesky(
            gram,
            lower=True,
            check_finite=False,
        )
        inverse_lower_transpose = scipy.linalg.solve_triangular(
            lower.T,
            np.eye(lower.shape[0], dtype=np.float64),
            lower=False,
            check_finite=False,
        )
        # Ambient column-error bounds enter the orthonormal physical frame
        # through the same right-side metric whitening as the vocabulary.
        whitening_gain = float(np.linalg.norm(inverse_lower_transpose, ord=2))
        local_action_objects = []
        physical_action_representations: dict[str, np.ndarray] = {}
        local_action_error_bound = 0.0
        for name in sorted(actions):
            action = actions[name]
            outside = np.setdiff1d(
                np.arange(matrix.shape[1], dtype=np.int64),
                columns,
                assume_unique=True,
            )
            if outside.size and action[outside, :][:, columns].nnz:
                raise GeneratorSpanClosureError(
                    f"direct action component {component_index} is not invariant under {name!r}"
                )
            raw_local_action = np.asarray(
                action[columns, :][:, columns].toarray(),
                dtype=np.float64,
            )
            local_action = (
                column_scales[:, None]
                * raw_local_action
                * inverse_column_scales[None, :]
            )
            physical_representation = (
                lower.T @ local_action @ inverse_lower_transpose
            )
            physical_action_representations[name] = physical_representation
            local_action_objects.append(
                RealGeneratorAction(
                    name,
                    physical_representation,
                    antiunitary=bool(antiunitary_parities[name]),
                )
            )
            if name in column_bounds:
                local_action_error_bound = max(
                    local_action_error_bound,
                    float(
                        np.linalg.norm(
                            column_bounds[name][columns]
                            * inverse_column_scales
                        )
                        * whitening_gain
                    ),
                )
        fixed = solve_generator_fixed_subspace(
            local_action_objects,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            action_absolute_error_bound=local_action_error_bound,
        )
        physical_fixed = np.asarray(fixed.basis, dtype=np.float64)
        scaled_fixed = inverse_lower_transpose @ physical_fixed
        fixed_coordinates = inverse_column_scales[:, None] * scaled_fixed
        normalized_gram = scaled_fixed.T @ gram @ scaled_fixed
        orthogonality_residual = float(
            np.linalg.norm(normalized_gram - np.eye(fixed.rank), ord="fro")
        )
        fixed_orthogonality_squared += orthogonality_residual**2
        local_projection = fixed_coordinates.T @ raw_gram
        fixed_coordinate_blocks.append((columns, fixed_coordinates))
        if materialize_ambient_vectors:
            ambient = _left_multiply(local_vocabulary, fixed_coordinates)
            fixed_ambient_blocks.append(ambient)

        for name in sorted(actions):
            physical_residual = (
                physical_action_representations[name] @ physical_fixed
                - physical_fixed
            )
            residual_operator = (
                0.0
                if fixed.rank == 0
                else float(np.linalg.norm(physical_residual, ord=2))
            )
            roundoff = float(
                np.finfo(np.float64).eps
                * max(1, len(indices), fixed.rank)
                * max(
                    1.0,
                    largest / max(smallest, np.finfo(np.float64).tiny),
                )
            )
            fixed_action_residuals[name].append(residual_operator)
            # The input column bounds certify authored/action construction and
            # enter the fixed-kernel solve above.  They are not an additional
            # residual of the already-compiled fixed physical action.  Adding
            # them again here double-counts the same uncertainty during null
            # classification and can turn an algebra-floor zero ambiguous.
            fixed_action_bounds[name].append(residual_operator + roundoff)

        component_records.append(
            {
                "label": f"direct_action_component_{component_index}",
                "size": int(len(indices)),
                "fixed_rank": int(fixed.rank),
                "rank_threshold": rank_threshold,
                "column_scale_min": float(np.min(column_scales)),
                "column_scale_max": float(np.max(column_scales)),
                "fixed_subspace": dict(fixed.metadata),
                "logical_projection_shape": tuple(int(value) for value in local_projection.shape),
            }
        )
        total_rank += int(fixed.rank)

    fixed_vocabulary_coordinates = np.zeros(
        (matrix.shape[1], total_rank),
        dtype=np.float64,
    )
    logical_projection_coordinates = np.zeros(
        (total_rank, matrix.shape[1]),
        dtype=np.float64,
    )
    rank_offset = 0
    for columns, normalized_fixed in fixed_coordinate_blocks:
        next_offset = rank_offset + normalized_fixed.shape[1]
        fixed_vocabulary_coordinates[columns, rank_offset:next_offset] = normalized_fixed
        gram = _dense_gram(matrix[:, columns], matrix[:, columns])
        logical_projection_coordinates[rank_offset:next_offset, columns] = (
            normalized_fixed.T @ gram
        )
        rank_offset = next_offset

    singular_values_all.sort(reverse=True)
    largest = max(singular_values_all, default=1.0)
    smallest = min(singular_values_all, default=1.0)
    fixed_orthogonality_residual = float(np.sqrt(fixed_orthogonality_squared))
    orthogonality_bound = float(
        absolute_tolerance
        + relative_tolerance * max(1.0, np.sqrt(float(total_rank)))
        + np.finfo(np.float64).eps * max(matrix.shape) * max(1.0, largest / smallest)
    )
    if fixed_orthogonality_residual > orthogonality_bound:
        raise VocabularyRankCertificationError(
            "direct-action fixed response basis orthogonality failed: "
            f"residual={fixed_orthogonality_residual:.6e}, "
            f"bound={orthogonality_bound:.6e}"
        )
    fixed_action_certification = tuple(
        {
            "generator": name,
            "residual_operator_norm": max(fixed_action_residuals[name], default=0.0),
            "certified_operator_bound": max(fixed_action_bounds[name], default=0.0),
            "subtraction_roundoff_bound": max(
                (
                    bound - residual
                    for bound, residual in zip(
                        fixed_action_bounds[name],
                        fixed_action_residuals[name],
                    )
                ),
                default=0.0,
            ),
        }
        for name in sorted(actions)
    )
    metadata = {
        "compiler": "direct_term_action__physical_fixed_metric_v1",
        "target_independent": True,
        "sample_grid_used": False,
        "logical_channel_count": int(matrix.shape[1]),
        "fixed_rank": int(total_rank),
        "logical_channel_ids": logical_ids,
        "antiunitary_parities": {
            name: bool(antiunitary_parities[name]) for name in sorted(actions)
        },
        "vocabulary_singular_values": tuple(singular_values_all),
        "vocabulary_rank_threshold": max(rank_thresholds, default=0.0),
        "rank_gray_factor": float(rank_gray_factor),
        "vocabulary_scaling_policy": "independent_positive_column_norm_v1",
        "vocabulary_condition_estimate": float(largest / smallest),
        "vocabulary_orthogonality_residual": 0.0,
        "vocabulary_orthogonality_bound": orthogonality_bound,
        "fixed_orthogonality_residual": fixed_orthogonality_residual,
        "generator_certification": tuple(
            {
                "generator": name,
                "antiunitary": bool(antiunitary_parities[name]),
                "outside_span_residual": 0.0,
                "span_certification_bound": float(
                    np.linalg.norm(column_bounds.get(name, np.zeros(matrix.shape[1])))
                ),
                "columns": (),
            }
            for name in sorted(actions)
        ),
        "fixed_physical_action_certification": fixed_action_certification,
        "fixed_subspace": {
            "algorithm": "direct_stacked_A_minus_I__physical_metric_v1",
            "components": tuple(component_records),
        },
        "invariant_component_certification": {
            "policy": "exact_metric_and_direct_action_support_graph_v1",
            "component_count": int(len(components)),
            "component_sizes": tuple(len(values) for values in components),
            "max_certified_outside_component_leakage": 0.0,
        },
        "early_direct_sum_certification": {
            "policy": "exact_coefficient_support_overlap_v1",
            "component_count": int(len(components)),
            "component_sizes": tuple(len(values) for values in components),
            "exact_zero_required": True,
            "nominal_degree_partition_used": False,
        },
        "ambient_fixed_vectors_materialized": bool(materialize_ambient_vectors),
        "block_rule": "exact_metric_support_plus_direct_generator_action",
    }
    fixed_vectors = (
        np.column_stack(fixed_ambient_blocks)
        if materialize_ambient_vectors and fixed_ambient_blocks
        else (
            np.zeros((matrix.shape[0], 0), dtype=np.float64)
            if materialize_ambient_vectors
            else None
        )
    )
    return FixedVocabularyCompilation(
        fixed_vectors=fixed_vectors,
        fixed_vocabulary_coordinates=fixed_vocabulary_coordinates,
        logical_projection_coordinates=logical_projection_coordinates,
        rank=total_rank,
        logical_channel_ids=logical_ids,
        metadata=metadata,
    )


__all__ = [
    "FIXED_VOCABULARY_COMPILER_V1",
    "FixedVocabularyCompilation",
    "GeneratorRepresentationError",
    "GeneratorSpanClosureError",
    "VocabularyRankCertificationError",
    "compile_generator_fixed_vocabulary",
    "compile_generator_fixed_vocabulary_from_actions",
]
