"""Small-matrix Reynolds compilation in exact logical-owner action blocks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
from scipy import sparse


LOCAL_REYNOLDS_ALGORITHM_V1 = "exact_action_block__small_matrix_reynolds_v1"
LOCAL_REYNOLDS_BLOCK_BACKEND_V1 = (
    "certified_action_graph_blocks__group_average_v1"
)
LOCAL_REYNOLDS_COMPONENT_POLICY_V1 = "generator__hermitian_action_graph_v1"
LOCAL_REYNOLDS_OWNER_POLICY_V1 = (
    "canonical_owner_order__small_projection_gram_schmidt_v1"
)
LOCAL_REYNOLDS_MATERIALIZATION_POLICY_V1 = (
    "selected_action_block_owners_only_v1"
)
LOCAL_REYNOLDS_REDUCE_POLICY_V1 = (
    "raw_owner_injectivity__no_post_projection_reduction_v1"
)
LOCAL_FIXED_STRUCTURAL_PLAN_POLICY_V1 = (
    "exact_physical_dedup__skip_legacy_cyclic_image_orbits_v1"
)
LOCAL_FIXED_PHYSICAL_SEED_DIGEST_V2 = (
    "canonical_csr_buffers__global_complex_scale_v2"
)
LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1 = 1.0e-12
LOCAL_FIXED_RELATIVE_TOLERANCE_V1 = 1.0e-12
LOCAL_FIXED_GRAY_ZONE_FACTOR_V1 = 100.0


def _immutable_array(
    value: object,
    *,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("object arrays require recursive freezing")
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _nonnegative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _canonical_indices(
    values: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be an integer sequence") from error
    result = tuple(_nonnegative_integer(value, name=name) for value in raw)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contain duplicate entries")
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must use canonical increasing order")
    return result


def _input_owner_indices(values: object) -> tuple[int, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("logical owner indices must be an integer sequence") from error
    result = tuple(
        _nonnegative_integer(value, name="logical owner index") for value in raw
    )
    if not result:
        raise ValueError("logical owner indices must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("logical owner indices contain duplicate entries")
    return result


def _mapping_names(mapping: Mapping[object, object], *, name: str) -> tuple[str, ...]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping")
    names = tuple(mapping.keys())
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError(f"{name} keys must be non-empty strings")
    if names != tuple(sorted(names)):
        raise ValueError(f"{name} must use canonical generator ordering")
    return tuple(str(value) for value in names)


def _bounded_mapping(
    value: Mapping[object, object],
    *,
    name: str,
) -> tuple[tuple[str, ...], Mapping[str, float]]:
    names = _mapping_names(value, name=name)
    normalized = {
        generator: _nonnegative_scalar(
            value[generator],
            name=f"{name} value",
        )
        for generator in names
    }
    return names, MappingProxyType(normalized)


def _real_dense(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if np.any(np.imag(array) != 0.0):
            raise ValueError(f"{name} must be real")
        array = np.real(array)
    result = np.asarray(array, dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return _immutable_array(result, dtype=np.dtype(np.float64))


def _real_vector(value: object, *, name: str) -> np.ndarray:
    result = _real_dense(value, name=name)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _real_square_action(
    value: object,
    *,
    dimension: int,
    name: str,
) -> np.ndarray | sparse.csc_matrix:
    if sparse.issparse(value):
        matrix = sparse.csc_matrix(value, copy=True)
        if np.iscomplexobj(matrix.data):
            if np.any(np.imag(matrix.data) != 0.0):
                raise ValueError(f"{name} must be real")
            matrix.data = np.asarray(np.real(matrix.data), dtype=np.float64)
        else:
            matrix = sparse.csc_matrix(matrix, dtype=np.float64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if matrix.shape != (dimension, dimension):
            raise ValueError(
                f"{name} has shape {matrix.shape}; expected {(dimension, dimension)}"
            )
        if np.any(~np.isfinite(matrix.data)):
            raise ValueError(f"{name} contains non-finite values")
        return matrix
    matrix = _real_dense(value, name=name)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} has shape {matrix.shape}; expected {(dimension, dimension)}"
        )
    return matrix


def _reordered_action(
    action: np.ndarray | sparse.csc_matrix,
    order: np.ndarray,
) -> np.ndarray | sparse.csc_matrix:
    if sparse.issparse(action):
        return sparse.csc_matrix(action[order, :][:, order], copy=True)
    return np.asarray(action[np.ix_(order, order)], dtype=np.float64)


def _restricted_dense_action(
    action: np.ndarray | sparse.csc_matrix,
    positions: np.ndarray,
) -> np.ndarray:
    if sparse.issparse(action):
        return np.asarray(action[positions, :][:, positions].toarray(), dtype=np.float64)
    return np.asarray(action[np.ix_(positions, positions)], dtype=np.float64)


class LocalResponseCompilationError(RuntimeError):
    """The small owner-space Reynolds backend cannot certify this input."""

    def __init__(
        self,
        reason: str,
        *,
        certificate: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason)
        self.certificate = _freeze(dict(certificate or {}))
        super().__init__(self.reason)


@dataclass(frozen=True)
class LocalActionBlock:
    """One invariant block in the real logical-owner representation."""

    block_index: int
    nominal_degrees: tuple[int, ...]
    owner_indices: tuple[int, ...]
    generator_actions: Mapping[str, np.ndarray]
    generator_error_bounds: Mapping[str, float]
    antiunitary_parities: Mapping[str, bool]
    hermitian_action: np.ndarray

    def __post_init__(self) -> None:
        block_index = _nonnegative_integer(self.block_index, name="block_index")
        owners = _canonical_indices(
            self.owner_indices,
            name="owner indices",
            allow_empty=False,
        )
        degrees = tuple(
            _nonnegative_integer(value, name="nominal degree")
            for value in tuple(self.nominal_degrees)
        )
        dimension = len(owners)
        if len(degrees) != dimension:
            raise ValueError("nominal degree count must match owner indices")

        names = _mapping_names(self.generator_actions, name="generator_actions")
        if not names:
            raise ValueError("generator_actions must not be empty")
        actions: dict[str, np.ndarray] = {}
        for name in names:
            action = _real_dense(
                self.generator_actions[name],
                name=f"generator action {name!r}",
            )
            if action.shape != (dimension, dimension):
                raise ValueError(
                    f"generator action {name!r} has shape {action.shape}; "
                    f"expected {(dimension, dimension)}"
                )
            actions[name] = action

        bound_names, error_bounds = _bounded_mapping(
            self.generator_error_bounds,
            name="generator error bounds",
        )
        if bound_names != names:
            raise ValueError("generator error-bound keys must match generator keys")
        parity_names = _mapping_names(
            self.antiunitary_parities,
            name="antiunitary_parities",
        )
        if parity_names != names:
            raise ValueError("antiunitary parity keys must match generator keys")
        if any(
            not isinstance(self.antiunitary_parities[name], (bool, np.bool_))
            for name in names
        ):
            raise ValueError("antiunitary parities must be boolean")
        parities = MappingProxyType(
            {name: bool(self.antiunitary_parities[name]) for name in names}
        )
        hermitian = _real_dense(self.hermitian_action, name="hermitian_action")
        if hermitian.shape != (dimension, dimension):
            raise ValueError(
                f"hermitian_action has shape {hermitian.shape}; "
                f"expected {(dimension, dimension)}"
            )

        object.__setattr__(self, "block_index", block_index)
        object.__setattr__(self, "nominal_degrees", degrees)
        object.__setattr__(self, "owner_indices", owners)
        object.__setattr__(self, "generator_actions", MappingProxyType(actions))
        object.__setattr__(self, "generator_error_bounds", error_bounds)
        object.__setattr__(self, "antiunitary_parities", parities)
        object.__setattr__(self, "hermitian_action", hermitian)


@dataclass(frozen=True)
class LocalFixedCompilation:
    """Certified fixed coordinates for one action block."""

    component_index: int
    component_owner_indices: tuple[int, ...]
    fixed_vocabulary_coordinates: np.ndarray
    logical_projection_coordinates: np.ndarray
    fixed_rank: int
    selected_global_owner_indices: tuple[int, ...]
    singular_values: np.ndarray
    gray_zone_certificate: Mapping[str, Any]
    generator_residuals: Mapping[str, float]
    hermitian_residual: float
    propagated_error_bounds: Mapping[str, float]
    certification_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        block_index = _nonnegative_integer(
            self.component_index,
            name="component_index",
        )
        owners = _canonical_indices(
            self.component_owner_indices,
            name="component owner indices",
            allow_empty=False,
        )
        rank = _nonnegative_integer(self.fixed_rank, name="fixed_rank")
        selected = _canonical_indices(
            self.selected_global_owner_indices,
            name="selected owner indices",
            allow_empty=True,
        )
        if len(selected) != rank:
            raise ValueError("selected owner count must equal fixed_rank")
        if any(owner not in set(owners) for owner in selected):
            raise ValueError("selected owner does not belong to the component")

        coordinates = _real_dense(
            self.fixed_vocabulary_coordinates,
            name="fixed_vocabulary_coordinates",
        )
        if coordinates.shape != (len(owners), rank):
            raise ValueError(
                "fixed_vocabulary_coordinates has shape "
                f"{coordinates.shape}; expected {(len(owners), rank)}"
            )
        projection = _real_dense(
            self.logical_projection_coordinates,
            name="logical_projection_coordinates",
        )
        if projection.shape != (rank, len(owners)):
            raise ValueError(
                "logical_projection_coordinates has shape "
                f"{projection.shape}; expected {(rank, len(owners))}"
            )
        singular_values = _real_vector(self.singular_values, name="singular_values")
        if singular_values.shape != (len(owners),):
            raise ValueError(
                f"singular_values has shape {singular_values.shape}; "
                f"expected {(len(owners),)}"
            )
        residual_names, residuals = _bounded_mapping(
            self.generator_residuals,
            name="generator residuals",
        )
        bound_names, propagated = _bounded_mapping(
            self.propagated_error_bounds,
            name="propagated error bounds",
        )
        if residual_names != bound_names:
            raise ValueError(
                "generator residual and propagated error-bound keys must match"
            )
        if not isinstance(self.gray_zone_certificate, Mapping):
            raise ValueError("gray_zone_certificate must be a mapping")
        if not isinstance(self.certification_metadata, Mapping):
            raise ValueError("certification_metadata must be a mapping")

        object.__setattr__(self, "component_index", block_index)
        object.__setattr__(self, "component_owner_indices", owners)
        object.__setattr__(self, "fixed_vocabulary_coordinates", coordinates)
        object.__setattr__(self, "logical_projection_coordinates", projection)
        object.__setattr__(self, "fixed_rank", rank)
        object.__setattr__(self, "selected_global_owner_indices", selected)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "gray_zone_certificate", _freeze(dict(self.gray_zone_certificate)))
        object.__setattr__(self, "generator_residuals", residuals)
        object.__setattr__(
            self,
            "hermitian_residual",
            _nonnegative_scalar(self.hermitian_residual, name="hermitian_residual"),
        )
        object.__setattr__(self, "propagated_error_bounds", propagated)
        object.__setattr__(self, "certification_metadata", _freeze(dict(self.certification_metadata)))


def build_local_action_blocks(
    *,
    nominal_degrees: Sequence[int],
    logical_owner_indices: Sequence[int],
    generator_actions: Mapping[str, object],
    generator_error_bounds: Mapping[str, float],
    antiunitary_parities: Mapping[str, bool],
    hermitian_action: object,
) -> tuple[LocalActionBlock, ...]:
    """Build connected components of generator/H action support only."""

    owners_input = _input_owner_indices(logical_owner_indices)
    dimension = len(owners_input)
    degrees_input = tuple(
        _nonnegative_integer(value, name="nominal degree")
        for value in tuple(nominal_degrees)
    )
    if len(degrees_input) != dimension:
        raise ValueError("nominal degree count must match logical owner count")
    names = _mapping_names(generator_actions, name="generator_actions")
    if not names:
        raise ValueError("generator_actions must not be empty")
    if set(generator_error_bounds) != set(names):
        raise ValueError("generator error-bound keys must match generator keys")
    if set(antiunitary_parities) != set(names):
        raise ValueError("antiunitary parity keys must match generator keys")

    order = np.asarray(
        sorted(range(dimension), key=lambda index: owners_input[index]),
        dtype=np.int64,
    )
    owners = tuple(owners_input[int(index)] for index in order)
    degrees = tuple(degrees_input[int(index)] for index in order)
    actions = {
        name: _reordered_action(
            _real_square_action(
                generator_actions[name],
                dimension=dimension,
                name=f"generator action {name!r}",
            ),
            order,
        )
        for name in names
    }
    hermitian = _reordered_action(
        _real_square_action(
            hermitian_action,
            dimension=dimension,
            name="hermitian_action",
        ),
        order,
    )

    adjacency = sparse.csr_matrix((dimension, dimension), dtype=np.float64)
    for raw_action in (*actions.values(), hermitian):
        support = sparse.csr_matrix(raw_action, dtype=np.float64)
        support.sum_duplicates()
        support.eliminate_zeros()
        support.data = np.ones(support.nnz, dtype=np.float64)
        adjacency = (adjacency + support + support.T).tocsr()
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    count, labels = sparse.csgraph.connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    position_blocks = tuple(
        tuple(np.flatnonzero(labels == label).tolist())
        for label in range(int(count))
    )
    position_blocks = tuple(
        sorted(position_blocks, key=lambda positions: owners[positions[0]])
    )

    return tuple(
        LocalActionBlock(
            block_index=block_index,
            nominal_degrees=tuple(degrees[position] for position in positions),
            owner_indices=tuple(owners[position] for position in positions),
            generator_actions={
                name: _restricted_dense_action(
                    actions[name],
                    np.asarray(positions, dtype=np.int64),
                )
                for name in names
            },
            generator_error_bounds={
                name: generator_error_bounds[name] for name in names
            },
            antiunitary_parities={
                name: antiunitary_parities[name] for name in names
            },
            hermitian_action=_restricted_dense_action(
                hermitian,
                np.asarray(positions, dtype=np.int64),
            ),
        )
        for block_index, positions in enumerate(position_blocks)
    )


def summarize_local_action_blocks(
    blocks: Sequence[LocalActionBlock],
) -> dict[str, Any]:
    """Summarize exact owner-action blocks without solving their fixed spaces."""

    ordered = tuple(blocks)
    if any(not isinstance(block, LocalActionBlock) for block in ordered):
        raise TypeError("action-block diagnostic requires LocalActionBlock values")
    generator_names = tuple(
        sorted(
            {
                str(name)
                for block in ordered
                for name in block.generator_actions
            }
        )
    )
    generator_nnz = {name: 0 for name in generator_names}
    generator_lower = {name: 0 for name in generator_names}
    hermitian_nnz = 0
    hermitian_lower = 0

    def counts(action: np.ndarray, degrees: tuple[int, ...]) -> tuple[int, int]:
        rows, columns = np.nonzero(np.asarray(action, dtype=np.float64))
        lower = sum(
            int(degrees[int(row)] < degrees[int(column)])
            for row, column in zip(rows, columns)
        )
        return int(len(rows)), int(lower)

    for block in ordered:
        for name in generator_names:
            if name not in block.generator_actions:
                raise ValueError(
                    "action-block diagnostic requires identical generator sets"
                )
            nonzero, lower = counts(
                block.generator_actions[name], block.nominal_degrees
            )
            generator_nnz[name] += nonzero
            generator_lower[name] += lower
        nonzero, lower = counts(block.hermitian_action, block.nominal_degrees)
        hermitian_nnz += nonzero
        hermitian_lower += lower

    dimensions = [len(block.owner_indices) for block in ordered]
    return {
        "schema_version": "local-action-block-diagnostic-v1",
        "closure_real_owner_count": int(sum(dimensions)),
        "action_block_count": int(len(dimensions)),
        "action_block_dimensions": dimensions,
        "maximum_action_block_dimension": int(max(dimensions, default=0)),
        "generator_action_nonzero_count": generator_nnz,
        "hermitian_action_nonzero_count": int(hermitian_nnz),
        "generator_lower_degree_edge_count": generator_lower,
        "hermitian_lower_degree_edge_count": int(hermitian_lower),
        "total_lower_degree_edge_count": int(
            sum(generator_lower.values()) + hermitian_lower
        ),
    }


def _gamma_n(dimension: int) -> float:
    epsilon = np.finfo(np.float64).eps
    product = max(1, int(dimension)) * epsilon
    if product >= 1.0:
        raise ValueError("floating-point error model dimension is too large")
    return float(product / (1.0 - product))


def _deterministic_logical_owners(
    logical_images: np.ndarray,
    owner_indices: tuple[int, ...],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    gray_zone_factor: float,
    block_index: int,
) -> tuple[int, ...]:
    rank = int(logical_images.shape[0])
    if rank == 0:
        return ()
    scale = max(1.0, float(np.linalg.norm(logical_images, ord=2)))
    threshold = max(
        absolute_tolerance
        + relative_tolerance * scale
        + _gamma_n(logical_images.shape[1]) * scale,
        np.finfo(np.float64).eps,
    )
    selected: list[int] = []
    orthonormal: list[np.ndarray] = []
    for position in range(logical_images.shape[1]):
        vector = np.asarray(logical_images[:, position], dtype=np.float64).copy()
        norm = float(np.linalg.norm(vector))
        if norm <= threshold:
            continue
        vector /= norm
        for basis_vector in orthonormal:
            vector -= basis_vector * float(basis_vector @ vector)
        residual = float(np.linalg.norm(vector))
        if residual <= threshold:
            continue
        if residual < gray_zone_factor * threshold:
            raise LocalResponseCompilationError(
                "ambiguous_owner_pivots",
                certificate={
                    "block_index": block_index,
                    "owner": owner_indices[position],
                    "residual": residual,
                    "threshold": threshold,
                },
            )
        orthonormal.append(vector / residual)
        selected.append(position)
        if len(selected) == rank:
            break
    if len(selected) != rank:
        raise LocalResponseCompilationError(
            "insufficient_logical_owner_rank",
            certificate={
                "block_index": block_index,
                "expected_rank": rank,
                "selected_rank": len(selected),
            },
        )
    return tuple(owner_indices[position] for position in selected)


def _canonical_group_words(
    group_words: Sequence[Sequence[str]],
    *,
    generator_names: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    words = tuple(tuple(str(name) for name in word) for word in group_words)
    if not words:
        raise ValueError("group_words must not be empty")
    if len(set(words)) != len(words):
        raise ValueError("group_words must be unique")
    if () not in words:
        raise ValueError("group_words must contain the identity word")
    missing = sorted(
        {
            name
            for word in words
            for name in word
            if name not in generator_names
        }
    )
    if missing:
        raise LocalResponseCompilationError(
            "group_word_missing_generator_action",
            certificate={"missing_generators": tuple(missing)},
        )
    return words


def _word_action(
    generator_actions: Mapping[str, np.ndarray],
    word: tuple[str, ...],
    *,
    dimension: int,
) -> np.ndarray:
    result = np.eye(dimension, dtype=np.float64)
    for name in word:
        result = np.asarray(generator_actions[name], dtype=np.float64) @ result
    return result


def solve_local_reynolds_block(
    block: LocalActionBlock,
    *,
    group_words: Sequence[Sequence[str]],
    absolute_tolerance: float = LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1,
    relative_tolerance: float = LOCAL_FIXED_RELATIVE_TOLERANCE_V1,
    gray_zone_factor: float = LOCAL_FIXED_GRAY_ZONE_FACTOR_V1,
) -> LocalFixedCompilation:
    """Average one finite group and Hermitian involution in owner space."""

    if not isinstance(block, LocalActionBlock):
        raise TypeError("block must be a LocalActionBlock")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("local Reynolds tolerances must be non-negative")
    if gray_zone_factor <= 1.0:
        raise ValueError("gray_zone_factor must exceed one")
    words = _canonical_group_words(
        group_words,
        generator_names=tuple(block.generator_actions),
    )
    dimension = len(block.owner_indices)
    actions = {
        name: np.asarray(action, dtype=np.float64)
        for name, action in block.generator_actions.items()
    }
    word_actions = tuple(
        _word_action(actions, word, dimension=dimension) for word in words
    )
    group_projector = sum(
        word_actions,
        np.zeros((dimension, dimension), dtype=np.float64),
    ) / float(len(word_actions))
    hermitian = np.asarray(block.hermitian_action, dtype=np.float64)
    projector = 0.5 * (np.eye(dimension) + hermitian) @ group_projector

    maximum_word_length = max((len(word) for word in words), default=0)
    action_scale = max(
        1.0,
        *(float(np.linalg.norm(action, ord=2)) for action in word_actions),
        float(np.linalg.norm(hermitian, ord=2)),
    )
    input_error = max(
        (float(value) for value in block.generator_error_bounds.values()),
        default=0.0,
    )
    certification_bound = float(
        absolute_tolerance
        + relative_tolerance * action_scale
        + (maximum_word_length + 2) * input_error * action_scale
        + 32.0
        * _gamma_n(max(1, dimension * max(1, maximum_word_length)))
        * action_scale
    )
    certification_limit = gray_zone_factor * max(
        certification_bound,
        np.finfo(np.float64).eps,
    )
    residuals = {
        "group_projector_idempotence": float(
            np.linalg.norm(group_projector @ group_projector - group_projector, ord=2)
        ),
        "hermitian_involution": float(
            np.linalg.norm(hermitian @ hermitian - np.eye(dimension), ord=2)
        ),
        "hermitian_group_commutator": float(
            np.linalg.norm(
                hermitian @ group_projector - group_projector @ hermitian,
                ord=2,
            )
        ),
        "combined_projector_idempotence": float(
            np.linalg.norm(projector @ projector - projector, ord=2)
        ),
        **{
            f"generator_fixed:{name}": float(
                np.linalg.norm(action @ group_projector - group_projector, ord=2)
            )
            for name, action in actions.items()
        },
    }
    failed = {
        name: value for name, value in residuals.items() if value > certification_limit
    }
    if failed:
        raise LocalResponseCompilationError(
            "local_reynolds_projector_certification_failed",
            certificate={
                "block_index": block.block_index,
                "certification_limit": certification_limit,
                "failed_residuals": failed,
            },
        )

    left_vectors, singular_values, _ = scipy.linalg.svd(
        projector,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesvd",
    )
    singular_scale = max(
        1.0,
        float(singular_values[0]) if singular_values.size else 0.0,
    )
    rank_threshold = float(
        certification_bound
        + absolute_tolerance
        + relative_tolerance * singular_scale
        + _gamma_n(max(1, dimension)) * singular_scale
    )
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    ambiguous = singular_values[
        (singular_values > rank_threshold)
        & (singular_values < gray_zone_factor * rank_threshold)
    ]
    if ambiguous.size:
        raise LocalResponseCompilationError(
            "local_reynolds_rank_gray_zone",
            certificate={
                "block_index": block.block_index,
                "rank_threshold": rank_threshold,
                "ambiguous_singular_values": tuple(float(value) for value in ambiguous),
            },
        )

    if rank:
        logical_images = left_vectors[:, :rank].T @ projector
        selected = _deterministic_logical_owners(
            logical_images,
            block.owner_indices,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            gray_zone_factor=gray_zone_factor,
            block_index=block.block_index,
        )
        positions = {owner: index for index, owner in enumerate(block.owner_indices)}
        fixed_coordinates = np.asarray(
            projector[:, [positions[owner] for owner in selected]],
            dtype=np.float64,
        )
        logical_projection = np.asarray(
            scipy.linalg.lstsq(
                fixed_coordinates,
                projector,
                cond=rank_threshold,
                check_finite=False,
                lapack_driver="gelsy",
            )[0],
            dtype=np.float64,
        )
    else:
        selected = ()
        fixed_coordinates = np.zeros((dimension, 0), dtype=np.float64)
        logical_projection = np.zeros((0, dimension), dtype=np.float64)

    reconstruction_residual = float(
        np.linalg.norm(fixed_coordinates @ logical_projection - projector, ord=2)
    )
    if reconstruction_residual > certification_limit:
        raise LocalResponseCompilationError(
            "local_reynolds_owner_frame_certification_failed",
            certificate={
                "block_index": block.block_index,
                "reconstruction_residual": reconstruction_residual,
                "certification_limit": certification_limit,
            },
        )
    generator_residuals = {
        name: float(np.linalg.norm(action @ fixed_coordinates - fixed_coordinates, ord=2))
        for name, action in actions.items()
    }
    hermitian_residual = float(
        np.linalg.norm(hermitian @ fixed_coordinates - fixed_coordinates, ord=2)
    )
    padded_singular_values = np.zeros(dimension, dtype=np.float64)
    padded_singular_values[: singular_values.size] = singular_values

    return LocalFixedCompilation(
        component_index=block.block_index,
        component_owner_indices=block.owner_indices,
        fixed_vocabulary_coordinates=fixed_coordinates,
        logical_projection_coordinates=logical_projection,
        fixed_rank=rank,
        selected_global_owner_indices=selected,
        singular_values=padded_singular_values,
        gray_zone_certificate={
            "threshold": rank_threshold,
            "gray_zone_upper": gray_zone_factor * rank_threshold,
            "gray_zone_factor": gray_zone_factor,
        },
        generator_residuals=generator_residuals,
        hermitian_residual=hermitian_residual,
        propagated_error_bounds={name: certification_bound for name in actions},
        certification_metadata={
            "algorithm": LOCAL_REYNOLDS_ALGORITHM_V1,
            "block_backend": LOCAL_REYNOLDS_BLOCK_BACKEND_V1,
            "component_policy": LOCAL_REYNOLDS_COMPONENT_POLICY_V1,
            "owner_policy": LOCAL_REYNOLDS_OWNER_POLICY_V1,
            "raw_dimension": dimension,
            "independent_dimension": dimension,
            "fixed_rank": rank,
            "group_order": len(words),
            "maximum_group_word_length": maximum_word_length,
            "projector_singular_values": tuple(float(value) for value in singular_values),
            "projector_rank_threshold": rank_threshold,
            "certification_bound": certification_bound,
            "certification_limit": certification_limit,
            "certification_residuals": residuals,
            "owner_frame_reconstruction_residual": reconstruction_residual,
            "generator_action_nnz": {
                name: int(np.count_nonzero(action)) for name, action in actions.items()
            },
            "raw_owner_injectivity_required": True,
        },
    )


__all__ = [
    "LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1",
    "LOCAL_FIXED_GRAY_ZONE_FACTOR_V1",
    "LOCAL_FIXED_PHYSICAL_SEED_DIGEST_V2",
    "LOCAL_FIXED_RELATIVE_TOLERANCE_V1",
    "LOCAL_FIXED_STRUCTURAL_PLAN_POLICY_V1",
    "LOCAL_REYNOLDS_ALGORITHM_V1",
    "LOCAL_REYNOLDS_BLOCK_BACKEND_V1",
    "LOCAL_REYNOLDS_COMPONENT_POLICY_V1",
    "LOCAL_REYNOLDS_MATERIALIZATION_POLICY_V1",
    "LOCAL_REYNOLDS_OWNER_POLICY_V1",
    "LOCAL_REYNOLDS_REDUCE_POLICY_V1",
    "LocalActionBlock",
    "LocalFixedCompilation",
    "LocalResponseCompilationError",
    "build_local_action_blocks",
    "summarize_local_action_blocks",
    "solve_local_reynolds_block",
]
