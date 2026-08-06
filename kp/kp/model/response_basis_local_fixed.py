"""Certified component-local fixed-response compilation.

The production p=0 compiler averages the finite group on exact small action
blocks in the real logical-owner vocabulary.  The older Gram-whitened
generator-nullspace solver remains available as an independent test oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
from scipy import sparse

from .response_basis_fixed_subspace import (
    FixedSubspaceCertificationError,
    RealGeneratorAction,
    solve_generator_fixed_subspace,
)


LOCAL_FIXED_ALGORITHM_V1 = "exact_joint_component__gram_whitened_generator_nullspace_v1"
LOCAL_FIXED_COMPONENT_POLICY_V1 = "generator__hermitian__ambient_support_union_v1"
LOCAL_FIXED_COLUMN_SCALING_V1 = "independent_owner_order__unit_column_norm_v1"
LOCAL_FIXED_OWNER_POLICY_V1 = "canonical_owner_order__small_projection_gram_schmidt_v1"
LOCAL_FIXED_BLOCK_BACKEND_V2 = (
    "exact_component__certified_action_graph_blocks_all_full_rank_v2"
)
LOCAL_REYNOLDS_ALGORITHM_V1 = "exact_action_block__small_matrix_reynolds_v1"
LOCAL_REYNOLDS_BLOCK_BACKEND_V1 = (
    "certified_action_graph_blocks__group_average_v1"
)
LOCAL_FIXED_STRUCTURAL_PLAN_POLICY_V1 = (
    "exact_physical_dedup__skip_legacy_cyclic_image_orbits_v1"
)
LOCAL_FIXED_PHYSICAL_SEED_DIGEST_V2 = (
    "canonical_csr_buffers__global_complex_scale_v2"
)
LOCAL_FIXED_MATERIALIZATION_POLICY_V1 = (
    "fixed_survivors_after_global_degree_tail_certificate_v1"
)
LOCAL_FIXED_REDUCE_POLICY_V1 = "small_global_degree_filtered_tail_certificate_v1"
LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1 = 1.0e-12
LOCAL_FIXED_RELATIVE_TOLERANCE_V1 = 1.0e-12
LOCAL_FIXED_GRAY_ZONE_FACTOR_V1 = 100.0
LOCAL_FIXED_DENSE_COMPONENT_CUTOFF_V1 = 512


def _immutable_numeric_array(value: object, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    """Return a byte-backed NumPy copy whose WRITEABLE flag cannot be restored."""

    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("object arrays require recursive freezing")
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _deep_freeze(value: Any) -> Any:
    """Defensively copy nested diagnostic data into immutable containers."""

    if sparse.issparse(value):
        matrix = sparse.csc_matrix(value, copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        matrix.data = _immutable_numeric_array(matrix.data)
        matrix.indices = _immutable_numeric_array(matrix.indices)
        matrix.indptr = _immutable_numeric_array(matrix.indptr)
        return matrix
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _deep_freeze(value.tolist())
        return _immutable_numeric_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _canonical_indices(
    values: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be an integer sequence") from error
    indices = tuple(
        _nonnegative_integer(value, name=name.rstrip("s")) for value in raw_values
    )
    if not allow_empty and not indices:
        raise ValueError(f"{name} must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contain duplicate values")
    if indices != tuple(sorted(indices)):
        raise ValueError(f"{name} must use canonical increasing ordering")
    return indices


def _real_dense_matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real matrix")
    if raw.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return _immutable_numeric_array(matrix, dtype=np.dtype(np.float64))


def _real_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    vector = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return _immutable_numeric_array(vector, dtype=np.dtype(np.float64))


def _real_sparse_csc(value: object, *, name: str) -> sparse.csc_matrix:
    if not sparse.issparse(value):
        raw = np.asarray(value)
        if raw.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
    elif np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    matrix = sparse.csc_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} contains non-finite values")
    matrix.data = _immutable_numeric_array(matrix.data)
    matrix.indices = _immutable_numeric_array(matrix.indices)
    matrix.indptr = _immutable_numeric_array(matrix.indptr)
    return matrix


def _generator_names(mapping: Mapping[object, object], *, name: str) -> tuple[str, ...]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping")
    raw_names = tuple(mapping.keys())
    if any(not isinstance(key, str) or not key for key in raw_names):
        raise ValueError(f"{name} keys must be non-empty generator names")
    names = tuple(raw_names)
    if names != tuple(sorted(names)):
        raise ValueError(f"{name} must use canonical generator ordering")
    return names


def _bounded_scalar_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> tuple[tuple[str, ...], Mapping[str, float]]:
    names = _generator_names(value, name=name)
    normalized: dict[str, float] = {}
    singular = name[:-1] if name.endswith("s") else name
    for generator_name in names:
        raw = value[generator_name]
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(
                f"{singular} for {generator_name!r} must be finite and non-negative"
            )
        try:
            bound = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{singular} for {generator_name!r} must be finite and non-negative"
            ) from error
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError(
                f"{singular} for {generator_name!r} must be finite and non-negative"
            )
        normalized[generator_name] = bound
    return names, MappingProxyType(normalized)


def _nonnegative_integer_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> Mapping[str, int]:
    names = _generator_names(value, name=name)
    normalized = {
        generator_name: _nonnegative_integer(
            value[generator_name],
            name=f"{name} value",
        )
        for generator_name in names
    }
    return MappingProxyType(normalized)


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


class LocalGeneratorNullspaceUnavailable(RuntimeError):
    """A component-local generator nullspace could not be certified."""

    def __init__(
        self,
        reason: str,
        *,
        certificate: Mapping[str, Any] | None = None,
    ) -> None:
        stable_reason = str(reason)
        super().__init__(stable_reason)
        self.reason = stable_reason
        self.certificate = _deep_freeze(dict(certificate or {}))


@dataclass(frozen=True)
class LocalVocabularyComponent:
    """One canonical local real vocabulary and its generator actions."""

    component_index: int
    nominal_degrees: tuple[int, ...]
    logical_owner_indices: tuple[int, ...]
    provenance_by_owner: Mapping[int, Any]
    ambient_columns: sparse.csc_matrix
    generator_actions: Mapping[str, np.ndarray]
    generator_error_bounds: Mapping[str, float]
    antiunitary_parities: Mapping[str, bool]
    hermitian_action: np.ndarray
    exact_support_rows: tuple[int, ...]
    column_absolute_error_bounds: np.ndarray | None = None
    source_generator_action_nnz: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        component_index = _nonnegative_integer(
            self.component_index,
            name="component_index",
        )
        owner_indices = _canonical_indices(
            self.logical_owner_indices,
            name="logical owner indices",
            allow_empty=False,
        )
        try:
            raw_degrees = tuple(self.nominal_degrees)
        except TypeError as error:
            raise ValueError("nominal degrees must be an integer sequence") from error
        nominal_degrees = tuple(
            _nonnegative_integer(value, name="nominal degree")
            for value in raw_degrees
        )
        if len(nominal_degrees) != len(owner_indices):
            raise ValueError("nominal degree count must match the local column count")

        ambient_columns = _real_sparse_csc(
            self.ambient_columns,
            name="ambient_columns",
        )
        local_dimension = len(owner_indices)
        if ambient_columns.shape[1] != local_dimension:
            raise ValueError(
                "ambient_columns column count must match logical owner indices"
            )

        if not isinstance(self.provenance_by_owner, Mapping):
            raise ValueError("provenance_by_owner must be a mapping")
        provenance_keys = tuple(self.provenance_by_owner.keys())
        if provenance_keys != owner_indices:
            raise ValueError(
                "provenance keys must exactly match logical owner indices in "
                "canonical order"
            )
        provenance = _deep_freeze(dict(self.provenance_by_owner))

        generator_names = _generator_names(
            self.generator_actions,
            name="generator_actions",
        )
        if not generator_names:
            raise ValueError("generator_actions must not be empty")
        actions: dict[str, np.ndarray] = {}
        for generator_name in generator_names:
            action = _real_dense_matrix(
                self.generator_actions[generator_name],
                name=f"generator action {generator_name!r}",
            )
            if action.shape != (local_dimension, local_dimension):
                raise ValueError(
                    f"generator action {generator_name!r} has shape {action.shape}; "
                    f"expected {(local_dimension, local_dimension)}"
                )
            actions[generator_name] = action

        parity_names = _generator_names(
            self.antiunitary_parities,
            name="antiunitary_parities",
        )
        if parity_names != generator_names:
            raise ValueError(
                "antiunitary parity keys must exactly match generator keys"
            )
        parities: dict[str, bool] = {}
        for generator_name in parity_names:
            parity = self.antiunitary_parities[generator_name]
            if not isinstance(parity, (bool, np.bool_)):
                raise ValueError(
                    f"antiunitary parity for {generator_name!r} must be boolean"
                )
            parities[generator_name] = bool(parity)

        bound_names, error_bounds = _bounded_scalar_mapping(
            self.generator_error_bounds,
            name="generator error bounds",
        )
        if bound_names != generator_names:
            raise ValueError(
                "generator error-bound keys must exactly match generator keys"
            )

        hermitian_action = _real_dense_matrix(
            self.hermitian_action,
            name="hermitian_action",
        )
        if hermitian_action.shape != (local_dimension, local_dimension):
            raise ValueError(
                f"hermitian_action has shape {hermitian_action.shape}; "
                f"expected {(local_dimension, local_dimension)}"
            )

        exact_support_rows = _canonical_indices(
            self.exact_support_rows,
            name="exact support rows",
            allow_empty=True,
        )
        if exact_support_rows and exact_support_rows[-1] >= ambient_columns.shape[0]:
            raise ValueError("exact support row lies outside the ambient row range")

        if self.column_absolute_error_bounds is None:
            column_absolute_error_bounds = _real_vector(
                np.zeros(local_dimension, dtype=np.float64),
                name="column_absolute_error_bounds",
            )
        else:
            column_absolute_error_bounds = _real_vector(
                self.column_absolute_error_bounds,
                name="column_absolute_error_bounds",
            )
        if column_absolute_error_bounds.shape != (local_dimension,):
            raise ValueError(
                "column_absolute_error_bounds must have shape "
                f"{(local_dimension,)}"
            )

        if self.source_generator_action_nnz is None:
            source_generator_action_nnz = MappingProxyType(
                {
                    name: int(np.count_nonzero(actions[name]))
                    for name in generator_names
                }
            )
        else:
            source_generator_action_nnz = _nonnegative_integer_mapping(
                self.source_generator_action_nnz,
                name="source_generator_action_nnz",
            )
            if tuple(source_generator_action_nnz) != generator_names:
                raise ValueError(
                    "source generator-action nnz keys must exactly match generator keys"
                )

        object.__setattr__(self, "component_index", component_index)
        object.__setattr__(self, "nominal_degrees", nominal_degrees)
        object.__setattr__(self, "logical_owner_indices", owner_indices)
        object.__setattr__(self, "provenance_by_owner", provenance)
        object.__setattr__(self, "ambient_columns", ambient_columns)
        object.__setattr__(self, "generator_actions", MappingProxyType(actions))
        object.__setattr__(self, "generator_error_bounds", error_bounds)
        object.__setattr__(self, "antiunitary_parities", MappingProxyType(parities))
        object.__setattr__(self, "hermitian_action", hermitian_action)
        object.__setattr__(self, "exact_support_rows", exact_support_rows)
        object.__setattr__(
            self,
            "column_absolute_error_bounds",
            column_absolute_error_bounds,
        )
        object.__setattr__(
            self,
            "source_generator_action_nnz",
            source_generator_action_nnz,
        )


@dataclass(frozen=True)
class LocalFixedCompilation:
    """Certified fixed coordinates produced for one local component."""

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
        component_index = _nonnegative_integer(
            self.component_index,
            name="component_index",
        )
        owner_indices = _canonical_indices(
            self.component_owner_indices,
            name="component owner indices",
            allow_empty=False,
        )
        fixed_rank = _nonnegative_integer(self.fixed_rank, name="fixed_rank")
        selected = _canonical_indices(
            self.selected_global_owner_indices,
            name="selected owner indices",
            allow_empty=True,
        )

        fixed_coordinates = _real_dense_matrix(
            self.fixed_vocabulary_coordinates,
            name="fixed_vocabulary_coordinates",
        )
        expected_fixed_shape = (len(owner_indices), fixed_rank)
        if fixed_coordinates.shape != expected_fixed_shape:
            raise ValueError(
                "fixed_vocabulary_coordinates has shape "
                f"{fixed_coordinates.shape}; expected {expected_fixed_shape}"
            )
        projection_coordinates = _real_dense_matrix(
            self.logical_projection_coordinates,
            name="logical_projection_coordinates",
        )
        expected_projection_shape = (fixed_rank, len(owner_indices))
        if projection_coordinates.shape != expected_projection_shape:
            raise ValueError(
                "logical_projection_coordinates has shape "
                f"{projection_coordinates.shape}; expected {expected_projection_shape}"
            )
        if len(selected) != fixed_rank:
            raise ValueError("selected owner count must equal fixed_rank")
        owner_set = set(owner_indices)
        outside = tuple(index for index in selected if index not in owner_set)
        if outside:
            raise ValueError(
                f"selected owner {outside[0]} does not belong to the component"
            )

        singular_values = _real_vector(self.singular_values, name="singular_values")
        expected_singular_shape = (len(owner_indices),)
        if singular_values.shape != expected_singular_shape:
            raise ValueError(
                f"singular_values has shape {singular_values.shape}; "
                f"expected {expected_singular_shape}"
            )
        residual_names, generator_residuals = _bounded_scalar_mapping(
            self.generator_residuals,
            name="generator residuals",
        )
        bound_names, propagated_error_bounds = _bounded_scalar_mapping(
            self.propagated_error_bounds,
            name="propagated error bounds",
        )
        if residual_names != bound_names:
            raise ValueError(
                "generator residual and propagated error-bound keys must match"
            )
        hermitian_residual = _nonnegative_scalar(
            self.hermitian_residual,
            name="hermitian_residual",
        )
        if not isinstance(self.gray_zone_certificate, Mapping):
            raise ValueError("gray_zone_certificate must be a mapping")
        if not isinstance(self.certification_metadata, Mapping):
            raise ValueError("certification_metadata must be a mapping")

        object.__setattr__(self, "component_index", component_index)
        object.__setattr__(self, "component_owner_indices", owner_indices)
        object.__setattr__(self, "fixed_vocabulary_coordinates", fixed_coordinates)
        object.__setattr__(
            self,
            "logical_projection_coordinates",
            projection_coordinates,
        )
        object.__setattr__(self, "fixed_rank", fixed_rank)
        object.__setattr__(self, "selected_global_owner_indices", selected)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(
            self,
            "gray_zone_certificate",
            _deep_freeze(dict(self.gray_zone_certificate)),
        )
        object.__setattr__(self, "generator_residuals", generator_residuals)
        object.__setattr__(self, "hermitian_residual", hermitian_residual)
        object.__setattr__(self, "propagated_error_bounds", propagated_error_bounds)
        object.__setattr__(
            self,
            "certification_metadata",
            _deep_freeze(dict(self.certification_metadata)),
        )


def _input_owner_indices(values: object) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("logical owner indices must be an integer sequence") from error
    indices = tuple(
        _nonnegative_integer(value, name="logical owner index")
        for value in raw_values
    )
    if not indices:
        raise ValueError("logical owner indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("logical owner indices contain duplicate values")
    return indices


def _real_square_action(
    value: object,
    *,
    dimension: int,
    name: str,
) -> np.ndarray | sparse.csc_matrix:
    if sparse.issparse(value):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        matrix = sparse.csc_matrix(value, dtype=np.float64, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if matrix.shape != (dimension, dimension):
            raise ValueError(
                f"{name} has shape {matrix.shape}; expected {(dimension, dimension)}"
            )
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"{name} contains non-finite values")
        return matrix
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} has shape {matrix.shape}; expected {(dimension, dimension)}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return np.array(matrix, dtype=np.float64, copy=True, order="C")


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


def _action_nonzero_pairs(
    action: np.ndarray | sparse.csc_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(action):
        coo = action.tocoo(copy=False)
        return (
            np.asarray(coo.row, dtype=np.int64),
            np.asarray(coo.col, dtype=np.int64),
        )
    return tuple(np.asarray(values, dtype=np.int64) for values in np.nonzero(action))  # type: ignore[return-value]


def _action_nnz(action: np.ndarray | sparse.csc_matrix) -> int:
    if sparse.issparse(action):
        return int(action.nnz)
    return int(np.count_nonzero(action))


def build_exact_joint_components(
    ambient_columns: np.ndarray | sparse.spmatrix,
    *,
    nominal_degrees: Sequence[int],
    logical_owner_indices: Sequence[int],
    provenance_by_owner: Mapping[int, Any],
    generator_actions: Mapping[str, object],
    generator_error_bounds: Mapping[str, float],
    antiunitary_parities: Mapping[str, bool],
    hermitian_action: object,
    column_absolute_error_bounds: Sequence[float],
) -> tuple[LocalVocabularyComponent, ...]:
    """Build exact joint components from action, adjoint, and ambient support.

    Input owner order is allowed to vary.  Every returned component uses the
    canonical increasing owner order, and components are ordered by their
    smallest owner ID.  Structural edges use exact stored nonzeros; no numerical
    threshold is used to infer route structure.
    """

    owners_input = _input_owner_indices(logical_owner_indices)
    dimension = len(owners_input)
    ambient = _real_sparse_csc(ambient_columns, name="ambient_columns")
    if ambient.shape[1] != dimension:
        raise ValueError(
            "ambient_columns column count must match logical owner indices"
        )

    try:
        degrees_input = tuple(nominal_degrees)
    except TypeError as error:
        raise ValueError("nominal degrees must be an integer sequence") from error
    degrees = tuple(
        _nonnegative_integer(value, name="nominal degree")
        for value in degrees_input
    )
    if len(degrees) != dimension:
        raise ValueError("nominal degree count must match logical owner count")

    if not isinstance(provenance_by_owner, Mapping):
        raise ValueError("provenance_by_owner must be a mapping")
    if set(provenance_by_owner) != set(owners_input):
        raise ValueError("provenance keys must exactly match logical owner indices")

    if not isinstance(generator_actions, Mapping):
        raise ValueError("generator_actions must be a mapping")
    raw_generator_names = tuple(generator_actions.keys())
    if any(
        not isinstance(name, str) or not name
        for name in raw_generator_names
    ):
        raise ValueError("generator_actions keys must be non-empty generator names")
    if len(set(raw_generator_names)) != len(raw_generator_names):
        raise ValueError("generator_actions contain duplicate generator names")
    generator_names = tuple(sorted(raw_generator_names))
    if not generator_names:
        raise ValueError("generator_actions must not be empty")
    if set(generator_error_bounds) != set(generator_names):
        raise ValueError("generator error-bound keys must exactly match generator keys")
    if set(antiunitary_parities) != set(generator_names):
        raise ValueError("antiunitary parity keys must exactly match generator keys")

    actions_input: dict[str, np.ndarray | sparse.csc_matrix] = {}
    error_bounds: dict[str, float] = {}
    parities: dict[str, bool] = {}
    source_nnz: dict[str, int] = {}
    for name in generator_names:
        action = _real_square_action(
            generator_actions[name],
            dimension=dimension,
            name=f"generator action {name!r}",
        )
        actions_input[name] = action
        source_nnz[name] = _action_nnz(action)
        error_bounds[name] = _nonnegative_scalar(
            generator_error_bounds[name],
            name=f"generator error bound {name!r}",
        )
        parity = antiunitary_parities[name]
        if not isinstance(parity, (bool, np.bool_)):
            raise ValueError(f"antiunitary parity for {name!r} must be boolean")
        parities[name] = bool(parity)

    hermitian_input = _real_square_action(
        hermitian_action,
        dimension=dimension,
        name="hermitian_action",
    )
    column_bounds_input = _real_vector(
        column_absolute_error_bounds,
        name="column_absolute_error_bounds",
    )
    if column_bounds_input.shape != (dimension,):
        raise ValueError(
            "column_absolute_error_bounds must have shape "
            f"{(dimension,)}"
        )

    canonical_order = np.asarray(
        sorted(range(dimension), key=lambda index: owners_input[index]),
        dtype=np.int64,
    )
    owners = tuple(owners_input[int(index)] for index in canonical_order)
    degrees = tuple(degrees[int(index)] for index in canonical_order)
    ambient = sparse.csc_matrix(ambient[:, canonical_order], copy=True)
    actions = {
        name: _reordered_action(actions_input[name], canonical_order)
        for name in generator_names
    }
    hermitian = _reordered_action(hermitian_input, canonical_order)
    column_bounds = np.asarray(column_bounds_input[canonical_order], dtype=np.float64)
    provenance = {
        owner: provenance_by_owner[owner]
        for owner in owners
    }

    parent = list(range(dimension))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if owners[root_left] <= owners[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    first_column_by_row: dict[int, int] = {}
    for column in range(dimension):
        start = int(ambient.indptr[column])
        stop = int(ambient.indptr[column + 1])
        for row_value in ambient.indices[start:stop]:
            row = int(row_value)
            previous = first_column_by_row.setdefault(row, column)
            union(previous, column)

    for action in tuple(actions.values()) + (hermitian,):
        rows, columns = _action_nonzero_pairs(action)
        for row, column in zip(rows.tolist(), columns.tolist()):
            union(int(row), int(column))

    members_by_root: dict[int, list[int]] = {}
    for position in range(dimension):
        members_by_root.setdefault(find(position), []).append(position)
    ordered_members = sorted(
        (tuple(sorted(values, key=lambda position: owners[position])) for values in members_by_root.values()),
        key=lambda values: owners[values[0]],
    )

    components: list[LocalVocabularyComponent] = []
    for component_index, member_tuple in enumerate(ordered_members):
        positions = np.asarray(member_tuple, dtype=np.int64)
        component_ambient = sparse.csc_matrix(ambient[:, positions], copy=True)
        support_rows = tuple(
            int(value) for value in np.unique(component_ambient.indices)
        )
        component_owners = tuple(owners[int(position)] for position in positions)
        component_actions = {
            name: _restricted_dense_action(actions[name], positions)
            for name in generator_names
        }
        component_hermitian = _restricted_dense_action(hermitian, positions)

        outside = np.setdiff1d(
            np.arange(dimension, dtype=np.int64),
            positions,
            assume_unique=True,
        )
        for name, action in tuple(actions.items()) + (("__hermitian__", hermitian),):
            if outside.size == 0:
                continue
            if sparse.issparse(action):
                leakage = action[outside, :][:, positions]
                leakage_nnz = int(leakage.nnz)
            else:
                leakage_nnz = int(np.count_nonzero(action[np.ix_(outside, positions)]))
            if leakage_nnz:
                raise LocalGeneratorNullspaceUnavailable(
                    "component_action_leakage",
                    certificate={
                        "component_index": component_index,
                        "action": name,
                        "outside_nonzeros": leakage_nnz,
                    },
                )

        components.append(
            LocalVocabularyComponent(
                component_index=component_index,
                nominal_degrees=tuple(degrees[int(position)] for position in positions),
                logical_owner_indices=component_owners,
                provenance_by_owner={
                    owner: provenance[owner]
                    for owner in component_owners
                },
                ambient_columns=component_ambient,
                generator_actions=component_actions,
                generator_error_bounds={
                    name: error_bounds[name]
                    for name in generator_names
                },
                antiunitary_parities={
                    name: parities[name]
                    for name in generator_names
                },
                hermitian_action=component_hermitian,
                exact_support_rows=support_rows,
                column_absolute_error_bounds=column_bounds[positions],
                source_generator_action_nnz={
                    name: source_nnz[name]
                    for name in generator_names
                },
            )
        )
    return tuple(components)


def _dense_gram(matrix: sparse.csc_matrix) -> np.ndarray:
    product = matrix.T @ matrix
    gram = np.asarray(product.toarray(), dtype=np.float64)
    return 0.5 * (gram + gram.T)


def _gamma_n(dimension: int) -> float:
    epsilon = np.finfo(np.float64).eps
    product = float(max(1, dimension)) * epsilon
    if product >= 1.0:
        return float("inf")
    return float(product / (1.0 - product))


def _certified_independent_positions(
    gram: np.ndarray,
    column_error_bounds: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    gray_zone_factor: float,
    component_index: int,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, float]:
    diagonal = np.maximum(0.0, np.diag(gram))
    norms = np.sqrt(diagonal)
    scale = max(1.0, float(np.max(norms)) if norms.size else 0.0)
    normalized = np.zeros_like(gram)
    nonzero = norms > 0.0
    if np.any(nonzero):
        normalized[np.ix_(nonzero, nonzero)] = (
            gram[np.ix_(nonzero, nonzero)]
            / norms[nonzero, None]
            / norms[None, nonzero]
        )
    normalized = 0.5 * (normalized + normalized.T)
    normalized_scale = max(1.0, float(np.linalg.norm(normalized, ord=2)))
    aggregate_relative_error = float(
        np.linalg.norm(
            np.divide(
                column_error_bounds,
                np.maximum(norms, np.finfo(np.float64).tiny),
            )
        )
    )
    rank_threshold = float(
        aggregate_relative_error
        + absolute_tolerance / scale
        + relative_tolerance * normalized_scale
        + _gamma_n(gram.shape[0]) * normalized_scale
    )
    rank_threshold = max(rank_threshold, np.finfo(np.float64).eps)

    selected: list[int] = []
    for position in range(gram.shape[0]):
        column_threshold = float(
            column_error_bounds[position]
            + absolute_tolerance
            + relative_tolerance * max(1.0, norms[position])
            + _gamma_n(gram.shape[0]) * max(1.0, norms[position])
        )
        if norms[position] <= column_threshold:
            continue
        if norms[position] < gray_zone_factor * column_threshold:
            raise LocalGeneratorNullspaceUnavailable(
                "vocabulary_rank_gray_zone",
                certificate={
                    "component_index": component_index,
                    "owner_position": position,
                    "column_norm": float(norms[position]),
                    "threshold": column_threshold,
                    "gray_zone_upper": gray_zone_factor * column_threshold,
                },
            )
        if not selected:
            residual = 1.0
        else:
            selected_array = np.asarray(selected, dtype=np.int64)
            selected_gram = normalized[np.ix_(selected_array, selected_array)]
            cross = normalized[selected_array, position]
            try:
                coefficients = scipy.linalg.solve(
                    selected_gram,
                    cross,
                    assume_a="sym",
                    check_finite=False,
                )
            except scipy.linalg.LinAlgError as error:
                raise LocalGeneratorNullspaceUnavailable(
                    "vocabulary_rank_factorization_failed",
                    certificate={"component_index": component_index},
                ) from error
            residual_squared = float(1.0 - cross @ coefficients)
            roundoff_floor = float(
                16.0
                * _gamma_n(len(selected) + 1)
                * max(1.0, float(np.linalg.cond(selected_gram)))
            )
            if residual_squared <= roundoff_floor:
                residual = 0.0
            else:
                residual = float(np.sqrt(max(0.0, residual_squared)))
        if residual <= rank_threshold:
            continue
        if residual < gray_zone_factor * rank_threshold:
            raise LocalGeneratorNullspaceUnavailable(
                "vocabulary_rank_gray_zone",
                certificate={
                    "component_index": component_index,
                    "owner_position": position,
                    "residual": residual,
                    "threshold": rank_threshold,
                    "gray_zone_upper": gray_zone_factor * rank_threshold,
                },
            )
        selected.append(position)

    if not selected:
        raise LocalGeneratorNullspaceUnavailable(
            "empty_independent_vocabulary",
            certificate={
                "component_index": component_index,
                "column_norms": tuple(float(value) for value in norms),
            },
        )
    return tuple(selected), norms, normalized, rank_threshold


def _action_on_independent_vocabulary(
    *,
    gram: np.ndarray,
    action: np.ndarray,
    selected_positions: tuple[int, ...],
    selected_norms: np.ndarray,
    normalized_selected_gram: np.ndarray,
    input_error_bound: float,
    column_error_bounds: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
    gray_zone_factor: float,
    component_index: int,
    action_name: str,
) -> tuple[np.ndarray, float, float, float]:
    selected = np.asarray(selected_positions, dtype=np.int64)
    scales = 1.0 / selected_norms
    image_coefficients = np.asarray(action[:, selected], dtype=np.float64)
    cross = gram[np.ix_(selected, np.arange(gram.shape[0]))] @ image_coefficients
    cross = scales[:, None] * cross * scales[None, :]
    image_gram = image_coefficients.T @ gram @ image_coefficients
    image_gram = scales[:, None] * image_gram * scales[None, :]
    try:
        coordinates = scipy.linalg.solve(
            normalized_selected_gram,
            cross,
            assume_a="sym",
            check_finite=False,
        )
    except scipy.linalg.LinAlgError as error:
        raise LocalGeneratorNullspaceUnavailable(
            "local_action_coordinate_solve_failed",
            certificate={
                "component_index": component_index,
                "action": action_name,
            },
        ) from error
    residual_gram = image_gram - cross.T @ coordinates
    residual_gram = 0.5 * (residual_gram + residual_gram.T)
    residual_eigenvalues = scipy.linalg.eigvalsh(residual_gram, check_finite=False)
    maximum_residual_eigenvalue = (
        max(0.0, float(residual_eigenvalues[-1]))
        if residual_eigenvalues.size
        else 0.0
    )
    residual_roundoff_floor = float(
        32.0
        * _gamma_n(max(1, gram.shape[0]))
        * max(
            1.0,
            float(np.linalg.norm(image_gram, ord=2)),
            float(np.linalg.norm(cross.T @ coordinates, ord=2)),
        )
    )
    residual_norm = (
        0.0
        if maximum_residual_eigenvalue <= residual_roundoff_floor
        else float(np.sqrt(maximum_residual_eigenvalue))
    )
    image_scale = max(
        1.0,
        float(np.sqrt(max(0.0, float(np.trace(image_gram))))),
    )
    propagated_columns = float(
        np.linalg.norm(column_error_bounds)
        * max(1.0, float(np.linalg.norm(image_coefficients, ord=2)))
        * max(1.0, float(np.max(scales)))
    )
    closure_bound = float(
        input_error_bound
        + propagated_columns
        + absolute_tolerance
        + relative_tolerance * image_scale
        + _gamma_n(gram.shape[0]) * image_scale
    )
    if residual_norm > closure_bound:
        raise LocalGeneratorNullspaceUnavailable(
            "generator_span_leakage",
            certificate={
                "component_index": component_index,
                "action": action_name,
                "outside_span_residual": residual_norm,
                "certification_bound": closure_bound,
            },
        )

    try:
        lower = scipy.linalg.cholesky(
            normalized_selected_gram,
            lower=True,
            check_finite=False,
        )
    except scipy.linalg.LinAlgError as error:
        raise LocalGeneratorNullspaceUnavailable(
            "vocabulary_whitening_failed",
            certificate={"component_index": component_index},
        ) from error
    inverse_lower_transpose = scipy.linalg.solve_triangular(
        lower.T,
        np.eye(lower.shape[0], dtype=np.float64),
        lower=False,
        check_finite=False,
    )
    whitened = lower.T @ coordinates @ inverse_lower_transpose
    orthogonality_residual = float(
        np.linalg.norm(whitened.T @ whitened - np.eye(whitened.shape[0]), ord=2)
    )
    representation_bound = float(
        closure_bound / image_scale
        + absolute_tolerance
        + relative_tolerance * max(1.0, float(np.linalg.norm(whitened, ord=2)))
        + _gamma_n(whitened.shape[0])
        * max(1.0, float(np.linalg.norm(whitened, ord=2)))
    )
    if orthogonality_residual > gray_zone_factor * representation_bound:
        raise LocalGeneratorNullspaceUnavailable(
            "generator_representation_not_orthogonal",
            certificate={
                "component_index": component_index,
                "action": action_name,
                "orthogonality_residual": orthogonality_residual,
                "certification_bound": representation_bound,
            },
        )
    return whitened, residual_norm, representation_bound, closure_bound


def _deterministic_logical_owners(
    logical_projection_coordinates: np.ndarray,
    owner_indices: tuple[int, ...],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    gray_zone_factor: float,
    component_index: int,
) -> tuple[int, ...]:
    rank = int(logical_projection_coordinates.shape[0])
    if rank == 0:
        return ()
    selected_positions: list[int] = []
    orthonormal: list[np.ndarray] = []
    scale = max(
        1.0,
        float(np.linalg.norm(logical_projection_coordinates, ord=2)),
    )
    threshold = float(
        absolute_tolerance
        + relative_tolerance * scale
        + _gamma_n(logical_projection_coordinates.shape[1]) * scale
    )
    threshold = max(threshold, np.finfo(np.float64).eps)
    for position in range(logical_projection_coordinates.shape[1]):
        vector = np.asarray(
            logical_projection_coordinates[:, position],
            dtype=np.float64,
        ).copy()
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
            raise LocalGeneratorNullspaceUnavailable(
                "ambiguous_owner_pivots",
                certificate={
                    "component_index": component_index,
                    "owner": owner_indices[position],
                    "residual": residual,
                    "threshold": threshold,
                    "gray_zone_upper": gray_zone_factor * threshold,
                },
            )
        vector /= residual
        orthonormal.append(vector)
        selected_positions.append(position)
        if len(selected_positions) == rank:
            break
    if len(selected_positions) != rank:
        raise LocalGeneratorNullspaceUnavailable(
            "insufficient_logical_owner_rank",
            certificate={
                "component_index": component_index,
                "expected_rank": rank,
                "selected_rank": len(selected_positions),
            },
        )
    return tuple(owner_indices[position] for position in selected_positions)


def solve_local_fixed_component(
    component: LocalVocabularyComponent,
    *,
    absolute_tolerance: float = LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1,
    relative_tolerance: float = LOCAL_FIXED_RELATIVE_TOLERANCE_V1,
    gray_zone_factor: float = LOCAL_FIXED_GRAY_ZONE_FACTOR_V1,
    dense_component_cutoff: int = LOCAL_FIXED_DENSE_COMPONENT_CUTOFF_V1,
) -> LocalFixedCompilation:
    """Solve one certified local generator/Hermitian fixed-space problem."""

    if not isinstance(component, LocalVocabularyComponent):
        raise TypeError("component must be a LocalVocabularyComponent")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("local fixed tolerances must be non-negative")
    if gray_zone_factor <= 1.0:
        raise ValueError("gray_zone_factor must exceed one")
    cutoff = _nonnegative_integer(
        dense_component_cutoff,
        name="dense_component_cutoff",
    )
    raw_dimension = len(component.logical_owner_indices)
    if raw_dimension > cutoff:
        raise LocalGeneratorNullspaceUnavailable(
            "oversize_component",
            certificate={
                "component_index": component.component_index,
                "raw_dimension": raw_dimension,
                "dense_component_cutoff": cutoff,
            },
        )

    gram = _dense_gram(component.ambient_columns)
    selected_positions, norms, normalized_gram, rank_threshold = (
        _certified_independent_positions(
            gram,
            np.asarray(component.column_absolute_error_bounds, dtype=np.float64),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            gray_zone_factor=gray_zone_factor,
            component_index=component.component_index,
        )
    )
    selected = np.asarray(selected_positions, dtype=np.int64)
    selected_norms = norms[selected]
    selected_normalized_gram = normalized_gram[np.ix_(selected, selected)]
    vocabulary_eigenvalues = scipy.linalg.eigvalsh(gram, check_finite=False)
    vocabulary_singular_values = np.sqrt(
        np.maximum(0.0, vocabulary_eigenvalues[::-1])
    )

    whitened_actions: dict[str, np.ndarray] = {}
    generator_residuals: dict[str, float] = {}
    propagated_error_bounds: dict[str, float] = {}
    closure_bounds: dict[str, float] = {}
    for name in sorted(component.generator_actions):
        whitened, residual, representation_bound, closure_bound = (
            _action_on_independent_vocabulary(
                gram=gram,
                action=np.asarray(component.generator_actions[name], dtype=np.float64),
                selected_positions=selected_positions,
                selected_norms=selected_norms,
                normalized_selected_gram=selected_normalized_gram,
                input_error_bound=float(component.generator_error_bounds[name]),
                column_error_bounds=np.asarray(
                    component.column_absolute_error_bounds,
                    dtype=np.float64,
                ),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                gray_zone_factor=gray_zone_factor,
                component_index=component.component_index,
                action_name=name,
            )
        )
        whitened_actions[name] = whitened
        generator_residuals[name] = residual
        propagated_error_bounds[name] = representation_bound
        closure_bounds[name] = closure_bound

    hermitian_whitened, hermitian_span_residual, hermitian_bound, hermitian_closure_bound = (
        _action_on_independent_vocabulary(
            gram=gram,
            action=np.asarray(component.hermitian_action, dtype=np.float64),
            selected_positions=selected_positions,
            selected_norms=selected_norms,
            normalized_selected_gram=selected_normalized_gram,
            input_error_bound=0.0,
            column_error_bounds=np.asarray(
                component.column_absolute_error_bounds,
                dtype=np.float64,
            ),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            gray_zone_factor=gray_zone_factor,
            component_index=component.component_index,
            action_name="__hermitian__",
        )
    )

    actions = [
        RealGeneratorAction(
            name,
            whitened_actions[name],
            antiunitary=bool(component.antiunitary_parities[name]),
        )
        for name in sorted(whitened_actions)
    ]
    actions.append(
        RealGeneratorAction(
            "__hermitian__",
            hermitian_whitened,
            antiunitary=True,
        )
    )
    action_error_terms = tuple(propagated_error_bounds.values()) + (
        hermitian_bound,
    )
    delta_k = float(np.sqrt(sum(value * value for value in action_error_terms)))
    gamma = _gamma_n(len(selected_positions))
    try:
        fixed = solve_generator_fixed_subspace(
            actions,
            absolute_tolerance=delta_k + absolute_tolerance,
            relative_tolerance=gamma + relative_tolerance,
            action_absolute_error_bound=0.0,
        )
    except FixedSubspaceCertificationError as error:
        reason = (
            "fixed_rank_gray_zone"
            if "gray zone" in str(error)
            else "fixed_subspace_certification_failed"
        )
        raise LocalGeneratorNullspaceUnavailable(
            reason,
            certificate={
                "component_index": component.component_index,
                "message": str(error),
                "delta_k": delta_k,
                "gamma_n": gamma,
            },
        ) from error

    try:
        lower = scipy.linalg.cholesky(
            selected_normalized_gram,
            lower=True,
            check_finite=False,
        )
    except scipy.linalg.LinAlgError as error:
        raise LocalGeneratorNullspaceUnavailable(
            "vocabulary_whitening_failed",
            certificate={"component_index": component.component_index},
        ) from error
    inverse_lower_transpose = scipy.linalg.solve_triangular(
        lower.T,
        np.eye(lower.shape[0], dtype=np.float64),
        lower=False,
        check_finite=False,
    )
    selected_to_raw = np.zeros(
        (raw_dimension, len(selected_positions)),
        dtype=np.float64,
    )
    selected_to_raw[selected, :] = (
        np.diag(1.0 / selected_norms) @ inverse_lower_transpose
    )
    orthonormal_fixed_coordinates = selected_to_raw @ fixed.basis
    physical_gram = (
        orthonormal_fixed_coordinates.T
        @ gram
        @ orthonormal_fixed_coordinates
    )
    orthogonality_residual = float(
        np.linalg.norm(physical_gram - np.eye(fixed.rank), ord=2)
    )
    orthogonality_bound = float(
        absolute_tolerance
        + relative_tolerance * max(1.0, float(fixed.rank))
        + _gamma_n(raw_dimension) * max(1.0, float(fixed.rank))
    )
    if orthogonality_residual > gray_zone_factor * orthogonality_bound:
        raise LocalGeneratorNullspaceUnavailable(
            "fixed_basis_not_orthonormal",
            certificate={
                "component_index": component.component_index,
                "residual": orthogonality_residual,
                "bound": orthogonality_bound,
            },
        )

    selected_cross_all = (
        (1.0 / selected_norms)[:, None]
        * gram[np.ix_(selected, np.arange(raw_dimension))]
    )
    whitened_cross_all = scipy.linalg.solve_triangular(
        lower,
        selected_cross_all,
        lower=True,
        check_finite=False,
    )
    orthonormal_logical_projection = fixed.basis.T @ whitened_cross_all
    selected_owner_indices = _deterministic_logical_owners(
        orthonormal_logical_projection,
        component.logical_owner_indices,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        gray_zone_factor=gray_zone_factor,
        component_index=component.component_index,
    )
    owner_position_by_id = {
        owner: position
        for position, owner in enumerate(component.logical_owner_indices)
    }
    selected_owner_positions = np.asarray(
        [owner_position_by_id[owner] for owner in selected_owner_indices],
        dtype=np.int64,
    )
    owner_projection_frame = orthonormal_logical_projection[
        :, selected_owner_positions
    ]
    owner_projection_singular_values = scipy.linalg.svdvals(
        owner_projection_frame,
        check_finite=False,
    )
    owner_projection_scale = max(
        1.0,
        float(owner_projection_singular_values[0])
        if owner_projection_singular_values.size
        else 0.0,
    )
    owner_projection_threshold = float(
        absolute_tolerance
        + relative_tolerance * owner_projection_scale
        + _gamma_n(max(1, fixed.rank)) * owner_projection_scale
    )
    if (
        owner_projection_singular_values.size
        and float(owner_projection_singular_values[-1])
        <= owner_projection_threshold
    ):
        raise LocalGeneratorNullspaceUnavailable(
            "ambiguous_owner_projection_frame",
            certificate={
                "component_index": component.component_index,
                "smallest_singular_value": float(
                    owner_projection_singular_values[-1]
                ),
                "threshold": owner_projection_threshold,
            },
        )
    fixed_coordinates = (
        orthonormal_fixed_coordinates @ owner_projection_frame
    )
    logical_projection = fixed_coordinates.T @ gram
    public_physical_gram = fixed_coordinates.T @ gram @ fixed_coordinates

    constraint_singular_values = np.zeros(raw_dimension, dtype=np.float64)
    fixed_singular_values = np.asarray(fixed.singular_values, dtype=np.float64)
    constraint_singular_values[: fixed_singular_values.size] = fixed_singular_values
    fixed_residuals = {
        name: float(
            np.linalg.norm(whitened_actions[name] @ fixed.basis - fixed.basis, ord=2)
        )
        for name in sorted(whitened_actions)
    }
    hermitian_residual = float(
        np.linalg.norm(hermitian_whitened @ fixed.basis - fixed.basis, ord=2)
    )
    generator_residuals = {
        name: max(generator_residuals[name], fixed_residuals[name])
        for name in sorted(generator_residuals)
    }

    fixed_threshold = float(
        delta_k
        + absolute_tolerance
        + (gamma + relative_tolerance)
        * max(
            1.0,
            float(np.max(fixed_singular_values))
            if fixed_singular_values.size
            else 0.0,
        )
    )
    metadata = {
        "algorithm": LOCAL_FIXED_ALGORITHM_V1,
        "component_policy": LOCAL_FIXED_COMPONENT_POLICY_V1,
        "column_scaling_policy": LOCAL_FIXED_COLUMN_SCALING_V1,
        "owner_policy": LOCAL_FIXED_OWNER_POLICY_V1,
        "raw_dimension": raw_dimension,
        "independent_dimension": len(selected_positions),
        "independent_owner_indices": tuple(
            component.logical_owner_indices[position]
            for position in selected_positions
        ),
        "vocabulary_singular_values": tuple(
            float(value) for value in vocabulary_singular_values
        ),
        "vocabulary_rank_threshold": rank_threshold,
        "generator_action_nnz": dict(component.source_generator_action_nnz),
        "generator_closure_bounds": closure_bounds,
        "hermitian_span_residual": hermitian_span_residual,
        "hermitian_closure_bound": hermitian_closure_bound,
        "internal_fixed_orthogonality_residual": orthogonality_residual,
        "internal_fixed_orthogonality_bound": orthogonality_bound,
        "owner_projection_singular_values": tuple(
            float(value) for value in owner_projection_singular_values
        ),
        "owner_projection_threshold": owner_projection_threshold,
        "public_fixed_gram": public_physical_gram,
        "fixed_subspace": dict(fixed.metadata),
        "dense_component_cutoff": cutoff,
    }
    return LocalFixedCompilation(
        component_index=component.component_index,
        component_owner_indices=component.logical_owner_indices,
        fixed_vocabulary_coordinates=fixed_coordinates,
        logical_projection_coordinates=logical_projection,
        fixed_rank=int(fixed.rank),
        selected_global_owner_indices=selected_owner_indices,
        singular_values=constraint_singular_values,
        gray_zone_certificate={
            "threshold": fixed_threshold,
            "gray_zone_upper": gray_zone_factor * fixed_threshold,
            "gray_zone_factor": gray_zone_factor,
            "delta_k": delta_k,
            "gamma_n": gamma,
        },
        generator_residuals=generator_residuals,
        hermitian_residual=max(hermitian_span_residual, hermitian_residual),
        propagated_error_bounds=propagated_error_bounds,
        certification_metadata=metadata,
    )


def _certified_action_graph_blocks(
    component: LocalVocabularyComponent,
) -> tuple[tuple[int, ...], ...]:
    """Return exact generator/H support components inside one ambient component."""

    dimension = len(component.logical_owner_indices)
    adjacency = sparse.csr_matrix((dimension, dimension), dtype=np.float64)
    for raw_action in (*component.generator_actions.values(), component.hermitian_action):
        action = sparse.csr_matrix(raw_action, dtype=np.float64)
        action.sum_duplicates()
        action.eliminate_zeros()
        action.data = np.ones(action.nnz, dtype=np.float64)
        adjacency = (adjacency + action + action.T).tocsr()
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    component_count, labels = sparse.csgraph.connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    blocks = tuple(
        tuple(
            sorted(
                np.flatnonzero(labels == label).tolist(),
                key=lambda position: component.logical_owner_indices[position],
            )
        )
        for label in range(int(component_count))
    )
    return tuple(
        sorted(
            blocks,
            key=lambda positions: component.logical_owner_indices[positions[0]],
        )
    )


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
    unknown = sorted(
        {
            name
            for word in words
            for name in word
            if name not in generator_names
        }
    )
    if unknown:
        raise LocalGeneratorNullspaceUnavailable(
            "group_word_missing_generator_action",
            certificate={"missing_generators": tuple(unknown)},
        )
    return words


def _owner_word_action(
    generator_actions: Mapping[str, np.ndarray],
    word: tuple[str, ...],
    *,
    dimension: int,
) -> np.ndarray:
    """Compose one canonical group word in the real owner representation."""

    result = np.eye(dimension, dtype=np.float64)
    for name in word:
        # Finite-group words append the next outer generator.  Since the raw
        # owner actions are already real-linear (including antiunitaries), the
        # corresponding coordinate action composes by ordinary left products.
        result = np.asarray(generator_actions[name], dtype=np.float64) @ result
    return result


def _solve_local_reynolds_block(
    component: LocalVocabularyComponent,
    *,
    group_words: tuple[tuple[str, ...], ...],
    absolute_tolerance: float,
    relative_tolerance: float,
    gray_zone_factor: float,
) -> LocalFixedCompilation:
    dimension = len(component.logical_owner_indices)
    generator_actions = {
        name: np.asarray(action, dtype=np.float64)
        for name, action in component.generator_actions.items()
    }
    word_actions = tuple(
        _owner_word_action(
            generator_actions,
            word,
            dimension=dimension,
        )
        for word in group_words
    )
    group_projector = sum(word_actions, np.zeros((dimension, dimension))) / float(
        len(word_actions)
    )
    hermitian_action = np.asarray(component.hermitian_action, dtype=np.float64)
    hermitian_projector = 0.5 * (
        np.eye(dimension, dtype=np.float64) + hermitian_action
    )
    projector = hermitian_projector @ group_projector

    maximum_word_length = max((len(word) for word in group_words), default=0)
    action_scale = max(
        1.0,
        *(float(np.linalg.norm(action, ord=2)) for action in word_actions),
        float(np.linalg.norm(hermitian_action, ord=2)),
    )
    input_action_error = max(
        (float(value) for value in component.generator_error_bounds.values()),
        default=0.0,
    )
    certification_bound = float(
        absolute_tolerance
        + relative_tolerance * action_scale
        + (maximum_word_length + 2) * input_action_error * action_scale
        + 32.0 * _gamma_n(max(1, dimension * max(1, maximum_word_length)))
        * action_scale
    )
    certification_limit = gray_zone_factor * max(
        certification_bound,
        np.finfo(np.float64).eps,
    )
    group_idempotence_residual = float(
        np.linalg.norm(group_projector @ group_projector - group_projector, ord=2)
    )
    hermitian_involution_residual = float(
        np.linalg.norm(
            hermitian_action @ hermitian_action - np.eye(dimension),
            ord=2,
        )
    )
    commutator_residual = float(
        np.linalg.norm(
            hermitian_action @ group_projector
            - group_projector @ hermitian_action,
            ord=2,
        )
    )
    projector_idempotence_residual = float(
        np.linalg.norm(projector @ projector - projector, ord=2)
    )
    generator_projector_residuals = {
        name: float(np.linalg.norm(action @ group_projector - group_projector, ord=2))
        for name, action in generator_actions.items()
    }
    certification_residuals = {
        "group_projector_idempotence": group_idempotence_residual,
        "hermitian_involution": hermitian_involution_residual,
        "hermitian_group_commutator": commutator_residual,
        "combined_projector_idempotence": projector_idempotence_residual,
        **{
            f"generator_fixed:{name}": value
            for name, value in generator_projector_residuals.items()
        },
    }
    failed_residuals = {
        name: value
        for name, value in certification_residuals.items()
        if value > certification_limit
    }
    if failed_residuals:
        raise LocalGeneratorNullspaceUnavailable(
            "local_reynolds_projector_certification_failed",
            certificate={
                "component_index": component.component_index,
                "certification_bound": certification_bound,
                "certification_limit": certification_limit,
                "failed_residuals": failed_residuals,
            },
        )

    left_vectors, singular_values, _right_vectors = scipy.linalg.svd(
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
    fixed_rank = int(np.count_nonzero(singular_values > rank_threshold))
    ambiguous = singular_values[
        (singular_values > rank_threshold)
        & (singular_values < gray_zone_factor * rank_threshold)
    ]
    if ambiguous.size:
        raise LocalGeneratorNullspaceUnavailable(
            "local_reynolds_rank_gray_zone",
            certificate={
                "component_index": component.component_index,
                "rank_threshold": rank_threshold,
                "gray_zone_upper": gray_zone_factor * rank_threshold,
                "ambiguous_singular_values": tuple(float(value) for value in ambiguous),
            },
        )

    if fixed_rank:
        fixed_left = left_vectors[:, :fixed_rank]
        logical_images = fixed_left.T @ projector
        selected_owner_indices = _deterministic_logical_owners(
            logical_images,
            component.logical_owner_indices,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            gray_zone_factor=gray_zone_factor,
            component_index=component.component_index,
        )
        owner_position = {
            owner: position
            for position, owner in enumerate(component.logical_owner_indices)
        }
        selected_positions = np.asarray(
            [owner_position[owner] for owner in selected_owner_indices],
            dtype=np.int64,
        )
        fixed_coordinates = np.asarray(
            projector[:, selected_positions],
            dtype=np.float64,
        )
        logical_projection, _residuals, _rank, _values = scipy.linalg.lstsq(
            fixed_coordinates,
            projector,
            cond=rank_threshold,
            check_finite=False,
            lapack_driver="gelsy",
        )
        logical_projection = np.asarray(logical_projection, dtype=np.float64)
    else:
        selected_owner_indices = ()
        fixed_coordinates = np.zeros((dimension, 0), dtype=np.float64)
        logical_projection = np.zeros((0, dimension), dtype=np.float64)

    reconstruction_residual = float(
        np.linalg.norm(fixed_coordinates @ logical_projection - projector, ord=2)
    )
    if reconstruction_residual > certification_limit:
        raise LocalGeneratorNullspaceUnavailable(
            "local_reynolds_owner_frame_certification_failed",
            certificate={
                "component_index": component.component_index,
                "reconstruction_residual": reconstruction_residual,
                "certification_limit": certification_limit,
            },
        )

    generator_residuals = {
        name: float(np.linalg.norm(action @ fixed_coordinates - fixed_coordinates, ord=2))
        for name, action in generator_actions.items()
    }
    hermitian_residual = float(
        np.linalg.norm(
            hermitian_action @ fixed_coordinates - fixed_coordinates,
            ord=2,
        )
    )
    propagated_error_bounds = {
        name: certification_bound for name in generator_actions
    }
    padded_singular_values = np.zeros(dimension, dtype=np.float64)
    padded_singular_values[: singular_values.size] = singular_values
    return LocalFixedCompilation(
        component_index=component.component_index,
        component_owner_indices=component.logical_owner_indices,
        fixed_vocabulary_coordinates=fixed_coordinates,
        logical_projection_coordinates=logical_projection,
        fixed_rank=fixed_rank,
        selected_global_owner_indices=selected_owner_indices,
        singular_values=padded_singular_values,
        gray_zone_certificate={
            "threshold": rank_threshold,
            "gray_zone_upper": gray_zone_factor * rank_threshold,
            "gray_zone_factor": gray_zone_factor,
        },
        generator_residuals=generator_residuals,
        hermitian_residual=hermitian_residual,
        propagated_error_bounds=propagated_error_bounds,
        certification_metadata={
            "algorithm": LOCAL_REYNOLDS_ALGORITHM_V1,
            "raw_dimension": dimension,
            "independent_dimension": dimension,
            "fixed_rank": fixed_rank,
            "group_order": len(group_words),
            "maximum_group_word_length": maximum_word_length,
            "projector_singular_values": tuple(float(value) for value in singular_values),
            "projector_rank_threshold": rank_threshold,
            "certification_bound": certification_bound,
            "certification_limit": certification_limit,
            "certification_residuals": certification_residuals,
            "owner_frame_reconstruction_residual": reconstruction_residual,
            "generator_action_nnz": dict(component.source_generator_action_nnz),
            "vocabulary_rank_required": False,
        },
    )


def solve_local_reynolds_component_blocked(
    component: LocalVocabularyComponent,
    *,
    group_words: Sequence[Sequence[str]],
    absolute_tolerance: float = LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1,
    relative_tolerance: float = LOCAL_FIXED_RELATIVE_TOLERANCE_V1,
    gray_zone_factor: float = LOCAL_FIXED_GRAY_ZONE_FACTOR_V1,
) -> LocalFixedCompilation:
    """Project one component by finite-group averaging on exact action blocks."""

    if not isinstance(component, LocalVocabularyComponent):
        raise TypeError("component must be a LocalVocabularyComponent")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("local Reynolds tolerances must be non-negative")
    if gray_zone_factor <= 1.0:
        raise ValueError("gray_zone_factor must exceed one")
    words = _canonical_group_words(
        group_words,
        generator_names=tuple(component.generator_actions),
    )
    blocks = _certified_action_graph_blocks(component)
    raw_dimension = len(component.logical_owner_indices)

    fixed_coordinate_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    logical_projection_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    selected_owner_indices: list[int] = []
    singular_values: list[float] = []
    generator_residuals = {name: 0.0 for name in component.generator_actions}
    propagated_error_bounds = {name: 0.0 for name in component.generator_actions}
    hermitian_residual = 0.0
    block_metadata: list[dict[str, Any]] = []
    thresholds: list[float] = []
    gray_uppers: list[float] = []

    for block_index, block_positions in enumerate(blocks):
        positions = np.asarray(block_positions, dtype=np.int64)
        sub_owners = tuple(
            component.logical_owner_indices[int(position)] for position in positions
        )
        sub_ambient = sparse.csc_matrix(
            component.ambient_columns[:, positions],
            copy=True,
        )
        sub_actions = {
            name: np.asarray(action[np.ix_(positions, positions)], dtype=np.float64)
            for name, action in component.generator_actions.items()
        }
        sub_component = LocalVocabularyComponent(
            component_index=block_index,
            nominal_degrees=tuple(
                component.nominal_degrees[int(position)] for position in positions
            ),
            logical_owner_indices=sub_owners,
            provenance_by_owner={
                owner: component.provenance_by_owner[owner] for owner in sub_owners
            },
            ambient_columns=sub_ambient,
            generator_actions=sub_actions,
            generator_error_bounds=dict(component.generator_error_bounds),
            antiunitary_parities=dict(component.antiunitary_parities),
            hermitian_action=np.asarray(
                component.hermitian_action[np.ix_(positions, positions)],
                dtype=np.float64,
            ),
            exact_support_rows=tuple(int(value) for value in np.unique(sub_ambient.indices)),
            column_absolute_error_bounds=np.asarray(
                component.column_absolute_error_bounds[positions],
                dtype=np.float64,
            ),
            source_generator_action_nnz={
                name: int(np.count_nonzero(action))
                for name, action in sub_actions.items()
            },
        )
        result = _solve_local_reynolds_block(
            sub_component,
            group_words=words,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            gray_zone_factor=gray_zone_factor,
        )
        if result.fixed_rank:
            fixed_coordinate_blocks.append(
                (positions, np.asarray(result.fixed_vocabulary_coordinates))
            )
            logical_projection_blocks.append(
                (positions, np.asarray(result.logical_projection_coordinates))
            )
            selected_owner_indices.extend(result.selected_global_owner_indices)
        singular_values.extend(float(value) for value in result.singular_values)
        thresholds.append(float(result.gray_zone_certificate["threshold"]))
        gray_uppers.append(float(result.gray_zone_certificate["gray_zone_upper"]))
        for name, value in result.generator_residuals.items():
            generator_residuals[name] = max(generator_residuals[name], float(value))
        for name, value in result.propagated_error_bounds.items():
            propagated_error_bounds[name] = max(
                propagated_error_bounds[name], float(value)
            )
        hermitian_residual = max(hermitian_residual, float(result.hermitian_residual))
        block_metadata.append(
            {
                "block_index": block_index,
                "raw_dimension": len(block_positions),
                "fixed_rank": int(result.fixed_rank),
                "owner_min": min(sub_owners),
                "owner_max": max(sub_owners),
                "projector": dict(result.certification_metadata),
            }
        )

    total_rank = sum(block.shape[1] for _positions, block in fixed_coordinate_blocks)
    fixed_coordinates = np.zeros((raw_dimension, total_rank), dtype=np.float64)
    logical_projection = np.zeros((total_rank, raw_dimension), dtype=np.float64)
    column_offset = 0
    for (positions, fixed_block), (_positions, projection_block) in zip(
        fixed_coordinate_blocks,
        logical_projection_blocks,
    ):
        block_rank = fixed_block.shape[1]
        columns = np.arange(column_offset, column_offset + block_rank)
        fixed_coordinates[np.ix_(positions, columns)] = fixed_block
        logical_projection[np.ix_(columns, positions)] = projection_block
        column_offset += block_rank

    selected_tuple = tuple(int(value) for value in selected_owner_indices)
    if selected_tuple != tuple(sorted(selected_tuple)):
        permutation = np.argsort(np.asarray(selected_tuple, dtype=np.int64))
        fixed_coordinates = fixed_coordinates[:, permutation]
        logical_projection = logical_projection[permutation, :]
        selected_tuple = tuple(selected_tuple[int(index)] for index in permutation)

    return LocalFixedCompilation(
        component_index=component.component_index,
        component_owner_indices=component.logical_owner_indices,
        fixed_vocabulary_coordinates=fixed_coordinates,
        logical_projection_coordinates=logical_projection,
        fixed_rank=total_rank,
        selected_global_owner_indices=selected_tuple,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        gray_zone_certificate={
            "threshold": max(thresholds, default=0.0),
            "gray_zone_upper": max(gray_uppers, default=0.0),
            "gray_zone_factor": gray_zone_factor,
            "block_count": len(blocks),
        },
        generator_residuals=generator_residuals,
        hermitian_residual=hermitian_residual,
        propagated_error_bounds=propagated_error_bounds,
        certification_metadata={
            "algorithm": LOCAL_REYNOLDS_ALGORITHM_V1,
            "block_backend": LOCAL_REYNOLDS_BLOCK_BACKEND_V1,
            "component_policy": LOCAL_FIXED_COMPONENT_POLICY_V1,
            "owner_policy": LOCAL_FIXED_OWNER_POLICY_V1,
            "raw_dimension": raw_dimension,
            "independent_dimension": raw_dimension,
            "fixed_rank": total_rank,
            "group_order": len(words),
            "action_block_count": len(blocks),
            "maximum_action_block_dimension": max(
                (len(block) for block in blocks),
                default=0,
            ),
            "action_blocks": block_metadata,
            "generator_action_nnz": dict(component.source_generator_action_nnz),
            "vocabulary_rank_required": False,
        },
    )


def solve_local_fixed_component_blocked(
    component: LocalVocabularyComponent,
    *,
    absolute_tolerance: float = LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1,
    relative_tolerance: float = LOCAL_FIXED_RELATIVE_TOLERANCE_V1,
    gray_zone_factor: float = LOCAL_FIXED_GRAY_ZONE_FACTOR_V1,
    dense_component_cutoff: int = LOCAL_FIXED_DENSE_COMPONENT_CUTOFF_V1,
) -> LocalFixedCompilation:
    """Solve an exact component through certified generator/H action blocks.

    Ambient-support edges remain part of the exact component identity.  They do
    not, however, create fixed-space constraints.  When an exact component is
    larger than the dense cutoff, the common generator/H support graph provides
    an exact direct-sum decomposition of the constraint equations.  Each block
    is solved by :func:`solve_local_fixed_component`; shared ambient tails are
    deliberately retained and are removed later by the global degree-filtered
    physical-space certificate.
    """

    raw_dimension = len(component.logical_owner_indices)
    cutoff = _nonnegative_integer(
        dense_component_cutoff,
        name="dense_component_cutoff",
    )
    blocks = _certified_action_graph_blocks(component)
    maximum_block_dimension = max((len(block) for block in blocks), default=0)
    if len(blocks) <= 1:
        if raw_dimension <= cutoff:
            return solve_local_fixed_component(
                component,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                gray_zone_factor=gray_zone_factor,
                dense_component_cutoff=cutoff,
            )
        raise LocalGeneratorNullspaceUnavailable(
            "oversize_irreducible_action_block",
            certificate={
                "component_index": component.component_index,
                "raw_dimension": raw_dimension,
                "action_block_count": len(blocks),
                "maximum_action_block_dimension": maximum_block_dimension,
                "dense_component_cutoff": cutoff,
            },
        )

    global_gram = _dense_gram(component.ambient_columns)
    global_diagonal = np.maximum(0.0, np.diag(global_gram))
    global_norms = np.sqrt(global_diagonal)
    global_scale = max(
        1.0,
        float(np.max(global_norms)) if global_norms.size else 0.0,
    )
    global_column_thresholds = (
        np.asarray(component.column_absolute_error_bounds, dtype=np.float64)
        + absolute_tolerance
        + relative_tolerance * np.maximum(1.0, global_norms)
        + _gamma_n(raw_dimension) * np.maximum(1.0, global_norms)
    )
    if np.any(global_norms <= global_column_thresholds):
        if raw_dimension <= cutoff:
            return solve_local_fixed_component(
                component,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                gray_zone_factor=gray_zone_factor,
                dense_component_cutoff=cutoff,
            )
        raise LocalGeneratorNullspaceUnavailable(
            "oversize_vocabulary_rank_deficient",
            certificate={
                "component_index": component.component_index,
                "raw_dimension": raw_dimension,
                "minimum_column_norm": float(np.min(global_norms)),
                "maximum_column_threshold": float(
                    np.max(global_column_thresholds)
                ),
            },
        )
    normalized_global_gram = (
        global_gram
        / global_norms[:, None]
        / global_norms[None, :]
    )
    normalized_global_gram = 0.5 * (
        normalized_global_gram + normalized_global_gram.T
    )
    global_gram_scale = max(
        1.0,
        float(np.linalg.norm(normalized_global_gram, ord=2)),
    )
    global_rank_threshold = float(
        np.linalg.norm(
            np.asarray(component.column_absolute_error_bounds, dtype=np.float64)
            / np.maximum(global_norms, np.finfo(np.float64).tiny)
        )
        + absolute_tolerance / global_scale
        + relative_tolerance * global_gram_scale
        + _gamma_n(raw_dimension) * global_gram_scale
    )
    global_rank_threshold = max(
        global_rank_threshold,
        np.finfo(np.float64).eps,
    )
    minimum_global_eigenvalue = float(
        scipy.linalg.eigvalsh(
            normalized_global_gram,
            subset_by_index=(0, 0),
            check_finite=False,
            driver="evr",
        )[0]
    )
    global_eigenvalue_backward_error = float(
        64.0 * _gamma_n(raw_dimension) * global_gram_scale
    )
    certified_minimum_global_eigenvalue = float(
        minimum_global_eigenvalue - global_eigenvalue_backward_error
    )
    if certified_minimum_global_eigenvalue <= global_rank_threshold**2:
        if raw_dimension <= cutoff:
            return solve_local_fixed_component(
                component,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                gray_zone_factor=gray_zone_factor,
                dense_component_cutoff=cutoff,
            )
        raise LocalGeneratorNullspaceUnavailable(
            "oversize_vocabulary_rank_deficient",
            certificate={
                "component_index": component.component_index,
                "raw_dimension": raw_dimension,
                "minimum_normalized_gram_eigenvalue": (
                    minimum_global_eigenvalue
                ),
                "eigenvalue_backward_error": (
                    global_eigenvalue_backward_error
                ),
                "rank_threshold": global_rank_threshold,
            },
        )
    minimum_global_singular_value = float(
        np.sqrt(certified_minimum_global_eigenvalue)
    )
    if minimum_global_singular_value < gray_zone_factor * global_rank_threshold:
        if raw_dimension <= cutoff:
            return solve_local_fixed_component(
                component,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                gray_zone_factor=gray_zone_factor,
                dense_component_cutoff=cutoff,
            )
        raise LocalGeneratorNullspaceUnavailable(
            "oversize_vocabulary_rank_gray_zone",
            certificate={
                "component_index": component.component_index,
                "raw_dimension": raw_dimension,
                "minimum_normalized_singular_value": (
                    minimum_global_singular_value
                ),
                "rank_threshold": global_rank_threshold,
                "gray_zone_upper": (
                    gray_zone_factor * global_rank_threshold
                ),
            },
        )

    if maximum_block_dimension > cutoff:
        raise LocalGeneratorNullspaceUnavailable(
            "oversize_irreducible_action_block",
            certificate={
                "component_index": component.component_index,
                "raw_dimension": raw_dimension,
                "action_block_count": len(blocks),
                "maximum_action_block_dimension": maximum_block_dimension,
                "dense_component_cutoff": cutoff,
            },
        )

    fixed_coordinate_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    logical_projection_blocks: list[tuple[np.ndarray, np.ndarray]] = []
    selected_owner_indices: list[int] = []
    singular_values: list[float] = []
    gray_certificates: list[Mapping[str, Any]] = []
    generator_residuals = {
        name: 0.0 for name in component.generator_actions
    }
    propagated_error_bounds = {
        name: 0.0 for name in component.generator_actions
    }
    hermitian_residual = 0.0
    block_metadata: list[dict[str, Any]] = []

    for block_index, block_positions in enumerate(blocks):
        positions = np.asarray(block_positions, dtype=np.int64)
        sub_ambient = sparse.csc_matrix(
            component.ambient_columns[:, positions],
            copy=True,
        )
        sub_support_rows = tuple(
            int(value) for value in np.unique(sub_ambient.indices)
        )
        sub_actions = {
            name: np.asarray(action[np.ix_(positions, positions)], dtype=np.float64)
            for name, action in component.generator_actions.items()
        }
        sub_hermitian = np.asarray(
            component.hermitian_action[np.ix_(positions, positions)],
            dtype=np.float64,
        )
        sub_owners = tuple(
            component.logical_owner_indices[int(position)]
            for position in positions
        )
        sub_component = LocalVocabularyComponent(
            component_index=block_index,
            nominal_degrees=tuple(
                component.nominal_degrees[int(position)]
                for position in positions
            ),
            logical_owner_indices=sub_owners,
            provenance_by_owner={
                owner: component.provenance_by_owner[owner]
                for owner in sub_owners
            },
            ambient_columns=sub_ambient,
            generator_actions=sub_actions,
            generator_error_bounds=dict(component.generator_error_bounds),
            antiunitary_parities=dict(component.antiunitary_parities),
            hermitian_action=sub_hermitian,
            exact_support_rows=sub_support_rows,
            column_absolute_error_bounds=np.asarray(
                component.column_absolute_error_bounds[positions],
                dtype=np.float64,
            ),
            source_generator_action_nnz={
                name: int(np.count_nonzero(sub_actions[name]))
                for name in sub_actions
            },
        )
        result = solve_local_fixed_component(
            sub_component,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            gray_zone_factor=gray_zone_factor,
            dense_component_cutoff=cutoff,
        )
        if result.fixed_rank:
            fixed_coordinate_blocks.append(
                (
                    positions,
                    np.asarray(result.fixed_vocabulary_coordinates, dtype=np.float64),
                )
            )
            logical_projection_blocks.append(
                (
                    positions,
                    np.asarray(result.logical_projection_coordinates, dtype=np.float64),
                )
            )
            selected_owner_indices.extend(result.selected_global_owner_indices)
        singular_values.extend(float(value) for value in result.singular_values)
        gray_certificates.append(result.gray_zone_certificate)
        for name, value in result.generator_residuals.items():
            generator_residuals[name] = max(
                generator_residuals[name], float(value)
            )
        for name, value in result.propagated_error_bounds.items():
            propagated_error_bounds[name] = max(
                propagated_error_bounds[name], float(value)
            )
        hermitian_residual = max(hermitian_residual, float(result.hermitian_residual))
        block_metadata.append(
            {
                "block_index": block_index,
                "raw_dimension": len(block_positions),
                "fixed_rank": int(result.fixed_rank),
                "owner_min": min(sub_owners),
                "owner_max": max(sub_owners),
                "independent_dimension": int(
                    result.certification_metadata["independent_dimension"]
                ),
                "fixed_subspace": dict(result.certification_metadata),
            }
        )

    total_rank = sum(block.shape[1] for _positions, block in fixed_coordinate_blocks)
    fixed_coordinates = np.zeros(
        (raw_dimension, total_rank),
        dtype=np.float64,
    )
    logical_projection = np.zeros(
        (total_rank, raw_dimension),
        dtype=np.float64,
    )
    column_offset = 0
    for (positions, fixed_block), (_logical_positions, projection_block) in zip(
        fixed_coordinate_blocks,
        logical_projection_blocks,
    ):
        block_rank = fixed_block.shape[1]
        fixed_coordinates[np.ix_(positions, np.arange(column_offset, column_offset + block_rank))] = (
            fixed_block
        )
        logical_projection[
            np.ix_(
                np.arange(column_offset, column_offset + block_rank),
                positions,
            )
        ] = projection_block
        column_offset += block_rank

    selected_tuple = tuple(int(value) for value in selected_owner_indices)
    if selected_tuple != tuple(sorted(selected_tuple)):
        permutation = np.argsort(np.asarray(selected_tuple, dtype=np.int64))
        fixed_coordinates = fixed_coordinates[:, permutation]
        logical_projection = logical_projection[permutation, :]
        selected_tuple = tuple(selected_tuple[int(index)] for index in permutation)

    thresholds = [
        float(record.get("threshold", 0.0))
        for record in gray_certificates
    ]
    gray_upper = [
        float(record.get("gray_zone_upper", 0.0))
        for record in gray_certificates
    ]
    return LocalFixedCompilation(
        component_index=component.component_index,
        component_owner_indices=component.logical_owner_indices,
        fixed_vocabulary_coordinates=fixed_coordinates,
        logical_projection_coordinates=logical_projection,
        fixed_rank=total_rank,
        selected_global_owner_indices=selected_tuple,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        gray_zone_certificate={
            "threshold": max(thresholds, default=0.0),
            "gray_zone_upper": max(gray_upper, default=0.0),
            "gray_zone_factor": gray_zone_factor,
            "block_count": len(blocks),
        },
        generator_residuals=generator_residuals,
        hermitian_residual=hermitian_residual,
        propagated_error_bounds=propagated_error_bounds,
        certification_metadata={
            "algorithm": LOCAL_FIXED_ALGORITHM_V1,
            "block_backend": LOCAL_FIXED_BLOCK_BACKEND_V2,
            "component_policy": LOCAL_FIXED_COMPONENT_POLICY_V1,
            "column_scaling_policy": LOCAL_FIXED_COLUMN_SCALING_V1,
            "owner_policy": LOCAL_FIXED_OWNER_POLICY_V1,
            "raw_dimension": raw_dimension,
            "global_vocabulary_full_rank_certified": True,
            "minimum_global_normalized_singular_value": (
                minimum_global_singular_value
            ),
            "minimum_global_normalized_gram_eigenvalue": (
                minimum_global_eigenvalue
            ),
            "global_gram_eigenvalue_backward_error": (
                global_eigenvalue_backward_error
            ),
            "global_vocabulary_rank_threshold": global_rank_threshold,
            "independent_dimension": sum(
                int(record["independent_dimension"])
                for record in block_metadata
            ),
            "fixed_rank": total_rank,
            "action_block_count": len(blocks),
            "maximum_action_block_dimension": maximum_block_dimension,
            "action_blocks": block_metadata,
            "generator_action_nnz": dict(component.source_generator_action_nnz),
            "dense_component_cutoff": cutoff,
        },
    )


__all__ = [
    "LOCAL_FIXED_ABSOLUTE_TOLERANCE_V1",
    "LOCAL_FIXED_ALGORITHM_V1",
    "LOCAL_FIXED_BLOCK_BACKEND_V2",
    "LOCAL_FIXED_COLUMN_SCALING_V1",
    "LOCAL_FIXED_COMPONENT_POLICY_V1",
    "LOCAL_FIXED_DENSE_COMPONENT_CUTOFF_V1",
    "LOCAL_FIXED_GRAY_ZONE_FACTOR_V1",
    "LOCAL_FIXED_MATERIALIZATION_POLICY_V1",
    "LOCAL_FIXED_OWNER_POLICY_V1",
    "LOCAL_FIXED_PHYSICAL_SEED_DIGEST_V2",
    "LOCAL_FIXED_REDUCE_POLICY_V1",
    "LOCAL_FIXED_STRUCTURAL_PLAN_POLICY_V1",
    "LOCAL_FIXED_RELATIVE_TOLERANCE_V1",
    "LOCAL_REYNOLDS_ALGORITHM_V1",
    "LOCAL_REYNOLDS_BLOCK_BACKEND_V1",
    "LocalFixedCompilation",
    "LocalGeneratorNullspaceUnavailable",
    "LocalVocabularyComponent",
    "build_exact_joint_components",
    "solve_local_fixed_component",
    "solve_local_fixed_component_blocked",
    "solve_local_reynolds_component_blocked",
]
