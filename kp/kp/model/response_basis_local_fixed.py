"""Data contracts for component-local fixed-response compilation.

This module intentionally contains no component construction, nullspace solver,
or production routing.  It only defines the immutable values exchanged by those
later stages and the typed failure used when a local nullspace cannot be
certified.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import sparse


def _deep_freeze(value: Any) -> Any:
    """Defensively copy nested diagnostic data into immutable containers."""

    if sparse.issparse(value):
        matrix = sparse.csc_matrix(value, copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        matrix.data.setflags(write=False)
        matrix.indices.setflags(write=False)
        matrix.indptr.setflags(write=False)
        return matrix
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        return array
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
    matrix = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    matrix.setflags(write=False)
    return matrix


def _real_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    vector = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be non-negative")
    vector.setflags(write=False)
    return vector


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
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} contains non-finite values")
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
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


__all__ = [
    "LocalFixedCompilation",
    "LocalGeneratorNullspaceUnavailable",
    "LocalVocabularyComponent",
]
