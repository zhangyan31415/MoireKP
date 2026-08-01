from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.linalg
from scipy import sparse

from ..identity import hash_array
from ..symmetry.joint_exactification import materialize_block_route_action


COMPLETE_LINEAR_V2 = "complete_linear_v2"
LEGACY_FROZEN_V1 = "legacy_frozen_v1"
COORDINATE_CONVENTION_V1 = "right_handed_model_cartesian_reciprocal_v1"
RESPONSE_NORMALIZATION_V1 = "reynolds_mean__hermitian_half_sum__dimensionless_k_v1"
COEFFICIENT_NORMALIZATION_V1 = "response_amplitude_abs_c_times_s_v1"
REGULARIZATION_NORMALIZATION_V1 = "coefficient_response_norm__energy_1eV_v1"
FIT_CHANNEL_POLICY_V2 = "confirmed_nonzero_only_v2"
FITTABLE_CHANNEL_CLASSIFICATIONS = frozenset({"confirmed_nonzero"})
FIT_SOLVER_POLICY_V2 = "real_svd_global_monomial_propagated_backward_error_v2"
FIT_SOLVER_POLICY_RIDGE_V3 = "real_ridge_all_confirmed__diagnostic_propagated_rank_v3"
FIT_SOLVER_POLICY_LEGACY = "legacy_real_qr_machine_tolerance_v0"
COMPILER_VERSION = "complete-response-basis-v2-symbolic-finite-p-v47"
TARGET_SPECTRAL_WEIGHTING_V1 = "target_spectral_linear_v1"
TARGET_SPECTRAL_TRACE_NORMALIZATION_V1 = "global_mean_trace_per_dimension_v1"
NORMALIZED_LOW_ENERGY_LINEAR_V1 = "normalized-low-energy-linear-v1"
EQUAL_MATRIX_FIT_OBJECTIVE_V1 = "equal_matrix_v1"


def _canonical_csr(value: sparse.spmatrix | np.ndarray, *, shape: tuple[int, int] | None = None) -> sparse.csr_matrix:
    # Always own the sparse buffers.  Compiled response channels expose their
    # coefficient matrices as read-only objects, so canonicalization must not
    # borrow and then mutate an input matrix's buffers.
    matrix = sparse.csr_matrix(value, dtype=np.complex128, copy=True)
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"coefficient matrix has shape {matrix.shape}, expected {shape}")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _readonly_csr(
    value: sparse.spmatrix | np.ndarray,
    *,
    shape: tuple[int, int] | None = None,
) -> sparse.csr_matrix:
    matrix = _canonical_csr(value, shape=shape)
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def _monomial_similarity_action(
    value: sparse.spmatrix | np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return source-index routes/phases when ``value`` is monomial."""

    matrix = _canonical_csr(value)
    if matrix.shape[0] != matrix.shape[1] or matrix.nnz != matrix.shape[0]:
        return None
    column = matrix.tocsc()
    if not np.all(np.diff(column.indptr) == 1):
        return None
    target_rows = np.asarray(column.indices, dtype=np.int64).copy()
    if np.unique(target_rows).size != matrix.shape[0]:
        return None
    phases = np.asarray(column.data, dtype=np.complex128).copy()
    if np.any(phases == 0.0j):
        return None
    target_rows.setflags(write=False)
    phases.setflags(write=False)
    return target_rows, phases


def _apply_monomial_similarity(
    matrix: sparse.csr_matrix,
    action: tuple[np.ndarray, np.ndarray],
) -> sparse.csr_matrix:
    """Evaluate ``U matrix U^dagger`` without sparse matrix products."""

    target_rows, phases = action
    source = matrix.tocoo(copy=False)
    dim = int(target_rows.size)
    if source.nnz == 0:
        return sparse.csr_matrix((dim, dim), dtype=np.complex128)
    rows = target_rows[np.asarray(source.row, dtype=np.int64)]
    cols = target_rows[np.asarray(source.col, dtype=np.int64)]
    data = (
        phases[np.asarray(source.row, dtype=np.int64)]
        * np.asarray(source.data, dtype=np.complex128)
        * np.conjugate(phases[np.asarray(source.col, dtype=np.int64)])
    )
    order = np.lexsort((cols, rows))
    sorted_rows = np.asarray(rows[order], dtype=np.int64)
    sorted_cols = np.asarray(cols[order], dtype=np.int64)
    sorted_data = np.asarray(data[order], dtype=np.complex128)
    indptr = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(np.bincount(sorted_rows, minlength=dim), dtype=np.int64),
        )
    )
    return sparse.csr_matrix(
        (sorted_data, sorted_cols, indptr),
        shape=(dim, dim),
        dtype=np.complex128,
        copy=False,
    )


def _deep_immutable_copy(value: Any) -> Any:
    """Defensively copy nested basis state into read-only containers."""

    if sparse.issparse(value):
        return _readonly_csr(value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _deep_immutable_copy(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_immutable_copy(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_immutable_copy(item) for item in value)
    return value


def _json_record(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind in {"b", "i", "u", "f", "c", "m", "M"}:
            little_dtype = array.dtype.newbyteorder("<")
            array = array.astype(little_dtype, copy=False)
        array = np.ascontiguousarray(array)
        return {
            "__array__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if sparse.issparse(value):
        matrix = _canonical_csr(value)
        return {
            "__csr__": True,
            "shape": list(matrix.shape),
            "data": _json_record(matrix.data),
            "indices": _json_record(matrix.indices),
            "indptr": _json_record(matrix.indptr),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_record(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_json_record(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, complex):
        return {"__complex__": [float(value.real), float(value.imag)]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_record(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _equal_matrix_fit_objective() -> dict[str, Any]:
    return {
        "mode": "equal_matrix",
        "version": EQUAL_MATRIX_FIT_OBJECTIVE_V1,
        "normalization": "unit_matrix_v1",
    }


@dataclass(frozen=True)
class TargetSpectralWeightingSpec:
    """Target-only spectral weighting for a linear complete-response fit."""

    mode: str = "target_spectral_linear"
    band_edge: str = "top"
    window_mode: str = "fixed_count_degeneracy_safe"
    n_bands: int | None = None
    min_bands: int | None = None
    max_bands: int | None = None
    floor: float = 0.05
    alpha: float = 1.0
    degeneracy_atol_ev: float = 1.0e-10
    degeneracy_rtol: float = 1.0e-10
    normalization: str = TARGET_SPECTRAL_TRACE_NORMALIZATION_V1
    two_sided_projector_weight: float = 0.0

    def __post_init__(self) -> None:
        if str(self.mode) != "target_spectral_linear":
            raise ValueError("target spectral weighting mode must be 'target_spectral_linear'")
        if str(self.band_edge) not in {"top", "bottom"}:
            raise ValueError("target spectral band_edge must be 'top' or 'bottom'")
        if str(self.window_mode) not in {"fixed_count_degeneracy_safe", "auto_gap"}:
            raise ValueError(
                "target spectral window_mode must be 'fixed_count_degeneracy_safe' or 'auto_gap'"
            )
        if self.window_mode == "fixed_count_degeneracy_safe":
            if self.n_bands is None or int(self.n_bands) <= 0:
                raise ValueError("fixed target spectral weighting requires positive n_bands")
        else:
            if self.min_bands is None or self.max_bands is None:
                raise ValueError("auto_gap target spectral weighting requires min_bands and max_bands")
            if int(self.min_bands) <= 0 or int(self.max_bands) < int(self.min_bands):
                raise ValueError("auto_gap target spectral band limits are invalid")
        floor = float(self.floor)
        alpha = float(self.alpha)
        if not np.isfinite(floor) or floor <= 0.0 or floor > 1.0:
            raise ValueError("target spectral floor must be finite in (0, 1]")
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("target spectral alpha must be finite and non-negative")
        if (
            not np.isfinite(float(self.degeneracy_atol_ev))
            or float(self.degeneracy_atol_ev) < 0.0
            or not np.isfinite(float(self.degeneracy_rtol))
            or float(self.degeneracy_rtol) < 0.0
        ):
            raise ValueError("target spectral degeneracy tolerances must be finite and non-negative")
        if str(self.normalization) != TARGET_SPECTRAL_TRACE_NORMALIZATION_V1:
            raise ValueError("unsupported target spectral trace normalization")
        projector_weight = float(self.two_sided_projector_weight)
        if not np.isfinite(projector_weight) or projector_weight < 0.0:
            raise ValueError(
                "target spectral two_sided_projector_weight must be finite and non-negative"
            )

    @classmethod
    def from_value(
        cls,
        value: "TargetSpectralWeightingSpec | Mapping[str, Any]",
    ) -> "TargetSpectralWeightingSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("spectral_weighting must be a TargetSpectralWeightingSpec or mapping")
        return cls(**dict(value))

    def artifact(self) -> dict[str, Any]:
        artifact = {
            "mode": str(self.mode),
            "version": TARGET_SPECTRAL_WEIGHTING_V1,
            "band_edge": str(self.band_edge),
            "window_mode": str(self.window_mode),
            "n_bands": None if self.n_bands is None else int(self.n_bands),
            "min_bands": None if self.min_bands is None else int(self.min_bands),
            "max_bands": None if self.max_bands is None else int(self.max_bands),
            "floor": float(self.floor),
            "alpha": float(self.alpha),
            "degeneracy_atol_ev": float(self.degeneracy_atol_ev),
            "degeneracy_rtol": float(self.degeneracy_rtol),
            "normalization": str(self.normalization),
        }
        if float(self.two_sided_projector_weight) > 0.0:
            artifact["two_sided_projector"] = {
                "enabled": True,
                "weight": float(self.two_sided_projector_weight),
                "projector_source": "target_spectral_window_v1",
            }
        return artifact


@dataclass(frozen=True)
class NormalizedLowEnergyWeightingSpec:
    mode: str = "normalized_low_energy_linear_v1"
    band_edge: str = "top"
    window_mode: str = "fixed_count_degeneracy_safe"
    n_bands: int = 1
    one_sided_weight: float = 1.0
    two_sided_weight: float = 1.0
    degeneracy_atol_ev: float = 1.0e-4
    degeneracy_rtol: float = 0.0

    def __post_init__(self) -> None:
        if self.mode != "normalized_low_energy_linear_v1":
            raise ValueError("normalized low-energy weighting has an invalid mode")
        if self.band_edge not in {"top", "bottom"}:
            raise ValueError("normalized low-energy band_edge must be 'top' or 'bottom'")
        if self.window_mode != "fixed_count_degeneracy_safe":
            raise ValueError("normalized low-energy weighting requires a fixed degeneracy-safe window")
        if int(self.n_bands) <= 0:
            raise ValueError("normalized low-energy weighting requires positive n_bands")
        for name in ("one_sided_weight", "two_sided_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"normalized low-energy {name} must be finite and non-negative")

    @classmethod
    def from_value(
        cls,
        value: "NormalizedLowEnergyWeightingSpec | Mapping[str, Any]",
    ) -> "NormalizedLowEnergyWeightingSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("normalized low-energy weighting must be a mapping")
        return cls(**dict(value))

    def artifact(self, *, resolved_band_counts: Sequence[int]) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "version": NORMALIZED_LOW_ENERGY_LINEAR_V1,
            "band_edge": self.band_edge,
            "window_mode": self.window_mode,
            "requested_bands": int(self.n_bands),
            "resolved_band_counts": [int(value) for value in resolved_band_counts],
            "one_sided_weight": float(self.one_sided_weight),
            "two_sided_weight": float(self.two_sided_weight),
            "normalization": "dimension_mean_square_v1",
        }


@dataclass(frozen=True)
class NormalizedLowEnergyWeighting:
    spec: NormalizedLowEnergyWeightingSpec
    eigenvectors: np.ndarray = field(repr=False, compare=False)
    selected_mask: np.ndarray = field(repr=False, compare=False)
    selected_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        vectors = np.array(self.eigenvectors, dtype=np.complex128, copy=True, order="C")
        mask = np.array(self.selected_mask, dtype=np.bool_, copy=True, order="C")
        vectors.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "selected_mask", mask)

    @property
    def maximum_weight_by_kpoint(self) -> np.ndarray:
        total = 1.0 + float(self.spec.one_sided_weight) + float(self.spec.two_sided_weight)
        return np.full(len(self.selected_counts), total, dtype=np.float64)

    def artifact(self) -> dict[str, Any]:
        return self.spec.artifact(resolved_band_counts=self.selected_counts)


@dataclass(frozen=True)
class TargetSpectralWeighting:
    """Prepared target spectral factors; eigenvectors remain fit-local state."""

    spec: TargetSpectralWeightingSpec
    eigenvalues: np.ndarray = field(repr=False, compare=False)
    eigenvectors: np.ndarray = field(repr=False, compare=False)
    spectral_weights: np.ndarray = field(repr=False, compare=False)
    selected_mask: np.ndarray = field(repr=False, compare=False)
    selected_counts: tuple[int, ...]
    normalization_factor: float
    effective_floor: float
    target_spectral_hash: str
    spectral_weight_hash: str

    def __post_init__(self) -> None:
        arrays = {
            "eigenvalues": (self.eigenvalues, np.float64),
            "eigenvectors": (self.eigenvectors, np.complex128),
            "spectral_weights": (self.spectral_weights, np.float64),
            "selected_mask": (self.selected_mask, np.bool_),
        }
        for name, (raw, dtype) in arrays.items():
            value = np.array(raw, dtype=dtype, copy=True, order="C")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def maximum_weight_by_kpoint(self) -> np.ndarray:
        return np.max(self.spectral_weights, axis=1)

    def artifact(self) -> dict[str, Any]:
        weights = np.asarray(self.spectral_weights, dtype=np.float64)
        return {
            **self.spec.artifact(),
            "selected_counts": [int(value) for value in self.selected_counts],
            "normalization_factor": float(self.normalization_factor),
            "effective_floor": float(self.effective_floor),
            "spectral_weight_min": float(np.min(weights)),
            "spectral_weight_max": float(np.max(weights)),
            "spectral_weight_mean": float(np.mean(weights)),
            "target_spectral_hash": self.target_spectral_hash,
            "spectral_weight_hash": self.spectral_weight_hash,
        }


def _spectral_degeneracy_tolerance(
    left: float,
    right: float,
    spec: TargetSpectralWeightingSpec,
) -> float:
    return float(spec.degeneracy_atol_ev) + float(spec.degeneracy_rtol) * max(
        abs(float(left)), abs(float(right)), 1.0
    )


def _complete_spectral_boundary(
    eigenvalues: np.ndarray,
    count: int,
    spec: TargetSpectralWeightingSpec,
) -> int:
    values = np.asarray(eigenvalues, dtype=np.float64)
    dim = int(values.size)
    selected = int(count)
    if spec.band_edge == "top":
        while selected < dim:
            boundary = dim - selected
            left = float(values[boundary - 1])
            right = float(values[boundary])
            if abs(right - left) > _spectral_degeneracy_tolerance(left, right, spec):
                break
            selected += 1
    else:
        while selected < dim:
            boundary = selected
            left = float(values[boundary - 1])
            right = float(values[boundary])
            if abs(right - left) > _spectral_degeneracy_tolerance(left, right, spec):
                break
            selected += 1
    return selected


def _target_spectral_window_count(
    eigenvalues: np.ndarray,
    spec: TargetSpectralWeightingSpec,
) -> int:
    values = np.asarray(eigenvalues, dtype=np.float64)
    dim = int(values.size)
    if spec.window_mode == "fixed_count_degeneracy_safe":
        count = int(spec.n_bands or 0)
        if count > dim:
            raise ValueError(f"target spectral n_bands={count} exceeds Hamiltonian dimension {dim}")
        return _complete_spectral_boundary(values, count, spec)
    minimum = int(spec.min_bands or 0)
    maximum = int(spec.max_bands or 0)
    if maximum >= dim:
        raise ValueError("target spectral auto_gap max_bands must be smaller than the Hamiltonian dimension")
    gaps: list[float] = []
    counts = list(range(minimum, maximum + 1))
    for count in counts:
        if spec.band_edge == "top":
            boundary = dim - count
            gap = float(values[boundary] - values[boundary - 1])
        else:
            boundary = count
            gap = float(values[boundary] - values[boundary - 1])
        gaps.append(gap)
    count = counts[int(np.argmax(np.asarray(gaps, dtype=np.float64)))]
    return _complete_spectral_boundary(values, count, spec)


def build_target_spectral_weighting(
    target_hamiltonians: np.ndarray,
    spec: TargetSpectralWeightingSpec | Mapping[str, Any],
) -> TargetSpectralWeighting:
    """Build a deterministic target-only hard spectral window and its factors."""

    policy = TargetSpectralWeightingSpec.from_value(spec)
    target = np.asarray(target_hamiltonians, dtype=np.complex128)
    if target.ndim != 3 or target.shape[1] != target.shape[2] or target.shape[0] == 0:
        raise ValueError("target_hamiltonians must have non-empty shape (Nk,dim,dim)")
    hermitian_residual = float(np.max(np.abs(target - target.conj().transpose(0, 2, 1))))
    target_scale = max(float(np.max(np.abs(target))), 1.0)
    if hermitian_residual > 1.0e-11 * target_scale:
        raise ValueError(
            "target spectral weighting requires Hermitian target Hamiltonians; "
            f"residual={hermitian_residual:.3e}"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(target)
    selected_mask = np.zeros_like(eigenvalues, dtype=bool)
    selected_counts: list[int] = []
    dim = int(target.shape[1])
    for k_index, values in enumerate(eigenvalues):
        count = _target_spectral_window_count(values, policy)
        selected_counts.append(count)
        if policy.band_edge == "top":
            selected_mask[k_index, dim - count :] = True
        else:
            selected_mask[k_index, :count] = True
    raw_weights = np.full_like(eigenvalues, float(policy.floor), dtype=np.float64)
    raw_weights += (
        float(policy.alpha)
        * (1.0 - float(policy.floor))
        * selected_mask.astype(np.float64)
    )
    normalization_factor = float(raw_weights.size / np.sum(raw_weights))
    spectral_weights = normalization_factor * raw_weights
    effective_floor = normalization_factor * float(policy.floor)
    target_spectral_hash = hashlib.sha256(
        _canonical_json({"target": target, "eigenvalues": eigenvalues}).encode("utf-8")
    ).hexdigest()
    spectral_weight_hash = hashlib.sha256(
        _canonical_json(spectral_weights).encode("utf-8")
    ).hexdigest()
    return TargetSpectralWeighting(
        spec=policy,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        spectral_weights=spectral_weights,
        selected_mask=selected_mask,
        selected_counts=tuple(selected_counts),
        normalization_factor=normalization_factor,
        effective_floor=effective_floor,
        target_spectral_hash=target_spectral_hash,
        spectral_weight_hash=spectral_weight_hash,
    )


def build_normalized_low_energy_weighting(
    target_hamiltonians: np.ndarray,
    spec: NormalizedLowEnergyWeightingSpec | Mapping[str, Any],
) -> NormalizedLowEnergyWeighting:
    """Resolve the target projector used by the public normalized fit loss."""

    policy = NormalizedLowEnergyWeightingSpec.from_value(spec)
    target = np.asarray(target_hamiltonians, dtype=np.complex128)
    if target.ndim != 3 or target.shape[1] != target.shape[2] or target.shape[0] == 0:
        raise ValueError("target_hamiltonians must have non-empty shape (Nk,dim,dim)")
    hermitian_residual = float(np.max(np.abs(target - target.conj().transpose(0, 2, 1))))
    target_scale = max(float(np.max(np.abs(target))), 1.0)
    if hermitian_residual > 1.0e-11 * target_scale:
        raise ValueError(
            "normalized low-energy weighting requires Hermitian target Hamiltonians; "
            f"residual={hermitian_residual:.3e}"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(target)
    selected_mask = np.zeros_like(eigenvalues, dtype=bool)
    selected_counts: list[int] = []
    dim = int(target.shape[1])
    for k_index, values in enumerate(eigenvalues):
        count = _target_spectral_window_count(values, policy)
        selected_counts.append(count)
        if policy.band_edge == "top":
            selected_mask[k_index, dim - count :] = True
        else:
            selected_mask[k_index, :count] = True
    return NormalizedLowEnergyWeighting(
        spec=policy,
        eigenvectors=eigenvectors,
        selected_mask=selected_mask,
        selected_counts=tuple(selected_counts),
    )


@dataclass(frozen=True)
class PolynomialCoordinateBasis:
    origin: tuple[float, float]
    reciprocal_basis: tuple[tuple[float, float], tuple[float, float]]
    scale: float
    max_degree: int
    monomials: tuple[tuple[int, int], ...]
    coordinate_convention: str = COORDINATE_CONVENTION_V1

    @classmethod
    def from_reciprocal_basis(
        cls,
        *,
        origin: Sequence[float],
        reciprocal_basis: Sequence[Sequence[float]],
        max_degree: int,
    ) -> "PolynomialCoordinateBasis":
        origin_array = np.asarray(origin, dtype=np.float64)
        basis_array = np.asarray(reciprocal_basis, dtype=np.float64)
        if origin_array.shape != (2,):
            raise ValueError(f"polynomial coordinate origin must have shape (2,), got {origin_array.shape}")
        if basis_array.shape != (2, 2):
            raise ValueError(f"reciprocal_basis must have shape (2,2), got {basis_array.shape}")
        if not np.all(np.isfinite(origin_array)) or not np.all(np.isfinite(basis_array)):
            raise ValueError("polynomial coordinate metadata must be finite")
        if abs(float(np.linalg.det(basis_array))) <= np.finfo(float).eps:
            raise ValueError("reciprocal_basis must be nonsingular")
        degree = int(max_degree)
        if degree < 0:
            raise ValueError("max_degree must be non-negative")
        scale = float(np.sqrt(np.mean(np.sum(basis_array**2, axis=1))))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("polynomial coordinate scale must be positive")
        monomials = tuple(
            (r, total - r)
            for total in range(degree + 1)
            for r in range(total + 1)
        )
        return cls(
            origin=(float(origin_array[0]), float(origin_array[1])),
            reciprocal_basis=(
                (float(basis_array[0, 0]), float(basis_array[0, 1])),
                (float(basis_array[1, 0]), float(basis_array[1, 1])),
            ),
            scale=scale,
            max_degree=degree,
            monomials=monomials,
        )

    def to_dimensionless(self, kpoints: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
        points = np.asarray(kpoints, dtype=np.float64)
        if points.shape[-1:] != (2,):
            raise ValueError(f"kpoints must end in a Cartesian dimension of 2, got {points.shape}")
        return (points - np.asarray(self.origin, dtype=np.float64)) / float(self.scale)

    def metadata(self) -> dict[str, Any]:
        return {
            "coordinate_convention": self.coordinate_convention,
            "origin": [float(value) for value in self.origin],
            "scale": float(self.scale),
            "reciprocal_basis": [list(row) for row in self.reciprocal_basis],
            "max_degree": int(self.max_degree),
            "monomials": [list(item) for item in self.monomials],
            "monomial_ordering": "total_degree_then_r_then_s",
            "complex_axis_orientation": "w=u_x+i*u_y",
            "cartesian_axes": [[1.0, 0.0], [0.0, 1.0]],
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "PolynomialCoordinateBasis":
        basis = cls.from_reciprocal_basis(
            origin=metadata["origin"],
            reciprocal_basis=metadata["reciprocal_basis"],
            max_degree=int(metadata["max_degree"]),
        )
        if str(metadata.get("coordinate_convention", COORDINATE_CONVENTION_V1)) != COORDINATE_CONVENTION_V1:
            raise ValueError("unsupported polynomial coordinate convention")
        if not np.isclose(float(metadata["scale"]), basis.scale, atol=0.0, rtol=1.0e-15):
            raise ValueError("frozen polynomial coordinate scale is inconsistent with reciprocal_basis")
        return basis


@dataclass(frozen=True)
class NullClassificationPolicy:
    numerical_factor: float = 1.0
    confirm_factor: float = 100.0
    policy_version: str = "absolute_propagated_error_gray_zone_v1"

    def __post_init__(self) -> None:
        if self.numerical_factor <= 0.0:
            raise ValueError("numerical_factor must be positive")
        if self.confirm_factor <= self.numerical_factor:
            raise ValueError("confirm_factor must exceed numerical_factor")

    def classify(self, norm: float, error_bound: float, *, structural: bool = False) -> str:
        if structural:
            return "structural_zero"
        magnitude = float(norm)
        error = max(float(error_bound), 0.0)
        if magnitude <= self.numerical_factor * error:
            return "numerical_zero"
        if magnitude < self.confirm_factor * error:
            return "ambiguous"
        return "confirmed_nonzero"

    def artifact(self) -> dict[str, Any]:
        return {
            "version": self.policy_version,
            "numerical_factor": float(self.numerical_factor),
            "confirm_factor": float(self.confirm_factor),
            "engineering_policy": True,
            "mathematical_constant": False,
        }


@dataclass(frozen=True)
class FiniteGroupElement:
    canonical_word: tuple[str, ...]
    antiunitary: bool
    canonical_k_map: tuple[tuple[int, int], tuple[int, int]]
    q_permutation: tuple[int, ...]
    sector_permutation: tuple[int, ...]
    k_pullback: tuple[tuple[float, float], tuple[float, float]]
    internal_u: np.ndarray = field(repr=False, compare=False)
    alternate_words: tuple[tuple[str, ...], ...] = ()
    alternate_word_residual: float = 0.0
    unitarity_residual: float = 0.0

    @property
    def discrete_key(self) -> tuple[Any, ...]:
        return (
            bool(self.antiunitary),
            self.canonical_k_map,
            self.q_permutation,
            self.sector_permutation,
        )


@dataclass(frozen=True)
class FiniteGroup:
    elements: tuple[FiniteGroupElement, ...]
    max_group_size: int = 256
    max_word_length: int = 32
    algebra_residual: float = 0.0

    def artifact(self) -> dict[str, Any]:
        return {
            "max_group_size": int(self.max_group_size),
            "max_word_length": int(self.max_word_length),
            "algebra_residual": float(self.algebra_residual),
            "elements": [
                {
                    "canonical_word": list(element.canonical_word),
                    "alternate_words": [list(word) for word in element.alternate_words],
                    "antiunitary": bool(element.antiunitary),
                    "canonical_k_map": [list(row) for row in element.canonical_k_map],
                    "k_pullback": [list(row) for row in element.k_pullback],
                    "q_permutation": list(element.q_permutation),
                    "sector_permutation": list(element.sector_permutation),
                    "alternate_word_residual": float(element.alternate_word_residual),
                    "unitarity_residual": float(element.unitarity_residual),
                }
                for element in self.elements
            ],
        }


@dataclass(frozen=True)
class FiniteGroupGenerator:
    name: str
    antiunitary: bool
    canonical_k_map: tuple[tuple[int, int], tuple[int, int]]
    q_permutation: tuple[int, ...]
    sector_permutation: tuple[int, ...]
    k_forward: tuple[tuple[float, float], tuple[float, float]]
    internal_u: np.ndarray = field(repr=False, compare=False)


def _validate_permutation(permutation: Sequence[int], *, name: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in permutation)
    if sorted(values) != list(range(len(values))):
        raise ValueError(f"{name} must be an integer permutation, got {values}")
    return values


def _canonicalize_global_phase(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.size == 0:
        return value.copy()
    flat = value.ravel()
    pivot = int(np.argmax(np.abs(flat)))
    if abs(flat[pivot]) == 0.0:
        return value.copy()
    phase = np.exp(-1j * np.angle(flat[pivot]))
    canonical = phase * value
    canonical.real[np.abs(canonical.real) < 4.0 * np.finfo(float).eps] = 0.0
    canonical.imag[np.abs(canonical.imag) < 4.0 * np.finfo(float).eps] = 0.0
    return canonical


def _phase_aligned_residual(reference: np.ndarray, candidate: np.ndarray) -> float:
    overlap = np.vdot(candidate, reference)
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    aligned = phase * candidate
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(reference - aligned) / denominator)


def _is_strict_identity_group_element(
    element: FiniteGroupElement,
    *,
    dim: int,
) -> bool:
    """Return whether an element is the exact identity up to internal phase."""

    expected_dim = int(dim)
    if expected_dim <= 0 or element.antiunitary or tuple(element.canonical_word):
        return False
    if not np.array_equal(
        np.asarray(element.canonical_k_map, dtype=np.int64),
        np.eye(2, dtype=np.int64),
    ):
        return False
    if tuple(int(value) for value in element.q_permutation) != tuple(
        range(len(element.q_permutation))
    ):
        return False
    if tuple(int(value) for value in element.sector_permutation) != tuple(
        range(len(element.sector_permutation))
    ):
        return False
    pullback = np.asarray(element.k_pullback, dtype=np.float64)
    pullback_tolerance = 64.0 * np.finfo(np.float64).eps
    if (
        pullback.shape != (2, 2)
        or not np.all(np.isfinite(pullback))
        or not np.allclose(
            pullback,
            np.eye(2, dtype=np.float64),
            rtol=0.0,
            atol=pullback_tolerance,
        )
    ):
        return False
    internal_u = np.asarray(element.internal_u, dtype=np.complex128)
    if (
        internal_u.shape != (expected_dim, expected_dim)
        or not np.all(np.isfinite(internal_u))
    ):
        return False
    phase_tolerance = float(
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, np.sqrt(float(internal_u.size)))
    )
    return bool(
        _phase_aligned_residual(
            np.eye(expected_dim, dtype=np.complex128),
            internal_u,
        )
        <= phase_tolerance
    )


def _compose_group_elements(
    current: FiniteGroupElement,
    generator: FiniteGroupGenerator,
) -> FiniteGroupElement:
    outer_u = np.asarray(generator.internal_u, dtype=np.complex128)
    inner_u = np.asarray(current.internal_u, dtype=np.complex128)
    internal_u = outer_u @ (np.conjugate(inner_u) if generator.antiunitary else inner_u)
    canonical_k_map = np.asarray(generator.canonical_k_map, dtype=np.int64) @ np.asarray(
        current.canonical_k_map,
        dtype=np.int64,
    )
    q_outer = np.asarray(generator.q_permutation, dtype=np.int64)
    q_inner = np.asarray(current.q_permutation, dtype=np.int64)
    sector_outer = np.asarray(generator.sector_permutation, dtype=np.int64)
    sector_inner = np.asarray(current.sector_permutation, dtype=np.int64)
    k_forward = np.asarray(generator.k_forward, dtype=np.float64) @ np.linalg.inv(
        np.asarray(current.k_pullback, dtype=np.float64)
    )
    return FiniteGroupElement(
        canonical_word=(*current.canonical_word, generator.name),
        antiunitary=bool(generator.antiunitary) ^ bool(current.antiunitary),
        canonical_k_map=tuple(tuple(int(value) for value in row) for row in canonical_k_map),
        q_permutation=tuple(int(value) for value in q_outer[q_inner]),
        sector_permutation=tuple(int(value) for value in sector_outer[sector_inner]),
        k_pullback=tuple(tuple(float(value) for value in row) for row in np.linalg.inv(k_forward)),
        internal_u=_canonicalize_global_phase(internal_u),
        unitarity_residual=float(
            np.linalg.norm(internal_u.conj().T @ internal_u - np.eye(internal_u.shape[0]))
            / max(1.0, np.sqrt(float(internal_u.shape[0])))
        ),
    )


def build_finite_group(
    generators: Sequence[FiniteGroupGenerator],
    *,
    max_group_size: int = 256,
    max_word_length: int = 32,
    closure_tolerance: float = 1.0e-8,
) -> FiniteGroup:
    if not generators:
        raise ValueError("finite-group BFS requires at least one generator")
    first_u = np.asarray(generators[0].internal_u, dtype=np.complex128)
    if first_u.ndim != 2 or first_u.shape[0] != first_u.shape[1]:
        raise ValueError("finite-group internal actions must be square matrices")
    dim = int(first_u.shape[0])
    q_size = len(generators[0].q_permutation)
    sector_size = len(generators[0].sector_permutation)
    normalized_generators: list[FiniteGroupGenerator] = []
    for generator in generators:
        if not generator.name:
            raise ValueError("finite-group generators require non-empty names")
        unitary = np.asarray(generator.internal_u, dtype=np.complex128)
        if unitary.shape != (dim, dim):
            raise ValueError("finite-group generator matrices must have a common dimension")
        unitarity = float(
            np.linalg.norm(unitary.conj().T @ unitary - np.eye(dim))
            / max(1.0, np.sqrt(float(dim)))
        )
        if unitarity > float(closure_tolerance):
            raise ValueError(
                f"finite-group generator {generator.name!r} is not unitary: residual={unitarity:.3e}"
            )
        canonical_map = np.asarray(generator.canonical_k_map, dtype=np.int64)
        forward = np.asarray(generator.k_forward, dtype=np.float64)
        if canonical_map.shape != (2, 2) or forward.shape != (2, 2):
            raise ValueError("finite-group k maps must have shape (2,2)")
        if abs(float(np.linalg.det(canonical_map))) != 1:
            raise ValueError(f"canonical k-map for {generator.name!r} must be unimodular")
        if abs(float(np.linalg.det(forward))) <= np.finfo(float).eps:
            raise ValueError(f"Cartesian k-map for {generator.name!r} must be nonsingular")
        q_permutation = _validate_permutation(generator.q_permutation, name=f"{generator.name}.q_permutation")
        sector_permutation = _validate_permutation(
            generator.sector_permutation,
            name=f"{generator.name}.sector_permutation",
        )
        if len(q_permutation) != q_size or len(sector_permutation) != sector_size:
            raise ValueError("finite-group generators must share Q/sector permutation dimensions")
        normalized_generators.append(
            replace(
                generator,
                q_permutation=q_permutation,
                sector_permutation=sector_permutation,
                internal_u=_canonicalize_global_phase(unitary),
            )
        )
    identity = FiniteGroupElement(
        canonical_word=(),
        antiunitary=False,
        canonical_k_map=((1, 0), (0, 1)),
        q_permutation=tuple(range(q_size)),
        sector_permutation=tuple(range(sector_size)),
        k_pullback=((1.0, 0.0), (0.0, 1.0)),
        internal_u=np.eye(dim, dtype=np.complex128),
    )
    elements: list[FiniteGroupElement] = [identity]
    index_by_key = {identity.discrete_key: 0}
    frontier = [0]
    algebra_residual = 0.0
    while frontier:
        current_index = frontier.pop(0)
        current = elements[current_index]
        for generator in normalized_generators:
            candidate = _compose_group_elements(current, generator)
            existing_index = index_by_key.get(candidate.discrete_key)
            if existing_index is not None:
                residual = _phase_aligned_residual(
                    np.asarray(elements[existing_index].internal_u),
                    np.asarray(candidate.internal_u),
                )
                algebra_residual = max(algebra_residual, residual)
                if residual > float(closure_tolerance):
                    raise ValueError(
                        "finite-group words with the same discrete action key are not numerically close "
                        f"modulo global phase: canonical={elements[existing_index].canonical_word}, "
                        f"alternate={candidate.canonical_word}, residual={residual:.3e}"
                    )
                if residual > elements[existing_index].alternate_word_residual:
                    next_residual = float(residual)
                else:
                    next_residual = float(elements[existing_index].alternate_word_residual)
                alternate_words = elements[existing_index].alternate_words
                if candidate.canonical_word not in alternate_words:
                    alternate_words = (*alternate_words, candidate.canonical_word)
                elements[existing_index] = replace(
                    elements[existing_index],
                    alternate_words=alternate_words,
                    alternate_word_residual=next_residual,
                )
                continue
            if len(candidate.canonical_word) > int(max_word_length):
                raise ValueError(
                    "finite-group BFS exceeded max_word_length before closure; refusing a numerically open group"
                )
            if len(elements) >= int(max_group_size):
                raise ValueError(
                    "finite-group BFS exceeded max_group_size before closure; refusing a numerically open group"
                )
            index_by_key[candidate.discrete_key] = len(elements)
            elements.append(candidate)
            frontier.append(len(elements) - 1)
    return FiniteGroup(
        elements=tuple(elements),
        max_group_size=int(max_group_size),
        max_word_length=int(max_word_length),
        algebra_residual=float(algebra_residual),
    )


def identity_finite_group(dim: int, *, q_size: int = 1, sector_size: int = 2) -> FiniteGroup:
    size = int(dim)
    if size <= 0:
        raise ValueError("identity group dimension must be positive")
    element = FiniteGroupElement(
        canonical_word=(),
        antiunitary=False,
        canonical_k_map=((1, 0), (0, 1)),
        q_permutation=tuple(range(int(q_size))),
        sector_permutation=tuple(range(int(sector_size))),
        k_pullback=((1.0, 0.0), (0.0, 1.0)),
        internal_u=np.eye(size, dtype=np.complex128),
    )
    return FiniteGroup(elements=(element,))


@dataclass(frozen=True)
class RawPolynomialSeed:
    seed_id: str
    coefficients: Mapping[tuple[int, int], sparse.spmatrix | np.ndarray]
    support_component: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("raw response seed requires a non-empty seed_id")
        if not self.coefficients:
            raise ValueError(f"raw response seed {self.seed_id!r} has no polynomial coefficients")
        shapes = {tuple(np.asarray(value.shape, dtype=int)) for value in self.coefficients.values()}
        if len(shapes) != 1:
            raise ValueError(f"raw response seed {self.seed_id!r} coefficient shapes differ: {shapes}")
        shape = next(iter(shapes))
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"raw response seed matrices must be square, got {shape}")

    @property
    def dim(self) -> int:
        return int(next(iter(self.coefficients.values())).shape[0])


@dataclass(frozen=True)
class ResponseChannel:
    channel_id: str
    seed_id: str
    component: str
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix]
    unnormalized_norm: float
    propagated_error_bound: float
    classification: str
    response_scale: float
    support_component: str
    metadata: Mapping[str, Any]
    support_leakage_before_cleanup: float = 0.0
    error_bound_components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coefficients = {
            (int(r), int(s)): _readonly_csr(matrix)
            for (r, s), matrix in self.coefficients.items()
        }
        if not coefficients:
            raise ValueError(f"response channel {self.channel_id!r} has no coefficients")
        object.__setattr__(self, "coefficients", MappingProxyType(coefficients))
        object.__setattr__(self, "metadata", _deep_immutable_copy(self.metadata))
        object.__setattr__(
            self,
            "error_bound_components",
            _deep_immutable_copy(self.error_bound_components),
        )

    def coefficient(self, r: int, s: int) -> sparse.csr_matrix:
        matrix = self.coefficients.get((int(r), int(s)))
        if matrix is not None:
            return matrix
        dim = next(iter(self.coefficients.values())).shape[0]
        return sparse.csr_matrix((dim, dim), dtype=np.complex128)


@dataclass(frozen=True)
class CandidateResponseSet:
    coordinate: PolynomialCoordinateBasis
    channels: tuple[ResponseChannel, ...]
    dim: int
    group_artifact: Mapping[str, Any]
    null_policy: NullClassificationPolicy
    support_artifact: Mapping[str, Any] = field(default_factory=dict)
    adjoint_artifact: Mapping[str, Any] = field(default_factory=dict)
    response_normalization: str = RESPONSE_NORMALIZATION_V1

    def artifact(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for channel in self.channels:
            counts[channel.classification] = counts.get(channel.classification, 0) + 1
        logical_count = int(
            self.adjoint_artifact.get(
                "logical_candidate_channel_count", len(self.channels)
            )
        )
        physical_count = int(
            self.adjoint_artifact.get(
                "physically_compiled_representative_channel_count", logical_count
            )
        )
        adjoint_drop_count = int(
            self.adjoint_artifact.get("adjoint_certified_dropped_channel_count", 0)
        )
        authored_seed_count = int(
            self.adjoint_artifact.get("authored_ordered_seed_count", logical_count // 2)
        )
        orbit_descriptor_count = int(
            self.adjoint_artifact.get(
                "adjoint_orbit_descriptor_count", authored_seed_count
            )
        )
        hermitian_ambient_count = int(
            self.adjoint_artifact.get(
                "hermitian_ambient_channel_count", physical_count
            )
        )
        materialized_count = int(
            self.adjoint_artifact.get(
                "physically_materialized_projected_channel_count", physical_count
            )
        )
        return {
            "response_semantics": COMPLETE_LINEAR_V2,
            "response_normalization": self.response_normalization,
            "coordinate": self.coordinate.metadata(),
            "dim": int(self.dim),
            "candidate_channel_count": logical_count,
            "logical_candidate_channel_count": logical_count,
            "authored_ordered_seed_count": authored_seed_count,
            "adjoint_orbit_descriptor_count": orbit_descriptor_count,
            "hermitian_ambient_channel_count": hermitian_ambient_count,
            "physically_materialized_projected_channel_count": materialized_count,
            "physically_compiled_representative_channel_count": physical_count,
            "adjoint_certified_dropped_channel_count": adjoint_drop_count,
            "classification_counts": counts,
            "null_policy": self.null_policy.artifact(),
            "group": _json_record(self.group_artifact),
            "support": _json_record(self.support_artifact),
            "adjoint": _json_record(self.adjoint_artifact),
            "channels": [
                {
                    "channel_id": channel.channel_id,
                    "seed_id": channel.seed_id,
                    "component": channel.component,
                    "classification": channel.classification,
                    "response_absolute_norm": float(channel.unnormalized_norm),
                    "algebra_error_absolute": float(channel.propagated_error_bound),
                    "norm_to_algebra_error_ratio": float(
                        channel.unnormalized_norm
                        / max(channel.propagated_error_bound, np.finfo(float).tiny)
                    ),
                    "support_component": channel.support_component,
                    "metadata": _json_record(channel.metadata),
                    "error_bound_components": _json_record(channel.error_bound_components),
                }
                for channel in self.channels
            ],
        }


def _poly_add(target: dict[tuple[int, int], complex], source: Mapping[tuple[int, int], complex]) -> None:
    for key, value in source.items():
        value_complex = complex(value)
        if value_complex == 0.0j:
            continue
        updated = target.get(key, 0.0j) + value_complex
        if updated == 0.0j:
            target.pop(key, None)
        else:
            target[key] = updated


def _poly_multiply(
    left: Mapping[tuple[int, int], complex],
    right: Mapping[tuple[int, int], complex],
) -> dict[tuple[int, int], complex]:
    out: dict[tuple[int, int], complex] = {}
    for (lr, ls), left_value in left.items():
        left_complex = complex(left_value)
        if left_complex == 0.0j:
            continue
        for (rr, rs), right_value in right.items():
            right_complex = complex(right_value)
            if right_complex == 0.0j:
                continue
            key = (int(lr + rr), int(ls + rs))
            updated = out.get(key, 0.0j) + left_complex * right_complex
            if updated == 0.0j:
                out.pop(key, None)
            else:
                out[key] = updated
    return out


def _poly_power(base: Mapping[tuple[int, int], complex], exponent: int) -> dict[tuple[int, int], complex]:
    out: dict[tuple[int, int], complex] = {(0, 0): 1.0 + 0.0j}
    for _ in range(int(exponent)):
        out = _poly_multiply(out, base)
    return out


def _complex_pullback_polynomials(
    element: FiniteGroupElement,
    coordinate: PolynomialCoordinateBasis,
) -> tuple[dict[tuple[int, int], complex], dict[tuple[int, int], complex]]:
    transform = np.asarray(element.k_pullback, dtype=np.float64)
    if transform.shape != (2, 2):
        raise ValueError(f"finite-group k_pullback must have shape (2,2), got {transform.shape}")
    origin = np.asarray(coordinate.origin, dtype=np.float64)
    translation = (transform @ origin - origin) / float(coordinate.scale)
    a = 0.5 * (transform[0, 0] + transform[1, 1]) + 0.5j * (transform[1, 0] - transform[0, 1])
    b = 0.5 * (transform[0, 0] - transform[1, 1]) + 0.5j * (transform[1, 0] + transform[0, 1])
    c = complex(translation[0], translation[1])
    # Exactified point-group maps are conformal or anti-conformal in this
    # complex coordinate.  Their forbidden coefficient is nevertheless often
    # represented by a machine-roundoff tail (for example 5e-17 for C3).  If
    # retained, an order-N monomial expands into N+1 formally nonzero branches
    # that are discarded only after thousands of sparse matrix operations.
    # Remove only coefficients below a backward-error bound for constructing
    # the 2x2 pullback and translated origin; resolved shear/mixing is retained.
    cleanup_bound = float(
        128.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            np.linalg.norm(transform, ord=2),
            np.linalg.norm(origin, ord=2) / float(coordinate.scale),
        )
    )

    def cleaned(value: complex) -> complex:
        value_complex = complex(value)
        return 0.0j if abs(value_complex) <= cleanup_bound else value_complex

    a = cleaned(a)
    b = cleaned(b)
    c = cleaned(c)
    w_map: dict[tuple[int, int], complex] = {}
    if a != 0.0j:
        w_map[(1, 0)] = a
    if b != 0.0j:
        w_map[(0, 1)] = b
    if c != 0.0j:
        w_map[(0, 0)] = c
    wbar_map: dict[tuple[int, int], complex] = {}
    if a != 0.0j:
        wbar_map[(0, 1)] = complex(np.conjugate(a))
    if b != 0.0j:
        wbar_map[(1, 0)] = complex(np.conjugate(b))
    if c != 0.0j:
        wbar_map[(0, 0)] = complex(np.conjugate(c))
    if not w_map or not wbar_map:
        raise ValueError("finite-group complex pullback is numerically zero")
    return w_map, wbar_map


def _monomial_pullback_table(
    element: FiniteGroupElement,
    coordinate: PolynomialCoordinateBasis,
) -> dict[tuple[int, int], dict[tuple[int, int], complex]]:
    """Compile the scalar polynomial pullback once for one group element."""

    w_map, wbar_map = _complex_pullback_polynomials(element, coordinate)
    w_powers: list[dict[tuple[int, int], complex]] = [{(0, 0): 1.0 + 0.0j}]
    wbar_powers: list[dict[tuple[int, int], complex]] = [
        {(0, 0): 1.0 + 0.0j}
    ]
    for _ in range(coordinate.max_degree):
        w_powers.append(_poly_multiply(w_powers[-1], w_map))
        wbar_powers.append(_poly_multiply(wbar_powers[-1], wbar_map))
    return {
        monomial: _poly_multiply(w_powers[monomial[0]], wbar_powers[monomial[1]])
        for monomial in coordinate.monomials
    }


def _substitute_polynomial_coefficients(
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
    element: FiniteGroupElement,
    coordinate: PolynomialCoordinateBasis,
    *,
    monomial_pullbacks: Mapping[
        tuple[int, int], Mapping[tuple[int, int], complex]
    ]
    | None = None,
) -> dict[tuple[int, int], sparse.csr_matrix]:
    pullbacks = (
        _monomial_pullback_table(element, coordinate)
        if monomial_pullbacks is None
        else monomial_pullbacks
    )
    out: dict[tuple[int, int], sparse.csr_matrix] = {}
    for (r, s), matrix in coefficients.items():
        try:
            scalar_poly = pullbacks[(int(r), int(s))]
        except KeyError as exc:
            raise ValueError(
                f"group pullback lacks declared monomial {(int(r), int(s))}"
            ) from exc
        for monomial, prefactor in scalar_poly.items():
            if sum(monomial) > coordinate.max_degree:
                raise ValueError("group pullback produced a monomial above the declared maximum degree")
            contribution = matrix * complex(prefactor)
            out[monomial] = contribution if monomial not in out else out[monomial] + contribution
    return {key: _canonical_csr(value) for key, value in out.items()}


def _apply_group_element(
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
    element: FiniteGroupElement,
    coordinate: PolynomialCoordinateBasis,
    *,
    internal_u_override: sparse.spmatrix | np.ndarray | None = None,
    internal_monomial_action: tuple[np.ndarray, np.ndarray] | None = None,
    monomial_pullbacks: Mapping[
        tuple[int, int], Mapping[tuple[int, int], complex]
    ]
    | None = None,
) -> dict[tuple[int, int], sparse.csr_matrix]:
    internal = (
        None
        if internal_monomial_action is not None
        else (
            _canonical_csr(internal_u_override)
            if internal_u_override is not None
            else _canonical_csr(np.asarray(element.internal_u, dtype=np.complex128))
        )
    )
    if (
        internal is not None
        and not element.antiunitary
        and np.array_equal(
            np.asarray(element.k_pullback, dtype=np.float64),
            np.eye(2, dtype=np.float64),
        )
        and (internal != sparse.eye(internal.shape[0], format="csr")).nnz == 0
    ):
        return {
            (int(r), int(s)): _canonical_csr(matrix)
            for (r, s), matrix in coefficients.items()
        }
    pulled = _substitute_polynomial_coefficients(
        coefficients,
        element,
        coordinate,
        monomial_pullbacks=monomial_pullbacks,
    )
    if element.antiunitary:
        conjugated: dict[tuple[int, int], sparse.csr_matrix] = {}
        for (r, s), matrix in pulled.items():
            key = (s, r)
            value = matrix.conjugate()
            conjugated[key] = value if key not in conjugated else conjugated[key] + value
        pulled = conjugated
    transformed = (
        {
            key: _apply_monomial_similarity(matrix, internal_monomial_action)
            for key, matrix in pulled.items()
        }
        if internal_monomial_action is not None
        else {
            key: _canonical_csr(internal @ matrix @ internal.getH())  # type: ignore[union-attr]
            for key, matrix in pulled.items()
        }
    )
    return transformed


def _sum_polynomial_coefficients(
    target: dict[tuple[int, int], sparse.csr_matrix],
    source: Mapping[tuple[int, int], sparse.csr_matrix],
) -> None:
    for key, matrix in source.items():
        target[key] = matrix.copy() if key not in target else target[key] + matrix


def _hermitian_project(
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
    *,
    dim: int,
) -> dict[tuple[int, int], sparse.csr_matrix]:
    keys = set(coefficients)
    keys.update((s, r) for r, s in coefficients)
    zero = sparse.csr_matrix((dim, dim), dtype=np.complex128)
    out: dict[tuple[int, int], sparse.csr_matrix] = {}
    for r, s in sorted(keys, key=lambda item: (sum(item), item[0], item[1])):
        direct = coefficients.get((r, s), zero)
        adjoint_partner = coefficients.get((s, r), zero).getH()
        matrix = _canonical_csr(0.5 * (direct + adjoint_partner))
        if matrix.nnz:
            out[(r, s)] = matrix
    return out


def _coefficient_norm(coefficients: Mapping[tuple[int, int], sparse.csr_matrix]) -> float:
    return float(np.sqrt(sum(float(np.vdot(matrix.data, matrix.data).real) for matrix in coefficients.values())))


def _propagated_error_components(
    raw_norm: float,
    *,
    dim: int,
    degree: int,
    group: FiniteGroup,
) -> dict[str, float]:
    eps = np.finfo(np.float64).eps
    operation_count = max(1, len(group.elements))
    gamma = eps * float(64 + 8 * operation_count + 4 * max(0, degree) + 2 * max(1, dim))
    unitary_residual = max((float(item.unitarity_residual) for item in group.elements), default=0.0)
    alternate_residual = max((float(item.alternate_word_residual) for item in group.elements), default=0.0)
    scale = max(float(raw_norm), np.finfo(float).tiny)
    return {
        "floating_point_accumulation": float(gamma * scale),
        "group_algebra": float(group.algebra_residual) * scale,
        "unitarity": unitary_residual * scale,
        "alternate_word": alternate_residual * scale,
    }


def _propagated_error_bound(
    raw_norm: float,
    *,
    dim: int,
    degree: int,
    group: FiniteGroup,
) -> float:
    return float(
        sum(
            _propagated_error_components(
                raw_norm,
                dim=dim,
                degree=degree,
                group=group,
            ).values()
        )
    )


class SupportClosureError(ValueError):
    """An authored support is not closed under the actual group action or adjoint."""


def _exact_structural_support_is_group_adjoint_closed(
    support_mask: np.ndarray,
    *,
    group: FiniteGroup,
) -> bool:
    """Prove matrix-entry support closure once, independently of seed values.

    For an internal action ``X -> U X U^dagger`` the possible output support is
    ``supp(U) @ supp(X) @ supp(U)^T`` in Boolean arithmetic.  Antiunitary
    conjugation does not change this support.  This exact support proof avoids
    applying every group element to every polynomial seed when an authored block
    is already structurally closed.

    No numerical threshold is used here: a nonzero entry in the actual exactified
    action participates in the proof.  If the exact support is not closed the
    caller must retain the slower per-seed, propagated-error path.
    """

    mask = np.asarray(support_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("support mask must be square")
    if np.any(mask.T & ~mask):
        return False
    mask_integer = sparse.csr_matrix(mask.astype(np.int32, copy=False))
    for element in group.elements:
        internal = np.asarray(element.internal_u, dtype=np.complex128)
        if internal.shape != mask.shape:
            raise ValueError(
                f"finite-group internal action has shape {internal.shape}, expected {mask.shape}"
            )
        pattern = sparse.csr_matrix((internal != 0.0).astype(np.int32, copy=False))
        image_support = pattern @ mask_integer @ pattern.T
        image_support.eliminate_zeros()
        coo = image_support.tocoo()
        if np.any(~mask[coo.row, coo.col]):
            return False
    return True


def _outside_support_norm(
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
    support_mask: np.ndarray,
) -> float:
    total = 0.0
    for matrix in coefficients.values():
        coo = matrix.tocoo()
        outside = ~support_mask[coo.row, coo.col]
        if np.any(outside):
            total += float(np.vdot(coo.data[outside], coo.data[outside]).real)
    return float(np.sqrt(total))


def _outside_support_components(
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
    support_mask: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for monomial, matrix in coefficients.items():
        coo = matrix.tocoo()
        outside = ~support_mask[coo.row, coo.col]
        for row, col, value in zip(coo.row[outside], coo.col[outside], coo.data[outside]):
            records.append(
                {
                    "monomial": [int(monomial[0]), int(monomial[1])],
                    "row": int(row),
                    "col": int(col),
                    "value_real": float(np.real(value)),
                    "value_imag": float(np.imag(value)),
                }
            )
    return records


def _adjoint_polynomial(
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
) -> dict[tuple[int, int], sparse.csr_matrix]:
    out: dict[tuple[int, int], sparse.csr_matrix] = {}
    for (r, s), matrix in coefficients.items():
        key = (s, r)
        value = matrix.getH()
        out[key] = value if key not in out else out[key] + value
    return {key: _canonical_csr(value) for key, value in out.items()}


def _mask_polynomial_support(
    coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
    support_mask: np.ndarray,
) -> dict[tuple[int, int], sparse.csr_matrix]:
    out: dict[tuple[int, int], sparse.csr_matrix] = {}
    for key, matrix in coefficients.items():
        coo = matrix.tocoo()
        keep = support_mask[coo.row, coo.col]
        cleaned = _canonical_csr(
            sparse.coo_matrix(
                (coo.data[keep], (coo.row[keep], coo.col[keep])),
                shape=matrix.shape,
            )
        )
        if cleaned.nnz:
            out[key] = cleaned
    return out


def certify_support_closure(
    seeds: Sequence[RawPolynomialSeed],
    *,
    group: FiniteGroup,
    coordinate: PolynomialCoordinateBasis,
    support_masks: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    by_component: dict[str, list[RawPolynomialSeed]] = {}
    for seed in seeds:
        by_component.setdefault(seed.support_component, []).append(seed)
    certificates: dict[str, dict[str, Any]] = {}
    for component, component_seeds in by_component.items():
        if component not in support_masks:
            raise SupportClosureError(f"support component {component!r} has no authored support mask")
        mask = np.asarray(support_masks[component], dtype=bool)
        dim = int(component_seeds[0].dim)
        if mask.shape != (dim, dim):
            raise SupportClosureError(
                f"support component {component!r} mask has shape {mask.shape}, expected {(dim, dim)}"
            )
        exact_structural_closure = _exact_structural_support_is_group_adjoint_closed(
            mask,
            group=group,
        )
        if exact_structural_closure:
            max_bound = 0.0
            all_raw_inside = True
            for seed in component_seeds:
                raw = {
                    key: _canonical_csr(value)
                    for key, value in seed.coefficients.items()
                }
                raw_norm = _coefficient_norm(raw)
                bound = _propagated_error_bound(
                    raw_norm,
                    dim=dim,
                    degree=coordinate.max_degree,
                    group=group,
                ) + float(seed.metadata.get("coordinate_conversion_error_bound", 0.0))
                max_bound = max(max_bound, bound)
                if _outside_support_norm(raw, mask) != 0.0:
                    all_raw_inside = False
                    break
            if all_raw_inside:
                certificates[component] = {
                    "certified": True,
                    "seed_count": int(len(component_seeds)),
                    "leakage_before_cleanup": 0.0,
                    "propagated_error_bound": float(max_bound),
                    "cleanup_allowed": True,
                    "cleaned_component_count": 0,
                    "cleaned_components": [],
                    "support_closure_compiler": (
                        "exact_boolean_adjoint_group_support_v1"
                    ),
                }
                continue
        max_leakage = 0.0
        max_bound = 0.0
        cleaned_by_location: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        for seed in component_seeds:
            raw = {key: _canonical_csr(value) for key, value in seed.coefficients.items()}
            raw_norm = _coefficient_norm(raw)
            bound = _propagated_error_bound(
                raw_norm,
                dim=dim,
                degree=coordinate.max_degree,
                group=group,
            ) + float(seed.metadata.get("coordinate_conversion_error_bound", 0.0))
            max_bound = max(max_bound, bound)
            raw_leakage = _outside_support_norm(raw, mask)
            for record in _outside_support_components(raw, mask):
                location = (*record["monomial"], int(record["row"]), int(record["col"]))
                cleaned_by_location[location] = record
            if raw_leakage > bound:
                raise SupportClosureError(
                    f"support component {component!r} excludes its raw seed {seed.seed_id!r}: "
                    f"leakage={raw_leakage:.3e}, bound={bound:.3e}"
                )
            adjoint = _adjoint_polynomial(raw)
            adjoint_leakage = _outside_support_norm(adjoint, mask)
            for record in _outside_support_components(adjoint, mask):
                location = (*record["monomial"], int(record["row"]), int(record["col"]))
                cleaned_by_location[location] = record
            max_leakage = max(max_leakage, raw_leakage, adjoint_leakage)
            if adjoint_leakage > bound:
                raise SupportClosureError(
                    f"support component {component!r} is not adjoint closed for seed {seed.seed_id!r}: "
                    f"leakage={adjoint_leakage:.3e}, bound={bound:.3e}"
                )
            for element in group.elements:
                transformed = _apply_group_element(raw, element, coordinate)
                leakage = _outside_support_norm(transformed, mask)
                for record in _outside_support_components(transformed, mask):
                    location = (*record["monomial"], int(record["row"]), int(record["col"]))
                    cleaned_by_location[location] = record
                max_leakage = max(max_leakage, leakage)
                if leakage > bound:
                    raise SupportClosureError(
                        f"support component {component!r} is not closed under word {element.canonical_word}: "
                        f"leakage={leakage:.3e}, bound={bound:.3e}"
                    )
                # The unitary and antiunitary actions both commute with the
                # polynomial adjoint.  Reuse the already evaluated group image
                # instead of repeating U X U^dagger for g(X^dagger).
                transformed_adjoint = _adjoint_polynomial(transformed)
                adjoint_leakage = _outside_support_norm(transformed_adjoint, mask)
                for record in _outside_support_components(transformed_adjoint, mask):
                    location = (*record["monomial"], int(record["row"]), int(record["col"]))
                    cleaned_by_location[location] = record
                max_leakage = max(max_leakage, adjoint_leakage)
                if adjoint_leakage > bound:
                    raise SupportClosureError(
                        f"support component {component!r} is not closed under adjoint word "
                        f"{element.canonical_word}: leakage={adjoint_leakage:.3e}, "
                        f"bound={bound:.3e}"
                    )
        certificates[component] = {
            "certified": True,
            "seed_count": int(len(component_seeds)),
            "leakage_before_cleanup": float(max_leakage),
            "propagated_error_bound": float(max_bound),
            "cleanup_allowed": bool(max_leakage <= max_bound),
            "cleaned_component_count": int(len(cleaned_by_location)),
            "cleaned_components": [
                cleaned_by_location[location]
                for location in sorted(cleaned_by_location)
            ],
            "support_closure_compiler": "per_seed_propagated_error_v1",
        }
    return certificates


def compile_candidate_responses(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: FiniteGroup,
    null_policy: NullClassificationPolicy | None = None,
    support_masks: Mapping[str, np.ndarray] | None = None,
    _components_by_seed: Mapping[str, Sequence[str]] | None = None,
    internal_actions_by_word: Mapping[
        tuple[str, ...], sparse.spmatrix | np.ndarray
    ]
    | None = None,
    internal_action_absolute_error_bound: float = 0.0,
) -> CandidateResponseSet:
    if not seeds:
        raise ValueError("complete response compilation requires at least one raw seed")
    if not group.elements:
        raise ValueError("complete response compilation requires a non-empty finite group")
    dim = int(seeds[0].dim)
    if any(int(seed.dim) != dim for seed in seeds):
        raise ValueError("all raw response seeds must use the same matrix dimension")
    for element in group.elements:
        if np.asarray(element.internal_u).shape != (dim, dim):
            raise ValueError(
                f"finite-group internal action has shape {np.asarray(element.internal_u).shape}, expected {(dim, dim)}"
            )
    internal_actions = (
        {
            tuple(str(value) for value in word): _canonical_csr(
                matrix,
                shape=(dim, dim),
            )
            for word, matrix in internal_actions_by_word.items()
        }
        if internal_actions_by_word is not None
        else None
    )
    if internal_actions is not None:
        expected_words = {
            tuple(str(value) for value in element.canonical_word)
            for element in group.elements
        }
        if set(internal_actions) != expected_words:
            raise ValueError(
                "factorized internal actions must cover every canonical group word exactly"
            )
    internal_monomial_actions = (
        {
            word: _monomial_similarity_action(matrix)
            for word, matrix in internal_actions.items()
        }
        if internal_actions is not None
        else {}
    )
    polynomial_pullbacks_by_word = {
        tuple(str(value) for value in element.canonical_word): (
            _monomial_pullback_table(element, coordinate)
        )
        for element in group.elements
    }
    internal_action_error = float(internal_action_absolute_error_bound)
    if not np.isfinite(internal_action_error) or internal_action_error < 0.0:
        raise ValueError("internal action absolute error bound must be finite and nonnegative")
    policy = null_policy or NullClassificationPolicy()
    support_artifact = (
        certify_support_closure(
            seeds,
            group=group,
            coordinate=coordinate,
            support_masks=support_masks,
        )
        if support_masks is not None
        else {}
    )
    identity_direct = False
    if len(group.elements) == 1:
        identity_direct = _is_strict_identity_group_element(
            group.elements[0],
            dim=dim,
        )
    channels: list[ResponseChannel] = []
    for seed in seeds:
        raw = {
            (int(r), int(s)): _canonical_csr(matrix, shape=(dim, dim))
            for (r, s), matrix in seed.coefficients.items()
        }
        if any(r < 0 or s < 0 or r + s > coordinate.max_degree for r, s in raw):
            raise ValueError(f"seed {seed.seed_id!r} contains monomials outside the coordinate basis")
        raw_norm = _coefficient_norm(raw)
        error_components = _propagated_error_components(
            raw_norm,
            dim=dim,
            degree=coordinate.max_degree,
            group=group,
        )
        error_components["support_cleanup"] = float(
            support_artifact.get(seed.support_component, {}).get("leakage_before_cleanup", 0.0)
        )
        error_components["coordinate_conversion"] = float(
            seed.metadata.get("coordinate_conversion_error_bound", 0.0)
        )
        if internal_actions is not None:
            error_components["factorized_internal_action"] = float(
                (2.0 * internal_action_error + internal_action_error**2)
                * max(raw_norm, np.finfo(float).tiny)
            )
        error_bound = float(sum(error_components.values()))
        requested_components = (
            tuple(_components_by_seed.get(seed.seed_id, ()))
            if _components_by_seed is not None
            else ("real", "imag")
        )
        if not requested_components or any(
            component not in {"real", "imag"} for component in requested_components
        ):
            raise ValueError(
                f"invalid response components for seed {seed.seed_id!r}: {requested_components}"
            )
        reynolds_by_component: dict[
            str,
            dict[tuple[int, int], sparse.csr_matrix],
        ] = {component: {} for component in requested_components}
        for element in group.elements:
            # Transform the authored raw seed once.  For an antiunitary action,
            # g(iX)=-i g(X); for a unitary action, g(iX)=+i g(X).  Accumulating
            # both real channels from the shared raw image preserves the exact
            # Reynolds/P_H formula while avoiding a second sparse U X U^dagger
            # transform for every seed and group element.
            transformed = (
                raw
                if identity_direct
                else _apply_group_element(
                    raw,
                    element,
                    coordinate,
                    monomial_pullbacks=polynomial_pullbacks_by_word[
                        tuple(str(value) for value in element.canonical_word)
                    ],
                    internal_monomial_action=internal_monomial_actions.get(
                        tuple(str(value) for value in element.canonical_word)
                    ),
                    internal_u_override=(
                        None
                        if internal_actions is None
                        else internal_actions[
                            tuple(str(value) for value in element.canonical_word)
                        ]
                    ),
                )
            )
            if support_masks is not None:
                transformed = _mask_polynomial_support(
                    transformed,
                    np.asarray(support_masks[seed.support_component], dtype=bool),
                )
            for component in requested_components:
                if component == "real":
                    contribution = transformed
                else:
                    phase = -1.0j if element.antiunitary else 1.0j
                    contribution = {
                        key: _canonical_csr(matrix * phase)
                        for key, matrix in transformed.items()
                    }
                _sum_polynomial_coefficients(
                    reynolds_by_component[component],
                    contribution,
                )
        for component in requested_components:
            reynolds = {
                key: _canonical_csr(matrix * (1.0 / float(len(group.elements))))
                for key, matrix in reynolds_by_component[component].items()
            }
            projected = _hermitian_project(reynolds, dim=dim)
            norm = _coefficient_norm(projected)
            # An empty numerical result is not, by itself, a proof of a structural
            # zero.  Until a discrete symbolic certificate is attached to the seed,
            # classify it conservatively against the propagated absolute error.
            classification = policy.classify(norm, error_bound, structural=False)
            channels.append(
                ResponseChannel(
                    channel_id=f"{seed.seed_id}:{component}",
                    seed_id=seed.seed_id,
                    component=component,
                    coefficients=projected or {(0, 0): sparse.csr_matrix((dim, dim), dtype=np.complex128)},
                    unnormalized_norm=norm,
                    propagated_error_bound=error_bound,
                    classification=classification,
                    response_scale=norm,
                    support_component=str(seed.support_component),
                    metadata=dict(seed.metadata),
                    support_leakage_before_cleanup=float(
                        support_artifact.get(seed.support_component, {}).get("leakage_before_cleanup", 0.0)
                    ),
                    error_bound_components=dict(error_components),
                )
            )
    return CandidateResponseSet(
        coordinate=coordinate,
        channels=tuple(channels),
        dim=dim,
        group_artifact=group.artifact(),
        null_policy=policy,
        support_artifact=support_artifact,
    )


def compile_candidate_responses_adjoint_canonicalized(
    seeds: Sequence[RawPolynomialSeed],
    *,
    joint_keys: Mapping[str, Any],
    coordinate: PolynomialCoordinateBasis,
    group: FiniteGroup,
    null_policy: NullClassificationPolicy | None = None,
    support_masks: Mapping[str, np.ndarray] | None = None,
    internal_actions_by_word: Mapping[
        tuple[str, ...], sparse.spmatrix | np.ndarray
    ]
    | None = None,
    internal_action_absolute_error_bound: float = 0.0,
) -> CandidateResponseSet:
    """Compile one Reynolds representative per raw-coefficient-certified adjoint orbit."""

    from .response_basis_adjoint import (
        JointAdjointCanonicalization,
        JointAdjointSeed,
        canonicalize_joint_adjoint_seeds,
        certify_joint_adjoint_coefficients,
    )

    by_id = {seed.seed_id: seed for seed in seeds}
    if set(by_id) != {str(seed_id) for seed_id in joint_keys}:
        raise ValueError("joint-adjoint keys must cover every logical raw seed exactly once")
    provisional = canonicalize_joint_adjoint_seeds(
        [JointAdjointSeed(seed.seed_id, joint_keys[seed.seed_id]) for seed in seeds]
    )
    absolute_bounds_by_seed = {
        seed.seed_id: (
            _propagated_error_bound(
                _coefficient_norm(
                    {
                        (int(r), int(s)): _canonical_csr(
                            matrix, shape=(seed.dim, seed.dim)
                        )
                        for (r, s), matrix in seed.coefficients.items()
                    }
                ),
                dim=seed.dim,
                degree=coordinate.max_degree,
                group=group,
            )
            + float(seed.metadata.get("coordinate_conversion_error_bound", 0.0))
        )
        for seed in seeds
    }
    certified_orbits = []
    certified_mappings = []
    for orbit in provisional.orbits:
        member_ids = set(orbit.member_seed_ids)
        orbit_canonicalization = JointAdjointCanonicalization(
            orbits=(orbit,),
            mappings=tuple(
                mapping
                for mapping in provisional.mappings
                if mapping.seed_id in member_ids
            ),
        )
        orbit_certified = certify_joint_adjoint_coefficients(
            orbit_canonicalization,
            {
                seed_id: by_id[seed_id].coefficients
                for seed_id in orbit.member_seed_ids
            },
            absolute_error_bound=sum(
                absolute_bounds_by_seed[seed_id]
                for seed_id in orbit.member_seed_ids
            ),
        )
        certified_orbits.extend(orbit_certified.orbits)
        certified_mappings.extend(orbit_certified.mappings)
    certified = JointAdjointCanonicalization(
        orbits=tuple(certified_orbits),
        mappings=tuple(certified_mappings),
    )
    representatives = [by_id[seed_id] for seed_id in certified.representative_seed_ids]
    components_by_representative = {
        orbit.representative_seed_id: tuple(
            channel_id.rsplit(":", 1)[1] for channel_id in orbit.channel_ids
        )
        for orbit in certified.orbits
    }
    physical = compile_candidate_responses(
        representatives,
        coordinate=coordinate,
        group=group,
        null_policy=null_policy,
        support_masks=support_masks,
        _components_by_seed=components_by_representative,
        internal_actions_by_word=internal_actions_by_word,
        internal_action_absolute_error_bound=internal_action_absolute_error_bound,
    )
    physical_by_id = {
        (channel.seed_id, channel.component): channel for channel in physical.channels
    }
    logical_channels: list[ResponseChannel] = []
    for seed in seeds:
        for component in ("real", "imag"):
            mapping = certified.mapping_for(seed.seed_id, component)
            if mapping.structural_zero:
                logical_channels.append(
                    ResponseChannel(
                        channel_id=f"{seed.seed_id}:{component}",
                        seed_id=seed.seed_id,
                        component=component,
                        coefficients={
                            (0, 0): sparse.csr_matrix(
                                (seed.dim, seed.dim), dtype=np.complex128
                            )
                        },
                        unnormalized_norm=0.0,
                        propagated_error_bound=absolute_bounds_by_seed[seed.seed_id],
                        classification="structural_zero",
                        response_scale=0.0,
                        support_component=seed.support_component,
                        metadata={
                            **dict(seed.metadata),
                            "adjoint_certification_status": mapping.certification_status,
                            "adjoint_structural_zero": True,
                        },
                    )
                )
                continue
            assert mapping.channel_id is not None
            representative_seed_id, representative_component = mapping.channel_id.rsplit(":", 1)
            representative = physical_by_id[(representative_seed_id, representative_component)]
            is_dependency = seed.seed_id != representative_seed_id
            logical_channels.append(
                replace(
                    representative,
                    channel_id=f"{seed.seed_id}:{component}",
                    seed_id=seed.seed_id,
                    coefficients={
                        key: _canonical_csr(matrix * int(mapping.sign))
                        for key, matrix in representative.coefficients.items()
                    },
                    metadata={
                        **dict(seed.metadata),
                        "adjoint_certification_status": mapping.certification_status,
                        "adjoint_representative_channel_id": mapping.channel_id,
                        "adjoint_sign": int(mapping.sign),
                        "adjoint_certified_dependency": bool(is_dependency),
                    },
                )
            )
    return CandidateResponseSet(
        coordinate=coordinate,
        channels=tuple(logical_channels),
        dim=physical.dim,
        group_artifact=physical.group_artifact,
        null_policy=physical.null_policy,
        support_artifact=physical.support_artifact,
        adjoint_artifact={
            "certification": "raw_polynomial_coefficient_adjoint_v1",
            "logical_candidate_channel_count": int(len(logical_channels)),
            "authored_ordered_seed_count": int(len(seeds)),
            "adjoint_orbit_descriptor_count": int(len(certified.orbits)),
            "hermitian_ambient_channel_count": int(len(physical.channels)),
            "physically_compiled_representative_channel_count": int(len(physical.channels)),
            "adjoint_certified_dropped_channel_count": int(
                len(logical_channels) - len(physical.channels)
            ),
        },
    )


def raw_polynomial_seed_from_term_key(
    key: Any,
    *,
    seed_id: str,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    n_orb1: int,
    n_orb2: int,
    coordinate: PolynomialCoordinateBasis,
    support_component: str,
    metadata: Mapping[str, Any],
    p_tolerance: float = 1.0e-5,
) -> RawPolynomialSeed:
    qset1 = np.asarray(Q_set1, dtype=np.float64)
    qset2 = np.asarray(Q_set2, dtype=np.float64)
    if qset1.ndim != 2 or qset1.shape[1] != 2 or qset2.ndim != 2 or qset2.shape[1] != 2:
        raise ValueError("Q sets must have shape (N,2)")
    layer_from = int(key.layer_from)
    layer_to = int(key.layer_to)
    q_rows = qset1 if layer_from == 1 else qset2
    q_cols = qset1 if layer_to == 1 else qset2
    n_from = int(n_orb1 if layer_from == 1 else n_orb2)
    n_to = int(n_orb1 if layer_to == 1 else n_orb2)
    orbital_from = int(key.orbital_from) - 1
    orbital_to = int(key.orbital_to) - 1
    if not (0 <= orbital_from < n_from and 0 <= orbital_to < n_to):
        raise ValueError(f"term key orbital indices are out of range: {key}")
    dim = int(qset1.shape[0] * int(n_orb1) + qset2.shape[0] * int(n_orb2))

    def global_index(layer: int, q_index: int, orbital: int) -> int:
        if layer == 1:
            return int(orbital * qset1.shape[0] + q_index)
        return int(qset1.shape[0] * int(n_orb1) + orbital * qset2.shape[0] + q_index)

    rows_by_monomial: dict[tuple[int, int], list[int]] = {}
    cols_by_monomial: dict[tuple[int, int], list[int]] = {}
    values_by_monomial: dict[tuple[int, int], list[complex]] = {}
    p_vector = np.asarray(key.p, dtype=np.float64)
    mz = int(key.Mz)
    mz_star = int(key.Mz_star)
    for row_q_index, q_row in enumerate(q_rows):
        matches = np.flatnonzero(np.linalg.norm(q_row - p_vector - q_cols, axis=1) < float(p_tolerance))
        if matches.size == 0:
            continue
        delta = complex(*(np.asarray(coordinate.origin) - q_row)) / float(coordinate.scale)
        centered_w = {(1, 0): 1.0 + 0.0j, (0, 0): delta}
        centered_wbar = {(0, 1): 1.0 + 0.0j, (0, 0): np.conjugate(delta)}
        polynomial = _poly_multiply(_poly_power(centered_w, mz), _poly_power(centered_wbar, mz_star))
        physical_scale = float(coordinate.scale) ** int(mz + mz_star)
        polynomial = {
            monomial: physical_scale * prefactor
            for monomial, prefactor in polynomial.items()
        }
        row = global_index(layer_from, row_q_index, orbital_from)
        for col_q_index in matches:
            col = global_index(layer_to, int(col_q_index), orbital_to)
            for monomial, prefactor in polynomial.items():
                rows_by_monomial.setdefault(monomial, []).append(row)
                cols_by_monomial.setdefault(monomial, []).append(col)
                values_by_monomial.setdefault(monomial, []).append(complex(prefactor))
    coefficients = {
        monomial: _canonical_csr(
            sparse.coo_matrix(
                (
                    np.asarray(values_by_monomial[monomial], dtype=np.complex128),
                    (
                        np.asarray(rows_by_monomial[monomial], dtype=np.int64),
                        np.asarray(cols_by_monomial[monomial], dtype=np.int64),
                    ),
                ),
                shape=(dim, dim),
            )
        )
        for monomial in rows_by_monomial
    }
    if not coefficients:
        coefficients = {(0, 0): sparse.csr_matrix((dim, dim), dtype=np.complex128)}
    conversion_norm = _coefficient_norm(coefficients)
    coordinate_conversion_error_bound = float(
        np.finfo(float).eps
        * float(32 + 8 * max(0, mz + mz_star))
        * max(conversion_norm, np.finfo(float).tiny)
    )
    return RawPolynomialSeed(
        seed_id=str(seed_id),
        coefficients=coefficients,
        support_component=str(support_component),
        metadata={
            **dict(metadata),
            "coordinate_conversion_error_bound": coordinate_conversion_error_bound,
            "term_key": {
                "Mz": mz,
                "Mz_star": mz_star,
                "layer_from": layer_from,
                "layer_to": layer_to,
                "orbital_from": int(key.orbital_from),
                "orbital_to": int(key.orbital_to),
                "p": [float(value) for value in key.p],
            },
        },
    )


def _zero_harmonic_joint_adjoint_keys(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
) -> dict[str, Any]:
    """Build fail-closed joint keys for the production kinetic/onsite p=0 vocabulary."""

    from .response_basis_adjoint import (
        CanonicalDimensionlessCenter,
        JointAdjointSeedKey,
    )

    center = CanonicalDimensionlessCenter.from_dimensionless((0.0, 0.0))
    keys: dict[str, Any] = {}
    for seed in seeds:
        term_key = seed.metadata.get("term_key")
        if not isinstance(term_key, Mapping):
            raise ValueError(
                f"adjoint-canonicalized production seed {seed.seed_id!r} lacks term_key metadata"
            )
        p_vector = np.asarray(term_key.get("p"), dtype=np.float64)
        if p_vector.shape != (2,) or not np.all(np.isfinite(p_vector)):
            raise ValueError(f"seed {seed.seed_id!r} has invalid explicit harmonic vector")
        if np.any(p_vector != 0.0):
            raise ValueError(
                "adjoint-canonicalized production currently supports only explicit p=0 "
                f"kinetic/onsite seeds; seed {seed.seed_id!r} has p={p_vector.tolist()}"
            )
        support = str(seed.support_component)
        layer_from = int(term_key["layer_from"])
        layer_to = int(term_key["layer_to"])
        keys[seed.seed_id] = JointAdjointSeedKey(
            sector_from=f"{support}|layer-{layer_from}",
            sector_to=f"{support}|layer-{layer_to}",
            orbital_from=int(term_key["orbital_from"]) - 1,
            orbital_to=int(term_key["orbital_to"]) - 1,
            monomial=(int(term_key["Mz"]), int(term_key["Mz_star"])),
            center=center,
            harmonic_id=f"{support}|p-zero",
            adjoint_harmonic_id=f"{support}|p-zero",
        )
    return keys


def _merge_support_certificates(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Combine independently certified seed partitions without weakening bounds."""

    merged: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        for component, raw_record in artifact.items():
            record = dict(raw_record)
            if component not in merged:
                merged[component] = record
                continue
            current = merged[component]
            current["certified"] = bool(current.get("certified", False)) and bool(
                record.get("certified", False)
            )
            current["seed_count"] = int(current.get("seed_count", 0)) + int(
                record.get("seed_count", 0)
            )
            current["leakage_before_cleanup"] = max(
                float(current.get("leakage_before_cleanup", 0.0)),
                float(record.get("leakage_before_cleanup", 0.0)),
            )
            current["propagated_error_bound"] = max(
                float(current.get("propagated_error_bound", 0.0)),
                float(record.get("propagated_error_bound", 0.0)),
            )
            current["cleanup_allowed"] = bool(current.get("cleanup_allowed", False)) and bool(
                record.get("cleanup_allowed", False)
            )
            cleaned: dict[str, Any] = {}
            for item in [
                *list(current.get("cleaned_components", [])),
                *list(record.get("cleaned_components", [])),
            ]:
                cleaned[_canonical_json(item)] = item
            current["cleaned_components"] = [cleaned[key] for key in sorted(cleaned)]
            current["cleaned_component_count"] = int(len(cleaned))
    return merged


def _compile_candidate_group_with_adjoint_fallback(
    seeds: Sequence[RawPolynomialSeed],
    *,
    coordinate: PolynomialCoordinateBasis,
    group: FiniteGroup,
    support_masks: Mapping[str, np.ndarray] | None,
    null_policy: NullClassificationPolicy | None = None,
    internal_actions_by_word: Mapping[
        tuple[str, ...], sparse.spmatrix | np.ndarray
    ]
    | None = None,
    internal_action_absolute_error_bound: float = 0.0,
    factorized_reynolds_artifact: Mapping[str, Any] | None = None,
) -> CandidateResponseSet:
    """Compile p=0 adjoint orbits and generic finite-p seeds in one channel set.

    A finite Fourier harmonic need not have an authored ``-p`` seed: the
    unconditional Hermitian projector already supplies its adjoint response.
    Moreover, center conversion makes a finite-p adjoint a binomial combination
    of the reversed seed's nominal degree and lower degrees, so the p=0
    one-to-one ``real/+imag`` mapping cannot be reused.  Finite-p seeds therefore
    take the complete per-seed Reynolds path; only the p=0 subset is deduplicated
    by the raw-coefficient-certified joint-adjoint optimization.
    """

    if not seeds:
        raise ValueError("complete response compilation requires at least one raw seed")
    zero_seeds: list[RawPolynomialSeed] = []
    finite_seeds: list[RawPolynomialSeed] = []
    for seed in seeds:
        term_key = seed.metadata.get("term_key")
        if not isinstance(term_key, Mapping):
            raise ValueError(
                f"adjoint-canonicalized production seed {seed.seed_id!r} lacks term_key metadata"
            )
        p_vector = np.asarray(term_key.get("p"), dtype=np.float64)
        if p_vector.shape != (2,) or not np.all(np.isfinite(p_vector)):
            raise ValueError(f"seed {seed.seed_id!r} has invalid explicit harmonic vector")
        if np.all(p_vector == 0.0):
            zero_seeds.append(seed)
        else:
            finite_seeds.append(seed)

    zero_candidates: CandidateResponseSet | None = None
    zero_fallback_reason: str | None = None
    if zero_seeds:
        from .response_basis_adjoint import (
            AdjointCertificationError,
            AdjointClosureError,
        )

        try:
            zero_candidates = compile_candidate_responses_adjoint_canonicalized(
                zero_seeds,
                joint_keys=_zero_harmonic_joint_adjoint_keys(
                    zero_seeds,
                    coordinate=coordinate,
                ),
                coordinate=coordinate,
                group=group,
                null_policy=null_policy,
                support_masks=support_masks,
                internal_actions_by_word=internal_actions_by_word,
                internal_action_absolute_error_bound=internal_action_absolute_error_bound,
            )
        except (AdjointClosureError, AdjointCertificationError) as exc:
            # Pairwise canonicalization is only an optimization.  A valid
            # authored vocabulary need not contain a raw adjoint partner;
            # Reynolds followed by P_H still yields its two complete real
            # response channels.
            zero_fallback_reason = type(exc).__name__
            zero_candidates = compile_candidate_responses(
                zero_seeds,
                coordinate=coordinate,
                group=group,
                null_policy=null_policy,
                support_masks=support_masks,
                internal_actions_by_word=internal_actions_by_word,
                internal_action_absolute_error_bound=internal_action_absolute_error_bound,
            )
    if not finite_seeds:
        assert zero_candidates is not None
        if zero_fallback_reason is not None:
            return replace(
                zero_candidates,
                adjoint_artifact={
                    "certification": "p0_full_candidate_reynolds_fallback_v1",
                    "logical_candidate_channel_count": int(len(zero_candidates.channels)),
                    "physically_compiled_representative_channel_count": int(
                        len(zero_candidates.channels)
                    ),
                    "adjoint_certified_dropped_channel_count": 0,
                    "p0_dense_fallback_seed_count": int(len(zero_seeds)),
                    "p0_dense_fallback_channel_count": int(len(zero_candidates.channels)),
                    "p0_dense_fallback_reason": zero_fallback_reason,
                    "reynolds_internal_action": dict(factorized_reynolds_artifact or {}),
                },
            )
        return replace(
            zero_candidates,
            adjoint_artifact={
                **dict(zero_candidates.adjoint_artifact),
                "reynolds_internal_action": dict(factorized_reynolds_artifact or {}),
            },
        )

    finite_candidates = compile_candidate_responses(
        finite_seeds,
        coordinate=coordinate,
        group=group,
        null_policy=null_policy,
        support_masks=support_masks,
        internal_actions_by_word=internal_actions_by_word,
        internal_action_absolute_error_bound=internal_action_absolute_error_bound,
    )
    if zero_candidates is None:
        return CandidateResponseSet(
            coordinate=coordinate,
            channels=finite_candidates.channels,
            dim=finite_candidates.dim,
            group_artifact=finite_candidates.group_artifact,
            null_policy=finite_candidates.null_policy,
            support_artifact=finite_candidates.support_artifact,
            adjoint_artifact={
                "certification": "finite_p_full_candidate_reynolds_v1",
                "logical_candidate_channel_count": int(len(finite_candidates.channels)),
                "authored_ordered_seed_count": int(len(finite_seeds)),
                "adjoint_orbit_descriptor_count": int(len(finite_seeds)),
                "hermitian_ambient_channel_count": int(
                    len(finite_candidates.channels)
                ),
                "physically_compiled_representative_channel_count": int(
                    len(finite_candidates.channels)
                ),
                "adjoint_certified_dropped_channel_count": 0,
                "finite_p_fallback_seed_count": int(len(finite_seeds)),
                "finite_p_fallback_channel_count": int(len(finite_candidates.channels)),
                "reynolds_internal_action": dict(factorized_reynolds_artifact or {}),
            },
        )

    seed_order = {seed.seed_id: index for index, seed in enumerate(seeds)}
    component_order = {"real": 0, "imag": 1}
    channels = tuple(
        sorted(
            (*zero_candidates.channels, *finite_candidates.channels),
            key=lambda channel: (
                seed_order[channel.seed_id],
                component_order[channel.component],
            ),
        )
    )
    zero_adjoint = dict(zero_candidates.adjoint_artifact)
    zero_physical = int(
        zero_adjoint.get(
            "physically_compiled_representative_channel_count",
            len(zero_candidates.channels),
        )
    )
    zero_dropped = int(zero_adjoint.get("adjoint_certified_dropped_channel_count", 0))
    return CandidateResponseSet(
        coordinate=coordinate,
        channels=channels,
        dim=zero_candidates.dim,
        group_artifact=zero_candidates.group_artifact,
        null_policy=zero_candidates.null_policy,
        support_artifact=_merge_support_certificates(
            [zero_candidates.support_artifact, finite_candidates.support_artifact]
        ),
        adjoint_artifact={
            "certification": "p0_adjoint_certified__finite_p_full_reynolds_v1",
            "zero_harmonic": zero_adjoint,
            "logical_candidate_channel_count": int(len(channels)),
            "physically_compiled_representative_channel_count": int(
                zero_physical + len(finite_candidates.channels)
            ),
            "adjoint_certified_dropped_channel_count": zero_dropped,
            "finite_p_fallback_seed_count": int(len(finite_seeds)),
            "finite_p_fallback_channel_count": int(len(finite_candidates.channels)),
            "p0_dense_fallback_seed_count": (
                int(len(zero_seeds)) if zero_fallback_reason is not None else 0
            ),
            "p0_dense_fallback_channel_count": (
                int(len(zero_candidates.channels)) if zero_fallback_reason is not None else 0
            ),
            "p0_dense_fallback_reason": zero_fallback_reason,
            "reynolds_internal_action": dict(factorized_reynolds_artifact or {}),
        },
    )


def _channel_sparse_vector(
    channel: ResponseChannel,
    coordinate: PolynomialCoordinateBasis,
    dim: int,
) -> sparse.csc_matrix:
    monomial_index = {key: index for index, key in enumerate(coordinate.monomials)}
    real_rows: list[np.ndarray] = []
    imag_rows: list[np.ndarray] = []
    real_values: list[np.ndarray] = []
    imag_values: list[np.ndarray] = []
    block = len(coordinate.monomials) * dim * dim
    for key, matrix in channel.coefficients.items():
        if key not in monomial_index:
            raise ValueError(f"channel {channel.channel_id!r} contains undeclared monomial {key}")
        coo = matrix.tocoo()
        base = monomial_index[key] * dim * dim
        positions = base + coo.row.astype(np.int64) * dim + coo.col.astype(np.int64)
        real_mask = coo.data.real != 0.0
        imag_mask = coo.data.imag != 0.0
        if np.any(real_mask):
            real_rows.append(positions[real_mask])
            real_values.append(coo.data.real[real_mask])
        if np.any(imag_mask):
            imag_rows.append(block + positions[imag_mask])
            imag_values.append(coo.data.imag[imag_mask])
    rows = np.concatenate([*real_rows, *imag_rows]) if real_rows or imag_rows else np.empty(0, dtype=np.int64)
    values = np.concatenate([*real_values, *imag_values]) if real_values or imag_values else np.empty(0, dtype=np.float64)
    return sparse.csc_matrix(
        (values, (rows, np.zeros(rows.size, dtype=np.int64))),
        shape=(2 * block, 1),
        dtype=np.float64,
    )


def _reduce_confirmed_channels(
    channels: Sequence[ResponseChannel],
    *,
    coordinate: PolynomialCoordinateBasis,
    dim: int,
    confirm_factor: float,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    confirmed_indices = [
        index
        for index, channel in enumerate(channels)
        if channel.classification == "confirmed_nonzero"
    ]
    normalized_vectors = {
        index: _channel_sparse_vector(channels[index], coordinate, dim)
        / max(channels[index].response_scale, np.finfo(float).tiny)
        for index in confirmed_indices
    }
    parent = list(range(len(confirmed_indices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    row_owner: dict[int, int] = {}
    for local_index, global_index in enumerate(confirmed_indices):
        for row in np.unique(normalized_vectors[global_index].indices):
            row_int = int(row)
            owner = row_owner.get(row_int)
            if owner is None:
                row_owner[row_int] = local_index
            else:
                union(local_index, owner)
    components_by_root: dict[int, list[int]] = {}
    for local_index, global_index in enumerate(confirmed_indices):
        components_by_root.setdefault(find(local_index), []).append(global_index)
    components = sorted(
        components_by_root.values(),
        key=lambda indices: tuple(indices),
    )
    selected: set[int] = {
        index
        for index, channel in enumerate(channels)
        if channel.classification == "ambiguous"
    }
    proofs: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for component_index, indices in enumerate(components):
        vectors = sparse.hstack(
            [normalized_vectors[index] for index in indices],
            format="csc",
        )
        active_rows = np.unique(vectors.indices)
        coefficient_matrix = np.asarray(vectors[active_rows, :].toarray(), dtype=np.float64)
        singular_values = scipy.linalg.svdvals(coefficient_matrix, check_finite=False)
        relative_errors = np.asarray(
            [
                channels[index].propagated_error_bound
                / max(channels[index].unnormalized_norm, np.finfo(float).tiny)
                for index in indices
            ],
            dtype=np.float64,
        )
        tolerance = max(
            (
                np.finfo(float).eps
                * max(coefficient_matrix.shape)
                * float(singular_values[0])
                if singular_values.size
                else 0.0
            ),
            float(confirm_factor) * float(np.max(relative_errors, initial=0.0)),
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank:
            _q, _r, pivots = scipy.linalg.qr(
                coefficient_matrix,
                pivoting=True,
                mode="economic",
                check_finite=False,
            )
            local_selected = [int(value) for value in pivots[:rank]]
        else:
            local_selected = []
        selected_global = {indices[local] for local in local_selected}
        selected.update(selected_global)
        certified_selected_global = set(selected_global)
        selected_vectors = (
            coefficient_matrix[:, local_selected]
            if local_selected
            else np.empty((coefficient_matrix.shape[0], 0), dtype=np.float64)
        )
        if selected_vectors.shape[1] and len(local_selected) < len(indices):
            dependency_coefficients, *_ = scipy.linalg.lstsq(
                selected_vectors,
                coefficient_matrix,
                cond=None,
                lapack_driver="gelsy",
                check_finite=False,
            )
            component_residuals = np.linalg.norm(
                coefficient_matrix - selected_vectors @ dependency_coefficients,
                axis=0,
            )
        elif selected_vectors.shape[1]:
            component_residuals = np.zeros(len(indices), dtype=np.float64)
        else:
            component_residuals = np.linalg.norm(coefficient_matrix, axis=0)
        condition_roundoff = (
            float(
                singular_values[0]
                / singular_values[rank - 1]
                * np.finfo(float).eps
            )
            if rank and singular_values[rank - 1] > 0.0
            else 0.0
        )
        for local, global_index in enumerate(indices):
            if global_index in selected_global:
                continue
            residual = float(component_residuals[local])
            algebra_floor = max(float(relative_errors[local]), np.finfo(float).tiny)
            response_norm = float(channels[global_index].unnormalized_norm)
            outside_absolute = float(residual * response_norm)
            algebra_floor_absolute = float(
                channels[global_index].propagated_error_bound
                + condition_roundoff * response_norm
            )
            # The component-wide SVD tolerance is only a cheap proposal.  A
            # barely-confirmed column must never relax the deletion threshold
            # for a different direction.  Fail closed unless this individual
            # column is actually inside its own propagated algebra/solver floor.
            if outside_absolute > algebra_floor_absolute:
                selected.add(global_index)
                certified_selected_global.add(global_index)
                continue
            dropped.append(
                {
                    "channel_id": channels[global_index].channel_id,
                    "reason": "target_independent_linear_dependency",
                    "response_absolute_norm": response_norm,
                    "outside_span_residual": residual,
                    "outside_span_residual_absolute": outside_absolute,
                    "algebra_floor": algebra_floor,
                    "algebra_floor_absolute": algebra_floor_absolute,
                    "algebra_floor_ratio": outside_absolute
                    / max(algebra_floor_absolute, np.finfo(float).tiny),
                }
            )
        proofs.append(
            {
                "support_component": f"joint_group_adjoint_component_{component_index}",
                "authored_support_components": sorted(
                    {str(channels[index].support_component) for index in indices}
                ),
                "channel_ids": [channels[index].channel_id for index in indices],
                "selected_channel_ids": [
                    channels[index].channel_id
                    for index in indices
                    if index in certified_selected_global
                ],
                "real_rank": int(len(certified_selected_global)),
                "rrqr_proposed_rank": rank,
                "dependency_acceptance": "per_column_outside_span_not_above_own_algebra_floor_v1",
                "singular_values": [float(value) for value in singular_values],
                "rank_tolerance": float(tolerance),
                "rank_solver_backward_error": float(
                    np.finfo(float).eps
                    * max(coefficient_matrix.shape)
                    * (float(singular_values[0]) if singular_values.size else 0.0)
                ),
                "propagated_relative_algebra_error": float(
                    confirm_factor * float(np.max(relative_errors, initial=0.0))
                ),
                "condition_estimate": (
                    float(singular_values[0] / singular_values[rank - 1])
                    if rank and singular_values[rank - 1] > 0.0
                    else None
                ),
                "condition_roundoff_estimate": condition_roundoff,
                "solver": "real_coefficient_space_svd_rrqr",
                "certification_space": "global_two_dimensional_polynomial_coefficient_space_real_imag_stack",
                "sample_grid_used": False,
                "active_coefficient_rows": int(active_rows.size),
                "block_rule": "joint_group_adjoint_support_component",
                "direct_sum_proof": (
                    "disjoint_global_real_polynomial_coefficient_rows_after_group_and_adjoint_projection"
                ),
            }
        )
    retained_vectors = {
        index: _channel_sparse_vector(channels[index], coordinate, dim)
        for index in sorted(selected)
    }
    retained_support = {
        index: set(int(row) for row in retained_vectors[index].indices)
        for index in retained_vectors
    }
    for numerical_index, channel in enumerate(channels):
        if channel.classification != "numerical_zero":
            continue
        target = _channel_sparse_vector(channel, coordinate, dim)
        target_rows = set(int(row) for row in target.indices)
        overlapping = [
            index
            for index in sorted(retained_vectors)
            if target_rows & retained_support[index]
        ]
        if target.nnz == 0 or not overlapping:
            outside_absolute = float(np.linalg.norm(target.data))
            condition_roundoff_absolute = 0.0
        else:
            active_rows = np.asarray(
                sorted(
                    target_rows.union(
                        *(retained_support[index] for index in overlapping)
                    )
                ),
                dtype=np.int64,
            )
            design = np.asarray(
                sparse.hstack(
                    [retained_vectors[index] for index in overlapping],
                    format="csc",
                )[active_rows, :].toarray(),
                dtype=np.float64,
            )
            target_vector = np.asarray(target[active_rows, :].toarray(), dtype=np.float64).ravel()
            projection_coefficients, *_ = scipy.linalg.lstsq(
                design,
                target_vector,
                cond=None,
                lapack_driver="gelsy",
                check_finite=False,
            )
            outside_absolute = float(
                np.linalg.norm(target_vector - design @ projection_coefficients)
            )
            singular_values = scipy.linalg.svdvals(design, check_finite=False)
            positive = singular_values[singular_values > 0.0]
            condition = (
                float(positive[0] / positive[-1])
                if positive.size
                else 0.0
            )
            condition_roundoff_absolute = float(
                condition * np.finfo(float).eps * max(float(channel.unnormalized_norm), np.finfo(float).tiny)
            )
        algebra_floor_absolute = float(
            channel.propagated_error_bound + condition_roundoff_absolute
        )
        dropped.append(
            {
                "channel_id": channel.channel_id,
                "reason": "certified_numerical_zero",
                "response_absolute_norm": float(channel.unnormalized_norm),
                "outside_span_residual_absolute": outside_absolute,
                "algebra_floor_absolute": algebra_floor_absolute,
                "algebra_floor_ratio": outside_absolute
                / max(algebra_floor_absolute, np.finfo(float).tiny),
                "propagated_error_bound": float(channel.propagated_error_bound),
            }
        )
    return sorted(selected), proofs, dropped


@dataclass(frozen=True)
class FittedResponseModel:
    basis_hash: str
    coefficients: np.ndarray = field(repr=False, compare=False)
    fit_pivots: tuple[int, ...]
    fit_selected_channel_ids: tuple[str, ...]
    nonzero_channel_ids: tuple[str, ...]
    regularization: float
    fit_indices: tuple[int, ...]
    band_window: tuple[int, ...]
    fit_hash: str
    coefficient_tolerance: float
    fit_solver_channel_indices: tuple[int, ...] = ()
    fit_solver_channel_ids: tuple[str, ...] = ()
    coefficient_normalization: str = COEFFICIENT_NORMALIZATION_V1
    regularization_normalization: str = REGULARIZATION_NORMALIZATION_V1
    fit_channel_policy: str = FIT_CHANNEL_POLICY_V2
    fit_solver_policy: str = FIT_SOLVER_POLICY_V2
    fit_design_singular_values: tuple[float, ...] = ()
    fit_design_machine_error_bound: float = 0.0
    fit_design_propagated_error_bound: float = 0.0
    fit_design_rank_tolerance: float = 0.0
    fit_design_certified_rank: int = 0
    fit_selected_condition_number: float = 0.0
    fit_selected_singular_value_min: float = 0.0
    fit_selected_error_bound: float = 0.0
    max_response_amplitude: float = 0.0
    fit_objective: Mapping[str, Any] = field(default_factory=_equal_matrix_fit_objective)

    def __post_init__(self) -> None:
        values = np.array(self.coefficients, dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(values)):
            raise ValueError("fitted response coefficients must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "coefficients", values)
        singular_values = tuple(float(value) for value in self.fit_design_singular_values)
        singular_array = np.asarray(singular_values, dtype=np.float64)
        if singular_array.size and (
            not np.all(np.isfinite(singular_array))
            or np.any(singular_array < 0.0)
            or np.any(np.diff(singular_array) > 0.0)
        ):
            raise ValueError(
                "fit design singular values must be finite, non-negative, and descending"
            )
        object.__setattr__(self, "fit_design_singular_values", singular_values)
        nonnegative_diagnostics = {
            "fit_design_machine_error_bound": self.fit_design_machine_error_bound,
            "fit_design_propagated_error_bound": self.fit_design_propagated_error_bound,
            "fit_design_rank_tolerance": self.fit_design_rank_tolerance,
            "fit_selected_condition_number": self.fit_selected_condition_number,
            "fit_selected_singular_value_min": self.fit_selected_singular_value_min,
            "fit_selected_error_bound": self.fit_selected_error_bound,
            "max_response_amplitude": self.max_response_amplitude,
        }
        for name, raw_value in nonnegative_diagnostics.items():
            numeric = float(raw_value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, numeric)
        certified_rank = int(self.fit_design_certified_rank)
        if certified_rank < 0:
            raise ValueError("fit design certified rank must be non-negative")
        object.__setattr__(self, "fit_design_certified_rank", certified_rank)
        solver_indices = tuple(int(value) for value in self.fit_solver_channel_indices)
        solver_ids = tuple(str(value) for value in self.fit_solver_channel_ids)
        if not solver_indices and not solver_ids and self.fit_selected_channel_ids:
            solver_indices = tuple(int(value) for value in self.fit_pivots)
            solver_ids = tuple(str(value) for value in self.fit_selected_channel_ids)
        if len(solver_indices) != len(solver_ids):
            raise ValueError("fit solver channel indices must match solver channel ids")
        object.__setattr__(self, "fit_solver_channel_indices", solver_indices)
        object.__setattr__(self, "fit_solver_channel_ids", solver_ids)
        objective = _deep_immutable_copy(self.fit_objective)
        if not isinstance(objective, Mapping) or not str(objective.get("mode", "")):
            raise ValueError("fitted response objective must record a non-empty mode")
        object.__setattr__(self, "fit_objective", objective)
        if not np.isfinite(float(self.regularization)) or float(self.regularization) < 0.0:
            raise ValueError("regularization must be finite and non-negative")
        if self.fit_solver_policy not in {
            FIT_SOLVER_POLICY_V2,
            FIT_SOLVER_POLICY_RIDGE_V3,
            FIT_SOLVER_POLICY_LEGACY,
        }:
            raise ValueError(f"unsupported fit solver policy: {self.fit_solver_policy!r}")
        if self.fit_solver_policy == FIT_SOLVER_POLICY_V2:
            if certified_rank != len(self.fit_pivots) or certified_rank != len(
                self.fit_selected_channel_ids
            ):
                raise ValueError(
                    "fit design certified rank must match fit pivots and selected channel ids"
                )
            if certified_rank > len(singular_values):
                raise ValueError("fit design certified rank exceeds stored singular spectrum")
            if certified_rank and self.fit_selected_singular_value_min <= self.fit_selected_error_bound:
                raise ValueError(
                    "selected fit response directions are not certified above their error bound"
                )
        elif self.fit_solver_policy == FIT_SOLVER_POLICY_RIDGE_V3:
            if self.regularization <= 0.0:
                raise ValueError("ridge fit solver policy requires positive regularization")
            if self.fit_pivots or self.fit_selected_channel_ids:
                raise ValueError("ridge fit must not report solver channels as RRQR pivots")
            if not solver_ids:
                raise ValueError("ridge fit must record its solver channels")
            if certified_rank > len(singular_values):
                raise ValueError("fit design certified rank exceeds stored singular spectrum")
            if solver_ids and self.fit_selected_singular_value_min <= 0.0:
                raise ValueError("regularized fit design must have positive stabilized singular value")

    def with_coefficients(
        self,
        coefficients: Sequence[float],
        *,
        channel_ids: Sequence[str],
        response_scales: Sequence[float],
        provenance: Mapping[str, Any],
    ) -> "FittedResponseModel":
        """Return a target-dependent fit state without mutating its compiled basis."""
        values = np.asarray(coefficients, dtype=np.float64).copy()
        ids = tuple(str(value) for value in channel_ids)
        scales = np.asarray(response_scales, dtype=np.float64)
        if values.shape != (len(ids),):
            raise ValueError(f"coefficient vector has shape {values.shape}, expected {(len(ids),)}")
        if scales.shape != values.shape:
            raise ValueError(f"response scale vector has shape {scales.shape}, expected {values.shape}")
        amplitude = np.abs(values) * scales
        values[amplitude <= float(self.coefficient_tolerance)] = 0.0
        amplitude = np.abs(values) * scales
        nonzero = tuple(
            ids[index]
            for index in np.flatnonzero(amplitude > float(self.coefficient_tolerance))
        )
        fit_payload = {
            "basis_hash": self.basis_hash,
            "parent_fit_hash": self.fit_hash,
            "coefficients": _json_record(values),
            "provenance": _json_record(provenance),
        }
        return FittedResponseModel(
            basis_hash=self.basis_hash,
            coefficients=values.copy(),
            fit_pivots=self.fit_pivots,
            fit_selected_channel_ids=self.fit_selected_channel_ids,
            nonzero_channel_ids=nonzero,
            regularization=self.regularization,
            fit_indices=self.fit_indices,
            band_window=self.band_window,
            fit_hash=hashlib.sha256(_canonical_json(fit_payload).encode("utf-8")).hexdigest(),
            coefficient_tolerance=self.coefficient_tolerance,
            fit_solver_channel_indices=self.fit_solver_channel_indices,
            fit_solver_channel_ids=self.fit_solver_channel_ids,
            coefficient_normalization=self.coefficient_normalization,
            regularization_normalization=self.regularization_normalization,
            fit_channel_policy=self.fit_channel_policy,
            fit_solver_policy=self.fit_solver_policy,
            fit_design_singular_values=self.fit_design_singular_values,
            fit_design_machine_error_bound=self.fit_design_machine_error_bound,
            fit_design_propagated_error_bound=self.fit_design_propagated_error_bound,
            fit_design_rank_tolerance=self.fit_design_rank_tolerance,
            fit_design_certified_rank=self.fit_design_certified_rank,
            fit_selected_condition_number=self.fit_selected_condition_number,
            fit_selected_singular_value_min=self.fit_selected_singular_value_min,
            fit_selected_error_bound=self.fit_selected_error_bound,
            max_response_amplitude=float(np.max(amplitude)) if amplitude.size else 0.0,
            fit_objective=self.fit_objective,
        )

    def artifact(self) -> dict[str, Any]:
        return {
            "basis_hash": self.basis_hash,
            "fit_hash": self.fit_hash,
            "fit_pivots": list(self.fit_pivots),
            "fit_selected_channel_ids": list(self.fit_selected_channel_ids),
            "fit_solver_channel_indices": list(self.fit_solver_channel_indices),
            "fit_solver_channel_ids": list(self.fit_solver_channel_ids),
            "nonzero_channel_ids": list(self.nonzero_channel_ids),
            "regularization": float(self.regularization),
            "fit_indices": list(self.fit_indices),
            "band_window": list(self.band_window),
            "coefficient_tolerance": float(self.coefficient_tolerance),
            "coefficient_normalization": self.coefficient_normalization,
            "regularization_normalization": self.regularization_normalization,
            "fit_channel_policy": self.fit_channel_policy,
            "fit_solver_policy": self.fit_solver_policy,
            "fit_design_singular_values": list(self.fit_design_singular_values),
            "fit_design_machine_error_bound": float(self.fit_design_machine_error_bound),
            "fit_design_propagated_error_bound": float(self.fit_design_propagated_error_bound),
            "fit_design_rank_tolerance": float(self.fit_design_rank_tolerance),
            "fit_design_certified_rank": int(self.fit_design_certified_rank),
            "fit_selected_condition_number": float(self.fit_selected_condition_number),
            "fit_selected_singular_value_min": float(self.fit_selected_singular_value_min),
            "fit_selected_error_bound": float(self.fit_selected_error_bound),
            "max_response_amplitude": float(self.max_response_amplitude),
            "fit_objective": _json_record(self.fit_objective),
        }


@dataclass(frozen=True)
class CompiledResponseRuntime:
    basis: "CompiledResponseBasis"
    fitted: FittedResponseModel

    def __post_init__(self) -> None:
        if self.fitted.basis_hash != self.basis.basis_hash:
            raise ValueError(
                "fitted response state belongs to a different CompiledResponseBasis: "
                f"{self.fitted.basis_hash} != {self.basis.basis_hash}"
            )
        coefficients = np.asarray(self.fitted.coefficients, dtype=np.float64)
        if coefficients.shape != (len(self.basis.channels),):
            raise ValueError(
                "fitted response coefficient vector has shape "
                f"{coefficients.shape}, expected {(len(self.basis.channels),)}"
            )
        if self.fitted.coefficient_normalization != self.basis.coefficient_normalization:
            raise ValueError(
                "fitted response coefficient normalization does not match its basis: "
                f"{self.fitted.coefficient_normalization!r} != {self.basis.coefficient_normalization!r}"
            )
        if self.fitted.regularization_normalization != self.basis.regularization_normalization:
            raise ValueError(
                "fitted response regularization normalization does not match its basis: "
                f"{self.fitted.regularization_normalization!r} != "
                f"{self.basis.regularization_normalization!r}"
            )
        if self.fitted.fit_channel_policy != FIT_CHANNEL_POLICY_V2:
            raise ValueError(
                "unsupported complete-response fit channel policy: "
                f"{self.fitted.fit_channel_policy!r}"
            )
        invalid_channels = [
            channel.channel_id
            for channel, coefficient in zip(self.basis.channels, coefficients)
            if channel.classification not in FITTABLE_CHANNEL_CLASSIFICATIONS and float(coefficient) != 0.0
        ]
        if invalid_channels:
            raise ValueError(
                "complete_linear_v2 production coefficients may use only confirmed_nonzero channels; "
                f"nonzero ineligible channels: {invalid_channels[:8]}"
            )

    @cached_property
    def polynomial_coefficients(self) -> Mapping[tuple[int, int], sparse.csr_matrix]:
        return self.basis.combine_channel_coefficients(self.fitted.coefficients)

    def hamiltonian(self, kpoint: Sequence[float]) -> np.ndarray:
        point = np.asarray(kpoint, dtype=np.float64)
        if point.shape != (2,):
            raise ValueError(f"kpoint must have shape (2,), got {point.shape}")
        return self.basis.evaluate_polynomial_coefficients(
            point[None, :],
            self.polynomial_coefficients,
        )[0]

    def hamiltonians(self, kpoints: Sequence[Sequence[float]]) -> np.ndarray:
        return self.basis.evaluate_polynomial_coefficients(
            kpoints,
            self.polynomial_coefficients,
        )

    def export_arrays(self) -> dict[str, np.ndarray]:
        frozen_basis = self.basis.freeze()
        fit_metadata = self.fitted.artifact()
        return {
            "response_semantics": np.asarray(COMPLETE_LINEAR_V2),
            "fitted_metadata_json": np.asarray(_canonical_json(fit_metadata)),
            "fitted_coefficients": np.asarray(self.fitted.coefficients, dtype=np.float64),
            **{
                f"basis_{key}": np.asarray(value)
                for key, value in frozen_basis.items()
            },
        }

    @classmethod
    def from_frozen_arrays(cls, arrays: Mapping[str, Any]) -> "CompiledResponseRuntime":
        semantics = str(np.asarray(arrays.get("response_semantics", "")).item())
        if semantics != COMPLETE_LINEAR_V2:
            raise ValueError(f"frozen runtime response_semantics is {semantics!r}, expected {COMPLETE_LINEAR_V2!r}")
        basis_arrays = {
            key[len("basis_") :]: value
            for key, value in arrays.items()
            if str(key).startswith("basis_")
        }
        basis = CompiledResponseBasis.from_frozen(basis_arrays)
        metadata = json.loads(str(np.asarray(arrays["fitted_metadata_json"]).item()))
        fitted = FittedResponseModel(
            basis_hash=str(metadata["basis_hash"]),
            coefficients=np.asarray(arrays["fitted_coefficients"], dtype=np.float64),
            fit_pivots=tuple(int(value) for value in metadata.get("fit_pivots", [])),
            fit_selected_channel_ids=tuple(str(value) for value in metadata.get("fit_selected_channel_ids", [])),
            nonzero_channel_ids=tuple(str(value) for value in metadata.get("nonzero_channel_ids", [])),
            regularization=float(metadata.get("regularization", 0.0)),
            fit_indices=tuple(int(value) for value in metadata.get("fit_indices", [])),
            band_window=tuple(int(value) for value in metadata.get("band_window", [])),
            fit_hash=str(metadata["fit_hash"]),
            coefficient_tolerance=float(metadata.get("coefficient_tolerance", 0.0)),
            fit_solver_channel_indices=tuple(
                int(value) for value in metadata.get("fit_solver_channel_indices", [])
            ),
            fit_solver_channel_ids=tuple(
                str(value) for value in metadata.get("fit_solver_channel_ids", [])
            ),
            coefficient_normalization=str(
                metadata.get("coefficient_normalization", COEFFICIENT_NORMALIZATION_V1)
            ),
            regularization_normalization=str(
                metadata.get("regularization_normalization", REGULARIZATION_NORMALIZATION_V1)
            ),
            fit_channel_policy=str(metadata.get("fit_channel_policy", FIT_CHANNEL_POLICY_V2)),
            fit_solver_policy=str(metadata.get("fit_solver_policy", FIT_SOLVER_POLICY_LEGACY)),
            fit_design_singular_values=tuple(
                float(value) for value in metadata.get("fit_design_singular_values", [])
            ),
            fit_design_machine_error_bound=float(
                metadata.get("fit_design_machine_error_bound", 0.0)
            ),
            fit_design_propagated_error_bound=float(
                metadata.get("fit_design_propagated_error_bound", 0.0)
            ),
            fit_design_rank_tolerance=float(metadata.get("fit_design_rank_tolerance", 0.0)),
            fit_design_certified_rank=int(
                metadata.get("fit_design_certified_rank", len(metadata.get("fit_pivots", [])))
            ),
            fit_selected_condition_number=float(
                metadata.get("fit_selected_condition_number", 0.0)
            ),
            fit_selected_singular_value_min=float(
                metadata.get("fit_selected_singular_value_min", 0.0)
            ),
            fit_selected_error_bound=float(metadata.get("fit_selected_error_bound", 0.0)),
            max_response_amplitude=float(metadata.get("max_response_amplitude", 0.0)),
            fit_objective=dict(metadata.get("fit_objective", _equal_matrix_fit_objective())),
        )
        return cls(basis=basis, fitted=fitted)


@dataclass(frozen=True)
class CompiledResponseBasis:
    coordinate: PolynomialCoordinateBasis
    channels: tuple[ResponseChannel, ...]
    dim: int
    identity_record: Mapping[str, Any]
    candidate_artifact: Mapping[str, Any]
    reduction_proofs: tuple[Mapping[str, Any], ...]
    dropped_channels: tuple[Mapping[str, Any], ...]
    basis_hash: str
    response_semantics: str = COMPLETE_LINEAR_V2
    response_normalization: str = RESPONSE_NORMALIZATION_V1
    coefficient_normalization: str = COEFFICIENT_NORMALIZATION_V1
    regularization_normalization: str = REGULARIZATION_NORMALIZATION_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(
            self,
            "identity_record",
            _deep_immutable_copy(self.identity_record),
        )
        object.__setattr__(
            self,
            "candidate_artifact",
            _deep_immutable_copy(self.candidate_artifact),
        )
        object.__setattr__(
            self,
            "reduction_proofs",
            tuple(_deep_immutable_copy(record) for record in self.reduction_proofs),
        )
        object.__setattr__(
            self,
            "dropped_channels",
            tuple(_deep_immutable_copy(record) for record in self.dropped_channels),
        )
        if self.basis_hash:
            computed = self._compute_hash()
            if computed != self.basis_hash:
                raise ValueError(
                    "compiled response basis hash mismatch: "
                    f"stored={self.basis_hash}, computed={computed}"
                )

    @classmethod
    def from_candidates(
        cls,
        candidates: CandidateResponseSet,
        *,
        identity_payload: Mapping[str, Any],
        reduce: bool,
    ) -> "CompiledResponseBasis":
        all_nonstructural = [
            channel
            for channel in candidates.channels
            if channel.classification != "structural_zero"
        ]
        proofs: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = [
            {
                "channel_id": channel.channel_id,
                "reason": "adjoint_certified_linear_dependency",
                "representative_channel_id": str(
                    channel.metadata["adjoint_representative_channel_id"]
                ),
                "adjoint_sign": int(channel.metadata["adjoint_sign"]),
                "certification_status": str(
                    channel.metadata["adjoint_certification_status"]
                ),
            }
            for channel in all_nonstructural
            if bool(channel.metadata.get("adjoint_certified_dependency", False))
        ]
        nonstructural = [
            channel
            for channel in all_nonstructural
            if not bool(channel.metadata.get("adjoint_certified_dependency", False))
        ]
        if reduce:
            keep, proofs, reducer_dropped = _reduce_confirmed_channels(
                nonstructural,
                coordinate=candidates.coordinate,
                dim=candidates.dim,
                confirm_factor=candidates.null_policy.confirm_factor,
            )
            dropped.extend(reducer_dropped)
            retained = [nonstructural[index] for index in keep]
        else:
            retained = nonstructural
        identity_record = _json_record(identity_payload)
        provisional = cls(
            coordinate=candidates.coordinate,
            channels=tuple(retained),
            dim=int(candidates.dim),
            identity_record=identity_record,
            candidate_artifact=candidates.artifact(),
            reduction_proofs=tuple(proofs),
            dropped_channels=tuple(dropped),
            basis_hash="",
        )
        object.__setattr__(provisional, "basis_hash", provisional._compute_hash())
        return provisional

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(channel.channel_id for channel in self.channels)

    @property
    def response_scales(self) -> np.ndarray:
        return np.asarray([channel.response_scale for channel in self.channels], dtype=np.float64)

    def _channel_hash_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for channel in self.channels:
            coefficient_records = []
            for monomial in self.coordinate.monomials:
                matrix = channel.coefficients.get(monomial)
                if matrix is None or matrix.nnz == 0:
                    continue
                coefficient_records.append(
                    {
                        "monomial": list(monomial),
                        "matrix": _json_record(matrix),
                    }
                )
            records.append(
                {
                    "channel_id": channel.channel_id,
                    "seed_id": channel.seed_id,
                    "component": channel.component,
                    "classification": channel.classification,
                    "response_scale": float(channel.response_scale),
                    "support_component": channel.support_component,
                    "metadata": _json_record(channel.metadata),
                    "error_bound_components": _json_record(channel.error_bound_components),
                    "coefficients": coefficient_records,
                }
            )
        return records

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "response_semantics": self.response_semantics,
            "response_normalization": self.response_normalization,
            "coefficient_normalization": self.coefficient_normalization,
            "regularization_normalization": self.regularization_normalization,
            "compiler_version": COMPILER_VERSION,
            "coordinate": self.coordinate.metadata(),
            "dim": int(self.dim),
            "identity": self.identity_record,
            "channels": self._channel_hash_records(),
            "candidate_artifact": _json_record(self.candidate_artifact),
            "reduction_proofs": _json_record(self.reduction_proofs),
            "dropped_channels": _json_record(self.dropped_channels),
        }

    def _compute_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._hash_payload()).encode("utf-8")).hexdigest()

    def artifact(self) -> dict[str, Any]:
        return {
            "response_semantics": self.response_semantics,
            "response_normalization": self.response_normalization,
            "coefficient_normalization": self.coefficient_normalization,
            "regularization_normalization": self.regularization_normalization,
            "compiler_version": COMPILER_VERSION,
            "basis_hash": self.basis_hash,
            "dim": int(self.dim),
            "coordinate": self.coordinate.metadata(),
            "candidate_channel_count": int(self.candidate_artifact.get("candidate_channel_count", 0)),
            "retained_basis_channel_count": int(len(self.channels)),
            "channel_ids": list(self.channel_ids),
            "reduction_proofs": _json_record(self.reduction_proofs),
            "dropped_channels": _json_record(self.dropped_channels),
        }

    def response_tensor(self, kpoints: Sequence[Sequence[float]]) -> np.ndarray:
        points = np.asarray(kpoints, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"kpoints must have shape (Nk,2), got {points.shape}")
        dimensionless = self.coordinate.to_dimensionless(points)
        w = dimensionless[:, 0] + 1j * dimensionless[:, 1]
        monomial_values = {
            monomial: (w ** monomial[0]) * (np.conjugate(w) ** monomial[1])
            for monomial in self.coordinate.monomials
        }
        out = np.zeros(
            (points.shape[0], len(self.channels), self.dim, self.dim),
            dtype=np.complex128,
        )
        for channel_index, channel in enumerate(self.channels):
            for monomial, matrix in channel.coefficients.items():
                dense = matrix.toarray()
                out[:, channel_index] += monomial_values[monomial][:, None, None] * dense[None, :, :]
        return out

    def hamiltonians(
        self,
        kpoints: Sequence[Sequence[float]],
        coefficients: Sequence[float],
    ) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.channels),):
            raise ValueError(f"coefficient vector has shape {values.shape}, expected {(len(self.channels),)}")
        return self.evaluate_polynomial_coefficients(
            kpoints,
            self.combine_channel_coefficients(values),
        )

    def combine_channel_coefficients(
        self,
        coefficients: Sequence[float],
    ) -> dict[tuple[int, int], sparse.csr_matrix]:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (len(self.channels),):
            raise ValueError(f"coefficient vector has shape {values.shape}, expected {(len(self.channels),)}")
        combined: dict[tuple[int, int], sparse.csr_matrix] = {}
        for coefficient, channel in zip(values, self.channels):
            if float(coefficient) == 0.0:
                continue
            for monomial, matrix in channel.coefficients.items():
                contribution = matrix * float(coefficient)
                combined[monomial] = (
                    contribution
                    if monomial not in combined
                    else combined[monomial] + contribution
                )
        return {
            monomial: _canonical_csr(matrix, shape=(self.dim, self.dim))
            for monomial, matrix in combined.items()
        }

    def evaluate_polynomial_coefficients(
        self,
        kpoints: Sequence[Sequence[float]],
        coefficients: Mapping[tuple[int, int], sparse.spmatrix | np.ndarray],
    ) -> np.ndarray:
        points = np.asarray(kpoints, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"kpoints must have shape (Nk,2), got {points.shape}")
        dimensionless = self.coordinate.to_dimensionless(points)
        w = dimensionless[:, 0] + 1j * dimensionless[:, 1]
        output = np.zeros((points.shape[0], self.dim, self.dim), dtype=np.complex128)
        for (r, s), raw_matrix in coefficients.items():
            matrix = _canonical_csr(raw_matrix, shape=(self.dim, self.dim)).tocoo()
            if matrix.nnz == 0:
                continue
            monomial = (w ** int(r)) * (np.conjugate(w) ** int(s))
            output[:, matrix.row, matrix.col] += monomial[:, None] * matrix.data[None, :]
        return output

    def _response_design_on_union_support(
        self,
        kpoints: Sequence[Sequence[float]],
        channel_indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate selected channels only where at least one is nonzero.

        Rows outside this union support are identically zero for every design
        column.  Removing them drops only a coefficient-independent constant
        from the full Frobenius least-squares loss.
        """

        points = np.asarray(kpoints, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"kpoints must have shape (Nk,2), got {points.shape}")
        selected = np.asarray(channel_indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("channel_indices must be one-dimensional")
        if selected.size == 0:
            return np.zeros((0, 0), dtype=np.complex128), np.zeros(0, dtype=np.int64)
        if np.min(selected) < 0 or np.max(selected) >= len(self.channels):
            raise ValueError("channel index is outside the compiled response basis")

        support_parts: list[np.ndarray] = []
        for channel_index in selected:
            channel = self.channels[int(channel_index)]
            for matrix in channel.coefficients.values():
                coo = matrix.tocoo()
                if coo.nnz:
                    support_parts.append(
                        np.asarray(coo.row, dtype=np.int64) * int(self.dim)
                        + np.asarray(coo.col, dtype=np.int64)
                    )
        if not support_parts:
            return (
                np.zeros((0, selected.size), dtype=np.complex128),
                np.zeros(0, dtype=np.int64),
            )
        support = np.unique(np.concatenate(support_parts))
        support_lookup = np.full(int(self.dim) * int(self.dim), -1, dtype=np.int64)
        support_lookup[support] = np.arange(support.size, dtype=np.int64)

        dimensionless = self.coordinate.to_dimensionless(points)
        w = dimensionless[:, 0] + 1j * dimensionless[:, 1]
        monomial_values = {
            monomial: (w ** monomial[0]) * (np.conjugate(w) ** monomial[1])
            for monomial in self.coordinate.monomials
        }
        design = np.zeros(
            (points.shape[0], support.size, selected.size),
            dtype=np.complex128,
        )
        for output_index, channel_index in enumerate(selected):
            channel = self.channels[int(channel_index)]
            for monomial, matrix in channel.coefficients.items():
                coo = matrix.tocoo()
                if coo.nnz == 0:
                    continue
                flat = (
                    np.asarray(coo.row, dtype=np.int64) * int(self.dim)
                    + np.asarray(coo.col, dtype=np.int64)
                )
                positions = support_lookup[flat]
                if np.any(positions < 0):
                    raise ValueError("compiled response coefficient escaped union support")
                design[:, positions, output_index] += (
                    monomial_values[monomial][:, None]
                    * np.asarray(coo.data, dtype=np.complex128)[None, :]
                )
        return design.reshape(points.shape[0] * support.size, selected.size), support

    def _target_spectral_weighted_complex_rows(
        self,
        complex_design: np.ndarray,
        union_support: np.ndarray,
        target_hamiltonians: np.ndarray,
        weighting: TargetSpectralWeighting,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one-sided target spectral factors without a dense response tensor."""

        target = np.asarray(target_hamiltonians, dtype=np.complex128)
        support = np.asarray(union_support, dtype=np.int64)
        design = np.asarray(complex_design, dtype=np.complex128)
        n_kpoints = int(target.shape[0])
        n_channels = int(design.shape[1])
        if target.shape != (n_kpoints, self.dim, self.dim):
            raise ValueError("target spectral row compiler received an invalid target shape")
        if weighting.spectral_weights.shape != (n_kpoints, self.dim):
            raise ValueError("target spectral weights do not match the fit target shape")
        if design.shape[0] != n_kpoints * support.size:
            raise ValueError("target spectral row compiler received an invalid sparse-union design")

        target_on_support = target.reshape(n_kpoints, -1)[:, support].reshape(-1)
        root_floor = float(np.sqrt(weighting.effective_floor))
        design_parts: list[np.ndarray] = [root_floor * design]
        target_parts: list[np.ndarray] = [root_floor * target_on_support]
        delta = np.asarray(weighting.spectral_weights, dtype=np.float64) - float(
            weighting.effective_floor
        )
        threshold = np.finfo(float).eps * max(
            1.0, float(np.max(weighting.spectral_weights))
        )
        support_rows = support // int(self.dim)
        support_cols = support % int(self.dim)
        active_rows = np.unique(support_rows)
        entries_by_row = {
            int(row): np.flatnonzero(support_rows == int(row))
            for row in active_rows
        }
        design_by_k = design.reshape(n_kpoints, support.size, n_channels)
        if np.any(delta > threshold):
            for k_index in range(n_kpoints):
                active_directions = np.flatnonzero(delta[k_index] > threshold)
                if active_directions.size == 0:
                    continue
                vectors = np.asarray(
                    weighting.eigenvectors[k_index][:, active_directions],
                    dtype=np.complex128,
                )
                if vectors.ndim == 1:
                    vectors = vectors[:, None]
                root_delta = np.sqrt(delta[k_index, active_directions])
                target_times_vectors = target[k_index] @ vectors
                for row in active_rows:
                    entries = entries_by_row[int(row)]
                    columns = support_cols[entries]
                    # (direction, support-entry) @ (support-entry, channel)
                    # gives all B_alpha @ u_n components for this output row.
                    row_design = vectors[columns, :].T @ design_by_k[k_index, entries, :]
                    design_parts.append(root_delta[:, None] * row_design)
                    target_parts.append(root_delta * target_times_vectors[int(row), :])

        projector_weight = float(weighting.spec.two_sided_projector_weight)
        if projector_weight > 0.0:
            root_projector_weight = float(np.sqrt(projector_weight))
            for k_index in range(n_kpoints):
                selected = np.flatnonzero(weighting.selected_mask[k_index])
                vectors = np.asarray(
                    weighting.eigenvectors[k_index][:, selected],
                    dtype=np.complex128,
                )
                left = np.conjugate(vectors[support_rows, :])
                right = vectors[support_cols, :]
                projector_design = np.einsum(
                    "eab,ec->abc",
                    left[:, :, None] * right[:, None, :],
                    design_by_k[k_index],
                    optimize=True,
                ).reshape(selected.size * selected.size, n_channels)
                projector_target = (
                    vectors.conj().T @ target[k_index] @ vectors
                ).reshape(-1)
                design_parts.append(root_projector_weight * projector_design)
                target_parts.append(root_projector_weight * projector_target)
        return np.vstack(design_parts), np.concatenate(target_parts)

    def _normalized_low_energy_complex_rows(
        self,
        complex_design: np.ndarray,
        union_support: np.ndarray,
        target_hamiltonians: np.ndarray,
        weighting: NormalizedLowEnergyWeighting,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compile the exact dimension-normalized public linear objective."""

        target = np.asarray(target_hamiltonians, dtype=np.complex128)
        support = np.asarray(union_support, dtype=np.int64)
        design = np.asarray(complex_design, dtype=np.complex128)
        n_kpoints = int(target.shape[0])
        n_channels = int(design.shape[1])
        if target.shape != (n_kpoints, self.dim, self.dim):
            raise ValueError("normalized low-energy row compiler received an invalid target shape")
        if weighting.selected_mask.shape != (n_kpoints, self.dim):
            raise ValueError("normalized low-energy projector does not match the target shape")
        if design.shape[0] != n_kpoints * support.size:
            raise ValueError("normalized low-energy row compiler received an invalid sparse-union design")

        dim = int(self.dim)
        target_on_support = target.reshape(n_kpoints, -1)[:, support].reshape(-1)
        design_parts: list[np.ndarray] = [design / float(dim)]
        target_parts: list[np.ndarray] = [target_on_support / float(dim)]
        support_rows = support // dim
        support_cols = support % dim
        active_rows = np.unique(support_rows)
        entries_by_row = {
            int(row): np.flatnonzero(support_rows == int(row))
            for row in active_rows
        }
        design_by_k = design.reshape(n_kpoints, support.size, n_channels)
        one_sided_weight = float(weighting.spec.one_sided_weight)
        two_sided_weight = float(weighting.spec.two_sided_weight)
        for k_index in range(n_kpoints):
            selected = np.flatnonzero(weighting.selected_mask[k_index])
            n_bands = int(selected.size)
            if n_bands == 0:
                raise ValueError("normalized low-energy projector is empty")
            vectors = np.asarray(
                weighting.eigenvectors[k_index][:, selected],
                dtype=np.complex128,
            )
            if one_sided_weight > 0.0:
                scale = float(np.sqrt(one_sided_weight / (dim * n_bands)))
                target_times_vectors = target[k_index] @ vectors
                for row in active_rows:
                    entries = entries_by_row[int(row)]
                    columns = support_cols[entries]
                    row_design = vectors[columns, :].T @ design_by_k[k_index, entries, :]
                    design_parts.append(scale * row_design)
                    target_parts.append(scale * target_times_vectors[int(row), :])
            if two_sided_weight > 0.0:
                scale = float(np.sqrt(two_sided_weight) / n_bands)
                left = np.conjugate(vectors[support_rows, :])
                right = vectors[support_cols, :]
                projector_design = np.einsum(
                    "eab,ec->abc",
                    left[:, :, None] * right[:, None, :],
                    design_by_k[k_index],
                    optimize=True,
                ).reshape(n_bands * n_bands, n_channels)
                projector_target = (
                    vectors.conj().T @ target[k_index] @ vectors
                ).reshape(-1)
                design_parts.append(scale * projector_design)
                target_parts.append(scale * projector_target)
        return np.vstack(design_parts), np.concatenate(target_parts)

    def fit(
        self,
        kpoints: Sequence[Sequence[float]],
        target_hamiltonians: np.ndarray,
        *,
        fit_indices: Sequence[int],
        regularization: float,
        band_window: Sequence[int],
        coefficient_tolerance: float = 0.0,
        spectral_weighting: (
            TargetSpectralWeightingSpec
            | NormalizedLowEnergyWeightingSpec
            | Mapping[str, Any]
            | None
        ) = None,
        progress_callback: Callable[..., None] | None = None,
    ) -> FittedResponseModel:
        points = np.asarray(kpoints, dtype=np.float64)
        target = np.asarray(target_hamiltonians, dtype=np.complex128)
        if target.shape != (points.shape[0], self.dim, self.dim):
            raise ValueError(
                f"target_hamiltonians has shape {target.shape}, expected {(points.shape[0], self.dim, self.dim)}"
            )
        eligible = np.asarray(
            [
                index
                for index, channel in enumerate(self.channels)
                if channel.classification in FITTABLE_CHANNEL_CLASSIFICATIONS
            ],
            dtype=np.int64,
        )
        if eligible.size == 0:
            raise ValueError(
                "complete response basis has no confirmed_nonzero channels eligible for production fitting"
            )
        tensor_started = time.perf_counter()
        _response_progress(progress_callback, "fit response sparse-union design start", state="start")
        complex_design, union_support = self._response_design_on_union_support(points, eligible)
        if union_support.size == 0:
            raise ValueError("complete response fit design has empty union support")
        complex_target = target.reshape(points.shape[0], -1)[:, union_support].reshape(-1)
        fit_objective: Mapping[str, Any] = _equal_matrix_fit_objective()
        maximum_weight_by_kpoint = np.ones(points.shape[0], dtype=np.float64)
        equal_weighting_requested = (
            isinstance(spectral_weighting, Mapping)
            and str(spectral_weighting.get("mode", "")).strip().lower()
            in {"equal_matrix", EQUAL_MATRIX_FIT_OBJECTIVE_V1}
        )
        normalized_weighting_requested = isinstance(
            spectral_weighting, NormalizedLowEnergyWeightingSpec
        ) or (
            isinstance(spectral_weighting, Mapping)
            and str(spectral_weighting.get("mode", "")).strip().lower()
            == "normalized_low_energy_linear_v1"
        )
        if normalized_weighting_requested:
            prepared_normalized_weighting = build_normalized_low_energy_weighting(
                target,
                spectral_weighting,
            )
            complex_design, complex_target = self._normalized_low_energy_complex_rows(
                complex_design,
                union_support,
                target,
                prepared_normalized_weighting,
            )
            fit_objective = prepared_normalized_weighting.artifact()
            maximum_weight_by_kpoint = (
                prepared_normalized_weighting.maximum_weight_by_kpoint
            )
        elif spectral_weighting is not None and not equal_weighting_requested:
            prepared_weighting = build_target_spectral_weighting(target, spectral_weighting)
            complex_design, complex_target = self._target_spectral_weighted_complex_rows(
                complex_design,
                union_support,
                target,
                prepared_weighting,
            )
            fit_objective = prepared_weighting.artifact()
            maximum_weight_by_kpoint = prepared_weighting.maximum_weight_by_kpoint
            if float(prepared_weighting.spec.two_sided_projector_weight) > 0.0:
                maximum_weight_by_kpoint = (
                    maximum_weight_by_kpoint
                    + float(prepared_weighting.spec.two_sided_projector_weight)
                )
        _response_progress(
            progress_callback,
            "fit response sparse-union design done "
            f"in {time.perf_counter() - tensor_started:.2f} s | "
            f"shape={complex_design.shape} support_entries={union_support.size} "
            f"eligible={eligible.size}",
            state="done",
        )
        design = np.vstack((complex_design.real, complex_design.imag))
        target_vector = np.concatenate((complex_target.real, complex_target.imag))
        response_scales = self.response_scales
        safe_scales = np.where(
            response_scales > np.finfo(float).tiny,
            response_scales,
            1.0,
        )
        normalized_design = design / safe_scales[eligible][None, :]
        solve_started = time.perf_counter()
        _response_progress(progress_callback, "fit response rank/solve start", state="start")
        ridge = float(regularization)
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("regularization must be finite and non-negative")
        ridge_left_vectors: np.ndarray | None = None
        ridge_right_vectors_h: np.ndarray | None = None
        if ridge > 0.0:
            ridge_left_vectors, singular_values, ridge_right_vectors_h = scipy.linalg.svd(
                normalized_design,
                full_matrices=False,
                check_finite=False,
            )
        else:
            singular_values = scipy.linalg.svdvals(normalized_design, check_finite=False)
        spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
        machine_error_bound = float(
            np.finfo(float).eps * max(normalized_design.shape) * spectral_norm
        )

        # A channel's propagated error is an absolute Frobenius bound in the
        # global polynomial coefficient basis.  Map that bound through the
        # fit-grid polynomial evaluation operator, then combine the independent
        # column bounds in Frobenius norm.  The result is an absolute spectral
        # perturbation bound for the target-dependent real design matrix.
        dimensionless = self.coordinate.to_dimensionless(points)
        w = dimensionless[:, 0] + 1j * dimensionless[:, 1]
        monomial_values = {
            monomial: (w ** monomial[0]) * (np.conjugate(w) ** monomial[1])
            for monomial in self.coordinate.monomials
        }
        # The scalar propagated bound is not accompanied by a certified
        # per-channel monomial error support.  Therefore every column must use
        # the full global polynomial evaluation operator, including monomials
        # whose compiled coefficient happened to cancel numerically.
        global_evaluation = np.column_stack(
            [monomial_values[monomial] for monomial in self.coordinate.monomials]
        )
        weighted_global_evaluation = (
            np.sqrt(maximum_weight_by_kpoint)[:, None] * global_evaluation
        )
        global_evaluation_norm = float(
            scipy.linalg.svdvals(weighted_global_evaluation, check_finite=False)[0]
        )
        propagated_column_bounds: list[float] = []
        for channel_index in eligible:
            channel = self.channels[int(channel_index)]
            relative_coefficient_error = float(
                channel.propagated_error_bound / safe_scales[int(channel_index)]
            )
            propagated_column_bounds.append(global_evaluation_norm * relative_coefficient_error)
        propagated_error_bound = float(np.linalg.norm(propagated_column_bounds))
        rank_tolerance = machine_error_bound + propagated_error_bound
        certified_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        propagated_column_bounds_array = np.asarray(propagated_column_bounds, dtype=np.float64)
        if ridge > 0.0:
            # Ridge makes the complete confirmed channel space well posed even
            # when the unregularized fit grid cannot identify every direction.
            # Pre-truncating with a target-dependent RRQR would defeat that
            # purpose and can discard lower-order channels when a higher-order
            # vocabulary is added.
            selected_local = np.arange(eligible.size, dtype=np.int64)
            selected = eligible.copy()
            selected_design = normalized_design
            selected_singular_values = singular_values
            unregularized_min = (
                float(selected_singular_values[-1])
                if selected_design.shape[0] >= selected_design.shape[1]
                else 0.0
            )
            regularized_singular_value_max = float(
                np.sqrt(float(selected_singular_values[0]) ** 2 + ridge)
            )
            selected_singular_value_min = float(
                np.sqrt(unregularized_min**2 + ridge)
            )
            selected_condition_number = float(
                regularized_singular_value_max / selected_singular_value_min
            )
            selected_error_bound = propagated_error_bound
            fit_solver_policy = FIT_SOLVER_POLICY_RIDGE_V3
            if ridge_left_vectors is None or ridge_right_vectors_h is None:
                raise RuntimeError("ridge SVD factors are unavailable")
            filter_factors = selected_singular_values / (
                selected_singular_values**2 + ridge
            )
            selected_theta = ridge_right_vectors_h.T @ (
                filter_factors * (ridge_left_vectors.T @ target_vector)
            )
            if not np.all(np.isfinite(selected_theta)):
                raise ValueError("complete response ridge solve produced non-finite coefficients")
        else:
            if certified_rank == 0:
                raise ValueError(
                    "complete response fit grid has no response direction certified above "
                    f"the propagated design error bound {rank_tolerance:.6e}"
                )
            rank = int(certified_rank)
            _q, _r, pivots = scipy.linalg.qr(
                normalized_design,
                mode="economic",
                pivoting=True,
                check_finite=False,
            )
            while rank > 0:
                selected_local = np.asarray(pivots[:rank], dtype=np.int64)
                selected_design = normalized_design[:, selected_local]
                selected_singular_values = scipy.linalg.svdvals(
                    selected_design,
                    check_finite=False,
                )
                selected_machine_error_bound = float(
                    np.finfo(float).eps
                    * max(selected_design.shape)
                    * float(selected_singular_values[0])
                )
                selected_propagated_error_bound = float(
                    np.linalg.norm(propagated_column_bounds_array[selected_local])
                )
                selected_error_bound = (
                    selected_machine_error_bound + selected_propagated_error_bound
                )
                if float(selected_singular_values[-1]) > selected_error_bound:
                    break
                rank -= 1
            if rank == 0:
                raise ValueError(
                    "complete response RRQR pivot subset has no direction certified above "
                    "its propagated design error bound"
                )
            selected = eligible[selected_local]
            selected_condition_number = float(
                selected_singular_values[0] / selected_singular_values[-1]
            )
            selected_singular_value_min = float(selected_singular_values[-1])
            fit_solver_policy = FIT_SOLVER_POLICY_V2
            solve_design = selected_design
            solve_target = target_vector
            selected_theta, _residuals, solve_rank, _solve_singular_values = scipy.linalg.lstsq(
                solve_design,
                solve_target,
                cond=None,
                lapack_driver="gelsy",
                check_finite=False,
            )
            if int(solve_rank) != int(selected.size) or not np.all(np.isfinite(selected_theta)):
                raise ValueError(
                    "complete response fit solve is not full rank in its selected solver space"
                )
        _response_progress(
            progress_callback,
            "fit response rank/solve done "
            f"in {time.perf_counter() - solve_started:.2f} s | "
            f"eligible={eligible.size} selected={selected.size} "
            f"certified_rank={certified_rank} ridge={ridge:.3g}",
            state="done",
        )
        coefficients = np.zeros(len(self.channels), dtype=np.float64)
        coefficients[selected] = selected_theta / safe_scales[selected]
        amplitude = np.abs(coefficients) * self.response_scales
        coefficients[amplitude <= float(coefficient_tolerance)] = 0.0
        amplitude = np.abs(coefficients) * self.response_scales
        nonzero = tuple(
            self.channels[index].channel_id
            for index in np.flatnonzero(coefficients != 0.0)
        )
        if fit_solver_policy == FIT_SOLVER_POLICY_RIDGE_V3:
            fit_pivots: tuple[int, ...] = ()
            fit_selected_channel_ids: tuple[str, ...] = ()
        else:
            fit_pivots = tuple(int(value) for value in selected)
            fit_selected_channel_ids = tuple(
                self.channels[int(index)].channel_id for index in selected
            )
        fit_solver_channel_indices = tuple(int(value) for value in selected)
        fit_solver_channel_ids = tuple(
            self.channels[int(index)].channel_id for index in selected
        )
        fit_payload = {
            "basis_hash": self.basis_hash,
            "target": _json_record(target),
            "fit_kpoints": _json_record(points),
            "fit_indices": [int(value) for value in fit_indices],
            "regularization": ridge,
            "band_window": [int(value) for value in band_window],
            "fit_pivots": list(fit_pivots),
            "fit_selected_channel_ids": list(fit_selected_channel_ids),
            "fit_solver_channel_indices": list(fit_solver_channel_indices),
            "fit_solver_channel_ids": list(fit_solver_channel_ids),
            "eligible_channel_ids": [self.channels[int(index)].channel_id for index in eligible],
            "fit_channel_policy": FIT_CHANNEL_POLICY_V2,
            "fit_solver_policy": fit_solver_policy,
            "fit_design_singular_values": _json_record(singular_values),
            "fit_design_machine_error_bound": machine_error_bound,
            "fit_design_propagated_error_bound": propagated_error_bound,
            "fit_design_rank_tolerance": rank_tolerance,
            "fit_design_certified_rank": certified_rank,
            "fit_selected_condition_number": selected_condition_number,
            "fit_selected_singular_value_min": selected_singular_value_min,
            "fit_selected_error_bound": selected_error_bound,
            "max_response_amplitude": float(np.max(amplitude)) if amplitude.size else 0.0,
            "coefficients": _json_record(coefficients),
            "coefficient_tolerance": float(coefficient_tolerance),
            "fit_objective": _json_record(fit_objective),
        }
        fit_hash = hashlib.sha256(_canonical_json(fit_payload).encode("utf-8")).hexdigest()
        return FittedResponseModel(
            basis_hash=self.basis_hash,
            coefficients=coefficients,
            fit_pivots=fit_pivots,
            fit_selected_channel_ids=fit_selected_channel_ids,
            nonzero_channel_ids=nonzero,
            regularization=ridge,
            fit_indices=tuple(int(value) for value in fit_indices),
            band_window=tuple(int(value) for value in band_window),
            fit_hash=fit_hash,
            coefficient_tolerance=float(coefficient_tolerance),
            fit_solver_channel_indices=fit_solver_channel_indices,
            fit_solver_channel_ids=fit_solver_channel_ids,
            coefficient_normalization=self.coefficient_normalization,
            fit_channel_policy=FIT_CHANNEL_POLICY_V2,
            fit_solver_policy=fit_solver_policy,
            fit_design_singular_values=tuple(float(value) for value in singular_values),
            fit_design_machine_error_bound=machine_error_bound,
            fit_design_propagated_error_bound=propagated_error_bound,
            fit_design_rank_tolerance=rank_tolerance,
            fit_design_certified_rank=certified_rank,
            fit_selected_condition_number=selected_condition_number,
            fit_selected_singular_value_min=selected_singular_value_min,
            fit_selected_error_bound=selected_error_bound,
            max_response_amplitude=float(np.max(amplitude)) if amplitude.size else 0.0,
            fit_objective=fit_objective,
        )

    def freeze(self) -> dict[str, np.ndarray]:
        channel_records = []
        entry_channel: list[int] = []
        entry_monomial: list[int] = []
        entry_row: list[int] = []
        entry_col: list[int] = []
        entry_value: list[complex] = []
        monomial_index = {key: index for index, key in enumerate(self.coordinate.monomials)}
        for channel_index, channel in enumerate(self.channels):
            channel_records.append(
                {
                    "channel_id": channel.channel_id,
                    "seed_id": channel.seed_id,
                    "component": channel.component,
                    "unnormalized_norm": float(channel.unnormalized_norm),
                    "propagated_error_bound": float(channel.propagated_error_bound),
                    "classification": channel.classification,
                    "response_scale": float(channel.response_scale),
                    "support_component": channel.support_component,
                    "metadata": _json_record(channel.metadata),
                    "support_leakage_before_cleanup": float(channel.support_leakage_before_cleanup),
                    "error_bound_components": _json_record(channel.error_bound_components),
                }
            )
            for monomial, matrix in channel.coefficients.items():
                coo = matrix.tocoo()
                entry_channel.extend([channel_index] * int(coo.nnz))
                entry_monomial.extend([monomial_index[monomial]] * int(coo.nnz))
                entry_row.extend(int(value) for value in coo.row)
                entry_col.extend(int(value) for value in coo.col)
                entry_value.extend(complex(value) for value in coo.data)
        metadata = {
            "response_semantics": self.response_semantics,
            "response_normalization": self.response_normalization,
            "coefficient_normalization": self.coefficient_normalization,
            "regularization_normalization": self.regularization_normalization,
            "compiler_version": COMPILER_VERSION,
            "basis_hash": self.basis_hash,
            "dim": int(self.dim),
            "coordinate": self.coordinate.metadata(),
            "identity_record": self.identity_record,
            "candidate_artifact": _json_record(self.candidate_artifact),
            "reduction_proofs": _json_record(self.reduction_proofs),
            "dropped_channels": _json_record(self.dropped_channels),
            "channels": channel_records,
        }
        return {
            "metadata_json": np.asarray(_canonical_json(metadata)),
            "entry_channel": np.asarray(entry_channel, dtype=np.int64),
            "entry_monomial": np.asarray(entry_monomial, dtype=np.int64),
            "entry_row": np.asarray(entry_row, dtype=np.int64),
            "entry_col": np.asarray(entry_col, dtype=np.int64),
            "entry_value": np.asarray(entry_value, dtype=np.complex128),
        }

    @classmethod
    def from_frozen(cls, arrays: Mapping[str, Any]) -> "CompiledResponseBasis":
        required = {
            "metadata_json",
            "entry_channel",
            "entry_monomial",
            "entry_row",
            "entry_col",
            "entry_value",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"frozen compiled response basis is missing arrays: {missing}")
        metadata = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
        if metadata.get("response_semantics") != COMPLETE_LINEAR_V2:
            raise ValueError("frozen response basis is not complete_linear_v2")
        coordinate = PolynomialCoordinateBasis.from_metadata(metadata["coordinate"])
        dim = int(metadata["dim"])
        entry_channel = np.asarray(arrays["entry_channel"], dtype=np.int64)
        entry_monomial = np.asarray(arrays["entry_monomial"], dtype=np.int64)
        entry_row = np.asarray(arrays["entry_row"], dtype=np.int64)
        entry_col = np.asarray(arrays["entry_col"], dtype=np.int64)
        entry_value = np.asarray(arrays["entry_value"], dtype=np.complex128)
        lengths = {array.size for array in (entry_channel, entry_monomial, entry_row, entry_col, entry_value)}
        if len(lengths) != 1:
            raise ValueError("frozen response basis entry arrays have inconsistent lengths")
        channel_records = list(metadata["channels"])
        n_channels = len(channel_records)
        n_monomials = len(coordinate.monomials)
        if entry_channel.size:
            if np.min(entry_channel) < 0 or np.max(entry_channel) >= n_channels:
                raise ValueError("frozen response basis contains an invalid channel index")
            if np.min(entry_monomial) < 0 or np.max(entry_monomial) >= n_monomials:
                raise ValueError("frozen response basis contains an invalid monomial index")
            if (
                np.min(entry_row) < 0
                or np.max(entry_row) >= dim
                or np.min(entry_col) < 0
                or np.max(entry_col) >= dim
            ):
                raise ValueError("frozen response basis contains an invalid matrix index")
        group_key = entry_channel * max(n_monomials, 1) + entry_monomial
        if group_key.size > 1 and np.any(np.diff(group_key) < 0):
            order = np.argsort(group_key, kind="stable")
            group_key = group_key[order]
            entry_channel = entry_channel[order]
            entry_monomial = entry_monomial[order]
            entry_row = entry_row[order]
            entry_col = entry_col[order]
            entry_value = entry_value[order]
        grouped_coefficients: list[dict[tuple[int, int], sparse.spmatrix]] = [
            {} for _ in range(n_channels)
        ]
        if group_key.size:
            boundaries = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.flatnonzero(np.diff(group_key)) + 1,
                    np.asarray([group_key.size], dtype=np.int64),
                )
            )
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                first = int(start)
                last = int(stop)
                channel_index = int(entry_channel[first])
                monomial_index = int(entry_monomial[first])
                grouped_coefficients[channel_index][coordinate.monomials[monomial_index]] = (
                    sparse.coo_matrix(
                        (
                            entry_value[first:last],
                            (entry_row[first:last], entry_col[first:last]),
                        ),
                        shape=(dim, dim),
                    )
                )
        channels: list[ResponseChannel] = []
        for channel_index, record in enumerate(channel_records):
            coefficients = grouped_coefficients[channel_index]
            channels.append(
                ResponseChannel(
                    channel_id=str(record["channel_id"]),
                    seed_id=str(record["seed_id"]),
                    component=str(record["component"]),
                    coefficients=coefficients or {(0, 0): sparse.csr_matrix((dim, dim), dtype=np.complex128)},
                    unnormalized_norm=float(record["unnormalized_norm"]),
                    propagated_error_bound=float(record["propagated_error_bound"]),
                    classification=str(record["classification"]),
                    response_scale=float(record["response_scale"]),
                    support_component=str(record["support_component"]),
                    metadata=dict(record.get("metadata", {})),
                    support_leakage_before_cleanup=float(record.get("support_leakage_before_cleanup", 0.0)),
                    error_bound_components={
                        str(key): float(value)
                        for key, value in record.get("error_bound_components", {}).items()
                    },
                )
            )
        basis = cls(
            coordinate=coordinate,
            channels=tuple(channels),
            dim=dim,
            identity_record=metadata["identity_record"],
            candidate_artifact=metadata["candidate_artifact"],
            reduction_proofs=tuple(metadata.get("reduction_proofs", [])),
            dropped_channels=tuple(metadata.get("dropped_channels", [])),
            basis_hash=str(metadata["basis_hash"]),
            response_semantics=str(metadata["response_semantics"]),
            response_normalization=str(metadata["response_normalization"]),
            coefficient_normalization=str(
                metadata.get("coefficient_normalization", COEFFICIENT_NORMALIZATION_V1)
            ),
            regularization_normalization=str(metadata["regularization_normalization"]),
        )
        return basis


@dataclass(frozen=True)
class GeneratorFixedResponseGroup:
    """One target-independent group compilation before cross-group assembly."""

    candidates: CandidateResponseSet
    retained_channels: tuple[ResponseChannel, ...]
    reduction_proofs: tuple[Mapping[str, Any], ...]
    dropped_channels: tuple[Mapping[str, Any], ...]


def _fixed_group_from_projected_candidates(
    candidates: CandidateResponseSet,
    *,
    reduce: bool,
    solver: str,
) -> GeneratorFixedResponseGroup:
    """Certify an already projected candidate set without a second group solve."""

    channels = [
        channel
        for channel in candidates.channels
        if channel.classification != "structural_zero"
    ]
    proofs: list[Mapping[str, Any]] = []
    dropped: list[Mapping[str, Any]] = [
        {
            "channel_id": channel.channel_id,
            "reason": "structural_zero",
            "solver": solver,
        }
        for channel in candidates.channels
        if channel.classification == "structural_zero"
    ]
    if reduce and channels:
        keep, reduction_proofs, reduction_dropped = _reduce_confirmed_channels(
            channels,
            coordinate=candidates.coordinate,
            dim=candidates.dim,
            confirm_factor=candidates.null_policy.confirm_factor,
        )
        channels = [channels[index] for index in keep]
        proofs.extend(reduction_proofs)
        dropped.extend(reduction_dropped)
    proofs.append(
        {
            "solver": solver,
            "block_rule": "declared_orbit_representative_reynolds_hermitian_projection",
            "selected_channel_ids": [channel.channel_id for channel in channels],
            "certification_space": (
                "global_two_dimensional_polynomial_coefficient_space_real_imag_stack"
            ),
            "sample_grid_used": False,
        }
    )
    return GeneratorFixedResponseGroup(
        candidates=candidates,
        retained_channels=tuple(channels),
        reduction_proofs=tuple(proofs),
        dropped_channels=tuple(dropped),
    )


def _coefficient_vector_to_response_map(
    vector: np.ndarray | sparse.spmatrix,
    *,
    coordinate: PolynomialCoordinateBasis,
    dim: int,
) -> dict[tuple[int, int], sparse.csr_matrix]:
    if sparse.issparse(vector):
        column = sparse.coo_matrix(vector, dtype=np.float64)
        block = len(coordinate.monomials) * int(dim) * int(dim)
        if column.shape != (2 * block, 1):
            raise ValueError(
                f"global real coefficient vector has shape {column.shape}, expected {(2 * block, 1)}"
            )
        entries: dict[tuple[int, int], dict[tuple[int, int], complex]] = {}
        for raw_row, raw_value in zip(column.row, column.data):
            coefficient_row = int(raw_row)
            is_imaginary = coefficient_row >= block
            local_row = coefficient_row - block if is_imaginary else coefficient_row
            monomial_index, matrix_offset = divmod(local_row, int(dim) * int(dim))
            matrix_row, matrix_col = divmod(matrix_offset, int(dim))
            monomial = coordinate.monomials[monomial_index]
            value = (1.0j if is_imaginary else 1.0) * float(raw_value)
            by_entry = entries.setdefault(monomial, {})
            matrix_key = (matrix_row, matrix_col)
            by_entry[matrix_key] = by_entry.get(matrix_key, 0.0j) + value
        coefficients: dict[tuple[int, int], sparse.csr_matrix] = {}
        for monomial, by_entry in entries.items():
            nonzero = [
                (row, col, value)
                for (row, col), value in by_entry.items()
                if value != 0.0j
            ]
            if not nonzero:
                continue
            rows, cols, values = zip(*nonzero)
            coefficients[monomial] = _canonical_csr(
                sparse.coo_matrix(
                    (values, (rows, cols)),
                    shape=(dim, dim),
                    dtype=np.complex128,
                )
            )
        return coefficients or {
            (0, 0): sparse.csr_matrix((dim, dim), dtype=np.complex128)
        }
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    block = len(coordinate.monomials) * int(dim) * int(dim)
    if values.shape != (2 * block,):
        raise ValueError(
            f"global real coefficient vector has shape {values.shape}, expected {(2 * block,)}"
        )
    real = values[:block].reshape(len(coordinate.monomials), dim, dim)
    imag = values[block:].reshape(len(coordinate.monomials), dim, dim)
    coefficients: dict[tuple[int, int], sparse.csr_matrix] = {}
    for monomial_index, monomial in enumerate(coordinate.monomials):
        matrix = real[monomial_index] + 1j * imag[monomial_index]
        if np.any(matrix != 0.0):
            coefficients[monomial] = _canonical_csr(matrix, shape=(dim, dim))
    return coefficients or {
        (0, 0): sparse.csr_matrix((dim, dim), dtype=np.complex128)
    }


def _actual_generator_elements(group: FiniteGroup) -> tuple[tuple[str, FiniteGroupElement], ...]:
    records: dict[str, FiniteGroupElement] = {}
    for element in group.elements:
        words = (element.canonical_word, *element.alternate_words)
        generator_names = sorted(
            {str(word[0]) for word in words if len(word) == 1}
        )
        for name in generator_names:
            previous = records.get(name)
            if previous is not None and previous.discrete_key != element.discrete_key:
                raise ValueError(
                    f"finite-group generator {name!r} has multiple discrete actions"
                )
            records[name] = element
    if not records:
        if len(group.elements) != 1 or group.elements[0].canonical_word:
            raise ValueError(
                "generator fixed-space compiler cannot recover the actual generators "
                "from canonical/alternate finite-group words"
            )
        records["identity"] = group.elements[0]
    return tuple((name, records[name]) for name in sorted(records))


def _exact_sparse_vector_key(vector: sparse.csc_matrix) -> bytes:
    canonical = vector.tocsc(copy=True)
    canonical.sort_indices()
    return b"".join(
        (
            np.asarray(canonical.shape, dtype="<i8").tobytes(),
            np.asarray(canonical.indptr, dtype="<i8").tobytes(),
            np.asarray(canonical.indices, dtype="<i8").tobytes(),
            np.asarray(canonical.data, dtype="<f8").tobytes(),
        )
    )


def _matvec_gamma(operation_count: int) -> float:
    count = max(0, int(operation_count))
    product = float(count) * np.finfo(np.float64).eps
    if product >= 1.0:
        raise ValueError("floating-point matvec error bound is undefined for this operation count")
    return float(product / (1.0 - product))


def _fixed_projection_roundoff_bound(
    raw_vector: sparse.csc_matrix,
    fixed_vectors: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    """Absolute two-matvec error bound for ``F (F.T v)``.

    The first operation count is the actual sparse support of ``v`` and the
    second is the fixed rank.  Zero ambient coefficient rows therefore cannot
    inflate a channel's classification bound.
    """

    raw = raw_vector.tocsc()
    fixed = np.asarray(fixed_vectors, dtype=np.float64)
    values = np.asarray(coordinates, dtype=np.float64).reshape(-1)
    if raw.shape[1] != 1 or raw.shape[0] != fixed.shape[0] or values.shape != (fixed.shape[1],):
        raise ValueError("fixed projection roundoff inputs have inconsistent shapes")
    absolute_raw = sparse.csc_matrix(
        (np.abs(raw.data), raw.indices.copy(), raw.indptr.copy()),
        shape=raw.shape,
    )
    coordinate_magnitude = np.asarray(
        absolute_raw.T @ np.abs(fixed),
        dtype=np.float64,
    ).reshape(-1)
    coordinate_bound = float(
        _matvec_gamma(raw.nnz) * np.linalg.norm(coordinate_magnitude)
    )
    reconstruction_magnitude = np.abs(fixed) @ np.abs(values)
    reconstruction_bound = float(
        _matvec_gamma(fixed.shape[1])
        * np.linalg.norm(reconstruction_magnitude)
    )
    return float(coordinate_bound + reconstruction_bound)


def _fixed_coordinate_projection_roundoff_bound(
    raw_vector: sparse.csc_matrix,
    vocabulary: sparse.csc_matrix,
    fixed_vocabulary_coordinates: np.ndarray,
    fixed_coordinates: np.ndarray,
    projected_vocabulary_coordinates: np.ndarray,
    *,
    vocabulary_operator_norm: float,
) -> float:
    """Bound roundoff in ``V C C.T V.T raw`` without ambient materialization."""

    raw = raw_vector.tocsc()
    fixed = np.asarray(fixed_vocabulary_coordinates, dtype=np.float64)
    coordinates = np.asarray(fixed_coordinates, dtype=np.float64).reshape(-1)
    projected = np.asarray(projected_vocabulary_coordinates, dtype=np.float64).reshape(-1)
    raw_to_vocabulary = np.asarray(
        (raw.T @ vocabulary).toarray(),
        dtype=np.float64,
    ).reshape(-1)
    fixed_norm = float(np.linalg.norm(fixed, ord=2)) if fixed.size else 0.0
    first = float(
        _matvec_gamma(max(1, raw.nnz, vocabulary.shape[1]))
        * float(vocabulary_operator_norm)
        * float(np.linalg.norm(raw.data))
    )
    second = float(
        _matvec_gamma(max(1, fixed.shape[0]))
        * fixed_norm
        * float(np.linalg.norm(raw_to_vocabulary))
    )
    third = float(
        _matvec_gamma(max(1, fixed.shape[1]))
        * fixed_norm
        * float(np.linalg.norm(coordinates))
    )
    fourth = float(
        _matvec_gamma(max(1, vocabulary.shape[1]))
        * float(vocabulary_operator_norm)
        * float(np.linalg.norm(projected))
    )
    return float(first + second + third + fourth)


def _certified_logical_projection_roundoff_bound(
    *,
    raw_norm: float,
    coordinate_norm: float,
    vocabulary_channel_count: int,
    fixed_rank: int,
    vocabulary_operator_norm: float,
    fixed_vocabulary_operator_norm_bound: float,
) -> float:
    """Bound direct use of fixed compiler logical projection coordinates.

    ``FixedVocabularyCompilation.logical_projection_coordinates`` is computed
    from the same certified Cholesky whitening used to construct the fixed
    basis.  A logical Hermitian channel is already proved to be a signed copy
    of one physical vocabulary column, so recomputing ``raw.T @ vocabulary``
    is mathematically redundant.  This bound covers the stored coordinate and
    sparse reconstruction matvecs in coefficient-space norm.
    """

    reconstruction_scale = float(
        vocabulary_operator_norm
        * fixed_vocabulary_operator_norm_bound
        * max(float(coordinate_norm), np.finfo(np.float64).tiny)
    )
    scale = max(
        float(raw_norm),
        reconstruction_scale,
        np.finfo(np.float64).tiny,
    )
    return float(
        _matvec_gamma(
            max(1, int(vocabulary_channel_count), int(fixed_rank))
        )
        * scale
    )


def compile_generator_fixed_response_group(
    seeds: Sequence[RawPolynomialSeed],
    *,
    joint_keys: Mapping[str, Any] | None,
    coordinate: PolynomialCoordinateBasis,
    group: FiniteGroup,
    reduce: bool,
    support_masks: Mapping[str, np.ndarray] | None = None,
    materialize_ambient_vectors: bool = True,
    factorized_actions: Mapping[str, Any] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> GeneratorFixedResponseGroup:
    """Compile responses through actual-generator fixed coefficient space.

    The complete authored logical channel list is retained in ``candidates``.
    Only exact raw duplicates are removed before whitening.  The returned basis
    channels keep original seed ids and ``term_index`` metadata, so fit/runtime
    coefficient synchronization does not acquire a second representation.
    """

    from .response_basis_fixed_compiler import compile_generator_fixed_vocabulary

    if not seeds:
        raise ValueError("generator fixed response compilation requires raw seeds")
    dim = int(seeds[0].dim)
    raw_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis group raw adjoint vocabulary start",
        state="start",
    )
    masks = (
        {str(key): np.asarray(value, dtype=bool) for key, value in support_masks.items()}
        if support_masks is not None
        else None
    )
    support_artifact = (
        certify_support_closure(
            seeds,
            group=group,
            coordinate=coordinate,
            support_masks=masks,
        )
        if masks is not None
        else {}
    )
    identity_group = identity_finite_group(
        dim,
        q_size=len(group.elements[0].q_permutation),
        sector_size=len(group.elements[0].sector_permutation),
    )
    if joint_keys is None:
        raw_logical = compile_candidate_responses(
            seeds,
            coordinate=coordinate,
            group=identity_group,
            support_masks=None,
        )
    else:
        raw_logical = compile_candidate_responses_adjoint_canonicalized(
            seeds,
            joint_keys=joint_keys,
            coordinate=coordinate,
            group=identity_group,
            support_masks=None,
        )
    physical_raw = [
        channel
        for channel in raw_logical.channels
        if channel.classification != "structural_zero"
        and not bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    if not physical_raw:
        raise ValueError("generator fixed response compilation has no Hermitian raw vocabulary")

    physical_vectors_by_id = {
        channel.channel_id: _channel_sparse_vector(channel, coordinate, dim)
        for channel in physical_raw
    }
    unique_raw: list[ResponseChannel] = []
    unique_vectors: list[sparse.csc_matrix] = []
    unique_by_key: dict[bytes, int] = {}
    exact_duplicate_of: dict[str, tuple[str, int]] = {}
    for channel in physical_raw:
        vector = physical_vectors_by_id[channel.channel_id]
        key = _exact_sparse_vector_key(vector)
        existing = unique_by_key.get(key)
        negative_existing = (
            None
            if existing is not None
            else unique_by_key.get(_exact_sparse_vector_key((-vector).tocsc()))
        )
        if existing is None and negative_existing is None:
            unique_by_key[key] = len(unique_raw)
            unique_raw.append(channel)
            unique_vectors.append(vector)
        elif existing is not None:
            reference = unique_vectors[existing]
            if reference.shape != vector.shape or (reference != vector).nnz:
                raise RuntimeError("exact vocabulary hash collision")
            exact_duplicate_of[channel.channel_id] = (
                unique_raw[existing].channel_id,
                1,
            )
        else:
            assert negative_existing is not None
            reference = unique_vectors[negative_existing]
            if reference.shape != vector.shape or ((-reference).tocsc() != vector).nnz:
                raise RuntimeError("exact signed vocabulary hash collision")
            exact_duplicate_of[channel.channel_id] = (
                unique_raw[negative_existing].channel_id,
                -1,
            )
    vocabulary = sparse.hstack(unique_vectors, format="csc")
    unique_index_by_id = {
        channel.channel_id: index for index, channel in enumerate(unique_raw)
    }

    def logical_physical_resolution(
        channel: ResponseChannel,
    ) -> tuple[int, int] | None:
        if channel.classification == "structural_zero":
            return None
        representative_id = str(
            channel.metadata.get(
                "adjoint_representative_channel_id",
                channel.channel_id,
            )
        )
        representative_id, duplicate_sign = exact_duplicate_of.get(
            representative_id,
            (representative_id, 1),
        )
        representative_index = unique_index_by_id.get(representative_id)
        if representative_index is None:
            raise ValueError(
                f"logical response {channel.channel_id!r} resolves to missing "
                f"physical vocabulary channel {representative_id!r}"
            )
        sign = int(channel.metadata.get("adjoint_sign", 1)) * int(duplicate_sign)
        if sign not in {-1, 1}:
            raise ValueError(
                f"logical response {channel.channel_id!r} has invalid adjoint sign {sign}"
            )
        return representative_index, sign

    zero_raw_vector = sparse.csc_matrix(
        (vocabulary.shape[0], 1),
        dtype=np.float64,
    )

    def logical_raw_vector(channel: ResponseChannel) -> sparse.csc_matrix:
        resolution = logical_physical_resolution(channel)
        if resolution is None:
            return zero_raw_vector
        representative_index, sign = resolution
        vector = unique_vectors[representative_index]
        return vector if sign == 1 else (-vector).tocsc()
    _response_progress(
        progress_callback,
        "response basis group raw adjoint vocabulary done "
        f"in {time.perf_counter() - raw_started:.2f} s | "
        f"ordered={len(seeds)} orbits="
        f"{raw_logical.adjoint_artifact.get('adjoint_orbit_descriptor_count', 0)} "
        f"ambient={len(unique_raw)}",
        state="done",
    )

    generator_images: dict[str, sparse.csc_matrix] = {}
    antiunitary_parities: dict[str, bool] = {}
    generator_error_bounds: dict[str, float] = {}
    generator_column_error_bounds: dict[str, tuple[float, ...]] = {}
    raw_error_bounds: dict[str, float] = {}
    raw_error_components: dict[str, dict[str, float]] = {}
    seeds_by_id = {seed.seed_id: seed for seed in seeds}
    physical_channels_by_seed: dict[str, list[ResponseChannel]] = {}
    for channel in unique_raw:
        if channel.seed_id not in seeds_by_id:
            raise ValueError(
                f"adjoint representative channel {channel.channel_id!r} has no raw seed"
            )
        physical_channels_by_seed.setdefault(channel.seed_id, []).append(channel)
    for channel in raw_logical.channels:
        resolution = logical_physical_resolution(channel)
        raw_norm = (
            0.0
            if resolution is None
            else float(np.linalg.norm(unique_vectors[resolution[0]].data))
        )
        components = _propagated_error_components(
            raw_norm,
            dim=dim,
            degree=coordinate.max_degree,
            group=group,
        )
        components["support_cleanup"] = float(
            support_artifact.get(channel.support_component, {}).get(
                "leakage_before_cleanup", 0.0
            )
        )
        components["coordinate_conversion"] = float(
            channel.metadata.get("coordinate_conversion_error_bound", 0.0)
        )
        raw_error_components[channel.channel_id] = components
        raw_error_bounds[channel.channel_id] = float(sum(components.values()))
    generator_elements = _actual_generator_elements(group)
    generator_image_raw_transform_count = 0
    factorized_action_nnz: dict[str, int] = {}
    factorized_action_hashes: dict[str, str] = {}
    factorized_action_match_residuals: dict[str, float] = {}
    factorized_fallback_reasons: dict[str, str] = {}
    factorized_generator_names: set[str] = set()
    builtin_exact_generator_names: set[str] = set()
    factorized_channel_actions: dict[str, sparse.csc_matrix] = {}
    factorized_actions_required = factorized_actions is not None
    available_factorized_actions = {
        str(key): value for key, value in (factorized_actions or {}).items()
    }
    from .response_basis_factorized import FactorizedTermActionError
    from ..symmetry.factorized_action import FactorizedActionCertificationError

    generator_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis group generator images start",
        state="start",
    )
    for name, element in generator_elements:
        factorized_action = available_factorized_actions.get(name)
        if (
            factorized_action is None
            and name == "identity"
            and len(group.elements) == 1
            and _is_strict_identity_group_element(element, dim=dim)
        ):
            direct_action = sparse.eye(
                vocabulary.shape[1],
                format="csc",
                dtype=np.float64,
            )
            generator_images[name] = vocabulary.copy()
            factorized_channel_actions[name] = direct_action
            builtin_exact_generator_names.add(name)
            factorized_action_nnz[name] = int(direct_action.nnz)
            factorized_action_hashes[name] = "builtin_exact_identity_v1"
            factorized_action_match_residuals[name] = 0.0
            antiunitary_parities[name] = False
            per_column_bounds = tuple(
                raw_error_bounds[channel.channel_id] for channel in unique_raw
            )
            generator_column_error_bounds[name] = per_column_bounds
            generator_error_bounds[name] = float(
                np.linalg.norm(np.asarray(per_column_bounds, dtype=np.float64))
            )
            continue
        if factorized_action is not None:
            try:
                from .response_basis_factorized import (
                    compile_factorized_real_channel_action,
                )
                from ..symmetry.factorized_action import (
                    materialize_factorized_matrix,
                )

                if str(getattr(factorized_action, "name", "")) != name:
                    raise FactorizedTermActionError(
                        f"artifact name {getattr(factorized_action, 'name', None)!r} "
                        f"does not match generator {name!r}"
                    )
                if bool(factorized_action.antiunitary) != bool(element.antiunitary):
                    raise FactorizedTermActionError("antiunitary parity mismatch")
                if tuple(int(value) for value in factorized_action.q_permutation) != tuple(
                    int(value) for value in element.q_permutation
                ):
                    raise FactorizedTermActionError("Q permutation mismatch")
                if tuple(int(value) for value in factorized_action.sector_permutation) != tuple(
                    int(value) for value in element.sector_permutation
                ):
                    raise FactorizedTermActionError("sector permutation mismatch")
                expected_forward = np.linalg.inv(
                    np.asarray(element.k_pullback, dtype=np.float64)
                )
                actual_forward = np.asarray(
                    factorized_action.k_forward,
                    dtype=np.float64,
                )
                map_tolerance = float(
                    128.0
                    * np.finfo(np.float64).eps
                    * max(1.0, np.linalg.norm(expected_forward, ord=2))
                )
                if not np.allclose(
                    actual_forward,
                    expected_forward,
                    rtol=0.0,
                    atol=map_tolerance,
                ):
                    raise FactorizedTermActionError("Cartesian k action mismatch")
                reconstructed = materialize_factorized_matrix(factorized_action)
                reference = np.asarray(element.internal_u, dtype=np.complex128)
                if reconstructed.shape != reference.shape:
                    raise FactorizedTermActionError(
                        f"full action shape mismatch: {reconstructed.shape} != {reference.shape}"
                    )
                overlap = complex(np.vdot(reconstructed, reference))
                phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
                action_match_residual = float(
                    np.linalg.norm(reference - phase * reconstructed, ord="fro")
                )
                action_match_bound = float(
                    factorized_action.off_route_residual
                    + np.sqrt(max(1, len(factorized_action.q_phases)))
                    * factorized_action.phase_alignment_residual
                    + factorized_action.matrix_certification_bound
                    + 128.0
                    * np.finfo(np.float64).eps
                    * max(1.0, np.linalg.norm(reference, ord="fro"))
                )
                if action_match_residual > action_match_bound:
                    raise FactorizedTermActionError(
                        "reconstructed full action does not match the active generator: "
                        f"residual={action_match_residual:.6e}, "
                        f"bound={action_match_bound:.6e}"
                    )
                direct_action = compile_factorized_real_channel_action(
                    seeds=seeds,
                    logical_channels=raw_logical.channels,
                    physical_channel_ids=tuple(
                        channel.channel_id for channel in unique_raw
                    ),
                    factorized_action=factorized_action,
                    physical_channel_aliases=exact_duplicate_of,
                )
                generator_images[name] = (vocabulary @ direct_action).tocsc()
                generator_images[name].eliminate_zeros()
                factorized_channel_actions[name] = direct_action
                factorized_generator_names.add(name)
                factorized_action_nnz[name] = int(direct_action.nnz)
                factorized_action_hashes[name] = str(
                    factorized_action.artifact_hash
                )
                factorized_action_match_residuals[name] = action_match_residual
                antiunitary_parities[name] = bool(element.antiunitary)
                per_column_bounds = []
                for channel in unique_raw:
                    raw_norm = float(
                        np.linalg.norm(
                            unique_vectors[unique_index_by_id[channel.channel_id]].data
                        )
                    )
                    factorization_bound = float(
                        (2.0 * action_match_residual + action_match_residual**2)
                        * raw_norm
                    )
                    per_column_bounds.append(
                        raw_error_bounds[channel.channel_id] + factorization_bound
                    )
                generator_column_error_bounds[name] = tuple(per_column_bounds)
                generator_error_bounds[name] = float(
                    np.linalg.norm(np.asarray(per_column_bounds, dtype=np.float64))
                )
                continue
            except (FactorizedTermActionError, FactorizedActionCertificationError) as exc:
                if factorized_actions_required:
                    raise
                factorized_fallback_reasons[name] = str(exc)
                _response_progress(
                    progress_callback,
                    "response basis factorized generator fallback "
                    f"{name}: {exc}",
                    state="diagnostic",
                )
        else:
            if factorized_actions_required:
                raise FactorizedTermActionError(
                    f"missing certified factorized action for generator {name!r}"
                )
            factorized_fallback_reasons[name] = "certified factorized action unavailable"
            _response_progress(
                progress_callback,
                "response basis factorized generator fallback "
                f"{name}: certified factorized action unavailable",
                state="diagnostic",
            )
        transformed_by_channel_id: dict[str, dict[tuple[int, int], sparse.csr_matrix]] = {}
        for seed_id, seed_channels in physical_channels_by_seed.items():
            raw_seed = seeds_by_id[seed_id]
            raw_coefficients = {
                (int(r), int(s)): _canonical_csr(matrix, shape=(dim, dim))
                for (r, s), matrix in raw_seed.coefficients.items()
            }
            transformed_raw = _apply_group_element(
                raw_coefficients,
                element,
                coordinate,
            )
            generator_image_raw_transform_count += 1
            if masks is not None:
                transformed_raw = _mask_polynomial_support(
                    transformed_raw,
                    masks[raw_seed.support_component],
                )
            for channel in seed_channels:
                if channel.component == "real":
                    phased = transformed_raw
                elif channel.component == "imag":
                    phase = -1.0j if element.antiunitary else 1.0j
                    phased = {
                        monomial: _canonical_csr(matrix * phase)
                        for monomial, matrix in transformed_raw.items()
                    }
                else:
                    raise ValueError(
                        f"unknown Hermitian response component {channel.component!r}"
                    )
                transformed_by_channel_id[channel.channel_id] = _hermitian_project(
                    phased,
                    dim=dim,
                )
        columns: list[sparse.csc_matrix] = []
        for channel in unique_raw:
            transformed_channel = replace(
                channel,
                coefficients=transformed_by_channel_id[channel.channel_id]
                or {
                    (0, 0): sparse.csr_matrix(
                        (dim, dim), dtype=np.complex128
                    )
                },
            )
            columns.append(
                _channel_sparse_vector(transformed_channel, coordinate, dim)
            )
        generator_images[name] = sparse.hstack(columns, format="csc")
        antiunitary_parities[name] = bool(element.antiunitary)
        generator_error_bounds[name] = float(
            np.sqrt(
                sum(raw_error_bounds[channel.channel_id] ** 2 for channel in unique_raw)
            )
        )
        generator_column_error_bounds[name] = tuple(
            raw_error_bounds[channel.channel_id] for channel in unique_raw
        )
    _response_progress(
        progress_callback,
        "response basis group generator images done "
        f"in {time.perf_counter() - generator_started:.2f} s | "
        f"generators={len(generator_elements)} factorized="
        f"{len(factorized_generator_names)} raw_transforms="
        f"{generator_image_raw_transform_count}",
        state="done",
    )
    fixed_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis group fixed-space solve start",
        state="start",
    )
    if len(factorized_channel_actions) == len(generator_elements):
        from .response_basis_fixed_compiler import (
            compile_generator_fixed_vocabulary_from_actions,
        )

        fixed = compile_generator_fixed_vocabulary_from_actions(
            vocabulary,
            generator_actions=factorized_channel_actions,
            antiunitary_parities=antiunitary_parities,
            logical_channel_ids=tuple(channel.channel_id for channel in unique_raw),
            generator_column_absolute_error_bounds=generator_column_error_bounds,
            materialize_ambient_vectors=bool(materialize_ambient_vectors),
        )
        fixed_solver = "generator_fixed_subspace__direct_term_action_metric_v1"
    else:
        fixed = compile_generator_fixed_vocabulary(
            vocabulary,
            generator_images=generator_images,
            antiunitary_parities=antiunitary_parities,
            logical_channel_ids=tuple(channel.channel_id for channel in unique_raw),
            generator_absolute_error_bounds=generator_error_bounds,
            generator_column_absolute_error_bounds=generator_column_error_bounds,
            materialize_ambient_vectors=bool(materialize_ambient_vectors),
        )
        fixed_solver = "generator_fixed_subspace__small_coordinate_rrqr"
    _response_progress(
        progress_callback,
        "response basis group fixed-space solve done "
        f"in {time.perf_counter() - fixed_started:.2f} s | "
        f"ambient={vocabulary.shape[1]} fixed={fixed.rank} components="
        f"{fixed.metadata.get('early_direct_sum_certification', {}).get('component_count', 1)}",
        state="done",
    )

    projection_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis group fixed-channel projection/classification start",
        state="start",
    )
    projected_channels: list[ResponseChannel] = []
    projection_records: dict[str, dict[str, Any]] = {}
    fixed_coordinates: dict[str, np.ndarray] = {}
    fixed_action_bounds = {
        str(record["generator"]): float(record["certified_operator_bound"])
        for record in fixed.metadata["fixed_physical_action_certification"]
    }
    fixed_action_operator_bound = max(fixed_action_bounds.values(), default=0.0)
    vocabulary_operator_norm = float(
        fixed.metadata["vocabulary_singular_values"][0]
    )
    vocabulary_smallest_singular = float(
        fixed.metadata["vocabulary_singular_values"][-1]
    )
    fixed_vocabulary_operator_norm_bound = float(
        1.0 / max(vocabulary_smallest_singular, np.finfo(np.float64).tiny)
    )
    for raw_channel in raw_logical.channels:
        raw_vector = logical_raw_vector(raw_channel)
        if bool(materialize_ambient_vectors):
            if fixed.fixed_vectors is None:
                raise RuntimeError("ambient fixed vectors were requested but not materialized")
            coordinates = np.asarray(
                raw_vector.T @ fixed.fixed_vectors,
                dtype=np.float64,
            ).reshape(-1)
            projected: np.ndarray | sparse.csc_matrix = fixed.fixed_vectors @ coordinates
            projection_roundoff = _fixed_projection_roundoff_bound(
                raw_vector,
                fixed.fixed_vectors,
                coordinates,
            )
            projected_vocabulary_coordinates = None
        else:
            if raw_channel.classification == "structural_zero":
                coordinates = np.zeros(fixed.rank, dtype=np.float64)
            else:
                resolution = logical_physical_resolution(raw_channel)
                if resolution is None:
                    raise RuntimeError("nonstructural logical channel resolved to zero")
                representative_index, sign = resolution
                coordinates = sign * np.asarray(
                    fixed.logical_projection_coordinates[:, representative_index],
                    dtype=np.float64,
                )
            projected_vocabulary_coordinates = None
            projected = None
            projection_roundoff = _certified_logical_projection_roundoff_bound(
                raw_norm=float(np.linalg.norm(raw_vector.data)),
                coordinate_norm=float(np.linalg.norm(coordinates)),
                vocabulary_channel_count=vocabulary.shape[1],
                fixed_rank=fixed.rank,
                vocabulary_operator_norm=vocabulary_operator_norm,
                fixed_vocabulary_operator_norm_bound=(
                    fixed_vocabulary_operator_norm_bound
                ),
            )
        norm = float(np.linalg.norm(coordinates))
        error_components = dict(raw_error_components[raw_channel.channel_id])
        error_components["generator_fixed_projection_roundoff"] = projection_roundoff
        structural = raw_channel.classification == "structural_zero"
        raw_norm = float(np.linalg.norm(raw_vector.data))
        fixed_action_approximation_bound = float(
            len(group.elements) * fixed_action_operator_bound * raw_norm
        )
        error_components["generator_fixed_action_approximation"] = (
            fixed_action_approximation_bound
        )
        error_bound = float(sum(error_components.values()))
        fixed_approximation_bound = float(
            fixed_action_approximation_bound + projection_roundoff
        )
        if structural:
            classification = "structural_zero"
        else:
            lower_norm = max(0.0, norm - fixed_approximation_bound)
            upper_norm = norm + fixed_approximation_bound
            if upper_norm <= raw_logical.null_policy.numerical_factor * error_bound:
                classification = "numerical_zero"
            elif lower_norm >= raw_logical.null_policy.confirm_factor * error_bound:
                classification = "confirmed_nonzero"
            else:
                classification = "ambiguous"
        projection_records[raw_channel.channel_id] = {
            "projected": projected,
            "projected_vocabulary_coordinates": projected_vocabulary_coordinates,
            "fixed_coordinates": np.asarray(coordinates, dtype=np.float64),
            "error_bound": error_bound,
            "fixed_approximation_bound": fixed_approximation_bound,
        }
        projected_channels.append(
            replace(
                raw_channel,
                coefficients={
                    (0, 0): sparse.csr_matrix((dim, dim), dtype=np.complex128)
                },
                unnormalized_norm=norm,
                propagated_error_bound=error_bound,
                classification=classification,
                response_scale=norm,
                support_leakage_before_cleanup=float(
                    support_artifact.get(raw_channel.support_component, {}).get(
                        "leakage_before_cleanup", 0.0
                    )
                ),
                error_bound_components=error_components,
                metadata={
                    **dict(raw_channel.metadata),
                    "generator_fixed_projection": True,
                    "generator_fixed_projection_materialized": False,
                    "generator_fixed_residuals": [],
                    "generator_fixed_approximation_bound": fixed_approximation_bound,
                },
            )
        )
        fixed_coordinates[raw_channel.channel_id] = coordinates
    _response_progress(
        progress_callback,
        "response basis group fixed-channel projection/classification done "
        f"in {time.perf_counter() - projection_started:.2f} s | "
        f"logical={len(projected_channels)}",
        state="done",
    )
    provisional_candidates = CandidateResponseSet(
        coordinate=coordinate,
        channels=tuple(projected_channels),
        dim=dim,
        group_artifact=group.artifact(),
        null_policy=raw_logical.null_policy,
        support_artifact=support_artifact,
        adjoint_artifact={
            **dict(raw_logical.adjoint_artifact),
            "compiler": "actual_generator_fixed_subspace_v1",
            "generator_names": [name for name, _ in generator_elements],
            "generator_image_raw_transform_count": int(
                generator_image_raw_transform_count
            ),
            "logical_projection_compiler": (
                "certified_physical_representative_coordinates_v1"
                if not bool(materialize_ambient_vectors)
                else "materialized_ambient_fixed_vectors_v1"
            ),
            "generator_image_compiler": (
                "analytic_identity_term_action_v1"
                if len(builtin_exact_generator_names) == len(generator_elements)
                else (
                    "factorized_q_route_term_action_v2"
                    if len(factorized_generator_names) == len(generator_elements)
                    else (
                        "mixed_factorized_and_full_matrix_v1"
                        if factorized_generator_names
                        else "full_matrix_seed_transform_v1"
                    )
                )
            ),
            "builtin_exact_generator_names": sorted(
                builtin_exact_generator_names
            ),
            "factorized_generator_names": sorted(factorized_generator_names),
            "factorized_action_nnz": dict(sorted(factorized_action_nnz.items())),
            "factorized_action_hashes": dict(
                sorted(factorized_action_hashes.items())
            ),
            "factorized_action_match_residuals": dict(
                sorted(factorized_action_match_residuals.items())
            ),
            "factorized_fallback_reasons": dict(
                sorted(factorized_fallback_reasons.items())
            ),
            "fixed_space_compiler": str(fixed.metadata.get("compiler", "")),
            "fixed_rank": int(fixed.rank),
            "exact_raw_duplicate_count": int(len(exact_duplicate_of)),
            "raw_vector_materialization_count": int(len(physical_vectors_by_id)),
        },
    )
    physical_projected = [
        channel
        for channel in provisional_candidates.channels
        if channel.classification != "structural_zero"
        and not bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    dropped: list[dict[str, Any]] = [
        {
            "channel_id": channel.channel_id,
            "reason": "adjoint_certified_linear_dependency",
            "representative_channel_id": str(
                channel.metadata["adjoint_representative_channel_id"]
            ),
            "adjoint_sign": int(channel.metadata["adjoint_sign"]),
            "certification_status": str(
                channel.metadata["adjoint_certification_status"]
            ),
        }
        for channel in provisional_candidates.channels
        if channel.classification != "structural_zero"
        and bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    if not reduce:
        retained = physical_projected
        selected_confirmed = [
            channel.channel_id
            for channel in retained
            if channel.classification == "confirmed_nonzero"
        ]
    else:
        confirmed = [
            channel
            for channel in physical_projected
            if channel.classification == "confirmed_nonzero"
        ]
        normalized = np.column_stack(
            [
                fixed_coordinates[channel.channel_id]
                / max(channel.unnormalized_norm, np.finfo(float).tiny)
                for channel in confirmed
            ]
        ) if confirmed else np.zeros((fixed.rank, 0), dtype=np.float64)
        if confirmed:
            _q, _r, pivots = scipy.linalg.qr(
                normalized,
                pivoting=True,
                mode="economic",
                check_finite=False,
            )
            order = [int(value) for value in pivots]
        else:
            order = []
        orthogonal: list[np.ndarray] = []
        selected_ids: set[str] = set()
        dependency_records: dict[str, dict[str, Any]] = {}
        for local_index in order:
            channel = confirmed[local_index]
            original = fixed_coordinates[channel.channel_id]
            residual = np.array(original, dtype=np.float64, copy=True)
            for _ in range(2):
                for direction in orthogonal:
                    residual -= direction * float(direction @ residual)
            outside = float(np.linalg.norm(residual))
            condition_roundoff = float(
                np.finfo(float).eps
                * max(1, fixed.rank, len(confirmed))
                * max(channel.unnormalized_norm, 1.0)
            )
            own_floor = float(channel.propagated_error_bound + condition_roundoff)
            if outside > own_floor:
                orthogonal.append(residual / outside)
                selected_ids.add(channel.channel_id)
            else:
                dependency_records[channel.channel_id] = {
                    "channel_id": channel.channel_id,
                    "reason": "target_independent_linear_dependency",
                    "response_absolute_norm": float(channel.unnormalized_norm),
                    "outside_span_residual_absolute": outside,
                    "algebra_floor_absolute": own_floor,
                    "algebra_floor_ratio": outside
                    / max(own_floor, np.finfo(float).tiny),
                    "dependency_policy": "per_column_propagated_error_fail_closed_v1",
                }
        retained = [
            channel
            for channel in physical_projected
            if channel.classification == "ambiguous"
            or (
                channel.classification == "confirmed_nonzero"
                and channel.channel_id in selected_ids
            )
        ]
        for channel in physical_projected:
            if channel.classification == "numerical_zero":
                dropped.append(
                    {
                        "channel_id": channel.channel_id,
                        "reason": "certified_numerical_zero",
                        "response_absolute_norm": float(channel.unnormalized_norm),
                        "algebra_floor_absolute": float(channel.propagated_error_bound),
                    }
                )
        dropped.extend(
            dependency_records[channel_id]
            for channel_id in sorted(dependency_records)
        )
        selected_confirmed = [
            channel.channel_id
            for channel in retained
            if channel.classification == "confirmed_nonzero"
        ]
    retained_ids = {channel.channel_id for channel in retained}
    materialize_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis group retained channel materialization start",
        state="start",
    )
    materialized_by_id: dict[str, ResponseChannel] = {}
    sparse_fixed_vocabulary_coordinates = sparse.csc_matrix(
        fixed.fixed_vocabulary_coordinates
    )
    for raw_channel in provisional_candidates.channels:
        if raw_channel.channel_id not in retained_ids:
            continue
        record = projection_records[raw_channel.channel_id]
        projected = record["projected"]
        if projected is None:
            coordinates = np.asarray(
                record["fixed_coordinates"],
                dtype=np.float64,
            )
            projected_vocabulary_coordinates = (
                sparse_fixed_vocabulary_coordinates
                @ sparse.csc_matrix(coordinates.reshape(-1, 1))
            ).tocsc()
            projected_vocabulary_coordinates.eliminate_zeros()
            projected = (
                vocabulary @ projected_vocabulary_coordinates
            ).tocsc()
            projected.eliminate_zeros()
        projected_coefficients = _coefficient_vector_to_response_map(
            projected,
            coordinate=coordinate,
            dim=dim,
        )
        generator_residuals: list[dict[str, float | str]] = []
        for name, element in generator_elements:
            transformed = _apply_group_element(
                projected_coefficients,
                element,
                coordinate,
            )
            if masks is not None:
                transformed = _mask_polynomial_support(
                    transformed,
                    masks[raw_channel.support_component],
                )
            transformed_channel = replace(
                raw_channel,
                coefficients=transformed
                or {
                    (0, 0): sparse.csr_matrix(
                        (dim, dim), dtype=np.complex128
                    )
                },
            )
            transformed_vector = _channel_sparse_vector(
                transformed_channel,
                coordinate,
                dim,
            ).tocsc()
            if sparse.issparse(projected):
                difference = (transformed_vector - projected).tocsc()
                difference.eliminate_zeros()
                invariance_residual = float(np.linalg.norm(difference.data))
            else:
                invariance_residual = float(
                    np.linalg.norm(
                        np.asarray(transformed_vector.toarray()).reshape(-1)
                        - np.asarray(projected).reshape(-1)
                    )
                )
            transformed_norm = float(np.linalg.norm(transformed_vector.data))
            projected_norm = float(
                np.linalg.norm(projected.data)
                if sparse.issparse(projected)
                else np.linalg.norm(projected)
            )
            invariance_roundoff = float(
                _matvec_gamma(2) * (transformed_norm + projected_norm)
            )
            invariance_bound = float(
                record["error_bound"]
                + record["fixed_approximation_bound"]
                + invariance_roundoff
            )
            if invariance_residual > invariance_bound:
                from .response_basis_fixed_subspace import FixedSubspaceCertificationError

                raise FixedSubspaceCertificationError(
                    f"projected response {raw_channel.channel_id!r} is not fixed by "
                    f"generator {name!r}: residual {invariance_residual:.6e} exceeds "
                    f"its own propagated bound {invariance_bound:.6e}"
                )
            generator_residuals.append(
                {
                    "generator": name,
                    "residual": invariance_residual,
                    "certification_bound": invariance_bound,
                }
            )
        materialized_by_id[raw_channel.channel_id] = replace(
            raw_channel,
            coefficients=projected_coefficients,
            metadata={
                **dict(raw_channel.metadata),
                "generator_fixed_projection_materialized": True,
                "generator_fixed_residuals": generator_residuals,
            },
        )
    _response_progress(
        progress_callback,
        "response basis group retained channel materialization done "
        f"in {time.perf_counter() - materialize_started:.2f} s | "
        f"retained={len(materialized_by_id)}",
        state="done",
    )
    final_channels = tuple(
        materialized_by_id.get(channel.channel_id, channel)
        for channel in provisional_candidates.channels
    )
    candidates = replace(
        provisional_candidates,
        channels=final_channels,
        adjoint_artifact={
            **dict(provisional_candidates.adjoint_artifact),
            "physically_materialized_projected_channel_count": int(
                len(materialized_by_id)
            ),
        },
    )
    retained = [materialized_by_id[channel.channel_id] for channel in retained]
    proof = {
        **dict(fixed.metadata),
        "fixed_vocabulary_block_rule": str(fixed.metadata.get("block_rule", "")),
        "block_rule": "joint_group_adjoint_support_component",
        "solver": fixed_solver,
        "certification_space": (
            "global_two_dimensional_polynomial_coefficient_space_real_imag_stack"
        ),
        "sample_grid_used": False,
        "selected_channel_ids": selected_confirmed,
        "retained_ambiguous_channel_ids": [
            channel.channel_id
            for channel in retained
            if channel.classification == "ambiguous"
        ],
        "dependency_policy": "per_column_propagated_error_fail_closed_v1",
        "logical_to_fixed_coordinates": {
            channel_id: [float(value) for value in fixed_coordinates[channel_id]]
            for channel_id in sorted(fixed_coordinates)
        },
    }
    return GeneratorFixedResponseGroup(
        candidates=candidates,
        retained_channels=tuple(retained),
        reduction_proofs=(proof,),
        dropped_channels=tuple(dropped),
    )


def _seeds_are_all_zero_harmonic(seeds: Sequence[RawPolynomialSeed]) -> bool:
    """Return whether every production seed has an explicit exact p=0 key."""

    for seed in seeds:
        term_key = seed.metadata.get("term_key")
        if not isinstance(term_key, Mapping):
            raise ValueError(
                f"complete response seed {seed.seed_id!r} lacks term_key metadata"
            )
        p_vector = np.asarray(term_key.get("p"), dtype=np.float64)
        if p_vector.shape != (2,) or not np.all(np.isfinite(p_vector)):
            raise ValueError(f"seed {seed.seed_id!r} has invalid explicit harmonic vector")
        if np.any(p_vector != 0.0):
            return False
    return True


def _fixed_groups_have_disjoint_support(
    groups: Sequence[GeneratorFixedResponseGroup],
    *,
    coordinate: PolynomialCoordinateBasis,
    dim: int,
) -> bool:
    """Certify a direct sum from exact disjoint coefficient-row support."""

    occupied: set[int] = set()
    for group in groups:
        group_rows: set[int] = set()
        for channel in group.retained_channels:
            vector = _channel_sparse_vector(channel, coordinate, dim)
            group_rows.update(int(row) for row in vector.indices)
        if occupied.intersection(group_rows):
            return False
        occupied.update(group_rows)
    return True


def _compiled_basis_from_fixed_groups(
    candidates: CandidateResponseSet,
    groups: Sequence[GeneratorFixedResponseGroup],
    *,
    identity_payload: Mapping[str, Any],
    reduce: bool,
) -> CompiledResponseBasis:
    """Assemble independently certified p=0 fixed groups without redoing full reduction."""

    retained = [channel for group in groups for channel in group.retained_channels]
    retained.sort(
        key=lambda channel: (
            int(channel.metadata.get("term_index", -1)),
            0 if channel.component == "real" else 1,
        )
    )
    proofs = [dict(proof) for group in groups for proof in group.reduction_proofs]
    dropped = [dict(record) for group in groups for record in group.dropped_channels]

    # Different symmetry-operation groups are not assumed to form a direct sum.
    # Their already-small retained spaces receive one final target-independent
    # coefficient-space certification; a single group needs no second pass.
    direct_sum_certified = bool(
        len(groups) > 1
        and retained
        and _fixed_groups_have_disjoint_support(
            groups,
            coordinate=candidates.coordinate,
            dim=candidates.dim,
        )
    )
    if direct_sum_certified:
        proofs.append(
            {
                "solver": "exact_disjoint_coefficient_support_direct_sum",
                "block_rule": "fixed_group_union_support",
                "group_count": int(len(groups)),
                "certification_space": (
                    "global_two_dimensional_polynomial_coefficient_space_"
                    "real_imag_stack"
                ),
                "sample_grid_used": False,
            }
        )
    if bool(reduce) and len(groups) > 1 and retained and not direct_sum_certified:
        keep, cross_proofs, cross_dropped = _reduce_confirmed_channels(
            retained,
            coordinate=candidates.coordinate,
            dim=candidates.dim,
            confirm_factor=candidates.null_policy.confirm_factor,
        )
        retained = [retained[index] for index in keep]
        proofs.extend(cross_proofs)
        dropped.extend(cross_dropped)

    provisional = CompiledResponseBasis(
        coordinate=candidates.coordinate,
        channels=tuple(retained),
        dim=int(candidates.dim),
        identity_record=_json_record(identity_payload),
        candidate_artifact=candidates.artifact(),
        reduction_proofs=tuple(proofs),
        dropped_channels=tuple(dropped),
        basis_hash="",
    )
    object.__setattr__(provisional, "basis_hash", provisional._compute_hash())
    return provisional


_RESPONSE_BASIS_CACHE: dict[str, CompiledResponseBasis] = {}
_RESPONSE_BASIS_COLD_COMPILE_COUNT: dict[str, int] = {}


def clear_response_basis_cache() -> None:
    _RESPONSE_BASIS_CACHE.clear()
    _RESPONSE_BASIS_COLD_COMPILE_COUNT.clear()


def response_basis_cache_stats() -> dict[str, Any]:
    keys = sorted(_RESPONSE_BASIS_COLD_COMPILE_COUNT)
    return {
        "keys": keys,
        "cold_compile_count_by_key": {
            key: int(_RESPONSE_BASIS_COLD_COMPILE_COUNT[key])
            for key in keys
        },
        "cached_basis_count": int(len(_RESPONSE_BASIS_CACHE)),
    }


def _cartesian_action_matrix(action_map: Mapping[str, Any], *, field: str) -> np.ndarray:
    if not isinstance(action_map, Mapping):
        raise ValueError(f"complete_linear_v2 requires explicit {field} metadata")
    map_type = str(action_map.get("type", "")).strip().lower()
    if map_type == "identity":
        return np.eye(2, dtype=np.float64)
    if map_type == "negation":
        return -np.eye(2, dtype=np.float64)
    if map_type == "rotation":
        angle = np.deg2rad(float(action_map.get("angle_deg", 0.0)))
        return np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
    if map_type == "reflection":
        angle = np.deg2rad(float(action_map.get("axis_deg", 0.0)))
        axis = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        return 2.0 * np.outer(axis, axis) - np.eye(2, dtype=np.float64)
    raise ValueError(f"unsupported explicit {field}.type {action_map.get('type')!r}")


def _canonical_lattice_map(
    cartesian_map: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    *,
    tolerance: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    reciprocal_columns = np.column_stack(
        (np.asarray(bM1, dtype=np.float64), np.asarray(bM2, dtype=np.float64))
    )
    if reciprocal_columns.shape != (2, 2) or abs(float(np.linalg.det(reciprocal_columns))) <= np.finfo(float).eps:
        raise ValueError("complete_linear_v2 requires a nonsingular explicit reciprocal basis")
    lattice_map = np.linalg.solve(reciprocal_columns, cartesian_map @ reciprocal_columns)
    rounded = np.rint(lattice_map).astype(np.int64)
    residual = float(np.linalg.norm(lattice_map - rounded))
    if residual > float(tolerance):
        raise ValueError(
            "explicit k_map is not an integer action in the model reciprocal basis: "
            f"residual={residual:.3e}"
        )
    if abs(int(round(np.linalg.det(rounded)))) != 1:
        raise ValueError("explicit canonical reciprocal-lattice k_map must be unimodular")
    return tuple(tuple(int(value) for value in row) for row in rounded)


def _sector_slot_records(sectors: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, int]]:
    names_by_slot = ["L1", "L2"]
    slot_by_name = {"L1": 0, "L2": 1, "1": 0, "2": 1}
    for record in sectors:
        if not isinstance(record, Mapping):
            continue
        qset = str(record.get("qset", ""))
        if qset not in {"qset1", "qset2"}:
            raise ValueError(f"sector {record.get('name')!r} has unsupported qset {qset!r}")
        slot = 0 if qset == "qset1" else 1
        name = str(record.get("name", names_by_slot[slot]))
        if names_by_slot[slot] in {"L1", "L2"}:
            names_by_slot[slot] = name
        slot_by_name[name] = slot
    return names_by_slot, slot_by_name


def _sector_permutation_from_metadata(
    sector_map: Any,
    sectors: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    names_by_slot, slot_by_name = _sector_slot_records(sectors)
    if isinstance(sector_map, str):
        normalized = sector_map.strip().lower()
        if normalized == "identity":
            return (0, 1)
        if normalized in {"layer_exchange", "sector_exchange", "exchange", "swap"}:
            return (1, 0)
        raise ValueError(f"unsupported explicit sector_map {sector_map!r}")
    if not isinstance(sector_map, Mapping):
        raise ValueError("complete_linear_v2 requires explicit sector_map metadata")
    result = []
    for slot, name in enumerate(names_by_slot):
        target = sector_map.get(name, sector_map.get(str(slot + 1), name))
        target_name = str(target)
        if target_name not in slot_by_name:
            raise ValueError(
                f"sector_map target {target_name!r} is not present in the model sector layout"
            )
        result.append(int(slot_by_name[target_name]))
    return _validate_permutation(result, name="sector_map")  # type: ignore[return-value]


def _q_permutation_from_metadata(
    q_map: Mapping[str, Any],
    sector_permutation: tuple[int, int],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    *,
    tolerance: float,
    active_slots: tuple[int, ...] = (0, 1),
) -> tuple[int, ...]:
    qsets = (
        np.asarray(Q_set1, dtype=np.float64),
        np.asarray(Q_set2, dtype=np.float64),
    )
    q_action = _cartesian_action_matrix(q_map, field="q_map")
    offsets = (0, int(qsets[0].shape[0]))
    permutation: list[int] = []
    scale = max(
        1.0,
        *(float(np.linalg.norm(vector)) for qset in qsets for vector in qset),
    )
    threshold = float(tolerance) * scale
    for source_slot in active_slots:
        qset = qsets[source_slot]
        target_slot = int(sector_permutation[source_slot])
        if target_slot not in active_slots:
            raise ValueError("q_map sends an active Q set to an inactive target sector")
        target_qset = qsets[target_slot]
        for vector in qset:
            mapped = q_action @ vector
            if target_qset.shape[0] == 0:
                raise ValueError("q_map sends a populated Q set to an empty target Q set")
            distances = np.linalg.norm(target_qset - mapped, axis=1)
            matches = np.flatnonzero(distances <= threshold)
            if matches.size != 1:
                raise ValueError(
                    "explicit q_map does not define a unique integer Q permutation: "
                    f"source_slot={source_slot + 1}, vector={vector.tolist()}, matches={matches.tolist()}"
                )
            permutation.append(offsets[target_slot] + int(matches[0]))
    return _validate_permutation(permutation, name="q_map permutation")


def _projected_basis_permutations(
    operation: Mapping[str, Any],
    sectors: Sequence[Mapping[str, Any]],
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    *,
    tolerance: float,
    active_slots: tuple[int, ...] = (0, 1),
) -> tuple[tuple[int, ...], tuple[int, int]] | None:
    """Resolve the certified discrete action in the projected model basis."""

    basis_action = operation.get("model_basis_action")
    if not isinstance(basis_action, Mapping) or not bool(
        basis_action.get("complete", False)
    ):
        return None
    name = str(operation.get("name", "<unnamed>"))
    items = basis_action.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError(
            f"operation {name!r} complete model_basis_action requires Q permutation items"
        )

    qsets = (
        np.asarray(Q_set1, dtype=np.float64),
        np.asarray(Q_set2, dtype=np.float64),
    )
    names_by_slot, slot_by_name = _sector_slot_records(sectors)
    offsets = (0, int(qsets[0].shape[0]))
    expected_active_sources = {
        (slot, q_index)
        for slot in active_slots
        for qset in (qsets[slot],)
        for q_index in range(int(qset.shape[0]))
    }
    expected_sources = {
        (slot, q_index)
        for slot, qset in enumerate(qsets)
        for q_index in range(int(qset.shape[0]))
    }
    targets_by_source: dict[tuple[int, int], tuple[int, int]] = {}
    seen_targets: set[tuple[int, int]] = set()
    target_slots_by_source: dict[int, set[int]] = {0: set(), 1: set()}
    q_scale = max(
        1.0,
        *(
            float(np.linalg.norm(vector))
            for qset in qsets
            for vector in qset
        ),
    )
    residual_limit = float(tolerance) * q_scale

    for item_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"operation {name!r} model_basis_action item {item_index} must be a mapping"
            )
        source_name = str(item.get("source_sector", ""))
        target_name = str(item.get("target_sector", ""))
        if source_name not in slot_by_name or target_name not in slot_by_name:
            raise ValueError(
                f"operation {name!r} model_basis_action references an unknown sector: "
                f"{source_name!r} -> {target_name!r}"
            )
        source_slot = int(slot_by_name[source_name])
        target_slot = int(slot_by_name[target_name])
        if source_slot not in active_slots:
            if target_slot in active_slots:
                raise ValueError(
                    f"operation {name!r} model_basis_action maps an inactive "
                    "source sector into the active model basis"
                )
            continue
        if target_slot not in active_slots:
            raise ValueError(
                f"operation {name!r} model_basis_action maps an active source "
                "sector outside the active model basis"
            )
        source_index = int(item.get("source_q_index", -1))
        target_index = int(item.get("target_q_index", -1))
        if not 0 <= source_index < int(qsets[source_slot].shape[0]):
            raise ValueError(
                f"operation {name!r} model_basis_action source Q index "
                f"{source_index} is out of range for {names_by_slot[source_slot]}"
            )
        if not 0 <= target_index < int(qsets[target_slot].shape[0]):
            raise ValueError(
                f"operation {name!r} model_basis_action target Q index "
                f"{target_index} is out of range for {names_by_slot[target_slot]}"
            )
        source_key = (source_slot, source_index)
        target_key = (target_slot, target_index)
        if source_key in targets_by_source:
            raise ValueError(
                f"operation {name!r} model_basis_action repeats source Q slot "
                f"{(names_by_slot[source_slot], source_index)}"
            )
        if target_key in seen_targets:
            raise ValueError(
                f"operation {name!r} model_basis_action is not bijective; target Q slot "
                f"{(names_by_slot[target_slot], target_index)} is repeated"
            )
        residual = float(item.get("q_residual", 0.0))
        if not np.isfinite(residual) or abs(residual) > residual_limit:
            raise ValueError(
                f"operation {name!r} model_basis_action Q residual {residual:.3e} "
                f"exceeds {residual_limit:.3e}"
            )
        targets_by_source[source_key] = target_key
        seen_targets.add(target_key)
        target_slots_by_source[source_slot].add(target_slot)

    actual_active_sources = {
        source for source in targets_by_source if source[0] in active_slots
    }
    if actual_active_sources != expected_active_sources:
        missing = sorted(expected_active_sources - actual_active_sources)
        extra = sorted(actual_active_sources - expected_active_sources)
        raise ValueError(
            f"operation {name!r} model_basis_action has incomplete source Q coverage; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    recorded_sector_map = basis_action.get("sector_map")
    if recorded_sector_map is None:
        raise ValueError(
            f"operation {name!r} complete model_basis_action requires sector_map"
        )
    recorded_sector_permutation = _sector_permutation_from_metadata(
        recorded_sector_map,
        sectors,
    )
    for source_slot, qset in enumerate(qsets):
        if source_slot in active_slots:
            continue
        target_slot = int(recorded_sector_permutation[source_slot])
        if target_slot in active_slots:
            raise ValueError(
                f"operation {name!r} maps an inactive sector into the active model basis"
            )
        if int(qset.shape[0]) != int(qsets[target_slot].shape[0]):
            raise ValueError(
                f"operation {name!r} inactive-sector Q counts are inconsistent"
            )
        for q_index in range(int(qset.shape[0])):
            source_key = (source_slot, q_index)
            target_key = (target_slot, q_index)
            if target_key in seen_targets:
                raise ValueError(
                    f"operation {name!r} inactive-sector placeholder is not bijective"
                )
            targets_by_source[source_key] = target_key
            seen_targets.add(target_key)
            target_slots_by_source[source_slot].add(target_slot)
    if set(targets_by_source) != expected_sources or len(seen_targets) != len(
        expected_sources
    ):
        raise ValueError(
            f"operation {name!r} model_basis_action Q permutation is not bijective"
        )
    derived_sector_permutation: list[int] = []
    for source_slot, targets in target_slots_by_source.items():
        if len(targets) > 1:
            raise ValueError(
                f"operation {name!r} model_basis_action maps one source sector "
                "to multiple target sectors"
            )
        derived_sector_permutation.append(
            next(iter(targets))
            if targets
            else int(recorded_sector_permutation[source_slot])
        )
    derived_sector_tuple = _validate_permutation(
        derived_sector_permutation,
        name="model_basis_action sector permutation",
    )
    if derived_sector_tuple != recorded_sector_permutation:
        raise ValueError(
            f"operation {name!r} model_basis_action sector_map is inconsistent "
            "with its Q permutation items"
        )

    q_permutation = tuple(
        offsets[target_slot] + target_index
        for source_key in sorted(expected_sources)
        for target_slot, target_index in (targets_by_source[source_key],)
    )
    return (
        _validate_permutation(q_permutation, name="model_basis_action Q permutation"),
        derived_sector_tuple,
    )


def finite_group_from_model_actions(
    operations: Sequence[Mapping[str, Any]],
    *,
    symmetry_gen: Any,
    bM1: np.ndarray,
    bM2: np.ndarray,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    sectors: Sequence[Mapping[str, Any]],
    n_orb: tuple[int, int] | None = None,
    factorized_actions: Mapping[str, Any] | None = None,
    max_group_size: int = 256,
    max_word_length: int = 32,
    closure_tolerance: float = 1.0e-8,
) -> FiniteGroup:
    active_slots = tuple(
        slot
        for slot, count in enumerate(n_orb if n_orb is not None else (1, 1))
        if int(count) > 0
    )
    if not active_slots:
        raise ValueError("complete_linear_v2 requires at least one active orbital sector")
    qsets = (np.asarray(Q_set1), np.asarray(Q_set2))
    q_count = int(sum(qset.shape[0] for qset in qsets))
    if not operations:
        if symmetry_gen is None:
            raise ValueError("complete_linear_v2 requires an explicit symmetry generator or an empty identity action")
        dim = int(getattr(symmetry_gen, "dim", 0) or 0)
        if dim <= 0:
            matrices = getattr(symmetry_gen, "matrices", {})
            if matrices:
                dim = int(np.asarray(next(iter(matrices.values()))).shape[0])
        if dim <= 0:
            raise ValueError("cannot infer identity action dimension from symmetry generator")
        return identity_finite_group(dim, q_size=q_count, sector_size=2)
    generators: list[FiniteGroupGenerator] = []
    for raw_operation in operations:
        if not isinstance(raw_operation, Mapping):
            raise ValueError("complete_linear_v2 symmetry operations must be explicit mappings")
        operation = dict(raw_operation)
        resolved_action = operation.get("internal_resolved_action")
        if isinstance(resolved_action, Mapping):
            for field in ("antiunitary", "k_map", "q_map", "sector_map"):
                if field in resolved_action:
                    operation[field] = resolved_action[field]
        missing = [
            field
            for field in ("antiunitary", "k_map", "q_map", "sector_map")
            if field not in operation
        ]
        if missing:
            if "q_map" in missing:
                raise ValueError("complete_linear_v2 requires explicit q_map metadata")
            raise ValueError(f"complete_linear_v2 operation is missing explicit action metadata: {missing}")
        name = str(operation.get("name", ""))
        if not name:
            raise ValueError("complete_linear_v2 operation requires an explicit family name")
        forward = _cartesian_action_matrix(operation["k_map"], field="k_map")
        canonical = _canonical_lattice_map(
            forward,
            np.asarray(bM1),
            np.asarray(bM2),
            tolerance=closure_tolerance,
        )
        projected_permutations = _projected_basis_permutations(
            operation,
            sectors,
            np.asarray(Q_set1),
            np.asarray(Q_set2),
            tolerance=closure_tolerance,
            active_slots=active_slots,
        )
        if projected_permutations is None:
            sector_permutation = _sector_permutation_from_metadata(
                operation["sector_map"], sectors
            )
            q_permutation = _q_permutation_from_metadata(
                operation["q_map"],
                sector_permutation,
                np.asarray(Q_set1),
                np.asarray(Q_set2),
                tolerance=closure_tolerance,
            )
        else:
            q_permutation, sector_permutation = projected_permutations
        if symmetry_gen is None:
            raise ValueError("complete_linear_v2 cannot infer internal U without exactified symmetry matrices")
        internal_u = np.asarray(
            symmetry_gen.get_operator(name, operation.get("params")),
            dtype=np.complex128,
        )
        factorized_action = (factorized_actions or {}).get(name)
        if factorized_action is not None and projected_permutations is not None:
            if str(getattr(factorized_action, "name", "")) != name:
                raise ValueError(
                    f"factorized projected action name mismatch for {name!r}: "
                    f"{getattr(factorized_action, 'name', None)!r}"
                )
            if bool(getattr(factorized_action, "antiunitary", False)) != bool(
                operation["antiunitary"]
            ):
                raise ValueError(
                    f"factorized antiunitary parity for {name!r} does not match "
                    "the projected continuum action"
                )
            factorized_q_permutation = tuple(
                int(value) for value in factorized_action.q_permutation
            )
            if factorized_q_permutation != tuple(q_permutation):
                raise ValueError(
                    f"factorized {name!r} Q permutation does not match the "
                    "projected model_basis_action"
                )
            factorized_sector_permutation = tuple(
                int(value) for value in factorized_action.sector_permutation
            )
            if factorized_sector_permutation != tuple(sector_permutation):
                raise ValueError(
                    f"factorized {name!r} sector permutation does not match the "
                    "projected model_basis_action"
                )
            factorized_forward = np.asarray(
                factorized_action.k_forward,
                dtype=np.float64,
            )
            map_bound = float(
                128.0
                * np.finfo(np.float64).eps
                * max(1.0, np.linalg.norm(forward, ord=2))
            )
            if factorized_forward.shape != (2, 2) or not np.allclose(
                factorized_forward,
                forward,
                rtol=0.0,
                atol=map_bound,
            ):
                raise ValueError(
                    f"factorized {name!r} Cartesian k action does not match the "
                    "physical model action"
                )
        generators.append(
            FiniteGroupGenerator(
                name=name,
                antiunitary=bool(operation["antiunitary"]),
                canonical_k_map=canonical,
                q_permutation=q_permutation,
                sector_permutation=sector_permutation,
                k_forward=tuple(tuple(float(value) for value in row) for row in forward),
                internal_u=internal_u,
            )
        )
    return build_finite_group(
        generators,
        max_group_size=max_group_size,
        max_word_length=max_word_length,
        closure_tolerance=closure_tolerance,
    )


def _support_masks_from_seeds(
    seeds: Sequence[RawPolynomialSeed],
    *,
    group: FiniteGroup,
    coordinate: PolynomialCoordinateBasis,
    internal_actions_by_word: Mapping[
        tuple[str, ...], sparse.spmatrix | np.ndarray
    ]
    | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Build authored masks and close declared representative support.

    A complete template may author one representative of a Fourier-harmonic
    orbit.  Its allowed structural support is the exact group+adjoint closure
    of that representative.  ``orbit_representative`` makes the same closure
    explicit while preserving the authored seed vocabulary.  An
    ``explicit_reduced`` template instead treats the authored mask as a
    contract and is checked fail-closed later.
    """

    masks: dict[str, np.ndarray] = {}
    policies_by_component: dict[str, set[str]] = {}
    for seed in seeds:
        mask = masks.setdefault(seed.support_component, np.zeros((seed.dim, seed.dim), dtype=bool))
        policy = str(seed.metadata.get("term_space_policy", ""))
        if policy not in {"complete", "explicit_reduced", "orbit_representative"}:
            raise ValueError(
                f"support component {seed.support_component!r} seed {seed.seed_id!r} "
                "lacks an explicit complete/explicit_reduced/orbit_representative policy"
            )
        policies_by_component.setdefault(seed.support_component, set()).add(policy)
        for matrix in seed.coefficients.values():
            coo = matrix.tocoo()
            mask[coo.row, coo.col] = True
    for component, policies in policies_by_component.items():
        if len(policies) != 1:
            raise ValueError(
                f"support component {component!r} mixes term-space policies: {sorted(policies)}"
            )

    authored_counts = {component: int(np.count_nonzero(mask)) for component, mask in masks.items()}
    structurally_closed_components = {
        component
        for component, mask in masks.items()
        if _exact_structural_support_is_group_adjoint_closed(mask, group=group)
    }
    polynomial_pullbacks_by_word = {
        tuple(str(value) for value in element.canonical_word): (
            _monomial_pullback_table(element, coordinate)
        )
        for element in group.elements
    }
    internal_actions = (
        {
            tuple(str(value) for value in word): _canonical_csr(matrix)
            for word, matrix in internal_actions_by_word.items()
        }
        if internal_actions_by_word is not None
        else None
    )
    if internal_actions is not None:
        expected_words = set(polynomial_pullbacks_by_word)
        if set(internal_actions) != expected_words:
            raise ValueError(
                "support-closure internal actions must cover every canonical group word"
            )
    internal_monomial_actions = (
        {
            word: _monomial_similarity_action(matrix)
            for word, matrix in internal_actions.items()
        }
        if internal_actions is not None
        else {}
    )
    if internal_monomial_actions and all(
        action is not None for action in internal_monomial_actions.values()
    ):
        # A certified monomial internal action maps each matrix entry to one
        # matrix entry.  Close the already-unioned authored component mask
        # directly under every finite-group route and adjoint, rather than
        # transforming every numerical seed separately.
        for component, mask in masks.items():
            if policies_by_component[component] not in (
                {"complete"},
                {"orbit_representative"},
            ):
                continue
            authored_mask = np.asarray(mask, dtype=bool).copy()
            source_rows, source_cols = np.nonzero(authored_mask)
            for action in internal_monomial_actions.values():
                assert action is not None
                target_rows = action[0]
                routed_rows = target_rows[source_rows]
                routed_cols = target_rows[source_cols]
                mask[routed_rows, routed_cols] = True
                mask[routed_cols, routed_rows] = True
            structurally_closed_components.add(component)

    def expand_until_certified(
        mask: np.ndarray,
        coefficients: Mapping[tuple[int, int], sparse.csr_matrix],
        *,
        error_bound: float,
    ) -> None:
        """Add the minimum deterministic pair support needed by one orbit image.

        Support is structural and therefore indexed only by ``(row, col)``.
        Numerical group-action tails stay outside the mask when their combined
        coefficient-space Frobenius norm is already within the propagated
        error bound.  This prevents exactified roundoff from turning a sparse
        structural orbit into an effectively dense one.
        """

        pair_squared_norms: dict[tuple[int, int], float] = {}
        for matrix in coefficients.values():
            coo = matrix.tocoo()
            for row, col, value in zip(coo.row, coo.col, coo.data):
                key = (int(row), int(col))
                if mask[key]:
                    continue
                pair_squared_norms[key] = pair_squared_norms.get(key, 0.0) + float(abs(value) ** 2)
        remaining_squared = float(sum(pair_squared_norms.values()))
        bound_squared = float(max(error_bound, 0.0) ** 2)
        if remaining_squared <= bound_squared:
            return
        ordered = sorted(
            pair_squared_norms.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
        for (row, col), squared_norm in ordered:
            mask[row, col] = True
            remaining_squared = max(0.0, remaining_squared - squared_norm)
            if remaining_squared <= bound_squared:
                break

    for seed in seeds:
        if policies_by_component[seed.support_component] not in (
            {"complete"},
            {"orbit_representative"},
        ):
            continue
        if seed.support_component in structurally_closed_components:
            continue
        mask = masks[seed.support_component]
        raw = {
            (int(r), int(s)): _canonical_csr(matrix, shape=(seed.dim, seed.dim))
            for (r, s), matrix in seed.coefficients.items()
        }
        error_bound = _propagated_error_bound(
            _coefficient_norm(raw),
            dim=seed.dim,
            degree=coordinate.max_degree,
            group=group,
        ) + float(seed.metadata.get("coordinate_conversion_error_bound", 0.0))
        adjoint = _adjoint_polynomial(raw)
        expand_until_certified(mask, adjoint, error_bound=error_bound)
        for element in group.elements:
            word = tuple(str(value) for value in element.canonical_word)
            transformed = _apply_group_element(
                raw,
                element,
                coordinate,
                internal_u_override=(
                    None if internal_actions is None else internal_actions[word]
                ),
                internal_monomial_action=internal_monomial_actions.get(word),
                monomial_pullbacks=polynomial_pullbacks_by_word[word],
            )
            expand_until_certified(
                mask,
                transformed,
                error_bound=error_bound,
            )
            expand_until_certified(
                mask,
                _adjoint_polynomial(transformed),
                error_bound=error_bound,
            )

    policy_artifact: dict[str, dict[str, Any]] = {}
    for component, mask in masks.items():
        policy = next(iter(policies_by_component[component]))
        closed_count = int(np.count_nonzero(mask))
        policy_artifact[component] = {
            "term_space_policy": policy,
            "support_mask_policy": (
                "complete_symmetry_adjoint_closure_v1"
                if policy == "complete"
                else (
                    "orbit_representative_symmetry_adjoint_closure_v1"
                    if policy == "orbit_representative"
                    else "explicit_reduced_authored_v1"
                )
            ),
            "authored_support_entry_count": authored_counts[component],
            "certified_support_entry_count": closed_count,
            "symmetry_adjoint_added_entry_count": closed_count - authored_counts[component],
            "structural_support_closure_certified": bool(
                component in structurally_closed_components
            ),
        }
    return masks, policy_artifact


def _basis_layout_identity(
    config: Any,
    groups: Sequence[FiniteGroup],
    coordinate: PolynomialCoordinateBasis,
    factorized_actions_by_group: Sequence[Mapping[str, Any] | None] | None = None,
    joint_route_actions_by_group: Sequence[Mapping[str, Any] | None] | None = None,
    joint_artifact_hashes_by_group: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    qset1 = np.asarray(config.Q_set1, dtype=np.float64)
    qset2 = np.asarray(config.Q_set2, dtype=np.float64)
    layers: list[int] = []
    q_indices: list[int] = []
    orbitals: list[int] = []
    for layer, qset, n_orb in (
        (1, qset1, int(config.n_orb1)),
        (2, qset2, int(config.n_orb2)),
    ):
        for orbital in range(n_orb):
            for q_index in range(qset.shape[0]):
                layers.append(layer)
                q_indices.append(int(q_index))
                orbitals.append(int(orbital))
    exactified_matrices: dict[str, np.ndarray] = {}
    q_permutations: dict[str, np.ndarray] = {}
    sector_permutations: dict[str, np.ndarray] = {}
    canonical_k_maps: dict[str, np.ndarray] = {}
    k_pullbacks: dict[str, np.ndarray] = {}
    antiunitary_parities: dict[str, bool] = {}
    group_limits: list[dict[str, int]] = []
    for group_index, group in enumerate(groups):
        group_limits.append(
            {
                "max_group_size": int(group.max_group_size),
                "max_word_length": int(group.max_word_length),
            }
        )
        for element_index, element in enumerate(group.elements):
            name = f"group_{group_index}_element_{element_index}"
            exactified_matrices[name] = np.asarray(element.internal_u, dtype=np.complex128)
            q_permutations[name] = np.asarray(element.q_permutation, dtype=np.int64)
            sector_permutations[name] = np.asarray(element.sector_permutation, dtype=np.int64)
            canonical_k_maps[name] = np.asarray(element.canonical_k_map, dtype=np.int64)
            k_pullbacks[name] = np.asarray(element.k_pullback, dtype=np.float64)
            antiunitary_parities[name] = bool(element.antiunitary)
    factorized_groups = list(
        factorized_actions_by_group
        if factorized_actions_by_group is not None
        else ({} for _ in groups)
    )
    if len(factorized_groups) != len(groups):
        raise ValueError("factorized action group layout must match finite groups")
    joint_route_groups = list(
        joint_route_actions_by_group
        if joint_route_actions_by_group is not None
        else ({} for _ in groups)
    )
    joint_hashes = list(
        joint_artifact_hashes_by_group
        if joint_artifact_hashes_by_group is not None
        else (None for _ in groups)
    )
    if len(joint_route_groups) != len(groups) or len(joint_hashes) != len(groups):
        raise ValueError("joint route action group layout must match finite groups")
    return {
        "basis_layout": {
            "order": "qset-slot-major_orbital-major_q-index-fastest",
            "basis_layer": np.asarray(layers, dtype=np.int16),
            "basis_q_index": np.asarray(q_indices, dtype=np.int32),
            "basis_orbital": np.asarray(orbitals, dtype=np.int16),
        },
        "q_vectors": {"qset1": qset1, "qset2": qset2},
        "basis_ordering": np.arange(len(layers), dtype=np.int64),
        "n_orb": np.asarray([int(config.n_orb1), int(config.n_orb2)], dtype=np.int64),
        "exactified_matrices": exactified_matrices,
        "antiunitary_parities": antiunitary_parities,
        "canonical_k_maps": canonical_k_maps,
        "k_pullbacks": k_pullbacks,
        "q_permutations": q_permutations,
        "sector_permutations": sector_permutations,
        "group_limits": group_limits,
        "factorized_action_hashes": [
            {
                str(name): str(getattr(action, "artifact_hash", ""))
                for name, action in sorted((actions or {}).items())
            }
            for actions in factorized_groups
        ],
        "joint_route_action_hashes": [
            {
                "joint_artifact_hash": (
                    None if joint_hash is None else str(joint_hash)
                ),
                "generator_matrix_hashes": {
                    str(name): hash_array(
                        materialize_block_route_action(action)
                    )
                    for name, action in sorted((actions or {}).items())
                },
            }
            for actions, joint_hash in zip(joint_route_groups, joint_hashes)
        ],
        "polynomial_coordinate": coordinate.metadata(),
        "dtype": "complex128",
        "response_normalization": RESPONSE_NORMALIZATION_V1,
        "coefficient_normalization": COEFFICIENT_NORMALIZATION_V1,
        "regularization_normalization": REGULARIZATION_NORMALIZATION_V1,
        "compiler_version": COMPILER_VERSION,
        "term_templates": getattr(config, "term_templates", []),
    }


_PERSISTENT_CACHE_INPUT_FIELDS = frozenset(
    {
        "identity",
        "coordinate",
        "reduce",
        "groups",
        "group_internal_u",
        "seeds",
    }
)
_PERSISTENT_CACHE_IDENTITY_FIELDS = frozenset(
    {
        "basis_layout",
        "q_vectors",
        "basis_ordering",
        "n_orb",
        "exactified_matrices",
        "antiunitary_parities",
        "canonical_k_maps",
        "k_pullbacks",
        "q_permutations",
        "sector_permutations",
        "group_limits",
        "factorized_action_hashes",
        "joint_route_action_hashes",
        "polynomial_coordinate",
        "dtype",
        "response_normalization",
        "coefficient_normalization",
        "regularization_normalization",
        "compiler_version",
        "term_templates",
    }
)


def _validate_persistent_cache_input_record(input_record: Mapping[str, Any]) -> None:
    """Fail closed unless the cache key sees the complete compiler identity."""

    if not isinstance(input_record, Mapping) or not input_record:
        raise ValueError("complete_linear_v2 persistent cache input is incomplete")
    missing_top = sorted(_PERSISTENT_CACHE_INPUT_FIELDS.difference(input_record))
    if missing_top:
        raise ValueError(
            "complete_linear_v2 persistent cache input is incomplete; missing "
            + ", ".join(missing_top)
        )
    unexpected_top = sorted(set(input_record).difference(_PERSISTENT_CACHE_INPUT_FIELDS))
    if unexpected_top:
        raise ValueError(
            "complete_linear_v2 persistent cache input has unexpected fields: "
            + ", ".join(str(value) for value in unexpected_top)
        )
    identity = input_record.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("complete_linear_v2 persistent cache input identity must be a mapping")
    missing_identity = sorted(_PERSISTENT_CACHE_IDENTITY_FIELDS.difference(identity))
    if missing_identity:
        raise ValueError(
            "complete_linear_v2 persistent cache input identity is incomplete; missing "
            + ", ".join(missing_identity)
        )
    unexpected_identity = sorted(set(identity).difference(_PERSISTENT_CACHE_IDENTITY_FIELDS))
    if unexpected_identity:
        raise ValueError(
            "complete_linear_v2 persistent cache input identity has unexpected fields: "
            + ", ".join(str(value) for value in unexpected_identity)
        )
    coordinate = input_record.get("coordinate")
    if not isinstance(coordinate, Mapping) or not coordinate:
        raise ValueError("complete_linear_v2 persistent cache input coordinate must be a non-empty mapping")
    if _canonical_json(coordinate) != _canonical_json(identity.get("polynomial_coordinate")):
        raise ValueError(
            "complete_linear_v2 persistent cache coordinate must match identity.polynomial_coordinate"
        )
    if not isinstance(input_record.get("reduce"), bool):
        raise ValueError("complete_linear_v2 persistent cache input reduce must be bool")
    groups = input_record.get("groups")
    group_internal_u = input_record.get("group_internal_u")
    if not isinstance(groups, list) or not groups:
        raise ValueError("complete_linear_v2 persistent cache input groups must be a non-empty list")
    if not isinstance(group_internal_u, list) or len(group_internal_u) != len(groups):
        raise ValueError(
            "complete_linear_v2 persistent cache input group_internal_u must match groups"
        )
    if not isinstance(input_record.get("seeds"), list) or not input_record["seeds"]:
        raise ValueError("complete_linear_v2 persistent cache input seeds must be a non-empty list")

    for seed_index, seed in enumerate(input_record["seeds"]):
        if not isinstance(seed, Mapping):
            raise ValueError(f"complete_linear_v2 persistent cache seed {seed_index} must be a mapping")
        required_seed_fields = {"seed_id", "support_component", "metadata", "coefficients"}
        if set(seed) != required_seed_fields:
            raise ValueError(
                f"complete_linear_v2 persistent cache seed {seed_index} fields must be "
                f"{sorted(required_seed_fields)}"
            )
        if not isinstance(seed.get("seed_id"), str) or not seed["seed_id"]:
            raise ValueError(f"complete_linear_v2 persistent cache seed {seed_index} requires seed_id")
        if not isinstance(seed.get("support_component"), str) or not seed["support_component"]:
            raise ValueError(
                f"complete_linear_v2 persistent cache seed {seed_index} requires support_component"
            )
        if not isinstance(seed.get("metadata"), Mapping):
            raise ValueError(f"complete_linear_v2 persistent cache seed {seed_index} metadata must be a mapping")
        coefficients = seed.get("coefficients")
        if not isinstance(coefficients, Mapping) or not coefficients:
            raise ValueError(
                f"complete_linear_v2 persistent cache seed {seed_index} coefficients must be non-empty"
            )

    action_fields = (
        "exactified_matrices",
        "antiunitary_parities",
        "canonical_k_maps",
        "k_pullbacks",
        "q_permutations",
        "sector_permutations",
    )
    action_keys: set[str] | None = None
    for field_name in action_fields:
        record = identity.get(field_name)
        if not isinstance(record, Mapping) or not record:
            raise ValueError(
                f"complete_linear_v2 persistent cache input identity.{field_name} must be a non-empty mapping"
            )
        keys = {str(key) for key in record}
        if action_keys is None:
            action_keys = keys
        elif keys != action_keys:
            raise ValueError(
                "complete_linear_v2 persistent cache input action metadata keys do not match"
            )

    group_limits = identity.get("group_limits")
    if not isinstance(group_limits, list) or len(group_limits) != len(groups):
        raise ValueError("complete_linear_v2 persistent cache identity.group_limits must match groups")
    factorized_hashes = identity.get("factorized_action_hashes")
    if not isinstance(factorized_hashes, list) or len(factorized_hashes) != len(groups):
        raise ValueError(
            "complete_linear_v2 persistent cache identity.factorized_action_hashes must match groups"
        )
    for group_index, hashes in enumerate(factorized_hashes):
        if not isinstance(hashes, Mapping):
            raise ValueError(
                "complete_linear_v2 persistent cache factorized action hashes "
                f"for group {group_index} must be a mapping"
            )
        for name, value in hashes.items():
            if not str(name) or len(str(value)) != 64:
                raise ValueError(
                    "complete_linear_v2 persistent cache factorized action hash "
                    f"for group {group_index} is invalid"
                )
    joint_route_hashes = identity.get("joint_route_action_hashes")
    if not isinstance(joint_route_hashes, list) or len(joint_route_hashes) != len(groups):
        raise ValueError(
            "complete_linear_v2 persistent cache identity.joint_route_action_hashes "
            "must match groups"
        )
    for group_index, record in enumerate(joint_route_hashes):
        if not isinstance(record, Mapping) or set(record) != {
            "joint_artifact_hash",
            "generator_matrix_hashes",
        }:
            raise ValueError(
                "complete_linear_v2 persistent cache joint route hash record "
                f"for group {group_index} is invalid"
            )
        artifact_hash = record["joint_artifact_hash"]
        generator_hashes = record["generator_matrix_hashes"]
        if not isinstance(generator_hashes, Mapping):
            raise ValueError(
                "complete_linear_v2 persistent cache joint route generator hashes "
                f"for group {group_index} must be a mapping"
            )
        if artifact_hash is None:
            if generator_hashes:
                raise ValueError(
                    "joint route generator hashes require a joint artifact hash"
                )
        elif (
            len(str(artifact_hash)) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(artifact_hash)
            )
            or any(
                not str(name) or len(str(value)) != 64
                for name, value in generator_hashes.items()
            )
        ):
            raise ValueError(
                "complete_linear_v2 persistent cache joint route hashes "
                f"for group {group_index} are invalid"
            )
    expected_action_keys: list[str] = []
    for group_index, (group, internal_matrices, limits) in enumerate(
        zip(groups, group_internal_u, group_limits)
    ):
        if not isinstance(group, Mapping):
            raise ValueError(f"complete_linear_v2 persistent cache group {group_index} must be a mapping")
        required_group_fields = {"max_group_size", "max_word_length", "algebra_residual", "elements"}
        if set(group) != required_group_fields:
            raise ValueError(
                f"complete_linear_v2 persistent cache group {group_index} fields must be "
                f"{sorted(required_group_fields)}"
            )
        elements = group.get("elements")
        if not isinstance(elements, list) or not elements:
            raise ValueError(f"complete_linear_v2 persistent cache group {group_index} elements must be non-empty")
        if not isinstance(internal_matrices, list) or len(internal_matrices) != len(elements):
            raise ValueError(
                f"complete_linear_v2 persistent cache group_internal_u[{group_index}] must match element order"
            )
        if not isinstance(limits, Mapping) or (
            int(limits.get("max_group_size", -1)) != int(group["max_group_size"])
            or int(limits.get("max_word_length", -1)) != int(group["max_word_length"])
        ):
            raise ValueError(
                f"complete_linear_v2 persistent cache group {group_index} limits do not match identity"
            )
        for element_index, (element, internal_u) in enumerate(zip(elements, internal_matrices)):
            action_key = f"group_{group_index}_element_{element_index}"
            expected_action_keys.append(action_key)
            if not isinstance(element, Mapping):
                raise ValueError(f"complete_linear_v2 persistent cache {action_key} must be a mapping")
            reference_u = np.asarray(identity["exactified_matrices"][action_key], dtype=np.complex128)
            candidate_u = np.asarray(internal_u, dtype=np.complex128)
            if (
                reference_u.ndim != 2
                or reference_u.shape[0] != reference_u.shape[1]
                or candidate_u.shape != reference_u.shape
                or not np.all(np.isfinite(reference_u))
                or not np.all(np.isfinite(candidate_u))
            ):
                raise ValueError(
                    f"complete_linear_v2 persistent cache {action_key} internal matrices are invalid"
                )
            phase_residual = _phase_aligned_residual(reference_u, candidate_u)
            phase_tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, np.sqrt(reference_u.size))
            if phase_residual > phase_tolerance:
                raise ValueError(
                    f"complete_linear_v2 persistent cache {action_key} internal U phase-aligned "
                    f"residual {phase_residual:.3e} exceeds {phase_tolerance:.3e}"
                )
            expected_antiunitary = bool(identity["antiunitary_parities"][action_key])
            if not isinstance(element.get("antiunitary"), bool) or element["antiunitary"] != expected_antiunitary:
                raise ValueError(
                    f"complete_linear_v2 persistent cache {action_key} antiunitary parity mismatch"
                )
            discrete_fields = (
                ("canonical_k_map", "canonical_k_maps"),
                ("q_permutation", "q_permutations"),
                ("sector_permutation", "sector_permutations"),
            )
            for element_field, identity_field in discrete_fields:
                element_value = np.asarray(element.get(element_field), dtype=np.int64)
                identity_value = np.asarray(identity[identity_field][action_key], dtype=np.int64)
                if element_value.shape != identity_value.shape or not np.array_equal(element_value, identity_value):
                    raise ValueError(
                        f"complete_linear_v2 persistent cache {action_key} {element_field} mismatch"
                    )
            element_pullback = np.asarray(element.get("k_pullback"), dtype=np.float64)
            identity_pullback = np.asarray(identity["k_pullbacks"][action_key], dtype=np.float64)
            if (
                element_pullback.shape != (2, 2)
                or identity_pullback.shape != (2, 2)
                or not np.all(np.isfinite(element_pullback))
                or not np.all(np.isfinite(identity_pullback))
                or not np.array_equal(element_pullback, identity_pullback)
            ):
                raise ValueError(
                    f"complete_linear_v2 persistent cache {action_key} k_pullback mismatch"
                )
    if action_keys != set(expected_action_keys):
        raise ValueError(
            "complete_linear_v2 persistent cache action metadata keys do not match group element order"
        )


def _response_progress(
    progress_callback: Callable[..., None] | None,
    message: str,
    *,
    state: str | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(message, state=state)


def _exactified_polynomial_origin(config: Any) -> np.ndarray:
    metadata = getattr(config, "symmetry_source_metadata", {})
    if not isinstance(metadata, Mapping) or str(metadata.get("exactification_owner", "")) != "kp_symm":
        raise ValueError(
            "complete_linear_v2 requires kp_symm-owned exactified symmetry metadata; "
            "rerun `kp symm` before compiling the response basis"
        )
    exactification = metadata.get("kp_symm_exactification")
    coordinate = exactification.get("polynomial_coordinate") if isinstance(exactification, Mapping) else None
    if not isinstance(coordinate, Mapping):
        raise ValueError(
            "complete_linear_v2 requires kp_symm_exactification.polynomial_coordinate metadata. "
            "Rerun `kp symm` with the current release before compiling the response basis"
        )
    if str(coordinate.get("coordinate_convention", "")) != COORDINATE_CONVENTION_V1:
        raise ValueError(
            "complete_linear_v2 received an unsupported exactified polynomial coordinate convention: "
            f"{coordinate.get('coordinate_convention')!r}"
        )
    if str(coordinate.get("origin_role", "")) != "exactified_valley_expansion_origin_in_model_cartesian":
        raise ValueError(
            "complete_linear_v2 requires the exactified valley expansion origin in model Cartesian coordinates"
        )
    origin = np.asarray(coordinate.get("origin"), dtype=np.float64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError(
            "kp_symm_exactification.polynomial_coordinate.origin must be a finite Cartesian vector of length 2"
        )
    return origin


def _cached_symmetry_group_key(
    operations: object,
    cache: dict[int, tuple[object, str]],
) -> str:
    """Serialize one shared term-operation object at most once."""

    object_id = id(operations)
    cached = cache.get(object_id)
    if cached is not None and cached[0] is operations:
        return cached[1]
    key = _canonical_json(operations)
    # Retaining the object also prevents an id from being reused during this
    # grouping pass.
    cache[object_id] = (operations, key)
    return key


def _compile_model_response_basis_uncached(
    *,
    coordinate: PolynomialCoordinateBasis,
    groups: Sequence[FiniteGroup],
    seeds_by_group: Sequence[Sequence[RawPolynomialSeed]],
    factorized_actions_by_group: Sequence[Mapping[str, Any] | None],
    joint_route_actions_by_group: Sequence[Mapping[str, Any] | None] | None = None,
    joint_artifact_hashes_by_group: Sequence[str | None] | None = None,
    dim: int,
    identity_payload: Mapping[str, Any],
    reduce: bool,
    cache_key: str,
    progress_callback: Callable[..., None] | None,
) -> CompiledResponseBasis:
    cold_started = time.perf_counter()
    _response_progress(progress_callback, "response basis cold compile start", state="start")
    _RESPONSE_BASIS_COLD_COMPILE_COUNT[cache_key] = _RESPONSE_BASIS_COLD_COMPILE_COUNT.get(cache_key, 0) + 1
    channel_rows: list[ResponseChannel] = []
    support_artifact: dict[str, Any] = {}
    group_artifacts: list[Any] = []
    adjoint_artifacts: list[Mapping[str, Any]] = []
    fixed_groups: list[GeneratorFixedResponseGroup] = []
    candidate_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis candidate compile (p0 fixed-space + finite-p symbolic atoms) start",
        state="start",
    )
    joint_route_groups = list(
        joint_route_actions_by_group
        if joint_route_actions_by_group is not None
        else (None for _ in groups)
    )
    joint_hashes = list(
        joint_artifact_hashes_by_group
        if joint_artifact_hashes_by_group is not None
        else (None for _ in groups)
    )
    if (
        len(factorized_actions_by_group) != len(groups)
        or len(joint_route_groups) != len(groups)
        or len(joint_hashes) != len(groups)
    ):
        raise ValueError("certified action group layout must match finite groups")
    for group_index, (
        seeds,
        group,
        factorized_actions,
        joint_route_actions,
        joint_artifact_hash,
    ) in enumerate(
        zip(
            seeds_by_group,
            groups,
            factorized_actions_by_group,
            joint_route_groups,
            joint_hashes,
        )
    ):
        zero_seeds: list[RawPolynomialSeed] = []
        finite_seeds: list[RawPolynomialSeed] = []
        for seed in seeds:
            term_key = seed.metadata.get("term_key")
            if not isinstance(term_key, Mapping):
                raise ValueError(
                    f"complete response seed {seed.seed_id!r} lacks term_key metadata"
                )
            p_vector = np.asarray(term_key.get("p"), dtype=np.float64)
            if p_vector.shape != (2,) or not np.all(np.isfinite(p_vector)):
                raise ValueError(f"seed {seed.seed_id!r} has invalid explicit harmonic vector")
            (zero_seeds if np.all(p_vector == 0.0) else finite_seeds).append(seed)

        internal_actions: Mapping[tuple[str, ...], sparse.spmatrix] | None = None
        reynolds_artifact: dict[str, Any] = {}
        if finite_seeds or joint_route_actions is not None:
            from .response_basis_factorized import (
                compile_factorized_group_element_actions,
                compile_joint_route_group_element_actions,
            )

            try:
                if factorized_actions is not None:
                    internal_actions, reynolds_artifact = (
                        compile_factorized_group_element_actions(
                            group=group,
                            factorized_generators=factorized_actions,
                        )
                    )
                elif (
                    joint_route_actions is not None
                    and joint_artifact_hash is not None
                ):
                    internal_actions, reynolds_artifact = (
                        compile_joint_route_group_element_actions(
                            group=group,
                            joint_route_generators=joint_route_actions,
                            joint_artifact_hash=joint_artifact_hash,
                        )
                    )
                else:
                    internal_actions, reynolds_artifact = (
                        compile_factorized_group_element_actions(
                            group=group,
                            factorized_generators={},
                        )
                    )
            except Exception as exc:
                exc.add_note(
                    "kp model response-basis compilation failed during certified "
                    f"sparse group actions for group {group_index} "
                    f"({len(finite_seeds)} seeds)"
                )
                raise

        support_masks, support_policy_artifact = _support_masks_from_seeds(
            seeds,
            group=group,
            coordinate=coordinate,
            internal_actions_by_word=internal_actions,
        )

        compiled_groups: list[CandidateResponseSet] = []
        zero_orbit_seeds = [
            seed
            for seed in zero_seeds
            if str(seed.metadata.get("term_space_policy", ""))
            == "orbit_representative"
        ]
        zero_closed_seeds = [
            seed
            for seed in zero_seeds
            if str(seed.metadata.get("term_space_policy", ""))
            != "orbit_representative"
        ]
        zero_joint_complete_seeds = (
            [
                seed
                for seed in zero_closed_seeds
                if str(seed.metadata.get("term_space_policy", "")) == "complete"
            ]
            if joint_route_actions is not None
            else []
        )
        zero_fixed_seeds = (
            [
                seed
                for seed in zero_closed_seeds
                if str(seed.metadata.get("term_space_policy", "")) != "complete"
            ]
            if joint_route_actions is not None
            else list(zero_closed_seeds)
        )
        if zero_fixed_seeds:
            try:
                fixed_group = compile_generator_fixed_response_group(
                    zero_fixed_seeds,
                    joint_keys=_zero_harmonic_joint_adjoint_keys(
                        zero_fixed_seeds,
                        coordinate=coordinate,
                    ),
                    coordinate=coordinate,
                    group=group,
                    reduce=bool(reduce),
                    support_masks=support_masks,
                    materialize_ambient_vectors=False,
                    factorized_actions=factorized_actions,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                exc.add_note(
                    "kp model response-basis compilation failed during p=0 "
                    f"fixed-space compilation for group {group_index} "
                    f"({len(zero_fixed_seeds)} seeds)"
                )
                raise
            fixed_groups.append(fixed_group)
            compiled_groups.append(fixed_group.candidates)
        if zero_joint_complete_seeds:
            assert internal_actions is not None
            try:
                from .response_basis_symmetry_first import (
                    compile_symbolic_atom_candidate_group,
                )

                joint_zero_candidates = compile_symbolic_atom_candidate_group(
                    zero_joint_complete_seeds,
                    coordinate=coordinate,
                    group=group,
                    factorized_actions={},
                    internal_actions_by_word=internal_actions,
                    factorized_group_artifact=reynolds_artifact,
                )
                joint_zero_fixed_group = _fixed_group_from_projected_candidates(
                    joint_zero_candidates,
                    reduce=bool(reduce),
                    solver="joint_route_symbolic_p0_reynolds_v1",
                )
            except Exception as exc:
                exc.add_note(
                    "kp model response-basis compilation failed during certified "
                    f"joint-route p=0 projection for group {group_index} "
                    f"({len(zero_joint_complete_seeds)} seeds)"
                )
                raise
            fixed_groups.append(joint_zero_fixed_group)
            compiled_groups.append(joint_zero_candidates)
        if zero_orbit_seeds:
            try:
                orbit_candidates = compile_candidate_responses(
                    zero_orbit_seeds,
                    coordinate=coordinate,
                    group=group,
                    support_masks=support_masks,
                    internal_actions_by_word=internal_actions,
                    internal_action_absolute_error_bound=float(
                        reynolds_artifact.get("maximum_action_error_bound", 0.0)
                    ),
                )
                orbit_candidates = replace(
                    orbit_candidates,
                    adjoint_artifact={
                        "certification": (
                            "declared_orbit_representative_"
                            "reynolds_hermitian_projection_v1"
                        ),
                        "logical_candidate_channel_count": int(
                            len(orbit_candidates.channels)
                        ),
                        "authored_ordered_seed_count": int(len(zero_orbit_seeds)),
                        "adjoint_orbit_descriptor_count": int(len(zero_orbit_seeds)),
                        "hermitian_ambient_channel_count": int(
                            len(orbit_candidates.channels)
                        ),
                        "physically_compiled_representative_channel_count": int(
                            len(orbit_candidates.channels)
                        ),
                        "adjoint_certified_dropped_channel_count": 0,
                        "fallback_used": False,
                    },
                )
                orbit_fixed_group = _fixed_group_from_projected_candidates(
                    orbit_candidates,
                    reduce=bool(reduce),
                    solver=(
                        "declared_orbit_representative_"
                        "reynolds_hermitian_projection_v1"
                    ),
                )
            except Exception as exc:
                exc.add_note(
                    "kp model response-basis compilation failed during declared "
                    f"p=0 orbit-representative projection for group {group_index} "
                    f"({len(zero_orbit_seeds)} seeds)"
                )
                raise
            fixed_groups.append(orbit_fixed_group)
            compiled_groups.append(orbit_candidates)
        if finite_seeds:
            assert internal_actions is not None
            _response_progress(
                progress_callback,
                "response basis finite-p certified sparse group actions "
                f"certified | elements={len(internal_actions)} "
                f"compiler={reynolds_artifact['compiler']} "
                f"max_nnz={int(reynolds_artifact['maximum_element_nnz'])}",
                state="done",
            )
            try:
                from .response_basis_symmetry_first import (
                    compile_symbolic_atom_candidate_group,
                )

                candidate_group = compile_symbolic_atom_candidate_group(
                    finite_seeds,
                    coordinate=coordinate,
                    group=group,
                    factorized_actions=factorized_actions,
                    internal_actions_by_word=internal_actions,
                    factorized_group_artifact=reynolds_artifact,
                )
            except Exception as exc:
                exc.add_note(
                    "kp model response-basis compilation failed during finite-p "
                    f"symbolic-atom compilation for group {group_index} "
                    f"({len(finite_seeds)} seeds)"
                )
                raise
            symbolic_timings = dict(
                candidate_group.adjoint_artifact.get(
                    "timings_seconds", {}
                )
            )
            selected_ids = {channel.channel_id for channel in candidate_group.channels}
            logical_ids = {
                f"{seed.seed_id}:{component}"
                for seed in finite_seeds
                for component in ("real", "imag")
            }
            symbolic_fixed_group = GeneratorFixedResponseGroup(
                candidates=candidate_group,
                retained_channels=tuple(candidate_group.channels),
                reduction_proofs=(
                    {
                        "solver": "closed_symbolic_atom_reynolds__two_stage_rrqr",
                        "block_rule": "symbolic_atom_support_components",
                        "selected_channel_ids": sorted(selected_ids),
                        "rank_proof_stage_count": int(
                            len(
                                candidate_group.adjoint_artifact.get(
                                    "rank_proofs", ()
                                )
                            )
                        ),
                        "certification_space": (
                            "global_two_dimensional_polynomial_"
                            "coefficient_space_real_imag_stack"
                        ),
                        "sample_grid_used": False,
                    },
                ),
                dropped_channels=tuple(
                    {
                        "channel_id": channel_id,
                        "reason": "target_independent_linear_dependency",
                        "certification": "closed_symbolic_atom_reynolds_v1",
                    }
                    for channel_id in sorted(logical_ids - selected_ids)
                ),
            )
            fixed_groups.append(symbolic_fixed_group)
            _response_progress(
                progress_callback,
                "response basis finite-p symbolic-atom projection done | "
                f"logical={2 * len(finite_seeds)} "
                f"retained={len(candidate_group.channels)} "
                f"atoms={float(symbolic_timings.get('seed_atoms', 0.0)):.2f}s "
                f"project={float(symbolic_timings.get('symbolic_projection', 0.0)):.2f}s "
                f"rank={float(symbolic_timings.get('rank_selection', 0.0)):.2f}s "
                f"materialize={float(symbolic_timings.get('channel_materialization', 0.0)):.2f}s",
                state="done",
            )
            compiled_groups.append(candidate_group)
        for candidate_group in compiled_groups:
            candidate_group = replace(
                candidate_group,
                support_artifact={
                    component: {
                        **dict(record),
                        **(
                            dict(support_policy_artifact.get(component, {}))
                            if int(
                                support_policy_artifact.get(component, {}).get(
                                    "symmetry_adjoint_added_entry_count", 0
                                )
                            )
                            > 0
                            else {}
                        ),
                    }
                    for component, record in candidate_group.support_artifact.items()
                },
            )
            channel_rows.extend(candidate_group.channels)
            support_artifact.update(candidate_group.support_artifact)
            group_artifacts.append(candidate_group.group_artifact)
            adjoint_artifacts.append(candidate_group.adjoint_artifact)
    _response_progress(
        progress_callback,
        "response basis candidate compile (p0 fixed-space + finite-p symbolic atoms) done "
        f"in {time.perf_counter() - candidate_started:.2f} s | "
        f"channels={len(channel_rows)}",
        state="done",
    )
    channel_rows.sort(
        key=lambda channel: (
            int(channel.metadata.get("term_index", -1)),
            0 if channel.component == "real" else 1,
        )
    )
    combined = CandidateResponseSet(
        coordinate=coordinate,
        channels=tuple(channel_rows),
        dim=int(dim),
        group_artifact={"action_groups": group_artifacts},
        support_artifact=support_artifact,
        adjoint_artifact={
            "certification": "raw_polynomial_coefficient_adjoint_v1",
            "groups": adjoint_artifacts,
            "logical_candidate_channel_count": int(
                sum(
                    int(
                        item.get(
                            "logical_candidate_channel_count",
                            item.get(
                                "physically_compiled_representative_channel_count",
                                0,
                            ),
                        )
                    )
                    for item in adjoint_artifacts
                )
            ),
            "authored_ordered_seed_count": int(
                sum(
                    int(item.get("authored_ordered_seed_count", 0))
                    for item in adjoint_artifacts
                )
            ),
            "adjoint_orbit_descriptor_count": int(
                sum(
                    int(item.get("adjoint_orbit_descriptor_count", 0))
                    for item in adjoint_artifacts
                )
            ),
            "hermitian_ambient_channel_count": int(
                sum(
                    int(item.get("hermitian_ambient_channel_count", 0))
                    for item in adjoint_artifacts
                )
            ),
            "physically_materialized_projected_channel_count": int(
                sum(
                    int(
                        item.get(
                            "physically_materialized_projected_channel_count",
                            item.get(
                                "physically_compiled_representative_channel_count", 0
                            ),
                        )
                    )
                    for item in adjoint_artifacts
                )
            ),
            "physically_compiled_representative_channel_count": int(
                sum(
                    int(item.get("physically_compiled_representative_channel_count", 0))
                    for item in adjoint_artifacts
                )
            ),
            "adjoint_certified_dropped_channel_count": int(
                sum(
                    int(item.get("adjoint_certified_dropped_channel_count", 0))
                    for item in adjoint_artifacts
                )
            ),
        },
        null_policy=NullClassificationPolicy(),
    )
    reduce_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis rank reduction (generator fixed-space certified) start",
        state="start",
    )
    basis = _compiled_basis_from_fixed_groups(
        combined,
        fixed_groups,
        identity_payload=identity_payload,
        reduce=bool(reduce),
    )
    _response_progress(
        progress_callback,
        "response basis rank reduction (generator fixed-space certified) done "
        f"in {time.perf_counter() - reduce_started:.2f} s | "
        f"retained={len(basis.channels)} dropped={len(basis.dropped_channels)}",
        state="done",
    )
    _response_progress(
        progress_callback,
        "response basis cold compile done "
        f"in {time.perf_counter() - cold_started:.2f} s",
        state="done",
    )
    return basis


def compile_model_response_basis(
    model: Any,
    config: Any,
    *,
    reduce: bool,
    progress_callback: Callable[..., None] | None = None,
) -> CompiledResponseBasis:
    if str(getattr(config, "response_semantics", LEGACY_FROZEN_V1)) != COMPLETE_LINEAR_V2:
        raise ValueError("compile_model_response_basis requires response_semantics=complete_linear_v2")
    terms = list(
        getattr(model, "candidate_terms", ())
        or getattr(model, "terms", {}).values()
    )
    if not terms:
        raise ValueError("complete_linear_v2 model contains no candidate terms")
    fingerprint_started = time.perf_counter()
    _response_progress(
        progress_callback,
        "response basis input fingerprint start",
        state="start",
    )
    max_degree = max(int(term.key.Mz) + int(term.key.Mz_star) for term in terms)
    origin = _exactified_polynomial_origin(config)
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=origin,
        reciprocal_basis=[np.asarray(config.bM1, dtype=float), np.asarray(config.bM2, dtype=float)],
        max_degree=max_degree,
    )
    symmetry_gen = getattr(model, "_moire_symmetry_gen", getattr(config, "symmetry_gen", None))
    q_count = int(
        sum(
            np.asarray(qset).shape[0]
            for qset, n_orb in (
                (config.Q_set1, int(config.n_orb1)),
                (config.Q_set2, int(config.n_orb2)),
            )
            if n_orb > 0
        )
    )
    dim = int(np.asarray(config.Q_set1).shape[0] * int(config.n_orb1) + np.asarray(config.Q_set2).shape[0] * int(config.n_orb2))
    grouped: dict[str, dict[str, Any]] = {}
    symmetry_group_key_cache: dict[int, tuple[object, str]] = {}
    for term_index, term in enumerate(terms):
        policy = str(getattr(term, "registry_metadata", {}).get("term_space_policy", ""))
        if policy not in {"complete", "explicit_reduced", "orbit_representative"}:
            raise ValueError(
                f"complete_linear_v2 term {term.key!r} lacks an explicit "
                "complete/explicit_reduced/orbit_representative policy"
            )
        authored_sym_ops = getattr(term, "symmetry_ops", [])
        sym_ops = list(authored_sym_ops)
        sym_key = _cached_symmetry_group_key(
            authored_sym_ops,
            symmetry_group_key_cache,
        )
        group_record = grouped.setdefault(sym_key, {"operations": sym_ops, "terms": []})
        term_name = str(getattr(term, "registry_metadata", {}).get("term_name", term.tag))
        support_component = f"{term_name}|{hashlib.sha256(sym_key.encode('utf-8')).hexdigest()[:16]}"
        seed = raw_polynomial_seed_from_term_key(
            term.key,
            seed_id=f"term:{term_index}",
            Q_set1=np.asarray(config.Q_set1),
            Q_set2=np.asarray(config.Q_set2),
            n_orb1=int(config.n_orb1),
            n_orb2=int(config.n_orb2),
            coordinate=coordinate,
            support_component=support_component,
            metadata={
                **dict(getattr(term, "registry_metadata", {})),
                "term_index": int(term_index),
                "tag": str(term.tag),
            },
        )
        group_record["terms"].append(seed)
    groups: list[FiniteGroup] = []
    seeds_by_group: list[list[RawPolynomialSeed]] = []
    factorized_actions_by_group: list[dict[str, Any] | None] = []
    joint_route_actions_by_group: list[dict[str, Any] | None] = []
    joint_artifact_hashes_by_group: list[str | None] = []
    for record in grouped.values():
        operations = record["operations"]
        factorized_actions: dict[str, Any] = {}
        joint_route_actions: dict[str, Any] = {}
        factorized_getter = getattr(symmetry_gen, "get_factorized_action", None)
        joint_route_getter = getattr(symmetry_gen, "get_joint_route_action", None)
        if callable(factorized_getter) or callable(joint_route_getter):
            for operation in operations:
                name = str(operation.get("name", ""))
                if not name:
                    continue
                if callable(factorized_getter):
                    action = factorized_getter(name)
                    if action is not None:
                        factorized_actions[name] = action
                if callable(joint_route_getter):
                    action = joint_route_getter(name)
                    if action is not None:
                        joint_route_actions[name] = action
        required_names = {
            str(operation.get("name", ""))
            for operation in operations
            if str(operation.get("name", ""))
        }
        joint_artifact_hash = getattr(symmetry_gen, "joint_artifact_hash", None)
        use_joint_routes = bool(
            required_names
            and set(joint_route_actions) == required_names
            and isinstance(joint_artifact_hash, str)
            and len(joint_artifact_hash) == 64
        )
        use_factorized = bool(
            not required_names or set(factorized_actions) == required_names
        )
        active_factorized_actions: dict[str, Any] | None = (
            factorized_actions if use_factorized or not use_joint_routes else None
        )
        active_joint_route_actions: dict[str, Any] | None = (
            joint_route_actions if not use_factorized and use_joint_routes else None
        )
        active_joint_artifact_hash = (
            str(joint_artifact_hash)
            if active_joint_route_actions is not None
            else None
        )
        if operations:
            group = finite_group_from_model_actions(
                operations,
                symmetry_gen=symmetry_gen,
                bM1=np.asarray(config.bM1),
                bM2=np.asarray(config.bM2),
                Q_set1=np.asarray(config.Q_set1),
                Q_set2=np.asarray(config.Q_set2),
                sectors=list(getattr(config, "sectors", [])),
                n_orb=(int(config.n_orb1), int(config.n_orb2)),
                factorized_actions=factorized_actions,
            )
        else:
            group = identity_finite_group(dim, q_size=q_count, sector_size=2)
        groups.append(group)
        seeds_by_group.append(list(record["terms"]))
        factorized_actions_by_group.append(active_factorized_actions)
        joint_route_actions_by_group.append(active_joint_route_actions)
        joint_artifact_hashes_by_group.append(active_joint_artifact_hash)
    identity_payload = _basis_layout_identity(
        config,
        groups,
        coordinate,
        factorized_actions_by_group,
        joint_route_actions_by_group,
        joint_artifact_hashes_by_group,
    )
    input_record = {
        "identity": identity_payload,
        "coordinate": coordinate.metadata(),
        "reduce": bool(reduce),
        "groups": [group.artifact() for group in groups],
        "group_internal_u": [
            [np.asarray(element.internal_u) for element in group.elements]
            for group in groups
        ],
        "seeds": [
            {
                "seed_id": seed.seed_id,
                "support_component": seed.support_component,
                "metadata": seed.metadata,
                "coefficients": seed.coefficients,
            }
            for seeds in seeds_by_group
            for seed in seeds
        ],
    }
    _validate_persistent_cache_input_record(input_record)
    cache_key = hashlib.sha256(_canonical_json(input_record).encode("utf-8")).hexdigest()
    _response_progress(
        progress_callback,
        "response basis input fingerprint done "
        f"in {time.perf_counter() - fingerprint_started:.2f} s | terms={len(terms)}",
        state="done",
    )
    cached = _RESPONSE_BASIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    persistent_cache = None
    output_dir = getattr(config, "output_dir", None)
    if output_dir is not None:
        from .response_basis_cache import PersistentResponseBasisCache

        persistent_cache = PersistentResponseBasisCache(
            Path(output_dir) / ".compiled_response_basis_cache"
        )
        lookup_started = time.perf_counter()
        _response_progress(
            progress_callback,
            "response basis persistent cache lookup start",
            state="start",
        )
        cache_path = persistent_cache.path_for(input_record)
        if cache_path.is_file():
            _response_progress(
                progress_callback,
                "response basis persistent cache lookup hit "
                f"in {time.perf_counter() - lookup_started:.2f} s",
                state="done",
            )
            load_started = time.perf_counter()
            basis = persistent_cache.load(input_record)
            _RESPONSE_BASIS_CACHE[cache_key] = basis
            _response_progress(
                progress_callback,
                "response basis persistent cache load done "
                f"in {time.perf_counter() - load_started:.2f} s | "
                f"channels={len(basis.channels)}",
                state="done",
            )
            return basis
        _response_progress(
            progress_callback,
            "response basis persistent cache miss "
            f"in {time.perf_counter() - lookup_started:.2f} s",
            state="done",
        )

    compiled_here = False

    def cold_compile() -> CompiledResponseBasis:
        nonlocal compiled_here
        compiled_here = True
        return _compile_model_response_basis_uncached(
            coordinate=coordinate,
            groups=groups,
            seeds_by_group=seeds_by_group,
            factorized_actions_by_group=factorized_actions_by_group,
            joint_route_actions_by_group=joint_route_actions_by_group,
            joint_artifact_hashes_by_group=joint_artifact_hashes_by_group,
            dim=dim,
            identity_payload=identity_payload,
            reduce=bool(reduce),
            cache_key=cache_key,
            progress_callback=progress_callback,
        )

    if persistent_cache is not None:
        cache_started = time.perf_counter()
        basis = persistent_cache.get_or_compile(input_record, cold_compile)
        action = "cold compile+store total" if compiled_here else "load done after lock"
        _response_progress(
            progress_callback,
            f"response basis persistent cache {action} "
            f"in {time.perf_counter() - cache_started:.2f} s | channels={len(basis.channels)}",
            state="done",
        )
    else:
        basis = cold_compile()
    _RESPONSE_BASIS_CACHE[cache_key] = basis
    return basis
