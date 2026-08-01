"""Certified row layout and routed-projector primitives for Gamma projection.

The v1 implementation is intentionally narrow.  It supports exactly two TAPW
source groups, equal ordered Q counts, a uniform physical-layer orbital width,
and the complete spinful basis.  Unsupported inputs are rejected before any row
indexing is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse

from ..projection_selection import CandidateRejected, CandidateRejectionReason

class GammaCertificationError(CandidateRejected):
    """Typed automatic-Gamma rejection using the public stable reason enum."""


class GammaLayoutError(GammaCertificationError):
    pass


class GammaRoutingError(GammaCertificationError):
    pass


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _hash_payload(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"kp:gamma-routing:v1\0")
    digest.update(_canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def _hash_array(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    if value.dtype.hasobject:
        raise TypeError("Gamma identities do not accept object arrays")
    descriptor = _canonical_json(
        {"dtype": value.dtype.str, "shape": list(value.shape)}
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"kp:gamma-array:v1\0")
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _freeze_complex_array(array: Any, *, message: str) -> np.ndarray:
    value = np.array(array, dtype=np.complex128, copy=True, order="C")
    if not np.all(np.isfinite(value)):
        raise ValueError(message)
    value.setflags(write=False)
    return value


def _strict_integral(value: Any) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
    )


def _canonical_matrix_hash(matrix: Any) -> str:
    """Hash dense and sparse matrices through the same sorted COO encoding."""

    if scipy.sparse.issparse(matrix):
        canonical = _canonical_sparse_csr(matrix).tocoo(copy=False)
        order = np.lexsort((canonical.col, canonical.row))
        rows = np.asarray(canonical.row[order], dtype=np.int64)
        columns = np.asarray(canonical.col[order], dtype=np.int64)
        values = np.ascontiguousarray(canonical.data[order], dtype=np.complex128)
        shape = canonical.shape
    else:
        dense = np.asarray(matrix, dtype=np.complex128)
        rows, columns = np.nonzero(dense)
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        values = np.ascontiguousarray(dense[rows, columns], dtype=np.complex128)
        shape = dense.shape
    digest = hashlib.sha256()
    digest.update(b"kp:gamma-matrix-coo:v1\0")
    digest.update(
        _canonical_json(
            {"dtype": np.dtype(np.complex128).str, "shape": list(shape)}
        ).encode("ascii")
    )
    for value in (rows, columns, values):
        digest.update(b"\0")
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def _canonical_sparse_csr(matrix: Any) -> scipy.sparse.csr_matrix:
    """Return one format-safe mathematical sparse matrix without densifying."""

    try:
        canonical = scipy.sparse.csr_matrix(
            matrix, dtype=np.complex128, copy=True
        )
        canonical.sum_duplicates()
        canonical.sort_indices()
        canonical.eliminate_zeros()
        canonical.sort_indices()
    except (IndexError, OverflowError, TypeError, ValueError) as error:
        raise ValueError("Gamma sparse action cannot be canonicalized") from error
    return canonical


def _matrix_is_finite(matrix: Any) -> bool:
    if scipy.sparse.issparse(matrix):
        return bool(np.all(np.isfinite(matrix.data)))
    return bool(np.all(np.isfinite(np.asarray(matrix))))


def _dense_matrix_block(
    matrix: Any, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    if scipy.sparse.issparse(matrix):
        block = matrix[rows, :][:, columns]
        return np.ascontiguousarray(block.toarray(), dtype=np.complex128)
    return np.ascontiguousarray(
        np.asarray(matrix, dtype=np.complex128)[np.ix_(rows, columns)]
    )


def _matrix_block_frobenius(
    matrix: Any, rows: np.ndarray, columns: np.ndarray
) -> float:
    if rows.size == 0 or columns.size == 0:
        return 0.0
    if scipy.sparse.issparse(matrix):
        block = matrix[rows, :][:, columns]
        return float(np.sqrt(np.vdot(block.data, block.data).real))
    return float(
        np.linalg.norm(
            np.asarray(matrix, dtype=np.complex128)[np.ix_(rows, columns)],
            ord="fro",
        )
    )


@dataclass(frozen=True)
class GammaRoutingThresholds:
    energy_same_ev: float
    energy_different_ev: float
    capture_zero_fraction: float
    capture_loss_max: float
    local_action_isometry: float
    off_route_leakage: float
    closure_residual: float
    route_zero_gap: float
    route_covariance: float
    projector_residual: float
    anchor_sigma_min: float
    max_rank: int
    max_iterations: int

    SCHEMA = "kp.gamma-routing-thresholds.v1"

    def __post_init__(self) -> None:
        scalar_names = (
            "energy_same_ev",
            "energy_different_ev",
            "capture_zero_fraction",
            "capture_loss_max",
            "local_action_isometry",
            "off_route_leakage",
            "closure_residual",
            "route_zero_gap",
            "route_covariance",
            "projector_residual",
            "anchor_sigma_min",
        )
        raw_values = {name: getattr(self, name) for name in scalar_names}
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in raw_values.values()
        ):
            raise ValueError(
                "Gamma routing thresholds must be strict finite numeric values"
            )
        try:
            values = {name: float(value) for name, value in raw_values.items()}
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "Gamma routing thresholds must be strict finite numeric values"
            ) from error
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError(
                "Gamma routing thresholds must be strict finite numeric values"
            )
        if not 0.0 <= values["energy_same_ev"] < values["energy_different_ev"]:
            raise ValueError("Gamma energy thresholds require 0 <= same < different")
        capture_full = 1.0 - values["capture_loss_max"]
        if not 0.0 <= values["capture_zero_fraction"] < capture_full <= 1.0:
            raise ValueError(
                "Gamma capture thresholds require 0 <= zero < 1-loss <= 1"
            )
        if not 0.0 <= values["route_zero_gap"] <= 1.0:
            raise ValueError("Gamma route_zero_gap must lie in [0, 1]")
        if not 0.0 < values["anchor_sigma_min"] <= 1.0:
            raise ValueError("Gamma anchor_sigma_min must lie in (0, 1]")
        nonnegative = scalar_names[4:]
        if any(values[name] < 0.0 for name in nonnegative):
            raise ValueError("Gamma residual thresholds must be non-negative")
        if (
            isinstance(self.max_rank, (bool, np.bool_))
            or isinstance(self.max_iterations, (bool, np.bool_))
            or not isinstance(self.max_rank, Integral)
            or not isinstance(self.max_iterations, Integral)
            or int(self.max_rank) <= 0
            or int(self.max_iterations) <= 0
        ):
            raise ValueError("Gamma rank and iteration limits must be positive integers")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "max_rank", int(self.max_rank))
        object.__setattr__(self, "max_iterations", int(self.max_iterations))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "energy_same_ev": self.energy_same_ev,
            "energy_different_ev": self.energy_different_ev,
            "capture_zero_fraction": self.capture_zero_fraction,
            "capture_loss_max": self.capture_loss_max,
            "local_action_isometry": self.local_action_isometry,
            "off_route_leakage": self.off_route_leakage,
            "closure_residual": self.closure_residual,
            "route_zero_gap": self.route_zero_gap,
            "route_covariance": self.route_covariance,
            "projector_residual": self.projector_residual,
            "anchor_sigma_min": self.anchor_sigma_min,
            "max_rank": self.max_rank,
            "max_iterations": self.max_iterations,
        }

    @property
    def identity_hash(self) -> str:
        return _hash_payload(self.to_payload())

    @classmethod
    def from_normalized_config(cls, config: Mapping[str, Any]) -> "GammaRoutingThresholds":
        if not isinstance(config, Mapping):
            raise ValueError("automatic Gamma routing thresholds must be a mapping")
        expected = {
            "energy_same_ev",
            "energy_different_ev",
            "capture_zero_fraction",
            "capture_loss_max",
            "local_action_isometry",
            "off_route_leakage",
            "closure_residual",
            "route_zero_gap",
            "route_covariance",
            "projector_residual",
            "anchor_sigma_min",
            "max_rank",
            "max_iterations",
        }
        missing = sorted(expected - set(config))
        unknown = sorted(set(config) - expected)
        if missing:
            raise ValueError(
                "automatic Gamma routing thresholds are missing: " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "automatic Gamma routing thresholds contain unknown fields: "
                + ", ".join(unknown)
            )
        return cls(**{name: config[name] for name in expected})


@dataclass(frozen=True)
class GammaRowAddress:
    same_q_local_row: int
    full_row: int
    q_index: int
    source_group: int
    physical_layer: int
    layer_in_group: int
    spin_index: int
    spin_label: str
    orbital: int


@dataclass(frozen=True)
class GammaRowLayout:
    ordered_qsets: tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]
    ordered_qset_hashes: tuple[str, str]
    num_layer_list: tuple[int, int]
    uniform_orbital_count: int
    spin_scope: str
    spin_labels: tuple[str, ...]
    tapw_source_basis_hash: str
    rows_by_q: tuple[tuple[GammaRowAddress, ...], ...]
    addresses_by_full_row: tuple[GammaRowAddress, ...]
    layout_hash: str

    SCHEMA = "kp.gamma-row-layout.v1"

    @classmethod
    def build(
        cls,
        *,
        qsets: Sequence[Any],
        num_layer_list: Sequence[int],
        num_orb_per_layer_list: Sequence[Sequence[int]],
        spin_convention: str,
        source_basis_hash: str,
    ) -> "GammaRowLayout":
        if len(qsets) != 2 or len(num_layer_list) != 2:
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "automatic Gamma v1 requires exactly two source groups",
            )
        if any(not _strict_integral(value) for value in num_layer_list):
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "source-group layer counts must be strict integers",
            )
        layers = tuple(int(value) for value in num_layer_list)
        if any(value <= 0 for value in layers):
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "source-group layer counts must be positive",
            )
        q_arrays = tuple(np.asarray(qset, dtype=np.float64) for qset in qsets)
        if any(array.ndim != 2 or array.shape[0] <= 0 for array in q_arrays):
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "ordered Q sets must be nonempty two-dimensional arrays",
            )
        if q_arrays[0].shape[0] != q_arrays[1].shape[0]:
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "automatic Gamma v1 requires equal ordered Q counts",
            )
        if q_arrays[0].shape[1] != q_arrays[1].shape[1]:
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "ordered Q sets must use the same coordinate dimensions",
            )
        if any(not np.all(np.isfinite(array)) for array in q_arrays):
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "ordered Q sets must be finite",
            )
        if len(num_orb_per_layer_list) != 2 or any(
            len(num_orb_per_layer_list[group]) != layers[group]
            for group in range(2)
        ):
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "orbital metadata must have one width per physical layer",
            )
        raw_widths = tuple(
            width for group in num_orb_per_layer_list for width in group
        )
        if any(not _strict_integral(width) for width in raw_widths):
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "per-physical-layer orbital widths must be strict integers",
            )
        widths = tuple(int(width) for width in raw_widths)
        if not widths or any(width <= 0 for width in widths) or len(set(widths)) != 1:
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_GAMMA_LAYOUT,
                "automatic Gamma v1 requires equal per-physical-layer orbital width",
            )
        spin_key = str(spin_convention).strip().lower()
        spin_map = {
            "all": ("spinful_all", ("up", "down")),
            "spinful_all": ("spinful_all", ("up", "down")),
        }
        if spin_key not in spin_map:
            raise GammaLayoutError(
                CandidateRejectionReason.UNSUPPORTED_SPIN_ROUTE,
                "automatic Gamma v1 supports spinful_all only",
            )
        spin_scope, spin_labels = spin_map[spin_key]
        if not isinstance(source_basis_hash, str) or not source_basis_hash.strip():
            raise GammaLayoutError(
                CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT,
                "TAPW source basis hash must be a literal nonempty string",
            )
        basis_hash = source_basis_hash.strip()

        q_count = int(q_arrays[0].shape[0])
        orbital_width = widths[0]
        orbitals_by_group = tuple(layer_count * orbital_width for layer_count in layers)
        per_spin_local = sum(orbitals_by_group)
        per_spin_full = q_count * per_spin_local
        rows_by_q: list[tuple[GammaRowAddress, ...]] = []
        all_addresses: list[GammaRowAddress] = []
        physical_offsets = (0, layers[0])
        for q_index in range(q_count):
            local_addresses: list[GammaRowAddress] = []
            for spin_index, spin_label in enumerate(spin_labels):
                for source_group in range(2):
                    group_full_offset = q_count * sum(orbitals_by_group[:source_group])
                    group_local_offset = sum(orbitals_by_group[:source_group])
                    for layer_in_group in range(layers[source_group]):
                        physical_layer = physical_offsets[source_group] + layer_in_group
                        for orbital in range(orbital_width):
                            local_row = (
                                spin_index * per_spin_local
                                + group_local_offset
                                + layer_in_group * orbital_width
                                + orbital
                            )
                            full_row = (
                                spin_index * per_spin_full
                                + group_full_offset
                                + q_index * orbitals_by_group[source_group]
                                + layer_in_group * orbital_width
                                + orbital
                            )
                            address = GammaRowAddress(
                                same_q_local_row=local_row,
                                full_row=full_row,
                                q_index=q_index,
                                source_group=source_group,
                                physical_layer=physical_layer,
                                layer_in_group=layer_in_group,
                                spin_index=spin_index,
                                spin_label=spin_label,
                                orbital=orbital,
                            )
                            local_addresses.append(address)
                            all_addresses.append(address)
            local_addresses.sort(key=lambda address: address.same_q_local_row)
            if [address.same_q_local_row for address in local_addresses] != list(
                range(len(local_addresses))
            ):
                raise GammaLayoutError(
                    CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT,
                    "same-Q local rows are not a bijection",
                )
            rows_by_q.append(tuple(local_addresses))

        full_dimension = len(spin_labels) * per_spin_full
        all_addresses.sort(key=lambda address: address.full_row)
        if [address.full_row for address in all_addresses] != list(range(full_dimension)):
            raise GammaLayoutError(
                CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT,
                "full Gamma rows are not a bijection",
            )
        ordered_qsets = tuple(
            tuple(tuple(float(value) for value in row) for row in array.tolist())
            for array in q_arrays
        )
        q_hashes = tuple(_hash_array(array) for array in q_arrays)
        full_rows_hash = _hash_array(
            np.asarray(
                [[address.full_row for address in addresses] for addresses in rows_by_q],
                dtype=np.int64,
            )
        )
        payload = {
            "schema": cls.SCHEMA,
            "tapw_source_basis_hash": basis_hash,
            "ordered_qset_hashes": list(q_hashes),
            "num_layer_list": list(layers),
            "uniform_orbital_count": orbital_width,
            "source_group_count": 2,
            "spin_scope": spin_scope,
            "semantic_spin_labels": list(spin_labels),
            "full_dim": full_dimension,
            "local_dim": len(spin_labels) * per_spin_local,
            "full_rows_by_q_hash": full_rows_hash,
            "basis_order": "spin->group->q->layer->orbital",
        }
        return cls(
            ordered_qsets=ordered_qsets,  # type: ignore[arg-type]
            ordered_qset_hashes=q_hashes,  # type: ignore[arg-type]
            num_layer_list=layers,  # type: ignore[arg-type]
            uniform_orbital_count=orbital_width,
            spin_scope=spin_scope,
            spin_labels=spin_labels,
            tapw_source_basis_hash=basis_hash,
            rows_by_q=tuple(rows_by_q),
            addresses_by_full_row=tuple(all_addresses),
            layout_hash=_hash_payload(payload),
        )

    @property
    def q_count(self) -> int:
        return len(self.rows_by_q)

    @property
    def same_q_dimension(self) -> int:
        return len(self.rows_by_q[0])

    @property
    def full_dimension(self) -> int:
        return len(self.addresses_by_full_row)

    def same_q_full_rows(self, q_index: int) -> np.ndarray:
        if not _strict_integral(q_index) or int(q_index) < 0 or int(q_index) >= self.q_count:
            raise IndexError(f"Gamma q index {q_index} outside 0..{self.q_count - 1}")
        return np.asarray(
            [address.full_row for address in self.rows_by_q[int(q_index)]],
            dtype=np.intp,
        )

    def address_for_full_row(self, full_row: int) -> GammaRowAddress:
        if (
            not _strict_integral(full_row)
            or int(full_row) < 0
            or int(full_row) >= self.full_dimension
        ):
            raise IndexError(
                f"Gamma full row {full_row} outside 0..{self.full_dimension - 1}"
            )
        return self.addresses_by_full_row[int(full_row)]

    def source_group_local_rows(self, source_group: int) -> np.ndarray:
        if not _strict_integral(source_group) or int(source_group) not in (0, 1):
            raise IndexError("Gamma source group must be 0 or 1")
        return np.asarray(
            [
                address.same_q_local_row
                for address in self.rows_by_q[0]
                if address.source_group == int(source_group)
            ],
            dtype=np.intp,
        )

    def source_group_local_projector(self, source_group: int) -> np.ndarray:
        projector = np.zeros(
            (self.same_q_dimension, self.same_q_dimension), dtype=np.complex128
        )
        rows = self.source_group_local_rows(source_group)
        projector[rows, rows] = 1.0
        return projector

    def physical_layer_local_projector(self, physical_layer: int) -> np.ndarray:
        total_layers = sum(self.num_layer_list)
        if (
            not _strict_integral(physical_layer)
            or int(physical_layer) < 0
            or int(physical_layer) >= total_layers
        ):
            raise IndexError(
                f"Gamma physical layer {physical_layer} outside 0..{total_layers - 1}"
            )
        projector = np.zeros(
            (self.same_q_dimension, self.same_q_dimension), dtype=np.complex128
        )
        rows = [
            address.same_q_local_row
            for address in self.rows_by_q[0]
            if address.physical_layer == int(physical_layer)
        ]
        projector[rows, rows] = 1.0
        return projector

    def to_payload(self) -> dict[str, Any]:
        full_rows_hash = _hash_array(
            np.asarray(
                [self.same_q_full_rows(q) for q in range(self.q_count)],
                dtype=np.int64,
            )
        )
        return {
            "schema": self.SCHEMA,
            "tapw_source_basis_hash": self.tapw_source_basis_hash,
            "ordered_qset_hashes": list(self.ordered_qset_hashes),
            "ordered_qsets": [[list(row) for row in qset] for qset in self.ordered_qsets],
            "num_layer_list": list(self.num_layer_list),
            "uniform_orbital_count": self.uniform_orbital_count,
            "source_group_count": 2,
            "spin_scope": self.spin_scope,
            "semantic_spin_labels": list(self.spin_labels),
            "full_dim": self.full_dimension,
            "local_dim": self.same_q_dimension,
            "full_rows_by_q_hash": full_rows_hash,
            "basis_order": "spin->group->q->layer->orbital",
            "layout_hash": self.layout_hash,
        }


def _computed_layout_hash(layout: GammaRowLayout) -> str:
    q_arrays = tuple(np.asarray(qset, dtype=np.float64) for qset in layout.ordered_qsets)
    q_hashes = tuple(_hash_array(array) for array in q_arrays)
    full_rows_hash = _hash_array(
        np.asarray(
            [
                [address.full_row for address in addresses]
                for addresses in layout.rows_by_q
            ],
            dtype=np.int64,
        )
    )
    payload = {
        "schema": layout.SCHEMA,
        "tapw_source_basis_hash": layout.tapw_source_basis_hash,
        "ordered_qset_hashes": list(q_hashes),
        "num_layer_list": list(layout.num_layer_list),
        "uniform_orbital_count": layout.uniform_orbital_count,
        "source_group_count": 2,
        "spin_scope": layout.spin_scope,
        "semantic_spin_labels": list(layout.spin_labels),
        "full_dim": layout.full_dimension,
        "local_dim": layout.same_q_dimension,
        "full_rows_by_q_hash": full_rows_hash,
        "basis_order": "spin->group->q->layer->orbital",
    }
    return _hash_payload(payload)


def _validate_layout_identity(layout: GammaRowLayout) -> None:
    try:
        canonical = GammaRowLayout.build(
            qsets=tuple(
                np.asarray(qset, dtype=np.float64) for qset in layout.ordered_qsets
            ),
            num_layer_list=layout.num_layer_list,
            num_orb_per_layer_list=tuple(
                tuple(
                    layout.uniform_orbital_count
                    for _layer in range(layout.num_layer_list[group])
                )
                for group in range(2)
            ),
            spin_convention=layout.spin_scope,
            source_basis_hash=layout.tapw_source_basis_hash,
        )
    except (GammaLayoutError, TypeError, ValueError) as error:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "Gamma row layout cannot be reconstructed from its certificate",
        ) from error
    if canonical != layout or _computed_layout_hash(layout) != layout.layout_hash:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "Gamma row layout identity is stale or internally inconsistent",
        )


def _validate_layout_thresholds(
    layout: GammaRowLayout, thresholds: GammaRoutingThresholds
) -> None:
    if thresholds.max_rank > layout.same_q_dimension:
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "materialized max_rank exceeds the validated Gamma local dimension",
        )


@dataclass(frozen=True)
class GammaEnergyCluster:
    band_indices: tuple[int, ...]
    energy_min: float
    energy_max: float
    frame: np.ndarray
    projector: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame",
            _freeze_complex_array(self.frame, message="Gamma cluster frame must be finite"),
        )
        object.__setattr__(
            self,
            "projector",
            _freeze_complex_array(
                self.projector, message="Gamma cluster projector must be finite"
            ),
        )


def _partition_gamma_eigensystem(
    eigenvalues: Any,
    eigenvectors: Any,
    *,
    thresholds: GammaRoutingThresholds,
) -> tuple[
    tuple[GammaEnergyCluster, ...],
    tuple[tuple[int, int, int, float], ...],
]:
    values = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if (
        values.ndim != 1
        or values.size == 0
        or vectors.shape != (values.size, values.size)
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
            "Gamma eigensystem must be square with one eigenvalue per column",
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise GammaRoutingError(
            CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
            "Gamma eigensystem must be finite",
        )
    if np.any(np.diff(values) < 0.0):
        raise GammaRoutingError(
            CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
            "Gamma eigenvalues must be sorted",
        )
    unitary_residual = np.linalg.norm(vectors.conj().T @ vectors - np.eye(values.size), ord="fro")
    if unitary_residual / np.sqrt(max(1, values.size)) > thresholds.projector_residual:
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            f"Gamma eigenvectors are not orthonormal ({unitary_residual:.3e})",
        )
    boundaries = [0]
    gray_boundaries: list[tuple[int, float]] = []
    for index, gap in enumerate(np.diff(values), start=1):
        if float(gap) <= thresholds.energy_same_ev:
            continue
        boundaries.append(index)
        if float(gap) < thresholds.energy_different_ev:
            gray_boundaries.append((index, float(gap)))
    boundaries.append(values.size)
    clusters: list[GammaEnergyCluster] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        frame = np.ascontiguousarray(vectors[:, start:stop])
        clusters.append(
            GammaEnergyCluster(
                band_indices=tuple(range(start, stop)),
                energy_min=float(values[start]),
                energy_max=float(values[stop - 1]),
                frame=frame,
                projector=frame @ frame.conj().T,
            )
        )
    cluster_index_by_band = {
        band: cluster_index
        for cluster_index, cluster in enumerate(clusters)
        for band in cluster.band_indices
    }
    gray_edges = tuple(
        (
            cluster_index_by_band[index - 1],
            cluster_index_by_band[index],
            index,
            gap,
        )
        for index, gap in gray_boundaries
    )
    return tuple(clusters), gray_edges


def _reject_relevant_gray_boundaries(
    gray_edges: Sequence[tuple[int, int, int, float]],
    selected_clusters: set[int],
    *,
    q_index: int | None = None,
) -> None:
    for left_cluster, right_cluster, band_index, gap in gray_edges:
        if (left_cluster in selected_clusters) == (right_cluster in selected_clusters):
            continue
        prefix = "" if q_index is None else f"q={q_index} "
        raise GammaRoutingError(
            CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
            f"{prefix}energy-cluster gray zone at bands "
            f"{band_index - 1}/{band_index}: gap={gap:.6g}",
        )


def cluster_gamma_eigensystem(
    eigenvalues: Any,
    eigenvectors: Any,
    *,
    thresholds: GammaRoutingThresholds,
) -> tuple[GammaEnergyCluster, ...]:
    clusters, gray_edges = _partition_gamma_eigensystem(
        eigenvalues,
        eigenvectors,
        thresholds=thresholds,
    )
    if gray_edges:
        _reject_relevant_gray_boundaries(gray_edges, {gray_edges[0][0]})
    return clusters


@dataclass(frozen=True)
class GammaCertifiedRawAction:
    name: str
    antiunitary: bool
    sector_map: tuple[int, int]
    q_permutation: tuple[int, ...]
    local_actions_by_source_q: tuple[np.ndarray, ...]
    tapw_source_basis_hash: str
    layout_hash: str
    thresholds_hash: str
    full_matrix_hash: str
    action_hash: str

    def __post_init__(self) -> None:
        if type(self.antiunitary) is not bool:
            raise ValueError("certified Gamma antiunitary must be a strict bool")
        frozen = tuple(
            _freeze_complex_array(
                local, message="certified Gamma local actions must be finite"
            )
            for local in self.local_actions_by_source_q
        )
        object.__setattr__(self, "local_actions_by_source_q", frozen)


def _raw_action_payload(
    *,
    name: str,
    antiunitary: bool,
    sector_map: tuple[int, int],
    q_permutation: tuple[int, ...],
    local_actions_by_source_q: Sequence[np.ndarray],
    tapw_source_basis_hash: str,
    layout_hash: str,
    thresholds_hash: str,
    full_matrix_hash: str,
) -> dict[str, Any]:
    return {
        "schema": "kp.gamma-raw-action.v1",
        "name": name,
        "antiunitary": antiunitary,
        "sector_map": list(sector_map),
        "q_permutation": list(q_permutation),
        "layout_hash": layout_hash,
        "tapw_source_basis_hash": tapw_source_basis_hash,
        "matrix_hash": full_matrix_hash,
        "local_action_hashes": [
            _hash_array(local) for local in local_actions_by_source_q
        ],
        "thresholds_hash": thresholds_hash,
    }


def _validate_action_certificate(
    action: GammaCertifiedRawAction,
    *,
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
) -> None:
    expected_q = tuple(range(layout.q_count))
    valid_routes = (
        len(action.q_permutation) == layout.q_count
        and all(_strict_integral(value) for value in action.q_permutation)
        and tuple(sorted(int(value) for value in action.q_permutation)) == expected_q
    )
    valid_sector = (
        len(action.sector_map) == 2
        and all(_strict_integral(value) for value in action.sector_map)
        and tuple(sorted(action.sector_map)) == (0, 1)
    )
    valid_locals = len(action.local_actions_by_source_q) == layout.q_count
    valid_antiunitary = type(action.antiunitary) is bool
    if (
        action.layout_hash != layout.layout_hash
        or action.thresholds_hash != thresholds.identity_hash
    ):
        valid_identity = False
    else:
        valid_identity = (
            action.tapw_source_basis_hash == layout.tapw_source_basis_hash
            and isinstance(action.full_matrix_hash, str)
            and bool(action.full_matrix_hash)
        )
    if not (
        valid_routes
        and valid_sector
        and valid_locals
        and valid_antiunitary
        and valid_identity
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"action {action.name!r} does not match the Gamma layout/threshold certificate",
        )
    for local in action.local_actions_by_source_q:
        value = np.asarray(local, dtype=np.complex128)
        if value.shape != (layout.same_q_dimension, layout.same_q_dimension) or not np.all(
            np.isfinite(value)
        ):
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"action {action.name!r} contains an invalid certified local block",
            )
    expected_hash = _hash_payload(
        _raw_action_payload(
            name=action.name,
            antiunitary=action.antiunitary,
            sector_map=action.sector_map,
            q_permutation=action.q_permutation,
            local_actions_by_source_q=action.local_actions_by_source_q,
            tapw_source_basis_hash=action.tapw_source_basis_hash,
            layout_hash=action.layout_hash,
            thresholds_hash=action.thresholds_hash,
            full_matrix_hash=action.full_matrix_hash,
        )
    )
    if expected_hash != action.action_hash:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"action {action.name!r} has a stale certified identity hash",
        )


def gamma_certified_action_route_contract(
    action: GammaCertifiedRawAction,
    *,
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
    full_action: Any | None = None,
) -> dict[str, Any]:
    """Return the canonical, independently verified Gamma route contract.

    The contract is suitable for ``CandidateOperationInput.route_contract``.
    It binds the factorized Q/sector route as well as the exact full matrix
    consumed by candidate certification.  Supplying ``full_action`` closes the
    handoff boundary: callers cannot pair a valid route certificate with a
    different raw action matrix.
    """

    if not isinstance(action, GammaCertifiedRawAction):
        raise TypeError("Gamma route contract requires a certified raw action")
    _validate_action_certificate(action, layout=layout, thresholds=thresholds)
    if full_action is not None:
        try:
            full_matrix_hash = _canonical_matrix_hash(full_action)
        except (IndexError, OverflowError, TypeError, ValueError) as error:
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"action {action.name!r} full matrix cannot be canonicalized",
            ) from error
        if full_matrix_hash != action.full_matrix_hash:
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"action {action.name!r} certified route does not match its raw full matrix",
            )
    return {
        "schema": "kp.gamma-certified-action-route.v1",
        "name": action.name,
        "antiunitary": action.antiunitary,
        "sector_map": list(action.sector_map),
        "q_permutation": list(action.q_permutation),
        "tapw_source_basis_hash": action.tapw_source_basis_hash,
        "layout_hash": action.layout_hash,
        "thresholds_hash": action.thresholds_hash,
        "full_matrix_hash": action.full_matrix_hash,
        "action_hash": action.action_hash,
    }


def infer_gamma_raw_action_q_permutations(
    *,
    full_action: Any,
    layout: GammaRowLayout,
    sector_map: Sequence[int],
    thresholds: GammaRoutingThresholds,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer the unique factorized Gamma Q route from raw matrix support.

    The manifest-derived ``sector_map`` remains authoritative.  For each
    source-Q/source-group column space, a target Q is admissible only when the
    mapped-sector block is locally isometric and every row outside that block
    is below the explicit routing leakage gate.  No symmetry-operation name or
    conventional TR/C3/C2 route is consulted.
    """

    _validate_layout_identity(layout)
    _validate_layout_thresholds(layout, thresholds)
    if len(sector_map) != 2 or any(
        not _strict_integral(value) for value in sector_map
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma sector_map must contain two strict integer indices",
        )
    sector = tuple(int(value) for value in sector_map)
    if tuple(sorted(sector)) != (0, 1):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma sector_map must be a permutation of two source groups",
        )
    if scipy.sparse.issparse(full_action):
        try:
            action = _canonical_sparse_csr(full_action)
        except ValueError as error:
            raise GammaRoutingError(
                CandidateRejectionReason.LOCAL_ACTION_ISOMETRY,
                "raw-H sparse action cannot be canonicalized safely",
            ) from error
    else:
        try:
            action = np.asarray(full_action, dtype=np.complex128)
        except (OverflowError, TypeError, ValueError) as error:
            raise GammaRoutingError(
                CandidateRejectionReason.LOCAL_ACTION_ISOMETRY,
                "raw-H action cannot be represented on the Gamma row space",
            ) from error
    valid_shape = action.shape == (layout.full_dimension, layout.full_dimension)
    if not valid_shape or not _matrix_is_finite(action):
        raise GammaRoutingError(
            CandidateRejectionReason.LOCAL_ACTION_ISOMETRY,
            "raw-H action must be a finite square matrix on the Gamma full row space",
        )

    all_rows = np.arange(layout.full_dimension, dtype=np.intp)
    routes_by_group: list[tuple[int, ...]] = []
    for source_group in range(2):
        target_group = sector[source_group]
        group_route: list[int] = []
        for source_q in range(layout.q_count):
            source_rows = np.asarray(
                [
                    address.full_row
                    for address in layout.rows_by_q[source_q]
                    if address.source_group == source_group
                ],
                dtype=np.intp,
            )
            admissible: list[int] = []
            for target_q in range(layout.q_count):
                target_rows = np.asarray(
                    [
                        address.full_row
                        for address in layout.rows_by_q[target_q]
                        if address.source_group == target_group
                    ],
                    dtype=np.intp,
                )
                outside = np.setdiff1d(all_rows, target_rows, assume_unique=True)
                leakage = _matrix_block_frobenius(action, outside, source_rows)
                leakage /= np.sqrt(source_rows.size)
                local = _dense_matrix_block(action, target_rows, source_rows)
                isometry = float(
                    np.linalg.norm(
                        local.conj().T @ local - np.eye(source_rows.size),
                        ord="fro",
                    )
                    / np.sqrt(source_rows.size)
                )
                if (
                    leakage <= thresholds.off_route_leakage
                    and isometry <= thresholds.local_action_isometry
                ):
                    admissible.append(target_q)
            if len(admissible) != 1:
                raise GammaRoutingError(
                    CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
                    "raw-H action does not determine a unique mapped-sector Q route "
                    f"for source group {source_group}, q {source_q}: {admissible}",
                )
            group_route.append(admissible[0])
        route = tuple(group_route)
        if tuple(sorted(route)) != tuple(range(layout.q_count)):
            raise GammaRoutingError(
                CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
                f"inferred source-group {source_group} Q route is not a permutation",
            )
        routes_by_group.append(route)
    if routes_by_group[0] != routes_by_group[1]:
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "inferred Gamma Q routes disagree between source groups",
        )
    return routes_by_group[0], routes_by_group[1]


def certify_gamma_raw_action(
    *,
    name: str,
    full_action: Any,
    layout: GammaRowLayout,
    q_permutations: Sequence[Sequence[int]],
    sector_map: Sequence[int],
    antiunitary: bool,
    thresholds: GammaRoutingThresholds,
    tapw_source_basis_hash: str,
) -> GammaCertifiedRawAction:
    _validate_layout_identity(layout)
    _validate_layout_thresholds(layout, thresholds)
    if type(antiunitary) is not bool:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "raw-H action antiunitary metadata must be a strict bool",
        )
    if (
        not isinstance(tapw_source_basis_hash, str)
        or tapw_source_basis_hash != layout.tapw_source_basis_hash
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT,
            "raw-H action TAPW source basis hash does not match Gamma layout",
        )
    if len(q_permutations) != 2:
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma raw action requires one Q permutation per source group",
        )
    if any(
        len(route) != layout.q_count
        or any(not _strict_integral(value) for value in route)
        for route in q_permutations
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma raw action Q routes must contain strict integer indices",
        )
    permutations = tuple(tuple(int(value) for value in route) for route in q_permutations)
    expected_q = tuple(range(layout.q_count))
    if any(tuple(sorted(route)) != expected_q for route in permutations):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma raw action Q routes must be permutations",
        )
    joint_route: list[int] = []
    for source_q in range(layout.q_count):
        targets = {permutations[group][source_q] for group in range(2)}
        if len(targets) != 1:
            raise GammaRoutingError(
                CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
                f"q route mismatch for joint source q {source_q}: {sorted(targets)}",
            )
        joint_route.append(targets.pop())
    if len(sector_map) != 2 or any(not _strict_integral(value) for value in sector_map):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma sector_map must contain two strict integer indices",
        )
    sector = tuple(int(value) for value in sector_map)
    if len(sector) != 2 or tuple(sorted(sector)) != (0, 1):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma sector_map must be a permutation of two source groups",
        )
    if scipy.sparse.issparse(full_action):
        try:
            action = _canonical_sparse_csr(full_action)
        except ValueError as error:
            raise GammaRoutingError(
                CandidateRejectionReason.LOCAL_ACTION_ISOMETRY,
                "raw-H sparse action cannot be canonicalized safely",
            ) from error
    else:
        try:
            action = np.asarray(full_action, dtype=np.complex128)
        except (OverflowError, TypeError, ValueError) as error:
            raise GammaRoutingError(
                CandidateRejectionReason.LOCAL_ACTION_ISOMETRY,
                "raw-H action cannot be represented on the Gamma row space",
            ) from error
    if action.shape != (layout.full_dimension, layout.full_dimension) or not _matrix_is_finite(
        action
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.LOCAL_ACTION_ISOMETRY,
            "raw-H action must be a finite square matrix on the Gamma full row space",
        )
    local_actions: list[np.ndarray] = []
    all_rows = np.arange(layout.full_dimension, dtype=np.intp)
    for source_q, target_q in enumerate(joint_route):
        source_rows = layout.same_q_full_rows(source_q)
        target_rows = layout.same_q_full_rows(target_q)
        outside = np.setdiff1d(all_rows, target_rows, assume_unique=True)
        off_route = _matrix_block_frobenius(action, outside, source_rows)
        off_route /= np.sqrt(layout.same_q_dimension)
        if off_route > thresholds.off_route_leakage:
            raise GammaRoutingError(
                CandidateRejectionReason.RAW_ACTION_ROUTE_LEAKAGE,
                f"raw-H off-route leakage {off_route:.3e} exceeds gate",
            )
        local = _dense_matrix_block(action, target_rows, source_rows)
        isometry = float(
            np.linalg.norm(
                local.conj().T @ local - np.eye(layout.same_q_dimension), ord="fro"
            )
            / np.sqrt(layout.same_q_dimension)
        )
        if isometry > thresholds.local_action_isometry:
            raise GammaRoutingError(
                CandidateRejectionReason.LOCAL_ACTION_ISOMETRY,
                f"raw-H local action isometry residual {isometry:.3e} exceeds gate",
            )
        for source_group in range(2):
            source_local = layout.source_group_local_rows(source_group)
            target_local = layout.source_group_local_rows(sector[source_group])
            outside_group = np.setdiff1d(
                np.arange(layout.same_q_dimension, dtype=np.intp),
                target_local,
                assume_unique=True,
            )
            group_leakage = float(
                np.linalg.norm(local[np.ix_(outside_group, source_local)], ord="fro")
                / np.sqrt(source_local.size)
            )
            if group_leakage > thresholds.off_route_leakage:
                raise GammaRoutingError(
                    CandidateRejectionReason.RAW_ACTION_ROUTE_LEAKAGE,
                    f"raw-H source-group route leakage {group_leakage:.3e} exceeds gate",
                )
        local_actions.append(local)
    frozen_locals = tuple(
        _freeze_complex_array(local, message="Gamma local actions must be finite")
        for local in local_actions
    )
    full_matrix_hash = _canonical_matrix_hash(action)
    fields = dict(
        name=str(name),
        antiunitary=antiunitary,
        sector_map=sector,  # type: ignore[arg-type]
        q_permutation=tuple(joint_route),
        local_actions_by_source_q=frozen_locals,
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
        layout_hash=layout.layout_hash,
        thresholds_hash=thresholds.identity_hash,
        full_matrix_hash=full_matrix_hash,
    )
    return GammaCertifiedRawAction(
        **fields,
        action_hash=_hash_payload(_raw_action_payload(**fields)),
    )


@dataclass(frozen=True)
class GammaClusterClosure:
    band_indices_by_q: tuple[tuple[int, ...], ...]
    rank_by_q: tuple[int, ...]
    projectors_by_q: tuple[np.ndarray, ...]
    closure_residuals: tuple[tuple[str, int, float], ...]
    closure_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projectors_by_q",
            tuple(
                _freeze_complex_array(
                    projector, message="Gamma closure projectors must be finite"
                )
                for projector in self.projectors_by_q
            ),
        )


def close_gamma_projector_clusters(
    eigenvalues_by_q: Sequence[Any],
    eigenvectors_by_q: Sequence[Any],
    *,
    layout: GammaRowLayout,
    seed_band_indices: Sequence[Sequence[int]],
    actions: Sequence[GammaCertifiedRawAction],
    thresholds: GammaRoutingThresholds,
) -> GammaClusterClosure:
    _validate_layout_identity(layout)
    _validate_layout_thresholds(layout, thresholds)
    q_count = layout.q_count
    if (
        len(eigenvalues_by_q) != q_count
        or len(eigenvectors_by_q) != q_count
        or len(seed_band_indices) != q_count
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "Gamma closure requires matching eigenvalue/vector/seed Q counts",
        )
    partitions_by_q = tuple(
        _partition_gamma_eigensystem(values, vectors, thresholds=thresholds)
        for values, vectors in zip(eigenvalues_by_q, eigenvectors_by_q, strict=True)
    )
    clusters_by_q = tuple(partition[0] for partition in partitions_by_q)
    gray_edges_by_q = tuple(partition[1] for partition in partitions_by_q)
    for action in actions:
        _validate_action_certificate(action, layout=layout, thresholds=thresholds)

    joint_bands: set[int] = set()
    for bands in seed_band_indices:
        if any(not _strict_integral(band) for band in bands):
            raise GammaRoutingError(
                CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                "Gamma closure seeds must contain strict integer band indices",
            )
        joint_bands.update(int(band) for band in bands)
    if not joint_bands:
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "Gamma closure seed is empty",
        )

    def materialize_joint() -> list[set[int]]:
        if len(joint_bands) > thresholds.max_rank:
            raise GammaRoutingError(
                CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
                f"joint Gamma closure rank {len(joint_bands)} exceeds configured maximum",
            )
        selected_by_q: list[set[int]] = []
        for q_index, clusters in enumerate(clusters_by_q):
            selected_q: set[int] = set()
            represented: set[int] = set()
            for cluster_index, cluster in enumerate(clusters):
                cluster_bands = set(cluster.band_indices)
                overlap = joint_bands.intersection(cluster_bands)
                if overlap and overlap != cluster_bands:
                    raise GammaRoutingError(
                        CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                        f"joint production bands split q={q_index} cluster {cluster.band_indices}",
                    )
                if overlap:
                    selected_q.add(cluster_index)
                    represented.update(cluster_bands)
            if represented != joint_bands:
                raise GammaRoutingError(
                    CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                    f"joint bands are not represented at q={q_index}",
                )
            _reject_relevant_gray_boundaries(
                gray_edges_by_q[q_index],
                selected_q,
                q_index=q_index,
            )
            selected_by_q.append(selected_q)
        return selected_by_q

    selected = materialize_joint()

    converged = False
    for _iteration in range(thresholds.max_iterations):
        previous_joint = set(joint_bands)
        snapshot = [set(items) for items in selected]
        for action in actions:
            for source_q in range(q_count):
                target_q = action.q_permutation[source_q]
                local_action = action.local_actions_by_source_q[source_q]
                for cluster_index in snapshot[source_q]:
                    source_cluster = clusters_by_q[source_q][cluster_index]
                    rank = len(source_cluster.band_indices)
                    source_projector = source_cluster.projector
                    image = local_action @ (
                        source_projector.conjugate()
                        if action.antiunitary
                        else source_projector
                    ) @ local_action.conj().T
                    captures = [
                        float(np.trace(cluster.projector @ image).real / rank)
                        for cluster in clusters_by_q[target_q]
                    ]
                    capture_numeric_tolerance = thresholds.projector_residual
                    if any(
                        capture < -capture_numeric_tolerance
                        or capture > 1.0 + capture_numeric_tolerance
                        for capture in captures
                    ):
                        raise GammaRoutingError(
                            CandidateRejectionReason.AMBIGUOUS_CLUSTER_CAPTURE,
                            f"action {action.name!r} produced out-of-range captures {captures}",
                        )
                    if abs(sum(captures) - 1.0) > thresholds.capture_loss_max:
                        raise GammaRoutingError(
                            CandidateRejectionReason.AMBIGUOUS_CLUSTER_CAPTURE,
                            f"action {action.name!r} target clusters do not cover "
                            f"the image: {captures}",
                        )
                    full = [
                        index
                        for index, capture in enumerate(captures)
                        if 1.0 - capture <= thresholds.capture_loss_max
                    ]
                    ambiguous = [
                        (index, capture)
                        for index, capture in enumerate(captures)
                        if index not in full
                        and capture > thresholds.capture_zero_fraction
                    ]
                    other_capture = (
                        sum(
                            capture
                            for index, capture in enumerate(captures)
                            if not full or index != full[0]
                        )
                        if len(full) == 1
                        else float("inf")
                    )
                    if (
                        len(full) != 1
                        or ambiguous
                        or other_capture > thresholds.capture_loss_max
                    ):
                        raise GammaRoutingError(
                            CandidateRejectionReason.AMBIGUOUS_CLUSTER_CAPTURE,
                            f"action {action.name!r} q={source_q}->{target_q} "
                            f"has captures {captures}",
                        )
                    target_cluster_index = full[0]
                    joint_bands.update(
                        clusters_by_q[target_q][target_cluster_index].band_indices
                    )
        selected = materialize_joint()
        if joint_bands == previous_joint:
            converged = True
            break
    if not converged:
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "Gamma closure did not converge within max_iterations",
        )

    projectors: list[np.ndarray] = []
    bands_by_q: list[tuple[int, ...]] = []
    for q_index in range(q_count):
        selected_clusters = sorted(selected[q_index])
        bands = tuple(
            band
            for cluster_index in selected_clusters
            for band in clusters_by_q[q_index][cluster_index].band_indices
        )
        bands_by_q.append(tuple(sorted(bands)))
        projector = sum(
            (clusters_by_q[q_index][index].projector for index in selected_clusters),
            start=np.zeros_like(clusters_by_q[q_index][0].projector),
        )
        projectors.append(projector)
    residuals: list[tuple[str, int, float]] = []
    for action in actions:
        for source_q, target_q in enumerate(action.q_permutation):
            rank = len(bands_by_q[source_q])
            if rank == 0:
                continue
            source_projector = projectors[source_q]
            image = action.local_actions_by_source_q[source_q] @ (
                source_projector.conjugate()
                if action.antiunitary
                else source_projector
            ) @ action.local_actions_by_source_q[source_q].conj().T
            residual = float(
                np.linalg.norm((np.eye(image.shape[0]) - projectors[target_q]) @ image, ord="fro")
                / np.sqrt(rank)
            )
            if residual > thresholds.closure_residual:
                raise GammaRoutingError(
                    CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
                    f"action {action.name!r} q={source_q} closure residual {residual:.3e}",
                )
            residuals.append((action.name, source_q, residual))
    payload = {
        "schema": "kp.gamma-cluster-closure.v1",
        "band_indices_by_q": [list(bands) for bands in bands_by_q],
        "projector_hashes": [_hash_array(projector) for projector in projectors],
        "actions": [action.action_hash for action in actions],
        "layout_hash": layout.layout_hash,
        "thresholds_hash": thresholds.identity_hash,
    }
    return GammaClusterClosure(
        band_indices_by_q=tuple(bands_by_q),
        rank_by_q=tuple(len(bands) for bands in bands_by_q),
        projectors_by_q=tuple(projectors),
        closure_residuals=tuple(residuals),
        closure_hash=_hash_payload(payload),
    )


def _residual(matrix: np.ndarray, *, rank: int) -> float:
    return float(np.linalg.norm(matrix, ord="fro") / np.sqrt(max(1, rank)))


def _certify_signed_routing_frames(
    groups: Sequence[np.ndarray],
    *,
    routing_operator: np.ndarray,
    thresholds: GammaRoutingThresholds,
    context: str,
    expected_gap: float | None = None,
) -> float:
    if len(groups) != 2 or any(
        group.ndim != 2
        or group.shape[0] != routing_operator.shape[0]
        or group.shape[1] <= 0
        or not np.all(np.isfinite(group))
        for group in groups
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            f"{context} does not contain two finite routed group frames",
        )
    compressed0 = groups[0].conj().T @ routing_operator @ groups[0]
    compressed1 = groups[1].conj().T @ routing_operator @ groups[1]
    compressed0 = 0.5 * (compressed0 + compressed0.conj().T)
    compressed1 = 0.5 * (compressed1 + compressed1.conj().T)
    cross = _residual(
        groups[0].conj().T @ routing_operator @ groups[1],
        rank=min(groups[0].shape[1], groups[1].shape[1]),
    )
    positive_gap = float(np.min(np.linalg.eigvalsh(compressed0)))
    negative_gap = float(-np.max(np.linalg.eigvalsh(compressed1)))
    actual_gap = min(positive_gap, negative_gap)
    invalid_expected_gap = (
        expected_gap is not None
        and abs(actual_gap - expected_gap) > thresholds.projector_residual
    )
    if (
        cross > thresholds.projector_residual
        or actual_gap <= thresholds.route_zero_gap
        or invalid_expected_gap
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            f"{context} does not preserve certified positive/negative routing",
        )
    return actual_gap


def _canonical_projector_frame(projector: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        return np.zeros((projector.shape[0], 0), dtype=np.complex128)
    _q, _r, pivots = scipy.linalg.qr(projector, mode="economic", pivoting=True)
    anchor = projector[:, np.asarray(pivots[:rank], dtype=np.intp)]
    u, singular_values, vh = np.linalg.svd(anchor, full_matrices=False)
    if singular_values.size != rank or float(np.min(singular_values)) <= 0.0:
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            "deterministic routed reference frame is rank deficient",
        )
    frame = u @ vh
    for column in range(rank):
        pivot = int(np.argmax(np.abs(frame[:, column])))
        value = frame[pivot, column]
        if abs(value) > 0.0:
            frame[:, column] *= np.exp(-1j * np.angle(value))
    return frame


def _align_group_frame(
    frame: np.ndarray,
    reference: np.ndarray,
    *,
    thresholds: GammaRoutingThresholds,
) -> np.ndarray:
    if frame.shape != reference.shape:
        raise GammaRoutingError(
            CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
            f"routed frame/reference shapes differ: {frame.shape} != {reference.shape}",
        )
    overlap = reference.conj().T @ frame
    left, singular_values, right_h = np.linalg.svd(overlap, full_matrices=False)
    sigma_min = float(np.min(singular_values)) if singular_values.size else 0.0
    if sigma_min < thresholds.anchor_sigma_min:
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            f"groupwise anchor rank loss: sigma_min={sigma_min:.3e}",
        )
    return frame @ right_h.conj().T @ left.conj().T


@dataclass(frozen=True)
class GammaRoutedFrames:
    layout_hash: str
    thresholds_hash: str
    joint_band_indices: tuple[int, ...]
    group_dimensions: tuple[int, int]
    group_offsets: tuple[int, int, int]
    local_frames_by_q: tuple[np.ndarray, ...]
    reference_frames_by_q_group: tuple[tuple[np.ndarray, np.ndarray], ...]
    routed_projectors_by_q: tuple[tuple[np.ndarray, np.ndarray], ...]
    route_gaps: tuple[float, ...]
    frame_hash: str
    reference_frame_hash: str

    def __post_init__(self) -> None:
        try:
            raw_route_gaps = tuple(self.route_gaps)
        except TypeError as error:
            raise ValueError(
                "routed Gamma route gaps must be strict finite numeric values"
            ) from error
        if not raw_route_gaps or any(
            isinstance(gap, (bool, np.bool_))
            or not isinstance(gap, Real)
            for gap in raw_route_gaps
        ):
            raise ValueError(
                "routed Gamma route gaps must be strict finite numeric values"
            )
        try:
            normalized_route_gaps = tuple(float(gap) for gap in raw_route_gaps)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "routed Gamma route gaps must be strict finite numeric values"
            ) from error
        if any(not np.isfinite(gap) for gap in normalized_route_gaps):
            raise ValueError(
                "routed Gamma route gaps must be strict finite numeric values"
            )
        object.__setattr__(self, "route_gaps", normalized_route_gaps)
        object.__setattr__(
            self,
            "local_frames_by_q",
            tuple(
                _freeze_complex_array(
                    frame, message="routed Gamma frames must be finite"
                )
                for frame in self.local_frames_by_q
            ),
        )
        object.__setattr__(
            self,
            "reference_frames_by_q_group",
            tuple(
                tuple(
                    _freeze_complex_array(
                        frame, message="routed Gamma reference frames must be finite"
                    )
                    for frame in groups
                )
                for groups in self.reference_frames_by_q_group
            ),
        )
        object.__setattr__(
            self,
            "routed_projectors_by_q",
            tuple(
                tuple(
                    _freeze_complex_array(
                        projector,
                        message="routed Gamma projectors must be finite",
                    )
                    for projector in groups
                )
                for groups in self.routed_projectors_by_q
            ),
        )


def build_gamma_routed_frames(
    eigenvalues_by_q: Sequence[Any],
    eigenvectors_by_q: Sequence[Any],
    *,
    joint_band_indices: Sequence[int],
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
    anchor_frames: Sequence[Any] | None = None,
    require_complete_clusters: bool = True,
) -> GammaRoutedFrames:
    _validate_layout_identity(layout)
    _validate_layout_thresholds(layout, thresholds)
    if len(eigenvalues_by_q) != layout.q_count or len(eigenvectors_by_q) != layout.q_count:
        raise GammaRoutingError(
            CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
            "routed Gamma eigensystems must cover every ordered Q",
        )
    if any(not _strict_integral(value) for value in joint_band_indices):
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "joint Gamma band indices must be strict integers",
        )
    joint = tuple(int(value) for value in joint_band_indices)
    if not joint or len(set(joint)) != len(joint) or len(joint) > thresholds.max_rank:
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "joint Gamma band indices must be unique, nonempty, and within max_rank",
        )
    projectors_by_q: list[tuple[np.ndarray, np.ndarray]] = []
    raw_frames_by_q: list[tuple[np.ndarray, np.ndarray]] = []
    route_gaps: list[float] = []
    group_dimensions: tuple[int, int] | None = None
    routing_operator = (
        layout.source_group_local_projector(0)
        - layout.source_group_local_projector(1)
    )
    for q_index, (raw_values, raw_vectors) in enumerate(
        zip(eigenvalues_by_q, eigenvectors_by_q, strict=True)
    ):
        values = np.asarray(raw_values, dtype=float)
        vectors = np.asarray(raw_vectors, dtype=np.complex128)
        if (
            values.shape != (layout.same_q_dimension,)
            or vectors.shape != (layout.same_q_dimension, layout.same_q_dimension)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(vectors))
            or np.any(np.diff(values) < 0.0)
        ):
            raise GammaRoutingError(
                CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
                f"q={q_index} eigensystem is not finite, sorted, and layout-complete",
            )
        full_orthonormality = _residual(
            vectors.conj().T @ vectors - np.eye(layout.same_q_dimension),
            rank=layout.same_q_dimension,
        )
        if full_orthonormality > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} full Gamma eigenframe is not orthonormal",
            )
        if any(band < 0 or band >= values.size for band in joint):
            raise GammaRoutingError(
                CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                f"joint band index is outside q={q_index} eigensystem",
            )
        if require_complete_clusters:
            clusters, gray_edges = _partition_gamma_eigensystem(
                values,
                vectors,
                thresholds=thresholds,
            )
            selected = set(joint)
            selected_clusters: set[int] = set()
            for cluster_index, cluster in enumerate(clusters):
                overlap = selected.intersection(cluster.band_indices)
                if overlap and overlap != set(cluster.band_indices):
                    raise GammaRoutingError(
                        CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                        f"joint bands split q={q_index} cluster {cluster.band_indices}",
                    )
                if overlap:
                    selected_clusters.add(cluster_index)
            _reject_relevant_gray_boundaries(
                gray_edges,
                selected_clusters,
                q_index=q_index,
            )
        frame = np.ascontiguousarray(vectors[:, np.asarray(joint, dtype=np.intp)])
        orthonormality = _residual(
            frame.conj().T @ frame - np.eye(len(joint)), rank=len(joint)
        )
        if orthonormality > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} selected joint frame is not orthonormal",
            )
        routing_matrix = frame.conj().T @ routing_operator @ frame
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (routing_matrix + routing_matrix.conj().T))
        route_gap = float(np.min(np.abs(eigenvalues)))
        if route_gap <= thresholds.route_zero_gap:
            raise GammaRoutingError(
                CandidateRejectionReason.AMBIGUOUS_SOURCE_GROUP_ROUTING,
                f"q={q_index} source-group route gap {route_gap:.3e} is ambiguous",
            )
        negative = np.flatnonzero(eigenvalues < -thresholds.route_zero_gap)
        positive = np.flatnonzero(eigenvalues > thresholds.route_zero_gap)
        dimensions = (int(positive.size), int(negative.size))
        if not all(dimensions):
            raise GammaRoutingError(
                CandidateRejectionReason.EMPTY_SOURCE_GROUP,
                f"q={q_index} routed group dimensions are {dimensions}",
            )
        if group_dimensions is None:
            group_dimensions = dimensions
        elif dimensions != group_dimensions:
            raise GammaRoutingError(
                CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
                f"q={q_index} routed group dimensions changed {group_dimensions}->{dimensions}",
            )
        frames = (
            np.ascontiguousarray(frame @ eigenvectors[:, positive]),
            np.ascontiguousarray(frame @ eigenvectors[:, negative]),
        )
        projectors = tuple(group_frame @ group_frame.conj().T for group_frame in frames)
        joint_projector = frame @ frame.conj().T
        residuals = (
            _residual(projectors[0] - projectors[0].conj().T, rank=dimensions[0]),
            _residual(projectors[1] - projectors[1].conj().T, rank=dimensions[1]),
            _residual(projectors[0] @ projectors[0] - projectors[0], rank=dimensions[0]),
            _residual(projectors[1] @ projectors[1] - projectors[1], rank=dimensions[1]),
            _residual(projectors[0] @ projectors[1], rank=min(dimensions)),
            _residual(projectors[0] + projectors[1] - joint_projector, rank=sum(dimensions)),
        )
        if max(residuals) > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} routed projector residual {max(residuals):.3e}",
            )
        raw_frames_by_q.append(frames)
        projectors_by_q.append(projectors)  # type: ignore[arg-type]
        route_gaps.append(route_gap)
    assert group_dimensions is not None

    reference_frames: list[tuple[np.ndarray, np.ndarray]] = []
    if anchor_frames is None:
        for projectors in projectors_by_q:
            reference_frames.append(
                (
                    _canonical_projector_frame(projectors[0], group_dimensions[0]),
                    _canonical_projector_frame(projectors[1], group_dimensions[1]),
                )
            )
    else:
        if len(anchor_frames) != layout.q_count:
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_K_COVERAGE,
                "Gamma anchor frames do not cover every q",
            )
        for q_index, raw_anchor in enumerate(anchor_frames):
            anchor = np.asarray(raw_anchor, dtype=np.complex128)
            if (
                anchor.shape != (layout.same_q_dimension, sum(group_dimensions))
                or not np.all(np.isfinite(anchor))
            ):
                raise GammaRoutingError(
                    CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                    f"q={q_index} anchor frame shape {anchor.shape} is invalid",
                )
            groups = (
                np.ascontiguousarray(anchor[:, : group_dimensions[0]]),
                np.ascontiguousarray(anchor[:, group_dimensions[0] :]),
            )
            anchor_residuals = (
                _residual(
                    groups[0].conj().T @ groups[0] - np.eye(group_dimensions[0]),
                    rank=group_dimensions[0],
                ),
                _residual(
                    groups[1].conj().T @ groups[1] - np.eye(group_dimensions[1]),
                    rank=group_dimensions[1],
                ),
                _residual(
                    groups[0].conj().T @ groups[1], rank=min(group_dimensions)
                ),
                _residual(
                    anchor.conj().T @ anchor - np.eye(sum(group_dimensions)),
                    rank=sum(group_dimensions),
                ),
            )
            if max(anchor_residuals) > thresholds.projector_residual:
                raise GammaRoutingError(
                    CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                    f"q={q_index} anchor frame is not an orthonormal group certificate",
                )
            compressed0 = groups[0].conj().T @ routing_operator @ groups[0]
            compressed1 = groups[1].conj().T @ routing_operator @ groups[1]
            compressed0 = 0.5 * (compressed0 + compressed0.conj().T)
            compressed1 = 0.5 * (compressed1 + compressed1.conj().T)
            routing_cross = _residual(
                groups[0].conj().T @ routing_operator @ groups[1],
                rank=min(group_dimensions),
            )
            positive_gap = float(np.min(np.linalg.eigvalsh(compressed0)))
            negative_gap = float(-np.max(np.linalg.eigvalsh(compressed1)))
            if (
                routing_cross > thresholds.projector_residual
                or min(positive_gap, negative_gap) <= thresholds.route_zero_gap
            ):
                raise GammaRoutingError(
                    CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                    f"q={q_index} anchor does not preserve signed source-group routing",
                )
            reference_frames.append(groups)

    aligned_combined: list[np.ndarray] = []
    for q_index, (raw_groups, references) in enumerate(
        zip(raw_frames_by_q, reference_frames, strict=True)
    ):
        aligned_groups = tuple(
            _align_group_frame(group, reference, thresholds=thresholds)
            for group, reference in zip(raw_groups, references, strict=True)
        )
        for group_index, (aligned, projector) in enumerate(
            zip(aligned_groups, projectors_by_q[q_index], strict=True)
        ):
            frame_residual = _residual(
                aligned @ aligned.conj().T - projector,
                rank=group_dimensions[group_index],
            )
            if frame_residual > thresholds.projector_residual:
                raise GammaRoutingError(
                    CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                    f"q={q_index} group={group_index} aligned frame changed projector",
                )
        combined = np.ascontiguousarray(np.column_stack(aligned_groups))
        cross = _residual(
            aligned_groups[0].conj().T @ aligned_groups[1],
            rank=min(group_dimensions),
        )
        if cross > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} aligned routed groups are not orthogonal",
            )
        combined_residual = _residual(
            combined.conj().T @ combined - np.eye(sum(group_dimensions)),
            rank=sum(group_dimensions),
        )
        if combined_residual > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} aligned routed frame is not orthonormal",
            )
        aligned_combined.append(combined)

    reference_tensor = np.stack(
        [np.column_stack(groups) for groups in reference_frames], axis=0
    )
    frame_tensor = np.stack(aligned_combined, axis=0)
    return GammaRoutedFrames(
        layout_hash=layout.layout_hash,
        thresholds_hash=thresholds.identity_hash,
        joint_band_indices=joint,
        group_dimensions=group_dimensions,
        group_offsets=(0, group_dimensions[0], sum(group_dimensions)),
        local_frames_by_q=tuple(aligned_combined),
        reference_frames_by_q_group=tuple(reference_frames),
        routed_projectors_by_q=tuple(projectors_by_q),
        route_gaps=tuple(route_gaps),
        frame_hash=_hash_array(frame_tensor),
        reference_frame_hash=_hash_array(reference_tensor),
    )


def certify_routed_covariance(
    routed_projectors_by_q: Sequence[Sequence[np.ndarray]],
    *,
    layout: GammaRowLayout,
    actions: Sequence[GammaCertifiedRawAction],
    thresholds: GammaRoutingThresholds,
) -> tuple[tuple[str, int, float], ...]:
    _validate_layout_identity(layout)
    _validate_layout_thresholds(layout, thresholds)
    q_count = layout.q_count
    if len(routed_projectors_by_q) != q_count:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_K_COVERAGE,
            "routed projectors do not cover every ordered Q",
        )
    projectors: list[tuple[np.ndarray, np.ndarray]] = []
    ranks_by_q: list[tuple[int, int]] = []
    for q_index, raw_groups in enumerate(routed_projectors_by_q):
        if len(raw_groups) != 2:
            raise GammaRoutingError(
                CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
                f"q={q_index} must contain two routed source-group projectors",
            )
        groups: list[np.ndarray] = []
        ranks: list[int] = []
        for group_index, raw_projector in enumerate(raw_groups):
            projector = np.asarray(raw_projector, dtype=np.complex128)
            if (
                projector.shape
                != (layout.same_q_dimension, layout.same_q_dimension)
                or not np.all(np.isfinite(projector))
            ):
                raise GammaRoutingError(
                    CandidateRejectionReason.SOURCE_GROUP_ROUTE_COVARIANCE,
                    f"q={q_index} group={group_index} routed projector is invalid",
                )
            raw_rank = float(np.trace(projector).real)
            rank = int(round(raw_rank))
            certificate_residual = max(
                abs(raw_rank - rank),
                _residual(projector - projector.conj().T, rank=max(1, rank)),
                _residual(projector @ projector - projector, rank=max(1, rank)),
            )
            if rank <= 0 or certificate_residual > thresholds.projector_residual:
                raise GammaRoutingError(
                    CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                    f"q={q_index} group={group_index} is not a certified projector",
                )
            groups.append(projector)
            ranks.append(rank)
        cross = _residual(groups[0] @ groups[1], rank=min(ranks))
        if cross > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} routed source-group projectors overlap",
            )
        projectors.append((groups[0], groups[1]))
        ranks_by_q.append((ranks[0], ranks[1]))

    residuals: list[tuple[str, int, float]] = []
    for action in actions:
        _validate_action_certificate(action, layout=layout, thresholds=thresholds)
        for source_q, target_q in enumerate(action.q_permutation):
            operation_residual = 0.0
            for source_group in range(2):
                target_group = action.sector_map[source_group]
                source = projectors[source_q][source_group]
                target = projectors[target_q][target_group]
                rank = ranks_by_q[source_q][source_group]
                target_rank = ranks_by_q[target_q][target_group]
                if rank != target_rank:
                    raise GammaRoutingError(
                        CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
                        f"action {action.name!r} maps routed rank {rank} to {target_rank}",
                    )
                image = action.local_actions_by_source_q[source_q] @ (
                    source.conjugate() if action.antiunitary else source
                ) @ action.local_actions_by_source_q[source_q].conj().T
                operation_residual = max(
                    operation_residual,
                    _residual(image - target, rank=rank),
                )
            if operation_residual > thresholds.route_covariance:
                raise GammaRoutingError(
                    CandidateRejectionReason.SOURCE_GROUP_ROUTE_COVARIANCE,
                    f"action {action.name!r} q={source_q} route covariance "
                    f"{operation_residual:.3e}",
                )
            residuals.append((action.name, source_q, operation_residual))
    return tuple(residuals)


def assemble_gamma_routed_projectors(
    routed: GammaRoutedFrames,
    *,
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
    include_high: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    _validate_layout_identity(layout)
    _validate_layout_thresholds(layout, thresholds)
    if routed.layout_hash != layout.layout_hash:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed frame layout hash does not match the canonical Gamma layout",
        )
    if routed.thresholds_hash != thresholds.identity_hash:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed frame threshold identity does not match the assembler thresholds",
        )
    if (
        len(routed.local_frames_by_q) != layout.q_count
        or len(routed.reference_frames_by_q_group) != layout.q_count
        or len(routed.routed_projectors_by_q) != layout.q_count
        or len(routed.route_gaps) != layout.q_count
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_K_COVERAGE,
            "routed frames do not cover every ordered Q",
        )
    try:
        raw_ranks = tuple(routed.group_dimensions)
        raw_offsets = tuple(routed.group_offsets)
        complete_group_slices = all(
            len(groups) == 2
            for groups in routed.reference_frames_by_q_group
        ) and all(
            len(groups) == 2 for groups in routed.routed_projectors_by_q
        )
    except (OverflowError, TypeError) as error:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed group metadata is not structurally valid",
        ) from error
    valid_ranks = (
        len(raw_ranks) == 2
        and all(_strict_integral(rank) and int(rank) > 0 for rank in raw_ranks)
    )
    ranks = tuple(int(rank) for rank in raw_ranks) if valid_ranks else ()
    expected_offsets = (0, ranks[0], sum(ranks)) if valid_ranks else ()
    valid_offsets = (
        len(raw_offsets) == 3
        and all(_strict_integral(offset) for offset in raw_offsets)
        and tuple(int(offset) for offset in raw_offsets) == expected_offsets
    )
    if not (valid_ranks and valid_offsets and complete_group_slices):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed group ranks, offsets, or group slices are invalid",
        )
    if (
        not routed.joint_band_indices
        or any(not _strict_integral(band) for band in routed.joint_band_indices)
        or len(set(routed.joint_band_indices)) != len(routed.joint_band_indices)
        or any(
            int(band) < 0 or int(band) >= layout.same_q_dimension
            for band in routed.joint_band_indices
        )
        or len(routed.joint_band_indices) != sum(ranks)
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed joint-band identity does not match routed group ranks",
        )
    if any(
        gap <= thresholds.route_zero_gap
        or gap > 1.0 + thresholds.projector_residual
        for gap in routed.route_gaps
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed source-group gaps do not satisfy the certified routing gate",
        )
    q_count = layout.q_count
    offsets = tuple(int(offset) for offset in raw_offsets)
    try:
        frame_tensor = np.stack(
            [np.asarray(frame, dtype=np.complex128) for frame in routed.local_frames_by_q],
            axis=0,
        )
        reference_tensor = np.stack(
            [
                np.column_stack(
                    tuple(np.asarray(frame, dtype=np.complex128) for frame in groups)
                )
                for groups in routed.reference_frames_by_q_group
            ],
            axis=0,
        )
    except (TypeError, ValueError) as error:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed frame tensors are not shape-consistent",
        ) from error
    if (
        not np.all(np.isfinite(frame_tensor))
        or not np.all(np.isfinite(reference_tensor))
        or _hash_array(frame_tensor) != routed.frame_hash
        or _hash_array(reference_tensor) != routed.reference_frame_hash
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed frame identity hash is stale or contains non-finite data",
        )
    low_dimension = q_count * sum(ranks)
    u_low = np.zeros((layout.full_dimension, low_dimension), dtype=np.complex128)
    high_frames: list[np.ndarray] = []
    routing_operator = (
        layout.source_group_local_projector(0)
        - layout.source_group_local_projector(1)
    )
    for q_index, raw_frame in enumerate(routed.local_frames_by_q):
        frame = np.asarray(raw_frame, dtype=np.complex128)
        if (
            frame.shape != (layout.same_q_dimension, sum(ranks))
            or not np.all(np.isfinite(frame))
        ):
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"q={q_index} routed frame shape is invalid",
            )
        rows = layout.same_q_full_rows(q_index)
        combined_residual = _residual(
            frame.conj().T @ frame - np.eye(sum(ranks)), rank=sum(ranks)
        )
        if combined_residual > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} routed local frame is not orthonormal",
            )
        reference_groups = routed.reference_frames_by_q_group[q_index]
        local_groups = tuple(
            frame[:, offsets[group] : offsets[group + 1]] for group in range(2)
        )
        reference_arrays = tuple(
            np.asarray(reference, dtype=np.complex128)
            for reference in reference_groups
        )
        for group in range(2):
            start, stop = offsets[group : group + 2]
            group_frame = local_groups[group]
            expected_projector = np.asarray(
                routed.routed_projectors_by_q[q_index][group],
                dtype=np.complex128,
            )
            reference = reference_arrays[group]
            if (
                expected_projector.shape
                != (layout.same_q_dimension, layout.same_q_dimension)
                or reference.shape != (layout.same_q_dimension, ranks[group])
                or not np.all(np.isfinite(expected_projector))
                or not np.all(np.isfinite(reference))
            ):
                raise GammaRoutingError(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    f"q={q_index} group={group} routed certificate shape is invalid",
                )
            projector_certificate = max(
                _residual(
                    expected_projector - expected_projector.conj().T,
                    rank=ranks[group],
                ),
                _residual(
                    expected_projector @ expected_projector - expected_projector,
                    rank=ranks[group],
                ),
                abs(float(np.trace(expected_projector).real) - ranks[group]),
                _residual(
                    reference.conj().T @ reference - np.eye(ranks[group]),
                    rank=ranks[group],
                ),
            )
            if projector_certificate > thresholds.projector_residual:
                raise GammaRoutingError(
                    CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                    f"q={q_index} group={group} routed certificate is not a rank projector/frame",
                )
            group_residual = _residual(
                group_frame @ group_frame.conj().T - expected_projector,
                rank=ranks[group],
            )
            if group_residual > thresholds.projector_residual:
                raise GammaRoutingError(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    f"q={q_index} routed group slice {group} does not match its projector",
                )
            for alpha in range(ranks[group]):
                column = q_count * sum(ranks[:group]) + alpha * q_count + q_index
                u_low[rows, column] = frame[:, start + alpha]
        _certify_signed_routing_frames(
            local_groups,
            routing_operator=routing_operator,
            thresholds=thresholds,
            context=f"q={q_index} routed local frame",
            expected_gap=routed.route_gaps[q_index],
        )
        _certify_signed_routing_frames(
            reference_arrays,
            routing_operator=routing_operator,
            thresholds=thresholds,
            context=f"q={q_index} routed reference frame",
        )
        reference0, reference1 = reference_arrays
        projector0 = np.asarray(
            routed.routed_projectors_by_q[q_index][0], dtype=np.complex128
        )
        projector1 = np.asarray(
            routed.routed_projectors_by_q[q_index][1], dtype=np.complex128
        )
        reference_cross = _residual(
            reference0.conj().T @ reference1, rank=min(ranks)
        )
        projector_cross = _residual(
            projector0 @ projector1, rank=min(ranks)
        )
        if max(reference_cross, projector_cross) > thresholds.projector_residual:
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} routed reference/projector groups are not orthogonal",
            )
        if include_high:
            high_frames.append(scipy.linalg.null_space(frame.conj().T))
    low_orthonormality = _residual(
        u_low.conj().T @ u_low - np.eye(low_dimension), rank=low_dimension
    )
    if low_orthonormality > thresholds.projector_residual:
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            f"assembled routed low frame is not orthonormal ({low_orthonormality:.3e})",
        )
    if not include_high:
        return u_low, None
    high_per_q = layout.same_q_dimension - sum(ranks)
    u_high = np.zeros(
        (layout.full_dimension, q_count * high_per_q), dtype=np.complex128
    )
    for q_index, high in enumerate(high_frames):
        high = np.asarray(high, dtype=np.complex128)
        if (
            high.shape != (layout.same_q_dimension, high_per_q)
            or not np.all(np.isfinite(high))
        ):
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                f"q={q_index} routed high frame has invalid shape or values",
            )
        rows = layout.same_q_full_rows(q_index)
        start = q_index * high_per_q
        u_high[rows, start : start + high_per_q] = high
    high_residual = _residual(
        u_high.conj().T @ u_high - np.eye(u_high.shape[1]), rank=u_high.shape[1]
    )
    cross_residual = _residual(
        u_low.conj().T @ u_high, rank=min(u_low.shape[1], u_high.shape[1])
    )
    completeness_residual = _residual(
        u_low @ u_low.conj().T
        + u_high @ u_high.conj().T
        - np.eye(layout.full_dimension),
        rank=layout.full_dimension,
    )
    if (
        max(high_residual, cross_residual, completeness_residual)
        > thresholds.projector_residual
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.PROJECTOR_FRAME_RANK,
            "assembled routed high frame failed orthonormality/cross/completeness gates",
        )
    return u_low, u_high
