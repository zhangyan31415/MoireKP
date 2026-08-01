"""Certified row layout and routed-projector primitives for Gamma projection.

The v1 implementation is intentionally narrow.  It supports exactly two TAPW
source groups, equal ordered Q counts, a uniform physical-layer orbital width,
and either the complete spinful basis or one explicitly identified spin slice.
Unsupported inputs are rejected before any row indexing is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg

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
        values = {name: float(getattr(self, name)) for name in scalar_names}
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError("Gamma routing thresholds must be finite")
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
        widths = tuple(
            int(width)
            for group in num_orb_per_layer_list
            for width in group
        )
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
        basis_hash = str(source_basis_hash).strip()
        if not basis_hash:
            raise GammaLayoutError(
                CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT,
                "TAPW source basis hash must be nonempty",
            )

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
        if int(q_index) < 0 or int(q_index) >= self.q_count:
            raise IndexError(f"Gamma q index {q_index} outside 0..{self.q_count - 1}")
        return np.asarray(
            [address.full_row for address in self.rows_by_q[int(q_index)]],
            dtype=np.intp,
        )

    def address_for_full_row(self, full_row: int) -> GammaRowAddress:
        if int(full_row) < 0 or int(full_row) >= self.full_dimension:
            raise IndexError(
                f"Gamma full row {full_row} outside 0..{self.full_dimension - 1}"
            )
        return self.addresses_by_full_row[int(full_row)]

    def source_group_local_rows(self, source_group: int) -> np.ndarray:
        if int(source_group) not in (0, 1):
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
        if int(physical_layer) < 0 or int(physical_layer) >= total_layers:
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


@dataclass(frozen=True)
class GammaEnergyCluster:
    band_indices: tuple[int, ...]
    energy_min: float
    energy_max: float
    frame: np.ndarray
    projector: np.ndarray


def cluster_gamma_eigensystem(
    eigenvalues: Any,
    eigenvectors: Any,
    *,
    thresholds: GammaRoutingThresholds,
) -> tuple[GammaEnergyCluster, ...]:
    values = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if values.ndim != 1 or vectors.shape != (values.size, values.size):
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
    for index, gap in enumerate(np.diff(values), start=1):
        if float(gap) <= thresholds.energy_same_ev:
            continue
        if float(gap) >= thresholds.energy_different_ev:
            boundaries.append(index)
            continue
        raise GammaRoutingError(
            CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
            f"energy-cluster gray zone at bands {index - 1}/{index}: gap={gap:.6g}",
        )
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
    return tuple(clusters)


@dataclass(frozen=True)
class GammaCertifiedRawAction:
    name: str
    antiunitary: bool
    sector_map: tuple[int, int]
    q_permutation: tuple[int, ...]
    local_actions_by_source_q: tuple[np.ndarray, ...]
    tapw_source_basis_hash: str
    action_hash: str


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
    if str(tapw_source_basis_hash) != layout.tapw_source_basis_hash:
        raise GammaRoutingError(
            CandidateRejectionReason.INVALID_GAMMA_ROW_LAYOUT,
            "raw-H action TAPW source basis hash does not match Gamma layout",
        )
    if len(q_permutations) != 2:
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma raw action requires one Q permutation per source group",
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
    sector = tuple(int(value) for value in sector_map)
    if len(sector) != 2 or tuple(sorted(sector)) != (0, 1):
        raise GammaRoutingError(
            CandidateRejectionReason.GAMMA_Q_ROUTE_MISMATCH,
            "Gamma sector_map must be a permutation of two source groups",
        )
    action = np.asarray(full_action, dtype=np.complex128)
    if action.shape != (layout.full_dimension, layout.full_dimension) or not np.all(
        np.isfinite(action)
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
        off_route = float(np.linalg.norm(action[np.ix_(outside, source_rows)], ord="fro"))
        off_route /= np.sqrt(layout.same_q_dimension)
        if off_route > thresholds.off_route_leakage:
            raise GammaRoutingError(
                CandidateRejectionReason.RAW_ACTION_ROUTE_LEAKAGE,
                f"raw-H off-route leakage {off_route:.3e} exceeds gate",
            )
        local = np.ascontiguousarray(action[np.ix_(target_rows, source_rows)])
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
    payload = {
        "schema": "kp.gamma-raw-action.v1",
        "name": str(name),
        "antiunitary": bool(antiunitary),
        "sector_map": list(sector),
        "q_permutation": joint_route,
        "layout_hash": layout.layout_hash,
        "tapw_source_basis_hash": layout.tapw_source_basis_hash,
        "matrix_hash": _hash_array(action),
        "thresholds_hash": thresholds.identity_hash,
    }
    return GammaCertifiedRawAction(
        name=str(name),
        antiunitary=bool(antiunitary),
        sector_map=sector,  # type: ignore[arg-type]
        q_permutation=tuple(joint_route),
        local_actions_by_source_q=tuple(local_actions),
        tapw_source_basis_hash=layout.tapw_source_basis_hash,
        action_hash=_hash_payload(payload),
    )


@dataclass(frozen=True)
class GammaClusterClosure:
    band_indices_by_q: tuple[tuple[int, ...], ...]
    rank_by_q: tuple[int, ...]
    projectors_by_q: tuple[np.ndarray, ...]
    closure_residuals: tuple[tuple[str, int, float], ...]
    closure_hash: str


def close_gamma_projector_clusters(
    eigenvalues_by_q: Sequence[Any],
    eigenvectors_by_q: Sequence[Any],
    *,
    seed_band_indices: Sequence[Sequence[int]],
    actions: Sequence[GammaCertifiedRawAction],
    thresholds: GammaRoutingThresholds,
) -> GammaClusterClosure:
    q_count = len(eigenvalues_by_q)
    if len(eigenvectors_by_q) != q_count or len(seed_band_indices) != q_count:
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "Gamma closure requires matching eigenvalue/vector/seed Q counts",
        )
    clusters_by_q = tuple(
        cluster_gamma_eigensystem(values, vectors, thresholds=thresholds)
        for values, vectors in zip(eigenvalues_by_q, eigenvectors_by_q, strict=True)
    )
    selected: list[set[int]] = [set() for _ in range(q_count)]
    for q_index, bands in enumerate(seed_band_indices):
        requested = {int(band) for band in bands}
        for cluster_index, cluster in enumerate(clusters_by_q[q_index]):
            overlap = requested.intersection(cluster.band_indices)
            if overlap and overlap != set(cluster.band_indices):
                raise GammaRoutingError(
                    CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                    f"production seed splits q={q_index} cluster {cluster.band_indices}",
                )
            if overlap:
                selected[q_index].add(cluster_index)
        represented = {
            band
            for cluster_index in selected[q_index]
            for band in clusters_by_q[q_index][cluster_index].band_indices
        }
        if represented != requested:
            raise GammaRoutingError(
                CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                f"seed bands at q={q_index} do not identify complete clusters",
            )
    if not any(selected):
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "Gamma closure seed is empty",
        )

    converged = False
    for _iteration in range(thresholds.max_iterations):
        changed = False
        snapshot = [set(items) for items in selected]
        for action in actions:
            if len(action.local_actions_by_source_q) != q_count:
                raise GammaRoutingError(
                    CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
                    f"action {action.name!r} does not cover every q route",
                )
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
                    capture_numeric_tolerance = max(
                        thresholds.projector_residual,
                        thresholds.local_action_isometry,
                    )
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
                            f"action {action.name!r} target clusters do not cover the image: {captures}",
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
                            f"action {action.name!r} q={source_q}->{target_q} has captures {captures}",
                        )
                    target_cluster_index = full[0]
                    if target_cluster_index not in selected[target_q]:
                        selected[target_q].add(target_cluster_index)
                        changed = True
        rank_total = sum(
            len(clusters_by_q[q][cluster].band_indices)
            for q in range(q_count)
            for cluster in selected[q]
        )
        if rank_total > thresholds.max_rank * q_count:
            raise GammaRoutingError(
                CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
                f"Gamma closure rank {rank_total} exceeds configured maximum",
            )
        if not changed:
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
    reference_q_index: int
    local_frames_by_q: tuple[np.ndarray, ...]
    reference_frames_by_q_group: tuple[tuple[np.ndarray, np.ndarray], ...]
    routed_projectors_by_q: tuple[tuple[np.ndarray, np.ndarray], ...]
    route_gaps: tuple[float, ...]
    frame_hash: str
    reference_frame_hash: str


def build_gamma_routed_frames(
    eigenvalues_by_q: Sequence[Any],
    eigenvectors_by_q: Sequence[Any],
    *,
    joint_band_indices: Sequence[int],
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
    reference_q_index: int,
    anchor_frames: Sequence[Any] | None = None,
    require_complete_clusters: bool = True,
) -> GammaRoutedFrames:
    if len(eigenvalues_by_q) != layout.q_count or len(eigenvectors_by_q) != layout.q_count:
        raise GammaRoutingError(
            CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
            "routed Gamma eigensystems must cover every ordered Q",
        )
    if thresholds.max_rank > layout.same_q_dimension:
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "materialized max_rank exceeds the validated Gamma local dimension",
        )
    joint = tuple(int(value) for value in joint_band_indices)
    if not joint or len(set(joint)) != len(joint) or len(joint) > thresholds.max_rank:
        raise GammaRoutingError(
            CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE,
            "joint Gamma band indices must be unique, nonempty, and within max_rank",
        )
    if int(reference_q_index) < 0 or int(reference_q_index) >= layout.q_count:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_K_COVERAGE,
            "Gamma reference q index is outside the ordered Q range",
        )
    projectors_by_q: list[tuple[np.ndarray, np.ndarray]] = []
    raw_frames_by_q: list[tuple[np.ndarray, np.ndarray]] = []
    route_gaps: list[float] = []
    group_dimensions: tuple[int, int] | None = None
    for q_index, (raw_values, raw_vectors) in enumerate(
        zip(eigenvalues_by_q, eigenvectors_by_q, strict=True)
    ):
        values = np.asarray(raw_values, dtype=float)
        vectors = np.asarray(raw_vectors, dtype=np.complex128)
        if vectors.shape != (layout.same_q_dimension, layout.same_q_dimension):
            raise GammaRoutingError(
                CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
                f"q={q_index} eigensystem dimension does not match Gamma local layout",
            )
        if any(band < 0 or band >= values.size for band in joint):
            raise GammaRoutingError(
                CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                f"joint band index is outside q={q_index} eigensystem",
            )
        if require_complete_clusters:
            clusters = cluster_gamma_eigensystem(values, vectors, thresholds=thresholds)
            selected = set(joint)
            for cluster in clusters:
                overlap = selected.intersection(cluster.band_indices)
                if overlap and overlap != set(cluster.band_indices):
                    raise GammaRoutingError(
                        CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER,
                        f"joint bands split q={q_index} cluster {cluster.band_indices}",
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
        r0 = layout.source_group_local_projector(0)
        r1 = layout.source_group_local_projector(1)
        routing_matrix = frame.conj().T @ (r0 - r1) @ frame
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
            if anchor.shape != (layout.same_q_dimension, sum(group_dimensions)):
                raise GammaRoutingError(
                    CandidateRejectionReason.SOURCE_GROUP_RANK_CHANGE,
                    f"q={q_index} anchor frame shape {anchor.shape} is invalid",
                )
            reference_frames.append(
                (
                    anchor[:, : group_dimensions[0]],
                    anchor[:, group_dimensions[0] :],
                )
            )

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
        reference_q_index=int(reference_q_index),
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
    actions: Sequence[GammaCertifiedRawAction],
    thresholds: GammaRoutingThresholds,
) -> tuple[tuple[str, int, float], ...]:
    q_count = len(routed_projectors_by_q)
    residuals: list[tuple[str, int, float]] = []
    for action in actions:
        if len(action.q_permutation) != q_count:
            raise GammaRoutingError(
                CandidateRejectionReason.SOURCE_GROUP_ROUTE_COVARIANCE,
                f"action {action.name!r} does not cover every routed q",
            )
        for source_q, target_q in enumerate(action.q_permutation):
            operation_residual = 0.0
            for source_group in range(2):
                target_group = action.sector_map[source_group]
                source = np.asarray(
                    routed_projectors_by_q[source_q][source_group], dtype=np.complex128
                )
                target = np.asarray(
                    routed_projectors_by_q[target_q][target_group], dtype=np.complex128
                )
                rank = int(round(float(np.trace(source).real)))
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
                    f"action {action.name!r} q={source_q} route covariance {operation_residual:.3e}",
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
    if len(routed.local_frames_by_q) != layout.q_count:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_K_COVERAGE,
            "routed frames do not cover every ordered Q",
        )
    if routed.group_offsets != (
        0,
        routed.group_dimensions[0],
        sum(routed.group_dimensions),
    ):
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed group offsets do not match group ranks",
        )
    q_count = layout.q_count
    ranks = routed.group_dimensions
    low_dimension = q_count * sum(ranks)
    u_low = np.zeros((layout.full_dimension, low_dimension), dtype=np.complex128)
    high_frames: list[np.ndarray] = []
    for q_index, raw_frame in enumerate(routed.local_frames_by_q):
        frame = np.asarray(raw_frame, dtype=np.complex128)
        if frame.shape != (layout.same_q_dimension, sum(ranks)):
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"q={q_index} routed frame shape is invalid",
            )
        rows = layout.same_q_full_rows(q_index)
        for group in range(2):
            start, stop = routed.group_offsets[group : group + 2]
            group_frame = frame[:, start:stop]
            expected_projector = np.asarray(
                routed.routed_projectors_by_q[q_index][group],
                dtype=np.complex128,
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
        rows = layout.same_q_full_rows(q_index)
        start = q_index * high_per_q
        u_high[rows, start : start + high_per_q] = high
    return u_low, u_high
