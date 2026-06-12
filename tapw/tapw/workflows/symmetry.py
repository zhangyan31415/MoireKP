from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.spatial import cKDTree

from ..symmetry.representations import (
    direct_sum,
    generate_direct_sum_params,
    get_g_vec_perlayer,
    get_any_rot_orb_twostep,
    legacy_single_valley_c3_layer_centers,
    rot,
    rotate_mat,
    spin_reps,
    supports_single_valley_c3,
)
from .band import (
    BandStructureCalculator,
    TAPW_parameters,
    _build_group_orbital_transport_between_groups,
    map_atom_types_by_fractional_symmetry,
    resolve_reference_m_valley_c2_symmetry,
    reference_m_valley_c2_linear_map,
    transform_k_by_cartesian_linear_map,
)


DETAIL_COLUMNS = [
    "valley",
    "operation",
    "spglib_index",
    "R_2d",
    "k_label",
    "residual_H_raw",
    "status",
    "not_supported_reason",
    "g_perm_max_delta",
    "nonzero_reciprocal_shift_count",
    "square_residual",
]

NOT_SUPPORTED_REASONS = (
    "valley_not_closed",
    "q_mapping_missing",
    "atom_mapping_missing",
    "orbital_rotation_unsupported",
    "spin_lift_unsupported",
    "antiunitary_not_order2",
    "orbit_not_closed",
    "S_not_positive_definite",
    "generic_C3_builder_failed",
    "reciprocal_shift_phase_unsupported",
    "seitz_translation_unsupported",
    "reference_operation_not_structure_symmetry",
    "spglib_operation_not_valley_closed",
)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PG_PHASE_CONVENTION = "P_G^atomic=diag(exp(-i DeltaG_group dot r_atom)); P_G^TAPW=g P_G^atomic g^dagger"
PG_EXTRACTED_PHASE_CONVENTION = (
    "source-side P_G extracted from raw transport; "
    "unitary: D_raw=D_g^(0) P_G; antiunitary: D_raw=D_g^(0) P_G^*"
)
RAW_H_ACTION_RULE = "unitary: D_raw H D_raw^dagger; antiunitary: D_raw H^* D_raw^dagger"


def displayed_operation_name(name: str) -> str:
    """Return the user-facing operation symbol without provisional axis names."""
    text = str(name)
    if text == "C2":
        return "C2"
    if text in {"K_C2T", "K_C2T"}:
        return "C2T"
    if text == "M_C2_eta":
        return "C2"
    if text.startswith("C2(axis="):
        return "C2"
    return text


def representation_operation_name(name: str) -> str:
    """Return the compact operation symbol used for saved representation files."""
    text = displayed_operation_name(name)
    if text == "C3z":
        return "C3"
    if text == "C3z^2":
        return "C3^2"
    return text


def _safe_path_component(text: Any) -> str:
    raw = str(text).strip() or "unknown"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+")
    return "".join(ch if ch in allowed else "_" for ch in raw)


def _json_optional_float(value):
    if value is None or (isinstance(value, str) and value == ""):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _json_optional_int(value):
    if value is None or (isinstance(value, str) and value == ""):
        return ""
    return int(value)


def _json_int_vector_dict(value) -> dict[str, list[int]]:
    if value is None:
        return {}
    return {
        str(group): np.asarray(coeffs, dtype=int).reshape(-1).tolist()
        for group, coeffs in dict(value).items()
    }


def _is_saved_minimal_generator(operation: str, already_saved: set[str]) -> bool:
    if operation == "E":
        return False
    if operation == "C3^2":
        return False
    if operation in already_saved:
        return False
    return True


def c2_axis_angle_deg_from_rotation(rotation_cart, atol: float = 1.0e-8):
    """Axis angle in the xy plane for a C2 rotation, modulo 180 degrees."""
    if rotation_cart is None:
        return None
    rotation = np.asarray(rotation_cart, dtype=float)
    if rotation.shape[0] < 2 or rotation.shape[1] < 2:
        return None
    r_2d = rotation[:2, :2]
    eigvals, eigvecs = np.linalg.eig(r_2d)
    target = int(np.argmin(np.abs(eigvals - 1.0)))
    if abs(eigvals[target] - 1.0) > atol:
        return None
    axis = np.real(eigvecs[:, target])
    norm = float(np.linalg.norm(axis))
    if norm < atol:
        return None
    axis = axis / norm
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    if abs(angle - 180.0) < 1.0e-10:
        angle = 0.0
    return angle


def _operation_markdown_label(entry: dict[str, Any]) -> str:
    name = displayed_operation_name(entry.get("operation", "unknown"))
    axis_angle = entry.get("axis_angle_deg")
    if axis_angle is not None and "C2" in name:
        angle = float(axis_angle)
        if abs(angle - round(angle)) < 0.05:
            angle_text = f"{angle:.0f}deg"
        else:
            angle_text = f"{angle:.1f}deg"
        return f"{name} (axis {angle_text})"
    return name


def _minimal_generators_by_valley(operations_summary: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    generators: dict[str, list[str]] = {}
    for valley, entries in operations_summary.items():
        selected: list[str] = []
        has_c3_generator = False
        for entry in entries:
            if not entry.get("supported", False):
                continue
            name = displayed_operation_name(entry.get("operation", "unknown"))
            if name == "E":
                continue
            if name == "C3z^2" and has_c3_generator:
                continue
            if name == "C3z":
                has_c3_generator = True
            if name not in selected:
                selected.append(name)
        generators[valley] = selected
    return generators


@dataclass(frozen=True)
class ValleyContext:
    valley: int
    valley_label: str
    valley_center_cart: np.ndarray
    partner_center_cart: np.ndarray
    group_k_centers: dict[int, np.ndarray]
    group_m_k_centers: dict[int, np.ndarray]
    group_g_vectors: dict[int, np.ndarray]
    moire_reciprocal_basis: np.ndarray
    calculator: BandStructureCalculator | None


def _basis_hash_for_valley_context(structure, valley_ctx: ValleyContext, projected_dim: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"valley={int(valley_ctx.valley)};label={valley_ctx.valley_label};dim={int(projected_dim)}".encode())
    digest.update(f";spin={bool(getattr(structure, 'spin', False))}".encode())
    for group_id in sorted(valley_ctx.group_g_vectors):
        vectors = np.ascontiguousarray(np.round(np.asarray(valley_ctx.group_g_vectors[group_id], dtype=np.float64), decimals=12))
        digest.update(f";group={int(group_id)};shape={vectors.shape}".encode())
        digest.update(vectors.tobytes())
    if hasattr(structure, "df") and structure.df is not None:
        columns = [
            column
            for column in ("twist_group", "atom_type", "species", "orb_name", "orb_num")
            if column in structure.df.columns
        ]
        if columns:
            basis_df = _sorted_structure_df(structure)[columns].copy()
            digest.update(basis_df.to_json(orient="records").encode())
    return digest.hexdigest()


def frobenius_relative_residual(lhs, rhs, denominator=None) -> float:
    if scipy.sparse.issparse(lhs) or scipy.sparse.issparse(rhs) or scipy.sparse.issparse(denominator):
        left = lhs if scipy.sparse.issparse(lhs) else scipy.sparse.csr_matrix(np.asarray(lhs, dtype=np.complex128))
        right = rhs if scipy.sparse.issparse(rhs) else scipy.sparse.csr_matrix(np.asarray(rhs, dtype=np.complex128))
        base = left if denominator is None else (
            denominator if scipy.sparse.issparse(denominator) else scipy.sparse.csr_matrix(np.asarray(denominator, dtype=np.complex128))
        )
        delta = (left - right).tocsr()
        diff_norm = float(np.sqrt(np.sum(np.abs(delta.data) ** 2)))
        base_csr = base.tocsr()
        base_norm = float(np.sqrt(np.sum(np.abs(base_csr.data) ** 2)))
        if base_norm == 0.0:
            return 0.0 if diff_norm == 0.0 else float("inf")
        return diff_norm / base_norm

    lhs = np.asarray(lhs, dtype=np.complex128)
    rhs = np.asarray(rhs, dtype=np.complex128)
    base = lhs if denominator is None else np.asarray(denominator, dtype=np.complex128)
    diff_norm = float(np.linalg.norm(lhs - rhs, ord="fro"))
    base_norm = float(np.linalg.norm(base, ord="fro"))
    if base_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")
    return diff_norm / base_norm


def is_positive_definite(matrix, atol: float = 1.0e-12) -> bool:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.allclose(matrix, matrix.conj().T, atol=atol):
        return False
    eigvals = scipy.linalg.eigvalsh(matrix, check_finite=False)
    if not np.all(np.isfinite(eigvals)):
        return False
    return bool(np.min(eigvals) > float(atol))


def inverse_sqrt_positive_definite(matrix, atol: float = 1.0e-12):
    matrix = np.asarray(matrix, dtype=np.complex128)
    eigvals, eigvecs = scipy.linalg.eigh(matrix, check_finite=False)
    if not np.all(np.isfinite(eigvals)) or float(np.min(eigvals)) <= float(atol):
        raise ValueError("S_not_positive_definite")
    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    return eigvecs @ inv_sqrt @ eigvecs.conj().T


def _candidate_name(base_name: str, antiunitary: bool) -> str:
    if not antiunitary:
        return base_name
    return "T" if base_name == "E" else f"{base_name}T"


def expand_spatial_operation_candidates(operations):
    candidates = []
    for operation in operations:
        base_name = str(operation.get("name", "op"))
        for antiunitary in (False, True):
            candidate = dict(operation)
            candidate["antiunitary"] = antiunitary
            candidate["name"] = _candidate_name(base_name, antiunitary)
            candidates.append(candidate)
    return candidates


def check_valley_closure(k_center, rotation_2d, reciprocal_basis_2d, antiunitary: bool, atol: float = 1.0e-6):
    k_center = np.asarray(k_center, dtype=float)
    rotation_2d = np.asarray(rotation_2d, dtype=float)
    reciprocal_basis_2d = np.asarray(reciprocal_basis_2d, dtype=float)
    if k_center.shape != (2,):
        raise ValueError(f"k_center must have shape (2,), got {k_center.shape}")
    if rotation_2d.shape != (2, 2):
        raise ValueError(f"rotation_2d must have shape (2, 2), got {rotation_2d.shape}")
    if reciprocal_basis_2d.shape != (2, 2):
        raise ValueError(f"reciprocal_basis_2d must have shape (2, 2), got {reciprocal_basis_2d.shape}")

    target = ((-rotation_2d) if antiunitary else rotation_2d) @ k_center
    delta = target - k_center
    coeffs = np.linalg.solve(reciprocal_basis_2d.T, delta)
    rounded = np.rint(coeffs)
    mismatch = reciprocal_basis_2d.T @ (coeffs - rounded)
    if float(np.linalg.norm(mismatch)) <= float(atol):
        return {
            "closed": True,
            "reason": "",
            "reciprocal_shift": rounded.astype(int),
        }
    return {
        "closed": False,
        "reason": "valley_not_closed",
        "reciprocal_shift": None,
    }


def _similarity_apply(transport, matrix, *, conjugate_input: bool = False):
    matrix_work = np.asarray(matrix.conj() if conjugate_input else matrix, dtype=np.complex128)
    if scipy.sparse.issparse(transport):
        transport_csr = transport.tocsr()
        return (transport_csr @ matrix_work) @ transport_csr.conj().T
    transport_dense = np.asarray(transport, dtype=np.complex128)
    return transport_dense @ matrix_work @ transport_dense.conj().T


def symmetrize_unitary_orbit(matrices: Iterable[np.ndarray], transports: Iterable[Any]):
    matrices = [np.asarray(matrix, dtype=np.complex128) for matrix in matrices]
    transports = list(transports)
    if not matrices or len(matrices) != len(transports):
        raise ValueError("matrices and transports must be non-empty and have the same length")

    accum = np.zeros_like(matrices[0], dtype=np.complex128)
    for matrix, transport in zip(matrices, transports):
        accum = accum + _similarity_apply(transport, matrix)
    return accum / float(len(matrices))


def lowdin_transform(hamiltonian, overlap, atol: float = 1.0e-12):
    overlap_inv_sqrt = inverse_sqrt_positive_definite(overlap, atol=atol)
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    return overlap_inv_sqrt @ hamiltonian @ overlap_inv_sqrt


def compute_lowdin_order_residual_unitary(
    h_orbit: Iterable[np.ndarray],
    s_orbit: Iterable[np.ndarray],
    transports: Iterable[Any],
    atol: float = 1.0e-12,
):
    h_orbit = [np.asarray(h, dtype=np.complex128) for h in h_orbit]
    s_orbit = [np.asarray(s, dtype=np.complex128) for s in s_orbit]
    transports = list(transports)

    if not h_orbit or len(h_orbit) != len(s_orbit) or len(h_orbit) != len(transports):
        return {
            "supported": False,
            "not_supported_reason": "orbit_not_closed",
            "residual": None,
        }

    if any(not is_positive_definite(s, atol=atol) for s in s_orbit):
        return {
            "supported": False,
            "not_supported_reason": "S_not_positive_definite",
            "residual": None,
        }

    h_sym = symmetrize_unitary_orbit(h_orbit, transports)
    s_sym = symmetrize_unitary_orbit(s_orbit, transports)
    if not is_positive_definite(s_sym, atol=atol):
        return {
            "supported": False,
            "not_supported_reason": "S_not_positive_definite",
            "residual": None,
        }

    strict_h = lowdin_transform(h_sym, s_sym, atol=atol)
    engineering_terms = [
        lowdin_transform(h, s, atol=atol)
        for h, s in zip(h_orbit, s_orbit)
    ]
    engineering_h = symmetrize_unitary_orbit(engineering_terms, transports)
    residual = frobenius_relative_residual(strict_h, engineering_h, denominator=strict_h)
    return {
        "supported": True,
        "not_supported_reason": "",
        "residual": residual,
    }


def resolve_tapw_valley_context(structure, compute_config, calculator: BandStructureCalculator | None = None) -> ValleyContext:
    cfg = copy.deepcopy(compute_config)
    if calculator is None:
        params = TAPW_parameters(structure, cfg)
    else:
        params = calculator.TAPW_parameters
    k1_calc, k2_calc, m_k1, m_k2, _ = params.calculate_K_points()
    if hasattr(structure, "df") and structure.df is not None and "twist_group" in structure.df.columns:
        group_ids = sorted(structure.df["twist_group"].astype(int).unique().tolist())
    else:
        group_ids = [0, 1]
    group_k_centers = {}
    group_m_k_centers = {}
    group_g_vectors = {}
    k1 = np.asarray(getattr(params, "K1", None) if getattr(params, "K1", None) is not None else k1_calc, dtype=float).reshape(2)
    k2 = np.asarray(getattr(params, "K2", None) if getattr(params, "K2", None) is not None else k2_calc, dtype=float).reshape(2)
    g1 = np.asarray(getattr(params, "g_vec_list_K1", np.zeros((0, 2))), dtype=float)
    g2 = np.asarray(getattr(params, "g_vec_list_K2", np.zeros((0, 2))), dtype=float)
    for group_id in group_ids:
        if group_id % 2 == 0:
            group_k_centers[group_id] = k1
            group_m_k_centers[group_id] = np.asarray(m_k1, dtype=float).reshape(2)
            group_g_vectors[group_id] = g1
        else:
            group_k_centers[group_id] = k2
            group_m_k_centers[group_id] = np.asarray(m_k2, dtype=float).reshape(2)
            group_g_vectors[group_id] = g2

    return ValleyContext(
        valley=int(cfg.valley),
        valley_label=getattr(calculator, "valley_flag", str(cfg.valley)),
        valley_center_cart=np.asarray(m_k1, dtype=float),
        partner_center_cart=np.asarray(m_k2, dtype=float),
        group_k_centers=group_k_centers,
        group_m_k_centers=group_m_k_centers,
        group_g_vectors=group_g_vectors,
        moire_reciprocal_basis=np.vstack(
            [
                np.asarray(getattr(structure, "reciprocal_Tmat", np.eye(3, dtype=float))[0][:2], dtype=float),
                np.asarray(getattr(structure, "reciprocal_Tmat", np.eye(3, dtype=float))[1][:2], dtype=float),
            ]
        ),
        calculator=calculator,
    )


def resolve_tapw_valley_center(structure, compute_config):
    return resolve_tapw_valley_context(structure, compute_config).valley_center_cart


def compare_transport_against_legacy_c3(transport, legacy):
    return frobenius_relative_residual(transport, legacy, denominator=legacy)


def classify_residual(value: float | None, tolerance: float) -> str:
    if value is None or not np.isfinite(value):
        return "not_supported"
    if value <= 1.0e-6:
        return "exact"
    if value <= float(tolerance):
        return "approximate/provisional"
    return "failed"


def _status_rank(status: str) -> int:
    return {
        "exact": 0,
        "approximate/provisional": 1,
        "failed": 2,
        "not_supported": 3,
    }.get(status, 3)


def combine_statuses(statuses: Iterable[str]) -> str:
    statuses = list(statuses)
    if not statuses:
        return "not_supported"
    return max(statuses, key=_status_rank)


class SymmetrySupportError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _rotation_name(rotation_cart: np.ndarray, index: int) -> str:
    rotation_cart = np.asarray(rotation_cart, dtype=float)
    det = float(np.linalg.det(rotation_cart))
    angle = float(np.degrees(np.arctan2(rotation_cart[1, 0], rotation_cart[0, 0])))
    angle = ((angle + 180.0) % 360.0) - 180.0
    if np.allclose(rotation_cart[:2, :2], np.eye(2), atol=1.0e-8):
        return "E"
    if round(det) == 1:
        if abs(angle - 120.0) <= 1.0e-6:
            return "C3z"
        if abs(angle + 120.0) <= 1.0e-6:
            return "C3z^2"
        if abs(abs(angle) - 180.0) <= 1.0e-6:
            return "C2"
    return f"op_{index}"


def _cartesian_positions_to_fractional(lattice: np.ndarray, cartesian_positions: np.ndarray) -> np.ndarray:
    lattice = np.asarray(lattice, dtype=float)
    cartesian_positions = np.asarray(cartesian_positions, dtype=float)
    frac = np.linalg.solve(lattice.T, cartesian_positions.T).T
    return frac - np.floor(frac)


def _round_trip_integer_coeffs(delta: np.ndarray, reciprocal_basis: np.ndarray) -> tuple[np.ndarray, float]:
    basis = np.asarray(reciprocal_basis, dtype=float).reshape(2, 2)
    coeffs = np.linalg.solve(basis.T, np.asarray(delta, dtype=float).reshape(2))
    coeffs_rounded = np.rint(coeffs).astype(int)
    residual = np.linalg.norm(coeffs_rounded @ basis - delta)
    return coeffs_rounded, float(residual)


def translation_removable_by_origin_shift(
    rotation_cart: np.ndarray,
    translation_cart: np.ndarray,
    tol: float = 1.0e-8,
) -> tuple[bool, np.ndarray, float]:
    rotation = np.asarray(rotation_cart, dtype=float).reshape(3, 3)
    translation = np.asarray(translation_cart, dtype=float).reshape(3)
    if np.linalg.norm(translation) < tol:
        return True, np.zeros(3, dtype=float), 0.0
    shift, *_ = np.linalg.lstsq(np.eye(3, dtype=float) - rotation, translation, rcond=None)
    residual = float(np.linalg.norm((np.eye(3, dtype=float) - rotation) @ shift - translation))
    return residual < tol, np.asarray(shift, dtype=float), residual


def _rounded_q_key(q, ndigits: int = 10):
    q = np.asarray(q, dtype=float).reshape(-1)
    return tuple(np.round(q, decimals=ndigits).tolist())


def build_projected_g_transport_by_physical_momentum(
    *,
    source_g_vectors: np.ndarray,
    target_g_vectors: np.ndarray,
    source_k_center: np.ndarray,
    target_k_center: np.ndarray,
    q_source_cart: np.ndarray,
    q_target_cart: np.ndarray,
    effective_linear_map: np.ndarray,
    reciprocal_basis: np.ndarray,
    tol: float = 1.0e-8,
):
    source_g = np.asarray(source_g_vectors, dtype=float)
    target_g = np.asarray(target_g_vectors, dtype=float)
    source_center = np.asarray(source_k_center, dtype=float).reshape(2)
    target_center = np.asarray(target_k_center, dtype=float).reshape(2)
    q_source = np.asarray(q_source_cart, dtype=float).reshape(2)
    q_target = np.asarray(q_target_cart, dtype=float).reshape(2)
    effective = np.asarray(effective_linear_map, dtype=float).reshape(2, 2)
    if source_g.shape != target_g.shape:
        raise ValueError(f"source and target G lists must have the same shape, got {source_g.shape} and {target_g.shape}")

    permutation = np.full(source_g.shape[0], -1, dtype=int)
    reciprocal_shifts = np.zeros((source_g.shape[0], 2), dtype=int)
    used_targets = set()
    for source_index, g_source in enumerate(source_g):
        # TAPW g_vec_list_* already stores the valley-shifted projected basis momenta.
        # The physical momentum represented by a basis row is therefore q + g_vec.
        p_source = q_source + np.asarray(g_source, dtype=float)
        p_target = effective @ p_source
        best: tuple[float, int, np.ndarray] | None = None
        for target_index, g_target in enumerate(target_g):
            if target_index in used_targets:
                continue
            delta = p_target - (q_target + np.asarray(g_target, dtype=float))
            coeffs, residual = _round_trip_integer_coeffs(delta, reciprocal_basis)
            if best is None or residual < best[0]:
                best = (residual, int(target_index), coeffs)
        if best is None or best[0] > tol:
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"Projected G-space momentum match failed for source index {source_index} with residual>{tol:.1e}.",
            )
        residual, target_index, coeffs = best
        used_targets.add(target_index)
        permutation[source_index] = target_index
        reciprocal_shifts[source_index] = coeffs

    return permutation, reciprocal_shifts


def build_c3_g_transport_from_tapw_convention(
    *,
    g_source: np.ndarray,
    g_target: np.ndarray,
    layer_center: np.ndarray,
    angle_deg: float,
    tol: float = 1.0e-8,
):
    g_source = np.asarray(g_source, dtype=float)
    g_target = np.asarray(g_target, dtype=float)
    layer_center = np.asarray(layer_center, dtype=float).reshape(2)
    if g_source.shape != g_target.shape:
        raise ValueError(f"source and target g lists must have the same shape, got {g_source.shape} and {g_target.shape}")

    theta = np.deg2rad(float(angle_deg))
    rot2d = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    rows = []
    cols = []
    data = []
    used_targets = set()
    for source_index, g_vec in enumerate(g_source):
        transformed = rot2d @ (np.asarray(g_vec, dtype=float) - layer_center) + layer_center
        deltas = np.linalg.norm(g_target - transformed, axis=1)
        target_index = int(np.argmin(deltas))
        if float(deltas[target_index]) > tol:
            raise SymmetrySupportError(
                "generic_C3_builder_failed",
                f"Generic C3 G-space match failed for source index {source_index}: min delta={deltas[target_index]:.3e}.",
            )
        if target_index in used_targets:
            raise SymmetrySupportError(
                "generic_C3_builder_failed",
                f"Generic C3 G-space match is not one-to-one; repeated target index {target_index}.",
            )
        used_targets.add(target_index)
        rows.append(target_index)
        cols.append(source_index)
        data.append(1.0 + 0.0j)

    return scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(g_target.shape[0], g_source.shape[0]),
        dtype=np.complex128,
    )


def build_time_reversal_g_transport(
    *,
    g_vectors: np.ndarray,
    tol: float = 1.0e-8,
):
    g_arr = np.asarray(g_vectors, dtype=float)
    rows = []
    cols = []
    data = []
    used_targets = set()
    max_delta = 0.0
    for source_index, g_source in enumerate(g_arr):
        deltas = np.linalg.norm(g_arr + np.asarray(g_source, dtype=float), axis=1)
        target_index = int(np.argmin(deltas))
        max_delta = max(max_delta, float(deltas[target_index]))
        if float(deltas[target_index]) > tol:
            raise SymmetrySupportError(
                "reciprocal_shift_phase_unsupported",
                f"Time-reversal direct G permutation requires nonzero reciprocal shifts; source index {source_index} best delta={deltas[target_index]:.3e}.",
            )
        if target_index in used_targets:
            raise SymmetrySupportError(
                "reciprocal_shift_phase_unsupported",
                f"Time-reversal direct G permutation is not one-to-one; repeated target index {target_index}.",
            )
        used_targets.add(target_index)
        rows.append(target_index)
        cols.append(source_index)
        data.append(1.0 + 0.0j)
    transport = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(g_arr.shape[0], g_arr.shape[0]),
        dtype=np.complex128,
    )
    return transport, max_delta


def build_m_time_reversal_g_transport(
    *,
    g_vectors: np.ndarray,
    center: np.ndarray,
    tol: float = 1.0e-8,
):
    g_arr = np.asarray(g_vectors, dtype=float)
    center_arr = np.asarray(center, dtype=float).reshape(2)
    rows = []
    cols = []
    data = []
    used_targets = set()
    max_delta = 0.0
    for source_index, g_source in enumerate(g_arr):
        transformed = -(np.asarray(g_source, dtype=float) - center_arr) + center_arr
        deltas = np.linalg.norm(g_arr - transformed, axis=1)
        target_index = int(np.argmin(deltas))
        max_delta = max(max_delta, float(deltas[target_index]))
        if float(deltas[target_index]) > tol:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"M time-reversal centered G mapping failed for source index {source_index}: min delta={deltas[target_index]:.3e}.",
            )
        if target_index in used_targets:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"M time-reversal centered G mapping is not one-to-one; repeated target index {target_index}.",
            )
        used_targets.add(target_index)
        rows.append(target_index)
        cols.append(source_index)
        data.append(1.0 + 0.0j)
    transport = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(g_arr.shape[0], g_arr.shape[0]),
        dtype=np.complex128,
    )
    return transport, max_delta


def build_k_c2t_g_transport(
    *,
    source_g_vectors: np.ndarray,
    target_g_vectors: np.ndarray,
    source_k_center: np.ndarray,
    target_k_center: np.ndarray,
    linear_map: np.ndarray,
    tol: float = 1.0e-8,
):
    source_g = np.asarray(source_g_vectors, dtype=float)
    target_g = np.asarray(target_g_vectors, dtype=float)
    source_k = np.asarray(source_k_center, dtype=float).reshape(2)
    target_k = np.asarray(target_k_center, dtype=float).reshape(2)
    linear = np.asarray(linear_map, dtype=float).reshape(2, 2)
    rows = []
    cols = []
    data = []
    used_targets = set()
    max_delta = 0.0
    for source_index, g_source in enumerate(source_g):
        transformed = linear @ (-(np.asarray(g_source, dtype=float) - source_k)) + target_k
        deltas = np.linalg.norm(target_g - transformed, axis=1)
        target_index = int(np.argmin(deltas))
        max_delta = max(max_delta, float(deltas[target_index]))
        if float(deltas[target_index]) > tol:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"K_C2T direct G mapping failed for source index {source_index}: min delta={deltas[target_index]:.3e}.",
            )
        if target_index in used_targets:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"K_C2T direct G mapping is not one-to-one; repeated target index {target_index}.",
            )
        used_targets.add(target_index)
        rows.append(target_index)
        cols.append(source_index)
        data.append(1.0 + 0.0j)
    transport = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(target_g.shape[0], source_g.shape[0]),
        dtype=np.complex128,
    )
    return transport, max_delta


def diagnose_k_c2t_axis_choices(
    *,
    source_g_vectors: np.ndarray,
    target_g_vectors: np.ndarray,
    source_k_center: np.ndarray,
    target_k_center: np.ndarray,
    axis_choices: dict[str, np.ndarray],
    tol: float = 1.0e-8,
):
    source_g = np.asarray(source_g_vectors, dtype=float)
    target_g = np.asarray(target_g_vectors, dtype=float)
    source_k = np.asarray(source_k_center, dtype=float).reshape(2)
    target_k = np.asarray(target_k_center, dtype=float).reshape(2)
    diagnostics: dict[str, dict[str, Any]] = {}
    for label, linear_map in axis_choices.items():
        linear = np.asarray(linear_map, dtype=float).reshape(2, 2)
        used_targets: set[int] = set()
        max_delta = 0.0
        unmatched_count = 0
        repeated_target_count = 0
        for g_source in source_g:
            transformed = linear @ (-(np.asarray(g_source, dtype=float) - source_k)) + target_k
            deltas = np.linalg.norm(target_g - transformed, axis=1)
            target_index = int(np.argmin(deltas))
            delta = float(deltas[target_index])
            max_delta = max(max_delta, delta)
            if delta > tol:
                unmatched_count += 1
            if target_index in used_targets:
                repeated_target_count += 1
            used_targets.add(target_index)
        diagnostics[str(label)] = {
            "max_delta": max_delta,
            "unmatched_count": unmatched_count,
            "repeated_target_count": repeated_target_count,
            "one_to_one": bool(
                unmatched_count == 0
                and repeated_target_count == 0
                and len(used_targets) == len(source_g)
            ),
        }
    return diagnostics


def _diagnose_affine_g_mapping(
    *,
    source_g_vectors: np.ndarray,
    target_g_vectors: np.ndarray,
    source_k_center: np.ndarray,
    target_k_center: np.ndarray,
    linear_map: np.ndarray,
    antiunitary_like: bool,
    tol: float = 1.0e-8,
):
    source_g = np.asarray(source_g_vectors, dtype=float)
    target_g = np.asarray(target_g_vectors, dtype=float)
    source_k = np.asarray(source_k_center, dtype=float).reshape(2)
    target_k = np.asarray(target_k_center, dtype=float).reshape(2)
    linear = np.asarray(linear_map, dtype=float).reshape(2, 2)
    used_targets: set[int] = set()
    max_delta = 0.0
    unmatched_count = 0
    repeated_target_count = 0
    for g_source in source_g:
        shifted = -(np.asarray(g_source, dtype=float) - source_k) if antiunitary_like else (np.asarray(g_source, dtype=float) - source_k)
        transformed = linear @ shifted + target_k
        deltas = np.linalg.norm(target_g - transformed, axis=1)
        target_index = int(np.argmin(deltas))
        delta = float(deltas[target_index])
        max_delta = max(max_delta, delta)
        if delta > tol:
            unmatched_count += 1
        if target_index in used_targets:
            repeated_target_count += 1
        used_targets.add(target_index)
    return {
        "max_delta": max_delta,
        "unmatched_count": unmatched_count,
        "repeated_target_count": repeated_target_count,
        "one_to_one": bool(
            unmatched_count == 0
            and repeated_target_count == 0
            and len(used_targets) == len(source_g)
        ),
    }


def diagnose_k_c2t_center_choices(
    *,
    source_g_vectors: np.ndarray,
    target_g_vectors: np.ndarray,
    linear_map: np.ndarray,
    center_choices: dict[str, tuple[np.ndarray, np.ndarray]],
    tol: float = 1.0e-8,
):
    source_g = np.asarray(source_g_vectors, dtype=float)
    target_g = np.asarray(target_g_vectors, dtype=float)
    linear = np.asarray(linear_map, dtype=float).reshape(2, 2)
    diagnostics: dict[str, dict[str, Any]] = {}

    for label, (source_k_center, target_k_center) in center_choices.items():
        source_k = np.asarray(source_k_center, dtype=float).reshape(2)
        target_k = np.asarray(target_k_center, dtype=float).reshape(2)
        used_targets: set[int] = set()
        max_delta = 0.0
        unmatched_count = 0
        repeated_target_count = 0
        for g_source in source_g:
            transformed = linear @ (-(np.asarray(g_source, dtype=float) - source_k)) + target_k
            deltas = np.linalg.norm(target_g - transformed, axis=1)
            target_index = int(np.argmin(deltas))
            delta = float(deltas[target_index])
            max_delta = max(max_delta, delta)
            if delta > tol:
                unmatched_count += 1
            if target_index in used_targets:
                repeated_target_count += 1
            used_targets.add(target_index)
        diagnostics[str(label)] = {
            "max_delta": max_delta,
            "unmatched_count": unmatched_count,
            "repeated_target_count": repeated_target_count,
            "one_to_one": bool(
                unmatched_count == 0
                and repeated_target_count == 0
                and len(used_targets) == len(source_g)
            ),
        }
    return diagnostics


def build_m_c2_eta_g_transport(
    *,
    source_g_vectors: np.ndarray,
    target_g_vectors: np.ndarray,
    source_k_center: np.ndarray,
    target_k_center: np.ndarray,
    linear_map: np.ndarray,
    tol: float = 1.0e-8,
):
    source_g = np.asarray(source_g_vectors, dtype=float)
    target_g = np.asarray(target_g_vectors, dtype=float)
    source_k = np.asarray(source_k_center, dtype=float).reshape(2)
    target_k = np.asarray(target_k_center, dtype=float).reshape(2)
    linear = np.asarray(linear_map, dtype=float).reshape(2, 2)
    rows = []
    cols = []
    data = []
    used_targets = set()
    max_delta = 0.0
    for source_index, g_source in enumerate(source_g):
        transformed = linear @ (np.asarray(g_source, dtype=float) - source_k) + target_k
        deltas = np.linalg.norm(target_g - transformed, axis=1)
        target_index = int(np.argmin(deltas))
        max_delta = max(max_delta, float(deltas[target_index]))
        if float(deltas[target_index]) > tol:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"M_C2_eta direct G mapping failed for source index {source_index}: min delta={deltas[target_index]:.3e}.",
            )
        if target_index in used_targets:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"M_C2_eta direct G mapping is not one-to-one; repeated target index {target_index}.",
            )
        used_targets.add(target_index)
        rows.append(target_index)
        cols.append(source_index)
        data.append(1.0 + 0.0j)
    transport = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(target_g.shape[0], source_g.shape[0]),
        dtype=np.complex128,
    )
    return transport, max_delta


def unitary_m_c2_spin_rep(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {rotation_matrix.shape}")
    return spin_reps(rotation_matrix)


def m_unitary_c2_target_group_shifts(
    *,
    group_centers: dict[int, np.ndarray],
    group_target_map: dict[int, int],
    linear_map: np.ndarray,
    reciprocal_basis: np.ndarray,
    tol: float = 1.0e-8,
) -> tuple[dict[int, np.ndarray], float]:
    linear = np.asarray(linear_map, dtype=float).reshape(2, 2)
    shifts: dict[int, np.ndarray] = {}
    max_residual = 0.0
    for source_group, target_group in group_target_map.items():
        source_center = np.asarray(group_centers[int(source_group)], dtype=float).reshape(2)
        target_center = np.asarray(group_centers[int(target_group)], dtype=float).reshape(2)
        delta = linear @ source_center - target_center
        coeffs, residual = _round_trip_integer_coeffs(delta, reciprocal_basis)
        max_residual = max(max_residual, residual)
        if residual > tol:
            raise SymmetrySupportError(
                "reciprocal_shift_phase_unsupported",
                f"M_C2_eta reciprocal shift for group {source_group}->{target_group} is not integral: residual={residual:.3e}.",
            )
        existing = shifts.get(int(target_group))
        if existing is not None and not np.array_equal(existing, coeffs):
            raise SymmetrySupportError(
                "reciprocal_shift_phase_unsupported",
                f"M_C2_eta assigns inconsistent reciprocal shifts to target group {target_group}.",
            )
        shifts[int(target_group)] = coeffs
    return shifts, max_residual


def build_direct_linear_g_transport(
    *,
    source_g_vectors: np.ndarray,
    target_g_vectors: np.ndarray,
    linear_map: np.ndarray,
    tol: float = 1.0e-8,
):
    source_g = np.asarray(source_g_vectors, dtype=float)
    target_g = np.asarray(target_g_vectors, dtype=float)
    linear = np.asarray(linear_map, dtype=float).reshape(2, 2)
    rows = []
    cols = []
    data = []
    used_targets = set()
    max_delta = 0.0
    for source_index, g_source in enumerate(source_g):
        transformed = linear @ np.asarray(g_source, dtype=float)
        deltas = np.linalg.norm(target_g - transformed, axis=1)
        target_index = int(np.argmin(deltas))
        max_delta = max(max_delta, float(deltas[target_index]))
        if float(deltas[target_index]) > tol:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"Direct linear G mapping failed for source index {source_index}: min delta={deltas[target_index]:.3e}.",
            )
        if target_index in used_targets:
            raise SymmetrySupportError(
                "g_mapping_missing",
                f"Direct linear G mapping is not one-to-one; repeated target index {target_index}.",
            )
        used_targets.add(target_index)
        rows.append(target_index)
        cols.append(source_index)
        data.append(1.0 + 0.0j)
    transport = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(target_g.shape[0], source_g.shape[0]),
        dtype=np.complex128,
    )
    return transport, max_delta


def _is_gamma_c2y_rotation(rotation_cart: np.ndarray) -> bool:
    rotation_cart = np.asarray(rotation_cart, dtype=float)
    det3d = float(np.linalg.det(rotation_cart))
    if abs(det3d - 1.0) > 1.0e-8:
        return False
    if not np.allclose(rotation_cart @ rotation_cart, np.eye(3), atol=1.0e-8):
        return False
    linear_map = rotation_cart[:2, :2]
    return bool(np.allclose(linear_map, np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float), atol=1.0e-8))


def _is_proper_order2_rotation(rotation_cart: np.ndarray) -> bool:
    rotation_cart = np.asarray(rotation_cart, dtype=float)
    if rotation_cart.shape != (3, 3):
        return False
    det3d = float(np.linalg.det(rotation_cart))
    if abs(det3d - 1.0) > 1.0e-8:
        return False
    return bool(np.allclose(rotation_cart @ rotation_cart, np.eye(3), atol=1.0e-8))


def find_layer_exchange_c2_spatial_operation(structure, spatial_operations: list[dict[str, Any]]):
    candidates = list_layer_exchange_c2_spatial_operations(structure, spatial_operations)
    if not candidates:
        return None
    for operation in candidates:
        if _is_gamma_c2y_rotation(np.asarray(operation["rotation_cart"], dtype=float)):
            return operation
    return candidates[0]


def list_layer_exchange_c2_spatial_operations(structure, spatial_operations: list[dict[str, Any]]):
    candidates = []
    for operation in spatial_operations or []:
        rotation_cart = np.asarray(operation.get("rotation_cart", np.eye(3, dtype=float)), dtype=float)
        if not _is_proper_order2_rotation(rotation_cart):
            continue
        try:
            atom_mapping = _build_atom_mapping(structure, operation)
        except SymmetrySupportError:
            continue
        if atom_mapping.get("group_target_map") != {0: 1, 1: 0}:
            continue
        candidates.append(operation)
    return candidates


def find_matching_spatial_operation_for_rotation(
    structure,
    spatial_operations: list[dict[str, Any]],
    rotation_cart: np.ndarray,
    *,
    require_layer_exchange: bool = False,
    tol: float = 1.0e-6,
):
    target_rotation = np.asarray(rotation_cart, dtype=float)
    for operation in spatial_operations or []:
        op_rotation = np.asarray(operation.get("rotation_cart", np.eye(3, dtype=float)), dtype=float)
        if np.linalg.norm(op_rotation - target_rotation) > tol:
            continue
        try:
            atom_mapping = _build_atom_mapping(structure, operation)
        except SymmetrySupportError:
            continue
        if require_layer_exchange and atom_mapping.get("group_target_map") != {0: 1, 1: 0}:
            continue
        return operation, atom_mapping
    return None, None


def _minimal_symmetry_candidates_for_valley(
    valley_ctx: ValleyContext,
    bravais: str,
    spatial_operations: list[dict[str, Any]] | None = None,
    structure=None,
):
    candidates = [
        {
            "index": 0,
            "name": "E",
            "antiunitary": False,
            "rotation_cart": np.eye(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
    ]

    valley = int(valley_ctx.valley)
    if valley == 5 or valley in {31, 32, 33}:
        candidates.append(
            {
                "index": 1,
                "name": "T",
                "antiunitary": True,
                "rotation_cart": np.eye(3, dtype=float),
                "translation_cart": np.zeros(3, dtype=float),
            }
        )

    if supports_single_valley_c3(bravais, valley):
        for index, (name, angle_deg) in enumerate((("C3z", 120.0), ("C3z^2", 240.0)), start=2):
            theta = np.deg2rad(angle_deg)
            rotation_cart = np.eye(3, dtype=float)
            rotation_cart[:2, :2] = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ],
                dtype=float,
            )
            candidates.append(
                {
                    "index": index,
                    "name": name,
                    "antiunitary": False,
                    "rotation_cart": rotation_cart,
                    "translation_cart": np.zeros(3, dtype=float),
                }
            )
    if valley in {1, 2, 11, 12}:
        candidates.append(
            {
                "index": 20,
                "name": "K_C2T",
                "antiunitary": True,
                "rotation_cart": np.diag([-1.0, 1.0, -1.0]),
                "translation_cart": np.zeros(3, dtype=float),
            }
        )
    if valley in {31, 32, 33}:
        candidates.append(
            {
                "index": 30,
                "name": "M_C2_eta",
                "antiunitary": False,
                "rotation_cart": np.diag([-1.0, 1.0, -1.0]),
                "translation_cart": np.zeros(3, dtype=float),
            }
        )
    if spatial_operations is None:
        spatial_operations = []
    if valley == 5:
        selected_operation = None
        if structure is not None:
            selected_operation = find_layer_exchange_c2_spatial_operation(structure, spatial_operations)
        if selected_operation is None:
            for operation in spatial_operations:
                if not _is_gamma_c2y_rotation(np.asarray(operation["rotation_cart"], dtype=float)):
                    continue
                selected_operation = operation
                break
        if selected_operation is not None:
            payload = {
                "index": int(selected_operation["index"]),
                "name": "C2",
                "antiunitary": False,
                "rotation_frac": np.asarray(selected_operation.get("rotation_frac", np.eye(3, dtype=float)), dtype=float),
                "translation_frac": np.asarray(selected_operation.get("translation_frac", np.zeros(3, dtype=float)), dtype=float),
                "rotation_cart": np.asarray(selected_operation["rotation_cart"], dtype=float),
                "translation_cart": np.asarray(selected_operation.get("translation_cart", np.zeros(3, dtype=float)), dtype=float),
            }
            candidates.append(payload)
    return candidates


def _species_to_atomic_numbers(species_list):
    from ase.data import atomic_numbers

    numbers = []
    for species in species_list:
        symbol = str(species).strip()
        atomic_number = atomic_numbers.get(symbol, 0)
        if atomic_number == 0:
            raise ValueError(f"Unknown chemical symbol {symbol!r}")
        numbers.append(atomic_number)
    return np.asarray(numbers, dtype=int)


def collect_spglib_spatial_operations(structure, symprec: float = 1.0e-3):
    import spglib

    lattice = np.asarray(structure.Tmat, dtype=float)
    df = structure.df.copy().reset_index(drop=True)
    positions_cart = df[["x", "y", "z"]].to_numpy(dtype=float)
    positions_frac = _cartesian_positions_to_fractional(lattice, positions_cart)
    numbers = _species_to_atomic_numbers(df["species"].tolist())

    symmetry = spglib.get_symmetry((lattice, positions_frac, numbers), symprec=symprec)
    if symmetry is None:
        raise RuntimeError(f"spglib.get_symmetry failed with symprec={symprec}.")

    lattice_inv_t = np.linalg.inv(lattice.T)
    operations = []
    for index, (rotation_frac, translation_frac) in enumerate(
        zip(symmetry["rotations"], symmetry["translations"])
    ):
        rotation_frac = np.asarray(rotation_frac, dtype=float)
        translation_frac = np.asarray(translation_frac, dtype=float)
        rotation_cart = lattice.T @ rotation_frac @ lattice_inv_t
        translation_cart = lattice.T @ translation_frac
        operations.append(
            {
                "index": int(index),
                "name": _rotation_name(rotation_cart, index),
                "rotation_frac": rotation_frac,
                "translation_frac": translation_frac,
                "rotation_cart": rotation_cart,
                "translation_cart": translation_cart,
            }
        )
    return operations


def _projected_identity_operator(projected_dim: int):
    return scipy.sparse.identity(int(projected_dim), dtype=np.complex128, format="csr")


def _projected_time_reversal_operator(projected_dim: int, spinful: bool):
    projected_dim = int(projected_dim)
    if not spinful:
        return scipy.sparse.identity(projected_dim, dtype=np.complex128, format="csr")
    if projected_dim % 2 != 0:
        raise SymmetrySupportError(
            "spin_lift_unsupported",
            f"Spinful projected dimension must be even, got {projected_dim}.",
        )
    half_dim = projected_dim // 2
    return scipy.sparse.kron(
        scipy.sparse.csr_matrix(1.0j * _SIGMA_Y),
        scipy.sparse.identity(half_dim, dtype=np.complex128, format="csr"),
        format="csr",
    )


def resolve_tapw_reciprocal_basis_2d(structure, compute_config) -> np.ndarray:
    reciprocal = np.asarray(getattr(structure, "reciprocal_Tmat", np.eye(3, dtype=float)), dtype=float)
    bravais = str(getattr(compute_config, "bravais", "hex")).lower()
    if bravais in {"square", "rect"}:
        g1 = reciprocal[0][:2]
        g2 = reciprocal[1][:2]
    else:
        if reciprocal[0] @ reciprocal[1] < -0.01:
            g1 = reciprocal[0][:2]
        else:
            g1 = -reciprocal[0][:2]
        g2 = reciprocal[1][:2]
    return np.vstack([g1, g2])


def _prepare_compute_config_for_valley(compute_config, valley: int):
    cfg = copy.deepcopy(compute_config)
    cfg.mode = "symmetry"
    cfg.valley = int(valley)
    cfg.valleys = [int(valley)]
    cfg.C3_H = False
    cfg.M_valley_D3_H = False
    cfg.hamk_save = False
    return cfg


def _default_validation_q_points():
    return [
        ("Gamma", np.array([0.0, 0.0, 0.0], dtype=float)),
        ("q1", np.array([0.173, -0.119, 0.0], dtype=float)),
    ]


def _format_k_frac(q) -> str:
    q = np.asarray(q, dtype=float)
    return "[" + ", ".join(f"{value:.6f}" for value in q.tolist()) + "]"


def _format_matrix_json(matrix, *, ndigits: int = 10) -> str:
    if matrix is None:
        return ""
    arr = np.asarray(matrix, dtype=float)
    rounded = np.round(arr, decimals=ndigits)
    return json.dumps(rounded.tolist(), separators=(",", ":"))


def _format_vector_json(vector, *, ndigits: int = 10):
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=float)
    return np.round(arr, decimals=ndigits).tolist()


def _candidate_spglib_index(candidate: dict[str, Any]):
    if candidate.get("spglib_layer_exchange_c2") is not None:
        return int(candidate["spglib_layer_exchange_c2"]["index"])
    if candidate.get("spglib_m_c2_operation") is not None:
        return int(candidate["spglib_m_c2_operation"]["index"])
    if candidate.get("name") == "C2" and "rotation_frac" in candidate:
        return int(candidate["index"])
    return ""


def _build_detail_debug_payload(base_debug: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in base_debug.items() if value is not None}
    for key in ("center_convention", "phase_sign", "phase_side"):
        if diagnostics.get(key) is not None:
            payload[key] = diagnostics.get(key)
    target_group_shift_coeffs = diagnostics.get("target_group_shift_coeffs")
    if target_group_shift_coeffs is not None:
        payload["target_group_shift_coeffs"] = {
            str(group): np.asarray(coeffs, dtype=int).tolist()
            for group, coeffs in dict(target_group_shift_coeffs).items()
        }
    if diagnostics.get("transport_unitarity_residual") is not None:
        payload["transport_unitarity_residual"] = diagnostics.get("transport_unitarity_residual")
    return payload


def _parse_orbitals_from_orb_name(orb_name: str):
    import re

    if "-" in orb_name:
        _, orb_part = orb_name.split("-", 1)
    else:
        orb_part = orb_name
    orbitals = {}
    for orb, count in re.findall(r"([spdf])(\d+)", orb_part):
        orbitals[orb] = int(count)
    if not orbitals:
        raise SymmetrySupportError(
            "orbital_rotation_unsupported",
            f"Could not parse orbital content from orb_name={orb_name!r}.",
        )
    return orbitals


def _orbital_rotation_block(orb_name: str, rotation_cart: np.ndarray):
    orbitals = _parse_orbitals_from_orb_name(str(orb_name))
    try:
        orbital_mapping = {
            "s": get_any_rot_orb_twostep("s", rotation_cart),
            "p": get_any_rot_orb_twostep("p", rotation_cart),
            "d": get_any_rot_orb_twostep("d", rotation_cart),
            "f": get_any_rot_orb_twostep("f", rotation_cart),
        }
        return direct_sum(*generate_direct_sum_params(orbitals, orbital_mapping))
    except Exception as exc:
        raise SymmetrySupportError(
            "orbital_rotation_unsupported",
            f"Failed to build orbital rotation for orb_name={orb_name!r}: {exc}",
        ) from exc


def _sorted_structure_df(structure):
    return structure.df.copy().sort_values(["atom_type"], kind="stable").reset_index(drop=True)


def _build_atom_mapping(structure, operation, tol: float = 5.0e-6):
    lattice = np.asarray(structure.Tmat, dtype=float)
    df = _sorted_structure_df(structure)
    positions_cart = df[["x", "y", "z"]].to_numpy(dtype=float)
    positions_frac = _cartesian_positions_to_fractional(lattice, positions_cart)
    rotation_frac = np.asarray(operation["rotation_frac"], dtype=float)
    translation_frac = np.asarray(operation["translation_frac"], dtype=float)

    records = []
    group_targets: dict[int, set[int]] = {}
    atom_type_targets: dict[int, set[int]] = {}
    image_shifts = np.array(
        [[n1, n2, n3] for n1 in (-1, 0, 1) for n2 in (-1, 0, 1) for n3 in (-1, 0, 1)],
        dtype=float,
    )

    def _match_group_to_group(source_group: int, target_group: int):
        try:
            atom_type_map = map_atom_types_by_fractional_symmetry(
                structure_df=df,
                lattice=lattice,
                rotation_frac=rotation_frac,
                translation_frac=translation_frac,
                source_group=source_group,
                target_group=target_group,
                tol=tol,
            )
        except Exception as exc:
            raise SymmetrySupportError(
                "atom_mapping_missing",
                f"Atom-type symmetry mapping failed for twist-group match {source_group}->{target_group}: {exc}",
            ) from exc

        assignments_local: dict[int, tuple[int, np.ndarray]] = {}
        source_df = df[df["twist_group"].astype(int) == int(source_group)]
        for source_atom_type, target_atom_type in atom_type_map.items():
            source_rows = source_df[source_df["atom_type"].astype(int) == int(source_atom_type)]
            target_rows = df[
                (df["twist_group"].astype(int) == int(target_group))
                & (df["atom_type"].astype(int) == int(target_atom_type))
            ]
            if len(source_rows) != len(target_rows):
                raise SymmetrySupportError(
                    "atom_mapping_missing",
                    f"Atom-type {source_atom_type}->{target_atom_type} row count mismatch under twist-group match {source_group}->{target_group}.",
                )
            target_indices = target_rows.index.to_numpy(dtype=int)
            target_frac = positions_frac[target_indices]
            tiled_positions = np.vstack([target_frac + shift for shift in image_shifts])
            tiled_target_indices = np.concatenate([target_indices for _ in range(len(image_shifts))])
            tiled_shifts = np.vstack([np.repeat(shift[None, :], len(target_indices), axis=0) for shift in image_shifts])
            tree = cKDTree(tiled_positions)

            source_indices = source_rows.index.to_numpy(dtype=int)
            transformed = (positions_frac[source_indices] @ rotation_frac.T) + translation_frac
            k = min(tiled_positions.shape[0], max(8, len(target_indices)))
            distances, matches = tree.query(transformed, k=k, distance_upper_bound=tol)
            distances = np.atleast_2d(distances)
            matches = np.atleast_2d(matches)
            order = np.argsort(distances[:, 0])
            used_targets_local = set()
            for local_row in order:
                source_index = int(source_indices[local_row])
                picked = None
                for rank in range(distances.shape[1]):
                    match_index = int(matches[local_row, rank])
                    if match_index >= tiled_positions.shape[0]:
                        continue
                    target_index = int(tiled_target_indices[match_index])
                    if target_index in used_targets_local:
                        continue
                    picked = (target_index, tiled_shifts[match_index].astype(int))
                    break
                if picked is None:
                    raise SymmetrySupportError(
                        "atom_mapping_missing",
                        f"Could not assign a unique target for source atom_type={source_atom_type} under twist-group match {source_group}->{target_group}.",
                    )
                assignments_local[source_index] = picked
                used_targets_local.add(picked[0])
        return assignments_local

    unique_groups = sorted(df["twist_group"].astype(int).unique().tolist())
    group_assignment: dict[int, dict[int, tuple[int, np.ndarray]]] = {}
    for source_group in unique_groups:
        best = None
        for target_group in unique_groups:
            try:
                assignments_local = _match_group_to_group(source_group, target_group)
            except SymmetrySupportError:
                continue
            score = 0.0
            for source_index, (target_index, shift) in assignments_local.items():
                transformed = positions_frac[source_index] @ rotation_frac.T + translation_frac
                target_frac = positions_frac[target_index] + shift.astype(float)
                score += float(np.linalg.norm(transformed - target_frac))
            if best is None or score < best[0]:
                best = (score, target_group, assignments_local)
        if best is None:
            raise SymmetrySupportError(
                "atom_mapping_missing",
                f"Could not determine a target twist_group for source_group={source_group}.",
            )
        _, target_group, assignments_local = best
        group_assignment[int(source_group)] = assignments_local
        group_targets.setdefault(int(source_group), set()).add(int(target_group))

    for source_group, assignments_local in group_assignment.items():
        for source_index, (target_index, shift) in assignments_local.items():
            source_row = df.iloc[source_index]
            source_group = int(source_row["twist_group"])
            target_group = int(df.iloc[target_index]["twist_group"])
            source_atom_type = int(source_row["atom_type"])
            target_atom_type = int(df.iloc[target_index]["atom_type"])
            atom_type_targets.setdefault(source_atom_type, set()).add(target_atom_type)

            records.append(
                {
                    "source_index": int(source_index),
                    "target_index": int(target_index),
                    "shift_frac": shift,
                    "shift_cart": lattice.T @ shift.astype(float),
                    "source_group": source_group,
                    "target_group": target_group,
                    "source_atom_type": source_atom_type,
                    "target_atom_type": target_atom_type,
                    "orb_name": str(source_row["orb_name"]),
                    "orb_num": int(source_row["orb_num"]),
                }
            )

    for source_group, targets in group_targets.items():
        if len(targets) != 1:
            raise SymmetrySupportError(
                "atom_mapping_missing",
                f"Source twist_group={source_group} maps to multiple target groups {sorted(targets)}.",
            )
    for source_atom_type, targets in atom_type_targets.items():
        if len(targets) != 1:
            raise SymmetrySupportError(
                "atom_mapping_missing",
                f"Source atom_type={source_atom_type} maps to multiple target atom types {sorted(targets)}.",
            )

    return {
        "df": df,
        "records": records,
        "group_target_map": {group: next(iter(targets)) for group, targets in group_targets.items()},
        "atom_type_target_map": {atom_type: next(iter(targets)) for atom_type, targets in atom_type_targets.items()},
    }


def _build_group_orbital_transport_data(
    structure_df,
    source_group: int,
    target_group: int,
    rotation_matrix: np.ndarray,
    spin: bool,
    spin_rep: np.ndarray | None,
    target_for_source: dict[int, int],
):
    orb_mapping = {
        "s": get_any_rot_orb_twostep("s", rotation_matrix),
        "p": get_any_rot_orb_twostep("p", rotation_matrix),
        "d": get_any_rot_orb_twostep("d", rotation_matrix),
        "f": get_any_rot_orb_twostep("f", rotation_matrix),
    }
    if spin and spin_rep is None:
        spin_rep = spin_reps(rotation_matrix)

    df_temp = structure_df.copy()
    atom_type_list_global = np.unique(df_temp["atom_type"].values)
    df_source = df_temp[df_temp["twist_group"] == source_group]
    df_target = df_temp[df_temp["twist_group"] == target_group]
    if df_source.empty or df_target.empty:
        raise SymmetrySupportError(
            "atom_mapping_missing",
            f"Missing atoms for twist_group pair {source_group}->{target_group}.",
        )

    source_atom_types = [at for at in atom_type_list_global if (df_source["atom_type"] == at).any()]
    target_atom_types = [at for at in atom_type_list_global if (df_target["atom_type"] == at).any()]
    source_offsets = {}
    source_blocks = {}
    source_dim_total = 0
    for atom_type in source_atom_types:
        orb_name = df_source.loc[df_source["atom_type"] == atom_type, "orb_name"].iloc[0]
        orbitals_dict = _parse_orbitals_from_orb_name(orb_name)
        block = direct_sum(*generate_direct_sum_params(orbitals_dict, orb_mapping))
        source_offsets[int(atom_type)] = (source_dim_total, block.shape[0], orbitals_dict)
        source_blocks[int(atom_type)] = block
        source_dim_total += block.shape[0]

    target_offsets = {}
    target_dim_total = 0
    for atom_type in target_atom_types:
        orb_name = df_target.loc[df_target["atom_type"] == atom_type, "orb_name"].iloc[0]
        orbitals_dict = _parse_orbitals_from_orb_name(orb_name)
        block = direct_sum(*generate_direct_sum_params(orbitals_dict, orb_mapping))
        target_offsets[int(atom_type)] = (target_dim_total, block.shape[0], orbitals_dict)
        target_dim_total += block.shape[0]

    for source_atom_type in source_atom_types:
        target_atom_type = int(target_for_source[int(source_atom_type)])
        source_dim = source_offsets[int(source_atom_type)][1]
        target_dim = target_offsets[target_atom_type][1]
        if source_dim != target_dim:
            raise SymmetrySupportError(
                "orbital_rotation_unsupported",
                f"Source/target orbital dimensions mismatch for atom_type {source_atom_type}->{target_atom_type}.",
            )

    return SimpleNamespace(
        source_atom_types=[int(v) for v in source_atom_types],
        target_atom_types=[int(v) for v in target_atom_types],
        source_offsets=source_offsets,
        target_offsets=target_offsets,
        source_blocks=source_blocks,
        source_dim=source_dim_total,
        target_dim=target_dim_total,
        spin_rep=spin_rep,
    )


def _representative_position_2d(df_group, atom_type: int) -> np.ndarray:
    atom_rows = df_group[df_group["atom_type"].astype(int) == int(atom_type)]
    if atom_rows.empty:
        raise SymmetrySupportError(
            "atom_mapping_missing",
            f"No rows found for target atom_type={atom_type}.",
        )
    if {"shifted_x", "shifted_y"}.issubset(atom_rows.columns):
        return atom_rows[["shifted_x", "shifted_y"]].to_numpy(dtype=float)[0]
    return atom_rows[["x", "y"]].to_numpy(dtype=float)[0]


def _target_group_reciprocal_phase_diagonal(structure, valley_ctx: ValleyContext, target_group_shifts: dict[int, np.ndarray]):
    groups = sorted(target_group_shifts)
    df = structure.df
    phase_blocks = []
    for target_group in groups:
        coeffs = np.asarray(target_group_shifts[target_group], dtype=float).reshape(2)
        delta_cart = coeffs @ np.asarray(valley_ctx.moire_reciprocal_basis, dtype=float)
        group_df = df[df["twist_group"].astype(int) == int(target_group)]
        atom_phases = []
        for atom_type in sorted(group_df["atom_type"].astype(int).unique().tolist()):
            atom_rows = group_df[group_df["atom_type"].astype(int) == int(atom_type)]
            if atom_rows.empty:
                raise SymmetrySupportError(
                    "atom_mapping_missing",
                    f"No rows found for target_group={target_group}, atom_type={atom_type}.",
                )
            orb_num = int(atom_rows["orb_num"].iloc[0])
            position = _representative_position_2d(atom_rows, int(atom_type))
            phase = np.exp(1.0j * float(delta_cart @ position))
            atom_phases.extend([phase] * orb_num)
        q_count = len(np.asarray(valley_ctx.group_g_vectors[int(target_group)], dtype=float))
        phase_blocks.append(np.tile(np.asarray(atom_phases, dtype=np.complex128), q_count))

    spinless_phase = np.concatenate(phase_blocks) if phase_blocks else np.ones(0, dtype=np.complex128)
    if getattr(structure, "spin", False):
        full_phase = np.concatenate([spinless_phase, spinless_phase])
    else:
        full_phase = spinless_phase
    return scipy.sparse.diags(full_phase, offsets=0, dtype=np.complex128, format="csr")


def build_reciprocal_shift_g_pin(
    *,
    g_vectors: np.ndarray,
    shift_cart: np.ndarray,
    tol: float = 1.0e-8,
):
    g_arr = np.asarray(g_vectors, dtype=float)
    shift = np.asarray(shift_cart, dtype=float).reshape(2)
    rows = []
    cols = []
    data = []
    used_targets: set[int] = set()
    max_delta = 0.0
    unmatched_count = 0
    repeated_target_count = 0
    for source_index, g_source in enumerate(g_arr):
        shifted = np.asarray(g_source, dtype=float) + shift
        deltas = np.linalg.norm(g_arr - shifted, axis=1)
        target_index = int(np.argmin(deltas))
        delta = float(deltas[target_index])
        max_delta = max(max_delta, delta)
        if delta > tol:
            unmatched_count += 1
            continue
        if target_index in used_targets:
            repeated_target_count += 1
            continue
        used_targets.add(target_index)
        rows.append(target_index)
        cols.append(source_index)
        data.append(1.0 + 0.0j)
    matrix = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.complex128), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(g_arr.shape[0], g_arr.shape[0]),
        dtype=np.complex128,
    )
    one_to_one = bool(
        unmatched_count == 0
        and repeated_target_count == 0
        and len(used_targets) == g_arr.shape[0]
    )
    return matrix, {
        "max_delta": max_delta,
        "unmatched_count": int(unmatched_count),
        "repeated_target_count": int(repeated_target_count),
        "one_to_one": one_to_one,
    }


def _group_orbital_identity_block(structure_df, group_id: int):
    df = structure_df
    group_df = df[df["twist_group"].astype(int) == int(group_id)]
    dim = 0
    for atom_type in sorted(group_df["atom_type"].astype(int).unique().tolist()):
        atom_rows = group_df[group_df["atom_type"].astype(int) == int(atom_type)]
        if atom_rows.empty:
            continue
        dim += int(atom_rows["orb_num"].iloc[0])
    return scipy.sparse.identity(dim, dtype=np.complex128, format="csr")


def _candidate_group_target_map_for_source_pin(candidate: dict[str, Any], groups: list[int]) -> dict[int, int]:
    name = str(candidate.get("name", ""))
    if name in {"E", "T", "C3z", "C3z^2"}:
        return {int(group): int(group) for group in groups}
    if name in {"C2", "K_C2T", "M_C2_eta"} and set(groups) == {0, 1}:
        return {0: 1, 1: 0}
    return {int(group): int(group) for group in groups}


def _source_pin_rule_payload(candidate: dict[str, Any], valley_ctx: ValleyContext) -> dict[str, Any]:
    groups = sorted(int(group) for group in valley_ctx.group_g_vectors)
    linear = np.asarray(
        candidate.get("linear_map_2d", np.asarray(candidate.get("rotation_cart", np.eye(3)))[:2, :2]),
        dtype=float,
    ).reshape(2, 2)
    effective = -linear if bool(candidate.get("antiunitary", False)) else linear
    inverse_effective = np.linalg.inv(effective)
    group_target_map = _candidate_group_target_map_for_source_pin(candidate, groups)

    shifts_cart: dict[int, np.ndarray] = {}
    shifts_coeffs: dict[int, np.ndarray] = {}
    shift_residuals: dict[int, float] = {}
    for source_group in groups:
        target_group = int(group_target_map[int(source_group)])
        source_center = np.asarray(valley_ctx.group_k_centers[int(source_group)], dtype=float).reshape(2)
        target_center = np.asarray(valley_ctx.group_k_centers[int(target_group)], dtype=float).reshape(2)
        shift = inverse_effective @ target_center - source_center
        coeffs, residual = _round_trip_integer_coeffs(shift, valley_ctx.moire_reciprocal_basis)
        shifts_cart[int(source_group)] = shift
        shifts_coeffs[int(source_group)] = coeffs
        shift_residuals[int(source_group)] = residual

    return {
        "source_form": "ordinary_with_pin",
        "ld_source_rule": "q_lambda = R_eff^{-1}(k + K_target(lambda)) - K_lambda",
        "effective_linear_2d": effective,
        "inverse_effective_linear_2d": inverse_effective,
        "group_target_map": group_target_map,
        "pin_shift_by_group_cart": shifts_cart,
        "pin_shift_by_group_coeffs": shifts_coeffs,
        "pin_shift_integrality_residual_by_group": shift_residuals,
    }


def build_source_pin_matrix_for_candidate(structure, valley_ctx: ValleyContext, candidate: dict[str, Any]):
    rule = _source_pin_rule_payload(candidate, valley_ctx)
    groups = sorted(int(group) for group in valley_ctx.group_g_vectors)
    group_blocks = []
    diagnostics: dict[str, Any] = {
        "pin_supported": True,
        "pin_reason": "",
        "pin_g_perm_max_delta": 0.0,
        "pin_unmatched_count": 0,
        "pin_repeated_target_count": 0,
    }
    for group in groups:
        q_pin, group_diag = build_reciprocal_shift_g_pin(
            g_vectors=valley_ctx.group_g_vectors[int(group)],
            shift_cart=rule["pin_shift_by_group_cart"][int(group)],
            tol=1.0e-8,
        )
        diagnostics["pin_g_perm_max_delta"] = max(
            float(diagnostics["pin_g_perm_max_delta"]),
            float(group_diag["max_delta"]),
        )
        diagnostics["pin_unmatched_count"] += int(group_diag["unmatched_count"])
        diagnostics["pin_repeated_target_count"] += int(group_diag["repeated_target_count"])
        if not group_diag["one_to_one"]:
            diagnostics["pin_supported"] = False
        orbital_identity = _group_orbital_identity_block(structure.df, int(group))
        group_blocks.append(scipy.sparse.kron(q_pin, orbital_identity, format="csr"))

    spinless = scipy.sparse.block_diag(group_blocks, format="csr")
    if getattr(structure, "spin", False):
        pin_matrix = scipy.sparse.kron(
            scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
            spinless,
            format="csr",
        )
    else:
        pin_matrix = spinless

    if not bool(diagnostics["pin_supported"]):
        diagnostics["pin_reason"] = "pin_not_closed_in_truncated_g_basis"
        rule["source_form"] = "ld_source_rule"
        pin_matrix.sort_indices()
        return pin_matrix, rule, diagnostics
    pin_matrix.sort_indices()
    return pin_matrix, rule, diagnostics


def _atomic_periodic_gauge_phase(structure, shift_by_group_cart: dict[int, np.ndarray]) -> np.ndarray:
    if not hasattr(structure, "df") or structure.df is None:
        raise SymmetrySupportError("atom_mapping_missing", "Structure dataframe is required to build P_G.")
    df = structure.df.copy().sort_values(["atom_type"], kind="stable").reset_index(drop=True)
    if {"x", "y"}.issubset(df.columns):
        coord_cols = ["x", "y"]
    elif {"shifted_x", "shifted_y"}.issubset(df.columns):
        coord_cols = ["shifted_x", "shifted_y"]
    else:
        raise SymmetrySupportError("atom_mapping_missing", "Structure dataframe lacks x/y coordinates for P_G.")
    coords = np.repeat(df[coord_cols].to_numpy(dtype=float), df["orb_num"].to_numpy(dtype=int), axis=0)
    groups = np.repeat(df["twist_group"].astype(int).to_numpy(), df["orb_num"].to_numpy(dtype=int), axis=0)
    phases = np.empty(coords.shape[0], dtype=np.complex128)
    for group in sorted(np.unique(groups).tolist()):
        shift = np.asarray(shift_by_group_cart.get(int(group), np.zeros(2)), dtype=float).reshape(2)
        mask = groups == int(group)
        phases[mask] = np.exp(-1.0j * (coords[mask] @ shift))
    return phases


def _projected_periodic_gauge_dim(structure, valley_ctx: ValleyContext) -> int:
    if valley_ctx.calculator is not None and getattr(valley_ctx.calculator, "TAPW_parameters", None) is not None:
        g_matrix = getattr(valley_ctx.calculator.TAPW_parameters, "g_matrix", None)
        if g_matrix is not None:
            return int(g_matrix.shape[0])
    projected_dim = sum(
        len(np.asarray(vectors)) * _group_orbital_identity_block(structure.df, int(group)).shape[0]
        for group, vectors in valley_ctx.group_g_vectors.items()
    )
    if getattr(structure, "spin", False):
        projected_dim *= 2
    return int(projected_dim)


def build_periodic_gauge_matrix_for_candidate(structure, valley_ctx: ValleyContext, candidate: dict[str, Any]):
    name = str(candidate.get("name", ""))
    if name not in {"C3z", "C3z^2"} or bool(candidate.get("antiunitary", False)):
        projected_dim = _projected_periodic_gauge_dim(structure, valley_ctx)
        groups = sorted(int(group) for group in valley_ctx.group_g_vectors)
        return scipy.sparse.identity(projected_dim, dtype=np.complex128, format="csr"), {
            "pg_shift_by_group_coeffs": {str(group): [0, 0] for group in groups},
            "pg_phase_convention": PG_PHASE_CONVENTION,
        }, {
            "pg_identity": True,
            "pg_phase_convention": PG_PHASE_CONVENTION,
        }

    rule = _source_pin_rule_payload(candidate, valley_ctx)
    shift_by_group_cart = {
        int(group): np.asarray(shift, dtype=float).reshape(2)
        for group, shift in dict(rule.get("pin_shift_by_group_cart", {})).items()
    }
    shift_by_group_coeffs = _json_int_vector_dict(rule.get("pin_shift_by_group_coeffs"))
    diagnostics = {
        "pg_identity": all(float(np.linalg.norm(shift)) < 1.0e-12 for shift in shift_by_group_cart.values()),
        "pg_phase_convention": PG_PHASE_CONVENTION,
    }
    g_matrix = None
    if valley_ctx.calculator is not None and getattr(valley_ctx.calculator, "TAPW_parameters", None) is not None:
        g_matrix = getattr(valley_ctx.calculator.TAPW_parameters, "g_matrix", None)
    projected_dim = _projected_periodic_gauge_dim(structure, valley_ctx)

    if diagnostics["pg_identity"]:
        pg_matrix = scipy.sparse.identity(projected_dim, dtype=np.complex128, format="csr")
        return pg_matrix, {
            "pg_shift_by_group_coeffs": shift_by_group_coeffs,
            "pg_phase_convention": PG_PHASE_CONVENTION,
        }, diagnostics

    if g_matrix is None:
        raise SymmetrySupportError("q_mapping_missing", "TAPW g_matrix is required to build nontrivial P_G.")
    g_matrix = g_matrix.tocsr() if scipy.sparse.issparse(g_matrix) else scipy.sparse.csr_matrix(g_matrix)
    spinless_phase = _atomic_periodic_gauge_phase(structure, shift_by_group_cart)
    if g_matrix.shape[1] == spinless_phase.shape[0]:
        phase = spinless_phase
    elif g_matrix.shape[1] == 2 * spinless_phase.shape[0]:
        phase = np.concatenate([spinless_phase, spinless_phase])
    else:
        raise SymmetrySupportError(
            "q_mapping_missing",
            f"P_G phase length {spinless_phase.shape[0]} is incompatible with g_matrix columns {g_matrix.shape[1]}.",
        )
    phase_diag = scipy.sparse.diags(phase, offsets=0, dtype=np.complex128, format="csr")
    pg_matrix = (g_matrix @ phase_diag @ g_matrix.conj().T).tocsr()
    pg_matrix.sort_indices()
    return pg_matrix, {
        "pg_shift_by_group_coeffs": shift_by_group_coeffs,
        "pg_phase_convention": PG_PHASE_CONVENTION,
    }, diagnostics


def source_side_periodic_gauge_from_raw_transport(pure_transport, raw_transport, *, antiunitary: bool):
    """Extract source-side P_G from a raw transport that already includes gauge phases."""
    pure = pure_transport.tocsr() if scipy.sparse.issparse(pure_transport) else scipy.sparse.csr_matrix(pure_transport)
    raw = raw_transport.tocsr() if scipy.sparse.issparse(raw_transport) else scipy.sparse.csr_matrix(raw_transport)
    if pure.shape != raw.shape:
        raise ValueError(f"pure/raw transport shapes differ: {pure.shape} vs {raw.shape}")
    source_pg = (pure.conj().T @ raw).tocsr()
    if bool(antiunitary):
        source_pg = source_pg.conj().tocsr()
    source_pg.sort_indices()
    return source_pg


def _closure_to_scalar_identity(matrix, atol: float = 1.0e-6):
    if scipy.sparse.issparse(matrix):
        matrix_csr = matrix.tocsr()
        identity = scipy.sparse.identity(matrix_csr.shape[0], dtype=np.complex128, format="csr")
        trace = np.sum(matrix_csr.diagonal())
        phase = trace / matrix_csr.shape[0]
        if abs(phase) > 0.0:
            phase = phase / abs(phase)
        residual = frobenius_relative_residual(matrix_csr, phase * identity, denominator=identity)
        return residual <= float(atol), phase, residual

    matrix = np.asarray(matrix, dtype=np.complex128)
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    phase = np.trace(matrix) / matrix.shape[0]
    if abs(phase) > 0.0:
        phase = phase / abs(phase)
    residual = frobenius_relative_residual(matrix, phase * identity, denominator=identity)
    return residual <= float(atol), phase, residual


def compute_lowdin_order_residual_antiunitary_order2(h0, s0, h1, s1, transport, atol: float = 1.0e-12):
    if not is_positive_definite(s0, atol=atol) or not is_positive_definite(s1, atol=atol):
        return {
            "supported": False,
            "not_supported_reason": "S_not_positive_definite",
            "residual": None,
        }

    h_sym = 0.5 * (h0 + transport @ h1.conj() @ transport.conj().T)
    s_sym = 0.5 * (s0 + transport @ s1.conj() @ transport.conj().T)
    if not is_positive_definite(s_sym, atol=atol):
        return {
            "supported": False,
            "not_supported_reason": "S_not_positive_definite",
            "residual": None,
        }

    strict_h = lowdin_transform(h_sym, s_sym, atol=atol)
    h0_lowdin = lowdin_transform(h0, s0, atol=atol)
    h1_lowdin = lowdin_transform(h1, s1, atol=atol)
    engineering_h = 0.5 * (h0_lowdin + transport @ h1_lowdin.conj() @ transport.conj().T)
    return {
        "supported": True,
        "not_supported_reason": "",
        "residual": frobenius_relative_residual(strict_h, engineering_h, denominator=strict_h),
    }


def _symmetrize_unitary_orbit_at_index(h_orbit, orbit_transports_to_ref, target_index: int):
    target_transport = orbit_transports_to_ref[target_index]
    if scipy.sparse.issparse(target_transport):
        target_transport_h = target_transport.conj().T.tocsr()
        transports_to_target = [
            (target_transport_h @ transport).tocsr() if scipy.sparse.issparse(transport)
            else target_transport_h @ np.asarray(transport, dtype=np.complex128)
            for transport in orbit_transports_to_ref
        ]
    else:
        target_transport_dense = np.asarray(target_transport, dtype=np.complex128)
        target_transport_h = target_transport_dense.conj().T
        transports_to_target = [
            target_transport_h @ (transport.toarray() if scipy.sparse.issparse(transport) else np.asarray(transport, dtype=np.complex128))
            for transport in orbit_transports_to_ref
        ]
    return symmetrize_unitary_orbit(h_orbit, transports_to_target)


class SymmetryAnalysisRunner:
    """First-class TAPW symmetry-analysis runner."""

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.structure = kwargs.get("structure")
        self.hr_supercell = kwargs.get("hr_supercell")
        self.sr_supercell = kwargs.get("sr_supercell")
        self.logger = kwargs.get("logger")

        base_output_dir = Path(self.config.paths.output_dir)
        relative_output_dir = getattr(self.config.symmetry_analysis, "output_dir", "symmetry_analysis")
        self.output_dir = str(base_output_dir / relative_output_dir)
        self._operation_mapping_cache: dict[tuple[int, bool], dict[str, Any]] = {}
        self._calculator_cache: dict[int, BandStructureCalculator] = {}
        self._raw_hs_cache: dict[tuple[int, tuple[float, float, float]], tuple[np.ndarray, np.ndarray | None]] = {}
        self._raw_c3_h_orbit_cache: dict[tuple[int, tuple[float, float, float]], list[np.ndarray]] = {}
        self._transport_cache: dict[Any, np.ndarray] = {}
        self._legacy_c3_matrix_cache: dict[int, np.ndarray] = {}

    def _calculator_for_valley(self, valley: int) -> BandStructureCalculator:
        calculator = self._calculator_cache.get(int(valley))
        if calculator is not None:
            return calculator
        compute_cfg = _prepare_compute_config_for_valley(self.config.compute, valley)
        calculator = BandStructureCalculator(
            hr_supercell=self.hr_supercell,
            sr_supercell=self.sr_supercell,
            structure=self.structure,
            config=compute_cfg,
            kpath_config=None,
        )
        self._calculator_cache[int(valley)] = calculator
        return calculator

    def _valley_context_for_valley(self, valley: int) -> ValleyContext:
        calculator = self._calculator_for_valley(valley)
        return resolve_tapw_valley_context(self.structure, calculator.config, calculator=calculator)

    def _raw_projected_hs(self, valley: int, q_local):
        q_local = np.asarray(q_local, dtype=float)
        cache_key = (int(valley), tuple(float(v) for v in q_local))
        cached = self._raw_hs_cache.get(cache_key)
        if cached is not None:
            return cached
        calculator = self._calculator_for_valley(valley)
        phase_ctx = calculator._build_getk_phase_context(q_local)
        h_full = calculator._assemble_sparse_realspace_matrix(
            calculator.hr_supercell,
            phase_ctx,
            type="H",
        )
        hamk = calculator.cal_TAPW_hamiltonian_k_cpu(
            h_full,
            tapw_parameters=calculator.TAPW_parameters,
            force_sparse_dot=True,
        )
        self._raw_hs_cache[cache_key] = (np.asarray(hamk, dtype=np.complex128), None)
        return self._raw_hs_cache[cache_key]

    def _raw_c3_h_orbit(self, valley: int, q_local):
        q_local = np.asarray(q_local, dtype=float)
        cache_key = (int(valley), tuple(float(v) for v in q_local))
        cached = self._raw_c3_h_orbit_cache.get(cache_key)
        if cached is not None:
            return cached
        calculator = self._calculator_for_valley(valley)
        orbit = calculator.Getk_super_gauge_sparse_symm(
            self.hr_supercell,
            q_local,
            force_sparse_dot=True,
        )
        if len(orbit) < 3:
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"Legacy C3 orbit builder returned {len(orbit)} matrices; expected at least 3.",
            )
        self._raw_c3_h_orbit_cache[cache_key] = [
            np.asarray(matrix, dtype=np.complex128)
            for matrix in orbit[:3]
        ]
        return self._raw_c3_h_orbit_cache[cache_key]

    def _legacy_c3_matrix_for_valley(self, valley: int) -> np.ndarray:
        cached = self._legacy_c3_matrix_cache.get(int(valley))
        if cached is not None:
            return cached

        cfg = _prepare_compute_config_for_valley(self.config.compute, valley)
        cfg.C3_H = True
        params = TAPW_parameters(self.structure, cfg)
        params.generate_all_parameters()
        if params.C3_matrix is None:
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"Legacy C3 matrix is unavailable for valley {valley}.",
            )
        matrix = params.C3_matrix.tocsr()
        self._legacy_c3_matrix_cache[int(valley)] = matrix
        return matrix

    def _validated_generic_c3_transport(self, valley: int, power: int):
        cache_key = ("validated_c3", int(valley), int(power))
        cached = self._transport_cache.get(cache_key)
        if cached is not None:
            return cached

        generic_c3 = self._build_generic_c3_projected_transport(valley, 120.0)
        legacy_c3 = self._legacy_c3_matrix_for_valley(valley).tocsr()
        residual_c3 = compare_transport_against_legacy_c3(generic_c3, legacy_c3)
        if residual_c3 >= 1.0e-8:
            raise SymmetrySupportError(
                "generic_C3_builder_failed",
                f"Generic C3 transport does not match legacy C3: residual={residual_c3:.3e}.",
            )

        generic_power = legacy_c3
        legacy_power = legacy_c3
        for _ in range(1, int(power)):
            generic_power = (generic_power @ legacy_c3).tocsr()
            legacy_power = (legacy_power @ legacy_c3).tocsr()

        residual_power = compare_transport_against_legacy_c3(generic_power, legacy_power)
        if residual_power >= 1.0e-8:
            raise SymmetrySupportError(
                "generic_C3_builder_failed",
                f"Generic C3^{power} transport does not match legacy C3^{power}: residual={residual_power:.3e}.",
            )
        self._transport_cache[cache_key] = generic_power
        return generic_power

    def _build_generic_c3_projected_transport(self, valley: int, angle_deg: float):
        valley_ctx = self._valley_context_for_valley(valley)
        rotation_cart = np.eye(3, dtype=float)
        theta = np.deg2rad(float(angle_deg))
        rotation_cart[:2, :2] = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=float,
        )
        group_blocks = []
        for group_id in sorted(valley_ctx.group_g_vectors):
            g_transport = build_c3_g_transport_from_tapw_convention(
                g_source=valley_ctx.group_g_vectors[group_id],
                g_target=valley_ctx.group_g_vectors[group_id],
                layer_center=valley_ctx.group_k_centers[group_id],
                angle_deg=angle_deg,
            )
            group_df = self.structure.df[self.structure.df["twist_group"].astype(int) == int(group_id)]
            atom_type_list_global = np.unique(self.structure.df["atom_type"].values)
            type_blocks = []
            for atom_type in atom_type_list_global:
                if not (group_df["atom_type"] == atom_type).any():
                    continue
                orb_name = group_df.loc[group_df["atom_type"] == atom_type, "orb_name"].iloc[0]
                type_blocks.append(_orbital_rotation_block(orb_name, rotation_cart))
            orbital_block = scipy.sparse.csr_matrix(direct_sum(*type_blocks))
            group_blocks.append(scipy.sparse.kron(g_transport, orbital_block, format="csr"))

        spinless = scipy.sparse.block_diag(group_blocks, format="csr")
        if getattr(self.structure, "spin", False):
            spin_block = scipy.sparse.csr_matrix(spin_reps(rotation_cart))
            return scipy.sparse.kron(spin_block, spinless, format="csr")
        return spinless

    def _build_projected_transport_direct(self, candidate: dict[str, Any], valley: int, q_target, q_source, atom_mapping):
        valley_ctx = self._valley_context_for_valley(valley)
        q_target = np.asarray(q_target, dtype=float)
        q_source = np.asarray(q_source, dtype=float)
        rotation_cart = np.asarray(candidate["rotation_cart"], dtype=float)
        linear_map = rotation_cart[:2, :2]
        antiunitary = bool(candidate.get("antiunitary", False))
        effective_linear = -linear_map if antiunitary else linear_map
        df = atom_mapping["df"]
        unique_groups = sorted(df["twist_group"].unique().tolist())
        group_index = {group: idx for idx, group in enumerate(unique_groups)}
        spinless_blocks = [[None for _ in unique_groups] for _ in unique_groups]
        returned_spin_rep = None

        q_target_cart = np.dot(q_target, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
        q_source_cart = np.dot(q_source, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
        spin_rep_override = None
        if getattr(self.structure, "spin", False) and antiunitary:
            spin_rep_override = spin_reps(rotation_cart) @ (1.0j * _SIGMA_Y)

        for source_group in unique_groups:
            target_group = int(atom_mapping["group_target_map"][source_group])
            source_g = np.asarray(valley_ctx.group_g_vectors[source_group], dtype=float)
            target_g = np.asarray(valley_ctx.group_g_vectors[target_group], dtype=float)
            permutation, reciprocal_shifts = build_projected_g_transport_by_physical_momentum(
                source_g_vectors=source_g,
                target_g_vectors=target_g,
                source_k_center=valley_ctx.group_k_centers[source_group],
                target_k_center=valley_ctx.group_k_centers[target_group],
                q_source_cart=q_source_cart,
                q_target_cart=q_target_cart,
                effective_linear_map=effective_linear,
                reciprocal_basis=valley_ctx.moire_reciprocal_basis,
                tol=1.0e-6,
            )
            if np.any(reciprocal_shifts != 0):
                raise SymmetrySupportError(
                    "reciprocal_shift_phase_unsupported",
                    f"Operation {candidate['name']} requires nonzero reciprocal shifts in projected G transport for group {source_group}->{target_group}.",
                )

            source_atom_types = sorted(df.loc[df["twist_group"] == source_group, "atom_type"].astype(int).unique().tolist())
            target_for_source = {
                atom_type: int(atom_mapping["atom_type_target_map"][atom_type])
                for atom_type in source_atom_types
            }
            group_data = _build_group_orbital_transport_data(
                structure_df=self.structure.df,
                source_group=source_group,
                target_group=target_group,
                rotation_matrix=rotation_cart,
                spin=bool(getattr(self.structure, "spin", False)),
                spin_rep=spin_rep_override,
                target_for_source=target_for_source,
            )
            if returned_spin_rep is None and group_data.spin_rep is not None:
                returned_spin_rep = group_data.spin_rep

            rows_parts = []
            cols_parts = []
            data_parts = []
            for source_g_index, target_g_index in enumerate(permutation.tolist()):
                for source_atom_type in group_data.source_atom_types:
                    target_atom_type = target_for_source[source_atom_type]
                    source_offset, block_dim, _ = group_data.source_offsets[source_atom_type]
                    target_offset, _, _ = group_data.target_offsets[target_atom_type]
                    block = group_data.source_blocks[source_atom_type]
                    row_base = int(target_g_index * group_data.target_dim + target_offset)
                    col_base = int(source_g_index * group_data.source_dim + source_offset)
                    row_idx = row_base + np.arange(block_dim, dtype=np.int64)
                    col_idx = col_base + np.arange(block_dim, dtype=np.int64)
                    rows_parts.append(np.repeat(row_idx, block_dim))
                    cols_parts.append(np.tile(col_idx, block_dim))
                    data_parts.append(block.reshape(-1))

            block_shape = (
                int(len(target_g) * group_data.target_dim),
                int(len(source_g) * group_data.source_dim),
            )
            if rows_parts:
                block_matrix = scipy.sparse.coo_matrix(
                    (
                        np.concatenate(data_parts).astype(np.complex128, copy=False),
                        (np.concatenate(rows_parts), np.concatenate(cols_parts)),
                    ),
                    shape=block_shape,
                    dtype=np.complex128,
                ).tocsr()
                block_matrix.sort_indices()
            else:
                block_matrix = scipy.sparse.csr_matrix(block_shape, dtype=np.complex128)
            spinless_blocks[group_index[target_group]][group_index[source_group]] = block_matrix

        spinless = scipy.sparse.bmat(spinless_blocks, format="csr")
        if getattr(self.structure, "spin", False):
            if returned_spin_rep is None:
                raise SymmetrySupportError(
                    "spin_lift_unsupported",
                    "Spinful projected transport did not yield a spin representation.",
                )
            return scipy.sparse.kron(scipy.sparse.csr_matrix(returned_spin_rep), spinless, format="csr")
        return spinless

    def _build_time_reversal_transport(self, valley: int, q_target, q_source, *, include_m_reciprocal_phase: bool = True):
        if int(valley) in {31, 32, 33}:
            return self._build_m_time_reversal_transport(
                valley,
                q_target,
                q_source,
                include_reciprocal_phase=include_m_reciprocal_phase,
            )

        valley_ctx = self._valley_context_for_valley(valley)
        q_target = np.asarray(q_target, dtype=float)
        q_source = np.asarray(q_source, dtype=float)
        if not np.allclose(q_source[:2], -q_target[:2], atol=1.0e-8):
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"Time reversal requires q_source=-q_target; got q_target={q_target.tolist()}, q_source={q_source.tolist()}.",
            )

        df = self.structure.df
        unique_groups = sorted(df["twist_group"].astype(int).unique().tolist())
        group_index = {group: idx for idx, group in enumerate(unique_groups)}
        spinless_blocks = [[None for _ in unique_groups] for _ in unique_groups]
        diagnostics = {
            "g_perm_max_delta": 0.0,
            "nonzero_reciprocal_shift_count": 0,
            "center_convention": "group_k_centers",
            "phase_sign": "+1",
            "phase_side": "left_target_rows",
        }

        for group_id in unique_groups:
            g_source = np.asarray(valley_ctx.group_g_vectors[group_id], dtype=float)
            q_transport, group_max_delta = build_time_reversal_g_transport(g_vectors=g_source, tol=1.0e-8)
            diagnostics["g_perm_max_delta"] = max(diagnostics["g_perm_max_delta"], group_max_delta)
            group_df = df[df["twist_group"].astype(int) == int(group_id)]
            atom_type_list = np.unique(group_df["atom_type"].values)
            type_offsets = {}
            type_dim_total = 0
            type_blocks = {}
            for atom_type in atom_type_list:
                orb_name = group_df.loc[group_df["atom_type"] == atom_type, "orb_name"].iloc[0]
                block = _orbital_rotation_block(orb_name, np.eye(3, dtype=float))
                type_offsets[int(atom_type)] = (type_dim_total, block.shape[0])
                type_blocks[int(atom_type)] = block
                type_dim_total += block.shape[0]
            orbital_identity = scipy.sparse.csr_matrix(
                direct_sum(*[type_blocks[int(atom_type)] for atom_type in atom_type_list])
            )
            block = scipy.sparse.kron(q_transport, orbital_identity, format="csr")
            spinless_blocks[group_index[group_id]][group_index[group_id]] = block

        spinless = scipy.sparse.bmat(spinless_blocks, format="csr")
        if getattr(self.structure, "spin", False):
            spin_block = scipy.sparse.csr_matrix(1.0j * _SIGMA_Y)
            transport = scipy.sparse.kron(spin_block, spinless, format="csr")
        else:
            transport = spinless

        ident = scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")
        delta = (transport.conj().T @ transport - ident).tocsr()
        diagnostics["transport_unitarity_residual"] = float(np.sqrt(np.sum(np.abs(delta.data) ** 2)) / np.sqrt(transport.shape[0]))
        t_sq = (transport @ transport.conj()).tocsr()
        expected = (-1.0 if getattr(self.structure, "spin", False) else 1.0) * ident
        t_sq_delta = (t_sq - expected).tocsr()
        diagnostics["t_square_residual"] = float(np.sqrt(np.sum(np.abs(t_sq_delta.data) ** 2)) / np.sqrt(transport.shape[0]))
        self._last_transport_diagnostics = diagnostics
        return transport

    def _build_m_time_reversal_transport(self, valley: int, q_target, q_source, *, include_reciprocal_phase: bool = True):
        valley_ctx = self._valley_context_for_valley(valley)
        q_target = np.asarray(q_target, dtype=float)
        q_source = np.asarray(q_source, dtype=float)
        if not np.allclose(q_source[:2], -q_target[:2], atol=1.0e-8):
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"M time reversal requires q_source=-q_target; got q_target={q_target.tolist()}, q_source={q_source.tolist()}.",
            )

        df = self.structure.df
        unique_groups = sorted(df["twist_group"].astype(int).unique().tolist())
        group_index = {group: idx for idx, group in enumerate(unique_groups)}
        spinless_blocks = [[None for _ in unique_groups] for _ in unique_groups]
        target_group_shifts: dict[int, np.ndarray] = {}
        diagnostics = {
            "g_perm_max_delta": 0.0,
            "nonzero_reciprocal_shift_count": 0,
        }

        for group_id in unique_groups:
            g_source = np.asarray(valley_ctx.group_g_vectors[int(group_id)], dtype=float)
            center = np.asarray(valley_ctx.group_k_centers[int(group_id)], dtype=float)
            q_transport, group_max_delta = build_m_time_reversal_g_transport(
                g_vectors=g_source,
                center=center,
                tol=1.0e-8,
            )
            diagnostics["g_perm_max_delta"] = max(diagnostics["g_perm_max_delta"], group_max_delta)

            shift_coeffs, shift_residual = _round_trip_integer_coeffs(
                -2.0 * center,
                valley_ctx.moire_reciprocal_basis,
            )
            if shift_residual >= 1.0e-8:
                self._last_transport_diagnostics = diagnostics
                raise SymmetrySupportError(
                    "reciprocal_shift_phase_unsupported",
                    f"M time reversal reciprocal shift for group {group_id} is not integral: residual={shift_residual:.3e}.",
                )
            target_group_shifts[int(group_id)] = shift_coeffs
            if np.any(shift_coeffs != 0):
                diagnostics["nonzero_reciprocal_shift_count"] += 1

            group_df = df[df["twist_group"].astype(int) == int(group_id)]
            atom_type_list = np.unique(group_df["atom_type"].values)
            type_blocks = {}
            for atom_type in atom_type_list:
                orb_name = group_df.loc[group_df["atom_type"] == atom_type, "orb_name"].iloc[0]
                type_blocks[int(atom_type)] = _orbital_rotation_block(orb_name, np.eye(3, dtype=float))
            orbital_identity = scipy.sparse.csr_matrix(
                direct_sum(*[type_blocks[int(atom_type)] for atom_type in atom_type_list])
            )
            block = scipy.sparse.kron(q_transport, orbital_identity, format="csr")
            spinless_blocks[group_index[int(group_id)]][group_index[int(group_id)]] = block

        diagnostics["target_group_shift_coeffs"] = {
            int(group): np.asarray(coeffs, dtype=int)
            for group, coeffs in target_group_shifts.items()
        }

        spinless = scipy.sparse.bmat(spinless_blocks, format="csr")
        if getattr(self.structure, "spin", False):
            spin_block = scipy.sparse.csr_matrix(1.0j * _SIGMA_Y)
            transport = scipy.sparse.kron(spin_block, spinless, format="csr")
        else:
            transport = spinless

        if include_reciprocal_phase:
            phase = _target_group_reciprocal_phase_diagonal(self.structure, valley_ctx, target_group_shifts)
            transport = (phase @ transport).tocsr()
        else:
            diagnostics["phase_side"] = "omitted_for_D_g^(0)_export"

        ident = scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")
        delta = (transport.conj().T @ transport - ident).tocsr()
        diagnostics["transport_unitarity_residual"] = float(np.sqrt(np.sum(np.abs(delta.data) ** 2)) / np.sqrt(transport.shape[0]))
        t_sq = (transport @ transport.conj()).tocsr()
        expected = (-1.0 if getattr(self.structure, "spin", False) else 1.0) * ident
        t_sq_delta = (t_sq - expected).tocsr()
        diagnostics["t_square_residual"] = float(np.sqrt(np.sum(np.abs(t_sq_delta.data) ** 2)) / np.sqrt(transport.shape[0]))
        self._last_transport_diagnostics = diagnostics
        return transport

    def _build_gamma_c2y_transport(self, candidate: dict[str, Any], valley: int, q_target, q_source):
        valley_ctx = self._valley_context_for_valley(valley)
        q_target = np.asarray(q_target, dtype=float)
        q_source = np.asarray(q_source, dtype=float)
        rotation_cart = np.asarray(candidate["rotation_cart"], dtype=float)
        linear_map = rotation_cart[:2, :2]
        translation_cart = np.asarray(candidate.get("translation_cart", np.zeros(3, dtype=float)), dtype=float)
        removable, origin_shift_cart, origin_shift_residual = translation_removable_by_origin_shift(
            rotation_cart,
            translation_cart,
            tol=1.0e-8,
        )
        if not removable:
            raise SymmetrySupportError(
                "seitz_translation_unsupported",
                f"C2 v1 requires removable Seitz translation; got translation={translation_cart.tolist()} with origin-shift residual={origin_shift_residual:.3e}.",
            )

        q_target_cart = np.dot(q_target, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
        q_source_cart = np.dot(q_source, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
        q_consistency = float(np.linalg.norm(linear_map @ q_source_cart - q_target_cart))
        if q_consistency >= 1.0e-8:
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"C2 q consistency failed: ||R2 q_source - q_target||={q_consistency:.3e}.",
            )

        atom_mapping = _build_atom_mapping(self.structure, candidate)
        if atom_mapping["group_target_map"] != {0: 1, 1: 0}:
            raise SymmetrySupportError(
                "atom_mapping_missing",
                f"C2 must exchange the two physical layers, got group_target_map={atom_mapping['group_target_map']}.",
            )

        df = self.structure.df
        unique_groups = [0, 1]
        group_index = {group: idx for idx, group in enumerate(unique_groups)}
        spinless_blocks = [[None for _ in unique_groups] for _ in unique_groups]
        returned_spin_rep = None
        diagnostics = {
            "q_consistency": q_consistency,
            "g_perm_max_delta": 0.0,
            "nonzero_reciprocal_shift_count": 0,
            "layer_exchange_detected": True,
            "origin_shift_residual": origin_shift_residual,
        }

        for source_group in unique_groups:
            target_group = atom_mapping["group_target_map"][source_group]
            source_g = np.asarray(valley_ctx.group_g_vectors[source_group], dtype=float)
            target_g = np.asarray(valley_ctx.group_g_vectors[target_group], dtype=float)
            q_transport, max_delta = build_direct_linear_g_transport(
                source_g_vectors=source_g,
                target_g_vectors=target_g,
                linear_map=linear_map,
                tol=1.0e-8,
            )
            diagnostics["g_perm_max_delta"] = max(diagnostics["g_perm_max_delta"], max_delta)

            source_atom_types = sorted(df.loc[df["twist_group"] == source_group, "atom_type"].astype(int).unique().tolist())
            target_for_source = {
                atom_type: int(atom_mapping["atom_type_target_map"][atom_type])
                for atom_type in source_atom_types
            }
            group_data = _build_group_orbital_transport_data(
                structure_df=self.structure.df,
                source_group=source_group,
                target_group=target_group,
                rotation_matrix=rotation_cart,
                spin=bool(getattr(self.structure, "spin", False)),
                spin_rep=None,
                target_for_source=target_for_source,
            )
            if returned_spin_rep is None and group_data.spin_rep is not None:
                returned_spin_rep = group_data.spin_rep

            q_perm = np.empty(len(source_g), dtype=int)
            q_coo = q_transport.tocoo()
            q_perm[np.asarray(q_coo.col, dtype=int)] = np.asarray(q_coo.row, dtype=int)

            rows_parts = []
            cols_parts = []
            data_parts = []
            for source_g_index, target_g_index in enumerate(q_perm.tolist()):
                for source_atom_type in group_data.source_atom_types:
                    target_atom_type = target_for_source[source_atom_type]
                    source_offset, block_dim, _ = group_data.source_offsets[source_atom_type]
                    target_offset, _, _ = group_data.target_offsets[target_atom_type]
                    block = group_data.source_blocks[source_atom_type]
                    row_base = int(target_g_index * group_data.target_dim + target_offset)
                    col_base = int(source_g_index * group_data.source_dim + source_offset)
                    row_idx = row_base + np.arange(block_dim, dtype=np.int64)
                    col_idx = col_base + np.arange(block_dim, dtype=np.int64)
                    rows_parts.append(np.repeat(row_idx, block_dim))
                    cols_parts.append(np.tile(col_idx, block_dim))
                    data_parts.append(block.reshape(-1))

            block_shape = (len(target_g) * group_data.target_dim, len(source_g) * group_data.source_dim)
            block = scipy.sparse.coo_matrix(
                (
                    np.concatenate(data_parts).astype(np.complex128, copy=False),
                    (np.concatenate(rows_parts), np.concatenate(cols_parts)),
                ),
                shape=block_shape,
                dtype=np.complex128,
            ).tocsr()
            block.sort_indices()
            spinless_blocks[group_index[target_group]][group_index[source_group]] = block

        spinless = scipy.sparse.bmat(spinless_blocks, format="csr")
        if getattr(self.structure, "spin", False):
            if returned_spin_rep is None:
                raise SymmetrySupportError(
                    "spin_lift_unsupported",
                    "Spinful C2 did not yield a spin representation.",
                )
            transport = scipy.sparse.kron(scipy.sparse.csr_matrix(returned_spin_rep), spinless, format="csr")
            expected = -scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")
        else:
            transport = spinless
            expected = scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")

        delta_u = (transport.conj().T @ transport - scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")).tocsr()
        diagnostics["transport_unitarity_residual"] = float(np.sqrt(np.sum(np.abs(delta_u.data) ** 2)) / np.sqrt(transport.shape[0]))
        square = (transport @ transport).tocsr()
        delta_sq = (square - expected).tocsr()
        diagnostics["t_square_residual"] = float(np.sqrt(np.sum(np.abs(delta_sq.data) ** 2)) / np.sqrt(transport.shape[0]))
        self._last_transport_diagnostics = diagnostics
        return transport

    def _build_k_c2t_transport(self, candidate: dict[str, Any], valley: int, q_target, q_source):
        valley_ctx = self._valley_context_for_valley(valley)
        q_target = np.asarray(q_target, dtype=float)
        q_source = np.asarray(q_source, dtype=float)
        diagnostics = {
            "g_perm_max_delta": 0.0,
            "nonzero_reciprocal_shift_count": 0,
            "layer_exchange_detected": True,
        }
        spglib_operation = candidate.get("spglib_layer_exchange_c2")
        if spglib_operation is None:
            spatial_operations = candidate.get("spatial_operations") or collect_spglib_spatial_operations(self.structure)
            spglib_operation = find_layer_exchange_c2_spatial_operation(self.structure, spatial_operations)
        axis_choices = {
            "diag(-1,1)": np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float),
            "diag(1,-1)": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        }
        if spglib_operation is not None:
            axis_choices["spglib_layer_exchange_c2"] = np.asarray(spglib_operation["rotation_cart"], dtype=float)[:2, :2]

        unique_groups = [0, 1]
        for source_group in unique_groups:
            target_group = 1 - source_group
            source_g = np.asarray(valley_ctx.group_g_vectors[source_group], dtype=float)
            target_g = np.asarray(valley_ctx.group_g_vectors[target_group], dtype=float)
            axis_diagnostics = diagnose_k_c2t_axis_choices(
                source_g_vectors=source_g,
                target_g_vectors=target_g,
                source_k_center=np.asarray(valley_ctx.group_k_centers[source_group], dtype=float),
                target_k_center=np.asarray(valley_ctx.group_k_centers[target_group], dtype=float),
                axis_choices=axis_choices,
                tol=1.0e-8,
            )
            diagnostics[f"axis_scan_{source_group}_to_{target_group}"] = axis_diagnostics

        if spglib_operation is None:
            self._last_transport_diagnostics = diagnostics
            raise SymmetrySupportError(
                "g_mapping_missing",
                "K_C2T could not find a layer-exchange order-2 spatial operation from spglib.",
            )

        spglib_axis_closed = True
        for source_group in unique_groups:
            target_group = 1 - source_group
            axis_result = diagnostics[f"axis_scan_{source_group}_to_{target_group}"]["spglib_layer_exchange_c2"]
            if not (axis_result["one_to_one"] and axis_result["max_delta"] < 1.0e-8):
                spglib_axis_closed = False
                break
        if not spglib_axis_closed:
            self._last_transport_diagnostics = diagnostics
            raise SymmetrySupportError(
                "g_mapping_missing",
                "K_C2T opposite-layer G mapping does not close for the spglib layer-exchange C2 axis.",
            )

        rotation_cart = np.asarray(spglib_operation["rotation_cart"], dtype=float)
        linear_map = rotation_cart[:2, :2]
        q_target_cart = np.dot(q_target, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
        q_source_cart = np.dot(q_source, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
        q_consistency = float(np.linalg.norm((-linear_map) @ q_source_cart - q_target_cart))
        diagnostics["q_consistency"] = q_consistency
        diagnostics["selected_axis"] = "spglib_layer_exchange_c2"
        if q_consistency >= 1.0e-8:
            self._last_transport_diagnostics = diagnostics
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"K_C2T q consistency failed for spglib layer-exchange C2 axis: ||-C2 q_source - q_target||={q_consistency:.3e}.",
            )

        atom_mapping = _build_atom_mapping(self.structure, spglib_operation)
        if atom_mapping.get("group_target_map") != {0: 1, 1: 0}:
            self._last_transport_diagnostics = diagnostics
            raise SymmetrySupportError(
                "atom_mapping_missing",
                f"K_C2T spglib axis does not exchange the two layers: group_target_map={atom_mapping.get('group_target_map')}.",
            )

        df = self.structure.df
        group_index = {group: idx for idx, group in enumerate(unique_groups)}
        spinless_blocks = [[None for _ in unique_groups] for _ in unique_groups]
        spin_rep_override = spin_reps(rotation_cart) @ (1.0j * _SIGMA_Y) if getattr(self.structure, "spin", False) else None

        for source_group in unique_groups:
            target_group = 1 - source_group
            source_g = np.asarray(valley_ctx.group_g_vectors[source_group], dtype=float)
            target_g = np.asarray(valley_ctx.group_g_vectors[target_group], dtype=float)
            q_transport, max_delta = build_k_c2t_g_transport(
                source_g_vectors=source_g,
                target_g_vectors=target_g,
                source_k_center=np.asarray(valley_ctx.group_k_centers[source_group], dtype=float),
                target_k_center=np.asarray(valley_ctx.group_k_centers[target_group], dtype=float),
                linear_map=linear_map,
                tol=1.0e-8,
            )
            diagnostics["g_perm_max_delta"] = max(diagnostics["g_perm_max_delta"], max_delta)

            source_atom_types = sorted(df.loc[df["twist_group"] == source_group, "atom_type"].astype(int).unique().tolist())
            target_for_source = {
                atom_type: int(atom_mapping["atom_type_target_map"][atom_type])
                for atom_type in source_atom_types
            }
            group_data = _build_group_orbital_transport_data(
                structure_df=self.structure.df,
                source_group=source_group,
                target_group=target_group,
                rotation_matrix=rotation_cart,
                spin=bool(getattr(self.structure, "spin", False)),
                spin_rep=spin_rep_override,
                target_for_source=target_for_source,
            )

            q_perm = np.empty(len(source_g), dtype=int)
            q_coo = q_transport.tocoo()
            q_perm[np.asarray(q_coo.col, dtype=int)] = np.asarray(q_coo.row, dtype=int)

            rows_parts = []
            cols_parts = []
            data_parts = []
            for source_g_index, target_g_index in enumerate(q_perm.tolist()):
                for source_atom_type in group_data.source_atom_types:
                    target_atom_type = target_for_source[source_atom_type]
                    source_offset, block_dim, _ = group_data.source_offsets[source_atom_type]
                    target_offset, _, _ = group_data.target_offsets[target_atom_type]
                    block = group_data.source_blocks[source_atom_type]
                    row_base = int(target_g_index * group_data.target_dim + target_offset)
                    col_base = int(source_g_index * group_data.source_dim + source_offset)
                    row_idx = row_base + np.arange(block_dim, dtype=np.int64)
                    col_idx = col_base + np.arange(block_dim, dtype=np.int64)
                    rows_parts.append(np.repeat(row_idx, block_dim))
                    cols_parts.append(np.tile(col_idx, block_dim))
                    data_parts.append(block.reshape(-1))

            block_shape = (len(target_g) * group_data.target_dim, len(source_g) * group_data.source_dim)
            block = scipy.sparse.coo_matrix(
                (
                    np.concatenate(data_parts).astype(np.complex128, copy=False),
                    (np.concatenate(rows_parts), np.concatenate(cols_parts)),
                ),
                shape=block_shape,
                dtype=np.complex128,
            ).tocsr()
            block.sort_indices()
            spinless_blocks[group_index[target_group]][group_index[source_group]] = block

        spinless = scipy.sparse.bmat(spinless_blocks, format="csr")
        if getattr(self.structure, "spin", False):
            transport = scipy.sparse.kron(scipy.sparse.csr_matrix(spin_rep_override), spinless, format="csr")
            expected = scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")
        else:
            transport = spinless
            expected = scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")

        delta_u = (transport.conj().T @ transport - scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")).tocsr()
        diagnostics["transport_unitarity_residual"] = float(np.sqrt(np.sum(np.abs(delta_u.data) ** 2)) / np.sqrt(transport.shape[0]))
        square = (transport @ transport.conj()).tocsr()
        delta_sq = (square - expected).tocsr()
        diagnostics["t_square_residual"] = float(np.sqrt(np.sum(np.abs(delta_sq.data) ** 2)) / np.sqrt(transport.shape[0]))
        self._last_transport_diagnostics = diagnostics
        return transport

    def _select_k_c2t_spatial_operation(
        self,
        valley: int,
        spatial_operations: list[dict[str, Any]],
        tolerance: float,
    ):
        q0 = np.zeros(3, dtype=float)
        candidates = list_layer_exchange_c2_spatial_operations(self.structure, spatial_operations)
        scan: list[dict[str, Any]] = []
        best: tuple[float, dict[str, Any]] | None = None
        h0 = None
        for operation in candidates:
            candidate = {
                "index": int(operation["index"]),
                "name": "K_C2T",
                "antiunitary": True,
                "closed": True,
                "rotation_frac": np.asarray(operation.get("rotation_frac", np.eye(3, dtype=float)), dtype=float),
                "translation_frac": np.asarray(operation.get("translation_frac", np.zeros(3, dtype=float)), dtype=float),
                "rotation_cart": np.asarray(operation["rotation_cart"], dtype=float),
                "translation_cart": np.asarray(operation.get("translation_cart", np.zeros(3, dtype=float)), dtype=float),
                "spatial_operations": spatial_operations,
                "spglib_layer_exchange_c2": operation,
            }
            try:
                transport = self._build_k_c2t_transport(candidate, valley, q0, q0)
                diagnostics = dict(getattr(self, "_last_transport_diagnostics", {}) or {})
                if h0 is None:
                    h0, _ = self._raw_projected_hs(valley, q0)
                residual = float(
                    frobenius_relative_residual(
                        h0,
                        transport @ h0.conj() @ transport.conj().T,
                        denominator=h0,
                    )
                )
                scan.append(
                    {
                        "spglib_index": int(operation["index"]),
                        "supported": True,
                        "residual_H_raw_gamma": residual,
                        "status": classify_residual(residual, tolerance),
                        "g_perm_max_delta": diagnostics.get("g_perm_max_delta"),
                        "t_square_residual": diagnostics.get("t_square_residual"),
                    }
                )
                if best is None or residual < best[0]:
                    best = (residual, operation)
            except SymmetrySupportError as exc:
                diagnostics = dict(getattr(self, "_last_transport_diagnostics", {}) or {})
                scan.append(
                    {
                        "spglib_index": int(operation["index"]),
                        "supported": False,
                        "not_supported_reason": exc.reason,
                        "g_perm_max_delta": diagnostics.get("g_perm_max_delta"),
                    }
                )

        self._last_k_c2t_operation_scan = scan
        if best is None:
            return None
        return best[1]

    def _select_m_c2_eta_spatial_operation(
        self,
        valley: int,
        spatial_operations: list[dict[str, Any]],
    ):
        if int(valley) not in {31, 32, 33}:
            return None
        valley_ctx = self._valley_context_for_valley(valley)
        scan: list[dict[str, Any]] = []
        for operation in list_layer_exchange_c2_spatial_operations(self.structure, spatial_operations):
            rotation_cart = np.asarray(operation["rotation_cart"], dtype=float)
            translation_cart = np.asarray(operation.get("translation_cart", np.zeros(3, dtype=float)), dtype=float)
            entry: dict[str, Any] = {
                "spglib_index": int(operation["index"]),
                "supported": False,
                "g_perm_max_delta": None,
            }
            if np.linalg.norm(translation_cart[:2]) > 1.0e-8:
                entry["not_supported_reason"] = "seitz_translation_unsupported"
                scan.append(entry)
                continue

            try:
                atom_mapping = _build_atom_mapping(self.structure, operation)
                if atom_mapping.get("group_target_map") != {0: 1, 1: 0}:
                    raise SymmetrySupportError(
                        "atom_mapping_missing",
                        f"M_C2_eta operation {operation['index']} does not exchange layers.",
                    )
                max_delta = 0.0
                for source_group, target_group in ((0, 1), (1, 0)):
                    g_diag = _diagnose_affine_g_mapping(
                        source_g_vectors=valley_ctx.group_g_vectors[source_group],
                        target_g_vectors=valley_ctx.group_g_vectors[target_group],
                        source_k_center=valley_ctx.group_k_centers[source_group],
                        target_k_center=valley_ctx.group_k_centers[target_group],
                        linear_map=rotation_cart[:2, :2],
                        antiunitary_like=False,
                        tol=1.0e-8,
                    )
                    max_delta = max(max_delta, float(g_diag["max_delta"]))
                    if not (g_diag["one_to_one"] and g_diag["max_delta"] < 1.0e-8):
                        raise SymmetrySupportError(
                            "spglib_operation_not_valley_closed",
                            f"M_C2_eta operation {operation['index']} is not K-center G-closed.",
                        )
                shifts, shift_residual = m_unitary_c2_target_group_shifts(
                    group_centers=valley_ctx.group_k_centers,
                    group_target_map={0: 1, 1: 0},
                    linear_map=rotation_cart[:2, :2],
                    reciprocal_basis=valley_ctx.moire_reciprocal_basis,
                    tol=1.0e-8,
                )
                entry.update(
                    {
                        "supported": True,
                        "g_perm_max_delta": max_delta,
                        "reciprocal_shift_residual_max": shift_residual,
                        "nonzero_reciprocal_shift_count": sum(
                            1 for coeffs in shifts.values() if np.any(np.asarray(coeffs, dtype=int) != 0)
                        ),
                    }
                )
                scan.append(entry)
                self._last_m_c2_eta_operation_scan = scan
                return operation
            except SymmetrySupportError as exc:
                entry["not_supported_reason"] = exc.reason
                scan.append(entry)
        self._last_m_c2_eta_operation_scan = scan
        return None

    def _build_m_c2_eta_transport(self, candidate: dict[str, Any], valley: int, q_target, q_source):
        valley_ctx = self._valley_context_for_valley(valley)
        q_target = np.asarray(q_target, dtype=float)
        q_source = np.asarray(q_source, dtype=float)
        params = valley_ctx.calculator.TAPW_parameters
        spatial_operations = candidate.get("spatial_operations") or collect_spglib_spatial_operations(self.structure)
        resolved = candidate.get("resolved_m_c2_symmetry")
        if resolved is None:
            resolved = resolve_reference_m_valley_c2_symmetry(self.structure, params)

        m_k1 = np.asarray(getattr(params, "m_K1", None), dtype=float) if getattr(params, "m_K1", None) is not None else None
        m_k2 = np.asarray(getattr(params, "m_K2", None), dtype=float) if getattr(params, "m_K2", None) is not None else None
        if m_k1 is None or m_k2 is None:
            _, _, m_k1, m_k2, _ = params.calculate_K_points()
        axis_vec = np.asarray(m_k1, dtype=float) + np.asarray(m_k2, dtype=float)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm < 1.0e-12:
            raise SymmetrySupportError(
                "q_mapping_missing",
                "M_C2_eta cannot determine the reference unitary C2 axis because m_K1 + m_K2 is numerically zero.",
            )
        axis_vec = axis_vec / axis_norm
        reference_rotation_cart = np.asarray(rotate_mat(np.array([axis_vec[0], axis_vec[1], 0.0]), np.pi), dtype=float)
        reference_linear_map = np.asarray(reference_m_valley_c2_linear_map(params), dtype=float)
        spglib_rotation_cart = np.asarray(resolved.rotation_cart, dtype=float)
        spglib_linear_map = np.asarray(resolved.linear_map_2d, dtype=float)

        def _build_consistent_transport(
            *,
            path_name: str,
            rotation_cart: np.ndarray,
            linear_map: np.ndarray,
            atom_mapping,
            not_structure_reason: str | None,
            not_valley_reason: str | None,
            center_by_group: dict[int, np.ndarray] | None = None,
            apply_target_reciprocal_phase: bool = False,
        ):
            local_diag = {
                "path": path_name,
                "g_perm_max_delta": 0.0,
                "nonzero_reciprocal_shift_count": 0,
                "layer_exchange_detected": True,
            }
            if center_by_group is None:
                center_by_group = valley_ctx.group_m_k_centers
            if atom_mapping is None:
                raise SymmetrySupportError(
                    not_structure_reason or "atom_mapping_missing",
                    f"{path_name} atom mapping is unavailable.",
                )
            if atom_mapping.get("group_target_map") != {0: 1, 1: 0}:
                raise SymmetrySupportError(
                    not_structure_reason or "atom_mapping_missing",
                    f"{path_name} does not exchange the two layers: group_target_map={atom_mapping.get('group_target_map')}.",
                )

            q_target_cart = np.dot(q_target, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
            q_source_cart = np.dot(q_source, np.asarray(self.structure.reciprocal_Tmat, dtype=float))[:2]
            q_consistency = float(np.linalg.norm(linear_map @ q_source_cart - q_target_cart))
            local_diag["q_consistency"] = q_consistency
            if q_consistency >= 1.0e-8:
                self._last_transport_diagnostics = local_diag
                raise SymmetrySupportError(
                    "q_mapping_missing",
                    f"{path_name} q consistency failed: ||R q_source - q_target||={q_consistency:.3e}.",
                )

            unique_groups = [0, 1]
            group_index = {group: idx for idx, group in enumerate(unique_groups)}
            spinless_blocks = [[None for _ in unique_groups] for _ in unique_groups]
            returned_spin_rep = None
            spin_rep_override = unitary_m_c2_spin_rep(rotation_cart) if getattr(self.structure, "spin", False) else None

            for source_group in unique_groups:
                target_group = 1 - source_group
                source_g = np.asarray(valley_ctx.group_g_vectors[source_group], dtype=float)
                target_g = np.asarray(valley_ctx.group_g_vectors[target_group], dtype=float)
                g_diag = _diagnose_affine_g_mapping(
                    source_g_vectors=source_g,
                    target_g_vectors=target_g,
                    source_k_center=np.asarray(center_by_group[source_group], dtype=float),
                    target_k_center=np.asarray(center_by_group[target_group], dtype=float),
                    linear_map=linear_map,
                    antiunitary_like=False,
                    tol=1.0e-8,
                )
                local_diag[f"g_mapping_{source_group}_to_{target_group}"] = g_diag
                if not (g_diag["one_to_one"] and g_diag["max_delta"] < 1.0e-8):
                    self._last_transport_diagnostics = local_diag
                    raise SymmetrySupportError(
                        not_valley_reason or "g_mapping_missing",
                        f"{path_name} is not valley/Gtilde-closed for group {source_group}->{target_group}: "
                        f"max_delta={g_diag['max_delta']:.3e}, unmatched={g_diag['unmatched_count']}, "
                        f"repeated={g_diag['repeated_target_count']}.",
                    )

                q_transport, max_delta = build_m_c2_eta_g_transport(
                    source_g_vectors=source_g,
                    target_g_vectors=target_g,
                    source_k_center=np.asarray(center_by_group[source_group], dtype=float),
                    target_k_center=np.asarray(center_by_group[target_group], dtype=float),
                    linear_map=linear_map,
                    tol=1.0e-8,
                )
                local_diag["g_perm_max_delta"] = max(local_diag["g_perm_max_delta"], max_delta)

                source_atom_types = sorted(
                    self.structure.df.loc[self.structure.df["twist_group"] == source_group, "atom_type"].astype(int).unique().tolist()
                )
                target_for_source = {
                    atom_type: int(atom_mapping["atom_type_target_map"][atom_type])
                    for atom_type in source_atom_types
                }
                group_data = _build_group_orbital_transport_data(
                    structure_df=self.structure.df,
                    source_group=source_group,
                    target_group=target_group,
                    rotation_matrix=rotation_cart,
                    spin=bool(getattr(self.structure, "spin", False)),
                    spin_rep=spin_rep_override,
                    target_for_source=target_for_source,
                )
                if returned_spin_rep is None and group_data.spin_rep is not None:
                    returned_spin_rep = group_data.spin_rep

                q_perm = np.empty(len(source_g), dtype=int)
                q_coo = q_transport.tocoo()
                q_perm[np.asarray(q_coo.col, dtype=int)] = np.asarray(q_coo.row, dtype=int)

                rows_parts = []
                cols_parts = []
                data_parts = []
                for source_g_index, target_g_index in enumerate(q_perm.tolist()):
                    for source_atom_type in group_data.source_atom_types:
                        target_atom_type = target_for_source[source_atom_type]
                        source_offset, block_dim, _ = group_data.source_offsets[source_atom_type]
                        target_offset, _, _ = group_data.target_offsets[target_atom_type]
                        block = group_data.source_blocks[source_atom_type]
                        row_base = int(target_g_index * group_data.target_dim + target_offset)
                        col_base = int(source_g_index * group_data.source_dim + source_offset)
                        row_idx = row_base + np.arange(block_dim, dtype=np.int64)
                        col_idx = col_base + np.arange(block_dim, dtype=np.int64)
                        rows_parts.append(np.repeat(row_idx, block_dim))
                        cols_parts.append(np.tile(col_idx, block_dim))
                        data_parts.append(block.reshape(-1))

                block_shape = (len(target_g) * group_data.target_dim, len(source_g) * group_data.source_dim)
                block = scipy.sparse.coo_matrix(
                    (
                        np.concatenate(data_parts).astype(np.complex128, copy=False),
                        (np.concatenate(rows_parts), np.concatenate(cols_parts)),
                    ),
                    shape=block_shape,
                    dtype=np.complex128,
                ).tocsr()
                block.sort_indices()
                spinless_blocks[group_index[target_group]][group_index[source_group]] = block

            transport = scipy.sparse.bmat(spinless_blocks, format="csr")
            if getattr(self.structure, "spin", False):
                if returned_spin_rep is None:
                    raise SymmetrySupportError(
                        "spin_lift_unsupported",
                        f"{path_name} did not yield a unitary spin representation.",
                    )
                transport = scipy.sparse.kron(scipy.sparse.csr_matrix(returned_spin_rep), transport, format="csr")
                expected = -scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")
            else:
                expected = scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")

            if apply_target_reciprocal_phase:
                target_group_shifts, shift_residual = m_unitary_c2_target_group_shifts(
                    group_centers=center_by_group,
                    group_target_map={0: 1, 1: 0},
                    linear_map=linear_map,
                    reciprocal_basis=valley_ctx.moire_reciprocal_basis,
                    tol=1.0e-8,
                )
                local_diag["reciprocal_shift_residual_max"] = shift_residual
                local_diag["nonzero_reciprocal_shift_count"] = sum(
                    1 for coeffs in target_group_shifts.values() if np.any(np.asarray(coeffs, dtype=int) != 0)
                )
                local_diag["center_convention"] = "group_k_centers"
                local_diag["target_group_shift_coeffs"] = {
                    int(group): np.asarray(coeffs, dtype=int)
                    for group, coeffs in target_group_shifts.items()
                }
                local_diag["phase_sign"] = "+1"
                local_diag["phase_side"] = "left_target_rows"
                phase = _target_group_reciprocal_phase_diagonal(
                    self.structure,
                    valley_ctx,
                    target_group_shifts,
                )
                transport = (phase @ transport).tocsr()

            delta_u = (transport.conj().T @ transport - scipy.sparse.identity(transport.shape[0], dtype=np.complex128, format="csr")).tocsr()
            local_diag["transport_unitarity_residual"] = float(np.sqrt(np.sum(np.abs(delta_u.data) ** 2)) / np.sqrt(transport.shape[0]))
            square = (transport @ transport).tocsr()
            delta_sq = (square - expected).tocsr()
            local_diag["t_square_residual"] = float(np.sqrt(np.sum(np.abs(delta_sq.data) ** 2)) / np.sqrt(transport.shape[0]))
            self._last_transport_diagnostics = local_diag
            return transport, local_diag

        preferred_operation = candidate.get("spglib_m_c2_operation")
        preferred_exc = None
        preferred_transport = None
        preferred_diag = {}
        if preferred_operation is not None:
            try:
                preferred_atom_mapping = _build_atom_mapping(self.structure, preferred_operation)
                preferred_transport, preferred_diag = _build_consistent_transport(
                    path_name=f"spglib_layer_exchange_C2_K_op{int(preferred_operation['index'])}",
                    rotation_cart=np.asarray(preferred_operation["rotation_cart"], dtype=float),
                    linear_map=np.asarray(preferred_operation["rotation_cart"], dtype=float)[:2, :2],
                    atom_mapping=preferred_atom_mapping,
                    not_structure_reason="atom_mapping_missing",
                    not_valley_reason="spglib_operation_not_valley_closed",
                    center_by_group=valley_ctx.group_k_centers,
                    apply_target_reciprocal_phase=not bool(candidate.get("export_pure_d0", False)),
                )
                self._last_transport_diagnostics = preferred_diag | {
                    "preferred_spglib_path": preferred_diag,
                    "selected_path": preferred_diag.get("path"),
                }
                return preferred_transport
            except SymmetrySupportError as exc:
                preferred_exc = exc
                preferred_diag = dict(getattr(self, "_last_transport_diagnostics", {}) or {})

        reference_match, reference_atom_mapping = find_matching_spatial_operation_for_rotation(
            self.structure,
            spatial_operations,
            reference_rotation_cart,
            require_layer_exchange=True,
        )
        reference_exc = None
        reference_transport = None
        reference_diag = {
            "reference_linear_map_delta": float(np.linalg.norm(reference_rotation_cart[:2, :2] - reference_linear_map)),
            "matched_spglib_rotation_delta": None if reference_match is None else float(np.linalg.norm(np.asarray(reference_match["rotation_cart"], dtype=float) - reference_rotation_cart)),
        }
        try:
            reference_transport, reference_build_diag = _build_consistent_transport(
                path_name="reference_R",
                rotation_cart=reference_rotation_cart,
                linear_map=reference_linear_map,
                atom_mapping=reference_atom_mapping,
                not_structure_reason="reference_operation_not_structure_symmetry",
                not_valley_reason="g_mapping_missing",
            )
            reference_diag.update(reference_build_diag)
        except SymmetrySupportError as exc:
            reference_exc = exc
            reference_diag.update(dict(getattr(self, "_last_transport_diagnostics", {}) or {}))

        spglib_operation = {
            "rotation_cart": spglib_rotation_cart,
            "translation_cart": np.asarray(resolved.translation_cart, dtype=float),
            "rotation_frac": np.asarray(resolved.rotation_frac, dtype=float),
            "translation_frac": np.asarray(resolved.translation_frac, dtype=float),
        }
        spglib_atom_mapping = {
            "group_target_map": {0: 1, 1: 0},
            "atom_type_target_map": {
                **{int(k): int(v) for k, v in resolved.atom_type_map_0_to_1.items()},
                **{int(k): int(v) for k, v in resolved.atom_type_map_1_to_0.items()},
            },
        }
        spglib_exc = None
        spglib_transport = None
        spglib_diag = {
            "spglib_reference_rotation_delta": float(np.linalg.norm(spglib_rotation_cart - reference_rotation_cart)),
        }
        try:
            spglib_transport, spglib_build_diag = _build_consistent_transport(
                path_name="spglib_R",
                rotation_cart=spglib_rotation_cart,
                linear_map=spglib_linear_map,
                atom_mapping=spglib_atom_mapping,
                not_structure_reason="atom_mapping_missing",
                not_valley_reason="spglib_operation_not_valley_closed",
            )
            spglib_diag.update(spglib_build_diag)
        except SymmetrySupportError as exc:
            spglib_exc = exc
            spglib_diag.update(dict(getattr(self, "_last_transport_diagnostics", {}) or {}))

        combined = {
            "preferred_spglib_path": preferred_diag,
            "reference_path": reference_diag,
            "spglib_path": spglib_diag,
        }
        if reference_transport is not None:
            self._last_transport_diagnostics = combined | {"selected_path": "reference_R"}
            return reference_transport
        if spglib_transport is not None:
            self._last_transport_diagnostics = combined | {"selected_path": "spglib_R"}
            return spglib_transport

        self._last_transport_diagnostics = combined
        if preferred_exc is not None:
            raise preferred_exc
        if reference_exc is not None:
            raise reference_exc
        if spglib_exc is not None:
            raise spglib_exc
        raise SymmetrySupportError("atom_mapping_missing", "M_C2_eta could not build any consistent transport path.")

    def _build_transport(self, candidate: dict[str, Any], valley: int, q_target, q_source):
        if not hasattr(self, "_transport_cache"):
            self._transport_cache = {}
        if not hasattr(self, "_operation_mapping_cache"):
            self._operation_mapping_cache = {}
        q_target = np.asarray(q_target, dtype=float)
        q_source = np.asarray(q_source, dtype=float)
        cache_key = (
            int(valley),
            int(candidate["index"]),
            bool(candidate.get("antiunitary", False)),
            bool(candidate.get("export_pure_d0", False)),
            _rounded_q_key(q_target),
            _rounded_q_key(q_source),
        )
        cached = self._transport_cache.get(cache_key)
        if cached is not None:
            return cached

        op_key = (int(candidate["index"]), bool(candidate.get("antiunitary", False)))
        calculator = self._calculator_for_valley(valley)
        projected_dim = int(calculator.TAPW_parameters.g_matrix.shape[0])

        if candidate.get("name") == "E" and not candidate.get("antiunitary", False):
            transport = scipy.sparse.identity(projected_dim, dtype=np.complex128, format="csr")
            self._transport_cache[cache_key] = transport
            return transport

        if candidate.get("name") == "T" and candidate.get("antiunitary", False):
            transport = self._build_time_reversal_transport(
                valley,
                q_target,
                q_source,
                include_m_reciprocal_phase=not bool(candidate.get("export_pure_d0", False)),
            )
            self._transport_cache[cache_key] = transport
            return transport

        if candidate.get("name") == "K_C2T" and candidate.get("antiunitary", False):
            transport = self._build_k_c2t_transport(candidate, valley, q_target, q_source)
            self._transport_cache[cache_key] = transport
            return transport

        if not candidate.get("antiunitary", False) and candidate.get("name") in {"C3z", "C3z^2"}:
            power = 1 if candidate.get("name") == "C3z" else 2
            generic_transport = self._validated_generic_c3_transport(valley, power)
            self._transport_cache[cache_key] = generic_transport
            return generic_transport

        if not candidate.get("antiunitary", False) and candidate.get("name") == "C2":
            transport = self._build_gamma_c2y_transport(candidate, valley, q_target, q_source)
            self._transport_cache[cache_key] = transport
            return transport

        if not candidate.get("antiunitary", False) and candidate.get("name") == "M_C2_eta":
            transport = self._build_m_c2_eta_transport(candidate, valley, q_target, q_source)
            self._transport_cache[cache_key] = transport
            return transport

        if not candidate.get("antiunitary", False) and str(candidate.get("name", "")).startswith("C2(axis="):
            raise SymmetrySupportError(
                "atom_mapping_missing",
                f"Generic C2 transport is disabled; only the dedicated Gamma C2 path is enabled.",
            )

        raise SymmetrySupportError(
            "atom_mapping_missing",
            f"Operation {candidate['name']} is not enabled in the current minimal reliable implementation.",
        )

    def _map_inverse_q(self, candidate: dict[str, Any], q_target):
        linear_map = np.asarray(
            candidate.get("linear_map_2d", np.asarray(candidate["rotation_cart"], dtype=float)[:2, :2]),
            dtype=float,
        )
        source_linear_map = -linear_map if bool(candidate.get("antiunitary", False)) else linear_map
        try:
            source_inverse = np.linalg.inv(source_linear_map)
        except np.linalg.LinAlgError as exc:
            raise SymmetrySupportError("q_mapping_missing", f"Operation linear map is singular: {exc}") from exc
        return transform_k_by_cartesian_linear_map(
            np.asarray(q_target, dtype=float),
            self.structure.reciprocal_Tmat,
            source_inverse,
        )

    def _make_detail_row(
        self,
        *,
        valley_label: str,
        operation_name: str,
        operation_type: str,
        spglib_index,
        r_2d: str,
        k_label: str,
        q_local,
        supported: bool,
        not_supported_reason: str,
        residual_h_raw,
        residual_s_raw,
        residual_h_sym,
        residual_s_sym,
        residual_lowdin_order,
        diagnostic_g_perm_max_delta=None,
        diagnostic_nonzero_reciprocal_shift_count=None,
        diagnostic_transport_unitarity_residual=None,
        diagnostic_t_square_residual=None,
        debug: dict[str, Any] | None = None,
        tolerance: float,
    ):
        covariance_status = "not_supported"
        if supported or residual_h_raw is not None or residual_s_raw is not None:
            covariance_status = combine_statuses(
                [
                    classify_residual(float(value), tolerance)
                    for value in (residual_h_raw, residual_s_raw)
                    if value is not None
                ]
            ) if any(value is not None for value in (residual_h_raw, residual_s_raw)) else "not_supported"

        symmetrization_status = "not_supported"
        if supported or residual_h_sym is not None or residual_s_sym is not None:
            symmetrization_status = combine_statuses(
                [
                    classify_residual(float(value), tolerance)
                    for value in (residual_h_sym, residual_s_sym)
                    if value is not None
                ]
            ) if any(value is not None for value in (residual_h_sym, residual_s_sym)) else "not_supported"

        lowdin_status = "not_computed"
        if residual_lowdin_order is not None:
            lowdin_status = classify_residual(float(residual_lowdin_order), tolerance)

        statuses = [status for status in (covariance_status,) if status != "not_supported"]
        status = "not_supported" if (not supported and not statuses) else (combine_statuses(statuses) if statuses else "not_supported")
        return {
            "valley": valley_label,
            "operation": displayed_operation_name(operation_name),
            "operation_type": operation_type,
            "spglib_index": spglib_index,
            "R_2d": r_2d,
            "k_label": k_label,
            "k_frac": _format_k_frac(q_local),
            "supported": bool(supported),
            "not_supported_reason": not_supported_reason,
            "covariance_status": covariance_status,
            "symmetrization_status": symmetrization_status,
            "lowdin_status": lowdin_status,
            "residual_H_raw": residual_h_raw,
            "residual_S_raw": residual_s_raw,
            "residual_H_sym": residual_h_sym,
            "residual_S_sym": residual_s_sym,
            "residual_lowdin_order": residual_lowdin_order,
            "g_perm_max_delta": diagnostic_g_perm_max_delta,
            "nonzero_reciprocal_shift_count": diagnostic_nonzero_reciprocal_shift_count,
            "square_residual": diagnostic_t_square_residual,
            "diagnostic_g_perm_max_delta": diagnostic_g_perm_max_delta,
            "diagnostic_nonzero_reciprocal_shift_count": diagnostic_nonzero_reciprocal_shift_count,
            "diagnostic_transport_unitarity_residual": diagnostic_transport_unitarity_residual,
            "diagnostic_t_square_residual": diagnostic_t_square_residual,
            "status": status,
            "debug": debug or {},
        }

    def _candidate_rows_for_q(self, candidate, valley: int, valley_label: str, q_label: str, q_target, tolerance: float):
        q_target = np.asarray(q_target, dtype=float)
        diagnostics = {}
        self._last_transport_diagnostics = {}
        rotation_cart = np.asarray(candidate.get("rotation_cart", np.eye(3, dtype=float)), dtype=float)
        r_2d = _format_matrix_json(rotation_cart[:2, :2])
        spglib_index = _candidate_spglib_index(candidate)
        base_debug = {
            "R_3d": _format_vector_json(rotation_cart),
            "translation_frac": _format_vector_json(candidate.get("translation_frac")),
            "center_convention": candidate.get("center_convention"),
            "phase_sign": candidate.get("phase_sign"),
            "phase_side": candidate.get("phase_side"),
        }
        if not candidate.get("closed", False):
            return self._make_detail_row(
                valley_label=valley_label,
                operation_name=candidate["name"],
                operation_type="antiunitary" if candidate.get("antiunitary", False) else "unitary",
                spglib_index=spglib_index,
                r_2d=r_2d,
                k_label=q_label,
                q_local=q_target,
                supported=False,
                not_supported_reason=candidate.get("closure_reason", "valley_not_closed"),
                residual_h_raw=None,
                residual_s_raw=None,
                residual_h_sym=None,
                residual_s_sym=None,
                residual_lowdin_order=None,
                diagnostic_g_perm_max_delta=None,
                diagnostic_nonzero_reciprocal_shift_count=None,
                diagnostic_transport_unitarity_residual=None,
                diagnostic_t_square_residual=None,
                debug={k: v for k, v in base_debug.items() if v is not None},
                tolerance=tolerance,
            )

        if not candidate.get("antiunitary", False) and candidate.get("name") in {"C3z", "C3z^2"}:
            try:
                power = 1 if candidate.get("name") == "C3z" else 2
                transport = self._build_transport(candidate, valley, q_target, q_target)
                diagnostics = dict(getattr(self, "_last_transport_diagnostics", {}) or {})
                h_orbit = self._raw_c3_h_orbit(valley, q_target)
            except SymmetrySupportError as exc:
                return self._make_detail_row(
                    valley_label=valley_label,
                    operation_name=candidate["name"],
                    operation_type="unitary",
                    spglib_index=spglib_index,
                    r_2d=r_2d,
                    k_label=q_label,
                    q_local=q_target,
                    supported=False,
                    not_supported_reason=exc.reason,
                    residual_h_raw=None,
                    residual_s_raw=None,
                    residual_h_sym=None,
                    residual_s_sym=None,
                    residual_lowdin_order=None,
                    diagnostic_g_perm_max_delta=diagnostics.get("g_perm_max_delta"),
                    diagnostic_nonzero_reciprocal_shift_count=diagnostics.get("nonzero_reciprocal_shift_count"),
                    diagnostic_transport_unitarity_residual=diagnostics.get("transport_unitarity_residual"),
                    diagnostic_t_square_residual=diagnostics.get("t_square_residual"),
                    debug=_build_detail_debug_payload(base_debug, diagnostics),
                    tolerance=tolerance,
                )

            h_target = h_orbit[0]
            h_source = h_orbit[power]
            h_cov = transport @ h_source @ transport.conj().T
            residual_h_raw = float(frobenius_relative_residual(h_target, h_cov, denominator=h_target))
            return self._make_detail_row(
                valley_label=valley_label,
                operation_name=candidate["name"],
                operation_type="unitary",
                spglib_index=spglib_index,
                r_2d=r_2d,
                k_label=q_label,
                q_local=q_target,
                supported=True,
                not_supported_reason="",
                residual_h_raw=residual_h_raw,
                residual_s_raw=None,
                residual_h_sym=None,
                residual_s_sym=None,
                residual_lowdin_order=None,
                diagnostic_g_perm_max_delta=diagnostics.get("g_perm_max_delta"),
                diagnostic_nonzero_reciprocal_shift_count=diagnostics.get("nonzero_reciprocal_shift_count"),
                diagnostic_transport_unitarity_residual=diagnostics.get("transport_unitarity_residual"),
                diagnostic_t_square_residual=diagnostics.get("t_square_residual"),
                debug=_build_detail_debug_payload(base_debug, diagnostics),
                tolerance=tolerance,
            )

        try:
            q_source = self._map_inverse_q(candidate, q_target)
            transport = self._build_transport(candidate, valley, q_target, q_source)
            diagnostics = dict(getattr(self, "_last_transport_diagnostics", {}) or {})
        except SymmetrySupportError as exc:
            return self._make_detail_row(
                valley_label=valley_label,
                operation_name=candidate["name"],
                operation_type="antiunitary" if candidate.get("antiunitary", False) else "unitary",
                spglib_index=spglib_index,
                r_2d=r_2d,
                k_label=q_label,
                q_local=q_target,
                supported=False,
                not_supported_reason=exc.reason,
                residual_h_raw=None,
                residual_s_raw=None,
                residual_h_sym=None,
                residual_s_sym=None,
                residual_lowdin_order=None,
                diagnostic_g_perm_max_delta=diagnostics.get("g_perm_max_delta"),
                diagnostic_nonzero_reciprocal_shift_count=diagnostics.get("nonzero_reciprocal_shift_count"),
                diagnostic_transport_unitarity_residual=diagnostics.get("transport_unitarity_residual"),
                diagnostic_t_square_residual=diagnostics.get("t_square_residual"),
                debug=_build_detail_debug_payload(base_debug, diagnostics),
                tolerance=tolerance,
            )

        h_target, _ = self._raw_projected_hs(valley, q_target)
        h_source, _ = self._raw_projected_hs(valley, q_source)
        antiunitary = bool(candidate.get("antiunitary", False))
        if antiunitary:
            h_cov = transport @ h_source.conj() @ transport.conj().T
        else:
            h_cov = transport @ h_source @ transport.conj().T

        residual_h_raw = float(frobenius_relative_residual(h_target, h_cov, denominator=h_target))
        residual_s_raw = None

        residual_h_sym = None
        residual_s_sym = None
        residual_lowdin_order = None
        supported = True
        not_supported_reason = ""

        return self._make_detail_row(
            valley_label=valley_label,
            operation_name=candidate["name"],
            operation_type="antiunitary" if antiunitary else "unitary",
            spglib_index=spglib_index,
            r_2d=r_2d,
            k_label=q_label,
            q_local=q_target,
            supported=supported,
            not_supported_reason=not_supported_reason,
            residual_h_raw=residual_h_raw,
            residual_s_raw=residual_s_raw,
            residual_h_sym=residual_h_sym,
            residual_s_sym=residual_s_sym,
            residual_lowdin_order=residual_lowdin_order,
            diagnostic_g_perm_max_delta=diagnostics.get("g_perm_max_delta"),
            diagnostic_nonzero_reciprocal_shift_count=diagnostics.get("nonzero_reciprocal_shift_count"),
            diagnostic_transport_unitarity_residual=diagnostics.get("transport_unitarity_residual"),
            diagnostic_t_square_residual=diagnostics.get("t_square_residual"),
            debug=_build_detail_debug_payload(base_debug, diagnostics),
            tolerance=tolerance,
        )

    def _representation_record_for_candidate(
        self,
        *,
        candidate: dict[str, Any],
        valley: int,
        valley_label: str,
        valley_ctx: ValleyContext,
        candidate_rows: list[dict[str, Any]],
    ):
        operation = representation_operation_name(candidate["name"])
        if not _is_saved_minimal_generator(operation, set()):
            return None

        q_target = np.zeros(3, dtype=float)
        q_source = self._map_inverse_q(candidate, q_target)
        raw_transport = self._build_transport(candidate, valley, q_target, q_source)
        raw_transport_diagnostics = dict(getattr(self, "_last_transport_diagnostics", {}) or {})
        export_candidate = dict(candidate)
        export_candidate["export_pure_d0"] = True
        transport = self._build_transport(export_candidate, valley, q_target, q_source)
        if not scipy.sparse.issparse(transport) or not scipy.sparse.issparse(raw_transport):
            raise SymmetrySupportError(
                "q_mapping_missing",
                f"Representation matrix for {operation} is not sparse; dense transport saving is disabled.",
            )
        matrix = transport.tocsr()
        raw_transport = raw_transport.tocsr()
        try:
            pin_matrix, pin_rule, pin_diagnostics = build_source_pin_matrix_for_candidate(
                self.structure,
                valley_ctx,
                candidate,
            )
        except (SymmetrySupportError, np.linalg.LinAlgError) as exc:
            reason = getattr(exc, "reason", "q_mapping_missing")
            pin_matrix = None
            pin_rule = {
                "source_form": "ld_source_rule",
                "ld_source_rule": "q_lambda = R_eff^{-1}(k + K_target(lambda)) - K_lambda",
                "pin_shift_by_group_coeffs": {},
            }
            pin_diagnostics = {
                "pin_supported": False,
                "pin_reason": reason,
            }
        raw_transport_delta = frobenius_relative_residual(raw_transport, matrix, denominator=raw_transport)
        if raw_transport_delta > 1.0e-12:
            pg_matrix = source_side_periodic_gauge_from_raw_transport(
                matrix,
                raw_transport,
                antiunitary=bool(candidate.get("antiunitary", False)),
            )
            pg_rule = {
                "pg_shift_by_group_coeffs": _json_int_vector_dict(raw_transport_diagnostics.get("target_group_shift_coeffs")),
                "pg_phase_convention": PG_EXTRACTED_PHASE_CONVENTION,
            }
            pg_diagnostics = {
                "pg_reason": "extracted_from_raw_transport",
                "raw_transport_delta": raw_transport_delta,
            }
        else:
            try:
                pg_matrix, pg_rule, pg_diagnostics = build_periodic_gauge_matrix_for_candidate(
                    self.structure,
                    valley_ctx,
                    candidate,
                )
            except (SymmetrySupportError, np.linalg.LinAlgError) as exc:
                reason = getattr(exc, "reason", "q_mapping_missing")
                pg_matrix = None
                pg_rule = {
                    "pg_shift_by_group_coeffs": {},
                    "pg_phase_convention": PG_PHASE_CONVENTION,
                }
                pg_diagnostics = {
                    "pg_reason": reason,
                }
        raw_h_matrix = None
        if pg_matrix is not None:
            pg_for_raw = pg_matrix.conj() if bool(candidate.get("antiunitary", False)) else pg_matrix
            raw_h_matrix = (matrix @ pg_for_raw).tocsr()
        gamma_row = next((row for row in candidate_rows if row.get("k_label") == "Gamma"), None)
        reference_row = gamma_row or (candidate_rows[0] if candidate_rows else {})
        projected_dim = int(matrix.shape[0])
        axis_angle_deg = None
        if "C2" in operation:
            axis_angle_deg = c2_axis_angle_deg_from_rotation(candidate.get("rotation_cart"))
        return {
            "valley": int(valley),
            "valley_label": str(valley_label),
            "operation": operation,
            "antiunitary": bool(candidate.get("antiunitary", False)),
            "spglib_index": _candidate_spglib_index(candidate),
            "axis_angle_deg": axis_angle_deg,
            "basis_hash": _basis_hash_for_valley_context(self.structure, valley_ctx, projected_dim),
            "residual_H_raw": reference_row.get("residual_H_raw"),
            "status": reference_row.get("status", ""),
            "g_perm_max_delta": reference_row.get("g_perm_max_delta"),
            "nonzero_reciprocal_shift_count": reference_row.get("nonzero_reciprocal_shift_count"),
            "square_residual": reference_row.get("square_residual"),
            "matrix": matrix,
            "matrix_role": "D_g^(0)",
            "pin_supported": bool(pin_diagnostics.get("pin_supported", False)),
            "pin_matrix": pin_matrix,
            "pin_reason": str(pin_diagnostics.get("pin_reason", "")),
            "pin_shift_by_group_coeffs": _json_int_vector_dict(pin_rule.get("pin_shift_by_group_coeffs")),
            "source_form": str(pin_rule.get("source_form", "ld_source_rule")),
            "ld_source_rule": str(pin_rule.get("ld_source_rule", "")),
            "pg_matrix": pg_matrix,
            "pg_reason": str(pg_diagnostics.get("pg_reason", "")),
            "pg_shift_by_group_coeffs": _json_int_vector_dict(pg_rule.get("pg_shift_by_group_coeffs")),
            "pg_phase_convention": str(pg_rule.get("pg_phase_convention", PG_PHASE_CONVENTION)),
            "raw_h_matrix": raw_h_matrix,
            "raw_h_action_rule": RAW_H_ACTION_RULE,
        }

    def _analyze(self) -> dict[str, Any]:
        valleys = getattr(self.config.symmetry_analysis, "valleys", None) or getattr(self.config.compute, "valleys", [])
        tolerance = float(getattr(self.config.symmetry_analysis, "tolerance", 1.0e-2))
        if not getattr(self.config.compute, "TAPW", False):
            raise ValueError("Symmetry-analysis mode currently supports TAPW only.")

        all_details = []
        operations_summary: dict[str, list[dict[str, Any]]] = {}
        representations: list[dict[str, Any]] = []
        validation_q_points = _default_validation_q_points()

        spatial_operations = collect_spglib_spatial_operations(self.structure)

        for valley in valleys:
            calculator = self._calculator_for_valley(int(valley))
            valley_label = getattr(calculator, "valley_flag", str(valley))
            valley_ctx = self._valley_context_for_valley(int(valley))
            valley_center = valley_ctx.valley_center_cart
            reciprocal_basis = valley_ctx.moire_reciprocal_basis

            valley_entries = []
            saved_representation_operations: set[str] = set()
            for candidate in _minimal_symmetry_candidates_for_valley(
                valley_ctx,
                getattr(self.config.twist, "bravais", "hex"),
                spatial_operations=spatial_operations,
                structure=self.structure,
            ):
                candidate = dict(candidate)
                candidate["index"] = int(candidate["index"])
                candidate["spatial_operations"] = spatial_operations
                if candidate["name"] == "K_C2T":
                    spglib_c2 = self._select_k_c2t_spatial_operation(
                        int(valley),
                        spatial_operations,
                        tolerance,
                    )
                    if spglib_c2 is None:
                        spglib_c2 = find_layer_exchange_c2_spatial_operation(
                            self.structure,
                            spatial_operations,
                        )
                    candidate["k_c2t_operation_scan"] = list(getattr(self, "_last_k_c2t_operation_scan", []) or [])
                    candidate["spglib_layer_exchange_c2"] = spglib_c2
                    if spglib_c2 is not None:
                        candidate["rotation_frac"] = np.asarray(spglib_c2["rotation_frac"], dtype=float)
                        candidate["translation_frac"] = np.asarray(spglib_c2["translation_frac"], dtype=float)
                        candidate["rotation_cart"] = np.asarray(spglib_c2["rotation_cart"], dtype=float)
                        candidate["translation_cart"] = np.asarray(spglib_c2["translation_cart"], dtype=float)
                if candidate["name"] == "M_C2_eta":
                    resolved_symmetry = resolve_reference_m_valley_c2_symmetry(
                        self.structure,
                        valley_ctx.calculator.TAPW_parameters,
                    )
                    candidate["resolved_m_c2_symmetry"] = resolved_symmetry
                    spglib_m_c2 = self._select_m_c2_eta_spatial_operation(
                        int(valley),
                        spatial_operations,
                    )
                    candidate["m_c2_eta_operation_scan"] = list(getattr(self, "_last_m_c2_eta_operation_scan", []) or [])
                    params = valley_ctx.calculator.TAPW_parameters
                    m_k1 = np.asarray(getattr(params, "m_K1", None), dtype=float) if getattr(params, "m_K1", None) is not None else None
                    m_k2 = np.asarray(getattr(params, "m_K2", None), dtype=float) if getattr(params, "m_K2", None) is not None else None
                    if m_k1 is None or m_k2 is None:
                        _, _, m_k1, m_k2, _ = params.calculate_K_points()
                    axis_vec = np.asarray(m_k1, dtype=float) + np.asarray(m_k2, dtype=float)
                    axis_vec = axis_vec / np.linalg.norm(axis_vec)
                    if spglib_m_c2 is not None:
                        candidate["spglib_m_c2_operation"] = spglib_m_c2
                        candidate["rotation_frac"] = np.asarray(spglib_m_c2["rotation_frac"], dtype=float)
                        candidate["translation_frac"] = np.asarray(spglib_m_c2["translation_frac"], dtype=float)
                        candidate["rotation_cart"] = np.asarray(spglib_m_c2["rotation_cart"], dtype=float)
                        candidate["translation_cart"] = np.asarray(spglib_m_c2["translation_cart"], dtype=float)
                    else:
                        candidate["rotation_frac"] = np.asarray(resolved_symmetry.rotation_frac, dtype=float)
                        candidate["translation_frac"] = np.asarray(resolved_symmetry.translation_frac, dtype=float)
                        candidate["rotation_cart"] = np.asarray(
                            rotate_mat(np.array([axis_vec[0], axis_vec[1], 0.0]), np.pi),
                            dtype=float,
                        )
                        candidate["translation_cart"] = np.asarray(resolved_symmetry.translation_cart, dtype=float)
                if candidate["name"] in {"K_C2T", "M_C2_eta"}:
                    candidate["closed"] = True
                    candidate["closure_reason"] = ""
                    candidate["reciprocal_shift"] = np.zeros(2, dtype=int)
                else:
                    candidate["closed"] = False
                    candidate["closure_reason"] = "valley_not_closed"
                    closure = check_valley_closure(
                        valley_center,
                        np.asarray(candidate["rotation_cart"], dtype=float)[:2, :2],
                        reciprocal_basis,
                        antiunitary=bool(candidate.get("antiunitary", False)),
                        atol=1.0e-6,
                    )
                    candidate["closed"] = bool(closure["closed"])
                    candidate["closure_reason"] = str(closure["reason"])
                    candidate["reciprocal_shift"] = closure["reciprocal_shift"]

                candidate_rows = [
                    self._candidate_rows_for_q(
                        candidate,
                        int(valley),
                        valley_label,
                        q_label,
                        q_value,
                        tolerance,
                    )
                    for q_label, q_value in validation_q_points
                ]
                all_details.extend(candidate_rows)

                candidate_status = combine_statuses(row["status"] for row in candidate_rows)
                candidate_supported = all(row["supported"] for row in candidate_rows)
                candidate_reason = ""
                for row in candidate_rows:
                    if row["not_supported_reason"]:
                        candidate_reason = row["not_supported_reason"]
                        break
                operation_display_name = displayed_operation_name(candidate["name"])
                axis_angle_deg = None
                if "C2" in operation_display_name:
                    axis_angle_deg = c2_axis_angle_deg_from_rotation(candidate.get("rotation_cart"))
                representation_name = representation_operation_name(candidate["name"])
                if candidate_supported and _is_saved_minimal_generator(representation_name, saved_representation_operations):
                    try:
                        representations.append(
                            self._representation_record_for_candidate(
                                candidate=candidate,
                                valley=int(valley),
                                valley_label=valley_label,
                                valley_ctx=valley_ctx,
                                candidate_rows=candidate_rows,
                            )
                        )
                        saved_representation_operations.add(representation_name)
                    except SymmetrySupportError:
                        pass
                valley_entries.append(
                    {
                        "operation": operation_display_name,
                        "supported": candidate_supported,
                        "status": candidate_status,
                        "not_supported_reason": candidate_reason,
                        "axis_angle_deg": axis_angle_deg,
                        "spglib_index": _candidate_spglib_index(candidate),
                    }
                )

            operations_summary[valley_label] = valley_entries

        valley_results = {}
        for valley_label, entries in operations_summary.items():
            valley_rows = [row for row in all_details if row["valley"] == valley_label]

            def _aggregate(status_key: str) -> str:
                metric_statuses = [
                    row[status_key]
                    for row in valley_rows
                    if row.get(status_key) not in (None, "not_supported")
                ]
                return combine_statuses(metric_statuses) if metric_statuses else "not_supported"

            unsupported_reasons = sorted(
                {
                    row["not_supported_reason"]
                    for row in valley_rows
                    if row["not_supported_reason"]
                }
            )
            valley_results[valley_label] = {
                "covariance_status": _aggregate("covariance_status"),
                "unsupported_reasons": unsupported_reasons,
                "operations_analyzed": [entry["operation"] for entry in entries],
            }

        minimal_generators = _minimal_generators_by_valley(operations_summary)

        return {
            "summary": {
                "valleys": list(valleys),
                "tolerance": tolerance,
                "eta_convention": (
                    "eta labels the three C3-related M valleys, eta=0,1,2. "
                    "In each M_eta valley, C2 denotes C3^eta C2 C3^{-eta}."
                ),
                "operations": operations_summary,
                "valley_results": valley_results,
                "minimal_generators": minimal_generators,
            },
            "details": all_details,
            "representations": [record for record in representations if record is not None],
        }

    def _summary_markdown_lines(self, summary: dict[str, Any]) -> list[str]:
        valleys = summary.get("valleys", [])
        tolerance = summary.get("tolerance")
        operations = summary.get("operations", {})
        valley_results = summary.get("valley_results", {})
        minimal_generators = summary.get("minimal_generators", {})

        lines = [
            "# Symmetry Analysis Summary",
            "",
            "Valleys analyzed: " + (", ".join(str(v) for v in valleys) if valleys else "None"),
            f"Tolerance: {tolerance}",
            "",
            "Eta convention: eta labels the three C3-related M valleys, eta=0,1,2.",
            "In each M_eta valley, C2 denotes C3^eta C2 C3^{-eta}.",
            "",
            "Per-valley assessment:",
        ]
        if valley_results:
            for valley, result in valley_results.items():
                lines.extend(
                    [
                        f"- {valley}:",
                        f"  - Raw H covariance: {result.get('covariance_status', 'not_supported')}",
                    ]
                )
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "Minimal generators:",
            ]
        )
        if minimal_generators:
            for valley, generators in minimal_generators.items():
                generator_text = ", ".join(str(name) for name in generators) if generators else "None (identity only)"
                lines.append(f"- {valley}: {generator_text}")
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "Operations:",
            ]
        )
        if operations:
            for valley, entries in operations.items():
                lines.append(f"- {valley}:")
                if entries:
                    for entry in entries:
                        lines.append(
                            "  - {operation}: supported={supported}, status={status}".format(
                                operation=_operation_markdown_label(entry),
                                supported=entry.get("supported", False),
                                status=entry.get("status", "unknown"),
                            )
                        )
                else:
                    lines.append("  - None")
        else:
            lines.append("- None")
        return lines

    def _write_summary_markdown(self, summary: dict[str, Any], output_dir: Path) -> None:
        path = output_dir / "summary.md"
        path.write_text("\n".join(self._summary_markdown_lines(summary)) + "\n", encoding="utf-8")

    def _write_summary_json(self, summary: dict[str, Any], output_dir: Path) -> None:
        path = output_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_debug_json(self, summary: dict[str, Any], details: list[dict[str, Any]], output_dir: Path) -> None:
        diagnostics_dir = output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / "debug.json"
        debug_details = []
        for row in details:
            debug_details.append(
                {
                    "valley": row.get("valley"),
                    "operation": row.get("operation"),
                    "spglib_index": row.get("spglib_index", ""),
                    "R_2d": row.get("R_2d", ""),
                    "k_label": row.get("k_label"),
                    "status": row.get("status"),
                    "not_supported_reason": row.get("not_supported_reason", ""),
                    "residual_H_raw": row.get("residual_H_raw"),
                    "residual_H_sym": row.get("residual_H_sym"),
                    "g_perm_max_delta": row.get("g_perm_max_delta"),
                    "nonzero_reciprocal_shift_count": row.get("nonzero_reciprocal_shift_count"),
                    "square_residual": row.get("square_residual"),
                    "debug": row.get("debug", {}),
                }
            )
        payload = {
            "summary": summary,
            "details": debug_details,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_details_csv(self, details: list[dict[str, Any]], output_dir: Path) -> None:
        path = output_dir / "details.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
            writer.writeheader()
            for row in details:
                payload = {column: row.get(column, "") for column in DETAIL_COLUMNS}
                writer.writerow(payload)

    def _write_representations(
        self,
        representations: list[dict[str, Any]],
        output_dir: Path,
        *,
        developer_outputs: bool = False,
    ) -> None:
        representations_dir = output_dir / "representations"
        representations_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "basis": "tapw_projected",
            "basis_order": "spin_outermost; group -> g_index -> atom_type -> orbital",
            "matrices": [],
        }

        for record in representations:
            matrix = record.get("matrix")
            if matrix is None:
                continue
            if not scipy.sparse.issparse(matrix):
                raise TypeError("Saved symmetry representation matrices must be scipy sparse matrices.")
            matrix = matrix.tocsr()
            valley_label = _safe_path_component(record.get("valley_label", record.get("valley", "unknown")))
            operation = representation_operation_name(record.get("operation", "unknown"))
            operation_file = _safe_path_component(operation) + ".npz"
            matrix_dir = representations_dir / valley_label
            matrix_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_dir = representations_dir / "diagnostics" / valley_label
            developer_files: dict[str, str] = {}
            for old_name in (operation_file, _safe_path_component(operation) + "_Pin.npz", _safe_path_component(operation) + "_PG.npz"):
                old_path = matrix_dir / old_name
                if old_path.exists():
                    old_path.unlink()
            if developer_outputs:
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                relative_path = Path("diagnostics") / valley_label / operation_file
                scipy.sparse.save_npz(representations_dir / relative_path, matrix)
                developer_files["file"] = relative_path.as_posix()

            pin_matrix = record.get("pin_matrix")
            pin_shape = None
            pin_nnz = None
            if pin_matrix is not None:
                if not scipy.sparse.issparse(pin_matrix):
                    raise TypeError("Saved source-shift P_in matrices must be scipy sparse matrices.")
                pin_matrix = pin_matrix.tocsr()
                pin_shape = [int(pin_matrix.shape[0]), int(pin_matrix.shape[1])]
                pin_nnz = int(pin_matrix.nnz)
                if developer_outputs:
                    pin_file = _safe_path_component(operation) + "_Pin.npz"
                    pin_relative_path = Path("diagnostics") / valley_label / pin_file
                    scipy.sparse.save_npz(representations_dir / pin_relative_path, pin_matrix)
                    developer_files["pin_file"] = pin_relative_path.as_posix()

            pg_matrix = record.get("pg_matrix")
            pg_shape = None
            pg_nnz = None
            if pg_matrix is not None:
                if not scipy.sparse.issparse(pg_matrix):
                    raise TypeError("Saved periodic-gauge P_G matrices must be scipy sparse matrices.")
                pg_matrix = pg_matrix.tocsr()
                pg_shape = [int(pg_matrix.shape[0]), int(pg_matrix.shape[1])]
                pg_nnz = int(pg_matrix.nnz)
                if developer_outputs:
                    pg_file = _safe_path_component(operation) + "_PG.npz"
                    pg_relative_path = Path("diagnostics") / valley_label / pg_file
                    scipy.sparse.save_npz(representations_dir / pg_relative_path, pg_matrix)
                    developer_files["pg_file"] = pg_relative_path.as_posix()

            raw_h_matrix = record.get("raw_h_matrix")
            raw_h_relative_path = None
            raw_h_shape = None
            raw_h_nnz = None
            if raw_h_matrix is not None:
                if not scipy.sparse.issparse(raw_h_matrix):
                    raise TypeError("Saved raw-H symmetry operators must be scipy sparse matrices.")
                raw_h_matrix = raw_h_matrix.tocsr()
                raw_h_file = _safe_path_component(operation) + "_rawH.npz"
                raw_h_relative_path = Path(valley_label) / raw_h_file
                scipy.sparse.save_npz(representations_dir / raw_h_relative_path, raw_h_matrix)
                raw_h_shape = [int(raw_h_matrix.shape[0]), int(raw_h_matrix.shape[1])]
                raw_h_nnz = int(raw_h_matrix.nnz)

            manifest_row = {
                "valley": int(record.get("valley")),
                "valley_label": str(record.get("valley_label", "")),
                "operation": operation,
                "antiunitary": bool(record.get("antiunitary", False)),
                "spglib_index": _json_optional_int(record.get("spglib_index", "")),
                "axis_deg": _json_optional_float(record.get("axis_angle_deg")),
                "dtype": str(raw_h_matrix.dtype if raw_h_matrix is not None else matrix.dtype),
                "pin_supported": bool(record.get("pin_supported", False)),
                "pin_reason": str(record.get("pin_reason", "")),
                "pin_shift_by_group_coeffs": _json_int_vector_dict(record.get("pin_shift_by_group_coeffs")),
                "pg_reason": str(record.get("pg_reason", "")),
                "pg_shift_by_group_coeffs": _json_int_vector_dict(record.get("pg_shift_by_group_coeffs")),
                "pg_phase_convention": str(record.get("pg_phase_convention", PG_PHASE_CONVENTION)),
                "raw_h_operator_file": None if raw_h_relative_path is None else raw_h_relative_path.as_posix(),
                "raw_h_operator_shape": raw_h_shape,
                "raw_h_operator_nnz": raw_h_nnz,
                "raw_h_action_rule": str(record.get("raw_h_action_rule", RAW_H_ACTION_RULE)),
                "source_form": str(record.get("source_form", "ld_source_rule")),
                "ld_source_rule": str(record.get("ld_source_rule", "")),
                "basis_hash": str(record.get("basis_hash", "")),
                "residual_H_raw": _json_optional_float(record.get("residual_H_raw")),
                "status": record.get("status", ""),
                "g_perm_max_delta": _json_optional_float(record.get("g_perm_max_delta")),
                "nonzero_reciprocal_shift_count": _json_optional_int(record.get("nonzero_reciprocal_shift_count", "")),
                "square_residual": _json_optional_float(record.get("square_residual")),
            }
            if developer_files:
                manifest_row["developer_outputs"] = developer_files
            manifest["matrices"].append(manifest_row)

        manifest_path = representations_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def run(self):
        payload = self._analyze()
        summary = dict(payload.get("summary", {}))
        details = list(payload.get("details", []))
        representations = list(payload.get("representations", []))

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_summary_markdown(summary, output_dir)
        self._write_summary_json(summary, output_dir)
        self._write_details_csv(details, output_dir)
        self._write_representations(
            representations,
            output_dir,
            developer_outputs=bool(getattr(self.config.symmetry_analysis, "developer_outputs", False)),
        )
        debug_path = output_dir / "debug.json"
        if debug_path.exists():
            debug_path.unlink()
        if bool(getattr(self.config.symmetry_analysis, "debug", False)):
            self._write_debug_json(summary, details, output_dir)
        else:
            diagnostics_debug_path = output_dir / "diagnostics" / "debug.json"
            if diagnostics_debug_path.exists():
                diagnostics_debug_path.unlink()

        if self.logger is not None:
            self.logger.info("Wrote symmetry-analysis outputs to %s", output_dir)
        return payload
