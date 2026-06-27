"""
TAPW (Twisted Atomic Plane Wave) Hamiltonian calculation module
"""
from __future__ import annotations

import numpy as np
import scipy
from scipy.linalg import lapack
import scipy.linalg
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import det
from scipy.spatial import cKDTree
import time
import os
import sys
import copy
import json
import multiprocessing as mp
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

from numpy.lib.format import open_memmap
from joblib import Parallel, delayed
import joblib.parallel as joblib_parallel
from ..symmetry.representations import (
    C3_MoTe2_all,
    C3_G_matrix,
    direct_sum,
    generate_direct_sum_params,
    rot_matrix,
    single_valley_c3_incompatibility_reason,
    spin_reps,
    supports_single_valley_c3,
    rotate_mat,
)
from tqdm import tqdm
from ..config import ComputeConfig, normalize_symmetrize_hamiltonian
from ..artifacts import (
    array_output_filename,
    band_output_filename,
    canonical_band_filename,
    canonical_topology_band_label,
    canonical_topology_grid_name,
    chern_flux_filename,
    chern_summary_filename,
    gvec_output_filename,
    raw_memmap_filename,
)
from ..chern_post import (
    compute_berry_flux_multiband,
    compute_berry_flux_single_band,
    reshape_wavefunction_grid,
)
from ..io.structure import StructureProcessorSpglib
from ..io.kpath import KPathGenerator
from ..geometry.rotations import get_any_rot_orb_twostep
from ..reporting import ensure_reporter
from ..utils import (
    timing_decorator_factory, rotate_vector, unique_sorted, 
    check_hermitian, is_positive_definite, print_sparse_matrix_info, 
    check_sparsity, HARTREE
)
import re
# 设置numpy的打印精度
np.set_printoptions(precision=6)


def _is_canonical_output_layout(config) -> bool:
    layout = getattr(config, "output_layout", None)
    return str(getattr(layout, "style", "")).lower() == "canonical_v1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_canonical_manifests(workflow_dir: Path, workflow: str) -> None:
    target_dir = workflow_dir.parent
    if len(target_dir.parents) < 2:
        return
    profile_dir = target_dir.parent
    root_dir = profile_dir.parent
    workflows = sorted(
        child.name
        for child in target_dir.iterdir()
        if child.is_dir() and (child / "manifest.json").exists()
    )
    _write_json(
        target_dir / "manifest.json",
        {
            "schema": "tapw_target_outputs/v1",
            "profile": profile_dir.name,
            "q_shell": target_dir.name,
            "workflows": workflows,
        },
    )
    targets = []
    for profile_path in sorted(root_dir.iterdir()):
        if not profile_path.is_dir():
            continue
        for q_path in sorted(profile_path.iterdir()):
            if q_path.is_dir() and (q_path / "manifest.json").exists():
                targets.append(
                    {
                        "profile": profile_path.name,
                        "q_shell": q_path.name,
                        "path": f"{profile_path.name}/{q_path.name}",
                    }
                )
    _write_json(
        root_dir / "manifest.json",
        {"schema": "tapw_outputs/v1", "updated_workflow": workflow, "targets": targets},
    )

# Optional MKL-accelerated sparse GEMM (can be a big speedup for g@H@g^H)
try:
    from sparse_dot_mkl import dot_product_mkl  # type: ignore
    _HAS_SPARSE_DOT_MKL = True
except Exception:  # pragma: no cover
    dot_product_mkl = None
    _HAS_SPARSE_DOT_MKL = False


_MEMMAP_CACHE: dict[str, np.memmap] = {}

_SLEPC_FACTOR_PROBE_CACHE: dict[tuple[str, str], str] = {}
_NOTAPW_SLEPC_FACTOR_CANDIDATES = (
    "mkl_pardiso",
    "mkl_cpardiso",
    "mumps",
    "superlu",
    "umfpack",
    "superlu_dist",
    "petsc",
)
_NOTAPW_SLEPC_SPD_SHIFT = 1.0e-10
_NOTAPW_SLEPC_TOL = 1.0e-8
_NOTAPW_SLEPC_MAX_IT = 5000
_M_VALLEY_TRIPLET = (31, 32, 33)
_M_VALLEY_ROTATIONS = {31: 0, 32: 120, 33: 240}
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)


def uses_m_valley_threefold_symmetrization(config: ComputeConfig) -> bool:
    bravais = getattr(config, "bravais", "hex")
    operations = requested_hamiltonian_symmetry_operations(config)
    return bool(
        getattr(config, "TAPW", False)
        and "C3z" in operations
        and str(bravais).lower() == "hex"
        and getattr(config, "valley", None) in _M_VALLEY_TRIPLET
    )


def uses_m_valley_d3_symmetrization(config: ComputeConfig) -> bool:
    operations = requested_hamiltonian_symmetry_operations(config)
    return bool(
        uses_m_valley_threefold_symmetrization(config)
        and "C2" in operations
    )


def requested_hamiltonian_symmetry_operations(config: ComputeConfig) -> list[str]:
    resolved_by_valley = getattr(config, "resolved_hamiltonian_symmetry_operations_by_valley", None)
    valley_label = BandStructureCalculator.VALLEY_MAP.get(getattr(config, "valley", None))
    if valley_label is None:
        valley_label = getattr(config, "valley_flag", None)
    if isinstance(resolved_by_valley, dict) and valley_label in resolved_by_valley:
        return list(resolved_by_valley[valley_label])

    spec = normalize_symmetrize_hamiltonian(getattr(config, "symmetrize_hamiltonian", False))
    if spec is False:
        return []
    if spec is True:
        # Direct calculator construction outside the CLI cannot know the manifest-selected
        # generators. Use the existing C3 backend as the conservative default.
        return ["C3z"]
    return list(spec)


def _validate_hamiltonian_symmetrization_backend(config: ComputeConfig, valley_label: str, operations: list[str]) -> None:
    if not operations:
        return
    bravais = str(getattr(config, "bravais", "hex")).lower()
    valley = getattr(config, "valley", None)
    op_set = set(operations)
    if valley in _M_VALLEY_TRIPLET and bravais == "hex":
        if op_set in ({"C3z"}, {"C3z", "C2"}):
            return
    elif op_set == {"C3z"} and supports_single_valley_c3(bravais, valley):
        return
    raise ValueError(
        "No TAPW Hamiltonian symmetrization backend is implemented for "
        f"valley {valley_label} with operations {operations}. "
        "Available release backends are single-valley [C3z] and M-valley [C3z] or [C3z, C2]."
    )


def resolve_kpoint_parallel_policy(config: ComputeConfig, os_name: str | None = None) -> SimpleNamespace:
    """Resolve the k-point parallel policy without touching non-TAPW solver constraints.

    The repository defaults are `joblib/loky` because the non-TAPW SLEPc path needs
    that spawn-based configuration. TAPW band calculations do not share that
    restriction, and they already have a dedicated `mp`/fork implementation that
    avoids repeatedly pickling a very large calculator object.
    """
    parallel_impl = str(getattr(config, "parallel_impl", "joblib"))
    parallel_backend = str(getattr(config, "parallel_backend", "loky"))
    num_processes = int(getattr(config, "num_processes", 1))
    auto_promoted = False

    if (
        getattr(config, "TAPW", False)
        and bool(getattr(config, "tapw_auto_fork", True))
        and num_processes > 1
        and parallel_impl == "joblib"
        and parallel_backend == "loky"
        and (os_name or os.name) == "posix"
    ):
        parallel_impl = "mp"
        auto_promoted = True

    return SimpleNamespace(
        parallel_impl=parallel_impl,
        parallel_backend=parallel_backend,
        auto_promoted=auto_promoted,
    )


def rotate_local_k_between_m_valleys(k_local, reciprocal_tmat, source_valley: int, target_valley: int):
    if source_valley not in _M_VALLEY_ROTATIONS or target_valley not in _M_VALLEY_ROTATIONS:
        raise ValueError(f"M-valley rotation only supports {_M_VALLEY_TRIPLET}, got {source_valley}->{target_valley}")

    reciprocal_tmat = np.asarray(reciprocal_tmat, dtype=float)
    k_local_arr = np.asarray(k_local, dtype=float)
    if k_local_arr.shape == (2,):
        k_local_arr = np.array([k_local_arr[0], k_local_arr[1], 0.0], dtype=float)
    elif k_local_arr.shape != (3,):
        raise ValueError(f"k_local must have shape (2,) or (3,), got {k_local_arr.shape}")

    delta_angle = _M_VALLEY_ROTATIONS[target_valley] - _M_VALLEY_ROTATIONS[source_valley]
    k_cart = np.dot(k_local_arr, reciprocal_tmat)
    k_cart_rot = np.array(k_cart, copy=True)
    k_cart_rot[:2] = rotate_vector(k_cart[:2], delta_angle)
    return np.linalg.solve(reciprocal_tmat.T, k_cart_rot)


def transport_matrix_between_m_valleys(matrix: np.ndarray, transport: np.ndarray) -> np.ndarray:
    return transport @ matrix @ transport.conj().T


def transform_k_by_cartesian_linear_map(k_local, reciprocal_tmat, linear_map: np.ndarray):
    reciprocal_tmat = np.asarray(reciprocal_tmat, dtype=float)
    linear_map = np.asarray(linear_map, dtype=float)
    k_local_arr = np.asarray(k_local, dtype=float)
    if k_local_arr.shape == (2,):
        k_local_arr = np.array([k_local_arr[0], k_local_arr[1], 0.0], dtype=float)
    elif k_local_arr.shape != (3,):
        raise ValueError(f"k_local must have shape (2,) or (3,), got {k_local_arr.shape}")
    if linear_map.shape != (2, 2):
        raise ValueError(f"linear_map must have shape (2, 2), got {linear_map.shape}")

    k_cart = np.dot(k_local_arr, reciprocal_tmat)
    k_cart_transformed = np.array(k_cart, copy=True)
    k_cart_transformed[:2] = linear_map @ k_cart[:2]
    return np.linalg.solve(reciprocal_tmat.T, k_cart_transformed)


def threefold_reference_hs_average(
    h_ref: np.ndarray,
    s_ref: np.ndarray | None,
    h_c1: np.ndarray,
    s_c1: np.ndarray | None,
    h_c2: np.ndarray,
    s_c2: np.ndarray | None,
    u_c1_to_ref: np.ndarray,
    u_c2_to_ref: np.ndarray,
):
    h_avg = (
        h_ref
        + transport_matrix_between_m_valleys(h_c1, u_c1_to_ref)
        + transport_matrix_between_m_valleys(h_c2, u_c2_to_ref)
    ) / 3.0

    if s_ref is None:
        return h_avg, None

    if s_c1 is None or s_c2 is None:
        raise ValueError("S-matrix averaging requires s_ref, s_c1, and s_c2 together.")

    s_avg = (
        s_ref
        + transport_matrix_between_m_valleys(s_c1, u_c1_to_ref)
        + transport_matrix_between_m_valleys(s_c2, u_c2_to_ref)
    ) / 3.0
    return h_avg, s_avg


def twofold_reference_hs_average(
    h_ref: np.ndarray,
    s_ref: np.ndarray | None,
    h_partner: np.ndarray,
    s_partner: np.ndarray | None,
    u_c2_ref: np.ndarray,
    antiunitary: bool = False,
):
    partner_h = h_partner.conj() if antiunitary else h_partner
    h_avg = 0.5 * (h_ref + transport_matrix_between_m_valleys(partner_h, u_c2_ref))

    if s_ref is None:
        return h_avg, None

    if s_partner is None:
        raise ValueError("S-matrix averaging requires s_ref and s_partner together.")

    partner_s = s_partner.conj() if antiunitary else s_partner
    s_avg = 0.5 * (s_ref + transport_matrix_between_m_valleys(partner_s, u_c2_ref))
    return h_avg, s_avg


def _m_valley_rotation_delta(source_valley: int, target_valley: int) -> int:
    if source_valley not in _M_VALLEY_ROTATIONS or target_valley not in _M_VALLEY_ROTATIONS:
        raise ValueError(f"M-valley rotation only supports {_M_VALLEY_TRIPLET}, got {source_valley}->{target_valley}")
    return (_M_VALLEY_ROTATIONS[target_valley] - _M_VALLEY_ROTATIONS[source_valley]) % 360


def _parse_orbitals_from_orb_name(orb_name: str):
    if not isinstance(orb_name, str):
        raise ValueError(f"orb_name must be str, got {type(orb_name)}")
    if "-" in orb_name:
        _, orb_part = orb_name.split("-", 1)
    else:
        orb_part = orb_name
    orbitals = {}
    for orb, count in re.findall(r"([spdf])(\d+)", orb_part):
        orbitals[orb] = int(count)
    if len(orbitals) == 0:
        raise ValueError(f"Cannot parse orbitals from orb_name='{orb_name}'")
    return orbitals


def _build_group_orbital_rotation_blocks(structure_df, angle_deg: int, spin: bool):
    rotation_matrix = rot_matrix(angle_deg)
    return _build_group_orbital_rotation_blocks_from_rotation_matrix(
        structure_df=structure_df,
        rotation_matrix=rotation_matrix,
        spin=spin,
        spin_rep=None,
    )


def _build_group_orbital_rotation_blocks_from_rotation_matrix(
    structure_df,
    rotation_matrix: np.ndarray,
    spin: bool,
    spin_rep: np.ndarray | None = None,
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
    unique_groups = sorted(df_temp["twist_group"].unique().tolist())
    group_blocks = []
    for gid in unique_groups:
        df_g = df_temp[df_temp["twist_group"] == gid]
        if df_g.empty:
            raise ValueError(f"No atoms found for twist_group={gid}")
        group_atom_types = [at for at in atom_type_list_global if (df_g["atom_type"] == at).any()]
        type_blocks = []
        for at in group_atom_types:
            orb_name = df_g.loc[df_g["atom_type"] == at, "orb_name"].iloc[0]
            orbitals_dict = _parse_orbitals_from_orb_name(orb_name)
            params = generate_direct_sum_params(orbitals_dict, orb_mapping)
            type_blocks.append(direct_sum(*params))
        group_blocks.append(direct_sum(*type_blocks))
    return group_blocks, spin_rep if spin else None


def select_reference_m_valley_c2_operation(
    lattice: np.ndarray,
    rotations,
    translations,
    invariant_k_cart,
    invariance_tol: float = 1.0e-6,
):
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {lattice.shape}")

    invariant_k_cart = np.asarray(invariant_k_cart, dtype=float)
    if invariant_k_cart.shape != (2,):
        raise ValueError(f"invariant_k_cart must have shape (2,), got {invariant_k_cart.shape}")

    lattice_inv_t = np.linalg.inv(lattice.T)
    best_candidate = None

    for idx, (rotation_frac, translation_frac) in enumerate(zip(rotations, translations)):
        rotation_frac = np.asarray(rotation_frac, dtype=float)
        translation_frac = np.asarray(translation_frac, dtype=float)
        if rotation_frac.shape != (3, 3):
            raise ValueError(f"rotation matrix at index {idx} must have shape (3, 3), got {rotation_frac.shape}")
        if translation_frac.shape != (3,):
            raise ValueError(
                f"translation vector at index {idx} must have shape (3,), got {translation_frac.shape}"
            )

        if np.allclose(rotation_frac, np.eye(3), atol=1.0e-12):
            continue
        if not np.allclose(rotation_frac @ rotation_frac, np.eye(3), atol=1.0e-12):
            continue

        det_rot = round(float(np.linalg.det(rotation_frac)))
        if det_rot != 1:
            continue

        rotation_cart = lattice.T @ rotation_frac @ lattice_inv_t
        linear_map = rotation_cart[:2, :2]
        score = float(np.linalg.norm(linear_map @ invariant_k_cart - invariant_k_cart))
        if best_candidate is None or score < best_candidate[0]:
            best_candidate = (score, idx, linear_map, rotation_cart, translation_frac)

    if best_candidate is None:
        raise ValueError("Could not find a proper order-2 symmetry operation for the reference M-valley C2.")
    if best_candidate[0] > invariance_tol:
        raise ValueError(
            "Could not find a reference M-valley C2 operation that leaves the chosen M point invariant; "
            + f"best invariance error is {best_candidate[0]:.3e}."
        )

    _, idx, linear_map, rotation_cart, translation_frac = best_candidate
    return idx, linear_map, rotation_cart, translation_frac


def _cartesian_positions_to_fractional(lattice: np.ndarray, cartesian_positions: np.ndarray) -> np.ndarray:
    lattice = np.asarray(lattice, dtype=float)
    cartesian_positions = np.asarray(cartesian_positions, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {lattice.shape}")
    if cartesian_positions.ndim != 2 or cartesian_positions.shape[1] != 3:
        raise ValueError(
            f"cartesian_positions must have shape (N, 3), got {cartesian_positions.shape}"
        )
    frac = np.linalg.solve(lattice.T, cartesian_positions.T).T
    return frac - np.floor(frac)


def map_atom_types_by_fractional_symmetry(
    structure_df,
    lattice: np.ndarray,
    rotation_frac: np.ndarray,
    translation_frac: np.ndarray,
    source_group: int,
    target_group: int,
    tol: float = 5.0e-6,
):
    required_columns = {"x", "y", "z", "species", "atom_type", "twist_group"}
    missing = required_columns.difference(structure_df.columns)
    if missing:
        raise ValueError(
            "structure_df is missing columns required for fractional symmetry mapping: "
            + ", ".join(sorted(missing))
        )

    df_temp = structure_df.copy().reset_index(drop=True)
    lattice = np.asarray(lattice, dtype=float)
    rotation_frac = np.asarray(rotation_frac, dtype=float)
    translation_frac = np.asarray(translation_frac, dtype=float)
    positions_frac = _cartesian_positions_to_fractional(
        lattice,
        df_temp[["x", "y", "z"]].to_numpy(dtype=float),
    )

    source_mask = df_temp["twist_group"] == source_group
    target_mask = df_temp["twist_group"] == target_group
    if not np.any(source_mask) or not np.any(target_mask):
        raise ValueError(f"Missing atoms for twist_group pair {source_group}->{target_group}.")

    source_atom_types = sorted(df_temp.loc[source_mask, "atom_type"].astype(int).unique().tolist())
    mapping: dict[int, int] = {}
    used_target_atom_types: set[int] = set()

    for source_atom_type in source_atom_types:
        source_rows = np.flatnonzero(
            source_mask.to_numpy() & (df_temp["atom_type"].to_numpy() == source_atom_type)
        )
        if source_rows.size == 0:
            continue

        species = df_temp.loc[source_rows[0], "species"]
        target_rows = np.flatnonzero(
            target_mask.to_numpy() & (df_temp["species"].to_numpy() == species)
        )
        if target_rows.size == 0:
            raise ValueError(
                f"No target-group atoms with species={species!r} for source atom_type={source_atom_type}."
            )

        target_frac = positions_frac[target_rows]
        target_images = []
        target_lookup = []
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                shift = np.array([float(n1), float(n2), 0.0], dtype=float)
                target_images.append(target_frac + shift)
                target_lookup.append(target_rows)
        target_images_arr = np.concatenate(target_images, axis=0)
        target_lookup_arr = np.concatenate(target_lookup, axis=0)

        transformed_frac = (positions_frac[source_rows] @ rotation_frac.T) + translation_frac
        transformed_frac = transformed_frac - np.floor(transformed_frac)
        distances, matched = cKDTree(target_images_arr).query(transformed_frac, k=1)
        matched = np.asarray(matched, dtype=int)
        if np.any(distances > tol):
            raise ValueError(
                f"Fractional symmetry mapping {source_group}->{target_group} failed for source atom_type="
                f"{source_atom_type}: max mismatch={float(np.max(distances)):.3e}"
            )

        matched_target_rows = target_lookup_arr[matched]
        if len(np.unique(matched_target_rows)) != len(source_rows):
            raise ValueError(
                f"Fractional symmetry mapping {source_group}->{target_group} is not one-to-one for "
                f"source atom_type={source_atom_type}."
            )

        matched_target_types = np.unique(df_temp.loc[matched_target_rows, "atom_type"].astype(int).to_numpy())
        if matched_target_types.size != 1:
            raise ValueError(
                f"Fractional symmetry mapping {source_group}->{target_group} sends source atom_type="
                f"{source_atom_type} to multiple target atom types {matched_target_types.tolist()}."
            )

        target_atom_type = int(matched_target_types[0])
        if target_atom_type in used_target_atom_types:
            raise ValueError(
                f"Fractional symmetry mapping {source_group}->{target_group} reuses target atom_type="
                f"{target_atom_type}."
            )
        used_target_atom_types.add(target_atom_type)
        mapping[int(source_atom_type)] = target_atom_type

    return mapping


def resolve_reference_m_valley_c2_symmetry(structure, reference_params, symprec: float = 1.0e-3):
    if getattr(reference_params, "valley", None) not in _M_VALLEY_TRIPLET:
        raise ValueError(f"Reference C2 symmetry only supports M valleys {_M_VALLEY_TRIPLET}.")
    if not hasattr(structure, "df") or structure.df is None:
        raise ValueError("structure must provide a populated df for reference M-valley C2 symmetry.")
    if not hasattr(structure, "Tmat"):
        raise ValueError("structure must provide Tmat for reference M-valley C2 symmetry.")

    try:
        import spglib  # type: ignore
    except ImportError as exc:
        raise ImportError("spglib is required for reference M-valley C2 symmetry resolution.") from exc

    try:
        from ase.data import atomic_numbers  # type: ignore
    except ImportError as exc:
        raise ImportError("ase is required for reference M-valley C2 symmetry resolution.") from exc

    structure_df = structure.df.copy().reset_index(drop=True)
    required_columns = {"x", "y", "z", "species", "twist_group", "atom_type"}
    missing = required_columns.difference(structure_df.columns)
    if missing:
        raise ValueError(
            "structure.df is missing columns required for reference M-valley C2 symmetry: "
            + ", ".join(sorted(missing))
        )

    lattice = np.asarray(structure.Tmat, dtype=float)
    positions_frac = _cartesian_positions_to_fractional(
        lattice,
        structure_df[["x", "y", "z"]].to_numpy(dtype=float),
    )

    try:
        numbers = np.asarray(
            [atomic_numbers[str(species)] for species in structure_df["species"].tolist()],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown chemical species {exc.args[0]!r} in structure.df for spglib mapping.") from exc

    symmetry = spglib.get_symmetry((lattice, positions_frac, numbers), symprec=symprec)
    if symmetry is None:
        raise ValueError(f"spglib.get_symmetry failed for reference M-valley C2 resolution with symprec={symprec}.")

    _, _, m_k1, _, _ = reference_params.calculate_K_points()
    operation_index, linear_map_2d, rotation_cart, translation_frac = select_reference_m_valley_c2_operation(
        lattice=lattice,
        rotations=symmetry["rotations"],
        translations=symmetry["translations"],
        invariant_k_cart=np.asarray(m_k1, dtype=float),
    )

    rotation_frac = np.asarray(symmetry["rotations"][operation_index], dtype=float)
    translation_frac = np.asarray(translation_frac, dtype=float)
    atom_type_map_0_to_1 = map_atom_types_by_fractional_symmetry(
        structure_df=structure_df,
        lattice=lattice,
        rotation_frac=rotation_frac,
        translation_frac=translation_frac,
        source_group=0,
        target_group=1,
    )
    atom_type_map_1_to_0 = map_atom_types_by_fractional_symmetry(
        structure_df=structure_df,
        lattice=lattice,
        rotation_frac=rotation_frac,
        translation_frac=translation_frac,
        source_group=1,
        target_group=0,
    )

    return SimpleNamespace(
        operation_index=int(operation_index),
        rotation_frac=rotation_frac,
        translation_frac=translation_frac,
        rotation_cart=np.asarray(rotation_cart, dtype=float),
        translation_cart=np.asarray(lattice.T @ translation_frac, dtype=float),
        linear_map_2d=np.asarray(linear_map_2d, dtype=float),
        atom_type_map_0_to_1=atom_type_map_0_to_1,
        atom_type_map_1_to_0=atom_type_map_1_to_0,
    )


def reference_m_valley_c2_spin_unitary(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {rotation_matrix.shape}")
    return spin_reps(rotation_matrix) @ (1.0j * _SIGMA_Y)


def _build_group_orbital_transport_between_groups(
    structure_df,
    source_group: int,
    target_group: int,
    rotation_matrix: np.ndarray,
    spin: bool,
    spin_rep: np.ndarray | None = None,
    target_for_source: dict[int, int] | None = None,
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
        raise ValueError(f"Missing atoms for twist_group pair {source_group}->{target_group}.")

    source_atom_types = [at for at in atom_type_list_global if (df_source["atom_type"] == at).any()]
    target_atom_types = [at for at in atom_type_list_global if (df_target["atom_type"] == at).any()]
    if len(source_atom_types) != len(target_atom_types):
        raise ValueError(
                "Reference M-valley C2 transport requires matched atom-type counts across the two twist groups; "
                + f"got {len(source_atom_types)} and {len(target_atom_types)}."
        )

    if target_for_source is not None:
        target_for_source = {int(k): int(v) for k, v in target_for_source.items()}
        missing_source_types = sorted(set(source_atom_types).difference(target_for_source))
        if missing_source_types:
            raise ValueError(
                "Explicit source->target atom-type map is missing entries for source atom types "
                + f"{missing_source_types}."
            )
    elif "z" in df_temp.columns:
        def _rep_z(df_group, atom_type):
            return float(df_group.loc[df_group["atom_type"] == atom_type, "z"].iloc[0])

        source_by_z = sorted(source_atom_types, key=lambda at: _rep_z(df_source, at))
        target_by_z = sorted(target_atom_types, key=lambda at: _rep_z(df_target, at), reverse=True)
        target_for_source = dict(zip(source_by_z, target_by_z))
    else:
        target_for_source = dict(zip(source_atom_types, target_atom_types))

    source_offsets = {}
    source_blocks = {}
    source_dim_total = 0
    for atom_type in source_atom_types:
        orb_name = df_source.loc[df_source["atom_type"] == atom_type, "orb_name"].iloc[0]
        orbitals_dict = _parse_orbitals_from_orb_name(orb_name)
        params = generate_direct_sum_params(orbitals_dict, orb_mapping)
        block = direct_sum(*params)
        source_offsets[atom_type] = (source_dim_total, block.shape[0], orbitals_dict)
        source_blocks[atom_type] = block
        source_dim_total += block.shape[0]

    target_offsets = {}
    target_dim_total = 0
    for atom_type in target_atom_types:
        orb_name = df_target.loc[df_target["atom_type"] == atom_type, "orb_name"].iloc[0]
        orbitals_dict = _parse_orbitals_from_orb_name(orb_name)
        target_block = direct_sum(*generate_direct_sum_params(orbitals_dict, orb_mapping))
        target_offsets[atom_type] = (target_dim_total, target_block.shape[0], orbitals_dict)
        target_dim_total += target_block.shape[0]

    transport = np.zeros((target_dim_total, source_dim_total), dtype=np.complex128)
    for source_atom_type in source_atom_types:
        target_atom_type = target_for_source[source_atom_type]
        source_start, source_dim, source_orbitals = source_offsets[source_atom_type]
        target_start, target_dim, target_orbitals = target_offsets[target_atom_type]
        if source_dim != target_dim or source_orbitals != target_orbitals:
            raise ValueError(
                "Reference M-valley C2 transport requires matched orbital blocks across mapped atom types; "
                + f"source atom_type={source_atom_type} {source_orbitals} vs "
                + f"target atom_type={target_atom_type} {target_orbitals}."
            )
        transport[target_start : target_start + target_dim, source_start : source_start + source_dim] = source_blocks[
            source_atom_type
        ]

    return transport, spin_rep if spin else None


def _build_rotation_match_matrix(source_gvecs, target_gvecs, angle_deg: int, tol: float = 1.0e-2):
    source = np.asarray(source_gvecs, dtype=float)
    target = np.asarray(target_gvecs, dtype=float)
    if source.shape != target.shape:
        raise ValueError(f"G-vector lists must have the same shape, got {source.shape} and {target.shape}")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"G-vector lists must have shape (N, 2), got {source.shape}")

    rotated_source = np.array([rotate_vector(vec, angle_deg) for vec in source], dtype=float)
    matrix = np.zeros((target.shape[0], source.shape[0]), dtype=np.complex128)
    used_targets: set[int] = set()

    for source_index, rotated_vec in enumerate(rotated_source):
        deltas = np.linalg.norm(target - rotated_vec, axis=1)
        target_index = int(np.argmin(deltas))
        if deltas[target_index] > tol:
            raise ValueError(
                f"Cannot match rotated G-vector at index {source_index}: min delta={deltas[target_index]:.3e}"
            )
        if target_index in used_targets:
            raise ValueError(f"Rotation map is not one-to-one; repeated target index {target_index}")
        used_targets.add(target_index)
        matrix[target_index, source_index] = 1.0

    return matrix


def _build_linear_match_matrix(source_gvecs, target_gvecs, linear_map: np.ndarray, tol: float = 1.0e-2):
    source = np.asarray(source_gvecs, dtype=float)
    target = np.asarray(target_gvecs, dtype=float)
    linear_map = np.asarray(linear_map, dtype=float)
    if source.shape != target.shape:
        raise ValueError(f"G-vector lists must have the same shape, got {source.shape} and {target.shape}")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"G-vector lists must have shape (N, 2), got {source.shape}")
    if linear_map.shape != (2, 2):
        raise ValueError(f"linear_map must have shape (2, 2), got {linear_map.shape}")

    transformed_source = (linear_map @ source.T).T
    matrix = np.zeros((target.shape[0], source.shape[0]), dtype=np.complex128)
    used_targets: set[int] = set()

    for source_index, transformed_vec in enumerate(transformed_source):
        deltas = np.linalg.norm(target - transformed_vec, axis=1)
        target_index = int(np.argmin(deltas))
        if deltas[target_index] > tol:
            raise ValueError(
                f"Cannot match transformed G-vector at index {source_index}: min delta={deltas[target_index]:.3e}"
            )
        if target_index in used_targets:
            raise ValueError(f"Linear map is not one-to-one; repeated target index {target_index}")
        used_targets.add(target_index)
        matrix[target_index, source_index] = 1.0

    return matrix


def build_reference_m_valley_c2_partner_projector(structure, reference_params, resolved_symmetry):
    if getattr(reference_params, "valley", None) not in _M_VALLEY_TRIPLET:
        raise ValueError(f"Reference C2 partner projector only supports M valleys {_M_VALLEY_TRIPLET}.")
    if not hasattr(structure, "df") or structure.df is None:
        raise ValueError("structure must provide a populated df for the reference M-valley C2 partner projector.")

    required_cols = {"atom_type", "orb_num", "twist_group", "shifted_x", "shifted_y", "orb_name"}
    missing = required_cols.difference(structure.df.columns)
    if missing:
        raise ValueError(
            "structure.df is missing columns required for the reference M-valley C2 partner projector: "
            + ", ".join(sorted(missing))
        )

    df_temp = structure.df.copy().sort_values(["atom_type"], kind="stable").reset_index(drop=True)
    unique_groups = sorted(df_temp["twist_group"].unique().tolist())
    if unique_groups != [0, 1]:
        raise ValueError(
            "Reference M-valley C2 partner projector currently supports bilayer twist_group=[0, 1], "
            + f"got {unique_groups}."
        )

    atom_type_all = df_temp["atom_type"].to_numpy(dtype=int, copy=False)
    orb_num_all = df_temp["orb_num"].to_numpy(dtype=int, copy=False)
    twist_group_all = df_temp["twist_group"].to_numpy(dtype=int, copy=False)
    pos_array = df_temp[["shifted_x", "shifted_y"]].to_numpy(dtype=float, copy=False)

    atom_type_list = np.unique(atom_type_all)
    if not np.array_equal(atom_type_list, np.arange(atom_type_list.size)):
        raise ValueError(f"atom_type must be 0..n_types-1, got {atom_type_list}")

    n_types = int(atom_type_list.size)
    atom_orb_num_list = np.zeros(n_types, dtype=int)
    atom_num_list = np.zeros(n_types, dtype=int)
    atom_twist_group_list = np.zeros(n_types, dtype=int)
    atom_orb_name_list = np.zeros(n_types, dtype=object)

    for atom_type in range(n_types):
        mask = atom_type_all == atom_type
        if not np.any(mask):
            raise ValueError(f"atom_type {atom_type} has no atoms")

        atom_orb_num_list[atom_type] = int(orb_num_all[mask][0])
        if not np.all(orb_num_all[mask] == atom_orb_num_list[atom_type]):
            raise ValueError(f"Inconsistent orb_num for atom_type {atom_type}")

        atom_num_list[atom_type] = int(np.sum(mask))
        atom_twist_group_list[atom_type] = int(twist_group_all[mask][0])
        if not np.all(twist_group_all[mask] == atom_twist_group_list[atom_type]):
            raise ValueError(f"Inconsistent twist_group for atom_type {atom_type}")

        atom_orb_name_list[atom_type] = df_temp.loc[mask, "orb_name"].iloc[0]

    factor_list = 1.0 / np.sqrt(atom_num_list.astype(np.float64))
    orb_group_num = np.zeros(2, dtype=int)
    for group_id in range(2):
        orb_group_num[group_id] = int(atom_orb_num_list[atom_twist_group_list == group_id].sum())

    source_g_lists = {
        0: np.asarray(reference_params.g_vec_list_K1, dtype=float),
        1: np.asarray(reference_params.g_vec_list_K2, dtype=float),
    }
    g_group_num = np.array([len(source_g_lists[0]), len(source_g_lists[1])], dtype=int)

    shift3 = np.zeros(2, dtype=int)
    for group_id in range(2):
        shift3[group_id] = int(np.dot(g_group_num[:group_id], orb_group_num[:group_id]))

    shift1 = np.zeros(n_types, dtype=int)
    for atom_type in range(n_types):
        group_id = atom_twist_group_list[atom_type]
        if atom_type > 0:
            shift1[atom_type] = int(
                atom_orb_num_list[:atom_type][atom_twist_group_list[:atom_type] == group_id].sum()
            )

    col_offsets = np.cumsum(np.r_[0, orb_num_all[:-1]]).astype(np.int64)
    rotation_matrix = np.asarray(resolved_symmetry.rotation_cart, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(
            f"resolved_symmetry.rotation_cart must have shape (3, 3), got {rotation_matrix.shape}"
        )

    translation_cart = np.asarray(getattr(resolved_symmetry, "translation_cart", np.zeros(3)), dtype=float)
    if translation_cart.shape != (3,):
        raise ValueError(
            f"resolved_symmetry.translation_cart must have shape (3,), got {translation_cart.shape}"
        )

    group_target_maps = {
        0: {int(k): int(v) for k, v in getattr(resolved_symmetry, "atom_type_map_0_to_1").items()},
        1: {int(k): int(v) for k, v in getattr(resolved_symmetry, "atom_type_map_1_to_0").items()},
    }

    orb_mapping = {
        "s": get_any_rot_orb_twostep("s", rotation_matrix),
        "p": get_any_rot_orb_twostep("p", rotation_matrix),
        "d": get_any_rot_orb_twostep("d", rotation_matrix),
        "f": get_any_rot_orb_twostep("f", rotation_matrix),
    }

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    for source_group in (0, 1):
        target_group = 1 - source_group
        source_atom_types = atom_type_list[atom_twist_group_list == source_group]
        source_g_list = source_g_lists[source_group]
        for gi, q_source in enumerate(source_g_list):
            q_source = np.asarray(q_source, dtype=float)
            q_target = rotation_matrix[:2, :2] @ q_source
            translation_phase = np.exp(1.0j * np.dot(q_target, translation_cart[:2]))
            group_row_shift = int(shift3[source_group] + gi * orb_group_num[source_group])

            for source_atom_type in source_atom_types:
                target_atom_type = group_target_maps[source_group][int(source_atom_type)]
                source_orbitals = _parse_orbitals_from_orb_name(str(atom_orb_name_list[source_atom_type]))
                target_orbitals = _parse_orbitals_from_orb_name(str(atom_orb_name_list[target_atom_type]))
                if source_orbitals != target_orbitals:
                    raise ValueError(
                        "Reference M-valley C2 partner projector requires matched orbital content across mapped "
                        + f"atom types; source atom_type={source_atom_type} {source_orbitals} vs "
                        + f"target atom_type={target_atom_type} {target_orbitals}."
                    )

                orb_block = direct_sum(*generate_direct_sum_params(source_orbitals, orb_mapping))
                row_block = orb_block.conj().T

                orb_num = int(atom_orb_num_list[source_atom_type])
                if row_block.shape != (orb_num, orb_num):
                    raise ValueError(
                        f"Orbital rotation block has shape {row_block.shape}, expected {(orb_num, orb_num)}."
                    )

                row_base = int(group_row_shift + shift1[source_atom_type])
                row_idx = row_base + np.arange(orb_num, dtype=np.int64)
                target_atom_idx = np.nonzero(
                    (twist_group_all == target_group) & (atom_type_all == target_atom_type)
                )[0]
                if target_atom_idx.size == 0:
                    raise ValueError(
                        f"No target atoms found for mapped atom_type={target_atom_type} in twist_group={target_group}."
                    )

                phase_atoms = translation_phase * np.exp(-1.0j * (pos_array[target_atom_idx] @ q_target))
                col_base = col_offsets[target_atom_idx]
                for atom_phase, atom_col_base in zip(phase_atoms, col_base):
                    rows_parts.append(np.repeat(row_idx, orb_num))
                    cols_parts.append(
                        np.tile(atom_col_base + np.arange(orb_num, dtype=np.int64), orb_num)
                    )
                    data_parts.append((atom_phase * factor_list[source_atom_type] * row_block).reshape(-1))

    dim_rows = int(np.dot(g_group_num, orb_group_num))
    dim_cols = int(np.sum(atom_num_list * atom_orb_num_list))
    projector = scipy.sparse.coo_matrix(
        (
            np.concatenate(data_parts).astype(np.complex128, copy=False),
            (np.concatenate(rows_parts), np.concatenate(cols_parts)),
        ),
        shape=(dim_rows, dim_cols),
        dtype=np.complex128,
    ).tocsr()
    projector.sort_indices()

    if getattr(structure, "spin", False):
        spin_rep = spin_reps(rotation_matrix).conj().T
        projector = scipy.sparse.csr_matrix(np.kron(spin_rep, projector.toarray()))
    return projector


def _build_reference_m_valley_transformed_projector(
    structure,
    reference_params,
    rotation_matrix: np.ndarray,
    group_target_maps: dict[int, dict[int, int]],
    translation_cart: np.ndarray | None = None,
):
    if getattr(reference_params, "valley", None) not in _M_VALLEY_TRIPLET:
        raise ValueError(f"Reference transformed projector only supports M valleys {_M_VALLEY_TRIPLET}.")
    if not hasattr(structure, "df") or structure.df is None:
        raise ValueError("structure must provide a populated df for the reference M-valley transformed projector.")

    required_cols = {"atom_type", "orb_num", "twist_group", "shifted_x", "shifted_y", "orb_name"}
    missing = required_cols.difference(structure.df.columns)
    if missing:
        raise ValueError(
            "structure.df is missing columns required for the reference M-valley transformed projector: "
            + ", ".join(sorted(missing))
        )

    df_temp = structure.df.copy().sort_values(["atom_type"], kind="stable").reset_index(drop=True)
    unique_groups = sorted(df_temp["twist_group"].unique().tolist())
    if unique_groups != [0, 1]:
        raise ValueError(
            "Reference M-valley transformed projector currently supports bilayer twist_group=[0, 1], "
            + f"got {unique_groups}."
        )

    atom_type_all = df_temp["atom_type"].to_numpy(dtype=int, copy=False)
    orb_num_all = df_temp["orb_num"].to_numpy(dtype=int, copy=False)
    twist_group_all = df_temp["twist_group"].to_numpy(dtype=int, copy=False)
    pos_array = df_temp[["shifted_x", "shifted_y"]].to_numpy(dtype=float, copy=False)

    atom_type_list = np.unique(atom_type_all)
    if not np.array_equal(atom_type_list, np.arange(atom_type_list.size)):
        raise ValueError(f"atom_type must be 0..n_types-1, got {atom_type_list}")

    n_types = int(atom_type_list.size)
    atom_orb_num_list = np.zeros(n_types, dtype=int)
    atom_num_list = np.zeros(n_types, dtype=int)
    atom_twist_group_list = np.zeros(n_types, dtype=int)
    atom_orb_name_list = np.zeros(n_types, dtype=object)

    for atom_type in range(n_types):
        mask = atom_type_all == atom_type
        if not np.any(mask):
            raise ValueError(f"atom_type {atom_type} has no atoms")

        atom_orb_num_list[atom_type] = int(orb_num_all[mask][0])
        if not np.all(orb_num_all[mask] == atom_orb_num_list[atom_type]):
            raise ValueError(f"Inconsistent orb_num for atom_type {atom_type}")

        atom_num_list[atom_type] = int(np.sum(mask))
        atom_twist_group_list[atom_type] = int(twist_group_all[mask][0])
        if not np.all(twist_group_all[mask] == atom_twist_group_list[atom_type]):
            raise ValueError(f"Inconsistent twist_group for atom_type {atom_type}")

        atom_orb_name_list[atom_type] = df_temp.loc[mask, "orb_name"].iloc[0]

    factor_list = 1.0 / np.sqrt(atom_num_list.astype(np.float64))
    orb_group_num = np.zeros(2, dtype=int)
    for group_id in range(2):
        orb_group_num[group_id] = int(atom_orb_num_list[atom_twist_group_list == group_id].sum())

    source_g_lists = {
        0: np.asarray(reference_params.g_vec_list_K1, dtype=float),
        1: np.asarray(reference_params.g_vec_list_K2, dtype=float),
    }
    g_group_num = np.array([len(source_g_lists[0]), len(source_g_lists[1])], dtype=int)

    shift3 = np.zeros(2, dtype=int)
    for group_id in range(2):
        shift3[group_id] = int(np.dot(g_group_num[:group_id], orb_group_num[:group_id]))

    shift1 = np.zeros(n_types, dtype=int)
    for atom_type in range(n_types):
        group_id = atom_twist_group_list[atom_type]
        if atom_type > 0:
            shift1[atom_type] = int(
                atom_orb_num_list[:atom_type][atom_twist_group_list[:atom_type] == group_id].sum()
            )

    normalized_group_maps = {
        int(source_group): {int(k): int(v) for k, v in target_map.items()}
        for source_group, target_map in group_target_maps.items()
    }
    for source_group in (0, 1):
        if source_group not in normalized_group_maps:
            raise ValueError(f"group_target_maps is missing source_group={source_group}.")

    col_offsets = np.cumsum(np.r_[0, orb_num_all[:-1]]).astype(np.int64)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {rotation_matrix.shape}")

    if translation_cart is None:
        translation_cart = np.zeros(3, dtype=float)
    translation_cart = np.asarray(translation_cart, dtype=float)
    if translation_cart.shape != (3,):
        raise ValueError(f"translation_cart must have shape (3,), got {translation_cart.shape}")

    orb_mapping = {
        "s": get_any_rot_orb_twostep("s", rotation_matrix),
        "p": get_any_rot_orb_twostep("p", rotation_matrix),
        "d": get_any_rot_orb_twostep("d", rotation_matrix),
        "f": get_any_rot_orb_twostep("f", rotation_matrix),
    }

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []

    for source_group in (0, 1):
        source_atom_types = atom_type_list[atom_twist_group_list == source_group]
        source_g_list = source_g_lists[source_group]
        target_map = normalized_group_maps[source_group]
        for gi, q_source in enumerate(source_g_list):
            q_source = np.asarray(q_source, dtype=float)
            q_target = rotation_matrix[:2, :2] @ q_source
            translation_phase = np.exp(1.0j * np.dot(q_target, translation_cart[:2]))
            group_row_shift = int(shift3[source_group] + gi * orb_group_num[source_group])

            for source_atom_type in source_atom_types:
                if int(source_atom_type) not in target_map:
                    raise ValueError(
                        f"group_target_maps[{source_group}] is missing source atom_type={int(source_atom_type)}."
                    )
                target_atom_type = int(target_map[int(source_atom_type)])
                target_group = int(atom_twist_group_list[target_atom_type])

                source_orbitals = _parse_orbitals_from_orb_name(str(atom_orb_name_list[source_atom_type]))
                target_orbitals = _parse_orbitals_from_orb_name(str(atom_orb_name_list[target_atom_type]))
                if source_orbitals != target_orbitals:
                    raise ValueError(
                        "Reference transformed projector requires matched orbital content across mapped atom types; "
                        + f"source atom_type={source_atom_type} {source_orbitals} vs "
                        + f"target atom_type={target_atom_type} {target_orbitals}."
                    )

                orb_block = direct_sum(*generate_direct_sum_params(source_orbitals, orb_mapping))
                row_block = orb_block.conj().T
                orb_num = int(atom_orb_num_list[source_atom_type])
                if row_block.shape != (orb_num, orb_num):
                    raise ValueError(
                        f"Orbital rotation block has shape {row_block.shape}, expected {(orb_num, orb_num)}."
                    )

                row_base = int(group_row_shift + shift1[source_atom_type])
                row_idx = row_base + np.arange(orb_num, dtype=np.int64)
                target_atom_idx = np.nonzero(
                    (twist_group_all == target_group) & (atom_type_all == target_atom_type)
                )[0]
                if target_atom_idx.size == 0:
                    raise ValueError(
                        f"No target atoms found for mapped atom_type={target_atom_type} in twist_group={target_group}."
                    )

                phase_atoms = translation_phase * np.exp(-1.0j * (pos_array[target_atom_idx] @ q_target))
                col_base = col_offsets[target_atom_idx]
                for atom_phase, atom_col_base in zip(phase_atoms, col_base):
                    rows_parts.append(np.repeat(row_idx, orb_num))
                    cols_parts.append(
                        np.tile(atom_col_base + np.arange(orb_num, dtype=np.int64), orb_num)
                    )
                    data_parts.append((atom_phase * factor_list[source_atom_type] * row_block).reshape(-1))

    dim_rows = int(np.dot(g_group_num, orb_group_num))
    dim_cols = int(np.sum(atom_num_list * atom_orb_num_list))
    projector = scipy.sparse.coo_matrix(
        (
            np.concatenate(data_parts).astype(np.complex128, copy=False),
            (np.concatenate(rows_parts), np.concatenate(cols_parts)),
        ),
        shape=(dim_rows, dim_cols),
        dtype=np.complex128,
    ).tocsr()
    projector.sort_indices()

    if getattr(structure, "spin", False):
        spin_rep = spin_reps(rotation_matrix).conj().T
        projector = scipy.sparse.csr_matrix(np.kron(spin_rep, projector.toarray()))
    return projector


def build_projected_m_valley_transport(structure, source_params, target_params):
    angle_deg = _m_valley_rotation_delta(source_params.valley, target_params.valley)
    df_temp = structure.df.copy()
    unique_groups = sorted(df_temp["twist_group"].unique().tolist())
    group_orb_blocks, spin_rep = _build_group_orbital_rotation_blocks(df_temp, angle_deg, structure.spin)
    blocks = []

    for gid in unique_groups:
        source_g = source_params.g_vec_list_K1 if (gid % 2 == 0) else source_params.g_vec_list_K2
        target_g = target_params.g_vec_list_K1 if (gid % 2 == 0) else target_params.g_vec_list_K2
        g_transport = _build_rotation_match_matrix(source_g, target_g, angle_deg)
        blocks.append(np.kron(g_transport, group_orb_blocks[gid]))

    transport = direct_sum(*blocks)
    if structure.spin:
        transport = np.kron(spin_rep, transport)
    return scipy.sparse.csr_matrix(transport)


def build_reference_m_valley_c2_transport(structure, reference_params):
    if reference_params.valley not in _M_VALLEY_TRIPLET:
        raise ValueError(f"Reference C2 transport only supports M valleys {_M_VALLEY_TRIPLET}.")
    if "twist_group" not in structure.df.columns:
        raise ValueError("structure.df must contain 'twist_group' column for reference M-valley C2 transport.")

    unique_groups = sorted(structure.df["twist_group"].unique().tolist())
    if unique_groups != [0, 1]:
        raise ValueError(
            f"Reference M-valley C2 transport currently supports bilayer twist_group=[0, 1], got {unique_groups}."
        )

    resolved_symmetry = None
    can_resolve_from_structure = bool(
        hasattr(structure, "Tmat")
        and {"x", "y", "z", "species", "twist_group", "atom_type"}.issubset(structure.df.columns)
    )
    if can_resolve_from_structure:
        resolved_symmetry = resolve_reference_m_valley_c2_symmetry(structure, reference_params)
        reflection_2d = reference_m_valley_c2_linear_map(reference_params)
        m_k1 = getattr(reference_params, "m_K1", None)
        m_k2 = getattr(reference_params, "m_K2", None)
        if m_k1 is None or m_k2 is None:
            _, _, m_k1, m_k2, _ = reference_params.calculate_K_points()
        axis_vec = np.asarray(m_k1, dtype=float) + np.asarray(m_k2, dtype=float)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm < 1.0e-12:
            raise ValueError("Cannot determine the reference C2 axis because m_K1 + m_K2 is numerically zero.")
        axis_vec = axis_vec / axis_norm
        # Keep the legacy projected-basis C2 axis for the orbital/spin rotation blocks.
        # The resolved spglib operation is still used to derive the cross-layer atom-type mapping.
        rotation_matrix = rotate_mat(np.array([axis_vec[0], axis_vec[1], 0.0]), np.pi)
        translation_cart = np.asarray(resolved_symmetry.translation_cart, dtype=float)
        if np.linalg.norm(translation_cart[:2]) > 1.0e-8:
            raise ValueError(
                "Reference M-valley C2 transport currently requires zero in-plane fractional translation; "
                + f"got translation_cart[:2]={translation_cart[:2]!r}."
            )
        target_map_0_to_1 = resolved_symmetry.atom_type_map_0_to_1
        target_map_1_to_0 = resolved_symmetry.atom_type_map_1_to_0
    else:
        # Legacy geometric fallback for synthetic/unit-test structures that do not
        # carry the full Cartesian coordinates needed for spglib symmetry recovery.
        m_k1 = getattr(reference_params, "m_K1", None)
        m_k2 = getattr(reference_params, "m_K2", None)
        if m_k1 is None or m_k2 is None:
            _, _, m_k1, m_k2, _ = reference_params.calculate_K_points()
        axis_vec = np.asarray(m_k1, dtype=float) + np.asarray(m_k2, dtype=float)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm < 1.0e-12:
            raise ValueError("Cannot determine the reference C2 axis because m_K1 + m_K2 is numerically zero.")
        axis_vec = axis_vec / axis_norm
        reflection_2d = reference_m_valley_c2_linear_map(reference_params)
        rotation_matrix = rotate_mat(np.array([axis_vec[0], axis_vec[1], 0.0]), np.pi)
        target_map_0_to_1 = None
        target_map_1_to_0 = None

    c2t_spin_rep = reference_m_valley_c2_spin_unitary(rotation_matrix) if structure.spin else None

    group_transport_0_to_1, spin_rep = _build_group_orbital_transport_between_groups(
        structure_df=structure.df.copy(),
        source_group=0,
        target_group=1,
        rotation_matrix=rotation_matrix,
        spin=structure.spin,
        spin_rep=c2t_spin_rep,
        target_for_source=target_map_0_to_1,
    )
    group_transport_1_to_0, _ = _build_group_orbital_transport_between_groups(
        structure_df=structure.df.copy(),
        source_group=1,
        target_group=0,
        rotation_matrix=rotation_matrix,
        spin=structure.spin,
        spin_rep=spin_rep,
        target_for_source=target_map_1_to_0,
    )

    if group_transport_0_to_1.shape[1] != group_transport_1_to_0.shape[0] or group_transport_0_to_1.shape[0] != group_transport_1_to_0.shape[1]:
        raise ValueError(
            "Reference M-valley C2 transport requires matched orbital transport blocks across the two twist groups; "
            + f"got {group_transport_0_to_1.shape} and {group_transport_1_to_0.shape}."
        )

    g_0_to_1 = _build_linear_match_matrix(
        reference_params.g_vec_list_K1,
        reference_params.g_vec_list_K2,
        reflection_2d,
    )
    g_1_to_0 = _build_linear_match_matrix(
        reference_params.g_vec_list_K2,
        reference_params.g_vec_list_K1,
        reflection_2d,
    )

    block_0_to_1 = np.kron(g_0_to_1, group_transport_0_to_1)
    block_1_to_0 = np.kron(g_1_to_0, group_transport_1_to_0)
    dim0 = block_0_to_1.shape[1]
    dim1 = block_1_to_0.shape[1]
    if block_1_to_0.shape[0] != dim0 or block_0_to_1.shape[0] != dim1:
        raise ValueError(
            "Reference M-valley C2 transport block dimensions are inconsistent with the projected basis: "
            f"0->1 shape={block_0_to_1.shape}, 1->0 shape={block_1_to_0.shape}."
        )

    zero_00 = np.zeros((dim0, dim0), dtype=np.complex128)
    zero_11 = np.zeros((dim1, dim1), dtype=np.complex128)
    transport = np.block(
        [
            [zero_00, block_1_to_0],
            [block_0_to_1, zero_11],
        ]
    )
    if structure.spin:
        transport = np.kron(spin_rep, transport)
    return scipy.sparse.csr_matrix(transport)


def reference_m_valley_c2_linear_map(reference_params) -> np.ndarray:
    m_k1 = getattr(reference_params, "m_K1", None)
    m_k2 = getattr(reference_params, "m_K2", None)
    if m_k1 is None or m_k2 is None:
        _, _, m_k1, m_k2, _ = reference_params.calculate_K_points()
    axis_vec = np.asarray(m_k1, dtype=float) + np.asarray(m_k2, dtype=float)
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1.0e-12:
        raise ValueError("Cannot determine the reference C2 axis because m_K1 + m_K2 is numerically zero.")
    axis_vec = axis_vec / axis_norm
    return 2.0 * np.outer(axis_vec, axis_vec) - np.eye(2, dtype=float)


def _write_progress_state(progress_dir: str | None, index: int, stage: str) -> None:
    # Progress state files were removed to avoid littering result directories.
    return


@contextmanager
def _progress_reporter(total: int):
    yield


@contextmanager
def _tqdm_joblib(total: int, desc: str):
    """Patch joblib callbacks so tqdm reflects completed tasks, not dispatched ones."""
    original_callback = joblib_parallel.BatchCompletionCallBack
    pbar = tqdm(total=total, desc=desc, unit="kpt", mininterval=5.0, dynamic_ncols=True)

    class _TqdmBatchCompletionCallBack(original_callback):
        def __call__(self, *args, **kwargs):
            pbar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib_parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallBack
    try:
        yield pbar
    finally:
        joblib_parallel.BatchCompletionCallBack = original_callback
        pbar.close()


def _get_memmap(path: str) -> np.memmap:
    mm = _MEMMAP_CACHE.get(path)
    if mm is None:
        mm = open_memmap(path, mode="r+")
        _MEMMAP_CACHE[path] = mm
    return mm


def _acquire_mkdir_lock(lock_dir: str, poll_s: float = 0.2, max_wait_s: float = 600.0) -> None:
    # Robust-enough cross-node lock using atomic mkdir.
    # If a job dies while holding the lock, users may need to manually delete the lock dir.
    deadline = time.time() + max_wait_s
    while True:
        try:
            os.mkdir(lock_dir)
            return
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for lock: {lock_dir}")
            time.sleep(poll_s)


def _release_mkdir_lock(lock_dir: str) -> None:
    with suppress(OSError):
        os.rmdir(lock_dir)


def _ensure_memmap_file(path: str, dtype, shape: tuple[int, ...]) -> None:
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    lock_dir = path + ".lock"
    _acquire_mkdir_lock(lock_dir)
    try:
        if not os.path.exists(path):
            mm = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            mm.flush()
            del mm
        else:
            mm = open_memmap(path, mode="r+")
            if mm.dtype != np.dtype(dtype) or mm.shape != tuple(shape):
                raise ValueError(f"Memmap {path} has dtype={mm.dtype}, shape={mm.shape}, expected {dtype}, {shape}")
            del mm
    finally:
        _release_mkdir_lock(lock_dir)


_MP_STATE: dict[str, object] = {}

_TPCTL_LIMITER = None
_TPCTL_NTHREADS: int | None = None
_AFFINITY_PINNED_PIDS: set[int] = set()


def _set_thread_limits(nthreads: int | None) -> None:
    """Best-effort BLAS/OpenMP thread limiting inside a process.

    Why:
      - In MKL builds, default thread counts can be large (tens of threads).
      - If you also parallelize k-points with multiprocessing, you can easily get
        massive oversubscription (hundreds of runnable threads) and slowdowns.
    """
    global _TPCTL_LIMITER, _TPCTL_NTHREADS
    if nthreads is None:
        return
    try:
        n = int(nthreads)
    except Exception:
        return
    if n <= 0:
        return
    if _TPCTL_NTHREADS == n:
        return

    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        # threadpool_limits applies limits on construction; keep a global ref so
        # it is not garbage-collected (which could restore defaults).
        _TPCTL_LIMITER = threadpool_limits(limits=n)
        _TPCTL_NTHREADS = n
    except Exception:
        _TPCTL_LIMITER = None
        _TPCTL_NTHREADS = None


def _maybe_pin_current_worker(slot_width: int | None, worker_count: int | None) -> None:
    """Best-effort CPU affinity for loky/joblib workers.

    Why:
      - On multi-socket nodes, many small OpenMP teams can drift across NUMA domains.
      - Pinning each worker to a disjoint chunk of the Slurm cpuset reduces cross-socket traffic.
      - This is only attempted inside worker processes whose names look like `LokyProcess-N`.
    """
    global _AFFINITY_PINNED_PIDS

    if slot_width is None or worker_count is None:
        return
    try:
        width = int(slot_width)
        nworkers = int(worker_count)
    except Exception:
        return
    if width <= 0 or nworkers <= 1:
        return
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return

    pid = os.getpid()
    if pid in _AFFINITY_PINNED_PIDS:
        return

    proc_name = mp.current_process().name
    match = re.search(r"(\d+)$", proc_name or "")
    if match is None:
        return
    worker_slot = max(int(match.group(1)) - 1, 0)

    try:
        allowed_cpus = sorted(os.sched_getaffinity(0))
    except Exception:
        return
    if not allowed_cpus:
        return

    width = min(width, len(allowed_cpus))
    slot_count = max(len(allowed_cpus) // width, 1)
    slot = worker_slot % slot_count
    start = slot * width
    cpu_chunk = allowed_cpus[start : start + width]
    if not cpu_chunk:
        cpu_chunk = allowed_cpus

    try:
        os.sched_setaffinity(0, set(cpu_chunk))
        _AFFINITY_PINNED_PIDS.add(pid)
    except Exception:
        return


def _mp_worker_init(blas_threads: int, worker_count: int | None = None) -> None:
    # Called once per multiprocessing worker process.
    _set_thread_limits(blas_threads)
    _maybe_pin_current_worker(blas_threads, worker_count)


def _solve_eigs_slepc(hamk, samk, cfg: ComputeConfig) -> np.ndarray:
    """SLEPc (slepc4py) eigenvalue solve for a single k-point (eigenvalues only)."""
    try:
        from petsc4py import PETSc  # type: ignore
        from slepc4py import SLEPc  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "eigensolver='slepc' requested but petsc4py/slepc4py is not available. "
            "Install them into the active environment (tapw-mkl) and retry. "
            f"Original import error: {e!r}"
        ) from e

    if getattr(cfg, "eig_vec_cal", False):
        raise ValueError("eigensolver='slepc' currently supports eig_vec_cal=false only.")

    slepc_comm = str(getattr(cfg, "slepc_comm", "self")).lower()
    comm = PETSc.COMM_WORLD if slepc_comm == "world" else PETSc.COMM_SELF

    A_csr_full = hamk.tocsr() if scipy.sparse.issparse(hamk) else scipy.sparse.csr_matrix(hamk)
    B_csr_full = None
    if samk is not None:
        B_csr_full = samk.tocsr() if scipy.sparse.issparse(samk) else scipy.sparse.csr_matrix(samk)

    # SLEPc GHEP assumes A and (especially) B define a valid Hermitian inner product.
    # Tiny non-Hermitian noise (from file truncation / k-phase rounding) can make v^H B v
    # have a small imaginary part and SLEPc will abort with error code 95.
    if bool(getattr(cfg, "slepc_make_hermitian", True)):
        A_csr_full = (A_csr_full + A_csr_full.getH()) * 0.5
        if B_csr_full is not None:
            B_csr_full = (B_csr_full + B_csr_full.getH()) * 0.5

    spd_shift = float(getattr(cfg, "slepc_spd_shift", 0.0) or 0.0)
    if spd_shift and B_csr_full is not None:
        # Very small diagonal regularization can help if S(k) is near-singular/indefinite.
        n = int(B_csr_full.shape[0])
        B_csr_full = B_csr_full + (spd_shift * scipy.sparse.identity(n, format="csr", dtype=B_csr_full.dtype))

    if np.iscomplexobj(A_csr_full.data) or (B_csr_full is not None and np.iscomplexobj(B_csr_full.data)):
        if np.dtype(PETSc.ScalarType).kind != "c":
            raise RuntimeError(
                "H(k)/S(k) is complex but PETSc in this environment is real-valued. "
                "Install complex PETSc/SLEPc (conda-forge complex build) and retry."
            )

    n_global = int(A_csr_full.shape[0])
    if comm.getSize() > 1:
        # Row-block distribution: each rank stores its local row slice in PETSc.
        # NOTE: each rank still constructs the full scipy CSR and then slices.
        rank = comm.getRank()
        size = comm.getSize()
        rstart = (rank * n_global) // size
        rend = ((rank + 1) * n_global) // size
        A_csr = A_csr_full[rstart:rend, :].tocsr()
        B_csr = B_csr_full[rstart:rend, :].tocsr() if B_csr_full is not None else None
        A_csr_full = None
        B_csr_full = None
    else:
        A_csr = A_csr_full
        B_csr = B_csr_full

    indptr = np.asarray(A_csr.indptr, dtype=np.int32)
    indices = np.asarray(A_csr.indices, dtype=np.int32)
    data = np.asarray(A_csr.data, dtype=PETSc.ScalarType, order="C")
    if comm.getSize() > 1:
        # petsc4py expects size=((m,M),(n,N)) where (m,n) are local sizes.
        A = PETSc.Mat().createAIJ(
            # Make the parallel row/col layout identical (required by EPS for square operators).
            size=((int(A_csr.shape[0]), n_global), (int(A_csr.shape[0]), n_global)),
            csr=(indptr, indices, data),
            comm=comm,
        )
    else:
        A = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(indptr, indices, data), comm=comm)
    A.assemble()
    A.setOption(PETSc.Mat.Option.HERMITIAN, True)

    B = None
    if B_csr is not None:
        indptr_b = np.asarray(B_csr.indptr, dtype=np.int32)
        indices_b = np.asarray(B_csr.indices, dtype=np.int32)
        data_b = np.asarray(B_csr.data, dtype=PETSc.ScalarType, order="C")
        if comm.getSize() > 1:
            B = PETSc.Mat().createAIJ(
                size=((int(B_csr.shape[0]), n_global), (int(B_csr.shape[0]), n_global)),
                csr=(indptr_b, indices_b, data_b),
                comm=comm,
            )
        else:
            B = PETSc.Mat().createAIJ(size=B_csr.shape, csr=(indptr_b, indices_b, data_b), comm=comm)
        B.assemble()
        B.setOption(PETSc.Mat.Option.HERMITIAN, True)
        B.setOption(PETSc.Mat.Option.SPD, True)

    eps = SLEPc.EPS().create(comm=comm)
    if B is None:
        eps.setOperators(A)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    else:
        eps.setOperators(A, B)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

    eps.setType(str(getattr(cfg, "slepc_eps_type", "krylovschur")))
    nev = int(getattr(cfg, "num_bands_cal", 0))
    if nev <= 0:
        raise ValueError("num_bands_cal must be > 0 for SLEPc.")
    eps.setDimensions(nev, PETSc.DECIDE)

    target = float(getattr(cfg, "efermi", 0.0))
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eps.setTarget(target)

    st = eps.getST()
    st.setType(str(getattr(cfg, "slepc_st_type", "sinvert")))
    st.setShift(target)

    ksp = st.getKSP()
    ksp.setType(str(getattr(cfg, "slepc_ksp_type", "preonly")))
    pc = ksp.getPC()
    pc.setType(str(getattr(cfg, "slepc_pc_type", "lu")))
    factor = str(getattr(cfg, "slepc_factor_mat_solver_type", "") or "").strip()
    if factor and factor.lower() != "petsc":
        pc.setFactorSolverType(factor)

    tol = float(getattr(cfg, "slepc_tol", 1e-8))
    max_it = int(getattr(cfg, "slepc_max_it", 5000))
    eps.setTolerances(tol, max_it)
    eps.setFromOptions()
    try:
        eps.solve()
    except Exception as e:
        # Re-raise with a more actionable message for the most common failure mode.
        msg = str(e)
        if "error code 95" in msg or "inner product is not well defined" in msg:
            raise RuntimeError(
                "SLEPc failed with an inner-product error (often caused by S(k) not being strictly Hermitian/HPD). "
                "Try setting compute.slepc_make_hermitian: true (default) and/or a small compute.slepc_spd_shift "
                "(e.g. 1e-10). If it still fails, your overlap matrix may not satisfy GHEP assumptions."
            ) from e
        raise

    nconv = int(eps.getConverged())
    if nconv <= 0:
        raise RuntimeError("SLEPc failed to converge any eigenpairs for this k-point.")
    nret = min(nev, nconv)
    evals = np.empty(nret, dtype=float)
    for i in range(nret):
        lam = eps.getEigenvalue(i)
        evals[i] = float(np.real(lam))
    return np.sort(evals)


def _pick_available_slepc_factor_backend() -> str:
    """Probe PETSc LU packages and return the best available backend."""
    key = (sys.executable, os.environ.get("LD_LIBRARY_PATH", ""))
    cached = _SLEPC_FACTOR_PROBE_CACHE.get(key)
    if cached:
        return cached

    from petsc4py import PETSc  # type: ignore

    probe = PETSc.Mat().createAIJ(
        size=(4, 4),
        csr=(
            np.array([0, 2, 4, 6, 8], dtype=np.int32),
            np.array([0, 1, 1, 2, 2, 3, 0, 3], dtype=np.int32),
            np.array([4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 1.0, 4.0], dtype=PETSc.ScalarType),
        ),
        comm=PETSc.COMM_SELF,
    )
    probe.assemble()

    for candidate in _NOTAPW_SLEPC_FACTOR_CANDIDATES:
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        try:
            ksp.setOperators(probe)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            if candidate.lower() != "petsc":
                pc.setFactorSolverType(candidate)
            ksp.setUp()
            _SLEPC_FACTOR_PROBE_CACHE[key] = candidate
            return candidate
        except Exception:
            continue
        finally:
            with suppress(Exception):
                ksp.destroy()

    _SLEPC_FACTOR_PROBE_CACHE[key] = "petsc"
    return "petsc"


def _solve_notapw_generalized_eigs(hamk, samk, cfg: ComputeConfig):
    """Solve the non-TAPW generalized problem using the configured backend.

    For `eigensolver="slepc"` we intentionally use a fixed single-node strategy
    that mirrors the benchmarked path:
    - COMM_SELF
    - shift-invert + LU
    - auto-probed PETSc LU backend (prefer MKL Pardiso when available)
    - small SPD shift on S
    """
    eigensolver = str(getattr(cfg, "eigensolver", "scipy")).lower()
    if eigensolver == "slepc":
        if getattr(cfg, "eig_vec_cal", False):
            raise ValueError("non-TAPW eigensolver='slepc' currently supports eig_vec_cal=false only.")

        factor_backend = _pick_available_slepc_factor_backend()
        slepc_cfg = SimpleNamespace(
            eig_vec_cal=False,
            slepc_comm="self",
            slepc_make_hermitian=True,
            slepc_spd_shift=_NOTAPW_SLEPC_SPD_SHIFT,
            num_bands_cal=int(getattr(cfg, "num_bands_cal", 0)),
            efermi=float(getattr(cfg, "efermi", 0.0)),
            slepc_eps_type="krylovschur",
            slepc_st_type="sinvert",
            slepc_ksp_type="preonly",
            slepc_pc_type="lu",
            slepc_factor_mat_solver_type=factor_backend,
            slepc_tol=_NOTAPW_SLEPC_TOL,
            slepc_max_it=_NOTAPW_SLEPC_MAX_IT,
        )
        return _solve_eigs_slepc(hamk, samk, slepc_cfg)

    return eigsh(
        hamk,
        k=cfg.num_bands_cal,
        M=samk,
        sigma=cfg.efermi,
        which="LM",
        return_eigenvectors=cfg.eig_vec_cal,
    )


def _mp_kpoint_worker(args: tuple[int, np.ndarray]) -> tuple[int, bool, str | None, object | None]:
    # Must be top-level for multiprocessing pickling.
    i, kpoint = args
    calc = _MP_STATE["calc"]  # BandStructureCalculator
    cfg = _MP_STATE["cfg"]  # ComputeConfig
    eig_path = _MP_STATE.get("eig_path")
    vec_path = _MP_STATE.get("vec_path")
    use_memmap = bool(_MP_STATE.get("use_memmap"))

    try:
        # Safety: in case the Pool initializer was not used for some reason.
        _set_thread_limits(getattr(cfg, "blas_threads", None))
        eig, vec, hamk, samk = calc.calculate_band_01(kpoint, i)

        if use_memmap and eig_path is not None:
            eig_mm = _get_memmap(str(eig_path))
            eig_mm[i, : eig.shape[0]] = eig
            if getattr(cfg, "eig_vec_cal", False) and vec_path is not None and vec is not None:
                vec_mm = _get_memmap(str(vec_path))
                vec_mm[i, :, : vec.shape[1]] = vec
            calc._set_progress_stage(i, "done")
            return i, True, None, None

        # In non-memmap mode, return the results to the parent so it can assemble
        # self.result["eig"]/["vec"] in memory (original behavior).
        calc._set_progress_stage(i, "done")
        return i, True, None, (eig, vec, hamk, samk)
    except Exception as e:  # pragma: no cover
        calc._set_progress_stage(i, "error")
        return i, False, str(e), None

class TAPW_parameters:
    """TAPW parameters for twisted material calculations"""
    
    def __init__(self, structure: StructureProcessorSpglib, config: ComputeConfig, reporter=None):
        self.config = config
        self.n_g = self.config.n_g
        self.valley = self.config.valley
        self.structure = structure
        self.reporter = ensure_reporter(reporter)
        self.bravais = getattr(self.config, "bravais", "hex")
        self.c3_h_disable_reason = None
        self.requested_symmetry_operations = requested_hamiltonian_symmetry_operations(self.config)
        self.use_single_valley_c3 = "C3z" in self.requested_symmetry_operations
        if getattr(self, "use_single_valley_c3", False):
            self.c3_h_disable_reason = single_valley_c3_incompatibility_reason(self.bravais, self.valley)
        self.use_single_valley_c3 = bool(self.use_single_valley_c3 and self.c3_h_disable_reason is None)

        # Initialize matrices
        self.g_matrix = None
        self.g_matrix_conj = None
        self.C3_matrix = None
        self.symm_matrix = None
        self.symm_matrix_inv = None
        self.g_symm_matrix = None
        self.g_symm_matrix_inv = None

        # K-points
        self.K1 = None
        self.K2 = None
        self.g_vec_list_K1 = None
        self.g_vec_list_K2 = None

        # 电场修正项
        self.electric_field_onsite = None  # shape: (num_atoms,)
        if self.config.Electric_field_in_eVpA or self.config.Inner_symmetrical_Electric_Field:
            self._compute_electric_field_onsite()

    def _compute_electric_field_onsite(self):
        """
        计算每个原子的电场能修正项（单位eV），存储在 self.electric_field_onsite
        """
        cfg = self.config
        df = self.structure.df
        if (cfg.Electric_field_in_eVpA is None and not cfg.Inner_symmetrical_Electric_Field):
            self.electric_field_onsite = None
            return
        # 计算零势能面z0

        # cfg.zero_potential_layers 是像 [2,3,4] 这样的全局 sublayer 列表
        global_idxs = set(cfg.zero_potential_layers)

        # 先构建 mask
        mask = False
        global_counter = 0

        # 按层遍历
        for layer in sorted(df['layer'].unique()):
            # 当前层有哪些 local sublayers
            local_subls = sorted(df.loc[df['layer']==layer, 'sublayer'].unique())
            for sub in local_subls:
                # 如果这个 global_counter 在你的列表里，就把对应的 (layer,sub) 全都选上
                if global_counter in global_idxs:
                    mask |= (df['layer']==layer) & (df['sublayer']==sub)
                global_counter += 1

        # 最后求 z 的平均
        if cfg.zero_potential_layers and mask.any():
            z0 = df.loc[mask, 'z'].mean()
        else:
            z0 = 0.0
        
        z = df['z'].values
        delta_z = z - z0
        onsite = np.zeros_like(z, dtype=np.float64)
        self.reporter.section("Electric field onsite correction")
        self.reporter.kv("Electric field (eV/Angstrom)", cfg.Electric_field_in_eVpA)
        self.reporter.array("delta_z", delta_z)
        # Electric_field_in_eVpA
        if cfg.Electric_field_in_eVpA is not None:
            onsite += cfg.Electric_field_in_eVpA * delta_z
        # Inner_symmetrical_Electric_Field
        if cfg.Inner_symmetrical_Electric_Field:
            isp = 0.005  # eV/Å
            # 指向零势能面
            onsite += isp * np.abs(delta_z)
        self.reporter.array("Onsite correction (eV)", onsite)
        self.electric_field_onsite = onsite

    def calculate_K_points(self):
        """Calculate the K1 and K2 points."""
        m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        m_g_unitvec_2 = self.structure.reciprocal_Tmat[1][:2]
        if self.structure.reciprocal_Tmat[0] @ self.structure.reciprocal_Tmat[1] < -0.01:
            m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]
        else:
            m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        
        n_moire = self.structure.twist_index
        
        if n_moire == 0:
            self.reporter.kv("n_moire=0 note", "only for non-twisted heterostructures")
            reciprocal_Tmat_layer1 = self.structure.monolayer_reciprocal_list[self.structure.twist_layer[0]-1]
            reciprocal_Tmat_layer2 = self.structure.monolayer_reciprocal_list[self.structure.twist_layer[0]]
            K1 = 1/3 * reciprocal_Tmat_layer1[0][:2] + 2/3 * reciprocal_Tmat_layer1[1][:2]
            K2 = 1/3 * reciprocal_Tmat_layer2[0][:2] + 2/3 * reciprocal_Tmat_layer2[1][:2]
            if self.config.valley == 2:
                K1 = -K1
                K2 = -K2
            elif self.config.valley != 1:
                raise ValueError("For non-twisted Heterostructures, only valley 1 (K1) and 2 (K2) are allowed now")
            offset = K2
            m_K1 = K1 - offset
            m_K2 = K2 - offset
            return K1, K2, m_K1, m_K2, offset

        bravais = getattr(self.config, "bravais", "hex")
        if bravais in {"square", "rect"}:
            # Use the moiré reciprocal basis (2D) to define Γ/X/Y/M in the moiré BZ
            g1 = m_g_unitvec_1
            g2 = m_g_unitvec_2

            # Moiré high-symmetry k-points (mBZ coordinates)
            m_Gamma = np.zeros(2)
            # m_X = 0.5 * g1
            # m_Y = 0.5 * g2
            # m_M = 0.5 * (g1 + g2)
            if n_moire%2 == 1:
                offset_X = 0.5 * (n_moire+1) * (g1 + g2)
                m_X1 = -0.5 * g2
                m_X2 = -0.5 * g1
            else:
                offset_X = 0.5 * n_moire * (g1 + g2)
                m_X1 = 0.5 * g1
                m_X2 = 0.5 * g2
            m_M1 = 0.5 * (g1 - g2)
            m_M2 = 0.5 * (g1 + g2)
            
            # Valley-dependent reciprocal-space offsets (commensurate indexing)
            offset_Gamma = np.zeros(2)
            # offset_X = n_moire * g1
            # offset_Y = n_moire * g2
            # offset_M = n_moire * (g1 + g2)
            
            # offset_X = 0.5 * n_moire * (g1 + g2)
            # offset_Y = 0.5 * n_moire * (-g1 + g2)
            offset_M = n_moire * g1
            
            self.reporter.section("Valley center setup")
            self.reporter.kv("g1", g1)
            self.reporter.kv("g2", g2)
            self.reporter.kv("n_moire", n_moire)
            self.reporter.kv("offset_X", offset_X)
            # print("offset_Y = ",offset_Y)
            self.reporter.kv("offset_M", offset_M)

            # For square/rect: valley_dict returns (K1, K2) centers used downstream;
            # mK_dict stores the moiré k-point (e.g. X = 1/2*g1).
            valley_dict = {
                5: (m_Gamma + offset_Gamma, m_Gamma + offset_Gamma),  # Γ
                41: (m_X1 + offset_X, m_X1 + offset_X),                  # X
                42: (rotate_vector(m_X1 + offset_X,90), rotate_vector(m_X2 + offset_X,90)),  # Y
                3: (m_M1 + offset_M, m_M2 + offset_M),                   # M
            }

            offset_dict = {
                5: offset_Gamma,
                41: offset_X,
                42: rotate_vector(offset_X,90),
                3: offset_M,
            }

            mK_dict = {
                5: (m_Gamma, m_Gamma),
                41: (m_X1, m_X2),
                42: (rotate_vector(m_X1,90), rotate_vector(m_X2,90)),
                3: (m_M1, m_M2),
            }

            if self.valley not in valley_dict:
                raise ValueError(
                    f"For bravais='{bravais}', supported valleys are {sorted(valley_dict.keys())}. Got {self.valley}"
                )

            K1, K2 = valley_dict[self.valley]
            m_K1, m_K2 = mK_dict[self.valley]
            offset = offset_dict[self.valley]
            self.reporter.section("Valley centers")
            self.reporter.kv("K1", K1)
            self.reporter.kv("K2", K2)
            self.reporter.kv("m_K1", m_K1)
            self.reporter.kv("m_K2", m_K2)
            self.reporter.kv("Offset", offset)
            return K1, K2, m_K1, m_K2, offset

        self.reporter.stage("Valley center setup", "Resolve the reciprocal-space centers for this valley.")
        self.reporter.detail("m_g_unitvec_1", m_g_unitvec_1, purpose="moire reciprocal basis vector")
        self.reporter.detail("m_g_unitvec_2", m_g_unitvec_2, purpose="moire reciprocal basis vector")
        offset = -n_moire * m_g_unitvec_1 + n_moire * m_g_unitvec_2

        m_K1 = -1/3 * m_g_unitvec_1 + 2/3 * m_g_unitvec_2
        m_K2 = -2/3 * m_g_unitvec_1 + 1/3 * m_g_unitvec_2
        
        if n_moire % 2 == 1:
            offset_1 = (n_moire + 1) * (m_g_unitvec_1 + m_g_unitvec_2) / 2
            offset_2 = rotate_vector(offset_1, 120)
            offset_3 = rotate_vector(offset_1, 240)
            m_M1 = -1/2 * m_g_unitvec_2
            m_M2 = -1/2 * m_g_unitvec_1
            m_M3 = rotate_vector(m_M2, 120)
        else:
            offset_1 = n_moire * (m_g_unitvec_1 + m_g_unitvec_2) / 2
            offset_2 = rotate_vector(offset_1, 120)
            offset_3 = rotate_vector(offset_1, 240)
            m_M1 = 1/2 * m_g_unitvec_1
            m_M2 = 1/2 * m_g_unitvec_2
            m_M3 = rotate_vector(m_M2, 120)

        valley_dict = {
            1: (m_K1 + offset, m_K2 + offset),
            2: (-m_K1 - offset, -m_K2 - offset),
            11: (rotate_vector(m_K1 + offset, 120), rotate_vector(m_K2 + offset, 120)),
            12: (rotate_vector(m_K1 + offset, 240), rotate_vector(m_K2 + offset, 240)),
            5: (np.zeros(2), np.zeros(2)),
            31: (m_M1 + offset_1, m_M2 + offset_1),
            32: (rotate_vector(m_M1 + offset_1, 120), rotate_vector(m_M2 + offset_1, 120)),
            33: (rotate_vector(m_M1 + offset_1, 240), rotate_vector(m_M2 + offset_1, 240)),
        }
        
        offset_dict = {
            1: offset, 2: -offset, 11: rotate_vector(offset, 120), 
            12: rotate_vector(offset, 240), 5: np.zeros(2),
            31: offset_1, 32: rotate_vector(offset_1, 120), 33: rotate_vector(offset_1, 240)
        }
        
        mK_dict = {
            1: (m_K1, m_K2), 2: (-m_K1, -m_K2),
            11: (rotate_vector(m_K1, 120), rotate_vector(m_K2, 120)),
            12: (rotate_vector(m_K1, 240), rotate_vector(m_K2, 240)),
            5: (np.zeros(2), np.zeros(2)),
            31: (m_M1, m_M2), 32: (rotate_vector(m_M1, 120), rotate_vector(m_M2, 120)),
            33: (rotate_vector(m_M1, 240), rotate_vector(m_M2, 240))
        }

        if self.valley not in mK_dict:
            raise ValueError("Invalid valley in set_const_mtrx_diff_Gn")

        K1, K2 = valley_dict[self.valley]
        m_K1, m_K2 = mK_dict[self.valley]
        offset = offset_dict[self.valley]
        self.reporter.stage("Valley centers", "Layer-resolved centers used for TAPW projection.")
        self.reporter.kv("K1", K1)
        self.reporter.kv("K2", K2)
        self.reporter.kv("m_K1", m_K1)
        self.reporter.kv("m_K2", m_K2)
        self.reporter.kv("Offset", offset)
        return K1, K2, m_K1, m_K2, offset

    def generate_g_vec_list(self):
        """Generate the g_vec_list for K1 and K2."""
        m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]
        m_g_unitvec_2 = self.structure.reciprocal_Tmat[1][:2]
        if self.structure.reciprocal_Tmat[0] @ self.structure.reciprocal_Tmat[1] < 0:
            m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        else:
            m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]

        K1, K2, m_K1, m_K2, offset = self.calculate_K_points()
        self.K1, self.K2 = K1, K2

        g_vec_list = []
        g_3 = -m_g_unitvec_1 - m_g_unitvec_2
        num_add = 2
        
        for i in range(self.n_g + num_add):
            for j in range(self.n_g + num_add):
                g_vec_list.append(i * m_g_unitvec_1 + j * m_g_unitvec_2)

        for i in range(1, self.n_g + num_add):
            for j in range(self.n_g + num_add):
                g_vec_list.append(i * g_3 + j * m_g_unitvec_1)

        for i in range(1, self.n_g + num_add):
            for j in range(1, self.n_g + num_add):
                g_vec_list.append(j * g_3 + i * m_g_unitvec_2)

        o_g_vec_list = np.array(g_vec_list)
        o_g_vec_list_m_K1 = o_g_vec_list - m_K1
        o_g_vec_list_m_K2 = o_g_vec_list - m_K2

        K1_distance = unique_sorted(np.linalg.norm(o_g_vec_list_m_K1, axis=1), 
                                   tolerance=0.1 * np.linalg.norm(m_g_unitvec_1))
        K2_distance = unique_sorted(np.linalg.norm(o_g_vec_list_m_K2, axis=1), 
                                   tolerance=0.1 * np.linalg.norm(m_g_unitvec_1))

        g_vec_list_K1 = []
        g_vec_list_K2 = []
        n_g = self.n_g - 0 if self.valley == 5 else self.n_g
        
        for i, vec in enumerate(o_g_vec_list_m_K1):
            if np.linalg.norm(vec) < K1_distance[n_g - 1] + 0.001:
                g_vec_list_K1.append(o_g_vec_list[i] + offset)
        
        for i, vec in enumerate(o_g_vec_list_m_K2):
            if np.linalg.norm(vec) < K2_distance[n_g - 1] + 0.001:
                g_vec_list_K2.append(o_g_vec_list[i] + offset)
        g_vec_list_K1 = np.array(g_vec_list_K1)
        g_vec_list_K2 = np.array(g_vec_list_K2)
        self.g_vec_list_K1 = np.array(g_vec_list_K1)
        self.g_vec_list_K2 = np.array(g_vec_list_K2)
        # self.g_vec_list_K1 = np.concatenate([g_vec_list_K1, -g_vec_list_K1])
        # self.g_vec_list_K2 = np.concatenate([g_vec_list_K2, -g_vec_list_K2])

        self.reporter.kv("G-vector count", f"K1 set={len(self.g_vec_list_K1)}, K2 set={len(self.g_vec_list_K2)}")
        self.reporter.detail(
            "K1 G-vectors",
            self.g_vec_list_K1,
            purpose="reciprocal vectors retained for the first source orientation group",
        )
        self.reporter.detail(
            "K2 G-vectors",
            self.g_vec_list_K2,
            purpose="reciprocal vectors retained for the second source orientation group",
        )

    @timing_decorator_factory(process_id=0)
    def generate_gr_matrix(self):
        """Generate the g_matrix for TAPW using twist_group assignment"""
        # Check that structure has twist_group column
        if 'twist_group' not in self.structure.df.columns:
            raise ValueError(
                "structure.df must contain 'twist_group' column. "
                "Please use StructureProcessorSpglib with the new multi-group support."
            )
        if 'phys_layer' not in self.structure.df.columns:
            raise ValueError(
                "structure.df must contain 'phys_layer' column. "
                "Please use StructureProcessorSpglib with the new multi-group support."
            )
        
        # Get unique twist groups (sorted)
        unique_groups = sorted(self.structure.df['twist_group'].unique())
        n_groups = len(unique_groups)
        
        # Validate twist_group is continuous 0..n_groups-1
        if unique_groups != list(range(n_groups)):
            raise ValueError(
                f"twist_group must be continuous 0..{n_groups-1}, got {unique_groups}"
            )
        
        # Assign g_vec_list: even groups -> K1, odd groups -> K2
        g_vec_list = []
        for group_id in range(n_groups):
            if group_id % 2 == 0:
                g_vec_list.append(self.g_vec_list_K1)
                self.reporter.kv(f"Source orientation group {group_id}", f"uses K1 G-vector set (count={len(self.g_vec_list_K1)})")
            else:
                g_vec_list.append(self.g_vec_list_K2)
                self.reporter.kv(f"Source orientation group {group_id}", f"uses K2 G-vector set (count={len(self.g_vec_list_K2)})")
        
        self.reporter.kv("Source orientation groups", n_groups)
        self.g_matrix = self.generate_gr_matrix_cpu(self.structure.df, g_vec_list, spin=self.structure.spin, reporter=self.reporter)
        if scipy.sparse.issparse(self.g_matrix):
            self.g_matrix = self.g_matrix.tocsr()
        else:
            self.g_matrix = scipy.sparse.csr_matrix(self.g_matrix)
        self.g_matrix.sort_indices()

        # Cache conjugate transpose for fast g@H@g^H multiplication.
        self.g_matrix_conj = self.g_matrix.conj().T.tocsr()
        self.g_matrix_conj.sort_indices()

        # Lightweight orthonormality check (much cheaper than det(g g^H)).
        row_norm2 = np.asarray(self.g_matrix.multiply(self.g_matrix.conj()).sum(axis=1)).ravel().real
        max_dev = float(np.max(np.abs(row_norm2 - 1.0))) if row_norm2.size else 0.0
        self.reporter.check(
            "Projection normalization check",
            passed=max_dev <= 1.0e-2,
            detail=f"max row-norm deviation {max_dev:.3e}",
        )
        if max_dev > 1.0e-2:
            raise Exception(f"g_matrix row norms deviate too much: max|norm^2-1|={max_dev:.3e}")
        
    @staticmethod
    def generate_gr_matrix_cpu(structure_df, g_vec_list, spin=None, reporter=None):
        """Generate gr_matrix (sparse) using twist_group assignment (generalized for n_groups >= 2).

        This version builds the matrix directly as sparse COO/CSR (instead of allocating a huge dense
        matrix and then converting), and avoids the extremely expensive determinant sanity checks.

        Expected/assumed invariants (same as the original implementation):
        - `atom_type` values are 0..n_types-1 and are stable identifiers.
        - All atoms of the same `atom_type` belong to the same `twist_group`.
        - All atoms of the same `atom_type` have the same `orb_num`.
        """
        required_cols = ['atom_type', 'orb_num', 'twist_group', 'shifted_x', 'shifted_y']
        for col in required_cols:
            if col not in structure_df.columns:
                raise ValueError(f"structure_df must contain '{col}' column")

        df_temp = structure_df.copy()
        # Preserve the original column-packing convention: atoms are grouped by atom_type.
        df_temp = df_temp.sort_values(['atom_type'], kind='stable').reset_index(drop=True)

        atom_type_all = df_temp['atom_type'].to_numpy(dtype=int, copy=False)
        orb_num_all = df_temp['orb_num'].to_numpy(dtype=int, copy=False)
        twist_group_all = df_temp['twist_group'].to_numpy(dtype=int, copy=False)
        pos_array = df_temp[['shifted_x', 'shifted_y']].to_numpy(dtype=float, copy=False)

        unique_groups = sorted(np.unique(twist_group_all).tolist())
        n_groups = len(unique_groups)
        if unique_groups != list(range(n_groups)):
            raise ValueError(f"twist_group must be continuous 0..{n_groups-1}, got {unique_groups}")
        if len(g_vec_list) != n_groups:
            raise ValueError(f"g_vec_list length ({len(g_vec_list)}) != n_groups ({n_groups})")

        atom_type_list = np.unique(atom_type_all)
        if not np.array_equal(atom_type_list, np.arange(atom_type_list.size)):
            raise ValueError(f"atom_type must be 0..n_types-1, got {atom_type_list}")
        n_types = int(atom_type_list.size)

        atom_orb_num_list = np.zeros(n_types, dtype=int)
        atom_num_list = np.zeros(n_types, dtype=int)
        atom_twist_group_list = np.zeros(n_types, dtype=int)
        atom_orb_name_list = np.array([str(i) for i in range(n_types)], dtype=object)
        if 'orb_name' in df_temp.columns:
            atom_orb_name_list = np.zeros(n_types, dtype=object)

        for t in range(n_types):
            mask = atom_type_all == t
            cnt = int(np.sum(mask))
            if cnt == 0:
                raise ValueError(f"atom_type {t} has no atoms")
            atom_num_list[t] = cnt

            orb0 = int(orb_num_all[mask][0])
            atom_orb_num_list[t] = orb0
            if not np.all(orb_num_all[mask] == orb0):
                raise ValueError(f"Inconsistent orb_num for atom_type {t}")

            grp0 = int(twist_group_all[mask][0])
            atom_twist_group_list[t] = grp0
            if not np.all(twist_group_all[mask] == grp0):
                raise ValueError(f"Inconsistent twist_group for atom_type {t}")

            if 'orb_name' in df_temp.columns:
                atom_orb_name_list[t] = df_temp.loc[mask, 'orb_name'].iloc[0]

        factor_list = 1.0 / np.sqrt(atom_num_list.astype(np.float64))

        reporter = ensure_reporter(reporter)
        reporter.detail(
            "TAPW sparse projector input packing",
            purpose="internal atom-type packing used to build the sparse projector",
        )
        reporter.detail("atom_type_list", atom_type_list, purpose="internal atom-type ids")
        reporter.detail("atom_orb_num_list", atom_orb_num_list, purpose="orbital count per atom type")
        reporter.detail("atom_num_list", atom_num_list, purpose="atom count per atom type")
        reporter.detail("atom_twist_group_list", atom_twist_group_list, purpose="source orientation group per atom type")
        reporter.detail("atom_orb_name_list", atom_orb_name_list, purpose="OpenMX orbital basis label per atom type")
        reporter.detail("n_groups", n_groups, purpose="number of source orientation groups")

        orb_group_num = np.zeros(n_groups, dtype=int)
        for group_id in range(n_groups):
            orb_group_num[group_id] = int(atom_orb_num_list[atom_twist_group_list == group_id].sum())
        g_group_num = np.array([len(g_vec_list[i]) for i in range(n_groups)], dtype=int)

        dim_gr_1 = int(np.dot(g_group_num, orb_group_num))
        dim_gr_2 = int(np.sum(atom_num_list * atom_orb_num_list))
        reporter.kv("Projected TAPW basis dimension", dim_gr_1)
        reporter.kv("Source orbital basis dimension", dim_gr_2)

        # shift3[group] = sum_{g'<group} g_group_num[g'] * orb_group_num[g']
        shift3 = np.zeros(n_groups, dtype=int)
        for group_id in range(n_groups):
            shift3[group_id] = int(np.dot(g_group_num[:group_id], orb_group_num[:group_id]))

        # shift1[type] = sum of orbitals of previous types in the same group
        shift1 = np.zeros(n_types, dtype=int)
        for j in range(n_types):
            group_id = atom_twist_group_list[j]
            if j > 0:
                shift1[j] = int(atom_orb_num_list[:j][atom_twist_group_list[:j] == group_id].sum())

        # Column offsets for each atom in df_temp order (grouped by atom_type).
        col_offsets = np.cumsum(np.r_[0, orb_num_all[:-1]]).astype(np.int64)

        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []

        for group_id in range(n_groups):
            gvecs = np.asarray(g_vec_list[group_id], dtype=np.float64)
            if gvecs.ndim != 2 or gvecs.shape[1] != 2:
                raise ValueError(f"g_vec_list[{group_id}] must have shape (Ng, 2), got {gvecs.shape}")

            for gi, g_vec in enumerate(gvecs):
                shift2 = int(gi * orb_group_num[group_id])
                for j in range(n_types):
                    if atom_twist_group_list[j] != group_id:
                        continue

                    orb_num = int(atom_orb_num_list[j])
                    row_base = int(shift1[j] + shift2 + shift3[group_id])
                    row_idx = (row_base + np.arange(orb_num, dtype=np.int64))[None, :]

                    atom_idx = np.nonzero((twist_group_all == group_id) & (atom_type_all == j))[0]
                    if atom_idx.size == 0:
                        continue

                    phase = np.exp(-1j * (pos_array[atom_idx] @ g_vec))
                    col_base = col_offsets[atom_idx]
                    col_idx = col_base[:, None] + np.arange(orb_num, dtype=np.int64)[None, :]

                    rows_parts.append(np.broadcast_to(row_idx, col_idx.shape).reshape(-1))
                    cols_parts.append(col_idx.reshape(-1))
                    data_parts.append(np.broadcast_to((phase * factor_list[j])[:, None], col_idx.shape).reshape(-1))

        if not rows_parts:
            raise ValueError("No entries generated for gr_matrix; check inputs.")

        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        data = np.concatenate(data_parts).astype(np.complex128, copy=False)

        gr = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(dim_gr_1, dim_gr_2), dtype=np.complex128).tocsr()
        gr.sort_indices()
        density = gr.nnz / float(dim_gr_1 * dim_gr_2) if dim_gr_1 and dim_gr_2 else 0.0
        reporter.kv("Sparse projector", f"shape={gr.shape}, nonzeros={gr.nnz}, density={density:.3e}")
        reporter.detail("factor_list", factor_list, purpose="normalization factor per atom type")

        if spin:
            gr = scipy.sparse.block_diag((gr, gr), format='csr')
            gr.sort_indices()
        return gr

    def generate_C3_matrix(self):
        """Generate the C3 transport matrix for alternating multi-group stacks.

        Representation conventions (must match generate_gr_matrix_cpu row ordering):
        - Rows are ordered by twist_group major blocks 0..G-1.
        - For each group g: basis is (G-vectors of that group) ⊗ (orbitals-of-atom-types in that group).
        - Even group -> A orientation (K1 g-set); Odd group -> B orientation (K2 g-set).

        This routine builds a block-diagonal C3 over groups (direct sum), where each group block is:
            C3_group = C3_G(group_orientation) ⊗ C3_orb(group_species)
        and optionally ⊗ C3_spin if spin is enabled.
        """
        if 'twist_group' not in self.structure.df.columns:
            raise ValueError("structure.df must contain 'twist_group' column for single-valley C3 symmetrization")

        df_temp = self.structure.df.copy()
        unique_groups = sorted(df_temp['twist_group'].unique().tolist())
        n_groups = len(unique_groups)
        if unique_groups != list(range(n_groups)):
            raise ValueError(f"twist_group must be continuous 0..{n_groups-1}, got {unique_groups}")

        # Ensure g-vectors exist
        if self.g_vec_list_K1 is None or self.g_vec_list_K2 is None:
            raise ValueError("g_vec_list_K1/K2 are not initialized. Call generate_g_vec_list() first.")

        # Build the two orientation-specific g-space C3 representations once.
        # C3_G_matrix is bilayer-oriented: it returns reps for the (K1) and (K2) g-spaces.
        C3_Gn_K1, C3_Gn_K2 = C3_G_matrix(
            self.g_vec_list_K1,
            self.g_vec_list_K2,
            self.structure.reciprocal_Tmat,
            self.structure.twist_index,
            valley=self.valley,
        )

        # Helpers for orbital representation (only needed for multi-group)
        from ..symmetry.representations import direct_sum, rot_matrix
        from ..geometry.rotations import get_any_rot_orb_twostep

        C3_rot_matrix = rot_matrix(120)
        sigma_z = np.array([[1, 0], [0, -1]])
        C3spin = scipy.linalg.expm(-1j * sigma_z / 2 * 2 * np.pi / 3)

        C3_s = get_any_rot_orb_twostep('s', C3_rot_matrix)
        C3_p = get_any_rot_orb_twostep('p', C3_rot_matrix)
        C3_d = get_any_rot_orb_twostep('d', C3_rot_matrix)
        C3_f = get_any_rot_orb_twostep('f', C3_rot_matrix)
        orbital_mapping = {'s': C3_s, 'p': C3_p, 'd': C3_d, 'f': C3_f}

        def parse_orbitals_from_orb_name(orb_name: str):
            """Parse an OpenMX orb_name like 'Mo7.0-s3p2d1' into orbital radial counts dict."""
            if not isinstance(orb_name, str):
                raise ValueError(f"orb_name must be str, got {type(orb_name)}")
            # Keep only the part after '-' (e.g. 's3p2d1')
            if '-' in orb_name:
                _, orb_part = orb_name.split('-', 1)
            else:
                orb_part = orb_name
            orbitals = {}
            for orb, count in re.findall(r'([spdf])(\d+)', orb_part):
                orbitals[orb] = int(count)
            if len(orbitals) == 0:
                raise ValueError(f"Cannot parse orbitals from orb_name='{orb_name}'")
            return orbitals

        from ..symmetry.representations import generate_direct_sum_params
        
        # Determine per-group atom-type ordering consistent with generate_gr_matrix_cpu:
        # atom_type_list is sorted unique over the whole df.
        atom_type_list_global = np.unique(df_temp['atom_type'].values)

        C3_blocks = []
        for gid in range(n_groups):
            df_g = df_temp[df_temp['twist_group'] == gid]
            if df_g.empty:
                raise ValueError(f"No atoms found for twist_group={gid}")

            # Orientation-specific g-space rep
            C3_G = C3_Gn_K1 if (gid % 2 == 0) else C3_Gn_K2

            # Build orbital rep over atom types in this group, ordered by global atom_type_list
            group_atom_types = [at for at in atom_type_list_global if (df_g['atom_type'] == at).any()]
            if len(group_atom_types) == 0:
                raise ValueError(f"Group {gid} has no atom types")

            C3_type_blocks = []
            for at in group_atom_types:
                orb_name = df_g.loc[df_g['atom_type'] == at, 'orb_name'].iloc[0]
                orbitals_dict = parse_orbitals_from_orb_name(orb_name)
                # Use original generate_direct_sum_params to match orbital ordering
                params = generate_direct_sum_params(orbitals_dict, orbital_mapping)
                C3_type_blocks.append(direct_sum(*params))

            C3_orb = direct_sum(*C3_type_blocks)

            # Combine: kron(C3_G, C3_orb)
            C3_group = np.kron(C3_G, C3_orb)
            C3_blocks.append(C3_group)
            self.reporter.kv(
                f"C3 group {gid}",
                f"orientation={'K1' if gid%2==0 else 'K2'}, C3_G={C3_G.shape}, "
                f"C3_orb={C3_orb.shape}, C3_group={C3_group.shape}",
            )

        # Direct sum across groups in gid order; this matches gr_matrix row block ordering.
        C3_all_rep = direct_sum(*C3_blocks)
        
        # Spin is applied at the very end (after direct_sum), matching original C3_MoTe2_all implementation
        if self.structure.spin:
            C3_all_rep = np.kron(C3spin, C3_all_rep)
        self.C3_matrix = scipy.sparse.csr_matrix(C3_all_rep)

        C3_matrix_2 = self.C3_matrix @ self.C3_matrix
        self.symm_matrix = [
            scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),
            self.C3_matrix,
            C3_matrix_2,
        ]
        self.symm_matrix_inv = [
            scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),
            self.C3_matrix.conj().T,
            C3_matrix_2.conj().T,
        ]

        self.reporter.kv("Combined C3 matrix", f"shape={self.C3_matrix.shape}, nnz={self.C3_matrix.nnz}")
        return self.C3_matrix
        
    def generate_C3_matrix1(self):
        """Generate the C3_matrix (only supports bilayer n_groups=2)."""
        # Check number of groups
        if 'twist_group' not in self.structure.df.columns:
            raise ValueError("structure.df must contain 'twist_group' column for single-valley C3 symmetrization")
        n_groups = len(self.structure.df['twist_group'].unique())
        if n_groups != 2:
            raise ValueError(
                f"Single-valley C3 symmetrization supports only bilayer (n_groups=2) in this release. "
                f"Current n_groups={n_groups}. Multi-group representation is outside the supported contract; "
                f"using it would be incorrect."
            )
        
        df_temp = self.structure.df.copy()
        
        atom_type_list = np.unique(df_temp['atom_type'].values)
        
        # Use twist_group instead of layer for grouping
        atom_twist_group_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['twist_group'].values)[0]
            for atom_type in atom_type_list
        ])
        # For backward compatibility, also get layer (should equal phys_layer)
        atom_layer_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['layer'].values)[0]
            for atom_type in atom_type_list
        ])
        
        atom_orb_name_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['orb_name'].values)[0]
            for atom_type in atom_type_list
        ])
        
        species_str_1layer = atom_orb_name_list[np.where(atom_layer_list == 0)[0]]
        species_str_2layer = atom_orb_name_list[np.where(atom_layer_list == 1)[0]]
        species_1layer = {}
        species_2layer = {}
        
        for idx, item in enumerate(species_str_1layer, start=1):
            # 使用'-'分割字符串，得到原子部分和轨道部分
            try:
                atom_part, orb_part = item.split('-')
            except ValueError:
                self.reporter.kv("C3 orbital parse warning", f"cannot split item: {item}")
                continue
            
            # 使用正则表达式提取原子符号（假设原子符号由字母组成）
            atom_match = re.match(r'([A-Za-z]+)', atom_part)
            if atom_match:
                atom = atom_match.group(1)
            else:
                self.reporter.kv("C3 orbital parse warning", f"cannot extract atom symbol: {atom_part}")
                continue
            
            # 使用正则表达式提取轨道类型和对应的数量
            orbitals = {}
            for orb, count in re.findall(r'([spdf])(\d+)', orb_part):
                orbitals[orb] = int(count)
            
            species_1layer[idx] = {"atom": atom, "orbitals": orbitals}
        
        for idx, item in enumerate(species_str_2layer, start=1):
            # 使用'-'分割字符串，得到原子部分和轨道部分
            try:
                atom_part, orb_part = item.split('-')
            except ValueError:
                self.reporter.kv("C3 orbital parse warning", f"cannot split item: {item}")
                continue
            
            # 使用正则表达式提取原子符号（假设原子符号由字母组成）
            atom_match = re.match(r'([A-Za-z]+)', atom_part)
            if atom_match:
                atom = atom_match.group(1)
            else:
                self.reporter.kv("C3 orbital parse warning", f"cannot extract atom symbol: {atom_part}")
                continue
            
            # 使用正则表达式提取轨道类型和对应的数量
            orbitals = {}
            for orb, count in re.findall(r'([spdf])(\d+)', orb_part):
                orbitals[orb] = int(count)
            
            species_2layer[idx] = {"atom": atom, "orbitals": orbitals}
        
        self.reporter.kv("C3 species layer 1", species_1layer)
        self.reporter.kv("C3 species layer 2", species_2layer)
        
        C3_G_rep_1layer, C3_G_rep_2layer = C3_G_matrix(
            self.g_vec_list_K1, self.g_vec_list_K2, 
            self.structure.reciprocal_Tmat, self.structure.twist_index, 
            valley=self.valley
        )
        
        C3_MoTe2_rep = C3_MoTe2_all(
            C3_G_rep_1layer, C3_G_rep_2layer,
            atoms_species_1layer=species_1layer,
            atoms_species_2layer=species_2layer,
            spin=self.structure.spin
        )
        
        self.C3_matrix = scipy.sparse.csr_matrix(C3_MoTe2_rep)
        C3_matrix_2 = self.C3_matrix @ self.C3_matrix
        np.save("C3_matrix",C3_MoTe2_rep)
        
        self.symm_matrix = [
            scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),
            self.C3_matrix, C3_matrix_2
        ]
        self.symm_matrix_inv = [
            scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),
            self.C3_matrix.conj().T, C3_matrix_2.conj().T
        ]
        return self.C3_matrix
    


    def generate_g_symm_matrix(self):
        """Generate the g_symm_matrix"""
        # Sanity check: C3 must act on the row-space of g_matrix
        if self.symm_matrix is None or self.symm_matrix[1] is None:
            raise ValueError("symm_matrix is not initialized. Call generate_C3_matrix() first.")
        if self.symm_matrix[1].shape[0] != self.g_matrix.shape[0]:
            raise ValueError(
                f"C3 matrix dimension mismatch: symm_matrix[1].shape={self.symm_matrix[1].shape} "
                f"but g_matrix.shape={self.g_matrix.shape}. "
                "This indicates inconsistent basis ordering between C3 representation and gr_matrix rows."
            )
        g_symm_matrix = self.symm_matrix[1] @ self.g_matrix
        g_symm_matrix_2 = self.symm_matrix[2] @ self.g_matrix
        g_symm_matrix_inv = self.g_matrix.conj().T @ self.symm_matrix_inv[1]
        g_symm_matrix_2_inv = self.g_matrix.conj().T @ self.symm_matrix_inv[2]
        self.g_symm_matrix = [self.g_matrix, g_symm_matrix, g_symm_matrix_2]
        self.g_symm_matrix_inv = [self.g_matrix.conj().T, g_symm_matrix_inv, g_symm_matrix_2_inv]

    def generate_all_parameters(self):
        """Generate all TAPW parameters"""
        self.generate_g_vec_list()
        self.generate_gr_matrix()
        # test1 = self.generate_C3_matrix_test().toarray()
        # test2 = self.generate_C3_matrix().toarray()
        # diff = np.sum(np.abs(test1 - test2))
        # print("diff = ", diff)
        # exit()
        if getattr(self, "use_single_valley_c3", False):
            self.generate_C3_matrix()
            self.generate_g_symm_matrix()
        self.reporter.kv("Spin-expanded TAPW basis dimension", self.g_matrix.shape[0])
        self.reporter.kv("TAPW basis construction", "complete")

class BandStructureCalculator:
    """Band structure and Chern number calculator for twisted materials"""
    
    # Valley mapping
    VALLEY_MAP = {
        1: "K1", 2: "K2", 11: "K1_120", 12: "K1_240", 5: "Gamma",
        31: "M1", 32: "M2", 33: "M3",
        3: "M", 41: "X", 42: "Y"
    }

    def __init__(self, hr_supercell, sr_supercell, structure: StructureProcessorSpglib, config: ComputeConfig, kpath_config: KPathGenerator=None, reporter=None):
        """Initialize the calculator
        
        Args:
            hr_supercell: Hamiltonian matrix in real space
            sr_supercell: Overlap matrix in real space
            structure: Processed structure information
            config: Computation configuration
            kpath_config: K-path configuration (optional, only needed for band structure calculation)
        """
        self.hr_supercell = hr_supercell
        self.sr_supercell = sr_supercell
        self.structure = structure
        self.config = config
        self.kpath_config = kpath_config
        self.reporter = ensure_reporter(reporter)
        
        self.result = {}
        self._progress_dir: str | None = None
        # Validate valley configuration
        if not hasattr(self.config, 'valley') or self.config.valley not in self.VALLEY_MAP:
            raise ValueError(f"Invalid valley configuration. Must be one of {list(self.VALLEY_MAP.keys())}")

        self.valley_flag = self.VALLEY_MAP[self.config.valley]
        self.requested_symmetry_operations = requested_hamiltonian_symmetry_operations(self.config)
        _validate_hamiltonian_symmetrization_backend(
            self.config,
            self.valley_flag,
            self.requested_symmetry_operations,
        )
        self.uses_hamiltonian_symmetrization = bool(self.requested_symmetry_operations)
        self.use_single_valley_c3 = "C3z" in self.requested_symmetry_operations
        self.use_M_valley_threefold_symm = uses_m_valley_threefold_symmetrization(self.config)
        self.use_M_valley_d3_symm = uses_m_valley_d3_symmetrization(self.config)
        self._m_valley_reference = 31
        self._m_valley_parameters: dict[int, TAPW_parameters] = {}
        self._m_valley_transport_ref_to_valley: dict[int, scipy.sparse.csr_matrix] = {}
        self._m_valley_transport_valley_to_ref: dict[int, scipy.sparse.csr_matrix] = {}
        self._m_valley_c3_reference_projectors: list[SimpleNamespace] = []
        self._m_valley_d3_reference_projectors: list[SimpleNamespace] = []
        self._m_valley_c2_reference_symmetry: SimpleNamespace | None = None
        self._m_valley_c2_reference_transport: scipy.sparse.csr_matrix | None = None
        self._m_valley_c2_reference_linear_map: np.ndarray | None = None
        
        # Cache for (orbital-expanded) Wannier coordinates used by Getk_super_gauge_sparse.
        # This avoids rebuilding `sorted_wann` for every rvec loop and every k-point.
        self._sorted_wann: np.ndarray | None = None
        self._num_wann: int | None = None
        self._ef_onsite_orb: np.ndarray | None = None
        self._realspace_block_cache: dict[int, SimpleNamespace] = {}
        
        if self.config.TAPW:
            if self.use_M_valley_threefold_symm:
                self.use_single_valley_c3 = False
                self._initialize_m_valley_threefold_symmetrization()
            else:
                self.TAPW_parameters = TAPW_parameters(self.structure, self.config, reporter=self.reporter)
                self.use_single_valley_c3 = self.TAPW_parameters.use_single_valley_c3
                self.uses_hamiltonian_symmetrization = self.use_single_valley_c3
                if "C3z" in self.requested_symmetry_operations and not self.use_single_valley_c3:
                    self.reporter.kv(
                        "symmetrize_hamiltonian",
                        self.TAPW_parameters.c3_h_disable_reason
                        + f" Proceeding with C3z disabled for valley {self.config.valley}.",
                    )
                self.TAPW_parameters.generate_all_parameters()
    def _set_progress_stage(self, index: int, stage: str) -> None:
        _write_progress_state(self._progress_dir, index, stage)

    def _refresh_ef_onsite_orb_cache(self) -> None:
        if self._sorted_wann is None or self._num_wann is None:
            return

        ef = getattr(getattr(self, "TAPW_parameters", None), "electric_field_onsite", None)
        if ef is None:
            self._ef_onsite_orb = None
            return

        orb_num = self.structure.df['orb_num'].to_numpy(dtype=int, copy=False)
        ef_orb = np.repeat(np.asarray(ef, dtype=np.float64), orb_num)
        if self.structure.spin:
            ef_orb = np.tile(ef_orb, 2)
        self._ef_onsite_orb = ef_orb

    def _ensure_sorted_wann_cache(self) -> None:
        if self._sorted_wann is not None and self._num_wann is not None:
            return
        df = self.structure.df
        coords = df[['x', 'y', 'z']].to_numpy(dtype=np.float64, copy=False)
        orb_num = df['orb_num'].to_numpy(dtype=int, copy=False)
        sorted_wann = np.repeat(coords, orb_num, axis=0)
        if self.structure.spin:
            sorted_wann = np.concatenate([sorted_wann, sorted_wann], axis=0)

        self._sorted_wann = sorted_wann
        self._num_wann = int(sorted_wann.shape[0])
        self._refresh_ef_onsite_orb_cache()

    def _get_or_build_realspace_block_cache(self, hr_blocks) -> SimpleNamespace:
        cache_key = id(hr_blocks)
        cached = self._realspace_block_cache.get(cache_key)
        if cached is not None:
            return cached

        rows_template_parts = []
        cols_template_parts = []
        blocks = []
        offset = 0
        for rvec, values_dic in hr_blocks.items():
            row_index = np.asarray(values_dic["row"], dtype=np.int64)
            col_index = np.asarray(values_dic["col"], dtype=np.int64)
            values = np.asarray(values_dic["val"], dtype=np.complex128)
            block_nnz = int(values.shape[0])
            rows_template_parts.append(row_index)
            cols_template_parts.append(col_index)
            blocks.append(
                SimpleNamespace(
                    row_index=row_index,
                    col_index=col_index,
                    values=values,
                    rvec_cart=np.dot(rvec, self.structure.Tmat),
                    data_slice=slice(offset, offset + block_nnz),
                )
            )
            offset += block_nnz

        rows_template = (
            np.concatenate(rows_template_parts, dtype=np.int64)
            if rows_template_parts
            else np.empty(0, dtype=np.int64)
        )
        cols_template = (
            np.concatenate(cols_template_parts, dtype=np.int64)
            if cols_template_parts
            else np.empty(0, dtype=np.int64)
        )
        sort_order = (
            np.lexsort((cols_template, rows_template)).astype(np.int64, copy=False)
            if offset
            else np.empty(0, dtype=np.int64)
        )
        rows_sorted = rows_template[sort_order] if offset else rows_template
        cols_sorted = cols_template[sort_order] if offset else cols_template
        if offset:
            unique_mask = np.empty(int(offset), dtype=bool)
            unique_mask[0] = True
            unique_mask[1:] = (
                (rows_sorted[1:] != rows_sorted[:-1])
                | (cols_sorted[1:] != cols_sorted[:-1])
            )
            compressed_positions_sorted = np.cumsum(unique_mask, dtype=np.int64) - 1
            compressed_positions = np.empty(int(offset), dtype=np.int64)
            compressed_positions[sort_order] = compressed_positions_sorted
            csr_unique_rows = rows_sorted[unique_mask]
            csr_indices = cols_sorted[unique_mask]
            csr_nnz = int(csr_indices.shape[0])
            has_duplicates = bool(csr_nnz != int(offset))
        else:
            unique_mask = np.empty(0, dtype=bool)
            compressed_positions = np.empty(0, dtype=np.int64)
            csr_unique_rows = np.empty(0, dtype=np.int64)
            csr_indices = np.empty(0, dtype=np.int64)
            csr_nnz = 0
            has_duplicates = False
        cached = SimpleNamespace(
            nnz_total=int(offset),
            rows_template=rows_template,
            cols_template=cols_template,
            blocks=blocks,
            sort_order=sort_order,
            compressed_positions=compressed_positions,
            csr_unique_rows=csr_unique_rows,
            csr_indices=csr_indices,
            csr_nnz=csr_nnz,
            has_duplicates=has_duplicates,
            csr_num_wann=None,
            csr_indptr=None,
        )
        self._realspace_block_cache[cache_key] = cached
        return cached

    def _compress_raw_realspace_data_to_csr(self, block_cache, raw_data, num_wann: int):
        raw_data = np.asarray(raw_data, dtype=np.complex128)
        if raw_data.shape[0] != block_cache.nnz_total:
            raise ValueError(
                f"raw_data length {raw_data.shape[0]} does not match cached nnz_total {block_cache.nnz_total}"
            )

        if block_cache.csr_num_wann != int(num_wann) or block_cache.csr_indptr is None:
            counts = np.bincount(
                block_cache.csr_unique_rows,
                minlength=int(num_wann),
            )
            indptr = np.empty(int(num_wann) + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            block_cache.csr_indptr = indptr
            block_cache.csr_num_wann = int(num_wann)

        if block_cache.has_duplicates:
            data = (
                np.bincount(
                    block_cache.compressed_positions,
                    weights=raw_data.real,
                    minlength=block_cache.csr_nnz,
                )
                + 1j
                * np.bincount(
                    block_cache.compressed_positions,
                    weights=raw_data.imag,
                    minlength=block_cache.csr_nnz,
                )
            )
        else:
            data = raw_data[block_cache.sort_order]

        return scipy.sparse.csr_matrix(
            (data, block_cache.csr_indices, block_cache.csr_indptr),
            shape=(int(num_wann), int(num_wann)),
            dtype=np.complex128,
        )

    def _build_getk_phase_context(self, k) -> SimpleNamespace:
        kvec = self.get_kvec(k)
        self._ensure_sorted_wann_cache()
        sorted_wann = self._sorted_wann
        num_wann = self._num_wann
        if sorted_wann is None or num_wann is None:
            raise RuntimeError("sorted_wann cache is not initialized")

        phase_wann = (
            sorted_wann[:, 0] * kvec[0]
            + sorted_wann[:, 1] * kvec[1]
            + sorted_wann[:, 2] * kvec[2]
        )
        return SimpleNamespace(
            kvec=np.asarray(kvec, dtype=float),
            sorted_wann=sorted_wann,
            num_wann=int(num_wann),
            phase_wann=phase_wann,
        )

    def _assemble_sparse_realspace_matrix(self, hr_blocks, phase_ctx, type="H"):
        sorted_wann = phase_ctx.sorted_wann
        num_wann = phase_ctx.num_wann
        kvec = phase_ctx.kvec
        phase_wann = phase_ctx.phase_wann

        use_fast = bool(getattr(self.config, "fast_getk", True))
        if use_fast:
            block_cache = self._get_or_build_realspace_block_cache(hr_blocks)
            data = np.empty(block_cache.nnz_total, dtype=np.complex128)
            for block in block_cache.blocks:
                exp_kR = np.exp(1j * np.dot(kvec, block.rvec_cart))
                dot_mn = phase_wann[block.row_index] - phase_wann[block.col_index]
                data[block.data_slice] = block.values * np.exp(-1j * dot_mn) * exp_kR
            mk = self._compress_raw_realspace_data_to_csr(block_cache, data, num_wann)
        else:
            mk = scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)
            for rvec, values_dic in hr_blocks.items():
                row_index = np.asarray(values_dic["row"], dtype=np.int64)
                col_index = np.asarray(values_dic["col"], dtype=np.int64)
                val_index = np.asarray(values_dic["val"], dtype=np.complex128)
                m_coor = sorted_wann[row_index]
                n_coor = sorted_wann[col_index]
                rvec_cart = np.dot(rvec, self.structure.Tmat)
                phase_factor = np.exp(-1j * np.dot(m_coor - n_coor, kvec)) * np.exp(1j * np.dot(kvec, rvec_cart))
                mk += scipy.sparse.csr_matrix(
                    (val_index * phase_factor, (row_index, col_index)),
                    shape=(num_wann, num_wann),
                )

        if type == "H" and self._ef_onsite_orb is not None:
            mk = mk + scipy.sparse.diags(self._ef_onsite_orb, 0, shape=(num_wann, num_wann), dtype=np.float64)
        return mk

    def _clone_compute_config_for_valley(self, valley: int) -> ComputeConfig:
        cfg = copy.deepcopy(self.config)
        cfg.valley = valley
        cfg.valleys = [valley]
        return cfg

    def _initialize_m_valley_threefold_symmetrization(self) -> None:
        self.reporter.kv(
            "M-valley C3 symmetrization",
            f"enabled for requested valley {self.config.valley} with reference valley {self._m_valley_reference}",
        )
        for valley in _M_VALLEY_TRIPLET:
            cfg = self._clone_compute_config_for_valley(valley)
            params = TAPW_parameters(self.structure, cfg, reporter=self.reporter)
            params.generate_all_parameters()
            self._m_valley_parameters[valley] = params

        self.TAPW_parameters = self._m_valley_parameters[self.config.valley]

        identity = scipy.sparse.identity(
            self._m_valley_parameters[self._m_valley_reference].g_matrix.shape[0],
            dtype=np.complex128,
            format="csr",
        )
        self._m_valley_transport_ref_to_valley[self._m_valley_reference] = identity
        self._m_valley_transport_valley_to_ref[self._m_valley_reference] = identity

        for valley in _M_VALLEY_TRIPLET:
            if valley == self._m_valley_reference:
                continue
            transport = build_projected_m_valley_transport(
                self.structure,
                self._m_valley_parameters[self._m_valley_reference],
                self._m_valley_parameters[valley],
            )
            self._m_valley_transport_ref_to_valley[valley] = transport
            self._m_valley_transport_valley_to_ref[valley] = transport.conj().T.tocsr()

        self._m_valley_c3_reference_projectors = self._build_m_valley_c3_reference_projectors()

        if self.use_M_valley_d3_symm:
            self.reporter.kv(
                "M-valley D3 symmetrization",
                f"enabled additional reference-valley C2 projection on M{self._m_valley_reference - 30}",
            )
            self._m_valley_c2_reference_symmetry = resolve_reference_m_valley_c2_symmetry(
                self.structure,
                self._m_valley_parameters[self._m_valley_reference],
            )
            self._m_valley_c2_reference_linear_map = None
            self._m_valley_c2_reference_transport = None
            self._m_valley_d3_reference_projectors = self._build_m_valley_d3_reference_projectors()

    def switch_m_valley(self, valley: int) -> None:
        if not getattr(self, "use_M_valley_threefold_symm", False):
            raise ValueError("switch_m_valley is only valid for M-valley threefold-symmetrized calculators.")
        if valley not in self._m_valley_parameters:
            raise ValueError(f"M-valley {valley} is not initialized in this calculator.")

        if hasattr(self.config, "set_valley"):
            self.config.set_valley(valley)
        else:
            self.config.valley = valley
            self.config.valley_flag = self.VALLEY_MAP[valley]
        self.valley_flag = self.VALLEY_MAP[valley]
        self.TAPW_parameters = self._m_valley_parameters[valley]
        self.result = {}
        self._progress_dir = None
        self._refresh_ef_onsite_orb_cache()

    def _normalize_projector_matrix(self, projector):
        if scipy.sparse.issparse(projector):
            projector = projector.tocsr()
            projector.sort_indices()
            return projector
        return np.asarray(projector, dtype=np.complex128)

    def _make_cached_projector_term(
        self,
        label: str,
        linear_map_2d: np.ndarray,
        projector,
        valley: int | None = None,
        tapw_parameters=None,
        transport_to_ref=None,
    ) -> SimpleNamespace:
        projector = self._normalize_projector_matrix(projector)
        if scipy.sparse.issparse(projector):
            projector_h = projector.conj().T.tocsr()
        else:
            projector_h = np.asarray(projector, dtype=np.complex128).conj().T
        return SimpleNamespace(
            label=label,
            linear_map_2d=np.asarray(linear_map_2d, dtype=float),
            projector=projector,
            projector_h=projector_h,
            valley=valley,
            tapw_parameters=tapw_parameters,
            transport_to_ref=transport_to_ref,
        )

    def _build_m_valley_c3_reference_projectors(self) -> list[SimpleNamespace]:
        reference_params = self._m_valley_parameters[self._m_valley_reference]
        projectors: list[SimpleNamespace] = []

        def _append_projector(
            label: str,
            linear_map_2d: np.ndarray,
            projector,
            valley: int,
            tapw_parameters,
            transport_to_ref,
        ) -> None:
            projectors.append(
                self._make_cached_projector_term(
                    label,
                    linear_map_2d,
                    projector,
                    valley=valley,
                    tapw_parameters=tapw_parameters,
                    transport_to_ref=transport_to_ref,
                )
            )

        _append_projector(
            "identity",
            np.eye(2, dtype=float),
            reference_params.g_matrix,
            valley=self._m_valley_reference,
            tapw_parameters=reference_params,
            transport_to_ref=self._m_valley_transport_valley_to_ref[self._m_valley_reference],
        )

        for valley in (32, 33):
            angle_deg = _m_valley_rotation_delta(self._m_valley_reference, valley)
            rotation = np.asarray(rot_matrix(angle_deg), dtype=float)
            projector = self._m_valley_transport_valley_to_ref[valley] @ self._m_valley_parameters[valley].g_matrix
            _append_projector(
                f"c3_valley_{valley}",
                rotation[:2, :2],
                projector,
                valley=valley,
                tapw_parameters=self._m_valley_parameters[valley],
                transport_to_ref=self._m_valley_transport_valley_to_ref[valley],
            )

        return projectors

    def _build_m_valley_d3_reference_projectors(self) -> list[SimpleNamespace]:
        if self._m_valley_c2_reference_symmetry is None:
            raise ValueError("Reference M-valley C2 symmetry must be resolved before building D3 projectors.")

        reference_params = self._m_valley_parameters[self._m_valley_reference]
        resolved_symmetry = self._m_valley_c2_reference_symmetry
        projectors: list[SimpleNamespace] = []

        def _append_projector(label: str, linear_map_2d: np.ndarray, projector) -> None:
            projectors.append(self._make_cached_projector_term(label, linear_map_2d, projector))

        _append_projector("identity", np.eye(2, dtype=float), reference_params.g_matrix)

        for valley in (32, 33):
            angle_deg = _m_valley_rotation_delta(self._m_valley_reference, valley)
            rotation = np.asarray(rot_matrix(angle_deg), dtype=float)
            projector = self._m_valley_transport_valley_to_ref[valley] @ self._m_valley_parameters[valley].g_matrix
            _append_projector(f"c3_valley_{valley}", rotation[:2, :2], projector)

        c2_rotation = np.asarray(resolved_symmetry.rotation_cart, dtype=float)
        c2_translation = np.asarray(resolved_symmetry.translation_cart, dtype=float)
        group_target_maps = {
            0: resolved_symmetry.atom_type_map_0_to_1,
            1: resolved_symmetry.atom_type_map_1_to_0,
        }

        for label, angle_deg in (
            ("c2", 0),
            ("c3_c2", 120),
            ("c3_sq_c2", 240),
        ):
            if angle_deg == 0:
                rotation = c2_rotation
                translation = c2_translation
            else:
                c3_rotation = np.asarray(rot_matrix(angle_deg), dtype=float)
                rotation = c3_rotation @ c2_rotation
                translation = c3_rotation @ c2_translation
            projector = _build_reference_m_valley_transformed_projector(
                structure=self.structure,
                reference_params=reference_params,
                rotation_matrix=rotation,
                group_target_maps=group_target_maps,
                translation_cart=translation,
            )
            _append_projector(label, rotation[:2, :2], projector)

        return projectors

    def generate_kmesh(self, num_k1, num_k2=None):
        """Generate a uniform fractional kappa-grid for Chern calculations.

        The flattened order is always the row-major order of an `indexing='ij'`
        mesh, matching `chern_post.reshape_wavefunction_grid`.
        """
        if num_k2 is None:
            num_k2 = num_k1

        kappa1 = np.linspace(-0.5, 0.5, int(num_k1), endpoint=True)
        kappa2 = np.linspace(-0.5, 0.5, int(num_k2), endpoint=True)
        kappa1_mesh, kappa2_mesh = np.meshgrid(kappa1, kappa2, indexing='ij')
        kpoints = np.stack(
            (
                kappa1_mesh,
                kappa2_mesh,
                np.zeros_like(kappa1_mesh),
            ),
            axis=-1,
        ).reshape(-1, 3)
        return kpoints

    def find_first_above_energy(self, energies, E):
        """Find first band index above energy E for each k-point"""
        idxs = []
        for band in energies:  # band: (nb,)
            idx = np.where(band > E)[0]
            idxs.append(idx[0] if len(idx) > 0 else len(band))
        return np.asarray(idxs, dtype=int)

    def align_bands_by_index(self, energies, E):
        """Align bands by first index above energy E"""
        first_indices = self.find_first_above_energy(energies, E)
        min_index = int(np.min(first_indices))
        max_index = int(np.max(first_indices))
        n_all = energies.shape[1]
        n_keep = n_all - (max_index - min_index)

        if n_keep <= 0:
            raise ValueError(f"No bands to keep after alignment (n_keep={n_keep}). Check energy E={E}")

        filtered = np.empty((energies.shape[0], n_keep), dtype=energies.dtype)
        for i in range(energies.shape[0]):
            start = first_indices[i] - min_index
            end = start + n_keep
            filtered[i] = energies[i, start:end]

        pivot_col = min_index
        return filtered, pivot_col

    def split_vbm_cbm(self, energies, E):
        """Split bands into VBM and CBM parts based on energy E"""
        # Sort each k-point's energies
        energies = np.sort(energies, axis=1)
        
        filtered, pivot_col = self.align_bands_by_index(energies, E)
        vbm = filtered[:, :pivot_col] if pivot_col > 0 else np.array([]).reshape(energies.shape[0], 0)
        cbm = filtered[:, pivot_col:] if pivot_col < filtered.shape[1] else np.array([]).reshape(energies.shape[0], 0)
        
        return filtered, vbm, cbm, pivot_col
    
    def split_vbm_cbm_with_vec(self, energies, vecs, E):
        """Split bands and corresponding eigenvectors into VBM and CBM parts"""
        # energies: (n_kpoints, n_bands)
        # vecs: (n_kpoints, n_orbitals, n_bands) or list of (n_orbitals, n_bands)
        
        # Sort each k-point's energies and get sorting indices
        sort_indices = np.argsort(energies, axis=1)
        energies_sorted = np.sort(energies, axis=1)
        
        filtered_energies, pivot_col = self.align_bands_by_index(energies_sorted, E)
        
        # Split energies
        vbm_energies = filtered_energies[:, :pivot_col] if pivot_col > 0 else np.array([]).reshape(energies.shape[0], 0)
        cbm_energies = filtered_energies[:, pivot_col:] if pivot_col < filtered_energies.shape[1] else np.array([]).reshape(energies.shape[0], 0)
        
        # Split eigenvectors if provided
        vbm_vecs = None
        cbm_vecs = None
        
        if vecs is not None and len(vecs) > 0:
            # Handle case where vecs is a list or array
            if isinstance(vecs, list):
                vecs_array = np.array(vecs)  # (n_kpoints, n_orbitals, n_bands)
            else:
                vecs_array = vecs
            
            # Get the alignment indices for each k-point
            first_indices = self.find_first_above_energy(energies_sorted, E)
            min_index = int(np.min(first_indices))
            max_index = int(np.max(first_indices))
            n_keep = energies.shape[1] - (max_index - min_index)
            
            if n_keep > 0:
                # Sort eigenvectors according to energy sorting
                vecs_sorted = np.zeros_like(vecs_array)
                for k in range(vecs_array.shape[0]):
                    vecs_sorted[k] = vecs_array[k][:, sort_indices[k]]
                
                # Align eigenvectors
                vecs_filtered = np.zeros((vecs_array.shape[0], vecs_array.shape[1], n_keep), dtype=vecs_array.dtype)
                for k in range(vecs_array.shape[0]):
                    start = first_indices[k] - min_index
                    end = start + n_keep
                    vecs_filtered[k] = vecs_sorted[k][:, start:end]
                
                # Split eigenvectors
                if pivot_col > 0:
                    vbm_vecs = vecs_filtered[:, :, :pivot_col]
                if pivot_col < vecs_filtered.shape[2]:
                    cbm_vecs = vecs_filtered[:, :, pivot_col:]
        
        return filtered_energies, vbm_energies, cbm_energies, pivot_col, vbm_vecs, cbm_vecs

    def save_band_static_metadata(self, path):
        metadata_path = os.path.join(path, "band") if _is_canonical_output_layout(self.config) else path
        os.makedirs(metadata_path, exist_ok=True)
        if self.config.TAPW:
            np.save(
                os.path.join(
                    metadata_path,
                    canonical_band_filename("g_vectors", 1)
                    if _is_canonical_output_layout(self.config)
                    else gvec_output_filename(n_g=self.config.n_g, valley_flag=self.valley_flag, layer=1),
                ),
                self.TAPW_parameters.g_vec_list_K1,
            )
            np.save(
                os.path.join(
                    metadata_path,
                    canonical_band_filename("g_vectors", 2)
                    if _is_canonical_output_layout(self.config)
                    else gvec_output_filename(n_g=self.config.n_g, valley_flag=self.valley_flag, layer=2),
                ),
                self.TAPW_parameters.g_vec_list_K2,
            )
        if getattr(self, "use_single_valley_c3", False):
            np.save(
                os.path.join(metadata_path, f"C3_matrix_{self.valley_flag}"),
                self.TAPW_parameters.C3_matrix.toarray(),
            )

    def calculate_band_structure(self, path, kpoints=None):
        """Calculate band structure
        
        Args:
            path: Output path for results
            kpoints: Optional k-points array. If None, uses k-path for band structure.
                    For Chern number calculation, should provide mesh k-points.
        """
        # Create output directories
        os.makedirs(path, exist_ok=True)
        # os.makedirs(os.path.join(path, "band_wave"), exist_ok=True)
        
        # Use provided k-points or generate from k-path
        if kpoints is None:
            if self.kpath_config is None:
                raise ValueError("Must provide either kpoints or kpath_config")
            kpoints = self.kpath_config.kpoints

        # Suffix used in filenames (also reused by memmap outputs)
        mode = getattr(self.config, "mode", None)
        suffix = "_2d" if mode == "chern" else ""
        if mode == "chern" and hasattr(self.config, "get_chern_grid_suffix"):
            suffix = self.config.get_chern_grid_suffix()

        # Optional k-point chunking for job arrays / multi-node runs.
        kpoints_all = kpoints
        nk_total = int(len(kpoints_all))
        chunk_id = int(getattr(self.config, "kpoint_chunk_id", 0))
        chunk_count = int(getattr(self.config, "kpoint_chunk_count", 1))
        if chunk_count < 1:
            raise ValueError("kpoint_chunk_count must be >= 1")
        if not (0 <= chunk_id < chunk_count):
            raise ValueError(f"kpoint_chunk_id must be in [0, {chunk_count - 1}], got {chunk_id}")

        if chunk_count > 1:
            start = (nk_total * chunk_id) // chunk_count
            end = (nk_total * (chunk_id + 1)) // chunk_count
            kpoints = kpoints_all[start:end]
            kpoint_indices = list(range(start, end))
            self.reporter.kv(
                "K-point chunk",
                f"chunk {chunk_id}/{chunk_count}: indices [{start}:{end}) ({len(kpoints)}/{nk_total})",
            )
        else:
            kpoint_indices = list(range(nk_total))
        
        self.save_band_static_metadata(path)
        # Calculate bands
        self.parallel_calculate_band_01(
            kpoints,
            kpoint_indices=kpoint_indices,
            nk_total=nk_total,
            out_dir=path,
            suffix=suffix,
        )

        # MPI COMM_WORLD SLEPc mode: all ranks participate in EPSSolve collectively, but only rank 0
        # should write outputs to avoid file clobbering.
        if (
            getattr(self.config, "eigensolver", "scipy") == "slepc"
            and str(getattr(self.config, "slepc_comm", "self")).lower() == "world"
        ):
            try:
                from mpi4py import MPI  # type: ignore
            except Exception:
                MPI = None

            if MPI is not None:
                comm = MPI.COMM_WORLD
                if comm.Get_size() > 1:
                    comm.Barrier()
                    if comm.Get_rank() != 0:
                        return

        # In chunked mode we only compute/write raw eig/vec (memmap) slices. Post-process after all chunks finish.
        if chunk_count > 1:
            kpoints_path = os.path.join(path, f"kpoints{suffix}.npy")
            _ensure_memmap_file(kpoints_path, np.float64, np.asarray(kpoints_all).shape)
            mm_k = _get_memmap(kpoints_path)
            mm_k[:] = np.asarray(kpoints_all)
            mm_k.flush()
            self.reporter.kv(
                "K-point chunk output",
                f"raw memmap slices written; run postprocess after all {chunk_count} chunks complete",
            )
            return
        
        # Save results
        band_data = self.result['eig']
        # 在chern模式下，添加num_chern标识
        canonical_layout = _is_canonical_output_layout(self.config)
        if mode == "chern":
            if canonical_layout:
                num_k1, num_k2 = self.config.get_chern_grid_shape()
                out_path = os.path.join(path, "topology", canonical_topology_grid_name(num_k1, num_k2))
            else:
                os.makedirs(os.path.join(path, "topo"), exist_ok=True)
                out_path = os.path.join(path, "topo")
        else:
            out_path = os.path.join(path, "band")
        os.makedirs(out_path, exist_ok=True)
        if canonical_layout:
            np.save(os.path.join(out_path, canonical_band_filename("kpoints", None)), np.asarray(kpoints_all))
        
        # Split bands and eigenvectors by fermi energy
        band_type = str(getattr(self.config, "band_type", "")).upper()
        want_vbm = band_type not in {"CBM"}
        want_cbm = band_type not in {"VBM"}

        vec_data = self.result['vec'] if self.config.eig_vec_cal else None
        if getattr(self.config, "vec_store", "memory") == "memmap" and vec_data is not None:
            # Streamed postprocess: avoid materializing (nk, dim, nb) in RAM.
            energies = np.asarray(band_data)
            first_indices = self.find_first_above_energy(energies, self.config.efermi)
            min_index = int(np.min(first_indices))
            max_index = int(np.max(first_indices))
            n_all = int(energies.shape[1])
            n_keep = int(n_all - (max_index - min_index))
            if n_keep <= 0:
                raise ValueError(f"No bands to keep after alignment (n_keep={n_keep}). Check efermi={self.config.efermi}")

            pivot_col = min_index
            filtered = np.empty((energies.shape[0], n_keep), dtype=energies.dtype)
            for k in range(energies.shape[0]):
                start = int(first_indices[k] - min_index)
                end = start + n_keep
                filtered[k] = energies[k, start:end]

            vbm = filtered[:, :pivot_col] if pivot_col > 0 else np.array([]).reshape(filtered.shape[0], 0)
            cbm = filtered[:, pivot_col:] if pivot_col < filtered.shape[1] else np.array([]).reshape(filtered.shape[0], 0)

            # Prepare output memmaps for requested band_type(s)
            vbm_mm = None
            cbm_mm = None
            if want_vbm and vbm.shape[1] > 0:
                vbm_path = os.path.join(
                    out_path,
                    canonical_band_filename("wavefunctions", "vbm")
                    if canonical_layout
                    else array_output_filename("vec_VBM", valley_flag=self.valley_flag, suffix=suffix, tapw=self.config.TAPW),
                )
                _ensure_memmap_file(vbm_path, np.complex128, (energies.shape[0], vec_data.shape[1], vbm.shape[1]))
                vbm_mm = open_memmap(vbm_path, mode="r+")
            if want_cbm and cbm.shape[1] > 0:
                cbm_path = os.path.join(
                    out_path,
                    canonical_band_filename("wavefunctions", "cbm")
                    if canonical_layout
                    else array_output_filename("vec_CBM", valley_flag=self.valley_flag, suffix=suffix, tapw=self.config.TAPW),
                )
                _ensure_memmap_file(cbm_path, np.complex128, (energies.shape[0], vec_data.shape[1], cbm.shape[1]))
                cbm_mm = open_memmap(cbm_path, mode="r+")

            for k in tqdm(range(energies.shape[0]), desc="Postprocess vec", total=energies.shape[0]):
                start = int(first_indices[k] - min_index)
                end = start + n_keep
                if vbm_mm is not None and cbm_mm is not None:
                    vec_keep = vec_data[k, :, start:end]
                    vbm_mm[k] = vec_keep[:, :pivot_col]
                    cbm_mm[k] = vec_keep[:, pivot_col:]
                elif vbm_mm is not None:
                    vbm_mm[k] = vec_data[k, :, start : start + pivot_col]
                elif cbm_mm is not None:
                    cbm_mm[k] = vec_data[k, :, start + pivot_col : end]

            if vbm_mm is not None:
                vbm_mm.flush()
            if cbm_mm is not None:
                cbm_mm.flush()

            vbm_vecs = None
            cbm_vecs = None
        else:
            filtered, vbm, cbm, pivot_col, vbm_vecs, cbm_vecs = self.split_vbm_cbm_with_vec(
                band_data, vec_data, self.config.efermi)
        
        # Save VBM data if exists
        if want_vbm and vbm.size > 0:
            np.savetxt(
                os.path.join(
                    out_path,
                    canonical_band_filename("energies", "vbm")
                    if canonical_layout
                    else band_output_filename("VBM", valley_flag=self.valley_flag, suffix=suffix, tapw=self.config.TAPW),
                ),
                vbm,
                fmt='%15.11f',
            )
        
        # Save CBM data if exists  
        if want_cbm and cbm.size > 0:
            np.savetxt(
                os.path.join(
                    out_path,
                    canonical_band_filename("energies", "cbm")
                    if canonical_layout
                    else band_output_filename("CBM", valley_flag=self.valley_flag, suffix=suffix, tapw=self.config.TAPW),
                ),
                cbm,
                fmt='%15.11f',
            )
        
        # Save eigenvectors if calculated
        if self.config.eig_vec_cal:
            if want_vbm and vbm_vecs is not None and vbm_vecs.size > 0:
                np.save(
                    os.path.join(
                        out_path,
                        canonical_band_filename("wavefunctions", "vbm")
                        if canonical_layout
                        else array_output_filename("vec_VBM", valley_flag=self.valley_flag, suffix=suffix, tapw=self.config.TAPW),
                    ),
                    vbm_vecs,
                )
            if want_cbm and cbm_vecs is not None and cbm_vecs.size > 0:
                np.save(
                    os.path.join(
                        out_path,
                        canonical_band_filename("wavefunctions", "cbm")
                        if canonical_layout
                        else array_output_filename("vec_CBM", valley_flag=self.valley_flag, suffix=suffix, tapw=self.config.TAPW),
                    ),
                    cbm_vecs,
                )
        
        # Save Hamiltonian if requested
        if self.config.hamk_save:
            np.save(
                os.path.join(
                    out_path,
                    canonical_band_filename("hamiltonian", None)
                    if canonical_layout
                    else array_output_filename("hamk", valley_flag=self.valley_flag, suffix=suffix, tapw=self.config.TAPW),
                ),
                self.result['hamk'],
            )
        if canonical_layout and mode != "chern":
            files = {
                "kpoints": canonical_band_filename("kpoints", None),
                "g_vectors_group1": canonical_band_filename("g_vectors", 1),
                "g_vectors_group2": canonical_band_filename("g_vectors", 2),
            }
            if want_vbm and vbm.size > 0:
                files["energies_vbm"] = canonical_band_filename("energies", "vbm")
                if self.config.eig_vec_cal:
                    files["wavefunctions_vbm"] = canonical_band_filename("wavefunctions", "vbm")
            if want_cbm and cbm.size > 0:
                files["energies_cbm"] = canonical_band_filename("energies", "cbm")
                if self.config.eig_vec_cal:
                    files["wavefunctions_cbm"] = canonical_band_filename("wavefunctions", "cbm")
            if self.config.hamk_save:
                files["hamiltonian_k"] = canonical_band_filename("hamiltonian", None)
            _write_json(
                Path(out_path) / "manifest.json",
                {
                    "schema": "tapw_band_outputs/v1",
                    "valley": self.valley_flag,
                    "q_shell": Path(path).name,
                    "files": files,
                    "group_mapping": {
                        "g_vectors_group1": {"source_group": 1},
                        "g_vectors_group2": {"source_group": 2},
                    },
                },
            )
            _update_canonical_manifests(Path(out_path), "band")

    def calculate_chern(self, path):
        """Calculate Chern number using uniform k-point mesh
        
        Args:
            path: Output path for results
        """
        if not getattr(self.config, "eig_vec_cal", False):
            raise ValueError("tapw chern requires compute.eig_vec_cal=true so Berry flux can be computed.")
        if bool(getattr(self.config, "ge", False)):
            raise ValueError(
                "tapw chern does not support ge=true in this release because generalized eigenvectors "
                "require overlap-metric Berry phases, not Euclidean overlaps."
            )
        if int(getattr(self.config, "kpoint_chunk_count", 1)) != 1:
            raise ValueError(
                "tapw chern cannot produce a Chern number from unfinished k-point chunks. "
                "Finalize chunked wavefunctions first, then run topology post-processing."
            )

        # Generate uniform k-point mesh
        num_k1, num_k2 = self.config.get_chern_grid_shape()
        kpoints = self.generate_kmesh(num_k1, num_k2)
        reporter = ensure_reporter(getattr(self, "reporter", None))
        reporter.kv("Chern k-points shape", kpoints.shape)
        
        # Calculate band structure on the mesh
        self.calculate_band_structure(path, kpoints)

        vec = getattr(self, "result", {}).get("vec")
        if vec is None:
            raise RuntimeError("tapw chern did not produce wavefunctions; check eig_vec_cal and solver output.")

        band_vec_grid = reshape_wavefunction_grid(np.asarray(vec), int(num_k1), int(num_k2))
        num_bands = int(band_vec_grid.shape[-1])
        raw_band_indices = getattr(self.config, "chern_band_indices", None)
        if raw_band_indices is None:
            if num_bands == 1:
                raw_band_indices = [0]
            else:
                raise ValueError(
                    "mode='chern' requires compute.chern_band_indices when more than one band is available."
                )

        raw_band_indices = [int(index) for index in raw_band_indices]
        resolved_band_indices = []
        for raw_band_index in [int(index) for index in raw_band_indices]:
            resolved = raw_band_index if raw_band_index >= 0 else num_bands + raw_band_index
            if resolved < 0 or resolved >= num_bands:
                raise ValueError(
                    "Band index {0} out of range for {1} bands.".format(raw_band_index, num_bands)
                )
            if resolved not in resolved_band_indices:
                resolved_band_indices.append(resolved)

        if not resolved_band_indices:
            raise ValueError("compute.chern_band_indices must not be empty.")

        canonical_layout = _is_canonical_output_layout(self.config)
        if canonical_layout:
            topo_path = os.path.join(os.fspath(path), "topology", canonical_topology_grid_name(num_k1, num_k2))
        else:
            topo_path = os.path.join(os.fspath(path), "topo")
        os.makedirs(topo_path, exist_ok=True)
        band_type = str(getattr(self.config, "band_type", "BOTH")).upper()
        suffix = self.config.get_chern_grid_suffix()
        entries = []
        files: dict[str, str] = {}
        for raw_band_index, band_index in zip(raw_band_indices, resolved_band_indices):
            berry_flux = compute_berry_flux_single_band(band_vec_grid, band_index)
            if canonical_layout:
                label = canonical_topology_band_label([raw_band_index])
                filename = f"berry_flux_{label}.npy"
                files[f"berry_flux_{label}"] = filename
            else:
                filename = chern_flux_filename(
                    band_type=band_type,
                    valley_flag=self.valley_flag,
                    suffix=suffix,
                    band_label=f"band{band_index}",
                )
            np.save(os.path.join(topo_path, filename), berry_flux)
            entries.append(
                {
                    "band_indices": [raw_band_index],
                    "resolved_band_indices": [band_index],
                    "chern_number": float(np.sum(berry_flux) / (2.0 * np.pi)),
                    "berry_flux_file": filename,
                }
            )

        if len(resolved_band_indices) > 1:
            berry_flux = compute_berry_flux_multiband(band_vec_grid, resolved_band_indices)
            if canonical_layout:
                label = canonical_topology_band_label(raw_band_indices)
                filename = f"berry_flux_{label}.npy"
                files[f"berry_flux_{label}"] = filename
            else:
                label = "bands" + "_".join(str(index) for index in resolved_band_indices)
                filename = chern_flux_filename(
                    band_type=band_type,
                    valley_flag=self.valley_flag,
                    suffix=suffix,
                    band_label=label,
                )
            np.save(os.path.join(topo_path, filename), berry_flux)
            entries.append(
                {
                    "band_indices": raw_band_indices,
                    "resolved_band_indices": resolved_band_indices,
                    "chern_number": float(np.sum(berry_flux) / (2.0 * np.pi)),
                    "berry_flux_file": filename,
                }
            )

        primary = entries[-1]
        summary = {
            "schema": "tapw_chern_summary/v1",
            "algorithm": "Fukui-Hatsugai-Suzuki Berry flux",
            "grid_shape": [int(num_k1), int(num_k2)],
            "grid_order": "ij",
            "valley": self.valley_flag,
            "band_type": band_type,
            "band_indices": primary["band_indices"],
            "resolved_band_indices": resolved_band_indices,
            "chern_number": primary["chern_number"],
            "berry_flux_file": primary["berry_flux_file"],
            "entries": entries,
        }
        summary_filename = (
            "chern_summary.json"
            if canonical_layout
            else chern_summary_filename(
                band_type=band_type,
                valley_flag=self.valley_flag,
                suffix=suffix,
                tapw=self.config.TAPW,
            )
        )
        summary["summary_file"] = summary_filename
        with open(os.path.join(topo_path, summary_filename), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if canonical_layout:
            files["chern_summary"] = summary_filename
            _write_json(
                Path(topo_path) / "manifest.json",
                {
                    "schema": "tapw_topology_outputs/v1",
                    "grid_shape": [int(num_k1), int(num_k2)],
                    "valley": self.valley_flag,
                    "files": files,
                },
            )
            topology_dir = Path(os.fspath(path)) / "topology"
            _write_json(
                topology_dir / "manifest.json",
                {
                    "schema": "tapw_topology_collection/v1",
                    "grids": [
                        {
                            "grid": canonical_topology_grid_name(num_k1, num_k2),
                            "manifest": f"{canonical_topology_grid_name(num_k1, num_k2)}/manifest.json",
                        }
                    ],
                },
            )
            _update_canonical_manifests(topology_dir, "topology")

        reporter.kv(
            "Chern number",
            "{0} {1}: {2:.8f}".format(
                self.valley_flag,
                primary["band_indices"],
                primary["chern_number"],
            )
        )
        return summary

    def run_calculation(self, path):
        """Main calculation entry point
        
        Args:
            path: Output path for results
        """
        if self.config.mode == "band":
            self.calculate_band_structure(path)
        elif self.config.mode == "chern":
            self.calculate_chern(path)
        else:
            raise ValueError(f"Unknown calculation mode: {self.config.mode}")

    def rot(self, vec, theta):
        """Rotate a vector by a given angle in degrees"""
        return rotate_vector(vec, theta)

    # @timing_decorator_factory(process_id=0)
    def get_kvec(self, k):
        """Get k vector in reciprocal space"""
        return np.dot(k, self.structure.reciprocal_Tmat)

    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse(self, Hr, k, type = "H"):
        """Get k-space Hamiltonian from real space Hamiltonian"""
        phase_ctx = self._build_getk_phase_context(k)
        return self._assemble_sparse_realspace_matrix(Hr, phase_ctx, type=type)

    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_symm(self, Hr, k, force_sparse_dot: bool = False):
        """Get k-space Hamiltonian with symmetry operations"""
        kvec = self.get_kvec(k)
        
        # Build sorted_wann array
        sorted_wann_x = self.structure.df['x'].values
        sorted_wann_y = self.structure.df['y'].values
        sorted_wann_z = self.structure.df['z'].values
        sorted_wann = np.repeat(np.array([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T,
                               self.structure.df['orb_num'].values, axis=0)
        sorted_layer_index = np.repeat(np.array(self.structure.df['layer']),
                                      self.structure.df['orb_num'].values, axis=0)
        
        if self.structure.spin:
            sorted_wann = np.concatenate([sorted_wann, sorted_wann], axis=0)
            sorted_layer_index = np.concatenate([sorted_layer_index, sorted_layer_index], axis=0)
        
        num_wann = len(sorted_wann)
        
        # Setup K points
        K1 = self.TAPW_parameters.K1
        K2 = self.TAPW_parameters.K2
        temp_k1 = np.zeros(3)
        temp_k2 = np.zeros(3)
        temp_k1[:2] = K1
        temp_k2[:2] = K2
        K1 = temp_k1
        K2 = temp_k2

        rotations = [0, -120, -240]
        kvec_K1 = [self.rot(kvec + K1, angle) - K1 for angle in rotations]
        kvec_K2 = [self.rot(kvec + K2, angle) - K2 for angle in rotations]

        mk_list = [np.zeros((self.TAPW_parameters.g_matrix.shape[0], 
                            self.TAPW_parameters.g_matrix.shape[0]), dtype=np.complex128) for _ in range(3)]

        for i in range(3):
            partial_mk = 0
            for rvec, values_dic in Hr.items():
                Rvec = np.dot(rvec, self.structure.Tmat)
                row_index, col_index, val = values_dic["row"], values_dic["col"], values_dic["val"]

                # Create boolean masks
                mask_row_1 = sorted_layer_index[row_index] == 0
                mask_row_2 = ~mask_row_1
                mask_col_1 = sorted_layer_index[col_index] == 0
                mask_col_2 = ~mask_col_1
                
                row_coords_1 = sorted_wann[row_index[mask_row_1]]
                row_coords_2 = sorted_wann[row_index[mask_row_2]]
                col_coords_1 = sorted_wann[col_index[mask_col_1]]
                col_coords_2 = sorted_wann[col_index[mask_col_2]]

                # Calculate phase factors
                exp_kvec_K1_Rvec = np.exp(1j * np.dot(kvec_K1[i], Rvec))
                exp_kvec_K2_Rvec = np.exp(1j * np.dot(kvec_K2[i], Rvec))

                phase_m_1 = np.exp(-1j * np.dot(row_coords_1, kvec_K1[i])) * exp_kvec_K1_Rvec
                phase_m_2 = np.exp(-1j * np.dot(row_coords_2, kvec_K2[i])) * exp_kvec_K2_Rvec
                phase_n_1 = np.exp(1j * np.dot(col_coords_1, kvec_K1[i]))
                phase_n_2 = np.exp(1j * np.dot(col_coords_2, kvec_K2[i]))

                phase_m = np.zeros(len(row_index), dtype=np.complex128)
                phase_n = np.zeros(len(col_index), dtype=np.complex128)
                phase_m[mask_row_1], phase_m[mask_row_2] = phase_m_1, phase_m_2
                phase_n[mask_col_1], phase_n[mask_col_2] = phase_n_1, phase_n_2

                phase_factors = phase_m * phase_n
                data_values = val * phase_factors
                partial_mk += scipy.sparse.csr_matrix((data_values, (row_index, col_index)), 
                                                     shape=(num_wann, num_wann))

            temp = self.cal_TAPW_hamiltonian_k_cpu(
                partial_mk,
                force_sparse_dot=force_sparse_dot,
            )
            mk_list[i] = temp

        return mk_list

    @timing_decorator_factory(process_id=0) 
    def Getk_super_gauge_sparse_final_HS(self, Hr, Sr, k):
        """Get final Hamiltonian for orthogonal or non-orthogonal basis (no symmetry)"""
        phase_ctx = self._build_getk_phase_context(k)
        Hk = self._assemble_sparse_realspace_matrix(Hr, phase_ctx, type="H")
        Hk = self.cal_TAPW_hamiltonian_k(Hk)
        if self.config.orthogonal_basis:
            return Hk, None
        else:
            Sk = self._assemble_sparse_realspace_matrix(Sr, phase_ctx, type="S")
            Sk = self.cal_TAPW_hamiltonian_k(Sk)
            if not self.config.ge:
                Hk = self.gen_H_new(Hk, Sk)
                return Hk, None
            else:
                return Hk, Sk

    
    @timing_decorator_factory(process_id=0)
    def C3_symm(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        """Apply C3 symmetry to Hamiltonian"""
        return self.C3_symm_cpu(hamk, hamk_C1, hamk_C2, C3_matrix)

    def C3_symm_cpu(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        """CPU version of C3 symmetry"""
        C3_matrix_2 = C3_matrix @ C3_matrix
        return (hamk + C3_matrix @ hamk_C1 @ C3_matrix.conj().T + 
                C3_matrix_2 @ hamk_C2 @ C3_matrix_2.conj().T) / 3

    @timing_decorator_factory(process_id=0)
    def gen_H_new(self, hamk, samk):
        """Generate new Hamiltonian from overlap matrix"""
        return self.gen_H_new_cpu(hamk, samk)
    
    @timing_decorator_factory(process_id=0)
    def gen_H_new_cpu(self, hamk, samk):
        """CPU version of Hamiltonian transformation"""
        # Use the legacy symmetric-orthogonalization path so TAPW eigenvectors
        # remain consistent with historical `tapw` outputs.
        s_eig, s_vec = scipy.linalg.eigh(samk, check_finite=False)
        s_inv_sqrt = np.diag(1.0 / np.sqrt(s_eig))
        uminvud = s_vec @ s_inv_sqrt @ s_vec.conj().T
        return uminvud @ hamk @ uminvud
    
    @timing_decorator_factory(process_id=0)
    def cal_TAPW_hamiltonian_k(self, hamk):
        return self.cal_TAPW_hamiltonian_k_cpu(hamk)

    def cal_TAPW_hamiltonian_k_cpu(self, hamk, tapw_parameters=None, force_sparse_dot: bool = False):
        """CPU version of TAPW Hamiltonian calculation"""
        tapw_parameters = self.TAPW_parameters if tapw_parameters is None else tapw_parameters
        g = tapw_parameters.g_matrix
        gH = tapw_parameters.g_matrix_conj

        use_sparse_dot = bool(force_sparse_dot or getattr(self.config, "use_sparse_dot_mkl", False))
        if use_sparse_dot and _HAS_SPARSE_DOT_MKL and scipy.sparse.issparse(g) and scipy.sparse.issparse(hamk) and scipy.sparse.issparse(gH):
            # MKL sparse GEMM is multi-threaded and usually much faster than SciPy's sparse matmul.
            tmp = dot_product_mkl(g, hamk, dense=False)
            out = dot_product_mkl(tmp, gH, dense=True)
            return out

        result = g @ hamk @ gH
        return result.toarray() if scipy.sparse.issparse(result) else np.asarray(result)

    def _get_raw_tapw_projected_hs_for_parameters(
        self,
        Hr,
        Sr,
        k,
        mpi_index,
        tapw_parameters,
        force_sparse_dot: bool = False,
    ):
        phase_ctx = self._build_getk_phase_context(k)
        h_full = self._assemble_sparse_realspace_matrix(Hr, phase_ctx, type="H")
        hamk = self.cal_TAPW_hamiltonian_k_cpu(
            h_full,
            tapw_parameters=tapw_parameters,
            force_sparse_dot=force_sparse_dot,
        )

        orthogonal_basis = bool(getattr(getattr(self, "config", None), "orthogonal_basis", False))
        if orthogonal_basis:
            return hamk, None

        s_full = self._assemble_sparse_realspace_matrix(Sr, phase_ctx, type="S")
        samk = self.cal_TAPW_hamiltonian_k_cpu(
            s_full,
            tapw_parameters=tapw_parameters,
            force_sparse_dot=force_sparse_dot,
        )
        return hamk, samk

    def _project_full_space_matrix_with_projector(self, full_matrix, projector, projector_h=None, force_sparse_dot: bool = False):
        if scipy.sparse.issparse(projector):
            projector = projector.tocsr()
            projector_h = projector.conj().T.tocsr() if projector_h is None else projector_h.tocsr()
            use_sparse_dot = bool(force_sparse_dot or getattr(self.config, "use_sparse_dot_mkl", False))
            if (
                use_sparse_dot
                and _HAS_SPARSE_DOT_MKL
                and scipy.sparse.issparse(full_matrix)
            ):
                tmp = dot_product_mkl(projector, full_matrix, dense=False)
                out = dot_product_mkl(tmp, projector_h, dense=True)
                return np.asarray(out)
            result = projector @ full_matrix @ projector_h
            return result.toarray() if scipy.sparse.issparse(result) else np.asarray(result)

        projector = np.asarray(projector, dtype=np.complex128)
        projector_h = projector.conj().T if projector_h is None else np.asarray(projector_h, dtype=np.complex128)
        full_matrix = full_matrix.toarray() if scipy.sparse.issparse(full_matrix) else np.asarray(full_matrix)
        return np.asarray(projector @ full_matrix @ projector_h)

    def _get_raw_projected_hs_with_projector(self, Hr, Sr, k, mpi_index, projector, projector_h=None, force_sparse_dot: bool = False):
        phase_ctx = self._build_getk_phase_context(k)
        h_full = self._assemble_sparse_realspace_matrix(Hr, phase_ctx, type="H")
        hamk = self._project_full_space_matrix_with_projector(
            h_full,
            projector,
            projector_h=projector_h,
            force_sparse_dot=force_sparse_dot,
        )

        orthogonal_basis = bool(getattr(getattr(self, "config", None), "orthogonal_basis", False))
        if orthogonal_basis:
            return hamk, None

        s_full = self._assemble_sparse_realspace_matrix(Sr, phase_ctx, type="S")
        samk = self._project_full_space_matrix_with_projector(
            s_full,
            projector,
            projector_h=projector_h,
            force_sparse_dot=force_sparse_dot,
        )
        return hamk, samk

    def _finalize_tapw_projected_hs(self, hamk, samk, mpi_index=0):
        if self.config.orthogonal_basis:
            return hamk, None

        if samk is None:
            raise ValueError("Non-orthogonal TAPW finalization requires an overlap matrix.")

        if not self.config.ge:
            return self.gen_H_new(hamk, samk), None
        return hamk, samk

    def _get_tapw_projected_hs_for_parameters(self, Hr, Sr, k, mpi_index, tapw_parameters):
        hamk, samk = self._get_raw_tapw_projected_hs_for_parameters(
            Hr, Sr, k, mpi_index, tapw_parameters
        )
        return self._finalize_tapw_projected_hs(hamk, samk, mpi_index)

    def _calculate_reference_m_valley_c3_hs(self, k_reference, mpi_index):
        if not getattr(self, "_m_valley_c3_reference_projectors", None):
            self._m_valley_c3_reference_projectors = self._build_m_valley_c3_reference_projectors()

        orthogonal_basis = bool(getattr(getattr(self, "config", None), "orthogonal_basis", False))
        h_sum = None
        s_sum = None

        for projector_term in self._m_valley_c3_reference_projectors:
            k_transformed = transform_k_by_cartesian_linear_map(
                k_reference,
                self.structure.reciprocal_Tmat,
                projector_term.linear_map_2d,
            )
            tapw_parameters = getattr(projector_term, "tapw_parameters", None)
            transport_to_ref = getattr(projector_term, "transport_to_ref", None)
            projector_valley = getattr(projector_term, "valley", None)
            if tapw_parameters is not None and transport_to_ref is not None:
                hamk, samk = self._get_raw_tapw_projected_hs_for_parameters(
                    self.hr_supercell,
                    self.sr_supercell,
                    k_transformed,
                    mpi_index,
                    tapw_parameters,
                    force_sparse_dot=True,
                )
                if projector_valley != self._m_valley_reference:
                    hamk = transport_matrix_between_m_valleys(hamk, transport_to_ref)
                    if samk is not None:
                        samk = transport_matrix_between_m_valleys(samk, transport_to_ref)
            else:
                projector_kwargs = {"force_sparse_dot": True}
                projector_h = getattr(projector_term, "projector_h", None)
                if projector_h is not None:
                    projector_kwargs["projector_h"] = projector_h
                hamk, samk = self._get_raw_projected_hs_with_projector(
                    self.hr_supercell,
                    self.sr_supercell,
                    k_transformed,
                    mpi_index,
                    projector_term.projector,
                    **projector_kwargs,
                )
            h_sum = np.array(hamk, copy=True) if h_sum is None else (h_sum + hamk)
            if orthogonal_basis:
                continue
            if samk is None:
                raise ValueError("Reference M-valley C3 averaging requires overlap matrices in non-orthogonal mode.")
            s_sum = np.array(samk, copy=True) if s_sum is None else (s_sum + samk)

        count = float(len(self._m_valley_c3_reference_projectors))
        h_avg = h_sum / count
        if orthogonal_basis:
            return h_avg, None
        if s_sum is None:
            raise ValueError("Reference M-valley C3 averaging did not accumulate any overlap matrices.")
        return h_avg, s_sum / count

    def _calculate_reference_m_valley_d3_hs(self, k_reference, mpi_index):
        if not self._m_valley_d3_reference_projectors:
            raise ValueError("Reference M-valley D3 projectors are not initialized.")

        orthogonal_basis = bool(getattr(getattr(self, "config", None), "orthogonal_basis", False))
        h_sum = None
        s_sum = None

        for projector_term in self._m_valley_d3_reference_projectors:
            k_transformed = transform_k_by_cartesian_linear_map(
                k_reference,
                self.structure.reciprocal_Tmat,
                projector_term.linear_map_2d,
            )
            projector_kwargs = {}
            projector_h = getattr(projector_term, "projector_h", None)
            if projector_h is not None:
                projector_kwargs["projector_h"] = projector_h
            hamk, samk = self._get_raw_projected_hs_with_projector(
                self.hr_supercell,
                self.sr_supercell,
                k_transformed,
                mpi_index,
                projector_term.projector,
                **projector_kwargs,
            )
            h_sum = np.array(hamk, copy=True) if h_sum is None else (h_sum + hamk)
            if orthogonal_basis:
                continue
            if samk is None:
                raise ValueError("Reference M-valley D3 averaging requires overlap matrices in non-orthogonal mode.")
            s_sum = np.array(samk, copy=True) if s_sum is None else (s_sum + samk)

        count = float(len(self._m_valley_d3_reference_projectors))
        h_avg = h_sum / count
        if orthogonal_basis:
            return h_avg, None
        if s_sum is None:
            raise ValueError("Reference M-valley D3 averaging did not accumulate any overlap matrices.")
        return h_avg, s_sum / count

    def _calculate_m_valley_threefold_hs(self, k, mpi_index=0):
        k_reference = rotate_local_k_between_m_valleys(
            k,
            self.structure.reciprocal_Tmat,
            source_valley=self.config.valley,
            target_valley=self._m_valley_reference,
        )

        if getattr(self, "use_M_valley_d3_symm", False):
            h_ref_sym, s_ref_sym = self._calculate_reference_m_valley_d3_hs(k_reference, mpi_index)
        else:
            h_ref_sym, s_ref_sym = self._calculate_reference_m_valley_c3_hs(k_reference, mpi_index)

        if self.config.valley == self._m_valley_reference:
            return self._finalize_tapw_projected_hs(h_ref_sym, s_ref_sym, mpi_index)

        # For eigenvalue-only band runs, transporting the averaged reference H/S into the
        # target M-valley basis is unnecessary: the generalized eigenvalues are invariant
        # under this basis change. Skipping the dense transport removes pure overhead.
        if (
            not getattr(self, "use_M_valley_d3_symm", False)
            and not getattr(self.config, "eig_vec_cal", False)
            and not getattr(self.config, "hamk_save", False)
        ):
            return self._finalize_tapw_projected_hs(h_ref_sym, s_ref_sym, mpi_index)

        transport = self._m_valley_transport_ref_to_valley[self.config.valley]
        hamk = transport_matrix_between_m_valleys(h_ref_sym, transport)
        samk = None if s_ref_sym is None else transport_matrix_between_m_valleys(s_ref_sym, transport)
        return self._finalize_tapw_projected_hs(hamk, samk, mpi_index)
    
    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_symm_final_HS(self, Hr, Sr, symm_matrix, symm_matrix_inv, k):
        """Get final Hamiltonian for orthogonal or non-orthogonal basis (with symmetry)"""
        Hk_list = self.Getk_super_gauge_sparse_symm(Hr, k)
        if self.config.orthogonal_basis:
            if self.config.ge:
                HK_new = self.C3_symm(Hk_list[0], Hk_list[1], Hk_list[2], self.TAPW_parameters.C3_matrix)
                return HK_new, None
            else:
                Hk_new = self.C3_symm(Hk_list[0], Hk_list[1], Hk_list[2], self.TAPW_parameters.C3_matrix)
                return Hk_new, None
        else:
            Sk_list = self.Getk_super_gauge_sparse_symm(Sr, k)
            if self.config.ge:
                HK_new = self.C3_symm(Hk_list[0], Hk_list[1], Hk_list[2], self.TAPW_parameters.C3_matrix)
                Sk_new = self.C3_symm(Sk_list[0], Sk_list[1], Sk_list[2], self.TAPW_parameters.C3_matrix)
                return HK_new, Sk_new
            else:
                Hk_new_list = [self.gen_H_new(Hk, Sk) for Hk, Sk in zip(Hk_list, Sk_list)]
                Hk_new = self.C3_symm(Hk_new_list[0], Hk_new_list[1], Hk_new_list[2], self.TAPW_parameters.C3_matrix)
                return Hk_new, None

    @timing_decorator_factory(process_id=0)
    def calculate_band_01(self, kpoints, i):
        """Calculate band structure for a single k-point"""
        if self.config.TAPW:
            self._set_progress_stage(i, "build_hs")
            if self.use_M_valley_threefold_symm:
                hamk, samk = self._calculate_m_valley_threefold_hs(kpoints[:3])
            elif self.use_single_valley_c3:
                hamk, samk = self.Getk_super_gauge_sparse_symm_final_HS(
                    self.hr_supercell, self.sr_supercell, 
                    self.TAPW_parameters.symm_matrix, self.TAPW_parameters.symm_matrix_inv, 
                    kpoints[:3]
                )
            else:
                hamk, samk = self.Getk_super_gauge_sparse_final_HS(
                    self.hr_supercell, self.sr_supercell, kpoints[:3]
                )
            # 正交基底下直接对角化Hk
            self._set_progress_stage(i, "solve")
            if self.config.orthogonal_basis:
                if self.config.eigsh_cal:
                    w = eigsh(hamk, k=self.config.num_bands_cal, sigma=self.config.efermi, 
                              which='LM', return_eigenvectors=self.config.eig_vec_cal)
                else:
                    a = hamk.toarray() if scipy.sparse.issparse(hamk) else hamk
                    w = scipy.linalg.eigh(a, eigvals_only=not self.config.eig_vec_cal, check_finite=False)
            else:
                if self.config.eigsh_cal:
                    if self.config.ge:
                        w = eigsh(hamk, k=self.config.num_bands_cal, M=samk, 
                                 sigma=self.config.efermi, which='LM', 
                                 return_eigenvectors=self.config.eig_vec_cal)
                    else:
                        w = eigsh(hamk, k=self.config.num_bands_cal, sigma=self.config.efermi, 
                                 which='LM', return_eigenvectors=self.config.eig_vec_cal)
                else:
                    a = hamk.toarray() if scipy.sparse.issparse(hamk) else hamk
                    if self.config.ge and samk is not None:
                        b = samk.toarray() if scipy.sparse.issparse(samk) else samk
                        w = scipy.linalg.eigh(a, b, eigvals_only=not self.config.eig_vec_cal, check_finite=False)
                    else:
                        w = scipy.linalg.eigh(a, eigvals_only=not self.config.eig_vec_cal, check_finite=False)
        else:
            if self.config.ge:
                self._set_progress_stage(i, "build_h")
                hamk = self.Getk_super_gauge_sparse(self.hr_supercell, kpoints[:3], type="H")
                if self.config.orthogonal_basis:
                    samk = None
                else:
                    self._set_progress_stage(i, "build_s")
                    samk = self.Getk_super_gauge_sparse(self.sr_supercell, kpoints[:3], type="S")
                self._set_progress_stage(i, "solve")
                w = _solve_notapw_generalized_eigs(hamk, samk, self.config)
            else:
                raise ValueError("non-TAPW calculations require ge=true in the release TAPW workflow.")
        if self.config.eig_vec_cal:
            eig = np.sort(np.real(w[0]))
            vec = w[1][:, np.argsort(np.real(w[0]))]
        else:
            eig = np.sort(np.real(w))
            vec = None

        # IMPORTANT: Do not return/store per-kpoint H(k)/S(k) unless explicitly requested.
        # Returning large matrices from joblib workers causes heavy pickling overhead and
        # can easily blow up memory when running many k-points or using many workers.
        if not self.config.hamk_save:
            hamk = None
            samk = None
        self._set_progress_stage(i, "post")
        return eig, vec, hamk, samk

    @timing_decorator_factory(process_id=0)
    def parallel_calculate_band_01(self, kpoints, kpoint_indices=None, nk_total=None, out_dir=None, suffix=""):
        """Calculate band structure for all k-points in parallel.

        Key performance points:
        - Limit BLAS/OpenMP threads inside each worker via `blas_threads` to avoid oversubscription.
        - For large k-mesh + wavefunctions, set `vec_store="memmap"` so workers write slices to disk and
          avoid returning huge arrays through joblib (pickle overhead + RAM blow-ups).
        """
        start_time = time.time()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sys.stdout.flush()
        self.reporter.stage("K-point solve", "Diagonalize the projected Hamiltonian along the requested k-path or mesh.")
        self.reporter.kv("Start time", current_time)
        self.reporter.kv("K-points", len(kpoints))

        num_processes = int(getattr(self.config, "num_processes", 1))
        blas_threads = int(getattr(self.config, "blas_threads", 1))
        parallel_policy = resolve_kpoint_parallel_policy(self.config)
        parallel_impl = parallel_policy.parallel_impl
        parallel_backend = parallel_policy.parallel_backend
        vec_store = getattr(self.config, "vec_store", "memory")

        if parallel_policy.auto_promoted:
            self.reporter.kv(
                "Parallel auto-switch",
                "TAPW k-point loop switched from joblib/loky to mp/fork to avoid large-calculator pickling overhead.",
            )

        self.reporter.kv(
            "Parallel config",
            f"impl={parallel_impl}, backend={parallel_backend}, "
            f"num_processes={num_processes}, blas_threads={blas_threads}, vec_store={vec_store}",
        )

        if kpoint_indices is None:
            kpoint_indices = list(range(len(kpoints)))
        if nk_total is None:
            nk_total = len(kpoints)
        if len(kpoint_indices) != len(kpoints):
            raise ValueError("kpoint_indices length must match kpoints length")
        kpoint_indices = [int(i) for i in kpoint_indices]

        use_memmap = (vec_store == "memmap") or (int(getattr(self.config, "kpoint_chunk_count", 1)) > 1)
        self._progress_dir = None

        eig_path = None
        vec_path = None
        if use_memmap:
            root = getattr(self.config, "memmap_dir", None) or out_dir or os.getcwd()
            memmap_dir = os.path.join(root, "memmap")
            os.makedirs(memmap_dir, exist_ok=True)

            # File names include the same suffix as the final outputs, so different modes/num_chern don't collide.
            eig_path = os.path.join(
                memmap_dir,
                raw_memmap_filename("eig", valley_flag=self.valley_flag, suffix=suffix),
            )
            vec_path = os.path.join(
                memmap_dir,
                raw_memmap_filename("vec", valley_flag=self.valley_flag, suffix=suffix),
            )

            nb = int(self.config.num_bands_cal) if getattr(self.config, "eigsh_cal", True) else None
            dim = int(self.TAPW_parameters.g_matrix.shape[0]) if getattr(self.config, "TAPW", False) else None
            if nb is None or dim is None:
                raise ValueError("memmap mode requires TAPW=True and eigsh_cal=True (fixed num_bands_cal).")

            _ensure_memmap_file(eig_path, np.float64, (int(nk_total), nb))
            if getattr(self.config, "eig_vec_cal", False):
                _ensure_memmap_file(vec_path, np.complex128, (int(nk_total), dim, nb))
            else:
                vec_path = None

            # Keep handles in this process for downstream post-processing.
            self.result["eig_path"] = eig_path
            self.result["vec_path"] = vec_path
            self.result["eig"] = open_memmap(eig_path, mode="r+")
            self.result["vec"] = open_memmap(vec_path, mode="r+") if vec_path is not None else None

        def _joblib_worker(i, kpoint):
            try:
                _maybe_pin_current_worker(blas_threads, num_processes)
                _set_thread_limits(blas_threads)
                eig, vec, hamk, samk = self.calculate_band_01(kpoint, i)

                if use_memmap and eig_path is not None:
                    eig_mm = _get_memmap(eig_path)
                    eig_mm[i, : eig.shape[0]] = eig
                    if getattr(self.config, "eig_vec_cal", False) and vec_path is not None and vec is not None:
                        vec_mm = _get_memmap(vec_path)
                        vec_mm[i, :, : vec.shape[1]] = vec
                    self._set_progress_stage(i, "done")
                    return i, True, None

                self._set_progress_stage(i, "done")
                return eig, vec, hamk, samk
            except Exception:
                self._set_progress_stage(i, "error")
                raise

        with _progress_reporter(total=len(kpoints)):
                if parallel_impl == "mp":
                    # Fork-based pool avoids repeatedly pickling a huge calculator object.
                    ctx = mp.get_context("fork")
                    _MP_STATE.clear()
                    _MP_STATE.update(
                        {
                            "calc": self,
                            "cfg": self.config,
                            "eig_path": eig_path,
                            "vec_path": vec_path,
                            "use_memmap": use_memmap,
                        }
                    )
                    with ctx.Pool(
                        processes=num_processes,
                        initializer=_mp_worker_init,
                        initargs=(blas_threads, num_processes),
                    ) as pool:
                        statuses = list(
                            tqdm(
                                pool.imap_unordered(_mp_kpoint_worker, zip(kpoint_indices, kpoints)),
                                total=len(kpoints),
                                desc="k-points",
                                unit="kpt",
                                mininterval=5.0,
                                dynamic_ncols=True,
                            )
                        )
                    failed = [(i, err) for (i, ok, err, _) in statuses if not ok]
                    if failed:
                        raise RuntimeError(f"{len(failed)} k-points failed, first: {failed[0]}")
                    if not use_memmap:
                        # Assemble results in memory, matching the original (joblib) behavior.
                        eig_arr = None
                        vec_arr = None
                        want_vec = bool(getattr(self.config, "eig_vec_cal", False))
                        want_hamk = bool(getattr(self.config, "hamk_save", False))
                        hamk_list = [None] * nk_total if want_hamk else None
                        samk_list = [None] * nk_total if want_hamk else None

                        for (i, ok, err, payload) in statuses:
                            if not ok:
                                raise RuntimeError(err or f"k-point {i} failed")
                            if payload is None:
                                raise RuntimeError(f"Missing payload for k-point {i} in non-memmap mp mode")
                            eig, vec, hamk, samk = payload
                            eig = np.asarray(eig)
                            if eig_arr is None:
                                eig_arr = np.empty((nk_total, eig.shape[0]), dtype=eig.dtype)
                            eig_arr[i, : eig.shape[0]] = eig

                            if want_vec:
                                if vec is None:
                                    raise RuntimeError(f"eig_vec_cal=True but vec is None for k-point {i}")
                                vec = np.asarray(vec)
                                if vec_arr is None:
                                    vec_arr = np.empty((nk_total, vec.shape[0], vec.shape[1]), dtype=vec.dtype)
                                vec_arr[i, :, : vec.shape[1]] = vec

                            if want_hamk and hamk_list is not None and samk_list is not None:
                                hamk_list[i] = hamk
                                samk_list[i] = samk

                        if eig_arr is None:
                            raise RuntimeError("No eigenvalues collected in non-memmap mp mode")
                        self.result["eig"] = eig_arr
                        self.result["vec"] = vec_arr if want_vec else None
                        if want_hamk and hamk_list is not None and samk_list is not None:
                            self.result["hamk"] = hamk_list
                            self.result["samk"] = samk_list
                        else:
                            self.result.pop("hamk", None)
                            self.result.pop("samk", None)
                else:
                    if num_processes == 1:
                        # Avoid joblib/loky overhead (and extra helper processes) for the common MPI-per-kpoint case.
                        results = [
                            _joblib_worker(i, kpoint)
                            for i, kpoint in tqdm(
                                zip(kpoint_indices, kpoints),
                                total=len(kpoints),
                                desc="k-points",
                                unit="kpt",
                                mininterval=5.0,
                                dynamic_ncols=True,
                            )
                        ]
                    else:
                        with _tqdm_joblib(total=len(kpoints), desc="k-points"):
                            results = Parallel(n_jobs=num_processes, backend=parallel_backend)(
                                delayed(_joblib_worker)(i, kpoint)
                                for i, kpoint in zip(kpoint_indices, kpoints)
                            )

                    if not use_memmap:
                        eig, vec, hamk, samk = zip(*results)
                        self.result["eig"] = np.array(eig)
                        self.result["vec"] = np.array(vec) if getattr(self.config, "eig_vec_cal", False) else None
                        if getattr(self.config, "hamk_save", False):
                            self.result["hamk"] = hamk
                            self.result["samk"] = samk
                        else:
                            self.result.pop("hamk", None)
                            self.result.pop("samk", None)
        self._progress_dir = None

        end_time = time.time()
        self.reporter.kv("Running time (seconds)", f"{end_time - start_time:.2f}")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.reporter.kv("End time", current_time)
        if use_memmap:
            with suppress(Exception):
                eig = self.result.get("eig")
                if eig is not None:
                    eig.flush()
            with suppress(Exception):
                vec = self.result.get("vec")
                if vec is not None:
                    vec.flush()

    def generate_indices(self, num_gn_all, num_Te, num_Mo, orbs_num):
        """Generate indices for spin up/down components"""
        up_index = np.concatenate((np.arange(num_Te), np.arange(num_Te) + num_Te * 2, 
                                  np.arange(num_Mo) + num_Te * 4))
        down_index = np.concatenate((np.arange(num_Te) + num_Te, np.arange(num_Te) + num_Te * 3, 
                                    np.arange(num_Mo) + num_Te * 4 + num_Mo))

        up_all_index = np.concatenate([up_index + orbs_num * 2 * i for i in range(num_gn_all)])
        down_all_index = np.concatenate([down_index + orbs_num * 2 * i for i in range(num_gn_all)])

        return up_all_index.astype(int), down_all_index.astype(int)

    def write_wave_function_spin(self, path):
        """Write wave function for all k-points and each spin to files"""
        num_Te = self.structure.X_orb
        num_Mo = self.structure.M_orb
        n_g = self.config.n_g
        valley_flag = self.config.valley_flag
        
        g_vec_list_K_1layer = np.load(
            os.path.join(path, gvec_output_filename(n_g=n_g, valley_flag=valley_flag, layer=1))
        )
        g_vec_list_K_2layer = np.load(
            os.path.join(path, gvec_output_filename(n_g=n_g, valley_flag=valley_flag, layer=2))
        )

        orb_num = self.structure.num_orbs_per_unit_cell
        band_wave = self.result['vec']
        self.reporter.kv("Band wavefunction shape", np.shape(band_wave))
        
        dir = os.path.join(path, f'{self.config.band_type}_{valley_flag}_valley')
        os.makedirs(dir, exist_ok=True)
        
        num_gn_all = len(g_vec_list_K_1layer) * 2
        up_all_index, down_all_index = self.generate_indices(num_gn_all, num_Te, num_Mo, orb_num)
        self.reporter.kv("Spin-up indices", up_all_index)
        self.reporter.kv("Spin-down indices", down_all_index)
        self.reporter.kv("Band wavefunction shape", band_wave.shape)
        
        np.save(os.path.join(dir, f'{self.config.band_type}_{valley_flag}_valley_up.npy'), band_wave[:, up_all_index])
        np.save(os.path.join(dir, f'{self.config.band_type}_{valley_flag}_valley_down.npy'), band_wave[:, down_all_index])

    def write_hamk_spin(self, path):
        """Write Hamiltonian matrix for all k-points and each spin to files"""
        hamk = self.result['hamk']
        dim_Hprime = np.shape(hamk[0])[0]

        num_Te = self.structure.X_orb
        num_Mo = self.structure.M_orb
        n_g = self.config.n_g
        valley_flag = self.config.valley_flag
        orb_num = self.structure.num_orbs_per_unit_cell
        
        g_vec_list_K_1layer = np.load(
            os.path.join(path, gvec_output_filename(n_g=n_g, valley_flag=valley_flag, layer=1))
        )
        num_gn_all = len(g_vec_list_K_1layer) * 2 
        up_all_index, down_all_index = self.generate_indices(num_gn_all, num_Te, num_Mo, orb_num)
        num_kpoints = len(self.kpath_config.kpoints)
        
        H_spin_kpoints = np.zeros((2, num_kpoints, int(dim_Hprime/2), int(dim_Hprime/2)), 
                                 dtype=np.complex64)
        
        for i in tqdm(range(num_kpoints)):
            gamma_hamk = hamk[i]
            H_gamma_up = gamma_hamk[up_all_index][:, up_all_index]
            H_gamma_down = gamma_hamk[down_all_index][:, down_all_index]
            
            H_spin_kpoints[0, i] = H_gamma_up
            H_spin_kpoints[1, i] = H_gamma_down

        os.makedirs(os.path.join(path, 'symm_Hprime_wave_npy'), exist_ok=True)
        np.save(os.path.join(path, 'symm_Hprime_wave_npy', f'Hprime_up_down_{self.config.band_type}_{self.config.valley_flag}.npy'), 
                H_spin_kpoints)

    def write_wave_2col(self, path, vec, g_vec_list_1layer, g_vec_list_2layer, orb_Te, orb_Mo):
        """Write wave function to 2-column format"""
        num_wann = len(vec)
        num_wann_perlayer = int(num_wann/2)
        
        with open(path, "w") as f:
            arr1 = np.arange(1, orb_Te+1)
            arr2 = np.arange(1, orb_Mo+1)

            orb_index_num = np.concatenate((arr1, arr1, arr1, arr1, arr2, arr2))
            orb_spin_index = np.concatenate((['up']*orb_Te, ['down']*orb_Te, ['up']*orb_Te, 
                                           ['down']*orb_Te, ['up']*orb_Mo, ['down']*orb_Mo))
            atoms_index = np.concatenate((['Te2']*orb_Te*2, ['Te1']*orb_Te*2, ['Mo']*orb_Mo*2))
            orb_all = orb_Te*4 + orb_Mo*2
            
            f.write("#layer  gvec     gvec_x     gvec_y  atom  orb  spin     real       imag\n")
            
            for i in range(num_wann):
                if i < num_wann_perlayer:
                    g_vec_index = i // orb_all
                    g_vec = g_vec_list_1layer[g_vec_index]
                    layer = 1
                else:
                    g_vec_index = (i - int(num_wann_perlayer)) // orb_all
                    g_vec = g_vec_list_2layer[g_vec_index]
                    layer = 2
                
                f.write(f"{layer:>5d} {g_vec_index+1:>5d} {g_vec[0]:>12.6f} {g_vec[1]:>10.6f} "
                       f"{atoms_index[i%orb_all]:>4s} {orb_index_num[i%orb_all]:>4d} "
                       f"{orb_spin_index[i%orb_all]:>5s} {vec[i].real:>10.6f} {vec[i].imag:>10.6f}\n")
