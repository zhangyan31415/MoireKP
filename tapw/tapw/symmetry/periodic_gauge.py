from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse

from ..identity import hash_array, hash_mapping, require_identity_fields


BOUNDARY_SEWING_SCHEMA = "tapw.boundary_sewing.v1"
BOUNDARY_SEWING_SCHEMA_VERSION = 1
_IDENTITY_FIELDS = (
    "identity_schema",
    "input_hash",
    "config_hash",
    "basis_hash",
    "package_version",
    "schema_version",
)


def atomic_bloch_phase(
    structure_df: Any,
    shift_by_group_cart: Mapping[int, np.ndarray],
) -> np.ndarray:
    if structure_df is None:
        raise ValueError("Structure dataframe is required to build the atomic Bloch phase")
    frame = structure_df.copy().sort_values(["atom_type"], kind="stable").reset_index(drop=True)
    required = {"twist_group", "atom_type", "orb_num"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Structure dataframe is missing periodic-gauge columns: {missing}")
    if {"x", "y"}.issubset(frame.columns):
        coordinate_columns = ["x", "y"]
    elif {"shifted_x", "shifted_y"}.issubset(frame.columns):
        coordinate_columns = ["shifted_x", "shifted_y"]
    else:
        raise ValueError("Structure dataframe lacks x/y coordinates for the atomic Bloch phase")

    orbital_counts = frame["orb_num"].to_numpy(dtype=int)
    if np.any(orbital_counts <= 0):
        raise ValueError("Structure orbital counts must be positive")
    coordinates = np.repeat(
        frame[coordinate_columns].to_numpy(dtype=float),
        orbital_counts,
        axis=0,
    )
    groups = np.repeat(frame["twist_group"].to_numpy(dtype=int), orbital_counts, axis=0)
    phases = np.empty(coordinates.shape[0], dtype=np.complex128)
    for group in np.unique(groups):
        shift = np.asarray(
            shift_by_group_cart.get(int(group), np.zeros(2)),
            dtype=float,
        ).reshape(2)
        mask = groups == int(group)
        phases[mask] = np.exp(-1.0j * (coordinates[mask] @ shift))
    return phases


def project_atomic_bloch_gauge(
    g_matrix: Any,
    spinless_phase: np.ndarray,
) -> scipy.sparse.csr_matrix:
    projector = (
        g_matrix.tocsr()
        if scipy.sparse.issparse(g_matrix)
        else scipy.sparse.csr_matrix(np.asarray(g_matrix, dtype=np.complex128))
    )
    phase = np.asarray(spinless_phase, dtype=np.complex128).reshape(-1)
    if projector.shape[1] == phase.size:
        full_phase = phase
    elif projector.shape[1] == 2 * phase.size:
        full_phase = np.concatenate([phase, phase])
    else:
        raise ValueError(
            "Atomic Bloch phase is incompatible with the TAPW projector: "
            f"phase_dim={phase.size}, projector_columns={projector.shape[1]}"
        )
    phase_matrix = scipy.sparse.diags(
        full_phase,
        offsets=0,
        dtype=np.complex128,
        format="csr",
    )
    projected = (projector @ phase_matrix @ projector.conj().T).tocsr()
    projected.sort_indices()
    return projected


def build_projected_boundary_operator(
    *,
    structure_df: Any,
    g_matrix: Any,
    reciprocal_shift: np.ndarray,
    spinful: bool,
) -> tuple[scipy.sparse.csr_matrix, dict[str, Any]]:
    frame = structure_df.copy()
    groups = sorted(frame["twist_group"].astype(int).unique().tolist())
    shift = np.asarray(reciprocal_shift, dtype=float).reshape(2)
    phase = atomic_bloch_phase(
        frame,
        {int(group): shift for group in groups},
    )
    projected_plus = project_atomic_bloch_gauge(g_matrix, phase)
    boundary = projected_plus.conj().T.tocsr()
    boundary.sort_indices()
    projected_dim = int(boundary.shape[0])
    if boundary.shape[0] != boundary.shape[1]:
        raise ValueError(f"Projected boundary operator must be square, got {boundary.shape}")
    expected_spin_factor = 2 if spinful else 1
    if projected_dim % expected_spin_factor != 0:
        raise ValueError(
            f"Projected dimension {projected_dim} is incompatible with spinful={spinful}"
        )
    identity = scipy.sparse.identity(projected_dim, dtype=np.complex128, format="csr")
    unitarity_delta = (boundary.conj().T @ boundary - identity).tocsr()
    unitarity_residual = float(np.sqrt(np.sum(np.abs(unitarity_delta.data) ** 2))) / max(
        1.0,
        float(np.sqrt(projected_dim)),
    )
    return boundary, {
        "atomic_spinless_dim": int(phase.size),
        "projected_dim": projected_dim,
        "spinful": bool(spinful),
        "reciprocal_shift": shift.tolist(),
        "operator_direction": "endpoint_to_start",
        "projected_unitarity_residual": unitarity_residual,
    }


def projected_basis_hash(
    *,
    structure_df: Any,
    g_matrix: Any,
    valley: str | int,
    q_shell: int,
    spinful: bool,
) -> str:
    projector = (
        g_matrix.tocsr()
        if scipy.sparse.issparse(g_matrix)
        else scipy.sparse.csr_matrix(np.asarray(g_matrix, dtype=np.complex128))
    )
    projector.sort_indices()
    frame = structure_df.copy().sort_values(["atom_type"], kind="stable").reset_index(drop=True)
    columns = [
        column
        for column in ("twist_group", "atom_type", "species", "orb_name", "orb_num")
        if column in frame.columns
    ]
    return hash_mapping(
        {
            "schema": "tapw.projected-basis.v1",
            "valley": str(valley),
            "q_shell": int(q_shell),
            "spinful": bool(spinful),
            "shape": list(projector.shape),
            "g_data_hash": hash_array(projector.data),
            "g_indices_hash": hash_array(projector.indices),
            "g_indptr_hash": hash_array(projector.indptr),
            "structure": frame[columns].to_dict(orient="records"),
        }
    )


def save_boundary_operators(
    path: str | Path,
    operators: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    if set(operators) != {"b1", "b2"}:
        raise ValueError("Boundary sewing pack requires exactly b1 and b2 operators")
    require_identity_fields(metadata, _IDENTITY_FIELDS, "boundary sewing metadata")
    payload: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for name in ("b1", "b2"):
        matrix = (
            operators[name].tocsr()
            if scipy.sparse.issparse(operators[name])
            else scipy.sparse.csr_matrix(np.asarray(operators[name], dtype=np.complex128))
        )
        matrix.sort_indices()
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Boundary operator {name} must be square, got {matrix.shape}")
        if shape is not None and matrix.shape != shape:
            raise ValueError(f"Boundary operator shapes differ: {shape} vs {matrix.shape}")
        shape = matrix.shape
        payload[f"{name}_data"] = matrix.data
        payload[f"{name}_indices"] = matrix.indices
        payload[f"{name}_indptr"] = matrix.indptr
        payload[f"{name}_shape"] = np.asarray(matrix.shape, dtype=np.int64)
    payload["metadata_json"] = np.asarray(
        json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False),
        dtype=str,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def load_boundary_operators(
    path: str | Path,
) -> tuple[dict[str, scipy.sparse.csr_matrix], dict[str, Any]]:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"Boundary sewing pack is missing metadata_json: {input_path}")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        require_identity_fields(metadata, _IDENTITY_FIELDS, f"boundary sewing pack {input_path}")
        operators = {}
        for name in ("b1", "b2"):
            required = [f"{name}_data", f"{name}_indices", f"{name}_indptr", f"{name}_shape"]
            missing = [key for key in required if key not in payload.files]
            if missing:
                raise ValueError(f"Boundary sewing pack is missing {name} fields: {missing}")
            shape = tuple(int(value) for value in np.asarray(payload[f"{name}_shape"]).tolist())
            operators[name] = scipy.sparse.csr_matrix(
                (
                    np.asarray(payload[f"{name}_data"], dtype=np.complex128),
                    np.asarray(payload[f"{name}_indices"], dtype=np.int64),
                    np.asarray(payload[f"{name}_indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
    if operators["b1"].shape != operators["b2"].shape:
        raise ValueError("Boundary sewing b1/b2 operator dimensions differ")
    return operators, metadata
