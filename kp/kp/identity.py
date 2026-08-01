from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__ as KP_VERSION


IDENTITY_SCHEMA = "moirekp.artifact-identity.v1"
_FILE_SCHEMA = b"moirekp:file:v1\0"
_ARRAY_SCHEMA = b"moirekp:array:v1\0"
_MAPPING_SCHEMA = b"moirekp:mapping:v1\0"
PROJECTION_BASIS_SCHEMA_VERSION = 2
PROJECTION_BASIS_HANDOFF_VERSION = "kp_project_basis_handoff_v2"
PROJECTION_ARTIFACT_IDENTITY_FIELDS = (
    "identity_schema",
    "input_hash",
    "config_hash",
    "basis_hash",
    "package_version",
    "schema_version",
    "k_indices_hash",
    "heff_hash",
)


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    digest.update(_FILE_SCHEMA)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: Any) -> str:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object-dtype arrays cannot be used for artifact identity")
    contiguous = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_ARRAY_SCHEMA)
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def hash_mapping(mapping: Mapping[str, Any]) -> str:
    if not isinstance(mapping, Mapping):
        raise TypeError("hash_mapping expects a mapping")
    encoded = json.dumps(
        _canonical_value(mapping),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_MAPPING_SCHEMA)
    digest.update(encoded)
    return digest.hexdigest()


def require_identity_fields(
    metadata: Mapping[str, Any],
    fields: Sequence[str],
    context: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    for field in fields:
        if field not in metadata:
            missing.append(str(field))
            continue
        value = _metadata_scalar(metadata[field], field=str(field), context=context)
        if value is None or value == "":
            missing.append(str(field))
            continue
        values[str(field)] = value
    if missing:
        raise KeyError(f"{context} is missing required identity fields: {', '.join(missing)}")
    return values


def require_matching_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields: Sequence[str],
    context: str,
) -> None:
    expected = require_identity_fields(left, fields, context)
    actual = require_identity_fields(right, fields, context)
    mismatched = [field for field in fields if expected[str(field)] != actual[str(field)]]
    if mismatched:
        details = ", ".join(
            f"{field}: {expected[str(field)]!r} != {actual[str(field)]!r}"
            for field in mismatched
        )
        raise ValueError(f"{context} identity mismatch ({details})")


def exactified_operation_provenance_is_complete(record: Mapping[str, Any]) -> bool:
    report = record.get("source_matrix_projection_report")
    report_status = None
    if isinstance(report, Mapping) and isinstance(report.get("report"), Mapping):
        report_status = str(report["report"].get("status", ""))
    return all(
        (
            str(record.get("matrix_kind", "")) == "continuum_internal_rep_exact",
            str(record.get("matrix_source", record.get("matrix_file_role", "")))
            == "kp_symm_exactified_action",
            str(record.get("status", "")) == "exactified",
            str(record.get("exactification_status", "")) == "exactified",
            str(record.get("exactification_owner", "")) == "kp_symm",
            bool(str(record.get("basis_hash", "")).strip()),
            report_status == "exactified",
        )
    )


def build_projection_basis_identity(
    *,
    qset1: Any,
    qset2: Any,
    source_hamiltonian_hash: str | None = None,
    spin: str,
    mode: str,
    energy_scale: float,
    nlow_state_list: Any,
    resolved_norb_fix_list: Any,
    gauge_mode: str,
    num_layer_list: Any,
    num_orb_per_layer_list: Any,
    orbital_block_dim: int,
    model_dim: int,
    k_indices: Any,
    gauge_frame_hash: str | None = None,
) -> dict[str, Any]:
    hamiltonian_hash = None if source_hamiltonian_hash is None else str(source_hamiltonian_hash).strip()
    if source_hamiltonian_hash is not None and not hamiltonian_hash:
        raise ValueError("source_hamiltonian_hash must be nonempty")
    qset1_hash = hash_array(np.asarray(qset1, dtype=float))
    qset2_hash = hash_array(np.asarray(qset2, dtype=float))
    k_indices_array = np.asarray(k_indices, dtype=np.int64)
    if k_indices_array.ndim != 1:
        raise ValueError(f"k_indices must be one-dimensional, got {k_indices_array.shape}")
    k_indices_hash = hash_array(k_indices_array)
    input_hash = hash_mapping(
        {
            "source_hamiltonian_hash": hamiltonian_hash,
            "qset1_hash": qset1_hash,
            "qset2_hash": qset2_hash,
        }
    )
    config_payload = {
        "spin": str(spin).lower(),
        "mode": str(mode).lower(),
        "energy_scale": float(energy_scale),
        "nlow_state_list": nlow_state_list,
        "resolved_norb_fix_list": resolved_norb_fix_list,
        "gauge_mode": str(gauge_mode),
        "num_layer_list": num_layer_list,
        "num_orb_per_layer_list": num_orb_per_layer_list,
        "orbital_block_dim": int(orbital_block_dim),
        "model_dim": int(model_dim),
        "k_indices": k_indices_array.tolist(),
        "gauge_frame_hash": None if gauge_frame_hash is None else str(gauge_frame_hash),
    }
    config_hash = hash_mapping(config_payload)
    basis_hash = hash_mapping(
        {
            "schema": "kp.projection-basis.v2",
            "input_hash": input_hash,
            "config_hash": config_hash,
            "qset1_hash": qset1_hash,
            "qset2_hash": qset2_hash,
            "k_indices_hash": k_indices_hash,
            "row_layout": {
                "num_layer_list": num_layer_list,
                "num_orb_per_layer_list": num_orb_per_layer_list,
                "orbital_block_dim": int(orbital_block_dim),
                "model_dim": int(model_dim),
            },
        }
    )
    identity = {
        "identity_schema": IDENTITY_SCHEMA,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "basis_hash": basis_hash,
        "package_version": KP_VERSION,
        "schema_version": PROJECTION_BASIS_SCHEMA_VERSION,
        "k_indices_hash": k_indices_hash,
    }
    if hamiltonian_hash is not None:
        identity["source_hamiltonian_hash"] = hamiltonian_hash
    return identity


def load_projection_k_indices(heff_file: str | Path) -> list[int]:
    heff_path = Path(heff_file)
    heff = np.load(heff_path, mmap_mode="r", allow_pickle=False)
    if heff.ndim < 3:
        raise ValueError(f"projection/heff.npy must have shape (Nk, dim, dim), got {heff.shape}")
    wavefunctions = heff_path.parent / "wavefunctions.npz"
    if not wavefunctions.is_file():
        raise FileNotFoundError(f"KP projection artifact is missing: {wavefunctions}")
    with np.load(wavefunctions, allow_pickle=False) as payload:
        if "k_indices" not in payload.files:
            raise KeyError(f"KP projection wavefunctions are missing k_indices: {wavefunctions}")
        raw = np.asarray(payload["k_indices"])
        stored_hash = require_identity_fields(
            payload,
            ("k_indices_hash",),
            f"KP projection artifacts {heff_path.parent}",
        )["k_indices_hash"]
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"projection k_indices must be a one-dimensional integer array, got {raw.shape}/{raw.dtype}")
    indices = [int(value) for value in raw.tolist()]
    if len(indices) != int(heff.shape[0]):
        raise ValueError(
            f"projection k_indices/Heff row mismatch: {len(indices)} != {int(heff.shape[0])}"
        )
    if any(index < 0 for index in indices):
        raise ValueError(f"projection k_indices must be non-negative, got {indices}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"projection k_indices must be unique, got {indices}")
    actual_hash = hash_array(np.asarray(indices, dtype=np.int64))
    if actual_hash != stored_hash:
        raise ValueError(
            f"KP projection artifacts {heff_path.parent} k_indices_hash mismatch: "
            f"{stored_hash} != {actual_hash}"
        )
    return indices


def load_projection_artifact_identity(
    project_dir: str | Path,
    *,
    verify_heff: bool = True,
) -> dict[str, Any]:
    project_dir = Path(project_dir)
    basis_path = project_dir / "basis.npz"
    wavefunctions_path = project_dir / "wavefunctions.npz"
    heff_path = project_dir / "heff.npy"
    for path in (basis_path, wavefunctions_path, heff_path):
        if not path.is_file():
            raise FileNotFoundError(f"KP projection artifact is missing: {path}")
    with np.load(basis_path, allow_pickle=True) as basis_payload:
        basis_identity = require_identity_fields(
            basis_payload,
            PROJECTION_ARTIFACT_IDENTITY_FIELDS,
            f"KP projection artifacts {project_dir}",
        )
        basis_kpoints_hash = (
            require_identity_fields(
                basis_payload,
                ("kpoints_hash",),
                f"KP projection artifacts {project_dir}",
            )["kpoints_hash"]
            if "kpoints_hash" in basis_payload.files
            else None
        )
        basis_hamiltonian_hash = (
            require_identity_fields(
                basis_payload,
                ("source_hamiltonian_hash",),
                f"KP projection artifacts {project_dir}",
            )["source_hamiltonian_hash"]
            if "source_hamiltonian_hash" in basis_payload.files
            else None
        )
    with np.load(wavefunctions_path, allow_pickle=False) as wavefunction_payload:
        wavefunction_identity = require_identity_fields(
            wavefunction_payload,
            PROJECTION_ARTIFACT_IDENTITY_FIELDS,
            f"KP projection artifacts {project_dir}",
        )
        wavefunction_kpoints_hash = (
            require_identity_fields(
                wavefunction_payload,
                ("kpoints_hash",),
                f"KP projection artifacts {project_dir}",
            )["kpoints_hash"]
            if "kpoints_hash" in wavefunction_payload.files
            else None
        )
        wavefunction_hamiltonian_hash = (
            require_identity_fields(
                wavefunction_payload,
                ("source_hamiltonian_hash",),
                f"KP projection artifacts {project_dir}",
            )["source_hamiltonian_hash"]
            if "source_hamiltonian_hash" in wavefunction_payload.files
            else None
        )
    require_matching_identity(
        basis_identity,
        wavefunction_identity,
        PROJECTION_ARTIFACT_IDENTITY_FIELDS,
        f"KP projection artifacts {project_dir}",
    )
    if (basis_hamiltonian_hash is None) != (wavefunction_hamiltonian_hash is None):
        raise KeyError(
            f"KP projection artifacts {project_dir} are missing matching "
            "source_hamiltonian_hash metadata"
        )
    if (
        basis_hamiltonian_hash is not None
        and wavefunction_hamiltonian_hash is not None
        and basis_hamiltonian_hash != wavefunction_hamiltonian_hash
    ):
        raise ValueError(
            f"KP projection artifacts {project_dir} identity mismatch "
            "(source_hamiltonian_hash: "
            f"{basis_hamiltonian_hash!r} != {wavefunction_hamiltonian_hash!r})"
        )
    if basis_hamiltonian_hash is not None:
        basis_identity["source_hamiltonian_hash"] = basis_hamiltonian_hash
    if basis_identity["identity_schema"] != IDENTITY_SCHEMA:
        raise ValueError(f"Unsupported KP projection identity schema: {basis_identity['identity_schema']!r}")
    if basis_identity["package_version"] != KP_VERSION:
        raise ValueError(
            f"KP projection package version {basis_identity['package_version']!r} does not match {KP_VERSION!r}"
        )
    if int(basis_identity["schema_version"]) != PROJECTION_BASIS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported KP projection basis schema version: {basis_identity['schema_version']!r}"
        )
    kpoints_path = project_dir / "kpoints.npy"
    has_kpoints_identity = basis_kpoints_hash is not None or wavefunction_kpoints_hash is not None
    if kpoints_path.is_file() or has_kpoints_identity:
        if not kpoints_path.is_file():
            raise FileNotFoundError(f"KP projection artifact is missing: {kpoints_path}")
        if basis_kpoints_hash is None or wavefunction_kpoints_hash is None:
            raise KeyError(
                f"KP projection artifacts {project_dir} are missing matching kpoints_hash metadata"
            )
        if basis_kpoints_hash != wavefunction_kpoints_hash:
            raise ValueError(
                f"KP projection artifacts {project_dir} identity mismatch "
                f"(kpoints_hash: {basis_kpoints_hash!r} != {wavefunction_kpoints_hash!r})"
            )
        kpoints = np.load(kpoints_path, mmap_mode="r", allow_pickle=False)
        if kpoints.ndim != 2 or kpoints.shape[1] != 2:
            raise ValueError(f"projection/kpoints.npy must have shape (Nk,2), got {kpoints.shape}")
        heff_rows = int(np.load(heff_path, mmap_mode="r", allow_pickle=False).shape[0])
        if int(kpoints.shape[0]) != heff_rows:
            raise ValueError(
                f"projection kpoints/Heff row mismatch: {int(kpoints.shape[0])} != {heff_rows}"
            )
        actual_kpoints_hash = hash_array(kpoints)
        if actual_kpoints_hash != basis_kpoints_hash:
            raise ValueError(
                f"KP projection artifacts {project_dir} kpoints_hash mismatch: "
                f"{basis_kpoints_hash} != {actual_kpoints_hash}"
            )
        basis_identity["kpoints_hash"] = basis_kpoints_hash
    if verify_heff:
        actual_heff_hash = hash_array(np.load(heff_path, mmap_mode="r", allow_pickle=False))
        if actual_heff_hash != basis_identity["heff_hash"]:
            raise ValueError(
                f"KP projection artifacts {project_dir} heff_hash mismatch: "
                f"{basis_identity['heff_hash']} != {actual_heff_hash}"
            )
        load_projection_k_indices(heff_path)
    return basis_identity


def _metadata_scalar(value: Any, *, field: str, context: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{context} identity field {field} must be scalar, got shape {value.shape}")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("identity mapping keys must be strings")
            output[key] = _canonical_value(item)
        return output
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot be used for artifact identity")
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported identity value type: {type(value).__name__}")
