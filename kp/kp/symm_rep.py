"""Post-process KP Heff symmetry actions into band-subspace reps."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config.case import normalize_case_config
from .io.tapw_loader import load_Q_sets
from .symmetry.exactify_representation import build_basis_labels
from .symmetry.geometry import bM_candidates_from_q_distances, canonical_bM_pair_from_candidates
from .symmetry.projection import _model_q_sets


_ANTIUNITARY_NAMES = {"T", "TR", "C2T"}
_METADATA_KEYS = {"__metadata_json__", "metadata_json"}


@dataclass(frozen=True)
class KpSymmetryOperation:
    name: str
    matrix: np.ndarray
    antiunitary: bool
    source: str
    k_map: dict[str, Any] | None = None
    validated_same_k_indices: frozenset[int] | None = None


@dataclass(frozen=True)
class KpSymmRepRequest:
    config_path: Path
    run_root: Path
    projection_dir: Path
    symmetry_dir: Path
    output_dir: Path
    heff_path: Path
    representations_path: Path
    fermi_energy: float
    valence_count: int
    conduction_count: int
    degeneracy_tol: float
    points: dict[str, tuple[float, float, float]]
    point_indices: dict[str, int]
    overwrite: bool


def _as_path(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _point_tuple(value: Any, *, label: str) -> tuple[float, float, float]:
    if isinstance(value, Mapping) and "coords" in value:
        value = value["coords"]
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
        raise ValueError(f"symm_rep.points.{label} must be a 2- or 3-entry fractional coordinate.")
    coords = tuple(float(item) for item in value)
    if len(coords) == 2:
        return coords[0], coords[1], 0.0
    return coords


def _normalize_label(label: str) -> str:
    text = str(label).strip()
    if text.lower() in {"g", "gamma", "Γ"}:
        return "Gamma"
    return text


def _parse_points(raw_points: Any, cfg: Mapping[str, Any]) -> dict[str, tuple[float, float, float]]:
    if raw_points is None:
        kpath = cfg.get("kpath", {}) if isinstance(cfg.get("kpath"), Mapping) else {}
        coordinates = kpath.get("coordinates", {})
        labels = kpath.get("labels", [])
        if not isinstance(coordinates, Mapping) or not coordinates:
            raise ValueError("symm_rep.points is required when kpath.coordinates is unavailable.")
        wanted = labels if labels else coordinates.keys()
        out: dict[str, tuple[float, float, float]] = {}
        for raw_label in wanted:
            label = _normalize_label(str(raw_label))
            coord = coordinates.get(raw_label)
            if coord is None and label == "Gamma":
                coord = coordinates.get("G")
            if coord is None:
                coord = coordinates.get(label)
            if coord is None:
                continue
            out.setdefault(label, _point_tuple(coord, label=label))
        if out:
            return out
        raise ValueError("Unable to infer symm_rep.points from kpath.coordinates.")
    if not isinstance(raw_points, Mapping) or not raw_points:
        raise ValueError("symm_rep.points must be a non-empty mapping like {Gamma: [0, 0]}.")
    return {_normalize_label(str(label)): _point_tuple(value, label=str(label)) for label, value in raw_points.items()}


def _explicit_point_index(raw_points: Any, label: str) -> int | None:
    if not isinstance(raw_points, Mapping):
        return None
    value = raw_points.get(label)
    if value is None and label == "Gamma":
        value = raw_points.get("G")
    if isinstance(value, Mapping) and value.get("index") is not None:
        return int(value["index"])
    return None


def _load_kpath_rows(config_path: Path, cfg: Mapping[str, Any]) -> np.ndarray | None:
    candidates: list[Path] = []
    material = cfg.get("material", {}) if isinstance(cfg.get("material"), Mapping) else {}
    hamk_file = material.get("hamk_file")
    if hamk_file not in (None, ""):
        hamk_path = _as_path(str(hamk_file), base_dir=config_path.parent)
        candidates.append(hamk_path.parent.parent / "KPATH.out")
        candidates.append(hamk_path.parent / "KPATH.out")
    candidates.append(config_path.parent.parent / "KPATH.out")
    candidates.append(config_path.parent / "KPATH.out")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        rows = np.loadtxt(candidate, dtype=float, comments="#")
        rows = np.atleast_2d(rows)
        if rows.shape[1] >= 2:
            return rows[:, : min(3, rows.shape[1])]
    return None


def _index_from_coords(kpath_rows: np.ndarray, coords: tuple[float, float, float], *, tol: float = 5.0e-5) -> int | None:
    rows = np.asarray(kpath_rows, dtype=float)
    target = np.asarray(coords[: rows.shape[1]], dtype=float)
    distances = np.linalg.norm(rows[:, : target.shape[0]] - target[None, :], axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) <= tol:
        return index
    return None


def _index_from_kpath_vertices(label: str, cfg: Mapping[str, Any], n_k: int) -> int | None:
    kpath = cfg.get("kpath", {}) if isinstance(cfg.get("kpath"), Mapping) else {}
    raw_labels = kpath.get("labels", [])
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)) or len(raw_labels) < 2:
        return None
    labels = [_normalize_label(str(item)) for item in raw_labels]
    try:
        vertex = labels.index(_normalize_label(label))
    except ValueError:
        return None
    segments = len(labels) - 1
    return int(round(vertex * (n_k - 1) / segments))


def _resolve_point_indices(
    *,
    config_path: Path,
    cfg: Mapping[str, Any],
    heff: np.ndarray,
    points: Mapping[str, tuple[float, float, float]],
    raw_points: Any,
) -> dict[str, int]:
    kpath_rows = _load_kpath_rows(config_path, cfg)
    indices: dict[str, int] = {}
    for label, coords in points.items():
        index = _explicit_point_index(raw_points, label)
        if index is None and kpath_rows is not None:
            index = _index_from_coords(kpath_rows, coords)
        if index is None:
            index = _index_from_kpath_vertices(label, cfg, int(heff.shape[0]))
        if index is None:
            raise ValueError(
                f"Cannot resolve heff k-index for point {label!r}; add symm_rep.points.{label}.index to the config."
            )
        if index < 0 or index >= int(heff.shape[0]):
            raise IndexError(f"Resolved point {label!r} to heff index {index}, outside 0..{heff.shape[0] - 1}.")
        indices[label] = int(index)
    return indices


def resolve_config_request(config_path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> KpSymmRepRequest:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = normalize_case_config(yaml.safe_load(handle), config_path=config_path)
    case = cfg.get("case", {})
    if not isinstance(case, Mapping):
        raise ValueError("KP symm-rep requires canonical case.profile/case.q_shell/case.output_root config.")
    output_root = _as_path(str(case["output_root"]), base_dir=config_path.parent)
    run_root = output_root / str(case["profile"]) / str(case["q_shell"])
    projection_dir = run_root / "projection"
    symmetry_dir = run_root / "symmetry"
    heff_path = projection_dir / "heff.npy"
    representations_path = symmetry_dir / "representations.npz"
    if not heff_path.is_file():
        raise FileNotFoundError(f"Missing projected Heff: {heff_path}\nRun `kp project -c {config_path}` first.")
    if not representations_path.is_file():
        raise FileNotFoundError(
            f"Missing KP symmetry representations: {representations_path}\nRun `kp symm -c {config_path}` first."
        )

    symm_rep_cfg = dict(cfg.get("symm_rep", {}) or {})
    overrides = dict(overrides or {})
    material = cfg.get("material", {}) if isinstance(cfg.get("material"), Mapping) else {}
    efermi = overrides.get("fermi_energy", symm_rep_cfg.get("efermi", material.get("efermi")))
    if efermi is None:
        raise ValueError("KP symm-rep requires material.efermi or symm_rep.efermi.")

    output_dir = Path(str(overrides.get("output_dir", symm_rep_cfg.get("output_dir", "symm_rep"))))
    if not output_dir.is_absolute():
        output_dir = run_root / output_dir
    heff = np.load(heff_path, mmap_mode="r")
    raw_points = symm_rep_cfg.get("points")
    points = _parse_points(raw_points, cfg)
    return KpSymmRepRequest(
        config_path=config_path,
        run_root=run_root,
        projection_dir=projection_dir,
        symmetry_dir=symmetry_dir,
        output_dir=output_dir,
        heff_path=heff_path,
        representations_path=representations_path,
        fermi_energy=float(efermi),
        valence_count=int(overrides.get("valence_count", symm_rep_cfg.get("valence_count", 20))),
        conduction_count=int(overrides.get("conduction_count", symm_rep_cfg.get("conduction_count", 20))),
        degeneracy_tol=float(overrides.get("degeneracy_tol", symm_rep_cfg.get("degeneracy_tol", 2.0e-3))),
        points=points,
        point_indices=_resolve_point_indices(
            config_path=config_path,
            cfg=cfg,
            heff=heff,
            points=points,
            raw_points=raw_points,
        ),
        overwrite=bool(overrides.get("overwrite", symm_rep_cfg.get("overwrite", True))),
    )


def _metadata_text(payload: np.lib.npyio.NpzFile) -> str | None:
    for key in _METADATA_KEYS:
        if key in payload.files:
            value = payload[key]
            if value.shape == ():
                return str(value.item())
            return str(value.tolist())
    return None


def _infer_antiunitary(name: str) -> bool:
    return name in _ANTIUNITARY_NAMES or str(name).endswith("T")


def _load_operations(path: Path) -> tuple[list[KpSymmetryOperation], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata: dict[str, Any] = {}
        text = _metadata_text(payload)
        if text:
            metadata = json.loads(text)
        records_by_key: dict[str, Mapping[str, Any]] = {}
        for item in metadata.get("operations", []) if isinstance(metadata.get("operations"), list) else []:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("operation") or item.get("matrix_array_key") or "").strip()
            key = str(item.get("matrix_array_key") or name).strip()
            if key:
                records_by_key[key] = item
        operations: list[KpSymmetryOperation] = []
        for key in payload.files:
            if key in _METADATA_KEYS:
                continue
            matrix = np.asarray(payload[key], dtype=np.complex128)
            record = records_by_key.get(key, {})
            name = str(record.get("name") or record.get("operation") or key).strip()
            antiunitary = bool(record.get("antiunitary", _infer_antiunitary(name)))
            matrix_kind = str(record.get("matrix_kind", ""))
            matrix_source = str(record.get("matrix_source", ""))
            action = record.get("model_action") or record.get("internal_resolved_action") or record.get("declared_model_action") or {}
            k_map = dict(action.get("k_map", {}) or {}) if isinstance(action, Mapping) else None
            validated_same_k_indices: set[int] | None = None
            pairs = record.get("pairs")
            is_point_independent_model_action = (
                matrix_kind == "continuum_internal_rep_exact" or matrix_source == "kp_symm_exactified_action"
            )
            if (
                not is_point_independent_model_action
                and isinstance(pairs, Sequence)
                and not isinstance(pairs, (str, bytes))
            ):
                validated_same_k_indices = set()
                for pair in pairs:
                    if not isinstance(pair, Mapping):
                        continue
                    source_index = pair.get("source_k_index")
                    target_index = pair.get("target_k_index")
                    if source_index is not None and target_index is not None and int(source_index) == int(target_index):
                        validated_same_k_indices.add(int(source_index))
            operations.append(
                KpSymmetryOperation(
                    name=name,
                    matrix=matrix,
                    antiunitary=antiunitary,
                    source=str(path),
                    k_map=k_map,
                    validated_same_k_indices=(
                        None if validated_same_k_indices is None else frozenset(validated_same_k_indices)
                    ),
                )
            )
    return sorted(operations, key=lambda item: item.name), metadata


def _reciprocal_basis_from_metadata(metadata: Mapping[str, Any]) -> np.ndarray | None:
    exact = metadata.get("kp_symm_exactification")
    if not isinstance(exact, Mapping):
        return None
    b1 = exact.get("bM1")
    b2 = exact.get("bM2")
    if b1 is None or b2 is None:
        return None
    basis = np.vstack([np.asarray(b1, dtype=float)[:2], np.asarray(b2, dtype=float)[:2]])
    if abs(float(np.linalg.det(basis))) < 1.0e-12:
        return None
    return basis


def _source_reciprocal_basis_from_request(request: KpSymmRepRequest) -> np.ndarray | None:
    with request.config_path.open("r", encoding="utf-8") as handle:
        cfg = normalize_case_config(yaml.safe_load(handle), config_path=request.config_path)
    material = cfg.get("material", {}) if isinstance(cfg.get("material"), Mapping) else {}
    qset1_file = material.get("qset1_file")
    qset2_file = material.get("qset2_file")
    if qset1_file in (None, "") or qset2_file in (None, ""):
        return None
    q1, q2 = load_Q_sets(
        str(_as_path(str(qset1_file), base_dir=request.config_path.parent)),
        str(_as_path(str(qset2_file), base_dir=request.config_path.parent)),
    )
    candidates = bM_candidates_from_q_distances(q1, q2)
    if not candidates:
        return None
    b1, b2 = canonical_bM_pair_from_candidates(candidates, angle_deg=60.0)
    return np.vstack([np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)])


def _point_to_model_linear_from_metadata(metadata: Mapping[str, Any]) -> np.ndarray | None:
    frame = metadata.get("frame")
    if not isinstance(frame, Mapping):
        return None
    transform = frame.get("k_transform")
    if not isinstance(transform, Mapping):
        return None
    matrix = transform.get("linear_matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape == (2, 2):
            return arr
    rotation_deg = transform.get("rotation_deg")
    if rotation_deg is None:
        return None
    angle = np.deg2rad(float(rotation_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s], [s, c]], dtype=float)


def _coords_to_model_fractional(
    coords: Sequence[float],
    *,
    model_reciprocal_basis: np.ndarray,
    source_reciprocal_basis: np.ndarray | None,
    point_to_model_linear: np.ndarray | None,
) -> np.ndarray:
    source_basis = (
        np.asarray(source_reciprocal_basis, dtype=float).reshape(2, 2)
        if source_reciprocal_basis is not None
        else np.asarray(model_reciprocal_basis, dtype=float).reshape(2, 2)
    )
    linear = (
        np.asarray(point_to_model_linear, dtype=float).reshape(2, 2)
        if point_to_model_linear is not None
        else np.eye(2, dtype=float)
    )
    coord2 = np.asarray(coords[:2], dtype=float)
    model_cart = linear @ (coord2 @ source_basis)
    return np.linalg.solve(np.asarray(model_reciprocal_basis, dtype=float).reshape(2, 2).T, model_cart)


def _linear_matrix_from_k_map(k_map: Mapping[str, Any]) -> np.ndarray | None:
    kind = str(k_map.get("type", "")).lower()
    if kind == "rotation":
        angle = np.deg2rad(float(k_map.get("angle_deg", 0.0)))
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.array([[c, -s], [s, c]], dtype=float)
    if kind == "negation":
        return -np.eye(2, dtype=float)
    if kind == "reflection":
        axis = np.deg2rad(float(k_map.get("axis_deg", 0.0)))
        c = float(np.cos(2.0 * axis))
        s = float(np.sin(2.0 * axis))
        return np.array([[c, s], [s, -c]], dtype=float)
    if kind == "identity":
        return np.eye(2, dtype=float)
    return None


def _operation_target_shift(
    operation: KpSymmetryOperation,
    coords: tuple[float, float, float],
    points: Mapping[str, tuple[float, float, float]],
    *,
    source_point: str,
    heff_index: int,
    reciprocal_basis: np.ndarray | None,
    source_reciprocal_basis: np.ndarray | None = None,
    point_to_model_linear: np.ndarray | None = None,
    tol: float = 5.0e-5,
) -> tuple[str, np.ndarray] | None:
    if operation.validated_same_k_indices is not None and int(heff_index) not in operation.validated_same_k_indices:
        return None
    if not operation.k_map:
        return source_point, np.zeros(2, dtype=int)
    linear = _linear_matrix_from_k_map(operation.k_map)
    if linear is None or reciprocal_basis is None:
        return source_point, np.zeros(2, dtype=int)
    source_frac = _coords_to_model_fractional(
        coords,
        model_reciprocal_basis=reciprocal_basis,
        source_reciprocal_basis=source_reciprocal_basis,
        point_to_model_linear=point_to_model_linear,
    )
    mapped_frac = np.linalg.solve(reciprocal_basis.T, linear @ (source_frac @ reciprocal_basis))
    best: tuple[float, str, np.ndarray] | None = None
    for label, target_coords in points.items():
        target_frac = _coords_to_model_fractional(
            target_coords,
            model_reciprocal_basis=reciprocal_basis,
            source_reciprocal_basis=source_reciprocal_basis,
            point_to_model_linear=point_to_model_linear,
        )
        lattice_delta = mapped_frac - target_frac
        shift_to_mapped = np.rint(lattice_delta).astype(int)
        residual = float(np.linalg.norm(lattice_delta - shift_to_mapped))
        if best is None or residual < best[0]:
            best = (residual, str(label), -shift_to_mapped)
    if best is None or best[0] > float(tol):
        return None
    return best[1], best[2]


def _load_model_basis_labels(request: KpSymmRepRequest, metadata: Mapping[str, Any]) -> list[Any] | None:
    exact = metadata.get("kp_symm_exactification")
    if not isinstance(exact, Mapping):
        return None
    sectors = exact.get("sectors")
    n_orb = exact.get("n_orb")
    b1 = exact.get("bM1")
    b2 = exact.get("bM2")
    if not isinstance(sectors, Sequence) or isinstance(sectors, (str, bytes)) or n_orb is None or b1 is None or b2 is None:
        return None
    with request.config_path.open("r", encoding="utf-8") as handle:
        cfg = normalize_case_config(yaml.safe_load(handle), config_path=request.config_path)
    material = cfg.get("material", {}) if isinstance(cfg.get("material"), Mapping) else {}
    qset1_file = material.get("qset1_file")
    qset2_file = material.get("qset2_file")
    if qset1_file in (None, "") or qset2_file in (None, ""):
        return None
    q1, q2 = load_Q_sets(
        str(_as_path(str(qset1_file), base_dir=request.config_path.parent)),
        str(_as_path(str(qset2_file), base_dir=request.config_path.parent)),
    )
    frame = metadata.get("frame", {}) if isinstance(metadata.get("frame"), Mapping) else {}
    k_transform = frame.get("k_transform", {}) if isinstance(frame.get("k_transform"), Mapping) else {}
    q_transform = frame.get("q_transform", {}) if isinstance(frame.get("q_transform"), Mapping) else {}
    rotation_deg = float(k_transform.get("rotation_deg", q_transform.get("rotation_deg", 0.0)))
    q_model1, q_model2 = _model_q_sets(q1, q2, rotation_deg=rotation_deg)
    exact_config = exact.get("config", {}) if isinstance(exact.get("config"), Mapping) else {}
    tol = float(exact_config.get("q_tol", 1.0e-6))
    q_offsets = {
        str(sector["name"]): np.asarray(sector["q_offset"], dtype=float)
        for sector in sectors
        if isinstance(sector, Mapping) and "name" in sector and "q_offset" in sector
    }
    return build_basis_labels(
        Q_set1=q_model1,
        Q_set2=q_model2,
        sectors=sectors,
        n_orb1=int(n_orb[0]),
        n_orb2=int(n_orb[1]),
        bM1=np.asarray(b1, dtype=float),
        bM2=np.asarray(b2, dtype=float),
        tol=tol,
        q_offsets=q_offsets,
    )


def _shift_matrix_from_labels(labels: Sequence[Any], shift: np.ndarray) -> np.ndarray:
    dim = len(labels)
    lookup = {
        (label.sector, int(label.q_integer[0]), int(label.q_integer[1]), int(label.orbital)): int(label.index)
        for label in labels
    }
    out = np.zeros((dim, dim), dtype=np.complex128)
    dn1, dn2 = int(shift[0]), int(shift[1])
    for label in labels:
        key = (
            label.sector,
            int(label.q_integer[0]) + dn1,
            int(label.q_integer[1]) + dn2,
            int(label.orbital),
        )
        target = lookup.get(key)
        if target is not None:
            out[int(target), int(label.index)] = 1.0
    return out


def _sewn_operation_matrix(
    operation: KpSymmetryOperation,
    shift: np.ndarray,
    *,
    model_basis_labels: Sequence[Any] | None,
) -> np.ndarray:
    if not np.any(np.asarray(shift, dtype=int)):
        return operation.matrix
    if model_basis_labels is None:
        raise ValueError(
            f"Operation {operation.name} requires reciprocal-shift sewing, but kp symm-rep cannot rebuild the "
            "model Q basis. Ensure the config has material.qset1_file/material.qset2_file and kp symm metadata "
            "contains kp_symm_exactification.sectors/n_orb/bM1/bM2."
        )
    if len(model_basis_labels) != operation.matrix.shape[0]:
        raise ValueError(
            f"Operation {operation.name} sewing basis has {len(model_basis_labels)} rows, "
            f"but matrix dimension is {operation.matrix.shape[0]}."
        )
    return _shift_matrix_from_labels(model_basis_labels, np.asarray(shift, dtype=int)) @ operation.matrix


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        for name in (
            "summary.md",
            "high_symmetry_wavefunctions.npz",
            "band_representations.npz",
            "characters.csv",
            "bands.csv",
        ):
            target = path / name
            if target.is_file() or target.is_symlink():
                target.unlink()
        stale_cache = path / "__pycache__"
        if stale_cache.is_dir():
            shutil.rmtree(stale_cache)
    path.mkdir(parents=True, exist_ok=True)


def _sort_eigensystem(ham: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    herm = 0.5 * (np.asarray(ham, dtype=np.complex128) + np.asarray(ham, dtype=np.complex128).conj().T)
    energies, vectors = np.linalg.eigh(herm)
    order = np.argsort(energies, kind="stable")
    return np.asarray(energies[order], dtype=float), np.asarray(vectors[:, order], dtype=np.complex128)


def _select_indices(
    energies: np.ndarray,
    fermi_energy: float,
    *,
    valence_count: int,
    conduction_count: int,
) -> dict[str, np.ndarray]:
    energies = np.asarray(energies, dtype=float)
    valence = [int(i) for i in np.where(energies < fermi_energy)[0]]
    conduction = [int(i) for i in np.where(energies >= fermi_energy)[0]]
    valence = sorted(valence[-valence_count:], key=lambda i: (-energies[i], i)) if valence_count > 0 else []
    conduction = sorted(conduction[:conduction_count], key=lambda i: (energies[i], i)) if conduction_count > 0 else []
    return {
        "valence": np.asarray(valence, dtype=int),
        "conduction": np.asarray(conduction, dtype=int),
    }


def _group_degenerate(indices: np.ndarray, energies: np.ndarray, *, tol: float) -> list[np.ndarray]:
    selected = [int(i) for i in np.asarray(indices, dtype=int)]
    if not selected:
        return []
    blocks: list[list[int]] = [[selected[0]]]
    for index in selected[1:]:
        reference = blocks[-1][-1]
        if abs(float(energies[index]) - float(energies[reference])) <= tol:
            blocks[-1].append(index)
        else:
            blocks.append([index])
    return [np.asarray(block, dtype=int) for block in blocks]


def _orthonormalize(vectors: np.ndarray, *, tol: float = 1.0e-10) -> np.ndarray:
    basis = np.asarray(vectors, dtype=np.complex128)
    if basis.ndim != 2 or basis.shape[1] == 0:
        return basis
    gram = basis.conj().T @ basis
    if np.linalg.norm(gram - np.eye(gram.shape[0])) <= tol * max(1, gram.shape[0]):
        return basis
    gram = 0.5 * (gram + gram.conj().T)
    values, rotation = np.linalg.eigh(gram)
    floor = max(tol, tol * float(np.max(np.abs(values))) if values.size else tol)
    if np.any(values <= floor):
        raise ValueError(f"Degenerate block basis is linearly dependent; min Gram eigenvalue={np.min(values):.3e}.")
    return basis @ (rotation @ np.diag(values**-0.5) @ rotation.conj().T)


def _project_operation(matrix: np.ndarray, vectors: np.ndarray, *, antiunitary: bool) -> np.ndarray:
    basis = _orthonormalize(vectors)
    rhs = basis.conj() if antiunitary else basis
    return basis.conj().T @ np.asarray(matrix, dtype=np.complex128) @ rhs


def _polar_unitary(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.size == 0:
        return arr.copy(), 0.0
    x, _singular_values, yh = np.linalg.svd(arr, full_matrices=False)
    unitary = x @ yh
    denom = max(float(np.linalg.norm(unitary)), 1.0e-30)
    return unitary, float(np.linalg.norm(arr - unitary) / denom)


def _unitarity_residual(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.size == 0:
        return 0.0
    eye = np.eye(arr.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(arr.conj().T @ arr - eye) / max(1, arr.shape[0]))


def _format_complex(value: complex, *, digits: int = 4) -> str:
    value = complex(value)
    real = 0.0 if abs(value.real) < 10 ** (-digits) else value.real
    imag = 0.0 if abs(value.imag) < 10 ** (-digits) else value.imag
    if imag == 0.0:
        return f"{real:.{digits}f}"
    if real == 0.0:
        return f"{imag:.{digits}f}j"
    return f"{real:.{digits}f}{imag:+.{digits}f}j"


def _format_d_block(matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix, dtype=np.complex128):
        rows.append("[" + ", ".join(_format_complex(value) for value in row) + "]")
    return "[" + "; ".join(rows) + "]"


def _phase_label(value: complex, *, tol: float = 0.15) -> str:
    value = complex(value)
    magnitude = abs(value)
    unit = value / magnitude if magnitude > 0.0 else value
    roots = [
        (1.0 + 0.0j, "1"),
        (-1.0 + 0.0j, "-1"),
        (1.0j, "i"),
        (-1.0j, "-i"),
        (np.exp(1.0j * np.pi / 3.0), "ω"),
        (np.exp(2.0j * np.pi / 3.0), "ω²"),
        (np.exp(-2.0j * np.pi / 3.0), "ω⁴"),
        (np.exp(-1.0j * np.pi / 3.0), "ω⁵"),
    ]
    target, label = min(roots, key=lambda item: abs(unit - item[0]))
    if abs(unit - target) <= tol and abs(magnitude - 1.0) <= 0.25:
        return label
    angle = np.angle(value) / np.pi
    return f"{magnitude:.3g}e^{{{angle:+.3g}iπ}}"


def _symm_rep_label(matrix: np.ndarray, *, antiunitary: bool) -> str:
    if antiunitary:
        return ""
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.shape == (1, 1):
        return f"({_phase_label(arr[0, 0])})"
    values = np.linalg.eigvals(arr)
    order = np.argsort(np.angle(values))
    return "(" + ", ".join(_phase_label(value) for value in values[order]) + ")"


def _energy_text(energies: Sequence[float], *, digits: int = 12) -> str:
    return ", ".join(f"{float(value):.{digits}f}" for value in energies)


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def _load_spin_operator(path: Path, n_k: int, dim: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.asarray(np.load(path), dtype=np.complex128)
    if arr.shape == (dim, dim):
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3 or arr.shape[1:] != (dim, dim):
        raise ValueError(f"spin_operator.npy has shape {arr.shape}, expected ({n_k}, {dim}, {dim}) or ({dim}, {dim}).")
    if arr.shape[0] not in {1, n_k}:
        raise ValueError(f"spin_operator.npy has {arr.shape[0]} k rows, expected 1 or {n_k}.")
    return arr


def _spin_operator_for_index(spin_operator: np.ndarray | None, index: int) -> np.ndarray | None:
    if spin_operator is None:
        return None
    if spin_operator.shape[0] == 1:
        return np.asarray(spin_operator[0], dtype=np.complex128)
    return np.asarray(spin_operator[int(index)], dtype=np.complex128)


def _spin_weights_from_expectation(value: float, *, threshold: float = 0.8) -> tuple[float, float, str]:
    sz = max(-1.0, min(1.0, float(value)))
    up = 0.5 * (1.0 + sz)
    down = 0.5 * (1.0 - sz)
    if up >= threshold:
        label = "up"
    elif down >= threshold:
        label = "down"
    else:
        label = f"↑{up:.3f},↓{down:.3f}"
    return float(up), float(down), label


def _spin_diagonalize_block(
    vectors: np.ndarray,
    spin_operator: np.ndarray | None,
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    basis = _orthonormalize(vectors)
    if spin_operator is None:
        return basis, [(float("nan"), float("nan"), "effective") for _ in range(basis.shape[1])]
    sz = np.asarray(spin_operator, dtype=np.complex128)
    projected = basis.conj().T @ sz @ basis
    projected = 0.5 * (projected + projected.conj().T)
    if basis.shape[1] > 1:
        values, rotation = np.linalg.eigh(projected)
        order = np.argsort(values)[::-1]
        values = values[order]
        basis = basis @ rotation[:, order]
    else:
        values = np.asarray([projected[0, 0].real], dtype=float)
    return basis, [_spin_weights_from_expectation(float(value.real)) for value in values]


def _write_bands_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["point", "sector", "band_index", "energy", "spin_up", "spin_down", "spin_label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_characters_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "point",
        "sector",
        "block_id",
        "band_indices",
        "energies",
        "operation",
        "antiunitary",
        "symm_rep",
        "trace_real",
        "trace_imag",
        "unitarity_residual",
        "raw_unitarity_residual",
        "polar_distance",
        "d_block",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_summary(
    path: Path,
    *,
    request: KpSymmRepRequest,
    band_rows: Sequence[Mapping[str, Any]],
    character_rows: Sequence[Mapping[str, Any]],
    operations: Sequence[KpSymmetryOperation],
    point_operations: Mapping[str, Sequence[str]],
) -> None:
    rows_by_point: dict[str, list[Mapping[str, Any]]] = {}
    chars_by_point: dict[str, list[Mapping[str, Any]]] = {}
    for row in band_rows:
        rows_by_point.setdefault(str(row["point"]), []).append(row)
    for row in character_rows:
        chars_by_point.setdefault(str(row["point"]), []).append(row)

    lines = [
        "# KP Symmetry Representation Summary",
        "",
        f"- config: `{request.config_path}`",
        f"- heff: `{request.heff_path}`",
        f"- symmetry: `{request.representations_path}`",
        f"- efermi: `{request.fermi_energy:.12f}`",
        f"- operations: {', '.join(op.name for op in operations) if operations else 'none'}",
        "",
    ]
    for point in request.points:
        lines.extend(
            [
                f"## {point}",
                "",
                f"- fractional coordinate: `{request.points[point]}`",
                f"- heff row: `{request.point_indices[point]}`",
                f"- little-group operations: {', '.join(point_operations.get(point, [])) or 'none'}",
                "",
            ]
        )
        for sector, title in (("valence", "Valence"), ("conduction", "Conduction")):
            sector_rows = [row for row in rows_by_point.get(point, []) if row["sector"] == sector]
            lines.extend([f"### {title}", "", "| band | energy (eV) | spin |", "|---:|---:|---|"])
            for row in sector_rows:
                lines.append(f"| {row['band_index']} | {float(row['energy']):.6f} | {row.get('spin_label', '')} |")
            if not sector_rows:
                lines.append("| - | - | - |")
            lines.append("")
        lines.extend(
            [
                "### Symmetry Representations",
                "",
                "| sector | block | bands | energies (eV) | operation | rep | trace | residual | D_block |",
                "|---|---:|---|---|---|---|---|---:|---|",
            ]
        )
        for row in chars_by_point.get(point, []):
            trace = f"{float(row['trace_real']):.6f}{float(row['trace_imag']):+.6f}i"
            energies = ", ".join(f"{float(item.strip()):.6f}" for item in str(row["energies"]).split(",") if item.strip())
            lines.append(
                f"| {row['sector']} | {row['block_id']} | {row['band_indices']} | {energies} | "
                f"{row['operation']} | {row.get('symm_rep', '')} | {trace} | "
                f"{float(row['unitarity_residual']):.3e} | `{row.get('d_block', '')}` |"
            )
        if not chars_by_point.get(point):
            lines.append("| - | - | - | - | - | - | - | - | - |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_configured_symm_rep(config_path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> Path:
    t0 = time.perf_counter()
    request = resolve_config_request(config_path, overrides=overrides)
    _prepare_output_dir(request.output_dir, overwrite=request.overwrite)
    heff = np.load(request.heff_path, mmap_mode="r")
    operations, metadata = _load_operations(request.representations_path)
    reciprocal_basis = _reciprocal_basis_from_metadata(metadata)
    source_reciprocal_basis = _source_reciprocal_basis_from_request(request)
    point_to_model_linear = _point_to_model_linear_from_metadata(metadata)
    model_basis_labels = _load_model_basis_labels(request, metadata)
    spin_operator_stack = _load_spin_operator(
        request.projection_dir / "spin_operator.npy",
        int(heff.shape[0]),
        int(heff.shape[1]),
    )

    band_rows: list[dict[str, Any]] = []
    character_rows: list[dict[str, Any]] = []
    point_operations: dict[str, list[str]] = {}
    wave_payload: dict[str, Any] = {
        "__metadata_json__": np.asarray(
            json.dumps(
                {
                    "schema": "kp.symm_rep.high_symmetry_wavefunctions.v1",
                    "config": str(request.config_path),
                    "fermi_energy": request.fermi_energy,
                    "point_indices": request.point_indices,
                },
                sort_keys=True,
            )
        ),
        "points": np.asarray(list(request.points), dtype="U64"),
    }
    rep_payload: dict[str, Any] = {
        "__metadata_json__": np.asarray(
            json.dumps(
                {
                    "schema": "kp.symm_rep.band_representations.v1",
                    "config": str(request.config_path),
                    "symmetry_metadata": metadata,
                },
                sort_keys=True,
            )
        )
    }

    for point, coords in request.points.items():
        index = request.point_indices[point]
        energies, vectors = _sort_eigensystem(np.asarray(heff[index]))
        point_spin_operator = _spin_operator_for_index(spin_operator_stack, index)
        active_operations: list[tuple[KpSymmetryOperation, np.ndarray]] = []
        for operation in operations:
            target = _operation_target_shift(
                operation,
                coords,
                request.points,
                source_point=point,
                heff_index=index,
                reciprocal_basis=reciprocal_basis,
                source_reciprocal_basis=source_reciprocal_basis,
                point_to_model_linear=point_to_model_linear,
            )
            if target is None:
                continue
            target_point, shift = target
            if target_point != point:
                continue
            active_operations.append(
                (
                    operation,
                    _sewn_operation_matrix(operation, shift, model_basis_labels=model_basis_labels),
                )
            )
        point_operations[point] = [operation.name for operation, _matrix in active_operations]
        selections = _select_indices(
            energies,
            request.fermi_energy,
            valence_count=request.valence_count,
            conduction_count=request.conduction_count,
        )
        selected = np.unique(np.concatenate([indices for indices in selections.values() if len(indices)]))
        wave_payload[f"{_safe_key(point)}__coords"] = np.asarray(coords, dtype=float)
        wave_payload[f"{_safe_key(point)}__heff_index"] = np.asarray(index, dtype=np.int64)
        wave_payload[f"{_safe_key(point)}__energies"] = energies
        wave_payload[f"{_safe_key(point)}__eigenvectors"] = vectors
        wave_payload[f"{_safe_key(point)}__selected_indices"] = selected.astype(np.int64)

        for sector in ("valence", "conduction"):
            blocks = _group_degenerate(selections[sector], energies, tol=request.degeneracy_tol)
            for block_id, block in enumerate(blocks):
                block_vectors, block_spin = _spin_diagonalize_block(vectors[:, block], point_spin_operator)
                block_energies = [float(energies[i]) for i in block]
                for band_index, (spin_up, spin_down, spin_label) in zip(block, block_spin):
                    band_rows.append(
                        {
                            "point": point,
                            "sector": sector,
                            "band_index": int(band_index),
                            "energy": f"{float(energies[band_index]):.12f}",
                            "spin_up": "" if np.isnan(spin_up) else f"{spin_up:.12f}",
                            "spin_down": "" if np.isnan(spin_down) else f"{spin_down:.12f}",
                            "spin_label": spin_label,
                        }
                    )
                for operation, operation_matrix in active_operations:
                    if operation_matrix.shape != (vectors.shape[0], vectors.shape[0]):
                        raise ValueError(
                            f"KP symmetry matrix {operation.name} has shape {operation_matrix.shape}, "
                            f"but Heff eigenvectors at {point} have dimension {vectors.shape[0]}."
                        )
                    raw_projected = _project_operation(
                        operation_matrix,
                        block_vectors,
                        antiunitary=operation.antiunitary,
                    )
                    projected, polar_distance = _polar_unitary(raw_projected)
                    trace = np.trace(projected)
                    key = "__".join(
                        [
                            _safe_key(point),
                            sector,
                            f"block{block_id}",
                            _safe_key(operation.name),
                        ]
                    )
                    rep_payload[key] = projected
                    rep_payload[f"{key}__raw_projected"] = raw_projected
                    character_rows.append(
                        {
                            "point": point,
                            "sector": sector,
                            "block_id": block_id,
                            "band_indices": " ".join(str(int(i)) for i in block),
                            "energies": _energy_text(block_energies),
                            "operation": operation.name,
                            "antiunitary": str(bool(operation.antiunitary)).lower(),
                            "symm_rep": _symm_rep_label(projected, antiunitary=operation.antiunitary),
                            "trace_real": f"{float(trace.real):.12f}",
                            "trace_imag": f"{float(trace.imag):.12f}",
                            "unitarity_residual": f"{_unitarity_residual(projected):.12e}",
                            "raw_unitarity_residual": f"{_unitarity_residual(raw_projected):.12e}",
                            "polar_distance": f"{polar_distance:.12e}",
                            "d_block": _format_d_block(projected),
                        }
                    )

    np.savez_compressed(request.output_dir / "high_symmetry_wavefunctions.npz", **wave_payload)
    np.savez_compressed(request.output_dir / "band_representations.npz", **rep_payload)
    _write_bands_csv(request.output_dir / "bands.csv", band_rows)
    _write_characters_csv(request.output_dir / "characters.csv", character_rows)
    _write_summary(
        request.output_dir / "summary.md",
        request=request,
        band_rows=band_rows,
        character_rows=character_rows,
        operations=operations,
        point_operations=point_operations,
    )
    elapsed = time.perf_counter() - t0
    print("=" * 72)
    print("[kp symm-rep] Complete")
    print("-" * 72)
    print(f"  output directory : {request.output_dir}")
    print(f"  summary          : {request.output_dir / 'summary.md'}")
    print(f"  band reps        : {request.output_dir / 'band_representations.npz'}")
    print(f"  characters       : {request.output_dir / 'characters.csv'}")
    print(f"  points           : {', '.join(f'{p}[{request.point_indices[p]}]' for p in request.points)}")
    print(f"  operations       : {', '.join(op.name for op in operations) if operations else 'none'}")
    print(f"  total time       : {elapsed:.2f}s")
    return request.output_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kp symm-rep", description="Compute symmetry reps of projected Heff bands")
    parser.add_argument("-c", "--config", required=True, help="YAML KP case config path")
    parser.add_argument("--valence-count", type=int)
    parser.add_argument("--conduction-count", type=int)
    parser.add_argument("--degeneracy-tol", type=float)
    parser.add_argument("--fermi-energy", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args(argv)
    overrides = {
        "valence_count": args.valence_count,
        "conduction_count": args.conduction_count,
        "degeneracy_tol": args.degeneracy_tol,
        "fermi_energy": args.fermi_energy,
        "output_dir": args.output_dir,
        "overwrite": False if args.no_overwrite else None,
    }
    run_configured_symm_rep(args.config, overrides={key: value for key, value in overrides.items() if value is not None})
