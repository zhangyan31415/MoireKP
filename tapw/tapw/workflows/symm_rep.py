"""Post-process TAPW raw-H symmetry actions into band-subspace reps."""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse

from ..chern_post import build_boundary_sewing


_HAMK_RE = re.compile(r"^hamk_(?P<point>.+)_valley\.npy$")
_SAVED_BAND_RE = re.compile(r"^band_(?P<edge>VBM|CBM)_(?P<point>.+)_valley\.txt$")
_RAW_H_SUFFIX = "_rawH.npz"
_ANTIUNITARY_NAMES = {"T", "TR", "C2T"}


@dataclass(frozen=True)
class RawHSymmetryOperation:
    point: str
    operation: str
    matrix: scipy.sparse.spmatrix
    antiunitary: bool
    source: str
    source_action: dict[str, Any] | None = None


@dataclass(frozen=True)
class EnergyBlock:
    block_id: int
    sector: str
    indices: tuple[int, ...]
    energies: tuple[float, ...]
    band_indices: tuple[int, ...]


@dataclass(frozen=True)
class SymmRepResult:
    output_dir: Path
    points: tuple[str, ...]
    fermi_energy: float


def spin_weights_and_label(vector: np.ndarray, *, threshold: float = 0.8) -> tuple[float | None, float | None, str]:
    """Return spin-up/down half weights and a coarse state label."""
    vector = np.asarray(vector)
    if vector.ndim != 1 or vector.shape[0] % 2 != 0:
        return None, None, "spinless"
    norm = float(np.vdot(vector, vector).real)
    if norm <= 0.0:
        return 0.0, 0.0, "mixed"
    half = vector.shape[0] // 2
    up = float(np.vdot(vector[:half], vector[:half]).real / norm)
    down = float(np.vdot(vector[half:], vector[half:]).real / norm)
    if up >= threshold:
        label = "up"
    elif down >= threshold:
        label = "down"
    else:
        label = "mixed"
    return up, down, label


def _spin_weight_label(vector: np.ndarray) -> str:
    up, down, _label = spin_weights_and_label(vector)
    if up is None or down is None:
        return "[spinless]"
    return f"[↑{up:.3f},↓{down:.3f}]"


def _spin_weight_text(up: Any, down: Any) -> str:
    try:
        up_value = float(up)
        down_value = float(down)
    except (TypeError, ValueError):
        return "spinless"
    if np.isnan(up_value) or np.isnan(down_value):
        return "spinless"
    return f"↑{up_value:.3f}, ↓{down_value:.3f}"


def _phase_label(value: complex, *, tol: float = 0.15) -> str:
    value = complex(value)
    magnitude = abs(value)
    if magnitude > 0.0:
        unit = value / magnitude
    else:
        unit = value
    roots = [
        (1.0 + 0.0j, "1"),
        (-1.0 + 0.0j, "-1"),
        (1.0j, "i"),
        (-1.0j, "-i"),
        (np.exp(2.0j * np.pi / 3.0), "ω"),
        (np.exp(-2.0j * np.pi / 3.0), "ω²"),
        (np.exp(1.0j * np.pi / 3.0), "e^{+iπ/3}"),
        (np.exp(-1.0j * np.pi / 3.0), "e^{-iπ/3}"),
    ]
    target, label = min(roots, key=lambda item: abs(unit - item[0]))
    if abs(unit - target) <= tol and abs(magnitude - 1.0) <= 0.25:
        return label
    angle = np.angle(value) / np.pi
    return f"{magnitude:.3g}e^{{{angle:+.3g}iπ}}"


def symmetry_rep_label(projected: np.ndarray, block_vectors: np.ndarray, *, antiunitary: bool = False) -> str:
    if antiunitary:
        return ""
    matrix = np.asarray(projected, dtype=np.complex128)
    vectors = np.asarray(block_vectors, dtype=np.complex128)
    if matrix.size == 0:
        return "()"
    if matrix.shape == (1, 1):
        symm_vectors = vectors
        values = np.array([matrix[0, 0]], dtype=np.complex128)
    else:
        values, coeffs = np.linalg.eig(matrix)
        order = np.argsort(np.angle(values))
        values = values[order]
        coeffs = coeffs[:, order]
        symm_vectors = vectors @ coeffs
    labels = []
    for value, vector in zip(values, symm_vectors.T):
        suffix = _spin_weight_label(vector)
        labels.append(f"{_phase_label(value)}{suffix}")
    return "(" + ", ".join(labels) + ")"


def _format_complex_entry(value: complex, *, digits: int = 4) -> str:
    value = complex(value)
    real = 0.0 if abs(value.real) < 10 ** (-digits) else value.real
    imag = 0.0 if abs(value.imag) < 10 ** (-digits) else value.imag
    if imag == 0.0:
        return f"{real:.{digits}f}"
    if real == 0.0:
        return f"{imag:.{digits}f}j"
    return f"{real:.{digits}f}{imag:+.{digits}f}j"


def _format_d_block(matrix: np.ndarray) -> str:
    arr = np.asarray(matrix, dtype=np.complex128)
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(_format_complex_entry(value) for value in row) + "]")
    return "[" + "; ".join(rows) + "]"


def _parse_energies_text(text: str) -> list[float]:
    try:
        return [float(item.strip()) for item in str(text).split(",") if item.strip()]
    except ValueError:
        return []


def _format_summary_energy(value: Any) -> str:
    return f"{float(value):.6f}"


def _format_summary_energies(text: str) -> str:
    energies = _parse_energies_text(text)
    if not energies:
        return str(text)
    return ", ".join(f"{energy:.6f}" for energy in energies)


def _symm_row_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str]:
    sector = str(row.get("sector", ""))
    energies = _parse_energies_text(str(row.get("energies", "")))
    if sector == "valence":
        sector_order = 0
        energy_key = -max(energies) if energies else float("inf")
    elif sector == "conduction":
        sector_order = 1
        energy_key = min(energies) if energies else float("inf")
    else:
        sector_order = 2
        energy_key = min(energies) if energies else float("inf")
    return (sector_order, energy_key, int(row.get("block_id", 0)), str(row.get("operation", "")))


def project_operation_to_subspace(raw_action: Any, vectors: np.ndarray, *, antiunitary: bool = False) -> np.ndarray:
    """Project a raw-H action into the column span of ``vectors``."""
    matrix = raw_action.toarray() if scipy.sparse.issparse(raw_action) else np.asarray(raw_action)
    basis = np.asarray(vectors, dtype=np.complex128)
    rhs = basis.conj() if antiunitary else basis
    return basis.conj().T @ matrix @ rhs


def _k_map_linear_matrix(k_map: dict[str, Any]) -> np.ndarray:
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
        c = float(np.cos(axis))
        s = float(np.sin(axis))
        direction = np.array([c, s], dtype=float)
        return 2.0 * np.outer(direction, direction) - np.eye(2, dtype=float)
    raise ValueError(f"Unsupported source_action k_map type: {k_map.get('type')!r}")


def _reciprocal_shift_for_closed_action(
    operation: RawHSymmetryOperation,
    coords: np.ndarray,
    reciprocal_basis: np.ndarray,
    *,
    atol: float = 1.0e-6,
) -> np.ndarray | None:
    if not operation.source_action:
        return np.zeros(2, dtype=int)
    k_map = dict(operation.source_action.get("k_map", {}) or {})
    linear = _k_map_linear_matrix(k_map)
    basis = np.asarray(reciprocal_basis, dtype=float).reshape(2, 2)
    coord2 = np.asarray(coords, dtype=float).reshape(-1)[:2]
    k_cart = coord2 @ basis
    mapped = linear @ k_cart
    delta = mapped - k_cart
    coeffs = np.linalg.solve(basis.T, delta)
    rounded = np.rint(coeffs).astype(int)
    residual = np.linalg.norm(delta - rounded @ basis)
    if float(residual) > float(atol):
        return None
    return rounded


def _sewing_matrix_from_shift(
    *,
    g_vectors_by_group: list[np.ndarray],
    reciprocal_basis: np.ndarray,
    reciprocal_shift_coeffs: np.ndarray,
    dim: int,
    spin_blocks: int,
    atol: float = 1.0e-6,
) -> scipy.sparse.csr_matrix:
    shift_cart = np.asarray(reciprocal_shift_coeffs, dtype=float).reshape(2) @ np.asarray(reciprocal_basis, dtype=float).reshape(2, 2)
    if float(np.linalg.norm(shift_cart)) <= float(atol):
        return scipy.sparse.identity(int(dim), dtype=np.complex128, format="csr")
    sewing = build_boundary_sewing(
        g_vectors_by_group,
        shift_cart,
        dim_h=int(dim),
        atol=atol,
        spin_blocks=int(spin_blocks),
    )
    if sewing.missing_blocks:
        raise ValueError(
            "Cannot build complete reciprocal sewing matrix: "
            f"matched={sewing.matched_blocks}, missing={sewing.missing_blocks}, shift={reciprocal_shift_coeffs.tolist()}."
        )
    data = np.ones_like(sewing.target_rows, dtype=np.complex128)
    return scipy.sparse.csr_matrix((data, (sewing.target_rows, sewing.source_rows)), shape=(int(dim), int(dim)))


def _sewn_raw_action(
    operation: RawHSymmetryOperation,
    point_source: dict[str, Any],
    dim: int,
) -> tuple[Any, str] | None:
    sewing_context = dict(point_source.get("sewing_context", {}) or {})
    if not sewing_context:
        return operation.matrix, ""
    coords = np.asarray(point_source.get("coords", (0.0, 0.0, 0.0)), dtype=float)
    reciprocal_basis = np.asarray(sewing_context["reciprocal_basis"], dtype=float)
    shift = _reciprocal_shift_for_closed_action(operation, coords, reciprocal_basis)
    if shift is None:
        return None
    sewing = _sewing_matrix_from_shift(
        g_vectors_by_group=[np.asarray(item, dtype=float) for item in sewing_context["g_vectors_by_group"]],
        reciprocal_basis=reciprocal_basis,
        reciprocal_shift_coeffs=shift,
        dim=int(dim),
        spin_blocks=int(sewing_context.get("spin_blocks", 1)),
    )
    raw = operation.matrix.tocsr() if scipy.sparse.issparse(operation.matrix) else scipy.sparse.csr_matrix(operation.matrix)
    return (sewing @ raw).tocsr(), f"{int(shift[0])} {int(shift[1])}"


def _safe_npz_key(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_") or "item"


def _infer_antiunitary(operation: str) -> bool:
    return operation in _ANTIUNITARY_NAMES or operation.endswith("T")


def _csr_from_packed(payload: np.lib.npyio.NpzFile, key: str) -> scipy.sparse.csr_matrix:
    required = [f"{key}_data", f"{key}_indices", f"{key}_indptr", f"{key}_shape"]
    missing = [name for name in required if name not in payload.files]
    if missing:
        raise ValueError(f"Packed CSR matrix {key!r} is missing entries: {missing}")
    shape = tuple(int(v) for v in payload[f"{key}_shape"].tolist())
    return scipy.sparse.csr_matrix(
        (payload[f"{key}_data"], payload[f"{key}_indices"], payload[f"{key}_indptr"]),
        shape=shape,
    )


def discover_high_symmetry_hamiltonians(band_dir: Path) -> dict[str, Path]:
    band_dir = Path(band_dir)
    hamiltonians: dict[str, Path] = {}
    for path in sorted(band_dir.glob("hamk_*_valley.npy")):
        match = _HAMK_RE.match(path.name)
        if match:
            hamiltonians[match.group("point")] = path
    if not hamiltonians and (band_dir / "hamiltonian_k.npy").is_file():
        hamiltonians["point"] = band_dir / "hamiltonian_k.npy"
    return hamiltonians


def discover_high_symmetry_sources(band_dir: Path) -> dict[str, dict[str, Any]]:
    band_dir = Path(band_dir)
    sources: dict[str, dict[str, Any]] = {
        point: {"kind": "hamiltonian", "hamiltonian": path}
        for point, path in discover_high_symmetry_hamiltonians(band_dir).items()
    }
    saved: dict[str, dict[str, Path]] = {}
    for energy_path in sorted(band_dir.glob("band_*_*_valley.txt")):
        match = _SAVED_BAND_RE.match(energy_path.name)
        if not match:
            continue
        edge = match.group("edge")
        point = match.group("point")
        vec_path = band_dir / f"vec_{edge}_{point}_valley.npy"
        if vec_path.is_file():
            saved.setdefault(point, {})[edge] = energy_path
            saved[point][f"vec_{edge}"] = vec_path
    for point, files in saved.items():
        if point in sources:
            continue
        if "VBM" in files or "CBM" in files:
            sources[point] = {"kind": "saved_wavefunctions", **files}
    return sources


def _load_operations_from_manifest(
    symmetry_dir: Path,
    point_labels: set[str],
) -> list[RawHSymmetryOperation]:
    manifest_path = symmetry_dir / "representations" / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("matrices", [])
    operations: list[RawHSymmetryOperation] = []
    packed_cache: dict[Path, np.lib.npyio.NpzFile] = {}
    try:
        for record in records:
            operation = str(record.get("operation", "")).strip()
            if not operation:
                continue
            point = str(
                record.get("target_valley")
                or record.get("valley_label")
                or record.get("source_valley")
                or ""
            ).strip()
            source_action = dict(record.get("source_action", {}) or {})
            if source_action:
                target_points = sorted(point_labels)
            else:
                if point and point not in point_labels:
                    continue
                if not point and len(point_labels) == 1:
                    point = next(iter(point_labels))
                if not point:
                    continue
                target_points = [point]

            matrix: scipy.sparse.csr_matrix | None = None
            raw_file = record.get("raw_h_operator_file")
            if raw_file:
                raw_path = symmetry_dir / "representations" / str(raw_file)
                if raw_path.is_file():
                    matrix = scipy.sparse.load_npz(raw_path).tocsr()
            if matrix is None and record.get("packed_matrix_key"):
                packed_file = record.get("packed_matrix_file") or "../representations.npz"
                packed_path = (symmetry_dir / "representations" / str(packed_file)).resolve()
                if packed_path.is_file():
                    if packed_path not in packed_cache:
                        packed_cache[packed_path] = np.load(packed_path, allow_pickle=False)
                    matrix = _csr_from_packed(packed_cache[packed_path], str(record["packed_matrix_key"]))
            if matrix is None:
                continue
            for target_point in target_points:
                operations.append(
                    RawHSymmetryOperation(
                        point=target_point,
                        operation=operation,
                        matrix=matrix,
                        antiunitary=bool(record.get("antiunitary", _infer_antiunitary(operation))),
                        source="manifest",
                        source_action=source_action or None,
                    )
                )
    finally:
        for payload in packed_cache.values():
            payload.close()
    return operations


def _load_legacy_operations(symmetry_dir: Path, point_labels: set[str]) -> list[RawHSymmetryOperation]:
    representations_dir = symmetry_dir / "representations"
    operations: list[RawHSymmetryOperation] = []
    if not representations_dir.is_dir():
        return operations
    for point_dir in sorted(path for path in representations_dir.iterdir() if path.is_dir()):
        point = point_dir.name
        if point not in point_labels:
            continue
        for matrix_path in sorted(point_dir.glob(f"*{_RAW_H_SUFFIX}")):
            operation = matrix_path.name[: -len(_RAW_H_SUFFIX)]
            operations.append(
                RawHSymmetryOperation(
                    point=point,
                    operation=operation,
                    matrix=scipy.sparse.load_npz(matrix_path).tocsr(),
                    antiunitary=_infer_antiunitary(operation),
                    source=str(matrix_path),
                )
            )
    return operations


def _load_packed_operations(symmetry_dir: Path, point_labels: set[str]) -> list[RawHSymmetryOperation]:
    packed_path = symmetry_dir / "representations.npz"
    if not packed_path.is_file():
        return []
    operations: list[RawHSymmetryOperation] = []
    profile_point = ""
    try:
        q_name = symmetry_dir.parent.name
        if not re.match(r"^q\d+$", q_name):
            q_name = ""
        if q_name:
            profile_point = symmetry_dir.parent.parent.name
    except IndexError:
        profile_point = ""
    with np.load(packed_path, allow_pickle=False) as payload:
        bases = sorted(name[: -len("_data")] for name in payload.files if name.endswith("_data"))
        for base in bases:
            matrix = _csr_from_packed(payload, base)
            matched = False
            for point in sorted(point_labels):
                prefix = f"{_safe_npz_key(point)}_"
                if base.startswith(prefix):
                    operation = base[len(prefix) :]
                    operations.append(
                        RawHSymmetryOperation(point, operation, matrix, _infer_antiunitary(operation), str(packed_path))
                    )
                    matched = True
            if matched:
                continue
            if profile_point in point_labels:
                target_points = [profile_point]
            elif not profile_point and len(point_labels) == 1:
                target_points = sorted(point_labels)
            else:
                target_points = []
            for point in target_points:
                operations.append(
                    RawHSymmetryOperation(point, base, matrix, _infer_antiunitary(base), str(packed_path))
                )
    return operations


def load_raw_h_symmetry_operations(symmetry_dir: Path, point_labels: set[str]) -> dict[str, list[RawHSymmetryOperation]]:
    symmetry_dir = Path(symmetry_dir)
    operations = _load_operations_from_manifest(symmetry_dir, point_labels)
    seen = {(op.point, op.operation) for op in operations}
    for op in _load_legacy_operations(symmetry_dir, point_labels):
        if (op.point, op.operation) not in seen:
            operations.append(op)
            seen.add((op.point, op.operation))
    for op in _load_packed_operations(symmetry_dir, point_labels):
        if (op.point, op.operation) not in seen:
            operations.append(op)
            seen.add((op.point, op.operation))

    grouped: dict[str, list[RawHSymmetryOperation]] = {point: [] for point in point_labels}
    for op in sorted(operations, key=lambda item: (item.point, item.operation)):
        grouped.setdefault(op.point, []).append(op)
    return grouped


def _load_hamiltonian(path: Path, *, hamiltonian_index: int = 0) -> tuple[np.ndarray, int, int]:
    hamk = np.asarray(np.load(path), dtype=np.complex128)
    if hamk.ndim == 3:
        count = int(hamk.shape[0])
        index = int(hamiltonian_index)
        if index < 0:
            index += count
        if index < 0 or index >= count:
            raise IndexError(f"{path} contains {count} Hamiltonians; index {hamiltonian_index} is out of range.")
        hamk = hamk[index]
        selected_index = index
    else:
        count = 1
        selected_index = 0
    if hamk.ndim != 2 or hamk.shape[0] != hamk.shape[1]:
        raise ValueError(f"{path} must contain a square Hamiltonian matrix, got shape {hamk.shape}.")
    return 0.5 * (hamk + hamk.conj().T), selected_index, count


def diagonalize_hamiltonian(path: Path, *, hamiltonian_index: int = 0) -> tuple[np.ndarray, np.ndarray, int, int]:
    hamk, selected_index, count = _load_hamiltonian(path, hamiltonian_index=hamiltonian_index)
    energies, vectors = np.linalg.eigh(hamk)
    order = np.argsort(energies)
    return (
        np.asarray(energies[order], dtype=float),
        np.asarray(vectors[:, order], dtype=np.complex128),
        selected_index,
        count,
    )


def _numeric_from_nested_dict(payload: Any, keys: set[str]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).replace("-", "_").lower() in keys:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
            found = _numeric_from_nested_dict(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _numeric_from_nested_dict(item, keys)
            if found is not None:
                return found
    return None


def infer_fermi_energy(band_dir: Path) -> float:
    band_dir = Path(band_dir)
    vbm_path = band_dir / "energies_vbm.txt"
    cbm_path = band_dir / "energies_cbm.txt"
    if vbm_path.is_file() and cbm_path.is_file():
        vbm = np.loadtxt(vbm_path, ndmin=1)
        cbm = np.loadtxt(cbm_path, ndmin=1)
        return float(0.5 * (np.nanmax(vbm) + np.nanmin(cbm)))
    legacy_vbm = sorted(band_dir.glob("band_VBM_*_valley.txt"))
    legacy_cbm = sorted(band_dir.glob("band_CBM_*_valley.txt"))
    if legacy_vbm and legacy_cbm:
        vbm_max = max(float(np.nanmax(np.loadtxt(path, ndmin=1))) for path in legacy_vbm)
        cbm_min = min(float(np.nanmin(np.loadtxt(path, ndmin=1))) for path in legacy_cbm)
        return float(0.5 * (vbm_max + cbm_min))

    try:
        import yaml
    except Exception:
        yaml = None
    if yaml is not None:
        candidates = [band_dir / "config.yaml"]
        for parent in [band_dir, *band_dir.parents[:5]]:
            candidates.extend(sorted(parent.glob("config*.yaml")))
        for path in candidates:
            if not path.is_file():
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            value = _numeric_from_nested_dict(payload, {"efermi", "fermi_energy", "fermi"})
            if value is not None:
                return value
    raise ValueError(
        "Cannot infer Fermi energy from band outputs or nearby config YAML; pass --fermi-energy explicitly."
    )


def select_band_indices(
    energies: np.ndarray,
    fermi_energy: float,
    *,
    valence_count: int,
    conduction_count: int,
) -> dict[str, np.ndarray]:
    valence = np.where(energies < fermi_energy)[0]
    conduction = np.where(energies >= fermi_energy)[0]
    return {
        "valence": valence[-max(0, int(valence_count)) :],
        "conduction": conduction[: max(0, int(conduction_count))],
    }


def group_degenerate_blocks(
    energies: np.ndarray,
    indices: np.ndarray,
    *,
    sector: str,
    degeneracy_tol: float,
    band_indices: np.ndarray | None = None,
) -> list[EnergyBlock]:
    if len(indices) == 0:
        return []
    labels = np.asarray(indices if band_indices is None else band_indices, dtype=int)
    label_by_index = {int(index): int(label) for index, label in zip(indices, labels)}
    blocks: list[EnergyBlock] = []
    current = [int(indices[0])]
    block_id = 0
    for idx in [int(value) for value in indices[1:]]:
        if abs(float(energies[idx] - energies[current[-1]])) <= degeneracy_tol:
            current.append(idx)
        else:
            blocks.append(
                EnergyBlock(
                    block_id,
                    sector,
                    tuple(current),
                    tuple(float(energies[i]) for i in current),
                    tuple(label_by_index[i] for i in current),
                )
            )
            block_id += 1
            current = [idx]
    blocks.append(
        EnergyBlock(
            block_id,
            sector,
            tuple(current),
            tuple(float(energies[i]) for i in current),
            tuple(label_by_index[i] for i in current),
        )
    )
    return blocks


def _resolve_stack_index(count: int, requested: int, path: Path) -> int:
    index = int(requested)
    if index < 0:
        index += count
    if index < 0 or index >= count:
        raise IndexError(f"{path} contains {count} entries; index {requested} is out of range.")
    return index


def _load_saved_sector(
    *,
    energy_path: Path,
    vector_path: Path,
    hamiltonian_index: int,
    count: int,
    sector: str,
) -> dict[str, Any]:
    all_energies = np.loadtxt(energy_path, ndmin=1)
    if all_energies.ndim == 1:
        all_energies = all_energies[None, :]
    k_index = _resolve_stack_index(all_energies.shape[0], hamiltonian_index, energy_path)
    edge_energies = np.asarray(all_energies[k_index], dtype=float)
    vectors = np.load(vector_path, mmap_mode="r")
    if vectors.ndim == 2:
        vectors_k = np.asarray(vectors)
    elif vectors.ndim == 3:
        vec_index = _resolve_stack_index(vectors.shape[0], hamiltonian_index, vector_path)
        vectors_k = np.asarray(vectors[vec_index])
    else:
        raise ValueError(f"{vector_path} must be a 2D or 3D saved wavefunction array, got {vectors.shape}.")
    if vectors_k.shape[1] == edge_energies.shape[0]:
        vectors_by_band = vectors_k
    elif vectors_k.shape[0] == edge_energies.shape[0]:
        vectors_by_band = vectors_k.T
    else:
        raise ValueError(
            f"{vector_path} shape {vectors_k.shape} is incompatible with {energy_path} band count "
            f"{edge_energies.shape[0]}."
        )
    n_select = min(max(0, int(count)), edge_energies.shape[0])
    if sector == "valence":
        cols = np.arange(edge_energies.shape[0] - n_select, edge_energies.shape[0], dtype=int)
    else:
        cols = np.arange(0, n_select, dtype=int)
    return {
        "energies": edge_energies[cols],
        "vectors": np.asarray(vectors_by_band[:, cols], dtype=np.complex128),
        "band_indices": cols,
        "source_index": k_index,
        "source_count": int(all_energies.shape[0]),
    }


def load_saved_wavefunction_point(
    source: dict[str, Any],
    *,
    hamiltonian_index: int,
    valence_count: int,
    conduction_count: int,
) -> dict[str, Any]:
    sectors: dict[str, dict[str, Any]] = {}
    if "VBM" in source and "vec_VBM" in source:
        sectors["valence"] = _load_saved_sector(
            energy_path=source["VBM"],
            vector_path=source["vec_VBM"],
            hamiltonian_index=hamiltonian_index,
            count=valence_count,
            sector="valence",
        )
    if "CBM" in source and "vec_CBM" in source:
        sectors["conduction"] = _load_saved_sector(
            energy_path=source["CBM"],
            vector_path=source["vec_CBM"],
            hamiltonian_index=hamiltonian_index,
            count=conduction_count,
            sector="conduction",
        )
    if not sectors:
        raise FileNotFoundError("Saved-wavefunction source has no usable VBM/CBM vector and energy pairs.")
    index = next(iter(sectors.values()))["source_index"]
    count = next(iter(sectors.values()))["source_count"]
    energies = np.concatenate([sectors[key]["energies"] for key in ("valence", "conduction") if key in sectors])
    vectors = np.hstack([sectors[key]["vectors"] for key in ("valence", "conduction") if key in sectors])
    selected_indices = np.concatenate(
        [sectors[key]["band_indices"] for key in ("valence", "conduction") if key in sectors]
    )
    return {
        "sectors": sectors,
        "energies": energies,
        "vectors": vectors,
        "selected_indices": selected_indices,
        "source_index": index,
        "source_count": count,
    }


def _unitarity_residual(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    eye = np.eye(matrix.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - eye) / np.sqrt(matrix.shape[0]))


def _format_float(value: float | None, *, digits: int = 12) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _format_indices(indices: tuple[int, ...] | list[int] | np.ndarray) -> str:
    return " ".join(str(int(value)) for value in indices)


def _format_energies(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return ", ".join(f"{float(value):.12f}" for value in values)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory {output_dir} already exists and is not empty; pass --overwrite.")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_wavefunctions_npz(
    output_dir: Path,
    point_results: dict[str, dict[str, Any]],
) -> None:
    payload: dict[str, np.ndarray] = {"points": np.array(list(point_results), dtype=str)}
    for point, data in point_results.items():
        prefix = _safe_npz_key(point)
        payload[f"{prefix}_energies"] = data["energies"]
        payload[f"{prefix}_eigenvectors"] = data["vectors"]
        payload[f"{prefix}_source_kind"] = np.array(data["source_kind"], dtype=str)
        payload[f"{prefix}_hamiltonian_index"] = np.array(data["hamiltonian_index"], dtype=int)
        payload[f"{prefix}_hamiltonian_count"] = np.array(data["hamiltonian_count"], dtype=int)
        payload[f"{prefix}_selected_indices"] = np.array(data["selected_indices"], dtype=int)
        payload[f"{prefix}_spin_up_weight"] = np.array(data["spin_up"], dtype=float)
        payload[f"{prefix}_spin_down_weight"] = np.array(data["spin_down"], dtype=float)
        payload[f"{prefix}_spin_label"] = np.array(data["spin_label"], dtype=str)
    np.savez_compressed(output_dir / "high_symmetry_wavefunctions.npz", **payload)


def _write_representation_npz(output_dir: Path, rep_entries: list[dict[str, Any]]) -> None:
    payload: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for entry in rep_entries:
        key = "__".join(
            [
                _safe_npz_key(entry["point"]),
                _safe_npz_key(entry["sector"]),
                f"block{entry['block_id']}",
                _safe_npz_key(entry["operation"]),
            ]
        )
        payload[key] = entry["matrix"]
        metadata.append({k: v for k, v in entry.items() if k != "matrix"})
    payload["metadata_json"] = np.array(json.dumps(metadata, indent=2, sort_keys=True), dtype=str)
    np.savez_compressed(output_dir / "band_representations.npz", **payload)


def _write_bands_csv(output_dir: Path, band_rows: list[dict[str, Any]]) -> None:
    fields = ["point", "sector", "band_index", "energy", "spin_up_weight", "spin_down_weight", "spin_label"]
    with (output_dir / "bands.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in band_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_characters_csv(output_dir: Path, character_rows: list[dict[str, Any]]) -> None:
    fields = [
        "point",
        "sector",
        "block_id",
        "band_indices",
        "energies",
        "operation",
        "antiunitary",
        "symm_rep",
        "sewing_shift",
        "trace_real",
        "trace_imag",
        "unitarity_residual",
    ]
    with (output_dir / "characters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in character_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_summary(
    output_dir: Path,
    *,
    band_dir: Path,
    symmetry_dir: Path,
    fermi_energy: float,
    degeneracy_tol: float,
    point_results: dict[str, dict[str, Any]],
    band_rows: list[dict[str, Any]],
    character_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# TAPW Symmetry Representation Summary",
        "",
        f"- band_dir: `{band_dir}`",
        f"- symmetry_dir: `{symmetry_dir}`",
        f"- fermi_energy: `{fermi_energy:.12f}`",
        f"- degeneracy_tol: `{degeneracy_tol:.6g}`",
        "",
    ]
    rows_by_point_sector: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in band_rows:
        rows_by_point_sector.setdefault((row["point"], row["sector"]), []).append(row)
    chars_by_point: dict[str, list[dict[str, Any]]] = {}
    for row in character_rows:
        chars_by_point.setdefault(row["point"], []).append(row)

    for point, data in point_results.items():
        lines.extend([f"## {point}", ""])
        if data.get("source_kind") == "saved_wavefunctions":
            lines.extend(["Source: saved wavefunctions.", ""])
        elif data.get("source_kind") == "computed_config_points":
            lines.extend(["Source: computed from config fractional points.", ""])
        if int(data["hamiltonian_count"]) > 1:
            lines.extend(
                [
                    f"Selected Hamiltonian slice: `{data['hamiltonian_index']}` of `{data['hamiltonian_count']}`.",
                    "",
                ]
            )
        if not data["operations"]:
            lines.extend(["No raw-H symmetry matrices were available for this point.", ""])
        for sector in ("valence", "conduction"):
            lines.extend([f"### {sector.capitalize()}", ""])
            lines.append("| band | energy | spin weights |")
            lines.append("| ---: | ---: | --- |")
            sector_rows = list(rows_by_point_sector.get((point, sector), []))
            if sector == "valence":
                sector_rows.sort(key=lambda row: float(row["energy"]), reverse=True)
            else:
                sector_rows.sort(key=lambda row: float(row["energy"]))
            for row in sector_rows:
                display_row = {
                    **row,
                    "energy": _format_summary_energy(row["energy"]),
                    "spin_text": _spin_weight_text(row.get("spin_up_weight"), row.get("spin_down_weight")),
                }
                lines.append(
                    "| {band_index} | {energy} | {spin_text} |".format(
                        **display_row
                    )
                )
            if not sector_rows:
                lines.append("|  |  | unavailable |")
            lines.append("")

        point_chars = chars_by_point.get(point, [])
        rep_rows = [
            row
            for row in point_chars
            if row.get("symm_rep") and str(row.get("antiunitary", "false")).lower() == "false"
        ]
        rep_rows.sort(key=_symm_row_sort_key)
        lines.extend(["### Symmetry Representations", ""])
        lines.append("| sector | block | bands | energies | operation | sewing G | rep | D_block |")
        lines.append("| --- | ---: | --- | --- | --- | --- | --- | --- |")
        for row in rep_rows:
            display_row = {
                **row,
                "energies": _format_summary_energies(row["energies"]),
                "d_block": row.get("d_block", ""),
            }
            lines.append(
                "| {sector} | {block_id} | {band_indices} | {energies} | {operation} | {sewing_shift} | {symm_rep} | `{d_block}` |".format(
                    **display_row,
                )
            )
        if not rep_rows:
            lines.append("|  |  |  |  | unavailable |  |  |  |")
        lines.append("")

        lines.extend(["### Characters", ""])
        lines.append("| sector | block | bands | energies | operation | antiunitary | trace | residual |")
        lines.append("| --- | ---: | --- | --- | --- | --- | ---: | ---: |")
        point_chars = sorted(point_chars, key=_symm_row_sort_key)
        for row in point_chars:
            trace = f"{float(row['trace_real']):.6f}"
            imag = float(row["trace_imag"])
            if abs(imag) > 1.0e-12:
                trace += f"{imag:+.6f}i"
            display_row = {**row, "energies": _format_summary_energies(row["energies"])}
            lines.append(
                "| {sector} | {block_id} | {band_indices} | {energies} | {operation} | {antiunitary} | {trace} | {unitarity_residual} |".format(
                    trace=trace,
                    **display_row,
                )
            )
        if not point_chars:
            lines.append("|  |  |  |  | unavailable |  |  |  |")
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_symm_rep_from_point_sources(
    *,
    point_sources: dict[str, dict[str, Any]],
    symmetry_dir: Path,
    output_dir: Path,
    fermi_energy: float,
    degeneracy_tol: float = 2.0e-3,
    overwrite: bool = False,
    source_label: str = "configured TAPW points",
) -> SymmRepResult:
    symmetry_dir = Path(symmetry_dir)
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, overwrite=overwrite)

    if not point_sources:
        raise ValueError("point_sources must contain at least one high-symmetry point.")
    operations_by_point = load_raw_h_symmetry_operations(symmetry_dir, set(point_sources))
    point_results: dict[str, dict[str, Any]] = {}
    band_rows: list[dict[str, Any]] = []
    character_rows: list[dict[str, Any]] = []
    rep_entries: list[dict[str, Any]] = []

    for point, source in point_sources.items():
        sectors = dict(source.get("sectors", {}) or {})
        if not sectors:
            raise ValueError(f"Point {point} has no valence/conduction sector data.")
        source_kind = str(source.get("source_kind", "computed_config_points"))
        sector_payloads: dict[str, dict[str, Any]] = {}
        vectors_by_sector: list[np.ndarray] = []
        energies_by_sector: list[np.ndarray] = []
        selected_indices: list[int] = []
        for sector in ("valence", "conduction"):
            if sector not in sectors:
                continue
            payload = sectors[sector]
            energies = np.asarray(payload["energies"], dtype=float)
            vectors = np.asarray(payload["vectors"], dtype=np.complex128)
            if vectors.ndim != 2 or vectors.shape[1] != len(energies):
                raise ValueError(f"Point {point} sector {sector} vectors must have shape (basis, bands).")
            band_indices = np.asarray(payload.get("band_indices", np.arange(len(energies))), dtype=int)
            if len(band_indices) != len(energies):
                raise ValueError(f"Point {point} sector {sector} band_indices length does not match energies.")
            positions = np.arange(len(energies), dtype=int)
            sector_payloads[sector] = {
                "energies": energies,
                "vectors": vectors,
                "positions": positions,
                "band_indices": band_indices,
            }
            energies_by_sector.append(energies)
            vectors_by_sector.append(vectors)
            selected_indices.extend(int(value) for value in band_indices)
        energies_all = np.concatenate(energies_by_sector)
        vectors_all = np.hstack(vectors_by_sector)

        spin_up: list[float] = []
        spin_down: list[float] = []
        spin_label: list[str] = []
        for state_index in range(vectors_all.shape[1]):
            up, down, label = spin_weights_and_label(vectors_all[:, state_index])
            spin_up.append(np.nan if up is None else up)
            spin_down.append(np.nan if down is None else down)
            spin_label.append(label)

        operations = operations_by_point.get(point, [])
        point_results[point] = {
            "energies": energies_all,
            "vectors": vectors_all,
            "source_kind": source_kind,
            "hamiltonian_index": int(source.get("source_index", 0)),
            "hamiltonian_count": int(source.get("source_count", 1)),
            "selected_indices": selected_indices,
            "spin_up": spin_up,
            "spin_down": spin_down,
            "spin_label": spin_label,
            "operations": operations,
        }

        spin_offset = 0
        for sector, payload in sector_payloads.items():
            sector_energies = payload["energies"]
            sector_vectors = payload["vectors"]
            positions = np.asarray(payload["positions"], dtype=int)
            band_indices = np.asarray(payload["band_indices"], dtype=int)
            row_positions = positions[::-1] if sector == "valence" else positions
            label_by_position = {int(pos): int(label) for pos, label in zip(positions, band_indices)}
            for idx in row_positions:
                spin_lookup_index = spin_offset + int(np.where(positions == idx)[0][0])
                band_rows.append(
                    {
                        "point": point,
                        "sector": sector,
                        "band_index": label_by_position[int(idx)],
                        "energy": _format_float(float(sector_energies[idx])),
                        "spin_up_weight": "" if np.isnan(spin_up[spin_lookup_index]) else _format_float(spin_up[spin_lookup_index], digits=6),
                        "spin_down_weight": "" if np.isnan(spin_down[spin_lookup_index]) else _format_float(spin_down[spin_lookup_index], digits=6),
                        "spin_label": spin_label[spin_lookup_index],
                    }
                )

            blocks = group_degenerate_blocks(
                sector_energies,
                positions,
                sector=sector,
                degeneracy_tol=degeneracy_tol,
                band_indices=band_indices,
            )
            for block in blocks:
                block_vectors = sector_vectors[:, list(block.indices)]
                for operation in operations:
                    if operation.matrix.shape != (sector_vectors.shape[0], sector_vectors.shape[0]):
                        raise ValueError(
                            f"Raw-H matrix {operation.operation} at {point} has shape {operation.matrix.shape}, "
                            f"but wavefunctions have dimension {sector_vectors.shape[0]}."
                        )
                    sewn_action = _sewn_raw_action(operation, source, sector_vectors.shape[0])
                    if sewn_action is None:
                        continue
                    action_matrix, sewing_shift = sewn_action
                    projected = project_operation_to_subspace(
                        action_matrix,
                        block_vectors,
                        antiunitary=operation.antiunitary,
                    )
                    trace = np.trace(projected)
                    residual = _unitarity_residual(projected)
                    rep_label = symmetry_rep_label(
                        projected,
                        block_vectors,
                        antiunitary=operation.antiunitary,
                    )
                    character_rows.append(
                        {
                            "point": point,
                            "sector": sector,
                            "block_id": block.block_id,
                            "band_indices": _format_indices(block.band_indices),
                            "energies": _format_energies(block.energies),
                            "operation": operation.operation,
                            "antiunitary": str(bool(operation.antiunitary)).lower(),
                            "symm_rep": rep_label,
                            "sewing_shift": sewing_shift,
                            "d_block": "" if operation.antiunitary else _format_d_block(projected),
                            "trace_real": f"{float(trace.real):.12g}",
                            "trace_imag": f"{float(trace.imag):.12g}",
                            "unitarity_residual": f"{residual:.12g}",
                        }
                    )
                    rep_entries.append(
                        {
                            "point": point,
                            "sector": sector,
                            "block_id": block.block_id,
                            "band_indices": list(block.band_indices),
                            "energies": list(block.energies),
                            "operation": operation.operation,
                            "antiunitary": bool(operation.antiunitary),
                            "unitarity_residual": residual,
                            "matrix": projected,
                        }
                    )
            spin_offset += len(sector_energies)

    _write_wavefunctions_npz(output_dir, point_results)
    _write_representation_npz(output_dir, rep_entries)
    _write_bands_csv(output_dir, band_rows)
    _write_characters_csv(output_dir, character_rows)
    _write_summary(
        output_dir,
        band_dir=Path(source_label),
        symmetry_dir=symmetry_dir,
        fermi_energy=float(fermi_energy),
        degeneracy_tol=degeneracy_tol,
        point_results=point_results,
        band_rows=band_rows,
        character_rows=character_rows,
    )
    return SymmRepResult(output_dir=output_dir, points=tuple(point_results), fermi_energy=float(fermi_energy))


def run_symm_rep(
    *,
    band_dir: Path,
    symmetry_dir: Path,
    output_dir: Path,
    valence_count: int = 20,
    conduction_count: int = 20,
    degeneracy_tol: float = 2.0e-3,
    fermi_energy: float | None = None,
    hamiltonian_index: int = 0,
    points: list[str] | tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> SymmRepResult:
    band_dir = Path(band_dir)
    symmetry_dir = Path(symmetry_dir)
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, overwrite=overwrite)

    sources = discover_high_symmetry_sources(band_dir)
    if points is not None:
        requested_points = {str(point) for point in points}
        sources = {point: source for point, source in sources.items() if point in requested_points}
    if not sources:
        raise FileNotFoundError(
            f"No high-symmetry hamk_<point>_valley.npy files or saved vec/band pairs found in {band_dir}."
        )
    if fermi_energy is None:
        fermi_energy = infer_fermi_energy(band_dir)
    fermi_energy = float(fermi_energy)

    operations_by_point = load_raw_h_symmetry_operations(symmetry_dir, set(sources))
    point_results: dict[str, dict[str, Any]] = {}
    band_rows: list[dict[str, Any]] = []
    character_rows: list[dict[str, Any]] = []
    rep_entries: list[dict[str, Any]] = []

    for point, source in sources.items():
        source_kind = str(source["kind"])
        sector_payloads: dict[str, dict[str, Any]]
        if source_kind == "hamiltonian":
            energies, vectors, selected_ham_index, ham_count = diagonalize_hamiltonian(
                source["hamiltonian"],
                hamiltonian_index=hamiltonian_index,
            )
            selections = select_band_indices(
                energies,
                fermi_energy,
                valence_count=valence_count,
                conduction_count=conduction_count,
            )
            sector_payloads = {
                sector: {
                    "energies": energies,
                    "vectors": vectors,
                    "positions": np.asarray(indices, dtype=int),
                    "band_indices": np.asarray(indices, dtype=int),
                }
                for sector, indices in selections.items()
            }
            selected_indices = sorted(set(int(idx) for values in selections.values() for idx in values))
        else:
            saved = load_saved_wavefunction_point(
                source,
                hamiltonian_index=hamiltonian_index,
                valence_count=valence_count,
                conduction_count=conduction_count,
            )
            energies = saved["energies"]
            vectors = saved["vectors"]
            selected_ham_index = int(saved["source_index"])
            ham_count = int(saved["source_count"])
            selected_indices = [int(value) for value in saved["selected_indices"]]
            sector_payloads = {
                sector: {
                    "energies": payload["energies"],
                    "vectors": payload["vectors"],
                    "positions": np.arange(len(payload["energies"]), dtype=int),
                    "band_indices": payload["band_indices"],
                }
                for sector, payload in saved["sectors"].items()
            }
        spin_up: list[float] = []
        spin_down: list[float] = []
        spin_label: list[str] = []
        for state_index in range(vectors.shape[1]):
            up, down, label = spin_weights_and_label(vectors[:, state_index])
            spin_up.append(np.nan if up is None else up)
            spin_down.append(np.nan if down is None else down)
            spin_label.append(label)

        operations = operations_by_point.get(point, [])
        point_results[point] = {
            "energies": energies,
            "vectors": vectors,
            "source_kind": source_kind,
            "hamiltonian_index": selected_ham_index,
            "hamiltonian_count": ham_count,
            "selected_indices": selected_indices,
            "spin_up": spin_up,
            "spin_down": spin_down,
            "spin_label": spin_label,
            "operations": operations,
        }

        for sector, payload in sector_payloads.items():
            sector_energies = payload["energies"]
            sector_vectors = payload["vectors"]
            positions = np.asarray(payload["positions"], dtype=int)
            band_indices = np.asarray(payload["band_indices"], dtype=int)
            row_positions = positions[::-1] if sector == "valence" else positions
            label_by_position = {int(pos): int(label) for pos, label in zip(positions, band_indices)}
            for idx in row_positions:
                spin_lookup_index = int(idx) if source_kind == "hamiltonian" else int(
                    np.where(positions == idx)[0][0]
                    + (0 if sector == "valence" else len(sector_payloads.get("valence", {}).get("energies", [])))
                )
                band_rows.append(
                    {
                        "point": point,
                        "sector": sector,
                        "band_index": label_by_position[int(idx)],
                        "energy": _format_float(float(sector_energies[idx])),
                        "spin_up_weight": "" if np.isnan(spin_up[spin_lookup_index]) else _format_float(spin_up[spin_lookup_index], digits=6),
                        "spin_down_weight": "" if np.isnan(spin_down[spin_lookup_index]) else _format_float(spin_down[spin_lookup_index], digits=6),
                        "spin_label": spin_label[spin_lookup_index],
                    }
                )

            blocks = group_degenerate_blocks(
                sector_energies,
                positions,
                sector=sector,
                degeneracy_tol=degeneracy_tol,
                band_indices=band_indices,
            )
            for block in blocks:
                block_vectors = sector_vectors[:, list(block.indices)]
                for operation in operations:
                    if operation.matrix.shape != (sector_vectors.shape[0], sector_vectors.shape[0]):
                        raise ValueError(
                            f"Raw-H matrix {operation.operation} at {point} has shape {operation.matrix.shape}, "
                            f"but wavefunctions have dimension {sector_vectors.shape[0]}."
                        )
                    sewn_action = _sewn_raw_action(operation, source, sector_vectors.shape[0])
                    if sewn_action is None:
                        continue
                    action_matrix, sewing_shift = sewn_action
                    projected = project_operation_to_subspace(
                        action_matrix,
                        block_vectors,
                        antiunitary=operation.antiunitary,
                    )
                    trace = np.trace(projected)
                    residual = _unitarity_residual(projected)
                    rep_label = symmetry_rep_label(
                        projected,
                        block_vectors,
                        antiunitary=operation.antiunitary,
                    )
                    character_rows.append(
                        {
                            "point": point,
                            "sector": sector,
                            "block_id": block.block_id,
                            "band_indices": _format_indices(block.band_indices),
                            "energies": _format_energies(block.energies),
                            "operation": operation.operation,
                            "antiunitary": str(bool(operation.antiunitary)).lower(),
                            "symm_rep": rep_label,
                            "sewing_shift": sewing_shift,
                            "d_block": "" if operation.antiunitary else _format_d_block(projected),
                            "trace_real": f"{float(trace.real):.12g}",
                            "trace_imag": f"{float(trace.imag):.12g}",
                            "unitarity_residual": f"{residual:.12g}",
                        }
                    )
                    rep_entries.append(
                        {
                            "point": point,
                            "sector": sector,
                            "block_id": block.block_id,
                            "band_indices": list(block.band_indices),
                            "energies": list(block.energies),
                            "operation": operation.operation,
                            "antiunitary": bool(operation.antiunitary),
                            "unitarity_residual": residual,
                            "matrix": projected,
                        }
                    )

    _write_wavefunctions_npz(output_dir, point_results)
    _write_representation_npz(output_dir, rep_entries)
    _write_bands_csv(output_dir, band_rows)
    _write_characters_csv(output_dir, character_rows)
    _write_summary(
        output_dir,
        band_dir=band_dir,
        symmetry_dir=symmetry_dir,
        fermi_energy=fermi_energy,
        degeneracy_tol=degeneracy_tol,
        point_results=point_results,
        band_rows=band_rows,
        character_rows=character_rows,
    )
    return SymmRepResult(output_dir=output_dir, points=tuple(point_results), fermi_energy=fermi_energy)
