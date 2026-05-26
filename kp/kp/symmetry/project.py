from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..blocks.blocks import calculate_energy_lists, get_H_block
from ..blocks.downfold import DownfoldingOptions, downfold_from_projectors
from ..io.tapw_loader import load_Q_sets, load_hamk


@dataclass
class ProjectionState:
    hamk: np.ndarray
    heff: np.ndarray
    u_low: np.ndarray


@dataclass
class RepresentationData:
    matrix: np.ndarray
    from_full_spinful: bool
    spin_leakage: float | None


def _resolve(path: str | None, base_dir: str) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_nlow_state_list(project_cfg: dict[str, Any]) -> list[list[int]]:
    active = project_cfg.get("active_indices")
    if active is not None:
        if isinstance(active, str):
            return [[int(part.strip()) for part in active.split(",") if part.strip()]]
        return [[int(value) for value in active]]
    nlow_state_list = project_cfg.get("nlow_state_list", [])
    if nlow_state_list and not isinstance(nlow_state_list[0], (list, tuple)):
        return [[int(x) for x in nlow_state_list]]
    return [[int(x) for x in layer] for layer in nlow_state_list]


def _downfold_method(project_cfg: dict[str, Any]) -> str:
    return str(project_cfg.get("downfold_method", project_cfg.get("method", "fixed_schur"))).lower()


def _e_ref(project_cfg: dict[str, Any]) -> float | None:
    value = project_cfg.get("e_ref", project_cfg.get("E_ref"))
    return None if value is None else float(value)


def _infer_orbitals_per_layer(hamk2d: np.ndarray, q_count: int, num_layers: int) -> int:
    base = int(hamk2d.shape[0]) // 2
    divisor = int(q_count) * int(num_layers)
    if divisor <= 0 or base % divisor != 0:
        raise ValueError(f"Cannot infer orbitals: base={base}, q_count={q_count}, num_layers={num_layers}")
    return base // divisor


def _spin_slice_hamk(hamk2d: np.ndarray, spin: str) -> np.ndarray:
    spin_lower = str(spin).lower()
    if spin_lower == "all":
        return np.asarray(hamk2d, dtype=np.complex128)
    half = hamk2d.shape[0] // 2
    if spin_lower == "up":
        return np.asarray(hamk2d[:half, :half], dtype=np.complex128)
    if spin_lower == "down":
        return np.asarray(hamk2d[half:, half:], dtype=np.complex128)
    raise ValueError(f"Unsupported spin value: {spin!r}")


def _fro_relative(lhs: np.ndarray, rhs: np.ndarray, denominator: np.ndarray) -> float:
    denom = float(np.linalg.norm(denominator))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(lhs - rhs) / denom)


def _unitarity_error(matrix: np.ndarray) -> float:
    ident = np.eye(matrix.shape[1], dtype=np.complex128)
    return _fro_relative(matrix.conj().T @ matrix, ident, ident)


def _load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Representation file missing: {path}")
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.complex128)
    if path.suffix == ".npz":
        try:
            import scipy.sparse

            sparse = scipy.sparse.load_npz(path)
            return np.asarray(sparse.toarray(), dtype=np.complex128)
        except Exception:
            data = np.load(path)
            for key in ("matrix", "D", "representation", "arr_0"):
                if key in data.files:
                    return np.asarray(data[key], dtype=np.complex128)
            raise ValueError(f"Cannot find matrix key in {path}; keys={data.files}")
    raise ValueError(f"Unsupported representation file extension: {path}")


def _slice_representation_for_spin(matrix: np.ndarray, spin: str, target_dim: int) -> RepresentationData:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape == (target_dim, target_dim):
        return RepresentationData(matrix=matrix, from_full_spinful=False, spin_leakage=None)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Representation matrix must be square, got shape={matrix.shape}")
    spin_lower = str(spin).lower()
    if spin_lower == "all":
        raise ValueError(f"D dimension {matrix.shape} does not match spin=all U dimension {target_dim}")
    if matrix.shape != (2 * target_dim, 2 * target_dim):
        raise ValueError(f"D dimension {matrix.shape} cannot be sliced to U dimension {target_dim}")
    if spin_lower == "up":
        row = slice(0, target_dim)
        other = slice(target_dim, 2 * target_dim)
    elif spin_lower == "down":
        row = slice(target_dim, 2 * target_dim)
        other = slice(0, target_dim)
    else:
        raise ValueError(f"Unsupported spin value: {spin!r}")
    leakage_blocks = [matrix[row, other], matrix[other, row]]
    leakage = float(np.sqrt(sum(float(np.linalg.norm(block) ** 2) for block in leakage_blocks)) / np.sqrt(target_dim))
    return RepresentationData(matrix=matrix[row, row], from_full_spinful=True, spin_leakage=leakage)


def _operation_entry(manifest: dict[str, Any], valley: str, operation: str) -> dict[str, Any]:
    def _matches(entry: dict[str, Any]) -> bool:
        entry_operation = str(entry.get("operation", entry.get("name", "")))
        entry_valley = str(entry.get("valley_label", entry.get("valley", valley)))
        return entry_operation == operation and entry_valley == valley

    operations = manifest.get("operations")
    if isinstance(operations, dict):
        if valley in operations:
            valley_ops = operations[valley]
            if isinstance(valley_ops, dict) and operation in valley_ops:
                entry = valley_ops[operation]
                return dict(entry) if isinstance(entry, dict) else {"filename": entry}
            if isinstance(valley_ops, dict) and isinstance(valley_ops.get("operations"), dict):
                entry = valley_ops["operations"].get(operation)
                if entry is not None:
                    return dict(entry) if isinstance(entry, dict) else {"filename": entry}
        if operation in operations:
            entry = operations[operation]
            return dict(entry) if isinstance(entry, dict) else {"filename": entry}
    if isinstance(operations, list):
        for entry in operations:
            if not isinstance(entry, dict):
                continue
            if _matches(entry):
                return dict(entry)
    matrices = manifest.get("matrices")
    if isinstance(matrices, list):
        for entry in matrices:
            if isinstance(entry, dict) and _matches(entry):
                return dict(entry)
    valleys = manifest.get("valleys")
    if isinstance(valleys, dict) and valley in valleys:
        valley_ops = valleys[valley].get("operations", {}) if isinstance(valleys[valley], dict) else {}
        if operation in valley_ops:
            entry = valley_ops[operation]
            return dict(entry) if isinstance(entry, dict) else {"filename": entry}
    raise KeyError(f"manifest missing operation {valley}/{operation}")


def _entry_filename(entry: dict[str, Any], valley: str, operation: str) -> str:
    for key in ("filename", "file", "path", "representation_filename"):
        if entry.get(key):
            return str(entry[key])
    return f"{valley}/{operation}.npz"


def _pairs_from_entry(entry: dict[str, Any], nk: int, *, default_k_index: int | None = None) -> list[tuple[int, int]]:
    raw_pairs = entry.get("k_pairs", entry.get("pairs"))
    if raw_pairs is not None:
        pairs: list[tuple[int, int]] = []
        for item in raw_pairs:
            if isinstance(item, dict):
                target = item.get("target", item.get("target_index", item.get("k_target")))
                source = item.get("source", item.get("source_index", item.get("k_source")))
            else:
                target, source = item
            pairs.append((int(target), int(source)))
        return pairs
    target_indices = entry.get("target_indices")
    source_indices = entry.get("source_indices")
    if target_indices is not None and source_indices is not None:
        if len(target_indices) != len(source_indices):
            raise ValueError("target_indices and source_indices have different lengths")
        return [(int(t), int(s)) for t, s in zip(target_indices, source_indices)]
    rule = str(entry.get("source_k_rule", entry.get("k_rule", ""))).lower()
    target_rule = str(entry.get("target_k_rule", "")).lower()
    if rule in {"same", "identity", "same_index"} or target_rule in {"same", "identity", "same_index"}:
        return [(idx, idx) for idx in range(nk)]
    if rule in {"gamma", "gamma_only"} or target_rule in {"gamma", "gamma_only"}:
        return [(0, 0)]
    if default_k_index is not None:
        return [(int(default_k_index), int(default_k_index))]
    raise ValueError("Manifest operation must provide k_pairs/source_indices or a supported k rule")


def _projectors_for_k(
    hamk_spin: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    *,
    orb0: int,
    spin: str,
    mode: str,
    nlow_state_list: list[list[int]],
    norb_fix_list: list[Any],
    method: str,
    e_ref: float | None,
    project_cfg: dict[str, Any],
) -> ProjectionState:
    q_layers = [[q1], [q2]]
    orb_layers = [[orb0], [orb0]]
    _, h_vec_blk, _, _ = get_H_block(
        hamk_spin,
        q_layers,
        [1, 1],
        orb_layers,
        nlow_state_list,
        norb_fix_list,
        spin=spin,
        mode=mode,
    )
    include_high = method != "first_order"
    u_low, u_high = calculate_energy_lists(
        h_vec_blk,
        nlow_state_list,
        norb_fix_list,
        q_layers,
        orb_layers,
        mode=mode,
        include_high=include_high,
    )
    u_low = np.asarray(u_low, dtype=np.complex128)
    u_high = None if u_high is None else np.asarray(u_high, dtype=np.complex128)
    result = downfold_from_projectors(
        hamk_spin,
        u_low,
        u_high,
        DownfoldingOptions(
            method=method,
            e_ref=e_ref,
            pole_warning_mev=float(project_cfg.get("pole_warning_mev", 10.0)),
            pole_danger_mev=float(project_cfg.get("pole_danger_mev", 1.0)),
            fail_on_near_pole=_as_bool(project_cfg.get("fail_on_near_pole", False)),
        ),
    )
    return ProjectionState(hamk=np.asarray(hamk_spin, dtype=np.complex128), heff=np.asarray(result.heff, dtype=np.complex128), u_low=u_low)


def _full_space_covariance_residual(
    *,
    d_full: np.ndarray,
    h_target: np.ndarray,
    h_source: np.ndarray,
    antiunitary: bool,
) -> float:
    h_source_work = h_source.conj() if antiunitary else h_source
    h_cov = d_full @ h_source_work @ d_full.conj().T
    return _fro_relative(h_target, h_cov, h_target)


def _project_operation(
    *,
    operation: str,
    antiunitary: bool,
    d_full: np.ndarray,
    states: dict[int, ProjectionState],
    pairs: list[tuple[int, int]],
    tolerance: float,
):
    raw_mats = []
    polar_mats = []
    pair_rows = []
    for target_idx, source_idx in pairs:
        target = states[target_idx]
        source = states[source_idx]
        u_source = source.u_low.conj() if antiunitary else source.u_low
        h_source = source.heff.conj() if antiunitary else source.heff
        d_raw = target.u_low.conj().T @ d_full @ u_source
        x, singular_values, yh = np.linalg.svd(d_raw, full_matrices=False)
        d_polar = x @ yh

        def metrics(d_matrix: np.ndarray) -> dict[str, Any]:
            image = d_full @ u_source
            leakage = float(np.linalg.norm(image - target.u_low @ d_matrix) / np.sqrt(d_matrix.shape[0]))
            h_cov = d_matrix @ h_source @ d_matrix.conj().T
            return {
                "d_unitarity_error": _unitarity_error(d_matrix),
                "subspace_leakage": leakage,
                "heff_covariance_residual": _fro_relative(target.heff, h_cov, target.heff),
            }

        raw_metrics = metrics(d_raw)
        polar_metrics = metrics(d_polar)
        sv_max_dev = float(np.max(np.abs(singular_values - 1.0))) if singular_values.size else 0.0
        if sv_max_dev > tolerance:
            raise ValueError(f"{operation} k=({target_idx},{source_idx}) singular values deviate from 1 by {sv_max_dev:.3e}")
        if raw_metrics["subspace_leakage"] > tolerance:
            raise ValueError(f"{operation} k=({target_idx},{source_idx}) subspace leakage {raw_metrics['subspace_leakage']:.3e} exceeds tolerance")
        if raw_metrics["heff_covariance_residual"] > tolerance:
            raise ValueError(
                f"{operation} k=({target_idx},{source_idx}) Heff covariance residual "
                f"{raw_metrics['heff_covariance_residual']:.3e} exceeds tolerance"
            )
        raw_mats.append(d_raw)
        polar_mats.append(d_polar)
        pair_rows.append(
            {
                "target_k_index": int(target_idx),
                "source_k_index": int(source_idx),
                "d_shape": list(d_raw.shape),
                "singular_values": [float(value) for value in singular_values],
                "raw": raw_metrics,
                "polar": polar_metrics,
            }
        )
    return raw_mats, polar_mats, pair_rows


def _save_matrix_stack(path: Path, matrices: list[np.ndarray]) -> None:
    if len(matrices) == 1:
        np.save(path, matrices[0])
    else:
        np.save(path, np.stack(matrices, axis=0))


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# KP Symmetry Projection Summary",
        "",
        f"Valley: {summary['valley']}",
        f"Spin: {summary['spin']}",
        f"Tolerance: {summary['tolerance']}",
        f"Full dimension: {summary['full_dim']}",
        f"Low dimension: {summary['low_dim']}",
        "",
        "## Operations",
        "",
    ]
    for op in summary["operations"]:
        lines.append(f"### {op['operation']}")
        lines.append("")
        lines.append(f"- antiunitary: {op['antiunitary']}")
        lines.append(f"- representation_file: `{op['representation_file']}`")
        if op.get("spin_leakage") is not None:
            lines.append(f"- spin_leakage: {op['spin_leakage']:.6e}")
        for pair in op["pairs"]:
            raw = pair["raw"]
            polar = pair["polar"]
            lines.append(
                f"- k target/source {pair['target_k_index']}/{pair['source_k_index']}: "
                f"full cov={pair.get('full_space_covariance_residual', float('nan')):.6e}, "
                f"raw cov={raw['heff_covariance_residual']:.6e}, "
                f"raw leakage={raw['subspace_leakage']:.6e}, "
                f"polar cov={polar['heff_covariance_residual']:.6e}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_symmetry_projection_from_config(cfg_path: str) -> dict[str, Any]:
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    material = cfg.get("material", {})
    plot_cfg = cfg.get("plot", {})
    project_cfg = cfg.get("project", {})
    symm_cfg = cfg.get("symm", {})
    if not _as_bool(symm_cfg.get("enable", True)):
        raise ValueError("symm.enable is false")

    valley = str(symm_cfg.get("valley", "K1"))
    spin = str(symm_cfg.get("spin", material.get("spin", "all"))).lower()
    operations = [str(op) for op in symm_cfg.get("operations", [])]
    if not operations:
        raise ValueError("symm.operations must not be empty")
    tolerance = float(symm_cfg.get("tolerance", 1.0e-2))

    tapw_symmetry_dir = _resolve(symm_cfg.get("tapw_symmetry_dir"), cfg_dir)
    if tapw_symmetry_dir is None:
        raise ValueError("symm.tapw_symmetry_dir is required")
    rep_root = Path(tapw_symmetry_dir) / "representations"
    manifest_path = rep_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Representation manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    hamk_file = _resolve(material.get("hamk_file"), cfg_dir)
    qset1_file = _resolve(material.get("qset1_file"), cfg_dir)
    qset2_file = _resolve(material.get("qset2_file"), cfg_dir)
    if hamk_file is None or qset1_file is None or qset2_file is None:
        raise ValueError("material.hamk_file/qset1_file/qset2_file are required")
    hamk = load_hamk(hamk_file, mmap_mode="r")
    q1, q2 = load_Q_sets(qset1_file, qset2_file)
    if q2 is None:
        raise ValueError("K-valley symmetry projection requires qset2_file")
    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    nk = int(hamk3d.shape[0])
    q_count = int(len(q1))
    num_layers = int(material.get("num_layers", 2))
    mode = str(project_cfg.get("mode", "K1")).lower()
    method = _downfold_method(project_cfg)
    e_ref = _e_ref(project_cfg)
    if method in {"fixed_schur", "linearized_lowdin"} and e_ref is None:
        raise ValueError(f"project.downfold_method={method!r} requires project.e_ref")
    nlow_state_list = _normalize_nlow_state_list(project_cfg)
    norb_fix_list = project_cfg.get("norb_fix_list", [])
    if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
        orb0 = int(material["num_orb_per_layer"][0])
    else:
        orb0 = _infer_orbitals_per_layer(np.asarray(hamk3d[0]), q_count, num_layers)

    operation_entries: dict[str, dict[str, Any]] = {}
    all_pairs: set[tuple[int, int]] = set()
    default_k_index = int(plot_cfg.get("hamk_index", 0))
    for operation in operations:
        entry = _operation_entry(manifest, valley, operation)
        pairs = _pairs_from_entry(entry, nk, default_k_index=default_k_index)
        for target_idx, source_idx in pairs:
            if target_idx < 0 or target_idx >= nk or source_idx < 0 or source_idx >= nk:
                raise IndexError(f"{operation} source/target k unavailable: target={target_idx}, source={source_idx}, nk={nk}")
            all_pairs.add((target_idx, source_idx))
        entry["_pairs"] = pairs
        operation_entries[operation] = entry

    required_k = sorted({idx for pair in all_pairs for idx in pair})
    if not required_k:
        raise ValueError("No source/target k points requested by symmetry operations")
    hamk_spin_by_k = {
        k_index: _spin_slice_hamk(np.asarray(hamk3d[k_index], dtype=np.complex128), spin)
        for k_index in required_k
    }
    full_dim = int(hamk_spin_by_k[required_k[0]].shape[0])

    operation_payloads: dict[str, dict[str, Any]] = {}
    for operation in operations:
        entry = operation_entries[operation]
        antiunitary = _as_bool(entry.get("antiunitary", False))
        filename = _entry_filename(entry, valley, operation)
        matrix_raw = _load_matrix(rep_root / filename)
        rep = _slice_representation_for_spin(matrix_raw, spin, full_dim)
        if rep.spin_leakage is not None and rep.spin_leakage > tolerance:
            raise ValueError(f"{operation} spin off-block leakage {rep.spin_leakage:.3e} exceeds tolerance")
        if rep.matrix.shape != (full_dim, full_dim):
            raise ValueError(f"{operation} D shape {rep.matrix.shape} does not match U_low full dimension {full_dim}")
        full_pair_rows = []
        for target_idx, source_idx in entry["_pairs"]:
            full_residual = _full_space_covariance_residual(
                d_full=rep.matrix,
                h_target=hamk_spin_by_k[target_idx],
                h_source=hamk_spin_by_k[source_idx],
                antiunitary=antiunitary,
            )
            if full_residual > tolerance:
                raise ValueError(
                    f"{operation} k=({target_idx},{source_idx}) full-space covariance residual "
                    f"{full_residual:.3e} exceeds tolerance. The TAPW representation is not "
                    "compatible with the KP input hamk/source-k rule."
                )
            full_pair_rows.append(
                {
                    "target_k_index": int(target_idx),
                    "source_k_index": int(source_idx),
                    "full_space_covariance_residual": full_residual,
                }
            )
        operation_payloads[operation] = {
            "entry": entry,
            "antiunitary": antiunitary,
            "filename": filename,
            "representation": rep,
            "full_pair_rows": full_pair_rows,
        }

    states: dict[int, ProjectionState] = {}
    for k_index in required_k:
        states[k_index] = _projectors_for_k(
            hamk_spin_by_k[k_index],
            q1,
            q2,
            orb0=orb0,
            spin=spin,
            mode=mode,
            nlow_state_list=nlow_state_list,
            norb_fix_list=norb_fix_list,
            method=method,
            e_ref=e_ref,
            project_cfg=project_cfg,
        )

    first_state = states[required_k[0]]
    low_dim = int(first_state.u_low.shape[1])
    output_dir = Path(_resolve(symm_cfg.get("output_dir", "symm_project"), cfg_dir) or "symm_project")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": cfg_path,
        "valley": valley,
        "spin": spin,
        "mode": mode,
        "tolerance": tolerance,
        "q_count": q_count,
        "orbital_block_dim": orb0,
        "full_dim": full_dim,
        "low_dim": low_dim,
        "operations": [],
    }

    for operation in operations:
        payload = operation_payloads[operation]
        entry = payload["entry"]
        antiunitary = bool(payload["antiunitary"])
        filename = str(payload["filename"])
        rep = payload["representation"]

        raw_mats, polar_mats, pair_rows = _project_operation(
            operation=operation,
            antiunitary=antiunitary,
            d_full=rep.matrix,
            states=states,
            pairs=entry["_pairs"],
            tolerance=tolerance,
        )
        for pair_row, full_pair_row in zip(pair_rows, payload["full_pair_rows"]):
            pair_row["full_space_covariance_residual"] = full_pair_row["full_space_covariance_residual"]
        _save_matrix_stack(output_dir / f"{operation}_low_raw.npy", raw_mats)
        _save_matrix_stack(output_dir / f"{operation}_low_polar.npy", polar_mats)
        summary["operations"].append(
            {
                "operation": operation,
                "antiunitary": antiunitary,
                "representation_file": str(rep_root / filename),
                "from_full_spinful": rep.from_full_spinful,
                "spin_leakage": rep.spin_leakage,
                "pairs": pair_rows,
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_md(output_dir / "summary.md", summary)
    return summary
