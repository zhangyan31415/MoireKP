from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.linalg
import yaml

from ..src.moire_refactored import (
    MoireConfig,
    generate_kpath_from_file,
    infer_bM_vectors_from_Q_set1,
    load_Q_sets_from_gvec_files,
    rot,
    run_end_to_end,
)


@dataclass
class ConfiguredModel:
    path: Path
    raw: dict[str, Any]
    source_config: Path
    source_raw: dict[str, Any]
    qset1_file: Path
    qset2_file: Path
    kpoints_file: Path | None
    heff_file: Path
    heff_eig_file: Path | None
    output_dir: Path
    q_rotation_deg: float
    fit_indices: list[int]
    band_indices: list[int] | None
    n_orb: tuple[int, int]
    nlow_state: list[int]
    bM_config: dict[str, Any]
    harmonics_config: dict[str, Any]
    max_order: dict[str, int]
    symmetry_map: dict[str, list[dict[str, Any]]]
    coeff_tol: float
    compare_to_heff: bool
    kpath_config: dict[str, Any] = field(default_factory=dict)
    band_slice: list[int] | None = None


def _resolve_path(value: str | Path | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _as_int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of integers, got {value!r}")
    return [int(item) for item in value]


def _model_section(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    section = raw.get("model", {})
    if not isinstance(section, Mapping):
        raise ValueError("model section must be a mapping")
    return section


def _get_path_value(raw: Mapping[str, Any], key: str) -> Any:
    if key in raw:
        return raw[key]
    input_section = raw.get("input", {})
    if isinstance(input_section, Mapping) and key in input_section:
        return input_section[key]
    return None


def load_model_config(path: str | Path) -> ConfiguredModel:
    cfg_path = Path(path).resolve()
    raw = _load_yaml(cfg_path)
    base = cfg_path.parent

    source_config = _resolve_path(raw.get("source_config"), base)
    if source_config is None:
        raise ValueError("Configured model YAML requires source_config")
    source_raw = _load_yaml(source_config)
    source_base = source_config.parent
    material = source_raw.get("material", {})
    project = source_raw.get("project", {})
    plot = source_raw.get("plot", {})
    if not isinstance(material, Mapping) or not isinstance(project, Mapping) or not isinstance(plot, Mapping):
        raise ValueError("source_config must contain mapping sections: material, project, plot")

    qset1_file = _resolve_path(material.get("qset1_file"), source_base)
    qset2_file = _resolve_path(material.get("qset2_file"), source_base)
    if qset1_file is None or qset2_file is None:
        raise ValueError("source_config material must provide qset1_file and qset2_file")

    heff_file = _resolve_path(_get_path_value(raw, "heff_file"), base)
    if heff_file is None:
        project_out = _resolve_path(project.get("out_dir"), source_base)
        if project_out is None:
            raise ValueError("heff_file is missing and source_config project.out_dir is unavailable")
        heff_file = project_out / "heff_list.npy"
    heff_eig_file = _resolve_path(_get_path_value(raw, "heff_eig_file"), base)
    if heff_eig_file is None:
        candidate = heff_file.with_name("heff_eig.npy")
        heff_eig_file = candidate if candidate.exists() else None

    kpoints_file = _resolve_path(_get_path_value(raw, "kpoints_file"), base)

    output_section = raw.get("output", {})
    if output_section is None:
        output_section = {}
    if not isinstance(output_section, Mapping):
        raise ValueError("output section must be a mapping")
    output_dir = _resolve_path(output_section.get("dir", "model"), base)
    if output_dir is None:
        raise ValueError("Failed to resolve output directory")

    model = _model_section(raw)
    n_orb_raw = model.get("n_orb", [model.get("n_orb1", 2), model.get("n_orb2", 2)])
    n_orb_values = _as_int_list(n_orb_raw, name="model.n_orb")
    if len(n_orb_values) != 2:
        raise ValueError(f"model.n_orb must have length 2, got {n_orb_values}")
    nlow_state = _as_int_list(model.get("nlow_state", n_orb_values), name="model.nlow_state")
    if len(nlow_state) != 2:
        raise ValueError(f"model.nlow_state must have length 2, got {nlow_state}")

    fit = raw.get("fit", {})
    if not isinstance(fit, Mapping):
        raise ValueError("fit section must be a mapping")
    fit_indices = _as_int_list(fit.get("indices", []), name="fit.indices")

    bands = raw.get("bands", {})
    if bands is None:
        bands = {}
    if not isinstance(bands, Mapping):
        raise ValueError("bands section must be a mapping")
    band_indices_raw = bands.get("indices")
    band_indices = None if band_indices_raw is None else _as_int_list(band_indices_raw, name="bands.indices")
    band_slice = None if bands.get("band_slice") is None else _as_int_list(bands.get("band_slice"), name="bands.band_slice")

    harmonics = model.get("harmonics", {})
    if not isinstance(harmonics, Mapping):
        raise ValueError("model.harmonics must be a mapping")
    symmetry_map = model.get("symmetry_map", {})
    if not isinstance(symmetry_map, Mapping):
        raise ValueError("model.symmetry_map must be a mapping")

    kpath_config = raw.get("kpath", {})
    if kpath_config is None:
        kpath_config = {}
    if not isinstance(kpath_config, Mapping):
        raise ValueError("kpath section must be a mapping")

    return ConfiguredModel(
        path=cfg_path,
        raw=dict(raw),
        source_config=source_config,
        source_raw=dict(source_raw),
        qset1_file=qset1_file,
        qset2_file=qset2_file,
        kpoints_file=kpoints_file,
        heff_file=heff_file,
        heff_eig_file=heff_eig_file,
        output_dir=output_dir,
        q_rotation_deg=float(model.get("q_rotation_deg", plot.get("q_rotation_deg", 0.0))),
        fit_indices=fit_indices,
        band_indices=band_indices,
        n_orb=(int(n_orb_values[0]), int(n_orb_values[1])),
        nlow_state=nlow_state,
        bM_config=dict(model.get("bM", {})),
        harmonics_config=dict(harmonics),
        max_order={str(key): int(value) for key, value in dict(model.get("max_order", {})).items()},
        symmetry_map={str(key): list(value) for key, value in dict(symmetry_map).items()},
        coeff_tol=float(fit.get("coeff_tol", 1.0e-6)),
        compare_to_heff=bool(bands.get("compare_to_heff", True)),
        kpath_config=dict(kpath_config),
        band_slice=band_slice,
    )


def evaluate_vector_expression(value: Any, variables: Mapping[str, Any]) -> np.ndarray:
    """Evaluate a small vector expression such as ``bM1 + bM2`` or ``rot(q1, 240)``."""
    if isinstance(value, np.ndarray):
        return _as_vector(value)
    if isinstance(value, (list, tuple)):
        return _as_vector(np.array(value, dtype=float))
    if isinstance(value, Mapping):
        if "expr" not in value:
            raise ValueError(f"Vector mapping must contain expr, got {value!r}")
        value = value["expr"]
    if not isinstance(value, str):
        raise ValueError(f"Unsupported vector expression {value!r}")
    expr = ast.parse(value, mode="eval")
    result = _eval_ast(expr.body, variables)
    return _as_vector(result)


def _as_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"Expected a 2-vector, got shape {arr.shape}: {value!r}")
    return arr


def _as_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.shape != ():
        raise ValueError(f"Expected scalar, got shape {arr.shape}: {value!r}")
    return float(arr)


def _eval_ast(node: ast.AST, variables: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown vector expression symbol: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant in vector expression: {node.value!r}")
    if isinstance(node, ast.List):
        return np.array([_as_scalar(_eval_ast(elt, variables)) for elt in node.elts], dtype=float)
    if isinstance(node, ast.Tuple):
        return np.array([_as_scalar(_eval_ast(elt, variables)) for elt in node.elts], dtype=float)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand, variables)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return np.asarray(left) + np.asarray(right)
        if isinstance(node.op, ast.Sub):
            return np.asarray(left) - np.asarray(right)
        if isinstance(node.op, ast.Mult):
            return np.asarray(left) * np.asarray(right)
        if isinstance(node.op, ast.Div):
            return np.asarray(left) / np.asarray(right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "rot" and len(node.args) == 2:
            vec = evaluate_vector_expression_ast(node.args[0], variables)
            angle = _as_scalar(_eval_ast(node.args[1], variables))
            return rot(vec, angle)
        if node.func.id == "norm" and len(node.args) == 1:
            vec = evaluate_vector_expression_ast(node.args[0], variables)
            return float(np.linalg.norm(vec))
    raise ValueError(f"Unsupported vector expression syntax: {ast.dump(node)}")


def evaluate_vector_expression_ast(node: ast.AST, variables: Mapping[str, Any]) -> np.ndarray:
    return _as_vector(_eval_ast(node, variables))


def _build_variables(Q_set1: np.ndarray, bM1: np.ndarray, bM2: np.ndarray, model: Mapping[str, Any]) -> dict[str, Any]:
    variables: dict[str, Any] = {
        "zero": np.zeros(2, dtype=float),
        "bM1": np.asarray(bM1, dtype=float),
        "bM2": np.asarray(bM2, dtype=float),
        "sqrt3": float(np.sqrt(3.0)),
        "pi": float(np.pi),
        "bM_norm": float(np.linalg.norm(bM1)),
    }
    variables["q1"] = variables["bM_norm"] * np.array([0.0, 1.0 / np.sqrt(3.0)], dtype=float)
    variables["q3"] = rot(variables["q1"], 240.0)

    custom = model.get("vectors", {})
    if custom is None:
        return variables
    if not isinstance(custom, Mapping):
        raise ValueError("model.vectors must be a mapping")
    for key, expr in custom.items():
        variables[str(key)] = evaluate_vector_expression(expr, variables)
    return variables


def _harmonics_map(raw: Mapping[str, Any], variables: Mapping[str, Any]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key, value in raw.items():
        out[int(key)] = evaluate_vector_expression(value, variables)
    return out


def _load_kpoints(config: ConfiguredModel) -> np.ndarray:
    if config.kpoints_file is not None:
        kpoints = np.load(config.kpoints_file)
        return _validate_kpoints(kpoints)

    if not config.kpath_config:
        raise ValueError("Configured model requires kpoints_file or kpath section")
    kpath = config.kpath_config
    file_path = _resolve_path(kpath.get("file"), config.path.parent)
    if file_path is None:
        raise ValueError("kpath.file is required when kpoints_file is absent")
    tmat = np.asarray(kpath.get("tmat"), dtype=float)
    phase_deg = float(kpath.get("phase_deg", config.q_rotation_deg))
    segment_points = kpath.get("segment_points")
    output_file = _resolve_path(kpath.get("output_file"), config.path.parent)
    generated = generate_kpath_from_file(
        Tmat=tmat,
        file_path=file_path,
        phase_deg=phase_deg,
        segment_points=None if segment_points is None else int(segment_points),
        output_file_path=output_file,
    )
    return _validate_kpoints(generated.kpoints_2d)


def _validate_kpoints(kpoints: Any) -> np.ndarray:
    arr = np.asarray(kpoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk,2), got {arr.shape}")
    return arr


def _select_rows(arr: np.ndarray, indices: Sequence[int] | None) -> np.ndarray:
    if indices is None:
        return np.asarray(arr)
    idx = np.asarray(list(indices), dtype=int)
    return np.asarray(arr)[idx]


def _block_diag_heff(heff_list: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if not indices:
        raise ValueError("fit.indices must contain at least one k-point index")
    blocks = [np.asarray(heff_list[int(i)]) for i in indices]
    return scipy.linalg.block_diag(*blocks)


def build_moire_config_from_file(path: str | Path) -> tuple[MoireConfig, ConfiguredModel]:
    config = load_model_config(path)
    heff_list = np.load(config.heff_file, mmap_mode="r")
    Q_set1, Q_set2 = load_Q_sets_from_gvec_files(
        config.qset1_file,
        config.qset2_file,
        rotation_deg=config.q_rotation_deg,
    )

    bM1, bM2 = _build_bM_vectors(Q_set1, config)
    model_section = _model_section(config.raw)
    variables = _build_variables(Q_set1, bM1, bM2, model_section)
    harmonics = config.harmonics_config
    intra = _harmonics_map(dict(harmonics.get("intra", {})), variables)
    inter = _harmonics_map(dict(harmonics.get("inter", {})), variables)
    kpoints_all = _load_kpoints(config)
    band_kpoints = _select_rows(kpoints_all, config.band_indices)
    fit_kpoints = _select_rows(kpoints_all, config.fit_indices)

    moire_config = MoireConfig(
        Q_set1=Q_set1,
        Q_set2=Q_set2,
        n_orb1=config.n_orb[0],
        n_orb2=config.n_orb[1],
        nlow_state=list(config.nlow_state),
        bM1=bM1,
        bM2=bM2,
        intra_harmonics_map=intra,
        inter_harmonics_map=inter,
        max_order=dict(config.max_order),
        symmetry_map=dict(config.symmetry_map),
        kpoints=band_kpoints,
        kpoints_fit=fit_kpoints,
        heff=_block_diag_heff(heff_list, config.fit_indices),
        coeff_tol=config.coeff_tol,
        output_dir=config.output_dir,
    )
    return moire_config, config


def _build_bM_vectors(Q_set1: np.ndarray, config: ConfiguredModel) -> tuple[np.ndarray, np.ndarray]:
    raw = config.bM_config
    if raw.get("infer_from_q", False) or not raw:
        angle = float(raw.get("angle_deg", 60.0))
        return infer_bM_vectors_from_Q_set1(Q_set1, angle_deg=angle)
    variables = {"zero": np.zeros(2), "pi": float(np.pi), "sqrt3": float(np.sqrt(3.0))}
    if "bM1" not in raw:
        raise ValueError("model.bM must provide bM1 or infer_from_q")
    bM1 = evaluate_vector_expression(raw["bM1"], variables)
    variables["bM1"] = bM1
    if "bM2" in raw:
        bM2 = evaluate_vector_expression(raw["bM2"], variables)
    else:
        bM2 = rot(bM1, float(raw.get("angle_deg", 60.0)))
    return bM1, bM2


def compare_bands(
    model_eigvals: np.ndarray,
    heff_eigvals: np.ndarray,
    *,
    band_slice: Sequence[int] | None = None,
) -> dict[str, float | int | list[int] | None]:
    model = np.asarray(model_eigvals, dtype=float)
    heff = np.asarray(heff_eigvals)
    if heff.ndim == 3:
        heff = np.linalg.eigvalsh(heff)
    heff = np.asarray(heff, dtype=float)
    if model.ndim != 2 or heff.ndim != 2:
        raise ValueError(f"model/heff eigvals must be 2D after eigensolve, got {model.shape} and {heff.shape}")
    if model.shape[0] != heff.shape[0]:
        raise ValueError(f"k-point counts differ: model={model.shape[0]}, heff={heff.shape[0]}")
    nbands = min(model.shape[1], heff.shape[1])
    start = 0
    stop = nbands
    if band_slice is not None:
        if len(band_slice) != 2:
            raise ValueError(f"band_slice must be [start, stop], got {band_slice}")
        start, stop = int(band_slice[0]), int(band_slice[1])
    model_sel = np.sort(model, axis=1)[:, start:stop]
    heff_sel = np.sort(heff, axis=1)[:, start:stop]
    if model_sel.shape != heff_sel.shape:
        raise ValueError(f"selected band shapes differ: model={model_sel.shape}, heff={heff_sel.shape}")
    delta = model_sel - heff_sel
    rms = float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    return {
        "num_kpoints": int(delta.shape[0]),
        "num_bands": int(delta.shape[1]),
        "band_slice": [start, stop],
        "rms_error": rms,
        "max_abs_error": max_abs,
        "rms_error_mev": 1000.0 * rms,
        "max_abs_error_mev": 1000.0 * max_abs,
    }


def run_configured_model(path: str | Path) -> dict[str, Any]:
    moire_config, model_config = build_moire_config_from_file(path)
    output_dir = model_config.output_dir
    moire_config.output_dir = None
    try:
        results = run_end_to_end(moire_config)
    finally:
        moire_config.output_dir = output_dir
    eigvals = results["eigvals"]
    if isinstance(eigvals, tuple):
        eigvals_array = eigvals[0]
    else:
        eigvals_array = eigvals

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "eigvals.npy", np.asarray(eigvals_array))

    comparison = None
    if model_config.compare_to_heff:
        if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
            heff_eig = np.load(model_config.heff_eig_file)
        else:
            heff_eig = np.linalg.eigvalsh(np.load(model_config.heff_file, mmap_mode="r"))
        heff_selected = _select_rows(heff_eig, model_config.band_indices)
        comparison = compare_bands(eigvals_array, heff_selected, band_slice=model_config.band_slice)
        with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2)
    results["configured_model"] = model_config
    results["moire_config"] = moire_config
    results["comparison"] = comparison
    return results
