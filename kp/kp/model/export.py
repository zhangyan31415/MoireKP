from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .core import (
    ContinuumModelBuilder,
    ContinuumTerm,
    ContinuumTermKey,
    _compute_one_k,
    _prepare_band_state,
    clear_symmetry_caches,
)
from .pipeline import build_moire_config_from_file

SCHEMA_VERSION = "standalone-kp-model-v1"
EXPORTER_VERSION = "0.1"
P_MATCH_TOLERANCE = 1.0e-5
FORBIDDEN_PRODUCTION_OPERATION_NAMES = {"C2x", "C2y", "C2yT", "mirror_x", "mirror_y"}
DEFAULT_TOP_LEVEL_FILES = {"README.md", "MODEL.md", "evaluate.py", "model_data.npz"}
SUPPORTED_VALLEY_TYPES = {"K", "M", "Gamma"}
SUPPORTED_SPIN_CONVENTIONS = {"spin_up_projected", "spin_down_projected", "spinful", "spinless_effective", "spinless"}


@dataclass
class _StandaloneExport:
    model_json: dict[str, Any]
    model_data: dict[str, np.ndarray]
    readme: str
    model_doc: str
    evaluate_py: str
    debug_files: dict[str, bytes | str]


def export_standalone_model(
    model_output_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
    debug_files: bool = False,
) -> Path:
    """Export a fitted configured model as a minimal NumPy-only standalone package."""
    model_output = Path(model_output_dir).resolve()
    out = Path(output_dir).resolve()
    if not model_output.is_dir():
        raise FileNotFoundError(f"model_output_dir does not exist: {model_output}")
    in_place = out == model_output
    if out.exists() and any(out.iterdir()) and not force and not in_place:
        raise FileExistsError(f"output_dir already exists and is not empty: {out}")
    if out.exists() and force and not in_place:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stale_model_json = out / "model.json"
    if stale_model_json.exists() and stale_model_json.is_file():
        stale_model_json.unlink()

    package = _build_standalone_export(model_output, include_debug=debug_files)
    _assert_clean_text("README.md", package.readme)
    _assert_clean_text("MODEL.md", package.model_doc)

    (out / "README.md").write_text(package.readme, encoding="utf-8")
    (out / "MODEL.md").write_text(package.model_doc, encoding="utf-8")
    (out / "evaluate.py").write_text(package.evaluate_py, encoding="utf-8")
    _write_npz(out / "model_data.npz", package.model_data)

    if debug_files:
        for rel, payload in package.debug_files.items():
            path = out / "debug" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                path.write_text(payload, encoding="utf-8")

    if not in_place:
        extra = {path.name for path in out.iterdir()} - DEFAULT_TOP_LEVEL_FILES - ({"debug"} if debug_files else set())
        if extra:
            raise RuntimeError(f"unexpected files in standalone export: {sorted(extra)}")
    return out


def export_all_standalone_models(
    examples_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    debug_files: bool = False,
    dry_run: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Export or inventory all standalone-capable model outputs under examples_root."""
    examples = Path(examples_root)
    out_root = Path(output_root)
    candidates = sorted(path for path in examples.glob("**/kp/outputs/*/*/model") if path.is_dir())
    report: dict[str, list[dict[str, Any]]] = {"exportable": [], "blocked": [], "exported": []}
    for model_output in candidates:
        profile = model_output.parent.parent.name
        q_shell = model_output.parent.name
        out_dir = out_root / f"{profile}_{q_shell}_numpy"
        try:
            if dry_run:
                _build_standalone_export(model_output.resolve(), include_debug=debug_files)
                report["exportable"].append(
                    {"model_output_dir": str(model_output.resolve()), "output_dir": str(out_dir)}
                )
            else:
                path = export_standalone_model(model_output, out_dir, force=force, debug_files=debug_files)
                report["exported"].append(
                    {"model_output_dir": str(model_output.resolve()), "output_dir": str(path)}
                )
        except Exception as exc:  # noqa: BLE001 - inventory must report every blocker.
            report["blocked"].append(
                {
                    "model_output_dir": str(model_output.resolve()),
                    "output_dir": str(out_dir),
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
    return report


def _build_standalone_export(model_output: Path, *, include_debug: bool) -> _StandaloneExport:
    clear_symmetry_caches()
    _ORBIT_RECORD_CACHE.clear()
    case_id = model_output.name
    config_path = _infer_model_config_path(model_output)

    active_terms_path = model_output / "active_terms.json"
    if not active_terms_path.exists():
        raise FileNotFoundError(f"active_terms.json is required: {active_terms_path}")
    active_terms_text = active_terms_path.read_text(encoding="utf-8")
    active_terms = json.loads(active_terms_text)
    if not isinstance(active_terms, list) or not active_terms:
        raise ValueError(f"active_terms.json must contain a non-empty list: {active_terms_path}")

    moire_config, model_config = build_moire_config_from_file(config_path)
    _validate_supported_export_model(model_config, moire_config)

    qsets = _qset_arrays(moire_config)
    n_orb_by_qset = _n_orb_by_qset(moire_config)
    basis_blocks = _basis_blocks(qsets, n_orb_by_qset)
    basis_metadata = _basis_row_metadata(qsets, n_orb_by_qset, basis_blocks)
    reciprocal_basis = _model_reciprocal_basis(moire_config)
    dim = sum(int(block["dim"]) for block in basis_blocks)
    exactified = _load_exactified_matrices(model_config, dim)
    semantic_terms, runtime_terms = _runtime_terms_from_active_terms(active_terms, moire_config)
    operator_data = _expand_operator_recipe(runtime_terms, moire_config, dim)

    reference_eigvals = _load_required_array(model_output / "eigvals.npy", "model reference eigenvalues")
    reference_kpoints = np.asarray(moire_config.kpoints, dtype=float)
    if reference_kpoints.ndim != 2 or reference_kpoints.shape[1] != 2:
        raise ValueError(f"reference kpoints must have shape (Nk, 2), got {reference_kpoints.shape}")
    if reference_eigvals.shape[0] != reference_kpoints.shape[0]:
        raise ValueError(
            f"reference eigvals/kpoints length mismatch: {reference_eigvals.shape[0]} vs {reference_kpoints.shape[0]}"
        )
    reference_heff_eig = None
    if model_config.heff_eig_file is not None and model_config.heff_eig_file.exists():
        reference_heff_eig = np.load(model_config.heff_eig_file, allow_pickle=False)

    run_summary = _load_json_if_exists(model_output / "run_summary.json")
    if not isinstance(run_summary, Mapping):
        run_summary = {}
    diagnostics_dir = model_output / "diagnostics"
    comparison = (
        _load_json_if_exists(model_output / "comparison.json")
        or _load_json_if_exists(diagnostics_dir / "comparison.json")
        or run_summary.get("comparison")
    )
    comparison_plot = (
        _load_json_if_exists(model_output / "comparison_plot.json")
        or _load_json_if_exists(diagnostics_dir / "comparison_plot.json")
        or run_summary.get("plot_comparison")
    )
    validation = _validation_summary(
        comparison,
        comparison_plot,
        reference_heff_eig is not None,
        reference_heff_eig_shape=None if reference_heff_eig is None else tuple(reference_heff_eig.shape),
    )

    model_data: dict[str, np.ndarray] = {
        "qset1": qsets["qset1"],
        "qset2": qsets["qset2"],
        "qset_layer1": qsets["qset1"],
        "qset_layer2": qsets["qset2"],
        "operator_term_index": operator_data["term_index"],
        "operator_row": operator_data["row"],
        "operator_col": operator_data["col"],
        "operator_mz": operator_data["mz"],
        "operator_mz_star": operator_data["mz_star"],
        "operator_q_center": operator_data["q_center"],
        "operator_prefactor_real": operator_data["prefactor_real"],
        "operator_prefactor_imag": operator_data["prefactor_imag"],
        "term_r_value_real": np.asarray([float(term["r_value_real"]) for term in semantic_terms], dtype=float),
        "term_r_value_imag": np.asarray([float(term["r_value_imag"]) for term in semantic_terms], dtype=float),
        "dimension_dim": np.asarray(dim, dtype=np.int64),
        "runtime_hermitianize_before_eigvalsh": np.asarray(True, dtype=np.bool_),
        "reference_kpoints": reference_kpoints,
        "reference_eigvals": np.asarray(reference_eigvals, dtype=float),
        "basis_qset_id": basis_metadata["basis_qset_id"],
        "basis_q_index": basis_metadata["basis_q_index"],
        "basis_orbital": basis_metadata["basis_orbital"],
        "basis_q_vector": basis_metadata["basis_q_vector"],
        "basis_block_offset": basis_metadata["basis_block_offset"],
        "basis_block_q_count": basis_metadata["basis_block_q_count"],
        "basis_block_n_orb": basis_metadata["basis_block_n_orb"],
        "model_reciprocal_basis": reciprocal_basis,
    }
    for name, matrix in exactified.items():
        model_data[f"exactified_{name}"] = matrix
    if reference_heff_eig is not None:
        model_data["reference_heff_eig"] = np.asarray(reference_heff_eig, dtype=float)

    operations = _portable_operations(model_config, exactified)
    coordinate_payload = _resolve_standalone_kpath_metadata(model_output, model_config, moire_config)
    array_hashes, combined_array_hash = _model_data_array_hashes(model_data)
    model_id = _model_name(case_id, model_config)
    model_json = {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "model_id": model_id,
        "model_name": model_id,
        "material": {"name": str(model_config.source_raw.get("material", {}).get("name", ""))},
        "valley": str(model_config.raw.get("valley", model_config.valley_model.get("active_valleys", ["K1"])[0])),
        "spin_convention": str(model_config.valley_model.get("spin_convention", "")),
        "valley_model": _json_safe(model_config.valley_model),
        "energy_unit": str(model_config.source_raw.get("material", {}).get("energy_unit", "eV")),
        "dimension": {
            "dim": int(dim),
            "n_orb": {"qset1": int(n_orb_by_qset["qset1"]), "qset2": int(n_orb_by_qset["qset2"])},
            "q_count": {"qset1": int(len(qsets["qset1"])), "qset2": int(len(qsets["qset2"]))},
            "q_count_by_qset": {"qset1": int(len(qsets["qset1"])), "qset2": int(len(qsets["qset2"]))},
            "n_orb_by_qset": {key: int(value) for key, value in n_orb_by_qset.items()},
            "basis_blocks": basis_blocks,
        },
        "basis": {
            "order": "qset-slot-major, orbital-major within qset slot, q-index fastest",
            "index_formula": {
                "qset1": "index = basis_blocks['qset1'].offset + orbital_zero_based * Nq1 + q_index",
                "qset2": "index = basis_blocks['qset2'].offset + orbital_zero_based * Nq2 + q_index",
            },
        },
        "qsets": [
            {"name": "qset1", "array_key": "qset1", "slot": 1, "friendly_array_key": "qset_layer1"},
            {"name": "qset2", "array_key": "qset2", "slot": 2, "friendly_array_key": "qset_layer2"},
        ],
        "sectors": _portable_sectors(getattr(moire_config, "sectors", [])),
        "coordinate_convention": coordinate_payload["coordinate_convention"],
        "high_symmetry_points": coordinate_payload["high_symmetry_points"],
        "default_kpath": coordinate_payload["default_kpath"],
        "points_per_segment": coordinate_payload["points_per_segment"],
        "default_band_slice": coordinate_payload["default_band_slice"],
        "energy_reference": _energy_reference(validation),
        "runtime": {"hermitianize_before_eigvalsh": True},
        "p_match_tolerance": P_MATCH_TOLERANCE,
        "runtime_recipe": {
            "array_file": "model_data.npz",
            "kind": "preexpanded_global_z_zstar_polynomial",
            "formula": (
                "H[row,col] += r_real[t] * prefactor_real[c] * z(k)**mz[c] * z*(k)**mz_star[c] "
                "+ r_imag[t] * prefactor_imag[c] * z(k)**mz[c] * z*(k)**mz_star[c]"
            ),
        },
        "terms": semantic_terms,
        "operations": operations,
        "validation": validation,
        "hashes": {
            "active_terms_sha256": hashlib.sha256(active_terms_text.encode("utf-8")).hexdigest(),
            "model_data_arrays": array_hashes,
            "model_data_arrays_combined_sha256": combined_array_hash,
        },
    }
    model_json["hashes"]["model_json_canonical_sha256"] = _canonical_json_hash(
        _model_json_for_hash(model_json)
    )
    _assert_strict_json("model.json", model_json)
    _assert_no_forbidden_production_operations(model_json["operations"])

    readme = _render_readme(model_json)
    model_doc = _render_model_doc(model_json, model_data)
    debug_payloads: dict[str, bytes | str] = {}
    if include_debug:
        debug_payloads = _debug_payloads(
            model_json=model_json,
            active_terms=semantic_terms,
            operations=operations,
            comparison=comparison or {},
            model_data=model_data,
        )

    return _StandaloneExport(
        model_json=model_json,
        model_data=model_data,
        readme=readme,
        model_doc=model_doc,
        evaluate_py=_evaluate_py_template(model_json),
        debug_files=debug_payloads,
    )


def _infer_model_config_path(model_output: Path) -> Path:
    run_summary = _load_json_if_exists(model_output / "run_summary.json")
    if isinstance(run_summary, Mapping):
        raw_path = run_summary.get("model_config_path") or run_summary.get("config_path")
        if raw_path:
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = (model_output / candidate).resolve()
            if candidate.exists():
                return candidate.resolve()

    if model_output.name == "model" and model_output.parent.parent.parent.name == "outputs":
        q_shell = model_output.parent.name
        profile = model_output.parent.parent.name
        kp_root = model_output.parent.parent.parent.parent
        for filename in (f"{profile}_{q_shell}.yaml", f"{profile}.yaml"):
            candidate = kp_root / "configs" / filename
            if candidate.exists():
                return candidate.resolve()

    nearby = _find_model_config_near_output(model_output)
    if nearby is not None:
        return nearby

    raise FileNotFoundError(
        "Unable to infer model config path. Expected model_output_dir like "
        "outputs/<profile>/<q_shell>/model or a model YAML next to the output directory "
        "with output.dir pointing at it."
    )


def _find_model_config_near_output(model_output: Path) -> Path | None:
    search_dirs = [model_output.parent]
    if model_output.parent.name == "model":
        search_dirs.append(model_output.parent.parent / "configs" / "model")
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for candidate in sorted([*search_dir.glob("*.yaml"), *search_dir.glob("*.yml")]):
            try:
                raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, Mapping):
                continue
            output_cfg = raw.get("output")
            if not isinstance(output_cfg, Mapping):
                continue
            raw_dir = output_cfg.get("dir")
            if raw_dir is None:
                continue
            configured_output = Path(str(raw_dir))
            if not configured_output.is_absolute():
                configured_output = (candidate.parent / configured_output).resolve()
            else:
                configured_output = configured_output.resolve()
            if configured_output == model_output.resolve():
                return candidate.resolve()
    return None


def _validate_supported_export_model(model_config: Any, moire_config: Any) -> None:
    valley_type = str(model_config.valley_model.get("valley_type", ""))
    mode = str(model_config.valley_model.get("mode", ""))
    spin = str(model_config.valley_model.get("spin_convention", ""))
    if valley_type not in SUPPORTED_VALLEY_TYPES:
        raise NotImplementedError(f"standalone export does not support valley_type={valley_type!r}")
    if mode != "single_valley":
        raise NotImplementedError(f"standalone export currently supports only single_valley mode, got {mode!r}")
    if spin not in SUPPORTED_SPIN_CONVENTIONS:
        raise NotImplementedError(f"standalone export does not support spin_convention={spin!r}")
    qsets = _qset_arrays(moire_config)
    for name, array in qsets.items():
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    n_orb = _n_orb_by_qset(moire_config)
    if any(value < 0 for value in n_orb.values()):
        raise ValueError(f"n_orb values must be non-negative, got {n_orb}")
    if sum(qsets[key].shape[0] * n_orb[key] for key in ("qset1", "qset2")) <= 0:
        raise ValueError("standalone export requires at least one active basis state")
    kpoints = np.asarray(getattr(moire_config, "kpoints", None), dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 2:
        raise ValueError(f"reference kpoints must have shape (Nk, 2), got {kpoints.shape}")


def _resolve_standalone_kpath_metadata(model_output: Path, model_config: Any, moire_config: Any) -> dict[str, Any]:
    """Resolve explicit user-facing coordinate and default kpath metadata."""
    for filename in ("standalone_export.json", "standalone_export.yaml", "standalone_export.yml"):
        payload = _load_mapping_file_if_exists(model_output / filename)
        if payload is not None:
            resolved = _standalone_payload_from_mapping(
                payload, model_config, moire_config, source=f"model output {filename}"
            )
            if resolved is not None:
                return resolved
    raw = getattr(model_config, "raw", {})
    if isinstance(raw, Mapping) and isinstance(raw.get("standalone_export"), Mapping):
        resolved = _standalone_payload_from_mapping(
            raw["standalone_export"], model_config, moire_config, source="model config standalone_export"
        )
        if resolved is not None:
            return resolved
    from_kpath = _standalone_payload_from_kpath_config(model_config, moire_config)
    if from_kpath is not None:
        return from_kpath
    manifest = _load_mapping_file_if_exists(model_config.path.parents[1] / "data-manifest.yaml")
    if manifest is not None and isinstance(manifest.get("standalone_export"), Mapping):
        resolved = _standalone_payload_from_mapping(
            manifest["standalone_export"], model_config, moire_config, source="example manifest standalone_export"
        )
        if resolved is not None:
            return resolved
    for filename in ("standalone_export.yaml", "standalone_export.yml", "standalone_export.json"):
        payload = _load_mapping_file_if_exists(model_config.path.parent / filename)
        if payload is not None:
            resolved = _standalone_payload_from_mapping(
                payload, model_config, moire_config, source=f"model config directory {filename}"
            )
            if resolved is not None:
                return resolved

    raise ValueError(
        "standalone export requires explicit coordinate convention, high_symmetry_points, and default_kpath "
        f"for {model_config.path}; add standalone_export.yaml near the model config or a complete kpath section"
    )


def _load_mapping_file_if_exists(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _standalone_payload_from_mapping(
    payload: Mapping[str, Any],
    model_config: Any,
    moire_config: Any,
    *,
    source: str,
) -> dict[str, Any] | None:
    coord = payload.get("coordinate_convention")
    hsp = payload.get("high_symmetry_points")
    kpath = payload.get("default_kpath")
    if coord is None and hsp is None and kpath is None:
        return None
    if not isinstance(coord, Mapping):
        raise ValueError(f"{source} is missing coordinate_convention mapping")
    if not isinstance(hsp, Mapping) or not hsp:
        raise ValueError(f"{source} is missing high_symmetry_points mapping")
    if not isinstance(kpath, Sequence) or isinstance(kpath, (str, bytes)) or not kpath:
        raise ValueError(f"{source} is missing default_kpath list")
    return _normalise_standalone_kpath_payload(
        coord,
        hsp,
        kpath,
        payload.get("points_per_segment", 80),
        payload.get("default_band_slice", None),
        moire_config,
        source=source,
    )


def _standalone_payload_from_kpath_config(model_config: Any, moire_config: Any) -> dict[str, Any] | None:
    kpath = getattr(model_config, "kpath_config", {})
    if not isinstance(kpath, Mapping) or not kpath:
        return None
    file_raw = kpath.get("file")
    if file_raw is None:
        return None
    path = Path(str(file_raw))
    if not path.is_absolute():
        path = (model_config.path.parent / path).resolve()
    if not path.exists():
        return None
    hsp, labels, default_points = _parse_line_mode_kpath_file(path)
    if not hsp or len(labels) < 2:
        return None
    points_per_segment = kpath.get("segment_points", default_points or 80)
    coord = {
        "type": "fractional_model_basis",
        "k_units": "fractional coordinates in bM1/bM2 basis",
        "q_units": "same as k",
        "hsp_coordinates_are": "fractional_model_basis",
    }
    return _normalise_standalone_kpath_payload(
        coord,
        hsp,
        labels,
        points_per_segment,
        getattr(model_config, "band_slice", None),
        moire_config,
        source=f"kpath file {path.name}",
    )


def _parse_line_mode_kpath_file(path: Path) -> tuple[dict[str, list[float]], list[str], int | None]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    default_points = None
    for line in lines[:4]:
        try:
            default_points = int(line.strip().split()[0])
            break
        except Exception:
            continue
    hsp: dict[str, list[float]] = {}
    sequence: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coords = [float(parts[0]), float(parts[1])]
        except ValueError:
            continue
        label = _normalise_kpath_label(parts[3])
        hsp.setdefault(label, coords)
        sequence.append(label)
    collapsed: list[str] = []
    for label in sequence:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)
    return hsp, collapsed, default_points


def _normalise_kpath_label(label: Any) -> str:
    text = str(label).strip()
    if text.lower() in {"gamma", "gam", "g"}:
        return "G"
    return text


def _normalise_standalone_kpath_payload(
    coord: Mapping[str, Any],
    hsp: Mapping[str, Any],
    kpath: Sequence[Any],
    points_per_segment: Any,
    default_band_slice: Any,
    moire_config: Any,
    *,
    source: str,
) -> dict[str, Any]:
    coord_type = str(coord.get("type", ""))
    if coord_type != "fractional_model_basis":
        raise ValueError(f"{source} coordinate_convention.type must be 'fractional_model_basis', got {coord_type!r}")
    hsp_out: dict[str, list[float]] = {}
    for raw_label, raw_point in hsp.items():
        arr = np.asarray(raw_point, dtype=float).ravel()
        if arr.shape[0] < 2:
            raise ValueError(f"{source} high_symmetry_points.{raw_label} must have at least two coordinates")
        hsp_out[_normalise_kpath_label(raw_label)] = [float(arr[0]), float(arr[1])]
    kpath_out = [_normalise_kpath_label(label) for label in kpath]
    missing = [label for label in kpath_out if label not in hsp_out]
    if missing:
        raise ValueError(f"{source} default_kpath references missing high_symmetry_points labels: {missing}")
    segment_count = int(points_per_segment)
    if segment_count <= 0:
        raise ValueError(f"{source} points_per_segment must be positive, got {points_per_segment!r}")
    bM1 = np.asarray(getattr(moire_config, "bM1", None), dtype=float).ravel()
    bM2 = np.asarray(getattr(moire_config, "bM2", None), dtype=float).ravel()
    if bM1.shape[0] < 2 or bM2.shape[0] < 2:
        raise ValueError(f"{source} requires bM1/bM2 to record fractional_model_basis coordinates")
    coord_out = {
        "type": "fractional_model_basis",
        "k_units": str(coord.get("k_units", "fractional coordinates in bM1/bM2 basis")),
        "q_units": str(coord.get("q_units", "same as k")),
        "bM1": [float(bM1[0]), float(bM1[1])],
        "bM2": [float(bM2[0]), float(bM2[1])],
        "hsp_coordinates_are": str(coord.get("hsp_coordinates_are", "fractional_model_basis")),
        "source": source,
    }
    if coord_out["hsp_coordinates_are"] != coord_out["type"]:
        raise ValueError(
            f"{source} hsp_coordinates_are={coord_out['hsp_coordinates_are']!r} does not match "
            f"coordinate_convention.type={coord_out['type']!r}"
        )
    band_slice = None if default_band_slice is None else [int(x) for x in default_band_slice]
    if band_slice is not None and len(band_slice) != 2:
        raise ValueError(f"{source} default_band_slice must be null or [start, stop], got {default_band_slice!r}")
    return {
        "coordinate_convention": coord_out,
        "high_symmetry_points": hsp_out,
        "default_kpath": kpath_out,
        "points_per_segment": segment_count,
        "default_band_slice": band_slice,
    }


def _qset_arrays(moire_config: Any) -> dict[str, np.ndarray]:
    if getattr(moire_config, "Q_set1", None) is None or getattr(moire_config, "Q_set2", None) is None:
        raise ValueError("standalone export requires two-qset MoireConfig with Q_set1 and Q_set2")
    return {
        "qset1": np.asarray(moire_config.Q_set1, dtype=float),
        "qset2": np.asarray(moire_config.Q_set2, dtype=float),
    }


def _n_orb_by_qset(moire_config: Any) -> dict[str, int]:
    return {"qset1": int(moire_config.n_orb1), "qset2": int(moire_config.n_orb2)}


def _basis_blocks(qsets: Mapping[str, np.ndarray], n_orb_by_qset: Mapping[str, int]) -> list[dict[str, int | str]]:
    blocks: list[dict[str, int | str]] = []
    offset = 0
    for qset_name in ("qset1", "qset2"):
        q_count = int(np.asarray(qsets[qset_name]).shape[0])
        n_orb = int(n_orb_by_qset[qset_name])
        dim = q_count * n_orb
        blocks.append({"qset": qset_name, "offset": int(offset), "q_count": q_count, "n_orb": n_orb, "dim": int(dim)})
        offset += dim
    return blocks


def _basis_row_metadata(
    qsets: Mapping[str, np.ndarray],
    n_orb_by_qset: Mapping[str, int],
    basis_blocks: Sequence[Mapping[str, int | str]],
) -> dict[str, np.ndarray]:
    qset_ids: list[int] = []
    q_indices: list[int] = []
    orbitals: list[int] = []
    q_vectors: list[np.ndarray] = []
    block_offsets: list[int] = []
    block_q_counts: list[int] = []
    block_n_orb: list[int] = []
    for slot, qset_name in enumerate(("qset1", "qset2"), start=1):
        qset = np.asarray(qsets[qset_name], dtype=float)
        n_orb = int(n_orb_by_qset[qset_name])
        block = next(item for item in basis_blocks if item["qset"] == qset_name)
        block_offsets.append(int(block["offset"]))
        block_q_counts.append(int(qset.shape[0]))
        block_n_orb.append(n_orb)
        for orbital in range(n_orb):
            for q_index, q_vector in enumerate(qset):
                qset_ids.append(slot)
                q_indices.append(int(q_index))
                orbitals.append(int(orbital))
                q_vectors.append(np.asarray(q_vector, dtype=float))
    if q_vectors:
        q_vector_array = np.vstack(q_vectors).astype(float)
    else:
        q_vector_array = np.zeros((0, 2), dtype=float)
    return {
        "basis_qset_id": np.asarray(qset_ids, dtype=np.int64),
        "basis_q_index": np.asarray(q_indices, dtype=np.int64),
        "basis_orbital": np.asarray(orbitals, dtype=np.int64),
        "basis_q_vector": q_vector_array,
        "basis_block_offset": np.asarray(block_offsets, dtype=np.int64),
        "basis_block_q_count": np.asarray(block_q_counts, dtype=np.int64),
        "basis_block_n_orb": np.asarray(block_n_orb, dtype=np.int64),
    }


def _model_reciprocal_basis(moire_config: Any) -> np.ndarray:
    b1 = np.asarray(getattr(moire_config, "bM1", None), dtype=float).ravel()
    b2 = np.asarray(getattr(moire_config, "bM2", None), dtype=float).ravel()
    if b1.shape[0] < 2 or b2.shape[0] < 2:
        raise ValueError("standalone export requires bM1/bM2 to save model_reciprocal_basis")
    return np.asarray([[float(b1[0]), float(b1[1])], [float(b2[0]), float(b2[1])]], dtype=float)


def _portable_sectors(sectors: Any) -> list[dict[str, Any]]:
    if not isinstance(sectors, Sequence) or isinstance(sectors, (str, bytes)):
        return []
    rows = []
    for item in sectors:
        if not isinstance(item, Mapping):
            continue
        row = {
            "name": str(item.get("name", "")),
            "qset": str(item.get("qset", "")),
            "q_offset": [float(x) for x in np.asarray(item.get("q_offset", [0.0, 0.0]), dtype=float).ravel()[:2]],
            "n_orb": int(item.get("n_orb", 0)),
        }
        rows.append(row)
    return rows


def _load_exactified_matrices(model_config: Any, dim: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    source_path = model_config.symmetry_source_config.get("path")
    if source_path is None:
        raise FileNotFoundError("symmetry_source.path is required for standalone export")
    symm_dir = Path(str(source_path))
    if not symm_dir.is_absolute():
        symm_dir = (model_config.path.parent / symm_dir).resolve()
    if not symm_dir.is_dir():
        raise FileNotFoundError(f"symmetry_source.path does not exist: {symm_dir}")

    requested = _production_operation_names(model_config, symm_dir)
    available: dict[str, Path] = {}
    for path in symm_dir.glob("exactified_*.npy"):
        raw_name = path.stem.removeprefix("exactified_")
        available.setdefault(_operation_family_name(raw_name), path)
    packed_path = symm_dir / "representations.npz"
    packed_keys: dict[str, str] = {}
    if packed_path.exists():
        with np.load(packed_path, allow_pickle=False) as packed:
            for key in packed.files:
                if str(key).startswith("__"):
                    continue
                packed_keys.setdefault(_operation_family_name(str(key)), str(key))
    names = requested or sorted(available)
    if not names and packed_keys:
        names = sorted(packed_keys)
    for name in names:
        if name in FORBIDDEN_PRODUCTION_OPERATION_NAMES:
            raise ValueError(f"forbidden production operation name in standalone export: {name}")
        path = available.get(name, symm_dir / f"exactified_{name}.npy")
        if path.exists():
            matrix = np.load(path, allow_pickle=False)
            if matrix.shape != (dim, dim):
                raise ValueError(f"{path.name} shape {matrix.shape} does not match model dim {dim}")
            out[name] = np.asarray(matrix, dtype=np.complex128)
        elif name in packed_keys:
            with np.load(packed_path, allow_pickle=False) as packed:
                matrix = np.asarray(packed[packed_keys[name]], dtype=np.complex128)
            if matrix.shape != (dim, dim):
                raise ValueError(f"{packed_path.name}:{packed_keys[name]} shape {matrix.shape} does not match model dim {dim}")
            out[name] = matrix
        else:
            raise FileNotFoundError(f"exactified production symmetry matrix is required: {path}")
    return out


def _production_operation_names(model_config: Any, symm_dir: Path) -> list[str]:
    names: list[str] = []

    def add_from_record(record: Any) -> None:
        if not isinstance(record, Mapping):
            return
        raw_name = str(record.get("canonical_operation", record.get("name", record.get("operation", ""))))
        name = _operation_family_name(raw_name)
        if name and name not in names:
            names.append(name)

    registry = _load_json_if_exists(model_config.output_dir / "operation_registry.json") if model_config.output_dir else None
    if isinstance(registry, list):
        for item in registry:
            add_from_record(item)
    for item in model_config.symmetry_source_metadata.get("operations", []):
        add_from_record(item)
    configured = model_config.symmetry_source_config.get("operations", [])
    if isinstance(configured, Mapping):
        configured = list(configured.values())
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        for item in configured:
            if isinstance(item, str):
                name = _operation_family_name(item)
                if name and name not in names:
                    names.append(name)
            else:
                add_from_record(item)
    if not names:
        manifest = _load_json_if_exists(symm_dir / "manifest.json")
        if isinstance(manifest, Mapping):
            for item in manifest.get("operations", []):
                add_from_record(item)
    return names


def _operation_family_name(name: str) -> str:
    raw = str(name).strip()
    if raw in {"C2x", "C2y", "mirror_x", "mirror_y"}:
        return "C2"
    if raw == "C2yT":
        return "C2T"
    if raw in {"T", "time_reversal"}:
        return "TR"
    return raw


def _operation_aliases(name: str, raw_operation: str | None = None) -> list[str]:
    family = _operation_family_name(name)
    aliases: list[str] = []
    if family == "TR":
        aliases.extend(["T", "time_reversal"])
    if raw_operation:
        aliases.append(str(raw_operation))
    aliases.append(str(name))
    out: list[str] = []
    for item in aliases:
        if item and item not in out:
            out.append(item)
    return out


def _runtime_terms_from_active_terms(active_terms: list[Mapping[str, Any]], moire_config: Any) -> tuple[list[dict[str, Any]], list[ContinuumTerm]]:
    semantic: list[dict[str, Any]] = []
    runtime: list[ContinuumTerm] = []
    for index, row in enumerate(active_terms):
        key_data = row.get("key")
        if not isinstance(key_data, Mapping):
            raise ValueError(f"active term {index} is missing key")
        key = _term_key_from_dict(key_data)
        metadata = _compact_term_metadata(row.get("registry_metadata", {}))
        sym_ops = row.get("symmetry_ops", [])
        if not isinstance(sym_ops, Sequence) or isinstance(sym_ops, (str, bytes)):
            raise ValueError(f"active term {index} has invalid symmetry_ops")
        operation_names = [str(op.get("name")) for op in sym_ops if isinstance(op, Mapping) and op.get("name") is not None]
        y_basis = ContinuumModelBuilder.make_Y_basis_function(
            key,
            np.asarray(moire_config.Q_set1, dtype=float),
            np.asarray(moire_config.Q_set2, dtype=float),
            int(moire_config.n_orb1),
            int(moire_config.n_orb2),
            tol=P_MATCH_TOLERANCE,
        )
        term = ContinuumTerm(
            key=key,
            Y_basis=y_basis,
            r_value_real=float(np.real(row.get("r_value_real", 0.0))),
            r_value_imag=float(np.real(row.get("r_value_imag", 0.0))),
            active=True,
            tag=str(row.get("tag", "")),
            symmetry_ops=[dict(op) for op in sym_ops if isinstance(op, Mapping)],
            registry_metadata=metadata,
        )
        runtime.append(term)
        semantic.append(
            {
                "index": index,
                "key": _term_key_to_json(key),
                "tag": term.tag,
                "r_value_real": term.r_value_real,
                "r_value_imag": term.r_value_imag,
                "operation_names": operation_names,
                "metadata": metadata,
            }
        )
    return semantic, runtime


def _term_key_from_dict(data: Mapping[str, Any]) -> ContinuumTermKey:
    return ContinuumTermKey(
        int(data["Mz"]),
        int(data["Mz_star"]),
        int(data["layer_from"]),
        int(data["layer_to"]),
        int(data["orbital_from"]),
        int(data["orbital_to"]),
        tuple(float(x) for x in data["p"]),
    )


def _term_key_to_json(key: ContinuumTermKey) -> dict[str, Any]:
    return {
        "Mz": int(key.Mz),
        "Mz_star": int(key.Mz_star),
        "layer_from": int(key.layer_from),
        "layer_to": int(key.layer_to),
        "orbital_from": int(key.orbital_from),
        "orbital_to": int(key.orbital_to),
        "p": [float(x) for x in key.p],
    }


def _compact_term_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    keep = (
        "term_name",
        "term_kind",
        "sector_pair",
        "orbital_pair",
        "harmonic_id",
        "harmonic_kind",
        "harmonic_vector",
        "harmonics_source",
        "monomial",
        "coefficient_unit",
        "coefficient_role",
    )
    return {key: _json_safe(metadata[key]) for key in keep if key in metadata}


_DENSE_ACTION_ENTRY_TOL = 1.0e-13
_ORBIT_RECORD_CACHE: dict[tuple[int, tuple[Any, ...]], list[tuple[np.ndarray, Mapping[str, Any] | None]]] = {}


def _expand_operator_recipe(runtime_terms: Sequence[ContinuumTerm], moire_config: Any, dim: int) -> dict[str, np.ndarray]:
    term_index: list[int] = []
    row: list[int] = []
    col: list[int] = []
    mz_values: list[int] = []
    mz_star_values: list[int] = []
    q_center: list[tuple[float, float]] = []
    prefactor_real: list[complex] = []
    prefactor_imag: list[complex] = []
    monomial_transform_cache: dict[
        tuple[tuple[float, ...], tuple[float, float], int, int],
        tuple[int, int, tuple[float, float], complex],
    ] = {}

    def cached_transform_monomial(
        transform: np.ndarray,
        q_base: np.ndarray,
        mz: int,
        mz_star: int,
    ) -> tuple[int, int, tuple[float, float], complex]:
        transform_key = tuple(float(x) for x in np.asarray(transform, dtype=float).ravel())
        q_key = (float(q_base[0]), float(q_base[1]))
        cache_key = (transform_key, q_key, int(mz), int(mz_star))
        cached = monomial_transform_cache.get(cache_key)
        if cached is not None:
            return cached
        mz_new, mz_star_new, q_new, transform_prefactor = _transform_monomial(
            transform,
            q_base,
            mz,
            mz_star,
        )
        cached = (
            int(mz_new),
            int(mz_star_new),
            (float(q_new[0]), float(q_new[1])),
            complex(transform_prefactor),
        )
        monomial_transform_cache[cache_key] = cached
        return cached

    for idx, term in enumerate(runtime_terms):
        if not hasattr(term.Y_basis, "eval_sparse"):
            raise NotImplementedError("standalone export requires sparse term basis metadata")

        rows = np.asarray(getattr(term.Y_basis, "_moire_sparse_rows"), dtype=int)
        cols = np.asarray(getattr(term.Y_basis, "_moire_sparse_cols"), dtype=int)
        row_q_idx = np.asarray(getattr(term.Y_basis, "_moire_sparse_row_q_idx"), dtype=int)
        q_rows = np.asarray(getattr(term.Y_basis, "_moire_sparse_Q_rows"), dtype=float)
        hermitize_in_basis = bool(getattr(term.Y_basis, "_moire_sparse_hermitize_in_basis", False))
        if rows.size == 0:
            continue

        orbit_records = _orbit_records(term.symmetry_ops, moire_config.symmetry_gen)
        base_components = [(int(term.key.Mz), int(term.key.Mz_star), 1.0 + 0.0j)]
        if hermitize_in_basis:
            base_components.append((int(term.key.Mz_star), int(term.key.Mz), 1.0 + 0.0j))

        term_contributions: list[tuple[int, int, int, int, np.ndarray, complex, complex]] = []
        for transform, action in orbit_records:
            for sparse_pos, r_out, c_out, matrix_factor, is_anti_total in _iter_transformed_sparse_entries(
                action,
                rows,
                cols,
            ):
                q_base = q_rows[row_q_idx[sparse_pos]]
                for mz_base, mz_star_base, component_prefactor in base_components:
                    mz_new, mz_star_new, q_new, transform_prefactor = cached_transform_monomial(
                        transform,
                        q_base,
                        mz_base,
                        mz_star_base,
                    )
                    prefactor = component_prefactor * transform_prefactor
                    if is_anti_total:
                        prefactor = np.conjugate(prefactor)
                        mz_new, mz_star_new = mz_star_new, mz_new
                    prefactor *= matrix_factor
                    real_pref = prefactor
                    imag_pref = (1j * (-1.0 if is_anti_total else 1.0)) * prefactor
                    term_contributions.append(
                        (
                            int(r_out),
                            int(c_out),
                            int(mz_new),
                            int(mz_star_new),
                            np.asarray(q_new, dtype=float),
                            complex(real_pref),
                            complex(imag_pref),
                        )
                    )
        needs_herm_real, needs_herm_imag = _hermitize_flags_from_operator_contributions(
            term_contributions,
            dim,
        )
        for r_out, c_out, mz_new, mz_star_new, q_new, real_pref, imag_pref in term_contributions:
            _append_operator_contribution(
                idx,
                int(r_out),
                int(c_out),
                mz_new,
                mz_star_new,
                q_new,
                real_pref,
                imag_pref,
                term_index,
                row,
                col,
                mz_values,
                mz_star_values,
                q_center,
                prefactor_real,
                prefactor_imag,
            )
            herm_real_pref = np.conjugate(real_pref) if needs_herm_real else 0.0j
            herm_imag_pref = np.conjugate(imag_pref) if needs_herm_imag else 0.0j
            if herm_real_pref != 0.0j or herm_imag_pref != 0.0j:
                _append_operator_contribution(
                    idx,
                    int(c_out),
                    int(r_out),
                    mz_star_new,
                    mz_new,
                    q_new,
                    herm_real_pref,
                    herm_imag_pref,
                    term_index,
                    row,
                    col,
                    mz_values,
                    mz_star_values,
                    q_center,
                    prefactor_real,
                    prefactor_imag,
                )

    if not term_index:
        raise ValueError("runtime recipe expansion produced no operator contributions")
    return {
        "term_index": np.asarray(term_index, dtype=np.int64),
        "row": np.asarray(row, dtype=np.int64),
        "col": np.asarray(col, dtype=np.int64),
        "mz": np.asarray(mz_values, dtype=np.int64),
        "mz_star": np.asarray(mz_star_values, dtype=np.int64),
        "q_center": np.asarray(q_center, dtype=float),
        "prefactor_real": np.asarray(prefactor_real, dtype=np.complex128),
        "prefactor_imag": np.asarray(prefactor_imag, dtype=np.complex128),
    }


def _hermitize_flags_from_operator_contributions(
    contributions: Sequence[tuple[int, int, int, int, np.ndarray, complex, complex]],
    dim: int,
) -> tuple[bool, bool]:
    if not contributions:
        return False, False
    probe_k = np.array([0.137, -0.219], dtype=float)
    real_matrix = np.zeros((int(dim), int(dim)), dtype=np.complex128)
    imag_matrix = np.zeros_like(real_matrix)
    for row_idx, col_idx, mz, mz_star, q, real_pref, imag_pref in contributions:
        z = (probe_k[0] - float(q[0])) + 1j * (probe_k[1] - float(q[1]))
        monomial = (z ** int(mz)) * (np.conjugate(z) ** int(mz_star))
        real_matrix[int(row_idx), int(col_idx)] += complex(real_pref) * monomial
        imag_matrix[int(row_idx), int(col_idx)] += complex(imag_pref) * monomial
    return (not np.allclose(real_matrix, real_matrix.conj().T), not np.allclose(imag_matrix, imag_matrix.conj().T))


def _append_operator_contribution(
    term_idx: int,
    row_idx: int,
    col_idx: int,
    mz: int,
    mz_star: int,
    q: np.ndarray,
    real_prefactor: complex,
    imag_prefactor: complex,
    term_index: list[int],
    rows: list[int],
    cols: list[int],
    mz_values: list[int],
    mz_star_values: list[int],
    q_center: list[tuple[float, float]],
    prefactor_real: list[complex],
    prefactor_imag: list[complex],
) -> None:
    if abs(real_prefactor) <= 1.0e-13 and abs(imag_prefactor) <= 1.0e-13:
        return
    term_index.append(int(term_idx))
    rows.append(int(row_idx))
    cols.append(int(col_idx))
    mz_values.append(int(mz))
    mz_star_values.append(int(mz_star))
    q_center.append((float(q[0]), float(q[1])))
    prefactor_real.append(complex(real_prefactor))
    prefactor_imag.append(complex(imag_prefactor))


def _iter_transformed_sparse_entries(
    action: Mapping[str, Any] | None,
    rows: np.ndarray,
    cols: np.ndarray,
):
    if action is None:
        for sparse_pos, (r_out, c_out) in enumerate(zip(rows, cols)):
            yield sparse_pos, int(r_out), int(c_out), 1.0 + 0.0j, False
        return

    kind = str(action.get("kind", ""))
    if kind == "monomial":
        perm = np.asarray(action["perm"], dtype=int)
        vals = np.asarray(action["vals"], dtype=np.complex128)
        inv_vals = np.asarray(action["inv_vals"], dtype=np.complex128)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(perm.shape[0], dtype=int)
        rr = inv_perm[rows]
        cc = inv_perm[cols]
        matrix_factor = vals[rr] * inv_vals[cc]
        is_anti_total = bool(action.get("is_anti_total", False))
        for sparse_pos, (r_out, c_out, factor) in enumerate(zip(rr, cc, matrix_factor)):
            yield sparse_pos, int(r_out), int(c_out), complex(factor), is_anti_total
        return

    if kind == "dense":
        left = np.asarray(action["left"], dtype=np.complex128)
        right = np.asarray(action["right"], dtype=np.complex128)
        is_anti_total = bool(action.get("is_anti_total", False))
        for sparse_pos, (row_in, col_in) in enumerate(zip(rows, cols)):
            left_col = left[:, int(row_in)]
            right_row = right[int(col_in), :]
            out_rows = np.flatnonzero(np.abs(left_col) > _DENSE_ACTION_ENTRY_TOL)
            out_cols = np.flatnonzero(np.abs(right_row) > _DENSE_ACTION_ENTRY_TOL)
            for r_out in out_rows:
                left_value = left_col[int(r_out)]
                for c_out in out_cols:
                    factor = left_value * right_row[int(c_out)]
                    if abs(factor) <= _DENSE_ACTION_ENTRY_TOL:
                        continue
                    yield sparse_pos, int(r_out), int(c_out), complex(factor), is_anti_total
        return

    raise ValueError(f"unknown standalone export symmetry action kind {kind!r}")


def _orbit_records(sym_ops: Sequence[Mapping[str, Any]], symmetry_gen: Any) -> list[tuple[np.ndarray, Mapping[str, Any] | None]]:
    if not sym_ops:
        return [(np.eye(2, dtype=float), None)]
    sym_ops_list = [dict(op) for op in sym_ops if isinstance(op, Mapping)]
    cache_key = (id(symmetry_gen), ContinuumModelBuilder._sym_ops_cache_key(sym_ops_list))
    cached = _ORBIT_RECORD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _points, op_seqs = ContinuumModelBuilder._generate_symmetry_orbit(np.zeros(2, dtype=float), sym_ops_list)
    records = []
    for op_seq in op_seqs:
        transform = _linear_transform_for_op_seq(op_seq, sym_ops_list)
        if not op_seq:
            records.append((transform, None))
            continue
        op_seq_applied = tuple(reversed(op_seq))
        action = ContinuumModelBuilder._get_composed_symmetry_action(symmetry_gen, op_seq_applied)
        if action is not None and ContinuumModelBuilder._validate_composed_symmetry_action_once(
            symmetry_gen,
            op_seq_applied,
            action,
        ):
            perm, vals, inv_vals, is_anti_total = action
            records.append(
                (
                    transform,
                    {
                        "kind": "monomial",
                        "perm": perm,
                        "vals": vals,
                        "inv_vals": inv_vals,
                        "is_anti_total": bool(is_anti_total),
                    },
                )
            )
            continue
        records.append((transform, _dense_composed_symmetry_action(symmetry_gen, op_seq_applied)))
    _ORBIT_RECORD_CACHE[cache_key] = records
    return records


def _dense_composed_symmetry_action(symmetry_gen: Any, op_seq_applied: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    if symmetry_gen is None:
        raise ValueError("standalone export requires symmetry_gen when sym_ops is non-empty")
    left_total: np.ndarray | None = None
    right_total: np.ndarray | None = None
    is_anti_total = False
    dim: int | None = None
    for op_name, param in op_seq_applied:
        left_op, right_op, is_anti_op = _dense_single_symmetry_action(symmetry_gen, str(op_name), param)
        left_op = np.asarray(left_op, dtype=np.complex128)
        right_op = np.asarray(right_op, dtype=np.complex128)
        if left_op.ndim != 2 or left_op.shape[0] != left_op.shape[1]:
            raise ValueError(f"symmetry operator {op_name!r} must be square, got {left_op.shape}")
        if right_op.shape != left_op.shape:
            raise ValueError(
                f"right action for symmetry operator {op_name!r} has shape {right_op.shape}, "
                f"expected {left_op.shape}"
            )
        if dim is None:
            dim = int(left_op.shape[0])
            left_total = np.eye(dim, dtype=np.complex128)
            right_total = np.eye(dim, dtype=np.complex128)
        elif left_op.shape != (dim, dim):
            raise ValueError(f"symmetry operator {op_name!r} shape {left_op.shape} does not match {(dim, dim)}")
        assert left_total is not None and right_total is not None
        if is_anti_op:
            left_total = left_op @ left_total.conj()
            right_total = right_total.conj() @ right_op
            is_anti_total = not is_anti_total
        else:
            left_total = left_op @ left_total
            right_total = right_total @ right_op
    if left_total is None or right_total is None:
        raise ValueError("dense composed symmetry action requires at least one operation")
    return {
        "kind": "dense",
        "left": left_total,
        "right": right_total,
        "is_anti_total": bool(is_anti_total),
    }


def _dense_single_symmetry_action(symmetry_gen: Any, op_name: str, param: Any) -> tuple[np.ndarray, np.ndarray, bool]:
    if op_name == "C3z":
        left = symmetry_gen.get_operator(op_name, param)
        right = symmetry_gen.get_operator(op_name, -int(param))
        return left, right, False
    if op_name in ContinuumModelBuilder._SYMM_ANTIUNITARY_OPS:
        left = symmetry_gen.get_operator(op_name, param)
        return left, np.asarray(left).conj().T, True
    if op_name in ContinuumModelBuilder._SYMM_UNITARY_OPS:
        left = symmetry_gen.get_operator(op_name, param)
        return left, np.asarray(left).conj().T, False
    raise ValueError(f"Unknown symmetry operation: {op_name}")


def _linear_transform_for_op_seq(op_seq: Sequence[tuple[str, Any]], sym_ops: Sequence[Mapping[str, Any]]) -> np.ndarray:
    op_by_name = {str(op["name"]): op for op in sym_ops if isinstance(op, Mapping) and "name" in op}
    basis = np.eye(2, dtype=float)
    columns = []
    for vector in basis:
        out = np.asarray(vector, dtype=float)
        for op_name, param in op_seq:
            operation = op_by_name[str(op_name)]
            out = ContinuumModelBuilder._apply_k_map_to_vector(
                out,
                operation,
                power=None if param is None else int(param),
            )
        columns.append(out)
    return np.column_stack(columns)


def _transform_monomial(transform: np.ndarray, q_base: np.ndarray, mz: int, mz_star: int) -> tuple[int, int, np.ndarray, complex]:
    q_center = np.linalg.solve(transform, np.asarray(q_base, dtype=float))
    z_e1 = complex(transform[0, 0], transform[1, 0])
    z_e2 = complex(transform[0, 1], transform[1, 1])
    det = float(np.linalg.det(transform))
    if det > 0:
        alpha = z_e1
        if not np.allclose(z_e2, 1j * alpha, atol=1.0e-8, rtol=0.0):
            raise NotImplementedError("standalone export supports only conformal rotation k maps")
        prefactor = (alpha**mz) * (np.conjugate(alpha) ** mz_star)
        return int(mz), int(mz_star), q_center, complex(prefactor)
    alpha = z_e1
    if not np.allclose(z_e2, -1j * alpha, atol=1.0e-8, rtol=0.0):
        raise NotImplementedError("standalone export supports only reflection k maps")
    prefactor = (alpha**mz) * (np.conjugate(alpha) ** mz_star)
    return int(mz_star), int(mz), q_center, complex(prefactor)


def _portable_operations(model_config: Any, exactified: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    registry = _load_json_if_exists(model_config.output_dir / "operation_registry.json") if model_config.output_dir else None
    source = registry if isinstance(registry, list) else model_config.symmetry_source_metadata.get("operations", [])
    for item in source:
        if not isinstance(item, Mapping):
            continue
        raw_name = str(item.get("canonical_operation", item.get("name", item.get("operation", ""))))
        name = _operation_family_name(raw_name)
        if not name or name not in exactified:
            continue
        raw_operation = str(item.get("source_operation", item.get("operation", raw_name)))
        rows.append(
            {
                "name": name,
                "family": "T" if name == "TR" else name,
                "aliases": _operation_aliases(name, raw_operation),
                "operation": raw_operation,
                "antiunitary": bool(item.get("antiunitary", False)),
                "k_map": _json_safe(item.get("k_map", item.get("internal_resolved_action", {}).get("k_map"))),
                "q_map": _json_safe(item.get("q_map", item.get("internal_resolved_action", {}).get("q_map"))),
                "sector_map": _json_safe(item.get("sector_map", item.get("internal_resolved_action", {}).get("sector_map"))),
                "matrix_array_key": f"exactified_{name}",
                "matrix_kind": str(item.get("matrix_kind", "continuum_internal_rep_exact")),
                "target_role": str(item.get("target_role", "continuum_internal_rep")),
            }
        )
    if not rows:
        for name in exactified:
            rows.append(
                {
                    "name": name,
                    "family": "T" if name == "TR" else name,
                    "aliases": _operation_aliases(name, name),
                    "operation": name,
                    "antiunitary": name in {"C2T", "TR"},
                    "matrix_array_key": f"exactified_{name}",
                    "matrix_kind": "continuum_internal_rep_exact",
                    "target_role": "continuum_internal_rep",
                }
            )
    return rows


def _assert_no_forbidden_production_operations(operations: Sequence[Mapping[str, Any]]) -> None:
    for op in operations:
        name = str(op.get("name", ""))
        if name in FORBIDDEN_PRODUCTION_OPERATION_NAMES:
            raise ValueError(f"forbidden production operation name in standalone export: {name}")


def _validation_summary(
    comparison: Any,
    comparison_plot: Any,
    has_heff_eig: bool,
    *,
    reference_heff_eig_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    comparison_record = comparison_plot if isinstance(comparison_plot, Mapping) else comparison
    if not isinstance(comparison_record, Mapping):
        comparison_record = {}
    band_window = comparison_record.get("band_slice")
    compared_band_count = comparison_record.get("num_bands")
    kpoint_count = comparison_record.get("num_kpoints")
    rms_mev = comparison_record.get("rms_error_mev", comparison_record.get("aligned_rms_error_meV"))
    max_mev = comparison_record.get("max_abs_error_mev", comparison_record.get("aligned_max_abs_error_meV"))
    reference_arrays = {"eigvals": "model_data.npz:reference_eigvals"}
    if has_heff_eig:
        reference_arrays["heff_eig"] = "model_data.npz:reference_heff_eig"
    summary: dict[str, Any] = {
        "reference_kind": str(comparison_record.get("reference_kind", "packaged_reference_eigvals")),
        "band_window": None if band_window is None else [int(x) for x in band_window],
        "compared_band_count": None if compared_band_count is None else int(compared_band_count),
        "kpoint_count": None if kpoint_count is None else int(kpoint_count),
        "alignment": comparison_record.get("align", comparison_record.get("alignment")),
        "rms_error_mev": None if rms_mev is None else float(rms_mev),
        "max_error_mev": None if max_mev is None else float(max_mev),
        "reference_arrays": reference_arrays,
        "reference_heff_eig_shape": None if reference_heff_eig_shape is None else [int(x) for x in reference_heff_eig_shape],
        "is_full_tapw_validation": False,
    }
    if isinstance(comparison, Mapping):
        summary["comparison_raw"] = _json_safe(comparison)
    if isinstance(comparison_plot, Mapping):
        summary["plot_comparison_raw"] = _json_safe(comparison_plot)
    return summary


def _energy_reference(validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_bands": "raw_model_eigenvalues",
        "zero_eV": None,
        "validation_alignment": validation.get("alignment"),
        "alignment_applied_to_output": False,
    }


def _model_name(case_id: str, model_config: Any) -> str:
    spin = str(model_config.valley_model.get("spin_convention", ""))
    return f"{case_id}_{spin}".strip("_")


def _load_required_array(path: Path, description: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{description} is required: {path}")
    return np.load(path, allow_pickle=False)


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _array_payload_hash(key: str, value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(key).encode("utf-8"))
    h.update(b"\0")
    h.update(arr.dtype.str.encode("ascii"))
    h.update(b"\0")
    h.update(json.dumps([int(x) for x in arr.shape], separators=(",", ":")).encode("ascii"))
    h.update(b"\0")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _model_data_array_hashes(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, str], str]:
    per_key = {key: _array_payload_hash(key, np.asarray(arrays[key])) for key in sorted(arrays)}
    combined = hashlib.sha256()
    for key in sorted(per_key):
        combined.update(key.encode("utf-8"))
        combined.update(b"\0")
        combined.update(per_key[key].encode("ascii"))
        combined.update(b"\0")
    return per_key, combined.hexdigest()


def _model_json_for_hash(model_json: Mapping[str, Any]) -> dict[str, Any]:
    safe = _json_safe(model_json)
    hashes = dict(safe.get("hashes", {}))
    hashes.pop("model_json_canonical_sha256", None)
    safe["hashes"] = hashes
    return safe


def _canonical_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_text(value).encode("utf-8")).hexdigest()


def _assert_strict_json(name: str, value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains non-strict JSON value: {exc}") from exc


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **{key: np.asarray(value) for key, value in arrays.items()})


def _debug_payloads(
    *,
    model_json: Mapping[str, Any],
    active_terms: Sequence[Mapping[str, Any]],
    operations: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    model_data: Mapping[str, np.ndarray],
) -> dict[str, bytes | str]:
    payloads: dict[str, bytes | str] = {
        "manifest.json": json.dumps(model_json, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "terms.json": json.dumps(list(active_terms), indent=2, sort_keys=True, allow_nan=False) + "\n",
        "operations.json": json.dumps(list(operations), indent=2, sort_keys=True, allow_nan=False) + "\n",
        "comparison.json": json.dumps(dict(comparison), indent=2, sort_keys=True, allow_nan=False) + "\n",
        "operator_terms_schema.md": _render_operator_terms_schema(),
        "symmetry_reconstruction.md": "# Symmetry Reconstruction\n\n" + _render_symmetry_reconstruction(model_json, model_data) + "\n",
    }
    return payloads


def _render_operator_terms_schema() -> str:
    return """# Operator Terms Schema

Runtime evaluation reads the pre-expanded arrays in `model_data.npz`:

- `operator_term_index`
- `operator_row`
- `operator_col`
- `operator_mz`
- `operator_mz_star`
- `operator_q_center`
- `operator_prefactor_real`
- `operator_prefactor_imag`

For contribution `c`, `operator_term_index[c]` selects the fitted coefficient
from `term_r_value_real` and `term_r_value_imag`; row/col select the matrix
element; `mz/mz_star` and `q_center` define the monomial; prefactors multiply
the real and imaginary coefficient channels.
"""


def _render_readme(model: Mapping[str, Any]) -> str:
    name = model["model_id"]
    return f"""# {name}

This directory contains a NumPy-only evaluator for `{name}`.

## Requirements

Python 3 and NumPy.

## Run

```bash
python evaluate.py
```

This writes:

- `kpoints.npy`
- `bands.npy`
- `bands.pdf`
- `kdist.npy`
- `kpath_ticks.json`

## Edit the k path

Open `evaluate.py` and edit:

- `HIGH_SYMMETRY_POINTS`
- `KPATH`
- `POINTS_PER_SEGMENT`
- `WINDOW_BANDS`
- `WINDOW_EDGE`

## Files

- `evaluate.py`: standalone evaluator
- `model_data.npz`: arrays used by the evaluator
- `MODEL.md`: model formula and basis notes
"""


def _material_display(model: Mapping[str, Any]) -> str:
    material = model.get("material", {})
    name = str(material.get("name", "")).strip() if isinstance(material, Mapping) else ""
    return name if name else "not recorded"


def _operation_names_text(model: Mapping[str, Any]) -> str:
    names = [str(op.get("name", "")).strip() for op in model.get("operations", []) if isinstance(op, Mapping)]
    names = [name for name in names if name]
    if not names:
        return "not recorded"
    if len(names) == 1:
        return f"`{names[0]}`"
    if len(names) == 2:
        return f"`{names[0]}` and `{names[1]}`"
    return ", ".join(f"`{name}`" for name in names[:-1]) + f", and `{names[-1]}`"


def _format_float(value: float) -> str:
    if abs(value) < 5.0e-13:
        value = 0.0
    return f"{value:.12g}"


def _format_vector(values: Any) -> str:
    arr = np.asarray(values, dtype=float).ravel()
    return "[" + ", ".join(_format_float(float(x)) for x in arr) + "]"


def _format_matrix(matrix: np.ndarray) -> str:
    return "[" + "; ".join("[" + ", ".join(_format_float(float(x)) for x in row) + "]" for row in matrix) + "]"


def _linear_map_matrix(map_metadata: Any) -> np.ndarray | None:
    if not isinstance(map_metadata, Mapping):
        return None
    map_type = str(map_metadata.get("type", "")).lower()
    if map_type == "identity":
        return np.eye(2, dtype=float)
    if map_type == "negation":
        return -np.eye(2, dtype=float)
    if map_type == "rotation":
        angle = -np.deg2rad(float(map_metadata.get("angle_deg", 0.0)))
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=float)
    if map_type == "reflection":
        theta = np.deg2rad(float(map_metadata.get("axis_deg", 0.0)))
        axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        return 2.0 * np.outer(axis, axis) - np.eye(2, dtype=float)
    return None


def _render_linear_action(map_metadata: Any, *, symbol: str) -> str:
    matrix = _linear_map_matrix(map_metadata)
    if matrix is None:
        return "metadata missing"
    return f"{symbol}' = {_format_matrix(matrix)} {symbol}"


def _coordinate_kind(model: Mapping[str, Any]) -> str:
    coord = model.get("coordinate_convention", {})
    if not isinstance(coord, Mapping):
        return "missing"
    text = " ".join(str(coord.get(key, "")) for key in ("frame", "k_units", "q_units", "coordinate_type", "units")).lower()
    if "dimensionless" in text or "fractional" in text:
        return "dimensionless"
    if "cartesian" in text or "inverse-length" in text or "1/" in text or "angstrom" in text:
        return "physical"
    return "unknown"


def _render_coordinate_section(model: Mapping[str, Any]) -> str:
    coord = model.get("coordinate_convention", {})
    if not isinstance(coord, Mapping) or not coord:
        return (
            "Coordinate convention: not recorded; do not use custom k-points until this metadata is added.\n"
            "\n`evaluate.py` expects k-points with shape `(Nk, 2)` when a convention is available."
        )
    frame = str(coord.get("frame", "not recorded"))
    k_units = str(coord.get("k_units", "not recorded"))
    q_units = str(coord.get("q_units", "not recorded"))
    kind = _coordinate_kind(model)
    lines = [
        "`evaluate.py` expects k-points as a NumPy array with shape `(Nk, 2)`.",
        "Column 0 is the first recorded model-frame coordinate; column 1 is the second recorded model-frame coordinate.",
        f"Recorded frame: `{frame}`.",
        f"Recorded k units: `{k_units}`.",
        f"Recorded Q units: `{q_units}`.",
    ]
    bM1 = coord.get("bM1")
    bM2 = coord.get("bM2")
    if bM1 is not None and bM2 is not None:
        lines.extend(
            [
                f"Recorded reciprocal basis vectors: `bM1 = {_format_vector(bM1)}`, `bM2 = {_format_vector(bM2)}`.",
                "`k_phys = k[0] * bM1 + k[1] * bM2` for fractional model-frame coordinates.",
            ]
        )
    if kind == "dimensionless":
        lines.append("Coordinate type: dimensionless model-frame coordinates.")
    elif kind == "physical":
        lines.append("Coordinate type: physical inverse-length coordinates.")
    else:
        lines.append(
            "Coordinate type: not explicitly recorded as dimensionless or Cartesian; do not use custom k-points unless they follow the same convention as `reference_kpoints`."
        )
    return "\n".join(lines)


def _render_coefficient_units(model: Mapping[str, Any]) -> str:
    kind = _coordinate_kind(model)
    if kind == "dimensionless":
        return "The exported k/Q coordinates are recorded as dimensionless model-frame coordinates, so stored coefficients are in the model energy unit."
    if kind == "physical":
        return (
            "The exported k/Q coordinates are recorded as physical inverse-length coordinates. "
            "Polynomial coefficient units depend on the total monomial order `Mz + Mz_star`."
        )
    return (
        "Coefficient units follow the exported coordinate convention and monomial order. "
        "Constant onsite terms are energy-like; polynomial terms require the recorded k/Q coordinate convention to assign physical units."
    )


def _render_operation_table(model: Mapping[str, Any]) -> str:
    rows = []
    for op in model.get("operations", []):
        if not isinstance(op, Mapping):
            continue
        name = str(op.get("name", "unknown"))
        antiunitary = "yes" if op.get("antiunitary") else "no"
        matrix_key = str(op.get("matrix_array_key", "not recorded"))
        k_action = _render_linear_action(op.get("k_map"), symbol="k")
        q_action = _render_linear_action(op.get("q_map"), symbol="Q")
        sector_map = json.dumps(_json_safe(op.get("sector_map", "not recorded")), sort_keys=True)
        rows.append(f"| `{name}` | {antiunitary} | `{matrix_key}` | `{k_action}` | `{q_action}` | `{sector_map}` |")
    if not rows:
        return "| none | n/a | n/a | metadata missing | metadata missing | n/a |"
    return "\n".join(rows)


def _format_complex(value: complex) -> str:
    real = float(np.real(value))
    imag = float(np.imag(value))
    if abs(real) < 5.0e-13:
        real = 0.0
    if abs(imag) < 5.0e-13:
        imag = 0.0
    if imag < 0:
        return f"{_format_float(real)} - {_format_float(abs(imag))}i"
    return f"{_format_float(real)} + {_format_float(imag)}i"


def _format_pi_angle(angle: float) -> str:
    coeff = angle / np.pi
    fraction = Fraction(float(coeff)).limit_denominator(12)
    if abs(float(fraction) - coeff) < 1.0e-10:
        numerator = fraction.numerator
        denominator = fraction.denominator
        if numerator == 0:
            return "0"
        sign = "-" if numerator < 0 else ""
        numerator = abs(numerator)
        if denominator == 1:
            factor = "" if numerator == 1 else f"{numerator}*"
            return f"{sign}{factor}pi"
        factor = "" if numerator == 1 else f"{numerator}*"
        return f"{sign}{factor}pi/{denominator}"
    return f"{_format_float(angle)} rad"


def _format_phase(value: complex) -> str:
    magnitude = abs(value)
    angle = float(np.angle(value))
    angle_text = _format_pi_angle(angle)
    if abs(magnitude - 1.0) < 1.0e-10:
        if angle_text == "0":
            return "0"
        return angle_text
    return f"{angle_text}; non-unit |phase|={_format_float(magnitude)}"


def _format_angle_latex(angle_text: str) -> str:
    if angle_text.endswith(" rad"):
        return angle_text[:-4] + r"\,\mathrm{rad}"
    return angle_text.replace("*", "").replace("pi", r"\pi")


def _format_phase_latex(value: complex) -> str:
    magnitude = abs(value)
    angle_text = _format_pi_angle(float(np.angle(value)))
    angle_latex = _format_angle_latex(angle_text)
    if abs(magnitude - 1.0) < 1.0e-10:
        if angle_text == "0":
            return "1"
        return f"e^{{i({angle_latex})}}"
    return f"{_format_float(magnitude)} e^{{i({angle_latex})}}"


def _math_label(value: str) -> str:
    return r"\mathrm{" + value.replace("_", r"\_") + "}"


def _basis_block_for_index(index: int, blocks: Sequence[Mapping[str, Any]]) -> tuple[str, int, int]:
    for block in blocks:
        offset = int(block.get("offset", 0))
        q_count = int(block.get("q_count", 0))
        dim = int(block.get("dim", 0))
        if offset <= index < offset + dim and q_count > 0:
            local = index - offset
            return str(block.get("qset", "unknown")), int(local // q_count), int(local % q_count)
    raise ValueError(f"basis index {index} is outside recorded basis blocks")


def _render_q_permutation(pairs: Sequence[tuple[int, int]]) -> str:
    return ", ".join(f"{col_q}\\mapsto {row_q}" for row_q, col_q in sorted(pairs, key=lambda item: item[1]))


def _render_symmetry_reconstruction(
    model: Mapping[str, Any],
    model_data: Mapping[str, np.ndarray],
    *,
    include_q_permutation: bool = True,
) -> str:
    blocks = list(model.get("dimension", {}).get("basis_blocks", []))
    if not blocks:
        return "Basis block metadata is missing, so `MODEL.md` cannot spell out matrix reconstruction rules."

    if include_q_permutation:
        reconstruction_note = (
            "The rules below are sufficient to reconstruct the full matrix without reading `model_data.npz`."
        )
        permutation_note = (
            "The listed permutation is \\(q'\\mapsto q=\\pi(q')\\). `sector_map` is only "
            "descriptive metadata; the equations below, or the `exactified_*` array, define "
            "the actual representation including phase angle, orbital exchange, spin exchange, "
            "and the projection gauge."
        )
    else:
        reconstruction_note = (
            "The rules below give the nonzero block structure and phase angles. "
            "The full q-index permutation is intentionally not expanded in this user-facing "
            "document; the `exactified_*` array is the authoritative matrix."
        )
        permutation_note = (
            "`sector_map` is descriptive metadata. The equations below, or the `exactified_*` "
            "array, define the actual representation including phase angle, orbital exchange, "
            "spin exchange, and the projection gauge."
        )

    sections = [
        "The table above gives the geometric action. The exact continuum representation "
        "`rho_g` is the matrix stored under the listed `exactified_*` key. Some spinful "
        f"or projected gauges produce dense representations; the array is authoritative. {reconstruction_note}",
        "",
        "Use the basis label \\(i=(\\alpha,a,q)\\), where \\(\\alpha\\) is the qset slot, "
        "\\(a\\) is the orbital index, and \\(q\\) is the Q-index inside that qset. "
        "The flat basis index is",
        "",
        "$$",
        "i(\\alpha,a,q)=\\mathrm{offset}(\\alpha)+aN_Q(\\alpha)+q.",
        "$$",
        "",
        "Each nonzero block rule has the form",
        "",
        "$$",
        "[\\rho_g]_{(\\alpha,a,q),(\\beta,b,q')} =",
        "e^{i\\theta^g_{\\alpha a,\\beta b}}\\,",
        "\\delta_{q,\\pi^g_{\\alpha a\\leftarrow\\beta b}(q')}.",
        "$$",
        "",
        "All omitted matrix elements are zero.",
        "",
        permutation_note,
    ]

    for op in model.get("operations", []):
        if not isinstance(op, Mapping):
            continue
        name = str(op.get("name", "unknown"))
        matrix_key = str(op.get("matrix_array_key", ""))
        matrix = model_data.get(matrix_key)
        if matrix is None:
            sections.extend(["", f"### `{name}`", f"Matrix key `{matrix_key}` is missing from `model_data.npz`."])
            continue
        array = np.asarray(matrix)
        rows, cols = np.nonzero(np.abs(array) > 1.0e-10)
        row_counts = np.sum(np.abs(array) > 1.0e-10, axis=1)
        col_counts = np.sum(np.abs(array) > 1.0e-10, axis=0)
        phase_permutation = bool(np.all(row_counts == 1) and np.all(col_counts == 1))
        groups: dict[tuple[str, int, str, int, complex], list[tuple[int, int]]] = {}
        for row, col in zip(rows, cols):
            row_qset, row_orb, row_q = _basis_block_for_index(int(row), blocks)
            col_qset, col_orb, col_q = _basis_block_for_index(int(col), blocks)
            value = array[int(row), int(col)]
            rounded = complex(round(float(np.real(value)), 12), round(float(np.imag(value)), 12))
            key = (row_qset, row_orb, col_qset, col_orb, rounded)
            groups.setdefault(key, []).append((row_q, col_q))
        sections.extend(
            [
                "",
                f"### `{name}`",
                f"Matrix key: `{matrix_key}`; shape `{tuple(array.shape)}`; nonzero entries `{len(rows)}`.",
                f"Phase-permutation form: `{'yes' if phase_permutation else 'no'}`.",
                "",
            ]
        )
        for key in sorted(groups, key=lambda item: (item[0], item[1], item[2], item[3], item[4].real, item[4].imag)):
            row_qset, row_orb, col_qset, col_orb, phase = key
            pairs = sorted(groups[key])
            op_label = _math_label(name)
            row_label = _math_label(row_qset)
            col_label = _math_label(col_qset)
            phase_latex = _format_phase_latex(phase)
            phase_angle = _format_phase(phase)
            block_lines = [
                f"- Nonzero block: `({col_qset}, orbital {col_orb}) -> ({row_qset}, orbital {row_orb})`; "
                f"phase angle `{phase_angle}`; count `{len(pairs)}`.",
                "",
                "  $$",
                f"  [\\rho_{{{op_label}}}]_{{({row_label},{row_orb},q),({col_label},{col_orb},q')}} =",
                f"  {phase_latex}\\,\\delta_{{q,\\pi(q')}}.",
                "  $$",
            ]
            if include_q_permutation:
                permutation = _render_q_permutation(pairs)
                block_lines.extend(
                    [
                        "",
                        "  $$",
                        f"  \\pi:\\quad {permutation}.",
                        "  $$",
                    ]
                )
            sections.extend(block_lines)
    return "\n".join(sections)


def _render_validation_section(model: Mapping[str, Any]) -> str:
    validation = model.get("validation_summary", {})
    if not isinstance(validation, Mapping):
        validation = {}
    comparison = validation.get("plot_comparison") or validation.get("comparison") or {}
    if not isinstance(comparison, Mapping):
        comparison = {}
    reference_source = str(comparison.get("reference_source", comparison.get("source", "not recorded")))
    reference_note = (
        "archived `reference_eigvals` shipped in `model_data.npz`; original source is not separately recorded"
        if reference_source == "not recorded"
        else reference_source
    )
    band_slice = comparison.get("band_slice", "not recorded")
    num_bands = comparison.get("num_bands", "not recorded")
    num_kpoints = comparison.get("num_kpoints", "not recorded")
    align = comparison.get("align", comparison.get("reference_kind", "not recorded"))
    rms = comparison.get("rms_error_mev", comparison.get("aligned_rms_error_meV", "not recorded"))
    max_abs = comparison.get("max_abs_error_mev", comparison.get("aligned_max_abs_error_meV", "not recorded"))
    heff_shape = validation.get("reference_heff_eig_shape")
    if heff_shape is None and validation.get("reference_heff_eig_available"):
        heff_text = "`reference_heff_eig` is present; shape was not recorded."
    elif heff_shape is not None:
        heff_text = f"`reference_heff_eig` is present with shape `{heff_shape}`. It is compact validation data."
    else:
        heff_text = "`reference_heff_eig` is not included."
    return f"""Reference source: {reference_note}.
Compared band slice/window: `{band_slice}`; compared bands: `{num_bands}`; k-point count: `{num_kpoints}`.
Alignment convention: `{align}`.
RMS error: `{rms}` meV; max error: `{max_abs}` meV.
Standalone runtime validation uses the packaged `reference_*` arrays when they are present.
{heff_text}

This check validates the standalone export against the archived reference arrays shipped in this package. It is not a substitute for full TAPW-to-KP validation unless the reference source is recorded as TAPW/heff-derived."""


def _render_model_doc(model: Mapping[str, Any], model_data: Mapping[str, np.ndarray] | None = None) -> str:
    dim = int(model["dimension"]["dim"])
    q_counts = model["dimension"].get("q_count_by_qset", model["dimension"].get("q_count", {}))
    n_orb = model["dimension"].get("n_orb_by_qset", model["dimension"].get("n_orb", {}))
    block_rows = [
        f"| `{block.get('qset')}` | {block.get('offset')} | {block.get('q_count')} | {block.get('n_orb')} | {block.get('dim')} |"
        for block in model["dimension"].get("basis_blocks", [])
    ]
    block_table = "\n".join(block_rows) if block_rows else "| n/a | n/a | n/a | n/a | n/a |"
    terms = list(model.get("terms", []))
    term_counts: dict[str, int] = {}
    for term in terms:
        tag = str(term.get("tag", "unknown"))
        term_counts[tag] = term_counts.get(tag, 0) + 1
    count_text = ", ".join(f"`{tag}` {term_counts.get(tag, 0)}" for tag in ["Kinect", "Onsite", "intra", "inter"])
    coord = model.get("coordinate_convention", {})
    validation = model.get("validation", {})
    energy = model.get("energy_reference", {})
    operations_text = _operation_names_text(model)
    operation_table = _render_operation_table(model)
    symmetry_rules = _render_symmetry_reconstruction(model, model_data or {}, include_q_permutation=False)
    return f"""# Model Description

## Scope

Standalone NumPy evaluator for `{model['model_id']}`.

- Material: {_material_display(model)}
- Valley: `{model.get('valley', 'not recorded')}`
- Spin convention: `{model.get('spin_convention', 'not recorded')}`
- Dimension: `{dim}`
- Energy unit: `{model.get('energy_unit', 'eV')}`

Files: `evaluate.py` runs the model; `model_data.npz` stores q sets, fitted
coefficients, exactified symmetry matrices, operator arrays, and compact
reference arrays.

## Basis And Q Sets

Basis order is qset-slot major, then orbital index, then q index:

```text
index = offset(qset) + orbital_zero_based * Nq(qset) + q_index
```

| Q-set | Offset | Q count | Orbitals | Block dim |
| --- | ---: | ---: | ---: | ---: |
{block_table}

Q sets are `qset1` and `qset2` in `model_data.npz`; friendly aliases
`qset_layer1` and `qset_layer2` may also be present.

## Coordinate Convention

`evaluate.py` expects k-points with shape `(Nk, 2)`.
Coordinate type: `{coord.get('type', 'not recorded')}`.
The high-symmetry points in `evaluate.py` use `{coord.get('hsp_coordinates_are', 'not recorded')}` coordinates.

$$
k_{{\\rm phys}} = k_1 b_{{M1}} + k_2 b_{{M2}}.
$$

- `bM1 = {_format_vector(coord.get('bM1', []))}`
- `bM2 = {_format_vector(coord.get('bM2', []))}`
- Default k path: `{model.get('default_kpath', [])}`
- Points per segment: `{model.get('points_per_segment', 'not recorded')}`

## Hamiltonian And Runtime Recipe

The exported Hamiltonian is evaluated by category:

$$
H_X(k)=
\\sum_{{t\\in T_X}}
\\sum_{{c\\in C_t}}
\\left[
r_t^R P_c^R+r_t^I P_c^I
\\right]
\\phi_c(k)|i_c\\rangle\\langle j_c|.
$$

$$
\\phi_c(k)=z(k-q_c)^{{m_c}}\\bar z(k-q_c)^{{\\bar m_c}},
\\qquad z(u)=u_x+i u_y.
$$

Runtime mapping: `operator_term_index -> t`, `operator_row/col -> i_c/j_c`,
`operator_mz/operator_mz_star -> m_c/\\bar m_c`, `operator_q_center -> q_c`,
and `operator_prefactor_real/imag -> P_c^R/P_c^I`.
`evaluate.py` combines these arrays with `term_r_value_real` and
`term_r_value_imag` from `model_data.npz`.

Output bands are `{energy.get('output_bands', 'not recorded')}`.
Validation alignment is `{energy.get('validation_alignment')}` and is applied
to output bands: `{energy.get('alignment_applied_to_output', False)}`.

## Terms

Active term counts: {count_text}; total `{len(terms)}`.
Kinetic (`"Kinect"` tag) terms, `Onsite`, `intra`, and `inter`
terms share the same runtime recipe. Semantic fields such as `key.Mz`,
`key.p`, `operation_names`, and `metadata.term_name` explain the source term
before export. Editing them does not rebuild `operator_*`; re-export with `kp`
if the term structure changes.

## Symmetry

Production operations: {operations_text}. The standalone evaluator does not
apply these operations dynamically; their effects are already folded into
`operator_*`.

| Operation | Antiunitary | Matrix | k action | Q action | Sector map |
| --- | --- | --- | --- | --- | --- |
{operation_table}

Compact matrix-element form:

$$
[\\rho_g]_{{(\\alpha,a,q),(\\beta,b,q')}} =
e^{{i\\theta^g_{{\\alpha a,\\beta b}}}}
\\delta_{{q,\\pi^g_{{\\alpha a\\leftarrow\\beta b}}(q')}}.
$$

For this exported model, the actual nonzero \\(\\rho_g\\) rules are:

{symmetry_rules}

## Validation

- Reference kind: `{validation.get('reference_kind')}`
- Band window: `{validation.get('band_window')}`
- Compared bands: `{validation.get('compared_band_count')}`
- K points: `{validation.get('kpoint_count')}`
- Alignment: `{validation.get('alignment')}`
- RMS error: `{validation.get('rms_error_mev')}` meV
- Max error: `{validation.get('max_error_mev')}` meV
- Full TAPW validation: `{validation.get('is_full_tapw_validation')}`

Missing fields are recorded as `null` in this report; no validation numbers are
invented during export.

## Limitations

This package uses the current two-qset KP core. Custom k-points must use the
recorded coordinate convention. Exactified matrices are included for diagnostics;
runtime evaluation uses the pre-expanded `operator_*` arrays. The default
`MODEL.md` includes the compact phase-permutation rules needed to reconstruct
the exported symmetry matrices. Use `debug_files=True` only for separated
developer-facing metadata files.
"""


def _evaluate_py_template(model: Mapping[str, Any]) -> str:
    coord = model.get("coordinate_convention", {})
    hsp = json.dumps(model.get("high_symmetry_points", {}), indent=4, sort_keys=True)
    kpath = repr(list(model.get("default_kpath", [])))
    points_per_segment = int(model.get("points_per_segment", 80))
    default_band_slice = model.get("default_band_slice", None)
    dim = int(model.get("dimension", {}).get("dim", 0)) if isinstance(model.get("dimension", {}), Mapping) else 0
    window_bands: int | None = None
    window_edge = "all"
    if isinstance(default_band_slice, Sequence) and not isinstance(default_band_slice, (str, bytes)) and len(default_band_slice) == 2:
        start, stop = int(default_band_slice[0]), int(default_band_slice[1])
        window_bands = max(0, stop - start)
        if dim > 0 and stop == dim:
            window_edge = "top"
        elif start == 0:
            window_edge = "bottom"
        else:
            window_edge = "top"
    window_bands_repr = repr(window_bands)
    bM1 = repr(list(coord.get("bM1", []))) if isinstance(coord, Mapping) else "[]"
    bM2 = repr(list(coord.get("bM2", []))) if isinstance(coord, Mapping) else "[]"
    coord_type = str(coord.get("type", "not recorded")) if isinstance(coord, Mapping) else "not recorded"
    return f'''from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Run this standalone evaluator with:
#   python evaluate.py
#
# Edit HIGH_SYMMETRY_POINTS, KPATH, POINTS_PER_SEGMENT, WINDOW_BANDS,
# and WINDOW_EDGE to change the path or plotted energy window.
# Coordinates are {coord_type}.

# =========================
# User-editable settings
# =========================

HIGH_SYMMETRY_POINTS = {hsp}

KPATH = {kpath}
POINTS_PER_SEGMENT = {points_per_segment}

# All bands are drawn. These settings only control the plotted y-range.
WINDOW_BANDS = {window_bands_repr}
WINDOW_EDGE = {window_edge!r}

BM1 = {bM1}
BM2 = {bM2}

OUT_KPOINTS = "kpoints.npy"
OUT_BANDS = "bands.npy"
OUT_BAND_PLOT = "bands.pdf"
OUT_KDIST = "kdist.npy"
OUT_TICKS = "kpath_ticks.json"

# Standalone topology. Keep this off by default for quick band checks; set it
# to True and tune the grid/band set below when you want model BC/QGT/WCC.
RUN_BANDS = True
RUN_TOPOLOGY = False

TOPO_N_B1 = 31
TOPO_N_B2 = 31
TOPO_RANGE_B1 = (-0.5, 0.5)
TOPO_RANGE_B2 = (-0.5, 0.5)
TOPO_BAND_SETS = {{
    "vbm2": {{"sector": "valence", "indices": [-1, -2]}},
}}
CALC_BERRY_CURVATURE = True
CALC_QUANTUM_GEOMETRY = True
CALC_WCC = False
WCC_LOOP = "b2"
TOPOLOGY_OUTPUT_DIR = "outputs/topology"

# End user-editable settings


class StandaloneModel:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.data = np.load(self.root / "model_data.npz", allow_pickle=False)
        self._required = [
            "dimension_dim",
            "operator_term_index",
            "operator_row",
            "operator_col",
            "operator_mz",
            "operator_mz_star",
            "operator_q_center",
            "operator_prefactor_real",
            "operator_prefactor_imag",
            "term_r_value_real",
            "term_r_value_imag",
            "runtime_hermitianize_before_eigvalsh",
        ]
        missing = [key for key in self._required if key not in self.data.files]
        if missing:
            raise KeyError(f"model_data.npz is missing runtime keys: {{missing}}")
        self.dim = int(np.asarray(self.data["dimension_dim"]).item())
        self.r_real = np.asarray(self.data["term_r_value_real"], dtype=float)
        self.r_imag = np.asarray(self.data["term_r_value_imag"], dtype=float)
        self.hermitianize_before_eigvalsh = bool(
            np.asarray(self.data["runtime_hermitianize_before_eigvalsh"]).item()
        )
        term_index = np.asarray(self.data["operator_term_index"], dtype=np.int64)
        if self.r_real.shape != self.r_imag.shape or self.r_real.ndim != 1:
            raise ValueError("term_r_value_real and term_r_value_imag must be one-dimensional arrays of equal shape")
        if term_index.size and int(np.max(term_index)) >= self.r_real.shape[0]:
            raise ValueError("operator_term_index references a missing fitted coefficient")

    def hamiltonian(self, k):
        k = np.asarray(k, dtype=float)
        if k.shape != (2,):
            raise ValueError(f"k must have shape (2,), got {{k.shape}}")
        term_index = np.asarray(self.data["operator_term_index"], dtype=np.int64)
        rows = np.asarray(self.data["operator_row"], dtype=np.int64)
        cols = np.asarray(self.data["operator_col"], dtype=np.int64)
        mz = np.asarray(self.data["operator_mz"], dtype=np.int64)
        mz_star = np.asarray(self.data["operator_mz_star"], dtype=np.int64)
        q_center = np.asarray(self.data["operator_q_center"], dtype=float)
        prefactor_real = np.asarray(self.data["operator_prefactor_real"], dtype=np.complex128)
        prefactor_imag = np.asarray(self.data["operator_prefactor_imag"], dtype=np.complex128)
        if q_center.ndim != 2 or q_center.shape[1] != 2:
            raise ValueError(f"operator_q_center must have shape (N, 2), got {{q_center.shape}}")
        z = (k[0] - q_center[:, 0]) + 1j * (k[1] - q_center[:, 1])
        monomial = (z ** mz) * (np.conjugate(z) ** mz_star)
        coeff = self.r_real[term_index] * prefactor_real + self.r_imag[term_index] * prefactor_imag
        h = np.zeros((self.dim, self.dim), dtype=np.complex128)
        np.add.at(h, (rows, cols), coeff * monomial)
        return h

    def _hamiltonian_for_eigvalsh(self, k):
        h = self.hamiltonian(k)
        if self.hermitianize_before_eigvalsh:
            return 0.5 * (h + h.conj().T)
        return h

    def bands(self, kpoints, band_slice=None):
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 2:
            raise ValueError(f"kpoints must have shape (Nk, 2), got {{kpoints.shape}}")
        out = np.empty((kpoints.shape[0], self.dim), dtype=float)
        for i, k in enumerate(kpoints):
            out[i] = np.linalg.eigvalsh(self._hamiltonian_for_eigvalsh(k))
        if band_slice is not None:
            if len(band_slice) != 2:
                raise ValueError(f"band_slice must be [start, stop], got {{band_slice!r}}")
            out = out[:, int(band_slice[0]):int(band_slice[1])]
        return out

    def eigensystem(self, k):
        h = self._hamiltonian_for_eigvalsh(k)
        return np.linalg.eigh(h)


def load_model(root="."):
    return StandaloneModel(root)


def _resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root).resolve() / path


def _load_kpoints(path, root):
    arr = np.load(_resolve_path(root, path), allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk, 2), got {{arr.shape}}")
    return arr


def _generate_kpath():
    if len(KPATH) < 2:
        raise ValueError("KPATH must contain at least two labels")
    b1 = np.asarray(BM1, dtype=float)
    b2 = np.asarray(BM2, dtype=float)
    if b1.shape != (2,) or b2.shape != (2,):
        raise ValueError("BM1 and BM2 must each contain two numbers")
    points = {{label: np.asarray(value, dtype=float) for label, value in HIGH_SYMMETRY_POINTS.items()}}
    missing = [label for label in KPATH if label not in points]
    if missing:
        raise ValueError(f"KPATH contains labels missing from HIGH_SYMMETRY_POINTS: {{missing}}")
    kpoints = []
    kdist = []
    tick_positions = [0.0]
    distance = 0.0
    for iseg, (start_label, stop_label) in enumerate(zip(KPATH[:-1], KPATH[1:])):
        start = points[start_label]
        stop = points[stop_label]
        if start.shape != (2,) or stop.shape != (2,):
            raise ValueError("high-symmetry points must be two-component coordinates")
        n = int(POINTS_PER_SEGMENT)
        if n <= 0:
            raise ValueError("POINTS_PER_SEGMENT must be positive")
        for i in range(n):
            if iseg > 0 and i == 0:
                continue
            t = i / float(n)
            k = (1.0 - t) * start + t * stop
            if kpoints:
                prev = kpoints[-1]
                dk = k - prev
                distance += float(np.linalg.norm(dk[0] * b1 + dk[1] * b2))
            kpoints.append(k)
            kdist.append(distance)
        if not np.allclose(kpoints[-1], stop):
            dk = stop - kpoints[-1]
            distance += float(np.linalg.norm(dk[0] * b1 + dk[1] * b2))
            kpoints.append(stop)
            kdist.append(distance)
        tick_positions.append(distance)
    return np.asarray(kpoints, dtype=float), np.asarray(kdist, dtype=float), {{"labels": list(KPATH), "positions": tick_positions}}


def _kdist_from_kpoints(kpoints):
    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 2:
        raise ValueError(f"kpoints must have shape (Nk, 2), got {{kpoints.shape}}")
    kdist = np.zeros(kpoints.shape[0], dtype=float)
    for i in range(1, kpoints.shape[0]):
        kdist[i] = kdist[i - 1] + float(np.linalg.norm(kpoints[i] - kpoints[i - 1]))
    return kdist


def _ticks_for_reference_kpath(kdist):
    segments = len(KPATH) - 1
    if segments <= 0:
        return {{"labels": [], "positions": []}}
    expected = segments * int(POINTS_PER_SEGMENT) + 1
    if int(kdist.shape[0]) != expected:
        return {{"labels": [], "positions": []}}
    indices = [i * int(POINTS_PER_SEGMENT) for i in range(segments + 1)]
    return {{"labels": list(KPATH), "positions": [float(kdist[i]) for i in indices]}}


def _band_window(dim):
    if WINDOW_BANDS is not None:
        count = max(1, min(int(WINDOW_BANDS), int(dim)))
        edge = str(WINDOW_EDGE).strip().lower()
        if edge == "bottom":
            start, stop = 0, count
        elif edge in {{"top", "all"}}:
            start, stop = int(dim) - count, int(dim)
        else:
            raise ValueError(f"WINDOW_EDGE must be 'top', 'bottom', or 'all', got {{WINDOW_EDGE!r}}")
    else:
        start, stop = 0, int(dim)
    if start < 0 or stop > dim or start >= stop:
        raise ValueError(f"Invalid band window [{{start}}, {{stop}}] for dim={{dim}}")
    return start, stop


def _plot_bands(kdist, bands, ticks, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped band plot")
        return None

    bands = np.asarray(bands, dtype=float)
    start, stop = _band_window(bands.shape[1])
    window = bands[:, start:stop]
    edge = str(WINDOW_EDGE).strip().lower()
    if edge == "bottom":
        reference = float(np.min(window[:, 0]))
        ylabel = "Energy - E_bottom (eV)"
    elif edge in {{"top", "slice"}}:
        reference = float(np.max(window[:, -1]))
        ylabel = "Energy - E_top (eV)"
    else:
        reference = 0.0
        ylabel = "Energy (eV)"
    shifted = bands - reference
    window_shifted = window - reference
    fig, ax = plt.subplots(figsize=(3.0, 5.0))
    for iband in range(shifted.shape[1]):
        ax.plot(kdist, shifted[:, iband], color="#1f77b4", linewidth=1.05, marker="o", markersize=2.3)
    if ticks.get("positions") and ticks.get("labels"):
        ax.set_xticks([float(item) for item in ticks["positions"]])
        ax.set_xticklabels([str(item) for item in ticks["labels"]])
        for tick in ticks["positions"]:
            ax.axvline(float(tick), color="0.86", linewidth=0.75, zorder=0)
    else:
        ax.set_xlabel("k-point index")
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", alpha=0.8, zorder=0)
    ax.set_ylabel(ylabel)
    ymin = float(np.min(window_shifted))
    ymax = float(np.max(window_shifted))
    pad = max(0.004, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(axis="y", color="0.88", linewidth=0.65)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _float_token(value):
    text = f"{{float(value):.6g}}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def _topology_grid_id():
    return (
        f"grid{{int(TOPO_N_B1)}}x{{int(TOPO_N_B2)}}"
        f"_b1_{{_float_token(TOPO_RANGE_B1[0])}}_{{_float_token(TOPO_RANGE_B1[1])}}"
        f"_b2_{{_float_token(TOPO_RANGE_B2[0])}}_{{_float_token(TOPO_RANGE_B2[1])}}"
    )


def _model_reciprocal_basis(model):
    if "model_reciprocal_basis" not in model.data.files:
        raise KeyError(
            "Topology requires model_reciprocal_basis in model_data.npz; "
            "regenerate this standalone package with kp model"
        )
    reciprocal_basis = np.asarray(model.data["model_reciprocal_basis"], dtype=float)
    if reciprocal_basis.shape != (2, 2):
        raise ValueError(
            f"model_reciprocal_basis must have shape (2, 2), got {{reciprocal_basis.shape}}"
        )
    determinant = float(np.linalg.det(reciprocal_basis))
    if not np.all(np.isfinite(reciprocal_basis)) or abs(determinant) < 1.0e-12:
        raise ValueError("model_reciprocal_basis must be finite and nonsingular")
    return reciprocal_basis


def _topology_mesh(model):
    n1 = int(TOPO_N_B1)
    n2 = int(TOPO_N_B2)
    if n1 < 2 or n2 < 2:
        raise ValueError("TOPO_N_B1 and TOPO_N_B2 must be at least 2")
    b1_values = np.linspace(float(TOPO_RANGE_B1[0]), float(TOPO_RANGE_B1[1]), n1)
    b2_values = np.linspace(float(TOPO_RANGE_B2[0]), float(TOPO_RANGE_B2[1]), n2)
    fractional_points = np.asarray([[x, y] for x in b1_values for y in b2_values], dtype=float)
    points = fractional_points @ _model_reciprocal_basis(model)
    return b1_values, b2_values, points


def _resolve_topology_band_indices(dim, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"band-set spec must be a dict, got {{spec!r}}")
    sector = str(spec.get("sector", "valence")).strip().lower()
    raw_indices = list(spec.get("indices", []))
    if not raw_indices:
        raise ValueError("topology band-set indices cannot be empty")
    resolved = []
    for raw in raw_indices:
        idx = int(raw)
        if sector == "valence":
            resolved_idx = dim + idx if idx < 0 else idx
        elif sector == "conduction":
            resolved_idx = idx if idx >= 0 else dim + idx
        else:
            raise ValueError(f"Unsupported topology sector {{sector!r}}")
        if resolved_idx < 0 or resolved_idx >= dim:
            raise ValueError(f"Resolved band index {{resolved_idx}} is outside 0..{{dim - 1}}")
        resolved.append(resolved_idx)
    return sorted(set(resolved))


def _eigensystem_on_grid(model, points, n1, n2):
    eigvals = np.empty((n1, n2, model.dim), dtype=float)
    eigvecs = np.empty((n1, n2, model.dim, model.dim), dtype=np.complex128)
    for flat_index, k in enumerate(points):
        i = flat_index // n2
        j = flat_index % n2
        vals, vecs = model.eigensystem(k)
        eigvals[i, j] = vals
        eigvecs[i, j] = vecs
    return eigvals, eigvecs


def _projectors_for_bandset(eigvecs, band_indices):
    n1, n2, dim, _ = eigvecs.shape
    projectors = np.empty((n1, n2, dim, dim), dtype=np.complex128)
    for i in range(n1):
        for j in range(n2):
            u = eigvecs[i, j][:, band_indices]
            projectors[i, j] = u @ u.conj().T
    return projectors


def _finite_difference(arr, values, axis, index):
    i, j = index
    if axis == 0:
        if i == 0:
            return (arr[1, j] - arr[0, j]) / float(values[1] - values[0])
        if i == arr.shape[0] - 1:
            return (arr[i, j] - arr[i - 1, j]) / float(values[i] - values[i - 1])
        return (arr[i + 1, j] - arr[i - 1, j]) / float(values[i + 1] - values[i - 1])
    if j == 0:
        return (arr[i, 1] - arr[i, 0]) / float(values[1] - values[0])
    if j == arr.shape[1] - 1:
        return (arr[i, j] - arr[i, j - 1]) / float(values[j] - values[j - 1])
    return (arr[i, j + 1] - arr[i, j - 1]) / float(values[j + 1] - values[j - 1])


def _berry_and_qgt_from_projectors(projectors, b1_values, b2_values):
    n1, n2 = projectors.shape[:2]
    berry = np.empty((n1, n2), dtype=float)
    qgt = np.empty((n1, n2, 4), dtype=float)
    for i in range(n1):
        for j in range(n2):
            p = projectors[i, j]
            dp1 = _finite_difference(projectors, b1_values, 0, (i, j))
            dp2 = _finite_difference(projectors, b2_values, 1, (i, j))
            berry[i, j] = float(-2.0 * np.imag(np.trace(p @ dp1 @ dp2)))
            g11 = float(np.real(np.trace(dp1 @ dp1)))
            g22 = float(np.real(np.trace(dp2 @ dp2)))
            g12 = float(np.real(np.trace(dp1 @ dp2)))
            qgt[i, j] = [g11 + g22, g11, g22, g12]
    return berry, qgt


def _geometry_to_cartesian(berry_fractional, qgt_fractional, reciprocal_basis):
    reciprocal_basis = np.asarray(reciprocal_basis, dtype=float)
    if reciprocal_basis.shape != (2, 2):
        raise ValueError(
            f"reciprocal_basis must have shape (2, 2), got {{reciprocal_basis.shape}}"
        )
    determinant = float(np.linalg.det(reciprocal_basis))
    if not np.all(np.isfinite(reciprocal_basis)) or abs(determinant) < 1.0e-12:
        raise ValueError("reciprocal_basis must be finite and nonsingular")

    berry_cartesian = np.asarray(berry_fractional, dtype=float) / determinant
    qgt_fractional = np.asarray(qgt_fractional, dtype=float)
    inverse_basis = np.linalg.inv(reciprocal_basis)
    metric_fractional = np.empty(qgt_fractional.shape[:-1] + (2, 2), dtype=float)
    metric_fractional[..., 0, 0] = qgt_fractional[..., 1]
    metric_fractional[..., 1, 1] = qgt_fractional[..., 2]
    metric_fractional[..., 0, 1] = qgt_fractional[..., 3]
    metric_fractional[..., 1, 0] = qgt_fractional[..., 3]
    metric_cartesian = np.einsum(
        "ab,...bc,dc->...ad",
        inverse_basis,
        metric_fractional,
        inverse_basis,
        optimize=True,
    )
    qgt_cartesian = np.empty_like(qgt_fractional)
    qgt_cartesian[..., 0] = np.trace(metric_cartesian, axis1=-2, axis2=-1)
    qgt_cartesian[..., 1] = metric_cartesian[..., 0, 0]
    qgt_cartesian[..., 2] = metric_cartesian[..., 1, 1]
    qgt_cartesian[..., 3] = metric_cartesian[..., 0, 1]
    return berry_cartesian, qgt_cartesian


def _save_grid_table(path, b1_values, b2_values, values, columns):
    rows = []
    for i, x in enumerate(b1_values):
        for j, y in enumerate(b2_values):
            value = np.asarray(values[i, j], dtype=float).ravel()
            rows.append([float(x), float(y), *[float(v) for v in value]])
    header = "kappa1 kappa2 " + " ".join(columns)
    np.savetxt(path, np.asarray(rows, dtype=float), header=header)


def _plot_grid_scalar(path, b1_values, b2_values, values, colorbar_label):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"matplotlib is not installed; skipped {{path.name}}")
        return None

    arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    image = ax.imshow(
        arr.T,
        origin="lower",
        extent=[float(b1_values[0]), float(b1_values[-1]), float(b2_values[0]), float(b2_values[-1])],
        aspect="equal",
        interpolation="bilinear",
    )
    ax.set_xlabel(r"$k_1/|b_M|$")
    ax.set_ylabel(r"$k_2/|b_M|$")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _sewing_permutation(model, shift):
    required = ["basis_qset_id", "basis_q_index", "basis_orbital", "basis_q_vector"]
    missing = [key for key in required if key not in model.data.files]
    if missing:
        raise KeyError(
            "WCC requires basis metadata in model_data.npz; regenerate this standalone package with a newer kp model export. "
            f"Missing: {{missing}}"
        )
    qset_id = np.asarray(model.data["basis_qset_id"], dtype=np.int64)
    orbital = np.asarray(model.data["basis_orbital"], dtype=np.int64)
    qvec = np.asarray(model.data["basis_q_vector"], dtype=float)
    shift = np.asarray(shift, dtype=float)
    target_by_source = np.empty(model.dim, dtype=np.int64)
    used = set()
    for source in range(model.dim):
        wanted = qvec[source] + shift
        matches = np.where(
            (qset_id == qset_id[source])
            & (orbital == orbital[source])
            & (np.linalg.norm(qvec - wanted[None, :], axis=1) < 1.0e-7)
        )[0]
        if matches.size != 1:
            raise ValueError(
                f"WCC boundary sewing failed for row {{source}} with shift={{shift.tolist()}}; matched {{matches.size}} rows"
            )
        target = int(matches[0])
        if target in used:
            raise ValueError("WCC boundary sewing is not one-to-one")
        used.add(target)
        target_by_source[source] = target
    return target_by_source


def _apply_sewing(matrix, target_by_source):
    out = np.zeros_like(matrix)
    out[target_by_source, :] = matrix
    return out


def _wilson_wcc(model, eigvecs, band_indices, loop):
    loop = str(loop).strip().lower()
    reciprocal_basis = _model_reciprocal_basis(model)
    if loop == "b1":
        shift = reciprocal_basis[0]
        target_by_source = _sewing_permutation(model, shift)
        sweep_count = eigvecs.shape[1]
        branches = []
        for j in range(sweep_count):
            product = np.eye(len(band_indices), dtype=np.complex128)
            for i in range(eigvecs.shape[0] - 1):
                u0 = eigvecs[i, j][:, band_indices]
                u1 = eigvecs[i + 1, j][:, band_indices]
                product = (u1.conj().T @ u0) @ product
            u_start = eigvecs[0, j][:, band_indices]
            u_end = _apply_sewing(eigvecs[-1, j, :, :], target_by_source)[:, band_indices]
            product = (u_start.conj().T @ u_end) @ product
            phases = np.sort((np.angle(np.linalg.eigvals(product)) / (2.0 * np.pi)) % 1.0)
            branches.append(phases)
        return np.asarray(branches, dtype=float)
    if loop == "b2":
        shift = reciprocal_basis[1]
        target_by_source = _sewing_permutation(model, shift)
        sweep_count = eigvecs.shape[0]
        branches = []
        for i in range(sweep_count):
            product = np.eye(len(band_indices), dtype=np.complex128)
            for j in range(eigvecs.shape[1] - 1):
                u0 = eigvecs[i, j][:, band_indices]
                u1 = eigvecs[i, j + 1][:, band_indices]
                product = (u1.conj().T @ u0) @ product
            u_start = eigvecs[i, 0][:, band_indices]
            u_end = _apply_sewing(eigvecs[i, -1, :, :], target_by_source)[:, band_indices]
            product = (u_start.conj().T @ u_end) @ product
            phases = np.sort((np.angle(np.linalg.eigvals(product)) / (2.0 * np.pi)) % 1.0)
            branches.append(phases)
        return np.asarray(branches, dtype=float)
    raise ValueError(f"WCC_LOOP must be 'b1' or 'b2', got {{loop!r}}")


def _plot_wcc(path, sweep_values, branches):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"matplotlib is not installed; skipped {{path.name}}")
        return None

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    for branch in range(branches.shape[1]):
        ax.scatter(sweep_values, branches[:, branch], s=8, color="#4a4f55")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$k/|b_M|$")
    ax.set_ylabel(r"$\\theta/2\\pi$")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _run_topology(root, model):
    reciprocal_basis = _model_reciprocal_basis(model)
    b1_values, b2_values, points = _topology_mesh(model)
    grid_dir = _resolve_path(root, Path(TOPOLOGY_OUTPUT_DIR) / _topology_grid_id())
    grid_dir.mkdir(parents=True, exist_ok=True)
    eigvals, eigvecs = _eigensystem_on_grid(model, points, int(TOPO_N_B1), int(TOPO_N_B2))
    np.save(grid_dir / "eigvals.npy", eigvals.reshape((-1, model.dim)))
    for name, spec in TOPO_BAND_SETS.items():
        band_indices = _resolve_topology_band_indices(model.dim, spec)
        projectors = _projectors_for_bandset(eigvecs, band_indices)
        berry_fractional, qgt_fractional = _berry_and_qgt_from_projectors(
            projectors,
            b1_values,
            b2_values,
        )
        berry, qgt = _geometry_to_cartesian(
            berry_fractional,
            qgt_fractional,
            reciprocal_basis,
        )
        if CALC_BERRY_CURVATURE:
            txt = grid_dir / f"berry_curvature_{{name}}.txt"
            _save_grid_table(txt, b1_values, b2_values, berry[:, :, None], ["omega"])
            _plot_grid_scalar(grid_dir / f"berry_curvature_{{name}}.pdf", b1_values, b2_values, berry, r"$\\Omega$")
        if CALC_QUANTUM_GEOMETRY:
            txt = grid_dir / f"quantum_geometry_{{name}}.txt"
            _save_grid_table(txt, b1_values, b2_values, qgt, ["trace_g", "g11", "g22", "g12"])
            _plot_grid_scalar(grid_dir / f"quantum_geometry_{{name}}.pdf", b1_values, b2_values, qgt[:, :, 0], r"$\\mathrm{{Tr}}\\,g$")
        if CALC_WCC:
            branches = _wilson_wcc(model, eigvecs, band_indices, WCC_LOOP)
            sweep_values = b2_values if str(WCC_LOOP).lower() == "b1" else b1_values
            txt = grid_dir / f"wcc_{{name}}_loop_{{str(WCC_LOOP).lower()}}.txt"
            np.savetxt(
                txt,
                np.column_stack([sweep_values, branches]),
                header="sweep " + " ".join(f"branch{{i}}" for i in range(branches.shape[1])),
            )
            _plot_wcc(grid_dir / f"wcc_{{name}}_loop_{{str(WCC_LOOP).lower()}}.pdf", sweep_values, branches)
    print(f"wrote topology outputs: {{grid_dir}}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate standalone NumPy continuum model bands.")
    parser.add_argument("--model-root", default=".", help="Standalone package root")
    parser.add_argument("--kpoints", default=None, help="Optional .npy kpoints file with shape (Nk, 2)")
    parser.add_argument("--out", default=None, help="Output .npy file for computed bands")
    args = parser.parse_args(argv)

    root = Path(args.model_root).resolve()
    model = load_model(root)
    if RUN_BANDS:
        if args.kpoints is None:
            if "reference_kpoints" in model.data.files:
                kpoints = np.asarray(model.data["reference_kpoints"], dtype=float)
                kdist = _kdist_from_kpoints(kpoints)
                ticks = _ticks_for_reference_kpath(kdist)
            else:
                kpoints, kdist, ticks = _generate_kpath()
        else:
            kpoints = _load_kpoints(args.kpoints, root)
            kdist = _kdist_from_kpoints(kpoints)
            ticks = {{"labels": [], "positions": []}}
        bands = model.bands(kpoints, band_slice=None)
        np.save(_resolve_path(root, OUT_KPOINTS), kpoints)
        np.save(_resolve_path(root, args.out or OUT_BANDS), bands)
        np.save(_resolve_path(root, OUT_KDIST), kdist)
        with _resolve_path(root, OUT_TICKS).open("w", encoding="utf-8") as handle:
            json.dump(ticks, handle, indent=2, allow_nan=False)
            handle.write("\\n")
        plot_path = _plot_bands(kdist, bands, ticks, _resolve_path(root, OUT_BAND_PLOT))
        print(f"wrote {{kpoints.shape[0]}} k-points and {{bands.shape[1]}} bands")
        if plot_path is not None:
            print(f"wrote band plot: {{plot_path}}")
    if RUN_TOPOLOGY:
        _run_topology(root, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _assert_clean_text(name: str, text: str) -> None:
    forbidden = (
        "/data/work",
        "/data/home",
        "review_bundles",
        "review_outputs",
        "review_packages",
        "validation_runs",
        "prompt",
        "raw TAPW",
        "hamk_file",
        "qset1_file",
        "qset2_file",
    )
    hits = [item for item in forbidden if item in text]
    if hits:
        raise ValueError(f"{name} contains forbidden local/export text: {hits}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    return value
