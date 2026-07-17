from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _valley_type_from_label(label: str) -> str:
    text = str(label)
    if text == "Gamma":
        return "Gamma"
    if text.startswith("K"):
        return "K"
    if text.startswith("M"):
        return "M"
    return text


def _set_default_path(section: dict[str, Any], key: str, value: str) -> None:
    if section.get(key) in (None, ""):
        section[key] = value


def _path_join(root: str | Path, *parts: str) -> str:
    return Path(str(root)).joinpath(*parts).as_posix()


def _canonical_case_base(case_raw: Any) -> str | None:
    if not isinstance(case_raw, Mapping):
        return None
    profile = case_raw.get("profile")
    q_shell = case_raw.get("q_shell")
    output_root = case_raw.get("output_root", case_raw.get("root"))
    if profile in (None, "") or q_shell in (None, "") or output_root in (None, ""):
        return None
    return _path_join(str(output_root), str(profile), str(q_shell))


def _require_canonical_case(case_raw: Any) -> str:
    case_base = _canonical_case_base(case_raw)
    if case_base is None:
        raise ValueError("KP case config requires case.profile, case.q_shell, and case.output_root.")
    return case_base


def _case_name(raw_case: Any, fallback: str) -> str:
    if isinstance(raw_case, Mapping):
        profile = raw_case.get("profile")
        q_shell = raw_case.get("q_shell")
        if profile not in (None, "") and q_shell not in (None, ""):
            return f"{profile}_{q_shell}"
    return str(raw_case or fallback)


def _merge_nested_section(raw: dict[str, Any], model: dict[str, Any], key: str) -> None:
    nested = model.pop(key, None)
    if nested is None:
        return
    existing = raw.get(key)
    if existing is not None and existing != nested:
        raise ValueError(f"Use either top-level {key!r} or model.{key}, not both")
    raw[key] = nested


def _resolve_relative_to(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _manifest_file_path(files: Mapping[str, Any], manifest_dir: Path, *keys: str) -> str | None:
    for key in keys:
        value = files.get(key)
        if value not in (None, ""):
            return str(_resolve_relative_to(str(value), manifest_dir))
    return None


def _apply_tapw_band_manifest(material: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    manifest_value = material.get("tapw_band_manifest")
    if manifest_value in (None, ""):
        return material

    manifest_path = _resolve_relative_to(str(manifest_value), config_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not isinstance(files, Mapping):
        raise ValueError(f"TAPW band manifest must contain a files mapping: {manifest_path}")
    manifest_dir = manifest_path.parent

    out = dict(material)
    out.setdefault("hamk_file", _manifest_file_path(files, manifest_dir, "hamiltonian_k", "hamk"))
    out.setdefault("qset1_file", _manifest_file_path(files, manifest_dir, "g_vectors_group1", "qset1"))
    out.setdefault("qset2_file", _manifest_file_path(files, manifest_dir, "g_vectors_group2", "qset2"))
    out.setdefault("band_file", _manifest_file_path(files, manifest_dir, "energies_vbm", "energies_cbm", "band_file"))

    missing = [key for key in ("hamk_file", "qset1_file", "qset2_file") if out.get(key) in (None, "")]
    if missing:
        raise ValueError(f"TAPW band manifest missing required file role(s) {missing}: {manifest_path}")
    return out


def normalize_case_config(raw: Mapping[str, Any] | None, *, config_path: str | Path) -> dict[str, Any]:
    """Normalize the compact one-file KP case YAML into internal section views."""
    out = dict(raw or {})
    path = Path(config_path)
    case_base = _require_canonical_case(out.get("case"))
    case_name = _case_name(out.get("case"), path.stem)

    material_raw = out.get("material")
    material = dict(material_raw) if isinstance(material_raw, Mapping) else {}
    material = _apply_tapw_band_manifest(material, config_dir=path.parent)
    top_spin = out.get("spin")
    if top_spin is not None:
        if material.get("spin") is not None and str(material["spin"]) != str(top_spin):
            raise ValueError("Top-level spin conflicts with material.spin")
        material["spin"] = str(top_spin)
    if material.get("num_layers") is None and isinstance(material.get("num_layer_list"), list):
        material["num_layers"] = int(sum(int(value) for value in material["num_layer_list"]))
    if material_raw is not None:
        out["material"] = material

    project = dict(out.get("project", {}) or {})
    valley = out.get("valley")
    if valley is not None:
        project.setdefault("mode", str(valley))
    if "gauge" not in project and "norb_fix_list" not in project:
        project["gauge"] = "auto"
    if "downfold_method" not in project and "method" not in project:
        project["downfold_method"] = "linearized_lowdin"
    explicit_low_states = project.get("nlow_state_list") not in (None, [])
    selection_raw = project.get("selection")
    if explicit_low_states:
        project["selection"] = {"mode": "explicit"}
    elif selection_raw is None:
        project["selection"] = {"mode": "auto"}
    elif isinstance(selection_raw, str):
        if selection_raw.strip().lower() != "auto":
            raise ValueError(
                "project.selection string must be 'auto'; use project.nlow_state_list "
                "for an explicit selection"
            )
        project["selection"] = {"mode": "auto"}
    elif isinstance(selection_raw, Mapping):
        selection = dict(selection_raw)
        selection.setdefault("mode", "auto")
        if str(selection["mode"]).strip().lower() != "auto":
            raise ValueError(
                "project.selection.mode must be 'auto' when project.nlow_state_list is omitted"
            )
        selection["mode"] = "auto"
        project["selection"] = selection
    else:
        raise ValueError("project.selection must be 'auto' or a mapping")
    if "project" in out or project:
        out["project"] = project

    plot = dict(out.get("plot", {}) or {})
    if "mode" not in plot and project.get("mode") is not None:
        plot["mode"] = project["mode"]
    if "nlow_state_list" not in plot and project.get("nlow_state_list") is not None:
        plot["nlow_state_list"] = project["nlow_state_list"]
    if "plot" in out or plot:
        out["plot"] = plot

    symm = dict(out.get("symm", {}) or {})
    if valley is not None:
        symm.setdefault("valley", str(valley))
    if top_spin is not None:
        symm.setdefault("spin", str(top_spin))
    if "symm" in out or symm:
        out["symm"] = symm

    model = dict(out.get("model", {}) or {})
    _merge_nested_section(out, model, "fit")
    _merge_nested_section(out, model, "bands")
    if "model" in out or model:
        out["model"] = model

    is_canonical_case = case_base is not None
    if is_canonical_case and "project" in out:
        project = dict(out.get("project", {}) or {})
        _set_default_path(
            project,
            "out_dir",
            _path_join(case_base, "projection"),
        )
        out["project"] = project

    if is_canonical_case and "symm" in out:
        symm = dict(out.get("symm", {}) or {})
        _set_default_path(
            symm,
            "output_dir",
            _path_join(case_base, "symmetry"),
        )
        out["symm"] = symm

    if is_canonical_case and ("model" in out or "output" in out):
        output = dict(out.get("output", {}) or {})
        _set_default_path(
            output,
            "dir",
            _path_join(case_base, "model"),
        )
        out["output"] = output

        plot = dict(out.get("plot", {}) or {})
        _set_default_path(
            plot,
            "out",
            _path_join(case_base, "inspect", "bands_and_qblocks.pdf"),
        )
        _set_default_path(plot, "data_out", _path_join(case_base, "inspect", "spectrum.txt"))
        out["plot"] = plot

        if valley is not None and "valley_model" not in out:
            spin_text = str(top_spin or material.get("spin", "all"))
            valley_type = _valley_type_from_label(str(valley))
            if spin_text in {"all", "spinful"}:
                spin_convention = "spinful"
            elif spin_text in {"up", "spin_up_projected"}:
                spin_convention = "spin_up_projected"
            elif spin_text in {"down", "spin_down_projected"}:
                spin_convention = "spin_down_projected"
            elif spin_text in {"spinless", "spinless_effective"}:
                spin_convention = "spinless_effective"
            else:
                raise ValueError(f"Unsupported top-level spin value: {spin_text!r}")
            out["valley_model"] = {
                "lattice": "hexagonal",
                "system": "bilayer",
                "valley_type": valley_type,
                "mode": "single_valley",
                "active_valleys": [str(valley)],
                "spin_convention": spin_convention,
            }

    return out
