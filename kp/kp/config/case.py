from __future__ import annotations

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


def _merge_nested_section(raw: dict[str, Any], model: dict[str, Any], key: str) -> None:
    nested = model.pop(key, None)
    if nested is None:
        return
    existing = raw.get(key)
    if existing is not None and existing != nested:
        raise ValueError(f"Use either top-level {key!r} or model.{key}, not both")
    raw[key] = nested


def normalize_case_config(raw: Mapping[str, Any] | None, *, config_path: str | Path) -> dict[str, Any]:
    """Normalize the compact one-file KP case YAML into legacy section views.

    The returned mapping keeps legacy split-config files unchanged while adding
    defaults for release-facing case YAMLs that contain material/project/model
    sections in one file.
    """
    out = dict(raw or {})
    path = Path(config_path)
    case_name = str(out.get("case") or path.stem)

    material_raw = out.get("material")
    material = dict(material_raw) if isinstance(material_raw, Mapping) else {}
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

    is_unified = all(key in out for key in ("material", "project", "model"))
    if is_unified:
        project = dict(out.get("project", {}) or {})
        _set_default_path(project, "out_dir", f"../outputs/project/{case_name}")
        out["project"] = project

        symm = dict(out.get("symm", {}) or {})
        _set_default_path(symm, "output_dir", f"../outputs/symm/{case_name}")
        out["symm"] = symm

        output = dict(out.get("output", {}) or {})
        _set_default_path(output, "dir", f"../outputs/model/{case_name}")
        out["output"] = output

        plot = dict(out.get("plot", {}) or {})
        _set_default_path(plot, "out", f"../outputs/project/{case_name}/heff_scatter.png")
        _set_default_path(plot, "data_out", f"../outputs/project/{case_name}/heff_spectrum.txt")
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
