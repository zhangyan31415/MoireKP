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


def _normalize_q_shell(value: Any) -> str:
    text = str(value).strip()
    if text.lower().startswith("q"):
        text = text[1:]
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError(f"project.q_shell must be an integer or qNN label, got {value!r}") from exc
    if index < 0:
        raise ValueError(f"project.q_shell must be non-negative, got {value!r}")
    return f"q{index:02d}"


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _expand_tapw_style_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the public TAPW-style schema into legacy internal section views."""

    out = dict(raw)
    system = _require_mapping(out.get("system"), label="system")
    project = _require_mapping(out.get("project"), label="project")
    legacy_keys = [
        key
        for key in ("case", "valley", "spin", "material", "plot", "symm", "kpath")
        if key in out
    ]
    if legacy_keys:
        raise ValueError(
            "A system-style KP config cannot mix legacy top-level key(s): "
            + ", ".join(legacy_keys)
        )
    public_sections = {"system", "project", "symmetry", "model", "bands", "symm_rep"}
    unknown_sections = sorted(set(out) - public_sections)
    if unknown_sections:
        raise ValueError(
            "A system-style KP config contains unsupported top-level section(s): "
            + ", ".join(unknown_sections)
        )

    required_system = ("name", "output", "tapw_output", "layers", "spin", "orbital_order", "cell")
    missing_system = [key for key in required_system if system.get(key) in (None, "")]
    if missing_system:
        raise ValueError(f"system is missing required field(s): {missing_system}")
    required_project = ("valley", "q_shell", "efermi", "target")
    missing_project = [key for key in required_project if project.get(key) in (None, "")]
    if missing_project:
        raise ValueError(f"project is missing required field(s): {missing_project}")

    layers = system["layers"]
    if not isinstance(layers, list) or not layers:
        raise ValueError("system.layers must be a non-empty list such as [1, 1]")
    cell = system["cell"]
    if not isinstance(cell, list) or len(cell) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in cell):
        raise ValueError("system.cell must be a 3x3 real-space lattice matrix")

    valley = str(project["valley"])
    q_shell = _normalize_q_shell(project["q_shell"])
    tapw_root = str(system["tapw_output"])
    band_root = _path_join(tapw_root, valley, q_shell, "band")
    target_raw = str(project["target"]).strip().lower()
    if target_raw in {"top", "vbm", "valence"}:
        target = "valence"
        band_filename = "energies_vbm.txt"
    elif target_raw in {"bottom", "cbm", "conduction"}:
        target = "conduction"
        band_filename = "energies_cbm.txt"
    else:
        raise ValueError(
            "project.target must be one of valence/top/vbm or "
            f"conduction/bottom/cbm, got {project['target']!r}"
        )

    material = {
        "name": str(system["name"]),
        "num_layer_list": [int(value) for value in layers],
        "spin": str(system["spin"]),
        "orbital_order": str(system["orbital_order"]),
        "efermi": float(project["efermi"]),
        "hamk_file": _path_join(band_root, "hamiltonian_k.npy"),
        "kpoints_file": _path_join(band_root, "kpoints.npy"),
        "qset1_file": _path_join(band_root, "g_vectors_group1.npy"),
        "qset2_file": _path_join(band_root, "g_vectors_group2.npy"),
        "band_file": _path_join(band_root, band_filename),
    }

    expanded_project = dict(project)
    expanded_project["mode"] = valley
    expanded_project.pop("valley", None)
    expanded_project["q_shell"] = q_shell
    expanded_project["target"] = target

    symmetry = out.get("symmetry", {})
    if symmetry is None:
        symmetry = {}
    symmetry = _require_mapping(symmetry, label="symmetry")
    symmetry.setdefault("tapw_symmetry_dir", _path_join(tapw_root, valley, q_shell, "symmetry"))

    bands = out.get("bands", {})
    if bands is None:
        bands = {}
    bands = _require_mapping(bands, label="bands")
    kpath = bands.get("kpath", {})
    if kpath is None:
        kpath = {}
    kpath = _require_mapping(kpath, label="bands.kpath")
    if "tmat" in kpath:
        raise ValueError("bands.kpath.tmat is not supported; put the real-space lattice in system.cell")
    internal_kpath = dict(kpath)
    internal_kpath["tmat"] = cell

    out.update(
        {
            "case": {
                "profile": valley,
                "q_shell": q_shell,
                "output_root": str(system["output"]),
            },
            "valley": valley,
            "spin": str(system["spin"]),
            "material": material,
            "project": expanded_project,
            "plot": {"mode": valley, "target": target},
            "symm": symmetry,
            "kpath": internal_kpath,
        }
    )
    return out


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
    """Normalize public TAPW-style or legacy KP YAML into internal section views."""
    out = dict(raw or {})
    if "system" in out:
        out = _expand_tapw_style_config(out)
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
        if selection_raw is None:
            selection = {}
        elif isinstance(selection_raw, str):
            if selection_raw.strip().lower() not in {"auto", "explicit"}:
                raise ValueError(
                    "project.selection string must be 'auto' or 'explicit' when "
                    "project.nlow_state_list is provided"
                )
            selection = {}
        elif isinstance(selection_raw, Mapping):
            selection = dict(selection_raw)
        else:
            raise ValueError("project.selection must be a string or mapping")
        selection["mode"] = "explicit"
        project["selection"] = selection
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
