from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kp.model.configured import build_moire_config_from_file, load_model_config, run_configured_model
from kp.src.moire_refactored import build_model


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"

GMK_BASE_MODEL_CONFIGS = {
    "mote2_3.89_K1": EXAMPLES_ROOT / "mote2/3.89/kp/configs/model/mote2_3.89_K1.yaml",
    "mgi2_3.89_Gamma": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/mgi2_3.89_Gamma.yaml",
    "mgi2_3.89_M1": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/mgi2_3.89_M1.yaml",
}

GMK_EXTENDED_MODEL_CONFIGS = {
    "mote2_3.89_K1": EXAMPLES_ROOT / "mote2/3.89/kp/configs/model/mote2_3.89_K1.yaml",
    "mote2_3.89_K1_spinful": EXAMPLES_ROOT / "mote2/3.89/kp/configs/model/mote2_3.89_K1_spinful.yaml",
    "mgi2_3.89_Gamma": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/mgi2_3.89_Gamma.yaml",
    "mgi2_3.89_M1": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/mgi2_3.89_M1.yaml",
    "mgi2_3.89_M1_spinful": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/mgi2_3.89_M1_spinful.yaml",
}

GMK_ACTIVE_SOURCE_CONFIGS = {
    "mote2_3.89_K1": EXAMPLES_ROOT / "mote2/3.89/kp/configs/source/mote2_3.89_K1.yaml",
    "mote2_3.89_K1_spinful": EXAMPLES_ROOT / "mote2/3.89/kp/configs/source/mote2_3.89_K1_spinful.yaml",
    "mgi2_3.89_Gamma": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/source/mgi2_3.89_Gamma.yaml",
    "mgi2_3.89_M1": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/source/mgi2_3.89_M1.yaml",
    "mgi2_3.89_M1_spinful": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/source/mgi2_3.89_M1_spinful.yaml",
}

GMK_REFERENCE_OUTPUTS = {
    "mote2_3.89_K1_toy": EXAMPLES_ROOT / "mote2/3.89/kp/outputs/model/reference/mote2_3.89_K1_toy",
    "mgi2_3.89_Gamma_toy_legacy": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/reference/mgi2_3.89_Gamma_toy_legacy",
    "mgi2_3.89_M1_q7": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/reference/mgi2_3.89_M1_q7",
}

GMK_SAVED_MODEL_OUTPUTS = {
    "mote2_3.89_K1": EXAMPLES_ROOT / "mote2/3.89/kp/outputs/model/mote2_3.89_K1",
    "mote2_3.89_K1_spinful": EXAMPLES_ROOT / "mote2/3.89/kp/outputs/model/mote2_3.89_K1_spinful",
    "mgi2_3.89_Gamma": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/mgi2_3.89_Gamma",
    "mgi2_3.89_M1": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/mgi2_3.89_M1",
    "mgi2_3.89_M1_spinful": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/mgi2_3.89_M1_spinful",
}

GMK_SAVED_OUTPUT_EXPECTATIONS = {
    "mgi2_3.89_Gamma": {
        "align": "top",
        "num_bands": 4,
        "band_slice": [120, 124],
        "max_aligned_rms_error_meV": 0.75,
        "max_aligned_max_abs_error_meV": 2.25,
    },
    "mgi2_3.89_M1": {
        "align": "bottom",
        "num_bands": 8,
        "band_slice": [0, 8],
        "max_aligned_rms_error_meV": 1.25,
        "max_aligned_max_abs_error_meV": 4.1,
    },
    "mgi2_3.89_M1_spinful": {
        "align": "bottom",
        "num_bands": 8,
        "band_slice": [0, 8],
        "max_aligned_rms_error_meV": 0.75,
        "max_aligned_max_abs_error_meV": 2.9,
    },
    "mote2_3.89_K1_spinful": {
        "align": "top",
        "num_bands": 8,
        "band_slice": [100, 108],
        "max_aligned_rms_error_meV": 1.35,
        "max_aligned_max_abs_error_meV": 5.0,
    },
}

GMK_READMES = [
    EXAMPLES_ROOT / "README.md",
    EXAMPLES_ROOT / "mgi2/3.89/README.md",
    EXAMPLES_ROOT / "mote2/3.89/README.md",
]


def _iter_path_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for child in value.values():
            paths.extend(_iter_path_strings(child))
        return paths
    if isinstance(value, list):
        paths = []
        for child in value:
            paths.extend(_iter_path_strings(child))
        return paths
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return [value]
    return []


def _has_forbidden_path_semantics(path_value: Path | str, forbidden_parts: set[str]) -> bool:
    return any(
        forbidden in part.lower()
        for part in Path(str(path_value)).parts
        for forbidden in forbidden_parts
    )


def _require_example_artifact(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"example artifact is not available in this checkout: {path}")
    return path


def test_gmk_base_configs_follow_unified_layout() -> None:
    for case_id, path in GMK_BASE_MODEL_CONFIGS.items():
        assert path.exists(), path
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["source_config"].startswith("../source/")
        assert raw["output"]["dir"] == f"../../outputs/model/{case_id}"
        _require_example_artifact((path.parent / raw["source_config"]).resolve())


def test_gmk_model_config_dirs_are_file_organization_only() -> None:
    forbidden_parts = {"production", "release", "public"}
    for path in [*GMK_BASE_MODEL_CONFIGS.values(), *GMK_EXTENDED_MODEL_CONFIGS.values()]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        output_dir = Path(str(raw["output"]["dir"]))
        assert not _has_forbidden_path_semantics(output_dir, forbidden_parts), (path, output_dir)
        _require_example_artifact((path.parent / raw["source_config"]).resolve())
        cfg = load_model_config(path)
        assert not _has_forbidden_path_semantics(cfg.output_dir, forbidden_parts), (path, cfg.output_dir)


def test_gmk_active_configs_do_not_reference_release_or_public_paths() -> None:
    forbidden_parts = {"production", "release", "public"}
    active_configs = [*GMK_ACTIVE_SOURCE_CONFIGS.values(), *GMK_EXTENDED_MODEL_CONFIGS.values()]

    for path in active_configs:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for path_string in _iter_path_strings(raw):
            assert not _has_forbidden_path_semantics(path_string, forbidden_parts), (path, path_string)


def test_gmk_readme_commands_are_portable() -> None:
    forbidden_fragments = (
        "/data/work",
        "/data/home",
        "mambaforge/envs",
        "miniconda",
        "anaconda",
        "/envs/",
    )

    for path in GMK_READMES:
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in text, (path, fragment)


def test_gmk_readmes_do_not_promote_historical_runs_as_active_paths() -> None:
    historical_path_fragments = (
        "kp/runs/",
        "kp/runs",
        "outputs/model/production",
    )

    for path in GMK_READMES:
        text = path.read_text(encoding="utf-8")
        active_section = text.split("## Historical Artifacts", maxsplit=1)[0]
        for fragment in historical_path_fragments:
            assert fragment not in active_section, (path, fragment)


def test_gmk_model_configs_use_canonical_source_configs() -> None:
    for path in GMK_EXTENDED_MODEL_CONFIGS.values():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        source_config = str(raw["source_config"])
        assert source_config.startswith("../source/"), (path, source_config)
        assert "_symm_config" not in source_config, (path, source_config)
        _require_example_artifact((path.parent / source_config).resolve())


def test_gmk_model_configs_keep_user_harmonics_minimal() -> None:
    for path in [*GMK_BASE_MODEL_CONFIGS.values(), *GMK_EXTENDED_MODEL_CONFIGS.values()]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "valley" in raw, path
        assert "spin" in raw, path
        model = raw.get("model", {})
        assert "vectors" not in model, path
        assert "term_templates" not in model, path
        for kind, spec in model.get("harmonics", {}).items():
            assert isinstance(spec, int), (path, kind, spec)
        text = path.read_text(encoding="utf-8")
        assert "harmonics_source" not in text
        assert "representatives:" not in text
        for forbidden in (
            "validation:",
            "valley_model:",
            "coordinate_frame:",
            "angle_deg:",
            "external_sewing_symmetries",
            "type: kp_symm_output",
            "symmetry_map:",
            "source_matrix_role",
            "source_gauge",
            "target_role",
            "gauge_correction",
            "antiunitary_convention",
            "spin_map",
            "valley_map",
            "matrix_file",
            "k_map:",
            "q_map:",
            "sector_map:",
            "exactification:",
        ):
            assert forbidden not in text, (path, forbidden)


def test_gmk_base_configs_load() -> None:
    for case_id, path in GMK_BASE_MODEL_CONFIGS.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw.get("spin") != "spinless", path
        _require_example_artifact((path.parent / raw["source_config"]).resolve())
        cfg = load_model_config(path)
        assert cfg.output_dir.name == case_id
        assert cfg.path == path
        assert cfg.valley_model["valley_type"] in {"Gamma", "K", "M"}


def test_gmk_base_configs_use_uniform_term_symmetry_sets() -> None:
    for path in GMK_BASE_MODEL_CONFIGS.values():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        _require_example_artifact((path.parent / raw["source_config"]).resolve())
        cfg = load_model_config(path)
        by_tag = {
            tag: [op["name"] for op in cfg.symmetry_map[tag]]
            for tag in ("Kinect", "Onsite", "intra", "inter")
        }
        assert len({tuple(names) for names in by_tag.values()}) == 1, by_tag


def test_gmk_reference_outputs_follow_unified_layout() -> None:
    for path in GMK_REFERENCE_OUTPUTS.values():
        _require_example_artifact(path)
        assert path.parent.name == "reference"


def test_gmk_saved_model_outputs_are_available_for_regressions() -> None:
    for path in GMK_SAVED_MODEL_OUTPUTS.values():
        _require_example_artifact(path)


def test_gmk_saved_model_outputs_match_recorded_quality_thresholds() -> None:
    for case_id, expectation in GMK_SAVED_OUTPUT_EXPECTATIONS.items():
        output_dir = _require_example_artifact(GMK_SAVED_MODEL_OUTPUTS[case_id])
        summary_path = _require_example_artifact(output_dir / "run_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        plot_comparison = summary["plot_comparison"]

        assert plot_comparison["align"] == expectation["align"], case_id
        assert plot_comparison["num_bands"] == expectation["num_bands"], case_id
        assert plot_comparison["band_slice"] == expectation["band_slice"], case_id
        assert plot_comparison["aligned_rms_error_meV"] <= expectation["max_aligned_rms_error_meV"], case_id
        assert (
            plot_comparison["aligned_max_abs_error_meV"]
            <= expectation["max_aligned_max_abs_error_meV"]
        ), case_id


def test_gamma_c2_rawh_action_matches_declared_model_support() -> None:
    symm_dir = _require_example_artifact(
        EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/symm/mgi2_3.89_Gamma_formal"
    )
    manifest = json.loads((symm_dir / "manifest.json").read_text(encoding="utf-8"))
    c2 = next(row for row in manifest["operations"] if row.get("operation") == "C2")

    assert c2["matrix_kind"] == "continuum_internal_rep_exact"
    assert c2["matrix_source"] == "kp_symm_exactified_action"
    assert c2["raw_matrix_file"] == "C2_low_raw.npy"
    assert c2["source_matrix_projection_report"]["report"]["status"] == "exactified"
    audit = c2["gamma_C2_action_audit"]
    assert audit["matrix_source"] == "raw_h_action_projection"
    assert audit["D_low_action_support_residual"] < 1.0e-6
    assert audit["D_low_rep_support_residual"] is None
    assert audit["representation_projection_equivalent"] is None
    assert audit["representation_projection_diagnostic"]["status"] == "not_selected"
    assert audit["representation_projection_diagnostic"]["support_matrix_source"] == "raw_action"


def test_m_base_config_builds_named_sector_model() -> None:
    path = GMK_BASE_MODEL_CONFIGS["mgi2_3.89_M1"]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require_example_artifact((path.parent / raw["source_config"]).resolve())
    moire_cfg, _model_cfg = build_moire_config_from_file(path)
    model = build_model(moire_cfg)
    assert moire_cfg.sectors
    assert {sector["name"] for sector in moire_cfg.sectors} == {"bottom", "top"}
    assert any(term.key.layer_from == 2 and term.key.layer_to == 1 for term in model.terms.values())


def test_k1_base_example_stays_mev_scale_and_has_active_terms(tmp_path: Path) -> None:
    src = GMK_BASE_MODEL_CONFIGS["mote2_3.89_K1"]
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    _require_example_artifact((src.parent / raw["source_config"]).resolve())
    _require_example_artifact((src.parent / raw["symmetry_source"]).resolve())
    raw["source_config"] = str((src.parent / raw["source_config"]).resolve())
    raw["kpath"]["file"] = str((src.parent / raw["kpath"]["file"]).resolve())
    raw["symmetry_source"] = str((src.parent / raw["symmetry_source"]).resolve())
    raw["output"]["dir"] = str((tmp_path / "k1_regression").resolve())
    cfg_path = tmp_path / "mote2_3.89_K1_regression.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    results = run_configured_model(cfg_path)

    plot_comparison = results["plot_comparison"]
    assert plot_comparison["aligned_rms_error_meV"] < 10.0
    assert plot_comparison["aligned_max_abs_error_meV"] < 20.0

    active_terms = json.loads(((tmp_path / "k1_regression") / "active_terms.json").read_text(encoding="utf-8"))
    assert len(active_terms) >= 80
    assert sum(1 for term in active_terms if term["tag"] == "inter") >= 30
