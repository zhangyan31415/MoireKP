from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples"
REPO_ROOT = EXAMPLES_ROOT.parent
for src_dir in (REPO_ROOT / "kp", REPO_ROOT / "tapw"):
    sys.path.insert(0, str(src_dir))

from kp.config.case import normalize_case_config
from kp.model.pipeline import build_moire_config_from_file, load_model_config, run_configured_model
from kp.model.core import build_model


pytestmark = pytest.mark.external_data


MOTE2_ROOT = EXAMPLES_ROOT / "mote2_3.89"
MGI2_ROOT = EXAMPLES_ROOT / "mgi2_3.89"

GMK_BASE_CASE_CONFIGS = {
    "mote2_3.89_K1_up_q06": MOTE2_ROOT / "kp/configs/mote2_3.89_K1_up_q06.yaml",
    "mgi2_3.89_Gamma_q05": MGI2_ROOT / "kp/configs/mgi2_3.89_Gamma_q05.yaml",
    "mgi2_3.89_M1_spinless_q07": MGI2_ROOT / "kp/configs/mgi2_3.89_M1_spinless_q07.yaml",
}

GMK_CASE_CONFIGS = {
    "mote2_3.89_K1_q06": MOTE2_ROOT / "kp/configs/mote2_3.89_K1_q06.yaml",
    "mote2_3.89_K1_up_q06": MOTE2_ROOT / "kp/configs/mote2_3.89_K1_up_q06.yaml",
    "mgi2_3.89_Gamma_q05": MGI2_ROOT / "kp/configs/mgi2_3.89_Gamma_q05.yaml",
    "mgi2_3.89_M1_q07": MGI2_ROOT / "kp/configs/mgi2_3.89_M1_q07.yaml",
    "mgi2_3.89_M1_spinless_q07": MGI2_ROOT / "kp/configs/mgi2_3.89_M1_spinless_q07.yaml",
}

GMK_REFERENCE_OUTPUTS = {
    "mote2_3.89_K1_toy": MOTE2_ROOT / "kp/outputs/model/reference/mote2_3.89_K1_toy",
    "mgi2_3.89_Gamma_toy_legacy": MGI2_ROOT / "kp/outputs/model/reference/mgi2_3.89_Gamma_toy_legacy",
    "mgi2_3.89_M1_q7": MGI2_ROOT / "kp/outputs/model/reference/mgi2_3.89_M1_q7",
}

GMK_SAVED_MODEL_OUTPUTS = {
    "mote2_3.89_K1": MOTE2_ROOT / "kp/outputs/model/mote2_3.89_K1",
    "mote2_3.89_K1_spinful": MOTE2_ROOT / "kp/outputs/model/mote2_3.89_K1_spinful",
    "mgi2_3.89_Gamma": MGI2_ROOT / "kp/outputs/model/mgi2_3.89_Gamma",
    "mgi2_3.89_M1": MGI2_ROOT / "kp/outputs/model/mgi2_3.89_M1",
    "mgi2_3.89_M1_spinful": MGI2_ROOT / "kp/outputs/model/mgi2_3.89_M1_spinful",
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
    MGI2_ROOT / "README.md",
    MOTE2_ROOT / "README.md",
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
    path = Path(str(path_value))
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError:
            pass
    return any(
        forbidden in part.lower()
        for part in path.parts
        for forbidden in forbidden_parts
    )


def _require_example_artifact(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"example artifact is not available in this checkout: {path}")
    return path


def _symmetry_source_path(config_path: Path, raw: dict) -> Path | None:
    source = raw.get("symmetry_source")
    if isinstance(source, str):
        return (config_path.parent / source).resolve()
    if isinstance(source, dict) and source.get("path"):
        path = Path(str(source["path"]))
        return path if path.is_absolute() else (config_path.parent / path).resolve()
    return None


def _require_model_load_artifacts(config_path: Path, raw: dict) -> None:
    normalized = normalize_case_config(raw, config_path=config_path)
    source_path_raw = normalized.get("source_config")
    if source_path_raw not in (None, ""):
        source_path = Path(str(source_path_raw))
        if not source_path.is_absolute():
            source_path = (config_path.parent / source_path).resolve()
        _require_example_artifact(source_path)
    else:
        for key in ("hamk_file", "qset1_file", "qset2_file"):
            value = normalized.get("material", {}).get(key)
            if value in (None, ""):
                pytest.skip(f"single-case config is missing material.{key}: {config_path}")
            path = Path(str(value))
            if not path.is_absolute():
                path = (config_path.parent / path).resolve()
            _require_example_artifact(path)
    symmetry_path = _symmetry_source_path(config_path, raw)
    if symmetry_path is None:
        symm_dir = normalized.get("symm", {}).get("output_dir")
        if symm_dir not in (None, ""):
            symmetry_path = Path(str(symm_dir))
            if not symmetry_path.is_absolute():
                symmetry_path = (config_path.parent / symmetry_path).resolve()
        else:
            pytest.skip(f"model config has no external kp symm source: {config_path}")
    _require_example_artifact(symmetry_path)


def test_gmk_case_configs_are_single_file_release_layout() -> None:
    for config_root in (MOTE2_ROOT / "kp/configs", MGI2_ROOT / "kp/configs"):
        assert config_root.exists(), config_root
        assert not any(child.is_dir() for child in config_root.iterdir()), config_root

    for case_id, path in GMK_CASE_CONFIGS.items():
        assert path.exists(), path
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "source_config" not in raw, path
        assert isinstance(raw.get("case"), dict), path
        assert raw["case"]["profile"] and raw["case"]["q_shell"], path
        normalized = normalize_case_config(raw, config_path=path)
        assert normalized["project"]["out_dir"].endswith("/projection"), (case_id, normalized["project"]["out_dir"])
        assert normalized["symm"]["output_dir"].endswith("/symmetry"), (case_id, normalized["symm"]["output_dir"])
        assert normalized["output"]["dir"].endswith("/model"), (case_id, normalized["output"]["dir"])


def test_gmk_model_config_dirs_are_file_organization_only() -> None:
    forbidden_parts = {"production", "release", "public"}
    for path in GMK_CASE_CONFIGS.values():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        normalized = normalize_case_config(raw, config_path=path)
        output_dir = Path(str(normalized["output"]["dir"]))
        assert not _has_forbidden_path_semantics(output_dir, forbidden_parts), (path, output_dir)
        _require_model_load_artifacts(path, raw)
        cfg = load_model_config(path)
        assert not _has_forbidden_path_semantics(cfg.output_dir, forbidden_parts), (path, cfg.output_dir)


def test_gmk_active_configs_do_not_reference_release_or_public_paths() -> None:
    forbidden_parts = {"production", "release", "public"}

    for path in GMK_CASE_CONFIGS.values():
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


def test_gmk_case_configs_do_not_use_split_source_model_configs() -> None:
    for path in GMK_CASE_CONFIGS.values():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "source_config" not in raw, path
        assert "material" in raw and "project" in raw and "symm" in raw and "model" in raw, path


def test_gmk_model_configs_keep_user_harmonics_minimal() -> None:
    for path in GMK_CASE_CONFIGS.values():
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
    for case_id, path in GMK_BASE_CASE_CONFIGS.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw.get("spin") != "spinless", path
        _require_model_load_artifacts(path, raw)
        cfg = load_model_config(path)
        assert cfg.output_dir.name == "model"
        assert cfg.path == path
        assert cfg.valley_model["valley_type"] in {"Gamma", "K", "M"}


def test_gmk_base_configs_use_uniform_term_symmetry_sets() -> None:
    for path in GMK_BASE_CASE_CONFIGS.values():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        _require_model_load_artifacts(path, raw)
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


def test_c2_action_for_gamma_from_rawh_matches_declared_model_support() -> None:
    symm_dir = _require_example_artifact(
        MGI2_ROOT / "kp/outputs/symm/mgi2_3.89_Gamma_formal"
    )
    manifest = json.loads((symm_dir / "manifest.json").read_text(encoding="utf-8"))
    c2 = next(row for row in manifest["operations"] if row.get("operation") == "C2")

    assert c2["matrix_kind"] == "continuum_internal_rep_exact"
    assert c2["matrix_source"] == "kp_symm_exactified_action"
    assert c2["raw_matrix_file"] == "C2_low_raw.npy"
    assert c2["source_matrix_projection_report"]["report"]["status"] == "exactified"
    audit = c2.get("C2_action_audit") or c2["gamma_" + "C2_action_audit"]
    assert audit["matrix_source"] == "raw_h_action_projection"
    assert audit["D_low_action_support_residual"] < 1.0e-6
    assert audit["D_low_rep_support_residual"] is None
    assert audit["representation_projection_equivalent"] is None
    assert audit["representation_projection_diagnostic"]["status"] == "not_selected"
    assert audit["representation_projection_diagnostic"]["support_matrix_source"] == "raw_action"


def test_m_base_config_builds_named_sector_model() -> None:
    path = GMK_BASE_CASE_CONFIGS["mgi2_3.89_M1_spinless_q07"]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require_model_load_artifacts(path, raw)
    moire_cfg, _model_cfg = build_moire_config_from_file(path)
    model = build_model(moire_cfg)
    assert moire_cfg.sectors
    assert {sector["name"] for sector in moire_cfg.sectors} == {"bottom", "top"}
    assert any(term.key.layer_from == 2 and term.key.layer_to == 1 for term in model.terms.values())


def test_k1_base_example_stays_mev_scale_and_has_active_terms(tmp_path: Path) -> None:
    src = GMK_BASE_CASE_CONFIGS["mote2_3.89_K1_up_q06"]
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    _require_model_load_artifacts(src, raw)
    normalized = normalize_case_config(raw, config_path=src)
    raw["kpath"]["file"] = str((src.parent / raw["kpath"]["file"]).resolve())
    raw["project"]["out_dir"] = str((src.parent / normalized["project"]["out_dir"]).resolve())
    raw["symm"]["output_dir"] = str((src.parent / normalized["symm"]["output_dir"]).resolve())
    raw["output"] = {"dir": str((tmp_path / "k1_regression").resolve())}
    cfg_path = tmp_path / "mote2_3.89_K1_regression.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    results = run_configured_model(cfg_path)

    plot_comparison = results["plot_comparison"]
    assert plot_comparison["aligned_rms_error_meV"] < 10.0
    assert plot_comparison["aligned_max_abs_error_meV"] < 20.0

    active_terms = json.loads(((tmp_path / "k1_regression") / "active_terms.json").read_text(encoding="utf-8"))
    assert len(active_terms) >= 80
    assert sum(1 for term in active_terms if term["tag"] == "inter") >= 30
