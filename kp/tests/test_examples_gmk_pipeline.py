from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from kp.model.configured import build_moire_config_from_file, load_model_config, run_configured_model
from kp.src.moire_refactored import build_model


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"

GMK_PRODUCTION_CONFIGS = {
    "mote2_3.89_K1": EXAMPLES_ROOT / "mote2/3.89/kp/configs/model/production/mote2_3.89_K1.yaml",
    "mgi2_3.89_Gamma": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/production/mgi2_3.89_Gamma.yaml",
    "mgi2_3.89_M1": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/production/mgi2_3.89_M1.yaml",
}

GMK_RELEASE_CONFIGS = {
    "mote2_3.89_K1": EXAMPLES_ROOT / "mote2/3.89/kp/configs/model/release/mote2_3.89_K1.yaml",
    "mote2_3.89_K1_spinful": EXAMPLES_ROOT / "mote2/3.89/kp/configs/model/release/mote2_3.89_K1_spinful.yaml",
    "mgi2_3.89_Gamma": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/release/mgi2_3.89_Gamma.yaml",
    "mgi2_3.89_M1": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/release/mgi2_3.89_M1.yaml",
    "mgi2_3.89_M1_spinful": EXAMPLES_ROOT / "mgi2/3.89/kp/configs/model/release/mgi2_3.89_M1_spinful.yaml",
}

GMK_REFERENCE_OUTPUTS = {
    "mote2_3.89_K1_toy": EXAMPLES_ROOT / "mote2/3.89/kp/outputs/model/reference/mote2_3.89_K1_toy",
    "mgi2_3.89_Gamma_toy_legacy": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/reference/mgi2_3.89_Gamma_toy_legacy",
    "mgi2_3.89_M1_q7": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/reference/mgi2_3.89_M1_q7",
}

GMK_PRODUCTION_OUTPUTS = {
    "mote2_3.89_K1": EXAMPLES_ROOT / "mote2/3.89/kp/outputs/model/production/mote2_3.89_K1",
    "mgi2_3.89_Gamma": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/production/mgi2_3.89_Gamma",
    "mgi2_3.89_M1": EXAMPLES_ROOT / "mgi2/3.89/kp/outputs/model/production/mgi2_3.89_M1",
}


def test_gmk_production_configs_follow_unified_layout() -> None:
    for case_id, path in GMK_PRODUCTION_CONFIGS.items():
        assert path.exists(), path
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["source_config"].startswith("../../source/")
        assert raw["output"]["dir"] == f"../../../outputs/model/production/{case_id}"
        assert (path.parent / raw["source_config"]).resolve().exists()


def test_gmk_model_configs_keep_user_harmonics_minimal() -> None:
    for path in [*GMK_PRODUCTION_CONFIGS.values(), *GMK_RELEASE_CONFIGS.values()]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = raw.get("model", {})
        assert "vectors" not in model, path
        assert "term_templates" not in model, path
        for kind, spec in model.get("harmonics", {}).items():
            assert isinstance(spec, int), (path, kind, spec)
        text = path.read_text(encoding="utf-8")
        assert "harmonics_source" not in text
        assert "representatives:" not in text
        for forbidden in (
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


def test_gmk_production_configs_load() -> None:
    for case_id, path in GMK_PRODUCTION_CONFIGS.items():
        cfg = load_model_config(path)
        assert cfg.output_dir.name == case_id
        assert cfg.path == path
        assert cfg.valley_model["valley_type"] in {"Gamma", "K", "M"}


def test_gmk_production_configs_use_uniform_term_symmetry_sets() -> None:
    for path in GMK_PRODUCTION_CONFIGS.values():
        cfg = load_model_config(path)
        by_tag = {
            tag: [op["name"] for op in cfg.symmetry_map[tag]]
            for tag in ("Kinect", "Onsite", "intra", "inter")
        }
        assert len({tuple(names) for names in by_tag.values()}) == 1, by_tag


def test_gmk_reference_outputs_follow_unified_layout() -> None:
    for path in GMK_REFERENCE_OUTPUTS.values():
        assert path.exists(), path
        assert path.parent.name == "reference"


def test_gmk_existing_production_outputs_follow_unified_layout() -> None:
    for path in GMK_PRODUCTION_OUTPUTS.values():
        assert path.exists(), path
        assert path.parent.name == "production"


def test_m_production_config_builds_named_sector_model() -> None:
    path = GMK_PRODUCTION_CONFIGS["mgi2_3.89_M1"]
    moire_cfg, _model_cfg = build_moire_config_from_file(path)
    model = build_model(moire_cfg)
    assert moire_cfg.sectors
    assert {sector["name"] for sector in moire_cfg.sectors} == {"bottom", "top"}
    assert any(term.key.layer_from == 2 and term.key.layer_to == 1 for term in model.terms.values())


def test_k1_production_example_stays_mev_scale_and_keeps_active_set(tmp_path: Path) -> None:
    src = GMK_PRODUCTION_CONFIGS["mote2_3.89_K1"]
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["source_config"] = str((src.parent / raw["source_config"]).resolve())
    raw["kpath"]["file"] = str((src.parent / raw["kpath"]["file"]).resolve())
    raw["symmetry_source"]["path"] = str((src.parent / raw["symmetry_source"]["path"]).resolve())
    raw["output"]["dir"] = str((tmp_path / "k1_regression").resolve())
    cfg_path = tmp_path / "mote2_3.89_K1_regression.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    results = run_configured_model(cfg_path)

    plot_comparison = results["plot_comparison"]
    assert math.isclose(plot_comparison["aligned_rms_error_meV"], 1.225565, abs_tol=1.0e-3)
    assert math.isclose(plot_comparison["aligned_max_abs_error_meV"], 4.941980, abs_tol=1.0e-3)

    active_terms = json.loads(((tmp_path / "k1_regression") / "active_terms.json").read_text(encoding="utf-8"))
    assert len(active_terms) == 94
    assert sum(1 for term in active_terms if term["tag"] == "inter") == 45
