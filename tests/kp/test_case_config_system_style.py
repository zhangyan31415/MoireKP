from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kp.config.case import normalize_case_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _tapw_style_system_config() -> dict:
    return {
        "system": {
            "name": "MgI2",
            "output": "../runs/mgi2_gamma",
            "tapw_output": "../../mgi2_3.89/tapw/outputs",
            "layers": [1, 1],
            "spin": "all",
            "orbital_order": "I-s3p2d2,Mg-s2p2,I-s3p2d2",
            "cell": [
                [60.0, 0.0, 0.0],
                [-30.0, 51.9615242271, 0.0],
                [0.0, 0.0, 50.0],
            ],
        },
        "project": {
            "valley": "Gamma",
            "q_shell": 4,
            "efermi": -4.168612,
            "target": "valence",
            "nlow_state_list": [[40, 41], [42, 43]],
            "downfold_method": "fixed_schur",
            "e_ref": -5.74,
            "workers": 64,
        },
        "symmetry": {"tolerance": 0.01},
        "model": {},
    }


def test_tracked_mgi2_gamma_q04_keeps_reviewed_automatic_acceptance_policy() -> None:
    config_path = (
        REPO_ROOT
        / "examples"
        / "mgi2_3.89"
        / "kp"
        / "configs"
        / "mgi2_3.89_Gamma_spinful_q04.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = raw["project"]
    model = raw["model"]

    assert model["target_bands"] == "top"
    assert model["fit"] == {
        "method": "linear",
        "kpoints": [0, 40],
        "bands": 10,
        "one_sided_weight": 0.0,
        "two_sided_weight": 0.0,
    }
    assert project["downfold_method"] == "fixed_schur"
    assert project["e_ref"] == -5.74
    assert model["harmonics"] == {"intralayer": 3, "interlayer": 3}
    assert model["max_order"] == {
        "kinetic": 10,
        "intralayer": 6,
        "interlayer": 10,
    }


def test_tapw_style_system_config_derives_source_and_run_layout(tmp_path: Path) -> None:
    cfg_path = tmp_path / "examples" / "autoproject" / "configs" / "mgi2_gamma.yaml"

    cfg = normalize_case_config(_tapw_style_system_config(), config_path=cfg_path)

    assert cfg["case"] == {
        "profile": "Gamma",
        "q_shell": "q04",
        "output_root": "../runs/mgi2_gamma",
    }
    assert cfg["material"] == {
        "name": "MgI2",
        "num_layer_list": [1, 1],
        "num_layers": 2,
        "spin": "all",
        "orbital_order": "I-s3p2d2,Mg-s2p2,I-s3p2d2",
        "efermi": -4.168612,
        "hamk_file": "../../mgi2_3.89/tapw/outputs/Gamma/q04/band/hamiltonian_k.npy",
        "kpoints_file": "../../mgi2_3.89/tapw/outputs/Gamma/q04/band/kpoints.npy",
        "qset1_file": "../../mgi2_3.89/tapw/outputs/Gamma/q04/band/g_vectors_group1.npy",
        "qset2_file": "../../mgi2_3.89/tapw/outputs/Gamma/q04/band/g_vectors_group2.npy",
        "band_file": "../../mgi2_3.89/tapw/outputs/Gamma/q04/band/energies_vbm.txt",
    }
    assert cfg["project"]["mode"] == "Gamma"
    assert cfg["project"]["selection"] == {"mode": "explicit"}
    assert cfg["project"]["out_dir"] == "../runs/mgi2_gamma/Gamma/q04/projection"
    assert cfg["plot"]["mode"] == "Gamma"
    assert cfg["plot"]["target"] == "valence"
    assert cfg["plot"]["out"] == (
        "../runs/mgi2_gamma/Gamma/q04/inspect/bands_and_qblocks.pdf"
    )
    assert cfg["plot"]["data_out"] == (
        "../runs/mgi2_gamma/Gamma/q04/inspect/spectrum.txt"
    )
    assert cfg["plot"]["nlow_state_list"] == [[40, 41], [42, 43]]
    assert cfg["symm"]["tapw_symmetry_dir"] == (
        "../../mgi2_3.89/tapw/outputs/Gamma/q04/symmetry"
    )
    assert cfg["symm"]["output_dir"] == "../runs/mgi2_gamma/Gamma/q04/symmetry"
    assert cfg["kpath"] == {"tmat": _tapw_style_system_config()["system"]["cell"]}


def test_tapw_style_system_config_uses_conduction_reference_file(tmp_path: Path) -> None:
    raw = _tapw_style_system_config()
    raw["project"]["valley"] = "M1"
    raw["project"]["q_shell"] = "q07"
    raw["project"]["target"] = "conduction"

    cfg = normalize_case_config(raw, config_path=tmp_path / "mgi2_m.yaml")

    assert cfg["case"]["q_shell"] == "q07"
    assert cfg["project"]["target"] == "conduction"
    assert cfg["plot"]["target"] == "conduction"
    assert cfg["material"]["band_file"].endswith("/M1/q07/band/energies_cbm.txt")


def test_tapw_style_system_config_accepts_bands_kpath_workflow(tmp_path: Path) -> None:
    raw = _tapw_style_system_config()
    raw["bands"] = {
        "kpath": {
            "labels": ["G", "M"],
            "coordinates": {"G": [0.0, 0.0], "M": [0.5, 0.0]},
        }
    }

    cfg = normalize_case_config(raw, config_path=tmp_path / "with_bands.yaml")

    assert cfg["kpath"] == {
        "labels": ["G", "M"],
        "coordinates": {"G": [0.0, 0.0], "M": [0.5, 0.0]},
        "tmat": raw["system"]["cell"],
    }


def test_tapw_style_system_config_rejects_invalid_or_mixed_schema(tmp_path: Path) -> None:
    raw = _tapw_style_system_config()
    raw["project"]["target"] = "conducton"
    with pytest.raises(ValueError, match="project.target"):
        normalize_case_config(raw, config_path=tmp_path / "bad_target.yaml")

    raw = _tapw_style_system_config()
    raw["output"] = {"dir": "internal-model-output"}
    with pytest.raises(ValueError, match="system.*top-level|top-level.*system"):
        normalize_case_config(raw, config_path=tmp_path / "unknown_section.yaml")

    raw = _tapw_style_system_config()
    raw["case"] = {}
    with pytest.raises(ValueError, match="system.*legacy|legacy.*system"):
        normalize_case_config(raw, config_path=tmp_path / "mixed.yaml")


def test_explicit_state_list_preserves_selection_diagnostics(tmp_path: Path) -> None:
    raw = _tapw_style_system_config()
    raw["project"]["selection"] = {
        "mode": "auto",
        "validation_bands": 12,
        "qblock_ylim": [-0.5, 0.2],
    }

    cfg = normalize_case_config(raw, config_path=tmp_path / "explicit.yaml")

    assert cfg["project"]["selection"] == {
        "mode": "explicit",
        "validation_bands": 12,
        "qblock_ylim": [-0.5, 0.2],
    }
