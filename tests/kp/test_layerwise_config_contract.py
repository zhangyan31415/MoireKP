from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from kp.model.pipeline import load_model_config
from kp.symmetry.projection import _sector_orbital_counts, _validate_project_layer_lists


def _write_minimal_layerwise_model(tmp_path: Path, *, n_orb: list[int], nlow_state: list[int] | None = None) -> Path:
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    internal_dim = len(q1) * int(n_orb[0]) + len(q2) * int(sum(n_orb[1:]))
    kpoints = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=float)
    heff = np.stack([np.diag(np.arange(internal_dim)), np.diag(np.arange(internal_dim) + 0.1)]).astype(np.complex128)

    np.save(tmp_path / "q1.npy", q1)
    np.save(tmp_path / "q2.npy", q2)
    np.save(tmp_path / "kpoints.npy", kpoints)
    project_dir = tmp_path / "outputs" / "K1" / "q06" / "projection"
    project_dir.mkdir(parents=True)
    np.save(project_dir / "heff.npy", heff)
    symm_dir = tmp_path / "outputs" / "K1" / "q06" / "symmetry"
    symm_dir.mkdir(parents=True)
    (symm_dir / "manifest.json").write_text(
        json.dumps({"frame": {"q_transform": {"rotation_deg": 0.0}}, "operations": []}),
        encoding="utf-8",
    )

    raw = {
        "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
        "material": {
            "num_layers": 3,
            "num_layer_list": [1, 2],
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "plot": {},
        "project": {},
        "symmetry_source": {"type": "kp_symm_output", "path": "outputs/K1/q06/symmetry"},
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": "Gamma",
            "mode": "single_valley",
            "active_valleys": ["Gamma"],
            "spin_convention": "spinful",
            "allowed_internal_symmetries": [],
            "external_sewing_symmetries": [],
        },
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": n_orb,
            "nlow_state": nlow_state if nlow_state is not None else n_orb,
            "bM": {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]},
            "harmonics": {"intra": {1: "zero"}, "inter": {1: "zero"}},
            "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
            "symmetry_map": {"Kinect": [], "intra": [], "inter": []},
        },
        "fit": {"indices": [0]},
        "output": {"dir": "out"},
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_model_layerwise_n_orb_resolves_to_qset_counts(tmp_path: Path) -> None:
    cfg_path = _write_minimal_layerwise_model(tmp_path, n_orb=[1, 1, 0])

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (1, 1)
    assert cfg.nlow_state == [1, 1]
    assert cfg.raw["model"]["n_orb_layerwise"] == [1, 1, 0]
    assert cfg.raw["model"]["n_orb_resolved_qset"] == [1, 1]


def test_model_layerwise_n_orb_can_empty_first_qset(tmp_path: Path) -> None:
    cfg_path = _write_minimal_layerwise_model(tmp_path, n_orb=[0, 2, 2])

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (0, 4)
    assert cfg.nlow_state == [0, 4]
    assert cfg.raw["model"]["n_orb_resolved_qset"] == [0, 4]


def test_model_infers_layerwise_n_orb_from_projection_basis(tmp_path: Path) -> None:
    cfg_path = _write_minimal_layerwise_model(tmp_path, n_orb=[1, 1, 0])
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"].pop("n_orb")
    raw["model"].pop("nlow_state")
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    projection_dir = tmp_path / "outputs" / "K1" / "q06" / "projection"
    np.savez(
        projection_dir / "basis.npz",
        nlow_state_list=np.asarray([[0], [0], []], dtype=object),
    )

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (1, 1)
    assert cfg.nlow_state == [1, 1]
    assert cfg.raw["model"]["n_orb_resolution"]["source"] == "projection_basis"


def test_model_rejects_n_orb_that_conflicts_with_projection_basis(tmp_path: Path) -> None:
    cfg_path = _write_minimal_layerwise_model(tmp_path, n_orb=[0, 2, 2])
    projection_dir = tmp_path / "outputs" / "K1" / "q06" / "projection"
    np.savez(
        projection_dir / "basis.npz",
        nlow_state_list=np.asarray([[0], [0], []], dtype=object),
    )

    with pytest.raises(ValueError, match="model.n_orb.*projection"):
        load_model_config(cfg_path)


def test_model_omitted_n_orb_requires_projection_metadata(tmp_path: Path) -> None:
    cfg_path = _write_minimal_layerwise_model(tmp_path, n_orb=[1, 1, 0])
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"].pop("n_orb")
    raw["model"].pop("nlow_state")
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.n_orb.*projection/basis.npz"):
        load_model_config(cfg_path)


def test_model_rejects_wrong_layerwise_n_orb_length(tmp_path: Path) -> None:
    cfg_path = _write_minimal_layerwise_model(tmp_path, n_orb=[1, 1, 0])
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["n_orb"] = [1, 1, 0, 0]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.n_orb"):
        load_model_config(cfg_path)


def test_model_rejects_legacy_qset_n_orb_for_multilayer_source(tmp_path: Path) -> None:
    cfg_path = _write_minimal_layerwise_model(tmp_path, n_orb=[1, 1, 0])
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["n_orb"] = [1, 1]
    raw["model"]["nlow_state"] = [1, 1]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.n_orb"):
        load_model_config(cfg_path)


def test_project_layerwise_counts_follow_num_layer_list() -> None:
    q1 = np.zeros((2, 2), dtype=float)
    q2 = np.zeros((2, 2), dtype=float)

    assert _sector_orbital_counts(
        q1,
        q2,
        [[134, 135], [136, 137], []],
        low_dim=None,
        num_layer_list=[1, 2],
    ) == (2, 2)
    assert _sector_orbital_counts(
        q1,
        q2,
        [[], [134, 135], [136, 137]],
        low_dim=None,
        num_layer_list=[1, 2],
    ) == (0, 4)


def test_project_rejects_legacy_source_group_rows_for_multilayer_source() -> None:
    with pytest.raises(ValueError, match="physical-layer rows"):
        _validate_project_layer_lists(
            [[134, 135], [136, 137]],
            [[[390], [177]], [[319], [106]]],
            num_layer_list=[1, 2],
            context="project",
        )


def test_project_rejects_wrong_layer_row_count_for_multilayer_source() -> None:
    with pytest.raises(ValueError, match="physical-layer rows"):
        _validate_project_layer_lists(
            [[134, 135]],
            [[[390], [177]]],
            num_layer_list=[1, 2],
            context="project",
        )
