from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import kp.cli as cli


def _write_tiny_project_config(tmp_path: Path, project: dict) -> Path:
    hamk = np.diag([0.0, 10.0, 1.0, 11.0, 100.0, 110.0, 101.0, 111.0]).astype(
        np.complex128
    )
    np.save(tmp_path / "hamk.npy", hamk[np.newaxis, :, :])
    np.save(tmp_path / "q1.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(tmp_path / "q2.npy", np.array([[0.0, 0.0]], dtype=float))

    cfg = {
        "material": {
            "name": "tiny",
            "hamk_file": "hamk.npy",
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_layer_list": [1, 1],
            "num_orb_per_layer": [2],
        },
        "plot": {"hamk_index": 0},
        "project": project,
    }
    cfg_path = tmp_path / "source.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_project_gauge_auto_writes_basis_selection_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )

    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": "auto",
        },
    )

    cli.cmd_project_from_config(str(cfg_path))

    out_dir = tmp_path / "project"
    assert (out_dir / "heff_list.npy").exists()
    assert (out_dir / "basis_selection.json").exists()
    assert (out_dir / "basis_selection.md").exists()
    assert (out_dir / "auto_norb_fix_list.yaml").exists()

    report = json.loads((out_dir / "basis_selection.json").read_text(encoding="utf-8"))
    assert report["gauge_mode"] == "auto_scdm"
    assert report["resolved_norb_fix_list"] == [[[[0, 1.0]]], [[[0, 1.0]]]]
    assert report["symmetry_closure_quality"]["status"] == "not_available"
    assert report["symmetry_closure_quality"]["subspace_leakage"] is None
    assert report["selections"][0]["sigma_min"] > 0.0
    assert "condition_number" in report["selections"][0]


def test_project_rejects_manual_norb_fix_with_auto_gauge(tmp_path: Path) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
            "gauge": "auto",
        },
    )

    with np.testing.assert_raises_regex(ValueError, "manual norb_fix_list.*gauge"):
        cli.cmd_project_from_config(str(cfg_path))


def test_project_requires_auto_gauge_or_manual_norb_fix_list(tmp_path: Path) -> None:
    cfg_path = _write_tiny_project_config(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
        },
    )

    with pytest.raises(ValueError, match="gauge: auto|manual norb_fix_list"):
        cli.cmd_project_from_config(str(cfg_path))
