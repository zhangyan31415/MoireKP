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
        "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
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


def _write_tiny_project_config_with_symm(tmp_path: Path, project: dict, symm: dict) -> Path:
    cfg_path = _write_tiny_project_config(tmp_path, project)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["symm"] = symm
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_cached_symmetry_basis(symm_dir: Path, report: dict) -> None:
    symm_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"project_basis": report}
    np.savez_compressed(
        symm_dir / "representations.npz",
        __metadata_json__=np.asarray(json.dumps(metadata, sort_keys=True)),
        TR=np.eye(2, dtype=np.complex128),
    )
    (symm_dir / "residuals.csv").write_text("operation,status\nTR,ok\n", encoding="utf-8")
    (symm_dir / "summary.md").write_text("# KP Symmetry Projection\n", encoding="utf-8")


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
            "gauge": {"method": "auto_scdm", "validate_with_symmetry": True},
        },
    )

    cli.cmd_project_from_config(str(cfg_path))

    out_dir = tmp_path / "project"
    assert (out_dir / "heff.npy").exists()
    assert (out_dir / "basis.npz").exists()
    assert (out_dir / "basis.md").exists()

    with np.load(out_dir / "basis.npz", allow_pickle=True) as payload:
        assert payload["norb_fix_list"].tolist() == [[[[0, 1.0]]], [[[0, 1.0]]]]
    report_text = (out_dir / "basis.md").read_text(encoding="utf-8")
    assert "gauge_mode: `auto_scdm`" in report_text
    assert "symmetry_closure_quality: `not_available`" in report_text
    assert "condition_number" in report_text


def test_project_auto_uses_symmetry_validated_resolved_anchors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    selected = [[[[1, 1.0]]], [[[1, 1.0]]]]

    def fake_symm(_cfg_path: str, *, developer_outputs=None):
        symm_dir = tmp_path / "symm"
        report = {
            "gauge_mode": "auto_scdm",
            "resolved_norb_fix_list": selected,
            "selections": [],
            "metric": {"type": "orthonormal", "basis_is_orthonormal": True},
            "state_selection_quality": {"status": "not_evaluated"},
            "gauge_anchor_quality": {"status": "ok", "sigma_min": 1.0, "condition_number": 1.0},
            "symmetry_closure_quality": {
                "status": "validated",
                "selected_candidate_id": "test_candidate",
            },
            "warnings": [],
        }
        _write_cached_symmetry_basis(symm_dir, report)
        return {
            "project_basis": {
                "gauge_mode": "auto_scdm",
                "resolved_norb_fix_list": selected,
            }
        }

    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fake_symm)

    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": {"method": "auto_scdm", "validate_with_symmetry": True},
        },
        {
            "enable": True,
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["TR"],
        },
    )
    (tmp_path / "tapw_symmetry").mkdir()

    cli.cmd_project_from_config(str(cfg_path))

    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as payload:
        assert payload["norb_fix_list"].tolist() == selected
    assert "symmetry_closure_quality: `validated`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


def test_project_auto_reuses_cached_symmetry_basis_selection(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    selected = [[[[1, 1.0]]], [[[1, 1.0]]]]
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    report = {
        "gauge_mode": "auto_scdm",
        "resolved_norb_fix_list": selected,
        "selections": [],
        "metric": {"type": "orthonormal", "basis_is_orthonormal": True},
        "state_selection_quality": {"status": "not_evaluated"},
        "gauge_anchor_quality": {"status": "ok", "sigma_min": 1.0, "condition_number": 1.0},
        "symmetry_closure_quality": {
            "status": "validated",
            "selected_candidate_id": "cached_candidate",
        },
        "warnings": [],
    }
    _write_cached_symmetry_basis(symm_dir, report)

    def fail_symm(_cfg_path: str, *, developer_outputs=None):
        raise AssertionError("kp project should reuse cached symmetry metadata")

    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fail_symm)

    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": "auto",
        },
        {
            "enable": True,
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["TR"],
        },
    )
    (tmp_path / "tapw_symmetry").mkdir()

    cli.cmd_project_from_config(str(cfg_path))

    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as basis_payload:
        assert basis_payload["norb_fix_list"].tolist() == selected
    assert "symmetry_closure_quality: `validated`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


def test_project_auto_defers_symmetry_validation_without_cached_basis(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )

    def fail_symm(_cfg_path: str, *, developer_outputs=None):
        raise AssertionError("kp project should not run heavy kp symm inline without a cache")

    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fail_symm)

    cfg_path = _write_tiny_project_config_with_symm(
        tmp_path,
        {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "out_dir": "project",
            "nlow_state_list": [[0], [0]],
            "gauge": "auto",
        },
        {
            "enable": True,
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["TR"],
        },
    )
    (tmp_path / "tapw_symmetry").mkdir()

    cli.cmd_project_from_config(str(cfg_path))

    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as basis_payload:
        assert basis_payload["norb_fix_list"].tolist() == [[[[0, 1.0]]], [[[0, 1.0]]]]
    assert "symmetry_closure_quality: `deferred`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


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
