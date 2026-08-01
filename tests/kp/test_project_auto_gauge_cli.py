from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import kp.cli as cli
from kp.symmetry.projection import resolve_symmetry_validated_project_gauge
from kp.basis.selection import GaugeAnchorReport
from kp.selection_artifact import CertificationStatus, load_selection_artifact


def _write_tiny_project_config(tmp_path: Path, project: dict) -> Path:
    hamk = np.diag([0.0, 10.0, 1.0, 11.0, 100.0, 110.0, 101.0, 111.0]).astype(
        np.complex128
    )
    np.save(tmp_path / "hamk.npy", hamk[np.newaxis, :, :])
    np.save(tmp_path / "q1.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(tmp_path / "q2.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(tmp_path / "kpoints.npy", np.zeros((1, 3), dtype=float))
    (tmp_path / "bands.txt").write_text("0.0 1.0\n", encoding="utf-8")

    cfg = {
        "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
        "material": {
            "name": "tiny",
            "hamk_file": "hamk.npy",
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
            "kpoints_file": "kpoints.npy",
            "band_file": "bands.txt",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_layer_list": [1, 1],
            "num_orb_per_layer": [2],
        },
        "plot": {"hamk_index": 0},
        "kpath": {"tmat": np.eye(3).tolist()},
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


def _write_tiny_tapw_c3_source(symm_dir: Path) -> None:
    rep_dir = symm_dir / "representations" / "K1"
    rep_dir.mkdir(parents=True)
    raw_h = np.eye(8, dtype=np.complex128)
    np.savez(rep_dir / "C3_rawH.npz", matrix=raw_h)
    manifest = {
        "operations": {
            "K1": {
                "C3": {
                    "antiunitary": False,
                    "raw_h_operator_file": "K1/C3_rawH.npz",
                    "k_pairs": [[0, 0]],
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                }
            }
        }
    }
    (symm_dir / "representations" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _explicit_project_config() -> dict:
    return {
        "mode": "K1",
        "workers": 1,
        "downfold_method": "first_order",
        "out_dir": "project",
        "nlow_state_list": [[0], [0]],
        "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
    }


def test_inspect_and_explicit_project_share_input_but_only_project_finalizes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    monkeypatch.setattr(
        cli,
        "plot_inspect_band_and_qblock",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())

    inspect = cli.cmd_plot_from_config(str(cfg_path))
    project = cli.cmd_project_from_config(str(cfg_path))

    assert inspect.status is CertificationStatus.PENDING
    assert inspect.identity is None
    assert project.status is CertificationStatus.UNVERIFIED_OVERRIDE
    assert project.identity is not None
    assert (
        inspect.selection_input.selection_input_identity_hash
        == project.selection_input.selection_input_identity_hash
    )
    marker = load_selection_artifact(tmp_path / "project" / "selection_artifact.json")
    assert marker == project.artifact


def test_active_indices_is_recorded_as_explicit_unverified_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())
    monkeypatch.setattr(
        cli,
        "_cmd_project_from_config_impl",
        lambda *_args, **_kwargs: cli._ExplicitProjectSelectionMaterialization(
            candidate_id="active-indices-0-1",
            candidate_dimension=2,
            basis_handoff_hash="1" * 64,
            authoritative_heff_hash="2" * 64,
            heff_k_indices_hash="3" * 64,
        ),
    )

    result = cli.cmd_project_from_config(str(cfg_path), {"active_indices": "0,1"})

    assert result.status is CertificationStatus.UNVERIFIED_OVERRIDE
    assert result.selection_input.selection_mode == "explicit"
    assert result.identity is not None
    assert result.identity.certification_evidence is None


def test_later_project_failure_leaves_pending_instead_of_previous_final(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())
    cli.cmd_project_from_config(str(cfg_path))

    def fail_projection(*_args, **_kwargs):
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(cli, "project_heff_full", fail_projection)
    with pytest.raises(RuntimeError, match="injected projection failure"):
        cli.cmd_project_from_config(str(cfg_path))

    marker = load_selection_artifact(tmp_path / "project" / "selection_artifact.json")
    assert marker.certification_status is CertificationStatus.PENDING


def test_keyboard_interrupt_releases_pending_selection_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_tiny_project_config(tmp_path, _explicit_project_config())
    real_begin = cli.begin_case_selection
    retained_sessions = []

    def retain_session(**kwargs):
        session = real_begin(**kwargs)
        retained_sessions.append(session)
        return session

    def interrupt_project(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "begin_case_selection", retain_session)
    monkeypatch.setattr(cli, "_cmd_project_from_config_impl", interrupt_project)
    with pytest.raises(KeyboardInterrupt):
        cli.cmd_project_from_config(str(cfg_path))

    project_dir = tmp_path / "project"
    marker = load_selection_artifact(project_dir / "selection_artifact.json")
    assert marker.certification_status is CertificationStatus.PENDING
    assert retained_sessions
    with (project_dir / ".selection-artifact.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_project_auto_requires_source_symmetry(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="auto gauge.*symm.tapw_symmetry_dir"):
        cli.cmd_project_from_config(str(cfg_path))

    assert not (tmp_path / "project" / "heff.npy").exists()


def test_symmetry_gauge_resolver_needs_no_project_artifacts_and_writes_nothing(tmp_path: Path) -> None:
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
            "valley": "K1",
            "spin": "up",
            "tapw_symmetry_dir": "tapw_symmetry",
            "output_dir": "symm",
            "operations": ["C3"],
            "tolerance": 1.0e-8,
        },
    )
    _write_tiny_tapw_c3_source(tmp_path / "tapw_symmetry")

    report = resolve_symmetry_validated_project_gauge(str(cfg_path))

    assert report.gauge_mode == "auto_scdm"
    assert report.symmetry_closure_quality["status"] == "validated"
    assert not (tmp_path / "project").exists()
    assert not (tmp_path / "symm").exists()


def test_project_auto_uses_symmetry_validated_resolved_anchors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    selected = [[[[1, 1.0]]], [[[1, 1.0]]]]

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

    calls: list[str] = []
    resolved_project_configs: list[dict] = []

    def fake_resolver(cfg_path: str, *, project_config=None):
        calls.append(cfg_path)
        resolved_project_configs.append(dict(project_config or {}))
        return GaugeAnchorReport(**report)

    def fail_full_symm(*_args, **_kwargs):
        raise AssertionError("kp project must not run the full kp symm writer")

    monkeypatch.setattr(cli, "resolve_symmetry_validated_project_gauge", fake_resolver, raising=False)
    monkeypatch.setattr(cli, "run_symmetry_projection_from_config", fail_full_symm)

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

    cli.cmd_project_from_config(str(cfg_path), {"k_indices": "0"})

    assert calls == [str(cfg_path)]
    assert resolved_project_configs[0]["k_indices"] == "0"
    with np.load(tmp_path / "project" / "basis.npz", allow_pickle=True) as payload:
        assert payload["norb_fix_list"].tolist() == selected
    assert "symmetry_closure_quality: `validated`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


def test_project_auto_does_not_trust_unbound_cached_symmetry_basis(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )
    stale_selected = [[[[1, 1.0]]], [[[1, 1.0]]]]
    resolved_selected = [[[[0, 1.0]]], [[[0, 1.0]]]]
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    report = {
        "gauge_mode": "auto_scdm",
        "resolved_norb_fix_list": stale_selected,
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

    resolved_report = dict(report)
    resolved_report["resolved_norb_fix_list"] = resolved_selected
    resolved_report["symmetry_closure_quality"] = {
        "status": "validated",
        "selected_candidate_id": "fresh_candidate",
    }

    def fake_resolver(_cfg_path: str, *, project_config=None):
        return GaugeAnchorReport(**resolved_report)

    def fail_symm(*_args, **_kwargs):
        raise AssertionError("kp project must not run the full kp symm writer")

    monkeypatch.setattr(cli, "resolve_symmetry_validated_project_gauge", fake_resolver, raising=False)
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
        assert basis_payload["norb_fix_list"].tolist() == resolved_selected
    assert "symmetry_closure_quality: `validated`" in (tmp_path / "project" / "basis.md").read_text(encoding="utf-8")


def test_project_auto_resolves_symmetry_without_running_full_symm(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "plot_eigs_scatter",
        lambda *_args, out, **_kwargs: Path(out).write_text("plot", encoding="utf-8"),
    )

    selected = [[[[1, 1.0]]], [[[1, 1.0]]]]
    report = {
        "gauge_mode": "auto_scdm",
        "resolved_norb_fix_list": selected,
        "selections": [],
        "metric": {"type": "orthonormal", "basis_is_orthonormal": True},
        "state_selection_quality": {"status": "not_evaluated"},
        "gauge_anchor_quality": {"status": "ok", "sigma_min": 1.0, "condition_number": 1.0},
        "symmetry_closure_quality": {
            "status": "validated",
            "selected_candidate_id": "resolved_candidate",
        },
        "warnings": [],
    }

    def fake_resolver(_cfg_path: str, *, project_config=None):
        return GaugeAnchorReport(**report)

    def fail_symm(*_args, **_kwargs):
        raise AssertionError("kp project must not run the full kp symm writer")

    monkeypatch.setattr(cli, "resolve_symmetry_validated_project_gauge", fake_resolver, raising=False)
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

    with pytest.raises(ValueError, match="auto gauge.*symm.tapw_symmetry_dir"):
        cli.cmd_project_from_config(str(cfg_path))
