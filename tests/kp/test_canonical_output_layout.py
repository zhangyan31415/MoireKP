from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import kp.cli as cli
from kp.config.case import normalize_case_config
from kp.symmetry import projection as projection_mod


def _canonical_case_config() -> dict:
    return {
        "case": {
            "profile": "K1",
            "q_shell": "q06",
            "output_root": "../outputs",
        },
        "valley": "K1",
        "spin": "up",
        "material": {
            "hamk_file": "hamk.npy",
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
            "spin": "up",
            "energy_unit": "eV",
            "num_layers": 2,
            "num_orb_per_layer": [1],
        },
        "project": {
            "mode": "K1",
            "workers": 1,
            "downfold_method": "first_order",
            "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        },
        "symm": {"operations": ["C3z"]},
        "model": {"n_orb": [1, 1]},
    }


def test_canonical_case_defaults_point_all_workflows_to_one_case_directory(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "kp" / "configs"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "K1_q06.yaml"

    cfg = normalize_case_config(_canonical_case_config(), config_path=cfg_path)

    assert cfg["project"]["out_dir"] == "../outputs/K1/q06/projection"
    assert cfg["plot"]["out"] == "../outputs/K1/q06/inspect/scatter.png"
    assert cfg["plot"]["data_out"] == "../outputs/K1/q06/inspect/spectrum.txt"
    assert cfg["symm"]["output_dir"] == "../outputs/K1/q06/symmetry"
    assert cfg["output"]["dir"] == "../outputs/K1/q06/model"


def test_legacy_unified_defaults_keep_existing_split_output_directories(tmp_path: Path) -> None:
    cfg_path = tmp_path / "kp" / "configs" / "legacy_case.yaml"
    raw = _canonical_case_config()
    raw["case"] = "legacy_case"

    cfg = normalize_case_config(raw, config_path=cfg_path)

    assert cfg["project"]["out_dir"] == "../outputs/project/legacy_case"
    assert cfg["plot"]["out"] == "../outputs/project/legacy_case/heff_scatter.png"
    assert cfg["symm"]["output_dir"] == "../outputs/symm/legacy_case"
    assert cfg["output"]["dir"] == "../outputs/model/legacy_case"


def test_kp_inspect_alias_dispatches_to_plot() -> None:
    calls = []
    original = cli.cmd_plot_from_config
    try:
        cli.cmd_plot_from_config = lambda config: calls.append(config)
        cli.main(["inspect", "-c", "K1_q06.yaml"])
    finally:
        cli.cmd_plot_from_config = original

    assert calls == ["K1_q06.yaml"]


def test_inspect_canonical_writes_user_facing_files(monkeypatch, tmp_path: Path) -> None:
    q = np.zeros((1, 2), dtype=float)
    hamk = np.zeros((1, 4, 4), dtype=np.complex128)

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: hamk)
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_get_H_block(*_args, **_kwargs):
        eigs = np.empty(1, dtype=object)
        eigs[0] = np.array([-0.2, 0.1])
        vecs = np.empty(1, dtype=object)
        vecs[0] = np.eye(2, dtype=np.complex128)
        return eigs, vecs, np.empty(0, dtype=object), np.empty(0, dtype=object)

    monkeypatch.setattr(cli, "get_H_block", fake_get_H_block)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg = _canonical_case_config()
    cfg["plot"] = {"mode": "gamma"}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cli.main(["inspect", "-c", str(cfg_path)])

    inspect_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "inspect"
    assert (inspect_dir / "spectrum.txt").exists()
    assert (inspect_dir / "scatter.png").read_text(encoding="utf-8") == "plot"
    assert (inspect_dir / "candidates.md").exists()
    assert (inspect_dir / "blocks.csv").exists()


def test_project_canonical_writes_compact_outputs_and_legacy_aliases(monkeypatch, tmp_path: Path) -> None:
    q = np.zeros((1, 2), dtype=float)
    hamk = np.zeros((1, 4, 4), dtype=np.complex128)

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: hamk)
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))
    monkeypatch.setattr(
        cli,
        "project_heff_full",
        lambda *_args, **_kwargs: (
            np.eye(2, dtype=np.complex128),
            np.array([0.0, 1.0], dtype=float),
            np.eye(2, dtype=np.complex128),
            SimpleNamespace(hermiticity_residual=0.0),
        ),
    )
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *_args, out, **_kwargs: Path(out).write_text("plot"))

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump(_canonical_case_config(), sort_keys=False), encoding="utf-8")

    cli.main(["project", "-c", str(cfg_path)])

    projection_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "projection"
    assert np.load(projection_dir / "heff.npy").shape == (1, 2, 2)
    assert (projection_dir / "eigvals.txt").exists()
    assert (projection_dir / "basis.md").exists()
    assert (projection_dir / "basis.npz").exists()
    assert (projection_dir / "scatter.png").read_text(encoding="utf-8") == "plot"
    assert (projection_dir / "heff_list.npy").exists()
    assert (projection_dir / "heff_eig.npy").exists()


def test_project_without_low_state_selection_points_user_to_inspect() -> None:
    try:
        cli._normalize_nlow_state_list({})
    except ValueError as exc:
        assert "kp inspect" in str(exc)
        assert "project.nlow_state_list" in str(exc)
    else:
        raise AssertionError("missing project.nlow_state_list should fail")


def test_symmetry_canonical_writer_combines_representations_and_residuals(tmp_path: Path) -> None:
    out_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "symmetry"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "exactified_C3z.npy", np.eye(2, dtype=np.complex128))
    np.save(out_dir / "exactified_C2T.npy", -np.eye(2, dtype=np.complex128))
    summary = {
        "operations": [
            {
                "name": "C3z",
                "matrix_file": "exactified_C3z.npy",
                "status": "exactified",
                "residuals": {"unitarity": 1.0e-12},
            },
            {
                "name": "C2T",
                "matrix_file": "exactified_C2T.npy",
                "status": "exactified",
                "exactification_distance": 2.0e-12,
            },
        ]
    }

    projection_mod._write_canonical_symmetry_outputs(out_dir, summary)

    with np.load(out_dir / "representations.npz") as payload:
        np.testing.assert_allclose(payload["C3z"], np.eye(2))
        np.testing.assert_allclose(payload["C2T"], -np.eye(2))
    residuals = (out_dir / "residuals.csv").read_text(encoding="utf-8")
    assert "operation,status,unitarity,exactification_distance" in residuals
    assert "C3z,exactified,1e-12," in residuals


def test_model_canonical_exports_standalone_in_model_directory(monkeypatch, tmp_path: Path) -> None:
    from kp.model import export as export_mod
    from kp.model import pipeline

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump(_canonical_case_config(), sort_keys=False), encoding="utf-8")
    model_output = tmp_path / "kp" / "outputs" / "K1" / "q06" / "model"
    seen: dict[str, object] = {}

    model_cfg = SimpleNamespace(
        path=cfg_path,
        output_dir=model_output,
        n_orb=(1, 1),
        fit_indices=[0],
        bM_diagnostics={},
        symmetry_source_metadata={},
    )
    moire_cfg = SimpleNamespace(
        Q_set1=[0],
        Q_set2=[0],
        n_orb1=1,
        n_orb2=1,
        kpoints=[0],
        intra_harmonics_map={},
        inter_harmonics_map={},
    )

    monkeypatch.setattr(pipeline, "run_configured_model", lambda _config: {"configured_model": model_cfg, "moire_config": moire_cfg})

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        seen["export"] = (Path(model_output_dir), Path(output_dir), bool(force), bool(debug_files))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for name in ("README.md", "MODEL.md", "evaluate.py", "model.json", "model_data.npz"):
            (Path(output_dir) / name).write_text("x", encoding="utf-8")
        return Path(output_dir)

    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "-c", str(cfg_path)])

    assert seen["export"] == (model_output, model_output, True, False)
    assert not (model_output / "standalone").exists()
