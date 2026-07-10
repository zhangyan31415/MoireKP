from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import kp.cli as cli
from kp.config.case import normalize_case_config
from kp.model.pipeline import _rotation_deg_from_symmetry_manifest
from kp.model.symmetry import load_symmetry_source
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
    assert cfg["plot"]["out"] == "../outputs/K1/q06/inspect/scatter.pdf"
    assert cfg["plot"]["data_out"] == "../outputs/K1/q06/inspect/spectrum.txt"
    assert cfg["symm"]["output_dir"] == "../outputs/K1/q06/symmetry"
    assert cfg["output"]["dir"] == "../outputs/K1/q06/model"


def test_legacy_unified_defaults_are_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "kp" / "configs" / "legacy_case.yaml"
    raw = _canonical_case_config()
    raw["case"] = "legacy_case"

    with pytest.raises(ValueError, match="case.*profile.*q_shell.*output_root"):
        normalize_case_config(raw, config_path=cfg_path)


def test_kp_inspect_alias_dispatches_to_plot() -> None:
    calls = []
    original = cli.cmd_plot_from_config
    try:
        cli.cmd_plot_from_config = lambda config: calls.append(config)
        cli.main(["inspect", "-c", "K1_q06.yaml"])
    finally:
        cli.cmd_plot_from_config = original

    assert calls == ["K1_q06.yaml"]


def test_inspect_canonical_writes_user_facing_files(monkeypatch, tmp_path: Path, capsys) -> None:
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
    cfg["material"]["efermi"] = 0.0
    cfg["plot"] = {"mode": "gamma", "report_below": 1, "report_above": 1}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cli.main(["inspect", "-c", str(cfg_path)])
    stdout = capsys.readouterr().out

    inspect_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "inspect"
    assert (inspect_dir / "spectrum.txt").exists()
    assert (inspect_dir / "scatter.pdf").read_text(encoding="utf-8") == "plot"
    wavefunctions = np.load(inspect_dir / "wavefunctions.npz")
    np.testing.assert_allclose(wavefunctions["eigenvalues"], np.array([[-0.2, 0.1]]))
    np.testing.assert_allclose(wavefunctions["eigenvectors"], np.eye(2, dtype=np.complex128)[None, :, :])
    np.testing.assert_array_equal(wavefunctions["q_index_order"], np.array([0]))
    assert float(wavefunctions["efermi"]) == 0.0
    assert int(wavefunctions["ref_q_index"]) == 0
    assert not (inspect_dir / "candidates.md").exists()
    assert not (inspect_dir / "blocks.csv").exists()
    assert "[kp] Selected bands at ref_Q=0: below EF [0], above EF [1]" in stdout


def test_inspect_with_band_file_uses_combined_kpath_qblock_plot(monkeypatch, tmp_path: Path) -> None:
    q = np.zeros((1, 2), dtype=float)
    hamk = np.zeros((1, 4, 4), dtype=np.complex128)
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: hamk)
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q, q.copy()))

    def fake_get_H_block(*_args, **_kwargs):
        eigs = np.empty(2, dtype=object)
        eigs[0] = np.array([-0.2, 0.1])
        eigs[1] = np.array([-0.1, 0.2])
        vecs = np.empty(2, dtype=object)
        vecs[0] = np.eye(2, dtype=np.complex128)
        vecs[1] = np.eye(2, dtype=np.complex128)
        return eigs, vecs, np.empty(0, dtype=object), np.empty(0, dtype=object)

    def fake_combined_plot(band_rows, qblock_rows, efermi, *, out, q_sector_lengths=None, **kwargs):
        captured["band_rows"] = len(band_rows)
        captured["qblock_rows"] = len(qblock_rows)
        captured["efermi"] = efermi
        captured["q_sector_lengths"] = q_sector_lengths
        Path(out).write_text("combined", encoding="utf-8")

    monkeypatch.setattr(cli, "get_H_block", fake_get_H_block)
    monkeypatch.setattr(cli, "plot_inspect_band_and_qblock", fake_combined_plot)

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    (cfg_path.parent / "bands.txt").write_text("-0.3 -0.2 -0.1\n-0.28 -0.18 -0.08\n", encoding="utf-8")
    cfg = _canonical_case_config()
    cfg["material"]["band_file"] = "bands.txt"
    cfg["material"]["efermi"] = -0.15
    cfg["plot"] = {"mode": "K1", "target": "valence", "ref_q_index": 0}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cli.main(["inspect", "-c", str(cfg_path)])

    inspect_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "inspect"
    assert (inspect_dir / "scatter.pdf").read_text(encoding="utf-8") == "combined"
    assert captured == {
        "band_rows": 2,
        "qblock_rows": 2,
        "efermi": -0.15,
        "q_sector_lengths": [1, 1],
    }
    assert (inspect_dir / "spectrum.txt").read_text(encoding="utf-8").splitlines()[0] == "-0.2000000000 0.1000000000"


def test_inspect_qsort_uses_each_qset_center(monkeypatch, tmp_path: Path) -> None:
    q1 = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=float)
    q2 = np.array([[100.0, 0.0], [101.0, 0.0], [103.0, 0.0]], dtype=float)
    hamk = np.zeros((1, 12, 12), dtype=np.complex128)
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_load_hamk_with_energy_unit", lambda *_args, **_kwargs: hamk)
    monkeypatch.setattr(cli, "load_Q_sets", lambda *_args, **_kwargs: (q1, q2))
    monkeypatch.setattr(
        cli,
        "_orbital_layout_from_material",
        lambda *_args, **_kwargs: ([1, 1], 1, [[1], [1]]),
    )

    def fake_get_H_block(*_args, **_kwargs):
        eigs = np.empty(6, dtype=object)
        vecs = np.empty(6, dtype=object)
        for i in range(6):
            eigs[i] = np.array([float(i)])
            vecs[i] = np.eye(1, dtype=np.complex128)
        return eigs, vecs, np.empty(0, dtype=object), np.empty(0, dtype=object)

    def fake_combined_plot(*_args, q_index_order=None, **_kwargs):
        captured["q_index_order"] = list(q_index_order)
        Path(_kwargs["out"]).write_text("combined", encoding="utf-8")

    monkeypatch.setattr(cli, "get_H_block", fake_get_H_block)
    monkeypatch.setattr(cli, "plot_inspect_band_and_qblock", fake_combined_plot)

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    (cfg_path.parent / "bands.txt").write_text("-0.3 -0.2\n-0.28 -0.18\n", encoding="utf-8")
    cfg = _canonical_case_config()
    cfg["material"]["band_file"] = "bands.txt"
    cfg["material"]["efermi"] = -0.15
    cfg["plot"] = {"mode": "K1", "target": "valence", "sort_by_qnorm": True}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cli.main(["inspect", "-c", str(cfg_path)])

    assert captured["q_index_order"] == [1, 0, 2, 4, 3, 5]


def test_project_canonical_writes_only_release_outputs(monkeypatch, tmp_path: Path) -> None:
    q = np.zeros((1, 2), dtype=float)
    hamk = np.zeros((1, 4, 4), dtype=np.complex128)
    captured_plot: dict[str, object] = {}

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
    def fake_project_plot(_eigs, efermi, *, out, figsize=None, box_aspect=None, font_family=None, **_kwargs):
        captured_plot["efermi"] = efermi
        captured_plot["figsize"] = figsize
        captured_plot["box_aspect"] = box_aspect
        captured_plot["font_family"] = font_family
        captured_plot["top_bands"] = _kwargs.get("top_bands")
        captured_plot["plot_all_bands"] = _kwargs.get("plot_all_bands")
        Path(out).write_text("plot")

    monkeypatch.setattr(cli, "plot_eigs_scatter", fake_project_plot)

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg = _canonical_case_config()
    cfg["material"]["efermi"] = 0.0
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cli.main(["project", "-c", str(cfg_path)])

    projection_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "projection"
    assert np.load(projection_dir / "heff.npy").shape == (1, 2, 2)
    wavefunctions = np.load(projection_dir / "wavefunctions.npz")
    assert wavefunctions["wavefunctions"].shape == (1, 2, 2)
    np.testing.assert_array_equal(wavefunctions["k_indices"], np.array([0], dtype=int))
    assert (projection_dir / "eigvals.txt").exists()
    assert (projection_dir / "basis.md").exists()
    assert (projection_dir / "basis.npz").exists()
    assert (projection_dir / "scatter.pdf").read_text(encoding="utf-8") == "plot"
    assert set(path.name for path in projection_dir.iterdir()) == {
        "basis.md",
        "basis.npz",
        "eigvals.txt",
        "heff.npy",
        "scatter.pdf",
        "wavefunctions.npz",
    }
    assert not (projection_dir / "heff_list.npy").exists()
    assert not (projection_dir / "heff_eig.npy").exists()
    assert not (projection_dir / "heff_vec.npy").exists()
    assert not (projection_dir / "vectors.npy").exists()
    assert not (projection_dir / "basis_selection.json").exists()
    assert not (projection_dir / "basis_selection.md").exists()
    assert not (projection_dir / "auto_norb_fix_list.yaml").exists()
    assert captured_plot["efermi"] == 0.0
    assert tuple(captured_plot["figsize"]) == (3.0, 5.0)
    assert round(float(captured_plot["box_aspect"]), 6) == round(5 / 3, 6)
    assert captured_plot["font_family"]
    assert captured_plot["top_bands"] == 10
    assert captured_plot["plot_all_bands"] is True


def test_project_without_low_state_selection_points_user_to_inspect() -> None:
    try:
        cli._normalize_nlow_state_list({})
    except ValueError as exc:
        assert "kp inspect" in str(exc)
        assert "project.nlow_state_list" in str(exc)
    else:
        raise AssertionError("missing project.nlow_state_list should fail")


def test_model_canonical_defaults_read_project_heff(tmp_path: Path) -> None:
    from kp.model.pipeline import load_model_config

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg = _canonical_case_config()
    cfg["symm"] = {}
    cfg["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "../outputs/K1/q06/symmetry",
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    project_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "projection"
    project_dir.mkdir(parents=True)
    np.save(project_dir / "heff.npy", np.zeros((1, 2, 2), dtype=np.complex128))
    symmetry_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "symmetry"
    symmetry_dir.mkdir(parents=True)
    np.savez(
        symmetry_dir / "representations.npz",
        C3z=np.eye(2),
        C2T=np.eye(2),
        __metadata_json__=np.asarray(
            json.dumps({"frame": {"q_transform": {"rotation_deg": 0.0}}}, sort_keys=True)
        ),
    )

    model_cfg = load_model_config(cfg_path)

    assert model_cfg.heff_file == project_dir / "heff.npy"
    assert model_cfg.heff_eig_file is None


def test_model_rejects_compact_symmetry_without_frame_metadata(tmp_path: Path) -> None:
    from kp.model.pipeline import load_model_config

    cfg_path = tmp_path / "kp" / "configs" / "K1_q06.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg = _canonical_case_config()
    cfg["symm"] = {}
    cfg["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "../outputs/K1/q06/symmetry",
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    project_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "projection"
    project_dir.mkdir(parents=True)
    np.save(project_dir / "heff.npy", np.zeros((1, 2, 2), dtype=np.complex128))
    symmetry_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "symmetry"
    symmetry_dir.mkdir(parents=True)
    np.savez(symmetry_dir / "representations.npz", C3z=np.eye(2), C2T=np.eye(2))

    try:
        load_model_config(cfg_path)
    except ValueError as exc:
        assert "Rerun `kp symm`" in str(exc)
    else:
        raise AssertionError("compact symmetry without frame metadata should fail")


def test_symmetry_canonical_writer_combines_representations_and_residuals(tmp_path: Path) -> None:
    out_dir = tmp_path / "kp" / "outputs" / "K1" / "q06" / "symmetry"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "exactified_C3z.npy", np.eye(2, dtype=np.complex128))
    np.save(out_dir / "exactified_C2T.npy", -np.eye(2, dtype=np.complex128))
    np.save(out_dir / "C3z_low_raw.npy", 2.0 * np.eye(2, dtype=np.complex128))
    np.save(out_dir / "q_model_layer1.npy", np.zeros((1, 2), dtype=float))
    np.save(out_dir / "q_model_layer2.npy", np.zeros((1, 2), dtype=float))
    for stale in ("manifest.json", "summary.json", "basis_selection.json", "basis_selection.md", "auto_norb_fix_list.yaml"):
        (out_dir / stale).write_text("stale", encoding="utf-8")
    summary = {
        "valley": "K1",
        "spin": "up",
        "tolerance": 1.0e-8,
        "full_dim": 2,
        "low_dim": 2,
        "artifact_identity": {
            "identity_schema": "moirekp.artifact-identity.v1",
            "input_hash": "input-a",
            "config_hash": "config-a",
            "basis_hash": "basis-a",
            "package_version": "0.1.0",
            "schema_version": 1,
            "k_indices_hash": "k-indices-a",
            "heff_hash": "heff-a",
        },
        "frame": {
            "q_transform": {"rotation_deg": 210.0},
            "k_transform": {"rotation_deg": 210.0},
        },
        "operations": [
                {
                    "name": "C3z",
                    "operation": "C3z",
                    "matrix_file": "exactified_C3z.npy",
                    "antiunitary": False,
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                    "matrix_kind": "continuum_internal_rep_exact",
                    "matrix_source": "kp_symm_exactified_action",
                    "source_matrix_role": "raw_h_sewing_action",
                    "source_gauge": "raw_saved_TAPW",
                    "target_role": "continuum_internal_rep",
                    "gauge_correction": {"kind": "none"},
                    "antiunitary_convention": "none",
                    "spin_map": "from_kp_symm_output",
                    "valley_map": "identity",
                    "target_block_dims": [2],
                    "residuals": {"unitarity": 1.0e-12},
                    "pairs": [],
                    "status": "exactified",
                    "exactification_status": "exactified",
                    "source_matrix_projection_report": {
                        "report": {"status": "exactified"}
                    },
                },
                {
                    "name": "C2T",
                    "operation": "C2T",
                    "matrix_file": "exactified_C2T.npy",
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 60.0},
                    "q_map": {"type": "reflection", "axis_deg": 60.0},
                    "sector_map": "layer_exchange",
                    "matrix_kind": "continuum_internal_rep_exact",
                    "matrix_source": "kp_symm_exactified_action",
                    "source_matrix_role": "raw_h_sewing_action",
                    "source_gauge": "raw_saved_TAPW",
                    "target_role": "continuum_internal_rep",
                    "gauge_correction": {"kind": "none"},
                    "antiunitary_convention": "U_K",
                    "spin_map": "from_kp_symm_output",
                    "valley_map": "identity",
                    "target_block_dims": [2],
                    "pairs": [],
                    "status": "exactified",
                    "exactification_status": "exactified",
                    "exactification_distance": 2.0e-12,
                    "source_matrix_projection_report": {
                        "report": {"status": "exactified"}
                    },
                },
            ]
        }

    projection_mod._write_canonical_symmetry_outputs(out_dir, summary)

    with np.load(out_dir / "representations.npz") as payload:
        np.testing.assert_allclose(payload["C3z"], np.eye(2))
        np.testing.assert_allclose(payload["C2T"], -np.eye(2))
        metadata = json.loads(str(payload["__metadata_json__"].item()))
    c2t = next(op for op in metadata["operations"] if op["name"] == "C2T")
    assert c2t["k_map"] == {"type": "reflection", "axis_deg": 60.0}
    assert metadata["artifact_identity"]["basis_hash"] == "basis-a"
    assert all(op["basis_hash"] == "basis-a" for op in metadata["operations"])
    assert all(op["heff_hash"] == "heff-a" for op in metadata["operations"])
    rotation_deg = _rotation_deg_from_symmetry_manifest(
        {"symmetry_source": {"type": "kp_symm_output", "path": str(out_dir)}},
        base=tmp_path,
    )
    assert rotation_deg == 210.0
    loaded = load_symmetry_source({"type": "kp_symm_output", "path": str(out_dir)}, base=tmp_path, expected_dim=2)
    loaded_c2t = next(op for op in loaded.metadata["operations"] if op["name"] == "C2T")
    assert loaded_c2t["k_map"] == {"type": "reflection", "axis_deg": 60.0}
    assert loaded.generator.get_operator("C2T").shape == (2, 2)
    residuals = (out_dir / "residuals.csv").read_text(encoding="utf-8")
    assert "operation,status,unitarity,exactification_distance" in residuals
    assert "C3z,exactified,1e-12," in residuals
    assert (out_dir / "summary.md").exists()
    assert set(path.name for path in out_dir.iterdir()) == {
        "representations.npz",
        "residuals.csv",
        "summary.md",
    }


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
        for name in ("README.md", "MODEL.md", "evaluate.py", "model_data.npz", "eigvals.npy", "band_comparison.pdf", "band_comparison_all.pdf", "q_lattice_harmonics.pdf"):
            (Path(output_dir) / name).write_text("x", encoding="utf-8")
        for stale in ("active_terms.json", "active_terms.sha256", "fit_selection.json", "run_summary.json", "model_run.log"):
            (Path(output_dir) / stale).write_text("stale", encoding="utf-8")
        return Path(output_dir)

    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "-c", str(cfg_path)])

    assert seen["export"] == (model_output, model_output, True, False)
    assert not (model_output / "standalone").exists()
    assert set(path.name for path in model_output.iterdir()) == {
        "MODEL.md",
        "README.md",
        "band_comparison.pdf",
        "band_comparison_all.pdf",
        "eigvals.npy",
        "evaluate.py",
        "model_data.npz",
        "q_lattice_harmonics.pdf",
    }
