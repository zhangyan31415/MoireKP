from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.model.configured import (  # noqa: E402
    build_moire_config_from_file,
    compare_bands,
    evaluate_vector_expression,
    load_model_config,
    run_configured_model,
)


def test_cli_model_subcommand_invokes_configured_runner(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.configured as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    seen: dict[str, str] = {}

    def fake_run(path: str) -> dict:
        seen["path"] = path
        return {"comparison": {"rms_error": 0.0, "max_abs_error": 0.0}}

    monkeypatch.setattr(configured, "run_configured_model", fake_run)

    cli.main(["model", "--config", str(cfg_path)])

    assert seen["path"] == str(cfg_path)


def _write_fixture(tmp_path: Path) -> Path:
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    kpoints = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=float)
    heff = np.stack(
        [
            np.diag([0.0, 1.0, 2.0, 3.0]),
            np.diag([0.1, 1.1, 2.1, 3.1]),
            np.diag([0.2, 1.2, 2.2, 3.2]),
        ]
    ).astype(np.complex128)

    np.save(tmp_path / "q1.npy", q1)
    np.save(tmp_path / "q2.npy", q2)
    np.save(tmp_path / "kpoints.npy", kpoints)
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    np.save(out_dir / "heff_list.npy", heff)
    np.save(out_dir / "heff_eig.npy", np.linalg.eigvalsh(heff))

    source_cfg = {
        "material": {
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "plot": {
            "q_rotation_deg": 0,
        },
        "project": {
            "out_dir": "project",
        },
    }
    (tmp_path / "source.yaml").write_text(yaml.safe_dump(source_cfg), encoding="utf-8")

    model_cfg = {
        "source_config": "source.yaml",
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": [1, 1],
            "nlow_state": [1, 1],
            "bM": {"infer_from_q": True},
            "harmonics": {
                "intra": {1: "zero", 2: "-bM1", 3: "bM1 + bM2"},
                "inter": {1: "zero", 2: "2*bM1"},
            },
            "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
            "symmetry_map": {
                "Kinect": [{"name": "TR"}],
                "intra": [{"name": "C3z"}],
                "inter": [{"name": "TR"}],
            },
        },
        "fit": {
            "indices": [0, 2],
            "coeff_tol": 1.0e-8,
        },
        "bands": {
            "indices": [0, 1, 2],
            "compare_to_heff": True,
        },
        "output": {
            "dir": "model_out",
        },
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    return path


def test_load_model_config_resolves_paths_relative_to_yaml(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)

    cfg = load_model_config(cfg_path)

    assert cfg.path == cfg_path
    assert cfg.source_config == tmp_path / "source.yaml"
    assert cfg.kpoints_file == tmp_path / "kpoints.npy"
    assert cfg.heff_file == tmp_path / "project" / "heff_list.npy"
    assert cfg.output_dir == tmp_path / "model_out"
    assert cfg.fit_indices == [0, 2]
    assert cfg.band_indices == [0, 1, 2]


def test_evaluate_vector_expression_supports_bM_symbols() -> None:
    variables = {
        "bM1": np.array([1.0, 0.0]),
        "bM2": np.array([0.0, 2.0]),
        "zero": np.zeros(2),
    }

    np.testing.assert_allclose(evaluate_vector_expression("bM1 + 0.5*bM2", variables), [1.0, 1.0])
    np.testing.assert_allclose(evaluate_vector_expression("-bM1", variables), [-1.0, 0.0])
    np.testing.assert_allclose(evaluate_vector_expression([3.0, 4.0], variables), [3.0, 4.0])


def test_build_moire_config_loads_fit_block_and_band_kpoints(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.heff_file == tmp_path / "project" / "heff_list.npy"
    assert moire_cfg.Q_set1.shape == (2, 2)
    assert moire_cfg.Q_set2.shape == (2, 2)
    assert moire_cfg.n_orb1 == 1
    assert moire_cfg.n_orb2 == 1
    assert moire_cfg.heff.shape == (8, 8)
    np.testing.assert_allclose(moire_cfg.kpoints_fit, [[0.0, 0.0], [0.2, 0.0]])
    np.testing.assert_allclose(moire_cfg.kpoints, [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    np.testing.assert_allclose(moire_cfg.intra_harmonics_map[3], moire_cfg.bM1 + moire_cfg.bM2)


def test_run_configured_model_saves_outputs_without_legacy_diagnostics_json(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_run_end_to_end(moire_config):
        assert moire_config.output_dir is None
        return {"eigvals": expected_eigvals, "diagnostics": {("tuple", "key"): 1}}

    monkeypatch.setattr("kp.model.configured.run_end_to_end", fake_run_end_to_end)

    results = run_configured_model(cfg_path)

    assert results["comparison"]["max_abs_error"] == 0.0
    assert results["moire_config"].output_dir == tmp_path / "model_out"
    assert (tmp_path / "model_out" / "eigvals.npy").exists()
    assert (tmp_path / "model_out" / "comparison.json").exists()


def test_compare_bands_reports_rms_and_max_error() -> None:
    model = np.array([[0.0, 1.1, 2.0], [0.0, 2.0, 4.0]])
    heff = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 4.0]])

    metrics = compare_bands(model, heff, band_slice=[0, 2])

    assert metrics["num_kpoints"] == 2
    assert metrics["num_bands"] == 2
    assert metrics["rms_error"] == np.sqrt((0.0**2 + 0.1**2 + 0.0**2 + 0.5**2) / 4)
    assert metrics["max_abs_error"] == 0.5
