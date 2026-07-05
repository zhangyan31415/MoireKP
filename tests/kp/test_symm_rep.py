from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml


def _write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "configs" / "case.yaml"
    cfg_path.parent.mkdir(parents=True)
    payload = {
        "case": {"profile": "Gamma", "q_shell": "q01", "output_root": "../outputs"},
        "material": {"efermi": 0.0},
        "kpath": {
            "labels": ["G", "M"],
            "points_per_segment": 1,
            "coordinates": {"G": [0.0, 0.0], "M": [0.5, 0.0]},
        },
        "symm_rep": {
            "output_dir": "symm_rep",
            "valence_count": 2,
            "conduction_count": 1,
            "degeneracy_tol": 1.0e-6,
            "points": {"Gamma": [0.0, 0.0]},
        },
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return cfg_path


def _write_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "outputs" / "Gamma" / "q01"
    projection_dir = root / "projection"
    symmetry_dir = root / "symmetry"
    projection_dir.mkdir(parents=True)
    symmetry_dir.mkdir(parents=True)

    heff = np.diag([-1.0, -1.0, 1.0, 2.0]).astype(np.complex128)[None, :, :]
    np.save(projection_dir / "heff.npy", heff)
    c2 = np.diag([1.0, -1.0, 1.0, -1.0]).astype(np.complex128)
    tr = np.eye(4, dtype=np.complex128)
    np.savez_compressed(symmetry_dir / "representations.npz", C2=c2, TR=tr)
    return root


def test_kp_symm_rep_projects_exactified_matrices_into_heff_blocks(tmp_path):
    from kp.symm_rep import run_configured_symm_rep

    cfg_path = _write_config(tmp_path)
    root = _write_inputs(tmp_path)

    output_dir = run_configured_symm_rep(cfg_path)

    assert output_dir == root / "symm_rep"
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "high_symmetry_wavefunctions.npz").is_file()
    assert (output_dir / "band_representations.npz").is_file()

    rows = list(csv.DictReader((output_dir / "characters.csv").open(newline="", encoding="utf-8")))
    c2_valence = [
        row
        for row in rows
        if row["point"] == "Gamma"
        and row["sector"] == "valence"
        and row["band_indices"] == "0 1"
        and row["operation"] == "C2"
    ]
    assert len(c2_valence) == 1
    assert float(c2_valence[0]["trace_real"]) == 0.0
    assert c2_valence[0]["energies"] == "-1.000000000000, -1.000000000000"
    assert "[1.0000, 0.0000]; [0.0000, -1.0000]" in c2_valence[0]["d_block"]

    with np.load(output_dir / "band_representations.npz", allow_pickle=False) as payload:
        assert np.allclose(payload["Gamma__valence__block0__C2"], np.diag([1.0, -1.0]))

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "# KP Symmetry Representation Summary" in summary
    assert "heff row: `0`" in summary
    assert "### Symmetry Representations" in summary


def test_kp_symm_rep_phase_label_uses_spinful_c3_omega():
    from kp.symm_rep import _phase_label

    assert _phase_label(np.exp(1.0j * np.pi / 3.0)) == "ω"
    assert _phase_label(np.exp(2.0j * np.pi / 3.0)) == "ω²"
    assert _phase_label(np.exp(-2.0j * np.pi / 3.0)) == "ω⁴"
    assert _phase_label(np.exp(-1.0j * np.pi / 3.0)) == "ω⁵"
    assert _phase_label(-1.0 + 0.0j) == "-1"


def test_kp_symm_rep_uses_polar_unitary_for_band_block(tmp_path):
    from kp.symm_rep import run_configured_symm_rep

    cfg_path = _write_config(tmp_path)
    root = _write_inputs(tmp_path)
    symmetry_dir = root / "symmetry"
    raw = np.diag([0.8, 1.2, 1.0, 1.0]).astype(np.complex128)
    np.savez_compressed(symmetry_dir / "representations.npz", C2=raw)

    output_dir = run_configured_symm_rep(cfg_path)

    rows = list(csv.DictReader((output_dir / "characters.csv").open(newline="", encoding="utf-8")))
    row = next(
        item
        for item in rows
        if item["point"] == "Gamma"
        and item["sector"] == "valence"
        and item["band_indices"] == "0 1"
        and item["operation"] == "C2"
    )
    assert float(row["trace_real"]) == 2.0
    assert float(row["unitarity_residual"]) < 1.0e-12
    assert float(row["raw_unitarity_residual"]) > 0.0
    assert float(row["polar_distance"]) > 0.0
    assert row["d_block"] == "[[1.0000, 0.0000]; [0.0000, 1.0000]]"


def test_kp_symm_rep_spin_diagonalizes_degenerate_block_from_spin_operator(tmp_path):
    from kp.symm_rep import run_configured_symm_rep

    cfg_path = _write_config(tmp_path)
    root = _write_inputs(tmp_path)
    spin_operator = np.diag([-1.0, 1.0, 1.0, -1.0]).astype(np.complex128)[None, :, :]
    np.save(root / "projection" / "spin_operator.npy", spin_operator)

    output_dir = run_configured_symm_rep(cfg_path)

    bands = list(csv.DictReader((output_dir / "bands.csv").open(newline="", encoding="utf-8")))
    valence = [row for row in bands if row["point"] == "Gamma" and row["sector"] == "valence"]
    assert [(row["spin_label"], float(row["spin_up"]), float(row["spin_down"])) for row in valence[:2]] == [
        ("up", 1.0, 0.0),
        ("down", 0.0, 1.0),
    ]


def test_kp_symm_rep_keeps_only_little_group_operations(tmp_path):
    from kp.symm_rep import run_configured_symm_rep

    cfg_path = _write_config(tmp_path)
    qset1_path = tmp_path / "qset1.npy"
    qset2_path = tmp_path / "qset2.npy"
    q_source = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
    np.save(qset1_path, q_source)
    np.save(qset2_path, q_source)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    payload["material"]["qset1_file"] = str(qset1_path)
    payload["material"]["qset2_file"] = str(qset2_path)
    payload["symm_rep"]["points"] = {"K": {"coords": [1.0 / 3.0, 1.0 / 3.0], "index": 0}}
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    root = _write_inputs(tmp_path)
    symmetry_dir = root / "symmetry"
    metadata = {
        "frame": {"k_transform": {"rotation_deg": 0.0}},
        "kp_symm_exactification": {
            "bM1": [1.0, 0.0],
            "bM2": [0.5, 0.8660254037844386],
            "n_orb": [1, 1],
            "sectors": [
                {"name": "L1", "qset": "qset1", "n_orb": 1, "q_offset": [0.5, 0.0]},
                {"name": "L2", "qset": "qset2", "n_orb": 1, "q_offset": [0.5, 0.0]},
            ],
        },
        "operations": [
            {
                "name": "C3z",
                "matrix_array_key": "C3z",
                "antiunitary": False,
                "model_action": {"k_map": {"type": "rotation", "angle_deg": 120.0}},
            },
            {
                "name": "C2",
                "matrix_array_key": "C2",
                "antiunitary": False,
                "model_action": {"k_map": {"type": "reflection", "axis_deg": 0.0}},
            },
        ],
    }
    np.savez_compressed(
        symmetry_dir / "representations.npz",
        C3z=np.eye(4, dtype=np.complex128),
        C2=np.eye(4, dtype=np.complex128),
        __metadata_json__=np.asarray(json.dumps(metadata)),
    )

    output_dir = run_configured_symm_rep(cfg_path)

    rows = list(csv.DictReader((output_dir / "characters.csv").open(newline="", encoding="utf-8")))
    assert {row["operation"] for row in rows} == {"C3z"}
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "little-group operations: C3z" in summary


def test_kp_symm_rep_uses_reciprocal_shift_sewing_for_little_group(tmp_path):
    from kp.symm_rep import run_configured_symm_rep

    cfg_path = _write_config(tmp_path)
    qset1_path = tmp_path / "qset1.npy"
    qset2_path = tmp_path / "qset2.npy"
    q_source = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
    np.save(qset1_path, q_source)
    np.save(qset2_path, q_source)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    payload["material"]["qset1_file"] = str(qset1_path)
    payload["material"]["qset2_file"] = str(qset2_path)
    payload["symm_rep"]["points"] = {"K": {"coords": [1.0 / 3.0, 1.0 / 3.0], "index": 0}}
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    root = _write_inputs(tmp_path)
    heff = np.diag([-1.0, -1.0, 1.0, 2.0]).astype(np.complex128)[None, :, :]
    np.save(root / "projection" / "heff.npy", heff)
    symmetry_dir = root / "symmetry"
    metadata = {
        "frame": {"k_transform": {"rotation_deg": 0.0}},
        "kp_symm_exactification": {
            "bM1": [1.0, 0.0],
            "bM2": [0.5, 0.8660254037844386],
            "n_orb": [1, 1],
            "sectors": [
                {"name": "L1", "qset": "qset1", "n_orb": 1, "q_offset": [0.5, 0.0]},
                {"name": "L2", "qset": "qset2", "n_orb": 1, "q_offset": [0.5, 0.0]},
            ],
        },
        "operations": [
            {
                "name": "C3z",
                "matrix_array_key": "C3z",
                "antiunitary": False,
                "matrix_kind": "continuum_internal_rep_exact",
                "matrix_source": "kp_symm_exactified_action",
                "model_action": {"k_map": {"type": "rotation", "angle_deg": 120.0}},
                "pairs": [{"source_k_index": 99, "target_k_index": 99}],
            },
        ],
    }
    np.savez_compressed(
        symmetry_dir / "representations.npz",
        C3z=np.eye(4, dtype=np.complex128),
        __metadata_json__=np.asarray(json.dumps(metadata)),
    )

    output_dir = run_configured_symm_rep(cfg_path)

    rows = list(csv.DictReader((output_dir / "characters.csv").open(newline="", encoding="utf-8")))
    assert {row["operation"] for row in rows} == {"C3z"}
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "little-group operations: C3z" in summary
    with np.load(output_dir / "band_representations.npz", allow_pickle=False) as payload:
        projected = payload["K__valence__block0__C3z"]
        raw_projected = payload["K__valence__block0__C3z__raw_projected"]
    assert np.allclose(projected.conj().T @ projected, np.eye(2))
    assert np.allclose(raw_projected, np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))


def test_kp_symm_rep_cli_dispatches(monkeypatch, tmp_path):
    from kp import cli
    from kp import symm_rep

    calls = []
    monkeypatch.setattr(symm_rep, "main", lambda argv=None: calls.append(argv))

    cli.main(["symm-rep", "-c", str(tmp_path / "case.yaml")])

    assert calls == [["-c", str(tmp_path / "case.yaml")]]
