from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.model.configured import (  # noqa: E402
    build_moire_config_from_file,
    compare_bands,
    compare_bands_for_plot,
    evaluate_vector_expression,
    load_model_config,
    matrix_residual,
    run_configured_model,
)
from kp.src import moire_refactored as moire_module  # noqa: E402
from kp.src.moire_refactored import (  # noqa: E402
    ContinuumModel,
    ContinuumModelBuilder,
    ContinuumTerm,
    ContinuumTermKey,
    MoireConfig,
    SymmetryGenerator,
    build_model,
    compute_bands,
)


def _source_meta(*, antiunitary: bool = False, representation: bool = False) -> dict[str, object]:
    return {
        "source_matrix_role": "bare_D0_internal_rep" if representation else "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "continuum_internal_rep",
        "gauge_correction": {"kind": "none"},
        "antiunitary_convention": "U_K" if antiunitary else "none",
                "spin_map": "from_kp_symm_output",
        "valley_map": "identity",
    }


def _triangular_q_shell() -> np.ndarray:
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    return np.array(
        [
            [0.0, 0.0],
            b1,
            -b1,
            b2,
            -b2,
            b1 - b2,
            -b1 + b2,
        ],
        dtype=float,
    )


def test_compute_bands_uses_eigvals_only_without_saved_eigenvectors(monkeypatch):
    def fail_eigh(*_args, **_kwargs):
        raise AssertionError("compute_bands should not compute eigenvectors when save_eigvecs=false")

    monkeypatch.setattr(moire_module.scipy.linalg, "eigh", fail_eigh)

    cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
        save_eigvecs=False,
    )
    key = ContinuumTermKey(Mz=0, Mz_star=0, layer_from=1, layer_to=1, orbital_from=1, orbital_to=1, p=(0.0, 0.0))
    term = ContinuumTerm(
        key=key,
        Y_basis=lambda _k: np.diag([1.0, 2.0]).astype(complex),
        r_value_real=1.0,
        r_value_imag=0.0,
        active=True,
        tag="Kinect",
    )
    model = ContinuumModel()
    model.terms[key] = term

    eigvals = compute_bands(cfg, model, cfg.kpoints)

    assert eigvals.shape == (1, 2)
    np.testing.assert_allclose(eigvals[0], [1.0, 2.0])


def test_monomial_extraction_tolerates_roundoff_leakage():
    matrix = np.eye(4, dtype=complex)
    matrix[0, 1] = 3.0e-9

    mono = ContinuumModelBuilder._extract_monomial_matrix(matrix)

    assert mono is not None
    perm, vals = mono
    np.testing.assert_array_equal(perm, np.arange(4))
    np.testing.assert_allclose(vals, np.ones(4))


def test_monomial_extraction_rejects_dense_leakage():
    matrix = np.eye(4, dtype=complex)
    matrix[0, 1] = 3.0e-5

    assert ContinuumModelBuilder._extract_monomial_matrix(matrix) is None


def test_clear_symmetry_caches_clears_orbit_and_kz_caches():
    moire_module.clear_symmetry_caches()
    ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE["orbit"] = object()
    ContinuumModelBuilder._KZ_POW_CACHE["kz"] = np.ones((1, 1), dtype=complex)

    moire_module.clear_symmetry_caches()

    assert not ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE
    assert not ContinuumModelBuilder._KZ_POW_CACHE


def _write_auto_fixture(tmp_path: Path) -> Path:
    qset = _triangular_q_shell()
    kpoints = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float)
    dim = 2 * len(qset)
    heff = np.stack([np.diag(np.arange(dim, dtype=float)), np.diag(np.arange(dim, dtype=float) + 0.1)]).astype(np.complex128)

    np.save(tmp_path / "q1.npy", -qset)
    np.save(tmp_path / "q2.npy", -qset)
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
        "plot": {},
        "project": {
            "out_dir": "project",
        },
    }
    (tmp_path / "source.yaml").write_text(yaml.safe_dump(source_cfg), encoding="utf-8")

    model_cfg = {
        "source_config": "source.yaml",
        "coordinate_frame": {"rotation_deg": 0},
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": "Gamma",
            "mode": "single_valley",
            "active_valleys": ["Gamma"],
            "spin_convention": "spinless_effective",
            "allowed_internal_symmetries": [],
            "external_sewing_symmetries": [],
        },
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": [1, 1],
            "nlow_state": [1, 1],
            "bM": {"source": "q_distance"},
            "harmonics": {
                "intra": 4,
                "inter": {"count": 4},
            },
            "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
            "symmetry_map": {
                "Kinect": [],
                "intra": [],
                "inter": [],
            },
        },
        "fit": {
            "indices": [0],
            "coeff_tol": 1.0e-8,
        },
        "bands": {
            "indices": [0, 1],
            "compare_to_heff": True,
        },
        "output": {
            "dir": "model_out",
        },
    }
    path = tmp_path / "model_auto.yaml"
    path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    return path


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


def test_cli_model_subcommand_prints_band_plot_path(monkeypatch, tmp_path: Path, capsys) -> None:
    import kp.cli as cli
    import kp.model.configured as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    plot_path = tmp_path / "model_out" / "band_comparison.png"

    def fake_run(path: str) -> dict:
        return {
            "band_plot": str(plot_path.resolve()),
            "comparison": {"rms_error": 0.0, "max_abs_error": 0.0},
        }

    monkeypatch.setattr(configured, "run_configured_model", fake_run)

    cli.main(["model", "--config", str(cfg_path)])

    assert f"[kp model]   band plot: {plot_path.resolve()}" in capsys.readouterr().out


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
        "plot": {},
        "project": {
            "out_dir": "project",
        },
    }
    (tmp_path / "source.yaml").write_text(yaml.safe_dump(source_cfg), encoding="utf-8")

    model_cfg = {
        "source_config": "source.yaml",
        "coordinate_frame": {"rotation_deg": 0},
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": "Gamma",
            "mode": "single_valley",
            "active_valleys": ["Gamma"],
            "spin_convention": "spinless_effective",
            "allowed_internal_symmetries": [],
            "external_sewing_symmetries": [],
        },
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": [1, 1],
            "nlow_state": [1, 1],
            "bM": {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]},
            "harmonics": {
                "intra": {1: "zero", 2: "-bM1", 3: "bM1 + bM2"},
                "inter": {1: "zero", 2: "2*bM1"},
            },
            "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
            "symmetry_map": {
                "Kinect": [],
                "intra": [],
                "inter": [],
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


def test_load_model_config_infers_n_orb_and_safe_defaults(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["source_config"] = "source.yaml"
    raw["valley_model"]["allowed_internal_symmetries"] = []
    raw["model"] = {
        "bM": {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]},
        "harmonics": {
            "intra": {1: "zero"},
            "inter": {1: "zero"},
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)
    moire_cfg, _ = build_moire_config_from_file(cfg_path)
    model = build_model(moire_cfg)

    assert cfg.n_orb == (1, 1)
    assert cfg.nlow_state == [1, 1]
    assert cfg.max_order == {"Kinect": 2, "intra": 0, "inter": 0}
    assert len(model.terms) < 30


def test_load_model_config_rejects_sector_n_orb_mismatch(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["n_orb"] = [2, 2]
    raw["sectors"] = [
        {"name": "bottom", "qset": "qset1", "n_orb": 1},
        {"name": "top", "qset": "qset2", "n_orb": 2},
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"sectors\.bottom\.n_orb=1.*model\.n_orb.*2"):
        load_model_config(cfg_path)


def test_load_model_config_rejects_user_exactification_knobs(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "operations": ["C3z"],
        "exactification": {"support_mode": "monomial"},
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source matrix projection is internal"):
        load_model_config(cfg_path)


def test_build_moire_config_reads_model_q_sets_from_symmetry_artifact(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["coordinate_frame"] = {"rotation_deg": 90.0}
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "representation"}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    q1_model = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]], dtype=float)
    q2_model = np.array([[20.0, 0.0], [21.0, 0.0], [22.0, 0.0]], dtype=float)
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    np.save(symm_dir / "q_model_layer1.npy", q1_model)
    np.save(symm_dir / "q_model_layer2.npy", q2_model)
    (symm_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frame": {"q_transform": {"rotation_deg": 90.0}},
                "q_model": {"files": {"layer1": "q_model_layer1.npy", "layer2": "q_model_layer2.npy"}},
                "operations": [],
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, _model_cfg = build_moire_config_from_file(cfg_path)

    np.testing.assert_allclose(moire_cfg.Q_set1, q1_model)
    np.testing.assert_allclose(moire_cfg.Q_set2, q2_model)


def test_model_max_order_overrides_safe_defaults(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["max_order"] = {"Kinect": 4, "intra": 1, "inter": 0}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.max_order == {"Kinect": 4, "intra": 1, "inter": 0}


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


def test_build_moire_config_auto_harmonics_selects_q_shell_stars(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    b = float(np.linalg.norm(moire_cfg.bM1))
    expected_norms = np.array([0.0, b, np.sqrt(3.0) * b, 2.0 * b])
    intra_norms = np.array([np.linalg.norm(moire_cfg.intra_harmonics_map[i]) for i in range(1, 5)])
    inter_norms = np.array([np.linalg.norm(moire_cfg.inter_harmonics_map[i]) for i in range(1, 5)])
    np.testing.assert_allclose(intra_norms, expected_norms, atol=1.0e-8)
    np.testing.assert_allclose(inter_norms, expected_norms, atol=1.0e-8)
    assert model_cfg.harmonics_diagnostics["auto_used"] is True
    assert model_cfg.harmonics_diagnostics["intra"]["selected"][1]["pair_count"] > 0


def test_build_moire_config_infers_bM_from_q_distances_not_q_norm(tmp_path: Path) -> None:
    radius = 0.5
    angles = np.deg2rad([0.0, 120.0, 240.0])
    qset = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    kpoints = np.array([[0.0, 0.0]], dtype=float)
    dim = 2 * len(qset)
    heff = np.diag(np.arange(dim, dtype=float))[None, :, :].astype(np.complex128)

    np.save(tmp_path / "q1.npy", -qset)
    np.save(tmp_path / "q2.npy", -qset)
    np.save(tmp_path / "kpoints.npy", kpoints)
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    np.save(out_dir / "heff_list.npy", heff)
    np.save(out_dir / "heff_eig.npy", np.linalg.eigvalsh(heff))
    (tmp_path / "source.yaml").write_text(
        yaml.safe_dump(
            {
                "material": {"qset1_file": "q1.npy", "qset2_file": "q2.npy"},
                "plot": {},
                "project": {"out_dir": "project"},
            }
        ),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "source_config": "source.yaml",
                "coordinate_frame": {"rotation_deg": 0},
                "valley_model": {
                    "lattice": "hexagonal",
                    "system": "bilayer",
                    "valley_type": "Gamma",
                    "mode": "single_valley",
                    "active_valleys": ["Gamma"],
                    "spin_convention": "spinless_effective",
                    "allowed_internal_symmetries": [],
                    "external_sewing_symmetries": [],
                },
                "kpoints_file": "kpoints.npy",
                "model": {
                    "n_orb": [1, 1],
                    "nlow_state": [1, 1],
                    "bM": {"source": "q_distance"},
                    "harmonics": {"intra": 1, "inter": 1},
                    "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
                    "symmetry_map": {},
                },
                "fit": {"indices": [0]},
                "bands": {"indices": [0], "compare_to_heff": False},
                "output": {"dir": "model_out"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    moire_cfg, _ = build_moire_config_from_file(cfg_path)

    assert np.linalg.norm(moire_cfg.bM1) > radius
    np.testing.assert_allclose(np.linalg.norm(moire_cfg.bM1), np.sqrt(3.0) * radius, atol=1.0e-12)


def test_build_moire_config_infers_bM_from_tmat_with_shared_rotation(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {"source": "tmat"}
    raw["kpath"] = {
        "file": "KPATH.in",
        "tmat": [
            [60.9508659523, 0.0, 0.0],
            [-30.4754329751, 52.7849982976, 0.0],
            [0.0, 0.0, 50.0],
        ],
    }
    raw["coordinate_frame"] = {"rotation_deg": 210}
    (tmp_path / "KPATH.in").write_text(
        "\n".join(
            [
                "K-Path Generated by VASPKIT.",
                "1",
                "Line-Mode",
                "Reciprocal",
                "0 0 0 Gamma",
                "0.5 0 0 M",
            ]
        ),
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.rotation_deg == 210
    np.testing.assert_allclose(moire_cfg.bM1, [0.11903354191339506, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(moire_cfg.bM2, [0.05951677095669753, 0.10308607119512161], atol=1.0e-10)


def test_load_model_config_rejects_duplicate_model_rotation_settings(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["q_rotation_deg"] = 0
    raw["kpath"] = {"phase_deg": 0}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Use only coordinate_frame.rotation_deg"):
        load_model_config(cfg_path)


def test_T_toy_generator_rejects_spinless_nlow_state() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    gen = SymmetryGenerator(q, q, [1, 1])

    with pytest.raises(ValueError, match="TR toy generator requires explicit spin/Kramers pair basis"):
        gen.get_time_reversal_matrix()


def test_noncanonical_internal_operation_rejected(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "Gamma_four_orbital"}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "bad_op"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported internal symmetry operation names"):
        load_model_config(cfg_path)


def test_C2_toy_generator_requires_template() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    gen = SymmetryGenerator(q, q, [2, 2])

    with pytest.raises(ValueError, match="C2 toy generator requires"):
        gen.get_C2_operator()


def test_C3z_toy_generator_requires_template() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    gen = SymmetryGenerator(q, q, [2, 2])

    with pytest.raises(ValueError, match="C3z toy generator requires"):
        gen.get_C3z_operator(1)


def test_K_single_valley_rejects_T_C2(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "K",
        "mode": "single_valley",
        "active_valleys": ["K1"],
        "spin_convention": "spinful",
        "allowed_internal_symmetries": ["C3z", "C2T"],
        "external_sewing_symmetries": ["TR", "C2"],
    }
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "identity_c3"}
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.5, 0.8660254037844386]}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "TR"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="K single_valley cannot use physical TR/C2"):
        load_model_config(cfg_path)

    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C3z"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_model_config(cfg_path)
    assert cfg.valley_model["valley_type"] == "K"


def test_K_single_valley_keeps_source_operation_separate_from_canonical_name(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "K",
        "mode": "single_valley",
        "active_valleys": ["K1"],
        "spin_convention": "spin_up_only",
        "allowed_internal_symmetries": ["C3z", "C2T"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "operations": [
            {
                **_source_meta(),
                "name": "C3z",
                "operation": "C3",
                "matrix_file": "C3_low_raw.npy",
                "antiunitary": False,
                "k_map": {"type": "rotation", "angle_deg": 120.0},
                "q_map": {"type": "rotation", "angle_deg": 120.0},
                "sector_map": "identity",
            },
            {
                **_source_meta(antiunitary=True),
                "name": "C2T",
                "operation": "C2T",
                "matrix_file": "C2T_low_raw.npy",
                "antiunitary": True,
                "k_map": {"type": "reflection", "axis_deg": 180.0},
                "q_map": {"type": "reflection", "axis_deg": 180.0},
                "sector_map": "identity",
            },
        ],
    }
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C3z"}, {"name": "C2T"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.valley_model["valley_type"] == "K"


def test_K_single_valley_C2T_requires_kp_symm_output(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "K",
        "mode": "single_valley",
        "active_valleys": ["K1"],
        "spin_convention": "spin_up_only",
        "allowed_internal_symmetries": ["C3z", "C2T"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "Gamma_four_orbital"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2T"}],
        "intra": [{"name": "C3z"}, {"name": "C2T"}],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="C2T toy generator is not supported"):
        load_model_config(cfg_path)


def test_generic_monomial_constraints_reproduce_legacy_kinetic_orders() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 6, "intra": 0, "inter": 0},
        symmetry_map={"Kinect": []},
        term_templates=[
            {
                "name": "generic_kinetic_layer1",
                "source": "diagonal_kp",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 6,
                "monomial_constraints": {
                    "exclude_m_sum_zero": True,
                    "difference_mod": 3,
                    "difference_residue": 0,
                    "require_mz_ge_mz_star": True,
                },
            }
        ],
    )

    model = build_model(cfg)
    orders = sorted((term.key.Mz, term.key.Mz_star) for term in model.terms.values())

    assert orders == [(1, 1), (2, 2), (3, 0), (3, 3), (4, 1), (6, 0)]


def test_M_spinless_uses_effective_T_name(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "M",
        "mode": "single_valley",
        "active_valleys": ["M1"],
        "spin_convention": "spinless_effective",
        "allowed_internal_symmetries": ["TR_eff", "C2_eff", "C2TR_eff"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "M_spinless_layer_exchange"}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "TR"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="M single_valley spinless_effective must use TR_eff"):
        load_model_config(cfg_path)

    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "TR_eff"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_model_config(cfg_path)
    assert cfg.valley_model["spin_convention"] == "spinless_effective"


def test_M_spinless_requires_effective_C2_internal_name(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "M",
        "mode": "single_valley",
        "active_valleys": ["M1"],
        "spin_convention": "spinless_effective",
        "allowed_internal_symmetries": ["TR_eff", "C2_eff"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "M_spinless_layer_exchange"}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2"}, {"name": "TR_eff"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must be effective names"):
        load_model_config(cfg_path)

    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2_eff"}, {"name": "TR_eff"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_model_config(cfg_path)

    assert cfg.valley_model["allowed_internal_symmetries"] == ["TR_eff", "C2_eff"]
    assert cfg.symmetry_map["Kinect"][0]["name"] == "C2_eff"


def test_Gamma_user_facing_C2_stays_standard_family(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["allowed_internal_symmetries"] = ["TR", "C2"]
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "Gamma_four_orbital"}
    raw["model"]["n_orb"] = [2, 2]
    raw["model"]["nlow_state"] = [2, 2]
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "TR"}, {"name": "C2"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.valley_model["allowed_internal_symmetries"] == ["TR", "C2"]
    assert cfg.symmetry_map["Kinect"][1]["name"] == "C2"


def test_monomial_filter_is_forbidden_in_release_schema(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["term_templates"] = [
        {
            "name": "bad_legacy_filter",
            "source": "diagonal_kp",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": [[1, 1]],
            "max_order": 4,
            "monomial_filter": "old_kinetic_filter",
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="not supported by the release model schema"):
        load_model_config(cfg_path)


def test_release_schema_rejects_unknown_generation_mode(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["term_templates"] = [
        {
            "name": "kinetic_layer1",
            "source": "diagonal_kp",
            "generation_mode": "unsupported_mode",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": [[1, 1]],
            "max_order": 4,
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="generation_mode must be one of"):
        load_model_config(cfg_path)


def test_release_schema_rejects_legacy_term_names(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["term_templates"] = [
        {
            "name": "legacy_intra_layer1",
            "source": "moire_potential",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": "all",
            "harmonics": {"kind": "intra", "indices": [1]},
            "max_order": 0,
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy-style name"):
        load_model_config(cfg_path)


def test_M_spinless_runtime_supports_effective_symmetry_names() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    sym = SymmetryGenerator(q, q, [1, 1], basis_template="M_spinless_layer_exchange")

    t_eff = sym.get_operator("TR_eff", None)
    c2_eff = sym.get_operator("C2_eff", None)
    c2t_eff = sym.get_operator("C2TR_eff", None)

    assert t_eff.shape == (2, 2)
    assert c2_eff.shape == (2, 2)
    assert c2t_eff.shape == (2, 2)
    np.testing.assert_allclose(t_eff @ t_eff.conj(), np.eye(2), atol=1.0e-12)


def test_M_spinless_effective_symmetry_names_have_k_actions() -> None:
    k = np.array([0.25, -0.5], dtype=float)

    np.testing.assert_allclose(
        ContinuumModelBuilder._apply_k_map_to_vector(k, {"name": "TR_eff", "k_map": {"type": "negation"}}),
        np.array([-0.25, 0.5]),
    )
    np.testing.assert_allclose(
        ContinuumModelBuilder._apply_k_map_to_vector(k, {"name": "C2_eff", "k_map": {"type": "reflection", "axis_deg": 0.0}}),
        np.array([0.25, 0.5]),
    )
    np.testing.assert_allclose(
        ContinuumModelBuilder._apply_k_map_to_vector(k, {"name": "C2TR_eff", "k_map": {"type": "reflection", "axis_deg": 90.0}}),
        np.array([-0.25, -0.5]),
    )


def test_k_action_requires_explicit_metadata() -> None:
    with pytest.raises(ValueError, match="requires explicit k_map"):
        ContinuumModelBuilder._apply_k_map_to_vector(np.array([0.25, -0.5]), {"name": "C2_eff"})


def test_bM_tmat_default(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {"source": "auto"}
    raw["kpath"] = {
        "file": "KPATH.in",
        "tmat": [
            [60.9508659523, 0.0, 0.0],
            [-30.4754329751, 52.7849982976, 0.0],
            [0.0, 0.0, 50.0],
        ],
    }
    (tmp_path / "KPATH.in").write_text(
        "\n".join(["K-Path Generated by VASPKIT.", "1", "Line-Mode", "Reciprocal", "0 0 0 G", "0.5 0 0 M"]),
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    _, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.bM_diagnostics["source"] == "tmat"
    assert "comparison_with_tmat" in model_cfg.bM_diagnostics


def test_explicit_bM_takes_precedence_over_tmat_auto(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]}
    raw["kpath"] = {
        "file": "KPATH.in",
        "tmat": [
            [60.9508659523, 0.0, 0.0],
            [-30.4754329751, 52.7849982976, 0.0],
            [0.0, 0.0, 50.0],
        ],
    }
    (tmp_path / "KPATH.in").write_text(
        "\n".join(["K-Path Generated by VASPKIT.", "1", "Line-Mode", "Reciprocal", "0 0 0 G", "0.5 0 0 M"]),
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    _, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.bM_diagnostics["source"] == "explicit"
    np.testing.assert_allclose(model_cfg.bM_diagnostics["b1"], [1.0, 0.0])
    np.testing.assert_allclose(model_cfg.bM_diagnostics["b2"], [0.0, 1.0])


def test_bM_q_distance_requires_validation(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    q_bad = np.array([[0.0, 0.0], [1.0, 0.0], [0.37, 0.91]], dtype=float)
    np.save(tmp_path / "q1.npy", q_bad)
    np.save(tmp_path / "q2.npy", q_bad)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {
        "source": "q_distance",
        "validation": {"max_error": 1.0e-12, "max_failure_fraction": 0.0},
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="q_distance bM reconstruction validation failed"):
        build_moire_config_from_file(cfg_path)


def test_inter_harmonics_include_sector_offsets(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = "K"
    raw["valley_model"]["active_valleys"] = ["K1"]
    raw["sectors"] = [
        {"name": "K1_L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
        {"name": "K1_L2", "qset": "qset2", "q_offset": [0.2, 0.0], "n_orb": 1},
    ]
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.5, 0.8660254037844386]}
    raw["model"]["harmonics"] = {"intra": 1, "inter": {"count": 1, "sector_pairs": [["K1_L2", "K1_L1"]]}}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    np.testing.assert_allclose(moire_cfg.inter_harmonics_map[1], [0.2, 0.0], atol=1.0e-12)
    selected = model_cfg.harmonics_diagnostics["inter"]["selected"][0]
    assert selected["sector_pair"] == ["K1_L2", "K1_L1"]
    assert selected["support_count"] >= 1


def test_k_inter_auto_harmonics_infers_sector_offsets(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = "K"
    raw["valley_model"]["active_valleys"] = ["K1"]
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.5, 0.8660254037844386]}
    raw["model"]["harmonics"] = {"intra": 1, "inter": {"count": 1}}
    raw.pop("sectors", None)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    selected = model_cfg.harmonics_diagnostics["inter"]["selected"][0]
    np.testing.assert_allclose(moire_cfg.inter_harmonics_map[1], selected["vector"], atol=1.0e-12)
    assert model_cfg.harmonics_diagnostics["inter"]["selection_rule"] == "K_valley_inter_geometry"
    np.testing.assert_allclose(selected["vector"], [0.0, 1.0 / np.sqrt(3.0)], atol=1.0e-12)


def test_build_terms_no_layer1_only() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=2,
        nlow_state=[1, 2],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 1, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"Kinect": [], "Onsite": [], "intra": [], "inter": []},
        term_templates=[
            {"name": "kinetic", "source": "diagonal_kp", "sector_pairs": "same", "orbital_pairs": "diagonal", "max_order": 1},
            {"name": "intra", "source": "moire_potential", "sector_pairs": "same", "orbital_pairs": "all", "harmonics": "intra"},
            {"name": "inter", "source": "tunneling", "sector_pairs": [[2, 1], [1, 2]], "orbital_pairs": "all", "harmonics": "inter"},
        ],
    )

    model = build_model(cfg)
    keys = list(model.terms)
    terms = list(model.terms.values())

    assert any(term.key.layer_from == 2 and term.key.layer_to == 2 and term.tag == "Kinect" for term in terms)
    assert any(key.layer_from == 2 and key.layer_to == 2 for key in keys)
    assert any(key.layer_from == 1 and key.layer_to == 2 for key in keys)
    assert any(key.layer_from == 2 and key.layer_to == 1 for key in keys)


def test_term_templates_accept_named_sector_pairs() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 1, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"inter": []},
        sectors=[
            {"name": "bottom", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
            {"name": "top", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
        ],
        term_templates=[
            {
                "name": "named_inter",
                "source": "tunneling",
                "sector_pairs": [["top", "bottom"]],
                "orbital_pairs": "all",
                "harmonics": {"kind": "inter", "indices": [1]},
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)
    keys = sorted((term.key.layer_from, term.key.layer_to) for term in model.terms.values())

    assert keys == [(2, 1)]


def test_compute_coefficients_skips_empty_orthogonalized_subgroup(monkeypatch) -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 0, "intra": 0, "inter": 0},
        symmetry_map={"intra": []},
        term_templates=[
            {"name": "intra", "source": "moire_potential", "sector_pairs": [[1, 1]], "orbital_pairs": "all", "harmonics": "intra"}
        ],
    )
    model = build_model(cfg)
    builder = model._moire_builder
    heff = np.eye(2, dtype=complex)
    k_points = np.array([[0.0, 0.0]])

    def empty_subset(self, sub_keys, k_points, tol=1.0e-6, tag=None):
        dim = heff.shape[0]
        initial = np.zeros((2 * len(sub_keys), dim, dim), dtype=complex)
        final = np.zeros((0,), dtype=complex)
        return sub_keys, initial, final, np.array([], dtype=int)

    monkeypatch.setattr(ContinuumModelBuilder, "get_orthogonalized_terms_subset", empty_subset)

    diagnostics = builder.compute_coefficients_by_tag(heff, k_points)

    assert diagnostics["intra"][(1, 1, 1, 1)].size == 0
    assert all(term.r_value_real == 0.0 and term.r_value_imag == 0.0 for term in model.terms.values())


def test_physical_tr_closed_fit_group_merges_orbital_partners() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    tr_block = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    tr_matrix = np.zeros((4, 4), dtype=complex)
    tr_matrix[:2, :2] = tr_block
    tr_matrix[2:, 2:] = tr_block

    class StaticGenerator:
        def get_operator(self, name, params=None):
            assert name == "TR"
            return tr_matrix

    tr_op = {
        "name": "TR",
        "antiunitary": True,
        "k_map": {"type": "negation"},
        "q_map": {"type": "negation"},
        "sector_map": "identity",
        "operation_physics_level": "physical",
    }
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=2,
        n_orb2=2,
        nlow_state=[2, 2],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        intra_harmonics_map={},
        inter_harmonics_map={},
        max_order={"Onsite": 0},
        symmetry_map={"Onsite": [tr_op]},
        symmetry_gen=StaticGenerator(),
        term_templates=[
            {
                "name": "onsite_bottom",
                "source": "onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            }
        ],
    )
    model = build_model(cfg)
    builder = model._moire_builder

    diagnostics = builder.compute_coefficients_by_tag(np.diag([2.0, 2.0, 0.0, 0.0]).astype(complex), np.array([[0.0, 0.0]]))

    onsite_terms = [term for term in model.terms.values() if term.tag == "Onsite"]
    assert len(onsite_terms) == 2
    assert sum(term.active for term in onsite_terms) == 1
    assert len(diagnostics["Onsite"]) == 1


def test_fit_uses_symmetry_closed_block_not_raw_subgroup(monkeypatch) -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=2,
        n_orb2=2,
        nlow_state=[2, 2],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        intra_harmonics_map={},
        inter_harmonics_map={},
        max_order={"Onsite": 0},
        symmetry_map={"Onsite": []},
        term_templates=[
            {
                "name": "onsite_bottom",
                "source": "onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            }
        ],
    )
    model = build_model(cfg)
    builder = model._moire_builder
    calls = []

    def keep_all(self, keys, k_points, *, tol, max_exact_group_size=16):
        return keys

    def same_fit_block(self, key):
        return (0, 1), (0, 1)

    def empty_subset(self, sub_keys, k_points, tol=1.0e-6, tag=None):
        calls.append(tuple((key.layer_from, key.layer_to, key.orbital_from, key.orbital_to) for key in sub_keys))
        dim = 2
        initial = np.zeros((2 * len(sub_keys), dim, dim), dtype=complex)
        final = np.zeros((0,), dtype=complex)
        return sub_keys, initial, final, np.array([], dtype=int)

    monkeypatch.setattr(ContinuumModelBuilder, "_filter_duplicate_symmetry_seed_keys", keep_all)
    monkeypatch.setattr(ContinuumModelBuilder, "_fit_block_signature_for_key", same_fit_block)
    monkeypatch.setattr(ContinuumModelBuilder, "get_orthogonalized_terms_subset", empty_subset)

    diagnostics = builder.compute_coefficients_by_tag(np.eye(4, dtype=complex), np.array([[0.0, 0.0]]))

    assert calls == [((1, 1, 1, 1), (1, 1, 2, 2))]
    assert len(diagnostics["Onsite"]) == 1


def test_coefficients_and_term_registry_written(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
        return {
            "eigvals": expected_eigvals,
            "diagnostics": {"fit_rank": 1},
            "model": type("Model", (), {"terms": {key: type("Term", (), {"key": key, "tag": "Kinect", "active": True, "r_value_real": 1.0, "r_value_imag": 0.0, "symmetry_ops": []})()}})(),
        }

    monkeypatch.setattr("kp.model.configured._run_model_pipeline", fake_pipeline)

    run_configured_model(cfg_path)

    assert (tmp_path / "model_out" / "coefficients.json").exists()
    assert (tmp_path / "model_out" / "terms.json").exists()
    assert (tmp_path / "model_out" / "fit_diagnostics.json").exists()
    assert (tmp_path / "model_out" / "operation_registry.json").exists()
    assert (tmp_path / "model_out" / "run_summary.json").exists()

    registry = json.loads((tmp_path / "model_out" / "operation_registry.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "model_out" / "run_summary.json").read_text(encoding="utf-8"))
    assert "production" + "_level" not in summary
    assert "validation_incomplete" in summary
    assert all("user_operation" in row and "canonical_operation" in row for row in registry)


def test_run_configured_model_quiet_writes_detailed_log(monkeypatch, capsys, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")
    fake_model = type("Model", (), {"terms": {}})()

    def fake_build_model(moire_config):
        print("very noisy coefficient dump")
        return fake_model

    def fake_compute_coefficients(moire_config, model):
        print("very noisy fit dump")
        return model, {}

    def fake_compute_bands(moire_config, model, kpoints, return_eigvecs=False):
        print("very noisy coefficient dump")
        return expected_eigvals

    monkeypatch.setattr("kp.model.configured.build_model", fake_build_model)
    monkeypatch.setattr("kp.model.configured.compute_coefficients", fake_compute_coefficients)
    monkeypatch.setattr("kp.model.configured.compute_bands", fake_compute_bands)

    results = run_configured_model(cfg_path)
    captured = capsys.readouterr()

    assert "very noisy coefficient dump" not in captured.out
    assert "[kp model] building continuum terms" in captured.out
    log_path = Path(results["model_log"])
    assert log_path.exists()
    assert "very noisy coefficient dump" in log_path.read_text(encoding="utf-8")
    assert "very noisy fit dump" in log_path.read_text(encoding="utf-8")


def test_matrix_residual_not_only_band() -> None:
    heff = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=complex)
    same_bands_different_matrix = np.array([[[1.5, 0.5], [0.5, 1.5]]], dtype=complex)

    band_metrics = compare_bands(np.linalg.eigvalsh(same_bands_different_matrix), np.linalg.eigvalsh(heff))
    residual = matrix_residual(same_bands_different_matrix, heff)

    assert band_metrics["max_abs_error"] < 1.0e-12
    assert residual["max_residual"] > 0.1


def test_symmetry_from_kp_symm_output(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C3_low_raw.npy", matrix)
    (tmp_path / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120},
                        "sector_map": "identity",
                        "spin_map": "identity",
                        "valley_map": "identity",
                        "residual": 0.0,
                        "leakage": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "operations": ["C3z"]}, base=tmp_path, expected_dim=2)

    np.testing.assert_allclose(source.generator.get_operator("C3z", 1), matrix)
    assert source.metadata["operations"][0]["name"] == "C3z"
    assert source.source_type == "kp_symm_output"


def test_symmetry_source_infers_operations_from_summary(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C3_low_raw.npy", matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "operation": "C3",
                        "matrix_file": "C3_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 1.2e-11,
                                    "subspace_leakage": 3.4e-9,
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)

    np.testing.assert_allclose(source.generator.get_operator("C3z", 1), matrix)
    op = source.metadata["operations"][0]
    assert op["name"] == "C3z"
    assert op["matrix_file"].endswith("C3_low_raw.npy")
    assert op["residual"] == 1.2e-11
    assert op["leakage"] == 3.4e-9


def test_symmetry_source_loads_effective_m_ops_from_tapw_sewing_labels(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    t_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    c2_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    np.save(tmp_path / "T_low_raw.npy", t_matrix)
    np.save(tmp_path / "C2_low_raw.npy", c2_matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(antiunitary=True),
                        "name": "TR_eff",
                        "operation": "TR",
                        "matrix_file": "T_low_raw.npy",
                        "antiunitary": True,
                        "k_map": {"type": "negation"},
                        "q_map": {"type": "negation"},
                        "sector_map": "identity",
                        "action_source": "raw_h_operator_file",
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 1.0e-12,
                                    "subspace_leakage": 2.0e-12,
                                }
                            }
                        ],
                    },
                    {
                        **_source_meta(),
                        "name": "C2_eff",
                        "operation": "C2",
                        "source_matrix_role": "D0_times_PG_action",
                        "gauge_correction": {"kind": "periodic_atomic_bloch_PG", "file": "representations/M1/C2_PG.npz"},
                        "matrix_file": "C2_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "sector_map": "layer_exchange",
                        "action_source": "representation_file+pg_file",
                        "pg_file": "representations/M1/C2_PG.npz",
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 3.0e-12,
                                    "subspace_leakage": 4.0e-12,
                                }
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source(
        {
            "type": "kp_symm_output",
            "path": str(tmp_path),
            "use": "raw",
            "matrix_kind": "action",
            "operations": [
                {"name": "TR_eff", "operation": "TR"},
                {"name": "C2_eff", "operation": "C2"},
            ],
                    },
        base=tmp_path,
        expected_dim=2,
    )

    np.testing.assert_allclose(source.generator.get_operator("TR_eff"), t_matrix)
    with pytest.raises(ValueError, match="not loaded"):
        source.generator.get_operator("TR")
    np.testing.assert_allclose(source.generator.get_operator("C2_eff"), c2_matrix)
    with pytest.raises(ValueError, match="not loaded"):
        source.generator.get_operator("C2")
    by_name = {op["name"]: op for op in source.metadata["operations"]}

    t_op = by_name["TR_eff"]
    assert t_op["operation"] == "TR"
    assert "aliases" not in t_op
    assert t_op["matrix_file"].endswith("T_low_raw.npy")
    assert t_op["matrix_kind"] == "action"
    assert t_op["source_matrix_role"] == "raw_h_sewing_action"
    assert t_op["target_role"] == "continuum_internal_rep"
    assert t_op["antiunitary_convention"] == "U_K"
    assert "production" + "_use" not in t_op

    c2_op = by_name["C2_eff"]
    assert c2_op["operation"] == "C2"
    assert "aliases" not in c2_op
    assert c2_op["matrix_file"].endswith("C2_low_raw.npy")
    assert c2_op["source_matrix_role"] == "D0_times_PG_action"
    assert c2_op["gauge_correction"] == {"kind": "periodic_atomic_bloch_PG", "file": "representations/M1/C2_PG.npz"}


def test_symmetry_source_requires_explicit_k_map_instead_of_axis_deg(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C2_low_raw.npy", matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C2",
                        "operation": "C2",
                        "matrix_file": "C2_low_raw.npy",
                        "axis_deg": 150.0,
                        "antiunitary": False,
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 0.0,
                                    "subspace_leakage": 0.0,
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires explicit k_map"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)


def test_symmetry_source_can_load_representation_low_matrix(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    action_matrix = np.eye(2, dtype=complex)
    representation_matrix = -np.eye(2, dtype=complex)
    np.save(tmp_path / "C2_low_raw.npy", action_matrix)
    np.save(tmp_path / "C2_low_representation_raw.npy", representation_matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(representation=True),
                        "name": "C2",
                        "operation": "C2",
                        "matrix_file": "C2_low_representation_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "sector_map": "layer_exchange",
                        "antiunitary": False,
                        "action_source": "raw_h_operator_file",
                        "pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source(
        {"type": "kp_symm_output", "path": str(tmp_path), "use": "raw", "matrix_kind": "representation"},
        base=tmp_path,
        expected_dim=2,
    )

    np.testing.assert_allclose(source.generator.get_operator("C2"), representation_matrix)
    op = source.metadata["operations"][0]
    assert op["matrix_kind"] == "representation"
    assert op["source_matrix_role"] == "bare_D0_internal_rep"
    assert op["target_role"] == "continuum_internal_rep"


def test_symmetry_source_rejects_invalid_representation_by_default(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C3_low_representation_raw.npy", matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                        {
                            **_source_meta(representation=True),
                            "name": "C3z",
                            "operation": "C3",
                            "matrix_file": "C3_low_representation_raw.npy",
                            "antiunitary": False,
                            "k_map": {"type": "rotation", "angle_deg": 120.0},
                            "q_map": {"type": "rotation", "angle_deg": 120.0},
                            "sector_map": "identity",
                            "representation_pairs": [
                            {
                                "quality_warnings": ["singular values deviate from 1 by 4.476e-01"],
                                "raw": {
                                    "heff_covariance_residual": 0.0,
                                    "subspace_leakage": 0.4476,
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="projection quality warnings"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path), "use": "raw", "matrix_kind": "representation"},
            base=tmp_path,
            expected_dim=2,
        )

    source = load_symmetry_source(
        {
            "type": "kp_symm_output",
            "path": str(tmp_path),
            "use": "raw",
            "matrix_kind": "representation",
            "allow_invalid_representation": True,
        },
        base=tmp_path,
        expected_dim=2,
    )

    assert source.metadata["operations"][0]["projection_quality_warnings"]


def test_build_moire_config_enriches_symmetry_map_with_rotated_k_map(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["coordinate_frame"] = {"rotation_deg": 30.0}
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "representation"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2"}],
        "Onsite": [],
        "intra": [],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    np.save(symm_dir / "C2_low_representation_raw.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                        {
                            **_source_meta(representation=True),
                            "name": "C2",
                            "matrix_file": "C2_low_representation_raw.npy",
                            "k_map": {"type": "reflection", "axis_deg": 150.0},
                            "q_map": {"type": "reflection", "axis_deg": 150.0},
                            "antiunitary": False,
                            "sector_map": "identity",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    op = moire_cfg.symmetry_map["Kinect"][0]
    assert model_cfg.symmetry_map["Kinect"][0]["k_map"]["type"] == "reflection"
    assert op["k_map"]["axis_deg"] == pytest.approx(180.0)


def test_build_moire_config_prefers_model_frame_action_from_symm_artifact(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["coordinate_frame"] = {"rotation_deg": 30.0}
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "representation"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2"}],
        "Onsite": [],
        "intra": [],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    np.save(symm_dir / "C2_low_representation_raw.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(representation=True),
                        "name": "C2",
                        "matrix_file": "C2_low_representation_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "model_action": {
                            "antiunitary": False,
                            "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                            "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                            "sector_map": "identity",
                        },
                        "antiunitary": False,
                        "sector_map": "identity",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, _model_cfg = build_moire_config_from_file(cfg_path)

    op = moire_cfg.symmetry_map["Kinect"][0]
    assert op["k_map"]["axis_deg"] == pytest.approx(180.0)
    assert op["q_map"]["axis_deg"] == pytest.approx(180.0)
    assert op["k_map"]["in_model_frame"] is True


def test_float_p_key_canonicalization() -> None:
    from kp.model.config_schema import canonical_vector_key

    assert canonical_vector_key([0.1 + 0.2, 0.0], tol=1.0e-9) == canonical_vector_key([0.3, 0.0], tol=1.0e-9)
    assert ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.1 + 0.2, 0.0)) == ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.3, 0.0))


def test_C2T_symmetrization_uses_antiunitary_conjugation() -> None:
    from kp.model.symmetry import MatrixSymmetryGenerator

    generator = MatrixSymmetryGenerator({"C2T": np.eye(2, dtype=complex)}, {"operations": []})
    Y = np.array([[1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 6.0j, 7.0 - 8.0j]], dtype=complex)

    transformed = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y.copy(), "C2T", None, generator)

    np.testing.assert_allclose(transformed, Y.conj())


def test_full_bilayer_block_detection_accepts_exactified_c2t_action() -> None:
    ops = [
        {
            "name": "C2T",
            "antiunitary": True,
            "sector_map": "layer_exchange",
            "k_map": {"type": "reflection", "axis_deg": 180.0},
        }
    ]

    assert ContinuumModelBuilder._uses_full_bilayer_block(ops) is True


def test_full_bilayer_block_detection_accepts_generic_two_sector_exchange() -> None:
    ops = [
        {
            "name": "TR_eff",
            "antiunitary": True,
            "sector_map": {"bottom": "top", "top": "bottom"},
        }
    ]

    assert ContinuumModelBuilder._uses_full_bilayer_block(ops, ["bottom", "top"]) is True


def test_run_configured_model_saves_auto_harmonics_diagnostic_plot(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.configured._run_model_pipeline", fake_pipeline)

    run_configured_model(cfg_path)

    assert (tmp_path / "model_out" / "harmonics_diagnostic.png").exists()
    assert (tmp_path / "model_out" / "harmonics_diagnostic.json").exists()


def test_run_configured_model_saves_outputs_without_legacy_diagnostics_json(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        assert moire_config.output_dir is None
        return {"eigvals": expected_eigvals, "diagnostics": {("tuple", "key"): 1}}

    monkeypatch.setattr("kp.model.configured._run_model_pipeline", fake_pipeline)

    results = run_configured_model(cfg_path)

    assert results["comparison"]["max_abs_error"] == 0.0
    assert results["moire_config"].output_dir == tmp_path / "model_out"
    assert (tmp_path / "model_out" / "eigvals.npy").exists()
    assert (tmp_path / "model_out" / "comparison.json").exists()
    assert (tmp_path / "model_out" / "band_comparison.png").exists()
    assert results["band_plot"] == str((tmp_path / "model_out" / "band_comparison.png").resolve())


def test_compare_bands_reports_rms_and_max_error() -> None:
    model = np.array([[0.0, 1.1, 2.0], [0.0, 2.0, 4.0]])
    heff = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 4.0]])

    metrics = compare_bands(model, heff, band_slice=[0, 2])

    assert metrics["num_kpoints"] == 2
    assert metrics["num_bands"] == 2
    assert metrics["rms_error"] == np.sqrt((0.0**2 + 0.1**2 + 0.0**2 + 0.5**2) / 4)
    assert metrics["max_abs_error"] == 0.5


def test_compare_bands_for_plot_supports_bottom_bands_with_bottom_alignment() -> None:
    model = np.array([[1.2, 2.0, 3.0], [1.4, 2.1, 3.2]])
    heff = np.array([[1.0, 2.0, 3.1], [1.1, 2.2, 3.2]])

    metrics = compare_bands_for_plot(
        model,
        heff,
        plot_config={"bottom_bands": 2, "align": "bottom"},
    )

    assert metrics["band_slice"] == [0, 2]
    assert metrics["align"] == "bottom"
    assert metrics["num_bands"] == 2
    assert metrics["aligned_rms_error_meV"] == pytest.approx(187.08286933869707)
    assert metrics["model_alignment_shift_meV"] == pytest.approx(200.0)


def test_run_configured_model_preserves_plot_ylim_in_config(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.setdefault("bands", {})
    raw["bands"].setdefault("plot", {})
    raw["bands"]["plot"]["ylim"] = [0.0, 0.16]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.configured._run_model_pipeline", fake_pipeline)
    results = run_configured_model(cfg_path)

    assert results["configured_model"].band_plot_config["ylim"] == [0.0, 0.16]


def test_M_spinless_kp_symm_output_physical_source_ops_are_relabelled_effective(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "M",
        "mode": "single_valley",
        "active_valleys": ["M1"],
        "spin_convention": "spinless_effective",
        "allowed_internal_symmetries": ["TR_eff", "C2_eff"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "use": "raw",
        "matrix_kind": "action",
        "operations": ["TR", "C2"],
                "source_matrix_role": "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "raw_low_heff_sewing",
        "gauge_correction": {"kind": "none"},
        "antiunitary_convention": "U_K",
    }
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2_eff"}, {"name": "TR_eff"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert [op["name"] for op in cfg.symmetry_source_config["operations"]] == ["TR_eff", "C2_eff"]
    assert all("aliases" not in op for op in cfg.symmetry_source_config["operations"])
    assert [op["name"] for op in cfg.symmetry_map["Kinect"]] == ["C2_eff", "TR_eff"]


def test_production_strict_rejects_unavailable_validation_outputs(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["validation"] = {"strict": True}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.configured._run_model_pipeline", fake_pipeline)

    with pytest.raises(ValueError, match="validation.strict=true requires"):
        run_configured_model(cfg_path)


def test_explicit_harmonics_use_neutral_registry_source() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0]), 2: np.array([1.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 1, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"intra": []},
        term_templates=[
            {
                "name": "intra_layer1_nonzero",
                "source": "moire_potential",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "all",
                "harmonics": {"kind": "intra", "indices": [2]},
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)
    term = next(iter(model.terms.values()))
    assert term.registry_metadata["harmonics_source"] == "explicit_indices"
    assert term.registry_metadata["generated_by"] == "representation_invariant_generator"


def test_symmetry_representative_harmonics_do_not_double_count_orbit() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    tr_op = {
        "name": "TR",
        "antiunitary": True,
        "k_map": {"type": "negation"},
        "q_map": {"type": "negation"},
        "sector_map": "identity",
    }
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([1.0, 0.0]), 2: np.array([-1.0, 0.0])},
        inter_harmonics_map={},
        max_order={"intra": 0},
        symmetry_map={"intra": [tr_op]},
        term_templates=[
            {
                "name": "intra_reps",
                "source": "moire_potential",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "all",
                "harmonics": "intra",
                "harmonics_source": "symmetry_representatives",
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)

    assert len(model.terms) == 1
    term = next(iter(model.terms.values()))
    assert term.registry_metadata["harmonics_source"] == "symmetry_representatives"


def test_symmetry_representative_harmonics_apply_sector_map() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    c2_op = {
        "name": "C2",
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0},
        "q_map": {"type": "reflection", "axis_deg": 0.0},
        "sector_map": "layer_exchange",
    }
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={},
        inter_harmonics_map={1: np.array([0.25, 0.0])},
        max_order={"inter": 0},
        symmetry_map={"inter": [c2_op]},
        term_templates=[
            {
                "name": "inter_reps",
                "source": "tunneling",
                "sector_pairs": [[2, 1], [1, 2]],
                "orbital_pairs": "all",
                "harmonics": "inter",
                "harmonics_source": "symmetry_representatives",
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)

    assert len(model.terms) == 1
    term = next(iter(model.terms.values()))
    assert term.registry_metadata["harmonic_orbit_size"] == 2
