from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.model.core import ContinuumModelBuilder, ContinuumTerm, ContinuumTermKey  # noqa: E402
from kp.model import export as export_module  # noqa: E402
from kp.model.export import (  # noqa: E402
    _dense_composed_symmetry_action,
    _expand_operator_recipe,
    _iter_transformed_sparse_entries,
    export_all_standalone_models,
    export_standalone_model,
)
from kp.model.pipeline import build_moire_config_from_file, run_configured_model  # noqa: E402


def _write_symm_frame_manifest(
    base: Path,
    *,
    case_id: str = "toy_K1",
    dim: int = 2,
    rotation_deg: float = 0.0,
    operations: tuple[str, ...] = ("C3z", "C2T"),
) -> Path:
    symm_dir = base / "outputs" / "symm" / case_id
    symm_dir.mkdir(parents=True, exist_ok=True)
    for operation in operations:
        np.save(symm_dir / f"exactified_{operation}.npy", np.eye(dim, dtype=np.complex128))
    np.save(symm_dir / "q_model_layer1.npy", np.array([[0.0, 0.0]], dtype=float))
    np.save(symm_dir / "q_model_layer2.npy", np.array([[0.0, 0.0]], dtype=float))
    manifest = {
        "exactification_owner": "kp_symm",
        "frame": {"q_transform": {"rotation_deg": float(rotation_deg)}},
        "q_model": {"files": {"layer1": "q_model_layer1.npy", "layer2": "q_model_layer2.npy"}},
        "operations": [
            {
                "name": operation,
                "operation": operation,
                "matrix_kind": "continuum_internal_rep_exact",
                "matrix_source": "kp_symm_exactified_action",
                "matrix_file": f"exactified_{operation}.npy",
                "antiunitary": operation in {"C2T", "TR"},
                "k_map": {"type": "identity"},
                "q_map": {"type": "identity"},
                "sector_map": "identity",
                "source_matrix_role": "raw_h_sewing_action",
                "source_gauge": "raw_saved_TAPW",
                "target_role": "continuum_internal_rep",
                "gauge_correction": {"kind": "none"},
                "antiunitary_convention": "U_K" if operation in {"C2T", "TR"} else "none",
                "spin_map": "from_kp_symm_output",
                "valley_map": "identity",
            }
            for operation in operations
        ],
    }
    (symm_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return symm_dir


def _write_export_fixture(
    tmp_path: Path,
    *,
    case_id: str = "toy_K1",
    valley: str = "K1",
    spin_convention: str = "spin_up_projected",
    valley_type: str = "K",
    n_orb: tuple[int, int] = (1, 1),
    operations: tuple[str, ...] = ("C3z", "C2T"),
    sectors: list[dict] | None = None,
    term_templates: list[dict] | None = None,
    standalone_override: bool = True,
) -> tuple[Path, Path]:
    root = tmp_path / "example" / "kp"
    root.mkdir(parents=True)

    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0]], dtype=float)
    kpoints = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]], dtype=float)
    dim = q1.shape[0] * int(n_orb[0]) + q2.shape[0] * int(n_orb[1])
    diagonal_base = np.linspace(0.25, 0.25 + 0.1 * max(dim - 1, 0), dim)
    heff = np.stack([np.diag(diagonal_base + 0.01 * k[0] + 0.02 * k[1]) for k in kpoints]).astype(np.complex128)

    np.save(root / "q1.npy", q1)
    np.save(root / "q2.npy", q2)
    np.save(root / "kpoints.npy", kpoints)
    project_dir = root / "outputs" / "project" / case_id
    project_dir.mkdir(parents=True)
    np.save(project_dir / "heff_list.npy", heff)
    np.save(project_dir / "heff_eig.npy", np.linalg.eigvalsh(heff))

    source_cfg = {
        "material": {
            "name": "Toy",
            "spin": "up",
            "energy_unit": "eV",
            "qset1_file": "../../q1.npy",
            "qset2_file": "../../q2.npy",
        },
        "plot": {},
        "project": {"out_dir": f"../../outputs/project/{case_id}"},
    }
    source_dir = root / "configs" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / f"{case_id}.yaml").write_text(yaml.safe_dump(source_cfg), encoding="utf-8")
    _write_symm_frame_manifest(root, case_id=case_id, dim=dim, operations=operations)

    if term_templates is None:
        term_templates = [
            {
                "name": "kinetic_layer1",
                "source": "diagonal_kp",
                "tag": "Kinect",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": [[1, 1]],
                "max_order": 1,
                "monomial_constraints": {"require_mz_ge_mz_star": True},
            },
            {
                "name": "onsite_layer2",
                "source": "onsite",
                "tag": "Onsite",
                "sector_pairs": [[2, 2]],
                "orbital_pairs": [[1, 1]],
                "max_order": 0,
            },
        ]

    model_cfg = {
        "source_config": f"../source/{case_id}.yaml",
        "symmetry_source": {"type": "kp_symm_output", "path": f"../../outputs/symm/{case_id}"},
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": valley_type,
            "mode": "single_valley",
            "active_valleys": [valley],
            "spin_convention": spin_convention,
            "allowed_internal_symmetries": list(operations),
            "external_sewing_symmetries": [],
        },
        "kpoints_file": "../../kpoints.npy",
        "model": {
            "n_orb": list(n_orb),
            "nlow_state": list(n_orb),
            "bM": {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]},
            "harmonics": {"intra": {1: "zero"}, "inter": {1: "zero"}},
            "max_order": {"Kinect": 1, "intra": 0, "inter": 0},
            "symmetry_map": {"Kinect": [], "Onsite": [], "intra": [], "inter": []},
            "term_templates": term_templates,
        },
        "fit": {"indices": [0, 1, 2], "coeff_tol": 1.0e-8},
        "bands": {"indices": [0, 1, 2], "compare_to_heff": True},
        "output": {"dir": f"../../outputs/model/{case_id}", "progress": False},
    }
    if sectors is not None:
        model_cfg["sectors"] = sectors
    model_dir = root / "configs" / "model"
    model_dir.mkdir(parents=True)
    cfg_path = model_dir / f"{case_id}.yaml"
    cfg_path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    if standalone_override:
        standalone_cfg = {
            "coordinate_convention": {
                "type": "fractional_model_basis",
                "k_units": "fractional coordinates in bM1/bM2 basis",
                "q_units": "same as k",
                "hsp_coordinates_are": "fractional_model_basis",
            },
            "high_symmetry_points": {"G": [0.0, 0.0], "X": [0.5, 0.0], "Y": [0.0, 0.5]},
            "default_kpath": ["G", "X", "Y", "G"],
            "points_per_segment": 4,
            "default_band_slice": None,
        }
        (model_dir / "standalone_export.yaml").write_text(
            yaml.safe_dump(standalone_cfg, sort_keys=False),
            encoding="utf-8",
        )

    run_configured_model(cfg_path)
    return root / "outputs" / "model" / case_id, cfg_path


def _load_exported_evaluator(package_dir: Path):
    spec = importlib.util.spec_from_file_location("standalone_eval", package_dir / "evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_export_does_not_require_or_write_license(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(tmp_path)
    out_dir = tmp_path / "standalone"

    export_standalone_model(model_output, out_dir)

    assert not (out_dir / "LICENSE").exists()


def test_standalone_export_removes_stale_model_json_in_place(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(tmp_path)
    (model_output / "model.json").write_text('{"stale": true}\n', encoding="utf-8")

    export_standalone_model(model_output, model_output, force=True)

    assert not (model_output / "model.json").exists()
    assert (model_output / "model_data.npz").exists()


def test_expand_operator_recipe_caches_duplicate_monomial_transforms(monkeypatch) -> None:
    def y_basis(_k):
        return np.eye(2, dtype=np.complex128)

    y_basis.eval_sparse = lambda _k: (np.array([0, 1]), np.array([0, 1]), np.ones(2, dtype=complex))
    y_basis._moire_sparse_rows = np.array([0, 1], dtype=int)
    y_basis._moire_sparse_cols = np.array([0, 1], dtype=int)
    y_basis._moire_sparse_row_q_idx = np.array([0, 0], dtype=int)
    y_basis._moire_sparse_Q_rows = np.array([[0.0, 0.0]], dtype=float)
    y_basis._moire_sparse_hermitize_in_basis = False
    term = ContinuumTerm(
        key=ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0)),
        Y_basis=y_basis,
        r_value_real=1.0,
        r_value_imag=0.0,
        tag="Onsite",
        symmetry_ops=[],
    )
    calls = []
    original = export_module._transform_monomial

    def spy(transform, q_base, mz, mz_star):
        calls.append((tuple(np.asarray(q_base, dtype=float)), int(mz), int(mz_star)))
        return original(transform, q_base, mz, mz_star)

    monkeypatch.setattr(export_module, "_transform_monomial", spy)

    data = _expand_operator_recipe([term], SimpleNamespace(symmetry_gen=None), dim=2)

    assert data["row"].size >= 2
    assert len(calls) == 1


def test_standalone_export_default_layout_and_user_run(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(tmp_path)
    out_dir = tmp_path / "standalone"

    export_standalone_model(model_output, out_dir)

    assert sorted(path.name for path in out_dir.iterdir()) == [
        "MODEL.md",
        "README.md",
        "evaluate.py",
        "model_data.npz",
    ]

    evaluator_text = (out_dir / "evaluate.py").read_text(encoding="utf-8")
    assert "import kp" not in evaluator_text
    assert "from kp" not in evaluator_text
    assert "allow_pickle=False" in evaluator_text
    assert "eval(" not in evaluator_text
    assert "exec(" not in evaluator_text
    assert "--self-test" not in evaluator_text
    assert "BM1 = [" in evaluator_text
    assert "np.array(" not in evaluator_text.split("# End user-editable settings", 1)[0]
    assert "WINDOW_BANDS = None" in evaluator_text
    assert "OUT_BAND_PLOT = \"bands.pdf\"" in evaluator_text
    assert "BAND_SLICE" not in evaluator_text
    assert "model.json" not in evaluator_text
    assert "max_antihermitian_norm" not in evaluator_text
    assert "anti-Hermitian residual" not in evaluator_text

    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    proc = subprocess.run(
        [sys.executable, "evaluate.py"],
        cwd=out_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out_dir / "kpoints.npy").exists()
    assert (out_dir / "bands.npy").exists()
    assert (out_dir / "bands.pdf").exists()
    assert (out_dir / "kdist.npy").exists()
    assert (out_dir / "kpath_ticks.json").exists()
    ticks = json.loads((out_dir / "kpath_ticks.json").read_text(encoding="utf-8"))
    assert ticks["labels"] in ([], ["G", "X", "Y", "G"])
    assert len(ticks["positions"]) == len(ticks["labels"])

    data = np.load(out_dir / "model_data.npz", allow_pickle=False)
    np.testing.assert_allclose(np.load(out_dir / "kpoints.npy"), data["reference_kpoints"])
    expected_keys = {
        "qset1",
        "qset2",
        "qset_layer1",
        "qset_layer2",
        "exactified_C3z",
        "exactified_C2T",
        "operator_term_index",
        "operator_row",
        "operator_col",
        "operator_mz",
        "operator_mz_star",
        "operator_q_center",
        "operator_prefactor_real",
        "operator_prefactor_imag",
        "term_r_value_real",
        "term_r_value_imag",
        "dimension_dim",
        "runtime_hermitianize_before_eigvalsh",
        "reference_kpoints",
        "reference_eigvals",
        "reference_heff_eig",
    }
    assert expected_keys.issubset(set(data.files))
    assert not (out_dir / "model.json").exists()
    assert int(np.asarray(data["dimension_dim"]).item()) == 2
    assert bool(np.asarray(data["runtime_hermitianize_before_eigvalsh"]).item()) is True
    assert "runtime_max_antihermitian_norm" not in data.files

    model_doc = (out_dir / "MODEL.md").read_text(encoding="utf-8")
    assert "$$" in model_doc
    assert len(model_doc.splitlines()) <= 190
    assert "H_X(k)" in model_doc
    assert "operator_prefactor_real" in model_doc
    assert "Kinetic (`\"Kinect\"` tag" in model_doc
    assert "`Kinect`" in model_doc
    assert "`Onsite`" in model_doc
    assert "`intra`" in model_doc
    assert "`inter`" in model_doc
    assert "[\\rho_{\\mathrm{C3z}}]" in model_doc
    assert "phase angle" in model_doc
    assert "\\pi:\\quad" not in model_doc
    assert "0\\mapsto" not in model_doc


def test_standalone_export_supports_spinful_two_orbital_models(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(
        tmp_path,
        case_id="toy_K1_spinful",
        spin_convention="spinful",
        n_orb=(2, 2),
        term_templates=[
            {
                "name": "onsite_layer1",
                "source": "onsite",
                "tag": "Onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            },
            {
                "name": "onsite_layer2",
                "source": "onsite",
                "tag": "Onsite",
                "sector_pairs": [[2, 2]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            },
        ],
    )
    out_dir = tmp_path / "standalone_spinful"

    export_standalone_model(model_output, out_dir)

    assert not (out_dir / "model.json").exists()
    data = np.load(out_dir / "model_data.npz", allow_pickle=False)
    assert int(np.asarray(data["dimension_dim"]).item()) == 4
    model_doc = (out_dir / "MODEL.md").read_text(encoding="utf-8")
    assert "Spin convention: `spinful`" in model_doc
    assert "| `qset2` | 2 | 1 | 2 | 2 |" in model_doc
    proc = subprocess.run(
        [sys.executable, "evaluate.py"],
        cwd=out_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_standalone_export_dense_symmetry_action_expands_sparse_entries() -> None:
    class DenseGenerator:
        def __init__(self) -> None:
            self.matrix = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)

        def get_operator(self, name: str, params=None):
            assert name == "C3z"
            assert params in {1, -1}
            return self.matrix

    action = _dense_composed_symmetry_action(DenseGenerator(), (("C3z", 1),))
    entries = list(
        _iter_transformed_sparse_entries(
            action,
            np.array([0], dtype=int),
            np.array([1], dtype=int),
        )
    )

    observed = {(row, col): value for _pos, row, col, value, is_anti in entries if not is_anti}
    assert set(observed) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert observed[(0, 0)] == pytest.approx(0.5 + 0.0j)
    assert observed[(0, 1)] == pytest.approx(-0.5 + 0.0j)
    assert observed[(1, 0)] == pytest.approx(0.5 + 0.0j)
    assert observed[(1, 1)] == pytest.approx(-0.5 + 0.0j)


def test_standalone_export_supports_single_active_qset_models(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(
        tmp_path,
        case_id="toy_K1_B",
        spin_convention="spin_down_projected",
        n_orb=(0, 1),
        operations=("C3z",),
        sectors=[{"name": "B_top", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1}],
        term_templates=[
            {
                "name": "onsite_B",
                "source": "onsite",
                "tag": "Onsite",
                "sector_pairs": [["B_top", "B_top"]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            },
        ],
    )
    out_dir = tmp_path / "standalone_single_slot"

    export_standalone_model(model_output, out_dir)

    assert not (out_dir / "model.json").exists()
    data = np.load(out_dir / "model_data.npz", allow_pickle=False)
    assert int(np.asarray(data["dimension_dim"]).item()) == 1
    model_doc = (out_dir / "MODEL.md").read_text(encoding="utf-8")
    assert "Spin convention: `spin_down_projected`" in model_doc
    assert "| `qset1` | 0 | 1 | 0 | 0 |" in model_doc
    assert "| `qset2` | 0 | 1 | 1 | 1 |" in model_doc
    assert "Production operations: `C3z`" in model_doc
    assert "exactified_C3z" in np.load(out_dir / "model_data.npz", allow_pickle=False).files
    assert "exactified_C2T" not in np.load(out_dir / "model_data.npz", allow_pickle=False).files


def test_standalone_export_matches_in_package_hamiltonian(tmp_path: Path) -> None:
    model_output, cfg_path = _write_export_fixture(tmp_path)
    out_dir = tmp_path / "standalone"
    export_standalone_model(model_output, out_dir)

    module = _load_exported_evaluator(out_dir)
    standalone_model = module.load_model(out_dir)

    moire_config, _model_config = build_moire_config_from_file(cfg_path)
    model = type("Model", (), {"terms": {}})()
    active_terms = json.loads((model_output / "active_terms.json").read_text(encoding="utf-8"))
    for row in active_terms:
        raw_key = row["key"]
        key = ContinuumTermKey(
            raw_key["Mz"],
            raw_key["Mz_star"],
            raw_key["layer_from"],
            raw_key["layer_to"],
            raw_key["orbital_from"],
            raw_key["orbital_to"],
            tuple(raw_key["p"]),
        )
        model.terms[key] = ContinuumTerm(
            key=key,
            Y_basis=ContinuumModelBuilder.make_Y_basis_function(
                key, moire_config.Q_set1, moire_config.Q_set2, moire_config.n_orb1, moire_config.n_orb2
            ),
            r_value_real=float(row["r_value_real"]),
            r_value_imag=float(row["r_value_imag"]),
            active=True,
            tag=row["tag"],
            symmetry_ops=row["symmetry_ops"],
        )

    from kp.model.core import _compute_one_k, _prepare_band_state  # noqa: WPS433

    state = _prepare_band_state(moire_config, model)
    for k in np.array([[0.0, 0.0], [0.31, -0.17], [-0.2, 0.11]], dtype=float):
        h_pkg, *_ = _compute_one_k(0, k, state, solve_eig=False)
        h_exported = standalone_model.hamiltonian(k)
        np.testing.assert_allclose(h_exported, h_pkg, atol=1.0e-9)


def test_standalone_export_debug_files_are_opt_in(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(tmp_path)
    out_dir = tmp_path / "standalone"

    export_standalone_model(model_output, out_dir)
    assert not (out_dir / "debug").exists()

    debug_dir = tmp_path / "standalone_debug"
    export_standalone_model(model_output, debug_dir, debug_files=True)
    assert (debug_dir / "debug" / "manifest.json").exists()
    assert (debug_dir / "debug" / "terms.json").exists()
    assert (debug_dir / "debug" / "operations.json").exists()
    assert (debug_dir / "debug" / "comparison.json").exists()
    assert (debug_dir / "debug" / "operator_terms_schema.md").exists()
    assert (debug_dir / "debug" / "symmetry_reconstruction.md").exists()
    debug_doc = (debug_dir / "debug" / "symmetry_reconstruction.md").read_text(encoding="utf-8")
    assert "\\pi:\\quad" in debug_doc
    assert "0\\mapsto" in debug_doc
    assert not (debug_dir / "debug" / "arrays").exists()


def test_standalone_export_requires_explicit_kpath_metadata(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(tmp_path, standalone_override=False)

    with pytest.raises(ValueError, match="standalone export requires explicit coordinate convention"):
        export_standalone_model(model_output, tmp_path / "standalone")


def test_standalone_export_is_deterministic(tmp_path: Path) -> None:
    model_output, _cfg_path = _write_export_fixture(tmp_path)
    out_a = tmp_path / "standalone_a"
    out_b = tmp_path / "standalone_b"

    export_standalone_model(model_output, out_a)
    export_standalone_model(model_output, out_b)

    assert not (out_a / "model.json").exists()
    assert not (out_b / "model.json").exists()
    data_a = np.load(out_a / "model_data.npz", allow_pickle=False)
    data_b = np.load(out_b / "model_data.npz", allow_pickle=False)
    assert set(data_a.files) == set(data_b.files)
    for key in data_a.files:
        np.testing.assert_array_equal(data_a[key], data_b[key])


def test_export_all_standalone_models_dry_run_reports_blockers(tmp_path: Path) -> None:
    ok_output, _cfg_path = _write_export_fixture(tmp_path / "ok", case_id="toy_ok")
    blocked_output, _blocked_cfg = _write_export_fixture(
        tmp_path / "blocked",
        case_id="toy_blocked",
        standalone_override=False,
    )
    examples_root = tmp_path
    output_root = tmp_path / "exports"

    report = export_all_standalone_models(examples_root, output_root, dry_run=True)

    assert not output_root.exists()
    assert any(Path(row["model_output_dir"]) == ok_output for row in report["exportable"])
    blocked = [row for row in report["blocked"] if Path(row["model_output_dir"]) == blocked_output]
    assert blocked
    assert "coordinate convention" in blocked[0]["reason"]


def test_cli_model_export_standalone_subcommand(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod

    model_output = tmp_path / "model_out"
    export_output = tmp_path / "standalone"
    calls: dict[str, object] = {}

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        calls["export"] = (model_output_dir, output_dir, force, debug_files)
        return Path(output_dir)

    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(
        [
            "model",
            "export-standalone",
            str(model_output),
            str(export_output),
            "--debug-files",
        ]
    )

    assert calls["export"] == (model_output, str(export_output), False, True)


def test_cli_model_config_exports_standalone_inside_output_dir_by_default(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"
    model_output.mkdir()
    (model_output / "run_summary.json").write_text("{}", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakeModelConfig:
        path = cfg_path
        output_dir = model_output
        n_orb = (1, 1)
        bM_diagnostics = {"source": "explicit"}
        symmetry_source_metadata = {"operations": []}
        fit_indices = [0]

    class FakeMoireConfig:
        Q_set1 = np.zeros((1, 2))
        Q_set2 = np.zeros((1, 2))
        n_orb1 = 1
        n_orb2 = 1
        kpoints = np.zeros((1, 2))

    def fake_run(path: str) -> dict:
        calls["run"] = path
        return {
            "configured_model": FakeModelConfig(),
            "moire_config": FakeMoireConfig(),
            "comparison": {"rms_error": 0.0, "max_abs_error": 0.0},
        }

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        calls["export"] = (Path(model_output_dir), Path(output_dir), force, debug_files)
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path)])

    assert calls["run"] == str(cfg_path)
    assert calls["export"] == (model_output, model_output / "standalone", True, False)
    summary = json.loads((model_output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["standalone_export"] == "standalone"


def test_cli_model_config_uses_explicit_standalone_export_path(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"
    model_output.mkdir()
    export_output = tmp_path / "portable"
    calls: dict[str, object] = {}

    class FakeModelConfig:
        output_dir = model_output

    def fake_run(path: str) -> dict:
        return {"configured_model": FakeModelConfig(), "comparison": None}

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        calls["export"] = (Path(model_output_dir), Path(output_dir), force, debug_files)
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path), "--export-standalone", str(export_output)])

    assert calls["export"] == (model_output, export_output, True, False)


def test_cli_model_config_propagates_explicit_standalone_export_failure(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"

    class FakeModelConfig:
        output_dir = model_output

    def fake_run(path: str) -> dict:
        return {"configured_model": FakeModelConfig(), "comparison": None}

    def fake_export(*_args, **_kwargs):
        raise RuntimeError("missing standalone metadata")

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    with pytest.raises(RuntimeError, match="missing standalone metadata"):
        cli.main(["model", "--config", str(cfg_path), "--export-standalone", str(tmp_path / "portable")])


def test_cli_model_export_standalone_all_examples_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod

    calls: dict[str, object] = {}

    def fake_export_all(examples_root, output_root, *, force=False, debug_files=False, dry_run=False):
        calls["export_all"] = (examples_root, output_root, force, debug_files, dry_run)
        return {
            "exportable": [{"model_output_dir": "a"}],
            "blocked": [{"model_output_dir": "b", "reason": "missing metadata"}],
            "exported": [],
        }

    monkeypatch.setattr(export_mod, "export_all_standalone_models", fake_export_all)

    cli.main(["model", "export-standalone", "--all-examples", "examples", str(tmp_path / "out"), "--dry-run"])

    assert calls["export_all"] == ("examples", str(tmp_path / "out"), False, False, True)
    captured = capsys.readouterr()
    assert "exportable" in captured.out
    assert "blocked" in captured.out
