import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse

import tapw.workflows.symmetry as symmetry_analysis


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None


def _make_runner(tmp_path: Path):
    config = SimpleNamespace(
        paths=SimpleNamespace(output_dir=str(tmp_path / "run")),
        symmetry_analysis=SimpleNamespace(
            output_dir="symmetry_analysis",
            tolerance=1.0e-2,
            valleys=[5],
            debug=False,
            enable=True,
        ),
    )
    return symmetry_analysis.SymmetryAnalysisRunner(
        config=config,
        structure=None,
        hr_supercell=None,
        sr_supercell=None,
        logger=_FakeLogger(),
    )


def test_runner_writes_summary_markdown_json_and_details_csv(tmp_path):
    runner = _make_runner(tmp_path)

    payload = {
        "summary": {
            "valleys": [5],
            "tolerance": 1.0e-2,
            "operations": {
                "Gamma": [
                    {
                        "operation": "C3z",
                        "supported": True,
                        "status": "approximate/provisional",
                    }
                ]
            },
        },
        "details": [
            {
                "valley": "Gamma",
                "operation": "C3z",
                "operation_type": "unitary",
                "spglib_index": 12,
                "R_2d": "[[0.0, -1.0], [1.0, 0.0]]",
                "k_label": "Gamma",
                "k_frac": "[0.0, 0.0, 0.0]",
                "supported": True,
                "not_supported_reason": "",
                "residual_H_raw": 1.0e-4,
                "residual_S_raw": 2.0e-4,
                "residual_H_sym": 1.0e-8,
                "residual_S_sym": 2.0e-8,
                "residual_lowdin_order": 3.0e-4,
                "g_perm_max_delta": 1.0e-10,
                "nonzero_reciprocal_shift_count": 0,
                "square_residual": 1.0e-9,
                "status": "approximate/provisional",
                "debug": {
                    "R_3d": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_frac": [0.0, 0.0, 0.0],
                    "transport_unitarity_residual": 1.0e-12,
                },
            },
            {
                "valley": "Gamma",
                "operation": "C2T",
                "operation_type": "antiunitary",
                "spglib_index": "",
                "R_2d": "",
                "k_label": "q1",
                "k_frac": "[0.17, -0.11, 0.0]",
                "supported": False,
                "not_supported_reason": "antiunitary_not_order2",
                "residual_H_raw": None,
                "residual_S_raw": None,
                "residual_H_sym": None,
                "residual_S_sym": None,
                "residual_lowdin_order": None,
                "g_perm_max_delta": None,
                "nonzero_reciprocal_shift_count": None,
                "square_residual": None,
                "status": "not_supported",
            },
        ],
    }

    runner._analyze = MethodType(lambda self: payload, runner)

    result = runner.run()

    output_dir = Path(runner.output_dir)
    assert result == payload
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "details.csv").is_file()

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["valleys"] == [5]

    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Valleys analyzed" in summary_md
    assert "Recommendation" not in summary_md
    assert "not_evaluated_lowdin_disabled" not in summary_md

    details = pd.read_csv(output_dir / "details.csv")
    assert list(details.columns) == [
        "valley",
        "operation",
        "spglib_index",
        "R_2d",
        "k_label",
        "residual_H_raw",
        "status",
        "not_supported_reason",
        "g_perm_max_delta",
        "nonzero_reciprocal_shift_count",
        "square_residual",
    ]
    assert "supported" not in details.columns
    assert "transport_unitarity_residual" not in details.columns
    assert "not_supported_reason" in details.columns
    assert details.loc[0, "spglib_index"] == 12
    assert details.loc[0, "g_perm_max_delta"] == 1.0e-10
    assert details.loc[0, "square_residual"] == 1.0e-9
    assert details.loc[1, "not_supported_reason"] == "antiunitary_not_order2"
    assert not (output_dir / "debug.json").exists()


def test_runner_writes_minimal_representation_matrices_and_manifest(tmp_path):
    runner = _make_runner(tmp_path)
    matrix = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    pin_matrix = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    pg_matrix = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    raw_h_matrix = matrix @ pg_matrix
    payload = {
        "summary": {
            "valleys": [1],
            "tolerance": 1.0e-2,
            "operations": {"K1": [{"operation": "C2T", "supported": True, "status": "exact"}]},
        },
        "details": [],
        "representations": [
            {
                "valley": 1,
                "valley_label": "K1",
                "operation": "C2T",
                "antiunitary": True,
                "spglib_index": 17,
                "axis_angle_deg": 3.7,
                "basis_hash": "basis-test-hash",
                "residual_H_raw": 1.0e-5,
                "status": "approximate/provisional",
                "g_perm_max_delta": 2.0e-12,
                "nonzero_reciprocal_shift_count": 0,
                "square_residual": 4.0e-9,
                "matrix": matrix,
                "matrix_role": "D_g^(0)",
                "pin_supported": True,
                "pin_matrix": pin_matrix,
                "pin_reason": "",
                "pin_shift_by_group_coeffs": {"0": [0, 0], "1": [0, 0]},
                "source_form": "ordinary_with_pin",
                "ld_source_rule": "q_lambda = R_eff^{-1}(k + K_target(lambda)) - K_lambda",
                "pg_matrix": pg_matrix,
                "pg_shift_by_group_coeffs": {"0": [0, 0], "1": [0, 0]},
                "pg_phase_convention": "P_G^atomic=diag(exp(-i DeltaG_group dot r_atom)); P_G^TAPW=g P_G^atomic g^dagger",
                "raw_h_matrix": raw_h_matrix,
                "raw_h_action_rule": "unitary: D_raw H D_raw^dagger; antiunitary: D_raw H^* D_raw^dagger",
            }
        ],
    }
    runner._analyze = MethodType(lambda self: payload, runner)

    runner.run()

    output_dir = Path(runner.output_dir)
    matrix_path = output_dir / "representations" / "K1" / "C2T.npz"
    pin_path = output_dir / "representations" / "K1" / "C2T_Pin.npz"
    pg_path = output_dir / "representations" / "K1" / "C2T_PG.npz"
    raw_h_path = output_dir / "representations" / "K1" / "C2T_rawH.npz"
    manifest_path = output_dir / "representations" / "manifest.json"
    assert matrix_path.is_file()
    assert pin_path.is_file()
    assert pg_path.is_file()
    assert raw_h_path.is_file()
    assert manifest_path.is_file()
    loaded = scipy.sparse.load_npz(matrix_path)
    loaded_pin = scipy.sparse.load_npz(pin_path)
    loaded_pg = scipy.sparse.load_npz(pg_path)
    loaded_raw_h = scipy.sparse.load_npz(raw_h_path)
    assert scipy.sparse.isspmatrix_csr(loaded)
    assert scipy.sparse.isspmatrix_csr(loaded_pin)
    assert scipy.sparse.isspmatrix_csr(loaded_pg)
    assert scipy.sparse.isspmatrix_csr(loaded_raw_h)
    assert np.allclose(loaded.toarray(), matrix.toarray())
    assert np.allclose(loaded_pin.toarray(), pin_matrix.toarray())
    assert np.allclose(loaded_pg.toarray(), pg_matrix.toarray())
    assert np.allclose(loaded_raw_h.toarray(), raw_h_matrix.toarray())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["basis"] == "tapw_projected"
    assert manifest["basis_order"] == "spin_outermost; group -> g_index -> atom_type -> orbital"
    assert manifest["matrices"] == [
        {
            "valley": 1,
            "valley_label": "K1",
            "operation": "C2T",
            "file": "K1/C2T.npz",
            "matrix_role": "D_g^(0)",
            "antiunitary": True,
            "spglib_index": 17,
            "axis_deg": 3.7,
            "shape": [2, 2],
            "nnz": 2,
            "dtype": "complex128",
            "pin_supported": True,
            "pin_file": "K1/C2T_Pin.npz",
            "pin_reason": "",
            "pin_shape": [2, 2],
            "pin_nnz": 2,
            "pin_shift_by_group_coeffs": {"0": [0, 0], "1": [0, 0]},
            "pg_file": "K1/C2T_PG.npz",
            "pg_reason": "",
            "pg_shape": [2, 2],
            "pg_nnz": 2,
            "pg_shift_by_group_coeffs": {"0": [0, 0], "1": [0, 0]},
            "pg_phase_convention": "P_G^atomic=diag(exp(-i DeltaG_group dot r_atom)); P_G^TAPW=g P_G^atomic g^dagger",
            "raw_h_operator_file": "K1/C2T_rawH.npz",
            "raw_h_operator_shape": [2, 2],
            "raw_h_operator_nnz": 2,
            "raw_h_action_rule": "unitary: D_raw H D_raw^dagger; antiunitary: D_raw H^* D_raw^dagger",
            "source_form": "ordinary_with_pin",
            "ld_source_rule": "q_lambda = R_eff^{-1}(k + K_target(lambda)) - K_lambda",
            "basis_hash": "basis-test-hash",
            "residual_H_raw": 1.0e-5,
            "status": "approximate/provisional",
            "g_perm_max_delta": 2.0e-12,
            "nonzero_reciprocal_shift_count": 0,
            "square_residual": 4.0e-9,
        }
    ]


def test_analyze_collects_only_minimal_supported_representation_generators(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    runner.config.compute = SimpleNamespace(TAPW=True, valleys=[5])
    runner.config.twist = SimpleNamespace(bravais="hex")
    runner.structure = SimpleNamespace(spin=False, reciprocal_Tmat=np.eye(3))
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=5,
        valley_label="Gamma",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.zeros((1, 2))},
        moire_reciprocal_basis=np.eye(2),
        calculator=SimpleNamespace(TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr"))),
    )
    runner._calculator_for_valley = MethodType(
        lambda self, valley: SimpleNamespace(valley_flag="Gamma", TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr"))),
        runner,
    )
    runner._valley_context_for_valley = MethodType(lambda self, valley: valley_ctx, runner)
    monkeypatch.setattr(symmetry_analysis, "collect_spglib_spatial_operations", lambda structure: [])
    monkeypatch.setattr(
        symmetry_analysis,
        "_minimal_symmetry_candidates_for_valley",
        lambda valley_ctx, bravais, spatial_operations=None, structure=None: [
            {"index": 0, "name": "E", "antiunitary": False, "closed": True, "rotation_cart": np.eye(3)},
            {"index": 2, "name": "C3z", "antiunitary": False, "closed": True, "rotation_cart": np.eye(3)},
            {"index": 3, "name": "C3z^2", "antiunitary": False, "closed": True, "rotation_cart": np.eye(3)},
        ],
    )
    monkeypatch.setattr(symmetry_analysis, "_default_validation_q_points", lambda: [("Gamma", np.zeros(3))])

    runner._candidate_rows_for_q = MethodType(
        lambda self, candidate, valley, valley_label, q_label, q_target, tolerance: {
            "valley": valley_label,
            "operation": symmetry_analysis.displayed_operation_name(candidate["name"]),
            "k_label": q_label,
            "supported": True,
            "status": "exact",
            "not_supported_reason": "",
            "residual_H_raw": 0.0,
        },
        runner,
    )
    runner._representation_record_for_candidate = MethodType(
        lambda self, candidate, valley, valley_label, valley_ctx, candidate_rows: {
            "valley": valley,
            "valley_label": valley_label,
            "operation": symmetry_analysis.representation_operation_name(candidate["name"]),
            "antiunitary": bool(candidate.get("antiunitary", False)),
            "matrix": scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
        },
        runner,
    )

    payload = runner._analyze()

    assert [record["operation"] for record in payload["representations"]] == ["C3"]


def test_runner_writes_debug_json_only_when_debug_enabled(tmp_path):
    runner = _make_runner(tmp_path)
    runner.config.symmetry_analysis.debug = True
    payload = {
        "summary": {
            "valleys": [31],
            "tolerance": 1.0e-2,
            "operations": {"M1": []},
        },
        "details": [
            {
                "valley": "M1",
                "operation": "M_C2_eta",
                "spglib_index": 3,
                "R_2d": "[[1.0, 0.0], [0.0, -1.0]]",
                "k_label": "q1",
                "residual_H_raw": 1.0e-3,
                "status": "approximate/provisional",
                "not_supported_reason": "",
                "g_perm_max_delta": 1.0e-10,
                "nonzero_reciprocal_shift_count": 2,
                "square_residual": 7.0e-6,
                "debug": {
                    "R_3d": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                    "translation_frac": [0.0, 0.0, 0.36],
                    "center_convention": "group_k_centers",
                    "target_group_shift_coeffs": {"0": [-5, -4], "1": [-4, -5]},
                    "phase_sign": "+1",
                    "phase_side": "left_target_rows",
                    "transport_unitarity_residual": 2.0e-10,
                },
            }
        ],
    }
    runner._analyze = MethodType(lambda self: payload, runner)

    runner.run()

    output_dir = Path(runner.output_dir)
    debug = json.loads((output_dir / "debug.json").read_text(encoding="utf-8"))
    assert debug["details"][0]["debug"]["center_convention"] == "group_k_centers"
    assert debug["details"][0]["debug"]["phase_side"] == "left_target_rows"


def test_summary_markdown_documents_m_eta_convention(tmp_path):
    runner = _make_runner(tmp_path)
    lines = runner._summary_markdown_lines(
        {
            "valleys": [31],
            "tolerance": 1.0e-2,
            "operations": {},
            "valley_results": {},
        }
    )
    text = "\n".join(lines)
    assert "eta labels the three C3-related M valleys" in text
    assert "C2 denotes C3^eta C2 C3^{-eta}" in text
    assert "M_C2_eta" not in text


def test_summary_markdown_uses_c2_display_name_axis_angle_and_generators(tmp_path):
    runner = _make_runner(tmp_path)
    lines = runner._summary_markdown_lines(
        {
            "valleys": [1, 5],
            "tolerance": 1.0e-2,
            "operations": {
                "Gamma": [
                    {
                        "operation": "C2",
                        "axis_angle_deg": 30.0,
                        "supported": True,
                        "status": "approximate/provisional",
                    },
                    {
                        "operation": "C3z",
                        "supported": True,
                        "status": "approximate/provisional",
                    },
                ],
                "K1": [
                    {
                        "operation": "C2T",
                        "axis_angle_deg": 60.0,
                        "supported": True,
                        "status": "approximate/provisional",
                    }
                ],
            },
            "valley_results": {},
            "minimal_generators": {
                "Gamma": ["C3z", "C2"],
                "K1": ["C2T"],
            },
        }
    )
    text = "\n".join(lines)
    assert "K_C2T" not in text
    assert "C2 (axis 30deg)" in text
    assert "C2T (axis 60deg)" in text
    assert "Minimal generators:" in text
    assert "- Gamma: C3z, C2" in text
    assert "- K1: C2T" in text


def test_displayed_operation_name_maps_internal_effective_names_to_family_names():
    assert symmetry_analysis.displayed_operation_name("C2") == "C2"
    assert symmetry_analysis.displayed_operation_name("K_C2T") == "C2T"
    assert symmetry_analysis.displayed_operation_name("M_C2_eta") == "C2"


def test_default_validation_points_are_gamma_plus_one_generic_point():
    points = symmetry_analysis._default_validation_q_points()

    assert [label for label, _ in points] == ["Gamma", "q1"]


def test_candidate_row_reports_raw_h_only_without_s_or_symmetrized_residuals(tmp_path):
    runner = _make_runner(tmp_path)
    runner.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    h0 = np.array([[2.0, 0.25], [0.25, 1.0]], dtype=np.complex128)
    calls = []
    runner._last_transport_diagnostics = {
        "g_perm_max_delta": 123.0,
        "nonzero_reciprocal_shift_count": 99,
        "t_square_residual": 456.0,
    }

    runner._map_inverse_q = MethodType(lambda self, candidate, q_target: np.asarray(q_target, dtype=float), runner)
    runner._build_transport = MethodType(lambda self, candidate, valley, q_target, q_source: np.eye(2, dtype=np.complex128), runner)

    def _fake_raw_projected_hs(self, valley, q_local):
        calls.append(tuple(np.asarray(q_local, dtype=float)))
        return h0, None

    runner._raw_projected_hs = MethodType(_fake_raw_projected_hs, runner)

    row = runner._candidate_rows_for_q(
        {
            "index": 0,
            "name": "E",
            "antiunitary": False,
            "closed": True,
            "rotation_cart": np.eye(3),
            "translation_cart": np.zeros(3),
        },
        valley=5,
        valley_label="Gamma",
        q_label="Gamma",
        q_target=np.zeros(3),
        tolerance=1.0e-2,
    )

    assert row["residual_H_raw"] == 0.0
    assert row["residual_S_raw"] is None
    assert row["residual_H_sym"] is None
    assert row["residual_S_sym"] is None
    assert row["residual_lowdin_order"] is None
    assert row["g_perm_max_delta"] is None
    assert row["nonzero_reciprocal_shift_count"] is None
    assert row["square_residual"] is None
    assert len(calls) == 2


def test_candidate_row_uses_legacy_c3_orbit_for_c3_covariance(tmp_path):
    runner = _make_runner(tmp_path)
    runner.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    transport = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    h_target = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
    h_source = transport.conj().T @ h_target @ transport
    h_source_2 = transport @ h_target @ transport.conj().T

    runner._build_transport = MethodType(lambda self, candidate, valley, q_target, q_source: transport, runner)
    runner._calculator_for_valley = MethodType(
        lambda self, valley: SimpleNamespace(
            Getk_super_gauge_sparse_symm=lambda hr, q, force_sparse_dot=False: [h_target, h_source, h_source_2]
        ),
        runner,
    )
    runner._raw_projected_hs = MethodType(
        lambda self, valley, q_local: (_ for _ in ()).throw(AssertionError("C3 must use the legacy C3 orbit")),
        runner,
    )

    row = runner._candidate_rows_for_q(
        {
            "index": 2,
            "name": "C3z",
            "antiunitary": False,
            "closed": True,
            "rotation_cart": np.eye(3),
            "translation_cart": np.zeros(3),
        },
        valley=1,
        valley_label="K1",
        q_label="Gamma",
        q_target=np.zeros(3),
        tolerance=1.0e-2,
    )

    assert row["residual_H_raw"] == 0.0
    assert row["status"] == "exact"


def test_source_pin_matrix_is_identity_for_zero_source_shift():
    structure = SimpleNamespace(
        spin=False,
        df=pd.DataFrame(
            {
                "twist_group": [0],
                "atom_type": [0],
                "orb_num": [1],
            }
        ),
    )
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=5,
        valley_label="Gamma",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.array([[0.0, 0.0], [1.0, 0.0]])},
        moire_reciprocal_basis=np.eye(2),
        calculator=None,
    )

    pin, rule, diagnostics = symmetry_analysis.build_source_pin_matrix_for_candidate(
        structure,
        valley_ctx,
        {"name": "E", "antiunitary": False, "rotation_cart": np.eye(3)},
    )

    assert diagnostics["pin_supported"] is True
    assert rule["source_form"] == "ordinary_with_pin"
    assert np.allclose(pin.toarray(), np.eye(2))


def test_source_pin_matrix_marks_nonclosed_truncated_g_basis_as_ld_source():
    theta = np.deg2rad(120.0)
    rotation = np.eye(3)
    rotation[:2, :2] = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    structure = SimpleNamespace(
        spin=False,
        df=pd.DataFrame(
            {
                "twist_group": [0],
                "atom_type": [0],
                "orb_num": [1],
            }
        ),
    )
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=1,
        valley_label="K1",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.array([1.0, 0.0])},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.array([[0.0, 0.0]])},
        moire_reciprocal_basis=np.eye(2),
        calculator=None,
    )

    pin, rule, diagnostics = symmetry_analysis.build_source_pin_matrix_for_candidate(
        structure,
        valley_ctx,
        {"name": "C3z", "antiunitary": False, "rotation_cart": rotation},
    )

    assert scipy.sparse.isspmatrix_csr(pin)
    assert pin.shape == (1, 1)
    assert pin.nnz == 0
    assert diagnostics["pin_supported"] is False
    assert diagnostics["pin_reason"] == "pin_not_closed_in_truncated_g_basis"
    assert rule["source_form"] == "ld_source_rule"


def test_periodic_gauge_matrix_is_identity_for_zero_source_shift():
    structure = SimpleNamespace(
        spin=False,
        df=pd.DataFrame(
            {
                "twist_group": [0],
                "atom_type": [0],
                "orb_num": [1],
                "x": [0.25],
                "y": [0.5],
            }
        ),
    )
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=5,
        valley_label="Gamma",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.array([[0.0, 0.0], [1.0, 0.0]])},
        moire_reciprocal_basis=np.eye(2),
        calculator=SimpleNamespace(TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr"))),
    )

    pg, rule, diagnostics = symmetry_analysis.build_periodic_gauge_matrix_for_candidate(
        structure,
        valley_ctx,
        {"name": "E", "antiunitary": False, "rotation_cart": np.eye(3)},
    )

    assert diagnostics["pg_identity"] is True
    assert rule["pg_shift_by_group_coeffs"] == {"0": [0, 0]}
    assert np.allclose(pg.toarray(), np.eye(2))


def test_periodic_gauge_matrix_uses_atomic_bloch_phase_for_nonzero_source_shift():
    rotation = np.eye(3)
    rotation[:2, :2] = -np.eye(2)
    structure = SimpleNamespace(
        spin=False,
        df=pd.DataFrame(
            {
                "twist_group": [0],
                "atom_type": [0],
                "orb_num": [1],
                "x": [0.25],
                "y": [0.0],
            }
        ),
    )
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=1,
        valley_label="K1",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.array([1.0, 0.0])},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.array([[0.0, 0.0]])},
        moire_reciprocal_basis=np.eye(2),
        calculator=SimpleNamespace(TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(1, format="csr"))),
    )

    pg, rule, diagnostics = symmetry_analysis.build_periodic_gauge_matrix_for_candidate(
        structure,
        valley_ctx,
        {"name": "C3z", "antiunitary": False, "rotation_cart": rotation},
    )

    expected = np.exp(-1.0j * (-2.0) * 0.25)
    assert diagnostics["pg_identity"] is False
    assert rule["pg_shift_by_group_coeffs"] == {"0": [-2, 0]}
    assert np.allclose(pg.toarray(), [[expected]])


def test_source_side_periodic_gauge_recovers_target_phase_transport():
    pure = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    target_phase = scipy.sparse.diags(
        [np.exp(0.2j), np.exp(-0.3j)],
        offsets=0,
        dtype=np.complex128,
        format="csr",
    )
    raw_transport = (target_phase @ pure).tocsr()

    pg_unitary = symmetry_analysis.source_side_periodic_gauge_from_raw_transport(
        pure,
        raw_transport,
        antiunitary=False,
    )
    assert np.allclose((pure @ pg_unitary).toarray(), raw_transport.toarray())

    pg_antiunitary = symmetry_analysis.source_side_periodic_gauge_from_raw_transport(
        pure,
        raw_transport,
        antiunitary=True,
    )
    assert np.allclose((pure @ pg_antiunitary.conj()).toarray(), raw_transport.toarray())
