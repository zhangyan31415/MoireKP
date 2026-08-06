import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import tapw.workflows.symmetry as symmetry_analysis


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None


def _make_runner(tmp_path: Path):
    h_path = tmp_path / "H_symm.npz"
    s_path = tmp_path / "S_symm.npz"
    structure_path = tmp_path / "openmx.dat"
    h_path.write_bytes(b"H-source-a")
    s_path.write_bytes(b"S-source-a")
    structure_path.write_bytes(b"structure-source-a")
    config = SimpleNamespace(
        paths=SimpleNamespace(
            output_dir=str(tmp_path / "run"),
            H_file=str(h_path),
            S_file=str(s_path),
            input_file=str(structure_path),
        ),
        output_layout=None,
        twist=SimpleNamespace(spin=True),
        compute=SimpleNamespace(n_g=6, valleys=[1]),
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


def _minimal_rawh_payload(operation: str = "C2T"):
    matrix = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    pg_matrix = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    return {
        "summary": {
            "valleys": [1],
            "tolerance": 1.0e-2,
            "operations": {"K1": [{"operation": operation, "supported": True, "status": "exact"}]},
        },
        "details": [],
        "representations": [
            {
                "valley": 1,
                "valley_label": "K1",
                "operation": operation,
                "antiunitary": operation in {"TR", "C2T"},
                "spglib_index": 17,
                "axis_angle_deg": 3.7,
                "basis_hash": "basis-test-hash",
                "residual_H_raw": 1.0e-5,
                "residual_S_raw": None,
                "production_validated": True,
                "status": "passed",
                "supported": True,
                "role": "internal",
                "g_perm_max_delta": 2.0e-12,
                "nonzero_reciprocal_shift_count": 0,
                "square_residual": 4.0e-9,
                "matrix": matrix,
                "matrix_role": "D_g^(0)",
                "pin_supported": True,
                "pin_matrix": pg_matrix,
                "pin_reason": "",
                "pin_shift_by_group_coeffs": {"0": [0, 0], "1": [0, 0]},
                "source_form": "ordinary_with_pin",
                "ld_source_rule": "q_lambda = R_eff^{-1}(k + K_target(lambda)) - K_lambda",
                "pg_matrix": pg_matrix,
                "pg_shift_by_group_coeffs": {"0": [0, 0], "1": [0, 0]},
                "pg_phase_convention": "P_G^atomic=diag(exp(-i DeltaG_group dot r_atom)); P_G^TAPW=g P_G^atomic g^dagger",
                "raw_h_matrix": matrix @ pg_matrix,
                "raw_h_action_rule": "unitary: D_raw H D_raw^dagger; antiunitary: D_raw H^* D_raw^dagger",
            }
        ],
    }


def test_c3_g_transport_accepts_moire_q_roundoff():
    angles = np.deg2rad(np.arange(0, 360, 60))
    g_vectors = np.array(
        [[0.0, 0.0], *[[np.cos(angle), np.sin(angle)] for angle in angles]],
        dtype=float,
    )
    g_target = g_vectors.copy()
    g_target[2] += np.array([5.0e-8, -4.0e-8])

    transport = symmetry_analysis.build_c3_g_transport_from_tapw_convention(
        g_source=g_vectors,
        g_target=g_target,
        layer_center=np.zeros(2),
        angle_deg=120.0,
    )

    assert transport.shape == (7, 7)
    assert transport.nnz == 7


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


def test_runner_writes_release_rawh_representation_and_manifest_by_default(tmp_path):
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
    assert not matrix_path.exists()
    assert not pin_path.exists()
    assert not pg_path.exists()
    assert not (output_dir / "diagnostics").exists()
    assert raw_h_path.is_file()
    assert manifest_path.is_file()
    loaded_raw_h = scipy.sparse.load_npz(raw_h_path)
    assert scipy.sparse.isspmatrix_csr(loaded_raw_h)
    assert np.allclose(loaded_raw_h.toarray(), raw_h_matrix.toarray())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["output_schema"] == "tapw_source_symmetry/v2"
    assert manifest["schema_version"] == 2
    assert manifest["basis"] == "tapw_projected"
    assert manifest["basis_order"] == "spin_outermost; group -> g_index -> atom_type -> orbital"
    assert manifest["operation_name_convention"]["time_reversal"] == "TR"
    assert manifest["operation_name_convention"]["legacy_input_aliases"] == {"T": "TR"}
    assert manifest["source_action_definition"]["required_fields"] == [
        "k_map",
        "q_map",
        "sector_map",
        "source_action",
    ]
    assert manifest["matrices"] == [
        {
            "valley": 1,
            "valley_label": "K1",
            "operation": "C2T",
            "antiunitary": True,
            "spglib_index": 17,
            "axis_deg": 3.7,
            "dtype": "complex128",
            "pin_supported": True,
            "pin_reason": "",
            "pin_shift_by_group_coeffs": {"0": [0, 0], "1": [0, 0]},
            "pg_reason": "",
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
            "source_valley": "K1",
            "target_valley": "K1",
            "closed_in_active_set": True,
            "role": "internal",
            "supported": True,
            "k_pairs": [[0, 0]],
            "k_pair_source": "symmetry_analysis_reference_k_index",
            "source_matrix_role": "raw_h_action",
            "source_gauge": "tapw_raw_hamiltonian",
            "target_role": "kp_source_action",
            "matrix_kind": "action",
            "conventions": {
                "antiunitary_convention": "U_K",
                "gauge_correction": {"kind": "none"},
                "source_action_frame": "tapw_source",
            },
            "k_map": {
                "type": "reflection",
                "axis_deg": 93.7,
                "reflection_axis_convention": "mirror_axis_deg",
            },
            "q_map": {
                "type": "reflection",
                "axis_deg": 93.7,
                "reflection_axis_convention": "mirror_axis_deg",
            },
            "sector_map": "layer_exchange",
            "source_action": {
                "antiunitary": True,
                "k_map": {
                    "type": "reflection",
                    "axis_deg": 93.7,
                    "reflection_axis_convention": "mirror_axis_deg",
                },
                "q_map": {
                    "type": "reflection",
                    "axis_deg": 93.7,
                    "reflection_axis_convention": "mirror_axis_deg",
                },
                "sector_map": "layer_exchange",
                "spin_map": "from_tapw_source",
                "valley_map": "identity",
            },
        }
    ]
    serialized_outputs = "\n".join(
        [
            (output_dir / "summary.json").read_text(encoding="utf-8"),
            (output_dir / "details.csv").read_text(encoding="utf-8"),
            manifest_path.read_text(encoding="utf-8"),
        ]
    )
    assert "K" + "_C2T" not in serialized_outputs
    assert "M" + "_C2_eta" not in serialized_outputs


def test_runner_writes_canonical_symmetry_two_file_layout_with_packed_metadata(tmp_path):
    runner = _make_runner(tmp_path)
    runner.config.output_layout = SimpleNamespace(
        style="canonical_v1",
        root=str(tmp_path / "outputs"),
        profile=None,
        q_shell="q06",
    )
    runner.config.symmetry_analysis.developer_outputs = True
    runner.output_dir = str(tmp_path / "outputs" / "K1" / "q06" / "symmetry")
    runner._analyze = MethodType(lambda self: _minimal_rawh_payload("C2T"), runner)

    runner.run()

    output_dir = Path(runner.output_dir)
    packed_path = output_dir / "representations.npz"
    residuals_path = output_dir / "residuals.csv"
    summary_path = output_dir / "summary.md"
    assert packed_path.is_file()
    assert not residuals_path.exists()
    assert summary_path.is_file()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "representations.npz",
        "summary.md",
    ]
    with np.load(packed_path, allow_pickle=False) as payload:
        assert payload["C2T_shape"].tolist() == [2, 2]
        assert payload["C2T_data"].dtype == np.complex128
        assert "C2T_indices" in payload.files
        assert "C2T_indptr" in payload.files
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["schema"] == "tapw.raw_h_representations.v1"
    assert metadata["identity_schema"] == "moirekp.artifact-identity.v1"
    assert len(metadata["input_hash"]) == 64
    assert len(metadata["config_hash"]) == 64
    assert metadata["basis_hash"] == "basis-test-hash"
    assert metadata["package_version"] == "0.1.0"
    assert metadata["schema_version"] == 1
    assert metadata["storage"] == "scipy_csr_components_v1"
    assert metadata["basis_order"] == "spin_outermost; group -> g_index -> atom_type -> orbital"
    assert metadata["matrices"][0]["key"] == "C2T"
    assert metadata["matrices"][0]["operation"] == "C2T"
    assert metadata["matrices"][0]["source_action"]["antiunitary"] is True
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Use `representations.npz` for release machine inputs; it contains CSR matrices and `metadata_json`." in summary_text
    assert "| K1 -> K1 | C2T | representations.npz:C2T | raw_h_action |" in summary_text
    assert "representations/K1/C2T_rawH.npz" not in summary_text


@pytest.mark.parametrize("source_name", ["H_file", "S_file", "input_file"])
def test_runner_source_identity_changes_with_each_input_file(tmp_path, source_name):
    runner = _make_runner(tmp_path)
    before = symmetry_analysis._source_input_hash(runner.config)
    source_path = Path(getattr(runner.config.paths, source_name))
    source_path.write_bytes(source_path.read_bytes() + b"-changed")

    after = symmetry_analysis._source_input_hash(runner.config)

    assert after != before


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("compute", "n_g", 7),
        ("twist", "spin", False),
        ("compute", "Electric_field_in_eVpA", 0.02),
    ],
)
def test_runner_config_identity_changes_with_physics_config(tmp_path, section, field, value):
    runner = _make_runner(tmp_path)
    before = symmetry_analysis._physics_config_hash(runner.config)
    setattr(getattr(runner.config, section), field, value)

    after = symmetry_analysis._physics_config_hash(runner.config)

    assert after != before


def test_runner_writes_developer_representation_matrices_under_diagnostics(tmp_path):
    runner = _make_runner(tmp_path)
    runner.config.symmetry_analysis.developer_outputs = True
    matrix = scipy.sparse.csr_matrix(np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128))
    pin_matrix = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    pg_matrix = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    raw_h_matrix = matrix @ pg_matrix
    payload = {
        "summary": {"valleys": [1], "tolerance": 1.0e-2, "operations": {"K1": []}},
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
    assert (output_dir / "representations" / "K1" / "C2T_rawH.npz").is_file()
    diag_dir = output_dir / "representations" / "diagnostics" / "K1"
    assert (diag_dir / "C2T.npz").is_file()
    assert (diag_dir / "C2T_Pin.npz").is_file()
    assert (diag_dir / "C2T_PG.npz").is_file()
    assert not (diag_dir / "C2T_rawH.npz").exists()

    manifest = json.loads((output_dir / "representations" / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["matrices"][0]
    assert row["raw_h_operator_file"] == "K1/C2T_rawH.npz"
    assert row["developer_outputs"] == {
        "file": "diagnostics/K1/C2T.npz",
        "pin_file": "diagnostics/K1/C2T_Pin.npz",
        "pg_file": "diagnostics/K1/C2T_PG.npz",
    }

    runner.config.symmetry_analysis.developer_outputs = False
    runner.run()

    assert not (diag_dir / "C2T.npz").exists()
    assert not (diag_dir / "C2T_Pin.npz").exists()
    assert not (diag_dir / "C2T_PG.npz").exists()


def test_runner_does_not_write_manifest_row_without_raw_h_matrix(tmp_path):
    runner = _make_runner(tmp_path)
    matrix = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    payload = {
        "summary": {"valleys": [1], "tolerance": 1.0e-2, "operations": {"K1": []}},
        "details": [],
        "representations": [
            {
                "valley": 1,
                "valley_label": "K1",
                "operation": "T",
                "antiunitary": True,
                "matrix": matrix,
                "matrix_role": "D_g^(0)",
                "export_raw_h_matrix": False,
            }
        ],
    }
    runner._analyze = MethodType(lambda self: payload, runner)

    runner.run()

    manifest = json.loads((Path(runner.output_dir) / "representations" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["matrices"] == []
    assert "T_rawH.npz" not in (Path(runner.output_dir) / "summary.md").read_text(encoding="utf-8")


def test_runner_canonicalizes_time_reversal_outputs_to_tr(tmp_path):
    runner = _make_runner(tmp_path)
    matrix = scipy.sparse.identity(2, dtype=np.complex128, format="csr")
    payload = {
        "summary": {
            "valleys": [31],
            "tolerance": 1.0e-2,
            "operations": {
                "M1": [
                    {
                        "operation": "T",
                        "antiunitary": True,
                        "supported": True,
                        "status": "exact",
                        "source_valley": "M1",
                        "target_valley": "M1",
                        "closed_in_active_set": True,
                        "role": "internal",
                        "export_raw_h_matrix": True,
                    }
                ]
            },
            "minimal_generators": {"M1": ["T"]},
        },
        "details": [],
        "representations": [
            {
                "valley": 31,
                "valley_label": "M1",
                "operation": "T",
                "antiunitary": True,
                "matrix": matrix,
                "matrix_role": "raw_h_action",
                "raw_h_matrix": matrix,
            }
        ],
    }
    runner._analyze = MethodType(lambda self: payload, runner)

    runner.run()

    output_dir = Path(runner.output_dir)
    assert (output_dir / "representations" / "M1" / "TR_rawH.npz").is_file()
    assert not (output_dir / "representations" / "M1" / "T_rawH.npz").exists()
    manifest = json.loads((output_dir / "representations" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["matrices"][0]["operation"] == "TR"
    assert manifest["matrices"][0]["raw_h_operator_file"] == "M1/TR_rawH.npz"
    text_outputs = "\n".join(
        [
            (output_dir / "summary.md").read_text(encoding="utf-8"),
            (output_dir / "summary.json").read_text(encoding="utf-8"),
            (output_dir / "details.csv").read_text(encoding="utf-8"),
            (output_dir / "representations" / "manifest.json").read_text(encoding="utf-8"),
        ]
    )
    assert "T_rawH.npz" not in text_outputs
    assert "| TR | built-in | yes | M1 | M1 | yes | yes | exact | yes | internal |  |" in text_outputs


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
    monkeypatch.setattr(symmetry_analysis, "collect_spglib_spatial_operations", lambda structure, **_kwargs: [])
    monkeypatch.setattr(
        symmetry_analysis,
        "_minimal_symmetry_candidates_for_valley",
        lambda valley_ctx, bravais, spatial_operations=None, structure=None, **_kwargs: [
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

    assert [record["operation"] for record in payload["representations"]] == ["C3z"]
    export_flags = {
        entry["operation"]: entry["export_raw_h_matrix"]
        for entry in payload["summary"]["operations"]["Gamma"]
    }
    assert export_flags == {"E": False, "C3z": True, "C3z^2": False}


def test_analyze_derives_c3_square_from_supported_c3_without_raw_validation(tmp_path, monkeypatch):
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
        lambda self, valley: SimpleNamespace(
            valley_flag="Gamma",
            TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr")),
        ),
        runner,
    )
    runner._valley_context_for_valley = MethodType(lambda self, valley: valley_ctx, runner)
    monkeypatch.setattr(symmetry_analysis, "collect_spglib_spatial_operations", lambda structure, **_kwargs: [])
    monkeypatch.setattr(
        symmetry_analysis,
        "_minimal_symmetry_candidates_for_valley",
        lambda valley_ctx, bravais, spatial_operations=None, structure=None, **_kwargs: [
            {"index": 0, "name": "E", "antiunitary": False, "closed": True, "rotation_cart": np.eye(3)},
            {"index": 2, "name": "C3z", "antiunitary": False, "closed": True, "rotation_cart": np.eye(3)},
            {"index": 3, "name": "C3z^2", "antiunitary": False, "closed": True, "rotation_cart": np.eye(3)},
        ],
    )
    monkeypatch.setattr(symmetry_analysis, "_default_validation_q_points", lambda: [("Gamma", np.zeros(3))])
    raw_validation_calls = []

    def fake_candidate_rows(self, candidate, valley, valley_label, q_label, q_target, tolerance):
        operation = symmetry_analysis.displayed_operation_name(candidate["name"])
        raw_validation_calls.append(operation)
        if operation == "C3z^2":
            raise AssertionError("C3z^2 should be derived from verified C3z")
        return {
            "valley": valley_label,
            "operation": operation,
            "k_label": q_label,
            "supported": True,
            "status": "exact",
            "covariance_status": "exact",
            "not_supported_reason": "",
            "residual_H_raw": 0.0,
        }

    runner._candidate_rows_for_q = MethodType(fake_candidate_rows, runner)
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

    assert raw_validation_calls == ["E", "C3z"]
    details_by_operation = {row["operation"]: row for row in payload["details"]}
    assert details_by_operation["C3z^2"]["status"] == "derived"
    assert details_by_operation["C3z^2"]["covariance_status"] == "derived"
    assert details_by_operation["C3z^2"]["residual_H_raw"] is None
    summary_by_operation = {
        entry["operation"]: entry
        for entry in payload["summary"]["operations"]["Gamma"]
    }
    assert summary_by_operation["C3z^2"]["supported"] is True
    assert summary_by_operation["C3z^2"]["export_raw_h_matrix"] is False


@pytest.mark.parametrize("validation", ["gamma", "export_only", "raw_h_only", "none"])
def test_runner_rejects_non_strict_validation_modes(tmp_path, validation):
    runner = _make_runner(tmp_path)
    runner.config.symmetry_analysis.validation = validation

    with pytest.raises(ValueError, match="symmetry.validation"):
        symmetry_analysis._symmetry_validation_mode(runner.config.symmetry_analysis)


@pytest.mark.parametrize(
    ("rows", "require_overlap", "expected"),
    [
        ([{"supported": True, "residual_H_raw": 1.0e-5, "residual_S_raw": None}], False, True),
        ([{"supported": True, "residual_H_raw": None, "residual_S_raw": None}], False, False),
        ([{"supported": True, "residual_H_raw": 2.0e-2, "residual_S_raw": None}], False, False),
        ([{"supported": True, "residual_H_raw": 1.0e-5, "residual_S_raw": None}], True, False),
        ([{"supported": True, "residual_H_raw": 1.0e-5, "residual_S_raw": 2.0e-2}], True, False),
        ([{"supported": True, "residual_H_raw": 1.0e-5, "residual_S_raw": 2.0e-5}], True, True),
        ([{"supported": False, "residual_H_raw": 0.0, "residual_S_raw": 0.0}], True, False),
    ],
)
def test_production_covariance_gate_requires_complete_h_and_s_validation(rows, require_overlap, expected):
    assert symmetry_analysis._candidate_rows_pass_production_validation(
        rows,
        tolerance=1.0e-2,
        require_overlap=require_overlap,
    ) is expected


def test_raw_projected_hs_projects_overlap_when_source_is_nonorthogonal(tmp_path):
    runner = _make_runner(tmp_path)

    class FakeCalculator:
        hr_supercell = "H"
        sr_supercell = "S"
        TAPW_parameters = object()

        @staticmethod
        def _build_getk_phase_context(_q):
            return object()

        @staticmethod
        def _assemble_sparse_realspace_matrix(_source, _phase, *, type):
            scale = 2.0 if type == "S" else 1.0
            return scipy.sparse.identity(2, dtype=np.complex128, format="csr") * scale

        @staticmethod
        def cal_TAPW_hamiltonian_k_cpu(matrix, **_kwargs):
            return matrix.toarray()

    runner._calculator_cache[5] = FakeCalculator()

    hamk, samk = runner._raw_projected_hs(5, np.zeros(3))

    assert hamk == pytest.approx(np.eye(2))
    assert samk == pytest.approx(2.0 * np.eye(2))


def test_canonical_writer_drops_unvalidated_raw_h_record(tmp_path):
    runner = _make_runner(tmp_path)
    runner.config.output_layout = SimpleNamespace(
        style="canonical_v1",
        root=str(tmp_path / "outputs"),
        profile=None,
        q_shell="q06",
    )
    output_dir = tmp_path / "symmetry"
    output_dir.mkdir()
    record = _minimal_rawh_payload("C2T")["representations"][0]
    record["production_validated"] = False

    runner._write_representations([record], output_dir)

    assert not (output_dir / "representations.npz").exists()


def test_c3_covariance_validation_uses_raw_h_action_with_periodic_gauge(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    runner.structure = SimpleNamespace(spin=False, reciprocal_Tmat=np.eye(3))
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=1,
        valley_label="K1",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.zeros((1, 2))},
        moire_reciprocal_basis=np.eye(2),
        calculator=SimpleNamespace(TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr"))),
    )
    runner._valley_context_for_valley = MethodType(lambda self, valley: valley_ctx, runner)

    d0 = scipy.sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128))
    pg = scipy.sparse.diags([1.0, -1.0], dtype=np.complex128, format="csr")
    raw_action = (d0 @ pg).toarray()
    h_raw = np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=np.complex128)
    np.testing.assert_allclose(raw_action @ h_raw @ raw_action.conj().T, h_raw, atol=1.0e-14)
    assert symmetry_analysis.frobenius_relative_residual(h_raw, d0 @ h_raw @ d0.conj().T, denominator=h_raw) > 1.0

    runner._build_transport = MethodType(lambda self, candidate, valley, q_target, q_source: d0, runner)
    runner._raw_projected_hs = MethodType(lambda self, valley, q_local: (h_raw, None), runner)
    monkeypatch.setattr(
        symmetry_analysis,
        "build_periodic_gauge_matrix_for_candidate",
        lambda structure, valley_ctx, candidate: (
            pg,
            {"pg_shift_by_group_coeffs": {"0": [1, 0]}, "pg_phase_convention": "test"},
            {"pg_reason": "test"},
        ),
    )

    row = runner._candidate_rows_for_q(
        {"index": 1, "name": "C3z", "antiunitary": False, "closed": True, "rotation_cart": np.eye(3)},
        valley=1,
        valley_label="K1",
        q_label="Gamma",
        q_target=np.zeros(3),
        tolerance=1.0e-10,
    )

    assert row["supported"] is True
    assert row["residual_H_raw"] == pytest.approx(0.0, abs=1.0e-14)
    assert row["status"] == "exact"


def test_k_source_c2_and_internal_c2t_use_same_spglib_axis(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    runner.config.compute = SimpleNamespace(TAPW=True, valleys=[1])
    runner.config.symmetry_analysis.valleys = [1]
    runner.config.twist = SimpleNamespace(bravais="hex")
    runner.structure = SimpleNamespace(spin=False, reciprocal_Tmat=np.eye(3))
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=1,
        valley_label="K1",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.zeros((1, 2))},
        moire_reciprocal_basis=np.eye(2),
        calculator=SimpleNamespace(TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr"))),
    )
    theta = np.deg2rad(120.0)
    spglib_c2 = {
        "index": 7,
        "rotation_frac": np.eye(3),
        "translation_frac": np.zeros(3),
        "rotation_cart": np.array(
            [
                [np.cos(theta), np.sin(theta), 0.0],
                [np.sin(theta), -np.cos(theta), 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        "translation_cart": np.zeros(3),
    }
    runner._calculator_for_valley = MethodType(
        lambda self, valley: SimpleNamespace(valley_flag="K1", TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr"))),
        runner,
    )
    runner._valley_context_for_valley = MethodType(lambda self, valley: valley_ctx, runner)
    runner._select_antiunitary_c2_layer_exchange_spatial_operation = MethodType(lambda self, valley, spatial_operations, tolerance: spglib_c2, runner)
    monkeypatch.setattr(symmetry_analysis, "collect_spglib_spatial_operations", lambda structure, **_kwargs: [spglib_c2])
    monkeypatch.setattr(symmetry_analysis, "_default_validation_q_points", lambda: [("Gamma", np.zeros(3))])
    runner._candidate_rows_for_q = MethodType(
        lambda self, candidate, valley, valley_label, q_label, q_target, tolerance: {
            "valley": valley_label,
            "operation": symmetry_analysis.displayed_operation_name(candidate["name"]),
            "k_label": q_label,
            "supported": bool(candidate.get("closed", False)),
            "status": "exact" if candidate.get("closed", False) else "not_supported",
            "not_supported_reason": "" if candidate.get("closed", False) else candidate.get("closure_reason", ""),
            "residual_H_raw": 0.0 if candidate.get("closed", False) else None,
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
            "raw_h_matrix": scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
        },
        runner,
    )

    payload = runner._analyze()

    entries = {entry["operation"]: entry for entry in payload["summary"]["operations"]["K1"]}
    assert entries["C2"]["axis_angle_deg"] == pytest.approx(60.0)
    assert entries["C2T"]["axis_angle_deg"] == pytest.approx(60.0)
    assert entries["C2"]["spglib_index"] == 7
    assert entries["C2T"]["spglib_index"] == 7


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
                "operation": "C2",
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
    debug = json.loads((output_dir / "diagnostics" / "debug.json").read_text(encoding="utf-8"))
    assert debug["details"][0]["debug"]["center_convention"] == "group_k_centers"
    assert debug["details"][0]["debug"]["phase_side"] == "left_target_rows"


def test_summary_markdown_documents_m_valley_convention_without_backend_names(tmp_path):
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
    assert "M-valley convention: M1, M2, and M3 are related by C3z." in text
    assert "single-M valley output exports only operations closed within the selected valley block" in text
    assert "M" + "_C2_eta" not in text


def test_summary_markdown_is_human_source_symmetry_report(tmp_path):
    runner = _make_runner(tmp_path)
    summary = {
        "valleys": [31, 5],
        "tolerance": 1.0e-2,
        "output_schema": "tapw_source_symmetry/v2",
        "operations": {
            "M1": [
                {
                    "operation": "TR",
                    "antiunitary": True,
                    "supported": True,
                    "status": "exact",
                    "source_valley": "M1",
                    "target_valley": "M1",
                    "closed_in_active_set": True,
                    "role": "internal",
                },
                {
                    "operation": "C2",
                    "antiunitary": False,
                    "supported": True,
                    "status": "exact",
                    "source_valley": "M1",
                    "target_valley": "M1",
                    "closed_in_active_set": True,
                    "role": "internal",
                },
                {
                    "operation": "C3z",
                    "antiunitary": False,
                    "supported": False,
                    "status": "not_supported",
                    "not_supported_reason": "valley_not_closed",
                    "source_valley": "M1",
                    "target_valley": "M2",
                    "closed_in_active_set": False,
                    "role": "inter-valley",
                },
            ],
            "Gamma": [
                {
                    "operation": "TR",
                    "antiunitary": True,
                    "supported": True,
                    "status": "exact",
                    "source_valley": "Gamma",
                    "target_valley": "Gamma",
                    "closed_in_active_set": True,
                    "role": "internal",
                },
                {
                    "operation": "C3z",
                    "antiunitary": False,
                    "supported": True,
                    "status": "exact",
                    "source_valley": "Gamma",
                    "target_valley": "Gamma",
                    "closed_in_active_set": True,
                    "role": "internal",
                },
            ],
        },
        "minimal_generators": {"M1": ["TR", "C2"], "Gamma": ["TR", "C3z"]},
    }
    representations = [
        {
            "valley_label": "M1",
            "operation": "TR",
            "matrix": scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
            "raw_h_operator_file": "M1/TR_rawH.npz",
            "matrix_role": "raw_h_action",
        },
        {
            "valley_label": "Gamma",
            "operation": "C3z",
            "matrix": scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
            "raw_h_operator_file": "Gamma/C3z_rawH.npz",
            "matrix_role": "raw_h_action",
        },
    ]

    text = "\n".join(
        runner._summary_markdown_lines(
            summary,
            representations=representations,
            developer_outputs=False,
        )
    )

    assert text.startswith("# TAPW Source Symmetry Summary")
    assert "## Run" in text
    assert "- Active valleys: M1, Gamma" in text
    assert "- Production matrix source: raw-H action only" in text
    assert "## Candidate Valley Actions" in text
    assert "| TR | built-in | yes | M1 | M1 | yes | yes | exact | yes | internal |  |" in text
    assert "| C3z | template | no | M1 | M2 | no | no | not_supported | no | inter-valley | valley_not_closed |" in text
    assert "## Exported Internal Generators For KP" in text
    assert "- M1: TR, C2 (E implicit)" in text
    assert "- Gamma: TR, C3z (E implicit)" in text
    assert "## Inter-Valley Operations" in text
    assert "- C3z maps M1 -> M2" in text
    assert "not exported as a single-M1 internal matrix" in text
    assert "## Exported Production Matrices" in text
    assert "| M1 -> M1 | TR | representations/M1/TR_rawH.npz | raw_h_action |" in text
    assert "| Gamma -> Gamma | C3z | representations/Gamma/C3z_rawH.npz | raw_h_action |" in text
    assert "## Diagnostics" in text
    assert "- Developer outputs: disabled" in text
    assert "D0/PG/Pin matrices are not production inputs." in text
    assert "## KP Model Guidance" in text
    assert "- Single-valley M1 KP model may use: TR, C2 (E implicit)." in text
    assert "- Do not use M1 C3z as a single-valley constraint from this output." in text
    assert "Minimal generators:" not in text


def test_summary_markdown_documents_single_k_internal_and_inter_valley_operations(tmp_path):
    runner = _make_runner(tmp_path)
    text = "\n".join(
        runner._summary_markdown_lines(
            {
                "valleys": [1],
                "tolerance": 1.0e-2,
                "operations": {
                    "K1": [
                        {
                            "operation": "C3z",
                            "antiunitary": False,
                            "supported": True,
                            "source_valley": "K1",
                            "target_valley": "K1",
                            "closed_in_active_set": True,
                            "role": "internal",
                        },
                        {
                            "operation": "C2T",
                            "antiunitary": True,
                            "supported": True,
                            "source_valley": "K1",
                            "target_valley": "K1",
                            "closed_in_active_set": True,
                            "role": "internal",
                        },
                        {
                            "operation": "TR",
                            "antiunitary": True,
                            "supported": False,
                            "source_valley": "K1",
                            "target_valley": "K2",
                            "closed_in_active_set": False,
                            "role": "inter-valley",
                        },
                        {
                            "operation": "C2",
                            "antiunitary": False,
                            "supported": False,
                            "source_valley": "K1",
                            "target_valley": "K2",
                            "closed_in_active_set": False,
                            "role": "inter-valley",
                        },
                    ]
                },
                "minimal_generators": {"K1": ["C3z", "C2T"]},
            },
            representations=[],
            developer_outputs=False,
        )
    )

    assert "| C3z | template | no | K1 | K1 | yes | yes | supported | no | internal |  |" in text
    assert "| C2T | template | yes | K1 | K1 | yes | yes | supported | no | internal |  |" in text
    assert "| TR | built-in | yes | K1 | K2 | no | no | not_supported | no | inter-valley |  |" in text
    assert "| C2 | template | no | K1 | K2 | no | no | not_supported | no | inter-valley |  |" in text
    assert "- Single-valley K1 KP model may use: C3z, C2T (E implicit)." in text
    assert "- TR and C2 are source physical symmetries but are not single-K1 internal constraints in this output." in text


def test_summary_markdown_separates_valley_closed_from_supported_candidates(tmp_path):
    runner = _make_runner(tmp_path)
    text = "\n".join(
        runner._summary_markdown_lines(
            {
                "valleys": [5],
                "tolerance": 2.0e-2,
                "operations": {
                    "Gamma": [
                        {
                            "operation": "TR",
                            "antiunitary": True,
                            "supported": True,
                            "status": "exact",
                            "source_valley": "Gamma",
                            "target_valley": "Gamma",
                            "closed_in_active_set": True,
                            "role": "internal",
                            "export_raw_h_matrix": True,
                        },
                        {
                            "operation": "C2",
                            "axis_angle_deg": 90.0,
                            "antiunitary": False,
                            "supported": False,
                            "status": "not_supported",
                            "not_supported_reason": "atom_mapping_missing",
                            "spglib_index": "",
                            "source_valley": "Gamma",
                            "target_valley": "Gamma",
                            "closed_in_active_set": True,
                            "role": "internal",
                            "export_raw_h_matrix": False,
                        },
                    ]
                },
                "minimal_generators": {"Gamma": ["TR"]},
            },
            representations=[],
            developer_outputs=False,
        )
    )

    assert "## Candidate Valley Actions" in text
    assert "valley-map closed only means the operation maps into the selected active valley block" in text
    assert (
        "| operation | candidate source | antiunitary | source valley | target valley | valley-map closed | "
        "supported | status | exported raw-H | role | reason |"
    ) in text
    assert "| TR | built-in | yes | Gamma | Gamma | yes | yes | exact | yes | internal |  |" in text
    assert "| C2 (axis 90deg) | template | no | Gamma | Gamma | yes | no | not_supported | no | internal | atom_mapping_missing |" in text
    assert "closed in active set" not in text
    assert "## Exported Internal Generators For KP" in text
    assert "- Gamma: TR (E implicit)" in text


def test_summary_markdown_hides_nonexported_redundant_candidates_but_keeps_identity(tmp_path):
    runner = _make_runner(tmp_path)
    text = "\n".join(
        runner._summary_markdown_lines(
            {
                "valleys": [5],
                "tolerance": 1.0e-2,
                "operations": {
                    "Gamma": [
                        {
                            "operation": "E",
                            "candidate_source": "built-in",
                            "supported": True,
                            "status": "derived",
                            "export_raw_h_matrix": False,
                            "source_valley": "Gamma",
                            "target_valley": "Gamma",
                            "closed": True,
                        },
                        {
                            "operation": "C3z",
                            "candidate_source": "spglib:1",
                            "supported": True,
                            "status": "derived",
                            "export_raw_h_matrix": True,
                            "source_valley": "Gamma",
                            "target_valley": "Gamma",
                            "closed": True,
                        },
                        {
                            "operation": "C3z^2",
                            "candidate_source": "spglib:2",
                            "supported": True,
                            "status": "derived",
                            "export_raw_h_matrix": False,
                            "source_valley": "Gamma",
                            "target_valley": "Gamma",
                            "closed": True,
                        },
                    ]
                },
                "minimal_generators": {"Gamma": ["C3z"]},
            },
            representations=[],
            developer_outputs=False,
        )
    )

    assert "| E | built-in | no | Gamma | Gamma | yes | yes | derived | no | internal |  |" in text
    assert "| C3z | template | no | Gamma | Gamma | yes | yes | derived | yes | internal |  |" in text
    assert "C3z^2" not in text


def test_summary_markdown_infers_antiunitary_for_legacy_t_and_c2t_entries(tmp_path):
    runner = _make_runner(tmp_path)
    text = "\n".join(
        runner._summary_markdown_lines(
            {
                "valleys": [1],
                "tolerance": 1.0e-2,
                "operations": {
                    "K1": [
                        {"operation": "T", "supported": True},
                        {"operation": "C2T", "supported": True},
                        {"operation": "C3z", "supported": True},
                    ]
                },
                "minimal_generators": {"K1": ["C3z", "C2T"]},
            }
        )
    )

    assert "| TR | built-in | yes | K1 | K1 | yes | yes | supported | no | internal |  |" in text
    assert "| C2T | template | yes | K1 | K1 | yes | yes | supported | no | internal |  |" in text
    assert "| C3z | template | no | K1 | K1 | yes | yes | supported | no | internal |  |" in text


def _fake_spglib_operation(index: int, rotation_cart):
    return {
        "index": int(index),
        "rotation_frac": np.eye(3),
        "translation_frac": np.zeros(3),
        "rotation_cart": np.asarray(rotation_cart, dtype=float),
        "translation_cart": np.zeros(3),
    }


def test_source_symmetry_candidates_include_m_inter_valley_c3_operations_from_spglib(tmp_path):
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=31,
        valley_label="M1",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.zeros((1, 2))},
        moire_reciprocal_basis=np.eye(2),
        calculator=None,
    )
    c2_operation = _fake_spglib_operation(7, np.diag([-1.0, 1.0, -1.0]))
    spatial_operations = [
        _fake_spglib_operation(2, symmetry_analysis._rotation_z_cart(120.0)),
        _fake_spglib_operation(3, symmetry_analysis._rotation_z_cart(240.0)),
        c2_operation,
    ]

    candidates = symmetry_analysis._minimal_symmetry_candidates_for_valley(
        valley_ctx,
        "hex",
        spatial_operations=spatial_operations,
        selected_c2_operation=c2_operation,
    )
    by_name = {candidate["name"]: candidate for candidate in candidates}

    assert by_name["C3z"]["target_valley_label"] == "M2"
    assert by_name["C3z"]["source_symmetry_role"] == "inter-valley"
    assert by_name["C3z"]["export_raw_h_matrix"] is False
    assert by_name["C3z^2"]["target_valley_label"] == "M3"
    assert by_name["C2"]["source_symmetry_role"] == "internal"
    assert by_name["C2"]["transport_backend"] == symmetry_analysis.BACKEND_C2_LAYER_EXCHANGE_UNITARY


def test_source_symmetry_candidates_include_k_inter_valley_tr_and_spglib_c2_operations(tmp_path):
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=1,
        valley_label="K1",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.zeros((1, 2))},
        moire_reciprocal_basis=np.eye(2),
        calculator=None,
    )
    c2_operation = _fake_spglib_operation(7, np.diag([-1.0, 1.0, -1.0]))

    candidates = symmetry_analysis._minimal_symmetry_candidates_for_valley(
        valley_ctx,
        "hex",
        spatial_operations=[c2_operation],
        selected_c2_operation=c2_operation,
    )
    by_name = {candidate["name"]: candidate for candidate in candidates}

    assert by_name["TR"]["antiunitary"] is True
    assert by_name["TR"]["target_valley_label"] == "K2"
    assert by_name["TR"]["source_symmetry_role"] == "inter-valley"
    assert by_name["TR"]["export_raw_h_matrix"] is False
    assert by_name["C2"]["target_valley_label"] == "K2"
    assert by_name["C2"]["source_symmetry_role"] == "inter-valley"
    assert by_name["C2T"]["target_valley_label"] == "K1"
    assert by_name["C2T"]["spatial_parent"] == "C2"
    assert by_name["C2T"]["transport_backend"] == symmetry_analysis.BACKEND_C2_LAYER_EXCHANGE_ANTIUNITARY
    assert np.allclose(by_name["C2"]["rotation_cart"], by_name["C2T"]["rotation_cart"])


def test_source_symmetry_candidates_are_generated_from_spglib_source_operations(tmp_path):
    def _context(valley, label):
        return symmetry_analysis.ValleyContext(
            valley=valley,
            valley_label=label,
            valley_center_cart=np.zeros(2),
            partner_center_cart=np.zeros(2),
            group_k_centers={0: np.zeros(2)},
            group_m_k_centers={0: np.zeros(2)},
            group_g_vectors={0: np.zeros((1, 2))},
            moire_reciprocal_basis=np.eye(2),
            calculator=None,
        )

    c2_operation = _fake_spglib_operation(7, np.diag([-1.0, 1.0, -1.0]))
    spatial_operations = [
        _fake_spglib_operation(2, symmetry_analysis._rotation_z_cart(120.0)),
        _fake_spglib_operation(3, symmetry_analysis._rotation_z_cart(240.0)),
        c2_operation,
    ]
    expected_source_ops = {"E", "TR", "C3z", "C3z^2", "C2"}
    for valley, label in [(5, "Gamma"), (31, "M1"), (1, "K1")]:
        candidates = symmetry_analysis._minimal_symmetry_candidates_for_valley(
            _context(valley, label),
            "hex",
            spatial_operations=spatial_operations,
            selected_c2_operation=c2_operation,
        )
        displayed = {symmetry_analysis.displayed_operation_name(candidate["name"]) for candidate in candidates}
        assert expected_source_ops.issubset(displayed)

    k_candidates = symmetry_analysis._minimal_symmetry_candidates_for_valley(
        _context(1, "K1"),
        "hex",
        spatial_operations=spatial_operations,
        selected_c2_operation=c2_operation,
    )
    assert "C2T" in {candidate["name"] for candidate in k_candidates}
    assert "K" + "_C2T" not in {candidate["name"] for candidate in k_candidates}


def test_minimal_generators_ignore_inter_valley_even_if_marked_supported():
    generators = symmetry_analysis._minimal_generators_by_valley(
        {
            "K1": [
                {"operation": "TR", "supported": True, "role": "inter-valley", "export_raw_h_matrix": False},
                {"operation": "C2", "supported": True, "role": "inter-valley", "export_raw_h_matrix": False},
                {"operation": "C3z", "supported": True, "role": "internal", "export_raw_h_matrix": True},
                {"operation": "C2T", "supported": True, "role": "internal", "export_raw_h_matrix": True},
            ]
        }
    )

    assert generators == {"K1": ["C3z", "C2T"]}


def test_inter_valley_reason_is_registered():
    assert "inter_valley_operation_not_exported_in_single_valley_block" in symmetry_analysis.NOT_SUPPORTED_REASONS


def test_spglib_symprec_defaults_to_symmetry_analysis_tolerance():
    config = SimpleNamespace(spglib_symprec=None)

    assert symmetry_analysis._spglib_symprec_from_config(config, 2.0e-2) == pytest.approx(2.0e-2)


def test_spglib_symprec_can_be_configured_independently():
    config = SimpleNamespace(spglib_symprec=5.0e-3)

    assert symmetry_analysis._spglib_symprec_from_config(config, 2.0e-2) == pytest.approx(5.0e-3)


def test_source_symmetry_candidates_do_not_emit_spatial_templates_without_spglib_support():
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=5,
        valley_label="Gamma",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.zeros((1, 2))},
        moire_reciprocal_basis=np.eye(2),
        calculator=None,
    )

    candidates = symmetry_analysis._minimal_symmetry_candidates_for_valley(
        valley_ctx,
        "hex",
        spatial_operations=[],
        structure=None,
    )

    assert [candidate["name"] for candidate in candidates] == ["E", "TR"]


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
    assert "K" + "_C2T" not in text
    assert "C2 (axis 30deg)" in text
    assert "C2T (axis 60deg)" in text
    assert "Exported Internal Generators For KP" in text
    assert "- Gamma: C3z, C2 (E implicit)" in text
    assert "- K1: C2T (E implicit)" in text


def test_displayed_operation_name_keeps_canonical_family_names():
    assert symmetry_analysis.displayed_operation_name("C2") == "C2"
    assert symmetry_analysis.displayed_operation_name("C2T") == "C2T"


def test_resolve_requested_symmetrization_operations_requires_supported_manifest_entries():
    summary = {
        "operations": {
            "K1": [
                {"operation": "C3z", "supported": True},
                {"operation": "C2T", "supported": True},
                {"operation": "C2", "supported": False, "not_supported_reason": "valley_not_closed"},
                {"operation": "TR", "supported": True, "role": "inter-valley", "export_raw_h_matrix": False},
            ]
        },
        "minimal_generators": {"K1": ["C3z", "C2T"]},
    }

    resolve = symmetry_analysis.resolve_requested_symmetrization_operations

    assert resolve(summary, "K1", True) == ["C3z", "C2T"]
    assert resolve(summary, "K1", ["C2T"]) == ["C2T"]

    with pytest.raises(ValueError, match="C2.*not supported"):
        resolve(summary, "K1", ["C2"])

    with pytest.raises(ValueError, match="TR.*not an internal"):
        resolve(summary, "K1", ["T"])


def test_default_validation_points_are_gamma_plus_one_generic_point():
    points = symmetry_analysis._default_validation_q_points()

    assert [label for label, _ in points] == ["Gamma", "q1"]


def test_candidate_row_shortcuts_identity_without_raw_h_projection(tmp_path):
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
    assert calls == []


def test_candidate_row_uses_mapped_raw_projected_h_for_c3_covariance(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    runner.structure = SimpleNamespace(spin=False, reciprocal_Tmat=np.eye(3))
    valley_ctx = symmetry_analysis.ValleyContext(
        valley=1,
        valley_label="K1",
        valley_center_cart=np.zeros(2),
        partner_center_cart=np.zeros(2),
        group_k_centers={0: np.zeros(2)},
        group_m_k_centers={0: np.zeros(2)},
        group_g_vectors={0: np.zeros((1, 2))},
        moire_reciprocal_basis=np.eye(2),
        calculator=SimpleNamespace(TAPW_parameters=SimpleNamespace(g_matrix=scipy.sparse.identity(2, format="csr"))),
    )
    transport = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    q_target = np.array([0.2, -0.1, 0.0], dtype=float)
    q_source = np.array([0.1, 0.2, 0.0], dtype=float)
    h_source = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.complex128)
    h_target = transport @ h_source @ transport.conj().T
    s_source = np.array([[1.0, 0.0], [0.0, 1.5]], dtype=np.complex128)
    s_target = transport @ s_source @ transport.conj().T
    raw_calls = []
    transport_calls = []
    runner.sr_supercell = object()

    runner._valley_context_for_valley = MethodType(lambda self, valley: valley_ctx, runner)

    def _fake_map_inverse_q(self, candidate, requested_q_target):
        np.testing.assert_allclose(requested_q_target, q_target)
        return q_source

    runner._map_inverse_q = MethodType(_fake_map_inverse_q, runner)

    def _fake_build_transport(self, candidate, valley, requested_q_target, requested_q_source):
        transport_calls.append(
            (
                tuple(np.asarray(requested_q_target, dtype=float)),
                tuple(np.asarray(requested_q_source, dtype=float)),
            )
        )
        return transport

    runner._build_transport = MethodType(_fake_build_transport, runner)
    runner._calculator_for_valley = MethodType(
        lambda self, valley: (_ for _ in ()).throw(AssertionError("C3 validation must not use the legacy C3 orbit")),
        runner,
    )

    def _fake_raw_projected_hs(self, valley, q_local):
        q_key = tuple(np.asarray(q_local, dtype=float))
        raw_calls.append(q_key)
        if np.allclose(q_local, q_target):
            return h_target, s_target
        if np.allclose(q_local, q_source):
            return h_source, s_source
        raise AssertionError(f"unexpected C3 covariance momentum {q_key}")

    runner._raw_projected_hs = MethodType(_fake_raw_projected_hs, runner)
    monkeypatch.setattr(
        symmetry_analysis,
        "build_periodic_gauge_matrix_for_candidate",
        lambda structure, valley_ctx, candidate: (
            scipy.sparse.identity(2, dtype=np.complex128, format="csr"),
            {"pg_shift_by_group_coeffs": {"0": [0, 0]}, "pg_phase_convention": "test"},
            {"pg_identity": True},
        ),
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
        q_label="q1",
        q_target=q_target,
        tolerance=1.0e-2,
    )

    assert row["residual_H_raw"] == 0.0
    assert row["residual_S_raw"] == 0.0
    assert row["status"] == "exact"
    assert raw_calls == [tuple(q_target), tuple(q_source)]
    assert transport_calls == [(tuple(q_target), tuple(q_source))]


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
