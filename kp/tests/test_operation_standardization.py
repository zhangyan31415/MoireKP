from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.model.config_schema import CANONICAL_INTERNAL_NAMES, canonical_operation_name_for_valley
from kp.model.configured import (
    ACTION_SPECS,
    _build_operation_registry,
    _resolved_action_for_source_operation,
    _symmetry_operation_index,
)
from kp.model.symmetry import MatrixSymmetryGenerator, load_symmetry_source
from kp.symmetry.project import (
    _model_action_metadata,
    _operation_action_metadata,
    _resolve_projected_model_action,
    _select_operation_matrix_kind,
    reflection_axis_equiv,
)


def _source_meta(antiunitary: bool = False) -> dict[str, object]:
    return {
        "name": "C2",
        "source_matrix_role": "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "continuum_internal_rep",
        "gauge_correction": {"kind": "none"},
        "antiunitary_convention": "U_K" if antiunitary else "none",
        "spin_map": "from_kp_symm_output",
        "valley_map": "identity",
    }


def test_standard_family_names_do_not_rewrite_unknown_families() -> None:
    gamma = {"valley_type": "Gamma", "mode": "single_valley", "spin_convention": "spinful"}
    k_single = {"valley_type": "K", "mode": "single_valley", "spin_convention": "spin_up_only"}
    m_effective = {"valley_type": "M", "mode": "single_valley", "spin_convention": "spinless_effective"}

    assert canonical_operation_name_for_valley("C4", gamma) == "C4"
    assert canonical_operation_name_for_valley("C4T", k_single) == "C4T"
    assert canonical_operation_name_for_valley("C4_eff", m_effective) == "C4_eff"
    assert canonical_operation_name_for_valley("TR_eff", m_effective) == "TR"
    assert canonical_operation_name_for_valley("C2_eff", m_effective) == "C2"
    assert canonical_operation_name_for_valley("C2TR_eff", m_effective) == "C2T"


def test_canonical_operation_registries_exclude_legacy_axis_names() -> None:
    allowed = {"C2", "C2T", "C3z", "TR"}
    assert CANONICAL_INTERNAL_NAMES <= allowed
    assert set(ACTION_SPECS) <= allowed
    assert {"TR_eff", "C2_eff", "C2TR_eff"}.isdisjoint(CANONICAL_INTERNAL_NAMES)
    assert {"TR_eff", "C2_eff", "C2TR_eff"}.isdisjoint(ACTION_SPECS)


def test_loaded_symmetry_source_keeps_source_and_canonical_family_separate(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.eye(2, dtype=complex))
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "operation": "C2",
                        "matrix_file": "C2_low_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "sector_map": "layer_exchange",
                        "antiunitary": False,
                        "pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)

    op = source.metadata["operations"][0]
    assert op["operation"] == "C2"
    assert op["name"] == "C2"
    assert op["k_map"] == {"type": "reflection", "axis_deg": 150.0}


def test_kp_symm_manifest_action_wins_over_default_operation_row(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.eye(2, dtype=complex))
    (tmp_path / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "operation": "C2",
                        "name": "C2",
                        "matrix_file": "C2_low_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                        "sector_map": "identity",
                        "antiunitary": False,
                        "model_action": {
                            "antiunitary": False,
                            "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                            "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                            "sector_map": "identity",
                        },
                        "pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    }
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
            "operations": [
                {
                    **_source_meta(),
                    "operation": "C2",
                    "name": "C2",
                    "matrix_file": "C2_low_raw.npy",
                    "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                    "sector_map": "layer_exchange",
                    "antiunitary": False,
                }
            ],
        },
        base=tmp_path,
        expected_dim=2,
    )

    op = source.metadata["operations"][0]
    assert op["sector_map"] == "identity"
    assert op["model_action"]["sector_map"] == "identity"


def test_kp_symm_representation_matrix_kind_loads_representation_file(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.zeros((2, 2), dtype=complex))
    np.save(tmp_path / "C2_low_representation_raw.npy", np.eye(2, dtype=complex))
    (tmp_path / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "operation": "C2",
                        "name": "C2",
                        "matrix_file": "C2_low_raw.npy",
                        "representation_matrix_file": "C2_low_representation_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                        "sector_map": "identity",
                        "antiunitary": False,
                        "representation_pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="production symmetry matrices must use matrix_kind='action'"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path), "use": "raw", "matrix_kind": "representation"},
            base=tmp_path,
            expected_dim=2,
        )


def test_rawh_action_is_only_production_matrix_source_even_if_rep_support_cleaner(tmp_path: Path) -> None:
    np.save(tmp_path / "C3_low_raw.npy", 2.0 * np.eye(2, dtype=complex))
    np.save(tmp_path / "C3_low_representation_raw.npy", 3.0 * np.eye(2, dtype=complex))
    np.save(tmp_path / "C2_low_raw.npy", 4.0 * np.eye(2, dtype=complex))
    np.save(tmp_path / "C2_low_representation_raw.npy", 5.0 * np.eye(2, dtype=complex))
    (tmp_path / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "mode": "gamma",
                "operations": [
                    {
                        **_source_meta(),
                        "operation": "C3",
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "representation_matrix_file": "C3_low_representation_raw.npy",
                        "k_map": {"type": "rotation", "angle_deg": 120.0, "in_model_frame": True},
                        "q_map": {"type": "rotation", "angle_deg": 120.0, "in_model_frame": True},
                        "sector_map": "identity",
                        "antiunitary": False,
                        "model_basis_action": {"support_resolution": {"support_matrix_source": "raw_action"}},
                        "pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    },
                    {
                        **_source_meta(),
                        "operation": "C2",
                        "name": "C2",
                        "matrix_file": "C2_low_raw.npy",
                        "representation_matrix_file": "C2_low_representation_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                        "sector_map": "layer_exchange",
                        "antiunitary": False,
                        "model_basis_action": {"support_resolution": {"support_matrix_source": "representation"}},
                        "representation_pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)

    np.testing.assert_allclose(source.generator.matrices["C3z"], 2.0 * np.eye(2, dtype=complex))
    np.testing.assert_allclose(source.generator.matrices["C2"], 4.0 * np.eye(2, dtype=complex))
    by_name = {row["name"]: row for row in source.metadata["operations"]}
    assert by_name["C3z"]["matrix_kind"] == "action"
    assert by_name["C2"]["matrix_kind"] == "action"
    assert by_name["C2"]["representation_projection_diagnostic"]["support_matrix_source"] == "representation"
    assert by_name["C2"]["representation_projection_diagnostic"]["representation_support_cleaner_than_raw_action"] is True


def test_sector_map_auto_rejected_for_production_manifest() -> None:
    with pytest.raises(ValueError, match="requires explicit sector_map"):
        _operation_action_metadata(
            {"axis_deg": 0.0, "q_map": {"type": "reflection", "axis_deg": 0.0}},
            operation="C2",
            antiunitary=False,
        )


def test_q_map_inferred_from_k_map_is_reported_in_compat_mode() -> None:
    source = _operation_action_metadata(
        {"axis_deg": 0.0, "sector_map": "identity"},
        operation="C2",
        antiunitary=False,
        strict=False,
    )

    assert source["q_map_inferred_from_k_map"] is True
    assert source["q_map"] == source["k_map"]


def test_support_discovery_default_off() -> None:
    source = _operation_action_metadata(
        {
            "axis_deg": 0.0,
            "q_map": {"type": "reflection", "axis_deg": 0.0},
            "sector_map": "identity",
        },
        operation="C2",
        antiunitary=False,
    )
    model_action = _model_action_metadata(source, valley="M", operation="C2", rotation_deg=0.0)
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    q2 = np.array([[0.0, 0.0], [1.0, 0.0]])
    matrix = np.zeros((4, 4), dtype=complex)
    matrix[2:, :2] = np.eye(2)
    matrix[:2, 2:] = np.eye(2)

    resolved, basis_action = _resolve_projected_model_action(
        D_low=matrix,
        support_matrices=[("raw_action", matrix)],
        model_action=model_action,
        q_model1=q1,
        q_model2=q2,
        nlow_state_list=[[0], [0]],
        tol=1.0e-8,
    )

    assert resolved["sector_map"] == "identity"
    assert basis_action["sector_map"] == "identity"
    assert basis_action["support_resolution"]["action_mismatch"] is False
    assert basis_action["support_resolution"]["selected_action_candidate"]["sector_map"] == "identity"
    assert basis_action["support_resolution"]["candidate_source"] == "manifest_model_action"


def test_exactification_mismatch_reports_without_overwriting_by_default() -> None:
    action = {
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "sector_map": "identity",
    }
    q = np.array([[0.0, 0.0]])
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    resolved, basis_action = _resolve_projected_model_action(
        D_low=matrix,
        support_matrices=[("raw_action", matrix)],
        model_action=action,
        q_model1=q,
        q_model2=q.copy(),
        nlow_state_list=[[0], [0]],
        tol=1.0e-8,
        discover_action_candidates=True,
    )

    assert resolved["sector_map"] == "identity"
    assert basis_action["support_resolution"]["action_mismatch"] is True
    assert basis_action["support_resolution"]["selected_action_candidate"]["sector_map"] == "layer_exchange"


def test_rep_projection_is_diagnostic_only() -> None:
    model_basis_action = {"support_resolution": {"support_matrix_source": "representation"}}

    kind, report = _select_operation_matrix_kind(
        model_basis_action,
        representation_pair_rows=[],
    )

    assert kind == "action"
    assert report["kind"] == "action"
    assert report["representation_projection_diagnostic"]["status"] == "raw_action_exactification_problem"


def test_operation_registry_records_canonical_names_and_source_metadata() -> None:
    source_meta = _source_meta(antiunitary=True)
    operation = {
        **source_meta,
        "user_name": "C2T",
        "name": "C2T",
        "operation": "C2T",
        "antiunitary": True,
        "k_map": {"type": "reflection", "axis_deg": 0.0},
        "q_map": {"type": "reflection", "axis_deg": 0.0},
        "sector_map": "identity",
        "internal_resolved_action": {"k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True}},
        "matrix_kind": "continuum_internal_rep_exact",
    }
    model_config = SimpleNamespace(
        valley_model={"valley_type": "K", "mode": "single_valley", "spin_convention": "spin_up_only"},
        symmetry_source_config={
            "type": "kp_symm_output",
            "matrix_kind": "continuum_internal_rep_exact",
            "operations": [operation],
        },
        symmetry_map={"Kinect": [operation], "intra": [dict(operation)], "inter": []},
    )

    registry = _build_operation_registry(model_config)

    assert len(registry) == 1
    row = registry[0]
    assert row["user_operation"] == "C2T"
    assert row["canonical_operation"] == "C2T"
    assert row["source_operation"] == "C2T"
    assert row["valley_type"] == "K"
    assert row["valley_mode"] == "single_valley"
    assert row["spin_convention"] == "spin_up_only"
    assert row["operation_physics_level"] == "physical"
    assert row["matrix_kind"] == "continuum_internal_rep_exact"
    assert row["source_matrix_role"] == "raw_h_sewing_action"
    assert row["antiunitary_convention"] == "U_K"
    assert row["internal_resolved_action"]["k_map"]["type"] == "reflection"


def test_internal_resolved_action_without_provenance_does_not_override_manifest_model_action() -> None:
    metadata = {
        "operations": [
            {
                "name": "C2T",
                "operation": "C2T",
                "antiunitary": True,
                "model_action": {
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "sector_map": "identity",
                },
                "internal_resolved_action": {
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "sector_map": "layer_exchange",
                },
            }
        ]
    }

    index = _symmetry_operation_index(metadata, rotation_deg=210.0)

    assert index["C2T"]["k_map"]["axis_deg"] == pytest.approx(90.0)
    assert index["C2T"]["sector_map"] == "identity"


def test_internal_resolved_action_with_provenance_can_override_manifest_model_action() -> None:
    metadata = {
        "operations": [
            {
                "name": "C2T",
                "operation": "C2T",
                "antiunitary": True,
                "model_action": {
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "sector_map": "identity",
                },
                "internal_resolved_action": {
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "sector_map": "layer_exchange",
                    "provenance": {
                        "source": "exactification_support_report",
                        "action_mismatch": True,
                        "accepted_by": "explicit_accept_support_resolved_action",
                    },
                },
            }
        ]
    }

    index = _symmetry_operation_index(metadata, rotation_deg=210.0)

    assert index["C2T"]["k_map"]["axis_deg"] == pytest.approx(180.0)
    assert index["C2T"]["sector_map"] == "layer_exchange"


def test_nonstandard_action_name_is_rejected(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.eye(2, dtype=complex))
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "operation": "C2",
                        "matrix_file": "C2_low_raw.npy",
                        "k_map": {"type": "nonstandard_reflection_name"},
                        "sector_map": "layer_exchange",
                        "antiunitary": False,
                        "pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported k_map.type"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)


def test_matrix_generator_rejects_axis_encoded_operation_name() -> None:
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    generator = MatrixSymmetryGenerator({"C2": matrix}, {"operations": []})

    with pytest.raises(ValueError, match="not loaded"):
        generator.get_operator("C4", None)


def _reflection_matrix(axis_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(axis_deg))
    axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    return 2.0 * np.outer(axis, axis) - np.eye(2)


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)


def test_reflection_action_frame_conjugation_preserves_sector_identity() -> None:
    source_action = {
        "antiunitary": True,
        "k_map": {"type": "reflection", "axis_deg": 60.0},
        "q_map": {"type": "reflection", "axis_deg": 60.0},
        "sector_map": "identity",
    }

    model_action = _model_action_metadata(source_action, valley="K1", operation="C2T", rotation_deg=210.0)

    assert reflection_axis_equiv(model_action["k_map"]["axis_deg"], 90.0)
    assert reflection_axis_equiv(model_action["q_map"]["axis_deg"], 90.0)
    assert model_action["sector_map"] == "identity"
    assert model_action["antiunitary"] is True
    assert model_action["action_source"] == "derived_by_frame_conjugation"


def test_k_c2t_model_action_does_not_use_notebook_support_hardcode() -> None:
    source_action = {
        "antiunitary": True,
        "k_map": {"type": "reflection", "axis_deg": 60.0},
        "q_map": {"type": "reflection", "axis_deg": 60.0},
        "sector_map": "identity",
    }

    model_action = _model_action_metadata(source_action, valley="K1", operation="C2T", rotation_deg=210.0)

    assert not reflection_axis_equiv(model_action["k_map"]["axis_deg"], 180.0)
    assert model_action["sector_map"] != "layer_exchange"
    assert reflection_axis_equiv(model_action["k_map"]["axis_deg"], 90.0)
    assert model_action["sector_map"] == "identity"


def test_q_model_center_minus_q_source_does_not_change_reflection_linear_part() -> None:
    phi = 210.0
    source_axis = 60.0
    center_before = np.array([1.3, -0.7])
    center_after = np.array([-0.2, 0.4])
    affine_offset = np.array([0.6, -0.3])
    R = _rotation_matrix(phi)
    A_source = _reflection_matrix(source_axis)
    expected_linear = R @ A_source @ R.T
    wrong_linear = _reflection_matrix(180.0)

    q_model_points = np.array([[0.0, 0.0], [1.0, 0.2], [-0.3, 0.8], [0.4, -1.1]], dtype=float)
    transformed = []
    for q_model in q_model_points:
        q_source = center_before - R.T @ q_model
        q_source_prime = A_source @ q_source + affine_offset
        q_model_prime = R @ (center_after - q_source_prime)
        transformed.append(q_model_prime)
    transformed = np.asarray(transformed)
    design = np.column_stack([q_model_points, np.ones(len(q_model_points))])
    fit, *_ = np.linalg.lstsq(design, transformed, rcond=None)
    linear = fit[:2, :].T

    np.testing.assert_allclose(linear, expected_linear, atol=1.0e-12)
    np.testing.assert_allclose(linear, _reflection_matrix(90.0), atol=1.0e-12)
    assert np.linalg.norm(linear - wrong_linear) > 1.0


def test_sector_identity_conjugation_stays_identity_under_relabel() -> None:
    tau = {"L1": "L2", "L2": "L1"}
    source_identity = {"L1": "L1", "L2": "L2"}
    tau_inv = {value: key for key, value in tau.items()}
    conjugated = {sector: tau[source_identity[tau_inv[sector]]] for sector in tau}

    assert conjugated == {"L1": "L1", "L2": "L2"}
    assert conjugated != {"L1": "L2", "L2": "L1"}


def test_k_single_valley_c2t_fallback_does_not_guess_notebook_support() -> None:
    action = _resolved_action_for_source_operation(
        "C2T",
        {"valley_type": "K", "mode": "single_valley", "spin_convention": "spin_up_only"},
    )

    assert action["sector_map"] == "identity"
    assert reflection_axis_equiv(action["k_map"]["axis_deg"], ACTION_SPECS["C2T"]["k_map"]["axis_deg"])
    assert float(action["k_map"]["axis_deg"]) == float(ACTION_SPECS["C2T"]["k_map"]["axis_deg"])
