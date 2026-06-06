from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.model.config_schema import CANONICAL_INTERNAL_NAMES, canonical_operation_name_for_valley, validate_model_config
from kp.model.configured import ACTION_SPECS, _build_operation_registry, _is_k_single_valley_notebook_c2t
from kp.model.symmetry import MatrixSymmetryGenerator, load_symmetry_source


LEGACY_AXIS_OPERATION_NAMES = (
    "C2" + "x",
    "C2" + "y",
    "C2" + "y" + "T",
    "mirror_" + "x",
    "mirror_" + "y",
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


def test_canonical_operation_registries_exclude_legacy_axis_names() -> None:
    assert not (set(LEGACY_AXIS_OPERATION_NAMES) & CANONICAL_INTERNAL_NAMES)
    assert not (set(LEGACY_AXIS_OPERATION_NAMES) & set(ACTION_SPECS))


@pytest.mark.parametrize("legacy_name", LEGACY_AXIS_OPERATION_NAMES)
def test_legacy_axis_operation_names_are_rejected_from_internal_symmetry_map(legacy_name: str) -> None:
    raw = {
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": "Gamma",
            "mode": "single_valley",
            "active_valleys": ["Gamma"],
            "spin_convention": "spinful",
            "allowed_internal_symmetries": [],
            "external_sewing_symmetries": [],
        },
        "symmetry_source": {"type": "toy_generator", "allow": True, "basis_template": "Gamma_four_orbital"},
        "model": {"symmetry_map": {"Kinect": [{"name": legacy_name}], "intra": [], "inter": []}},
    }

    with pytest.raises(ValueError, match=f"Unsupported internal symmetry operation names.*{legacy_name}"):
        validate_model_config(raw, nlow_state=[2, 2])


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
        "source_resolved_action": {"k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True}},
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
    assert not (set(LEGACY_AXIS_OPERATION_NAMES) & {str(value) for value in row.values()})


def test_K_notebook_profile_is_explicitly_recognized_for_canonical_C2T() -> None:
    valley_model = {"valley_type": "K", "mode": "single_valley", "spin_convention": "spin_up_only"}

    assert _is_k_single_valley_notebook_c2t(
        valley_model,
        {"Kinect": [{"name": "C2T", "model_action_profile": "K_notebook"}]},
    )
    assert not _is_k_single_valley_notebook_c2t(
        valley_model,
        {"Kinect": [{"name": "C2" + "y" + "T", "model_action_profile": "K_notebook"}]},
    )
    assert not _is_k_single_valley_notebook_c2t(valley_model, {"Kinect": [{"name": "C2T"}]})


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
