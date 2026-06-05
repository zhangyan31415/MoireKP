from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from kp.model.config_schema import canonical_operation_name_for_valley
from kp.model.symmetry import MatrixSymmetryGenerator, load_symmetry_source


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
