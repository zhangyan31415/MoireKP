from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from kp.model.symmetry import load_symmetry_source
from kp.symmetry.action_schema import complete_action_operation_metadata


def _record(**overrides):
    data = {
        "name": "C2",
        "operation": "C2",
        "matrix_file": "C2_low_raw.npy",
        "matrix_kind": "action",
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0},
        "q_map": {"type": "reflection", "axis_deg": 0.0},
        "sector_map": "identity",
        "source_matrix_role": "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "continuum_internal_rep",
        "gauge_correction": {"kind": "none"},
        "antiunitary_convention": "none",
        "spin_map": "from_kp_symm_output",
        "valley_map": "identity",
    }
    data.update(overrides)
    return data


def test_production_operation_requires_explicit_matrix_kind() -> None:
    record = _record()
    record.pop("matrix_kind")

    with pytest.raises(ValueError, match="matrix_kind"):
        complete_action_operation_metadata(record)


def test_production_operation_does_not_infer_q_map_from_k_map() -> None:
    record = _record()
    record.pop("q_map")

    with pytest.raises(ValueError, match="q_map"):
        complete_action_operation_metadata(record)


def test_diagnostic_mode_records_inferred_and_defaulted_fields() -> None:
    record = _record()
    for key in ("matrix_kind", "q_map", "source_matrix_role", "source_gauge", "target_role", "gauge_correction"):
        record.pop(key)

    completed = complete_action_operation_metadata(record, allow_inferred=True)

    assert completed["matrix_kind"] == "action"
    assert completed["q_map"] == completed["k_map"]
    assert completed["inferred_fields"] == ["q_map"]
    assert set(completed["defaulted_fields"]) >= {
        "matrix_kind",
        "source_matrix_role",
        "source_gauge",
        "target_role",
        "gauge_correction",
    }


def test_load_symmetry_source_missing_operation_matrix_kind_raises(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.eye(2, dtype=complex))
    row = _record()
    row.pop("matrix_kind")
    (tmp_path / "manifest.json").write_text(yaml.safe_dump({"operations": [row]}), encoding="utf-8")

    with pytest.raises(ValueError, match="matrix_kind"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path)}, base=tmp_path, expected_dim=2)


def test_load_symmetry_source_diagnostic_allows_inferred_q_map(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.eye(2, dtype=complex))
    row = _record()
    row.pop("q_map")
    (tmp_path / "manifest.json").write_text(yaml.safe_dump({"operations": [row]}), encoding="utf-8")

    loaded = load_symmetry_source(
        {
            "type": "kp_symm_output",
            "path": str(tmp_path),
            "mode": "diagnostic",
        },
        base=tmp_path,
        expected_dim=2,
    )

    operation = loaded.metadata["operations"][0]
    assert operation["q_map"] == operation["k_map"]
    assert operation["inferred_fields"] == ["q_map"]
