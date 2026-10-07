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


@pytest.mark.parametrize(
    "field",
    ["q_map", "sector_map", "matrix_kind", "source_gauge", "source_matrix_role", "target_role"],
)
def test_production_operation_requires_strict_action_schema_fields(field: str) -> None:
    record = _record()
    record.pop(field)

    with pytest.raises(ValueError, match=field):
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


def test_load_symmetry_source_root_matrix_kind_does_not_complete_production_operation(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.eye(2, dtype=complex))
    row = _record()
    row.pop("matrix_kind")
    (tmp_path / "manifest.json").write_text(yaml.safe_dump({"operations": [row]}), encoding="utf-8")

    with pytest.raises(ValueError, match="matrix_kind"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path), "matrix_kind": "action"},
            base=tmp_path,
            expected_dim=2,
        )


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


def test_load_symmetry_source_production_rejects_inferred_metadata_marker(tmp_path: Path) -> None:
    np.save(tmp_path / "C2_low_raw.npy", np.eye(2, dtype=complex))
    row = _record(inferred_fields=["q_map"])
    (tmp_path / "manifest.json").write_text(yaml.safe_dump({"operations": [row]}), encoding="utf-8")

    with pytest.raises(ValueError, match="inferred_fields"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path)}, base=tmp_path, expected_dim=2)


def test_load_symmetry_source_canonicalizes_legacy_t_operation_name(tmp_path: Path) -> None:
    np.save(tmp_path / "TR_low_raw.npy", np.eye(2, dtype=complex))
    row = _record(
        name="T",
        operation="T",
        matrix_file="TR_low_raw.npy",
        antiunitary=True,
        k_map={"type": "negation"},
        q_map={"type": "negation"},
        antiunitary_convention="U_K",
    )
    (tmp_path / "manifest.json").write_text(yaml.safe_dump({"operations": [row]}), encoding="utf-8")

    loaded = load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path)}, base=tmp_path, expected_dim=2)

    assert loaded.metadata["operations"][0]["name"] == "TR"
    assert loaded.generator.get_operator("TR").shape == (2, 2)
    with pytest.raises(ValueError, match="T"):
        loaded.generator.get_operator("T")


def test_load_symmetry_source_rejects_exactified_matrix_without_kp_symm_provenance(tmp_path: Path) -> None:
    np.save(tmp_path / "exactified_C2.npy", np.eye(2, dtype=complex))
    row = _record(matrix_file="exactified_C2.npy", matrix_kind="continuum_internal_rep_exact")
    (tmp_path / "manifest.json").write_text(yaml.safe_dump({"operations": [row]}), encoding="utf-8")

    with pytest.raises(ValueError, match="continuum_internal_rep_exact.*kp_symm"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path)}, base=tmp_path, expected_dim=2)


def test_load_symmetry_source_rejects_exactified_matrix_from_non_raw_h_role(tmp_path: Path) -> None:
    np.save(tmp_path / "exactified_C2.npy", np.eye(2, dtype=complex))
    row = _record(
        matrix_file="exactified_C2.npy",
        matrix_kind="continuum_internal_rep_exact",
        matrix_source="kp_symm_exactified_action",
        source_matrix_role="bare_D0_internal_rep",
    )
    (tmp_path / "manifest.json").write_text(
        yaml.safe_dump({"exactification_owner": "kp_symm", "operations": [row]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_h_sewing_action"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path)}, base=tmp_path, expected_dim=2)
