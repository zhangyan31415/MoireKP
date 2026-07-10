from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kp.identity import (
    build_projection_basis_identity,
    exactified_operation_provenance_is_complete,
    hash_array,
    hash_file,
    hash_mapping,
    require_identity_fields,
    require_matching_identity,
)


def _exactified_operation_record() -> dict:
    return {
        "matrix_kind": "continuum_internal_rep_exact",
        "matrix_source": "kp_symm_exactified_action",
        "status": "exactified",
        "exactification_status": "exactified",
        "exactification_owner": "kp_symm",
        "basis_hash": "basis-a",
        "source_matrix_projection_report": {"report": {"status": "exactified"}},
    }


@pytest.mark.parametrize(
    "field",
    [
        "matrix_kind",
        "matrix_source",
        "status",
        "exactification_status",
        "exactification_owner",
        "basis_hash",
        "source_matrix_projection_report",
    ],
)
def test_exactified_operation_provenance_requires_every_contract_field(field: str) -> None:
    complete = _exactified_operation_record()
    assert exactified_operation_provenance_is_complete(complete)

    incomplete = dict(complete)
    incomplete.pop(field)
    assert not exactified_operation_provenance_is_complete(incomplete)


def test_kp_identity_hashes_are_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    payload = b"moirekp-identity" * 100_000
    first.write_bytes(payload)
    second.write_bytes(payload)

    assert hash_file(first) == hash_file(second)
    second.write_bytes(payload + b"changed")
    assert hash_file(first) != hash_file(second)

    source = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    assert not source.flags.c_contiguous
    assert hash_array(source) == hash_array(np.ascontiguousarray(source))
    assert hash_array(source) != hash_array(source.astype(np.float32))
    assert hash_array(source) != hash_array(source.reshape(2, 6))

    left = {"spin": "up", "q_shell": np.int64(4), "points": [0.0, 0.5]}
    right = {"points": [0.0, 0.5], "q_shell": 4, "spin": "up"}
    assert hash_mapping(left) == hash_mapping(right)
    assert hash_mapping(left) != hash_mapping({**right, "q_shell": 5})


def test_kp_identity_metadata_is_strict() -> None:
    expected = {"input_hash": np.asarray("input-a"), "basis_hash": "basis-a"}
    actual = {"input_hash": "input-a", "basis_hash": np.asarray("basis-a")}

    required = require_identity_fields(expected, ("input_hash", "basis_hash"), "projection")
    assert required == {"input_hash": "input-a", "basis_hash": "basis-a"}
    require_matching_identity(expected, actual, ("input_hash", "basis_hash"), "projection")

    with pytest.raises(KeyError, match="projection.*config_hash"):
        require_identity_fields(expected, ("config_hash",), "projection")
    with pytest.raises(ValueError, match="projection.*basis_hash"):
        require_matching_identity(
            expected,
            {**actual, "basis_hash": "basis-b"},
            ("input_hash", "basis_hash"),
            "projection",
        )


def test_projection_basis_identity_tracks_gauge_and_row_layout(tmp_path: Path) -> None:
    hamk = tmp_path / "hamk.npy"
    np.save(hamk, np.eye(4, dtype=np.complex128)[None, :, :])
    common = {
        "hamk_file": hamk,
        "hamk_fallback": np.eye(4, dtype=np.complex128),
        "qset1": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "qset2": np.array([[0.0, 0.0], [0.0, 1.0]]),
        "spin": "up",
        "mode": "k1",
        "energy_scale": 1.0,
        "nlow_state_list": [[0], [0]],
        "resolved_norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
        "gauge_mode": "manual",
        "num_layer_list": [1, 1],
        "num_orb_per_layer_list": [1, 1],
        "orbital_block_dim": 1,
        "model_dim": 4,
        "k_indices": [0, 2],
    }

    first = build_projection_basis_identity(**common)
    second = build_projection_basis_identity(**common)
    changed_gauge = build_projection_basis_identity(
        **{**common, "resolved_norb_fix_list": [[[[1, 1.0]]], [[[0, 1.0]]]]}
    )
    changed_rows = build_projection_basis_identity(
        **{**common, "num_orb_per_layer_list": [1, 2]}
    )
    changed_k_mapping = build_projection_basis_identity(
        **{**common, "k_indices": [1, 3]}
    )

    assert first == second
    assert len(first["basis_hash"]) == 64
    assert changed_gauge["basis_hash"] != first["basis_hash"]
    assert changed_rows["basis_hash"] != first["basis_hash"]
    assert changed_k_mapping["k_indices_hash"] != first["k_indices_hash"]
    assert changed_k_mapping["basis_hash"] != first["basis_hash"]
