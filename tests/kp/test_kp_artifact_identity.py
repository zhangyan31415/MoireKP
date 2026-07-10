from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kp.identity import (
    hash_array,
    hash_file,
    hash_mapping,
    require_identity_fields,
    require_matching_identity,
)


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
