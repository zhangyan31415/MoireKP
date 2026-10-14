from __future__ import annotations

import numpy as np

from kp.symmetry.projection import _save_exactified_matrix


def test_save_exactified_matrix_preserves_every_route_block_entry(tmp_path) -> None:
    path = tmp_path / "exactified_C3z.npy"
    matrix = np.array(
        [
            [1.0, 5.0e-5 + 2.0e-17j],
            [-3.0e-12j, -1.0],
        ],
        dtype=np.complex128,
    )
    original = matrix.copy()

    saved, report = _save_exactified_matrix(path, matrix)

    np.testing.assert_array_equal(matrix, original)
    np.testing.assert_array_equal(saved, original)
    np.testing.assert_array_equal(np.load(path), original)
    assert report == {
        "policy": "structural_route_support_v1",
        "numeric_entries_modified": 0,
    }


def test_save_exactified_matrix_preserves_structural_zeros(tmp_path) -> None:
    path = tmp_path / "exactified_C2.npy"
    matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )

    saved, _report = _save_exactified_matrix(path, matrix)

    np.testing.assert_array_equal(saved, matrix)
    assert saved[0, 0] == 0.0
    assert saved[1, 1] == 0.0
