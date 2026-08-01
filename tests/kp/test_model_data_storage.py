from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from kp.model.export import _write_npz


def test_write_npz_uses_lossless_deflate_compression(tmp_path: Path) -> None:
    arrays = {
        "complex_matrix": np.asarray(
            [[1.0 + 2.0j, -3.5j], [4.25, -7.0 - 0.5j]],
            dtype=np.complex128,
        ),
        "integer_vector": np.asarray([0, 2, 5, 9], dtype=np.int64),
        "metadata": np.asarray("standalone-kp-model-v1"),
    }
    output_path = tmp_path / "model_data.npz"

    _write_npz(output_path, arrays)

    with zipfile.ZipFile(output_path) as archive:
        assert archive.namelist()
        assert all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist())

    with np.load(output_path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key, expected in arrays.items():
            actual = loaded[key]
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(actual, expected)
