import importlib
from pathlib import Path

import numpy as np


def test_save_qgt_outputs_includes_plot_coordinate_columns(tmp_path):
    cp = importlib.import_module("tapw.chern_post")

    fractional_mesh = np.array(
        [
            [[0.0, 0.0], [0.0, 0.5]],
            [[0.5, 0.0], [0.5, 0.5]],
        ],
        dtype=float,
    )
    cartesian_mesh = np.array(
        [
            [[1.0, 2.0], [1.0, 4.0]],
            [[3.0, 2.0], [3.0, 4.0]],
        ],
        dtype=float,
    )
    plot_mesh = np.array(
        [
            [[0.1, 0.2], [0.1, 0.4]],
            [[0.3, 0.2], [0.3, 0.4]],
        ],
        dtype=float,
    )
    qgt_fields = {
        "g11_frac": np.array([[1.0, 1.1], [1.2, 1.3]]),
        "g12_frac": np.array([[0.2, 0.3], [0.4, 0.5]]),
        "g22_frac": np.array([[1.5, 1.6], [1.7, 1.8]]),
        "omega12_frac": np.array([[0.3, 0.4], [0.5, 0.6]]),
        "gxx_cart": np.array([[2.0, 2.1], [2.2, 2.3]]),
        "gxy_cart": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "gyy_cart": np.array([[4.0, 4.1], [4.2, 4.3]]),
        "trace_g_cart": np.array([[6.0, 6.2], [6.4, 6.6]]),
        "omega_xy_cart": np.array([[5.0, 5.1], [5.2, 5.3]]),
    }
    prefix = tmp_path / "qgt_demo"

    cp.save_qgt_outputs(
        str(prefix),
        fractional_mesh,
        cartesian_mesh,
        plot_mesh,
        qgt_fields,
        "Tr g for Band -1 in K1",
        np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float),
        0.5,
        0.5,
    )

    txt_path = Path(str(prefix) + ".txt")
    header_lines = [line for line in txt_path.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert header_lines
    header = header_lines[-1]
    assert "kx_plot" in header
    assert "ky_plot" in header

    table = np.loadtxt(txt_path)
    # Columns 4 and 5 should be the flattened normalized plot coordinates.
    expected_plot = plot_mesh.reshape(-1, 2)
    np.testing.assert_allclose(table[:, 4], expected_plot[:, 0])
    np.testing.assert_allclose(table[:, 5], expected_plot[:, 1])
