import importlib
from pathlib import Path

import numpy as np


def _demo_mesh():
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
        "omega_xy_cart": np.array([[5.0, -5.1], [5.2, -5.3]]),
    }
    return fractional_mesh, cartesian_mesh, plot_mesh, qgt_fields


def test_save_qgt_outputs_writes_single_band_trace_condition_file(tmp_path):
    cp = importlib.import_module("tapw.chern_post")
    fractional_mesh, cartesian_mesh, plot_mesh, qgt_fields = _demo_mesh()
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
        band_indices=[-1],
    )

    trace_path = Path(str(prefix) + "_trace_condition.txt")
    text = trace_path.read_text(encoding="utf-8")
    assert "mode single_band" in text
    assert "band_indices -1" in text
    assert "# Tr g(k) = g_xx(k) + g_yy(k)" in text
    assert "# <Tr g> = <Tr g(k)>_BZ" in text
    assert "# delta_g = sqrt(<(Tr g - <Tr g>)^2>) / <Tr g>" in text
    assert "delta_tr " in text
    assert "delta_g " in text
    assert "mean_trace_g_scaled " in text
    assert "mean_abs_omega_scaled " in text
    assert "rms_residual_scaled " in text
    assert "max_abs_residual_scaled " in text
    assert "rms_trace_g_fluctuation_scaled " in text
    assert "mean_trace_g_cart" not in text
    assert "delta_tr_subspace" not in text


def test_save_qgt_outputs_writes_subspace_trace_condition_file(tmp_path):
    cp = importlib.import_module("tapw.chern_post")
    fractional_mesh, cartesian_mesh, plot_mesh, qgt_fields = _demo_mesh()
    prefix = tmp_path / "qgt_demo_subspace"

    cp.save_qgt_outputs(
        str(prefix),
        fractional_mesh,
        cartesian_mesh,
        plot_mesh,
        qgt_fields,
        "Tr g for Bands -1_-2 in K1",
        np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float),
        0.5,
        0.5,
        band_indices=[-1, -2],
    )

    trace_path = Path(str(prefix) + "_trace_condition.txt")
    text = trace_path.read_text(encoding="utf-8")
    assert "mode subspace" in text
    assert "band_indices -1,-2" in text
    assert "delta_tr_subspace " in text
    assert "delta_g " in text
    assert "mean_trace_g_scaled " in text
    assert "mean_abs_omega_scaled " in text
    assert "rms_residual_scaled " in text
    assert "max_abs_residual_scaled " in text
    assert "rms_trace_g_fluctuation_scaled " in text
    assert "mean_trace_g_cart" not in text
    assert (
        "warning This is a subspace trace-condition diagnostic, not a single-band ideal-Chern-band diagnostic."
        in text
    )
