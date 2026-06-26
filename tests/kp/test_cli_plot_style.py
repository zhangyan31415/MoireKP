from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_projected_band_plot_uses_single_heff_color(tmp_path: Path) -> None:
    from kp import cli

    fig, ax = cli.plot_eigs_scatter(
        [
            np.array([-0.30, -0.20, -0.10]),
            np.array([-0.28, -0.18, -0.08]),
            np.array([-0.27, -0.17, -0.07]),
        ],
        efermi=0.0,
        out=str(tmp_path / "bands.png"),
        original_eigs_list=[
            np.array([-0.31, -0.21, -0.11]),
            np.array([-0.29, -0.19, -0.09]),
            np.array([-0.26, -0.16, -0.06]),
        ],
        return_fig=True,
    )

    try:
        heff_lines = [line for line in ax.lines if line.get_zorder() == 2]
        assert len(heff_lines) == 3
        assert {line.get_color() for line in heff_lines} == {"#1f77b4"}
        assert {line.get_marker() for line in heff_lines} == {"None"}
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_projected_band_plot_can_show_top_bands_aligned_to_zero(tmp_path: Path) -> None:
    from kp import cli

    fig, ax = cli.plot_eigs_scatter(
        [
            np.array([-0.40, -0.30, -0.20, -0.10]),
            np.array([-0.35, -0.25, -0.15, -0.05]),
            np.array([-0.45, -0.33, -0.22, -0.12]),
        ],
        efermi=None,
        out=str(tmp_path / "top.png"),
        top_bands=2,
        align="top",
        xlabel="k-path point",
        return_fig=True,
    )

    try:
        heff_lines = [line for line in ax.lines if line.get_zorder() == 2]
        assert len(heff_lines) == 2
        assert max(float(np.max(line.get_ydata())) for line in heff_lines) == 0.0
        assert ax.get_xlabel() == "k-path point"
        assert ax.get_ylabel() == r"$E - E_{\mathrm{top}}$ (eV)"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
