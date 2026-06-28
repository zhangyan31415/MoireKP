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


def test_projected_band_plot_can_use_release_style(tmp_path: Path) -> None:
    from kp import cli
    from kp.plot_style import KP_BAND_BOX_ASPECT, KP_BAND_FIGSIZE

    fig, ax = cli.plot_eigs_scatter(
        [
            np.array([-4.6, -4.5]),
            np.array([-4.55, -4.45]),
        ],
        efermi=-4.5,
        out=str(tmp_path / "project.png"),
        figsize=KP_BAND_FIGSIZE,
        box_aspect=KP_BAND_BOX_ASPECT,
        font_family="DejaVu Serif",
        return_fig=True,
    )

    try:
        assert tuple(round(float(item), 6) for item in fig.get_size_inches()) == KP_BAND_FIGSIZE
        assert round(float(ax.get_box_aspect()), 6) == round(KP_BAND_BOX_ASPECT, 6)
        assert ax.get_ylabel() == "Energy - E_F (eV)"
        assert ax.lines[1].get_ydata()[0] == 0.0
        assert ax.title.get_fontfamily()[0] == "DejaVu Serif" or ax.yaxis.label.get_fontfamily()[0] == "DejaVu Serif"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_inspect_plot_combines_kpath_and_qblock_panels(tmp_path: Path) -> None:
    from kp import cli

    fig, axes = cli.plot_inspect_band_and_qblock(
        band_eigs_list=[
            np.array([-0.30, -0.20, -0.10]),
            np.array([-0.28, -0.18, -0.08]),
            np.array([-0.27, -0.17, -0.07]),
        ],
        qblock_eigs_list=[
            np.array([-0.31, -0.21, -0.11]),
            np.array([-0.29, -0.19, -0.09]),
            np.array([-0.26, -0.16, -0.06]),
            np.array([-0.25, -0.15, -0.05]),
        ],
        efermi=-0.18,
        out=str(tmp_path / "inspect.png"),
        q_sector_lengths=[2, 2],
        q_window_bands=2,
        return_fig=True,
    )

    try:
        ax_band, ax_q = axes
        assert len(fig.axes) == 2
        assert ax_band.get_shared_y_axes().joined(ax_band, ax_q)
        assert ax_band.get_title() == "Band path"
        assert ax_q.get_title() == "Q-block diagonalization"
        assert tuple(round(float(item), 6) for item in ax_band.get_ylim()) == (-1.0, 1.0)
        assert round(float(ax_band.get_box_aspect()), 6) == round(5 / 3, 6)
        assert round(float(ax_q.get_box_aspect()), 6) == round(5 / 3, 6)
        q_band_lines = [
            line for line in ax_q.lines
            if line.get_linestyle() == "-" and len(set(line.get_xdata())) > 1
        ]
        assert q_band_lines, "Q-block bands should be rendered as lines with point markers"
        assert {line.get_marker() for line in q_band_lines} == {"o"}
        divider_lines = [
            line for line in ax_q.lines
            if line.get_linestyle() == "--" and len(set(line.get_xdata())) == 1
        ]
        assert divider_lines
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
