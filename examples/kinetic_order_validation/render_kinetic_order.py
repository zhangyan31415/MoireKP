#!/usr/bin/env python3
"""Draw Figure R2 from the archived kinetic-order sweep (RMS and Max errors vs m_kin)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np


CASES = (
    ("mote2_k", r"Bilayer MoTe$_2$", r"$K$, $3.89^\circ$"),
    ("zrs2_m", r"Bilayer ZrS$_2$", r"$M_1$, $3.89^\circ$"),
    ("zrs2_gamma", r"Bilayer ZrS$_2$", r"$\Gamma$, $3.89^\circ$"),
    ("aab_ka", r"Trilayer MoTe$_2$", r"$K_A$, $5.09^\circ$"),
    ("aab_kb", r"Trilayer MoTe$_2$", r"$K_B$, $5.09^\circ$"),
    ("aab_gamma", r"Trilayer MoTe$_2$", r"$\Gamma$, $5.09^\circ$"),
)
ORDERS = (4, 6, 8, 10)
BAND_COUNTS = (8, 8, 10, 6, 3, 8)


def read_curves(root: Path):
    data = json.loads((root / "data/summary.json").read_text())
    curves = {}
    with np.load(root / "data/path_bands.npz", allow_pickle=False) as archive:
        for (case, _, _), count in zip(CASES, BAND_COUNTS):
            group = 2 if case == "zrs2_gamma" else 0
            records = {r["kinetic_order"]: r for r in data["groups"][group]["rows"]
                       if r["case"] == case}
            rms, maximum = [], []
            for order in ORDERS:
                prefix = f"s{group}_{case}_k{order}"
                delta = archive[prefix + "_model"] - archive[prefix + "_tapw"]
                assert delta.shape == (61, count)
                row = records[order]
                delta = 1000 * delta + row["paper_alignment_shift_mev"]
                rms.append(float(np.sqrt(np.mean(delta**2))))
                maximum.append(float(np.max(np.abs(delta))))
                reference = row["native_paper_aligned"]
                np.testing.assert_allclose([rms[-1], maximum[-1]],
                    [reference["rms_mev"], reference["max_mev"]], atol=1e-8, rtol=0)
            curves[case] = rms, maximum, data["stars"][case]["kinetic_order"]
    return curves


def draw(curves, out: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.1,
        "axes.edgecolor": "#4D535A",
        "axes.linewidth": 0.58,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "ytick.minor.size": 1.3,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 6, figsize=(6.55, 2.22), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.082, right=0.994, bottom=0.265, top=0.775, wspace=0.25)
    for idx, (ax, (case, system, valley)) in enumerate(zip(axes, CASES)):
        rms_list, max_list, original = curves[case]
        ax.plot(
            ORDERS, max_list, color="#8A99A8", lw=0.90, ls="--",
            marker="s", ms=2.5, mfc="white", mec="#738291", mew=0.65, zorder=2,
        )
        ax.plot(
            ORDERS, rms_list, color="#264E78", lw=1.15,
            marker="o", ms=3.1, zorder=3,
        )
        ax.plot(
            original, rms_list[ORDERS.index(original)],
            marker="*", ms=8.5, mfc="#F0CC52", mec="#20242A", mew=0.58, zorder=4,
        )
        ax.set_title(f"{system}\n{valley}", pad=4.5, fontsize=6.85)
        ax.text(-0.14, 1.16, f"({chr(97 + idx)})", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5)
        ax.set(yscale="log", xlim=(3.4, 10.6), ylim=(0.42, 240.0), xticks=ORDERS)
        ax.set_yticks([1, 10, 100], labels=["1", "10", "100"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", which="major", color="#DADEE3", lw=0.5)
        ax.tick_params(which="major", direction="in", top=True, right=True, pad=2.0)
        ax.tick_params(axis="y", which="minor", direction="in", right=True)

    axes[0].set_ylabel("Band error vs. TAPW (meV)")
    fig.text(0.36, 0.055, r"Channel-diagonal kinetic expansion order $m_{\mathrm{kin}}$", ha="center", va="center", fontsize=7.6)

    h_rms = Line2D([], [], color="#264E78", lw=1.15, marker="o", ms=3.1, label="RMS error")
    h_max = Line2D([], [], color="#8A99A8", lw=0.90, ls="--", marker="s", ms=2.5, mfc="white", mec="#738291", mew=0.65, label="Max. error")
    h_star = Line2D([], [], linestyle="none", marker="*", ms=7.8, mfc="#F0CC52", mec="#20242A", mew=0.58, label="Selected order")
    fig.legend(
        handles=[h_rms, h_max, h_star],
        loc="lower right",
        bbox_to_anchor=(0.996, 0.015),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
        handlelength=1.6,
        borderaxespad=0.0,
        fontsize=6.6,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    draw(read_curves(root), root / "figures/kinetic_order.pdf")
