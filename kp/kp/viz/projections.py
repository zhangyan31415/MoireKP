from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_orbital_scatter_multi(
    band_indices: Sequence[int],
    H_GM_diag_eig_vec: Sequence[np.ndarray],
    Q_set1: np.ndarray,
    orbit_labels: Sequence[str] | None,
    index_1: Sequence[int],
    *,
    max_id: int = -1,
    cmap_name: str = "coolwarm_r",
):
    """Plot multiple orbital-scatter panels in a row (ported from notebook).

    Parameters
    - band_indices: list of band indices to visualize
    - H_GM_diag_eig_vec: list/array of eigenvector matrices (per Q block)
    - Q_set1: (N, 2) or (N, 3) Q-point coordinates
    - orbit_labels: optional labels for orbitals (unused in default plot)
    - index_1: indices of Q points to plot
    - max_id: index selector in sorted orbital weights (default -1: strongest)
    - cmap_name: matplotlib colormap name
    """

    n = len(band_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    max_possible = H_GM_diag_eig_vec[0].shape[0] - 1
    norm = plt.Normalize(vmin=0, vmax=max_possible)
    cmap = plt.get_cmap(cmap_name)

    q_xy = Q_set1[:, :2]

    for ax, band_index in zip(axes, band_indices):
        for idx in index_1:
            vec = H_GM_diag_eig_vec[idx][:, band_index]
            vec_abs = np.abs(vec) ** 2
            sorted_indices = np.argsort(vec_abs)
            mi = int(sorted_indices[max_id])

            # symmetric color mapping around center
            mid = max_possible / 2
            val = mi if mi < mid else mid + (max_possible - mi)
            color = cmap(norm(val))

            xy = q_xy[idx % len(q_xy)]
            # size encodes weight (quadratic as in notebook)
            size = float(vec_abs[mi] ** 2 * 5000)
            alpha = float(vec_abs[mi] / (np.max(vec_abs) if np.max(vec_abs) > 0 else 1.0) * 0 + 0.9)

            ax.scatter(xy[0], xy[1], marker="o", s=size, color=color, alpha=alpha, linewidth=0.5)
            ax.annotate(f"{mi}", (xy[0], xy[1]), fontsize=12, ha="center", va="center")

            # radius ring at |Q|
            r = float(np.linalg.norm(xy))
            circle = plt.Circle((0, 0), r, color=cmap(norm(r)), alpha=0.2, fill=False, linewidth=0.5, linestyle="--")
            ax.add_artist(circle)

            ax.annotate(
                f"{vec_abs[mi]:.2f}", (xy[0] + 0.01, xy[1] + 0.01), fontsize=8, color="black", ha="center", va="center"
            )

        y_max = float(np.max(np.linalg.norm(q_xy, axis=1)) * 1.1) if len(q_xy) else 1.0
        ax.set_xlim(-y_max, y_max)
        ax.set_ylim(-y_max, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Band index: {band_index}")

    plt.tight_layout()
    plt.show()


def plot_orbital_scatter(
    band_index: int,
    H_GM_diag_eig_vec: Sequence[np.ndarray],
    Q_set1: np.ndarray,
    orbit_labels: Sequence[str] | None,
    index_1: Sequence[int],
    *,
    max_id: int = -1,
    cmap_name: str = "coolwarm_r",
):
    """Single-panel convenience wrapper around plot_orbital_scatter_multi."""

    plot_orbital_scatter_multi(
        [band_index], H_GM_diag_eig_vec, Q_set1, orbit_labels, index_1, max_id=max_id, cmap_name=cmap_name
    )
