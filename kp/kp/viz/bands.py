from __future__ import annotations

import numpy as np


def plot_band_structure(*args, **kwargs):
    """Create a color-mapped band-structure plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(args) < 2:
        raise ValueError("plot_band_structure requires x values and a band array")
    x = np.asarray(args[0], dtype=float)
    bands = np.asarray(args[1], dtype=float)
    if bands.ndim == 1:
        bands = bands[:, np.newaxis]
    if bands.shape[0] != x.shape[0]:
        raise ValueError(f"x length {x.shape[0]} does not match band rows {bands.shape[0]}")
    path = kwargs.pop("path", kwargs.pop("out", None))
    if path is None:
        raise ValueError("plot_band_structure requires path= or out=")
    title = kwargs.pop("title", None)
    ylabel = kwargs.pop("ylabel", "Energy")
    xlabel = kwargs.pop("xlabel", "k-path")
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5.0, 3.2)))
    for band_index in range(bands.shape[1]):
        ax.plot(x, bands[:, band_index], lw=kwargs.get("lw", 1.1), color=kwargs.get("color", "C0"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(str(title))
    ax.grid(True, axis="y", lw=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=int(kwargs.get("dpi", 160)))
    plt.close(fig)
    return str(path)
