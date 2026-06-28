from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

KP_BAND_FIGSIZE: tuple[float, float] = (3.0, 5.0)
KP_BAND_BOX_ASPECT: float = 5.0 / 3.0
KP_INSPECT_FIGSIZE: tuple[float, float] = (7.4, 5.0)
KP_DPI: int = 220
KP_REFERENCE_STYLE = {"color": "0.30", "linewidth": 0.85, "alpha": 0.78, "linestyle": "-"}
KP_PRIMARY_STYLE = {"color": "#1f77b4", "linewidth": 0.95, "alpha": 0.92, "linestyle": "-"}
KP_MODEL_STYLE = {"color": "#1f77b4", "linewidth": 1.0, "alpha": 0.95, "linestyle": "-"}
KP_MARKER_STYLE = {"marker": "o", "markersize": 2.4, "markeredgewidth": 0.0}
KP_LEGEND_KWARGS = {"loc": "best", "frameon": False, "fontsize": 9}


def kp_font_family() -> str:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def apply_kp_axis_style(ax, *, box_aspect: float | None = KP_BAND_BOX_ASPECT, font_family: str | None = None) -> None:
    if box_aspect is not None:
        ax.set_box_aspect(float(box_aspect))
    family = font_family or kp_font_family()
    text_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
    text_items.extend(ax.get_xticklabels())
    text_items.extend(ax.get_yticklabels())
    legend = ax.get_legend()
    if legend is not None:
        text_items.extend(legend.get_texts())
    for item in text_items:
        item.set_fontfamily(family)


@contextmanager
def kp_plot_rc_context() -> Iterator[None]:
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": kp_font_family(), "mathtext.fontset": "stix"}):
        yield
