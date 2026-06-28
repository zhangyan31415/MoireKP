from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

KP_BAND_FIGSIZE: tuple[float, float] = (3.0, 5.0)
KP_BAND_BOX_ASPECT: float = 5.0 / 3.0
KP_INSPECT_FIGSIZE: tuple[float, float] = (7.4, 6.2)
KP_DPI: int = 220


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
