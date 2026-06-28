from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml
import numpy as np

from .io.tapw_loader import load_hamk, load_Q_sets
from .orbitals import (
    expand_orbital_order_by_sector,
    expand_orbital_order_pattern as _expand_orbital_order_pattern,
)
from .blocks import (
    get_H_block,
    project_heff_full,
    resolve_project_gauge_anchors,
    set_projector_blas_threads,
)
from .basis.selection import GaugeAnchorReport, write_basis_selection_report
# Reporting-only downfold helpers were removed from the active core. Keep the
# old imports here as a reference while the workflow is simplified.
# from .blocks.downfold import (
#     compute_top_n_metrics,
#     print_top_n_metrics,
#     write_sweep_csv,
#     write_sweep_json,
#     SweepRow,
# )
from .symmetry.projection import run_symmetry_projection_from_config
from .config.case import normalize_case_config

HARTREE_TO_EV = 27.2113845


def _is_canonical_case_config(cfg: dict[str, Any]) -> bool:
    case = cfg.get("case")
    if not isinstance(case, dict):
        return False
    return all(case.get(key) not in (None, "") for key in ("profile", "q_shell", "output_root"))


def _config_path_uses_canonical_case(config_path: str | Path) -> bool:
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = normalize_case_config(yaml.safe_load(handle), config_path=config_path)
        return _is_canonical_case_config(cfg)
    except Exception:
        return False


def _write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _write_case_summary(cfg: dict[str, Any], workflow_dir: str | Path, workflow: str) -> None:
    if not _is_canonical_case_config(cfg):
        return
    workflow_path = Path(workflow_dir)
    case_dir = workflow_path.parent
    summary = case_dir / "summary.md"
    case = cfg.get("case", {})
    lines = [
        "# KP Case Summary",
        "",
        f"- profile: `{case.get('profile')}`",
        f"- q_shell: `{case.get('q_shell')}`",
        f"- updated_workflow: `{workflow}`",
        "",
        "## Workflows",
    ]
    for name in ("inspect", "projection", "symmetry", "model"):
        status = "ready" if (case_dir / name).exists() else "pending"
        lines.append(f"- {name}: {status}")
    _write_text(summary, "\n".join(lines) + "\n")

    root = case_dir.parent.parent
    manifest = root / "manifest.yaml"
    profile_dir = case_dir.parent
    targets = []
    if root.exists():
        for profile in sorted(p for p in root.iterdir() if p.is_dir()):
            for q_shell in sorted(q for q in profile.iterdir() if q.is_dir()):
                targets.append(
                    {
                        "profile": profile.name,
                        "q_shell": q_shell.name,
                        "path": f"{profile.name}/{q_shell.name}",
                    }
                )
    _write_text(
        manifest,
        yaml.safe_dump(
            {
                "schema": "kp_outputs/v1",
                "updated_workflow": workflow,
                "targets": targets
                or [
                    {
                        "profile": profile_dir.name,
                        "q_shell": case_dir.name,
                        "path": f"{profile_dir.name}/{case_dir.name}",
                    }
                ],
            },
            sort_keys=False,
        ),
    )


def _write_inspect_sidecars(
    cfg: dict[str, Any],
    *,
    output_dir: str | Path,
    eigs_list: Sequence[np.ndarray],
    efermi: float,
    ref_q_index: int,
) -> None:
    if not _is_canonical_case_config(cfg):
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_case_summary(cfg, out, "inspect")


def _default_standalone_export_dir(model_output_dir: Path) -> Path:
    return model_output_dir / "standalone"


def _record_standalone_export(model_output_dir: Path, export_path: Path) -> None:
    summary_path = model_output_dir / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    if not isinstance(summary, dict):
        summary = {}
    model_output_resolved = model_output_dir.resolve()
    export_resolved = export_path.resolve()
    try:
        summary["standalone_export"] = str(export_resolved.relative_to(model_output_resolved))
    except ValueError:
        summary["standalone_export"] = str(export_resolved)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _energy_scale_from_material(material: dict[str, Any]) -> float:
    unit = material.get("energy_unit", "eV")
    normalized = str(unit).strip().lower().replace("_", "").replace("-", "")
    if normalized in {"ev", "electronvolt", "electronvolts"}:
        return 1.0
    if normalized in {"hartree", "hartrees", "ha"}:
        return HARTREE_TO_EV
    raise ValueError(f"material.energy_unit must be 'eV' or 'Hartree', got {unit!r}")


def _load_hamk_with_energy_unit(path: str, material: dict[str, Any], *, mmap_mode: str | None = "r") -> np.ndarray:
    hamk = load_hamk(path, mmap_mode=mmap_mode)
    scale = _energy_scale_from_material(material)
    if scale == 1.0:
        return hamk
    return hamk * scale


def _selected_spin_block_for_projection(hamk2d: np.ndarray, spin: str) -> tuple[np.ndarray, str]:
    spin_norm = str(spin).lower()
    if spin_norm == "all":
        return hamk2d, "all"
    if spin_norm not in {"up", "down"}:
        raise ValueError(f"Unsupported spin value {spin!r}; expected 'up', 'down', or 'all'")
    if hamk2d.ndim != 2 or hamk2d.shape[0] != hamk2d.shape[1]:
        raise ValueError(f"Spin projection requires a square Hamiltonian block, got shape={hamk2d.shape}")
    if hamk2d.shape[0] % 2 != 0:
        raise ValueError(f"Spin projection requires an even Hamiltonian dimension, got {hamk2d.shape[0]}")
    half = hamk2d.shape[0] // 2
    if spin_norm == "up":
        return hamk2d[:half, :half], "up"
    return hamk2d[half:, half:], "up"


def _selected_spin_project_input(hamk2d: np.ndarray, spin: str) -> np.ndarray:
    block, _ = _selected_spin_block_for_projection(hamk2d, spin)
    return np.asarray(block, dtype=np.complex128)


def load_efermi_from_vbm_txt(path: str) -> float:
    """Load VBM text file and return max energy as efermi (in eV)."""
    vals: List[float] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    continue
    if not vals:
        raise ValueError(f"No numeric values found in {path}")
    return max(vals)


def _plot_report_layer_for_ref_q(
    ref_q: int,
    *,
    num_layer_list: Sequence[int],
    q_lengths: Sequence[int],
    index_order: Sequence[int] | np.ndarray | None,
) -> int:
    """Return the physical layer for a non-gamma plot reference block."""
    if index_order is not None:
        if ref_q < 0 or ref_q >= len(index_order):
            raise IndexError(f"ref_q_index {ref_q} is outside ordered block range 0..{len(index_order) - 1}")
        block_index = int(index_order[ref_q])
    else:
        block_index = int(ref_q)
    if block_index < 0:
        raise IndexError(f"ref_q_index maps to negative block index {block_index}")

    offset = 0
    physical_layer_offset = 0
    for group_index, n_layers_raw in enumerate(num_layer_list):
        if group_index >= len(q_lengths):
            raise ValueError(
                f"q_lengths has {len(q_lengths)} entries but num_layer_list needs group {group_index}"
            )
        n_layers = int(n_layers_raw)
        q_len = int(q_lengths[group_index])
        if n_layers < 0 or q_len <= 0:
            raise ValueError(f"invalid non-gamma layout: n_layers={n_layers}, q_len={q_len}")

        group_block_count = n_layers * q_len
        if block_index < offset + group_block_count:
            local_index = block_index - offset
            return physical_layer_offset + int(local_index // q_len)

        offset += group_block_count
        physical_layer_offset += n_layers

    raise IndexError(f"ref_q_index maps to block {block_index}, outside non-gamma block count {offset}")


def infer_orbitals_per_layer(
    hamk2d: np.ndarray,
    q_count: int,
    num_layers: int,
    *,
    spin: str,
    mode: str,
) -> int:
    """Infer number of orbitals per layer per Q for a single spin channel."""
    N = hamk2d.shape[0]
    # if mode == "gamma":
    #     N = N // 2
    # if spin != "all":
    #     N = N // 2
    N = N // 2 # spin
    base = N
    if base % (q_count * num_layers) != 0:
        raise ValueError(
            f"Cannot infer orbitals: base={base}, q_count={q_count}, num_layers={num_layers}"
        )
    return base // (q_count * num_layers)


def _num_layer_list_from_material(material: dict[str, Any]) -> list[int]:
    raw = material.get("num_layer_list")
    if raw is not None:
        layers = [int(x) for x in raw]
        if not layers or any(x <= 0 for x in layers):
            raise ValueError(f"material.num_layer_list must contain positive integers, got {raw!r}")
        return layers
    return [1 for _ in range(int(material.get("num_layers", 2)))]


def _orbital_layout_from_material(
    material: dict[str, Any],
    hamk2d: np.ndarray,
    q_count: int,
    *,
    spin: str,
    mode: str,
) -> tuple[list[int], int, list[list[int]]]:
    num_layer_list = _num_layer_list_from_material(material)
    total_layers = sum(num_layer_list)
    raw = material.get("num_orb_per_layer")
    if raw:
        vals = [int(x) for x in raw]
        if len(vals) == 1:
            orb0 = vals[0]
            return num_layer_list, orb0, [[orb0 for _ in range(n)] for n in num_layer_list]
        if len(vals) == len(num_layer_list):
            if len(set(vals)) != 1:
                raise ValueError("This release requires equal orbital counts for each physical layer.")
            orb0 = vals[0]
            return num_layer_list, orb0, [[vals[i] for _ in range(n)] for i, n in enumerate(num_layer_list)]
        if len(vals) == total_layers:
            if len(set(vals)) != 1:
                raise ValueError("This release requires equal orbital counts for each physical layer.")
            orb0 = vals[0]
            out: list[list[int]] = []
            pos = 0
            for n in num_layer_list:
                out.append(vals[pos:pos + n])
                pos += n
            return num_layer_list, orb0, out
        raise ValueError(
            "material.num_orb_per_layer must have length 1, len(num_layer_list), "
            f"or sum(num_layer_list); got {len(vals)} values for {num_layer_list}"
        )
    orb0 = infer_orbitals_per_layer(hamk2d, q_count, total_layers, spin=spin, mode=mode)
    return num_layer_list, orb0, [[orb0 for _ in range(n)] for n in num_layer_list]


def plot_eigs_scatter(
    eigs_list: Sequence[np.ndarray],
    efermi: float | None,
    *,
    out: str,
    # target: str = "valence",
    # ref_q_index: int = 0,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    index_order: Sequence[int] | None = None,
    original_eigs_list: Sequence[np.ndarray] | None = None,
    top_bands: int | None = None,
    bottom_bands: int | None = None,
    band_slice: Sequence[int] | None = None,
    align: str = "fermi",
    x_values: Sequence[float] | None = None,
    x_ticks: Sequence[float] | None = None,
    x_ticklabels: Sequence[str] | None = None,
    xlabel: str = "k-path point",
    figsize: tuple[float, float] | list[float] | None = None,
    box_aspect: float | None = None,
    font_family: str | None = None,
    return_fig: bool = False,
):
    """Plot all bands as lines across Q (2nd dim is band index).

    If index_order is provided, reorder Q indices before plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stack_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
        arrs = [np.asarray(ev, dtype=float).ravel() for ev in rows]
        if not arrs:
            raise ValueError("No band rows to plot")
        width = arrs[0].shape[0]
        for arr in arrs:
            if arr.shape[0] != width:
                raise ValueError("Band rows have inconsistent widths")
        return np.stack(arrs, axis=0)

    # Stack into (num_Q, num_bands)
    E = stack_rows(eigs_list)
    if index_order is not None:
        idx = np.asarray(index_order, dtype=int)
        E = E[idx]
    E0 = None
    if original_eigs_list is not None:
        E0 = stack_rows(original_eigs_list)
        if index_order is not None:
            idx = np.asarray(index_order, dtype=int)
            E0 = E0[idx]

    def select_window(arr: np.ndarray) -> np.ndarray:
        nbands = int(arr.shape[1])
        if band_slice is not None:
            if len(band_slice) != 2:
                raise ValueError(f"band_slice must have two entries, got {band_slice!r}")
            start = max(0, int(band_slice[0]))
            stop = min(nbands, int(band_slice[1]))
        elif top_bands is not None:
            count = max(1, int(top_bands))
            start = max(0, nbands - count)
            stop = nbands
        elif bottom_bands is not None:
            count = max(1, int(bottom_bands))
            start = 0
            stop = min(nbands, count)
        else:
            start = 0
            stop = nbands
        if stop <= start:
            raise ValueError(f"Empty band plotting window start={start}, stop={stop}, nbands={nbands}")
        return arr[:, start:stop]

    E = select_window(E)
    if E0 is not None:
        E0 = select_window(E0)

    x = np.asarray(x_values, dtype=float) if x_values is not None else np.arange(E.shape[0], dtype=float)
    if x.shape[0] != E.shape[0]:
        raise ValueError(f"x-axis length {x.shape[0]} does not match band rows {E.shape[0]}")
    fig, ax = plt.subplots(figsize=tuple(figsize) if figsize is not None else (5.6, 9.0))
    if box_aspect is not None:
        ax.set_box_aspect(float(box_aspect))
    align_key = str(align or "fermi").strip().lower()

    def shift_for(arr: np.ndarray) -> float:
        if align_key in {"top", "top_band", "top-band"}:
            return float(np.max(arr[:, -1]))
        if align_key in {"bottom", "bottom_band", "bottom-band"}:
            return float(np.min(arr[:, 0]))
        if align_key in {"fermi", "ef", "efermi"}:
            return float(efermi) if (efermi is not None) else 0.0
        if align_key in {"none", "false", "0"}:
            return 0.0
        raise ValueError(f"Unsupported band plot align={align!r}; expected 'top', 'bottom', 'fermi', or 'none'")

    shift = shift_for(E)
    original_shift = shift_for(E0) if E0 is not None else shift
    if E0 is not None:
        for i in range(E0.shape[1]):
            ax.plot(
                x,
                E0[:, i] - original_shift,
                color="#9AA0A6",
                lw=0.7,
                alpha=0.45,
                zorder=1,
                label="Original" if i == 0 else "_nolegend_",
            )
    heff_color = "#1f77b4"
    for i in range(E.shape[1]):
        ax.plot(
            x,
            E[:, i] - shift,
            lw=0.95,
            alpha=0.95,
            color=heff_color,
            zorder=2,
            label="Heff" if i == 0 else "_nolegend_",
        )

    ax.axhline(0.0, color="#555555", lw=0.8, ls="--", alpha=0.8, zorder=0)
    if x_ticks is not None and x_ticklabels is not None and len(x_ticks) == len(x_ticklabels):
        ax.set_xticks([float(item) for item in x_ticks])
        ax.set_xticklabels([str(item) for item in x_ticklabels])
        for tick in x_ticks:
            ax.axvline(float(tick), color="0.86", linewidth=0.75, zorder=0)
        ax.set_xlim(float(x[0]), float(x[-1]))
    else:
        ax.set_xlabel(xlabel)
    if align_key in {"top", "top_band", "top-band"}:
        ax.set_ylabel(r"$E - E_{\mathrm{top}}$ (eV)")
    elif align_key in {"bottom", "bottom_band", "bottom-band"}:
        ax.set_ylabel(r"$E - E_{\mathrm{bottom}}$ (eV)")
    elif align_key in {"fermi", "ef", "efermi"} and efermi is not None:
        ax.set_ylabel("Energy - E_F (eV)")
    else:
        ax.set_ylabel("Energy (eV)")
    ax.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.65)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if E0 is not None:
        ax.legend(loc="best", frameon=False, fontsize=9)
    if font_family:
        text_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        text_items.extend(ax.get_xticklabels())
        text_items.extend(ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            text_items.extend(legend.get_texts())
        for item in text_items:
            item.set_fontfamily(font_family)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=220)
    if return_fig:
        return fig, ax
    plt.close(fig)
    return None


def _stack_band_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    arrs = [np.asarray(ev, dtype=float).ravel() for ev in rows]
    if not arrs:
        raise ValueError("No band rows to plot")
    width = arrs[0].shape[0]
    for arr in arrs:
        if arr.shape[0] != width:
            raise ValueError("Band rows have inconsistent widths")
    return np.stack(arrs, axis=0)


def _fermi_window_band_indices(
    rows: Sequence[np.ndarray],
    *,
    efermi: float,
    ref_q_index: int,
    count: int,
) -> list[int]:
    E = _stack_band_rows(rows)
    ref = max(0, min(int(ref_q_index), E.shape[0] - 1))
    nbands = int(E.shape[1])
    width = max(1, min(int(count), nbands))
    order = np.argsort(np.abs(E[ref] - float(efermi)))
    return sorted(int(item) for item in order[:width])


def _default_inspect_relative_ylim() -> tuple[float, float]:
    return (-1.0, 1.0)


def _inspect_plot_font_family() -> str:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def plot_inspect_band_and_qblock(
    band_eigs_list: Sequence[np.ndarray],
    qblock_eigs_list: Sequence[np.ndarray],
    efermi: float,
    *,
    out: str,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    q_index_order: Sequence[int] | None = None,
    q_sector_lengths: Sequence[int] | None = None,
    ref_q_index: int = 0,
    q_window_bands: int = 10,
    k_x_values: Sequence[float] | None = None,
    k_x_ticks: Sequence[float] | None = None,
    k_x_ticklabels: Sequence[str] | None = None,
    return_fig: bool = False,
):
    """Plot normal k-path bands beside Q-block diagonalization for inspect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_cfg = {"font.family": _inspect_plot_font_family(), "mathtext.fontset": "stix"}
    with plt.rc_context(font_cfg):
        E_band = _stack_band_rows(band_eigs_list)
        E_q = _stack_band_rows(qblock_eigs_list)
        if q_index_order is not None:
            idx = np.asarray(q_index_order, dtype=int)
            E_q = E_q[idx]

        q_band_indices = _fermi_window_band_indices(
            [row for row in E_q],
            efermi=efermi,
            ref_q_index=ref_q_index,
            count=q_window_bands,
        )
        if ylim is None:
            ylim = _default_inspect_relative_ylim()

        fig, (ax_band, ax_q) = plt.subplots(
            1,
            2,
            figsize=(7.4, 6.2),
            sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.08},
        )

        x_band = (
            np.asarray(k_x_values, dtype=float)
            if k_x_values is not None
            else np.arange(E_band.shape[0], dtype=float)
        )
        if x_band.shape[0] != E_band.shape[0]:
            raise ValueError(f"k-path axis length {x_band.shape[0]} does not match band rows {E_band.shape[0]}")
        for band_index in range(E_band.shape[1]):
            ax_band.plot(x_band, E_band[:, band_index] - efermi, color="#222222", lw=0.75, alpha=0.82)
        if k_x_ticks is not None and k_x_ticklabels is not None and len(k_x_ticks) == len(k_x_ticklabels):
            ax_band.set_xticks([float(item) for item in k_x_ticks])
            ax_band.set_xticklabels([str(item) for item in k_x_ticklabels])
            for tick in k_x_ticks:
                ax_band.axvline(float(tick), color="0.84", linewidth=0.7, zorder=0)
            if x_band.size:
                ax_band.set_xlim(float(x_band[0]), float(x_band[-1]))
        else:
            ax_band.set_xlabel("k-path point")
        ax_band.set_title("Band path")

        x_q = np.arange(E_q.shape[0], dtype=float)
        for band_index in q_band_indices:
            ax_q.plot(
                x_q,
                E_q[:, band_index] - efermi,
                marker="o",
                markersize=2.5,
                color="#1f77b4",
                lw=0.75,
                alpha=0.9,
            )
        if q_sector_lengths:
            cumulative = 0
            for sector_index, length in enumerate(q_sector_lengths[:-1], start=1):
                cumulative += int(length)
                if 0 < cumulative < E_q.shape[0]:
                    ax_q.axvline(float(cumulative) - 0.5, color="#555555", lw=0.9, ls="--", alpha=0.8)
                    ax_q.text(
                        float(cumulative) - 0.5,
                        0.98,
                        f"sector {sector_index + 1}",
                        transform=ax_q.get_xaxis_transform(),
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="#555555",
                        rotation=90,
                    )
        ax_q.set_xlabel("Q block index")
        ax_q.set_title("Q-block diagonalization")

        for ax in (ax_band, ax_q):
            ax.set_box_aspect(5 / 3)
            ax.axhline(0.0, color="#555555", lw=0.8, ls=":", alpha=0.8, zorder=0)
            ax.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.65)
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
        ax_band.set_ylabel("Energy - E_F (eV)")
        ax_q.tick_params(labelleft=False)
        ax_band.set_ylim(ylim)
        if title:
            fig.suptitle(title, y=0.995)
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.90 if title else 0.94, wspace=0.08)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=220)
        if return_fig:
            return fig, (ax_band, ax_q)
        plt.close(fig)
    return None


def save_spectrum_txt(eigs_list: Sequence[np.ndarray], path: str, *, index_order: Sequence[int] | None = None) -> None:
    """Save full spectrum per Q to a txt file, one line per Q.

    If index_order is provided, reorder Q indices before saving.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [np.asarray(ev) for ev in eigs_list]
    if index_order is not None:
        idx = np.asarray(index_order, dtype=int)
        rows = [rows[i] for i in idx]
    with open(path, "w") as f:
        for evs in rows:
            f.write(" ".join(f"{x:.10f}" for x in evs) + "\n")


def _load_bands_from_text(path: str) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    continue
            if vals:
                rows.append(np.array(vals, dtype=float))
    if not rows:
        raise ValueError(f"No numeric rows found in band file: {path}")
    return rows


def _normalize_kpath_tick_label(label: Any) -> str:
    text = str(label).strip()
    if text.lower() in {"gamma", "gam", "g"}:
        return r"$\Gamma$"
    return text


def _kpath_axis_from_config(
    cfg: dict[str, Any],
    *,
    cfg_dir: str,
    project_indices: Sequence[int],
    row_count: int,
) -> dict[str, Any]:
    kpath_cfg = cfg.get("kpath", {})
    if not isinstance(kpath_cfg, dict):
        return {}
    file_raw = kpath_cfg.get("file")
    tmat_raw = kpath_cfg.get("tmat")
    if file_raw is None or tmat_raw is None:
        return {}
    path = Path(str(file_raw))
    if not path.is_absolute():
        path = Path(cfg_dir) / path
    try:
        from .model.core import generate_kpath_from_file

        kpath = generate_kpath_from_file(
            Tmat=np.asarray(tmat_raw, dtype=float),
            file_path=path,
            phase_deg=float(kpath_cfg.get("phase_deg", 0.0)),
            segment_points=kpath_cfg.get("segment_points"),
        )
        x_all = np.asarray(kpath.x, dtype=float)
        if x_all.shape[0] == row_count:
            x_values = x_all
            full_path = True
        else:
            indices = np.asarray(project_indices, dtype=int)
            if x_all.shape[0] <= int(np.max(indices, initial=-1)):
                return {}
            x_values = x_all[indices]
            full_path = indices.shape[0] == x_all.shape[0] and np.array_equal(indices, np.arange(x_all.shape[0]))
        payload: dict[str, Any] = {"x_values": x_values}
        if full_path and len(kpath.x_ticks) == len(kpath.labels_ticks):
            payload["x_ticks"] = [float(item) for item in kpath.x_ticks]
            payload["x_ticklabels"] = [_normalize_kpath_tick_label(item) for item in kpath.labels_ticks]
        return payload
    except Exception as exc:
        print(f"[kp] K-path axis warning: {exc}; use k-path point index")
        return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _is_auto_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"auto", "auto_scdm"}


def _gauge_requests_auto(gauge_config: Any) -> bool:
    if _is_auto_token(gauge_config):
        return True
    if isinstance(gauge_config, dict):
        return _is_auto_token(gauge_config.get("method", gauge_config.get("mode")))
    return False


def _project_requests_auto_gauge(project_cfg: dict[str, Any]) -> bool:
    norb_fix_list = project_cfg.get("norb_fix_list")
    has_manual = norb_fix_list is not None and not _is_auto_token(norb_fix_list)
    if has_manual:
        return False
    return _is_auto_token(norb_fix_list) or _gauge_requests_auto(project_cfg.get("gauge"))


def _symm_can_validate_auto_gauge(symm_cfg: Any) -> bool:
    if not isinstance(symm_cfg, dict):
        return False
    if not _as_bool(symm_cfg.get("enable", True)):
        return False
    return bool(symm_cfg.get("tapw_symmetry_dir"))


def _project_requests_inline_symmetry_gauge_validation(project_cfg: dict[str, Any]) -> bool:
    value = project_cfg.get("validate_auto_gauge_with_symmetry")
    if value is not None:
        return _as_bool(value)
    gauge = project_cfg.get("gauge")
    if isinstance(gauge, dict):
        value = gauge.get("validate_with_symmetry", gauge.get("symmetry_validation"))
        if isinstance(value, str):
            return value.strip().lower() in {"inline", "required", "require", "true", "yes", "1"}
        if value is not None:
            return _as_bool(value)
    return False


def _cached_symmetry_basis_payload(symm_cfg: Any, resolve) -> dict[str, Any] | None:
    if not isinstance(symm_cfg, dict):
        return None
    output_dir = symm_cfg.get("output_dir")
    if output_dir is None:
        return None
    resolved = resolve(output_dir)
    if resolved is None:
        return None
    basis_path = Path(resolved) / "basis_selection.json"
    if not basis_path.exists():
        return None
    payload = json.loads(basis_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cached basis selection must be a JSON object: {basis_path}")
    if not isinstance(payload.get("resolved_norb_fix_list"), list):
        raise ValueError(f"Cached basis selection lacks resolved_norb_fix_list: {basis_path}")
    return payload


def _with_deferred_symmetry_validation(report: GaugeAnchorReport) -> GaugeAnchorReport:
    warnings = list(report.warnings)
    message = "symmetry validation deferred to kp symm"
    if message not in warnings:
        warnings.append(message)
    return GaugeAnchorReport(
        gauge_mode=report.gauge_mode,
        resolved_norb_fix_list=report.resolved_norb_fix_list,
        selections=report.selections,
        metric=report.metric,
        state_selection_quality=report.state_selection_quality,
        gauge_anchor_quality=report.gauge_anchor_quality,
        symmetry_closure_quality={
            "status": "deferred",
            "source": "kp_symm",
            "subspace_leakage": None,
            "reason": "Run `kp symm` to validate auto gauge anchors against TAPW source symmetry.",
        },
        warnings=warnings,
    )


def _gauge_report_from_basis_payload(payload: dict[str, Any]) -> GaugeAnchorReport:
    return GaugeAnchorReport(
        gauge_mode=str(payload.get("gauge_mode", "auto_scdm")),
        resolved_norb_fix_list=list(payload.get("resolved_norb_fix_list", [])),
        selections=list(payload.get("selections", [])),
        metric=dict(payload.get("metric", {"type": "orthonormal", "basis_is_orthonormal": True})),
        state_selection_quality=dict(payload.get("state_selection_quality", {"status": "not_evaluated"})),
        gauge_anchor_quality=dict(payload.get("gauge_anchor_quality", {"status": "unknown"})),
        symmetry_closure_quality=dict(
            payload.get(
                "symmetry_closure_quality",
                {
                    "status": "validated",
                    "source": "kp_symm",
                    "subspace_leakage": None,
                },
            )
        ),
        warnings=list(payload.get("warnings", [])),
    )


def parse_int_list(value: str | Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(v) for v in value]


def parse_float_list(value: str | Sequence[float] | None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(v) for v in value]


def _apply_project_overrides(project_cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(project_cfg)
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if value is None:
            continue
        cfg[key] = value
    return cfg


def _project_heff_full_kwargs_supports(name: str) -> bool:
    try:
        sig = inspect.signature(project_heff_full)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


_PROJECT_WORKER_CONTEXT: dict[str, Any] | None = None


def _project_batches(indices: Sequence[int], workers: int, batch_size: int | str | None) -> list[list[int]]:
    values = [int(i) for i in indices]
    if not values:
        return []
    if isinstance(batch_size, str) and batch_size.lower() == "auto":
        batch_size = None
    if batch_size is None:
        effective_workers = max(1, min(int(workers), len(values)))
        batch_size = max(1, int(np.ceil(len(values) / (2 * effective_workers))))
    batch_size = max(1, int(batch_size))
    return [values[i:i + batch_size] for i in range(0, len(values), batch_size)]


def _project_blas_threads(project_cfg: dict[str, Any], workers: int) -> int:
    configured = project_cfg.get("blas_threads")
    if configured is not None:
        return max(1, int(configured))
    cores = os.cpu_count() or 1
    return max(1, min(8, int(cores) // max(1, int(workers))))


def _init_project_worker(context: dict[str, Any]) -> None:
    global _PROJECT_WORKER_CONTEXT
    worker_context = dict(context)
    set_projector_blas_threads(worker_context.get("blas_threads", 1))
    if "hamk" not in worker_context:
        worker_context["hamk"] = load_hamk(worker_context["hamk_file"], mmap_mode="r")
        worker_context["hamk_is_scaled"] = False
    _PROJECT_WORKER_CONTEXT = worker_context


def _project_one_from_context(k_index: int, context: dict[str, Any]) -> tuple[Any, ...]:
    hamk_data = context["hamk"]
    ham_for_projection = hamk_data[int(k_index)] if hamk_data.ndim == 3 else hamk_data
    ham_for_projection = _selected_spin_project_input(ham_for_projection, context["spin"])
    if not context.get("hamk_is_scaled", True):
        energy_scale = float(context.get("energy_scale", 1.0))
        if energy_scale != 1.0:
            ham_for_projection = ham_for_projection * energy_scale
    kwargs = {
        "spin": context["projection_spin"],
        "bands": context["nlow_state_list"],
        "comps": context["norb_fix_list"],
        "Qlayer_list": [[context["q1"]], [context["q2"]]],
        "num_orb_per_layer_list": context["num_orb_per_layer_list"],
        "nlow_state_list": context["nlow_state_list"],
        "norb_fix_list": context["norb_fix_list"],
        "downfold_method": context["method"],
        "E_ref": context["e_ref"],
        "pole_warning_mev": context["pole_warning_mev"],
        "pole_danger_mev": context["pole_danger_mev"],
        "fail_on_near_pole": context["fail_on_near_pole"],
        "return_diagnostics": True,
        "mode": context["mode"],
    }
    if context["supports_compute_condition_number"]:
        kwargs["compute_condition_number"] = context["compute_condition_number"]
    if context["supports_compute_pole_diagnostics"]:
        kwargs["compute_pole_diagnostics"] = context["compute_pole_diagnostics"]
    return project_heff_full(
        ham_for_projection,
        context["q_count"],
        context["orb0"],
        context["num_layer_list"],
        **kwargs,
    )


def _project_batch_from_worker_context(indices: Sequence[int]) -> list[tuple[int, tuple[Any, ...]]]:
    if _PROJECT_WORKER_CONTEXT is None:
        raise RuntimeError("KP project worker context is not initialized")
    return [(int(i), _project_one_from_context(int(i), _PROJECT_WORKER_CONTEXT)) for i in indices]


def _project_batch_from_context(
    indices: Sequence[int],
    context: dict[str, Any],
) -> list[tuple[int, tuple[Any, ...]]]:
    if (
        _PROJECT_WORKER_CONTEXT is None
        or _PROJECT_WORKER_CONTEXT.get("context_id") != context.get("context_id")
    ):
        _init_project_worker(context)
    return _project_batch_from_worker_context(indices)


def _active_indices_from_project_cfg(project_cfg: dict[str, Any]) -> list[int]:
    active = project_cfg.get("active_indices")
    if active is not None:
        return parse_int_list(active)
    nlow_state_list = project_cfg.get("nlow_state_list", [])
    if nlow_state_list and isinstance(nlow_state_list[0], (list, tuple)):
        return sorted({int(x) for layer_bands in nlow_state_list for x in layer_bands})
    if nlow_state_list:
        return [int(x) for x in nlow_state_list]
    return []


def _normalize_nlow_state_list(project_cfg: dict[str, Any]) -> list[list[int]]:
    active = project_cfg.get("active_indices")
    if active is not None:
        return [parse_int_list(active)]
    nlow_state_list = project_cfg.get("nlow_state_list", [])
    if not nlow_state_list:
        raise ValueError(
            "project.nlow_state_list is required before projection/model fitting; "
            "run `kp inspect -c <config>` and choose low-state bands first."
        )
    if nlow_state_list and not isinstance(nlow_state_list[0], (list, tuple)):
        return [[int(x) for x in nlow_state_list]]
    return nlow_state_list


def _downfold_method(project_cfg: dict[str, Any]) -> str:
    return str(project_cfg.get("downfold_method", project_cfg.get("method", "fixed_schur"))).lower()


def _e_ref_from_project_cfg(project_cfg: dict[str, Any]) -> float | None:
    value = project_cfg.get("e_ref", project_cfg.get("E_ref"))
    return None if value is None else float(value)


def _top_n_list_from_project_cfg(project_cfg: dict[str, Any]) -> list[int]:
    value = project_cfg.get("top_n", project_cfg.get("top_n_list", [2, 4, 6, 10, 20]))
    return parse_int_list(value)


def _downfold_method_label(method: str) -> str:
    labels = {
        "first_order": "first_order (projected block only)",
        "fixed_schur": "fixed_schur (fixed-energy downfolding)",
        "linearized_lowdin": "linearized_lowdin (linearized downfolding)",
    }
    return labels.get(method, method)


def _print_project_diagnostics(
    diag_list: Sequence[Any],
    *,
    pole_warning_mev: float,
    pole_danger_mev: float,
    print_k_diagnostics: bool = False,
) -> None:
    if not diag_list:
        return

    print("[kp] Downfolding diagnostics:")

    pole_rows = [
        (ik, d)
        for ik, d in enumerate(diag_list)
        if getattr(d, "pole_distance_min_mev", None) is not None
    ]
    if pole_rows:
        pole_vals = np.array([float(d.pole_distance_min_mev) for _, d in pole_rows], dtype=float)
        min_pos = int(np.argmin(pole_vals))
        min_k, _ = pole_rows[min_pos]
        median_margin = float(np.median(pole_vals))
        print(
            "[kp]   E_ref margin to discarded bands: "
            f"min={pole_vals[min_pos]:.3f} meV at k={min_k:03d}, "
            f"median={median_margin:.3f} meV"
        )
        print(
            "[kp]   status: "
            f"{'OK' if not any(getattr(d, 'near_pole', False) for _, d in pole_rows) else 'CHECK'} "
            f"(warning < {pole_warning_mev:.3f} meV, danger < {pole_danger_mev:.3f} meV)"
        )

        cond_rows = [
            (ik, float(d.pole_condition_number))
            for ik, d in pole_rows
            if getattr(d, "pole_condition_number", None) is not None
        ]
        if cond_rows:
            cond_k, cond_max = max(cond_rows, key=lambda item: item[1])
            if print_k_diagnostics or cond_max >= 1.0e8:
                print(f"[kp]   Schur solve conditioning: max={cond_max:.3e} at k={cond_k:03d}")

        flagged = [
            (ik, d)
            for ik, d in pole_rows
            if getattr(d, "near_pole", False) or getattr(d, "warnings", None)
        ]
        if flagged:
            print("[kp]   k-points requiring attention:")
            for ik, d in flagged[:20]:
                print(f"[kp]     k={ik:03d}: margin={float(d.pole_distance_min_mev):.3f} meV")
                for warning in getattr(d, "warnings", []):
                    print(f"[kp]       {warning}")
            if len(flagged) > 20:
                print(f"[kp]     ... {len(flagged) - 20} more")

        if print_k_diagnostics:
            print("[kp]   per-k diagnostics:")
            for ik, d in pole_rows:
                cond = getattr(d, "pole_condition_number", None)
                cond_text = "" if cond is None else f", Schur condition={float(cond):.3e}"
                print(f"[kp]     k={ik:03d}: margin={float(d.pole_distance_min_mev):.3f} meV{cond_text}")

    herm_vals = [
        float(d.hermiticity_residual)
        for d in diag_list
        if getattr(d, "hermiticity_residual", None) is not None
    ]
    if herm_vals:
        print(f"[kp]   Hermiticity check: max residual={max(herm_vals):.3e}")


def cmd_plot_from_config(cfg_path: str) -> None:
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    print(f"[kp] Loading config: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = normalize_case_config(yaml.safe_load(f), config_path=cfg_path)

    material = cfg.get("material", {})
    plot_cfg = cfg.get("plot", {})

    def resolve(p: str | None) -> str | None:
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(cfg_dir, p)

    band_file = resolve(material.get("band_file"))
    vbm_file = resolve(material.get("vbm_file"))
    explicit_ef = material.get("efermi")

    # Extract common material parameters that may be needed later
    spin = material.get("spin", "all")
    mode = plot_cfg.get("mode", "gamma").lower()
    num_layers = int(material.get("num_layers", 2))

    eigs_list: list[np.ndarray]
    band_eigs_list: list[np.ndarray] | None = None
    target = str(plot_cfg.get("target", "valence"))
    print(f"[kp] Target: {target}")

    # Initialize variables that may be needed later
    orb0 = None
    q1 = None
    q2 = None
    vecs_list = None

    if band_file:
        print(f"[kp] Using band file for k-path panel: {band_file}")
        rows = _load_bands_from_text(band_file)
        band_eigs_list = [np.array(sorted(r)) for r in rows]
        print(f"[kp] Loaded {len(band_eigs_list)} k-path rows from band file.")

    if band_eigs_list is not None and target.lower() != "all":
        # Keep normal bands for the left panel, then compute Q-block bands below.
        eigs_list = band_eigs_list
    else:
        # Build from Hamiltonian (heavy). Keep for generality.
        hamk_file = resolve(material["hamk_file"])
        qset1_file = resolve(material["qset1_file"])
        qset2_file = resolve(material["qset2_file"])
        print(f"[kp] Using Hamiltonian: {hamk_file}")
        print(f"[kp] Using Q-set files: {qset1_file}, {qset2_file}")

        hamk = _load_hamk_with_energy_unit(hamk_file, material, mmap_mode="r")
        q1, q2 = load_Q_sets(qset1_file, qset2_file)
        # print(f"[kp] hamk shape={hamk.shape}, dtype={hamk.dtype}")
        print(f"[kp] Q1 shape={q1.shape}, Q2 shape={q2.shape}")

        # hamk index selection (if 3D)
        hamk_index = int(plot_cfg.get("hamk_index", 0))
        if hamk.ndim == 3:
            hamk2d = hamk[hamk_index]
        elif hamk.ndim == 2:
            hamk2d = hamk
        else:
            raise ValueError(f"Unexpected hamk ndim: {hamk.ndim}")

        print(f"[kp] spin={spin}, num_layers={num_layers}")

        num_layer_list, orb0, num_orb_per_layer_list = _orbital_layout_from_material(
            material,
            hamk2d,
            len(q1),
            spin=spin,
            mode=mode,
        )
        print(f"[kp] inferred per-layer orbitals per Q (single spin) = {orb0}")


        Qlayer_list = [[q1], [q2]]
        print(f"[kp] num_layer_list = {num_layer_list}")
        print(f"[kp] num_orb_per_layer_list = {num_orb_per_layer_list}")
        block_n = (sum(num_layer_list) * orb0) * (2 if spin == "all" else 1)
        print(f"[kp] block dimension per Q (with spin) = {block_n}")

        # Not used in this branch of get_H_block but kept for API compatibility
        # nlow_state_list = [[], []]
        # norb_fix_list = [[], []]

        nlow_state_list = plot_cfg.get("nlow_state_list", [])
        norb_fix_list = plot_cfg.get("norb_fix_list", [])
        print("[kp] nlow_state_list = ",nlow_state_list)
        print("[kp] norb_fix_list = ",norb_fix_list)
        print(f"[kp] Using mode: {mode}")
        hamk2d, block_spin = _selected_spin_block_for_projection(hamk2d, spin)

        print(f"[kp] q1 shape = {q1.shape}, q2 shape = {q2.shape}")
        print(f"[kp] hamk2d shape = {hamk2d.shape}")
        print(f"[kp] orb0 = {orb0}")

        H_eig, H_vec, H_blocks, U_new = get_H_block(
            hamk2d,
            Qlayer_list,
            num_layer_list,
            num_orb_per_layer_list,
            nlow_state_list,
            norb_fix_list,
            spin=block_spin,
            mode=mode,
        )
        eigs_list = [np.asarray(ev, dtype=np.float64) for ev in H_eig]
        vecs_list = [np.asarray(v, dtype=np.complex128) for v in H_vec]
        print(f"[kp] Diagonalized {len(eigs_list)} Q blocks.")
        print(f"[kp] {np.array(eigs_list).shape}")

    # Compute efermi
    if explicit_ef is not None:
        efermi = float(explicit_ef)
        print(f"[kp] Using explicit efermi: {efermi:.6f} eV")
    elif vbm_file:
        efermi = load_efermi_from_vbm_txt(vbm_file)
        print(f"[kp] Using VBM file for efermi: {vbm_file} -> {efermi:.6f} eV")
    else:
        # fallback: use median of top quartile as rough VBM
        concat = np.sort(np.concatenate([np.asarray(e) for e in eigs_list]))
        efermi = float(np.median(concat[-max(10, len(concat)//10):]))
        print(f"[kp] Using fallback efermi estimate: {efermi:.6f} eV")

    if band_eigs_list is not None and target.lower() != "all":
        hamk_file = resolve(material["hamk_file"])
        qset1_file = resolve(material["qset1_file"])
        qset2_file = resolve(material["qset2_file"])
        print(f"[kp] Using Hamiltonian for Q-block panel: {hamk_file}")
        print(f"[kp] Using Q-set files: {qset1_file}, {qset2_file}")

        hamk = _load_hamk_with_energy_unit(hamk_file, material, mmap_mode="r")
        q1, q2 = load_Q_sets(qset1_file, qset2_file)
        print(f"[kp] Q1 shape={q1.shape}, Q2 shape={q2.shape}")

        hamk_index = int(plot_cfg.get("hamk_index", 0))
        if hamk.ndim == 3:
            hamk2d = hamk[hamk_index]
        elif hamk.ndim == 2:
            hamk2d = hamk
        else:
            raise ValueError(f"Unexpected hamk ndim: {hamk.ndim}")

        print(f"[kp] spin={spin}, num_layers={num_layers}")
        num_layer_list, orb0, num_orb_per_layer_list = _orbital_layout_from_material(
            material,
            hamk2d,
            len(q1),
            spin=spin,
            mode=mode,
        )
        Qlayer_list = [[q1], [q2]]
        print(f"[kp] num_layer_list = {num_layer_list}")
        print(f"[kp] num_orb_per_layer_list = {num_orb_per_layer_list}")
        block_n = (sum(num_layer_list) * orb0) * (2 if spin == "all" else 1)
        print(f"[kp] block dimension per Q (with spin) = {block_n}")
        nlow_state_list = plot_cfg.get("nlow_state_list", [])
        norb_fix_list = plot_cfg.get("norb_fix_list", [])
        print("[kp] nlow_state_list = ", nlow_state_list)
        print("[kp] norb_fix_list = ", norb_fix_list)
        hamk2d, block_spin = _selected_spin_block_for_projection(hamk2d, spin)
        H_eig, H_vec, H_blocks, U_new = get_H_block(
            hamk2d,
            Qlayer_list,
            num_layer_list,
            num_orb_per_layer_list,
            nlow_state_list,
            norb_fix_list,
            spin=block_spin,
            mode=mode,
        )
        eigs_list = [np.asarray(ev, dtype=np.float64) for ev in H_eig]
        vecs_list = [np.asarray(v, dtype=np.complex128) for v in H_vec]
        print(f"[kp] Diagonalized {len(eigs_list)} Q blocks.")
        print(f"[kp] {np.array(eigs_list).shape}")

    out_path = resolve(plot_cfg.get("out", f"plot_{mode}_scatter.png"))
    data_out = resolve(plot_cfg.get("data_out", os.path.splitext(out_path)[0] + ".txt"))
    ref_q_index = int(plot_cfg.get("ref_q_index", 0))
    title = plot_cfg.get("title", None)
    ylim_cfg = plot_cfg.get("ylim")
    if ylim_cfg is not None and isinstance(ylim_cfg, (list, tuple)) and len(ylim_cfg) == 2:
        ylim = (float(ylim_cfg[0]), float(ylim_cfg[1]))
    else:
        ymin = plot_cfg.get("ymin")
        ymax = plot_cfg.get("ymax")
        ylim = (float(ymin), float(ymax)) if (ymin is not None and ymax is not None) else None
    print(f"[kp] Saving spectrum txt to: {data_out}")
    print(f"[kp] Saving plot to: {out_path}")
    if ylim is not None:
        print(f"[kp] Using y-lim: {ylim}")

    # Optional Q ordering by |Q| after rotation
    index_order = None
    sort_by_qnorm = bool(plot_cfg.get("sort_by_qnorm", True))
    q_rotation_deg = float(plot_cfg.get("q_rotation_deg", 0.0))
    if sort_by_qnorm and q1 is not None and q2 is not None:
        q_base = q1
        q_base2 = q2
        center = q_base.mean(axis=0)
        ca, sa = np.cos(np.deg2rad(q_rotation_deg)), np.sin(np.deg2rad(q_rotation_deg))
        R = np.array([[ca, -sa], [sa, ca]])
        Q_set1 = np.array([(R @ (center - q_base[i])[:2]) for i in range(len(q_base))])
        Q_set2 = np.array([(R @ (center - q_base2[i])[:2]) for i in range(len(q_base2))])

        if mode == "gamma":
            # gamma mode: eigs_list length = len(q1) (one block per Q combining all layers)
            index_order = np.argsort(np.linalg.norm(Q_set1, axis=1))
        else:
            # non-gamma mode: eigs_list is grouped by physical layer, then Q.
            index_order1 = np.argsort(np.linalg.norm(Q_set1, axis=1))
            index_order2 = np.argsort(np.linalg.norm(Q_set2, axis=1))
            pieces = []
            offset = 0
            for group_index, n_layers in enumerate(num_layer_list):
                order = index_order1 if group_index == 0 else index_order2
                q_len = len(q1) if group_index == 0 else len(q2)
                for layer_in_group in range(n_layers):
                    pieces.append(offset + layer_in_group * q_len + order)
                offset += n_layers * q_len
            index_order = np.concatenate(pieces)
        print(f"[kp] Using Q-norm sort with rotation {q_rotation_deg} deg. Mode: {mode}")

    # H_eig is an array of object vectors; normalize to list
    # Save spectrum to text
    save_spectrum_txt(eigs_list, data_out, index_order=index_order)
    if band_eigs_list is not None and eigs_list is not band_eigs_list:
        q_sector_lengths = None
        if q1 is not None and q2 is not None:
            if mode == "gamma":
                q_sector_lengths = [len(eigs_list)]
            else:
                q_lengths = [len(q1), len(q2)]
                q_sector_lengths = [
                    int(n_layers) * int(q_lengths[group_index])
                    for group_index, n_layers in enumerate(num_layer_list)
                ]
        k_axis = _kpath_axis_from_config(
            cfg,
            cfg_dir=cfg_dir,
            project_indices=list(range(len(band_eigs_list))),
            row_count=len(band_eigs_list),
        )
        plot_inspect_band_and_qblock(
            band_eigs_list,
            eigs_list,
            efermi,
            out=out_path,
            title=title,
            ylim=ylim,
            q_index_order=index_order,
            q_sector_lengths=q_sector_lengths,
            ref_q_index=ref_q_index,
            q_window_bands=int(plot_cfg.get("q_window_bands", 10)),
            k_x_values=k_axis.get("x_values"),
            k_x_ticks=k_axis.get("x_ticks"),
            k_x_ticklabels=k_axis.get("x_ticklabels"),
        )
    else:
        plot_eigs_scatter(
            eigs_list,
            efermi,
            out=out_path,
            # target=target,
            # ref_q_index=ref_q_index,
            title=title,
            ylim=ylim,
            index_order=index_order,
        )
    _write_inspect_sidecars(
        cfg,
        output_dir=os.path.dirname(out_path) or ".",
        eigs_list=eigs_list,
        efermi=efermi,
        ref_q_index=ref_q_index,
    )

    # # -------------------- Optional low-energy projection (Heff) --------------------
    # project_cfg = cfg.get("project", {})
    # if project_cfg and project_cfg.get("enable", False):
    #     try:
    #         from joblib import Parallel, delayed  # noqa: F401
    #         have_joblib = True
    #     except Exception:
    #         have_joblib = False

    #     nlow_state_list = project_cfg.get("nlow_state_list", [])
    #     norb_fix_list = project_cfg.get("norb_fix_list", [])
    #     workers = int(project_cfg.get("workers", 4))
    #     out_heff = resolve(project_cfg.get("out_heff", os.path.join(os.path.dirname(out_path) or ".", "heff_list.npy")))
    #     out_eig = resolve(project_cfg.get("out_eig", os.path.join(os.path.dirname(out_path) or ".", "heff_eig.npy")))
    #     out_vec = resolve(project_cfg.get("out_vec", os.path.join(os.path.dirname(out_path) or ".", "heff_vec.npy")))

    #     # Prepare per-Q blocks; reuse H_diag_block logic by recomputing indices via get_H_block
    #     # We already have eigs_list & vecs_list, but need the block Hamiltonians as input to projection.
    #     # So recompute H_diag_block using get_H_block result kept earlier? Not stored — recompute quickly.
    #     H_eig2, H_vec2, H_blocks, _ = get_H_block(
    #         hamk2d,
    #         [[q1], [q2]],
    #         [1, 1],
    #         [[orb0], [orb0]],
    #         [[], []],
    #         [[], []],
    #         spin=spin,
    #         mode="gamma",
    #     )
    #     blocks = [np.asarray(b) for b in H_blocks]
    #     # Apply same Q ordering
    #     if index_order is not None:
    #         blocks = [blocks[i] for i in index_order]

    #     print(f"[kp] Projecting to low-energy subspace: workers={workers}")
    #     if have_joblib and workers > 1:
    #         from joblib import Parallel, delayed
    #         from tqdm import tqdm
    #         results = Parallel(n_jobs=workers)(
    #             delayed(get_h_dft_low)(blocks[i], nlow_state_list, norb_fix_list, q1, q2)
    #             for i in tqdm(range(len(blocks)))
    #         )
    #     else:
    #         results = [get_h_dft_low(blocks[i], nlow_state_list, norb_fix_list, q1, q2) for i in range(len(blocks))]

    #     heff_list = [r[0] for r in results]
    #     heig_list = [r[1] for r in results]
    #     hvec_list = [r[2] for r in results]

    #     os.makedirs(os.path.dirname(out_heff) or ".", exist_ok=True)
    #     np.save(out_heff, np.array(heff_list, dtype=object))
    #     np.save(out_eig, np.array(heig_list, dtype=object))
    #     np.save(out_vec, np.array(hvec_list, dtype=object))
    #     print(f"[kp] Saved Heff to: {out_heff}")
    #     print(f"[kp] Saved Heff eig to: {out_eig}")
    #     print(f"[kp] Saved Heff vec to: {out_vec}")

    # # -------------------- Inline band analysis (around EF) --------------------
    # try:
    #     tt
    #     E2 = np.vstack([np.asarray(ev, dtype=np.float64) for ev in eigs_list])
    #     if index_order is not None:
    #         E2 = E2[index_order]
    #         if 'vecs_list' in locals():
    #             vecs_ord = [vecs_list[i] for i in index_order]
    #         else:
    #             vecs_ord = None
    #     else:
    #         vecs_ord = vecs_list if 'vecs_list' in locals() else None

    #     ref_q = ref_q_index if 0 <= ref_q_index < E2.shape[0] else 0
    #     below_n = int(plot_cfg.get('report_below', 2))
    #     above_n = int(plot_cfg.get('report_above', 2))
    #     top_n = int(plot_cfg.get('report_components', 4))

    #     e_row = E2[ref_q]
    #     below_idx = np.where(e_row < efermi)[0]
    #     above_idx = np.where(e_row >= efermi)[0]
    #     sel_below = below_idx[-below_n:].tolist() if below_idx.size else []
    #     sel_above = above_idx[:above_n].tolist() if above_idx.size else []
    #     print(f"[kp] Selected bands at ref_Q={ref_q}: below EF {sel_below}, above EF {sel_above}")

    #     if vecs_ord is not None and len(vecs_ord) == E2.shape[0]:
    #         labels_pattern = material.get("orbital_order")
    #         per_layer_labels = None
    #         if labels_pattern:
    #             try:
    #                 per_layer_labels = expand_orbital_order_pattern(str(labels_pattern))
    #             except Exception:
    #                 per_layer_labels = None
    #         if per_layer_labels is not None and len(per_layer_labels) != orb0:
    #             if len(per_layer_labels) < orb0:
    #                 per_layer_labels += [f"orb_{i}" for i in range(len(per_layer_labels), orb0)]
    #             else:
    #                 per_layer_labels = per_layer_labels[:orb0]

    #         block_dim = int(vecs_ord[0].shape[0])
    #         half = block_dim // 2 if spin == 'all' else block_dim

    #         def map_index(idx: int):
    #             if spin == 'all':
    #                 spin_tag = 'up' if idx < half else 'down'
    #                 idx0 = idx if idx < half else idx - half
    #             else:
    #                 spin_tag = spin
    #                 idx0 = idx
    #             layer = idx0 // orb0
    #             orb_local = idx0 % orb0
    #             label = None
    #             if per_layer_labels is not None and 0 <= orb_local < len(per_layer_labels):
    #                 label = per_layer_labels[orb_local]
    #             return spin_tag, int(layer), int(orb_local), label

    #         for b in sel_below + sel_above:
    #             w = np.zeros(block_dim, dtype=np.float64)
    #             for V in vecs_ord:
    #                 vec = np.asarray(V[:, b], dtype=np.complex128)
    #                 w += np.abs(vec) ** 2
    #             print(np.shape(vec))
    #             top_idx = np.argsort(-w)[:top_n]
    #             w_top = w[top_idx]
    #             tot = float(w.sum()) if float(w.sum()) != 0.0 else 1.0
    #             frac = w_top / tot
    #             print(f"[kp] Band {b}: top-{top_n} orbital components (aggregated over Q)")
    #             print("        rank   idx   spin  layer  orb   weight        frac     label")
    #             for rk, (t, wt, fr) in enumerate(zip(top_idx.tolist(), w_top.tolist(), frac.tolist()), start=1):
    #                 spin_tag, layer, orb_local, label = map_index(int(t))
    #                 label_str = label if label is not None else "-"
    #                 print(f"        {rk:>4}  {int(t):>5}  {spin_tag:<5}  {layer:>5}  {orb_local:>3}  {wt:>9.4f}   {fr*100:>7.2f}%   {label_str}")
    #     else:
    #         print("[kp] Eigenvectors not available; skip orbital decomposition. Set target:'all' or remove band_file.")
    # except Exception as ex:
    #     print(f"[kp] Analysis step warning: {ex}")

    # -------------------- Inline band analysis (around EF) --------------------

    def _format_complex(z: complex) -> str:
        # 固定宽度便于对齐：real(>11.6f) + imag(+11.6f)i  → 总宽约 24
        return f"{z.real:>11.6f}{z.imag:+11.6f}i"


    try:
        # 组装 E2: (nQ, nBands)
        E2 = np.vstack([np.asarray(ev, dtype=np.float64) for ev in eigs_list])

        # 可选重排 Q 顺序
        if index_order is not None:
            E2 = E2[index_order]
            if 'vecs_list' in locals():
                vecs_ord = [vecs_list[i] for i in index_order]
            else:
                vecs_ord = None
        else:
            vecs_ord = vecs_list if 'vecs_list' in locals() else None

        # 参考 Q
        ref_q = ref_q_index if 0 <= ref_q_index < E2.shape[0] else 0

        # 选择 EF 附近的带（在 ref_q 这一行）
        below_n = int(plot_cfg.get('report_below', 2))
        above_n = int(plot_cfg.get('report_above', 2))
        top_n   = int(plot_cfg.get('report_components', 8))  # 你示例里是 top-8

        e_row = E2[ref_q]
        below_idx = np.where(e_row < efermi)[0]
        above_idx = np.where(e_row >= efermi)[0]
        sel_below = below_idx[-below_n:].tolist() if below_idx.size else []
        sel_above = above_idx[:above_n].tolist() if above_idx.size else []
        print(f"[kp] Selected bands at ref_Q={ref_q}: below EF {sel_below}, above EF {sel_above}")

        # 若没有本征矢，跳过
        if vecs_ord is None or len(vecs_ord) != E2.shape[0]:
            print("[kp] Eigenvectors not available; skip orbital decomposition. Set target:'all' or remove band_file.")
        elif orb0 is None:
            print("[kp] orb0 not available; skip orbital decomposition. Need to load Hamiltonian.")
        else:
            # 轨道标签准备（"层内轨道名表"，长度对齐到 orb0）
            labels_pattern = material.get("orbital_order") if isinstance(material, dict) else None
            layer_names = [f"L{i + 1}" for i in range(int(sum(num_layer_list)))]
            labels_by_layer = (
                expand_orbital_order_by_sector(
                    labels_pattern,
                    layer_names,
                    expected_count=orb0,
                )
                if labels_pattern
                else None
            )

            Vq = np.asarray(vecs_ord[ref_q])  # 形状 (block_dim, nBands)
            if Vq.ndim != 2:
                raise RuntimeError(f"Eigenvector matrix at ref_q has wrong shape: {Vq.shape}")

            block_dim, nBands = Vq.shape
            half = block_dim // 2 if spin == 'all' else block_dim
            report_block_layer = None
            if mode != "gamma" and q1 is not None and q2 is not None:
                report_block_layer = _plot_report_layer_for_ref_q(
                    ref_q,
                    num_layer_list=num_layer_list,
                    q_lengths=[len(q1), len(q2)],
                    index_order=index_order,
                )

            def map_index(idx: int, block_layer: int | None = None):
                """将平面波/原子轨道基底索引 -> (spin_tag, layer, orb_local, label)

                Args:
                    idx: Orbital index within the reported block.
                    block_layer: Physical layer for non-gamma single-layer blocks.
                """
                if spin == 'all':
                    spin_tag = 'up' if idx < half else 'down'
                    idx0 = idx if idx < half else idx - half
                else:
                    spin_tag = spin
                    idx0 = idx

                if mode == "gamma":
                    # gamma mode: block contains all layers, calculate layer from idx0
                    layer = idx0 // orb0
                else:
                    # non-gamma mode: block contains only one layer
                    if block_layer is not None:
                        layer = block_layer
                    else:
                        layer = 0

                orb_local = idx0 % orb0
                label = None
                if labels_by_layer is not None:
                    per_layer_labels = labels_by_layer.get(f"L{int(layer) + 1}")
                    if per_layer_labels is not None and 0 <= orb_local < len(per_layer_labels):
                        label = per_layer_labels[orb_local]
                return spin_tag, int(layer), int(orb_local), (label if label is not None else "-")

            # 仅在 ref_q 做分解：对每条被选中的带打印 top-N 分量 + 复系数
            for b in (sel_below + sel_above):
                if not (0 <= b < nBands):
                    print(f"[kp] Skip band {b}: out of range 0..{nBands-1}")
                    continue

                vec = np.asarray(Vq[:, b], dtype=np.complex128)  # (block_dim,)
                w = np.abs(vec) ** 2
                tot = float(w.sum()) if float(w.sum()) != 0.0 else 1.0

                top_idx = np.argsort(-w)[:top_n]
                w_top = w[top_idx]
                frac = w_top / tot

                print(f"[kp] Band {b}: top-{top_n} orbital components ({ref_q} Q, mode={mode})")
                # 列定义：rank idx spin layer orb coeff weight frac label
                print("        rank   idx   spin  layer  orb        coeff (complex)       frac     label")
                for rk, t in enumerate(top_idx.tolist(), start=1):
                    coeff = complex(vec[int(t)])
                    wt = float(w[int(t)])
                    fr = float(frac[rk-1])
                    spin_tag, layer, orb_local, label = map_index(int(t), block_layer=report_block_layer)
                    # 与你示例的对齐风格一致（数值宽度匹配），coeff 另外增加一列
                    print(f"        {rk:>4}  {int(t):>5}  {spin_tag:<5}  {layer:>5}  {orb_local:>3}  {_format_complex(coeff)}   {fr*100:>7.2f}%   {label}")

                # 如需一次性输出该带“所有基底分量”的复系数（可能很长），可打开以下注释：
                # print(f"[kp] Band {b}: full coefficient vector at ref_Q={ref_q} (index, coeff)")
                # for idx_full, c_full in enumerate(vec.tolist()):
                #     print(f"        {idx_full:>5}  {_format_complex(complex(c_full))}")

    except Exception as ex:
        print(f"[kp] Analysis step warning: {ex}")




    print("[kp] DONE")


def cmd_project_from_config(cfg_path: str, overrides: dict[str, Any] | None = None) -> None:
    """Project selected bands into an effective Hamiltonian (Heff) across Q blocks.

    Reads the same material/plot settings for data and Q ordering, and the
    'project' section for projection parameters and outputs.
    """
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    print(f"[kp] Loading config: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = normalize_case_config(yaml.safe_load(f), config_path=cfg_path)

    material = cfg.get("material", {})
    plot_cfg = cfg.get("plot", {})
    project_cfg = _apply_project_overrides(cfg.get("project", {}), overrides)

    def resolve(p: str | None) -> str | None:
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(cfg_dir, p)

    hamk_file = resolve(material["hamk_file"])
    qset1_file = resolve(material["qset1_file"])
    qset2_file = resolve(material["qset2_file"])
    spin = material.get("spin", "all")

    print("[kp] Loading input data")
    hamk = _load_hamk_with_energy_unit(hamk_file, material, mmap_mode="r")
    q1, q2 = load_Q_sets(qset1_file, qset2_file)

    hamk_index = int(plot_cfg.get("hamk_index", 0))
    if hamk.ndim == 3:
        hamk2d = hamk[hamk_index]
    elif hamk.ndim == 2:
        hamk2d = hamk
    else:
        raise ValueError(f"Unexpected hamk ndim: {hamk.ndim}")

    # Projection parameters
    nlow_state_list = _normalize_nlow_state_list(project_cfg)
    norb_fix_list = project_cfg["norb_fix_list"] if "norb_fix_list" in project_cfg else None
    workers = int(project_cfg.get("workers", 4))
    mode = project_cfg.get("mode", "gamma").lower()
    method = _downfold_method(project_cfg)
    e_ref = _e_ref_from_project_cfg(project_cfg)
    pole_warning_mev = float(project_cfg.get("pole_warning_mev", 10.0))
    pole_danger_mev = float(project_cfg.get("pole_danger_mev", 1.0))
    fail_on_near_pole = _as_bool(project_cfg.get("fail_on_near_pole", False))
    compute_pole_diagnostics = _as_bool(project_cfg.get("compute_pole_diagnostics", False))
    compute_condition_number = _as_bool(project_cfg.get("compute_condition_number", False))
    verbose = _as_bool(project_cfg.get("verbose", False))
    print_k_diagnostics = _as_bool(project_cfg.get("print_k_diagnostics", False))
    # top_n_list = _top_n_list_from_project_cfg(project_cfg)
    if method in {"fixed_schur", "linearized_lowdin"} and e_ref is None:
        raise ValueError(f"project.downfold_method={method!r} requires project.e_ref")

    num_layer_list, orb0, num_orb_per_layer_list = _orbital_layout_from_material(
        material,
        hamk2d,
        len(q1),
        spin=spin,
        mode=mode,
    )

    # For projection, we do not reorder Q to avoid changing Heff basis.
    q_count = len(q1)
    out_dir = resolve(project_cfg.get("out_dir", "plots"))
    if out_dir is None:
        out_dir = os.path.join(cfg_dir, "plots")
    canonical_project = _is_canonical_case_config(cfg)
    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    nk = hamk3d.shape[0]
    has_explicit_k_indices = project_cfg.get("k_indices") is not None
    project_indices = (
        parse_int_list(project_cfg.get("k_indices"))
        if has_explicit_k_indices
        else list(range(nk))
    )
    for idx in project_indices:
        if idx < 0 or idx >= nk:
            raise IndexError(f"project.k_indices contains {idx}, but available k indices are 0..{nk - 1}")
    active_indices = _active_indices_from_project_cfg(project_cfg)
    model_dim = q_count * sum(len(layer_bands) for layer_bands in nlow_state_list)
    print("[kp] Project setup:")
    print(f"[kp]   material={material.get('name', 'unknown')}, mode={mode}, spin={spin}")
    print(f"[kp]   k-points={len(project_indices)}/{nk}, Q-points={q_count}, orbitals/layer/Q={orb0}")
    if len(project_indices) != nk:
        print(f"[kp]   k-indices={project_indices}")
    print(f"[kp]   active bands={active_indices}, model dimension={model_dim}")
    method_line = f"[kp]   method={_downfold_method_label(method)}"
    if e_ref is not None:
        method_line += f", E_ref={e_ref:.6f} eV"
    print(method_line)
    projection_spin = "all" if str(spin).lower() == "all" else "up"
    q2_for_projection = q2 if q2 is not None else q1
    gauge_report: GaugeAnchorReport | None = None
    resolved_norb_fix_list: list[Any] | None = None
    symm_cfg = cfg.get("symm", {})
    if _project_requests_auto_gauge(project_cfg) and _symm_can_validate_auto_gauge(symm_cfg):
        basis_payload = _cached_symmetry_basis_payload(symm_cfg, resolve)
        if basis_payload is not None:
            print("[kp]   resolving auto gauge from cached kp symm basis selection")
            resolved_norb_fix_list = list(basis_payload["resolved_norb_fix_list"])
            gauge_report = _gauge_report_from_basis_payload(basis_payload)
        elif _project_requests_inline_symmetry_gauge_validation(project_cfg):
            print("[kp]   resolving auto gauge with kp symm validation")
            symm_summary = run_symmetry_projection_from_config(cfg_path)
            project_basis = symm_summary.get("project_basis", {}) if isinstance(symm_summary, dict) else {}
            resolved_from_symm = project_basis.get("resolved_norb_fix_list") if isinstance(project_basis, dict) else None
            if not isinstance(resolved_from_symm, list):
                raise ValueError("kp symm did not return project_basis.resolved_norb_fix_list for auto gauge")
            basis_payload = _cached_symmetry_basis_payload(symm_cfg, resolve)
            if basis_payload is None:
                basis_payload = {
                    "gauge_mode": project_basis.get("gauge_mode", "auto_scdm") if isinstance(project_basis, dict) else "auto_scdm",
                    "resolved_norb_fix_list": resolved_from_symm,
                    "selections": [],
                    "metric": {"type": "orthonormal", "basis_is_orthonormal": True},
                    "state_selection_quality": {"status": "not_evaluated"},
                    "gauge_anchor_quality": {"status": "unknown"},
                    "symmetry_closure_quality": {
                        "status": "validated",
                        "source": "kp_symm",
                        "subspace_leakage": None,
                    },
                    "warnings": [],
                }
            resolved_norb_fix_list = resolved_from_symm
            gauge_report = _gauge_report_from_basis_payload(basis_payload)
        else:
            print("[kp]   resolving auto gauge locally; symmetry validation deferred to kp symm")
            resolved_norb_fix_list, local_report = resolve_project_gauge_anchors(
                _selected_spin_project_input(np.asarray(hamk2d), spin),
                q_count,
                orb0,
                num_layer_list,
                spin=projection_spin,
                Qlayer_list=[[q1], [q2_for_projection]],
                num_orb_per_layer_list=num_orb_per_layer_list,
                nlow_state_list=nlow_state_list,
                norb_fix_list=norb_fix_list,
                gauge_config=project_cfg.get("gauge"),
                mode=mode,
            )
            gauge_report = _with_deferred_symmetry_validation(local_report)
    else:
        resolved_norb_fix_list, gauge_report = resolve_project_gauge_anchors(
            _selected_spin_project_input(np.asarray(hamk2d), spin),
            q_count,
            orb0,
            num_layer_list,
            spin=projection_spin,
            Qlayer_list=[[q1], [q2_for_projection]],
            num_orb_per_layer_list=num_orb_per_layer_list,
            nlow_state_list=nlow_state_list,
            norb_fix_list=norb_fix_list,
            gauge_config=project_cfg.get("gauge"),
            mode=mode,
        )
    norb_fix_list = resolved_norb_fix_list
    write_basis_selection_report(out_dir, gauge_report)
    if canonical_project:
        basis_md = Path(out_dir) / "basis_selection.md"
        if basis_md.exists():
            (Path(out_dir) / "basis.md").write_text(basis_md.read_text(encoding="utf-8"), encoding="utf-8")
        np.savez(
            Path(out_dir) / "basis.npz",
            nlow_state_list=np.asarray(nlow_state_list, dtype=object),
            norb_fix_list=np.asarray(norb_fix_list, dtype=object),
        )
    print(f"[kp]   gauge={gauge_report.gauge_mode}")
    print(f"[kp]   basis selection report={os.path.join(out_dir, 'basis_selection.json')}")
    print(f"[kp]   output directory={out_dir}")
    if verbose:
        print(f"[kp]   hamk={hamk_file}")
        print(f"[kp]   qset1={qset1_file}")
        print(f"[kp]   qset2={qset2_file}")
        print(f"[kp]   hamk shape={hamk.shape}, reference block shape={hamk2d.shape}")
        print(f"[kp]   nlow_state_list={nlow_state_list}")
        print(f"[kp]   norb_fix_list={norb_fix_list}")
    # Unified output directory for all artifacts
    out_heff = os.path.join(out_dir, "heff.npy" if canonical_project else "heff_list.npy")
    out_eig = os.path.join(out_dir, "eigvals.npy" if canonical_project else "heff_eig.npy")
    out_vec = os.path.join(out_dir, "vectors.npy" if canonical_project else "heff_vec.npy")
    plot_out = os.path.join(out_dir, "scatter.png" if canonical_project else "heff_scatter.png")
    data_out = os.path.join(out_dir, "eigvals.txt" if canonical_project else "heff_spectrum.txt")
    original_eigs_list = None
    band_file = resolve(material.get("band_file"))
    if band_file:
        try:
            original_eigs_list = _load_bands_from_text(band_file)
            if len(project_indices) != nk:
                original_eigs_list = [original_eigs_list[i] for i in project_indices]
        except Exception as ex:
            print(f"[kp] Overlay warning: failed to load original bands from {band_file}: {ex}")

    project_context_id = uuid.uuid4().hex
    project_context = {
        "context_id": project_context_id,
        "hamk": hamk3d,
        "hamk_file": hamk_file,
        "hamk_is_scaled": True,
        "energy_scale": _energy_scale_from_material(material),
        "spin": spin,
        "projection_spin": projection_spin,
        "q1": q1,
        "q2": q2,
        "q_count": q_count,
        "orb0": orb0,
        "num_layer_list": num_layer_list,
        "num_orb_per_layer_list": num_orb_per_layer_list,
        "nlow_state_list": nlow_state_list,
        "norb_fix_list": norb_fix_list,
        "method": method,
        "e_ref": e_ref,
        "pole_warning_mev": pole_warning_mev,
        "pole_danger_mev": pole_danger_mev,
        "fail_on_near_pole": fail_on_near_pole,
        "compute_condition_number": compute_condition_number,
        "compute_pole_diagnostics": compute_pole_diagnostics,
        "supports_compute_condition_number": _project_heff_full_kwargs_supports("compute_condition_number"),
        "supports_compute_pole_diagnostics": _project_heff_full_kwargs_supports("compute_pole_diagnostics"),
        "mode": mode,
    }

    # Run projection for each k (first dim of hamk)
    effective_workers = max(1, min(workers, len(project_indices)))
    inner_blas_threads = _project_blas_threads(project_cfg, effective_workers)
    set_projector_blas_threads(inner_blas_threads)
    print(f"[kp] Running projection: {len(project_indices)} k-points, workers={workers}")
    if workers > effective_workers:
        print(f"[kp]   effective workers={effective_workers} (limited by k-point count)")
    if verbose:
        print(f"[kp]   BLAS threads per worker={inner_blas_threads}")

    if workers <= 1:
        results = [_project_one_from_context(i, project_context) for i in project_indices]
    else:
        try:
            from joblib import Parallel, delayed
            from tqdm import tqdm

            parallel_context = dict(project_context)
            parallel_context.pop("hamk", None)
            parallel_context["blas_threads"] = inner_blas_threads
            batches = _project_batches(project_indices, effective_workers, project_cfg.get("batch_size"))
            if len(batches) != len(project_indices):
                print(f"[kp]   parallel batches={len(batches)}, batch_size~{len(batches[0])}; progress counts completed k-points")
            parallel_results = Parallel(
                n_jobs=effective_workers,
                backend="loky",
                return_as="generator",
                batch_size=1,
                initializer=_init_project_worker,
                initargs=(parallel_context,),
            )(
                delayed(_project_batch_from_context)(batch, parallel_context)
                for batch in batches
            )
            progress = tqdm(
                total=len(project_indices),
                desc="[kp] project",
                unit="k",
                ncols=80,
                leave=True,
                disable=not sys.stderr.isatty(),
            )
            batch_results = []
            try:
                for batch in parallel_results:
                    batch_results.append(batch)
                    progress.update(len(batch))
            finally:
                progress.close()
            result_by_index = {k_index: result for batch in batch_results for k_index, result in batch}
            results = [result_by_index[int(i)] for i in project_indices]
        except Exception as ex:
            print(f"[kp] Parallel projection failed, falling back to serial: {ex}")
            results = [_project_one_from_context(i, project_context) for i in project_indices]

    heff_list = [np.asarray(r[0], dtype=np.complex128) for r in results]  # (nk, M, M)
    heig_list = [np.asarray(r[1], dtype=np.float64) for r in results]      # (nk, M)
    hvec_list = [np.asarray(r[2], dtype=np.complex128) for r in results]  # (nk, M, M)
    diag_list = [r[3] for r in results if len(r) > 3]

    # Stack to numeric arrays if possible
    try:
        heff_arr = np.stack(heff_list, axis=0)  # (Q, M, M)
        heig_arr = np.stack(heig_list, axis=0)  # (Q, M)
        hvec_arr = np.stack(hvec_list, axis=0)  # (Q, M, M)
    except Exception:
        heff_arr = np.array(heff_list, dtype=object)
        heig_arr = np.array(heig_list, dtype=object)
        hvec_arr = np.array(hvec_list, dtype=object)

    # Save
    os.makedirs(os.path.dirname(out_heff) or ".", exist_ok=True)
    np.save(out_heff, heff_arr)
    np.save(out_eig, heig_arr)
    np.save(out_vec, hvec_arr)
    if canonical_project:
        np.save(os.path.join(out_dir, "heff_list.npy"), heff_arr)
        np.save(os.path.join(out_dir, "heff_eig.npy"), heig_arr)
        np.save(os.path.join(out_dir, "heff_vec.npy"), hvec_arr)
    if has_explicit_k_indices:
        np.save(os.path.join(out_dir, "k_indices.npy"), np.asarray(project_indices, dtype=int))
    if isinstance(heff_arr, np.ndarray) and heff_arr.dtype != object:
        print(f"[kp] Heff shape: {heff_arr.shape}")
    print("[kp] Saved arrays:")
    print(f"[kp]   {out_heff}")
    print(f"[kp]   {out_eig}")
    print(f"[kp]   {out_vec}")
    _print_project_diagnostics(
        diag_list,
        pole_warning_mev=pole_warning_mev,
        pole_danger_mev=pole_danger_mev,
        print_k_diagnostics=print_k_diagnostics,
    )

    # Plot projected bands.  Project-specific E_F wins; canonical one-file
    # configs normally define it once under material.
    proj_ef = project_cfg.get("efermi", material.get("efermi"))
    efermi = float(proj_ef) if proj_ef is not None else None
    ylim_cfg = project_cfg.get("ylim")
    ylim = (float(ylim_cfg[0]), float(ylim_cfg[1])) if isinstance(ylim_cfg, (list, tuple)) and len(ylim_cfg) == 2 else None
    project_plot_cfg = project_cfg.get("plot", {})
    if project_plot_cfg is None:
        project_plot_cfg = {}
    if not isinstance(project_plot_cfg, dict):
        raise ValueError("project.plot must be a mapping when provided")
    project_top_bands = project_plot_cfg.get("top_bands", project_cfg.get("plot_top_bands"))
    project_bottom_bands = project_plot_cfg.get("bottom_bands", project_cfg.get("plot_bottom_bands"))
    project_band_slice = project_plot_cfg.get("band_slice", project_cfg.get("plot_band_slice"))
    project_align = str(project_plot_cfg.get("align", project_cfg.get("plot_align", "fermi")))
    kpath_axis = _kpath_axis_from_config(
        cfg,
        cfg_dir=cfg_dir,
        project_indices=project_indices,
        row_count=len(heig_list),
    )
    if original_eigs_list is not None and len(original_eigs_list) != len(heig_list):
        print(
            f"[kp] Overlay warning: original band rows ({len(original_eigs_list)}) "
            f"do not match Heff rows ({len(heig_list)}); skip overlay."
        )
        original_eigs_list = None
    # Top-N comparison is reporting-only, so keep it disabled while downfold.py
    # is limited to the core downfolding math.
    # if original_eigs_list is not None:
    #     try:
    #         metrics = compute_top_n_metrics(original_eigs_list, heig_list, top_n_list)
    #         print_top_n_metrics(metrics)
    #     except ValueError as ex:
    #         print(f"[kp] Top-N comparison warning: {ex}")
    print(f"[kp] Saving Heff plot: {plot_out}")
    plot_eigs_scatter(
        heig_list,
        efermi,
        out=plot_out,
        title=project_cfg.get("title"),
        ylim=ylim,
        index_order=None,
        original_eigs_list=original_eigs_list,
        top_bands=None if project_top_bands is None else int(project_top_bands),
        bottom_bands=None if project_bottom_bands is None else int(project_bottom_bands),
        band_slice=project_band_slice,
        align=project_align,
        xlabel="k-path point",
        figsize=(3.0, 5.0) if canonical_project else None,
        box_aspect=(5 / 3) if canonical_project else None,
        font_family=_inspect_plot_font_family() if canonical_project else None,
        **kpath_axis,
    )
    save_spectrum_txt(heig_list, data_out)
    print(f"[kp] Saved Heff spectrum: {data_out}")
    if canonical_project:
        _write_case_summary(cfg, out_dir, "projection")


def cmd_sweep_from_config(cfg_path: str, overrides: dict[str, Any] | None = None) -> None:
    """Sweep E_ref for fixed-energy Schur downfolding and report pole distances."""
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    print(f"[kp] Loading config: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = normalize_case_config(yaml.safe_load(f), config_path=cfg_path)

    material = cfg.get("material", {})
    plot_cfg = cfg.get("plot", {})
    project_cfg = _apply_project_overrides(cfg.get("project", {}), overrides)

    def resolve(p: str | None) -> str | None:
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(cfg_dir, p)

    hamk_file = resolve(material["hamk_file"])
    qset1_file = resolve(material["qset1_file"])
    qset2_file = resolve(material["qset2_file"])
    band_file = resolve(material.get("band_file"))
    # Top-N comparison is disabled with the reporting helpers below, so the
    # sweep no longer requires material.band_file.
    # if not band_file:
    #     raise ValueError("E_ref sweep requires material.band_file for top-N comparison")

    e_ref_values = parse_float_list(project_cfg.get("e_ref_values", project_cfg.get("e_ref_sweep")))
    if not e_ref_values:
        raise ValueError("E_ref sweep requires e_ref_values/e_ref_sweep")
    # top_n_list = _top_n_list_from_project_cfg(project_cfg)
    k_indices = parse_int_list(project_cfg.get("k_indices")) if project_cfg.get("k_indices") is not None else None
    workers = int(project_cfg.get("workers", 1))
    mode = project_cfg.get("mode", "gamma").lower()
    spin = material.get("spin", "all")
    pole_warning_mev = float(project_cfg.get("pole_warning_mev", 10.0))
    pole_danger_mev = float(project_cfg.get("pole_danger_mev", 1.0))
    fail_on_near_pole = _as_bool(project_cfg.get("fail_on_near_pole", False))
    compute_pole_diagnostics = _as_bool(project_cfg.get("compute_pole_diagnostics", False))
    compute_condition_number = _as_bool(project_cfg.get("compute_condition_number", False))
    nlow_state_list = _normalize_nlow_state_list(project_cfg)
    norb_fix_list = project_cfg["norb_fix_list"] if "norb_fix_list" in project_cfg else None

    print(f"[kp] Sweep downfold_method = fixed_schur")
    print(f"[kp] active indices = {_active_indices_from_project_cfg(project_cfg)}")
    print(f"[kp] E_ref values = {e_ref_values}")
    # print(f"[kp] top-N list = {top_n_list}")

    hamk = _load_hamk_with_energy_unit(hamk_file, material, mmap_mode="r")
    q1, q2 = load_Q_sets(qset1_file, qset2_file)
    hamk_index = int(plot_cfg.get("hamk_index", 0))
    hamk2d = hamk[hamk_index] if hamk.ndim == 3 else hamk
    num_layer_list, orb0, num_orb_per_layer_list = _orbital_layout_from_material(
        material,
        hamk2d,
        len(q1),
        spin=spin,
        mode=mode,
    )
    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    if k_indices is None:
        k_indices = list(range(hamk3d.shape[0]))
    # original_rows_all = _load_bands_from_text(band_file)
    # original_rows = [original_rows_all[i] for i in k_indices]
    q_count = len(q1)
    projection_spin = "all" if str(spin).lower() == "all" else "up"
    q2_for_projection = q2 if q2 is not None else q1
    norb_fix_list, _gauge_report = resolve_project_gauge_anchors(
        _selected_spin_project_input(np.asarray(hamk2d), spin),
        q_count,
        orb0,
        num_layer_list,
        spin=projection_spin,
        Qlayer_list=[[q1], [q2_for_projection]],
        num_orb_per_layer_list=num_orb_per_layer_list,
        nlow_state_list=nlow_state_list,
        norb_fix_list=norb_fix_list,
        gauge_config=project_cfg.get("gauge"),
        mode=mode,
    )

    # rows: list[SweepRow] = []
    for e_ref in e_ref_values:
        print(f"[kp] Sweeping E_ref={e_ref:.6f} eV over {len(k_indices)} k points")
        results = []

        def sweep_one(k_index: int):
            ham_for_projection = _selected_spin_project_input(hamk3d[k_index], spin)
            kwargs = {
                "spin": projection_spin,
                "Qlayer_list": [[q1], [q2]],
                "num_orb_per_layer_list": num_orb_per_layer_list,
                "nlow_state_list": nlow_state_list,
                "norb_fix_list": norb_fix_list,
                "downfold_method": "fixed_schur",
                "E_ref": float(e_ref),
                "pole_warning_mev": pole_warning_mev,
                "pole_danger_mev": pole_danger_mev,
                "fail_on_near_pole": fail_on_near_pole,
                "return_diagnostics": True,
                "mode": mode,
            }
            if _project_heff_full_kwargs_supports("compute_condition_number"):
                kwargs["compute_condition_number"] = compute_condition_number
            if _project_heff_full_kwargs_supports("compute_pole_diagnostics"):
                kwargs["compute_pole_diagnostics"] = compute_pole_diagnostics
            return project_heff_full(
                ham_for_projection,
                q_count,
                orb0,
                num_layer_list,
                **kwargs,
            )

        if workers > 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=workers)(delayed(sweep_one)(i) for i in k_indices)
            except Exception as ex:
                print(f"[kp] Parallel sweep failed, falling back to serial: {ex}")
                results = []
        if not results:
            results = [sweep_one(i) for i in k_indices]
        # heig_list = [np.asarray(r[1], dtype=np.float64) for r in results]
        diag_list = [r[3] for r in results if len(r) > 3]
        # metrics = compute_top_n_metrics(original_rows, heig_list, top_n_list)
        pole_vals = [d.pole_distance_min_mev for d in diag_list if d.pole_distance_min_mev is not None]
        global_pole = float(np.min(pole_vals)) if pole_vals else None
        # rows.append(SweepRow(e_ref=float(e_ref), global_pole_distance_min_mev=global_pole, metrics=metrics))
        if global_pole is not None:
            print(f"[kp]   global min |E_ref - E_high| = {global_pole:.3f} meV")
        # print_top_n_metrics(metrics)

    # out_dir = resolve(project_cfg.get("out_dir", "plots"))
    # if out_dir is None:
    #     out_dir = os.path.join(cfg_dir, "plots")
    # csv_out = resolve(project_cfg.get("sweep_csv", os.path.join(out_dir, "e_ref_sweep.csv")))
    # json_out = resolve(project_cfg.get("sweep_json", os.path.join(out_dir, "e_ref_sweep.json")))
    # write_sweep_csv(rows, csv_out)
    # write_sweep_json(rows, json_out)
    # print(f"[kp] Saved E_ref sweep CSV to: {csv_out}")
    # print(f"[kp] Saved E_ref sweep JSON to: {json_out}")


# -------------------- Band analysis --------------------

def _expand_orbital_pattern(segment: str) -> list[str]:
    return _expand_orbital_order_pattern(segment)


def expand_orbital_order_pattern(pattern: str) -> list[str]:
    """Expand 'I-s3p2d2,Mg-s2p2,I-s3p2d2' into a flat list of labels (length should match orb_per_layer)."""
    return _expand_orbital_order_pattern(pattern)


def _add_project_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", required=True, help="YAML config path")
    parser.add_argument("--active-indices", help="Comma-separated active band indices, e.g. 46,47")
    parser.add_argument("--downfold-method", choices=["first_order", "fixed_schur", "linearized_lowdin"])
    parser.add_argument("--e-ref", type=float, help="Reference energy for fixed_schur/linearized_lowdin in eV")
    parser.add_argument("--top-n", help="Comma-separated top-N values for diagnostics, e.g. 2,4,6,10,20")
    parser.add_argument("--k-indices", help="Comma-separated k indices; default is all")
    parser.add_argument("--pole-warning-mev", type=float)
    parser.add_argument("--pole-danger-mev", type=float)
    parser.add_argument("--fail-on-near-pole", action="store_true")
    parser.add_argument("--compute-pole-diagnostics", action="store_true")
    parser.add_argument("--compute-condition-number", action="store_true")


def _project_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "active_indices": args.active_indices,
        "downfold_method": args.downfold_method,
        "e_ref": args.e_ref,
        "top_n": args.top_n,
        "k_indices": args.k_indices,
        "pole_warning_mev": args.pole_warning_mev,
        "pole_danger_mev": args.pole_danger_mev,
        "fail_on_near_pole": args.fail_on_near_pole if args.fail_on_near_pole else None,
        "compute_pole_diagnostics": args.compute_pole_diagnostics if args.compute_pole_diagnostics else None,
        "compute_condition_number": args.compute_condition_number if args.compute_condition_number else None,
    }


def _model_output_dir_from_config(cfg_path: str) -> Path:
    config_path = Path(cfg_path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = normalize_case_config(yaml.safe_load(f), config_path=config_path)

    output_cfg = cfg.get("output", {})
    if not isinstance(output_cfg, dict) or not output_cfg.get("dir"):
        raise SystemExit("kp export -c/--config requires output.dir in the model config")

    model_output_dir = Path(str(output_cfg["dir"])).expanduser()
    if model_output_dir.is_absolute():
        return model_output_dir
    return config_path.parent / model_output_dir


def _run_standalone_export_command(args: argparse.Namespace, action_args: Sequence[str]) -> None:
    from .model import export as export_mod

    if args.all_examples:
        if getattr(args, "config", None):
            raise SystemExit("kp export --all-examples cannot be combined with -c/--config")
        all_output_root = action_args[0] if action_args else None
        if all_output_root is None:
            all_output_root = getattr(args, "export_output_dir", None)
        if all_output_root is None:
            raise SystemExit("kp export --all-examples requires <output_root>")
        report = export_mod.export_all_standalone_models(
            args.all_examples,
            all_output_root,
            force=bool(args.force),
            debug_files=bool(args.debug_files),
            dry_run=bool(args.dry_run),
        )
        print(f"[kp model] standalone exportable: {len(report['exportable'])}")
        print(f"[kp model] standalone exported: {len(report['exported'])}")
        print(f"[kp model] standalone blocked: {len(report['blocked'])}")
        for row in report["blocked"]:
            print(f"[kp model]   blocked: {row['model_output_dir']}: {row['reason']}")
        return

    if getattr(args, "config", None):
        if action_args:
            raise SystemExit("kp export -c/--config cannot be combined with <model_output_dir>")
        output_dir = getattr(args, "export_output_dir", None)
        if output_dir is None:
            raise SystemExit("kp export -c/--config requires -o/--out")
        model_output_dir = _model_output_dir_from_config(args.config)
    else:
        if len(action_args) < 2:
            raise SystemExit("kp export requires <model_output_dir> <output_dir>")
        model_output_dir = Path(action_args[0])
        output_dir = action_args[1]

    export_path = export_mod.export_standalone_model(
        model_output_dir,
        output_dir,
        force=bool(args.force),
        debug_files=bool(args.debug_files),
    )
    print(f"[kp model]   standalone export: {export_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kp", description="kp CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Inspect source bands and Q-block spectra")
    p_show.add_argument("-c", "--config", required=True, help="YAML config path")
    p_inspect = sub.add_parser("inspect", help="Inspect source bands and Q-block spectra")
    p_inspect.add_argument("-c", "--config", required=True, help="YAML config path")

    p_plot = sub.add_parser("plot", help="Plot scatter of band vs Q from config")
    p_plot.add_argument("-c", "--config", required=True, help="YAML config path")

    p_proj_short = sub.add_parser("proj", help="Project selected bands to Heff across Q")
    _add_project_arguments(p_proj_short)

    p_proj = sub.add_parser("project", help="Project selected bands to Heff across Q and plot")
    _add_project_arguments(p_proj)

    p_sweep = sub.add_parser("sweep", help="Sweep E_ref for fixed Schur downfolding")
    p_sweep.add_argument("-c", "--config", required=True, help="YAML config path")
    p_sweep.add_argument("--active-indices", help="Comma-separated active band indices, e.g. 46,47")
    p_sweep.add_argument("--e-ref-values", help="Comma-separated E_ref values in eV")
    p_sweep.add_argument("--top-n", help="Comma-separated top-N values for diagnostics")
    p_sweep.add_argument("--k-indices", help="Comma-separated k indices; default is all")
    p_sweep.add_argument("--pole-warning-mev", type=float)
    p_sweep.add_argument("--pole-danger-mev", type=float)
    p_sweep.add_argument("--fail-on-near-pole", action="store_true")
    p_sweep.add_argument("--compute-pole-diagnostics", action="store_true")
    p_sweep.add_argument("--compute-condition-number", action="store_true")

    p_symm = sub.add_parser("symm", help="Project TAPW symmetry representations into the KP basis")
    p_symm.add_argument("-c", "--config", required=True, help="YAML config path")
    p_symm.add_argument("--developer-outputs", action="store_true", help="Write developer-only projection matrices under diagnostics/")

    p_fit = sub.add_parser("fit", help="Fit/build a configured continuum model")
    p_fit.add_argument("-c", "--config", required=True, help="YAML model config path")
    p_fit.add_argument("--export-standalone", help="Write a minimal NumPy-only standalone model package")
    p_fit.add_argument("--debug-files", action="store_true", help="Include debug files in standalone export")

    p_model = sub.add_parser("model", help="Build/fit/export a configured continuum model")
    p_model.add_argument("model_action", nargs="?", help="Optional action, e.g. export-standalone")
    p_model.add_argument("model_args", nargs="*", help="Arguments for optional model action")
    p_model.add_argument("-c", "--config", help="YAML model config path")
    p_model.add_argument("--export-standalone", help="Write a minimal NumPy-only standalone model package")
    p_model.add_argument("--all-examples", help="Inventory/export all examples under this root")
    p_model.add_argument("--dry-run", action="store_true", help="Report standalone exportability without writing packages")
    p_model.add_argument("--force", action="store_true", help="Overwrite existing standalone export dirs")
    p_model.add_argument("--debug-files", action="store_true", help="Include debug files in standalone export")

    p_export = sub.add_parser("export", help="Export a standalone continuum model package")
    p_export.add_argument("model_output_dir", nargs="?", help="Model output directory")
    p_export.add_argument("output_dir", nargs="?", help="Standalone package output directory")
    p_export.add_argument("-c", "--config", help="YAML model config path")
    p_export.add_argument("-o", "--out", dest="export_output_dir", help="Standalone package output directory")
    p_export.add_argument("--all-examples", help="Inventory/export all examples under this root")
    p_export.add_argument("--dry-run", action="store_true", help="Report standalone exportability without writing packages")
    p_export.add_argument("--force", action="store_true", help="Overwrite existing standalone export dirs")
    p_export.add_argument("--debug-files", action="store_true", help="Include debug files in standalone export")

    return p


def main(argv: Sequence[str] | None = None) -> None:
    p = build_argparser()
    args, extra_args = p.parse_known_args(argv)
    if extra_args and not (args.cmd == "model" and args.model_action == "export-standalone"):
        p.error(f"unrecognized arguments: {' '.join(extra_args)}")
    if args.cmd in {"plot", "show", "inspect"}:
        cmd_plot_from_config(args.config)
    elif args.cmd in {"project", "proj"}:
        overrides = _project_overrides_from_args(args)
        cmd_project_from_config(args.config, overrides)
    elif args.cmd == "sweep":
        overrides = {
            "active_indices": args.active_indices,
            "e_ref_values": args.e_ref_values,
            "top_n": args.top_n,
            "k_indices": args.k_indices,
            "pole_warning_mev": args.pole_warning_mev,
            "pole_danger_mev": args.pole_danger_mev,
            "fail_on_near_pole": args.fail_on_near_pole if args.fail_on_near_pole else None,
            "compute_pole_diagnostics": args.compute_pole_diagnostics if args.compute_pole_diagnostics else None,
            "compute_condition_number": args.compute_condition_number if args.compute_condition_number else None,
        }
        cmd_sweep_from_config(args.config, overrides)
    elif args.cmd == "symm":
        run_symmetry_projection_from_config(
            args.config,
            developer_outputs=True if args.developer_outputs else None,
        )
    elif args.cmd == "export":
        action_args = [x for x in [args.model_output_dir, args.output_dir] if x is not None]
        _run_standalone_export_command(args, action_args)
    elif args.cmd in {"model", "fit"}:
        if args.cmd == "model" and args.model_action == "export-standalone":
            action_args = list(args.model_args) + list(extra_args)
            _run_standalone_export_command(args, action_args)
            return
        if not args.config:
            raise SystemExit("kp model requires --config unless using 'export-standalone'")
        from .model.pipeline import run_configured_model

        results = run_configured_model(args.config)
        model_cfg = results.get("configured_model")
        moire_cfg = results.get("moire_config")
        comparison = results.get("comparison")
        plot_comparison = results.get("plot_comparison")
        if model_cfg is not None and moire_cfg is not None:
            q1 = len(moire_cfg.Q_set1) if moire_cfg.Q_set1 is not None else 0
            q2 = len(moire_cfg.Q_set2) if moire_cfg.Q_set2 is not None else 0
            dim = q1 * int(moire_cfg.n_orb1) + q2 * int(moire_cfg.n_orb2)
            sym_ops = [op.get("name") for op in model_cfg.symmetry_source_metadata.get("operations", [])]
            print("[kp model] Completed configured continuum model")
            print(f"[kp model]   config: {model_cfg.path}")
            print(f"[kp model]   output: {model_cfg.output_dir}")
            print(f"[kp model]   basis: dim={dim}, Q=({q1}, {q2}), n_orb={model_cfg.n_orb}")
            print(f"[kp model]   fit k-points: {len(model_cfg.fit_indices)}, band k-points: {len(moire_cfg.kpoints)}")
            print(f"[kp model]   bM source: {model_cfg.bM_diagnostics.get('source', 'unknown')}")
            intra_count = len(getattr(moire_cfg, "intra_harmonics_map", {}) or {})
            inter_count = len(getattr(moire_cfg, "inter_harmonics_map", {}) or {})
            print(f"[kp model]   harmonics: intra={intra_count}, inter={inter_count}")
            if sym_ops:
                print(f"[kp model]   symmetry: {', '.join(str(op) for op in sym_ops)}")
            if results.get("model_log"):
                print(f"[kp model]   detailed log: {results['model_log']}")
            if results.get("runtime_s") is not None:
                print(f"[kp model]   runtime: {float(results['runtime_s']):.2f} s")
        if results.get("band_plot"):
            print(f"[kp model]   band plot: {results['band_plot']}")
        if results.get("all_band_plot"):
            print(f"[kp model]   all-band plot: {results['all_band_plot']}")
        if comparison:
            rms = float(comparison["rms_error"])
            max_abs = float(comparison["max_abs_error"])
            print(f"[kp model]   RMS error: {rms:.6e} eV ({1000.0 * rms:.3f} meV)")
            print(f"[kp model]   Max error: {max_abs:.6e} eV ({1000.0 * max_abs:.3f} meV)")
        if plot_comparison:
            rms_mev = float(plot_comparison.get("rms_error_mev", 1000.0 * float(plot_comparison["rms_error"])))
            max_mev = float(plot_comparison.get("max_abs_error_mev", 1000.0 * float(plot_comparison["max_abs_error"])))
            bands = int(plot_comparison.get("num_bands", 0))
            align = str(plot_comparison.get("align", "none"))
            print(f"[kp model]   plot bands RMS: {rms_mev:.3f} meV, Max: {max_mev:.3f} meV (bands={bands}, align={align})")
            if plot_comparison.get("model_alignment_shift_meV") is not None:
                print(f"[kp model]   plot alignment shift: {float(plot_comparison['model_alignment_shift_meV']):.3f} meV")
        all_band_comparison = results.get("all_band_plot_comparison")
        if all_band_comparison:
            rms_mev = float(
                all_band_comparison.get(
                    "rms_error_mev",
                    1000.0 * float(all_band_comparison["rms_error"]),
                )
            )
            max_mev = float(
                all_band_comparison.get(
                    "max_abs_error_mev",
                    1000.0 * float(all_band_comparison["max_abs_error"]),
                )
            )
            bands = int(all_band_comparison.get("num_bands", 0))
            align = str(all_band_comparison.get("align", "none"))
            print(f"[kp model]   all-band RMS: {rms_mev:.3f} meV, Max: {max_mev:.3f} meV (bands={bands}, align={align})")
        if model_cfg is None:
            raise RuntimeError("standalone export requires configured model results")
        from .model.export import export_standalone_model

        standalone_dir = (
            Path(args.export_standalone)
            if args.export_standalone
            else (
                Path(model_cfg.output_dir)
                if _config_path_uses_canonical_case(args.config)
                else _default_standalone_export_dir(Path(model_cfg.output_dir))
            )
        )
        export_path = export_standalone_model(
            model_cfg.output_dir,
            standalone_dir,
            force=True,
            debug_files=bool(args.debug_files),
        )
        _record_standalone_export(Path(model_cfg.output_dir), Path(export_path))
        print(f"[kp model]   standalone export: {export_path}")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
