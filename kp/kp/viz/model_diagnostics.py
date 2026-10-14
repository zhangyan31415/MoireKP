from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from ..plot_style import (
    KP_BAND_BOX_ASPECT,
    KP_BAND_FIGSIZE,
    KP_DPI,
    KP_LEGEND_KWARGS,
    KP_MODEL_STYLE,
    KP_REFERENCE_STYLE,
    apply_kp_axis_style,
    kp_plot_rc_context,
    kp_font_family,
    relative_energy_ylabel,
)
from ..symmetry.geometry import sector_qset
from ..model.core import (
    MoireConfig,
    generate_kpath_from_file,
    generate_kpath_from_symbols,
)
from ..model import configuration
from ..model import harmonic_selection
from ..model import validation


def _hamiltonian_element_comparison_arrays(
    model_hamiltonians: np.ndarray,
    heff_hamiltonians: np.ndarray,
    harmonic_mask: np.ndarray,
    *,
    positions: Sequence[int] | None = None,
    qset1: np.ndarray | None = None,
    qset2: np.ndarray | None = None,
    n_orb: tuple[int, int] | None = None,
    subtract_layer_diagonal_mean: bool = False,
    q_order: str = "native",
    q_norm_tol: float = 1.0e-6,
) -> dict[str, Any]:
    model = np.asarray(model_hamiltonians, dtype=np.complex128)
    heff = np.asarray(heff_hamiltonians, dtype=np.complex128)
    mask = np.asarray(harmonic_mask, dtype=bool)
    if model.ndim == 2:
        model = model[None, :, :]
    if heff.ndim == 2:
        heff = heff[None, :, :]
    if model.shape != heff.shape:
        raise ValueError(f"model/heff Hamiltonian arrays must have same shape, got {model.shape} and {heff.shape}")
    if model.ndim != 3 or model.shape[-1] != model.shape[-2]:
        raise ValueError(f"Hamiltonian arrays must have shape (Nk, dim, dim), got {model.shape}")
    if mask.shape != model.shape[-2:]:
        raise ValueError(f"harmonic mask shape {mask.shape} does not match Hamiltonian dimension {model.shape[-2:]}")
    if positions is None:
        selected = list(range(model.shape[0]))
    else:
        selected = [int(item) for item in positions]
    if not selected:
        raise ValueError("Hamiltonian element comparison requires at least one k-point position")
    for pos in selected:
        if pos < 0 or pos >= model.shape[0]:
            raise IndexError(f"k-point position {pos} is outside 0..{model.shape[0] - 1}")
    model_selected = model[np.asarray(selected, dtype=int)]
    heff_selected = heff[np.asarray(selected, dtype=int)]

    display_order = list(range(model.shape[-1]))
    q_block_boundaries: list[int] = []
    q_shell_boundaries: list[int] = []
    layer_boundaries: list[int] = []
    display_layers: np.ndarray | None = None
    if qset1 is not None and qset2 is not None and n_orb is not None:
        display = _q_block_display_metadata(
            qset1,
            qset2,
            n_orb,
            dim=model.shape[-1],
            q_order=q_order,
            q_norm_tol=q_norm_tol,
        )
        display_order = display["order"]
        q_block_boundaries = display["q_block_boundaries"]
        q_shell_boundaries = display["q_shell_boundaries"]
        layer_boundaries = display["layer_boundaries"]
        display_layers = display["layers"]
    else:
        display_layers = np.ones(model.shape[-1], dtype=np.int16)

    heff_offsets: list[list[float]] = []
    model_offsets: list[list[float]] = []
    if subtract_layer_diagonal_mean:
        heff_selected, heff_offsets = _subtract_layer_diagonal_means(heff_selected, display_layers)
        model_selected, model_offsets = _subtract_layer_diagonal_means(model_selected, display_layers)

    order = np.asarray(display_order, dtype=int)
    model_selected = model_selected[:, order][:, :, order]
    heff_selected = heff_selected[:, order][:, :, order]
    mask = mask[np.ix_(order, order)]
    diff_selected = model_selected - heff_selected
    mask3 = mask[None, :, :]
    heff_abs = np.abs(heff_selected)
    model_abs = np.abs(model_selected)
    diff_abs = np.abs(diff_selected)
    return {
        "positions": selected,
        "heff": heff_abs,
        "model": model_abs,
        "diff": diff_abs,
        "support_heff": np.where(mask3, heff_abs, np.nan),
        "support_model": np.where(mask3, model_abs, np.nan),
        "support_diff": np.where(mask3, diff_abs, np.nan),
        "mask": mask,
        "support_fraction": float(np.count_nonzero(mask) / mask.size),
        "display_order": display_order,
        "q_block_boundaries": q_block_boundaries,
        "q_shell_boundaries": q_shell_boundaries,
        "layer_boundaries": layer_boundaries,
        "q_order": str(q_order).strip().lower(),
        "diag_offsets": {
            "heff": heff_offsets,
            "model": model_offsets,
        },
    }



def _q_block_display_metadata(
    qset1: np.ndarray,
    qset2: np.ndarray,
    n_orb: tuple[int, int],
    *,
    dim: int,
    q_order: str = "native",
    q_norm_tol: float = 1.0e-6,
) -> dict[str, Any]:
    rows = harmonic_selection._model_row_metadata_for_harmonic_scan(qset1, qset2, n_orb)
    if len(rows) != int(dim):
        raise ValueError(f"basis metadata dimension {len(rows)} does not match Hamiltonian dimension {dim}")
    order_mode = str(q_order).strip().lower()
    if order_mode not in {"native", "norm_shell"}:
        raise ValueError(f"q_order must be 'native' or 'norm_shell', got {q_order!r}")
    tol = float(q_norm_tol)
    if tol <= 0.0:
        raise ValueError(f"q_norm_tol must be positive, got {q_norm_tol}")

    shell_index: dict[tuple[int, int], int] = {}
    polar_angle: dict[tuple[int, int], float] = {}
    for layer in (1, 2):
        q_rows = {
            int(row["q_index"]): np.asarray(row["q"], dtype=float)
            for row in rows
            if int(row["layer"]) == layer
        }
        shell_norms = harmonic_selection._unique_nonnegative_norms(
            [float(np.linalg.norm(qvec)) for qvec in q_rows.values()],
            tol=tol,
        )
        for q_index, qvec in q_rows.items():
            norm = float(np.linalg.norm(qvec))
            shell_index[(layer, q_index)] = int(
                np.argmin(np.abs(np.asarray(shell_norms, dtype=float) - norm))
            )
            angle = 0.0 if norm <= tol else float(np.mod(np.arctan2(qvec[1], qvec[0]), 2.0 * np.pi))
            polar_angle[(layer, q_index)] = angle

    def display_key(idx: int) -> tuple[Any, ...]:
        row = rows[idx]
        layer = int(row["layer"])
        q_index = int(row["q_index"])
        orbital = int(row["orbital"])
        if order_mode == "native":
            return (layer, q_index, orbital)
        return (
            layer,
            shell_index[(layer, q_index)],
            polar_angle[(layer, q_index)],
            q_index,
            orbital,
        )

    order = sorted(
        range(len(rows)),
        key=display_key,
    )
    ordered_rows = [rows[idx] for idx in order]
    layers = np.asarray([int(row["layer"]) for row in ordered_rows], dtype=np.int16)
    q_block_boundaries: list[int] = []
    q_shell_boundaries: list[int] = []
    layer_boundaries: list[int] = []
    for idx in range(1, len(ordered_rows)):
        prev = ordered_rows[idx - 1]
        curr = ordered_rows[idx]
        if int(prev["layer"]) != int(curr["layer"]):
            layer_boundaries.append(idx)
            q_block_boundaries.append(idx)
        elif int(prev["q_index"]) != int(curr["q_index"]):
            q_block_boundaries.append(idx)
            if order_mode == "norm_shell":
                prev_key = (int(prev["layer"]), int(prev["q_index"]))
                curr_key = (int(curr["layer"]), int(curr["q_index"]))
                if shell_index[prev_key] != shell_index[curr_key]:
                    q_shell_boundaries.append(idx)
    return {
        "order": [int(idx) for idx in order],
        "layers": layers,
        "q_block_boundaries": q_block_boundaries,
        "q_shell_boundaries": q_shell_boundaries,
        "layer_boundaries": layer_boundaries,
        "q_order": order_mode,
    }



def _subtract_layer_diagonal_means(
    hamiltonians: np.ndarray,
    layers: np.ndarray,
) -> tuple[np.ndarray, list[list[float]]]:
    adjusted = np.asarray(hamiltonians, dtype=np.complex128).copy()
    layer_arr = np.asarray(layers, dtype=int)
    unique_layers = [int(item) for item in np.unique(layer_arr)]
    offsets: list[list[float]] = []
    for ik in range(adjusted.shape[0]):
        row_offsets: list[float] = []
        diag = np.diagonal(adjusted[ik]).copy()
        for layer in unique_layers:
            indices = np.flatnonzero(layer_arr == layer)
            if indices.size == 0:
                row_offsets.append(0.0)
                continue
            offset = complex(np.mean(diag[indices]))
            adjusted[ik, indices, indices] -= offset
            row_offsets.append(float(np.real_if_close(offset).real))
        offsets.append(row_offsets)
    return adjusted, offsets



def _finite_heatmap_vmax(*arrays: np.ndarray) -> float:
    values = []
    for arr in arrays:
        data = np.asarray(arr, dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size:
            values.append(float(np.max(finite)))
    vmax = max(values) if values else 0.0
    return vmax if vmax > 0.0 else 1.0



def _matrix_index_ticks(dim: int, *, max_ticks: int = 8) -> list[int]:
    size = int(dim)
    if size <= 0:
        return []
    if size <= max_ticks:
        return list(range(size))
    step = max(1, int(math.ceil((size - 1) / float(max_ticks - 1))))
    ticks = list(range(0, size, step))
    if ticks[-1] != size - 1:
        ticks.append(size - 1)
    return ticks



def save_hamiltonian_element_comparison_plot(
    model_hamiltonians: np.ndarray,
    heff_hamiltonians: np.ndarray,
    path: str | Path,
    *,
    harmonic_mask: np.ndarray,
    positions: Sequence[int] | None = None,
    k_indices: Sequence[int] | None = None,
    title: str | None = None,
    qset1: np.ndarray | None = None,
    qset2: np.ndarray | None = None,
    n_orb: tuple[int, int] | None = None,
    subtract_layer_diagonal_mean: bool = True,
    view: str = "full",
    q_order: str = "native",
    q_norm_tol: float = 1.0e-6,
) -> Path:
    panels = _hamiltonian_element_comparison_arrays(
        model_hamiltonians,
        heff_hamiltonians,
        harmonic_mask,
        positions=positions,
        qset1=qset1,
        qset2=qset2,
        n_orb=n_orb,
        subtract_layer_diagonal_mean=subtract_layer_diagonal_mean,
        q_order=q_order,
        q_norm_tol=q_norm_tol,
    )
    if view not in {"full", "selected_support"}:
        raise ValueError("Hamiltonian element comparison view must be 'full' or 'selected_support'")
    prefix = "support_" if view == "selected_support" else ""
    heff = np.asarray(panels[f"{prefix}heff"], dtype=float)
    model = np.asarray(panels[f"{prefix}model"], dtype=float)
    diff = np.asarray(panels[f"{prefix}diff"], dtype=float)
    nrows = int(heff.shape[0])
    if k_indices is None:
        labels = [f"k position {idx}" for idx in panels["positions"]]
    else:
        labels = [f"k index {int(idx)}" for idx in k_indices]
        if len(labels) != nrows:
            raise ValueError(f"k_indices length {len(labels)} does not match plotted row count {nrows}")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out = out.with_suffix(".pdf")
    raster_dpi = max(int(KP_DPI), 360)
    main_vmax = _finite_heatmap_vmax(heff, model)
    diff_vmax = _finite_heatmap_vmax(diff)
    dim = int(heff.shape[-1])
    index_ticks = _matrix_index_ticks(dim)
    q_boundaries = [int(item) for item in panels.get("q_block_boundaries", [])]
    q_shell_boundaries = [int(item) for item in panels.get("q_shell_boundaries", [])]
    layer_boundaries = [int(item) for item in panels.get("layer_boundaries", [])]
    diag_offsets = panels.get("diag_offsets", {})

    def format_offsets(kind: str, row_index: int) -> str:
        rows = diag_offsets.get(kind, []) if isinstance(diag_offsets, Mapping) else []
        if not rows or row_index >= len(rows):
            return ""
        values = rows[row_index]
        if not values:
            return ""
        parts = [f"L{idx + 1}={float(value):+.4g}" for idx, value in enumerate(values)]
        return "diag mean subtracted\n" + ", ".join(parts) + " eV"

    def draw_boundaries(ax: Any) -> None:
        for boundary in q_boundaries:
            pos = float(boundary) - 0.5
            ax.axhline(pos, color="0.65", linewidth=0.3, alpha=0.85)
            ax.axvline(pos, color="0.65", linewidth=0.3, alpha=0.85)
        for boundary in q_shell_boundaries:
            pos = float(boundary) - 0.5
            ax.axhline(pos, color="0.32", linewidth=0.7, alpha=0.95)
            ax.axvline(pos, color="0.32", linewidth=0.7, alpha=0.95)
        for boundary in layer_boundaries:
            pos = float(boundary) - 0.5
            ax.axhline(pos, color="black", linewidth=1.0, alpha=0.9)
            ax.axvline(pos, color="black", linewidth=1.0, alpha=0.9)

    with kp_plot_rc_context():
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=3,
            figsize=(8.4, max(2.6, 2.35 * nrows)),
            dpi=KP_DPI,
            squeeze=False,
            constrained_layout=True,
        )
        main_cmap = plt.get_cmap("viridis").copy()
        diff_cmap = plt.get_cmap("magma").copy()
        main_cmap.set_bad(color="white")
        diff_cmap.set_bad(color="white")
        main_norm = Normalize(vmin=0.0, vmax=main_vmax)
        diff_norm = Normalize(vmin=0.0, vmax=diff_vmax)
        columns = [
            ("Heff |H|", heff, main_cmap, main_norm),
            ("Model |H|", model, main_cmap, main_norm),
            ("|Model - Heff|", diff, diff_cmap, diff_norm),
        ]
        for irow in range(nrows):
            for icol, (col_title, data, cmap, norm) in enumerate(columns):
                ax = axes[irow, icol]
                ax.imshow(np.ma.masked_invalid(data[irow]), origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
                draw_boundaries(ax)
                ax.set_xticks(index_ticks)
                ax.set_yticks(index_ticks)
                ax.tick_params(axis="both", labelsize=6, length=2)
                if irow != nrows - 1:
                    ax.set_xticklabels([])
                if icol != 0:
                    ax.set_yticklabels([])
                ax.set_box_aspect(1.0)
                if irow == 0:
                    ax.set_title(col_title)
                if icol == 0:
                    ax.set_ylabel(labels[irow], rotation=90, labelpad=14)
                    text = format_offsets("heff", irow)
                    if text:
                        ax.text(
                            0.02,
                            0.98,
                            text,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=5.5,
                            color="white",
                            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"},
                        )
                elif icol == 1:
                    text = format_offsets("model", irow)
                    if text:
                        ax.text(
                            0.02,
                            0.98,
                            text,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=5.5,
                            color="white",
                            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.5, "edgecolor": "none"},
                        )
                if irow == nrows - 1:
                    ax.set_xlabel("column: layer/Q/orbital")
        if title:
            order_suffix = "\nQ ordered by |Q| shell" if panels.get("q_order") == "norm_shell" else ""
            if view == "selected_support":
                support_fraction = 100.0 * float(panels["support_fraction"])
                fig.suptitle(
                    f"{title}{order_suffix}\n"
                    f"selected harmonic support ({support_fraction:.2f}% shown; white = excluded, not zero)"
                )
            else:
                fig.suptitle(f"{title}{order_suffix}\nfull matrix")
        main_sm = ScalarMappable(norm=main_norm, cmap=main_cmap)
        diff_sm = ScalarMappable(norm=diff_norm, cmap=diff_cmap)
        fig.colorbar(main_sm, ax=axes[:, :2].ravel().tolist(), fraction=0.025, pad=0.02, label="|H| (eV)")
        fig.colorbar(diff_sm, ax=axes[:, 2].ravel().tolist(), fraction=0.025, pad=0.02, label="|ΔH| (eV)")
        fig.savefig(out, dpi=raster_dpi, bbox_inches="tight")
        fig.savefig(pdf_out, bbox_inches="tight")
        plt.close(fig)
    return out



def _normalise_kpath_label(label: Any) -> str:
    text = str(label).strip()
    if text.lower() in {"gamma", "g", "γ"}:
        return r"$\Gamma$"
    return text



def _plot_comparison_filename(plot_config: Mapping[str, Any] | None) -> str:
    plot_options = dict(plot_config or {})
    align = str(plot_options.get("align", "none")).lower()
    suffix = "_aligned" if align in {"top", "top_band", "top-band", "bottom", "bottom_band", "bottom-band"} else ""
    if plot_options.get("bottom_bands") is not None:
        return f"comparison_bottom{int(plot_options['bottom_bands'])}{suffix}.json"
    if plot_options.get("top_bands") is not None:
        return f"comparison_top{int(plot_options['top_bands'])}{suffix}.json"
    return "comparison_plot.json"



def _plot_axis_from_kpath(config: configuration.ConfiguredModel, npoints: int) -> tuple[np.ndarray, list[float] | None, list[str] | None]:
    if not config.kpath_config:
        return np.arange(npoints, dtype=float), None, None
    if config.band_indices is not None:
        return np.arange(npoints, dtype=float), None, None

    tmat = config.kpath_config.get("tmat")
    if tmat is None:
        return np.arange(npoints, dtype=float), None, None
    labels = config.kpath_config.get("labels")
    coordinates = config.kpath_config.get("coordinates")
    if labels is not None or coordinates is not None:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ValueError("inline kpath.labels must be a sequence")
        if not isinstance(coordinates, Mapping):
            raise ValueError("inline kpath.coordinates must be a mapping")
        points_per_segment = config.kpath_config.get(
            "points_per_segment", config.kpath_config.get("segment_points")
        )
        if points_per_segment is None:
            raise ValueError("inline kpath.points_per_segment is required")
        generated = generate_kpath_from_symbols(
            Tmat=np.asarray(tmat, dtype=float),
            symbols=[str(label) for label in labels],
            segment_points=int(points_per_segment),
            phase_deg=float(config.rotation_deg),
            coords=coordinates,
        )
    else:
        file_path = configuration._resolve_path(config.kpath_config.get("file"), config.path.parent)
        if file_path is None or not file_path.exists():
            return np.arange(npoints, dtype=float), None, None
        generated = generate_kpath_from_file(
            Tmat=np.asarray(tmat, dtype=float),
            file_path=file_path,
            phase_deg=float(config.rotation_deg),
            segment_points=None
            if config.kpath_config.get("segment_points") is None
            else int(config.kpath_config.get("segment_points")),
        )
    x = np.asarray(generated.x, dtype=float)
    if x.shape[0] != npoints:
        return np.arange(npoints, dtype=float), None, None
    return x, list(generated.x_ticks), [_normalise_kpath_label(label) for label in generated.labels_ticks]



def _band_plot_title(config: configuration.ConfiguredModel) -> str:
    if config.raw.get("title"):
        return str(config.raw["title"])
    source_plot = config.source_raw.get("plot", {})
    if isinstance(source_plot, Mapping) and source_plot.get("title"):
        return str(source_plot["title"])
    return config.path.stem



def _band_overlap_weights(
    model_eigvecs: np.ndarray,
    reference_eigvecs: np.ndarray,
    *,
    model_eigvals: np.ndarray | None = None,
    reference_eigvals: np.ndarray | None = None,
    degeneracy_tol: float = 0.0,
) -> np.ndarray:
    model_vec = np.asarray(model_eigvecs)
    ref_vec = np.asarray(reference_eigvecs)
    if model_vec.ndim != 3 or ref_vec.ndim != 3:
        raise ValueError(
            "band overlap expects eigenvectors with shape (Nk,dim,N), "
            f"got {model_vec.shape} and {ref_vec.shape}"
        )
    if model_vec.shape != ref_vec.shape:
        raise ValueError(f"band overlap eigenvector shapes differ: model={model_vec.shape}, reference={ref_vec.shape}")
    weights = np.abs(np.einsum("kdn,kdn->kn", np.conjugate(model_vec), ref_vec)) ** 2
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    tol = float(degeneracy_tol)
    if tol <= 0.0:
        return weights
    if model_eigvals is None or reference_eigvals is None:
        return weights
    model_val = np.asarray(model_eigvals, dtype=float)
    ref_val = np.asarray(reference_eigvals, dtype=float)
    if model_val.shape != weights.shape or ref_val.shape != weights.shape:
        raise ValueError(
            "band overlap degeneracy eigvals must match overlap shape, "
            f"got model={model_val.shape}, reference={ref_val.shape}, overlap={weights.shape}"
        )

    nk, nbands = weights.shape
    clustered = weights.copy()
    for ik in range(nk):
        start = 0
        for ib in range(nbands - 1):
            model_gap = abs(float(model_val[ik, ib + 1] - model_val[ik, ib]))
            ref_gap = abs(float(ref_val[ik, ib + 1] - ref_val[ik, ib]))
            if model_gap <= tol or ref_gap <= tol:
                continue
            stop = ib + 1
            if stop - start > 1:
                model_subspace = model_vec[ik, :, start:stop]
                ref_subspace = ref_vec[ik, :, start:stop]
                value = float(np.linalg.norm(model_subspace.conj().T @ ref_subspace, ord="fro") ** 2 / (stop - start))
                clustered[ik, start:stop] = np.clip(value, 0.0, 1.0)
            start = stop
        stop = nbands
        if stop - start > 1:
            model_subspace = model_vec[ik, :, start:stop]
            ref_subspace = ref_vec[ik, :, start:stop]
            value = float(np.linalg.norm(model_subspace.conj().T @ ref_subspace, ord="fro") ** 2 / (stop - start))
            clustered[ik, start:stop] = np.clip(value, 0.0, 1.0)
    return clustered



def save_band_comparison_plot(
    model_eigvals: np.ndarray,
    heff_eigvals: np.ndarray,
    path: str | Path,
    *,
    band_slice: Sequence[int] | None = None,
    plot_config: Mapping[str, Any] | None = None,
    x: Sequence[float] | None = None,
    x_ticks: Sequence[float] | None = None,
    x_ticklabels: Sequence[str] | None = None,
    title: str | None = None,
    overlap_weights: np.ndarray | None = None,
    current_support_eigvals: np.ndarray | None = None,
    reference_label: str = "Reference",
) -> Path:
    model = np.asarray(model_eigvals, dtype=float)
    heff = np.asarray(heff_eigvals)
    if heff.ndim == 3:
        heff = np.linalg.eigvalsh(heff)
    heff = np.asarray(heff, dtype=float)
    if model.ndim != 2 or heff.ndim != 2:
        raise ValueError(f"model/heff eigvals must be 2D after eigensolve, got {model.shape} and {heff.shape}")
    if model.shape[0] != heff.shape[0]:
        raise ValueError(f"k-point counts differ: model={model.shape[0]}, heff={heff.shape[0]}")
    current_support = None
    if current_support_eigvals is not None:
        current_support = np.asarray(current_support_eigvals)
        if current_support.ndim == 3:
            current_support = np.linalg.eigvalsh(current_support)
        current_support = np.asarray(current_support, dtype=float)
        if current_support.ndim != 2:
            raise ValueError(
                "current_support_eigvals must be 2D after eigensolve, "
                f"got {current_support.shape}"
            )
        if current_support.shape[0] != heff.shape[0]:
            raise ValueError(
                "current-support/heff k-point counts differ: "
                f"current_support={current_support.shape[0]}, heff={heff.shape[0]}"
            )

    nbands = min(model.shape[1], heff.shape[1])
    if current_support is not None:
        nbands = min(nbands, current_support.shape[1])
    plot_options = dict(plot_config or {})
    start, stop = validation._resolve_plot_band_slice(nbands=nbands, metric_band_slice=band_slice, plot_config=plot_options)
    model_sorted = np.sort(model, axis=1)[:, :nbands]
    heff_sorted = np.sort(heff, axis=1)[:, :nbands]
    current_support_sorted = np.sort(current_support, axis=1)[:, :nbands] if current_support is not None else None
    overlap_sorted = None
    if overlap_weights is not None:
        overlap = np.asarray(overlap_weights, dtype=float)
        if overlap.ndim != 2:
            raise ValueError(f"overlap_weights must be 2D, got {overlap.shape}")
        if overlap.shape[0] != model.shape[0] or overlap.shape[1] < nbands:
            raise ValueError(
                "overlap_weights must have one row per k-point and at least the plotted band count, "
                f"got overlap={overlap.shape}, model={model.shape}, heff={heff.shape}"
            )
        overlap_sorted = np.clip(overlap[:, :nbands], 0.0, 1.0)
    model_sel = model_sorted[:, start:stop]
    heff_sel = heff_sorted[:, start:stop]
    current_support_sel = current_support_sorted[:, start:stop] if current_support_sorted is not None else None
    overlap_sel = overlap_sorted[:, start:stop] if overlap_sorted is not None else None
    align = str(plot_options.get("align", "none")).lower()
    model_ref = 0.0
    heff_ref = 0.0
    current_support_ref = 0.0
    if align in {"top", "top_band", "top-band"}:
        model_ref = float(np.max(model_sel[:, -1]))
        heff_ref = float(np.max(heff_sel[:, -1]))
        if current_support_sel is not None:
            current_support_ref = float(np.max(current_support_sel[:, -1]))
        model_sel = model_sel - model_ref
        heff_sel = heff_sel - heff_ref
        if current_support_sel is not None:
            current_support_sel = current_support_sel - current_support_ref
    elif align in {"bottom", "bottom_band", "bottom-band"}:
        model_ref = float(np.min(model_sel[:, 0]))
        heff_ref = float(np.min(heff_sel[:, 0]))
        if current_support_sel is not None:
            current_support_ref = float(np.min(current_support_sel[:, 0]))
        model_sel = model_sel - model_ref
        heff_sel = heff_sel - heff_ref
        if current_support_sel is not None:
            current_support_sel = current_support_sel - current_support_ref
    elif align not in {"none", "false", "0"}:
        raise ValueError(f"Unsupported bands.plot.align={plot_options.get('align')!r}; expected 'none', 'top', or 'bottom'")
    plot_all_bands = bool(plot_options.get("plot_all_bands", plot_options.get("show_all_bands", True)))
    if plot_all_bands:
        model_plot = model_sorted - model_ref
        heff_plot = heff_sorted - heff_ref
        current_support_plot = (
            current_support_sorted - current_support_ref if current_support_sorted is not None else None
        )
        overlap_plot = overlap_sorted
    else:
        model_plot = model_sel
        heff_plot = heff_sel
        current_support_plot = current_support_sel
        overlap_plot = overlap_sel

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray(x, dtype=float) if x is not None else np.arange(model_plot.shape[0], dtype=float)
    if x_values.shape[0] != model_plot.shape[0]:
        raise ValueError(f"plot x-axis length {x_values.shape[0]} does not match k-point count {model_sel.shape[0]}")
    figsize_raw = plot_options.get("figsize", list(KP_BAND_FIGSIZE))
    if not isinstance(figsize_raw, Sequence) or isinstance(figsize_raw, (str, bytes)) or len(figsize_raw) != 2:
        raise ValueError(f"bands.plot.figsize must be [width, height], got {figsize_raw!r}")
    dpi = int(plot_options.get("dpi", KP_DPI))
    fig, ax = plt.subplots(figsize=(float(figsize_raw[0]), float(figsize_raw[1])), dpi=dpi)
    overlap_scalar = None
    overlap_label = str(plot_options.get("overlap_label", "Wavefunction overlap"))
    for ib in range(heff_plot.shape[1]):
        ax.plot(x_values, heff_plot[:, ib], **KP_REFERENCE_STYLE)
    current_support_style = {
        "color": "#E69F00",
        "linestyle": "--",
        "linewidth": 1.05,
        "alpha": 0.86,
    }
    if current_support_plot is not None:
        for ib in range(current_support_plot.shape[1]):
            ax.plot(x_values, current_support_plot[:, ib], **current_support_style)
    if overlap_plot is not None:
        cmap = plt.get_cmap(str(plot_options.get("overlap_cmap", "Blues")))
        colorbar_values = overlap_sel if overlap_sel is not None and overlap_sel.size else overlap_plot
        default_vmin = float(np.nanmin(colorbar_values))
        default_vmax = float(np.nanmax(colorbar_values))
        vmin = float(plot_options.get("overlap_vmin", default_vmin))
        vmax = float(plot_options.get("overlap_vmax", default_vmax))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            center = 0.5 * (vmin + vmax)
            pad = max(1.0e-6, 0.01 * max(abs(center), 1.0))
            vmin = center - pad
            vmax = center + pad
        norm = Normalize(
            vmin=vmin,
            vmax=vmax,
        )
        linewidth = float(plot_options.get("overlap_linewidth", KP_MODEL_STYLE.get("linewidth", 1.0)))
        alpha = float(plot_options.get("overlap_alpha", KP_MODEL_STYLE.get("alpha", 0.95)))
        linestyle = str(plot_options.get("overlap_linestyle", KP_MODEL_STYLE.get("linestyle", "-")))
        for ib in range(model_plot.shape[1]):
            points = np.column_stack([x_values, model_plot[:, ib]])
            if points.shape[0] < 2:
                ax.plot(
                    x_values,
                    model_plot[:, ib],
                    color=cmap(norm(float(overlap_plot[0, ib]))),
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle=linestyle,
                )
                continue
            segments = np.stack([points[:-1], points[1:]], axis=1)
            values = 0.5 * (overlap_plot[:-1, ib] + overlap_plot[1:, ib])
            collection = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidths=linewidth,
                alpha=alpha,
                linestyles=linestyle,
            )
            collection.set_array(np.asarray(values, dtype=float))
            ax.add_collection(collection)
        overlap_scalar = ScalarMappable(norm=norm, cmap=cmap)
        overlap_scalar.set_array([])
    else:
        for ib in range(model_plot.shape[1]):
            ax.plot(x_values, model_plot[:, ib], **KP_MODEL_STYLE)
    if x_ticks is not None and x_ticklabels is not None and len(x_ticks) == len(x_ticklabels):
        ax.set_xticks([float(item) for item in x_ticks])
        ax.set_xticklabels(list(x_ticklabels))
        for tick in x_ticks:
            ax.axvline(float(tick), color="0.86", linewidth=0.75, zorder=0)
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    else:
        ax.set_xlabel("k-point index")
    if model_ref or heff_ref:
        ref_label = "top" if align in {"top", "top_band", "top-band"} else "bottom"
        ax.set_ylabel(relative_energy_ylabel(ref_label))
    else:
        ax.set_ylabel("Energy (eV)")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", color="0.88", linewidth=0.65)
    ax.axhline(0.0, color="0.88", linewidth=0.7, zorder=0)
    if model_sel.size and heff_sel.size:
        plotted_windows = [model_sel, heff_sel]
        if current_support_sel is not None:
            plotted_windows.append(current_support_sel)
        ymin = float(min(np.min(values) for values in plotted_windows))
        ymax = float(max(np.max(values) for values in plotted_windows))
        pad = max(0.004, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
        ylim_cfg = plot_options.get("ylim")
        if ylim_cfg is not None:
            if not isinstance(ylim_cfg, Sequence) or isinstance(ylim_cfg, (str, bytes)) or len(ylim_cfg) != 2:
                raise ValueError(f"bands.plot.ylim must be [ymin, ymax], got {ylim_cfg!r}")
            values = [float(ylim_cfg[0]), float(ylim_cfg[1])]
            if len(values) != 2:
                raise ValueError(f"bands.plot.ylim must be [ymin, ymax], got {ylim_cfg!r}")
            ax.set_ylim(float(values[0]), float(values[1]))
        else:
            ax.set_ylim(ymin - pad, ymax + pad)
    show_metrics = bool(plot_options.get("show_metrics", plot_options.get("annotate_metrics", True)))
    if show_metrics and align in {"top", "top_band", "top-band", "bottom", "bottom_band", "bottom-band"}:
        delta = model_sel - heff_sel
        rms_mev = 1000.0 * float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
        shift_mev = 1000.0 * (model_ref - heff_ref)
        band_label = "top" if align in {"top", "top_band", "top-band"} else "bottom"
        ax.text(
            0.06,
            0.08,
            f"{band_label} {stop - start} bands\naligned RMS {rms_mev:.2f} meV\nshift {shift_mev:.2f} meV",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
    legend_handles = [
        Line2D([0], [0], label=str(reference_label), **KP_REFERENCE_STYLE),
    ]
    if current_support_plot is not None:
        legend_handles.append(
            Line2D([0], [0], label="Current Heff support mask", **current_support_style)
        )
    legend_handles.append(Line2D([0], [0], label="KP model", **KP_MODEL_STYLE))
    if bool(plot_options.get("legend_outside", False)):
        ax.legend(
            handles=legend_handles,
            loc=str(plot_options.get("legend_loc", "upper center")),
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=KP_LEGEND_KWARGS["frameon"],
            fontsize=KP_LEGEND_KWARGS["fontsize"],
        )
    else:
        ax.legend(
            handles=legend_handles,
            loc=str(plot_options.get("legend_loc", KP_LEGEND_KWARGS["loc"])),
            frameon=KP_LEGEND_KWARGS["frameon"],
            fontsize=KP_LEGEND_KWARGS["fontsize"],
        )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    box_aspect = plot_options.get("box_aspect", KP_BAND_BOX_ASPECT)
    if box_aspect in {None, False, "none", "None"}:
        resolved_box_aspect = None
    else:
        resolved_box_aspect = float(box_aspect)
    font_family = str(plot_options.get("font_family") or kp_font_family())
    apply_kp_axis_style(
        ax,
        box_aspect=resolved_box_aspect,
        font_family=font_family,
    )
    fig.tight_layout()
    if overlap_scalar is not None:
        colorbar_ax = inset_axes(
            ax,
            width=str(plot_options.get("overlap_colorbar_size", "4%")),
            height="100%",
            loc="lower left",
            bbox_to_anchor=(float(plot_options.get("overlap_colorbar_x", 1.07)), 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        vmin = float(overlap_scalar.norm.vmin)
        vmax = float(overlap_scalar.norm.vmax)
        gradient = np.linspace(vmin, vmax, 256, dtype=float).reshape(-1, 1)
        colorbar_ax.imshow(
            gradient,
            aspect="auto",
            origin="lower",
            cmap=overlap_scalar.cmap,
            norm=overlap_scalar.norm,
            extent=(0.0, 1.0, vmin, vmax),
        )
        colorbar_ax.set_xticks([])
        colorbar_ax.set_yticks(np.linspace(vmin, vmax, 6))
        colorbar_ax.yaxis.tick_right()
        colorbar_ax.yaxis.set_label_position("right")
        colorbar_ax.set_ylabel(overlap_label)
        apply_kp_axis_style(colorbar_ax, box_aspect=None, font_family=font_family)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out



def _first_bz_vertices(bM1: np.ndarray, bM2: np.ndarray) -> np.ndarray:
    from scipy.spatial import Voronoi

    points = []
    origin_index = 0
    for m in range(-2, 3):
        for n in range(-2, 3):
            if m == 0 and n == 0:
                origin_index = len(points)
            points.append(m * np.asarray(bM1, dtype=float) + n * np.asarray(bM2, dtype=float))
    points_arr = np.asarray(points, dtype=float)
    vor = Voronoi(points_arr)
    region_index = vor.point_region[origin_index]
    region = vor.regions[region_index]
    if not region or -1 in region:
        raise ValueError("Failed to construct a bounded first Brillouin-zone polygon")
    vertices = np.asarray(vor.vertices[region], dtype=float)
    center = np.mean(vertices, axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]



def save_harmonics_diagnostic_plot(
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    diagnostics: Mapping[str, Any],
    path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=KP_DPI)
    ax.scatter(Q_set1[:, 0], Q_set1[:, 1], s=28, marker="o", color="#1f77b4", alpha=0.75, label="layer 1 Q")
    ax.scatter(Q_set2[:, 0], Q_set2[:, 1], s=34, marker="^", color="#ff7f0e", alpha=0.75, label="layer 2 Q")

    bz = _first_bz_vertices(bM1, bM2)
    ax.add_patch(Polygon(bz, closed=True, fill=False, edgecolor="0.25", linewidth=1.2, linestyle="-", label="1st BZ"))

    selected_vectors: list[tuple[str, int, np.ndarray]] = []
    for kind in ("intra", "inter"):
        for row in diagnostics.get(kind, {}).get("selected", []):
            vector = np.asarray(row["vector"], dtype=float)
            selected_vectors.append((kind, int(row["index"]), vector))

    radii = sorted({round(float(np.linalg.norm(vector)), 12) for _kind, _index, vector in selected_vectors if np.linalg.norm(vector) > 1.0e-10})
    for radius in radii:
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor="0.75", linewidth=0.8, linestyle="--"))

    colors = {"intra": "#6a3d9a", "inter": "#d62728"}
    linestyles = {"intra": "-", "inter": "--"}
    for kind, index, vector in selected_vectors:
        if np.linalg.norm(vector) <= 1.0e-10:
            ax.scatter([0.0], [0.0], s=48, marker="x", color=colors[kind], linewidths=1.2)
            ax.text(0.0, 0.0, f" {kind[0]}{index}", color=colors[kind], fontsize=8, va="bottom")
            continue
        ax.annotate(
            "",
            xy=vector,
            xytext=(0.0, 0.0),
            arrowprops={
                "arrowstyle": "->",
                "color": colors[kind],
                "linewidth": 1.4,
                "linestyle": linestyles[kind],
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        ax.text(vector[0], vector[1], f" {kind[0]}{index}", color=colors[kind], fontsize=8, va="center")

    all_points = [np.asarray(Q_set1), np.asarray(Q_set2), np.zeros((1, 2)), np.asarray([bM1, bM2])]
    if bz.size:
        all_points.append(bz)
    if selected_vectors:
        all_points.append(np.asarray([vector for _kind, _index, vector in selected_vectors], dtype=float))
    stacked = np.vstack(all_points)
    span = float(max(np.ptp(stacked[:, 0]), np.ptp(stacked[:, 1]), np.linalg.norm(bM1), np.linalg.norm(bM2), 1.0))
    center = np.mean(stacked, axis=0)
    pad = 0.12 * span
    ax.set_xlim(center[0] - 0.55 * span - pad, center[0] + 0.55 * span + pad)
    ax.set_ylim(center[1] - 0.55 * span - pad, center[1] + 0.55 * span + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="0.88", linewidth=0.7)
    ax.axvline(0.0, color="0.88", linewidth=0.7)
    ax.set_xlabel("Qx")
    ax.set_ylabel("Qy")
    ax.set_title("Auto-selected harmonic vectors")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color=colors["intra"], lw=1.5, linestyle=linestyles["intra"]),
            Line2D([0], [0], color=colors["inter"], lw=1.5, linestyle=linestyles["inter"]),
        ]
    )
    labels.extend(["intra selected", "inter selected"])
    ax.legend(handles, labels, **KP_LEGEND_KWARGS)
    apply_kp_axis_style(ax, box_aspect=None, font_family=kp_font_family())
    fig.tight_layout()
    fig.savefig(out, dpi=KP_DPI)
    plt.close(fig)
    return out



def save_harmonics_diagnostics(
    *,
    moire_config: MoireConfig,
    model_config: configuration.ConfiguredModel,
    output_dir: Path,
) -> None:
    diagnostics = model_config.harmonics_diagnostics
    if not diagnostics:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "harmonics_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    save_harmonics_diagnostic_plot(
        Q_set1=np.asarray(moire_config.Q_set1, dtype=float),
        Q_set2=np.asarray(moire_config.Q_set2, dtype=float),
        bM1=np.asarray(moire_config.bM1, dtype=float),
        bM2=np.asarray(moire_config.bM2, dtype=float),
        diagnostics=diagnostics,
        path=output_dir / "harmonics_diagnostic.pdf",
    )



_Q_LATTICE_PALETTE = {
    "red": "#d1495b",
    "blue": "#3b6fb6",
    "gold": "#d99000",
    "green": "#2a9d64",
    "edge": "#737b86",
    "frame": "#737b86",
    "dark": "#20242a",
    "paper": "white",
}



def _nearest_point(points: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(points, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if arr.size == 0:
        return tgt, float("inf")
    distances = np.linalg.norm(arr - tgt, axis=1)
    idx = int(np.argmin(distances))
    return arr[idx], float(distances[idx])



def _harmonic_vectors_from_map(mapping: Mapping[int, np.ndarray]) -> list[tuple[int, np.ndarray]]:
    return [(int(index), np.asarray(vector, dtype=float)) for index, vector in sorted(mapping.items(), key=lambda item: int(item[0]))]



def _q_lattice_match_tolerance(qset1: np.ndarray, qset2: np.ndarray, bM1: np.ndarray, bM2: np.ndarray) -> float:
    stacked = np.vstack([np.asarray(qset1, dtype=float), np.asarray(qset2, dtype=float)])
    span = max(float(np.ptp(stacked[:, 0])), float(np.ptp(stacked[:, 1])), float(np.linalg.norm(bM1)), float(np.linalg.norm(bM2)), 1.0)
    return max(1.0e-7, 1.0e-5 * span)



def _choose_q_lattice_anchor(
    qset1: np.ndarray,
    qset2: np.ndarray,
    intra_vectors: Sequence[tuple[int, np.ndarray]],
    inter_vectors: Sequence[tuple[int, np.ndarray]],
    *,
    tol: float,
) -> np.ndarray:
    q1 = np.asarray(qset1, dtype=float)
    q2 = np.asarray(qset2, dtype=float)
    best_score: tuple[int, float, float] | None = None
    best = q1[0]
    for point in q1:
        miss_count = 0
        residual = 0.0
        for _index, vector in intra_vectors:
            _nearest, distance = _nearest_point(q1, point - vector)
            residual += distance
            miss_count += int(distance > tol)
        for _index, vector in inter_vectors:
            _nearest, distance = _nearest_point(q2, point + vector)
            residual += distance
            miss_count += int(distance > tol)
        score = (miss_count, round(residual, 12), round(float(np.linalg.norm(point)), 12))
        if best_score is None or score < best_score:
            best_score = score
            best = point
    return np.asarray(best, dtype=float)



def _draw_q_lattice_basis_arrows(ax: Any, bM1: np.ndarray, bM2: np.ndarray) -> None:
    origin = np.zeros(2, dtype=float)
    for vector, label, offset in (
        (np.asarray(bM1, dtype=float), r"$\mathbf{b}_{M1}$", np.array([0.04, -0.08])),
        (np.asarray(bM2, dtype=float), r"$\mathbf{b}_{M2}$", np.array([0.04, 0.06])),
    ):
        ax.annotate(
            "",
            xy=vector,
            xytext=origin,
            arrowprops=dict(arrowstyle="-|>", lw=1.15, color=_Q_LATTICE_PALETTE["dark"], mutation_scale=8.5),
            zorder=6,
        )
        text_xy = vector + offset * max(float(np.linalg.norm(vector)), 1.0)
        ax.text(text_xy[0], text_xy[1], label, fontsize=8.4, color=_Q_LATTICE_PALETTE["dark"], zorder=7)



def _draw_q_lattice_segments(ax: Any, points: np.ndarray, *, alpha: float = 1.0, linewidth: float = 1.05) -> None:
    from matplotlib.collections import LineCollection

    arr = np.asarray(points, dtype=float)
    distances = [
        float(np.linalg.norm(arr[i] - arr[j]))
        for i in range(len(arr))
        for j in range(i + 1, len(arr))
        if np.linalg.norm(arr[i] - arr[j]) > 1.0e-10
    ]
    if not distances:
        return
    nearest = min(distances)
    threshold = 1.04 * nearest
    segments = [(arr[i], arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr)) if np.linalg.norm(arr[i] - arr[j]) <= threshold]
    ax.add_collection(LineCollection(segments, colors=_Q_LATTICE_PALETTE["edge"], linewidths=linewidth, alpha=alpha, zorder=1))



def _q_lattice_hex_shell_from_radius(points: np.ndarray, bM1: np.ndarray, bM2: np.ndarray) -> int:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return 0
    b1 = np.asarray(bM1, dtype=float)
    b2 = np.asarray(bM2, dtype=float)
    spacing = max(float(np.linalg.norm(b1)), float(np.linalg.norm(b2)), 1.0e-12)
    radius = float(np.max(np.linalg.norm(arr, axis=1)))
    outer_vertex_allowance = 1.0 / math.sqrt(3.0)
    return max(0, int(math.ceil(max(0.0, radius / spacing - outer_vertex_allowance) - 1.0e-10)))



def _q_lattice_hex_centers(bM1: np.ndarray, bM2: np.ndarray, shell: int) -> np.ndarray:
    b1 = np.asarray(bM1, dtype=float)
    b2 = np.asarray(bM2, dtype=float)
    centers = []
    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            if max(abs(i), abs(j), abs(i + j)) <= shell:
                centers.append(i * b1 + j * b2)
    return np.asarray(centers, dtype=float)



def _draw_q_lattice_hex_cells(ax: Any, *, bM1: np.ndarray, bM2: np.ndarray, points: np.ndarray, linewidth: float = 1.0) -> np.ndarray:
    from matplotlib.collections import LineCollection

    shell = _q_lattice_hex_shell_from_radius(points, bM1, bM2)
    centers = _q_lattice_hex_centers(bM1, bM2, shell)
    base_hex = _first_bz_vertices(np.asarray(bM1, dtype=float), np.asarray(bM2, dtype=float))
    if centers.size == 0 or base_hex.size == 0:
        return base_hex
    segments = []
    vertices = []
    for center in centers:
        poly = base_hex + center
        vertices.append(poly)
        for idx in range(len(poly)):
            segments.append((poly[idx], poly[(idx + 1) % len(poly)]))
    ax.add_collection(
        LineCollection(
            segments,
            colors=_Q_LATTICE_PALETTE["edge"],
            linewidths=linewidth,
            alpha=0.92,
            zorder=1,
        )
    )
    return np.vstack(vertices)



def _q_lattice_nearest_spacing(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=float)
    distances = [
        float(np.linalg.norm(arr[i] - arr[j]))
        for i in range(len(arr))
        for j in range(i + 1, len(arr))
        if np.linalg.norm(arr[i] - arr[j]) > 1.0e-10
    ]
    return min(distances) if distances else 1.0



def _draw_q_lattice_harmonic_arrow(
    ax: Any,
    start: np.ndarray,
    stop: np.ndarray,
    *,
    color: str,
    spacing: float,
    curvature: float = 0.045,
) -> None:
    from matplotlib.patches import FancyArrow

    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(stop, dtype=float)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        return

    unit = direction / length
    normal = np.array([-unit[1], unit[0]], dtype=float)
    head_length = min(0.30 * spacing, 0.34 * length)
    head_base = p1 - head_length * unit
    shaft_end = p1 - 0.62 * head_length * unit
    control = 0.5 * (p0 + shaft_end) + curvature * length * normal
    t = np.linspace(0.0, 1.0, 40)[:, None]
    curve = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * control + t**2 * shaft_end
    ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.35, solid_capstyle="round", zorder=6)
    arrow_head = FancyArrow(
        head_base[0],
        head_base[1],
        p1[0] - head_base[0],
        p1[1] - head_base[1],
        width=0.001 * spacing,
        head_width=0.28 * spacing,
        head_length=head_length,
        length_includes_head=True,
        overhang=0.30,
        color=color,
        linewidth=0.0,
        zorder=7,
    )
    ax.add_patch(arrow_head)



def _q_lattice_inter_arrow_endpoints(
    *,
    qset1: np.ndarray,
    qset2: np.ndarray,
    sectors: Sequence[Mapping[str, Any]] | None,
    inter_sector_pairs: Sequence[Sequence[str]] | None,
    vector: np.ndarray,
    fallback_anchor: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sectors and inter_sector_pairs:
        by_name = {str(sector.get("name")): sector for sector in sectors}
        for pair in inter_sector_pairs:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                continue
            sector_from = by_name.get(str(pair[0]))
            sector_to = by_name.get(str(pair[1]))
            if sector_from is None or sector_to is None:
                continue
            q_from = sector_qset(sector_from, qset1, qset2)
            q_to = sector_qset(sector_to, qset1, qset2)
            for point in q_from:
                target, distance = _nearest_point(q_to, np.asarray(point, dtype=float) - np.asarray(vector, dtype=float))
                if distance <= tol:
                    return np.asarray(point, dtype=float), np.asarray(target, dtype=float)
    return np.asarray(fallback_anchor, dtype=float), np.asarray(fallback_anchor, dtype=float) + np.asarray(vector, dtype=float)



def save_q_lattice_harmonics_plot(
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
    intra_harmonics: Mapping[int, np.ndarray],
    inter_harmonics: Mapping[int, np.ndarray],
    path: str | Path,
    title: str | None = None,
    sectors: Sequence[Mapping[str, Any]] | None = None,
    inter_sector_pairs: Sequence[Sequence[str]] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    q1 = np.asarray(Q_set1, dtype=float)
    q2 = np.asarray(Q_set2, dtype=float)
    b1 = np.asarray(bM1, dtype=float)
    b2 = np.asarray(bM2, dtype=float)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    intra_vectors = _harmonic_vectors_from_map(intra_harmonics)
    inter_vectors = _harmonic_vectors_from_map(inter_harmonics)
    tol = _q_lattice_match_tolerance(q1, q2, b1, b2)
    anchor = _choose_q_lattice_anchor(q1, q2, intra_vectors, inter_vectors, tol=tol)
    spacing = _q_lattice_nearest_spacing(np.vstack([q1, q2]))

    with kp_plot_rc_context():
        fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=KP_DPI, constrained_layout=True)
        hex_vertices = _draw_q_lattice_hex_cells(ax, bM1=b1, bM2=b2, points=np.vstack([q1, q2]), linewidth=1.0)
        ax.scatter(q1[:, 0], q1[:, 1], s=40, color=_Q_LATTICE_PALETTE["red"], edgecolor="none", linewidth=0.0, zorder=3)
        ax.scatter(q2[:, 0], q2[:, 1], s=40, color=_Q_LATTICE_PALETTE["blue"], edgecolor="none", linewidth=0.0, zorder=3)

        arrow_targets: list[np.ndarray] = [anchor]
        styles = {
            "intra": {"color": _Q_LATTICE_PALETTE["gold"], "label": "intra"},
            "inter": {"color": _Q_LATTICE_PALETTE["green"], "label": "inter"},
        }
        for _index, vector in intra_vectors:
            ideal_target = anchor - vector
            target, distance = _nearest_point(q1, ideal_target)
            if distance > tol:
                target = ideal_target
            arrow_targets.append(np.asarray(target, dtype=float))
            _draw_q_lattice_harmonic_arrow(
                ax,
                anchor,
                target,
                color=styles["intra"]["color"],
                spacing=spacing,
            )
        for _index, vector in inter_vectors:
            start, target = _q_lattice_inter_arrow_endpoints(
                qset1=q1,
                qset2=q2,
                sectors=sectors,
                inter_sector_pairs=inter_sector_pairs,
                vector=vector,
                fallback_anchor=anchor,
                tol=tol,
            )
            arrow_targets.extend([np.asarray(start, dtype=float), np.asarray(target, dtype=float)])
            _draw_q_lattice_harmonic_arrow(
                ax,
                start,
                target,
                color=styles["inter"]["color"],
                spacing=spacing,
            )

        all_points = [q1, q2, np.asarray([anchor]), np.asarray(arrow_targets, dtype=float)]
        if hex_vertices.size:
            all_points.append(hex_vertices)
        stacked = np.vstack(all_points)
        span = max(float(np.ptp(stacked[:, 0])), float(np.ptp(stacked[:, 1])), float(np.linalg.norm(b1)), float(np.linalg.norm(b2)), 1.0)
        center = np.mean(stacked, axis=0)
        half = 0.58 * span
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if title:
            ax.set_title(title, fontsize=13, pad=7, weight="semibold")
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor=_Q_LATTICE_PALETTE["red"], markeredgecolor="none", markersize=6, label=r"$Q_1$"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=_Q_LATTICE_PALETTE["blue"], markeredgecolor="none", markersize=6, label=r"$Q_2$"),
                Line2D([0], [0], color=_Q_LATTICE_PALETTE["gold"], lw=1.45, linestyle="-", label="intra"),
                Line2D([0], [0], color=_Q_LATTICE_PALETTE["green"], lw=1.45, linestyle="-", label="inter"),
            ],
            **KP_LEGEND_KWARGS,
        )
        apply_kp_axis_style(ax, box_aspect=None, font_family=kp_font_family())
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out



def save_bM_diagnostics(*, model_config: configuration.ConfiguredModel, output_dir: Path) -> None:
    if not model_config.bM_diagnostics:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bM_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(model_config.bM_diagnostics, handle, indent=2)
