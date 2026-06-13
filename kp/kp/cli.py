from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io.tapw_loader import load_hamk, load_Q_sets
from .blocks import get_H_block, project_heff_full#, get_h_dft_low
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

HARTREE_TO_EV = 27.2113845


def _energy_scale_from_material(material: dict[str, Any]) -> float:
    unit = material.get("energy_unit")
    if unit is None:
        raise ValueError("material.energy_unit is required and must be 'eV' or 'Hartree'")
    normalized = str(unit).strip().lower().replace("_", "").replace("-", "")
    if normalized in {"ev", "electronvolt", "electronvolts"}:
        return 1.0
    if normalized in {"hartree", "hartrees", "ha"}:
        return HARTREE_TO_EV
    raise ValueError(f"material.energy_unit must be 'eV' or 'Hartree', got {unit!r}")


def _load_hamk_with_energy_unit(path: str, material: dict[str, Any], *, mmap_mode: str | None = "r") -> np.ndarray:
    return load_hamk(path, mmap_mode=mmap_mode) * _energy_scale_from_material(material)

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
    return_fig: bool = False,
):
    """Plot all bands as lines across Q (2nd dim is band index).

    If index_order is provided, reorder Q indices before plotting.
    """
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
    x = np.arange(E.shape[0])
    fig, ax = plt.subplots(figsize=(5.6, 9.0))
    shift = float(efermi) if (efermi is not None) else 0.0
    if E0 is not None:
        for i in range(E0.shape[1]):
            ax.plot(
                x,
                E0[:, i] - shift,
                color="#9AA0A6",
                lw=0.7,
                alpha=0.45,
                zorder=1,
                label="Original" if i == 0 else "_nolegend_",
            )
    colors = list(plt.get_cmap("tab10").colors)
    for i in range(E.shape[1]):
        color = colors[i % len(colors)]
        ax.plot(
            x,
            E[:, i] - shift,
            lw=1.2,
            alpha=0.95,
            color=color,
            marker='o',
            markersize=3.3,
            markeredgewidth=0.0,
            zorder=2,
            label="Heff" if i == 0 else "_nolegend_",
        )

    ax.axhline(0.0, color="#555555", lw=0.8, ls="--", alpha=0.8, zorder=0)
    ax.set_xlabel("Q index")
    ax.set_ylabel("Energy - E_F (eV)" if (efermi is not None) else "Energy (eV)")
    ax.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.65)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if E0 is not None:
        ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=220)
    if return_fig:
        return fig, ax
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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


def _active_indices_from_project_cfg(project_cfg: dict[str, Any]) -> list[int]:
    active = project_cfg.get("active_indices")
    if active is not None:
        return parse_int_list(active)
    nlow_state_list = project_cfg.get("nlow_state_list", [])
    if nlow_state_list and isinstance(nlow_state_list[0], (list, tuple)):
        return [int(x) for x in nlow_state_list[0]]
    if nlow_state_list:
        return [int(x) for x in nlow_state_list]
    return []


def _normalize_nlow_state_list(project_cfg: dict[str, Any]) -> list[list[int]]:
    active = project_cfg.get("active_indices")
    if active is not None:
        return [parse_int_list(active)]
    nlow_state_list = project_cfg.get("nlow_state_list", [])
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
        cfg = yaml.safe_load(f)

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
    target = str(plot_cfg.get("target", "valence"))
    print(f"[kp] Target: {target}")

    # Initialize variables that may be needed later
    orb0 = None
    q1 = None
    q2 = None
    vecs_list = None

    if band_file and target.lower() != "all":
        # Prefer precomputed band energies (fast path)
        print(f"[kp] Using band file (fast path): {band_file}")
        rows = _load_bands_from_text(band_file)
        eigs_list = [np.array(sorted(r)) for r in rows]
        print(f"[kp] Loaded {len(eigs_list)} Q points from band file.")
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

        # determine orbitals per layer per Q (single spin)
        if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
            orb0 = int(material["num_orb_per_layer"][0])
        else:
            orb0 = infer_orbitals_per_layer(hamk2d, len(q1), num_layers, spin=spin,mode=mode)
        print(f"[kp] inferred per-layer orbitals per Q (single spin) = {orb0}")


        Qlayer_list = [[q1], [q2]]
        num_layer_list = [1, 1]
        num_orb_per_layer_list = [[orb0], [orb0]]
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
        if spin == "up":
            hamk2d = hamk2d[:hamk2d.shape[0]//2,:hamk2d.shape[0]//2]
        elif spin == "down":
            hamk2d = hamk2d[hamk2d.shape[0]//2:,hamk2d.shape[0]//2:]

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
            spin=spin,
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
            # non-gamma mode: eigs_list length = len(q1) + len(q2)
            # Structure: [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, layer1_Q1, ..., layer1_Qn]
            index_order1 = np.argsort(np.linalg.norm(Q_set1, axis=1))
            index_order2 = np.argsort(np.linalg.norm(Q_set2, axis=1))
            # For non-gamma mode, blocks are ordered as: all layer0 blocks, then all layer1 blocks
            index_order = np.concatenate([index_order1, len(q1) + index_order2])
        print(f"[kp] Using Q-norm sort with rotation {q_rotation_deg} deg. Mode: {mode}")

    # H_eig is an array of object vectors; normalize to list
    # Save spectrum to text
    save_spectrum_txt(eigs_list, data_out, index_order=index_order)
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
            per_layer_labels = expand_orbital_order_pattern(labels_pattern) if labels_pattern else None
            if per_layer_labels is not None and len(per_layer_labels) != orb0:
                if len(per_layer_labels) < orb0:
                    per_layer_labels = per_layer_labels + [f"orb_{i}" for i in range(len(per_layer_labels), orb0)]
                else:
                    per_layer_labels = per_layer_labels[:orb0]

            Vq = np.asarray(vecs_ord[ref_q])  # 形状 (block_dim, nBands)
            if Vq.ndim != 2:
                raise RuntimeError(f"Eigenvector matrix at ref_q has wrong shape: {Vq.shape}")

            block_dim, nBands = Vq.shape
            half = block_dim // 2 if spin == 'all' else block_dim

            def map_index(idx: int, block_layer: int | None = None):
                """将平面波/原子轨道基底索引 -> (spin_tag, layer, orb_local, label)

                Args:
                    idx: 轨道索引
                    block_layer: 对于非gamma mode，指定该块属于哪个layer (0或1)
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
                        # Infer from ref_q index: if ref_q < len(q1), it's layer 0, else layer 1
                        layer = 0 if ref_q < len(q1) else 1

                orb_local = idx0 % orb0
                label = None
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

                # Determine block layer for non-gamma mode
                block_layer = None
                if mode != "gamma" and q1 is not None:
                    # For non-gamma mode, ref_q indicates which (layer, Q) block
                    block_layer = 0 if ref_q < len(q1) else 1

                print(f"[kp] Band {b}: top-{top_n} orbital components ({ref_q} Q, mode={mode})")
                # 列定义：rank idx spin layer orb coeff weight frac label
                print("        rank   idx   spin  layer  orb        coeff (complex)       frac     label")
                for rk, t in enumerate(top_idx.tolist(), start=1):
                    coeff = complex(vec[int(t)])
                    wt = float(w[int(t)])
                    fr = float(frac[rk-1])
                    spin_tag, layer, orb_local, label = map_index(int(t), block_layer=block_layer)
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
        cfg = yaml.safe_load(f)

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
    num_layers = int(material.get("num_layers", 2))

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
    norb_fix_list = project_cfg.get("norb_fix_list", [])
    workers = int(project_cfg.get("workers", 4))
    mode = project_cfg.get("mode", "gamma").lower()
    method = _downfold_method(project_cfg)
    e_ref = _e_ref_from_project_cfg(project_cfg)
    pole_warning_mev = float(project_cfg.get("pole_warning_mev", 10.0))
    pole_danger_mev = float(project_cfg.get("pole_danger_mev", 1.0))
    fail_on_near_pole = _as_bool(project_cfg.get("fail_on_near_pole", False))
    verbose = _as_bool(project_cfg.get("verbose", False))
    print_k_diagnostics = _as_bool(project_cfg.get("print_k_diagnostics", False))
    # top_n_list = _top_n_list_from_project_cfg(project_cfg)
    if method in {"fixed_schur", "linearized_lowdin"} and e_ref is None:
        raise ValueError(f"project.downfold_method={method!r} requires project.e_ref")

    # Determine orbitals per layer per Q (single spin)
    if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
        orb0 = int(material["num_orb_per_layer"][0])
    else:
        orb0 = infer_orbitals_per_layer(hamk2d, len(q1), num_layers, spin=spin, mode=mode)

    # For projection, we do not reorder Q to avoid changing Heff basis.
    q_count = len(q1)
    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    nk = hamk3d.shape[0]
    if spin == "up":
        hamk3d_new = []
        for i in range(len(hamk3d)):
            hamk3d_new.append(hamk3d[i][:hamk3d[i].shape[0]//2,:hamk3d[i].shape[0]//2])
        hamk3d = np.array(hamk3d_new)
    elif spin == "down":
        hamk3d_new = []
        for i in range(len(hamk3d)):
            hamk3d_new.append(hamk3d[i][hamk3d[i].shape[0]//2:,hamk3d[i].shape[0]//2:])
        hamk3d = np.array(hamk3d_new)
    active_indices = _active_indices_from_project_cfg(project_cfg)
    model_dim = q_count * len(active_indices)
    print("[kp] Project setup:")
    print(f"[kp]   material={material.get('name', 'unknown')}, mode={mode}, spin={spin}")
    print(f"[kp]   k-points={nk}, Q-points={q_count}, orbitals/layer/Q={orb0}")
    print(f"[kp]   active bands={active_indices}, model dimension={model_dim}")
    method_line = f"[kp]   method={_downfold_method_label(method)}"
    if e_ref is not None:
        method_line += f", E_ref={e_ref:.6f} eV"
    print(method_line)
    print(f"[kp]   output directory={resolve(project_cfg.get('out_dir', 'plots'))}")
    if verbose:
        print(f"[kp]   hamk={hamk_file}")
        print(f"[kp]   qset1={qset1_file}")
        print(f"[kp]   qset2={qset2_file}")
        print(f"[kp]   hamk shape={hamk.shape}, reference block shape={hamk2d.shape}")
        print(f"[kp]   nlow_state_list={nlow_state_list}")
        print(f"[kp]   norb_fix_list={norb_fix_list}")
    # Unified output directory for all artifacts
    out_dir = resolve(project_cfg.get("out_dir", "plots"))
    if out_dir is None:
        out_dir = os.path.join(cfg_dir, "plots")
    out_heff = os.path.join(out_dir, "heff_list.npy")
    out_eig = os.path.join(out_dir, "heff_eig.npy")
    out_vec = os.path.join(out_dir, "heff_vec.npy")
    plot_out = os.path.join(out_dir, "heff_scatter.png")
    data_out = os.path.join(out_dir, "heff_spectrum.txt")
    original_eigs_list = None
    band_file = resolve(material.get("band_file"))
    if band_file:
        try:
            original_eigs_list = _load_bands_from_text(band_file)
        except Exception as ex:
            print(f"[kp] Overlay warning: failed to load original bands from {band_file}: {ex}")

    # Run projection for each k (first dim of hamk)
    print(f"[kp] Running projection: {nk} k-points, workers={workers}")

    try:
        from joblib import Parallel, delayed
        from tqdm import tqdm
        results = Parallel(n_jobs=workers)(
            delayed(project_heff_full)(
                np.asarray(hamk3d[i], dtype=np.complex128),
                q_count,
                orb0,
                [1, 1],
                spin=spin,
                bands=nlow_state_list,
                comps=norb_fix_list,
                Qlayer_list=[[q1], [q2]],
                num_orb_per_layer_list=[[orb0], [orb0]],
                nlow_state_list=nlow_state_list,
                norb_fix_list=norb_fix_list,
                downfold_method=method,
                E_ref=e_ref,
                pole_warning_mev=pole_warning_mev,
                pole_danger_mev=pole_danger_mev,
                fail_on_near_pole=fail_on_near_pole,
                return_diagnostics=True,
                mode=mode,
            )
            for i in tqdm(
                range(nk),
                desc="[kp] project",
                unit="k",
                ncols=80,
                leave=False,
                disable=not sys.stderr.isatty(),
            )
        )
    except Exception as ex:
        print(f"[kp] Parallel projection failed, falling back to serial: {ex}")
        results = [
            project_heff_full(
                np.asarray(hamk3d[i], dtype=np.complex128),
                q_count,
                orb0,
                [1, 1],
                spin=spin,
                bands=nlow_state_list,
                comps=norb_fix_list,
                Qlayer_list=[[q1], [q2]],
                num_orb_per_layer_list=[[orb0], [orb0]],
                nlow_state_list=nlow_state_list,
                norb_fix_list=norb_fix_list,
                downfold_method=method,
                E_ref=e_ref,
                pole_warning_mev=pole_warning_mev,
                pole_danger_mev=pole_danger_mev,
                fail_on_near_pole=fail_on_near_pole,
                return_diagnostics=True,
                mode=mode,
            )
            for i in range(nk)
        ]

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

    # Plot projected bands
    # Shift: subtract only if project.efermi is provided; otherwise no shift
    proj_ef = project_cfg.get("efermi")
    efermi = float(proj_ef) if proj_ef is not None else None
    ylim_cfg = project_cfg.get("ylim")
    ylim = (float(ylim_cfg[0]), float(ylim_cfg[1])) if isinstance(ylim_cfg, (list, tuple)) and len(ylim_cfg) == 2 else None
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
    )
    save_spectrum_txt(heig_list, data_out)
    print(f"[kp] Saved Heff spectrum: {data_out}")


def cmd_sweep_from_config(cfg_path: str, overrides: dict[str, Any] | None = None) -> None:
    """Sweep E_ref for fixed-energy Schur downfolding and report pole distances."""
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    print(f"[kp] Loading config: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

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
    num_layers = int(material.get("num_layers", 2))
    pole_warning_mev = float(project_cfg.get("pole_warning_mev", 10.0))
    pole_danger_mev = float(project_cfg.get("pole_danger_mev", 1.0))
    fail_on_near_pole = _as_bool(project_cfg.get("fail_on_near_pole", False))
    nlow_state_list = _normalize_nlow_state_list(project_cfg)
    norb_fix_list = project_cfg.get("norb_fix_list", [])

    print(f"[kp] Sweep downfold_method = fixed_schur")
    print(f"[kp] active indices = {_active_indices_from_project_cfg(project_cfg)}")
    print(f"[kp] E_ref values = {e_ref_values}")
    # print(f"[kp] top-N list = {top_n_list}")

    hamk = _load_hamk_with_energy_unit(hamk_file, material, mmap_mode="r")
    q1, q2 = load_Q_sets(qset1_file, qset2_file)
    hamk_index = int(plot_cfg.get("hamk_index", 0))
    hamk2d = hamk[hamk_index] if hamk.ndim == 3 else hamk
    if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
        orb0 = int(material["num_orb_per_layer"][0])
    else:
        orb0 = infer_orbitals_per_layer(hamk2d, len(q1), num_layers, spin=spin, mode=mode)
    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    if k_indices is None:
        k_indices = list(range(hamk3d.shape[0]))
    # original_rows_all = _load_bands_from_text(band_file)
    # original_rows = [original_rows_all[i] for i in k_indices]
    q_count = len(q1)

    # rows: list[SweepRow] = []
    for e_ref in e_ref_values:
        print(f"[kp] Sweeping E_ref={e_ref:.6f} eV over {len(k_indices)} k points")
        results = []
        if workers > 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=workers)(
                    delayed(project_heff_full)(
                        np.asarray(hamk3d[i], dtype=np.complex128),
                        q_count,
                        orb0,
                        [1, 1],
                        spin=spin,
                        Qlayer_list=[[q1], [q2]],
                        num_orb_per_layer_list=[[orb0], [orb0]],
                        nlow_state_list=nlow_state_list,
                        norb_fix_list=norb_fix_list,
                        downfold_method="fixed_schur",
                        E_ref=float(e_ref),
                        pole_warning_mev=pole_warning_mev,
                        pole_danger_mev=pole_danger_mev,
                        fail_on_near_pole=fail_on_near_pole,
                        return_diagnostics=True,
                        mode=mode,
                    )
                    for i in k_indices
                )
            except Exception as ex:
                print(f"[kp] Parallel sweep failed, falling back to serial: {ex}")
                results = []
        if not results:
            results = [
                project_heff_full(
                    np.asarray(hamk3d[i], dtype=np.complex128),
                    q_count,
                    orb0,
                    [1, 1],
                    spin=spin,
                    Qlayer_list=[[q1], [q2]],
                    num_orb_per_layer_list=[[orb0], [orb0]],
                    nlow_state_list=nlow_state_list,
                    norb_fix_list=norb_fix_list,
                    downfold_method="fixed_schur",
                    E_ref=float(e_ref),
                    pole_warning_mev=pole_warning_mev,
                    pole_danger_mev=pole_danger_mev,
                    fail_on_near_pole=fail_on_near_pole,
                    return_diagnostics=True,
                    mode=mode,
                )
                for i in k_indices
            ]
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
    """Expand a segment like 'I-s3p2d2' into a list of orbital labels for that species.
    Ordering: s, p(x,y,z), d(z2, x2-y2, xy, xz, yz), f(7 real harmonics).
    """
    segment = segment.strip()
    if not segment:
        return []
    if '-' not in segment:
        return [segment]
    species, patterns = segment.split('-', 1)
    s_list = ['s']
    p_list = ['px', 'py', 'pz']
    d_list = ['dz2', 'dx2-y2', 'dxy', 'dxz', 'dyz']
    f_list = ['f5z2', 'f5xz2', 'f5yz2', 'fzx2', 'fxyz', 'fx3', 'f3yx2']
    out: list[str] = []
    import re
    for m in re.finditer(r'([spdf])(\d+)', patterns):
        orb = m.group(1)
        rep = int(m.group(2))
        base = {'s': s_list, 'p': p_list, 'd': d_list, 'f': f_list}[orb]
        for r in range(1, rep + 1):
            for b in base:
                out.append(f"{species}_{b}_r{r}")
    return out


def expand_orbital_order_pattern(pattern: str) -> list[str]:
    """Expand 'I-s3p2d2,Mg-s2p2,I-s3p2d2' into a flat list of labels (length should match orb_per_layer)."""
    parts = [p for p in pattern.split(',') if p.strip()]
    labels: list[str] = []
    for seg in parts:
        labels.extend(_expand_orbital_pattern(seg))
    return labels


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kp", description="kp CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plot = sub.add_parser("plot", help="Plot scatter of band vs Q from config")
    p_plot.add_argument("-c", "--config", required=True, help="YAML config path")

    p_proj = sub.add_parser("project", help="Project selected bands to Heff across Q and plot")
    p_proj.add_argument("-c", "--config", required=True, help="YAML config path")
    p_proj.add_argument("--active-indices", help="Comma-separated active band indices, e.g. 46,47")
    p_proj.add_argument("--downfold-method", choices=["first_order", "fixed_schur", "linearized_lowdin"])
    p_proj.add_argument("--e-ref", type=float, help="Reference energy for fixed_schur/linearized_lowdin in eV")
    p_proj.add_argument("--top-n", help="Comma-separated top-N values for diagnostics, e.g. 2,4,6,10,20")
    p_proj.add_argument("--pole-warning-mev", type=float)
    p_proj.add_argument("--pole-danger-mev", type=float)
    p_proj.add_argument("--fail-on-near-pole", action="store_true")

    p_sweep = sub.add_parser("sweep", help="Sweep E_ref for fixed Schur downfolding")
    p_sweep.add_argument("-c", "--config", required=True, help="YAML config path")
    p_sweep.add_argument("--active-indices", help="Comma-separated active band indices, e.g. 46,47")
    p_sweep.add_argument("--e-ref-values", help="Comma-separated E_ref values in eV")
    p_sweep.add_argument("--top-n", help="Comma-separated top-N values for diagnostics")
    p_sweep.add_argument("--k-indices", help="Comma-separated k indices; default is all")
    p_sweep.add_argument("--pole-warning-mev", type=float)
    p_sweep.add_argument("--pole-danger-mev", type=float)
    p_sweep.add_argument("--fail-on-near-pole", action="store_true")

    p_symm = sub.add_parser("symm", help="Project TAPW symmetry representations into the KP basis")
    p_symm.add_argument("-c", "--config", required=True, help="YAML config path")
    p_symm.add_argument("--developer-outputs", action="store_true", help="Write developer-only projection matrices under diagnostics/")

    p_model = sub.add_parser("model", help="Build/fit a configured continuum model and compare to Heff")
    p_model.add_argument("-c", "--config", required=True, help="YAML model config path")

    return p


def main(argv: Sequence[str] | None = None) -> None:
    p = build_argparser()
    args = p.parse_args(argv)
    if args.cmd == "plot":
        cmd_plot_from_config(args.config)
    elif args.cmd == "project":
        overrides = {
            "active_indices": args.active_indices,
            "downfold_method": args.downfold_method,
            "e_ref": args.e_ref,
            "top_n": args.top_n,
            "pole_warning_mev": args.pole_warning_mev,
            "pole_danger_mev": args.pole_danger_mev,
            "fail_on_near_pole": args.fail_on_near_pole if args.fail_on_near_pole else None,
        }
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
        }
        cmd_sweep_from_config(args.config, overrides)
    elif args.cmd == "symm":
        run_symmetry_projection_from_config(
            args.config,
            developer_outputs=True if args.developer_outputs else None,
        )
    elif args.cmd == "model":
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
            print(f"[kp model]   harmonics: intra={len(moire_cfg.intra_harmonics_map)}, inter={len(moire_cfg.inter_harmonics_map)}")
            if sym_ops:
                print(f"[kp model]   symmetry: {', '.join(str(op) for op in sym_ops)}")
            if results.get("model_log"):
                print(f"[kp model]   detailed log: {results['model_log']}")
            if results.get("runtime_s") is not None:
                print(f"[kp model]   runtime: {float(results['runtime_s']):.2f} s")
        if results.get("band_plot"):
            print(f"[kp model]   band plot: {results['band_plot']}")
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
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
