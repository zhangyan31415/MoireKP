from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Sequence

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io.tapw_loader import load_hamk, load_Q_sets
from .blocks import get_H_block, project_heff_full#, get_h_dft_low
hartree = 27.2113845
hartree = 1

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
) -> int:
    """Infer number of orbitals per layer per Q for a single spin channel."""
    N = hamk2d.shape[0]
    if spin == "all":
        base = N // 2
    else:
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
    target: str = "valence",
    ref_q_index: int = 0,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    index_order: Sequence[int] | None = None,
):
    """Plot all bands as lines across Q (2nd dim is band index).

    If index_order is provided, reorder Q indices before plotting.
    """
    # Stack into (num_Q, num_bands)
    E = np.array(eigs_list)
    if index_order is not None:
        idx = np.asarray(index_order, dtype=int)
        E = E[idx]
    x = np.arange(E.shape[0])
    fig, ax = plt.subplots(figsize=(5, 9))
    shift = float(efermi) if (efermi is not None) else 0.0
    for i in range(E.shape[1]):
        ax.plot(x, E[:, i] - shift, lw=0.6, alpha=0.9, marker='.')

    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Q index")
    ax.set_ylabel("Energy - E_F (eV)" if (efermi is not None) else "Energy (eV)")
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


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

    eigs_list: list[np.ndarray]
    target = str(plot_cfg.get("target", "valence"))
    print(f"[kp] Target: {target}")
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

        hamk = load_hamk(hamk_file, mmap_mode="r")*hartree
        q1, q2 = load_Q_sets(qset1_file, qset2_file)
        print(f"[kp] hamk shape={hamk.shape}, dtype={hamk.dtype}")
        print(f"[kp] Q1 shape={q1.shape}, Q2 shape={q2.shape}")

        # hamk index selection (if 3D)
        hamk_index = int(plot_cfg.get("hamk_index", 0))
        if hamk.ndim == 3:
            hamk2d = hamk[hamk_index]
        elif hamk.ndim == 2:
            hamk2d = hamk
        else:
            raise ValueError(f"Unexpected hamk ndim: {hamk.ndim}")

        spin = material.get("spin", "all")
        num_layers = int(material.get("num_layers", 2))
        print(f"[kp] spin={spin}, num_layers={num_layers}")

        # determine orbitals per layer per Q (single spin)
        if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
            orb0 = int(material["num_orb_per_layer"][0])
        else:
            orb0 = infer_orbitals_per_layer(hamk2d, len(q1), num_layers, spin=spin)
        print(f"[kp] inferred per-layer orbitals per Q (single spin) = {orb0}")

        Qlayer_list = [[q1], [q2]]
        num_layer_list = [1, 1]
        num_orb_per_layer_list = [[orb0], [orb0]]
        block_n = (sum(num_layer_list) * orb0) * (2 if spin == "all" else 1)
        print(f"[kp] block dimension per Q (with spin) = {block_n}")

        # Not used in this branch of get_H_block but kept for API compatibility
        # nlow_state_list = [[], []]
        # norb_fix_list = [[], []]
        
        nlow_state_list = plot_cfg.get("nlow_state_list", [])
        norb_fix_list = plot_cfg.get("norb_fix_list", [])
        print(nlow_state_list)
        print(norb_fix_list)

        H_eig, H_vec, H_blocks, U_new = get_H_block(
            hamk2d,
            Qlayer_list,
            num_layer_list,
            num_orb_per_layer_list,
            nlow_state_list,
            norb_fix_list,
            spin=spin,
            mode="gamma",
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

    out_path = resolve(plot_cfg.get("out", "plot_gamma_scatter.png"))
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
    q_rotation_deg = float(plot_cfg.get("q_rotation_deg", 210.0))
    if sort_by_qnorm:
        q_base = q1
        center = q_base.mean(axis=0)
        ca, sa = np.cos(np.deg2rad(q_rotation_deg)), np.sin(np.deg2rad(q_rotation_deg))
        R = np.array([[ca, -sa], [sa, ca]])
        Q_set1 = np.array([(R @ (center - q_base[i])[:2]) for i in range(len(q_base))])
        index_order = np.argsort(np.linalg.norm(Q_set1, axis=1))
        print(f"[kp] Using Q-norm sort with rotation {q_rotation_deg} deg.")

    # H_eig is an array of object vectors; normalize to list
    # Save spectrum to text
    save_spectrum_txt(eigs_list, data_out, index_order=index_order)
    plot_eigs_scatter(
        eigs_list,
        efermi,
        out=out_path,
        target=target,
        ref_q_index=ref_q_index,
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
        else:
            # 轨道标签准备（“层内轨道名表”，长度对齐到 orb0）
            labels_pattern = material.get("orbital_order") if isinstance(material, dict) else None
            per_layer_labels = expand_orbital_order_pattern(labels_pattern)
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

            def map_index(idx: int):
                """将平面波/原子轨道基底索引 -> (spin_tag, layer, orb_local, label)"""
                if spin == 'all':
                    spin_tag = 'up' if idx < half else 'down'
                    idx0 = idx if idx < half else idx - half
                else:
                    spin_tag = spin
                    idx0 = idx
                layer = idx0 // orb0
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

                print(f"[kp] Band {b}: top-{top_n} orbital components ({ref_q} Q)")
                # 列定义：rank idx spin layer orb coeff weight frac label
                print("        rank   idx   spin  layer  orb        coeff (complex)       frac     label")
                for rk, t in enumerate(top_idx.tolist(), start=1):
                    coeff = complex(vec[int(t)])
                    wt = float(w[int(t)])
                    fr = float(frac[rk-1])
                    spin_tag, layer, orb_local, label = map_index(int(t))
                    # 与你示例的对齐风格一致（数值宽度匹配），coeff 另外增加一列
                    print(f"        {rk:>4}  {int(t):>5}  {spin_tag:<5}  {layer:>5}  {orb_local:>3}  {_format_complex(coeff)}   {fr*100:>7.2f}%   {label}")

                # 如需一次性输出该带“所有基底分量”的复系数（可能很长），可打开以下注释：
                # print(f"[kp] Band {b}: full coefficient vector at ref_Q={ref_q} (index, coeff)")
                # for idx_full, c_full in enumerate(vec.tolist()):
                #     print(f"        {idx_full:>5}  {_format_complex(complex(c_full))}")

    except Exception as ex:
        print(f"[kp] Analysis step warning: {ex}")




    print("[kp] DONE")


def cmd_project_from_config(cfg_path: str) -> None:
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
    project_cfg = cfg.get("project", {})

    def resolve(p: str | None) -> str | None:
        if p is None:
            return None
        return p if os.path.isabs(p) else os.path.join(cfg_dir, p)

    hamk_file = resolve(material["hamk_file"])
    qset1_file = resolve(material["qset1_file"])
    qset2_file = resolve(material["qset2_file"])
    spin = material.get("spin", "all")
    num_layers = int(material.get("num_layers", 2))

    print(f"[kp] Using Hamiltonian: {hamk_file}")
    hamk = load_hamk(hamk_file, mmap_mode="r")*hartree
    q1, q2 = load_Q_sets(qset1_file, qset2_file)

    hamk_index = int(plot_cfg.get("hamk_index", 0))
    if hamk.ndim == 3:
        hamk2d = hamk[hamk_index]
    elif hamk.ndim == 2:
        hamk2d = hamk
    else:
        raise ValueError(f"Unexpected hamk ndim: {hamk.ndim}")

    # Determine orbitals per layer per Q (single spin)
    if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
        orb0 = int(material["num_orb_per_layer"][0])
    else:
        orb0 = infer_orbitals_per_layer(hamk2d, len(q1), num_layers, spin=spin)

    # For projection, we do not reorder Q to avoid changing Heff basis.
    q_count = len(q1)

    # Projection parameters
    nlow_state_list = project_cfg.get("nlow_state_list", [])
    norb_fix_list = project_cfg.get("norb_fix_list", [])
    workers = int(project_cfg.get("workers", 4))
    # Unified output directory for all artifacts
    out_dir = resolve(project_cfg.get("out_dir", "plots"))
    if out_dir is None:
        out_dir = os.path.join(cfg_dir, "plots")
    out_heff = os.path.join(out_dir, "heff_list.npy")
    out_eig = os.path.join(out_dir, "heff_eig.npy")
    out_vec = os.path.join(out_dir, "heff_vec.npy")
    plot_out = os.path.join(out_dir, "heff_scatter.png")
    data_out = os.path.join(out_dir, "heff_spectrum.txt")

    # Run projection for each k (first dim of hamk)
    print(f"[kp] Projecting Heff across k (first-order U^†HU): workers={workers}")
    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    nk = hamk3d.shape[0]
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
            )
            for i in tqdm(range(nk))
        )
    except Exception:
        results = [
            project_heff_full(
                np.asarray(hamk3d[i], dtype=np.complex128),
                q_count,
                orb0,
                [1, 1],
                spin=spin,
                bands=nlow_state_list,
                comps=norb_fix_list,
            )
            for i in range(nk)
        ]

    heff_list = [np.asarray(r[0], dtype=np.complex128) for r in results]  # (nk, M, M)
    heig_list = [np.asarray(r[1], dtype=np.float64) for r in results]      # (nk, M)
    hvec_list = [np.asarray(r[2], dtype=np.complex128) for r in results]  # (nk, M, M)

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
    print(f"[kp] Saved Heff to: {out_heff}")
    print(f"[kp] Saved Heff eig to: {out_eig}")
    print(f"[kp] Saved Heff vec to: {out_vec}")

    # Plot projected bands
    # Shift: subtract only if project.efermi is provided; otherwise no shift
    proj_ef = project_cfg.get("efermi")
    efermi = float(proj_ef) if proj_ef is not None else None
    ylim_cfg = project_cfg.get("ylim")
    ylim = (float(ylim_cfg[0]), float(ylim_cfg[1])) if isinstance(ylim_cfg, (list, tuple)) and len(ylim_cfg) == 2 else None
    print(f"[kp] Plotting Heff bands to: {plot_out}")
    plot_eigs_scatter(heig_list, efermi, out=plot_out, target="all", ref_q_index=0, title=project_cfg.get("title"), ylim=ylim, index_order=None)
    save_spectrum_txt(heig_list, data_out)
    print(f"[kp] Saved Heff spectrum to: {data_out}")


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


def cmd_analyze_from_config(cfg_path: str, bands_below: list[int], bands_above: list[int]) -> None:
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

    hamk_file = resolve(material["hamk_file"])
    qset1_file = resolve(material["qset1_file"])
    qset2_file = resolve(material["qset2_file"])
    explicit_ef = material.get("efermi")

    spin = material.get("spin", "all")
    num_layers = int(material.get("num_layers", 2))

    hamk = load_hamk(hamk_file, mmap_mode="r")
    q1, q2 = load_Q_sets(qset1_file, qset2_file)

    hamk_index = int(plot_cfg.get("hamk_index", 0))
    if hamk.ndim == 3:
        hamk2d = hamk[hamk_index]
    elif hamk.ndim == 2:
        hamk2d = hamk
    else:
        raise ValueError(f"Unexpected hamk ndim: {hamk.ndim}")

    if "num_orb_per_layer" in material and material["num_orb_per_layer"]:
        orb0 = int(material["num_orb_per_layer"][0])
    else:
        orb0 = infer_orbitals_per_layer(hamk2d, len(q1), num_layers, spin=spin)

    Qlayer_list = [[q1], [q2]]
    num_layer_list = [1, 1]
    num_orb_per_layer_list = [[orb0], [orb0]]

    H_eig, H_vec, H_blocks, U_new = get_H_block(
        hamk2d,
        Qlayer_list,
        num_layer_list,
        num_orb_per_layer_list,
        [[], []],
        [[], []],
        spin=spin,
        mode="gamma",
    )

    eigs_list = [np.asarray(ev) for ev in H_eig]
    vecs_list = [np.asarray(v) for v in H_vec]

    # Q order same as plot
    sort_by_qnorm = bool(plot_cfg.get("sort_by_qnorm", True))
    q_rotation_deg = float(plot_cfg.get("q_rotation_deg", 210.0))
    if sort_by_qnorm:
        q_base = q1
        center = q_base.mean(axis=0)
        ca, sa = np.cos(np.deg2rad(q_rotation_deg)), np.sin(np.deg2rad(q_rotation_deg))
        R = np.array([[ca, -sa], [sa, ca]])
        Q_set1 = np.array([(R @ (center - q_base[i])[:2]) for i in range(len(q_base))])
        order = np.argsort(np.linalg.norm(Q_set1, axis=1))
        eigs_list = [eigs_list[i] for i in order]
        vecs_list = [vecs_list[i] for i in order]

    if explicit_ef is None:
        concat = np.sort(np.concatenate(eigs_list))
        efermi = float(np.median(concat[-max(10, len(concat)//10):]))
    else:
        efermi = float(explicit_ef)
    print(f"[kp] efermi = {efermi:.6f} eV")

    selected = list(bands_below) + list(bands_above)
    print(f"[kp] Bands below EF: {bands_below}")
    print(f"[kp] Bands above EF: {bands_above}")

    labels_pattern = material.get("orbital_order")
    per_layer_labels: list[str] | None = None
    if labels_pattern:
        per_layer_labels = expand_orbital_order_pattern(str(labels_pattern))
        if len(per_layer_labels) != orb0:
            print(f"[kp] Warning: labels length {len(per_layer_labels)} != orb_per_layer {orb0}")
            if len(per_layer_labels) < orb0:
                per_layer_labels += [f"orb_{i}" for i in range(len(per_layer_labels), orb0)]
            else:
                per_layer_labels = per_layer_labels[:orb0]

    block_dim = vecs_list[0].shape[0]
    half = block_dim // 2 if spin == 'all' else block_dim

    def map_index(idx: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if spin == 'all':
            info['spin'] = 'up' if idx < half else 'down'
            idx0 = idx if idx < half else idx - half
        else:
            info['spin'] = spin
            idx0 = idx
        layer = idx0 // orb0
        orb_local = idx0 % orb0
        info['layer'] = int(layer)
        info['orbital_index'] = int(orb_local)
        if per_layer_labels is not None:
            info['label'] = per_layer_labels[orb_local]
        return info

    for b in selected:
        if b < 0 or b >= vecs_list[0].shape[1]:
            print(f"[kp] Band {b}: out of range; skipping")
            continue
        w = np.zeros(block_dim, dtype=float)
        for V in vecs_list:
            vec = V[:, b]
            w += (np.abs(vec) ** 2)
        top_idx = np.argsort(-w)[:4]
        print(f"[kp] Band {b}: top-4 orbital indices = {top_idx.tolist()}")
        for i in top_idx:
            mi = map_index(int(i))
            lbl = mi.get('label', None)
            if lbl is None:
                print(f"    idx {int(i)} -> spin={mi['spin']}, layer={mi['layer']}, orb_local={mi['orbital_index']}")
            else:
                print(f"    idx {int(i)} -> spin={mi['spin']}, layer={mi['layer']}, orb_local={mi['orbital_index']}, label={lbl}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kp", description="kp CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plot = sub.add_parser("plot", help="Plot scatter of band vs Q from config")
    p_plot.add_argument("-c", "--config", required=True, help="YAML config path")

    p_proj = sub.add_parser("project", help="Project selected bands to Heff across Q and plot")
    p_proj.add_argument("-c", "--config", required=True, help="YAML config path")

    return p


def main(argv: Sequence[str] | None = None) -> None:
    p = build_argparser()
    args = p.parse_args(argv)
    if args.cmd == "plot":
        cmd_plot_from_config(args.config)
    elif args.cmd == "project":
        cmd_project_from_config(args.config)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
