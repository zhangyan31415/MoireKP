from __future__ import annotations

from typing import List, Tuple, Literal

import numpy as np
import scipy
import scipy.linalg


def align_eigenstates(U_low: np.ndarray, Phi_ref: np.ndarray) -> np.ndarray:
    """Align low-energy eigenstates to a reference basis using Procrustes via SVD.

    Parameters
    - U_low: (M, N) eigenvector matrix (columns are states to align)
    - Phi_ref: (M, N) reference vectors (columns)

    Returns
    - U_aligned: (M, N) aligned eigenvectors
    - V: (N, N) unitary rotation applied in the low-energy subspace
    """

    # Overlap O = Phi_ref^† U_low
    O = Phi_ref.conj().T @ U_low
    # SVD: O = X Σ Y^†
    X, s, Yh = np.linalg.svd(O, full_matrices=False)
    # Optimal unitary: V = Y X^†
    V = Yh.conj().T @ X.conj().T
    # Apply rotation in subspace
    U_aligned = U_low @ V
    return U_aligned, V


def get_H_block(
    Hamk_list: np.ndarray,
    Qlayer_list: List[np.ndarray],
    num_layer_list: List[int],
    num_orb_per_layer_list: List[List[int]],
    nlow_state_list: List[int],
    norb_fix_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    mode: Literal["gamma", "K1", "K2"] = "gamma",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble Hamiltonian sub-blocks and diagonalize per-Q selection.

    This ports the core indexing logic from the notebook's get_H_block
    to extract same-Q blocks across layers, handle spins, and compute
    eigenvalues/eigenvectors per block.

    Returns
    - H_GM_diag_eig: array of eigenvalues per block (list -> array)
    - H_GM_diag_eig_vec: array of eigenvectors per block (list -> array)
    - H_diag_block: list/array of extracted Hamiltonian blocks
    - U_new_block: placeholder for potential unitary transforms (empty)
    """

    # Inputs can be large dense arrays
    hamk = Hamk_list

    # Normalize list-like shapes
    num_layer_arr = np.array(num_layer_list)
    num_orb_per_layer = [np.array(x) for x in num_orb_per_layer_list]
    # Number of Q points per group
    num_q_list = [np.array([len(q_layer) for q_layer in q_group]) for q_group in Qlayer_list]

    H_diag_block: List[np.ndarray] = []
    H_GM_diag_eig: List[np.ndarray] = []
    H_GM_diag_eig_vec: List[np.ndarray] = []
    U_new_block: List[np.ndarray] = []

    # helper for spin handling
    def apply_spin(idx: np.ndarray) -> np.ndarray:
        if spin == "all":
            return np.concatenate((idx, idx + hamk.shape[1] // 2))
        if spin == "down":
            return idx + hamk.shape[1] // 2
        return idx

    # Expected orbitals per layer (assuming group 0 orbitals define block width)
    orb_per_layer_0 = int(num_orb_per_layer[0][0])
    q_count_0 = int(num_q_list[0][0])

    if mode == "gamma":
        # Same-Q across all layers grouped together
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover
            tqdm = lambda x, total=None, miniters=None: x  # fallback

        for iqx in tqdm(range(q_count_0), total=q_count_0, miniters=1):
            same_q_index_parts: List[np.ndarray] = []
            for ilx in range(len(num_layer_arr)):
                for j in range(num_layer_arr[ilx]):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer_0, (iqx_ + 1) * orb_per_layer_0)
                    shift = q_count_0 * orb_per_layer_0
                    same_q_index_parts.append(base + shift * num_layer_arr[:ilx].sum())
            # print(same_q_index_parts)
            same_q_index = np.concatenate(same_q_index_parts)

            expected = int(num_layer_arr.sum()) * orb_per_layer_0
            if same_q_index.shape[0] != expected:
                raise ValueError(
                    f"Mismatch in block size: got {same_q_index.shape[0]} vs expected {expected}"
                )

            same_q_index = apply_spin(same_q_index)
            block = hamk[np.ix_(same_q_index, same_q_index)]
            eig, vec = np.linalg.eigh(block)
            # Layered alignment with linear-combo references (minimal and explicit)
            if isinstance(nlow_state_list, list) and len(nlow_state_list) >= 1 \
               and isinstance(norb_fix_list, list) and len(norb_fix_list) >= 1:
                # Flatten per-layer definitions into band list and per-band combos
                bands_flat: list[int] = []
                ref_flat: list[list[tuple[int, complex]]] = []
                for layer in range(1):
                    for b in nlow_state_list[layer]:
                        bands_flat.append(int(b))
                    for combos in norb_fix_list[layer]:
                        parsed: list[tuple[int, complex]] = []
                        for it in combos:
                            if isinstance(it, (list, tuple)) and len(it) == 2:
                                idxc, coef = it
                                coef_c = complex(coef) if not isinstance(coef, complex) else coef
                                parsed.append((int(idxc), coef_c))
                            else:
                                parsed.append((int(it), complex(1.0)))
                        ref_flat.append(parsed)
                print("bands_flat :",bands_flat)
                # ref_flat = [[(50, (1+0j))], [(142, (1+0j))], [(123, (1+0j))], [(31, (1+0j))], [(50, (1+0j))], [(142, (1+0j))], [(123, (1+0j))], [(31, (1+0j))]]
                
                print("ref_flat :", ref_flat)
                if bands_flat and len(ref_flat) == len(bands_flat):
                    # Phi_ref = np.zeros((vec.shape[0], len(bands_flat)), dtype=complex)
                    # for j, combos in enumerate(ref_flat):
                    #     for idxc, coef in combos:
                    #         if 0 <= int(idxc) < Phi_ref.shape[0]:
                    #             Phi_ref[int(idxc), j] += complex(coef)

                    Phi_ref = np.zeros((vec.shape[0], len(bands_flat)), dtype=complex)
                    for j, items in enumerate(ref_flat):
                        col = np.zeros(Phi_ref.shape[0], dtype=complex)
                        for idxc, coef in items:
                            if 0 <= idxc < Phi_ref.shape[0]:
                                col[idxc] += coef
                        # 逐列归一，避免幅值偏置
                        nrm = np.linalg.norm(col)
                        if nrm > 0:
                            col /= nrm
                        Phi_ref[:, j] = col
                    
                    
                    U_low = vec[:, np.array(bands_flat, dtype=int)]
                    U_aligned, _ = align_eigenstates(U_low, Phi_ref)
                    for col_j, b in enumerate(bands_flat):
                        vec[:, int(b)] = U_aligned[:, col_j]
            
            V0 = np.eye(len(bands_flat), dtype=complex)
            for i, b in enumerate(bands_flat):
                a = np.vdot(Phi_ref[:, i], vec[:, b])
                if abs(a) > 1e-14:
                    V0[i, i] = np.conj(a) / abs(a)
            vec[:, bands_flat] = vec[:, bands_flat] @ V0
            
            
            bands = np.array(nlow_state_list[0])
            comps = norb_fix_list[0]
            
            # Phi_ref = np.zeros_like(vec[:,bands], dtype=complex)
            # for i in range(len(nlow_state_list[0])):
            #     # for j in comps:
            #     Phi_ref[comps[i], i] = 1
            # # Phi_ref[30,7] = 1
            # # Phi_ref[31,7] = 1j
            # # Phi_ref[248, 0] = 1
            # # Phi_ref[35, 1] = 1
            # # Phi_ref[319, 2] = np.sqrt(2)/2
            # # Phi_ref[390, 2] = 1j*np.sqrt(2)/2
            # # Phi_ref[106, 3] = np.sqrt(2)/2
            # # Phi_ref[177, 3] = -1j*np.sqrt(2)/2
            # U_aligned, V_1 = align_eigenstates(vec[:,bands], Phi_ref)
            # vec[:, bands] = U_aligned
            # # vec[:, bands] = Phi_ref wrong
            
            # V_0 = np.zeros((len(bands), len(bands)), dtype=complex)
            # for i in range(len(bands)):
            #     phase = np.abs(vec[comps[i], bands[i]]) / vec[comps[i], bands[i]]
            #     V_0[i, i] = phase
            # vec[:, bands] = vec[:, bands] @ V_0
            
            

            H_diag_block.append(block)
            H_GM_diag_eig.append(eig)
            
            H_GM_diag_eig_vec.append(vec)

        H_GM_diag_eig = np.array(H_GM_diag_eig, dtype=object)
        H_GM_diag_eig_vec = np.array(H_GM_diag_eig_vec, dtype=object)
        H_diag_block = np.array(H_diag_block, dtype=object)
    else:
        # Per-layer, per-Q blocks
        for ilx in range(len(num_layer_arr)):
            for j in range(num_layer_arr[ilx]):
                for iq in range(int(num_q_list[ilx][0])):
                    iqx = iq * num_layer_arr[ilx] + j
                    base = np.arange(iqx * orb_per_layer_0, (iqx + 1) * orb_per_layer_0)
                    shift = q_count_0 * orb_per_layer_0
                    same_q_index = base + shift * num_layer_arr[:ilx].sum()
                    same_q_index = apply_spin(same_q_index)

                    block = hamk[np.ix_(same_q_index, same_q_index)]
                    eig, vec = np.linalg.eigh(block)

                    H_diag_block.append(block)
                    H_GM_diag_eig.append(eig)
                    H_GM_diag_eig_vec.append(vec)

    return (
        np.array(H_GM_diag_eig, dtype=object),
        np.array(H_GM_diag_eig_vec, dtype=object),
        np.array(H_diag_block, dtype=object),
        np.array(U_new_block, dtype=object),
    )


# def get_h_dft_low(
#     hamk_block: np.ndarray,
#     nlow_state_list: List[List[int]] | List[int],
#     norb_fix_list: List[List[int]] | List[int],
#     Q_set1: np.ndarray | None = None,
#     Q_set2: np.ndarray | None = None,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Second-order effective Hamiltonian for a single k (full H(k)).

#     Heff = h00 + h01 (E_ref I - h11)^{-1} h10,
#     where h00 = Uproj^† H Uproj, h11 = UHE^† H UHE, h01 = Uproj^† H UHE.
#     Uproj is built by selecting per-Q bands and aligning to norb_fix via SVD.
#     """

#     H = np.asarray(hamk_block, dtype=np.complex128)
#     # Use spin-up block only if two-spin present (as in notebook)
#     if H.shape[0] % 2 == 0:
#         N_up = H.shape[0] // 2
#         h_dft = H[:N_up, :N_up]
#     else:
#         h_dft = H

#     # Flatten helpers
#     def _flatten(x):
#         if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
#             out = []
#             for s in x:
#                 out.extend(list(s))
#             return list(out)
#         return list(x)

#     bands_flat = np.array(_flatten(nlow_state_list), dtype=int)
#     comps_flat = np.array(_flatten(norb_fix_list), dtype=int)
#     if bands_flat.size == 0:
#         raise ValueError("nlow_state_list is empty")

#     if Q_set1 is None:
#         raise ValueError("Q_set1 is required to infer q_count")
#     q_count = int(len(Q_set1))
#     num_layers = 2
#     # Infer orbitals per layer per Q (single spin)
#     if q_count * num_layers == 0:
#         raise ValueError("Invalid q_count/num_layers")
#     orb_per_layer0 = h_dft.shape[0] // (q_count * num_layers)
#     if orb_per_layer0 * q_count * num_layers != h_dft.shape[0]:
#         raise ValueError("Dimension mismatch: cannot infer orb_per_layer per Q")

#     # Build per-Q index mapping into the full H_up(k)
#     num_layer_arr = np.array([1, 1])
#     shift = q_count * orb_per_layer0
#     idx_list: List[np.ndarray] = []
#     for iqx in range(q_count):
#         parts = []
#         for ilx in range(len(num_layer_arr)):
#             for j in range(num_layer_arr[ilx]):
#                 iqx_ = iqx * num_layer_arr[ilx] + j
#                 base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
#                 parts.append(base + shift * num_layer_arr[:ilx].sum())
#         same_q_index = np.concatenate(parts)
#         idx_list.append(same_q_index)

#     # Construct Uproj and UHighEnergy
#     m_bands = bands_flat.size
#     N = h_dft.shape[0]
#     M = m_bands * q_count
#     Uproj = np.zeros((N, M), dtype=np.complex128)
#     Uhe_cols: List[np.ndarray] = []

#     for iq, idx in enumerate(idx_list):
#         block = h_dft[np.ix_(idx, idx)]
#         eigvals, vec = np.linalg.eigh(block)
#         # low subspace inside this block (bands are per-block indices)
#         U_low_blk = vec[:, bands_flat]
#         # Align using global norb_fix mapped to block-local positions
#         Phi_blk = np.zeros_like(U_low_blk)
#         if comps_flat.size == m_bands:
#             inv_map = {int(g): i for i, g in enumerate(idx.tolist())}
#             for j, g in enumerate(comps_flat.tolist()):
#                 if int(g) in inv_map:
#                     Phi_blk[inv_map[int(g)], j] = 1.0
#         U_align_blk, _ = align_eigenstates(U_low_blk, Phi_blk) if np.any(Phi_blk) else (U_low_blk, np.eye(m_bands, dtype=np.complex128))
#         for j in range(m_bands):
#             Uproj[idx, iq * m_bands + j] = U_align_blk[:, j]
#         # high-energy complement in block
#         all_idx = np.arange(vec.shape[0], dtype=int)
#         mask = np.ones_like(all_idx, dtype=bool)
#         mask[bands_flat] = False
#         Uhigh_blk = vec[:, mask]
#         for col in range(Uhigh_blk.shape[1]):
#             v = np.zeros((N,), dtype=np.complex128)
#             v[idx] = Uhigh_blk[:, col]
#             Uhe_cols.append(v)

#     UHighEnergy = np.stack(Uhe_cols, axis=1) if Uhe_cols else np.zeros((N, 0), dtype=np.complex128)

#     # Schrieffer–Wolff second-order effective Hamiltonian
#     h00 = Uproj.conj().T @ h_dft @ Uproj
#     X = h_dft @ UHighEnergy
#     h11 = UHighEnergy.conj().T @ X
#     h01 = Uproj.conj().T @ X
#     h10 = h01.conj().T

#     # Reference energy: mean eigenvalue of h00
#     eff_energy = float(np.min(np.linalg.eigvalsh(h00))) if h00.size else 0.0
#     print(eff_energy)
#     if h11.shape[0] > 0:
#         temp = np.linalg.solve(eff_energy * np.eye(h11.shape[0], dtype=np.complex128) - h11, h10)
#         Heff = h00 + h01 @ temp
#     else:
#         Heff = h00.copy()
#     Heff = 0.5 * (Heff + Heff.conj().T)
#     heig, hvec = np.linalg.eigh(Heff)
#     sort_idx = np.argsort(heig)
#     heig = heig[sort_idx]
#     hvec = hvec[:, sort_idx]
#     return Heff, heig, hvec


# def project_heff_full(
#     hamk_full: np.ndarray,
#     q_count: int,
#     orb_per_layer0: int,
#     num_layer_list: List[int],
#     *,
#     spin: Literal["up", "down", "all"] = "up",
#     bands: List[int] | List[List[int]] = None,
#     comps: List[int] | List[List[int]] = None,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Project full H(k) to Heff(k) with basis size M = bands_per_Q * q_count.

#     - hamk_full: (N, N) complex Hermitian for one k
#     - q_count: number of Q points (from len(Q_set1))
#     - orb_per_layer0: orbitals per layer per Q (single spin)
#     - num_layer_list: list per valley side, e.g., [1,1]
#     - spin: 'up'|'down'|'all' (affects index mapping)
#     - bands: per-Q selected band indices (flattened or nested); length = M_bands
#     - comps: per-band reference orbital global indices (flattened or nested); length = M_bands

#     Returns Heff(k) (M, M), its eigenvalues (M,), and eigenvectors (M, M).
#     """

#     if bands is None:
#         raise ValueError("bands (nlow_state_list) must be provided")
#     if comps is None:
#         comps = []

#     # Flatten helpers
#     def _flatten(x):
#         if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
#             out = []
#             for s in x:
#                 out.extend(list(s))
#             return list(out)
#         return list(x)

#     bands_flat = np.array(_flatten(bands), dtype=int)
#     comps_flat = np.array(_flatten(comps), dtype=int) if comps else np.array([], dtype=int)
#     m_bands = bands_flat.size
#     if m_bands == 0:
#         raise ValueError("Empty bands list for projection")

#     # Build per-Q index mapping into full H(k)
#     num_layer_arr = np.array(num_layer_list)
#     shift = q_count * orb_per_layer0
#     idx_list: List[np.ndarray] = []
#     for iqx in range(q_count):
#         parts = []
#         for ilx in range(len(num_layer_arr)):
#             for j in range(num_layer_arr[ilx]):
#                 iqx_ = iqx * num_layer_arr[ilx] + j
#                 base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
#                 parts.append(base + shift * num_layer_arr[:ilx].sum())
#         same_q_index = np.concatenate(parts)
#         if spin == "all":
#             same_q_index = np.concatenate((same_q_index, same_q_index + hamk_full.shape[0] // 2))
#         elif spin == "down":
#             same_q_index = same_q_index + hamk_full.shape[0] // 2
#         idx_list.append(same_q_index)

#     # First-order projection: Heff = U^† H U
#     N = hamk_full.shape[0]
#     M = m_bands * q_count
#     U = np.zeros((N, M), dtype=np.complex128)

#     for iq, idx in enumerate(idx_list):
#         block = np.asarray(hamk_full[np.ix_(idx, idx)], dtype=np.complex128)
#         _, vec = np.linalg.eigh(block)
#         U_low = vec[:, bands_flat]
#         Phi = np.zeros((block.shape[0], m_bands), dtype=np.complex128)
#         if comps_flat.size == m_bands:
#             inv_map = {int(g): i for i, g in enumerate(idx.tolist())}
#             for j, g in enumerate(comps_flat.tolist()):
#                 if int(g) in inv_map:
#                     Phi[inv_map[int(g)], j] = 1.0
#         if np.any(Phi):
#             U_aligned, _ = align_eigenstates(U_low, Phi)
#         else:
#             U_aligned = U_low
#         for j in range(m_bands):
#             U[idx, iq * m_bands + j] = U_aligned[:, j]

#     Heff = U.conj().T @ np.asarray(hamk_full, dtype=np.complex128) @ U
#     print("heff mean=",np.mean(np.linalg.eigvalsh(Heff)))
#     Heff = 0.5 * (Heff + Heff.conj().T)
#     heig, hvec = np.linalg.eigh(Heff)
#     return Heff, heig, hvec

def project_heff_full(
    hamk_full: np.ndarray,
    q_count: int,
    orb_per_layer0: int,
    num_layer_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    bands: List[int] | List[List[int]] = None,
    comps: List[int] | List[List[int]] = None,
    second_order: bool = True,
    E_ref: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project full H(k) to Heff(k).

    一阶：Heff = U_low^† H U_low
    二阶（Löwdin/SW 下折）：Heff = H00 + H01 (E_ref I - H11)^{-1} H10

    参数
    - hamk_full: (N, N) 该 k 点的全空间厄米矩阵
    - q_count: Q 的个数（如 len(Q_set1)）
    - orb_per_layer0: 单自旋、每层、每 Q 的轨道数（用于索引）
    - num_layer_list: 每个“侧/谷”的层数，如 [1,1]
    - spin: 'up'|'down'|'all'（影响索引映射）
    - bands: 每个 Q 选取的“低能带”索引（可平铺或嵌套）；长度 = m_bands
    - comps: 每个低能带希望对齐的“参考全局轨道”索引（与 bands 对应；可选）
    - second_order: False 则一阶；True 则按二阶下折
    - E_ref: 二阶中的参考能量，若 None 则取 H00 本征值均值

    返回
    - Heff(k) (M, M)；其本征值 heig (M,) 与本征矢 hvec (M, M)
    """
    if bands is None:
        raise ValueError("bands (nlow_state_list) must be provided")

    # 将 bands/comps 摊平
    def _flatten(x):
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
            out = []
            for s in x:
                out.extend(list(s))
            return list(out)
        return list(x)

    bands_flat = np.array(_flatten(bands), dtype=int)
    comps_flat = np.array(_flatten(comps), dtype=int) if comps else np.array([], dtype=int)
    m_bands = bands_flat.size
    if m_bands == 0:
        raise ValueError("Empty bands list for projection")

    # 构建每个 Q 的全局索引
    num_layer_arr = np.array(num_layer_list)
    shift = q_count * orb_per_layer0
    idx_list: List[np.ndarray] = []
    for iqx in range(q_count):
        parts = []
        for ilx in range(len(num_layer_arr)):
            for j in range(num_layer_arr[ilx]):
                iqx_ = iqx * num_layer_arr[ilx] + j
                base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
                parts.append(base + shift * num_layer_arr[:ilx].sum())
        same_q_index = np.concatenate(parts)
        if spin == "all":
            same_q_index = np.concatenate((same_q_index, same_q_index + hamk_full.shape[0] // 2))
        elif spin == "down":
            same_q_index = same_q_index + hamk_full.shape[0] // 2
        idx_list.append(same_q_index)

    # 低能投影矩阵 U_low（N x M）
    N = hamk_full.shape[0]
    M = m_bands * q_count
    U_low_full = np.zeros((N, M), dtype=np.complex128)

    # 为二阶准备的高能子空间收集器
    high_parts: List[Tuple[np.ndarray, np.ndarray]] = []  # (block_global_idx, U_high_block)

    for iq, idx in enumerate(idx_list):
        block = np.asarray(hamk_full[np.ix_(idx, idx)], dtype=np.complex128)
        # 注意：np.linalg.eigh 返回升序本征
        _, vec = np.linalg.eigh(block)

        # 处理负索引：映射到 [0, block_dim)
        block_dim = vec.shape[0]
        bands_pos = np.mod(bands_flat, block_dim)

        # 低能矢量
        U_low = vec[:, bands_pos]

        # 参考基对齐（如果提供 comps）
        if comps_flat.size == m_bands:
            pos_map = {int(g): i for i, g in enumerate(idx.tolist())}  # 全局->局部
            Phi = np.zeros((block_dim, m_bands), dtype=np.complex128)
            for j, g in enumerate(comps_flat.tolist()):
                if int(g) in pos_map:
                    Phi[pos_map[int(g)], j] = 1.0
            if np.any(Phi):
                U_low, _ = align_eigenstates(U_low, Phi)

        # 填回全空间 U_low_full
        for j in range(m_bands):
            U_low_full[idx, iq * m_bands + j] = U_low[:, j]

        # 若要二阶，下折需要高能正交补
        if second_order:
            all_cols = np.arange(block_dim)
            high_cols = np.setdiff1d(all_cols, bands_pos, assume_unique=False)
            U_high = vec[:, high_cols]  # 已正交（来自厄米对角化）
            high_parts.append((idx, U_high))

    if not second_order:
        # 一阶：Heff = U† H U
        Heff = U_low_full.conj().T @ np.asarray(hamk_full, dtype=np.complex128) @ U_low_full
        Heff = 0.5 * (Heff + Heff.conj().T)
        heig, hvec = np.linalg.eigh(Heff)
        print("heff mean=", float(np.mean(heig)))
        return Heff, heig, hvec

    # ---------- 二阶：组装 U_high_full ----------
    high_dim_total = sum(Uh.shape[1] for _, Uh in high_parts)
    U_high_full = np.zeros((N, high_dim_total), dtype=np.complex128)
    col = 0
    for idx, Uh in high_parts:
        k = Uh.shape[1]
        U_high_full[idx, col : col + k] = Uh
        col += k

    # 分块：H00/H11/H01/H10
    H = np.asarray(hamk_full, dtype=np.complex128)
    H00 = U_low_full.conj().T @ H @ U_low_full
    X = H @ U_high_full
    H11 = U_high_full.conj().T @ X
    H01 = U_low_full.conj().T @ X
    H10 = H01.conj().T

    # 参考能量
    if E_ref is None:
        E_ref = float(np.min(np.linalg.eigvalsh(H00)))-0
        print("E ref = ", E_ref)
    # 二阶下折
    # Heff = H00 + H01 @ (E_ref*I - H11)^{-1} @ H10
    A = (E_ref * np.eye(H11.shape[0], dtype=np.complex128)) - H11
    # 先解 A * Y = H10，数值更稳
    Y = np.linalg.solve(A, H10)
    Heff = H00 + H01 @ Y
    Heff = 0.5 * (Heff + Heff.conj().T)

    heig, hvec = np.linalg.eigh(Heff)
    # print("heff mean=", float(np.mean(heig)))
    return Heff, heig, hvec
