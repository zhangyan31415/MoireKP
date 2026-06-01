from __future__ import annotations

from typing import List, Tuple, Literal

import numpy as np
import scipy
import scipy.linalg

from .downfold import DownfoldingOptions, downfold_from_projectors


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

        # for iqx in tqdm(range(q_count_0), total=q_count_0, miniters=1):
        for iqx in range(q_count_0):
            same_q_index_parts: List[np.ndarray] = []
            for ilx in range(len(num_layer_arr)):
                for j in range(num_layer_arr[ilx]):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer_0, (iqx_ + 1) * orb_per_layer_0)
                    # index2 = np.array([15, 18, 20, 16, 19, 21, 24, 26, 25, 27, 14, 22, 17, 23])-14
                    # iqx2 = index2[iqx] * num_layer_arr[ilx] + j
                    # base_2 = np.arange(iqx2 * orb_per_layer_0, (iqx2 + 1) * orb_per_layer_0)
                    # if ilx != 0:
                    #     print("base_2 = ",base_2)
                    #     print("base = ",base)
                    shift = q_count_0 * orb_per_layer_0
                    if ilx == 0:
                        same_q_index_parts.append(base + shift * num_layer_arr[:ilx].sum())
                    else:
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
            # print(block[10,20])
            eig, vec = np.linalg.eigh(block)
            # print(np.sort(eig)[:5],np.linalg.norm(block))
            # Layered alignment with linear-combo references (minimal and explicit)
            # print("nlow_state_list = ",nlow_state_list)
            # print("norb_fix_list = ",norb_fix_list)
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
                # print("bands_flat :",bands_flat)
                # ref_flat = [[(50, (1+0j))], [(142, (1+0j))], [(123, (1+0j))], [(31, (1+0j))], [(50, (1+0j))], [(142, (1+0j))], [(123, (1+0j))], [(31, (1+0j))]]

                # print("ref_flat :", ref_flat)
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

                    # print("none zero index and value = ",Phi_ref.nonzero(),Phi_ref[Phi_ref.nonzero()])
                    U_low = vec[:, np.array(bands_flat, dtype=int)]
                    U_aligned, _ = align_eigenstates(U_low, Phi_ref)
                    # for col_j, b in enumerate(bands_flat):
                    #     vec[:, int(b)] = U_aligned[:, col_j]
                    vec[:, bands_flat] = U_aligned#*np.exp(-1j*7*np.pi/6)

            V_0 = np.zeros((len(bands_flat), len(bands_flat)), dtype=complex)
            comps = norb_fix_list[0]
            for i in range(len(bands_flat)):
                # phase = np.abs(vec[comps[i], bands_flat[i]]) / vec[comps[i], bands_flat[i]]
                phase = np.exp(-1j * 2 * 210 * np.pi / 180)
                V_0[i, i] = phase
                # print("phase = ",phase)
            # vec[:, bands_flat] = vec[:, bands_flat] @ V_0

            V0 = np.eye(len(bands_flat), dtype=complex)
            for i, b in enumerate(bands_flat):
                a = np.vdot(Phi_ref[:, i], vec[:, b])
                if abs(a) > 1e-14:
                    V0[i, i] = np.conj(a) / abs(a)
                    #  V0[i, i] =  abs(a)/ np.conj(a)
            # print("none zero V0 = ",V0.nonzero(),V0[V0.nonzero()])
            # vec[:, bands_flat] = vec[:, bands_flat] @ V0


            # bands = np.array(nlow_state_list[0])
            # comps = norb_fix_list[0]

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
            for jj in range(num_layer_arr[ilx]):
                for iq in range(int(num_q_list[ilx][0])):
                    iqx = iq * num_layer_arr[ilx] + jj
                    base = np.arange(iqx * orb_per_layer_0, (iqx + 1) * orb_per_layer_0)
                    shift = q_count_0 * orb_per_layer_0
                    same_q_index = base + shift * num_layer_arr[:ilx].sum()
                    same_q_index = apply_spin(same_q_index)
                    # print(f"ilx = {ilx}, j = {jj}, iq = {iq}, iqx = {iqx}, base = {base}, shift = {shift}, same_q_index = {same_q_index}")
                    block = hamk[np.ix_(same_q_index, same_q_index)]
                    eig, vec = np.linalg.eigh(block)

                    if isinstance(nlow_state_list, list) and len(nlow_state_list) >= 1 \
                        and isinstance(norb_fix_list, list) and len(norb_fix_list) >= 1:
                            # Flatten per-layer definitions into band list and per-band combos
                            bands_flat: list[int] = []
                            ref_flat: list[list[tuple[int, complex]]] = []
                            layer = int(ilx)
                            if layer >= len(nlow_state_list) or layer >= len(norb_fix_list):
                                raise IndexError(
                                    f"Missing low-state/reference config for layer {layer}: "
                                    f"nlow_state_list has {len(nlow_state_list)} layers, "
                                    f"norb_fix_list has {len(norb_fix_list)} layers"
                                )
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
                            # print("bands_flat :",bands_flat)
                            # ref_flat = [[(50, (1+0j))], [(142, (1+0j))], [(123, (1+0j))], [(31, (1+0j))], [(50, (1+0j))], [(142, (1+0j))], [(123, (1+0j))], [(31, (1+0j))]]

                            # print("ref_flat :", ref_flat)
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

                                # print("none zero index and value = ",Phi_ref.nonzero(),Phi_ref[Phi_ref.nonzero()])
                                U_low = vec[:, np.array(bands_flat, dtype=int)]
                                U_aligned, _ = align_eigenstates(U_low, Phi_ref)
                                # for col_j, b in enumerate(bands_flat):
                                #     vec[:, int(b)] = U_aligned[:, col_j]
                                vec[:, bands_flat] = U_aligned



                    H_diag_block.append(block)
                    H_GM_diag_eig.append(eig)
                    H_GM_diag_eig_vec.append(vec)
    # np.save("//data/work/zy/software/1.tapw_code/moirekp/kp/H_GM_diag_eig_vec.npy",H_GM_diag_eig_vec)
    return (
        np.array(H_GM_diag_eig, dtype=object),
        np.array(H_GM_diag_eig_vec, dtype=object),
        np.array(H_diag_block, dtype=object),
        np.array(U_new_block, dtype=object),
    )


def calculate_energy_lists(
    H_GM_diag_eig_vec,
    nlow_state_list,
    norb_fix_list,
    Qlayer_list,
    num_orb_per_layer_list,
    mode="gamma",
    *,
    include_high: bool = True,
):
    """
    Calculate low energy and high energy lists from the given eigenvector matrix.

    Parameters:
    H_GM_diag_eig_vec (numpy.ndarray): Eigenvector matrix.
    nlow_state_list: List of low energy state indices per layer.
    norb_fix_list: List of fixed orbital indices per layer.
    Qlayer_list: List of Q layers.
    num_orb_per_layer_list: Number of orbitals per layer.
    mode: "gamma" or "K1"/"K2" - determines indexing structure.

    Returns:
    tuple: Low energy list (numpy.ndarray), High energy list (numpy.ndarray).
    """
    # Normalize mode to lowercase
    mode_lower = mode.lower() if isinstance(mode, str) else mode

    theta_rot = 0
    U_low_proj = None
    U_low_proj_list = []
    Qlayer1, Qlayer2 = Qlayer_list
    Qlayer1 = Qlayer1[0]
    Qlayer2 = Qlayer2[0]
    phase = 1

    # Convert H_GM_diag_eig_vec to list if it's an array
    if isinstance(H_GM_diag_eig_vec, np.ndarray):
        H_vec_list = H_GM_diag_eig_vec.tolist()
    else:
        H_vec_list = H_GM_diag_eig_vec

    # Ensure all elements are numpy arrays
    H_vec_list = [np.asarray(v, dtype=np.complex128) for v in H_vec_list]

    for ilayer in range(len(nlow_state_list)):
        nlow_state = nlow_state_list[ilayer]
        if nlow_state == []:
            continue
        norb_fix = norb_fix_list[ilayer]
        Qlayer = Qlayer1 if ilayer == 0 else Qlayer2
        ULowEnergyList = [[] for _ in range(len(nlow_state))]

        for j in range(len(nlow_state)):
            for i in range(len(Qlayer)):
                # Index calculation depends on mode
                if mode_lower == "gamma":
                    # gamma mode: structure is [Q0_all_layers, Q1_all_layers, ...]
                    # But get_H_block returns [Q0, Q1, ...] for gamma mode
                    vec_idx = i
                else:
                    # non-gamma mode: structure is [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, ...]
                    # Index = layer_offset + Q_index
                    # Assuming Qlayer1 and Qlayer2 have same length for simplicity
                    layer_offset = ilayer * len(Qlayer1)
                    vec_idx = layer_offset + i

                if vec_idx >= len(H_vec_list):
                    raise IndexError(f"vec_idx {vec_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}, i={i}")

                # Ensure vec is a numpy array
                vec = np.asarray(H_vec_list[vec_idx], dtype=np.complex128)
                vecpart = vec[:, nlow_state[j]]
                # vecpart = vecpart * ((vecpart[norb_fix[j]] / np.abs(vecpart[norb_fix[j]])) ** (-1)) * np.exp(-1j * 2 * theta_rot * np.pi / 180)

                ULowEnergyList[j].append(vecpart)
            ULowEnergyList[j] = scipy.linalg.block_diag(*ULowEnergyList[j])
        U_low_proj_list.append(np.concatenate(ULowEnergyList,axis=0))
    U_low_proj = scipy.linalg.block_diag(*U_low_proj_list).T

    # The following reordering is only for gamma mode
    # In gamma mode, the structure needs to be reordered for spin and layer separation
    if mode_lower == "gamma":
        # spin_up = np.array([j for i in range(57) for j in range(i * 142, i * 142 + 71)])
        # spin_down = np.array([j for i in range(57) for j in range(i * 142 + 71, i * 142 + 142)])
        n_g = len(Qlayer1)
        nlayers = 2
        orb0 = num_orb_per_layer_list[0][0]
        orb1 = num_orb_per_layer_list[1][0]
        spin_up = np.array([j for i in range(n_g) for j in range(i * orb0 * 2 *nlayers, i * orb0 *2*nlayers + orb0*nlayers)])
        spin_down = np.array([j for i in range(n_g) for j in range(i * orb0 * 2 *nlayers + orb0*nlayers, i * orb0 * 2*nlayers + orb0*2*nlayers)])
        # print("max of spin_up = ",np.max(spin_up))
        # print("max of spin_down = ",np.max(spin_down))
        orb_layer1_up = np.array([j for i in range(n_g) for j in range(i * orb0*nlayers, i * orb0*nlayers + orb0)])
        orb_layer1_down = orb_layer1_up+np.shape(U_low_proj)[0]//2
        orb_layer2_up = np.array([j for i in range(n_g) for j in range(i * orb0*nlayers + orb0, i * orb0*nlayers + orb0*nlayers)])
        orb_layer2_down = orb_layer2_up+np.shape(U_low_proj)[0]//2
        # print(len(spin_up),len(spin_down),len(orb_layer1_up),len(orb_layer1_down),len(orb_layer2_up),len(orb_layer2_down))

        # U_low_proj = np.concatenate((U_low_proj[orb_layer1_up], U_low_proj[orb_layer2_up], U_low_proj[orb_layer1_down], U_low_proj[orb_layer2_down]), axis=0)

        # U_low_proj = np.concatenate((U_low_proj[spin_up], U_low_proj[spin_down]), axis=0)
        # U_low_proj = np.concatenate((U_low_proj[orb_layer1_up], U_low_proj[orb_layer2_up]), axis=0)



        U_low_proj = np.concatenate((U_low_proj[spin_up], U_low_proj[spin_down]), axis=0)
        U_low_proj = np.concatenate((U_low_proj[orb_layer1_up], U_low_proj[orb_layer2_up], U_low_proj[orb_layer1_down], U_low_proj[orb_layer2_down]), axis=0)

    if len(nlow_state_list) == 2:
        if nlow_state_list[0] == []:
            vec0_shape = np.asarray(H_vec_list[0], dtype=np.complex128).shape[0]
            U_low_proj = np.concatenate((np.zeros((vec0_shape*len(Qlayer1),np.shape(U_low_proj)[1])),U_low_proj),axis=0)
        if nlow_state_list[1] == []:
            vec_last_shape = np.asarray(H_vec_list[-1], dtype=np.complex128).shape[0]
            U_low_proj = np.concatenate((U_low_proj,np.zeros((vec_last_shape*len(Qlayer2),np.shape(U_low_proj)[1]))),axis=0)

    # print("shape of Uproj = ",np.shape(U_low_proj))
    if not include_high:
        return U_low_proj, None

    U_high_proj = None
    UHighEnergyList = []
    for ilayer in range(len(nlow_state_list)):
        nlow_state = nlow_state_list[ilayer]
        if nlow_state == []:
            continue
        Qlayer = Qlayer1 if ilayer == 0 else Qlayer2
        for i in range(len(Qlayer)):
            # Index calculation depends on mode
            if mode_lower == "gamma":
                vec_idx = i
            else:
                # non-gamma mode: structure is [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, ...]
                layer_offset = ilayer * len(Qlayer1)
                vec_idx = layer_offset + i

            if vec_idx >= len(H_vec_list):
                raise IndexError(f"vec_idx {vec_idx} >= len(H_vec_list) {len(H_vec_list)} for ilayer={ilayer}, i={i}")

            # Ensure vec is a numpy array
            vec = np.asarray(H_vec_list[vec_idx], dtype=np.complex128)
            orb_num = vec.shape[1]
            vecpart = vec[:, np.delete(np.arange(orb_num), nlow_state)]
            # for j in range(vecpart.shape[1]):
            #     vecpart[:,j] = vecpart[:,j] * ((vecpart[norb_fix,j] / np.abs(vecpart[norb_fix,j])) ** (-1)) * np.exp(-1j * 2 * theta_rot * np.pi / 180)
            vecpart_transposed = vecpart
            UHighEnergyList.append(vecpart_transposed)
    U_high_proj = scipy.linalg.block_diag(*UHighEnergyList)

    # The following reordering is only for gamma mode
    if mode_lower == "gamma":
        # U_high_proj = np.concatenate((U_high_proj[orb_layer1], U_high_proj[orb_layer2]), axis=0)
        # U_high_proj = np.concatenate((U_high_proj[orb_layer1_up], U_high_proj[orb_layer2_up], U_high_proj[orb_layer1_down], U_high_proj[orb_layer2_down]), axis=0)

        # U_high_proj = np.concatenate((U_high_proj[spin_up], U_high_proj[spin_down]), axis=0)
        # U_high_proj = np.concatenate((U_high_proj[orb_layer1_up], U_high_proj[orb_layer2_up]), axis=0)


        U_high_proj = np.concatenate((U_high_proj[spin_up], U_high_proj[spin_down]), axis=0)
        U_high_proj = np.concatenate((U_high_proj[orb_layer1_up], U_high_proj[orb_layer2_up], U_high_proj[orb_layer1_down], U_high_proj[orb_layer2_down]), axis=0)

    if len(nlow_state_list) == 2:
        if nlow_state_list[0] == []:
            vec0_shape = np.asarray(H_vec_list[0], dtype=np.complex128).shape[0]
            U_high_proj = np.concatenate((np.zeros((vec0_shape*len(Qlayer1),np.shape(U_high_proj)[1])),U_high_proj),axis=0)
        if nlow_state_list[1] == []:
            vec_last_shape = np.asarray(H_vec_list[-1], dtype=np.complex128).shape[0]
            U_high_proj = np.concatenate((U_high_proj,np.zeros((vec_last_shape*len(Qlayer2),np.shape(U_high_proj)[1]))),axis=0)
    # print("shape of U_high_proj = ",np.shape(U_high_proj))
    # ULowEnergyList = np.array(ULowEnergyList)
    # UHighEnergyList = np.array(UHighEnergyList)

    # return ULowEnergyList, UHighEnergyList
    return U_low_proj, U_high_proj



def project_heff_full(
    hamk_full: np.ndarray,
    q_count: int,
    orb_per_layer0: int,
    num_layer_list: List[int],
    *,
    spin: Literal["up", "down", "all"] = "up",
    bands: List[int] | List[List[int]] = None,
    comps: List[int] | List[List[int]] = None,
    # NEW: 可直接转调 get_H_block 以复用“对齐”逻辑
    Qlayer_list: List[List[np.ndarray]] | None = None,
    num_orb_per_layer_list: List[List[int]] | None = None,
    nlow_state_list: List[List[int]] | None = None,
    norb_fix_list: List[List[List[Tuple[int, complex]]]] | None = None,
    second_order: bool = True,
    E_ref: float | None = None,
    mode = "Gamma",
    downfold_method: str | None = None,
    pole_warning_mev: float = 10.0,
    pole_danger_mev: float = 1.0,
    fail_on_near_pole: bool = False,
    return_diagnostics: bool = False,
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
    # if bands is None:
    #     raise ValueError("bands (nlow_state_list) must be provided")

    # 将 bands 摊平（comps 另行处理，因其可能含复系数组合）
    # def _flatten(x):
    #     if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
    #         out = []
    #         for s in x:
    #             out.extend(list(s))
    #         return list(out)
    #     return list(x)

    # bands_flat = np.array(_flatten(bands), dtype=int)

    # m_bands = bands_flat.size
    # Normalize mode to lowercase for consistent comparison
    mode_lower = mode.lower() if isinstance(mode, str) else mode
    if mode_lower == "gamma":
        m_bands = len(nlow_state_list[0])
        bands_flat = np.array(nlow_state_list[0])
    else:
        # non-gamma mode: each layer has its own bands
        # For now, assume same bands for both layers, but structure allows per-layer bands
        if len(nlow_state_list) >= 2 and len(nlow_state_list[0]) > 0:
            m_bands = len(nlow_state_list[0])  # bands per layer
            bands_flat = np.array(nlow_state_list[0])
        else:
            raise ValueError("For non-gamma mode, nlow_state_list must have bands for at least layer 0")
    if m_bands == 0:
        raise ValueError("Empty bands list for projection")

    # 构建每个 Q 的全局索引
    num_layer_arr = np.array(num_layer_list)
    shift = q_count * orb_per_layer0
    idx_list: List[np.ndarray] = []

    if mode_lower == "gamma":
        # gamma mode: each Q combines all layers
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
    else:
        # non-gamma mode: each (layer, Q) pair is a separate block
        # Structure: [layer0_Q0, layer0_Q1, ..., layer0_Qn, layer1_Q0, layer1_Q1, ..., layer1_Qn]
        for ilx in range(len(num_layer_arr)):
            for j in range(num_layer_arr[ilx]):
                for iqx in range(q_count):
                    iqx_ = iqx * num_layer_arr[ilx] + j
                    base = np.arange(iqx_ * orb_per_layer0, (iqx_ + 1) * orb_per_layer0)
                    same_q_index = base + shift * num_layer_arr[:ilx].sum()
                    if spin == "all":
                        same_q_index = np.concatenate((same_q_index, same_q_index + hamk_full.shape[0] // 2))
                    elif spin == "down":
                        same_q_index = same_q_index + hamk_full.shape[0] // 2
                    idx_list.append(same_q_index)

        # 低能投影矩阵 U_low（N x M）
        N = hamk_full.shape[0]
        # For non-gamma mode: M = m_bands * total_blocks
        M = m_bands * len(idx_list)

    U_low_full = np.zeros((N, M), dtype=np.complex128)

    # 为二阶准备的高能子空间收集器
    high_parts: List[Tuple[np.ndarray, np.ndarray]] = []  # (block_global_idx, U_high_block)

    # ---------- 若提供对齐所需信息，则先取对齐后的 vec（逐 Q） ----------
    aligned_vec_blocks: List[np.ndarray] | None = None
    if nlow_state_list and norb_fix_list:
        # 若未显式给 Qlayer_list / num_orb_per_layer_list，则按当前参数合成一个“最小可用”布局
        if Qlayer_list is None:
            Qlayer_list = [[np.arange(q_count) for _ in range(n)] for n in num_layer_list]
        if num_orb_per_layer_list is None:
            num_orb_per_layer_list = [[int(orb_per_layer0) for _ in range(n)] for n in num_layer_list]

        # 调用 get_H_block：其中已经做了 Procrustes + 相位规范
        H_eig_blk, H_vec_blk, _, _ = get_H_block(
            hamk_full,                        # ← 原 hamk_full 即 get_H_block 的 Hamk_list
            Qlayer_list,
            num_layer_list,
            num_orb_per_layer_list,
            nlow_state_list,                 # ← 两层/多层写法均可
            norb_fix_list,                   # ← 线性组合 [[idx, coeff], ...]
            spin=spin,
            mode=mode_lower,
        )
        # 规范成列表（每个元素为该 Q 的对齐后 vec）
        aligned_vec_blocks = [np.asarray(v, dtype=np.complex128) for v in H_vec_blk.tolist()]

    for iq, idx in enumerate(idx_list):
        continue
        # block = np.asarray(hamk_full[np.ix_(idx, idx)], dtype=np.complex128)
        if aligned_vec_blocks is not None:
            # 直接复用 get_H_block 的对齐后本征矢
            vec = aligned_vec_blocks[iq]
        # else:
            # 回退：本地对角化（无对齐）
        # _, vec = np.linalg.eigh(block)
        # 处理负索引：映射到 [0, block_dim)
        block_dim = vec.shape[0]
        bands_pos = np.mod(bands_flat, block_dim)

        # 低能矢量
        U_low = vec[:, bands_pos]

        # 填回全空间 U_low_full
        for j in range(m_bands):
            U_low_full[idx, iq * m_bands + j] = U_low[:, j]

        # 若要二阶，下折需要高能正交补
        if second_order:
            all_cols = np.arange(block_dim)
            high_cols = np.setdiff1d(all_cols, bands_pos, assume_unique=False)
            U_high = vec[:, high_cols]  # 已正交（来自厄米对角化）
            high_parts.append((idx, U_high))
    if False:
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
    method = (downfold_method or ("fixed_schur" if second_order else "first_order")).lower()
    include_high = method != "first_order"

    U_low_full,U_high_full = calculate_energy_lists(
        H_vec_blk,
        nlow_state_list,
        norb_fix_list,
        Qlayer_list,
        num_orb_per_layer_list,
        mode=mode_lower,
        include_high=include_high,
    )
    U_low_full = np.array(U_low_full, dtype=np.complex128)
    if U_high_full is not None:
        U_high_full = np.array(U_high_full, dtype=np.complex128)

    result = downfold_from_projectors(
        hamk_full,
        U_low_full,
        U_high_full,
        DownfoldingOptions(
            method=method,
            e_ref=E_ref,
            pole_warning_mev=float(pole_warning_mev),
            pole_danger_mev=float(pole_danger_mev),
            fail_on_near_pole=bool(fail_on_near_pole),
        ),
    )

    Heff = result.heff
    heig, hvec = np.linalg.eigh(Heff)
    if return_diagnostics:
        return Heff, heig, hvec, result
    return Heff, heig, hvec
