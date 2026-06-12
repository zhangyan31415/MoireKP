"""
TAPW (Twisted Atomic Plane Wave) Hamiltonian calculation module
"""
import numpy as np
import scipy
from scipy.linalg import lapack
import scipy.linalg
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import det
import time
import os
import sys
from joblib import Parallel, delayed
from .C3_symm_01 import C3_MoTe2_all, C3_G_matrix
from tqdm import tqdm
from .config import ComputeConfig
from .read_pos_01 import StructureProcessorSpglib
from .read_kpath_01 import KPathGenerator
from .utils import (
    timing_decorator_factory, rotate_vector, unique_sorted, 
    check_hermitian, is_positive_definite, print_sparse_matrix_info, 
    check_sparsity, HARTREE
)
import re
# 设置numpy的打印精度
np.set_printoptions(precision=6)

# Try to import CuPy for GPU support
try:
    import cupy as cp
except ImportError:
    cp = None

class TAPW_parameters:
    """TAPW parameters for twisted material calculations"""
    
    def __init__(self, structure: StructureProcessorSpglib, config: ComputeConfig):
        self.config = config
        self.n_g = self.config.n_g
        self.valley = self.config.valley
        self.structure = structure

        # Initialize matrices
        self.g_matrix = None
        self.g_matrix_conj = None
        self.C3_matrix = None
        self.symm_matrix = None
        self.symm_matrix_inv = None
        self.g_symm_matrix = None
        self.g_symm_matrix_inv = None

        # K-points
        self.K1 = None
        self.K2 = None
        self.g_vec_list_K1 = None
        self.g_vec_list_K2 = None

        # 电场修正项
        self.electric_field_onsite = None  # shape: (num_atoms,)
        if self.config.Electric_field_in_eVpA or self.config.Inner_symmetrical_Electric_Field:
            self._compute_electric_field_onsite()

    def _compute_electric_field_onsite(self):
        """
        计算每个原子的电场能修正项（单位eV），存储在 self.electric_field_onsite
        """
        cfg = self.config
        df = self.structure.df
        if (cfg.Electric_field_in_eVpA is None and not cfg.Inner_symmetrical_Electric_Field):
            self.electric_field_onsite = None
            return
        # 计算零势能面z0

        # cfg.zero_potential_layers 是像 [2,3,4] 这样的全局 sublayer 列表
        global_idxs = set(cfg.zero_potential_layers)

        # 先构建 mask
        mask = False
        global_counter = 0

        # 按层遍历
        for layer in sorted(df['layer'].unique()):
            # 当前层有哪些 local sublayers
            local_subls = sorted(df.loc[df['layer']==layer, 'sublayer'].unique())
            for sub in local_subls:
                # 如果这个 global_counter 在你的列表里，就把对应的 (layer,sub) 全都选上
                if global_counter in global_idxs:
                    mask |= (df['layer']==layer) & (df['sublayer']==sub)
                global_counter += 1

        # 最后求 z 的平均
        if cfg.zero_potential_layers and mask.any():
            z0 = df.loc[mask, 'z'].mean()
        else:
            z0 = 0.0
        
        z = df['z'].values
        delta_z = z - z0
        onsite = np.zeros_like(z, dtype=np.float64)
        print("Electric_field_in_eVpA", cfg.Electric_field_in_eVpA)
        print("delta_z = ", delta_z)
        # Electric_field_in_eVpA
        if cfg.Electric_field_in_eVpA is not None:
            onsite += cfg.Electric_field_in_eVpA * delta_z
        # Inner_symmetrical_Electric_Field
        if cfg.Inner_symmetrical_Electric_Field:
            isp = 0.005  # eV/Å
            # 指向零势能面
            onsite += isp * np.abs(delta_z)
        print("onsite = ", onsite)
        self.electric_field_onsite = onsite

    def calculate_K_points(self):
        """Calculate the K1 and K2 points."""
        m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        m_g_unitvec_2 = self.structure.reciprocal_Tmat[1][:2]
        if self.structure.reciprocal_Tmat[0] @ self.structure.reciprocal_Tmat[1] < -0.01:
            m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]
        else:
            m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        
        n_moire = self.structure.twist_index
        
        if n_moire == 0:
            print("n_moire == 0 is only for non-twisted Heterostructures")
            reciprocal_Tmat_layer1 = self.structure.monolayer_reciprocal_list[self.structure.twist_layer[0]-1]
            reciprocal_Tmat_layer2 = self.structure.monolayer_reciprocal_list[self.structure.twist_layer[0]]
            K1 = 1/3 * reciprocal_Tmat_layer1[0][:2] + 2/3 * reciprocal_Tmat_layer1[1][:2]
            K2 = 1/3 * reciprocal_Tmat_layer2[0][:2] + 2/3 * reciprocal_Tmat_layer2[1][:2]
            if self.config.valley == 1:
                pass
            elif self.config.valley == 2:
                K1 = -K1
                K2 = -K2
            else:
                raise ValueError("For non-twisted Heterostructures, only valley 1 (K1) and 2 (K2) are allowed now")
            offset = K2
            m_K1 = K1 - offset
            m_K2 = K2 - offset
            return K1, K2, m_K1, m_K2, offset

        bravais = getattr(self.config, "bravais", "hex")
        if bravais in {"square", "rect"}:
            # Use the moiré reciprocal basis (2D) to define Γ/X/Y/M in the moiré BZ
            g1 = m_g_unitvec_1
            g2 = m_g_unitvec_2

            # Moiré high-symmetry k-points (mBZ coordinates)
            m_Gamma = np.zeros(2)
            # m_X = 0.5 * g1
            # m_Y = 0.5 * g2
            # m_M = 0.5 * (g1 + g2)
            if n_moire%2 == 1:
                offset_X = 0.5 * (n_moire+1) * (g1 + g2)
                m_X1 = -0.5 * g2
                m_X2 = -0.5 * g1
            else:
                offset_X = 0.5 * n_moire * (g1 + g2)
                m_X1 = 0.5 * g1
                m_X2 = 0.5 * g2
            m_M1 = 0.5 * (g1 - g2)
            m_M2 = 0.5 * (g1 + g2)
            
            # Valley-dependent reciprocal-space offsets (commensurate indexing)
            offset_Gamma = np.zeros(2)
            # offset_X = n_moire * g1
            # offset_Y = n_moire * g2
            # offset_M = n_moire * (g1 + g2)
            
            # offset_X = 0.5 * n_moire * (g1 + g2)
            # offset_Y = 0.5 * n_moire * (-g1 + g2)
            offset_M = n_moire * g1
            
                
            print("g1 = ",g1)
            print("g2 = ",g2)
            print("n_moire = ",n_moire)
            print("offset_X = ",offset_X)
            # print("offset_Y = ",offset_Y)
            print("offset_M = ",offset_M)

            # For square/rect: valley_dict returns (K1, K2) centers used downstream;
            # mK_dict stores the moiré k-point (e.g. X = 1/2*g1).
            valley_dict = {
                5: (m_Gamma + offset_Gamma, m_Gamma + offset_Gamma),  # Γ
                41: (m_X1 + offset_X, m_X1 + offset_X),                  # X
                42: (rotate_vector(m_X1 + offset_X,90), rotate_vector(m_X2 + offset_X,90)),  # Y
                3: (m_M1 + offset_M, m_M2 + offset_M),                   # M
            }

            offset_dict = {
                5: offset_Gamma,
                41: offset_X,
                42: rotate_vector(offset_X,90),
                3: offset_M,
            }

            mK_dict = {
                5: (m_Gamma, m_Gamma),
                41: (m_X1, m_X2),
                42: (rotate_vector(m_X1,90), rotate_vector(m_X2,90)),
                3: (m_M1, m_M2),
            }

            if self.valley not in valley_dict:
                raise ValueError(
                    f"For bravais='{bravais}', supported valleys are {sorted(valley_dict.keys())}. Got {self.valley}"
                )

            K1, K2 = valley_dict[self.valley]
            m_K1, m_K2 = mK_dict[self.valley]
            offset = offset_dict[self.valley]
            print("K1, K2, m_K1, m_K2, offset = ", K1, K2, m_K1, m_K2, offset)
            return K1, K2, m_K1, m_K2, offset

        print("m_g_unitvec_1 = ", m_g_unitvec_1)
        print("m_g_unitvec_2 = ", m_g_unitvec_2)
        offset = -n_moire * m_g_unitvec_1 + n_moire * m_g_unitvec_2

        m_K1 = -1/3 * m_g_unitvec_1 + 2/3 * m_g_unitvec_2
        m_K2 = -2/3 * m_g_unitvec_1 + 1/3 * m_g_unitvec_2
        
        if n_moire % 2 == 1:
            offset_1 = (n_moire + 1) * (m_g_unitvec_1 + m_g_unitvec_2) / 2
            offset_2 = rotate_vector(offset_1, 120)
            offset_3 = rotate_vector(offset_1, 240)
            m_M1 = -1/2 * m_g_unitvec_2
            m_M2 = -1/2 * m_g_unitvec_1
            m_M3 = rotate_vector(m_M2, 120)
        else:
            offset_1 = n_moire * (m_g_unitvec_1 + m_g_unitvec_2) / 2
            offset_2 = rotate_vector(offset_1, 120)
            offset_3 = rotate_vector(offset_1, 240)
            m_M1 = 1/2 * m_g_unitvec_1
            m_M2 = 1/2 * m_g_unitvec_2
            m_M3 = rotate_vector(m_M2, 120)

        valley_dict = {
            1: (m_K1 + offset, m_K2 + offset),
            2: (-m_K1 - offset, -m_K2 - offset),
            11: (rotate_vector(m_K1 + offset, 120), rotate_vector(m_K2 + offset, 120)),
            12: (rotate_vector(m_K1 + offset, 240), rotate_vector(m_K2 + offset, 240)),
            5: (np.zeros(2), np.zeros(2)),
            31: (m_M1 + offset_1, m_M2 + offset_1),
            32: (rotate_vector(m_M1 + offset_1, 120), rotate_vector(m_M2 + offset_1, 120)),
            33: (rotate_vector(m_M1 + offset_1, 240), rotate_vector(m_M2 + offset_1, 240)),
        }
        
        offset_dict = {
            1: offset, 2: -offset, 11: rotate_vector(offset, 120), 
            12: rotate_vector(offset, 240), 5: np.zeros(2),
            31: offset_1, 32: rotate_vector(offset_1, 120), 33: rotate_vector(offset_1, 240)
        }
        
        mK_dict = {
            1: (m_K1, m_K2), 2: (-m_K1, -m_K2),
            11: (rotate_vector(m_K1, 120), rotate_vector(m_K2, 120)),
            12: (rotate_vector(m_K1, 240), rotate_vector(m_K2, 240)),
            5: (np.zeros(2), np.zeros(2)),
            31: (m_M1, m_M2), 32: (rotate_vector(m_M1, 120), rotate_vector(m_M2, 120)),
            33: (rotate_vector(m_M1, 240), rotate_vector(m_M2, 240))
        }

        if self.valley not in mK_dict:
            raise ValueError("Invalid valley in set_const_mtrx_diff_Gn")

        K1, K2 = valley_dict[self.valley]
        m_K1, m_K2 = mK_dict[self.valley]
        offset = offset_dict[self.valley]
        print("K1, K2, m_K1, m_K2, offset = ", K1, K2, m_K1, m_K2, offset)
        return K1, K2, m_K1, m_K2, offset

    def generate_g_vec_list(self):
        """Generate the g_vec_list for K1 and K2."""
        m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]
        m_g_unitvec_2 = self.structure.reciprocal_Tmat[1][:2]
        if self.structure.reciprocal_Tmat[0] @ self.structure.reciprocal_Tmat[1] < 0:
            m_g_unitvec_1 = self.structure.reciprocal_Tmat[0][:2]
        else:
            m_g_unitvec_1 = -self.structure.reciprocal_Tmat[0][:2]

        K1, K2, m_K1, m_K2, offset = self.calculate_K_points()
        self.K1, self.K2 = K1, K2

        g_vec_list = []
        g_3 = -m_g_unitvec_1 - m_g_unitvec_2
        num_add = 2
        
        for i in range(self.n_g + num_add):
            for j in range(self.n_g + num_add):
                g_vec_list.append(i * m_g_unitvec_1 + j * m_g_unitvec_2)

        for i in range(1, self.n_g + num_add):
            for j in range(self.n_g + num_add):
                g_vec_list.append(i * g_3 + j * m_g_unitvec_1)

        for i in range(1, self.n_g + num_add):
            for j in range(1, self.n_g + num_add):
                g_vec_list.append(j * g_3 + i * m_g_unitvec_2)

        o_g_vec_list = np.array(g_vec_list)
        o_g_vec_list_m_K1 = o_g_vec_list - m_K1
        o_g_vec_list_m_K2 = o_g_vec_list - m_K2

        K1_distance = unique_sorted(np.linalg.norm(o_g_vec_list_m_K1, axis=1), 
                                   tolerance=0.1 * np.linalg.norm(m_g_unitvec_1))
        K2_distance = unique_sorted(np.linalg.norm(o_g_vec_list_m_K2, axis=1), 
                                   tolerance=0.1 * np.linalg.norm(m_g_unitvec_1))

        g_vec_list_K1 = []
        g_vec_list_K2 = []
        n_g = self.n_g - 0 if self.valley == 5 else self.n_g
        
        for i, vec in enumerate(o_g_vec_list_m_K1):
            if np.linalg.norm(vec) < K1_distance[n_g - 1] + 0.001:
                g_vec_list_K1.append(o_g_vec_list[i] + offset)
        
        for i, vec in enumerate(o_g_vec_list_m_K2):
            if np.linalg.norm(vec) < K2_distance[n_g - 1] + 0.001:
                g_vec_list_K2.append(o_g_vec_list[i] + offset)
        g_vec_list_K1 = np.array(g_vec_list_K1)
        g_vec_list_K2 = np.array(g_vec_list_K2)
        self.g_vec_list_K1 = np.array(g_vec_list_K1)
        self.g_vec_list_K2 = np.array(g_vec_list_K2)
        # self.g_vec_list_K1 = np.concatenate([g_vec_list_K1, -g_vec_list_K1])
        # self.g_vec_list_K2 = np.concatenate([g_vec_list_K2, -g_vec_list_K2])

        print(self.g_vec_list_K1)
        print("======================")
        print(self.g_vec_list_K2)
        print("num G vectors per layer = ", len(self.g_vec_list_K1),len(self.g_vec_list_K2))

    @timing_decorator_factory(process_id=0)
    def generate_gr_matrix(self):
        """Generate the g_matrix for TAPW using twist_group assignment"""
        # Check that structure has twist_group column
        if 'twist_group' not in self.structure.df.columns:
            raise ValueError(
                "structure.df must contain 'twist_group' column. "
                "Please use StructureProcessorSpglib with the new multi-group support."
            )
        if 'phys_layer' not in self.structure.df.columns:
            raise ValueError(
                "structure.df must contain 'phys_layer' column. "
                "Please use StructureProcessorSpglib with the new multi-group support."
            )
        
        # Get unique twist groups (sorted)
        unique_groups = sorted(self.structure.df['twist_group'].unique())
        n_groups = len(unique_groups)
        
        # Validate twist_group is continuous 0..n_groups-1
        if unique_groups != list(range(n_groups)):
            raise ValueError(
                f"twist_group must be continuous 0..{n_groups-1}, got {unique_groups}"
            )
        
        # Assign g_vec_list: even groups -> K1, odd groups -> K2
        g_vec_list = []
        for group_id in range(n_groups):
            if group_id % 2 == 0:
                g_vec_list.append(self.g_vec_list_K1)
                print(f"twist_group={group_id} -> K1 (g_vec_list length={len(self.g_vec_list_K1)})")
            else:
                g_vec_list.append(self.g_vec_list_K2)
                print(f"twist_group={group_id} -> K2 (g_vec_list length={len(self.g_vec_list_K2)})")
        
        print(f"Total twist groups: {n_groups}")
        self.g_matrix = self.generate_gr_matrix_cpu(self.structure.df, g_vec_list, spin=self.structure.spin)
        self.g_matrix_conj = self.g_matrix.T.conj()
        
        delta = np.abs(det(self.g_matrix @ self.g_matrix_conj))
        self.g_matrix = scipy.sparse.csr_matrix(self.g_matrix)
        self.g_matrix_conj = scipy.sparse.csr_matrix(self.g_matrix_conj)
        
        if delta < 1.0e-2:
            raise Exception("np.abs(det(gr_mtrx @ gr_mtrx.T.conj())) < 1.0e-2")
        else:
            print("np.abs(det(gr_mtrx @ gr_mtrx.T.conj())) = ", delta)
        
    @staticmethod
    def generate_gr_matrix_cpu(structure_df, g_vec_list, spin=None):
        """
        Generate gr_matrix using twist_group assignment (generalized for n_groups >= 2).
        
        Parameters:
        - structure_df (pd.DataFrame): Must contain:
            - 'atom_type': Atom type
            - 'orb_num': Number of orbitals per atom
            - 'twist_group': Twist group index (0..n_groups-1)
            - 'shifted_x', 'shifted_y': Atom coordinates
        - g_vec_list (list): List of g-vector arrays, one per twist_group
            g_vec_list[i] corresponds to twist_group=i
        - spin: Whether to include spin (block diagonal)
        
        Returns:
        - gr_matrix (np.ndarray): Shape (dim_gr_1, dim_gr_2)
        """
        # Check required columns
        required_cols = ['atom_type', 'orb_num', 'twist_group', 'shifted_x', 'shifted_y']
        for col in required_cols:
            if col not in structure_df.columns:
                raise ValueError(f"structure_df must contain '{col}' column")
        
        # Copy DataFrame to avoid modifying original
        df_temp = structure_df.copy()
        
        # Get unique twist groups and validate
        unique_groups = sorted(df_temp['twist_group'].unique())
        n_groups = len(unique_groups)
        if unique_groups != list(range(n_groups)):
            raise ValueError(f"twist_group must be continuous 0..{n_groups-1}, got {unique_groups}")
        
        if len(g_vec_list) != n_groups:
            raise ValueError(f"g_vec_list length ({len(g_vec_list)}) != n_groups ({n_groups})")
        
        # Get unique atom types
        atom_type_list = np.unique(df_temp['atom_type'].values)
        
        # Get orbital numbers per atom type
        atom_orb_num_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['orb_num'].values)[0]
            for atom_type in atom_type_list
        ])
        
        # Get orbital numbers for all atoms
        atom_orb_num_list_all_atom = np.concatenate([
            df_temp[df_temp['atom_type'] == atom_type]['orb_num'].values
            for atom_type in atom_type_list
        ]).flatten()
        
        # Get atom counts per type
        atom_num_list = np.array([
            len(df_temp[df_temp['atom_type'] == atom_type])
            for atom_type in atom_type_list
        ])
        
        # Get twist_group per atom type (assume all atoms of same type have same group)
        atom_twist_group_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['twist_group'].values)[0]
            for atom_type in atom_type_list
        ])
        
        # Get twist_group for all atoms
        atom_twist_group_list_all_atom = np.concatenate([
            df_temp[df_temp['atom_type'] == atom_type]['twist_group'].values
            for atom_type in atom_type_list
        ]).flatten()
        
        # Get atom type for all atoms
        atom_type_list_all_atom = df_temp['atom_type'].values
        
        atom_orb_name_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['orb_name'].values)[0]
            for atom_type in atom_type_list
        ])
        
        # Compute normalization factors
        factor_list = np.array([
            1 / np.sqrt(atom_num)
            for atom_num in atom_num_list
        ])
        
        # Print info
        print("atom_type_list = ", atom_type_list)
        print("atom_orb_num_list = ", atom_orb_num_list)
        print("atom_num_list = ", atom_num_list)
        print("atom_twist_group_list = ", atom_twist_group_list)
        print("atom_orb_name_list = ", atom_orb_name_list)
        print(f"n_groups = {n_groups}")
        
        # Compute dimensions
        # dim_gr_1: sum over all groups of (orbitals in group * g_vecs in group)
        dim_gr_1 = 0
        for group_id in range(n_groups):
            mask_group = atom_twist_group_list == group_id
            n_orbs_in_group = atom_orb_num_list[mask_group].sum()
            n_g_vecs_in_group = len(g_vec_list[group_id])
            dim_gr_1 += n_orbs_in_group * n_g_vecs_in_group
        
        dim_gr_2 = np.sum(atom_num_list * atom_orb_num_list)
        
        print(f"dim_gr_1 = {dim_gr_1}")
        print(f"dim_gr_2 = {dim_gr_2}")
        
        # Initialize gr_matrix
        gr_matrix = np.zeros((dim_gr_1, dim_gr_2), dtype=np.complex128)
        
        # Compute number of orbitals per group
        orb_group_num = np.zeros(n_groups, dtype=int)
        for group_id in range(n_groups):
            mask = atom_twist_group_list == group_id
            orb_group_num[group_id] = atom_orb_num_list[mask].sum()
        
        # Compute number of g-vectors per group
        g_group_num = np.array([len(g_vec_list[i]) for i in range(n_groups)], dtype=int)
        
        # Generate index_list_g
        index_list_g = []
        g_list_index = 0
        for group_id in range(n_groups):
            for i in range(len(g_vec_list[group_id])):
                for j, atom_orb_num in enumerate(atom_orb_num_list):
                    if atom_twist_group_list[j] != group_id:
                        continue
                    g_list_index += 1
                    # shift1: offset within this group for this atom type
                    mask_same_group = atom_twist_group_list[:j] == group_id
                    shift1 = atom_orb_num_list[:j][mask_same_group].sum()
                    # shift2: offset for this g-vector within this group
                    shift2 = i * orb_group_num[group_id]
                    # shift3: offset for previous groups
                    shift3 = np.dot(g_group_num[:group_id], orb_group_num[:group_id])
                    index_list = np.arange(atom_orb_num) + shift1 + shift2 + shift3
                    index_list_g.append(index_list)
        
        # Get atom positions
        pos_array = np.array(df_temp[['shifted_x', 'shifted_y']].values)
        
        # Fill gr_matrix
        g_list_index = -1
        for group_id in range(n_groups):
            for i in range(len(g_vec_list[group_id])):
                for j, atom_orb_num in enumerate(atom_orb_num_list):
                    if atom_twist_group_list[j] != group_id:
                        continue
                    g_list_index += 1
                    for iatom in range(len(atom_type_list_all_atom)):
                        if (atom_twist_group_list_all_atom[iatom] != group_id or
                            atom_type_list_all_atom[iatom] != j):
                            continue
                        r_vec = pos_array[iatom]
                        g_vec = g_vec_list[group_id][i]
                        gr_i_index_include = index_list_g[g_list_index]
                        gr_j_start = np.sum(atom_orb_num_list_all_atom[:iatom])
                        gr_j_end = gr_j_start + atom_orb_num_list_all_atom[iatom]
                        gr_j_index_include = np.arange(gr_j_start, gr_j_end)
                        
                        exp_val = np.exp(-1j * np.dot(g_vec, r_vec))
                        
                        gr_matrix[gr_i_index_include, gr_j_index_include] = [exp_val] * atom_orb_num_list_all_atom[iatom]
                        gr_matrix[gr_i_index_include, gr_j_index_include] *= factor_list[j]
        
        print("factor_list = ", factor_list)
        
        # Compute conjugate transpose
        gr_matrix_conj = gr_matrix.T.conj()
        
        # Check determinant (print summary only, not full matrix)
        det_product = gr_matrix @ gr_matrix_conj
        det_val = np.linalg.det(det_product)
        max_abs = np.max(np.abs(det_product))
        fro_norm = np.linalg.norm(det_product, 'fro')
        print(f"gr_matrix @ gr_matrix_conj: shape={det_product.shape}, max_abs={max_abs:.6e}, fro_norm={fro_norm:.6e}")
        print(f"det(gr_matrix @ gr_matrix_conj) = {det_val:.6e}")
        
        return scipy.linalg.block_diag(gr_matrix, gr_matrix) if spin else gr_matrix

    def generate_gr_matrix_gpu(self):
        """Generate the g_matrix for TAPW using GPU (only supports bilayer n_groups=2)."""
        # Check number of groups
        if 'twist_group' not in self.structure.df.columns:
            raise ValueError("structure.df must contain 'twist_group' column for GPU gr_matrix")
        n_groups = len(self.structure.df['twist_group'].unique())
        if n_groups != 2:
            raise NotImplementedError(
                f"GPU gr_matrix only supports bilayer (n_groups=2). "
                f"Current n_groups={n_groups}. Use CPU for multi-group alternating."
            )
        
        n_wann_perlayer = int(len(self.structure.sort_wann) / 2)
        n_wann = n_wann_perlayer * 2

        num_orbs = self.structure.num_orbs_per_unit_cell * 2
        sel_orbs = np.arange(num_orbs)
        len_type = len(sel_orbs) * 2
        num_g_vec = len(self.g_vec_list_K1)
        gr_mtrx = cp.zeros((num_g_vec * len_type, n_wann), dtype=cp.complex128)

        iter = -1
        print(f"num of orb per cell = {num_orbs}, len type= {len_type} num g vec = {num_g_vec}")

        sort_wann = cp.asarray(self.structure.sort_wann)
        g_vec_list_K1 = cp.asarray(self.g_vec_list_K1)
        g_vec_list_K2 = cp.asarray(self.g_vec_list_K2)

        for ilayer in range(2):
            for i in tqdm(range(num_g_vec)):
                for k, iorb in enumerate(sel_orbs):
                    iter += 1
                    for j in range(n_wann):
                        if j % num_orbs == iorb and int(j / n_wann_perlayer) == ilayer:
                            wann_coord = sort_wann[j, :2]
                            if ilayer == 0:
                                gr_mtrx[iter, j] = cp.exp(-1j * cp.dot(g_vec_list_K1[i], wann_coord))
                            elif ilayer == 1:
                                gr_mtrx[iter, j] = cp.exp(-1j * cp.dot(g_vec_list_K2[i], wann_coord))

        factor = 1 / cp.sqrt(self.structure.num_unit_cell)
        gr_mtrx = factor * gr_mtrx

        delta = cp.abs(det(gr_mtrx @ gr_mtrx.T.conj()))
        if delta < 1.0e-2:
            raise Exception("cp.abs(det(gr_mtrx @ gr_mtrx.T.conj())) < 1.0e-2")
        else:
            print("cp.abs(det(gr_mtrx @ gr_mtrx.T.conj())) = ", delta)

        self.g_matrix = scipy.sparse.csr_matrix(cp.asnumpy(gr_mtrx))

    def generate_C3_matrix(self):
        """Generate the C3_matrix (only supports bilayer n_groups=2)."""
        # Check number of groups
        if 'twist_group' not in self.structure.df.columns:
            raise ValueError("structure.df must contain 'twist_group' column for C3_H")
        n_groups = len(self.structure.df['twist_group'].unique())
        if n_groups != 2:
            raise NotImplementedError(
                f"C3_H is implemented only for bilayer (n_groups=2) in this codebase. "
                f"Current n_groups={n_groups}. Multi-group representation is not implemented; "
                f"using it would be incorrect (avoid silent wrong)."
            )
        
        df_temp = self.structure.df.copy()
        
        atom_type_list = np.unique(df_temp['atom_type'].values)
        
        # Use twist_group instead of layer for grouping
        atom_twist_group_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['twist_group'].values)[0]
            for atom_type in atom_type_list
        ])
        # For backward compatibility, also get layer (should equal phys_layer)
        atom_layer_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['layer'].values)[0]
            for atom_type in atom_type_list
        ])
        
        atom_orb_name_list = np.array([
            np.unique(df_temp[df_temp['atom_type'] == atom_type]['orb_name'].values)[0]
            for atom_type in atom_type_list
        ])
        
        species_str_1layer = atom_orb_name_list[np.where(atom_layer_list == 0)[0]]
        species_str_2layer = atom_orb_name_list[np.where(atom_layer_list == 1)[0]]
        species_1layer = {}
        species_2layer = {}
        
        for idx, item in enumerate(species_str_1layer, start=1):
            # 使用'-'分割字符串，得到原子部分和轨道部分
            try:
                atom_part, orb_part = item.split('-')
            except ValueError:
                print(f"Cannot split item: {item}")
                continue
            
            # 使用正则表达式提取原子符号（假设原子符号由字母组成）
            atom_match = re.match(r'([A-Za-z]+)', atom_part)
            if atom_match:
                atom = atom_match.group(1)
            else:
                print(f"Cannot extract atom symbol: {atom_part}")
                continue
            
            # 使用正则表达式提取轨道类型和对应的数量
            orbitals = {}
            for orb, count in re.findall(r'([spdf])(\d+)', orb_part):
                orbitals[orb] = int(count)
            
            species_1layer[idx] = {"atom": atom, "orbitals": orbitals}
        
        for idx, item in enumerate(species_str_2layer, start=1):
            # 使用'-'分割字符串，得到原子部分和轨道部分
            try:
                atom_part, orb_part = item.split('-')
            except ValueError:
                print(f"Cannot split item: {item}")
                continue
            
            # 使用正则表达式提取原子符号（假设原子符号由字母组成）
            atom_match = re.match(r'([A-Za-z]+)', atom_part)
            if atom_match:
                atom = atom_match.group(1)
            else:
                print(f"Cannot extract atom symbol: {atom_part}")
                continue
            
            # 使用正则表达式提取轨道类型和对应的数量
            orbitals = {}
            for orb, count in re.findall(r'([spdf])(\d+)', orb_part):
                orbitals[orb] = int(count)
            
            species_2layer[idx] = {"atom": atom, "orbitals": orbitals}
        
        print("species 1layer = ", species_1layer)
        print("species 2layer = ", species_2layer)
        
        C3_G_rep_1layer, C3_G_rep_2layer = C3_G_matrix(
            self.g_vec_list_K1, self.g_vec_list_K2, 
            self.structure.reciprocal_Tmat, self.structure.twist_index, 
            valley=self.valley
        )
        
        C3_MoTe2_rep = C3_MoTe2_all(
            C3_G_rep_1layer, C3_G_rep_2layer,
            atoms_species_1layer=species_1layer,
            atoms_species_2layer=species_2layer,
            spin=self.structure.spin
        )
        
        self.C3_matrix = scipy.sparse.csr_matrix(C3_MoTe2_rep)
        C3_matrix_2 = self.C3_matrix @ self.C3_matrix
        np.save("C3_matrix",C3_MoTe2_rep)
        
        self.symm_matrix = [
            scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),
            self.C3_matrix, C3_matrix_2
        ]
        self.symm_matrix_inv = [
            scipy.sparse.csr_matrix(np.eye(self.C3_matrix.shape[0])),
            self.C3_matrix.conj().T, C3_matrix_2.conj().T
        ]
    
    def generate_g_symm_matrix(self):
        """Generate the g_symm_matrix"""
        g_symm_matrix = self.symm_matrix[1] @ self.g_matrix
        g_symm_matrix_2 = self.symm_matrix[2] @ self.g_matrix
        g_symm_matrix_inv = self.g_matrix.conj().T @ self.symm_matrix_inv[1]
        g_symm_matrix_2_inv = self.g_matrix.conj().T @ self.symm_matrix_inv[2]
        
        self.g_symm_matrix = [self.g_matrix, g_symm_matrix, g_symm_matrix_2]
        self.g_symm_matrix_inv = [self.g_matrix.conj().T, g_symm_matrix_inv, g_symm_matrix_2_inv]

    def generate_all_parameters(self):
        """Generate all TAPW parameters"""
        # Check number of groups for C3_H and GPU
        if 'twist_group' in self.structure.df.columns:
            n_groups = len(self.structure.df['twist_group'].unique())
            if self.config.C3_H and n_groups != 2:
                raise NotImplementedError(
                    f"C3_H is implemented only for bilayer (n_groups=2) in this codebase. "
                    f"Current n_groups={n_groups}. Multi-group representation is not implemented; "
                    f"using it would be incorrect (avoid silent wrong)."
                )
            if self.config.gpu and n_groups != 2:
                raise NotImplementedError(
                    f"GPU gr_matrix only supports bilayer (n_groups=2). "
                    f"Current n_groups={n_groups}. Use CPU for multi-group alternating."
                )
        
        self.generate_g_vec_list()
        self.generate_gr_matrix()
        # self.generate_C3_matrix()
        if self.config.C3_H:
            self.generate_C3_matrix()
            self.generate_g_symm_matrix()
        print("g matrix shape = ", self.g_matrix.shape)
        print("TAPW parameters generated successfully!")

class BandStructureCalculator:
    """Band structure and Chern number calculator for twisted materials"""
    
    # Valley mapping
    VALLEY_MAP = {
        1: "K1", 2: "K2", 11: "K1_120", 12: "K1_240", 5: "Gamma",
        31: "M1", 32: "M2", 33: "M3",
        3: "M", 41: "X", 42: "Y"
    }

    def __init__(self, hr_supercell, sr_supercell, structure: StructureProcessorSpglib, config: ComputeConfig, kpath_config: KPathGenerator=None):
        """Initialize the calculator
        
        Args:
            hr_supercell: Hamiltonian matrix in real space
            sr_supercell: Overlap matrix in real space
            structure: Processed structure information
            config: Computation configuration
            kpath_config: K-path configuration (optional, only needed for band structure calculation)
        """
        self.hr_supercell = hr_supercell
        self.sr_supercell = sr_supercell
        self.structure = structure
        self.config = config
        self.kpath_config = kpath_config
        
        self.result = {}
        
        # Validate valley configuration
        if not hasattr(self.config, 'valley') or self.config.valley not in self.VALLEY_MAP:
            raise ValueError(f"Invalid valley configuration. Must be one of {list(self.VALLEY_MAP.keys())}")
        
        self.valley_flag = self.VALLEY_MAP[self.config.valley]
        
        if self.config.TAPW:
            self.TAPW_parameters = TAPW_parameters(self.structure, self.config)
            self.TAPW_parameters.generate_all_parameters()
            
        if self.config.gpu and cp is None:
            raise ImportError("CuPy is not installed. Please install CuPy to use GPU acceleration.")

    def generate_kmesh(self, num_k):
        """Generate a uniform k-point mesh for Chern number calculation
        
        Args:
            num_k: Number of k-points in each direction
            
        Returns:
            numpy.ndarray: Array of k-points in the first Brillouin zone
        """
        
        # Generate uniform mesh
        kx = np.linspace(0, 1, num_k, endpoint=True)
        ky = np.linspace(0, 1, num_k, endpoint=True)
        kx = np.linspace(-0.5, 0.5, num_k, endpoint=True)
        ky = np.linspace(-0.5, 0.5, num_k, endpoint=True)
        K_mesh = np.meshgrid(kx, ky)
        kpoints = np.array(K_mesh).reshape(2, -1).T
        kpoints = np.hstack((kpoints, np.zeros((kpoints.shape[0], 1))))
        return kpoints

    def find_first_above_energy(self, energies, E):
        """Find first band index above energy E for each k-point"""
        idxs = []
        for band in energies:  # band: (nb,)
            idx = np.where(band > E)[0]
            idxs.append(idx[0] if len(idx) > 0 else len(band))
        return np.asarray(idxs, dtype=int)

    def align_bands_by_index(self, energies, E):
        """Align bands by first index above energy E"""
        first_indices = self.find_first_above_energy(energies, E)
        min_index = int(np.min(first_indices))
        max_index = int(np.max(first_indices))
        n_all = energies.shape[1]
        n_keep = n_all - (max_index - min_index)

        if n_keep <= 0:
            raise ValueError(f"No bands to keep after alignment (n_keep={n_keep}). Check energy E={E}")

        filtered = np.empty((energies.shape[0], n_keep), dtype=energies.dtype)
        for i in range(energies.shape[0]):
            start = first_indices[i] - min_index
            end = start + n_keep
            filtered[i] = energies[i, start:end]

        pivot_col = min_index
        return filtered, pivot_col

    def split_vbm_cbm(self, energies, E):
        """Split bands into VBM and CBM parts based on energy E"""
        # Sort each k-point's energies
        energies = np.sort(energies, axis=1)
        
        filtered, pivot_col = self.align_bands_by_index(energies, E)
        vbm = filtered[:, :pivot_col] if pivot_col > 0 else np.array([]).reshape(energies.shape[0], 0)
        cbm = filtered[:, pivot_col:] if pivot_col < filtered.shape[1] else np.array([]).reshape(energies.shape[0], 0)
        
        return filtered, vbm, cbm, pivot_col
    
    def split_vbm_cbm_with_vec(self, energies, vecs, E):
        """Split bands and corresponding eigenvectors into VBM and CBM parts"""
        # energies: (n_kpoints, n_bands)
        # vecs: (n_kpoints, n_orbitals, n_bands) or list of (n_orbitals, n_bands)
        
        # Sort each k-point's energies and get sorting indices
        sort_indices = np.argsort(energies, axis=1)
        energies_sorted = np.sort(energies, axis=1)
        
        filtered_energies, pivot_col = self.align_bands_by_index(energies_sorted, E)
        
        # Split energies
        vbm_energies = filtered_energies[:, :pivot_col] if pivot_col > 0 else np.array([]).reshape(energies.shape[0], 0)
        cbm_energies = filtered_energies[:, pivot_col:] if pivot_col < filtered_energies.shape[1] else np.array([]).reshape(energies.shape[0], 0)
        
        # Split eigenvectors if provided
        vbm_vecs = None
        cbm_vecs = None
        
        if vecs is not None and len(vecs) > 0:
            # Handle case where vecs is a list or array
            if isinstance(vecs, list):
                vecs_array = np.array(vecs)  # (n_kpoints, n_orbitals, n_bands)
            else:
                vecs_array = vecs
            
            # Get the alignment indices for each k-point
            first_indices = self.find_first_above_energy(energies_sorted, E)
            min_index = int(np.min(first_indices))
            max_index = int(np.max(first_indices))
            n_keep = energies.shape[1] - (max_index - min_index)
            
            if n_keep > 0:
                # Sort eigenvectors according to energy sorting
                vecs_sorted = np.zeros_like(vecs_array)
                for k in range(vecs_array.shape[0]):
                    vecs_sorted[k] = vecs_array[k][:, sort_indices[k]]
                
                # Align eigenvectors
                vecs_filtered = np.zeros((vecs_array.shape[0], vecs_array.shape[1], n_keep), dtype=vecs_array.dtype)
                for k in range(vecs_array.shape[0]):
                    start = first_indices[k] - min_index
                    end = start + n_keep
                    vecs_filtered[k] = vecs_sorted[k][:, start:end]
                
                # Split eigenvectors
                if pivot_col > 0:
                    vbm_vecs = vecs_filtered[:, :, :pivot_col]
                if pivot_col < vecs_filtered.shape[2]:
                    cbm_vecs = vecs_filtered[:, :, pivot_col:]
        
        return filtered_energies, vbm_energies, cbm_energies, pivot_col, vbm_vecs, cbm_vecs

    def calculate_band_structure(self, path, kpoints=None):
        """Calculate band structure
        
        Args:
            path: Output path for results
            kpoints: Optional k-points array. If None, uses k-path for band structure.
                    For Chern number calculation, should provide mesh k-points.
        """
        # Create output directories
        os.makedirs(path, exist_ok=True)
        # os.makedirs(os.path.join(path, "band_wave"), exist_ok=True)
        
        # Use provided k-points or generate from k-path
        if kpoints is None:
            if self.kpath_config is None:
                raise ValueError("Must provide either kpoints or kpath_config")
            kpoints = self.kpath_config.kpoints
        
        # Save g-vectors if using TAPW
        if self.config.TAPW:
            np.save(os.path.join(path, f"g_vec_list_{self.config.n_g}_{self.valley_flag}_1layer"),
                   self.TAPW_parameters.g_vec_list_K1)
            np.save(os.path.join(path, f"g_vec_list_{self.config.n_g}_{self.valley_flag}_2layer"),
                   self.TAPW_parameters.g_vec_list_K2)
        if self.config.C3_H:
            np.save(os.path.join(path, f"C3_matrix_{self.valley_flag}"),
                   self.TAPW_parameters.C3_matrix.toarray())
        # Calculate bands
        self.parallel_calculate_band_01(kpoints)
        
        # Save results
        band_data = self.result['eig']
        suffix = "_2d" if getattr(self.config, "mode", None) == "chern" else ""
        # 在chern模式下，添加num_chern标识
        if getattr(self.config, "mode", None) == "chern" and hasattr(self.config, "num_chern"):
            suffix += f"_{self.config.num_chern}"
            os.makedirs(os.path.join(path, "topo"), exist_ok=True)
            path = os.path.join(path, "topo")
        else:
            os.makedirs(os.path.join(path, "band"), exist_ok=True)
            path = os.path.join(path, "band")
        
        # Split bands and eigenvectors by fermi energy
        vec_data = self.result['vec'] if self.config.eig_vec_cal else None
        filtered, vbm, cbm, pivot_col, vbm_vecs, cbm_vecs = self.split_vbm_cbm_with_vec(
            band_data, vec_data, self.config.efermi)
        
        # Save VBM data if exists
        if vbm.size > 0:
            np.savetxt(os.path.join(path, f"band_VBM_{self.valley_flag}_valley{suffix}.txt"),
                      vbm, fmt='%15.11f')
        
        # Save CBM data if exists  
        if cbm.size > 0:
            np.savetxt(os.path.join(path, f"band_CBM_{self.valley_flag}_valley{suffix}.txt"),
                      cbm, fmt='%15.11f')
        
        # Save eigenvectors if calculated
        if self.config.eig_vec_cal:
            if vbm_vecs is not None and vbm_vecs.size > 0:
                np.save(os.path.join(path, f"vec_VBM_{self.valley_flag}_valley{suffix}"), vbm_vecs)
            if cbm_vecs is not None and cbm_vecs.size > 0:
                np.save(os.path.join(path, f"vec_CBM_{self.valley_flag}_valley{suffix}"), cbm_vecs)
        
        # Save Hamiltonian if requested
        if self.config.hamk_save:
            np.save(os.path.join(path, f"hamk_{self.valley_flag}_valley{suffix}"), self.result['hamk'])

    def calculate_chern(self, path):
        """Calculate Chern number using uniform k-point mesh
        
        Args:
            path: Output path for results
        """
        # Generate uniform k-point mesh
        kpoints = self.generate_kmesh(self.config.num_chern)
        print("kpoints shape = ", kpoints.shape)
        print("kpoints = ", kpoints)
        
        # Calculate band structure on the mesh
        self.calculate_band_structure(path, kpoints)
        
        # TODO: Implement actual Chern number calculation using Berry curvature
        # This would involve calculating the Berry curvature at each k-point
        # and integrating over the Brillouin zone
        
        return NotImplemented

    def run_calculation(self, path):
        """Main calculation entry point
        
        Args:
            path: Output path for results
        """
        if self.config.mode == "band":
            self.calculate_band_structure(path)
        elif self.config.mode == "chern":
            self.calculate_chern(path)
        else:
            raise ValueError(f"Unknown calculation mode: {self.config.mode}")

    def rot(self, vec, theta):
        """Rotate a vector by a given angle in degrees"""
        return rotate_vector(vec, theta)
    
    def rot_gpu(self, vec, theta):
        """GPU version of rotation"""
        if cp is None:
            raise ImportError("CuPy is required for GPU operations")
        theta = theta / 180 * cp.pi
        rot_mat = cp.array([[cp.cos(theta), -cp.sin(theta)], [cp.sin(theta), cp.cos(theta)]])
        if len(vec) == 2:
            return cp.dot(rot_mat, vec)
        elif len(vec) == 3:
            temp = cp.zeros(3)
            temp[:2] = cp.dot(rot_mat, vec[:2])
            temp[2] = vec[2]
            return temp

    # @timing_decorator_factory(process_id=0)
    def get_kvec(self, k):
        """Get k vector in reciprocal space"""
        return np.dot(k, self.structure.reciprocal_Tmat)

    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse(self, Hr, k, type = "H"):
        """Get k-space Hamiltonian from real space Hamiltonian"""
        kvec = self.get_kvec(k)
        # print(self.structure.df)
        # exit()
        sorted_wann_x = self.structure.df['x'].values
        sorted_wann_y = self.structure.df['y'].values
        sorted_wann_z = self.structure.df['z'].values
        sorted_wann = np.repeat(np.array([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T,
                               self.structure.df['orb_num'].values, axis=0)

        if self.structure.spin:
            sorted_wann = np.concatenate([sorted_wann, sorted_wann], axis=0)
        
        num_wann = len(sorted_wann)
        mk = scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)
        
        for rvec, values_dic in Hr.items():
            row_index, col_index, val_index = values_dic["row"], values_dic["col"], values_dic["val"]
            m_coor, n_coor = sorted_wann[row_index], sorted_wann[col_index]
            Rvec = np.dot(rvec, self.structure.Tmat)
            phase_factor = np.exp(-1j * np.dot(m_coor - n_coor, kvec)) * np.exp(1j * np.dot(kvec, Rvec))
            mk += scipy.sparse.csr_matrix((val_index * phase_factor, (row_index, col_index)), 
                                         shape=(num_wann, num_wann))

        # --- 电场修正：加到对角元 ---
        ef_onsite = None
        if type == "H":
            if hasattr(self, 'TAPW_parameters') and self.TAPW_parameters.electric_field_onsite is not None:
                # 需要扩展到所有轨道（每个原子有多个轨道）
                orb_num = self.structure.df['orb_num'].values
                ef_onsite = np.repeat(self.TAPW_parameters.electric_field_onsite, orb_num)
                if self.structure.spin:
                    ef_onsite = np.tile(ef_onsite, 2)
                # print("orb_num = ", orb_num)
                # exit()
                # print("self.TAPW_parameters.electric_field_onsite = ", self.TAPW_parameters.electric_field_onsite)
                # print("ef_onsite = ", ef_onsite.shape,ef_onsite)
                mk = mk + scipy.sparse.diags(ef_onsite, 0, shape=(num_wann, num_wann), dtype=np.float64)
        return mk

    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_symm(self, Hr, k):
        """Get k-space Hamiltonian with symmetry operations"""
        kvec = self.get_kvec(k)
        
        # Build sorted_wann array
        sorted_wann_x = self.structure.df['x'].values
        sorted_wann_y = self.structure.df['y'].values
        sorted_wann_z = self.structure.df['z'].values
        sorted_wann = np.repeat(np.array([sorted_wann_x, sorted_wann_y, sorted_wann_z]).T,
                               self.structure.df['orb_num'].values, axis=0)
        sorted_layer_index = np.repeat(np.array(self.structure.df['layer']),
                                      self.structure.df['orb_num'].values, axis=0)
        
        if self.structure.spin:
            sorted_wann = np.concatenate([sorted_wann, sorted_wann], axis=0)
            sorted_layer_index = np.concatenate([sorted_layer_index, sorted_layer_index], axis=0)
        
        num_wann = len(sorted_wann)
        
        # Setup K points
        K1 = self.TAPW_parameters.K1
        K2 = self.TAPW_parameters.K2
        temp_k1 = np.zeros(3)
        temp_k2 = np.zeros(3)
        temp_k1[:2] = K1
        temp_k2[:2] = K2
        K1 = temp_k1
        K2 = temp_k2

        rotations = [0, -120, -240]
        kvec_K1 = [self.rot(kvec + K1, angle) - K1 for angle in rotations]
        kvec_K2 = [self.rot(kvec + K2, angle) - K2 for angle in rotations]

        mk_list = [np.zeros((self.TAPW_parameters.g_matrix.shape[0], 
                            self.TAPW_parameters.g_matrix.shape[0]), dtype=np.complex128) for _ in range(3)]

        for i in range(3):
            partial_mk = 0
            for rvec, values_dic in Hr.items():
                Rvec = np.dot(rvec, self.structure.Tmat)
                row_index, col_index, val = values_dic["row"], values_dic["col"], values_dic["val"]

                # Create boolean masks
                mask_row_1 = sorted_layer_index[row_index] == 0
                mask_row_2 = ~mask_row_1
                mask_col_1 = sorted_layer_index[col_index] == 0
                mask_col_2 = ~mask_col_1
                
                row_coords_1 = sorted_wann[row_index[mask_row_1]]
                row_coords_2 = sorted_wann[row_index[mask_row_2]]
                col_coords_1 = sorted_wann[col_index[mask_col_1]]
                col_coords_2 = sorted_wann[col_index[mask_col_2]]

                # Calculate phase factors
                exp_kvec_K1_Rvec = np.exp(1j * np.dot(kvec_K1[i], Rvec))
                exp_kvec_K2_Rvec = np.exp(1j * np.dot(kvec_K2[i], Rvec))

                phase_m_1 = np.exp(-1j * np.dot(row_coords_1, kvec_K1[i])) * exp_kvec_K1_Rvec
                phase_m_2 = np.exp(-1j * np.dot(row_coords_2, kvec_K2[i])) * exp_kvec_K2_Rvec
                phase_n_1 = np.exp(1j * np.dot(col_coords_1, kvec_K1[i]))
                phase_n_2 = np.exp(1j * np.dot(col_coords_2, kvec_K2[i]))

                phase_m = np.zeros(len(row_index), dtype=np.complex128)
                phase_n = np.zeros(len(col_index), dtype=np.complex128)
                phase_m[mask_row_1], phase_m[mask_row_2] = phase_m_1, phase_m_2
                phase_n[mask_col_1], phase_n[mask_col_2] = phase_n_1, phase_n_2

                phase_factors = phase_m * phase_n
                data_values = val * phase_factors
                partial_mk += scipy.sparse.csr_matrix((data_values, (row_index, col_index)), 
                                                     shape=(num_wann, num_wann))

            temp = self.cal_TAPW_hamiltonian_k(partial_mk)
            mk_list[i] = temp

        return mk_list

    @timing_decorator_factory(process_id=0) 
    def Getk_super_gauge_sparse_final_HS(self, Hr, Sr, k, mpi_index):
        """Get final Hamiltonian for orthogonal or non-orthogonal basis (no symmetry)"""
        Hk = self.Getk_super_gauge_sparse(Hr, k, type = "H")
        Hk = self.cal_TAPW_hamiltonian_k(Hk)
        if self.config.orthogonal_basis:
            return Hk, None
        else:
            Sk = self.Getk_super_gauge_sparse(Sr, k, type = "S")
            Sk = self.cal_TAPW_hamiltonian_k(Sk)
            if not self.config.ge:
                Hk = self.gen_H_new(Hk, Sk, mpi_index)
                return Hk, None
            else:
                return Hk, Sk

    
    @timing_decorator_factory(process_id=0)
    def C3_symm(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        """Apply C3 symmetry to Hamiltonian"""
        if not self.config.gpu:
            return self.C3_symm_cpu(hamk, hamk_C1, hamk_C2, C3_matrix)
        else:
            return self.C3_symm_gpu(hamk, hamk_C1, hamk_C2, C3_matrix)

    def C3_symm_cpu(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        """CPU version of C3 symmetry"""
        C3_matrix_2 = C3_matrix @ C3_matrix
        return (hamk + C3_matrix @ hamk_C1 @ C3_matrix.conj().T + 
                C3_matrix_2 @ hamk_C2 @ C3_matrix_2.conj().T) / 3

    def C3_symm_gpu(self, hamk, hamk_C1, hamk_C2, C3_matrix):
        """GPU version of C3 symmetry"""
        C3_matrix_2 = C3_matrix @ C3_matrix
        return (hamk + C3_matrix @ hamk_C1 @ C3_matrix.conj().T + 
                C3_matrix_2 @ hamk_C2 @ C3_matrix_2.conj().T) / 3

    @timing_decorator_factory(process_id=0)
    def gen_H_new(self, hamk, samk, gpu_index=0):
        """Generate new Hamiltonian from overlap matrix"""
        if not self.config.gpu:
            return self.gen_H_new_cpu(hamk, samk)
        else:
            return self.gen_H_new_gpu(hamk, samk, gpu_index)
    
    @timing_decorator_factory(process_id=0)
    def gen_H_new_cpu(self, hamk, samk):
        """CPU version of Hamiltonian transformation"""
        S_eig, S_vec = scipy.linalg.eigh(samk)
        M_inv = np.diag(1 / np.sqrt(S_eig))
        UMinvUd = S_vec @ M_inv @ S_vec.conj().T
        return UMinvUd @ hamk @ UMinvUd
    
    @timing_decorator_factory(process_id=0)
    def gen_H_new_gpu(self, hamk, samk, gpu_index=0):
        """GPU version of Hamiltonian transformation"""
        with cp.cuda.Device(gpu_index):
            samk_gpu = cp.asarray(samk)
            S_eig_gpu, S_vec_gpu = cp.linalg.eigh(samk_gpu)
            self.del_cupy_gpu(samk_gpu)
            
            M_inv_gpu = cp.diag(1 / cp.sqrt(S_eig_gpu))
            UMinvUd_gpu = S_vec_gpu @ M_inv_gpu @ S_vec_gpu.conj().T
            self.del_cupy_gpu(S_vec_gpu, S_eig_gpu, M_inv_gpu)

            hamk_gpu = cp.asarray(hamk)
            UH_gpu = UMinvUd_gpu @ hamk_gpu
            hamk_new_gpu = UH_gpu @ UMinvUd_gpu
            self.del_cupy_gpu(UMinvUd_gpu, UH_gpu)
            
            result = cp.asnumpy(hamk_new_gpu)
            self.del_cupy_gpu(hamk_gpu, hamk_new_gpu)
            return result

    def del_cupy_gpu(self, *args):
        """Delete CuPy GPU arrays and free memory"""
        for arg in args:
            del arg
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    @timing_decorator_factory(process_id=0)
    def cal_TAPW_hamiltonian_k(self, hamk):
        # if not self.config.gpu:
        #     return self.cal_TAPW_hamiltonian_k_cpu(hamk)
        # else:
        #     return self.cal_TAPW_hamiltonian_k_gpu(hamk)
        return self.cal_TAPW_hamiltonian_k_cpu(hamk)

    def cal_TAPW_hamiltonian_k_cpu(self, hamk):
        """CPU version of TAPW Hamiltonian calculation"""
        result = self.TAPW_parameters.g_matrix @ hamk @ self.TAPW_parameters.g_matrix_conj
        return result.toarray()
    
    @timing_decorator_factory(process_id=0)
    def Getk_super_gauge_sparse_symm_final_HS(self, Hr, Sr, symm_matrix, symm_matrix_inv, k, mpi_index):
        """Get final Hamiltonian for orthogonal or non-orthogonal basis (with symmetry)"""
        Hk_list = self.Getk_super_gauge_sparse_symm(Hr, k)
        if self.config.orthogonal_basis:
            if self.config.ge:
                HK_new = self.C3_symm(Hk_list[0], Hk_list[1], Hk_list[2], self.TAPW_parameters.C3_matrix)
                return HK_new, None
            else:
                Hk_new = self.C3_symm(Hk_list[0], Hk_list[1], Hk_list[2], self.TAPW_parameters.C3_matrix)
                return Hk_new, None
        else:
            Sk_list = self.Getk_super_gauge_sparse_symm(Sr, k)
            if self.config.ge:
                HK_new = self.C3_symm(Hk_list[0], Hk_list[1], Hk_list[2], self.TAPW_parameters.C3_matrix)
                Sk_new = self.C3_symm(Sk_list[0], Sk_list[1], Sk_list[2], self.TAPW_parameters.C3_matrix)
                return HK_new, Sk_new
            else:
                Hk_new_list = [self.gen_H_new(Hk, Sk, mpi_index) for Hk, Sk in zip(Hk_list, Sk_list)]
                Hk_new = self.C3_symm(Hk_new_list[0], Hk_new_list[1], Hk_new_list[2], self.TAPW_parameters.C3_matrix)
                return Hk_new, None

    @timing_decorator_factory(process_id=0)
    def calculate_band_01(self, kpoints, i):
        """Calculate band structure for a single k-point"""
        if self.config.TAPW:
            if self.config.C3_H:
                hamk, samk = self.Getk_super_gauge_sparse_symm_final_HS(
                    self.hr_supercell, self.sr_supercell, 
                    self.TAPW_parameters.symm_matrix, self.TAPW_parameters.symm_matrix_inv, 
                    kpoints[:3], self.config.gpu_index[i % self.config.gpu_num]
                )
            else:
                hamk, samk = self.Getk_super_gauge_sparse_final_HS(
                    self.hr_supercell, self.sr_supercell, kpoints[:3],
                    self.config.gpu_index[i % self.config.gpu_num]
                )
            # 正交基底下直接对角化Hk
            if self.config.orthogonal_basis:
                if self.config.eigsh_cal:
                    w = eigsh(hamk, k=self.config.num_bands_cal, sigma=self.config.efermi, 
                              which='LM', return_eigenvectors=self.config.eig_vec_cal)
                else:
                    w = scipy.linalg.eigh(hamk.toarray() if scipy.sparse.issparse(hamk) else hamk)
            else:
                if self.config.eigsh_cal:
                    if self.config.ge:
                        w = eigsh(hamk, k=self.config.num_bands_cal, M=samk, 
                                 sigma=self.config.efermi, which='LM', 
                                 return_eigenvectors=self.config.eig_vec_cal)
                    else:
                        w = eigsh(hamk, k=self.config.num_bands_cal, sigma=self.config.efermi, 
                                 which='LM', return_eigenvectors=self.config.eig_vec_cal)
                else:
                    w = lapack.zhegv(hamk, samk, itype=1, 
                                    jobz='V' if self.config.eig_vec_cal else 'N')
        else:
            if self.config.eigsh_cal:
                if self.config.ge:
                    hamk = self.Getk_super_gauge_sparse(self.hr_supercell, kpoints[:3], type = "H")
                    samk = self.Getk_super_gauge_sparse(self.sr_supercell, kpoints[:3], type = "S")
                    hamk = hamk.toarray()
                    samk = samk.toarray()
                    w = eigsh(hamk, k=self.config.num_bands_cal, M=samk, 
                             sigma=self.config.efermi, which='LM', 
                             return_eigenvectors=self.config.eig_vec_cal)
                else:
                    raise ValueError("Not implemented! Recommend to use generalized eigenvalue solver.")
        if self.config.eig_vec_cal:
            eig = np.sort(np.real(w[0]))
            vec = w[1][:, np.argsort(np.real(w[0]))]
            if not self.config.hamk_save:
                hamk, samk = 0, 0
        else:
            eig = np.sort(np.real(w))
            vec = 0
        return eig, vec, hamk, samk

    @timing_decorator_factory(process_id=0)
    def parallel_calculate_band_01(self, kpoints):
        """Calculate band structure for all k-points in parallel"""
        start_time = time.time()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sys.stdout.flush() 
        print(f"Current Time: {current_time}")
        print(f"Running with num_processes = {self.config.num_processes}")
                
        def print_time(stage):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"{stage} Current Time: {current_time}")
            sys.stdout.flush()

        def delayed_calculate_band_01(kpoint, delay):
            time.sleep(delay * self.config.delay_time)
            result = self.calculate_band_01(kpoint, delay)
            print(f"=============================     Kpoint {delay} {kpoint} finished    =============================")
            return result

        delays = [i for i in range(len(kpoints))]
        tasks = [delayed(delayed_calculate_band_01)(kpoint, delay) 
                for kpoint, delay in tqdm(zip(kpoints, delays))]
        result = Parallel(n_jobs=self.config.num_processes)(tasks)

        eig, vec, hamk, samk = zip(*result)

        self.result['eig'] = np.array(eig)
        self.result['vec'] = np.array(vec)
        self.result['hamk'] = hamk
        self.result['samk'] = samk

        end_time = time.time()
        print(f"Running time: {end_time - start_time:.2f} seconds")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Current Time: {current_time}")

    def generate_indices(self, num_gn_all, num_Te, num_Mo, orbs_num):
        """Generate indices for spin up/down components"""
        up_index = np.concatenate((np.arange(num_Te), np.arange(num_Te) + num_Te * 2, 
                                  np.arange(num_Mo) + num_Te * 4))
        down_index = np.concatenate((np.arange(num_Te) + num_Te, np.arange(num_Te) + num_Te * 3, 
                                    np.arange(num_Mo) + num_Te * 4 + num_Mo))

        up_all_index = np.concatenate([up_index + orbs_num * 2 * i for i in range(num_gn_all)])
        down_all_index = np.concatenate([down_index + orbs_num * 2 * i for i in range(num_gn_all)])

        return up_all_index.astype(int), down_all_index.astype(int)

    def write_wave_function_spin(self, path):
        """Write wave function for all k-points and each spin to files"""
        num_Te = self.structure.X_orb
        num_Mo = self.structure.M_orb
        n_g = self.config.n_g
        valley_flag = self.config.valley_flag
        
        g_vec_list_K_1layer = np.load(path + f"/g_vec_list_{n_g}_{valley_flag}_1layer.npy")
        g_vec_list_K_2layer = np.load(path + f"/g_vec_list_{n_g}_{valley_flag}_2layer.npy")

        orb_num = self.structure.num_orbs_per_unit_cell
        band_wave = self.result['vec']
        print(np.shape(band_wave))
        
        dir = os.path.join(path, f'{self.config.band_type}_{valley_flag}_valley')
        os.makedirs(dir, exist_ok=True)
        
        num_gn_all = len(g_vec_list_K_1layer) * 2
        up_all_index, down_all_index = self.generate_indices(num_gn_all, num_Te, num_Mo, orb_num)
        print(up_all_index, down_all_index, band_wave.shape)
        
        np.save(os.path.join(dir, f'{self.config.band_type}_{valley_flag}_valley_up.npy'), band_wave[:, up_all_index])
        np.save(os.path.join(dir, f'{self.config.band_type}_{valley_flag}_valley_down.npy'), band_wave[:, down_all_index])

    def write_hamk_spin(self, path):
        """Write Hamiltonian matrix for all k-points and each spin to files"""
        hamk = self.result['hamk']
        dim_Hprime = np.shape(hamk[0])[0]

        num_Te = self.structure.X_orb
        num_Mo = self.structure.M_orb
        n_g = self.config.n_g
        valley_flag = self.config.valley_flag
        orb_num = self.structure.num_orbs_per_unit_cell
        
        g_vec_list_K_1layer = np.load(path + f"/g_vec_list_{n_g}_{valley_flag}_1layer.npy")
        num_gn_all = len(g_vec_list_K_1layer) * 2 
        up_all_index, down_all_index = self.generate_indices(num_gn_all, num_Te, num_Mo, orb_num)
        num_kpoints = len(self.kpath_config.kpoints)
        
        H_spin_kpoints = np.zeros((2, num_kpoints, int(dim_Hprime/2), int(dim_Hprime/2)), 
                                 dtype=np.complex64)
        
        for i in tqdm(range(num_kpoints)):
            gamma_hamk = hamk[i]
            H_gamma_up = gamma_hamk[up_all_index][:, up_all_index]
            H_gamma_down = gamma_hamk[down_all_index][:, down_all_index]
            
            H_spin_kpoints[0, i] = H_gamma_up
            H_spin_kpoints[1, i] = H_gamma_down

        os.makedirs(os.path.join(path, 'symm_Hprime_wave_npy'), exist_ok=True)
        np.save(os.path.join(path, 'symm_Hprime_wave_npy', f'Hprime_up_down_{self.config.band_type}_{self.config.valley_flag}.npy'), 
                H_spin_kpoints)

    def write_wave_2col(self, path, vec, g_vec_list_1layer, g_vec_list_2layer, orb_Te, orb_Mo):
        """Write wave function to 2-column format"""
        num_wann = len(vec)
        num_wann_perlayer = int(num_wann/2)
        
        with open(path, "w") as f:
            arr1 = np.arange(1, orb_Te+1)
            arr2 = np.arange(1, orb_Mo+1)

            orb_index_num = np.concatenate((arr1, arr1, arr1, arr1, arr2, arr2))
            orb_spin_index = np.concatenate((['up']*orb_Te, ['down']*orb_Te, ['up']*orb_Te, 
                                           ['down']*orb_Te, ['up']*orb_Mo, ['down']*orb_Mo))
            atoms_index = np.concatenate((['Te2']*orb_Te*2, ['Te1']*orb_Te*2, ['Mo']*orb_Mo*2))
            orb_all = orb_Te*4 + orb_Mo*2
            
            f.write("#layer  gvec     gvec_x     gvec_y  atom  orb  spin     real       imag\n")
            
            for i in range(num_wann):
                if i < num_wann_perlayer:
                    g_vec_index = i // orb_all
                    g_vec = g_vec_list_1layer[g_vec_index]
                    layer = 1
                else:
                    g_vec_index = (i - int(num_wann_perlayer)) // orb_all
                    g_vec = g_vec_list_2layer[g_vec_index]
                    layer = 2
                
                f.write(f"{layer:>5d} {g_vec_index+1:>5d} {g_vec[0]:>12.6f} {g_vec[1]:>10.6f} "
                       f"{atoms_index[i%orb_all]:>4s} {orb_index_num[i%orb_all]:>4d} "
                       f"{orb_spin_index[i%orb_all]:>5s} {vec[i].real:>10.6f} {vec[i].imag:>10.6f}\n")