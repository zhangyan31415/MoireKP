"""
TAPW Slab Calculator
===================
This module implements slab calculations within the TAPW (Twisted Atomic Plane Wave) framework.
It adapts the standard slab construction method to work with TAPW transformations.

The approach follows these steps:
1. Calculate interlayer hopping matrices Hij(ic) from real-space H(R), S(R)
2. Apply TAPW transformation to get effective interlayer matrices
3. Construct slab Hamiltonian by assembling layers with open boundaries
4. Solve eigenvalue problem and analyze surface states

Key differences from standard slab method:
- TAPW transformation is applied to both intra-layer and inter-layer terms
- G-vector basis is used instead of Wannier basis
- C3 symmetry can be enforced if enabled
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import time

from .cal_ham_01 import BandStructureCalculator
from .utils import timing_decorator_factory
from .read_kpath_01 import KPathGenerator


class TAPWSlab(BandStructureCalculator):
    """TAPW Slab calculator extending BandStructureCalculator"""
    
    def __init__(self, hr_supercell, sr_supercell, structure, config, kpath_config=None):
        """Initialize TAPW slab calculator
        
        Args:
            hr_supercell: Hamiltonian matrix in real space
            sr_supercell: Overlap matrix in real space  
            structure: Processed structure information
            config: Config object with slab configuration
            kpath_config: K-path configuration for ribbon bands
        """
        super().__init__(hr_supercell, sr_supercell, structure, config.compute, kpath_config)
        
        # Get slab configuration
        if hasattr(config, 'slab') and config.slab is not None:
            slab_config = config.slab
        else:
            # Fallback to default values if no slab config
            from .config import SlabConfig
            slab_config = SlabConfig()
        
        # Slab-specific parameters from configuration
        self.nslab = slab_config.nslab
        self.ijmax = slab_config.ijmax
        self.slab_direction = slab_config.slab_direction  # [x, y, z] vector
        self.direction_index = slab_config.direction_index  # 0=x, 1=y, 2=z
        self.direction_name = slab_config.direction_name  # 'x', 'y', or 'z'
        self.analyze_surface = slab_config.analyze_surface
        self.surface_threshold = slab_config.surface_threshold
        self.kpath_slab_in = slab_config.kpath_slab_in
        self.kpath_slab_out = slab_config.kpath_slab_out
        
        # Results storage
        self.slab_results = {}
        
        self.nwnn = np.max(hr_supercell[(0,0,0)]['row'])+1
        print(f"nwnn = {self.nwnn}")
        
        # if not self.config.TAPW:
        #     raise ValueError("TAPW must be enabled for slab calculations")
            
    def calculate_interlayer_hopping_tapw(self, k_parallel, Hr, Sr=None):
        """
        Calculate interlayer hopping matrices with TAPW transformation for slab
        
        For configurable slab direction:
        - k_parallel: 2D k-point in directions parallel to slab (periodic directions)  
        - slab_direction: cut direction (no Fourier transform)
        
        Args:
            k_parallel: 2D k-point parallel to slab
            Hr: Real-space Hamiltonian matrices dict
            Sr: Real-space overlap matrices dict (optional)
            
        Returns:
            Hij_tapw: dict mapping ic -> TAPW-transformed interlayer hopping matrix
            Sij_tapw: dict mapping ic -> TAPW-transformed interlayer overlap matrix (if Sr provided)
        """
        Hij_tapw = {}
        Sij_tapw = {} if Sr is not None else None
        
        # For twisted materials, only nearest layer coupling (ijmax=1)
        # Construct 3D k-vector based on slab direction
        # k_parallel contains k-components in the two directions parallel to slab
        k_3d = [0.0, 0.0, 0.0]
        parallel_indices = [i for i in range(3) if i != self.direction_index]
        
        # Fill in parallel k-components
        k_3d[parallel_indices[0]] = k_parallel[0]
        k_3d[parallel_indices[1]] = k_parallel[1]
        # Cut direction component remains 0
        
        kvec = self.get_kvec(np.array(k_3d))
        print(f"Slab direction: {self.direction_name}, k_parallel: {k_parallel}, k_3d: {k_3d}")
                
        # Get sorted wannier positions for layer identification (only needed for TAPW)
        if self.config.TAPW:
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
        else:
             num_wann = self.nwnn
        
        
        # Choose matrix dimension based on whether TAPW is used
        if self.config.TAPW:
            matrix_dim = self.TAPW_parameters.g_matrix.shape[0]  # TAPW transformed dimension
            print(f"Using TAPW: matrix dimension = {matrix_dim}")
            # TAPW matrices are typically dense and smaller
            for ic in range(-self.ijmax, self.ijmax + 1):
                Hij_tapw[ic] = np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)
                if Sr is not None:
                    Sij_tapw[ic] = np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)
        else:
            matrix_dim = num_wann
            print(f"Not using TAPW: matrix dimension = {matrix_dim}")
            # Wannier matrices are typically sparse and larger
            for ic in range(-self.ijmax, self.ijmax + 1):
                Hij_tapw[ic] = scipy.sparse.csr_matrix((matrix_dim, matrix_dim), dtype=np.complex128)
                if Sr is not None:
                    Sij_tapw[ic] = scipy.sparse.csr_matrix((matrix_dim, matrix_dim), dtype=np.complex128)
        
        
        
        # Build interlayer coupling matrices directly from R-space
        # Separate by R-component in slab direction (layer separation)
        interlayer_matrices = {}
        for ic in range(-self.ijmax, self.ijmax + 1):
            interlayer_matrices[ic] = {'H': scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)}
            if Sr is not None:
                interlayer_matrices[ic]['S'] = scipy.sparse.csr_matrix((num_wann, num_wann), dtype=np.complex128)
        
        # Sum over R vectors, separating by R-component in slab direction
        for rvec, values_dic in Hr.items():
            # Get interlayer separation in slab direction
            ic = rvec[self.direction_index]  # R-component in slab direction
            
            if abs(ic) <= self.ijmax:
                row_index, col_index, val_index = values_dic["row"], values_dic["col"], values_dic["val"]

                if self.config.TAPW:
                    m_coor, n_coor = sorted_wann[row_index], sorted_wann[col_index]
                    # Calculate position difference only in directions parallel to slab
                    parallel_pos_diff = np.zeros((len(row_index), 3))
                    for i, idx in enumerate(parallel_indices):
                        parallel_pos_diff[:, idx] = m_coor[:, idx] - n_coor[:, idx]
                else:
                    parallel_pos_diff = np.zeros((len(row_index), 3))
                    
                Rvec = np.dot(rvec, self.structure.Tmat)
                
                # Only do Fourier transform in directions parallel to slab
                # Phase from position difference and R-vector in parallel directions only
                parallel_kvec = np.zeros(3)
                parallel_Rvec = np.zeros(3)
                for idx in parallel_indices:
                    parallel_kvec[idx] = kvec[idx]
                    parallel_Rvec[idx] = Rvec[idx]
                
                
                phase_factor = (np.exp(-1j * np.dot(parallel_pos_diff, parallel_kvec)) * 
                               np.exp(1j * np.dot(parallel_kvec, parallel_Rvec)))
                
                interlayer_matrices[ic]['H'] += scipy.sparse.csr_matrix(
                    (val_index * phase_factor, (row_index, col_index)), shape=(num_wann, num_wann))
        
        if Sr is not None:
            for rvec, values_dic in Sr.items():
                # Get interlayer separation in slab direction
                ic = rvec[self.direction_index]  # R-component in slab direction
                
                if abs(ic) <= self.ijmax:
                    row_index, col_index, val_index = values_dic["row"], values_dic["col"], values_dic["val"]
                    if self.config.TAPW:
                        m_coor, n_coor = sorted_wann[row_index], sorted_wann[col_index]
                        # Calculate position difference only in directions parallel to slab
                        parallel_pos_diff = np.zeros((len(row_index), 3))
                        for i, idx in enumerate(parallel_indices):
                            parallel_pos_diff[:, idx] = m_coor[:, idx] - n_coor[:, idx]
                    else:
                        parallel_pos_diff = np.zeros((len(row_index), 3))
                        
                    Rvec = np.dot(rvec, self.structure.Tmat)
                    
                    # Only do Fourier transform in directions parallel to slab
                    parallel_kvec = np.zeros(3)
                    parallel_Rvec = np.zeros(3)
                    for idx in parallel_indices:
                        parallel_kvec[idx] = kvec[idx]
                        parallel_Rvec[idx] = Rvec[idx]
                    
                    phase_factor = (np.exp(-1j * np.dot(parallel_pos_diff, parallel_kvec)) * 
                                   np.exp(1j * np.dot(parallel_kvec, parallel_Rvec)))
                    
                    interlayer_matrices[ic]['S'] += scipy.sparse.csr_matrix(
                        (val_index * phase_factor, (row_index, col_index)), shape=(num_wann, num_wann))
        
        # Apply TAPW transformation to each interlayer matrix (if TAPW is enabled)
        for ic in range(-self.ijmax, self.ijmax + 1):
            if ic in interlayer_matrices:
                if self.config.TAPW:
                    # Apply TAPW transformation
                    Hij_tapw[ic] = self.cal_TAPW_hamiltonian_k(interlayer_matrices[ic]['H'])
                    if Sr is not None and 'S' in interlayer_matrices[ic]:
                        Sij_tapw[ic] = self.cal_TAPW_hamiltonian_k(interlayer_matrices[ic]['S'])
                else:
                    # Use original sparse matrices without TAPW transformation (keep sparse for efficiency)
                    Hij_tapw[ic] = interlayer_matrices[ic]['H']  # Keep as sparse matrix
                    if Sr is not None and 'S' in interlayer_matrices[ic]:
                        Sij_tapw[ic] = interlayer_matrices[ic]['S']  # Keep as sparse matrix
        
        return Hij_tapw, Sij_tapw
    
    def _create_default_slab_kpath(self, output_file):
        """Create default KPATH_SLAB.in file based on slab direction"""
        # Define k-path in the plane perpendicular to slab direction
        direction_names = ['x', 'y', 'z']
        parallel_dirs = [d for i, d in enumerate(direction_names) if i != self.direction_index]
        
        default_content = f"""K-Path for Slab calculation ({self.direction_name}-direction cut).
   40
Line-Mode
Reciprocal
   0.0000000000   0.0000000000   0.0000000000     Gamma
   0.5000000000   0.0000000000   0.0000000000     {parallel_dirs[0].upper()}

   0.5000000000   0.0000000000   0.0000000000     {parallel_dirs[0].upper()}
   0.5000000000   0.0000000000   0.5000000000     M      

   0.5000000000   0.0000000000   0.5000000000     M 
   0.0000000000   0.0000000000   0.0000000000     Gamma
"""
        with open(output_file, 'w') as f:
            f.write(default_content)
        print(f"Created default slab k-path: {output_file}")
        print(f"Slab k-path: Gamma -> {parallel_dirs[0].upper()} -> M -> Gamma ({self.direction_name}-direction cut, {parallel_dirs[0]}-{parallel_dirs[1]} plane)")
    

        
    def construct_tapw_slab_hamiltonian(self, Hij_tapw, Sij_tapw=None):
        """
        Construct TAPW slab Hamiltonian from interlayer hopping matrices
        
        Args:
            Hij_tapw: dict mapping ic -> TAPW interlayer hopping matrix
            Sij_tapw: dict mapping ic -> TAPW interlayer overlap matrix (optional)
            
        Returns:
            Hamk_slab: TAPW slab Hamiltonian
            Sk_slab: TAPW slab overlap matrix (if provided)
        """
        # Get dimensions
        first_matrix = list(Hij_tapw.values())[0]
        dim_per_layer = first_matrix.shape[0]
        total_dim = self.nslab * dim_per_layer
        
        # Check if we're working with sparse matrices (non-TAPW case)
        use_sparse = scipy.sparse.issparse(first_matrix)
        
        if use_sparse:
            print(f"Using sparse matrices for slab construction (total dimension: {total_dim})")
            memory_estimate = total_dim * total_dim * 16 / (1024**3)  # 16 bytes per complex128, convert to GB
            print(f"Estimated memory for dense matrix: {memory_estimate:.2f} GB (using sparse instead)")
            
            # Build sparse slab matrix using block matrix construction
            H_blocks = []
            S_blocks = [] if Sij_tapw is not None else None
            
            for i2 in range(1, self.nslab + 1):  # row index
                H_row = []
                S_row = [] if Sij_tapw is not None else None
                
                for i1 in range(1, self.nslab + 1):  # column index
                    ic = i1 - i2
                    if abs(ic) <= self.ijmax and ic in Hij_tapw:
                        H_row.append(Hij_tapw[ic])
                        if Sij_tapw is not None:
                            S_row.append(Sij_tapw[ic])
                    else:
                        # Zero block
                        zero_block = scipy.sparse.csr_matrix((dim_per_layer, dim_per_layer), dtype=np.complex128)
                        H_row.append(zero_block)
                        if Sij_tapw is not None:
                            S_row.append(zero_block)
                
                H_blocks.append(H_row)
                if Sij_tapw is not None:
                    S_blocks.append(S_row)
            
            Hamk_slab = scipy.sparse.bmat(H_blocks, format='csr')
            Sk_slab = scipy.sparse.bmat(S_blocks, format='csr') if Sij_tapw is not None else None
            
        else:
            print(f"Using dense matrices for slab construction (total dimension: {total_dim})")
            memory_estimate = total_dim * total_dim * 16 / (1024**3)  # 16 bytes per complex128, convert to GB
            print(f"Estimated memory for dense matrix: {memory_estimate:.2f} GB")
            # Initialize dense slab matrices (TAPW case)
            Hamk_slab = np.zeros((total_dim, total_dim), dtype=np.complex128)
            Sk_slab = np.zeros((total_dim, total_dim), dtype=np.complex128) if Sij_tapw is not None else None
            
            # Fill dense slab matrices following standard slab construction
            for i1 in range(1, self.nslab + 1):  # column index (1-based)
                for i2 in range(1, self.nslab + 1):  # row index (1-based)
                    if abs(i2 - i1) <= self.ijmax:
                        # Convert to 0-based Python indices
                        row_start = (i2 - 1) * dim_per_layer
                        row_end = i2 * dim_per_layer
                        col_start = (i1 - 1) * dim_per_layer
                        col_end = i1 * dim_per_layer
                        
                        # Hij_tapw(i1-i2) gives hopping from layer i2 to layer i1
                        ic = i1 - i2
                        if ic in Hij_tapw:
                            Hamk_slab[row_start:row_end, col_start:col_end] = Hij_tapw[ic]
                            if Sij_tapw is not None:
                                Sk_slab[row_start:row_end, col_start:col_end] = Sij_tapw[ic]
        
        return Hamk_slab, Sk_slab
        
    def solve_tapw_slab_bands(self, Hamk_slab, Sk_slab=None, return_vectors=False):
        """Solve slab eigenvalue problem (TAPW or Wannier)"""
        is_sparse = scipy.sparse.issparse(Hamk_slab)
        # is_sparse = False
        print("is sparse: ", is_sparse)
        # For sparse matrices or when eigsh is explicitly requested
        if (is_sparse and (self.config.eigsh_cal and hasattr(self.config, 'num_bands_cal'))) and self.config.num_bands_cal < Hamk_slab.shape[0]:
            
            # num_bands = getattr(self.config, 'num_bands_cal', min(50, Hamk_slab.shape[0] - 1))
            num_bands = self.config.num_bands_cal
            print(f"Using eigsh to calculate {num_bands} bands of total {Hamk_slab.shape[0]} bands near Fermi level ({self.config.efermi} eV)")
            print(f"Matrix type: {'sparse' if is_sparse else 'dense'}")
            
            if Sk_slab is not None:
                # Generalized eigenvalue problem H ψ = λ S ψ
                if return_vectors:
                    vals, vecs = eigsh(Hamk_slab, k=num_bands, M=Sk_slab,
                                     sigma=self.config.efermi, which='LM', return_eigenvectors=True)
                    return np.real(vals), np.array(vecs, dtype=np.complex128)
                else:
                    
                    from scipy.sparse.linalg import LinearOperator, gmres
                    
                    Ashift = Hamk_slab - self.config.efermi * Sk_slab

                    def OPinv_mv(b):
                        # Use GMRES without preconditioner for better stability
                        x, info = gmres(Ashift, b, rtol=1e-6, maxiter=500)
                        if info != 0:
                            print(f"Warning: GMRES did not converge, info={info}")
                        return x

                    OPinv = LinearOperator(Ashift.shape, matvec=OPinv_mv, dtype=Ashift.dtype)

                    vals = eigsh(Hamk_slab, k=min(num_bands, 300), M=Sk_slab,
                                sigma=self.config.efermi, which='LM',
                                OPinv=OPinv, ncv=min(Hamk_slab.shape[0]-1, min(num_bands,300)+40),
                                return_eigenvectors=False)

                    # vals = eigsh(Hamk_slab, k=num_bands, M=Sk_slab,
                    #            sigma=self.config.efermi, which='LM', return_eigenvectors=False)
                    return np.real(vals)
            else:
                # Standard eigenvalue problem H ψ = λ ψ
                if return_vectors:
                    vals, vecs = eigsh(Hamk_slab, k=num_bands,
                                     sigma=self.config.efermi, which='LM', return_eigenvectors=True)
                    return np.real(vals), np.array(vecs, dtype=np.complex128)
                else:
                    vals = eigsh(Hamk_slab, k=num_bands,
                               sigma=self.config.efermi, which='LM', return_eigenvectors=False)
                    return np.real(vals)
        else:
            # Use full diagonalization for dense matrices
            print("Using full matrix diagonalization")
            print(f"Matrix type: dense")
            
            # Convert sparse to dense if needed for full diagonalization
            if type(Hamk_slab) == scipy.sparse.csr_matrix:
                Hamk_slab = Hamk_slab.toarray()
                if Sk_slab is not None:
                    Sk_slab = Sk_slab.toarray()
            
            if Sk_slab is not None:
                # Generalized eigenvalue problem H ψ = λ S ψ
                if return_vectors:
                    vals, vecs = la.eigh(Hamk_slab, Sk_slab)
                    return np.real(vals), np.array(vecs, dtype=np.complex128)
                else:
                    vals = la.eigh(Hamk_slab, Sk_slab, eigvals_only=True)
                    return np.real(vals)
            else:
                # Standard eigenvalue problem H ψ = λ ψ
                if return_vectors:
                    vals, vecs = la.eigh(Hamk_slab)
                    return np.real(vals), np.array(vecs, dtype=np.complex128)
                else:
                    vals = la.eigh(Hamk_slab, eigvals_only=True)
                    return np.real(vals)
                
    def analyze_slab_surface_character(self, eigenvectors):
        """
        Analyze surface character of slab eigenstates
        
        Args:
            eigenvectors: (nslab*dim_per_layer, nslab*dim_per_layer) eigenvector matrix
            
        Returns:
            surface_data: dict with surface character analysis
        """
        total_dim = eigenvectors.shape[0]
        dim_per_layer = total_dim // self.nslab
        
        surface_data = {
            'total': np.zeros(total_dim),
            'top': np.zeros(total_dim),
            'bottom': np.zeros(total_dim),
            'bulk': np.zeros(total_dim),
            'surface_index': np.zeros(total_dim)
        }
        
        for i in range(total_dim):
            vec = eigenvectors[:, i]
            
            # Calculate weights on different layers
            layer_weights = np.zeros(self.nslab)
            for layer in range(self.nslab):
                start_idx = layer * dim_per_layer
                end_idx = (layer + 1) * dim_per_layer
                layer_weights[layer] = np.sum(np.abs(vec[start_idx:end_idx])**2)
            
            # Surface character analysis
            surface_data['top'][i] = layer_weights[0]
            surface_data['bottom'][i] = layer_weights[-1]
            surface_data['total'][i] = layer_weights[0] + layer_weights[-1]
            
            if self.nslab > 2:
                surface_data['bulk'][i] = np.sum(layer_weights[1:-1])
            else:
                surface_data['bulk'][i] = 0
                
            # Surface index: +1 for top, -1 for bottom, 0 for bulk
            top_weight = layer_weights[0]
            bottom_weight = layer_weights[-1]
            
            if top_weight + bottom_weight > 0:
                surface_data['surface_index'][i] = (top_weight - bottom_weight) / (top_weight + bottom_weight)
            else:
                surface_data['surface_index'][i] = 0.0
        
        return surface_data
        
    @timing_decorator_factory(process_id=0)
    def calculate_tapw_slab_bands(self, klist):
        """
        Calculate TAPW slab band structure
        
        Args:
            klist: array of 2D k-points for ribbon calculation
            
        Returns:
            bands: (nk, nslab*tapw_dim) array of eigenvalues
            surface_weights: surface character data (if analyze_surface=True)
        """
        bands = []
        surface_weights = {
            'total': [],
            'top': [],
            'bottom': [],
            'bulk': [],
            'surface_index': []
        } if self.analyze_surface else None
        
        # Calculate dimensions based on whether TAPW is used
        if self.config.TAPW:
            matrix_dim_per_layer = self.TAPW_parameters.g_matrix.shape[0] // len(self.structure.df['layer'].unique())
            method_name = "TAPW"
        else:

            
            num_wann = self.nwnn
            
            matrix_dim_per_layer = num_wann
            method_name = "Wannier"
            
        total_slab_dim = self.nslab * matrix_dim_per_layer
        
        print(f"Calculating {method_name} slab with {self.nslab} layers")
        print(f"{method_name} dimension per layer: {matrix_dim_per_layer}")
        print(f"Total slab dimension: {total_slab_dim}")
        
        if (self.config.eigsh_cal and hasattr(self.config, 'num_bands_cal') and 
            self.config.num_bands_cal < total_slab_dim):
            print(f"Will calculate {self.config.num_bands_cal} bands near Fermi level ({self.config.efermi} eV)")
        else:
            print(f"Will calculate all {total_slab_dim} bands")
        
        # Determine number of processes
        num_processes = getattr(self.config, 'num_processes', 1)
        # Limit parallel processes for sparse matrix calculations to avoid memory issues
        # if not self.config.TAPW and num_processes > 4:
        #     print(f"Limiting parallel processes from {num_processes} to 4 for sparse matrix calculations")
        #     num_processes = 4
            
        if num_processes > 1:
            print(f"Using parallel calculation with {num_processes} processes")
            
            # Define function for single k-point calculation
            def calculate_single_kpoint(k_point):
                try:
                    # Extract k-components parallel to slab (skip slab direction component)
                    if len(k_point) == 3:
                        parallel_indices = [i for i in range(3) if i != self.direction_index]
                        k_parallel = [k_point[parallel_indices[0]], k_point[parallel_indices[1]]]
                    else:
                        k_parallel = k_point  # assume already 2D parallel components
                    
                    # Calculate TAPW interlayer hopping matrices
                    Hij_tapw, Sij_tapw = self.calculate_interlayer_hopping_tapw(
                        k_parallel, self.hr_supercell, self.sr_supercell if not self.config.orthogonal_basis else None
                    )
                    
                    # Construct TAPW slab Hamiltonian
                    Hamk_slab, Sk_slab = self.construct_tapw_slab_hamiltonian(Hij_tapw, Sij_tapw)
                    
                    # Solve eigenvalue problem
                    if self.analyze_surface:
                        vals, vecs = self.solve_tapw_slab_bands(Hamk_slab, Sk_slab, return_vectors=True)
                        # Analyze surface character
                        surf_data = self.analyze_slab_surface_character(vecs)
                        # Convert to numpy arrays to ensure picklability
                        return np.array(vals, dtype=np.float64), {k: np.array(v, dtype=np.float64) for k, v in surf_data.items()}
                    else:
                        vals = self.solve_tapw_slab_bands(Hamk_slab, Sk_slab)
                        # Convert to numpy array to ensure picklability
                        return np.array(vals, dtype=np.float64), None
                except Exception as e:
                    print(f"Error in k-point calculation: {e}")
                    # Return NaN arrays to maintain consistency
                    if self.analyze_surface:
                        nan_vals = np.full(self.config.num_bands_cal, np.nan, dtype=np.float64)
                        nan_surf = {'total': np.full(self.config.num_bands_cal, np.nan), 
                                   'top': np.full(self.config.num_bands_cal, np.nan),
                                   'bottom': np.full(self.config.num_bands_cal, np.nan),
                                   'bulk': np.full(self.config.num_bands_cal, np.nan),
                                   'surface_index': np.full(self.config.num_bands_cal, np.nan)}
                        return nan_vals, nan_surf
                    else:
                        nan_vals = np.full(self.config.num_bands_cal, np.nan, dtype=np.float64)
                        return nan_vals, None
            
            # Parallel calculation with better error handling
            try:
                results = Parallel(n_jobs=num_processes, prefer="processes")(
                    delayed(calculate_single_kpoint)(k_point) for k_point in tqdm(klist, desc="Calculating k-points")
                )
            except Exception as e:
                print(f"Parallel calculation failed: {e}")
                print("Falling back to serial calculation...")
                results = []
                for k_point in tqdm(klist, desc="Calculating k-points (serial)"):
                    results.append(calculate_single_kpoint(k_point))
            
            # Extract results
            for vals, surf_data in results:
                bands.append(vals)
                if self.analyze_surface and surf_data is not None:
                    for key in surface_weights:
                        surface_weights[key].append(surf_data[key])
                        
        else:
            print("Using serial calculation")
            # Serial calculation (original code)
            for i, k_point in enumerate(tqdm(klist, desc="Calculating k-points")):
                # Extract k-components parallel to slab (skip slab direction component)
                if len(k_point) == 3:
                    parallel_indices = [i for i in range(3) if i != self.direction_index]
                    k_parallel = [k_point[parallel_indices[0]], k_point[parallel_indices[1]]]
                else:
                    k_parallel = k_point  # assume already 2D parallel components
                
                # Calculate TAPW interlayer hopping matrices
                Hij_tapw, Sij_tapw = self.calculate_interlayer_hopping_tapw(
                    k_parallel, self.hr_supercell, self.sr_supercell if not self.config.orthogonal_basis else None
                )
                
                # Construct TAPW slab Hamiltonian
                Hamk_slab, Sk_slab = self.construct_tapw_slab_hamiltonian(Hij_tapw, Sij_tapw)
                
                # Solve eigenvalue problem
                if self.analyze_surface:
                    vals, vecs = self.solve_tapw_slab_bands(Hamk_slab, Sk_slab, return_vectors=True)
                    bands.append(vals)
                    
                    # Analyze surface character
                    surf_data = self.analyze_slab_surface_character(vecs)
                    for key in surface_weights:
                        surface_weights[key].append(surf_data[key])
                else:
                    vals = self.solve_tapw_slab_bands(Hamk_slab, Sk_slab)
                    bands.append(vals)
        
        bands_array = np.array(bands)
        print(f"Calculated band structure: {bands_array.shape[0]} k-points, {bands_array.shape[1]} bands")
        
        if self.analyze_surface:
            # Convert to numpy arrays
            for key in surface_weights:
                surface_weights[key] = np.array(surface_weights[key])
            return bands_array, surface_weights
        else:
            return bands_array
            
    def calculate_ribbon_bands(self, path, slab_kpoints=None):
        """
        Calculate ribbon band structure (slab with periodic boundary in parallel directions)
        
        Args:
            path: Output path for results
            slab_kpoints: Array of slab k-points. If None, uses slab k-path configuration
        """
        # Create output directories
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "slab_data"), exist_ok=True)
        
        # Use provided k-points or read from slab k-path file
        if slab_kpoints is None:
            # Check if slab k-path file exists
            if not os.path.exists(self.kpath_slab_in):
                print(f"Slab k-path file not found: {self.kpath_slab_in}")
                print("Creating default slab k-path...")
                # self._create_default_slab_kpath(self.kpath_slab_in)
                raise ValueError("Slab k-path file not found. Please create a default slab k-path file.")
            
            # Use existing KPathGenerator to read slab k-path
            
            slab_kpath_gen = KPathGenerator(self.structure.Tmat)
            slab_kpath_gen.read_and_generate_kpath(self.kpath_slab_in, self.kpath_slab_out)
            
            # Extract k-points (first 3 columns, ignore the 4th distance column)
            slab_kpoints = np.array([kp[:3] for kp in slab_kpath_gen.kpoints])
            
            # Store ticks for plotting
            self.slab_ticks = dict(zip(slab_kpath_gen.labels_ticks, slab_kpath_gen.x_ticks))
        
        print(f"Using {len(slab_kpoints)} k-points for slab calculation")
        
        # Calculate slab bands
        if self.analyze_surface:
            bands, surface_weights = self.calculate_tapw_slab_bands(slab_kpoints)
            self.slab_results['surface_weights'] = surface_weights
            
            # Analyze surface states
            total_surf = surface_weights['total']
            avg_surface_char = np.mean(total_surf, axis=0)
            n_surface_states = np.sum(avg_surface_char > self.surface_threshold)
            
            print(f"Surface states identified: {n_surface_states} bands")
        else:
            bands = self.calculate_tapw_slab_bands(slab_kpoints)
        
        self.slab_results['bands'] = bands
        self.slab_results['kpoints'] = slab_kpoints
        
        # Save results
        suffix = f"_{self.nslab}layers_{self.valley_flag}"
        np.savetxt(os.path.join(path, f"slab_data/slab_bands{suffix}.txt"),
                  bands, fmt='%15.11f')
        
        if self.analyze_surface:
            np.save(os.path.join(path, f"slab_data/surface_weights{suffix}"), surface_weights)
        
        # Save TAPW parameters
        if self.config.TAPW:
            np.save(os.path.join(path, f"slab_data/g_vec_list_{self.config.n_g}_{suffix}_1layer"),
                   self.TAPW_parameters.g_vec_list_K1)
            np.save(os.path.join(path, f"slab_data/g_vec_list_{self.config.n_g}_{suffix}_2layer"),
                   self.TAPW_parameters.g_vec_list_K2)
        
        print(f"TAPW slab calculation completed!")
        print(f"Results saved to {path}/slab_data/")
        
    def plot_ribbon_bands(self, energy_range=(-2, 2), outfile=None, show_surface=True):
        """
        Plot ribbon band structure with surface state analysis
        
        Args:
            energy_range: Energy range for plotting [min, max] in eV
            outfile: Output file path for saving plot
            show_surface: Whether to highlight surface states
        """
        if 'bands' not in self.slab_results:
            raise ValueError("No slab bands calculated. Run calculate_ribbon_bands first.")
            
        bands = self.slab_results['bands']
        kpoints = self.slab_results['kpoints']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(kpoints))
        
        # Plot bands
        for b in range(bands.shape[1]):
            band = bands[:, b]
            # Only plot if band has values in the energy range
            if np.any((band >= energy_range[0]) & (band <= energy_range[1])):
                
                if show_surface and self.analyze_surface and 'surface_weights' in self.slab_results:
                    # Color by surface character
                    surface_weights = self.slab_results['surface_weights']
                    total_surf = surface_weights['total'][:, b]
                    surface_index = surface_weights['surface_index'][:, b]
                    
                    avg_total = np.mean(total_surf)
                    
                    if avg_total > self.surface_threshold:
                        # Surface state - use color mapping
                        scatter = ax.scatter(x, band, c=surface_index, cmap='RdBu_r', 
                                           s=15, alpha=0.8, vmin=-1, vmax=1)
                    else:
                        # Bulk state
                        ax.plot(x, band, 'k-', alpha=0.3, linewidth=0.5)
                else:
                    # Standard plotting
                    ax.plot(x, band, 'k-', alpha=0.7, linewidth=0.8)
        
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(energy_range)
        ax.set_ylabel('Energy (eV)')
        ax.set_title(f'TAPW Slab Band Structure ({self.nslab} layers, {self.valley_flag})')
        ax.grid(True, alpha=0.3)
        
        # Add high-symmetry point labels and vertical lines
        if hasattr(self, 'slab_ticks'):
            # Convert x_tick positions to k-point indices
            tick_positions = []
            tick_labels = []
            for label, x_pos in self.slab_ticks.items():
                # Find closest k-point index to this x position
                closest_idx = int(x_pos * len(kpoints) / max(self.slab_ticks.values()))
                tick_positions.append(closest_idx)
                tick_labels.append(label)
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            
            # Add vertical lines at high-symmetry points
            for tick_pos in tick_positions:
                ax.axvline(x=tick_pos, color='black', linestyle='-', alpha=0.3)
        
        # Add Fermi level
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add colorbar for surface states if applicable
        if (show_surface and self.analyze_surface and 'surface_weights' in self.slab_results):
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Surface Index', rotation=270, labelpad=20)
            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
            cbar.set_ticklabels(['Bottom\n(-1)', '-0.5', 'Bulk\n(0)', '0.5', 'Top\n(+1)'])
        
        plt.tight_layout()
        
        if outfile:
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {outfile}")
        
        return fig, ax 