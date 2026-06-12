"""
Correct Slab Band Structure Calculator for Bi2Se3
-------------------------------------------------
This module implements the correct slab construction method as used in WannierTools.

The slab Hamiltonian is constructed by:
1. First calculating interlayer hopping matrices Hij(ic, :, :) for different layer separations ic
2. Then assembling these into a large (nslab*Num_wann) × (nslab*Num_wann) matrix
3. Each layer has Num_wann orbitals

This follows the same approach as WannierTools' ham_slab subroutine.

Dependencies: numpy, scipy, matplotlib
"""
import argparse
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

PI2 = 2.0 * np.pi

def read_sparse_tb(path):
    """Return (orbitals, dict{R: matrix}) where R is (Rx,Ry,Rz) ints."""
    with open(path, "r") as f:
        # skip comment line(s) until we reach first numeric token
        line = f.readline()
        while line.strip().startswith("!") or not line.strip():
            line = f.readline()
        # now line has total number of non-zeros
        n_nonzero = int(line.split()[0])
        norb = int(f.readline().split()[0])
        nR = int(f.readline().split()[0])  # unused explicitly
        # container: dict maps R-tuple to dense matrix
        mats = {}
        for _ in range(n_nonzero):
            elems = f.readline().split()
            Rx, Ry, Rz = map(int, elems[:3])
            i, j = int(elems[3]) - 1, int(elems[4]) - 1  # 1-based → 0-based
            Re, Im = map(float, elems[5:7])
            R_key = (Rx, Ry, Rz)
            if R_key not in mats:
                mats[R_key] = np.zeros((norb, norb), dtype=np.complex128)
            mats[R_key][i, j] = Re + 1j * Im
        return norb, mats

def read_poscar(path):
    """Return lattice vectors a1,a2,a3 (3×3) and scale factor."""
    with open(path, "r") as f:
        title = f.readline()
        scale = float(f.readline().strip())
        a1 = np.fromstring(f.readline(), sep=" ")[:3]
        a2 = np.fromstring(f.readline(), sep=" ")[:3]
        a3 = np.fromstring(f.readline(), sep=" ")[:3]
    return scale * np.vstack([a1, a2, a3])

def reciprocal_lattice(a):
    """Return reciprocal lattice vectors b1,b2,b3 (3×3)."""
    vol = np.dot(a[0], np.cross(a[1], a[2]))
    b1 = PI2 * np.cross(a[1], a[2]) / vol
    b2 = PI2 * np.cross(a[2], a[0]) / vol
    b3 = PI2 * np.cross(a[0], a[1]) / vol
    return np.vstack([b1, b2, b3])

def kpath_k_g_m(nseg=40, lattice=None):
    """Return fractional 2D k-points along K→Γ→M and segment labels."""
    # For hexagonal lattice (like Bi2Se3), these are the standard high-symmetry points
    K = np.array([1/3, 2/3])
    G = np.array([0.0, 0.0])
    M = np.array([0.5, 0.5])
    
    seg1 = np.linspace(K, G, nseg, endpoint=False)
    seg2 = np.linspace(G, M, nseg + 1)  # include M
    klist = np.vstack([seg1, seg2])
    ticks = {"K": 0, "Γ": len(seg1), "M": len(klist) - 1}
    return klist, ticks

def kpath_g_m_k(nseg=40, lattice=None):
    """Return fractional 2D k-points along Γ→M→K→Γ path."""
    G = np.array([0.0, 0.0])
    M = np.array([0.5, 0.5])
    K = np.array([1/3, 2/3])
    
    seg1 = np.linspace(G, M, nseg, endpoint=False)
    seg2 = np.linspace(M, K, nseg, endpoint=False)
    seg3 = np.linspace(K, G, nseg + 1)  # include G
    klist = np.vstack([seg1, seg2, seg3])
    
    ticks = {"Γ": 0, "M": len(seg1), "K": len(seg1) + len(seg2), "Γ": len(klist) - 1}
    return klist, ticks

def calculate_interlayer_hopping(k2d, mats3d, norb, ijmax=5):
    """
    Calculate interlayer hopping matrices Hij(ic, :, :) following WannierTools approach.
    
    This is equivalent to ham_qlayer2qlayer2 in WannierTools.
    
    Parameters:
    - k2d: 2D k-point [kx, ky] in fractional coordinates
    - mats3d: dict with 'H' and 'S' containing R-space matrices
    - norb: number of orbitals per layer
    - ijmax: maximum interlayer distance to consider
    
    Returns:
    - Hij: dict mapping ic -> (norb, norb) matrix for interlayer hopping
    """
    Hij = {}
    
    # Initialize all interlayer matrices
    for ic in range(-ijmax, ijmax + 1):
        Hij[ic] = np.zeros((norb, norb), dtype=np.complex128)
    
    kx, ky = k2d
    
    # Sum over R vectors
    for (Rx, Ry, Rz), Hmat in mats3d['H'].items():
        int_ic = int(Rz)  # interlayer separation
        
        if abs(int_ic) <= ijmax:
            # In-plane Fourier transform: exp(i k·R_parallel)
            kdotr = kx * Rx + ky * Ry
            ratio = np.exp(1j * PI2 * kdotr)
            
            # Add contribution to interlayer hopping matrix
            Hij[int_ic] += Hmat * ratio
    
    return Hij

def construct_slab_hamiltonian(Hij, norb, nslab, ijmax):
    """
    Construct slab Hamiltonian from interlayer hopping matrices.
    
    This follows the same logic as ham_slab in WannierTools:
    - Total dimension: (nslab * norb) × (nslab * norb)  
    - Each layer has norb orbitals
    - Hij[i1-i2] gives hopping from layer i2 to layer i1
    
    Parameters:
    - Hij: dict mapping ic -> (norb, norb) interlayer hopping matrix
    - norb: number of orbitals per layer
    - nslab: number of layers in slab
    - ijmax: maximum interlayer distance
    
    Returns:
    - Hamk_slab: (nslab*norb, nslab*norb) slab Hamiltonian
    """
    total_dim = nslab * norb
    Hamk_slab = np.zeros((total_dim, total_dim), dtype=np.complex128)
    
    # Loop over layer indices (following WannierTools convention)
    for i1 in range(1, nslab + 1):  # column index (1-based like Fortran)
        for i2 in range(1, nslab + 1):  # row index (1-based like Fortran)
            if abs(i2 - i1) <= ijmax:
                # Convert to 0-based Python indices
                row_start = (i2 - 1) * norb
                row_end = i2 * norb
                col_start = (i1 - 1) * norb
                col_end = i1 * norb
                
                # Hij(i1-i2) gives hopping from layer i2 to layer i1
                ic = i1 - i2
                if ic in Hij:
                    Hamk_slab[row_start:row_end, col_start:col_end] = Hij[ic]
    
    return Hamk_slab

def solve_slab_bands(Hamk_slab, Sk_slab=None, return_vectors=False):
    """Solve slab eigenvalue problem."""
    if Sk_slab is not None:
        # Generalized eigenvalue problem H ψ = λ S ψ
        if return_vectors:
            vals, vecs = la.eigh(Hamk_slab, Sk_slab)
            return np.real_if_close(vals), vecs
        else:
            vals = la.eigh(Hamk_slab, Sk_slab, eigvals_only=True)
            return np.real_if_close(vals)
    else:
        # Standard eigenvalue problem H ψ = λ ψ
        if return_vectors:
            vals, vecs = la.eigh(Hamk_slab)
            return np.real_if_close(vals), vecs
        else:
            vals = la.eigh(Hamk_slab, eigvals_only=True)
            return np.real_if_close(vals)

def analyze_surface_character(eigenvectors, norb, nslab):
    """
    Analyze surface character of slab eigenstates, distinguishing top and bottom surfaces.
    
    Parameters:
    - eigenvectors: (nslab*norb, nslab*norb) eigenvector matrix
    - norb: number of orbitals per layer
    - nslab: number of layers
    
    Returns:
    - surface_data: dict with keys:
        - 'total': total surface character (top + bottom)
        - 'top': top surface character
        - 'bottom': bottom surface character
        - 'bulk': bulk character (middle layers)
        - 'surface_index': surface index (+1 for top, -1 for bottom, 0 for bulk)
    """
    total_dim = nslab * norb
    surface_data = {
        'total': np.zeros(total_dim),
        'top': np.zeros(total_dim),
        'bottom': np.zeros(total_dim),
        'bulk': np.zeros(total_dim),
        'surface_index': np.zeros(total_dim)  # New: -1 to +1 scale
    }
    
    for i in range(total_dim):
        vec = eigenvectors[:, i]
        
        # Calculate weights on different layers
        layer_weights = np.zeros(nslab)
        for layer in range(nslab):
            start_idx = layer * norb
            end_idx = (layer + 1) * norb
            layer_weights[layer] = np.sum(np.abs(vec[start_idx:end_idx])**2)
        
        # Top surface (first layer)
        surface_data['top'][i] = layer_weights[0]
        
        # Bottom surface (last layer)
        surface_data['bottom'][i] = layer_weights[-1]
        
        # Total surface character
        surface_data['total'][i] = layer_weights[0] + layer_weights[-1]
        
        # Bulk character (middle layers)
        if nslab > 2:
            surface_data['bulk'][i] = np.sum(layer_weights[1:-1])
        else:
            surface_data['bulk'][i] = 0
            
        # Surface index: +1 for top surface, -1 for bottom surface, 0 for bulk
        top_weight = layer_weights[0]
        bottom_weight = layer_weights[-1]
        
        if top_weight + bottom_weight > 0:
            # Normalize to [-1, +1] range
            surface_data['surface_index'][i] = (top_weight - bottom_weight) / (top_weight + bottom_weight)
        else:
            surface_data['surface_index'][i] = 0.0
    
    return surface_data

def get_layer_projections(eigenvectors, norb, nslab):
    """
    Get projections of eigenstates onto different layers.
    
    Parameters:
    - eigenvectors: (nslab*norb, nslab*norb) eigenvector matrix
    - norb: number of orbitals per layer
    - nslab: number of layers
    
    Returns:
    - layer_projections: (nslab*norb, nslab) array of layer projections
    """
    total_dim = nslab * norb
    layer_projections = np.zeros((total_dim, nslab))
    
    for i in range(total_dim):
        vec = eigenvectors[:, i]
        
        for layer in range(nslab):
            start_idx = layer * norb
            end_idx = (layer + 1) * norb
            layer_projections[i, layer] = np.sum(np.abs(vec[start_idx:end_idx])**2)
    
    return layer_projections

def calculate_bulk_hamiltonian(k3d, mats3d, norb):
    """
    Calculate bulk Hamiltonian H(k) for 3D k-point.
    
    Parameters:
    - k3d: 3D k-point [kx, ky, kz] in fractional coordinates
    - mats3d: dict with 'H' and optionally 'S' containing R-space matrices
    - norb: number of orbitals
    
    Returns:
    - Hamk_bulk: (norb, norb) bulk Hamiltonian
    - Sk_bulk: (norb, norb) overlap matrix (if 'S' in mats3d)
    """
    Hamk_bulk = np.zeros((norb, norb), dtype=np.complex128)
    Sk_bulk = None
    
    kx, ky, kz = k3d
    
    # Sum over all R vectors for bulk Hamiltonian
    for (Rx, Ry, Rz), Hmat in mats3d['H'].items():
        kdotr = kx * Rx + ky * Ry + kz * Rz
        ratio = np.exp(1j * PI2 * kdotr)
        Hamk_bulk += Hmat * ratio
    
    # Calculate overlap matrix if available
    if 'S' in mats3d:
        Sk_bulk = np.zeros((norb, norb), dtype=np.complex128)
        for (Rx, Ry, Rz), Smat in mats3d['S'].items():
            kdotr = kx * Rx + ky * Ry + kz * Rz
            ratio = np.exp(1j * PI2 * kdotr)
            Sk_bulk += Smat * ratio
    
    return Hamk_bulk, Sk_bulk

def calculate_bulk_bands(klist, mats, norb, kz=0.0, include_overlap=False):
    """
    Calculate bulk band structure.
    
    Parameters:
    - klist: array of 2D k-points (will be extended to 3D with fixed kz)
    - mats: dict with 'H' and optionally 'S' containing R-space matrices
    - norb: number of orbitals
    - kz: fixed kz value for 2D cut of 3D bulk
    - include_overlap: whether to include overlap matrix
    
    Returns:
    - bands: (nk, norb) array of eigenvalues
    """
    bands = []
    
    print(f"Calculating bulk bands with {norb} orbitals")
    print(f"Using kz = {kz}")
    
    for i, k2d in enumerate(klist):
        if i % 20 == 0:
            print(f"  k-point {i+1}/{len(klist)}")
        
        # Extend 2D k-point to 3D
        k3d = np.array([k2d[0], k2d[1], kz])
        
        # Calculate bulk Hamiltonian
        Hamk_bulk, Sk_bulk = calculate_bulk_hamiltonian(k3d, mats, norb)
        
        # Solve eigenvalue problem
        if include_overlap and Sk_bulk is not None:
            vals = la.eigh(Hamk_bulk, Sk_bulk, eigvals_only=True)
        else:
            vals = la.eigh(Hamk_bulk, eigvals_only=True)
        
        bands.append(np.real_if_close(vals))
    
    return np.array(bands)

def calculate_slab_bands(klist, mats, norb, nslab, ijmax=5, include_overlap=False, 
                        analyze_surface=False):
    """
    Calculate slab band structure.
    
    Parameters:
    - klist: array of 2D k-points
    - mats: dict with 'H' and optionally 'S' containing R-space matrices  
    - norb: number of orbitals per layer
    - nslab: number of layers in slab
    - ijmax: maximum interlayer distance
    - include_overlap: whether to include overlap matrix
    - analyze_surface: whether to analyze surface character
    
    Returns:
    - bands: (nk, nslab*norb) array of eigenvalues
    - surface_weights: (nk, nslab*norb) array of surface character (if analyze_surface=True)
    """
    bands = []
    surface_weights = {
        'total': [],
        'top': [],
        'bottom': [],
        'bulk': [],
        'surface_index': []
    } if analyze_surface else None
    
    print(f"Calculating slab with {nslab} layers, {norb} orbitals per layer")
    print(f"Total slab dimension: {nslab * norb}")
    print(f"Maximum interlayer distance: {ijmax}")
    if analyze_surface:
        print("Surface state analysis enabled")
    
    for i, k2d in enumerate(klist):
        if i % 10 == 0:
            print(f"  k-point {i+1}/{len(klist)}")
        
        # Calculate interlayer hopping matrices
        Hij = calculate_interlayer_hopping(k2d, mats, norb, ijmax)
        
        # Construct slab Hamiltonian
        Hamk_slab = construct_slab_hamiltonian(Hij, norb, nslab, ijmax)
        
        # Handle overlap matrix if needed
        Sk_slab = None
        if include_overlap and 'S' in mats:
            # Calculate interlayer overlap matrices
            mats_S = {'H': mats['S']}  # Reuse the same function
            Sij = calculate_interlayer_hopping(k2d, mats_S, norb, ijmax)
            Sk_slab = construct_slab_hamiltonian(Sij, norb, nslab, ijmax)
        
        # Solve eigenvalue problem
        if analyze_surface:
            vals, vecs = solve_slab_bands(Hamk_slab, Sk_slab, return_vectors=True)
            bands.append(vals)
            
            # Analyze surface character
            surf_data = analyze_surface_character(vecs, norb, nslab)
            for key in surface_weights:
                surface_weights[key].append(surf_data[key])
        else:
            vals = solve_slab_bands(Hamk_slab, Sk_slab)
            bands.append(vals)
    
    if analyze_surface:
        # Convert to numpy arrays
        for key in surface_weights:
            surface_weights[key] = np.array(surface_weights[key])
        return np.array(bands), surface_weights
    else:
        return np.array(bands)

def calculate_single_k_slab(k2d, mats, norb, nslab, ijmax, include_overlap, analyze_surface):
    """Calculate slab bands for a single k-point (for parallel processing)."""
    # Calculate interlayer hopping matrices
    Hij = calculate_interlayer_hopping(k2d, mats, norb, ijmax)
    
    # Construct slab Hamiltonian
    Hamk_slab = construct_slab_hamiltonian(Hij, norb, nslab, ijmax)
    
    # Handle overlap matrix if needed
    Sk_slab = None
    if include_overlap and 'S' in mats:
        # Calculate interlayer overlap matrices
        mats_S = {'H': mats['S']}  # Reuse the same function
        Sij = calculate_interlayer_hopping(k2d, mats_S, norb, ijmax)
        Sk_slab = construct_slab_hamiltonian(Sij, norb, nslab, ijmax)
    
    # Solve eigenvalue problem
    if analyze_surface:
        vals, vecs = solve_slab_bands(Hamk_slab, Sk_slab, return_vectors=True)
        
        # Analyze surface character
        surf_data = analyze_surface_character(vecs, norb, nslab)
        return vals, surf_data
    else:
        vals = solve_slab_bands(Hamk_slab, Sk_slab)
        return vals

def calculate_slab_bands_parallel(klist, mats, norb, nslab, ijmax=5, include_overlap=False, 
                                analyze_surface=False, n_jobs=None):
    """
    Calculate slab band structure using parallel processing.
    
    Parameters:
    - klist: array of 2D k-points
    - mats: dict with 'H' and optionally 'S' containing R-space matrices  
    - norb: number of orbitals per layer
    - nslab: number of layers in slab
    - ijmax: maximum interlayer distance
    - include_overlap: whether to include overlap matrix
    - analyze_surface: whether to analyze surface character
    - n_jobs: number of parallel jobs (None for auto)
    
    Returns:
    - bands: (nk, nslab*norb) array of eigenvalues
    - surface_weights: dict with surface character data (if analyze_surface=True)
    """
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(klist))
    
    print(f"Calculating slab with {nslab} layers, {norb} orbitals per layer")
    print(f"Total slab dimension: {nslab * norb}")
    print(f"Maximum interlayer distance: {ijmax}")
    print(f"Using {n_jobs} parallel processes")
    if analyze_surface:
        print("Surface state analysis enabled")
    
    # Create partial function for parallel processing
    calc_func = partial(calculate_single_k_slab, 
                       mats=mats, norb=norb, nslab=nslab, ijmax=ijmax,
                       include_overlap=include_overlap, analyze_surface=analyze_surface)
    
    # Parallel calculation
    with Pool(n_jobs) as pool:
        results = pool.map(calc_func, klist)
    
    # Process results
    bands = []
    surface_weights = {
        'total': [],
        'top': [],
        'bottom': [],
        'bulk': [],
        'surface_index': []
    } if analyze_surface else None
    
    for result in results:
        if analyze_surface:
            vals, surf_data = result
            bands.append(vals)
            for key in surface_weights:
                surface_weights[key].append(surf_data[key])
        else:
            bands.append(result)
    
    if analyze_surface:
        # Convert to numpy arrays
        for key in surface_weights:
            surface_weights[key] = np.array(surface_weights[key])
        return np.array(bands), surface_weights
    else:
        return np.array(bands)

def plot_slab_comparison(klist, bands_dict, ticks, title="Slab Band Structure", 
                        energy_range=(-2, 2), outfile=None, surface_weights=None,
                        surface_threshold=0.5):
    """Plot comparison of different slab/bulk calculations with surface state highlighting."""
    # Create figure with space for colorbar if needed
    has_surface_analysis = (surface_weights is not None and 
                           any(label != "Bulk" for label in surface_weights.keys()))
    
    if has_surface_analysis:
        fig, (ax, cax) = plt.subplots(1, 2, figsize=(12, 6), 
                                     gridspec_kw={'width_ratios': [10, 0.5]})
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = None
    
    x = np.arange(len(klist))
    
    colors = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '-', '--', '-.', ':', '-', '--']
    linewidths = [1.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    
    # Track surface state legends
    legend_added = {'top': False, 'bottom': False, 'mixed': False}
    
    # For colorbar
    all_surface_values = []
    
    for i, (label, bands) in enumerate(bands_dict.items()):
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        lw = linewidths[i % len(linewidths)]
        
        # Special formatting for bulk bands
        if label == "Bulk":
            alpha = 0.9
            lw = 1.2
        else:
            alpha = 0.7
        
        # Check if we have surface weights for this slab
        has_surface_data = (surface_weights is not None and 
                           label in surface_weights and 
                           label != "Bulk")
        
        # Plot bands within energy range
        for b in range(bands.shape[1]):
            band = bands[:, b]
            # Only plot if band has values in the energy range
            if np.any((band >= energy_range[0]) & (band <= energy_range[1])):
                
                if has_surface_data:
                    # Get surface character for this band
                    total_surf = surface_weights[label]['total'][:, b]  # (nk,) array
                    surface_index = surface_weights[label]['surface_index'][:, b]  # -1 to +1
                    
                    # Average surface character
                    avg_total = np.mean(total_surf)
                    
                    if avg_total > surface_threshold:
                        # Surface state - use colorbar mapping
                        if cax is not None:
                            # Use RdBu_r colormap: red for +1 (top), blue for -1 (bottom)
                            scatter = ax.scatter(x, band, c=surface_index, cmap='RdBu_r', 
                                               s=15, alpha=0.8, vmin=-1, vmax=1)
                            all_surface_values.extend(surface_index)
                            
                            # Add legend entries only once
                            if not legend_added['top']:
                                ax.plot([], [], 'ro', markersize=4, label='Top surface', alpha=0.8)
                                legend_added['top'] = True
                            if not legend_added['bottom']:
                                ax.plot([], [], 'bo', markersize=4, label='Bottom surface', alpha=0.8)
                                legend_added['bottom'] = True
                        else:
                            # Fallback without colorbar
                            avg_index = np.mean(surface_index)
                            if avg_index > 0.3:
                                surf_color = 'red'
                                surf_label = 'Top surface' if not legend_added['top'] else ""
                                legend_added['top'] = True
                            elif avg_index < -0.3:
                                surf_color = 'blue'  
                                surf_label = 'Bottom surface' if not legend_added['bottom'] else ""
                                legend_added['bottom'] = True
                            else:
                                surf_color = 'purple'
                                surf_label = 'Mixed surface' if not legend_added['mixed'] else ""
                                legend_added['mixed'] = True
                            
                            ax.plot(x, band, color=surf_color, lw=1.5, 
                                   alpha=0.9, label=surf_label if surf_label else "")
                    else:
                        # Bulk-like state - use original formatting but lighter
                        ax.plot(x, band, color=color, lw=lw, linestyle=ls, 
                               alpha=alpha*0.4, label=label if b == 0 else "")
                else:
                    # No surface analysis - use original formatting
                    ax.plot(x, band, color=color, lw=lw, linestyle=ls, 
                           alpha=alpha, label=label if b == 0 else "")
    
    ax.set_xticks(list(ticks.values()))
    ax.set_xticklabels(list(ticks.keys()))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(energy_range)
    ax.set_ylabel('Energy (eV)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add vertical lines at high-symmetry points
    for tick_pos in ticks.values():
        ax.axvline(x=tick_pos, color='black', linestyle='-', alpha=0.3)
    
    # Add Fermi level
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add colorbar if we have surface analysis
    if cax is not None and all_surface_values:
        # Create colorbar
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Create a ScalarMappable for the colorbar using RdBu_r
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        sm = cm.ScalarMappable(norm=norm, cmap='RdBu_r')
        sm.set_array([])
        
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Surface Index', rotation=270, labelpad=20)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['Bottom\n(-1)', '-0.5', 'Bulk\n(0)', '0.5', 'Top\n(+1)'])
        
        # Add text annotations
        cbar.ax.text(1.5, 1.05, 'Top Surface', transform=cbar.ax.transAxes, 
                    ha='left', va='center', color='red', fontweight='bold', fontsize=9)
        cbar.ax.text(1.5, -0.05, 'Bottom Surface', transform=cbar.ax.transAxes,
                    ha='left', va='center', color='blue', fontweight='bold', fontsize=9)
    elif cax is not None:
        # Remove empty colorbar axis
        cax.remove()
    
    plt.tight_layout()
    
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {outfile}")
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description='Calculate slab band structure (correct method)')
    parser.add_argument('--H', default='../H.dat', help='H.dat path')
    parser.add_argument('--S', default='../S.dat', help='S.dat path')
    parser.add_argument('--poscar', default='../POSCAR', help='POSCAR path')
    parser.add_argument('--nslab', type=int, default=5, help='Number of layers in slab')
    parser.add_argument('--ijmax', type=int, default=5, help='Maximum interlayer distance')
    parser.add_argument('--nseg', type=int, default=40, help='Points per k-path segment')
    parser.add_argument('--kpath', choices=['KGM', 'GMKG'], default='KGM',
                       help='K-path choice: KGM (K→Γ→M) or GMKG (Γ→M→K→Γ)')
    parser.add_argument('--outfile', default='slab_bands_correct.png', help='Output plot file')
    parser.add_argument('--energy_range', nargs=2, type=float, default=[-10.0, 0.0],
                       help='Energy range for plotting [min, max] in eV')
    parser.add_argument('--compare_thickness', action='store_true',
                       help='Compare different slab thicknesses')
    parser.add_argument('--include_bulk', action='store_true',
                       help='Include bulk band calculation for comparison')
    parser.add_argument('--bulk_kz', type=float, default=0.0,
                       help='kz value for bulk band calculation (default: 0.0)')
    parser.add_argument('--include_overlap', action='store_true', default=False,
                       help='Include overlap matrix (non-orthogonal basis)')
    parser.add_argument('--analyze_surface', action='store_true',
                       help='Analyze surface character of slab states')
    parser.add_argument('--surface_threshold', type=float, default=0.5,
                       help='Threshold for identifying surface states (default: 0.5)')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs (1=serial, >1=parallel, 0=auto)')
    
    args = parser.parse_args()
    
    # Read TB data
    print(f"Reading Hamiltonian from {args.H}...")
    norb_H, Hmats = read_sparse_tb(args.H)
    mats = {'H': Hmats}
    
    if args.include_overlap:
        print(f"Reading overlap from {args.S}...")
        norb_S, Smats = read_sparse_tb(args.S)
        assert norb_H == norb_S, "H.dat / S.dat orbital mismatch"
        mats['S'] = Smats
    
    norb = norb_H
    print(f"Orbitals per layer: {norb}")
    
    # Analyze R vectors
    Rz_values = [Rz for (Rx, Ry, Rz) in Hmats.keys()]
    print(f"Available Rz values: {min(Rz_values)} to {max(Rz_values)}")
    
    # Read lattice structure from POSCAR
    print(f"Reading lattice from {args.poscar}...")
    lattice = read_poscar(args.poscar)
    reciprocal = reciprocal_lattice(lattice)
    print(f"Real space lattice vectors (Å):")
    for i, vec in enumerate(lattice):
        print(f"  a{i+1} = [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")
    print(f"Reciprocal lattice vectors (Å⁻¹):")
    for i, vec in enumerate(reciprocal):
        print(f"  b{i+1} = [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")
    
    # Set up k-path
    if args.kpath == 'KGM':
        klist, ticks = kpath_k_g_m(args.nseg, lattice)
    else:
        klist, ticks = kpath_g_m_k(args.nseg, lattice)
    print(f"K-path: {' → '.join(list(ticks.keys()))}, {len(klist)} points")
    
    # Calculate bands
    bands_dict = {}
    surface_weights_dict = {}
    
    # Calculate bulk bands if requested
    if args.include_bulk:
        print(f"\nCalculating bulk bands...")
        bulk_bands = calculate_bulk_bands(klist, mats, norb, args.bulk_kz, args.include_overlap)
        bands_dict["Bulk"] = bulk_bands
        
        # Print bulk band statistics
        n_bands = bulk_bands.shape[1]
        e_min, e_max = np.min(bulk_bands), np.max(bulk_bands)
        print(f"  Bulk (kz={args.bulk_kz}): {n_bands} bands, E ∈ [{e_min:.3f}, {e_max:.3f}] eV")
    
    # Calculate slab bands
    if args.compare_thickness:
        nslab_list = [3, 5, 7, 9, args.nslab] if args.nslab not in [3, 5, 7, 9] else [3, 5, 7, 9]
        nslab_list = sorted(list(set(nslab_list)))
    else:
        nslab_list = [args.nslab]
    
    for nslab in nslab_list:
        label = f"{nslab}L slab"
        print(f"\nCalculating {label}...")
        
        # Determine if we use parallel processing
        use_parallel = (args.n_jobs > 1) or (args.n_jobs == 0)
        n_jobs = args.n_jobs if args.n_jobs > 0 else None
        
        # Calculate with or without surface analysis
        if args.analyze_surface:
            if use_parallel:
                bands, surf_weights = calculate_slab_bands_parallel(
                    klist, mats, norb, nslab, args.ijmax, args.include_overlap, 
                    analyze_surface=True, n_jobs=n_jobs)
            else:
                bands, surf_weights = calculate_slab_bands(
                    klist, mats, norb, nslab, args.ijmax, args.include_overlap, 
                    analyze_surface=True)
            
            surface_weights_dict[label] = surf_weights
            
            # Analyze surface states
            total_surf = surf_weights['total']  # (nk, nbands)
            avg_surface_char = np.mean(total_surf, axis=0)  # Average over k-points
            n_surface_states = np.sum(avg_surface_char > args.surface_threshold)
            
            # Analyze top vs bottom surface states
            top_surf = surf_weights['top']
            bottom_surf = surf_weights['bottom']
            avg_top = np.mean(top_surf, axis=0)
            avg_bottom = np.mean(bottom_surf, axis=0)
            
            n_top_surface = np.sum((avg_surface_char > args.surface_threshold) & 
                                  (avg_top > 2 * avg_bottom))
            n_bottom_surface = np.sum((avg_surface_char > args.surface_threshold) & 
                                     (avg_bottom > 2 * avg_top))
            n_mixed_surface = n_surface_states - n_top_surface - n_bottom_surface
            
            print(f"  Surface states identified: {n_surface_states} bands")
            print(f"    - Top surface: {n_top_surface} bands")
            print(f"    - Bottom surface: {n_bottom_surface} bands") 
            print(f"    - Mixed surface: {n_mixed_surface} bands")
        else:
            if use_parallel:
                bands = calculate_slab_bands_parallel(
                    klist, mats, norb, nslab, args.ijmax, args.include_overlap, 
                    n_jobs=n_jobs)
            else:
                bands = calculate_slab_bands(klist, mats, norb, nslab, args.ijmax, args.include_overlap)
        
        bands_dict[label] = bands
        
        # Print band statistics
        n_bands = bands.shape[1]
        e_min, e_max = np.min(bands), np.max(bands)
        print(f"  {label}: {n_bands} bands, E ∈ [{e_min:.3f}, {e_max:.3f}] eV")
    
    # Plot results
    title = f"Bi2Se3 Band Structure (Correct Method)"
    if args.include_bulk and args.compare_thickness:
        title += " - Bulk vs. Slab Comparison"
    elif args.include_bulk:
        title += " - Bulk vs. Slab"
    elif args.compare_thickness:
        title += " - Slab Thickness Comparison"
    else:
        title += " - Slab"
    
    # Pass surface weights if available
    surface_weights = surface_weights_dict if args.analyze_surface else None
    
    fig, ax = plot_slab_comparison(klist, bands_dict, ticks, title, 
                                  tuple(args.energy_range), args.outfile,
                                  surface_weights, args.surface_threshold)
    
    print(f"\nCalculation completed!")
    print(f"Results saved to {args.outfile}")
    
    plt.show()

if __name__ == '__main__':
    main() 