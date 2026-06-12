"""
Bi2Se3 slab tight-binding band structure calculator
--------------------------------------------------
Reads sparse-format Wannier TB files ``H.dat`` and ``S.dat`` (as produced by
*wannier90* / *WannierTools* with ``write_hr = T`` / ``write_sr = T``) and a
VASP-style ``POSCAR``.  It constructs a slab Hamiltonian by selecting matrix 
elements within a specified Rz range, solves the generalized eigenproblem
H(k) ψ = E S(k) ψ along high-symmetry paths, and plots the resulting bands.

Features:
- Single layer: only Rz = 0
- Finite slab: -max_Rz <= Rz <= max_Rz
- Surface states identification
- Bulk vs surface band comparison

Usage (bash):
    python slab.py --H H.dat --S S.dat --poscar POSCAR \
        --max_Rz 2 --nseg 40 --outfile bands_slab.png

Dependencies: numpy, scipy, matplotlib
"""
import argparse
from collections import defaultdict
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

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
        mats = defaultdict(lambda: np.zeros((norb, norb), dtype=np.complex128))
        for _ in range(n_nonzero):
            elems = f.readline().split()
            Rx, Ry, Rz = map(int, elems[:3])
            i, j = int(elems[3]) - 1, int(elems[4]) - 1  # 1-based → 0-based
            Re, Im = map(float, elems[5:7])
            mats[(Rx, Ry, Rz)][i, j] = Re + 1j * Im
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

def kpath_k_g_m(nseg=40):
    """Return fractional 2D k-points along K→Γ→M and a list of segment labels."""
    K = np.array([1/3, 2/3])
    G = np.array([0.0, 0.0])
    M = np.array([0.5, 0.5])
    seg1 = np.linspace(K, G, nseg, endpoint=False)
    seg2 = np.linspace(G, M, nseg + 1)  # include M
    klist = np.vstack([seg1, seg2])
    # high-symmetry indices for x-ticks
    ticks = {"K": 0, "Γ": len(seg1), "M": len(klist) - 1}
    return klist, ticks

def kpath_g_m_k(nseg=40):
    """Return fractional 2D k-points along Γ→M→K→Γ path."""
    G = np.array([0.0, 0.0])
    M = np.array([0.5, 0.5])
    K = np.array([1/3, 2/3])
    
    seg1 = np.linspace(G, M, nseg, endpoint=False)
    seg2 = np.linspace(M, K, nseg, endpoint=False)
    seg3 = np.linspace(K, G, nseg + 1)  # include G
    klist = np.vstack([seg1, seg2, seg3])
    
    # high-symmetry indices for x-ticks
    ticks = {"Γ": 0, "M": len(seg1), "K": len(seg1) + len(seg2), "Γ": len(klist) - 1}
    return klist, ticks

def build_slab_matrices(k2d, mats3d, norb, max_Rz=0):
    """
    Sum over R with -max_Rz <= Rz <= max_Rz → return H(k), S(k).
    max_Rz = 0: single layer (only Rz = 0)
    max_Rz > 0: slab with finite thickness
    """
    Hk = np.zeros((norb, norb), dtype=np.complex128)
    Sk = np.zeros((norb, norb), dtype=np.complex128)
    kx, ky = k2d  # fractional coords along b1,b2
    
    for (Rx, Ry, Rz), Hmat in mats3d['H'].items():
        if abs(Rz) > max_Rz:
            continue
        phase = np.exp(1j * PI2 * (kx * Rx + ky * Ry))
        Hk += Hmat * phase
        
    for (Rx, Ry, Rz), Smat in mats3d['S'].items():
        if abs(Rz) > max_Rz:
            continue
        phase = np.exp(1j * PI2 * (kx * Rx + ky * Ry))
        Sk += Smat * phase
        
    return Hk, Sk

def solve_bands(Hk, Sk, return_vectors=False):
    """Solve generalized Hermitian eigenproblem H ψ = E S ψ."""
    if return_vectors:
        vals, vecs = la.eigh(Hk, Sk)
        return np.real_if_close(vals), vecs
    else:
        vals = la.eigh(Hk, Sk, eigvals_only=True)
        return np.real_if_close(vals)

def analyze_surface_character(eigenvectors, norb, layer_orbs):
    """
    Analyze the surface character of eigenstates.
    Assumes orbitals are ordered by layers.
    
    Parameters:
    - eigenvectors: (norb, norb) array of eigenvectors
    - norb: total number of orbitals
    - layer_orbs: number of orbitals per layer
    
    Returns:
    - surface_weight: (norb,) array of surface character for each eigenstate
    """
    n_layers = norb // layer_orbs
    surface_weight = np.zeros(norb)
    
    for i in range(norb):
        vec = eigenvectors[:, i]
        # Calculate weight on top and bottom layers
        top_weight = np.sum(np.abs(vec[:layer_orbs])**2)
        bottom_weight = np.sum(np.abs(vec[-layer_orbs:])**2)
        surface_weight[i] = top_weight + bottom_weight
    
    return surface_weight

def plot_slab_bands(klist, bands_data, ticks, title="Slab bands", outfile=None, 
                   surface_weights=None, surface_threshold=0.5):
    """
    Plot slab band structure with optional surface state highlighting.
    
    Parameters:
    - klist: k-point list
    - bands_data: dict with keys like 'single_layer', 'slab_2', etc.
    - ticks: high-symmetry point labels
    - surface_weights: optional surface character data
    - surface_threshold: threshold for identifying surface states
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(klist))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, (label, bands) in enumerate(bands_data.items()):
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        
        if surface_weights is not None and label in surface_weights:
            # Plot with surface character
            weights = surface_weights[label]
            for b, band in enumerate(bands.T):
                if np.mean(weights[b]) > surface_threshold:
                    ax.plot(x, band, color='red', lw=2.0, alpha=0.8, label='Surface' if b == 0 else "")
                else:
                    ax.plot(x, band, color=color, lw=0.8, alpha=0.6, linestyle=ls, 
                           label=f'{label} (bulk)' if b == 0 else "")
        else:
            # Regular plot
            for b, band in enumerate(bands.T):
                ax.plot(x, band, color=color, lw=1.0, linestyle=ls, 
                       label=label if b == 0 else "")
    
    ax.set_xticks(list(ticks.values()))
    ax.set_xticklabels(list(ticks.keys()))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-10, 0)
    ax.set_ylabel('Energy (eV)')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend()
    
    # Add vertical lines at high-symmetry points
    for tick_pos in ticks.values():
        ax.axvline(x=tick_pos, color='black', linestyle='-', alpha=0.3)
    
    fig.tight_layout()
    
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f'Plot written to {outfile}')
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description='Calculate slab band structure')
    parser.add_argument('--H', default='H.dat', help='H.dat path')
    parser.add_argument('--S', default='S.dat', help='S.dat path')
    parser.add_argument('--poscar', default='POSCAR', help='POSCAR path')
    parser.add_argument('--max_Rz', type=int, default=2, help='Maximum |Rz| for slab')
    parser.add_argument('--nseg', type=int, default=40, help='Points per k-path segment')
    parser.add_argument('--outfile', default='slab_bands.png', help='Output plot file')
    parser.add_argument('--kpath', choices=['KGM', 'GMKG'], default='KGM', 
                       help='K-path choice: KGM (K→Γ→M) or GMKG (Γ→M→K→Γ)')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare different slab thicknesses')
    parser.add_argument('--surface_analysis', action='store_true',
                       help='Perform surface state analysis')
    parser.add_argument('--layer_orbs', type=int, default=15,
                       help='Number of orbitals per layer (for surface analysis)')
    
    args = parser.parse_args()
    
    # Read TB data
    print(f"Reading Hamiltonian from {args.H}...")
    norb_H, Hmats = read_sparse_tb(args.H)
    print(f"Reading overlap from {args.S}...")
    norb_S, Smats = read_sparse_tb(args.S)
    assert norb_H == norb_S, "H.dat / S.dat orbital mismatch"
    norb = norb_H
    mats = {'H': Hmats, 'S': Smats}
    
    print(f"Total orbitals: {norb}")
    print(f"Available R vectors: {len(Hmats)}")
    
    # Print some R vector statistics
    Rz_values = [Rz for (Rx, Ry, Rz) in Hmats.keys()]
    print(f"Rz range: {min(Rz_values)} to {max(Rz_values)}")
    
    # Lattice (only used for annotation; k is fractional)
    a = read_poscar(args.poscar)
    b = reciprocal_lattice(a)
    
    # K-path
    if args.kpath == 'KGM':
        klist, ticks = kpath_k_g_m(args.nseg)
    else:
        klist, ticks = kpath_g_m_k(args.nseg)
    
    print(f"K-path: {list(ticks.keys())}, {len(klist)} points")
    
    # Calculate bands for different slab thicknesses
    bands_data = {}
    surface_weights = {}
    
    if args.compare:
        # Compare multiple slab thicknesses
        max_Rz_list = [0, 1, 2, args.max_Rz] if args.max_Rz > 2 else [0, 1, 2]
        max_Rz_list = sorted(list(set(max_Rz_list)))  # Remove duplicates
    else:
        max_Rz_list = [args.max_Rz]
    
    for max_Rz in max_Rz_list:
        label = 'single_layer' if max_Rz == 0 else f'slab_{2*max_Rz+1}L'
        print(f"\nCalculating {label} (max_Rz = {max_Rz})...")
        
        bands = []
        if args.surface_analysis and max_Rz > 0:
            surface_chars = []
        
        for i, k2d in enumerate(klist):
            if i % 20 == 0:
                print(f"  k-point {i+1}/{len(klist)}")
            
            Hk, Sk = build_slab_matrices(k2d, mats, norb, max_Rz)
            
            if args.surface_analysis and max_Rz > 0:
                vals, vecs = solve_bands(Hk, Sk, return_vectors=True)
                bands.append(vals)
                surf_char = analyze_surface_character(vecs, norb, args.layer_orbs)
                surface_chars.append(surf_char)
            else:
                vals = solve_bands(Hk, Sk)
                bands.append(vals)
        
        bands_data[label] = np.array(bands)  # Nk × norb
        
        if args.surface_analysis and max_Rz > 0:
            surface_weights[label] = np.array(surface_chars).T  # norb × Nk
    
    # Plot results
    title = f'Bi2Se3 slab TB bands ({args.kpath})'
    if args.compare:
        title += f' - Thickness comparison'
    
    fig, ax = plot_slab_bands(klist, bands_data, ticks, title, args.outfile,
                             surface_weights if args.surface_analysis else None)
    
    # Print summary
    print(f"\nBand calculation completed:")
    for label, bands in bands_data.items():
        n_bands = bands.shape[1]
        e_min, e_max = np.min(bands), np.max(bands)
        print(f"  {label}: {n_bands} bands, E ∈ [{e_min:.3f}, {e_max:.3f}] eV")
    
    plt.show()

if __name__ == '__main__':
    main() 