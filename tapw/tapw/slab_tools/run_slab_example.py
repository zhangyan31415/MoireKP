#!/usr/bin/env python3
"""
Simple example script for Bi2Se3 slab band structure calculation

This script demonstrates basic usage of the slab calculation functionality.
"""

import sys
import os

# Add the current directory to Python path to import slab module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the slab calculation functions
from slab import *

def run_basic_example():
    """Run a basic slab calculation example."""
    
    print("=" * 60)
    print("Bi2Se3 Slab Band Structure Calculation Example")
    print("=" * 60)
    
    # Check if required files exist
    required_files = ['H.dat', 'S.dat', 'POSCAR']
    for filename in required_files:
        if not os.path.exists(filename):
            print(f"ERROR: Required file '{filename}' not found in current directory")
            return False
    
    # Read TB data
    print("\n1. Loading tight-binding data...")
    try:
        norb_H, Hmats = read_sparse_tb('H.dat')
        norb_S, Smats = read_sparse_tb('S.dat')
        assert norb_H == norb_S, "H.dat / S.dat orbital mismatch"
        norb = norb_H
        mats = {'H': Hmats, 'S': Smats}
        
        print(f"   ✓ Total orbitals: {norb}")
        print(f"   ✓ Available R vectors: {len(Hmats)}")
        
        # Print R vector statistics
        Rz_values = [Rz for (Rx, Ry, Rz) in Hmats.keys()]
        print(f"   ✓ Rz range: {min(Rz_values)} to {max(Rz_values)}")
        
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Set up k-path
    print("\n2. Setting up k-point path...")
    klist, ticks = kpath_k_g_m(40)  # 40 points per segment
    print(f"   ✓ K-path: {' → '.join(list(ticks.keys()))}")
    print(f"   ✓ Total k-points: {len(klist)}")
    
    # Calculate bands for different slab thicknesses
    print("\n3. Calculating slab band structures...")
    max_Rz_list = [0, 1, 2]  # Single layer, 3-layer, 5-layer
    bands_data = {}
    
    for max_Rz in max_Rz_list:
        label = 'single_layer' if max_Rz == 0 else f'slab_{2*max_Rz+1}L'
        print(f"   Calculating {label} (max_Rz = {max_Rz})...")
        
        bands = []
        for i, k2d in enumerate(klist):
            if i % 20 == 0:
                print(f"     Progress: {i+1}/{len(klist)} k-points", end='\r')
            
            Hk, Sk = build_slab_matrices(k2d, mats, norb, max_Rz)
            vals = solve_bands(Hk, Sk)
            bands.append(vals)
        
        bands_data[label] = np.array(bands)
        print(f"     ✓ {label} completed")
    
    # Plot results
    print("\n4. Generating plots...")
    try:
        fig, ax = plot_slab_bands(
            klist, bands_data, ticks, 
            title="Bi2Se3 Slab Band Structure Comparison",
            outfile="slab_bands_example.png"
        )
        print("   ✓ Plot saved as 'slab_bands_example.png'")
    except Exception as e:
        print(f"   ✗ Error generating plot: {e}")
        return False
    
    # Print summary
    print("\n5. Calculation Summary:")
    print("-" * 40)
    for label, bands in bands_data.items():
        n_bands = bands.shape[1]
        e_min, e_max = np.min(bands), np.max(bands)
        
        # Find bands near Fermi level
        fermi_bands = bands[(bands >= -0.5) & (bands <= 0.5)]
        n_fermi_bands = len(fermi_bands)
        
        print(f"   {label}:")
        print(f"     - Total bands: {n_bands}")
        print(f"     - Energy range: [{e_min:.3f}, {e_max:.3f}] eV")
        print(f"     - Bands near Fermi level (±0.5 eV): {n_fermi_bands}")
    
    print("\n" + "=" * 60)
    print("Calculation completed successfully!")
    print("Check 'slab_bands_example.png' for the band structure plot.")
    print("=" * 60)
    
    return True

def run_advanced_example():
    """Run an advanced example with surface state analysis."""
    
    print("\n" + "=" * 60)
    print("Advanced Example: Surface State Analysis")
    print("=" * 60)
    
    # This would include more sophisticated analysis
    # For now, just show the command line usage
    
    print("\nFor advanced analysis, you can use the command line interface:")
    print("\nBasic usage:")
    print("  python slab.py --max_Rz 2 --nseg 40")
    
    print("\nCompare different thicknesses:")
    print("  python slab.py --compare --max_Rz 3")
    
    print("\nWith surface state analysis:")
    print("  python slab.py --surface_analysis --layer_orbs 15 --max_Rz 2")
    
    print("\nCustom k-path:")
    print("  python slab.py --kpath GMKG --max_Rz 2")
    
    print("\nAll options:")
    print("  python slab.py --help")

if __name__ == '__main__':
    # Run basic example
    success = run_basic_example()
    
    if success:
        # Show advanced options
        run_advanced_example()
    else:
        print("\nBasic example failed. Please check your input files and try again.")
        sys.exit(1) 