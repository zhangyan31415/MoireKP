#!/usr/bin/env python3
"""
Test script for surface state analysis functionality

This script tests the surface state analysis and visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slab_correct import *

def test_surface_analysis():
    """Test surface state analysis functionality."""
    
    print("=" * 60)
    print("Testing Surface State Analysis")
    print("=" * 60)
    
    # Check if required files exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h_file = os.path.join(base_dir, 'H.dat')
    s_file = os.path.join(base_dir, 'S.dat')
    
    if not os.path.exists(h_file) or not os.path.exists(s_file):
        print("ERROR: Required files not found")
        return False
    
    try:
        # Read data
        print("\n1. Loading tight-binding data...")
        norb_H, Hmats = read_sparse_tb(h_file)
        norb_S, Smats = read_sparse_tb(s_file)
        mats = {'H': Hmats, 'S': Smats}
        norb = norb_H
        
        print(f"   ✓ Orbitals per layer: {norb}")
        
        # Set up a small k-path for testing
        print("\n2. Setting up k-path...")
        klist, ticks = kpath_k_g_m(10)  # Small for fast testing
        print(f"   ✓ K-points: {len(klist)}")
        
        # Test surface analysis with different slab thicknesses
        print("\n3. Testing surface state analysis...")
        
        bands_dict = {}
        surface_weights_dict = {}
        
        for nslab in [5, 7]:  # Test with 5 and 7 layers
            print(f"\n   Analyzing {nslab}-layer slab...")
            
            # Calculate with surface analysis
            bands, surf_weights = calculate_slab_bands(
                klist, mats, norb, nslab, ijmax=2, 
                include_overlap=True, analyze_surface=True
            )
            
            bands_dict[f"{nslab}L"] = bands
            surface_weights_dict[f"{nslab}L"] = surf_weights
            
            # Analyze surface character statistics
            avg_surface_char = np.mean(surf_weights, axis=0)  # Average over k-points
            
            surface_threshold = 0.5
            n_surface_states = np.sum(avg_surface_char > surface_threshold)
            n_bulk_states = np.sum(avg_surface_char <= surface_threshold)
            
            print(f"   ✓ Total bands: {len(avg_surface_char)}")
            print(f"   ✓ Surface states (>{surface_threshold}): {n_surface_states}")
            print(f"   ✓ Bulk-like states (≤{surface_threshold}): {n_bulk_states}")
            print(f"   ✓ Max surface character: {np.max(avg_surface_char):.3f}")
            print(f"   ✓ Min surface character: {np.min(avg_surface_char):.3f}")
            
            # Show distribution of surface character
            hist, bins = np.histogram(avg_surface_char, bins=10, range=(0, 1))
            print(f"   ✓ Surface character distribution:")
            for i in range(len(hist)):
                if hist[i] > 0:
                    print(f"      [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]} bands")
        
        # Create visualization
        print("\n4. Creating surface state visualization...")
        
        fig, ax = plot_slab_comparison(
            klist, bands_dict, ticks,
            "Surface State Analysis Test",
            energy_range=(-3, 1),
            surface_weights=surface_weights_dict,
            surface_threshold=0.5,
            outfile="test_surface_states.png"
        )
        
        print("   ✓ Plot saved as 'test_surface_states.png'")
        print("   ✓ Red lines indicate surface states")
        print("   ✓ Lighter lines indicate bulk-like states")
        
        print("\n" + "=" * 60)
        print("✓ Surface state analysis test completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_surface_analysis():
    """Demonstrate how to use surface analysis features."""
    
    print("\n" + "=" * 60)
    print("Surface State Analysis Usage Examples")
    print("=" * 60)
    
    print("\n1. Basic surface analysis:")
    print("   python slab_correct.py --analyze_surface --nslab 7")
    
    print("\n2. Surface analysis with bulk comparison:")
    print("   python slab_correct.py --include_bulk --analyze_surface --nslab 5")
    
    print("\n3. Multiple slabs with surface analysis:")
    print("   python slab_correct.py --compare_thickness --analyze_surface --nslab 9")
    
    print("\n4. Adjust surface threshold:")
    print("   python slab_correct.py --analyze_surface --surface_threshold 0.7 --nslab 5")
    
    print("\nSurface State Interpretation:")
    print("- Red bands: Surface states (localized at slab surfaces)")
    print("- Light colored bands: Bulk-like states (extended through slab)")
    print("- Surface character > threshold → surface state")
    print("- For topological insulators like Bi2Se3:")
    print("  * Surface states should appear in the bulk gap")
    print("  * Linear dispersion near Γ point")
    print("  * Dirac cone structure")

if __name__ == '__main__':
    success = test_surface_analysis()
    
    if success:
        demonstrate_surface_analysis()
    else:
        print("\nSurface analysis test failed. Please check your setup.")
        sys.exit(1) 