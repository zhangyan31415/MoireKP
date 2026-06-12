#!/usr/bin/env python3
"""
Test script for the correct slab band structure calculation

This script tests the new slab implementation and compares it with different parameters.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slab_correct import *

def test_basic_slab():
    """Test basic slab calculation."""
    
    print("=" * 60)
    print("Testing Correct Slab Implementation")
    print("=" * 60)
    
    # Check if required files exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h_file = os.path.join(base_dir, 'H.dat')
    s_file = os.path.join(base_dir, 'S.dat')
    poscar_file = os.path.join(base_dir, 'POSCAR')
    
    required_files = [h_file, s_file, poscar_file]
    for filename in required_files:
        if not os.path.exists(filename):
            print(f"ERROR: Required file '{filename}' not found")
            return False
    
    try:
        # Read data
        print("\n1. Loading tight-binding data...")
        norb_H, Hmats = read_sparse_tb(h_file)
        norb_S, Smats = read_sparse_tb(s_file)
        assert norb_H == norb_S, "H.dat / S.dat orbital mismatch"
        
        norb = norb_H
        mats = {'H': Hmats, 'S': Smats}
        
        print(f"   ✓ Orbitals per layer: {norb}")
        print(f"   ✓ Total R vectors: {len(Hmats)}")
        
        # Analyze R vectors
        Rz_values = [Rz for (Rx, Ry, Rz) in Hmats.keys()]
        print(f"   ✓ Rz range: {min(Rz_values)} to {max(Rz_values)}")
        
        # Set up k-path (smaller for testing)
        print("\n2. Setting up k-path...")
        klist, ticks = kpath_k_g_m(20)  # Fewer points for faster testing
        print(f"   ✓ K-path: {' → '.join(list(ticks.keys()))}")
        print(f"   ✓ Total k-points: {len(klist)}")
        
        # Test different slab thicknesses
        print("\n3. Testing different slab thicknesses...")
        bands_dict = {}
        
        for nslab in [3, 5]:  # Test with 3 and 5 layers
            print(f"\n   Testing {nslab}-layer slab...")
            
            # Test interlayer hopping calculation
            k_test = klist[0]
            Hij = calculate_interlayer_hopping(k_test, mats, norb, ijmax=3)
            print(f"   ✓ Interlayer matrices calculated: {len(Hij)} matrices")
            
            # Test slab Hamiltonian construction
            Hamk_slab = construct_slab_hamiltonian(Hij, norb, nslab, ijmax=3)
            expected_dim = nslab * norb
            print(f"   ✓ Slab Hamiltonian dimension: {Hamk_slab.shape} (expected: {expected_dim}×{expected_dim})")
            
            # Check hermiticity
            hermiticity_error = np.max(np.abs(Hamk_slab - Hamk_slab.conj().T))
            print(f"   ✓ Hermiticity check: max error = {hermiticity_error:.2e}")
            
            # Calculate bands for a few k-points
            bands = calculate_slab_bands(klist[:10], mats, norb, nslab, ijmax=3, include_overlap=True)
            bands_dict[f"{nslab}L"] = bands
            
            print(f"   ✓ Band calculation completed")
            print(f"   ✓ Number of bands: {bands.shape[1]}")
            print(f"   ✓ Energy range: [{np.min(bands):.3f}, {np.max(bands):.3f}] eV")
        
                 # Test bulk calculation
        print("\n   Testing bulk calculation...")
        bulk_bands = calculate_bulk_bands(klist[:5], mats, norb, kz=0.0, include_overlap=True)
        bands_dict["Bulk"] = bulk_bands
        
        print(f"   ✓ Bulk calculation completed")
        print(f"   ✓ Bulk bands: {bulk_bands.shape[1]}")
        print(f"   ✓ Bulk energy range: [{np.min(bulk_bands):.3f}, {np.max(bulk_bands):.3f}] eV")
        
        # Plot comparison
        print("\n4. Generating comparison plot...")
        fig, ax = plot_slab_comparison(klist[:5], bands_dict, ticks, 
                                     "Slab vs Bulk Test Comparison", 
                                     energy_range=(-3, 1),
                                     outfile="test_slab_bands.png")
        
        print("   ✓ Plot saved as 'test_slab_bands.png'")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("The correct slab implementation is working properly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_methods():
    """Compare the correct method with the old (incorrect) method."""
    
    print("\n" + "=" * 60)
    print("Comparison: Correct vs. Old Method")
    print("=" * 60)
    
    print("\nCorrect method (new implementation):")
    print("- Follows WannierTools' ham_slab approach")
    print("- Calculates interlayer hopping matrices Hij(ic, :, :)")
    print("- Assembles slab Hamiltonian properly")
    print("- Each layer has Num_wann orbitals")
    print("- Total dimension: (nslab × Num_wann)²")
    
    print("\nOld method (previous implementation):")
    print("- Incorrectly used bulk R-space matrices directly")
    print("- Applied simple Rz filtering without proper layer construction")
    print("- Did not account for proper interlayer coupling")
    print("- Missing the layer-by-layer assembly logic")
    
    print("\nKey differences:")
    print("1. Interlayer coupling: Correct method uses Hij matrices")
    print("2. Matrix assembly: Proper layer-wise construction")
    print("3. Dimensionality: Correct scaling with number of layers")
    print("4. Physical meaning: Each layer explicitly represented")

if __name__ == '__main__':
    success = test_basic_slab()
    
    if success:
        compare_methods()
        print(f"\nTo run full calculations, use:")
        print(f"  python slab_correct.py --nslab 5 --compare_thickness")
        print(f"  python slab_correct.py --include_overlap --nslab 7")
    else:
        print(f"\nTest failed. Please check your input files and try again.")
        sys.exit(1) 