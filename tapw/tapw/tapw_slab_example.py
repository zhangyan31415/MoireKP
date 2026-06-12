"""
TAPW Slab Calculation Example
============================
This example demonstrates how to use the TAPW slab calculator for twisted materials.

Usage:
    python tapw_slab_example.py --config slab_config.yaml

The example shows:
1. Setting up slab calculation parameters
2. Running ribbon band structure calculation
3. Analyzing surface states
4. Plotting results with surface character
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import TAPW modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tapw_slab import TAPWSlab
from config import Config
from read_pos_01 import StructureProcessor
from read_kpath_01 import KPathGenerator
from read_hr_01 import HrSparseHandler


def create_example_slab_config():
    """Create example configuration for slab calculation"""
    # Use the existing config.yaml and modify mode to slab
    config = Config.from_yaml('config.yaml')
    config.compute.mode = "slab"
    config.compute.valleys = [1]  # K valley
    
    # Ensure slab config exists
    if config.slab is None:
        from config import SlabConfig
        config.slab = SlabConfig()
    
    return config


def setup_kpath_for_ribbon():
    """Create k-path for 010 slab calculation (parallel directions kx, kz)"""
    # For 010 slab: scan along kx while keeping ky=0, kz=0
    # This gives a 1D cut through the 2D parallel space
    nseg = 100
    kx_points = np.linspace(-0.5, 0.5, nseg)
    kpoints = np.array([[kx, 0.0, 0.0] for kx in kx_points])  # [kx, ky=0, kz=0]
    
    # Define high-symmetry points for kx direction
    ticks = {
        "Γ": 0,
        "X": nseg // 2,  # X point in kx direction  
        "Γ": nseg - 1
    }
    
    return kpoints, ticks


def run_tapw_slab_calculation(structure_file, hamiltonian_file, overlap_file, 
                             output_path, config=None):
    """
    Run complete TAPW slab calculation
    
    Args:
        structure_file: Path to structure file (POSCAR, etc.)
        hamiltonian_file: Path to Hamiltonian file (hr_supercell.dat)
        overlap_file: Path to overlap file (sr_supercell.dat)
        output_path: Output directory path
        config: ComputeConfig object (optional)
    """
    
    # Use default config if not provided
    if config is None:
        config = create_example_slab_config()
    
    print("=== TAPW Slab Calculation ===")
    print(f"Structure file: {structure_file}")
    print(f"Hamiltonian file: {hamiltonian_file}")
    print(f"Overlap file: {overlap_file}")
    print(f"Output path: {output_path}")
    print(f"Slab layers: {config.nslab}")
    print(f"Valley: {config.valley}")
    print(f"n_g: {config.n_g}")
    print()
    
    # Read structure
    print("Reading structure...")
    structure = StructureProcessor(structure_file, config)
    structure.process_structure()
    
    # Read Hamiltonian and overlap matrices
    print("Reading Hamiltonian and overlap matrices...")
    ham_reader = HamiltonianReader(hamiltonian_file, overlap_file, config)
    hr_supercell, sr_supercell = ham_reader.read_hamiltonian()
    
    # Setup k-path for ribbon calculation
    print("Setting up k-path for ribbon calculation...")
    kpoints, ticks = setup_kpath_for_ribbon()
    kpath_config = type('KPathConfig', (), {
        'kpoints': kpoints,
        'ticks': ticks
    })()
    
    # Initialize TAPW slab calculator
    print("Initializing TAPW slab calculator...")
    slab_calc = TAPWSlab(hr_supercell, sr_supercell, structure, config, kpath_config)
    
    # Calculate ribbon bands
    print("Calculating ribbon band structure...")
    slab_calc.calculate_ribbon_bands(output_path, kpoints)
    
    # Plot results
    print("Plotting results...")
    fig, ax = slab_calc.plot_ribbon_bands(
        energy_range=(-1.5, 1.5),
        outfile=os.path.join(output_path, f"slab_bands_{config.nslab}layers.png"),
        show_surface=True
    )
    
    # Additional analysis
    if config.analyze_surface and 'surface_weights' in slab_calc.slab_results:
        analyze_surface_states(slab_calc.slab_results, output_path, config)
    
    print("TAPW slab calculation completed!")
    return slab_calc


def analyze_surface_states(slab_results, output_path, config):
    """Analyze and save surface state information"""
    
    surface_weights = slab_results['surface_weights']
    bands = slab_results['bands']
    
    # Calculate average surface character for each band
    avg_total_surf = np.mean(surface_weights['total'], axis=0)
    avg_top_surf = np.mean(surface_weights['top'], axis=0)
    avg_bottom_surf = np.mean(surface_weights['bottom'], axis=0)
    avg_surface_index = np.mean(surface_weights['surface_index'], axis=0)
    
    # Identify surface states
    surface_mask = avg_total_surf > config.surface_threshold
    surface_bands = np.where(surface_mask)[0]
    
    # Classify surface states
    top_surface_mask = (surface_mask & (avg_surface_index > 0.3))
    bottom_surface_mask = (surface_mask & (avg_surface_index < -0.3))
    mixed_surface_mask = (surface_mask & 
                         (np.abs(avg_surface_index) <= 0.3))
    
    n_top = np.sum(top_surface_mask)
    n_bottom = np.sum(bottom_surface_mask)
    n_mixed = np.sum(mixed_surface_mask)
    
    print(f"\n=== Surface State Analysis ===")
    print(f"Total surface states: {len(surface_bands)}")
    print(f"Top surface states: {n_top}")
    print(f"Bottom surface states: {n_bottom}")
    print(f"Mixed surface states: {n_mixed}")
    
    # Save analysis results
    analysis_file = os.path.join(output_path, "slab_data", "surface_analysis.txt")
    with open(analysis_file, 'w') as f:
        f.write("TAPW Slab Surface State Analysis\n")
        f.write("================================\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Slab layers: {config.nslab}\n")
        f.write(f"  Valley: {config.valley}\n")
        f.write(f"  n_g: {config.n_g}\n")
        f.write(f"  Surface threshold: {config.surface_threshold}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Total bands: {len(avg_total_surf)}\n")
        f.write(f"  Surface states: {len(surface_bands)}\n")
        f.write(f"  Top surface: {n_top}\n")
        f.write(f"  Bottom surface: {n_bottom}\n")
        f.write(f"  Mixed surface: {n_mixed}\n\n")
        
        f.write("Band-by-band analysis:\n")
        f.write("Band  Total_Surf  Top_Surf  Bottom_Surf  Surf_Index  Type\n")
        f.write("-" * 60 + "\n")
        
        for i, (total, top, bottom, index) in enumerate(zip(
            avg_total_surf, avg_top_surf, avg_bottom_surf, avg_surface_index)):
            
            if total > config.surface_threshold:
                if index > 0.3:
                    state_type = "Top"
                elif index < -0.3:
                    state_type = "Bottom"
                else:
                    state_type = "Mixed"
            else:
                state_type = "Bulk"
            
            f.write(f"{i:4d}  {total:8.4f}  {top:8.4f}  {bottom:8.4f}  "
                   f"{index:8.4f}  {state_type}\n")
    
    print(f"Surface analysis saved to {analysis_file}")


def plot_comparison_with_bulk(slab_calc, bulk_bands=None, output_path=None):
    """Plot slab bands comparison with bulk (if available)"""
    
    if bulk_bands is None:
        print("No bulk bands provided for comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot slab bands
    slab_calc.plot_ribbon_bands(energy_range=(-1.5, 1.5), show_surface=True)
    ax1 = plt.gca()
    ax1.set_title(f'TAPW Slab ({slab_calc.nslab} layers)')
    
    # Plot bulk bands
    kpoints = slab_calc.slab_results['kpoints']
    x = np.arange(len(kpoints))
    
    for b in range(bulk_bands.shape[1]):
        band = bulk_bands[:, b]
        ax2.plot(x, band, 'k-', alpha=0.7, linewidth=0.8)
    
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_ylabel('Energy (eV)')
    ax2.set_title('Bulk Bands')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(os.path.join(output_path, "slab_bulk_comparison.png"), 
                   dpi=300, bbox_inches='tight')
    
    return fig


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='TAPW Slab Calculation Example')
    parser.add_argument('--structure', required=True,
                       help='Structure file (POSCAR, etc.)')
    parser.add_argument('--hamiltonian', required=True,
                       help='Hamiltonian file (hr_supercell.dat)')
    parser.add_argument('--overlap', required=True,
                       help='Overlap file (sr_supercell.dat)')
    parser.add_argument('--output', default='./slab_results',
                       help='Output directory')
    parser.add_argument('--nslab', type=int, default=7,
                       help='Number of layers in slab')
    parser.add_argument('--valley', type=int, default=1,
                       help='Valley index (1=K, 2=K\')')
    parser.add_argument('--n_g', type=int, default=3,
                       help='Number of G vectors')
    parser.add_argument('--energy_range', nargs=2, type=float, default=[-1.5, 1.5],
                       help='Energy range for plotting [min, max] in eV')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_example_slab_config()
    config.nslab = args.nslab
    config.valley = args.valley
    config.n_g = args.n_g
    
    # Run calculation
    slab_calc = run_tapw_slab_calculation(
        args.structure, args.hamiltonian, args.overlap, 
        args.output, config
    )
    
    print(f"\nResults saved to: {args.output}")
    print("Files created:")
    print(f"  - slab_bands_{args.nslab}layers.png")
    print(f"  - slab_data/slab_bands_{args.nslab}layers_K1.txt")
    print(f"  - slab_data/surface_weights_{args.nslab}layers_K1.npy")
    print(f"  - slab_data/surface_analysis.txt")


if __name__ == '__main__':
    main() 