#!/usr/bin/env python3
"""
Simple TAPW Slab Calculation Example
===================================
This example shows how to run slab calculations using the configuration file.

Usage:
    1. Set mode: "slab" in config.yaml
    2. Configure slab parameters in the slab section
    3. Run: python slab_example_simple.py

The script will automatically use the slab configuration from config.yaml.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main as tapw_main

def create_slab_config_yaml(output_file="config_slab.yaml"):
    """Create a sample slab configuration YAML file"""
    
    slab_config = """# TAPW Slab Configuration Example
# Twisted Material Configuration File

# Twist parameters
twist:
  twist_index_m: 3          # Twist index m, determines the twist angle
  num_layers: 2             # Total number of layers in the structure
  type_structure: [2, 2]    # Structure type for each layer: 1 for graphene-like, 2 for TMD-like
  twist_layer: [1, 1]       # Which layers are twisted relative to each other
  spin: true               # Whether to include spin in calculations

# File paths and IO configuration
paths:
  H_file: "H.dat"   # Hamiltonian file, can be .dat or .npz
  S_file: "S.dat"   # Overlap file, can be .dat or .npz
  input_file: "openmx.dat"   # Input structure file name
  output_dir: "slab_output"   # Output directory for results
  kpath_in: "KPATH.in"   # K-path input file
  kpath_out: "KPATH.out"   # K-path output file

# Computation parameters
compute:
  # Valley configuration
  valleys: [1]    # List of valleys to calculate (K valley for slab)
  
  # Calculation mode - SET TO SLAB
  mode: "slab"            # Slab calculation mode
  
  # Energy parameters
  efermi: -4.6           # Fermi energy level, unit: eV
  n_g: 3                  # Harmonic of G vectors
  
  # Band structure calculation parameters
  num_processes: 4       # Number of parallel processes
  num_bands_cal: 40      # Number of bands to calculate
  band_type: "CBM"       # Band type to analyze
  
  # Hardware acceleration
  gpu: false             # Whether to use GPU acceleration
  gpu_index: [0, 1]      # GPU device indices to use
  delay_time: 0          # Delay time between GPU operations
  
  # Calculation flags
  hamk_save: false       # Whether to save Hamiltonian matrices
  TAPW: true            # Use TAPW method (REQUIRED for slab)
  eigsh_cal: true       # Use eigsh for eigenvalue calculation
  C3_H: false           # Apply C3 symmetry to Hamiltonian
  ge: false             # Use generalized eigenvalue solver
  eig_vec_cal: false     # Calculate eigenvectors 

  # Electric field parameters
  Electric_field_in_eVpA: 0.000  # Electric field strength in eV/Å
  zero_potential_layers: [1]     # Layers with zero potential
  Inner_symmetrical_Electric_Field: false  # Whether to use inner symmetrical electric field

  orthogonal_basis: false

# Slab calculation parameters (ACTIVE when mode: "slab")
slab:
  nslab: 7                    # Number of layers in the slab
  ijmax: 1                    # Maximum interlayer distance (twisted materials)
  slab_direction: "y"         # Slab normal direction (010 cut)
  analyze_surface: true       # Whether to analyze surface character
  surface_threshold: 0.5      # Threshold for identifying surface states
  kpath_slab_in: "KPATH_SLAB.in"   # Slab k-path input file
  kpath_slab_out: "KPATH_SLAB.out" # Slab k-path output file
"""
    
    with open(output_file, 'w') as f:
        f.write(slab_config)
    
    print(f"Sample slab configuration saved to: {output_file}")
    print("You can modify this file and use it with --config option")
    return output_file

def main():
    """Main function for slab example"""
    parser = argparse.ArgumentParser(description='TAPW Slab Calculation Example')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file (default: config.yaml)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a sample slab configuration file')
    parser.add_argument('--nslab', type=int,
                       help='Override number of slab layers')
    parser.add_argument('--valleys', type=int, nargs='+',
                       help='Override valleys to calculate')
    
    args = parser.parse_args()
    
    if args.create_config:
        config_file = create_slab_config_yaml()
        print(f"\nTo run slab calculation:")
        print(f"python {__file__} --config {config_file}")
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file '{args.config}' not found!")
        print("Create one using: python slab_example_simple.py --create-config")
        return
    
    # Modify sys.argv to pass arguments to main TAPW program
    original_argv = sys.argv.copy()
    sys.argv = ['tapw_main', '--config', args.config, '--mode', 'slab']
    
    # Add overrides if provided
    if args.nslab:
        print(f"Note: nslab override not implemented in main program")
        print(f"Please modify the slab.nslab value in {args.config}")
    
    if args.valleys:
        sys.argv.extend(['--valleys'] + [str(v) for v in args.valleys])
    
    try:
        print("=" * 60)
        print("Starting TAPW Slab Calculation")
        print("=" * 60)
        print(f"Configuration file: {args.config}")
        print(f"Mode: slab")
        
        # Call the main TAPW program
        tapw_main()
        
        print("=" * 60)
        print("TAPW Slab Calculation Completed!")
        print("=" * 60)
        print("Check the output directory for results:")
        print("- slab_bands_*.png: Band structure plots")
        print("- slab_data/: Numerical data and analysis")
        
    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main() 