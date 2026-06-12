# TAPW Slab Tools

This directory contains tools for calculating slab/ribbon band structures using the TAPW (Twisted Atomic Plane Wave) method.

## Overview

The TAPW slab method extends the standard slab construction approach to work with twisted materials by applying the TAPW transformation to both intra-layer and inter-layer terms. This is essential for twisted systems where the standard Wannier-based approach may not capture the physics correctly.

## Key Features

- **TAPW-transformed slab Hamiltonian**: Applies G-vector basis transformation to slab matrices
- **010 slab geometry**: Default slab cut in y-direction (b2 lattice vector)
- **Twisted material optimization**: ijmax=1 for nearest-layer coupling only
- **Correct Fourier transform**: Only in parallel directions (x,z), not slab direction (y)
- **Dedicated slab k-path system**: KPATH_SLAB.in/out files for 2D k-paths
- **Surface state analysis**: Automatically identifies and classifies surface states
- **Electric field support**: Includes electric field corrections for gated structures  
- **C3 symmetry**: Optional C3 rotational symmetry enforcement
- **Non-orthogonal basis**: Full support for overlap matrices

## Method Comparison

### Standard Slab Method (slab_correct.py)

The standard approach follows WannierTools methodology:

1. **Direct real-space construction**: Build interlayer hopping matrices `Hij(ic)` directly from `H(R)` matrices
2. **Layer-by-layer assembly**: Construct slab Hamiltonian by assembling layers with open boundaries
3. **Wannier basis**: Works in the original Wannier orbital basis

```python
# Standard approach
Hij = calculate_interlayer_hopping(k2d, mats3d, norb, ijmax)
Hamk_slab = construct_slab_hamiltonian(Hij, norb, nslab, ijmax)
eigenvals = solve_slab_bands(Hamk_slab, Sk_slab)
```

### TAPW Slab Method (tapw_slab.py)

The TAPW approach adds an additional transformation step:

1. **Full k-space construction**: First build complete k-space Hamiltonian from `H(R)`
2. **TAPW transformation**: Apply `G * H(k) * G†` transformation to get effective matrices
3. **Extract interlayer blocks**: Extract layer-to-layer coupling from transformed matrices
4. **Slab assembly**: Assemble transformed blocks into slab Hamiltonian

```python
# TAPW approach  
mk_full = build_kspace_hamiltonian(Hr, k2d)
mk_tapw_full = apply_tapw_transformation(mk_full)
Hij_tapw = extract_interlayer_blocks(mk_tapw_full)
Hamk_slab = construct_tapw_slab_hamiltonian(Hij_tapw)
eigenvals = solve_tapw_slab_bands(Hamk_slab, Sk_slab)
```

## Why TAPW for Slabs?

For twisted materials, the TAPW method offers several advantages:

1. **Correct valley physics**: Properly handles K/K' valley structure in twisted systems
2. **Moiré effects**: Captures long-range moiré modulation through G-vector expansion
3. **Interlayer coupling**: More accurate treatment of twist-dependent interlayer hopping
4. **Symmetry preservation**: Maintains crystal symmetries in the effective model

## Usage

### Basic Usage

**Method 1: Using Configuration File (Recommended)**

1. Set mode to "slab" in config.yaml:
```yaml
compute:
  mode: "slab"
  valleys: [1]
  TAPW: true
  n_g: 3

slab:
  nslab: 7
  ijmax: 1
  slab_direction: "y"
  analyze_surface: true
  surface_threshold: 0.5
  kpath_slab_in: "KPATH_SLAB.in"
  kpath_slab_out: "KPATH_SLAB.out"
```

2. Create or modify KPATH_SLAB.in file:
```
4
Gamma  0.0  0.0
X      0.5  0.0
M      0.5  0.5
Gamma  0.0  0.0
40
```

3. Run calculation:
```bash
python main.py --config config.yaml
# or use the simple example:
python slab_example_simple.py --config config.yaml
```

**Method 2: Direct Programming Interface**

```python
from tapw_slab import TAPWSlab
from config import Config

# Load configuration
config = Config.from_yaml('config.yaml')
config.compute.mode = "slab"

# Initialize calculator
slab_calc = TAPWSlab(hr_supercell, sr_supercell, structure, config)

# Calculate ribbon bands
slab_calc.calculate_ribbon_bands(output_path)

# Plot with surface analysis
slab_calc.plot_ribbon_bands(show_surface=True)
```

### Command Line Usage

**Option 1: Using Main TAPW Program**
```bash
# First, set mode: "slab" in config.yaml
python main.py --config config.yaml --valleys 1
```

**Option 2: Using Simple Slab Example**
```bash
# Create a sample configuration
python slab_example_simple.py --create-config

# Run with custom config
python slab_example_simple.py --config config_slab.yaml --valleys 1
```

**Option 3: Legacy Example (requires manual setup)**
```bash
python tapw_slab_example.py \
    --structure POSCAR \
    --hamiltonian hr_supercell.dat \
    --overlap sr_supercell.dat \
    --nslab 7 \
    --valley 1 \
    --n_g 3 \
    --output ./slab_results
```

## Configuration Parameters

### Slab-Specific Parameters

- `nslab`: Number of layers in the slab (default: 7)
- `ijmax`: Maximum interlayer distance to consider (default: 1, for twisted materials)  
- `slab_direction`: Direction perpendicular to slab ('y', default for 010 cut)
- `analyze_surface`: Whether to analyze surface character (default: True)
- `surface_threshold`: Threshold for identifying surface states (default: 0.5)
- `kpath_slab_in`: Slab k-path input file (default: "KPATH_SLAB.in")
- `kpath_slab_out`: Slab k-path output file (default: "KPATH_SLAB.out")

### TAPW Parameters

- `TAPW`: Enable TAPW transformation (required: True)
- `n_g`: Number of G vectors per layer (default: 3)
- `valley`: Valley index (1=K, 2=K', etc.)
- `C3_H`: Enable C3 symmetry (default: True)

### Electric Field Parameters

- `Electric_field_in_eVpA`: External electric field in eV/Å
- `Inner_symmetrical_Electric_Field`: Enable symmetric internal field
- `zero_potential_layers`: Layers for zero potential reference

## Output Files

The calculation produces several output files:

```
slab_results/
├── slab_bands_7layers.png              # Band structure plot
├── slab_data/
│   ├── slab_bands_7layers_K1.txt       # Numerical band data
│   ├── surface_weights_7layers_K1.npy  # Surface character data
│   ├── surface_analysis.txt            # Surface state analysis
│   ├── g_vec_list_3_7layers_K1_1layer.npy  # G-vectors layer 1
│   └── g_vec_list_3_7layers_K1_2layer.npy  # G-vectors layer 2
```

## Surface State Analysis

The TAPW slab calculator automatically analyzes surface states:

- **Surface character**: Fraction of wavefunction on surface layers
- **Surface index**: +1 for top surface, -1 for bottom surface, 0 for bulk
- **Classification**: Automatic identification of top/bottom/mixed surface states

Surface states are visualized using a red-blue colormap where:
- Red: Top surface states
- Blue: Bottom surface states  
- Gray: Bulk states

## Implementation Details

### TAPW Transformation and Dimensions

**Key insight**: The TAPW transformation `G * H(k) * G†` transforms the Hamiltonian from Wannier basis to G-vector basis:

- **Input dimension**: `(num_wannier_orbs, num_wannier_orbs)` 
- **Output dimension**: `(dim_gr_1, dim_gr_1)` where `dim_gr_1 = Σ(orbs_per_layer × num_g_vectors_per_layer)`

**For twisted materials**:
- Each layer gets its own G-vector set (K1 for even layers, K2 for odd layers)
- `ijmax=1` because twisted systems have only nearest-layer coupling
- The transformation preserves interlayer coupling structure

**Layer coupling extraction**:
1. Build interlayer matrices `H_ic` for each layer separation `ic` from R-space
2. Apply TAPW transformation to each: `H_ic_tapw = G * H_ic * G†`
3. Assemble into slab Hamiltonian with open boundaries

### Memory Considerations

For large systems:
- Use sparse matrix operations where possible
- Consider reducing `n_g` for initial tests
- Monitor memory usage for large `nslab` values

### Convergence Parameters

Key parameters affecting accuracy:
- `n_g`: Higher values give better accuracy but increase computational cost
- `ijmax`: Should include all significant interlayer couplings
- `nslab`: Must be large enough to avoid finite-size effects

## Comparison with Experiments

The TAPW slab method is particularly useful for:
- **ARPES experiments**: Direct comparison with surface-sensitive measurements
- **STM/STS studies**: Local density of states at surfaces
- **Transport measurements**: Edge state contributions to conductivity

## Limitations and Future Work

Current limitations:
- Assumes layer structure is preserved by TAPW transformation
- May require large `nslab` for convergence in some systems
- Computational cost scales as O(nslab³)

Future improvements:
- Implementation of López-Sancho decimation for semi-infinite surfaces
- GPU acceleration for large systems
- Integration with Berry curvature calculations

## References

1. Standard slab method: López-Sancho et al., J. Phys. F: Met. Phys. 15, 851 (1985)
2. TAPW method: [Original TAPW paper reference]
3. WannierTools methodology: Wu et al., Comput. Phys. Commun. 224, 405 (2018) 