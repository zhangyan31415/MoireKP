# moirekp

A Python package for moiré physics calculations.

## Overview

`moirekp` provides tools for studying moiré systems through two approaches:

1. **TAPW (Twisted Angle Plane Wave)**: Ab initio calculations for twisted bilayer systems
2. **Continuum Model**: Effective low-energy descriptions of moiré systems (in development)

## Features

### TAPW Module
- Band structure calculation for twisted bilayer systems
- Support for various valleys (K, K', Γ, M points)
- C3 symmetry consideration
- GPU acceleration support
- Parallel computation capabilities
- Chern number calculations

### Continuum Model Module
- Low-energy effective Hamiltonians for moiré systems
- Continuum Model for moiré systems
- (More features coming soon...)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:zhangyan31415/moirekp.git
cd moirekp
```

2. Create a conda environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate moirekp
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### TAPW Calculations

1. Prepare configuration:
```bash
tapw-config -o output_dir
```

2. Edit configuration files and set paths to DFT data files.

3. Run calculation:
```bash
cd output_dir
tapw-calc --config config.yaml
```

4. Plot results:
```bash
tapw-plot --config bands.yaml
```

### Continuum Model
(Documentation coming soon...)

## Examples

See the `examples/` directory for sample calculations.

## License

MIT License 