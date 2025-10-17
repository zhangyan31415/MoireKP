kp — From TAPW to moiré k·p (developer skeleton)

Overview
- Developer-first skeleton to migrate the current Jupyter notebook workflow to a Python package.
- Mirrors the notebook structure: IO, k-path generation, block diagonalization, symmetry, continuum model build, and visualization.

Install (editable)
- pip install -e .

CLI
- Plot example (MgI2, Gamma valley):
  - Edit paths if needed in `configs/mgi2_gamma.yaml`.
  - Run: `kp plot -c configs/mgi2_gamma.yaml`
  - Output: `plots/mgi2_gamma_scatter.png`

- Project low-energy Heff (separate command):
  - Configure `project` section in config (select `nlow_state_list`, `norb_fix_list`, `workers`).
  - Run: `kp project -c configs/mgi2_gamma.yaml`
  - Outputs:
    - `plots/heff_list.npy` (Heff per Q)
    - `plots/heff_eig.npy` (Heff eigenvalues per Q)
    - `plots/heff_vec.npy` (Heff eigenvectors per Q)
    - `plots/heff_scatter.png` and `plots/heff_spectrum.txt`

Package Layout (key modules)
- kp/io: Load TAPW outputs (hamiltonians, Q-sets, orbital order)
- kp/kmesh: k-path and BZ utilities
- kp/blocks: Low-energy block assembly and diagonalization helpers
- kp/analysis: Orbital weight computation and selection strategies
- kp/symmetry: Symmetry operators and checks
- kp/model: Continuum terms, builder, orthogonalization, fitting
- kp/viz: Band plots and orbital projection diagnostics
- kp/config: Config schemas (dataclasses)

Next Steps
- Port functions from model.ipynb into corresponding modules.
- Replace hard-coded paths with config-driven inputs (see configs/example.yaml).
- Add minimal scripts to run selection, build, and plot for a target material.
