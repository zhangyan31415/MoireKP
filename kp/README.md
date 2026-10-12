# kp package

The `kp` package projects TAPW Hamiltonians and symmetry actions into selected
low-energy spaces, constructs moiré continuum models, and exports standalone NumPy
evaluators.

## Installation

Install from the repository root:

```bash
conda env create -f environment.yml
conda activate moirekp
```

For an existing compatible environment, editable installation is also
available with `pip install -e .`.

## Quick check

The CLI can be checked without external TAPW arrays:

```bash
python -c "import kp"
kp --help
```

## Workflow

Example configs are under the repository-level `examples/` directory. A
complete KP workflow is:

```bash
CFG=examples/mote2_3.89/kp/kp_K.yaml
kp inspect  -c "$CFG"  # optional band plots
kp project  -c "$CFG"
kp symm     -c "$CFG"
kp model    -c "$CFG"
kp symm-rep -c "$CFG"  # optional band representations
```

These commands have distinct roles:

- `kp inspect` plots the TAPW bands and reference Q-block spectrum.
- `kp project` selects and downfolds the low-energy subspace and writes Heff.
- `kp symm` projects and exactifies the symmetry generators required by the
  continuum model.
- `kp model` selects or loads harmonics, determines the model coefficients, and
  exports the standalone evaluator.
- `kp symm-rep` optionally analyzes high-symmetry band blocks after
  `kp project` and `kp symm`.

The [paper examples](../examples/README.md) provide first-principles inputs and
commands to regenerate the large H/S arrays, run TAPW, and construct the six
models. The arrays are generated locally.

## Band representations

`kp symm-rep` reads `projection/heff.npy`, `projection/kpoints.npy`, and
`symmetry/representations.npz`. It checks that these inputs belong to the same
projection, uses the saved k-points to determine the little group and
reciprocal sewing shift, and writes:

```text
symm_rep/
  summary.md
  bands.csv
  characters.csv
  high_symmetry_wavefunctions.npz
  band_representations.npz
```

The output includes each projected block, its unitarity residual, and the
change introduced by polar decomposition.

An optional top-level `symm_rep` section may override `points`,
`valence_count`, `conduction_count`, `degeneracy_tol`, or `output_dir`.
Otherwise the command uses `project.efermi` and the configured `bands.kpath`.

## Model export

`kp model` writes directly to the `model/` directory. The standalone files
include:

```text
model/
  README.md
  MODEL.md
  evaluate.py
  physical_model.py
  model_data.npz
  model.toml
  symmetry.toml
  terms.csv
```

There is no separate `kp export` command.

## Package layout

- `kp/io`: TAPW Hamiltonian and Q-set loading
- `kp/blocks`: Q-block assembly and low-energy projection
- `kp/symmetry`: projected symmetry actions and exactification
- `kp/model`: continuum-model construction, coefficient extraction, and export
- `kp/viz`: plotting utilities
