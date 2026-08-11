# kp package

The `kp` package projects TAPW Hamiltonians and symmetry actions into selected
low-energy spaces, fits moiré continuum models, and exports standalone NumPy
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
CFG=examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_q06.yaml
kp inspect  -c "$CFG"  # optional source-spectrum diagnostic
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
- `kp model` selects or loads harmonic support, fits the configured model, and
  exports the standalone evaluator.
- `kp symm-rep` optionally analyzes high-symmetry band blocks after
  `kp project` and `kp symm`.

The scientific runs require external arrays that are not committed to the
source repository. `examples/README.md` describes the tracked case configs and
automatic low-energy and harmonic selection.

## Band representations

`kp symm-rep` reads `projection/heff.npy`, `projection/kpoints.npy`, and
`symmetry/representations.npz`. It validates their shared artifact identity,
uses the persisted projected k-points to determine the little group and
reciprocal sewing shift, and writes:

```text
symm_rep/
  summary.md
  bands.csv
  characters.csv
  high_symmetry_wavefunctions.npz
  band_representations.npz
```

The output preserves each raw projected block, its raw unitarity residual, and
the distance to the reported polar-unitary block. A poor raw block remains
marked as such; polar decomposition is not a replacement for the diagnostic.

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
- `kp/model`: continuum-model construction, fitting, and export
- `kp/viz`: plotting utilities
