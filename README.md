# moirekp

`moirekp` provides Python workflows for moire electronic-structure studies:

- `tapw`: truncated atomic plane-wave calculations and post-processing for twisted bilayer systems.
- `kp`: continuum-model construction, symmetry projection, and model validation utilities.

The repository is packaged as a single editable Python project with command-line entry points for both modules.

## Installation

Create the release environment from the repository root:

```bash
conda env create -f environment.yml
conda activate moirekp
```

The environment file installs the package in editable mode with `python -m pip install -e .`.

## Quick Checks

```bash
python -m pytest tests/test_release_contract.py -q
tapw --help
tapw init --help
kp --help
```

## TAPW Workflow

Generate starter configuration files:

```bash
tapw init -o output_dir
```

Edit `output_dir/config.yaml` and `output_dir/bands.yaml` to point to the required OpenMX-derived input files, then run:

```bash
cd output_dir
tapw run --config config.yaml
tapw plot --config bands.yaml
```

For topology post-processing, run `tapw run` with a Chern configuration and then use `tapw topo` in the generated `Q_shell_*` directory.

Legacy TAPW aliases remain available for existing scripts: `tapw-calc`, `tapw-config`, `tapw-plot`, `tapw-chernpost`, `tapw-orbital`, and `tapw-plot-orbital`.

## Continuum-Model Workflow

The `kp` command works with YAML source and model configurations under `examples/<material>_<angle>/kp/configs/`.

Typical operations are:

```bash
kp plot --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml
kp project --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml
kp symm --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml
```

See `examples/README.md` for the release-facing example layout and `examples/data-manifest.yaml` for the current dataset provenance status.

## Release Metadata

Release license, DOI, and public data URL metadata are not finalized in this checkout. Track those items in `RELEASE_BLOCKERS.md` before publishing an archival release.
