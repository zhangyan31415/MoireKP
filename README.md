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

These checks are the clean-clone smoke surface. They do not require external
OpenMX matrices, TAPW Q-shell arrays, or precomputed KP outputs.

```bash
python -m pytest tests/test_release_contract.py tests/kp/test_example_dependency_contract.py -q  # clean-clone
tapw --help  # clean-clone
tapw init --help  # clean-clone
kp --help  # clean-clone
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
The release example commands consume external TAPW arrays or precomputed KP
outputs; clean-clone checks are limited to package metadata and CLI surfaces.

Typical operations are:

```bash
kp plot --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml  # external-data
kp project --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml  # external-data
kp symm --config examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml  # external-data
```

For new `kp project` configs, prefer `project.gauge: auto` instead of hand
writing `project.norb_fix_list`. See `docs/project_auto_gauge.md` for the
finite-basis auto-gauge anchor report and failure checks.

For `kp model`, prefer `model.fit.mode: auto_low_energy` when the target is a
low-energy continuum model. See `docs/project_auto_low_energy.md` for the
automatic band, subspace, matrix, and parameter-quality reports.

See `examples/README.md` for the release-facing example layout and `examples/data-manifest.yaml` for the current dataset provenance status.

## Release Metadata

Release license, DOI, and public data URL metadata are not finalized in this
checkout. Track those items in `RELEASE_BLOCKERS.md` before publishing an
archival release. Do not infer a license, DOI, or public dataset URL from local
paths or unpublished artifacts.

Before tagging a formal CPC archive, run:

```bash
scripts/release_gate.sh
```

This enables `MOIREKP_RELEASE_FINAL=1` and must fail until the license, DOI,
public data URL, checksums, and release blockers are resolved.
