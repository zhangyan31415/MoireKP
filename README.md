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

Edit `output_dir/config.yaml` and `output_dir/bands.yaml` to point to the required OpenMX-derived input files, then run the same config through the requested TAPW workflows:

```bash
cd output_dir
tapw run -c config.yaml
tapw symm -c config.yaml
tapw topo -c config.yaml  # when topology settings are present
tapw plot -c bands.yaml
```

New TAPW configs use `case.output_root` plus per-workflow sections
(`bands`, `symmetry`, `topology`) and write
`outputs/<valley>/qNN/<workflow>/`. For example, `bands.valley: K1` with
`bands.q_shell: 6` writes `outputs/K1/q06`.

TAPW symmetry writes three release files:
`symmetry/representations.npz`, `symmetry/residuals.csv`, and
`symmetry/summary.md`.
Canonical topology grid directories include the sampled reciprocal-coordinate
range, for example `grid21x21_b1_m0p5_0p5_b2_m0p5_0p5`.

## Continuum-Model Workflow

The `kp` command works with YAML source and model configurations under `examples/<material>_<angle>/kp/configs/`.
The release example commands consume external TAPW arrays or precomputed KP
outputs; clean-clone checks are limited to package metadata and CLI surfaces.

New KP configs should use one case file for inspect, projection, symmetry, and
model fitting:

```text
kp/
  configs/K1_q06.yaml
  outputs/K1/q06/
    inspect/
    projection/
    symmetry/
    model/
```

Typical operations are:

```bash
kp inspect -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml  # external-data
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml  # external-data
kp symm    -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml  # external-data
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml  # external-data
```

`kp model` writes the standalone evaluator and data directly into
`kp/outputs/<profile>/<q_shell>/model/` as part of the release workflow.

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
