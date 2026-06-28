# MoireKP Examples

This directory contains release-facing examples grouped by material and twist
angle. The repository tracks configuration files and small template inputs; TAPW
arrays, OpenMX matrices, projected Heff files, and paper-scale outputs are
external unless `examples/data-manifest.yaml` explicitly lists them as tracked
release data.

## Command Classes

- Clean-clone smoke: runs from a fresh checkout without external scientific
  data. These commands validate packaging, metadata, CLI availability, and the
  tiny synthetic example.
- External-data scientific examples: require TAPW/OpenMX arrays or symmetry
  exports listed in `examples/data-manifest.yaml`.
- Paper reproduction: requires the finalized public data archive, DOI, license,
  checksums, and the full reproduction instructions for the CPC submission.

## Clean-Clone Smoke

```bash
python -m pytest tests/test_release_contract.py tests/kp/test_example_dependency_contract.py -q  # clean-clone
tapw --help  # clean-clone
tapw init --help  # clean-clone
kp --help  # clean-clone
```

Clean-clone smoke checks intentionally do not include synthetic scientific
examples. Release-facing scientific examples are tied to the external datasets
listed in `examples/data-manifest.yaml`.

## Canonical Layout

```text
examples/<material>_<angle>/
  openmx/
  tapw/
    configs/
      K1_q06.yaml
      K1_up_q06.yaml
    outputs/
      manifest.json
      K1/
        q06/
          manifest.json
          band/
          symmetry/
          topology/
  kp/
    configs/
      K1_q06.yaml
      source/
      model/
        reference/
        diagnostics/
    notebooks/
    outputs/
      plot/<case_id>/
      project/<case_id>/
      symm/<case_id>/
      model/
        <case_id>/
        reference/<case_id>/
        diagnostics/<case_id>/
```

`case_id` uses `<material>_<angle>_<valley>`, for example
`mote2_3.89_K1`, `mgi2_3.89_Gamma`, or `mgi2_3.89_M1`.

For TAPW, use one config per material/angle/valley/q-shell case. The default
spinful profile omits a spin suffix, for example `tapw/configs/K1_q06.yaml`
writes `tapw/outputs/K1/q06`. Non-default spin profiles must be explicit:
`K1_up/q06`, `K1_down/q06`, or `K1_spinless/q06`.

## Config Roles

- `configs/source/<case_id>.yaml`: shared input for `kp show`, `kp proj`,
  and `kp symm`.
- `configs/K1_q06.yaml`: preferred single-case config for `kp inspect`,
  `kp project`, `kp symm`, and `kp model`.
- `configs/model/<case_id>.yaml`: canonical continuum-model configuration.
  Production configs consume validated symmetry/action metadata from `kp symm`
  outputs.
- `configs/model/reference/<case_id>.yaml`: notebook, toy, or legacy comparison
  paths.
- `configs/model/diagnostics/<case_id>_<tag>.yaml`: diagnostic experiments that
  are not the active release path.

## Active GMK Cases

- MoTe2 K:
  - `examples/mote2_3.89/kp/configs/source/mote2_3.89_K1.yaml`
  - `examples/mote2_3.89/kp/configs/model/mote2_3.89_K1.yaml`
  - `examples/mote2_3.89/kp/configs/source/mote2_3.89_K1_spinful.yaml`
  - `examples/mote2_3.89/kp/configs/model/mote2_3.89_K1_spinful.yaml`
- MgI2 Gamma:
  - `examples/mgi2_3.89/kp/configs/source/mgi2_3.89_Gamma.yaml`
  - `examples/mgi2_3.89/kp/configs/model/mgi2_3.89_Gamma.yaml`
- MgI2 M:
  - `examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1.yaml`
  - `examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1.yaml`
  - `examples/mgi2_3.89/kp/configs/source/mgi2_3.89_M1_spinful.yaml`
  - `examples/mgi2_3.89/kp/configs/model/mgi2_3.89_M1_spinful.yaml`

## External-Data Scientific Examples

The active KP workflow is documented as a dependency DAG rather than as a
clean-clone test:

- `tapw run -c examples/<case>/tapw/configs/K1_q06.yaml` (external-data: writes canonical TAPW band outputs)
- `tapw symm -c examples/<case>/tapw/configs/K1_q06.yaml` (external-data: writes canonical TAPW raw-H symmetry outputs)
- `tapw chern -c examples/<case>/tapw/configs/K1_q06.yaml` (external-data: writes canonical topology outputs)
- `kp inspect -c examples/<case>/kp/configs/K1_q06.yaml` (external-data: inspect source bands before selecting low states)
- `kp show -c examples/<case>/kp/configs/source/<case_id>.yaml` (external-data: consumes TAPW band and Q arrays)
- `kp proj -c examples/<case>/kp/configs/source/<case_id>.yaml` (external-data: consumes TAPW band and Q arrays; produces projected Heff)
- `kp symm -c examples/<case>/kp/configs/source/<case_id>.yaml` (external-data: consumes TAPW symmetry-analysis exports)
- `kp fit -c examples/<case>/kp/configs/model/<case_id>.yaml` (precomputed: consumes `kp proj` and `kp symm` outputs)
- `kp export -c examples/<case>/kp/configs/model/<case_id>.yaml -o exported/<case_id>` (precomputed: optionally writes a standalone package from the model output)

New source configs should use `project.gauge: auto` with an explicit
`project.nlow_state_list`. Existing hand-written `project.norb_fix_list` entries
remain valid as expert overrides, but they should not be required for ordinary
finite-basis projection. Auto-gauge reports are written next to `heff_list.npy`
as `basis_selection.json` and `basis_selection.md`.

When TAPW outputs use the canonical layout, KP source configs should prefer the
band manifest instead of repeating every TAPW filename:

```yaml
material:
  tapw_band_manifest: ../../../tapw/outputs/K1/q06/band/manifest.json
```

If the same source config has a TAPW raw-H symmetry source in `symm`, `kp proj`
and `kp symm` both use the symmetry-scored auto-gauge resolver. The report then
lists all gauge candidates and rejected residuals. Auto gauge still does not
choose `nlow_state_list`; it only fixes the gauge of the low subspace the user
already selected.

Dataset provenance and unresolved release metadata are tracked in
`examples/data-manifest.yaml`. The release cannot be archived until every
required external file has a public URL, license, DOI, checksum, runtime class,
and expected-output description.

## TAPW Templates

- `examples/tapw/basic/`: small TAPW configuration templates.
- `examples/tapw/kpaths/`: canonical hexagonal KMGMK K-path input.
- `examples/tapw/mote2_9.43/`: MoTe2 rigid-OpenMX TAPW and direct
  diagonalization configs.
- `examples/tapw/mgi2_9.43/`: MgI2 rigid-OpenMX TAPW and direct
  diagonalization configs.

The `tapw init` command uses package templates under `tapw/tapw/templates/`.
Those templates are not material-specific examples.

## Paper Reproduction

Paper-scale reproduction is intentionally separated from clean-clone tests. It
requires the finalized public archive named in `examples/data-manifest.yaml` and
the release checklist in `RELEASE_BLOCKERS.md` to be resolved. Until then, the
repository should be treated as a pre-release checkout.

## Historical Artifacts

Older local runs may exist in developer worktrees, but active README commands
and active YAML configs must not use historical `kp/runs/...` or production
scratch directories as command targets.
