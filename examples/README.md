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
      K1/
        q06/
          band/
          symmetry/
          topology/
  kp/
    configs/
      K1_q06.yaml
    outputs/
      K1/
        q06/
          inspect/
          projection/
          symmetry/
          model/
```

For TAPW, use one config per material/angle/valley/q-shell case. The default
spinful profile omits a spin suffix, for example `tapw/configs/K1_q06.yaml`
writes `tapw/outputs/K1/q06`. Non-default spin profiles must be explicit:
`K1_up/q06`, `K1_down/q06`, or `K1_spinless/q06`.

## Config Roles

KP release examples use one YAML file per material/angle/valley/q-shell case.
The same file drives `kp inspect`, `kp project`, `kp symm`, and `kp model`.
Do not split release examples into separate source and model subdirectories.

## Active GMK Cases

- MoTe2 K:
  - `examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml`
  - `examples/mote2_3.89/kp/configs/mote2_3.89_K1_up_q06.yaml`
- MgI2 Gamma:
  - `examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_q05.yaml`
- MgI2 M:
  - `examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_q07.yaml`
  - `examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_spinless_q07.yaml`

## External-Data Scientific Examples

The active KP workflow is documented as a dependency DAG rather than as a
clean-clone test:

- `tapw run -c examples/<case>/tapw/configs/K1_q06.yaml` (external-data: writes canonical TAPW band outputs)
- `tapw symm -c examples/<case>/tapw/configs/K1_q06.yaml` (external-data: writes canonical TAPW raw-H symmetry outputs)
- `tapw topo -c examples/<case>/tapw/configs/K1_q06.yaml` (external-data: writes canonical topology outputs)
- `kp project -c examples/<case>/kp/configs/K1_q06.yaml` (external-data: automatically selects the low-energy subspace, consumes TAPW band, Q, and raw-H symmetry outputs, and produces projected Heff plus exactified continuum symmetry)
- `kp model -c examples/<case>/kp/configs/K1_q06.yaml` (external-data: consumes projection and symmetry outputs)

New KP case configs should omit `project.nlow_state_list`; `kp project`
automatically selects the smallest acceptable symmetry-compatible low-energy
subspace and reports its band, layer, spin, and orbital content. Experts may
write `nlow_state_list` only to reuse an already reviewed subspace and skip the
search. The projection reference belongs in `project.e_ref`.

For symmetry projection, KP reads TAPW raw-H symmetry output inferred from
`system.tapw_output`, `project.valley`, and `project.q_shell`. Users do not list
symmetry operations or matrices in the KP case config; the operation set comes
from the TAPW symmetry export.

Dataset provenance and unresolved release metadata are tracked in
`examples/data-manifest.yaml`. The release cannot be archived until every
required external file has a public URL, license, DOI, checksum, runtime class,
and expected-output description.

## TAPW Templates

- `examples/tapw/basic/`: small TAPW configuration templates.
- `examples/tapw/kpaths/`: canonical hexagonal KMGMK K-path input.
- `examples/tapw/mote2_9.43/`: MoTe2 rigid-OpenMX TAPW configs.
- `examples/tapw/mgi2_9.43/`: MgI2 rigid-OpenMX TAPW configs.

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
