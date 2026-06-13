# MoireKP CPC Release Validation

Date: 2026-06-13

This file records the release gate definition and current release blockers. It
does not include local node names, private paths, transient validation outputs,
or historical command results.

## Required Gate

Run these commands from a clean checkout with the release environment active:

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
python -m build
kp --help
tapw --help
MOIREKP_RELEASE_FINAL=1 scripts/release_gate.sh
```

Targeted CPC hardening checks:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_action_schema.py tests/kp/test_symm_projection.py tests/kp/test_exactify_representation.py
python -m pytest -q -p no:cacheprovider tests/tapw/test_c3_symm_valleys.py tests/tapw/test_tapw_chern_workflow.py tests/tapw/test_cli_surface.py
python -m pytest -q -p no:cacheprovider tests/test_release_contract.py tests/kp/test_example_dependency_contract.py
```

## Static Gates

- Release-facing code must not contain unimplemented sentinels, unreported hard process-exit calls, or unit placeholder overrides.
- TAPW long-calculation exit paths must report the real shell status without bypassing the Python CLI boundary.
- Public docs, examples, package code, and tests must not reference local work directories, review bundles, validation outputs, private node names, prompt records, or AI-conversation traces.
- Source archives must exclude generated outputs, build artifacts, review material, local validation workspaces, and paper build products.

## Current Blockers

The checkout remains pre-release until the following values are supplied and
encoded in `examples/data-manifest.yaml` and project metadata:

- final license
- CPC DOI or submission DOI metadata
- public data archive URL
- final external data checksums
- removal of `pending_external` / `pending_public_archive` markers

`scripts/release_gate.sh` is expected to fail in final mode until those items are
resolved.
