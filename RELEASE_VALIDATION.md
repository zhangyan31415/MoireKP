# Release validation

Run from the repository root with the project environment active:

```bash
bash scripts/release_gate.sh
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
python scripts/release/build_distribution.py --output dist
```

The build verifies current wheel/source equality and rejects generated outputs, paper files and author tools in the source archive. Install the wheel in a separate environment and check `tapw --help`, `kp --help` and a small fixture outside the checkout.

The Python CLI boundary reports real exit codes. Release-facing code must not contain unimplemented sentinels or unit placeholder overrides.

`bash scripts/release_gate.sh --final` additionally checks public data metadata. External dataset licenses, DOI and data URL remain recorded in `examples/data-manifest.yaml` and `RELEASE_BLOCKERS.md` until publication. Software checks and local reproduction checks can run before those metadata are supplied.
