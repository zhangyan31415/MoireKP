# Minimal Synthetic Example

This clean-clone example contains only small text files. It exercises public KP
helpers for orbital-order loading, orbital selection, and k-path interpolation
without requiring TAPW arrays or external OpenMX data.

Run from the repository root:

```bash
python -m pytest examples/minimal_synthetic -q  # clean-clone
```

The test compares generated outputs against the files under `expected/`.
