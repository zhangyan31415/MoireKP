# Shared H/S processing

All three backend `tapw prepare-hs` commands produce raw H(R)/S(R).
To add symmetry-averaged matrices without replacing that pair, pass
`--symmetrize` during preparation. To process an already imported directory:

```bash
python scripts/common/symmetrize_hs.py \
  --input /absolute/prepared --output /absolute/new/symmetry \
  --format both
```

The output records the structural operations and measured matrix residuals.
`--cutoff` applies only to symmetry outputs; its default is zero.
`--symprec` is in Å. For spinful input supply actual per-site
`--magnetic-moments` or explicitly mark a nonmagnetic calculation with
`--assume-nonmagnetic`.

See the [English guide](../README.md) or [中文指南](../README.zh.md).
