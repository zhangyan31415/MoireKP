# Legacy script implementations

These files support historical command lines. New OpenMX, ABACUS and SIESTA
H/S preparation uses `tapw prepare-hs BACKEND`; optional symmetry processing
uses `--symmetrize` or `scripts/common/symmetrize_hs.py`.

The original top-level `scripts/openmx_symm_hs_python.py` and
`scripts/openmx_symm_hs_scfout.py` paths remain as compatibility launchers.
Their implementations live in `openmx/` here. They write symmetry-processed
matrices directly to the working directory and have different output/overwrite
rules from the common preparation interface.
