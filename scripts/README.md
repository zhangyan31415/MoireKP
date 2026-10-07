# H/S input preparation

Use `tapw prepare-hs BACKEND` to read completed DFT outputs. H(R) is written in eV and S(R) is dimensionless. The output also contains structure, orbital/spin metadata and a configuration fragment.

```bash
python -m pip install .
tapw prepare-hs --help
```

| Backend | Required input | Reader |
|---|---|---|
| [OpenMX](openmx/README.md) | `.scfout` and matching structure | Build the C++ reader once using `openmx/build_openmx_symm_hs.sh` |
| [ABACUS](abacus/README.md) | Real-space LCAO H/S CSR output | Included Python converter |
| [SIESTA](siesta/README.md) | HSX | Optional `sisl`; install `.[siesta]` |

```bash
tapw prepare-hs openmx --input work/openmx.scfout --structure work/openmx.dat \
  --binary build/openmx_import_hs/analysis_symm_hs --output prepared/openmx --format both
tapw prepare-hs abacus --input work/abacus --output prepared/abacus --format both
tapw prepare-hs siesta --input work/system.HSX --output prepared/siesta --format both
```

Use a new output directory. Add layer, twist, valley, Fermi energy and k-path settings to the generated fragment before running TAPW. Keep the relative H/S paths beside the config, or set explicit paths.

`--symmetrize` writes separate symmetry-averaged matrices. `common/symmetrize_hs.py` processes an existing prepared input. Spinful nonmagnetic symmetry processing uses `--assume-nonmagnetic`; magnetic inputs require their site moments. Details and conventions are in each backend guide.

The per-backend `prepare_hs.py` files and top-level OpenMX launchers preserve existing commands. Maintained import and symmetry algorithms live in `tapw.io`; historical implementations are under `legacy/`. The distribution builder is under `release/`.
