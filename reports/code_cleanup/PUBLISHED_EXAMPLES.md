# Published example acceptance

All six published configurations passed the complete 61-point KP project → symmetry → model replay and independent export checks. Inputs were real cached TAPW matrices; no DFT calculation or TAPW matrix generation is claimed.

| Case | Dimension | Points | Maximum model-spectrum difference from historical oracle (eV) |
|---|---:|---:|---:|
| mote2_k | 54 | 61 | 8.438e-14 |
| aab_ka | 36 | 61 | 1.284e-12 |
| aab_kb | 18 | 61 | 1.006e-12 |
| aab_gamma | 76 | 61 | 3.464e-14 |
| zrs2_m | 88 | 61 | 4.174e-14 |
| zrs2_gamma | 152 | 61 | 6.484e-14 |

ZrS2 Gamma was rerun on bigmem003 using `examples/zrs2_3.89/tapw/outputs_gmkg_km_neighbors`. All 61 fractional momenta match the published GMKG path within 5.551e-17. The physical configuration and numerical thresholds were unchanged.

Durable results: `/data/work/zy/software/1.tapw_code/moirekp-release/validation_runs/code_cleanup_20261014/published`. Full provenance and validation values are in `PUBLISHED_EXAMPLES.json`.
