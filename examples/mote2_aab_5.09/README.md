# A–AB trilayer MoTe2, 5.09 degrees

Runtime class: **external-data**. The scientific H/S and TAPW arrays are separate from the source-only installation.

The active example uses the historical OpenMX H/S with a reconstructed, strictly hexagonal rigid geometry for TAPW. The rigid file preserves the 1143 source atom identities; it is a coordinate/basis input and does not represent a new DFT calculation. `openmx/soc/POSCAR_source_relaxed` and `openmx_source_original.dat` retain the actual old source geometry. `openmx/soc/POSCAR_rigid_selected_20260929` is the TAPW geometry, with a=b=39.7113764531 Å and gamma=120 degrees. Its construction preserves the old in-plane area, mean layer heights and registry; each layer has 127 copies of the monolayer basis.

The source basis is **Mo7.0-s3p2d2, Te7.0-s3p2d2f1**, with Mo_PBE19/Te_PBE19 potentials: 71 spatial orbitals per MoTe2 unit, 27051 spatial and 54102 spinor orbitals in the source supercell. The selected bilayer `mote2_3.89` example uses the same 71-orbital PAO specifications.

| Model | Spin block | Dimension | Independent coefficients | Orders kinetic/intralayer/interlayer | Harmonics intra/inter | Fit rows | Eref (eV) |
|---|---|---:|---:|---|---|---|---:|
| KA | up | 36 | 188 | 10/4/5 | 2/2 | [10,50] | -4.63485648238 |
| KB | down | 18 | 34 | 10/2/0 | 2/0 | [10,50] | -4.68485648238 |
| Gamma | all | 76 | 440 | 8/2/6 | 4/2 | [0,40] | -4.71680097242 |

All use q4, automatic low-state selection, energy-linearized Lowdin reduction and linear full-matrix fitting. One- and two-sided target-subspace weights are both zero. Configured target-band counts are 12/8/10; the primary validation windows are top6/top3/top8. Fit rows are zero-based explicit samples on the 61-vertex Gamma-M-K-Gamma path, not index ranges. The single-spin labels are specific to this source: up selects the two A layers and down the B layer. Do not transplant the spin labels from the former small-basis source.

The source Fermi energy is -4.182383731612424 eV. Relative to the former model setup, Eref was deterministically registered to retain its distance below each native valley VBM; no Eref optimization was performed. KA/KB share the same registration rule and a common KA-VBM display shift. The native Gamma VBM is 37.09192579 meV below the native K VBM.

```bash
tapw run  -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml  # external-data
tapw symm -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml  # external-data
tapw run  -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml  # external-data
kp project -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml  # external-data
kp symm    -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml  # external-data
kp model   -c examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_K1_A_q04.yaml  # external-data
```

Repeat the three KP commands with `K1_B` or `Gamma_spinful` in the config filename. Long calculations must run on an authorized compute node with process/thread counts matched to its allocation.

Active output aliases point to `outputs_selected_20260929`; preceding example outputs and input aliases are retained with `before_20260929` names. `selected_models_20260929.json` records source models, parameter counts, checksums and model-to-original-Heff errors. The top6/top3/top8 raw RMS errors are 0.494420/0.318135/0.644772 meV on all 61 path rows. These are not total TAPW-to-model errors or full-Brillouin-zone convergence claims. Geometry/overlap evidence belonging to earlier models must not be reused for this selection without matching input fingerprints.
