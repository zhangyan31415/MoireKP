# Additional moiré continuum models

[Band comparisons in Fig. R1](band_comparisons.pdf) · [Model and error data](models.json)

| Structure | Twist | Valley | Path points (construction / validation) | Dimension | Coefficients | RMS (meV) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| [AA bilayer MoTe₂](../mote2_2.13/README.md) | 2.13° | K₁ | 61 / 61 | 54 | 246 | 0.422 |
| [AB bilayer ZrS₂](../zrs2_2.13/README.md) | 2.13° | Γ | 61 / 61 | 152 | 1350 | 0.385 |
| [AB bilayer ZrS₂](../zrs2_2.13/README.md) | 2.13° | M₁ | 61 / 61 | 88 | 1039 | 1.057 |
| [AA bilayer WSe₂](../wse2_2.13/README.md) | 2.13° | K₁ | 31 / 61 | 54 | 144 | 0.549 |
| [AA bilayer WSe₂](../wse2_3.89/README.md) | 3.89° | K₁ | 31 / 61 | 54 | 101 | 1.353 |
| [Alternating trilayer WSe₂](../wse2_trilayer_3.89/README.md) | 3.89° | K₁ | 61 / 61 | 81 | 680 | 1.143 |
| [AA bilayer SnSe₂](../snse2_3.89/README.md) | 3.89° | Γ | 31 / 61 | 148 | 941 | 0.744 |
| [AA bilayer SnSe₂](../snse2_3.89/README.md) | 3.89° | M₁ | 31 / 61 | 88 | 770 | 0.710 |
| [AB bilayer CaI₂](../cai2_ab_3.89/README.md) | 3.89° | Γ | 61 / 61 | 222 | 20 | 0.269 |
| [AB bilayer CaI₂](../cai2_ab_3.89/README.md) | 3.89° | M₁ | 31 / 61 | 72 | 932 | 0.034 |
| [AB bilayer CaI₂](../cai2_ab_2.28/README.md) | 2.28° | Γ | 61 / 61 | 186 | 393 | 0.039 |
| [AB bilayer CaI₂](../cai2_ab_2.28/README.md) | 2.28° | M₁ | 31 / 61 | 96 | 937 | 0.627 |

The table lists the twelve models in Fig. R1. A separate [alternating MoTe₂ trilayer example](../mote2_trilayer_3.89/README.md) is also available.

The first path count is the grid on which the original Hamiltonians were constructed. Coefficient extraction can use a subset of those momenta. The second is the grid used for reporting, including any points shared with construction. All reported energies and overlaps use the 61 actual TAPW samples. For the six 31-point construction grids, the original points occupy the even rows of the 61-point path. `eigvals.npy` preserves the construction-grid spectrum; `reference_bands.npz` contains the R1 comparison.

CaI₂ Γ inputs: [3.89° fitting data and script](../cai2_ab_3.89/kp/) · [2.28° input archive and KP configuration](../cai2_ab_2.28/kp/).

Each structure uses the same layout:
- `openmx/`: relaxed and rigid structures and the NSOC/SOC inputs.
- `tapw/tapw_<valley>.yaml`: TAPW inputs and construction grid.
- `kp/kp_<valley>.yaml`: model settings. Each bundled input is named `kp/input_<valley>.zip`. Unpack it into the example directory before running the commands in its README.
- `model/<valley>/`: coefficients, evaluator, and one `reference_bands.npz` validation archive.

Use the material README for the commands that consume each input. The CaI₂ 3.89° Γ YAML is read by its fixed-angle refinement driver. SnSe₂ Γ additionally uses `kp/refine_Gamma.yaml` after its linear reference stage.
The coefficients and NumPy evaluator under each `model/` directory can be used
directly from a checkout. The stored band validation can be replayed with
NumPy from the repository root:

```bash
python examples/additional_models/verify_models.py
```

Each `reference_bands.npz` contains Cartesian momenta in inverse angstroms, TAPW
eigenvalues in eV for the validation window, the energy shift applied to the
model, and the path indices used in the error calculation. The evaluator
recomputes the full continuum spectrum, and the verifier compares the
selected bands and reproduces the table from the stored arrays.

The twelve models in Fig. R1 use the full 61-point Gamma–M–K–Gamma path. Each model edge is aligned to its TAPW edge at Gamma. The [model index](models.json) records the errors and coefficient checksums. Each R1 validation archive also contains TAPW-to-model overlap matrices, the fixed band assignments, and the 3 meV cluster-averaged overlaps used for coloring. The NumPy verifier replays both the energies and these overlaps. The three cases using an orthonormalized resolvent embedding are the WSe₂ trilayer and the two CaI₂ Γ models. The other nine use the direct retained-state embedding.
