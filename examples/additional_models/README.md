# Additional moiré continuum models

[Band comparisons in Fig. R1](band_comparisons.pdf) · [Model and error data](models.json)

| Structure | Twist | Valley | Dimension | Coefficients | RMS (meV) |
| --- | ---: | --- | ---: | ---: | ---: |
| [AA bilayer MoTe₂](../mote2_2.13/README.md) | 2.13° | K₁ | 54 | 246 | 0.422 |
| [AB bilayer ZrS₂](../zrs2_2.13/README.md) | 2.13° | Γ | 152 | 1350 | 0.385 |
| [AB bilayer ZrS₂](../zrs2_2.13/README.md) | 2.13° | M₁ | 88 | 1039 | 1.057 |
| [AA bilayer WSe₂](../wse2_2.13/README.md) | 2.13° | K₁ | 54 | 144 | 0.549 |
| [AA bilayer WSe₂](../wse2_3.89/README.md) | 3.89° | K₁ | 54 | 101 | 1.353 |
| [Alternating trilayer WSe₂](../wse2_trilayer_3.89/README.md) | 3.89° | K₁ | 81 | 680 | 1.143 |
| [AA bilayer SnSe₂](../snse2_3.89/README.md) | 3.89° | Γ | 148 | 941 | 0.744 |
| [AA bilayer SnSe₂](../snse2_3.89/README.md) | 3.89° | M₁ | 88 | 770 | 0.710 |
| [AB bilayer CaI₂](../cai2_ab_3.89/README.md) | 3.89° | Γ | 222 | 20 | 0.269 |
| [AB bilayer CaI₂](../cai2_ab_3.89/README.md) | 3.89° | M₁ | 72 | 932 | 0.034 |
| [AB bilayer CaI₂](../cai2_ab_2.28/README.md) | 2.28° | Γ | 186 | 393 | 0.039 |
| [AB bilayer CaI₂](../cai2_ab_2.28/README.md) | 2.28° | M₁ | 96 | 937 | 0.627 |

The table lists the twelve models in Fig. R1. A separate [alternating MoTe₂ trilayer example](../mote2_trilayer_3.89/README.md) is also available.

Each example describes its structural inputs, TAPW and KP configuration, and
the steps available for regenerating the model from those inputs.
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

The twelve models in Fig. R1 use the full 61-point Gamma–M–K–Gamma path. Each model edge is aligned to its TAPW edge at Gamma. The [61-point validation summary](validation61.json) records the errors and coefficient checksums. Each model directory includes `native_overlaps.npz`, containing TAPW-to-model overlap matrices, the fixed band assignments, and the 3 meV cluster-averaged overlaps used for coloring. The NumPy verifier replays both the energies and these overlaps. The three cases using an orthonormalized resolvent embedding are the WSe₂ trilayer and the two CaI₂ Γ models. The other nine use the direct retained-state embedding.
