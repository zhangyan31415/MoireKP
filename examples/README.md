# Paper examples

Each example has `openmx/` for the NSOC/SOC inputs and relaxed/rigid POSCAR
files, `tapw/` and `kp/` for the model configurations, and a top-level README
with the commands to run them.
NSOC is run to convergence before one-step SOC. Large H/S matrices and
subsequent results are generated locally.

| Example | Models | Instructions |
|---|---|---|
| AA bilayer MoTe2 | K | [MoTe2](mote2_3.89/README.md) |
| AB bilayer ZrS2 | Gamma, M | [ZrS2](zrs2_3.89/README.md) |
| A–AB trilayer MoTe2 | KA, KB, Gamma | [Trilayer MoTe2](mote2_aab_5.09/README.md) |

An additional [zero-twist hBN / bilayer graphene example](graphene_hbn/r2g_zero_twist/README.md)
reproduces an archived orthogonal Slater–Koster TAPW band calculation, retaining
both materials explicitly.
