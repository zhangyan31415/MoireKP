# Bilayer MoTe2 3.89 Degree K Valley

The selected paper model uses the archived 2024 SOC calculation with
Mo7.0-s3p2d2 and Te7.0-s3p2d2f1: 71 spatial orbitals per MoTe2 unit,
the same PAO specifications as the selected trilayer example.

Its K-valley Hamiltonian has dimension 54, one spin-up state per layer and
27 Q channels per layer. The model stores 156 real coefficients spanning
154 independent directions. The polynomial orders are 10/5/5
(kinetic/intralayer/interlayer), with harmonic-count settings 2/3
(intralayer/interlayer; the intralayer count includes zero transfer).
The linear solve uses the full matrices at path rows [0, 2, 20], with zero
one-sided and two-sided weights. The target comparison window is the highest
eight bands. Linearized Löwdin reduction
uses Eref = -4.614105470129998 eV.

The real-space H/S average uses the original relaxed 1302-atom coordinates
and six operations detected at 0.05 Å. The archived atom-corresponding rigid
reference defines the TAPW projector coordinates; its cell is 51.8951368053 Å
in plane with a 120-degree angle and an 80 Å perpendicular height. TAPW source
symmetry uses a 0.005 Å site tolerance and retains C3z and C2T. No new DFT
calculation was performed for this coordinate reference.

From the repository root, using the current installed TAPW/KP commands:

```bash
# external-data
tapw run  -c examples/mote2_3.89/tapw/configs/mote2_3.89_K1_paper_20261001_q06.yaml
tapw symm -c examples/mote2_3.89/tapw/configs/mote2_3.89_K1_paper_20261001_q06.yaml
kp project -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_tuned_q06.yaml
kp symm    -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_tuned_q06.yaml
kp model   -c examples/mote2_3.89/kp/configs/mote2_3.89_K1_spinless_tuned_q06.yaml
```

TAPW uses 61 processes with one BLAS thread each. The KP configuration uses
10 workers with eight BLAS threads each. Run these only on resources assigned
to the task. Both configurations use explicit versioned output directories:
`tapw/outputs_selected_20261001/` and `kp/outputs_selected_20261001/`.
The existing `kp/outputs/k1_spinless_tuned` and `tapw/outputs/K1/q06` aliases
resolve to these selected artifacts. The source inputs are under
`openmx/selected_20261001/`.

The complete 61-point, eight-band comparison gives raw RMS/maximum errors
0.696/1.997 meV. With each spectrum's VBM aligned, these become
0.818/1.976 meV. The paper reports all three error stages on the same complete
window and uses the same export for Fig. 2 bands and geometry. Numerical
records and update commands are in
`paper/moirekp/code_paper/review/evidence/revision_20261001_bilayer154/`.

The other spinful and earlier spinless configuration files preserve the
previous smaller-basis example. Their original inputs remain available;
they are not the selected paper model. The pre-update outputs are retained
under `kp/outputs/k1_spinless_tuned_before_20261001` and
`tapw/outputs/K1/q06_before_20261001`.
