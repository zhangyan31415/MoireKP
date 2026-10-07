# Bilayer ZrS2 3.89 Degree Gamma And M Valleys

This example uses the 3.89-degree ZrS2 OpenMX dataset from the AB data family.
The TAPW inputs require the symmetry-restored `H_symm.npz` and `S_symm.npz`
matrices generated from that dataset. The matching `openmx.dat_rigid` structure
input is included here.

The spinful Gamma q04 model uses energy-linearized Löwdin downfolding at
-6.21 eV, automatic low-energy and harmonic selection, orders 8/4/4, and
a 2.25/22.5 target-subspace-weighted linear fit on the closed Gamma-M-K-Gamma path. The selected spinful M1 q07
model uses the same reduction at -5.22 eV, orders 6/4/6,
and an unweighted linear fit. The spinless M1 q07 model uses
orders 10/6/6 and a weighted linear fit. Automatic harmonic selection chose
two intralayer and two interlayer shells for all three models.

```bash
tapw run  -c examples/zrs2_3.89/tapw/configs/zrs2_3.89_Gamma_spinful_q04.yaml  # external-data
tapw symm -c examples/zrs2_3.89/tapw/configs/zrs2_3.89_Gamma_spinful_q04.yaml  # external-data
kp project -c examples/zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml  # external-data
kp symm   -c examples/zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml  # external-data
kp model  -c examples/zrs2_3.89/kp/configs/zrs2_3.89_Gamma_spinful_q04.yaml  # external-data

tapw run  -c examples/zrs2_3.89/tapw/configs/zrs2_3.89_M1_spinful_q07.yaml  # external-data
tapw symm -c examples/zrs2_3.89/tapw/configs/zrs2_3.89_M1_spinful_q07.yaml  # external-data
kp project -c examples/zrs2_3.89/kp/configs/zrs2_3.89_M1_spinful_linearized_q07.yaml  # external-data
kp symm   -c examples/zrs2_3.89/kp/configs/zrs2_3.89_M1_spinful_linearized_q07.yaml  # external-data
kp model  -c examples/zrs2_3.89/kp/configs/zrs2_3.89_M1_spinful_linearized_q07.yaml  # external-data

kp project -c examples/zrs2_3.89/kp/configs/zrs2_3.89_M1_spinless_q07.yaml  # external-data
kp symm   -c examples/zrs2_3.89/kp/configs/zrs2_3.89_M1_spinless_q07.yaml  # external-data
kp model  -c examples/zrs2_3.89/kp/configs/zrs2_3.89_M1_spinless_q07.yaml  # external-data
```

The selected Gamma fit uses closed-path rows [5,20,40,55]. In reciprocal
fractional coordinates these are (1/8,0), (1/2,0), (1/3,1/3), and (1/12,1/12).
Row 55 is on K-to-central-Gamma; row 60 repeats row 0 and is not fitted.
The 152-dimensional model has 630 independent real coefficients. Both matrix
weights, 2.25/22.5, act on the highest 20 reference states. The fit is ordinary
linear weighted least squares with zero ridge and no exterior spectral constraint.
The selected export is `kp/outputs_selected_20261006/gamma_spinful/Gamma/q04/model/`.
The normal `kp project`, `kp symm`, `kp model` workflow reproduces the selected
Hamiltonian to better than 1e-12 eV. The TAPW Gamma configuration now writes the
closed-path source to `tapw/outputs_gmkg_km_neighbors/` and uses the Top-2 group
for its 41x41 topology example.
The selected M1 fit uses rows [0,20] (Gamma and M), with both weights zero.
The separately supplied spinless M1 example is unchanged.

The selected M1 export is in `kp/outputs/m1_spinful_linearized/M1/q07/model/`.
On the stored 61-point path, its lowest eight bands differ from TAPW by
1.440 meV RMS before display alignment, or 0.637 meV after aligning its CBM
to the TAPW CBM. The former fixed-Schur configuration
`zrs2_3.89_M1_spinful_q07.yaml` remains available for reproducing that result.
The CLI defaults to energy-linearized downfolding; the older fixed-Schur M1
configuration remains an explicit comparison. The stored path informed the
selected M1 choice, so its reported path error is an in-sample check.

The `validation/` directory beside each selected model contains the manuscript
band/overlap comparisons and, for Gamma, the 41x41 BC/Trg maps and Top-2 Wilson
loop. Wilson-loop boundary singular values and loop-mesh comparisons are kept
in the validation records. The standalone exports were verified against the
selected models at both path and off-path momenta.

On the common 61-point closed path, the selected Gamma model's highest ten
bands have 0.618 meV raw RMS and 2.737 meV maximum
error against full-spin TAPW. Direct TAPW clustered overlaps have
mean 99.33% and minimum 96.66% (3 meV cluster
tolerance and the same fixed target sets). These primary statistics do not
extend to all lower context bands in the manuscript plot. At the fixed Q cutoff,
a finite post-fit extension screen covers +/-b1 with about 0.0032 meV maximum
upturn, but the first >1 meV upturn is already near 1.001b1. No extrapolated
TAPW accuracy or continuous all-direction stability is claimed.
