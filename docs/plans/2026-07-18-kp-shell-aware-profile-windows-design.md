# Shell-Aware KP Model Profile Windows

## Goal

Define physically meaningful and comparable band windows for the four automatic
continuum-model profiles.  A compact profile must minimize model complexity while
remaining accurate at the target band edge.  A broad profile may use more terms,
but it must not degrade the compact profile's edge bands.

## Band-window convention

Let

\[
N_0 = \sum_l n_{\mathrm{orb},l}
\]

be the internal dimension of the central projected manifold.  The compact
window is

\[
N_{\mathrm{low}} = \min(N_{\mathrm{total}}, 2N_0).
\]

The broad window is derived from the first two distinct radial Q shells in each
layer.  A Q=0 shell, when present, is the first shell.  All Q points at the same
radius are retained together, so a symmetry star is never split.  For layer
`l`, let `NQ_l(2)` be the cumulative number of Q points in its first two radial
shells.  Then

\[
N_{\mathrm{high}} = \min\left(
N_{\mathrm{total}},
\sum_l n_{\mathrm{orb},l} NQ_l(2)
\right).
\]

For shifted K/M sets without Q=0, the two smallest distinct Q norms are used.
The counting is layer-aware and supports unequal Q sets and unequal orbital
counts.

For the current examples this gives:

| Case | `n_orb` | low | high |
|---|---:|---:|---:|
| MgI2 Gamma 19Q | `[2, 2]` | 8 | 28 |
| MgI2 Gamma 31Q | `[2, 2]` | 8 | 28 |
| PtSe2 Gamma 19Q | `[1, 1]` | 4 | 14 |

## Linear fitting objectives

The linear low profile uses a two-sided spectral weight in the target-Heff
eigenbasis.  The top or bottom `N_low` states receive the primary weight.  A
small cross-window floor may be retained so that low/high mixing, which shifts
edge eigenvalues at second order, is not ignored.  The objective remains linear
in continuum coefficients.

The linear high profile uses the same edge weight plus the broad `N_high`
window.  Broad-window accuracy is optimized only among candidates that preserve
the selected low profile at the shared `N_low` window.

Nonlinear low and high profiles use the same window definitions and dominance
rules.  The solver family changes, but the physical meaning of `low` and `high`
does not.

## Candidate metrics

Every complete candidate is evaluated on both windows, regardless of the
profile for which it was fitted.  Reports include, for `N_low` and `N_high`:

- aligned RMS band error;
- maximum absolute band error;
- mean principal-angle subspace overlap;
- minimum principal singular value.

Symmetry certification and acceptance guards remain hard eligibility
requirements.  Expanded/all-band RMS remains diagnostic and does not replace
the two physical windows.

## Low-profile selection

Low is a constrained complexity minimization, not simply the candidate with the
smallest RMS.  First form the fidelity-eligible set using edge RMS, maximum error,
subspace-overlap targets, and the existing statistical tolerance around the best
edge result.  Then select lexicographically by:

1. independent real parameters;
2. active symmetry-closed term groups;
3. effective harmonic support;
4. total derivative order;
5. edge RMS and maximum error.

If no candidate meets the fidelity targets, continue with the best Pareto
candidate and record an explicit warning and unmet targets.

## High-profile dominance

After selecting low, a high candidate is eligible only if, on the same
`N_low` window:

\[
\mathrm{RMS}_{\mathrm{high}} \leq \mathrm{RMS}_{\mathrm{low}} + \epsilon_R,
\]

\[
\mathrm{Max}_{\mathrm{high}} \leq \mathrm{Max}_{\mathrm{low}} + \epsilon_M,
\]

and its mean overlap and minimum principal singular value do not fall below the
low values by more than their numerical tolerances.  Among eligible candidates,
high is selected for `N_high` accuracy, followed by overlap and complexity.

If no broad candidate dominates low, the best broad candidate is still written
so the workflow can continue, but its profile status is `WARN` with the failed
low-window guards printed in the report.

## Reporting and compatibility

The automatic report prints the resolved Q-shell counts and both band windows.
Each of the four profiles reports both low-window and high-window metrics, so
different primary windows are never presented as directly comparable numbers.
Explicit user overrides remain available for both counts.  Legacy linear-only
mode retains its existing output layout, but uses the resolved window metadata
when automatic four-profile selection is enabled.

## Validation

Unit tests cover Q=0 inclusion, shifted Q sets, unequal layer Q sets, dimension
clamping, low complexity ordering, high dominance, and warning fallback.  The
MgI2 Gamma 19Q and 31Q bigmem runs are repeated with low/high windows 8/28 and a
3000-variable ceiling.  Final validation compares band errors, principal-angle
overlap, symmetry covariance residuals, and band plots for all four profiles.
