# Degeneracy in optional nonlinear band fitting

The linear matrix-coefficient solve is unchanged. Optional nonlinear refinement
uses sorted band energies only at samples where the selected bands are separated
from their neighbours in both the reference and initial model spectra.

For `fit.method: nonlinear`, `fit.degeneracy_tol_ev` sets the absolute gap
tolerance in eV (default `1e-8`). The legacy `fit.refine_bands` mapping accepts
the same key. The tolerance is separate from the much larger degeneracy tolerance
used to complete a target window or to group overlaps in a plot.

Before optimization, a point is excluded from the band term if a gap no larger
than the tolerance touches any selected band, including the edge of the selected
window. Degeneracies entirely outside that window do not exclude a point.
This point selection is frozen throughout the optimizer call. Residuals and
Jacobians use the same selection; mean-square band losses are normalized over
the remaining points. The existing matrix-fit samples and weights are unchanged.

The legacy energy-alignment anchor remains its original point and band. An
initially degenerate anchor, an empty usable band set, or a new degeneracy at an
active sample/anchor aborts optional refinement. The input coefficients are
retained and the report records `skipped: true`, `accepted: false`, and the
reason. Finite-difference refinement uses the same checks; it is not a remedy
for nondifferentiable eigenvalues.

The `degeneracy_guard` report records the tolerance, kept/excluded point indices,
initial/reference degenerate pairs, and any pair triggering an abort. Pair
`[i, n]` means bands `n,n+1` at row `i` of the supplied band-sample array. For the
public interface, map these rows through `band_kpoints` to recover the original
construction-path indices. Legacy routes use `fit_kpoints.selected_indices`.

Matrix-only refinement branches do not need eigenvalue derivatives and remain
unchanged. The policy is intentionally conservative: even a symmetry-protected
degenerate pair is excluded from the band term. It does not implement directional
eigenvalue derivatives or a new differentiable cluster objective.

Regression coverage is in `tests/kp/test_band_degeneracy.py`, together with the
existing configured-model and complete-response-basis tests.
