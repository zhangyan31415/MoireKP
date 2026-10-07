"""Fixed sample selection for differentiable, sorted-eigenvalue residuals."""
from __future__ import annotations

from functools import wraps
from collections.abc import Mapping
import numpy as np

DEFAULT_DEGENERACY_TOL_EV = 1.0e-8


class BandDegeneracyError(ValueError):
    def __init__(self, reason, report):
        self.reason = reason
        self.report = report
        super().__init__(f"band degeneracy: {reason}; {report}")


class BandDegeneracyGuard:
    """Exclude initial degenerate k points, then reject new degeneracies.

    All indices in the report address the supplied band-sample array. A gap
    touching a selected state is checked even across the selection boundary.
    The alignment anchor is the original first point, never the first kept one.
    """

    def __init__(self, target, initial, selected_mask, *,
                 tolerance_ev=DEFAULT_DEGENERACY_TOL_EV, alignment="none"):
        self.tolerance_ev = float(tolerance_ev)
        if not np.isfinite(self.tolerance_ev) or self.tolerance_ev <= 0:
            raise ValueError("band degeneracy tolerance must be positive and finite")
        target = np.asarray(target, dtype=float)
        initial = np.asarray(initial, dtype=float)
        self.selected_mask = np.asarray(selected_mask, dtype=bool).copy()
        if (target.ndim != 2 or initial.shape != target.shape
                or self.selected_mask.shape != target.shape or target.shape[0] == 0
                or target.shape[1] == 0):
            raise ValueError("band spectra and selected mask must have shape (Nk,dim)")
        self._validate(target)
        self._validate(initial)
        self.anchor = None
        if alignment in {"top", "bottom"}:
            indices = np.flatnonzero(self.selected_mask[0])
            if indices.size == 0:
                raise ValueError("alignment anchor has no selected band")
            self.anchor = int(indices[-1] if alignment == "top" else indices[0])
        elif alignment not in {"none", ""}:
            raise ValueError(f"unsupported band alignment {alignment!r}")
        self.reference_pairs = self._pairs(target)
        self.initial_pairs = self._pairs(initial)
        self.active_points = self.selected_mask.any(axis=1)
        for row, _band in self.reference_pairs + self.initial_pairs:
            self.active_points[row] = False
        self.anchor_degenerate = bool(self._anchor_pairs(target) or self._anchor_pairs(initial))
        self.active_points.setflags(write=False)
        self.selected_mask.setflags(write=False)

    def _validate(self, values):
        if values.shape != self.selected_mask.shape or not np.isfinite(values).all():
            raise ValueError("band eigenvalues must be finite and match selected mask")
        if np.any(np.diff(values, axis=1) < 0):
            raise ValueError("band eigenvalues must be sorted")

    def _pairs(self, values):
        touch = self.selected_mask[:, :-1] | self.selected_mask[:, 1:]
        return np.argwhere((np.diff(values, axis=1) <= self.tolerance_ev) & touch).tolist()

    def _anchor_pairs(self, values):
        if self.anchor is None:
            return []
        n = values.shape[1]
        return [[0, i] for i in [self.anchor - 1, self.anchor]
                if 0 <= i < n - 1 and values[0, i + 1] - values[0, i] <= self.tolerance_ev]

    def report(self):
        return dict(policy="exclude_initial_points_abort_new_degeneracy",
                    tolerance_ev=self.tolerance_ev,
                    active_point_indices=np.flatnonzero(self.active_points).tolist(),
                    excluded_point_indices=np.flatnonzero(~self.active_points).tolist(),
                    reference_degenerate_pairs=self.reference_pairs,
                    initial_degenerate_pairs=self.initial_pairs,
                    alignment_anchor_band=self.anchor,
                    matrix_samples_unchanged=True)

    def require_usable(self):
        if self.anchor_degenerate:
            raise BandDegeneracyError("degenerate_alignment_anchor", self.report())
        if not self.active_points.any():
            raise BandDegeneracyError("no_nondegenerate_band_points", self.report())

    def check(self, values):
        values = np.asarray(values, dtype=float)
        self._validate(values)
        self.require_usable()
        anchor = self._anchor_pairs(values)
        pairs = [pair for pair in self._pairs(values) if self.active_points[pair[0]]]
        if anchor or pairs:
            report = self.report()
            report["trigger_pairs"] = anchor or pairs
            raise BandDegeneracyError(
                "degenerate_alignment_anchor" if anchor else "new_target_degeneracy", report)


def retain_coefficients_on_degeneracy(function):
    """Abort optional refinement before a singular eigenvalue Jacobian is used."""
    @wraps(function)
    def wrapped(moire_config, model_config, model, *args, **kwargs):
        fitted = getattr(model, "_fitted_response_model", None)
        terms = list(getattr(model, "terms", {}).values())
        terms += list(getattr(model, "candidate_terms", ()) or ())
        saved = [(term, {key: getattr(term, key) for key in
                         ("r_value_real", "r_value_imag", "active") if hasattr(term, key)})
                 for term in terms]
        try:
            return function(moire_config, model_config, model, *args, **kwargs)
        except BandDegeneracyError as exc:
            if fitted is not None:
                model._fitted_response_model = fitted
            for term, fields in saved:
                for key, value in fields.items():
                    setattr(term, key, value)
            cfg = args[0] if args and isinstance(args[0], Mapping) else kwargs.get(
                "raw_cfg", getattr(model_config, "band_refinement_config", {}))
            print(f"[kp model] nonlinear refinement skipped: {exc.reason}; "
                  "input coefficients retained", flush=True)
            return dict(enabled=True, skipped=True, accepted=False, reason=exc.reason,
                        mode=cfg.get("mode", "band_refinement"),
                        degeneracy_guard=exc.report,
                        acceptance_guard=dict(enabled=True, accepted=False,
                                              failed_windows=[exc.reason]))
    return wrapped
