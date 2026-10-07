import importlib

import numpy as np


def test_trace_condition_diagnostic_is_zero_for_ideal_match():
    cp = importlib.import_module("tapw.chern_post")
    trace_g = np.array([[1.0, 2.0], [3.0, 4.0]])
    omega_xy = -trace_g

    diagnostic = cp.compute_trace_condition_diagnostic(trace_g, omega_xy)

    assert diagnostic["warning"] == ""
    assert diagnostic["num_points"] == 4
    assert np.isclose(diagnostic["mean_trace_g"], 2.5)
    assert np.isclose(diagnostic["mean_abs_omega"], 2.5)
    assert np.isclose(diagnostic["rms_residual"], 0.0)
    assert np.isclose(diagnostic["max_abs_residual"], 0.0)
    assert np.isclose(diagnostic["delta_tr"], 0.0)
    assert np.isclose(diagnostic["rms_trace_g_fluctuation"], np.sqrt(1.25))
    assert np.isclose(diagnostic["delta_g"], np.sqrt(1.25) / 2.5)


def test_trace_condition_diagnostic_is_invariant_under_positive_rescaling():
    cp = importlib.import_module("tapw.chern_post")
    trace_g = np.array([[2.0, 5.0], [7.0, 11.0]])
    omega_xy = np.array([[1.0, -4.0], [8.0, -10.0]])

    reference = cp.compute_trace_condition_diagnostic(trace_g, omega_xy)
    scaled = cp.compute_trace_condition_diagnostic(3.5 * trace_g, 3.5 * omega_xy)

    assert np.isclose(reference["delta_tr"], scaled["delta_tr"])
    assert np.isclose(reference["delta_g"], scaled["delta_g"])


def test_delta_g_is_zero_for_uniform_trace_g():
    cp = importlib.import_module("tapw.chern_post")
    trace_g = np.full((2, 3), 4.0)
    omega_xy = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])

    diagnostic = cp.compute_trace_condition_diagnostic(trace_g, omega_xy)

    assert np.isclose(diagnostic["rms_trace_g_fluctuation"], 0.0)
    assert np.isclose(diagnostic["delta_g"], 0.0)


def test_trace_condition_diagnostic_ignores_non_finite_points():
    cp = importlib.import_module("tapw.chern_post")
    trace_g = np.array([[1.0, np.nan], [np.inf, 3.0]])
    omega_xy = np.array([[-1.0, 2.0], [5.0, -1.0]])

    diagnostic = cp.compute_trace_condition_diagnostic(trace_g, omega_xy)

    assert diagnostic["num_points"] == 2
    assert diagnostic["warning"] == ""
    assert np.isclose(diagnostic["mean_trace_g"], 2.0)
    assert np.isclose(diagnostic["mean_abs_omega"], 1.0)
    assert np.isclose(diagnostic["rms_residual"], np.sqrt(2.0))
    assert np.isclose(diagnostic["max_abs_residual"], 2.0)
    assert np.isclose(diagnostic["rms_trace_g_fluctuation"], 1.0)
    assert np.isclose(diagnostic["delta_g"], 0.5)


def test_trace_condition_diagnostic_returns_nan_when_mean_abs_omega_is_too_small():
    cp = importlib.import_module("tapw.chern_post")
    trace_g = np.array([[1.0, 2.0], [3.0, 4.0]])
    omega_xy = np.zeros((2, 2), dtype=float)

    diagnostic = cp.compute_trace_condition_diagnostic(trace_g, omega_xy, eps=1e-14)

    assert np.isnan(diagnostic["delta_tr"])
    assert "mean_abs_omega" in diagnostic["warning"]
    assert np.isfinite(diagnostic["delta_g"])


def test_trace_condition_diagnostic_returns_nan_delta_g_when_mean_trace_g_is_too_small():
    cp = importlib.import_module("tapw.chern_post")
    trace_g = np.zeros((2, 2), dtype=float)
    omega_xy = np.ones((2, 2), dtype=float)

    diagnostic = cp.compute_trace_condition_diagnostic(trace_g, omega_xy, eps=1e-14)

    assert np.isnan(diagnostic["delta_g"])
    assert "mean_trace_g" in diagnostic["warning"]


def test_trace_condition_diagnostic_rejects_shape_mismatch():
    cp = importlib.import_module("tapw.chern_post")

    try:
        cp.compute_trace_condition_diagnostic(np.zeros((2, 2)), np.zeros((2, 3)))
    except ValueError as exc:
        assert "Cannot compute trace-condition diagnostic" in str(exc)
        assert "(2, 2)" in str(exc)
        assert "(2, 3)" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for mismatched trace_g/omega_xy shapes.")
