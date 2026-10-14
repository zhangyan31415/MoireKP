from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kp.projection_selection import (
    CandidateRejected,
    CandidateRejectionReason,
    FixedWindowBandMetrics,
    FrozenTargetWindow,
    ProjectionOverlapMetrics,
    TargetWindowSpec,
    certify_minimum_principal_overlap,
    evaluate_fixed_target_window,
    projection_overlap_metrics,
    resolve_target_window,
)


def _window_spec(
    *,
    edge: str = "valence",
    band_count: int = 1,
    degeneracy_tolerance_mev: float = 1.0e-3,
) -> TargetWindowSpec:
    return TargetWindowSpec(
        edge=edge,
        band_count=band_count,
        validation_k_indices=(0, 1),
        energy_reference_ev=0.0,
        degeneracy_tolerance_mev=degeneracy_tolerance_mev,
    )


def test_target_window_spec_rejects_invalid_or_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="edge"):
        _window_spec(edge="middle")
    with pytest.raises(ValueError, match="band_count"):
        _window_spec(band_count=0)
    with pytest.raises(ValueError, match="validation_k_indices"):
        TargetWindowSpec(
            edge="valence",
            band_count=1,
            validation_k_indices=(0, 0),
            energy_reference_ev=0.0,
            degeneracy_tolerance_mev=1.0e-3,
        )
    with pytest.raises(ValueError, match="energy_reference_ev"):
        TargetWindowSpec(
            edge="valence",
            band_count=1,
            validation_k_indices=(0,),
            energy_reference_ev=float("nan"),
            degeneracy_tolerance_mev=1.0e-3,
        )


def test_fixed_window_rejects_candidate_with_too_few_bands_instead_of_shrinking() -> None:
    target = np.array(
        [
            [-0.40, -0.20, 0.30],
            [-0.35, -0.15, 0.35],
        ]
    )
    candidate = np.array([[-0.19], [-0.14]])
    target_window = resolve_target_window(target, _window_spec(band_count=2))

    with pytest.raises(CandidateRejected) as exc_info:
        evaluate_fixed_target_window(
            target_window,
            candidate,
        )

    rejection = exc_info.value
    assert rejection.reason is CandidateRejectionReason.INSUFFICIENT_CANDIDATE_BANDS
    assert rejection.required_band_count == 2
    assert rejection.available_band_count == 1
    assert rejection.k_index == 0


def test_fixed_window_preserves_twenty_mev_offset_for_one_band() -> None:
    target = np.array([[-0.100], [-0.120]])
    candidate = np.array([[-0.080], [-0.100]])
    target_window = resolve_target_window(target, _window_spec())

    metrics = evaluate_fixed_target_window(target_window, candidate)

    assert metrics.band_count == 1
    assert metrics.validation_k_indices == (0, 1)
    assert metrics.rms_error_mev == pytest.approx(20.0)
    assert metrics.maximum_abs_error_mev == pytest.approx(20.0)
    np.testing.assert_allclose(metrics.errors_mev, np.full((2, 1), 20.0))


def test_valence_candidate_can_cross_global_reference_without_rejection() -> None:
    target = np.array([[-0.010], [-0.010]])
    candidate = np.array([[-0.200, 0.010], [-0.200, 0.010]])
    target_window = resolve_target_window(target, _window_spec(edge="valence"))

    metrics = evaluate_fixed_target_window(target_window, candidate)

    np.testing.assert_allclose(metrics.errors_mev, np.full((2, 1), 20.0))


def test_conduction_candidate_can_cross_global_reference_without_rejection() -> None:
    target = np.array([[0.010], [0.010]])
    candidate = np.array([[-0.010, 0.200], [-0.010, 0.200]])
    target_window = resolve_target_window(
        target,
        _window_spec(edge="conduction"),
    )

    metrics = evaluate_fixed_target_window(target_window, candidate)

    np.testing.assert_allclose(metrics.errors_mev, np.full((2, 1), -20.0))


def test_target_window_rejects_boundary_that_splits_degenerate_multiplet() -> None:
    target = np.array(
        [
            [-0.3000000, -0.1000000, -0.0999995, 0.4000000],
            [-0.2500000, -0.0800000, -0.0500000, 0.4500000],
        ]
    )
    with pytest.raises(CandidateRejected) as exc_info:
        resolve_target_window(
            target,
            _window_spec(band_count=1, degeneracy_tolerance_mev=1.0e-3),
        )

    rejection = exc_info.value
    assert rejection.reason is CandidateRejectionReason.TARGET_BOUNDARY_DEGENERACY
    assert rejection.k_index == 0
    assert rejection.boundary_gap_mev == pytest.approx(5.0e-4)


def test_frozen_target_window_is_reused_across_candidates_and_target_mutation() -> None:
    target = np.array(
        [
            [-0.40, -0.20, 0.30, 0.40],
            [-0.35, -0.15, 0.35, 0.45],
        ]
    )
    spec = _window_spec(band_count=1)

    target_window = resolve_target_window(target, spec)

    assert isinstance(target_window, FrozenTargetWindow)
    assert target_window.target_band_ids == ((1,), (1,))
    np.testing.assert_allclose(
        target_window.target_energies_ev,
        np.array([[-0.20], [-0.15]]),
    )
    assert isinstance(target_window.target_energies_ev, tuple)
    with pytest.raises(TypeError, match="item assignment"):
        target_window.target_energies_ev[0][0] = -99.0
    changed_energies = FrozenTargetWindow(
        spec=spec,
        target_band_ids=((1,), (1,)),
        target_energies_ev=((-0.21,), (-0.16,)),
    )
    assert changed_energies != target_window
    assert hash(target_window) == hash(target_window)

    target[:] = np.array(
        [
            [-0.40, 0.10, 0.30, 0.40],
            [-0.35, 0.15, 0.35, 0.45],
        ]
    )
    candidate_a = np.array(
        [
            [-0.40, -0.30, -0.25, -0.18],
            [-0.35, -0.30, -0.25, -0.13],
        ]
    )
    candidate_b = np.array(
        [
            [-0.40, -0.30, -0.25, -0.19],
            [-0.35, -0.30, -0.25, -0.14],
        ]
    )

    metrics_a = evaluate_fixed_target_window(target_window, candidate_a)
    metrics_b = evaluate_fixed_target_window(target_window, candidate_b)

    assert target_window.target_band_ids == ((1,), (1,))
    np.testing.assert_allclose(metrics_a.errors_mev, np.full((2, 1), 20.0))
    np.testing.assert_allclose(metrics_b.errors_mev, np.full((2, 1), 10.0))
    assert isinstance(metrics_a.errors_mev, tuple)
    with pytest.raises(TypeError, match="item assignment"):
        metrics_a.errors_mev[0][0] = -99.0
    assert hash(metrics_a) == hash(metrics_a)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("rms_error_mev", float("nan")),
        ("rms_error_mev", float("inf")),
        ("maximum_abs_error_mev", float("nan")),
        ("maximum_abs_error_mev", float("inf")),
    ),
)
def test_fixed_window_metrics_reject_nonfinite_scalars(
    field_name: str,
    value: float,
) -> None:
    values = {
        "band_count": 1,
        "validation_k_indices": (0,),
        "rms_error_mev": 1.0,
        "maximum_abs_error_mev": 1.0,
        "errors_mev": ((1.0,),),
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        FixedWindowBandMetrics(**values)


def test_fixed_window_metrics_reject_nonfinite_or_inconsistent_errors() -> None:
    with pytest.raises(ValueError, match="errors_mev"):
        FixedWindowBandMetrics(
            band_count=1,
            validation_k_indices=(0,),
            rms_error_mev=1.0,
            maximum_abs_error_mev=1.0,
            errors_mev=((float("inf"),),),
        )
    with pytest.raises(ValueError, match="inconsistent"):
        FixedWindowBandMetrics(
            band_count=1,
            validation_k_indices=(0,),
            rms_error_mev=2.0,
            maximum_abs_error_mev=2.0,
            errors_mev=((1.0,),),
        )


def test_projection_metrics_keep_principal_capture_and_anchor_quality_separate() -> None:
    target = np.zeros((1, 3, 2), dtype=complex)
    target[0, 0, 0] = 1.0
    target[0, 1, 1] = 1.0

    model = np.zeros_like(target)
    model[0, 0, 0] = 1.0
    model[0, 1, 1] = np.sqrt(0.75)
    model[0, 2, 1] = 0.5

    projection = np.eye(3, dtype=complex)[None, :, :]
    metrics = projection_overlap_metrics(
        model,
        target,
        validation_k_indices=(0,),
        projection_basis=projection,
        gauge_anchor_quality=0.123,
    )

    assert metrics.minimum_principal_overlap_squared == pytest.approx(0.75)
    assert metrics.mean_principal_overlap_squared == pytest.approx(0.875)
    assert metrics.target_capture == pytest.approx(1.0)
    assert metrics.gauge_anchor_quality == pytest.approx(0.123)
    assert metrics.worst_k_index == 0


def test_target_capture_measures_projection_space_not_model_band_gauge() -> None:
    target = np.zeros((1, 3, 2), dtype=complex)
    target[0, 0, 0] = 1.0
    target[0, 1, 1] = 1.0
    model = target.copy()

    projection = np.zeros((1, 3, 2), dtype=complex)
    projection[0, 0, 0] = 1.0
    projection[0, 1, 1] = 0.8
    projection[0, 2, 1] = 0.6

    metrics = projection_overlap_metrics(
        model,
        target,
        validation_k_indices=(0,),
        projection_basis=projection,
        gauge_anchor_quality=None,
    )

    assert metrics.minimum_principal_overlap_squared == pytest.approx(1.0)
    assert metrics.mean_principal_overlap_squared == pytest.approx(1.0)
    assert metrics.target_capture == pytest.approx(0.82)
    assert metrics.gauge_anchor_quality is None


def test_projection_overlap_requires_explicit_complete_projection_space() -> None:
    basis = np.ones((1, 1, 1), dtype=complex)

    with pytest.raises(TypeError, match="projection_basis"):
        projection_overlap_metrics(
            basis,
            basis,
            validation_k_indices=(0,),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("minimum_principal_overlap_squared", float("nan")),
        ("mean_principal_overlap_squared", float("inf")),
        ("target_capture", float("nan")),
        ("gauge_anchor_quality", float("inf")),
    ),
)
def test_projection_overlap_metrics_reject_nonfinite_scalars(
    field_name: str,
    value: float,
) -> None:
    metrics = ProjectionOverlapMetrics(
        minimum_principal_overlap_squared=0.9,
        mean_principal_overlap_squared=0.95,
        target_capture=0.98,
        gauge_anchor_quality=0.8,
        validation_k_indices=(3, 7),
        worst_k_index=7,
    )

    with pytest.raises(ValueError, match=field_name):
        replace(metrics, **{field_name: value})


def test_projection_overlap_metrics_reject_inconsistent_range_and_k_context() -> None:
    with pytest.raises(ValueError, match="minimum.*mean"):
        ProjectionOverlapMetrics(
            minimum_principal_overlap_squared=0.9,
            mean_principal_overlap_squared=0.8,
            target_capture=0.95,
            gauge_anchor_quality=None,
            validation_k_indices=(3, 7),
            worst_k_index=7,
        )
    with pytest.raises(ValueError, match="worst_k_index"):
        ProjectionOverlapMetrics(
            minimum_principal_overlap_squared=0.8,
            mean_principal_overlap_squared=0.9,
            target_capture=0.95,
            gauge_anchor_quality=None,
            validation_k_indices=(3, 7),
            worst_k_index=1,
        )


def test_minimum_overlap_certifier_rejects_manually_corrupted_nan_metric() -> None:
    metrics = ProjectionOverlapMetrics(
        minimum_principal_overlap_squared=0.9,
        mean_principal_overlap_squared=0.95,
        target_capture=0.98,
        gauge_anchor_quality=0.8,
        validation_k_indices=(3, 7),
        worst_k_index=7,
    )
    object.__setattr__(metrics, "minimum_principal_overlap_squared", float("nan"))

    with pytest.raises(ValueError, match="minimum_principal_overlap_squared"):
        certify_minimum_principal_overlap(metrics, threshold=0.90)


def test_projection_metrics_are_invariant_under_simultaneous_unitary_gauge() -> None:
    rng = np.random.default_rng(12345)
    nk = 2
    ambient = 4
    states = 2
    target = np.zeros((nk, ambient, states), dtype=complex)
    model = np.zeros_like(target)
    projection = np.zeros((nk, ambient, 3), dtype=complex)
    for k_index in range(nk):
        target[k_index, :states] = np.eye(states)
        angle = 0.2 + 0.1 * k_index
        model[k_index, 0, 0] = 1.0
        model[k_index, 1, 1] = np.cos(angle)
        model[k_index, 2, 1] = np.sin(angle)
        projection[k_index, :3] = np.eye(3)

    reference = projection_overlap_metrics(
        model,
        target,
        validation_k_indices=(0, 1),
        projection_basis=projection,
        gauge_anchor_quality=0.75,
    )

    gauged_target = np.empty_like(target)
    gauged_model = np.empty_like(model)
    gauged_projection = np.empty_like(projection)
    for k_index in range(nk):
        ambient_q, _ = np.linalg.qr(
            rng.normal(size=(ambient, ambient))
            + 1.0j * rng.normal(size=(ambient, ambient))
        )
        target_q, _ = np.linalg.qr(
            rng.normal(size=(states, states))
            + 1.0j * rng.normal(size=(states, states))
        )
        model_q, _ = np.linalg.qr(
            rng.normal(size=(states, states))
            + 1.0j * rng.normal(size=(states, states))
        )
        projection_q, _ = np.linalg.qr(
            rng.normal(size=(3, 3)) + 1.0j * rng.normal(size=(3, 3))
        )
        gauged_target[k_index] = ambient_q @ target[k_index] @ target_q
        gauged_model[k_index] = ambient_q @ model[k_index] @ model_q
        gauged_projection[k_index] = (
            ambient_q @ projection[k_index] @ projection_q
        )

    transformed = projection_overlap_metrics(
        gauged_model,
        gauged_target,
        validation_k_indices=(0, 1),
        projection_basis=gauged_projection,
        gauge_anchor_quality=0.75,
    )

    assert transformed.minimum_principal_overlap_squared == pytest.approx(
        reference.minimum_principal_overlap_squared
    )
    assert transformed.mean_principal_overlap_squared == pytest.approx(
        reference.mean_principal_overlap_squared
    )
    assert transformed.target_capture == pytest.approx(reference.target_capture)
    assert transformed.gauge_anchor_quality == reference.gauge_anchor_quality


def test_one_bad_k_point_reports_actual_noncontiguous_validation_k_index() -> None:
    target = np.zeros((8, 2, 1), dtype=complex)
    target[:, 0, 0] = 1.0
    model = target.copy()
    model[7, :, 0] = np.array([0.0, 1.0])
    projection = target.copy()

    metrics = projection_overlap_metrics(
        model,
        target,
        validation_k_indices=(3, 7),
        projection_basis=projection,
    )

    assert metrics.mean_principal_overlap_squared == pytest.approx(0.5)
    assert metrics.minimum_principal_overlap_squared == pytest.approx(0.0)
    assert metrics.validation_k_indices == (3, 7)
    assert metrics.worst_k_index == 7
    with pytest.raises(CandidateRejected) as exc_info:
        certify_minimum_principal_overlap(metrics, threshold=0.90)
    assert exc_info.value.reason is CandidateRejectionReason.SUBSPACE_OVERLAP
    assert exc_info.value.k_index == 7


def test_projection_metrics_reject_out_of_bounds_actual_k_index() -> None:
    basis = np.ones((2, 1, 1), dtype=complex)

    with pytest.raises(ValueError, match="validation_k_indices"):
        projection_overlap_metrics(
            basis,
            basis,
            validation_k_indices=(0, 2),
            projection_basis=basis,
        )
