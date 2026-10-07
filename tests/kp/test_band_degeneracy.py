"""Small-matrix regression tests for optional nonlinear eigenvalue fitting."""
from types import SimpleNamespace

import numpy as np
import pytest

from kp.model import pipeline


def guard(target, initial, mask, **kwargs):
    cls = getattr(pipeline, "BandDegeneracyGuard", None)
    assert cls is not None, "nonlinear fitting needs a shared degeneracy guard"
    return cls(np.asarray(target), np.asarray(initial), np.asarray(mask), **kwargs)


def test_guard_catches_selected_boundary_but_not_remote_degeneracy():
    values = np.array([[0., 0., 2., 3.], [0., 1., 2., 2.], [0., 1., 2., 3.]])
    mask = np.zeros_like(values, dtype=bool)
    mask[:, -1] = True
    g = guard(values, values, mask)
    np.testing.assert_array_equal(g.active_points, [True, False, True])
    assert g.report()["excluded_point_indices"] == [1]


def test_guard_checks_initial_model_and_preserves_fixed_mask():
    target = np.array([[0., 1.], [0., 1.], [0., 1.]])
    initial = np.array([[0., 0.], [0., 1.], [0., 1.]])
    g = guard(target, initial, np.ones_like(target, dtype=bool))
    np.testing.assert_array_equal(g.active_points, [False, True, True])
    g.check(target)
    np.testing.assert_array_equal(g.active_points, [False, True, True])
    with pytest.raises(ValueError, match="degeneracy"):
        g.check(np.array([[0., 1.], [0., 0.], [0., 1.]]))
    np.testing.assert_array_equal(g.active_points, [False, True, True])


def test_alignment_anchor_is_not_silently_replaced():
    values = np.array([[0., 0.], [0., 1.]])
    g = guard(values, values, np.ones_like(values, dtype=bool), alignment="top")
    with pytest.raises(ValueError, match="alignment"):
        g.require_usable()


def test_runtime_alignment_guard_even_if_anchor_point_was_excluded():
    values = np.array([[0., 0., 2.], [0., 1., 2.]])
    g = guard(values, values, np.ones_like(values, dtype=bool), alignment="top")
    g.require_usable()
    with pytest.raises(ValueError, match="alignment"):
        g.check(np.array([[0., 2., 2.], [0., 1., 2.]]))


def test_no_usable_points_is_explicit_and_tolerance_is_small():
    values = np.array([[0., 5e-9], [0., 2e-8]])
    g = guard(values, values, np.ones_like(values, dtype=bool))
    np.testing.assert_array_equal(g.active_points, [False, True])
    g = guard(values[:1], values[:1], np.ones((1, 2), dtype=bool))
    with pytest.raises(ValueError, match="no_nondegenerate"):
        g.require_usable()


@pytest.mark.parametrize("tol", [0., -1., np.nan, np.inf])
def test_invalid_gap_tolerance_is_rejected(tol):
    with pytest.raises(ValueError, match="tolerance"):
        guard([[0., 1.]], [[0., 1.]], [[True, True]], tolerance_ev=tol)


def test_public_residual_and_jacobian_use_same_frozen_points():
    target = np.array([np.diag([0., 0.]), np.diag([0., 2.])], dtype=complex)
    mask = np.ones((2, 2), dtype=bool)
    g = guard(np.linalg.eigvalsh(target), np.linalg.eigvalsh(target), mask)

    class Evaluator:
        def hamiltonians(self, y):
            return target + float(y[0]) * np.eye(2)[None]

        def band_jacobian(self, eigenvectors, selected_masks):
            assert not selected_masks[0].any()
            return np.ones((int(selected_masks.sum()), 1))

    def evaluate(y):
        return pipeline._public_band_residual_and_jacobian(
            None, np.zeros(1), np.array([0]), np.array([y]), np.zeros((2, 2)),
            target, target_bands="top", bands=2, band_loss_weight=1.,
            evaluator=Evaluator(),
            weighting=SimpleNamespace(selected_mask=mask, selected_counts=(2, 2)),
            degeneracy_guard=g,
        )

    r, j, counts = evaluate(.1)
    assert counts == (0, 2)
    assert r.shape == (2,) and j.shape == (2, 1)
    np.testing.assert_allclose(r, .1 / np.sqrt(2))
    finite_difference = (evaluate(.1 + 1e-6)[0] - evaluate(.1 - 1e-6)[0]) / 2e-6
    np.testing.assert_allclose(j[:, 0], finite_difference, atol=1e-9)


def test_legacy_jacobian_mask_keeps_original_alignment_anchor():
    vectors = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    basis = np.array([[np.diag([0., 1.]), np.diag([0., 3.]), np.diag([0., 5.])]])
    j = pipeline._band_refinement_eigenvalue_jacobian(
        vectors, basis, band_slice=(1, 2), align="top", band_sigma=1.,
        normalize=True, point_mask=np.array([False, True, True]),
    )
    np.testing.assert_allclose(j[:, 0], np.array([2., 4.]) / np.sqrt(2))


def public_fit(monkeypatch, *, all_degenerate=False, optimizer=None):
    target = np.array([np.diag([0., 0.]), np.diag([0., 3.])], dtype=complex)
    if all_degenerate:
        target[:] = 0

    class Fitted:
        coefficients = np.array([0.])
        fit_solver_channel_indices = (0,)

        def with_coefficients(self, values, **kwargs):
            result = Fitted()
            result.coefficients = np.asarray(values).copy()
            return result

    class Evaluator:
        backend = "toy"
        support = np.arange(2)
        cache_bytes = 0
        base_hamiltonians = np.array([np.diag([0., 0.]), np.diag([0., 2.])], dtype=complex)

        def hamiltonians(self, values):
            return self.base_hamiltonians + values[0] * np.diag([0., 1.])[None]

        def band_jacobian(self, vectors, masks):
            return np.concatenate([abs(vectors[i, 1, masks[i]])**2 for i in range(2)])[:, None]

    model = SimpleNamespace(_fitted_response_model=Fitted(),
                            _compiled_response_basis=SimpleNamespace(channel_ids=('x',), response_scales=np.ones(1)))
    original = model._fitted_response_model
    monkeypatch.setattr(pipeline, '_load_kpoints', lambda _: np.array([[0., 0.], [.1, .2]]))
    monkeypatch.setattr(pipeline, '_load_heff_in_model_basis', lambda _: target)
    monkeypatch.setattr(pipeline, '_current_heff_support_hamiltonians', lambda h, **kw: h)
    monkeypatch.setattr(pipeline, '_prepare_public_band_response_evaluator', lambda *a, **kw: Evaluator())
    monkeypatch.setattr(pipeline, '_sync_complete_response_coefficients_to_terms', lambda *a: None)

    def matrix_objective(basis, indices, points, targets, **kw):
        assert len(points) == 2  # The degenerate point is still a matrix sample.
        np.testing.assert_array_equal(targets, target)
        return np.ones((1, 1)), np.array([.2]), {}

    monkeypatch.setattr(pipeline, '_public_hamiltonian_quadratic_residual', matrix_objective)
    if optimizer:
        monkeypatch.setattr(pipeline.scipy.optimize, 'least_squares', optimizer)
    cfg = dict(hamiltonian_kpoints=[0, 1], band_kpoints=[0, 1], target_bands='top',
               bands=1, one_sided_weight=0., two_sided_weight=0., band_loss_weight=1.,
               max_steps=20, mode='public_nonlinear_v1')
    report = pipeline._refine_public_nonlinear_complete_response(
        SimpleNamespace(Q_set1=np.zeros((1, 2)), Q_set2=np.zeros((1, 2))),
        SimpleNamespace(n_orb=(1, 1), harmonics_config={}), model, cfg)
    return report, model, original


def test_public_fit_preserves_matrix_term_and_optimizes_nondegenerate_point(monkeypatch):
    report, model, _ = public_fit(monkeypatch)
    assert report['accepted']
    assert report['degeneracy_guard']['active_point_indices'] == [1]
    assert report['resolved_band_counts'] == [0, 1]
    # min (y-.2)^2 + (y-1)^2; matrix loss has not disappeared with point 0.
    np.testing.assert_allclose(model._fitted_response_model.coefficients, [.6], atol=1e-8)


def test_public_fit_skips_all_degenerate_reference_points(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('optimizer must not run without nondegenerate band samples')
    report, model, original = public_fit(monkeypatch, all_degenerate=True, optimizer=forbidden)
    assert report['skipped'] and not report['accepted']
    assert report['reason'] == 'no_nondegenerate_band_points'
    assert model._fitted_response_model is original


def test_public_fit_aborts_trial_crossing_and_keeps_input_model(monkeypatch):
    def crossing(fun, x0, **kwargs):
        fun(np.array([-2.]))  # The retained k point is now exactly degenerate.
        pytest.fail('trial degeneracy must abort before the derivative is used')
    report, model, original = public_fit(monkeypatch, optimizer=crossing)
    assert report['reason'] == 'new_target_degeneracy'
    assert report['degeneracy_guard']['trigger_pairs'] == [[1, 0]]
    assert not report['accepted']
    assert model._fitted_response_model is original


def test_public_config_accepts_gap_tolerance():
    config = dict(method='nonlinear', kpoints=[0], bands=1, one_sided_weight=0.,
                  two_sided_weight=0., band_kpoints=[0, 1], band_loss_weight=1.,
                  degeneracy_tol_ev=2e-8)
    parsed = pipeline._canonical_public_fit_method(config, heff_shape=(2, 2, 2))
    assert parsed['degeneracy_tol_ev'] == 2e-8


def legacy_fit(monkeypatch, *, compiled=False, alignment='none', crossing=False,
               initial_degenerate=True):
    from kp.model.core import ContinuumTermKey

    base = np.array([np.diag([0., 0. if initial_degenerate else 1.]),
                     np.diag([0., 2.])], dtype=complex)
    target = base.copy()
    target[1, 1, 1] = 3.
    term = SimpleNamespace(tag='Kinect', active=True, r_value_real=0., r_value_imag=0.,
                           key=ContinuumTermKey(0, 0, 1, 1, 1, 1, (0., 0.)))
    model = SimpleNamespace(terms={'t': term})
    response = np.broadcast_to(np.diag([0., 1.]), (2, 2, 2)).astype(complex)
    raw = dict(enabled=True, band_slice=[1, 2], align=alignment, components=['real'],
               variable_tags=['Kinect'], max_nfev=5, optimizer='scipy_least_squares',
               matrix_weight=1., matrix_compression='disabled')
    cfg = SimpleNamespace(band_refinement_config=raw, band_indices=None, n_orb=(2, 0),
                          raw={}, band_plot_config={}, harmonics_config={}, heff_file='unused')
    moire = pipeline.MoireConfig(Q_set1=np.zeros((1, 2)), Q_set2=np.zeros((0, 2)),
                               n_orb1=2, n_orb2=0, kpoints=np.zeros((2, 2)))
    monkeypatch.setattr(pipeline, '_refinement_target_hamiltonians',
                        lambda *a: (target, np.linalg.eigvalsh(target), 'test'))
    monkeypatch.setattr(pipeline, '_model_hamiltonians_for_kpoints',
                        lambda *a: base + term.r_value_real * response)
    monkeypatch.setattr(pipeline, '_prepare_band_state', lambda *a: None)
    monkeypatch.setattr(pipeline, '_apply_refinement_acceptance_guard',
                        lambda **kw: (kw['candidate_y'], None))
    monkeypatch.setattr(pipeline, '_sync_complete_response_coefficients_to_terms', lambda *a: None)
    original = None
    if compiled:
        class Fit:
            basis_hash = 'toy'
            coefficients = np.array([0.])
            fit_solver_channel_ids = ('x',)
            fit_selected_channel_ids = ('x',)

            def with_coefficients(self, values, **kw):
                new = Fit()
                new.coefficients = np.asarray(values).copy()
                return new

        model._fitted_response_model = original = Fit()
        model._compiled_response_basis = SimpleNamespace(
            basis_hash='toy', response_tensor=lambda k: response[:, None],
            channels=[SimpleNamespace(metadata={'tag': 'Kinect'}, component='real',
                                      classification='confirmed_nonzero', channel_id='x')],
            candidate_artifact={}, response_scales=np.ones(1), channel_ids=('x',))

    observed = {}

    def optimizer(fun, x0, **kwargs):
        if crossing:
            fun(np.array([-2.]))
            pytest.fail('new degeneracy must abort refinement')
        observed['rows'] = len(fun(x0))
        # One retained band residual plus the full 2-point complex matrix loss.
        assert observed['rows'] == 1 + 2 * target.size
        if 'jac' in kwargs:
            jac = kwargs['jac'](x0)
            if hasattr(jac, 'toarray'):
                jac = jac.toarray()
            numerical = (fun(x0 + 1e-6) - fun(x0 - 1e-6)) / 2e-6
            np.testing.assert_allclose(jac[:, 0], numerical, atol=1e-6)
        return SimpleNamespace(x=np.asarray(x0) + .2, nfev=1, njev=1,
                               cost=1., status=1, message='test')

    monkeypatch.setattr(pipeline.scipy.optimize, 'least_squares', optimizer)
    report = pipeline.refine_band_coefficients(moire, cfg, model)
    return report, model, original, observed


@pytest.mark.parametrize('compiled', [False, True])
def test_legacy_routes_filter_band_samples_but_keep_matrix_samples(monkeypatch, compiled):
    report, model, _, observed = legacy_fit(monkeypatch, compiled=compiled)
    assert not report.get('skipped', False)
    assert report['degeneracy_guard']['active_point_indices'] == [1]
    assert observed['rows'] == 17


@pytest.mark.parametrize('compiled', [False, True])
def test_legacy_routes_keep_coefficients_on_new_degeneracy(monkeypatch, compiled):
    report, model, original, _ = legacy_fit(monkeypatch, compiled=compiled, crossing=True)
    assert report['skipped'] and not report['accepted']
    assert report['reason'] == 'new_target_degeneracy'
    assert model.terms['t'].r_value_real == 0.
    if compiled:
        assert model._fitted_response_model is original


@pytest.mark.parametrize('compiled', [False, True])
def test_legacy_routes_skip_degenerate_alignment_anchor(monkeypatch, compiled):
    report, model, original, observed = legacy_fit(monkeypatch, compiled=compiled, alignment='top')
    assert report['reason'] == 'degenerate_alignment_anchor'
    assert not observed
    assert model.terms['t'].r_value_real == 0.
