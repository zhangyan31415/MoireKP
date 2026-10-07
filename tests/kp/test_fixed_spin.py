import numpy as np
import pytest

from kp.experimental.fixed_spin import molecular_frame, product_frame, reduce_fixed_frame


def test_explicit_molecular_orbitals_and_spin_closure():
    spatial, labels = molecular_frame(components=('px', 'py'), parities=(1,), radial=[3., 4.])
    assert spatial.shape == (80, 4)
    assert len(labels) == 4
    # L1, symmetric S px: both S atoms, both radial functions.
    expected = np.zeros(80)
    expected[[2, 29]] = .6 / np.sqrt(2)
    expected[[5, 32]] = .8 / np.sqrt(2)
    np.testing.assert_allclose(spatial[:, 0], expected)
    f, s0 = product_frame(spatial)
    sz = np.diag(np.r_[np.ones(80), -np.ones(80)])
    np.testing.assert_allclose(f.conj().T @ f, np.eye(8), atol=1e-14)
    np.testing.assert_allclose(sz @ f, f @ s0, atol=1e-14)


def test_fixed_spin_does_not_remove_soc_and_reconstruction_dresses_spin():
    # Spatial order a,b; spin-major full basis a-up,b-up,a-down,b-down.
    f, s0 = product_frame(np.array([[1.], [0.]]))
    h = np.diag([0., 3., .1, 4.]).astype(complex)
    h[0, 2] = h[2, 0] = .07  # retained spin flip
    h[0, 3] = h[3, 0] = .5   # eliminated opposite spin
    out = reduce_fixed_frame(h, f, np.diag([1., 1., -1., -1.]), e_ref=0.)
    assert out['direct_h'][0, 1] == pytest.approx(.07)
    np.testing.assert_allclose(out['bare_spin'], s0)
    np.testing.assert_allclose(out['schur_h'], [[-.0625, .07], [.07, .1]])
    z = out['embedding']
    np.testing.assert_allclose(z.conj().T @ z, np.eye(2), atol=1e-13)
    np.testing.assert_allclose(out['dressed_h'], z.conj().T @ h @ z)
    assert out['dressed_spin'][0, 0].real == pytest.approx(63/65)
    assert out['pole_distance_ev'] == pytest.approx(3.)


def test_rejects_nonorthonormal_basis_and_near_pole():
    f, _ = product_frame(np.array([[1.], [0.]]))
    with pytest.raises(ValueError, match='orthonormal'):
        reduce_fixed_frame(np.eye(4), 2*f, np.eye(4), e_ref=0.)
    with pytest.raises(ValueError, match='pole'):
        reduce_fixed_frame(np.diag([1., 0., 2., 3.]), f, np.eye(4), e_ref=0.)


def test_spin_conserving_elimination_preserves_physical_sz():
    f, s0 = product_frame(np.array([[1.], [0.]]))
    h = np.diag([0., 3., .1, 4.]).astype(complex)
    h[0, 1] = h[1, 0] = .5
    h[2, 3] = h[3, 2] = .7
    out = reduce_fixed_frame(h, f, np.array([1., 1., -1., -1.]), e_ref=0.)
    np.testing.assert_allclose(out['dressed_spin'], s0, atol=1e-14)


def test_no_elimination_recovers_full_hamiltonian():
    h = np.array([[1., .2j], [-.2j, 2.]])
    out = reduce_fixed_frame(h, np.eye(2), np.array([1., -1.]), e_ref=0.)
    np.testing.assert_allclose(out['dressed_h'], h)
    np.testing.assert_allclose(out['dressed_spin'], np.diag([1., -1.]))
    assert out['pole_distance_ev'] is None


@pytest.mark.parametrize('components,parities,dimension', [
    (('px','py'), (1,), 16),
    (('px','py'), (1,-1), 32),
    (('px','py','pz'), (1,-1), 48),
])
def test_complete_radial_orbital_families(components, parities, dimension):
    spatial, _ = molecular_frame(components=components, parities=parities)
    f, _ = product_frame(spatial)
    assert f.shape == (160, dimension)
    np.testing.assert_allclose(f.conj().T @ f, np.eye(dimension), atol=1e-14)
