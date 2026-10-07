"""Orientation and sewing regressions for the Wilson implementation used in the paper."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from tapw.chern_post import wilson_loop


def _phase_spectrum(phases):
    # Comparing points on the unit circle avoids a spurious 0/1 branch mismatch.
    return np.sort_complex(np.exp(2j * np.pi * np.asarray(phases)))


def test_wilson_spinor_orientation_and_reversed_loop():
    theta = 0.91
    phi = np.linspace(0.0, 2 * np.pi, 513)
    vectors = np.empty((len(phi), 2, 1), dtype=complex)
    vectors[:, 0, 0] = np.cos(theta / 2)
    vectors[:, 1, 0] = np.exp(1j * phi) * np.sin(theta / 2)
    forward = wilson_loop(vectors)
    backward = wilson_loop(vectors[::-1])
    # The code transports with <u_next|u_current>; the oriented spinor loop
    # therefore has +integral(A) for A=i<u|du>, namely -half the solid angle.
    expected = -0.5 * (1 - np.cos(theta))
    np.testing.assert_allclose(_phase_spectrum(forward), _phase_spectrum([expected]), atol=2.0e-5)
    np.testing.assert_allclose(_phase_spectrum(backward), _phase_spectrum(-forward), atol=1.0e-13)


def test_wilson_nonabelian_gauge_invariance_and_inverse_sewing():
    rng = np.random.default_rng(20260928)
    vectors = np.asarray([
        np.linalg.qr(rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2)))[0]
        for _ in range(9)
    ])
    sewing = np.diag(np.exp(1j * np.array([0.2, -0.3, 0.7, 0.9, -0.5])))
    # Sewing maps the endpoint frame back to the initial coordinate frame.
    vectors[-1] = sewing.conj().T @ vectors[0]
    forward = wilson_loop(vectors, boundary_sewing=sparse.csr_matrix(sewing))
    backward = wilson_loop(vectors[::-1], boundary_sewing=sparse.csr_matrix(sewing.conj().T))
    gauges = np.asarray([
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in vectors
    ])
    rotated = wilson_loop(vectors @ gauges, boundary_sewing=sparse.csr_matrix(sewing))
    np.testing.assert_allclose(_phase_spectrum(rotated), _phase_spectrum(forward), atol=2.0e-13)
    np.testing.assert_allclose(_phase_spectrum(backward), _phase_spectrum(-forward), atol=2.0e-13)
    # A wrong sewing orientation must fail before polar projection could hide it.
    with pytest.raises(ValueError, match="not isometric"):
        wilson_loop(vectors, boundary_sewing=sparse.csr_matrix(sewing.conj().T))


def test_wilson_reports_raw_boundary_singular_value_before_polar():
    frames = np.broadcast_to(np.eye(2, dtype=complex), (3, 2, 2)).copy()
    boundary = sparse.diags([1.0, 0.8], dtype=complex)
    with pytest.raises(ValueError, match=r"max\|sigma-1\|=2\.000000e-01"):
        wilson_loop(frames, boundary_sewing=boundary)
