"""Constant physical-spin product frames in orthonormal TAPW coordinates.

The bare spin is exact in this selected parent space. Eliminating additional
orbitals generally dresses the physical spin; the reconstructed operator is
returned separately and is never replaced with a Pauli label.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
from scipy import sparse


def molecular_frame(*, components=('px', 'py'), parities=(1,), radial=None):
    """ZrS2 spatial frame, two layers of S-s2p2d1,Zr-s3p2d1,S-s2p2d1.

    Rows are layer-major (40 AO coordinates/layer). Both S sites share the
    same radial combination and Cartesian axes. ``radial=None`` keeps both
    radial p functions independently. These are orthogonal-coordinate weights,
    not nonorthogonal atomic populations or literal Wannier functions.
    """
    if not components or len(set(components)) != len(components) or any(
        x not in ('px', 'py', 'pz') for x in components
    ):
        raise ValueError('components must be distinct Cartesian p orbitals')
    if not parities or len(set(parities)) != len(parities) or any(p not in (-1, 1) for p in parities):
        raise ValueError('parities must be distinct +1/-1 values')
    if radial is None:
        radial_vectors = np.eye(2)
    else:
        r = np.asarray(radial, dtype=float)
        if r.shape != (2,) or not np.all(np.isfinite(r)) or la.norm(r) == 0:
            raise ValueError('radial must be two finite nonzero real coefficients')
        radial_vectors = (r / la.norm(r))[None, :]
    cols, labels = [], []
    for layer in range(2):
        for parity in parities:
            for component in components:
                for ri, r in enumerate(radial_vectors):
                    v = np.zeros(80)
                    axis = ('px', 'py', 'pz').index(component)
                    for site, factor in ((0, 1), (27, parity)):
                        for radial_index in range(2):
                            v[40*layer+site+2+3*radial_index+axis] = factor*r[radial_index]/np.sqrt(2)
                    cols.append(v)
                    labels.append(f'L{layer+1}:S1{ "+" if parity == 1 else "-" }S2:{component}:radial{ri}')
    return np.column_stack(cols), labels


def product_frame(spatial):
    """Return F=I_spin tensor spatial and its exact spin-major bare Sz."""
    spatial = np.asarray(spatial, dtype=complex)
    if spatial.ndim != 2 or not np.allclose(spatial.conj().T@spatial, np.eye(spatial.shape[1]), atol=1e-12, rtol=0):
        raise ValueError('spatial columns must be orthonormal')
    f = la.block_diag(spatial, spatial)
    n = spatial.shape[1]
    return f, np.diag(np.r_[np.ones(n), -np.ones(n)])


def _apply_spin(spin, frame):
    return spin[:, None]*frame if spin.ndim == 1 else spin@frame


def reduce_fixed_frame(ham, frame, spin, *, e_ref, complement=None, pole_floor_ev=1e-6):
    """Direct/Schur/reconstructed reductions with consistent observables.

    Z=(F+G Y)(I+Y†Y)^(-1/2), Y=(Eref-H_GG)^(-1)H_GF.
    Z†HZ equals the energy-linearized Lowdin Hamiltonian at Eref. Z†sZ is
    its dressed physical spin. All output energies are in eV.
    """
    ham, frame, spin = np.asarray(ham), np.asarray(frame), np.asarray(spin)
    if ham.ndim != 2 or ham.shape[0] != ham.shape[1] or not np.allclose(ham, ham.conj().T, atol=1e-10, rtol=0):
        raise ValueError('ham must be Hermitian and square')
    n, r = frame.shape
    if ham.shape != (n, n) or not np.allclose(frame.conj().T@frame, np.eye(r), atol=1e-10, rtol=0):
        raise ValueError('frame must be an orthonormal isometry compatible with ham')
    if spin.shape not in ((n,), (n, n)):
        raise ValueError('spin must be a diagonal vector or square matrix')
    if not np.isfinite(e_ref) or pole_floor_ev <= 0:
        raise ValueError('finite reference energy and positive pole floor required')
    if complement is None:
        complement = la.null_space(frame.conj().T)
    g = np.asarray(complement)
    if g.shape != (n, n-r) or not np.allclose(g.conj().T@g, np.eye(n-r), atol=1e-10, rtol=0) or la.norm(frame.conj().T@g) > 1e-9:
        raise ValueError('complement must be complete and orthonormal')
    # Sparse multiplication exploits Q-local support without requiring the
    # production projector/downfold code or changing its current defaults.
    fs, gs = sparse.csr_matrix(frame), sparse.csr_matrix(g)
    hf = ham@fs
    hpp = frame.conj().T@hf
    hgp = g.conj().T@hf
    hgg = (gs.conj().T@ham)@gs
    if n == r:
        y = np.zeros((0, r), complex)
        pole = None
    else:
        e, u = la.eigh(hgg, check_finite=False)
        delta = e_ref-e
        pole = float(np.min(abs(delta)))
        if pole < pole_floor_ev:
            raise ValueError(f'excluded-space pole at {pole:g} eV from Eref')
        y = u@((u.conj().T@hgp)/delta[:, None])
    schur = hpp+hgp.conj().T@y
    mass = np.eye(r)+y.conj().T@y
    em, um = la.eigh(mass, check_finite=False)
    inv_sqrt = (um/np.sqrt(em))@um.conj().T
    z = (frame+g@y)@inv_sqrt
    dressed = inv_sqrt@(schur+e_ref*(mass-np.eye(r)))@inv_sqrt
    return dict(direct_h=(hpp+hpp.conj().T)/2,
                schur_h=(schur+schur.conj().T)/2,
                dressed_h=(dressed+dressed.conj().T)/2,
                bare_spin=frame.conj().T@_apply_spin(spin, frame),
                dressed_spin=z.conj().T@_apply_spin(spin, z),
                embedding=z, pole_distance_ev=pole,
                mass_eigenvalue_range=[float(em.min()), float(em.max())])
