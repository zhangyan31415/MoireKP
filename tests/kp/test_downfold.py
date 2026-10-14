from __future__ import annotations

import numpy as np
import pytest


def test_projected_hamiltonian_matches_dense_contraction() -> None:
    from kp.blocks.downfold import _projected_hamiltonian

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    ham = raw + raw.conj().T
    u_left, _ = np.linalg.qr(rng.normal(size=(9, 4)) + 1j * rng.normal(size=(9, 4)))
    u_right, _ = np.linalg.qr(rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3)))

    expected = u_left.conj().T @ ham @ u_right
    actual = _projected_hamiltonian(ham, u_left, u_right, chunk_cols=2)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_grouped_projected_hamiltonian_preserves_original_column_order() -> None:
    from kp.blocks.downfold import _grouped_projected_hamiltonian

    rng = np.random.default_rng(17)
    raw = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    ham = raw + raw.conj().T
    u_left = np.zeros((6, 4), dtype=np.complex128)
    u_right = np.zeros((6, 4), dtype=np.complex128)
    q0, _ = np.linalg.qr(rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4)))
    q1, _ = np.linalg.qr(rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4)))
    # Interleave columns by support, matching the Gamma projector ordering that
    # caught an earlier block-local prototype.
    u_left[:3, [0, 2]] = q0[:, :2]
    u_left[3:, [1, 3]] = q1[:, :2]
    u_right[:3, [0, 2]] = q0[:, 2:]
    u_right[3:, [1, 3]] = q1[:, 2:]

    expected = u_left.conj().T @ ham @ u_right
    actual = _grouped_projected_hamiltonian(ham, u_left, u_right)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_downfold_first_order_matches_dense_projection() -> None:
    from kp.blocks.downfold import DownfoldingOptions, downfold_from_projectors

    rng = np.random.default_rng(11)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    ham = raw + raw.conj().T
    u_low, _ = np.linalg.qr(rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3)))

    result = downfold_from_projectors(
        ham,
        u_low,
        None,
        DownfoldingOptions(method="first_order"),
    )

    expected = 0.5 * ((u_low.conj().T @ ham @ u_low) + (u_low.conj().T @ ham @ u_low).conj().T)
    np.testing.assert_allclose(result.heff, expected, atol=1e-12)


def test_downfold_fixed_schur_prepares_projector_groups_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from kp.blocks import downfold

    rng = np.random.default_rng(21)
    raw = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    ham = raw + raw.conj().T
    u_low = np.zeros((10, 2), dtype=np.complex128)
    u_high = np.zeros((10, 4), dtype=np.complex128)
    q0, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    q1, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    u_low[:5, 0] = q0[:, 0]
    u_low[5:, 1] = q1[:, 0]
    u_high[:5, :2] = q0[:, 1:3]
    u_high[5:, 2:] = q1[:, 1:3]

    real_groups = downfold._build_projector_groups
    calls = 0

    def counting_groups(basis: np.ndarray):
        nonlocal calls
        calls += 1
        return real_groups(basis)

    monkeypatch.setattr(downfold, "_build_projector_groups", counting_groups)

    downfold.downfold_from_projectors(
        ham,
        u_low,
        u_high,
        downfold.DownfoldingOptions(method="fixed_schur", e_ref=0.1),
    )

    assert calls == 2


def test_grouped_projected_hamiltonian_reuses_groups_for_self_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kp.blocks import downfold

    rng = np.random.default_rng(23)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    ham = raw + raw.conj().T
    basis = np.zeros((8, 3), dtype=np.complex128)
    q0, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    q1, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    basis[:4, [0, 2]] = q0[:, :2]
    basis[4:, [1]] = q1[:, :1]

    real_groups = downfold._build_projector_groups
    calls = 0

    def counting_groups(projector: np.ndarray):
        nonlocal calls
        calls += 1
        return real_groups(projector)

    monkeypatch.setattr(downfold, "_build_projector_groups", counting_groups)

    actual = downfold._grouped_projected_hamiltonian(ham, basis)

    assert calls == 1
    np.testing.assert_allclose(actual, basis.conj().T @ ham @ basis, atol=1e-12)


def test_fixed_schur_skips_expensive_pole_diagnostics_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from kp.blocks.downfold import DownfoldingOptions, downfold_blocks

    def fail_eigvalsh(_matrix: np.ndarray) -> np.ndarray:
        raise AssertionError("pole eigenspectrum should not be computed by default")

    def fail_cond(_matrix: np.ndarray) -> float:
        raise AssertionError("condition number should not be computed by default")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail_eigvalsh)
    monkeypatch.setattr(np.linalg, "cond", fail_cond)

    h_pp = np.array([[0.0]], dtype=np.complex128)
    h_ph = np.array([[0.2, 0.1]], dtype=np.complex128)
    h_hh = np.diag([0.1005, 0.3]).astype(np.complex128)

    result = downfold_blocks(
        h_pp,
        h_ph,
        h_hh,
        DownfoldingOptions(
            method="fixed_schur",
            e_ref=0.1,
            pole_warning_mev=1.0,
            pole_danger_mev=0.1,
        ),
    )

    assert result.pole_distance_min_mev is None
    assert result.pole_condition_number is None
    assert result.near_pole is False
    assert result.danger_pole is False
    assert result.warnings == []


def test_fixed_schur_computes_pole_diagnostics_when_requested() -> None:
    from kp.blocks.downfold import DownfoldingOptions, downfold_blocks

    h_pp = np.array([[0.0]], dtype=np.complex128)
    h_ph = np.array([[0.2, 0.1]], dtype=np.complex128)
    h_hh = np.diag([0.1005, 0.3]).astype(np.complex128)

    result = downfold_blocks(
        h_pp,
        h_ph,
        h_hh,
        DownfoldingOptions(
            method="fixed_schur",
            e_ref=0.1,
            pole_warning_mev=1.0,
            pole_danger_mev=0.1,
            compute_pole_diagnostics=True,
        ),
    )

    assert result.pole_condition_number is None
    assert result.near_pole is True
    assert result.danger_pole is False
    assert result.warnings
    assert result.pole_distance_min_mev == pytest.approx(0.5)


def test_fixed_schur_fail_on_near_pole_forces_pole_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    from kp.blocks.downfold import DownfoldingOptions, NearPoleError, downfold_blocks

    real_eigvalsh = np.linalg.eigvalsh
    eigvalsh_calls: list[np.ndarray] = []

    def tracking_eigvalsh(matrix: np.ndarray) -> np.ndarray:
        eigvalsh_calls.append(np.asarray(matrix).copy())
        return real_eigvalsh(matrix)

    def fail_cond(_matrix: np.ndarray) -> float:
        raise AssertionError("condition number should remain opt-in")

    monkeypatch.setattr(np.linalg, "eigvalsh", tracking_eigvalsh)
    monkeypatch.setattr(np.linalg, "cond", fail_cond)

    h_pp = np.array([[0.0]], dtype=np.complex128)
    h_ph = np.array([[0.2, 0.1]], dtype=np.complex128)
    h_hh = np.diag([0.1005, 0.3]).astype(np.complex128)

    with pytest.raises(NearPoleError, match=r"min \|E_ref - E_high\| = 0\.500 meV"):
        downfold_blocks(
            h_pp,
            h_ph,
            h_hh,
            DownfoldingOptions(
                method="fixed_schur",
                e_ref=0.1,
                pole_warning_mev=1.0,
                pole_danger_mev=0.1,
                fail_on_near_pole=True,
                compute_pole_diagnostics=False,
                compute_condition_number=False,
            ),
        )

    assert len(eigvalsh_calls) == 1


def test_fixed_schur_computes_condition_number_when_requested() -> None:
    from kp.blocks.downfold import DownfoldingOptions, downfold_blocks

    h_pp = np.array([[0.0]], dtype=np.complex128)
    h_ph = np.array([[0.2, 0.1]], dtype=np.complex128)
    h_hh = np.diag([0.05, 0.3]).astype(np.complex128)
    e_ref = 0.1

    result = downfold_blocks(
        h_pp,
        h_ph,
        h_hh,
        DownfoldingOptions(
            method="fixed_schur",
            e_ref=e_ref,
            compute_condition_number=True,
        ),
    )

    expected_cond = np.linalg.cond(e_ref * np.eye(2, dtype=np.complex128) - h_hh)
    assert result.pole_condition_number == pytest.approx(expected_cond)

@pytest.mark.parametrize('method',['fixed_schur','linearized_lowdin'])
def test_exact_pole_fails_even_without_expensive_diagnostics(method):
    from kp.blocks.downfold import downfold_blocks,DownfoldingOptions,NearPoleError
    with pytest.raises(NearPoleError,match='singular'):
        downfold_blocks(np.zeros((1,1)),np.array([[.1]]),np.zeros((1,1)),
                        DownfoldingOptions(method=method,e_ref=0.))

@pytest.mark.parametrize('method',['first_order','fixed_schur','linearized_lowdin'])
def test_nonfinite_downfold_input_is_rejected(method):
    from kp.blocks.downfold import downfold_blocks,DownfoldingOptions
    with pytest.raises(ValueError,match='finite'):
        downfold_blocks(np.array([[np.nan]]),np.array([[.1]]),np.array([[1.]]),
                        DownfoldingOptions(method=method,e_ref=0.))

def test_linearized_reconstruction_is_rayleigh_ritz_and_shift_covariant():
    from kp.blocks.downfold import downfold_blocks,DownfoldingOptions
    rng=np.random.default_rng(408);h=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));h=(h+h.conj().T)/2
    ref=5.;x=np.linalg.solve(ref*np.eye(4)-h[2:,2:],h[2:,:2]);e,u=np.linalg.eigh(np.eye(2)+x.conj().T@x);w=np.vstack([np.eye(2),x])@(u/np.sqrt(e))@u.conj().T
    result=downfold_blocks(h[:2,:2],h[:2,2:],h[2:,2:],DownfoldingOptions(e_ref=ref))
    np.testing.assert_allclose(w.conj().T@w,np.eye(2),atol=1e-12)
    np.testing.assert_allclose(result.heff,w.conj().T@h@w,atol=1e-12)
    c=.31;shifted=h+c*np.eye(6)
    new=downfold_blocks(shifted[:2,:2],shifted[:2,2:],shifted[2:,2:],DownfoldingOptions(e_ref=ref+c))
    np.testing.assert_allclose(new.heff,result.heff+c*np.eye(2),atol=1e-12)
