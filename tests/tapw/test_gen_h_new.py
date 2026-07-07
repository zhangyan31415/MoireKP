from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

import tapw.workflows.band as band_mod
from tapw.workflows.band import BandStructureCalculator

ROOT = Path(__file__).resolve().parents[2]


def _reference_transform(hamk: np.ndarray, samk: np.ndarray) -> np.ndarray:
    s_eig, s_vec = scipy.linalg.eigh(samk, check_finite=False)
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s_eig))
    uminvud = s_vec @ s_inv_sqrt @ s_vec.conj().T
    return uminvud @ hamk @ uminvud


def test_gen_h_new_cpu_matches_reference_for_complex_generalized_problem():
    rng = np.random.default_rng(20260315)

    raw_h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hamk = raw_h + raw_h.conj().T

    raw_s = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    samk = raw_s @ raw_s.conj().T
    samk += 2.0 * np.eye(4)

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    expected = np.linalg.eigvalsh(_reference_transform(hamk, samk))
    actual = np.linalg.eigvalsh(calculator.gen_H_new_cpu(hamk, samk))

    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_gen_h_new_cpu_matches_legacy_reference_matrix_for_complex_generalized_problem():
    rng = np.random.default_rng(20260315)

    raw_h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hamk = raw_h + raw_h.conj().T

    raw_s = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    samk = raw_s @ raw_s.conj().T
    samk += 2.0 * np.eye(4)

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    expected = _reference_transform(hamk, samk)
    actual = calculator.gen_H_new_cpu(hamk, samk)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_gen_h_new_cpu_does_not_force_extra_hermitian_symmetrization():
    rng = np.random.default_rng(1)

    raw_h = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    hamk = raw_h + raw_h.conj().T

    # Deliberately ill-conditioned SPD overlap so the legacy transform keeps a
    # measurable anti-Hermitian residue instead of being symmetrized away.
    q_mat, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    samk = q_mat @ np.diag(np.geomspace(1.0e-10, 1.0e2, 8)) @ q_mat.conj().T

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)

    actual = calculator.gen_H_new_cpu(hamk, samk)
    hermitian_part = (actual + actual.conj().T) / 2

    assert np.max(np.abs(actual - hermitian_part)) > 1.0e-8


def test_gen_h_new_cpu_uses_spin_degenerate_real_overlap_fast_path(monkeypatch):
    rng = np.random.default_rng(20260707)

    raw_h = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    hamk = raw_h + raw_h.conj().T

    raw_s0 = rng.normal(size=(3, 3))
    s0 = raw_s0 @ raw_s0.T + 3.0 * np.eye(3)
    samk = scipy.linalg.block_diag(s0, s0)

    expected = _reference_transform(hamk, samk)
    seen_shapes = []
    original_eigh = scipy.linalg.eigh

    def recording_eigh(matrix, *args, **kwargs):
        seen_shapes.append(np.asarray(matrix).shape)
        return original_eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "eigh", recording_eigh)

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    actual = calculator.gen_H_new_cpu(hamk, samk)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert seen_shapes == [(3, 3)]


def test_gen_h_new_cpu_uses_spin_degenerate_complex_overlap_fast_path(monkeypatch):
    rng = np.random.default_rng(20260708)

    raw_h = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    hamk = raw_h + raw_h.conj().T

    raw_s0 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    s0 = raw_s0 @ raw_s0.conj().T + 3.0 * np.eye(3)
    samk = scipy.linalg.block_diag(s0, s0)

    expected = _reference_transform(hamk, samk)
    seen_shapes = []
    original_eigh = scipy.linalg.eigh

    def recording_eigh(matrix, *args, **kwargs):
        seen_shapes.append(np.asarray(matrix).shape)
        return original_eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "eigh", recording_eigh)

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    actual = calculator.gen_H_new_cpu(hamk, samk)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert seen_shapes == [(3, 3)]


def test_spin_degenerate_overlap_projection_matches_full_sparse_projection():
    rng = np.random.default_rng(20260708)

    raw_g0 = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    g0 = scipy.sparse.csr_matrix(raw_g0)
    g = scipy.sparse.block_diag((g0, g0), format="csr")

    raw_s0 = rng.normal(size=(5, 5))
    s0 = raw_s0 @ raw_s0.T + 5.0 * np.eye(5)
    s = scipy.sparse.block_diag(
        (scipy.sparse.csr_matrix(s0), scipy.sparse.csr_matrix(s0)),
        format="csr",
    )

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(use_sparse_dot_mkl=False)
    calculator.TAPW_parameters = SimpleNamespace(
        g_matrix=g,
        g_matrix_conj=g.conj().T.tocsr(),
    )

    expected = calculator.cal_TAPW_hamiltonian_k_cpu(s)
    actual = calculator._cal_TAPW_spin_degenerate_overlap_k_cpu(s)

    assert actual is not None
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(actual[:3, 3:], 0.0, atol=1e-12)
    assert np.allclose(actual[3:, :3], 0.0, atol=1e-12)
    assert np.allclose(actual[:3, :3], actual[3:, 3:], rtol=1e-12, atol=1e-12)


def test_sparse_dot_mkl_import_normalizes_conda_mkl_interface_layer():
    env = os.environ.copy()
    env["MKL_INTERFACE_LAYER"] = "LP64,GNU"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import tapw.workflows.band; print(os.environ.get('MKL_INTERFACE_LAYER'))",
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == "LP64"
    assert "MKL_INTERFACE_LAYER value LP64,GNU invalid" not in completed.stderr


def test_forced_sparse_dot_requires_sparse_dot_mkl(monkeypatch):
    monkeypatch.setattr(band_mod, "_HAS_SPARSE_DOT_MKL", False)
    monkeypatch.setattr(band_mod, "dot_product_mkl", None)

    g = scipy.sparse.eye(4, format="csr")
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(use_sparse_dot_mkl=False)
    calculator.TAPW_parameters = SimpleNamespace(
        g_matrix=g,
        g_matrix_conj=g.conj().T.tocsr(),
    )

    with pytest.raises(RuntimeError) as error:
        calculator.cal_TAPW_hamiltonian_k_cpu(
            scipy.sparse.eye(4, format="csr"),
            force_sparse_dot=True,
        )

    message = str(error.value)
    assert "sparse_dot_mkl" in message
    assert "python -m pip install sparse-dot-mkl" in message
    assert "g @ H/S @ g†" in message
    assert "compute.use_sparse_dot_mkl: false" in message
    assert "force_sparse_dot=True" in message


def test_configured_sparse_dot_requires_sparse_dot_mkl(monkeypatch):
    monkeypatch.setattr(band_mod, "_HAS_SPARSE_DOT_MKL", False)
    monkeypatch.setattr(band_mod, "dot_product_mkl", None)

    g = scipy.sparse.eye(4, format="csr")
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(use_sparse_dot_mkl=True)
    calculator.TAPW_parameters = SimpleNamespace(
        g_matrix=g,
        g_matrix_conj=g.conj().T.tocsr(),
    )

    with pytest.raises(RuntimeError) as error:
        calculator.cal_TAPW_hamiltonian_k_cpu(scipy.sparse.eye(4, format="csr"))

    message = str(error.value)
    assert "sparse_dot_mkl" in message
    assert "python -m pip install sparse-dot-mkl" in message
    assert "compute.use_sparse_dot_mkl: false" in message


def test_spin_degenerate_overlap_forced_sparse_dot_requires_sparse_dot_mkl(monkeypatch):
    monkeypatch.setattr(band_mod, "_HAS_SPARSE_DOT_MKL", False)
    monkeypatch.setattr(band_mod, "dot_product_mkl", None)

    g0 = scipy.sparse.eye(2, 3, format="csr")
    g = scipy.sparse.block_diag((g0, g0), format="csr")
    s0 = scipy.sparse.eye(3, format="csr")
    overlap = scipy.sparse.block_diag((s0, s0), format="csr")

    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(use_sparse_dot_mkl=False)
    calculator.TAPW_parameters = SimpleNamespace(
        g_matrix=g,
        g_matrix_conj=g.conj().T.tocsr(),
    )

    with pytest.raises(RuntimeError) as error:
        calculator._cal_TAPW_spin_degenerate_overlap_k_cpu(
            overlap,
            force_sparse_dot=True,
        )

    message = str(error.value)
    assert "sparse_dot_mkl" in message
    assert "python -m pip install sparse-dot-mkl" in message
    assert "compute.use_sparse_dot_mkl: false" in message


def test_getk_final_hs_forces_sparse_dot_when_configured():
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(
        orthogonal_basis=False,
        ge=False,
        use_sparse_dot_mkl=True,
    )
    calculator._build_getk_phase_context = lambda _k: object()

    def assemble(_blocks, _phase_ctx, type="H"):
        return type

    projection_calls = []

    def project_cpu(matrix, **kwargs):
        projection_calls.append((matrix, bool(kwargs.get("force_sparse_dot", False))))
        return np.eye(2)

    overlap_calls = []

    def project_overlap(matrix, **kwargs):
        overlap_calls.append((matrix, bool(kwargs.get("force_sparse_dot", False))))
        return np.eye(2)

    calculator._assemble_sparse_realspace_matrix = assemble
    calculator.cal_TAPW_hamiltonian_k_cpu = project_cpu
    calculator._cal_TAPW_spin_degenerate_overlap_k_cpu = project_overlap
    calculator.gen_H_new = lambda h, s: ("orthogonalized", h, s)

    result, overlap = BandStructureCalculator.Getk_super_gauge_sparse_final_HS(
        calculator,
        Hr=object(),
        Sr=object(),
        k=np.array([0.0, 0.0, 0.0]),
    )

    assert projection_calls == [("H", True)]
    assert overlap_calls == [("S", True)]
    assert result[0] == "orthogonalized"
    assert overlap is None


def test_getk_final_hs_allows_sparse_dot_fallback_when_disabled():
    calculator = BandStructureCalculator.__new__(BandStructureCalculator)
    calculator.config = SimpleNamespace(
        orthogonal_basis=False,
        ge=False,
        use_sparse_dot_mkl=False,
    )
    calculator._build_getk_phase_context = lambda _k: object()

    def assemble(_blocks, _phase_ctx, type="H"):
        return type

    projection_calls = []

    def project_cpu(matrix, **kwargs):
        projection_calls.append((matrix, bool(kwargs.get("force_sparse_dot", False))))
        return np.eye(2)

    overlap_calls = []

    def project_overlap(matrix, **kwargs):
        overlap_calls.append((matrix, bool(kwargs.get("force_sparse_dot", False))))
        return np.eye(2)

    calculator._assemble_sparse_realspace_matrix = assemble
    calculator.cal_TAPW_hamiltonian_k_cpu = project_cpu
    calculator._cal_TAPW_spin_degenerate_overlap_k_cpu = project_overlap
    calculator.gen_H_new = lambda h, s: ("orthogonalized", h, s)

    BandStructureCalculator.Getk_super_gauge_sparse_final_HS(
        calculator,
        Hr=object(),
        Sr=object(),
        k=np.array([0.0, 0.0, 0.0]),
    )

    assert projection_calls == [("H", False)]
    assert overlap_calls == [("S", False)]
