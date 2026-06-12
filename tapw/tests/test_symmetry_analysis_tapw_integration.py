from types import SimpleNamespace

import numpy as np
import scipy.sparse

import tapw.workflows.symmetry as symmetry_analysis


def test_valley_center_resolution_uses_tapw_parameter_convention(monkeypatch):
    helper = getattr(symmetry_analysis, "resolve_tapw_valley_center", None)
    assert helper is not None

    seen = {}

    class FakeTAPWParameters:
        def __init__(self, structure, config):
            seen["valley"] = config.valley

        def calculate_K_points(self):
            return (
                np.array([10.0, 20.0]),
                np.array([30.0, 40.0]),
                np.array([1.5, -2.5]),
                np.array([3.5, -4.5]),
                np.array([0.0, 0.0]),
            )

    monkeypatch.setattr(symmetry_analysis, "TAPW_parameters", FakeTAPWParameters, raising=False)

    structure = object()
    config = SimpleNamespace(valley=31)
    center = helper(structure, config)

    assert seen["valley"] == 31
    assert np.allclose(center, np.array([1.5, -2.5]))


def test_debug_c3_comparison_hook_reports_zero_for_matching_matrices():
    helper = getattr(symmetry_analysis, "compare_transport_against_legacy_c3", None)
    assert helper is not None

    transport = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    legacy = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    residual = helper(transport, legacy)

    assert residual == 0.0


def test_projected_time_reversal_operator_squares_to_minus_identity_for_spinful_basis():
    helper = getattr(symmetry_analysis, "_projected_time_reversal_operator", None)
    assert helper is not None

    operator = helper(projected_dim=8, spinful=True)
    product = operator @ operator.conj()

    assert scipy.sparse.issparse(operator)
    assert np.allclose(product.toarray(), -np.eye(8, dtype=np.complex128))


def test_build_transport_shortcuts_identity_and_time_reversal(monkeypatch):
    runner = symmetry_analysis.SymmetryAnalysisRunner.__new__(symmetry_analysis.SymmetryAnalysisRunner)
    runner.structure = SimpleNamespace(spin=True)
    runner._operation_mapping_cache = {}
    fake_params = SimpleNamespace(g_matrix=np.zeros((8, 16), dtype=np.complex128))
    fake_calculator = SimpleNamespace(TAPW_parameters=fake_params)
    runner._calculator_for_valley = lambda valley: fake_calculator

    identity = symmetry_analysis.SymmetryAnalysisRunner._build_transport(
        runner,
        {"index": 0, "name": "E", "antiunitary": False},
        valley=5,
        q_target=np.zeros(3),
        q_source=np.zeros(3),
    )

    assert scipy.sparse.issparse(identity)
    assert np.allclose(identity.toarray(), np.eye(8, dtype=np.complex128))


def test_raw_projected_hs_requests_sparse_dot_mkl_projection_path():
    runner = symmetry_analysis.SymmetryAnalysisRunner.__new__(symmetry_analysis.SymmetryAnalysisRunner)
    runner._raw_hs_cache = {}
    calls = {}

    class FakeCalculator:
        hr_supercell = {"H": object()}
        TAPW_parameters = object()

        def _build_getk_phase_context(self, q_local):
            return ("phase", tuple(q_local))

        def _assemble_sparse_realspace_matrix(self, hr_supercell, phase_ctx, type="H"):
            assert hr_supercell is self.hr_supercell
            assert phase_ctx[0] == "phase"
            assert type == "H"
            return scipy.sparse.identity(2, dtype=np.complex128, format="csr")

        def cal_TAPW_hamiltonian_k_cpu(self, h_full, tapw_parameters=None, force_sparse_dot=False):
            calls["force_sparse_dot"] = bool(force_sparse_dot)
            calls["tapw_parameters"] = tapw_parameters
            return np.eye(2, dtype=np.complex128)

    fake_calculator = FakeCalculator()
    runner._calculator_for_valley = lambda valley: fake_calculator

    hamk, samk = symmetry_analysis.SymmetryAnalysisRunner._raw_projected_hs(
        runner,
        valley=5,
        q_local=np.zeros(3),
    )

    assert samk is None
    assert np.allclose(hamk, np.eye(2, dtype=np.complex128))
    assert calls["force_sparse_dot"] is True
    assert calls["tapw_parameters"] is fake_calculator.TAPW_parameters


def test_raw_c3_h_orbit_requests_sparse_dot_mkl_projection_path():
    runner = symmetry_analysis.SymmetryAnalysisRunner.__new__(symmetry_analysis.SymmetryAnalysisRunner)
    runner.hr_supercell = {"H": object()}
    runner._raw_c3_h_orbit_cache = {}
    calls = {}

    class FakeCalculator:
        def Getk_super_gauge_sparse_symm(self, hr_supercell, q_local, force_sparse_dot=False):
            calls["hr_supercell"] = hr_supercell
            calls["q_local"] = np.asarray(q_local, dtype=float)
            calls["force_sparse_dot"] = bool(force_sparse_dot)
            return [
                np.eye(2, dtype=np.complex128),
                2.0 * np.eye(2, dtype=np.complex128),
                3.0 * np.eye(2, dtype=np.complex128),
            ]

    runner._calculator_for_valley = lambda valley: FakeCalculator()

    orbit = symmetry_analysis.SymmetryAnalysisRunner._raw_c3_h_orbit(
        runner,
        valley=5,
        q_local=np.zeros(3),
    )

    assert calls["hr_supercell"] is runner.hr_supercell
    assert np.allclose(calls["q_local"], np.zeros(3))
    assert calls["force_sparse_dot"] is True
    assert len(orbit) == 3
