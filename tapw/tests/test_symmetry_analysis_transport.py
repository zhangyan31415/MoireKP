from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import tapw.workflows.symmetry as symmetry_analysis
from tapw.symmetry.representations import C3_G_matrix


def test_transport_cache_key_distinguishes_valley_and_q_source(monkeypatch):
    runner = symmetry_analysis.SymmetryAnalysisRunner.__new__(symmetry_analysis.SymmetryAnalysisRunner)
    runner.structure = SimpleNamespace(spin=False)
    runner.config = SimpleNamespace(twist=SimpleNamespace(bravais="hex"))
    runner._operation_mapping_cache = {}
    runner._transport_cache = {}
    runner._legacy_c3_matrix_cache = {}

    fake_params = SimpleNamespace(g_matrix=scipy.sparse.identity(4, dtype=np.complex128, format="csr"))
    fake_calculator = SimpleNamespace(TAPW_parameters=fake_params)
    runner._calculator_for_valley = lambda valley: fake_calculator

    calls = []

    def fake_direct(self, candidate, valley, q_target, q_source, atom_mapping=None):
        calls.append(
            (
                valley,
                tuple(np.asarray(q_target, dtype=float)),
                tuple(np.asarray(q_source, dtype=float)),
            )
        )
        return scipy.sparse.identity(4, dtype=np.complex128, format="csr")

    monkeypatch.setattr(
        symmetry_analysis.SymmetryAnalysisRunner,
        "_build_projected_transport_direct",
        fake_direct,
    )
    monkeypatch.setattr(
        symmetry_analysis,
        "_build_atom_mapping",
        lambda structure, operation: {},
    )
    candidate = {
        "index": 7,
        "name": "C3z",
        "antiunitary": False,
        "translation_cart": np.zeros(3, dtype=float),
    }
    monkeypatch.setattr(
        symmetry_analysis.SymmetryAnalysisRunner,
        "_validated_generic_c3_transport",
        lambda self, valley, power: fake_direct(
            self,
            candidate,
            valley,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ),
    )

    symmetry_analysis.SymmetryAnalysisRunner._build_transport(
        runner,
        candidate,
        valley=5,
        q_target=np.array([0.1, 0.2, 0.0]),
        q_source=np.array([0.3, 0.4, 0.0]),
    )
    symmetry_analysis.SymmetryAnalysisRunner._build_transport(
        runner,
        candidate,
        valley=5,
        q_target=np.array([0.1, 0.2, 0.0]),
        q_source=np.array([0.35, 0.4, 0.0]),
    )
    symmetry_analysis.SymmetryAnalysisRunner._build_transport(
        runner,
        candidate,
        valley=1,
        q_target=np.array([0.1, 0.2, 0.0]),
        q_source=np.array([0.3, 0.4, 0.0]),
    )

    assert len(calls) == 3


def test_build_transport_keeps_sparse_projected_transport(monkeypatch):
    runner = symmetry_analysis.SymmetryAnalysisRunner.__new__(symmetry_analysis.SymmetryAnalysisRunner)
    runner.structure = SimpleNamespace(spin=False)
    runner.config = SimpleNamespace(twist=SimpleNamespace(bravais="hex"))
    runner._operation_mapping_cache = {}
    runner._transport_cache = {}
    runner._legacy_c3_matrix_cache = {}

    fake_params = SimpleNamespace(g_matrix=scipy.sparse.identity(4, dtype=np.complex128, format="csr"))
    fake_calculator = SimpleNamespace(TAPW_parameters=fake_params)
    runner._calculator_for_valley = lambda valley: fake_calculator

    monkeypatch.setattr(
        symmetry_analysis,
        "_build_atom_mapping",
        lambda structure, operation: {},
    )
    monkeypatch.setattr(
        symmetry_analysis.SymmetryAnalysisRunner,
        "_build_projected_transport_direct",
        lambda self, candidate, valley, q_target, q_source, atom_mapping=None: scipy.sparse.identity(4, dtype=np.complex128, format="csr"),
    )
    candidate = {
        "index": 3,
        "name": "C3z",
        "antiunitary": False,
        "translation_cart": np.zeros(3, dtype=float),
    }
    monkeypatch.setattr(
        symmetry_analysis.SymmetryAnalysisRunner,
        "_validated_generic_c3_transport",
        lambda self, valley, power: scipy.sparse.identity(4, dtype=np.complex128, format="csr"),
    )
    transport = symmetry_analysis.SymmetryAnalysisRunner._build_transport(
        runner,
        candidate,
        valley=5,
        q_target=np.array([0.1, 0.2, 0.0]),
        q_source=np.array([0.1, 0.2, 0.0]),
    )

    assert scipy.sparse.issparse(transport)


def test_physical_momentum_g_mapping_depends_on_k_centers_and_q_points():
    helper = getattr(symmetry_analysis, "build_projected_g_transport_by_physical_momentum", None)
    assert helper is not None

    source_g = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    target_g = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    reciprocal_basis = np.eye(2, dtype=float)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)

    permutation, reciprocal_shifts = helper(
        source_g_vectors=source_g,
        target_g_vectors=target_g,
        source_k_center=np.array([1.0, 0.0], dtype=float),
        target_k_center=np.array([0.0, 1.0], dtype=float),
        q_source_cart=np.array([0.1, 0.0], dtype=float),
        q_target_cart=np.array([0.0, 0.1], dtype=float),
        effective_linear_map=rotation,
        reciprocal_basis=reciprocal_basis,
        tol=1.0e-9,
    )

    assert permutation.tolist() == [0, 1]
    assert np.all(reciprocal_shifts == 0)


def test_generic_c3_transport_matches_legacy_c3_for_supported_gamma_case(monkeypatch):
    helper = getattr(symmetry_analysis, "compare_transport_against_legacy_c3", None)
    assert helper is not None

    legacy = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )
    residual = helper(legacy, legacy)

    assert residual == 0.0


def test_generic_c3_g_transport_matches_legacy_c3_g_matrix_for_gamma():
    helper = getattr(symmetry_analysis, "build_c3_g_transport_from_tapw_convention", None)
    assert helper is not None

    m_g_vec = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    g_vec_list_k1 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
        dtype=float,
    )
    g_vec_list_k2 = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [-1.0, np.sqrt(3.0)],
            [-1.0, -np.sqrt(3.0)],
        ],
        dtype=float,
    )

    generic_k1 = helper(
        g_source=g_vec_list_k1,
        g_target=g_vec_list_k1,
        layer_center=np.zeros(2, dtype=float),
        angle_deg=120.0,
        tol=1.0e-8,
    )
    generic_k2 = helper(
        g_source=g_vec_list_k2,
        g_target=g_vec_list_k2,
        layer_center=np.zeros(2, dtype=float),
        angle_deg=120.0,
        tol=1.0e-8,
    )
    legacy_k1, legacy_k2 = C3_G_matrix(g_vec_list_k1, g_vec_list_k2, m_g_vec, twisted_index_m=4, valley=5)

    assert np.allclose(generic_k1.toarray(), legacy_k1)
    assert np.allclose(generic_k2.toarray(), legacy_k2)


def test_translation_removable_by_origin_shift_detects_removable_c2_seitz_shift():
    helper = getattr(symmetry_analysis, "translation_removable_by_origin_shift", None)
    assert helper is not None

    rotation = np.diag([1.0, -1.0, -1.0])
    translation = np.array([0.0, 0.2, 0.0], dtype=float)

    removable, shift, residual = helper(rotation, translation, tol=1.0e-8)

    assert removable is True
    assert residual < 1.0e-8
    assert np.allclose((np.eye(3) - rotation) @ shift, translation)


def test_build_k_c2t_g_transport_matches_named_formula():
    helper = getattr(symmetry_analysis, "build_k_c2t_g_transport", None)
    assert helper is not None

    k_source = np.array([1.0, 0.0], dtype=float)
    k_target = np.array([-1.0, 0.0], dtype=float)
    g_source = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    g_target = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    linear_map = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float)

    transport, max_delta = helper(
        source_g_vectors=g_source,
        target_g_vectors=g_target,
        source_k_center=k_source,
        target_k_center=k_target,
        linear_map=linear_map,
        tol=1.0e-8,
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    assert np.allclose(transport.toarray(), expected)
    assert max_delta < 1.0e-8


def test_build_m_c2_eta_transport_does_not_reject_m2_upfront(monkeypatch):
    runner = symmetry_analysis.SymmetryAnalysisRunner.__new__(symmetry_analysis.SymmetryAnalysisRunner)
    runner.structure = SimpleNamespace(
        spin=False,
        reciprocal_Tmat=np.eye(3, dtype=float),
        df=pd.DataFrame(
            {
                "twist_group": [0, 1],
                "atom_type": [0, 1],
                "orb_name": ["s1", "s1"],
            }
        ),
    )

    fake_params = SimpleNamespace(
        m_K1=np.array([1.0, 0.0], dtype=float),
        m_K2=np.array([0.0, 1.0], dtype=float),
    )
    fake_ctx = SimpleNamespace(
        calculator=SimpleNamespace(TAPW_parameters=fake_params),
        group_k_centers={0: np.zeros(2, dtype=float), 1: np.zeros(2, dtype=float)},
        group_m_k_centers={0: np.zeros(2, dtype=float), 1: np.zeros(2, dtype=float)},
        group_g_vectors={0: np.zeros((1, 2), dtype=float), 1: np.zeros((1, 2), dtype=float)},
        moire_reciprocal_basis=np.eye(2, dtype=float),
    )
    runner._valley_context_for_valley = lambda valley: fake_ctx
    monkeypatch.setattr(
        symmetry_analysis,
        "resolve_reference_m_valley_c2_symmetry",
        lambda structure, params: SimpleNamespace(
            rotation_cart=np.diag([-1.0, 1.0, -1.0]),
            linear_map_2d=np.diag([-1.0, 1.0]),
            translation_cart=np.zeros(3, dtype=float),
            rotation_frac=np.eye(3, dtype=float),
            translation_frac=np.zeros(3, dtype=float),
            atom_type_map_0_to_1={0: 1},
            atom_type_map_1_to_0={1: 0},
        ),
    )
    monkeypatch.setattr(
        symmetry_analysis,
        "reference_m_valley_c2_linear_map",
        lambda params: np.diag([-1.0, 1.0]),
    )
    monkeypatch.setattr(
        symmetry_analysis,
        "find_matching_spatial_operation_for_rotation",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        symmetry_analysis,
        "collect_spglib_spatial_operations",
        lambda structure: [],
    )

    candidate = {
        "name": "M_C2_eta",
        "spatial_operations": [],
    }

    transport = runner._build_m_c2_eta_transport(candidate, 32, np.zeros(3, dtype=float), np.zeros(3, dtype=float))

    assert scipy.sparse.issparse(transport)
    assert transport.shape == (2, 2)


def test_build_m_c2_eta_g_transport_matches_named_formula():
    helper = getattr(symmetry_analysis, "build_m_c2_eta_g_transport", None)
    assert helper is not None

    k_source = np.array([0.5, 0.0], dtype=float)
    k_target = np.array([-0.5, 0.0], dtype=float)
    g_source = np.array(
        [
            [0.5, 0.0],
            [1.5, 0.0],
        ],
        dtype=float,
    )
    g_target = np.array(
        [
            [-0.5, 0.0],
            [-1.5, 0.0],
        ],
        dtype=float,
    )
    linear_map = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float)

    transport, max_delta = helper(
        source_g_vectors=g_source,
        target_g_vectors=g_target,
        source_k_center=k_source,
        target_k_center=k_target,
        linear_map=linear_map,
        tol=1.0e-8,
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    assert np.allclose(transport.toarray(), expected)
    assert max_delta < 1.0e-8


def test_legacy_c3_layer_centers_helper_matches_authoritative_k_centers():
    helper = getattr(symmetry_analysis, "legacy_single_valley_c3_layer_centers", None)
    assert helper is not None

    m_g_vec = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    centers = helper(m_g_vec=m_g_vec, twisted_index_m=4, valley=1)

    g_vec_1l, g_vec_2l = symmetry_analysis.get_g_vec_perlayer(m_g_vec, 4)
    expected_k1_1 = symmetry_analysis.rot((g_vec_1l[0] + g_vec_1l[1]) / 3.0, 120)
    expected_k1_2 = symmetry_analysis.rot((g_vec_2l[0] + g_vec_2l[1]) / 3.0, 120)

    assert np.allclose(centers[0], expected_k1_1)
    assert np.allclose(centers[1], expected_k1_2)


def test_k_c2t_center_diagnostics_reports_metrics_per_center_choice():
    helper = getattr(symmetry_analysis, "diagnose_k_c2t_center_choices", None)
    assert helper is not None

    linear_map = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float)
    source_k = np.array([1.0, 0.0], dtype=float)
    target_k = np.array([-1.0, 0.0], dtype=float)
    source_g = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float)
    target_g = np.array([[-1.0, 0.0], [0.0, 0.0]], dtype=float)

    diagnostics = helper(
        source_g_vectors=source_g,
        target_g_vectors=target_g,
        linear_map=linear_map,
        center_choices={
            "A": (source_k, target_k),
            "B": (np.zeros(2, dtype=float), np.zeros(2, dtype=float)),
            "C": (source_k + np.array([0.25, 0.0]), target_k),
        },
        tol=1.0e-8,
    )

    assert set(diagnostics) == {"A", "B", "C"}
    assert diagnostics["A"]["one_to_one"] is True
    assert diagnostics["A"]["max_delta"] < 1.0e-8
    assert diagnostics["B"]["one_to_one"] is False
    assert diagnostics["C"]["one_to_one"] is False


def test_k_c2t_axis_diagnostics_reports_metrics_per_axis():
    helper = getattr(symmetry_analysis, "diagnose_k_c2t_axis_choices", None)
    assert helper is not None

    source_k = np.array([1.0, 0.0], dtype=float)
    target_k = np.array([-1.0, 0.0], dtype=float)
    source_g = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float)
    target_g = np.array([[-1.0, 0.0], [0.0, 0.0]], dtype=float)

    diagnostics = helper(
        source_g_vectors=source_g,
        target_g_vectors=target_g,
        source_k_center=source_k,
        target_k_center=target_k,
        axis_choices={
            "diag(-1,1)": np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float),
            "diag(1,-1)": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        },
        tol=1.0e-8,
    )

    assert set(diagnostics) == {"diag(-1,1)", "diag(1,-1)"}
    assert diagnostics["diag(-1,1)"]["one_to_one"] is True
    assert diagnostics["diag(-1,1)"]["max_delta"] < 1.0e-8
    assert diagnostics["diag(1,-1)"]["one_to_one"] is False


def test_m_time_reversal_g_transport_uses_centered_momentum_mapping():
    helper = getattr(symmetry_analysis, "build_m_time_reversal_g_transport", None)
    assert helper is not None

    center = np.array([1.0, 0.0], dtype=float)
    g_vectors = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    transport, max_delta = helper(
        g_vectors=g_vectors,
        center=center,
        tol=1.0e-8,
    )

    dense = transport.toarray()
    assert max_delta == 0.0
    assert np.allclose(dense[:, 0], [1.0, 0.0, 0.0])
    assert np.allclose(dense[:, 1], [0.0, 0.0, 1.0])
    assert np.allclose(dense[:, 2], [0.0, 1.0, 0.0])


def test_m_time_reversal_transport_uses_centered_mapping_and_target_group_phase():
    runner = symmetry_analysis.SymmetryAnalysisRunner.__new__(symmetry_analysis.SymmetryAnalysisRunner)
    runner.structure = SimpleNamespace(
        spin=False,
        reciprocal_Tmat=np.eye(3, dtype=float),
        df=pd.DataFrame(
            [
                {"twist_group": 0, "atom_type": 0, "orb_name": "s1", "orb_num": 1, "x": 0.25, "y": 0.0},
                {"twist_group": 1, "atom_type": 1, "orb_name": "s1", "orb_num": 1, "x": 0.0, "y": 0.5},
            ]
        ),
    )
    valley_ctx = SimpleNamespace(
        group_k_centers={
            0: np.array([1.0, 0.0], dtype=float),
            1: np.array([0.0, 1.0], dtype=float),
        },
        group_g_vectors={
            0: np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=float),
            1: np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 0.0]], dtype=float),
        },
        moire_reciprocal_basis=np.eye(2, dtype=float),
    )
    runner._valley_context_for_valley = lambda valley: valley_ctx

    transport = symmetry_analysis.SymmetryAnalysisRunner._build_time_reversal_transport(
        runner,
        31,
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
    )

    dense = transport.toarray()
    expected_group0_phase = np.exp(1.0j * np.array([-2.0, 0.0]) @ np.array([0.25, 0.0]))
    expected_group1_phase = np.exp(1.0j * np.array([0.0, -2.0]) @ np.array([0.0, 0.5]))
    expected = scipy.sparse.block_diag(
        [
            expected_group0_phase
            * np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.complex128,
            ),
            expected_group1_phase
            * np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.complex128,
            ),
        ],
        format="csr",
    ).toarray()

    assert np.allclose(dense, expected)
    assert runner._last_transport_diagnostics["g_perm_max_delta"] == 0.0
    assert runner._last_transport_diagnostics["nonzero_reciprocal_shift_count"] == 2


def test_m_unitary_c2_target_group_shifts_use_physical_momentum_offset():
    helper = getattr(symmetry_analysis, "m_unitary_c2_target_group_shifts", None)
    assert helper is not None

    linear_map = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=float)
    reciprocal_basis = np.eye(2, dtype=float)
    group_centers = {
        0: np.array([1.0, 0.0], dtype=float),
        1: np.array([0.0, 0.0], dtype=float),
    }

    shifts, max_residual = helper(
        group_centers=group_centers,
        group_target_map={0: 1, 1: 0},
        linear_map=linear_map,
        reciprocal_basis=reciprocal_basis,
        tol=1.0e-8,
    )

    assert max_residual == 0.0
    assert np.array_equal(shifts[1], np.array([0, -1]))
    assert np.array_equal(shifts[0], np.array([-1, 0]))


def test_unitary_m_c2_spin_helper_uses_pure_spin_rotation_without_time_reversal_piece():
    helper = getattr(symmetry_analysis, "unitary_m_c2_spin_rep", None)
    assert helper is not None

    rotation = np.diag([-1.0, 1.0, -1.0])
    spin_rep = helper(rotation)
    expected = symmetry_analysis.spin_reps(rotation)

    assert np.allclose(spin_rep, expected)
    assert np.allclose(spin_rep @ spin_rep, -np.eye(2, dtype=np.complex128))
