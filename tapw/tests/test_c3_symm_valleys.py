import numpy as np
import pandas as pd
import pytest
from types import MethodType, SimpleNamespace

import tapw.cal_ham_01 as cal_ham_01
from tapw.C3_symm_01 import C3_G_matrix, rot
from tapw.config import ComputeConfig


def _hex_moire_basis():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
        ],
        dtype=float,
    )


def _m_valley_centers(m_g_vec: np.ndarray, twisted_index_m: int, valley: int):
    m_g1 = m_g_vec[0][:2]
    m_g2 = m_g_vec[1][:2]

    if twisted_index_m % 2 == 1:
        offset = (twisted_index_m + 1) * (m_g1 + m_g2) / 2.0
        base_1 = -0.5 * m_g2
        base_2 = -0.5 * m_g1
    else:
        offset = twisted_index_m * (m_g1 + m_g2) / 2.0
        base_1 = 0.5 * m_g1
        base_2 = 0.5 * m_g2

    angle = {31: 0, 32: 120, 33: 240}[valley]
    return rot(base_1 + offset, angle), rot(base_2 + offset, angle)


def _c3_orbit(center: np.ndarray):
    seed = np.array([0.19, -0.11], dtype=float)
    return np.stack([center + rot(seed, angle) for angle in (0, 120, 240)], axis=0)


def test_single_valley_c3_helper_marks_only_c3_closed_valleys_as_compatible():
    helper = getattr(cal_ham_01, "supports_single_valley_c3", None)
    assert helper is not None

    for valley in (1, 2, 11, 12, 5):
        assert helper("hex", valley) is True

    for valley in (31, 32, 33, 3, 41, 42):
        assert helper("hex", valley) is False


def test_tapw_parameters_disable_single_valley_c3_for_m_valleys():
    config = ComputeConfig(C3_H=True, TAPW=True)
    config.valley = 31
    config.bravais = "hex"

    tapw_parameters = cal_ham_01.TAPW_parameters(structure=object(), config=config)

    assert tapw_parameters.use_C3_H is False
    assert "M valleys" in tapw_parameters.c3_h_disable_reason


def test_m_valley_threefold_symmetrization_helper_enables_only_tapw_hex_m_triplet():
    helper = getattr(cal_ham_01, "uses_m_valley_threefold_symmetrization", None)
    assert helper is not None

    config = ComputeConfig(C3_H=True, TAPW=True)
    config.bravais = "hex"

    for valley in (31, 32, 33):
        config.valley = valley
        assert helper(config) is True

    config.valley = 1
    assert helper(config) is False

    config.valley = 31
    config.bravais = "square"
    assert helper(config) is False

    config.bravais = "hex"
    config.TAPW = False
    assert helper(config) is False


def test_m_valley_d3_symmetrization_helper_requires_explicit_flag():
    helper = getattr(cal_ham_01, "uses_m_valley_d3_symmetrization", None)
    assert helper is not None

    config = ComputeConfig(C3_H=True, TAPW=True)
    config.bravais = "hex"
    config.valley = 31

    assert helper(config) is False

    config.M_valley_D3_H = True
    assert helper(config) is True

    config.valley = 1
    assert helper(config) is False

    config.valley = 31
    config.bravais = "square"
    assert helper(config) is False


def test_rotate_local_k_between_m_valleys_is_invertible():
    rotate_k = getattr(cal_ham_01, "rotate_local_k_between_m_valleys", None)
    assert rotate_k is not None

    reciprocal_tmat = np.eye(3)
    k_local = np.array([0.17, -0.23, 0.0], dtype=float)

    k_m2 = rotate_k(k_local, reciprocal_tmat, source_valley=31, target_valley=32)
    k_back = rotate_k(k_m2, reciprocal_tmat, source_valley=32, target_valley=31)

    assert np.allclose(k_back, k_local)


def test_threefold_reference_average_and_derived_valley_transport_are_consistent():
    average_hs = getattr(cal_ham_01, "threefold_reference_hs_average", None)
    derive_matrix = getattr(cal_ham_01, "transport_matrix_between_m_valleys", None)
    assert average_hs is not None
    assert derive_matrix is not None

    h_ref = np.array([[2.0, 1.0 - 0.5j], [1.0 + 0.5j, -1.0]], dtype=np.complex128)
    s_ref = np.array([[3.0, 0.2j], [-0.2j, 2.5]], dtype=np.complex128)

    u_ref_to_m2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    u_ref_to_m3 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    h_m2 = derive_matrix(h_ref, u_ref_to_m2)
    h_m3 = derive_matrix(h_ref, u_ref_to_m3)
    s_m2 = derive_matrix(s_ref, u_ref_to_m2)
    s_m3 = derive_matrix(s_ref, u_ref_to_m3)

    h_avg, s_avg = average_hs(
        h_ref,
        s_ref,
        h_m2,
        s_m2,
        h_m3,
        s_m3,
        u_ref_to_m2.conj().T,
        u_ref_to_m3.conj().T,
    )

    assert np.allclose(h_avg, h_ref)
    assert np.allclose(s_avg, s_ref)
    assert np.allclose(derive_matrix(h_avg, u_ref_to_m2), h_m2)
    assert np.allclose(derive_matrix(s_avg, u_ref_to_m3), s_m3)


def test_m_valley_threefold_hs_ge_false_standardizes_only_after_raw_average():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    raw_hs = {
        "identity": (
            np.array([[2.0, 0.0], [0.0, 8.0]], dtype=np.complex128),
            np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.complex128),
        ),
        "c1": (
            np.array([[18.0, 0.0], [0.0, 32.0]], dtype=np.complex128),
            np.array([[9.0, 0.0], [0.0, 16.0]], dtype=np.complex128),
        ),
        "c2": (
            np.array([[50.0, 0.0], [0.0, 72.0]], dtype=np.complex128),
            np.array([[25.0, 0.0], [0.0, 36.0]], dtype=np.complex128),
        ),
    }

    fake = SimpleNamespace()
    fake.config = SimpleNamespace(valley=31, orthogonal_basis=False, ge=False)
    fake.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    fake._m_valley_reference = 31
    fake._m_valley_c3_reference_projectors = [
        SimpleNamespace(label="identity", linear_map_2d=np.eye(2, dtype=float), projector="identity"),
        SimpleNamespace(label="c3_valley_32", linear_map_2d=np.array([[-0.5, -np.sqrt(3.0) / 2.0], [np.sqrt(3.0) / 2.0, -0.5]], dtype=float), projector="c1"),
        SimpleNamespace(label="c3_valley_33", linear_map_2d=np.array([[-0.5, np.sqrt(3.0) / 2.0], [-np.sqrt(3.0) / 2.0, -0.5]], dtype=float), projector="c2"),
    ]
    fake._m_valley_transport_ref_to_valley = {
        31: np.eye(2, dtype=np.complex128),
        32: np.eye(2, dtype=np.complex128),
        33: np.eye(2, dtype=np.complex128),
    }
    fake.hr_supercell = object()
    fake.sr_supercell = object()

    def _get_raw_projected_hs_with_projector(self, Hr, Sr, k, mpi_index, projector, force_sparse_dot=False):
        return raw_hs[projector]

    def _finalize_tapw_projected_hs(self, hamk, samk, mpi_index):
        return hamk + 10.0 * samk, None

    fake._get_raw_projected_hs_with_projector = MethodType(
        _get_raw_projected_hs_with_projector,
        fake,
    )
    fake._finalize_tapw_projected_hs = MethodType(_finalize_tapw_projected_hs, fake)
    fake._calculate_reference_m_valley_c3_hs = MethodType(
        calculator_cls._calculate_reference_m_valley_c3_hs,
        fake,
    )

    hamk, samk = calculator_cls._calculate_m_valley_threefold_hs(fake, np.zeros(3), mpi_index=0)

    h_avg = (raw_hs["identity"][0] + raw_hs["c1"][0] + raw_hs["c2"][0]) / 3.0
    s_avg = (raw_hs["identity"][1] + raw_hs["c1"][1] + raw_hs["c2"][1]) / 3.0
    expected = h_avg + 10.0 * s_avg

    assert np.allclose(hamk, expected)
    assert samk is None


def test_build_projected_m_valley_transport_matches_rotated_g_permutation():
    builder = getattr(cal_ham_01, "build_projected_m_valley_transport", None)
    assert builder is not None

    rot120 = lambda vec: rot(np.asarray(vec, dtype=float), 120)

    source_k1 = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float)
    source_k2 = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=float)
    target_k1 = np.array([rot120(source_k1[1]), rot120(source_k1[0])], dtype=float)
    target_k2 = np.array([rot120(source_k2[1]), rot120(source_k2[0])], dtype=float)

    structure = SimpleNamespace(
        df=pd.DataFrame(
            {
                "twist_group": [0, 1],
                "atom_type": [0, 1],
                "orb_name": ["A-s1", "B-s1"],
            }
        ),
        spin=False,
    )
    source_params = SimpleNamespace(valley=31, g_vec_list_K1=source_k1, g_vec_list_K2=source_k2)
    target_params = SimpleNamespace(valley=32, g_vec_list_K1=target_k1, g_vec_list_K2=target_k2)

    transport = builder(structure, source_params, target_params).toarray()
    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )

    assert np.allclose(transport, expected)


def test_build_reference_m_valley_c2_transport_swaps_projected_k1_k2_groups():
    builder = getattr(cal_ham_01, "build_reference_m_valley_c2_transport", None)
    assert builder is not None

    structure = SimpleNamespace(
        df=pd.DataFrame(
            {
                "twist_group": [0, 1],
                "atom_type": [0, 1],
                "orb_name": ["A-s1", "B-s1"],
            }
        ),
        spin=False,
    )
    reference_params = SimpleNamespace(
        valley=31,
        g_vec_list_K1=np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float),
        g_vec_list_K2=np.array([[-1.0, 0.0], [-2.0, 0.0]], dtype=float),
        m_K1=np.array([0.1, -0.1], dtype=float),
        m_K2=np.array([-0.1, -0.1], dtype=float),
    )

    transport = builder(structure, reference_params).toarray()
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    assert np.allclose(transport, expected)


def test_build_reference_m_valley_c2_transport_reverses_atom_type_block_order_between_groups():
    builder = getattr(cal_ham_01, "build_reference_m_valley_c2_transport", None)
    assert builder is not None

    structure = SimpleNamespace(
        df=pd.DataFrame(
            {
                "twist_group": [0, 0, 1, 1],
                "atom_type": [0, 1, 2, 3],
                "orb_name": ["A-s1", "B-s1", "B-s1", "A-s1"],
                "z": [0.0, 1.0, 2.0, 3.0],
            }
        ),
        spin=False,
    )
    reference_params = SimpleNamespace(
        valley=31,
        g_vec_list_K1=np.array([[1.0, 0.0]], dtype=float),
        g_vec_list_K2=np.array([[-1.0, 0.0]], dtype=float),
        m_K1=np.array([0.1, -0.1], dtype=float),
        m_K2=np.array([-0.1, -0.1], dtype=float),
    )

    transport = builder(structure, reference_params).toarray()
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    assert np.allclose(transport, expected)


def test_build_reference_m_valley_c2_partner_projector_uses_explicit_transformed_q_on_target_group():
    builder = getattr(cal_ham_01, "build_reference_m_valley_c2_partner_projector", None)
    assert builder is not None

    structure = SimpleNamespace(
        df=pd.DataFrame(
            {
                "twist_group": [0, 1],
                "atom_type": [0, 1],
                "orb_name": ["A-s1", "A-s1"],
                "orb_num": [1, 1],
                "shifted_x": [0.25, 0.75],
                "shifted_y": [0.0, 0.0],
            }
        ),
        spin=False,
    )
    reference_params = SimpleNamespace(
        valley=31,
        g_vec_list_K1=np.array([[1.0, 0.0]], dtype=float),
        g_vec_list_K2=np.array([[-2.0, 0.0]], dtype=float),
    )
    resolved_symmetry = SimpleNamespace(
        rotation_cart=np.diag([-1.0, 1.0, -1.0]),
        translation_cart=np.zeros(3, dtype=float),
        atom_type_map_0_to_1={0: 1},
        atom_type_map_1_to_0={1: 0},
    )

    projector = builder(structure, reference_params, resolved_symmetry).toarray()
    expected = np.array(
        [
            [0.0, np.exp(1.0j * 0.75)],
            [np.exp(-1.0j * 0.5), 0.0],
        ],
        dtype=np.complex128,
    )

    assert np.allclose(projector, expected)


def test_select_reference_m_valley_c2_operation_chooses_candidate_that_fixes_m1():
    selector = getattr(cal_ham_01, "select_reference_m_valley_c2_operation", None)
    assert selector is not None

    lattice = np.array(
        [
            [44.35455773, 27.93612638, 0.0],
            [-46.37067399, 24.44411058, 0.0],
            [0.0, 0.0, 60.0],
        ],
        dtype=float,
    )
    rotations = [
        np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=int),
        np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int),
        np.array([[1, 0, 0], [1, -1, 0], [0, 0, -1]], dtype=int),
    ]
    translations = [
        np.array([0.0, 0.0, 0.3632873], dtype=float),
        np.array([0.0, 0.0, 0.3632873], dtype=float),
        np.array([0.0, 0.0, 0.3632873], dtype=float),
    ]
    m_k1 = np.array([0.036881, -0.058557], dtype=float)

    selected_index, linear_map, rotation_cart, translation_frac = selector(
        lattice=lattice,
        rotations=rotations,
        translations=translations,
        invariant_k_cart=m_k1,
    )

    assert selected_index == 1
    assert np.allclose(linear_map @ m_k1, m_k1, atol=1.0e-5)
    assert np.allclose(rotation_cart[:2, :2], linear_map)
    assert np.allclose(translation_frac, translations[1])


def test_spin_reps_accepts_nearly_exact_order_two_rotation_axis():
    spin_reps = getattr(cal_ham_01, "spin_reps", None)
    assert spin_reps is not None

    rotation = np.array(
        [
            [-0.431953, -0.901896, 0.0],
            [-0.901896, 0.431953, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    spin = spin_reps(rotation)

    assert spin.shape == (2, 2)
    assert np.isfinite(spin).all()


def test_spin_reps_clips_arccos_argument_under_tiny_trace_drift():
    spin_reps = getattr(cal_ham_01, "spin_reps", None)
    assert spin_reps is not None

    rotation = np.eye(3, dtype=float)
    rotation[0, 0] += 2.0e-7
    rotation[1, 1] -= 1.0e-7
    rotation[2, 2] -= 1.0e-7

    spin = spin_reps(rotation)

    assert spin.shape == (2, 2)
    assert np.isfinite(spin).all()


def test_map_atom_types_by_fractional_symmetry_uses_explicit_operation():
    mapper = getattr(cal_ham_01, "map_atom_types_by_fractional_symmetry", None)
    assert mapper is not None

    lattice = np.eye(3, dtype=float)
    rotation_frac = np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int)
    translation_frac = np.array([0.0, 0.0, 0.8], dtype=float)

    source_positions = {
        0: np.array([0.10, 0.20, 0.10], dtype=float),
        1: np.array([0.25, 0.15, 0.20], dtype=float),
        2: np.array([0.35, 0.40, 0.30], dtype=float),
    }
    target_positions = {
        5: (rotation_frac @ source_positions[0]) + translation_frac,
        4: (rotation_frac @ source_positions[1]) + translation_frac,
        3: (rotation_frac @ source_positions[2]) + translation_frac,
    }

    rows = []
    for atom_type, pos in source_positions.items():
        rows.append(
            {
                "twist_group": 0,
                "atom_type": atom_type,
                "species": "A" if atom_type != 1 else "B",
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
            }
        )
    for atom_type, pos in target_positions.items():
        pos_mod = np.mod(pos, 1.0)
        rows.append(
            {
                "twist_group": 1,
                "atom_type": atom_type,
                "species": "A" if atom_type != 4 else "B",
                "x": pos_mod[0],
                "y": pos_mod[1],
                "z": pos_mod[2],
            }
        )

    structure_df = pd.DataFrame(rows)

    mapping = mapper(
        structure_df=structure_df,
        lattice=lattice,
        rotation_frac=rotation_frac,
        translation_frac=translation_frac,
        source_group=0,
        target_group=1,
    )

    assert mapping == {0: 5, 1: 4, 2: 3}


def test_m_valley_d3_hs_uses_cached_reference_projector_average_then_derives_target_valley():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    h_ref = np.array([[1.0, 2.0 + 1.0j], [2.0 - 1.0j, 5.0]], dtype=np.complex128)
    s_ref = np.array([[3.0, 0.4], [0.4, 7.0]], dtype=np.complex128)
    u_ref_to_m2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    fake = SimpleNamespace()
    fake.config = SimpleNamespace(valley=32, orthogonal_basis=True, ge=True)
    fake.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    fake._m_valley_reference = 31
    fake.use_M_valley_d3_symm = True
    fake._m_valley_transport_ref_to_valley = {32: u_ref_to_m2}

    def _finalize_tapw_projected_hs(self, hamk, samk, mpi_index):
        return hamk, samk

    fake._finalize_tapw_projected_hs = MethodType(_finalize_tapw_projected_hs, fake)
    seen_k = []

    def _calculate_reference_m_valley_d3_hs(self, k_reference, mpi_index):
        seen_k.append(np.asarray(k_reference, dtype=float))
        return h_ref, s_ref

    fake._calculate_reference_m_valley_d3_hs = MethodType(_calculate_reference_m_valley_d3_hs, fake)

    k_input = np.array([0.2, 0.1, 0.0], dtype=float)
    hamk, samk = calculator_cls._calculate_m_valley_threefold_hs(fake, k_input, mpi_index=0)

    expected_k_reference = cal_ham_01.rotate_local_k_between_m_valleys(
        k_input,
        fake.structure.reciprocal_Tmat,
        source_valley=32,
        target_valley=31,
    )
    expected_h = u_ref_to_m2 @ h_ref @ u_ref_to_m2.conj().T
    expected_s = u_ref_to_m2 @ s_ref @ u_ref_to_m2.conj().T

    assert np.allclose(hamk, expected_h)
    assert np.allclose(samk, expected_s)
    assert len(seen_k) == 1
    assert np.allclose(seen_k[0], expected_k_reference)


def test_m_valley_c3_hs_skips_target_transport_for_eigenvalue_only_runs():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    h_ref = np.array([[1.0, 2.0 + 1.0j], [2.0 - 1.0j, 5.0]], dtype=np.complex128)
    s_ref = np.array([[3.0, 0.4], [0.4, 7.0]], dtype=np.complex128)
    u_ref_to_m2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    fake = SimpleNamespace()
    fake.config = SimpleNamespace(
        valley=32,
        orthogonal_basis=False,
        ge=False,
        eig_vec_cal=False,
        hamk_save=False,
    )
    fake.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    fake._m_valley_reference = 31
    fake.use_M_valley_d3_symm = False
    fake._m_valley_transport_ref_to_valley = {32: u_ref_to_m2}

    def _finalize_tapw_projected_hs(self, hamk, samk, mpi_index):
        return hamk + 10.0 * samk, None

    fake._finalize_tapw_projected_hs = MethodType(_finalize_tapw_projected_hs, fake)
    seen_k = []

    def _calculate_reference_m_valley_c3_hs(self, k_reference, mpi_index):
        seen_k.append(np.asarray(k_reference, dtype=float))
        return h_ref, s_ref

    fake._calculate_reference_m_valley_c3_hs = MethodType(_calculate_reference_m_valley_c3_hs, fake)

    k_input = np.array([0.2, 0.1, 0.0], dtype=float)
    hamk, samk = calculator_cls._calculate_m_valley_threefold_hs(fake, k_input, mpi_index=0)

    expected_k_reference = cal_ham_01.rotate_local_k_between_m_valleys(
        k_input,
        fake.structure.reciprocal_Tmat,
        source_valley=32,
        target_valley=31,
    )

    assert np.allclose(hamk, h_ref + 10.0 * s_ref)
    assert samk is None
    assert len(seen_k) == 1
    assert np.allclose(seen_k[0], expected_k_reference)


def test_calculate_reference_m_valley_c3_hs_averages_cached_projector_terms():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    h_ref = np.array([[1.0, 0.2], [0.2, 3.0]], dtype=np.complex128)
    s_ref = np.array([[2.0, 0.1], [0.1, 4.0]], dtype=np.complex128)
    h_c1 = np.array([[5.0, -0.4j], [0.4j, 7.0]], dtype=np.complex128)
    s_c1 = np.array([[6.0, -0.3], [-0.3, 8.0]], dtype=np.complex128)
    h_c2 = np.array([[9.0, 0.6j], [-0.6j, 11.0]], dtype=np.complex128)
    s_c2 = np.array([[10.0, 0.2], [0.2, 12.0]], dtype=np.complex128)

    fake = SimpleNamespace()
    fake.hr_supercell = object()
    fake.sr_supercell = object()
    fake.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    fake._m_valley_c3_reference_projectors = [
        SimpleNamespace(label="identity", linear_map_2d=np.eye(2, dtype=float), projector="identity"),
        SimpleNamespace(label="c3_valley_32", linear_map_2d=np.array([[-0.5, -np.sqrt(3.0) / 2.0], [np.sqrt(3.0) / 2.0, -0.5]], dtype=float), projector="c1"),
        SimpleNamespace(label="c3_valley_33", linear_map_2d=np.array([[-0.5, np.sqrt(3.0) / 2.0], [-np.sqrt(3.0) / 2.0, -0.5]], dtype=float), projector="c2"),
    ]

    seen_terms = []

    def _get_raw_projected_hs_with_projector(self, Hr, Sr, k, mpi_index, projector, force_sparse_dot=False):
        seen_terms.append((projector, np.asarray(k, dtype=float), bool(force_sparse_dot)))
        if projector == "identity":
            return h_ref, s_ref
        if projector == "c1":
            return h_c1, s_c1
        if projector == "c2":
            return h_c2, s_c2
        raise AssertionError(f"unexpected projector {projector!r}")

    fake._get_raw_projected_hs_with_projector = MethodType(_get_raw_projected_hs_with_projector, fake)

    k_input = np.array([0.2, 0.1, 0.0], dtype=float)
    hamk, samk = calculator_cls._calculate_reference_m_valley_c3_hs(fake, k_input, mpi_index=0)

    rot120 = np.array([[-0.5, -np.sqrt(3.0) / 2.0], [np.sqrt(3.0) / 2.0, -0.5]], dtype=float)
    rot240 = np.array([[-0.5, np.sqrt(3.0) / 2.0], [-np.sqrt(3.0) / 2.0, -0.5]], dtype=float)

    assert np.allclose(hamk, (h_ref + h_c1 + h_c2) / 3.0)
    assert np.allclose(samk, (s_ref + s_c1 + s_c2) / 3.0)
    assert len(seen_terms) == 3
    assert np.allclose(seen_terms[0][1], k_input)
    assert np.allclose(seen_terms[1][1], np.array([*(rot120 @ k_input[:2]), 0.0], dtype=float))
    assert np.allclose(seen_terms[2][1], np.array([*(rot240 @ k_input[:2]), 0.0], dtype=float))
    assert all(term[2] is True for term in seen_terms)


def test_calculate_reference_m_valley_c3_hs_prefers_valley_projection_then_transport_when_metadata_is_available():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    h_ref = np.array([[1.0, 0.2], [0.2, 3.0]], dtype=np.complex128)
    s_ref = np.array([[2.0, 0.1], [0.1, 4.0]], dtype=np.complex128)
    h_m2 = np.array([[5.0, -0.4j], [0.4j, 7.0]], dtype=np.complex128)
    s_m2 = np.array([[6.0, -0.3], [-0.3, 8.0]], dtype=np.complex128)
    h_m3 = np.array([[9.0, 0.6j], [-0.6j, 11.0]], dtype=np.complex128)
    s_m3 = np.array([[10.0, 0.2], [0.2, 12.0]], dtype=np.complex128)

    u_m2_to_ref = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    u_m3_to_ref = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    fake = SimpleNamespace()
    fake.hr_supercell = object()
    fake.sr_supercell = object()
    fake.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    fake.config = SimpleNamespace(orthogonal_basis=False)
    fake._m_valley_reference = 31
    fake._m_valley_c3_reference_projectors = [
        SimpleNamespace(
            label="identity",
            valley=31,
            linear_map_2d=np.eye(2, dtype=float),
            tapw_parameters="ref_params",
            transport_to_ref=np.eye(2, dtype=np.complex128),
        ),
        SimpleNamespace(
            label="c3_valley_32",
            valley=32,
            linear_map_2d=np.array(
                [
                    [-0.5, -np.sqrt(3.0) / 2.0],
                    [np.sqrt(3.0) / 2.0, -0.5],
                ],
                dtype=float,
            ),
            tapw_parameters="m2_params",
            transport_to_ref=u_m2_to_ref,
        ),
        SimpleNamespace(
            label="c3_valley_33",
            valley=33,
            linear_map_2d=np.array(
                [
                    [-0.5, np.sqrt(3.0) / 2.0],
                    [-np.sqrt(3.0) / 2.0, -0.5],
                ],
                dtype=float,
            ),
            tapw_parameters="m3_params",
            transport_to_ref=u_m3_to_ref,
        ),
    ]

    seen_terms = []

    def _get_raw_tapw_projected_hs_for_parameters(
        self,
        Hr,
        Sr,
        k,
        mpi_index,
        tapw_parameters,
        force_sparse_dot=False,
    ):
        seen_terms.append((tapw_parameters, np.asarray(k, dtype=float), bool(force_sparse_dot)))
        if tapw_parameters == "ref_params":
            return h_ref, s_ref
        if tapw_parameters == "m2_params":
            return h_m2, s_m2
        if tapw_parameters == "m3_params":
            return h_m3, s_m3
        raise AssertionError(f"unexpected tapw_parameters {tapw_parameters!r}")

    def _get_raw_projected_hs_with_projector(self, *args, **kwargs):
        raise AssertionError(
            "projector path should not run when per-valley TAPW metadata is available"
        )

    fake._get_raw_tapw_projected_hs_for_parameters = MethodType(
        _get_raw_tapw_projected_hs_for_parameters,
        fake,
    )
    fake._get_raw_projected_hs_with_projector = MethodType(
        _get_raw_projected_hs_with_projector,
        fake,
    )

    k_input = np.array([0.2, 0.1, 0.0], dtype=float)
    hamk, samk = calculator_cls._calculate_reference_m_valley_c3_hs(fake, k_input, mpi_index=0)

    expected_h, expected_s = cal_ham_01.threefold_reference_hs_average(
        h_ref,
        s_ref,
        h_m2,
        s_m2,
        h_m3,
        s_m3,
        u_m2_to_ref,
        u_m3_to_ref,
    )
    rot120 = np.array(
        [
            [-0.5, -np.sqrt(3.0) / 2.0],
            [np.sqrt(3.0) / 2.0, -0.5],
        ],
        dtype=float,
    )
    rot240 = np.array(
        [
            [-0.5, np.sqrt(3.0) / 2.0],
            [-np.sqrt(3.0) / 2.0, -0.5],
        ],
        dtype=float,
    )

    assert np.allclose(hamk, expected_h)
    assert np.allclose(samk, expected_s)
    assert len(seen_terms) == 3
    assert np.allclose(seen_terms[0][1], k_input)
    assert np.allclose(seen_terms[1][1], np.array([*(rot120 @ k_input[:2]), 0.0], dtype=float))
    assert np.allclose(seen_terms[2][1], np.array([*(rot240 @ k_input[:2]), 0.0], dtype=float))
    assert all(term[2] is True for term in seen_terms)


def test_calculate_reference_m_valley_d3_hs_averages_cached_projector_terms():
    calculator_cls = getattr(cal_ham_01, "BandStructureCalculator", None)
    assert calculator_cls is not None

    h_identity = np.array([[1.0, 0.2], [0.2, 3.0]], dtype=np.complex128)
    s_identity = np.array([[2.0, 0.1], [0.1, 4.0]], dtype=np.complex128)
    h_partner = np.array([[5.0, -0.4j], [0.4j, 7.0]], dtype=np.complex128)
    s_partner = np.array([[6.0, -0.3], [-0.3, 8.0]], dtype=np.complex128)

    fake = SimpleNamespace()
    fake.hr_supercell = object()
    fake.sr_supercell = object()
    fake.structure = SimpleNamespace(reciprocal_Tmat=np.eye(3))
    fake._m_valley_d3_reference_projectors = [
        SimpleNamespace(linear_map_2d=np.eye(2, dtype=float), projector="identity"),
        SimpleNamespace(linear_map_2d=np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float), projector="partner"),
    ]

    seen_kpoints = []

    def _get_raw_projected_hs_with_projector(self, Hr, Sr, k, mpi_index, projector):
        seen_kpoints.append((projector, np.asarray(k, dtype=float)))
        if projector == "identity":
            return h_identity, s_identity
        if projector == "partner":
            return h_partner, s_partner
        raise AssertionError(f"unexpected projector {projector!r}")

    fake._get_raw_projected_hs_with_projector = MethodType(_get_raw_projected_hs_with_projector, fake)

    hamk, samk = calculator_cls._calculate_reference_m_valley_d3_hs(
        fake,
        np.array([0.2, 0.1, 0.0], dtype=float),
        mpi_index=0,
    )

    assert np.allclose(hamk, 0.5 * (h_identity + h_partner))
    assert np.allclose(samk, 0.5 * (s_identity + s_partner))
    assert len(seen_kpoints) == 2
    assert np.allclose(seen_kpoints[0][1], np.array([0.2, 0.1, 0.0], dtype=float))
    assert np.allclose(seen_kpoints[1][1], np.array([-0.2, 0.1, 0.0], dtype=float))


def test_build_reference_m_valley_c2_transport_uses_c2t_spin_unitary_for_spinful_case():
    builder = getattr(cal_ham_01, "build_reference_m_valley_c2_transport", None)
    assert builder is not None

    structure = SimpleNamespace(
        df=pd.DataFrame(
            {
                "twist_group": [0, 1],
                "atom_type": [0, 1],
                "orb_name": ["A-s1", "B-s1"],
            }
        ),
        spin=True,
    )
    reference_params = SimpleNamespace(
        valley=31,
        g_vec_list_K1=np.array([[1.0, 0.0]], dtype=float),
        g_vec_list_K2=np.array([[-1.0, 0.0]], dtype=float),
        m_K1=np.array([0.1, -0.1], dtype=float),
        m_K2=np.array([-0.1, -0.1], dtype=float),
    )

    transport = builder(structure, reference_params).toarray()
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    axis = np.array([0.0, -1.0, 0.0], dtype=float)
    rotation_matrix = cal_ham_01.rotate_mat(axis, np.pi)
    expected_spin = cal_ham_01.spin_reps(rotation_matrix) @ (1.0j * sigma_y)
    expected_orbital = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    expected = np.kron(expected_spin, expected_orbital)

    assert np.allclose(transport, expected)


@pytest.mark.parametrize("valley", [31, 32, 33])
def test_c3_g_matrix_rejects_hex_m_valleys_in_single_valley_basis(valley):
    m_g_vec = _hex_moire_basis()
    twisted_index_m = 1
    center_1layer, center_2layer = _m_valley_centers(m_g_vec, twisted_index_m, valley)

    g_vec_list_1layer = _c3_orbit(center_1layer)
    g_vec_list_2layer = _c3_orbit(center_2layer)

    with pytest.raises(ValueError, match="M valleys.*single-valley C3_H"):
        C3_G_matrix(
            g_vec_list_1layer,
            g_vec_list_2layer,
            m_g_vec,
            twisted_index_m,
            valley=valley,
        )
