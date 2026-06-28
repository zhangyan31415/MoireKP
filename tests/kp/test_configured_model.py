from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kp.model.export as export_module  # noqa: E402
from kp.model.pipeline import (  # noqa: E402
    _auto_harmonics_from_q_sets,
    _auto_harmonics_from_support,
    _all_band_plot_config,
    _apply_refinement_acceptance_guard,
    _auto_low_energy_windows,
    _auto_low_energy_refinement_indices,
    _build_operation_registry,
    _default_term_template_profile_metadata,
    _default_term_templates_for_model,
    _band_refinement_band_residual,
    _band_refinement_eigenvalue_jacobian,
    _band_refinement_global_matrix_jacobian,
    _band_refinement_reduced_global_matrix_loss,
    _default_auto_low_energy_order_config,
    _edge_weighted_band_weights,
    _harmonic_ablation_candidate_is_accepted,
    _harmonic_selection_threshold_values,
    _run_harmonic_ablation_selection,
    _select_harmonic_ablation_candidate,
    _auto_low_energy_fit_candidate_sets,
    _choose_auto_fit_candidate_record,
    _low_subspace_matrix_residual,
    _matrix_loss_block_masks,
    _matrix_loss_residual,
    _max_derivative_order_values,
    _principal_angle_subspace_residual,
    _q_shell_row_indices,
    _representative_score,
    _select_adaptive_fit_indices,
    _select_band_refinement_variables,
    _shell_band_window_from_target_eig,
    _shell_subspace_overlap_report,
    _solve_band_refinement_gauss_newton,
    _subspace_overlap_metrics,
    _symmetry_operation_index,
    _term_component_hamiltonians_for_kpoints,
    _term_response_pair_hamiltonians_for_kpoints,
    _write_auto_model_selection_outputs,
    _window_band_plot_config,
    ConfiguredModel,
    build_moire_config_from_file,
    compare_bands,
    compare_bands_for_plot,
    evaluate_vector_expression,
    load_model_config,
    matrix_residual,
    run_configured_model,
    save_band_comparison_plot,
)
import kp.model.core as model_core  # noqa: E402
import kp.model.pipeline as pipeline_module  # noqa: E402
from kp.model.core import (  # noqa: E402
    ContinuumModel,
    ContinuumModelBuilder,
    ContinuumTerm,
    ContinuumTermKey,
    MoireConfig,
    SymmetryGenerator,
    _prepare_band_state,
    build_model,
    compute_bands,
)
from kp.model.export import _expand_operator_recipe  # noqa: E402
from kp.symmetry.projection import (  # noqa: E402
    _kp_symm_exactification_config,
    _merged_exactification_overrides,
    _operation_power_relation,
)


def _source_meta(*, antiunitary: bool = False, representation: bool = False) -> dict[str, object]:
    return {
        "matrix_kind": "action",
        "source_matrix_role": "bare_D0_internal_rep" if representation else "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "continuum_internal_rep",
        "gauge_correction": {"kind": "none"},
        "antiunitary_convention": "U_K" if antiunitary else "none",
                "spin_map": "from_kp_symm_output",
        "valley_map": "identity",
    }


def _support_grouping_builder() -> ContinuumModelBuilder:
    return ContinuumModelBuilder(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        np.zeros((0, 2), dtype=float),
        1,
        0,
        np.eye(2, dtype=float),
        np.eye(2, dtype=float),
        {},
        {},
        {},
        None,
        {"Kinect": [], "Onsite": [], "intra": [], "inter": []},
    )


def _add_matrix_term(
    builder: ContinuumModelBuilder,
    *,
    mz: int,
    matrix: np.ndarray,
    tag: str = "intra",
) -> ContinuumTermKey:
    key = ContinuumTermKey(mz, 0, 1, 1, 1, 1, (float(mz), 0.0))
    arr = np.asarray(matrix, dtype=np.complex128)
    builder.model.add_term(key, lambda _k, value=arr: value.copy(), tag=tag, symmetry_ops=[])
    return key


def test_expand_operator_recipe_caches_duplicate_monomial_transforms(monkeypatch) -> None:
    def y_basis(_k):
        return np.eye(2, dtype=np.complex128)

    y_basis.eval_sparse = lambda _k: (np.array([0, 1]), np.array([0, 1]), np.ones(2, dtype=complex))
    y_basis._moire_sparse_rows = np.array([0, 1], dtype=int)
    y_basis._moire_sparse_cols = np.array([0, 1], dtype=int)
    y_basis._moire_sparse_row_q_idx = np.array([0, 0], dtype=int)
    y_basis._moire_sparse_Q_rows = np.array([[0.0, 0.0]], dtype=float)
    y_basis._moire_sparse_hermitize_in_basis = False
    term = ContinuumTerm(
        key=ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0)),
        Y_basis=y_basis,
        r_value_real=1.0,
        r_value_imag=0.0,
        tag="Onsite",
        symmetry_ops=[],
    )
    calls = []
    original = export_module._transform_monomial

    def spy(transform, q_base, mz, mz_star):
        calls.append((tuple(np.asarray(q_base, dtype=float)), int(mz), int(mz_star)))
        return original(transform, q_base, mz, mz_star)

    monkeypatch.setattr(export_module, "_transform_monomial", spy)

    data = _expand_operator_recipe([term], SimpleNamespace(symmetry_gen=None), dim=2)

    assert data["row"].size >= 2
    assert len(calls) == 1


def test_support_fit_groups_split_disjoint_actual_support() -> None:
    builder = _support_grouping_builder()
    key_a = _add_matrix_term(builder, mz=0, matrix=np.diag([1.0, 0.0]))
    key_b = _add_matrix_term(builder, mz=1, matrix=np.diag([0.0, 1.0]))

    groups = builder._support_connected_components_for_keys(
        [key_a, key_b],
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
        term_matrix_cache={},
    )

    assert groups == [[key_a], [key_b]]


def test_support_fit_groups_keep_overlapping_actual_support_together() -> None:
    builder = _support_grouping_builder()
    key_a = _add_matrix_term(builder, mz=0, matrix=np.diag([1.0, 0.0]))
    key_b = _add_matrix_term(builder, mz=1, matrix=np.diag([2.0, 0.0]))

    groups = builder._support_connected_components_for_keys(
        [key_a, key_b],
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
        term_matrix_cache={},
    )

    assert groups == [[key_a, key_b]]


def test_support_fit_groups_use_symmetrized_actual_support() -> None:
    builder = _support_grouping_builder()
    key_upper = _add_matrix_term(builder, mz=0, matrix=np.array([[0.0, 1.0], [0.0, 0.0]]))
    key_lower = _add_matrix_term(builder, mz=1, matrix=np.array([[0.0, 0.0], [1.0, 0.0]]))

    groups = builder._support_connected_components_for_keys(
        [key_upper, key_lower],
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
        term_matrix_cache={},
    )

    assert groups == [[key_upper, key_lower]]


def test_support_fit_support_indices_are_empty_for_zero_terms() -> None:
    builder = _support_grouping_builder()
    key = _add_matrix_term(builder, mz=0, matrix=np.zeros((2, 2), dtype=np.complex128))

    support = builder._term_fit_support_indices(
        key,
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
        term_matrix_cache={},
    )

    assert support.size == 0


def test_compute_coefficients_by_tag_splits_disjoint_support_fit_groups(monkeypatch) -> None:
    builder = _support_grouping_builder()
    key_a = _add_matrix_term(builder, mz=0, matrix=np.diag([1.0, 0.0]))
    key_b = _add_matrix_term(builder, mz=1, matrix=np.diag([0.0, 1.0]))
    calls: list[list[ContinuumTermKey]] = []
    original = builder._initialterm_support_vectors_for_fit_keys

    def spy(keys, *args, **kwargs):
        calls.append(list(keys))
        return original(keys, *args, **kwargs)

    monkeypatch.setattr(builder, "_initialterm_support_vectors_for_fit_keys", spy)

    builder.compute_coefficients_by_tag(
        np.diag([3.0, 5.0]).astype(np.complex128),
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
    )

    assert calls == [[key_a], [key_b]]
    assert builder.model.terms[key_a].r_value_real == pytest.approx(3.0)
    assert builder.model.terms[key_b].r_value_real == pytest.approx(5.0)


def test_compute_coefficients_by_tag_keeps_overlapping_support_joint(monkeypatch) -> None:
    builder = _support_grouping_builder()
    key_a = _add_matrix_term(builder, mz=0, matrix=np.diag([1.0, 1.0]))
    key_b = _add_matrix_term(builder, mz=1, matrix=np.diag([2.0, 0.0]))
    calls: list[list[ContinuumTermKey]] = []
    original = builder._initialterm_support_vectors_for_fit_keys

    def spy(keys, *args, **kwargs):
        calls.append(list(keys))
        return original(keys, *args, **kwargs)

    monkeypatch.setattr(builder, "_initialterm_support_vectors_for_fit_keys", spy)

    builder.compute_coefficients_by_tag(
        np.diag([3.0, 0.0]).astype(np.complex128),
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
    )

    assert calls == [[key_a, key_b]]


def test_support_vector_coefficient_solver_recovers_real_coefficients_on_actual_support() -> None:
    builder = _support_grouping_builder()
    initialterms = np.zeros((3, 4, 4), dtype=np.complex128)
    initialterms[0, 0, 0] = 1.0
    initialterms[0, 3, 3] = -0.5
    initialterms[1, 1, 2] = 2.0 + 1.0j
    initialterms[1, 2, 1] = 2.0 - 1.0j
    initialterms[2, 0, 3] = -1.0j
    initialterms[2, 3, 0] = 1.0j
    expected = np.array([1.25, -0.5, 2.0], dtype=float)
    heff = np.tensordot(expected, initialterms, axes=(0, 0))

    coeffs, includinglist, support_idx = builder._solve_coefficients_from_support_vectors(
        initialterms,
        heff,
        tol=1.0e-12,
    )

    assert includinglist.tolist() == [0, 1, 2]
    assert support_idx.size < heff.size
    assert coeffs == pytest.approx(expected)


def test_compute_coefficients_by_tag_uses_support_matrix_solver(monkeypatch) -> None:
    builder = _support_grouping_builder()
    key_a = _add_matrix_term(builder, mz=0, matrix=np.diag([1.0, 0.0]))
    key_b = _add_matrix_term(builder, mz=1, matrix=np.diag([0.0, 1.0]))
    calls: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    original = builder._solve_coefficients_from_support_matrix

    def spy(initial_vectors, target_vector, *args, **kwargs):
        calls.append((tuple(initial_vectors.shape), tuple(target_vector.shape)))
        return original(initial_vectors, target_vector, *args, **kwargs)

    monkeypatch.setattr(builder, "_solve_coefficients_from_support_matrix", spy)

    builder.compute_coefficients_by_tag(
        np.diag([3.0, 5.0]).astype(np.complex128),
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
    )

    assert calls == [((2, 1), (1,)), ((2, 1), (1,))]
    assert builder.model.terms[key_a].r_value_real == pytest.approx(3.0)
    assert builder.model.terms[key_b].r_value_real == pytest.approx(5.0)


def test_compute_coefficients_by_tag_skips_known_zero_support_components(monkeypatch) -> None:
    builder = _support_grouping_builder()
    key_zero = _add_matrix_term(builder, mz=0, matrix=np.zeros((2, 2), dtype=np.complex128))
    key_live = _add_matrix_term(builder, mz=1, matrix=np.diag([0.0, 1.0]))
    materialized: list[list[ContinuumTermKey]] = []
    original = builder._initialterm_support_vectors_for_fit_keys

    def keep_all(self, keys, k_points, *, tol, max_exact_group_size=16, **_kwargs):
        return keys

    def spy(keys, *args, **kwargs):
        materialized.append(list(keys))
        return original(keys, *args, **kwargs)

    monkeypatch.setattr(ContinuumModelBuilder, "_filter_duplicate_symmetry_seed_keys", keep_all)
    monkeypatch.setattr(builder, "_initialterm_support_vectors_for_fit_keys", spy)

    builder.compute_coefficients_by_tag(
        np.diag([0.0, 5.0]).astype(np.complex128),
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
    )

    assert materialized == [[key_live]]
    assert not builder.model.terms[key_zero].active
    assert builder.model.terms[key_zero].r_value_real == 0.0
    assert builder.model.terms[key_zero].r_value_imag == 0.0
    assert builder.model.terms[key_live].r_value_real == pytest.approx(5.0)


def test_compute_coefficients_by_tag_uses_support_vectors_without_dense_initialterms(monkeypatch) -> None:
    builder = _support_grouping_builder()
    key = _add_matrix_term(builder, mz=0, matrix=np.diag([1.0, 2.0]))

    def fail_dense_materialization(*_args, **_kwargs):
        raise AssertionError("dense initialterms should not be materialized in support-vector fit")

    monkeypatch.setattr(builder, "_initialterms_for_fit_keys", fail_dense_materialization)

    builder.compute_coefficients_by_tag(
        np.diag([3.0, 6.0]).astype(np.complex128),
        np.array([[0.0, 0.0]], dtype=float),
        tol=1.0e-10,
    )

    assert builder.model.terms[key].active
    assert builder.model.terms[key].r_value_real == pytest.approx(3.0)


def test_fit_block_materialization_matches_stacked_dense_blocks() -> None:
    builder = _support_grouping_builder()
    key = _add_matrix_term(
        builder,
        mz=0,
        matrix=np.array([[1.0, 2.0j], [-2.0j, 3.0]], dtype=np.complex128),
    )
    k_points = np.array([[0.0, 0.0], [0.25, 0.0]], dtype=float)

    mat_real, mat_imag = builder.stack_Y_for_term(builder.model.terms[key], k_points)
    expected_real, expected_imag = builder.get_mat_blocks([mat_real, mat_imag], key, len(k_points))
    got_real, got_imag = builder._fit_blocks_for_term_key(key, k_points, term_matrix_cache={})

    np.testing.assert_allclose(got_real, expected_real)
    np.testing.assert_allclose(got_imag, expected_imag)


def test_fit_block_cache_reuses_readonly_matrix_objects() -> None:
    builder = _support_grouping_builder()
    key = _add_matrix_term(builder, mz=0, matrix=np.diag([1.0, 2.0]))
    k_points = np.array([[0.0, 0.0], [0.25, 0.0]], dtype=float)
    cache = {}

    first_real, first_imag = builder._fit_blocks_for_term_key(key, k_points, term_matrix_cache=cache)
    assert len(cache) == 1
    cached_real, cached_imag = next(iter(cache.values()))
    assert first_real is cached_real
    assert first_imag is cached_imag

    second_real, second_imag = builder._fit_blocks_for_term_key(key, k_points, term_matrix_cache=cache)
    assert second_real is cached_real
    assert second_imag is cached_imag


def test_fit_block_support_vectors_match_dense_initialterms_on_support() -> None:
    builder = _support_grouping_builder()
    key = _add_matrix_term(
        builder,
        mz=0,
        matrix=np.array([[1.0, 2.0j], [-2.0j, 3.0]], dtype=np.complex128),
    )
    k_points = np.array([[0.0, 0.0], [0.25, 0.0]], dtype=float)
    cache = {}
    support = builder._term_fit_support_indices(key, k_points, tol=1.0e-10, term_matrix_cache=cache)

    dense = builder._initialterms_for_fit_keys([key], k_points, term_matrix_cache=cache)
    vectors = builder._initialterm_support_vectors_for_fit_keys(
        [key],
        k_points,
        support_idx=support,
        term_matrix_cache=cache,
    )

    np.testing.assert_allclose(vectors, dense.reshape(dense.shape[0], -1)[:, support])


def _triangular_q_shell() -> np.ndarray:
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    return np.array(
        [
            [0.0, 0.0],
            b1,
            -b1,
            b2,
            -b2,
            b1 - b2,
            -b1 + b2,
        ],
        dtype=float,
    )


def test_compute_bands_uses_eigvals_only_without_saved_eigenvectors(monkeypatch):
    def fail_eigh(*_args, **_kwargs):
        raise AssertionError("compute_bands should not compute eigenvectors when save_eigvecs=false")

    monkeypatch.setattr(model_core.scipy.linalg, "eigh", fail_eigh)

    cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
        save_eigvecs=False,
    )
    key = ContinuumTermKey(Mz=0, Mz_star=0, layer_from=1, layer_to=1, orbital_from=1, orbital_to=1, p=(0.0, 0.0))
    term = ContinuumTerm(
        key=key,
        Y_basis=lambda _k: np.diag([1.0, 2.0]).astype(complex),
        r_value_real=1.0,
        r_value_imag=0.0,
        active=True,
        tag="Kinect",
    )
    model = ContinuumModel()
    model.terms[key] = term

    eigvals = compute_bands(cfg, model, cfg.kpoints)

    assert eigvals.shape == (1, 2)
    np.testing.assert_allclose(eigvals[0], [1.0, 2.0])


def test_compute_bands_hermitizes_assembled_hamiltonian(monkeypatch):
    def add_nonhermitian(matrix, *_args, **_kwargs):
        matrix[0, 1] += 1.0

    monkeypatch.setattr(
        model_core.ContinuumModelBuilder,
        "add_symmetrized_term_to_matrix_static",
        staticmethod(add_nonhermitian),
    )
    cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
        save_eigvecs=False,
    )
    key = ContinuumTermKey(Mz=0, Mz_star=0, layer_from=1, layer_to=1, orbital_from=1, orbital_to=1, p=(0.0, 0.0))
    term = ContinuumTerm(
        key=key,
        Y_basis=lambda _k: np.zeros((2, 2), dtype=complex),
        r_value_real=1.0,
        r_value_imag=0.0,
        active=True,
        tag="Kinect",
    )
    model = ContinuumModel()
    model.terms[key] = term

    eigvals = compute_bands(cfg, model, cfg.kpoints)

    np.testing.assert_allclose(eigvals[0], [-0.5, 0.5])


def test_monomial_extraction_tolerates_roundoff_leakage():
    matrix = np.eye(4, dtype=complex)
    matrix[0, 1] = 3.0e-9

    mono = ContinuumModelBuilder._extract_monomial_matrix(matrix)

    assert mono is not None
    perm, vals = mono
    np.testing.assert_array_equal(perm, np.arange(4))
    np.testing.assert_allclose(vals, np.ones(4))


def test_monomial_extraction_rejects_dense_leakage():
    matrix = np.eye(4, dtype=complex)
    matrix[0, 1] = 3.0e-5

    assert ContinuumModelBuilder._extract_monomial_matrix(matrix) is None


def test_clear_symmetry_caches_clears_orbit_and_kz_caches():
    model_core.clear_symmetry_caches()
    ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE["orbit"] = object()
    ContinuumModelBuilder._KZ_POW_CACHE["kz"] = np.ones((1, 1), dtype=complex)

    model_core.clear_symmetry_caches()

    assert not ContinuumModelBuilder._SYMMETRIZE_ORBIT_CACHE
    assert not ContinuumModelBuilder._KZ_POW_CACHE


def test_term_template_harmonic_filter_is_honored_by_core_builder():
    qset = _triangular_q_shell()
    intra = {
        1: np.array([0.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.5, np.sqrt(3.0) / 2.0]),
    }
    builder = ContinuumModelBuilder(
        qset,
        qset,
        1,
        1,
        np.array([1.0, 0.0]),
        np.array([0.5, np.sqrt(3.0) / 2.0]),
        intra,
        {},
        {"intra": 0},
        SymmetryGenerator(qset, qset, [1, 1]),
        {"intra": []},
        [
            {
                "name": "filtered_intra",
                "source": "moire_potential",
                "tag": "intra",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "harmonic_filter": {"kind": "intra", "indices": [2], "sign": -1.0},
                "max_order": 0,
            }
        ],
    )

    builder.build_terms()

    assert list(builder.model.terms) == [
        ContinuumTermKey(Mz=0, Mz_star=0, layer_from=1, layer_to=1, orbital_from=1, orbital_to=1, p=(-1.0, -0.0))
    ]
    term = next(iter(builder.model.terms.values()))
    assert term.registry_metadata["harmonic_id"] == 2
    assert term.registry_metadata["harmonics_source"] == "explicit_indices"


def test_term_template_rejects_harmonics_and_harmonic_filter_together():
    qset = _triangular_q_shell()
    builder = ContinuumModelBuilder(
        qset,
        qset,
        1,
        1,
        np.array([1.0, 0.0]),
        np.array([0.5, np.sqrt(3.0) / 2.0]),
        {1: np.array([0.0, 0.0])},
        {},
        {"intra": 0},
        SymmetryGenerator(qset, qset, [1, 1]),
        {"intra": []},
        [
            {
                "name": "ambiguous_intra",
                "source": "moire_potential",
                "tag": "intra",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "harmonics": "intra",
                "harmonic_filter": {"kind": "intra", "indices": [1]},
                "max_order": 0,
            }
        ],
    )

    with pytest.raises(ValueError, match="use either harmonics or harmonic_filter"):
        builder.build_terms()


def _write_auto_fixture(tmp_path: Path) -> Path:
    qset = _triangular_q_shell()
    kpoints = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float)
    dim = 2 * len(qset)
    heff = np.stack([np.diag(np.arange(dim, dtype=float)), np.diag(np.arange(dim, dtype=float) + 0.1)]).astype(np.complex128)

    np.save(tmp_path / "q1.npy", -qset)
    np.save(tmp_path / "q2.npy", -qset)
    np.save(tmp_path / "kpoints.npy", kpoints)
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    np.save(out_dir / "heff_list.npy", heff)
    np.save(out_dir / "heff_eig.npy", np.linalg.eigvalsh(heff))

    source_cfg = {
        "material": {
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "plot": {},
        "project": {
            "out_dir": "project",
        },
    }
    (tmp_path / "source.yaml").write_text(yaml.safe_dump(source_cfg), encoding="utf-8")
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)

    model_cfg = {
        "source_config": "source.yaml",
        "symmetry_source": {"type": "kp_symm_output", "path": "symm"},
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": "Gamma",
            "mode": "single_valley",
            "active_valleys": ["Gamma"],
            "spin_convention": "spinless_effective",
            "allowed_internal_symmetries": [],
            "external_sewing_symmetries": [],
        },
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": [1, 1],
            "nlow_state": [1, 1],
            "bM": {"source": "q_distance"},
            "harmonics": {
                "intra": 4,
                "inter": {"count": 4},
            },
            "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
            "symmetry_map": {
                "Kinect": [],
                "intra": [],
                "inter": [],
            },
        },
        "fit": {
            "indices": [0],
            "coeff_tol": 1.0e-8,
        },
        "bands": {
            "indices": [0, 1],
            "compare_to_heff": True,
        },
        "output": {
            "dir": "model_out",
        },
    }
    path = tmp_path / "model_auto.yaml"
    path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    return path


def _write_symm_frame_manifest(tmp_path: Path, *, rotation_deg: float, path_name: str = "symm") -> Path:
    symm_dir = tmp_path / path_name
    symm_dir.mkdir(exist_ok=True)
    (symm_dir / "manifest.json").write_text(
        json.dumps({"frame": {"q_transform": {"rotation_deg": float(rotation_deg)}}, "operations": []}),
        encoding="utf-8",
    )
    return symm_dir


def _write_unified_case_fixture(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "kp" / "configs"
    cfg_dir.mkdir(parents=True)
    qset = _triangular_q_shell()
    np.save(cfg_dir / "q1.npy", -qset)
    np.save(cfg_dir / "q2.npy", -qset)
    np.save(cfg_dir / "kpoints.npy", np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=float))

    case = "tiny_K1_B_Q4"
    project_out = tmp_path / "kp" / "outputs" / "project" / case
    project_out.mkdir(parents=True)
    dim = len(qset)
    heff = np.stack([np.diag(np.arange(dim, dtype=float) + shift) for shift in (0.0, 0.1, 0.2)]).astype(np.complex128)
    np.save(project_out / "heff_list.npy", heff)
    np.save(project_out / "heff_eig.npy", np.linalg.eigvalsh(heff))

    symm_out = tmp_path / "kp" / "outputs" / "symm" / case
    symm_out.mkdir(parents=True)
    (symm_out / "manifest.json").write_text(
        json.dumps(
            {
                "frame": {"q_transform": {"rotation_deg": 0.0}},
                "operations": [
                    {
                        "name": "C3z",
                        "operation": "C3z",
                        "matrix_file": "exactified_C3z.npy",
                        "matrix_kind": "continuum_internal_rep_exact",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "case": case,
        "valley": "K1",
        "spin": "down",
        "material": {
            "name": "Tiny",
            "num_layer_list": [1, 2],
            "orbital_order": "x",
            "efermi": 0.0,
            "hamk_file": "hamk.npy",
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "project": {
            "nlow_state_list": [[], [], [0]],
            "gauge": "auto",
            "downfold_method": "fixed_schur",
            "e_ref": 0.0,
        },
        "symm": {},
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": [0, 0, 1],
            "harmonics": {"intralayer": 1},
            "max_order": {"kinetic": 2, "intralayer": 0},
            "fit": {"mode": "auto_compact", "max_points": 2, "coeff_tol": 1.0e-8},
            "bands": {"compare_to_heff": True, "band_slice": [0, dim]},
        },
    }
    path = cfg_dir / f"{case}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_cli_model_subcommand_invokes_configured_runner(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"
    seen: dict[str, object] = {}

    class FakeModelConfig:
        output_dir = model_output

    def fake_run(path: str) -> dict:
        seen["path"] = path
        return {"configured_model": FakeModelConfig(), "comparison": {"rms_error": 0.0, "max_abs_error": 0.0}}

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        seen["export"] = (Path(model_output_dir), Path(output_dir), bool(force), bool(debug_files))
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path)])

    assert seen["path"] == str(cfg_path)
    assert seen["export"] == (model_output, model_output / "standalone", True, False)


def test_cli_model_subcommand_prints_band_plot_path(monkeypatch, tmp_path: Path, capsys) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    plot_path = tmp_path / "model_out" / "band_comparison.pdf"
    all_plot_path = tmp_path / "model_out" / "band_comparison_all.pdf"

    class FakeModelConfig:
        output_dir = tmp_path / "model_out"

    def fake_run(path: str) -> dict:
        return {
            "configured_model": FakeModelConfig(),
            "band_plot": str(plot_path.resolve()),
            "all_band_plot": str(all_plot_path.resolve()),
            "comparison": {"rms_error": 0.0, "max_abs_error": 0.0},
            "all_band_plot_comparison": {
                "rms_error": 0.002,
                "max_abs_error": 0.003,
                "rms_error_mev": 2.0,
                "max_abs_error_mev": 3.0,
                "num_bands": 124,
                "align": "top",
            },
        }

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert f"[kp model]   band plot: {plot_path.resolve()}" in out
    assert f"[kp model]   all-band plot: {all_plot_path.resolve()}" in out
    assert "[kp model]   all-band RMS: 2.000 meV, Max: 3.000 meV (bands=124, align=top)" in out


def test_cli_model_subcommand_uses_explicit_standalone_export_path(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"
    export_output = tmp_path / "portable"
    calls: dict[str, object] = {}

    class FakeModelConfig:
        output_dir = model_output

    def fake_run(path: str) -> dict:
        calls["run"] = path
        return {"configured_model": FakeModelConfig(), "comparison": None}

    def fake_export(model_output_dir, output_dir, *, force=False, debug_files=False):
        calls["export"] = (Path(model_output_dir), Path(output_dir), bool(force), bool(debug_files))
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path), "--export-standalone", str(export_output)])

    assert calls["run"] == str(cfg_path)
    assert calls["export"] == (model_output, export_output, True, False)


def test_standalone_export_dense_symmetry_action_expands_sparse_entries() -> None:
    from kp.model.export import _dense_composed_symmetry_action, _iter_transformed_sparse_entries

    class DenseGenerator:
        def __init__(self) -> None:
            self.matrix = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)

        def get_operator(self, name: str, params=None):
            assert name == "C3z"
            assert params in {1, -1}
            return self.matrix

    action = _dense_composed_symmetry_action(DenseGenerator(), (("C3z", 1),))
    entries = list(
        _iter_transformed_sparse_entries(
            action,
            np.array([0], dtype=int),
            np.array([1], dtype=int),
        )
    )

    observed = {(row, col): value for _pos, row, col, value, is_anti in entries if not is_anti}
    assert set(observed) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert observed[(0, 0)] == pytest.approx(0.5 + 0.0j)
    assert observed[(0, 1)] == pytest.approx(-0.5 + 0.0j)
    assert observed[(1, 0)] == pytest.approx(0.5 + 0.0j)
    assert observed[(1, 1)] == pytest.approx(-0.5 + 0.0j)


def test_cli_project_rejects_unimplemented_qdpt2() -> None:
    import kp.cli as cli

    parser = cli.build_argparser()

    with pytest.raises(SystemExit):
        parser.parse_args(["project", "-c", "model.yaml", "--downfold-method", "qdpt2"])

    args = parser.parse_args(["project", "-c", "model.yaml", "--downfold-method", "fixed_schur"])
    assert args.downfold_method == "fixed_schur"


def test_cli_energy_unit_policy_is_explicit() -> None:
    import kp.cli as cli

    assert cli._energy_scale_from_material({}) == pytest.approx(1.0)
    assert cli._energy_scale_from_material({"energy_unit": "eV"}) == pytest.approx(1.0)
    assert cli._energy_scale_from_material({"energy_unit": "Hartree"}) == pytest.approx(cli.HARTREE_TO_EV)

    with pytest.raises(ValueError, match="material.energy_unit"):
        cli._energy_scale_from_material({"energy_unit": "Rydberg"})


def test_project_orbital_layout_infers_missing_num_orb_per_layer() -> None:
    import kp.cli as cli

    q_count = 19
    total_layers = 3
    orbitals_per_layer = 71
    dim = 2 * q_count * total_layers * orbitals_per_layer
    hamk2d = np.zeros((dim, dim), dtype=np.complex128)

    for spin in ("all", "up", "down"):
        num_layer_list, orb0, layout = cli._orbital_layout_from_material(
            {"num_layer_list": [1, 2]},
            hamk2d,
            q_count,
            spin=spin,
            mode="K1",
        )

        assert num_layer_list == [1, 2]
        assert orb0 == orbitals_per_layer
        assert layout == [[orbitals_per_layer], [orbitals_per_layer, orbitals_per_layer]]


def test_project_auto_gauge_validation_allows_omitted_symmetry_operations() -> None:
    import kp.cli as cli

    assert cli._symm_can_validate_auto_gauge({"tapw_symmetry_dir": "symmetry_analysis"})
    assert not cli._symm_can_validate_auto_gauge({"tapw_symmetry_dir": "symmetry_analysis", "enable": False})


def test_public_release_helpers_have_basic_stable_behavior(tmp_path: Path) -> None:
    from kp.analysis import compute_orbital_weights, rank_orbitals, select_orbit_set
    from kp.io import load_orbital_order
    from kp.kmesh import KPathGenerator
    from kp.symmetry import SymmetryGenerator as PublicSymmetryGenerator
    from kp.symmetry import check_symmetry_consistency
    from kp.viz import plot_band_structure

    orbital_csv = tmp_path / "orbitals.csv"
    orbital_csv.write_text("index,name,layer\n0,dz2,1\n1,dxy,2\n", encoding="utf-8")
    assert load_orbital_order(str(orbital_csv)) == [
        {"index": 0, "name": "dz2", "layer": 1},
        {"index": 1, "name": "dxy", "layer": 2},
    ]

    real_lattice = np.eye(2)
    np.testing.assert_allclose(KPathGenerator.calculate_reciprocal_vectors(real_lattice), 2.0 * np.pi * np.eye(2))
    np.testing.assert_allclose(KPathGenerator.direct_cart_real(real_lattice, [0.25, 0.5]), [0.25, 0.5])

    eigvecs = np.zeros((2, 4, 2), dtype=complex)
    eigvecs[:, 0, 0] = 1.0
    eigvecs[:, 3, 1] = 2.0
    weights = compute_orbital_weights(eigvecs, np.zeros((2, 2)), ["a", "b"], [0, 1], score="sum")
    np.testing.assert_allclose(weights, [2.0, 8.0])
    assert rank_orbitals(weights, top_k=1) == [1]
    assert select_orbit_set(weights, {"top_k": 1}) == {"indices": [1], "weights": [8.0]}

    def h_of_k(k: np.ndarray) -> np.ndarray:
        return np.diag([float(k[0]), float(k[1])])

    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    residual = check_symmetry_consistency(
        h_of_k,
        swap,
        lambda k: np.array([k[1], k[0]], dtype=float),
        k_points=np.array([[0.2, 0.7]], dtype=float),
    )
    assert residual == pytest.approx(0.0)
    k_identity, d_identity = PublicSymmetryGenerator(np.zeros((1, 2)), np.zeros((1, 2)), [1, 1]).get_operator("identity")
    np.testing.assert_allclose(k_identity, np.eye(2))
    np.testing.assert_allclose(d_identity, np.eye(2))
    with pytest.raises(ValueError, match="explicit symmetry matrices"):
        PublicSymmetryGenerator(np.zeros((1, 2)), np.zeros((1, 2)), [1, 1]).get_operator("C3z")

    plot_path = plot_band_structure([0.0, 1.0], np.array([[0.0, 0.5], [0.2, 0.7]]), path=tmp_path / "bands.png")
    assert Path(plot_path).exists()


def _write_fixture(tmp_path: Path) -> Path:
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    kpoints = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=float)
    heff = np.stack(
        [
            np.diag([0.0, 1.0, 2.0, 3.0]),
            np.diag([0.1, 1.1, 2.1, 3.1]),
            np.diag([0.2, 1.2, 2.2, 3.2]),
        ]
    ).astype(np.complex128)

    np.save(tmp_path / "q1.npy", q1)
    np.save(tmp_path / "q2.npy", q2)
    np.save(tmp_path / "kpoints.npy", kpoints)
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    np.save(out_dir / "heff_list.npy", heff)
    np.save(out_dir / "heff_eig.npy", np.linalg.eigvalsh(heff))

    source_cfg = {
        "material": {
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "plot": {},
        "project": {
            "out_dir": "project",
        },
    }
    (tmp_path / "source.yaml").write_text(yaml.safe_dump(source_cfg), encoding="utf-8")
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)

    model_cfg = {
        "source_config": "source.yaml",
        "symmetry_source": {"type": "kp_symm_output", "path": "symm"},
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": "Gamma",
            "mode": "single_valley",
            "active_valleys": ["Gamma"],
            "spin_convention": "spinless_effective",
            "allowed_internal_symmetries": [],
            "external_sewing_symmetries": [],
        },
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": [1, 1],
            "nlow_state": [1, 1],
            "bM": {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]},
            "harmonics": {
                "intra": {1: "zero", 2: "-bM1", 3: "bM1 + bM2"},
                "inter": {1: "zero", 2: "2*bM1"},
            },
            "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
            "symmetry_map": {
                "Kinect": [],
                "intra": [],
                "inter": [],
            },
        },
        "fit": {
            "indices": [0, 2],
            "coeff_tol": 1.0e-8,
            "coeff_prune_threshold": 2.0e-4,
        },
        "bands": {
            "indices": [0, 1, 2],
            "compare_to_heff": True,
        },
        "output": {
            "dir": "model_out",
        },
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    return path


def test_load_model_config_resolves_paths_relative_to_yaml(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)

    cfg = load_model_config(cfg_path)

    assert cfg.path == cfg_path
    assert cfg.source_config == tmp_path / "source.yaml"
    assert cfg.kpoints_file == tmp_path / "kpoints.npy"
    assert cfg.heff_file == tmp_path / "project" / "heff_list.npy"
    assert cfg.output_dir == tmp_path / "model_out"
    assert cfg.fit_indices == [0, 2]
    assert cfg.coeff_prune_threshold == 2.0e-4
    assert cfg.band_indices == [0, 1, 2]


def test_load_model_config_accepts_single_case_yaml_with_nested_model_sections(tmp_path: Path) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)

    cfg = load_model_config(cfg_path)

    case = "tiny_K1_B_Q4"
    assert cfg.source_config == cfg_path
    assert cfg.source_raw["case"] == case
    assert cfg.heff_file == tmp_path / "kp" / "outputs" / "project" / case / "heff_list.npy"
    assert cfg.output_dir == tmp_path / "kp" / "outputs" / "model" / case
    assert cfg.valley_model["valley_type"] == "K"
    assert cfg.valley_model["active_valleys"] == ["K1"]
    assert cfg.valley_model["spin_convention"] == "spin_down_projected"
    assert cfg.symmetry_source_config["type"] == "kp_symm_output"
    assert cfg.symmetry_source_config["path"] == str((tmp_path / "kp" / "outputs" / "symm" / case).resolve())
    assert "operations" not in cfg.symmetry_source_config
    assert {tag: [row["name"] for row in rows] for tag, rows in cfg.symmetry_map.items()} == {
        "Kinect": ["C3z"],
        "Onsite": ["C3z"],
        "intra": ["C3z"],
        "inter": ["C3z"],
    }
    assert cfg.fit_selection_metadata["mode"] == "auto_compact"
    assert len(cfg.fit_indices) == 2
    assert cfg.band_slice == [0, 7]
    assert cfg.compare_to_heff is True
    assert cfg.harmonics_config == {"intra": 1}
    assert cfg.max_order["Kinect"] == 2
    assert cfg.orbital_count_metadata["n_orb"]["input_kind"] == "physical_layer"
    assert cfg.sectors_config == [{"name": "L3", "qset": "qset2", "n_orb": 1}]
    assert cfg.term_template_metadata["profiles"] == ["k_sector_aware"]
    assert [row["name"] for row in cfg.term_templates] == ["kinetic_L3", "onsite_L3"]


def test_load_model_config_accepts_band_refinement_block(tmp_path: Path) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["fit"]["refine_bands"] = {
        "enabled": True,
        "variable_tags": "kinetic",
        "components": "real",
        "top_bands": 1,
        "matrix_weight": 0.003,
        "reweight": {
            "rounds": 1,
            "threshold_mev": 0.8,
            "alpha": 2.0,
            "cap": 5.0,
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.band_refinement_config == {
        "enabled": True,
        "variable_tags": "kinetic",
        "components": "real",
        "top_bands": 1,
        "matrix_weight": 0.003,
        "reweight": {
            "rounds": 1,
            "threshold_mev": 0.8,
            "alpha": 2.0,
            "cap": 5.0,
        },
    }


def test_load_model_config_accepts_block_matrix_loss(tmp_path: Path) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["fit"]["refine_bands"] = {
        "enabled": True,
        "variable_tags": ["kinetic", "intralayer"],
        "components": ["real", "imag"],
        "top_bands": 1,
        "matrix_loss": {
            "enabled": True,
            "mode": "block_normalized",
            "sigma_mev": 10.0,
            "weight": 0.2,
            "blocks": {
                "kinetic_diagonal": 2.0,
                "intralayer": 1.0,
                "interlayer": 0.5,
            },
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.band_refinement_config["matrix_loss"] == {
        "enabled": True,
        "mode": "block_normalized",
        "sigma_mev": 10.0,
        "weight": 0.2,
        "blocks": {
            "kinetic_diagonal": 2.0,
            "intralayer": 1.0,
            "interlayer": 0.5,
        },
    }


def test_block_normalized_matrix_loss_balances_matrix_blocks() -> None:
    moire_cfg = MoireConfig(
        Q_set1=np.zeros((2, 2), dtype=float),
        Q_set2=np.zeros((1, 2), dtype=float),
        n_orb1=1,
        n_orb2=1,
    )
    masks = _matrix_loss_block_masks(moire_cfg, dim=3)
    assert int(np.count_nonzero(masks["kinetic_diagonal"])) == 3
    assert int(np.count_nonzero(masks["intralayer"])) == 2
    assert int(np.count_nonzero(masks["interlayer"])) == 4

    delta = np.zeros((1, 3, 3), dtype=np.complex128)
    delta[:, masks["kinetic_diagonal"]] = 2.0
    delta[:, masks["intralayer"]] = 4.0
    delta[:, masks["interlayer"]] = 6.0

    residual, report = _matrix_loss_residual(
        delta,
        masks,
        {
            "enabled": True,
            "mode": "block_normalized",
            "sigma_mev": 1000.0,
            "weight": 1.0,
            "blocks": {
                "kinetic_diagonal": 1.0,
                "intralayer": 1.0,
                "interlayer": 1.0,
            },
        },
    )

    assert residual.shape == (18,)
    assert report["mode"] == "block_normalized"
    assert report["blocks"]["kinetic_diagonal"]["count"] == 3
    assert report["blocks"]["intralayer"]["count"] == 2
    assert report["blocks"]["interlayer"]["count"] == 4
    assert report["blocks"]["kinetic_diagonal"]["rms_mev"] == pytest.approx(2000.0)
    assert report["blocks"]["intralayer"]["rms_mev"] == pytest.approx(4000.0)
    assert report["blocks"]["interlayer"]["rms_mev"] == pytest.approx(6000.0)
    assert report["blocks"]["kinetic_diagonal"]["loss_norm_sq"] == pytest.approx(4.0)
    assert report["blocks"]["intralayer"]["loss_norm_sq"] == pytest.approx(16.0)
    assert report["blocks"]["interlayer"]["loss_norm_sq"] == pytest.approx(36.0)


def test_global_matrix_loss_matches_legacy_dense_residual() -> None:
    delta = np.array(
        [
            [
                [1.0 + 2.0j, 3.0 - 4.0j],
                [5.0 + 6.0j, 7.0 - 8.0j],
            ]
        ],
        dtype=np.complex128,
    )

    residual, report = _matrix_loss_residual(
        delta,
        {"all": np.ones((2, 2), dtype=bool)},
        {"enabled": True, "mode": "global", "sigma_mev": 2000.0, "weight": 0.5},
    )

    legacy = np.sqrt(0.5) * (delta.reshape(-1) / 2.0)
    expected = np.concatenate([legacy.real, legacy.imag])
    np.testing.assert_allclose(residual, expected)
    assert report["mode"] == "global"
    assert report["rms_mev"] == pytest.approx(float(np.sqrt(np.mean(np.abs(delta) ** 2)) * 1000.0))


def test_band_loss_normalization_scales_selected_window_by_rms_count() -> None:
    model = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float)
    target = np.zeros_like(model)

    unnormalized = _band_refinement_band_residual(model, target, band_sigma=2.0, normalize=False)
    normalized = _band_refinement_band_residual(model, target, band_sigma=2.0, normalize=True)

    np.testing.assert_allclose(unnormalized, model / 2.0)
    np.testing.assert_allclose(normalized, model / (2.0 * np.sqrt(model.size)))
    assert np.sum(normalized**2) == pytest.approx(np.mean((model / 2.0) ** 2))


def test_band_refinement_eigenvalue_jacobian_matches_finite_difference() -> None:
    basis = np.array(
        [
            [
                [[1.0, 0.2], [0.2, -0.3]],
                [[0.4, -0.1j], [0.1j, 0.7]],
            ],
            [
                [[0.1, 0.3j], [-0.3j, 0.2]],
                [[-0.2, 0.5], [0.5, 0.9]],
            ],
        ],
        dtype=np.complex128,
    )
    y = np.array([0.35, -0.2], dtype=float)
    h = np.tensordot(y, basis, axes=(0, 0))
    eig, vec = np.linalg.eigh(h)
    target = eig[:, 0:2] + np.array([[0.01, -0.02], [0.02, -0.01]])

    def residual(values: np.ndarray) -> np.ndarray:
        current = np.tensordot(values, basis, axes=(0, 0))
        current_eig = np.linalg.eigvalsh(current)
        selected = current_eig[:, 0:2]
        selected = selected + (target[0, -1] - selected[0, -1])
        return _band_refinement_band_residual(selected, target, band_sigma=0.5, normalize=True).ravel()

    analytic = _band_refinement_eigenvalue_jacobian(
        vec,
        basis,
        band_slice=(0, 2),
        align="top",
        band_sigma=0.5,
        normalize=True,
    )
    numeric = np.empty_like(analytic)
    step = 1.0e-6
    for index in range(y.size):
        delta = np.zeros_like(y)
        delta[index] = step
        numeric[:, index] = (residual(y + delta) - residual(y - delta)) / (2.0 * step)

    assert analytic == pytest.approx(numeric, abs=1.0e-6)


def test_band_refinement_global_matrix_jacobian_matches_residual_layout() -> None:
    basis = np.array(
        [
            [
                [[1.0, 2.0j], [-2.0j, 3.0]],
                [[0.5, 0.25], [0.25, -0.5]],
            ],
            [
                [[0.2, -0.4], [-0.4, 0.1]],
                [[0.0, 1.0j], [-1.0j, 0.3]],
            ],
        ],
        dtype=np.complex128,
    )
    y = np.array([0.4, -0.7], dtype=float)
    matrix_sigma = 0.01
    matrix_weight = 0.2

    def residual(values: np.ndarray) -> np.ndarray:
        delta = np.tensordot(values, basis, axes=(0, 0)).reshape(-1) / matrix_sigma
        return np.sqrt(matrix_weight) * np.concatenate([delta.real, delta.imag])

    analytic = _band_refinement_global_matrix_jacobian(
        basis,
        matrix_weight=matrix_weight,
        matrix_sigma=matrix_sigma,
    )
    numeric = np.empty_like(analytic)
    step = 1.0e-6
    for index in range(y.size):
        delta = np.zeros_like(y)
        delta[index] = step
        numeric[:, index] = (residual(y + delta) - residual(y - delta)) / (2.0 * step)

    assert analytic == pytest.approx(numeric, abs=1.0e-8)


def test_band_refinement_reduced_matrix_loss_preserves_quadratic_gradient() -> None:
    basis = np.array(
        [
            [
                [[1.0, 0.2j], [-0.2j, 0.4]],
                [[0.3, 0.1], [0.1, -0.7]],
            ],
            [
                [[0.2, -0.5], [-0.5, 0.6]],
                [[-0.1, 0.4j], [-0.4j, 0.8]],
            ],
            [
                [[0.0, 0.3], [0.3, -0.2]],
                [[0.5, -0.2j], [0.2j, 0.1]],
            ],
        ],
        dtype=np.complex128,
    )
    target_delta = np.array(
        [
            [[0.3, 0.1j], [-0.1j, -0.2]],
            [[-0.4, 0.2], [0.2, 0.5]],
        ],
        dtype=np.complex128,
    )
    matrix_weight = 0.4
    matrix_sigma = 0.02
    reduced = _band_refinement_reduced_global_matrix_loss(
        basis,
        target_delta,
        matrix_weight=matrix_weight,
        matrix_sigma=matrix_sigma,
    )
    jac = reduced["jacobian"]
    center = reduced["center_delta"]
    scale = np.sqrt(matrix_weight) / matrix_sigma

    def original_objective(delta: np.ndarray) -> float:
        matrix_delta = target_delta + np.tensordot(delta, basis, axes=(0, 0))
        flat = matrix_delta.reshape(-1)
        residual = scale * np.concatenate([flat.real, flat.imag])
        return float(residual @ residual)

    def reduced_objective(delta: np.ndarray) -> float:
        residual = jac @ (delta - center)
        return float(residual @ residual)

    first = np.array([0.1, -0.2, 0.05], dtype=float)
    second = np.array([-0.3, 0.4, 0.2], dtype=float)
    assert original_objective(first) - original_objective(second) == pytest.approx(
        reduced_objective(first) - reduced_objective(second),
        abs=1.0e-8,
    )

    flat = basis.reshape(basis.shape[0], -1)
    target = target_delta.reshape(-1)
    gram = (matrix_weight / matrix_sigma**2) * np.real(flat.conj() @ flat.T)
    linear = (matrix_weight / matrix_sigma**2) * np.real(flat.conj() @ target)
    np.testing.assert_allclose(
        jac.T @ (jac @ (first - center)),
        gram @ first + linear,
        atol=1.0e-8,
    )


def test_term_response_pair_matches_component_responses() -> None:
    builder = _support_grouping_builder()
    key = _add_matrix_term(
        builder,
        mz=0,
        matrix=np.array([[1.0, 0.4j], [-0.4j, -0.2]], dtype=np.complex128),
        tag="Kinect",
    )
    term = builder.model.terms[key]
    term.active = True
    moire_cfg = MoireConfig(
        Q_set1=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=1,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0], [0.2, 0.1]], dtype=float),
    )
    state = _prepare_band_state(moire_cfg, builder.model)
    real_component = _term_component_hamiltonians_for_kpoints(
        moire_cfg,
        state,
        term,
        "real",
        moire_cfg.kpoints,
    )
    imag_component = _term_component_hamiltonians_for_kpoints(
        moire_cfg,
        state,
        term,
        "imag",
        moire_cfg.kpoints,
    )

    real_pair, imag_pair = _term_response_pair_hamiltonians_for_kpoints(state, term, moire_cfg.kpoints)

    np.testing.assert_allclose(real_pair, real_component, atol=1.0e-12)
    np.testing.assert_allclose(imag_pair, imag_component, atol=1.0e-12)


def test_term_response_pair_matches_sparse_component_responses() -> None:
    qset = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    key = ContinuumTermKey(1, 0, 1, 1, 1, 1, (0.0, 0.0))
    term = ContinuumTerm(
        key=key,
        Y_basis=ContinuumModelBuilder.make_Y_basis_function(
            key,
            qset,
            np.zeros((0, 2), dtype=float),
            1,
            0,
        ),
        r_value_real=0.0,
        r_value_imag=0.0,
        active=True,
        tag="Kinect",
        symmetry_ops=[],
    )
    model = SimpleNamespace(terms={key: term})
    moire_cfg = MoireConfig(
        Q_set1=qset,
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=1,
        n_orb2=0,
        kpoints=np.array([[0.1, 0.0], [0.2, 0.3]], dtype=float),
    )
    state = _prepare_band_state(moire_cfg, model)
    real_component = _term_component_hamiltonians_for_kpoints(moire_cfg, state, term, "real", moire_cfg.kpoints)
    imag_component = _term_component_hamiltonians_for_kpoints(moire_cfg, state, term, "imag", moire_cfg.kpoints)

    real_pair, imag_pair = _term_response_pair_hamiltonians_for_kpoints(state, term, moire_cfg.kpoints)

    np.testing.assert_allclose(real_pair, real_component, atol=1.0e-12)
    np.testing.assert_allclose(imag_pair, imag_component, atol=1.0e-12)


def test_band_refinement_gauss_newton_solver_fits_nonlinear_residual() -> None:
    def residual(values: np.ndarray) -> np.ndarray:
        x, y = values
        return np.array([x * x + y - 1.0, x - y], dtype=float)

    def jacobian(values: np.ndarray) -> np.ndarray:
        x, _y = values
        return np.array([[2.0 * x, 1.0], [1.0, -1.0]], dtype=float)

    result = _solve_band_refinement_gauss_newton(
        residual,
        jacobian,
        np.array([0.2, 0.8], dtype=float),
        max_nfev=40,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )

    assert result.cost < 1.0e-20
    np.testing.assert_allclose(result.x, [0.6180339887, 0.6180339887], atol=1.0e-10)
    assert result.njev > 0


def test_edge_weighted_band_weights_prioritize_top_bands() -> None:
    weights = _edge_weighted_band_weights(
        n_bands=6,
        primary_bands=2,
        target_bands="top",
        decay=0.5,
        floor=0.1,
        normalize_mean=True,
    )

    assert weights.shape == (6,)
    assert weights[-1] == pytest.approx(weights[-2])
    assert weights[-1] > weights[-3] > weights[-4] > weights[0]
    assert np.mean(weights) == pytest.approx(1.0)


def test_edge_weighted_band_weights_prioritize_bottom_bands() -> None:
    weights = _edge_weighted_band_weights(
        n_bands=6,
        primary_bands=2,
        target_bands="bottom",
        decay=0.5,
        floor=0.1,
        normalize_mean=True,
    )

    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] > weights[2] > weights[3] > weights[-1]
    assert np.mean(weights) == pytest.approx(1.0)


def test_principal_angle_subspace_residual_is_phase_and_rotation_invariant() -> None:
    target = np.eye(4, 2, dtype=np.complex128)[None, :, :]
    rotation = np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128)
    model = (target[0] @ rotation)[None, :, :]

    residual, report = _principal_angle_subspace_residual(
        model,
        target,
        weight=1.0,
        normalize=True,
        gap_weights=np.ones(1),
    )

    np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)
    assert report["mean_overlap"] == pytest.approx(1.0)
    assert report["max_leakage"] == pytest.approx(0.0)


def test_principal_angle_subspace_residual_detects_orthogonal_direction() -> None:
    target = np.eye(4, 2, dtype=np.complex128)[None, :, :]
    model = np.zeros((1, 4, 2), dtype=np.complex128)
    model[0, 0, 0] = 1.0
    model[0, 2, 1] = 1.0

    residual, report = _principal_angle_subspace_residual(
        model,
        target,
        weight=1.0,
        normalize=False,
        gap_weights=np.ones(1),
    )

    np.testing.assert_allclose(np.sort(residual), [0.0, 1.0], atol=1.0e-12)
    assert report["mean_overlap"] == pytest.approx(0.5)
    assert report["max_leakage"] == pytest.approx(0.5)
    assert report["min_singular_value"] == pytest.approx(0.0)


def test_subspace_overlap_metrics_reports_boundary_instability() -> None:
    target = np.eye(4, 2, dtype=np.complex128)[None, :, :]
    model = np.zeros((1, 4, 2), dtype=np.complex128)
    model[0, 0, 0] = 1.0
    model[0, 2, 1] = 1.0

    report = _subspace_overlap_metrics(model, target, gap_weights=np.array([0.25]))

    assert report["mean_overlap"] == pytest.approx(0.5)
    assert report["max_leakage"] == pytest.approx(0.5)
    assert report["min_singular_value"] == pytest.approx(0.0)
    assert report["gap_weight_min"] == pytest.approx(0.25)


def test_low_subspace_matrix_residual_uses_target_projector_not_full_matrix() -> None:
    target = np.eye(3, 1, dtype=np.complex128)[None, :, :]
    delta = np.zeros((1, 3, 3), dtype=np.complex128)
    delta[0, 0, 0] = 2.0
    delta[0, 2, 2] = 100.0

    residual, report = _low_subspace_matrix_residual(
        delta,
        target,
        weight=1.0,
        sigma_mev=1000.0,
        normalize=False,
        gap_weights=np.ones(1),
    )

    np.testing.assert_allclose(residual, [2.0, 0.0], atol=1.0e-12)
    assert report["rms_mev"] == pytest.approx(2000.0)
    assert report["max_mev"] == pytest.approx(2000.0)


def test_q_shell_row_indices_are_cumulative_and_orbital_complete() -> None:
    qset1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    qset2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)

    shells = _q_shell_row_indices(qset1, qset2, (2, 1), max_shells=2, tol=1.0e-8)

    assert [item["shell_index"] for item in shells] == [0, 1]
    assert shells[0]["dimension"] == 3
    assert shells[0]["subspace_bands"] == 1
    assert shells[0]["rows"] == [0, 3, 6]
    assert shells[1]["dimension"] == 6
    assert shells[1]["subspace_bands"] == 3
    assert shells[1]["rows"] == [0, 1, 3, 4, 6, 7]


def test_shell_subspace_overlap_uses_half_shell_dimension_and_ignores_internal_rotation() -> None:
    qset = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    shells = _q_shell_row_indices(qset, np.zeros((0, 2), dtype=float), (2, 0), max_shells=2)
    target = np.diag([0.0, 0.1, 1.0, 1.1])[None, :, :].astype(np.complex128)
    rotation = np.array(
        [
            [np.sqrt(0.5), -np.sqrt(0.5)],
            [np.sqrt(0.5), np.sqrt(0.5)],
        ],
        dtype=np.complex128,
    )
    rotated_top = rotation @ np.diag([1.0, 1.1]) @ rotation.conj().T
    model = target.copy()
    model[0, 2:4, 2:4] = rotated_top

    report = _shell_subspace_overlap_report(
        model,
        target,
        shells,
        target_bands="top",
    )

    shell1 = report["shells"][1]
    assert shell1["dimension"] == 4
    assert shell1["subspace_bands"] == 2
    assert shell1["subspace"]["mean_overlap"] == pytest.approx(1.0)
    assert shell1["subspace"]["max_leakage"] == pytest.approx(0.0)


def test_shell_gap_aware_window_prefers_stable_boundary_near_half() -> None:
    eig = np.array(
        [
            [0.0, 0.1, 0.2, 1.8, 1.9, 2.0],
            [0.0, 0.2, 0.3, 1.7, 1.8, 1.9],
        ],
        dtype=float,
    )

    window = _shell_band_window_from_target_eig(
        eig,
        target_bands="top",
        window_config={
            "mode": "gap_aware",
            "center_fraction": 0.5,
            "search_fraction": [0.35, 0.65],
            "gap_tolerance_mev": 50.0,
        },
    )

    assert window["mode"] == "gap_aware"
    assert window["n_bands"] == 3
    assert window["band_slice"] == [3, 6]
    assert window["boundary_gap_mev"] == pytest.approx(1400.0)
    assert window["window_quality"] == "ok"


def test_shell_gap_aware_window_reports_small_boundary_gap() -> None:
    eig = np.array([[0.0, 0.1, 0.11, 0.12, 0.13, 0.14]], dtype=float)

    window = _shell_band_window_from_target_eig(
        eig,
        target_bands="top",
        window_config={
            "mode": "gap_aware",
            "center_fraction": 0.5,
            "search_fraction": [0.35, 0.65],
            "gap_tolerance_mev": 50.0,
        },
    )

    assert window["n_bands"] == 3
    assert window["boundary_gap_mev"] < 50.0
    assert window["window_quality"] == "boundary_gap_small"


def test_harmonic_ablation_selects_smallest_acceptable_support() -> None:
    qset = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    heff = np.zeros((1, 6, 6), dtype=np.complex128)
    heff[0] = np.diag([0.0, 0.10, 2.0, 3.0, 4.0, 5.0])
    heff[0, 0, 1] = heff[0, 1, 0] = 0.10
    heff[0, 0, 2] = heff[0, 2, 0] = 1.0e-5

    report = _run_harmonic_ablation_selection(
        heff,
        qset,
        qset,
        n_orb=(1, 1),
        target_bands="bottom",
        primary_bands=2,
        plot_bands=2,
        max_shell=2,
        thresholds={
            "plot_rms_mev": 1.0,
            "plot_max_mev": 2.0,
            "min_overlap": 0.99,
        },
    )

    selected = report["selected"]
    assert selected["intra_shells"] == 1
    assert selected["inter_shells"] == 0
    assert selected["accepted"] is True
    assert selected["plot_rms_mev"] < 1.0
    assert selected["subspace_mean_overlap"] > 0.99
    too_small = next(
        item for item in report["candidates"]
        if item["intra_shells"] == 0 and item["inter_shells"] == 0
    )
    assert too_small["accepted"] is False
    assert too_small["plot_rms_mev"] > 10.0


def test_harmonic_selection_default_thresholds_are_conservative() -> None:
    thresholds = _harmonic_selection_threshold_values({})

    marginal_compact = {
        "plot_rms_mev": 0.44,
        "plot_max_mev": 1.28,
        "subspace_mean_overlap": 0.9986,
        "subspace_max_leakage": 0.002,
        "low_matrix_rms_mev": 0.15,
    }
    accurate_candidate = {
        "plot_rms_mev": 0.08,
        "plot_max_mev": 0.22,
        "subspace_mean_overlap": 0.9999,
        "subspace_max_leakage": 1.0e-4,
        "low_matrix_rms_mev": 0.02,
    }

    assert thresholds["plot_rms_mev"] < 1.0
    assert thresholds["plot_max_mev"] < 3.0
    assert not _harmonic_ablation_candidate_is_accepted(marginal_compact, thresholds)
    assert _harmonic_ablation_candidate_is_accepted(accurate_candidate, thresholds)


def test_harmonic_ablation_selection_uses_quality_plateau_not_first_compact_candidate() -> None:
    candidates = [
        {
            "intra_shells": 3,
            "inter_shells": 3,
            "accepted": True,
            "plot_rms_mev": 0.18,
            "plot_max_mev": 0.55,
            "low_matrix_rms_mev": 0.09,
        },
        {
            "intra_shells": 4,
            "inter_shells": 4,
            "accepted": True,
            "plot_rms_mev": 0.045,
            "plot_max_mev": 0.18,
            "low_matrix_rms_mev": 0.035,
        },
        {
            "intra_shells": 5,
            "inter_shells": 5,
            "accepted": True,
            "plot_rms_mev": 0.020,
            "plot_max_mev": 0.10,
            "low_matrix_rms_mev": 0.010,
        },
    ]

    selected, status = _select_harmonic_ablation_candidate(candidates)

    assert status == "accepted_quality_plateau_candidate"
    assert (selected["intra_shells"], selected["inter_shells"]) == (4, 4)


def test_default_harmonic_candidate_ladder_includes_balanced_four_shell() -> None:
    from kp.model.pipeline import _default_harmonic_candidate_pairs

    pairs = _default_harmonic_candidate_pairs(5)

    assert (4, 4) in pairs


def test_harmonic_ablation_can_scan_explicit_candidate_pairs_only() -> None:
    qset = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    heff = np.zeros((1, 6, 6), dtype=np.complex128)
    heff[0] = np.diag([0.0, 0.10, 2.0, 3.0, 4.0, 5.0])
    heff[0, 0, 1] = heff[0, 1, 0] = 0.10

    report = _run_harmonic_ablation_selection(
        heff,
        qset,
        qset,
        n_orb=(1, 1),
        target_bands="bottom",
        primary_bands=2,
        plot_bands=2,
        max_shell=5,
        candidate_pairs=[(0, 0), (1, 0)],
        thresholds={"plot_rms_mev": 1.0, "plot_max_mev": 2.0, "min_overlap": 0.99},
    )

    assert [(row["intra_shells"], row["inter_shells"]) for row in report["candidates"]] == [(0, 0), (1, 0)]
    assert report["selected"]["intra_shells"] == 1
    assert report["selected"]["inter_shells"] == 0


def test_auto_low_energy_windows_reports_small_boundary_gap() -> None:
    eig = np.array(
        [
            [0.0, 1.0, 1.00001, 2.0, 4.0],
            [0.0, 1.1, 1.10001, 2.1, 4.1],
        ],
        dtype=float,
    )

    windows = _auto_low_energy_windows(
        eig,
        n_primary=2,
        target_bands="top",
        gap_tolerance_mev=0.1,
        max_expanded=2,
    )

    assert windows["primary"]["band_slice"] == [3, 5]
    assert windows["primary"]["n_bands"] == 2
    assert windows["primary"]["boundary_gap_mev"] == pytest.approx(999.99)
    assert windows["expanded"][0]["n_bands"] == 3
    assert windows["expanded"][0]["state"] == "boundary_gap_small"
    assert windows["expanded"][0]["use_for_loss"] is False


def test_select_adaptive_fit_indices_keeps_endpoints_and_residual_peaks() -> None:
    seg1 = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21)])
    seg2 = np.column_stack([np.ones(20), np.linspace(0.05, 1.0, 20)])
    seg3 = np.column_stack([np.linspace(0.95, 0.0, 20), np.ones(20)])
    kpoints = np.vstack([seg1, seg2, seg3])
    scores = np.zeros(len(kpoints))
    scores[13] = 10.0
    scores[47] = 10.0

    selected, meta = _select_adaptive_fit_indices(
        kpoints,
        initial_points=2,
        max_points=7,
        residual_scores=scores,
    )

    assert selected == [0, 40]
    assert meta["mode"] == "auto_low_energy"
    assert meta["source"] == "candidate_scan_pending"
    candidates = {row["name"]: row["indices"] for row in meta["fit_candidate_sets"]}
    assert candidates["minimal"] == [0, 40]
    assert candidates["junctions_3"] == [0, 20, 40]
    assert candidates["junctions_4"] == [0, 20, 40, 60]
    assert candidates["residual_augmented_5"] == [0, 20, 40, 60, 13]
    assert candidates["residual_augmented_6"] == [0, 20, 40, 60, 13, 47]


def test_load_model_config_auto_low_energy_sets_default_refinement(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"]["target_bands"] = "top"
    data["fit"] = {
        "mode": "auto_low_energy",
        "coeff_tol": 1.0e-8,
    }
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.fit_selection_metadata["mode"] == "auto_low_energy"
    assert config.fit_selection_metadata["max_points"] == 2
    assert config.band_refinement_config["enabled"] is True
    assert config.band_refinement_config["solver"] == "linear_low_subspace"
    assert config.band_refinement_config["variable_tags"] == ["Kinect", "Onsite", "intra"]
    assert config.band_refinement_config["components"] == ["real"]
    assert [row["name"] for row in config.band_refinement_config["refinement_candidates"]] == [
        "A_intra_real",
        "B_add_inter_real",
        "C_add_imag",
    ]
    assert config.band_refinement_config["subspace_loss"]["enabled"] is False
    shell_loss = config.band_refinement_config["shell_projected_matrix_loss"]
    assert shell_loss["enabled"] is True
    assert shell_loss["weight"] == pytest.approx(1.0)
    assert shell_loss["max_shells"] == 3
    assert shell_loss["window"]["mode"] == "fixed_fraction"
    assert shell_loss["window"]["center_fraction"] == pytest.approx(0.5)
    assert shell_loss["legacy_alias"] == "shell_subspace_loss"
    assert "shell_subspace_loss" not in config.band_refinement_config
    assert config.band_refinement_config["low_subspace_matrix_loss"]["enabled"] is False
    assert config.band_refinement_config["target_bands"] == "top"
    assert config.band_refinement_config["band_slice"][1] - config.band_refinement_config["band_slice"][0] > sum(config.n_orb)
    assert config.band_refinement_config["weighted_band_loss"]["enabled"] is True
    assert config.band_refinement_config["weighted_band_loss"]["primary_bands"] == sum(config.n_orb)
    assert config.band_refinement_config["weighted_band_loss"]["decay"] == pytest.approx(0.75)
    assert config.band_refinement_config["weighted_band_loss"]["floor"] == pytest.approx(0.45)
    assert config.band_refinement_config["subspace_loss"]["band_slice"] == config.band_refinement_config["band_slice"]
    assert config.band_refinement_config["low_subspace_matrix_loss"]["band_slice"] == config.band_refinement_config["band_slice"]
    assert config.band_refinement_config["subspace_loss"]["weight"] == pytest.approx(2.0)
    assert config.band_refinement_config["max_nfev"] == 8
    assert config.band_refinement_config["indices"] == config.fit_selection_metadata["refine_indices"]
    assert config.band_refinement_config["indices"] == [0, 1]
    assert config.band_refinement_config["use_fit_kpoints"] is False
    assert config.band_refinement_config["max_variables"] == 900
    assert config.band_refinement_config["acceptance_guard"]["enabled"] is True
    assert config.band_refinement_config["acceptance_guard"]["profile"] == "low_energy"
    assert config.band_refinement_config["acceptance_guard"]["selection"] == "best_validation_window"
    assert config.band_refinement_config["acceptance_guard"]["line_search_alphas"] == [0.5]
    assert config.band_refinement_config["acceptance_guard"]["guard_all_bands"] is False
    assert config.band_refinement_config["acceptance_guard"]["max_rms_increase_mev"] == pytest.approx(999.0)
    assert config.band_refinement_config["acceptance_guard"]["max_max_increase_mev"] == pytest.approx(999.0)
    assert config.coeff_prune_threshold == pytest.approx(1.0e-4)


def test_load_model_config_auto_low_energy_fills_harmonics_and_orders(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"].pop("harmonics", None)
    data["model"].pop("max_order", None)
    data["model"].pop("max_derivative_order", None)
    data["model"]["target_bands"] = "top"
    data["fit"] = {"mode": "auto_low_energy", "max_points": 2}
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.harmonics_config["intra"]["count"] == 2
    assert config.harmonics_config["inter"]["count"] == 2
    assert config.max_order["Kinect"] == 6
    assert config.max_order["intra"] == 4
    assert config.max_order["inter"] == 6
    assert config.max_order["moire_intra_zero"] == 0
    assert config.max_order["tunneling_zero"] == 6
    assert config.raw["model"]["auto_low_energy_order_profile"]["profile"] == "gamma_compact_ladder_start"
    assert config.raw["model"]["auto_low_energy_defaults"]["harmonics"] is True


def test_auto_low_energy_order_defaults_are_valley_aware() -> None:
    gamma = _default_auto_low_energy_order_config({"valley_type": "Gamma"})
    k = _default_auto_low_energy_order_config({"valley_type": "K"})
    m = _default_auto_low_energy_order_config({"valley_type": "M"})

    assert gamma["max_order"] == {"Kinect": 6, "intra": 4, "inter": 6}
    assert gamma["max_derivative_order"]["moire_intra_zero"] == 0
    assert gamma["max_derivative_order"]["moire_intra_nonzero"] == 4
    assert gamma["max_derivative_order"]["tunneling_zero"] == 6
    assert k["max_order"] == {"Kinect": 6, "intra": 4, "inter": 4}
    assert k["max_derivative_order"] == {}
    assert m["max_order"] == {"Kinect": 10, "intra": 4, "inter": 6}


def test_load_model_config_auto_low_energy_can_select_harmonics_from_heff(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"].pop("harmonics", None)
    data["model"]["target_bands"] = "bottom"
    data["fit"] = {
        "mode": "auto_low_energy",
        "max_points": 2,
        "harmonic_selection": {
            "enabled": True,
            "max_shell": 2,
            "sample_kpoints": 0,
            "plot_bands": 2,
            "thresholds": {
                "plot_rms_mev": 1.0,
                "plot_max_mev": 2.0,
                "min_overlap": 0.99,
            },
        },
    }
    qset = np.load(tmp_path / "q1.npy")
    heff = np.zeros((2, 2 * len(qset), 2 * len(qset)), dtype=np.complex128)
    for ik in range(2):
        heff[ik] = np.diag(np.arange(2 * len(qset), dtype=float) + 10.0)
        heff[ik, 0, 0] = 0.0
        heff[ik, 1, 1] = 0.10
        heff[ik, 0, 1] = heff[ik, 1, 0] = 0.10
        heff[ik, 0, 2] = heff[ik, 2, 0] = 1.0e-5
    np.save(tmp_path / "project" / "heff_list.npy", heff)
    np.save(tmp_path / "project" / "heff_eig.npy", np.linalg.eigvalsh(heff))
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.harmonics_config["intra"]["count"] == 1
    assert config.harmonics_config["inter"]["count"] == 0
    report = config.raw["model"]["auto_low_energy_harmonic_selection"]
    assert report["selection_status"] == "accepted_quality_plateau_candidate"
    assert report["selected"]["accepted"] is True
    assert config.band_refinement_config["auto_harmonic_selection"]["selected"]["intra_shells"] == 1


def test_load_model_config_auto_low_energy_preserves_explicit_harmonics(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"]["harmonics"] = {"intralayer": 3, "interlayer": 2}
    data["model"]["target_bands"] = "bottom"
    data["fit"] = {
        "mode": "auto_low_energy",
        "max_points": 2,
        "harmonic_selection": {"enabled": True, "max_shell": 1},
    }
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.harmonics_config == {"intra": 3, "inter": 2}
    assert "auto_low_energy_harmonic_selection" not in config.raw["model"]
    assert config.band_refinement_config["auto_harmonic_selection"]["enabled"] is False


def test_band_refinement_invokes_subspace_and_low_matrix_losses(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    class Term:
        tag = "Kinect"
        active = True
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))

        def __init__(self) -> None:
            self.r_value_real = 0.0
            self.r_value_imag = 0.0

    term = Term()

    class Model:
        terms = {"term": term}

    def fake_hamiltonians(_moire_config, model, kpoints):
        value = next(iter(model.terms.values())).r_value_real
        return np.array([[[0.0, value], [value, 1.0]] for _ in range(len(kpoints))], dtype=np.complex128)

    calls = {"subspace": 0, "matrix": 0}

    def fake_subspace(model_basis, target_basis, **kwargs):
        calls["subspace"] += 1
        assert model_basis.shape == target_basis.shape == (1, 2, 1)
        return np.zeros(1, dtype=float), {"enabled": True, "mean_overlap": 1.0, "max_leakage": 0.0}

    def fake_low_matrix(delta, target_basis, **kwargs):
        calls["matrix"] += 1
        assert delta.shape == (1, 2, 2)
        assert target_basis.shape == (1, 2, 1)
        return np.zeros(1, dtype=float), {"enabled": True, "rms_mev": 0.0, "max_mev": 0.0}

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)
    monkeypatch.setattr(pipeline, "_principal_angle_subspace_residual", fake_subspace)
    monkeypatch.setattr(pipeline, "_low_subspace_matrix_residual", fake_low_matrix)

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_refinement_config={
            "enabled": True,
            "band_slice": [1, 2],
            "align": "top",
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "max_nfev": 1,
            "subspace_loss": {"enabled": True, "top_bands": 1, "weight": 1.0},
            "low_subspace_matrix_loss": {"enabled": True, "top_bands": 1, "weight": 1.0, "sigma_mev": 10.0},
        },
    )

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, Model())

    assert report["subspace_loss"]["enabled"] is True
    assert report["low_subspace_matrix_loss"]["enabled"] is True
    assert calls["subspace"] > 0
    assert calls["matrix"] > 0


def test_band_refinement_passes_analytic_jacobian_to_least_squares(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    class Term:
        tag = "Kinect"
        active = True
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))

        def __init__(self) -> None:
            self.r_value_real = 0.1
            self.r_value_imag = 0.0

    term = Term()

    class Model:
        terms = {"term": term}

    def fake_hamiltonians(_moire_config, model, kpoints):
        value = next(iter(model.terms.values())).r_value_real
        return np.array([[[value, 0.0], [0.0, 1.0 - value]] for _ in range(len(kpoints))], dtype=np.complex128)

    captured = {}

    class Result:
        x = np.array([0.1], dtype=float)
        nfev = 1
        cost = 0.0

    def fake_least_squares(fun, y0, **kwargs):
        captured["has_jac"] = "jac" in kwargs
        residual = fun(y0)
        jac = kwargs["jac"](y0)
        captured["residual_shape"] = residual.shape
        captured["jac_shape"] = jac.shape
        return Result()

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)
    monkeypatch.setattr(pipeline.scipy.optimize, "least_squares", fake_least_squares)

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_refinement_config={
            "enabled": True,
            "band_slice": [0, 2],
            "align": "none",
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "matrix_weight": 0.1,
            "matrix_sigma_mev": 10.0,
            "optimizer": "scipy_least_squares",
            "max_nfev": 1,
        },
    )

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, Model())

    assert captured == {
        "has_jac": True,
        "residual_shape": (3,),
        "jac_shape": (3, 1),
    }
    assert report["jacobian"]["enabled"] is True
    assert report["jacobian"]["mode"] == "analytic_hellmann_feynman"
    assert report["jacobian"]["matrix_loss"]["compressed"] is True
    assert report["jacobian"]["matrix_loss"]["source_rows"] == 8
    assert report["jacobian"]["matrix_loss"]["compressed_rows"] == 1


def test_band_refinement_acceptance_guard_reverts_worse_plot_window(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    class Term:
        tag = "Kinect"
        active = True
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))

        def __init__(self) -> None:
            self.r_value_real = 0.0
            self.r_value_imag = 0.0

    class Model:
        def __init__(self) -> None:
            self.term = Term()
            self.terms = {"term": self.term}

    model = Model()

    def fake_hamiltonians(_moire_config, model_obj, kpoints):
        value = float(model_obj.term.r_value_real)
        return np.array([[[value, 0.0], [0.0, 1.0]] for _k in kpoints], dtype=np.complex128)

    class Result:
        x = np.array([1.0])
        nfev = 1
        cost = 0.0

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)
    monkeypatch.setattr(pipeline.scipy.optimize, "least_squares", lambda *_args, **_kwargs: Result())

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={"mode": "auto_low_energy"},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_plot_config={"top_bands": 2, "align": "top"},
        band_refinement_config={
            "enabled": True,
            "mode": "auto_low_energy",
            "band_slice": [1, 2],
            "align": "top",
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "max_nfev": 1,
            "subspace_loss": {"enabled": False},
            "low_subspace_matrix_loss": {"enabled": False},
            "acceptance_guard": {"enabled": True, "max_rms_increase_mev": 0.0, "max_max_increase_mev": 0.0},
        },
    )

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, model)

    assert report["acceptance_guard"]["accepted"] is False
    assert report["acceptance_guard"]["reverted"] is True
    assert model.term.r_value_real == pytest.approx(0.0)
    assert report["max_scaled_coefficient_drift"] == pytest.approx(0.0)


def test_refinement_acceptance_guard_can_select_best_validation_alpha(tmp_path: Path) -> None:
    heff = np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
    heff_eig = np.linalg.eigvalsh(heff)
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={"model": {"target_bands": "top"}},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=tmp_path / "heff.npy",
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={"mode": "manual"},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_plot_config={"top_bands": 1, "align": "none"},
    )

    def h_from_y(y: np.ndarray) -> np.ndarray:
        value = float(np.asarray(y)[0])
        return np.array([[[0.0, 0.0], [0.0, value]]], dtype=np.complex128)

    selected_y, report = _apply_refinement_acceptance_guard(
        raw_cfg={
            "align": "none",
            "target_bands": "top",
            "acceptance_guard": {
                "enabled": True,
                "selection": "best_validation_window",
                "line_search_alphas": [1.0, 0.5],
                "max_rms_increase_mev": 2000.0,
                "max_max_increase_mev": 2000.0,
                "guard_primary_window": False,
                "guard_all_bands": False,
            },
        },
        model_config=model_cfg,
        base_h=h_from_y(np.array([0.0])),
        heff_eig=heff_eig,
        heff_all=heff,
        y0=np.array([0.0]),
        candidate_y=np.array([2.0]),
        h_from_y=h_from_y,
    )

    assert selected_y == pytest.approx([1.0])
    assert report is not None
    assert report["accepted"] is True
    assert report["selected_alpha"] == pytest.approx(0.5)
    assert len(report["trials"]) == 2


def test_band_refinement_uses_fit_kpoints_not_full_band_path(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.2]],
            [[0.0, 0.0], [0.0, 1.4]],
        ],
        dtype=np.complex128,
    )
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    class Term:
        tag = "Kinect"
        active = True
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))

        def __init__(self) -> None:
            self.r_value_real = 0.0
            self.r_value_imag = 0.0

    class Model:
        terms = {"term": Term()}

    seen_lengths: list[int] = []

    def fake_hamiltonians(_moire_config, model, kpoints):
        seen_lengths.append(len(kpoints))
        value = next(iter(model.terms.values())).r_value_real
        return np.array([[[0.0, value], [value, 1.0 + 0.1 * idx]] for idx, _k in enumerate(kpoints)], dtype=np.complex128)

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float),
        kpoints_fit=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0, 2],
        fit_selection_metadata={"mode": "auto_low_energy", "selected_indices": [0, 2]},
        band_indices=[0, 1, 2],
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_refinement_config={
            "enabled": True,
            "mode": "auto_low_energy",
            "band_slice": [1, 2],
            "align": "top",
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "max_nfev": 1,
            "subspace_loss": {"enabled": False},
            "low_subspace_matrix_loss": {"enabled": False},
        },
    )

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, Model())

    assert report["fit_kpoints"]["source"] == "fit.indices"
    assert report["fit_kpoints"]["count"] == 2
    assert seen_lengths
    assert set(seen_lengths) == {2}


def test_band_refinement_accepts_explicit_refine_indices(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.2]],
            [[0.0, 0.0], [0.0, 1.4]],
        ],
        dtype=np.complex128,
    )
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    class Term:
        tag = "Kinect"
        active = True
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))

        def __init__(self) -> None:
            self.r_value_real = 0.0
            self.r_value_imag = 0.0

    class Model:
        terms = {"term": Term()}

    seen_kpoints: list[np.ndarray] = []

    def fake_hamiltonians(_moire_config, model, kpoints):
        seen_kpoints.append(np.asarray(kpoints, dtype=float).copy())
        value = next(iter(model.terms.values())).r_value_real
        return np.array([[[0.0, value], [value, 1.0]] for _k in kpoints], dtype=np.complex128)

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float),
        kpoints_fit=np.array([[0.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={"mode": "manual", "selected_indices": [0]},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_refinement_config={
            "enabled": True,
            "solver": "linear_low_subspace",
            "indices": [0, 2],
            "band_slice": [1, 2],
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "regularization": 1.0,
        },
    )

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, Model())

    assert report["fit_kpoints"]["source"] == "fit.refine_bands.indices"
    assert report["fit_kpoints"]["selected_indices"] == [0, 2]
    assert seen_kpoints
    assert all(len(kpoints) == 2 for kpoints in seen_kpoints)


def test_band_refinement_linear_low_subspace_solver_updates_coefficients(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array([[[0.0, 0.0], [0.0, 2.0]]], dtype=np.complex128)
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    class Term:
        tag = "Kinect"
        active = True
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))

        def __init__(self) -> None:
            self.r_value_real = 1.0
            self.r_value_imag = 0.0

    class Model:
        def __init__(self) -> None:
            self.term = Term()
            self.terms = {"term": self.term}

    def fake_hamiltonians(_moire_config, model, kpoints):
        value = model.term.r_value_real
        return np.array([[[0.0, 0.0], [0.0, value]] for _k in kpoints], dtype=np.complex128)

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
        kpoints_fit=np.array([[0.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={"mode": "manual", "selected_indices": [0]},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_refinement_config={
            "enabled": True,
            "solver": "linear_low_subspace",
            "band_slice": [1, 2],
            "align": "none",
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "regularization": 0.0,
            "subspace_loss": {"enabled": False},
            "shell_projected_matrix_loss": {"enabled": True, "max_shells": 1},
            "low_subspace_matrix_loss": {"enabled": False},
        },
    )
    model = Model()

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, model)

    assert report["solver"] == "linear_low_subspace"
    assert model.term.r_value_real == pytest.approx(2.0)
    assert report["initial"]["top_band_rms_mev"] == pytest.approx(1000.0)
    assert report["refined"]["top_band_rms_mev"] == pytest.approx(0.0, abs=1.0e-9)
    assert report["shell_projected_matrix_loss"]["enabled"] is True
    assert report["shell_projected_matrix_loss"]["shells"][0]["subspace_bands"] == 1
    assert report["shell_projected_matrix_loss"]["refined"]["shells"][0]["subspace"]["mean_overlap"] == pytest.approx(1.0)
    assert report["shell_subspace_loss"]["legacy_alias_of"] == "shell_projected_matrix_loss"


def test_band_refinement_linear_low_subspace_guard_reverts_all_band_degradation(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    class Term:
        tag = "Kinect"
        active = True
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))

        def __init__(self) -> None:
            self.r_value_real = 0.0
            self.r_value_imag = 0.0

    class Model:
        def __init__(self) -> None:
            self.term = Term()
            self.terms = {"term": self.term}

    def fake_hamiltonians(_moire_config, model, kpoints):
        value = float(model.term.r_value_real)
        return np.array([[[1.0 - value, 0.0], [0.0, 1.0 + 10.0 * value]] for _k in kpoints], dtype=np.complex128)

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={"model": {"target_bands": "bottom"}},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={"mode": "manual", "selected_indices": [0]},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_plot_config={"bottom_bands": 1, "align": "none"},
        band_refinement_config={
            "enabled": True,
            "solver": "linear_low_subspace",
            "band_slice": [0, 1],
            "align": "none",
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "regularization": 0.0,
            "acceptance_guard": {
                "enabled": True,
                "max_rms_increase_mev": 0.0,
                "max_max_increase_mev": 0.0,
                "line_search_alphas": [1.0],
            },
        },
    )
    model = Model()

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, model)

    assert report["solver"] == "linear_low_subspace"
    assert report["acceptance_guard"]["accepted"] is False
    assert report["acceptance_guard"]["reverted"] is True
    assert report["acceptance_guard"]["failed_windows"] == ["all_bands"]
    assert model.term.r_value_real == pytest.approx(0.0)
    assert report["refined"]["top_band_rms_mev"] == report["initial"]["top_band_rms_mev"]
    assert report["max_scaled_coefficient_drift"] == pytest.approx(0.0)


def test_band_refinement_linear_low_subspace_uses_direct_term_response(tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array([[[0.0, 0.0], [0.0, 2.0]]], dtype=np.complex128)
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
    term = ContinuumTerm(
        key=key,
        Y_basis=lambda _k: np.diag([0.0, 1.0]).astype(np.complex128),
        r_value_real=1.0,
        r_value_imag=0.0,
        active=True,
        tag="Kinect",
        symmetry_ops=[],
    )

    class Model:
        terms = {key: term}

    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0]], dtype=float),
        kpoints_fit=np.array([[0.0, 0.0]], dtype=float),
    )
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=tmp_path / "out",
        rotation_deg=0.0,
        fit_indices=[0],
        fit_selection_metadata={"mode": "manual", "selected_indices": [0]},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_refinement_config={
            "enabled": True,
            "solver": "linear_low_subspace",
            "band_slice": [1, 2],
            "align": "none",
            "variable_tags": ["Kinect"],
            "components": ["real"],
            "regularization": 0.0,
        },
    )

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, Model())

    assert report["solver"] == "linear_low_subspace"
    assert report["n_variables"] == 1
    assert term.r_value_real == pytest.approx(2.0)
    assert report["refined"]["top_band_rms_mev"] == pytest.approx(0.0, abs=1.0e-9)


def test_auto_model_selection_outputs_write_user_facing_reports(monkeypatch, tmp_path: Path) -> None:
    import kp.model.pipeline as pipeline

    heff = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.1]],
        ],
        dtype=np.complex128,
    )
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    def fake_hamiltonians(_moire_config, _model, _kpoints):
        return heff.copy()

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={"model": {"n_orb": [1, 0], "target_bands": "top"}, "fit": {"mode": "auto_low_energy"}},
        source_config=tmp_path / "source.yaml",
        source_raw={},
        qset1_file=tmp_path / "q1.npy",
        qset2_file=tmp_path / "q2.npy",
        kpoints_file=None,
        heff_file=heff_file,
        heff_eig_file=None,
        output_dir=output_dir,
        rotation_deg=0.0,
        fit_indices=[0, 1],
        fit_selection_metadata={"mode": "auto_low_energy", "selected_indices": [0, 1]},
        band_indices=None,
        n_orb=(1, 0),
        nlow_state=[1, 0],
        bM_config={},
        harmonics_config={},
        max_order={},
        symmetry_map={},
        coeff_tol=1.0e-8,
        coeff_prune_threshold=0.0,
        compare_to_heff=True,
        band_refinement_config={
            "enabled": True,
            "mode": "auto_low_energy",
            "target_bands": "top",
            "auto_harmonic_selection": {
                "enabled": True,
                "method": "heff_harmonic_ablation",
                "selection_status": "accepted_compact_candidate",
                "selected": {"intra_shells": 1, "inter_shells": 0, "accepted": True},
                "candidates": [],
            },
            "auto_windows": {
                "primary": {
                    "role": "primary",
                    "n_bands": 1,
                    "band_slice": [1, 2],
                    "boundary_gap_mev": 1000.0,
                    "state": "ok",
                    "use_for_loss": True,
                },
                "expanded": [],
            },
        },
    )
    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((0, 2), dtype=float),
        n_orb1=1,
        n_orb2=0,
        kpoints=np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float),
    )
    results = {
        "model": object(),
        "band_refinement": {
            "enabled": True,
            "n_variables": 3,
            "subspace_loss": {"enabled": True},
            "low_subspace_matrix_loss": {"enabled": True},
        },
    }

    summary = _write_auto_model_selection_outputs(
        results=results,
        output_dir=output_dir,
        model_config=model_cfg,
        moire_config=moire_cfg,
    )

    assert summary["selected_candidate"] == "configured_auto_low_energy"
    assert (output_dir / "auto_model_selection.json").exists()
    assert (output_dir / "auto_model_selection.md").exists()
    assert (output_dir / "selected_model_config.yaml").exists()
    assert (output_dir / "candidate_metrics.csv").exists()
    assert (output_dir / "matrix_residual_report.json").exists()
    assert (output_dir / "subspace_leakage.pdf").exists()
    payload = json.loads((output_dir / "auto_model_selection.json").read_text(encoding="utf-8"))
    assert payload["harmonic_selection"]["selected"]["intra_shells"] == 1
    md = (output_dir / "auto_model_selection.md").read_text(encoding="utf-8")
    assert "Harmonic Selection" in md


def test_band_refinement_variable_selection_accepts_string_aliases() -> None:
    class Term:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.active = True

    kinetic = Term("Kinect")
    inter = Term("inter")

    class Model:
        terms = {"kinetic": kinetic, "inter": inter}

    variables = _select_band_refinement_variables(
        Model(),
        {"variable_tags": "kinetic", "components": "real"},
    )

    assert variables == [(kinetic, "real")]


def test_load_model_config_accepts_human_readable_model_aliases(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"].pop("nlow_state")
    raw["model"].pop("bM")
    raw["model"]["harmonics"] = {"intralayer": 3, "interlayer": 2}
    raw["model"]["max_order"] = {"kinetic": 12, "intralayer": 2, "interlayer": 1}
    raw["model"]["symmetry_map"] = {"kinetic": [], "onsite": [], "intralayer": [], "interlayer": []}
    raw["model"]["max_derivative_order"] = {
        "intralayer_nonzero": 4,
        "interlayer_zero": 5,
        "interlayer_nonzero": 6,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.nlow_state == [1, 1]
    assert cfg.orbital_count_metadata["nlow_state"]["input_kind"] == "default_from_n_orb"
    assert cfg.harmonics_config == {"intra": 3, "inter": 2}
    assert {key: cfg.max_order[key] for key in ("Kinect", "intra", "inter")} == {
        "Kinect": 12,
        "intra": 2,
        "inter": 1,
    }
    assert cfg.max_order["moire_intra_nonzero"] == 4
    assert cfg.max_order["tunneling_zero"] == 5
    assert cfg.max_order["tunneling_nonzero"] == 6
    assert set(cfg.symmetry_map) == {"Kinect", "Onsite", "intra", "inter"}


def test_missing_symmetry_map_defaults_to_symmetry_source_operations(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"].pop("allowed_internal_symmetries", None)
    raw["valley_model"].pop("external_sewing_symmetries", None)
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "operations": [{"name": "C3z"}]}
    raw["model"].pop("symmetry_map")
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert set(cfg.symmetry_map) == {"Kinect", "Onsite", "intra", "inter"}
    assert {op["name"] for op in cfg.symmetry_map["Kinect"]} == {"C3z"}
    assert cfg.symmetry_map["Kinect"] == cfg.symmetry_map["Onsite"]
    assert cfg.symmetry_map["Kinect"] == cfg.symmetry_map["intra"]
    assert cfg.symmetry_map["Kinect"] == cfg.symmetry_map["inter"]


def test_explicit_symmetry_map_overrides_symmetry_source_default(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"].pop("allowed_internal_symmetries", None)
    raw["valley_model"].pop("external_sewing_symmetries", None)
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "operations": [{"name": "C3z"}]}
    raw["model"]["symmetry_map"] = {"kinetic": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.symmetry_map == {"Kinect": []}


def test_human_readable_model_aliases_reject_conflicts(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["harmonics"] = {"intra": 2, "intralayer": 3}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.harmonics uses both"):
        load_model_config(cfg_path)


def test_auto_compact_fit_indices_are_selected_when_indices_are_omitted(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"] = {
        "mode": "auto_compact",
        "max_points": 2,
        "coeff_tol": 1.0e-8,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.fit_indices == [0, 2]
    assert cfg.fit_selection_metadata == {
        "mode": "auto_compact",
        "source": "auto_compact_path_spacing",
        "requested_max_points": 2,
        "candidate_count": 3,
        "selected_indices": [0, 2],
    }


def test_auto_compact_fit_indices_use_deterministic_path_spacing() -> None:
    import kp.model.pipeline as configured_pipeline

    kpoints = np.zeros((61, 2), dtype=float)

    indices, metadata = configured_pipeline._select_auto_fit_indices(
        kpoints,
        {"mode": "auto_compact", "max_points": 4},
    )

    assert indices == [0, 20, 40, 60]
    assert metadata["selected_indices"] == [0, 20, 40, 60]
    assert metadata["candidate_count"] == 61


def test_load_model_config_rejects_negative_coeff_prune_threshold(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["coeff_prune_threshold"] = -1.0e-3
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="coeff_prune_threshold"):
        load_model_config(cfg_path)


def test_load_model_config_infers_n_orb_and_safe_defaults(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["source_config"] = "source.yaml"
    raw["valley_model"]["allowed_internal_symmetries"] = []
    raw["model"] = {
        "bM": {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]},
        "harmonics": {
            "intra": {1: "zero"},
            "inter": {1: "zero"},
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)
    moire_cfg, _ = build_moire_config_from_file(cfg_path)
    model = build_model(moire_cfg)

    assert cfg.n_orb == (1, 1)
    assert cfg.nlow_state == [1, 1]
    assert {key: cfg.max_order[key] for key in ("Kinect", "intra", "inter")} == {
        "Kinect": 2,
        "intra": 0,
        "inter": 0,
    }
    assert cfg.max_order["tunneling_zero"] == 0
    assert cfg.max_order["tunneling_nonzero"] == 0
    assert len(model.terms) < 30


def test_load_model_config_resolves_layerwise_n_orb_from_num_layer_list(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_path = tmp_path / "source.yaml"
    source_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_raw["material"]["num_layer_list"] = [1, 2]
    source_path.write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")
    raw["model"]["n_orb"] = [1, 1, 0]
    raw["model"].pop("nlow_state", None)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (1, 1)
    assert cfg.nlow_state == [1, 1]
    assert cfg.orbital_count_metadata["num_layer_list"] == [1, 2]
    assert cfg.orbital_count_metadata["n_orb"]["input_kind"] == "physical_layer"
    assert cfg.orbital_count_metadata["n_orb"]["raw"] == [1, 1, 0]
    assert cfg.orbital_count_metadata["n_orb"]["resolved_qset"] == [1, 1]
    assert cfg.orbital_count_metadata["n_orb"]["groups"] == [
        {"qset": "qset1", "source_group": 1, "layers": [1], "values": [1], "total": 1},
        {"qset": "qset2", "source_group": 2, "layers": [2, 3], "values": [1, 0], "total": 1},
    ]
    assert cfg.orbital_count_metadata["nlow_state"]["input_kind"] == "default_from_n_orb"


def test_load_model_config_drops_inactive_layer_for_gamma_1plus2_model(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_path = tmp_path / "source.yaml"
    source_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_raw["material"]["num_layer_list"] = [1, 2]
    source_path.write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")
    raw["model"]["n_orb"] = [0, 2, 2]
    raw["model"]["nlow_state"] = [0, 2, 2]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (2, 2)
    assert cfg.nlow_state == [2, 2]
    assert cfg.orbital_count_metadata["n_orb"]["resolved_model_sectors"] == [2, 2]
    assert cfg.term_template_metadata["profiles"] == ["gamma_2x2_independent_sectors"]


def test_load_model_config_resolves_layerwise_nlow_state_independently(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_path = tmp_path / "source.yaml"
    source_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_raw["material"]["num_layer_list"] = [1, 2]
    source_path.write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")
    raw["model"]["n_orb"] = [2, 2, 0]
    raw["model"]["nlow_state"] = [0, 2, 2]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (2, 2)
    assert cfg.nlow_state == [2, 2]
    assert cfg.orbital_count_metadata["n_orb"]["resolved_qset"] == [2, 2]
    assert cfg.orbital_count_metadata["nlow_state"]["resolved_qset"] == [0, 4]
    assert cfg.orbital_count_metadata["nlow_state"]["resolved_model_sectors"] == [2, 2]


def test_load_model_config_rejects_legacy_two_entry_n_orb_with_num_layer_list(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    source_path = tmp_path / "source.yaml"
    source_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_raw["material"]["num_layer_list"] = [1, 2]
    source_path.write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.n_orb"):
        load_model_config(cfg_path)


def test_short_spinless_label_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley"] = "K1"
    raw["spin"] = "spinless"
    raw.pop("valley_model", None)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="spin_up_projected.*spinless_effective"):
        load_model_config(cfg_path)


def test_load_model_config_rejects_sector_n_orb_mismatch(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["n_orb"] = [2, 2]
    raw["sectors"] = [
        {"name": "bottom", "qset": "qset1", "n_orb": 1},
        {"name": "top", "qset": "qset2", "n_orb": 2},
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"sectors\.bottom\.n_orb=1.*model\.n_orb.*2"):
        load_model_config(cfg_path)


def test_load_model_config_rejects_user_exactification_knobs(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "operations": ["C3z"],
        "exactification": {"support_mode": "monomial"},
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source matrix projection is internal"):
        load_model_config(cfg_path)


def test_ptse2_release_config_hides_internal_exactification_knobs() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "ptse2_gamma_q4"
        / "kp"
        / "configs"
        / "ptse2_gamma_q4_symmhamk_fourpz_Q4.yaml"
    )
    if not path.exists():
        pytest.skip("PtSe2 release example is not present in this checkout")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert "exactification" not in raw.get("symm", {})
    model = raw.get("model", {})
    assert "harmonics" not in model
    assert "max_order" not in model
    assert "max_derivative_order" not in model
    assert model.get("fit") == {"mode": "auto_low_energy"}


def test_load_model_config_marks_symmetry_source_inferred_from_source_config(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    symm_dir = _write_symm_frame_manifest(tmp_path, rotation_deg=0.0, path_name="symm_from_source")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.pop("symmetry_source", None)
    source_raw = yaml.safe_load((tmp_path / "source.yaml").read_text(encoding="utf-8"))
    source_raw["symm"] = {"output_dir": "symm_from_source"}
    (tmp_path / "source.yaml").write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.symmetry_source_config["type"] == "kp_symm_output"
    assert cfg.symmetry_source_config["path"] == str(symm_dir.resolve())
    assert cfg.symmetry_source_metadata["inferred_from_source_config"] is True


def test_build_moire_config_reads_model_q_sets_from_symmetry_artifact(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "action"}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    q1_model = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]], dtype=float)
    q2_model = np.array([[20.0, 0.0], [21.0, 0.0], [22.0, 0.0]], dtype=float)
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    np.save(symm_dir / "q_model_layer1.npy", q1_model)
    np.save(symm_dir / "q_model_layer2.npy", q2_model)
    (symm_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frame": {"q_transform": {"rotation_deg": 90.0}},
                "q_model": {"files": {"layer1": "q_model_layer1.npy", "layer2": "q_model_layer2.npy"}},
                "operations": [],
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, _model_cfg = build_moire_config_from_file(cfg_path)

    np.testing.assert_allclose(moire_cfg.Q_set1, q1_model)
    np.testing.assert_allclose(moire_cfg.Q_set2, q2_model)


def test_short_kp_symm_source_does_not_infer_matrix_kind_from_valley(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = "symm"
    raw["valley_model"]["valley_type"] = "Gamma"
    raw["valley_model"]["active_valleys"] = ["Gamma"]
    raw["valley_model"]["allowed_internal_symmetries"] = ["C2"]
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert "matrix_kind" not in cfg.symmetry_source_config
    assert "operations" not in cfg.symmetry_source_config
    assert cfg.symmetry_map["Kinect"][0]["name"] == "C2"


def test_matrix_kind_operations_require_explicit_action_metadata(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "use": "raw",
        "matrix_kind": "action",
        "operations": ["C2T"],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="requires explicit action metadata"):
        load_model_config(cfg_path)


def test_matrix_kind_operations_require_explicit_source_semantics(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "use": "raw",
        "matrix_kind": "action",
        "operations": [
            {
                "name": "C2T",
                "antiunitary": True,
                "k_map": {"type": "identity"},
                "q_map": {"type": "identity"},
                "sector_map": "layer_exchange",
            }
        ],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source_matrix_role"):
        load_model_config(cfg_path)


def test_model_config_rejects_representation_matrix_kind_during_normalization(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "use": "raw",
        "matrix_kind": "representation",
        "operations": [
            {
                "name": "C2",
                "antiunitary": False,
                "k_map": {"type": "reflection", "axis_deg": 0.0},
                "q_map": {"type": "reflection", "axis_deg": 0.0},
                "sector_map": "identity",
            }
        ],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="matrix_kind must be 'action'"):
        load_model_config(cfg_path)


def test_model_max_order_overrides_safe_defaults(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["max_order"] = {"Kinect": 4, "intra": 1, "inter": 0}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert {key: cfg.max_order[key] for key in ("Kinect", "intra", "inter")} == {
        "Kinect": 4,
        "intra": 1,
        "inter": 0,
    }
    assert cfg.max_order["tunneling_zero"] == 0
    assert cfg.max_order["tunneling_nonzero"] == 0


def test_evaluate_vector_expression_supports_bM_symbols() -> None:
    variables = {
        "bM1": np.array([1.0, 0.0]),
        "bM2": np.array([0.0, 2.0]),
        "zero": np.zeros(2),
    }

    np.testing.assert_allclose(evaluate_vector_expression("bM1 + 0.5*bM2", variables), [1.0, 1.0])
    np.testing.assert_allclose(evaluate_vector_expression("-bM1", variables), [-1.0, 0.0])
    np.testing.assert_allclose(evaluate_vector_expression([3.0, 4.0], variables), [3.0, 4.0])


def test_default_term_templates_are_profile_driven_and_preserve_k_gamma_m_behavior() -> None:
    max_order = _max_derivative_order_values({"max_order": {"Kinect": 10, "intra": 4, "inter": 3}})

    k_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "K"},
        n_orb=(1, 1),
        max_order=max_order,
    )
    assert [row["name"] for row in k_templates] == [
        "kinetic_layer1",
        "onsite_layer1",
        "intra_layer1",
        "inter_21_positive",
        "inter_21_negative",
    ]
    assert k_templates[0]["max_order"] == 10
    assert k_templates[2]["harmonics"] == {"kind": "intra", "indices": [2, 3, 4]}
    assert k_templates[4]["harmonics"] == {"kind": "inter", "indices": [2, 3, 4], "sign": -1.0}

    gamma_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=max_order,
    )
    assert [row["name"] for row in gamma_templates][-2:] == [
        "gamma_inter_nonzero",
        "gamma_inter_nonzero_negative",
    ]
    gamma_by_name = {row["name"]: row for row in gamma_templates}
    for name in ["gamma_kinetic", "gamma_onsite", "gamma_intra_zero", "gamma_intra_zero_kdependent", "gamma_intra_nonzero"]:
        assert gamma_by_name[name]["sector_pairs"] == [[1, 1], [2, 2]]
    assert gamma_by_name["gamma_intra_zero_kdependent"]["harmonics"] == {"kind": "intra", "indices": [1]}
    assert gamma_by_name["gamma_intra_zero_kdependent"]["max_order"] == 4
    assert gamma_by_name["gamma_intra_zero_kdependent"]["monomial_constraints"]["exclude_m_sum_zero"] is True
    assert gamma_templates[-2]["max_order"] == 3
    assert gamma_templates[-1]["max_order"] == 3


def test_gamma_2x2_term_templates_collapse_only_with_sector_exchange_symmetry() -> None:
    max_order = _max_derivative_order_values({"max_order": {"Kinect": 10, "intra": 4, "inter": 3}})

    gamma_1x1 = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(1, 1),
        max_order=max_order,
        symmetry_operations=[{"name": "C2", "sector_map": "layer_exchange"}],
    )
    gamma_1x1_by_name = {row["name"]: row for row in gamma_1x1}
    assert [row["name"] for row in gamma_1x1] == [
        "gamma_1x1_kinetic",
        "gamma_1x1_onsite",
        "gamma_1x1_intra_nonzero",
        "gamma_1x1_inter_zero",
        "gamma_1x1_inter_nonzero",
    ]
    for name in ["gamma_1x1_kinetic", "gamma_1x1_onsite", "gamma_1x1_intra_nonzero"]:
        assert gamma_1x1_by_name[name]["sector_pairs"] == [[1, 1], [2, 2]]
    assert gamma_1x1_by_name["gamma_1x1_intra_nonzero"]["harmonics"] == {"kind": "intra", "indices": [2, 3, 4]}
    assert gamma_1x1_by_name["gamma_1x1_inter_zero"]["sector_pairs"] == [[2, 1], [1, 2]]
    assert gamma_1x1_by_name["gamma_1x1_inter_nonzero"]["sector_pairs"] == [[2, 1], [1, 2]]
    assert gamma_1x1_by_name["gamma_1x1_inter_zero"]["max_order"] == 3
    assert gamma_1x1_by_name["gamma_1x1_inter_nonzero"]["max_order"] == 3
    assert all(
        row["source"] != "moire_potential" or 1 not in row.get("harmonics", {}).get("indices", [])
        for row in gamma_1x1
    )

    gamma_1x1_non_exchanged = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(1, 1),
        max_order=max_order,
        symmetry_operations=[{"name": "TR", "sector_map": "identity"}, {"name": "C3z", "sector_map": "identity"}],
    )
    gamma_1x1_non_exchanged_by_name = {row["name"]: row for row in gamma_1x1_non_exchanged}
    for name in ["gamma_1x1_kinetic", "gamma_1x1_onsite", "gamma_1x1_intra_nonzero"]:
        assert gamma_1x1_non_exchanged_by_name[name]["sector_pairs"] == [[1, 1], [2, 2]]

    exchanged = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=max_order,
        symmetry_operations=[{"name": "C2", "sector_map": "layer_exchange"}],
    )
    exchanged_by_name = {row["name"]: row for row in exchanged}
    for name in ["gamma_kinetic", "gamma_onsite", "gamma_intra_zero", "gamma_intra_zero_kdependent", "gamma_intra_nonzero"]:
        assert exchanged_by_name[name]["sector_pairs"] == [[1, 1]]

    generic_exchanged = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=max_order,
        symmetry_operations=[{"name": "C2", "sector_map": {"bottom": "top", "top": "bottom"}}],
    )
    generic_exchanged_by_name = {row["name"]: row for row in generic_exchanged}
    for name in ["gamma_kinetic", "gamma_onsite", "gamma_intra_zero", "gamma_intra_zero_kdependent", "gamma_intra_nonzero"]:
        assert generic_exchanged_by_name[name]["sector_pairs"] == [[1, 1]]

    non_exchanged = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=max_order,
        symmetry_operations=[{"name": "TR", "sector_map": "identity"}, {"name": "C3z", "sector_map": "identity"}],
    )
    non_exchanged_by_name = {row["name"]: row for row in non_exchanged}
    for name in ["gamma_kinetic", "gamma_onsite", "gamma_intra_zero", "gamma_intra_zero_kdependent", "gamma_intra_nonzero"]:
        assert non_exchanged_by_name[name]["sector_pairs"] == [[1, 1], [2, 2]]

    metadata = _default_term_template_profile_metadata(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        symmetry_operations=[{"name": "C2", "sector_map": "layer_exchange"}],
    )
    assert metadata["profiles"] == ["gamma_2x2_symmetry_aware"]
    assert metadata["gamma_2x2_sector_diagonal_pairs"] == [[1, 1]]

    m_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "M"},
        n_orb=(1, 1),
        max_order=max_order,
    )
    assert [row["name"] for row in m_templates] == [
        "m1_kinetic_bottom",
        "m1_onsite_bottom",
        "m1_intra_bottom",
        "m1_inter_top_to_bottom",
    ]
    assert m_templates[-1]["harmonics"] == "inter"


def test_gamma_max_derivative_order_uses_semantic_tunneling_names() -> None:
    max_order = _max_derivative_order_values(
        {
            "max_order": {"Kinect": 10, "intra": 4, "inter": 3},
            "max_derivative_order": {"tunneling_nonzero": 6},
        }
    )

    gamma_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=max_order,
    )
    by_name = {row["name"]: row for row in gamma_templates}

    assert by_name["gamma_inter_zero"]["max_order"] == 3
    assert by_name["gamma_inter_nonzero"]["max_order"] == 6
    assert by_name["gamma_inter_nonzero_negative"]["max_order"] == 6


def test_gamma_2x2_derivative_split_is_case_specific_override() -> None:
    default_max_order = _max_derivative_order_values(
        {"max_order": {"Kinect": 10, "intra": 6, "inter": 6}},
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
    )
    default_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=default_max_order,
    )
    default_by_name = {row["name"]: row for row in default_templates}
    assert default_by_name["gamma_inter_zero"]["max_order"] == 6

    case_max_order = _max_derivative_order_values(
        {
            "max_order": {"Kinect": 10, "intra": 6, "inter": 6},
            "max_derivative_order": {
                "moire_intra_zero": 0,
                "moire_intra_nonzero": 6,
                "tunneling_zero": 10,
                "tunneling_nonzero": 6,
            },
        }
    )
    case_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=case_max_order,
    )
    case_by_name = {row["name"]: row for row in case_templates}

    assert case_by_name["gamma_intra_zero"]["max_order"] == 0
    assert case_by_name["gamma_intra_zero_kdependent"]["max_order"] == 6
    assert case_by_name["gamma_intra_nonzero"]["max_order"] == 6
    assert case_by_name["gamma_inter_zero"]["max_order"] == 10
    assert case_by_name["gamma_inter_nonzero"]["max_order"] == 6
    assert case_by_name["gamma_inter_nonzero_negative"]["max_order"] == 6


def test_default_term_templates_clamp_profile_harmonics_to_requested_count() -> None:
    max_order = {"Kinect": 6, "intra": 4, "inter": 4}

    k_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "K"},
        n_orb=(1, 1),
        max_order=max_order,
        harmonic_counts={"intra": 3, "inter": 2},
    )

    assert k_templates[2]["harmonics"] == {"kind": "intra", "indices": [2, 3]}
    assert k_templates[3]["harmonics"] == {"kind": "inter", "indices": [1, 2]}
    assert k_templates[4]["harmonics"] == {"kind": "inter", "indices": [2], "sign": -1.0}

    gamma_templates = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order=max_order,
        harmonic_counts={"intra": 7, "inter": 6},
    )
    gamma_by_name = {row["name"]: row for row in gamma_templates}
    assert gamma_by_name["gamma_intra_nonzero"]["harmonics"] == {"kind": "intra", "indices": [2, 3, 4, 5, 6, 7]}
    assert gamma_by_name["gamma_inter_nonzero"]["harmonics"] == {"kind": "inter", "indices": [2, 3, 4, 5, 6]}
    assert gamma_by_name["gamma_inter_nonzero_negative"]["harmonics"] == {"kind": "inter", "indices": [2, 3, 4, 5, 6], "sign": -1.0}


def test_default_k_term_templates_are_generated_from_active_sectors() -> None:
    max_order = _max_derivative_order_values({"max_order": {"Kinect": 12, "intra": 2, "inter": 2}})
    sectors = [
        {"name": "A_bottom", "qset": "qset1", "n_orb": 1},
        {"name": "A_top", "qset": "qset2", "n_orb": 1},
    ]

    templates = _default_term_templates_for_model(
        valley_model={"valley_type": "K"},
        n_orb=(1, 1),
        max_order=max_order,
        harmonic_counts={"intra": 3, "inter": 2},
        sectors=sectors,
    )

    by_name = {row["name"]: row for row in templates}
    assert "kinetic_A_bottom" in by_name
    assert "kinetic_A_top" in by_name
    assert "intra_A_bottom_first_shell" in by_name
    assert "intra_A_top_second_shell" in by_name
    assert "inter_A_top_to_A_bottom" in by_name
    assert by_name["kinetic_A_bottom"]["sector_pairs"] == [["A_bottom", "A_bottom"]]
    assert by_name["kinetic_A_top"]["sector_pairs"] == [["A_top", "A_top"]]
    assert by_name["kinetic_A_bottom"]["monomial_constraints"] == {
        "exclude_m_sum_zero": True,
        "difference_mod": 3,
        "difference_residue": 0,
        "require_mz_ge_mz_star": True,
    }
    assert by_name["intra_A_bottom_first_shell"]["harmonics"] == {"kind": "intra", "indices": [2], "sign": -1.0}
    assert by_name["intra_A_top_second_shell"]["harmonics"] == {"kind": "intra", "indices": [3]}
    assert by_name["inter_A_top_to_A_bottom"]["harmonics"] == {"kind": "inter", "indices": [1, 2]}


def test_default_k_term_templates_omit_inter_for_single_active_sector() -> None:
    max_order = _max_derivative_order_values({"max_order": {"Kinect": 12, "intra": 2, "inter": 2}})

    templates = _default_term_templates_for_model(
        valley_model={"valley_type": "K"},
        n_orb=(0, 1),
        max_order=max_order,
        harmonic_counts={"intra": 3, "inter": 2},
        sectors=[{"name": "B_top", "qset": "qset2", "n_orb": 1}],
    )

    by_name = {row["name"]: row for row in templates}
    assert set(by_name) == {
        "kinetic_B_top",
        "onsite_B_top",
        "intra_B_top_first_shell",
        "intra_B_top_second_shell",
    }
    assert by_name["kinetic_B_top"]["sector_pairs"] == [["B_top", "B_top"]]
    assert all(row["source"] != "tunneling" for row in templates)


def test_auto_harmonics_are_generated_from_qset_symmetry_orbits() -> None:
    bM1 = np.array([1.0, 0.0], dtype=float)
    bM2 = np.array([0.5, np.sqrt(3.0) / 2.0], dtype=float)
    qset = _triangular_q_shell()

    intra, intra_diag = _auto_harmonics_from_q_sets(
        kind="intra",
        count=4,
        Q_set1=qset,
        Q_set2=qset,
        bM1=bM1,
        bM2=bM2,
    )
    assert intra_diag["generation"] == "symmetry_orbit"
    assert intra_diag["orbit_generators"] == ["HermitianPair"]
    assert intra_diag["selected"][1]["orbit_size"] > 1
    np.testing.assert_allclose(intra[1], [0.0, 0.0], atol=1.0e-12)

    sectors = [
        {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0]},
        {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0]},
    ]
    c3_op = {
        "name": "C3z",
        "antiunitary": False,
        "k_map": {"type": "rotation", "angle_deg": 120.0},
        "q_map": {"type": "rotation", "angle_deg": 120.0},
        "sector_map": "identity",
    }
    intra_with_c3, intra_c3_diag = _auto_harmonics_from_support(
        raw=4,
        kind="intra",
        count=4,
        sectors=sectors,
        Q_set1=qset,
        Q_set2=qset,
        bM1=bM1,
        bM2=bM2,
        symmetry_operations=[c3_op],
    )
    assert intra_c3_diag["generation"] == "qset_support_symmetry_orbit"
    assert intra_c3_diag["orbit_generators"] == ["HermitianPair", "C3z"]
    assert intra_c3_diag["selected"][1]["orbit_size"] > intra_diag["selected"][1]["orbit_size"]
    np.testing.assert_allclose(intra_with_c3[1], [0.0, 0.0], atol=1.0e-12)

    intra_with_named_c3, named_c3_diag = _auto_harmonics_from_support(
        raw=4,
        kind="intra",
        count=4,
        sectors=sectors,
        Q_set1=qset,
        Q_set2=qset,
        bM1=bM1,
        bM2=bM2,
        symmetry_operations=[{"name": "C3z"}],
    )
    assert named_c3_diag["orbit_generators"] == ["HermitianPair", "C3z"]
    for index, vector in intra_with_c3.items():
        np.testing.assert_allclose(intra_with_named_c3[index], vector, atol=1.0e-12)


def test_harmonic_representative_score_ignores_roundoff_tie_breaks() -> None:
    positive = np.array([0.17855031287009285, 0.10308607119637303])
    negative = np.array([-0.17855031287009285, -0.103086071196373])

    assert _representative_score(positive) < _representative_score(negative)


def test_symmetry_operation_index_rotates_q_map_when_model_action_absent() -> None:
    metadata = {
        "operations": [
            {
                "name": "C2T",
                "operation": "C2T",
                "antiunitary": True,
                "k_map": {"type": "reflection", "axis_deg": 60.0},
                "q_map": {"type": "reflection", "axis_deg": 60.0},
                "sector_map": "identity",
            }
        ]
    }

    index = _symmetry_operation_index(metadata, rotation_deg=210.0)

    assert index["C2T"]["k_map"]["axis_deg"] == pytest.approx(270.0)
    assert index["C2T"]["q_map"]["axis_deg"] == pytest.approx(270.0)
    assert index["C2T"]["sector_map"] == "identity"


def test_operation_registry_keeps_distinct_geometry_and_records_usage_tags(tmp_path: Path) -> None:
    cfg = load_model_config(_write_fixture(tmp_path))
    cfg.symmetry_source_config = {
        "type": "kp_symm_output",
        "operations": [
            {"name": "C2", "operation": "C2_raw"},
            {"name": "C2", "operation": "C2_shifted"},
        ],
    }
    first = {
        "user_name": "C2",
        "name": "C2",
        "operation": "C2_raw",
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0},
        "q_map": {"type": "reflection", "axis_deg": 0.0},
        "sector_map": "identity",
    }
    same_geometry_other_tag = dict(first)
    second = {
        **first,
        "operation": "C2_shifted",
        "q_map": {"type": "reflection", "axis_deg": 60.0},
        "sector_map": "layer_exchange",
    }
    cfg.symmetry_map = {
        "Kinect": [first, second],
        "intra": [same_geometry_other_tag],
    }
    cfg.term_templates = [
        {"name": "kinetic_layer1", "source": "diagonal_kp"},
        {"name": "intra_layer1", "source": "moire_potential"},
    ]

    registry = _build_operation_registry(cfg)

    assert len(registry) == 2
    by_source = {row["source_operation"]: row for row in registry}
    assert by_source["C2_raw"]["q_map"]["axis_deg"] == 0.0
    assert by_source["C2_shifted"]["q_map"]["axis_deg"] == 60.0
    assert by_source["C2_shifted"]["sector_map"] == "layer_exchange"
    assert by_source["C2_raw"]["used_by_tags"] == ["Kinect", "intra"]
    assert by_source["C2_raw"]["term_tags"] == ["intra_layer1", "kinetic_layer1"]


def test_kp_symm_exactification_uses_manifest_actions_without_model_discovery() -> None:
    exact_cfg = _kp_symm_exactification_config()

    assert not exact_cfg.get("discover_action_candidates", False)
    assert not exact_cfg.get("accept_support_resolved_action", False)
    assert "central_phase" not in exact_cfg
    assert exact_cfg["require_group_relations"] is True
    assert exact_cfg["source"] == "kp_symm"
    assert exact_cfg["reject_if_off_support_rel_gt"] == pytest.approx(1.0e-5)
    assert exact_cfg["reject_if_amplitude_deviation_gt"] == pytest.approx(0.05)


def test_kp_symm_exactification_tolerance_is_not_valley_dependent() -> None:
    assert _kp_symm_exactification_config() == _kp_symm_exactification_config()


def test_kp_symm_exactification_allows_explicit_projection_noise_override() -> None:
    exact_cfg = _kp_symm_exactification_config({"reject_if_off_support_rel_gt": 5.0e-3})

    assert exact_cfg["reject_if_off_support_rel_gt"] == pytest.approx(5.0e-3)
    assert _kp_symm_exactification_config()["reject_if_off_support_rel_gt"] == pytest.approx(1.0e-5)


def test_kp_symm_gamma_exactification_default_uses_symmetry_tolerance() -> None:
    exact_cfg = _merged_exactification_overrides("Gamma", None, symmetry_tolerance=2.0e-2)

    assert exact_cfg["reject_if_off_support_rel_gt"] == pytest.approx(2.0e-2)
    assert exact_cfg["operations"] == {"C3z": {"support_mode": "monomial"}}


def test_kp_symm_explicit_exactification_override_wins_over_symmetry_tolerance() -> None:
    exact_cfg = _merged_exactification_overrides(
        "Gamma",
        {"reject_if_off_support_rel_gt": 3.0e-3},
        symmetry_tolerance=2.0e-2,
    )

    assert exact_cfg["reject_if_off_support_rel_gt"] == pytest.approx(3.0e-3)


def test_kp_symm_exactification_preserves_operation_overrides() -> None:
    exact_cfg = _kp_symm_exactification_config(
        {
            "support_mode": "monomial",
            "operations": {"TR": {"support_mode": "monomial"}},
        }
    )

    assert exact_cfg["support_mode"] == "monomial"
    assert exact_cfg["operations"] == {"TR": {"support_mode": "monomial"}}


def test_kp_symm_exactification_rejects_unknown_override_keys() -> None:
    with pytest.raises(ValueError, match="Unsupported symm.exactification keys"):
        _kp_symm_exactification_config({"discover_action_candidates": True})


def test_kp_symm_operation_group_relations_are_manifest_metadata() -> None:
    assert _operation_power_relation("C2", spin_convention="spinful") == {
        "type": "power",
        "name": "C2^2",
        "operation": "C2",
        "power": 2,
        "phase": -1,
        "source": "kp_symm_manifest",
    }
    assert _operation_power_relation("TR", spin_convention="spinless_effective")["phase"] == 1
    assert _operation_power_relation("C3z", spin_convention="spinful")["phase"] == -1
    assert _operation_power_relation("C2T", spin_convention="spinful")["phase"] == 1


def test_spinful_gamma_kp_symm_exactification_requires_manifest_group_relations() -> None:
    exact_cfg = _kp_symm_exactification_config()

    assert "central_phase" not in exact_cfg
    assert exact_cfg["require_group_relations"] is True


def test_kp_symm_exactification_config_does_not_use_operation_entry_shape_for_phases() -> None:
    exact_cfg = _kp_symm_exactification_config()

    assert "central_phase" not in exact_cfg
    assert exact_cfg["require_group_relations"] is True


def test_model_rotation_reads_symmetry_manifest_without_coordinate_frame(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.pop("coordinate_frame", None)
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm"}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    (symm_dir / "manifest.json").write_text(
        json.dumps({"frame": {"q_transform": {"rotation_deg": 210.0}}, "operations": []}),
        encoding="utf-8",
    )

    cfg = load_model_config(cfg_path)

    assert cfg.rotation_deg == 210.0


def test_model_rotation_fails_without_manifest_frame(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.pop("coordinate_frame", None)
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm"}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    (symm_dir / "manifest.json").write_text(json.dumps({"operations": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="symmetry manifest frame rotation"):
        load_model_config(cfg_path)


def test_build_moire_config_loads_fit_block_and_band_kpoints(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.heff_file == tmp_path / "project" / "heff_list.npy"
    assert moire_cfg.Q_set1.shape == (2, 2)
    assert moire_cfg.Q_set2.shape == (2, 2)
    assert moire_cfg.n_orb1 == 1
    assert moire_cfg.n_orb2 == 1
    assert moire_cfg.heff.shape == (8, 8)
    np.testing.assert_allclose(moire_cfg.kpoints_fit, [[0.0, 0.0], [0.2, 0.0]])
    np.testing.assert_allclose(moire_cfg.kpoints, [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    np.testing.assert_allclose(moire_cfg.intra_harmonics_map[3], moire_cfg.bM1 + moire_cfg.bM2)


def test_build_moire_config_auto_harmonics_selects_q_shell_stars(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    b = float(np.linalg.norm(moire_cfg.bM1))
    expected_norms = np.array([0.0, b, b, b])
    intra_norms = np.array([np.linalg.norm(moire_cfg.intra_harmonics_map[i]) for i in range(1, 5)])
    inter_norms = np.array([np.linalg.norm(moire_cfg.inter_harmonics_map[i]) for i in range(1, 5)])
    np.testing.assert_allclose(intra_norms, expected_norms, atol=1.0e-8)
    np.testing.assert_allclose(inter_norms, expected_norms, atol=1.0e-8)
    assert model_cfg.harmonics_diagnostics["auto_used"] is True
    assert model_cfg.harmonics_diagnostics["intra"]["selected"][1]["pair_count"] > 0


def test_build_moire_config_infers_bM_from_q_distances_not_q_norm(tmp_path: Path) -> None:
    radius = 0.5
    angles = np.deg2rad([0.0, 120.0, 240.0])
    qset = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    kpoints = np.array([[0.0, 0.0]], dtype=float)
    dim = 2 * len(qset)
    heff = np.diag(np.arange(dim, dtype=float))[None, :, :].astype(np.complex128)

    np.save(tmp_path / "q1.npy", -qset)
    np.save(tmp_path / "q2.npy", -qset)
    np.save(tmp_path / "kpoints.npy", kpoints)
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    np.save(out_dir / "heff_list.npy", heff)
    np.save(out_dir / "heff_eig.npy", np.linalg.eigvalsh(heff))
    (tmp_path / "source.yaml").write_text(
        yaml.safe_dump(
            {
                "material": {"qset1_file": "q1.npy", "qset2_file": "q2.npy"},
                "plot": {},
                "project": {"out_dir": "project"},
            }
        ),
        encoding="utf-8",
    )
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "source_config": "source.yaml",
                "symmetry_source": {"type": "kp_symm_output", "path": "symm"},
                "valley_model": {
                    "lattice": "hexagonal",
                    "system": "bilayer",
                    "valley_type": "Gamma",
                    "mode": "single_valley",
                    "active_valleys": ["Gamma"],
                    "spin_convention": "spinless_effective",
                    "allowed_internal_symmetries": [],
                    "external_sewing_symmetries": [],
                },
                "kpoints_file": "kpoints.npy",
                "model": {
                    "n_orb": [1, 1],
                    "nlow_state": [1, 1],
                    "bM": {"source": "q_distance"},
                    "harmonics": {"intra": 1, "inter": 1},
                    "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
                    "symmetry_map": {},
                },
                "fit": {"indices": [0]},
                "bands": {"indices": [0], "compare_to_heff": False},
                "output": {"dir": "model_out"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    moire_cfg, _ = build_moire_config_from_file(cfg_path)

    assert np.linalg.norm(moire_cfg.bM1) > radius
    np.testing.assert_allclose(np.linalg.norm(moire_cfg.bM1), np.sqrt(3.0) * radius, atol=1.0e-12)


def test_build_moire_config_infers_bM_from_tmat_with_shared_rotation(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {"source": "tmat"}
    raw["kpath"] = {
        "file": "KPATH.in",
        "tmat": [
            [60.9508659523, 0.0, 0.0],
            [-30.4754329751, 52.7849982976, 0.0],
            [0.0, 0.0, 50.0],
        ],
    }
    _write_symm_frame_manifest(tmp_path, rotation_deg=210.0)
    (tmp_path / "KPATH.in").write_text(
        "\n".join(
            [
                "K-Path Generated by VASPKIT.",
                "1",
                "Line-Mode",
                "Reciprocal",
                "0 0 0 Gamma",
                "0.5 0 0 M",
            ]
        ),
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.rotation_deg == 210
    np.testing.assert_allclose(moire_cfg.bM1, [0.11903354191339506, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(moire_cfg.bM2, [0.05951677095669753, 0.10308607119512161], atol=1.0e-10)


def test_load_model_config_rejects_duplicate_model_rotation_settings(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["q_rotation_deg"] = 0
    raw["kpath"] = {"phase_deg": 0}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Model frame rotation must come from the kp_symm manifest"):
        load_model_config(cfg_path)


def test_load_model_config_rejects_coordinate_frame_rotation(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["coordinate_frame"] = {"rotation_deg": 210}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="coordinate_frame is not a supported model input"):
        load_model_config(cfg_path)


def test_T_template_generator_rejects_spinless_nlow_state() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    gen = SymmetryGenerator(q, q, [1, 1])

    with pytest.raises(ValueError, match="TR template generator requires explicit spin/Kramers pair basis"):
        gen.get_time_reversal_matrix()


def test_noncanonical_internal_operation_rejected(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "Gamma_four_orbital"}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "bad_op"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported internal symmetry operation names"):
        load_model_config(cfg_path)


def test_toy_generator_uses_current_q_frame_without_kp_symm_manifest(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "Gamma_four_orbital"}
    raw["model"]["symmetry_map"] = {"Kinect": [], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.rotation_deg == pytest.approx(0.0)


def test_C2_template_generator_requires_template() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    gen = SymmetryGenerator(q, q, [2, 2])

    with pytest.raises(ValueError, match="C2 template generator requires"):
        gen.get_C2_operator()


def test_C3z_template_generator_requires_template() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    gen = SymmetryGenerator(q, q, [2, 2])

    with pytest.raises(ValueError, match="C3z template generator requires"):
        gen.get_C3z_operator(1)


def test_K_single_valley_rejects_T_C2(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "K",
        "mode": "single_valley",
        "active_valleys": ["K1"],
        "spin_convention": "spinful",
        "allowed_internal_symmetries": ["C3z", "C2T"],
        "external_sewing_symmetries": ["TR", "C2"],
    }
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "operations": []}
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.5, 0.8660254037844386]}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "TR"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="K single_valley cannot use physical TR/C2"):
        load_model_config(cfg_path)

    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C3z"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_model_config(cfg_path)
    assert cfg.valley_model["valley_type"] == "K"


def test_K_single_valley_keeps_source_operation_separate_from_canonical_name(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "K",
        "mode": "single_valley",
        "active_valleys": ["K1"],
        "spin_convention": "spin_up_projected",
        "allowed_internal_symmetries": ["C3z", "C2T"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "operations": [
            {
                **_source_meta(),
                "name": "C3z",
                "operation": "C3",
                "matrix_file": "C3_low_raw.npy",
                "antiunitary": False,
                "k_map": {"type": "rotation", "angle_deg": 120.0},
                "q_map": {"type": "rotation", "angle_deg": 120.0},
                "sector_map": "identity",
            },
            {
                **_source_meta(antiunitary=True),
                "name": "C2T",
                "operation": "C2T",
                "matrix_file": "C2T_low_raw.npy",
                "antiunitary": True,
                "k_map": {"type": "reflection", "axis_deg": 180.0},
                "q_map": {"type": "reflection", "axis_deg": 180.0},
                "sector_map": "identity",
            },
        ],
    }
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C3z"}, {"name": "C2T"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.valley_model["valley_type"] == "K"


def test_K_single_valley_C2T_requires_kp_symm_output(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "K",
        "mode": "single_valley",
        "active_valleys": ["K1"],
        "spin_convention": "spin_up_projected",
        "allowed_internal_symmetries": ["C3z", "C2T"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True, "basis_template": "Gamma_four_orbital"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2T"}],
        "intra": [{"name": "C3z"}, {"name": "C2T"}],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="C2T template generator is not supported"):
        load_model_config(cfg_path)


def test_generic_monomial_constraints_reproduce_legacy_kinetic_orders() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 6, "intra": 0, "inter": 0},
        symmetry_map={"Kinect": []},
        term_templates=[
            {
                "name": "generic_kinetic_layer1",
                "source": "diagonal_kp",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 6,
                "monomial_constraints": {
                    "exclude_m_sum_zero": True,
                    "difference_mod": 3,
                    "difference_residue": 0,
                    "require_mz_ge_mz_star": True,
                },
            }
        ],
    )

    model = build_model(cfg)
    orders = sorted((term.key.Mz, term.key.Mz_star) for term in model.terms.values())

    assert orders == [(1, 1), (2, 2), (3, 0), (3, 3), (4, 1), (6, 0)]


def test_M_spinless_uses_physical_T_with_effective_metadata(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "M",
        "mode": "single_valley",
        "active_valleys": ["M1"],
        "spin_convention": "spinless_effective",
        "allowed_internal_symmetries": ["TR_eff", "C2_eff", "C2TR_eff"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "operations": []}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "TR"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)
    assert cfg.valley_model["spin_convention"] == "spinless_effective"
    assert cfg.valley_model["allowed_internal_symmetries"] == ["TR", "C2", "C2T"]
    assert cfg.symmetry_map["Kinect"][0]["name"] == "TR"
    assert cfg.symmetry_map["Kinect"][0]["representation_level"] == "effective_single_spin"
    assert cfg.symmetry_map["Kinect"][0]["effective_name"] == "TR_eff"


def test_M_spinless_legacy_effective_C2_alias_normalizes_to_physical_name(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "M",
        "mode": "single_valley",
        "active_valleys": ["M1"],
        "spin_convention": "spinless_effective",
        "allowed_internal_symmetries": ["TR_eff", "C2_eff"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "operations": []}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2"}, {"name": "TR_eff"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.valley_model["allowed_internal_symmetries"] == ["TR", "C2"]
    assert cfg.symmetry_map["Kinect"][0]["name"] == "C2"
    assert cfg.symmetry_map["Kinect"][0]["effective_name"] == "C2_eff"
    assert cfg.symmetry_map["Kinect"][1]["name"] == "TR"
    assert cfg.symmetry_map["Kinect"][1]["operation_alias"] == "TR_eff"


def test_Gamma_user_facing_C2_stays_standard_family(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["allowed_internal_symmetries"] = ["TR", "C2"]
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "operations": []}
    raw["model"]["n_orb"] = [2, 2]
    raw["model"]["nlow_state"] = [2, 2]
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "TR"}, {"name": "C2"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.valley_model["allowed_internal_symmetries"] == ["TR", "C2"]
    assert cfg.symmetry_map["Kinect"][1]["name"] == "C2"


def test_monomial_filter_is_forbidden_in_model_schema(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["term_templates"] = [
        {
            "name": "bad_legacy_filter",
            "source": "diagonal_kp",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": [[1, 1]],
            "max_order": 4,
            "monomial_filter": "old_kinetic_filter",
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="not supported by the model schema"):
        load_model_config(cfg_path)


def test_model_schema_rejects_unknown_generation_mode(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["term_templates"] = [
        {
            "name": "kinetic_layer1",
            "source": "diagonal_kp",
            "generation_mode": "unsupported_mode",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": [[1, 1]],
            "max_order": 4,
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="generation_mode must be one of"):
        load_model_config(cfg_path)


def test_model_schema_rejects_legacy_term_names(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["term_templates"] = [
        {
            "name": "legacy_intra_layer1",
            "source": "moire_potential",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": "all",
            "harmonics": {"kind": "intra", "indices": [1]},
            "max_order": 0,
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy-style name"):
        load_model_config(cfg_path)


def test_M_spinless_runtime_supports_physical_names_for_effective_representation() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    sym = SymmetryGenerator(q, q, [1, 1], basis_template="M_spinless_layer_exchange")

    t_eff = sym.get_operator("TR", None)
    c2_eff = sym.get_operator("C2", None)
    c2t_eff = sym.get_operator("C2T", None)

    assert t_eff.shape == (2, 2)
    assert c2_eff.shape == (2, 2)
    assert c2t_eff.shape == (2, 2)
    np.testing.assert_allclose(t_eff @ t_eff.conj(), np.eye(2), atol=1.0e-12)


def test_M_spinless_effective_representation_uses_physical_k_actions() -> None:
    k = np.array([0.25, -0.5], dtype=float)

    np.testing.assert_allclose(
        ContinuumModelBuilder._apply_k_map_to_vector(k, {"name": "TR", "k_map": {"type": "negation"}}),
        np.array([-0.25, 0.5]),
    )
    np.testing.assert_allclose(
        ContinuumModelBuilder._apply_k_map_to_vector(k, {"name": "C2", "k_map": {"type": "reflection", "axis_deg": 0.0}}),
        np.array([0.25, 0.5]),
    )
    np.testing.assert_allclose(
        ContinuumModelBuilder._apply_k_map_to_vector(k, {"name": "C2T", "k_map": {"type": "reflection", "axis_deg": 90.0}}),
        np.array([-0.25, -0.5]),
    )


def test_k_action_requires_explicit_metadata() -> None:
    with pytest.raises(ValueError, match="requires explicit k_map"):
        ContinuumModelBuilder._apply_k_map_to_vector(np.array([0.25, -0.5]), {"name": "C2_eff"})


def test_bM_tmat_default(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {"source": "auto"}
    raw["kpath"] = {
        "file": "KPATH.in",
        "tmat": [
            [60.9508659523, 0.0, 0.0],
            [-30.4754329751, 52.7849982976, 0.0],
            [0.0, 0.0, 50.0],
        ],
    }
    (tmp_path / "KPATH.in").write_text(
        "\n".join(["K-Path Generated by VASPKIT.", "1", "Line-Mode", "Reciprocal", "0 0 0 G", "0.5 0 0 M"]),
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    _, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.bM_diagnostics["source"] == "tmat"
    assert "comparison_with_tmat" in model_cfg.bM_diagnostics


def test_missing_bM_defaults_to_q_distance_even_with_kpath_tmat(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"].pop("bM", None)
    raw["kpath"] = {
        "file": "KPATH.in",
        "tmat": [
            [60.9508659523, 0.0, 0.0],
            [-30.4754329751, 52.7849982976, 0.0],
            [0.0, 0.0, 50.0],
        ],
    }
    (tmp_path / "KPATH.in").write_text(
        "\n".join(["K-Path Generated by VASPKIT.", "1", "Line-Mode", "Reciprocal", "0 0 0 G", "0.5 0 0 M"]),
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    _, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.bM_config == {}
    assert model_cfg.bM_diagnostics["source"] == "q_distance"
    assert "comparison_with_tmat" in model_cfg.bM_diagnostics


def test_explicit_bM_takes_precedence_over_tmat_auto(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]}
    raw["kpath"] = {
        "file": "KPATH.in",
        "tmat": [
            [60.9508659523, 0.0, 0.0],
            [-30.4754329751, 52.7849982976, 0.0],
            [0.0, 0.0, 50.0],
        ],
    }
    (tmp_path / "KPATH.in").write_text(
        "\n".join(["K-Path Generated by VASPKIT.", "1", "Line-Mode", "Reciprocal", "0 0 0 G", "0.5 0 0 M"]),
        encoding="utf-8",
    )
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    _, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.bM_diagnostics["source"] == "explicit"
    np.testing.assert_allclose(model_cfg.bM_diagnostics["b1"], [1.0, 0.0])
    np.testing.assert_allclose(model_cfg.bM_diagnostics["b2"], [0.0, 1.0])


def test_bM_q_distance_requires_validation(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    q_bad = np.array([[0.0, 0.0], [1.0, 0.0], [0.37, 0.91]], dtype=float)
    np.save(tmp_path / "q1.npy", q_bad)
    np.save(tmp_path / "q2.npy", q_bad)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["bM"] = {
        "source": "q_distance",
        "validation": {"max_error": 1.0e-12, "max_failure_fraction": 0.0},
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="q_distance bM reconstruction validation failed"):
        build_moire_config_from_file(cfg_path)


def test_inter_harmonics_include_sector_offsets(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = "K"
    raw["valley_model"]["active_valleys"] = ["K1"]
    raw["sectors"] = [
        {"name": "K1_L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
        {"name": "K1_L2", "qset": "qset2", "q_offset": [0.2, 0.0], "n_orb": 1},
    ]
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.5, 0.8660254037844386]}
    raw["model"]["harmonics"] = {"intra": 1, "inter": {"count": 1, "sector_pairs": [["K1_L2", "K1_L1"]]}}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    np.testing.assert_allclose(moire_cfg.inter_harmonics_map[1], [0.2, 0.0], atol=1.0e-12)
    selected = model_cfg.harmonics_diagnostics["inter"]["selected"][0]
    assert selected["sector_pair"] == ["K1_L2", "K1_L1"]
    assert selected["support_count"] >= 1


def test_k_inter_auto_harmonics_infers_sector_offsets(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = "K"
    raw["valley_model"]["active_valleys"] = ["K1"]
    raw["model"]["bM"] = {"bM1": [1.0, 0.0], "bM2": [0.5, 0.8660254037844386]}
    raw["model"]["harmonics"] = {"intra": 1, "inter": {"count": 1}}
    raw.pop("sectors", None)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    selected = model_cfg.harmonics_diagnostics["inter"]["selected"][0]
    np.testing.assert_allclose(moire_cfg.inter_harmonics_map[1], selected["vector"], atol=1.0e-12)
    assert model_cfg.harmonics_diagnostics["inter"]["generation"] == "valley_geometry_orbit"


def test_build_terms_no_layer1_only() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=2,
        nlow_state=[1, 2],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 1, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"Kinect": [], "Onsite": [], "intra": [], "inter": []},
        term_templates=[
            {"name": "kinetic", "source": "diagonal_kp", "sector_pairs": "same", "orbital_pairs": "diagonal", "max_order": 1},
            {"name": "intra", "source": "moire_potential", "sector_pairs": "same", "orbital_pairs": "all", "harmonics": "intra"},
            {"name": "inter", "source": "tunneling", "sector_pairs": [[2, 1], [1, 2]], "orbital_pairs": "all", "harmonics": "inter"},
        ],
    )

    model = build_model(cfg)
    keys = list(model.terms)
    terms = list(model.terms.values())

    assert any(term.key.layer_from == 2 and term.key.layer_to == 2 and term.tag == "Kinect" for term in terms)
    assert any(key.layer_from == 2 and key.layer_to == 2 for key in keys)
    assert any(key.layer_from == 1 and key.layer_to == 2 for key in keys)
    assert any(key.layer_from == 2 and key.layer_to == 1 for key in keys)


def test_term_templates_accept_named_sector_pairs() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 1, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"inter": []},
        sectors=[
            {"name": "bottom", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
            {"name": "top", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
        ],
        term_templates=[
            {
                "name": "named_inter",
                "source": "tunneling",
                "sector_pairs": [["top", "bottom"]],
                "orbital_pairs": "all",
                "harmonics": {"kind": "inter", "indices": [1]},
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)
    keys = sorted((term.key.layer_from, term.key.layer_to) for term in model.terms.values())

    assert keys == [(2, 1)]


def test_compute_coefficients_skips_empty_orthogonalized_subgroup(monkeypatch) -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        intra_harmonics_map={1: np.array([0.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 0, "intra": 0, "inter": 0},
        symmetry_map={"intra": []},
        term_templates=[
            {"name": "intra", "source": "moire_potential", "sector_pairs": [[1, 1]], "orbital_pairs": "all", "harmonics": "intra"}
        ],
    )
    model = build_model(cfg)
    builder = model._moire_builder
    heff = np.eye(2, dtype=complex)
    k_points = np.array([[0.0, 0.0]])

    def empty_solver(self, initial_vectors, target_vector, *, tol=1.0e-6):
        return np.array([], dtype=float), np.array([], dtype=int)

    monkeypatch.setattr(ContinuumModelBuilder, "_solve_coefficients_from_support_matrix", empty_solver)

    diagnostics = builder.compute_coefficients_by_tag(heff, k_points)

    assert diagnostics["intra"][(1, 1, 1, 1)].size == 0
    assert all(term.r_value_real == 0.0 and term.r_value_imag == 0.0 for term in model.terms.values())


def test_physical_tr_closed_fit_group_merges_orbital_partners() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    tr_block = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    tr_matrix = np.zeros((4, 4), dtype=complex)
    tr_matrix[:2, :2] = tr_block
    tr_matrix[2:, 2:] = tr_block

    class StaticGenerator:
        def get_operator(self, name, params=None):
            assert name == "TR"
            return tr_matrix

    tr_op = {
        "name": "TR",
        "antiunitary": True,
        "k_map": {"type": "negation"},
        "q_map": {"type": "negation"},
        "sector_map": "identity",
        "operation_physics_level": "physical",
    }
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=2,
        n_orb2=2,
        nlow_state=[2, 2],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        intra_harmonics_map={},
        inter_harmonics_map={},
        max_order={"Onsite": 0},
        symmetry_map={"Onsite": [tr_op]},
        symmetry_gen=StaticGenerator(),
        term_templates=[
            {
                "name": "onsite_bottom",
                "source": "onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            }
        ],
    )
    model = build_model(cfg)
    builder = model._moire_builder

    diagnostics = builder.compute_coefficients_by_tag(np.diag([2.0, 2.0, 0.0, 0.0]).astype(complex), np.array([[0.0, 0.0]]))

    onsite_terms = [term for term in model.terms.values() if term.tag == "Onsite"]
    assert len(onsite_terms) == 2
    assert sum(term.active for term in onsite_terms) == 1
    assert len(diagnostics["Onsite"]) == 1


def test_fit_uses_symmetry_closed_block_not_raw_subgroup(monkeypatch) -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=2,
        n_orb2=2,
        nlow_state=[2, 2],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        intra_harmonics_map={},
        inter_harmonics_map={},
        max_order={"Onsite": 0},
        symmetry_map={"Onsite": []},
        term_templates=[
            {
                "name": "onsite_bottom",
                "source": "onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            }
        ],
    )
    model = build_model(cfg)
    builder = model._moire_builder
    calls = []

    def keep_all(self, keys, k_points, *, tol, max_exact_group_size=16, **_kwargs):
        return keys

    def same_fit_block(self, key):
        return (0, 1), (0, 1)

    def same_fit_block_indices(self, key):
        return np.array([0, 1], dtype=int), np.array([0, 1], dtype=int)

    def one_support_component(self, keys, k_points, *, tol, term_matrix_cache=None):
        return [(list(keys), np.array([0, 3], dtype=int))]

    def empty_support_vectors(self, sub_keys, k_points, *, support_idx, **_kwargs):
        calls.append(tuple((key.layer_from, key.layer_to, key.orbital_from, key.orbital_to) for key in sub_keys))
        return np.zeros((2 * len(sub_keys), len(support_idx)), dtype=complex)

    def empty_solver(self, initial_vectors, target_vector, *, tol=1.0e-6):
        return np.array([], dtype=float), np.array([], dtype=int)

    monkeypatch.setattr(ContinuumModelBuilder, "_filter_duplicate_symmetry_seed_keys", keep_all)
    monkeypatch.setattr(ContinuumModelBuilder, "_fit_block_signature_for_key", same_fit_block)
    monkeypatch.setattr(ContinuumModelBuilder, "_fit_block_indices_for_key", same_fit_block_indices)
    monkeypatch.setattr(ContinuumModelBuilder, "_support_connected_components_with_support_for_keys", one_support_component)
    monkeypatch.setattr(ContinuumModelBuilder, "_initialterm_support_vectors_for_fit_keys", empty_support_vectors)
    monkeypatch.setattr(ContinuumModelBuilder, "_solve_coefficients_from_support_matrix", empty_solver)

    diagnostics = builder.compute_coefficients_by_tag(np.eye(4, dtype=complex), np.array([[0.0, 0.0]]))

    assert calls == [((1, 1, 1, 1), (1, 1, 2, 2))]
    assert len(diagnostics["Onsite"]) == 1


def test_fit_reuses_stacked_term_matrices_between_duplicate_filter_and_orthogonalization(monkeypatch) -> None:
    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.zeros((0, 2), dtype=float)
    k_points = np.array([[0.0, 0.0], [0.25, 0.0]], dtype=float)
    builder = ContinuumModelBuilder(
        q1,
        q2,
        1,
        0,
        np.array([1.0, 0.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        {},
        {},
        {"intra": 1},
        SymmetryGenerator(q1, q2, [1, 0]),
        {"intra": []},
    )
    for mz in (0, 1):
        key = ContinuumTermKey(mz, 0, 1, 1, 1, 1, (0.0, 0.0))
        builder.model.add_term(
            key,
            ContinuumModelBuilder.make_Y_basis_function(key, q1, q2, 1, 0),
            tag="intra",
            symmetry_ops=[],
        )

    model_core.clear_symmetry_caches()
    original = ContinuumModelBuilder.symmetrize_Y_and_iY_basis_static
    calls: list[tuple[ContinuumTermKey, tuple[float, float]]] = []

    def counted_symmetrize(Y_basis, k, sym_ops, symmetry_gen=None, term=None, use_cache=False, **kwargs):
        calls.append((term.key, tuple(float(x) for x in k)))
        return original(Y_basis, k, sym_ops, symmetry_gen=symmetry_gen, term=term, use_cache=use_cache, **kwargs)

    monkeypatch.setattr(
        ContinuumModelBuilder,
        "symmetrize_Y_and_iY_basis_static",
        staticmethod(counted_symmetrize),
    )

    builder.compute_coefficients_by_tag(np.diag([1.0, 1.25]).astype(complex), k_points, tol=1.0e-10)

    assert len(calls) == len(builder.model.terms) * len(k_points)


def test_joint_fit_does_not_double_count_duplicate_diagonal_terms_across_tags() -> None:
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    q2 = np.zeros((0, 2), dtype=float)
    k_points = np.array([[0.0, 0.0]], dtype=float)
    target = np.diag([1.0, -1.0]).astype(complex)
    builder = ContinuumModelBuilder(
        q1,
        q2,
        1,
        0,
        np.array([1.0, 0.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        {},
        {},
        {"Kinect": 0, "intra": 0},
        SymmetryGenerator(q1, q2, [1, 0]),
        {"Kinect": [], "intra": []},
    )

    for tag, key in [
        ("Kinect", ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))),
        ("intra", ContinuumTermKey(1, 1, 1, 1, 1, 1, (0.0, 0.0))),
    ]:
        builder.model.add_term(key, lambda _k, matrix=target: matrix.copy(), tag=tag, symmetry_ops=[])

    builder.compute_coefficients_by_tag(target, k_points, tol=1.0e-12)

    h_model = builder.model.assemble_hamiltonian(k_points[0], builder.symmetry_gen)
    np.testing.assert_allclose(h_model, target, atol=1.0e-12)
    coeff_sum = sum(float(term.r_value_real) for term in builder.model.terms.values())
    assert coeff_sum == pytest.approx(1.0, abs=1.0e-12)


def test_coefficients_and_term_registry_written(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
        return {
            "eigvals": expected_eigvals,
            "diagnostics": {"fit_rank": 1},
            "model": type("Model", (), {"terms": {key: type("Term", (), {"key": key, "tag": "Kinect", "active": True, "r_value_real": 1.0, "r_value_imag": 0.0, "symmetry_ops": []})()}})(),
        }

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    run_configured_model(cfg_path)

    assert (tmp_path / "model_out" / "run_summary.json").exists()
    assert (tmp_path / "model_out" / "active_terms.sha256").exists()
    assert not (tmp_path / "model_out" / "coefficients.json").exists()
    assert not (tmp_path / "model_out" / "terms.json").exists()
    assert not (tmp_path / "model_out" / "fit_diagnostics.json").exists()
    assert not (tmp_path / "model_out" / "operation_registry.json").exists()

    active_terms_bytes = (tmp_path / "model_out" / "active_terms.json").read_bytes()
    active_terms_hash = (tmp_path / "model_out" / "active_terms.sha256").read_text(encoding="utf-8").strip()
    summary = json.loads((tmp_path / "model_out" / "run_summary.json").read_text(encoding="utf-8"))
    assert active_terms_hash == hashlib.sha256(active_terms_bytes).hexdigest()
    assert summary["active_terms_hash"] == active_terms_hash
    assert summary["term_template_profile"]["input_kind"] == "default"
    assert "production" + "_level" not in summary
    assert "validation_incomplete" in summary
    assert all("user_operation" in row and "canonical_operation" in row for row in summary["operations"])


def test_release_output_profile_serializes_only_active_terms(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")
    active_key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
    inactive_key = ContinuumTermKey(1, 0, 1, 1, 1, 1, (0.0, 0.0))

    def fake_term(key, *, active: bool):
        return type(
            "Term",
            (),
            {
                "key": key,
                "tag": "Kinect",
                "active": active,
                "r_value_real": 1.0 if active else 0.0,
                "r_value_imag": 0.0,
                "symmetry_ops": [],
            },
        )()

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {
            "eigvals": expected_eigvals,
            "diagnostics": {"fit_rank": 1},
            "model": type(
                "Model",
                (),
                {
                    "terms": {
                        active_key: fake_term(active_key, active=True),
                        inactive_key: fake_term(inactive_key, active=False),
                    }
                },
            )(),
        }

    seen_keys = []
    original_term_to_dict = pipeline_module._term_to_release_dict

    def spy_term_to_dict(term):
        seen_keys.append(term.key)
        return original_term_to_dict(term)

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)
    monkeypatch.setattr(pipeline_module, "_term_to_release_dict", spy_term_to_dict)

    run_configured_model(cfg_path)

    assert seen_keys == [active_key]
    active_terms = json.loads((tmp_path / "model_out" / "active_terms.json").read_text(encoding="utf-8"))
    assert len(active_terms) == 1
    assert active_terms[0]["active"] is True


def test_release_active_terms_use_compact_symmetry_ops(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")
    key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
    heavy_symmetry = [
        {
            "name": "C3z",
            "antiunitary": False,
            "k_map": {"type": "rotation", "angle_deg": 120.0},
            "q_map": {"type": "rotation", "angle_deg": 120.0},
            "sector_map": "identity",
            "internal_resolved_action": {"k_map": {"type": "rotation", "angle_deg": 120.0}},
            "model_basis_action": {"items": [{"source_q_index": i, "target_q_index": i} for i in range(50)]},
            "source_matrix_projection_report": {"items": [{"residual": float(i)} for i in range(50)]},
        }
    ]

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {
            "eigvals": expected_eigvals,
            "diagnostics": {"fit_rank": 1},
            "model": type(
                "Model",
                (),
                {
                    "terms": {
                        key: type(
                            "Term",
                            (),
                            {
                                "key": key,
                                "tag": "Kinect",
                                "active": True,
                                "r_value_real": 1.0,
                                "r_value_imag": 0.0,
                                "symmetry_ops": heavy_symmetry,
                                "registry_metadata": {},
                            },
                        )()
                    }
                },
            )(),
        }

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    run_configured_model(cfg_path)

    active_terms = json.loads((tmp_path / "model_out" / "active_terms.json").read_text(encoding="utf-8"))
    op = active_terms[0]["symmetry_ops"][0]
    assert op["name"] == "C3z"
    assert op["k_map"] == {"type": "rotation", "angle_deg": 120.0}
    assert "model_basis_action" not in op
    assert "source_matrix_projection_report" not in op


def test_debug_output_profile_writes_diagnostics_subdir(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "debug"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
        return {
            "eigvals": expected_eigvals,
            "diagnostics": {"fit_rank": 1},
            "model": type("Model", (), {"terms": {key: type("Term", (), {"key": key, "tag": "Kinect", "active": True, "r_value_real": 1.0, "r_value_imag": 0.0, "symmetry_ops": []})()}})(),
        }

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    run_configured_model(cfg_path)

    diagnostics = tmp_path / "model_out" / "diagnostics"
    assert (diagnostics / "coefficients.json").exists()
    assert (diagnostics / "terms.json").exists()
    assert (diagnostics / "fit_diagnostics.json").exists()
    assert (diagnostics / "operation_registry.json").exists()
    assert (diagnostics / "comparison.json").exists()
    assert (diagnostics / "comparison_plot.json").exists()
    assert (diagnostics / "bM_diagnostic.json").exists()
    assert (diagnostics / "harmonics_diagnostic.json").exists()
    assert not (tmp_path / "model_out" / "coefficients.json").exists()


def test_run_configured_model_quiet_writes_detailed_log(monkeypatch, capsys, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")
    fake_model = type("Model", (), {"terms": {}})()

    def fake_build_model(moire_config):
        print("very noisy coefficient dump")
        return fake_model

    def fake_compute_coefficients(moire_config, model):
        print("very noisy fit dump")
        return model, {}

    def fake_compute_bands(moire_config, model, kpoints, return_eigvecs=False):
        print("very noisy coefficient dump")
        return expected_eigvals

    monkeypatch.setattr("kp.model.pipeline.build_model", fake_build_model)
    monkeypatch.setattr("kp.model.pipeline.compute_coefficients", fake_compute_coefficients)
    monkeypatch.setattr("kp.model.pipeline.compute_bands", fake_compute_bands)

    results = run_configured_model(cfg_path)
    captured = capsys.readouterr()

    assert "very noisy coefficient dump" not in captured.out
    assert "[kp model] building continuum terms" in captured.out
    log_path = Path(results["model_log"])
    assert log_path.exists()
    assert "very noisy coefficient dump" in log_path.read_text(encoding="utf-8")
    assert "very noisy fit dump" in log_path.read_text(encoding="utf-8")


def test_matrix_residual_not_only_band() -> None:
    heff = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=complex)
    same_bands_different_matrix = np.array([[[1.5, 0.5], [0.5, 1.5]]], dtype=complex)

    band_metrics = compare_bands(np.linalg.eigvalsh(same_bands_different_matrix), np.linalg.eigvalsh(heff))
    residual = matrix_residual(same_bands_different_matrix, heff)

    assert band_metrics["max_abs_error"] < 1.0e-12
    assert residual["max_residual"] > 0.1


def test_symmetry_from_kp_symm_output(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C3_low_raw.npy", matrix)
    (tmp_path / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120},
                        "q_map": {"type": "rotation", "angle_deg": 120},
                        "sector_map": "identity",
                        "spin_map": "identity",
                        "valley_map": "identity",
                        "residual": 0.0,
                        "leakage": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "operations": ["C3z"]}, base=tmp_path, expected_dim=2)

    np.testing.assert_allclose(source.generator.get_operator("C3z", 1), matrix)
    assert source.metadata["operations"][0]["name"] == "C3z"
    assert source.source_type == "kp_symm_output"


def test_symmetry_source_infers_operations_from_summary(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C3_low_raw.npy", matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "operation": "C3",
                        "matrix_file": "C3_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 1.2e-11,
                                    "subspace_leakage": 3.4e-9,
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)

    np.testing.assert_allclose(source.generator.get_operator("C3z", 1), matrix)
    op = source.metadata["operations"][0]
    assert op["name"] == "C3z"
    assert op["matrix_file"].endswith("C3_low_raw.npy")
    assert op["residual"] == 1.2e-11
    assert op["leakage"] == 3.4e-9


def test_symmetry_source_loads_m_effective_aliases_under_physical_keys(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    t_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    c2_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    np.save(tmp_path / "TR_low_raw.npy", t_matrix)
    np.save(tmp_path / "C2_low_raw.npy", c2_matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "requires_model_exactification": True,
                "operations": [
                    {
                        **_source_meta(antiunitary=True),
                        "name": "TR_eff",
                        "operation": "TR",
                        "matrix_file": "TR_low_raw.npy",
                        "antiunitary": True,
                        "k_map": {"type": "negation"},
                        "q_map": {"type": "negation"},
                        "sector_map": "identity",
                        "action_source": "raw_h_operator_file",
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 1.0e-12,
                                    "subspace_leakage": 2.0e-12,
                                }
                            }
                        ],
                    },
                    {
                        **_source_meta(),
                        "name": "C2_eff",
                        "operation": "C2",
                        "source_matrix_role": "D0_times_PG_action",
                        "gauge_correction": {"kind": "periodic_atomic_bloch_PG", "file": "representations/M1/C2_PG.npz"},
                        "matrix_file": "C2_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "sector_map": "layer_exchange",
                        "action_source": "representation_file+pg_file",
                        "pg_file": "representations/M1/C2_PG.npz",
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 3.0e-12,
                                    "subspace_leakage": 4.0e-12,
                                }
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    source = load_symmetry_source(
        {
            "type": "kp_symm_output",
            "path": str(tmp_path),
            "use": "raw",
            "matrix_kind": "action",
            "operations": [
                {"name": "TR_eff", "operation": "TR"},
                {"name": "C2_eff", "operation": "C2"},
            ],
                    },
        base=tmp_path,
        expected_dim=2,
    )

    np.testing.assert_allclose(source.generator.get_operator("TR"), t_matrix)
    with pytest.raises(ValueError, match="not loaded"):
        source.generator.get_operator("TR_eff")
    np.testing.assert_allclose(source.generator.get_operator("C2"), c2_matrix)
    with pytest.raises(ValueError, match="not loaded"):
        source.generator.get_operator("C2_eff")
    by_name = {op["name"]: op for op in source.metadata["operations"]}

    t_op = by_name["TR"]
    assert t_op["operation"] == "TR"
    assert t_op["operation_alias"] == "TR_eff"
    assert t_op["canonical_physical_operation"] == "TR"
    assert t_op["representation_level"] == "effective_single_spin"
    assert t_op["effective_name"] == "TR_eff"
    assert t_op["approximation"] == {"kind": "spin_SU2_effective_block"}
    assert t_op["matrix_file"].endswith("TR_low_raw.npy")
    assert t_op["matrix_kind"] == "action"
    assert t_op["source_matrix_role"] == "raw_h_sewing_action"
    assert t_op["target_role"] == "continuum_internal_rep"
    assert t_op["antiunitary_convention"] == "U_K"
    assert "production" + "_use" not in t_op

    c2_op = by_name["C2"]
    assert c2_op["operation"] == "C2"
    assert c2_op["operation_alias"] == "C2_eff"
    assert c2_op["canonical_physical_operation"] == "C2"
    assert c2_op["representation_level"] == "effective_single_spin"
    assert c2_op["effective_name"] == "C2_eff"
    assert c2_op["matrix_file"].endswith("C2_low_raw.npy")
    assert c2_op["source_matrix_role"] == "D0_times_PG_action"
    assert c2_op["gauge_correction"] == {"kind": "periodic_atomic_bloch_PG", "file": "representations/M1/C2_PG.npz"}


def test_symmetry_source_requires_explicit_k_map_instead_of_axis_deg(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C2_low_raw.npy", matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "requires_model_exactification": True,
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C2",
                        "operation": "C2",
                        "matrix_file": "C2_low_raw.npy",
                        "axis_deg": 150.0,
                        "antiunitary": False,
                        "pairs": [
                            {
                                "raw": {
                                    "heff_covariance_residual": 0.0,
                                    "subspace_leakage": 0.0,
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="k_map"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)


def test_symmetry_source_rejects_representation_low_matrix_as_production_source(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    action_matrix = np.eye(2, dtype=complex)
    representation_matrix = -np.eye(2, dtype=complex)
    np.save(tmp_path / "C2_low_raw.npy", action_matrix)
    np.save(tmp_path / "C2_low_representation_raw.npy", representation_matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {
                        **{k: v for k, v in _source_meta(representation=True).items() if k != "matrix_kind"},
                        "name": "C2",
                        "operation": "C2",
                        "matrix_file": "C2_low_representation_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "sector_map": "layer_exchange",
                        "antiunitary": False,
                        "action_source": "raw_h_operator_file",
                        "pairs": [{"raw": {"heff_covariance_residual": 0.0, "subspace_leakage": 0.0}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="production symmetry matrices must use raw-action or kp_symm exactified matrices"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path), "use": "raw", "matrix_kind": "representation"},
            base=tmp_path,
            expected_dim=2,
        )


def test_symmetry_source_rejects_manifest_default_representation_matrix_kind(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    np.save(tmp_path / "C2_low_representation_raw.npy", -np.eye(2, dtype=complex))
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "default_matrix_kind": "representation",
                "operations": [
                    {
                        **{k: v for k, v in _source_meta(representation=True).items() if k != "matrix_kind"},
                        "name": "C2",
                        "operation": "C2",
                        "matrix_file": "C2_low_representation_raw.npy",
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "sector_map": "layer_exchange",
                        "antiunitary": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matrix_kind"):
        load_symmetry_source({"type": "kp_symm_output", "path": str(tmp_path), "use": "raw"}, base=tmp_path, expected_dim=2)


def test_symmetry_source_rejects_invalid_representation_by_default(tmp_path: Path) -> None:
    from kp.model.symmetry import load_symmetry_source

    matrix = np.eye(2, dtype=complex)
    np.save(tmp_path / "C3_low_representation_raw.npy", matrix)
    (tmp_path / "summary.json").write_text(
        yaml.safe_dump(
            {
                "operations": [
                        {
                            **_source_meta(representation=True),
                            "name": "C3z",
                            "operation": "C3",
                            "matrix_file": "C3_low_representation_raw.npy",
                            "antiunitary": False,
                            "k_map": {"type": "rotation", "angle_deg": 120.0},
                            "q_map": {"type": "rotation", "angle_deg": 120.0},
                            "sector_map": "identity",
                            "representation_pairs": [
                            {
                                "quality_warnings": ["singular values deviate from 1 by 4.476e-01"],
                                "raw": {
                                    "heff_covariance_residual": 0.0,
                                    "subspace_leakage": 0.4476,
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="production symmetry matrices must use raw-action or kp_symm exactified matrices"):
        load_symmetry_source(
            {"type": "kp_symm_output", "path": str(tmp_path), "use": "raw", "matrix_kind": "representation"},
            base=tmp_path,
            expected_dim=2,
        )


def test_build_moire_config_enriches_symmetry_map_with_rotated_k_map(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "action"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2"}],
        "Onsite": [],
        "intra": [],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    v = np.array([np.sqrt(3.0) / 2.0, -0.5], dtype=float)
    np.save(tmp_path / "q1.npy", np.array([-v, v], dtype=float))
    np.save(tmp_path / "q2.npy", np.array([-v, v], dtype=float))

    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    np.save(symm_dir / "exactified_C2.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "exactification_owner": "kp_symm",
                "frame": {"q_transform": {"rotation_deg": 30.0}},
                "requires_model_exactification": False,
                "operations": [
                        {
                            **_source_meta(),
                            "name": "C2",
                            "matrix_file": "exactified_C2.npy",
                            "matrix_kind": "continuum_internal_rep_exact",
                            "matrix_source": "kp_symm_exactified_action",
                            "k_map": {"type": "reflection", "axis_deg": 150.0},
                            "q_map": {"type": "reflection", "axis_deg": 150.0},
                            "antiunitary": False,
                            "sector_map": "identity",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    op = moire_cfg.symmetry_map["Kinect"][0]
    assert model_cfg.symmetry_map["Kinect"][0]["k_map"]["type"] == "reflection"
    assert op["k_map"]["axis_deg"] == pytest.approx(180.0)


def test_build_moire_config_prefers_model_frame_action_from_symm_artifact(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "action"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2"}],
        "Onsite": [],
        "intra": [],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    v = np.array([np.sqrt(3.0) / 2.0, -0.5], dtype=float)
    np.save(tmp_path / "q1.npy", np.array([-v, v], dtype=float))
    np.save(tmp_path / "q2.npy", np.array([-v, v], dtype=float))

    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    np.save(symm_dir / "exactified_C2.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "exactification_owner": "kp_symm",
                "frame": {"q_transform": {"rotation_deg": 30.0}},
                "requires_model_exactification": False,
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C2",
                        "matrix_file": "exactified_C2.npy",
                        "matrix_kind": "continuum_internal_rep_exact",
                        "matrix_source": "kp_symm_exactified_action",
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "model_action": {
                            "antiunitary": False,
                            "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                            "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                            "sector_map": "identity",
                        },
                        "antiunitary": False,
                        "sector_map": "identity",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, _model_cfg = build_moire_config_from_file(cfg_path)

    op = moire_cfg.symmetry_map["Kinect"][0]
    assert op["k_map"]["axis_deg"] == pytest.approx(180.0)
    assert op["q_map"]["axis_deg"] == pytest.approx(180.0)
    assert op["k_map"]["in_model_frame"] is True


def test_build_moire_config_loads_strict_symm_artifact_without_model_exactification(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "action"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2"}],
        "Onsite": [],
        "intra": [],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    v = np.array([np.sqrt(3.0) / 2.0, -0.5], dtype=float)
    np.save(tmp_path / "q1.npy", np.array([-v, v], dtype=float))
    np.save(tmp_path / "q2.npy", np.array([-v, v], dtype=float))

    model_action = {
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        "sector_map": "identity",
    }
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    np.save(symm_dir / "exactified_C2.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "strict_metadata": True,
                "exactification_owner": "kp_symm",
                "frame": {"q_transform": {"rotation_deg": 30.0}},
                "requires_model_exactification": False,
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C2",
                        "matrix_file": "exactified_C2.npy",
                        "matrix_kind": "continuum_internal_rep_exact",
                        "matrix_source": "kp_symm_exactified_action",
                        "k_map": {"type": "reflection", "axis_deg": 150.0},
                        "q_map": {"type": "reflection", "axis_deg": 150.0},
                        "model_action": model_action,
                        "antiunitary": False,
                        "sector_map": "identity",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    operation = model_cfg.symmetry_source_metadata["operations"][0]
    assert "source_matrix_projection_reports" not in model_cfg.symmetry_source_metadata
    assert "source_matrix_projection_report" not in operation
    assert operation["matrix_kind"] == "continuum_internal_rep_exact"
    assert model_cfg.symmetry_map["Kinect"][0]["k_map"]["axis_deg"] == pytest.approx(180.0)
    assert moire_cfg.symmetry_map["Kinect"][0]["k_map"]["axis_deg"] == pytest.approx(180.0)


def test_build_moire_config_reads_kp_symm_projection_report_without_action_mismatch(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "action"}
    raw["model"]["symmetry_map"] = {
        "Kinect": [{"name": "C2T"}],
        "Onsite": [],
        "intra": [],
        "inter": [],
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    manifest_action = {
        "antiunitary": True,
        "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        "sector_map": "layer_exchange",
    }
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    np.save(symm_dir / "exactified_C2T.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "frame": {"q_transform": {"rotation_deg": 0.0}},
                "exactification_owner": "kp_symm",
                "requires_model_exactification": False,
                "operations": [
                    {
                        **_source_meta(antiunitary=True),
                        "name": "C2T",
                        "matrix_file": "exactified_C2T.npy",
                        "matrix_kind": "continuum_internal_rep_exact",
                        "matrix_source": "kp_symm_exactified_action",
                        "k_map": {"type": "reflection", "axis_deg": 60.0},
                        "q_map": {"type": "reflection", "axis_deg": 60.0},
                        "sector_map": "identity",
                        "model_action": manifest_action,
                        "internal_resolved_action": manifest_action,
                        "source_matrix_projection_report": {
                            "report": {"status": "exactified"},
                            "resolved_action": manifest_action,
                            "manifest_model_action": manifest_action,
                            "support_resolution": {
                                "action_mismatch": False,
                                "candidate_source": "manifest_model_action",
                            },
                        },
                        "model_basis_action": {
                            "complete": True,
                            "sector_map": "layer_exchange",
                            "items": [
                                {
                                    "source_sector": "L1",
                                    "source_q_index": 0,
                                    "target_sector": "L2",
                                    "target_q_index": 0,
                                    "q_residual": 0.0,
                                }
                            ],
                        },
                        "antiunitary": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    operation = model_cfg.symmetry_source_metadata["operations"][0]
    report = operation["source_matrix_projection_report"]
    assert report["support_resolution"]["action_mismatch"] is False
    assert operation["internal_resolved_action"]["sector_map"] == "layer_exchange"
    assert "provenance" not in operation["internal_resolved_action"]
    assert model_cfg.symmetry_map["Kinect"][0]["sector_map"] == "layer_exchange"
    assert moire_cfg.symmetry_map["Kinect"][0]["k_map"]["in_model_frame"] is True


def test_action_mismatch_default_does_not_write_internal_resolved_action(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "action"}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2"}], "Onsite": [], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    declared = {
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "sector_map": "identity",
    }
    selected = {
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "sector_map": "layer_exchange",
    }
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    np.save(symm_dir / "exactified_C2.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "frame": {"q_transform": {"rotation_deg": 0.0}},
                "exactification_owner": "kp_symm",
                "requires_model_exactification": False,
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C2",
                        "matrix_file": "exactified_C2.npy",
                        "matrix_kind": "continuum_internal_rep_exact",
                        "matrix_source": "kp_symm_exactified_action",
                        "k_map": declared["k_map"],
                        "q_map": declared["q_map"],
                        "model_action": declared,
                        "source_matrix_projection_report": {
                            "report": {"status": "exactified"},
                            "resolved_action": declared,
                            "manifest_model_action": declared,
                            "selected_action_candidate": selected,
                            "support_resolution": {
                                "action_mismatch": True,
                                "candidate_source": "support_discovery",
                                "declared_model_action": declared,
                                "selected_action_candidate": selected,
                                "declared_support_residual": 0.5,
                                "selected_support_residual": 0.0,
                            },
                        },
                        "antiunitary": False,
                        "sector_map": "identity",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    operation = model_cfg.symmetry_source_metadata["operations"][0]
    assert operation["source_matrix_projection_report"]["support_resolution"]["action_mismatch"] is True
    assert "internal_resolved_action" not in operation
    assert model_cfg.symmetry_map["Kinect"][0]["sector_map"] == "identity"
    assert moire_cfg.symmetry_map["Kinect"][0]["sector_map"] == "identity"


def test_action_mismatch_explicit_accept_writes_internal_resolved_action_with_provenance(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["symmetry_source"] = {"type": "kp_symm_output", "path": "symm", "matrix_kind": "action"}
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2"}], "Onsite": [], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    declared = {
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "sector_map": "identity",
    }
    selected = {
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
        "sector_map": "layer_exchange",
        "provenance": {
            "source": "support_exactification",
            "accepted_by_user": True,
            "declared_model_action": declared,
            "selected_action_candidate": {
                "antiunitary": False,
                "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
                "sector_map": "layer_exchange",
            },
            "declared_support_residual": 0.5,
            "selected_support_residual": 0.0,
        },
    }
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    np.save(symm_dir / "exactified_C2.npy", np.eye(4, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        yaml.safe_dump(
            {
                "frame": {"q_transform": {"rotation_deg": 0.0}},
                "exactification_owner": "kp_symm",
                "requires_model_exactification": False,
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C2",
                        "matrix_file": "exactified_C2.npy",
                        "matrix_kind": "continuum_internal_rep_exact",
                        "matrix_source": "kp_symm_exactified_action",
                        "k_map": declared["k_map"],
                        "q_map": declared["q_map"],
                        "model_action": declared,
                        "internal_resolved_action": selected,
                        "source_matrix_projection_report": {
                            "report": {"status": "exactified"},
                            "resolved_action": selected,
                            "manifest_model_action": declared,
                            "selected_action_candidate": selected,
                            "support_resolution": {
                                "action_mismatch": True,
                                "candidate_source": "support_discovery",
                                "declared_model_action": declared,
                                "selected_action_candidate": selected,
                                "declared_support_residual": 0.5,
                                "selected_support_residual": 0.0,
                                "provenance": selected["provenance"],
                            },
                        },
                        "antiunitary": False,
                        "sector_map": "identity",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    _moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    operation = model_cfg.symmetry_source_metadata["operations"][0]
    assert operation["internal_resolved_action"]["sector_map"] == "layer_exchange"
    assert operation["internal_resolved_action"]["provenance"]["accepted_by_user"] is True


def test_float_p_key_canonicalization() -> None:
    from kp.model.schema import canonical_vector_key

    assert canonical_vector_key([0.1 + 0.2, 0.0], tol=1.0e-9) == canonical_vector_key([0.3, 0.0], tol=1.0e-9)
    assert ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.1 + 0.2, 0.0)) == ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.3, 0.0))


def test_C2T_symmetrization_uses_antiunitary_conjugation() -> None:
    from kp.model.symmetry import MatrixSymmetryGenerator

    generator = MatrixSymmetryGenerator({"C2T": np.eye(2, dtype=complex)}, {"operations": []})
    Y = np.array([[1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 6.0j, 7.0 - 8.0j]], dtype=complex)

    transformed = ContinuumModelBuilder._apply_symmetry_op_to_matrix(Y.copy(), "C2T", None, generator)

    np.testing.assert_allclose(transformed, Y.conj())


def test_full_bilayer_block_detection_accepts_exactified_c2t_action() -> None:
    ops = [
        {
            "name": "C2T",
            "antiunitary": True,
            "sector_map": "layer_exchange",
            "k_map": {"type": "reflection", "axis_deg": 180.0},
        }
    ]

    assert ContinuumModelBuilder._uses_full_bilayer_block(ops) is True


def test_full_bilayer_block_detection_accepts_generic_two_sector_exchange() -> None:
    ops = [
        {
            "name": "TR_eff",
            "antiunitary": True,
            "sector_map": {"bottom": "top", "top": "bottom"},
        }
    ]

    assert ContinuumModelBuilder._uses_full_bilayer_block(ops, ["bottom", "top"]) is True


def test_run_configured_model_saves_auto_harmonics_diagnostic_plot(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    run_configured_model(cfg_path)

    assert not (tmp_path / "model_out" / "harmonics_diagnostic.pdf").exists()
    assert not (tmp_path / "model_out" / "harmonics_diagnostic.json").exists()

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "debug"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    run_configured_model(cfg_path)

    assert (tmp_path / "model_out" / "diagnostics" / "harmonics_diagnostic.pdf").exists()
    assert (tmp_path / "model_out" / "diagnostics" / "harmonics_diagnostic.json").exists()


def test_run_configured_model_saves_outputs_without_legacy_diagnostics_json(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        assert moire_config.output_dir is None
        return {"eigvals": expected_eigvals, "diagnostics": {("tuple", "key"): 1}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    results = run_configured_model(cfg_path)

    assert results["comparison"]["max_abs_error"] == 0.0
    assert results["moire_config"].output_dir == tmp_path / "model_out"
    assert (tmp_path / "model_out" / "eigvals.npy").exists()
    assert not (tmp_path / "model_out" / "comparison.json").exists()
    assert not (tmp_path / "model_out" / "comparison_plot.json").exists()
    assert (tmp_path / "model_out" / "band_comparison.pdf").exists()
    assert results["band_plot"] == str((tmp_path / "model_out" / "band_comparison.pdf").resolve())
    assert (tmp_path / "model_out" / "band_comparison_all.pdf").exists()
    assert results["all_band_plot"] == str((tmp_path / "model_out" / "band_comparison_all.pdf").resolve())
    assert results["all_band_plot_comparison"]["num_bands"] == expected_eigvals.shape[1]
    summary = json.loads((tmp_path / "model_out" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["comparison"]["max_abs_error"] == 0.0
    assert summary["plot_comparison"]["max_abs_error"] == 0.0
    assert summary["all_band_plot"] == "band_comparison_all.pdf"
    assert summary["all_band_plot_comparison"]["num_bands"] == expected_eigvals.shape[1]


def test_compare_bands_reports_rms_and_max_error() -> None:
    model = np.array([[0.0, 1.1, 2.0], [0.0, 2.0, 4.0]])
    heff = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 4.0]])

    metrics = compare_bands(model, heff, band_slice=[0, 2])

    assert metrics["num_kpoints"] == 2
    assert metrics["num_bands"] == 2
    assert metrics["rms_error"] == np.sqrt((0.0**2 + 0.1**2 + 0.0**2 + 0.5**2) / 4)
    assert metrics["max_abs_error"] == 0.5


def test_compare_bands_for_plot_supports_bottom_bands_with_bottom_alignment() -> None:
    model = np.array([[1.2, 2.0, 3.0], [1.4, 2.1, 3.2]])
    heff = np.array([[1.0, 2.0, 3.1], [1.1, 2.2, 3.2]])

    metrics = compare_bands_for_plot(
        model,
        heff,
        plot_config={"bottom_bands": 2, "align": "bottom"},
    )

    assert metrics["band_slice"] == [0, 2]
    assert metrics["align"] == "bottom"
    assert metrics["num_bands"] == 2
    assert metrics["aligned_rms_error_meV"] == pytest.approx(187.08286933869707)
    assert metrics["model_alignment_shift_meV"] == pytest.approx(200.0)


def test_save_band_comparison_plot_can_plot_all_bands_with_top_band_window(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes
    from kp.plot_style import KP_MODEL_STYLE, KP_REFERENCE_STYLE

    model = np.array(
        [
            [-100.0, -1.0, 0.0, 2.0, 3.0],
            [-99.0, -0.8, 0.2, 2.2, 3.2],
        ]
    )
    heff = model + 0.01
    plotted_colors: list[str] = []
    ylims: list[tuple[float, float]] = []

    original_plot = matplotlib.axes.Axes.plot
    original_set_ylim = matplotlib.axes.Axes.set_ylim

    def spy_plot(self, *args, **kwargs):
        color = kwargs.get("color")
        if color in {KP_REFERENCE_STYLE["color"], KP_MODEL_STYLE["color"]}:
            plotted_colors.append(str(color))
        return original_plot(self, *args, **kwargs)

    def spy_set_ylim(self, bottom=None, top=None, *args, **kwargs):
        if bottom is not None and top is not None:
            ylims.append((float(bottom), float(top)))
        return original_set_ylim(self, bottom, top, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", spy_set_ylim)

    save_band_comparison_plot(
        model,
        heff,
        tmp_path / "bands.png",
        plot_config={"top_bands": 2, "plot_all_bands": True},
    )

    assert plotted_colors.count(KP_REFERENCE_STYLE["color"]) == 5
    assert plotted_colors.count(KP_MODEL_STYLE["color"]) == 5
    assert ylims
    assert ylims[-1][0] > 1.0
    assert ylims[-1][1] < 3.5


def test_save_band_comparison_plot_uses_release_style_by_default(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes
    import matplotlib.pyplot as plt

    from kp.plot_style import KP_BAND_BOX_ASPECT, KP_BAND_FIGSIZE, kp_font_family

    model = np.array([[0.0, 0.1], [0.02, 0.12]])
    heff = model + 0.01
    captured: dict[str, object] = {}

    original_subplots = plt.subplots
    original_set_box_aspect = matplotlib.axes.Axes.set_box_aspect

    def spy_subplots(*args, **kwargs):
        captured["figsize"] = kwargs.get("figsize")
        return original_subplots(*args, **kwargs)

    def spy_set_box_aspect(self, aspect=None, *args, **kwargs):
        captured["box_aspect"] = aspect
        return original_set_box_aspect(self, aspect, *args, **kwargs)

    saved_axes = []

    def spy_close(fig=None):
        saved_axes.extend(fig.axes if fig is not None else [])

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_box_aspect", spy_set_box_aspect)
    monkeypatch.setattr(plt, "close", spy_close)

    save_band_comparison_plot(model, heff, tmp_path / "bands.png", plot_config={"align": "top"})

    assert tuple(captured["figsize"]) == KP_BAND_FIGSIZE
    assert round(float(captured["box_aspect"]), 6) == round(KP_BAND_BOX_ASPECT, 6)
    assert saved_axes
    assert saved_axes[0].get_ylabel() == "Energy - E_top (eV)"
    assert "$" not in saved_axes[0].get_ylabel()
    assert saved_axes[0].yaxis.label.get_fontfamily()[0] == kp_font_family()


def test_all_band_plot_config_drops_zoom_selection_and_annotations() -> None:
    config = _all_band_plot_config(
        {
            "top_bands": 10,
            "bottom_bands": 4,
            "band_slice": [100, 124],
            "ylim": [-0.1, 0.02],
            "show_metrics": True,
            "figsize": [3.2, 5.8],
            "align": "top",
        }
    )

    assert config["plot_all_bands"] is True
    assert config["show_metrics"] is False
    assert config["legend_outside"] is True
    assert config["figsize"] == [3.2, 5.8]
    assert config["align"] == "top"
    assert "top_bands" not in config
    assert "bottom_bands" not in config
    assert "band_slice" not in config
    assert "ylim" not in config


def test_window_band_plot_config_uses_at_least_twelve_bands() -> None:
    config = _window_band_plot_config({"top_bands": 10, "align": "top"}, target_bands="top")

    assert config["top_bands"] == 12
    assert config["align"] == "top"
    assert config["show_metrics"] is False
    assert config["legend_outside"] is True


def test_window_band_plot_config_respects_larger_or_explicit_slices() -> None:
    assert _window_band_plot_config({"top_bands": 16}, target_bands="top")["top_bands"] == 16

    sliced = _window_band_plot_config({"band_slice": [4, 18], "align": "top"}, target_bands="top")
    assert sliced == {"band_slice": [4, 18], "align": "top", "show_metrics": False, "legend_outside": True}


def test_window_band_plot_config_defaults_from_target_edge() -> None:
    top = _window_band_plot_config({}, target_bands="top")
    bottom = _window_band_plot_config({}, target_bands="bottom")

    assert top["top_bands"] == 12
    assert bottom["bottom_bands"] == 12


def test_run_configured_model_preserves_plot_ylim_in_config(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.setdefault("bands", {})
    raw["bands"].setdefault("plot", {})
    raw["bands"]["plot"]["ylim"] = [0.0, 0.16]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)
    results = run_configured_model(cfg_path)

    assert results["configured_model"].band_plot_config["ylim"] == [0.0, 0.16]


def test_M_spinless_kp_symm_output_physical_source_ops_stay_physical_with_effective_metadata(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir(exist_ok=True)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"] = {
        "lattice": "hexagonal",
        "system": "bilayer",
        "valley_type": "M",
        "mode": "single_valley",
        "active_valleys": ["M1"],
        "spin_convention": "spinless_effective",
        "allowed_internal_symmetries": ["TR_eff", "C2_eff"],
        "external_sewing_symmetries": [],
    }
    raw["symmetry_source"] = {
        "type": "kp_symm_output",
        "path": "symm",
        "use": "raw",
        "matrix_kind": "action",
        "operations": [
            {
                "name": "TR",
                "antiunitary": True,
                "k_map": {"type": "negation"},
                "q_map": {"type": "negation"},
                "sector_map": "identity",
            },
            {
                "name": "C2",
                "antiunitary": False,
                "k_map": {"type": "reflection", "axis_deg": 0.0},
                "q_map": {"type": "reflection", "axis_deg": 0.0},
                "sector_map": "layer_exchange",
            },
        ],
                "source_matrix_role": "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "raw_low_heff_sewing",
        "gauge_correction": {"kind": "none"},
        "antiunitary_convention": "U_K",
    }
    raw["model"]["symmetry_map"] = {"Kinect": [{"name": "C2_eff"}, {"name": "TR_eff"}], "intra": [], "inter": []}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert [op["name"] for op in cfg.symmetry_source_config["operations"]] == ["TR", "C2"]
    assert [op["effective_name"] for op in cfg.symmetry_source_config["operations"]] == ["TR_eff", "C2_eff"]
    assert [op["name"] for op in cfg.symmetry_map["Kinect"]] == ["C2", "TR"]
    assert [op["effective_name"] for op in cfg.symmetry_map["Kinect"]] == ["C2_eff", "TR_eff"]


def test_validation_strict_rejects_unavailable_validation_outputs(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["validation"] = {"strict": True}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    with pytest.raises(ValueError, match="validation.strict=true requires"):
        run_configured_model(cfg_path)


def test_validation_production_mode_rejects_unavailable_validation_outputs(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["validation"] = {"mode": "production"}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_eigvals = np.load(tmp_path / "project" / "heff_eig.npy")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    with pytest.raises(ValueError, match="validation.strict=true requires"):
        run_configured_model(cfg_path)


def test_validation_production_band_level_only_allows_missing_projector_validations(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["validation"] = {"mode": "production", "band_level_only": True}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    results = run_configured_model(cfg_path)

    validations = results["validations"]
    assert validations["matrix_residuals"]["available"] is True
    assert validations["block_residuals"]["available"] is False
    assert validations["wavefunction_overlap"]["available"] is False
    summary = json.loads((tmp_path / "model_out" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["validation_incomplete"] is True


def test_explicit_harmonics_use_neutral_registry_source() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([0.0, 0.0]), 2: np.array([1.0, 0.0])},
        inter_harmonics_map={1: np.array([0.0, 0.0])},
        max_order={"Kinect": 1, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"intra": []},
        term_templates=[
            {
                "name": "intra_layer1_nonzero",
                "source": "moire_potential",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "all",
                "harmonics": {"kind": "intra", "indices": [2]},
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)
    term = next(iter(model.terms.values()))
    assert term.registry_metadata["harmonics_source"] == "explicit_indices"
    assert term.registry_metadata["generated_by"] == "representation_invariant_generator"


def test_symmetry_representative_harmonics_do_not_double_count_orbit() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    tr_op = {
        "name": "TR",
        "antiunitary": True,
        "k_map": {"type": "negation"},
        "q_map": {"type": "negation"},
        "sector_map": "identity",
    }
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={1: np.array([1.0, 0.0]), 2: np.array([-1.0, 0.0])},
        inter_harmonics_map={},
        max_order={"intra": 0},
        symmetry_map={"intra": [tr_op]},
        term_templates=[
            {
                "name": "intra_reps",
                "source": "moire_potential",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "all",
                "harmonics": "intra",
                "harmonics_source": "symmetry_representatives",
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)

    assert len(model.terms) == 1
    term = next(iter(model.terms.values()))
    assert term.registry_metadata["harmonics_source"] == "symmetry_representatives"


def test_symmetry_representative_harmonics_apply_sector_map() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    c2_op = {
        "name": "C2",
        "antiunitary": False,
        "k_map": {"type": "reflection", "axis_deg": 0.0},
        "q_map": {"type": "reflection", "axis_deg": 0.0},
        "sector_map": "layer_exchange",
    }
    cfg = MoireConfig(
        Q_set1=q,
        Q_set2=q,
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254037844386]),
        intra_harmonics_map={},
        inter_harmonics_map={1: np.array([0.25, 0.0])},
        max_order={"inter": 0},
        symmetry_map={"inter": [c2_op]},
        term_templates=[
            {
                "name": "inter_reps",
                "source": "tunneling",
                "sector_pairs": [[2, 1], [1, 2]],
                "orbital_pairs": "all",
                "harmonics": "inter",
                "harmonics_source": "symmetry_representatives",
                "max_order": 0,
            }
        ],
    )

    model = build_model(cfg)

    assert len(model.terms) == 1
    term = next(iter(model.terms.values()))
    assert term.registry_metadata["harmonic_orbit_size"] == 2


def test_auto_low_energy_fit_candidates_include_compact_path_anchors() -> None:
    seg1 = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21)])
    seg2 = np.column_stack([np.ones(20), np.linspace(0.05, 1.0, 20)])
    seg3 = np.column_stack([np.linspace(0.95, 0.0, 20), np.ones(20)])
    kpoints = np.vstack([seg1, seg2, seg3])

    candidates = _auto_low_energy_fit_candidate_sets(kpoints, initial_points=2, max_points=7)
    by_name = {row["name"]: row["indices"] for row in candidates}

    assert list(by_name)[:3] == ["minimal", "junctions_3", "junctions_4"]
    assert by_name["minimal"] == [0, 40]
    assert by_name["junctions_3"] == [0, 20, 40]
    assert by_name["junctions_4"] == [0, 20, 40, 60]
    assert by_name["uniform_5"] == [0, 15, 30, 45, 60]


def test_auto_low_energy_fit_candidates_use_initial_fit_indices_override() -> None:
    kpoints = np.column_stack([np.linspace(0.0, 1.0, 61), np.zeros(61)])

    candidates = _auto_low_energy_fit_candidate_sets(
        kpoints,
        initial_points=2,
        max_points=5,
        initial_indices=[0, 40],
    )

    assert candidates[0]["name"] == "minimal"
    assert candidates[0]["indices"] == [0, 40]


def test_auto_low_energy_fit_candidates_do_not_truncate_semantic_junctions() -> None:
    seg1 = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21)])
    seg2 = np.column_stack([np.ones(20), np.linspace(0.05, 1.0, 20)])
    seg3 = np.column_stack([np.linspace(0.95, 0.0, 20), np.ones(20)])
    kpoints = np.vstack([seg1, seg2, seg3])

    candidates = _auto_low_energy_fit_candidate_sets(
        kpoints,
        initial_points=2,
        max_points=2,
        initial_indices=[0, 40],
    )

    assert [row["name"] for row in candidates] == ["minimal"]
    assert candidates[0]["indices"] == [0, 40]
    assert [0, 20] not in [row["indices"] for row in candidates]


def test_auto_low_energy_refinement_indices_use_vertices_midpoints_and_edge_point() -> None:
    seg1 = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21)])
    seg2 = np.column_stack([np.ones(20), np.linspace(0.05, 1.0, 20)])
    seg3 = np.column_stack([np.linspace(0.95, 0.0, 20), np.ones(20)])
    kpoints = np.vstack([seg1, seg2, seg3])

    indices = _auto_low_energy_refinement_indices(
        kpoints,
        base_indices=[0, 40],
        max_points=8,
    )

    assert indices == [0, 2, 10, 20, 30, 40, 50, 60]


def test_auto_low_energy_fit_candidate_selection_rejects_low_band_overfit() -> None:
    records = [
        {
            "name": "center",
            "indices": [0],
            "plot_rms_mev": 1.18,
            "plot_max_mev": 3.58,
            "all_rms_mev": 79.8,
            "all_max_mev": 597.0,
            "fit_count": 1,
        },
        {
            "name": "center_last_junction",
            "indices": [0, 40],
            "plot_rms_mev": 1.50,
            "plot_max_mev": 4.26,
            "all_rms_mev": 3.35,
            "all_max_mev": 11.46,
            "fit_count": 2,
        },
        {
            "name": "high_symmetry_vertices",
            "indices": [0, 20, 40, 60],
            "plot_rms_mev": 1.44,
            "plot_max_mev": 4.53,
            "all_rms_mev": 3.15,
            "all_max_mev": 11.27,
            "fit_count": 4,
        },
    ]

    selected, report = _choose_auto_fit_candidate_record(records)

    assert selected["name"] == "high_symmetry_vertices"
    assert report["selection_status"] == "selected_sane_low_energy_candidate"
    assert report["rejected"][0]["name"] == "center"
    assert report["rejected"][0]["reason"] == "global_error_outlier"


def test_auto_low_energy_fit_candidate_selection_uses_one_standard_error_rule() -> None:
    records = [
        {
            "name": "minimal",
            "indices": [0, 40],
            "plot_rms_mev": 0.90,
            "plot_max_mev": 3.10,
            "all_rms_mev": 3.10,
            "all_max_mev": 11.0,
            "fit_count": 2,
            "active_terms": 500,
            "n_variables": 120,
        },
        {
            "name": "junctions_4",
            "indices": [0, 20, 40, 60],
            "plot_rms_mev": 0.88,
            "plot_max_mev": 3.00,
            "all_rms_mev": 3.05,
            "all_max_mev": 10.8,
            "fit_count": 4,
            "active_terms": 500,
            "n_variables": 120,
        },
    ]

    selected, report = _choose_auto_fit_candidate_record(records)

    assert selected["name"] == "minimal"
    assert report["selection_status"] == "selected_simplest_near_best_candidate"


def test_auto_low_energy_fit_candidate_selection_penalizes_extra_variables_when_quality_is_close() -> None:
    records = [
        {
            "name": "refine_A",
            "indices": [0, 20, 40, 60],
            "plot_rms_mev": 0.84,
            "plot_max_mev": 3.20,
            "all_rms_mev": 3.20,
            "all_max_mev": 11.5,
            "fit_count": 4,
            "active_terms": 634,
            "n_variables": 176,
        },
        {
            "name": "refine_B_interlayer",
            "indices": [0, 20, 40, 60],
            "plot_rms_mev": 0.83,
            "plot_max_mev": 3.15,
            "all_rms_mev": 3.18,
            "all_max_mev": 11.4,
            "fit_count": 4,
            "active_terms": 634,
            "n_variables": 619,
        },
    ]

    selected, report = _choose_auto_fit_candidate_record(records)

    assert selected["name"] == "refine_A"
    assert report["selection_status"] == "selected_simplest_near_best_candidate"
