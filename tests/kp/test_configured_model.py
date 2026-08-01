from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kp.model.export as export_module  # noqa: E402
import kp.model.operator_runtime as operator_runtime_module  # noqa: E402
from kp.model.pipeline import (  # noqa: E402
    _auto_harmonics_from_q_sets,
    _automatic_term_group_keep_sets,
    _apply_model_selection_term_filter,
    _auto_harmonics_from_support,
    _all_band_plot_config,
    _apply_refinement_acceptance_guard,
    _auto_low_energy_windows,
    _auto_low_energy_refinement_indices,
    _automatic_low_profile_band_count,
    _balanced_family_order_candidates,
    _build_operation_registry,
    _default_term_template_profile_metadata,
    _default_term_templates_for_model,
    _LEGACY_TERM_TEMPLATE_PROFILES,
    _TERM_TEMPLATE_PROFILES,
    _band_refinement_band_residual,
    _band_refinement_eigenvalue_jacobian,
    _band_refinement_global_matrix_jacobian,
    _band_refinement_reduced_global_matrix_loss,
    _default_auto_low_energy_order_config,
    _edge_weighted_band_weights,
    _harmonic_ablation_candidate_is_accepted,
    _harmonic_ablation_mask_from_shell_maps,
    _harmonic_ablation_shell_maps,
    _hamiltonian_element_comparison_arrays,
    _harmonic_recommendation_plot_candidates,
    _harmonic_selection_threshold_values,
    _selected_harmonic_support_mask,
    _run_harmonic_ablation_selection,
    _select_harmonic_ablation_candidate,
    _auto_low_energy_fit_candidate_sets,
    _band_overlap_weights,
    _choose_auto_fit_candidate_record,
    _low_subspace_matrix_residual,
    _matrix_loss_block_masks,
    _matrix_loss_residual,
    _load_model_artifact_identity,
    _low_cost_harmonic_recommendation_candidate,
    _load_kpoints,
    _load_kpoints_from_inputs,
    _operation_matrix_is_exactified,
    _max_derivative_order_values,
    _principal_angle_subspace_residual,
    _q_shell_row_indices,
    _representative_score,
    _resolve_layerwise_counts,
    _select_adaptive_fit_indices,
    _select_band_refinement_variables,
    _shell_band_window_from_target_eig,
    _shell_subspace_overlap_report,
    _solve_band_refinement_gauss_newton,
    _band_refinement_gauss_newton_available,
    _subspace_overlap_metrics,
    _symmetry_operation_index,
    _term_component_hamiltonians_for_kpoints,
    _term_response_pair_hamiltonians_for_kpoints,
    _print_automatic_harmonic_selection_report,
    _progress_line,
    _run_model_pipeline,
    _write_auto_model_selection_outputs,
    _select_nonlinear_frontier_scores,
    _refine_nonlinear_frontier_candidates,
    _refit_linear_profile_candidates,
    _harmonic_support_size,
    _linear_profile_fit_objective,
    _write_high_low_model_selection_outputs,
    _materialize_automatic_profile_output,
    _window_band_plot_config,
    ConfiguredModel,
    build_moire_config_from_file,
    compare_bands,
    compare_bands_for_plot,
    evaluate_vector_expression,
    load_model_config,
    matrix_residual,
    run_configured_model,
    save_hamiltonian_element_comparison_plot,
    save_band_comparison_plot,
    save_q_lattice_harmonics_plot,
)
import kp.model.core as model_core  # noqa: E402
import kp.model.pipeline as pipeline_module  # noqa: E402
from kp.model.model_selection import (  # noqa: E402
    CandidateScore,
    FamilyOrders,
    ModelSelectionConfig,
    select_high_low_profiles,
)
from kp.identity import hash_array  # noqa: E402
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


def _exactified_test_meta() -> dict[str, object]:
    return {
        "status": "exactified",
        "exactification_status": "exactified",
        "exactification_owner": "kp_symm",
        "basis_hash": "basis-fixture",
        "source_matrix_projection_report": {"report": {"status": "exactified"}},
    }


def test_model_artifact_identity_rejects_projection_symmetry_mismatch(tmp_path: Path) -> None:
    project_dir = tmp_path / "projection"
    symmetry_dir = tmp_path / "symmetry"
    project_dir.mkdir()
    symmetry_dir.mkdir()
    heff = np.eye(2, dtype=np.complex128)[None, :, :]
    np.save(project_dir / "heff.npy", heff)
    identity = {
        "identity_schema": "moirekp.artifact-identity.v1",
        "input_hash": "input-a",
        "config_hash": "config-a",
        "basis_hash": "basis-a",
        "package_version": "0.1.0",
        "schema_version": 2,
        "k_indices_hash": hash_array(np.asarray([0], dtype=np.int64)),
        "heff_hash": hash_array(heff),
    }
    scalar_identity = {key: np.asarray(value) for key, value in identity.items()}
    np.savez(project_dir / "basis.npz", **scalar_identity)
    np.savez(
        project_dir / "wavefunctions.npz",
        wavefunctions=np.eye(2, dtype=np.complex128)[None, :, :],
        k_indices=np.asarray([0]),
        **scalar_identity,
    )
    symmetry_identity = dict(identity)
    operation = {
        "name": "C3z",
        "matrix_kind": "continuum_internal_rep_exact",
        "matrix_source": "kp_symm_exactified_action",
        "status": "exactified",
        "exactification_status": "exactified",
        "exactification_owner": "kp_symm",
        "basis_hash": "basis-b",
        "k_indices_hash": identity["k_indices_hash"],
        "heff_hash": identity["heff_hash"],
        "source_matrix_projection_report": {"report": {"status": "exactified"}},
    }
    np.savez(
        symmetry_dir / "representations.npz",
        C3z=np.eye(2, dtype=np.complex128),
        __metadata_json__=np.asarray(
            json.dumps(
                {"artifact_identity": symmetry_identity, "operations": [operation]},
                sort_keys=True,
            )
        ),
    )

    with pytest.raises(ValueError, match="kp model.*basis_hash"):
        _load_model_artifact_identity(
            project_dir / "heff.npy",
            {"type": "kp_symm_output", "path": str(symmetry_dir)},
        )


def test_model_kpoints_follow_projection_source_indices(tmp_path: Path) -> None:
    kpoints = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
        dtype=float,
    )
    kpoints_file = tmp_path / "kpoints.npy"
    np.save(kpoints_file, kpoints)
    config = SimpleNamespace(
        kpoints_file=kpoints_file,
        project_k_indices=[1, 3],
        kpath_config={},
        path=tmp_path / "case.yaml",
        rotation_deg=0.0,
    )

    np.testing.assert_allclose(_load_kpoints(config), kpoints[[1, 3]])


def test_model_kpoints_from_projection_are_aligned_and_rotated(tmp_path: Path) -> None:
    source_cartesian = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)
    kpoints_file = tmp_path / "project" / "kpoints.npy"
    kpoints_file.parent.mkdir()
    np.save(kpoints_file, source_cartesian)
    heff_file = kpoints_file.with_name("heff.npy")
    np.save(heff_file, np.zeros((2, 1, 1), dtype=np.complex128))
    config = SimpleNamespace(
        kpoints_file=kpoints_file,
        kpoints_from_projection=True,
        project_k_indices=[4, 7],
        kpath_config={},
        path=tmp_path / "case.yaml",
        rotation_deg=90.0,
        heff_file=heff_file,
        artifact_identity={"kpoints_hash": hash_array(source_cartesian)},
    )

    np.testing.assert_allclose(
        _load_kpoints(config),
        np.array([[0.0, 1.0], [-2.0, 0.0]]),
        atol=1.0e-14,
    )


def test_auto_fit_kpoints_from_projection_are_not_source_reindexed(tmp_path: Path) -> None:
    source_cartesian = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)
    kpoints_file = tmp_path / "project" / "kpoints.npy"
    kpoints_file.parent.mkdir()
    np.save(kpoints_file, source_cartesian)

    loaded = _load_kpoints_from_inputs(
        kpoints_file=kpoints_file,
        kpath_config={},
        base=tmp_path,
        rotation_deg=90.0,
        kpoints_from_projection=True,
    )

    np.testing.assert_allclose(
        loaded,
        np.array([[0.0, 1.0], [-2.0, 0.0]]),
        atol=1.0e-14,
    )

def test_exactified_operation_requires_complete_production_provenance() -> None:
    complete = {
        "matrix_kind": "continuum_internal_rep_exact",
        "matrix_source": "kp_symm_exactified_action",
        "status": "exactified",
        "exactification_status": "exactified",
        "exactification_owner": "kp_symm",
        "basis_hash": "basis-a",
        "source_matrix_projection_report": {"report": {"status": "exactified"}},
    }

    assert _operation_matrix_is_exactified(complete)
    assert not _operation_matrix_is_exactified({**complete, "matrix_kind": "action"})
    assert not _operation_matrix_is_exactified({**complete, "matrix_source": "raw_h_action_projection"})
    assert not _operation_matrix_is_exactified({**complete, "status": "projected"})
    assert not _operation_matrix_is_exactified({**complete, "basis_hash": ""})
    assert not _operation_matrix_is_exactified(
        {**complete, "source_matrix_projection_report": {"report": {"status": "failed"}}}
    )


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
    original = operator_runtime_module.transform_monomial

    def spy(transform, q_base, mz, mz_star):
        calls.append((tuple(np.asarray(q_base, dtype=float)), int(mz), int(mz_star)))
        return original(transform, q_base, mz, mz_star)

    monkeypatch.setattr(operator_runtime_module, "transform_monomial", spy)

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


def test_support_matrix_solver_filters_numerical_null_channels_before_qr() -> None:
    builder = _support_grouping_builder()
    initial_vectors = np.array(
        [
            [1.0, 0.0],
            [1.0e-12, 0.0],
        ],
        dtype=np.complex128,
    )
    target_vector = np.array([2.0, 0.0], dtype=np.complex128)

    coeffs, includinglist = builder._solve_coefficients_from_support_matrix(
        initial_vectors,
        target_vector,
        tol=1.0e-13,
        null_channel_abs_tol=1.0e-10,
    )

    assert includinglist.tolist() == [0]
    assert coeffs == pytest.approx([2.0])


def test_support_matrix_solver_filters_small_symmetrized_to_raw_ratio_before_qr() -> None:
    builder = _support_grouping_builder()
    initial_vectors = np.array(
        [
            [1.0, 0.0],
            [1.0e-6, 0.0],
        ],
        dtype=np.complex128,
    )
    raw_norms = np.array([1.0, 1.0], dtype=float)
    target_vector = np.array([2.0, 0.0], dtype=np.complex128)

    coeffs, includinglist = builder._solve_coefficients_from_support_matrix(
        initial_vectors,
        target_vector,
        tol=1.0e-13,
        raw_channel_norms=raw_norms,
        null_channel_rel_tol=1.0e-3,
    )

    assert includinglist.tolist() == [0]
    assert coeffs == pytest.approx([2.0])


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


def test_compute_bands_can_collect_assembled_hamiltonians(monkeypatch):
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
    hamiltonians: list[np.ndarray] = []

    eigvals = compute_bands(cfg, model, cfg.kpoints, hamiltonians_out=hamiltonians)

    np.testing.assert_allclose(eigvals[0], [-0.5, 0.5])
    assert len(hamiltonians) == 1
    np.testing.assert_allclose(hamiltonians[0], [[0.0, 0.5], [0.5, 0.0]])


def test_validation_reuses_band_stage_hamiltonians(monkeypatch):
    q1 = np.zeros((1, 2), dtype=float)
    q2 = np.zeros((0, 2), dtype=float)
    moire_config = MoireConfig(
        Q_set1=q1,
        Q_set2=q2,
        n_orb1=1,
        n_orb2=0,
        nlow_state=[1, 0],
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        kpoints=np.array([[0.0, 0.0], [0.25, 0.0]], dtype=float),
        symmetry_gen=SymmetryGenerator(q1, q2, [1, 0]),
    )
    model_config = SimpleNamespace(
        validation_config={},
        compare_to_heff=False,
        symmetry_source_config={"operations": []},
        symmetry_map={},
        term_templates=[],
        valley_model={"valley_type": "Gamma", "mode": "single_valley", "spin_convention": "spinful"},
        band_indices=None,
        fit_indices=[],
    )
    model = SimpleNamespace(terms={})
    band_hamiltonians = np.array([[[1.0]], [[2.0]]], dtype=complex)
    assembly_calls = 0

    def fail_reassembly(*_args, **_kwargs):
        nonlocal assembly_calls
        assembly_calls += 1
        raise AssertionError("validation must reuse band-stage Hamiltonians")

    monkeypatch.setattr(pipeline_module, "_compute_one_k", fail_reassembly)

    pipeline_module._compute_validation_outputs(
        results={"model": model, "band_hamiltonians": band_hamiltonians},
        model_config=model_config,
        moire_config=moire_config,
    )

    assert assembly_calls == 0


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
    np.save(out_dir / "heff.npy", heff)
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)

    model_cfg = {
        "case": {"profile": "test", "q_shell": "q00", "output_root": "."},
        "material": {
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "plot": {},
        "project": {
            "out_dir": "project",
        },
        "symm": {"output_dir": "symm"},
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


def _write_k_inter_direction_fixture(tmp_path: Path) -> Path:
    p_vec = np.array([0.0, 1.0], dtype=float)
    q1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    q2 = q1 - p_vec
    kpoints = np.array([[0.0, 0.0]], dtype=float)
    dim = q1.shape[0] + q2.shape[0]
    heff = np.diag(np.arange(dim, dtype=float))[None, :, :].astype(np.complex128)

    np.save(tmp_path / "q1.npy", q1)
    np.save(tmp_path / "q2.npy", q2)
    np.save(tmp_path / "kpoints.npy", kpoints)
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    np.save(out_dir / "heff.npy", heff)
    _write_symm_frame_manifest(
        tmp_path,
        rotation_deg=0.0,
        q_model_files={"layer1": "../q1.npy", "layer2": "../q2.npy"},
    )

    model_cfg = {
        "case": {"profile": "K1", "q_shell": "q00", "output_root": "."},
        "material": {
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "plot": {},
        "project": {
            "out_dir": "project",
        },
        "symm": {"output_dir": "symm"},
        "symmetry_source": {"type": "kp_symm_output", "path": "symm"},
        "valley_model": {
            "lattice": "hexagonal",
            "system": "bilayer",
            "valley_type": "K",
            "mode": "single_valley",
            "active_valleys": ["K1"],
            "spin_convention": "spin_down_projected",
            "allowed_internal_symmetries": [],
            "external_sewing_symmetries": [],
        },
        "sectors": [
            {"name": "L1", "qset": "qset1", "n_orb": 1},
            {"name": "L2", "qset": "qset2", "n_orb": 1},
        ],
        "kpoints_file": "kpoints.npy",
        "model": {
            "n_orb": [1, 1],
            "nlow_state": [1, 1],
            "bM": {"bM1": [1.0, 0.0], "bM2": [0.0, 1.0]},
            "harmonics": {
                "intra": 1,
                "inter": {1: "[0.0, 1.0]"},
            },
            "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
            "symmetry_map": {
                "Kinect": [],
                "Onsite": [],
                "intra": [],
                "inter": [],
            },
        },
        "fit": {
            "indices": [0],
            "coeff_tol": 1.0e-8,
        },
        "bands": {
            "indices": [0],
            "compare_to_heff": True,
        },
        "output": {
            "dir": "model_out",
        },
    }
    path = tmp_path / "model_k_inter_direction.yaml"
    path.write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")
    return path


def _write_symm_frame_manifest(
    tmp_path: Path,
    *,
    rotation_deg: float,
    path_name: str = "symm",
    q_model_files: dict[str, str] | None = None,
    project_k_indices: list[int] | None = None,
) -> Path:
    symm_dir = tmp_path / path_name
    symm_dir.mkdir(exist_ok=True)
    manifest = {"frame": {"q_transform": {"rotation_deg": float(rotation_deg)}}, "operations": []}
    if q_model_files is not None:
        manifest["q_model"] = {"files": dict(q_model_files)}
    (symm_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    project_dir = tmp_path / "project"
    heff_path = project_dir / "heff.npy"
    if heff_path.exists():
        heff = np.load(heff_path, allow_pickle=False)
        if project_k_indices is None:
            project_k_indices = list(range(int(heff.shape[0])))
        if len(project_k_indices) != int(heff.shape[0]):
            raise ValueError("test fixture project_k_indices must match Heff rows")
        identity = {
            "identity_schema": "moirekp.artifact-identity.v1",
            "input_hash": "input-fixture",
            "config_hash": "config-fixture",
            "basis_hash": "basis-fixture",
            "package_version": "0.1.0",
            "schema_version": 2,
            "k_indices_hash": hash_array(
                np.asarray(project_k_indices, dtype=np.int64)
            ),
            "heff_hash": hash_array(heff),
        }
        scalar_identity = {key: np.asarray(value) for key, value in identity.items()}
        dim = int(heff.shape[-1])
        np.savez(project_dir / "basis.npz", **scalar_identity)
        np.savez(
            project_dir / "wavefunctions.npz",
            wavefunctions=np.stack([np.eye(dim, dtype=np.complex128)] * int(heff.shape[0])),
            k_indices=np.asarray(project_k_indices, dtype=int),
            **scalar_identity,
        )
        np.savez_compressed(
            symm_dir / "representations.npz",
            __metadata_json__=np.asarray(
                json.dumps({**manifest, "artifact_identity": identity}, sort_keys=True)
            ),
        )
    return symm_dir


def _write_unified_case_fixture(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "kp" / "configs"
    cfg_dir.mkdir(parents=True)
    qset = _triangular_q_shell()
    np.save(cfg_dir / "q1.npy", -qset)
    np.save(cfg_dir / "q2.npy", -qset)
    np.save(cfg_dir / "kpoints.npy", np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=float))

    profile = "tiny_K1_B"
    q_shell = "q04"
    project_out = tmp_path / "kp" / "outputs" / profile / q_shell / "projection"
    project_out.mkdir(parents=True)
    dim = len(qset)
    heff = np.stack([np.diag(np.arange(dim, dtype=float) + shift) for shift in (0.0, 0.1, 0.2)]).astype(np.complex128)
    np.save(project_out / "heff.npy", heff)

    symm_out = tmp_path / "kp" / "outputs" / profile / q_shell / "symmetry"
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
        "case": {"profile": profile, "q_shell": q_shell, "output_root": "../outputs"},
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
    path = cfg_dir / f"{profile}_{q_shell}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_public_fit_method_parses_linear_contract(tmp_path: Path) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["fit"] = {
        "method": "linear",
        "kpoints": [0, 2],
        "bands": 2,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.response_semantics == "complete_linear_v2"
    assert config.fit_indices == [0, 2]
    assert config.model_selection_config is None
    assert config.fit_method_config == {
        "method": "linear",
        "hamiltonian_kpoints": [0, 2],
        "band_kpoints": None,
        "requested_bands": 2,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
        "band_loss_weight": None,
        "max_steps": None,
        "normalization": "dimension_mean_square_v1",
    }
    assert config.response_fit_objective == {
        "mode": "normalized_low_energy_linear_v1",
        "target_reference": "current_heff_support_mask",
        "edge": "top",
        "window": {
            "mode": "fixed_count_degeneracy_safe",
            "bands": 2,
            "degeneracy_tol_mev": 0.1,
        },
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
        "normalization": "dimension_mean_square_v1",
    }
    assert config.band_refinement_config == {"enabled": False}


def test_public_fit_method_parses_nonlinear_contract(tmp_path: Path) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["fit"] = {
        "method": "nonlinear",
        "kpoints": [0, 2],
        "band_kpoints": "all",
        "bands": 2,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
        "band_loss_weight": 2.0,
        "max_steps": 17,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.response_semantics == "complete_linear_v2"
    assert config.fit_indices == [0, 2]
    assert config.model_selection_config is None
    assert config.fit_method_config == {
        "method": "nonlinear",
        "hamiltonian_kpoints": [0, 2],
        "band_kpoints": [0, 1, 2],
        "requested_bands": 2,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
        "band_loss_weight": 2.0,
        "max_steps": 17,
        "normalization": "dimension_mean_square_v1",
    }
    assert config.response_fit_objective["mode"] == "normalized_low_energy_linear_v1"
    assert config.band_refinement_config == {
        "enabled": True,
        "mode": "public_nonlinear_v1",
        "reference": "current_heff_support_mask",
        "hamiltonian_kpoints": [0, 2],
        "band_kpoints": [0, 1, 2],
        "target_bands": "top",
        "bands": 2,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
        "band_loss_weight": 2.0,
        "max_steps": 17,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: model.pop("max_order"), "model.max_order"),
        (lambda model: model.update({"profiles": ["linear/high"]}), "model.profiles"),
        (
            lambda model: model.update({"max_derivative_order": {"intralayer_zero": 0}}),
            "max_derivative_order",
        ),
    ],
)
def test_public_fit_method_requires_explicit_simple_model_space(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["fit"] = {
        "method": "linear",
        "kpoints": [0, 2],
        "bands": 2,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
    }
    mutation(raw["model"])
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_model_config(cfg_path)


@pytest.mark.parametrize("method", ["linear", "nonlinear"])
def test_public_fit_method_automatically_selects_omitted_harmonics(
    tmp_path: Path,
    method: str,
) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    fit = {
        "method": method,
        "kpoints": [0, 2],
        "bands": 2,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
    }
    if method == "nonlinear":
        fit.update(
            {
                "band_kpoints": [0, 2],
                "band_loss_weight": 2.0,
                "max_steps": 5,
            }
        )
    raw["model"]["fit"] = fit
    raw["model"].pop("harmonics")
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    report = config.automatic_harmonic_selection
    assert config.raw["model"]["automatic_defaults"]["harmonics"] is True
    assert config.harmonics_config["intra"]["count"] == report["selected"]["intra_shells"]
    assert config.harmonics_config["inter"]["count"] == report["selected"]["inter_shells"]
    assert config.fit_method_config["method"] == method


@pytest.mark.parametrize(
    ("fit", "message"),
    [
        (
            {
                "method": "other",
                "kpoints": [0],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
            },
            "fit.method",
        ),
        (
            {
                "method": "linear",
                "kpoints": [],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
            },
            "fit.kpoints",
        ),
        (
            {
                "method": "linear",
                "kpoints": [0, 0],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
            },
            "fit.kpoints.*unique",
        ),
        (
            {
                "method": "linear",
                "kpoints": [3],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
            },
            "fit.kpoints.*0..2",
        ),
        (
            {
                "method": "linear",
                "kpoints": [0],
                "bands": 8,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
            },
            "fit.bands.*dimension",
        ),
        (
            {
                "method": "linear",
                "kpoints": [0],
                "bands": 1,
                "one_sided_weight": -1.0,
                "two_sided_weight": 1.0,
            },
            "one_sided_weight",
        ),
        (
            {
                "method": "linear",
                "kpoints": [0],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": float("inf"),
            },
            "two_sided_weight",
        ),
        (
            {
                "method": "linear",
                "kpoints": [0],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
                "band_loss_weight": 1.0,
            },
            "band_loss_weight.*nonlinear",
        ),
        (
            {
                "method": "nonlinear",
                "kpoints": [0],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
                "band_loss_weight": 1.0,
            },
            "band_kpoints",
        ),
        (
            {
                "method": "nonlinear",
                "kpoints": [0],
                "band_kpoints": [1, 1],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
                "band_loss_weight": 1.0,
            },
            "band_kpoints.*unique",
        ),
        (
            {
                "method": "nonlinear",
                "kpoints": [0],
                "band_kpoints": [1],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
                "band_loss_weight": 0.0,
            },
            "band_loss_weight",
        ),
        (
            {
                "method": "nonlinear",
                "kpoints": [0],
                "band_kpoints": [1],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
                "band_loss_weight": 1.0,
                "max_steps": 0,
            },
            "max_steps",
        ),
        (
            {
                "method": "linear",
                "kpoints": [0],
                "bands": 1,
                "one_sided_weight": 1.0,
                "two_sided_weight": 1.0,
                "objective": {"mode": "equal_matrix_v1"},
            },
            "cannot be mixed.*objective",
        ),
    ],
)
def test_public_fit_method_rejects_invalid_contract(
    tmp_path: Path,
    fit: dict[str, object],
    message: str,
) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["fit"] = fit
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_model_config(cfg_path)


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

    def fake_export(
        model_output_dir,
        output_dir,
        *,
        force=False,
        debug_files=False,
        operator_data=None,
    ):
        seen["export"] = (Path(model_output_dir), Path(output_dir), bool(force), bool(debug_files))
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path)])

    assert seen["path"] == str(cfg_path)
    assert seen["export"] == (model_output, model_output, True, False)


def test_cli_model_subcommand_does_not_reexport_materialized_profiles(monkeypatch, tmp_path: Path) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"
    exports: list[tuple[Path, Path]] = []

    class FakeModelConfig:
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)

    high_cfg = FakeModelConfig(model_output / "high")
    low_cfg = FakeModelConfig(model_output / "low")
    root_cfg = FakeModelConfig(model_output)

    def fake_run(path: str) -> dict:
        return {
            "configured_model": root_cfg,
            "comparison": {"rms_error": 0.0, "max_abs_error": 0.0},
            "profile_results": {
                "high": {"model_config": high_cfg, "results": {}},
                "low": {"model_config": low_cfg, "results": {}},
            },
        }

    def fake_export(model_output_dir, output_dir, **kwargs):
        exports.append((Path(model_output_dir), Path(output_dir)))
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path)])

    assert exports == []


def test_cli_model_subcommand_preserves_four_materialized_profile_directories(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"
    exports: list[tuple[Path, Path]] = []

    class FakeModelConfig:
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)

    nested = {
        family: {
            quality: {
                "model_config": FakeModelConfig(model_output / family / quality),
                "results": {},
            }
            for quality in ("high", "low")
        }
        for family in ("linear", "nonlinear")
    }

    monkeypatch.setattr(
        configured,
        "run_configured_model",
        lambda path: {
            "configured_model": nested["nonlinear"]["high"]["model_config"],
            "comparison": {"rms_error": 0.0, "max_abs_error": 0.0},
            "profile_results": nested,
        },
    )
    monkeypatch.setattr(
        export_mod,
        "export_standalone_model",
        lambda source, output, **kwargs: exports.append((Path(source), Path(output))),
    )

    cli.main(["model", "--config", str(cfg_path)])

    assert exports == []


@pytest.mark.parametrize(
    "profile_paths",
    [
        ("linear/low", "linear/high"),
        ("nonlinear/high",),
    ],
)
def test_cli_model_subcommand_preserves_requested_profile_subset(
    monkeypatch,
    tmp_path: Path,
    profile_paths: tuple[str, ...],
) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    model_output = tmp_path / "model_out"
    exports: list[tuple[Path, Path]] = []

    class FakeModelConfig:
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)

    nested: dict[str, dict[str, dict[str, object]]] = {}
    for profile_path in profile_paths:
        family, quality = profile_path.split("/")
        nested.setdefault(family, {})[quality] = {
            "model_config": FakeModelConfig(model_output / family / quality),
            "results": {},
        }
    primary_family, primary_quality = profile_paths[-1].split("/")
    monkeypatch.setattr(
        configured,
        "run_configured_model",
        lambda path: {
            "configured_model": nested[primary_family][primary_quality]["model_config"],
            "comparison": {"rms_error": 0.0, "max_abs_error": 0.0},
            "profile_results": nested,
        },
    )
    monkeypatch.setattr(
        export_mod,
        "export_standalone_model",
        lambda source, output, **kwargs: exports.append((Path(source), Path(output))),
    )

    cli.main(["model", "--config", str(cfg_path)])

    assert exports == []


def test_cli_model_subcommand_prints_band_plot_path(monkeypatch, tmp_path: Path, capsys) -> None:
    import kp.cli as cli
    import kp.model.export as export_mod
    import kp.model.pipeline as configured

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    plot_path = tmp_path / "model_out" / "band_comparison.pdf"
    all_plot_path = tmp_path / "model_out" / "band_comparison_all.pdf"
    full_heff_plot_path = tmp_path / "model_out" / "band_comparison_vs_full_heff.pdf"

    class FakeModelConfig:
        output_dir = tmp_path / "model_out"

    def fake_run(path: str) -> dict:
        return {
            "configured_model": FakeModelConfig(),
            "band_plot": str(plot_path.resolve()),
            "all_band_plot": str(all_plot_path.resolve()),
            "full_heff_band_plot": str(full_heff_plot_path.resolve()),
            "comparison": {
                "rms_error": 0.0,
                "max_abs_error": 0.0,
                "reference": "current_heff_support_mask",
            },
            "plot_comparison": {
                "rms_error": 0.001,
                "max_abs_error": 0.002,
                "rms_error_mev": 1.0,
                "max_abs_error_mev": 2.0,
                "num_bands": 6,
                "align": "top",
                "model_alignment_shift_meV": -0.5,
                "reference": "current_heff_support_mask",
            },
            "all_band_plot_comparison": {
                "rms_error": 0.002,
                "max_abs_error": 0.003,
                "rms_error_mev": 2.0,
                "max_abs_error_mev": 3.0,
                "num_bands": 124,
                "align": "top",
                "reference": "current_heff_support_mask",
            },
            "plot_comparison_vs_full_heff": {
                "rms_error": 0.019,
                "max_abs_error": 0.037,
                "rms_error_mev": 19.0,
                "max_abs_error_mev": 37.0,
                "num_bands": 6,
                "align": "top",
                "model_alignment_shift_meV": -37.0,
                "reference": "original_heff",
            },
        }

    def fake_export(
        model_output_dir,
        output_dir,
        *,
        force=False,
        debug_files=False,
        operator_data=None,
    ):
        return Path(output_dir)

    monkeypatch.setattr(configured, "run_configured_model", fake_run)
    monkeypatch.setattr(export_mod, "export_standalone_model", fake_export)

    cli.main(["model", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert "[kp model] Results" in out
    assert f"  band plot  {plot_path.resolve()}" in out
    assert f"  all-band plot  {all_plot_path.resolve()}" in out
    assert f"  full-Heff band plot  {full_heff_plot_path.resolve()}" in out
    assert "[kp model] Validation" in out
    assert "  RMS error vs current Heff support mask  0.000000e+00 eV (0.000 meV)" in out
    assert "  plot bands RMS vs current Heff support mask  1.000 meV, Max: 2.000 meV" in out
    assert "  all-band RMS vs current Heff support mask  2.000 meV, Max: 3.000 meV" in out
    assert "  plot bands RMS vs original Heff  19.000 meV, Max: 37.000 meV" in out


def test_canonical_model_cleanup_preserves_harmonic_recommendation_plot(tmp_path: Path) -> None:
    import kp.cli as cli

    output = tmp_path / "model_out"
    output.mkdir()
    keep = output / "harmonic_recommendation_bands.png"
    keep.write_bytes(b"plot")
    heatmap = output / "hamiltonian_element_comparison.png"
    heatmap.write_bytes(b"heatmap")
    heatmap_pdf = output / "hamiltonian_element_comparison.pdf"
    heatmap_pdf.write_bytes(b"%PDF-1.4\n")
    support_heatmap = output / "hamiltonian_element_comparison_selected_support.png"
    support_heatmap.write_bytes(b"support heatmap")
    support_heatmap_pdf = output / "hamiltonian_element_comparison_selected_support.pdf"
    support_heatmap_pdf.write_bytes(b"%PDF-1.4\n")
    heatmap_dir = output / "hamiltonian_element_comparisons"
    heatmap_dir.mkdir()
    grouped_heatmap = heatmap_dir / "native_full.png"
    grouped_heatmap.write_bytes(b"grouped heatmap")
    stale = output / "temporary.txt"
    stale.write_text("remove", encoding="utf-8")

    cli._cleanup_canonical_model_output(output)

    assert keep.exists()
    assert heatmap.exists()
    assert heatmap_pdf.exists()
    assert not support_heatmap.exists()
    assert not support_heatmap_pdf.exists()
    assert grouped_heatmap.exists()
    assert not stale.exists()


def test_model_band_cleanup_removes_stale_primary_and_secondary_comparisons(tmp_path: Path) -> None:
    output = tmp_path / "model_out"
    output.mkdir()
    primary = output / "band_comparison.pdf"
    secondary = output / "band_comparison_vs_full_heff.pdf"
    harmonic = output / "harmonic_recommendation_bands.png"
    primary.write_bytes(b"old primary")
    secondary.write_bytes(b"old secondary")
    harmonic.write_bytes(b"keep")

    pipeline_module._cleanup_stale_band_outputs(output)

    assert not primary.exists()
    assert not secondary.exists()
    assert harmonic.exists()


def test_cli_model_subcommand_rejects_explicit_standalone_export_path(tmp_path: Path) -> None:
    import kp.cli as cli

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("source_config: source.yaml\n", encoding="utf-8")
    export_output = tmp_path / "portable"

    with pytest.raises(SystemExit):
        cli.main(["model", "--config", str(cfg_path), "--export-standalone", str(export_output)])


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


def test_compiled_operator_recipe_keeps_onsite_terms_within_same_layer_q_block() -> None:
    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0]], dtype=float)

    key = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
    y_basis = ContinuumModelBuilder.make_Y_basis_function(key, q1, q2, 1, 1)
    term = ContinuumTerm(
        key=key,
        Y_basis=y_basis,
        r_value_real=1.0,
        r_value_imag=0.0,
        active=True,
        tag="Onsite",
        symmetry_ops=[{"name": "C2", "k_map": {"type": "identity"}}],
    )

    class DenseLayerExchange:
        def get_operator(self, name: str, params=None):
            assert name == "C2"
            return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)

    recipe = operator_runtime_module.compile_operator_recipe(
        [term],
        SimpleNamespace(symmetry_gen=DenseLayerExchange()),
        dim=2,
    )

    rows = np.asarray(recipe["row"], dtype=int)
    cols = np.asarray(recipe["col"], dtype=int)
    assert set(zip(rows.tolist(), cols.tolist())) <= {(0, 0), (1, 1)}
    assert (0, 1) not in set(zip(rows.tolist(), cols.tolist()))
    assert (1, 0) not in set(zip(rows.tolist(), cols.tolist()))


def test_hamiltonian_element_panels_reorder_q_blocks_and_subtract_layer_diagonal_means() -> None:
    # Native basis order is layer/orbital/Q.  Plot order should be layer/Q/orbital,
    # so a same-Q orbital block becomes adjacent in the heatmap.
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0]], dtype=float)
    heff = np.zeros((1, 5, 5), dtype=np.complex128)
    model = np.zeros_like(heff)
    heff[0, np.arange(5), np.arange(5)] = [10.0, 14.0, 20.0, 24.0, 100.0]
    model[0, np.arange(5), np.arange(5)] = [11.0, 15.0, 21.0, 25.0, 102.0]
    heff[0, 0, 2] = 3.0
    heff[0, 2, 0] = 3.0
    model[0, 0, 2] = 4.0
    model[0, 2, 0] = 4.0

    panels = _hamiltonian_element_comparison_arrays(
        model,
        heff,
        np.ones((5, 5), dtype=bool),
        positions=[0],
        qset1=q1,
        qset2=q2,
        n_orb=(2, 1),
        subtract_layer_diagonal_mean=True,
    )

    assert panels["display_order"] == [0, 2, 1, 3, 4]
    assert panels["q_block_boundaries"] == [2, 4]
    assert panels["layer_boundaries"] == [4]
    assert panels["diag_offsets"]["heff"][0] == pytest.approx([17.0, 100.0])
    assert panels["diag_offsets"]["model"][0] == pytest.approx([18.0, 102.0])
    assert panels["heff"][0, 0, 0] == pytest.approx(7.0)
    assert panels["heff"][0, 0, 1] == pytest.approx(3.0)
    assert panels["model"][0, 0, 0] == pytest.approx(7.0)
    assert panels["diff"][0, 0, 1] == pytest.approx(1.0)


def test_q_block_display_metadata_orders_each_layer_by_q_norm_shell() -> None:
    q1 = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0 + 1.0e-10, 0.0],
        ],
        dtype=float,
    )
    q2 = np.array([[0.0, 0.0]], dtype=float)

    metadata = pipeline_module._q_block_display_metadata(
        q1,
        q2,
        (2, 1),
        dim=9,
        q_order="norm_shell",
        q_norm_tol=1.0e-6,
    )

    assert metadata["order"] == [1, 5, 3, 7, 2, 6, 0, 4, 8]
    assert metadata["q_block_boundaries"] == [2, 4, 6, 8]
    assert metadata["q_shell_boundaries"] == [2, 6]
    assert metadata["layer_boundaries"] == [8]


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
    np.save(out_dir / "heff.npy", heff)
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)

    model_cfg = {
        "case": {"profile": "test", "q_shell": "q00", "output_root": "."},
        "material": {
            "qset1_file": "q1.npy",
            "qset2_file": "q2.npy",
        },
        "plot": {},
        "project": {
            "out_dir": "project",
        },
        "symm": {"output_dir": "symm"},
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


def _disable_model_selection_for_mock_pipeline(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.setdefault("fit", {})["model_selection"] = False
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _expected_project_eigvals(tmp_path: Path) -> np.ndarray:
    return np.linalg.eigvalsh(np.load(tmp_path / "project" / "heff.npy"))


def test_response_semantics_defaults_legacy_and_v2_requires_explicit_template_policy(
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    legacy = load_model_config(cfg_path)
    assert legacy.response_semantics == "legacy_frozen_v1"

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["response_semantics"] = "complete_linear_v2"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    complete = load_model_config(cfg_path)
    assert complete.response_semantics == "complete_linear_v2"
    assert complete.term_templates == []
    assert complete.term_template_metadata["generator"] == "gamma_case_derived_v1"

    raw["model"]["term_templates"] = [
        {
            "name": "custom",
            "source": "onsite",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": "all",
            "max_order": 0,
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="term_space_policy"):
        load_model_config(cfg_path)


def test_complete_gamma_default_defers_to_case_derived_generator(
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["response_semantics"] = "complete_linear_v2"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    configured = load_model_config(cfg_path)

    assert configured.term_templates == []
    assert configured.term_template_metadata == {
        "input_kind": "default",
        "valley_type": "Gamma",
        "n_orb": [1, 1],
        "profiles": [],
        "generator": "gamma_case_derived_v1",
        "term_space_policies": ["complete"],
    }


@pytest.mark.parametrize(
    ("valley_type", "generator"),
    [
        ("K", "k_case_derived_v1"),
        ("M", "m_case_derived_v1"),
    ],
)
def test_complete_k_m_default_defers_to_case_derived_generator(
    tmp_path: Path,
    valley_type: str,
    generator: str,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = valley_type
    raw["valley_model"]["active_valleys"] = [valley_type]
    raw["model"]["response_semantics"] = "complete_linear_v2"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    configured = load_model_config(cfg_path)

    assert configured.term_templates == []
    assert configured.term_template_metadata == {
        "input_kind": "default",
        "valley_type": valley_type,
        "n_orb": [1, 1],
        "profiles": [],
        "generator": generator,
        "term_space_policies": ["complete"],
    }


@pytest.mark.parametrize("valley_type", ["K", "M"])
def test_complete_k_m_explicit_templates_bypass_case_generator(
    tmp_path: Path,
    valley_type: str,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = valley_type
    raw["valley_model"]["active_valleys"] = [valley_type]
    raw["model"]["response_semantics"] = "complete_linear_v2"
    raw["model"]["term_templates"] = [
        {
            "name": "custom",
            "source": "onsite",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": "all",
            "max_order": 0,
            "term_space_policy": "complete",
        }
    ]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    configured = load_model_config(cfg_path)

    assert configured.term_templates == raw["model"]["term_templates"]
    assert configured.term_template_metadata == {"input_kind": "explicit"}


def test_default_profile_registry_contains_no_complete_valley_profiles() -> None:
    assert not any(
        profile.valley_type in {"Gamma", "K", "M"}
        for profile in _TERM_TEMPLATE_PROFILES
    )


def test_gamma_case_derived_templates_follow_actual_q_support() -> None:
    qset1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    qset2 = np.array([[0.0, 0.0], [-1.0, 0.0]], dtype=float)
    sectors = [
        {"name": "L1", "qset": "qset1", "n_orb": 2},
        {"name": "L2", "qset": "qset2", "n_orb": 2},
    ]

    templates, diagnostics = pipeline_module._case_derived_gamma_term_templates(
        sectors=sectors,
        n_orb=(2, 2),
        max_order={
            "Kinect": 2,
            "Onsite": 0,
            "intra": 1,
            "inter": 1,
        },
        Q_set1=qset1,
        Q_set2=qset2,
        intra_harmonics_map={
            1: np.array([0.0, 0.0]),
            2: np.array([1.0, 0.0]),
        },
        inter_harmonics_map={
            1: np.array([0.0, 0.0]),
            2: np.array([1.0, 0.0]),
            3: np.array([7.0, 0.0]),
        },
    )

    assert {row["term_space_policy"] for row in templates} == {"complete"}
    assert {row["source"] for row in templates} == {
        "diagonal_kp",
        "onsite",
        "moire_potential",
        "tunneling",
    }
    assert all(row["orbital_pairs"] == "all" for row in templates)
    assert all("positive" not in row["name"] for row in templates)
    assert all("negative" not in row["name"] for row in templates)
    assert diagnostics["generator"] == "gamma_case_derived_v1"

    by_name = {str(sector["name"]): sector for sector in sectors}
    inter_directions: set[tuple[str, str]] = set()
    for row in templates:
        records = row.get("harmonic_records", [])
        for from_name, to_name in row["sector_pairs"]:
            if row["source"] == "tunneling":
                inter_directions.add((from_name, to_name))
            for record in records:
                support = pipeline_module._model_q_pair_support_count_for_sector_pair(
                    sector_from=by_name[from_name],
                    sector_to=by_name[to_name],
                    p_vector=np.asarray(record["vector"], dtype=float),
                    Q_set1=qset1,
                    Q_set2=qset2,
                    tol=1.0e-8,
                )
                assert support == record["support_count"]
                assert support > 0

    assert inter_directions == {("L1", "L2"), ("L2", "L1")}
    assert any(
        record["kind"] == "inter"
        and np.allclose(np.abs(record["vector"]), [7.0, 0.0])
        and record["reason"] == "no_q_pair_support"
        for record in diagnostics["omitted_harmonics"]
    )


@pytest.mark.parametrize(
    ("valley_type", "generator", "name_prefix"),
    [
        ("K", "k_case_derived_v1", "k_case_"),
        ("M", "m_case_derived_v1", "m_case_"),
    ],
)
def test_k_m_case_derived_templates_follow_actual_q_support(
    valley_type: str,
    generator: str,
    name_prefix: str,
) -> None:
    qset1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    qset2 = np.array([[0.0, 0.0], [-1.0, 0.0]], dtype=float)
    sectors = [
        {"name": "L1", "qset": "qset1", "n_orb": 2},
        {"name": "L2", "qset": "qset2", "n_orb": 2},
    ]

    templates, diagnostics = pipeline_module._case_derived_term_templates(
        valley_type=valley_type,
        sectors=sectors,
        n_orb=(2, 2),
        max_order={"Kinect": 3, "intra": 2, "inter": 1},
        Q_set1=qset1,
        Q_set2=qset2,
        intra_harmonics_map={
            1: np.array([0.0, 0.0]),
            2: np.array([1.0, 0.0]),
        },
        inter_harmonics_map={
            1: np.array([0.0, 0.0]),
            2: np.array([1.0, 0.0]),
            3: np.array([7.0, 0.0]),
        },
    )

    assert templates
    assert all(row["name"].startswith(name_prefix) for row in templates)
    assert all(row["orbital_pairs"] == "all" for row in templates)
    assert {row["term_space_policy"] for row in templates} == {"complete"}
    assert not any("monomial_constraints" in row for row in templates)
    assert diagnostics["generator"] == generator
    assert diagnostics["valley_type"] == valley_type
    assert diagnostics["orders"] == {"Kinect": 3, "intra": 2, "inter": 1}

    kinetic = [row for row in templates if row["source"] == "diagonal_kp"]
    onsite = [row for row in templates if row["source"] == "onsite"]
    intra = [row for row in templates if row["source"] == "moire_potential"]
    inter = [row for row in templates if row["source"] == "tunneling"]
    assert {tuple(row["sector_pairs"][0]) for row in kinetic} == {
        ("L1", "L1"),
        ("L2", "L2"),
    }
    assert {row["max_order"] for row in kinetic} == {3}
    assert {tuple(row["sector_pairs"][0]) for row in onsite} == {
        ("L1", "L1"),
        ("L2", "L2"),
    }
    assert {row["max_order"] for row in onsite} == {0}
    assert {tuple(row["sector_pairs"][0]) for row in intra} == {
        ("L1", "L1"),
        ("L2", "L2"),
    }
    assert {row["max_order"] for row in intra} == {2}
    assert {tuple(row["sector_pairs"][0]) for row in inter} == {
        ("L1", "L2"),
        ("L2", "L1"),
    }
    assert {row["max_order"] for row in inter} == {1}

    by_name = {str(sector["name"]): sector for sector in sectors}
    for row in intra + inter:
        for record in row["harmonic_records"]:
            from_name, to_name = row["sector_pairs"][0]
            assert record["support_count"] == pipeline_module._model_q_pair_support_count_for_sector_pair(
                sector_from=by_name[from_name],
                sector_to=by_name[to_name],
                p_vector=np.asarray(record["vector"], dtype=float),
                Q_set1=qset1,
                Q_set2=qset2,
                tol=1.0e-8,
            )
            assert record["support_count"] > 0

    assert any(
        record["kind"] == "inter"
        and np.allclose(np.abs(record["vector"]), [7.0, 0.0])
        and record["reason"] == "no_q_pair_support"
        for record in diagnostics["omitted_harmonics"]
    )


def test_case_support_count_matches_core_q_pair_convention_with_sector_offsets() -> None:
    qset1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    qset2 = np.array([[0.5, 0.0], [1.5, 0.0]], dtype=float)
    sector1 = {
        "name": "L1",
        "qset": "qset1",
        "n_orb": 1,
        "q_offset": [0.2, 0.0],
    }
    sector2 = {
        "name": "L2",
        "qset": "qset2",
        "n_orb": 1,
        "q_offset": [-0.1, 0.0],
    }

    for layer_from, layer_to, sector_from, sector_to, p_vector in (
        (1, 2, sector1, sector2, np.array([-0.5, 0.0])),
        (2, 1, sector2, sector1, np.array([0.5, 0.0])),
    ):
        key = ContinuumTermKey(
            Mz=0,
            Mz_star=0,
            layer_from=layer_from,
            layer_to=layer_to,
            orbital_from=1,
            orbital_to=1,
            p=tuple(p_vector),
        )
        core_response = ContinuumModelBuilder.make_Y_basis_function(
            key,
            qset1,
            qset2,
            1,
            1,
        )(np.zeros(2, dtype=float))

        support_count = pipeline_module._model_q_pair_support_count_for_sector_pair(
            sector_from=sector_from,
            sector_to=sector_to,
            p_vector=p_vector,
            Q_set1=qset1,
            Q_set2=qset2,
            tol=1.0e-8,
        )

        assert support_count == np.count_nonzero(core_response) == 2


def test_build_materializes_default_gamma_case_envelope(
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    symm_dir = tmp_path / "symm"
    np.save(symm_dir / "q_model_layer1.npy", np.load(tmp_path / "q1.npy"))
    np.save(symm_dir / "q_model_layer2.npy", np.load(tmp_path / "q2.npy"))
    _write_symm_frame_manifest(
        tmp_path,
        rotation_deg=0.0,
        q_model_files={
            "layer1": "q_model_layer1.npy",
            "layer2": "q_model_layer2.npy",
        },
    )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["response_semantics"] = "complete_linear_v2"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire, configured = build_moire_config_from_file(cfg_path)

    assert configured.term_template_metadata["generator"] == "gamma_case_derived_v1"
    assert configured.term_template_metadata["generation_status"] == "materialized"
    assert moire.term_templates
    assert moire.term_templates == configured.term_templates
    assert {
        row["term_space_policy"]
        for row in moire.term_templates
    } == {"complete"}
    names = {row["name"] for row in moire.term_templates}
    assert not any(name.startswith("gamma_inter_") for name in names)
    assert "gamma_case_kinetic" in names


@pytest.mark.parametrize(
    ("valley_type", "generator", "name_prefix"),
    [
        ("K", "k_case_derived_v1", "k_case_"),
        ("M", "m_case_derived_v1", "m_case_"),
    ],
)
def test_build_materializes_default_k_m_case_envelope(
    tmp_path: Path,
    valley_type: str,
    generator: str,
    name_prefix: str,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    symm_dir = tmp_path / "symm"
    np.save(symm_dir / "q_model_layer1.npy", np.load(tmp_path / "q1.npy"))
    np.save(symm_dir / "q_model_layer2.npy", np.load(tmp_path / "q2.npy"))
    _write_symm_frame_manifest(
        tmp_path,
        rotation_deg=0.0,
        q_model_files={
            "layer1": "q_model_layer1.npy",
            "layer2": "q_model_layer2.npy",
        },
    )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = valley_type
    raw["valley_model"]["active_valleys"] = [valley_type]
    raw["model"]["response_semantics"] = "complete_linear_v2"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire, configured = build_moire_config_from_file(cfg_path)

    assert configured.term_template_metadata["generator"] == generator
    assert configured.term_template_metadata["generation_status"] == "materialized"
    assert configured.term_template_metadata["valley_type"] == valley_type
    assert "inter_sector_pair_resolution" not in configured.term_template_metadata
    assert moire.term_templates == configured.term_templates
    assert moire.term_templates
    assert all(
        row["name"].startswith(name_prefix)
        for row in moire.term_templates
    )
    assert {
        row["term_space_policy"]
        for row in moire.term_templates
    } == {"complete"}


def test_explicit_gamma_templates_bypass_case_derived_generation(
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    symm_dir = tmp_path / "symm"
    np.save(symm_dir / "q_model_layer1.npy", np.load(tmp_path / "q1.npy"))
    np.save(symm_dir / "q_model_layer2.npy", np.load(tmp_path / "q2.npy"))
    _write_symm_frame_manifest(
        tmp_path,
        rotation_deg=0.0,
        q_model_files={
            "layer1": "q_model_layer1.npy",
            "layer2": "q_model_layer2.npy",
        },
    )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["response_semantics"] = "complete_linear_v2"
    explicit = {
        "name": "user_gamma_onsite",
        "source": "onsite",
        "sector_pairs": [["L1", "L1"]],
        "orbital_pairs": "all",
        "max_order": 0,
        "term_space_policy": "complete",
    }
    raw["model"]["term_templates"] = [explicit]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire, configured = build_moire_config_from_file(cfg_path)

    assert configured.term_template_metadata == {"input_kind": "explicit"}
    assert moire.term_templates == [explicit]


def test_complete_v2_parses_target_spectral_fit_objective(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["response_semantics"] = "complete_linear_v2"
    raw["model"]["target_bands"] = "top"
    raw["fit"]["objective"] = {
        "mode": "target_spectral_linear",
        "target_reference": "current_heff_support_mask",
        "window": {
            "mode": "fixed_count_degeneracy_safe",
            "bands": 10,
            "degeneracy_tol_mev": 0.1,
        },
        "floor": 0.05,
        "alpha": 1.0,
        "normalization": "mean_trace_per_dimension_v1",
        "two_sided_projector": {
            "enabled": True,
            "weight": 300.0,
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.response_fit_objective == {
        "mode": "target_spectral_linear",
        "target_reference": "current_heff_support_mask",
        "edge": "top",
        "window": {
            "mode": "fixed_count_degeneracy_safe",
            "bands": 10,
            "degeneracy_tol_mev": 0.1,
        },
        "floor": 0.05,
        "alpha": 1.0,
        "normalization": "mean_trace_per_dimension_v1",
        "two_sided_projector": {
            "enabled": True,
            "weight": 300.0,
        },
    }


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_target_spectral_projector_rejects_invalid_weight(
    tmp_path: Path,
    weight: float,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["response_semantics"] = "complete_linear_v2"
    raw["fit"]["objective"] = {
        "mode": "target_spectral_linear",
        "window": {"mode": "fixed_count_degeneracy_safe", "bands": 2},
        "two_sided_projector": {"enabled": True, "weight": weight},
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="two_sided_projector.weight"):
        load_model_config(cfg_path)


def test_target_spectral_fit_objective_rejects_legacy_semantics(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["objective"] = {
        "mode": "target_spectral_linear",
        "window": {"mode": "fixed_count_degeneracy_safe", "bands": 2},
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="complete_linear_v2"):
        load_model_config(cfg_path)


def test_target_spectral_fit_uses_current_heff_support_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["response_semantics"] = "complete_linear_v2"
    raw["symmetry_source"] = {"type": "toy_generator", "allow": True}
    raw["fit"]["objective"] = {
        "mode": "target_spectral_linear",
        "window": {"mode": "fixed_count_degeneracy_safe", "bands": 2},
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    original = np.load(tmp_path / "project" / "heff.npy")
    replacement = np.asarray(original, dtype=np.complex128).copy()
    replacement += 7.0 * np.eye(replacement.shape[-1], dtype=np.complex128)[None, :, :]
    calls: list[dict[str, object]] = []

    def fake_support(values: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append(dict(kwargs))
        np.testing.assert_allclose(values, original)
        return replacement

    monkeypatch.setattr(
        pipeline_module,
        "_current_heff_support_hamiltonians",
        fake_support,
    )

    moire_config, model_config = build_moire_config_from_file(cfg_path)

    assert calls
    assert moire_config.response_fit_objective["mode"] == "target_spectral_linear"
    dim = replacement.shape[-1]
    fitted_blocks = np.asarray(
        [
            moire_config.heff[i * dim : (i + 1) * dim, i * dim : (i + 1) * dim]
            for i in range(len(model_config.fit_indices))
        ]
    )
    np.testing.assert_allclose(fitted_blocks, replacement[model_config.fit_indices])


def test_all_builtin_term_profiles_and_templates_declare_space_policy() -> None:
    for profile in _LEGACY_TERM_TEMPLATE_PROFILES:
        assert profile.term_space_policy in {
            "complete",
            "explicit_reduced",
            "orbit_representative",
        }
        for template in profile.templates:
            assert template["term_space_policy"] == profile.term_space_policy


def test_builtin_curated_reduced_profiles_are_declared_as_orbit_representatives() -> None:
    policies = {
        (profile.valley_type, profile.n_orb): profile.term_space_policy
        for profile in _LEGACY_TERM_TEMPLATE_PROFILES
    }

    assert policies[("K", None)] == "orbit_representative"
    assert policies[("M", None)] == "orbit_representative"
    assert policies[("Gamma", (2, 2))] == "orbit_representative"


def test_build_moire_config_remaps_projected_heff_rows_to_source_kpoints(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    source_kpoints = np.load(tmp_path / "kpoints.npy")
    full_heff = np.load(tmp_path / "project" / "heff.npy")
    selected_source_rows = [0, 2]
    np.save(tmp_path / "project" / "heff.npy", full_heff[selected_source_rows])
    _write_symm_frame_manifest(
        tmp_path,
        rotation_deg=0.0,
        project_k_indices=selected_source_rows,
    )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["indices"] = [0, 1]
    raw["bands"]["indices"] = [0, 1]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    moire_config, model_config = build_moire_config_from_file(cfg_path)

    assert model_config.project_k_indices == selected_source_rows
    np.testing.assert_allclose(moire_config.kpoints, source_kpoints[selected_source_rows])
    np.testing.assert_allclose(moire_config.kpoints_fit, source_kpoints[selected_source_rows])


def test_load_model_config_resolves_paths_relative_to_yaml(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)

    cfg = load_model_config(cfg_path)

    assert cfg.path == cfg_path
    assert cfg.source_config == cfg_path
    assert cfg.kpoints_file == tmp_path / "kpoints.npy"
    assert cfg.heff_file == tmp_path / "project" / "heff.npy"
    assert cfg.output_dir == tmp_path / "model_out"
    assert cfg.fit_indices == [0, 2]
    assert cfg.coeff_prune_threshold == 2.0e-4
    assert cfg.band_indices == [0, 1, 2]


def test_load_model_config_parses_opt_in_model_selection(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["max_order"] = {
        "kinetic": 4,
        "intralayer": 2,
        "interlayer": 1,
    }
    raw["fit"]["model_selection"] = {
        "enabled": True,
        "folds": 3,
        "orders": {
            "kinetic": [1, 2, 4],
            "intralayer": [0, 1, 2],
            "interlayer": [0, 1],
        },
        "quality": {
            "overlap_target": 0.95,
            "overlap_safety_floor": 0.90,
        },
        "profiles": {
            "low": {
                "standard_error_multiplier": 2.0,
                "term_keep_fractions": [0.25, 0.5, 0.75],
            }
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.model_selection_config.enabled is True
    assert cfg.model_selection_config.n_folds == 3
    assert cfg.model_selection_config.order_candidates == {
        "kinetic": (1, 2, 4),
        "intra": (0, 1, 2),
        "inter": (0, 1),
    }
    assert cfg.model_selection_config.pruning_keep_fractions == (0.25, 0.5, 0.75)


def test_automatic_term_group_keep_sets_preserve_most_important_groups() -> None:
    ranked = [("weak", 0.1), ("medium", 1.0), ("strong", 10.0), ("dominant", 100.0)]

    keep_sets = _automatic_term_group_keep_sets(
        ranked,
        keep_fractions=(0.25, 0.5, 0.75),
    )

    assert keep_sets == (
        ("dominant",),
        ("strong", "dominant"),
        ("medium", "strong", "dominant"),
    )


def test_apply_model_selection_term_filter_removes_whole_closed_terms() -> None:
    keep = SimpleNamespace(key="keep")
    remove = SimpleNamespace(key="remove")
    model = SimpleNamespace(
        terms={"keep": keep, "remove": remove},
        candidate_terms=[keep, remove],
    )

    report = _apply_model_selection_term_filter(model, ("remove",))

    assert model.terms == {"keep": keep}
    assert model.candidate_terms == [keep]
    assert report == {"requested": 1, "removed": 1, "remaining": 1}


def test_load_model_config_enables_model_selection_by_default(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)

    cfg = load_model_config(cfg_path)

    assert cfg.model_selection_config.enabled is True


def test_automatic_family_order_scan_rebuilds_and_refits_each_stage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)
    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)
    model_cfg = replace(
        model_cfg,
        band_refinement_config={
            "enabled": True,
            "solver": "nonlinear_band",
            "frontier_max_candidates": 3,
            "max_variables": 3000,
        },
        model_selection_config=ModelSelectionConfig(
            enabled=True,
            order_candidates={
                "kinetic": (1, 2),
                "intra": (0, 1),
                "inter": (0, 1),
            },
            n_folds=2,
            pruning_keep_fractions=(),
        ),
    )
    dim = int(
        len(moire_cfg.Q_set1) * moire_cfg.n_orb1
        + len(moire_cfg.Q_set2) * moire_cfg.n_orb2
    )
    target = np.stack(
        [
            np.diag(np.arange(dim, dtype=float) + 0.01 * index)
            for index in range(len(moire_cfg.kpoints))
        ]
    ).astype(complex)
    calls: list[tuple[FamilyOrders, tuple[str, ...]]] = []

    monkeypatch.setattr(
        pipeline_module,
        "_current_heff_support_hamiltonians",
        lambda *args, **kwargs: target.copy(),
    )

    def fake_pipeline(candidate_moire, candidate_model, log_path, *, verbose, progress):
        del log_path, verbose, progress
        orders = FamilyOrders(
            int(candidate_moire.max_order["Kinect"]),
            int(candidate_moire.max_order["intra"]),
            int(candidate_moire.max_order["inter"]),
        )
        sources = tuple(
            sorted(
                {
                    str(item.get("source"))
                    for item in candidate_moire.term_templates
                    if str(item.get("source")) != "onsite"
                }
            )
        )
        calls.append((orders, sources))
        error_mev = 80.0 / (10 ** sum(orders.as_tuple()))
        profile = np.linspace(0.5, 1.5, len(target)) * error_mev * 1.0e-3
        hamiltonians = target + profile[:, None, None] * np.eye(dim)[None, :, :]
        eigvals, eigvecs = np.linalg.eigh(hamiltonians)
        terms = {
            f"p{index}": SimpleNamespace(
                active=True,
                r_value_real=1.0,
                r_value_imag=0.0,
            )
            for index in range(1 + sum(orders.as_tuple()))
        }
        return {
            "model": SimpleNamespace(terms=terms),
            "eigvals": (eigvals, eigvecs),
            "band_hamiltonians": hamiltonians,
            "band_refinement": {"enabled": False},
        }

    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)

    scan = pipeline_module._run_automatic_family_order_scan(
        moire_config=moire_cfg,
        model_config=model_cfg,
        output_dir=tmp_path / "scan",
        verbose=False,
        progress=False,
    )

    assert scan["staged_selection"].selected_orders == FamilyOrders(2, 1, 1)
    assert scan["profiles"].high.selected.orders == FamilyOrders(2, 1, 1)
    assert scan["profile_families"]["linear"] is scan["profiles"]
    assert scan["profile_families"]["nonlinear"].high.selected is not None
    assert scan["profile_families"]["nonlinear"].low.selected is not None
    assert all(
        run["score"].solver_family == "nonlinear"
        for name, run in scan["runs"].items()
        if name.endswith("__nonlinear")
    )
    assert scan["correction_sweep"] is not None
    assert any(orders == FamilyOrders(1, 1, 1) for orders, _sources in calls)
    assert calls[:2] == [
        (FamilyOrders(1, 0, 0), ("diagonal_kp",)),
        (FamilyOrders(2, 0, 0), ("diagonal_kp",)),
    ]
    assert all("tunneling" not in sources for _orders, sources in calls[:4])


def test_load_model_config_accepts_single_case_yaml_with_nested_model_sections(tmp_path: Path) -> None:
    cfg_path = _write_unified_case_fixture(tmp_path)

    cfg = load_model_config(cfg_path)

    assert cfg.source_config == cfg_path
    assert cfg.source_raw["case"] == {"profile": "tiny_K1_B", "q_shell": "q04", "output_root": "../outputs"}
    assert cfg.heff_file == tmp_path / "kp" / "outputs" / "tiny_K1_B" / "q04" / "projection" / "heff.npy"
    assert cfg.output_dir == tmp_path / "kp" / "outputs" / "tiny_K1_B" / "q04" / "model"
    assert cfg.valley_model["valley_type"] == "K"
    assert cfg.valley_model["active_valleys"] == ["K1"]
    assert cfg.valley_model["spin_convention"] == "spin_down_projected"
    assert cfg.symmetry_source_config["type"] == "kp_symm_output"
    assert cfg.symmetry_source_config["path"] == str((tmp_path / "kp" / "outputs" / "tiny_K1_B" / "q04" / "symmetry").resolve())
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
    assert {row["term_space_policy"] for row in cfg.term_templates} == {
        "orbit_representative"
    }


def test_gamma_layerwise_orbitals_resolve_by_source_qset_not_nonempty_layers() -> None:
    gamma_counts, gamma_metadata = _resolve_layerwise_counts(
        [0, 2, 2],
        name="model.n_orb",
        num_layer_list=[1, 2],
    )
    k_counts, _ = _resolve_layerwise_counts(
        [0, 2, 2],
        name="model.n_orb",
        num_layer_list=[1, 2],
        prefer_active_layer_sectors=True,
    )

    assert gamma_counts == [0, 4]
    assert gamma_metadata["resolved_qset"] == [0, 4]
    assert k_counts == [2, 2]


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


def test_band_only_analytic_refinement_can_use_gauss_newton() -> None:
    assert _band_refinement_gauss_newton_available(
        analytic_jacobian_enabled=True,
        matrix_jacobian=None,
    )
    assert not _band_refinement_gauss_newton_available(
        analytic_jacobian_enabled=False,
        matrix_jacobian=None,
    )
    assert not _band_refinement_gauss_newton_available(
        analytic_jacobian_enabled=True,
        matrix_jacobian=object(),
    )


def test_nonlinear_frontier_keeps_linear_high_low_and_pareto_vocabularies() -> None:
    compact = CandidateScore(
        name="compact",
        orders=FamilyOrders(3, 1, 1),
        independent_real_parameters=10,
        weighted_rms_mev=1.0,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=2.0,
        mean_subspace_overlap=0.98,
    )
    middle = CandidateScore(
        name="middle",
        orders=FamilyOrders(4, 2, 2),
        independent_real_parameters=20,
        weighted_rms_mev=0.5,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=1.0,
        mean_subspace_overlap=0.98,
    )
    dominated = CandidateScore(
        name="dominated",
        orders=FamilyOrders(5, 3, 3),
        independent_real_parameters=30,
        weighted_rms_mev=0.8,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=1.5,
        mean_subspace_overlap=0.98,
    )
    accurate = CandidateScore(
        name="accurate",
        orders=FamilyOrders(6, 4, 4),
        independent_real_parameters=40,
        weighted_rms_mev=0.2,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=0.5,
        mean_subspace_overlap=0.98,
    )
    oversized = CandidateScore(
        name="oversized",
        orders=FamilyOrders(8, 6, 6),
        independent_real_parameters=3001,
        weighted_rms_mev=0.1,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=0.3,
        mean_subspace_overlap=0.98,
    )
    over_order = CandidateScore(
        name="over-order",
        orders=FamilyOrders(10, 6, 10),
        independent_real_parameters=50,
        weighted_rms_mev=0.1,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=0.3,
        mean_subspace_overlap=0.98,
    )
    profiles = select_high_low_profiles((compact, middle, dominated, accurate))

    selected = _select_nonlinear_frontier_scores(
        (compact, middle, dominated, accurate, oversized),
        profiles=profiles,
        maximum_candidates=4,
        maximum_variables=3000,
    )

    assert [candidate.name for candidate in selected] == [
        "compact",
        "middle",
        "accurate",
    ]

    capped = _select_nonlinear_frontier_scores(
        (compact, middle, accurate, over_order),
        profiles=profiles,
        maximum_candidates=4,
        maximum_variables=3000,
        maximum_orders=FamilyOrders(6, 4, 4),
    )
    assert [candidate.name for candidate in capped] == [
        "compact",
        "middle",
        "accurate",
    ]

    dominated_endpoint = replace(
        middle,
        name="dominated-endpoint",
        weighted_rms_mev=1.2,
    )
    endpoint_capped = _select_nonlinear_frontier_scores(
        (compact, dominated_endpoint),
        profiles=select_high_low_profiles((compact, dominated_endpoint)),
        maximum_candidates=4,
        maximum_orders=FamilyOrders(4, 2, 2),
    )
    assert [candidate.name for candidate in endpoint_capped] == [
        "compact",
        "dominated-endpoint",
    ]


def test_harmonic_support_size_uses_active_fitted_harmonics_not_configured_ceiling() -> None:
    model_config = SimpleNamespace(harmonics_config={"intra": 4, "inter": 4})
    model = SimpleNamespace(
        terms={
            "a": SimpleNamespace(
                active=True,
                r_value_real=1.0,
                r_value_imag=0.0,
                tag="intra",
                registry_metadata={"harmonic_id": 1},
            ),
            "b": SimpleNamespace(
                active=True,
                r_value_real=2.0,
                r_value_imag=0.0,
                tag="intra",
                registry_metadata={"harmonic_id": 1},
            ),
            "c": SimpleNamespace(
                active=True,
                r_value_real=0.0,
                r_value_imag=0.0,
                tag="inter",
                registry_metadata={"harmonic_id": 4},
            ),
        }
    )

    assert _harmonic_support_size(model_config, model=model, tolerance=1.0e-8) == 1


def test_nonlinear_frontier_refines_each_selected_vocabulary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    moire_config, model_config = build_moire_config_from_file(cfg_path)
    model_config = replace(
        model_config,
        band_refinement_config={
            "enabled": True,
            "solver": "nonlinear_band",
            "weighted_band_loss": {"enabled": True, "primary_bands": 4},
        },
    )
    target = np.asarray(
        pipeline_module._current_heff_support_hamiltonians(
            np.load(model_config.heff_file),
            qset1=np.asarray(moire_config.Q_set1),
            qset2=np.asarray(moire_config.Q_set2),
            n_orb=model_config.n_orb,
            harmonics_config=model_config.harmonics_config,
        )
    )
    target = pipeline_module._select_rows(target, model_config.band_indices)
    scores = (
        CandidateScore(
            name="compact",
            orders=FamilyOrders(2, 1, 1),
            independent_real_parameters=8,
            weighted_rms_mev=0.8,
            weighted_rms_se_mev=0.1,
            weighted_max_mev=1.0,
            mean_subspace_overlap=0.98,
        ),
        CandidateScore(
            name="accurate",
            orders=FamilyOrders(4, 2, 2),
            independent_real_parameters=16,
            weighted_rms_mev=0.4,
            weighted_rms_se_mev=0.1,
            weighted_max_mev=0.7,
            mean_subspace_overlap=0.99,
        ),
    )
    linear_config = replace(
        model_config,
        band_refinement_config={"enabled": False},
    )
    runs = {
        score.name: {
            "score": score,
            "model_config": linear_config,
            "moire_config": moire_config,
        }
        for score in scores
    }
    seen: list[tuple[str, dict[str, object]]] = []

    def fake_pipeline(candidate_moire, candidate_model, log_path, **kwargs):
        seen.append((Path(log_path).stem, dict(candidate_model.band_refinement_config)))
        assert candidate_model.band_refinement_config["enabled"] is True
        eigvals, eigvecs = np.linalg.eigh(target)
        return {
            "model": SimpleNamespace(terms={}),
            "eigvals": (eigvals, eigvecs),
            "band_hamiltonians": target,
            "band_refinement": {"enabled": True},
        }

    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)

    refined = _refine_nonlinear_frontier_candidates(
        frontier_scores=scores,
        linear_runs=runs,
        model_config=model_config,
        target_hamiltonians=target,
        kpoints=np.asarray(moire_config.kpoints),
        n_folds=3,
        low_primary_count=4,
        high_primary_count=3,
        candidate_dir=tmp_path / "candidate_logs",
        verbose=False,
    )

    assert set(refined) == {
        "compact__nonlinear_high",
        "compact__nonlinear_low",
        "accurate__nonlinear_high",
        "accurate__nonlinear_low",
    }
    assert [name for name, _config in seen] == [
        "compact__nonlinear_high",
        "compact__nonlinear_low",
        "accurate__nonlinear_high",
        "accurate__nonlinear_low",
    ]
    assert all(run["score"].solver_family == "nonlinear" for run in refined.values())
    assert refined["compact__nonlinear_high"]["score"].selection_scope == "high"
    assert refined["compact__nonlinear_low"]["score"].selection_scope == "low"
    low_configs = [config for name, config in seen if name.endswith("_low")]
    high_configs = [config for name, config in seen if name.endswith("_high")]
    assert all(config["matrix_weight"] == 0.0 for config in low_configs)
    assert all(config["optimizer"] == "auto" for config in low_configs)
    assert all(config["band_slice"] == [0, 4] for config in low_configs)
    assert all(config["acceptance_guard"]["guard_all_bands"] is False for config in low_configs)
    assert all(config["band_slice"] == [target.shape[-1] - 3, target.shape[-1]] for config in high_configs)
    assert all(config["weighted_band_loss"]["primary_bands"] == 3 for config in high_configs)
    assert {run["score"].orders for run in refined.values()} == {
        FamilyOrders(2, 1, 1),
        FamilyOrders(4, 2, 2),
    }

    seen.clear()
    scoped = _refine_nonlinear_frontier_candidates(
        frontier_scores=(
            replace(scores[0], selection_scope="high"),
            replace(scores[1], selection_scope="low"),
        ),
        linear_runs=runs,
        model_config=model_config,
        target_hamiltonians=target,
        kpoints=np.asarray(moire_config.kpoints),
        n_folds=3,
        low_primary_count=4,
        high_primary_count=4,
        candidate_dir=tmp_path / "candidate_logs_scoped",
        verbose=False,
    )
    assert set(scoped) == {
        "compact__nonlinear_high",
        "accurate__nonlinear_low",
    }


def test_uniform_nonlinear_frontier_keeps_equal_band_weights(
    monkeypatch,
    tmp_path: Path,
) -> None:
    moire_config, model_config = build_moire_config_from_file(_write_fixture(tmp_path))
    model_config = replace(
        model_config,
        fit_weighting="uniform",
        requested_profiles=("nonlinear/high",),
        band_refinement_config={
            "enabled": True,
            "solver": "nonlinear_band",
            "weighted_band_loss": {"enabled": False},
        },
    )
    target = np.asarray(
        pipeline_module._current_heff_support_hamiltonians(
            np.load(model_config.heff_file),
            qset1=np.asarray(moire_config.Q_set1),
            qset2=np.asarray(moire_config.Q_set2),
            n_orb=model_config.n_orb,
            harmonics_config=model_config.harmonics_config,
        )
    )
    target = pipeline_module._select_rows(target, model_config.band_indices)
    score = CandidateScore(
        name="uniform-source",
        orders=FamilyOrders(2, 1, 1),
        independent_real_parameters=8,
        weighted_rms_mev=0.8,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=1.0,
        mean_subspace_overlap=0.98,
        selection_scope="high",
    )
    linear_config = replace(model_config, band_refinement_config={"enabled": False})
    seen: list[dict[str, object]] = []

    def fake_pipeline(candidate_moire, candidate_model, log_path, **kwargs):
        seen.append(dict(candidate_model.band_refinement_config))
        eigvals, eigvecs = np.linalg.eigh(target)
        return {
            "model": SimpleNamespace(terms={}),
            "eigvals": (eigvals, eigvecs),
            "band_hamiltonians": target,
            "band_refinement": dict(candidate_model.band_refinement_config),
        }

    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)

    refined = _refine_nonlinear_frontier_candidates(
        frontier_scores=(score,),
        linear_runs={
            score.name: {
                "score": score,
                "model_config": linear_config,
                "moire_config": moire_config,
            }
        },
        model_config=model_config,
        target_hamiltonians=target,
        kpoints=np.asarray(moire_config.kpoints),
        n_folds=3,
        low_primary_count=2,
        high_primary_count=4,
        candidate_dir=tmp_path / "uniform_nonlinear_logs",
        verbose=False,
    )

    assert set(refined) == {"uniform-source__nonlinear_high"}
    assert len(seen) == 1
    assert seen[0]["weighted_band_loss"] == {
        "enabled": False,
        "primary_bands": 4,
    }


def test_linear_profile_refit_uses_complete_two_sided_objective(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    moire_config, model_config = build_moire_config_from_file(cfg_path)
    target = np.asarray(
        pipeline_module._current_heff_support_hamiltonians(
            np.load(model_config.heff_file),
            qset1=np.asarray(moire_config.Q_set1),
            qset2=np.asarray(moire_config.Q_set2),
            n_orb=model_config.n_orb,
            harmonics_config=model_config.harmonics_config,
        )
    )
    target = pipeline_module._select_rows(target, model_config.band_indices)
    high = CandidateScore(
        name="high-source",
        orders=FamilyOrders(4, 2, 2),
        independent_real_parameters=16,
        weighted_rms_mev=0.4,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=0.7,
        mean_subspace_overlap=0.99,
        selection_scope="high",
    )
    low = CandidateScore(
        name="low-source",
        orders=FamilyOrders(2, 1, 1),
        independent_real_parameters=8,
        weighted_rms_mev=0.8,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=1.0,
        mean_subspace_overlap=0.98,
        selection_scope="low",
    )
    linear_config = replace(model_config, band_refinement_config={"enabled": False})
    runs = {
        score.name: {
            "score": score,
            "model_config": linear_config,
            "moire_config": moire_config,
        }
        for score in (high, low)
    }
    seen: list[tuple[str, str, int, float, tuple[str, ...], tuple[str, ...]]] = []

    def fake_pipeline(candidate_moire, candidate_model, log_path, **kwargs):
        objective = candidate_model.response_fit_objective
        seen.append(
            (
                Path(log_path).stem,
                candidate_model.response_semantics,
                objective["window"]["bands"],
                objective["two_sided_projector"]["weight"],
                tuple(row["name"] for row in candidate_model.term_templates),
                tuple(row["term_space_policy"] for row in candidate_model.term_templates),
            )
        )
        return {
            "model": SimpleNamespace(terms={}),
            "band_hamiltonians": target,
            "band_refinement": {"enabled": False},
        }

    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)

    refitted = _refit_linear_profile_candidates(
        frontier_scores=(high, low),
        linear_runs=runs,
        target_hamiltonians=target,
        kpoints=np.asarray(moire_config.kpoints),
        n_folds=3,
        low_primary_count=2,
        high_primary_count=4,
        candidate_dir=tmp_path / "linear_profile_logs",
        verbose=False,
    )

    assert set(refitted) == {
        "high-source__linear_high_weighted",
        "low-source__linear_low_weighted",
    }
    assert [row[:4] for row in seen] == [
        ("high-source__linear_high_weighted", "complete_linear_v2", 4, 300.0),
        ("low-source__linear_low_weighted", "complete_linear_v2", 2, 300.0),
    ]
    for row in seen:
        assert row[4] == (
            "gamma_1x1_kinetic",
            "gamma_1x1_onsite",
            "gamma_1x1_intra_nonzero",
            "gamma_1x1_inter_zero",
            "gamma_1x1_inter_nonzero",
        )
        assert set(row[5]) == {"complete"}


def test_uniform_linear_profile_refit_reuses_equal_matrix_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    moire_config, model_config = build_moire_config_from_file(_write_fixture(tmp_path))
    model_config = replace(
        model_config,
        fit_weighting="uniform",
        response_fit_objective={"mode": "equal_matrix_v1"},
        band_refinement_config={"enabled": False},
    )
    target = np.asarray(
        pipeline_module._current_heff_support_hamiltonians(
            np.load(model_config.heff_file),
            qset1=np.asarray(moire_config.Q_set1),
            qset2=np.asarray(moire_config.Q_set2),
            n_orb=model_config.n_orb,
            harmonics_config=model_config.harmonics_config,
        )
    )
    target = pipeline_module._select_rows(target, model_config.band_indices)
    eigvals, eigvecs = np.linalg.eigh(target)
    base_result = {
        "model": SimpleNamespace(terms={}),
        "eigvals": (eigvals, eigvecs),
        "band_hamiltonians": target,
        "band_refinement": {"enabled": False},
    }
    high = CandidateScore(
        name="high-source",
        orders=FamilyOrders(4, 2, 2),
        independent_real_parameters=16,
        weighted_rms_mev=0.4,
        weighted_rms_se_mev=0.1,
        weighted_max_mev=0.7,
        mean_subspace_overlap=0.99,
        selection_scope="high",
    )
    low = replace(
        high,
        name="low-source",
        orders=FamilyOrders(2, 1, 1),
        independent_real_parameters=8,
        selection_scope="low",
    )
    runs = {
        score.name: {
            "score": score,
            "result": base_result,
            "model_config": model_config,
            "moire_config": moire_config,
        }
        for score in (high, low)
    }
    monkeypatch.setattr(
        pipeline_module,
        "_run_model_pipeline",
        lambda *args, **kwargs: pytest.fail("uniform linear profiles must reuse equal-matrix fits"),
    )

    refitted = _refit_linear_profile_candidates(
        frontier_scores=(high, low),
        linear_runs=runs,
        target_hamiltonians=target,
        kpoints=np.asarray(moire_config.kpoints),
        n_folds=3,
        low_primary_count=2,
        high_primary_count=4,
        candidate_dir=tmp_path / "uniform_linear_logs",
        verbose=False,
    )

    assert set(refitted) == {
        "high-source__linear_high_uniform",
        "low-source__linear_low_uniform",
    }
    assert all(
        run["model_config"].response_fit_objective == {"mode": "equal_matrix_v1"}
        for run in refitted.values()
    )
    assert all(run["result"] is base_result for run in refitted.values())


def test_complete_profile_refit_uses_resolved_gamma_case_envelope(
    tmp_path: Path,
) -> None:
    moire_config, model_config = build_moire_config_from_file(_write_fixture(tmp_path))
    legacy = replace(
        model_config,
        term_templates=[
            {
                "name": "gamma_kinetic",
                "source": "diagonal_kp",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": [[1, 1]],
                "max_order": 2,
                "term_space_policy": "explicit_reduced",
            }
        ],
    )

    templates = pipeline_module._complete_term_templates_for_profile_refit(
        legacy,
        moire_config=moire_config,
    )

    assert templates
    assert all(row["name"].startswith("gamma_case_") for row in templates)
    assert {row["term_space_policy"] for row in templates} == {"complete"}


@pytest.mark.parametrize(
    ("valley_type", "name_prefix"),
    [
        ("K", "k_case_"),
        ("M", "m_case_"),
    ],
)
def test_complete_profile_refit_uses_resolved_k_m_case_envelope(
    tmp_path: Path,
    valley_type: str,
    name_prefix: str,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = valley_type
    raw["valley_model"]["active_valleys"] = [valley_type]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    moire_config, model_config = build_moire_config_from_file(cfg_path)
    legacy = replace(
        model_config,
        term_templates=[
            {
                "name": "legacy_kinetic",
                "source": "diagonal_kp",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 2,
                "term_space_policy": "orbit_representative",
            }
        ],
    )

    templates = pipeline_module._complete_term_templates_for_profile_refit(
        legacy,
        moire_config=moire_config,
    )

    assert templates
    assert all(row["name"].startswith(name_prefix) for row in templates)
    assert {row["term_space_policy"] for row in templates} == {"complete"}
    assert all(row["orbital_pairs"] == "all" for row in templates)


@pytest.mark.parametrize("valley_type", ["K", "M"])
def test_complete_profile_refit_k_m_requires_resolved_case(
    tmp_path: Path,
    valley_type: str,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["valley_type"] = valley_type
    raw["valley_model"]["active_valleys"] = [valley_type]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    model_config = load_model_config(cfg_path)

    with pytest.raises(
        ValueError,
        match=rf"{valley_type} complete-response profile refit requires resolved",
    ):
        pipeline_module._complete_term_templates_for_profile_refit(model_config)


def test_reverted_refinement_baseline_remains_selectable() -> None:
    assert pipeline_module._band_refinement_result_is_selectable(
        {
            "band_refinement": {
                "acceptance_guard": {
                    "enabled": True,
                    "accepted": False,
                    "reverted": True,
                }
            }
        }
    )
    assert not pipeline_module._band_refinement_result_is_selectable(
        {
            "band_refinement": {
                "acceptance_guard": {
                    "enabled": True,
                    "accepted": False,
                    "reverted": False,
                }
            }
        }
    )


def test_selected_profile_reuses_completed_frontier_result(monkeypatch, tmp_path: Path) -> None:
    cached = {"model": object(), "eigvals": np.zeros((1, 1))}
    monkeypatch.setattr(
        pipeline_module,
        "_run_model_pipeline",
        lambda *args, **kwargs: pytest.fail("completed frontier result must be reused"),
    )

    result = pipeline_module._selected_profile_pipeline_result(
        selected_run={"result": cached},
        selected_moire=SimpleNamespace(),
        selected_model=SimpleNamespace(),
        profile_log=tmp_path / "unused.log",
        verbose=False,
        progress=False,
    )

    assert result is cached


def test_automatic_low_profile_uses_twice_projected_model_dimension() -> None:
    model_config = SimpleNamespace(n_orb=(2, 2))

    assert _automatic_low_profile_band_count(model_config, total_bands=76) == 8


def test_linear_profile_fit_objective_uses_fixed_window_and_two_sided_weight() -> None:
    low = _linear_profile_fit_objective(
        target_bands="top",
        band_count=8,
        two_sided_projector_weight=300.0,
    )
    high = _linear_profile_fit_objective(
        target_bands="top",
        band_count=28,
        two_sided_projector_weight=300.0,
    )

    assert low["mode"] == "target_spectral_linear"
    assert low["window"] == {
        "mode": "fixed_count_degeneracy_safe",
        "bands": 8,
        "degeneracy_tol_mev": 0.1,
    }
    assert low["two_sided_projector"] == {"enabled": True, "weight": 300.0}
    assert high["window"]["bands"] == 28


def test_balanced_family_order_candidates_include_compact_joint_models() -> None:
    assert _balanced_family_order_candidates(
        {
            "kinetic": (2, 4, 6, 10),
            "intra": (0, 2, 4, 6),
            "inter": (0, 2, 4, 10),
        }
    ) == (
        FamilyOrders(2, 0, 0),
        FamilyOrders(4, 2, 2),
        FamilyOrders(6, 4, 4),
    )


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
    assert selected["intra_shells"] == 2
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


def test_low_cost_harmonic_recommendation_requires_relaxed_accuracy() -> None:
    report = {
        "candidates": [
            {
                "intra_shells": 1,
                "inter_shells": 1,
                "plot_rms_mev": 2.0,
                "plot_max_mev": 4.0,
                "subspace_mean_overlap": 0.99,
            }
        ]
    }

    assert _low_cost_harmonic_recommendation_candidate(report) is None


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
        candidate_pairs=[(0, 0), (1, 0), (2, 0)],
        thresholds={"plot_rms_mev": 1.0, "plot_max_mev": 2.0, "min_overlap": 0.99},
    )

    assert [(row["intra_shells"], row["inter_shells"]) for row in report["candidates"]] == [(0, 0), (1, 0), (2, 0)]
    assert report["selected"]["intra_shells"] == 2
    assert report["selected"]["inter_shells"] == 0


def test_harmonic_ablation_precomputed_shell_maps_match_mask_stats() -> None:
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=float)
    rows = [
        {"q": q1[0], "layer": 1},
        {"q": q1[1], "layer": 1},
        {"q": q2[0], "layer": 2},
        {"q": q2[1], "layer": 2},
    ]
    shell_norms = {"intra": [1.0, 2.0], "inter": [1.0, 2.0, np.sqrt(5.0)]}

    direct_mask, direct_stats = pipeline_module._harmonic_ablation_mask(
        rows,
        shell_norms,
        intra_shells=1,
        inter_shells=2,
        tol=1.0e-8,
    )
    shell_maps = _harmonic_ablation_shell_maps(rows, shell_norms, tol=1.0e-8)
    fast_mask, fast_stats = _harmonic_ablation_mask_from_shell_maps(
        shell_maps,
        intra_shells=1,
        inter_shells=2,
    )

    np.testing.assert_array_equal(fast_mask, direct_mask)
    assert fast_stats == direct_stats


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

    report = config.automatic_harmonic_selection
    assert config.harmonics_config["intra"]["count"] == report["selected"]["intra_shells"]
    assert config.harmonics_config["inter"]["count"] == report["selected"]["inter_shells"]
    assert config.max_order["Kinect"] == 6
    assert config.max_order["intra"] == 4
    assert config.max_order["inter"] == 6
    assert config.max_order["moire_intra_zero"] == 0
    assert config.max_order["tunneling_zero"] == 6
    assert config.raw["model"]["auto_low_energy_order_profile"]["profile"] == "gamma_compact_ladder_start"
    assert config.raw["model"]["automatic_defaults"]["harmonics"] is True
    assert report["selection_profile"] == "low_cost"


def test_load_model_config_omitted_harmonics_uses_low_cost_scan_in_manual_fit_mode(tmp_path: Path) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"].pop("harmonics", None)
    data["fit"] = {"indices": [0]}
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    report = config.automatic_harmonic_selection
    assert report["selection_profile"] == "low_cost"
    assert config.harmonics_config["intra"]["count"] == report["selected"]["intra_shells"]
    assert config.harmonics_config["inter"]["count"] == report["selected"]["inter_shells"]


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
    np.save(tmp_path / "project" / "heff.npy", heff)
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)

    assert config.harmonics_config["intra"]["count"] == 2
    assert config.harmonics_config["inter"]["count"] == 0
    report = config.automatic_harmonic_selection
    assert report["selection_status"] == "selected_low_cost_relaxed_accuracy"
    assert report["selected"]["accepted"] is True
    assert config.band_refinement_config["auto_harmonic_selection"]["enabled"] is False


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
    assert config.automatic_harmonic_selection == {}
    assert config.band_refinement_config["auto_harmonic_selection"]["enabled"] is False


def test_load_model_config_does_not_print_harmonic_recommendation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"]["harmonics"] = {"intralayer": 3, "interlayer": 2}
    data["model"]["target_bands"] = "bottom"
    data["fit"] = {"mode": "auto_low_energy", "max_points": 2}
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_model_config(cfg_path)
    stdout = capsys.readouterr().out

    assert config.harmonics_config == {"intra": 3, "inter": 2}
    assert stdout == ""
    assert not (tmp_path / "model_out" / "harmonic_recommendation_bands.png").exists()


def test_run_configured_model_prints_single_compact_harmonic_recommendation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"]["harmonics"] = {"intralayer": 3, "interlayer": 2}
    data["model"]["target_bands"] = "bottom"
    data["fit"] = {
        "mode": "auto_low_energy",
        "max_points": 2,
        "harmonic_recommendation": True,
        "model_selection": False,
    }
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    result = run_configured_model(cfg_path)
    stdout = capsys.readouterr().out

    assert result["configured_model"].harmonics_config == {"intra": 3, "inter": 2}
    assert result["configured_model"].automatic_harmonic_selection == {}
    assert stdout.count("Harmonic recommendation") == 1
    assert "--- Harmonic recommendation (diagnostic only) ---" in stdout
    assert "actual model.harmonics: intra=3 inter=2 (unchanged)" in stdout
    assert "low-cost" in stdout
    assert "high-acc" in stdout
    assert "faithful recommended" not in stdout
    assert "plot " in stdout
    assert "mean subspace overlap" in stdout
    assert "band plot" in stdout
    assert (tmp_path / "model_out" / "harmonic_recommendation_bands.png").exists()


def test_run_configured_model_prints_automatic_low_cost_harmonics_without_plot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = _write_auto_fixture(tmp_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["model"].pop("harmonics", None)
    data["fit"] = {
        "method": "linear",
        "kpoints": [0],
        "bands": 2,
        "one_sided_weight": 0.0,
        "two_sided_weight": 0.0,
    }
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _write_symm_frame_manifest(
        tmp_path,
        rotation_deg=0.0,
        q_model_files={"layer1": "../q1.npy", "layer2": "../q2.npy"},
    )

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    result = run_configured_model(cfg_path)
    stdout = capsys.readouterr().out

    assert result["configured_model"].raw["model"]["automatic_defaults"]["harmonics"] is True
    assert "auto_low_energy_harmonic_selection" not in result["configured_model"].raw["model"]
    assert stdout.count("Automatic harmonic selection  PASS") == 1
    assert "Choice" in stdout
    assert "Plot RMS/Max (meV)" in stdout
    assert "Primary RMS/Max (meV)" in stdout
    assert stdout.count("selected") == 1
    assert "quality" in stdout
    assert "low-cost" in stdout
    assert "high-accuracy" in stdout
    assert "[kp model] Model setup" in stdout
    assert "basis" in stdout and "n_orb=[1, 1]" in stdout
    assert "method" in stdout and "linear" in stdout
    assert "target" in stdout and "top 2 bands" in stdout
    assert "fit k rows" in stdout and "[0] · projected Heff row indices" in stdout
    assert "H weights" in stdout and "full=1 · one-sided=0 · two-sided=0" in stdout
    assert not (tmp_path / "model_out" / "harmonic_recommendation_bands.png").exists()


def test_automatic_harmonic_selection_always_shows_high_reference(
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = {
        "intra_shells": 3,
        "inter_shells": 3,
        "plot_rms_mev": 0.842,
        "plot_max_mev": 1.721,
        "primary_rms_mev": 0.678,
        "primary_max_mev": 1.446,
        "subspace_mean_overlap": 0.999084,
    }

    _print_automatic_harmonic_selection_report(
        {"selected": candidate, "quality_selected": dict(candidate)}
    )
    stdout = capsys.readouterr().out

    assert "Automatic harmonic selection  PASS" in stdout
    assert stdout.count("selected") == 1
    assert any(line.lstrip().startswith("high") for line in stdout.splitlines())
    assert stdout.count("0.842 / 1.721") == 2
    assert stdout.count("0.678 / 1.446") == 2


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
        n_orb=(2, 0),
        nlow_state=[2, 0],
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
    support_targets: list[np.ndarray] = []

    def capture_support_target(target, **_kwargs):
        support_targets.append(np.asarray(target))
        return np.asarray(target)

    monkeypatch.setattr(pipeline, "_current_heff_support_hamiltonians", capture_support_target)

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
            "optimizer": "scipy_least_squares",
            "subspace_loss": {"enabled": False},
            "low_subspace_matrix_loss": {"enabled": False},
            "acceptance_guard": {"enabled": True, "max_rms_increase_mev": 0.0, "max_max_increase_mev": 0.0},
        },
    )

    report = pipeline.refine_band_coefficients(moire_cfg, model_cfg, model)

    assert report["acceptance_guard"]["accepted"] is False
    assert report["acceptance_guard"]["reverted"] is True
    assert report["reference"] == "current_heff_support_mask"
    assert support_targets
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
        n_orb=(2, 0),
        nlow_state=[2, 0],
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
            [[0.0, 0.4], [0.4, 1.0]],
            [[0.0, 0.4], [0.4, 1.1]],
        ],
        dtype=np.complex128,
    )
    support_heff = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.1]],
        ],
        dtype=np.complex128,
    )
    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, heff)

    def fake_hamiltonians(_moire_config, _model, _kpoints):
        return support_heff.copy()

    monkeypatch.setattr(pipeline, "_model_hamiltonians_for_kpoints", fake_hamiltonians)
    monkeypatch.setattr(
        pipeline,
        "_selected_harmonic_support_mask",
        lambda *args, **kwargs: np.eye(2, dtype=bool),
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    model_cfg = ConfiguredModel(
        path=tmp_path / "model.yaml",
        raw={"model": {"n_orb": [1, 1], "target_bands": "top"}, "fit": {"mode": "auto_low_energy"}},
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
        n_orb=(1, 1),
        nlow_state=[1, 1],
        bM_config={},
        harmonics_config={"intra": 0, "inter": 0},
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
        Q_set2=np.zeros((1, 2), dtype=float),
        n_orb1=1,
        n_orb2=1,
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
    candidate = payload["candidates"][0]
    assert candidate["reference"] == "current_heff_support_mask"
    assert candidate["windows"][0]["band"]["top_band_rms_mev"] == pytest.approx(0.0)
    assert candidate["secondary_original_heff"]["reference"] == "original_heff"
    assert candidate["secondary_original_heff"]["plot_rms_mev"] > 0.0
    md = (output_dir / "auto_model_selection.md").read_text(encoding="utf-8")
    assert "Harmonic Selection" in md
    assert "current Heff support mask" in md


def test_high_low_model_selection_outputs_write_dual_profile_frontier(tmp_path: Path) -> None:
    candidates = (
        CandidateScore(
            name="k2_i1_t1",
            orders=FamilyOrders(2, 1, 1),
            independent_real_parameters=18,
            weighted_rms_mev=0.40,
            weighted_rms_se_mev=0.05,
            weighted_max_mev=1.2,
            mean_subspace_overlap=0.98,
            expanded_weighted_rms_mev=1.0,
        ),
        CandidateScore(
            name="k1_i1_t1",
            orders=FamilyOrders(1, 1, 1),
            independent_real_parameters=8,
            weighted_rms_mev=0.47,
            weighted_rms_se_mev=0.05,
            weighted_max_mev=1.8,
            mean_subspace_overlap=0.96,
            expanded_weighted_rms_mev=4.0,
        ),
    )
    profiles = select_high_low_profiles(
        candidates,
        overlap_target=0.95,
        overlap_safety_floor=0.90,
        low_se_multiplier=2.0,
    )
    scan = {
        "profiles": profiles,
        "runs": {candidate.name: {"score": candidate} for candidate in candidates},
    }

    summary = _write_high_low_model_selection_outputs(
        scan=scan,
        output_dir=tmp_path,
    )

    assert summary["profiles"]["high"]["selected"]["name"] == "k2_i1_t1"
    assert summary["profiles"]["low"]["selected"]["name"] == "k1_i1_t1"
    assert summary["profiles"]["low"]["standard_error_multiplier"] == 2.0
    assert (tmp_path / "auto_model_selection.json").exists()
    assert (tmp_path / "auto_model_selection.md").exists()
    assert (tmp_path / "candidate_metrics.csv").exists()
    assert (tmp_path / "model_complexity_frontier.pdf").exists()
    markdown = (tmp_path / "auto_model_selection.md").read_text(encoding="utf-8")
    assert "High Accuracy" in markdown
    assert "Low Parameter" in markdown
    assert "expanded/all-band error is diagnostic only" in markdown


def test_four_profile_report_records_solver_and_complexity_metadata(tmp_path: Path) -> None:
    linear = CandidateScore(
        name="linear",
        orders=FamilyOrders(4, 2, 2),
        independent_real_parameters=40,
        weighted_rms_mev=0.4,
        weighted_rms_se_mev=0.05,
        weighted_max_mev=1.0,
        mean_subspace_overlap=0.98,
        active_group_count=20,
        harmonic_support_size=8,
        primary_band_count=10,
    )
    nonlinear = CandidateScore(
        name="nonlinear",
        orders=FamilyOrders(3, 1, 1),
        independent_real_parameters=18,
        weighted_rms_mev=0.2,
        weighted_rms_se_mev=0.03,
        weighted_max_mev=0.6,
        mean_subspace_overlap=0.97,
        solver_family="nonlinear",
        active_group_count=9,
        harmonic_support_size=4,
        primary_band_count=4,
    )
    linear_profiles = select_high_low_profiles((linear,))
    nonlinear_profiles = select_high_low_profiles((nonlinear,))
    scan = {
        "profiles": linear_profiles,
        "profile_families": {
            "linear": linear_profiles,
            "nonlinear": nonlinear_profiles,
        },
        "runs": {
            linear.name: {"score": linear},
            nonlinear.name: {"score": nonlinear},
        },
        "nonlinear_frontier": {"enabled": True, "linear_sources": ["linear"]},
    }

    summary = _write_high_low_model_selection_outputs(scan=scan, output_dir=tmp_path)

    assert summary["mode"] == "automatic_four_model_profiles"
    assert summary["profile_families"]["nonlinear"]["low"]["selected"][
        "solver_family"
    ] == "nonlinear"
    assert summary["profile_families"]["nonlinear"]["low"]["selected"][
        "active_group_count"
    ] == 9
    assert summary["profile_families"]["nonlinear"]["low"]["selected"][
        "primary_band_count"
    ] == 4
    assert summary["nonlinear_frontier"]["enabled"] is True
    csv_text = (tmp_path / "candidate_metrics.csv").read_text(encoding="utf-8")
    assert "solver_family" in csv_text.splitlines()[0]
    assert "primary_band_count" in csv_text.splitlines()[0]
    assert "nonlinear" in csv_text
    markdown = (tmp_path / "auto_model_selection.md").read_text(encoding="utf-8")
    assert "Nonlinear High" in markdown
    assert "Nonlinear Low" in markdown


def test_materialize_automatic_profile_output_writes_runnable_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    moire_config, model_config = build_moire_config_from_file(cfg_path)
    model_config = replace(
        model_config,
        response_semantics="complete_linear_v2",
        response_fit_objective={"mode": "target_spectral_linear"},
    )
    target = np.asarray(
        pipeline_module._current_heff_support_hamiltonians(
            np.load(model_config.heff_file),
            qset1=np.asarray(moire_config.Q_set1),
            qset2=np.asarray(moire_config.Q_set2),
            n_orb=model_config.n_orb,
            harmonics_config=model_config.harmonics_config,
        )
    )
    target = pipeline_module._select_rows(target, model_config.band_indices)
    eigvals, eigvecs = np.linalg.eigh(target)
    score = CandidateScore(
        name="k1_i0_t0",
        orders=FamilyOrders(1, 0, 0),
        independent_real_parameters=2,
        weighted_rms_mev=0.0,
        weighted_rms_se_mev=0.0,
        weighted_max_mev=0.0,
        mean_subspace_overlap=1.0,
    )
    result = {"model": SimpleNamespace(terms={}), "eigvals": (eigvals, eigvecs)}
    profile = {
        "results": result,
        "moire_config": moire_config,
        "model_config": model_config,
        "score": score,
        "status": "PASS",
    }
    registry_calls: list[Path] = []

    def fake_band_plot(*args, **kwargs):
        path = Path(args[2])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def fake_q_plot(*args, **kwargs):
        path = Path(kwargs["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def fake_registry(**kwargs):
        registry_calls.append(Path(kwargs["output_dir"]))
        (Path(kwargs["output_dir"]) / "active_terms.json").write_text("[]\n", encoding="utf-8")
        (Path(kwargs["output_dir"]) / "run_summary.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "save_band_comparison_plot", fake_band_plot)
    monkeypatch.setattr(pipeline_module, "save_q_lattice_harmonics_plot", fake_q_plot)
    monkeypatch.setattr(
        pipeline_module,
        "_compute_validation_outputs",
        lambda **kwargs: ({}, {"validation_incomplete": False, "missing": []}),
    )
    monkeypatch.setattr(pipeline_module, "_write_model_registry_outputs", fake_registry)
    def fake_export(model_output, output_dir, **kwargs):
        output = Path(output_dir)
        (output / "evaluate.py").write_text("# standalone\n", encoding="utf-8")
        np.savez(output / "model_data.npz", dimension_dim=np.asarray(4))
        return output

    monkeypatch.setattr("kp.model.export.export_standalone_model", fake_export)

    materialized = _materialize_automatic_profile_output(
        profile_name="low",
        profile=profile,
        target_hamiltonians=target,
        root_output_dir=tmp_path / "out",
    )

    profile_dir = tmp_path / "out" / "low"
    assert registry_calls == [profile_dir]
    assert (profile_dir / "eigvals.npy").exists()
    assert (profile_dir / "current_heff_support_eigvals.npy").exists()
    assert (profile_dir / "selected_model_config.yaml").exists()
    selected_config = yaml.safe_load(
        (profile_dir / "selected_model_config.yaml").read_text(encoding="utf-8")
    )
    assert selected_config["model"]["response_semantics"] == "complete_linear_v2"
    assert selected_config["model"]["fit"]["objective"] == {
        "mode": "target_spectral_linear"
    }
    assert (profile_dir / "band_comparison.pdf").exists()
    assert (profile_dir / "band_comparison_all.pdf").exists()
    assert materialized["comparison"]["reference"] == "current_heff_support_mask"
    assert materialized["auto_model_selection"]["profile"] == "low"


def test_run_configured_model_default_auto_writes_high_and_low_profiles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)

    def fake_scan(*, moire_config, model_config, **kwargs):
        target = np.asarray(
            pipeline_module._current_heff_support_hamiltonians(
                np.load(model_config.heff_file),
                qset1=np.asarray(moire_config.Q_set1),
                qset2=np.asarray(moire_config.Q_set2),
                n_orb=model_config.n_orb,
                harmonics_config=model_config.harmonics_config,
            )
        )
        target = pipeline_module._select_rows(target, model_config.band_indices)
        score = CandidateScore(
            name="k2_i0_t0",
            orders=FamilyOrders(2, 0, 0),
            independent_real_parameters=4,
            weighted_rms_mev=0.0,
            weighted_rms_se_mev=0.0,
            weighted_max_mev=0.0,
            mean_subspace_overlap=1.0,
        )
        profiles = select_high_low_profiles(
            (score,),
            overlap_target=0.95,
            overlap_safety_floor=0.90,
            low_se_multiplier=2.0,
        )
        return {
            "profiles": profiles,
            "runs": {
                score.name: {
                    "score": score,
                    "model_config": model_config,
                    "moire_config": moire_config,
                }
            },
            "target_hamiltonians": target,
            "staged_selection": None,
            "correction_sweep": None,
        }

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        target = np.asarray(
            pipeline_module._current_heff_support_hamiltonians(
                np.load(model_config.heff_file),
                qset1=np.asarray(moire_config.Q_set1),
                qset2=np.asarray(moire_config.Q_set2),
                n_orb=model_config.n_orb,
                harmonics_config=model_config.harmonics_config,
            )
        )
        target = pipeline_module._select_rows(target, model_config.band_indices)
        eigvals, eigvecs = np.linalg.eigh(target)
        return {
            "model": SimpleNamespace(terms={}),
            "eigvals": (eigvals, eigvecs),
            "band_hamiltonians": target,
            "diagnostics": {},
            "band_refinement": {"enabled": False},
        }

    monkeypatch.setattr(
        pipeline_module,
        "_run_automatic_family_order_scan",
        fake_scan,
    )
    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)
    def fake_export(model_output, output_dir, **kwargs):
        output = Path(output_dir)
        (output / "evaluate.py").write_text("# standalone\n", encoding="utf-8")
        np.savez(output / "model_data.npz", dimension_dim=np.asarray(4))
        return output

    monkeypatch.setattr("kp.model.export.export_standalone_model", fake_export)
    monkeypatch.setattr(
        pipeline_module,
        "_compute_validation_outputs",
        lambda **kwargs: ({}, {"validation_incomplete": False, "missing": []}),
    )

    results = run_configured_model(cfg_path)

    output_dir = results["configured_model"].output_dir
    assert results["auto_model_selection"]["mode"] == "automatic_high_low_family_orders"
    for profile_name in ("high", "low"):
        profile_dir = output_dir / profile_name
        assert (profile_dir / "eigvals.npy").exists()
        assert (profile_dir / "active_terms.json").exists()
        assert (profile_dir / "run_summary.json").exists()
        assert (profile_dir / "selected_model_config.yaml").exists()
        assert results["profile_results"][profile_name]["results"]["auto_model_selection"]["profile"] == profile_name


def test_run_configured_model_writes_four_linear_nonlinear_profiles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)

    def fake_scan(*, moire_config, model_config, **kwargs):
        target = np.asarray(
            pipeline_module._current_heff_support_hamiltonians(
                np.load(model_config.heff_file),
                qset1=np.asarray(moire_config.Q_set1),
                qset2=np.asarray(moire_config.Q_set2),
                n_orb=model_config.n_orb,
                harmonics_config=model_config.harmonics_config,
            )
        )
        target = pipeline_module._select_rows(target, model_config.band_indices)
        linear = CandidateScore(
            name="linear",
            orders=FamilyOrders(2, 0, 0),
            independent_real_parameters=4,
            weighted_rms_mev=0.1,
            weighted_rms_se_mev=0.01,
            weighted_max_mev=0.2,
            mean_subspace_overlap=1.0,
        )
        nonlinear = CandidateScore(
            name="nonlinear",
            orders=FamilyOrders(2, 0, 0),
            independent_real_parameters=4,
            weighted_rms_mev=0.05,
            weighted_rms_se_mev=0.01,
            weighted_max_mev=0.1,
            mean_subspace_overlap=1.0,
            solver_family="nonlinear",
        )
        linear_profiles = select_high_low_profiles((linear,))
        nonlinear_profiles = select_high_low_profiles((nonlinear,))
        return {
            "profiles": linear_profiles,
            "profile_families": {
                "linear": linear_profiles,
                "nonlinear": nonlinear_profiles,
            },
            "runs": {
                linear.name: {
                    "score": linear,
                    "model_config": replace(
                        model_config,
                        band_refinement_config={"enabled": False},
                    ),
                    "moire_config": moire_config,
                },
                nonlinear.name: {
                    "score": nonlinear,
                    "model_config": replace(
                        model_config,
                        band_refinement_config={"enabled": True},
                    ),
                    "moire_config": moire_config,
                },
            },
            "target_hamiltonians": target,
            "staged_selection": None,
            "correction_sweep": None,
            "nonlinear_frontier": {"enabled": True},
        }

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        target = np.asarray(
            pipeline_module._current_heff_support_hamiltonians(
                np.load(model_config.heff_file),
                qset1=np.asarray(moire_config.Q_set1),
                qset2=np.asarray(moire_config.Q_set2),
                n_orb=model_config.n_orb,
                harmonics_config=model_config.harmonics_config,
            )
        )
        target = pipeline_module._select_rows(target, model_config.band_indices)
        eigvals, eigvecs = np.linalg.eigh(target)
        return {
            "model": SimpleNamespace(terms={}),
            "eigvals": (eigvals, eigvecs),
            "band_hamiltonians": target,
            "diagnostics": {},
            "band_refinement": dict(model_config.band_refinement_config),
        }

    monkeypatch.setattr(pipeline_module, "_run_automatic_family_order_scan", fake_scan)
    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)
    def fake_registry(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "active_terms.json").write_text("[{}]\n", encoding="utf-8")
        (output / "run_summary.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "_write_model_registry_outputs", fake_registry)
    def fake_export(model_output, output_dir, **kwargs):
        output = Path(output_dir)
        (output / "evaluate.py").write_text("# standalone\n", encoding="utf-8")
        np.savez(output / "model_data.npz", dimension_dim=np.asarray(4))
        return output

    monkeypatch.setattr("kp.model.export.export_standalone_model", fake_export)
    monkeypatch.setattr(
        pipeline_module,
        "_compute_validation_outputs",
        lambda **kwargs: ({}, {"validation_incomplete": False, "missing": []}),
    )

    results = run_configured_model(cfg_path)

    output_dir = tmp_path / "model_out"
    assert results["auto_model_selection"]["mode"] == "automatic_four_model_profiles"
    for solver_family in ("linear", "nonlinear"):
        for quality in ("high", "low"):
            profile_dir = output_dir / solver_family / quality
            assert (profile_dir / "eigvals.npy").exists()
            assert (profile_dir / "active_terms.json").exists()
            assert (profile_dir / "run_summary.json").exists()
            assert (profile_dir / "evaluate.py").exists()
            assert (profile_dir / "model_data.npz").exists()
            profile_result = results["profile_results"][solver_family][quality]["results"]
            assert profile_result["auto_model_selection"]["profile"] == (
                f"{solver_family}/{quality}"
            )
    assert results["configured_model"].output_dir == output_dir / "nonlinear" / "high"


@pytest.mark.parametrize(
    ("requested", "expected_primary"),
    [
        (("linear/low", "linear/high"), "linear/high"),
        (("nonlinear/high",), "nonlinear/high"),
    ],
)
def test_run_configured_model_materializes_only_requested_profile_outputs(
    monkeypatch,
    tmp_path: Path,
    requested: tuple[str, ...],
    expected_primary: str,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["profiles"] = list(requested)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    def fake_scan(*, moire_config, model_config, **kwargs):
        target = np.asarray(
            pipeline_module._current_heff_support_hamiltonians(
                np.load(model_config.heff_file),
                qset1=np.asarray(moire_config.Q_set1),
                qset2=np.asarray(moire_config.Q_set2),
                n_orb=model_config.n_orb,
                harmonics_config=model_config.harmonics_config,
            )
        )
        target = pipeline_module._select_rows(target, model_config.band_indices)
        linear = CandidateScore(
            name="linear",
            orders=FamilyOrders(2, 0, 0),
            independent_real_parameters=4,
            weighted_rms_mev=0.1,
            weighted_rms_se_mev=0.01,
            weighted_max_mev=0.2,
            mean_subspace_overlap=1.0,
        )
        nonlinear = replace(
            linear,
            name="nonlinear",
            solver_family="nonlinear",
            weighted_rms_mev=0.05,
        )
        linear_profiles = select_high_low_profiles((linear,))
        nonlinear_profiles = select_high_low_profiles((nonlinear,))
        return {
            "profiles": linear_profiles,
            "profile_families": {
                "linear": linear_profiles,
                "nonlinear": nonlinear_profiles,
            },
            "runs": {
                "linear": {
                    "score": linear,
                    "model_config": replace(
                        model_config,
                        band_refinement_config={"enabled": False},
                    ),
                    "moire_config": moire_config,
                },
                "nonlinear": {
                    "score": nonlinear,
                    "model_config": replace(
                        model_config,
                        band_refinement_config={"enabled": True},
                    ),
                    "moire_config": moire_config,
                },
            },
            "target_hamiltonians": target,
            "staged_selection": None,
            "correction_sweep": None,
            "nonlinear_frontier": {"enabled": True},
        }

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        target = np.asarray(
            pipeline_module._current_heff_support_hamiltonians(
                np.load(model_config.heff_file),
                qset1=np.asarray(moire_config.Q_set1),
                qset2=np.asarray(moire_config.Q_set2),
                n_orb=model_config.n_orb,
                harmonics_config=model_config.harmonics_config,
            )
        )
        target = pipeline_module._select_rows(target, model_config.band_indices)
        eigvals, eigvecs = np.linalg.eigh(target)
        return {
            "model": SimpleNamespace(terms={}),
            "eigvals": (eigvals, eigvecs),
            "band_hamiltonians": target,
            "diagnostics": {},
            "band_refinement": dict(model_config.band_refinement_config),
        }

    monkeypatch.setattr(pipeline_module, "_run_automatic_family_order_scan", fake_scan)
    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)

    def fake_registry(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "active_terms.json").write_text("[{}]\n", encoding="utf-8")
        (output / "run_summary.json").write_text("{}\n", encoding="utf-8")

    def fake_export(model_output, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluate.py").write_text("# standalone\n", encoding="utf-8")
        np.savez(output / "model_data.npz", dimension_dim=np.asarray(4))
        return output

    monkeypatch.setattr(pipeline_module, "_write_model_registry_outputs", fake_registry)
    monkeypatch.setattr("kp.model.export.export_standalone_model", fake_export)
    monkeypatch.setattr(
        pipeline_module,
        "_compute_validation_outputs",
        lambda **kwargs: ({}, {"validation_incomplete": False, "missing": []}),
    )

    results = run_configured_model(cfg_path)

    output_dir = tmp_path / "model_out"
    supported = {
        "linear/low",
        "linear/high",
        "nonlinear/low",
        "nonlinear/high",
    }
    for profile_name in supported:
        profile_dir = output_dir.joinpath(*profile_name.split("/"))
        if profile_name in requested:
            assert (profile_dir / "evaluate.py").is_file()
            assert (profile_dir / "model_data.npz").is_file()
        else:
            assert not profile_dir.exists()
    assert results["configured_model"].output_dir == output_dir.joinpath(
        *expected_primary.split("/")
    )


def test_auto_fit_candidate_scan_uses_current_support_as_primary_target(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    heff_path = tmp_path / "project" / "heff.npy"
    original_heff = np.load(heff_path)
    original_heff[:, 1, 3] = np.array([0.35, 0.40, 0.45])
    original_heff[:, 3, 1] = original_heff[:, 1, 3]
    np.save(heff_path, original_heff)
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)
    moire_config, model_config = build_moire_config_from_file(cfg_path)
    support_mask = np.eye(original_heff.shape[-1], dtype=bool)
    support_heff = original_heff * support_mask[None, :, :]
    support_eigvals = np.linalg.eigvalsh(support_heff)
    raw = dict(model_config.raw)
    raw["fit"] = {
        "mode": "auto_low_energy",
        "candidate_scan": True,
        "disable_residual_augmented_fit": True,
    }
    model_config = replace(
        model_config,
        raw=raw,
        fit_selection_metadata={
            "mode": "auto_low_energy",
            "fit_candidate_sets": [
                {"name": "center", "indices": [0]},
                {"name": "ends", "indices": [0, 2]},
            ],
        },
    )
    seen_fit_heff: list[np.ndarray] = []

    def fake_pipeline(candidate_moire, *_args, **_kwargs):
        seen_fit_heff.append(np.asarray(candidate_moire.heff))
        return {"eigvals": support_eigvals, "diagnostics": {}}

    monkeypatch.setattr(pipeline_module, "_run_model_pipeline", fake_pipeline)
    monkeypatch.setattr(
        pipeline_module,
        "_selected_harmonic_support_mask",
        lambda *args, **kwargs: support_mask,
    )

    scan = pipeline_module._run_auto_low_energy_fit_candidate_scan(
        moire_config=moire_config,
        model_config=model_config,
        output_dir=tmp_path / "scan",
        log_path=tmp_path / "scan" / "selected.log",
        verbose=False,
        progress=False,
    )

    assert scan is not None
    selected_results, selected_moire, _selected_model = scan
    report = selected_results["auto_fit_candidate_scan"]
    assert report["reference"] == "current_heff_support_mask"
    assert report["secondary_reference"] == "original_heff"
    for candidate in report["candidates"]:
        assert candidate["reference"] == "current_heff_support_mask"
        assert candidate["plot_rms_mev"] == pytest.approx(0.0)
        assert candidate["secondary_original_heff"]["plot_rms_mev"] > 0.0
    assert seen_fit_heff
    assert np.max(np.abs(np.asarray(selected_moire.heff)[1, 3])) == pytest.approx(0.0)


def test_auto_low_energy_without_candidate_scan_still_uses_current_support_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"] = {
        "mode": "auto_low_energy",
        "candidate_scan": False,
        "max_points": 2,
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    heff_path = tmp_path / "project" / "heff.npy"
    original_heff = np.load(heff_path)
    original_heff[:, 1, 3] = np.array([0.35, 0.40, 0.45])
    original_heff[:, 3, 1] = original_heff[:, 1, 3]
    np.save(heff_path, original_heff)
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)
    support_mask = np.eye(original_heff.shape[-1], dtype=bool)
    support_heff = original_heff * support_mask[None, :, :]
    seen_window_targets: list[np.ndarray] = []
    real_windows = pipeline_module._auto_low_energy_windows

    def capture_windows(target_eigvals, **kwargs):
        seen_window_targets.append(np.asarray(target_eigvals))
        return real_windows(target_eigvals, **kwargs)

    monkeypatch.setattr(pipeline_module, "_auto_low_energy_windows", capture_windows)
    monkeypatch.setattr(
        pipeline_module,
        "_selected_harmonic_support_mask",
        lambda *args, **kwargs: support_mask,
    )

    moire_config, model_config = build_moire_config_from_file(cfg_path)

    assert seen_window_targets
    np.testing.assert_allclose(seen_window_targets[-1], np.linalg.eigvalsh(support_heff))
    assert model_config.band_refinement_config["reference"] == "current_heff_support_mask"
    assert model_config.band_refinement_config["auto_windows"]["reference"] == (
        "current_heff_support_mask"
    )
    fit_target = np.asarray(moire_config.heff)
    assert fit_target[1, 3] == pytest.approx(0.0)
    assert fit_target[5, 7] == pytest.approx(0.0)


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


def test_missing_fit_section_defaults_to_auto_low_energy(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.pop("fit", None)
    if isinstance(raw.get("model"), dict):
        raw["model"].pop("fit", None)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.fit_selection_metadata["mode"] == "auto_low_energy"
    assert cfg.fit_indices
    assert cfg.raw["model"]["automatic_defaults"]["fit"] is True


def test_load_model_config_parses_requested_profiles_and_uniform_fit_weighting(
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["profiles"] = ["linear/low", "linear/high"]
    raw["fit"] = {
        "mode": "auto_low_energy",
        "weighting": "uniform",
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.requested_profiles == ("linear/low", "linear/high")
    assert cfg.fit_weighting == "uniform"
    assert cfg.response_fit_objective == {"mode": "equal_matrix_v1"}
    assert cfg.band_refinement_config["weighted_band_loss"]["enabled"] is False


@pytest.mark.parametrize(
    ("profiles", "match"),
    [
        ([], "model.profiles must not be empty"),
        (["linear/high", "linear/high"], "model.profiles contains duplicate"),
        (["linear/high", "quadratic/ultra"], "unsupported model profile"),
    ],
)
def test_load_model_config_rejects_invalid_requested_profiles(
    tmp_path: Path,
    profiles: list[str],
    match: str,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["profiles"] = profiles
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_model_config(cfg_path)


def test_load_model_config_rejects_unknown_fit_weighting(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["weighting"] = "mystery"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="fit.weighting"):
        load_model_config(cfg_path)


def test_model_target_edge_defaults_from_project_plot_target(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["plot"]["target"] = "conduction"
    raw["model"].pop("target_bands", None)
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.raw["model"]["target_bands"] == "bottom"


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


def test_load_model_config_reads_null_channel_abs_tol(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["null_channel_abs_tol"] = 1.0e-10
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.null_channel_abs_tol == pytest.approx(1.0e-10)


def test_load_model_config_rejects_negative_null_channel_abs_tol(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["null_channel_abs_tol"] = -1.0e-10
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="null_channel_abs_tol"):
        load_model_config(cfg_path)


def test_load_model_config_reads_null_channel_rel_tol(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["null_channel_rel_tol"] = 1.0e-3
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.null_channel_rel_tol == pytest.approx(1.0e-3)


def test_load_model_config_rejects_negative_null_channel_rel_tol(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["fit"]["null_channel_rel_tol"] = -1.0e-3
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="null_channel_rel_tol"):
        load_model_config(cfg_path)


def test_load_model_config_infers_n_orb_and_safe_defaults(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["valley_model"]["allowed_internal_symmetries"] = []
    raw["project"]["nlow_state_list"] = [[0], [0]]
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
    assert cfg.raw["model"]["n_orb_resolution"]["source"] == "source_project"
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
    raw["material"]["num_layer_list"] = [1, 2]
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


def test_load_model_config_preserves_gamma_source_qset_totals_for_1plus2_model(
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["material"]["num_layer_list"] = [1, 2]
    raw["model"]["n_orb"] = [0, 2, 2]
    raw["model"]["nlow_state"] = [0, 2, 2]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (0, 4)
    assert cfg.nlow_state == [0, 4]
    assert cfg.orbital_count_metadata["n_orb"]["resolved_model_sectors"] == [0, 4]
    assert cfg.term_template_metadata["profiles"] == ["gamma_default"]


def test_load_model_config_resolves_layerwise_nlow_state_independently(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["material"]["num_layer_list"] = [1, 2]
    raw["model"]["n_orb"] = [2, 2, 0]
    raw["model"]["nlow_state"] = [0, 2, 2]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_model_config(cfg_path)

    assert cfg.n_orb == (2, 2)
    assert cfg.nlow_state == [0, 4]
    assert cfg.orbital_count_metadata["n_orb"]["resolved_qset"] == [2, 2]
    assert cfg.orbital_count_metadata["nlow_state"]["resolved_qset"] == [0, 4]
    assert cfg.orbital_count_metadata["nlow_state"]["resolved_model_sectors"] == [0, 4]


def test_load_model_config_rejects_legacy_two_entry_n_orb_with_num_layer_list(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["material"]["num_layer_list"] = [1, 2]
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

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


def test_ptse2_release_config_exposes_only_public_model_knobs() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "ptse2_7.34"
        / "kp"
        / "configs"
        / "ptse2_7.34_Gamma_spinful_q04.yaml"
    )
    if not path.exists():
        pytest.skip("PtSe2 release example is not present in this checkout")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    symmetry = raw.get("symmetry", {})
    assert "exactification" not in symmetry
    assert "operation_actions" not in symmetry
    assert "spin_sector_sewing" not in symmetry
    model = raw.get("model", {})
    assert model["harmonics"] == {"intralayer": 5, "interlayer": 5}
    assert model["max_order"] == {
        "kinetic": 10,
        "intralayer": 8,
        "interlayer": 8,
    }
    assert "max_derivative_order" not in model
    fit = model.get("fit", {})
    assert fit == {
        "method": "nonlinear",
        "kpoints": [0, 20, 40],
        "band_kpoints": [0, 20, 40],
        "bands": 38,
        "one_sided_weight": 1.0,
        "two_sided_weight": 1.0,
        "band_loss_weight": 1.0,
    }


def test_load_model_config_marks_symmetry_source_inferred_from_case_symm_section(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    symm_dir = _write_symm_frame_manifest(tmp_path, rotation_deg=0.0, path_name="symm_from_source")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.pop("symmetry_source", None)
    raw["symm"] = {"output_dir": "symm_from_source"}
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


def test_default_gamma_templates_use_generic_onsite_fallback_for_arbitrary_orbital_count() -> None:
    max_order = _max_derivative_order_values(
        {"max_order": {"Kinect": 4, "intra": 0, "inter": 0}}
    )

    templates = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(4, 4),
        max_order=max_order,
        harmonic_counts={"intra": 0, "inter": 0},
    )

    assert [row["name"] for row in templates] == [
        "gamma_generic_kinetic",
        "gamma_generic_onsite",
        "gamma_generic_intra",
        "gamma_generic_inter",
    ]
    by_name = {row["name"]: row for row in templates}
    assert by_name["gamma_generic_kinetic"] == {
        "name": "gamma_generic_kinetic",
        "source": "diagonal_kp",
        "sector_pairs": [[1, 1], [2, 2]],
        "orbital_pairs": "diagonal",
        "term_space_policy": "complete",
        "max_order": 4,
    }
    assert by_name["gamma_generic_onsite"] == {
        "name": "gamma_generic_onsite",
        "source": "onsite",
        "sector_pairs": [[1, 1], [2, 2]],
        "orbital_pairs": "diagonal",
        "term_space_policy": "complete",
        "max_order": 0,
    }
    assert by_name["gamma_generic_intra"]["harmonics"] == "intra"
    assert by_name["gamma_generic_inter"]["harmonics"] == "inter"

    metadata = _default_term_template_profile_metadata(
        valley_model={"valley_type": "Gamma"},
        n_orb=(4, 4),
        symmetry_operations=[],
    )
    assert metadata["profiles"] == ["gamma_default"]


def test_exact_gamma_profiles_do_not_also_include_generic_fallback() -> None:
    templates = _default_term_templates_for_model(
        valley_model={"valley_type": "Gamma"},
        n_orb=(2, 2),
        max_order={"Kinect": 4, "intra": 0, "inter": 0},
    )

    names = {row["name"] for row in templates}
    assert "gamma_kinetic" in names
    assert not any(name.startswith("gamma_generic_") for name in names)


def test_core_default_templates_include_onsite_safety_fallback() -> None:
    templates = _support_grouping_builder()._default_term_templates()

    assert [row["source"] for row in templates] == [
        "diagonal_kp",
        "onsite",
        "moire_potential",
        "tunneling",
    ]


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


def test_default_k_inter_template_uses_supported_raw_q_direction(tmp_path: Path) -> None:
    cfg_path = _write_k_inter_direction_fixture(tmp_path)

    moire_cfg, _model_cfg = build_moire_config_from_file(cfg_path)
    model = build_model(moire_cfg)

    inter_terms = [
        term
        for term in model.terms.values()
        if (term.registry_metadata or {}).get("term_kind") == "inter"
    ]
    assert inter_terms
    support = sum(
        int(np.count_nonzero(np.abs(term.Y_basis(np.array([0.0, 0.0], dtype=float))) > 1.0e-14))
        for term in inter_terms
    )
    assert support > 0
    assert {(term.registry_metadata or {})["sector_pair"][0] for term in inter_terms} == {"L1"}
    assert {(term.registry_metadata or {})["sector_pair"][1] for term in inter_terms} == {"L2"}


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


def test_kp_symm_exactification_infers_projected_basis_support_by_default() -> None:
    exact_cfg = _kp_symm_exactification_config()

    assert exact_cfg["discover_action_candidates"] is True
    assert exact_cfg["accept_support_resolved_action"] is True
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


def test_kp_symm_gamma_exactification_default_uses_auto_c3_support() -> None:
    exact_cfg = _merged_exactification_overrides("Gamma", None, symmetry_tolerance=2.0e-2)

    assert exact_cfg["reject_if_off_support_rel_gt"] == pytest.approx(2.0e-2)
    assert exact_cfg["operations"] == {
        "C3z": {"support_mode": "auto", "algebraic_template": "auto"}
    }


def test_kp_symm_explicit_exactification_override_wins_over_symmetry_tolerance() -> None:
    exact_cfg = _merged_exactification_overrides(
        "Gamma",
        {"reject_if_off_support_rel_gt": 3.0e-3},
        symmetry_tolerance=2.0e-2,
    )

    assert exact_cfg["reject_if_off_support_rel_gt"] == pytest.approx(3.0e-3)


def test_kp_symm_public_operation_action_is_lowered_to_internal_candidate() -> None:
    tr_action = {
        "antiunitary": True,
        "k_map": {"type": "negation", "in_model_frame": True},
        "q_map": {"type": "negation", "in_model_frame": True},
        "sector_map": "layer_exchange",
    }

    exact_cfg = _merged_exactification_overrides(
        "Gamma",
        None,
        symmetry_tolerance=2.0e-2,
        operation_actions={"TR": tr_action},
    )

    assert exact_cfg["operations"]["TR"] == {"action_candidates": [tr_action]}
    assert exact_cfg["operations"]["C3z"] == {
        "support_mode": "auto",
        "algebraic_template": "auto",
    }


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
    (symm_dir / "representations.npz").unlink()

    with pytest.raises(ValueError, match="symmetry manifest frame rotation"):
        load_model_config(cfg_path)


def test_build_moire_config_loads_fit_block_and_band_kpoints(tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)

    moire_cfg, model_cfg = build_moire_config_from_file(cfg_path)

    assert model_cfg.heff_file == tmp_path / "project" / "heff.npy"
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
    np.save(out_dir / "heff.npy", heff)
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "case": {"profile": "test", "q_shell": "q00", "output_root": "."},
                "material": {"qset1_file": "q1.npy", "qset2_file": "q2.npy"},
                "plot": {},
                "project": {"out_dir": "project"},
                "symm": {"output_dir": "symm"},
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

    def empty_solver(self, initial_vectors, target_vector, *, tol=1.0e-6, **_kwargs):
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


def test_coefficient_fit_uses_full_dense_symmetry_responses() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    angle = 2.0 * np.pi / 3.0
    c3_block = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.complex128,
    )
    c3_matrix = np.block(
        [
            [c3_block, np.zeros((2, 2), dtype=np.complex128)],
            [np.zeros((2, 2), dtype=np.complex128), c3_block],
        ]
    )

    class DenseC3Generator:
        def get_operator(self, name, params=None):
            assert name == "C3z"
            power = 1 if params is None else int(params)
            return np.linalg.matrix_power(c3_matrix, power)

    c3_op = {
        "name": "C3z",
        "antiunitary": False,
        "k_map": {"type": "rotation", "angle_deg": 120.0},
        "q_map": {"type": "rotation", "angle_deg": 120.0},
        "sector_map": "identity",
    }
    builder = ContinuumModelBuilder(
        q,
        np.zeros((0, 2), dtype=float),
        4,
        0,
        np.array([1.0, 0.0]),
        np.array([0.5, np.sqrt(3.0) / 2.0]),
        {},
        {},
        {"Onsite": 0},
        DenseC3Generator(),
        {"Onsite": [c3_op]},
        term_templates=[
            {
                "name": "onsite",
                "source": "onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
            }
        ],
    )
    builder.build_terms()
    builder.null_channel_rel_tol = 1.0e-12
    kpoint = np.array([0.13, -0.07], dtype=float)
    known_coefficients = [0.7, -0.2, 1.1, -0.4]
    for term, coefficient in zip(builder.model.terms.values(), known_coefficients):
        term.active = True
        term.r_value_real = coefficient
        term.r_value_imag = 0.0
    target = builder.model.assemble_hamiltonian(kpoint, builder.symmetry_gen)

    for term in builder.model.terms.values():
        term.active = True
        term.r_value_real = None
        term.r_value_imag = None
    builder.compute_coefficients_by_tag(
        target,
        np.asarray([kpoint]),
        tol=1.0e-12,
    )
    reconstructed = builder.model.assemble_hamiltonian(kpoint, builder.symmetry_gen)

    np.testing.assert_allclose(reconstructed, target, atol=1.0e-11, rtol=0.0)
    np.testing.assert_allclose(
        np.diag(reconstructed),
        np.diag(target),
        atol=1.0e-11,
        rtol=0.0,
    )
    assert np.trace(reconstructed) == pytest.approx(np.trace(target), abs=1.0e-11)


def test_coefficient_fit_custom_terms_fall_back_to_full_dense_responses() -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    angle = 2.0 * np.pi / 3.0
    c3_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.complex128,
    )

    class DenseC3Generator:
        def get_operator(self, name, params=None):
            assert name == "C3z"
            power = 1 if params is None else int(params)
            return np.linalg.matrix_power(c3_matrix, power)

    c3_op = {
        "name": "C3z",
        "antiunitary": False,
        "k_map": {"type": "rotation", "angle_deg": 120.0},
        "q_map": {"type": "rotation", "angle_deg": 120.0},
        "sector_map": "identity",
    }
    builder = ContinuumModelBuilder(
        q,
        np.zeros((0, 2), dtype=float),
        2,
        0,
        np.array([1.0, 0.0]),
        np.array([0.5, np.sqrt(3.0) / 2.0]),
        {},
        {},
        {"Onsite": 0},
        DenseC3Generator(),
        {"Onsite": [c3_op]},
    )
    for orbital in (1, 2):
        key = ContinuumTermKey(0, 0, 1, 1, orbital, orbital, (0.0, 0.0))
        projector = np.zeros((2, 2), dtype=np.complex128)
        projector[orbital - 1, orbital - 1] = 1.0
        builder.model.add_term(
            key,
            lambda _k, matrix=projector: matrix.copy(),
            tag="Onsite",
            symmetry_ops=[c3_op],
        )
    kpoint = np.array([0.09, 0.04], dtype=float)
    for term, coefficient in zip(builder.model.terms.values(), [0.7, -0.2]):
        term.active = True
        term.r_value_real = coefficient
        term.r_value_imag = 0.0
    target = builder.model.assemble_hamiltonian(kpoint, builder.symmetry_gen)

    for term in builder.model.terms.values():
        term.r_value_real = None
        term.r_value_imag = None
    builder.compute_coefficients_by_tag(target, np.asarray([kpoint]), tol=1.0e-12)
    reconstructed = builder.model.assemble_hamiltonian(kpoint, builder.symmetry_gen)

    np.testing.assert_allclose(reconstructed, target, atol=1.0e-11, rtol=0.0)


def test_production_response_components_follow_actual_sparse_support() -> None:
    response = model_core.sparse.csc_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 4.0],
            ],
            dtype=np.complex128,
        )
    )

    components = ContinuumModelBuilder._response_column_components(response)

    assert [component.tolist() for component in components] == [[0], [1], [2, 3]]


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

    def empty_solver(self, initial_vectors, target_vector, *, tol=1.0e-6, **_kwargs):
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
    _disable_model_selection_for_mock_pipeline(cfg_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)

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
    _disable_model_selection_for_mock_pipeline(cfg_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)
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
    _disable_model_selection_for_mock_pipeline(cfg_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)
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
    _disable_model_selection_for_mock_pipeline(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "debug"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_eigvals = _expected_project_eigvals(tmp_path)

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
    expected_eigvals = _expected_project_eigvals(tmp_path)
    fake_model = type("Model", (), {"terms": {}})()
    compiled_runtime = object()

    def fake_build_model(moire_config):
        print("very noisy coefficient dump")
        return fake_model

    def fake_compute_coefficients(moire_config, model, *, progress_callback=None):
        print("very noisy fit dump")
        return model, {}

    def fake_compute_bands(
        moire_config,
        model,
        kpoints,
        return_eigvecs=False,
        hamiltonians_out=None,
        compiled_runtime_out=None,
    ):
        print("very noisy coefficient dump")
        if hamiltonians_out is not None:
            hamiltonians_out.extend(np.diag(row).astype(complex) for row in expected_eigvals)
        assert compiled_runtime_out is not None
        compiled_runtime_out.append(compiled_runtime)
        return expected_eigvals

    monkeypatch.setattr("kp.model.pipeline.build_model", fake_build_model)
    monkeypatch.setattr("kp.model.pipeline.compute_coefficients", fake_compute_coefficients)
    monkeypatch.setattr("kp.model.pipeline.compute_bands", fake_compute_bands)

    results = run_configured_model(cfg_path)
    captured = capsys.readouterr()

    assert "very noisy coefficient dump" not in captured.out
    assert "[kp model] automatic family-order scan selected" in captured.out
    log_path = Path(results["model_log"])
    assert log_path.exists()
    assert "very noisy coefficient dump" in log_path.read_text(encoding="utf-8")
    assert "very noisy fit dump" in log_path.read_text(encoding="utf-8")
    assert results["compiled_operator_runtime"] is compiled_runtime


def test_run_model_pipeline_prints_term_summary_and_fit_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    key_onsite = ContinuumTermKey(0, 0, 1, 1, 1, 1, (0.0, 0.0))
    key_inter = ContinuumTermKey(1, 0, 2, 1, 1, 1, (0.0, 0.0))
    terms = {
        key_onsite: SimpleNamespace(tag="Onsite", active=True, r_value_real=0.0, r_value_imag=0.0),
        key_inter: SimpleNamespace(tag="inter", active=True, r_value_real=0.0, r_value_imag=0.0),
    }
    fake_model = SimpleNamespace(terms=terms)
    expected_eigvals = np.array([[0.0, 1.0]], dtype=float)
    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((1, 2), dtype=float),
        n_orb1=1,
        n_orb2=1,
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        kpoints=np.array([[0.0, 0.0]], dtype=float),
        kpoints_fit=np.array([[0.0, 0.0]], dtype=float),
        heff=np.diag([0.0, 1.0])[None, :, :].astype(np.complex128),
    )
    model_cfg = SimpleNamespace(
        null_channel_abs_tol=0.0,
        null_channel_rel_tol=0.0,
        coeff_prune_threshold=0.0,
        band_refinement_config={},
        compare_to_heff=False,
    )

    def fake_compute_coefficients(moire_config, model, *, progress_callback=None):
        assert progress_callback is not None
        progress_callback("fit block 1/1 tag=Onsite terms=1 start", state="start")
        progress_callback("fit block 1/1 done in 0.01 s | variables=2 support_components=1 skipped_empty=0", state="done")
        model.terms[key_inter].active = False
        model._compiled_response_basis = SimpleNamespace(channels=("c0", "c1", "c2"))
        model._fitted_response_model = SimpleNamespace(
            coefficients=np.array([1.0, 0.0, 2.0]),
            nonzero_channel_ids=("c0", "c2"),
            fit_objective={},
        )
        return model, {}

    def fake_compute_bands(moire_config, model, kpoints, return_eigvecs=False, hamiltonians_out=None, compiled_runtime_out=None):
        if hamiltonians_out is not None:
            hamiltonians_out.append(np.diag(expected_eigvals[0]).astype(complex))
        if compiled_runtime_out is not None:
            compiled_runtime_out.append(None)
        return expected_eigvals

    monkeypatch.setattr("kp.model.pipeline.build_model", lambda _moire_config: fake_model)
    monkeypatch.setattr("kp.model.pipeline.compute_coefficients", fake_compute_coefficients)
    monkeypatch.setattr("kp.model.pipeline.compute_bands", fake_compute_bands)

    _run_model_pipeline(moire_cfg, model_cfg, tmp_path / "model.log", verbose=False, progress=True)
    stdout = capsys.readouterr().out

    assert "[kp model] Continuum basis" in stdout
    assert "Family  Terms  Components" in stdout
    assert "Onsite" in stdout
    assert "inter" in stdout
    basis_lines = stdout.split("[kp model] Continuum basis", maxsplit=1)[1].splitlines()
    onsite_index = next(index for index, line in enumerate(basis_lines) if "Onsite" in line)
    inter_index = next(index for index, line in enumerate(basis_lines) if "inter" in line)
    assert onsite_index < inter_index
    assert "[kp model] Fitted continuum basis" in stdout
    assert "Family  Seed terms  Active terms" in stdout
    assert "active semantic terms" in stdout
    assert "1 / 2" in stdout
    assert "independent coefficients  3" in stdout
    assert "nonzero coefficients      2" in stdout
    assert "Onsite: terms=1, fit_components=2" not in stdout
    assert "inter: terms=1, fit_components=2" not in stdout
    assert "fit block 1/1 tag=Onsite terms=1 start" in stdout
    assert "fit block 1/1 done in 0.01 s | variables=2 support_components=1 skipped_empty=0" in stdout
    assert "variables=0 support=0" not in stdout


def test_run_model_pipeline_exposes_output_directory_only_to_complete_v2_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_model = SimpleNamespace(terms={})
    expected_eigvals = np.asarray([[0.0]], dtype=float)
    moire_cfg = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.empty((0, 2), dtype=float),
        n_orb1=1,
        n_orb2=0,
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.0, 1.0]),
        kpoints=np.array([[0.0, 0.0]], dtype=float),
        kpoints_fit=np.array([[0.0, 0.0]], dtype=float),
        heff=np.zeros((1, 1, 1), dtype=np.complex128),
        response_semantics="complete_linear_v2",
    )
    model_cfg = SimpleNamespace(
        null_channel_abs_tol=0.0,
        null_channel_rel_tol=0.0,
        coeff_prune_threshold=0.0,
        band_refinement_config={},
        compare_to_heff=False,
    )
    log_path = tmp_path / "model" / "model_run.log"

    def fake_compute_coefficients(moire_config, model, *, progress_callback=None):
        assert Path(moire_config.output_dir) == log_path.parent
        return model, {}

    def fake_compute_bands(
        moire_config,
        model,
        kpoints,
        return_eigvecs=False,
        hamiltonians_out=None,
        compiled_runtime_out=None,
    ):
        if hamiltonians_out is not None:
            hamiltonians_out.append(np.zeros((1, 1), dtype=np.complex128))
        if compiled_runtime_out is not None:
            compiled_runtime_out.append(None)
        return expected_eigvals

    monkeypatch.setattr("kp.model.pipeline.build_model", lambda _config: fake_model)
    monkeypatch.setattr("kp.model.pipeline.compute_coefficients", fake_compute_coefficients)
    monkeypatch.setattr("kp.model.pipeline.compute_bands", fake_compute_bands)

    _run_model_pipeline(
        moire_cfg,
        model_cfg,
        log_path,
        verbose=False,
        progress=False,
    )

    assert moire_cfg.output_dir is None


def test_progress_line_can_emit_ansi_color(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    _progress_line("fitting coefficients done in 1.23 s", enabled=True, style="done")
    stdout = capsys.readouterr().out

    assert "\033[" in stdout
    assert "fitting coefficients done in 1.23 s" in stdout


def test_progress_line_reserves_yellow_for_warnings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    _progress_line("fitting coefficients ...", enabled=True, style="fit_start")
    stdout = capsys.readouterr().out

    assert "\033[1;33m" not in stdout
    assert "fitting coefficients ..." in stdout


def test_harmonic_recommendation_plot_candidates_include_current_low_and_high() -> None:
    report = {
        "selected": {"intra_shells": 4, "inter_shells": 3, "plot_rms_mev": 0.01, "plot_max_mev": 0.02, "low_matrix_rms_mev": 0.0, "primary_rms_mev": 0.0, "primary_max_mev": 0.0, "subspace_mean_overlap": 1.0, "accepted": True},
        "candidates": [
            {"intra_shells": 1, "inter_shells": 1, "plot_rms_mev": 1.8, "plot_max_mev": 4.0, "low_matrix_rms_mev": 0.0, "primary_rms_mev": 0.1, "primary_max_mev": 0.2, "subspace_mean_overlap": 0.95, "accepted": False},
            {"intra_shells": 2, "inter_shells": 1, "plot_rms_mev": 0.5, "plot_max_mev": 1.0, "low_matrix_rms_mev": 0.0, "primary_rms_mev": 0.05, "primary_max_mev": 0.1, "subspace_mean_overlap": 0.995, "accepted": True},
            {"intra_shells": 4, "inter_shells": 3, "plot_rms_mev": 0.01, "plot_max_mev": 0.02, "low_matrix_rms_mev": 0.0, "primary_rms_mev": 0.0, "primary_max_mev": 0.0, "subspace_mean_overlap": 1.0, "accepted": True},
        ],
    }

    selected = _harmonic_recommendation_plot_candidates(report, current_counts={"intra": 1, "inter": 1})

    assert [item[0] for item in selected] == ["current", "low", "high"]
    assert (selected[0][1]["intra_shells"], selected[0][1]["inter_shells"]) == (1, 1)
    assert (selected[1][1]["intra_shells"], selected[1][1]["inter_shells"]) == (2, 1)
    assert (selected[2][1]["intra_shells"], selected[2][1]["inter_shells"]) == (2, 1)


def test_harmonic_recommendation_uses_balanced_tiers_before_strict_plateau() -> None:
    report = {
        "selected": {
            "intra_shells": 5,
            "inter_shells": 5,
            "plot_rms_mev": 0.006,
            "plot_max_mev": 0.024,
            "low_matrix_rms_mev": 0.0,
            "primary_rms_mev": 0.001,
            "primary_max_mev": 0.002,
            "subspace_mean_overlap": 1.0,
            "accepted": True,
        },
        "candidates": [
            {
                "intra_shells": 1,
                "inter_shells": 1,
                "plot_rms_mev": 10.558,
                "plot_max_mev": 26.418,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 11.038,
                "primary_max_mev": 26.418,
                "subspace_mean_overlap": 0.799516,
                "accepted": False,
            },
            {
                "intra_shells": 2,
                "inter_shells": 2,
                "plot_rms_mev": 0.550,
                "plot_max_mev": 3.470,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 0.203,
                "primary_max_mev": 0.395,
                "subspace_mean_overlap": 0.999477,
                "accepted": False,
            },
            {
                "intra_shells": 3,
                "inter_shells": 3,
                "plot_rms_mev": 0.623,
                "plot_max_mev": 2.274,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 0.282,
                "primary_max_mev": 0.649,
                "subspace_mean_overlap": 0.999583,
                "accepted": False,
            },
            {
                "intra_shells": 5,
                "inter_shells": 5,
                "plot_rms_mev": 0.006,
                "plot_max_mev": 0.024,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 0.001,
                "primary_max_mev": 0.002,
                "subspace_mean_overlap": 1.0,
                "accepted": True,
            },
        ],
    }

    selected = _harmonic_recommendation_plot_candidates(report, current_counts={"intra": 1, "inter": 1})

    assert [item[0] for item in selected] == ["current", "low", "high"]
    assert (selected[1][1]["intra_shells"], selected[1][1]["inter_shells"]) == (2, 2)
    assert (selected[2][1]["intra_shells"], selected[2][1]["inter_shells"]) == (3, 3)
    assert (report["selected"]["intra_shells"], report["selected"]["inter_shells"]) == (5, 5)


def test_harmonic_recommendation_uses_requested_low_and_high_accuracy_thresholds() -> None:
    report = {
        "selected": {
            "intra_shells": 4,
            "inter_shells": 4,
            "plot_rms_mev": 0.99,
            "plot_max_mev": 2.99,
            "low_matrix_rms_mev": 0.0,
            "primary_rms_mev": 0.01,
            "primary_max_mev": 0.02,
            "subspace_mean_overlap": 0.951,
            "accepted": True,
        },
        "candidates": [
            {
                "intra_shells": 1,
                "inter_shells": 1,
                "plot_rms_mev": 0.9,
                "plot_max_mev": 0.9,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 0.1,
                "primary_max_mev": 0.2,
                "subspace_mean_overlap": 0.9,
                "accepted": False,
            },
            {
                "intra_shells": 2,
                "inter_shells": 2,
                "plot_rms_mev": 1.0,
                "plot_max_mev": 30.0,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 0.1,
                "primary_max_mev": 0.2,
                "subspace_mean_overlap": 0.91,
                "accepted": False,
            },
            {
                "intra_shells": 3,
                "inter_shells": 3,
                "plot_rms_mev": 0.99,
                "plot_max_mev": 2.99,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 0.05,
                "primary_max_mev": 0.1,
                "subspace_mean_overlap": 0.95,
                "accepted": False,
            },
            {
                "intra_shells": 4,
                "inter_shells": 4,
                "plot_rms_mev": 0.99,
                "plot_max_mev": 2.99,
                "low_matrix_rms_mev": 0.0,
                "primary_rms_mev": 0.01,
                "primary_max_mev": 0.02,
                "subspace_mean_overlap": 0.951,
                "accepted": True,
            },
        ],
    }

    selected = _harmonic_recommendation_plot_candidates(report, current_counts={"intra": 1, "inter": 1})

    assert (selected[1][1]["intra_shells"], selected[1][1]["inter_shells"]) == (2, 2)
    assert (selected[2][1]["intra_shells"], selected[2][1]["inter_shells"]) == (4, 4)


def test_hamiltonian_element_comparison_arrays_keep_full_and_selected_support_views() -> None:
    heff = np.array(
        [
            [[1.0, 2.0j, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ],
        dtype=np.complex128,
    )
    model = heff + 1.0
    mask = np.array(
        [
            [True, False, True],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )

    panels = _hamiltonian_element_comparison_arrays(
        model,
        heff,
        mask,
        positions=[1],
    )

    assert panels["positions"] == [1]
    assert panels["heff"].shape == (1, 3, 3)
    assert panels["model"].shape == (1, 3, 3)
    assert panels["diff"].shape == (1, 3, 3)
    assert panels["heff"][0, 0, 0] == pytest.approx(10.0)
    assert panels["model"][0, 0, 0] == pytest.approx(11.0)
    assert panels["diff"][0, 0, 0] == pytest.approx(1.0)
    assert panels["heff"][0, 0, 1] == pytest.approx(11.0)
    assert panels["model"][0, 1, 2] == pytest.approx(16.0)
    assert panels["diff"][0, 1, 0] == pytest.approx(1.0)
    assert np.isnan(panels["support_heff"][0, 0, 1])
    assert np.isnan(panels["support_model"][0, 1, 2])
    assert np.isnan(panels["support_diff"][0, 1, 0])
    assert panels["support_fraction"] == pytest.approx(5.0 / 9.0)


def test_selected_harmonic_support_mask_shape_and_zero_support() -> None:
    q1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0]], dtype=float)

    mask = _selected_harmonic_support_mask(
        q1,
        q2,
        n_orb=(1, 1),
        current_counts={"intra": 0, "inter": 0},
    )

    assert mask.shape == (3, 3)
    assert mask[0, 0]
    assert mask[1, 1]
    assert mask[2, 2]
    assert not mask[0, 2]  # interlayer zero-q support is not part of inter=0 model support
    assert not mask[0, 1]


def test_selected_harmonic_support_mask_keeps_all_nonzero_inter_shells() -> None:
    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)

    mask = _selected_harmonic_support_mask(
        q1,
        q2,
        n_orb=(1, 1),
        current_counts={"intra": 0, "inter": 3},
    )

    np.testing.assert_array_equal(mask[0, 1:], np.ones(3, dtype=bool))
    np.testing.assert_array_equal(mask[1:, 0], np.ones(3, dtype=bool))


def test_selected_harmonic_support_mask_counts_shared_zero_inter_shell() -> None:
    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)

    mask = _selected_harmonic_support_mask(
        q1,
        q2,
        n_orb=(1, 1),
        current_counts={"intra": 0, "inter": 3},
    )

    np.testing.assert_array_equal(mask[0, 1:], np.ones(3, dtype=bool))
    np.testing.assert_array_equal(mask[1:, 0], np.ones(3, dtype=bool))


def test_save_hamiltonian_element_comparison_plot_writes_n_by_three_panel(tmp_path: Path) -> None:
    heff = np.stack([np.eye(3), 2.0 * np.eye(3)]).astype(np.complex128)
    model = heff + 0.1 * np.ones_like(heff)
    mask = np.eye(3, dtype=bool)

    path = save_hamiltonian_element_comparison_plot(
        model,
        heff,
        tmp_path / "hamiltonian_element_comparison.png",
        harmonic_mask=mask,
        positions=[0, 1],
        k_indices=[3, 7],
    )

    assert path.exists()
    assert path.stat().st_size > 0
    pdf_path = path.with_suffix(".pdf")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0

    support_path = save_hamiltonian_element_comparison_plot(
        model,
        heff,
        tmp_path / "hamiltonian_element_comparison_selected_support.png",
        harmonic_mask=mask,
        positions=[0, 1],
        k_indices=[3, 7],
        view="selected_support",
    )
    assert support_path.exists()
    assert support_path.with_suffix(".pdf").exists()


def test_save_hamiltonian_element_comparison_plot_draws_q_norm_shell_boundaries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import matplotlib.axes

    q1 = np.array(
        [[2.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0 + 1.0e-10, 0.0]],
        dtype=float,
    )
    q2 = np.array([[0.0, 0.0]], dtype=float)
    heff = np.eye(5, dtype=np.complex128)[None, :, :]
    shell_lines: list[tuple[float, float]] = []
    original_axhline = matplotlib.axes.Axes.axhline

    def spy_axhline(self, y=0, *args, **kwargs):
        linewidth = float(kwargs.get("linewidth", kwargs.get("lw", 0.0)))
        if linewidth == pytest.approx(0.7):
            shell_lines.append((float(y), linewidth))
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", spy_axhline)

    save_hamiltonian_element_comparison_plot(
        heff,
        heff,
        tmp_path / "qnorm.png",
        harmonic_mask=np.eye(5, dtype=bool),
        qset1=q1,
        qset2=q2,
        n_orb=(1, 1),
        q_order="norm_shell",
        q_norm_tol=1.0e-6,
    )

    assert {position for position, _linewidth in shell_lines} == {0.5, 2.5}


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
                            **_exactified_test_meta(),
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
                        **_exactified_test_meta(),
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
                        **_exactified_test_meta(),
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
    assert operation["source_matrix_projection_report"]["report"]["status"] == "exactified"
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
                        **_exactified_test_meta(),
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
                        **_exactified_test_meta(),
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
                        **_exactified_test_meta(),
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
    expected_eigvals = _expected_project_eigvals(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["fit"]["model_selection"] = False
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

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


def test_save_q_lattice_harmonics_plot_draws_model_qsets(tmp_path: Path) -> None:
    qset1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254]], dtype=float)
    qset2 = qset1 + np.array([1.0 / 3.0, -0.2], dtype=float)
    out = tmp_path / "q_lattice_harmonics.pdf"

    path = save_q_lattice_harmonics_plot(
        Q_set1=qset1,
        Q_set2=qset2,
        bM1=np.array([1.0, 0.0]),
        bM2=np.array([0.5, 0.8660254]),
        intra_harmonics={1: np.array([-1.0, 0.0])},
        inter_harmonics={1: qset2[0] - qset1[0]},
        path=out,
    )

    assert path == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_q_lattice_inter_arrow_uses_selected_sector_pair_direction() -> None:
    p_vec = np.array([0.0, 1.0], dtype=float)
    qset1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    qset2 = qset1 - p_vec

    start, target = pipeline_module._q_lattice_inter_arrow_endpoints(
        qset1=qset1,
        qset2=qset2,
        sectors=[
            {"name": "L1", "qset": "qset1"},
            {"name": "L2", "qset": "qset2"},
        ],
        inter_sector_pairs=[["L1", "L2"]],
        vector=p_vec,
        fallback_anchor=np.array([10.0, 10.0], dtype=float),
        tol=1.0e-8,
    )

    np.testing.assert_allclose(start, qset1[0], atol=1.0e-12)
    np.testing.assert_allclose(target, qset2[0], atol=1.0e-12)


def test_q_lattice_hex_shell_uses_outer_vertex_radius_for_offset_valleys() -> None:
    b1 = np.array([1.0, 0.0], dtype=float)
    b2 = np.array([0.5, np.sqrt(3.0) / 2.0], dtype=float)
    angle = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    offset_valley_points = np.column_stack([2.52 * np.cos(angle), 2.52 * np.sin(angle)])
    gamma_points = np.column_stack([2.65 * np.cos(angle), 2.65 * np.sin(angle)])

    assert pipeline_module._q_lattice_hex_shell_from_radius(offset_valley_points, b1, b2) == 2
    assert pipeline_module._q_lattice_hex_shell_from_radius(gamma_points, b1, b2) == 3


def test_run_configured_model_saves_outputs_without_legacy_diagnostics_json(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["fit"]["model_selection"] = False
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

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
    assert (tmp_path / "model_out" / "q_lattice_harmonics.pdf").exists()
    assert results["q_lattice_plot"] == str((tmp_path / "model_out" / "q_lattice_harmonics.pdf").resolve())
    assert results["band_plot"] == str((tmp_path / "model_out" / "band_comparison.pdf").resolve())
    assert (tmp_path / "model_out" / "band_comparison_all.pdf").exists()
    assert results["all_band_plot"] == str((tmp_path / "model_out" / "band_comparison_all.pdf").resolve())
    assert (tmp_path / "model_out" / "band_comparison_vs_full_heff.pdf").exists()
    assert results["full_heff_band_plot"] == str(
        (tmp_path / "model_out" / "band_comparison_vs_full_heff.pdf").resolve()
    )
    assert results["all_band_plot_comparison"]["num_bands"] == expected_eigvals.shape[1]
    assert results["comparison"]["reference"] == "current_heff_support_mask"
    assert results["comparison_vs_full_heff"]["reference"] == "original_heff"
    summary = json.loads((tmp_path / "model_out" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["comparison"]["max_abs_error"] == 0.0
    assert summary["plot_comparison"]["max_abs_error"] == 0.0
    assert summary["all_band_plot"] == "band_comparison_all.pdf"
    assert summary["q_lattice_plot"] == "q_lattice_harmonics.pdf"
    assert summary["all_band_plot_comparison"]["num_bands"] == expected_eigvals.shape[1]
    assert summary["comparison"]["reference"] == "current_heff_support_mask"
    assert summary["comparison_vs_full_heff"]["reference"] == "original_heff"
    assert summary["full_heff_band_plot"] == "band_comparison_vs_full_heff.pdf"


def test_run_configured_model_uses_current_support_as_primary_energy_and_overlap_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw_config["bands"]["plot"] = {"top_bands": 3, "align": "top"}
    raw_config["fit"]["model_selection"] = False
    cfg_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    heff_path = tmp_path / "project" / "heff.npy"
    heff = np.load(heff_path)
    heff[:, 1, 3] = np.array([0.35, 0.40, 0.45])
    heff[:, 3, 1] = heff[:, 1, 3]
    np.save(heff_path, heff)
    _write_symm_frame_manifest(tmp_path, rotation_deg=0.0)

    q1 = np.load(tmp_path / "q1.npy")
    q2 = np.load(tmp_path / "q2.npy")
    support_mask = _selected_harmonic_support_mask(
        q1,
        q2,
        n_orb=(1, 1),
        current_counts={"intra": 3, "inter": 2},
    )
    support_heff = heff * support_mask[None, :, :]
    support_heff = 0.5 * (support_heff + np.swapaxes(support_heff.conj(), -1, -2))
    support_eigvals, support_eigvecs = np.linalg.eigh(support_heff)
    full_eigvals = np.linalg.eigvalsh(heff)
    assert not np.allclose(support_eigvals, full_eigvals)

    plot_calls: dict[str, dict[str, object]] = {}
    overlap_target: dict[str, np.ndarray] = {}

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": (support_eigvals, support_eigvecs), "diagnostics": {}}

    def fake_save_band_comparison_plot(model, heff, path, **kwargs):
        output = Path(path)
        plot_calls[output.name] = {"heff": np.asarray(heff), **dict(kwargs)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        return output

    def fake_band_overlap_weights(model_basis, target_basis, **kwargs):
        overlap_target["basis"] = np.asarray(target_basis)
        overlap_target["eigvals"] = np.asarray(kwargs["reference_eigvals"])
        return np.ones(np.asarray(kwargs["model_eigvals"]).shape, dtype=float)

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)
    monkeypatch.setattr("kp.model.pipeline.save_band_comparison_plot", fake_save_band_comparison_plot)
    monkeypatch.setattr("kp.model.pipeline._band_overlap_weights", fake_band_overlap_weights)
    monkeypatch.setattr(
        "kp.model.pipeline._selected_harmonic_support_mask",
        lambda *args, **kwargs: support_mask,
    )

    results = run_configured_model(cfg_path)

    np.testing.assert_allclose(plot_calls["band_comparison.pdf"]["heff"], support_eigvals)
    assert plot_calls["band_comparison.pdf"].get("current_support_eigvals") is None
    assert plot_calls["band_comparison.pdf"]["plot_config"]["top_bands"] == 3
    np.testing.assert_allclose(plot_calls["band_comparison_vs_full_heff.pdf"]["heff"], full_eigvals)
    assert plot_calls["band_comparison_vs_full_heff.pdf"]["plot_config"]["top_bands"] == 3
    np.testing.assert_allclose(overlap_target["basis"], support_eigvecs)
    np.testing.assert_allclose(overlap_target["eigvals"], support_eigvals)
    assert results["comparison"]["max_abs_error"] == pytest.approx(0.0)
    assert results["comparison"]["reference"] == "current_heff_support_mask"
    assert results["plot_comparison"]["max_abs_error"] == pytest.approx(0.0)
    assert results["plot_comparison"]["reference"] == "current_heff_support_mask"
    assert results["all_band_plot_comparison"]["max_abs_error"] == pytest.approx(0.0)
    assert results["all_band_plot_comparison"]["reference"] == "current_heff_support_mask"
    assert results["comparison_vs_full_heff"]["max_abs_error"] > 0.0
    assert results["comparison_vs_full_heff"]["reference"] == "original_heff"
    assert results["plot_comparison_vs_full_heff"]["reference"] == "original_heff"
    assert results["all_band_plot_comparison_vs_full_heff"]["reference"] == "original_heff"


def test_run_configured_model_fails_closed_when_current_support_reference_cannot_be_built(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    def fail_support(*args, **kwargs):
        raise RuntimeError("support construction failed")

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)
    monkeypatch.setattr("kp.model.pipeline._selected_harmonic_support_mask", fail_support)

    with pytest.raises(
        ValueError,
        match="current Heff support mask is required as the primary model-comparison reference",
    ):
        run_configured_model(cfg_path)


def test_run_configured_model_does_not_reapply_band_indices_to_model_eigenvectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["bands"]["indices"] = [1, 2]
    raw["fit"]["model_selection"] = False
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    selected_eigvals = _expected_project_eigvals(tmp_path)[[1, 2]]
    model_eigvecs = np.stack(
        [
            np.eye(4, dtype=np.complex128),
            np.eye(4, dtype=np.complex128)[:, [1, 0, 2, 3]],
        ]
    )
    captured: dict[str, np.ndarray] = {}

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": (selected_eigvals, model_eigvecs), "diagnostics": {}}

    def fake_band_overlap_weights(model_basis, target_basis, **kwargs):
        captured["model_basis"] = np.asarray(model_basis)
        captured["target_basis"] = np.asarray(target_basis)
        return np.ones(np.asarray(kwargs["model_eigvals"]).shape, dtype=float)

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)
    monkeypatch.setattr("kp.model.pipeline._band_overlap_weights", fake_band_overlap_weights)

    run_configured_model(cfg_path)

    np.testing.assert_allclose(captured["model_basis"], model_eigvecs)
    assert captured["target_basis"].shape == model_eigvecs.shape


def test_run_configured_model_writes_all_heatmaps_only_to_subdirectory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = _write_fixture(tmp_path)
    expected_eigvals = _expected_project_eigvals(tmp_path)
    heff = np.load(tmp_path / "project" / "heff.npy")
    heatmap_calls: list[tuple[Path, str, str]] = []

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {
            "eigvals": expected_eigvals,
            "band_hamiltonians": heff,
            "diagnostics": {},
        }

    def fake_save_hamiltonian_element_comparison_plot(model, target, path, **kwargs):
        output = Path(path)
        heatmap_calls.append(
            (
                output,
                str(kwargs.get("view", "full")),
                str(kwargs.get("q_order", "native")),
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        output.with_suffix(".pdf").touch()
        return output

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)
    monkeypatch.setattr(
        "kp.model.pipeline.save_hamiltonian_element_comparison_plot",
        fake_save_hamiltonian_element_comparison_plot,
    )

    output = tmp_path / "model_out"
    output.mkdir(parents=True, exist_ok=True)
    legacy_names = (
        "hamiltonian_element_comparison.png",
        "hamiltonian_element_comparison.pdf",
        "hamiltonian_element_comparison_selected_support.png",
        "hamiltonian_element_comparison_selected_support.pdf",
    )
    for name in legacy_names:
        (output / name).touch()

    run_configured_model(cfg_path)

    qnorm_dir = output / "hamiltonian_element_comparisons"
    assert heatmap_calls == [
        (qnorm_dir / "native_full.png", "full", "native"),
        (qnorm_dir / "native_selected_support.png", "selected_support", "native"),
        (qnorm_dir / "qnorm_full.png", "full", "norm_shell"),
        (qnorm_dir / "qnorm_selected_support.png", "selected_support", "norm_shell"),
    ]
    assert all(not (output / name).exists() for name in legacy_names)
    assert {path.name for path in qnorm_dir.iterdir()} == {
        "native_full.png",
        "native_full.pdf",
        "native_selected_support.png",
        "native_selected_support.pdf",
        "qnorm_full.png",
        "qnorm_full.pdf",
        "qnorm_selected_support.png",
        "qnorm_selected_support.pdf",
    }


def test_json_safe_recurses_into_complex_numpy_arrays() -> None:
    payload = {
        "coefficients": np.asarray([1.25 - 0.5j, -2.0 + 3.0j], dtype=np.complex128),
    }

    converted = pipeline_module._json_safe(payload)

    assert converted == {
        "coefficients": [
            {"real": 1.25, "imag": -0.5},
            {"real": -2.0, "imag": 3.0},
        ]
    }
    json.dumps(converted)


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


def test_save_band_comparison_plot_draws_all_bands_by_default(monkeypatch, tmp_path: Path) -> None:
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

    original_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        color = kwargs.get("color")
        if color in {KP_REFERENCE_STYLE["color"], KP_MODEL_STYLE["color"]}:
            plotted_colors.append(str(color))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    save_band_comparison_plot(
        model,
        heff,
        tmp_path / "bands.png",
        plot_config={"top_bands": 2},
    )

    assert plotted_colors.count(KP_REFERENCE_STYLE["color"]) == 5
    assert plotted_colors.count(KP_MODEL_STYLE["color"]) == 5


def test_save_band_comparison_plot_draws_current_heff_support_mask(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes
    import matplotlib.pyplot as plt

    model = np.array([[0.0, 0.1], [0.02, 0.12]])
    heff = model + 0.01
    current_support = model + 0.005
    support_color = "#E69F00"
    plotted_support_colors: list[str] = []
    legend_labels: list[str] = []

    original_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        if kwargs.get("color") == support_color:
            plotted_support_colors.append(support_color)
        return original_plot(self, *args, **kwargs)

    def spy_close(fig=None):
        if fig is not None and fig.axes:
            legend = fig.axes[0].get_legend()
            if legend is not None:
                legend_labels.extend(item.get_text() for item in legend.get_texts())

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)
    monkeypatch.setattr(plt, "close", spy_close)

    save_band_comparison_plot(
        model,
        heff,
        tmp_path / "bands.png",
        current_support_eigvals=current_support,
    )

    assert plotted_support_colors == [support_color, support_color]
    assert "Current Heff support mask" in legend_labels


def test_save_band_comparison_plot_labels_the_primary_reference_explicitly(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    model = np.array([[0.0, 0.1], [0.02, 0.12]])
    reference = model + 0.005
    legend_labels: list[str] = []

    def spy_close(fig=None):
        if fig is not None and fig.axes:
            legend = fig.axes[0].get_legend()
            if legend is not None:
                legend_labels.extend(item.get_text() for item in legend.get_texts())

    monkeypatch.setattr(plt, "close", spy_close)

    save_band_comparison_plot(
        model,
        reference,
        tmp_path / "bands.png",
        reference_label="Current Heff support mask",
    )

    assert "Current Heff support mask" in legend_labels
    assert "Reference" not in legend_labels


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


def test_save_band_comparison_plot_colors_model_bands_by_overlap(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    model = np.array([[0.0, 0.1, 0.2], [0.02, 0.12, 0.24], [0.03, 0.15, 0.27]])
    heff = model + 0.001
    overlap = np.array([[0.5, 0.8, 0.95], [0.6, 0.85, 0.96], [0.7, 0.9, 0.98]])
    collections: list[LineCollection] = []
    ylims: list[tuple[float, float]] = []
    saved_axes = []

    original_add_collection = matplotlib.axes.Axes.add_collection
    original_set_ylim = matplotlib.axes.Axes.set_ylim

    def spy_add_collection(self, collection, *args, **kwargs):
        if isinstance(collection, LineCollection):
            collections.append(collection)
        return original_add_collection(self, collection, *args, **kwargs)

    def spy_set_ylim(self, bottom=None, top=None, *args, **kwargs):
        if bottom is not None and top is not None:
            ylims.append((float(bottom), float(top)))
        return original_set_ylim(self, bottom, top, *args, **kwargs)

    def spy_close(fig=None):
        saved_axes.extend(fig.axes if fig is not None else [])

    monkeypatch.setattr(matplotlib.axes.Axes, "add_collection", spy_add_collection)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", spy_set_ylim)
    monkeypatch.setattr(plt, "close", spy_close)

    save_band_comparison_plot(
        model,
        heff,
        tmp_path / "bands.pdf",
        plot_config={"top_bands": 2, "plot_all_bands": True, "align": "top"},
        overlap_weights=overlap,
    )

    assert (tmp_path / "bands.pdf").exists()
    assert len(saved_axes) >= 2
    assert saved_axes[-1].get_ylabel() == "Wavefunction overlap"
    np.testing.assert_allclose(saved_axes[-1].get_ylim(), (0.8, 0.98))
    model_collections = [item for item in collections if item.get_array() is not None and len(item.get_array()) == model.shape[0] - 1]
    assert len(model_collections) == model.shape[1]
    assert ylims
    assert ylims[-1][0] > -0.22


def test_band_overlap_weights_clusters_when_either_side_is_nearly_degenerate() -> None:
    reference_vec = np.broadcast_to(np.eye(3, dtype=complex), (1, 3, 3)).copy()
    theta = np.pi / 4.0
    model_vec = reference_vec.copy()
    model_vec[0, :2, :2] = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=complex,
    )

    naive = _band_overlap_weights(model_vec, reference_vec)
    np.testing.assert_allclose(naive[0, :2], [0.5, 0.5])

    reference_touch = _band_overlap_weights(
        model_vec,
        reference_vec,
        model_eigvals=np.array([[0.0, 0.02, 0.1]]),
        reference_eigvals=np.array([[0.0, 1.0e-5, 0.1]]),
        degeneracy_tol=1.0e-4,
    )
    model_touch = _band_overlap_weights(
        model_vec,
        reference_vec,
        model_eigvals=np.array([[0.0, 1.0e-5, 0.1]]),
        reference_eigvals=np.array([[0.0, 0.02, 0.1]]),
        degeneracy_tol=1.0e-4,
    )

    np.testing.assert_allclose(reference_touch[0, :2], [1.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(model_touch[0, :2], [1.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(reference_touch[0, 2], 1.0, atol=1.0e-12)


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


def test_window_band_plot_config_respects_explicit_top_bands() -> None:
    config = _window_band_plot_config({"top_bands": 10, "align": "top"}, target_bands="top")

    assert config["top_bands"] == 10
    assert config["align"] == "top"
    assert config["show_metrics"] is False
    assert config["legend_outside"] is True


def test_window_band_plot_config_respects_larger_or_explicit_slices() -> None:
    assert _window_band_plot_config({"top_bands": 16}, target_bands="top")["top_bands"] == 16

    sliced = _window_band_plot_config({"band_slice": [4, 18], "align": "top"}, target_bands="top")
    assert sliced == {
        "band_slice": [4, 18],
        "align": "top",
        "show_metrics": False,
        "legend_outside": True,
        "plot_all_bands": True,
    }


def test_window_band_plot_config_defaults_from_target_edge() -> None:
    top = _window_band_plot_config({}, target_bands="top")
    bottom = _window_band_plot_config({}, target_bands="bottom")

    assert top["top_bands"] == 10
    assert bottom["bottom_bands"] == 10


def test_run_configured_model_preserves_plot_ylim_in_config(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw.setdefault("bands", {})
    raw["bands"].setdefault("plot", {})
    raw["bands"]["plot"]["ylim"] = [0.0, 0.16]
    raw["fit"]["model_selection"] = False
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    expected_eigvals = _expected_project_eigvals(tmp_path)

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
    _disable_model_selection_for_mock_pipeline(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["validation"] = {"strict": True}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_eigvals = _expected_project_eigvals(tmp_path)

    def fake_pipeline(moire_config, model_config, log_path, *, verbose, progress):
        return {"eigvals": expected_eigvals, "diagnostics": {}}

    monkeypatch.setattr("kp.model.pipeline._run_model_pipeline", fake_pipeline)

    with pytest.raises(ValueError, match="validation.strict=true requires"):
        run_configured_model(cfg_path)


def test_validation_production_mode_rejects_unavailable_validation_outputs(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_fixture(tmp_path)
    _disable_model_selection_for_mock_pipeline(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["validation"] = {"mode": "production"}
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_eigvals = _expected_project_eigvals(tmp_path)

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
