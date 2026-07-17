from __future__ import annotations

import importlib

import numpy as np
import pytest


def _selection_module():
    try:
        return importlib.import_module("kp.low_energy_selection")
    except ModuleNotFoundError:
        pytest.fail("kp.low_energy_selection is not implemented")


def test_reference_k_uses_exact_expansion_origin_not_array_zero() -> None:
    selection = _selection_module()
    kpoints = np.asarray(
        [
            [0.25, 0.0],
            [0.10, -0.20],
            [0.0, 0.0],
            [-0.15, 0.05],
        ],
        dtype=float,
    )
    qsets = [np.asarray([[0.2, 0.0], [0.0, 0.0]], dtype=float)]

    reference = selection.resolve_reference_point(kpoints, qsets)

    assert reference.k_index == 2
    assert reference.k_coordinate == (0.0, 0.0)


def test_reference_k_rejects_path_without_expansion_origin() -> None:
    selection = _selection_module()
    kpoints = np.asarray([[0.1, 0.0], [-0.1, 0.0]], dtype=float)
    qsets = [np.asarray([[0.0, 0.0]], dtype=float)]

    with pytest.raises(ValueError, match="expansion origin"):
        selection.resolve_reference_point(kpoints, qsets, origin_tolerance=1.0e-8)


def test_minimum_q_is_resolved_per_source_group_with_stable_tie_break() -> None:
    selection = _selection_module()
    kpoints = np.asarray([[0.0, 0.0]], dtype=float)
    qsets = [
        np.asarray([[0.3, 0.0], [0.1, 0.0], [-0.1, 0.0]], dtype=float),
        np.asarray([[0.0, 0.4], [0.0, 0.2]], dtype=float),
    ]

    reference = selection.resolve_reference_point(kpoints, qsets)

    assert reference.q_indices == (1, 1)
    assert reference.q_vectors == ((0.1, 0.0), (0.0, 0.2))


def _diagonal_block(
    selection,
    key: str,
    energies: list[float],
    *,
    physical_layer: int | None = None,
    joint_layers: tuple[int, ...] = (),
):
    dimension = len(energies)
    return selection.ReferenceBlock(
        key=key,
        eigenvalues=np.asarray(energies, dtype=float),
        eigenvectors=np.eye(dimension, dtype=np.complex128),
        physical_layer=physical_layer,
        joint_layers=joint_layers,
    )


def test_spinless_nondegenerate_reference_state_produces_dimension_one_seed() -> None:
    selection = _selection_module()
    block = _diagonal_block(selection, "L1", [-0.4, -0.1, 0.3], physical_layer=0)
    states = selection.build_reference_state_pool([block])

    clusters = selection.cluster_edge_states(
        states,
        edge="valence",
        efermi=0.0,
        degeneracy_tolerance=1.0e-6,
    )
    seeds = selection.cumulative_energy_seeds(clusters)

    assert len(seeds[0].states) == 1
    assert seeds[0].states[0].band_index == 1


def test_near_degenerate_states_form_one_energy_cluster() -> None:
    selection = _selection_module()
    block = _diagonal_block(selection, "joint", [-0.3, -0.1000, -0.0995, 0.2], joint_layers=(0, 1))
    states = selection.build_reference_state_pool([block])

    clusters = selection.cluster_edge_states(
        states,
        edge="valence",
        efermi=0.0,
        degeneracy_tolerance=1.0e-3,
    )

    assert [state.band_index for state in clusters[0].states] == [2, 1]


def test_valence_and_conduction_edges_sort_from_the_requested_side() -> None:
    selection = _selection_module()
    block = _diagonal_block(selection, "L1", [-0.4, -0.1, 0.05, 0.3], physical_layer=0)
    states = selection.build_reference_state_pool([block])

    valence = selection.cluster_edge_states(states, edge="valence", efermi=0.0)
    conduction = selection.cluster_edge_states(states, edge="conduction", efermi=0.0)

    assert [cluster.states[0].band_index for cluster in valence] == [1, 0]
    assert [cluster.states[0].band_index for cluster in conduction] == [2, 3]


def test_non_gamma_pool_keeps_physical_layer_labels_and_allows_inactive_layers() -> None:
    selection = _selection_module()
    blocks = [
        _diagonal_block(selection, "L1", [-0.8, 0.4], physical_layer=0),
        _diagonal_block(selection, "L2", [-0.05, 0.5], physical_layer=1),
        _diagonal_block(selection, "L3", [-0.7, 0.6], physical_layer=2),
    ]
    states = selection.build_reference_state_pool(blocks)

    clusters = selection.cluster_edge_states(states, edge="valence", efermi=0.0)
    first_seed = selection.cumulative_energy_seeds(clusters)[0]

    assert [(state.physical_layer, state.band_index) for state in first_seed.states] == [(1, 0)]


def test_gamma_pool_keeps_joint_block_identity_without_fake_layer_ownership() -> None:
    selection = _selection_module()
    block = _diagonal_block(selection, "Gamma_joint", [-0.2, -0.1], joint_layers=(0, 1))

    states = selection.build_reference_state_pool([block])

    assert {state.physical_layer for state in states} == {None}
    assert {state.joint_layers for state in states} == {(0, 1)}


def _seed_for(selection, states, *keys: tuple[str, int]):
    by_key = {(state.block_key, state.band_index): state for state in states}
    return selection.LowEnergySeed(states=tuple(by_key[key] for key in keys))


def test_identity_closed_spinless_seed_remains_dimension_one() -> None:
    selection = _selection_module()
    block = _diagonal_block(selection, "L1", [-0.1, 0.2], physical_layer=0)
    states = selection.build_reference_state_pool([block])
    action = selection.ReferenceAction(
        name="identity",
        source_block="L1",
        target_block="L1",
        matrix=np.eye(2, dtype=np.complex128),
        antiunitary=False,
    )

    closure = selection.close_seed_under_symmetry(
        _seed_for(selection, states, ("L1", 0)),
        states,
        [action],
    )

    assert [(state.block_key, state.band_index) for state in closure.states] == [("L1", 0)]
    assert closure.additions == ()


def test_spinful_tr_seed_adds_its_kramers_partner() -> None:
    selection = _selection_module()
    block = _diagonal_block(selection, "M1", [-0.1, -0.1], physical_layer=0)
    states = selection.build_reference_state_pool([block])
    time_reversal = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    action = selection.ReferenceAction(
        name="TR",
        source_block="M1",
        target_block="M1",
        matrix=time_reversal,
        antiunitary=True,
    )

    closure = selection.close_seed_under_symmetry(
        _seed_for(selection, states, ("M1", 0)),
        states,
        [action],
    )

    assert {(state.block_key, state.band_index) for state in closure.states} == {("M1", 0), ("M1", 1)}
    assert [(addition.operation, addition.target_band) for addition in closure.additions] == [("TR", 1)]


def test_layer_exchange_adds_partner_from_target_layer() -> None:
    selection = _selection_module()
    blocks = [
        _diagonal_block(selection, "L1", [-0.1, 0.3], physical_layer=0),
        _diagonal_block(selection, "L2", [-0.1, 0.4], physical_layer=1),
    ]
    states = selection.build_reference_state_pool(blocks)
    action = selection.ReferenceAction(
        name="exchange",
        source_block="L1",
        target_block="L2",
        matrix=np.eye(2, dtype=np.complex128),
        antiunitary=False,
    )

    closure = selection.close_seed_under_symmetry(
        _seed_for(selection, states, ("L1", 0)),
        states,
        [action],
    )

    assert {(state.block_key, state.band_index) for state in closure.states} == {("L1", 0), ("L2", 0)}


def test_ptse2_like_joint_gamma_seed_54_closes_with_55() -> None:
    selection = _selection_module()
    energies = np.linspace(-2.0, 1.0, 56).tolist()
    block = _diagonal_block(selection, "Gamma_joint", energies, joint_layers=(0, 1))
    states = selection.build_reference_state_pool([block])
    time_reversal = np.eye(56, dtype=np.complex128)
    time_reversal[54, 54] = 0.0
    time_reversal[55, 55] = 0.0
    time_reversal[55, 54] = 1.0
    time_reversal[54, 55] = -1.0
    action = selection.ReferenceAction(
        name="TR",
        source_block="Gamma_joint",
        target_block="Gamma_joint",
        matrix=time_reversal,
        antiunitary=True,
    )

    closure = selection.close_seed_under_symmetry(
        _seed_for(selection, states, ("Gamma_joint", 54)),
        states,
        [action],
    )

    assert [state.band_index for state in closure.states] == [54, 55]
    assert closure.additions[0].source_band == 54
    assert closure.additions[0].target_band == 55


def test_impossible_symmetry_image_is_a_structural_failure() -> None:
    selection = _selection_module()
    source = _diagonal_block(selection, "source", [-0.1, 0.2], physical_layer=0)
    target = selection.ReferenceBlock(
        key="target",
        eigenvalues=np.asarray([-0.1]),
        eigenvectors=np.asarray([[1.0], [0.0]], dtype=np.complex128),
        physical_layer=1,
    )
    states = selection.build_reference_state_pool([source, target])
    action = selection.ReferenceAction(
        name="bad_exchange",
        source_block="source",
        target_block="target",
        matrix=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        antiunitary=False,
    )

    with pytest.raises(selection.SymmetryClosureError, match="cannot represent"):
        selection.close_seed_under_symmetry(
            _seed_for(selection, states, ("source", 0)),
            states,
            [action],
        )


def _thresholds(selection):
    return selection.SelectionThresholds(
        band_rms_mev=1.0,
        band_max_mev=3.0,
        subspace_overlap=0.95,
        symmetry_residual=1.0e-6,
        symmetry_leakage=1.0e-5,
    )


def _candidate(selection, candidate_id: str, dimension: int, **overrides):
    values = {
        "band_rms_mev": 0.5,
        "band_max_mev": 2.0,
        "subspace_overlap": 0.98,
        "symmetry_residual": 1.0e-8,
        "symmetry_leakage": 1.0e-7,
        "structural_failure": None,
    }
    values.update(overrides)
    return selection.CandidateMetrics(
        candidate_id=candidate_id,
        dimension=dimension,
        **values,
    )


def test_smallest_candidate_passing_every_threshold_is_selected() -> None:
    selection = _selection_module()
    candidates = [
        _candidate(selection, "eight", 8, band_rms_mev=0.1),
        _candidate(selection, "four", 4, band_rms_mev=0.8),
        _candidate(selection, "two", 2, band_rms_mev=1.2),
    ]

    decision = selection.select_projection_candidate(candidates, _thresholds(selection))

    assert decision.status == "PASS"
    assert decision.selected.candidate_id == "four"
    assert decision.violations == ()


def test_same_dimension_uses_band_error_overlap_and_symmetry_tie_breaks() -> None:
    selection = _selection_module()
    candidates = [
        _candidate(selection, "larger-rms", 4, band_rms_mev=0.7, subspace_overlap=0.999),
        _candidate(
            selection,
            "worse-overlap",
            4,
            band_rms_mev=0.4,
            subspace_overlap=0.97,
            symmetry_residual=1.0e-10,
        ),
        _candidate(
            selection,
            "best",
            4,
            band_rms_mev=0.4,
            subspace_overlap=0.99,
            symmetry_residual=5.0e-8,
        ),
    ]

    decision = selection.select_projection_candidate(candidates, _thresholds(selection))

    assert decision.selected.candidate_id == "best"


def test_no_passing_candidate_selects_minimax_normalized_violation_with_warning() -> None:
    selection = _selection_module()
    candidates = [
        _candidate(selection, "bad-rms", 2, band_rms_mev=2.0),
        _candidate(
            selection,
            "balanced",
            4,
            band_rms_mev=1.2,
            band_max_mev=3.3,
            subspace_overlap=0.94,
        ),
        _candidate(selection, "bad-overlap", 8, subspace_overlap=0.70),
    ]

    decision = selection.select_projection_candidate(candidates, _thresholds(selection))

    assert decision.status == "WARN"
    assert decision.selected.candidate_id == "balanced"
    assert {violation.metric for violation in decision.violations} == {
        "band_rms_mev",
        "band_max_mev",
        "subspace_overlap",
    }


def test_structural_failure_is_never_selected_as_warning_fallback() -> None:
    selection = _selection_module()
    candidates = [
        _candidate(selection, "numerically-perfect", 1, structural_failure="projector rank loss"),
        _candidate(selection, "valid-warning", 4, band_rms_mev=1.5),
    ]

    decision = selection.select_projection_candidate(candidates, _thresholds(selection))

    assert decision.status == "WARN"
    assert decision.selected.candidate_id == "valid-warning"
    assert decision.structural_failures == (("numerically-perfect", "projector rank loss"),)


def test_all_structurally_failed_candidates_raise_instead_of_warning() -> None:
    selection = _selection_module()
    candidates = [
        _candidate(selection, "rank-loss", 2, structural_failure="projector rank loss"),
        _candidate(selection, "singular", 4, structural_failure="singular downfolding"),
    ]

    with pytest.raises(selection.CandidateSelectionError, match="no structurally valid"):
        selection.select_projection_candidate(candidates, _thresholds(selection))
