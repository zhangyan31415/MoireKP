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
