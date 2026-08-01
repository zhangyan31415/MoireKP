from __future__ import annotations

import importlib

import numpy as np
import pytest


def _api():
    return importlib.import_module("kp.symmetry.factorized_action")


def _basis_index(
    sector: int,
    orbital: int,
    q_index: int,
    *,
    q_counts: tuple[int, ...],
    n_orb: tuple[int, ...],
) -> int:
    offset = sum(q_counts[item] * n_orb[item] for item in range(sector))
    return int(offset + orbital * q_counts[sector] + q_index)


def _factorized_matrix(
    blocks: tuple[np.ndarray, ...],
    phases: tuple[complex, ...],
    *,
    q_counts: tuple[int, ...],
    n_orb: tuple[int, ...],
    q_permutation: tuple[int, ...],
    sector_permutation: tuple[int, ...],
) -> np.ndarray:
    dim = sum(q_counts[item] * n_orb[item] for item in range(len(q_counts)))
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    q_offsets = np.cumsum((0, *q_counts[:-1]))
    source_global_q = 0
    for source_sector, source_q_count in enumerate(q_counts):
        target_sector = int(sector_permutation[source_sector])
        block = np.asarray(blocks[source_sector], dtype=np.complex128)
        for source_q in range(source_q_count):
            target_global_q = int(q_permutation[source_global_q])
            target_q = target_global_q - int(q_offsets[target_sector])
            rows = [
                _basis_index(
                    target_sector,
                    orbital,
                    target_q,
                    q_counts=q_counts,
                    n_orb=n_orb,
                )
                for orbital in range(n_orb[target_sector])
            ]
            columns = [
                _basis_index(
                    source_sector,
                    orbital,
                    source_q,
                    q_counts=q_counts,
                    n_orb=n_orb,
                )
                for orbital in range(n_orb[source_sector])
            ]
            matrix[np.ix_(rows, columns)] = phases[source_global_q] * block
            source_global_q += 1
    return matrix


def test_exactified_action_factorization_removes_only_q_dependent_phase() -> None:
    api = _api()
    q_counts = (2, 2)
    n_orb = (2, 2)
    sector_permutation = (0, 1)
    q_permutation = (1, 0, 3, 2)
    blocks = (
        np.asarray([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128),
        np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128),
    )
    phases = (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)
    matrix = _factorized_matrix(
        blocks,
        phases,
        q_counts=q_counts,
        n_orb=n_orb,
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
    )
    q_vectors = (
        np.asarray([[-0.2, 0.0], [0.2, 0.0]], dtype=np.float64),
        np.asarray([[0.0, -0.3], [0.0, 0.3]], dtype=np.float64),
    )

    action = api.certify_factorized_action(
        name="TR",
        matrix=matrix,
        antiunitary=True,
        k_forward=-np.eye(2),
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
        q_vectors=q_vectors,
        q_counts=q_counts,
        n_orb=n_orb,
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    assert action.version == "factorized_response_action_v1"
    assert action.name == "TR"
    assert action.antiunitary is True
    assert action.q_permutation == q_permutation
    assert action.sector_permutation == sector_permutation
    assert action.off_route_residual == 0.0
    assert action.phase_alignment_residual == 0.0
    assert action.q_covariance_residual == 0.0
    for actual, expected in zip(action.orbital_blocks, blocks):
        phase = np.vdot(expected, actual)
        phase /= abs(phase)
        np.testing.assert_allclose(actual, phase * expected, atol=1.0e-15)
    assert len(action.q_phases) == sum(q_counts)
    assert len(action.artifact_hash) == 64


def test_factorized_action_certifies_inactive_zero_orbital_sector() -> None:
    api = _api()
    q_counts = (2, 2)
    n_orb = (0, 1)
    q_permutation = (0, 1, 2, 3)
    sector_permutation = (0, 1)
    blocks = (
        np.zeros((0, 0), dtype=np.complex128),
        np.ones((1, 1), dtype=np.complex128),
    )
    matrix = _factorized_matrix(
        blocks,
        (1.0 + 0.0j,) * 4,
        q_counts=q_counts,
        n_orb=n_orb,
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
    )
    q_vectors = (
        np.asarray([[-0.2, 0.0], [0.2, 0.0]], dtype=np.float64),
        np.asarray([[0.0, -0.3], [0.0, 0.3]], dtype=np.float64),
    )

    action = api.certify_factorized_action(
        name="C3z",
        matrix=matrix,
        antiunitary=False,
        k_forward=np.eye(2),
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
        q_vectors=q_vectors,
        q_counts=q_counts,
        n_orb=n_orb,
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    assert action.n_orb == n_orb
    assert action.orbital_blocks[0].shape == (0, 0)
    np.testing.assert_allclose(api.materialize_factorized_matrix(action), matrix)


def test_factorization_rejects_matrix_weight_outside_declared_q_route() -> None:
    api = _api()
    q_counts = (1, 1)
    n_orb = (2, 2)
    q_permutation = (0, 1)
    sector_permutation = (0, 1)
    blocks = (np.eye(2, dtype=complex), np.eye(2, dtype=complex))
    matrix = _factorized_matrix(
        blocks,
        (1.0 + 0.0j, 1.0 + 0.0j),
        q_counts=q_counts,
        n_orb=n_orb,
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
    )
    matrix[2, 0] = 1.0e-4

    with pytest.raises(api.FactorizedActionCertificationError, match="off-route"):
        api.certify_factorized_action(
            name="bad",
            matrix=matrix,
            antiunitary=False,
            k_forward=np.eye(2),
            q_permutation=q_permutation,
            sector_permutation=sector_permutation,
            q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
            q_counts=q_counts,
            n_orb=n_orb,
            matrix_absolute_error_bound=1.0e-12,
            q_absolute_error_bound=1.0e-12,
        )


def test_factorization_rejects_q_blocks_that_are_not_phase_equivalent() -> None:
    api = _api()
    q_counts = (2, 1)
    n_orb = (2, 2)
    q_permutation = (0, 1, 2)
    sector_permutation = (0, 1)
    matrix = _factorized_matrix(
        (np.eye(2, dtype=complex), np.eye(2, dtype=complex)),
        (1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j),
        q_counts=q_counts,
        n_orb=n_orb,
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
    )
    rows = [
        _basis_index(0, orbital, 1, q_counts=q_counts, n_orb=n_orb)
        for orbital in range(2)
    ]
    columns = [
        _basis_index(0, orbital, 1, q_counts=q_counts, n_orb=n_orb)
        for orbital in range(2)
    ]
    matrix[np.ix_(rows, columns)] = np.diag([1.0, -1.0])

    with pytest.raises(api.FactorizedActionCertificationError, match="phase-equivalent"):
        api.certify_factorized_action(
            name="bad-blocks",
            matrix=matrix,
            antiunitary=False,
            k_forward=np.eye(2),
            q_permutation=q_permutation,
            sector_permutation=sector_permutation,
            q_vectors=(np.zeros((2, 2)), np.zeros((1, 2))),
            q_counts=q_counts,
            n_orb=n_orb,
            matrix_absolute_error_bound=1.0e-12,
            q_absolute_error_bound=1.0e-12,
        )


def test_factorized_action_package_round_trip_verifies_hashes() -> None:
    api = _api()
    q_counts = (1, 1)
    n_orb = (2, 2)
    action = api.certify_factorized_action(
        name="C2",
        matrix=_factorized_matrix(
            (np.eye(2, dtype=complex), np.eye(2, dtype=complex)),
            (1.0 + 0.0j, -1.0 + 0.0j),
            q_counts=q_counts,
            n_orb=n_orb,
            q_permutation=(1, 0),
            sector_permutation=(1, 0),
        ),
        antiunitary=False,
        k_forward=np.eye(2),
        q_permutation=(1, 0),
        sector_permutation=(1, 0),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=q_counts,
        n_orb=n_orb,
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    metadata, arrays = api.pack_factorized_actions([action])
    restored = api.load_factorized_actions(metadata, arrays)

    assert metadata["version"] == "factorized_response_action_v1"
    assert metadata["status"] == "certified"
    assert len(metadata["package_hash"]) == 64
    assert set(restored) == {"C2"}
    assert restored["C2"].artifact_hash == action.artifact_hash
    for actual, expected in zip(restored["C2"].orbital_blocks, action.orbital_blocks):
        np.testing.assert_array_equal(actual, expected)


def test_factorized_action_package_rejects_corrupted_block_array() -> None:
    api = _api()
    action = api.certify_factorized_action(
        name="identity",
        matrix=np.eye(2, dtype=complex),
        antiunitary=False,
        k_forward=np.eye(2),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )
    metadata, arrays = api.pack_factorized_actions([action])
    corrupted = {key: np.array(value, copy=True) for key, value in arrays.items()}
    block_key = metadata["actions"][0]["orbital_block_array_keys"][0]
    corrupted[block_key][0, 0] += 0.25

    with pytest.raises(api.FactorizedActionCertificationError, match="hash mismatch"):
        api.load_factorized_actions(metadata, corrupted)


def test_matrix_symmetry_generator_exposes_loaded_factorized_action() -> None:
    api = _api()
    from kp.model.symmetry import MatrixSymmetryGenerator

    action = api.certify_factorized_action(
        name="C3z",
        matrix=np.eye(2, dtype=complex),
        antiunitary=False,
        k_forward=np.eye(2),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )
    generator = MatrixSymmetryGenerator(
        {"C3z": np.eye(2, dtype=complex)},
        {},
        factorized_actions={"C3z": action},
    )

    assert generator.get_factorized_action("C3z").artifact_hash == action.artifact_hash
    assert generator.get_factorized_action("TR") is None


def test_factorized_action_materializes_the_certified_full_matrix() -> None:
    api = _api()
    q_counts = (2, 2)
    n_orb = (2, 2)
    q_permutation = (3, 2, 1, 0)
    sector_permutation = (1, 0)
    blocks = (
        np.asarray([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128),
        np.asarray([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128),
    )
    phases = (1.0, 1.0j, -1.0, -1.0j)
    matrix = _factorized_matrix(
        blocks,
        phases,
        q_counts=q_counts,
        n_orb=n_orb,
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
    )
    action = api.certify_factorized_action(
        name="C2",
        matrix=matrix,
        antiunitary=False,
        k_forward=np.diag([1.0, -1.0]),
        q_permutation=q_permutation,
        sector_permutation=sector_permutation,
        q_vectors=(np.zeros((2, 2)), np.zeros((2, 2))),
        q_counts=q_counts,
        n_orb=n_orb,
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    np.testing.assert_allclose(
        api.materialize_factorized_matrix(action),
        matrix,
        atol=2.0e-14,
    )
