from __future__ import annotations

import numpy as np

from kp.basis.symmetry_gauge import (
    SymmetryAdaptedBasisFrame,
    SymmetryGaugeOperation,
    derive_symmetry_adapted_basis_frame,
    derive_symmetry_adapted_internal_frame,
    transform_basis_operation,
    transform_basis_hamiltonian,
    transform_basis_vectors,
    transform_internal_operation,
)


def _gamma_spinful_four_state_operations() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sqrt3 = np.sqrt(3.0)
    rotation = np.array(
        [
            [-0.25 - 0.25j * sqrt3, -0.25 * sqrt3 - 0.75j, 0.0, 0.0],
            [0.25 * sqrt3 + 0.75j, -0.25 - 0.25j * sqrt3, 0.0, 0.0],
            [0.0, 0.0, -0.25 + 0.25j * sqrt3, -0.25 * sqrt3 + 0.75j],
            [0.0, 0.0, 0.25 * sqrt3 - 0.75j, -0.25 + 0.25j * sqrt3],
        ],
        dtype=np.complex128,
    )
    time_reversal = np.array(
        [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    layer_exchange = np.array(
        [
            [0.0, 0.0, -1.0j, 0.0],
            [0.0, 0.0, 0.0, 1.0j],
            [-1.0j, 0.0, 0.0, 0.0],
            [0.0, 1.0j, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    return rotation, time_reversal, layer_exchange


def test_symmetry_adapted_frame_orders_unitary_phases_and_kramers_pairs() -> None:
    rotation, time_reversal, layer_exchange = _gamma_spinful_four_state_operations()
    frame = derive_symmetry_adapted_internal_frame(
        [
            SymmetryGaugeOperation("finite_rotation", rotation, antiunitary=False, power=3),
            SymmetryGaugeOperation("antiunitary_pairing", time_reversal, antiunitary=True, power=2),
            SymmetryGaugeOperation("secondary_exchange", layer_exchange, antiunitary=False, power=2),
        ]
    )

    assert frame.status == "applied"
    assert frame.primary_operation == "finite_rotation"
    transformed_rotation = transform_internal_operation(rotation, frame.unitary, antiunitary=False)
    transformed_tr = transform_internal_operation(time_reversal, frame.unitary, antiunitary=True)
    transformed_exchange = transform_internal_operation(layer_exchange, frame.unitary, antiunitary=False)
    expected_phases = np.array(
        [np.exp(-1.0j * np.pi / 3.0), np.exp(1.0j * np.pi / 3.0), -1.0, -1.0],
        dtype=np.complex128,
    )
    expected_pairing = np.kron(np.eye(2), np.array([[0.0, -1.0], [1.0, 0.0]]))
    expected_exchange = np.kron(np.eye(2), -1.0j * np.array([[0.0, 1.0], [1.0, 0.0]]))

    np.testing.assert_allclose(transformed_rotation, np.diag(expected_phases), atol=1.0e-12)
    np.testing.assert_allclose(transformed_tr, expected_pairing, atol=1.0e-12)
    np.testing.assert_allclose(transformed_exchange, expected_exchange, atol=1.0e-12)
    np.testing.assert_allclose(frame.unitary.conj().T @ frame.unitary, np.eye(4), atol=1.0e-12)


def test_symmetry_adapted_frame_falls_back_when_no_unitary_resolves_the_basis() -> None:
    time_reversal = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    frame = derive_symmetry_adapted_internal_frame(
        [SymmetryGaugeOperation("only_antiunitary", time_reversal, antiunitary=True, power=2)]
    )

    assert frame.status == "not_applicable"
    assert frame.primary_operation is None
    np.testing.assert_array_equal(frame.unitary, np.eye(2, dtype=np.complex128))


def test_symmetry_adapted_frame_uses_an_available_order_two_generator() -> None:
    generator = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    frame = derive_symmetry_adapted_internal_frame(
        [SymmetryGaugeOperation("generic_order_two", generator, antiunitary=False, power=2)]
    )

    assert frame.status == "applied"
    assert frame.primary_operation == "generic_order_two"
    transformed = transform_internal_operation(generator, frame.unitary, antiunitary=False)
    np.testing.assert_allclose(transformed, np.diag([1.0, -1.0]), atol=1.0e-12)


def _block_monomial_operation(internal: np.ndarray, *, n_q: int, permutation: list[int]) -> np.ndarray:
    n_orb = int(internal.shape[0])
    matrix = np.zeros((n_orb * n_q, n_orb * n_q), dtype=np.complex128)
    for source_q, target_q in enumerate(permutation):
        source = [orbital * n_q + source_q for orbital in range(n_orb)]
        target = [orbital * n_q + target_q for orbital in range(n_orb)]
        matrix[np.ix_(target, source)] = internal
    return matrix


def test_basis_frame_is_derived_from_available_actions_without_family_names() -> None:
    rotation, time_reversal, secondary = _gamma_spinful_four_state_operations()
    n_q = 3
    matrices = {
        "g": _block_monomial_operation(rotation, n_q=n_q, permutation=[1, 2, 0]),
        "a": _block_monomial_operation(time_reversal, n_q=n_q, permutation=[0, 2, 1]),
        "s": _block_monomial_operation(secondary, n_q=n_q, permutation=[0, 1, 2]),
    }
    frame = derive_symmetry_adapted_basis_frame(
        matrices,
        operations=[
            {"name": "g", "antiunitary": False, "power": 3},
            {"name": "a", "antiunitary": True, "power": 2},
            {"name": "s", "antiunitary": False, "power": 2},
        ],
        sectors=[{"name": "only", "n_orb": 4, "n_q": n_q}],
    )

    assert frame.status == "applied"
    transformed = transform_basis_operation(matrices["g"], frame.full_unitary, antiunitary=False)
    for source_q, target_q in enumerate([1, 2, 0]):
        source = [orbital * n_q + source_q for orbital in range(4)]
        target = [orbital * n_q + target_q for orbital in range(4)]
        block = transformed[np.ix_(target, source)]
        np.testing.assert_allclose(block, np.diag(np.diag(block)), atol=1.0e-12)

    restored = SymmetryAdaptedBasisFrame.from_artifact(frame.artifact())
    np.testing.assert_allclose(restored.full_unitary, frame.full_unitary, atol=1.0e-15)


def test_basis_frame_keeps_identity_when_little_group_has_no_resolving_unitary() -> None:
    antiunitary = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
    full = _block_monomial_operation(antiunitary, n_q=2, permutation=[1, 0])
    frame = derive_symmetry_adapted_basis_frame(
        {"generic_a": full},
        operations=[{"name": "generic_a", "antiunitary": True, "power": 2}],
        sectors=[{"name": "only", "n_orb": 2, "n_q": 2}],
    )

    assert frame.status == "not_applicable"
    np.testing.assert_array_equal(frame.full_unitary, np.eye(4, dtype=np.complex128))


def test_hamiltonian_and_wavefunctions_are_rotated_in_the_same_frame() -> None:
    generator = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    frame = derive_symmetry_adapted_basis_frame(
        {"generic": generator},
        operations=[{"name": "generic", "antiunitary": False, "power": 2}],
        sectors=[{"name": "only", "n_orb": 2, "n_q": 1}],
    )
    hamiltonian = np.array([[2.0, 1.0j], [-1.0j, -0.5]], dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    transformed_hamiltonian = transform_basis_hamiltonian(hamiltonian, frame.full_unitary)
    transformed_vectors = transform_basis_vectors(eigenvectors, frame.full_unitary)

    np.testing.assert_allclose(
        transformed_hamiltonian @ transformed_vectors,
        transformed_vectors * eigenvalues[np.newaxis, :],
        atol=1.0e-12,
    )


def test_cross_sector_action_fixes_relative_kramers_pair_phases() -> None:
    rotation, time_reversal, exchange = _gamma_spinful_four_state_operations()
    rotation_full = np.kron(np.eye(2), rotation)
    time_reversal_full = np.kron(np.eye(2), time_reversal)
    exchange_full = np.block(
        [
            [np.zeros_like(exchange), exchange],
            [exchange, np.zeros_like(exchange)],
        ]
    )
    frame = derive_symmetry_adapted_basis_frame(
        {"g": rotation_full, "a": time_reversal_full, "x": exchange_full},
        operations=[
            {"name": "g", "antiunitary": False, "power": 3},
            {"name": "a", "antiunitary": True, "power": 2},
            {"name": "x", "antiunitary": False, "power": 2},
        ],
        sectors=[
            {"name": "left", "n_orb": 4, "n_q": 1},
            {"name": "right", "n_orb": 4, "n_q": 1},
        ],
    )
    transformed = transform_basis_operation(exchange_full, frame.full_unitary, antiunitary=False)
    expected = np.kron(np.eye(2), -1.0j * np.array([[0.0, 1.0], [1.0, 0.0]]))

    np.testing.assert_allclose(transformed[4:8, 0:4], expected, atol=1.0e-12)
    np.testing.assert_allclose(transformed[0:4, 4:8], expected, atol=1.0e-12)
