from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import kp.model.response_basis as response_basis
from kp.model.response_basis import (
    FiniteGroupElement,
    PolynomialCoordinateBasis,
    RawPolynomialSeed,
    compile_candidate_responses_adjoint_canonicalized,
    identity_finite_group,
    raw_polynomial_seed_from_term_key,
)
from kp.symmetry.factorized_action import certify_factorized_action


def _api():
    return importlib.import_module("kp.model.response_basis_factorized")


def _complete_degree_one_seeds() -> list[RawPolynomialSeed]:
    seeds: list[RawPolynomialSeed] = []
    seed_index = 0
    for layer in (1, 2):
        offset = 0 if layer == 1 else 2
        for orbital_from in range(2):
            for orbital_to in range(2):
                for r, s in ((1, 0), (0, 1)):
                    matrix = sparse.csr_matrix(
                        (
                            np.asarray([1.0 + 0.0j]),
                            (
                                np.asarray([offset + orbital_from]),
                                np.asarray([offset + orbital_to]),
                            ),
                        ),
                        shape=(4, 4),
                    )
                    seeds.append(
                        RawPolynomialSeed(
                            seed_id=f"seed:{seed_index}",
                            coefficients={(r, s): matrix},
                            support_component="kinetic",
                            metadata={
                                "term_index": seed_index,
                                "term_key": {
                                    "Mz": r,
                                    "Mz_star": s,
                                    "layer_from": layer,
                                    "layer_to": layer,
                                    "orbital_from": orbital_from + 1,
                                    "orbital_to": orbital_to + 1,
                                    "p": [0.0, 0.0],
                                },
                            },
                        )
                    )
                    seed_index += 1
    return seeds


def _complete_degree_two_seeds() -> list[RawPolynomialSeed]:
    seeds: list[RawPolynomialSeed] = []
    seed_index = 0
    monomials = tuple(
        (r, degree - r)
        for degree in range(3)
        for r in range(degree + 1)
    )
    for layer in (1, 2):
        offset = 0 if layer == 1 else 2
        for orbital_from in range(2):
            for orbital_to in range(2):
                for r, s in monomials:
                    matrix = sparse.csr_matrix(
                        (
                            np.asarray([1.0 + 0.0j]),
                            (
                                np.asarray([offset + orbital_from]),
                                np.asarray([offset + orbital_to]),
                            ),
                        ),
                        shape=(4, 4),
                    )
                    seeds.append(
                        RawPolynomialSeed(
                            seed_id=f"degree-two:{seed_index}",
                            coefficients={(r, s): matrix},
                            support_component="kinetic",
                            metadata={
                                "term_index": seed_index,
                                "term_key": {
                                    "Mz": r,
                                    "Mz_star": s,
                                    "layer_from": layer,
                                    "layer_to": layer,
                                    "orbital_from": orbital_from + 1,
                                    "orbital_to": orbital_to + 1,
                                    "p": [0.0, 0.0],
                                },
                            },
                        )
                    )
                    seed_index += 1
    return seeds


def test_factored_term_action_matches_full_matrix_formula() -> None:
    api = _api()
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=1,
    )
    seeds = _complete_degree_one_seeds()
    logical = compile_candidate_responses_adjoint_canonicalized(
        seeds,
        joint_keys=response_basis._zero_harmonic_joint_adjoint_keys(
            seeds,
            coordinate=coordinate,
        ),
        coordinate=coordinate,
        group=identity_finite_group(4, q_size=2, sector_size=2),
    )
    physical = [
        channel
        for channel in logical.channels
        if channel.classification != "structural_zero"
        and not bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    vocabulary = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 4)
            for channel in physical
        ],
        format="csc",
    )

    internal_u = np.zeros((4, 4), dtype=np.complex128)
    internal_u[2:, :2] = np.eye(2)
    internal_u[:2, 2:] = np.eye(2)
    reflection = np.diag([1.0, -1.0])
    element = FiniteGroupElement(
        canonical_word=("C2",),
        antiunitary=False,
        canonical_k_map=((1, 0), (0, -1)),
        q_permutation=(1, 0),
        sector_permutation=(1, 0),
        k_pullback=((1.0, 0.0), (0.0, -1.0)),
        internal_u=internal_u,
    )
    dense_images = []
    for channel in physical:
        transformed = response_basis._apply_group_element(
            channel.coefficients,
            element,
            coordinate,
        )
        dense_images.append(
            response_basis._channel_sparse_vector(
                replace(channel, coefficients=transformed),
                coordinate,
                4,
            )
        )
    dense_generator_image = sparse.hstack(dense_images, format="csc")
    factorized = certify_factorized_action(
        name="C2",
        matrix=internal_u,
        antiunitary=False,
        k_forward=reflection,
        q_permutation=(1, 0),
        sector_permutation=(1, 0),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(2, 2),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    action = api.compile_factorized_real_channel_action(
        seeds=seeds,
        logical_channels=logical.channels,
        physical_channel_ids=tuple(channel.channel_id for channel in physical),
        factorized_action=factorized,
    )
    factored_generator_image = (vocabulary @ action).tocsc()

    np.testing.assert_allclose(
        factored_generator_image.toarray(),
        dense_generator_image.toarray(),
        atol=2.0e-14,
    )
    assert action.shape == (len(physical), len(physical))
    assert action.nnz == len(physical)


def test_factored_term_action_matches_dense_orbital_block_formula() -> None:
    api = _api()
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=1,
    )
    seeds = _complete_degree_one_seeds()
    logical = compile_candidate_responses_adjoint_canonicalized(
        seeds,
        joint_keys=response_basis._zero_harmonic_joint_adjoint_keys(
            seeds,
            coordinate=coordinate,
        ),
        coordinate=coordinate,
        group=identity_finite_group(4, q_size=2, sector_size=2),
    )
    physical = [
        channel
        for channel in logical.channels
        if channel.classification != "structural_zero"
        and not bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    vocabulary = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 4)
            for channel in physical
        ],
        format="csc",
    )

    hadamard = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    internal_u = np.zeros((4, 4), dtype=np.complex128)
    internal_u[2:, :2] = hadamard
    internal_u[:2, 2:] = hadamard
    reflection = np.diag([1.0, -1.0])
    element = FiniteGroupElement(
        canonical_word=("C2",),
        antiunitary=False,
        canonical_k_map=((1, 0), (0, -1)),
        q_permutation=(1, 0),
        sector_permutation=(1, 0),
        k_pullback=((1.0, 0.0), (0.0, -1.0)),
        internal_u=internal_u,
    )
    dense_generator_image = sparse.hstack(
        [
            response_basis._channel_sparse_vector(
                replace(
                    channel,
                    coefficients=response_basis._apply_group_element(
                        channel.coefficients,
                        element,
                        coordinate,
                    ),
                ),
                coordinate,
                4,
            )
            for channel in physical
        ],
        format="csc",
    )
    factorized = certify_factorized_action(
        name="C2",
        matrix=internal_u,
        antiunitary=False,
        k_forward=reflection,
        q_permutation=(1, 0),
        sector_permutation=(1, 0),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(2, 2),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    action = api.compile_factorized_real_channel_action(
        seeds=seeds,
        logical_channels=logical.channels,
        physical_channel_ids=tuple(channel.channel_id for channel in physical),
        factorized_action=factorized,
    )

    np.testing.assert_allclose(
        (vocabulary @ action).toarray(),
        dense_generator_image.toarray(),
        atol=2.0e-14,
    )
    assert action.nnz > len(physical)


def test_factored_antiunitary_term_action_matches_full_matrix_formula() -> None:
    api = _api()
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=1,
    )
    seeds = _complete_degree_one_seeds()
    logical = compile_candidate_responses_adjoint_canonicalized(
        seeds,
        joint_keys=response_basis._zero_harmonic_joint_adjoint_keys(
            seeds,
            coordinate=coordinate,
        ),
        coordinate=coordinate,
        group=identity_finite_group(4, q_size=2, sector_size=2),
    )
    physical = [
        channel
        for channel in logical.channels
        if channel.classification != "structural_zero"
        and not bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    vocabulary = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 4)
            for channel in physical
        ],
        format="csc",
    )
    orbital_tr = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    internal_u = np.zeros((4, 4), dtype=np.complex128)
    internal_u[:2, :2] = orbital_tr
    internal_u[2:, 2:] = orbital_tr
    element = FiniteGroupElement(
        canonical_word=("TR",),
        antiunitary=True,
        canonical_k_map=((-1, 0), (0, -1)),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        k_pullback=((-1.0, 0.0), (0.0, -1.0)),
        internal_u=internal_u,
    )
    dense_generator_image = sparse.hstack(
        [
            response_basis._channel_sparse_vector(
                replace(
                    channel,
                    coefficients=response_basis._apply_group_element(
                        channel.coefficients,
                        element,
                        coordinate,
                    ),
                ),
                coordinate,
                4,
            )
            for channel in physical
        ],
        format="csc",
    )
    factorized = certify_factorized_action(
        name="TR",
        matrix=internal_u,
        antiunitary=True,
        k_forward=-np.eye(2),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(2, 2),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    action = api.compile_factorized_real_channel_action(
        seeds=seeds,
        logical_channels=logical.channels,
        physical_channel_ids=tuple(channel.channel_id for channel in physical),
        factorized_action=factorized,
    )

    np.testing.assert_allclose(
        (vocabulary @ action).toarray(),
        dense_generator_image.toarray(),
        atol=2.0e-14,
    )


def test_generator_fixed_compiler_uses_factorized_term_action() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=1,
    )
    seeds = _complete_degree_one_seeds()
    internal_u = np.zeros((4, 4), dtype=np.complex128)
    internal_u[2:, :2] = np.eye(2)
    internal_u[:2, 2:] = np.eye(2)
    group = response_basis.build_finite_group(
        [
            response_basis.FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((1, 0), (0, -1)),
                q_permutation=(1, 0),
                sector_permutation=(1, 0),
                k_forward=((1.0, 0.0), (0.0, -1.0)),
                internal_u=internal_u,
            )
        ]
    )
    factorized = certify_factorized_action(
        name="C2",
        matrix=internal_u,
        antiunitary=False,
        k_forward=np.diag([1.0, -1.0]),
        q_permutation=(1, 0),
        sector_permutation=(1, 0),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(2, 2),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    compiled = response_basis.compile_generator_fixed_response_group(
        seeds,
        joint_keys=response_basis._zero_harmonic_joint_adjoint_keys(
            seeds,
            coordinate=coordinate,
        ),
        coordinate=coordinate,
        group=group,
        reduce=True,
        materialize_ambient_vectors=False,
        factorized_actions={"C2": factorized},
    )

    artifact = compiled.candidates.adjoint_artifact
    assert artifact["generator_image_compiler"] == "factorized_q_route_term_action_v2"
    assert artifact["generator_image_raw_transform_count"] == 0

    assert artifact["factorized_action_nnz"]["C2"] > 0
    assert (
        compiled.reduction_proofs[0]["solver"]
        == "generator_fixed_subspace__direct_term_action_metric_v1"
    )

    reference = response_basis.compile_generator_fixed_response_group(
        seeds,
        joint_keys=response_basis._zero_harmonic_joint_adjoint_keys(
            seeds,
            coordinate=coordinate,
        ),
        coordinate=coordinate,
        group=group,
        reduce=True,
        materialize_ambient_vectors=False,
    )
    direct_vectors = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 4)
            for channel in compiled.retained_channels
        ],
        format="csc",
    ).toarray()
    reference_vectors = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 4)
            for channel in reference.retained_channels
        ],
        format="csc",
    ).toarray()
    direct_projector = direct_vectors @ np.linalg.pinv(direct_vectors)
    reference_projector = reference_vectors @ np.linalg.pinv(reference_vectors)
    np.testing.assert_allclose(direct_projector, reference_projector, atol=3.0e-12)


def test_c3_degree_two_action_cleans_only_certified_structural_zero_tails() -> None:
    api = _api()
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=2,
    )
    seeds = _complete_degree_two_seeds()
    logical = compile_candidate_responses_adjoint_canonicalized(
        seeds,
        joint_keys=response_basis._zero_harmonic_joint_adjoint_keys(
            seeds,
            coordinate=coordinate,
        ),
        coordinate=coordinate,
        group=identity_finite_group(4, q_size=2, sector_size=2),
    )
    physical = [
        channel
        for channel in logical.channels
        if channel.classification != "structural_zero"
        and not bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    vocabulary = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 4)
            for channel in physical
        ],
        format="csc",
    )
    angle = 2.0 * np.pi / 3.0
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    orbital = np.diag(
        np.asarray(
            [
                np.exp(-1.0j * angle),
                np.exp(1.0j * angle),
            ]
        )
    )
    internal_u = np.zeros((4, 4), dtype=np.complex128)
    internal_u[:2, :2] = orbital
    internal_u[2:, 2:] = orbital
    factorized = certify_factorized_action(
        name="C3z",
        matrix=internal_u,
        antiunitary=False,
        k_forward=rotation,
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(2, 2),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )
    element = FiniteGroupElement(
        canonical_word=("C3z",),
        antiunitary=False,
        canonical_k_map=((-1, -1), (1, 0)),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        k_pullback=tuple(tuple(float(value) for value in row) for row in np.linalg.inv(rotation)),
        internal_u=internal_u,
    )
    dense = sparse.hstack(
        [
            response_basis._channel_sparse_vector(
                replace(
                    channel,
                    coefficients=response_basis._apply_group_element(
                        channel.coefficients,
                        element,
                        coordinate,
                    ),
                ),
                coordinate,
                4,
            )
            for channel in physical
        ],
        format="csc",
    )

    action = api.compile_factorized_real_channel_action(
        seeds=seeds,
        logical_channels=logical.channels,
        physical_channel_ids=tuple(channel.channel_id for channel in physical),
        factorized_action=factorized,
    )

    np.testing.assert_allclose((vocabulary @ action).toarray(), dense.toarray(), atol=2.0e-12)


def test_factorized_action_matches_dense_formula_for_cross_sector_p0_seeds() -> None:
    api = _api()
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seeds: list[RawPolynomialSeed] = []
    seed_index = 0
    for layer_from, layer_to in ((1, 2), (2, 1)):
        row_offset = 0 if layer_from == 1 else 2
        col_offset = 0 if layer_to == 1 else 2
        for orbital_from in range(2):
            for orbital_to in range(2):
                matrix = sparse.csr_matrix(
                    (
                        np.asarray([1.0 + 0.0j]),
                        (
                            np.asarray([row_offset + orbital_from]),
                            np.asarray([col_offset + orbital_to]),
                        ),
                    ),
                    shape=(4, 4),
                )
                seeds.append(
                    RawPolynomialSeed(
                        seed_id=f"cross-p0:{seed_index}",
                        coefficients={(0, 0): matrix},
                        support_component="inter",
                        metadata={
                            "term_index": seed_index,
                            "term_key": {
                                "Mz": 0,
                                "Mz_star": 0,
                                "layer_from": layer_from,
                                "layer_to": layer_to,
                                "orbital_from": orbital_from + 1,
                                "orbital_to": orbital_to + 1,
                                "p": [0.0, 0.0],
                            },
                        },
                    )
                )
                seed_index += 1
    logical = compile_candidate_responses_adjoint_canonicalized(
        seeds,
        joint_keys=response_basis._zero_harmonic_joint_adjoint_keys(
            seeds,
            coordinate=coordinate,
        ),
        coordinate=coordinate,
        group=identity_finite_group(4, q_size=2, sector_size=2),
    )
    physical = [
        channel
        for channel in logical.channels
        if channel.classification != "structural_zero"
        and not bool(channel.metadata.get("adjoint_certified_dependency", False))
    ]
    vocabulary = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 4)
            for channel in physical
        ],
        format="csc",
    )
    block1 = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    block2 = np.asarray(
        [[1.0, -1.0j], [-1.0j, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    internal_u = np.zeros((4, 4), dtype=np.complex128)
    internal_u[:2, :2] = block1
    internal_u[2:, 2:] = np.exp(0.37j) * block2
    element = FiniteGroupElement(
        canonical_word=("C2",),
        antiunitary=False,
        canonical_k_map=((1, 0), (0, 1)),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        k_pullback=((1.0, 0.0), (0.0, 1.0)),
        internal_u=internal_u,
    )
    dense = sparse.hstack(
        [
            response_basis._channel_sparse_vector(
                replace(
                    channel,
                    coefficients=response_basis._apply_group_element(
                        channel.coefficients,
                        element,
                        coordinate,
                    ),
                ),
                coordinate,
                4,
            )
            for channel in physical
        ],
        format="csc",
    )
    factorized = certify_factorized_action(
        name="C2",
        matrix=internal_u,
        antiunitary=False,
        k_forward=np.eye(2),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(2, 2),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    action = api.compile_factorized_real_channel_action(
        seeds=seeds,
        logical_channels=logical.channels,
        physical_channel_ids=tuple(channel.channel_id for channel in physical),
        factorized_action=factorized,
    )

    np.testing.assert_allclose(
        (vocabulary @ action).toarray(),
        dense.toarray(),
        atol=2.0e-13,
    )


@pytest.mark.parametrize("antiunitary", [False, True])
def test_factorized_action_matches_dense_formula_for_finite_p_and_q_phases(
    antiunitary: bool,
) -> None:
    api = _api()
    q_vectors = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seeds: list[RawPolynomialSeed] = []
    seed_index = 0
    for layer_from, layer_to in ((1, 2), (2, 1)):
        for p_x in (-1.0, 1.0):
            key = SimpleNamespace(
                Mz=0,
                Mz_star=0,
                layer_from=layer_from,
                layer_to=layer_to,
                orbital_from=1,
                orbital_to=1,
                p=(p_x, 0.0),
            )
            seeds.append(
                raw_polynomial_seed_from_term_key(
                    key,
                    seed_id=f"finite-p:{seed_index}",
                    Q_set1=q_vectors,
                    Q_set2=q_vectors,
                    n_orb1=1,
                    n_orb2=1,
                    coordinate=coordinate,
                    support_component="inter",
                    metadata={
                        "term_index": seed_index,
                        "term_space_policy": "explicit_reduced",
                    },
                )
            )
            seed_index += 1
    logical = response_basis.compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=identity_finite_group(6, q_size=6, sector_size=2),
    )
    physical = [
        channel for channel in logical.channels
        if channel.classification != "structural_zero"
    ]
    vocabulary = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 6)
            for channel in physical
        ],
        format="csc",
    )
    q_permutation = (2, 1, 0, 5, 4, 3)
    alpha = 0.31
    beta = -0.27
    phases = tuple(
        [np.exp(1.0j * alpha * q[0]) for q in q_vectors]
        + [np.exp(1.0j * (beta + alpha * q[0])) for q in q_vectors]
    )
    internal_u = np.zeros((6, 6), dtype=np.complex128)
    for source_q, target_q in enumerate(q_permutation):
        internal_u[target_q, source_q] = phases[source_q]
    reflection = -np.eye(2)
    element = FiniteGroupElement(
        canonical_word=(("TR",) if antiunitary else ("C2",)),
        antiunitary=antiunitary,
        canonical_k_map=((-1, 0), (0, -1)),
        q_permutation=q_permutation,
        sector_permutation=(0, 1),
        k_pullback=tuple(tuple(float(value) for value in row) for row in reflection),
        internal_u=internal_u,
    )
    dense = sparse.hstack(
        [
            response_basis._channel_sparse_vector(
                replace(
                    channel,
                    coefficients=response_basis._apply_group_element(
                        channel.coefficients,
                        element,
                        coordinate,
                    ),
                ),
                coordinate,
                6,
            )
            for channel in physical
        ],
        format="csc",
    )
    factorized = certify_factorized_action(
        name="TR" if antiunitary else "C2",
        matrix=internal_u,
        antiunitary=antiunitary,
        k_forward=reflection,
        q_permutation=q_permutation,
        sector_permutation=(0, 1),
        q_vectors=(q_vectors, q_vectors),
        q_counts=(3, 3),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    action = api.compile_factorized_real_channel_action(
        seeds=seeds,
        logical_channels=logical.channels,
        physical_channel_ids=tuple(channel.channel_id for channel in physical),
        factorized_action=factorized,
    )

    np.testing.assert_allclose(
        (vocabulary @ action).toarray(),
        dense.toarray(),
        atol=3.0e-13,
    )

    raw_channels = []
    raw_vectors = []
    raw_dense_images = []
    for seed in seeds:
        for component, phase in (("real", 1.0 + 0.0j), ("imag", 1.0j)):
            coefficients = {
                monomial: sparse.csr_matrix(matrix * phase)
                for monomial, matrix in seed.coefficients.items()
            }
            channel = SimpleNamespace(
                channel_id=f"{seed.seed_id}:{component}",
                seed_id=seed.seed_id,
                component=component,
                coefficients=coefficients,
            )
            raw_channels.append(channel)
            raw_vectors.append(
                response_basis._channel_sparse_vector(channel, coordinate, 6)
            )
            raw_dense_images.append(
                response_basis._channel_sparse_vector(
                    SimpleNamespace(
                        channel_id=channel.channel_id,
                        coefficients=response_basis._apply_group_element(
                            coefficients,
                            element,
                            coordinate,
                        ),
                    ),
                    coordinate,
                    6,
                )
            )
    raw_vocabulary = sparse.hstack(raw_vectors, format="csc")
    raw_dense = sparse.hstack(raw_dense_images, format="csc")
    raw_action = api.compile_factorized_raw_seed_action(
        seeds=seeds,
        factorized_action=factorized,
    )

    assert raw_action.shape == (len(raw_channels), len(raw_channels))
    np.testing.assert_allclose(
        (raw_vocabulary @ raw_action).toarray(),
        raw_dense.toarray(),
        atol=3.0e-13,
    )


def test_factorized_action_rejects_nonconstant_q_phase_ratio() -> None:
    api = _api()
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    matrix = sparse.csr_matrix(
        (
            np.asarray([1.0 + 0.0j, 1.0 + 0.0j]),
            (
                np.asarray([0, 1]),
                np.asarray([2, 3]),
            ),
        ),
        shape=(4, 4),
    )
    seed = RawPolynomialSeed(
        seed_id="nonconstant-phase",
        coefficients={(0, 0): matrix},
        support_component="inter",
        metadata={
            "term_index": 0,
            "term_key": {
                "Mz": 0,
                "Mz_star": 0,
                "layer_from": 1,
                "layer_to": 2,
                "orbital_from": 1,
                "orbital_to": 1,
                "p": [0.0, 0.0],
            },
        },
    )
    logical = response_basis.compile_candidate_responses(
        [seed],
        coordinate=coordinate,
        group=identity_finite_group(4, q_size=4, sector_size=2),
    )
    physical = [
        channel for channel in logical.channels
        if channel.classification != "structural_zero"
    ]
    phases = np.asarray([1.0, 1.0, 1.0, 1.0j], dtype=np.complex128)
    internal_u = np.diag(phases)
    factorized = certify_factorized_action(
        name="phase-test",
        matrix=internal_u,
        antiunitary=False,
        k_forward=np.eye(2),
        q_permutation=(0, 1, 2, 3),
        sector_permutation=(0, 1),
        q_vectors=(
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        ),
        q_counts=(2, 2),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    with pytest.raises(api.FactorizedTermActionError, match="not constant"):
        api.compile_factorized_real_channel_action(
            seeds=[seed],
            logical_channels=logical.channels,
            physical_channel_ids=tuple(channel.channel_id for channel in physical),
            factorized_action=factorized,
        )


def test_generator_fixed_compiler_accepts_finite_p_without_joint_keys() -> None:
    q_vectors = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seeds: list[RawPolynomialSeed] = []
    seed_index = 0
    for layer_from, layer_to in ((1, 2), (2, 1)):
        for p_x in (-1.0, 1.0):
            seeds.append(
                raw_polynomial_seed_from_term_key(
                    SimpleNamespace(
                        Mz=0,
                        Mz_star=0,
                        layer_from=layer_from,
                        layer_to=layer_to,
                        orbital_from=1,
                        orbital_to=1,
                        p=(p_x, 0.0),
                    ),
                    seed_id=f"fixed-finite-p:{seed_index}",
                    Q_set1=q_vectors,
                    Q_set2=q_vectors,
                    n_orb1=1,
                    n_orb2=1,
                    coordinate=coordinate,
                    support_component="inter",
                    metadata={
                        "term_index": seed_index,
                        "term_space_policy": "explicit_reduced",
                    },
                )
            )
            seed_index += 1
    q_permutation = (2, 1, 0, 5, 4, 3)
    internal_u = np.zeros((6, 6), dtype=np.complex128)
    for source_q, target_q in enumerate(q_permutation):
        internal_u[target_q, source_q] = 1.0
    reflection = -np.eye(2)
    group = response_basis.build_finite_group(
        [
            response_basis.FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=q_permutation,
                sector_permutation=(0, 1),
                k_forward=tuple(
                    tuple(float(value) for value in row) for row in reflection
                ),
                internal_u=internal_u,
            )
        ]
    )
    factorized = certify_factorized_action(
        name="C2",
        matrix=internal_u,
        antiunitary=False,
        k_forward=reflection,
        q_permutation=q_permutation,
        sector_permutation=(0, 1),
        q_vectors=(q_vectors, q_vectors),
        q_counts=(3, 3),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    compiled = response_basis.compile_generator_fixed_response_group(
        seeds,
        joint_keys=None,
        coordinate=coordinate,
        group=group,
        reduce=True,
        materialize_ambient_vectors=False,
        factorized_actions={"C2": factorized},
    )

    artifact = compiled.candidates.adjoint_artifact
    assert artifact["generator_image_compiler"] == "factorized_q_route_term_action_v2"
    assert artifact["generator_image_raw_transform_count"] == 0

    production = response_basis._compile_model_response_basis_uncached(
        coordinate=coordinate,
        groups=[group],
        seeds_by_group=[seeds],
        factorized_actions_by_group=[{"C2": factorized}],
        dim=6,
        identity_payload={"test": "finite-p-production-dispatch"},
        reduce=True,
        cache_key="finite-p-production-dispatch-test",
        progress_callback=None,
    )
    production_artifact = production.candidate_artifact["adjoint"]["groups"][0]
    reynolds_action = production_artifact["reynolds_internal_action"]
    assert reynolds_action["compiler"] == "factorized_sparse_group_elements_v1"
    assert reynolds_action["dense_internal_transform_count"] == 0

    zero_harmonic_seeds = [
        raw_polynomial_seed_from_term_key(
            SimpleNamespace(
                Mz=0,
                Mz_star=0,
                layer_from=layer_from,
                layer_to=layer_to,
                orbital_from=1,
                orbital_to=1,
                p=(0.0, 0.0),
            ),
            seed_id=f"mixed-p0:{offset}",
            Q_set1=q_vectors,
            Q_set2=q_vectors,
            n_orb1=1,
            n_orb2=1,
            coordinate=coordinate,
            support_component="inter",
            metadata={
                "term_index": len(seeds) + offset,
                "term_space_policy": "explicit_reduced",
            },
        )
        for offset, (layer_from, layer_to) in enumerate(((1, 2), (2, 1)))
    ]
    mixed = response_basis._compile_model_response_basis_uncached(
        coordinate=coordinate,
        groups=[group],
        seeds_by_group=[[*seeds, *zero_harmonic_seeds]],
        factorized_actions_by_group=[{"C2": factorized}],
        dim=6,
        identity_payload={"test": "mixed-p0-finite-p-production-dispatch"},
        reduce=True,
        cache_key="mixed-p0-finite-p-production-dispatch-test",
        progress_callback=None,
    )
    mixed_artifacts = mixed.candidate_artifact["adjoint"]["groups"]
    assert len(mixed_artifacts) == 2
    assert mixed_artifacts[0]["generator_image_compiler"] == "factorized_q_route_term_action_v2"
    assert mixed_artifacts[0]["generator_image_raw_transform_count"] == 0
    assert (
        mixed_artifacts[1]["reynolds_internal_action"]["compiler"]
        == "factorized_sparse_group_elements_v1"
    )
    assert (
        mixed_artifacts[1]["reynolds_internal_action"]["dense_internal_transform_count"]
        == 0
    )


def test_sparse_factorized_reynolds_matches_dense_for_nonclosed_harmonic_orbit() -> None:
    api = _api()
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seed = RawPolynomialSeed(
        seed_id="finite-p-orbit-representative",
        coefficients={(0, 0): sparse.csr_matrix(([1.0], ([0], [1])), shape=(2, 2))},
        support_component="finite-p",
        metadata={
            "term_index": 0,
            "term_key": {
                "Mz": 0,
                "Mz_star": 0,
                "layer_from": 1,
                "layer_to": 1,
                "orbital_from": 1,
                "orbital_to": 1,
                "p": [2.0, 0.0],
            },
        },
    )
    internal_u = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    group = response_basis.build_finite_group(
        [
            response_basis.FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(1, 0),
                sector_permutation=(0,),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=internal_u,
            )
        ]
    )
    factorized = certify_factorized_action(
        name="C2",
        matrix=internal_u,
        antiunitary=False,
        k_forward=-np.eye(2),
        q_permutation=(1, 0),
        sector_permutation=(0,),
        q_vectors=(np.asarray([[1.0, 0.0], [-1.0, 0.0]]),),
        q_counts=(2,),
        n_orb=(1,),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    dense = response_basis.compile_candidate_responses(
        [seed],
        coordinate=coordinate,
        group=group,
    )
    internal_actions, artifact = api.compile_factorized_group_element_actions(
        group=group,
        factorized_generators={"C2": factorized},
    )
    fast = response_basis.compile_candidate_responses(
        [seed],
        coordinate=coordinate,
        group=group,
        internal_actions_by_word=internal_actions,
        internal_action_absolute_error_bound=artifact["maximum_action_error_bound"],
    )

    assert artifact["compiler"] == "factorized_sparse_group_elements_v1"
    assert artifact["dense_internal_transform_count"] == 0
    assert len(internal_actions) == len(group.elements)
    for dense_channel, fast_channel in zip(dense.channels, fast.channels):
        assert dense_channel.channel_id == fast_channel.channel_id
        assert dense_channel.classification == fast_channel.classification
        for monomial in set(dense_channel.coefficients) | set(fast_channel.coefficients):
            np.testing.assert_allclose(
                dense_channel.coefficient(*monomial).toarray(),
                fast_channel.coefficient(*monomial).toarray(),
                atol=2.0e-14,
            )
