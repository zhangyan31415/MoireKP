from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import kp.model.response_basis as response_basis
from kp.model.response_basis import (
    FiniteGroupGenerator,
    PolynomialCoordinateBasis,
    build_finite_group,
    raw_polynomial_seed_from_term_key,
)
from kp.symmetry.factorized_action import certify_factorized_action
from kp.model.response_basis_factorized import FactorizedTermActionError


def _api():
    return importlib.import_module("kp.model.response_basis_symmetry_first")


def _finite_p_degree_one_seed():
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=1,
    )
    q_vectors = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    seed = raw_polynomial_seed_from_term_key(
        SimpleNamespace(
            Mz=1,
            Mz_star=0,
            layer_from=1,
            layer_to=2,
            orbital_from=1,
            orbital_to=1,
            p=(1.0, 0.0),
        ),
        seed_id="finite-p-degree-one",
        Q_set1=q_vectors,
        Q_set2=q_vectors,
        n_orb1=1,
        n_orb2=1,
        coordinate=coordinate,
        support_component="inter",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )
    return coordinate, seed


def _vector(coefficients, coordinate, dim):
    return response_basis._channel_sparse_vector(
        SimpleNamespace(channel_id="oracle", coefficients=coefficients),
        coordinate,
        dim,
    )


def test_raw_real_vocabulary_contains_real_and_imaginary_seed_directions() -> None:
    api = _api()
    coordinate, seed = _finite_p_degree_one_seed()

    vocabulary = api.build_raw_real_vocabulary([seed], coordinate=coordinate)

    real = _vector(seed.coefficients, coordinate, seed.dim)
    imaginary = _vector(
        {
            monomial: sparse.csr_matrix(1.0j * matrix)
            for monomial, matrix in seed.coefficients.items()
        },
        coordinate,
        seed.dim,
    )
    expected = sparse.hstack([real, imaginary], format="csc")
    assert vocabulary.channel_ids == (
        "finite-p-degree-one:real",
        "finite-p-degree-one:imag",
    )
    np.testing.assert_allclose(vocabulary.matrix.toarray(), expected.toarray())


def test_finite_p_adjoint_image_matches_direct_polynomial_adjoint() -> None:
    api = _api()
    coordinate, seed = _finite_p_degree_one_seed()

    adjoint = api.build_raw_adjoint_image([seed], coordinate=coordinate)

    real_adjoint = response_basis._adjoint_polynomial(seed.coefficients)
    imaginary_adjoint = response_basis._adjoint_polynomial(
        {
            monomial: sparse.csr_matrix(1.0j * matrix)
            for monomial, matrix in seed.coefficients.items()
        }
    )
    expected = sparse.hstack(
        [
            _vector(real_adjoint, coordinate, seed.dim),
            _vector(imaginary_adjoint, coordinate, seed.dim),
        ],
        format="csc",
    )
    np.testing.assert_allclose(adjoint.toarray(), expected.toarray())

    monomial_block = seed.dim * seed.dim
    degree_zero_rows = np.r_[
        0:monomial_block,
        len(coordinate.monomials) * monomial_block :
        (len(coordinate.monomials) + 1) * monomial_block,
    ]
    assert adjoint[degree_zero_rows, :].nnz > 0

    twice = response_basis._adjoint_polynomial(real_adjoint)
    np.testing.assert_allclose(
        _vector(twice, coordinate, seed.dim).toarray(),
        _vector(seed.coefficients, coordinate, seed.dim).toarray(),
    )


def _closed_finite_p_case():
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    q_vectors = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    seeds = []
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
                    seed_id=f"finite:{layer_from}:{layer_to}:{p_x:+.0f}",
                    Q_set1=q_vectors,
                    Q_set2=q_vectors,
                    n_orb1=1,
                    n_orb2=1,
                    coordinate=coordinate,
                    support_component="inter",
                    metadata={"term_space_policy": "complete"},
                )
            )
    q_permutation = (2, 1, 0, 5, 4, 3)
    internal_u = np.zeros((6, 6), dtype=np.complex128)
    for source_q, target_q in enumerate(q_permutation):
        internal_u[target_q, source_q] = 1.0
    reflection = -np.eye(2)
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
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=q_permutation,
                sector_permutation=(0, 1),
                k_forward=tuple(tuple(float(value) for value in row) for row in reflection),
                internal_u=internal_u,
            )
        ]
    )
    return coordinate, seeds, factorized, group


def test_symmetry_first_fixed_space_matches_complete_reynolds_span() -> None:
    api = _api()
    coordinate, seeds, factorized, group = _closed_finite_p_case()
    dense_candidates = response_basis.compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=group,
    )
    dense_basis = response_basis.CompiledResponseBasis.from_candidates(
        dense_candidates,
        identity_payload={"test": "finite-p-symmetry-first"},
        reduce=True,
    )
    dense_vectors = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 6)
            for channel in dense_basis.channels
        ],
        format="csc",
    ).toarray()

    compiled = api.compile_symmetry_first_fixed_space(
        seeds,
        coordinate=coordinate,
        factorized_actions={"C2": factorized},
    )

    assert compiled.rank == len(dense_basis.channels)
    fast_vectors = compiled.physical_basis.toarray()
    dense_projector = dense_vectors @ np.linalg.pinv(dense_vectors)
    fast_projector = fast_vectors @ np.linalg.pinv(fast_vectors)
    np.testing.assert_allclose(fast_projector, dense_projector, atol=5.0e-12)


def test_symmetry_first_projection_does_not_require_authored_seed_closure() -> None:
    api = _api()
    coordinate, closed_seeds, factorized, group = _closed_finite_p_case()
    seeds = [closed_seeds[0]]

    with pytest.raises(FactorizedTermActionError, match="outside the authored seed vocabulary"):
        api.compile_factorized_raw_seed_action(
            seeds=seeds,
            factorized_action=factorized,
        )

    dense_candidates = response_basis.compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=group,
    )
    dense_basis = response_basis.CompiledResponseBasis.from_candidates(
        dense_candidates,
        identity_payload={"test": "finite-p-nonclosed-symbolic-atoms"},
        reduce=True,
    )
    dense_vectors = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 6)
            for channel in dense_basis.channels
        ],
        format="csc",
    ).toarray()

    compiled = api.compile_symmetry_first_fixed_space(
        seeds,
        coordinate=coordinate,
        factorized_actions={"C2": factorized},
        group=group,
    )

    assert compiled.rank == len(dense_basis.channels)
    fast_vectors = compiled.physical_basis.toarray()
    dense_projector = dense_vectors @ np.linalg.pinv(dense_vectors)
    fast_projector = fast_vectors @ np.linalg.pinv(fast_vectors)
    np.testing.assert_allclose(fast_projector, dense_projector, atol=5.0e-12)


def test_symbolic_atom_candidate_group_materializes_only_independent_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinate, closed_seeds, factorized, group = _closed_finite_p_case()
    seeds = [closed_seeds[0]]
    dense = response_basis.compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=group,
    )
    dense_basis = response_basis.CompiledResponseBasis.from_candidates(
        dense,
        identity_payload={"test": "symbolic-atom-candidate-group"},
        reduce=True,
    )

    def fail_dense_reynolds(*_args, **_kwargs):
        raise AssertionError("symbolic compiler entered per-seed CSR Reynolds")

    monkeypatch.setattr(response_basis, "compile_candidate_responses", fail_dense_reynolds)
    compiled = api.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"C2": factorized},
    )

    assert compiled.adjoint_artifact["certification"] == "closed_symbolic_atom_reynolds_v1"
    assert set(compiled.adjoint_artifact["timings_seconds"]) == {
        "group_context",
        "seed_atoms",
        "symbolic_projection",
        "rank_selection",
        "channel_materialization",
    }
    assert compiled.adjoint_artifact["logical_candidate_channel_count"] == 2
    assert len(compiled.channels) == len(dense_basis.channels)
    fast_vectors = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 6)
            for channel in compiled.channels
        ],
        format="csc",
    ).toarray()
    dense_vectors = sparse.hstack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, 6)
            for channel in dense_basis.channels
        ],
        format="csc",
    ).toarray()
    np.testing.assert_allclose(
        fast_vectors @ np.linalg.pinv(fast_vectors),
        dense_vectors @ np.linalg.pinv(dense_vectors),
        atol=5.0e-12,
    )


def test_symbolic_atom_compiles_each_group_context_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinate, seeds, factorized, group = _closed_finite_p_case()
    calls = 0
    original = response_basis._monomial_pullback_table

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(response_basis, "_monomial_pullback_table", counted)
    api.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"C2": factorized},
    )

    assert calls == len(group.elements)


def test_symbolic_atom_extracts_each_seed_support_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinate, seeds, factorized, group = _closed_finite_p_case()
    calls = 0
    original = api._seed_symbolic_atoms

    def counted(seed):
        nonlocal calls
        calls += 1
        return original(seed)

    monkeypatch.setattr(api, "_seed_symbolic_atoms", counted)
    api.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"C2": factorized},
    )

    assert calls == len(seeds)


def test_symbolic_rank_uses_each_columns_propagated_error_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    projected = sparse.csc_matrix(
        np.asarray(
            [
                [1.0, 1.0],
                [0.0, 1.0e-10],
            ],
            dtype=np.float64,
        )
    )

    def fail_redundant_factorization(*_args, **_kwargs):
        raise AssertionError("symbolic rank used a redundant dense factorization")

    monkeypatch.setattr(api.scipy.linalg, "svdvals", fail_redundant_factorization)
    monkeypatch.setattr(api.scipy.linalg, "lstsq", fail_redundant_factorization)
    selected, _proofs = api._independent_projected_columns(
        projected,
        relative_error_bounds=np.asarray([1.0e-8, 1.0e-8]),
        confirm_factor=100.0,
    )

    assert len(selected) == 1


def test_symbolic_projection_does_not_materialize_raw_physical_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinate, seeds, factorized, group = _closed_finite_p_case()

    def fail_raw_vocabulary(*_args, **_kwargs):
        raise AssertionError("symbolic path materialized the raw physical vocabulary")

    monkeypatch.setattr(api, "build_raw_real_vocabulary", fail_raw_vocabulary)
    compiled = api.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"C2": factorized},
    )

    assert compiled.channels


def test_symbolic_projection_reuses_precompiled_factorized_group_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinate, seeds, factorized, group = _closed_finite_p_case()
    internal_actions, artifact = api.compile_factorized_group_element_actions(
        group=group,
        factorized_generators={"C2": factorized},
    )

    def fail_duplicate_compile(*_args, **_kwargs):
        raise AssertionError("symbolic path recompiled certified group actions")

    monkeypatch.setattr(
        api,
        "compile_factorized_group_element_actions",
        fail_duplicate_compile,
    )
    compiled = api.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"C2": factorized},
        internal_actions_by_word=internal_actions,
        factorized_group_artifact=artifact,
    )

    assert compiled.channels


def test_symbolic_projection_batches_all_seeds_without_per_seed_reynolds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    coordinate, seeds, factorized, group = _closed_finite_p_case()

    def fail_per_seed(*_args, **_kwargs):
        raise AssertionError("symbolic projection entered the per-seed Reynolds loop")

    monkeypatch.setattr(api, "_symbolic_reynolds_pair", fail_per_seed)
    compiled = api.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"C2": factorized},
    )

    assert compiled.channels


def test_symbolic_atom_batch_matches_complete_antiunitary_reynolds() -> None:
    api = _api()
    coordinate, seeds, _factorized, _group = _closed_finite_p_case()
    q_vectors = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    q_permutation = (2, 1, 0, 5, 4, 3)
    internal_u = np.zeros((6, 6), dtype=np.complex128)
    for source_q, target_q in enumerate(q_permutation):
        internal_u[target_q, source_q] = 1.0
    reflection = -np.eye(2)
    factorized = certify_factorized_action(
        name="T",
        matrix=internal_u,
        antiunitary=True,
        k_forward=reflection,
        q_permutation=q_permutation,
        sector_permutation=(0, 1),
        q_vectors=(q_vectors, q_vectors),
        q_counts=(3, 3),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="T",
                antiunitary=True,
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
    dense = response_basis.compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=group,
    )
    dense_basis = response_basis.CompiledResponseBasis.from_candidates(
        dense,
        identity_payload={"test": "antiunitary-symbolic-batch"},
        reduce=True,
    )
    compiled = api.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"T": factorized},
    )
    dense_vectors = sparse.hstack(
        [response_basis._channel_sparse_vector(channel, coordinate, 6) for channel in dense_basis.channels],
        format="csc",
    ).toarray()
    fast_vectors = sparse.hstack(
        [response_basis._channel_sparse_vector(channel, coordinate, 6) for channel in compiled.channels],
        format="csc",
    ).toarray()

    assert fast_vectors.shape[1] == dense_vectors.shape[1]
    np.testing.assert_allclose(
        fast_vectors @ np.linalg.pinv(fast_vectors),
        dense_vectors @ np.linalg.pinv(dense_vectors),
        atol=5.0e-12,
    )
