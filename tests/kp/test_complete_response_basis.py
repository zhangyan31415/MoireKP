from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import kp.model.response_basis as response_basis_module

from kp.model.response_basis import (
    COMPLETE_LINEAR_V2,
    COORDINATE_CONVENTION_V1,
    RESPONSE_NORMALIZATION_V1,
    CandidateResponseSet,
    CompiledResponseRuntime,
    CompiledResponseBasis,
    FiniteGroupGenerator,
    NullClassificationPolicy,
    PolynomialCoordinateBasis,
    RawPolynomialSeed,
    SupportClosureError,
    build_finite_group,
    certify_support_closure,
    clear_response_basis_cache,
    compile_candidate_responses,
    compile_candidate_responses_adjoint_canonicalized,
    compile_generator_fixed_response_group,
    compile_model_response_basis,
    finite_group_from_model_actions,
    identity_finite_group,
    raw_polynomial_seed_from_term_key,
    response_basis_cache_stats,
)
from kp.model.response_basis_adjoint import (
    AdjointCertificationError,
    CanonicalDimensionlessCenter,
    JointAdjointSeedKey,
)
from kp.model.response_basis_oracle import dense_direct_response
from kp.model.response_basis_cache import ResponseBasisCacheCorruptionError
from kp.symmetry.factorized_action import certify_factorized_action
from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    materialize_block_route_action,
)
from kp.model.core import (
    ContinuumTermKey,
    MoireConfig,
    build_model,
    compute_bands,
    compute_coefficients,
)
from kp.model.pipeline import (
    _case_derived_term_templates,
    _public_band_residual_and_jacobian,
    _public_hamiltonian_quadratic_residual,
    _refine_public_nonlinear_complete_response,
    _model_hamiltonians_for_kpoints,
    _prune_small_coefficients,
    _sync_complete_response_coefficients_to_terms,
    refine_band_coefficients,
)


def _identity_payload(dim: int) -> dict[str, object]:
    return {
        "basis_layout": {"order": "toy", "dim": dim},
        "q_vectors": np.zeros((1, 2), dtype=np.float64),
        "basis_ordering": np.arange(dim, dtype=np.int64),
        "n_orb": np.asarray([dim, 0], dtype=np.int64),
        "exactified_matrices": {"identity": np.eye(dim, dtype=np.complex128)},
        "k_pullbacks": {"identity": np.eye(2, dtype=np.float64)},
        "q_permutations": {"identity": np.arange(1, dtype=np.int64)},
        "sector_permutations": {"identity": np.asarray([0, 1], dtype=np.int64)},
        "dtype": "complex128",
        "compiler_version": "test-v1",
    }


def _toy_coordinate(max_degree: int = 0) -> PolynomialCoordinateBasis:
    return PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=[0.0, 0.0],
        reciprocal_basis=[[2.0, 0.0], [0.0, 4.0]],
        max_degree=max_degree,
    )


def test_coordinate_basis_is_dimensionless_global_and_records_ordering() -> None:
    coordinate = _toy_coordinate(max_degree=2)

    assert coordinate.coordinate_convention == COORDINATE_CONVENTION_V1
    assert coordinate.scale == pytest.approx(np.sqrt(10.0))
    assert coordinate.monomials == (
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
    )
    np.testing.assert_allclose(
        coordinate.to_dimensionless([np.sqrt(10.0), 0.0]),
        [1.0, 0.0],
    )
    metadata = coordinate.metadata()
    assert metadata["origin"] == [0.0, 0.0]
    assert metadata["monomial_ordering"] == "total_degree_then_r_then_s"
    assert metadata["cartesian_axes"] == [[1.0, 0.0], [0.0, 1.0]]


def test_polynomial_pullback_prunes_roundoff_only_complex_mixing() -> None:
    coordinate = _toy_coordinate(max_degree=10)
    element = response_basis_module.FiniteGroupElement(
        canonical_word=("C3z",),
        antiunitary=False,
        canonical_k_map=((-1, -1), (1, 0)),
        q_permutation=(0,),
        sector_permutation=(0,),
        k_pullback=(
            (-0.4999999999999999, 0.8660254037844387),
            (-0.8660254037844388, -0.4999999999999999),
        ),
        internal_u=np.eye(1, dtype=np.complex128),
    )
    seed = {
        (10, 0): sparse.csr_matrix(
            np.asarray([[2.0 - 0.5j]], dtype=np.complex128)
        )
    }

    transformed = response_basis_module._substitute_polynomial_coefficients(
        seed,
        element,
        coordinate,
    )

    assert set(transformed) == {(10, 0)}
    a = -0.4999999999999999 - 0.8660254037844388j
    np.testing.assert_allclose(
        transformed[(10, 0)].toarray(),
        seed[(10, 0)].toarray() * a**10,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_polynomial_pullback_keeps_resolved_complex_mixing() -> None:
    coordinate = _toy_coordinate(max_degree=2)
    element = response_basis_module.FiniteGroupElement(
        canonical_word=("shear",),
        antiunitary=False,
        canonical_k_map=((1, 0), (0, 1)),
        q_permutation=(0,),
        sector_permutation=(0,),
        k_pullback=((1.0, 0.25), (0.0, 1.0)),
        internal_u=np.eye(1, dtype=np.complex128),
    )
    seed = {(2, 0): sparse.csr_matrix(np.asarray([[1.0]], dtype=np.complex128))}

    transformed = response_basis_module._substitute_polynomial_coefficients(
        seed,
        element,
        coordinate,
    )

    assert set(transformed) == {(2, 0), (1, 1), (0, 2)}


def test_candidate_compiler_reuses_polynomial_pullbacks_across_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = _toy_coordinate(max_degree=4)
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(0,),
                sector_permutation=(0,),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=np.eye(1, dtype=np.complex128),
            )
        ]
    )
    seeds = [
        RawPolynomialSeed(
            seed_id=f"seed-{index}",
            coefficients={
                (4, 0): sparse.csr_matrix(
                    np.asarray([[index + 1.0]], dtype=np.complex128)
                )
            },
            support_component="all",
            metadata={},
        )
        for index in range(3)
    ]
    calls = 0
    original = response_basis_module._complex_pullback_polynomials

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        response_basis_module,
        "_complex_pullback_polynomials",
        counted,
    )

    compile_candidate_responses(seeds, coordinate=coordinate, group=group)

    assert calls == len(group.elements)


def test_support_closure_uses_monomial_structural_orbit_without_seed_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = _toy_coordinate(max_degree=0)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(1, 0),
                sector_permutation=(0,),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=swap,
            )
        ]
    )
    seed = RawPolynomialSeed(
        seed_id="support-seed",
        coefficients={
            (0, 0): sparse.csr_matrix(([1.0], ([0], [0])), shape=(2, 2))
        },
        support_component="onsite",
        metadata={"term_space_policy": "complete"},
    )
    internal_actions = {
        tuple(element.canonical_word): sparse.csr_matrix(element.internal_u)
        for element in group.elements
    }
    override_calls = 0
    original = response_basis_module._apply_group_element

    def counted(*args: object, **kwargs: object):
        nonlocal override_calls
        if kwargs.get("internal_u_override") is not None:
            override_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(response_basis_module, "_apply_group_element", counted)

    masks, _artifact = response_basis_module._support_masks_from_seeds(
        [seed],
        group=group,
        coordinate=coordinate,
        internal_actions_by_word=internal_actions,
    )

    assert override_calls == 0
    np.testing.assert_array_equal(masks["onsite"], np.eye(2, dtype=bool))


def test_support_closure_uses_certified_boolean_orbit_for_dense_internal_mixing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = _toy_coordinate(max_degree=0)
    mixing = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="mix",
                antiunitary=False,
                canonical_k_map=((1, 0), (0, 1)),
                q_permutation=(1, 0),
                sector_permutation=(0,),
                k_forward=((1.0, 0.0), (0.0, 1.0)),
                internal_u=mixing,
            )
        ]
    )
    seed = RawPolynomialSeed(
        seed_id="dense-mixing-support-seed",
        coefficients={
            (0, 0): sparse.csr_matrix(([1.0], ([0], [0])), shape=(2, 2))
        },
        support_component="onsite",
        metadata={"term_space_policy": "complete"},
    )
    internal_actions = {
        tuple(element.canonical_word): sparse.csr_matrix(element.internal_u)
        for element in group.elements
    }

    def fail_per_seed_transform(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("certified dense support transformed every seed")

    monkeypatch.setattr(
        response_basis_module,
        "_apply_group_element",
        fail_per_seed_transform,
    )
    masks, policy = response_basis_module._support_masks_from_seeds(
        [seed],
        group=group,
        coordinate=coordinate,
        internal_actions_by_word=internal_actions,
    )

    np.testing.assert_array_equal(masks["onsite"], np.ones((2, 2), dtype=bool))
    assert policy["onsite"]["structural_support_closure_certified"] is True
    assert policy["onsite"]["support_mask_compiler"] == (
        "certified_factorized_boolean_group_adjoint_closure_v1"
    )


def test_rank_reduction_solves_component_dependencies_in_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = _toy_coordinate(max_degree=0)
    channels = [
        response_basis_module.ResponseChannel(
            channel_id=f"channel-{index}",
            seed_id=f"seed-{index}",
            component="real",
            coefficients={
                (0, 0): sparse.csr_matrix(
                    np.asarray([[index + 1.0]], dtype=np.complex128)
                )
            },
            unnormalized_norm=float(index + 1),
            propagated_error_bound=1.0e-12,
            classification="confirmed_nonzero",
            response_scale=float(index + 1),
            support_component="same-row",
            metadata={},
        )
        for index in range(12)
    ]
    calls = 0
    original = response_basis_module.scipy.linalg.lstsq

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(response_basis_module.scipy.linalg, "lstsq", counted)

    retained, _proofs, dropped = response_basis_module._reduce_confirmed_channels(
        channels,
        coordinate=coordinate,
        dim=1,
        confirm_factor=64.0,
    )

    assert len(retained) == 1
    assert len(dropped) == 11
    assert calls == 1


def test_monomial_internal_similarity_fast_path_matches_sparse_product() -> None:
    coordinate = _toy_coordinate(max_degree=1)
    internal = np.asarray(
        [[0.0, np.exp(0.3j)], [np.exp(-0.2j), 0.0]],
        dtype=np.complex128,
    )
    element = response_basis_module.FiniteGroupElement(
        canonical_word=("C2",),
        antiunitary=True,
        canonical_k_map=((-1, 0), (0, -1)),
        q_permutation=(1, 0),
        sector_permutation=(0,),
        k_pullback=((-1.0, 0.0), (0.0, -1.0)),
        internal_u=internal,
    )
    coefficients = {
        (1, 0): sparse.csr_matrix(
            np.asarray(
                [[1.0 + 0.5j, 2.0 - 0.25j], [0.0, -0.75j]],
                dtype=np.complex128,
            )
        )
    }
    canonical_internal = sparse.csr_matrix(internal)
    monomial_action = response_basis_module._monomial_similarity_action(
        canonical_internal
    )
    assert monomial_action is not None

    reference = response_basis_module._apply_group_element(
        coefficients,
        element,
        coordinate,
        internal_u_override=canonical_internal,
    )
    fast = response_basis_module._apply_group_element(
        coefficients,
        element,
        coordinate,
        internal_u_override=canonical_internal,
        internal_monomial_action=monomial_action,
    )

    assert set(fast) == set(reference)
    for monomial in reference:
        np.testing.assert_allclose(
            fast[monomial].toarray(),
            reference[monomial].toarray(),
            rtol=0.0,
            atol=1.0e-14,
        )


def test_symmetry_group_key_serializes_each_shared_operation_object_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = [{"name": "C3z", "antiunitary": False}]
    equal_copy = copy.deepcopy(shared)
    calls = 0
    original = response_basis_module._canonical_json

    def counted(value: object) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(response_basis_module, "_canonical_json", counted)
    cache: dict[int, tuple[object, str]] = {}

    first = response_basis_module._cached_symmetry_group_key(shared, cache)
    second = response_basis_module._cached_symmetry_group_key(shared, cache)
    copied = response_basis_module._cached_symmetry_group_key(equal_copy, cache)

    assert first == second == copied
    assert calls == 2


def test_identity_e12_produces_two_hermitian_real_channels() -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    seed = RawPolynomialSeed(
        seed_id="E12",
        coefficients={(0, 0): e12},
        support_component="toy",
        metadata={"term_space_policy": "complete"},
    )

    candidates = compile_candidate_responses(
        [seed],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )

    assert isinstance(candidates, CandidateResponseSet)
    assert [channel.channel_id for channel in candidates.channels] == ["E12:real", "E12:imag"]
    assert [channel.classification for channel in candidates.channels] == [
        "confirmed_nonzero",
        "confirmed_nonzero",
    ]
    real = candidates.channels[0].coefficient(0, 0).toarray()
    imag = candidates.channels[1].coefficient(0, 0).toarray()
    np.testing.assert_allclose(real, np.array([[0.0, 0.5], [0.5, 0.0]]))
    np.testing.assert_allclose(imag, np.array([[0.0, 0.5j], [-0.5j, 0.0]]))
    assert np.linalg.matrix_rank(
        np.column_stack(
            [
                np.concatenate([real.real.ravel(), real.imag.ravel()]),
                np.concatenate([imag.real.ravel(), imag.imag.ravel()]),
            ]
        )
    ) == 2


def test_real_and_imag_channels_share_each_raw_group_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = RawPolynomialSeed(
        "E12-shared-transform",
        {
            (0, 0): sparse.csr_matrix(
                np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
            )
        },
        "toy",
        {"term_space_policy": "complete"},
    )
    group = identity_finite_group(2)
    calls = 0
    original = response_basis_module._apply_group_element

    def counted_apply(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(response_basis_module, "_apply_group_element", counted_apply)
    candidates = compile_candidate_responses(
        [seed],
        coordinate=_toy_coordinate(),
        group=group,
    )

    assert len(candidates.channels) == 2
    assert calls == 0


def test_exact_identity_action_bypasses_polynomial_and_sparse_unitary_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = _toy_coordinate(max_degree=2)
    coefficients = {
        (2, 0): sparse.csr_matrix(
            np.asarray([[0.0, 1.25 - 0.5j], [0.0, 0.0]], dtype=np.complex128)
        ),
        (0, 1): sparse.csr_matrix(
            np.asarray([[0.0, -0.2j], [0.0, 0.0]], dtype=np.complex128)
        ),
    }

    def fail_substitution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact identity entered polynomial substitution")

    monkeypatch.setattr(
        response_basis_module,
        "_substitute_polynomial_coefficients",
        fail_substitution,
    )
    transformed = response_basis_module._apply_group_element(
        coefficients,
        identity_finite_group(2).elements[0],
        coordinate,
    )

    assert set(transformed) == set(coefficients)
    for monomial in coefficients:
        np.testing.assert_array_equal(
            transformed[monomial].toarray(),
            coefficients[monomial].toarray(),
        )


def test_certified_joint_adjoint_compile_keeps_logical_channels_but_compiles_one_seed() -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    e21 = e12.getH().tocsr()
    center = CanonicalDimensionlessCenter.from_dimensionless((0.0, 0.0))
    forward = JointAdjointSeedKey(
        sector_from="layer-1",
        sector_to="layer-1",
        orbital_from=0,
        orbital_to=1,
        monomial=(0, 0),
        center=center,
        harmonic_id="zero",
        adjoint_harmonic_id="zero",
    )
    seeds = [
        RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {}),
        RawPolynomialSeed("E21", {(0, 0): e21}, "toy", {}),
    ]
    dense = compile_candidate_responses(
        seeds,
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )

    certified = compile_candidate_responses_adjoint_canonicalized(
        seeds,
        joint_keys={"E12": forward, "E21": forward.adjoint()},
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )

    assert [channel.channel_id for channel in certified.channels] == [
        "E12:real",
        "E12:imag",
        "E21:real",
        "E21:imag",
    ]
    dense_by_id = {channel.channel_id: channel for channel in dense.channels}
    for channel in certified.channels:
        np.testing.assert_allclose(
            channel.coefficient(0, 0).toarray(),
            dense_by_id[channel.channel_id].coefficient(0, 0).toarray(),
            atol=1.0e-15,
        )
    artifact = certified.artifact()
    assert artifact["candidate_channel_count"] == 4
    assert artifact["logical_candidate_channel_count"] == 4
    assert artifact["physically_compiled_representative_channel_count"] == 2
    assert artifact["adjoint_certified_dropped_channel_count"] == 2

    basis = CompiledResponseBasis.from_candidates(
        certified,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    assert basis.channel_ids == ("E12:real", "E12:imag")
    adjoint_drops = [
        record
        for record in basis.dropped_channels
        if record["reason"] == "adjoint_certified_linear_dependency"
    ]
    assert [record["channel_id"] for record in adjoint_drops] == [
        "E21:real",
        "E21:imag",
    ]


def test_finite_harmonic_fallback_matches_dense_reynolds_without_an_adjoint_partner() -> None:
    """Finite-p seeds remain complete when the authored vocabulary has no -p seed."""

    coordinate = _toy_coordinate(max_degree=1)
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    e21 = e12.getH().tocsr()
    finite = {
        (0, 0): sparse.csr_matrix(0.25 * e12),
        (1, 0): sparse.csr_matrix(1.5 * e12),
    }

    def metadata(*, row: int, col: int, p: tuple[float, float]) -> dict[str, object]:
        return {
            "term_index": row * 2 + col,
            "term_key": {
                "Mz": 0,
                "Mz_star": 0,
                "layer_from": 1,
                "layer_to": 1,
                "orbital_from": row + 1,
                "orbital_to": col + 1,
                "p": list(p),
            },
        }

    seeds = [
        RawPolynomialSeed(
            "zero-E12",
            {(0, 0): e12},
            "zero-support",
            metadata(row=0, col=1, p=(0.0, 0.0)),
        ),
        RawPolynomialSeed(
            "zero-E21",
            {(0, 0): e21},
            "zero-support",
            metadata(row=1, col=0, p=(0.0, 0.0)),
        ),
        RawPolynomialSeed(
            "finite-p-E12",
            finite,
            "finite-support",
            metadata(row=0, col=1, p=(0.25, -0.125)),
        ),
    ]
    group = identity_finite_group(2)
    dense = compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=group,
    )

    hybrid = response_basis_module._compile_candidate_group_with_adjoint_fallback(
        seeds,
        coordinate=coordinate,
        group=group,
        support_masks=None,
    )

    assert [channel.channel_id for channel in hybrid.channels] == [
        channel.channel_id for channel in dense.channels
    ]
    dense_by_id = {channel.channel_id: channel for channel in dense.channels}
    for channel in hybrid.channels:
        expected = dense_by_id[channel.channel_id]
        for monomial in coordinate.monomials:
            np.testing.assert_allclose(
                channel.coefficient(*monomial).toarray(),
                expected.coefficient(*monomial).toarray(),
                atol=2.0e-15,
            )

    artifact = hybrid.artifact()
    assert artifact["logical_candidate_channel_count"] == 6
    assert artifact["physically_compiled_representative_channel_count"] == 4
    assert artifact["adjoint_certified_dropped_channel_count"] == 2
    assert artifact["adjoint"]["finite_p_fallback_seed_count"] == 1
    assert artifact["adjoint"]["finite_p_fallback_channel_count"] == 2


def test_unpaired_zero_harmonic_seed_falls_back_to_complete_dense_channels() -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    seed = RawPolynomialSeed(
        "unpaired-zero-E12",
        {(0, 0): e12},
        "explicit-zero-support",
        {
            "term_space_policy": "explicit_reduced",
            "term_key": {
                "Mz": 0,
                "Mz_star": 0,
                "layer_from": 1,
                "layer_to": 1,
                "orbital_from": 1,
                "orbital_to": 2,
                "p": [0.0, 0.0],
            },
        },
    )
    coordinate = _toy_coordinate(max_degree=0)
    group = identity_finite_group(2)
    dense = compile_candidate_responses([seed], coordinate=coordinate, group=group)

    compiled = response_basis_module._compile_candidate_group_with_adjoint_fallback(
        [seed],
        coordinate=coordinate,
        group=group,
        support_masks=None,
    )

    assert [channel.channel_id for channel in compiled.channels] == [
        "unpaired-zero-E12:real",
        "unpaired-zero-E12:imag",
    ]
    for actual, expected in zip(compiled.channels, dense.channels):
        np.testing.assert_allclose(
            actual.coefficient(0, 0).toarray(),
            expected.coefficient(0, 0).toarray(),
            atol=2.0e-15,
        )
    artifact = compiled.artifact()["adjoint"]
    assert artifact["p0_dense_fallback_seed_count"] == 1
    assert artifact["p0_dense_fallback_channel_count"] == 2


def test_complete_support_policy_closes_authored_harmonic_representative_under_group() -> None:
    cycle = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C3z",
                antiunitary=False,
                canonical_k_map=((1, 0), (0, 1)),
                q_permutation=(1, 2, 0),
                sector_permutation=(0,),
                k_forward=((1.0, 0.0), (0.0, 1.0)),
                internal_u=cycle,
            )
        ]
    )
    e12 = sparse.csr_matrix(
        np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.complex128,
        )
    )
    complete = RawPolynomialSeed(
        "harmonic-representative",
        {(0, 0): e12},
        "complete-harmonic",
        {"term_space_policy": "complete"},
    )

    masks, policy = response_basis_module._support_masks_from_seeds(
        [complete],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
    )

    certificate = response_basis_module.certify_support_closure(
        [complete],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
        support_masks=masks,
    )["complete-harmonic"]
    assert certificate["certified"] is True
    assert policy["complete-harmonic"]["support_mask_policy"] == (
        "complete_symmetry_adjoint_closure_v1"
    )
    assert policy["complete-harmonic"]["symmetry_adjoint_added_entry_count"] > 0

    reduced = RawPolynomialSeed(
        "reduced-harmonic",
        {(0, 0): e12},
        "reduced-harmonic",
        {"term_space_policy": "explicit_reduced"},
    )
    reduced_masks, reduced_policy = response_basis_module._support_masks_from_seeds(
        [reduced],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
    )
    assert reduced_policy["reduced-harmonic"]["support_mask_policy"] == (
        "explicit_reduced_authored_v1"
    )
    with pytest.raises(response_basis_module.SupportClosureError, match=r"not (?:adjoint )?closed"):
        response_basis_module.certify_support_closure(
            [reduced],
            group=group,
            coordinate=_toy_coordinate(max_degree=0),
            support_masks=reduced_masks,
        )


def test_explicit_reduced_support_does_not_silently_add_adjoint_partner() -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    seed = RawPolynomialSeed(
        "explicit-e12",
        {(0, 0): e12},
        "explicit-e12",
        {"term_space_policy": "explicit_reduced"},
    )

    masks, policy = response_basis_module._support_masks_from_seeds(
        [seed],
        group=identity_finite_group(2),
        coordinate=_toy_coordinate(max_degree=0),
    )

    assert policy["explicit-e12"]["authored_support_entry_count"] == 1
    assert not bool(masks["explicit-e12"][1, 0])
    with pytest.raises(SupportClosureError, match="not adjoint closed"):
        certify_support_closure(
            [seed],
            group=identity_finite_group(2),
            coordinate=_toy_coordinate(max_degree=0),
            support_masks=masks,
        )


def test_orbit_representative_support_is_closed_by_group_and_adjoint() -> None:
    exchange = np.array(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2T",
                antiunitary=True,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(0,),
                sector_permutation=(1, 0),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=exchange,
            )
        ]
    )
    seed = RawPolynomialSeed(
        "layer-one-representative",
        {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]))},
        "layer-orbit",
        {"term_space_policy": "orbit_representative"},
    )

    masks, policy = response_basis_module._support_masks_from_seeds(
        [seed],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
    )

    np.testing.assert_array_equal(masks["layer-orbit"], np.eye(2, dtype=bool))
    certificate = certify_support_closure(
        [seed],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
        support_masks=masks,
    )["layer-orbit"]
    assert certificate["certified"] is True
    assert policy["layer-orbit"]["support_mask_policy"] == (
        "orbit_representative_symmetry_adjoint_closure_v1"
    )


def test_complete_support_keeps_numerical_group_tail_outside_structural_mask() -> None:
    eps = 1.0e-15
    cosine = np.sqrt(1.0 - (2.0 * eps) ** 2)
    reflection = np.array(
        [[cosine, 2.0 * eps], [2.0 * eps, -cosine]],
        dtype=np.complex128,
    )
    identity = identity_finite_group(2).elements[0]
    reflected = response_basis_module.FiniteGroupElement(
        canonical_word=("tail-reflection",),
        antiunitary=False,
        canonical_k_map=((-1, 0), (0, -1)),
        q_permutation=(0,),
        sector_permutation=(0, 1),
        k_pullback=((-1.0, 0.0), (0.0, -1.0)),
        internal_u=reflection,
    )
    group = response_basis_module.FiniteGroup(elements=(identity, reflected))
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    seed = RawPolynomialSeed(
        "complete-e12",
        {(0, 0): e12},
        "complete-e12",
        {"term_space_policy": "complete"},
    )

    masks, policy = response_basis_module._support_masks_from_seeds(
        [seed],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
    )
    mask = masks["complete-e12"]
    certificate = certify_support_closure(
        [seed],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
        support_masks=masks,
    )["complete-e12"]

    assert policy["complete-e12"]["authored_support_entry_count"] == 1
    assert policy["complete-e12"]["symmetry_adjoint_added_entry_count"] == 1
    np.testing.assert_array_equal(mask, np.array([[False, True], [True, False]]))
    assert 0.0 < certificate["leakage_before_cleanup"]
    assert certificate["leakage_before_cleanup"] <= certificate["propagated_error_bound"]
    assert certificate["cleaned_component_count"] > 0


def test_support_closure_reuses_group_image_for_adjoint_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = RawPolynomialSeed(
        "complete-e12-shared-adjoint",
        {
            (0, 0): sparse.csr_matrix(
                np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
            )
        },
        "complete-e12-shared-adjoint",
        {"term_space_policy": "complete"},
    )
    group = identity_finite_group(2)
    calls = 0
    original = response_basis_module._apply_group_element

    def counted_apply(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(response_basis_module, "_apply_group_element", counted_apply)
    masks, _ = response_basis_module._support_masks_from_seeds(
        [seed],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
    )
    assert calls == len(group.elements)
    certify_support_closure(
        [seed],
        group=group,
        coordinate=_toy_coordinate(max_degree=0),
        support_masks=masks,
    )
    # Once the complete authored mask has been closed, the exact Boolean
    # group+adjoint support certificate replaces another per-seed transform.
    assert calls == len(group.elements)


def test_generator_fixed_group_matches_dense_reynolds_response_span() -> None:
    coordinate = _toy_coordinate()
    matrices = []
    seeds = []
    center = CanonicalDimensionlessCenter.from_dimensionless((0.0, 0.0))
    joint_keys = {}
    for row in range(2):
        for col in range(2):
            matrix = np.zeros((2, 2), dtype=np.complex128)
            matrix[row, col] = 1.0
            seed_id = f"E{row + 1}{col + 1}"
            matrices.append(matrix)
            seeds.append(
                RawPolynomialSeed(
                    seed_id,
                    {(0, 0): sparse.csr_matrix(matrix)},
                    "complete-two-orbital",
                    {"term_index": 2 * row + col},
                )
            )
            joint_keys[seed_id] = JointAdjointSeedKey(
                sector_from="layer-1",
                sector_to="layer-1",
                orbital_from=row,
                orbital_to=col,
                monomial=(0, 0),
                center=center,
                harmonic_id="zero",
                adjoint_harmonic_id="zero",
            )
    generator = FiniteGroupGenerator(
        name="C2",
        antiunitary=False,
        canonical_k_map=((-1, 0), (0, -1)),
        q_permutation=(0,),
        sector_permutation=(0, 1),
        k_forward=((-1.0, 0.0), (0.0, -1.0)),
        internal_u=np.diag([1.0, -1.0]).astype(np.complex128),
    )
    group = build_finite_group([generator])
    dense = compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=group,
    )

    fixed = compile_generator_fixed_response_group(
        seeds,
        joint_keys=joint_keys,
        coordinate=coordinate,
        group=group,
        reduce=True,
        materialize_ambient_vectors=False,
    )

    dense_vectors = sparse.hstack(
        [
            response_basis_module._channel_sparse_vector(channel, coordinate, 2)
            for channel in dense.channels
        ],
        format="csc",
    ).toarray()
    fixed_vectors = sparse.hstack(
        [
            response_basis_module._channel_sparse_vector(channel, coordinate, 2)
            for channel in fixed.retained_channels
        ],
        format="csc",
    ).toarray()
    dense_projector = dense_vectors @ np.linalg.pinv(dense_vectors)
    fixed_projector = fixed_vectors @ np.linalg.pinv(fixed_vectors)

    assert len(fixed.candidates.channels) == 8
    assert len(fixed.retained_channels) == 2
    assert (
        fixed.candidates.adjoint_artifact[
            "physically_materialized_projected_channel_count"
        ]
        == 2
    )
    np.testing.assert_allclose(fixed_projector, dense_projector, atol=2.0e-12)
    dense_by_id = {channel.channel_id: channel for channel in dense.channels}
    for channel in fixed.retained_channels:
        dense_channel = dense_by_id[channel.channel_id]
        np.testing.assert_allclose(
            channel.coefficient(0, 0).toarray(),
            dense_channel.coefficient(0, 0).toarray(),
            atol=2.0e-12,
        )
        transformed = response_basis_module._apply_group_element(
            channel.coefficients,
            group.elements[1],
            coordinate,
        )
        np.testing.assert_allclose(
            transformed[(0, 0)].toarray(),
            channel.coefficient(0, 0).toarray(),
            atol=2.0e-12,
        )
    coefficients = np.asarray([0.37, -1.21])
    fixed_h = sum(
        coefficient * channel.coefficient(0, 0).toarray()
        for coefficient, channel in zip(coefficients, fixed.retained_channels)
    )
    dense_h = sum(
        coefficient * dense_by_id[channel.channel_id].coefficient(0, 0).toarray()
        for coefficient, channel in zip(coefficients, fixed.retained_channels)
    )
    np.testing.assert_allclose(fixed_h, dense_h, atol=2.0e-12)


def test_generator_images_transform_once_per_adjoint_orbit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = _toy_coordinate(max_degree=0)
    center = CanonicalDimensionlessCenter.from_dimensionless((0.0, 0.0))
    seeds: list[RawPolynomialSeed] = []
    joint_keys: dict[str, JointAdjointSeedKey] = {}
    for row in range(2):
        for col in range(2):
            seed_id = f"E{row + 1}{col + 1}"
            matrix = sparse.csr_matrix(
                (
                    np.asarray([1.0 + 0.0j]),
                    (np.asarray([row]), np.asarray([col])),
                ),
                shape=(2, 2),
            )
            seeds.append(RawPolynomialSeed(seed_id, {(0, 0): matrix}, "toy", {}))
            joint_keys[seed_id] = JointAdjointSeedKey(
                sector_from="layer-1",
                sector_to="layer-1",
                orbital_from=row,
                orbital_to=col,
                monomial=(0, 0),
                center=center,
                harmonic_id="zero",
                adjoint_harmonic_id="zero",
            )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(0,),
                sector_permutation=(0, 1),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=np.diag([1.0, -1.0]).astype(np.complex128),
            )
        ]
    )
    calls = 0
    original = response_basis_module._apply_group_element

    def counted_apply(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(response_basis_module, "_apply_group_element", counted_apply)

    def fail_repeated_raw_vocabulary_gram(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "logical fixed projection recomputed raw.T @ vocabulary per channel"
        )

    monkeypatch.setattr(
        response_basis_module,
        "_fixed_coordinate_projection_roundoff_bound",
        fail_repeated_raw_vocabulary_gram,
    )
    progress_messages: list[str] = []
    fixed = compile_generator_fixed_response_group(
        seeds,
        joint_keys=joint_keys,
        coordinate=coordinate,
        group=group,
        reduce=True,
        materialize_ambient_vectors=False,
        progress_callback=lambda message, **_kwargs: progress_messages.append(
            str(message)
        ),
    )

    # The identity raw vocabulary is compiled directly, without calling the
    # general polynomial/group transformation.  Three C2 transforms build the
    # generator images and two retained channels are certified at the end.
    assert calls == 5
    assert fixed.candidates.adjoint_artifact["adjoint_orbit_descriptor_count"] == 3
    assert fixed.candidates.adjoint_artifact["generator_image_raw_transform_count"] == 3
    assert (
        fixed.candidates.adjoint_artifact["raw_vector_materialization_count"]
        == fixed.candidates.adjoint_artifact["hermitian_ambient_channel_count"]
    )
    assert (
        fixed.candidates.adjoint_artifact["logical_projection_compiler"]
        == "certified_physical_representative_coordinates_v1"
    )
    assert fixed.reduction_proofs[0]["solver"] == "generator_fixed_subspace__small_coordinate_rrqr"
    assert fixed.reduction_proofs[0]["sample_grid_used"] is False
    assert any("raw adjoint vocabulary done" in message for message in progress_messages)
    assert any("generator images done" in message for message in progress_messages)
    assert any("fixed-space solve done" in message for message in progress_messages)
    assert any("retained channel materialization done" in message for message in progress_messages)


def test_sparse_projection_roundoff_does_not_scale_with_zero_ambient_rows() -> None:
    small_raw = sparse.csc_matrix(
        ([1.0], ([0], [0])),
        shape=(8, 1),
        dtype=np.float64,
    )
    large_raw = sparse.csc_matrix(
        ([1.0], ([0], [0])),
        shape=(80_000, 1),
        dtype=np.float64,
    )
    small_fixed = np.zeros((8, 1), dtype=np.float64)
    large_fixed = np.zeros((80_000, 1), dtype=np.float64)
    small_fixed[0, 0] = 1.0
    large_fixed[0, 0] = 1.0

    small_bound = response_basis_module._fixed_projection_roundoff_bound(
        small_raw,
        small_fixed,
        np.asarray([1.0]),
    )
    large_bound = response_basis_module._fixed_projection_roundoff_bound(
        large_raw,
        large_fixed,
        np.asarray([1.0]),
    )

    assert large_bound == pytest.approx(small_bound, rel=0.0, abs=0.0)


def test_unrelated_large_seed_cannot_relax_adjoint_certification_bound() -> None:
    dim = 3
    e12 = np.zeros((dim, dim), dtype=np.complex128)
    e12[0, 1] = 1.0
    mismatched_e21 = 1.0001 * e12.conj().T
    large_e33 = np.zeros((dim, dim), dtype=np.complex128)
    large_e33[2, 2] = 1.0e12
    center = CanonicalDimensionlessCenter.from_dimensionless((0.0, 0.0))
    forward = JointAdjointSeedKey(
        sector_from="layer-1",
        sector_to="layer-1",
        orbital_from=0,
        orbital_to=1,
        monomial=(0, 0),
        center=center,
        harmonic_id="zero",
        adjoint_harmonic_id="zero",
    )
    unrelated_self_adjoint = JointAdjointSeedKey(
        sector_from="layer-1",
        sector_to="layer-1",
        orbital_from=2,
        orbital_to=2,
        monomial=(0, 0),
        center=center,
        harmonic_id="zero",
        adjoint_harmonic_id="zero",
    )
    seeds = [
        RawPolynomialSeed("E12", {(0, 0): sparse.csr_matrix(e12)}, "toy", {}),
        RawPolynomialSeed(
            "E21-mismatch",
            {(0, 0): sparse.csr_matrix(mismatched_e21)},
            "toy",
            {},
        ),
        RawPolynomialSeed(
            "unrelated-large-E33",
            {(0, 0): sparse.csr_matrix(large_e33)},
            "toy",
            {},
        ),
    ]

    with pytest.raises(
        AdjointCertificationError,
        match="raw coefficient adjoint residual",
    ):
        compile_candidate_responses_adjoint_canonicalized(
            seeds,
            joint_keys={
                "E12": forward,
                "E21-mismatch": forward.adjoint(),
                "unrelated-large-E33": unrelated_self_adjoint,
            },
            coordinate=_toy_coordinate(),
            group=identity_finite_group(dim),
        )


def test_null_classification_uses_unnormalized_norm_and_propagated_error() -> None:
    policy = NullClassificationPolicy(numerical_factor=1.0, confirm_factor=100.0)

    assert policy.classify(0.0, 1.0e-12, structural=True) == "structural_zero"
    assert policy.classify(1.0e-12, 1.0e-12) == "numerical_zero"
    assert policy.classify(20.0e-12, 1.0e-12) == "ambiguous"
    assert policy.classify(100.0e-12, 1.0e-12) == "confirmed_nonzero"
    assert policy.artifact()["engineering_policy"] is True


def test_exact_numeric_cancellation_is_not_called_structural_without_algebraic_proof() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )

    assert candidates.channels[0].classification == "confirmed_nonzero"
    assert candidates.channels[1].component == "imag"
    assert candidates.channels[1].classification == "numerical_zero"


def test_uncompressed_basis_retains_each_nonstructural_candidate_exactly_once() -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )

    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )

    assert basis.channel_ids == ("E12:real", "E12:imag")
    assert basis.artifact()["candidate_channel_count"] == 2
    assert basis.artifact()["retained_basis_channel_count"] == 2
    records = basis.candidate_artifact["channels"]
    assert [record["channel_id"] for record in records] == ["E12:real", "E12:imag"]
    assert all(record["response_absolute_norm"] > 0.0 for record in records)
    assert all(record["algebra_error_absolute"] >= 0.0 for record in records)


def test_candidate_artifact_preserves_logical_count_when_only_independent_channels_materialize() -> None:
    raw = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "logical-count",
                {(0, 0): sparse.csr_matrix(np.asarray([[1.0]]))},
                "diag",
                {},
            )
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(1),
    )
    compact = dataclasses.replace(
        raw,
        channels=(raw.channels[0],),
        adjoint_artifact={
            "logical_candidate_channel_count": 2,
            "physically_compiled_representative_channel_count": 1,
            "physically_materialized_projected_channel_count": 1,
        },
    )

    artifact = compact.artifact()

    assert artifact["candidate_channel_count"] == 2
    assert artifact["logical_candidate_channel_count"] == 2
    assert artifact["physically_materialized_projected_channel_count"] == 1


def test_reducer_reports_absolute_outside_span_and_algebra_floor_for_removed_channels() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("first", {(0, 0): e11}, "diag", {}),
            RawPolynomialSeed("duplicate", {(0, 0): e11}, "diag", {}),
            RawPolynomialSeed("numerical", {(0, 0): e11}, "diag", {}),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    dropped = {record["channel_id"]: record for record in basis.dropped_channels}

    dependent = dropped["duplicate:real"]
    assert dependent["response_absolute_norm"] > 0.0
    assert dependent["outside_span_residual_absolute"] < 1.0e-14
    assert dependent["algebra_floor_absolute"] > 0.0
    numerical = dropped["numerical:imag"]
    assert numerical["reason"] == "certified_numerical_zero"
    assert numerical["outside_span_residual_absolute"] == 0.0
    assert numerical["algebra_floor_ratio"] <= 1.0


def test_reducer_never_drops_a_direction_outside_its_own_algebra_floor() -> None:
    anchor = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    independent = sparse.csr_matrix(np.diag([1.0, 0.1]).astype(complex))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("anchor", {(0, 0): anchor}, "diag", {}),
            RawPolynomialSeed("independent", {(0, 0): independent}, "diag", {}),
            RawPolynomialSeed("barely-confirmed", {(0, 0): anchor}, "diag", {}),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    poisoned_channels = []
    for channel in candidates.channels:
        if channel.channel_id == "barely-confirmed:real":
            channel = dataclasses.replace(
                channel,
                propagated_error_bound=channel.unnormalized_norm / 101.0,
                classification="confirmed_nonzero",
            )
        poisoned_channels.append(channel)
    candidates = dataclasses.replace(candidates, channels=tuple(poisoned_channels))

    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )

    dependency_drops = [
        record
        for record in basis.dropped_channels
        if record["reason"] == "target_independent_linear_dependency"
    ]
    assert all(record["algebra_floor_ratio"] <= 1.0 for record in dependency_drops)
    assert "independent:real" in basis.channel_ids


def test_basis_hash_is_target_independent_and_fit_does_not_mutate_basis() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(np.complex128))
    e22 = sparse.csr_matrix(np.diag([0.0, 1.0]).astype(np.complex128))
    seeds = [
        RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {}),
        RawPolynomialSeed("E22", {(0, 0): e22}, "diag", {}),
    ]
    candidates = compile_candidate_responses(
        seeds,
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    original_hash = basis.basis_hash
    original_channels = basis.channel_ids
    original_artifact = copy.deepcopy(basis.artifact())

    fit_a = basis.fit(
        np.asarray([[0.0, 0.0]]),
        np.asarray([np.diag([1.0, 2.0])], dtype=np.complex128),
        fit_indices=[3],
        regularization=0.0,
        band_window=[0, 2],
    )
    fit_b = basis.fit(
        np.asarray([[0.0, 0.0]]),
        np.asarray([np.diag([3.0, -4.0])], dtype=np.complex128),
        fit_indices=[17],
        regularization=2.5,
        band_window=[1, 2],
    )
    fit_c = basis.fit(
        np.asarray([[0.1, 0.0]]),
        np.asarray([np.diag([1.0, 2.0])], dtype=np.complex128),
        fit_indices=[3],
        regularization=0.0,
        band_window=[0, 2],
    )

    assert basis.basis_hash == original_hash
    assert basis.channel_ids == original_channels
    assert basis.artifact() == original_artifact
    assert fit_a.basis_hash == fit_b.basis_hash == original_hash
    assert fit_a.fit_hash != fit_b.fit_hash
    assert fit_a.fit_hash != fit_c.fit_hash
    assert fit_a.fit_pivots == ()
    assert fit_b.fit_pivots == ()
    assert fit_a.fit_solver_channel_ids == fit_b.fit_solver_channel_ids
    assert fit_a.fit_response_gram_rank == 2
    assert fit_a.fit_design_certified_rank == 2
    np.testing.assert_allclose(fit_a.coefficients, [1.0, 2.0], atol=1.0e-12)
    assert fit_a.nonzero_channel_ids == ("E11:real", "E22:real")


def test_compiled_basis_defensively_freezes_hash_bearing_nested_state() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(np.complex128))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    baseline = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    mutable_identity = {"nested": {"values": [1, 2]}}
    mutable_candidate_artifact = {"channels": [{"channel_id": "source"}]}
    mutable_proof = {"members": ["E11:real"]}
    provisional = dataclasses.replace(
        baseline,
        identity_record=mutable_identity,
        candidate_artifact=mutable_candidate_artifact,
        reduction_proofs=(mutable_proof,),
        basis_hash="",
    )
    basis = dataclasses.replace(provisional, basis_hash=provisional._compute_hash())
    original_hash = basis.basis_hash

    mutable_identity["nested"]["values"][0] = 99
    mutable_candidate_artifact["channels"][0]["channel_id"] = "mutated"
    mutable_proof["members"].append("mutated")

    assert basis.identity_record["nested"]["values"] == (1, 2)
    assert basis.candidate_artifact["channels"][0]["channel_id"] == "source"
    assert basis.reduction_proofs[0]["members"] == ("E11:real",)
    assert basis._compute_hash() == basis.basis_hash == original_hash

    channel = basis.channels[0]
    coefficient = channel.coefficients[(0, 0)]
    assert coefficient.data.flags.writeable is False
    assert coefficient.indices.flags.writeable is False
    assert coefficient.indptr.flags.writeable is False
    with pytest.raises(TypeError):
        channel.coefficients[(1, 0)] = coefficient
    with pytest.raises(ValueError):
        coefficient.data[0] = 7.0
    with pytest.raises(TypeError):
        basis.identity_record["new"] = "not allowed"
    with pytest.raises(TypeError):
        basis.candidate_artifact["channels"][0]["channel_id"] = "not allowed"
    assert basis._compute_hash() == basis.basis_hash == original_hash


def test_new_compiled_basis_computes_canonical_hash_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "hash-once",
                {(0, 0): sparse.csr_matrix(np.asarray([[1.0]]))},
                "diag",
                {},
            )
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(1),
    )
    calls = 0
    original = CompiledResponseBasis._compute_hash

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(CompiledResponseBasis, "_compute_hash", counted)
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(1),
        reduce=True,
    )

    assert basis.basis_hash
    assert calls == 1


def test_fit_retains_ambiguous_basis_channel_but_excludes_it_from_production() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(np.complex128))
    e22 = sparse.csr_matrix(np.diag([0.0, 1.0]).astype(np.complex128))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {}),
            RawPolynomialSeed("E22", {(0, 0): e22}, "diag", {}),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    channels = tuple(
        dataclasses.replace(channel, classification="ambiguous")
        if channel.channel_id == "E22:real"
        else channel
        for channel in candidates.channels
    )
    candidates = dataclasses.replace(candidates, channels=channels)
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )

    assert "E22:real" in basis.channel_ids
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([0.0, 2.0])], dtype=np.complex128),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )

    ambiguous_index = basis.channel_ids.index("E22:real")
    assert fitted.fit_channel_policy == "confirmed_nonzero_only_v2"
    assert "E22:real" not in fitted.fit_selected_channel_ids
    assert fitted.coefficients[ambiguous_index] == pytest.approx(0.0)
    assert "E22:real" not in fitted.nonzero_channel_ids
    runtime = CompiledResponseRuntime(basis, fitted)
    np.testing.assert_allclose(runtime.hamiltonian([0.0, 0.0]), np.zeros((2, 2)))


def test_runtime_rejects_nonzero_ambiguous_production_coefficient() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(np.complex128))
    e22 = sparse.csr_matrix(np.diag([0.0, 1.0]).astype(np.complex128))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {}),
            RawPolynomialSeed("E22", {(0, 0): e22}, "diag", {}),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(channel, classification="ambiguous")
            if channel.channel_id == "E22:real"
            else channel
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([1.0, 0.0])], dtype=np.complex128),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )
    coefficients = np.asarray(fitted.coefficients).copy()
    coefficients[basis.channel_ids.index("E22:real")] = 1.0
    invalid = dataclasses.replace(fitted, coefficients=coefficients)

    with pytest.raises(ValueError, match="confirmed_nonzero channels"):
        CompiledResponseRuntime(basis, invalid)


def test_fitted_response_model_owns_finite_readonly_coefficients() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(np.complex128))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([1.0, 0.0])], dtype=np.complex128),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )
    source = np.asarray(fitted.coefficients).copy()
    owned = dataclasses.replace(fitted, coefficients=source)
    expected = np.asarray(owned.coefficients).copy()

    source[:] = 123.0

    np.testing.assert_array_equal(owned.coefficients, expected)
    assert not owned.coefficients.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        owned.coefficients[0] = 1.0
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(fitted, coefficients=np.asarray([np.nan] * len(expected)))
    with pytest.raises(ValueError, match="singular values"):
        dataclasses.replace(fitted, fit_design_singular_values=(1.0, np.nan))
    with pytest.raises(ValueError, match="non-negative"):
        dataclasses.replace(fitted, fit_design_propagated_error_bound=-1.0)
    with pytest.raises(ValueError, match="certified rank"):
        dataclasses.replace(
            fitted,
            fit_design_certified_rank=fitted.fit_response_gram_rank + 1,
        )
    with pytest.raises(ValueError, match="fit solver policy"):
        dataclasses.replace(fitted, fit_solver_policy="untrusted-bypass")


def test_ambiguous_guard_cannot_be_bypassed_by_mutating_fitted_coefficients() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(np.complex128))
    e22 = sparse.csr_matrix(np.diag([0.0, 1.0]).astype(np.complex128))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {}),
            RawPolynomialSeed("E22", {(0, 0): e22}, "diag", {}),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(channel, classification="ambiguous")
            if channel.channel_id == "E22:real"
            else channel
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([1.0, 0.0])], dtype=np.complex128),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )
    source = np.asarray(fitted.coefficients).copy()
    guarded = dataclasses.replace(fitted, coefficients=source)
    ambiguous_index = basis.channel_ids.index("E22:real")

    source[ambiguous_index] = 1.0

    runtime = CompiledResponseRuntime(basis, guarded)
    assert runtime.fitted.coefficients[ambiguous_index] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        runtime.fitted.coefficients[ambiguous_index] = 1.0


def test_fit_rank_and_coefficients_use_dimensionless_response_normalized_variables() -> None:
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "unit_E11",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))},
                "diag",
                {},
            ),
            RawPolynomialSeed(
                "tiny_E22",
                {(0, 0): sparse.csr_matrix(np.diag([0.0, 1.0e-16]).astype(complex))},
                "diag",
                {},
            ),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    target = np.asarray([np.diag([0.3, 0.2])], dtype=complex)

    fitted = basis.fit(
        [[0.0, 0.0]],
        target,
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )

    assert fitted.fit_selected_channel_ids == ()
    assert len(fitted.fit_solver_channel_ids) == 2
    assert fitted.fit_response_gram_rank == 2
    np.testing.assert_allclose(
        basis.hamiltonians([[0.0, 0.0]], fitted.coefficients),
        target,
        atol=1.0e-14,
    )
    assert fitted.coefficients[1] == pytest.approx(2.0e15)


def test_basis_hash_uses_canonical_little_endian_array_serialization() -> None:
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "E11",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))},
                "diag",
                {},
            )
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    little_identity = _identity_payload(2)
    big_identity = _identity_payload(2)
    little_identity["q_vectors"] = np.asarray([[0.25, -0.5]], dtype="<f8")
    big_identity["q_vectors"] = np.asarray([[0.25, -0.5]], dtype=">f8")

    little = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=little_identity,
        reduce=True,
    )
    big = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=big_identity,
        reduce=True,
    )

    assert little.basis_hash == big.basis_hash


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("q_vectors", np.asarray([[0.5, -0.5]], dtype=float)),
        ("basis_ordering", np.asarray([1, 0], dtype=np.int64)),
        ("exactified_matrices", {"identity": np.diag([1.0, -1.0]).astype(complex)}),
        ("k_pullbacks", {"identity": np.asarray([[1.0, 1.0e-10], [0.0, 1.0]])}),
        ("q_permutations", {"identity": np.asarray([1, 0], dtype=np.int64)}),
        ("sector_permutations", {"identity": np.asarray([1, 0], dtype=np.int64)}),
        ("dtype", "complex64"),
    ],
)
def test_basis_hash_changes_with_each_basis_identity_field(
    field: str,
    replacement: object,
) -> None:
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "E11",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))},
                "diag",
                {},
            )
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    identity = _identity_payload(2)
    if field in {"basis_ordering", "q_permutations"}:
        identity["q_vectors"] = np.zeros((2, 2), dtype=float)
        identity["basis_ordering"] = np.asarray([0, 1], dtype=np.int64)
        identity["q_permutations"] = {"identity": np.asarray([0, 1], dtype=np.int64)}
    baseline = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=identity,
        reduce=True,
    )
    changed_identity = copy.deepcopy(identity)
    changed_identity[field] = replacement
    changed = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=changed_identity,
        reduce=True,
    )

    assert changed.basis_hash != baseline.basis_hash


def test_basis_hash_changes_with_coordinate_and_normalization_versions() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    baseline = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    scaled_candidates = compile_candidate_responses(
        [RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {})],
        coordinate=PolynomialCoordinateBasis.from_reciprocal_basis(
            origin=[0.0, 0.0],
            reciprocal_basis=[[3.0, 0.0], [0.0, 6.0]],
            max_degree=0,
        ),
        group=identity_finite_group(2),
    )
    scaled = CompiledResponseBasis.from_candidates(
        scaled_candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    changed_normalization = dataclasses.replace(
        baseline,
        response_normalization="different-response-normalization",
        basis_hash="",
    )
    changed_normalization = dataclasses.replace(
        changed_normalization,
        basis_hash=changed_normalization._compute_hash(),
    )

    assert scaled.basis_hash != baseline.basis_hash
    assert changed_normalization.basis_hash != baseline.basis_hash


def test_target_independent_reducer_uses_real_coefficient_space_not_gram_squared_rank() -> None:
    e11 = np.diag([1.0, 0.0]).astype(complex)
    almost_e11 = np.diag([1.0, 1.0e-9]).astype(complex)
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("E11", {(0, 0): sparse.csr_matrix(e11)}, "diag", {}),
            RawPolynomialSeed(
                "E11_plus_tiny_E22",
                {(0, 0): sparse.csr_matrix(almost_e11)},
                "diag",
                {},
            ),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )

    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )

    assert basis.channel_ids == ("E11:real", "E11_plus_tiny_E22:real")
    proof = basis.reduction_proofs[0]
    assert proof["solver"] == "real_coefficient_space_svd_rrqr"
    assert proof["real_rank"] == 2
    assert proof["sample_grid_used"] is False
    assert proof["certification_space"].startswith("global_two_dimensional_polynomial")


def test_rank_components_are_proved_from_joint_response_support_not_authored_labels() -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("first", {(0, 0): e11}, "author_label_a", {}),
            RawPolynomialSeed("duplicate", {(0, 0): e11}, "author_label_b", {}),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )

    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )

    assert len(basis.channels) == 1
    assert basis.reduction_proofs[0]["direct_sum_proof"] == (
        "disjoint_global_real_polynomial_coefficient_rows_after_group_and_adjoint_projection"
    )
    assert set(basis.reduction_proofs[0]["authored_support_components"]) == {
        "author_label_a",
        "author_label_b",
    }


def test_frozen_basis_round_trip_reproduces_responses_and_hamiltonian() -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    frozen = basis.freeze()
    restored = CompiledResponseBasis.from_frozen(frozen)

    assert restored.response_semantics == COMPLETE_LINEAR_V2
    assert restored.response_normalization == RESPONSE_NORMALIZATION_V1
    assert restored.basis_hash == basis.basis_hash
    np.testing.assert_allclose(
        restored.response_tensor([[0.0, 0.0], [0.2, -0.3]]),
        basis.response_tensor([[0.0, 0.0], [0.2, -0.3]]),
    )
    coefficients = np.asarray([2.0, -3.0])
    np.testing.assert_allclose(
        restored.hamiltonians([[0.0, 0.0]], coefficients),
        basis.hamiltonians([[0.0, 0.0]], coefficients),
    )


def test_frozen_basis_loader_groups_entries_once_and_hashes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    frozen = dict(basis.freeze())
    entry_names = (
        "entry_channel",
        "entry_monomial",
        "entry_row",
        "entry_col",
        "entry_value",
    )
    permutation = np.arange(np.asarray(frozen["entry_channel"]).size)[::-1]
    for name in entry_names:
        frozen[name] = np.asarray(frozen[name])[permutation]

    hash_calls = 0
    original_compute_hash = CompiledResponseBasis._compute_hash

    def counted_compute_hash(self: CompiledResponseBasis) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return original_compute_hash(self)

    monkeypatch.setattr(CompiledResponseBasis, "_compute_hash", counted_compute_hash)
    restored = CompiledResponseBasis.from_frozen(frozen)

    assert restored.basis_hash == basis.basis_hash
    assert hash_calls == 1
    np.testing.assert_allclose(
        restored.response_tensor([[0.0, 0.0]]),
        basis.response_tensor([[0.0, 0.0]]),
    )


@pytest.mark.parametrize("audit_field", ["candidate_artifact", "dropped_channels"])
def test_frozen_basis_hash_covers_target_independent_audit_state(
    audit_field: str,
) -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    frozen = dict(basis.freeze())
    metadata = json.loads(str(np.asarray(frozen["metadata_json"]).item()))
    if audit_field == "candidate_artifact":
        metadata["candidate_artifact"]["candidate_channel_count"] += 1
    else:
        metadata["dropped_channels"].append(
            {"channel_id": "tampered", "reason": "not-a-compiler-proof"}
        )
    frozen["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(ValueError, match="compiled response basis hash mismatch"):
        CompiledResponseBasis.from_frozen(frozen)


def test_hamiltonian_evaluation_does_not_materialize_channel_response_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )

    def fail_response_tensor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Hamiltonian evaluation materialized the per-channel response tensor")

    monkeypatch.setattr(CompiledResponseBasis, "response_tensor", fail_response_tensor)
    coefficients = np.asarray([2.0, -3.0])
    hamiltonians = basis.hamiltonians(
        [[0.0, 0.0], [0.2, -0.3]],
        coefficients,
    )

    expected = np.asarray(
        [
            [[0.0, 1.0 - 1.5j], [1.0 + 1.5j, 0.0]],
            [[0.0, 1.0 - 1.5j], [1.0 + 1.5j, 0.0]],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(hamiltonians, expected)


def test_finite_group_uses_discrete_action_key_and_accepts_projective_phase() -> None:
    generator = FiniteGroupGenerator(
        name="C3z",
        antiunitary=False,
        canonical_k_map=((0, -1), (1, -1)),
        q_permutation=(1, 2, 0),
        sector_permutation=(0, 1),
        k_forward=((0.0, -1.0), (1.0, -1.0)),
        internal_u=np.asarray([[np.exp(1j * np.pi / 3.0)]], dtype=np.complex128),
    )

    group = build_finite_group([generator], max_group_size=8, max_word_length=6)

    assert len(group.elements) == 3
    assert [element.canonical_word for element in group.elements] == [
        (),
        ("C3z",),
        ("C3z", "C3z"),
    ]
    identity = group.elements[0]
    assert identity.discrete_key == (
        False,
        ((1, 0), (0, 1)),
        (0, 1, 2),
        (0, 1),
    )
    assert group.algebra_residual < 1.0e-12
    identity_record = group.artifact()["elements"][0]
    assert ["C3z", "C3z", "C3z"] in identity_record["alternate_words"]
    assert identity_record["alternate_word_residual"] < 1.0e-12


def test_finite_group_fails_closed_when_same_action_key_has_inconsistent_u() -> None:
    inconsistent = FiniteGroupGenerator(
        name="bad",
        antiunitary=False,
        canonical_k_map=((1, 0), (0, 1)),
        q_permutation=(0,),
        sector_permutation=(0, 1),
        k_forward=((1.0, 0.0), (0.0, 1.0)),
        internal_u=np.diag([1.0, -1.0]).astype(np.complex128),
    )

    with pytest.raises(ValueError, match="numerically close"):
        build_finite_group([inconsistent], max_group_size=4, max_word_length=3)


@pytest.mark.parametrize(
    ("max_group_size", "max_word_length", "message"),
    [(3, 32, "max_group_size"), (256, 1, "max_word_length")],
)
def test_finite_group_fails_closed_at_configured_bfs_limits(
    max_group_size: int,
    max_word_length: int,
    message: str,
) -> None:
    shear = FiniteGroupGenerator(
        name="shear",
        antiunitary=False,
        canonical_k_map=((1, 1), (0, 1)),
        q_permutation=(0,),
        sector_permutation=(0, 1),
        k_forward=((1.0, 1.0), (0.0, 1.0)),
        internal_u=np.eye(1, dtype=complex),
    )

    with pytest.raises(ValueError, match=message):
        build_finite_group(
            [shear],
            max_group_size=max_group_size,
            max_word_length=max_word_length,
        )


def test_support_certification_checks_adjoint_before_cleanup() -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    seed = RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {})
    incomplete = np.array([[False, True], [False, False]], dtype=bool)
    closed = np.array([[False, True], [True, False]], dtype=bool)

    with pytest.raises(SupportClosureError, match="adjoint"):
        certify_support_closure(
            [seed],
            group=identity_finite_group(2),
            coordinate=_toy_coordinate(),
            support_masks={"toy": incomplete},
        )

    certificates = certify_support_closure(
        [seed],
        group=identity_finite_group(2),
        coordinate=_toy_coordinate(),
        support_masks={"toy": closed},
    )
    assert certificates["toy"]["certified"] is True
    assert certificates["toy"]["leakage_before_cleanup"] == 0.0


def test_structurally_closed_support_is_certified_once_without_per_seed_group_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    e21 = e12.getH().tocsr()
    seeds = [
        RawPolynomialSeed(
            "E12",
            {(0, 0): e12},
            "toy",
            {"term_space_policy": "complete"},
        ),
        RawPolynomialSeed(
            "E21",
            {(0, 0): e21},
            "toy",
            {"term_space_policy": "complete"},
        ),
    ]
    closed = np.array([[False, True], [True, False]], dtype=bool)

    def fail_per_seed_transform(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("structurally closed support transformed every seed")

    monkeypatch.setattr(response_basis_module, "_apply_group_element", fail_per_seed_transform)

    certificates = certify_support_closure(
        seeds,
        group=identity_finite_group(2),
        coordinate=_toy_coordinate(),
        support_masks={"toy": closed},
    )
    masks, policies = response_basis_module._support_masks_from_seeds(
        seeds,
        group=identity_finite_group(2),
        coordinate=_toy_coordinate(),
    )

    np.testing.assert_array_equal(masks["toy"], closed)
    assert certificates["toy"]["support_closure_compiler"] == (
        "exact_boolean_adjoint_group_support_v1"
    )
    assert policies["toy"]["structural_support_closure_certified"] is True


def test_support_cleanup_leakage_is_propagated_into_channel_error_bound() -> None:
    raw = sparse.csr_matrix(
        np.asarray([[1.0, 1.0e-15], [0.0, 0.0]], dtype=np.complex128)
    )
    support = np.zeros((2, 2), dtype=bool)
    support[0, 0] = True
    seed = RawPolynomialSeed("cleanup", {(0, 0): raw}, "toy", {})

    candidates = compile_candidate_responses(
        [seed],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
        support_masks={"toy": support},
    )

    certificate = candidates.support_artifact["toy"]
    channel = candidates.channels[0]
    assert certificate["cleanup_allowed"] is True
    assert certificate["leakage_before_cleanup"] > 0.0
    assert certificate["cleaned_component_count"] == 2
    assert certificate["cleaned_components"]
    assert channel.error_bound_components["support_cleanup"] == pytest.approx(
        certificate["leakage_before_cleanup"]
    )
    assert channel.propagated_error_bound == pytest.approx(
        sum(channel.error_bound_components.values())
    )


def test_finite_q_center_globalization_keeps_lower_orders() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=[0.0, 0.0],
        reciprocal_basis=[[2.0, 0.0], [0.0, 2.0]],
        max_degree=1,
    )
    key = ContinuumTermKey(
        Mz=1,
        Mz_star=0,
        layer_from=1,
        layer_to=1,
        orbital_from=1,
        orbital_to=2,
        p=(0.0, 0.0),
    )
    seed = raw_polynomial_seed_from_term_key(
        key,
        seed_id="finite-p",
        Q_set1=np.asarray([[1.0, 0.0]]),
        Q_set2=np.asarray([[0.0, 0.0]]),
        n_orb1=2,
        n_orb2=1,
        coordinate=coordinate,
        support_component="kinetic",
        metadata={},
    )

    constant = seed.coefficients[(0, 0)].toarray()
    linear = seed.coefficients[(1, 0)].toarray()
    assert constant[0, 1] == pytest.approx(-1.0)
    assert linear[0, 1] == pytest.approx(2.0)
    assert set(seed.coefficients) == {(0, 0), (1, 0)}
    physical_k = np.asarray([3.0, 0.0])
    dimensionless = coordinate.to_dimensionless(physical_k)
    w = complex(dimensionless[0], dimensionless[1])
    compiled_value = constant[0, 1] + linear[0, 1] * w
    assert compiled_value == pytest.approx(complex(*(physical_k - np.asarray([1.0, 0.0]))))


def test_true_finite_p_adjoint_mixes_lower_nominal_orders() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=[0.0, 0.0],
        reciprocal_basis=[[1.0, 0.0], [0.0, 1.0]],
        max_degree=1,
    )
    qset = np.asarray([[1.0, 0.0], [0.0, 0.0]])

    def seed_for(key: ContinuumTermKey, seed_id: str) -> RawPolynomialSeed:
        return raw_polynomial_seed_from_term_key(
            key,
            seed_id=seed_id,
            Q_set1=qset,
            Q_set2=np.asarray([[0.0, 0.0]]),
            n_orb1=2,
            n_orb2=1,
            coordinate=coordinate,
            support_component="finite-p-adjoint",
            metadata={},
        )

    forward = seed_for(
        ContinuumTermKey(1, 0, 1, 1, 1, 2, (1.0, 0.0)),
        "forward-linear",
    )
    reverse_linear = seed_for(
        ContinuumTermKey(0, 1, 1, 1, 2, 1, (-1.0, 0.0)),
        "reverse-linear",
    )
    reverse_constant = seed_for(
        ContinuumTermKey(0, 0, 1, 1, 2, 1, (-1.0, 0.0)),
        "reverse-constant",
    )
    adjoint = response_basis_module._adjoint_polynomial(forward.coefficients)

    def residual(
        left: dict[tuple[int, int], sparse.csr_matrix],
        right: dict[tuple[int, int], sparse.csr_matrix],
    ) -> float:
        total = 0.0
        for monomial in set(left) | set(right):
            delta = (
                left.get(monomial, sparse.csr_matrix((5, 5), dtype=np.complex128))
                - right.get(monomial, sparse.csr_matrix((5, 5), dtype=np.complex128))
            )
            total += float(np.vdot(delta.data, delta.data).real)
        return float(np.sqrt(total))

    reverse_only = dict(reverse_linear.coefficients)
    reverse_with_lower_order = {
        monomial: reverse_linear.coefficients.get(
            monomial, sparse.csr_matrix((5, 5), dtype=np.complex128)
        )
        - reverse_constant.coefficients.get(
            monomial, sparse.csr_matrix((5, 5), dtype=np.complex128)
        )
        for monomial in set(reverse_linear.coefficients) | set(reverse_constant.coefficients)
    }
    assert residual(adjoint, reverse_only) == pytest.approx(1.0)
    assert residual(adjoint, reverse_with_lower_order) == pytest.approx(0.0, abs=1.0e-15)


def test_dense_direct_oracle_matches_compiled_unitary_antiunitary_formula() -> None:
    coordinate = _toy_coordinate(max_degree=1)
    e12 = sparse.csr_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    )
    seed = RawPolynomialSeed(
        "linear-E12",
        {(1, 0): e12, (0, 0): 0.2 * e12},
        "toy",
        {},
    )
    time_reversal = FiniteGroupGenerator(
        name="TR",
        antiunitary=True,
        canonical_k_map=((-1, 0), (0, -1)),
        q_permutation=(0,),
        sector_permutation=(0, 1),
        k_forward=((-1.0, 0.0), (0.0, -1.0)),
        internal_u=np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128),
    )
    group = build_finite_group([time_reversal])
    candidates = compile_candidate_responses(
        [seed],
        coordinate=coordinate,
        group=group,
    )
    kpoints = np.asarray([[0.17, -0.31], [-0.4, 0.23], [0.0, 0.0]])
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    compiled = basis.response_tensor(kpoints)

    for channel_index, component in enumerate(("real", "imag")):
        oracle = dense_direct_response(
            seed,
            kpoints,
            coordinate=coordinate,
            group=group,
            component=component,
        )
        np.testing.assert_allclose(compiled[:, channel_index], oracle, atol=2.0e-13)


def test_dense_direct_oracle_does_not_call_sparse_compiler_transform_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = _toy_coordinate(max_degree=1)
    seed = RawPolynomialSeed(
        "linear-E12",
        {
            (1, 0): sparse.csr_matrix(
                np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
            )
        },
        "toy",
        {},
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dense direct oracle imported a sparse compiler helper")

    for helper in (
        "_apply_group_element",
        "_substitute_polynomial_coefficients",
        "_mask_polynomial_support",
        "_sum_polynomial_coefficients",
    ):
        monkeypatch.setattr(f"kp.model.response_basis.{helper}", fail)

    result = dense_direct_response(
        seed,
        [[0.2, -0.1]],
        coordinate=coordinate,
        group=identity_finite_group(2),
        component="real",
    )

    assert np.linalg.norm(result) > 0.0


def test_dense_direct_oracle_matches_joint_group_with_shifted_global_origin() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=[0.13, -0.07],
        reciprocal_basis=[[1.0, 0.0], [0.0, 1.0]],
        max_degree=2,
    )
    seed = RawPolynomialSeed(
        "quadratic-E12",
        {
            (2, 0): sparse.csr_matrix(
                np.asarray([[0.0, 0.7 - 0.2j], [0.0, 0.0]], dtype=complex)
            ),
            (0, 1): sparse.csr_matrix(
                np.asarray([[0.0, -0.1 + 0.3j], [0.0, 0.0]], dtype=complex)
            ),
        },
        "toy",
        {},
    )
    reflection = FiniteGroupGenerator(
        name="C2",
        antiunitary=False,
        canonical_k_map=((1, 0), (0, -1)),
        q_permutation=(0,),
        sector_permutation=(0, 1),
        k_forward=((1.0, 0.0), (0.0, -1.0)),
        internal_u=np.diag([1.0, -1.0]).astype(complex),
    )
    time_reversal = FiniteGroupGenerator(
        name="T",
        antiunitary=True,
        canonical_k_map=((-1, 0), (0, -1)),
        q_permutation=(0,),
        sector_permutation=(0, 1),
        k_forward=((-1.0, 0.0), (0.0, -1.0)),
        internal_u=np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=complex),
    )
    group = build_finite_group([reflection, time_reversal])
    candidates = compile_candidate_responses(
        [seed],
        coordinate=coordinate,
        group=group,
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    kpoints = np.asarray([[0.2, -0.1], [-0.31, 0.24], [0.07, 0.11]])
    compiled = basis.response_tensor(kpoints)

    for index, component in enumerate(("real", "imag")):
        np.testing.assert_allclose(
            compiled[:, index],
            dense_direct_response(
                seed,
                kpoints,
                coordinate=coordinate,
                group=group,
                component=component,
            ),
            atol=3.0e-13,
        )


class _MatrixGenerator:
    def __init__(self, matrices: dict[str, np.ndarray]) -> None:
        self.matrices = matrices

    def get_operator(self, name: str, params: object = None) -> np.ndarray:
        del params
        return self.matrices[name]


def test_model_action_adapter_requires_explicit_maps_and_builds_discrete_keys() -> None:
    generator = _MatrixGenerator({"T": np.eye(2, dtype=np.complex128)})
    operation = {
        "name": "T",
        "antiunitary": True,
        "k_map": {"type": "negation"},
        "q_map": {"type": "negation"},
        "sector_map": "identity",
    }
    group = finite_group_from_model_actions(
        [operation],
        symmetry_gen=generator,
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([0.0, 1.0]),
        Q_set1=np.asarray([[0.0, 0.0]]),
        Q_set2=np.asarray([[0.0, 0.0]]),
        sectors=[
            {"name": "L1", "qset": "qset1"},
            {"name": "L2", "qset": "qset2"},
        ],
    )

    assert len(group.elements) == 2
    assert group.elements[1].discrete_key == (
        True,
        ((-1, 0), (0, -1)),
        (0, 1),
        (0, 1),
    )

    missing_q_map = dict(operation)
    missing_q_map.pop("q_map")
    with pytest.raises(ValueError, match="explicit q_map"):
        finite_group_from_model_actions(
            [missing_q_map],
            symmetry_gen=generator,
            bM1=np.asarray([1.0, 0.0]),
            bM2=np.asarray([0.0, 1.0]),
            Q_set1=np.asarray([[0.0, 0.0]]),
            Q_set2=np.asarray([[0.0, 0.0]]),
            sectors=[
                {"name": "L1", "qset": "qset1"},
                {"name": "L2", "qset": "qset2"},
            ],
        )


def test_model_action_adapter_uses_resolved_action_that_matches_exactified_u() -> None:
    generator = _MatrixGenerator(
        {"TR": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)}
    )
    operation = {
        "name": "TR",
        "antiunitary": True,
        "k_map": {"type": "negation"},
        "q_map": {"type": "negation"},
        "sector_map": "identity",
        "internal_resolved_action": {
            "antiunitary": True,
            "k_map": {"type": "negation"},
            "q_map": {"type": "negation"},
            "sector_map": "layer_exchange",
        },
    }

    group = finite_group_from_model_actions(
        [operation],
        symmetry_gen=generator,
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([0.0, 1.0]),
        Q_set1=np.asarray([[0.0, 0.0]]),
        Q_set2=np.asarray([[0.0, 0.0]]),
        sectors=[
            {"name": "L1", "qset": "qset1"},
            {"name": "L2", "qset": "qset2"},
        ],
    )

    assert group.elements[1].q_permutation == (1, 0)
    assert group.elements[1].sector_permutation == (1, 0)


def _projected_exchange_operation(
    *,
    items: list[dict[str, object]] | None = None,
    sector_map: str = "layer_exchange",
) -> dict[str, object]:
    if items is None:
        items = [
            {
                "source_sector": "L1",
                "source_q_index": 0,
                "target_sector": "L2",
                "target_q_index": 0,
                "q_residual": 0.0,
            },
            {
                "source_sector": "L2",
                "source_q_index": 0,
                "target_sector": "L1",
                "target_q_index": 0,
                "q_residual": 0.0,
            },
        ]
    return {
        "name": "TR",
        "antiunitary": True,
        "k_map": {"type": "negation"},
        "q_map": {"type": "negation"},
        "sector_map": "identity",
        "model_basis_action": {
            "complete": True,
            "sector_map": sector_map,
            "items": items,
            "support_resolution": {
                "action_mismatch": True,
                "candidate_source": "support_discovery",
                "provenance": {
                    "source": "support_exactification",
                    "accepted_by": "kp_projected_basis_inference",
                    "accepted_by_user": False,
                },
            },
        },
    }


def _two_sector_tr_group(
    operation: dict[str, object],
    *,
    factorized_actions: dict[str, object] | None = None,
):
    generator = _MatrixGenerator(
        {"TR": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)}
    )
    extra = (
        {}
        if factorized_actions is None
        else {"factorized_actions": factorized_actions}
    )
    return finite_group_from_model_actions(
        [operation],
        symmetry_gen=generator,
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([0.0, 1.0]),
        Q_set1=np.asarray([[0.0, 0.0]]),
        Q_set2=np.asarray([[0.0, 0.0]]),
        sectors=[
            {"name": "L1", "qset": "qset1"},
            {"name": "L2", "qset": "qset2"},
        ],
        **extra,
    )


def test_model_action_adapter_uses_complete_projected_basis_action() -> None:
    group = _two_sector_tr_group(_projected_exchange_operation())

    assert group.elements[1].q_permutation == (1, 0)
    assert group.elements[1].sector_permutation == (1, 0)


def test_model_action_adapter_rejects_factorized_projected_permutation_mismatch() -> None:
    factorized = certify_factorized_action(
        name="TR",
        matrix=np.eye(2, dtype=np.complex128),
        antiunitary=True,
        k_forward=-np.eye(2, dtype=np.float64),
        q_permutation=(0, 1),
        sector_permutation=(0, 1),
        q_vectors=(np.zeros((1, 2)), np.zeros((1, 2))),
        q_counts=(1, 1),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )

    with pytest.raises(ValueError, match="factorized.*Q permutation.*projected"):
        _two_sector_tr_group(
            _projected_exchange_operation(),
            factorized_actions={"TR": factorized},
        )


@pytest.mark.parametrize(
    ("items", "sector_map", "message"),
    [
        (
            [
                {
                    "source_sector": "L1",
                    "source_q_index": 0,
                    "target_sector": "L2",
                    "target_q_index": 0,
                    "q_residual": 0.0,
                }
            ],
            "layer_exchange",
            "incomplete.*source",
        ),
        (
            [
                {
                    "source_sector": "L1",
                    "source_q_index": 0,
                    "target_sector": "L2",
                    "target_q_index": 0,
                    "q_residual": 0.0,
                },
                {
                    "source_sector": "L2",
                    "source_q_index": 0,
                    "target_sector": "L2",
                    "target_q_index": 0,
                    "q_residual": 0.0,
                },
            ],
            "layer_exchange",
            "not bijective",
        ),
        (
            [
                {
                    "source_sector": "L3",
                    "source_q_index": 0,
                    "target_sector": "L2",
                    "target_q_index": 0,
                    "q_residual": 0.0,
                },
                {
                    "source_sector": "L2",
                    "source_q_index": 0,
                    "target_sector": "L1",
                    "target_q_index": 0,
                    "q_residual": 0.0,
                },
            ],
            "layer_exchange",
            "unknown.*sector",
        ),
        (
            None,
            "identity",
            "sector_map.*inconsistent",
        ),
    ],
)
def test_model_action_adapter_rejects_malformed_projected_basis_action(
    items: list[dict[str, object]] | None,
    sector_map: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _two_sector_tr_group(
            _projected_exchange_operation(items=items, sector_map=sector_map)
        )


def test_model_action_adapter_ignores_inactive_zero_orbital_qset() -> None:
    operation = {
        "name": "C3z",
        "antiunitary": False,
        "k_map": {"type": "identity"},
        "q_map": {"type": "identity"},
        "sector_map": "identity",
        "model_basis_action": {
            "complete": True,
            "sector_map": "identity",
            "items": [
                {
                    "source_sector": "L2",
                    "source_q_index": 0,
                    "target_sector": "L2",
                    "target_q_index": 0,
                    "q_residual": 0.0,
                }
            ],
        },
    }

    class Symmetry:
        dim = 1

        @staticmethod
        def get_operator(_name: str, _params: object = None) -> np.ndarray:
            return np.eye(1, dtype=np.complex128)

    group = finite_group_from_model_actions(
        [operation],
        symmetry_gen=Symmetry(),
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([0.0, 1.0]),
        Q_set1=np.asarray([[0.0, 0.0]]),
        Q_set2=np.asarray([[0.0, 0.0]]),
        sectors=[
            {"name": "L1", "qset": "qset1"},
            {"name": "L2", "qset": "qset2"},
        ],
        n_orb=(0, 1),
    )

    assert group.elements[0].q_permutation == (0, 1)


def _complete_onsite_config(*, q_shift: float = 0.0) -> MoireConfig:
    return MoireConfig(
        Q_set1=np.asarray([[q_shift, 0.0]], dtype=float),
        Q_set2=np.empty((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        nlow_state=[2, 0],
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([0.0, 1.0]),
        intra_harmonics_map={},
        inter_harmonics_map={},
        max_order={"Kinect": 0, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"Onsite": [], "Kinect": [], "intra": [], "inter": []},
        term_templates=[
            {
                "name": "complete_onsite",
                "source": "onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
                "term_space_policy": "complete",
            }
        ],
        response_semantics=COMPLETE_LINEAR_V2,
        symmetry_source_metadata={
            "exactification_owner": "kp_symm",
            "kp_symm_exactification": {
                "status": "exactified",
                "polynomial_coordinate": {
                    "coordinate_convention": "right_handed_model_cartesian_reciprocal_v1",
                    "origin": [0.0, 0.0],
                    "origin_role": "exactified_valley_expansion_origin_in_model_cartesian",
                    "valley": "Gamma",
                },
            },
        },
    )


def _literal_harmonic_record_config(
    records: list[dict[str, object]],
) -> MoireConfig:
    return MoireConfig(
        Q_set1=np.asarray([[0.0, 0.0]], dtype=float),
        Q_set2=np.asarray([[1.0, 0.0]], dtype=float),
        n_orb1=2,
        n_orb2=2,
        nlow_state=[2, 2],
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([0.0, 1.0]),
        intra_harmonics_map={},
        inter_harmonics_map={},
        max_order={"Kinect": 0, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"Onsite": [], "Kinect": [], "intra": [], "inter": []},
        term_templates=[
            {
                "name": "gamma_case_inter_L2_to_L1",
                "source": "tunneling",
                "sector_pairs": [["L2", "L1"]],
                "orbital_pairs": "all",
                "harmonic_records": records,
                "max_order": 0,
                "term_space_policy": "complete",
            }
        ],
        sectors=[
            {"name": "L1", "qset": "qset1", "n_orb": 2},
            {"name": "L2", "qset": "qset2", "n_orb": 2},
        ],
        response_semantics=COMPLETE_LINEAR_V2,
    )


def test_literal_harmonic_records_materialize_all_ordered_orbital_pairs() -> None:
    config = _literal_harmonic_record_config(
        [
            {
                "id": "inter:1",
                "kind": "inter",
                "vector": [1.0, 0.0],
                "source": "case_q_pair_support",
                "support_count": 1,
            }
        ]
    )

    model = build_model(config)

    assert len(model.candidate_terms) == 4
    assert {
        (term.key.orbital_from, term.key.orbital_to)
        for term in model.candidate_terms
    } == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert {term.key.p for term in model.candidate_terms} == {(1.0, 0.0)}
    assert {
        term.registry_metadata["harmonic_id"]
        for term in model.candidate_terms
    } == {"inter:1"}


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {
                    "id": "bad-shape",
                    "kind": "inter",
                    "vector": [1.0],
                    "source": "case_q_pair_support",
                    "support_count": 1,
                }
            ],
            "finite two-vector",
        ),
        (
            [
                {
                    "id": "not-finite",
                    "kind": "inter",
                    "vector": [np.nan, 0.0],
                    "source": "case_q_pair_support",
                    "support_count": 1,
                }
            ],
            "finite two-vector",
        ),
        (
            [
                {
                    "id": "no-support",
                    "kind": "inter",
                    "vector": [1.0, 0.0],
                    "source": "case_q_pair_support",
                    "support_count": 0,
                }
            ],
            "positive support_count",
        ),
        (
            [
                {
                    "id": "duplicate-a",
                    "kind": "inter",
                    "vector": [1.0, 0.0],
                    "source": "case_q_pair_support",
                    "support_count": 1,
                },
                {
                    "id": "duplicate-b",
                    "kind": "inter",
                    "vector": [1.0, 0.0],
                    "source": "case_q_pair_support",
                    "support_count": 1,
                },
            ],
            "duplicate harmonic record",
        ),
    ],
)
def test_literal_harmonic_records_fail_closed(
    records: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_model(_literal_harmonic_record_config(records))


class _GaugeCovarianceSymmetryGenerator(_MatrixGenerator):
    def __init__(
        self,
        matrices: dict[str, np.ndarray],
        factorized_actions: dict[str, object],
    ) -> None:
        super().__init__(matrices)
        self.factorized_actions = factorized_actions
        self.dim = next(iter(matrices.values())).shape[0]
        self.metadata: dict[str, object] = {}

    def get_factorized_action(self, name: str) -> object | None:
        return self.factorized_actions.get(name)


def test_nonclosed_factorized_seed_vocabulary_has_the_same_physical_projector_as_joint_and_dense() -> None:
    from kp.model.response_basis_factorized import (
        FactorizedTermActionError,
        compile_factorized_group_element_actions,
        compile_factorized_raw_seed_action,
        compile_joint_route_group_element_actions,
    )
    from kp.model.response_basis_symmetry_first import (
        compile_symmetry_first_fixed_space,
    )

    qset = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    # A quarter-turn relative phase makes the missing authored Fourier
    # direction well separated instead of manufacturing a nearly singular
    # epsilon-scale counterexample.
    q_phases = np.asarray(
        [1.0, 1.0j, 1.0, -1.0j],
        dtype=np.complex128,
    )
    q_permutation = (2, 3, 0, 1)
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[np.asarray(q_permutation), np.arange(4)] = q_phases
    factorized = certify_factorized_action(
        name="C2",
        matrix=matrix,
        antiunitary=False,
        k_forward=np.eye(2),
        q_permutation=q_permutation,
        sector_permutation=(1, 0),
        q_vectors=(qset, qset),
        q_counts=(2, 2),
        n_orb=(1, 1),
        matrix_absolute_error_bound=1.0e-13,
        q_absolute_error_bound=1.0e-13,
    )
    joint = BlockRouteAction(
        name="C2",
        antiunitary=False,
        fiber_permutation=q_permutation,
        fiber_dimensions=(1, 1, 1, 1),
        fiber_indices=((0,), (1,), (2,), (3,)),
        route_blocks=tuple(
            np.asarray([[phase]], dtype=np.complex128) for phase in q_phases
        ),
    )
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seeds = tuple(
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
            seed_id=f"inter:{layer_from}->{layer_to}",
            Q_set1=qset,
            Q_set2=qset,
            n_orb1=1,
            n_orb2=1,
            coordinate=coordinate,
            support_component="inter",
            metadata={
                "term_index": term_index,
                "term_space_policy": "complete",
            },
        )
        for term_index, (layer_from, layer_to) in enumerate(((1, 2), (2, 1)))
    )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((1, 0), (0, 1)),
                q_permutation=q_permutation,
                sector_permutation=(1, 0),
                k_forward=((1.0, 0.0), (0.0, 1.0)),
                internal_u=matrix,
            )
        ]
    )

    # The authored equal-amplitude interlayer seeds are deliberately not
    # closed under the Q-dependent phase pattern.
    with pytest.raises(
        FactorizedTermActionError,
        match="not constant on its support|outside the authored seed vocabulary",
    ):
        compile_factorized_raw_seed_action(
            seeds=seeds,
            factorized_action=factorized,
        )

    factorized_actions, factorized_artifact = (
        compile_factorized_group_element_actions(
            group=group,
            factorized_generators={"C2": factorized},
        )
    )
    factorized_symbolic = compile_symmetry_first_fixed_space(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={"C2": factorized},
        internal_actions_by_word=factorized_actions,
        factorized_group_artifact=factorized_artifact,
    )
    joint_actions, joint_artifact = compile_joint_route_group_element_actions(
        group=group,
        joint_route_generators={"C2": joint},
        joint_artifact_hash="a" * 64,
    )
    joint_symbolic = compile_symmetry_first_fixed_space(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={},
        internal_actions_by_word=joint_actions,
        factorized_group_artifact=joint_artifact,
    )
    dense = compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=group,
    )
    dense_vectors = sparse.hstack(
        [
            response_basis_module._channel_sparse_vector(
                channel,
                coordinate,
                4,
            )
            for channel in dense.channels
        ],
        format="csc",
    ).toarray()

    physical_bases = (
        factorized_symbolic.physical_basis.toarray(),
        joint_symbolic.physical_basis.toarray(),
        dense_vectors,
    )
    ranks = [np.linalg.matrix_rank(basis, tol=1.0e-12) for basis in physical_bases]
    assert factorized_symbolic.rank == joint_symbolic.rank == ranks[-1] > 0
    assert ranks[0] == ranks[1] == ranks[2]
    reference_projector = physical_bases[-1] @ np.linalg.pinv(
        physical_bases[-1],
        rcond=1.0e-12,
    )
    for basis in physical_bases[:-1]:
        projector = basis @ np.linalg.pinv(basis, rcond=1.0e-12)
        np.testing.assert_allclose(projector, reference_projector, atol=5.0e-12)


def _gamma_gauge_covariance_config(
    *,
    matrices: dict[str, np.ndarray],
    valley_type: str = "Gamma",
    action_specs: dict[str, dict[str, object]] | None = None,
    operations: list[dict[str, object]] | None = None,
) -> MoireConfig:
    qset1 = np.asarray([[0.0, 0.0]], dtype=float)
    qset2 = np.asarray([[0.0, 0.0]], dtype=float)
    sectors = [
        {"name": "L1", "qset": "qset1", "n_orb": 2},
        {"name": "L2", "qset": "qset2", "n_orb": 2},
    ]
    if action_specs is None:
        action_specs = {
            "C2": {
                "antiunitary": False,
                "k_forward": -np.eye(2, dtype=float),
                "q_permutation": (1, 0),
                "sector_permutation": (1, 0),
            },
            "T": {
                "antiunitary": True,
                "k_forward": -np.eye(2, dtype=float),
                "q_permutation": (0, 1),
                "sector_permutation": (0, 1),
            },
        }
    factorized = {
        name: certify_factorized_action(
            name=name,
            matrix=matrix,
            antiunitary=bool(action_specs[name]["antiunitary"]),
            k_forward=np.asarray(action_specs[name]["k_forward"], dtype=float),
            q_permutation=tuple(action_specs[name]["q_permutation"]),
            sector_permutation=tuple(action_specs[name]["sector_permutation"]),
            q_vectors=(qset1, qset2),
            q_counts=(1, 1),
            n_orb=(2, 2),
            matrix_absolute_error_bound=1.0e-13,
            q_absolute_error_bound=1.0e-13,
        )
        for name, matrix in matrices.items()
    }
    if operations is None:
        operations = [
            {
                "name": "C2",
                "antiunitary": False,
                "k_map": {"type": "negation"},
                "q_map": {"type": "negation"},
                "sector_map": "layer_exchange",
            },
            {
                "name": "T",
                "antiunitary": True,
                "k_map": {"type": "negation"},
                "q_map": {"type": "negation"},
                "sector_map": "identity",
            },
        ]
    templates, _diagnostics = _case_derived_term_templates(
        valley_type=valley_type,
        sectors=sectors,
        n_orb=(2, 2),
        max_order={"Kinect": 0, "Onsite": 0, "intra": 0, "inter": 0},
        Q_set1=qset1,
        Q_set2=qset2,
        intra_harmonics_map={1: np.zeros(2, dtype=float)},
        inter_harmonics_map={1: np.zeros(2, dtype=float)},
    )
    return MoireConfig(
        Q_set1=qset1,
        Q_set2=qset2,
        n_orb1=2,
        n_orb2=2,
        nlow_state=[2, 2],
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([-0.5, np.sqrt(3.0) / 2.0]),
        intra_harmonics_map={1: np.zeros(2, dtype=float)},
        inter_harmonics_map={1: np.zeros(2, dtype=float)},
        max_order={"Kinect": 0, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_gen=_GaugeCovarianceSymmetryGenerator(matrices, factorized),
        symmetry_map={
            "Kinect": copy.deepcopy(operations),
            "Onsite": copy.deepcopy(operations),
            "intra": copy.deepcopy(operations),
            "inter": copy.deepcopy(operations),
        },
        term_templates=templates,
        sectors=sectors,
        response_semantics=COMPLETE_LINEAR_V2,
        symmetry_source_metadata={
            "exactification_owner": "kp_symm",
            "kp_symm_exactification": {
                "status": "exactified",
                "polynomial_coordinate": {
                    "coordinate_convention": (
                        "right_handed_model_cartesian_reciprocal_v1"
                    ),
                    "origin": [0.0, 0.0],
                    "origin_role": (
                        "exactified_valley_expansion_origin_in_model_cartesian"
                    ),
                    "valley": valley_type,
                },
            },
        },
    )


def _real_response_columns(
    basis: CompiledResponseBasis,
    *,
    transport: np.ndarray | None = None,
) -> np.ndarray:
    responses = basis.response_tensor([[0.0, 0.0]])[0]
    if transport is not None:
        responses = np.asarray(
            [
                transport.conj().T @ response @ transport
                for response in responses
            ],
            dtype=np.complex128,
        )
    return np.stack(
        [
            np.concatenate([response.real.ravel(), response.imag.ravel()])
            for response in responses
        ],
        axis=1,
    )


def test_gamma_case_basis_is_covariant_under_l2_orbital_swap() -> None:
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    W = sparse.block_diag((identity, swap), format="csr").toarray()
    matrices_a = {
        "C2": np.block(
            [
                [np.zeros((2, 2), dtype=np.complex128), sigma_z],
                [sigma_z, np.zeros((2, 2), dtype=np.complex128)],
            ]
        ),
        "T": np.eye(4, dtype=np.complex128),
    }
    matrices_b = {
        "C2": W.conj().T @ matrices_a["C2"] @ W,
        "T": W.conj().T @ matrices_a["T"] @ W.conj(),
    }

    basis_a = compile_model_response_basis(
        build_model(_gamma_gauge_covariance_config(matrices=matrices_a)),
        _gamma_gauge_covariance_config(matrices=matrices_a),
        reduce=True,
    )
    basis_b = compile_model_response_basis(
        build_model(_gamma_gauge_covariance_config(matrices=matrices_b)),
        _gamma_gauge_covariance_config(matrices=matrices_b),
        reduce=True,
    )

    assert len(basis_a.channels) == len(basis_b.channels)
    columns_a_in_b = _real_response_columns(basis_a, transport=W)
    columns_b = _real_response_columns(basis_b)
    projector_a_in_b = columns_a_in_b @ np.linalg.pinv(columns_a_in_b)
    projector_b = columns_b @ np.linalg.pinv(columns_b)
    np.testing.assert_allclose(projector_b, projector_a_in_b, atol=5.0e-11)

    kpoints = np.asarray([[0.0, 0.0], [0.17, -0.09]], dtype=float)
    coefficients = np.linspace(0.1, 1.0, len(basis_a.channels))
    target_a = np.einsum(
        "c,kcij->kij",
        coefficients,
        basis_a.response_tensor(kpoints),
        optimize=True,
    )
    target_b = np.asarray(
        [W.conj().T @ hamiltonian @ W for hamiltonian in target_a]
    )
    fitted_a = basis_a.fit(
        kpoints,
        target_a,
        fit_indices=[0, 1],
        regularization=0.0,
        band_window=[0, 4],
    )
    fitted_b = basis_b.fit(
        kpoints,
        target_b,
        fit_indices=[0, 1],
        regularization=0.0,
        band_window=[0, 4],
    )
    runtime_a = CompiledResponseRuntime(basis_a, fitted_a)
    runtime_b = CompiledResponseRuntime(basis_b, fitted_b)
    predicted_a = runtime_a.hamiltonians(kpoints)
    predicted_b = runtime_b.hamiltonians(kpoints)

    residual_a = np.linalg.norm(predicted_a - target_a)
    residual_b = np.linalg.norm(predicted_b - target_b)
    assert residual_a == pytest.approx(residual_b, abs=5.0e-11)
    np.testing.assert_allclose(
        predicted_b,
        np.asarray([W.conj().T @ value @ W for value in predicted_a]),
        atol=5.0e-11,
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(predicted_a),
        np.linalg.eigvalsh(predicted_b),
        atol=5.0e-11,
    )


@pytest.mark.parametrize("valley_type", ["K", "M"])
def test_k_m_case_basis_is_covariant_under_dense_l2_orbital_gauge(
    valley_type: str,
) -> None:
    identity = np.eye(2, dtype=np.complex128)
    hadamard = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    W = sparse.block_diag((identity, hadamard), format="csr").toarray()
    if valley_type == "K":
        omega = np.exp(2.0j * np.pi / 3.0)
        orbital_c3 = np.diag([omega, np.conjugate(omega)])
        matrices_a = {
            "C3z": sparse.block_diag(
                (orbital_c3, orbital_c3),
                format="csr",
            ).toarray()
        }
        angle = 2.0 * np.pi / 3.0
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=float,
        )
        action_specs = {
            "C3z": {
                "antiunitary": False,
                "k_forward": rotation,
                "q_permutation": (0, 1),
                "sector_permutation": (0, 1),
            }
        }
        operations = [
            {
                "name": "C3z",
                "antiunitary": False,
                "k_map": {"type": "rotation", "angle_deg": 120.0},
                "q_map": {"type": "rotation", "angle_deg": 120.0},
                "sector_map": "identity",
            }
        ]
    else:
        sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
        matrices_a = {
            "C2": np.block(
                [
                    [np.zeros((2, 2), dtype=np.complex128), sigma_z],
                    [sigma_z, np.zeros((2, 2), dtype=np.complex128)],
                ]
            ),
            "T": np.eye(4, dtype=np.complex128),
        }
        action_specs = {
            "C2": {
                "antiunitary": False,
                "k_forward": -np.eye(2, dtype=float),
                "q_permutation": (1, 0),
                "sector_permutation": (1, 0),
            },
            "T": {
                "antiunitary": True,
                "k_forward": -np.eye(2, dtype=float),
                "q_permutation": (0, 1),
                "sector_permutation": (0, 1),
            },
        }
        operations = [
            {
                "name": "C2",
                "antiunitary": False,
                "k_map": {"type": "negation"},
                "q_map": {"type": "negation"},
                "sector_map": "layer_exchange",
            },
            {
                "name": "T",
                "antiunitary": True,
                "k_map": {"type": "negation"},
                "q_map": {"type": "negation"},
                "sector_map": "identity",
            },
        ]
    matrices_b = {
        name: (
            W.conj().T @ matrix @ W.conj()
            if bool(action_specs[name]["antiunitary"])
            else W.conj().T @ matrix @ W
        )
        for name, matrix in matrices_a.items()
    }
    config_a = _gamma_gauge_covariance_config(
        matrices=matrices_a,
        valley_type=valley_type,
        action_specs=action_specs,
        operations=operations,
    )
    config_b = _gamma_gauge_covariance_config(
        matrices=matrices_b,
        valley_type=valley_type,
        action_specs=action_specs,
        operations=operations,
    )

    basis_a = compile_model_response_basis(
        build_model(config_a),
        config_a,
        reduce=True,
    )
    basis_b = compile_model_response_basis(
        build_model(config_b),
        config_b,
        reduce=True,
    )

    assert len(basis_a.channels) == len(basis_b.channels)
    columns_a_in_b = _real_response_columns(basis_a, transport=W)
    columns_b = _real_response_columns(basis_b)
    projector_a_in_b = columns_a_in_b @ np.linalg.pinv(columns_a_in_b)
    projector_b = columns_b @ np.linalg.pinv(columns_b)
    np.testing.assert_allclose(projector_b, projector_a_in_b, atol=5.0e-11)

    kpoints = np.asarray([[0.0, 0.0], [0.17, -0.09]], dtype=float)
    coefficients = np.linspace(0.1, 1.0, len(basis_a.channels))
    target_a = np.einsum(
        "c,kcij->kij",
        coefficients,
        basis_a.response_tensor(kpoints),
        optimize=True,
    )
    target_b = np.asarray(
        [W.conj().T @ hamiltonian @ W for hamiltonian in target_a]
    )
    fitted_a = basis_a.fit(
        kpoints,
        target_a,
        fit_indices=[0, 1],
        regularization=0.0,
        band_window=[0, 4],
    )
    fitted_b = basis_b.fit(
        kpoints,
        target_b,
        fit_indices=[0, 1],
        regularization=0.0,
        band_window=[0, 4],
    )
    predicted_a = CompiledResponseRuntime(basis_a, fitted_a).hamiltonians(kpoints)
    predicted_b = CompiledResponseRuntime(basis_b, fitted_b).hamiltonians(kpoints)

    assert np.linalg.norm(predicted_a - target_a) == pytest.approx(
        np.linalg.norm(predicted_b - target_b),
        abs=5.0e-11,
    )
    np.testing.assert_allclose(
        predicted_b,
        np.asarray([W.conj().T @ value @ W for value in predicted_a]),
        atol=5.0e-11,
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(predicted_a),
        np.linalg.eigvalsh(predicted_b),
        atol=5.0e-11,
    )


def test_orbital_swap_is_not_a_sufficient_gauge_covariance_regression() -> None:
    identity = np.eye(2, dtype=np.complex128)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    hadamard = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    diagonal_seeds = [
        np.diag([1.0, 0.0]).astype(np.complex128),
        np.diag([0.0, 1.0]).astype(np.complex128),
    ]

    def columns(
        transport: np.ndarray | None,
    ) -> np.ndarray:
        matrices = (
            diagonal_seeds
            if transport is None
            else [
                transport.conj().T @ matrix @ transport
                for matrix in diagonal_seeds
            ]
        )
        return np.stack(
            [
                np.concatenate([matrix.real.ravel(), matrix.imag.ravel()])
                for matrix in matrices
            ],
            axis=1,
        )

    regenerated = columns(None)
    regenerated_projector = regenerated @ np.linalg.pinv(regenerated)
    swap_columns = columns(swap)
    dense_columns = columns(hadamard)
    swap_projector_delta = np.linalg.norm(
        regenerated_projector - swap_columns @ np.linalg.pinv(swap_columns)
    )
    dense_rotation_old_profile_projector_delta = np.linalg.norm(
        regenerated_projector - dense_columns @ np.linalg.pinv(dense_columns)
    )

    assert swap_projector_delta < 5.0e-11
    assert dense_rotation_old_profile_projector_delta > 1.0e-3


def test_complete_policy_enumerates_ordered_pairs_and_reduces_in_coefficient_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _complete_onsite_config()
    model = build_model(config)
    assert len(model.terms) == 4

    def fail_legacy_complex_rank(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("complete_linear_v2 entered legacy complex-rank path")

    monkeypatch.setattr(
        "kp.model.core.ContinuumModelBuilder._orthogonalize_hermitian_matrices_with_support",
        fail_legacy_complex_rank,
    )
    basis = compile_model_response_basis(model, config, reduce=True)

    assert basis.candidate_artifact["candidate_channel_count"] == 8
    assert basis.candidate_artifact["logical_candidate_channel_count"] == 8
    assert basis.candidate_artifact["authored_ordered_seed_count"] == 4
    assert basis.candidate_artifact["adjoint_orbit_descriptor_count"] == 3
    assert basis.candidate_artifact["hermitian_ambient_channel_count"] == 4
    assert (
        basis.candidate_artifact["physically_materialized_projected_channel_count"]
        == 4
    )
    assert (
        basis.candidate_artifact["physically_compiled_representative_channel_count"]
        == 4
    )
    assert basis.candidate_artifact["adjoint_certified_dropped_channel_count"] == 4
    assert len(basis.channels) == 4
    assert all(
        proof["block_rule"] == "filtered_degree_residual_owner_pivots"
        for proof in basis.reduction_proofs
    )


def test_complete_p0_model_routes_through_graded_symbolic_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kp.model.response_basis_graded as graded_module

    clear_response_basis_cache()
    config = _complete_onsite_config()
    model = build_model(config)
    calls = 0
    original = graded_module.compile_graded_candidate_group

    def counted_graded_compile(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        graded_module,
        "compile_graded_candidate_group",
        counted_graded_compile,
    )

    basis = compile_model_response_basis(model, config, reduce=True)
    assert calls == 1
    assert any(
        proof["solver"] == "graded_filtered_symbolic_p0_reynolds_v1"
        for proof in basis.reduction_proofs
    )
    identity_artifact = basis.candidate_artifact["adjoint"]["groups"][0]
    assert identity_artifact["certification"] == "graded_filtered_reynolds_v1"
    outer_timings = basis.candidate_artifact["adjoint"]["outer_timings_seconds"]
    for key in (
        "certified_group_actions",
        "support_mask_closure",
        "p0_complete_compile",
        "finite_complete_compile",
        "group_artifact_merge",
    ):
        assert outer_timings[key] >= 0.0


def test_complete_p0_graded_compiler_does_not_repeat_postcompile_adjoint_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_response_basis_cache()

    def fail_repeated_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("p=0 graded compile repeated the raw-adjoint scan")

    monkeypatch.setattr(
        response_basis_module,
        "_certify_joint_adjoint_seed_orbits",
        fail_repeated_scan,
    )
    config = _complete_onsite_config()
    basis = compile_model_response_basis(build_model(config), config, reduce=True)

    assert basis.candidate_artifact["adjoint_orbit_descriptor_count"] == 3
    assert basis.candidate_artifact["hermitian_ambient_channel_count"] == 4
    assert basis.candidate_artifact["adjoint_certified_dropped_channel_count"] == 4


def test_declared_p0_orbit_representative_uses_primary_reynolds_route() -> None:
    clear_response_basis_cache()
    config = _complete_onsite_config()
    config.term_templates = [
        {
            "name": "onsite_orbit_representative",
            "source": "onsite",
            "sector_pairs": [[1, 1]],
            "orbital_pairs": [[1, 2]],
            "max_order": 0,
            "term_space_policy": "orbit_representative",
        }
    ]
    model = build_model(config)
    assert len(model.terms) == 1

    basis = compile_model_response_basis(model, config, reduce=True)

    assert len(basis.channels) == 2
    orbit_artifact = basis.candidate_artifact["adjoint"]["groups"][0]
    assert orbit_artifact["certification"] == (
        "declared_orbit_representative_reynolds_hermitian_projection_v1"
    )
    assert orbit_artifact["fallback_used"] is False
    coefficients = [channel.coefficient(0, 0).toarray() for channel in basis.channels]
    assert all(np.allclose(matrix, matrix.conj().T) for matrix in coefficients)
    real_vectors = np.column_stack(
        [
            np.concatenate([matrix.real.ravel(), matrix.imag.ravel()])
            for matrix in coefficients
        ]
    )
    assert np.linalg.matrix_rank(real_vectors) == 2


def test_complete_finite_p_routes_through_graded_symbolic_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    q_vectors = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    seed = raw_polynomial_seed_from_term_key(
        SimpleNamespace(
            Mz=0,
            Mz_star=0,
            layer_from=1,
            layer_to=1,
            orbital_from=1,
            orbital_to=1,
            p=(1.0, 0.0),
        ),
        seed_id="finite-p-production-route",
        Q_set1=q_vectors,
        Q_set2=np.empty((0, 2)),
        n_orb1=1,
        n_orb2=0,
        coordinate=coordinate,
        support_component="intra",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )
    group = identity_finite_group(2, q_size=2, sector_size=2)
    duplicate_seed = dataclasses.replace(
        seed,
        seed_id="finite-p-production-route-duplicate",
        metadata={**dict(seed.metadata), "term_index": 1},
    )

    def fail_full_reynolds(*_args, **_kwargs):
        raise AssertionError("finite-p production entered per-seed CSR Reynolds")

    def fail_secondary_reduction(*_args, **_kwargs):
        raise AssertionError("symbolically certified finite-p span was reduced twice")

    monkeypatch.setattr(
        response_basis_module,
        "_compile_candidate_group_with_adjoint_fallback",
        fail_full_reynolds,
    )
    monkeypatch.setattr(
        response_basis_module,
        "_reduce_confirmed_channels",
        fail_secondary_reduction,
    )
    basis = response_basis_module._compile_model_response_basis_uncached(
        coordinate=coordinate,
        groups=[group],
        seeds_by_group=[[seed, duplicate_seed]],
        factorized_actions_by_group=[{}],
        dim=2,
        identity_payload=_identity_payload(2),
        reduce=True,
        cache_key="test-finite-p-symbolic-route",
        progress_callback=None,
    )

    assert len(basis.channels) == 2
    assert basis.candidate_artifact["candidate_channel_count"] == 4
    assert basis.candidate_artifact["logical_candidate_channel_count"] == 4
    assert basis.candidate_artifact["adjoint"]["groups"][0]["certification"] == (
        "graded_filtered_reynolds_v1"
    )


def test_complete_finite_p_typed_graded_fallback_records_symbolic_oracle_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kp.model.response_basis_graded as graded_module

    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    q_vectors = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    seed = raw_polynomial_seed_from_term_key(
        SimpleNamespace(
            Mz=0,
            Mz_star=0,
            layer_from=1,
            layer_to=1,
            orbital_from=1,
            orbital_to=1,
            p=(1.0, 0.0),
        ),
        seed_id="finite-p-typed-graded-fallback",
        Q_set1=q_vectors,
        Q_set2=np.empty((0, 2)),
        n_orb1=1,
        n_orb2=0,
        coordinate=coordinate,
        support_component="intra",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )

    def reject_graded(*_args: object, **_kwargs: object) -> None:
        raise graded_module.GradedCompilationUnavailable(
            "synthetic finite-p typed fallback"
        )

    monkeypatch.setattr(
        graded_module,
        "compile_graded_candidate_group",
        reject_graded,
    )

    basis = response_basis_module._compile_model_response_basis_uncached(
        coordinate=coordinate,
        groups=[identity_finite_group(2, q_size=2, sector_size=2)],
        seeds_by_group=[[seed]],
        factorized_actions_by_group=[{}],
        dim=2,
        identity_payload=_identity_payload(2),
        reduce=True,
        cache_key="test-finite-p-typed-graded-fallback-provenance",
        progress_callback=None,
    )

    assert basis.candidate_artifact["adjoint"]["groups"][0]["certification"] == (
        "closed_symbolic_atom_reynolds_v1"
    )
    assert [
        (proof["solver"], proof["block_rule"])
        for proof in basis.reduction_proofs
    ] == [
        (
            "symbolic_reynolds_typed_fallback_v1",
            "symbolic_atom_support_components",
        )
    ]


def test_fixed_group_direct_sum_is_certified_from_disjoint_coefficient_rows() -> None:
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "left",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]))},
                "left",
                {},
            ),
            RawPolynomialSeed(
                "right",
                {(0, 0): sparse.csr_matrix(np.diag([0.0, 1.0]))},
                "right",
                {},
            ),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    left = next(channel for channel in candidates.channels if channel.channel_id == "left:real")
    right = next(channel for channel in candidates.channels if channel.channel_id == "right:real")
    group_type = response_basis_module.GeneratorFixedResponseGroup
    left_group = group_type(candidates, (left,), (), ())
    right_group = group_type(candidates, (right,), (), ())
    overlap_group = group_type(candidates, (left,), (), ())

    assert response_basis_module._fixed_groups_have_disjoint_support(
        [left_group, right_group],
        coordinate=_toy_coordinate(),
        dim=2,
    )
    assert not response_basis_module._fixed_groups_have_disjoint_support(
        [left_group, overlap_group],
        coordinate=_toy_coordinate(),
        dim=2,
    )


def test_complete_p0_uses_symbolic_oracle_for_typed_graded_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kp.model.response_basis_graded as graded_module
    import kp.model.response_basis_symmetry_first as symbolic_module

    clear_response_basis_cache()
    config = _complete_onsite_config()
    symbolic_calls = 0
    original_symbolic = symbolic_module.compile_symbolic_atom_candidate_group

    def reject_graded(*_args: object, **_kwargs: object) -> None:
        raise graded_module.GradedCompilationUnavailable("synthetic typed fallback")

    def counted_symbolic(*args: object, **kwargs: object):
        nonlocal symbolic_calls
        symbolic_calls += 1
        return original_symbolic(*args, **kwargs)

    monkeypatch.setattr(
        graded_module,
        "compile_graded_candidate_group",
        reject_graded,
    )
    monkeypatch.setattr(
        symbolic_module,
        "compile_symbolic_atom_candidate_group",
        counted_symbolic,
    )

    basis = compile_model_response_basis(
        model=build_model(config), config=config, reduce=True
    )
    assert symbolic_calls == 1
    assert any(
        proof["solver"] == "symbolic_p0_reynolds_typed_fallback_v1"
        for proof in basis.reduction_proofs
    )


def _p0_factorized_fail_closed_case():
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seed = raw_polynomial_seed_from_term_key(
        SimpleNamespace(
            Mz=0,
            Mz_star=0,
            layer_from=1,
            layer_to=1,
            orbital_from=1,
            orbital_to=1,
            p=(0.0, 0.0),
        ),
        seed_id="p0-factorized-fail-closed",
        Q_set1=np.asarray([[0.0, 0.0]]),
        Q_set2=np.empty((0, 2)),
        n_orb1=1,
        n_orb2=0,
        coordinate=coordinate,
        support_component="intra",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(0,),
                sector_permutation=(0,),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=np.ones((1, 1), dtype=np.complex128),
            )
        ]
    )
    return coordinate, seed, group


def test_complete_p0_propagates_factorized_term_action_error() -> None:
    from kp.model.response_basis_factorized import FactorizedTermActionError

    coordinate, seed, group = _p0_factorized_fail_closed_case()

    with pytest.raises(FactorizedTermActionError, match="does not match generator") as caught:
        response_basis_module._compile_model_response_basis_uncached(
            coordinate=coordinate,
            groups=[group],
            seeds_by_group=[[seed]],
            factorized_actions_by_group=[
                {"C2": SimpleNamespace(name="not-C2")}
            ],
            dim=1,
            identity_payload=_identity_payload(1),
            reduce=True,
            cache_key="test-p0-factorized-error-propagates",
            progress_callback=None,
        )

    assert any(
        "sparse group actions" in note and "group 0" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_complete_p0_rejects_missing_certified_factorized_action() -> None:
    from kp.model.response_basis_factorized import FactorizedTermActionError

    coordinate, seed, group = _p0_factorized_fail_closed_case()

    with pytest.raises(
        FactorizedTermActionError,
        match="missing group words.*C2",
    ) as caught:
        response_basis_module._compile_model_response_basis_uncached(
            coordinate=coordinate,
            groups=[group],
            seeds_by_group=[[seed]],
            factorized_actions_by_group=[{}],
            dim=1,
            identity_payload=_identity_payload(1),
            reduce=True,
            cache_key="test-p0-missing-factorized-action-rejected",
            progress_callback=None,
        )

    assert any(
        "sparse group actions" in note and "group 0" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_complete_p0_uses_certified_joint_routes_for_nonclosed_authored_span() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    qset = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    seeds = [
        raw_polynomial_seed_from_term_key(
            SimpleNamespace(
                Mz=0,
                Mz_star=0,
                layer_from=1,
                layer_to=1,
                orbital_from=orbital_from + 1,
                orbital_to=orbital_to + 1,
                p=(0.0, 0.0),
            ),
            seed_id=f"p0-joint:{orbital_from}:{orbital_to}",
            Q_set1=qset,
            Q_set2=np.empty((0, 2)),
            n_orb1=2,
            n_orb2=0,
            coordinate=coordinate,
            support_component="kinetic",
            metadata={"term_index": 2 * orbital_from + orbital_to, "term_space_policy": "complete"},
        )
        for orbital_from in range(2)
        for orbital_to in range(2)
    ]
    x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    route = BlockRouteAction(
        name="C2",
        antiunitary=False,
        fiber_permutation=(0, 1),
        fiber_dimensions=(2, 2),
        fiber_indices=((0, 2), (1, 3)),
        route_blocks=(x, z),
    )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(0, 1),
                sector_permutation=(0,),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=materialize_block_route_action(route),
            )
        ]
    )

    compiled = response_basis_module._compile_model_response_basis_uncached(
        coordinate=coordinate,
        groups=[group],
        seeds_by_group=[seeds],
        factorized_actions_by_group=[None],
        joint_route_actions_by_group=[{"C2": route}],
        joint_artifact_hashes_by_group=["a" * 64],
        dim=4,
        identity_payload=_identity_payload(4),
        reduce=True,
        cache_key="test-p0-joint-route-physical-closure",
        progress_callback=None,
    )

    assert compiled.channels
    assert "joint_route_sparse_group_elements_v1" in str(
        compiled.candidate_artifact
    )


def test_complete_p0_rejects_malformed_singleton_identity_action() -> None:
    from kp.model.response_basis_factorized import FactorizedTermActionError

    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seed = raw_polynomial_seed_from_term_key(
        SimpleNamespace(
            Mz=0,
            Mz_star=0,
            layer_from=1,
            layer_to=1,
            orbital_from=1,
            orbital_to=1,
            p=(0.0, 0.0),
        ),
        seed_id="malformed-singleton-identity",
        Q_set1=np.asarray([[0.0, 0.0]]),
        Q_set2=np.empty((0, 2)),
        n_orb1=1,
        n_orb2=0,
        coordinate=coordinate,
        support_component="intra",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )
    identity_group = identity_finite_group(1, q_size=1, sector_size=1)
    malformed_element = dataclasses.replace(
        identity_group.elements[0],
        k_pullback=((1.0, 0.0), (0.0, 2.0)),
    )
    malformed_group = dataclasses.replace(
        identity_group,
        elements=(malformed_element,),
    )

    with pytest.raises(
        FactorizedTermActionError,
        match="empty-word finite-group element is not a strict identity",
    ):
        response_basis_module._compile_model_response_basis_uncached(
            coordinate=coordinate,
            groups=[malformed_group],
            seeds_by_group=[[seed]],
            factorized_actions_by_group=[{}],
            dim=1,
            identity_payload=_identity_payload(1),
            reduce=True,
            cache_key="test-malformed-singleton-identity-rejected",
            progress_callback=None,
        )


def test_complete_finite_p_propagates_factorized_term_action_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kp.model import response_basis_factorized
    from kp.model.response_basis_factorized import FactorizedTermActionError

    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seed = raw_polynomial_seed_from_term_key(
        SimpleNamespace(
            Mz=0,
            Mz_star=0,
            layer_from=1,
            layer_to=1,
            orbital_from=1,
            orbital_to=1,
            p=(1.0, 0.0),
        ),
        seed_id="finite-p-factorized-error",
        Q_set1=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        Q_set2=np.empty((0, 2)),
        n_orb1=1,
        n_orb2=0,
        coordinate=coordinate,
        support_component="intra",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )

    def reject_factorized(*_args: object, **_kwargs: object) -> None:
        raise FactorizedTermActionError("synthetic factorized action failure")

    monkeypatch.setattr(
        response_basis_factorized,
        "compile_factorized_group_element_actions",
        reject_factorized,
    )

    with pytest.raises(
        FactorizedTermActionError,
        match="synthetic factorized action failure",
    ) as caught:
        response_basis_module._compile_model_response_basis_uncached(
            coordinate=coordinate,
            groups=[identity_finite_group(2, q_size=2, sector_size=2)],
            seeds_by_group=[[seed]],
            factorized_actions_by_group=[{}],
            dim=2,
            identity_payload=_identity_payload(2),
            reduce=True,
            cache_key="test-factorized-error-propagates",
            progress_callback=None,
        )

    assert any(
        "certified sparse group actions" in note and "group 0" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_complete_finite_p_propagates_unexpected_graded_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kp.model import response_basis_graded

    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    seed = raw_polynomial_seed_from_term_key(
        SimpleNamespace(
            Mz=0,
            Mz_star=0,
            layer_from=1,
            layer_to=1,
            orbital_from=1,
            orbital_to=1,
            p=(1.0, 0.0),
        ),
        seed_id="finite-p-symbolic-error",
        Q_set1=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        Q_set2=np.empty((0, 2)),
        n_orb1=1,
        n_orb2=0,
        coordinate=coordinate,
        support_component="intra",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )

    def crash_graded(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic graded implementation bug")

    monkeypatch.setattr(
        response_basis_graded,
        "compile_graded_candidate_group",
        crash_graded,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic graded implementation bug",
    ) as caught:
        response_basis_module._compile_model_response_basis_uncached(
            coordinate=coordinate,
            groups=[identity_finite_group(2, q_size=2, sector_size=2)],
            seeds_by_group=[[seed]],
            factorized_actions_by_group=[{}],
            dim=2,
            identity_payload=_identity_payload(2),
            reduce=True,
            cache_key="test-graded-error-propagates",
            progress_callback=None,
        )

    assert any(
        "finite-p symbolic-atom" in note and "group 0" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_complete_fit_reports_authored_orbit_ambient_fixed_and_materialized_counts() -> None:
    clear_response_basis_cache()
    config = _complete_onsite_config()
    config.kpoints_fit = np.asarray([[0.0, 0.0]], dtype=float)
    config.heff = np.zeros((1, 2, 2), dtype=np.complex128)
    config.response_fit_indices = [0]
    messages: list[str] = []

    compute_coefficients(
        config,
        build_model(config),
        progress_callback=lambda message, **_kwargs: messages.append(str(message)),
    )

    vocabulary_lines = [
        message for message in messages if message.startswith("response basis vocabulary:")
    ]
    assert vocabulary_lines == [
        "response basis vocabulary: authored=4 adjoint_orbits=3 "
        "hermitian_ambient=4 fixed=4 materialized=4"
    ]


def test_complete_compiler_preserves_authored_seeds_before_key_deduplication() -> None:
    config = _complete_onsite_config()
    duplicate = copy.deepcopy(config.term_templates[0])
    duplicate["name"] = "complete_onsite_second_author"
    config.term_templates = [config.term_templates[0], duplicate]

    model = build_model(config)

    # The legacy runtime dictionary may keep one value per physical key, but the
    # target-independent candidate registry must retain both authored templates.
    assert len(model.terms) == 4
    assert len(model.candidate_terms) == 8

    basis = compile_model_response_basis(model, config, reduce=False)

    assert basis.candidate_artifact["candidate_channel_count"] == 16
    group_artifact = basis.candidate_artifact["adjoint"]["groups"][0]
    assert group_artifact["physical_seed_count"] == 4
    assert group_artifact["duplicate_seed_groups"] == (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    authored_names = {
        source["family"]
        for channel in basis.candidate_artifact["channels"]
        for source in channel["metadata"]["graded_source_provenance"]
    }
    assert authored_names == {"complete_onsite", "complete_onsite_second_author"}


def test_model_basis_requires_exactified_polynomial_origin_metadata() -> None:
    config = _complete_onsite_config()
    config.symmetry_source_metadata = {
        "exactification_owner": "kp_symm",
        "kp_symm_exactification": {"status": "exactified"},
    }

    with pytest.raises(ValueError, match="polynomial_coordinate.*Rerun `kp symm`"):
        compile_model_response_basis(build_model(config), config, reduce=True)


def test_model_basis_cache_compiles_once_per_unique_target_independent_key() -> None:
    clear_response_basis_cache()
    config = _complete_onsite_config()
    model = build_model(config)

    first = compile_model_response_basis(model, config, reduce=True)
    second = compile_model_response_basis(model, config, reduce=True)
    stats = response_basis_cache_stats()

    assert first is second
    assert stats["cold_compile_count_by_key"] == {stats["keys"][0]: 1}

    changed_config = _complete_onsite_config(q_shift=0.2)
    changed_model = build_model(changed_config)
    changed = compile_model_response_basis(changed_model, changed_config, reduce=True)
    changed_stats = response_basis_cache_stats()
    assert changed.basis_hash != first.basis_hash
    assert len(changed_stats["cold_compile_count_by_key"]) == 2
    assert set(changed_stats["cold_compile_count_by_key"].values()) == {1}


def test_model_basis_persistent_cache_survives_memory_cache_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_response_basis_cache()
    config = _complete_onsite_config()
    config.output_dir = tmp_path / "model"
    cold_compile_calls = 0

    import kp.model.response_basis as response_basis_module

    original_cold_compile = response_basis_module._compile_model_response_basis_uncached

    def counted_cold_compile(*args: object, **kwargs: object):
        nonlocal cold_compile_calls
        cold_compile_calls += 1
        return original_cold_compile(*args, **kwargs)

    monkeypatch.setattr(
        response_basis_module,
        "_compile_model_response_basis_uncached",
        counted_cold_compile,
    )

    first = compile_model_response_basis(build_model(config), config, reduce=True)
    assert cold_compile_calls == 1
    cache_files = list(
        (Path(config.output_dir) / ".compiled_response_basis_cache").glob("*.npz")
    )
    assert len(cache_files) == 1

    clear_response_basis_cache()
    second = compile_model_response_basis(build_model(config), config, reduce=True)

    assert second.basis_hash == first.basis_hash
    assert cold_compile_calls == 1
    assert str(config.output_dir) not in str(second.artifact())


def test_model_basis_persistent_cache_v52_cannot_bypass_v53_structural_generators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_response_basis_cache()
    assert response_basis_module.COMPILER_VERSION == (
        "complete-response-basis-v2-structural-generators-v53"
    )
    config = _complete_onsite_config()
    config.output_dir = tmp_path / "model"

    monkeypatch.setattr(
        response_basis_module,
        "COMPILER_VERSION",
        "complete-response-basis-v2-graded-finite-p-v52",
    )
    compile_model_response_basis(build_model(config), config, reduce=True)
    cache_dir = Path(config.output_dir) / ".compiled_response_basis_cache"
    v52_files = set(cache_dir.glob("*.npz"))
    assert len(v52_files) == 1

    clear_response_basis_cache()
    monkeypatch.setattr(
        response_basis_module,
        "COMPILER_VERSION",
        "complete-response-basis-v2-structural-generators-v53",
    )
    cold_calls = 0
    original_cold_compile = response_basis_module._compile_model_response_basis_uncached

    def counted_cold_compile(*args: object, **kwargs: object):
        nonlocal cold_calls
        cold_calls += 1
        return original_cold_compile(*args, **kwargs)

    monkeypatch.setattr(
        response_basis_module,
        "_compile_model_response_basis_uncached",
        counted_cold_compile,
    )
    compile_model_response_basis(build_model(config), config, reduce=True)

    assert cold_calls == 1
    v53_files = set(cache_dir.glob("*.npz"))
    assert len(v53_files) == 2
    assert v52_files < v53_files


def test_model_basis_persistent_cache_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    clear_response_basis_cache()
    config = _complete_onsite_config()
    config.output_dir = tmp_path / "model"
    compile_model_response_basis(build_model(config), config, reduce=True)
    cache_files = list(
        (Path(config.output_dir) / ".compiled_response_basis_cache").glob("*.npz")
    )
    assert len(cache_files) == 1
    cache_files[0].write_bytes(b"not-an-npz")

    clear_response_basis_cache()
    with pytest.raises(ResponseBasisCacheCorruptionError, match="will not be recompiled"):
        compile_model_response_basis(build_model(config), config, reduce=True)


def test_model_basis_persistent_cache_requires_complete_identity_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_response_basis_cache()
    config = _complete_onsite_config()
    config.output_dir = tmp_path / "model"

    import kp.model.response_basis as response_basis_module

    original_identity = response_basis_module._basis_layout_identity

    def incomplete_identity(*args: object, **kwargs: object) -> dict[str, object]:
        identity = original_identity(*args, **kwargs)
        identity.pop("q_permutations")
        return identity

    monkeypatch.setattr(
        response_basis_module,
        "_basis_layout_identity",
        incomplete_identity,
    )

    with pytest.raises(ValueError, match="persistent cache input.*q_permutations"):
        compile_model_response_basis(build_model(config), config, reduce=True)


def test_complete_fit_progress_distinguishes_compile_fit_and_warm_load_stages(
    tmp_path: Path,
) -> None:
    clear_response_basis_cache()
    config = _complete_onsite_config()
    config.output_dir = tmp_path / "model"
    config.kpoints_fit = np.asarray([[0.0, 0.0]])
    config.heff = np.asarray([np.diag([1.5, -0.25])], dtype=np.complex128)
    config.coeff_tol = 0.0
    cold_messages: list[str] = []

    compute_coefficients(
        config,
        build_model(config),
        progress_callback=lambda message, **_kwargs: cold_messages.append(str(message)),
    )

    assert any("response basis persistent cache lookup" in item for item in cold_messages)
    assert any("response basis persistent cache miss" in item for item in cold_messages)
    assert any("response basis cold compile" in item for item in cold_messages)
    assert any("response basis candidate compile" in item for item in cold_messages)
    assert any("response basis rank reduction" in item for item in cold_messages)
    assert any("fit response sparse-union design" in item for item in cold_messages)
    assert any("fit response rank/solve" in item for item in cold_messages)

    clear_response_basis_cache()
    warm_messages: list[str] = []
    compute_coefficients(
        config,
        build_model(config),
        progress_callback=lambda message, **_kwargs: warm_messages.append(str(message)),
    )

    assert any("response basis persistent cache load" in item for item in warm_messages)
    assert not any("response basis cold compile" in item for item in warm_messages)
    assert not any("response basis candidate compile" in item for item in warm_messages)
    assert not any("response basis rank reduction" in item for item in warm_messages)
    assert any("fit response sparse-union design" in item for item in warm_messages)
    assert any("fit response rank/solve" in item for item in warm_messages)


def test_model_basis_persistent_cache_is_warm_in_second_python_process(
    tmp_path: Path,
) -> None:
    cache_output = tmp_path / "model"
    script = textwrap.dedent(
        f"""
        import numpy as np
        from kp.model.core import MoireConfig, build_model
        from kp.model.response_basis import COMPLETE_LINEAR_V2, compile_model_response_basis

        config = MoireConfig(
            Q_set1=np.asarray([[0.0, 0.0]], dtype=float),
            Q_set2=np.empty((0, 2), dtype=float),
            n_orb1=2,
            n_orb2=0,
            nlow_state=[2, 0],
            bM1=np.asarray([1.0, 0.0]),
            bM2=np.asarray([0.0, 1.0]),
            intra_harmonics_map={{}},
            inter_harmonics_map={{}},
            max_order={{"Kinect": 0, "Onsite": 0, "intra": 0, "inter": 0}},
            symmetry_map={{"Onsite": [], "Kinect": [], "intra": [], "inter": []}},
            term_templates=[{{
                "name": "complete_onsite",
                "source": "onsite",
                "sector_pairs": [[1, 1]],
                "orbital_pairs": "diagonal",
                "max_order": 0,
                "term_space_policy": "complete",
            }}],
            response_semantics=COMPLETE_LINEAR_V2,
            output_dir={str(cache_output)!r},
            symmetry_source_metadata={{
                "exactification_owner": "kp_symm",
                "kp_symm_exactification": {{
                    "status": "exactified",
                    "polynomial_coordinate": {{
                        "coordinate_convention": "right_handed_model_cartesian_reciprocal_v1",
                        "origin": [0.0, 0.0],
                        "origin_role": "exactified_valley_expansion_origin_in_model_cartesian",
                        "valley": "Gamma",
                    }},
                }},
            }},
        )
        messages = []
        basis = compile_model_response_basis(
            build_model(config),
            config,
            reduce=True,
            progress_callback=lambda message, **kwargs: messages.append(str(message)),
        )
        print(basis.basis_hash)
        print("\\n".join(messages))
        """
    )

    first = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "response basis cold compile" in first.stdout
    assert "response basis persistent cache load" in second.stdout
    assert "response basis cold compile" not in second.stdout


def test_model_basis_same_key_concurrent_processes_compile_once(tmp_path: Path) -> None:
    cache_output = tmp_path / "model"
    count_path = tmp_path / "cold-compile-count"
    script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        import kp.model.response_basis as response_basis_module
        from kp.model.core import build_model
        from tests.kp.test_complete_response_basis import _complete_onsite_config

        config = _complete_onsite_config()
        config.output_dir = {str(cache_output)!r}
        count_path = Path({str(count_path)!r})
        original = response_basis_module._compile_model_response_basis_uncached
        def counted(*args, **kwargs):
            with count_path.open("a", encoding="utf-8") as handle:
                handle.write("compile\\n")
            time.sleep(0.5)
            return original(*args, **kwargs)
        response_basis_module._compile_model_response_basis_uncached = counted
        response_basis_module.compile_model_response_basis(
            build_model(config), config, reduce=True
        )
        """
    )
    processes = [subprocess.Popen([sys.executable, "-c", script]) for _ in range(2)]
    return_codes = [process.wait(timeout=30) for process in processes]

    assert return_codes == [0, 0]
    assert count_path.read_text(encoding="utf-8").splitlines() == ["compile"]


def test_model_basis_hash_ignores_target_identity_metadata_from_projection() -> None:
    clear_response_basis_cache()
    config_a = _complete_onsite_config()
    config_a.symmetry_source_metadata.update({
        "heff_hash": "target-a",
        "k_indices_hash": "fit-grid-a",
        "band_window": [0, 2],
    })
    basis_a = compile_model_response_basis(build_model(config_a), config_a, reduce=True)

    config_b = _complete_onsite_config()
    config_b.symmetry_source_metadata.update({
        "heff_hash": "target-b",
        "k_indices_hash": "fit-grid-b",
        "band_window": [1, 2],
    })
    basis_b = compile_model_response_basis(build_model(config_b), config_b, reduce=True)

    assert basis_a.basis_hash == basis_b.basis_hash
    assert response_basis_cache_stats()["cached_basis_count"] == 1


def test_complete_fit_and_bands_consume_one_compiled_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_response_basis_cache()
    target = np.asarray(
        [[[1.0, 0.3 + 0.4j], [0.3 - 0.4j, 2.0]]],
        dtype=np.complex128,
    )
    config = _complete_onsite_config()
    config.kpoints_fit = np.asarray([[0.0, 0.0]])
    config.kpoints = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    config.heff = target.copy()
    config.coeff_tol = 0.0
    model = build_model(config)

    def fail_legacy_complex_rank(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("complete_linear_v2 entered legacy complex-rank path")

    monkeypatch.setattr(
        "kp.model.core.ContinuumModelBuilder._orthogonalize_hermitian_matrices_with_support",
        fail_legacy_complex_rank,
    )
    fitted_model, diagnostics = compute_coefficients(config, model)
    basis = fitted_model._compiled_response_basis
    fitted = fitted_model._fitted_response_model

    assert diagnostics["response_semantics"] == COMPLETE_LINEAR_V2
    assert diagnostics["candidate_response_channels"] == 8
    assert diagnostics["retained_basis_channels"] == 4
    assert diagnostics["fit_selected_channels"] == 4
    assert diagnostics["nonzero_coefficient_channels"] == 4
    np.testing.assert_allclose(
        basis.hamiltonians([[0.0, 0.0]], fitted.coefficients),
        target,
        atol=2.0e-12,
    )

    runtime_sink: list[object] = []
    hamiltonians: list[np.ndarray] = []
    bands = compute_bands(
        config,
        fitted_model,
        config.kpoints,
        hamiltonians_out=hamiltonians,
        compiled_runtime_out=runtime_sink,
    )
    assert isinstance(runtime_sink[0], CompiledResponseRuntime)
    np.testing.assert_allclose(hamiltonians[0], target[0], atol=2.0e-12)
    np.testing.assert_allclose(bands[0], np.linalg.eigvalsh(target[0]), atol=2.0e-12)
    stats = response_basis_cache_stats()
    assert set(stats["cold_compile_count_by_key"].values()) == {1}


def test_complete_runtime_frozen_arrays_reload_without_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _complete_onsite_config()
    config.kpoints_fit = np.asarray([[0.0, 0.0]])
    config.heff = np.asarray([np.diag([1.5, -0.25])], dtype=np.complex128)
    config.coeff_tol = 0.0
    model, _diagnostics = compute_coefficients(config, build_model(config))
    runtime = CompiledResponseRuntime(
        model._compiled_response_basis,
        model._fitted_response_model,
    )
    arrays = runtime.export_arrays()

    def fail_compile(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("frozen complete response runtime attempted to recompile")

    monkeypatch.setattr("kp.model.response_basis.compile_model_response_basis", fail_compile)
    restored = CompiledResponseRuntime.from_frozen_arrays(arrays)

    assert restored.basis.basis_hash == runtime.basis.basis_hash
    np.testing.assert_allclose(
        restored.hamiltonian([0.13, -0.07]),
        runtime.hamiltonian([0.13, -0.07]),
        atol=2.0e-13,
    )


def test_runtime_rejects_coefficient_length_or_normalization_mismatch() -> None:
    config = _complete_onsite_config()
    config.kpoints_fit = np.asarray([[0.0, 0.0]])
    config.heff = np.asarray([np.diag([1.5, -0.25])], dtype=np.complex128)
    config.coeff_tol = 0.0
    model, _diagnostics = compute_coefficients(config, build_model(config))
    basis = model._compiled_response_basis
    fitted = model._fitted_response_model

    with pytest.raises(ValueError, match="coefficient vector"):
        CompiledResponseRuntime(
            basis,
            dataclasses.replace(fitted, coefficients=fitted.coefficients[:-1]),
        )
    with pytest.raises(ValueError, match="coefficient normalization"):
        CompiledResponseRuntime(
            basis,
            dataclasses.replace(fitted, coefficient_normalization="wrong-version"),
        )


def test_fit_grid_rank_and_pivots_never_change_compiled_basis() -> None:
    coordinate = _toy_coordinate(max_degree=1)
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "constant-E11",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))},
                "diag",
                {},
            ),
            RawPolynomialSeed(
                "linear-E22",
                {
                    (1, 0): sparse.csr_matrix(np.diag([0.0, 0.5]).astype(complex)),
                    (0, 1): sparse.csr_matrix(np.diag([0.0, 0.5]).astype(complex)),
                },
                "diag",
                {},
            ),
        ],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    artifact_before = copy.deepcopy(basis.artifact())
    rank_one = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([1.0, 0.0])], dtype=complex),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )
    rank_two = basis.fit(
        [[0.0, 0.0], [0.2, 0.0]],
        np.asarray([np.diag([1.0, 0.0]), np.diag([1.0, 0.2])], dtype=complex),
        fit_indices=[0, 1],
        regularization=0.0,
        band_window=[0, 2],
    )

    assert rank_one.fit_pivots == rank_two.fit_pivots == ()
    assert rank_one.fit_design_certified_rank == 1
    assert rank_two.fit_design_certified_rank == 2
    assert basis.artifact() == artifact_before
    assert rank_one.basis_hash == rank_two.basis_hash == basis.basis_hash
    assert rank_one.fit_hash != rank_two.fit_hash


def test_fit_rank_excludes_directions_below_propagated_fit_design_error() -> None:
    """Fit-grid dependence is target state, not a mutation of term-space rank."""

    coordinate = _toy_coordinate(max_degree=1)
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("constant", {(0, 0): e11}, "diag", {}),
            RawPolynomialSeed(
                "nearly-constant-on-fit-grid",
                {
                    (0, 0): e11,
                    (1, 0): 5.0e-4 * e11,
                    (0, 1): 5.0e-4 * e11,
                },
                "diag",
                {},
            ),
        ],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(
                channel,
                propagated_error_bound=1.0e-2 * channel.response_scale,
                error_bound_components={"certified_test_bound": 1.0e-2 * channel.response_scale},
                classification="confirmed_nonzero",
            )
            if channel.component == "real"
            else channel
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    artifact_before = copy.deepcopy(basis.artifact())
    points = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    target = np.asarray([2.0 * np.diag([1.0, 0.0]), 2.001 * np.diag([1.0, 0.0])])

    fitted = basis.fit(
        points,
        target,
        fit_indices=[0, 1],
        regularization=0.0,
        band_window=[0, 2],
    )

    assert fitted.fit_pivots == ()
    assert fitted.fit_response_gram_rank == 1
    assert (
        fitted.fit_response_gram_eigenvalues[1]
        <= fitted.fit_response_gram_rank_tolerance
    )
    assert fitted.fit_design_rank_tolerance >= fitted.fit_design_propagated_error_bound
    assert (
        fitted.fit_solver_policy
        == "real_physical_coefficient_gram_whitened_minimum_norm_svd_v4"
    )
    assert fitted.fit_selected_singular_value_min > fitted.fit_selected_error_bound
    assert basis.artifact() == artifact_before


def test_fit_records_certified_rank_in_whitened_physical_response_span() -> None:
    """Redundant owner channels do not reduce the physical fit-design rank."""

    coordinate = _toy_coordinate(max_degree=0)
    root_three_over_two = np.sqrt(3.0) / 2.0
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "direction-a",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))},
                "diag",
                {},
            ),
            RawPolynomialSeed(
                "direction-b",
                {
                    (0, 0): sparse.csr_matrix(
                        np.diag([0.5, root_three_over_two]).astype(complex)
                    )
                },
                "diag",
                {},
            ),
            RawPolynomialSeed(
                "direction-c",
                {
                    (0, 0): sparse.csr_matrix(
                        np.diag([-0.5, root_three_over_two]).astype(complex)
                    )
                },
                "diag",
                {},
            ),
        ],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(
                channel,
                propagated_error_bound=0.51 * channel.response_scale,
                error_bound_components={
                    "certified_test_bound": 0.51 * channel.response_scale
                },
                classification="confirmed_nonzero",
            )
            if channel.component == "real"
            else channel
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )

    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([2.0, 3.0])], dtype=complex),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )

    assert fitted.fit_pivots == ()
    assert fitted.fit_response_gram_rank == 2
    assert fitted.fit_design_certified_rank == 2
    assert fitted.fit_selected_singular_value_min > fitted.fit_selected_error_bound


def test_regularized_fit_uses_all_confirmed_channels_instead_of_pretruncating() -> None:
    """Tikhonov regularization must stabilize, not discard, dependent channels."""

    coordinate = _toy_coordinate(max_degree=1)
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed("constant", {(0, 0): e11}, "diag", {}),
            RawPolynomialSeed(
                "nearly-constant-on-fit-grid",
                {
                    (0, 0): e11,
                    (1, 0): 5.0e-4 * e11,
                    (0, 1): 5.0e-4 * e11,
                },
                "diag",
                {},
            ),
        ],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(
                channel,
                propagated_error_bound=1.0e-2 * channel.response_scale,
                error_bound_components={"certified_test_bound": 1.0e-2 * channel.response_scale},
                classification="confirmed_nonzero",
            )
            if channel.component == "real"
            else channel
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    points = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    target = np.asarray([2.0 * np.diag([1.0, 0.0]), 2.001 * np.diag([1.0, 0.0])])

    fitted = basis.fit(
        points,
        target,
        fit_indices=[0, 1],
        regularization=1.0e-4,
        band_window=[0, 2],
    )

    confirmed_ids = tuple(
        channel.channel_id
        for channel in basis.channels
        if channel.classification == "confirmed_nonzero"
    )
    assert fitted.fit_pivots == ()
    assert fitted.fit_selected_channel_ids == ()
    assert fitted.fit_solver_channel_ids == confirmed_ids
    assert len(fitted.fit_solver_channel_indices) == len(confirmed_ids) == 2
    assert fitted.fit_design_certified_rank == 1
    assert fitted.regularization == pytest.approx(1.0e-4)
    assert np.max(np.abs(fitted.coefficients) * basis.response_scales) < 2.0
    solver_indices = np.asarray(fitted.fit_solver_channel_indices, dtype=int)
    complex_design, support = basis._response_design_on_union_support(points, solver_indices)
    normalized = complex_design / basis.response_scales[solver_indices]
    real_design = np.vstack((normalized.real, normalized.imag))
    complex_target = target.reshape(points.shape[0], -1)[:, support].reshape(-1)
    real_target = np.concatenate((complex_target.real, complex_target.imag))
    expected_theta = np.linalg.solve(
        real_design.T @ real_design + 1.0e-4 * np.eye(2),
        real_design.T @ real_target,
    )
    np.testing.assert_allclose(
        fitted.coefficients[solver_indices] * basis.response_scales[solver_indices],
        expected_theta,
        atol=1.0e-10,
    )


def _spectral_fit_toy_basis(*, only_e11: bool = False) -> CompiledResponseBasis:
    e11 = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=complex))
    seeds = [RawPolynomialSeed("E11", {(0, 0): e11}, "toy", {})]
    if not only_e11:
        e22 = sparse.csr_matrix(np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=complex))
        e12 = sparse.csr_matrix(np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex))
        seeds.extend(
            [
                RawPolynomialSeed("E22", {(0, 0): e22}, "toy", {}),
                RawPolynomialSeed("E12", {(0, 0): e12}, "toy", {}),
            ]
        )
    candidates = compile_candidate_responses(
        seeds,
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    return CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )


def test_response_action_matches_dense_response_tensor() -> None:
    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]], dtype=float)
    frames = np.asarray(
        [
            [[1.0], [0.0]],
            [[1.0 / np.sqrt(2.0)], [1.0j / np.sqrt(2.0)]],
        ],
        dtype=np.complex128,
    )
    selected = np.asarray([0, 2], dtype=np.int64)

    action = basis.response_action(points, frames, channel_indices=selected)
    dense = basis.response_tensor(points)[:, selected]
    expected = np.einsum("kcmn,kns->kcms", dense, frames, optimize=True)

    np.testing.assert_allclose(action, expected, atol=1.0e-13)


def _dense_target_spectral_ridge_oracle(
    basis: CompiledResponseBasis,
    points: np.ndarray,
    target: np.ndarray,
    *,
    floor: float,
    alpha: float,
    ridge: float,
    n_bands: int = 1,
    two_sided_projector_weight: float = 0.0,
) -> np.ndarray:
    """Independent dense formula for the target-spectral production loss."""

    responses = basis.response_tensor(points)
    n_kpoints, _n_channels, dim, _ = responses.shape
    raw_weights = np.full((n_kpoints, dim), float(floor), dtype=float)
    raw_weights[:, -int(n_bands) :] += float(alpha) * (1.0 - float(floor))
    weights = raw_weights * (n_kpoints * dim / float(np.sum(raw_weights)))
    design_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for k_index in range(n_kpoints):
        _values, vectors = np.linalg.eigh(target[k_index])
        square_root = (vectors * np.sqrt(weights[k_index])) @ vectors.conj().T
        design_parts.append(
            np.stack(
                [(response @ square_root).reshape(-1) for response in responses[k_index]],
                axis=1,
            )
        )
        target_parts.append((target[k_index] @ square_root).reshape(-1))
        if float(two_sided_projector_weight) > 0.0:
            selected = vectors[:, -int(n_bands) :]
            root_weight = float(np.sqrt(two_sided_projector_weight))
            design_parts.append(
                root_weight
                * np.stack(
                    [
                        (selected.conj().T @ response @ selected).reshape(-1)
                        for response in responses[k_index]
                    ],
                    axis=1,
                )
            )
            target_parts.append(
                root_weight
                * (selected.conj().T @ target[k_index] @ selected).reshape(-1)
            )
    complex_design = np.vstack(design_parts)
    complex_target = np.concatenate(target_parts)
    scales = basis.response_scales
    real_design = np.vstack((complex_design.real, complex_design.imag)) / scales[None, :]
    real_target = np.concatenate((complex_target.real, complex_target.imag))
    theta = np.linalg.solve(
        real_design.T @ real_design + float(ridge) * np.eye(real_design.shape[1]),
        real_design.T @ real_target,
    )
    return theta / scales


def _dense_normalized_low_energy_ridge_oracle(
    basis: CompiledResponseBasis,
    points: np.ndarray,
    target: np.ndarray,
    *,
    n_bands: int,
    one_sided_weight: float,
    two_sided_weight: float,
    ridge: float,
) -> np.ndarray:
    """Independent dense formula for the public dimension-normalized loss."""

    responses = basis.response_tensor(points)
    n_kpoints, _n_channels, dim, _ = responses.shape
    design_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for k_index in range(n_kpoints):
        _values, vectors = np.linalg.eigh(target[k_index])
        selected = vectors[:, -int(n_bands) :]
        projector = selected @ selected.conj().T
        design_parts.append(
            np.stack(
                [(response / dim).reshape(-1) for response in responses[k_index]],
                axis=1,
            )
        )
        target_parts.append((target[k_index] / dim).reshape(-1))
        design_parts.append(
            np.sqrt(one_sided_weight / (dim * n_bands))
            * np.stack(
                [(response @ projector).reshape(-1) for response in responses[k_index]],
                axis=1,
            )
        )
        target_parts.append(
            np.sqrt(one_sided_weight / (dim * n_bands))
            * (target[k_index] @ projector).reshape(-1)
        )
        design_parts.append(
            np.sqrt(two_sided_weight) / n_bands
            * np.stack(
                [
                    (projector @ response @ projector).reshape(-1)
                    for response in responses[k_index]
                ],
                axis=1,
            )
        )
        target_parts.append(
            np.sqrt(two_sided_weight)
            / n_bands
            * (projector @ target[k_index] @ projector).reshape(-1)
        )
    complex_design = np.vstack(design_parts)
    complex_target = np.concatenate(target_parts)
    scales = basis.response_scales
    real_design = np.vstack((complex_design.real, complex_design.imag)) / scales[None, :]
    real_target = np.concatenate((complex_target.real, complex_target.imag))
    theta = np.linalg.solve(
        real_design.T @ real_design + float(ridge) * np.eye(real_design.shape[1]),
        real_design.T @ real_target,
    )
    return theta / scales


def test_normalized_public_linear_loss_matches_independent_dense_oracle() -> None:
    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    target = np.asarray(
        [
            [[0.4, 0.3 + 0.2j], [0.3 - 0.2j, 1.5]],
            [[-0.2, -0.15 + 0.35j], [-0.15 - 0.35j, 0.9]],
        ],
        dtype=complex,
    )
    ridge = 0.17
    one_sided_weight = 1.5
    two_sided_weight = 3.0

    fitted = basis.fit(
        points,
        target,
        fit_indices=[3, 9],
        regularization=ridge,
        band_window=[0, 2],
        spectral_weighting={
            "mode": "normalized_low_energy_linear_v1",
            "band_edge": "top",
            "window_mode": "fixed_count_degeneracy_safe",
            "n_bands": 1,
            "one_sided_weight": one_sided_weight,
            "two_sided_weight": two_sided_weight,
        },
    )
    expected = _dense_normalized_low_energy_ridge_oracle(
        basis,
        points,
        target,
        n_bands=1,
        one_sided_weight=one_sided_weight,
        two_sided_weight=two_sided_weight,
        ridge=ridge,
    )

    np.testing.assert_allclose(fitted.coefficients, expected, rtol=2.0e-11, atol=2.0e-11)
    assert fitted.fit_objective == {
        "mode": "normalized_low_energy_linear_v1",
        "version": "normalized-low-energy-linear-v1",
        "band_edge": "top",
        "window_mode": "fixed_count_degeneracy_safe",
        "requested_bands": 1,
        "resolved_band_counts": (1, 1),
        "one_sided_weight": one_sided_weight,
        "two_sided_weight": two_sided_weight,
        "normalization": "dimension_mean_square_v1",
    }


@pytest.mark.parametrize("target_bands", ["top", "bottom"])
def test_public_nonlinear_loss_and_jacobian_match_dense_formula(
    target_bands: str,
) -> None:
    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    target = np.asarray(
        [
            [[0.4, 0.3 + 0.2j], [0.3 - 0.2j, 1.5]],
            [[-0.2, -0.15 + 0.35j], [-0.15 - 0.35j, 0.9]],
        ],
        dtype=complex,
    )
    channel_indices = np.arange(len(basis.channels), dtype=int)
    coefficients = np.asarray([0.25, -0.35, 0.4, -0.1], dtype=float)[
        : len(basis.channels)
    ]
    one_sided_weight = 1.5
    two_sided_weight = 3.0
    band_loss_weight = 2.25

    root, center, _report = _public_hamiltonian_quadratic_residual(
        basis,
        channel_indices,
        points,
        target,
        target_bands=target_bands,
        bands=1,
        one_sided_weight=one_sided_weight,
        two_sided_weight=two_sided_weight,
    )
    h_residual = root @ coefficients - center
    model_h = basis.hamiltonians(points, coefficients)
    dense_loss = 0.0
    for k_index in range(len(points)):
        _values, vectors = np.linalg.eigh(target[k_index])
        selected = vectors[:, -1:] if target_bands == "top" else vectors[:, :1]
        projector = selected @ selected.conj().T
        delta = model_h[k_index] - target[k_index]
        dense_loss += (
            np.linalg.norm(delta) ** 2 / 4.0
            + one_sided_weight * np.linalg.norm(delta @ projector) ** 2 / 2.0
            + two_sided_weight * np.linalg.norm(projector @ delta @ projector) ** 2
        ) / len(points)
    assert float(h_residual @ h_residual) == pytest.approx(dense_loss, rel=2.0e-12)

    residual, jacobian, counts = _public_band_residual_and_jacobian(
        basis,
        coefficients,
        channel_indices,
        coefficients,
        points,
        target,
        target_bands=target_bands,
        bands=1,
        band_loss_weight=band_loss_weight,
    )
    assert counts == (1, 1)
    step = 1.0e-6
    finite_difference = np.empty_like(jacobian)
    for index in range(len(coefficients)):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[index] += step
        minus[index] -= step
        plus_residual = _public_band_residual_and_jacobian(
            basis,
            coefficients,
            channel_indices,
            plus,
            points,
            target,
            target_bands=target_bands,
            bands=1,
            band_loss_weight=band_loss_weight,
        )[0]
        minus_residual = _public_band_residual_and_jacobian(
            basis,
            coefficients,
            channel_indices,
            minus,
            points,
            target,
            target_bands=target_bands,
            bands=1,
            band_loss_weight=band_loss_weight,
        )[0]
        finite_difference[:, index] = (plus_residual - minus_residual) / (2.0 * step)
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2.0e-6, atol=2.0e-7)


@pytest.mark.parametrize("target_bands", ["top", "bottom"])
def test_public_band_response_cache_matches_sparse_oracle(target_bands: str) -> None:
    from kp.model.pipeline import (
        _compiled_response_band_jacobian,
        _prepare_public_band_response_evaluator,
    )

    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    coefficients0 = np.linspace(0.1, 0.1 * len(basis.channels), len(basis.channels))
    channel_indices = np.asarray([0, len(basis.channels) - 1], dtype=int)
    y = coefficients0[channel_indices] + np.asarray([0.17, -0.23])

    evaluator = _prepare_public_band_response_evaluator(
        basis,
        coefficients0,
        channel_indices,
        points,
    )
    expected_coefficients = coefficients0.copy()
    expected_coefficients[channel_indices] = y
    expected_hamiltonians = basis.hamiltonians(points, expected_coefficients)
    actual_hamiltonians = evaluator.hamiltonians(y)
    np.testing.assert_allclose(
        actual_hamiltonians,
        expected_hamiltonians,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    _eigenvalues, eigenvectors = np.linalg.eigh(actual_hamiltonians)
    selected_masks = np.zeros((len(points), basis.dim), dtype=bool)
    if target_bands == "top":
        selected_masks[:, -1] = True
    else:
        selected_masks[:, 0] = True
    expected_jacobian = _compiled_response_band_jacobian(
        basis,
        channel_indices,
        points,
        eigenvectors,
        selected_masks,
    )
    actual_jacobian = evaluator.band_jacobian(eigenvectors, selected_masks)
    np.testing.assert_allclose(
        actual_jacobian,
        expected_jacobian,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert evaluator.backend == "union_support_vectorized"


def test_public_band_response_cache_compiles_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kp.model.pipeline import _prepare_public_band_response_evaluator

    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    coefficients0 = np.linspace(0.1, 0.1 * len(basis.channels), len(basis.channels))
    channel_indices = np.arange(len(basis.channels), dtype=int)
    original = CompiledResponseBasis._response_design_on_union_support
    calls = 0

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        CompiledResponseBasis,
        "_response_design_on_union_support",
        counted,
    )
    evaluator = _prepare_public_band_response_evaluator(
        basis,
        coefficients0,
        channel_indices,
        points,
    )
    for offset in (0.0, 0.2):
        hamiltonians = evaluator.hamiltonians(coefficients0 + offset)
        _eigenvalues, eigenvectors = np.linalg.eigh(hamiltonians)
        masks = np.ones((len(points), basis.dim), dtype=bool)
        evaluator.band_jacobian(eigenvectors, masks)
    assert calls == 1


def test_public_band_response_cache_uses_low_memory_fallback() -> None:
    from kp.model.pipeline import (
        _compiled_response_band_jacobian,
        _prepare_public_band_response_evaluator,
    )

    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    coefficients0 = np.linspace(0.1, 0.1 * len(basis.channels), len(basis.channels))
    channel_indices = np.arange(len(basis.channels), dtype=int)
    evaluator = _prepare_public_band_response_evaluator(
        basis,
        coefficients0,
        channel_indices,
        points,
        cache_limit_bytes=1,
    )
    assert evaluator.backend == "sparse_low_memory"
    assert evaluator.design_by_k is None
    assert evaluator.cache_bytes == 0

    y = coefficients0 + 0.2
    expected_hamiltonians = basis.hamiltonians(points, y)
    actual_hamiltonians = evaluator.hamiltonians(y)
    np.testing.assert_allclose(actual_hamiltonians, expected_hamiltonians)
    _eigenvalues, eigenvectors = np.linalg.eigh(actual_hamiltonians)
    masks = np.ones((len(points), basis.dim), dtype=bool)
    expected_jacobian = _compiled_response_band_jacobian(
        basis,
        channel_indices,
        points,
        eigenvectors,
        masks,
    )
    np.testing.assert_allclose(
        evaluator.band_jacobian(eigenvectors, masks),
        expected_jacobian,
    )


def test_public_nonlinear_starts_from_exact_linear_coefficients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kp.model.pipeline as pipeline_module

    heff_file = tmp_path / "heff.npy"
    np.save(heff_file, np.zeros((3, 2, 2), dtype=complex))
    fitted = SimpleNamespace(
        fit_solver_channel_indices=(0, 2),
        coefficients=np.asarray([0.25, 9.0, -0.75]),
        with_coefficients=lambda *args, **kwargs: SimpleNamespace(
            coefficients=np.asarray(args[0], dtype=float)
        ),
    )
    basis = SimpleNamespace(
        channel_ids=("a", "fixed", "b"),
        response_scales=np.ones(3),
    )
    model = SimpleNamespace(
        _compiled_response_basis=basis,
        _fitted_response_model=fitted,
    )
    model_config = SimpleNamespace(
        heff_file=heff_file,
        n_orb=(1, 1),
        harmonics_config={},
    )
    moire_config = SimpleNamespace(
        Q_set1=np.zeros((1, 2)),
        Q_set2=np.zeros((1, 2)),
    )
    raw_cfg = {
        "hamiltonian_kpoints": [0, 2],
        "band_kpoints": [0, 1, 2],
        "target_bands": "top",
        "bands": 1,
        "one_sided_weight": 1.5,
        "two_sided_weight": 3.0,
        "band_loss_weight": 2.0,
        "max_steps": 17,
    }
    monkeypatch.setattr(pipeline_module, "_load_kpoints", lambda _config: np.zeros((3, 2)))
    monkeypatch.setattr(
        pipeline_module,
        "_current_heff_support_hamiltonians",
        lambda heff, **kwargs: np.asarray(heff),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_public_hamiltonian_quadratic_residual",
        lambda *args, **kwargs: (np.eye(2), np.zeros(2), {"mode": "test"}),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_public_band_residual_and_jacobian",
        lambda *args, **kwargs: (
            np.asarray([float(np.sum(args[3]))]),
            np.ones((1, 2)),
            (1, 1, 1),
        ),
    )
    prepared_evaluators: list[object] = []

    def fake_prepare_evaluator(*args, **kwargs):
        evaluator = SimpleNamespace(
            backend="union_support_vectorized",
            support=np.arange(3, dtype=int),
            cache_bytes=128,
        )
        prepared_evaluators.append(evaluator)
        return evaluator

    monkeypatch.setattr(
        pipeline_module,
        "_prepare_public_band_response_evaluator",
        fake_prepare_evaluator,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_sync_complete_response_coefficients_to_terms",
        lambda *args, **kwargs: None,
    )
    captured: dict[str, np.ndarray | int] = {}

    progress_messages: list[str] = []

    def fake_least_squares(fun, x0, *, jac, max_nfev):
        captured["x0"] = np.asarray(x0).copy()
        captured["max_nfev"] = int(max_nfev)
        assert fun(np.asarray(x0)).shape == (3,)
        assert jac(np.asarray(x0)).shape == (3, 2)
        assert fun(np.asarray(x0) + 0.1).shape == (3,)
        return SimpleNamespace(
            x=np.asarray(x0) + 0.1,
            nfev=2,
            njev=1,
            cost=0.5,
            status=1,
            message="test",
        )

    monkeypatch.setattr(pipeline_module.scipy.optimize, "least_squares", fake_least_squares)

    report = _refine_public_nonlinear_complete_response(
        moire_config,
        model_config,
        model,
        raw_cfg,
        progress_callback=lambda message, **_kwargs: progress_messages.append(message),
    )

    np.testing.assert_allclose(captured["x0"], [0.25, -0.75])
    assert captured["max_nfev"] == 17
    assert report["initial_coefficients"] == [0.25, -0.75]
    assert len(prepared_evaluators) == 1
    assert report["band_response_backend"] == "union_support_vectorized"
    assert report["band_response_support_entries"] == 3
    assert report["band_response_cache_bytes"] == 128
    assert progress_messages[0].startswith(
        "nonlinear band response backend: union_support_vectorized, "
        "support=3, cache=0.0 MiB, "
    )
    assert progress_messages[1:] == [
        "nonlinear loss evaluation 1/17",
        "nonlinear loss evaluation 2/17",
    ]


def test_target_spectral_two_sided_projector_matches_independent_dense_oracle() -> None:
    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    target = np.asarray(
        [
            [[0.4, 0.3 + 0.2j], [0.3 - 0.2j, 1.5]],
            [[-0.2, -0.15 + 0.35j], [-0.15 - 0.35j, 0.9]],
        ],
        dtype=complex,
    )
    ridge = 0.17
    projector_weight = 13.0
    spec = response_basis_module.TargetSpectralWeightingSpec(
        mode="target_spectral_linear",
        band_edge="top",
        window_mode="fixed_count_degeneracy_safe",
        n_bands=1,
        floor=0.2,
        alpha=0.7,
        two_sided_projector_weight=projector_weight,
    )

    fitted = basis.fit(
        points,
        target,
        fit_indices=[3, 9],
        regularization=ridge,
        band_window=[0, 2],
        spectral_weighting=spec,
    )
    expected = _dense_target_spectral_ridge_oracle(
        basis,
        points,
        target,
        floor=0.2,
        alpha=0.7,
        ridge=ridge,
        two_sided_projector_weight=projector_weight,
    )

    np.testing.assert_allclose(fitted.coefficients, expected, rtol=2.0e-11, atol=2.0e-11)
    assert fitted.fit_objective["two_sided_projector"] == {
        "enabled": True,
        "weight": projector_weight,
        "projector_source": "target_spectral_window_v1",
    }


def test_target_spectral_linear_fit_matches_independent_dense_sqrt_oracle() -> None:
    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    target = np.asarray(
        [
            [[0.4, 0.3 + 0.2j], [0.3 - 0.2j, 1.5]],
            [[-0.2, -0.15 + 0.35j], [-0.15 - 0.35j, 0.9]],
        ],
        dtype=complex,
    )
    ridge = 0.17
    spec = response_basis_module.TargetSpectralWeightingSpec(
        mode="target_spectral_linear",
        band_edge="top",
        window_mode="fixed_count_degeneracy_safe",
        n_bands=1,
        floor=0.2,
        alpha=0.7,
    )

    fitted = basis.fit(
        points,
        target,
        fit_indices=[3, 9],
        regularization=ridge,
        band_window=[0, 2],
        spectral_weighting=spec,
    )
    expected = _dense_target_spectral_ridge_oracle(
        basis,
        points,
        target,
        floor=0.2,
        alpha=0.7,
        ridge=ridge,
    )

    np.testing.assert_allclose(fitted.coefficients, expected, rtol=2.0e-11, atol=2.0e-11)
    assert fitted.fit_objective["mode"] == "target_spectral_linear"
    assert fitted.fit_objective["normalization"] == "global_mean_trace_per_dimension_v1"


def test_target_spectral_linear_uses_full_target_for_one_sided_off_union_rows() -> None:
    basis = _spectral_fit_toy_basis(only_e11=True)
    points = np.asarray([[0.0, 0.0]])
    target = np.asarray([[[0.6, 0.8], [0.8, -0.4]]], dtype=complex)
    ridge = 0.11
    spec = response_basis_module.TargetSpectralWeightingSpec(
        mode="target_spectral_linear",
        band_edge="top",
        window_mode="fixed_count_degeneracy_safe",
        n_bands=1,
        floor=0.1,
        alpha=1.0,
    )

    fitted = basis.fit(
        points,
        target,
        fit_indices=[0],
        regularization=ridge,
        band_window=[0, 2],
        spectral_weighting=spec,
    )
    expected = _dense_target_spectral_ridge_oracle(
        basis,
        points,
        target,
        floor=0.1,
        alpha=1.0,
        ridge=ridge,
    )

    np.testing.assert_allclose(fitted.coefficients, expected, rtol=2.0e-11, atol=2.0e-11)


@pytest.mark.parametrize(
    ("floor", "alpha"),
    [(0.2, 0.0), (1.0, 7.0)],
)
def test_target_spectral_equal_limits_match_equal_matrix_fit(floor: float, alpha: float) -> None:
    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0]])
    target = np.asarray([[[0.2, 0.4j], [-0.4j, 1.1]]], dtype=complex)
    baseline = basis.fit(
        points,
        target,
        fit_indices=[0],
        regularization=0.03,
        band_window=[0, 2],
    )
    weighted = basis.fit(
        points,
        target,
        fit_indices=[0],
        regularization=0.03,
        band_window=[0, 2],
        spectral_weighting=response_basis_module.TargetSpectralWeightingSpec(
            mode="target_spectral_linear",
            band_edge="top",
            window_mode="fixed_count_degeneracy_safe",
            n_bands=1,
            floor=floor,
            alpha=alpha,
        ),
    )

    np.testing.assert_allclose(weighted.coefficients, baseline.coefficients, atol=2.0e-12)
    np.testing.assert_allclose(
        weighted.fit_design_singular_values,
        baseline.fit_design_singular_values,
        atol=2.0e-12,
    )


def test_target_spectral_fixed_count_completes_degenerate_boundary() -> None:
    target = np.asarray([np.diag([0.0, 1.0, 1.0])], dtype=complex)
    weighting = response_basis_module.build_target_spectral_weighting(
        target,
        response_basis_module.TargetSpectralWeightingSpec(
            mode="target_spectral_linear",
            band_edge="top",
            window_mode="fixed_count_degeneracy_safe",
            n_bands=1,
            floor=0.1,
            alpha=1.0,
            degeneracy_atol_ev=1.0e-10,
        ),
    )

    assert weighting.selected_counts == (2,)
    assert weighting.spectral_weights[0, 1] == pytest.approx(weighting.spectral_weights[0, 2])
    assert weighting.spectral_weights[0, 0] < weighting.spectral_weights[0, 1]


@pytest.mark.parametrize(
    ("band_edge", "expected_count"),
    [("top", 1), ("bottom", 3)],
)
def test_target_spectral_auto_gap_selects_largest_allowed_boundary_gap(
    band_edge: str,
    expected_count: int,
) -> None:
    target = np.asarray([np.diag([0.0, 1.0, 1.2, 3.0])], dtype=complex)
    weighting = response_basis_module.build_target_spectral_weighting(
        target,
        response_basis_module.TargetSpectralWeightingSpec(
            mode="target_spectral_linear",
            band_edge=band_edge,
            window_mode="auto_gap",
            min_bands=1,
            max_bands=3,
            floor=0.1,
            alpha=1.0,
        ),
    )

    assert weighting.selected_counts == (expected_count,)


def test_target_spectral_objective_changes_fit_hash_not_basis_hash_and_round_trips() -> None:
    basis = _spectral_fit_toy_basis()
    points = np.asarray([[0.0, 0.0]])
    target = np.asarray([[[0.2, 0.3], [0.3, 1.0]]], dtype=complex)
    original_basis_hash = basis.basis_hash
    fit_a = basis.fit(
        points,
        target,
        fit_indices=[0],
        regularization=0.01,
        band_window=[0, 2],
        spectral_weighting=response_basis_module.TargetSpectralWeightingSpec(
            mode="target_spectral_linear",
            band_edge="top",
            window_mode="fixed_count_degeneracy_safe",
            n_bands=1,
            floor=0.1,
            alpha=1.0,
        ),
    )
    fit_b = basis.fit(
        points,
        target,
        fit_indices=[0],
        regularization=0.01,
        band_window=[0, 2],
        spectral_weighting=response_basis_module.TargetSpectralWeightingSpec(
            mode="target_spectral_linear",
            band_edge="top",
            window_mode="fixed_count_degeneracy_safe",
            n_bands=1,
            floor=0.1,
            alpha=1.0,
            two_sided_projector_weight=3.0,
        ),
    )

    assert basis.basis_hash == original_basis_hash
    assert fit_a.basis_hash == fit_b.basis_hash == original_basis_hash
    assert fit_a.fit_hash != fit_b.fit_hash
    runtime = CompiledResponseRuntime(basis, fit_b)
    restored = CompiledResponseRuntime.from_frozen_arrays(runtime.export_arrays())
    assert restored.fitted.fit_objective == fit_b.fit_objective
    assert restored.fitted.fit_hash == fit_b.fit_hash
    np.testing.assert_allclose(restored.fitted.coefficients, fit_b.coefficients, atol=0.0)


def test_target_spectral_with_coefficients_preserves_objective_artifact() -> None:
    basis = _spectral_fit_toy_basis()
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([[[0.2, 0.3], [0.3, 1.0]]], dtype=complex),
        fit_indices=[0],
        regularization=0.01,
        band_window=[0, 2],
        spectral_weighting=response_basis_module.TargetSpectralWeightingSpec(
            mode="target_spectral_linear",
            band_edge="top",
            window_mode="fixed_count_degeneracy_safe",
            n_bands=1,
            floor=0.1,
            alpha=1.0,
        ),
    )

    updated = fitted.with_coefficients(
        fitted.coefficients,
        channel_ids=basis.channel_ids,
        response_scales=basis.response_scales,
        provenance={"test": "preserve-objective"},
    )

    assert updated.fit_objective == fitted.fit_objective


def test_compute_coefficients_translates_configured_target_spectral_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeBasis:
        dim = 2
        channels: tuple[object, ...] = ()
        candidate_artifact = {"candidate_channel_count": 0}
        basis_hash = "fake-basis"

        def artifact(self) -> dict[str, object]:
            return {"basis_hash": self.basis_hash}

        def fit(self, *args: object, **kwargs: object) -> object:
            captured["spectral_weighting"] = kwargs["spectral_weighting"]
            return SimpleNamespace(
                coefficients=np.zeros(0, dtype=float),
                fit_hash="fake-fit",
                fit_solver_channel_ids=(),
                fit_selected_channel_ids=(),
                nonzero_channel_ids=(),
                artifact=lambda: {"fit_hash": "fake-fit"},
            )

    monkeypatch.setattr(
        response_basis_module,
        "compile_model_response_basis",
        lambda *args, **kwargs: FakeBasis(),
    )
    config = MoireConfig(
        Q_set1=np.zeros((1, 2), dtype=float),
        Q_set2=np.zeros((1, 2), dtype=float),
        bM1=np.eye(2, dtype=float),
        bM2=np.eye(2, dtype=float),
        kpoints_fit=np.zeros((1, 2), dtype=float),
        heff=np.asarray([np.diag([0.0, 1.0])], dtype=complex),
        response_semantics="complete_linear_v2",
        response_fit_objective={
            "mode": "target_spectral_linear",
            "target_reference": "current_heff_support_mask",
            "edge": "top",
            "window": {
                "mode": "fixed_count_degeneracy_safe",
                "bands": 2,
                "degeneracy_tol_mev": 0.5,
            },
            "floor": 0.05,
            "alpha": 1.0,
            "normalization": "mean_trace_per_dimension_v1",
            "two_sided_projector": {"enabled": True, "weight": 300.0},
        },
    )

    compute_coefficients(config, SimpleNamespace(candidate_terms=[], terms={}))

    assert captured["spectral_weighting"] == {
        "mode": "target_spectral_linear",
        "band_edge": "top",
        "window_mode": "fixed_count_degeneracy_safe",
        "n_bands": 2,
        "floor": 0.05,
        "alpha": 1.0,
        "degeneracy_atol_ev": 0.0005,
        "degeneracy_rtol": 0.0,
        "normalization": "global_mean_trace_per_dimension_v1",
        "two_sided_projector_weight": 300.0,
    }


def test_fit_builds_design_only_on_union_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E11", {(0, 0): e11}, "diag", {})],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )

    def fail_dense_tensor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit constructed the full dense response tensor")

    monkeypatch.setattr(CompiledResponseBasis, "response_tensor", fail_dense_tensor)
    # The off-support E22 target contributes a coefficient-independent constant
    # to the full Frobenius loss and therefore must not alter the E11 fit.
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([3.0, 1.0e6])], dtype=complex),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )

    np.testing.assert_allclose(fitted.coefficients, [3.0], atol=1.0e-12)


def test_fit_design_error_maps_constant_complex_channel_without_extra_sqrt_two() -> None:
    coordinate = _toy_coordinate(max_degree=0)
    e12 = sparse.csr_matrix(np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E12", {(0, 0): e12}, "offdiag", {})],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(
                channel,
                propagated_error_bound=1.0e-2 * channel.response_scale,
                error_bound_components={"certified_test_bound": 1.0e-2 * channel.response_scale},
                classification="confirmed_nonzero" if channel.component == "imag" else "ambiguous",
            )
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    points = np.zeros((4, 2), dtype=float)
    imag_channel = next(channel for channel in basis.channels if channel.component == "imag")
    target = np.repeat(imag_channel.coefficients[(0, 0)].toarray()[None, :, :], 4, axis=0)

    fitted = basis.fit(
        points,
        target,
        fit_indices=[0, 1, 2, 3],
        regularization=0.0,
        band_window=[0, 2],
    )

    # ||[1,1,1,1]||_2 * (e/s) = 2 * 0.01.  Real-stacking is
    # an isometry, so a complex Hermitian response adds no sqrt(2).
    assert fitted.fit_design_propagated_error_bound == pytest.approx(2.0e-2)
    assert fitted.fit_design_certified_rank == 1


def test_normalized_low_energy_fit_scales_design_error_with_full_matrix_rows() -> None:
    """The error certificate must use the same 1/dim scaling as the objective."""

    coordinate = _toy_coordinate(max_degree=0)
    e12 = sparse.csr_matrix(np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("E12", {(0, 0): e12}, "offdiag", {})],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(
                channel,
                propagated_error_bound=1.0e-2 * channel.response_scale,
                error_bound_components={"certified_test_bound": 1.0e-2 * channel.response_scale},
                classification="confirmed_nonzero" if channel.component == "imag" else "ambiguous",
            )
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    points = np.zeros((4, 2), dtype=float)
    imag_channel = next(channel for channel in basis.channels if channel.component == "imag")
    target = np.repeat(imag_channel.coefficients[(0, 0)].toarray()[None, :, :], 4, axis=0)

    fitted = basis.fit(
        points,
        target,
        fit_indices=[0, 1, 2, 3],
        regularization=0.0,
        band_window=[0, 2],
        spectral_weighting={
            "mode": "normalized_low_energy_linear_v1",
            "band_edge": "top",
            "window_mode": "fixed_count_degeneracy_safe",
            "n_bands": 1,
            "one_sided_weight": 0.0,
            "two_sided_weight": 0.0,
        },
    )

    # Four identical k points give ||[1,1,1,1]||_2 = 2.  The public
    # full-matrix objective divides both the response and its error by dim=2.
    assert fitted.fit_design_propagated_error_bound == pytest.approx(1.0e-2)
    assert fitted.fit_design_certified_rank == 1


def test_fit_design_error_uses_global_monomial_space_without_support_certificate() -> None:
    coordinate = _toy_coordinate(max_degree=1)
    e11 = sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))
    candidates = compile_candidate_responses(
        [RawPolynomialSeed("constant-only", {(0, 0): e11}, "diag", {})],
        coordinate=coordinate,
        group=identity_finite_group(2),
    )
    candidates = dataclasses.replace(
        candidates,
        channels=tuple(
            dataclasses.replace(
                channel,
                propagated_error_bound=1.0e-2 * channel.response_scale,
                error_bound_components={"certified_test_bound": 1.0e-2 * channel.response_scale},
                classification="confirmed_nonzero",
            )
            if channel.component == "real"
            else channel
            for channel in candidates.channels
        ),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=False,
    )
    points = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    target = np.repeat(np.diag([1.0, 0.0])[None, :, :], 2, axis=0)
    fitted = basis.fit(
        points,
        target,
        fit_indices=[0, 1],
        regularization=0.0,
        band_window=[0, 2],
    )

    u = coordinate.to_dimensionless(points)
    w = u[:, 0] + 1j * u[:, 1]
    global_evaluation = np.column_stack(
        [(w ** r) * (np.conjugate(w) ** s) for r, s in coordinate.monomials]
    )
    expected = np.linalg.norm(global_evaluation, ord=2) * 1.0e-2
    assert fitted.fit_design_propagated_error_bound == pytest.approx(expected)
    assert expected > np.sqrt(2.0) * 1.0e-2


def test_complete_refinement_uses_retained_basis_responses_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _complete_onsite_config()
    kpoints = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    target = np.asarray(
        [
            [[1.0, 0.25 + 0.1j], [0.25 - 0.1j, 2.0]],
            [[1.0, 0.25 + 0.1j], [0.25 - 0.1j, 2.0]],
        ],
        dtype=np.complex128,
    )
    config.kpoints_fit = kpoints[:1]
    config.kpoints = kpoints
    config.heff = target[:1]
    config.coeff_tol = 0.0
    model, _diagnostics = compute_coefficients(config, build_model(config))
    heff_path = tmp_path / "heff.npy"
    eig_path = tmp_path / "heff_eig.npy"
    np.save(heff_path, target)
    np.save(eig_path, np.linalg.eigvalsh(target))
    model_config = SimpleNamespace(
        band_refinement_config={
            "enabled": True,
            "indices": [0, 1],
            "components": ["real", "imag"],
            "variable_tags": ["Onsite"],
            "top_bands": 2,
            "max_nfev": 2,
            "band_sigma_mev": 1.0,
            "matrix_weight": 1.0,
            "matrix_sigma_mev": 1.0,
            "coefficient_weight": 0.0,
        },
        band_indices=None,
        fit_indices=[0],
        heff_eig_file=eig_path,
        heff_file=heff_path,
        band_plot_config={"align": "none"},
        band_slice=None,
        n_orb=(2, 0),
        harmonics_config={},
        fit_selection_metadata={"mode": "manual"},
        response_semantics="complete_linear_v2",
        raw={"model": {"target_bands": "top"}},
    )

    def fail_legacy_prepare(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("complete refinement entered legacy term response path")

    monkeypatch.setattr("kp.model.pipeline._prepare_band_state", fail_legacy_prepare)
    support_targets: list[np.ndarray] = []

    def capture_support_target(values: np.ndarray, **_kwargs: object) -> np.ndarray:
        support_targets.append(np.asarray(values))
        return np.asarray(values)

    monkeypatch.setattr(
        "kp.model.pipeline._current_heff_support_hamiltonians",
        capture_support_target,
    )
    report = refine_band_coefficients(config, model_config, model)

    assert report["enabled"] is True
    assert report["reference"] == "current_heff_support_mask"
    assert support_targets
    assert report["response_basis"]["mode"] == "compiled_response_basis_v2"
    assert report["fit_kpoints"]["count"] == 2
    hamiltonians = _model_hamiltonians_for_kpoints(config, model, kpoints)
    np.testing.assert_allclose(hamiltonians, target, atol=2.0e-9)


def test_complete_pruning_uses_response_amplitude_and_updates_fitted_runtime() -> None:
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "unit_E11",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))},
                "diag",
                {"term_index": 0, "tag": "Onsite"},
            ),
            RawPolynomialSeed(
                "large_E22",
                {(0, 0): sparse.csr_matrix(np.diag([0.0, 1000.0]).astype(complex))},
                "diag",
                {"term_index": 1, "tag": "Onsite"},
            ),
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([0.01, 0.1])], dtype=complex),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
    )
    terms = {
        "unit": SimpleNamespace(active=True, r_value_real=0.01, r_value_imag=0.0),
        "large": SimpleNamespace(active=True, r_value_real=0.0001, r_value_imag=0.0),
    }
    model = SimpleNamespace(
        terms=terms,
        _compiled_response_basis=basis,
        _fitted_response_model=fitted,
    )

    report = _prune_small_coefficients(model, 0.05)

    assert report["normalization"] == "response_amplitude_abs_c_times_s_v1"
    assert report["dropped"] == 1
    assert report["kept"] == 1
    refined = model._fitted_response_model
    np.testing.assert_allclose(refined.coefficients, [0.0, 0.0001], atol=1.0e-14)
    np.testing.assert_allclose(
        CompiledResponseRuntime(basis, refined).hamiltonian([0.0, 0.0]),
        np.diag([0.0, 0.1]),
        atol=1.0e-12,
    )


def test_refined_fit_zeros_coefficients_below_versioned_response_tolerance() -> None:
    candidates = compile_candidate_responses(
        [
            RawPolynomialSeed(
                "E11",
                {(0, 0): sparse.csr_matrix(np.diag([1.0, 0.0]).astype(complex))},
                "diag",
                {},
            )
        ],
        coordinate=_toy_coordinate(),
        group=identity_finite_group(2),
    )
    basis = CompiledResponseBasis.from_candidates(
        candidates,
        identity_payload=_identity_payload(2),
        reduce=True,
    )
    fitted = basis.fit(
        [[0.0, 0.0]],
        np.asarray([np.diag([1.0, 0.0])], dtype=complex),
        fit_indices=[0],
        regularization=0.0,
        band_window=[0, 2],
        coefficient_tolerance=0.05,
    )

    refined = fitted.with_coefficients(
        [0.01],
        channel_ids=basis.channel_ids,
        response_scales=basis.response_scales,
        provenance={"operation": "test"},
    )

    np.testing.assert_array_equal(refined.coefficients, [0.0])
    assert refined.nonzero_channel_ids == ()


def test_complete_response_coefficient_sync_uses_compiler_candidate_order() -> None:
    first_authored = SimpleNamespace(
        active=True,
        r_value_real=91.0,
        r_value_imag=92.0,
    )
    last_duplicate_kept_by_legacy_dict = SimpleNamespace(
        active=True,
        r_value_real=81.0,
        r_value_imag=82.0,
    )
    basis = SimpleNamespace(
        channels=(
            SimpleNamespace(
                channel_id="term:0:real",
                component="real",
                metadata={"term_index": 0},
            ),
        )
    )
    model = SimpleNamespace(
        candidate_terms=[first_authored, last_duplicate_kept_by_legacy_dict],
        terms={"deduplicated-key": last_duplicate_kept_by_legacy_dict},
        _compiled_response_basis=basis,
    )

    _sync_complete_response_coefficients_to_terms(model, np.asarray([3.5]))

    assert first_authored.active is True
    assert first_authored.r_value_real == pytest.approx(3.5)
    assert first_authored.r_value_imag == pytest.approx(0.0)
    assert last_duplicate_kept_by_legacy_dict.active is False
    assert last_duplicate_kept_by_legacy_dict.r_value_real == pytest.approx(0.0)
    assert last_duplicate_kept_by_legacy_dict.r_value_imag == pytest.approx(0.0)
