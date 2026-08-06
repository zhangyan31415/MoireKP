from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse

import kp.model.response_basis as response_basis
import kp.model.response_basis_graded as graded
import kp.model.response_basis_symmetry_first as symbolic
from kp.model.response_basis import (
    FiniteGroupGenerator,
    PolynomialCoordinateBasis,
    RawPolynomialSeed,
    build_finite_group,
    identity_finite_group,
    raw_polynomial_seed_from_term_key,
)
from kp.model.response_basis_graded import _certified_component_elimination


def test_connected_column_components_follow_shared_sparse_rows() -> None:
    value = sparse.csc_matrix(
        np.asarray(
            (
                (1.0, 0.0, 2.0, 0.0, 0.0),
                (0.0, 0.0, 3.0, 0.0, 4.0),
                (0.0, 5.0, 0.0, 0.0, 0.0),
            )
        )
    )

    components = graded._connected_column_components(
        value,
        np.asarray((0, 1, 2, 4), dtype=np.int64),
    )

    assert components == ((0, 2, 4), (1,))


def test_deferred_degree_insertion_preserves_certified_leakage_charge() -> None:
    leakage = 1.0e-12
    projected = sparse.csc_matrix(
        np.asarray(
            (
                (1.0, leakage, leakage),
                (0.0, 1.0, 1.0),
            )
        )
    )
    row_degrees = np.asarray((1, 0), dtype=np.int64)
    nominal_degrees = np.asarray((1, 0, 0), dtype=np.int64)
    error_bounds = np.asarray((0.0, 2.0 * leakage, 2.0 * leakage))
    reference = graded.select_filtered_independent_columns(
        projected,
        row_degrees=row_degrees,
        nominal_degrees=nominal_degrees,
        absolute_error_bounds=error_bounds,
    )

    actual, owner_map, artifact = (
        graded._select_filtered_independent_column_blocks(
            (projected,),
            owner_index_blocks=((0, 1, 2),),
            nominal_degree_blocks=(nominal_degrees,),
            error_bound_blocks=(error_bounds,),
            row_degrees=row_degrees,
        )
    )

    assert owner_map == (0, 1, 2)
    assert actual.selected_owner_indices == reference.selected_owner_indices
    np.testing.assert_allclose(
        actual.residual_error_bounds,
        reference.residual_error_bounds,
        rtol=1.0e-14,
        atol=1.0e-30,
    )
    assert artifact["strategy"] == "deferred_owner_degree_slices_v2"


def test_component_elimination_is_invariant_to_degree_block_column_scale() -> None:
    """RRQR and dependency solves must use the same column normalization."""

    epsilon = 2.0**-80
    block = sparse.csc_matrix(
        np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, epsilon, 1.0],
            ],
            dtype=float,
        )
    )

    pivots, nonpivots, coefficients, _errors, proof = (
        _certified_component_elimination(
            block,
            (0, 1, 2),
            error_bounds=np.zeros(3, dtype=float),
        )
    )

    assert len(pivots) == 2
    assert len(nonpivots) == 1
    reconstructed = block[:, list(pivots)] @ coefficients
    np.testing.assert_allclose(
        reconstructed,
        block[:, list(nonpivots)].toarray(),
        rtol=1.0e-13,
        atol=1.0e-30,
    )
    assert proof["maximum_solve_roundoff_bound"] < 1.0e-10


def _seed(
    seed_id: str,
    *,
    monomial: tuple[int, int],
    top: np.ndarray,
    family: str,
    term_index: int,
    lower: dict[tuple[int, int], np.ndarray] | None = None,
) -> RawPolynomialSeed:
    coefficients = {
        key: sparse.csr_matrix(value, dtype=np.complex128)
        for key, value in (lower or {}).items()
    }
    coefficients[monomial] = sparse.csr_matrix(top, dtype=np.complex128)
    return RawPolynomialSeed(
        seed_id=seed_id,
        coefficients=coefficients,
        support_component=family,
        metadata={
            "term_index": term_index,
            "term_name": family,
            "tag": family,
            "term_space_policy": "complete",
            "term_key": {
                "Mz": monomial[0],
                "Mz_star": monomial[1],
                "layer_from": 1,
                "layer_to": 1,
                "orbital_from": 1,
                "orbital_to": 1,
                "p": [0.0, 0.0],
            },
        },
    )


def _candidate_vectors(candidate) -> np.ndarray:
    return sparse.hstack(
        [
            response_basis._channel_sparse_vector(
                channel,
                candidate.coordinate,
                candidate.dim,
            )
            for channel in candidate.channels
        ],
        format="csc",
    ).toarray()


def _d3_cubic_case():
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=3,
    )
    angle = 2.0 * np.pi / 3.0
    c3_matrix = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    reflection = np.asarray(((1.0, 0.0), (0.0, -1.0)))
    sigma_x = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128)
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C3",
                antiunitary=False,
                canonical_k_map=((0, -1), (1, -1)),
                q_permutation=(0,),
                sector_permutation=(0,),
                k_forward=tuple(
                    tuple(float(value) for value in row) for row in c3_matrix
                ),
                internal_u=np.eye(2, dtype=np.complex128),
            ),
            FiniteGroupGenerator(
                name="C2",
                antiunitary=False,
                canonical_k_map=((0, 1), (1, 0)),
                q_permutation=(0,),
                sector_permutation=(0,),
                k_forward=tuple(
                    tuple(float(value) for value in row) for row in reflection
                ),
                internal_u=sigma_x,
            ),
        ]
    )
    sigma_z = np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=np.complex128)
    seeds = tuple(
        _seed(
            f"degree-{degree}-{r}",
            monomial=(r, degree - r),
            top=sigma_z,
            family="intra",
            term_index=sum(range(1, degree + 1)) + r,
        )
        for degree in range(4)
        for r in range(degree + 1)
    )
    internal_actions = {
        tuple(str(value) for value in element.canonical_word): sparse.csr_matrix(
            element.internal_u
        )
        for element in group.elements
    }
    artifact = {"maximum_action_error_bound": 0.0, "status": "test-certified"}
    return coordinate, seeds, group, internal_actions, artifact


def test_each_production_degree_is_projected_independently_and_cubic_is_new(
    monkeypatch,
) -> None:
    coordinate, seeds, group, internal_actions, artifact = _d3_cubic_case()
    projected_batch_sizes: list[int] = []
    homogeneous_degrees: set[int] = set()
    original_batch = graded._symbolic_reynolds_batch
    original_homogeneous = graded.compile_homogeneous_momentum_action

    def record_batch(batch, atoms, **kwargs):
        projected_batch_sizes.append(len(batch))
        return original_batch(batch, atoms, **kwargs)

    def record_homogeneous(element, *, coordinate, degree):
        homogeneous_degrees.add(int(degree))
        return original_homogeneous(element, coordinate=coordinate, degree=degree)

    monkeypatch.setattr(graded, "_symbolic_reynolds_batch", record_batch)
    monkeypatch.setattr(
        graded, "compile_homogeneous_momentum_action", record_homogeneous
    )
    compiled = graded.compile_graded_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={},
        internal_actions_by_word=internal_actions,
        factorized_group_artifact=artifact,
    )

    assert len(compiled.channels) == 1
    assert compiled.channels[0].metadata["term_key"]["Mz"] + compiled.channels[
        0
    ].metadata["term_key"]["Mz_star"] == 3
    assert homogeneous_degrees == {0, 1, 2, 3}
    assert max(projected_batch_sizes) <= 4
    assert compiled.adjoint_artifact["structural_support_count"] == 1
    assert compiled.adjoint_artifact["structural_support_count"] < len(seeds)
    assert compiled.adjoint_artifact["global_filtered_owner_certificate"][
        "strategy"
    ] == "deferred_owner_degree_slices_v2"


def test_family_duplicates_merge_but_keep_provenance_and_residual_tail() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=1,
    )
    e00 = np.asarray(((1.0, 0.0), (0.0, 0.0)), dtype=np.complex128)
    e11 = np.asarray(((0.0, 0.0), (0.0, 1.0)), dtype=np.complex128)
    onsite = _seed(
        "onsite-d0",
        monomial=(0, 0),
        top=e00,
        family="onsite",
        term_index=0,
    )
    intra_duplicate = _seed(
        "intra-d0",
        monomial=(0, 0),
        top=e00,
        family="intra",
        term_index=1,
    )
    first_tail = _seed(
        "kinetic-d1",
        monomial=(1, 0),
        top=e00,
        lower={(0, 0): e00},
        family="kinetic",
        term_index=2,
    )
    first_tail_duplicate = _seed(
        "intra-d1-duplicate",
        monomial=(1, 0),
        top=e00,
        lower={(0, 0): e00},
        family="intra",
        term_index=3,
    )
    independent_tail = _seed(
        "intra-d1-independent-tail",
        monomial=(1, 0),
        top=e00,
        lower={(0, 0): e11},
        family="intra",
        term_index=4,
    )
    seeds = (
        onsite,
        intra_duplicate,
        first_tail,
        first_tail_duplicate,
        independent_tail,
    )
    group = identity_finite_group(2, q_size=1, sector_size=1)
    internal_actions = {(): sparse.eye(2, format="csr", dtype=np.complex128)}
    artifact = {"maximum_action_error_bound": 0.0, "status": "test-certified"}
    oracle = symbolic.compile_symbolic_atom_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={},
        internal_actions_by_word=internal_actions,
        factorized_group_artifact=artifact,
    )
    compiled = graded.compile_graded_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={},
        internal_actions_by_word=internal_actions,
        factorized_group_artifact=artifact,
    )

    compiled_vectors = _candidate_vectors(compiled)
    oracle_vectors = _candidate_vectors(oracle)
    assert len(compiled.channels) == len(oracle.channels)
    np.testing.assert_allclose(
        compiled_vectors @ np.linalg.pinv(compiled_vectors),
        oracle_vectors @ np.linalg.pinv(oracle_vectors),
        atol=1.0e-10,
    )
    assert compiled.adjoint_artifact["authored_ordered_seed_count"] == 5
    assert compiled.adjoint_artifact["physical_seed_count"] == 3
    assert compiled.adjoint_artifact["structural_support_count"] == 1
    provenance = [
        source
        for channel in compiled.channels
        for source in channel.metadata["graded_source_provenance"]
    ]
    assert {source["family"] for source in provenance} >= {
        "onsite",
        "intra",
        "kinetic",
    }
    degree_one_proofs = [
        proof
        for proof in compiled.adjoint_artifact["rank_proofs"]
        if proof["degree"] == 1
    ]
    degree_zero_proofs = [
        proof
        for proof in compiled.adjoint_artifact["rank_proofs"]
        if proof["degree"] == 0
    ]
    assert sum(len(item["selected_owner_indices"]) for item in degree_one_proofs) > 0
    assert sum(len(item["selected_owner_indices"]) for item in degree_zero_proofs) > 0
    assert compiled.adjoint_artifact["global_filtered_owner_certificate"][
        "strategy"
    ] == "deferred_owner_degree_slices_v2"


def test_single_sign_harmonic_generates_virtual_adjoint_closure_without_fallback(
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
        seed_id="single-sign-p",
        Q_set1=q_vectors,
        Q_set2=np.empty((0, 2)),
        n_orb1=1,
        n_orb2=0,
        coordinate=coordinate,
        support_component="intra",
        metadata={"term_index": 0, "term_space_policy": "complete"},
    )
    group = identity_finite_group(2, q_size=2, sector_size=2)
    compiled = graded.compile_graded_candidate_group(
        [seed],
        coordinate=coordinate,
        group=group,
        factorized_actions={},
    )

    assert compiled.adjoint_artifact["certification"] == "graded_filtered_reynolds_v1"
    assert compiled.adjoint_artifact["structural_support_count"] == 1
    assert compiled.adjoint_artifact["closed_structural_direction_count"] == 2
    assert any(
        support_id.startswith("virtual:")
        for orbit in compiled.adjoint_artifact["structural_orbits"]
        for support_id in orbit["support_ids"]
    )


def test_high_degree_only_support_records_its_actual_structural_probe_degree() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=3,
    )
    seed = _seed(
        "cubic-only-support",
        monomial=(3, 0),
        top=np.asarray(((1.0,),), dtype=np.complex128),
        family="inter",
        term_index=0,
    )
    compiled = graded.compile_graded_candidate_group(
        [seed],
        coordinate=coordinate,
        group=identity_finite_group(1, q_size=1, sector_size=1),
        factorized_actions={},
    )

    assert compiled.adjoint_artifact["structural_probe_seed_indices"] == [0]
    assert compiled.adjoint_artifact["structural_probe_degrees"] == [3]
    assert compiled.adjoint_artifact["structural_probe_max_degree"] == 3


def test_structural_orbits_do_not_drop_tiny_nonzero_orbital_gauge_mixing() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=0,
    )
    epsilon = 1.0e-11
    c2_internal = np.asarray(
        ((1.0, epsilon), (epsilon, -1.0)), dtype=np.complex128
    )
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2-small-gauge-mixing",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(0,),
                sector_permutation=(0,),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=c2_internal,
            )
        ]
    )
    e00 = np.asarray(((1.0, 0.0), (0.0, 0.0)), dtype=np.complex128)
    e01 = np.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=np.complex128)
    seeds = (
        _seed(
            "diagonal-support",
            monomial=(0, 0),
            top=e00,
            family="intra",
            term_index=0,
        ),
        _seed(
            "offdiagonal-support",
            monomial=(0, 0),
            top=e01,
            family="intra",
            term_index=1,
        ),
    )
    internal_actions = {
        tuple(str(value) for value in element.canonical_word): sparse.csr_matrix(
            element.internal_u
        )
        for element in group.elements
    }

    plan = graded.build_structural_compilation_plan(
        seeds,
        coordinate=coordinate,
        group=group,
        internal_actions_by_word=internal_actions,
    )

    assert len(plan.structural_orbits) == 1
    assert plan.structural_orbits[0].representative_seed_indices == (0, 1)


def test_global_owner_certificate_removes_shared_tail_across_top_orbits() -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=1,
    )
    e00 = np.asarray(((1.0, 0.0), (0.0, 0.0)), dtype=np.complex128)
    e11 = np.asarray(((0.0, 0.0), (0.0, 1.0)), dtype=np.complex128)
    shared_tail = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128)
    seeds = (
        _seed(
            "orbit-0-head",
            monomial=(1, 0),
            top=e00,
            family="intra",
            term_index=0,
        ),
        _seed(
            "orbit-0-tail",
            monomial=(1, 0),
            top=e00,
            lower={(0, 0): shared_tail},
            family="intra",
            term_index=1,
        ),
        _seed(
            "orbit-1-head",
            monomial=(1, 0),
            top=e11,
            family="intra",
            term_index=2,
        ),
        _seed(
            "orbit-1-tail",
            monomial=(1, 0),
            top=e11,
            lower={(0, 0): shared_tail},
            family="intra",
            term_index=3,
        ),
    )
    compiled = graded.compile_graded_candidate_group(
        seeds,
        coordinate=coordinate,
        group=identity_finite_group(2, q_size=1, sector_size=1),
        factorized_actions={},
    )
    vectors = _candidate_vectors(compiled)

    assert compiled.adjoint_artifact["structural_orbit_count"] == 2
    assert len(compiled.channels) == np.linalg.matrix_rank(vectors, tol=1.0e-12)
    certificate = compiled.adjoint_artifact["global_filtered_owner_certificate"]
    assert compiled.adjoint_artifact["logical_candidate_channel_count"] == 8
    assert compiled.adjoint_artifact[
        "global_authored_projected_matrix_avoided"
    ]
    assert "global_projected_matrix_avoided" not in compiled.adjoint_artifact
    assert "global_retained_rank_certificate" not in compiled.adjoint_artifact
    assert certificate["input_real_column_count"] == 8
    assert certificate["retained_real_column_count"] == 5
    assert certificate["maximum_degree_slice_column_count"] == 8
    assert certificate["maximum_degree_slice_nonzero_count"] < certificate[
        "input_nonzero_count"
    ]
    assert certificate["single_pass_filtered_selection"]
    assert not certificate["global_authored_projected_matrix_materialized"]
    assert compiled.adjoint_artifact["orbit_hstack_materialized"]
    assert certificate["structural_projection_block_count"] == (
        compiled.adjoint_artifact["structural_orbit_count"]
    )
    assert certificate["owner_column_materialization_policy"] == (
        "selected_global_owner_only_v1"
    )
    assert certificate["materialized_owner_column_count"] == len(
        compiled.channels
    )
    assert certificate["avoided_owner_column_materialization_count"] == (
        certificate["input_real_column_count"] - len(compiled.channels)
    )
    timings = compiled.adjoint_artifact["timings_seconds"]
    assert timings["raw_adjoint_certification"] == 0.0
    for key in (
        "seed_validation_and_descriptors",
        "error_bound_construction",
        "orbit_hstack",
        "selected_owner_column_lookup",
        "function_body",
    ):
        assert timings[key] >= 0.0
    assert timings["function_body"] >= sum(
        timings[key]
        for key in (
            "seed_validation_and_descriptors",
            "error_bound_construction",
            "orbit_hstack",
            "selected_owner_column_lookup",
        )
    )
    assert compiled.adjoint_artifact[
        "maximum_orbit_hstack_real_column_count"
    ] == max(
        orbit["candidate_real_column_count"]
        for orbit in compiled.adjoint_artifact["structural_orbits"]
    )
    assert compiled.adjoint_artifact["maximum_orbit_hstack_nonzero_count"] == max(
        orbit["candidate_nonzero_count"]
        for orbit in compiled.adjoint_artifact["structural_orbits"]
    )


def test_structural_batches_preserve_single_global_filtered_owner_ids(
    monkeypatch,
) -> None:
    coordinate = PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=2,
    )
    e00 = np.asarray(((1.0, 0.0), (0.0, 0.0)), dtype=np.complex128)
    e11 = np.asarray(((0.0, 0.0), (0.0, 1.0)), dtype=np.complex128)
    tail = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128)
    nominal_degrees = (1, 2, 1, 2, 2, 1)
    tops = (e00, e00, e00, e11, e11, e11)
    seeds = tuple(
        _seed(
            f"owner-{index}",
            monomial=(degree, 0),
            top=top,
            lower={(0, 0): (index + 1) * tail},
            family="intra",
            term_index=index,
        )
        for index, (degree, top) in enumerate(zip(nominal_degrees, tops))
    )
    group = identity_finite_group(2, q_size=1, sector_size=1)
    internal_actions = {(): sparse.eye(2, format="csr", dtype=np.complex128)}
    action_artifact = {
        "maximum_action_error_bound": 0.0,
        "status": "test-certified",
    }
    row_degrees = graded.projected_row_degrees(coordinate=coordinate, dim=2)
    embedded_rows = np.concatenate(
        [
            np.flatnonzero(row_degrees == degree)[:2]
            for degree in (2, 1, 0)
        ]
    )
    abstract_real_columns = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 1.0, 0.0],
            [2.0, -2.0, 2.0, 1.0, -1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0, 0.0, -2.0],
            [0.0, 1.0, -1.0, 2.0, -1.0, -1.0],
            [1.0, -2.0, -1.0, -1.0, 0.0, 1.0],
        ]
    )
    global_projected = sparse.lil_matrix(
        (len(row_degrees), 2 * len(seeds)), dtype=np.float64
    )
    for seed_index in range(len(seeds)):
        global_projected[embedded_rows, 2 * seed_index] = (
            abstract_real_columns[:, seed_index].reshape(-1, 1)
        )
    global_projected = global_projected.tocsc()
    seed_index_by_id = {
        str(seed.seed_id): index for index, seed in enumerate(seeds)
    }

    def projected_batch(batch, atoms, **kwargs):
        del atoms, kwargs
        owners = [
            owner
            for seed in batch
            for owner in (
                2 * seed_index_by_id[str(seed.seed_id)],
                2 * seed_index_by_id[str(seed.seed_id)] + 1,
            )
        ]
        return global_projected[:, owners].tocsc()

    monkeypatch.setattr(graded, "_symbolic_reynolds_batch", projected_batch)
    absolute_errors = []
    for seed in seeds:
        components = response_basis._propagated_error_components(
            response_basis._coefficient_norm(seed.coefficients),
            dim=seed.dim,
            degree=coordinate.max_degree,
            group=group,
        )
        absolute_errors.extend((sum(components.values()),) * 2)
    reference = graded.select_filtered_independent_columns(
        global_projected,
        row_degrees=row_degrees,
        nominal_degrees=np.repeat(nominal_degrees, 2),
        absolute_error_bounds=np.asarray(absolute_errors),
    )
    logical_ids = tuple(
        f"{seed.seed_id}:{component}"
        for seed in seeds
        for component in ("real", "imag")
    )
    reference_ids = tuple(logical_ids[index] for index in reference.selected_indices)
    assert reference_ids == tuple(f"owner-{index}:real" for index in range(5))

    compiled = graded.compile_graded_candidate_group(
        seeds,
        coordinate=coordinate,
        group=group,
        factorized_actions={},
        internal_actions_by_word=internal_actions,
        factorized_group_artifact=action_artifact,
    )

    assert compiled.adjoint_artifact["structural_orbit_count"] == 2
    assert tuple(channel.channel_id for channel in compiled.channels) == reference_ids
