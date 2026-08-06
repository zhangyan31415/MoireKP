from __future__ import annotations

import numpy as np

import kp.model.response_basis as response_basis
from kp.model.core import ContinuumTermKey, MoireConfig, build_model
from kp.model.pipeline import _signed_case_harmonic_records
from kp.model.response_basis import COMPLETE_LINEAR_V2


def test_case_harmonics_author_one_representative_per_adjoint_orbit() -> None:
    p = np.asarray([0.75, 0.25], dtype=float)

    records = _signed_case_harmonic_records(
        {
            1: np.zeros(2, dtype=float),
            2: p,
        },
        kind="intra",
        tol=1.0e-10,
    )

    # Hermitian projection supplies the -p response.  It must not be authored
    # as another candidate record before symmetry/rank reduction.
    assert len(records) == 2
    zero, finite = records
    np.testing.assert_allclose(zero["vector"], np.zeros(2), atol=0.0)
    np.testing.assert_allclose(finite["vector"], p, atol=0.0)
    assert not any(
        np.allclose(record["vector"], -p, rtol=0.0, atol=1.0e-12)
        for record in records
    )

    # Keep enough explicit provenance for the later joint symmetry/adjoint
    # compiler to reconstruct and certify the complete physical orbit.
    assert finite["adjoint_harmonic_id"] == finite["id"]
    assert finite["adjoint_generation"] == "hermitian_projection"
    np.testing.assert_allclose(finite["adjoint_vector"], -p, atol=0.0)
    assert finite["source_orbit_harmonic_ids"] == [2]
    np.testing.assert_allclose(
        finite["source_orbit_vectors"],
        np.asarray([p, -p]),
        atol=0.0,
    )


def test_literal_harmonic_adjoint_provenance_reaches_candidate_term_metadata() -> None:
    p = np.asarray([1.0, 0.0], dtype=float)
    [record] = _signed_case_harmonic_records(
        {7: p},
        kind="inter",
        tol=1.0e-10,
    )
    literal_record = {
        **record,
        "vector": np.asarray(record["vector"], dtype=float).tolist(),
        "support_count": 1,
    }
    config = MoireConfig(
        Q_set1=np.asarray([[0.0, 0.0]], dtype=float),
        Q_set2=np.asarray([[1.0, 0.0]], dtype=float),
        n_orb1=1,
        n_orb2=1,
        nlow_state=[1, 1],
        bM1=np.asarray([1.0, 0.0]),
        bM2=np.asarray([0.0, 1.0]),
        intra_harmonics_map={},
        inter_harmonics_map={},
        max_order={"Kinect": 0, "Onsite": 0, "intra": 0, "inter": 0},
        symmetry_map={"Onsite": [], "Kinect": [], "intra": [], "inter": []},
        term_templates=[
            {
                "name": "case_inter_L2_to_L1",
                "source": "tunneling",
                "sector_pairs": [["L2", "L1"]],
                "orbital_pairs": "all",
                "harmonic_records": [literal_record],
                "max_order": 0,
                "term_space_policy": "complete",
            }
        ],
        sectors=[
            {"name": "L1", "qset": "qset1", "n_orb": 1},
            {"name": "L2", "qset": "qset2", "n_orb": 1},
        ],
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
                    "valley": "Gamma",
                },
            },
        },
    )

    model = build_model(config)

    [term] = model.candidate_terms
    metadata = term.registry_metadata
    assert metadata["adjoint_harmonic_id"] == record["adjoint_harmonic_id"]
    assert metadata["adjoint_generation"] == "hermitian_projection"
    np.testing.assert_allclose(metadata["adjoint_vector"], -p, atol=0.0)
    assert metadata["source_orbit_harmonic_ids"] == [7]
    np.testing.assert_allclose(
        metadata["source_orbit_vectors"],
        np.asarray([p, -p]),
        atol=0.0,
    )

    response_basis.clear_response_basis_cache()
    compiled = response_basis.compile_model_response_basis(
        model,
        config,
        reduce=True,
    )
    support_records = compiled.candidate_artifact["support"]
    assert len(support_records) == 1
    [support_certificate] = support_records.values()
    assert support_certificate["support_mask_policy"] == (
        "complete_symmetry_adjoint_closure_v1"
    )
    assert support_certificate["symmetry_adjoint_added_entry_count"] > 0
    assert support_certificate["structural_support_closure_certified"] is True
    assert any(
        channel["metadata"].get("adjoint_generation")
        == "hermitian_projection"
        for channel in compiled.candidate_artifact["channels"]
    )


def _finite_harmonic_response_columns(
    harmonic_signs: tuple[int, ...],
) -> np.ndarray:
    qset = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    empty_qset = np.empty((0, 2), dtype=float)
    n_orb = 2
    dim = len(qset) * n_orb
    coordinate = response_basis.PolynomialCoordinateBasis.from_reciprocal_basis(
        origin=(0.0, 0.0),
        reciprocal_basis=((1.0, 0.0), (0.0, 1.0)),
        max_degree=2,
    )
    seeds = []
    for sign in harmonic_signs:
        for degree in range(coordinate.max_degree + 1):
            for mz in range(degree + 1):
                mz_star = degree - mz
                for orbital_from in range(1, n_orb + 1):
                    for orbital_to in range(1, n_orb + 1):
                        key = ContinuumTermKey(
                            mz,
                            mz_star,
                            1,
                            1,
                            orbital_from,
                            orbital_to,
                            (float(sign), 0.0),
                        )
                        seed_id = (
                            f"p{sign:+d}:m{mz}:{mz_star}:"
                            f"o{orbital_from}:{orbital_to}"
                        )
                        seeds.append(
                            response_basis.raw_polynomial_seed_from_term_key(
                                key,
                                seed_id=seed_id,
                                Q_set1=qset,
                                Q_set2=empty_qset,
                                n_orb1=n_orb,
                                n_orb2=0,
                                coordinate=coordinate,
                                support_component="finite-harmonic",
                                metadata={"term_space_policy": "complete"},
                            )
                        )

    candidates = response_basis.compile_candidate_responses(
        seeds,
        coordinate=coordinate,
        group=response_basis.identity_finite_group(dim),
    )
    return np.column_stack(
        [
            response_basis._channel_sparse_vector(channel, coordinate, dim)
            .toarray()
            .ravel()
            for channel in candidates.channels
        ]
    )


def test_hermitian_projection_of_one_harmonic_representative_spans_both_signs() -> None:
    # The complete degree filtration is important: the adjoint of a finite-p
    # Q-centred monomial may contain lower global polynomial degrees.  With all
    # ordered orbital pairs and all degrees through the cutoff present, adding
    # explicitly authored -p seeds cannot enlarge the physical response span.
    canonical = _finite_harmonic_response_columns((1,))
    redundantly_signed = _finite_harmonic_response_columns((1, -1))

    assert np.linalg.matrix_rank(canonical, tol=1.0e-10) == np.linalg.matrix_rank(
        redundantly_signed,
        tol=1.0e-10,
    )
    canonical_projector = canonical @ np.linalg.pinv(canonical, rcond=1.0e-12)
    signed_projector = redundantly_signed @ np.linalg.pinv(
        redundantly_signed,
        rcond=1.0e-12,
    )
    np.testing.assert_allclose(
        canonical_projector,
        signed_projector,
        rtol=0.0,
        atol=2.0e-12,
    )
