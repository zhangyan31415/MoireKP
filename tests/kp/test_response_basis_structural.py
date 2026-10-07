from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import kp.model.response_basis as response_basis
from kp.model.core import ContinuumModel, ContinuumTermKey, MoireConfig
from kp.model.response_basis import (
    COMPLETE_LINEAR_V2,
    FiniteGroupGenerator,
    build_finite_group,
    compile_model_response_basis,
)
from kp.model.response_basis_structural import compile_structural_generator_plan


def _term(
    index: int,
    *,
    row: int,
    column: int,
    monomial: tuple[int, int] = (0, 0),
    p: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, SimpleNamespace]:
    return (
        index,
        SimpleNamespace(
            key=ContinuumTermKey(
                monomial[0],
                monomial[1],
                1,
                1,
                row + 1,
                column + 1,
                p,
            ),
            tag="intra",
            registry_metadata={
                "term_name": "gamma_case_dense_complete_intra",
                "term_kind": "intra",
                "term_space_policy": "complete",
            },
        ),
    )


def test_dense_orbital_mixing_selects_cyclic_generators_before_seed_materialization() -> None:
    hadamard = np.asarray(
        ((1.0, 1.0), (1.0, -1.0)), dtype=np.complex128
    ) / np.sqrt(2.0)
    group = build_finite_group(
        [
            FiniteGroupGenerator(
                name="C2-dense-orbital-mixing",
                antiunitary=False,
                canonical_k_map=((-1, 0), (0, -1)),
                q_permutation=(0,),
                sector_permutation=(0,),
                k_forward=((-1.0, 0.0), (0.0, -1.0)),
                internal_u=hadamard,
            )
        ]
    )
    records = tuple(
        _term(2 * row + column, row=row, column=column)
        for row in range(2)
        for column in range(2)
    )

    plan = compile_structural_generator_plan(
        records,
        group=group,
        Q_set1=np.asarray(((0.0, 0.0),)),
        Q_set2=np.empty((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        maximum_action_error_bound=0.0,
    )

    assert plan.full_structural_support_count == 4
    assert plan.selected_generator_count == 2
    assert len(plan.selected_term_indices) == 2
    assert plan.selected_closure_rank == plan.full_closure_rank == 8
    assert plan.maximum_omitted_closure_residual < 2.0e-12


def test_selected_finite_p_generator_keeps_every_allowed_monomial() -> None:
    monomials = tuple(
        (mz, degree - mz)
        for degree in range(3)
        for mz in range(degree + 1)
    )
    records = tuple(
        _term(
            support * len(monomials) + monomial_index,
            row=row,
            column=column,
            monomial=monomial,
            p=p,
        )
        for support, (row, column, p) in enumerate(
            ((0, 1, (1.0, 0.0)), (1, 0, (-1.0, 0.0)))
        )
        for monomial_index, monomial in enumerate(monomials)
    )

    plan = compile_structural_generator_plan(
        records,
        group=response_basis.identity_finite_group(6),
        Q_set1=np.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))),
        Q_set2=np.empty((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
    )

    assert plan.full_logical_term_count == 12
    assert plan.selected_generator_count == 1
    assert plan.materialized_term_count == 6
    selected_monomials = {
        (records[index][1].key.Mz, records[index][1].key.Mz_star)
        for index in plan.selected_term_indices
    }
    assert selected_monomials == set(monomials)


def test_model_compiler_materializes_only_selected_adjoint_generator(
    monkeypatch,
) -> None:
    model = ContinuumModel()
    metadata = {
        "term_name": "gamma_case_complete_offdiagonal_intra",
        "term_kind": "intra",
        "term_space_policy": "complete",
    }
    for index, (row, column, p_value) in enumerate(
        ((1, 2, (1.0, 0.0)), (2, 1, (-1.0, 0.0)))
    ):
        model.add_term(
            ContinuumTermKey(0, 0, 1, 1, row, column, p_value),
            lambda _k: np.zeros((6, 6), dtype=np.complex128),
            tag="intra",
            symmetry_ops=[],
            registry_metadata={**metadata, "authored_index": index},
        )
    config = MoireConfig(
        Q_set1=np.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))),
        Q_set2=np.empty((0, 2), dtype=float),
        n_orb1=2,
        n_orb2=0,
        nlow_state=[2, 0],
        bM1=np.asarray((1.0, 0.0)),
        bM2=np.asarray((0.0, 1.0)),
        response_semantics=COMPLETE_LINEAR_V2,
    )
    config.symmetry_source_metadata = {
        "exactification_owner": "kp_symm",
        "kp_symm_exactification": {
            "polynomial_coordinate": {
                "coordinate_convention": (
                    "right_handed_model_cartesian_reciprocal_v1"
                ),
                "origin": [0.0, 0.0],
                "origin_role": (
                    "exactified_valley_expansion_origin_in_model_cartesian"
                ),
            }
        },
    }
    calls = 0
    original = response_basis.raw_polynomial_seed_from_term_key

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        response_basis, "raw_polynomial_seed_from_term_key", counted
    )

    basis = compile_model_response_basis(model, config, reduce=True)

    structural = basis.candidate_artifact["structural_preselection"]
    assert calls == structural["materialized_term_count"] == 1
    assert structural["full_logical_term_count"] == 2
    assert structural["selected_generator_count"] == 1
    assert len(basis.channels) == 2
