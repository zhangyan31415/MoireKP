from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kp import gamma_auto_producer as producer_mod
from kp.blocks import blocks as blocks_impl
from kp.blocks.downfold import DownfoldingOptions, downfold_from_projectors
from kp.blocks.gamma_layout import GammaRoutingError, GammaRowLayout
from kp.gamma_auto_producer import (
    GammaAutomaticProducerInputs,
    GammaAutomaticSelectionConfig,
    GammaDownfoldConfig,
    GammaRawOperationSpec,
    evaluate_gamma_automatic_selection,
    prepare_gamma_automatic_selection,
    produce_gamma_automatic_selection,
)
from kp.identity import hash_array
from kp.projection_selection import CandidateRejectionReason
from kp.symmetry.joint_exactification import (
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
)


def _common_config(*, validation_k_indices: tuple[int, ...] = (1,)):
    """Keep non-routing policy explicit while the common producer is introduced."""

    return GammaAutomaticSelectionConfig.from_normalized_config(
        {
            "mode": "auto",
            "producer_integrity_thresholds": {
                "source_hamiltonian_covariance_residual": 1.0e-10,
                "routed_model_heff_covariance_residual": 1.0e-10,
            },
            "routing_thresholds": {
                "energy_same_ev": 1.0e-9,
                "energy_different_ev": 1.0e-5,
                "capture_zero_fraction": 1.0e-9,
                "capture_loss_max": 1.0e-9,
                "local_action_isometry": 1.0e-10,
                "off_route_leakage": 1.0e-10,
                "closure_residual": 1.0e-10,
                "route_zero_gap": 1.0e-8,
                "route_covariance": 1.0e-10,
                "projector_residual": 1.0e-10,
                "anchor_sigma_min": 1.0e-8,
                "max_rank": 4,
                "max_iterations": 8,
            },
            "selection_thresholds": {
                "band_rms_mev": 1.0e-6,
                "band_max_mev": 1.0e-6,
                "subspace_overlap": 1.0 - 1.0e-10,
                "symmetry_residual": 1.0e-9,
                "symmetry_leakage": 1.0e-9,
            },
            "candidate_symmetry_thresholds": {
                "raw_h_leakage": 1.0e-10,
                "exactification_distance": 1.0e-10,
                "intertwining_residual": 1.0e-10,
                "heff_covariance_residual": 1.0e-10,
                "relation_residual": 1.0e-10,
                "antiunitary_square_residual": 1.0e-10,
                "exact_action_unitarity_residual": 1.0e-10,
                "projection_orthonormality_residual": 1.0e-10,
                "heff_hermiticity_residual": 1.0e-10,
                "raw_h_action_unitarity_residual": 1.0e-10,
            },
            "exactification": {
                "enabled": True,
                "max_rms_correction": 1.0e-8,
                "max_route_correction": 1.0e-8,
                "central_branch_margin": 1.0e-6,
                "max_iterations": 8,
                "condition_limit": 1.0e8,
            },
            "target_window": {
                "edge": "valence",
                "band_count": 4,
                "validation_k_indices": list(validation_k_indices),
                "energy_reference_ev": 0.0,
                "degeneracy_tolerance_mev": 1.0e-6,
            },
            "candidate_seed_band_indices": [[0, 1]],
            "reference_k_index": 0,
            "downfold": {
                "method": "first_order",
                "e_ref": None,
                "pole_warning_mev": 10.0,
                "pole_danger_mev": 1.0,
                "fail_on_near_pole": True,
                "compute_pole_diagnostics": True,
                "compute_condition_number": False,
            },
        }
    )


def _common_inputs() -> GammaAutomaticProducerInputs:
    """Two k points and two Q channels with one low state per model owner."""

    qset = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    hamiltonians: list[np.ndarray] = []
    for shift in (0.0, 0.05):
        hamiltonian = np.zeros((8, 8), dtype=np.complex128)
        hamiltonian[np.ix_((0, 2, 4, 6), (0, 2, 4, 6))] = np.diag(
            np.asarray([-2.0, -1.0, 1.0, 2.0]) + shift
        )
        hamiltonian[np.ix_((1, 3, 5, 7), (1, 3, 5, 7))] = np.diag(
            np.asarray([-1.8, -0.8, 1.2, 2.2]) + shift
        )
        hamiltonians.append(hamiltonian)
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation(
                "E^2",
                lhs=("E", "E"),
                rhs=(),
                central_phase=1.0,
            ),
        ),
        central_phases=(1.0,),
        source="gamma_common_anchor_producer_test",
    )
    return GammaAutomaticProducerInputs(
        source_hamiltonians=np.stack(hamiltonians, axis=0),
        k_indices=(0, 1),
        kpoints=np.asarray([[0.0, 0.0], [0.25, 0.0]], dtype=np.float64),
        qsets=(qset, qset),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        tapw_source_basis_hash=hash_array(np.arange(8, dtype=np.int64)),
        operations=(
            GammaRawOperationSpec(
                name="E",
                full_action=np.eye(8, dtype=np.complex128),
                antiunitary=False,
                q_permutations=((0, 1), (0, 1)),
                sector_map=(0, 1),
                pairs=((0, 0), (1, 1)),
            ),
        ),
        presentation=presentation,
    )


def _layout(inputs: GammaAutomaticProducerInputs) -> GammaRowLayout:
    return GammaRowLayout.build(
        qsets=inputs.qsets,
        num_layer_list=inputs.num_layer_list,
        num_orb_per_layer_list=inputs.num_orb_per_layer_list,
        spin_convention="all",
        source_basis_hash=inputs.tapw_source_basis_hash,
    )


def _reordered_q_inputs(
    inputs: GammaAutomaticProducerInputs,
    order: tuple[int, ...],
) -> GammaAutomaticProducerInputs:
    old_layout = _layout(inputs)
    reordered_qsets = tuple(qset[np.asarray(order)] for qset in inputs.qsets)
    new_layout = GammaRowLayout.build(
        qsets=reordered_qsets,
        num_layer_list=inputs.num_layer_list,
        num_orb_per_layer_list=inputs.num_orb_per_layer_list,
        spin_convention="all",
        source_basis_hash=inputs.tapw_source_basis_hash,
    )
    old_rows_for_new = np.empty(old_layout.full_dimension, dtype=np.intp)
    for new_q, old_q in enumerate(order):
        old_rows_for_new[new_layout.same_q_full_rows(new_q)] = (
            old_layout.same_q_full_rows(old_q)
        )
    hamiltonians = inputs.source_hamiltonians[:, old_rows_for_new][
        :, :, old_rows_for_new
    ]
    operations = tuple(
        GammaRawOperationSpec(
            name=operation.name,
            full_action=np.asarray(operation.full_action)[
                np.ix_(old_rows_for_new, old_rows_for_new)
            ],
            antiunitary=operation.antiunitary,
            q_permutations=operation.q_permutations,
            sector_map=operation.sector_map,
            pairs=operation.pairs,
        )
        for operation in inputs.operations
    )
    return GammaAutomaticProducerInputs(
        **{
            **inputs.__dict__,
            "source_hamiltonians": hamiltonians,
            "qsets": reordered_qsets,
            "operations": operations,
        }
    )


def _q_symmetric_inputs(
    *,
    q_permutation: tuple[int, int],
    local_actions: tuple[np.ndarray, np.ndarray],
    degenerate_pairs: bool = False,
) -> GammaAutomaticProducerInputs:
    inputs = _common_inputs()
    layout = _layout(inputs)
    local_values = (
        np.asarray([-2.0, -2.0, 1.0, 1.0])
        if degenerate_pairs
        else np.asarray([-2.0, -1.0, 1.0, 2.0])
    )
    hamiltonians: list[np.ndarray] = []
    for shift in (0.0, 0.05):
        hamiltonian = np.zeros(
            (layout.full_dimension, layout.full_dimension),
            dtype=np.complex128,
        )
        for q_index in range(layout.q_count):
            rows = layout.same_q_full_rows(q_index)
            hamiltonian[np.ix_(rows, rows)] = np.diag(local_values + shift)
        hamiltonians.append(hamiltonian)
    full_action = np.zeros(
        (layout.full_dimension, layout.full_dimension),
        dtype=np.complex128,
    )
    for source_q, target_q in enumerate(q_permutation):
        source_rows = layout.same_q_full_rows(source_q)
        target_rows = layout.same_q_full_rows(target_q)
        full_action[np.ix_(target_rows, source_rows)] = local_actions[source_q]
    operation = GammaRawOperationSpec(
        name="E",
        full_action=full_action,
        antiunitary=False,
        q_permutations=(q_permutation, q_permutation),
        sector_map=(0, 1),
        pairs=((0, 0), (1, 1)),
    )
    return GammaAutomaticProducerInputs(
        **{
            **inputs.__dict__,
            "source_hamiltonians": np.stack(hamiltonians, axis=0),
            "operations": (operation,),
        }
    )


def _strict_routing_must_not_run(*_args, **_kwargs):
    raise AssertionError("common-anchor production called strict routed-projector code")


def test_default_gamma_producer_is_common_anchor_and_never_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "close_gamma_projector_clusters",
        "build_gamma_routed_frames",
        "certify_routed_covariance",
        "assemble_gamma_routed_projectors",
        "_routing_certificate_hash",
    ):
        monkeypatch.setattr(
            producer_mod,
            name,
            _strict_routing_must_not_run,
            raising=False,
        )

    qrcp_calls = 0
    real_qrcp = blocks_impl.select_anchor_rows_qrcp

    def count_qrcp(*args, **kwargs):
        nonlocal qrcp_calls
        qrcp_calls += 1
        return real_qrcp(*args, **kwargs)

    monkeypatch.setattr(blocks_impl, "select_anchor_rows_qrcp", count_qrcp)

    result = produce_gamma_automatic_selection(_common_inputs(), _common_config())

    handoff = result.handoff
    assert handoff.projection_basis_kind == "gamma_common_anchor"
    assert handoff.local_frames.shape == (2, 2, 4, 2)
    assert handoff.physical_layer_anchor_counts == (1, 1)
    assert handoff.source_qset_anchor_counts == (1, 1)
    assert handoff.active_model_layers == (0, 1)
    assert handoff.model_group_ranks == (1, 1)
    assert handoff.model_group_qset_indices == (0, 1)
    assert not hasattr(handoff, "routing_certificate_hashes")
    assert qrcp_calls == 1


def test_public_common_anchor_does_not_gate_on_full_source_h_covariance() -> None:
    """High-energy source drift must not veto an exact low-energy model."""

    inputs = _q_symmetric_inputs(
        q_permutation=(0, 1),
        local_actions=(np.eye(4), np.eye(4)),
    )
    layout = _layout(inputs)
    hamiltonians = np.array(inputs.source_hamiltonians, copy=True)
    for q_index in range(layout.q_count):
        rows = layout.same_q_full_rows(q_index)
        hamiltonians[1][np.ix_(rows, rows)] = np.diag(
            [-2.0, -1.0, 10.0, 20.0]
        )
    operation = GammaRawOperationSpec(
        name="E",
        full_action=np.eye(layout.full_dimension, dtype=np.complex128),
        antiunitary=False,
        q_permutations=((0, 1), (0, 1)),
        sector_map=(0, 1),
        pairs=((0, 1), (1, 0)),
    )
    inputs = GammaAutomaticProducerInputs(
        **{
            **inputs.__dict__,
            "source_hamiltonians": hamiltonians,
            "operations": (operation,),
        }
    )
    config = GammaAutomaticSelectionConfig.from_normalized_project_config(
        {
            "target": "valence",
            "efermi": 0.0,
            "selection": {
                "mode": "auto",
                "max_dimension": 2,
                "validation_indices": [0],
                "validation_bands": 4,
            },
        }
    )

    result = produce_gamma_automatic_selection(inputs, config)

    assert result.handoff.anchor_spec.joint_band_indices == (0, 1)


def test_common_producer_diagonalizes_each_local_block_once_and_downfolds_once_per_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _common_inputs()
    config = _common_config()
    real_eigh = np.linalg.eigh
    local_eigh_shapes: list[tuple[int, int]] = []

    def count_local_eigh(matrix):
        shape = tuple(np.shape(matrix))
        if shape == (4, 4):
            local_eigh_shapes.append(shape)
        return real_eigh(matrix)

    monkeypatch.setattr(producer_mod.np.linalg, "eigh", count_local_eigh)
    preparation = prepare_gamma_automatic_selection(inputs, config)
    assert local_eigh_shapes == [(4, 4)] * 4

    # Candidate Heff diagonalization is not a local source-block solve.
    monkeypatch.setattr(producer_mod.np.linalg, "eigh", real_eigh)
    downfold_k_positions: list[int] = []

    def count_downfold(hamiltonian, *args, **kwargs):
        matches = [
            position
            for position, source in enumerate(inputs.source_hamiltonians)
            if np.array_equal(hamiltonian, source)
        ]
        assert len(matches) == 1
        downfold_k_positions.append(matches[0])
        return downfold_from_projectors(hamiltonian, *args, **kwargs)

    monkeypatch.setattr(producer_mod, "downfold_from_projectors", count_downfold)
    evaluate_gamma_automatic_selection(preparation)

    assert sorted(downfold_k_positions) == [0, 1]


def test_target_full_h_eigensolver_is_not_used_for_validation_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _common_inputs()
    preparation = prepare_gamma_automatic_selection(
        inputs,
        _common_config(validation_k_indices=(1,)),
    )

    assert preparation.target_values.shape == (1, 8)


def test_geometric_reference_q_and_anchor_policy_are_q_order_covariant() -> None:
    inputs = _common_inputs()
    config = _common_config()
    original = produce_gamma_automatic_selection(inputs, config)
    order = (1, 0)
    reordered = produce_gamma_automatic_selection(
        _reordered_q_inputs(inputs, order),
        config,
    )

    assert original.handoff.anchor_spec.reference_q_index == 0
    assert reordered.handoff.anchor_spec.reference_q_index == 1
    assert reordered.handoff.anchor_spec.min_sigma == pytest.approx(
        config.routing_thresholds.anchor_sigma_min
    )
    assert reordered.handoff.anchor_spec.max_condition == pytest.approx(
        1.0 / config.routing_thresholds.anchor_sigma_min
    )

    q_count = original.handoff.layout.q_count
    ranks = original.handoff.model_group_ranks
    old_columns_for_new = np.asarray(
        [
            q_count * sum(ranks[:group]) + orbital * q_count + order[new_q]
            for group, rank in enumerate(ranks)
            for orbital in range(rank)
            for new_q in range(q_count)
        ],
        dtype=np.intp,
    )
    for position in range(len(original.handoff.k_indices)):
        expected = original.handoff.authoritative_heff[position][
            np.ix_(old_columns_for_new, old_columns_for_new)
        ]
        np.testing.assert_allclose(
            reordered.handoff.authoritative_heff[position],
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_geometric_reference_q_must_be_unique() -> None:
    inputs = _common_inputs()
    ambiguous_qset = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    ambiguous = GammaAutomaticProducerInputs(
        **{**inputs.__dict__, "qsets": (ambiguous_qset, ambiguous_qset)}
    )

    with pytest.raises(GammaRoutingError, match="unique geometric reference Q"):
        prepare_gamma_automatic_selection(ambiguous, _common_config())


def test_candidate_seed_cannot_exceed_max_rank() -> None:
    config = _common_config()
    limited = replace(
        config,
        routing_thresholds=replace(config.routing_thresholds, max_rank=1),
    )

    with pytest.raises(GammaRoutingError, match="max_rank") as error:
        produce_gamma_automatic_selection(_common_inputs(), limited)

    assert error.value.reason is CandidateRejectionReason.SYMMETRY_CLOSURE_FAILURE


def test_degenerate_cluster_cut_is_rejected_and_larger_candidate_is_selected() -> None:
    identity = np.eye(4, dtype=np.complex128)
    inputs = _q_symmetric_inputs(
        q_permutation=(0, 1),
        local_actions=(identity, identity),
        degenerate_pairs=True,
    )
    config = replace(
        _common_config(),
        candidate_seed_band_indices=((0,), (0, 1)),
    )

    result = produce_gamma_automatic_selection(inputs, config)

    assert len(result.rejected_candidates) == 1
    assert result.rejected_candidates[0].seed_band_indices == (0,)
    assert (
        result.rejected_candidates[0].reason
        is CandidateRejectionReason.AMBIGUOUS_ENERGY_CLUSTER
    )
    assert result.handoff.anchor_spec.joint_band_indices == (0, 1)


@pytest.mark.parametrize(
    "unexpected",
    (
        ValueError("bad candidate shape"),
        TypeError("bad candidate type"),
        np.linalg.LinAlgError("candidate solver failed"),
    ),
)
def test_unexpected_candidate_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
    unexpected: Exception,
) -> None:
    preparation = prepare_gamma_automatic_selection(
        _common_inputs(),
        _common_config(),
    )

    def fail_candidate(**_kwargs):
        raise unexpected

    monkeypatch.setattr(producer_mod, "_evaluate_candidate", fail_candidate)

    with pytest.raises(type(unexpected), match=str(unexpected)):
        evaluate_gamma_automatic_selection(preparation)


def test_typed_candidate_rejection_falls_through_to_larger_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _common_config(),
        candidate_seed_band_indices=((0,), (0, 1)),
    )
    preparation = prepare_gamma_automatic_selection(_common_inputs(), config)
    real_evaluate_candidate = producer_mod._evaluate_candidate

    def reject_small_candidate(*, seed, **kwargs):
        if seed == (0,):
            raise GammaRoutingError(
                CandidateRejectionReason.PROJECTOR_FRAME_RANK,
                "small candidate has no stable anchor",
            )
        return real_evaluate_candidate(seed=seed, **kwargs)

    monkeypatch.setattr(
        producer_mod,
        "_evaluate_candidate",
        reject_small_candidate,
    )

    result = evaluate_gamma_automatic_selection(preparation)

    assert result.handoff.anchor_spec.joint_band_indices == (0, 1)
    assert len(result.rejected_candidates) == 1
    assert (
        result.rejected_candidates[0].reason
        is CandidateRejectionReason.PROJECTOR_FRAME_RANK
    )


def test_q_dependent_internal_action_is_rejected() -> None:
    identity = np.eye(4, dtype=np.complex128)
    nonuniform = np.diag([1.0, -1.0, 1.0, -1.0]).astype(np.complex128)
    inputs = _q_symmetric_inputs(
        q_permutation=(0, 1),
        local_actions=(identity, nonuniform),
    )

    with pytest.raises(GammaRoutingError, match="Q-independent internal action") as error:
        produce_gamma_automatic_selection(inputs, _common_config())

    assert error.value.reason is CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED


def test_public_selection_leaves_q_dependent_candidate_symmetry_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public candidate choice depends only on bands and common-anchor coverage."""

    identity = np.eye(4, dtype=np.complex128)
    nonuniform = np.diag([1.0, -1.0, 1.0, -1.0]).astype(np.complex128)
    inputs = _q_symmetric_inputs(
        q_permutation=(0, 1),
        local_actions=(identity, nonuniform),
    )
    config = GammaAutomaticSelectionConfig.from_normalized_project_config(
        {
            "target": "valence",
            "efermi": 0.0,
            "selection": {
                "mode": "auto",
                "max_dimension": 2,
                "validation_indices": [0],
                "validation_bands": 2,
                "thresholds": {
                    "band_rms_mev": 1.0e6,
                    "band_max_mev": 1.0e6,
                    "subspace_overlap": 0.01,
                },
            },
        }
    )

    def reject_preselection_symmetry(*args, **kwargs):
        raise AssertionError("public candidate selection called symmetry certification")

    monkeypatch.setattr(
        producer_mod,
        "_exactified_candidate_actions",
        reject_preselection_symmetry,
    )
    monkeypatch.setattr(
        producer_mod,
        "certify_candidate_symmetries",
        reject_preselection_symmetry,
    )

    result = produce_gamma_automatic_selection(inputs, config)

    assert result.decision.status == "PASS"
    assert result.evaluations[0].symmetry_certificate is None
    assert result.candidates[0].symmetry_residual is None
    assert result.candidates[0].symmetry_leakage is None


def test_uniform_internal_action_materializes_on_actual_q_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = np.eye(4, dtype=np.complex128)
    inputs = _q_symmetric_inputs(
        q_permutation=(1, 0),
        local_actions=(identity, identity),
    )
    captured: dict[str, np.ndarray] = {}
    internal_dimensions: list[tuple[int, ...]] = []
    real_certify = producer_mod.certify_candidate_symmetries
    real_joint_certify = producer_mod.certify_joint_block_actions

    def capture_exact_actions(**kwargs):
        captured.update(kwargs["exactified_actions"])
        return real_certify(**kwargs)

    def capture_internal_actions(actions, presentation):
        internal_dimensions.extend(
            action.fiber_dimensions for action in actions.values()
        )
        return real_joint_certify(actions, presentation)

    monkeypatch.setattr(
        producer_mod,
        "certify_candidate_symmetries",
        capture_exact_actions,
    )
    monkeypatch.setattr(
        producer_mod,
        "certify_joint_block_actions",
        capture_internal_actions,
    )

    result = produce_gamma_automatic_selection(inputs, _common_config())

    ranks = result.handoff.model_group_ranks
    q_count = result.handoff.layout.q_count
    expected = np.zeros(
        (result.handoff.model_dim, result.handoff.model_dim),
        dtype=np.complex128,
    )
    for source_q, target_q in enumerate((1, 0)):
        source_columns = producer_mod._model_columns_for_q(
            q_index=source_q,
            q_count=q_count,
            model_group_ranks=ranks,
        )
        target_columns = producer_mod._model_columns_for_q(
            q_index=target_q,
            q_count=q_count,
            model_group_ranks=ranks,
        )
        expected[np.ix_(target_columns, source_columns)] = np.eye(sum(ranks))
    np.testing.assert_allclose(captured["E"], expected, rtol=0.0, atol=1.0e-12)
    assert internal_dimensions == [(sum(ranks),)]
    assert result.evaluations[0].symmetry_certificate.status.value == "certified"


def test_fixed_schur_grouped_downfold_matches_dense_complement() -> None:
    config = _common_config()
    fixed = replace(
        config,
        downfold=GammaDownfoldConfig(
            DownfoldingOptions(
                method="fixed_schur",
                e_ref=0.0,
                pole_warning_mev=10.0,
                pole_danger_mev=1.0,
                fail_on_near_pole=False,
                compute_pole_diagnostics=True,
                compute_condition_number=False,
            )
        ),
    )
    preparation = prepare_gamma_automatic_selection(_common_inputs(), fixed)
    result = evaluate_gamma_automatic_selection(preparation)
    selected = set(result.handoff.anchor_spec.joint_band_indices)
    high_indices = np.asarray(
        [
            index
            for index in range(result.handoff.layout.same_q_dimension)
            if index not in selected
        ],
        dtype=np.intp,
    )
    high_rank = len(high_indices)

    for position, k_index in enumerate(result.handoff.k_indices):
        dense_high = np.zeros(
            (result.handoff.layout.full_dimension, high_rank * 2),
            dtype=np.complex128,
        )
        for q_index in range(result.handoff.layout.q_count):
            rows = result.handoff.layout.same_q_full_rows(q_index)
            columns = np.arange(
                q_index * high_rank,
                (q_index + 1) * high_rank,
                dtype=np.intp,
            )
            dense_high[np.ix_(rows, columns)] = preparation.local_vectors[
                position
            ][q_index][:, high_indices]
        expected = downfold_from_projectors(
            preparation.inputs.source_hamiltonians[position],
            result.handoff.assemble_model_for_k(k_index),
            dense_high,
            fixed.downfold.options,
        ).heff
        np.testing.assert_allclose(
            result.handoff.authoritative_heff[position],
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )
