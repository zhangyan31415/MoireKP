from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import threading

import numpy as np
import pytest

from kp import gamma_auto_producer as producer_mod
from kp.gamma_auto_producer import (
    GammaAutomaticProducerInputs,
    GammaAutomaticSelectionConfig,
    GammaRawOperationSpec,
    evaluate_gamma_automatic_selection,
    prepare_gamma_automatic_selection,
    produce_gamma_automatic_selection,
)
from kp.blocks.gamma_layout import GammaRowLayout
from kp.identity import hash_array
from kp.selection_artifact import (
    build_certified_gamma_selection_identity,
    gamma_routed_ordered_q_identity_hash,
)
from kp.symmetry.joint_exactification import (
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
)
from kp.symmetry.candidate_certificate import (
    CandidateOperationInput,
    candidate_raw_action_package_hash,
)


def _config_payload() -> dict[str, object]:
    return {
        "mode": "auto",
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
            "band_count": 2,
            "validation_k_indices": [0, 1],
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


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("routing_thresholds", "route_covariance"),
        ("selection_thresholds", "band_max_mev"),
        ("candidate_symmetry_thresholds", "heff_covariance_residual"),
    ],
)
def test_gamma_auto_config_rejects_each_missing_hard_threshold(
    section: str,
    field: str,
) -> None:
    payload = _config_payload()
    del payload[section][field]  # type: ignore[index]

    with pytest.raises(ValueError, match=field):
        GammaAutomaticSelectionConfig.from_normalized_config(payload)


def test_gamma_auto_config_rejects_defaults_unknown_fields_and_bool_numbers() -> None:
    missing = _config_payload()
    del missing["selection_thresholds"]
    with pytest.raises(ValueError, match="selection_thresholds"):
        GammaAutomaticSelectionConfig.from_normalized_config(missing)

    unknown = _config_payload()
    unknown["legacy_tolerance"] = 1.0e-3
    with pytest.raises(ValueError, match="unknown"):
        GammaAutomaticSelectionConfig.from_normalized_config(unknown)

    boolean = _config_payload()
    boolean["routing_thresholds"]["route_covariance"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="strict finite numeric"):
        GammaAutomaticSelectionConfig.from_normalized_config(boolean)


def _presentation() -> MagneticPresentation:
    return MagneticPresentation(
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
        source="gamma_auto_producer_test",
    )


def _producer_inputs(
    *,
    action: np.ndarray | None = None,
    kpoints: np.ndarray | None = None,
) -> GammaAutomaticProducerInputs:
    h0 = np.diag([-2.0, -1.0, 1.0, 2.0]).astype(np.complex128)
    h1 = np.diag([-1.8, -0.8, 1.2, 2.2]).astype(np.complex128)
    source = np.stack((h0, h1), axis=0)
    identity = np.eye(4, dtype=np.complex128) if action is None else action
    return GammaAutomaticProducerInputs(
        source_hamiltonians=source,
        k_indices=(0, 1),
        kpoints=(
            np.asarray([[0.0, 0.0], [0.25, 0.0]], dtype=np.float64)
            if kpoints is None
            else kpoints
        ),
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        tapw_source_basis_hash=hash_array(np.arange(4, dtype=np.int64)),
        operations=(
            GammaRawOperationSpec(
                name="E",
                full_action=identity,
                antiunitary=False,
                q_permutations=((0,), (0,)),
                sector_map=(0, 1),
                pairs=((0, 0), (1, 1)),
            ),
        ),
        presentation=_presentation(),
    )


def test_real_gamma_auto_producer_builds_all_k_handoff_and_certified_identity() -> None:
    config = GammaAutomaticSelectionConfig.from_normalized_config(_config_payload())
    inputs = _producer_inputs()

    result = produce_gamma_automatic_selection(inputs, config)

    assert result.decision.selected == result.candidates[0]
    assert result.rejected_candidates == ()
    assert result.handoff.k_indices == (0, 1)
    assert result.handoff.heff_k_indices == (0, 1)
    np.testing.assert_array_equal(result.handoff.kpoints, inputs.kpoints)
    assert result.handoff.kpoints_hash == hash_array(inputs.kpoints)
    assert result.handoff.joint_band_indices == (0, 1)
    assert result.handoff.candidate_id == result.decision.selected.candidate_id
    assert result.handoff.source_hamiltonian_hash == hash_array(
        inputs.source_hamiltonians
    )
    assert all(value != "0" * 64 for value in result.handoff.closure_certificate_hashes)
    assert all(value != "0" * 64 for value in result.handoff.routing_certificate_hashes)
    assert result.selection_input.action_package_hash == result.handoff.raw_action_package_hash
    assert result.selection_input.ordered_q_hash == gamma_routed_ordered_q_identity_hash(
        result.handoff
    )
    identity = build_certified_gamma_selection_identity(
        selection_input=result.selection_input,
        handoff=result.handoff,
        metrics=result.decision.selected,
    )
    assert identity.resolved_candidate.candidate_id == result.handoff.candidate_id
    for k_index in result.handoff.k_indices:
        u_low, _ = result.handoff.assemble_for_k(k_index, include_high=False)
        np.testing.assert_allclose(
            u_low.conj().T @ u_low,
            np.eye(result.handoff.model_dim),
            atol=1.0e-12,
        )


def test_gamma_auto_preselection_identity_binds_factorized_route_metadata() -> None:
    preparation = prepare_gamma_automatic_selection(
        _producer_inputs(),
        GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
    )
    operation = preparation.operation_inputs["E"]
    assert operation.route_contract is not None
    drifted_contract = dict(operation.route_contract)
    drifted_contract["sector_map"] = [1, 0]
    drifted_operation = CandidateOperationInput(
        name=operation.name,
        antiunitary=operation.antiunitary,
        d_full=operation.d_full,
        pairs=operation.pairs,
        route_contract=drifted_contract,
    )
    drifted_hash = candidate_raw_action_package_hash(
        operations={"E": drifted_operation},
        presentation=preparation.inputs.presentation,
        required_pairs=preparation.required_pairs,
    )

    assert drifted_hash != preparation.selection_input.action_package_hash


def test_gamma_auto_preselection_identity_binds_sampled_kpoints_and_pairs() -> None:
    config = GammaAutomaticSelectionConfig.from_normalized_config(_config_payload())
    original = prepare_gamma_automatic_selection(_producer_inputs(), config)
    shifted = prepare_gamma_automatic_selection(
        _producer_inputs(
            kpoints=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float64)
        ),
        config,
    )

    contract = original.operation_inputs["E"].route_contract
    assert contract is not None
    assert contract["sampled_k_route"]["pairs"] == [[0, 0], [1, 1]]
    assert original.selection_input.action_package_hash != shifted.selection_input.action_package_hash


def test_gamma_target_eigensystems_parallelize_with_one_outer_blas_limit_and_keep_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hamiltonians = np.stack(
        tuple(
            np.diag([10.0 + k_position, -1.0 - k_position]).astype(np.complex128)
            for k_position in range(3)
        ),
        axis=0,
    )
    original_eigh = np.linalg.eigh
    rendezvous = threading.Barrier(3, timeout=5.0)
    worker_threads: set[int] = set()
    executor_workers: list[int] = []
    scope_events: list[tuple[object, ...]] = []
    scope_active = False
    original_executor = producer_mod.ThreadPoolExecutor

    class RecordingThreadPoolExecutor(original_executor):
        def __init__(self, max_workers: int):
            executor_workers.append(max_workers)
            super().__init__(max_workers=max_workers)

    @contextmanager
    def fake_threadpool_limits(*, limits, user_api):
        nonlocal scope_active
        scope_events.append(("enter", limits, user_api))
        scope_active = True
        try:
            yield
        finally:
            scope_active = False
            scope_events.append(("exit",))

    def synchronized_eigh(matrix: np.ndarray):
        assert scope_active
        assert np.shares_memory(matrix, hamiltonians)
        worker_threads.add(threading.get_ident())
        rendezvous.wait()
        return original_eigh(matrix)

    import threadpoolctl

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", fake_threadpool_limits)
    monkeypatch.setattr(producer_mod.np.linalg, "eigh", synchronized_eigh)
    monkeypatch.setattr(producer_mod, "ThreadPoolExecutor", RecordingThreadPoolExecutor)

    values, vectors = producer_mod._target_eigensystems(
        hamiltonians,
        workers=99,
    )

    assert scope_events == [("enter", 1, "blas"), ("exit",)]
    assert executor_workers == [len(hamiltonians)]
    assert len(worker_threads) == len(hamiltonians)
    np.testing.assert_allclose(
        values,
        np.asarray([[-1.0, 10.0], [-2.0, 11.0], [-3.0, 12.0]]),
    )
    for k_position, eigvecs in enumerate(vectors):
        expected_values, expected_vectors = original_eigh(hamiltonians[k_position])
        np.testing.assert_allclose(values[k_position], expected_values)
        np.testing.assert_allclose(eigvecs, expected_vectors)


def test_gamma_preselection_identity_is_invariant_to_target_worker_count() -> None:
    inputs = _producer_inputs()
    config = GammaAutomaticSelectionConfig.from_normalized_config(_config_payload())

    serial = prepare_gamma_automatic_selection(inputs, config, workers=1)
    parallel = prepare_gamma_automatic_selection(inputs, config, workers=2)

    assert serial.selection_input == parallel.selection_input
    assert serial.source_hamiltonian_hash == parallel.source_hamiltonian_hash
    np.testing.assert_array_equal(serial.target_values, parallel.target_values)
    for serial_vectors, parallel_vectors in zip(
        serial.target_vectors,
        parallel.target_vectors,
        strict=True,
    ):
        np.testing.assert_array_equal(serial_vectors, parallel_vectors)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_gamma_preselection_rejects_non_positive_strict_integer_workers(
    workers: object,
) -> None:
    with pytest.raises(ValueError, match="workers.*strict positive integer"):
        prepare_gamma_automatic_selection(
            _producer_inputs(),
            GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
            workers=workers,  # type: ignore[arg-type]
        )


def test_real_gamma_auto_producer_fails_closed_when_raw_action_breaks_source_route() -> None:
    angle = 0.1
    leaked = np.eye(4, dtype=np.complex128)
    leaked[:2, :2] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    with pytest.raises(Exception) as rejected:
        produce_gamma_automatic_selection(
            _producer_inputs(action=leaked),
            GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
        )

    assert "route leakage" in str(rejected.value).lower()


def test_real_gamma_auto_producer_binds_all_k_states_beyond_operation_pairs() -> None:
    inputs = _producer_inputs()
    incomplete = GammaAutomaticProducerInputs(
        **{
            **inputs.__dict__,
            "operations": (
                GammaRawOperationSpec(
                    name="E",
                    full_action=np.eye(4),
                    antiunitary=False,
                    q_permutations=((0,), (0,)),
                    sector_map=(0, 1),
                    pairs=((0, 0),),
                ),
            ),
        }
    )

    result = produce_gamma_automatic_selection(
        incomplete,
        GammaAutomaticSelectionConfig.from_normalized_config(
            deepcopy(_config_payload())
        ),
    )

    state_keys = {
        (state.role, state.k_index)
        for state in result.evaluations[0].symmetry_certificate.states
    }
    assert ("source", 1) in state_keys
    assert ("target", 1) in state_keys


def test_gamma_auto_nq2_spinful_tr_keeps_complete_kramers_clusters() -> None:
    payload = _config_payload()
    payload["target_window"] = {
        "edge": "valence",
        "band_count": 8,
        "validation_k_indices": [0, 1],
        "energy_reference_ev": 0.0,
        "degeneracy_tolerance_mev": 1.0e-6,
    }
    payload["candidate_seed_band_indices"] = [[0, 1, 2, 3]]
    qsets = (
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )
    basis_hash = hash_array(np.arange(8, dtype=np.int64))
    layout = GammaRowLayout.build(
        qsets=qsets,
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=basis_hash,
    )
    hamiltonians = []
    for shift in (0.0, 0.1):
        hamiltonian = np.zeros((8, 8), dtype=np.complex128)
        for q_index in range(2):
            rows = layout.same_q_full_rows(q_index)
            hamiltonian[np.ix_(rows, rows)] = np.diag(
                [-2.0 + shift, -1.0 + shift, -2.0 + shift, -1.0 + shift]
            )
        hamiltonians.append(hamiltonian)
    local_tr = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    full_tr = np.zeros((8, 8), dtype=np.complex128)
    for q_index in range(2):
        rows = layout.same_q_full_rows(q_index)
        full_tr[np.ix_(rows, rows)] = local_tr
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("TR", True),),
        relations=(
            MagneticRelation(
                "TR^2",
                lhs=("TR", "TR"),
                rhs=(),
                central_phase=-1.0,
            ),
        ),
        central_phases=(1.0, -1.0),
        source="gamma_auto_nq2_tr_test",
    )
    inputs = GammaAutomaticProducerInputs(
        source_hamiltonians=np.stack(hamiltonians, axis=0),
        k_indices=(0, 1),
        kpoints=np.asarray([[0.0, 0.0], [0.25, 0.0]], dtype=np.float64),
        qsets=qsets,
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        tapw_source_basis_hash=basis_hash,
        operations=(
            GammaRawOperationSpec(
                name="TR",
                full_action=full_tr,
                antiunitary=True,
                q_permutations=((0, 1), (0, 1)),
                sector_map=(0, 1),
                pairs=((0, 0), (1, 1)),
            ),
        ),
        presentation=presentation,
    )

    result = produce_gamma_automatic_selection(
        inputs,
        GammaAutomaticSelectionConfig.from_normalized_config(payload),
    )

    assert result.handoff.layout.q_count == 2
    assert result.handoff.group_ranks == (2, 2)
    assert result.handoff.model_dim == 8
    assert result.handoff.joint_band_indices == (0, 1, 2, 3)
    assert result.evaluations[0].symmetry_certificate.operations[
        0
    ].antiunitary_square_residual == pytest.approx(0.0)


def test_gamma_auto_skips_smaller_structural_failure_and_selects_larger_candidate() -> None:
    payload = _config_payload()
    payload["candidate_seed_band_indices"] = [[0], [0, 1]]

    result = produce_gamma_automatic_selection(
        _producer_inputs(),
        GammaAutomaticSelectionConfig.from_normalized_config(payload),
    )

    assert len(result.rejected_candidates) == 1
    assert result.rejected_candidates[0].seed_band_indices == (0,)
    assert result.candidates == (result.decision.selected,)
    assert result.handoff.joint_band_indices == (0, 1)


def test_gamma_auto_v1_rejects_physical_k_ids_that_could_be_mistaken_for_positions() -> None:
    inputs = _producer_inputs()
    noncanonical = GammaAutomaticProducerInputs(
        **{**inputs.__dict__, "k_indices": (4, 7)}
    )

    with pytest.raises(ValueError, match="contiguous positional k_indices"):
        produce_gamma_automatic_selection(
            noncanonical,
            GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
        )


def test_gamma_auto_rejects_raw_full_h_covariance_failure_outside_low_space() -> None:
    inputs = _producer_inputs()
    hamiltonians = np.array(inputs.source_hamiltonians, copy=True)
    hamiltonians[1, :2, :2] = hamiltonians[0, :2, :2]
    hamiltonians[1, 2:, 2:] = np.diag([10.0, 20.0])
    cross_k = GammaAutomaticProducerInputs(
        **{
            **inputs.__dict__,
            "source_hamiltonians": hamiltonians,
            "operations": (
                GammaRawOperationSpec(
                    name="E",
                    full_action=np.eye(4),
                    antiunitary=False,
                    q_permutations=((0,), (0,)),
                    sector_map=(0, 1),
                    pairs=((1, 0), (0, 1)),
                ),
            ),
        }
    )

    with pytest.raises(Exception, match="source Hamiltonian covariance"):
        produce_gamma_automatic_selection(
            cross_k,
            GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
        )


def test_gamma_auto_preparation_exposes_final_identity_before_first_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = prepare_gamma_automatic_selection(
        _producer_inputs(),
        GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
    )
    with pytest.raises(TypeError):
        preparation.operation_inputs["mutated"] = preparation.operation_inputs["E"]
    with pytest.raises(ValueError):
        preparation.target_values[0, 0] = 99.0
    pending_published = False
    original = producer_mod._evaluate_candidate

    def checked_evaluator(**kwargs):
        assert pending_published is True
        assert kwargs["raw_action_package_hash"] == (
            preparation.selection_input.action_package_hash
        )
        return original(**kwargs)

    monkeypatch.setattr(producer_mod, "_evaluate_candidate", checked_evaluator)
    pending_published = True

    result = evaluate_gamma_automatic_selection(preparation)

    assert result.selection_input is preparation.selection_input
