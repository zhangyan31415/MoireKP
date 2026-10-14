from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from kp import gamma_auto_producer as producer_mod
from kp.gamma_auto_producer import (
    GammaAutomaticProducerInputs,
    GammaAutomaticSelectionConfig,
    GammaProducerIntegrityThresholds,
    GammaRawOperationSpec,
    evaluate_gamma_automatic_selection,
    prepare_gamma_automatic_selection,
    produce_gamma_automatic_selection,
)
from kp.blocks.gamma_layout import GammaRowLayout
from kp.identity import hash_array
from kp.selection_artifact import (
    build_certified_gamma_selection_identity,
    gamma_common_anchor_ordered_q_identity_hash,
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
        (
            "producer_integrity_thresholds",
            "source_hamiltonian_covariance_residual",
        ),
        (
            "producer_integrity_thresholds",
            "routed_model_heff_covariance_residual",
        ),
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

    negative = _config_payload()
    negative["producer_integrity_thresholds"][  # type: ignore[index]
        "source_hamiltonian_covariance_residual"
    ] = -1.0
    with pytest.raises(ValueError, match="strict finite numeric"):
        GammaAutomaticSelectionConfig.from_normalized_config(negative)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_hamiltonian_covariance_residual", True),
        ("source_hamiltonian_covariance_residual", np.nan),
        ("source_hamiltonian_covariance_residual", np.inf),
        ("source_hamiltonian_covariance_residual", -1.0),
        ("routed_model_heff_covariance_residual", False),
        ("routed_model_heff_covariance_residual", np.nan),
        ("routed_model_heff_covariance_residual", np.inf),
        ("routed_model_heff_covariance_residual", 0.0),
    ],
)
def test_gamma_producer_integrity_direct_constructor_fails_closed(
    field: str,
    value: object,
) -> None:
    thresholds = {
        "source_hamiltonian_covariance_residual": 0.0,
        "routed_model_heff_covariance_residual": 1.0e-10,
    }
    thresholds[field] = value

    with pytest.raises(ValueError, match="producer integrity threshold"):
        GammaProducerIntegrityThresholds(**thresholds)  # type: ignore[arg-type]


def test_gamma_producer_integrity_accepts_zero_source_and_rejects_unknown_key() -> None:
    thresholds = GammaProducerIntegrityThresholds(
        source_hamiltonian_covariance_residual=0.0,
        routed_model_heff_covariance_residual=1.0e-10,
    )
    assert thresholds.source_hamiltonian_covariance_residual == 0.0

    unknown = _config_payload()
    unknown["producer_integrity_thresholds"][  # type: ignore[index]
        "unexpected"
    ] = 1.0
    with pytest.raises(ValueError, match="field mismatch.*unknown"):
        GammaAutomaticSelectionConfig.from_normalized_config(unknown)


def test_gamma_auto_integrity_gates_are_independent_and_policy_bound() -> None:
    payloads = []
    for field, value in (
        ("source_hamiltonian_covariance_residual", 2.0e-10),
        ("routed_model_heff_covariance_residual", 3.0e-10),
    ):
        payload = _config_payload()
        payload["producer_integrity_thresholds"][field] = value  # type: ignore[index]
        payloads.append(payload)

    baseline = prepare_gamma_automatic_selection(
        _producer_inputs(),
        GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
    ).selection_input.selection_policy_hash
    changed = {
        prepare_gamma_automatic_selection(
            _producer_inputs(),
            GammaAutomaticSelectionConfig.from_normalized_config(payload),
        ).selection_input.selection_policy_hash
        for payload in payloads
    }

    assert len(changed) == 2
    assert baseline not in changed


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


def _dual_frame_q2_case(
    method: str,
    *,
    source_covariance_tolerance: float = 1.0e-10,
    candidate_covariance_tolerance: float = 1.0e-10,
    routed_model_covariance_tolerance: float = 1.0e-10,
) -> tuple[GammaAutomaticProducerInputs, GammaAutomaticSelectionConfig]:
    payload = _config_payload()
    payload["producer_integrity_thresholds"] = {
        "source_hamiltonian_covariance_residual": source_covariance_tolerance,
        "routed_model_heff_covariance_residual": routed_model_covariance_tolerance,
    }
    payload["candidate_symmetry_thresholds"][  # type: ignore[index]
        "heff_covariance_residual"
    ] = candidate_covariance_tolerance
    payload["selection_thresholds"] = {
        "band_rms_mev": 1.0e3,
        "band_max_mev": 1.0e3,
        "subspace_overlap": 1.0e-12,
        "symmetry_residual": 1.0e-9,
        "symmetry_leakage": 1.0e-9,
    }
    payload["target_window"] = {
        "edge": "valence",
        "band_count": 4,
        "validation_k_indices": [0, 1],
        "energy_reference_ev": 0.0,
        "degeneracy_tolerance_mev": 1.0e-6,
    }
    payload["downfold"] = {
        **payload["downfold"],  # type: ignore[dict-item]
        "method": method,
        "e_ref": None if method == "first_order" else -1.4,
        "fail_on_near_pole": False,
        "compute_pole_diagnostics": False,
    }
    qsets = (
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
    )
    basis_hash = hash_array(np.arange(8, dtype=np.int64))
    layout = GammaRowLayout.build(
        qsets=qsets,
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash=basis_hash,
    )
    hamiltonians: list[np.ndarray] = []
    for k_position in range(2):
        hamiltonian = np.zeros((layout.full_dimension, layout.full_dimension), dtype=np.complex128)
        for q_index, angle in enumerate(
            (0.20 + 0.03 * k_position, -0.27 + 0.02 * k_position)
        ):
            rotation = np.eye(layout.same_q_dimension, dtype=np.complex128)
            rotation[:2, :2] = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            local_values = np.asarray(
                [-2.2, -1.2, 1.1, 2.1]
                if q_index == 0
                else [-1.9, -0.9, 1.3, 2.3],
                dtype=np.float64,
            )
            local_values += 0.04 * k_position
            rows = layout.same_q_full_rows(q_index)
            hamiltonian[np.ix_(rows, rows)] = (
                rotation @ np.diag(local_values) @ rotation.conj().T
            )
        q0_rows = layout.same_q_full_rows(0)
        q1_rows = layout.same_q_full_rows(1)
        q_coupling = np.zeros(
            (layout.same_q_dimension, layout.same_q_dimension),
            dtype=np.complex128,
        )
        q_coupling[0, 2] = 0.12 + 0.01j
        q_coupling[1, 3] = -0.09 + 0.02j
        q_coupling[2, 0] = 0.07 - 0.01j
        q_coupling[3, 1] = -0.05
        hamiltonian[np.ix_(q0_rows, q1_rows)] = q_coupling
        hamiltonian[np.ix_(q1_rows, q0_rows)] = q_coupling.conj().T
        hamiltonians.append(hamiltonian)
    return (
        GammaAutomaticProducerInputs(
            source_hamiltonians=np.stack(hamiltonians, axis=0),
            k_indices=(0, 1),
            kpoints=np.asarray([[0.0, 0.0], [0.25, 0.0]], dtype=np.float64),
            qsets=qsets,
            num_layer_list=(1, 1),
            num_orb_per_layer_list=((1,), (1,)),
            tapw_source_basis_hash=basis_hash,
            operations=(
                GammaRawOperationSpec(
                    name="E",
                    full_action=np.eye(layout.full_dimension, dtype=np.complex128),
                    antiunitary=False,
                    q_permutations=((0, 1), (0, 1)),
                    sector_map=(0, 1),
                    pairs=((0, 0), (1, 1)),
                ),
            ),
            presentation=_presentation(),
        ),
        GammaAutomaticSelectionConfig.from_normalized_config(payload),
    )


def test_gamma_auto_producer_hands_off_common_anchor_model_frame() -> None:
    inputs = _producer_inputs()
    hamiltonians = []
    for angle in (0.2, -0.35):
        rotation = np.eye(4, dtype=np.complex128)
        rotation[:2, :2] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        hamiltonians.append(
            rotation
            @ np.diag([-2.0, -1.0, 1.0, 2.0])
            @ rotation.conj().T
        )
    mixed_inputs = GammaAutomaticProducerInputs(
        **{
            **inputs.__dict__,
            "source_hamiltonians": np.stack(hamiltonians, axis=0),
        }
    )

    result = produce_gamma_automatic_selection(
        mixed_inputs,
        GammaAutomaticSelectionConfig.from_normalized_config(_config_payload()),
    )

    for position, k_index in enumerate(result.handoff.k_indices):
        model = result.handoff.assemble_model_for_k(k_index)
        state_frame, state_heff = result.handoff.model_state_for_k(k_index)
        np.testing.assert_allclose(
            state_frame,
            model,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(
            state_heff,
            result.handoff.authoritative_heff_for_k(k_index),
        )
        np.testing.assert_allclose(
            state_heff,
            model.conj().T
            @ mixed_inputs.source_hamiltonians[position]
            @ model,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model.conj().T @ model,
            np.eye(result.handoff.model_dim),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_gamma_auto_model_builder_reuses_prepared_local_eigensystems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, config = _dual_frame_q2_case("first_order")
    preparation = prepare_gamma_automatic_selection(inputs, config)
    original_anchor = producer_mod.build_gamma_common_anchor_spec
    original_frames = producer_mod.build_gamma_common_anchor_frames
    anchor_calls = 0
    frame_positions: list[int] = []

    def anchor_spy(**kwargs):
        nonlocal anchor_calls
        anchor_calls += 1
        reference_position = inputs.k_indices.index(config.reference_k_index)
        assert kwargs["reference_eigenvalues_by_q"] is preparation.local_values[
            reference_position
        ]
        assert kwargs["reference_eigenvectors_by_q"] is preparation.local_vectors[
            reference_position
        ]
        return original_anchor(**kwargs)

    def frames_spy(**kwargs):
        position = next(
            position
            for position, values in enumerate(preparation.local_values)
            if kwargs["eigenvalues_by_q"] is values
        )
        assert kwargs["eigenvectors_by_q"] is preparation.local_vectors[position]
        frame_positions.append(position)
        return original_frames(**kwargs)

    monkeypatch.setattr(
        producer_mod,
        "build_gamma_common_anchor_spec",
        anchor_spy,
    )
    monkeypatch.setattr(producer_mod, "build_gamma_common_anchor_frames", frames_spy)
    monkeypatch.setattr(
        producer_mod,
        "_local_eigensystems",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate evaluation must reuse prepared local eigensystems")
        ),
    )

    evaluate_gamma_automatic_selection(preparation)

    assert anchor_calls == 1
    assert frame_positions == list(range(len(inputs.k_indices)))


def test_gamma_auto_candidate_and_artifact_identity_bind_model_gauge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, config = _dual_frame_q2_case("first_order")
    original_identity = producer_mod.build_projection_basis_identity
    identity_calls: list[dict[str, object]] = []

    def identity_spy(**kwargs):
        identity_calls.append(dict(kwargs))
        return original_identity(**kwargs)

    monkeypatch.setattr(producer_mod, "build_projection_basis_identity", identity_spy)

    result = produce_gamma_automatic_selection(inputs, config)

    assert len(identity_calls) == 1
    identity_call = identity_calls[0]
    assert identity_call["gauge_mode"] == "auto_common_anchor"
    assert identity_call["gauge_frame_hash"] == hash_array(result.handoff.local_frames)
    assert (
        identity_call["resolved_norb_fix_list"]
        == result.handoff.anchor_spec.to_payload()["resolved_reference_terms"]
    )
    state_records = result.evaluations[0].symmetry_certificate.input_identity_payload[
        "states"
    ]
    for state in state_records:
        k_index = int(state["k_index"])
        model, model_heff = result.handoff.model_state_for_k(k_index)
        assert state["present"] is True
        assert state["u_low_hash"] == hash_array(model)
        assert state["heff_hash"] == hash_array(model_heff)


def test_real_gamma_auto_producer_builds_all_k_handoff_and_certified_identity() -> None:
    config = GammaAutomaticSelectionConfig.from_normalized_config(_config_payload())
    inputs = _producer_inputs()

    result = produce_gamma_automatic_selection(inputs, config)

    assert result.decision.selected == result.candidates[0]
    assert result.rejected_candidates == ()
    assert result.handoff.k_indices == (0, 1)
    assert result.handoff.authoritative_heff.shape[0] == 2
    np.testing.assert_array_equal(result.handoff.kpoints, inputs.kpoints)
    assert result.handoff.kpoints_hash == hash_array(inputs.kpoints)
    assert result.handoff.anchor_spec.joint_band_indices == (0, 1)
    assert result.handoff.candidate_id == result.decision.selected.candidate_id
    assert result.handoff.source_hamiltonian_hash == hash_array(
        inputs.source_hamiltonians
    )
    assert result.handoff.candidate_certificate_hash != "0" * 64
    assert result.selection_input.action_package_hash == result.handoff.action_package_hash
    assert result.selection_input.ordered_q_hash == gamma_common_anchor_ordered_q_identity_hash(
        result.handoff
    )
    identity = build_certified_gamma_selection_identity(
        selection_input=result.selection_input,
        handoff=result.handoff,
        metrics=result.decision.selected,
    )
    assert identity.resolved_candidate.candidate_id == result.handoff.candidate_id
    for k_index in result.handoff.k_indices:
        u_low = result.handoff.assemble_model_for_k(k_index)
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


def test_gamma_preselection_identity_is_invariant_to_target_worker_count() -> None:
    inputs = _producer_inputs()
    config = GammaAutomaticSelectionConfig.from_normalized_config(_config_payload())

    serial = prepare_gamma_automatic_selection(inputs, config, workers=1)
    parallel = prepare_gamma_automatic_selection(inputs, config, workers=2)

    assert serial.selection_input == parallel.selection_input
    assert serial.source_hamiltonian_hash == parallel.source_hamiltonian_hash
    np.testing.assert_array_equal(serial.target_values, parallel.target_values)
    assert not hasattr(serial, "target_vectors")
    assert not hasattr(parallel, "target_vectors")


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
    assert result.handoff.model_group_ranks == (2, 2)
    assert result.handoff.model_dim == 8
    assert result.handoff.anchor_spec.joint_band_indices == (0, 1, 2, 3)
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
    assert result.handoff.anchor_spec.joint_band_indices == (0, 1)


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

    permissive_source = _config_payload()
    permissive_source["producer_integrity_thresholds"][  # type: ignore[index]
        "source_hamiltonian_covariance_residual"
    ] = 10.0
    permissive_source["candidate_symmetry_thresholds"][  # type: ignore[index]
        "heff_covariance_residual"
    ] = 0.0
    prepare_gamma_automatic_selection(
        cross_k,
        GammaAutomaticSelectionConfig.from_normalized_config(permissive_source),
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
