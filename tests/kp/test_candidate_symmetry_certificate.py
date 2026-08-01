from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from kp.symmetry.candidate_certificate import (
    CANDIDATE_SYMMETRY_CERTIFICATE_SCHEMA_VERSION,
    CANDIDATE_SYMMETRY_METRIC_HASH_SCHEMA_VERSION,
    CANDIDATE_SYMMETRY_METRIC_THRESHOLD_FIELDS,
    CANDIDATE_SYMMETRY_PHASE_HASH_SCHEMA_VERSION,
    CandidateJointFailureCode,
    CandidateOperationInput,
    CandidateProjectionState,
    CandidateStateFailureCode,
    CandidateSymmetryStatus,
    CandidateSymmetryThresholds,
    candidate_certificate_envelope,
    certify_candidate_symmetries,
    evaluate_projected_pair,
    verify_candidate_certificate_envelope,
)
from kp.symmetry.joint_exactification import (
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
)


def _presentation(
    name: str,
    *,
    antiunitary: bool = False,
    phase: complex = 1.0,
) -> MagneticPresentation:
    return MagneticPresentation(
        generators=(MagneticGenerator(name, antiunitary),),
        relations=(
            MagneticRelation(
                f"{name}^2",
                lhs=(name, name),
                rhs=(),
                central_phase=phase,
            ),
        ),
        central_phases=(1.0,) if phase == 1.0 else (1.0, -1.0),
        source="synthetic_candidate_certificate",
    )


def _state(u_low: np.ndarray, heff: np.ndarray | None = None) -> CandidateProjectionState:
    columns = int(np.asarray(u_low).shape[1])
    return CandidateProjectionState(
        u_low=np.asarray(u_low, dtype=np.complex128),
        heff=(
            np.eye(columns, dtype=np.complex128)
            if heff is None
            else np.asarray(heff, dtype=np.complex128)
        ),
    )


def _certify(
    *,
    name: str,
    state: CandidateProjectionState,
    d_full: object,
    exact: np.ndarray | None,
    antiunitary: bool = False,
    phase: complex = 1.0,
    pairs: tuple[tuple[int, int], ...] = ((0, 0),),
    required_pairs: tuple[tuple[int, int], ...] = ((0, 0),),
    thresholds: CandidateSymmetryThresholds | None = None,
):
    return certify_candidate_symmetries(
        candidate_id="synthetic",
        states={0: state},
        operations={
            name: CandidateOperationInput(
                name=name,
                antiunitary=antiunitary,
                d_full=d_full,
                pairs=pairs,
            )
        },
        exactified_actions={} if exact is None else {name: exact},
        presentation=_presentation(name, antiunitary=antiunitary, phase=phase),
        required_pairs={name: required_pairs},
        thresholds=(
            CandidateSymmetryThresholds.uniform(1.0e-10)
            if thresholds is None
            else thresholds
        ),
    )


def test_certifies_identity_action_and_records_every_metric() -> None:
    result = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.eye(2),
        exact=np.eye(2),
    )

    assert result.status is CandidateSymmetryStatus.CERTIFIED
    assert result.operation_coverage_complete is True
    assert result.pair_coverage_complete is True
    operation = result.operations[0]
    pair = operation.pairs[0]
    assert pair.raw_h_leakage == pytest.approx(0.0)
    assert pair.exactification_distance == pytest.approx(0.0)
    assert pair.intertwining_residual == pytest.approx(0.0)
    assert pair.heff_covariance_residual == pytest.approx(0.0)
    assert result.relations[0].residual == pytest.approx(0.0)
    assert len(result.certificate_hash) == 64
    assert result.thresholds == CandidateSymmetryThresholds.uniform(1.0e-10)
    assert operation.exact_action_unitarity_residual == pytest.approx(0.0)
    assert result.joint_certification_status == "certified"
    assert result.joint_certification_failure is None


def test_rejects_candidate_with_raw_h_subspace_leakage() -> None:
    angle = 0.2
    d_full = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.complex128,
    )
    result = _certify(
        name="E",
        state=_state(np.asarray([[1.0], [0.0]])),
        d_full=d_full,
        exact=np.eye(1),
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    pair = result.operations[0].pairs[0]
    assert pair.raw_h_leakage == pytest.approx(np.sin(angle))
    assert "raw_h_leakage" in pair.failures


def test_rejects_exact_action_that_does_not_intertwine_raw_action() -> None:
    result = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        exact=np.eye(2),
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    pair = result.operations[0].pairs[0]
    assert pair.raw_h_leakage == pytest.approx(0.0)
    assert pair.exactification_distance == pytest.approx(np.sqrt(2.0))
    assert pair.intertwining_residual == pytest.approx(np.sqrt(2.0))
    assert {"exactification_distance", "intertwining_residual"} <= set(pair.failures)


def test_certifies_spinful_time_reversal_and_records_square_residual() -> None:
    time_reversal = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    result = _certify(
        name="TR",
        state=_state(np.eye(2)),
        d_full=time_reversal,
        exact=time_reversal,
        antiunitary=True,
        phase=-1.0,
    )

    assert result.status is CandidateSymmetryStatus.CERTIFIED
    operation = result.operations[0]
    assert operation.antiunitary_square_phase == -1.0
    assert operation.antiunitary_square_residual == pytest.approx(0.0)
    assert result.relations[0].residual == pytest.approx(0.0)


def test_rejects_missing_operation_or_pair_coverage() -> None:
    missing_operation = certify_candidate_symmetries(
        candidate_id="missing-operation",
        states={0: _state(np.eye(1))},
        operations={},
        exactified_actions={},
        presentation=_presentation("E"),
        required_pairs={"E": ((0, 0),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )
    assert missing_operation.status is CandidateSymmetryStatus.FAILED
    assert missing_operation.operation_coverage_complete is False
    assert "missing_operation:E" in missing_operation.failures

    missing_pair = _certify(
        name="E",
        state=_state(np.eye(1)),
        d_full=np.eye(1),
        exact=np.eye(1),
        pairs=((0, 0),),
        required_pairs=((0, 0), (1, 1)),
    )
    assert missing_pair.status is CandidateSymmetryStatus.FAILED
    assert missing_pair.pair_coverage_complete is False
    assert "missing_pair:E:1<-1" in missing_pair.failures


@pytest.mark.parametrize(
    "exact",
    (
        pytest.param(np.asarray([[np.nan]], dtype=np.complex128), id="nan"),
        pytest.param(np.asarray([[np.inf]], dtype=np.complex128), id="inf"),
        pytest.param(None, id="none"),
    ),
)
def test_nonfinite_exact_action_fails_closed_instead_of_becoming_zero(
    exact: np.ndarray | None,
) -> None:
    result = _certify(
        name="E",
        state=_state(np.eye(1)),
        d_full=np.eye(1),
        exact=exact,
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    assert result.operations[0].exact_action_finite is False
    assert result.operations[0].pairs == ()
    assert "nonfinite_exactified_action:E" in result.failures


def test_sparse_raw_h_action_is_not_materialized_and_hashes_like_dense() -> None:
    class NoDenseCSR(sparse.csr_matrix):
        def toarray(self, *args, **kwargs):
            raise AssertionError("full raw-H action must remain sparse")

    dimension = 100_000
    sparse_action = NoDenseCSR(sparse.eye(dimension, format="csr"))
    state = _state(np.eye(dimension, 1))
    sparse_result = _certify(
        name="E",
        state=state,
        d_full=sparse_action,
        exact=np.eye(1),
    )

    assert sparse_result.status is CandidateSymmetryStatus.CERTIFIED


def test_small_dense_and_sparse_raw_actions_have_identical_identity_and_hash() -> None:
    dimension = 8
    state = _state(np.eye(dimension, 1))
    sparse_result = _certify(
        name="E",
        state=state,
        d_full=sparse.eye(dimension, format="csr"),
        exact=np.eye(1),
    )
    dense_result = _certify(
        name="E",
        state=state,
        d_full=np.eye(dimension),
        exact=np.eye(1),
    )

    assert sparse_result.status is CandidateSymmetryStatus.CERTIFIED
    assert sparse_result == dense_result
    assert sparse_result.input_identity_hash == dense_result.input_identity_hash
    assert sparse_result.certificate_hash == dense_result.certificate_hash


def test_nontrivial_unitary_dense_and_sparse_have_identical_certificate_hash() -> None:
    rng = np.random.default_rng(20260801)
    trial = rng.normal(size=(8, 8)) + 1.0j * rng.normal(size=(8, 8))
    gauge, _ = np.linalg.qr(trial)
    signs = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    action = gauge @ signs @ gauge.conj().T
    state = _state(np.eye(8), np.zeros((8, 8)))

    dense_result = _certify(
        name="E",
        state=state,
        d_full=action,
        exact=action,
    )
    sparse_result = _certify(
        name="E",
        state=state,
        d_full=sparse.csr_matrix(action),
        exact=action,
    )

    assert dense_result.status is CandidateSymmetryStatus.CERTIFIED
    assert sparse_result.status is CandidateSymmetryStatus.CERTIFIED
    assert dense_result.input_identity_hash == sparse_result.input_identity_hash
    assert (
        dense_result.operations[0].raw_h_action_unitarity_residual
        != sparse_result.operations[0].raw_h_action_unitarity_residual
    )
    assert dense_result.certificate_hash == sparse_result.certificate_hash


def test_sparse_input_identity_sorts_coordinates_sums_duplicates_and_drops_zeros() -> None:
    sparse_action = sparse.coo_matrix(
        (
            np.asarray([1.0, 0.25, 0.75, 0.0, 0.0]),
            (
                np.asarray([1, 0, 0, 1, 0]),
                np.asarray([1, 0, 0, 0, 1]),
            ),
        ),
        shape=(2, 2),
    )
    state = _state(np.eye(2))
    sparse_result = _certify(
        name="E",
        state=state,
        d_full=sparse_action,
        exact=np.eye(2),
    )
    dense_result = _certify(
        name="E",
        state=state,
        d_full=np.eye(2),
        exact=np.eye(2),
    )

    assert sparse_result.input_identity_hash == dense_result.input_identity_hash
    assert sparse_result.certificate_hash == dense_result.certificate_hash


def test_certificate_hash_is_stable_under_mapping_and_pair_order() -> None:
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("A", False), MagneticGenerator("B", False)),
        relations=(
            MagneticRelation("A^2", lhs=("A", "A"), rhs=(), central_phase=1.0),
            MagneticRelation("B^2", lhs=("B", "B"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="synthetic_hash_order",
    )
    state = _state(np.eye(1))

    def certify(reverse: bool):
        names = ("B", "A") if reverse else ("A", "B")
        pairs = ((1, 1), (0, 0)) if reverse else ((0, 0), (1, 1))
        return certify_candidate_symmetries(
            candidate_id="hash-order",
            states={1: state, 0: state} if reverse else {0: state, 1: state},
            operations={
                name: CandidateOperationInput(
                    name=name,
                    antiunitary=False,
                    d_full=np.eye(1),
                    pairs=pairs,
                )
                for name in names
            },
            exactified_actions={name: np.eye(1) for name in names},
            presentation=presentation,
            required_pairs={name: pairs for name in names},
            thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
        )

    forward = certify(False)
    reverse = certify(True)
    assert forward.status is CandidateSymmetryStatus.CERTIFIED
    assert reverse.status is CandidateSymmetryStatus.CERTIFIED
    assert forward.certificate_hash == reverse.certificate_hash


def test_presentation_and_input_identity_bind_full_physical_contract() -> None:
    baseline = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.eye(2),
        exact=np.eye(2),
    )
    gauged_state = _state(np.diag([1.0, 1.0j]))
    changed_state = _certify(
        name="E",
        state=gauged_state,
        d_full=np.eye(2),
        exact=np.eye(2),
    )
    changed_action = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        exact=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    )
    fourth_power = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation(
                "E^4",
                lhs=("E", "E", "E", "E"),
                rhs=(),
                central_phase=1.0,
            ),
        ),
        central_phases=(1.0,),
        source="different-description",
    )
    changed_presentation = certify_candidate_symmetries(
        candidate_id="synthetic",
        states={0: _state(np.eye(2))},
        operations={
            "E": CandidateOperationInput(
                name="E",
                antiunitary=False,
                d_full=np.eye(2),
                pairs=((0, 0),),
            )
        },
        exactified_actions={"E": np.eye(2)},
        presentation=fourth_power,
        required_pairs={"E": ((0, 0),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )
    changed_phase_presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation(
                "E^2",
                lhs=("E", "E"),
                rhs=(),
                central_phase=-1.0,
            ),
        ),
        central_phases=(1.0, -1.0),
        source="phase-change",
    )
    changed_phase = certify_candidate_symmetries(
        candidate_id="synthetic",
        states={0: _state(np.eye(2))},
        operations={
            "E": CandidateOperationInput("E", False, np.eye(2), ((0, 0),))
        },
        exactified_actions={"E": np.eye(2)},
        presentation=changed_phase_presentation,
        required_pairs={"E": ((0, 0),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    assert baseline.presentation_payload["version"]
    assert len(baseline.presentation_hash) == 64
    assert len(baseline.input_identity_hash) == 64
    assert baseline.input_identity_hash != changed_state.input_identity_hash
    assert baseline.input_identity_hash != changed_action.input_identity_hash
    assert baseline.presentation_hash != changed_presentation.presentation_hash
    assert baseline.presentation_hash != changed_phase.presentation_hash
    assert baseline.certificate_hash != changed_state.certificate_hash
    assert baseline.certificate_hash != changed_action.certificate_hash
    assert baseline.certificate_hash != changed_presentation.certificate_hash
    assert baseline.certificate_hash != changed_phase.certificate_hash


def test_presentation_source_description_is_explicitly_not_identity_bound() -> None:
    first = _presentation("E")
    second = MagneticPresentation(
        generators=first.generators,
        relations=first.relations,
        central_phases=first.central_phases,
        source="another-description-only-source",
    )

    def certify(presentation: MagneticPresentation):
        return certify_candidate_symmetries(
            candidate_id="presentation-source",
            states={0: _state(np.eye(1))},
            operations={
                "E": CandidateOperationInput("E", False, np.eye(1), ((0, 0),))
            },
            exactified_actions={"E": np.eye(1)},
            presentation=presentation,
            required_pairs={"E": ((0, 0),)},
            thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
        )

    assert certify(first).presentation_hash == certify(second).presentation_hash


def test_presentation_payload_property_is_a_defensive_serializable_copy() -> None:
    certificate = _certify(
        name="E",
        state=_state(np.eye(1)),
        d_full=np.eye(1),
        exact=np.eye(1),
    )
    original_payload = certificate.presentation_payload
    original_hash = certificate.certificate_hash
    exposed = certificate.presentation_payload

    exposed["version"] = "ordinary-mutation"
    exposed["relations"][0]["name"] = "nested-mutation"
    exposed["relations"].append({"name": "extra"})
    dict.__setitem__(exposed, "base-class-attack", True)
    dict.__setitem__(exposed["generators"][0], "name", "base-class-nested")
    dict.__init__(exposed, {"reinitialized": True})

    assert certificate.presentation_payload == original_payload
    assert certificate.presentation_payload is not certificate.presentation_payload
    assert certificate.presentation_payload_copy() == original_payload
    assert certificate.certificate_hash == original_hash
    assert json.loads(json.dumps(certificate.presentation_payload)) == original_payload
    restored = pickle.loads(pickle.dumps(certificate))
    assert restored == certificate
    assert restored.presentation_payload == original_payload
    assert restored.certificate_hash == original_hash


def test_candidate_certificate_envelope_roundtrips_full_canonical_input() -> None:
    certificate = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.eye(2),
        exact=np.eye(2),
    )

    envelope = candidate_certificate_envelope(certificate)
    verified = verify_candidate_certificate_envelope(envelope)

    assert verified["candidate_id"] == certificate.candidate_id
    assert verified["status"] == "certified"
    assert verified["certificate_hash"] == certificate.certificate_hash
    assert verified["input_identity_hash"] == certificate.input_identity_hash
    assert verified["input_identity_payload"]["states"][0]["u_low_hash"]
    assert verified["input_identity_payload"]["operations"][0]["raw_action_hash"]


@pytest.mark.parametrize(
    "tamper",
    ["status", "certificate_hash", "input_identity_payload"],
)
def test_candidate_certificate_envelope_rejects_tamper(tamper: str) -> None:
    certificate = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.eye(2),
        exact=np.eye(2),
    )
    envelope = candidate_certificate_envelope(certificate)
    if tamper == "status":
        envelope["certificate_payload"]["status"] = "failed"
    elif tamper == "certificate_hash":
        envelope["certificate_hash"] = "0" * 64
    else:
        envelope["input_identity_payload"]["states"][0]["u_low_hash"] = "0" * 64

    with pytest.raises(ValueError, match="candidate symmetry certificate envelope"):
        verify_candidate_certificate_envelope(envelope)


def test_certificate_hash_binds_metrics_and_thresholds() -> None:
    baseline = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.eye(2),
        exact=np.eye(2),
        thresholds=CandidateSymmetryThresholds.uniform(1.0),
    )
    angle = 0.2
    changed_metric = _certify(
        name="E",
        state=_state(np.asarray([[1.0], [0.0]])),
        d_full=np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        ),
        exact=np.eye(1),
        thresholds=CandidateSymmetryThresholds.uniform(1.0),
    )
    changed_threshold = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.eye(2),
        exact=np.eye(2),
        thresholds=CandidateSymmetryThresholds.uniform(2.0),
    )

    assert baseline.certificate_hash != changed_metric.certificate_hash
    assert baseline.certificate_hash != changed_threshold.certificate_hash


def test_metric_hash_canonicalization_absorbs_roundoff_but_not_gate_scale_changes() -> None:
    baseline = _certify(
        name="E",
        state=_state(np.eye(1)),
        d_full=np.eye(1),
        exact=np.eye(1),
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-13),
    )
    operation = baseline.operations[0]
    roundoff_a = replace(
        baseline,
        operations=(
            replace(operation, raw_h_action_unitarity_residual=2.0e-15),
        ),
    )
    roundoff_b = replace(
        baseline,
        operations=(
            replace(operation, raw_h_action_unitarity_residual=3.0e-15),
        ),
    )
    above_a = replace(
        baseline,
        operations=(
            replace(operation, raw_h_action_unitarity_residual=2.0e-13),
        ),
    )
    above_b = replace(
        baseline,
        operations=(
            replace(operation, raw_h_action_unitarity_residual=3.0e-13),
        ),
    )

    assert roundoff_a.certificate_hash == roundoff_b.certificate_hash
    assert above_a.certificate_hash != above_b.certificate_hash


def test_derived_complex_phase_hash_absorbs_roundoff_but_preserves_changes() -> None:
    baseline = _certify(
        name="TR",
        state=_state(np.eye(2)),
        d_full=np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        exact=np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        antiunitary=True,
        phase=-1.0,
    )
    operation = baseline.operations[0]
    roundoff_a = replace(
        baseline,
        operations=(replace(operation, antiunitary_square_phase=-1.0 + 2.0e-15j),),
    )
    roundoff_b = replace(
        baseline,
        operations=(replace(operation, antiunitary_square_phase=-1.0 + 3.0e-15j),),
    )
    changed_a = replace(
        baseline,
        operations=(replace(operation, antiunitary_square_phase=-1.0 + 2.0e-10j),),
    )
    changed_b = replace(
        baseline,
        operations=(replace(operation, antiunitary_square_phase=-1.0 + 3.0e-10j),),
    )

    assert roundoff_a.certificate_hash == roundoff_b.certificate_hash
    assert changed_a.certificate_hash != changed_b.certificate_hash


def test_certificate_and_metric_hash_schema_versions_are_explicit() -> None:
    assert (
        CANDIDATE_SYMMETRY_CERTIFICATE_SCHEMA_VERSION
        == "candidate_symmetry_certificate_v3"
    )
    assert (
        CANDIDATE_SYMMETRY_METRIC_HASH_SCHEMA_VERSION
        == "candidate_symmetry_metric_hash_v1"
    )
    assert (
        CANDIDATE_SYMMETRY_PHASE_HASH_SCHEMA_VERSION
        == "candidate_symmetry_phase_hash_v1"
    )
    assert dict(CANDIDATE_SYMMETRY_METRIC_THRESHOLD_FIELDS) == {
        "certificate.relation_residual_max": "relation_residual",
        "operation.antiunitary_square_residual": "antiunitary_square_residual",
        "operation.exact_action_unitarity_residual": (
            "exact_action_unitarity_residual"
        ),
        "operation.raw_h_action_unitarity_residual": (
            "raw_h_action_unitarity_residual"
        ),
        "pair.exactification_distance": "exactification_distance",
        "pair.heff_covariance_residual": "heff_covariance_residual",
        "pair.intertwining_residual": "intertwining_residual",
        "pair.raw_action_unitarity_residual": (
            "raw_h_action_unitarity_residual"
        ),
        "pair.raw_h_leakage": "raw_h_leakage",
        "relation.maximum_entry": "relation_residual",
        "relation.residual": "relation_residual",
        "state.heff_hermiticity_residual": "heff_hermiticity_residual",
        "state.projection_orthonormality_residual": (
            "projection_orthonormality_residual"
        ),
    }


def test_certificate_hash_binds_each_typed_metric_and_threshold() -> None:
    baseline = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=np.eye(2),
        exact=np.eye(2),
    )
    operation = baseline.operations[0]
    pair = operation.pairs[0]
    for metric_name in (
        "raw_h_leakage",
        "raw_action_unitarity_residual",
        "exactification_distance",
        "intertwining_residual",
        "heff_covariance_residual",
    ):
        changed_pair = replace(pair, **{metric_name: 1.0e-12})
        changed = replace(
            baseline,
            operations=(replace(operation, pairs=(changed_pair,)),),
        )
        assert changed.certificate_hash != baseline.certificate_hash, metric_name

    changed_unitarity = replace(
        baseline,
        operations=(
            replace(operation, exact_action_unitarity_residual=1.0e-12),
        ),
    )
    assert changed_unitarity.certificate_hash != baseline.certificate_hash
    changed_raw_unitarity = replace(
        baseline,
        operations=(
            replace(operation, raw_h_action_unitarity_residual=1.0e-12),
        ),
    )
    assert changed_raw_unitarity.certificate_hash != baseline.certificate_hash

    state = baseline.states[0]
    for metric_name in (
        "projection_orthonormality_residual",
        "heff_hermiticity_residual",
    ):
        changed_states = tuple(
            replace(item, **{metric_name: 1.0e-12})
            if item == state
            else item
            for item in baseline.states
        )
        changed = replace(baseline, states=changed_states)
        assert changed.certificate_hash != baseline.certificate_hash, metric_name

    relation = baseline.relations[0]
    for metric_name in ("residual", "maximum_entry"):
        changed = replace(
            baseline,
            relations=(replace(relation, **{metric_name: 1.0e-12}),),
        )
        assert changed.certificate_hash != baseline.certificate_hash, metric_name

    tr = _certify(
        name="TR",
        state=_state(np.eye(2)),
        d_full=np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        exact=np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        antiunitary=True,
        phase=-1.0,
    )
    changed_square = replace(
        tr,
        operations=(
            replace(tr.operations[0], antiunitary_square_residual=1.0e-12),
        ),
    )
    assert changed_square.certificate_hash != tr.certificate_hash
    changed_phase = replace(
        tr,
        operations=(
            replace(tr.operations[0], antiunitary_square_phase=1.0),
        ),
    )
    assert changed_phase.certificate_hash != tr.certificate_hash

    for threshold_field in CandidateSymmetryThresholds.__dataclass_fields__:
        changed_thresholds = replace(
            baseline.thresholds,
            **{
                threshold_field: getattr(baseline.thresholds, threshold_field)
                + 1.0e-12
            },
        )
        changed = replace(baseline, thresholds=changed_thresholds)
        assert changed.certificate_hash != baseline.certificate_hash, threshold_field

    assert replace(
        baseline,
        candidate_id="different",
    ).certificate_hash != baseline.certificate_hash
    assert replace(
        baseline,
        status=CandidateSymmetryStatus.FAILED,
    ).certificate_hash != baseline.certificate_hash
    assert replace(
        baseline,
        failures=("changed",),
    ).certificate_hash != baseline.certificate_hash
    ordered_failures = replace(baseline, failures=("alpha", "beta"))
    reversed_failures = replace(baseline, failures=("beta", "alpha"))
    assert ordered_failures.certificate_hash == reversed_failures.certificate_hash


def test_small_leaky_candidate_fails_while_closed_larger_candidate_certifies() -> None:
    raw_action = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    small = _certify(
        name="E",
        state=_state(np.asarray([[1.0], [0.0]])),
        d_full=raw_action,
        exact=np.eye(1),
    )
    larger = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=raw_action,
        exact=raw_action,
    )

    assert small.status is CandidateSymmetryStatus.FAILED
    assert small.operations[0].pairs[0].raw_h_leakage == pytest.approx(1.0)
    assert larger.status is CandidateSymmetryStatus.CERTIFIED
    assert larger.operations[0].pairs[0].raw_h_leakage == pytest.approx(0.0)


def test_nonorthonormal_projection_and_nonunitary_raw_action_cannot_cancel() -> None:
    source_state = _state(np.asarray([[2.0]]), np.eye(1))
    target_state = _state(np.asarray([[1.0]]), np.eye(1))
    result = certify_candidate_symmetries(
        candidate_id="invalid-cancellation",
        states={},
        source_states={1: source_state},
        target_states={0: target_state},
        operations={
            "E": CandidateOperationInput(
                name="E",
                antiunitary=False,
                d_full=np.asarray([[0.5]]),
                pairs=((0, 1),),
            )
        },
        exactified_actions={"E": np.eye(1)},
        presentation=_presentation("E"),
        required_pairs={"E": ((0, 1),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    source_certificate = next(
        state for state in result.states if state.role == "source"
    )
    operation = result.operations[0]
    assert result.status is CandidateSymmetryStatus.FAILED
    assert source_certificate.projection_orthonormality_residual == pytest.approx(3.0)
    assert "projection_orthonormality_residual" in source_certificate.failures
    assert operation.raw_h_action_unitarity_residual == pytest.approx(0.75)
    assert "raw_h_action_unitarity_residual" in operation.failures


def test_nonhermitian_heff_fails_with_typed_state_residual() -> None:
    nonhermitian = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    result = _certify(
        name="E",
        state=_state(np.eye(2), nonhermitian),
        d_full=np.eye(2),
        exact=np.eye(2),
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    assert result.states
    assert result.states[0].heff_hermiticity_residual == pytest.approx(np.sqrt(2.0))
    assert "heff_hermiticity_residual" in result.states[0].failures


def test_empty_required_and_observed_pair_coverage_fails_closed() -> None:
    result = certify_candidate_symmetries(
        candidate_id="empty-pairs",
        states={},
        operations={
            "E": CandidateOperationInput(
                name="E",
                antiunitary=False,
                d_full=np.eye(1),
                pairs=(),
            )
        },
        exactified_actions={"E": np.eye(1)},
        presentation=_presentation("E"),
        required_pairs={"E": ()},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    assert result.pair_coverage_complete is False
    assert "empty_pair_coverage:E" in result.failures
    assert "empty_pair_coverage" in result.operations[0].failures


def test_evaluated_unexpected_pair_does_not_count_as_required_pair_coverage() -> None:
    result = certify_candidate_symmetries(
        candidate_id="unexpected-only-pair",
        states={0: _state(np.eye(1)), 1: _state(np.eye(1))},
        operations={
            "E": CandidateOperationInput("E", False, np.eye(1), ((1, 1),))
        },
        exactified_actions={"E": np.eye(1)},
        presentation=_presentation("E"),
        required_pairs={"E": ((0, 0),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    assert result.pair_coverage_complete is False
    assert "no_evaluated_required_pair" in result.operations[0].failures


def test_required_only_pair_states_are_identity_bound_and_typed() -> None:
    def certify(state_zero: CandidateProjectionState):
        return certify_candidate_symmetries(
            candidate_id="required-only-state",
            states={0: state_zero, 1: _state(np.eye(1))},
            operations={
                "E": CandidateOperationInput("E", False, np.eye(1), ((1, 1),))
            },
            exactified_actions={"E": np.eye(1)},
            presentation=_presentation("E"),
            required_pairs={"E": ((0, 0),)},
            thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
        )

    baseline = certify(_state(np.eye(1)))
    changed = certify(_state(np.asarray([[1.0j]])))

    assert {(state.role, state.k_index) for state in baseline.states} == {
        ("source", 0),
        ("source", 1),
        ("target", 0),
        ("target", 1),
    }
    assert baseline.input_identity_hash != changed.input_identity_hash
    assert baseline.certificate_hash != changed.certificate_hash


def test_missing_required_only_state_has_stable_typed_reason() -> None:
    result = certify_candidate_symmetries(
        candidate_id="missing-required-only-state",
        states={1: _state(np.eye(1))},
        operations={
            "E": CandidateOperationInput("E", False, np.eye(1), ((1, 1),))
        },
        exactified_actions={"E": np.eye(1)},
        presentation=_presentation("E"),
        required_pairs={"E": ((0, 0),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    required_only = {
        (state.role, state.k_index): state for state in result.states
    }
    assert required_only[("source", 0)].failures == (
        CandidateStateFailureCode.MISSING_PROJECTION_STATE,
    )
    assert required_only[("target", 0)].failures == (
        CandidateStateFailureCode.MISSING_PROJECTION_STATE,
    )


def test_projected_pair_exposes_projected_and_evaluated_actions_separately() -> None:
    evaluation = evaluate_projected_pair(
        d_full=np.eye(1),
        target_u_low=np.eye(1),
        source_u_low=np.eye(1),
        target_heff=np.zeros((1, 1)),
        source_heff=np.zeros((1, 1)),
        antiunitary=False,
        action=np.asarray([[2.0]]),
    )

    np.testing.assert_array_equal(evaluation.projected_action, np.eye(1))
    np.testing.assert_array_equal(evaluation.evaluated_action, np.asarray([[2.0]]))
    assert not hasattr(evaluation, "action")


def test_complex_antiunitary_conjugates_basis_and_heff() -> None:
    root_two = np.sqrt(2.0)
    source_basis = np.asarray([[1.0, 1.0j], [1.0j, 1.0]]) / root_two
    target_basis = np.diag([np.exp(0.3j), np.exp(-0.2j)])
    exact = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    raw_action = target_basis @ exact @ source_basis.T
    source_heff = np.asarray([[1.0, 0.3j], [-0.3j, 2.0]], dtype=np.complex128)
    target_heff = exact @ source_heff.conj() @ exact.conj().T
    source_state = _state(source_basis, source_heff)
    target_state = _state(target_basis, target_heff)

    result = certify_candidate_symmetries(
        candidate_id="complex-antiunitary",
        states={},
        source_states={1: source_state},
        target_states={0: target_state},
        operations={
            "TR": CandidateOperationInput(
                name="TR",
                antiunitary=True,
                d_full=raw_action,
                pairs=((0, 1),),
            )
        },
        exactified_actions={"TR": exact},
        presentation=_presentation("TR", antiunitary=True, phase=-1.0),
        required_pairs={"TR": ((0, 1),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    without_basis_conjugation = target_basis.conj().T @ raw_action @ source_basis
    without_heff_conjugation = exact @ source_heff @ exact.conj().T
    assert np.linalg.norm(without_basis_conjugation - exact) > 0.1
    assert np.linalg.norm(without_heff_conjugation - target_heff) > 0.1
    assert result.status is CandidateSymmetryStatus.CERTIFIED
    assert result.operations[0].pairs[0].heff_covariance_residual == pytest.approx(0.0)


def test_exact_action_unitarity_has_typed_metric_and_independent_gate() -> None:
    rng = np.random.default_rng(123)
    trial = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    gauge, _ = np.linalg.qr(trial)
    exact = gauge @ np.diag([1.0, -1.0]) @ gauge.conj().T
    thresholds = CandidateSymmetryThresholds(
        raw_h_leakage=1.0e-10,
        exactification_distance=1.0e-10,
        intertwining_residual=1.0e-10,
        heff_covariance_residual=1.0e-10,
        relation_residual=1.0e-10,
        antiunitary_square_residual=1.0e-10,
        exact_action_unitarity_residual=0.0,
        projection_orthonormality_residual=1.0e-10,
        heff_hermiticity_residual=1.0e-10,
        raw_h_action_unitarity_residual=1.0e-10,
    )
    result = _certify(
        name="E",
        state=_state(np.eye(2), np.zeros((2, 2))),
        d_full=exact,
        exact=exact,
        thresholds=thresholds,
    )

    operation = result.operations[0]
    assert operation.exact_action_unitarity_residual > 0.0
    assert "exact_action_unitarity_residual" in operation.failures
    assert result.status is CandidateSymmetryStatus.FAILED
    assert result.joint_certification_status == "certified"
    assert result.joint_certification_failure is None


def test_overflowed_metrics_become_failed_nulls_not_nan_hash_payloads() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _certify(
            name="E",
            state=_state(np.eye(1)),
            d_full=np.asarray([[1.0e308]], dtype=np.complex128),
            exact=np.eye(1),
            thresholds=CandidateSymmetryThresholds.uniform(1.0e307),
        )

    pair = result.operations[0].pairs[0]
    assert result.status is CandidateSymmetryStatus.FAILED
    assert pair.raw_action_unitarity_residual is None
    assert pair.exactification_distance is None
    assert "nonfinite_raw_action_unitarity_residual" in pair.failures
    assert "exactification_distance" in pair.failures
    assert len(result.certificate_hash) == 64


def test_relation_mismatch_is_recorded_and_fails_closed() -> None:
    exact = np.diag([1.0, 1.0j]).astype(np.complex128)
    result = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=exact,
        exact=exact,
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    assert result.relations[0].name == "E^2"
    assert result.relations[0].residual == pytest.approx(np.sqrt(2.0))
    assert "relation:E^2" in result.failures
    assert result.joint_certification_status == "failed"
    assert (
        result.joint_certification_failure
        is CandidateJointFailureCode.RELATION_CERTIFICATION_FAILED
    )
    assert "joint relation certification failed" in result.joint_certification_diagnostic
    assert all("residual=" not in failure for failure in result.failures)
    changed_diagnostic = replace(
        result,
        joint_certification_diagnostic="platform-specific diagnostic changed",
    )
    assert changed_diagnostic.certificate_hash == result.certificate_hash
    changed_code = replace(
        result,
        joint_certification_failure=CandidateJointFailureCode.ACTION_NOT_UNITARY,
    )
    assert changed_code.certificate_hash != result.certificate_hash


def test_strict_joint_relation_failure_has_relation_code_after_user_gate_passes() -> None:
    phases = np.diag(np.exp(1.0j * np.asarray([1.0e-8, -1.0e-8])))
    result = _certify(
        name="E",
        state=_state(np.eye(2), np.zeros((2, 2))),
        d_full=phases,
        exact=phases,
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-6),
    )

    assert result.relations[0].passed is True
    assert result.relations[0].residual == pytest.approx(2.0e-8)
    assert result.joint_certification_status == "failed"
    assert (
        result.joint_certification_failure
        is CandidateJointFailureCode.RELATION_CERTIFICATION_FAILED
    )


def test_strict_joint_action_construction_failure_has_unitarity_code() -> None:
    nonunitary = np.diag([1.0 + 1.0e-8, 1.0]).astype(np.complex128)
    result = _certify(
        name="E",
        state=_state(np.eye(2), np.zeros((2, 2))),
        d_full=np.eye(2),
        exact=nonunitary,
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-6),
    )

    assert result.relations[0].passed is True
    assert result.joint_certification_status == "failed"
    assert (
        result.joint_certification_failure
        is CandidateJointFailureCode.ACTION_NOT_UNITARY
    )


def test_duplicate_pair_does_not_satisfy_exact_coverage() -> None:
    result = _certify(
        name="E",
        state=_state(np.eye(1)),
        d_full=np.eye(1),
        exact=np.eye(1),
        pairs=((0, 0), (0, 0)),
        required_pairs=((0, 0),),
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    assert result.pair_coverage_complete is False
    assert "duplicate_pair:E:0<-0" in result.failures


def test_inconsistent_exact_action_dimensions_fail_closed() -> None:
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("A", False), MagneticGenerator("B", False)),
        relations=(
            MagneticRelation("A^2", lhs=("A", "A"), rhs=(), central_phase=1.0),
            MagneticRelation("B^2", lhs=("B", "B"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="synthetic_dimension_mismatch",
    )
    result = certify_candidate_symmetries(
        candidate_id="dimension-mismatch",
        states={0: _state(np.eye(1))},
        operations={
            name: CandidateOperationInput(
                name=name,
                antiunitary=False,
                d_full=np.eye(1),
                pairs=((0, 0),),
            )
            for name in ("A", "B")
        },
        exactified_actions={"A": np.eye(1), "B": np.eye(2)},
        presentation=presentation,
        required_pairs={"A": ((0, 0),), "B": ((0, 0),)},
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )

    assert result.status is CandidateSymmetryStatus.FAILED
    assert "joint_action_dimension_mismatch" in result.failures
