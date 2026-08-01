from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from kp.symmetry.candidate_certificate import (
    CandidateOperationInput,
    CandidateProjectionState,
    CandidateSymmetryStatus,
    CandidateSymmetryThresholds,
    certify_candidate_symmetries,
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

    dimension = 4096
    sparse_action = NoDenseCSR(sparse.eye(dimension, format="csr"))
    state = _state(np.eye(dimension, 1))
    sparse_result = _certify(
        name="E",
        state=state,
        d_full=sparse_action,
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
    assert "joint relation certification failed" in result.joint_certification_failure


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
