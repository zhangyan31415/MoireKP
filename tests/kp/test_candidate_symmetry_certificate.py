from __future__ import annotations

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
    d_full: np.ndarray,
    exact: np.ndarray | None,
    antiunitary: bool = False,
    phase: complex = 1.0,
    pairs: tuple[tuple[int, int], ...] = ((0, 0),),
    required_pairs: tuple[tuple[int, int], ...] = ((0, 0),),
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
        thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
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


def test_sparse_raw_h_action_uses_the_same_in_memory_projection_path() -> None:
    result = _certify(
        name="E",
        state=_state(np.eye(2)),
        d_full=sparse.csr_matrix(np.eye(2)),
        exact=np.eye(2),
    )

    assert result.status is CandidateSymmetryStatus.CERTIFIED


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
