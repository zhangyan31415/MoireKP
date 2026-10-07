"""Regression boundaries used by the final-model diagnostics audit."""
import numpy as np
from kp.symmetry.candidate_certificate import (
    CandidateOperationInput, CandidateProjectionState, CandidateSymmetryStatus,
    CandidateSymmetryThresholds, certify_candidate_symmetries,
)
from kp.symmetry.joint_exactification import MagneticGenerator, MagneticPresentation, MagneticRelation


def test_group_exact_action_does_not_certify_non_covariant_hamiltonian():
    """An exact involution still must pass the independent Heff covariance gate."""
    swap=np.array([[0.,1.],[1.,0.]],dtype=complex)
    presentation=MagneticPresentation(
        generators=(MagneticGenerator('X',False),),
        relations=(MagneticRelation('X^2',lhs=('X','X'),rhs=(),central_phase=1.),),
        central_phases=(1.,), source='revision_small_matrix_regression')
    result=certify_candidate_symmetries(
        candidate_id='closed_but_not_covariant',
        states={0:CandidateProjectionState(u_low=np.eye(2,dtype=complex),heff=np.diag([0.,1.]).astype(complex))},
        operations={'X':CandidateOperationInput(name='X',antiunitary=False,d_full=swap,pairs=((0,0),))},
        exactified_actions={'X':swap},presentation=presentation,
        required_pairs={'X':((0,0),)},thresholds=CandidateSymmetryThresholds.uniform(1.e-10))
    assert result.status is not CandidateSymmetryStatus.CERTIFIED
    assert result.relations[0].residual < 1.e-12
    pair=result.operations[0].pairs[0]
    assert pair.raw_h_leakage < 1.e-12
    assert pair.heff_covariance_residual > 1.
