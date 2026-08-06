from __future__ import annotations

from pathlib import Path

import numpy as np

from kp.identity import hash_mapping
from kp.low_energy_selection import CandidateMetrics, SelectionThresholds
from kp.projection_handoff import ExplicitLegacyBasisSpec
from kp.selection_artifact import (
    CertificationStatus,
    ResolvedCandidateIdentity,
    SelectionArtifactStore,
    SelectionBindingError,
    SelectionInputIdentity,
    verify_certified_generic_selection_artifact,
)
from kp.selection_orchestration import (
    CaseSelectionInputs,
    begin_case_selection,
    resolve_case_selection,
)
from kp.symmetry import projection as projection_mod


def _digest(label: str) -> str:
    return hash_mapping({"label": label})


def _pending_generic_selection(
    project_dir: Path,
) -> tuple[ExplicitLegacyBasisSpec, object]:
    qset1_hash = _digest("qset-1")
    qset2_hash = _digest("qset-2")
    selection_input = SelectionInputIdentity.create(
        selection_mode="auto",
        frozen_target_window_hash=_digest("target"),
        validation_k_indices_hash=_digest("validation-k"),
        ordered_q_hash=hash_mapping(
            {
                "schema": "kp.cli-ordered-q.v1",
                "qset1_hash": qset1_hash,
                "qset2_hash": qset2_hash,
            }
        ),
        source_hamiltonian_hash=_digest("source-h"),
        action_package_hash=_digest("actions"),
        row_layout_hash=_digest("layout"),
        selection_policy_hash=_digest("policy"),
    )
    metrics = CandidateMetrics(
        candidate_id="k-dim004",
        dimension=4,
        band_rms_mev=0.1,
        band_max_mev=0.2,
        subspace_overlap=0.9,
        symmetry_residual=None,
        symmetry_leakage=None,
    )
    artifact_identity = {
        "basis_hash": _digest("basis"),
        "heff_hash": _digest("heff"),
        "k_indices_hash": _digest("k-indices"),
    }
    handoff = ExplicitLegacyBasisSpec(
        artifact_identity=artifact_identity,
        nlow_state_list=[[], [4, 5]],
        resolved_norb_fix_list=[[], [[0, 1]]],
        gauge_mode="auto_scdm",
        frame_artifact=None,
        model_dim=4,
    )
    resolved = ResolvedCandidateIdentity.create(
        selection_mode="auto",
        candidate_id=metrics.candidate_id,
        candidate_dimension=metrics.dimension,
        projection_basis_kind=handoff.projection_basis_kind,
        basis_handoff_hash=artifact_identity["basis_hash"],
        authoritative_heff_hash=artifact_identity["heff_hash"],
        heff_k_indices_hash=artifact_identity["k_indices_hash"],
    )
    pending_payload = {
        "schema": "kp.non-gamma-post-selection-symmetry-pending.v2",
        "status": "pending",
        "candidate_id": metrics.candidate_id,
        "candidate_dimension": metrics.dimension,
        "nlow_state_list": [[], [4, 5]],
        "mode": "k",
        "spin": "all",
        "qset1_hash": qset1_hash,
        "qset2_hash": qset2_hash,
        "selection_input_identity_hash": selection_input.selection_input_identity_hash,
        "resolved_candidate_hash": resolved.resolved_candidate_hash,
        "basis_handoff_hash": resolved.basis_handoff_hash,
    }
    input_hash = hash_mapping(
        {
            "schema": "kp.non-gamma-post-selection-symmetry-input.v2",
            **{
                key: value
                for key, value in pending_payload.items()
                if key not in {"schema", "status"}
            },
        }
    )
    certificate_hash = hash_mapping(
        {
            "schema": "kp.non-gamma-post-selection-symmetry-certificate.v2",
            "status": "pending",
            "input_identity_hash": input_hash,
        }
    )
    store = SelectionArtifactStore(project_dir)
    session = begin_case_selection(
        store=store,
        selection_input=selection_input,
        transaction_id="generic-pending-symmetry",
    )
    try:
        result = resolve_case_selection(
            CaseSelectionInputs.generic_auto(
                selection_input=selection_input,
                candidates=(metrics,),
                thresholds=SelectionThresholds(1.0, 1.0, 0.5, 0.1, 0.1),
                resolved_candidate=resolved,
                certificate_hash=certificate_hash,
                certificate_input_identity_hash=input_hash,
                pending_symmetry_payload=pending_payload,
                payloads={"basis.npz": b"basis"},
                allow_pending_symmetry=True,
            ),
            session=session,
        )
    finally:
        session.close()
    assert result.status is CertificationStatus.PENDING_SYMMETRY
    return handoff, result.artifact


def test_kp_symm_promotes_generic_pending_selection_with_actual_evidence(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projection"
    handoff, pending = _pending_generic_selection(project_dir)
    binding = projection_mod._load_current_generic_selection_binding(
        project_dir,
        handoff,
    )
    raw = np.eye(handoff.model_dim, dtype=np.complex128)
    exact = raw.copy()
    summary = {
        "artifact_identity": dict(handoff.artifact_identity),
        "operations": [
            {
                "name": "C3z",
                "antiunitary": False,
                "status": "exactified",
                "exactification_status": "exactified",
                "pairs": [
                    {
                        "raw": {
                            "d_unitarity_error": 2.0e-8,
                            "subspace_leakage": 3.0e-8,
                            "heff_covariance_residual": 4.0e-8,
                        },
                        "full_space_covariance_residual": 5.0e-8,
                    }
                ],
            }
        ],
    }

    certified = projection_mod._certify_pending_generic_selection(
        binding,
        summary=summary,
        raw_low_matrices={"C3z": raw},
        exact_matrices={"C3z": exact},
    )

    assert certified is not None
    assert certified.certification_status is CertificationStatus.CERTIFIED
    assert certified.metrics is not None
    assert certified.metrics.symmetry_residual == 5.0e-8
    assert certified.metrics.symmetry_leakage == 3.0e-8
    assert pending.identity is not None
    assert certified.identity is not None
    assert certified.identity.selection_identity_hash != pending.identity.selection_identity_hash
    assert summary["selection_identity_hash"] == certified.identity.selection_identity_hash
    assert summary["operations"][0]["selection_identity_hash"] == (
        certified.identity.selection_identity_hash
    )

    binding.store.promote_pending_symmetry(certified)
    assert binding.store.load_current(require_certified=True) == certified
    assert (
        verify_certified_generic_selection_artifact(
            certified,
            artifact_identity=handoff.artifact_identity,
            candidate_dimension=handoff.model_dim,
        )
        is certified
    )

    mismatched_identity = {
        **handoff.artifact_identity,
        "basis_hash": _digest("different-basis"),
    }
    try:
        verify_certified_generic_selection_artifact(
            certified,
            artifact_identity=mismatched_identity,
            candidate_dimension=handoff.model_dim,
        )
    except SelectionBindingError as error:
        assert "basis_handoff_hash" in str(error)
    else:  # pragma: no cover - regression guard
        raise AssertionError("generic selection accepted a different project basis")
