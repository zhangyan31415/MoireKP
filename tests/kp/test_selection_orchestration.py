from __future__ import annotations

from pathlib import Path

import numpy as np

from kp.blocks import GammaRoutingThresholds, GammaRowLayout, build_gamma_routed_frames
from kp.identity import hash_mapping
from kp.low_energy_selection import CandidateMetrics, SelectionThresholds
from kp.projection_handoff import GammaRoutedBasisSpec, certify_gamma_routed_basis_spec
from kp.selection_artifact import (
    CertificationStatus,
    ResolvedCandidateIdentity,
    SelectionArtifactStore,
    SelectionInputIdentity,
    gamma_routed_ordered_q_identity_hash,
)
from kp.selection_orchestration import (
    CaseSelectionInputs,
    begin_case_selection,
    preview_case_selection,
    resolve_case_selection,
)
from kp.symmetry.candidate_certificate import (
    CandidateOperationInput,
    CandidateSymmetryThresholds,
)
from kp.symmetry.joint_exactification import (
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
)


def _digest(label: str) -> str:
    return hash_mapping({"label": label})


def _routing_thresholds() -> GammaRoutingThresholds:
    return GammaRoutingThresholds(
        energy_same_ev=1.0e-8,
        energy_different_ev=1.0e-5,
        capture_zero_fraction=1.0e-8,
        capture_loss_max=1.0e-8,
        local_action_isometry=1.0e-10,
        off_route_leakage=1.0e-10,
        closure_residual=1.0e-10,
        route_zero_gap=1.0e-8,
        route_covariance=1.0e-10,
        projector_residual=1.0e-10,
        anchor_sigma_min=1.0e-8,
        max_rank=4,
        max_iterations=8,
    )


def _gamma_handoff() -> GammaRoutedBasisSpec:
    layout = GammaRowLayout.build(
        qsets=(np.zeros((1, 2)), np.zeros((1, 2))),
        num_layer_list=(1, 1),
        num_orb_per_layer_list=((1,), (1,)),
        spin_convention="all",
        source_basis_hash="source-basis",
    )
    thresholds = _routing_thresholds()
    routed = build_gamma_routed_frames(
        (np.arange(4, dtype=float),),
        (np.eye(4, dtype=np.complex128),),
        joint_band_indices=(0, 1),
        layout=layout,
        thresholds=thresholds,
        require_complete_clusters=False,
    )
    k_indices = (0, 1)
    heff = np.asarray(
        [np.diag([0.25, 0.75]), np.diag([0.5, 1.0])],
        dtype=np.complex128,
    )
    pairs = ((0, 0), (1, 1))
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("E", False),),
        relations=(
            MagneticRelation("E^2", lhs=("E", "E"), rhs=(), central_phase=1.0),
        ),
        central_phases=(1.0,),
        source="selection_orchestration_test",
    )
    return certify_gamma_routed_basis_spec(
        candidate_id="gamma-two",
        artifact_identity={
            "identity_schema": "moirekp.artifact-identity.v1",
            "input_hash": _digest("input"),
            "config_hash": _digest("config"),
            "basis_hash": _digest("base-basis"),
            "package_version": "0.1.0",
            "schema_version": 2,
            "k_indices_hash": "replaced",
            "heff_hash": "replaced",
        },
        layout=layout,
        thresholds=thresholds,
        k_indices=k_indices,
        routed_frames=(routed, routed),
        authoritative_heff=heff,
        heff_k_indices=k_indices,
        closure_certificate_hashes=(_digest("closure-0"), _digest("closure-1")),
        routing_certificate_hashes=(_digest("route-0"), _digest("route-1")),
        source_hamiltonian_hash=_digest("source-h"),
        ordered_q_hashes=layout.ordered_qset_hashes,
        raw_action_package_hash="0" * 64,
        operations={
            "E": CandidateOperationInput(
                name="E",
                antiunitary=False,
                d_full=np.eye(layout.full_dimension, dtype=np.complex128),
                pairs=pairs,
            )
        },
        exactified_actions={"E": np.eye(2, dtype=np.complex128)},
        presentation=presentation,
        required_pairs={"E": pairs},
        candidate_thresholds=CandidateSymmetryThresholds.uniform(1.0e-10),
    )


def _selection_input(handoff: GammaRoutedBasisSpec, *, mode: str) -> SelectionInputIdentity:
    return SelectionInputIdentity.create(
        selection_mode=mode,
        frozen_target_window_hash=_digest("target"),
        validation_k_indices_hash=_digest("validation-k"),
        ordered_q_hash=gamma_routed_ordered_q_identity_hash(handoff),
        source_hamiltonian_hash=handoff.source_hamiltonian_hash,
        action_package_hash=handoff.raw_action_package_hash,
        row_layout_hash=handoff.layout.layout_hash,
        selection_policy_hash=_digest("policy"),
    )


def _metrics(handoff: GammaRoutedBasisSpec) -> CandidateMetrics:
    return CandidateMetrics(
        candidate_id=handoff.candidate_id,
        dimension=handoff.model_dim,
        band_rms_mev=0.1,
        band_max_mev=0.2,
        subspace_overlap=0.999,
        symmetry_residual=1.0e-10,
        symmetry_leakage=2.0e-10,
        structural_failure=None,
    )


def _thresholds() -> SelectionThresholds:
    return SelectionThresholds(1.0, 2.0, 0.99, 1.0e-8, 1.0e-8)


def test_auto_resolution_certifies_only_real_gamma_handoff(tmp_path: Path) -> None:
    handoff = _gamma_handoff()
    store = SelectionArtifactStore(tmp_path / "projection")
    session = begin_case_selection(
        store=store,
        selection_input=_selection_input(handoff, mode="auto"),
        transaction_id="auto-a",
    )

    result = resolve_case_selection(
        CaseSelectionInputs.auto(
            selection_input=_selection_input(handoff, mode="auto"),
            candidates=(_metrics(handoff),),
            thresholds=_thresholds(),
            projection_handoff=handoff,
            payloads={"basis.npz": b"real-routed-handoff"},
        ),
        session=session,
    )

    assert result.status is CertificationStatus.CERTIFIED
    assert result.identity is not None
    assert store.load_current(require_certified=True) == result.artifact


def test_auto_failure_replaces_previous_certified_marker(tmp_path: Path) -> None:
    handoff = _gamma_handoff()
    store = SelectionArtifactStore(tmp_path / "projection")
    first = begin_case_selection(
        store=store,
        selection_input=_selection_input(handoff, mode="auto"),
        transaction_id="auto-good",
    )
    resolve_case_selection(
        CaseSelectionInputs.auto(
            selection_input=_selection_input(handoff, mode="auto"),
            candidates=(_metrics(handoff),),
            thresholds=_thresholds(),
            projection_handoff=handoff,
            payloads={"basis.npz": b"real-routed-handoff"},
        ),
        session=first,
    )

    second = begin_case_selection(
        store=store,
        selection_input=_selection_input(handoff, mode="auto"),
        transaction_id="auto-bad",
    )
    failed = resolve_case_selection(
        CaseSelectionInputs.auto(
            selection_input=_selection_input(handoff, mode="auto"),
            candidates=(_metrics(handoff),),
            thresholds=_thresholds(),
            projection_handoff=None,
            payloads={},
        ),
        session=second,
    )

    assert failed.status is CertificationStatus.FAILED
    assert store.load_current().certification_status is CertificationStatus.FAILED


def test_explicit_and_active_index_resolution_is_never_certified(tmp_path: Path) -> None:
    handoff = _gamma_handoff()
    selection_input = _selection_input(handoff, mode="explicit")
    resolved = ResolvedCandidateIdentity.create(
        selection_mode="explicit",
        candidate_id="active-indices-0-1",
        candidate_dimension=2,
        projection_basis_kind="explicit_legacy",
        basis_handoff_hash=_digest("basis"),
        authoritative_heff_hash=_digest("heff"),
        heff_k_indices_hash=_digest("k-map"),
    )
    store = SelectionArtifactStore(tmp_path / "projection")
    session = begin_case_selection(
        store=store,
        selection_input=selection_input,
        transaction_id="explicit-a",
    )

    result = resolve_case_selection(
        CaseSelectionInputs.explicit(
            selection_input=selection_input,
            resolved_candidate=resolved,
            diagnostic="user supplied --active-indices",
        ),
        session=session,
    )

    assert result.status is CertificationStatus.UNVERIFIED_OVERRIDE
    assert result.identity is not None
    assert result.identity.certification_evidence is None


def test_inspect_preview_is_honest_pending_without_final_identity() -> None:
    handoff = _gamma_handoff()

    preview = preview_case_selection(_selection_input(handoff, mode="auto"))

    assert preview.status is CertificationStatus.PENDING
    assert preview.identity is None
    assert preview.artifact is None
