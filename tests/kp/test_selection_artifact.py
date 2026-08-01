from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from kp.identity import hash_mapping
from kp.low_energy_selection import CandidateMetrics
from kp.projection_selection import FrozenTargetWindow, TargetWindowSpec
from kp.selection_artifact import (
    CertificationEvidence,
    CertificationStatus,
    ResolvedCandidateIdentity,
    SelectionArtifact,
    SelectionArtifactStore,
    SelectionFailureCode,
    SelectionIdentity,
    SelectionInputIdentity,
    SelectionMetricEvidence,
    SelectionTransactionError,
    build_selection_input_identity_hash,
    build_selection_policy_hash,
    hash_frozen_target_window,
    hash_validation_k_indices,
    load_selection_artifact,
)


def _digest(label: str) -> str:
    return hash_mapping({"label": label})


def _metrics(candidate_id: str = "candidate-4") -> CandidateMetrics:
    return CandidateMetrics(
        candidate_id=candidate_id,
        dimension=4,
        band_rms_mev=0.25,
        band_max_mev=0.75,
        subspace_overlap=0.99,
        symmetry_residual=1.0e-9,
        symmetry_leakage=2.0e-9,
        structural_failure=None,
    )


def _resolved(*, mode: str = "auto", heff: str = "heff") -> ResolvedCandidateIdentity:
    return ResolvedCandidateIdentity.create(
        selection_mode=mode,
        candidate_id="candidate-4",
        candidate_dimension=4,
        projection_basis_kind="fixed_window",
        basis_handoff_hash=_digest("basis"),
        authoritative_heff_hash=_digest(heff),
        heff_k_indices_hash=_digest("k-map"),
    )


def _metric_evidence() -> SelectionMetricEvidence:
    return SelectionMetricEvidence.create(
        candidate_id="candidate-4",
        frozen_target_window_hash=_digest("target"),
        validation_k_indices_hash=_digest("validation-k"),
        basis_handoff_hash=_digest("basis"),
        metrics=_metrics(),
    )


def _certification_evidence() -> CertificationEvidence:
    return CertificationEvidence.create(
        metric_evidence=_metric_evidence(),
        symmetry_certificate_hash=_digest("candidate-certificate"),
        symmetry_input_identity_hash=_digest("candidate-input"),
    )


def _selection_input(**changes: str) -> SelectionInputIdentity:
    values = {
        "selection_mode": "auto",
        "frozen_target_window_hash": _digest("target"),
        "ordered_q_hash": _digest("q"),
        "source_hamiltonian_hash": _digest("source-h"),
        "action_package_hash": _digest("actions"),
        "row_layout_hash": _digest("layout"),
        "selection_policy_hash": _digest("policy"),
    }
    values.update(changes)
    return SelectionInputIdentity.create(**values)


def _selection_input_hash(**changes: str) -> str:
    return _selection_input(**changes).selection_input_identity_hash


def _identity(
    *,
    mode: str = "auto",
    with_evidence: bool = True,
    selection_input: SelectionInputIdentity | None = None,
) -> SelectionIdentity:
    input_identity = (
        _selection_input(selection_mode=mode)
        if selection_input is None
        else selection_input
    )
    return SelectionIdentity.create(
        selection_input=input_identity,
        selection_policy_hash=_digest("policy"),
        resolved_candidate=_resolved(mode=mode),
        certification_evidence=(
            _certification_evidence() if with_evidence else None
        ),
    )


def _certified(transaction_id: str = "transaction-a") -> SelectionArtifact:
    identity = _identity()
    return SelectionArtifact.certified(
        transaction_id=transaction_id,
        identity=identity,
    )


def test_frozen_target_hash_binds_complete_resolved_window() -> None:
    spec = TargetWindowSpec(
        edge="valence",
        band_count=2,
        validation_k_indices=(0, 3),
        energy_reference_ev=0.0,
        degeneracy_tolerance_mev=1.0e-3,
    )
    baseline = FrozenTargetWindow(
        spec=spec,
        target_band_ids=((2, 3), (4, 5)),
        target_energies_ev=((-0.2, -0.1), (-0.3, -0.15)),
    )
    changed_ids = replace(baseline, target_band_ids=((1, 3), (4, 5)))
    changed_energies = replace(
        baseline,
        target_energies_ev=((-0.21, -0.1), (-0.3, -0.15)),
    )
    changed_reference = replace(
        baseline,
        spec=replace(spec, energy_reference_ev=0.1),
    )

    assert hash_frozen_target_window(baseline) == hash_frozen_target_window(baseline)
    assert hash_frozen_target_window(changed_ids) != hash_frozen_target_window(baseline)
    assert hash_frozen_target_window(changed_energies) != hash_frozen_target_window(baseline)
    assert hash_frozen_target_window(changed_reference) != hash_frozen_target_window(baseline)
    baseline_k_hash = hash_validation_k_indices(baseline)
    changed_k_hash = hash_validation_k_indices(
        replace(
            baseline,
            spec=replace(spec, validation_k_indices=(1, 3)),
        )
    )
    assert len(baseline_k_hash) == 64
    assert hash_validation_k_indices(baseline) == baseline_k_hash
    assert changed_k_hash != baseline_k_hash


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("frozen_target_window_hash", "different-target"),
        ("ordered_q_hash", "different-q"),
        ("source_hamiltonian_hash", "different-source"),
        ("action_package_hash", "different-actions"),
        ("row_layout_hash", "different-layout"),
        ("selection_policy_hash", "different-policy"),
    ),
)
def test_selection_input_identity_binds_every_physical_input(
    field: str,
    changed: str,
) -> None:
    baseline = _selection_input_hash()

    assert _selection_input_hash(**{field: _digest(changed)}) != baseline


def test_resolved_metric_and_certification_evidence_are_frozen_and_self_hashed() -> None:
    resolved = _resolved()
    metric_evidence = _metric_evidence()
    evidence = _certification_evidence()

    assert len(resolved.resolved_candidate_hash) == 64
    assert len(metric_evidence.metrics_hash) == 64
    assert len(evidence.evidence_hash) == 64
    with pytest.raises(FrozenInstanceError):
        resolved.candidate_id = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(resolved, resolved_candidate_hash=_digest("tampered"))
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(metric_evidence, metrics_hash=_digest("tampered"))
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(evidence, evidence_hash=_digest("tampered"))


@pytest.mark.parametrize(
    "metric",
    (
        "band_rms_mev",
        "band_max_mev",
        "subspace_overlap",
        "symmetry_residual",
        "symmetry_leakage",
    ),
)
def test_metric_evidence_rejects_nonfinite_metric(metric: str) -> None:
    values = _metrics().__dict__ | {metric: float("nan")}

    with pytest.raises(ValueError, match="finite"):
        SelectionMetricEvidence.create(
            candidate_id="candidate-4",
            frozen_target_window_hash=_digest("target"),
            validation_k_indices_hash=_digest("validation-k"),
            basis_handoff_hash=_digest("basis"),
            metrics=CandidateMetrics(**values),
        )


def test_metric_evidence_requires_candidate_and_basis_identity_match() -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        SelectionMetricEvidence.create(
            candidate_id="different",
            frozen_target_window_hash=_digest("target"),
            validation_k_indices_hash=_digest("validation-k"),
            basis_handoff_hash=_digest("basis"),
            metrics=_metrics(),
        )

    evidence = _certification_evidence()
    with pytest.raises(ValueError, match="candidate"):
        SelectionIdentity.create(
            selection_input=_selection_input(),
            selection_policy_hash=_digest("policy"),
            resolved_candidate=ResolvedCandidateIdentity.create(
                selection_mode="auto",
                candidate_id="different",
                candidate_dimension=4,
                projection_basis_kind="fixed_window",
                basis_handoff_hash=_digest("basis"),
                authoritative_heff_hash=_digest("heff"),
                heff_k_indices_hash=_digest("k-map"),
            ),
            certification_evidence=evidence,
        )


def test_explicit_override_binds_actual_candidate_heff_and_k_map_without_metrics() -> None:
    identity = _identity(mode="explicit", with_evidence=False)
    artifact = SelectionArtifact.unverified_override(
        transaction_id="explicit-transaction",
        identity=identity,
        diagnostic="Explicit bands requested by the user",
    )

    assert artifact.certification_status is CertificationStatus.UNVERIFIED_OVERRIDE
    assert artifact.status_reason is SelectionFailureCode.EXPLICIT_UNVERIFIED
    assert artifact.failure_codes == (SelectionFailureCode.EXPLICIT_UNVERIFIED,)
    assert artifact.metrics is None
    assert artifact.identity is not None
    assert artifact.identity.resolved_candidate.authoritative_heff_hash == _digest("heff")
    assert artifact.identity.resolved_candidate.heff_k_indices_hash == _digest("k-map")
    assert artifact.identity.certification_evidence is None


def test_certified_artifact_requires_complete_evidence_and_single_metrics_source() -> None:
    artifact = _certified()

    assert artifact.certification_status is CertificationStatus.CERTIFIED
    assert artifact.metrics == artifact.identity.certification_evidence.metric_evidence.metrics
    assert artifact.failure_codes == ()
    assert artifact.status_reason is None
    assert len(artifact.artifact_hash) == 64

    with pytest.raises(ValueError, match="certification evidence"):
        SelectionArtifact.certified(
            transaction_id="incomplete",
            identity=_identity(with_evidence=False),
        )


def test_certified_artifact_rejects_explicit_mode_and_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="automatic"):
        SelectionArtifact.certified(
            transaction_id="explicit-cannot-pass",
            identity=_identity(mode="explicit"),
        )

    identity = _identity(
        selection_input=_selection_input(source_hamiltonian_hash=_digest("different-input"))
    )
    with pytest.raises(ValueError, match="input identity"):
        SelectionArtifact(
            schema_version="kp.selection-artifact.v1",
            transaction_id="bad-input",
            selection_input_identity_hash=_selection_input_hash(),
            identity=identity,
            metrics=identity.certification_evidence.metric_evidence.metrics,
            certification_status=CertificationStatus.CERTIFIED,
            status_reason=None,
            failure_codes=(),
            diagnostic=None,
            artifact_hash=_digest("irrelevant"),
        )


def test_failed_and_pending_artifacts_cannot_claim_identity_or_metrics() -> None:
    pending = SelectionArtifact.pending(
        transaction_id="pending",
        selection_input_identity_hash=_selection_input_hash(),
    )
    failed = SelectionArtifact.failed(
        transaction_id="failed",
        selection_input_identity_hash=_selection_input_hash(),
        reason=SelectionFailureCode.HARD_METRIC_FAILED,
        failure_codes=(SelectionFailureCode.HARD_METRIC_FAILED,),
        diagnostic="No candidate passed",
    )

    assert pending.identity is None and pending.metrics is None
    assert failed.identity is None and failed.metrics is None
    assert failed.certification_status is CertificationStatus.FAILED
    assert failed.status_reason is SelectionFailureCode.HARD_METRIC_FAILED


def test_strict_json_roundtrip_rejects_tampering_and_nonfinite_values(tmp_path: Path) -> None:
    artifact = _certified()
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(artifact.to_dict(), sort_keys=True), encoding="utf-8")

    loaded = load_selection_artifact(path)
    assert loaded == artifact

    payload = artifact.to_dict()
    payload["identity"]["resolved_candidate"]["authoritative_heff_hash"] = _digest(
        "tampered-heff"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_selection_artifact(path)

    path.write_text(
        json.dumps(artifact.to_dict()).replace('"band_rms_mev": 0.25', '"band_rms_mev": NaN'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_selection_artifact(path)


def test_loader_rejects_top_level_metric_as_a_second_source_of_truth(tmp_path: Path) -> None:
    payload = _certified().to_dict()
    payload["metrics"]["band_rms_mev"] = 0.5
    unhashed = dict(payload)
    unhashed.pop("artifact_hash")
    payload["artifact_hash"] = hash_mapping(unhashed)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="single source of truth"):
        load_selection_artifact(path)


def test_selection_identity_is_command_independent_but_artifact_is_transaction_specific() -> None:
    inspect = _certified("inspect-transaction")
    project = _certified("project-transaction")

    assert inspect.identity.selection_identity_hash == project.identity.selection_identity_hash
    assert inspect.artifact_hash != project.artifact_hash


def test_nested_hashes_bind_basis_heff_kmap_metrics_and_certificates() -> None:
    common = {
        "selection_mode": "auto",
        "candidate_id": "candidate-4",
        "candidate_dimension": 4,
        "projection_basis_kind": "fixed_window",
        "basis_handoff_hash": _digest("basis"),
        "authoritative_heff_hash": _digest("heff"),
        "heff_k_indices_hash": _digest("k-map"),
    }
    baseline = ResolvedCandidateIdentity.create(**common)
    assert ResolvedCandidateIdentity.create(
        **(common | {"basis_handoff_hash": _digest("changed-basis")})
    ).resolved_candidate_hash != baseline.resolved_candidate_hash
    assert ResolvedCandidateIdentity.create(
        **(common | {"authoritative_heff_hash": _digest("changed-heff")})
    ).resolved_candidate_hash != baseline.resolved_candidate_hash
    assert ResolvedCandidateIdentity.create(
        **(common | {"heff_k_indices_hash": _digest("changed-k-map")})
    ).resolved_candidate_hash != baseline.resolved_candidate_hash

    baseline_metric = _metric_evidence()
    changed_metric = SelectionMetricEvidence.create(
        candidate_id="candidate-4",
        frozen_target_window_hash=_digest("target"),
        validation_k_indices_hash=_digest("validation-k"),
        basis_handoff_hash=_digest("basis"),
        metrics=CandidateMetrics(**(_metrics().__dict__ | {"band_rms_mev": 0.3})),
    )
    assert changed_metric.metrics_hash != baseline_metric.metrics_hash

    baseline_certificate = _certification_evidence()
    changed_certificate = CertificationEvidence.create(
        metric_evidence=baseline_metric,
        symmetry_certificate_hash=_digest("changed-certificate"),
        symmetry_input_identity_hash=_digest("candidate-input"),
    )
    assert changed_certificate.evidence_hash != baseline_certificate.evidence_hash


@pytest.mark.parametrize("mutation", ("unknown", "missing", "duplicate"))
def test_strict_json_schema_rejects_unknown_missing_and_duplicate_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _certified().to_dict()
    path = tmp_path / "selection.json"
    if mutation == "unknown":
        payload["unknown"] = "value"
        path.write_text(json.dumps(payload), encoding="utf-8")
        match = "unknown fields"
    elif mutation == "missing":
        payload.pop("diagnostic")
        path.write_text(json.dumps(payload), encoding="utf-8")
        match = "missing fields"
    else:
        encoded = json.dumps(payload)
        encoded = encoded.replace(
            '"transaction_id": "transaction-a"',
            '"transaction_id": "transaction-a", "transaction_id": "transaction-b"',
        )
        path.write_text(encoded, encoding="utf-8")
        match = "duplicate field"

    with pytest.raises(ValueError, match=match):
        load_selection_artifact(path)


@pytest.mark.parametrize("status", (CertificationStatus.PENDING, CertificationStatus.FAILED))
def test_nonfinal_artifacts_reject_fabricated_candidate_even_with_rehashed_payload(
    tmp_path: Path,
    status: CertificationStatus,
) -> None:
    if status is CertificationStatus.PENDING:
        artifact = SelectionArtifact.pending(
            transaction_id="pending",
            selection_input_identity_hash=_selection_input_hash(),
        )
    else:
        artifact = SelectionArtifact.failed(
            transaction_id="failed",
            selection_input_identity_hash=_selection_input_hash(),
            reason=SelectionFailureCode.HARD_METRIC_FAILED,
            failure_codes=(SelectionFailureCode.HARD_METRIC_FAILED,),
            diagnostic="failed",
        )
    payload = artifact.to_dict()
    payload["identity"] = _identity().to_dict()
    unhashed = dict(payload)
    unhashed.pop("artifact_hash")
    payload["artifact_hash"] = hash_mapping(unhashed)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot"):
        load_selection_artifact(path)


def test_pending_first_invalidates_older_certified_generation(tmp_path: Path) -> None:
    store = SelectionArtifactStore(tmp_path)
    first_pending = SelectionArtifact.pending(
        transaction_id="first",
        selection_input_identity_hash=_selection_input_hash(),
    )
    with store.begin(first_pending) as transaction:
        transaction.publish(_certified("first"))
    assert store.load_current(require_certified=True).transaction_id == "first"

    second_pending = SelectionArtifact.pending(
        transaction_id="second",
        selection_input_identity_hash=_selection_input_hash(),
    )
    transaction = store.begin(second_pending)
    try:
        assert store.load_current().certification_status is CertificationStatus.PENDING
        with pytest.raises(ValueError, match="not certified"):
            store.load_current(require_certified=True)
    finally:
        transaction.close()


def test_interrupted_transaction_cannot_leave_a_current_certified_marker(tmp_path: Path) -> None:
    store = SelectionArtifactStore(tmp_path)
    pending = SelectionArtifact.pending(
        transaction_id="crashed",
        selection_input_identity_hash=_selection_input_hash(),
    )

    transaction = store.begin(pending)
    transaction.close()

    assert store.load_current().certification_status is CertificationStatus.PENDING
    with pytest.raises(ValueError, match="not certified"):
        store.load_current(require_certified=True)


def test_output_directory_lock_is_exclusive_across_processes(tmp_path: Path) -> None:
    store = SelectionArtifactStore(tmp_path)
    pending = SelectionArtifact.pending(
        transaction_id="lock-holder",
        selection_input_identity_hash=_selection_input_hash(),
    )
    transaction = store.begin(pending)
    script = """
import fcntl
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open("a+b") as handle:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
raise SystemExit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(store.lock_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        transaction.close()


def test_older_transaction_cannot_overwrite_newer_pending_generation(tmp_path: Path) -> None:
    store = SelectionArtifactStore(tmp_path)
    first_pending = SelectionArtifact.pending(
        transaction_id="transaction-a",
        selection_input_identity_hash=_selection_input_hash(),
    )
    transaction_a = store.begin(first_pending)
    transaction_b = SelectionArtifact.pending(
        transaction_id="transaction-b",
        selection_input_identity_hash=_selection_input_hash(),
    )
    store.marker_path.write_text(
        json.dumps(transaction_b.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    try:
        with pytest.raises(SelectionTransactionError) as exc_info:
            transaction_a.publish(_certified("transaction-a"))
        assert exc_info.value.failure_code is SelectionFailureCode.TRANSACTION_SUPERSEDED
        assert store.load_current().transaction_id == "transaction-b"
    finally:
        transaction_a.close()


def test_same_id_replacement_marker_cannot_be_overwritten_by_old_transaction(
    tmp_path: Path,
) -> None:
    store = SelectionArtifactStore(tmp_path)
    first_pending = SelectionArtifact.pending(
        transaction_id="reused-id",
        selection_input_identity_hash=_selection_input_hash(),
    )
    transaction = store.begin(first_pending)
    replacement = SelectionArtifact.pending(
        transaction_id="reused-id",
        selection_input_identity_hash=_digest("replacement-input"),
    )
    store.marker_path.write_text(
        json.dumps(replacement.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    try:
        with pytest.raises(SelectionTransactionError) as exc_info:
            transaction.publish(_certified("reused-id"))
        assert exc_info.value.failure_code is SelectionFailureCode.TRANSACTION_SUPERSEDED
        assert store.load_current() == replacement
    finally:
        transaction.close()


def test_transaction_rejects_wrong_final_id_and_illegal_state_transition(tmp_path: Path) -> None:
    store = SelectionArtifactStore(tmp_path)
    pending = SelectionArtifact.pending(
        transaction_id="transaction-a",
        selection_input_identity_hash=_selection_input_hash(),
    )
    transaction = store.begin(pending)
    try:
        with pytest.raises(SelectionTransactionError, match="transaction ID"):
            transaction.publish(_certified("transaction-b"))
        with pytest.raises(SelectionTransactionError, match="transition"):
            transaction.publish(
                SelectionArtifact.unverified_override(
                    transaction_id="transaction-a",
                    identity=_identity(mode="explicit", with_evidence=False),
                    diagnostic="explicit",
                )
            )
    finally:
        transaction.close()


def test_selection_input_identity_is_canonical_frozen_and_strict() -> None:
    identity = _selection_input()

    assert identity.selection_input_identity_hash == build_selection_input_identity_hash(
        selection_mode="auto",
        frozen_target_window_hash=_digest("target"),
        ordered_q_hash=_digest("q"),
        source_hamiltonian_hash=_digest("source-h"),
        action_package_hash=_digest("actions"),
        row_layout_hash=_digest("layout"),
        selection_policy_hash=_digest("policy"),
    )
    assert SelectionInputIdentity.from_dict(identity.to_dict()) == identity
    with pytest.raises(FrozenInstanceError):
        identity.selection_mode = "explicit"
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(identity, selection_input_identity_hash=_digest("tampered"))
    with pytest.raises(ValueError, match="unknown fields"):
        SelectionInputIdentity.from_dict(identity.to_dict() | {"unknown": True})


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("hard_thresholds", {"band_rms_mev": 0.4}),
        ("candidate_envelope_config", {"dimensions": [4, 8]}),
        ("candidate_generator_version", "generator-v2"),
        ("candidate_schema_version", "candidate-v2"),
        ("metric_schema_version", "metrics-v2"),
        ("ordering_rule_version", "dimension-band-overlap-symmetry-v2"),
    ),
)
def test_selection_policy_hash_binds_every_policy_input(
    field: str,
    replacement: object,
) -> None:
    values = {
        "hard_thresholds": {"band_rms_mev": 0.5, "subspace_overlap": 0.98},
        "candidate_envelope_config": {"dimensions": [4, 6]},
        "candidate_generator_version": "generator-v1",
        "candidate_schema_version": "candidate-v1",
        "metric_schema_version": "metrics-v1",
        "ordering_rule_version": "dimension-band-overlap-symmetry-v1",
    }
    baseline = build_selection_policy_hash(**values)

    assert build_selection_policy_hash(**(values | {field: replacement})) != baseline


def test_selection_policy_hash_rejects_nonfinite_thresholds() -> None:
    common = {
        "candidate_envelope_config": {"dimensions": [4]},
        "candidate_generator_version": "generator-v1",
        "candidate_schema_version": "candidate-v1",
        "metric_schema_version": "metrics-v1",
        "ordering_rule_version": "ordering-v1",
    }
    with pytest.raises(ValueError, match="finite"):
        build_selection_policy_hash(
            hard_thresholds={"band_rms_mev": float("nan")},
            **common,
        )
    with pytest.raises(ValueError, match="real scalar"):
        build_selection_policy_hash(
            hard_thresholds={"band_rms_mev": True},
            **common,
        )


def test_selection_identity_cross_checks_input_mode_policy_and_metric_dimension() -> None:
    with pytest.raises(ValueError, match="mode"):
        SelectionIdentity.create(
            selection_input=_selection_input(selection_mode="explicit"),
            selection_policy_hash=_digest("policy"),
            resolved_candidate=_resolved(mode="auto"),
            certification_evidence=_certification_evidence(),
        )
    mismatched_target = CertificationEvidence.create(
        metric_evidence=SelectionMetricEvidence.create(
            candidate_id="candidate-4",
            frozen_target_window_hash=_digest("different-target"),
            validation_k_indices_hash=_digest("validation-k"),
            basis_handoff_hash=_digest("basis"),
            metrics=_metrics(),
        ),
        symmetry_certificate_hash=_digest("candidate-certificate"),
        symmetry_input_identity_hash=_digest("candidate-input"),
    )
    with pytest.raises(ValueError, match="target"):
        SelectionIdentity.create(
            selection_input=_selection_input(),
            selection_policy_hash=_digest("policy"),
            resolved_candidate=_resolved(),
            certification_evidence=mismatched_target,
        )
    with pytest.raises(ValueError, match="policy"):
        SelectionIdentity.create(
            selection_input=_selection_input(),
            selection_policy_hash=_digest("different-policy"),
            resolved_candidate=_resolved(),
            certification_evidence=_certification_evidence(),
        )
    with pytest.raises(ValueError, match="dimension"):
        SelectionIdentity.create(
            selection_input=_selection_input(),
            selection_policy_hash=_digest("policy"),
            resolved_candidate=ResolvedCandidateIdentity.create(
                selection_mode="auto",
                candidate_id="candidate-4",
                candidate_dimension=5,
                projection_basis_kind="fixed_window",
                basis_handoff_hash=_digest("basis"),
                authoritative_heff_hash=_digest("heff"),
                heff_k_indices_hash=_digest("k-map"),
            ),
            certification_evidence=_certification_evidence(),
        )


def test_frozen_target_hash_binds_edge_count_tolerance_and_row_order() -> None:
    baseline = FrozenTargetWindow(
        spec=TargetWindowSpec(
            edge="valence",
            band_count=2,
            validation_k_indices=(0, 3),
            energy_reference_ev=0.0,
            degeneracy_tolerance_mev=1.0e-3,
        ),
        target_band_ids=((2, 3), (4, 5)),
        target_energies_ev=((-0.2, -0.1), (-0.3, -0.15)),
    )
    changed_edge = replace(baseline, spec=replace(baseline.spec, edge="conduction"))
    changed_count = FrozenTargetWindow(
        spec=replace(baseline.spec, band_count=1),
        target_band_ids=((2,), (4,)),
        target_energies_ev=((-0.2,), (-0.3,)),
    )
    changed_tolerance = replace(
        baseline,
        spec=replace(baseline.spec, degeneracy_tolerance_mev=2.0e-3),
    )
    reversed_rows = FrozenTargetWindow(
        spec=replace(baseline.spec, validation_k_indices=(3, 0)),
        target_band_ids=tuple(reversed(baseline.target_band_ids)),
        target_energies_ev=tuple(reversed(baseline.target_energies_ev)),
    )

    baseline_hash = hash_frozen_target_window(baseline)
    for changed in (changed_edge, changed_count, changed_tolerance, reversed_rows):
        assert hash_frozen_target_window(changed) != baseline_hash


def test_loader_rejects_display_only_pass_status_and_stale_nested_hash(
    tmp_path: Path,
) -> None:
    payload = _certified().to_dict()
    payload["certification_status"] = "PASS"
    unhashed = dict(payload)
    unhashed.pop("artifact_hash")
    payload["artifact_hash"] = hash_mapping(unhashed)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="certification_status"):
        load_selection_artifact(path)

    payload = _certified().to_dict()
    payload["identity"]["resolved_candidate"]["authoritative_heff_hash"] = _digest(
        "changed-heff"
    )
    unhashed = dict(payload)
    unhashed.pop("artifact_hash")
    payload["artifact_hash"] = hash_mapping(unhashed)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved candidate hash mismatch"):
        load_selection_artifact(path)


def test_transaction_stages_and_reloads_generation_payloads_before_certifying(
    tmp_path: Path,
) -> None:
    store = SelectionArtifactStore(tmp_path)
    pending = SelectionArtifact.pending(
        transaction_id="payload-generation",
        selection_input_identity_hash=_selection_input_hash(),
    )
    payloads = {
        "basis.bin": b"basis-payload",
        "nested/heff.bin": b"heff-payload",
    }

    with store.begin(pending) as transaction:
        manifest_hash = transaction.stage_payloads(payloads)
        assert len(manifest_hash) == 64
        assert transaction.payload_manifest_hash == manifest_hash
        assert store.load_current().certification_status is CertificationStatus.PENDING
        transaction.publish(_certified("payload-generation"))

    assert store.load_current(require_certified=True).transaction_id == "payload-generation"
    assert store.load_current_payloads(require_certified=True) == payloads


def test_explicit_transaction_allows_only_unverified_override(tmp_path: Path) -> None:
    store = SelectionArtifactStore(tmp_path)
    input_identity = _selection_input(selection_mode="explicit")
    pending = SelectionArtifact.pending(
        transaction_id="explicit-generation",
        selection_input_identity_hash=input_identity.selection_input_identity_hash,
    )
    final = SelectionArtifact.unverified_override(
        transaction_id="explicit-generation",
        identity=_identity(
            mode="explicit",
            with_evidence=False,
            selection_input=input_identity,
        ),
        diagnostic="explicit user override",
    )

    with store.begin(pending, selection_mode="explicit") as transaction:
        transaction.stage_payloads({"basis.bin": b"explicit-basis"})
        transaction.publish(final)

    assert (
        store.load_current().certification_status
        is CertificationStatus.UNVERIFIED_OVERRIDE
    )
    with pytest.raises(ValueError, match="not certified"):
        store.load_current(require_certified=True)


def test_payload_tampering_blocks_final_marker_and_leaves_pending(tmp_path: Path) -> None:
    store = SelectionArtifactStore(tmp_path)
    pending = SelectionArtifact.pending(
        transaction_id="tampered-payload",
        selection_input_identity_hash=_selection_input_hash(),
    )
    with store.begin(pending) as transaction:
        transaction.stage_payloads({"basis.bin": b"original"})
        (transaction.generation_directory / "basis.bin").write_bytes(b"tampered")
        with pytest.raises(SelectionTransactionError) as exc_info:
            transaction.publish(_certified("tampered-payload"))
        assert exc_info.value.failure_code is SelectionFailureCode.PERSISTENCE_FAILURE
        assert store.load_current().certification_status is CertificationStatus.PENDING


def test_fault_before_final_marker_leaves_pending_and_releases_lock(tmp_path: Path) -> None:
    def fail(stage: str) -> None:
        if stage == "before_final_marker":
            raise RuntimeError("injected crash")

    store = SelectionArtifactStore(tmp_path, fault_injector=fail)
    pending = SelectionArtifact.pending(
        transaction_id="faulted",
        selection_input_identity_hash=_selection_input_hash(),
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        with store.begin(pending) as transaction:
            transaction.stage_payloads({"basis.bin": b"basis"})
            transaction.publish(_certified("faulted"))

    assert store.load_current().certification_status is CertificationStatus.PENDING
    replacement_store = SelectionArtifactStore(tmp_path)
    replacement = SelectionArtifact.pending(
        transaction_id="replacement",
        selection_input_identity_hash=_selection_input_hash(),
    )
    replacement_transaction = replacement_store.begin(replacement)
    replacement_transaction.close()


def test_process_exit_releases_transaction_lock_and_leaves_pending(tmp_path: Path) -> None:
    script = """
import os
import pathlib
import sys
from kp.selection_artifact import SelectionArtifact, SelectionArtifactStore

output = pathlib.Path(sys.argv[1])
pending = SelectionArtifact.pending(
    transaction_id="crash-process",
    selection_input_identity_hash=sys.argv[2],
)
SelectionArtifactStore(output).begin(pending)
os._exit(23)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), _selection_input_hash()],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 23

    store = SelectionArtifactStore(tmp_path)
    assert store.load_current().certification_status is CertificationStatus.PENDING
    replacement = SelectionArtifact.pending(
        transaction_id="after-crash",
        selection_input_identity_hash=_selection_input_hash(),
    )
    transaction = store.begin(replacement)
    transaction.close()
