"""Command-independent projection-selection resolution.

This module owns the state transition between a canonical selection input and
its persisted selection artifact.  CLI commands may prepare the inputs in
different ways, but command names, output paths, and plot options never enter
the selection identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .low_energy_selection import (
    CandidateMetrics,
    CandidateSelectionError,
    SelectionDecision,
    SelectionThresholds,
    select_projection_candidate,
)
from .selection_artifact import (
    CertificationStatus,
    ResolvedCandidateIdentity,
    SelectionArtifact,
    SelectionArtifactStore,
    SelectionArtifactTransaction,
    SelectionBindingError,
    SelectionFailureCode,
    SelectionIdentity,
    SelectionInputIdentity,
    build_certified_gamma_selection_identity,
)


@dataclass(frozen=True)
class CaseSelectionInputs:
    """Fully materialized physical inputs for one selection decision."""

    selection_input: SelectionInputIdentity
    candidates: tuple[CandidateMetrics, ...]
    thresholds: SelectionThresholds | None
    projection_handoff: Any | None
    resolved_candidate: ResolvedCandidateIdentity | None
    payloads: tuple[tuple[str, bytes], ...]
    diagnostic: str | None
    preview_only: bool

    @classmethod
    def auto(
        cls,
        *,
        selection_input: SelectionInputIdentity,
        candidates: Sequence[CandidateMetrics],
        thresholds: SelectionThresholds,
        projection_handoff: Any | None,
        payloads: Mapping[str, bytes],
    ) -> "CaseSelectionInputs":
        if selection_input.selection_mode != "auto":
            raise ValueError("automatic selection requires auto input identity")
        if not isinstance(payloads, Mapping):
            raise TypeError("selection payloads must be a mapping")
        frozen_payloads = tuple(
            (str(name), bytes(content)) for name, content in sorted(payloads.items())
        )
        return cls(
            selection_input=selection_input,
            candidates=tuple(candidates),
            thresholds=thresholds,
            projection_handoff=projection_handoff,
            resolved_candidate=None,
            payloads=frozen_payloads,
            diagnostic=None,
            preview_only=False,
        )

    @classmethod
    def explicit(
        cls,
        *,
        selection_input: SelectionInputIdentity,
        resolved_candidate: ResolvedCandidateIdentity,
        diagnostic: str,
    ) -> "CaseSelectionInputs":
        if selection_input.selection_mode != "explicit":
            raise ValueError("explicit selection requires explicit input identity")
        if resolved_candidate.selection_mode != "explicit":
            raise ValueError("explicit selection requires an explicit resolved candidate")
        if not isinstance(diagnostic, str) or not diagnostic.strip():
            raise ValueError("explicit selection requires a diagnostic")
        return cls(
            selection_input=selection_input,
            candidates=(),
            thresholds=None,
            projection_handoff=None,
            resolved_candidate=resolved_candidate,
            payloads=(),
            diagnostic=diagnostic.strip(),
            preview_only=False,
        )

    @classmethod
    def preview(
        cls,
        selection_input: SelectionInputIdentity,
    ) -> "CaseSelectionInputs":
        return cls(
            selection_input=selection_input,
            candidates=(),
            thresholds=None,
            projection_handoff=None,
            resolved_candidate=None,
            payloads=(),
            diagnostic=None,
            preview_only=True,
        )


@dataclass(frozen=True)
class ResolvedProjectionSelection:
    selection_input: SelectionInputIdentity
    status: CertificationStatus
    identity: SelectionIdentity | None
    artifact: SelectionArtifact | None
    decision: SelectionDecision | None


@dataclass
class PendingCaseSelection:
    selection_input: SelectionInputIdentity
    pending: SelectionArtifact
    transaction: SelectionArtifactTransaction

    def close(self) -> None:
        self.transaction.close()


def begin_case_selection(
    *,
    store: SelectionArtifactStore,
    selection_input: SelectionInputIdentity,
    transaction_id: str,
) -> PendingCaseSelection:
    """Publish PENDING before any command performs selection-dependent work."""

    if not isinstance(store, SelectionArtifactStore):
        raise TypeError("store must be SelectionArtifactStore")
    if not isinstance(selection_input, SelectionInputIdentity):
        raise TypeError("selection_input must be SelectionInputIdentity")
    pending = SelectionArtifact.pending(
        transaction_id=transaction_id,
        selection_input_identity_hash=selection_input.selection_input_identity_hash,
    )
    transaction = store.begin(
        pending,
        selection_mode=selection_input.selection_mode,
    )
    return PendingCaseSelection(
        selection_input=selection_input,
        pending=pending,
        transaction=transaction,
    )


def preview_case_selection(
    selection_input: SelectionInputIdentity,
) -> ResolvedProjectionSelection:
    """Return an honest dry-run state when no final basis/Heff exists yet."""

    if not isinstance(selection_input, SelectionInputIdentity):
        raise TypeError("selection_input must be SelectionInputIdentity")
    return resolve_case_selection(CaseSelectionInputs.preview(selection_input))


def _failure_code(error: CandidateSelectionError) -> SelectionFailureCode:
    return SelectionFailureCode(error.failure_code.value)


def _publish_auto_failure(
    session: PendingCaseSelection,
    *,
    reason: SelectionFailureCode,
    diagnostic: str,
) -> ResolvedProjectionSelection:
    artifact = SelectionArtifact.failed(
        transaction_id=session.pending.transaction_id,
        selection_input_identity_hash=(
            session.selection_input.selection_input_identity_hash
        ),
        reason=reason,
        failure_codes=(reason,),
        diagnostic=diagnostic,
    )
    session.transaction.publish(artifact)
    return ResolvedProjectionSelection(
        selection_input=session.selection_input,
        status=CertificationStatus.FAILED,
        identity=None,
        artifact=artifact,
        decision=None,
    )


def resolve_case_selection(
    canonical_inputs: CaseSelectionInputs,
    *,
    session: PendingCaseSelection | None = None,
) -> ResolvedProjectionSelection:
    """Resolve and persist one canonical selection transaction fail-closed."""

    if not isinstance(canonical_inputs, CaseSelectionInputs):
        raise TypeError("canonical_inputs must be CaseSelectionInputs")
    if canonical_inputs.preview_only:
        if session is not None:
            raise ValueError("selection previews cannot own a persistence transaction")
        return ResolvedProjectionSelection(
            selection_input=canonical_inputs.selection_input,
            status=CertificationStatus.PENDING,
            identity=None,
            artifact=None,
            decision=None,
        )
    if not isinstance(session, PendingCaseSelection):
        raise TypeError("final selection resolution requires PendingCaseSelection")
    if canonical_inputs.selection_input != session.selection_input:
        session.close()
        raise SelectionBindingError(
            "selection inputs do not match the PENDING transaction"
        )

    try:
        if canonical_inputs.selection_input.selection_mode == "explicit":
            resolved = canonical_inputs.resolved_candidate
            if resolved is None:
                raise SelectionBindingError(
                    "explicit selection is missing its resolved candidate"
                )
            identity = SelectionIdentity.create(
                selection_input=canonical_inputs.selection_input,
                selection_policy_hash=(
                    canonical_inputs.selection_input.selection_policy_hash
                ),
                resolved_candidate=resolved,
                certification_evidence=None,
            )
            artifact = SelectionArtifact.unverified_override(
                transaction_id=session.pending.transaction_id,
                identity=identity,
                diagnostic=str(canonical_inputs.diagnostic),
            )
            session.transaction.publish(artifact)
            return ResolvedProjectionSelection(
                selection_input=canonical_inputs.selection_input,
                status=CertificationStatus.UNVERIFIED_OVERRIDE,
                identity=identity,
                artifact=artifact,
                decision=None,
            )

        thresholds = canonical_inputs.thresholds
        if thresholds is None:
            return _publish_auto_failure(
                session,
                reason=SelectionFailureCode.STRUCTURAL_REJECTION,
                diagnostic="automatic selection is missing hard thresholds",
            )
        try:
            decision = select_projection_candidate(
                canonical_inputs.candidates,
                thresholds,
            )
        except CandidateSelectionError as error:
            return _publish_auto_failure(
                session,
                reason=_failure_code(error),
                diagnostic=str(error),
            )
        except (TypeError, ValueError) as error:
            return _publish_auto_failure(
                session,
                reason=SelectionFailureCode.STRUCTURAL_REJECTION,
                diagnostic=str(error),
            )
        if canonical_inputs.projection_handoff is None:
            return _publish_auto_failure(
                session,
                reason=SelectionFailureCode.CANDIDATE_CERTIFICATE_FAILED,
                diagnostic=(
                    "automatic certification requires a real routed Gamma handoff"
                ),
            )
        try:
            identity = build_certified_gamma_selection_identity(
                selection_input=canonical_inputs.selection_input,
                handoff=canonical_inputs.projection_handoff,
                metrics=decision.selected,
            )
        except (SelectionBindingError, TypeError, ValueError) as error:
            return _publish_auto_failure(
                session,
                reason=SelectionFailureCode.CANDIDATE_CERTIFICATE_FAILED,
                diagnostic=str(error),
            )
        if not canonical_inputs.payloads:
            return _publish_auto_failure(
                session,
                reason=SelectionFailureCode.PERSISTENCE_FAILURE,
                diagnostic="automatic certification requires persisted payloads",
            )
        payload_manifest_hash = session.transaction.stage_payloads(
            dict(canonical_inputs.payloads)
        )
        artifact = SelectionArtifact.certified(
            transaction_id=session.pending.transaction_id,
            identity=identity,
            payload_manifest_hash=payload_manifest_hash,
            projection_handoff=canonical_inputs.projection_handoff,
        )
        session.transaction.publish(artifact)
        return ResolvedProjectionSelection(
            selection_input=canonical_inputs.selection_input,
            status=CertificationStatus.CERTIFIED,
            identity=identity,
            artifact=artifact,
            decision=decision,
        )
    finally:
        session.close()
