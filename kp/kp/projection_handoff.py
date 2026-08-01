"""Typed, fail-closed projection-basis handoffs.

The routed Gamma variant owns the exact local frames and effective
Hamiltonians consumed by downstream symmetry projection.  Its archive is
numeric-only and every identity is recomputed on load.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence, TypeAlias

import numpy as np

from .blocks.gamma_layout import (
    GammaRoutedFrames,
    GammaRoutingError,
    GammaRoutingThresholds,
    GammaRowLayout,
    assemble_gamma_routed_projectors,
)
from .identity import (
    PROJECTION_ARTIFACT_IDENTITY_FIELDS,
    hash_array,
    hash_mapping,
)
from .projection_selection import CandidateRejectionReason


GAMMA_ROUTED_BASIS_HANDOFF_VERSION = "kp_project_gamma_routed_handoff_v1"
EXPLICIT_LEGACY_BASIS_KIND = "explicit_legacy"
GAMMA_ROUTED_BASIS_KIND = "gamma_routed"

_GAMMA_ROUTED_ARCHIVE_KEYS = frozenset(
    {
        "projection_basis_kind",
        "projection_basis_handoff_version",
        "base_basis_hash",
        "layout_payload",
        "layout_hash",
        "routing_thresholds_payload",
        "routing_thresholds_hash",
        "joint_band_indices",
        "group_ranks",
        "group_offsets",
        "k_indices",
        "frames",
        "reference_frames",
        "routed_frame_hashes",
        "routed_reference_frame_hashes",
        "frame_hash",
        "reference_frame_hash",
        "route_gaps",
        "heff_k_indices",
        "authoritative_heff",
        "heff_hash",
        "closure_certificate_hashes",
        "routing_certificate_hashes",
        "source_hamiltonian_hash",
        "ordered_q_hashes",
        "raw_action_package_hash",
        "candidate_id",
        "candidate_certificate_envelope_json",
        "candidate_certificate_hash",
        "candidate_input_identity_hash",
        "handoff_identity_hash",
        *PROJECTION_ARTIFACT_IDENTITY_FIELDS,
    }
)

GAMMA_ROUTED_ONLY_BASIS_FIELDS = frozenset(
    _GAMMA_ROUTED_ARCHIVE_KEYS
    - {
        "projection_basis_kind",
        "projection_basis_handoff_version",
        "heff_hash",
        "source_hamiltonian_hash",
        *PROJECTION_ARTIFACT_IDENTITY_FIELDS,
    }
)

_INT64_FIELDS = frozenset(
    {
        "joint_band_indices",
        "group_ranks",
        "group_offsets",
        "k_indices",
        "heff_k_indices",
    }
)
_COMPLEX128_FIELDS = frozenset(
    {"frames", "reference_frames", "authoritative_heff"}
)
_FLOAT64_FIELDS = frozenset({"route_gaps"})
_SHA256_SCALAR_FIELDS = frozenset(
    {
        "base_basis_hash",
        "layout_hash",
        "routing_thresholds_hash",
        "frame_hash",
        "reference_frame_hash",
        "heff_hash",
        "source_hamiltonian_hash",
        "raw_action_package_hash",
        "candidate_certificate_hash",
        "candidate_input_identity_hash",
        "handoff_identity_hash",
        "input_hash",
        "config_hash",
        "basis_hash",
        "k_indices_hash",
    }
)
_SHA256_VECTOR_FIELDS = frozenset(
    {
        "routed_frame_hashes",
        "routed_reference_frame_hashes",
        "closure_certificate_hashes",
        "routing_certificate_hashes",
        "ordered_q_hashes",
    }
)


def _reject(reason: CandidateRejectionReason, message: str) -> GammaRoutingError:
    return GammaRoutingError(reason, message)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    if name not in payload:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma handoff is missing {name}",
        )
    value = np.asarray(payload[name])
    if value.shape != () or value.dtype.kind != "U":
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma handoff {name} must be a Unicode scalar",
        )
    scalar = value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    text = str(scalar)
    if not text:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma handoff {name} must be nonempty",
        )
    return text


def _strict_index_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype != np.dtype(np.int64):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} must be a one-dimensional int64 array",
        )
    indices = tuple(int(item) for item in array.tolist())
    if not indices or any(item < 0 for item in indices) or len(set(indices)) != len(indices):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} must contain unique nonnegative indices",
        )
    return indices


def _strict_hash_tuple(value: Any, *, name: str, length: int) -> tuple[str, ...]:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.dtype != np.dtype("<U64")
        or len(array) != length
    ):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} must contain {length} scalar hashes",
        )
    hashes = tuple(str(item) for item in array.tolist())
    if any(not item for item in hashes):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} contains an empty identity",
        )
    return hashes


def _require_exact_archive_keys(files: Sequence[str]) -> None:
    actual = frozenset(str(name) for name in files)
    missing = sorted(_GAMMA_ROUTED_ARCHIVE_KEYS - actual)
    extra = sorted(actual - _GAMMA_ROUTED_ARCHIVE_KEYS)
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma archive key set mismatch (" + "; ".join(details) + ")",
        )


def _require_disk_dtypes(payload: Mapping[str, np.ndarray]) -> None:
    for name in _INT64_FIELDS:
        if np.asarray(payload[name]).dtype != np.dtype(np.int64):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must have dtype int64",
            )
    for name in _COMPLEX128_FIELDS:
        if np.asarray(payload[name]).dtype != np.dtype(np.complex128):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must have dtype complex128",
            )
    for name in _FLOAT64_FIELDS:
        if np.asarray(payload[name]).dtype != np.dtype(np.float64):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must have dtype float64",
            )
    for name in _SHA256_SCALAR_FIELDS:
        value = np.asarray(payload[name])
        if value.shape != () or value.dtype != np.dtype("<U64"):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must be a <U64 scalar",
            )
    for name in _SHA256_VECTOR_FIELDS:
        value = np.asarray(payload[name])
        if value.ndim != 1 or value.dtype != np.dtype("<U64"):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must be a <U64 vector",
            )
    schema_version = np.asarray(payload["schema_version"])
    if schema_version.shape != () or schema_version.dtype != np.dtype(np.int64):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma archive schema_version must be an int64 scalar",
        )


def _freeze(array: Any, *, dtype: Any, name: str) -> np.ndarray:
    try:
        value = np.array(array, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} cannot be represented as a numeric tensor",
        ) from error
    if value.dtype.hasobject or not np.all(np.isfinite(value)):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} must be finite and numeric",
        )
    value.setflags(write=False)
    return value


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _bound_routed_basis_hash(
    *,
    base_basis_hash: str,
    layout_hash: str,
    thresholds_hash: str,
    k_indices_hash: str,
    frame_hash: str,
    reference_frame_hash: str,
    heff_hash: str,
    closure_certificate_hashes: Sequence[str],
    routing_certificate_hashes: Sequence[str],
    source_hamiltonian_hash: str,
    ordered_q_hashes: Sequence[str],
    raw_action_package_hash: str,
    candidate_certificate_hash: str,
    candidate_input_identity_hash: str,
) -> str:
    return hash_mapping(
        {
            "schema": "kp.gamma-routed-projection-basis.v1",
            "base_basis_hash": str(base_basis_hash),
            "layout_hash": str(layout_hash),
            "thresholds_hash": str(thresholds_hash),
            "k_indices_hash": str(k_indices_hash),
            "frame_hash": str(frame_hash),
            "reference_frame_hash": str(reference_frame_hash),
            "heff_hash": str(heff_hash),
            "closure_certificate_hashes": tuple(closure_certificate_hashes),
            "routing_certificate_hashes": tuple(routing_certificate_hashes),
            "source_hamiltonian_hash": str(source_hamiltonian_hash),
            "ordered_q_hashes": tuple(ordered_q_hashes),
            "raw_action_package_hash": str(raw_action_package_hash),
            "candidate_certificate_hash": str(candidate_certificate_hash),
            "candidate_input_identity_hash": str(candidate_input_identity_hash),
        }
    )


@dataclass(frozen=True)
class ExplicitLegacyBasisSpec:
    """Versioned compatibility view of the explicit physical-layer basis."""

    artifact_identity: Mapping[str, Any]
    nlow_state_list: list[Any]
    resolved_norb_fix_list: list[Any]
    gauge_mode: str
    frame_artifact: Mapping[str, Any] | None
    model_dim: int
    frame: Any | None = None

    projection_basis_kind: ClassVar[str] = EXPLICIT_LEGACY_BASIS_KIND


@dataclass(frozen=True)
class GammaRoutedBasisSpec:
    """Exact per-k routed frames and authoritative projected Hamiltonians."""

    artifact_identity: Mapping[str, Any]
    base_basis_hash: str
    layout: GammaRowLayout
    thresholds: GammaRoutingThresholds
    joint_band_indices: tuple[int, ...]
    group_ranks: tuple[int, int]
    group_offsets: tuple[int, int, int]
    k_indices: tuple[int, ...]
    frames: np.ndarray
    reference_frames: np.ndarray
    routed_frame_hashes: tuple[str, ...]
    routed_reference_frame_hashes: tuple[str, ...]
    frame_hash: str
    reference_frame_hash: str
    route_gaps: np.ndarray
    heff_k_indices: tuple[int, ...]
    authoritative_heff: np.ndarray
    heff_hash: str
    closure_certificate_hashes: tuple[str, ...]
    routing_certificate_hashes: tuple[str, ...]
    source_hamiltonian_hash: str
    ordered_q_hashes: tuple[str, str]
    raw_action_package_hash: str
    candidate_id: str
    candidate_certificate_envelope_json: str
    candidate_certificate_hash: str
    candidate_input_identity_hash: str

    projection_basis_kind: ClassVar[str] = GAMMA_ROUTED_BASIS_KIND
    handoff_version: ClassVar[str] = GAMMA_ROUTED_BASIS_HANDOFF_VERSION

    def __post_init__(self) -> None:
        try:
            k_indices = tuple(int(item) for item in self.k_indices)
            heff_k_indices = tuple(int(item) for item in self.heff_k_indices)
            joint = tuple(int(item) for item in self.joint_band_indices)
            ranks = tuple(int(item) for item in self.group_ranks)
            offsets = tuple(int(item) for item in self.group_offsets)
        except (TypeError, ValueError) as error:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma integer metadata is invalid",
            ) from error
        if (
            not k_indices
            or any(item < 0 for item in k_indices)
            or len(set(k_indices)) != len(k_indices)
            or heff_k_indices != k_indices
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma frame and authoritative-Heff k mappings differ",
            )
        if (
            len(ranks) != 2
            or any(rank <= 0 for rank in ranks)
            or offsets != (0, ranks[0], sum(ranks))
            or not joint
            or len(joint) != sum(ranks)
            or len(set(joint)) != len(joint)
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma joint-band ranks or offsets are invalid",
            )
        frames = _freeze(self.frames, dtype=np.complex128, name="frames")
        references = _freeze(
            self.reference_frames,
            dtype=np.complex128,
            name="reference_frames",
        )
        route_gaps = _freeze(self.route_gaps, dtype=np.float64, name="route_gaps")
        heff = _freeze(
            self.authoritative_heff,
            dtype=np.complex128,
            name="authoritative_heff",
        )
        expected_frame_shape = (
            len(k_indices),
            self.layout.q_count,
            self.layout.same_q_dimension,
            sum(ranks),
        )
        model_dim = self.layout.q_count * sum(ranks)
        if frames.shape != expected_frame_shape or references.shape != expected_frame_shape:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma frame tensors do not match layout/k/group dimensions",
            )
        if route_gaps.shape != expected_frame_shape[:2]:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma route gaps do not match ordered k/Q coverage",
            )
        if heff.shape != (len(k_indices), model_dim, model_dim):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma authoritative Heff shape is invalid",
            )
        if tuple(self.ordered_q_hashes) != tuple(self.layout.ordered_qset_hashes):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma ordered-Q identity does not match the row layout",
            )
        scalar_identities = (
            self.source_hamiltonian_hash,
            self.raw_action_package_hash,
            self.candidate_certificate_hash,
            self.candidate_input_identity_hash,
            *self.ordered_q_hashes,
            *self.closure_certificate_hashes,
            *self.routing_certificate_hashes,
            *self.routed_frame_hashes,
            *self.routed_reference_frame_hashes,
        )
        if any(not _is_sha256(value) for value in scalar_identities):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma source/Q/action/frame/certificate identities must be SHA-256 hashes",
            )
        if any(
            len(values) != len(k_indices)
            for values in (
                self.closure_certificate_hashes,
                self.routing_certificate_hashes,
                self.routed_frame_hashes,
                self.routed_reference_frame_hashes,
            )
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma frame/closure/routing identities do not cover ordered k",
            )
        provisional_certificate = (
            self.candidate_certificate_hash == "0" * 64
            and self.candidate_input_identity_hash == "0" * 64
            and self.raw_action_package_hash == "0" * 64
            and not self.candidate_id
            and not self.candidate_certificate_envelope_json
        )
        if not provisional_certificate:
            try:
                from .symmetry.candidate_certificate import (
                    candidate_action_package_hash,
                    verify_candidate_certificate_envelope,
                )

                envelope_raw = json.loads(self.candidate_certificate_envelope_json)
                verified_envelope = verify_candidate_certificate_envelope(envelope_raw)
            except (ImportError, TypeError, ValueError) as error:
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    f"routed Gamma candidate certificate envelope is invalid: {error}",
                ) from error
            if (
                verified_envelope["candidate_id"] != self.candidate_id
                or verified_envelope["certificate_hash"]
                != self.candidate_certificate_hash
                or verified_envelope["input_identity_hash"]
                != self.candidate_input_identity_hash
                or candidate_action_package_hash(
                    verified_envelope["input_identity_payload"]
                )
                != self.raw_action_package_hash
            ):
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    "routed Gamma candidate/action certificate identity mismatch",
                )
            state_records = verified_envelope["input_identity_payload"].get(
                "states", []
            )
            if not isinstance(state_records, list):
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    "routed Gamma candidate state identity payload is invalid",
                )
            observed_k: set[int] = set()
            for record in state_records:
                if not isinstance(record, Mapping):
                    raise _reject(
                        CandidateRejectionReason.HANDOFF_IDENTITY,
                        "routed Gamma candidate state record is invalid",
                    )
                k_index = int(record.get("k_index", -1))
                if k_index not in k_indices:
                    raise _reject(
                        CandidateRejectionReason.HANDOFF_K_COVERAGE,
                        "candidate certificate references k outside routed handoff",
                    )
                position = k_indices.index(k_index)
                routed_for_k = self.routed_frames_for_k(k_index)
                u_low = assemble_gamma_routed_projectors(
                    routed_for_k,
                    layout=self.layout,
                    thresholds=self.thresholds,
                    include_high=False,
                )[0]
                if (
                    record.get("present") is not True
                    or record.get("u_low_hash")
                    != hash_array(np.asarray(u_low, dtype=np.dtype("<c16")))
                    or record.get("heff_hash")
                    != hash_array(np.asarray(heff[position], dtype=np.dtype("<c16")))
                ):
                    raise _reject(
                        CandidateRejectionReason.HANDOFF_IDENTITY,
                        f"candidate certificate state binding differs at k={k_index}",
                    )
                observed_k.add(k_index)
            if observed_k != set(k_indices):
                raise _reject(
                    CandidateRejectionReason.HANDOFF_K_COVERAGE,
                    "candidate certificate does not cover every routed k index",
                )
        if self.frame_hash != hash_array(frames):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma aggregate frame hash mismatch",
            )
        if self.reference_frame_hash != hash_array(references):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma aggregate reference-frame hash mismatch",
            )
        if self.heff_hash != hash_array(heff):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma authoritative Heff hash mismatch",
            )
        artifact_identity = dict(self.artifact_identity)
        if artifact_identity.get("heff_hash") != self.heff_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection identity does not bind the authoritative routed Heff",
            )
        if artifact_identity.get("k_indices_hash") != hash_array(
            np.asarray(k_indices, dtype=np.int64)
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection identity does not bind the routed k mapping",
            )
        if artifact_identity.get("source_hamiltonian_hash") != self.source_hamiltonian_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection identity does not bind the routed source Hamiltonian",
            )
        expected_basis_hash = _bound_routed_basis_hash(
            base_basis_hash=self.base_basis_hash,
            layout_hash=self.layout.layout_hash,
            thresholds_hash=self.thresholds.identity_hash,
            k_indices_hash=str(artifact_identity["k_indices_hash"]),
            frame_hash=self.frame_hash,
            reference_frame_hash=self.reference_frame_hash,
            heff_hash=self.heff_hash,
            closure_certificate_hashes=self.closure_certificate_hashes,
            routing_certificate_hashes=self.routing_certificate_hashes,
            source_hamiltonian_hash=self.source_hamiltonian_hash,
            ordered_q_hashes=self.ordered_q_hashes,
            raw_action_package_hash=self.raw_action_package_hash,
            candidate_certificate_hash=self.candidate_certificate_hash,
            candidate_input_identity_hash=self.candidate_input_identity_hash,
        )
        if artifact_identity.get("basis_hash") != expected_basis_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection basis identity does not bind the complete routed handoff",
            )
        object.__setattr__(self, "artifact_identity", artifact_identity)
        object.__setattr__(self, "k_indices", k_indices)
        object.__setattr__(self, "heff_k_indices", heff_k_indices)
        object.__setattr__(self, "joint_band_indices", joint)
        object.__setattr__(self, "group_ranks", ranks)
        object.__setattr__(self, "group_offsets", offsets)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "reference_frames", references)
        object.__setattr__(self, "route_gaps", route_gaps)
        object.__setattr__(self, "authoritative_heff", heff)
        # The production assembler is also the structural validator.  This
        # catches per-k hashes, group slices, projectors, gaps, and layout drift.
        for k_index in k_indices:
            assemble_gamma_routed_projectors(
                self.routed_frames_for_k(k_index),
                layout=self.layout,
                thresholds=self.thresholds,
                include_high=False,
            )

    @classmethod
    def create(
        cls,
        *,
        artifact_identity: Mapping[str, Any],
        layout: GammaRowLayout,
        thresholds: GammaRoutingThresholds,
        k_indices: Sequence[int],
        routed_frames: Sequence[GammaRoutedFrames],
        authoritative_heff: Any,
        heff_k_indices: Sequence[int],
        closure_certificate_hashes: Sequence[str],
        routing_certificate_hashes: Sequence[str],
        source_hamiltonian_hash: str,
        ordered_q_hashes: Sequence[str],
        raw_action_package_hash: str,
        candidate_certificate_hash: str,
        candidate_input_identity_hash: str,
        candidate_id: str = "",
        candidate_certificate_envelope_json: str = "",
    ) -> "GammaRoutedBasisSpec":
        indices = tuple(int(item) for item in k_indices)
        routed = tuple(routed_frames)
        if len(routed) != len(indices) or not routed:
            raise _reject(
                CandidateRejectionReason.HANDOFF_K_COVERAGE,
                "routed Gamma frame rows do not cover ordered k indices",
            )
        first = routed[0]
        if any(
            (
                item.layout_hash != layout.layout_hash
                or item.thresholds_hash != thresholds.identity_hash
                or item.joint_band_indices != first.joint_band_indices
                or item.group_dimensions != first.group_dimensions
                or item.group_offsets != first.group_offsets
            )
            for item in routed
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma per-k frame metadata is inconsistent",
            )
        try:
            frames = np.stack(
                [np.stack(item.local_frames_by_q, axis=0) for item in routed],
                axis=0,
            )
            references = np.stack(
                [
                    np.stack(
                        [np.column_stack(groups) for groups in item.reference_frames_by_q_group],
                        axis=0,
                    )
                    for item in routed
                ],
                axis=0,
            )
            route_gaps = np.asarray([item.route_gaps for item in routed], dtype=np.float64)
        except ValueError as error:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma per-k frames cannot form fixed numeric tensors",
            ) from error
        heff = np.asarray(authoritative_heff, dtype=np.complex128)
        identity = dict(artifact_identity)
        base_basis_hash = str(identity.get("basis_hash", ""))
        if not base_basis_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma handoff requires a base projection basis hash",
            )
        identity["k_indices_hash"] = hash_array(np.asarray(indices, dtype=np.int64))
        identity["heff_hash"] = hash_array(heff)
        identity["source_hamiltonian_hash"] = str(source_hamiltonian_hash)
        identity["basis_hash"] = _bound_routed_basis_hash(
            base_basis_hash=base_basis_hash,
            layout_hash=layout.layout_hash,
            thresholds_hash=thresholds.identity_hash,
            k_indices_hash=str(identity["k_indices_hash"]),
            frame_hash=hash_array(frames),
            reference_frame_hash=hash_array(references),
            heff_hash=str(identity["heff_hash"]),
            closure_certificate_hashes=closure_certificate_hashes,
            routing_certificate_hashes=routing_certificate_hashes,
            source_hamiltonian_hash=str(source_hamiltonian_hash),
            ordered_q_hashes=ordered_q_hashes,
            raw_action_package_hash=str(raw_action_package_hash),
            candidate_certificate_hash=str(candidate_certificate_hash),
            candidate_input_identity_hash=str(candidate_input_identity_hash),
        )
        return cls(
            artifact_identity=identity,
            base_basis_hash=base_basis_hash,
            layout=layout,
            thresholds=thresholds,
            joint_band_indices=first.joint_band_indices,
            group_ranks=first.group_dimensions,
            group_offsets=first.group_offsets,
            k_indices=indices,
            frames=frames,
            reference_frames=references,
            routed_frame_hashes=tuple(item.frame_hash for item in routed),
            routed_reference_frame_hashes=tuple(
                item.reference_frame_hash for item in routed
            ),
            frame_hash=hash_array(frames),
            reference_frame_hash=hash_array(references),
            route_gaps=route_gaps,
            heff_k_indices=tuple(int(item) for item in heff_k_indices),
            authoritative_heff=heff,
            heff_hash=hash_array(heff),
            closure_certificate_hashes=tuple(
                str(item) for item in closure_certificate_hashes
            ),
            routing_certificate_hashes=tuple(
                str(item) for item in routing_certificate_hashes
            ),
            source_hamiltonian_hash=str(source_hamiltonian_hash),
            ordered_q_hashes=tuple(str(item) for item in ordered_q_hashes),  # type: ignore[arg-type]
            raw_action_package_hash=str(raw_action_package_hash),
            candidate_id=str(candidate_id),
            candidate_certificate_envelope_json=str(
                candidate_certificate_envelope_json
            ),
            candidate_certificate_hash=str(candidate_certificate_hash),
            candidate_input_identity_hash=str(candidate_input_identity_hash),
        )

    @property
    def model_dim(self) -> int:
        return int(self.authoritative_heff.shape[-1])

    @property
    def handoff_identity_hash(self) -> str:
        return hash_mapping(
            {
                "version": self.handoff_version,
                "projection_basis_kind": self.projection_basis_kind,
                "base_basis_hash": self.base_basis_hash,
                "layout_hash": self.layout.layout_hash,
                "thresholds_hash": self.thresholds.identity_hash,
                "joint_band_indices": self.joint_band_indices,
                "group_ranks": self.group_ranks,
                "group_offsets": self.group_offsets,
                "k_indices_hash": hash_array(np.asarray(self.k_indices, dtype=np.int64)),
                "frame_hash": self.frame_hash,
                "reference_frame_hash": self.reference_frame_hash,
                "routed_frame_hashes": self.routed_frame_hashes,
                "routed_reference_frame_hashes": self.routed_reference_frame_hashes,
                "route_gaps_hash": hash_array(self.route_gaps),
                "heff_k_indices_hash": hash_array(
                    np.asarray(self.heff_k_indices, dtype=np.int64)
                ),
                "heff_hash": self.heff_hash,
                "closure_certificate_hashes": self.closure_certificate_hashes,
                "routing_certificate_hashes": self.routing_certificate_hashes,
                "source_hamiltonian_hash": self.source_hamiltonian_hash,
                "ordered_q_hashes": self.ordered_q_hashes,
                "raw_action_package_hash": self.raw_action_package_hash,
                "candidate_id": self.candidate_id,
                "candidate_certificate_hash": self.candidate_certificate_hash,
                "candidate_input_identity_hash": self.candidate_input_identity_hash,
            }
        )

    def require_k_indices(self, required: Sequence[int]) -> None:
        missing = sorted(set(int(item) for item in required) - set(self.k_indices))
        if missing:
            raise _reject(
                CandidateRejectionReason.HANDOFF_K_COVERAGE,
                f"routed Gamma handoff is missing required k indices {missing}",
            )

    def _k_position(self, k_index: int) -> int:
        self.require_k_indices((k_index,))
        return self.k_indices.index(int(k_index))

    def routed_frames_for_k(self, k_index: int) -> GammaRoutedFrames:
        position = self._k_position(k_index)
        ranks = self.group_ranks
        offsets = self.group_offsets
        local_frames = tuple(self.frames[position, q] for q in range(self.layout.q_count))
        references = tuple(
            tuple(
                self.reference_frames[position, q, :, offsets[group] : offsets[group + 1]]
                for group in range(2)
            )
            for q in range(self.layout.q_count)
        )
        projectors = tuple(
            tuple(
                frame[:, offsets[group] : offsets[group + 1]]
                @ frame[:, offsets[group] : offsets[group + 1]].conj().T
                for group in range(2)
            )
            for frame in local_frames
        )
        return GammaRoutedFrames(
            layout_hash=self.layout.layout_hash,
            thresholds_hash=self.thresholds.identity_hash,
            joint_band_indices=self.joint_band_indices,
            group_dimensions=ranks,
            group_offsets=offsets,
            local_frames_by_q=local_frames,
            reference_frames_by_q_group=references,
            routed_projectors_by_q=projectors,
            route_gaps=tuple(float(item) for item in self.route_gaps[position]),
            frame_hash=self.routed_frame_hashes[position],
            reference_frame_hash=self.routed_reference_frame_hashes[position],
        )

    def assemble_for_k(
        self, k_index: int, *, include_high: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return assemble_gamma_routed_projectors(
            self.routed_frames_for_k(k_index),
            layout=self.layout,
            thresholds=self.thresholds,
            include_high=include_high,
        )

    def authoritative_heff_for_k(self, k_index: int) -> np.ndarray:
        return self.authoritative_heff[self._k_position(k_index)]

    @property
    def candidate_certificate_envelope(self) -> dict[str, object]:
        if not self.candidate_certificate_envelope_json:
            raise ValueError("routed Gamma handoff has no candidate certificate envelope")
        payload = json.loads(self.candidate_certificate_envelope_json)
        if not isinstance(payload, dict):
            raise ValueError("routed Gamma candidate certificate envelope is invalid")
        return payload


ProjectionBasisSpec: TypeAlias = ExplicitLegacyBasisSpec | GammaRoutedBasisSpec


def certify_gamma_routed_basis_spec(
    *,
    candidate_id: str,
    artifact_identity: Mapping[str, Any],
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
    k_indices: Sequence[int],
    routed_frames: Sequence[GammaRoutedFrames],
    authoritative_heff: Any,
    heff_k_indices: Sequence[int],
    closure_certificate_hashes: Sequence[str],
    routing_certificate_hashes: Sequence[str],
    source_hamiltonian_hash: str,
    ordered_q_hashes: Sequence[str],
    raw_action_package_hash: str,
    operations: Mapping[str, Any],
    exactified_actions: Mapping[str, np.ndarray | None],
    presentation: Any,
    required_pairs: Mapping[str, Sequence[tuple[int, int]]],
    candidate_thresholds: Any,
) -> GammaRoutedBasisSpec:
    """Create a routed handoff only after Task-7 candidate certification.

    The certificate is evaluated from the exact assembled persisted frames and
    the same authoritative Heff tensor that will be written to ``basis.npz``.
    """

    from .symmetry.candidate_certificate import (
        CandidateProjectionState,
        CandidateSymmetryStatus,
        candidate_action_package_hash,
        candidate_certificate_envelope,
        certify_candidate_symmetries,
    )

    provisional = GammaRoutedBasisSpec.create(
        artifact_identity=artifact_identity,
        layout=layout,
        thresholds=thresholds,
        k_indices=k_indices,
        routed_frames=routed_frames,
        authoritative_heff=authoritative_heff,
        heff_k_indices=heff_k_indices,
        closure_certificate_hashes=closure_certificate_hashes,
        routing_certificate_hashes=routing_certificate_hashes,
        source_hamiltonian_hash=source_hamiltonian_hash,
        ordered_q_hashes=ordered_q_hashes,
        raw_action_package_hash="0" * 64,
        candidate_certificate_hash="0" * 64,
        candidate_input_identity_hash="0" * 64,
    )
    referenced_k = sorted(
        {
            int(index)
            for pairs in required_pairs.values()
            for pair in pairs
            for index in pair
        }
    )
    provisional.require_k_indices(referenced_k)
    states = {
        k_index: CandidateProjectionState(
            u_low=provisional.assemble_for_k(k_index, include_high=False)[0],
            heff=provisional.authoritative_heff_for_k(k_index),
        )
        for k_index in referenced_k
    }
    certificate = certify_candidate_symmetries(
        candidate_id=str(candidate_id),
        states=states,
        operations=operations,
        exactified_actions=exactified_actions,
        presentation=presentation,
        required_pairs=required_pairs,
        thresholds=candidate_thresholds,
    )
    if certificate.status is not CandidateSymmetryStatus.CERTIFIED:
        raise _reject(
            CandidateRejectionReason.CANDIDATE_SYMMETRY_FAILED,
            "routed Gamma candidate failed Task-7 symmetry certification: "
            + ", ".join(certificate.failures),
        )
    certificate_envelope = candidate_certificate_envelope(certificate)
    certificate_envelope_json = _canonical_json(certificate_envelope)
    certified_action_package_hash = candidate_action_package_hash(
        certificate.input_identity_payload
    )
    if raw_action_package_hash not in {"0" * 64, certified_action_package_hash}:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma raw action package hash does not match the certified "
            "Task-7 candidate input",
        )
    certified_identity = dict(provisional.artifact_identity)
    certified_identity["basis_hash"] = _bound_routed_basis_hash(
        base_basis_hash=provisional.base_basis_hash,
        layout_hash=provisional.layout.layout_hash,
        thresholds_hash=provisional.thresholds.identity_hash,
        k_indices_hash=str(certified_identity["k_indices_hash"]),
        frame_hash=provisional.frame_hash,
        reference_frame_hash=provisional.reference_frame_hash,
        heff_hash=provisional.heff_hash,
        closure_certificate_hashes=provisional.closure_certificate_hashes,
        routing_certificate_hashes=provisional.routing_certificate_hashes,
        source_hamiltonian_hash=provisional.source_hamiltonian_hash,
        ordered_q_hashes=provisional.ordered_q_hashes,
        raw_action_package_hash=certified_action_package_hash,
        candidate_certificate_hash=certificate.certificate_hash,
        candidate_input_identity_hash=certificate.input_identity_hash,
    )
    return replace(
        provisional,
        artifact_identity=certified_identity,
        raw_action_package_hash=certified_action_package_hash,
        candidate_id=certificate.candidate_id,
        candidate_certificate_envelope_json=certificate_envelope_json,
        candidate_certificate_hash=certificate.certificate_hash,
        candidate_input_identity_hash=certificate.input_identity_hash,
    )


def save_gamma_routed_basis_spec(
    path: str | Path, spec: GammaRoutedBasisSpec
) -> None:
    """Persist a routed handoff without object arrays or pickle."""

    if not spec.candidate_id or not spec.candidate_certificate_envelope_json:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma basis cannot persist an uncertified candidate",
        )
    payload: dict[str, np.ndarray] = {
        "projection_basis_kind": np.asarray(spec.projection_basis_kind),
        "projection_basis_handoff_version": np.asarray(spec.handoff_version),
        "base_basis_hash": np.asarray(spec.base_basis_hash),
        "layout_payload": np.asarray(_canonical_json(spec.layout.to_payload())),
        "layout_hash": np.asarray(spec.layout.layout_hash),
        "routing_thresholds_payload": np.asarray(
            _canonical_json(spec.thresholds.to_payload())
        ),
        "routing_thresholds_hash": np.asarray(spec.thresholds.identity_hash),
        "joint_band_indices": np.asarray(spec.joint_band_indices, dtype=np.int64),
        "group_ranks": np.asarray(spec.group_ranks, dtype=np.int64),
        "group_offsets": np.asarray(spec.group_offsets, dtype=np.int64),
        "k_indices": np.asarray(spec.k_indices, dtype=np.int64),
        "frames": spec.frames,
        "reference_frames": spec.reference_frames,
        "routed_frame_hashes": np.asarray(spec.routed_frame_hashes),
        "routed_reference_frame_hashes": np.asarray(
            spec.routed_reference_frame_hashes
        ),
        "frame_hash": np.asarray(spec.frame_hash),
        "reference_frame_hash": np.asarray(spec.reference_frame_hash),
        "route_gaps": spec.route_gaps,
        "heff_k_indices": np.asarray(spec.heff_k_indices, dtype=np.int64),
        "authoritative_heff": spec.authoritative_heff,
        "heff_hash": np.asarray(spec.heff_hash),
        "closure_certificate_hashes": np.asarray(spec.closure_certificate_hashes),
        "routing_certificate_hashes": np.asarray(spec.routing_certificate_hashes),
        "source_hamiltonian_hash": np.asarray(spec.source_hamiltonian_hash),
        "ordered_q_hashes": np.asarray(spec.ordered_q_hashes),
        "raw_action_package_hash": np.asarray(spec.raw_action_package_hash),
        "candidate_id": np.asarray(spec.candidate_id),
        "candidate_certificate_envelope_json": np.asarray(
            spec.candidate_certificate_envelope_json
        ),
        "candidate_certificate_hash": np.asarray(spec.candidate_certificate_hash),
        "candidate_input_identity_hash": np.asarray(
            spec.candidate_input_identity_hash
        ),
        "handoff_identity_hash": np.asarray(spec.handoff_identity_hash),
    }
    for name, value in spec.artifact_identity.items():
        array = np.asarray(value)
        if array.shape != () or array.dtype.hasobject:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"projection identity field {name} is not a numeric/string scalar",
            )
        payload[str(name)] = array
    if any(value.dtype.hasobject for value in payload.values()):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma basis archives cannot contain object arrays",
        )
    np.savez(Path(path), **payload)


def _restore_layout(payload_json: str, stored_hash: str) -> GammaRowLayout:
    try:
        payload = json.loads(payload_json)
        if not isinstance(payload, Mapping):
            raise TypeError("layout payload is not a mapping")
        qsets = payload["ordered_qsets"]
        layers = payload["num_layer_list"]
        width = int(payload["uniform_orbital_count"])
        layout = GammaRowLayout.build(
            qsets=qsets,
            num_layer_list=layers,
            num_orb_per_layer_list=tuple(
                tuple(width for _ in range(int(count))) for count in layers
            ),
            spin_convention=str(payload["spin_scope"]),
            source_basis_hash=str(payload["tapw_source_basis_hash"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma layout payload cannot be reconstructed",
        ) from error
    if layout.layout_hash != stored_hash or layout.to_payload() != dict(payload):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma layout payload/hash mismatch",
        )
    return layout


def _restore_thresholds(payload_json: str, stored_hash: str) -> GammaRoutingThresholds:
    try:
        payload = json.loads(payload_json)
        if not isinstance(payload, Mapping):
            raise TypeError("threshold payload is not a mapping")
        if payload.get("schema") != GammaRoutingThresholds.SCHEMA:
            raise ValueError("unsupported threshold schema")
        thresholds = GammaRoutingThresholds.from_normalized_config(
            {key: value for key, value in payload.items() if key != "schema"}
        )
    except (TypeError, ValueError) as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma threshold payload cannot be reconstructed",
        ) from error
    if thresholds.identity_hash != stored_hash or thresholds.to_payload() != dict(payload):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma threshold payload/hash mismatch",
        )
    return thresholds


def load_gamma_routed_basis_spec(path: str | Path) -> GammaRoutedBasisSpec:
    """Load and fully revalidate one numeric routed Gamma handoff."""

    archive_path = Path(path)
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            _require_exact_archive_keys(archive.files)
            payload = {
                name: np.array(archive[name], copy=True)
                for name in _GAMMA_ROUTED_ARCHIVE_KEYS
            }
        _require_disk_dtypes(payload)
    except GammaRoutingError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma basis archive cannot be loaded without pickle: {archive_path}",
        ) from error
    if _required_text(payload, "projection_basis_kind") != GAMMA_ROUTED_BASIS_KIND:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "projection basis discriminator is not gamma_routed",
        )
    if (
        _required_text(payload, "projection_basis_handoff_version")
        != GAMMA_ROUTED_BASIS_HANDOFF_VERSION
    ):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "unsupported routed Gamma handoff version",
        )
    layout = _restore_layout(
        _required_text(payload, "layout_payload"),
        _required_text(payload, "layout_hash"),
    )
    thresholds = _restore_thresholds(
        _required_text(payload, "routing_thresholds_payload"),
        _required_text(payload, "routing_thresholds_hash"),
    )
    k_indices = _strict_index_tuple(payload["k_indices"], name="k_indices")
    heff_k_indices = _strict_index_tuple(
        payload["heff_k_indices"], name="heff_k_indices"
    )
    artifact_identity = {
        field: np.asarray(payload[field]).item()
        for field in PROJECTION_ARTIFACT_IDENTITY_FIELDS
    }
    artifact_identity["source_hamiltonian_hash"] = _required_text(
        payload, "source_hamiltonian_hash"
    )
    spec = GammaRoutedBasisSpec(
        artifact_identity=artifact_identity,
        base_basis_hash=_required_text(payload, "base_basis_hash"),
        layout=layout,
        thresholds=thresholds,
        joint_band_indices=tuple(
            int(item) for item in payload["joint_band_indices"].tolist()
        ),
        group_ranks=tuple(
            int(item) for item in payload["group_ranks"].tolist()
        ),  # type: ignore[arg-type]
        group_offsets=tuple(
            int(item) for item in payload["group_offsets"].tolist()
        ),  # type: ignore[arg-type]
        k_indices=k_indices,
        frames=payload["frames"],
        reference_frames=payload["reference_frames"],
        routed_frame_hashes=_strict_hash_tuple(
            payload["routed_frame_hashes"],
            name="routed_frame_hashes",
            length=len(k_indices),
        ),
        routed_reference_frame_hashes=_strict_hash_tuple(
            payload["routed_reference_frame_hashes"],
            name="routed_reference_frame_hashes",
            length=len(k_indices),
        ),
        frame_hash=_required_text(payload, "frame_hash"),
        reference_frame_hash=_required_text(payload, "reference_frame_hash"),
        route_gaps=payload["route_gaps"],
        heff_k_indices=heff_k_indices,
        authoritative_heff=payload["authoritative_heff"],
        heff_hash=_required_text(payload, "heff_hash"),
        closure_certificate_hashes=_strict_hash_tuple(
            payload["closure_certificate_hashes"],
            name="closure_certificate_hashes",
            length=len(k_indices),
        ),
        routing_certificate_hashes=_strict_hash_tuple(
            payload["routing_certificate_hashes"],
            name="routing_certificate_hashes",
            length=len(k_indices),
        ),
        source_hamiltonian_hash=_required_text(payload, "source_hamiltonian_hash"),
        ordered_q_hashes=tuple(
            str(item) for item in payload["ordered_q_hashes"].tolist()
        ),  # type: ignore[arg-type]
        raw_action_package_hash=_required_text(payload, "raw_action_package_hash"),
        candidate_id=_required_text(payload, "candidate_id"),
        candidate_certificate_envelope_json=_required_text(
            payload, "candidate_certificate_envelope_json"
        ),
        candidate_certificate_hash=_required_text(
            payload, "candidate_certificate_hash"
        ),
        candidate_input_identity_hash=_required_text(
            payload, "candidate_input_identity_hash"
        ),
    )
    if _required_text(payload, "handoff_identity_hash") != spec.handoff_identity_hash:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma handoff identity hash mismatch",
        )
    return spec
