"""Typed, fail-closed projection-basis handoffs.

The automatic-Gamma variant certifies routed selection frames and canonical
SCDM model frames for the same subspace.  Effective Hamiltonians are bound to
the model frame; the numeric archive and every identity are revalidated on
load.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence, TypeAlias

import numpy as np

from .blocks.gamma_layout import (
    GammaCertifiedRawAction,
    GammaRoutedFrames,
    GammaRoutingError,
    GammaRoutingThresholds,
    GammaRowLayout,
    assemble_gamma_routed_projectors,
    gamma_certified_action_route_contract,
)
from .blocks.blocks import (
    GammaModelAnchorSpec,
    GammaModelFrames,
    _gamma_reference_matrix,
    align_eigenstates,
)
from .identity import (
    IDENTITY_SCHEMA,
    KP_VERSION,
    PROJECTION_ARTIFACT_IDENTITY_FIELDS,
    PROJECTION_BASIS_SCHEMA_VERSION,
    hash_array,
    hash_mapping,
)
from .projection_selection import CandidateRejectionReason


GAMMA_ROUTED_BASIS_HANDOFF_VERSION = "kp_project_gamma_routed_handoff_v3"
EXPLICIT_LEGACY_BASIS_KIND = "explicit_legacy"
GAMMA_ROUTED_BASIS_KIND = "gamma_routed"
GAMMA_SAMPLED_K_MATCH_TOLERANCE = 1.0e-10
GAMMA_SAMPLED_K_GRAY_TOLERANCE = 1.0e-8


def gamma_sampled_k_route_contract(
    kpoints: Any,
    pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Bind a sampled-domain k route to its numeric k-point set."""

    points = np.asarray(kpoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= 0 or points.shape[1] != 2:
        raise ValueError("Gamma sampled kpoints must have shape (Nk, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("Gamma sampled kpoints must be finite")
    canonical_pairs = sorted((int(target), int(source)) for target, source in pairs)
    if not canonical_pairs or len(set(canonical_pairs)) != len(canonical_pairs):
        raise ValueError("Gamma sampled k route must contain unique nonempty pairs")
    nk = int(points.shape[0])
    if any(
        target < 0 or source < 0 or target >= nk or source >= nk
        for target, source in canonical_pairs
    ):
        raise ValueError("Gamma sampled k route references k outside its numeric domain")
    return {
        "schema": "kp.gamma-sampled-k-route.v1",
        "coverage": "actual_sampled_k_set_intersection",
        "kpoints_hash": hash_array(points),
        "pairs": [list(pair) for pair in canonical_pairs],
        "match_tolerance": GAMMA_SAMPLED_K_MATCH_TOLERANCE,
        "gray_tolerance": GAMMA_SAMPLED_K_GRAY_TOLERANCE,
    }

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
        "kpoints",
        "kpoints_hash",
        "frames",
        "reference_frames",
        "routed_frame_hashes",
        "routed_reference_frame_hashes",
        "frame_hash",
        "reference_frame_hash",
        "route_gaps",
        "model_reference_k_index",
        "model_reference_projector",
        "model_reference_projector_hash",
        "model_anchor_spec_payload",
        "model_anchor_spec_hash",
        "model_frames",
        "model_frame_hashes",
        "model_frame_hash",
        "routing_to_model",
        "routing_to_model_hashes",
        "routing_to_model_hash",
        "bridge_certificate_hash",
        "routed_heff",
        "routed_heff_hash",
        "heff_covariance_residuals",
        "heff_covariance_residuals_hash",
        "heff_covariance_tolerance",
        "heff_covariance_evidence_hash",
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
        # Explicit projection handoffs also bind their sampled k-domain.
        # This field predates the routed Gamma archive and is not variant-only.
        "kpoints_hash",
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
_INT64_SCALAR_FIELDS = frozenset({"model_reference_k_index"})
_COMPLEX128_FIELDS = frozenset(
    {
        "frames",
        "reference_frames",
        "model_reference_projector",
        "model_frames",
        "routing_to_model",
        "routed_heff",
        "authoritative_heff",
    }
)
_FLOAT64_FIELDS = frozenset({"route_gaps", "kpoints", "heff_covariance_residuals"})
_FLOAT64_SCALAR_FIELDS = frozenset({"heff_covariance_tolerance"})
_SHA256_SCALAR_FIELDS = frozenset(
    {
        "base_basis_hash",
        "layout_hash",
        "routing_thresholds_hash",
        "frame_hash",
        "reference_frame_hash",
        "model_reference_projector_hash",
        "model_anchor_spec_hash",
        "model_frame_hash",
        "routing_to_model_hash",
        "bridge_certificate_hash",
        "routed_heff_hash",
        "heff_covariance_residuals_hash",
        "heff_covariance_evidence_hash",
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
        "kpoints_hash",
    }
)
_SHA256_VECTOR_FIELDS = frozenset(
    {
        "routed_frame_hashes",
        "routed_reference_frame_hashes",
        "model_frame_hashes",
        "routing_to_model_hashes",
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


def _strict_integral_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    try:
        items = tuple(value)
    except TypeError as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} must contain strict integers (bool excluded)",
        ) from error
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
        for item in items
    ):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} must contain strict integers (bool excluded)",
        )
    return tuple(int(item) for item in items)


def _strict_integral_scalar(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"routed Gamma {name} must be a strict integer (bool excluded)",
        )
    return int(value)


def _strict_positive_real_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"{name} must be a strict numeric value that is finite and positive",
        )
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"{name} must be a strict numeric value that is finite and positive",
        )
    return result


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
    identity_schema = _required_text(payload, "identity_schema")
    package_version = _required_text(payload, "package_version")
    if identity_schema != IDENTITY_SCHEMA:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma archive identity_schema is unsupported",
        )
    if package_version != KP_VERSION:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma archive package_version differs from the active KP package",
        )
    for name in _INT64_FIELDS:
        value = np.asarray(payload[name])
        expected_shape = {
            "group_ranks": (2,),
            "group_offsets": (3,),
        }.get(name)
        if (
            value.dtype != np.dtype(np.int64)
            or value.ndim != 1
            or (expected_shape is not None and value.shape != expected_shape)
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must be a canonical int64 vector",
            )
    for name in _INT64_SCALAR_FIELDS:
        value = np.asarray(payload[name])
        if value.shape != () or value.dtype != np.dtype(np.int64):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must be an int64 scalar",
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
    for name in _FLOAT64_SCALAR_FIELDS:
        value = np.asarray(payload[name])
        if value.shape != () or value.dtype != np.dtype(np.float64):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma archive field {name} must be a float64 scalar",
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
    if int(schema_version.item()) != PROJECTION_BASIS_SCHEMA_VERSION:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma archive schema_version is unsupported",
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


def _model_global_columns(
    *, q_index: int, q_count: int, group_ranks: Sequence[int]
) -> np.ndarray:
    return np.asarray(
        [
            q_count * sum(group_ranks[:group]) + orbital * q_count + q_index
            for group, rank in enumerate(group_ranks)
            for orbital in range(int(rank))
        ],
        dtype=np.intp,
    )


def assemble_gamma_model_frame(
    model_frames: GammaModelFrames,
    *,
    layout: GammaRowLayout,
) -> np.ndarray:
    """Assemble compact local model frames in sector-orbital-Q column order."""

    if not isinstance(model_frames, GammaModelFrames):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "Gamma model assembly requires a GammaModelFrames record",
        )
    if model_frames.layout_hash != layout.layout_hash:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "Gamma model frame layout hash does not match the canonical Gamma layout",
        )
    ranks = tuple(int(value) for value in model_frames.group_ranks)
    local_rank = sum(ranks)
    if len(model_frames.local_frames_by_q) != layout.q_count:
        raise _reject(
            CandidateRejectionReason.HANDOFF_K_COVERAGE,
            "Gamma model frames do not cover every ordered Q",
        )
    assembled = np.zeros(
        (layout.full_dimension, layout.q_count * local_rank),
        dtype=np.complex128,
    )
    for q_index, raw_frame in enumerate(model_frames.local_frames_by_q):
        frame = np.asarray(raw_frame, dtype=np.complex128)
        if frame.shape != (layout.same_q_dimension, local_rank) or not np.all(
            np.isfinite(frame)
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"Gamma model q={q_index} frame shape or values are invalid",
            )
        rows = layout.same_q_full_rows(q_index)
        columns = _model_global_columns(
            q_index=q_index,
            q_count=layout.q_count,
            group_ranks=ranks,
        )
        assembled[np.ix_(rows, columns)] = frame
    return assembled


def _assemble_model_tensor(
    compact_frames: np.ndarray,
    *,
    layout: GammaRowLayout,
    group_ranks: Sequence[int],
) -> np.ndarray:
    local_rank = sum(int(value) for value in group_ranks)
    raw = np.asarray(compact_frames, dtype=np.complex128)
    if raw.shape != (layout.q_count, layout.same_q_dimension, local_rank):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "compact Gamma model frame has an invalid layout shape",
        )
    assembled = np.zeros(
        (layout.full_dimension, layout.q_count * local_rank),
        dtype=np.complex128,
    )
    for q_index in range(layout.q_count):
        rows = layout.same_q_full_rows(q_index)
        columns = _model_global_columns(
            q_index=q_index,
            q_count=layout.q_count,
            group_ranks=group_ranks,
        )
        assembled[np.ix_(rows, columns)] = raw[q_index]
    return assembled


def _assemble_bridge_tensor(
    compact_bridge: np.ndarray,
    *,
    q_count: int,
    group_ranks: Sequence[int],
) -> np.ndarray:
    local_rank = sum(int(value) for value in group_ranks)
    raw = np.asarray(compact_bridge, dtype=np.complex128)
    if raw.shape != (q_count, local_rank, local_rank):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "compact Gamma routing-to-model bridge has an invalid layout shape",
        )
    assembled = np.zeros(
        (q_count * local_rank, q_count * local_rank), dtype=np.complex128
    )
    for q_index in range(q_count):
        columns = _model_global_columns(
            q_index=q_index,
            q_count=q_count,
            group_ranks=group_ranks,
        )
        assembled[np.ix_(columns, columns)] = raw[q_index]
    return assembled


def _bridge_certificate_hash(
    *,
    layout_hash: str,
    thresholds_hash: str,
    frame_hash: str,
    model_frame_hash: str,
    model_reference_k_index: int,
    model_reference_projector_hash: str,
    model_anchor_spec_hash: str,
    routing_to_model_hash: str,
    k_indices_hash: str,
    kpoints_hash: str,
    model_heff_hash: str,
    routed_heff_hash: str,
    heff_covariance_evidence_hash: str,
    projector_tolerance: float,
) -> str:
    return hash_mapping(
        {
            "schema": "kp.gamma-routing-to-model-bridge-certificate.v2",
            "direction": "routing_to_model",
            "layout_hash": str(layout_hash),
            "thresholds_hash": str(thresholds_hash),
            "routed_frame_hash": str(frame_hash),
            "model_frame_hash": str(model_frame_hash),
            "model_reference_k_index": int(model_reference_k_index),
            "model_reference_projector_hash": str(model_reference_projector_hash),
            "model_anchor_spec_hash": str(model_anchor_spec_hash),
            "routing_to_model_hash": str(routing_to_model_hash),
            "k_indices_hash": str(k_indices_hash),
            "kpoints_hash": str(kpoints_hash),
            "model_heff_hash": str(model_heff_hash),
            "routed_heff_hash": str(routed_heff_hash),
            "heff_covariance_evidence_hash": str(heff_covariance_evidence_hash),
            "projector_tolerance": float(projector_tolerance),
        }
    )


def _heff_covariance_evidence_hash(
    *,
    k_indices_hash: str,
    kpoints_hash: str,
    routing_to_model_hash: str,
    model_heff_hash: str,
    routed_heff_hash: str,
    residuals_hash: str,
    tolerance: float,
) -> str:
    return hash_mapping(
        {
            "schema": "kp.gamma-heff-covariance-evidence.v1",
            "direction": "routing_to_model",
            "k_indices_hash": str(k_indices_hash),
            "kpoints_hash": str(kpoints_hash),
            "routing_to_model_hash": str(routing_to_model_hash),
            "model_heff_hash": str(model_heff_hash),
            "routed_heff_hash": str(routed_heff_hash),
            "residuals_hash": str(residuals_hash),
            "tolerance": float(tolerance),
        }
    )


def _heff_covariance_residuals(
    *,
    routed_heff: np.ndarray,
    model_heff: np.ndarray,
    compact_bridge: np.ndarray,
    q_count: int,
    group_ranks: Sequence[int],
) -> np.ndarray:
    if routed_heff.shape != model_heff.shape or routed_heff.ndim != 3:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed/model Heff covariance tensors must have identical rank-3 shapes",
        )
    if compact_bridge.shape[0] != routed_heff.shape[0]:
        raise _reject(
            CandidateRejectionReason.HANDOFF_K_COVERAGE,
            "routed/model Heff covariance evidence does not cover every k",
        )
    residuals: list[float] = []
    for k_position in range(routed_heff.shape[0]):
        bridge = _assemble_bridge_tensor(
            compact_bridge[k_position],
            q_count=q_count,
            group_ranks=group_ranks,
        )
        expected = bridge.conj().T @ routed_heff[k_position] @ bridge
        denominator = float(np.linalg.norm(model_heff[k_position], ord="fro"))
        if denominator == 0.0:
            denominator = 1.0
        residuals.append(
            float(
                np.linalg.norm(model_heff[k_position] - expected, ord="fro")
                / denominator
            )
        )
    return np.asarray(residuals, dtype=np.float64)


def _validate_model_anchor_contract(
    anchor: GammaModelAnchorSpec,
    *,
    layout: GammaRowLayout,
    joint_band_indices: Sequence[int],
    group_ranks: Sequence[int],
) -> None:
    if not isinstance(anchor, GammaModelAnchorSpec):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma handoff requires a GammaModelAnchorSpec",
        )
    try:
        anchor.to_payload()
    except (TypeError, ValueError) as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"Gamma model anchor contract is invalid: {error}",
        ) from error
    if (
        anchor.layout_hash != layout.layout_hash
        or anchor.same_q_dimension != layout.same_q_dimension
        or anchor.joint_band_indices != tuple(joint_band_indices)
        or anchor.group_ranks != tuple(group_ranks)
        or anchor.reference_q_index >= layout.q_count
    ):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "Gamma model anchor contract differs from routed layout/bands/ranks",
        )


def _validate_dual_frame_geometry(
    *,
    routed_frames: np.ndarray,
    model_frames: np.ndarray,
    routing_to_model: np.ndarray,
    anchor: GammaModelAnchorSpec,
    layout: GammaRowLayout,
    thresholds: GammaRoutingThresholds,
) -> None:
    tolerance = float(thresholds.projector_residual)
    local_rank = int(model_frames.shape[-1])
    identity = np.eye(local_rank, dtype=np.complex128)
    try:
        phi = _gamma_reference_matrix(
            anchor.resolved_reference_terms,
            row_count=layout.same_q_dimension,
        )
    except (IndexError, TypeError, ValueError) as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"Gamma model anchor references are invalid: {error}",
        ) from error
    for k_position in range(model_frames.shape[0]):
        for q_index in range(layout.q_count):
            routed = routed_frames[k_position, q_index]
            model = model_frames[k_position, q_index]
            bridge = routing_to_model[k_position, q_index]
            canonical, _ = align_eigenstates(model, phi)
            residuals = {
                "model_orthonormality": np.linalg.norm(
                    model.conj().T @ model - identity, ord="fro"
                ),
                "bridge_left_unitarity": np.linalg.norm(
                    bridge.conj().T @ bridge - identity, ord="fro"
                ),
                "bridge_right_unitarity": np.linalg.norm(
                    bridge @ bridge.conj().T - identity, ord="fro"
                ),
                "bridge_reconstruction": np.linalg.norm(
                    routed @ bridge - model, ord="fro"
                ),
                "projector": np.linalg.norm(
                    routed @ routed.conj().T - model @ model.conj().T,
                    ord="fro",
                ),
                "canonical_scdm": np.linalg.norm(canonical - model, ord="fro"),
            }
            worst_name, worst_value = max(
                residuals.items(), key=lambda item: float(item[1])
            )
            if not np.isfinite(worst_value) or float(worst_value) > tolerance:
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    "Gamma dual-frame geometry failed at "
                    f"k_position={k_position}, q={q_index}: "
                    f"{worst_name}={float(worst_value):.3e} exceeds {tolerance:.3e}",
                )


def _bound_routed_basis_hash(
    *,
    base_basis_hash: str,
    layout_hash: str,
    thresholds_hash: str,
    k_indices_hash: str,
    kpoints_hash: str,
    frame_hash: str,
    reference_frame_hash: str,
    model_reference_k_index: int,
    model_reference_projector_hash: str,
    model_anchor_spec_hash: str,
    model_frame_hash: str,
    routing_to_model_hash: str,
    bridge_certificate_hash: str,
    routed_heff_hash: str,
    heff_covariance_evidence_hash: str,
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
            "schema": "kp.gamma-routed-projection-basis.v3",
            "base_basis_hash": str(base_basis_hash),
            "layout_hash": str(layout_hash),
            "thresholds_hash": str(thresholds_hash),
            "k_indices_hash": str(k_indices_hash),
            "kpoints_hash": str(kpoints_hash),
            "frame_hash": str(frame_hash),
            "reference_frame_hash": str(reference_frame_hash),
            "model_reference_k_index": int(model_reference_k_index),
            "model_reference_projector_hash": str(model_reference_projector_hash),
            "model_anchor_spec_hash": str(model_anchor_spec_hash),
            "model_frame_hash": str(model_frame_hash),
            "routing_to_model_hash": str(routing_to_model_hash),
            "bridge_certificate_hash": str(bridge_certificate_hash),
            "routed_heff_hash": str(routed_heff_hash),
            "heff_covariance_evidence_hash": str(
                heff_covariance_evidence_hash
            ),
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
    """Exact routed/model frames and model-gauge projected Hamiltonians."""

    artifact_identity: Mapping[str, Any]
    base_basis_hash: str
    layout: GammaRowLayout
    thresholds: GammaRoutingThresholds
    joint_band_indices: tuple[int, ...]
    group_ranks: tuple[int, int]
    group_offsets: tuple[int, int, int]
    k_indices: tuple[int, ...]
    kpoints: np.ndarray
    kpoints_hash: str
    frames: np.ndarray
    reference_frames: np.ndarray
    routed_frame_hashes: tuple[str, ...]
    routed_reference_frame_hashes: tuple[str, ...]
    frame_hash: str
    reference_frame_hash: str
    route_gaps: np.ndarray
    model_reference_k_index: int
    model_reference_projector: np.ndarray
    model_reference_projector_hash: str
    model_anchor_spec: GammaModelAnchorSpec
    model_frames: np.ndarray
    model_frame_hashes: tuple[str, ...]
    model_frame_hash: str
    routing_to_model: np.ndarray
    routing_to_model_hashes: tuple[str, ...]
    routing_to_model_hash: str
    bridge_certificate_hash: str
    routed_heff: np.ndarray
    routed_heff_hash: str
    heff_covariance_residuals: np.ndarray
    heff_covariance_residuals_hash: str
    heff_covariance_tolerance: float
    heff_covariance_evidence_hash: str
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
        k_indices = _strict_integral_tuple(self.k_indices, name="k_indices")
        heff_k_indices = _strict_integral_tuple(
            self.heff_k_indices, name="heff_k_indices"
        )
        joint = _strict_integral_tuple(
            self.joint_band_indices, name="joint_band_indices"
        )
        ranks = _strict_integral_tuple(self.group_ranks, name="group_ranks")
        offsets = _strict_integral_tuple(self.group_offsets, name="group_offsets")
        model_reference_k_index = _strict_integral_scalar(
            self.model_reference_k_index,
            name="model_reference_k_index",
        )
        object.__setattr__(self, "k_indices", k_indices)
        object.__setattr__(self, "heff_k_indices", heff_k_indices)
        object.__setattr__(self, "joint_band_indices", joint)
        object.__setattr__(self, "group_ranks", ranks)
        object.__setattr__(self, "group_offsets", offsets)
        object.__setattr__(
            self, "model_reference_k_index", model_reference_k_index
        )
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
        if model_reference_k_index not in k_indices:
            raise _reject(
                CandidateRejectionReason.HANDOFF_K_COVERAGE,
                "routed Gamma model reference k is outside the persisted k mapping",
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
        kpoints = _freeze(self.kpoints, dtype=np.float64, name="kpoints")
        references = _freeze(
            self.reference_frames,
            dtype=np.complex128,
            name="reference_frames",
        )
        route_gaps = _freeze(self.route_gaps, dtype=np.float64, name="route_gaps")
        model_reference_projector = _freeze(
            self.model_reference_projector,
            dtype=np.complex128,
            name="model_reference_projector",
        )
        model_frames = _freeze(
            self.model_frames,
            dtype=np.complex128,
            name="model_frames",
        )
        routing_to_model = _freeze(
            self.routing_to_model,
            dtype=np.complex128,
            name="routing_to_model",
        )
        routed_heff = _freeze(
            self.routed_heff,
            dtype=np.complex128,
            name="routed_heff",
        )
        covariance_residuals = _freeze(
            self.heff_covariance_residuals,
            dtype=np.float64,
            name="heff_covariance_residuals",
        )
        covariance_tolerance = _strict_positive_real_scalar(
            self.heff_covariance_tolerance,
            name="Gamma Heff covariance tolerance",
        )
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
        expected_bridge_shape = (
            len(k_indices),
            self.layout.q_count,
            sum(ranks),
            sum(ranks),
        )
        model_dim = self.layout.q_count * sum(ranks)
        if kpoints.shape != (len(k_indices), 2):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma kpoints must have shape (Nk, 2) in routed-k order",
            )
        if frames.shape != expected_frame_shape or references.shape != expected_frame_shape:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma frame tensors do not match layout/k/group dimensions",
            )
        if model_frames.shape != expected_frame_shape:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma model frame tensor does not match layout/k/group dimensions",
            )
        if routing_to_model.shape != expected_bridge_shape:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma routing-to-model tensor does not match layout/k/group dimensions",
            )
        if model_reference_projector.shape != (
            self.layout.same_q_dimension,
            self.layout.same_q_dimension,
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma model reference projector has an invalid same-Q shape",
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
        if routed_heff.shape != heff.shape or covariance_residuals.shape != (
            len(k_indices),
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma routed-Heff covariance evidence shape is invalid",
            )
        if tuple(self.ordered_q_hashes) != tuple(self.layout.ordered_qset_hashes):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma ordered-Q identity does not match the row layout",
            )
        _validate_model_anchor_contract(
            self.model_anchor_spec,
            layout=self.layout,
            joint_band_indices=joint,
            group_ranks=ranks,
        )
        if (
            self.model_reference_projector_hash
            != hash_array(model_reference_projector)
            or self.model_reference_projector_hash
            != self.model_anchor_spec.reference_projector_hash
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma model reference projector does not match the anchor projector hash",
            )
        reference_position = k_indices.index(model_reference_k_index)
        reference_model_frame = model_frames[
            reference_position, self.model_anchor_spec.reference_q_index
        ]
        reference_projector_residual = float(
            np.linalg.norm(
                reference_model_frame @ reference_model_frame.conj().T
                - model_reference_projector,
                ord="fro",
            )
        )
        reference_certificate_residual = max(
            reference_projector_residual,
            float(
                np.linalg.norm(
                    model_reference_projector - model_reference_projector.conj().T,
                    ord="fro",
                )
            ),
            float(
                np.linalg.norm(
                    model_reference_projector @ model_reference_projector
                    - model_reference_projector,
                    ord="fro",
                )
            ),
            abs(float(np.trace(model_reference_projector).real) - sum(ranks)),
        )
        if reference_certificate_residual > self.thresholds.projector_residual:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma model_reference_k_index does not select the anchor reference projector",
            )
        scalar_identities = (
            self.source_hamiltonian_hash,
            self.kpoints_hash,
            self.raw_action_package_hash,
            self.candidate_certificate_hash,
            self.candidate_input_identity_hash,
            self.model_anchor_spec.identity_hash,
            self.model_reference_projector_hash,
            self.model_frame_hash,
            self.routing_to_model_hash,
            self.bridge_certificate_hash,
            self.routed_heff_hash,
            self.heff_covariance_residuals_hash,
            self.heff_covariance_evidence_hash,
            *self.ordered_q_hashes,
            *self.closure_certificate_hashes,
            *self.routing_certificate_hashes,
            *self.routed_frame_hashes,
            *self.routed_reference_frame_hashes,
            *self.model_frame_hashes,
            *self.routing_to_model_hashes,
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
                self.model_frame_hashes,
                self.routing_to_model_hashes,
            )
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma frame/closure/routing identities do not cover ordered k",
            )
        if self.model_frame_hash != hash_array(model_frames):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma aggregate model-frame hash mismatch",
            )
        if self.routing_to_model_hash != hash_array(routing_to_model):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma aggregate routing-to-model hash mismatch",
            )
        if tuple(hash_array(row) for row in model_frames) != tuple(
            self.model_frame_hashes
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma per-k model-frame hash mismatch",
            )
        if tuple(hash_array(row) for row in routing_to_model) != tuple(
            self.routing_to_model_hashes
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma per-k routing-to-model hash mismatch",
            )
        if self.routed_heff_hash != hash_array(routed_heff):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma routed-Heff covariance tensor hash mismatch",
            )
        if self.heff_covariance_residuals_hash != hash_array(covariance_residuals):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma Heff covariance residual hash mismatch",
            )
        recomputed_covariance_residuals = _heff_covariance_residuals(
            routed_heff=routed_heff,
            model_heff=heff,
            compact_bridge=routing_to_model,
            q_count=self.layout.q_count,
            group_ranks=ranks,
        )
        if (
            not np.allclose(
                recomputed_covariance_residuals,
                covariance_residuals,
                rtol=0.0,
                atol=np.finfo(np.float64).eps,
            )
            or float(np.max(recomputed_covariance_residuals))
            > covariance_tolerance
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma routed/model Heff covariance evidence failed its numerical gate",
            )
        k_indices_hash = hash_array(np.asarray(k_indices, dtype=np.int64))
        expected_covariance_evidence_hash = _heff_covariance_evidence_hash(
            k_indices_hash=k_indices_hash,
            kpoints_hash=self.kpoints_hash,
            routing_to_model_hash=self.routing_to_model_hash,
            model_heff_hash=self.heff_hash,
            routed_heff_hash=self.routed_heff_hash,
            residuals_hash=self.heff_covariance_residuals_hash,
            tolerance=covariance_tolerance,
        )
        if self.heff_covariance_evidence_hash != expected_covariance_evidence_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma Heff covariance evidence hash mismatch",
            )
        expected_bridge_certificate = _bridge_certificate_hash(
            layout_hash=self.layout.layout_hash,
            thresholds_hash=self.thresholds.identity_hash,
            frame_hash=self.frame_hash,
            model_frame_hash=self.model_frame_hash,
            model_reference_k_index=model_reference_k_index,
            model_reference_projector_hash=self.model_reference_projector_hash,
            model_anchor_spec_hash=self.model_anchor_spec.identity_hash,
            routing_to_model_hash=self.routing_to_model_hash,
            k_indices_hash=k_indices_hash,
            kpoints_hash=self.kpoints_hash,
            model_heff_hash=self.heff_hash,
            routed_heff_hash=self.routed_heff_hash,
            heff_covariance_evidence_hash=self.heff_covariance_evidence_hash,
            projector_tolerance=self.thresholds.projector_residual,
        )
        if self.bridge_certificate_hash != expected_bridge_certificate:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma routing-to-model bridge certificate hash mismatch",
            )
        _validate_dual_frame_geometry(
            routed_frames=frames,
            model_frames=model_frames,
            routing_to_model=routing_to_model,
            anchor=self.model_anchor_spec,
            layout=self.layout,
            thresholds=self.thresholds,
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
                    CandidateSymmetryThresholds,
                    candidate_action_package_hash,
                    verify_candidate_certificate_envelope,
                )

                envelope_raw = json.loads(self.candidate_certificate_envelope_json)
                verified_envelope = verify_candidate_certificate_envelope(envelope_raw)
                verified_threshold_payload = verified_envelope[
                    "certificate_payload"
                ]["thresholds"]
                CandidateSymmetryThresholds(
                    **dict(verified_threshold_payload)
                )
            except (ImportError, KeyError, TypeError, ValueError) as error:
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
                u_low = _assemble_model_tensor(
                    model_frames[position],
                    layout=self.layout,
                    group_ranks=ranks,
                )
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
        if self.kpoints_hash != hash_array(kpoints):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma kpoints hash mismatch",
            )
        artifact_identity = dict(self.artifact_identity)
        if artifact_identity.get("heff_hash") != self.heff_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection identity does not bind the authoritative model Heff",
            )
        if artifact_identity.get("k_indices_hash") != hash_array(
            np.asarray(k_indices, dtype=np.int64)
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection identity does not bind the routed k mapping",
            )
        if artifact_identity.get("kpoints_hash") != self.kpoints_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection identity does not bind the routed kpoints",
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
            kpoints_hash=self.kpoints_hash,
            frame_hash=self.frame_hash,
            reference_frame_hash=self.reference_frame_hash,
            model_reference_k_index=self.model_reference_k_index,
            model_reference_projector_hash=self.model_reference_projector_hash,
            model_anchor_spec_hash=self.model_anchor_spec.identity_hash,
            model_frame_hash=self.model_frame_hash,
            routing_to_model_hash=self.routing_to_model_hash,
            bridge_certificate_hash=self.bridge_certificate_hash,
            routed_heff_hash=self.routed_heff_hash,
            heff_covariance_evidence_hash=self.heff_covariance_evidence_hash,
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
        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "reference_frames", references)
        object.__setattr__(self, "route_gaps", route_gaps)
        object.__setattr__(
            self, "model_reference_projector", model_reference_projector
        )
        object.__setattr__(self, "model_frames", model_frames)
        object.__setattr__(self, "routing_to_model", routing_to_model)
        object.__setattr__(self, "routed_heff", routed_heff)
        object.__setattr__(
            self, "heff_covariance_residuals", covariance_residuals
        )
        object.__setattr__(
            self, "heff_covariance_tolerance", covariance_tolerance
        )
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
        kpoints: Any,
        routed_frames: Sequence[GammaRoutedFrames],
        model_reference_k_index: int,
        model_anchor_spec: GammaModelAnchorSpec,
        model_frames: Sequence[GammaModelFrames],
        model_reference_projector: Any,
        routed_heff: Any,
        heff_covariance_tolerance: float,
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
        indices = _strict_integral_tuple(k_indices, name="k_indices")
        heff_indices = _strict_integral_tuple(
            heff_k_indices, name="heff_k_indices"
        )
        routed = tuple(routed_frames)
        model_rows = tuple(model_frames)
        if len(routed) != len(indices) or not routed:
            raise _reject(
                CandidateRejectionReason.HANDOFF_K_COVERAGE,
                "routed Gamma frame rows do not cover ordered k indices",
            )
        if len(model_rows) != len(indices):
            raise _reject(
                CandidateRejectionReason.HANDOFF_K_COVERAGE,
                "Gamma model frame rows do not cover ordered k indices",
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
        _validate_model_anchor_contract(
            model_anchor_spec,
            layout=layout,
            joint_band_indices=first.joint_band_indices,
            group_ranks=first.group_dimensions,
        )
        if any(
            (
                not isinstance(item, GammaModelFrames)
                or item.anchor_spec_identity_hash != model_anchor_spec.identity_hash
                or item.layout_hash != layout.layout_hash
                or item.joint_band_indices != first.joint_band_indices
                or item.group_ranks != first.group_dimensions
                or len(item.local_frames_by_q) != layout.q_count
                or any(
                    residual > model_anchor_spec.orthonormality_tolerance
                    for residual in item.orthonormality_residuals_by_q
                )
                or any(
                    residual > model_anchor_spec.projector_tolerance
                    for residual in item.projector_residuals_by_q
                )
            )
            for item in model_rows
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma per-k model-frame metadata differs from its anchor/routing contract",
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
            compact_model_frames = np.stack(
                [np.stack(item.local_frames_by_q, axis=0) for item in model_rows],
                axis=0,
            )
            routing_to_model = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(routed[k].local_frames_by_q[q]).conj().T
                            @ np.asarray(model_rows[k].local_frames_by_q[q])
                            for q in range(layout.q_count)
                        ],
                        axis=0,
                    )
                    for k in range(len(indices))
                ],
                axis=0,
            )
        except ValueError as error:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma per-k frames cannot form fixed numeric tensors",
            ) from error
        heff = np.asarray(authoritative_heff, dtype=np.complex128)
        routed_heff_tensor = np.asarray(routed_heff, dtype=np.complex128)
        reference_projector = np.asarray(
            model_reference_projector, dtype=np.complex128
        )
        frozen_kpoints = _freeze(kpoints, dtype=np.float64, name="kpoints")
        if frozen_kpoints.shape != (len(indices), 2):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma kpoints must have shape (Nk, 2) in routed-k order",
            )
        identity = dict(artifact_identity)
        base_basis_hash = str(identity.get("basis_hash", ""))
        if not base_basis_hash:
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma handoff requires a base projection basis hash",
            )
        identity["k_indices_hash"] = hash_array(np.asarray(indices, dtype=np.int64))
        identity["kpoints_hash"] = hash_array(frozen_kpoints)
        identity["heff_hash"] = hash_array(heff)
        identity["source_hamiltonian_hash"] = str(source_hamiltonian_hash)
        model_frame_hash = hash_array(compact_model_frames)
        routing_to_model_hash = hash_array(routing_to_model)
        routed_heff_hash = hash_array(routed_heff_tensor)
        covariance_residuals = _heff_covariance_residuals(
            routed_heff=routed_heff_tensor,
            model_heff=heff,
            compact_bridge=routing_to_model,
            q_count=layout.q_count,
            group_ranks=first.group_dimensions,
        )
        covariance_tolerance = _strict_positive_real_scalar(
            heff_covariance_tolerance,
            name="Gamma producer routed/model Heff covariance tolerance",
        )
        if (
            covariance_residuals.size != len(indices)
            or not np.all(np.isfinite(covariance_residuals))
            or float(np.max(covariance_residuals)) > covariance_tolerance
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "Gamma producer routed/model Heff covariance evidence failed",
            )
        covariance_residuals_hash = hash_array(covariance_residuals)
        covariance_evidence_hash = _heff_covariance_evidence_hash(
            k_indices_hash=str(identity["k_indices_hash"]),
            kpoints_hash=str(identity["kpoints_hash"]),
            routing_to_model_hash=routing_to_model_hash,
            model_heff_hash=str(identity["heff_hash"]),
            routed_heff_hash=routed_heff_hash,
            residuals_hash=covariance_residuals_hash,
            tolerance=covariance_tolerance,
        )
        reference_projector_hash = hash_array(reference_projector)
        bridge_certificate_hash = _bridge_certificate_hash(
            layout_hash=layout.layout_hash,
            thresholds_hash=thresholds.identity_hash,
            frame_hash=hash_array(frames),
            model_frame_hash=model_frame_hash,
            model_reference_k_index=int(model_reference_k_index),
            model_reference_projector_hash=reference_projector_hash,
            model_anchor_spec_hash=model_anchor_spec.identity_hash,
            routing_to_model_hash=routing_to_model_hash,
            k_indices_hash=str(identity["k_indices_hash"]),
            kpoints_hash=str(identity["kpoints_hash"]),
            model_heff_hash=str(identity["heff_hash"]),
            routed_heff_hash=routed_heff_hash,
            heff_covariance_evidence_hash=covariance_evidence_hash,
            projector_tolerance=thresholds.projector_residual,
        )
        identity["basis_hash"] = _bound_routed_basis_hash(
            base_basis_hash=base_basis_hash,
            layout_hash=layout.layout_hash,
            thresholds_hash=thresholds.identity_hash,
            k_indices_hash=str(identity["k_indices_hash"]),
            kpoints_hash=str(identity["kpoints_hash"]),
            frame_hash=hash_array(frames),
            reference_frame_hash=hash_array(references),
            model_reference_k_index=int(model_reference_k_index),
            model_reference_projector_hash=reference_projector_hash,
            model_anchor_spec_hash=model_anchor_spec.identity_hash,
            model_frame_hash=model_frame_hash,
            routing_to_model_hash=routing_to_model_hash,
            bridge_certificate_hash=bridge_certificate_hash,
            routed_heff_hash=routed_heff_hash,
            heff_covariance_evidence_hash=covariance_evidence_hash,
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
            kpoints=frozen_kpoints,
            kpoints_hash=str(identity["kpoints_hash"]),
            frames=frames,
            reference_frames=references,
            routed_frame_hashes=tuple(item.frame_hash for item in routed),
            routed_reference_frame_hashes=tuple(
                item.reference_frame_hash for item in routed
            ),
            frame_hash=hash_array(frames),
            reference_frame_hash=hash_array(references),
            route_gaps=route_gaps,
            model_reference_k_index=model_reference_k_index,
            model_reference_projector=reference_projector,
            model_reference_projector_hash=reference_projector_hash,
            model_anchor_spec=model_anchor_spec,
            model_frames=compact_model_frames,
            model_frame_hashes=tuple(
                hash_array(row) for row in compact_model_frames
            ),
            model_frame_hash=model_frame_hash,
            routing_to_model=routing_to_model,
            routing_to_model_hashes=tuple(
                hash_array(row) for row in routing_to_model
            ),
            routing_to_model_hash=routing_to_model_hash,
            bridge_certificate_hash=bridge_certificate_hash,
            routed_heff=routed_heff_tensor,
            routed_heff_hash=routed_heff_hash,
            heff_covariance_residuals=covariance_residuals,
            heff_covariance_residuals_hash=covariance_residuals_hash,
            heff_covariance_tolerance=covariance_tolerance,
            heff_covariance_evidence_hash=covariance_evidence_hash,
            heff_k_indices=heff_indices,
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
                "kpoints_hash": self.kpoints_hash,
                "frame_hash": self.frame_hash,
                "reference_frame_hash": self.reference_frame_hash,
                "routed_frame_hashes": self.routed_frame_hashes,
                "routed_reference_frame_hashes": self.routed_reference_frame_hashes,
                "route_gaps_hash": hash_array(self.route_gaps),
                "model_reference_k_index": self.model_reference_k_index,
                "model_reference_projector_hash": self.model_reference_projector_hash,
                "model_anchor_spec_hash": self.model_anchor_spec.identity_hash,
                "model_frame_hash": self.model_frame_hash,
                "model_frame_hashes": self.model_frame_hashes,
                "routing_to_model_hash": self.routing_to_model_hash,
                "routing_to_model_hashes": self.routing_to_model_hashes,
                "bridge_certificate_hash": self.bridge_certificate_hash,
                "routed_heff_hash": self.routed_heff_hash,
                "heff_covariance_residuals_hash": self.heff_covariance_residuals_hash,
                "heff_covariance_tolerance": self.heff_covariance_tolerance,
                "heff_covariance_evidence_hash": self.heff_covariance_evidence_hash,
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

    def assemble_routing_for_k(
        self, k_index: int, *, include_high: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return assemble_gamma_routed_projectors(
            self.routed_frames_for_k(k_index),
            layout=self.layout,
            thresholds=self.thresholds,
            include_high=include_high,
        )

    def assemble_model_for_k(self, k_index: int) -> np.ndarray:
        return _assemble_model_tensor(
            self.model_frames[self._k_position(k_index)],
            layout=self.layout,
            group_ranks=self.group_ranks,
        )

    def routing_to_model_for_k(self, k_index: int) -> np.ndarray:
        return _assemble_bridge_tensor(
            self.routing_to_model[self._k_position(k_index)],
            q_count=self.layout.q_count,
            group_ranks=self.group_ranks,
        )

    def model_state_for_k(self, k_index: int) -> tuple[np.ndarray, np.ndarray]:
        position = self._k_position(k_index)
        return (
            _assemble_model_tensor(
                self.model_frames[position],
                layout=self.layout,
                group_ranks=self.group_ranks,
            ),
            self.authoritative_heff[position],
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
    kpoints: Any,
    routed_frames: Sequence[GammaRoutedFrames],
    model_reference_k_index: int,
    model_anchor_spec: GammaModelAnchorSpec,
    model_frames: Sequence[GammaModelFrames],
    model_reference_projector: Any,
    routed_heff: Any,
    heff_covariance_tolerance: float,
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
    certified_gamma_actions: Sequence[GammaCertifiedRawAction] | None = None,
) -> GammaRoutedBasisSpec:
    """Create a routed handoff only after Task-7 candidate certification.

    The certificate is evaluated from the exact assembled persisted frames and
    the same authoritative Heff tensor that will be written to ``basis.npz``.
    """

    from .symmetry.candidate_certificate import (
        CandidateOperationInput,
        CandidateProjectionState,
        CandidateSymmetryStatus,
        candidate_action_package_hash,
        candidate_certificate_envelope,
        certify_candidate_symmetries,
    )

    if certified_gamma_actions is None:
        if any(
            isinstance(operation, CandidateOperationInput)
            and operation.route_contract is not None
            for operation in operations.values()
        ):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma route contracts require certified Gamma actions",
            )
    else:
        certified_by_name: dict[str, GammaCertifiedRawAction] = {}
        for action in certified_gamma_actions:
            if action.name in certified_by_name:
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    f"duplicate certified Gamma action {action.name!r}",
                )
            certified_by_name[action.name] = action
        if set(certified_by_name) != set(operations):
            raise _reject(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "certified Gamma actions do not cover the candidate operations",
            )
        for name, action in certified_by_name.items():
            operation = operations[name]
            if not isinstance(operation, CandidateOperationInput):
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    f"operation {name!r} is not a candidate operation input",
                )
            if operation.name != action.name or operation.antiunitary != action.antiunitary:
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    f"operation {name!r} metadata differs from its certified Gamma action",
                )
            expected_contract = gamma_certified_action_route_contract(
                action,
                layout=layout,
                thresholds=thresholds,
                full_action=operation.d_full,
            )
            if "sampled_k_route" in operation.route_contract:
                expected_contract["sampled_k_route"] = gamma_sampled_k_route_contract(
                    kpoints,
                    required_pairs[name],
                )
            if operation.route_contract != expected_contract:
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    f"operation {name!r} route contract differs from its certified Gamma action",
                )

    configured_covariance_tolerance = _strict_positive_real_scalar(
        heff_covariance_tolerance,
        name="Gamma routed/model Heff covariance tolerance",
    )

    provisional = GammaRoutedBasisSpec.create(
        artifact_identity=artifact_identity,
        layout=layout,
        thresholds=thresholds,
        k_indices=k_indices,
        kpoints=kpoints,
        routed_frames=routed_frames,
        model_reference_k_index=model_reference_k_index,
        model_anchor_spec=model_anchor_spec,
        model_frames=model_frames,
        model_reference_projector=model_reference_projector,
        routed_heff=routed_heff,
        heff_covariance_tolerance=configured_covariance_tolerance,
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
    states: dict[int, CandidateProjectionState] = {}
    for k_index in provisional.k_indices:
        model_frame, model_heff = provisional.model_state_for_k(k_index)
        states[k_index] = CandidateProjectionState(
            u_low=model_frame,
            heff=model_heff,
        )
    certificate = certify_candidate_symmetries(
        candidate_id=str(candidate_id),
        states=states,
        operations=operations,
        exactified_actions=exactified_actions,
        presentation=presentation,
        required_pairs=required_pairs,
        thresholds=candidate_thresholds,
        required_state_k_indices=k_indices,
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
        kpoints_hash=provisional.kpoints_hash,
        frame_hash=provisional.frame_hash,
        reference_frame_hash=provisional.reference_frame_hash,
        model_reference_k_index=provisional.model_reference_k_index,
        model_reference_projector_hash=provisional.model_reference_projector_hash,
        model_anchor_spec_hash=provisional.model_anchor_spec.identity_hash,
        model_frame_hash=provisional.model_frame_hash,
        routing_to_model_hash=provisional.routing_to_model_hash,
        bridge_certificate_hash=provisional.bridge_certificate_hash,
        routed_heff_hash=provisional.routed_heff_hash,
        heff_covariance_evidence_hash=provisional.heff_covariance_evidence_hash,
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
        "kpoints": spec.kpoints,
        "kpoints_hash": np.asarray(spec.kpoints_hash),
        "frames": spec.frames,
        "reference_frames": spec.reference_frames,
        "routed_frame_hashes": np.asarray(spec.routed_frame_hashes),
        "routed_reference_frame_hashes": np.asarray(
            spec.routed_reference_frame_hashes
        ),
        "frame_hash": np.asarray(spec.frame_hash),
        "reference_frame_hash": np.asarray(spec.reference_frame_hash),
        "route_gaps": spec.route_gaps,
        "model_reference_k_index": np.asarray(
            spec.model_reference_k_index, dtype=np.int64
        ),
        "model_reference_projector": spec.model_reference_projector,
        "model_reference_projector_hash": np.asarray(
            spec.model_reference_projector_hash
        ),
        "model_anchor_spec_payload": np.asarray(
            _canonical_json(spec.model_anchor_spec.to_payload())
        ),
        "model_anchor_spec_hash": np.asarray(spec.model_anchor_spec.identity_hash),
        "model_frames": spec.model_frames,
        "model_frame_hashes": np.asarray(spec.model_frame_hashes),
        "model_frame_hash": np.asarray(spec.model_frame_hash),
        "routing_to_model": spec.routing_to_model,
        "routing_to_model_hashes": np.asarray(spec.routing_to_model_hashes),
        "routing_to_model_hash": np.asarray(spec.routing_to_model_hash),
        "bridge_certificate_hash": np.asarray(spec.bridge_certificate_hash),
        "routed_heff": spec.routed_heff,
        "routed_heff_hash": np.asarray(spec.routed_heff_hash),
        "heff_covariance_residuals": spec.heff_covariance_residuals,
        "heff_covariance_residuals_hash": np.asarray(
            spec.heff_covariance_residuals_hash
        ),
        "heff_covariance_tolerance": np.asarray(
            spec.heff_covariance_tolerance, dtype=np.float64
        ),
        "heff_covariance_evidence_hash": np.asarray(
            spec.heff_covariance_evidence_hash
        ),
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
            discriminator = _required_text(
                {"projection_basis_kind": np.array(archive["projection_basis_kind"], copy=True)},
                "projection_basis_kind",
            )
            version = _required_text(
                {
                    "projection_basis_handoff_version": np.array(
                        archive["projection_basis_handoff_version"], copy=True
                    )
                },
                "projection_basis_handoff_version",
            )
            if discriminator != GAMMA_ROUTED_BASIS_KIND:
                raise _reject(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    "projection basis discriminator is not gamma_routed",
                )
            if version != GAMMA_ROUTED_BASIS_HANDOFF_VERSION:
                detail = (
                    "legacy v2 routed Gamma archives are unsupported; rerun kp project"
                    if version == "kp_project_gamma_routed_handoff_v2"
                    else f"unsupported routed Gamma handoff version {version!r}; rerun kp project"
                )
                raise _reject(CandidateRejectionReason.HANDOFF_IDENTITY, detail)
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
    layout = _restore_layout(
        _required_text(payload, "layout_payload"),
        _required_text(payload, "layout_hash"),
    )
    thresholds = _restore_thresholds(
        _required_text(payload, "routing_thresholds_payload"),
        _required_text(payload, "routing_thresholds_hash"),
    )
    try:
        anchor_payload_text = _required_text(payload, "model_anchor_spec_payload")
        anchor_payload = json.loads(anchor_payload_text)
        if not isinstance(anchor_payload, Mapping):
            raise TypeError("anchor payload is not a mapping")
        model_anchor_spec = GammaModelAnchorSpec.from_payload(anchor_payload)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"Gamma model anchor payload cannot be reconstructed: {error}",
        ) from error
    if (
        _canonical_json(model_anchor_spec.to_payload()) != anchor_payload_text
        or model_anchor_spec.identity_hash
        != _required_text(payload, "model_anchor_spec_hash")
    ):
        raise _reject(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "Gamma model anchor payload/hash is not canonical",
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
    artifact_identity["kpoints_hash"] = _required_text(payload, "kpoints_hash")
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
        kpoints=payload["kpoints"],
        kpoints_hash=_required_text(payload, "kpoints_hash"),
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
        model_reference_k_index=int(
            np.asarray(payload["model_reference_k_index"]).item()
        ),
        model_reference_projector=payload["model_reference_projector"],
        model_reference_projector_hash=_required_text(
            payload, "model_reference_projector_hash"
        ),
        model_anchor_spec=model_anchor_spec,
        model_frames=payload["model_frames"],
        model_frame_hashes=_strict_hash_tuple(
            payload["model_frame_hashes"],
            name="model_frame_hashes",
            length=len(k_indices),
        ),
        model_frame_hash=_required_text(payload, "model_frame_hash"),
        routing_to_model=payload["routing_to_model"],
        routing_to_model_hashes=_strict_hash_tuple(
            payload["routing_to_model_hashes"],
            name="routing_to_model_hashes",
            length=len(k_indices),
        ),
        routing_to_model_hash=_required_text(payload, "routing_to_model_hash"),
        bridge_certificate_hash=_required_text(
            payload, "bridge_certificate_hash"
        ),
        routed_heff=payload["routed_heff"],
        routed_heff_hash=_required_text(payload, "routed_heff_hash"),
        heff_covariance_residuals=payload["heff_covariance_residuals"],
        heff_covariance_residuals_hash=_required_text(
            payload, "heff_covariance_residuals_hash"
        ),
        heff_covariance_tolerance=float(
            np.asarray(payload["heff_covariance_tolerance"]).item()
        ),
        heff_covariance_evidence_hash=_required_text(
            payload, "heff_covariance_evidence_hash"
        ),
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
