"""Project-owned, bit-exact targets for KP symmetry exactification.

The authoritative payload is the literal little-endian ``complex128`` route
block.  A cyclotomic-monomial description is stored only when it reconstructs
the same bits; it is an algebraic audit hint and never owns production data.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..identity import hash_array, hash_mapping
from .joint_exactification import (
    BlockRouteAction,
    JointExactificationConfig,
    MagneticPresentation,
    _cyclotomic_root,
    certify_joint_block_actions,
    magnetic_presentation_artifact,
    magnetic_presentation_from_artifact,
    materialize_block_route_action,
)


CANONICAL_TARGET_ACTIONS_V1 = "canonical_target_actions_v1"
LITERAL_ROUTE_SOURCE_ACTIONS_V1 = "literal_route_source_actions_v1"
_LITERAL_STORAGE = "literal_little_endian_complex128_block_routes_v1"
_POST_STORAGE_CLEANUP = "post_exactification_storage_cleanup_v1"


class CanonicalTargetError(ValueError):
    """A persisted project target is missing, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class CanonicalTargetBundle:
    """Fully decoded and hash-checked project target."""

    actions: Mapping[str, BlockRouteAction]
    presentation: MagneticPresentation
    joint_config: JointExactificationConfig
    certificate_identity: Mapping[str, Any]
    artifact_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(
            self,
            "certificate_identity",
            MappingProxyType(dict(self.certificate_identity)),
        )


@dataclass(frozen=True)
class LiteralRouteSourceBundle:
    """Bit-exact relation-certified intermediate source action."""

    actions: Mapping[str, BlockRouteAction]
    presentation: MagneticPresentation
    joint_config: JointExactificationConfig
    relation_certificate: Mapping[str, Any]
    structure_identity: Mapping[str, Any]
    artifact_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(
            self,
            "relation_certificate",
            MappingProxyType(dict(self.relation_certificate)),
        )
        object.__setattr__(
            self,
            "structure_identity",
            MappingProxyType(dict(self.structure_identity)),
        )


def _config_from_mapping(value: Mapping[str, Any] | None) -> JointExactificationConfig:
    default = JointExactificationConfig()
    raw = {} if value is None else value
    if not isinstance(raw, Mapping):
        raise CanonicalTargetError("canonical target joint config must be a mapping")
    return JointExactificationConfig(
        enabled=bool(raw.get("enabled", default.enabled)),
        max_rms_correction=float(
            raw.get("max_rms_correction", default.max_rms_correction)
        ),
        max_route_correction=float(
            raw.get("max_route_correction", default.max_route_correction)
        ),
        central_branch_margin=float(
            raw.get("central_branch_margin", default.central_branch_margin)
        ),
        max_iterations=int(raw.get("max_iterations", default.max_iterations)),
        condition_limit=float(raw.get("condition_limit", default.condition_limit)),
    )


def _config_artifact(
    value: JointExactificationConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    config = value if isinstance(value, JointExactificationConfig) else _config_from_mapping(value)
    return {
        "enabled": bool(config.enabled),
        "max_rms_correction": float(config.max_rms_correction),
        "max_route_correction": float(config.max_route_correction),
        "central_branch_margin": float(config.central_branch_margin),
        "max_iterations": int(config.max_iterations),
        "condition_limit": float(config.condition_limit),
    }


def _little_endian_complex_bytes(block: np.ndarray) -> tuple[np.ndarray, bytes]:
    value = np.asarray(block, dtype=np.complex128, order="C")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise CanonicalTargetError("canonical route block must be square")
    if not np.all(np.isfinite(value)):
        raise CanonicalTargetError("canonical route block must be finite")
    little = np.asarray(value, dtype=np.dtype("<c16"), order="C")
    return value, little.tobytes(order="C")


def _literal_payload_hash(shape: tuple[int, int], payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"moirekp:canonical-target:block:c16le:v1\0")
    digest.update(np.asarray(shape, dtype="<i8").tobytes(order="C"))
    digest.update(payload)
    return digest.hexdigest()


def _strict_cyclotomic_monomial(
    block: np.ndarray,
    *,
    root_order: int,
) -> dict[str, Any] | None:
    value, authoritative = _little_endian_complex_bytes(block)
    dimension = int(value.shape[0])
    support: list[int] = []
    exponents: list[int] = []
    for source in range(dimension):
        rows = np.flatnonzero(value[:, source] != 0.0)
        if rows.size != 1:
            return None
        target = int(rows[0])
        phase = complex(value[target, source])
        matches = tuple(
            exponent
            for exponent in range(root_order)
            if phase == _cyclotomic_root(root_order, exponent)
        )
        if not matches:
            return None
        support.append(target)
        exponents.append(int(matches[0]))
    if sorted(support) != list(range(dimension)):
        return None
    reconstructed = np.zeros_like(value)
    for source, (target, exponent) in enumerate(zip(support, exponents)):
        reconstructed[target, source] = _cyclotomic_root(root_order, exponent)
    _, reconstructed_bytes = _little_endian_complex_bytes(reconstructed)
    if reconstructed_bytes != authoritative:
        return None
    return {
        "status": "certified",
        "codec": "cyclotomic_monomial_hint_v1",
        "root_order": int(root_order),
        "support_target_by_source": support,
        "root_exponent_by_source": exponents,
        "bitwise_equal_to_authoritative_payload": True,
    }


def _block_record(block: np.ndarray, *, root_order: int) -> dict[str, Any]:
    value, payload = _little_endian_complex_bytes(block)
    algebraic = _strict_cyclotomic_monomial(value, root_order=root_order)
    if algebraic is None:
        algebraic = {
            "status": "unavailable",
            "codec": "cyclotomic_monomial_hint_v1",
            "reason": "route_block_is_not_bitwise_cyclotomic_monomial",
        }
    return {
        "shape": [int(value.shape[0]), int(value.shape[1])],
        "dtype": "<c16",
        "byte_order": "little",
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "payload_hash": _literal_payload_hash(value.shape, payload),
        "algebraic_codec": algebraic,
    }


def _decode_block(record: Mapping[str, Any]) -> np.ndarray:
    if record.get("dtype") != "<c16" or record.get("byte_order") != "little":
        raise CanonicalTargetError("canonical target block has unsupported dtype")
    shape_raw = record.get("shape")
    if not isinstance(shape_raw, list) or len(shape_raw) != 2:
        raise CanonicalTargetError("canonical target block shape is invalid")
    shape = (int(shape_raw[0]), int(shape_raw[1]))
    if shape[0] <= 0 or shape[0] != shape[1]:
        raise CanonicalTargetError("canonical target block must be square")
    encoded = record.get("payload_base64")
    if not isinstance(encoded, str):
        raise CanonicalTargetError("canonical target block lacks its literal payload")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as exc:
        raise CanonicalTargetError("canonical target block payload is invalid") from exc
    expected_size = shape[0] * shape[1] * np.dtype("<c16").itemsize
    if len(payload) != expected_size:
        raise CanonicalTargetError("canonical target block payload size is invalid")
    if _literal_payload_hash(shape, payload) != str(record.get("payload_hash", "")):
        raise CanonicalTargetError("canonical target block payload hash mismatch")
    little = np.frombuffer(payload, dtype=np.dtype("<c16")).reshape(shape)
    block = np.array(little, dtype=np.complex128, order="C", copy=True)
    algebraic = record.get("algebraic_codec")
    if isinstance(algebraic, Mapping) and algebraic.get("status") == "certified":
        order = int(algebraic.get("root_order", 0))
        support = tuple(
            int(item) for item in algebraic.get("support_target_by_source", ())
        )
        exponents = tuple(
            int(item) for item in algebraic.get("root_exponent_by_source", ())
        )
        if (
            order <= 0
            or len(support) != shape[0]
            or len(exponents) != shape[0]
            or sorted(support) != list(range(shape[0]))
        ):
            raise CanonicalTargetError("canonical algebraic hint is invalid")
        reconstructed = np.zeros(shape, dtype=np.complex128)
        for source, (target, exponent) in enumerate(zip(support, exponents)):
            reconstructed[target, source] = _cyclotomic_root(order, exponent)
        _, reconstructed_payload = _little_endian_complex_bytes(reconstructed)
        if reconstructed_payload != payload:
            raise CanonicalTargetError(
                "canonical algebraic hint differs from authoritative payload"
            )
    return block


def _certificate_identity(certificate: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "status",
        "basis_frame_hash",
        "layout_hash",
        "presentation_hash",
        "declared_cocycle_hash",
        "actions_hash",
    )
    if certificate.get("status") != "certified" or any(
        field not in certificate for field in required
    ):
        raise CanonicalTargetError(
            "canonical target requires a complete certified structure identity"
        )
    return {
        "version": str(certificate.get("version", "unspecified")),
        **{field: certificate[field] for field in required},
        "certificate_hash": str(
            certificate.get("certificate_hash", hash_mapping(certificate))
        ),
    }


def _encode_action_records(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    root_order: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    names = tuple(generator.name for generator in presentation.generators)
    if set(actions) != set(names):
        raise CanonicalTargetError(
            "literal route actions must match the declared presentation"
        )
    records: list[dict[str, Any]] = []
    dense_hashes: dict[str, str] = {}
    for generator in presentation.generators:
        action = actions[generator.name]
        if bool(action.antiunitary) != bool(generator.antiunitary):
            raise CanonicalTargetError(
                f"antiunitary parity mismatch for {generator.name!r}"
            )
        dense_hashes[generator.name] = hash_array(
            materialize_block_route_action(action)
        )
        records.append(
            {
                "name": generator.name,
                "antiunitary": bool(action.antiunitary),
                "fiber_permutation": list(action.fiber_permutation),
                "fiber_dimensions": list(action.fiber_dimensions),
                "fiber_indices": [list(indices) for indices in action.fiber_indices],
                "unitarity_certification_bound": float(
                    action.unitarity_certification_bound
                ),
                "route_blocks": [
                    _block_record(block, root_order=root_order)
                    for block in action.route_blocks
                ],
            }
        )
    return records, dense_hashes


def encode_canonical_target_actions(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    certificate: Mapping[str, Any],
    joint_config: JointExactificationConfig | Mapping[str, Any] | None = None,
    root_order: int = 24,
) -> dict[str, Any]:
    """Encode a general project-owned ``U(d)`` route action bit for bit."""

    order = int(root_order)
    if order <= 0:
        raise CanonicalTargetError("canonical target root order must be positive")
    records, dense_hashes = _encode_action_records(
        actions,
        presentation,
        root_order=order,
    )
    payload: dict[str, Any] = {
        "version": CANONICAL_TARGET_ACTIONS_V1,
        "status": "certified",
        "storage": _LITERAL_STORAGE,
        "root_order": order,
        # Compatibility field: its scope is explicitly only the storage pass.
        "numeric_entries_modified": 0,
        "numeric_entries_modified_scope": _POST_STORAGE_CLEANUP,
        "post_exactification_storage_cleanup": {
            "version": _POST_STORAGE_CLEANUP,
            "numeric_entries_modified": 0,
            "threshold_cleanup_performed": False,
        },
        "certificate_identity": _certificate_identity(certificate),
        "presentation": magnetic_presentation_artifact(presentation),
        "joint_config": _config_artifact(joint_config),
        "dense_action_hashes": dense_hashes,
        "actions": records,
    }
    payload["artifact_hash"] = hash_mapping(payload)
    return payload


def encode_literal_route_source_actions(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    structure_identity: Mapping[str, Any],
    joint_config: JointExactificationConfig | Mapping[str, Any] | None = None,
    root_order: int = 24,
    source_role: str = "phase_preserving_numerical_source",
) -> dict[str, Any]:
    """Encode a relation-certified numerical source without calling it a target."""

    order = int(root_order)
    if order <= 0:
        raise CanonicalTargetError("literal source root order must be positive")
    role = str(source_role).strip()
    if not role:
        raise CanonicalTargetError("literal source role must be nonempty")
    records, dense_hashes = _encode_action_records(
        actions,
        presentation,
        root_order=order,
    )
    try:
        relation_certificate = certify_joint_block_actions(actions, presentation)
    except Exception as exc:
        raise CanonicalTargetError(
            "literal source actions are not relation certified"
        ) from exc
    identity = dict(structure_identity)
    if not identity.get("status"):
        raise CanonicalTargetError(
            "literal source structure identity must report its classifier status"
        )
    payload: dict[str, Any] = {
        "version": LITERAL_ROUTE_SOURCE_ACTIONS_V1,
        "status": "relation_certified",
        "role": role,
        "storage": _LITERAL_STORAGE,
        "root_order": order,
        "threshold_cleanup_performed": False,
        "relation_certificate": relation_certificate,
        "structure_identity": identity,
        "presentation": magnetic_presentation_artifact(presentation),
        "joint_config": _config_artifact(joint_config),
        "dense_action_hashes": dense_hashes,
        "actions": records,
    }
    payload["artifact_hash"] = hash_mapping(payload)
    return payload


def _decode_literal_action_records(
    root: Mapping[str, Any],
    presentation: MagneticPresentation,
) -> dict[str, BlockRouteAction]:
    records_raw = root.get("actions")
    if not isinstance(records_raw, list):
        raise CanonicalTargetError("literal source action records must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    for record in records_raw:
        if not isinstance(record, Mapping) or not str(record.get("name", "")):
            raise CanonicalTargetError("literal source action record is invalid")
        name = str(record["name"])
        if name in records:
            raise CanonicalTargetError(f"duplicate literal source action {name!r}")
        records[name] = record
    names = tuple(generator.name for generator in presentation.generators)
    if set(records) != set(names):
        raise CanonicalTargetError(
            "literal source action names do not match its presentation"
        )
    dense_hashes = root.get("dense_action_hashes")
    if not isinstance(dense_hashes, Mapping):
        raise CanonicalTargetError("literal source lacks dense action hashes")
    actions: dict[str, BlockRouteAction] = {}
    for generator in presentation.generators:
        record = records[generator.name]
        parity = bool(record.get("antiunitary"))
        if parity != bool(generator.antiunitary):
            raise CanonicalTargetError(
                f"literal source antiunitary mismatch for {generator.name!r}"
            )
        permutation = tuple(int(value) for value in record.get("fiber_permutation", ()))
        dimensions = tuple(int(value) for value in record.get("fiber_dimensions", ()))
        indices = tuple(
            tuple(int(value) for value in members)
            for members in record.get("fiber_indices", ())
        )
        block_records = record.get("route_blocks")
        if not isinstance(block_records, list) or len(block_records) != len(dimensions):
            raise CanonicalTargetError(
                f"literal source has invalid route blocks for {generator.name!r}"
            )
        blocks = tuple(_decode_block(block_record) for block_record in block_records)
        if any(
            block.shape != (dimension, dimension)
            for block, dimension in zip(blocks, dimensions)
        ):
            raise CanonicalTargetError(
                f"literal source route dimensions mismatch for {generator.name!r}"
            )
        try:
            action = BlockRouteAction(
                generator.name,
                parity,
                permutation,
                dimensions,
                blocks,
                fiber_indices=indices,
                unitarity_certification_bound=float(
                    record["unitarity_certification_bound"]
                ),
            )
        except Exception as exc:
            raise CanonicalTargetError(
                f"literal source route layout is invalid for {generator.name!r}"
            ) from exc
        if hash_array(materialize_block_route_action(action)) != str(
            dense_hashes.get(generator.name, "")
        ):
            raise CanonicalTargetError(
                f"literal source dense hash mismatch for {generator.name!r}"
            )
        actions[generator.name] = action
    return actions


def load_literal_route_source_artifact(
    artifact: Mapping[str, Any],
) -> LiteralRouteSourceBundle:
    """Hash-check and restore a noncanonical relation-certified source snapshot."""

    root = dict(artifact)
    expected_artifact_hash = str(root.pop("artifact_hash", ""))
    if not expected_artifact_hash or expected_artifact_hash != hash_mapping(root):
        raise CanonicalTargetError("literal source artifact hash mismatch")
    if (
        root.get("version") != LITERAL_ROUTE_SOURCE_ACTIONS_V1
        or root.get("status") != "relation_certified"
        or root.get("storage") != _LITERAL_STORAGE
        or bool(root.get("threshold_cleanup_performed"))
        or not str(root.get("role", "")).strip()
    ):
        raise CanonicalTargetError("unsupported or uncertified literal source")
    presentation_raw = root.get("presentation")
    if not isinstance(presentation_raw, Mapping):
        raise CanonicalTargetError("literal source lacks its presentation")
    try:
        presentation = magnetic_presentation_from_artifact(presentation_raw)
    except Exception as exc:
        raise CanonicalTargetError("literal source presentation is invalid") from exc
    config_raw = root.get("joint_config")
    config = _config_from_mapping(
        config_raw if isinstance(config_raw, Mapping) else None
    )
    relation_certificate = root.get("relation_certificate")
    structure_identity = root.get("structure_identity")
    if (
        not isinstance(relation_certificate, Mapping)
        or relation_certificate.get("status") != "certified"
    ):
        raise CanonicalTargetError("literal source lacks relation certification")
    if not isinstance(structure_identity, Mapping) or not structure_identity.get(
        "status"
    ):
        raise CanonicalTargetError("literal source lacks structure classifier identity")
    actions = _decode_literal_action_records(root, presentation)
    live_certificate = certify_joint_block_actions(actions, presentation)
    if hash_mapping(live_certificate) != hash_mapping(relation_certificate):
        raise CanonicalTargetError("literal source relation certificate mismatch")
    return LiteralRouteSourceBundle(
        actions=actions,
        presentation=presentation,
        joint_config=config,
        relation_certificate=relation_certificate,
        structure_identity=structure_identity,
        artifact_hash=expected_artifact_hash,
    )


def load_canonical_target_artifact(
    artifact: Mapping[str, Any],
) -> CanonicalTargetBundle:
    """Hash-check and reconstruct a persisted target without a reference solver."""

    root = dict(artifact)
    expected_artifact_hash = str(root.pop("artifact_hash", ""))
    actual_artifact_hash = hash_mapping(root)
    if not expected_artifact_hash or expected_artifact_hash != actual_artifact_hash:
        raise CanonicalTargetError("canonical target artifact hash mismatch")
    cleanup = root.get("post_exactification_storage_cleanup")
    if (
        root.get("version") != CANONICAL_TARGET_ACTIONS_V1
        or root.get("status") != "certified"
        or root.get("storage") != _LITERAL_STORAGE
        or root.get("numeric_entries_modified") != 0
        or root.get("numeric_entries_modified_scope") != _POST_STORAGE_CLEANUP
        or not isinstance(cleanup, Mapping)
        or cleanup.get("numeric_entries_modified") != 0
        or bool(cleanup.get("threshold_cleanup_performed"))
    ):
        raise CanonicalTargetError("unsupported or uncertified canonical target")
    presentation_raw = root.get("presentation")
    if not isinstance(presentation_raw, Mapping):
        raise CanonicalTargetError("canonical target lacks its presentation")
    try:
        presentation = magnetic_presentation_from_artifact(presentation_raw)
    except Exception as exc:
        raise CanonicalTargetError("canonical target presentation is invalid") from exc
    config_raw = root.get("joint_config")
    config = _config_from_mapping(
        config_raw if isinstance(config_raw, Mapping) else None
    )
    identity = root.get("certificate_identity")
    if not isinstance(identity, Mapping):
        raise CanonicalTargetError("canonical target lacks certificate identity")
    required_identity = (
        "version",
        "status",
        "basis_frame_hash",
        "layout_hash",
        "presentation_hash",
        "declared_cocycle_hash",
        "actions_hash",
        "certificate_hash",
    )
    if identity.get("status") != "certified" or any(
        field not in identity for field in required_identity
    ):
        raise CanonicalTargetError("canonical target certificate identity is incomplete")
    records_raw = root.get("actions")
    if not isinstance(records_raw, list):
        raise CanonicalTargetError("canonical target action records must be a list")
    records: dict[str, Mapping[str, Any]] = {}
    for record in records_raw:
        if not isinstance(record, Mapping) or not str(record.get("name", "")):
            raise CanonicalTargetError("canonical target action record is invalid")
        name = str(record["name"])
        if name in records:
            raise CanonicalTargetError(f"duplicate canonical target action {name!r}")
        records[name] = record
    names = tuple(generator.name for generator in presentation.generators)
    if set(records) != set(names):
        raise CanonicalTargetError(
            "canonical target action names do not match its presentation"
        )
    dense_hashes = root.get("dense_action_hashes")
    if not isinstance(dense_hashes, Mapping):
        raise CanonicalTargetError("canonical target lacks dense action hashes")
    actions: dict[str, BlockRouteAction] = {}
    for generator in presentation.generators:
        record = records[generator.name]
        parity = bool(record.get("antiunitary"))
        if parity != bool(generator.antiunitary):
            raise CanonicalTargetError(
                f"canonical target antiunitary mismatch for {generator.name!r}"
            )
        permutation = tuple(int(value) for value in record.get("fiber_permutation", ()))
        dimensions = tuple(int(value) for value in record.get("fiber_dimensions", ()))
        indices = tuple(
            tuple(int(value) for value in members)
            for members in record.get("fiber_indices", ())
        )
        block_records = record.get("route_blocks")
        if not isinstance(block_records, list) or len(block_records) != len(dimensions):
            raise CanonicalTargetError(
                f"canonical target has invalid route blocks for {generator.name!r}"
            )
        blocks = tuple(_decode_block(block_record) for block_record in block_records)
        if any(block.shape != (dimension, dimension) for block, dimension in zip(blocks, dimensions)):
            raise CanonicalTargetError(
                f"canonical target route dimensions mismatch for {generator.name!r}"
            )
        try:
            action = BlockRouteAction(
                generator.name,
                parity,
                permutation,
                dimensions,
                blocks,
                fiber_indices=indices,
                unitarity_certification_bound=float(
                    record["unitarity_certification_bound"]
                ),
            )
        except Exception as exc:
            raise CanonicalTargetError(
                f"canonical target route layout is invalid for {generator.name!r}"
            ) from exc
        if hash_array(materialize_block_route_action(action)) != str(
            dense_hashes.get(generator.name, "")
        ):
            raise CanonicalTargetError(
                f"canonical target dense hash mismatch for {generator.name!r}"
            )
        actions[generator.name] = action
    return CanonicalTargetBundle(
        actions=actions,
        presentation=presentation,
        joint_config=config,
        certificate_identity=identity,
        artifact_hash=expected_artifact_hash,
    )


def decode_canonical_target_actions(
    artifact: Mapping[str, Any],
    reference_actions: Mapping[str, BlockRouteAction] | None = None,
    presentation: MagneticPresentation | None = None,
    *,
    expected_certificate_identity: Mapping[str, Any] | None = None,
) -> dict[str, BlockRouteAction]:
    """Decode a target and optionally bind it to a current route context."""

    bundle = load_canonical_target_artifact(artifact)
    if presentation is not None and magnetic_presentation_artifact(
        presentation
    ) != magnetic_presentation_artifact(bundle.presentation):
        raise CanonicalTargetError("canonical target presentation mismatch")
    if expected_certificate_identity is not None:
        fields = (
            "version",
            "status",
            "basis_frame_hash",
            "layout_hash",
            "presentation_hash",
            "declared_cocycle_hash",
            "actions_hash",
            "certificate_hash",
        )
        mismatch = tuple(
            field
            for field in fields
            if bundle.certificate_identity.get(field)
            != expected_certificate_identity.get(field)
        )
        if mismatch:
            raise CanonicalTargetError(
                "canonical target certificate identity mismatch: "
                + ", ".join(mismatch)
            )
    if reference_actions is not None:
        if set(reference_actions) != set(bundle.actions):
            raise CanonicalTargetError(
                "canonical target actions do not match current route actions"
            )
        for name, action in bundle.actions.items():
            reference = reference_actions[name]
            if (
                action.antiunitary != reference.antiunitary
                or action.fiber_permutation != reference.fiber_permutation
                or action.fiber_dimensions != reference.fiber_dimensions
                or action.fiber_indices != reference.fiber_indices
            ):
                raise CanonicalTargetError(
                    f"canonical target route layout mismatch for {name!r}"
                )
    return dict(bundle.actions)


def _monomial_branch_margin(
    raw_block: np.ndarray,
    target_block: np.ndarray,
    *,
    root_order: int,
) -> float | None:
    codec = _strict_cyclotomic_monomial(target_block, root_order=root_order)
    if codec is None:
        return None
    roots = tuple(_cyclotomic_root(root_order, exponent) for exponent in range(root_order))
    support = tuple(int(value) for value in codec["support_target_by_source"])
    exponents = tuple(int(value) for value in codec["root_exponent_by_source"])
    minimum = float("inf")
    for source, (target_row, target_exponent) in enumerate(zip(support, exponents)):
        column = np.asarray(raw_block[:, source], dtype=np.complex128)
        candidates: list[tuple[float, int, int]] = []
        for row in range(target_block.shape[0]):
            for exponent, root in enumerate(roots):
                candidate = np.zeros(target_block.shape[0], dtype=np.complex128)
                candidate[row] = root
                candidates.append(
                    (float(np.linalg.norm(column - candidate)), row, exponent)
                )
        target_distance = next(
            distance
            for distance, row, exponent in candidates
            if row == target_row and exponent == target_exponent
        )
        alternative_distance = min(
            distance
            for distance, row, exponent in candidates
            if (row, exponent) != (target_row, target_exponent)
        )
        margin = float(alternative_distance - target_distance)
        minimum = min(minimum, margin)
    return minimum


def certify_fixed_target_exactification(
    raw_matrices: Mapping[str, np.ndarray],
    target_actions: Mapping[str, BlockRouteAction],
    *,
    root_order: int = 24,
    max_rms_correction: float,
    max_route_correction: float,
    persisted_frame_hash: str | None = None,
    target_artifact_hash: str | None = None,
    target_certificate_identity: Mapping[str, Any] | None = None,
    source_matrix_hashes: Mapping[str, str] | None = None,
    operation_antiunitary: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Certify transformed raw matrices directly against the project target."""

    if set(raw_matrices) != set(target_actions):
        raise CanonicalTargetError(
            "fixed-target matrices and actions must have identical names"
        )
    order = int(root_order)
    if order <= 0:
        raise CanonicalTargetError("fixed-target root order must be positive")
    rms_limit = float(max_rms_correction)
    route_limit = float(max_route_correction)
    if not np.isfinite(rms_limit) or rms_limit <= 0.0:
        raise CanonicalTargetError("fixed-target RMS limit must be finite and positive")
    if not np.isfinite(route_limit) or route_limit <= 0.0:
        raise CanonicalTargetError("fixed-target route limit must be finite and positive")

    route_values_all: list[float] = []
    route_rms_by_operation: dict[str, float] = {}
    route_max_by_operation: dict[str, float] = {}
    dense_relative_by_operation: dict[str, float] = {}
    dense_normalized_by_operation: dict[str, float] = {}
    off_route_absolute_by_operation: dict[str, float] = {}
    off_route_relative_by_operation: dict[str, float] = {}
    off_route_max_entry_by_operation: dict[str, float] = {}
    raw_unitarity_by_operation: dict[str, float] = {}
    raw_unitarity_bound_by_operation: dict[str, float] = {}
    target_hashes: dict[str, str] = {}
    transformed_hashes: dict[str, str] = {}
    changed_entries: dict[str, int] = {}
    branch_status: dict[str, str] = {}
    branch_margins: list[float] = []

    for name, action in target_actions.items():
        if operation_antiunitary is not None and bool(
            operation_antiunitary.get(name)
        ) != bool(action.antiunitary):
            raise CanonicalTargetError(
                f"fixed-target antiunitary parity mismatch for {name!r}"
            )
        raw = np.asarray(raw_matrices[name], dtype=np.complex128)
        target = materialize_block_route_action(action)
        if raw.shape != target.shape or not np.all(np.isfinite(raw)):
            raise CanonicalTargetError(
                f"fixed-target input {name!r} has an invalid shape or values"
            )
        route_mask = np.zeros(raw.shape, dtype=bool)
        route_values: list[float] = []
        operation_margins: list[float] = []
        for source, target_fiber in enumerate(action.fiber_permutation):
            rows = action.fiber_indices[target_fiber]
            columns = action.fiber_indices[source]
            route_mask[np.ix_(rows, columns)] = True
            raw_block = np.asarray(raw[np.ix_(rows, columns)], dtype=np.complex128)
            target_block = np.asarray(action.route_blocks[source], dtype=np.complex128)
            correction = float(
                np.linalg.norm(raw_block - target_block, ord="fro")
                / np.sqrt(max(1, target_block.shape[1]))
            )
            route_values.append(correction)
            margin = _monomial_branch_margin(
                raw_block,
                target_block,
                root_order=order,
            )
            if margin is not None:
                operation_margins.append(margin)
                branch_margins.append(margin)
        values = np.asarray(route_values, dtype=np.float64)
        route_rms = float(np.sqrt(np.mean(np.square(values))))
        route_max = float(np.max(values))
        route_values_all.extend(float(value) for value in values)
        route_rms_by_operation[name] = route_rms
        route_max_by_operation[name] = route_max

        off_route = np.where(route_mask, 0.0, raw)
        raw_norm = float(np.linalg.norm(raw, ord="fro"))
        off_absolute = float(np.linalg.norm(off_route, ord="fro"))
        off_relative = float(off_absolute / max(raw_norm, np.finfo(np.float64).tiny))
        off_max_entry = float(np.max(np.abs(off_route)))
        off_route_absolute_by_operation[name] = off_absolute
        off_route_relative_by_operation[name] = off_relative
        off_route_max_entry_by_operation[name] = off_max_entry

        difference = raw - target
        difference_norm = float(np.linalg.norm(difference, ord="fro"))
        dense_normalized = float(difference_norm / np.sqrt(max(1, raw.shape[0])))
        dense_relative = float(
            difference_norm
            / max(float(np.linalg.norm(target, ord="fro")), np.finfo(np.float64).tiny)
        )
        dense_normalized_by_operation[name] = dense_normalized
        dense_relative_by_operation[name] = dense_relative
        unitarity = float(
            np.linalg.norm(
                raw.conjugate().T @ raw
                - np.eye(raw.shape[0], dtype=np.complex128),
                ord="fro",
            )
            / np.sqrt(max(1, raw.shape[0]))
        )
        # With T unitary and E=raw-T, this bound follows directly from
        # T^H E + E^H T + E^H E, plus a dimension-scaled roundoff allowance.
        unitarity_bound = float(
            (
                2.0 * difference_norm
                + difference_norm * difference_norm
                + 512.0
                * np.finfo(np.float64).eps
                * max(1, raw.shape[0] ** 2)
            )
            / np.sqrt(max(1, raw.shape[0]))
        )
        raw_unitarity_by_operation[name] = unitarity
        raw_unitarity_bound_by_operation[name] = unitarity_bound
        if unitarity > unitarity_bound:
            raise CanonicalTargetError(
                f"fixed-target input {name!r} is too non-unitary: "
                f"residual={unitarity:.6e}, derived_bound={unitarity_bound:.6e}"
            )
        if off_relative > rms_limit or off_max_entry > route_limit:
            raise CanonicalTargetError(
                f"fixed-target input {name!r} has excessive off-route support: "
                f"relative={off_relative:.6e}, max_entry={off_max_entry:.6e}"
            )
        if dense_relative > rms_limit and dense_normalized > rms_limit:
            raise CanonicalTargetError(
                f"fixed-target RMS correction exceeds configured limit for {name!r}: "
                f"relative={dense_relative:.6e}, normalized={dense_normalized:.6e}, "
                f"limit={rms_limit:.6e}"
            )
        target_hashes[name] = hash_array(target)
        transformed_hashes[name] = hash_array(raw)
        changed_entries[name] = int(np.count_nonzero(raw != target))
        if len(operation_margins) == len(action.route_blocks):
            branch_status[name] = (
                "diagnostic_local_target_nearest"
                if min(operation_margins) > 0.0
                else "diagnostic_local_alternative_closer_non_authoritative"
            )
        else:
            branch_status[name] = "not_applicable_continuous_Ud"

    combined = np.asarray(route_values_all, dtype=np.float64)
    correction_rms = float(np.sqrt(np.mean(np.square(combined))))
    correction_max = float(np.max(combined))
    if correction_rms > rms_limit:
        raise CanonicalTargetError(
            "fixed-target RMS correction exceeds configured limit: "
            f"correction={correction_rms:.6e}, limit={rms_limit:.6e}"
        )
    if correction_max > route_limit:
        raise CanonicalTargetError(
            "fixed-target route correction exceeds configured limit: "
            f"correction={correction_max:.6e}, limit={route_limit:.6e}"
        )
    all_discrete = all(
        value.startswith("diagnostic_local_")
        for value in branch_status.values()
    )
    minimum_margin: float | None = min(branch_margins) if all_discrete else None
    return {
        "version": "fixed_target_exactification_certificate_v2",
        "status": "certified",
        "selection_policy": (
            "persisted_project_owned_target_with_direct_correction_budget"
        ),
        "selection_scope": (
            "project_owned_joint_representation; local_column_branch_is_diagnostic_only"
        ),
        "exactification_target_applied": True,
        "additional_basis_gauge_applied": False,
        "root_order": order,
        "branch_check_by_operation": branch_status,
        "minimum_branch_margin": minimum_margin,
        "correction_rms_by_operation": route_rms_by_operation,
        "correction_max_by_operation": route_max_by_operation,
        "correction_rms": correction_rms,
        "correction_max": correction_max,
        "dense_correction_relative_fro_by_operation": dense_relative_by_operation,
        "dense_correction_normalized_fro_by_operation": dense_normalized_by_operation,
        "off_route_absolute_fro_by_operation": off_route_absolute_by_operation,
        "off_route_relative_fro_by_operation": off_route_relative_by_operation,
        "off_route_max_entry_by_operation": off_route_max_entry_by_operation,
        "raw_unitarity_residual_by_operation": raw_unitarity_by_operation,
        "raw_unitarity_derived_bound_by_operation": raw_unitarity_bound_by_operation,
        "max_rms_correction": rms_limit,
        "max_route_correction": route_limit,
        "source_matrix_hashes": (
            {} if source_matrix_hashes is None else dict(source_matrix_hashes)
        ),
        "transformed_raw_matrix_hashes": transformed_hashes,
        "target_matrix_hashes": target_hashes,
        "numeric_entries_changed_by_exactification": changed_entries,
        # Compatibility alias: zero refers only to the subsequent storage pass.
        "numeric_entries_modified": 0,
        "numeric_entries_modified_scope": _POST_STORAGE_CLEANUP,
        "post_exactification_storage_cleanup": {
            "version": _POST_STORAGE_CLEANUP,
            "numeric_entries_modified": 0,
            "threshold_cleanup_performed": False,
        },
        "persisted_frame_hash": persisted_frame_hash,
        "target_artifact_hash": target_artifact_hash,
        "target_certificate_identity": (
            {} if target_certificate_identity is None else dict(target_certificate_identity)
        ),
        "operation_antiunitary": {
            name: bool(action.antiunitary) for name, action in target_actions.items()
        },
        "route_layout": {
            name: {
                "fiber_permutation": list(action.fiber_permutation),
                "fiber_dimensions": list(action.fiber_dimensions),
                "fiber_indices": [list(group) for group in action.fiber_indices],
            }
            for name, action in target_actions.items()
        },
    }


__all__ = [
    "CANONICAL_TARGET_ACTIONS_V1",
    "LITERAL_ROUTE_SOURCE_ACTIONS_V1",
    "CanonicalTargetBundle",
    "CanonicalTargetError",
    "LiteralRouteSourceBundle",
    "certify_fixed_target_exactification",
    "decode_canonical_target_actions",
    "encode_canonical_target_actions",
    "encode_literal_route_source_actions",
    "load_canonical_target_artifact",
    "load_literal_route_source_artifact",
]
