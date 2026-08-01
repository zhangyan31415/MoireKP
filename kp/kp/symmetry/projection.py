from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..blocks.blocks import (
    _assemble_gamma_projectors_from_block_eigenvectors,
    _assemble_projectors_from_block_eigenvectors,
    _layer_reference_entries,
    get_H_block,
    ProjectGaugeAnchorCandidate,
    resolve_project_gauge_anchor_candidates,
)
from ..blocks.gamma_layout import GammaRoutingError
from ..blocks.downfold import DownfoldingOptions, downfold_from_projectors
from ..basis.selection import (
    GaugeAnchorReport,
    GaugeCandidateSymmetryMetrics,
    select_gauge_candidate_by_symmetry,
    write_basis_selection_report,
)
from ..basis.symmetry_gauge import (
    SymmetryAdaptedBasisFrame,
    derive_symmetry_adapted_basis_frame,
    transform_basis_operation,
)
from ..config.case import normalize_case_config
from ..io.tapw_loader import load_Q_sets, load_hamk
from ..identity import (
    PROJECTION_ARTIFACT_IDENTITY_FIELDS,
    PROJECTION_BASIS_HANDOFF_VERSION,
    build_projection_basis_identity,
    hash_array,
    load_projection_artifact_identity,
    require_identity_fields,
    require_matching_identity,
)
from ..model.schema import M_EFFECTIVE_OPERATION_ALIASES
from ..projection_handoff import (
    ExplicitLegacyBasisSpec,
    GAMMA_ROUTED_ONLY_BASIS_FIELDS,
    GammaRoutedBasisSpec,
    ProjectionBasisSpec,
    load_gamma_routed_basis_spec,
)
from ..projection_selection import CandidateRejectionReason
from .candidate_certificate import evaluate_projected_pair
from .exactify_representation import exactify_loaded_symmetry_source
from .canonical_target import (
    CanonicalTargetError,
    certify_fixed_target_exactification,
    encode_canonical_target_actions,
    encode_literal_route_source_actions,
    load_canonical_target_artifact,
)
from .factorized_action import (
    FACTORIZED_RESPONSE_ACTION_V1,
    FactorizedActionCertificationError,
    certify_factorized_action,
    explicit_linear_action_matrix,
    pack_factorized_actions,
)
from .geometry import (
    bM_candidates_from_q_distances,
    canonical_bM_pair_from_candidates,
    sectors_with_q_offsets,
)
from .joint_exactification import (
    _cyclotomic_root,
    BlockRouteAction,
    JointExactificationError,
    JointExactificationConfig,
    JointExactificationResult,
    certify_fixed_target_joint_result,
    derive_closest_cyclotomic_u1_gauge,
    derive_standard_generator_fiber_gauge,
    extract_block_route_action,
    joint_exactify_block_actions,
    load_joint_exactification_artifact,
    materialize_block_route_action,
)
from .q_canonicalization import CanonicalQResult, canonicalize_q_geometry
from .structure_certificate import (
    CertificationStatus,
    FiberSectorLayout,
    StructureGrade,
    certify_common_fiber_gauge,
    certify_joint_structure,
    common_fiber_gauge_hash,
    compare_joint_structure,
    derive_target_basis_hash,
)
from .structure_recovery import (
    recover_fixed_target_common_gauge,
    recover_uniform_internal_action,
)

try:
    import scipy.sparse as _sparse
except Exception:  # pragma: no cover - scipy is optional for dense-only inputs
    _sparse = None


@dataclass
class ProjectionState:
    hamk: np.ndarray
    heff: np.ndarray
    u_low: np.ndarray


@dataclass
class RepresentationData:
    matrix: np.ndarray
    from_full_spinful: bool
    spin_leakage: float | None


@dataclass
class ActionRepresentation:
    matrix: Any
    representation: RepresentationData | None
    pg: RepresentationData | None
    raw_h: RepresentationData | None
    pg_filename: str | None
    raw_h_filename: str | None
    action_source: str
    combined_raw_h_residual: float | None


class ProjectionTransactionIntegrityError(RuntimeError):
    """A certified handoff is internally inconsistent and must not roll back."""


def _select_validated_auto_gauge_candidate(
    candidates: Sequence[Any],
    metrics: Sequence[GaugeCandidateSymmetryMetrics],
    *,
    max_exactification_distance: float = 1.0e-3,
):
    """Select a resolved auto-gauge candidate from symmetry validation metrics."""

    decision = select_gauge_candidate_by_symmetry(
        metrics,
        max_exactification_distance=max_exactification_distance,
    )
    by_id = {str(candidate.candidate_id): candidate for candidate in candidates}
    selected = by_id.get(str(decision.selected.candidate_id))
    if selected is None:
        raise ValueError(
            f"symmetry validation selected unknown gauge candidate {decision.selected.candidate_id!r}"
        )
    return selected, decision


def _resolve(path: str | None, base_dir: str) -> str | None:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


_SUPPORTED_OPERATION_LABELS = frozenset({"C3", "C3z", "C2", "C2T", "T", "TR"})
HARTREE_TO_EV = 27.211386245988
_PROJECT_ARTIFACT_IDENTITY_FIELDS = PROJECTION_ARTIFACT_IDENTITY_FIELDS
_PROJECT_BASIS_IDENTITY_FIELDS = _PROJECT_ARTIFACT_IDENTITY_FIELDS[:-1]


def _save_exactified_matrix(
    path: str | Path,
    matrix: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    saved = np.array(matrix, dtype=np.complex128, copy=True)
    np.save(path, saved)
    return saved, {
        "policy": "structural_route_support_v1",
        "numeric_entries_modified": 0,
    }


def _invalidate_stale_canonical_symmetry_outputs(output_dir: Path) -> None:
    """Make a failed rerun unambiguously non-consumable.

    Raw projection diagnostics may remain for debugging, but no production
    package or exact matrix from an older successful run may survive into a
    new certification attempt.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "representations.npz",
        "manifest.json",
        "summary.json",
        "summary.md",
        "residuals.csv",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()
    for pattern in ("exactified_*.npy", "*_exactification_report.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _validate_persisted_project_structure_frame(
    frame: SymmetryAdaptedBasisFrame,
    *,
    expected_dimension: int,
) -> dict[str, Any]:
    """Validate that persisted ``W`` is the gauge certified by project."""

    if frame.status != "applied":
        raise ValueError(
            "a persisted joint structure transaction requires an applied frame"
        )
    transaction = frame.components.get("joint_structure_transaction")
    if not isinstance(transaction, Mapping):
        raise ValueError("project symmetry frame lacks a joint structure transaction")
    pre = transaction.get("pre_certificate")
    post = transaction.get("post_certificate")
    witness = transaction.get("common_gauge_witness")
    transition = transaction.get("transition")
    if not all(
        isinstance(value, Mapping)
        for value in (pre, post, witness, transition)
    ):
        raise ValueError("project joint structure transaction is incomplete")
    fixed_alignment = frame.components.get(
        "fixed_target_common_gauge_alignment"
    )
    alignment_report = (
        fixed_alignment.get("alignment")
        if isinstance(fixed_alignment, Mapping)
        else None
    )
    source_classifier = (
        fixed_alignment.get("source_structure_classifier")
        if isinstance(fixed_alignment, Mapping)
        else None
    )
    literal_source = (
        fixed_alignment.get("phase_preserving_source_actions")
        if isinstance(fixed_alignment, Mapping)
        else None
    )
    direct_certificate = (
        fixed_alignment.get("direct_raw_to_target_certificate")
        if isinstance(fixed_alignment, Mapping)
        else None
    )
    phase_source_report = (
        fixed_alignment.get("phase_preserving_source")
        if isinstance(fixed_alignment, Mapping)
        else None
    )
    alignment_error_bounds = (
        alignment_report.get("source_action_absolute_error_bounds")
        if isinstance(alignment_report, Mapping)
        else None
    )
    reported_source_error_bounds = (
        phase_source_report.get(
            "joint_relation_correction_max_by_operation"
        )
        if isinstance(phase_source_report, Mapping)
        else None
    )
    bounded_alignment_evidence = bool(
        isinstance(alignment_error_bounds, Mapping)
        and isinstance(reported_source_error_bounds, Mapping)
        and set(alignment_error_bounds) == set(reported_source_error_bounds)
        and all(
            isinstance(alignment_error_bounds[name], (int, float, np.number))
            and isinstance(
                reported_source_error_bounds[name],
                (int, float, np.number),
            )
            and np.isfinite(float(alignment_error_bounds[name]))
            and float(alignment_error_bounds[name]) >= 0.0
            and float(alignment_error_bounds[name])
            == float(reported_source_error_bounds[name])
            for name in alignment_error_bounds
        )
    )
    alignment_certification_basis_is_bound = bool(
        isinstance(alignment_report, Mapping)
        and (
            alignment_report.get("certification_basis")
            == "exact_source_relations+exact_target_structure+common_gauge_witness"
            or (
                alignment_report.get("certification_basis")
                == "certified_source_relations+exact_target_structure+"
                "bounded_common_gauge_witness"
                and bounded_alignment_evidence
            )
        )
    )
    conventional_transaction = bool(
        pre.get("status") == CertificationStatus.CERTIFIED.value
        and transition.get("status") == CertificationStatus.CERTIFIED.value
        and transition.get("is_monotone") is True
    )
    fixed_target_ambiguous_source = bool(
        pre.get("status") == CertificationStatus.UNKNOWN_OR_UNSUPPORTED.value
        and isinstance(fixed_alignment, Mapping)
        and fixed_alignment.get("status") == CertificationStatus.CERTIFIED.value
        and isinstance(alignment_report, Mapping)
        and alignment_report.get("status") == CertificationStatus.CERTIFIED.value
        and alignment_certification_basis_is_bound
        and alignment_report.get("target_was_modified") is False
        and isinstance(source_classifier, Mapping)
        and source_classifier.get("status")
        == CertificationStatus.UNKNOWN_OR_UNSUPPORTED.value
        and source_classifier.get("production_gate") is False
        and isinstance(literal_source, Mapping)
        and literal_source.get("status") == "relation_certified"
        and isinstance(direct_certificate, Mapping)
        and direct_certificate.get("status") == CertificationStatus.CERTIFIED.value
    )
    if (
        post.get("status") != CertificationStatus.CERTIFIED.value
        or witness.get("status") != CertificationStatus.CERTIFIED.value
        or not (conventional_transaction or fixed_target_ambiguous_source)
    ):
        raise ValueError("project joint structure transaction is not certified")
    invariant_fields = (
        "layout_hash",
        "presentation_hash",
        "declared_cocycle_hash",
    )
    changed_invariants = tuple(
        field for field in invariant_fields if pre.get(field) != post.get(field)
    )
    if changed_invariants:
        raise ValueError(
            "project gauge transaction changed protected structure identity: "
            + ", ".join(changed_invariants)
        )

    operations = post.get("operations")
    if not isinstance(operations, Mapping) or not operations:
        raise ValueError("project post certificate lacks operation fiber metadata")
    reference_operation = next(iter(operations.values()))
    if not isinstance(reference_operation, Mapping):
        raise ValueError("project post certificate has invalid operation metadata")
    raw_fiber_indices = reference_operation.get("fiber_indices")
    if not isinstance(raw_fiber_indices, Sequence) or isinstance(
        raw_fiber_indices, (str, bytes)
    ):
        raise ValueError("project post certificate lacks protected fiber indices")
    fiber_indices = tuple(
        tuple(int(index) for index in members)
        for members in raw_fiber_indices
    )
    flattened = tuple(index for members in fiber_indices for index in members)
    if tuple(sorted(flattened)) != tuple(range(int(expected_dimension))):
        raise ValueError(
            "project protected fibers do not partition the persisted basis"
        )
    for name, operation in operations.items():
        if not isinstance(operation, Mapping) or tuple(
            tuple(int(index) for index in members)
            for members in operation.get("fiber_indices", ())
        ) != fiber_indices:
            raise ValueError(
                f"project post certificate has inconsistent fibers for {name!r}"
            )

    full_gauge = np.asarray(frame.full_unitary, dtype=np.complex128)
    expected_shape = (int(expected_dimension), int(expected_dimension))
    if full_gauge.shape != expected_shape or not np.all(np.isfinite(full_gauge)):
        raise ValueError(
            "persisted project frame has an invalid shape or non-finite values"
        )
    assembled = np.zeros_like(full_gauge)
    fiber_gauge: list[np.ndarray] = []
    for indices in fiber_indices:
        block = np.asarray(full_gauge[np.ix_(indices, indices)], dtype=np.complex128)
        dimension = int(block.shape[0])
        residual = float(
            np.linalg.norm(
                block.conjugate().T @ block
                - np.eye(dimension, dtype=np.complex128),
                ord="fro",
            )
        )
        bound = float(
            128.0
            * np.finfo(np.float64).eps
            * max(1, dimension * dimension)
            * max(1.0, float(np.linalg.norm(block, ord="fro") ** 2))
        )
        if residual > bound:
            raise ValueError(
                "persisted project frame has a non-unitary protected-fiber block: "
                f"residual={residual:.6e}, bound={bound:.6e}"
            )
        assembled[np.ix_(indices, indices)] = block
        fiber_gauge.append(block)
    if not np.array_equal(full_gauge, assembled):
        raise ValueError("persisted project frame mixes protected continuum fibers")
    gauge_hash = common_fiber_gauge_hash(fiber_gauge)
    if gauge_hash != str(witness.get("gauge_hash", "")):
        raise ValueError(
            "persisted project frame does not match the certified transaction gauge"
        )
    if (
        witness.get("source_actions_hash") != pre.get("actions_hash")
        or witness.get("target_actions_hash") != post.get("actions_hash")
        or witness.get("presentation_hash") != post.get("presentation_hash")
        or witness.get("source_basis_hash") != pre.get("basis_frame_hash")
        or witness.get("target_basis_hash") != post.get("basis_frame_hash")
    ):
        raise ValueError(
            "project common-gauge witness is not bound to its structure certificates"
        )
    expected_target_basis_hash = derive_target_basis_hash(
        str(pre.get("basis_frame_hash", "")),
        fiber_gauge,
    )
    if expected_target_basis_hash != str(post.get("basis_frame_hash", "")):
        raise ValueError(
            "project post-certificate basis hash is not derived from persisted W"
        )
    return {
        "transaction": transaction,
        "pre_certificate": pre,
        "post_certificate": post,
        "common_gauge_witness": witness,
        "fiber_indices": fiber_indices,
        "gauge_hash": gauge_hash,
        "fixed_target_ambiguous_source_classifier": (
            fixed_target_ambiguous_source
        ),
    }


def _transform_projected_operations_to_persisted_frame(
    matrices: Mapping[str, np.ndarray],
    operations: Sequence[Mapping[str, Any]],
    frame: SymmetryAdaptedBasisFrame | None,
) -> dict[str, np.ndarray]:
    """Put raw projected operations in the basis already written by project.

    Exactification is deliberately downstream of this transformation.  The
    two operations do not commute numerically, and the production symmetry
    matrices must live in the same persisted basis as Heff and wavefunctions.
    """

    parity_by_name = {
        str(operation["name"]): bool(operation.get("antiunitary", False))
        for operation in operations
    }
    if set(matrices) != set(parity_by_name):
        raise ValueError(
            "raw projected symmetry matrices do not match operation metadata"
        )
    if frame is None or frame.status != "applied":
        return {
            name: np.array(value, dtype=np.complex128, order="C", copy=True)
            for name, value in matrices.items()
        }
    dimension = int(frame.full_unitary.shape[0])
    if frame.full_unitary.shape != (dimension, dimension):
        raise ValueError("persisted symmetry frame must be square")
    if isinstance(frame.components.get("joint_structure_transaction"), Mapping):
        _validate_persisted_project_structure_frame(
            frame,
            expected_dimension=dimension,
        )
    else:
        full = np.asarray(frame.full_unitary, dtype=np.complex128)
        if not np.all(np.isfinite(full)):
            raise ValueError("persisted symmetry frame contains non-finite values")
        residual = float(
            np.linalg.norm(
                full.conjugate().T @ full
                - np.eye(dimension, dtype=np.complex128),
                ord="fro",
            )
        )
        bound = float(
            128.0
            * np.finfo(np.float64).eps
            * max(1, dimension * dimension)
            * max(1.0, float(np.linalg.norm(full, ord="fro") ** 2))
        )
        if residual > bound:
            raise ValueError(
                "persisted symmetry frame is not unitary: "
                f"residual={residual:.6e}, bound={bound:.6e}"
            )
    transformed: dict[str, np.ndarray] = {}
    for name, value in matrices.items():
        matrix = np.asarray(value, dtype=np.complex128)
        if matrix.shape != (dimension, dimension):
            raise ValueError(
                f"{name} projected symmetry/frame dimension mismatch: "
                f"{matrix.shape} != {(dimension, dimension)}"
            )
        transformed[name] = transform_basis_operation(
            matrix,
            frame.full_unitary,
            antiunitary=parity_by_name[name],
        )
    return transformed


def _certify_exact_target_in_persisted_project_basis(
    canonical_target_artifact: Mapping[str, Any],
    *,
    persisted_frame: SymmetryAdaptedBasisFrame,
    persisted_frame_artifact: Mapping[str, Any],
    sectors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-certify the persisted target without running a second solver."""

    try:
        target = load_canonical_target_artifact(canonical_target_artifact)
    except CanonicalTargetError as exc:
        raise ValueError(f"invalid persisted canonical target: {exc}") from exc
    reference = target.actions[target.presentation.generators[0].name]
    expected_dimension = int(sum(reference.fiber_dimensions))
    validated = _validate_persisted_project_structure_frame(
        persisted_frame,
        expected_dimension=expected_dimension,
    )
    persisted_post = validated["post_certificate"]
    persisted_witness = validated["common_gauge_witness"]
    persisted_fiber_indices = validated["fiber_indices"]
    gauge_hash = str(validated["gauge_hash"])
    if tuple(persisted_fiber_indices) != tuple(reference.fiber_indices):
        raise ValueError(
            "project target and persisted frame use different protected fiber layouts"
        )

    current_layout = _joint_fiber_sector_layout(
        target.actions,
        target.presentation,
        sectors,
        basis_frame_hash=str(persisted_post.get("basis_frame_hash", "")),
    )
    current_post = certify_joint_structure(
        target.actions,
        target.presentation,
        layout=current_layout,
    )
    if current_post.status is CertificationStatus.UNKNOWN_OR_UNSUPPORTED:
        propagated_bounds = persisted_witness.get(
            "certification_bound_by_operation",
            {},
        )
        if isinstance(propagated_bounds, Mapping) and propagated_bounds:
            current_post = certify_joint_structure(
                target.actions,
                target.presentation,
                layout=current_layout,
                block_absolute_error_bounds={
                    str(name): float(value)
                    for name, value in propagated_bounds.items()
                },
            )
    if current_post.status is not CertificationStatus.CERTIFIED:
        raise ValueError("current project-basis joint actions are not certified")
    current_post_artifact = current_post.artifact()
    structure_identity_fields = (
        "layout_hash",
        "presentation_hash",
        "declared_cocycle_hash",
        "actions_hash",
    )
    post_mismatches = tuple(
        field
        for field in structure_identity_fields
        if persisted_post.get(field) != current_post_artifact.get(field)
    )
    if post_mismatches:
        raise ValueError(
            "decoded project target does not reproduce its post certificate: "
            + ", ".join(post_mismatches)
        )
    certificate_identity_fields = (
        "version",
        "status",
        "basis_frame_hash",
        *structure_identity_fields,
        "certificate_hash",
    )
    identity_mismatches = tuple(
        field
        for field in certificate_identity_fields
        if target.certificate_identity.get(field) != persisted_post.get(field)
    )
    if identity_mismatches:
        raise ValueError(
            "persisted canonical target certificate identity mismatch: "
            + ", ".join(identity_mismatches)
        )
    return {
        "version": "kp_persisted_project_frame_revalidation_v3",
        "status": "certified",
        "policy": "project_owned_target_recertified_without_free_solver",
        "persisted_frame_hash": str(persisted_frame_artifact.get("frame_hash", "")),
        "canonical_target_artifact_hash": target.artifact_hash,
        "second_gauge_selection_performed": False,
        "additional_basis_gauge_required": False,
        "additional_basis_gauge_applied": False,
        "basis_transform": {
            "unitary": "W_dagger_D_W",
            "antiunitary": "W_dagger_D_W_conjugate",
        },
        "canonical_action_identity": {
            field: current_post_artifact[field]
            for field in structure_identity_fields
        },
        "target_certificate_identity": dict(target.certificate_identity),
        "current_project_basis_certificate": current_post_artifact,
        "common_gauge_witness": dict(persisted_witness),
        "persisted_gauge_hash_recomputed": gauge_hash,
        "production_free_joint_solver": {
            "status": "not_run",
            "production_used": False,
        },
    }


def _joint_fiber_sector_layout(
    actions: Mapping[str, Any],
    presentation: Any,
    sectors: Sequence[Mapping[str, Any]],
    *,
    basis_frame_hash: str,
) -> FiberSectorLayout:
    reference = actions[presentation.generators[0].name]
    names: list[str] = []
    owners: list[int] = []
    expected_dimensions: list[int] = []
    for sector_index, raw in enumerate(sectors):
        name = str(raw.get("name", f"sector_{sector_index}"))
        n_q = int(raw["n_q"])
        n_orb = int(raw["n_orb"])
        if n_q <= 0 or n_orb < 0:
            raise ValueError("joint gauge sectors require positive n_q and n_orb")
        if n_orb == 0:
            continue
        compact_sector_index = len(names)
        names.append(name)
        owners.extend([compact_sector_index] * n_q)
        expected_dimensions.extend([n_orb] * n_q)
    if len(owners) != len(reference.fiber_dimensions):
        raise ValueError(
            "joint gauge sector layout does not cover every exactified fiber: "
            f"{len(owners)} != {len(reference.fiber_dimensions)}"
        )
    if tuple(expected_dimensions) != tuple(reference.fiber_dimensions):
        raise ValueError(
            "joint gauge sector dimensions disagree with the exactified fiber layout"
        )
    return FiberSectorLayout(
        sector_names=tuple(names),
        fiber_sectors=tuple(owners),
        basis_frame_hash=str(basis_frame_hash),
    )


def _joint_target_frame_hash(
    source_frame_hash: str,
    fiber_gauge: Sequence[np.ndarray],
) -> str:
    return derive_target_basis_hash(source_frame_hash, fiber_gauge)


def _identity_fiber_gauge(actions: Mapping[str, Any], presentation: Any) -> tuple[np.ndarray, ...]:
    reference = actions[presentation.generators[0].name]
    return tuple(
        np.eye(int(dimension), dtype=np.complex128)
        for dimension in reference.fiber_dimensions
    )


def _uniform_algebraic_identity_report(
    actions: Mapping[str, Any],
    presentation: Any,
    certificate: Any,
) -> tuple[bool, dict[str, Any]]:
    """Recognize a zero-cost, already algebraic monomial joint representation."""

    if all(
        dimension == 1
        for action in actions.values()
        for dimension in action.fiber_dimensions
    ):
        return False, {
            "status": "not_applicable",
            "reason": "scalar_routes_use_the_existing_closest_cyclotomic_solver",
        }
    if certificate.status is not CertificationStatus.CERTIFIED:
        return False, {
            "status": "unknown_or_unsupported",
            "reason": "pre_structure_certificate_is_not_decisive",
        }
    for operation in certificate.operations.values():
        if any(
            row.grade is not StructureGrade.UNIFORM_INTERNAL
            for row in operation.source_sectors.values()
        ):
            return False, {
                "status": "not_applicable",
                "reason": "not_all_source_sector_actions_are_uniform_internal",
            }

    root_order = 1
    power_generators: set[str] = set()
    for relation in presentation.relations:
        if relation.rhs or not relation.lhs:
            continue
        name = relation.lhs[0]
        if any(value != name for value in relation.lhs):
            continue
        central_order = 1 if relation.central_phase == complex(1.0, 0.0) else 2
        root_order = math.lcm(root_order, len(relation.lhs) * central_order)
        power_generators.add(name)
    expected_generators = {generator.name for generator in presentation.generators}
    if power_generators != expected_generators or root_order <= 1 or root_order > 4096:
        return False, {
            "status": "unknown_or_unsupported",
            "reason": "presentation_lacks_bounded_power_relations_for_all_generators",
        }
    roots = np.asarray(
        [_cyclotomic_root(root_order, exponent) for exponent in range(root_order)],
        dtype=np.complex128,
    )
    reference = actions[presentation.generators[0].name]
    bound = float(
        8192.0
        * np.finfo(np.float64).eps
        * max(
            1,
            root_order,
            len(reference.fiber_dimensions),
            max(reference.fiber_dimensions),
        )
    )
    encodings: dict[str, dict[str, Any]] = {}
    for name, operation_certificate in certificate.operations.items():
        action = actions[name]
        operation_encoding: dict[str, Any] = {}
        for sector_name, row in operation_certificate.source_sectors.items():
            if row.reference_fiber is None:
                return False, {
                    "status": "unknown_or_unsupported",
                    "reason": f"{name}/{sector_name}_lacks_reference_fiber",
                }
            block = np.asarray(
                action.route_blocks[row.reference_fiber],
                dtype=np.complex128,
            )
            if any(
                not np.array_equal(
                    np.asarray(action.route_blocks[source], dtype=np.complex128),
                    block,
                )
                for source in row.source_fibers
            ):
                return False, {
                    "status": "not_applicable",
                    "reason": f"{name}/{sector_name}_is_only_numerically_uniform",
                }
            dimension = int(block.shape[0])
            targets = tuple(
                int(np.argmax(np.abs(block[:, source])))
                for source in range(dimension)
            )
            if sorted(targets) != list(range(dimension)):
                return False, {
                    "status": "not_applicable",
                    "reason": f"{name}/{sector_name}_is_not_monomial",
                }
            exact = np.zeros_like(block)
            exponents: list[int] = []
            for source, target in enumerate(targets):
                value = complex(block[target, source])
                distances = np.abs(roots - value)
                exponent = int(np.argmin(distances))
                root = complex(roots[exponent])
                if value != root:
                    return False, {
                        "status": "not_applicable",
                        "reason": f"{name}/{sector_name}_is_only_near_a_cyclotomic_phase",
                    }
                exact[target, source] = root
                exponents.append(exponent)
            if not np.array_equal(block, exact):
                return False, {
                    "status": "not_applicable",
                    "reason": f"{name}/{sector_name}_has_literal_off_monomial_entries",
                }
            operation_encoding[sector_name] = {
                "support_target_by_source": list(targets),
                "root_exponents": exponents,
            }
        encodings[name] = operation_encoding
    return True, {
        "status": "certified",
        "reason": "zero_cost_uniform_algebraic_monomial_representation",
        "cyclotomic_root_order": int(root_order),
        "certification_bound": bound,
        "literal_algebraic_equality": True,
        "encodings": encodings,
    }


def _joint_config_from_artifact(
    value: Mapping[str, Any],
) -> JointExactificationConfig:
    return JointExactificationConfig(
        enabled=bool(value.get("enabled", True)),
        max_rms_correction=float(value["max_rms_correction"]),
        max_route_correction=float(value["max_route_correction"]),
        central_branch_margin=float(value["central_branch_margin"]),
        max_iterations=int(value["max_iterations"]),
        condition_limit=float(value["condition_limit"]),
    )


def _phase_preserving_joint_source(
    raw_matrices: Mapping[str, np.ndarray],
    reference_actions: Mapping[str, BlockRouteAction],
    presentation: Any,
    *,
    config: JointExactificationConfig,
) -> tuple[JointExactificationResult, dict[str, Any]]:
    """Remove only floating noise while retaining continuous raw gauge phases."""

    names = tuple(generator.name for generator in presentation.generators)
    if set(raw_matrices) != set(names) or set(reference_actions) != set(names):
        raise ValueError(
            "raw phase-preserving source must match the complete presentation"
        )
    polar_rms_by_operation: dict[str, float] = {}
    polar_max_by_operation: dict[str, float] = {}
    off_route_absolute_by_operation: dict[str, float] = {}
    off_route_relative_by_operation: dict[str, float] = {}
    off_route_max_entry_by_operation: dict[str, float] = {}
    raw_hashes: dict[str, str] = {}
    polar_actions: dict[str, BlockRouteAction] = {}
    all_polar_corrections: list[float] = []
    epsilon = float(np.finfo(np.float64).eps)
    for generator in presentation.generators:
        name = generator.name
        reference = reference_actions[name]
        raw = np.asarray(raw_matrices[name], dtype=np.complex128)
        dimension = int(sum(reference.fiber_dimensions))
        if raw.shape != (dimension, dimension) or not np.all(np.isfinite(raw)):
            raise ValueError(
                f"raw phase-preserving source {name!r} has invalid shape or values"
            )
        route_mask = np.zeros(raw.shape, dtype=bool)
        blocks: list[np.ndarray] = []
        corrections: list[float] = []
        unitarity_bounds: list[float] = []
        for source, target in enumerate(reference.fiber_permutation):
            rows = reference.fiber_indices[target]
            columns = reference.fiber_indices[source]
            route_mask[np.ix_(rows, columns)] = True
            block = np.asarray(raw[np.ix_(rows, columns)], dtype=np.complex128)
            left, _singular_values, right = np.linalg.svd(block)
            unitary = np.asarray(left @ right, dtype=np.complex128)
            correction = float(
                np.linalg.norm(block - unitary, ord="fro")
                / np.sqrt(max(1, unitary.shape[1]))
            )
            residual = float(
                np.linalg.norm(
                    unitary.conjugate().T @ unitary
                    - np.eye(unitary.shape[1], dtype=np.complex128),
                    ord="fro",
                )
            )
            bound = float(
                residual
                + 512.0
                * epsilon
                * max(1, unitary.shape[0] ** 2)
                * max(1.0, float(np.linalg.norm(unitary, ord="fro") ** 2))
            )
            blocks.append(unitary)
            corrections.append(correction)
            unitarity_bounds.append(bound)
        values = np.asarray(corrections, dtype=np.float64)
        polar_rms = float(np.sqrt(np.mean(np.square(values))))
        polar_max = float(np.max(values))
        polar_rms_by_operation[name] = polar_rms
        polar_max_by_operation[name] = polar_max
        all_polar_corrections.extend(float(value) for value in values)
        off_route = np.where(route_mask, 0.0, raw)
        off_absolute = float(np.linalg.norm(off_route, ord="fro"))
        off_relative = float(
            off_absolute
            / max(float(np.linalg.norm(raw, ord="fro")), np.finfo(np.float64).tiny)
        )
        off_max = float(np.max(np.abs(off_route)))
        off_route_absolute_by_operation[name] = off_absolute
        off_route_relative_by_operation[name] = off_relative
        off_route_max_entry_by_operation[name] = off_max
        if off_relative > config.max_rms_correction or off_max > config.max_route_correction:
            raise ValueError(
                f"raw phase-preserving source {name!r} has excessive off-route support: "
                f"relative={off_relative:.6e}, max_entry={off_max:.6e}"
            )
        raw_hashes[name] = hash_array(raw)
        polar_actions[name] = BlockRouteAction(
            name,
            generator.antiunitary,
            reference.fiber_permutation,
            reference.fiber_dimensions,
            tuple(blocks),
            fiber_indices=reference.fiber_indices,
            unitarity_certification_bound=max(unitarity_bounds),
        )
    combined = np.asarray(all_polar_corrections, dtype=np.float64)
    polar_rms = float(np.sqrt(np.mean(np.square(combined))))
    polar_max = float(np.max(combined))
    if polar_rms > config.max_rms_correction:
        raise ValueError(
            "phase-preserving polar RMS correction exceeds the joint budget: "
            f"{polar_rms:.6e} > {config.max_rms_correction:.6e}"
        )
    if polar_max > config.max_route_correction:
        raise ValueError(
            "phase-preserving polar route correction exceeds the joint budget: "
            f"{polar_max:.6e} > {config.max_route_correction:.6e}"
        )
    result = joint_exactify_block_actions(
        polar_actions,
        presentation,
        config=config,
    )
    return result, {
        "version": "phase_preserving_joint_source_v1",
        "status": "certified",
        "policy": "route_polar_then_minimum_joint_relation_correction",
        "raw_matrix_hashes": raw_hashes,
        "polar_correction_rms_by_operation": polar_rms_by_operation,
        "polar_correction_max_by_operation": polar_max_by_operation,
        "polar_correction_rms": polar_rms,
        "polar_correction_max": polar_max,
        "off_route_absolute_fro_by_operation": off_route_absolute_by_operation,
        "off_route_relative_fro_by_operation": off_route_relative_by_operation,
        "off_route_max_entry_by_operation": off_route_max_entry_by_operation,
        "joint_relation_correction_rms": float(result.report["route_correction_rms"]),
        "joint_relation_correction_max": float(result.report["route_correction_max"]),
        "joint_relation_correction_rms_by_operation": dict(
            result.report["route_correction_rms_by_operation"]
        ),
        "joint_relation_correction_max_by_operation": dict(
            result.report["route_correction_max_by_operation"]
        ),
        "joint_artifact_hash": str(result.artifact_metadata["artifact_hash"]),
    }


def _derive_projection_basis_frame(
    matrices: Mapping[str, np.ndarray],
    reports: Mapping[str, Any],
    *,
    operations: Sequence[Mapping[str, Any]],
    sectors: Sequence[Mapping[str, Any]],
    tolerance: float,
    raw_source_matrices: Mapping[str, np.ndarray] | None = None,
) -> tuple[SymmetryAdaptedBasisFrame, dict[str, np.ndarray] | None]:
    """Derive the one frame shared by projection and final symmetry output."""

    joint_payload = reports.get("__joint_exactification__")
    if not isinstance(joint_payload, Mapping):
        return (
            derive_symmetry_adapted_basis_frame(
                matrices,
                operations=operations,
                sectors=sectors,
                tolerance=tolerance,
            ),
            None,
        )
    metadata = joint_payload.get("metadata")
    arrays = joint_payload.get("arrays")
    if not isinstance(metadata, Mapping) or not isinstance(arrays, Mapping):
        raise ValueError("joint exactification handoff lacks metadata or route arrays")
    previous = load_joint_exactification_artifact(metadata, arrays)
    artifact_hash = str(previous.artifact_metadata.get("artifact_hash", ""))
    source_frame_hash = hashlib.sha256(
        ("kp:joint-source-frame:v1\0" + artifact_hash).encode("utf-8")
    ).hexdigest()
    pre_layout = _joint_fiber_sector_layout(
        previous.actions,
        previous.presentation,
        sectors,
        basis_frame_hash=source_frame_hash,
    )
    pre_certificate = certify_joint_structure(
        previous.actions,
        previous.presentation,
        layout=pre_layout,
    )
    identity_gauge = _identity_fiber_gauge(
        previous.actions,
        previous.presentation,
    )
    identity_ready, identity_report = _uniform_algebraic_identity_report(
        previous.actions,
        previous.presentation,
        pre_certificate,
    )
    dimensions = {
        dimension
        for action in previous.actions.values()
        for dimension in action.fiber_dimensions
    }
    components: dict[str, Any] = {}
    candidate_report: dict[str, Any] = {}
    closest_report: dict[str, Any] | None = None
    recovery = None
    # Recovery is also the safe path for an already uniform U(d) action that is
    # merely *near* an algebraic standard.  Restricting this to GENERAL routes
    # would leave literal 1e-13 off-support entries untouched after rollback.
    if not identity_ready and dimensions != {1}:
        recovery = recover_uniform_internal_action(
            previous.actions,
            previous.presentation,
            layout=pre_layout,
        )
        components["uniform_structure_recovery"] = recovery.artifact()
    if identity_ready:
        selected_actions = previous.actions
        selected_gauge = identity_gauge
        post_certificate = pre_certificate
        witness = certify_common_fiber_gauge(
            previous.actions,
            selected_actions,
            selected_gauge,
            previous.presentation,
            source_basis_hash=source_frame_hash,
            target_basis_hash=source_frame_hash,
        )
        transition = compare_joint_structure(
            pre_certificate,
            post_certificate,
            gauge_witness=witness,
        )
        decision = "identity_preserved"
        solver_outcome = CertificationStatus.CERTIFIED.value
        standard_component = {
            "status": "certified",
            "fiber_mode": "identity_preserved",
            "selection_policy": "zero_cost_uniform_algebraic_identity_short_circuit",
            "common_gauge_proximity_objective": 0.0,
            "relation_certification": pre_certificate.artifact()[
                "relation_certificate"
            ],
        }
        canonicality = "canonical_up_to_discrete_stabilizer"
    elif (
        recovery is not None
        and recovery.status is CertificationStatus.CERTIFIED
        and recovery.actions is not None
        and recovery.fiber_gauge is not None
        and recovery.certificate is not None
        and recovery.common_gauge_witness is not None
    ):
        selected_actions = recovery.actions
        selected_gauge = recovery.fiber_gauge
        post_certificate = recovery.certificate
        witness = recovery.common_gauge_witness
        transition = compare_joint_structure(
            pre_certificate,
            post_certificate,
            gauge_witness=witness,
        )
        if (
            transition.status is not CertificationStatus.CERTIFIED
            or not transition.is_monotone
        ):
            raise ValueError(
                "certified uniform recovery failed the projection transaction"
            )
        decision = "uniform_recovery_accepted"
        solver_outcome = CertificationStatus.CERTIFIED.value
        recovery_report = recovery.artifact()
        candidate_report = recovery_report
        sector_standardization = recovery_report.get("sector_standardization")
        standard_component = {
            **(
                dict(sector_standardization)
                if isinstance(sector_standardization, Mapping)
                else {}
            ),
            "status": "certified",
            "fiber_mode": (
                "q_uniform_Ud"
                if recovery_report["selection_policy"]
                == "uniform_sector_reduction_standardization"
                else "q_uniform_recovered"
            ),
            "selection_policy": recovery_report["selection_policy"],
            "common_gauge_proximity_objective": recovery_report[
                "closest_to_source_objective"
            ],
            "relation_certification": recovery_report[
                "relation_certification"
            ],
        }
        canonicality = str(recovery_report["canonicality"])
    else:
        closest = None
        base_actions = previous.actions
        solver_error: str | None = None
        try:
            if dimensions == {1}:
                closest = derive_closest_cyclotomic_u1_gauge(
                    previous.actions,
                    previous.presentation,
                )
                base_actions = closest.actions
                stored = previous.report.get("closest_cyclotomic_u1_gauge")
                if not isinstance(stored, Mapping):
                    raise ProjectionTransactionIntegrityError(
                        "certified U(1) joint artifact lacks closest-cyclotomic provenance"
                    )
                stored_exponents = {
                    str(name): tuple(int(value) for value in values)
                    for name, values in dict(stored.get("root_exponents", {})).items()
                }
                if stored_exponents != dict(closest.root_exponents):
                    raise ProjectionTransactionIntegrityError(
                        "closest cyclotomic branch changed between joint exactification and "
                        "projection-frame construction"
                    )
                closest_report = dict(stored)
                closest_report["live_common_gauge_residual_max"] = float(
                    closest.report["common_gauge_residual_max"]
                )
            standard = derive_standard_generator_fiber_gauge(
                base_actions,
                previous.presentation,
            )
            candidate_report = dict(standard.report)
            candidate_gauge = tuple(
                np.asarray(
                    (
                        np.eye(value.shape[0], dtype=np.complex128)
                        if closest is None
                        else closest.fiber_gauge[fiber]
                    )
                    @ standard.fiber_gauge[fiber],
                    dtype=np.complex128,
                )
                for fiber, value in enumerate(identity_gauge)
            )
            target_frame_hash = _joint_target_frame_hash(
                source_frame_hash,
                candidate_gauge,
            )
            post_layout = _joint_fiber_sector_layout(
                standard.actions,
                previous.presentation,
                sectors,
                basis_frame_hash=target_frame_hash,
            )
            post_certificate = certify_joint_structure(
                standard.actions,
                previous.presentation,
                layout=post_layout,
            )
            witness = certify_common_fiber_gauge(
                previous.actions,
                standard.actions,
                candidate_gauge,
                previous.presentation,
                source_basis_hash=source_frame_hash,
                target_basis_hash=target_frame_hash,
            )
            transition = compare_joint_structure(
                pre_certificate,
                post_certificate,
                gauge_witness=witness,
            )
            candidate_certified = (
                standard.report.get("status") == "certified"
                or (
                    closest is not None
                    and standard.report.get("status") == "not_applicable"
                )
            )
        except (JointExactificationError, ValueError, np.linalg.LinAlgError) as exc:
            solver_error = str(exc)
            candidate_certified = False
            candidate_gauge = identity_gauge
            post_certificate = pre_certificate
            witness = certify_common_fiber_gauge(
                previous.actions,
                previous.actions,
                identity_gauge,
                previous.presentation,
                source_basis_hash=source_frame_hash,
                target_basis_hash=source_frame_hash,
            )
            transition = compare_joint_structure(
                pre_certificate,
                pre_certificate,
                gauge_witness=witness,
            )
            standard = None
            candidate_report = {
                "status": "unknown_or_unsupported",
                "reason": solver_error,
            }
        if (
            candidate_certified
            and transition.status is CertificationStatus.CERTIFIED
            and witness.status is CertificationStatus.CERTIFIED
            and standard is not None
        ):
            selected_actions = standard.actions
            selected_gauge = candidate_gauge
            decision = "candidate_accepted"
            solver_outcome = CertificationStatus.CERTIFIED.value
            standard_component = dict(standard.report)
            if closest_report is not None:
                components["closest_cyclotomic_u1_gauge"] = closest_report
            canonicality = "canonical_up_to_continuous_stabilizer"
        else:
            rejected_candidate_transaction = {
                "post_certificate": post_certificate.artifact(),
                "common_gauge_witness": witness.artifact(),
                "transition": transition.artifact(),
            }
            selected_actions = previous.actions
            selected_gauge = identity_gauge
            if transition.regressions:
                decision = "candidate_rejected_structure_regression"
                solver_outcome = CertificationStatus.CERTIFIED_INFEASIBLE.value
            else:
                decision = "candidate_rejected_unknown_or_unsupported"
                solver_outcome = CertificationStatus.UNKNOWN_OR_UNSUPPORTED.value
            post_certificate = pre_certificate
            witness = certify_common_fiber_gauge(
                previous.actions,
                previous.actions,
                identity_gauge,
                previous.presentation,
                source_basis_hash=source_frame_hash,
                target_basis_hash=source_frame_hash,
            )
            transition = compare_joint_structure(
                pre_certificate,
                post_certificate,
                gauge_witness=witness,
            )
            components["rejected_candidate_transaction"] = (
                rejected_candidate_transaction
            )
            standard_component = {
                "status": "certified",
                "fiber_mode": "identity_rollback",
                "selection_policy": "fail_closed_preserve_pre_exact_representation",
                "common_gauge_proximity_objective": 0.0,
                "candidate_report": candidate_report,
                "relation_certification": pre_certificate.artifact()[
                    "relation_certificate"
                ],
            }
            canonicality = "canonical_up_to_continuous_stabilizer"

    target_selection_transaction = {
        "decision": decision,
        "solver_outcome": solver_outcome,
        "source_joint_artifact_hash": artifact_hash,
        "pre_certificate": pre_certificate.artifact(),
        "post_certificate": post_certificate.artifact(),
        "common_gauge_witness": witness.artifact(),
        "transition": transition.artifact(),
    }
    if raw_source_matrices is not None:
        config_raw = previous.artifact_metadata.get("config")
        if not isinstance(config_raw, Mapping):
            raise ProjectionTransactionIntegrityError(
                "project joint artifact lacks its effective configuration"
            )
        joint_config = _joint_config_from_artifact(config_raw)
        target_reference = selected_actions[
            previous.presentation.generators[0].name
        ]

        def raw_to_target_distance(
            fiber_gauge: Sequence[np.ndarray],
        ) -> dict[str, Any]:
            full = np.eye(
                int(sum(target_reference.fiber_dimensions)),
                dtype=np.complex128,
            )
            for fiber, indices in enumerate(target_reference.fiber_indices):
                full[np.ix_(indices, indices)] = fiber_gauge[fiber]
            by_operation: dict[str, float] = {}
            for generator in previous.presentation.generators:
                transformed = transform_basis_operation(
                    np.asarray(
                        raw_source_matrices[generator.name],
                        dtype=np.complex128,
                    ),
                    full,
                    antiunitary=generator.antiunitary,
                )
                target_dense = materialize_block_route_action(
                    selected_actions[generator.name]
                )
                by_operation[generator.name] = float(
                    np.linalg.norm(transformed - target_dense, ord="fro")
                    / np.sqrt(max(1, transformed.shape[0]))
                )
            return {
                "normalized_fro_by_operation": by_operation,
                "max_normalized_fro": max(by_operation.values(), default=0.0),
            }

        before_alignment_distance = raw_to_target_distance(selected_gauge)
        phase_source, phase_source_report = _phase_preserving_joint_source(
            raw_source_matrices,
            previous.actions,
            previous.presentation,
            config=joint_config,
        )
        phase_source_hash = hashlib.sha256(
            (
                "kp:phase-preserving-joint-source:v1\0"
                + str(phase_source.artifact_metadata["artifact_hash"])
            ).encode("utf-8")
        ).hexdigest()
        phase_layout = _joint_fiber_sector_layout(
            phase_source.actions,
            phase_source.presentation,
            sectors,
            basis_frame_hash=phase_source_hash,
        )
        phase_pre_certificate = certify_joint_structure(
            phase_source.actions,
            phase_source.presentation,
            layout=phase_layout,
        )
        alignment = recover_fixed_target_common_gauge(
            phase_source.actions,
            selected_actions,
            previous.presentation,
            layout=phase_layout,
            reference_fiber_gauge=selected_gauge,
            source_action_absolute_error_bounds=dict(
                phase_source_report[
                    "joint_relation_correction_max_by_operation"
                ]
            ),
        )
        if (
            alignment.status is not CertificationStatus.CERTIFIED
            or alignment.fiber_gauge is None
            or alignment.certificate is None
            or alignment.common_gauge_witness is None
        ):
            raise ProjectionTransactionIntegrityError(
                "raw phase-preserving actions cannot be aligned to the selected "
                f"canonical target: {alignment.artifact().get('reason')}"
            )
        alignment_transition = compare_joint_structure(
            phase_pre_certificate,
            alignment.certificate,
            gauge_witness=alignment.common_gauge_witness,
        )
        if (
            phase_pre_certificate.status is CertificationStatus.CERTIFIED
            and (
                alignment_transition.status is not CertificationStatus.CERTIFIED
                or not alignment_transition.is_monotone
            )
        ):
            raise ProjectionTransactionIntegrityError(
                "raw fixed-target alignment failed its monotone transaction"
            )
        alignment_reference = phase_source.actions[
            phase_source.presentation.generators[0].name
        ]
        alignment_dimension = int(sum(alignment_reference.fiber_dimensions))
        alignment_full_gauge = np.eye(
            alignment_dimension,
            dtype=np.complex128,
        )
        for fiber, indices in enumerate(alignment_reference.fiber_indices):
            alignment_full_gauge[np.ix_(indices, indices)] = (
                alignment.fiber_gauge[fiber]
            )
        parity = {
            generator.name: bool(generator.antiunitary)
            for generator in phase_source.presentation.generators
        }
        transformed_raw = {
            name: transform_basis_operation(
                np.asarray(raw_source_matrices[name], dtype=np.complex128),
                alignment_full_gauge,
                antiunitary=parity[name],
            )
            for name in parity
        }
        direct_alignment_certificate = certify_fixed_target_exactification(
            transformed_raw,
            selected_actions,
            root_order=24,
            max_rms_correction=joint_config.max_rms_correction,
            max_route_correction=joint_config.max_route_correction,
            persisted_frame_hash=None,
            target_artifact_hash=None,
            target_certificate_identity=alignment.certificate.artifact(),
            source_matrix_hashes={
                name: hash_array(raw_source_matrices[name]) for name in parity
            },
            operation_antiunitary=parity,
        )
        after_alignment_distance = raw_to_target_distance(
            alignment.fiber_gauge
        )
        phase_source_artifact = encode_literal_route_source_actions(
            phase_source.actions,
            phase_source.presentation,
            structure_identity=phase_pre_certificate.artifact(),
            joint_config=joint_config,
            root_order=24,
            source_role="phase_preserving_joint_source",
        )
        components["fixed_target_common_gauge_alignment"] = {
            "version": "kp_fixed_target_common_gauge_alignment_v2",
            "status": "certified",
            "policy": (
                "phase_preserving_source_to_fixed_algebraic_target_"
                "closest_to_certified_standard_gauge"
            ),
            "replaced_target_selection_gauge": True,
            "target_selection_transaction": target_selection_transaction,
            "phase_preserving_source": phase_source_report,
            "phase_preserving_source_actions": phase_source_artifact,
            "alignment": alignment.artifact(),
            "source_structure_classifier": {
                "status": phase_pre_certificate.status.value,
                "production_gate": (
                    phase_pre_certificate.status
                    is CertificationStatus.CERTIFIED
                ),
                "reason": (
                    "a certified source classifier must remain monotone; "
                    "an ambiguous intermediate classifier is superseded by "
                    "the exact target plus certified common-gauge witness"
                ),
            },
            "direct_raw_to_target_certificate": direct_alignment_certificate,
            "raw_to_target_correction": {
                "before_alignment": before_alignment_distance,
                "after_alignment": after_alignment_distance,
            },
        }
        selected_gauge = alignment.fiber_gauge
        pre_certificate = phase_pre_certificate
        post_certificate = alignment.certificate
        witness = alignment.common_gauge_witness
        transition = alignment_transition
        source_frame_hash = phase_source_hash
        decision = "fixed_target_raw_alignment_accepted"
        solver_outcome = CertificationStatus.CERTIFIED.value

    reference = previous.actions[previous.presentation.generators[0].name]
    dimension = int(np.asarray(next(iter(matrices.values()))).shape[0])
    full_gauge = np.eye(dimension, dtype=np.complex128)
    for fiber, indices in enumerate(reference.fiber_indices):
        full_gauge[np.ix_(indices, indices)] = selected_gauge[fiber]
    canonical_matrices = {
        name: materialize_block_route_action(action)
        for name, action in selected_actions.items()
    }
    identity_frame = derive_symmetry_adapted_basis_frame(
        {},
        operations=(),
        sectors=sectors,
        tolerance=tolerance,
    )
    components["standard_generator_fiber_gauge"] = standard_component
    components["canonical_target_actions"] = encode_canonical_target_actions(
        selected_actions,
        previous.presentation,
        certificate=post_certificate.artifact(),
        joint_config=previous.artifact_metadata.get("config"),
        root_order=24,
    )
    components["joint_structure_transaction"] = {
        "version": "kp_joint_structure_transaction_v1",
        "decision": decision,
        "solver_outcome": solver_outcome,
        "pre_certificate": pre_certificate.artifact(),
        "post_certificate": post_certificate.artifact(),
        "common_gauge_witness": witness.artifact(),
        "transition": transition.artifact(),
        "identity_short_circuit": identity_report,
        "candidate_report": candidate_report,
        "canonicality": canonicality,
        "residual_stabilizer_dimension": None,
    }
    components["internal_frame"] = {
        "status": "not_applicable",
        "reason": "joint_fibers_are_fixed_by_certified_standard_generator_gauge",
    }
    frame = SymmetryAdaptedBasisFrame(
        status="applied",
        full_unitary=full_gauge,
        sector_frames=identity_frame.sector_frames,
        reason=None,
        components=components,
    )
    return frame, canonical_matrices


def _load_project_artifact_identity(
    project_dir: str | Path,
    *,
    verify_heff: bool = True,
) -> dict[str, Any]:
    return load_projection_artifact_identity(project_dir, verify_heff=verify_heff)


def _required_scalar_text(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> str:
    if key not in payload:
        raise KeyError(f"{context} is missing {key}")
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError(f"{context} field {key} must be scalar, got {value.shape}")
    scalar = value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return str(scalar)


def _load_persisted_projection_basis_handoff(
    project_dir: str | Path,
) -> ProjectionBasisSpec | None:
    """Load the authoritative project basis without running gauge selection.

    The basis and wavefunction packages carry duplicate frame metadata.  Both
    copies must agree byte-for-byte before ``kp symm`` is allowed to project a
    source operation into that basis.
    """

    project_path = Path(project_dir)
    basis_path = project_path / "basis.npz"
    wavefunctions_path = project_path / "wavefunctions.npz"
    heff_path = project_path / "heff.npy"
    try:
        with np.load(basis_path, allow_pickle=False) as discriminator_payload:
            basis_files = frozenset(discriminator_payload.files)
            basis_kind = (
                _required_scalar_text(
                    discriminator_payload,
                    "projection_basis_kind",
                    context=f"KP projection basis handoff {basis_path}",
                )
                if "projection_basis_kind" in discriminator_payload.files
                else "explicit_legacy"
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"projection basis discriminator is invalid: {error}",
        ) from error
    if basis_kind == GammaRoutedBasisSpec.projection_basis_kind:
        try:
            routed = load_gamma_routed_basis_spec(basis_path)
            artifact_identity = _load_project_artifact_identity(project_path)
            require_matching_identity(
                routed.artifact_identity,
                artifact_identity,
                PROJECTION_ARTIFACT_IDENTITY_FIELDS,
                "routed Gamma project basis handoff",
            )
        except GammaRoutingError:
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                f"routed Gamma projection artifact identity failed: {error}",
            ) from error
        companion_heff = np.load(heff_path, allow_pickle=False)
        if not np.array_equal(companion_heff, routed.authoritative_heff):
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "projection/heff.npy differs from routed authoritative_heff",
            )
        with np.load(wavefunctions_path, allow_pickle=False) as wave_payload:
            if "k_indices" not in wave_payload.files:
                raise GammaRoutingError(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    "routed Gamma wavefunctions package is missing k_indices",
                )
            wave_k_indices = tuple(
                int(item) for item in np.asarray(wave_payload["k_indices"]).tolist()
            )
        if wave_k_indices != routed.k_indices:
            raise GammaRoutingError(
                CandidateRejectionReason.HANDOFF_IDENTITY,
                "routed Gamma wavefunction k mapping differs from basis handoff",
            )
        return routed
    if basis_kind != ExplicitLegacyBasisSpec.projection_basis_kind:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            f"unsupported projection_basis_kind: {basis_kind!r}",
        )
    forbidden_routed_fields = sorted(
        basis_files.intersection(GAMMA_ROUTED_ONLY_BASIS_FIELDS)
    )
    if forbidden_routed_fields:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "explicit legacy basis contains routed-only fields: "
            + ", ".join(forbidden_routed_fields),
        )
    artifact_identity = _load_project_artifact_identity(project_path)
    with np.load(basis_path, allow_pickle=True) as basis_payload:
        basis_context = f"KP projection basis handoff {basis_path}"
        if "projection_basis_handoff_version" not in basis_payload.files:
            return None
        basis_version = _required_scalar_text(
            basis_payload,
            "projection_basis_handoff_version",
            context=basis_context,
        )
        basis_gauge_mode = _required_scalar_text(
            basis_payload,
            "gauge_mode",
            context=basis_context,
        )
        basis_frame_json = _required_scalar_text(
            basis_payload,
            "symmetry_adapted_frame_json",
            context=basis_context,
        )
        basis_wavefunctions_hash = _required_scalar_text(
            basis_payload,
            "wavefunctions_hash",
            context=basis_context,
        )
        basis_spin_operator_hash = _required_scalar_text(
            basis_payload,
            "spin_operator_hash",
            context=basis_context,
        )
        if "nlow_state_list" not in basis_payload.files:
            raise KeyError(f"{basis_context} is missing nlow_state_list")
        if "norb_fix_list" not in basis_payload.files:
            raise KeyError(f"{basis_context} is missing norb_fix_list")
        nlow_state_list = np.asarray(
            basis_payload["nlow_state_list"], dtype=object
        ).tolist()
        resolved_norb_fix_list = np.asarray(
            basis_payload["norb_fix_list"], dtype=object
        ).tolist()
    with np.load(wavefunctions_path, allow_pickle=False) as wave_payload:
        wave_context = f"KP projection basis handoff {wavefunctions_path}"
        wave_version = _required_scalar_text(
            wave_payload,
            "projection_basis_handoff_version",
            context=wave_context,
        )
        wave_gauge_mode = _required_scalar_text(
            wave_payload,
            "gauge_mode",
            context=wave_context,
        )
        wave_frame_json = _required_scalar_text(
            wave_payload,
            "symmetry_adapted_frame_json",
            context=wave_context,
        )
        wave_wavefunctions_hash = _required_scalar_text(
            wave_payload,
            "wavefunctions_hash",
            context=wave_context,
        )
        wave_spin_operator_hash = _required_scalar_text(
            wave_payload,
            "spin_operator_hash",
            context=wave_context,
        )
        if "wavefunctions" not in wave_payload.files:
            raise KeyError(f"{wave_context} is missing wavefunctions")
        actual_wavefunctions_hash = hash_array(wave_payload["wavefunctions"])
        has_spin_operator = "spin_operator" in wave_payload.files
        actual_spin_operator_hash = (
            hash_array(wave_payload["spin_operator"])
            if has_spin_operator
            else "absent"
        )
    if basis_version != PROJECTION_BASIS_HANDOFF_VERSION:
        raise ValueError(
            "unsupported KP projection basis handoff version: "
            f"{basis_version!r} != {PROJECTION_BASIS_HANDOFF_VERSION!r}"
        )
    if wave_version != basis_version:
        raise ValueError(
            "KP projection basis/wavefunction handoff version mismatch: "
            f"{basis_version!r} != {wave_version!r}"
        )
    if not basis_gauge_mode or wave_gauge_mode != basis_gauge_mode:
        raise ValueError(
            "KP projection basis/wavefunction gauge mode mismatch: "
            f"{basis_gauge_mode!r} != {wave_gauge_mode!r}"
        )
    if wave_frame_json != basis_frame_json:
        raise ValueError(
            "KP projection basis/wavefunction symmetry frame mismatch"
        )
    if wave_wavefunctions_hash != basis_wavefunctions_hash:
        raise ValueError(
            "KP projection basis/wavefunction wavefunctions_hash mismatch: "
            f"{basis_wavefunctions_hash!r} != {wave_wavefunctions_hash!r}"
        )
    if actual_wavefunctions_hash != basis_wavefunctions_hash:
        raise ValueError(
            "KP projection wavefunctions content hash mismatch: "
            f"{basis_wavefunctions_hash!r} != {actual_wavefunctions_hash!r}"
        )
    if wave_spin_operator_hash != basis_spin_operator_hash:
        raise ValueError(
            "KP projection basis/wavefunction spin_operator_hash mismatch: "
            f"{basis_spin_operator_hash!r} != {wave_spin_operator_hash!r}"
        )
    if actual_spin_operator_hash != basis_spin_operator_hash:
        raise ValueError(
            "KP projection spin-operator content hash mismatch: "
            f"{basis_spin_operator_hash!r} != {actual_spin_operator_hash!r}"
        )

    frame_artifact: Mapping[str, Any] | None = None
    frame: SymmetryAdaptedBasisFrame | None = None
    if basis_frame_json:
        decoded = json.loads(basis_frame_json)
        if not isinstance(decoded, Mapping):
            raise ValueError("persisted KP projection symmetry frame is not a mapping")
        frame_artifact = dict(decoded)
        frame = SymmetryAdaptedBasisFrame.from_artifact(frame_artifact)

    heff = np.load(heff_path, mmap_mode="r", allow_pickle=False)
    if heff.ndim != 3 or heff.shape[-2] != heff.shape[-1]:
        raise ValueError(
            "projection/heff.npy must have shape (Nk, model_dim, model_dim), "
            f"got {heff.shape}"
        )
    model_dim = int(heff.shape[-1])
    if frame is not None and frame.full_unitary.shape != (model_dim, model_dim):
        raise ValueError(
            "persisted project frame/model dimension mismatch: "
            f"{frame.full_unitary.shape} != {(model_dim, model_dim)}"
        )
    return ExplicitLegacyBasisSpec(
        artifact_identity=dict(artifact_identity),
        nlow_state_list=list(nlow_state_list),
        resolved_norb_fix_list=list(resolved_norb_fix_list),
        gauge_mode=basis_gauge_mode,
        frame_artifact=frame_artifact,
        frame=frame,
        model_dim=model_dim,
    )


def _validate_symmetry_project_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    require_matching_identity(
        expected,
        actual,
        _PROJECT_BASIS_IDENTITY_FIELDS,
        "kp symm projection basis",
    )
    resolved = require_identity_fields(
        actual,
        _PROJECT_ARTIFACT_IDENTITY_FIELDS,
        "kp symm projection artifacts",
    )
    return resolved


def _energy_scale_from_material(material: Mapping[str, Any]) -> float:
    unit = material.get("energy_unit", "eV")
    text = str(unit).strip().lower()
    if text in {"ev", "electronvolt", "electron_volt"}:
        return 1.0
    if text in {"hartree", "ha"}:
        return HARTREE_TO_EV
    raise ValueError(f"material.energy_unit must be 'eV' or 'Hartree', got {unit!r}")


def _load_hamk_with_energy_unit(path: str, material: Mapping[str, Any], *, mmap_mode: str | None = "r") -> np.ndarray:
    hamk = load_hamk(path, mmap_mode=mmap_mode)
    scale = _energy_scale_from_material(material)
    if scale == 1.0:
        return hamk
    return hamk * scale


def _validate_operation_label(label: str) -> str:
    text = str(label)
    if text in M_EFFECTIVE_OPERATION_ALIASES:
        text = str(M_EFFECTIVE_OPERATION_ALIASES[text]["canonical"])
    if text not in _SUPPORTED_OPERATION_LABELS:
        raise ValueError(
            f"Unsupported symm operation {text!r}; use a standard operation family and explicit action metadata"
        )
    return text


def _canonical_internal_operation_name(operation: str) -> str:
    if operation in {"C3", "C3z"}:
        return "C3z"
    if operation in {"T", "TR"}:
        return "TR"
    return str(operation)


def _source_manifest_operation_name(operation: str) -> str:
    if operation in {"C3", "C3z"}:
        return operation
    if operation in {"T", "TR"}:
        return "TR"
    return str(operation)


def _output_operation_name(requested: str, validated: str) -> str:
    if validated in {"T", "TR"}:
        return "TR"
    return str(requested)


def _manifest_operation_lookup_names(operation: str) -> tuple[str, ...]:
    if operation == "C3z":
        return ("C3z", "C3")
    if operation == "T":
        return ("T", "TR")
    if operation == "TR":
        return ("TR", "T")
    return (operation,)


def _valley_family(valley: str) -> str:
    text = str(valley).lower()
    if text.startswith("k"):
        return "K"
    if text.startswith("m"):
        return "M"
    if text.startswith("g"):
        return "Gamma"
    return str(valley)


def _spin_convention_for_exactification(spin: str, n_orb: tuple[int, int], *, valley: str) -> str:
    spin_text = str(spin).lower()
    if str(spin).lower() == "all" or max(int(n_orb[0]), int(n_orb[1])) > 1:
        return "spinful"
    if _valley_family(valley) == "K" and spin_text in {"up", "down"}:
        return f"spin_{spin_text}_projected"
    return "spinless_effective"


def _operation_power_relation(operation: str, *, spin_convention: str) -> dict[str, Any]:
    name = _canonical_internal_operation_name(operation)
    if name == "C3z":
        power, phase = 3, -1
    elif name == "C2":
        power, phase = 2, (-1 if spin_convention == "spinful" else 1)
    elif name == "C2T":
        power, phase = 2, 1
    elif name == "TR":
        power, phase = 2, (-1 if spin_convention == "spinful" else 1)
    else:
        raise ValueError(f"Unsupported operation family for group relation: {operation!r}")
    return {
        "type": "power",
        "name": f"{name}^{power}",
        "operation": name,
        "power": power,
        "phase": phase,
        "source": "kp_symm_manifest",
    }


def _kp_symm_exactification_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = {
        "support_source": "geometry",
        "discover_action_candidates": True,
        "accept_support_resolved_action": True,
        "require_group_relations": True,
        "phase_classes": "global",
        "reject_if_off_support_rel_gt": 1.0e-5,
        "reject_if_amplitude_deviation_gt": 0.05,
        "inferred": True,
        "source": "kp_symm",
    }
    if overrides:
        allowed = {
            "accept_support_resolved_action",
            "action_candidates",
            "forbid_inferred_q_offset",
            "joint_exactification",
            "reject_if_off_support_rel_gt",
            "reject_if_amplitude_deviation_gt",
            "monomial_cleanup_tol",
            "monomial_root_order_max",
            "operations",
            "phase_classes",
            "require_explicit_action_candidates",
            "strict",
            "support_mode",
        }
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"Unsupported symm.exactification keys: {unknown}")
        config.update({key: overrides[key] for key in allowed if key in overrides})
    return config


def _normalize_nlow_state_list(project_cfg: dict[str, Any]) -> list[list[int]]:
    active = project_cfg.get("active_indices")
    if active is not None:
        if isinstance(active, str):
            return [[int(part.strip()) for part in active.split(",") if part.strip()]]
        return [[int(value) for value in active]]
    nlow_state_list = project_cfg.get("nlow_state_list", [])
    if nlow_state_list and not isinstance(nlow_state_list[0], (list, tuple)):
        return [[int(x) for x in nlow_state_list]]
    return [[int(x) for x in layer] for layer in nlow_state_list]


def _source_group_nlow_state_list(
    nlow_state_list: Sequence[Sequence[int]],
    num_layer_list: Sequence[int] | None,
) -> list[list[int]]:
    rows = [[int(band) for band in row] for row in nlow_state_list]
    if num_layer_list is None:
        return rows
    group_layers = [int(n) for n in num_layer_list]
    total_layers = int(sum(group_layers))
    if len(rows) != total_layers:
        raise ValueError(
            f"project.nlow_state_list must have {total_layers} physical-layer rows "
            f"(sum(material.num_layer_list)); got {len(rows)}. "
            "Use [] for layers that do not contribute."
        )
    grouped: list[list[int]] = []
    offset = 0
    for n_layers in group_layers:
        bands: list[int] = []
        for local in range(int(n_layers)):
            bands.extend(rows[offset + local])
        grouped.append(bands)
        offset += int(n_layers)
    return grouped


def _validate_project_layer_lists(
    nlow_state_list: Sequence[Sequence[int]],
    norb_fix_list: Sequence[Any],
    *,
    num_layer_list: Sequence[int] | None,
    context: str = "project",
) -> None:
    nlow_rows = [[int(band) for band in row] for row in nlow_state_list]
    if not nlow_rows:
        raise ValueError(f"{context}.nlow_state_list must not be empty")

    if num_layer_list is not None:
        total_layers = int(sum(int(n) for n in num_layer_list))
        if len(nlow_rows) != total_layers:
            raise ValueError(
                f"{context}.nlow_state_list must have {total_layers} physical-layer rows "
                f"(sum(material.num_layer_list)); got {len(nlow_rows)}. "
                "Use [] for layers that do not contribute."
            )

    if len(norb_fix_list) != len(nlow_rows):
        raise ValueError(
            f"{context}.norb_fix_list must have the same number of rows as "
            f"{context}.nlow_state_list; got {len(norb_fix_list)} vs {len(nlow_rows)}"
        )

    for layer, bands in enumerate(nlow_rows):
        ref_entries = _layer_reference_entries(norb_fix_list[layer], len(bands))
        if len(ref_entries) != len(bands):
            raise ValueError(
                f"{context}.nlow_state_list layer {layer} has {len(bands)} bands but "
                f"{context}.norb_fix_list layer {layer} has {len(ref_entries)} references"
            )


def _downfold_method(project_cfg: dict[str, Any]) -> str:
    return str(project_cfg.get("downfold_method", project_cfg.get("method", "linearized_lowdin"))).lower()


def _e_ref(project_cfg: dict[str, Any]) -> float | None:
    value = project_cfg.get("e_ref", project_cfg.get("E_ref"))
    return None if value is None else float(value)


def _infer_symmetry_operations_from_manifest(manifest: Mapping[str, Any], valley: str) -> list[str]:
    rows: list[Mapping[str, Any]] = []
    matrices = manifest.get("matrices")
    if isinstance(matrices, list):
        rows.extend(row for row in matrices if isinstance(row, Mapping))
    operations = manifest.get("operations")
    if isinstance(operations, Mapping):
        valley_ops = operations.get(str(valley))
        if isinstance(valley_ops, Mapping):
            nested = valley_ops.get("operations", valley_ops)
            if isinstance(nested, Mapping):
                for name, value in nested.items():
                    row = dict(value) if isinstance(value, Mapping) else {}
                    row.setdefault("operation", str(name))
                    row.setdefault("source_valley", str(valley))
                    row.setdefault("target_valley", str(valley))
                    rows.append(row)
        else:
            for name, value in operations.items():
                row = dict(value) if isinstance(value, Mapping) else {}
                row.setdefault("operation", str(name))
                rows.append(row)
    elif isinstance(operations, list):
        rows.extend(row for row in operations if isinstance(row, Mapping))

    selected: set[str] = set()
    for row in rows:
        source_valley = str(row.get("source_valley", row.get("valley_label", row.get("valley", valley))))
        target_valley = str(row.get("target_valley", source_valley))
        if source_valley != str(valley) or target_valley != str(valley):
            continue
        operation = str(row.get("operation", row.get("name", "")))
        canonical = _canonical_internal_operation_name(operation)
        if canonical in {"", "E"} or "^" in operation:
            continue
        if row.get("supported") is False:
            continue
        if not any(row.get(key) for key in ("raw_h_operator_file", "matrix_file", "filename", "file", "path")):
            continue
        try:
            _validate_operation_label(canonical)
        except ValueError:
            continue
        selected.add(canonical)

    preferred = ["TR", "C3z", "C2", "C2T"]
    ordered = [name for name in preferred if name in selected]
    ordered.extend(sorted(selected - set(ordered)))
    if not ordered:
        raise ValueError(f"Could not infer symm.operations for valley {valley!r} from TAPW symmetry manifest")
    return ordered


def _default_exactification_overrides_for_valley(
    valley: str,
    *,
    symmetry_tolerance: float | None = None,
) -> dict[str, Any]:
    family = _valley_family(str(valley))
    if family == "K":
        defaults: dict[str, Any] = {"reject_if_off_support_rel_gt": 5.0e-3}
    elif family == "Gamma":
        defaults = {
            "reject_if_off_support_rel_gt": 1.0e-3,
            "operations": {
                "C3z": {"support_mode": "auto", "algebraic_template": "auto"},
            },
        }
    else:
        defaults = {}
    if symmetry_tolerance is not None and "reject_if_off_support_rel_gt" in defaults:
        tolerance = float(symmetry_tolerance)
        if tolerance > 0.0:
            defaults["reject_if_off_support_rel_gt"] = max(
                float(defaults["reject_if_off_support_rel_gt"]),
                tolerance,
            )
    return defaults


def _merged_exactification_overrides(
    valley: str,
    user_overrides: Any,
    *,
    symmetry_tolerance: float | None = None,
    operation_actions: Any = None,
) -> dict[str, Any]:
    defaults = _default_exactification_overrides_for_valley(
        valley,
        symmetry_tolerance=symmetry_tolerance,
    )
    merged = dict(defaults)
    if user_overrides is not None:
        if not isinstance(user_overrides, Mapping):
            raise ValueError("symm.exactification must be a mapping when provided")
        for key, value in user_overrides.items():
            if key == "operations" and isinstance(value, Mapping) and isinstance(merged.get("operations"), Mapping):
                op_merged = {str(name): dict(spec) for name, spec in merged["operations"].items()}
                for op_name, op_spec in value.items():
                    op_merged[str(op_name)] = dict(op_spec) if isinstance(op_spec, Mapping) else op_spec
                merged[key] = op_merged
            else:
                merged[str(key)] = value

    if operation_actions is not None:
        if not isinstance(operation_actions, Mapping):
            raise ValueError("symm.operation_actions must be a mapping when provided")
        operation_specs = {
            str(name): dict(spec) if isinstance(spec, Mapping) else spec
            for name, spec in (
                merged.get("operations", {}).items()
                if isinstance(merged.get("operations"), Mapping)
                else ()
            )
        }
        for raw_name, raw_action in operation_actions.items():
            name = _canonical_internal_operation_name(str(raw_name))
            _validate_operation_label(name)
            if not isinstance(raw_action, Mapping):
                raise ValueError(f"symm.operation_actions.{raw_name} must be a mapping")
            action = dict(raw_action)
            current_raw = operation_specs.get(name, {})
            if not isinstance(current_raw, Mapping):
                raise ValueError(f"symm.exactification.operations.{name} must be a mapping")
            current = dict(current_raw)
            declared = [action]
            if "action_candidates" in current and current["action_candidates"] != declared:
                raise ValueError(
                    f"symm.operation_actions.{name} conflicts with "
                    f"symm.exactification.operations.{name}.action_candidates"
                )
            current["action_candidates"] = declared
            operation_specs[name] = current
        merged["operations"] = operation_specs
    return merged


def _infer_orbitals_per_layer(hamk2d: np.ndarray, q_count: int, num_layers: int) -> int:
    base = int(hamk2d.shape[0]) // 2
    divisor = int(q_count) * int(num_layers)
    if divisor <= 0 or base % divisor != 0:
        raise ValueError(f"Cannot infer orbitals: base={base}, q_count={q_count}, num_layers={num_layers}")
    return base // divisor


def _num_layer_list_from_material(material: Mapping[str, Any]) -> list[int]:
    raw = material.get("num_layer_list")
    if raw is not None:
        layers = [int(x) for x in raw]
        if not layers or any(x <= 0 for x in layers):
            raise ValueError(f"material.num_layer_list must contain positive integers, got {raw!r}")
        return layers
    return [1 for _ in range(int(material.get("num_layers", 2)))]


def _orbital_layout_from_material(
    material: Mapping[str, Any],
    hamk2d: np.ndarray,
    q_count: int,
) -> tuple[list[int], int, list[list[int]]]:
    num_layer_list = _num_layer_list_from_material(material)
    total_layers = sum(num_layer_list)
    raw = material.get("num_orb_per_layer")
    if raw:
        vals = [int(x) for x in raw]
        if len(vals) == 1:
            orb0 = vals[0]
            return num_layer_list, orb0, [[orb0 for _ in range(n)] for n in num_layer_list]
        if len(vals) == len(num_layer_list):
            if len(set(vals)) != 1:
                raise ValueError("This release requires equal orbital counts for each physical layer.")
            orb0 = vals[0]
            return num_layer_list, orb0, [[vals[i] for _ in range(n)] for i, n in enumerate(num_layer_list)]
        if len(vals) == total_layers:
            if len(set(vals)) != 1:
                raise ValueError("This release requires equal orbital counts for each physical layer.")
            orb0 = vals[0]
            out: list[list[int]] = []
            pos = 0
            for n in num_layer_list:
                out.append(vals[pos:pos + n])
                pos += n
            return num_layer_list, orb0, out
        raise ValueError(
            "material.num_orb_per_layer must have length 1, len(num_layer_list), "
            f"or sum(num_layer_list); got {len(vals)} values for {num_layer_list}"
        )
    orb0 = _infer_orbitals_per_layer(hamk2d, q_count, total_layers)
    return num_layer_list, orb0, [[orb0 for _ in range(n)] for n in num_layer_list]


def _spin_slice_hamk(hamk2d: np.ndarray, spin: str) -> np.ndarray:
    spin_lower = str(spin).lower()
    if spin_lower == "all":
        return np.asarray(hamk2d, dtype=np.complex128)
    half = hamk2d.shape[0] // 2
    if spin_lower == "up":
        return np.asarray(hamk2d[:half, :half], dtype=np.complex128)
    if spin_lower == "down":
        return np.asarray(hamk2d[half:, half:], dtype=np.complex128)
    raise ValueError(f"Unsupported spin value: {spin!r}")


def _spin_label_for_sliced_block(spin: str) -> str:
    return "all" if str(spin).lower() == "all" else "up"


def _is_sparse(matrix: Any) -> bool:
    return _sparse is not None and _sparse.issparse(matrix)


def _matrix_norm(matrix: Any) -> float:
    if _is_sparse(matrix):
        return float(_sparse.linalg.norm(matrix))
    return float(np.linalg.norm(matrix))


def _infer_spin_route_from_action_matrix(
    matrix: Any,
    *,
    source_spin: str,
    ambiguity_tolerance: float = 1.0e-8,
) -> tuple[str | None, dict[str, Any]]:
    """Infer whether a full-spin action stays in or leaves one spin sector."""

    if getattr(matrix, "ndim", None) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"spin-route inference requires a square matrix, got {matrix.shape}")
    if int(matrix.shape[0]) % 2:
        raise ValueError(
            "spin-route inference requires an even full-spin dimension, "
            f"got {matrix.shape[0]}"
        )
    source = str(source_spin).strip().lower()
    if source not in {"up", "down"}:
        raise ValueError(f"spin-route inference requires source_spin up/down, got {source_spin!r}")
    tolerance = float(ambiguity_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("spin-route ambiguity_tolerance must be finite and non-negative")

    half = int(matrix.shape[0]) // 2
    slices = {"up": slice(0, half), "down": slice(half, 2 * half)}
    block_norms = {
        f"{target}<-{candidate_source}": _matrix_norm(
            matrix[slices[target], slices[candidate_source]]
        )
        for candidate_source in ("up", "down")
        for target in ("up", "down")
    }
    normalized_support: dict[str, float] = {}
    for candidate_source in ("up", "down"):
        column_norm = math.sqrt(
            sum(
                block_norms[f"{target}<-{candidate_source}"] ** 2
                for target in ("up", "down")
            )
        )
        for target in ("up", "down"):
            key = f"{target}<-{candidate_source}"
            normalized_support[key] = (
                0.0 if column_norm == 0.0 else block_norms[key] / column_norm
            )

    source_support = {
        target: normalized_support[f"{target}<-{source}"]
        for target in ("up", "down")
    }
    selected_target = max(source_support, key=lambda target: source_support[target])
    dominant = float(source_support[selected_target])
    secondary = float(source_support["down" if selected_target == "up" else "up"])
    if dominant == 0.0:
        raise ValueError(f"spin action has zero support from source spin {source}")
    if secondary > tolerance * dominant:
        raise ValueError(
            "ambiguous spin target for source spin "
            f"{source}: normalized support={source_support}"
        )
    route = None if selected_target == source else f"{source}_to_{selected_target}"
    return route, {
        "source_spin": source,
        "selected_target_spin": selected_target,
        "route": route,
        "ambiguity_tolerance": tolerance,
        "block_norms": block_norms,
        "normalized_support": normalized_support,
    }


def _combine_spin_routes(routes: Mapping[str, str | None]) -> str | None:
    """Require one source/target spin Hilbert-space route for a run context."""

    unique = set(routes.values())
    if not unique:
        return None
    if len(unique) != 1:
        rendered = {str(name): route for name, route in sorted(routes.items())}
        raise ValueError(f"symmetry operations require incompatible spin routes: {rendered}")
    return next(iter(unique))


def _as_dense(matrix: Any) -> np.ndarray:
    if _is_sparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.complex128)
    return np.asarray(matrix, dtype=np.complex128)


def _fro_relative(lhs: Any, rhs: Any, denominator: Any) -> float:
    denom = _matrix_norm(denominator)
    if denom == 0.0:
        denom = 1.0
    return float(_matrix_norm(lhs - rhs) / denom)


def _load_matrix(path: Path) -> np.ndarray:
    path, matrix_key = _split_packed_matrix_selector(path)
    if not path.exists():
        raise FileNotFoundError(f"Representation file missing: {path}")
    if matrix_key is not None:
        return _load_packed_matrix(path, matrix_key)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.complex128)
    if path.suffix == ".npz":
        try:
            import scipy.sparse

            return scipy.sparse.load_npz(path).tocsr()
        except Exception:
            data = np.load(path)
            for key in ("matrix", "D", "representation", "arr_0"):
                if key in data.files:
                    return np.asarray(data[key], dtype=np.complex128)
            raise ValueError(f"Cannot find matrix key in {path}; keys={data.files}")
    raise ValueError(f"Unsupported representation file extension: {path}")


def _split_packed_matrix_selector(path: Path) -> tuple[Path, str | None]:
    text = str(path)
    marker = ".npz:"
    index = text.rfind(marker)
    if index < 0:
        return path, None
    return Path(text[: index + len(".npz")]), text[index + len(marker) :]


def _load_packed_matrix(path: Path, key: str) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if key in payload.files:
        return np.asarray(payload[key], dtype=np.complex128)
    csr_keys = (f"{key}_data", f"{key}_indices", f"{key}_indptr", f"{key}_shape")
    if all(name in payload.files for name in csr_keys):
        if _sparse is None:
            raise RuntimeError(f"scipy is required to load sparse packed matrix {path}:{key}")
        return _sparse.csr_matrix(
            (
                np.asarray(payload[csr_keys[0]], dtype=np.complex128),
                np.asarray(payload[csr_keys[1]], dtype=np.int64),
                np.asarray(payload[csr_keys[2]], dtype=np.int64),
            ),
            shape=tuple(int(value) for value in payload[csr_keys[3]]),
        )
    raise ValueError(f"Cannot find packed matrix {key!r} in {path}; keys={payload.files}")


def _slice_representation_for_spin(matrix: np.ndarray, spin: str, target_dim: int) -> RepresentationData:
    if matrix.shape == (target_dim, target_dim):
        matrix_out = matrix.tocsr() if _is_sparse(matrix) else np.asarray(matrix, dtype=np.complex128)
        return RepresentationData(matrix=matrix_out, from_full_spinful=False, spin_leakage=None)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Representation matrix must be square, got shape={matrix.shape}")
    spin_lower = str(spin).lower()
    if spin_lower == "all":
        raise ValueError(f"D dimension {matrix.shape} does not match spin=all U dimension {target_dim}")
    if matrix.shape != (2 * target_dim, 2 * target_dim):
        raise ValueError(f"D dimension {matrix.shape} cannot be sliced to U dimension {target_dim}")
    if spin_lower == "up":
        row = slice(0, target_dim)
        other = slice(target_dim, 2 * target_dim)
    elif spin_lower == "down":
        row = slice(target_dim, 2 * target_dim)
        other = slice(0, target_dim)
    else:
        raise ValueError(f"Unsupported spin value: {spin!r}")
    leakage_blocks = [matrix[row, other], matrix[other, row]]
    leakage = float(np.sqrt(sum(float(_matrix_norm(block) ** 2) for block in leakage_blocks)) / np.sqrt(target_dim))
    matrix_out = matrix[row, row].tocsr() if _is_sparse(matrix) else np.asarray(matrix[row, row], dtype=np.complex128)
    return RepresentationData(matrix=matrix_out, from_full_spinful=True, spin_leakage=leakage)


def _operation_entry(manifest: dict[str, Any], valley: str, operation: str) -> dict[str, Any]:
    operation_names = _manifest_operation_lookup_names(operation)

    def _matches(entry: dict[str, Any]) -> bool:
        entry_operation = str(entry.get("operation", entry.get("name", "")))
        entry_valley = str(entry.get("valley_label", entry.get("valley", valley)))
        return entry_operation in operation_names and entry_valley == valley

    operations = manifest.get("operations")
    if isinstance(operations, dict):
        if valley in operations:
            valley_ops = operations[valley]
            if isinstance(valley_ops, dict):
                for name in operation_names:
                    if name in valley_ops:
                        entry = valley_ops[name]
                        return dict(entry) if isinstance(entry, dict) else {"filename": entry}
                if isinstance(valley_ops.get("operations"), dict):
                    for name in operation_names:
                        entry = valley_ops["operations"].get(name)
                        if entry is not None:
                            return dict(entry) if isinstance(entry, dict) else {"filename": entry}
        for name in operation_names:
            if name in operations:
                entry = operations[name]
                return dict(entry) if isinstance(entry, dict) else {"filename": entry}
    if isinstance(operations, list):
        for entry in operations:
            if not isinstance(entry, dict):
                continue
            if _matches(entry):
                return dict(entry)
    matrices = manifest.get("matrices")
    if isinstance(matrices, list):
        for entry in matrices:
            if isinstance(entry, dict) and _matches(entry):
                return dict(entry)
    valleys = manifest.get("valleys")
    if isinstance(valleys, dict) and valley in valleys:
        valley_ops = valleys[valley].get("operations", {}) if isinstance(valleys[valley], dict) else {}
        for name in operation_names:
            if name in valley_ops:
                entry = valley_ops[name]
                return dict(entry) if isinstance(entry, dict) else {"filename": entry}
    raise KeyError(f"manifest missing operation {valley}/{operation}")


def _entry_filename(entry: dict[str, Any], valley: str, operation: str) -> str:
    for key in ("filename", "file", "path", "representation_filename"):
        if entry.get(key):
            return str(entry[key])
    return f"{valley}/{operation}.npz"


def _optional_representation_filename(entry: dict[str, Any]) -> str | None:
    return _optional_entry_filename(entry, "filename", "file", "path", "representation_filename")


def _operation_action_metadata(entry: dict[str, Any], operation: str, antiunitary: bool, *, strict: bool = True) -> dict[str, Any]:
    if isinstance(entry.get("k_map"), dict):
        k_map = dict(entry["k_map"])
    elif strict:
        raise ValueError(f"Operation {operation!r} requires explicit k_map metadata in the TAPW symmetry manifest")
    elif entry.get("axis_deg") is not None:
        k_map = {"type": "reflection", "axis_deg": float(entry["axis_deg"])}
    elif operation in {"C3", "C3z"}:
        k_map = {"type": "rotation", "angle_deg": 120.0}
    elif operation in {"T", "TR"}:
        k_map = {"type": "negation"}
    else:
        raise ValueError(f"Operation {operation!r} requires explicit k_map metadata in the TAPW symmetry manifest")
    if strict and "q_map" not in entry:
        raise ValueError(f"Operation {operation!r} requires explicit q_map metadata in the TAPW symmetry manifest")
    q_map_inferred = "q_map" not in entry
    q_map_raw = entry.get("q_map", k_map)
    if "sector_map" not in entry or entry.get("sector_map") == "auto":
        raise ValueError(f"Operation {operation!r} requires explicit sector_map metadata in the TAPW symmetry manifest")
    out = {
        "k_map": k_map,
        "q_map": dict(q_map_raw) if isinstance(q_map_raw, dict) else q_map_raw,
        "sector_map": entry["sector_map"],
        "sector_map_source": "manifest",
        "spin_map": entry.get("spin_map", "from_kp_symm_output"),
        "valley_map": entry.get("valley_map", "identity"),
        "antiunitary": bool(antiunitary),
    }
    if q_map_inferred:
        out["q_map_inferred_from_k_map"] = True
    return out


def _rotation_matrix_2d(angle_deg: float) -> list[list[float]]:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return [[c, -s], [s, c]]


def _frame_metadata(*, rotation_deg: float, inference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rotation = _rotation_matrix_2d(rotation_deg)
    metadata = {
        "source": "tapw_q_lists",
        "model": "continuum_model_q_basis",
        "k_transform": {
            "formula": "k_model = R(rotation_deg) @ k_source",
            "rotation_deg": float(rotation_deg),
            "linear_matrix": rotation,
        },
        "q_transform": {
            "formula": "q_model = R(rotation_deg) @ (layer_mean - q_source)",
            "rotation_deg": float(rotation_deg),
            "center": "layer_mean",
            "linear_matrix": [[-value for value in row] for row in rotation],
        },
    }
    if inference is not None:
        metadata["inference"] = dict(inference)
    return metadata


def _model_q_sets(q1: np.ndarray, q2: np.ndarray, *, rotation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    q1_arr = np.asarray(q1, dtype=float)
    q2_arr = np.asarray(q2, dtype=float)
    theta = np.deg2rad(float(rotation_deg))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    center1 = np.mean(q1_arr, axis=0)
    center2 = np.mean(q2_arr, axis=0)
    return (center1 - q1_arr) @ rotation.T, (center2 - q2_arr) @ rotation.T


def _angle_deg(vector: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(float(vector[1]), float(vector[0]))))


def _normalize_angle_360(angle_deg: float) -> float:
    value = float(angle_deg) % 360.0
    return 0.0 if abs(value) < 1.0e-10 or abs(value - 360.0) < 1.0e-10 else value


def _infer_model_frame_rotation(
    q1: np.ndarray,
    q2: np.ndarray,
    operation_entries: Mapping[str, Mapping[str, Any]],
    *,
    target_reflection_axis_deg: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    for operation_name in ("C2T", "C2"):
        entry = operation_entries.get(operation_name)
        if not isinstance(entry, Mapping):
            continue
        for map_key in ("q_map", "k_map"):
            action_map = entry.get(map_key)
            if not isinstance(action_map, Mapping):
                continue
            if str(action_map.get("type", "")).lower() != "reflection" or "axis_deg" not in action_map:
                continue
            source_axis = float(action_map["axis_deg"])
            rotation_deg = _normalize_angle_360(float(target_reflection_axis_deg) - source_axis)
            return rotation_deg, {
                "source": "reflection_axis",
                "operation": operation_name,
                "action_map": map_key,
                "source_axis_deg": source_axis,
                "target_model_axis_deg": float(target_reflection_axis_deg),
                "formula": "rotation_deg = target_model_axis_deg - source_axis_deg",
            }

    q0_1, q0_2 = _model_q_sets(q1, q2, rotation_deg=0.0)
    candidates = bM_candidates_from_q_distances(q0_1, q0_2)
    if candidates:
        bM1, bM2 = canonical_bM_pair_from_candidates(candidates, angle_deg=60.0)
        bM1_angle = _angle_deg(bM1)
        rotation_deg = _normalize_angle_360(-bM1_angle)
        return rotation_deg, {
            "source": "q_lattice_bM1",
            "source_bM1": bM1.tolist(),
            "source_bM2": bM2.tolist(),
            "source_bM1_angle_deg": bM1_angle,
            "target_bM1_angle_deg": 0.0,
            "formula": "rotation_deg = -angle(canonical_bM1(center - q_source))",
        }

    return 0.0, {
        "source": "degenerate_q_lattice",
        "reason": "No nonzero Q differences and no reflection action were available",
        "formula": "rotation_deg = 0 for degenerate diagnostic inputs",
    }


def _model_frame_map(raw: Any, *, rotation_deg: float) -> Any:
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    if bool(out.get("in_model_frame", False)):
        return out
    if str(out.get("type", "")).lower() == "reflection" and "axis_deg" in out:
        out["axis_deg"] = float(out["axis_deg"]) + float(rotation_deg)
    out["in_model_frame"] = True
    return out


def normalize_reflection_axis_deg(axis_deg: float) -> float:
    return float(axis_deg) % 180.0


def reflection_axis_equiv(lhs: float, rhs: float, *, tol: float = 1.0e-8) -> bool:
    delta = (normalize_reflection_axis_deg(lhs) - normalize_reflection_axis_deg(rhs) + 90.0) % 180.0 - 90.0
    return abs(delta) <= float(tol)


def _conjugate_action_to_model_frame(source_action: dict[str, Any], *, rotation_deg: float) -> dict[str, Any]:
    model_action = dict(source_action)
    model_action["k_map"] = _model_frame_map(source_action.get("k_map"), rotation_deg=rotation_deg)
    model_action["q_map"] = _model_frame_map(source_action.get("q_map", source_action.get("k_map")), rotation_deg=rotation_deg)
    model_action["action_source"] = "derived_by_frame_conjugation"
    model_action["derivation"] = {
        "formula": "A_model = R(rotation_deg) @ A_source @ R(-rotation_deg)",
        "rotation_deg": float(rotation_deg),
        "reflection_axis_convention": "mirror_axis_deg",
        "q_transform": "q_model = R(rotation_deg) @ (sector_center - q_source)",
        "q_affine_offset_ignored_for_linear_action": True,
    }
    return model_action


def _model_frame_action_metadata(source_action: dict[str, Any], *, rotation_deg: float) -> dict[str, Any]:
    return _conjugate_action_to_model_frame(source_action, rotation_deg=rotation_deg)


def _model_action_metadata(
    source_action: dict[str, Any],
    *,
    valley: str,
    operation: str,
    rotation_deg: float,
) -> dict[str, Any]:
    return _conjugate_action_to_model_frame(source_action, rotation_deg=rotation_deg)


def _rotation_from_action_map(action_map: Any) -> np.ndarray:
    if not isinstance(action_map, dict):
        return np.eye(2, dtype=float)
    map_type = str(action_map.get("type", "")).lower()
    if map_type == "rotation":
        theta = np.deg2rad(float(action_map.get("angle_deg", 0.0)))
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    if map_type == "reflection":
        theta = np.deg2rad(float(action_map.get("axis_deg", 0.0)))
        axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        return 2.0 * np.outer(axis, axis) - np.eye(2, dtype=float)
    if map_type == "negation":
        return -np.eye(2, dtype=float)
    return np.eye(2, dtype=float)


def _shift_reflection_axis(action_map: Any, shift_deg: float) -> Any:
    if not isinstance(action_map, dict):
        return action_map
    out = dict(action_map)
    if str(out.get("type", "")).lower() == "reflection" and "axis_deg" in out:
        out["axis_deg"] = float(out["axis_deg"]) + float(shift_deg)
    return out


def _sector_target(sector: str, sector_map: Any) -> str:
    if isinstance(sector_map, dict):
        return str(sector_map.get(sector, sector))
    if str(sector_map) == "layer_exchange":
        return {"L1": "L2", "L2": "L1"}.get(sector, sector)
    return sector


def _mapping_get_int(mapping: Mapping[Any, Any], key: int) -> int | None:
    for candidate in (key, str(key)):
        if candidate in mapping:
            return int(mapping[candidate])
    return None


def _mapping_int_keys(mapping: Mapping[Any, Any]) -> set[int]:
    keys: set[int] = set()
    for key in mapping:
        try:
            keys.add(int(key))
        except (TypeError, ValueError):
            continue
    return keys


def _mapping_get_projected_orbital(mapping: Mapping[Any, Any], orbital: int) -> int | None:
    # User-facing orbital_map is 1-based to match exactification labels. A map
    # that explicitly contains slot 0 is treated as internal 0-based form.
    if 0 in _mapping_int_keys(mapping):
        return _mapping_get_int(mapping, int(orbital))
    mapped = _mapping_get_int(mapping, int(orbital) + 1)
    return None if mapped is None else int(mapped) - 1


def _target_orbital_for_action(
    orbital: int,
    *,
    source_sector: str,
    target_sector: str,
    action: Mapping[str, Any],
) -> int:
    raw = action.get("orbital_map")
    if not isinstance(raw, Mapping):
        return int(orbital)
    candidates: list[Any] = []
    for key in (source_sector, f"{source_sector}->{target_sector}", f"{source_sector}:{target_sector}", "*"):
        if key in raw:
            candidates.append(raw[key])
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = candidate.get(target_sector)
            if isinstance(nested, Mapping):
                mapped = _mapping_get_projected_orbital(nested, int(orbital))
                if mapped is not None:
                    return mapped
            mapped = _mapping_get_projected_orbital(candidate, int(orbital))
            if mapped is not None:
                return mapped
    return int(orbital)


def _action_key_for_mismatch(action: dict[str, Any]) -> str:
    comparable = {
        key: value
        for key, value in action.items()
        if key not in {"sector_map_source", "action_source", "derivation"}
    }
    return json.dumps(comparable, sort_keys=True)


def _action_candidates_from_model_action(model_action: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]]
    sector_map = model_action.get("sector_map")
    if sector_map in {"auto", None}:
        raise ValueError("model_action requires explicit sector_map metadata")
    else:
        candidates = [dict(model_action)]
        if sector_map == "identity":
            swapped = dict(model_action)
            swapped["sector_map"] = "layer_exchange"
            candidates.append(swapped)
    if (
        bool(model_action.get("antiunitary", False))
        and isinstance(model_action.get("q_map"), dict)
        and str(model_action["q_map"].get("type", "")).lower() == "reflection"
    ):
        for shift in (-90.0, 90.0):
            for base in list(candidates):
                shifted = dict(base)
                shifted["k_map"] = _shift_reflection_axis(base.get("k_map"), shift)
                shifted["q_map"] = _shift_reflection_axis(base.get("q_map"), shift)
                candidates.append(shifted)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _sector_orbital_counts(
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    *,
    low_dim: int | None,
    num_layer_list: Sequence[int] | None = None,
) -> tuple[int, int]:
    if num_layer_list is not None:
        total_layers = int(sum(int(n) for n in num_layer_list))
        if len(nlow_state_list) != total_layers:
            raise ValueError(
                f"project.nlow_state_list must have {total_layers} physical-layer rows "
                f"(sum(material.num_layer_list)); got {len(nlow_state_list)}. "
                "Use [] for layers that do not contribute."
            )
        counts: list[int] = []
        offset = 0
        for n_layers in list(num_layer_list)[:2]:
            counts.append(sum(len(nlow_state_list[offset + local]) for local in range(int(n_layers))))
            offset += int(n_layers)
        while len(counts) < 2:
            counts.append(0)
        return int(counts[0]), int(counts[1])
    if len(nlow_state_list) == 2:
        return len(nlow_state_list[0]), len(nlow_state_list[1])
    raise ValueError(
        f"project.nlow_state_list must have two qset rows when material.num_layer_list is unavailable; "
        f"got {len(nlow_state_list)} rows."
    )


def _model_basis_labels(
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    *,
    low_dim: int | None = None,
    num_layer_list: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    n_orb1, n_orb2 = _sector_orbital_counts(
        q_model1,
        q_model2,
        nlow_state_list,
        low_dim=low_dim,
        num_layer_list=num_layer_list,
    )
    for sector, qset, bands in (
        ("L1", np.asarray(q_model1, dtype=float), range(n_orb1)),
        ("L2", np.asarray(q_model2, dtype=float), range(n_orb2)),
    ):
        for orbital_slot, _band in enumerate(bands):
            for q_index, q_vector in enumerate(qset):
                labels.append(
                    {
                        "sector": sector,
                        "q_index": int(q_index),
                        "orbital": int(orbital_slot),
                        "q_vector": np.asarray(q_vector, dtype=float),
                    }
                )
    return labels


def _basis_action_for_candidate(
    *,
    action: dict[str, Any],
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    low_dim: int | None,
    num_layer_list: Sequence[int] | None = None,
    tol: float,
) -> dict[str, Any]:
    qsets = {"L1": np.asarray(q_model1, dtype=float), "L2": np.asarray(q_model2, dtype=float)}
    labels = _model_basis_labels(
        q_model1,
        q_model2,
        nlow_state_list,
        low_dim=low_dim,
        num_layer_list=num_layer_list,
    )
    target_index = {
        (str(label["sector"]), int(label["q_index"]), int(label["orbital"])): idx
        for idx, label in enumerate(labels)
    }
    q_action_items: list[dict[str, Any]] = []
    q_action_seen: set[tuple[str, int]] = set()
    perm = np.full(len(labels), -1, dtype=int)
    missing: list[dict[str, Any]] = []
    R = _rotation_from_action_map(action.get("q_map", action.get("k_map")))
    for src_idx, label in enumerate(labels):
        source_sector = str(label["sector"])
        target_sector = _sector_target(source_sector, action.get("sector_map", "identity"))
        target_qset = qsets.get(target_sector)
        if target_qset is None or target_qset.size == 0:
            missing.append({"source_index": int(src_idx), "reason": "missing_target_sector"})
            continue
        mapped = R @ np.asarray(label["q_vector"], dtype=float)
        distances = np.linalg.norm(target_qset - mapped, axis=1)
        target_q_index = int(np.argmin(distances))
        residual = float(distances[target_q_index])
        if residual > float(tol):
            missing.append(
                {
                    "source_index": int(src_idx),
                    "source_sector": source_sector,
                    "target_sector": target_sector,
                    "q_residual": residual,
                }
            )
            continue
        target_orbital = _target_orbital_for_action(
            int(label["orbital"]),
            source_sector=source_sector,
            target_sector=target_sector,
            action=action,
        )
        key = (target_sector, target_q_index, target_orbital)
        if key not in target_index:
            missing.append(
                {
                    "source_index": int(src_idx),
                    "reason": "missing_target_orbital",
                    "source_orbital": int(label["orbital"]),
                    "target_orbital": int(target_orbital),
                }
            )
            continue
        perm[src_idx] = int(target_index[key])
        q_key = (source_sector, int(label["q_index"]))
        if q_key not in q_action_seen:
            q_action_seen.add(q_key)
            q_action_items.append(
                {
                    "source_sector": source_sector,
                    "source_q_index": int(label["q_index"]),
                    "target_sector": target_sector,
                    "target_q_index": target_q_index,
                    "q_residual": residual,
                }
            )
    return {
        "complete": bool(np.all(perm >= 0)),
        "perm": perm,
        "items": q_action_items,
        "missing": missing,
        "sector_map": action.get("sector_map", "identity"),
        "orbital_map": action.get("orbital_map"),
    }


def _block_support_residual(D: np.ndarray, labels: list[dict[str, Any]], perm: np.ndarray) -> float:
    arr = np.asarray(D, dtype=np.complex128)
    if arr.ndim == 3:
        arr = arr[0]
    mask = np.zeros(arr.shape, dtype=bool)
    target_groups: dict[tuple[str, int], list[int]] = {}
    for idx, label in enumerate(labels):
        target_groups.setdefault((str(label["sector"]), int(label["q_index"])), []).append(idx)
    for src_idx, target_idx in enumerate(perm):
        if target_idx < 0:
            continue
        src_label = labels[src_idx]
        tgt_label = labels[int(target_idx)]
        src_cols = target_groups[(str(src_label["sector"]), int(src_label["q_index"]))]
        tgt_rows = target_groups[(str(tgt_label["sector"]), int(tgt_label["q_index"]))]
        mask[np.ix_(tgt_rows, src_cols)] = True
    off = arr.copy()
    off[mask] = 0.0
    denom = float(np.linalg.norm(arr))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(off) / denom)


def _resolve_projected_model_action(
    *,
    D_low: np.ndarray,
    support_matrices: Sequence[tuple[str, np.ndarray]] | None = None,
    model_action: dict[str, Any],
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    nlow_state_list: list[list[int]],
    num_layer_list: Sequence[int] | None = None,
    tol: float,
    discover_action_candidates: bool = False,
    accept_support_resolved_action: bool = False,
    strict: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if model_action.get("sector_map") in {"auto", None}:
        raise ValueError("model_action requires explicit sector_map metadata")
    matrix_options = list(support_matrices or [("raw", D_low)])
    low_dim = int(np.asarray(matrix_options[0][1]).shape[-1])
    labels = _model_basis_labels(
        q_model1,
        q_model2,
        nlow_state_list,
        low_dim=low_dim,
        num_layer_list=num_layer_list,
    )
    declared_basis_action = _basis_action_for_candidate(
        action=model_action,
        q_model1=q_model1,
        q_model2=q_model2,
        nlow_state_list=nlow_state_list,
        low_dim=low_dim,
        num_layer_list=num_layer_list,
        tol=tol,
    )
    declared_residuals: list[dict[str, Any]] = []
    if declared_basis_action["complete"]:
        for matrix_label, matrix in matrix_options:
            residual = _block_support_residual(matrix, labels, np.asarray(declared_basis_action["perm"], dtype=int))
            declared_residuals.append({"matrix": matrix_label, "block_off_support_rel": residual})
    best: tuple[float, str, dict[str, Any], dict[str, Any]] | None = None
    diagnostics: list[dict[str, Any]] = []
    candidate_actions = _action_candidates_from_model_action(model_action) if discover_action_candidates else [dict(model_action)]
    for candidate in candidate_actions:
        basis_action = _basis_action_for_candidate(
            action=candidate,
            q_model1=q_model1,
            q_model2=q_model2,
            nlow_state_list=nlow_state_list,
            low_dim=low_dim,
            num_layer_list=num_layer_list,
            tol=tol,
        )
        residuals: list[dict[str, Any]] = []
        if basis_action["complete"]:
            for matrix_label, matrix in matrix_options:
                residual = _block_support_residual(matrix, labels, np.asarray(basis_action["perm"], dtype=int))
                residuals.append({"matrix": matrix_label, "block_off_support_rel": residual})
                if best is None or residual < best[0]:
                    best = (residual, matrix_label, candidate, basis_action)
        diagnostics.append(
            {
                "action": {
                    "k_map": candidate.get("k_map"),
                    "q_map": candidate.get("q_map"),
                    "sector_map": candidate.get("sector_map"),
                    "orbital_map": candidate.get("orbital_map"),
                    "antiunitary": candidate.get("antiunitary"),
                },
                "complete": bool(basis_action["complete"]),
                "missing_count": int(len(basis_action["missing"])),
                "support_residuals": residuals,
                "block_off_support_rel": min((row["block_off_support_rel"] for row in residuals), default=None),
            }
        )
    selected_candidate = dict(model_action)
    selected_basis_action = declared_basis_action
    if best is None:
        support_matrix_source = None
        selected_residual = None
    else:
        selected_residual, support_matrix_source, selected_candidate, selected_basis_action = best
    action_mismatch = _action_key_for_mismatch(selected_candidate) != _action_key_for_mismatch(model_action)
    if strict and action_mismatch:
        raise ValueError("support discovery selected an action that differs from declared model_action in strict mode")
    use_selected = bool(accept_support_resolved_action and action_mismatch)
    resolved_action = dict(selected_candidate if use_selected else model_action)
    basis_action = selected_basis_action if use_selected else declared_basis_action
    residual = selected_residual if use_selected else min((row["block_off_support_rel"] for row in declared_residuals), default=None)
    selected_report = selected_candidate if discover_action_candidates else dict(model_action)
    selected_report_residual = selected_residual if discover_action_candidates else residual
    provenance = None
    if use_selected:
        provenance = {
            "source": "support_exactification",
            "accepted_by": "kp_projected_basis_inference",
            "accepted_by_user": False,
            "declared_model_action": model_action,
            "selected_action_candidate": selected_candidate,
            "declared_support_residual": min((row["block_off_support_rel"] for row in declared_residuals), default=None),
            "selected_support_residual": selected_residual,
        }
        resolved_action["provenance"] = provenance
    basis_action_out = {
        "complete": bool(basis_action["complete"]),
        "sector_map": basis_action["sector_map"],
        "orbital_map": basis_action.get("orbital_map"),
        "items": basis_action["items"],
        "missing": basis_action["missing"],
        "support_resolution": {
            "block_off_support_rel": residual,
            "support_matrix_source": support_matrix_source,
            "matrix_kind": "action",
            "matrix_source": "raw_h_action_projection",
            "action_mismatch": bool(action_mismatch),
            "candidate_source": "support_discovery" if discover_action_candidates else "manifest_model_action",
            "declared_model_action": model_action,
            "selected_model_action": selected_report,
            "selected_action_candidate": selected_report,
            "declared_support_residual": min((row["block_off_support_rel"] for row in declared_residuals), default=None),
            "declared_support_residuals": declared_residuals,
            "selected_support_residual": selected_report_residual,
            "candidates": diagnostics,
        },
    }
    if provenance is not None:
        basis_action_out["support_resolution"]["provenance"] = provenance
    return resolved_action, basis_action_out


def _has_projection_quality_warnings(pair_rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(bool(row.get("quality_warnings")) for row in pair_rows if isinstance(row, Mapping))


def _select_operation_matrix_kind(
    model_basis_action: Mapping[str, Any],
    *,
    representation_pair_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    support_source = (model_basis_action.get("support_resolution") or {}).get("support_matrix_source")
    representation_invalid = _has_projection_quality_warnings(representation_pair_rows)
    projection_warnings = (
        [str(item) for row in representation_pair_rows for item in row.get("quality_warnings", [])]
        if representation_pair_rows
        else []
    )
    rep_support_selected = support_source == "representation"
    return "action", {
        "kind": "action",
        "matrix_source": "raw_h_action_projection",
        "reason": "representation_projection_diagnostic_only" if rep_support_selected else "support_source_selected_action",
        "support_matrix_source": support_source,
        "representation_quality_warnings": representation_invalid,
        "representation_projection_diagnostic": {
            "status": "raw_action_exactification_problem" if rep_support_selected else "not_selected",
            "support_matrix_source": support_source,
            "projection_warnings": projection_warnings,
            "representation_support_cleaner_than_raw_action": rep_support_selected,
        },
    }


def _c2_action_audit_for_gamma(
    *,
    mode: str,
    operation: str,
    matrix_kind: str,
    matrix_selection: Mapping[str, Any],
    model_basis_action: Mapping[str, Any],
    raw_matrix: np.ndarray,
    representation_matrix: np.ndarray | None,
    pair_rows: Sequence[Mapping[str, Any]],
    representation_pair_rows: Sequence[Mapping[str, Any]],
    declared_model_action: Mapping[str, Any],
    combined_raw_h_residual: float | None,
) -> dict[str, Any] | None:
    if str(mode).lower() != "gamma" or operation != "C2":
        return None
    support_resolution = model_basis_action.get("support_resolution", {})
    residuals = support_resolution.get("declared_support_residuals", []) if isinstance(support_resolution, Mapping) else []
    by_matrix = {
        str(row.get("matrix")): row.get("block_off_support_rel")
        for row in residuals
        if isinstance(row, Mapping)
    }
    raw = np.asarray(raw_matrix, dtype=np.complex128)
    denom = max(float(np.linalg.norm(raw)), 1.0)
    if representation_matrix is None:
        rep_diff = None
        rep_equivalent = None
    else:
        rep = np.asarray(representation_matrix, dtype=np.complex128)
        rep_diff = float(np.linalg.norm(raw - rep) / denom)
        rep_equivalent = bool(rep_diff < 1.0e-10)
    rep_diag = matrix_selection.get("representation_projection_diagnostic", {}) if isinstance(matrix_selection, Mapping) else {}
    return {
        "matrix_kind": matrix_kind,
        "matrix_source": "raw_h_action_projection",
        "D_low_action_support_residual": by_matrix.get("raw_action", support_resolution.get("declared_support_residual") if isinstance(support_resolution, Mapping) else None),
        "D_low_rep_support_residual": by_matrix.get("representation"),
        "D_low_action_vs_rep_norm": rep_diff,
        "rawH_full_space_covariance_residual": (
            pair_rows[0].get("full_space_covariance_residual")
            if pair_rows and isinstance(pair_rows[0], Mapping)
            else None
        ),
        "combined_raw_h_residual": combined_raw_h_residual,
        "declared_model_action": dict(declared_model_action),
        "sector_map": declared_model_action.get("sector_map"),
        "q_map": declared_model_action.get("q_map"),
        "group_relation_residuals_action": {},
        "exactification_status": rep_diag.get("status", "not_run_in_kp_symm_projection") if isinstance(rep_diag, Mapping) else "not_run_in_kp_symm_projection",
        "representation_projection_equivalent": rep_equivalent,
        "representation_projection_diagnostic": rep_diag,
        "representation_projection_warnings": [
            str(item)
            for row in representation_pair_rows
            if isinstance(row, Mapping)
            for item in row.get("quality_warnings", [])
        ],
    }


def _optional_entry_filename(entry: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if entry.get(key):
            return str(entry[key])
    developer_outputs = entry.get("developer_outputs")
    if isinstance(developer_outputs, Mapping):
        for key in keys:
            if developer_outputs.get(key):
                return str(developer_outputs[key])
    return None


def _spin_route_endpoints(
    spin: str,
    spin_sector_sewing: str | None,
) -> tuple[str, str]:
    source = str(spin).strip().lower()
    if source == "all" and spin_sector_sewing is None:
        return "all", "all"
    if source not in {"up", "down"}:
        raise ValueError(f"cross-spin routing requires source spin up/down, got {spin!r}")
    if spin_sector_sewing is None:
        return source, source
    route = str(spin_sector_sewing).strip().lower()
    endpoints = {
        "up_to_down": ("up", "down"),
        "down_to_up": ("down", "up"),
    }
    if route not in endpoints:
        raise ValueError(f"Unsupported spin_sector_sewing mode: {spin_sector_sewing!r}")
    route_source, route_target = endpoints[route]
    if source != route_source:
        raise ValueError(
            f"spin_sector_sewing={route} requires spin: {route_source}, got {spin!r}"
        )
    return route_source, route_target


def _resolve_spin_sector_sewing(
    *,
    spin: str,
    rep_root: Path,
    operation_entries: Mapping[str, Mapping[str, Any]],
    explicit_route: Any,
    ambiguity_tolerance: float = 1.0e-8,
) -> tuple[str | None, dict[str, Any]]:
    """Infer the run-wide spin route from TAPW raw-H action support."""

    spin_text = str(spin).strip().lower()
    if spin_text == "all":
        if explicit_route not in (None, "", "auto"):
            raise ValueError("spin_sector_sewing is incompatible with spin: all")
        return None, {
            "source": "full_spin_action",
            "source_spin": "all",
            "target_spin": "all",
            "route": None,
            "operations": {},
        }
    if spin_text not in {"up", "down"}:
        raise ValueError(f"Unsupported spin value for symmetry routing: {spin!r}")

    routes: dict[str, str | None] = {}
    operation_diagnostics: dict[str, dict[str, Any]] = {}
    for operation, raw_entry in operation_entries.items():
        entry = dict(raw_entry)
        filename = _optional_entry_filename(
            entry,
            "raw_h_operator_file",
            "raw_operator_file",
            "full_space_action_file",
            "action_operator_file",
        )
        if filename is None:
            raise ValueError(
                f"Operation {operation!r} cannot infer its spin route without raw_h_operator_file"
            )
        matrix = _load_matrix(rep_root / filename)
        route, diagnostics = _infer_spin_route_from_action_matrix(
            matrix,
            source_spin=spin_text,
            ambiguity_tolerance=ambiguity_tolerance,
        )
        routes[str(operation)] = route
        operation_diagnostics[str(operation)] = diagnostics

    inferred = _combine_spin_routes(routes)
    explicit = None if explicit_route in (None, "", "auto") else str(explicit_route).strip().lower()
    if explicit is not None:
        _spin_route_endpoints(spin_text, explicit)
        if explicit != inferred:
            raise ValueError(
                "explicit spin_sector_sewing conflicts with raw-H action support: "
                f"explicit={explicit!r}, inferred={inferred!r}"
            )
    source_spin, target_spin = _spin_route_endpoints(spin_text, inferred)
    return inferred, {
        "source": "raw_h_action_support",
        "source_spin": source_spin,
        "target_spin": target_spin,
        "route": inferred,
        "explicit_assertion": explicit,
        "ambiguity_tolerance": float(ambiguity_tolerance),
        "operations": operation_diagnostics,
    }


def _pairs_from_entry(
    entry: dict[str, Any],
    nk: int,
    *,
    default_k_index: int | None = None,
    allow_default_k_index: bool = False,
) -> list[tuple[int, int]]:
    raw_pairs = entry.get("k_pairs", entry.get("pairs"))
    if raw_pairs is not None:
        pairs: list[tuple[int, int]] = []
        for item in raw_pairs:
            if isinstance(item, dict):
                target = item.get("target", item.get("target_index", item.get("k_target")))
                source = item.get("source", item.get("source_index", item.get("k_source")))
            else:
                target, source = item
            pairs.append((int(target), int(source)))
        return pairs
    target_indices = entry.get("target_indices")
    source_indices = entry.get("source_indices")
    if target_indices is not None and source_indices is not None:
        if len(target_indices) != len(source_indices):
            raise ValueError("target_indices and source_indices have different lengths")
        return [(int(t), int(s)) for t, s in zip(target_indices, source_indices)]
    rule = str(entry.get("source_k_rule", entry.get("k_rule", ""))).lower()
    target_rule = str(entry.get("target_k_rule", "")).lower()
    if rule in {"same", "identity", "same_index"} or target_rule in {"same", "identity", "same_index"}:
        return [(idx, idx) for idx in range(nk)]
    if rule in {"gamma", "gamma_only"} or target_rule in {"gamma", "gamma_only"}:
        return [(0, 0)]
    if default_k_index is not None and allow_default_k_index:
        entry["k_pairs_inferred_from_default_k_index"] = True
        entry["candidate_source"] = "diagnostic_default_k_index"
        entry["_k_pairs_provenance"] = {
            "authored_in_manifest": False,
            "candidate_source": "diagnostic_default_k_index",
            "default_k_index": int(default_k_index),
        }
        return [(int(default_k_index), int(default_k_index))]
    raise ValueError("Manifest operation must provide k_pairs/source_indices or a supported k rule")


def _check_spin_leakage(operation: str, label: str, rep: RepresentationData, tolerance: float) -> None:
    if rep.spin_leakage is not None and rep.spin_leakage > tolerance:
        raise ValueError(f"{operation} {label} spin off-block leakage {rep.spin_leakage:.3e} exceeds tolerance")


def _load_spin_sliced_representation(
    *,
    path: Path,
    spin: str,
    full_dim: int,
    spin_sector_sewing: str | None = None,
) -> RepresentationData:
    matrix = _load_matrix(path)
    if spin_sector_sewing is None:
        return _slice_representation_for_spin(matrix, spin, full_dim)
    source_spin, target_spin = _spin_route_endpoints(spin, spin_sector_sewing)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Representation matrix must be square, got shape={matrix.shape}")
    if matrix.shape != (2 * full_dim, 2 * full_dim):
        raise ValueError(
            f"D dimension {matrix.shape} cannot be sliced to {spin_sector_sewing} "
            f"block with full_dim {full_dim}"
        )
    slices = {
        "up": slice(0, full_dim),
        "down": slice(full_dim, 2 * full_dim),
    }
    row = slices[target_spin]
    col = slices[source_spin]
    block = matrix[row, col].tocsr() if _is_sparse(matrix) else np.asarray(matrix[row, col], dtype=np.complex128)
    return RepresentationData(matrix=block, from_full_spinful=True, spin_leakage=None)


def _build_action_representation(
    *,
    operation: str,
    entry: dict[str, Any],
    rep_root: Path,
    filename: str | None,
    antiunitary: bool,
    spin: str,
    full_dim: int,
    tolerance: float,
    spin_sector_sewing: str | None = None,
) -> ActionRepresentation:
    rep: RepresentationData | None = None
    if filename is not None:
        rep = _load_spin_sliced_representation(
            path=rep_root / filename,
            spin=spin,
            full_dim=full_dim,
            spin_sector_sewing=spin_sector_sewing,
        )
        _check_spin_leakage(operation, "representation", rep, tolerance)
        if rep.matrix.shape != (full_dim, full_dim):
            raise ValueError(f"{operation} D shape {rep.matrix.shape} does not match U_low full dimension {full_dim}")

    pg_filename = _optional_entry_filename(entry, "pg_file", "periodic_gauge_file", "source_pg_file")
    raw_h_filename = _optional_entry_filename(
        entry,
        "raw_h_operator_file",
        "raw_operator_file",
        "full_space_action_file",
        "action_operator_file",
    )
    if raw_h_filename is None:
        raise ValueError(
            f"{operation} source manifest must provide raw_h_operator_file; "
            "bare representation_file or representation_file+pg_file is not a valid action source"
        )
    pg_rep: RepresentationData | None = None
    raw_h_rep: RepresentationData | None = None
    combined: np.ndarray | None = None
    combined_residual: float | None = None

    if pg_filename is not None and rep is not None:
        if spin_sector_sewing is not None:
            pg_rep = _load_spin_sliced_representation(
                path=rep_root / pg_filename,
                spin=spin,
                full_dim=full_dim,
                spin_sector_sewing=None,
            )
        else:
            pg_rep = _load_spin_sliced_representation(
                path=rep_root / pg_filename,
                spin=spin,
                full_dim=full_dim,
                spin_sector_sewing=spin_sector_sewing,
            )
        _check_spin_leakage(operation, "periodic-gauge", pg_rep, tolerance)
        if pg_rep.matrix.shape != (full_dim, full_dim):
            raise ValueError(f"{operation} PG shape {pg_rep.matrix.shape} does not match U_low full dimension {full_dim}")
        pg_for_raw_h = pg_rep.matrix.conj() if antiunitary else pg_rep.matrix
        combined = rep.matrix @ pg_for_raw_h

    raw_h_rep = _load_spin_sliced_representation(
        path=rep_root / raw_h_filename,
        spin=spin,
        full_dim=full_dim,
        spin_sector_sewing=spin_sector_sewing,
    )
    _check_spin_leakage(operation, "raw-H operator", raw_h_rep, tolerance)
    if raw_h_rep.matrix.shape != (full_dim, full_dim):
        raise ValueError(f"{operation} raw-H operator shape {raw_h_rep.matrix.shape} does not match U_low full dimension {full_dim}")
    if combined is not None:
        combined_residual = _fro_relative(raw_h_rep.matrix, combined, raw_h_rep.matrix)
        if combined_residual > tolerance:
            raise ValueError(
                f"{operation} manifest raw-H operator disagrees with D_g^(0)+PG by "
                f"{combined_residual:.3e}"
            )
    return ActionRepresentation(
        matrix=raw_h_rep.matrix,
        representation=rep,
        pg=pg_rep,
        raw_h=raw_h_rep,
        pg_filename=pg_filename,
        raw_h_filename=raw_h_filename,
        action_source="raw_h_operator_file",
        combined_raw_h_residual=combined_residual,
    )


def _projectors_for_k(
    hamk_spin: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    *,
    orb0: int,
    num_layer_list: list[int] | None = None,
    num_orb_per_layer_list: list[list[int]] | None = None,
    spin: str,
    mode: str,
    nlow_state_list: list[list[int]],
    norb_fix_list: list[Any],
    method: str,
    e_ref: float | None,
    project_cfg: dict[str, Any],
    compute_heff: bool = True,
    eigensystem_cache: dict[Any, tuple[np.ndarray, np.ndarray]] | None = None,
) -> ProjectionState:
    if num_layer_list is None:
        num_layer_list = [1, 1]
    if num_orb_per_layer_list is None:
        num_orb_per_layer_list = [[int(orb0)] for _ in num_layer_list]
    _validate_project_layer_lists(
        nlow_state_list,
        norb_fix_list,
        num_layer_list=num_layer_list,
        context="project",
    )
    q_layers = [[q1], [q2]]
    orb_layers = num_orb_per_layer_list
    method = str(method).lower()
    mode_lower = str(mode).lower()
    include_high = method != "first_order"
    _, h_vec_blk, _, _ = get_H_block(
        hamk_spin,
        q_layers,
        num_layer_list,
        orb_layers,
        nlow_state_list,
        norb_fix_list,
        spin=spin,
        mode=mode,
        selected_bands_by_layer=None if include_high else nlow_state_list,
        eigensystem_cache=eigensystem_cache,
    )
    if mode_lower != "gamma":
        q_count = int(len(q1))
        shift = q_count * int(orb0)
        idx_list: list[np.ndarray] = []
        layer_offset = 0
        per_spin_offset = hamk_spin.shape[0] // 2
        bands_by_physical_layer: list[list[int]] = []
        for group_index, n_layers in enumerate(num_layer_list):
            for layer_in_group in range(n_layers):
                band_entry = len(bands_by_physical_layer)
                bands_by_physical_layer.append([int(band) for band in nlow_state_list[band_entry]])
                for iq in range(q_count):
                    base = np.arange((iq * n_layers + layer_in_group) * int(orb0), (iq * n_layers + layer_in_group + 1) * int(orb0))
                    base = base + shift * layer_offset
                    if spin == "all":
                        idx = np.concatenate((base, base + per_spin_offset))
                    elif spin == "down":
                        idx = base + per_spin_offset
                    else:
                        idx = base
                    idx_list.append(idx)
            layer_offset += n_layers
        u_low, u_high = _assemble_projectors_from_block_eigenvectors(
            h_vec_blk,
            idx_list,
            bands_by_physical_layer,
            include_high=include_high,
        )
    else:
        q_count = int(len(q1))
        shift = q_count * int(orb0)
        per_spin_offset = hamk_spin.shape[0] // 2
        idx_list = []
        for iq in range(q_count):
            parts = []
            layer_offset = 0
            for group_index, n_layers in enumerate(num_layer_list):
                for layer_in_group in range(n_layers):
                    base = np.arange(
                        (iq * n_layers + layer_in_group) * int(orb0),
                        (iq * n_layers + layer_in_group + 1) * int(orb0),
                    )
                    parts.append(base + shift * layer_offset)
                layer_offset += n_layers
            idx = np.concatenate(parts)
            if spin == "all":
                idx = np.concatenate((idx, idx + per_spin_offset))
            elif spin == "down":
                idx = idx + per_spin_offset
            idx_list.append(idx)
        u_low, u_high = _assemble_gamma_projectors_from_block_eigenvectors(
            h_vec_blk,
            idx_list,
            _source_group_nlow_state_list(nlow_state_list, num_layer_list),
            include_high=include_high,
        )
    u_low = np.asarray(u_low, dtype=np.complex128)
    if not compute_heff:
        return ProjectionState(
            hamk=np.asarray(hamk_spin, dtype=np.complex128),
            heff=np.zeros((u_low.shape[1], u_low.shape[1]), dtype=np.complex128),
            u_low=u_low,
        )
    result = downfold_from_projectors(
        hamk_spin,
        u_low,
        u_high,
        DownfoldingOptions(
            method=method,
            e_ref=e_ref,
            pole_warning_mev=float(project_cfg.get("pole_warning_mev", 10.0)),
            pole_danger_mev=float(project_cfg.get("pole_danger_mev", 1.0)),
            fail_on_near_pole=_as_bool(project_cfg.get("fail_on_near_pole", False)),
        ),
    )
    return ProjectionState(hamk=np.asarray(hamk_spin, dtype=np.complex128), heff=np.asarray(result.heff, dtype=np.complex128), u_low=u_low)


def _full_space_covariance_residual(
    *,
    d_full: np.ndarray,
    h_target: np.ndarray,
    h_source: np.ndarray,
    antiunitary: bool,
) -> float:
    h_source_work = h_source.conj() if antiunitary else h_source
    d_dag = d_full.conjugate().transpose() if _is_sparse(d_full) else d_full.conj().T
    h_cov = (d_full @ h_source_work) @ d_dag
    if _is_sparse(h_cov):
        h_cov = h_cov.toarray()
    return _fro_relative(h_target, h_cov, h_target)


def _project_operation(
    *,
    operation: str,
    antiunitary: bool,
    d_full: np.ndarray,
    states: dict[int, ProjectionState],
    pairs: list[tuple[int, int]],
    tolerance: float,
    enforce_heff_covariance: bool = True,
    compute_heff_covariance: bool = True,
    raise_on_quality_failure: bool = True,
    compute_polar: bool = False,
    target_states: dict[int, ProjectionState] | None = None,
    source_states: dict[int, ProjectionState] | None = None,
):
    raw_mats = []
    polar_mats = []
    pair_rows = []
    if target_states is None:
        target_states = states
    if source_states is None:
        source_states = states
    for target_idx, source_idx in pairs:
        target = target_states[target_idx]
        source = source_states[source_idx]
        raw_evaluation = evaluate_projected_pair(
            d_full=d_full,
            target_u_low=target.u_low,
            source_u_low=source.u_low,
            target_heff=target.heff,
            source_heff=source.heff,
            antiunitary=antiunitary,
            compute_heff_covariance=compute_heff_covariance,
        )
        d_raw = raw_evaluation.projected_action

        def metrics(d_matrix: np.ndarray) -> dict[str, Any]:
            evaluation = (
                raw_evaluation
                if d_matrix is d_raw
                else evaluate_projected_pair(
                    d_full=d_full,
                    target_u_low=target.u_low,
                    source_u_low=source.u_low,
                    target_heff=target.heff,
                    source_heff=source.heff,
                    antiunitary=antiunitary,
                    action=d_matrix,
                    compute_heff_covariance=compute_heff_covariance,
                )
            )
            return {
                "d_unitarity_error": evaluation.action_unitarity_residual,
                "subspace_leakage": evaluation.subspace_residual,
                "heff_covariance_residual": evaluation.heff_covariance_residual,
            }

        raw_metrics = metrics(d_raw)
        if compute_polar:
            x, singular_values, yh = np.linalg.svd(d_raw, full_matrices=False)
            d_polar = x @ yh
            polar_metrics = metrics(d_polar)
            sv_max_dev = float(np.max(np.abs(singular_values - 1.0))) if singular_values.size else 0.0
            singular_values_out = [float(value) for value in singular_values]
            polar_mats.append(d_polar)
        else:
            d_unitarity = raw_metrics["d_unitarity_error"]
            sv_max_dev = float(min(d_unitarity, 1.0))
            singular_values_out = []
            polar_metrics = None
        quality_warnings = []
        if sv_max_dev > tolerance:
            message = (
                f"unitarity error {raw_metrics['d_unitarity_error']:.3e} exceeds tolerance"
                if not compute_polar
                else f"singular values deviate from 1 by {sv_max_dev:.3e}"
            )
            if raise_on_quality_failure:
                raise ValueError(f"{operation} k=({target_idx},{source_idx}) {message}")
            quality_warnings.append(message)
        if raw_metrics["subspace_leakage"] > tolerance:
            message = f"subspace leakage {raw_metrics['subspace_leakage']:.3e} exceeds tolerance"
            if raise_on_quality_failure:
                raise ValueError(f"{operation} k=({target_idx},{source_idx}) {message}")
            quality_warnings.append(message)
        if (
            enforce_heff_covariance
            and raw_metrics["heff_covariance_residual"] is not None
            and raw_metrics["heff_covariance_residual"] > tolerance
        ):
            message = f"Heff covariance residual {raw_metrics['heff_covariance_residual']:.3e} exceeds tolerance"
            if raise_on_quality_failure:
                raise ValueError(f"{operation} k=({target_idx},{source_idx}) {message}")
            quality_warnings.append(message)
        raw_mats.append(d_raw)
        pair_row = {
            "target_k_index": int(target_idx),
            "source_k_index": int(source_idx),
            "d_shape": list(d_raw.shape),
            "singular_values": singular_values_out,
            "singular_value_max_deviation": sv_max_dev,
            "quality_warnings": quality_warnings,
            "raw": raw_metrics,
        }
        if polar_metrics is not None:
            pair_row["polar"] = polar_metrics
        pair_rows.append(pair_row)
    return raw_mats, polar_mats, pair_rows


def _save_matrix_stack(path: Path, matrices: list[np.ndarray]) -> None:
    if len(matrices) == 1:
        np.save(path, matrices[0])
    else:
        np.save(path, np.stack(matrices, axis=0))


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# KP Symmetry Projection Summary",
        "",
        f"Valley: {summary['valley']}",
        f"Spin: {summary['spin']}",
        f"Tolerance: {summary['tolerance']}",
        f"Full dimension: {summary['full_dim']}",
        f"Low dimension: {summary['low_dim']}",
        "",
        "## Operations",
        "",
    ]
    for op in summary["operations"]:
        lines.append(f"### {op['operation']}")
        lines.append("")
        lines.append(f"- antiunitary: {op['antiunitary']}")
        if op.get("representation_file"):
            lines.append(f"- representation_file: `{op['representation_file']}`")
        developer_outputs = op.get("developer_outputs")
        if isinstance(developer_outputs, Mapping) and developer_outputs.get("representation_file"):
            lines.append(f"- representation_file: `{developer_outputs['representation_file']}`")
        lines.append(f"- action_source: {op.get('action_source', 'representation_file')}")
        if op.get("pg_file") is not None:
            lines.append(f"- pg_file: `{op['pg_file']}`")
        if op.get("raw_h_operator_file") is not None:
            lines.append(f"- raw_h_operator_file: `{op['raw_h_operator_file']}`")
        if op.get("combined_raw_h_residual") is not None:
            lines.append(f"- combined_raw_h_residual: {op['combined_raw_h_residual']:.6e}")
        if op.get("spin_leakage") is not None:
            lines.append(f"- spin_leakage: {op['spin_leakage']:.6e}")
        if op.get("pg_spin_leakage") is not None:
            lines.append(f"- pg_spin_leakage: {op['pg_spin_leakage']:.6e}")
        if op.get("raw_h_spin_leakage") is not None:
            lines.append(f"- raw_h_spin_leakage: {op['raw_h_spin_leakage']:.6e}")
        if op.get("matrix_scope"):
            lines.append(f"- matrix_scope: {op['matrix_scope']}")
        if op.get("valid_k_domain"):
            lines.append(f"- valid_k_domain: {op['valid_k_domain']}")
        if op.get("reference_pairs_are_not_domain_restrictions") is not None:
            lines.append(
                "- reference_pairs_are_not_domain_restrictions: "
                f"{op['reference_pairs_are_not_domain_restrictions']}"
            )
        for pair in op["pairs"]:
            raw = pair["raw"]
            full_covariance = pair.get("full_space_covariance_residual")
            full_covariance_text = (
                "skipped" if full_covariance is None else f"{float(full_covariance):.6e}"
            )
            raw_covariance = raw.get("heff_covariance_residual")
            raw_covariance_text = (
                "skipped" if raw_covariance is None else f"{float(raw_covariance):.6e}"
            )
            line = (
                f"- reference projection diagnostic target/source "
                f"{pair['target_k_index']}/{pair['source_k_index']}: "
                f"full cov={full_covariance_text}, "
                f"raw cov={raw_covariance_text}, "
                f"raw leakage={raw['subspace_leakage']:.6e}"
            )
            if "polar" in pair:
                line += f", polar cov={pair['polar']['heff_covariance_residual']:.6e}"
            lines.append(line)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class _ProjectionRunConfig:
    cfg_path: str
    cfg_dir: str
    material: Mapping[str, Any]
    plot_cfg: Mapping[str, Any]
    project_cfg: Mapping[str, Any]
    symm_cfg: Mapping[str, Any]
    save_projection_diagnostics: bool
    valley: str
    spin: str
    spin_sector_sewing: Any
    q_rotation_raw: Any
    tolerance: float
    canonical_layout: bool


@dataclass
class _ProjectionRunContext:
    config: _ProjectionRunConfig
    rep_root: Path
    operation_requests: list[dict[str, Any]]
    spin_route_inference: Mapping[str, Any]
    q1: np.ndarray
    q2: np.ndarray
    q_count: int
    mode: str
    method: str
    e_ref: float | None
    nlow_state_list: list[list[int]]
    num_layer_list: list[int]
    orb0: int
    num_orb_per_layer_list: list[int]
    required_k: list[int]
    hamk_source_by_k: dict[int, np.ndarray]
    hamk_target_by_k: dict[int, np.ndarray]
    full_dim: int
    output_dir: Path
    q_model1: np.ndarray
    q_model2: np.ndarray
    q_rotation_deg: float
    frame_inference: Mapping[str, Any]
    operation_payloads: dict[str, dict[str, Any]]
    block_eigensystem_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    selected_gauge_states: tuple[
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        ProjectionState,
    ] | None = None


@dataclass(frozen=True)
class ProjectSymmetryPreparation:
    """In-memory symmetry state resolved once for the active project run."""

    context: _ProjectionRunContext
    selected_candidate: ProjectGaugeAnchorCandidate
    gauge_report: GaugeAnchorReport


def _load_projection_run_config(
    cfg_path: str,
    *,
    developer_outputs: bool | None,
) -> _ProjectionRunConfig:
    cfg_path = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = normalize_case_config(yaml.safe_load(handle), config_path=cfg_path)

    material = cfg.get("material", {})
    plot_cfg = cfg.get("plot", {})
    project_cfg = cfg.get("project", {})
    symm_cfg = cfg.get("symm", {})
    diagnostics_cfg = symm_cfg.get("diagnostics", {})
    if isinstance(diagnostics_cfg, Mapping) and "projection_matrices" in diagnostics_cfg:
        raise ValueError("symm.diagnostics.projection_matrices is no longer supported; use symm.developer_outputs")
    if diagnostics_cfg not in ({}, None):
        raise ValueError("symm.diagnostics is no longer supported; use symm.developer_outputs")
    save_projection_diagnostics = (
        bool(developer_outputs)
        if developer_outputs is not None
        else _as_bool(symm_cfg.get("developer_outputs", False))
    )
    if not _as_bool(symm_cfg.get("enable", True)):
        raise ValueError("symm.enable is false")

    return _ProjectionRunConfig(
        cfg_path=cfg_path,
        cfg_dir=cfg_dir,
        material=material,
        plot_cfg=plot_cfg,
        project_cfg=project_cfg,
        symm_cfg=symm_cfg,
        save_projection_diagnostics=save_projection_diagnostics,
        valley=str(symm_cfg.get("valley", "K1")),
        spin=str(symm_cfg.get("spin", material.get("spin", "all"))).lower(),
        spin_sector_sewing=symm_cfg.get("spin_sector_sewing"),
        q_rotation_raw=plot_cfg.get("q_rotation_deg", symm_cfg.get("q_rotation_deg")),
        tolerance=float(symm_cfg.get("tolerance", 1.0e-2)),
        canonical_layout=(
            isinstance(cfg.get("case"), Mapping)
            and all(cfg["case"].get(key) not in (None, "") for key in ("profile", "q_shell", "output_root"))
        ),
    )


def _load_manifest_and_operation_requests(
    run_cfg: _ProjectionRunConfig,
    *,
    announce: bool = True,
) -> tuple[Path, Mapping[str, Any], list[dict[str, Any]]]:
    tapw_symmetry_dir = _resolve(run_cfg.symm_cfg.get("tapw_symmetry_dir"), run_cfg.cfg_dir)
    if tapw_symmetry_dir is None:
        raise ValueError("symm.tapw_symmetry_dir is required")
    packed_path = Path(tapw_symmetry_dir) / "representations.npz"
    if packed_path.exists():
        rep_root, manifest = _load_packed_tapw_symmetry_manifest(packed_path)
        operations_raw = run_cfg.symm_cfg.get("operations")
        inferred_operations = operations_raw in (None, [])
        source_operations = (
            _infer_symmetry_operations_from_manifest(manifest, run_cfg.valley)
            if inferred_operations
            else [str(op) for op in operations_raw]
        )
        if inferred_operations and announce:
            print(f"[kp symm] inferred symmetry operations: {', '.join(source_operations)}")
        operation_requests: list[dict[str, Any]] = []
        for operation in source_operations:
            canonical = _validate_operation_label(operation)
            operation_requests.append(
                {
                    "source": _source_manifest_operation_name(canonical),
                    "output": _output_operation_name(operation, canonical),
                    "requested": operation,
                }
            )
        return rep_root, manifest, operation_requests
    rep_root = Path(tapw_symmetry_dir) / "representations"
    manifest_path = rep_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Representation manifest missing: {manifest_path}; packed TAPW file also missing: {packed_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    operations_raw = run_cfg.symm_cfg.get("operations")
    inferred_operations = operations_raw in (None, [])
    source_operations = (
        _infer_symmetry_operations_from_manifest(manifest, run_cfg.valley)
        if inferred_operations
        else [str(op) for op in operations_raw]
    )
    if inferred_operations and announce:
        print(f"[kp symm] inferred symmetry operations: {', '.join(source_operations)}")

    operation_requests: list[dict[str, Any]] = []
    for operation in source_operations:
        canonical = _validate_operation_label(operation)
        operation_requests.append(
            {
                "source": _source_manifest_operation_name(canonical),
                "output": _output_operation_name(operation, canonical),
                "requested": operation,
            }
        )
    return rep_root, manifest, operation_requests


def _load_packed_tapw_symmetry_manifest(packed_path: Path) -> tuple[Path, Mapping[str, Any]]:
    payload = np.load(packed_path, allow_pickle=False)
    if "metadata_json" not in payload.files:
        raise ValueError(f"Packed TAPW symmetry file lacks metadata_json: {packed_path}")
    metadata = json.loads(str(payload["metadata_json"].item()))
    matrices_raw = metadata.get("matrices")
    if not isinstance(matrices_raw, list):
        raise ValueError(f"Packed TAPW symmetry metadata lacks matrices list: {packed_path}")

    matrices: list[dict[str, Any]] = []
    for row in matrices_raw:
        if not isinstance(row, Mapping):
            continue
        entry = dict(row)
        key = str(entry.get("key", entry.get("operation", entry.get("name", ""))))
        if not key:
            continue
        if entry.get("supported") is False:
            continue
        entry.setdefault("operation", key)
        entry.setdefault("name", entry["operation"])
        entry.setdefault("source_valley", entry.get("source_valley", entry.get("valley", "Gamma")))
        entry.setdefault("target_valley", entry.get("target_valley", entry["source_valley"]))
        entry["raw_h_operator_file"] = f"{packed_path.name}:{key}"
        entry["packed_matrix_file"] = packed_path.name
        entry["packed_matrix_key"] = key
        entry.setdefault("k_pairs", [[0, 0]])
        entry.setdefault("k_pairs_source", "packed_tapw_single_point_default")
        matrices.append(entry)

    manifest = dict(metadata)
    manifest["matrices"] = matrices
    return packed_path.parent, manifest


def _load_projection_arrays_and_layout(
    run_cfg: _ProjectionRunConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, float | None, list[list[int]], list[int], int, list[int]]:
    hamk_file = _resolve(run_cfg.material.get("hamk_file"), run_cfg.cfg_dir)
    qset1_file = _resolve(run_cfg.material.get("qset1_file"), run_cfg.cfg_dir)
    qset2_file = _resolve(run_cfg.material.get("qset2_file"), run_cfg.cfg_dir)
    if hamk_file is None or qset1_file is None or qset2_file is None:
        raise ValueError("material.hamk_file/qset1_file/qset2_file are required")
    hamk = _load_hamk_with_energy_unit(hamk_file, run_cfg.material, mmap_mode="r")
    q1, q2 = load_Q_sets(qset1_file, qset2_file)
    if q2 is None:
        raise ValueError("K-valley symmetry projection requires qset2_file")

    hamk3d = hamk if hamk.ndim == 3 else hamk[np.newaxis, ...]
    q_count = int(len(q1))
    mode = str(run_cfg.project_cfg.get("mode", "K1")).lower()
    method = _downfold_method(run_cfg.project_cfg)
    e_ref = _e_ref(run_cfg.project_cfg)
    if method in {"fixed_schur", "linearized_lowdin"} and e_ref is None:
        raise ValueError(f"project.downfold_method={method!r} requires project.e_ref")
    nlow_state_list = _normalize_nlow_state_list(run_cfg.project_cfg)
    num_layer_list, orb0, num_orb_per_layer_list = _orbital_layout_from_material(
        run_cfg.material,
        np.asarray(hamk3d[0]),
        q_count,
    )
    return (
        hamk3d,
        q1,
        q2,
        mode,
        method,
        e_ref,
        nlow_state_list,
        num_layer_list,
        orb0,
        num_orb_per_layer_list,
    )


def _resolve_operation_entries(
    *,
    manifest: Mapping[str, Any],
    valley: str,
    operation_requests: Sequence[Mapping[str, Any]],
    nk: int,
    default_k_index: int,
) -> tuple[dict[str, dict[str, Any]], set[tuple[int, int]]]:
    operation_entries: dict[str, dict[str, Any]] = {}
    all_pairs: set[tuple[int, int]] = set()
    for request in operation_requests:
        operation = request["source"]
        output_operation = request["output"]
        entry = _operation_entry(manifest, valley, operation)
        pairs = _pairs_from_entry(entry, nk, default_k_index=default_k_index)
        for target_idx, source_idx in pairs:
            if target_idx < 0 or target_idx >= nk or source_idx < 0 or source_idx >= nk:
                raise IndexError(
                    f"{output_operation} source/target k unavailable: "
                    f"target={target_idx}, source={source_idx}, nk={nk}"
                )
            all_pairs.add((target_idx, source_idx))
        entry["_pairs"] = pairs
        operation_entries[output_operation] = entry
    return operation_entries, all_pairs


def _source_target_hamk_by_k(
    *,
    hamk3d: np.ndarray,
    required_k: Sequence[int],
    spin: str,
    spin_sector_sewing: Any,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if spin_sector_sewing is None:
        hamk_source_by_k = {
            k_index: _spin_slice_hamk(np.asarray(hamk3d[k_index], dtype=np.complex128), spin)
            for k_index in required_k
        }
        return hamk_source_by_k, hamk_source_by_k

    source_spin, target_spin = _spin_route_endpoints(spin, str(spin_sector_sewing))
    hamk_source_by_k = {
        k_index: _spin_slice_hamk(
            np.asarray(hamk3d[k_index], dtype=np.complex128), source_spin
        )
        for k_index in required_k
    }
    hamk_target_by_k = {
        k_index: _spin_slice_hamk(
            np.asarray(hamk3d[k_index], dtype=np.complex128), target_spin
        )
        for k_index in required_k
    }
    return hamk_source_by_k, hamk_target_by_k


def _build_operation_payloads(
    *,
    run_cfg: _ProjectionRunConfig,
    rep_root: Path,
    operation_requests: Sequence[Mapping[str, Any]],
    operation_entries: Mapping[str, Mapping[str, Any]],
    hamk_source_by_k: Mapping[int, np.ndarray],
    hamk_target_by_k: Mapping[int, np.ndarray],
    full_dim: int,
    validate_full_space_covariance: bool = True,
) -> dict[str, dict[str, Any]]:
    operation_payloads: dict[str, dict[str, Any]] = {}
    for request in operation_requests:
        output_operation = request["output"]
        entry = operation_entries[output_operation]
        antiunitary = _as_bool(entry.get("antiunitary", False))
        filename = _optional_representation_filename(entry)
        action = _build_action_representation(
            operation=output_operation,
            entry=entry,
            rep_root=rep_root,
            filename=filename,
            antiunitary=antiunitary,
            spin=run_cfg.spin,
            full_dim=full_dim,
            tolerance=run_cfg.tolerance,
            spin_sector_sewing=None if run_cfg.spin_sector_sewing is None else str(run_cfg.spin_sector_sewing),
        )
        full_pair_rows = []
        for target_idx, source_idx in entry["_pairs"]:
            full_residual = None
            if validate_full_space_covariance:
                full_residual = _full_space_covariance_residual(
                    d_full=action.matrix,
                    h_target=hamk_target_by_k[target_idx],
                    h_source=hamk_source_by_k[source_idx],
                    antiunitary=antiunitary,
                )
                if full_residual > run_cfg.tolerance:
                    raise ValueError(
                        f"{output_operation} k=({target_idx},{source_idx}) full-space covariance residual "
                        f"{full_residual:.3e} exceeds tolerance. The TAPW representation is not "
                        "compatible with the KP input hamk/source-k rule."
                    )
            full_pair_rows.append(
                {
                    "target_k_index": int(target_idx),
                    "source_k_index": int(source_idx),
                    "full_space_covariance_residual": full_residual,
                }
            )
        operation_payloads[output_operation] = {
            "entry": entry,
            "antiunitary": antiunitary,
            "filename": filename,
            "action": action,
            "full_pair_rows": full_pair_rows,
        }
    return operation_payloads


def _build_projection_run_context(
    run_cfg: _ProjectionRunConfig,
    *,
    create_output_dir: bool = True,
    validate_full_space_covariance: bool = True,
) -> _ProjectionRunContext:
    rep_root, manifest, operation_requests = _load_manifest_and_operation_requests(run_cfg)
    (
        hamk3d,
        q1,
        q2,
        mode,
        method,
        e_ref,
        nlow_state_list,
        num_layer_list,
        orb0,
        num_orb_per_layer_list,
    ) = _load_projection_arrays_and_layout(run_cfg)

    nk = int(hamk3d.shape[0])
    default_k_index = int(run_cfg.plot_cfg.get("hamk_index", 0))
    operation_entries, all_pairs = _resolve_operation_entries(
        manifest=manifest,
        valley=run_cfg.valley,
        operation_requests=operation_requests,
        nk=nk,
        default_k_index=default_k_index,
    )
    spin_sector_sewing, spin_route_inference = _resolve_spin_sector_sewing(
        spin=run_cfg.spin,
        rep_root=rep_root,
        operation_entries=operation_entries,
        explicit_route=run_cfg.spin_sector_sewing,
    )
    run_cfg = replace(run_cfg, spin_sector_sewing=spin_sector_sewing)
    if run_cfg.q_rotation_raw is None:
        q_rotation_deg, frame_inference = _infer_model_frame_rotation(q1, q2, operation_entries)
    else:
        q_rotation_deg = float(run_cfg.q_rotation_raw)
        frame_inference = {
            "source": "explicit_config",
            "field": "plot.q_rotation_deg" if "q_rotation_deg" in run_cfg.plot_cfg else "symm.q_rotation_deg",
        }
    required_k = sorted({idx for pair in all_pairs for idx in pair})
    if not required_k:
        raise ValueError("No source/target k points requested by symmetry operations")
    hamk_source_by_k, hamk_target_by_k = _source_target_hamk_by_k(
        hamk3d=hamk3d,
        required_k=required_k,
        spin=run_cfg.spin,
        spin_sector_sewing=run_cfg.spin_sector_sewing,
    )
    full_dim = int(hamk_source_by_k[required_k[0]].shape[0])
    output_dir = Path(_resolve(run_cfg.symm_cfg.get("output_dir", "symm_project"), run_cfg.cfg_dir) or "symm_project")
    if create_output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    operation_payloads = _build_operation_payloads(
        run_cfg=run_cfg,
        rep_root=rep_root,
        operation_requests=operation_requests,
        operation_entries=operation_entries,
        hamk_source_by_k=hamk_source_by_k,
        hamk_target_by_k=hamk_target_by_k,
        full_dim=full_dim,
        validate_full_space_covariance=validate_full_space_covariance,
    )
    q_model1, q_model2 = _model_q_sets(q1, q2, rotation_deg=q_rotation_deg)

    return _ProjectionRunContext(
        config=run_cfg,
        rep_root=rep_root,
        operation_requests=operation_requests,
        spin_route_inference=spin_route_inference,
        q1=q1,
        q2=q2,
        q_count=int(len(q1)),
        mode=mode,
        method=method,
        e_ref=e_ref,
        nlow_state_list=nlow_state_list,
        num_layer_list=num_layer_list,
        orb0=orb0,
        num_orb_per_layer_list=num_orb_per_layer_list,
        required_k=required_k,
        hamk_source_by_k=hamk_source_by_k,
        hamk_target_by_k=hamk_target_by_k,
        full_dim=full_dim,
        output_dir=output_dir,
        q_model1=q_model1,
        q_model2=q_model2,
        q_rotation_deg=q_rotation_deg,
        frame_inference=frame_inference,
        operation_payloads=operation_payloads,
    )


def _resolve_projection_gauge_candidates(ctx: _ProjectionRunContext) -> Sequence[ProjectGaugeAnchorCandidate]:
    run_cfg = ctx.config
    resolver_spin = _spin_label_for_sliced_block(run_cfg.spin) if run_cfg.spin_sector_sewing is None else "up"
    reference_k = int(run_cfg.plot_cfg.get("hamk_index", 0))
    if reference_k not in ctx.hamk_source_by_k:
        reference_k = ctx.required_k[0]
    q2_for_projection = ctx.q2 if ctx.q2 is not None else ctx.q1
    norb_fix_list = run_cfg.project_cfg["norb_fix_list"] if "norb_fix_list" in run_cfg.project_cfg else None
    return resolve_project_gauge_anchor_candidates(
        ctx.hamk_source_by_k[reference_k],
        ctx.q_count,
        ctx.orb0,
        ctx.num_layer_list,
        spin=resolver_spin,
        Qlayer_list=[[ctx.q1], [q2_for_projection]],
        num_orb_per_layer_list=ctx.num_orb_per_layer_list,
        nlow_state_list=ctx.nlow_state_list,
        norb_fix_list=norb_fix_list,
        gauge_config=run_cfg.project_cfg.get("gauge"),
        mode=ctx.mode,
        eigensystem_cache=ctx.block_eigensystem_cache,
    )


def _states_for_resolved_anchors(
    ctx: _ProjectionRunContext,
    resolved_norb_fix_list: list[Any],
    *,
    projection_method: str | None = None,
    compute_heff: bool = True,
    reference_states: tuple[
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        ProjectionState,
    ]
    | None = None,
    reuse_diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    ProjectionState,
]:
    run_cfg = ctx.config
    requested_method = ctx.method if projection_method is None else projection_method
    states_local: dict[int, ProjectionState] = {}
    source_states_local: dict[int, ProjectionState] = {}
    target_states_local: dict[int, ProjectionState] = {}

    def state_for_k(
        hamk: np.ndarray,
        *,
        block_spin: str,
        reference: ProjectionState | None,
        k_index: int,
        role: str,
    ) -> ProjectionState:
        common = {
            "orb0": ctx.orb0,
            "num_layer_list": ctx.num_layer_list,
            "num_orb_per_layer_list": ctx.num_orb_per_layer_list,
            "spin": block_spin,
            "mode": ctx.mode,
            "nlow_state_list": ctx.nlow_state_list,
            "norb_fix_list": resolved_norb_fix_list,
            "e_ref": ctx.e_ref,
            "project_cfg": run_cfg.project_cfg,
            "eigensystem_cache": getattr(ctx, "block_eigensystem_cache", None),
        }
        if reference is None:
            return _projectors_for_k(
                hamk,
                ctx.q1,
                ctx.q2,
                method=requested_method,
                compute_heff=compute_heff,
                **common,
            )

        basis_state = _projectors_for_k(
            hamk,
            ctx.q1,
            ctx.q2,
            method="first_order",
            **common,
        )
        reference_u = np.asarray(reference.u_low, dtype=np.complex128)
        candidate_u = np.asarray(basis_state.u_low, dtype=np.complex128)
        try:
            if reference_u.shape != candidate_u.shape:
                raise ValueError(
                    f"projector shape changed: {reference_u.shape} != {candidate_u.shape}"
                )
            transform = reference_u.conj().T @ candidate_u
            identity = np.eye(transform.shape[0], dtype=np.complex128)
            unitarity_residual = float(
                np.linalg.norm(transform.conj().T @ transform - identity, ord="fro")
                / np.sqrt(max(1, transform.shape[0]))
            )
            subspace_residual = float(
                np.linalg.norm(candidate_u - reference_u @ transform, ord="fro")
                / np.sqrt(max(1, transform.shape[0]))
            )
            certification_tolerance = float(
                max(
                    1.0e-12,
                    64.0
                    * np.finfo(np.float64).eps
                    * max(1, transform.shape[0]),
                )
            )
            if max(unitarity_residual, subspace_residual) > certification_tolerance:
                raise ValueError(
                    "candidate low-energy subspace is not a certified unitary frame of "
                    f"the reference: unitarity={unitarity_residual:.3e}, "
                    f"subspace={subspace_residual:.3e}, "
                    f"tolerance={certification_tolerance:.3e}"
                )
            transformed_heff = transform.conj().T @ reference.heff @ transform
            transformed_heff = 0.5 * (transformed_heff + transformed_heff.conj().T)
            if reuse_diagnostics is not None:
                reuse_diagnostics.append(
                    {
                        "status": "reused",
                        "k_index": int(k_index),
                        "role": role,
                        "unitarity_residual": unitarity_residual,
                        "subspace_residual": subspace_residual,
                        "certification_tolerance": certification_tolerance,
                    }
                )
            return ProjectionState(
                hamk=np.asarray(hamk, dtype=np.complex128),
                heff=np.asarray(transformed_heff, dtype=np.complex128),
                u_low=candidate_u,
            )
        except ValueError as exc:
            if reuse_diagnostics is not None:
                reuse_diagnostics.append(
                    {
                        "status": "fallback",
                        "k_index": int(k_index),
                        "role": role,
                        "reason": str(exc),
                    }
                )
            return _projectors_for_k(
                hamk,
                ctx.q1,
                ctx.q2,
                method=requested_method,
                **common,
            )

    reference_source = None if reference_states is None else reference_states[1]
    reference_target = None if reference_states is None else reference_states[2]
    if run_cfg.spin_sector_sewing is None:
        block_spin = _spin_label_for_sliced_block(run_cfg.spin)
        for k_index in ctx.required_k:
            states_local[k_index] = state_for_k(
                ctx.hamk_source_by_k[k_index],
                block_spin=block_spin,
                reference=None if reference_source is None else reference_source[k_index],
                k_index=k_index,
                role="source_target",
            )
        return states_local, states_local, states_local, states_local[ctx.required_k[0]]

    _spin_route_endpoints(run_cfg.spin, str(run_cfg.spin_sector_sewing))
    for k_index in ctx.required_k:
        source_states_local[k_index] = state_for_k(
            ctx.hamk_source_by_k[k_index],
            block_spin="up",
            reference=None if reference_source is None else reference_source[k_index],
            k_index=k_index,
            role="source",
        )
        target_states_local[k_index] = state_for_k(
            ctx.hamk_target_by_k[k_index],
            block_spin="up",
            reference=None if reference_target is None else reference_target[k_index],
            k_index=k_index,
            role="target",
        )
    return {}, source_states_local, target_states_local, source_states_local[ctx.required_k[0]]


def _basis_states_for_resolved_anchors(
    ctx: _ProjectionRunContext,
    resolved_norb_fix_list: list[Any],
) -> tuple[
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    ProjectionState,
]:
    """Build candidate projector frames without running the configured downfold."""

    return _states_for_resolved_anchors(
        ctx,
        resolved_norb_fix_list,
        projection_method="first_order",
        compute_heff=False,
    )


def _configured_basis_states_for_resolved_anchors(
    ctx: _ProjectionRunContext,
    resolved_norb_fix_list: list[Any],
) -> tuple[
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    ProjectionState,
]:
    """Build the production-method frame without evaluating its downfold."""

    return _states_for_resolved_anchors(
        ctx,
        resolved_norb_fix_list,
        projection_method=ctx.method,
        compute_heff=False,
    )


def _projection_basis_frame_residual(
    candidate_states: tuple[
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        ProjectionState,
    ],
    configured_states: tuple[
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        ProjectionState,
    ],
) -> float:
    residual = 0.0
    for candidate_by_k, configured_by_k in zip(candidate_states[1:3], configured_states[1:3]):
        if set(candidate_by_k) != set(configured_by_k):
            return float("inf")
        for k_index in candidate_by_k:
            candidate_u = np.asarray(candidate_by_k[k_index].u_low, dtype=np.complex128)
            configured_u = np.asarray(configured_by_k[k_index].u_low, dtype=np.complex128)
            if candidate_u.shape != configured_u.shape:
                return float("inf")
            scale = np.sqrt(max(1, candidate_u.shape[1]))
            residual = max(
                residual,
                float(np.linalg.norm(candidate_u - configured_u, ord="fro") / scale),
            )
    return residual


def _certified_frame_equivalence(
    project_frame: np.ndarray,
    recomputed_frame: np.ndarray,
) -> tuple[float, float]:
    """Compare independently derived certified frames at floating-point precision."""

    project = np.asarray(project_frame, dtype=np.complex128)
    recomputed = np.asarray(recomputed_frame, dtype=np.complex128)
    if project.shape != recomputed.shape or project.ndim != 2:
        return float("inf"), 0.0
    scale = np.sqrt(max(1, project.shape[1]))
    residual = float(np.linalg.norm(project - recomputed, ord="fro") / scale)
    dimension = max(project.shape, default=1)
    bound = float(max(1.0e-12, 64.0 * dimension * np.finfo(np.float64).eps))
    return residual, bound


def _candidate_symmetry_metrics(
    ctx: _ProjectionRunContext,
    candidate: ProjectGaugeAnchorCandidate,
    *,
    candidate_states: tuple[
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        ProjectionState,
    ]
    | None = None,
    state_cache: dict[
        str,
        tuple[
            dict[int, ProjectionState],
            dict[int, ProjectionState],
            dict[int, ProjectionState],
            ProjectionState,
        ],
    ]
    | None = None,
) -> GaugeCandidateSymmetryMetrics:
    run_cfg = ctx.config
    candidate_id = str(candidate.candidate_id)
    diagnostic_stage = "candidate_setup"
    try:
        _validate_project_layer_lists(
            ctx.nlow_state_list,
            candidate.resolved_norb_fix_list,
            num_layer_list=ctx.num_layer_list,
            context=f"auto gauge candidate {candidate.candidate_id}",
        )
        if candidate_states is None:
            candidate_states = _basis_states_for_resolved_anchors(
                ctx,
                candidate.resolved_norb_fix_list,
            )
        if state_cache is not None:
            state_cache[candidate_id] = candidate_states
        _states, source_states_candidate, target_states_candidate, first_state_candidate = candidate_states
        low_dim_candidate = int(first_state_candidate.u_low.shape[1])
        n_orb_candidate = _sector_orbital_counts(
            ctx.q_model1,
            ctx.q_model2,
            ctx.nlow_state_list,
            low_dim=low_dim_candidate,
            num_layer_list=ctx.num_layer_list,
        )
        spin_convention_candidate = _spin_convention_for_exactification(
            run_cfg.spin,
            n_orb_candidate,
            valley=run_cfg.valley,
        )
        raw_candidate_matrices: dict[str, np.ndarray] = {}
        operation_records: list[dict[str, Any]] = []
        for request in ctx.operation_requests:
            output_operation = request["output"]
            canonical_name = _canonical_internal_operation_name(output_operation)
            payload = ctx.operation_payloads[output_operation]
            entry = payload["entry"]
            antiunitary = bool(payload["antiunitary"])
            action = payload["action"]
            source_action_metadata = _operation_action_metadata(entry, output_operation, antiunitary)
            model_action_metadata = _model_action_metadata(
                source_action_metadata,
                valley=run_cfg.valley,
                operation=output_operation,
                rotation_deg=ctx.q_rotation_deg,
            )
            raw_mats, _polar_mats, _pair_rows = _project_operation(
                operation=output_operation,
                antiunitary=antiunitary,
                d_full=action.matrix,
                states=source_states_candidate,
                pairs=entry["_pairs"],
                tolerance=run_cfg.tolerance,
                compute_polar=False,
                target_states=target_states_candidate,
                source_states=source_states_candidate,
                enforce_heff_covariance=False,
                compute_heff_covariance=False,
            )
            raw_candidate_matrices[canonical_name] = np.asarray(raw_mats[0], dtype=np.complex128)
            operation_records.append(
                {
                    "name": canonical_name,
                    "operation": output_operation,
                    "antiunitary": antiunitary,
                    "matrix_file": f"{output_operation}_low_raw.npy",
                    "source_matrix_role": "raw_h_sewing_action",
                    "source_gauge": "raw_saved_TAPW",
                    "target_role": "continuum_internal_rep",
                    **model_action_metadata,
                    "source_action": source_action_metadata,
                    "model_action": model_action_metadata,
                    "declared_model_action": model_action_metadata,
                    "group_relations": [
                        _operation_power_relation(canonical_name, spin_convention=spin_convention_candidate)
                    ],
                }
            )
        bM_candidates = bM_candidates_from_q_distances(ctx.q_model1, ctx.q_model2)
        if bM_candidates:
            bM1_candidate, bM2_candidate = canonical_bM_pair_from_candidates(bM_candidates, angle_deg=60.0)
        else:
            bM1_candidate = np.array([1.0, 0.0], dtype=float)
            bM2_candidate = np.array([0.5, float(np.sqrt(3.0) / 2.0)], dtype=float)
        sectors_candidate = sectors_with_q_offsets(
            [
                {"name": "L1", "qset": "qset1", "n_orb": int(n_orb_candidate[0])},
                {"name": "L2", "qset": "qset2", "n_orb": int(n_orb_candidate[1])},
            ],
            Q_set1=ctx.q_model1,
            Q_set2=ctx.q_model2,
            bM1=bM1_candidate,
            bM2=bM2_candidate,
        )
        exact_config_candidate = _kp_symm_exactification_config(
            _merged_exactification_overrides(
                run_cfg.valley,
                run_cfg.symm_cfg.get("exactification"),
                symmetry_tolerance=run_cfg.tolerance,
                operation_actions=run_cfg.symm_cfg.get("operation_actions"),
            )
        )
        diagnostic_stage = "initial_exactification"
        exact_matrices_candidate, exact_reports_candidate = exactify_loaded_symmetry_source(
            loaded_metadata={"operations": operation_records},
            matrices=raw_candidate_matrices,
            Q_set1=ctx.q_model1,
            Q_set2=ctx.q_model2,
            sectors=sectors_candidate,
            n_orb=n_orb_candidate,
            bM1=bM1_candidate,
            bM2=bM2_candidate,
            raw_config={"exactification": exact_config_candidate},
            rotation_deg=0.0,
            output_dir=None,
        )
        diagnostic_stage = "basis_frame_derivation"
        adapted_frame_candidate, _canonical_candidate_matrices = _derive_projection_basis_frame(
            exact_matrices_candidate,
            exact_reports_candidate,
            operations=operation_records,
            sectors=[
                {"name": "L1", "n_orb": int(n_orb_candidate[0]), "n_q": int(len(ctx.q_model1))},
                {"name": "L2", "n_orb": int(n_orb_candidate[1]), "n_q": int(len(ctx.q_model2))},
            ],
            # The input matrices have already been exactified.  The frame
            # solver therefore needs an algebraic floating-point threshold,
            # not the (possibly percent-level) source-symmetry acceptance
            # tolerance from the user configuration.
            tolerance=min(max(float(run_cfg.tolerance), 1.0e-12), 1.0e-8),
            raw_source_matrices=_basis_frame_raw_source_matrices(
                raw_candidate_matrices,
                spin_sector_sewing=run_cfg.spin_sector_sewing,
            ),
        )
        distances: dict[str, float] = {}
        support_off: dict[str, float] = {}
        phase_branch: dict[str, float] = {}
        for op_name, report in exact_reports_candidate.items():
            report_payload = report.get("report", {}) if isinstance(report, Mapping) else {}
            support_payload = report.get("support_diagnostics", {}) if isinstance(report, Mapping) else {}
            if isinstance(report_payload, Mapping) and "distance_mod_global_phase" in report_payload:
                distances[str(op_name)] = float(report_payload["distance_mod_global_phase"])
            if isinstance(support_payload, Mapping) and "off_support_rel" in support_payload:
                support_off[str(op_name)] = float(support_payload["off_support_rel"])
            if isinstance(report_payload, Mapping) and "phase_std_deg" in report_payload:
                phase_branch[str(op_name)] = float(report_payload["phase_std_deg"])
        return GaugeCandidateSymmetryMetrics(
            candidate_id=candidate_id,
            exactification_distance_by_op=distances,
            phase_branch_distance_by_op=phase_branch,
            support_off_by_op=support_off,
            metadata={
                "status": "evaluated",
                "candidate_priority": int(candidate.priority),
                "sigma_min": candidate.report.gauge_anchor_quality.get("sigma_min"),
                "condition_number": candidate.report.gauge_anchor_quality.get("condition_number"),
                "resolved_norb_fix_list": candidate.resolved_norb_fix_list,
                "symmetry_adapted_frame": adapted_frame_candidate.artifact(),
                "candidate_projection": {
                    "status": "basis_only_no_downfold",
                    "projection_method": "first_order",
                },
            },
        )
    except Exception as exc:
        return GaugeCandidateSymmetryMetrics(
            candidate_id=candidate_id,
            exactification_distance_by_op={},
            metadata={
                "status": "failed",
                "candidate_priority": int(candidate.priority),
                "error": f"{diagnostic_stage}: {exc}",
                "resolved_norb_fix_list": candidate.resolved_norb_fix_list,
            },
        )


def _basis_frame_raw_source_matrices(
    raw_matrices: Mapping[str, np.ndarray],
    *,
    spin_sector_sewing: Any,
) -> Mapping[str, np.ndarray] | None:
    """Return raw matrices only when they act within one Hilbert space."""

    if spin_sector_sewing is not None:
        return None
    return raw_matrices


def _select_fixed_target_source_matrices(
    raw_matrices: Mapping[str, np.ndarray],
    *,
    exactified_internal_matrices: Mapping[str, np.ndarray] | None,
    spin_sector_sewing: Any,
) -> tuple[Mapping[str, np.ndarray], str]:
    """Choose actions that may be compared with an internal canonical target.

    A cross-sector sewing block maps between two source Hilbert spaces.  It is
    valid covariance evidence, but it is not itself an internal representation
    matrix and therefore cannot be sent directly to fixed-target certification.
    """

    if spin_sector_sewing is None:
        return raw_matrices, "raw_projected_internal_action"
    if exactified_internal_matrices is None:
        raise ValueError(
            "cross-sector fixed-target certification requires joint-exactified "
            "internal actions"
        )
    if set(exactified_internal_matrices) != set(raw_matrices):
        raise ValueError(
            "joint-exactified internal actions do not match raw sewing operations"
        )
    return (
        exactified_internal_matrices,
        "joint_exactified_cross_sector_internal_action",
    )


def _select_projection_gauge(
    ctx: _ProjectionRunContext,
    gauge_candidates: Sequence[ProjectGaugeAnchorCandidate],
):
    selected_gauge_candidate = gauge_candidates[0]
    gauge_report = selected_gauge_candidate.report
    if gauge_report.gauge_mode != "auto_scdm":
        return selected_gauge_candidate, gauge_report

    basis_state_cache: dict[
        str,
        tuple[
            dict[int, ProjectionState],
            dict[int, ProjectionState],
            dict[int, ProjectionState],
            ProjectionState,
        ],
    ] = {}
    candidate_metrics = [
        _candidate_symmetry_metrics(ctx, candidate, state_cache=basis_state_cache)
        for candidate in gauge_candidates
    ]
    validation_cfg = ctx.config.symm_cfg.get("gauge_validation", {})
    if not isinstance(validation_cfg, Mapping):
        validation_cfg = {}
    selected_gauge_candidate, validation_decision = _select_validated_auto_gauge_candidate(
        gauge_candidates,
        candidate_metrics,
        max_exactification_distance=float(validation_cfg.get("max_exactification_distance", ctx.config.tolerance)),
    )
    selected_id = str(selected_gauge_candidate.candidate_id)
    configured_selected_states = _configured_basis_states_for_resolved_anchors(
        ctx,
        selected_gauge_candidate.resolved_norb_fix_list,
    )
    frame_residual = _projection_basis_frame_residual(
        basis_state_cache[selected_id],
        configured_selected_states,
    )
    frame_tolerance = float(
        max(
            1.0e-10,
            64.0 * np.finfo(np.float64).eps * max(1, ctx.full_dim),
        )
    )
    frame_certification = {
        "status": "certified" if frame_residual <= frame_tolerance else "configured_basis_fallback",
        "frame_residual": frame_residual,
        "tolerance": frame_tolerance,
        "configured_method": ctx.method,
    }
    if frame_residual > frame_tolerance:
        configured_state_cache = {
            str(candidate.candidate_id): _configured_basis_states_for_resolved_anchors(
                ctx,
                candidate.resolved_norb_fix_list,
            )
            for candidate in gauge_candidates
        }
        candidate_metrics = [
            _candidate_symmetry_metrics(
                ctx,
                candidate,
                candidate_states=configured_state_cache[str(candidate.candidate_id)],
            )
            for candidate in gauge_candidates
        ]
        candidate_metrics = [
            replace(
                metric,
                metadata={
                    **dict(metric.metadata),
                    "candidate_projection": {
                        "status": "configured_basis_fallback",
                        "projection_method": ctx.method,
                    },
                },
            )
            for metric in candidate_metrics
        ]
        selected_gauge_candidate, validation_decision = _select_validated_auto_gauge_candidate(
            gauge_candidates,
            candidate_metrics,
            max_exactification_distance=float(
                validation_cfg.get("max_exactification_distance", ctx.config.tolerance)
            ),
        )
    selected_metric = next(
        metric for metric in candidate_metrics if metric.candidate_id == selected_gauge_candidate.candidate_id
    )
    selected_metric = replace(
        selected_metric,
        metadata={
            **dict(selected_metric.metadata),
            "production_basis_certification": frame_certification,
        },
    )
    candidate_metrics = [
        selected_metric if metric.candidate_id == selected_metric.candidate_id else metric
        for metric in candidate_metrics
    ]
    # Candidate scoring intentionally skips the configured downfold.  The
    # selected frame is downfolded once by the normal production path.
    ctx.selected_gauge_states = None
    gauge_report = replace(
        selected_gauge_candidate.report,
        symmetry_closure_quality={
            "status": "validated",
            "selection_policy": "symmetry_exactification_residual",
            "selected_candidate_id": selected_gauge_candidate.candidate_id,
            "symmetry_adapted_frame": selected_metric.metadata.get("symmetry_adapted_frame"),
            "candidate_rankings": validation_decision.rankings,
            "metrics": [
                {
                    "candidate_id": metric.candidate_id,
                    "exactification_distance_by_op": dict(metric.exactification_distance_by_op),
                    "phase_branch_distance_by_op": dict(metric.phase_branch_distance_by_op),
                    "support_off_by_op": dict(metric.support_off_by_op),
                    "metadata": dict(metric.metadata),
                }
                for metric in candidate_metrics
            ],
        },
    )
    return selected_gauge_candidate, gauge_report


def _resolve_validated_projection_gauge(
    ctx: _ProjectionRunContext,
) -> tuple[ProjectGaugeAnchorCandidate, GaugeAnchorReport]:
    gauge_candidates = _resolve_projection_gauge_candidates(ctx)
    selected_gauge_candidate, gauge_report = _select_projection_gauge(ctx, gauge_candidates)
    _validate_project_layer_lists(
        ctx.nlow_state_list,
        selected_gauge_candidate.resolved_norb_fix_list,
        num_layer_list=ctx.num_layer_list,
        context="project",
    )
    return selected_gauge_candidate, gauge_report


def _resolved_gauge_for_symmetry(
    ctx: _ProjectionRunContext,
    *,
    gauge_report: GaugeAnchorReport | None = None,
) -> tuple[ProjectGaugeAnchorCandidate, GaugeAnchorReport]:
    """Use the gauge already selected by project, or resolve it for legacy callers."""

    if gauge_report is None:
        return _resolve_validated_projection_gauge(ctx)
    closure = gauge_report.symmetry_closure_quality
    selected_id = (
        str(closure.get("selected_candidate_id", "project_resolved_gauge"))
        if isinstance(closure, Mapping)
        else "project_resolved_gauge"
    )
    return (
        ProjectGaugeAnchorCandidate(
            candidate_id=selected_id,
            resolved_norb_fix_list=list(gauge_report.resolved_norb_fix_list),
            report=gauge_report,
            priority=0,
        ),
        gauge_report,
    )


def infer_project_spin_route_from_config(
    cfg_path: str,
    *,
    project_config: Mapping[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve the source/target spin route without constructing projectors."""

    run_cfg = _load_projection_run_config(cfg_path, developer_outputs=None)
    if project_config is not None:
        run_cfg = replace(run_cfg, project_cfg=dict(project_config))
    rep_root, manifest, operation_requests = _load_manifest_and_operation_requests(
        run_cfg,
        announce=False,
    )
    hamk_file = _resolve(run_cfg.material.get("hamk_file"), run_cfg.cfg_dir)
    if hamk_file is None:
        raise ValueError("material.hamk_file is required")
    hamk = _load_hamk_with_energy_unit(hamk_file, run_cfg.material, mmap_mode="r")
    nk = int(hamk.shape[0]) if hamk.ndim == 3 else 1
    operation_entries, _all_pairs = _resolve_operation_entries(
        manifest=manifest,
        valley=run_cfg.valley,
        operation_requests=operation_requests,
        nk=nk,
        default_k_index=int(run_cfg.plot_cfg.get("hamk_index", 0)),
    )
    return _resolve_spin_sector_sewing(
        spin=run_cfg.spin,
        rep_root=rep_root,
        operation_entries=operation_entries,
        explicit_route=run_cfg.spin_sector_sewing,
    )


def resolve_symmetry_validated_project_gauge(
    cfg_path: str,
    *,
    project_config: Mapping[str, Any] | None = None,
    developer_outputs: bool | None = None,
) -> GaugeAnchorReport:
    """Resolve auto-gauge anchors against source symmetry without writing KP artifacts."""

    return prepare_symmetry_validated_project_gauge(
        cfg_path,
        project_config=project_config,
        developer_outputs=developer_outputs,
    ).gauge_report


def prepare_symmetry_validated_project_gauge(
    cfg_path: str,
    *,
    project_config: Mapping[str, Any] | None = None,
    developer_outputs: bool | None = None,
) -> ProjectSymmetryPreparation:
    """Resolve Auto-gauge and retain the loaded source symmetry for package preparation."""

    run_cfg = _load_projection_run_config(cfg_path, developer_outputs=developer_outputs)
    if project_config is not None:
        run_cfg = replace(run_cfg, project_cfg=dict(project_config))
    ctx = _build_projection_run_context(
        run_cfg,
        create_output_dir=False,
        validate_full_space_covariance=False,
    )
    selected_gauge_candidate, gauge_report = _resolve_validated_projection_gauge(ctx)
    return ProjectSymmetryPreparation(
        context=ctx,
        selected_candidate=selected_gauge_candidate,
        gauge_report=gauge_report,
    )


def load_project_prepared_symmetry_package(
    cfg_path: str,
    *,
    project_config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load a symmetry package prepared by the matching project artifacts."""

    run_cfg = _load_projection_run_config(cfg_path, developer_outputs=None)
    if project_config is not None:
        run_cfg = replace(run_cfg, project_cfg=dict(project_config))
    output_dir = Path(
        _resolve(run_cfg.symm_cfg.get("output_dir", "symm_project"), run_cfg.cfg_dir)
        or "symm_project"
    )
    packed_path = output_dir / "representations.npz"
    if not packed_path.exists():
        return None
    with np.load(packed_path, allow_pickle=False) as payload:
        if "__metadata_json__" not in payload.files:
            return None
        metadata = json.loads(str(np.asarray(payload["__metadata_json__"]).item()))
    if not isinstance(metadata, Mapping):
        return None
    preparation = metadata.get("package_preparation", {})
    if not isinstance(preparation, Mapping):
        return None
    if preparation.get("owner") != "kp_project" or preparation.get("status") != "prepared":
        return None
    project_dir = _resolve(run_cfg.project_cfg.get("out_dir"), run_cfg.cfg_dir)
    if project_dir is None:
        raise ValueError("project.out_dir is required to reuse a project-prepared symmetry package")
    actual_identity = _load_project_artifact_identity(project_dir)
    cached_identity = metadata.get("artifact_identity", {})
    if not isinstance(cached_identity, Mapping):
        raise ValueError("project-prepared symmetry package is missing artifact_identity")
    require_matching_identity(
        cached_identity,
        actual_identity,
        PROJECTION_ARTIFACT_IDENTITY_FIELDS,
        "project-prepared symmetry package",
    )
    return dict(metadata)


def _gauge_report_from_persisted_handoff(
    handoff: ExplicitLegacyBasisSpec,
) -> GaugeAnchorReport:
    """Build diagnostic summary data without re-running gauge selection."""

    frame_artifact = (
        None
        if handoff.frame_artifact is None
        else dict(handoff.frame_artifact)
    )
    return GaugeAnchorReport(
        gauge_mode=handoff.gauge_mode,
        resolved_norb_fix_list=list(handoff.resolved_norb_fix_list),
        selections=[],
        metric={
            "type": "persisted_project_basis_handoff",
            "basis_is_orthonormal": True,
        },
        state_selection_quality={
            "status": "persisted_project_basis",
            "selection_policy": "consume_project_artifact",
        },
        gauge_anchor_quality={
            "status": "persisted_project_basis",
            "selection_policy": "consume_project_artifact",
        },
        symmetry_closure_quality={
            "status": "persisted_project_frame",
            "selection_policy": "consume_then_recertify",
            "symmetry_adapted_frame": frame_artifact,
        },
        warnings=[],
    )


def _gauge_report_from_gamma_routed_handoff(
    handoff: GammaRoutedBasisSpec,
) -> GaugeAnchorReport:
    """Describe a routed frame without inventing explicit orbital anchors."""

    return GaugeAnchorReport(
        gauge_mode="gamma_routed",
        resolved_norb_fix_list=[],
        selections=[],
        metric={
            "type": "persisted_gamma_routed_basis_handoff",
            "basis_is_orthonormal": True,
            "handoff_identity_hash": handoff.handoff_identity_hash,
        },
        state_selection_quality={
            "status": "persisted_gamma_routed_basis",
            "selection_policy": "consume_project_artifact",
        },
        gauge_anchor_quality={
            "status": "not_applicable",
            "reason": "routed_frames_are_authoritative",
        },
        symmetry_closure_quality={
            "status": "persisted_gamma_routed_basis",
            "selection_policy": "consume_then_recertify",
            "symmetry_adapted_frame": None,
        },
        warnings=[],
    )


def _states_from_gamma_routed_handoff(
    ctx: _ProjectionRunContext,
    handoff: GammaRoutedBasisSpec,
) -> tuple[
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    dict[int, ProjectionState],
    ProjectionState,
]:
    """Materialize persisted routed frames without diagonalization/downfolding."""

    handoff.require_k_indices(ctx.required_k)
    if handoff.layout.q_count != ctx.q_count or handoff.layout.full_dimension != ctx.full_dim:
        raise GammaRoutingError(
            CandidateRejectionReason.HANDOFF_IDENTITY,
            "routed Gamma layout does not match the active symmetry row space",
        )
    states: dict[int, ProjectionState] = {}
    for k_index in ctx.required_k:
        u_low, _u_high = handoff.assemble_for_k(k_index, include_high=False)
        # Keep the persisted numeric values exact.  No polar alignment, anchor
        # fallback, diagonalization, or Heff reconstruction is permitted here.
        states[int(k_index)] = ProjectionState(
            hamk=np.asarray(ctx.hamk_source_by_k[int(k_index)], dtype=np.complex128),
            heff=handoff.authoritative_heff_for_k(int(k_index)),
            u_low=u_low,
        )
    first = states[int(ctx.required_k[0])]
    return states, states, states, first


def _resolve_symmetry_project_identity(
    ctx: _ProjectionRunContext,
    *,
    handoff: ProjectionBasisSpec | None = None,
    gauge_report: Any | None = None,
    resolved_norb_fix_list: Any | None = None,
    low_dim: int | None = None,
) -> dict[str, Any]:
    run_cfg = ctx.config
    if handoff is not None:
        if isinstance(handoff, GammaRoutedBasisSpec):
            handoff.require_k_indices(ctx.required_k)
            layout_qsets = tuple(
                np.asarray(qset, dtype=np.float64)
                for qset in handoff.layout.ordered_qsets
            )
            if not (
                np.array_equal(layout_qsets[0], np.asarray(ctx.q1, dtype=np.float64))
                and np.array_equal(layout_qsets[1], np.asarray(ctx.q2, dtype=np.float64))
            ):
                raise GammaRoutingError(
                    CandidateRejectionReason.HANDOFF_IDENTITY,
                    "routed Gamma ordered-Q identity differs from the active config",
                )
            return dict(handoff.artifact_identity)
        hamk_file = _resolve(run_cfg.material.get("hamk_file"), run_cfg.cfg_dir)
        if hamk_file is None:
            raise ValueError(
                "material.hamk_file is required for kp symm projection identity"
            )
        hamk_source = load_hamk(hamk_file, mmap_mode="r")
        source_k_count = int(hamk_source.shape[0]) if hamk_source.ndim == 3 else 1
        configured_k_indices_raw = run_cfg.project_cfg.get("k_indices")
        if configured_k_indices_raw is None:
            configured_k_indices = list(range(source_k_count))
        elif isinstance(configured_k_indices_raw, str):
            configured_k_indices = [
                int(part.strip())
                for part in configured_k_indices_raw.split(",")
                if part.strip()
            ]
        elif isinstance(configured_k_indices_raw, Sequence):
            configured_k_indices = [int(value) for value in configured_k_indices_raw]
        else:
            raise ValueError(
                "project.k_indices must be a list or comma-separated string"
            )
        gauge_frame_hash = (
            str(handoff.frame_artifact.get("frame_hash", "")) or None
            if handoff.frame is not None and handoff.frame.status == "applied"
            else None
        )
        expected = build_projection_basis_identity(
            qset1=ctx.q1,
            qset2=ctx.q2,
            spin=run_cfg.spin,
            mode=ctx.mode,
            energy_scale=_energy_scale_from_material(run_cfg.material),
            nlow_state_list=ctx.nlow_state_list,
            resolved_norb_fix_list=handoff.resolved_norb_fix_list,
            gauge_mode=handoff.gauge_mode,
            num_layer_list=ctx.num_layer_list,
            num_orb_per_layer_list=ctx.num_orb_per_layer_list,
            orbital_block_dim=ctx.orb0,
            model_dim=int(handoff.model_dim),
            k_indices=configured_k_indices,
            gauge_frame_hash=gauge_frame_hash,
        )
        return _validate_symmetry_project_identity(
            expected,
            handoff.artifact_identity,
        )
    if gauge_report is None or resolved_norb_fix_list is None or low_dim is None:
        raise ValueError(
            "live symmetry identity requires gauge_report, resolved anchors, and low_dim"
        )
    project_dir_raw = run_cfg.project_cfg.get("out_dir")
    project_dir = _resolve(project_dir_raw, run_cfg.cfg_dir)
    if project_dir is None:
        raise ValueError("project.out_dir is required before kp symm can validate the projection basis")
    actual = _load_project_artifact_identity(project_dir)
    hamk_file = _resolve(run_cfg.material.get("hamk_file"), run_cfg.cfg_dir)
    if hamk_file is None:
        raise ValueError("material.hamk_file is required for kp symm projection identity")
    hamk_source = load_hamk(hamk_file, mmap_mode="r")
    source_k_count = int(hamk_source.shape[0]) if hamk_source.ndim == 3 else 1
    configured_k_indices_raw = run_cfg.project_cfg.get("k_indices")
    if configured_k_indices_raw is None:
        configured_k_indices = list(range(source_k_count))
    elif isinstance(configured_k_indices_raw, str):
        configured_k_indices = [
            int(part.strip())
            for part in configured_k_indices_raw.split(",")
            if part.strip()
        ]
    elif isinstance(configured_k_indices_raw, Sequence):
        configured_k_indices = [int(value) for value in configured_k_indices_raw]
    else:
        raise ValueError("project.k_indices must be a list or comma-separated string")
    for index in configured_k_indices:
        if index < 0 or index >= source_k_count:
            raise IndexError(
                f"project.k_indices contains {index}, but available k indices are 0..{source_k_count - 1}"
            )
    closure_quality = getattr(gauge_report, "symmetry_closure_quality", {})
    frame_artifact_raw = (
        closure_quality.get("symmetry_adapted_frame") if isinstance(closure_quality, Mapping) else None
    )
    frame_artifact = frame_artifact_raw if isinstance(frame_artifact_raw, Mapping) else None
    adapted_frame = None if frame_artifact is None else SymmetryAdaptedBasisFrame.from_artifact(frame_artifact)
    gauge_frame_hash = (
        str(frame_artifact.get("frame_hash", "")) or None
        if adapted_frame is not None and adapted_frame.status == "applied"
        else None
    )
    expected = build_projection_basis_identity(
        qset1=ctx.q1,
        qset2=ctx.q2,
        spin=run_cfg.spin,
        mode=ctx.mode,
        energy_scale=_energy_scale_from_material(run_cfg.material),
        nlow_state_list=ctx.nlow_state_list,
        resolved_norb_fix_list=resolved_norb_fix_list,
        gauge_mode=gauge_report.gauge_mode,
        num_layer_list=ctx.num_layer_list,
        num_orb_per_layer_list=ctx.num_orb_per_layer_list,
        orbital_block_dim=ctx.orb0,
        model_dim=int(low_dim),
        k_indices=configured_k_indices,
        gauge_frame_hash=gauge_frame_hash,
    )
    return _validate_symmetry_project_identity(expected, actual)


def _initial_projection_summary(
    ctx: _ProjectionRunContext,
    *,
    gauge_report: Any,
    low_dim: int,
    n_orb_for_exactification: tuple[int, int],
    artifact_identity: Mapping[str, Any],
) -> dict[str, Any]:
    run_cfg = ctx.config
    frame_artifact_raw = gauge_report.symmetry_closure_quality.get("symmetry_adapted_frame")
    frame_artifact = frame_artifact_raw if isinstance(frame_artifact_raw, Mapping) else None
    return {
        "config": run_cfg.cfg_path,
        "valley": run_cfg.valley,
        "spin": run_cfg.spin,
        "mode": ctx.mode,
        "tolerance": run_cfg.tolerance,
        "q_count": ctx.q_count,
        "orbital_block_dim": ctx.orb0,
        "full_dim": ctx.full_dim,
        "low_dim": low_dim,
        "artifact_identity": dict(artifact_identity),
        "frame": _frame_metadata(rotation_deg=ctx.q_rotation_deg, inference=ctx.frame_inference),
        "q_model": {
            "files": {"layer1": "q_model_layer1.npy", "layer2": "q_model_layer2.npy"},
            "source_files": {
                "layer1": str(_resolve(run_cfg.material.get("qset1_file"), run_cfg.cfg_dir)),
                "layer2": str(_resolve(run_cfg.material.get("qset2_file"), run_cfg.cfg_dir)),
            },
            "formula": "q_model = R(rotation_deg) @ (layer_mean - q_source)",
        },
        "operations": [],
        "strict_metadata": True,
        "requires_model_exactification": False,
        "exactification_owner": "kp_symm",
        "project_basis": {
            "nlow_state_list_layout": "physical_layer",
            "num_layer_list": [int(n) for n in ctx.num_layer_list],
            "gauge_mode": gauge_report.gauge_mode,
            "symmetry_adapted_frame": frame_artifact,
            "resolved_norb_fix_list": gauge_report.resolved_norb_fix_list,
            "basis_selection_report": "basis_selection.json",
            "resolved_sector_orbital_counts": {
                "L1": int(n_orb_for_exactification[0]),
                "L2": int(n_orb_for_exactification[1]),
            },
            "resolved_source_group_nlow_state_list": _source_group_nlow_state_list(
                ctx.nlow_state_list,
                ctx.num_layer_list,
            ),
        },
    }


def _cleanup_stale_projection_diagnostics(output_dir: Path, output_operation: str) -> None:
    diagnostics_dir = output_dir / "diagnostics"
    for stale in (
        output_dir / f"{output_operation}_low_polar.npy",
        output_dir / f"{output_operation}_low_representation_raw.npy",
        output_dir / f"{output_operation}_low_representation_polar.npy",
        diagnostics_dir / f"{output_operation}_low_polar.npy",
        diagnostics_dir / f"{output_operation}_low_representation_raw.npy",
        diagnostics_dir / f"{output_operation}_low_representation_polar.npy",
    ):
        if stale.exists():
            stale.unlink()


def _append_projected_operation_summaries(
    ctx: _ProjectionRunContext,
    *,
    summary: dict[str, Any],
    source_states: Mapping[int, ProjectionState],
    target_states: Mapping[int, ProjectionState],
    n_orb_for_exactification: tuple[int, int],
    compute_heff_covariance: bool = True,
) -> dict[str, np.ndarray]:
    run_cfg = ctx.config
    raw_low_matrices: dict[str, np.ndarray] = {}
    spin_convention = _spin_convention_for_exactification(
        run_cfg.spin,
        n_orb_for_exactification,
        valley=run_cfg.valley,
    )
    source_spin, target_spin = _spin_route_endpoints(
        run_cfg.spin,
        None if run_cfg.spin_sector_sewing is None else str(run_cfg.spin_sector_sewing),
    )
    for request in ctx.operation_requests:
        output_operation = request["output"]
        canonical_name = _canonical_internal_operation_name(output_operation)
        payload = ctx.operation_payloads[output_operation]
        entry = payload["entry"]
        antiunitary = bool(payload["antiunitary"])
        filename = None if payload["filename"] is None else str(payload["filename"])
        action = payload["action"]
        rep = action.representation
        source_action_metadata = _operation_action_metadata(entry, output_operation, antiunitary)
        model_action_metadata = _model_action_metadata(
            source_action_metadata,
            valley=run_cfg.valley,
            operation=output_operation,
            rotation_deg=ctx.q_rotation_deg,
        )

        raw_mats, polar_mats, pair_rows = _project_operation(
            operation=output_operation,
            antiunitary=antiunitary,
            d_full=action.matrix,
            states=source_states,
            pairs=entry["_pairs"],
            tolerance=run_cfg.tolerance,
            compute_polar=run_cfg.save_projection_diagnostics,
            compute_heff_covariance=compute_heff_covariance,
            enforce_heff_covariance=compute_heff_covariance,
            target_states=target_states,
            source_states=source_states,
        )
        rep_raw_mats: list[np.ndarray] = []
        rep_polar_mats: list[np.ndarray] = []
        rep_pair_rows: list[dict[str, Any]] = []
        if run_cfg.save_projection_diagnostics and rep is not None:
            rep_raw_mats, rep_polar_mats, rep_pair_rows = _project_operation(
                operation=output_operation,
                antiunitary=antiunitary,
                d_full=rep.matrix,
                states=source_states,
                pairs=entry["_pairs"],
                tolerance=run_cfg.tolerance,
                enforce_heff_covariance=False,
                raise_on_quality_failure=False,
                compute_polar=True,
                target_states=target_states,
                source_states=source_states,
            )
        for pair_row, full_pair_row in zip(pair_rows, payload["full_pair_rows"]):
            pair_row["full_space_covariance_residual"] = full_pair_row["full_space_covariance_residual"]
        support_matrices = [("raw_action", np.asarray(raw_mats[0], dtype=np.complex128))]
        if rep_raw_mats:
            support_matrices.append(("representation", np.asarray(rep_raw_mats[0], dtype=np.complex128)))
        resolved_model_action, model_basis_action = _resolve_projected_model_action(
            D_low=np.asarray(raw_mats[0], dtype=np.complex128),
            support_matrices=support_matrices,
            model_action=model_action_metadata,
            q_model1=ctx.q_model1,
            q_model2=ctx.q_model2,
            nlow_state_list=ctx.nlow_state_list,
            num_layer_list=ctx.num_layer_list,
            tol=max(float(run_cfg.tolerance), 1.0e-8),
            discover_action_candidates=True,
            accept_support_resolved_action=True,
        )
        _save_matrix_stack(ctx.output_dir / f"{output_operation}_low_raw.npy", raw_mats)
        _cleanup_stale_projection_diagnostics(ctx.output_dir, output_operation)
        if run_cfg.save_projection_diagnostics:
            diagnostics_dir = ctx.output_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            _save_matrix_stack(diagnostics_dir / f"{output_operation}_low_polar.npy", polar_mats)
            if rep_raw_mats:
                _save_matrix_stack(diagnostics_dir / f"{output_operation}_low_representation_raw.npy", rep_raw_mats)
                _save_matrix_stack(diagnostics_dir / f"{output_operation}_low_representation_polar.npy", rep_polar_mats)
        matrix_kind, matrix_selection = _select_operation_matrix_kind(
            model_basis_action,
            representation_pair_rows=rep_pair_rows,
        )
        support_resolution = model_basis_action.get("support_resolution", {})
        accepted_internal_action = (
            isinstance(support_resolution, Mapping)
            and isinstance(support_resolution.get("provenance"), Mapping)
            and support_resolution["provenance"].get("accepted_by")
            in {"kp_projected_basis_inference", "explicit_action_candidates"}
        )
        operation_summary = {
            "name": canonical_name,
            "operation": output_operation,
            "antiunitary": antiunitary,
            "matrix_file": f"{output_operation}_low_raw.npy",
            "matrix_kind": matrix_kind,
            "source_matrix_role": "raw_h_sewing_action",
            "source_gauge": "raw_saved_TAPW",
            "target_role": "continuum_internal_rep",
            "gauge_correction": {"kind": "none"},
            "antiunitary_convention": "U_K" if antiunitary else "none",
            **model_action_metadata,
            "source_action": source_action_metadata,
            "model_action": model_action_metadata,
            "declared_model_action": model_action_metadata,
            "model_basis_action": model_basis_action,
            "matrix_selection": matrix_selection,
            "axis_deg": entry.get("axis_deg"),
            "status": "projected",
            "source_covariance_status": entry.get("status"),
            "source_residual_H_raw": entry.get("residual_H_raw"),
            "square_residual": entry.get("square_residual"),
            "spglib_index": entry.get("spglib_index"),
            "ld_source_rule": entry.get("ld_source_rule"),
            "pg_file": None if action.pg_filename is None else str(ctx.rep_root / action.pg_filename),
            "raw_h_operator_file": None if action.raw_h_filename is None else str(ctx.rep_root / action.raw_h_filename),
            "action_source": action.action_source,
            "combined_raw_h_residual": action.combined_raw_h_residual,
            "from_full_spinful": None if rep is None else rep.from_full_spinful,
            "spin_leakage": None if rep is None else rep.spin_leakage,
            "pg_spin_leakage": None if action.pg is None else action.pg.spin_leakage,
            "raw_h_spin_leakage": None if action.raw_h is None else action.raw_h.spin_leakage,
            "spin_sector_sewing": run_cfg.spin_sector_sewing,
            "source_spin": source_spin,
            "target_spin": target_spin,
            "spin_route_inference": dict(ctx.spin_route_inference),
            "pairs": pair_rows,
            "group_relations": [
                _operation_power_relation(canonical_name, spin_convention=spin_convention)
            ],
        }
        if run_cfg.save_projection_diagnostics:
            developer_output_files = {
                "polar_matrix_file": f"diagnostics/{output_operation}_low_polar.npy",
            }
            if filename is not None:
                developer_output_files["representation_file"] = str(ctx.rep_root / filename)
            if rep_raw_mats:
                developer_output_files["representation_matrix_file"] = f"diagnostics/{output_operation}_low_representation_raw.npy"
                developer_output_files["representation_polar_matrix_file"] = (
                    f"diagnostics/{output_operation}_low_representation_polar.npy"
                )
                operation_summary["representation_pairs"] = rep_pair_rows
            operation_summary["developer_outputs"] = developer_output_files
        raw_low_matrices[canonical_name] = np.asarray(raw_mats[0], dtype=np.complex128)
        if entry.get("k_pairs_inferred_from_default_k_index"):
            operation_summary["k_pairs_inferred_from_default_k_index"] = True
            operation_summary["candidate_source"] = entry.get("candidate_source", "diagnostic_default_k_index")
            operation_summary["k_pairs_provenance"] = dict(entry.get("_k_pairs_provenance", {}))
        if accepted_internal_action:
            operation_summary["internal_resolved_action"] = resolved_model_action
        c2_action_audit = _c2_action_audit_for_gamma(
            mode=ctx.mode,
            operation=output_operation,
            matrix_kind=matrix_kind,
            matrix_selection=matrix_selection,
            model_basis_action=model_basis_action,
            raw_matrix=np.asarray(raw_mats[0], dtype=np.complex128),
            representation_matrix=None if not rep_raw_mats else np.asarray(rep_raw_mats[0], dtype=np.complex128),
            pair_rows=pair_rows,
            representation_pair_rows=rep_pair_rows,
            declared_model_action=model_action_metadata,
            combined_raw_h_residual=action.combined_raw_h_residual,
        )
        if c2_action_audit is not None:
            operation_summary["C2_action_audit"] = c2_action_audit
        summary["operations"].append(operation_summary)
    return raw_low_matrices


def _factorized_response_action_package(
    metadata: Mapping[str, Any],
    matrices: Mapping[str, np.ndarray],
    *,
    q_geometry: CanonicalQResult,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Certify all packed exactified actions for the reusable model fast path."""

    sector_order = tuple(str(value) for value in q_geometry.artifact.get("sector_order", ()))
    exactification = metadata.get("kp_symm_exactification", {})
    n_orb_raw = exactification.get("n_orb") if isinstance(exactification, Mapping) else None
    if len(sector_order) != 2 or not isinstance(n_orb_raw, Sequence):
        return (
            {
                "version": FACTORIZED_RESPONSE_ACTION_V1,
                "status": "unavailable",
                "reason": "missing_two_sector_basis_layout",
            },
            {},
        )
    n_orb = tuple(int(value) for value in n_orb_raw)
    if len(n_orb) != len(sector_order):
        return (
            {
                "version": FACTORIZED_RESPONSE_ACTION_V1,
                "status": "unavailable",
                "reason": "n_orb_sector_count_mismatch",
            },
            {},
        )
    q_vectors = tuple(
        np.asarray(q_geometry.canonical_q[name], dtype=np.float64)
        for name in sector_order
    )
    q_counts = tuple(int(array.shape[0]) for array in q_vectors)
    q_offsets = np.cumsum((0, *q_counts[:-1])).astype(np.int64)
    sector_by_name = {name: index for index, name in enumerate(sector_order)}
    q_bound = float(q_geometry.artifact.get("canonical_closure_max", 0.0))
    operations = metadata.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        operations = []
    actions = []
    failures: list[dict[str, str]] = []
    joint_root = (
        exactification.get("joint_block_representation")
        if isinstance(exactification, Mapping)
        else None
    )
    joint_actions = {
        str(record.get("name", "")): record
        for record in (
            joint_root.get("actions", [])
            if isinstance(joint_root, Mapping)
            and str(joint_root.get("status", "")) == "certified"
            else []
        )
        if isinstance(record, Mapping) and record.get("name")
    }

    expected_fibers: list[tuple[int, ...]] = []
    for sector, (count, orbitals) in enumerate(zip(q_counts, n_orb)):
        basis_offset = int(sum(q_counts[item] * n_orb[item] for item in range(sector)))
        for q_index in range(count):
            expected_fibers.append(
                tuple(
                    basis_offset + orbital * count + q_index
                    for orbital in range(orbitals)
                )
            )

    def joint_q_route(name: str) -> tuple[list[int], list[int]] | None:
        if any(orbitals == 0 for orbitals in n_orb):
            return None
        record = joint_actions.get(name)
        if record is None:
            return None
        fibers_raw = record.get("fiber_indices", [])
        permutation_raw = record.get("fiber_permutation", [])
        if not isinstance(fibers_raw, Sequence) or isinstance(fibers_raw, (str, bytes)):
            raise ValueError(f"joint action {name!r} has invalid fiber_indices")
        fibers = [tuple(int(value) for value in group) for group in fibers_raw]
        if len(fibers) != len(expected_fibers) or set(fibers) != set(expected_fibers):
            raise ValueError(
                f"joint action {name!r} fiber layout does not match the model Q/orbital basis"
            )
        if not isinstance(permutation_raw, Sequence) or isinstance(
            permutation_raw, (str, bytes)
        ):
            raise ValueError(f"joint action {name!r} has invalid fiber_permutation")
        permutation = tuple(int(value) for value in permutation_raw)
        if sorted(permutation) != list(range(len(fibers))):
            raise ValueError(f"joint action {name!r} fiber_permutation is not bijective")
        record_for_fiber = {fiber: index for index, fiber in enumerate(fibers)}
        global_q_for_record = {
            record_for_fiber[fiber]: global_q
            for global_q, fiber in enumerate(expected_fibers)
        }
        q_permutation = [
            global_q_for_record[permutation[record_for_fiber[fiber]]]
            for fiber in expected_fibers
        ]
        q_offsets_with_end = np.cumsum((0, *q_counts)).astype(np.int64)
        sector_permutation: list[int] = []
        for source_sector in range(len(q_counts)):
            source_start = int(q_offsets_with_end[source_sector])
            source_end = int(q_offsets_with_end[source_sector + 1])
            target_sectors = {
                int(np.searchsorted(q_offsets_with_end, q_permutation[source], side="right") - 1)
                for source in range(source_start, source_end)
            }
            if len(target_sectors) != 1:
                raise ValueError(
                    f"joint action {name!r} maps one source sector to multiple target sectors"
                )
            sector_permutation.append(next(iter(target_sectors)))
        return q_permutation, sector_permutation

    for raw_operation in operations:
        if not isinstance(raw_operation, Mapping):
            continue
        operation = dict(raw_operation)
        name = str(operation.get("name", operation.get("operation", "")))
        if name not in matrices:
            failures.append({"name": name, "reason": "missing_exactified_matrix"})
            continue
        try:
            joint_route = joint_q_route(name)
        except ValueError as exc:
            failures.append({"name": name, "reason": str(exc)})
            continue
        if joint_route is not None:
            q_permutation, sector_permutation = joint_route
        else:
            basis_action = operation.get("model_basis_action", {})
            items = basis_action.get("items", []) if isinstance(basis_action, Mapping) else []
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                failures.append({"name": name, "reason": "missing_model_basis_action_items"})
                continue
            routes: dict[tuple[int, int], tuple[int, int]] = {}
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                source_name = str(item.get("source_sector", ""))
                target_name = str(item.get("target_sector", ""))
                if source_name not in sector_by_name or target_name not in sector_by_name:
                    continue
                routes[(sector_by_name[source_name], int(item["source_q_index"]))] = (
                    sector_by_name[target_name],
                    int(item["target_q_index"]),
                )
            action_metadata_for_route = None
            for field in (
                "internal_resolved_action",
                "model_action",
                "declared_model_action",
            ):
                candidate = operation.get(field)
                if isinstance(candidate, Mapping):
                    action_metadata_for_route = candidate
                    break
            q_permutation = []
            sector_permutation = []
            route_error: str | None = None
            for source_sector, count in enumerate(q_counts):
                targets: set[int] = set()
                for source_q in range(count):
                    route = routes.get((source_sector, source_q))
                    if (
                        route is None
                        and n_orb[source_sector] == 0
                    ):
                        basis_sector_map = (
                            basis_action.get("sector_map")
                            if isinstance(basis_action, Mapping)
                            else None
                        )
                        if basis_sector_map is None and action_metadata_for_route is not None:
                            basis_sector_map = action_metadata_for_route.get(
                                "sector_map", "identity"
                            )
                        source_name = sector_order[source_sector]
                        target_name = _sector_target(
                            source_name,
                            "identity" if basis_sector_map is None else basis_sector_map,
                        )
                        if target_name not in sector_by_name:
                            route_error = "inactive_sector_target_unknown"
                            break
                        target_sector = int(sector_by_name[target_name])
                        if n_orb[target_sector] != 0:
                            route_error = "inactive_sector_maps_to_active_sector"
                            break
                        if q_counts[source_sector] != q_counts[target_sector]:
                            route_error = "inactive_sector_q_count_mismatch"
                            break
                        # A zero-orbital sector contributes no rows or columns to
                        # the continuum basis.  Its Q route is therefore a
                        # canonical placeholder, not a physical covariance
                        # datum.  Preserve q-index order so the packed action
                        # remains a full two-sector permutation.
                        route = (target_sector, int(source_q))
                    if route is None:
                        route_error = "incomplete_model_basis_action"
                        break
                    target_sector, target_q = route
                    if not 0 <= target_q < q_counts[target_sector]:
                        route_error = "target_q_out_of_range"
                        break
                    targets.add(int(target_sector))
                    q_permutation.append(int(q_offsets[target_sector] + target_q))
                if route_error is not None:
                    break
                if len(targets) != 1:
                    route_error = "source_sector_has_multiple_target_sectors"
                    break
                sector_permutation.append(next(iter(targets)))
            if route_error is not None:
                failures.append({"name": name, "reason": route_error})
                continue
        action_metadata = None
        for field in ("internal_resolved_action", "model_action", "declared_model_action"):
            candidate = operation.get(field)
            if isinstance(candidate, Mapping):
                action_metadata = candidate
                break
        if action_metadata is None or not isinstance(action_metadata.get("k_map"), Mapping):
            failures.append({"name": name, "reason": "missing_explicit_k_map"})
            continue
        try:
            actions.append(
                certify_factorized_action(
                    name=name,
                    matrix=np.asarray(matrices[name], dtype=np.complex128),
                    antiunitary=bool(operation.get("antiunitary", False)),
                    k_forward=explicit_linear_action_matrix(action_metadata["k_map"]),
                    q_permutation=q_permutation,
                    sector_permutation=sector_permutation,
                    q_vectors=q_vectors,
                    q_counts=q_counts,
                    n_orb=n_orb,
                    matrix_absolute_error_bound=0.0,
                    q_absolute_error_bound=q_bound,
                )
            )
        except FactorizedActionCertificationError as exc:
            failures.append({"name": name, "reason": str(exc)})
    if failures or len(actions) != len(matrices):
        return (
            {
                "version": FACTORIZED_RESPONSE_ACTION_V1,
                "status": "unavailable",
                "reason": "one_or_more_actions_not_factorizable",
                "failures": failures,
            },
            {},
        )
    return pack_factorized_actions(actions)


def _write_canonical_symmetry_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    q_geometry: CanonicalQResult | None = None,
    joint_artifact_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    operations = summary.get("operations", [])
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        return

    matrices: dict[str, np.ndarray] = {}
    packed_operations: list[dict[str, Any]] = []
    artifact_identity = summary.get("artifact_identity", {})
    if not isinstance(artifact_identity, Mapping):
        artifact_identity = {}
    residual_rows = ["operation,status,unitarity,exactification_distance"]
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        name = str(operation.get("name", ""))
        if not name:
            continue
        matrix_file = operation.get("matrix_file")
        if matrix_file not in (None, ""):
            matrix_path = Path(str(matrix_file))
            if not matrix_path.is_absolute():
                matrix_path = output_dir / matrix_path
            if matrix_path.exists():
                matrices[name] = np.asarray(np.load(matrix_path), dtype=np.complex128)
                packed_operation = dict(operation)
                packed_operation["matrix_file"] = "representations.npz"
                packed_operation["matrix_array_key"] = name
                packed_operation["exactification_owner"] = "kp_symm"
                packed_operation.pop("developer_outputs", None)
                for field in _PROJECT_ARTIFACT_IDENTITY_FIELDS:
                    if field in artifact_identity:
                        packed_operation[field] = artifact_identity[field]
                packed_operations.append(packed_operation)
        residuals = operation.get("residuals", {})
        unitarity = ""
        if isinstance(residuals, Mapping) and residuals.get("unitarity") is not None:
            unitarity = str(residuals.get("unitarity"))
        exactification_distance = operation.get("exactification_distance")
        if exactification_distance is None and isinstance(operation.get("source_matrix_projection_report"), Mapping):
            report = operation["source_matrix_projection_report"]
            if isinstance(report.get("report"), Mapping):
                exactification_distance = report["report"].get("distance")
        residual_rows.append(
            ",".join(
                [
                    name,
                    str(operation.get("status", "")),
                    unitarity,
                    "" if exactification_distance is None else str(exactification_distance),
                ]
            )
        )

    if matrices:
        metadata = dict(summary)
        metadata["operations"] = packed_operations
        metadata.setdefault("exactification_owner", "kp_symm")
        metadata.setdefault(
            "kp_symm_exactification",
            {"status": "exactified", "matrix_source": "kp_symm_exactified_action"},
        )
        payload = dict(matrices)
        if joint_artifact_arrays is not None:
            payload.update(
                {
                    str(key): np.asarray(value, dtype=np.complex128)
                    for key, value in joint_artifact_arrays.items()
                }
            )
        if q_geometry is not None:
            sector_order = tuple(q_geometry.artifact.get("sector_order", ()))
            if len(sector_order) != 2:
                raise ValueError(
                    "canonical symmetry output currently requires exactly two ordered Q sectors"
                )
            layer1, layer2 = sector_order
            payload.update(
                {
                    "__q_model_raw_layer1__": np.asarray(q_geometry.raw_q[layer1], dtype=np.float64),
                    "__q_model_raw_layer2__": np.asarray(q_geometry.raw_q[layer2], dtype=np.float64),
                    "__q_model_canonical_layer1__": np.asarray(
                        q_geometry.canonical_q[layer1], dtype=np.float64
                    ),
                    "__q_model_canonical_layer2__": np.asarray(
                        q_geometry.canonical_q[layer2], dtype=np.float64
                    ),
                }
            )
            factorized_metadata, factorized_arrays = (
                _factorized_response_action_package(
                    metadata,
                    matrices,
                    q_geometry=q_geometry,
                )
            )
            exactification = dict(metadata.get("kp_symm_exactification", {}))
            exactification["factorized_response_action"] = factorized_metadata
            metadata["kp_symm_exactification"] = exactification
            payload.update(factorized_arrays)
        payload["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(output_dir / "representations.npz", **payload)
    (output_dir / "residuals.csv").write_text("\n".join(residual_rows) + "\n", encoding="utf-8")
    _write_summary_md(output_dir / "summary.md", summary)
    keep = {"representations.npz", "residuals.csv", "summary.md"}
    for path in output_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
        elif path.is_dir() and path.name == "diagnostics":
            shutil.rmtree(path)


def _exactify_and_write_projection_summary(
    ctx: _ProjectionRunContext,
    *,
    summary: dict[str, Any],
    raw_low_matrices: Mapping[str, np.ndarray],
    n_orb: tuple[int, int],
) -> dict[str, Any]:
    _invalidate_stale_canonical_symmetry_outputs(ctx.output_dir)
    run_cfg = ctx.config
    bM_candidates = bM_candidates_from_q_distances(ctx.q_model1, ctx.q_model2)
    if bM_candidates:
        bM1, bM2 = canonical_bM_pair_from_candidates(bM_candidates, angle_deg=60.0)
    else:
        bM1 = np.array([1.0, 0.0], dtype=float)
        bM2 = np.array([0.5, float(np.sqrt(3.0) / 2.0)], dtype=float)
    sectors = sectors_with_q_offsets(
        [
            {"name": "L1", "qset": "qset1", "n_orb": int(n_orb[0])},
            {"name": "L2", "qset": "qset2", "n_orb": int(n_orb[1])},
        ],
        Q_set1=ctx.q_model1,
        Q_set2=ctx.q_model2,
        bM1=bM1,
        bM2=bM2,
    )
    exact_overrides = _merged_exactification_overrides(
        run_cfg.valley,
        run_cfg.symm_cfg.get("exactification"),
        symmetry_tolerance=run_cfg.tolerance,
        operation_actions=run_cfg.symm_cfg.get("operation_actions"),
    )
    exact_config = _kp_symm_exactification_config(exact_overrides)
    project_basis = summary.get("project_basis", {})
    frame_artifact_raw = (
        project_basis.get("symmetry_adapted_frame")
        if isinstance(project_basis, Mapping)
        else None
    )
    frame_artifact = (
        frame_artifact_raw if isinstance(frame_artifact_raw, Mapping) else None
    )
    adapted_frame = (
        None
        if frame_artifact is None
        else SymmetryAdaptedBasisFrame.from_artifact(frame_artifact)
    )
    exact_dimension = (
        next(iter(raw_low_matrices.values())).shape[0]
        if raw_low_matrices
        else 0
    )
    if adapted_frame is not None and adapted_frame.full_unitary.shape != (
        exact_dimension,
        exact_dimension,
    ):
        raise ValueError(
            "symmetry-adapted frame is inconsistent with the projected model basis: "
            f"{adapted_frame.full_unitary.shape} != "
            f"{(exact_dimension, exact_dimension)}"
        )
    project_basis_matrices = _transform_projected_operations_to_persisted_frame(
        raw_low_matrices,
        summary["operations"],
        adapted_frame,
    )
    frame_components = (
        adapted_frame.components if adapted_frame is not None else {}
    )
    structure_transaction = frame_components.get("joint_structure_transaction")
    transaction_decision = (
        str(structure_transaction.get("decision"))
        if isinstance(structure_transaction, Mapping)
        else None
    )
    uses_closest_cyclotomic = isinstance(
        frame_components.get("closest_cyclotomic_u1_gauge"), Mapping
    )
    uses_standard_generator = isinstance(
        frame_components.get("standard_generator_fiber_gauge"), Mapping
    ) and frame_components["standard_generator_fiber_gauge"].get("status") == "certified"
    uses_structure_transaction = isinstance(
        structure_transaction, Mapping
    )
    project_frame_revalidation: Mapping[str, Any] | None = None
    final_joint_result: JointExactificationResult | None = None
    fixed_target_certificate: Mapping[str, Any] | None = None
    fixed_target_source_role: str | None = None
    fixed_target_source_hashes: dict[str, str] | None = None
    joint_payload: Mapping[str, Any] | None = None
    if uses_structure_transaction:
        if adapted_frame is None or frame_artifact is None:
            raise ValueError(
                "persisted project structure transaction lacks its basis frame"
            )
        canonical_target_raw = frame_components.get("canonical_target_actions")
        if not isinstance(canonical_target_raw, Mapping):
            raise ValueError(
                "persisted project structure transaction lacks its canonical target; "
                "rerun kp project with the current handoff version"
            )
        try:
            canonical_target = load_canonical_target_artifact(
                canonical_target_raw
            )
        except CanonicalTargetError as exc:
            raise ValueError(f"invalid persisted canonical target: {exc}") from exc
        project_frame_revalidation = _certify_exact_target_in_persisted_project_basis(
            canonical_target_raw,
            persisted_frame=adapted_frame,
            persisted_frame_artifact=frame_artifact,
            sectors=[
                {
                    "name": "L1",
                    "n_orb": int(n_orb[0]),
                    "n_q": int(len(ctx.q_model1)),
                },
                {
                    "name": "L2",
                    "n_orb": int(n_orb[1]),
                    "n_q": int(len(ctx.q_model2)),
                },
            ],
        )
        operation_names = tuple(
            str(operation["name"]) for operation in summary["operations"]
        )
        if set(operation_names) != set(canonical_target.actions):
            raise ValueError(
                "current kp symm operations do not match the project-owned target: "
                f"current={sorted(operation_names)}, "
                f"target={sorted(canonical_target.actions)}"
            )
        parity_by_name = {
            str(operation["name"]): bool(operation.get("antiunitary", False))
            for operation in summary["operations"]
        }
        spin_sector_sewing = getattr(run_cfg, "spin_sector_sewing", None)
        exactified_internal_source: Mapping[str, np.ndarray] | None = None
        if spin_sector_sewing is not None:
            exactified_internal_source, _source_reports = (
                exactify_loaded_symmetry_source(
                    loaded_metadata=summary,
                    matrices=raw_low_matrices,
                    Q_set1=ctx.q_model1,
                    Q_set2=ctx.q_model2,
                    sectors=sectors,
                    n_orb=n_orb,
                    bM1=bM1,
                    bM2=bM2,
                    raw_config={"exactification": exact_config},
                    rotation_deg=0.0,
                    output_dir=None,
                )
            )
        fixed_target_pre_frame, fixed_target_source_role = (
            _select_fixed_target_source_matrices(
                raw_low_matrices,
                exactified_internal_matrices=exactified_internal_source,
                spin_sector_sewing=spin_sector_sewing,
            )
        )
        fixed_target_source_hashes = {
            name: hash_array(fixed_target_pre_frame[name])
            for name in operation_names
        }
        fixed_target_project_basis_matrices = (
            _transform_projected_operations_to_persisted_frame(
                fixed_target_pre_frame,
                summary["operations"],
                adapted_frame,
            )
        )
        fixed_target_certificate = certify_fixed_target_exactification(
            fixed_target_project_basis_matrices,
            canonical_target.actions,
            root_order=int(canonical_target_raw.get("root_order", 24)),
            max_rms_correction=float(
                canonical_target.joint_config.max_rms_correction
            ),
            max_route_correction=float(
                canonical_target.joint_config.max_route_correction
            ),
            persisted_frame_hash=str(frame_artifact.get("frame_hash", "")),
            target_artifact_hash=canonical_target.artifact_hash,
            target_certificate_identity=canonical_target.certificate_identity,
            source_matrix_hashes=fixed_target_source_hashes,
            operation_antiunitary=parity_by_name,
        )
        fixed_target_certificate = dict(fixed_target_certificate)
        fixed_target_certificate["source_matrix_role"] = fixed_target_source_role
        fixed_target_certificate["raw_sewing_matrix_hashes"] = {
            name: hash_array(raw_low_matrices[name]) for name in operation_names
        }
        final_joint_result = certify_fixed_target_joint_result(
            canonical_target.actions,
            canonical_target.presentation,
            config=canonical_target.joint_config,
            provenance={
                "kind": "persisted_project_basis_fixed_target",
                "frame_hash": str(frame_artifact.get("frame_hash", "")),
                "target_artifact_hash": canonical_target.artifact_hash,
                "target_certificate_identity": dict(
                    canonical_target.certificate_identity
                ),
            },
            fixed_target_certificate=fixed_target_certificate,
            diagnostic_result=None,
        )
        exact_matrices = {
            name: materialize_block_route_action(action)
            for name, action in final_joint_result.actions.items()
        }
        artifact_hash = str(final_joint_result.artifact_metadata["artifact_hash"])
        exact_reports: dict[str, Any] = {}
        for operation in summary["operations"]:
            name = str(operation["name"])
            resolved_action = dict(operation.get("model_action", {}))
            exact_reports[name] = {
                "input_matrix_file": operation.get("matrix_file"),
                "operation_alias": operation.get("operation_alias"),
                "canonical_physical_operation": operation.get(
                    "canonical_physical_operation", name
                ),
                "physical_parent": operation.get(
                    "physical_parent",
                    operation.get("canonical_physical_operation", name),
                ),
                "representation_level": operation.get("representation_level"),
                "effective_name": operation.get("effective_name"),
                "approximation": operation.get("approximation"),
                "derived_from": operation.get("derived_from"),
                "source_matrix_role": operation.get("source_matrix_role"),
                "source_gauge": operation.get("source_gauge"),
                "target_role": operation.get("target_role"),
                "resolved_action": resolved_action,
                "manifest_model_action": resolved_action,
                "selected_action_candidate": resolved_action,
                "support_resolution": {
                    "action_mismatch": False,
                    "candidate_source": "persisted_project_canonical_target",
                    "declared_model_action": resolved_action,
                    "selected_action_candidate": resolved_action,
                    "matrix_kind": "action",
                    "matrix_source": "raw_h_action_projection",
                },
                "preferred_mode": "project_owned_general_Ud_block_route",
                "report": {
                    "status": "exactified",
                    "method": (
                        "direct_transformed_raw_to_persisted_project_target"
                        if fixed_target_source_role
                        == "raw_projected_internal_action"
                        else "joint_exactified_cross_sector_to_persisted_project_target"
                    ),
                    "correction_rms": float(
                        fixed_target_certificate["correction_rms_by_operation"][name]
                    ),
                    "correction_max": float(
                        fixed_target_certificate["correction_max_by_operation"][name]
                    ),
                    "numeric_entries_modified": 0,
                    "numeric_entries_modified_scope": (
                        "post_exactification_storage_cleanup_v1"
                    ),
                },
                "joint_exactification": {
                    "status": "certified",
                    "artifact_hash": artifact_hash,
                    "presentation_source": canonical_target.presentation.source,
                    "relation_certification_bound": float(
                        final_joint_result.report["relation_certification_bound"]
                    ),
                    "pre_relation_residual_max": float(
                        final_joint_result.report["pre_relation_residual_max"]
                    ),
                    "post_relation_residual_max": float(
                        final_joint_result.report["post_relation_residual_max"]
                    ),
                    "route_correction_rms": float(
                        fixed_target_certificate["correction_rms_by_operation"][name]
                    ),
                    "route_correction_max": float(
                        fixed_target_certificate["correction_max_by_operation"][name]
                    ),
                    "root_report": dict(final_joint_result.report),
                    "production_used_free_solver": False,
                },
            }
        joint_payload = {
            "metadata": dict(final_joint_result.artifact_metadata),
            "arrays": dict(final_joint_result.artifact_arrays),
            "report": dict(final_joint_result.report),
        }
    elif uses_closest_cyclotomic or uses_standard_generator:
        raise ValueError(
            "persisted standard-generator frame lacks its joint structure "
            "transaction; rerun kp project before kp symm"
        )
    else:
        exact_matrices, exact_reports = exactify_loaded_symmetry_source(
            loaded_metadata=summary,
            matrices=project_basis_matrices,
            Q_set1=ctx.q_model1,
            Q_set2=ctx.q_model2,
            sectors=sectors,
            n_orb=n_orb,
            bM1=bM1,
            bM2=bM2,
            raw_config={"exactification": exact_config},
            rotation_deg=0.0,
            output_dir=None,
        )
        joint_payload = exact_reports.get("__joint_exactification__")
        if isinstance(joint_payload, Mapping):
            metadata = joint_payload.get("metadata")
            arrays = joint_payload.get("arrays")
            if not isinstance(metadata, Mapping) or not isinstance(arrays, Mapping):
                raise ValueError(
                    "joint exactification handoff lacks metadata or route arrays"
                )
            final_joint_result = load_joint_exactification_artifact(metadata, arrays)
            exact_matrices = {
                name: materialize_block_route_action(action)
                for name, action in final_joint_result.actions.items()
            }

    for operation_summary in summary["operations"]:
        name = str(operation_summary["name"])
        report = exact_reports.get(name)
        if not isinstance(report, Mapping):
            raise ValueError(f"{name} kp_symm exactification did not produce a report")
        status = report.get("report", {}).get("status") if isinstance(report.get("report"), Mapping) else None
        if status != "exactified":
            raise ValueError(f"{name} kp_symm exactification did not finish: status={status!r}")
        if name not in exact_matrices:
            raise ValueError(f"{name} kp_symm exactification did not produce a matrix")
        input_basis_transform = {
            "version": "kp_exactification_input_basis_transform_v1",
            "source_matrix_file": operation_summary.get("matrix_file"),
            "source_matrix_hash": hash_array(raw_low_matrices[name]),
            "source_basis": "raw_projected_pre_project_frame",
            "target_basis": "persisted_project_final_basis",
            "frame_hash": (
                None if frame_artifact is None else frame_artifact.get("frame_hash")
            ),
            "antiunitary": bool(operation_summary.get("antiunitary", False)),
            "formula": (
                "W_dagger_D_W_conjugate"
                if bool(operation_summary.get("antiunitary", False))
                else "W_dagger_D_W"
            ),
            "transformed_matrix_hash": hash_array(project_basis_matrices[name]),
            "transformation_applied": bool(
                adapted_frame is not None and adapted_frame.status == "applied"
            ),
        }
        report["input_basis_transform"] = input_basis_transform
        if fixed_target_certificate is not None:
            report["fixed_target_exactification"] = dict(
                fixed_target_certificate
            )
        if adapted_frame is not None and adapted_frame.status == "applied":
            operation_summary["gauge_correction"] = {
                "kind": (
                    "persisted_project_frame_before_fixed_target_certification"
                    if uses_structure_transaction
                    else "persisted_project_frame_before_joint_exactification"
                ),
                "frame_hash": frame_artifact.get("frame_hash"),
                "version": frame_artifact.get("version"),
                "transaction_decision": transaction_decision,
                "formula": input_basis_transform["formula"],
                "additional_basis_gauge_applied": False,
            }
        exact_matrices[name], storage_cleanup = _save_exactified_matrix(
            ctx.output_dir / f"exactified_{name}.npy",
            exact_matrices[name],
        )
        operation_summary["matrix_storage_cleanup"] = storage_cleanup
        operation_summary["raw_matrix_file"] = operation_summary["matrix_file"]
        operation_summary["matrix_file"] = f"exactified_{name}.npy"
        operation_summary["matrix_kind"] = "continuum_internal_rep_exact"
        operation_summary["matrix_source"] = "kp_symm_exactified_action"
        operation_summary["matrix_scope"] = "point_independent_continuum_action"
        operation_summary["valid_k_domain"] = "all_model_k"
        operation_summary["reference_pairs_are_not_domain_restrictions"] = True
        operation_summary["exactification_reference_pairs"] = list(operation_summary.get("pairs", []))
        operation_summary["projection_diagnostic_pairs"] = list(operation_summary.get("pairs", []))
        operation_summary["status"] = "exactified"
        operation_summary["exactification_status"] = status
        operation_summary["exactification_report_file"] = f"{name.lower()}_exactification_report.json"
        operation_summary["source_matrix_projection_report"] = report
        if fixed_target_certificate is not None:
            operation_summary["fixed_target_exactification"] = dict(
                fixed_target_certificate
            )
        operation_summary["internal_resolved_action"] = report.get("resolved_action", operation_summary["model_action"])
    if final_joint_result is not None:
        final_artifact_hash = str(
            final_joint_result.artifact_metadata["artifact_hash"]
        )
        for generator in final_joint_result.presentation.generators:
            name = generator.name
            operation_report = exact_reports[name]
            operation_report["joint_exactification"]["artifact_hash"] = (
                final_artifact_hash
            )
            operation_report["joint_exactification"][
                "post_gauge_certification"
            ] = dict(final_joint_result.report)
            operation_report["joint_exactification"][
                "post_gauge_certification_compatibility"
            ] = {
                "deprecated_alias_of": "joint_exactification_certification",
                "additional_gauge_applied": False,
            }
            operation_report["joint_exactification"][
                "joint_exactification_certification"
            ] = dict(final_joint_result.report)
            operation_report["joint_exactification"][
                "project_target_certification"
            ] = (
                {
                    "status": "not_applicable",
                    "reason": "no_persisted_joint_structure_transaction",
                }
                if project_frame_revalidation is None
                else dict(project_frame_revalidation)
            )
    # Emit reports only after the authoritative final transaction is known.
    for operation_summary in summary["operations"]:
        name = str(operation_summary["name"])
        operation_report = exact_reports[name]
        with (
            ctx.output_dir / f"{name.lower()}_exactification_report.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(operation_report, handle, indent=2)
    kp_symm_exactification = {
        "status": "exactified",
        "matrix_source": "kp_symm_exactified_action",
        "config": exact_config,
        "bM1": bM1.tolist(),
        "bM2": bM2.tolist(),
        "n_orb": [int(n_orb[0]), int(n_orb[1])],
        "sectors": sectors,
        "polynomial_coordinate": {
            "coordinate_convention": "right_handed_model_cartesian_reciprocal_v1",
            "origin": [0.0, 0.0],
            "origin_role": "exactified_valley_expansion_origin_in_model_cartesian",
            "valley": str(ctx.config.valley),
        },
        "basis_transform_order": (
            (
                "raw_cross_sector_sewing__joint_internal_exactification__"
                "persisted_project_frame__project_owned_fixed_target"
            )
            if fixed_target_source_role
            == "joint_exactified_cross_sector_internal_action"
            else "raw_projected_action__persisted_project_frame__project_owned_fixed_target"
            if uses_structure_transaction
            else "raw_projected_action__persisted_project_frame__joint_exactification"
        ),
        "input_project_basis_frame": frame_artifact,
        "post_exactification_gauge": frame_artifact,
        "post_exactification_gauge_compatibility": {
            "deprecated_alias_of": "input_project_basis_frame",
            "applied_before_exactification": True,
        },
    }
    if project_frame_revalidation is not None:
        kp_symm_exactification["project_frame_revalidation"] = dict(
            project_frame_revalidation
        )
    if fixed_target_certificate is not None:
        kp_symm_exactification["fixed_target_exactification"] = dict(
            fixed_target_certificate
        )
        kp_symm_exactification["production_joint_solver"] = {
            "kind": "persisted_project_owned_fixed_target",
            "free_solver_status": "not_run",
            "production_used_free_solver": False,
            "additional_basis_gauge_applied": False,
        }
    if final_joint_result is not None and isinstance(joint_payload, Mapping):
        kp_symm_exactification["joint_block_representation"] = dict(
            final_joint_result.artifact_metadata
        )
        kp_symm_exactification["joint_certification"] = {
            "joint_exactification": dict(joint_payload["report"]),
            "production_source": (
                "persisted_project_canonical_target"
                if uses_structure_transaction
                else "current_run_joint_exactification"
            ),
            "fixed_target_exactification": (
                None
                if fixed_target_certificate is None
                else dict(fixed_target_certificate)
            ),
            "project_target": (
                {
                    "status": "not_applicable",
                    "reason": "no_persisted_joint_structure_transaction",
                }
                if project_frame_revalidation is None
                else dict(project_frame_revalidation)
            ),
            "legacy_aliases": {
                "pre_gauge": "joint_exactification",
                "post_gauge": "joint_exactification",
            },
        }
    else:
        kp_symm_exactification["joint_block_representation"] = {
            "status": "not_applicable",
            "reason": "unsupported_or_incomplete_generator_set",
        }
    summary["kp_symm_exactification"] = kp_symm_exactification

    q_geometry = canonicalize_q_geometry(
        {"L1": ctx.q_model1, "L2": ctx.q_model2},
        summary["operations"],
        active_sectors=tuple(
            name
            for name, count in zip(("L1", "L2"), n_orb, strict=True)
            if int(count) > 0
        ),
    )
    q_model_metadata = summary.setdefault("q_model", {})
    q_model_metadata.update(
        {
            "role": "raw_model_q",
            "production_role": "symmetry_canonical_q",
            "canonicalization": dict(q_geometry.artifact),
            "package_arrays": {
                "raw_layer1": "__q_model_raw_layer1__",
                "raw_layer2": "__q_model_raw_layer2__",
                "canonical_layer1": "__q_model_canonical_layer1__",
                "canonical_layer2": "__q_model_canonical_layer2__",
            },
        }
    )

    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (ctx.output_dir / "manifest.json").write_text(payload, encoding="utf-8")
    (ctx.output_dir / "summary.json").write_text(payload, encoding="utf-8")
    _write_summary_md(ctx.output_dir / "summary.md", summary)
    if ctx.config.canonical_layout:
        _write_canonical_symmetry_outputs(
            ctx.output_dir,
            summary,
            q_geometry=q_geometry,
            joint_artifact_arrays=(
                None
                if final_joint_result is None
                else final_joint_result.artifact_arrays
            ),
        )
    return summary


def run_symmetry_projection_from_config(
    cfg_path: str,
    *,
    developer_outputs: bool | None = None,
    project_config: Mapping[str, Any] | None = None,
    gauge_report: GaugeAnchorReport | None = None,
    validate_full_space_covariance: bool = True,
    package_owner: str = "kp_symm",
    project_preparation: ProjectSymmetryPreparation | None = None,
) -> dict[str, Any]:
    persisted_handoff: ProjectionBasisSpec | None = None
    persisted_states: tuple[
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        dict[int, ProjectionState],
        ProjectionState,
    ] | None = None
    consume_persisted_project = (
        project_preparation is None
        and project_config is None
        and gauge_report is None
        and str(package_owner) == "kp_symm"
    )
    if consume_persisted_project:
        run_cfg = _load_projection_run_config(cfg_path, developer_outputs=developer_outputs)
        configured_output_dir = Path(
            _resolve(
                run_cfg.symm_cfg.get("output_dir", "symm_project"),
                run_cfg.cfg_dir,
            )
            or "symm_project"
        )
        _invalidate_stale_canonical_symmetry_outputs(configured_output_dir)
        ctx = _build_projection_run_context(
            run_cfg,
            validate_full_space_covariance=validate_full_space_covariance,
        )
        _invalidate_stale_canonical_symmetry_outputs(ctx.output_dir)
        project_dir = _resolve(run_cfg.project_cfg.get("out_dir"), run_cfg.cfg_dir)
        if project_dir is None:
            raise ValueError(
                "project.out_dir is required before kp symm can consume the projection basis"
            )
        persisted_handoff = _load_persisted_projection_basis_handoff(project_dir)
        if persisted_handoff is None:
            project_gauge_reused = False
            selected_gauge_candidate, gauge_report = _resolved_gauge_for_symmetry(
                ctx,
                gauge_report=None,
            )
            norb_fix_list = selected_gauge_candidate.resolved_norb_fix_list
        elif isinstance(persisted_handoff, GammaRoutedBasisSpec):
            persisted_states = _states_from_gamma_routed_handoff(
                ctx,
                persisted_handoff,
            )
            norb_fix_list = []
            gauge_report = _gauge_report_from_gamma_routed_handoff(
                persisted_handoff
            )
            project_gauge_reused = True
        else:
            if persisted_handoff.nlow_state_list != ctx.nlow_state_list:
                raise ValueError(
                    "persisted project nlow_state_list does not match the current config: "
                    f"{persisted_handoff.nlow_state_list!r} != {ctx.nlow_state_list!r}"
                )
            norb_fix_list = list(persisted_handoff.resolved_norb_fix_list)
            _validate_project_layer_lists(
                ctx.nlow_state_list,
                norb_fix_list,
                num_layer_list=ctx.num_layer_list,
                context="persisted project basis",
            )
            gauge_report = _gauge_report_from_persisted_handoff(persisted_handoff)
            project_gauge_reused = True
    elif project_preparation is None:
        run_cfg = _load_projection_run_config(cfg_path, developer_outputs=developer_outputs)
        if project_config is not None:
            run_cfg = replace(run_cfg, project_cfg=dict(project_config))
        ctx = _build_projection_run_context(
            run_cfg,
            validate_full_space_covariance=validate_full_space_covariance,
        )
        project_gauge_reused = gauge_report is not None
        selected_gauge_candidate, gauge_report = _resolved_gauge_for_symmetry(
            ctx,
            gauge_report=gauge_report,
        )
        norb_fix_list = selected_gauge_candidate.resolved_norb_fix_list
    else:
        ctx = project_preparation.context
        run_cfg = ctx.config
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        selected_gauge_candidate = project_preparation.selected_candidate
        prepared_report = project_preparation.gauge_report
        if gauge_report is not None and gauge_report is not prepared_report:
            raise ValueError("project symmetry preparation and gauge report do not match")
        gauge_report = prepared_report
        project_gauge_reused = True
        norb_fix_list = selected_gauge_candidate.resolved_norb_fix_list
    selected_states = (
        persisted_states
        if persisted_states is not None
        else ctx.selected_gauge_states
    )
    if selected_states is None:
        selected_states = _states_for_resolved_anchors(
            ctx,
            norb_fix_list,
            compute_heff=validate_full_space_covariance,
        )
    _states, source_states, target_states, first_state = selected_states
    low_dim = int(first_state.u_low.shape[1])
    if persisted_handoff is not None:
        if low_dim != persisted_handoff.model_dim:
            raise ValueError(
                "persisted project basis/projector dimension mismatch: "
                f"{persisted_handoff.model_dim} != {low_dim}"
            )
        artifact_identity = _resolve_symmetry_project_identity(
            ctx,
            handoff=persisted_handoff,
        )
    else:
        artifact_identity = _resolve_symmetry_project_identity(
            ctx,
            gauge_report=gauge_report,
            resolved_norb_fix_list=norb_fix_list,
            low_dim=low_dim,
        )
    write_basis_selection_report(ctx.output_dir, gauge_report)
    np.save(ctx.output_dir / "q_model_layer1.npy", ctx.q_model1)
    np.save(ctx.output_dir / "q_model_layer2.npy", ctx.q_model2)

    n_orb_for_exactification = (
        persisted_handoff.group_ranks
        if isinstance(persisted_handoff, GammaRoutedBasisSpec)
        else _sector_orbital_counts(
            ctx.q_model1,
            ctx.q_model2,
            ctx.nlow_state_list,
            low_dim=low_dim,
            num_layer_list=ctx.num_layer_list,
        )
    )
    summary = _initial_projection_summary(
        ctx,
        gauge_report=gauge_report,
        low_dim=low_dim,
        n_orb_for_exactification=n_orb_for_exactification,
        artifact_identity=artifact_identity,
    )
    if isinstance(persisted_handoff, GammaRoutedBasisSpec):
        summary["project_basis"].update(
            {
                "projection_basis_kind": persisted_handoff.projection_basis_kind,
                "nlow_state_list_layout": "not_applicable_gamma_routed",
                "layout_hash": persisted_handoff.layout.layout_hash,
                "frame_hash": persisted_handoff.frame_hash,
                "reference_frame_hash": persisted_handoff.reference_frame_hash,
                "heff_hash": persisted_handoff.heff_hash,
                "candidate_certificate_hash": (
                    persisted_handoff.candidate_certificate_hash
                ),
                "candidate_input_identity_hash": (
                    persisted_handoff.candidate_input_identity_hash
                ),
            }
        )
    summary["package_preparation"] = {
        "owner": str(package_owner),
        "status": "prepared",
        "project_resolved_gauge_reused": project_gauge_reused,
    }
    raw_low_matrices = _append_projected_operation_summaries(
        ctx,
        summary=summary,
        source_states=source_states,
        target_states=target_states,
        n_orb_for_exactification=n_orb_for_exactification,
        compute_heff_covariance=validate_full_space_covariance,
    )
    return _exactify_and_write_projection_summary(
        ctx,
        summary=summary,
        raw_low_matrices=raw_low_matrices,
        n_orb=n_orb_for_exactification,
    )
