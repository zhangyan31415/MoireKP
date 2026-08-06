from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .schema import M_EFFECTIVE_OPERATION_ALIASES
from .schema import canonical_operation_name_for_valley
from ..symmetry.action_schema import (
    PRODUCTION_MATRIX_KINDS,
    SOURCE_MATRIX_SEMANTICS,
    allows_inferred_action_metadata,
    complete_action_operation_metadata,
    validate_action_map,
)


_CANONICAL_OPERATION_NAMES = {"C3z", "C2", "TR", "C2T"}
_ARTIFACT_METADATA_KEYS = (
    "strict_metadata",
    "requires_model_exactification",
    "exactification_owner",
    "kp_symm_exactification",
    "source_action_definition",
    "model_action_definition",
)
_MANIFEST_AUTHORED_OPERATION_FIELDS = {
    "antiunitary",
    "k_map",
    "q_map",
    "sector_map",
    "matrix_file",
    "matrix_kind",
    "kind",
    "source_matrix_role",
    "source_gauge",
    "target_role",
    "gauge_correction",
    "antiunitary_convention",
    "spin_map",
    "valley_map",
    "model_action",
    "declared_model_action",
    "model_basis_action",
    "support_resolution",
    "pairs",
    "representation_pairs",
    "combined_raw_h_residual",
}


def _canonical_manifest_operation_name(name: str) -> str:
    if name == "C3":
        return "C3z"
    if name == "T":
        return "TR"
    return name


@dataclass
class LoadedSymmetrySource:
    source_type: str
    generator: Any | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _CertifiedPackedJoint:
    matrices: Mapping[str, np.ndarray]
    power_relations: Mapping[str, Mapping[str, Any]]
    actions: Mapping[str, Any]
    artifact_hash: str


@dataclass(frozen=True)
class _CertifiedResponseGauge:
    matrices: Mapping[str, np.ndarray]
    joint_route_actions: Mapping[str, Any]
    factorized_actions: Mapping[str, Any]
    basis_gauge: np.ndarray
    metadata: Mapping[str, Any]


def _certified_cyclotomic_response_gauge(
    *,
    matrices: Mapping[str, np.ndarray],
    joint_route_actions: Mapping[str, Any],
    factorized_actions: Mapping[str, Any],
    report: Mapping[str, Any],
    q_vectors: tuple[np.ndarray, ...],
) -> _CertifiedResponseGauge:
    """Materialize the already-certified scalar cyclotomic response gauge.

    ``kp symm`` persists the exact joint actions together with the common fiber
    gauge that maps them to their closest cyclotomic routes.  Model response
    generation needs that canonical gauge: otherwise harmless per-Q phases can
    turn a compact Fourier seed into many near-zero projected columns.
    """

    if str(report.get("status", "")) != "certified":
        raise ValueError("cyclotomic response gauge must be certified")
    if not joint_route_actions:
        raise ValueError("cyclotomic response gauge requires certified joint routes")
    angles = np.asarray(report.get("gauge_angles", ()), dtype=np.float64)
    if angles.ndim != 1 or not angles.size or not np.all(np.isfinite(angles)):
        raise ValueError("cyclotomic response gauge has invalid gauge angles")
    root_order = int(report.get("root_order", 0))
    root_exponents_raw = report.get("root_exponents")
    if root_order <= 1 or not isinstance(root_exponents_raw, Mapping):
        raise ValueError("cyclotomic response gauge lacks discrete route roots")
    certification_bound = float(report.get("common_gauge_certification_bound", -1.0))
    recorded_residual = float(report.get("common_gauge_residual_max", np.inf))
    if (
        not np.isfinite(certification_bound)
        or certification_bound < 0.0
        or not np.isfinite(recorded_residual)
        or recorded_residual > certification_bound
    ):
        raise ValueError("cyclotomic response gauge certificate is invalid")

    from ..symmetry.factorized_action import certify_factorized_action
    from ..symmetry.joint_exactification import (
        BlockRouteAction,
        materialize_block_route_action,
    )

    reference = next(iter(joint_route_actions.values()))
    if len(reference.fiber_dimensions) != angles.size or any(
        int(value) != 1 for value in reference.fiber_dimensions
    ):
        raise ValueError("cyclotomic response gauge requires scalar joint fibers")
    dimension = int(sum(reference.fiber_dimensions))
    diagonal = np.empty(dimension, dtype=np.complex128)
    for fiber, indices in enumerate(reference.fiber_indices):
        if len(indices) != 1:
            raise ValueError("cyclotomic response gauge requires scalar fiber indices")
        diagonal[int(indices[0])] = np.exp(1.0j * angles[fiber])
    basis_gauge = np.diag(diagonal)

    transformed_routes: dict[str, Any] = {}
    transformed_matrices: dict[str, np.ndarray] = {}
    actual_residual = 0.0
    for name, action in joint_route_actions.items():
        if (
            action.fiber_dimensions != reference.fiber_dimensions
            or action.fiber_indices != reference.fiber_indices
        ):
            raise ValueError(f"joint route layout differs for {name!r}")
        exponents_raw = root_exponents_raw.get(name)
        if not isinstance(exponents_raw, (list, tuple)) or len(exponents_raw) != angles.size:
            raise ValueError(f"cyclotomic response gauge lacks route roots for {name!r}")
        blocks: list[np.ndarray] = []
        for source, target in enumerate(action.fiber_permutation):
            source_phase = diagonal[action.fiber_indices[source][0]]
            if action.antiunitary:
                source_phase = source_phase.conjugate()
            target_phase = diagonal[action.fiber_indices[target][0]]
            block = np.asarray(
                target_phase.conjugate()
                * action.route_blocks[source]
                * source_phase,
                dtype=np.complex128,
            )
            expected_root = np.exp(
                2.0j * np.pi * int(exponents_raw[source]) / root_order
            )
            actual_residual = max(
                actual_residual,
                float(abs(complex(block[0, 0]) - expected_root)),
            )
            blocks.append(block)
        transformed = BlockRouteAction(
            name=action.name,
            antiunitary=action.antiunitary,
            fiber_permutation=action.fiber_permutation,
            fiber_dimensions=action.fiber_dimensions,
            fiber_indices=action.fiber_indices,
            route_blocks=tuple(blocks),
            unitarity_certification_bound=float(
                action.unitarity_certification_bound + 8.0 * certification_bound
            ),
        )
        transformed_routes[str(name)] = transformed
        transformed_matrices[str(name)] = materialize_block_route_action(transformed)
    if actual_residual > certification_bound:
        raise ValueError(
            "certified cyclotomic response gauge does not reproduce its route roots: "
            f"residual={actual_residual:.6e}, bound={certification_bound:.6e}"
        )

    for name, matrix in matrices.items():
        if name not in transformed_matrices:
            raise ValueError(f"response matrix {name!r} lacks a transformed joint route")
        source = np.asarray(matrix, dtype=np.complex128)
        right_gauge = basis_gauge.conjugate() if joint_route_actions[name].antiunitary else basis_gauge
        expected = basis_gauge.conjugate().T @ source @ right_gauge
        mismatch = float(np.linalg.norm(expected - transformed_matrices[name], ord="fro"))
        if mismatch > certification_bound * max(1, dimension):
            raise ValueError(
                f"cyclotomic response matrix mismatch for {name!r}: "
                f"residual={mismatch:.6e}"
            )

    transformed_factorized: dict[str, Any] = {}
    for name, action in factorized_actions.items():
        transformed_factorized[str(name)] = certify_factorized_action(
            name=str(name),
            matrix=transformed_matrices[str(name)],
            antiunitary=bool(action.antiunitary),
            k_forward=np.asarray(action.k_forward, dtype=np.float64),
            q_permutation=action.q_permutation,
            sector_permutation=action.sector_permutation,
            q_vectors=q_vectors,
            q_counts=action.q_counts,
            n_orb=action.n_orb,
            matrix_absolute_error_bound=float(
                certification_bound + action.matrix_certification_bound
            ),
            q_absolute_error_bound=float(action.q_certification_bound),
        )

    metadata = {
        "status": "certified",
        "version": "model_response_cyclotomic_gauge_v1",
        "source": "kp_symm_exactification.joint_block_representation.report.closest_cyclotomic_u1_gauge",
        "root_order": root_order,
        "gauge_angles": angles.tolist(),
        "recorded_common_gauge_residual_max": recorded_residual,
        "reconstructed_common_gauge_residual_max": actual_residual,
        "common_gauge_certification_bound": certification_bound,
        "hamiltonian_transform": "G_dagger_H_G",
        "antiunitary_action_transform": "G_dagger_D_G_conjugate",
    }
    return _CertifiedResponseGauge(
        matrices={
            str(name): transformed_matrices[str(name)] for name in matrices
        },
        joint_route_actions=transformed_routes,
        factorized_actions=transformed_factorized,
        basis_gauge=basis_gauge,
        metadata=metadata,
    )


def _canonical_structural_zero_copy(matrix: np.ndarray) -> np.ndarray:
    """Copy a complex matrix while encoding exact structural zeros as +0+0j."""

    result = np.array(
        matrix,
        dtype=np.complex128,
        order="C",
        copy=True,
    )
    structural_zero = result == 0.0
    if np.any(structural_zero):
        result.real[structural_zero] = 0.0
        result.imag[structural_zero] = 0.0
    return result


def _exact_central_sign_copy(matrix: np.ndarray, sign: int) -> np.ndarray:
    """Apply an exact central sign without changing structural-zero bits."""

    result = _canonical_structural_zero_copy(matrix)
    if int(sign) == -1:
        nonzero = result != 0.0
        result[nonzero] = np.negative(result[nonzero])
    return result


class MatrixSymmetryGenerator:
    """Low-energy symmetry representation loaded from kp symm/TAPW projection output."""

    def __init__(
        self,
        matrices: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
        *,
        factorized_actions: Mapping[str, Any] | None = None,
        joint_route_actions: Mapping[str, Any] | None = None,
        joint_artifact_hash: str | None = None,
        certified_power_relations: Mapping[str, Mapping[str, Any]] | None = None,
        model_basis_gauge: np.ndarray | None = None,
    ):
        self.matrices = {str(key): np.asarray(value, dtype=complex) for key, value in matrices.items()}
        self.metadata = dict(metadata)
        self.factorized_actions = {
            str(key): value for key, value in (factorized_actions or {}).items()
        }
        self.joint_route_actions = {
            str(key): value for key, value in (joint_route_actions or {}).items()
        }
        self.joint_artifact_hash = (
            None if joint_artifact_hash is None else str(joint_artifact_hash)
        )
        self.certified_power_relations = {
            str(key): dict(value)
            for key, value in (certified_power_relations or {}).items()
        }
        self.model_basis_gauge = (
            None
            if model_basis_gauge is None
            else np.asarray(model_basis_gauge, dtype=np.complex128)
        )
        unknown_relations = set(self.certified_power_relations) - set(self.matrices)
        if unknown_relations:
            raise ValueError(
                "certified power relations lack loaded matrices: "
                f"{sorted(unknown_relations)}"
            )
        c3_relation = self.certified_power_relations.get("C3z")
        if c3_relation is not None:
            power_raw = c3_relation.get("power")
            sign_raw = c3_relation.get("central_sign")
            artifact_hash = str(c3_relation.get("joint_artifact_hash", ""))
            if (
                isinstance(power_raw, (bool, np.bool_))
                or not isinstance(power_raw, (int, np.integer))
                or int(power_raw) != 3
                or isinstance(sign_raw, (bool, np.bool_))
                or not isinstance(sign_raw, (int, np.integer))
                or int(sign_raw) not in {-1, 1}
                or c3_relation.get("certification_source")
                != "joint_block_representation.presentation"
                or len(artifact_hash) != 64
                or any(character not in "0123456789abcdef" for character in artifact_hash)
            ):
                raise ValueError("invalid certified C3z cubic power relation")
        if "C3z" in self.matrices:
            self.metadata["c3_power_derivation"] = (
                {
                    "status": "certified",
                    "relation": dict(c3_relation),
                    "formulas": {
                        "-1": "D_dagger",
                        "2": "central_sign_times_D_dagger",
                        "-2": "central_sign_times_D",
                    },
                    "numeric_inverse_or_matrix_power_performed": False,
                    "structural_zero_policy": "exact_positive_zero_v1",
                }
                if c3_relation is not None
                else {
                    "status": "legacy_numeric",
                    "reason": "no_certified_joint_cubic_power_relation",
                    "numeric_inverse_or_matrix_power_performed": True,
                }
            )
        self.cached_operators: dict[tuple[str, Any], np.ndarray] = {}

    def get_factorized_action(self, name: str) -> Any | None:
        return self.factorized_actions.get(str(name))

    def get_joint_route_action(self, name: str) -> Any | None:
        return self.joint_route_actions.get(str(name))

    def get_operator(self, name: str, params: Any = None) -> np.ndarray:
        key = str(name)
        if key not in self.matrices:
            raise ValueError(f"Symmetry matrix for operation {name!r} is not loaded")
        param_key = params
        if key == "C3z":
            if params in {2, "2"}:
                param_key = 2
            elif params in {-1, "-1"}:
                param_key = -1
            elif params in {-2, "-2"}:
                param_key = -2
        cache_key = (key, param_key)
        if cache_key in self.cached_operators:
            return self.cached_operators[cache_key]
        matrix = self.matrices[key]
        power_relation = self.certified_power_relations.get(key)
        if key == "C3z" and param_key in {2, -1, -2} and power_relation is not None:
            sign = int(power_relation["central_sign"])
            if (
                int(power_relation.get("power", 0)) != 3
                or sign not in {-1, 1}
            ):
                raise ValueError("invalid certified C3z cubic power relation")
            if param_key == -1:
                operator = _canonical_structural_zero_copy(
                    matrix.conjugate().T,
                )
            elif param_key == 2:
                adjoint = _canonical_structural_zero_copy(
                    matrix.conjugate().T,
                )
                operator = _exact_central_sign_copy(adjoint, sign)
            else:
                operator = _exact_central_sign_copy(matrix, sign)
        elif key == "C3z" and param_key == 2:
            operator = matrix @ matrix
        elif key == "C3z" and param_key == -1:
            operator = np.linalg.inv(matrix)
        elif key == "C3z" and param_key == -2:
            inverse = self.get_operator(key, -1)
            operator = inverse @ inverse
        else:
            operator = matrix
        self.cached_operators[cache_key] = operator
        return operator


def _load_manifest(path: Path) -> dict[str, Any]:
    for name in ("manifest.json", "summary.json"):
        candidate = path / name
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Symmetry manifest root must be a mapping: {candidate}")
            return data
    return {}


def _packed_operation_metadata(name: str, valley_model: Mapping[str, Any]) -> dict[str, Any]:
    family = canonical_operation_name_for_valley(str(name), valley_model)
    if family == "C3z":
        action = {
            "antiunitary": False,
            "k_map": {"type": "rotation", "angle_deg": 120.0},
            "q_map": {"type": "rotation", "angle_deg": 120.0},
            "sector_map": "identity",
        }
    elif family == "C2T":
        action = {
            "antiunitary": True,
            "k_map": {"type": "reflection", "axis_deg": 180.0},
            "q_map": {"type": "reflection", "axis_deg": 180.0},
            "sector_map": "identity",
        }
    elif family == "TR":
        action = {
            "antiunitary": True,
            "k_map": {"type": "negation"},
            "q_map": {"type": "negation"},
            "sector_map": "identity",
        }
    elif family == "C2":
        action = {
            "antiunitary": False,
            "k_map": {"type": "reflection", "axis_deg": 0.0},
            "q_map": {"type": "reflection", "axis_deg": 0.0},
            "sector_map": "layer_exchange",
        }
    else:
        action = {}
    if action:
        action["antiunitary_convention"] = "U_K" if action["antiunitary"] else "none"
        action["matrix_kind"] = "continuum_internal_rep_exact"
        action["matrix_source"] = "kp_symm_exactified_action"
        action["spin_map"] = "from_kp_symm_output"
        action["valley_map"] = "identity"
        action.update({key: value for key, value in SOURCE_MATRIX_SEMANTICS.items()})
    return action


def _packed_manifest(path: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    packed = path / "representations.npz"
    if not packed.exists():
        return {}
    with np.load(packed, allow_pickle=False) as payload:
        if "__metadata_json__" in payload.files:
            metadata = json.loads(str(payload["__metadata_json__"].item()))
            return dict(metadata) if isinstance(metadata, Mapping) else {}
    valley_model = raw.get("valley_model", {})
    if not isinstance(valley_model, Mapping):
        valley_model = {}
    with np.load(packed, allow_pickle=False) as payload:
        operations = [
            {
                "name": _canonical_manifest_operation_name(str(name)),
                "operation": _canonical_manifest_operation_name(str(name)),
                "matrix_file": "representations.npz",
                "matrix_array_key": str(name),
                **_packed_operation_metadata(str(name), valley_model),
            }
            for name in payload.files
            if not str(name).startswith("__")
        ]
    return {
        "operations": operations,
        "exactification_owner": "kp_symm",
        "kp_symm_exactification": {"status": "exactified", "matrix_source": "kp_symm_exactified_action"},
    }


def _certified_joint_power_relations(
    restored: Any,
    joint_metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract only power identities certified by the loaded presentation."""

    if "C3z" not in restored.actions:
        return {}
    generators = {
        generator.name: generator
        for generator in restored.presentation.generators
    }
    generator = generators.get("C3z")
    if (
        generator is None
        or bool(generator.antiunitary)
        or bool(restored.actions["C3z"].antiunitary)
    ):
        raise ValueError("C3z must be a certified unitary generator")
    relations = [
        relation
        for relation in restored.presentation.relations
        if relation.lhs == ("C3z", "C3z", "C3z")
        and relation.rhs == ()
    ]
    if not relations:
        raise ValueError("C3z lacks a certified cubic power relation")
    phases = {complex(relation.central_phase) for relation in relations}
    if len(phases) != 1 or next(iter(phases)) not in {1.0 + 0.0j, -1.0 + 0.0j}:
        raise ValueError("C3z has an incompatible certified cubic power relation")
    phase = next(iter(phases))
    sign = 1 if phase == 1.0 + 0.0j else -1
    return {
        "C3z": {
            "power": 3,
            "central_sign": sign,
            "central_phase": [float(sign), 0.0],
            "relation_names": sorted(relation.name for relation in relations),
            "certification_source": "joint_block_representation.presentation",
            "joint_artifact_hash": str(joint_metadata.get("artifact_hash", "")),
            "presentation_source": str(restored.presentation.source),
        }
    }


def _certified_packed_joint_matrices(path: Path) -> _CertifiedPackedJoint | None:
    """Reconstruct and bind packed production matrices to their joint routes."""

    packed = path / "representations.npz"
    if not packed.is_file():
        return None
    try:
        with np.load(packed, allow_pickle=False) as payload:
            if "__metadata_json__" not in payload.files:
                return None
            metadata = json.loads(str(payload["__metadata_json__"].item()))
            if not isinstance(metadata, Mapping):
                raise ValueError("packed symmetry metadata must be a mapping")
            exactification = metadata.get("kp_symm_exactification")
            if not isinstance(exactification, Mapping):
                return None
            joint_metadata = exactification.get("joint_block_representation")
            if not isinstance(joint_metadata, Mapping):
                return None
            joint_status = str(joint_metadata.get("status", ""))
            if joint_status == "not_applicable":
                return None
            if joint_status != "certified":
                raise ValueError(
                    "packed joint artifact must declare status='certified' or "
                    "status='not_applicable'"
                )
            action_records = joint_metadata.get("actions")
            if not isinstance(action_records, list):
                raise ValueError("packed joint artifact lacks action records")
            route_keys: set[str] = set()
            for record in action_records:
                if not isinstance(record, Mapping):
                    raise ValueError("packed joint artifact has an invalid action record")
                keys = record.get("route_array_keys")
                if not isinstance(keys, list):
                    raise ValueError("packed joint artifact lacks route array keys")
                route_keys.update(str(key) for key in keys)
            joint_arrays = {
                key: np.array(payload[key], dtype=np.complex128, order="C", copy=True)
                for key in route_keys
            }

            from ..symmetry.joint_exactification import (
                JointExactificationConfig,
                certify_fixed_target_joint_result,
                load_joint_exactification_artifact,
                materialize_block_route_action,
            )

            restored = load_joint_exactification_artifact(
                joint_metadata,
                joint_arrays,
            )
            certified: dict[str, np.ndarray] = {}
            for name, action in restored.actions.items():
                if name not in payload.files:
                    raise ValueError(
                        f"packed production matrix {name!r} is missing"
                    )
                actual = np.asarray(payload[name])
                expected = materialize_block_route_action(action)
                if (
                    actual.dtype != np.dtype(np.complex128)
                    or actual.shape != expected.shape
                    or actual.tobytes(order="C") != expected.tobytes(order="C")
                ):
                    raise ValueError(
                        "packed production matrix does not match its certified joint "
                        f"routes for {name}"
                    )
                certified[name] = np.array(
                    expected,
                    dtype=np.complex128,
                    order="C",
                    copy=True,
                )

            if (
                joint_metadata.get("production_action_source")
                == "persisted_project_canonical_target"
            ):
                fixed_record = joint_metadata.get("fixed_target_exactification")
                if not isinstance(fixed_record, Mapping):
                    raise ValueError(
                        "packed fixed-target joint artifact lacks exactification metadata"
                    )
                if fixed_record.get("additional_basis_gauge_applied") is not False:
                    raise ValueError(
                        "packed fixed-target joint artifact declares an additional basis gauge"
                    )
                certificate = fixed_record.get("certificate")
                if not isinstance(certificate, Mapping):
                    raise ValueError(
                        "packed fixed-target joint artifact lacks its certificate"
                    )
                top_level_certificate = exactification.get(
                    "fixed_target_exactification"
                )
                if (
                    not isinstance(top_level_certificate, Mapping)
                    or dict(top_level_certificate) != dict(certificate)
                ):
                    raise ValueError(
                        "packed fixed-target top-level certificate does not match "
                        "the certified joint artifact"
                    )
                joint_report = joint_metadata.get("report")
                if (
                    not isinstance(joint_report, Mapping)
                    or joint_report.get("fixed_target_exactification")
                    != certificate
                ):
                    raise ValueError(
                        "packed fixed-target joint report does not match its certificate"
                    )
                joint_certification = exactification.get("joint_certification")
                if (
                    not isinstance(joint_certification, Mapping)
                    or joint_certification.get("fixed_target_exactification")
                    != certificate
                ):
                    raise ValueError(
                        "packed fixed-target certification summary does not match "
                        "the certified joint artifact"
                    )
                provenance = fixed_record.get("provenance")
                if not isinstance(provenance, Mapping):
                    raise ValueError(
                        "packed fixed-target joint artifact lacks target provenance"
                    )
                config_record = joint_metadata.get("config")
                if not isinstance(config_record, Mapping):
                    raise ValueError(
                        "packed fixed-target joint artifact lacks exactification config"
                    )
                try:
                    certify_fixed_target_joint_result(
                        restored.actions,
                        restored.presentation,
                        config=JointExactificationConfig(**dict(config_record)),
                        provenance=provenance,
                        fixed_target_certificate=certificate,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "packed fixed-target certificate validation failed: "
                        f"{exc}"
                    ) from exc
            power_relations = _certified_joint_power_relations(
                restored,
                joint_metadata,
            )
            return _CertifiedPackedJoint(
                matrices=certified,
                power_relations=power_relations,
                actions=dict(restored.actions),
                artifact_hash=str(restored.artifact_metadata["artifact_hash"]),
            )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid packed joint symmetry artifact: {exc}") from exc


def _load_matrix(path: Path) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = list(arr.files)
        if not keys:
            raise ValueError(f"Empty npz symmetry matrix file: {path}")
        return np.asarray(arr[keys[0]], dtype=complex)
    return np.asarray(arr, dtype=complex)


def _load_operation_matrix(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    if path.suffix == ".npz" and record.get("matrix_array_key") is not None:
        with np.load(path, allow_pickle=False) as payload:
            return np.asarray(payload[str(record["matrix_array_key"])], dtype=complex)
    return _load_matrix(path)


def _manifest_operations(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_ops = manifest.get("operations", manifest.get("matrices", []))
    if isinstance(manifest_ops, dict):
        flattened = []
        for valley_value in manifest_ops.values():
            if isinstance(valley_value, list):
                flattened.extend([item for item in valley_value if isinstance(item, Mapping)])
            elif isinstance(valley_value, dict):
                for op_name, op_value in valley_value.items():
                    if isinstance(op_value, Mapping):
                        row = dict(op_value)
                        row.setdefault("name", str(op_name))
                        flattened.append(row)
        manifest_ops = flattened
    elif not isinstance(manifest_ops, list):
        manifest_ops = []
    out: list[dict[str, Any]] = []
    for op in manifest_ops:
        if not isinstance(op, Mapping):
            continue
        row = dict(op)
        operation_name = row.get("operation", row.get("name"))
        if row.get("name") is None and operation_name is not None:
            row["name"] = _canonical_manifest_operation_name(str(operation_name))
        out.append(row)
    return out


def _operation_records(raw_operations: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_ops = _manifest_operations(manifest)
    manifest_index: dict[str, dict[str, Any]] = {}
    for op in manifest_ops:
        for key in (op.get("operation"), op.get("name")):
            if key is None:
                continue
            manifest_index[str(key)] = dict(op)
            manifest_index[_canonical_manifest_operation_name(str(key))] = dict(op)
    if raw_operations is None:
        raw_operations = []
    if isinstance(raw_operations, Mapping):
        mapped: list[dict[str, Any]] = []
        for key, value in raw_operations.items():
            row = dict(value) if isinstance(value, Mapping) else {}
            row.setdefault("name", str(key))
            mapped.append(row)
        raw_operations = mapped
    if not isinstance(raw_operations, list):
        raise ValueError("symmetry_source.operations must be a list")
    if not raw_operations:
        return [dict(op) for op in manifest_ops if isinstance(op, Mapping)]
    records: list[dict[str, Any]] = []
    for item in raw_operations:
        if isinstance(item, str):
            matched = manifest_index.get(item)
            if matched is None:
                raise ValueError(f"Operation {item!r} not found in kp_symm_output manifest")
            records.append(matched)
        elif isinstance(item, Mapping):
            row = dict(item)
            requested = row.get("name", row.get("operation"))
            if requested is None:
                records.append(row)
                continue
            source_key = str(row.get("operation", requested))
            matched = manifest_index.get(source_key)
            if matched is None:
                records.append(row)
                continue
            merged = dict(matched)
            merged.setdefault("operation", source_key)
            for key, value in row.items():
                if key in {"name", "user_name"}:
                    merged[key] = value
                elif key == "operation":
                    merged.setdefault("operation", value)
                elif key in _MANIFEST_AUTHORED_OPERATION_FIELDS and key in matched:
                    continue
                elif key not in matched:
                    merged[key] = value
            merged["name"] = str(requested)
            records.append(merged)
        else:
            raise ValueError(f"Unsupported symmetry operation metadata: {item!r}")
    return records


def _validate_map(raw: Any, *, field: str) -> Any:
    return validate_action_map(raw, field=field)


def _pairs_for_matrix_kind(record: Mapping[str, Any], matrix_kind: str) -> list[Any]:
    pairs = record.get("pairs", [])
    return pairs if isinstance(pairs, list) else []


def _quality_warnings_from_pairs(record: Mapping[str, Any], matrix_kind: str) -> list[str]:
    warnings_out: list[str] = []
    for pair in _pairs_for_matrix_kind(record, matrix_kind):
        if isinstance(pair, Mapping):
            warnings_raw = pair.get("quality_warnings", [])
            if isinstance(warnings_raw, list):
                warnings_out.extend(str(item) for item in warnings_raw)
    return warnings_out


def _metric_from_pairs(record: Mapping[str, Any], use: str, key: str, matrix_kind: str) -> float | None:
    pairs = _pairs_for_matrix_kind(record, matrix_kind)
    if not isinstance(pairs, list) or not pairs:
        return None
    first = pairs[0]
    if not isinstance(first, Mapping):
        return None
    use_block = first.get(use, {})
    if isinstance(use_block, Mapping) and key in use_block:
        value = use_block[key]
        if value is None:
            return None
        return float(value)
    return None


def _manifest_matrix_kind(record: Mapping[str, Any]) -> str:
    raw = record.get("matrix_kind", record.get("kind"))
    if raw:
        return str(raw)
    raise ValueError("kp_symm_output operation requires explicit matrix_kind metadata")


def _manifest_default_matrix_kind(manifest: Mapping[str, Any]) -> str | None:
    raw = manifest.get("default_matrix_kind", manifest.get("matrix_kind"))
    if raw:
        return str(raw)
    return None


def _complete_operation_record(record: Mapping[str, Any], *, use: str, allow_inferred: bool = False) -> dict[str, Any]:
    out = dict(record)
    if not allow_inferred:
        inferred_markers = [
            key
            for key in ("inferred_fields", "defaulted_fields")
            if out.get(key)
        ]
        if inferred_markers:
            raise ValueError(
                f"kp_symm_output operation {out.get('name', out.get('operation', ''))!r} "
                f"cannot use inferred production action metadata: {inferred_markers}"
            )
    raw_name = str(out.get("name", ""))
    alias = M_EFFECTIVE_OPERATION_ALIASES.get(raw_name)
    if alias is not None:
        out.setdefault("operation_alias", raw_name)
        out.setdefault("effective_name", str(alias["effective_name"]))
        out.setdefault("representation_level", "effective_single_spin")
        out.setdefault("canonical_physical_operation", str(alias["canonical"]))
        out.setdefault("physical_parent", str(alias["canonical"]))
        out.setdefault("approximation", {"kind": "spin_SU2_effective_block"})
        if "derived_from" in alias:
            out.setdefault("derived_from", list(alias["derived_from"]))
        out["name"] = str(alias["canonical"])
    elif raw_name:
        out["name"] = _canonical_manifest_operation_name(raw_name)
    if out.get("operation") is not None:
        out["operation"] = _canonical_manifest_operation_name(str(out["operation"]))
    canonical_name = str(out.get("name", ""))
    if canonical_name not in _CANONICAL_OPERATION_NAMES:
        raise ValueError(f"Unsupported canonical operation {canonical_name!r}")
    out = complete_action_operation_metadata(
        out,
        allow_inferred=allow_inferred,
        context="kp_symm_output operation",
    )
    matrix_kind = str(out["matrix_kind"])
    support_resolution = out.get("support_resolution")
    if not isinstance(support_resolution, Mapping):
        model_basis_action = out.get("model_basis_action")
        if isinstance(model_basis_action, Mapping):
            support_resolution = model_basis_action.get("support_resolution")
    support_matrix_source = None
    if isinstance(support_resolution, Mapping):
        support_matrix_source = support_resolution.get("support_matrix_source")
    if support_matrix_source == "representation":
        out.setdefault(
            "representation_projection_diagnostic",
            {
                "status": "raw_action_exactification_problem",
                "support_matrix_source": "representation",
                "representation_support_cleaner_than_raw_action": True,
                "projection_warnings": _quality_warnings_from_pairs(out, "representation"),
            },
        )
    quality_warnings = _quality_warnings_from_pairs(out, matrix_kind)
    if quality_warnings:
        out["projection_quality_warnings"] = quality_warnings
    out["residual"] = out.get("residual", _metric_from_pairs(out, use, "heff_covariance_residual", matrix_kind))
    out["leakage"] = out.get("leakage", _metric_from_pairs(out, use, "subspace_leakage", matrix_kind))
    if out["residual"] is None:
        out["residual"] = out.get("combined_raw_h_residual", 0.0)
    if out["leakage"] is None:
        out["leakage"] = 0.0
    return out


def _validate_exactified_operation_record(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if str(record.get("matrix_kind", "")) != "continuum_internal_rep_exact":
        return
    name = str(record.get("name", record.get("operation", "")))
    if str(manifest.get("exactification_owner", "")) != "kp_symm":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires kp_symm exactification provenance"
        )
    if str(record.get("matrix_source", "")) != "kp_symm_exactified_action":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires matrix_source='kp_symm_exactified_action'"
        )
    if str(record.get("source_matrix_role", "")) != "raw_h_sewing_action":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires source_matrix_role='raw_h_sewing_action'"
        )
    if str(record.get("target_role", "")) != "continuum_internal_rep":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires target_role='continuum_internal_rep'"
        )
    report = record.get("source_matrix_projection_report")
    if not isinstance(report, Mapping):
        return
    if report.get("source_matrix_role") is not None and str(report.get("source_matrix_role")) != "raw_h_sewing_action":
        raise ValueError(
            f"continuum_internal_rep_exact operation {name!r} requires raw-H action exactification"
        )
    support_resolution = report.get("support_resolution")
    if isinstance(support_resolution, Mapping):
        if support_resolution.get("matrix_kind") is not None and str(support_resolution.get("matrix_kind")) != "action":
            raise ValueError(
                f"continuum_internal_rep_exact operation {name!r} requires source support matrix_kind='action'"
            )
        if support_resolution.get("matrix_source") is not None and str(support_resolution.get("matrix_source")) != "raw_h_action_projection":
            raise ValueError(
                f"continuum_internal_rep_exact operation {name!r} requires raw-H action exactification"
            )


def _add_inferred_field(record: dict[str, Any], field: str) -> None:
    raw_fields = record.get("inferred_fields", [])
    fields = [str(item) for item in raw_fields] if isinstance(raw_fields, list) else []
    record["inferred_fields"] = sorted({*fields, str(field)})


def load_symmetry_source(raw: Mapping[str, Any] | None, *, base: Path, expected_dim: int | None = None) -> LoadedSymmetrySource:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("symmetry_source must be a mapping")
    source_type = str(raw.get("type", "none"))
    if source_type == "none":
        return LoadedSymmetrySource(source_type="none", generator=None, metadata={"operations": []})
    if source_type == "toy_generator":
        if not bool(raw.get("allow", False)):
            raise ValueError("toy_generator requires explicit symmetry_source.allow: true")
        warnings.warn("toy_generator is not a validated TAPW low-energy representation", RuntimeWarning, stacklevel=2)
        return LoadedSymmetrySource(source_type="toy_generator", generator=None, metadata={"operations": [], "basis_template": raw.get("basis_template")})
    if source_type != "kp_symm_output":
        raise ValueError(f"Unsupported symmetry_source.type: {source_type}")

    path_raw = raw.get("path")
    if path_raw is None:
        raise ValueError("symmetry_source.type=kp_symm_output requires path")
    path = Path(path_raw)
    if not path.is_absolute():
        path = (base / path).resolve()
    manifest = _load_manifest(path)
    if not manifest:
        manifest = _packed_manifest(path, raw)
    certified_packed_joint = _certified_packed_joint_matrices(path)
    records = _operation_records(raw.get("operations"), manifest)
    matrices: dict[str, np.ndarray] = {}
    metadata_records: list[dict[str, Any]] = []
    use = str(raw.get("use", "raw"))
    allow_inferred = allows_inferred_action_metadata(raw) or allows_inferred_action_metadata(manifest)
    root_matrix_kind = raw.get("matrix_kind", raw.get("kind"))
    manifest_matrix_kind = _manifest_default_matrix_kind(manifest) if allow_inferred else None
    for matrix_kind_raw in (root_matrix_kind, manifest_matrix_kind):
        if matrix_kind_raw is not None and str(matrix_kind_raw) not in PRODUCTION_MATRIX_KINDS:
            raise ValueError("production symmetry matrices must use raw-action or kp_symm exactified matrices")
    inferred_matrix_kind = root_matrix_kind if root_matrix_kind is not None else manifest_matrix_kind
    if not allow_inferred:
        inferred_matrix_kind = None
    for record in records:
        record = dict(record)
        used_inferred_matrix_kind = False
        if allow_inferred and inferred_matrix_kind is not None and ("matrix_kind" not in record and "kind" not in record):
            record["matrix_kind"] = str(inferred_matrix_kind)
            _add_inferred_field(record, "matrix_kind")
            used_inferred_matrix_kind = True
        elif "matrix_kind" in record or "kind" in record:
            record["matrix_kind"] = _manifest_matrix_kind(record)
        record = _complete_operation_record(record, use=use, allow_inferred=allow_inferred)
        if used_inferred_matrix_kind:
            _add_inferred_field(record, "matrix_kind")
        _validate_exactified_operation_record(record, manifest)
        name = str(record["name"])
        matrix_file = Path(str(record["matrix_file"]))
        if not matrix_file.is_absolute():
            matrix_file = path / matrix_file
        matrix = _load_operation_matrix(matrix_file, record)
        if (
            str(record.get("matrix_kind", "")) == "continuum_internal_rep_exact"
            and matrix_file == path / "representations.npz"
            and certified_packed_joint is not None
        ):
            expected = certified_packed_joint.matrices.get(name)
            if expected is None:
                raise ValueError(
                    f"packed exact symmetry matrix {name!r} lacks a certified joint action"
                )
            if (
                matrix.dtype != np.dtype(np.complex128)
                or matrix.shape != expected.shape
                or matrix.tobytes(order="C") != expected.tobytes(order="C")
            ):
                raise ValueError(
                    "packed exact symmetry matrix does not match its certified joint "
                    f"action for {name}"
                )
        if expected_dim is not None and matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"Symmetry matrix {matrix_file} has shape {matrix.shape}, expected {(expected_dim, expected_dim)}")
        matrices[name] = matrix
        metadata_record = dict(record)
        metadata_record["matrix_file"] = str(matrix_file)
        metadata_record["use"] = use
        metadata_records.append(metadata_record)
    artifact_metadata = {
        key: manifest[key]
        for key in _ARTIFACT_METADATA_KEYS
        if key in manifest
    }
    certified_power_relations = (
        {}
        if certified_packed_joint is None
        else {
            name: dict(record)
            for name, record in certified_packed_joint.power_relations.items()
            if name in matrices
        }
    )
    metadata = {
        "operations": metadata_records,
        "path": str(path),
        "use": use,
        **artifact_metadata,
    }
    if certified_power_relations:
        metadata["certified_power_relations"] = certified_power_relations
    factorized_actions: dict[str, Any] = {}
    packed_path = path / "representations.npz"
    if packed_path.is_file():
        from ..symmetry.factorized_action import load_factorized_actions_from_npz

        factorized_actions = load_factorized_actions_from_npz(packed_path)
    joint_route_actions = (
        {} if certified_packed_joint is None else dict(certified_packed_joint.actions)
    )
    model_basis_gauge: np.ndarray | None = None
    exactification = manifest.get("kp_symm_exactification")
    joint_metadata = (
        exactification.get("joint_block_representation")
        if isinstance(exactification, Mapping)
        else None
    )
    joint_report = (
        joint_metadata.get("report")
        if isinstance(joint_metadata, Mapping)
        else None
    )
    closest_report = (
        joint_report.get("closest_cyclotomic_u1_gauge")
        if isinstance(joint_report, Mapping)
        else None
    )
    if (
        isinstance(closest_report, Mapping)
        and str(closest_report.get("status", "")) == "certified"
        and joint_route_actions
    ):
        with np.load(packed_path, allow_pickle=False) as payload:
            q_keys = (
                "__q_model_canonical_layer1__",
                "__q_model_canonical_layer2__",
            )
            if not all(key in payload.files for key in q_keys):
                raise ValueError(
                    "certified cyclotomic response gauge requires canonical Q arrays"
                )
            q_vectors = tuple(
                np.asarray(payload[key], dtype=np.float64) for key in q_keys
            )
        reframed = _certified_cyclotomic_response_gauge(
            matrices=matrices,
            joint_route_actions=joint_route_actions,
            factorized_actions=factorized_actions,
            report=closest_report,
            q_vectors=q_vectors,
        )
        matrices = dict(reframed.matrices)
        joint_route_actions = dict(reframed.joint_route_actions)
        factorized_actions = dict(reframed.factorized_actions)
        model_basis_gauge = reframed.basis_gauge
        metadata["model_response_basis_gauge"] = dict(reframed.metadata)
    generator = MatrixSymmetryGenerator(
        matrices,
        metadata,
        factorized_actions=factorized_actions,
        joint_route_actions=joint_route_actions,
        joint_artifact_hash=(
            None
            if certified_packed_joint is None
            else certified_packed_joint.artifact_hash
        ),
        certified_power_relations=certified_power_relations,
        model_basis_gauge=model_basis_gauge,
    )
    return LoadedSymmetrySource(
        source_type="kp_symm_output",
        generator=generator,
        metadata=dict(generator.metadata),
    )
