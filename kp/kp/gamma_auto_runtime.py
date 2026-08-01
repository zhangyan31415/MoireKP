"""Filesystem/CLI adapter for the pure automatic Gamma producer.

The physics producer remains side-effect free.  This module owns only strict
config materialization, the PENDING-to-final transaction boundary, and the
canonical projection files committed by ``kp project``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import io
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .gamma_auto_producer import (
    GammaAutomaticProducerInputs,
    GammaAutomaticSelectionConfig,
    GammaAutomaticSelectionPreparation,
    GammaRawOperationSpec,
    evaluate_gamma_automatic_selection,
    prepare_gamma_automatic_selection,
)
from .blocks.gamma_layout import (
    GammaRowLayout,
    infer_gamma_raw_action_q_permutations,
)
from .identity import hash_array
from .projection_handoff import (
    GAMMA_SAMPLED_K_GRAY_TOLERANCE,
    GAMMA_SAMPLED_K_MATCH_TOLERANCE,
    GammaRoutedBasisSpec,
    save_gamma_routed_basis_spec,
)
from .selection_orchestration import (
    CaseSelectionInputs,
    PendingCaseSelection,
    ResolvedProjectionSelection,
    resolve_case_selection,
)
from .symmetry.joint_exactification import (
    compile_continuum_magnetic_presentation,
)


@dataclass(frozen=True)
class GammaAutomaticRuntimePreparation:
    """Identity-complete auto preparation plus release output coordinates."""

    selection_preparation: GammaAutomaticSelectionPreparation
    output_directory: Path


def _load_strict_gamma_auto_config(value: Any) -> GammaAutomaticSelectionConfig:
    """Parse the complete hard policy without inserting runtime defaults."""

    return GammaAutomaticSelectionConfig.from_normalized_config(value)


def _strict_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a non-placeholder lowercase SHA-256 string"
        )
    text = value
    if (
        len(text) != 64
        or text == "0" * 64
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(
            f"{field} must be a non-placeholder lowercase SHA-256 string"
        )
    return text


def _selected_manifest_basis_hash(ctx: Any) -> str:
    """Resolve one certified source basis from the selected operation entries."""

    selected_hashes: list[str] = []
    for request in ctx.operation_requests:
        output_name = str(request.get("output", ""))
        payload = ctx.operation_payloads.get(output_name)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"selected TAPW symmetry operation {output_name!r} lacks a payload"
            )
        entry = payload.get("entry")
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"selected TAPW symmetry operation {output_name!r} lacks a manifest entry"
            )
        selected_hashes.append(
            _strict_sha256(
                entry.get("basis_hash"),
                field=(
                    f"selected TAPW symmetry operation {output_name!r} basis_hash"
                ),
            )
        )
    if not selected_hashes:
        raise ValueError("automatic Gamma requires selected TAPW symmetry operations")
    unique_hashes = set(selected_hashes)
    if len(unique_hashes) != 1:
        raise ValueError(
            "selected TAPW symmetry operations must share one common basis_hash"
        )
    source_basis_hash = selected_hashes[0]

    if "basis_hash" in ctx.manifest:
        top_level_hash = _strict_sha256(
            ctx.manifest["basis_hash"],
            field="TAPW symmetry top-level manifest basis_hash",
        )
        if top_level_hash != source_basis_hash:
            raise ValueError(
                "TAPW symmetry top-level manifest basis_hash disagrees with "
                "the selected operation basis_hash"
            )
    return source_basis_hash


def _manifest_sector_map(value: Any, projection: Any) -> tuple[int, int]:
    names = ("L1", "L2")
    targets = tuple(
        projection._sector_target(source, value) for source in names
    )
    if any(target not in names for target in targets):
        raise ValueError(f"Gamma manifest sector_map has unknown targets {targets}")
    mapped = tuple(names.index(target) for target in targets)
    if tuple(sorted(mapped)) != (0, 1):
        raise ValueError("Gamma manifest sector_map must bijectively map L1/L2")
    return mapped  # type: ignore[return-value]


def _strict_sampled_k_map_matrix(operation: str, value: Any) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ValueError(f"operation {operation!r} k_map must be a mapping")
    map_type = str(value.get("type", "")).strip().lower()
    allowed_keys: dict[str, set[str]] = {
        "identity": {"type"},
        "negation": {"type"},
        "rotation": {"type", "angle_deg"},
        "reflection": {"type", "axis_deg", "reflection_axis_convention"},
    }
    if map_type not in allowed_keys:
        raise ValueError(
            f"operation {operation!r} has unsupported k_map.type {value.get('type')!r}"
        )
    if set(value) != allowed_keys[map_type]:
        raise ValueError(f"operation {operation!r} k_map has an invalid strict schema")
    if map_type == "identity":
        return np.eye(2, dtype=np.float64)
    if map_type == "negation":
        return -np.eye(2, dtype=np.float64)
    field = "angle_deg" if map_type == "rotation" else "axis_deg"
    raw_angle = value[field]
    if isinstance(raw_angle, (bool, np.bool_)) or not isinstance(raw_angle, Real):
        raise ValueError(
            f"operation {operation!r} k_map angle must be finite numeric"
        )
    angle = float(raw_angle)
    if not np.isfinite(angle):
        raise ValueError(f"operation {operation!r} k_map angle must be finite")
    theta = np.deg2rad(angle)
    if map_type == "rotation":
        return np.asarray(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )
    if value["reflection_axis_convention"] != "mirror_axis_deg":
        raise ValueError(
            f"operation {operation!r} reflection k_map requires mirror_axis_deg"
        )
    axis = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    return 2.0 * np.outer(axis, axis) - np.eye(2, dtype=np.float64)


def _infer_sampled_k_pairs(
    *,
    operation: str,
    kpoints: Any,
    k_map: Any,
) -> tuple[tuple[int, int], ...]:
    """Enumerate the exact intersection of a sampled k-set with its image."""

    points = np.asarray(kpoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= 0 or points.shape[1] != 2:
        raise ValueError("automatic Gamma sampled kpoints must have shape (Nk, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("automatic Gamma sampled kpoints must be finite")
    matrix = _strict_sampled_k_map_matrix(operation, k_map)
    pairs: list[tuple[int, int]] = []
    for source_index, source in enumerate(points):
        mapped = matrix @ source
        distances = np.linalg.norm(points - mapped, axis=1)
        exact_targets = np.flatnonzero(
            distances <= GAMMA_SAMPLED_K_MATCH_TOLERANCE
        )
        gray_targets = np.flatnonzero(
            (distances > GAMMA_SAMPLED_K_MATCH_TOLERANCE)
            & (distances <= GAMMA_SAMPLED_K_GRAY_TOLERANCE)
        )
        if gray_targets.size:
            raise ValueError(
                f"operation {operation!r} sampled k-route has a numerical gray zone "
                f"at source index {source_index}"
            )
        pairs.extend(
            (int(target_index), int(source_index))
            for target_index in exact_targets
        )
    if not pairs:
        raise ValueError(
            f"operation {operation!r} has no exact sampled k-pairs on the actual k-set"
        )
    return tuple(pairs)


def _geometry_q_route(
    *,
    projection: Any,
    model_action: Mapping[str, Any],
    q_model1: np.ndarray,
    q_model2: np.ndarray,
    num_layer_list: list[int],
    sector_map: tuple[int, int],
    q_count: int,
    tolerance: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Cross-check the raw-H support route against the manifest Q map."""

    basis_action = projection._basis_action_for_candidate(
        action=dict(model_action),
        q_model1=q_model1,
        q_model2=q_model2,
        nlow_state_list=[[0] for _layer in range(sum(num_layer_list))],
        low_dim=None,
        num_layer_list=num_layer_list,
        tol=tolerance,
    )
    if basis_action.get("complete") is not True:
        raise ValueError(
            "Gamma manifest q_map does not define a complete ordered Q route"
        )
    by_name = {"L1": 0, "L2": 1}
    routes: list[list[int | None]] = [
        [None for _ in range(q_count)],
        [None for _ in range(q_count)],
    ]
    for raw_item in basis_action.get("items", []):
        if not isinstance(raw_item, Mapping):
            raise ValueError("Gamma manifest q_map route item is invalid")
        source_name = str(raw_item.get("source_sector", ""))
        target_name = str(raw_item.get("target_sector", ""))
        if source_name not in by_name or target_name not in by_name:
            raise ValueError("Gamma manifest q_map route uses an unknown sector")
        source_group = by_name[source_name]
        target_group = by_name[target_name]
        source_q = int(raw_item["source_q_index"])
        target_q = int(raw_item["target_q_index"])
        if target_group != sector_map[source_group]:
            raise ValueError("Gamma manifest q_map and sector_map disagree")
        if not (0 <= source_q < q_count and 0 <= target_q < q_count):
            raise ValueError("Gamma manifest q_map route index is out of range")
        previous = routes[source_group][source_q]
        if previous is not None and previous != target_q:
            raise ValueError("Gamma manifest q_map route is ambiguous")
        routes[source_group][source_q] = target_q
    frozen: list[tuple[int, ...]] = []
    for group, route in enumerate(routes):
        if any(value is None for value in route):
            raise ValueError(
                f"Gamma manifest q_map route is incomplete for source group {group}"
            )
        concrete = tuple(int(value) for value in route if value is not None)
        if tuple(sorted(concrete)) != tuple(range(q_count)):
            raise ValueError("Gamma manifest q_map route is not bijective")
        frozen.append(concrete)
    return frozen[0], frozen[1]


def _bytes_buffer(content: bytes) -> io.BytesIO:
    return io.BytesIO(content)


def _npy_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _load_npy_bytes(content: bytes) -> np.ndarray:
    return np.load(_bytes_buffer(content), allow_pickle=False)


def _npz_bytes(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    arrays = {str(name): np.asarray(value) for name, value in payload.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("automatic Gamma release payloads cannot contain object arrays")
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _serialize_gamma_handoff(handoff: GammaRoutedBasisSpec) -> bytes:
    """Use the authoritative strict writer, then return the exact archive bytes."""

    with tempfile.TemporaryDirectory(prefix="kp-gamma-handoff-") as directory:
        path = Path(directory) / "basis.npz"
        save_gamma_routed_basis_spec(path, handoff)
        return path.read_bytes()


def _gamma_projected_spin_operator(
    handoff: GammaRoutedBasisSpec,
) -> np.ndarray:
    """Project canonical full-space Sz through each certified routed frame."""

    if not isinstance(handoff, GammaRoutedBasisSpec):
        raise TypeError("handoff must be a GammaRoutedBasisSpec")
    layout = handoff.layout
    if layout.spin_scope != "spinful_all" or layout.spin_labels != ("up", "down"):
        raise ValueError(
            "automatic Gamma spin projection requires the canonical full-spin layout"
        )
    spin_eigenvalue = {"up": 1.0, "down": -1.0}
    full_spin_diagonal = np.asarray(
        [
            spin_eigenvalue[address.spin_label]
            for address in layout.addresses_by_full_row
        ],
        dtype=np.complex128,
    )
    if full_spin_diagonal.shape != (layout.full_dimension,):
        raise ValueError("canonical Gamma spin rows do not cover the full source space")

    projected_rows: list[np.ndarray] = []
    for k_index in handoff.k_indices:
        routed_frame, _ = handoff.assemble_for_k(k_index, include_high=False)
        if routed_frame.shape != (layout.full_dimension, handoff.model_dim):
            raise ValueError(
                f"routed Gamma frame at k={k_index} has shape {routed_frame.shape}, "
                f"expected {(layout.full_dimension, handoff.model_dim)}"
            )
        raw = routed_frame.conj().T @ (
            full_spin_diagonal[:, np.newaxis] * routed_frame
        )
        projected = 0.5 * (raw + raw.conj().T)
        if not np.all(np.isfinite(projected)):
            raise ValueError(
                f"routed Gamma spin projection is nonfinite at k={k_index}"
            )
        projected_rows.append(np.ascontiguousarray(projected, dtype=np.complex128))
    return np.stack(projected_rows, axis=0)


def _gamma_project_payloads(
    handoff: GammaRoutedBasisSpec,
    kpoints: Any,
) -> dict[str, bytes]:
    """Serialize the four canonical, mutually bound routed projection files."""

    points = np.asarray(kpoints, dtype=np.float64)
    authoritative_points = np.asarray(handoff.kpoints, dtype=np.float64)
    if points.shape != authoritative_points.shape or not np.array_equal(
        points, authoritative_points
    ):
        raise ValueError("runtime kpoints differ from the certified Gamma handoff")
    if handoff.kpoints_hash != hash_array(points):
        raise ValueError("certified Gamma handoff kpoints hash mismatch")

    heff = np.asarray(handoff.authoritative_heff, dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(heff)
    spin_operator = _gamma_projected_spin_operator(handoff)
    wavefunction_payload = {
        "wavefunctions": eigenvectors,
        "eigenvalues": eigenvalues,
        "k_indices": np.asarray(handoff.k_indices, dtype=np.int64),
        "spin_convention": np.asarray("all"),
        "spin_operator": spin_operator,
        **{
            str(name): np.asarray(value)
            for name, value in handoff.artifact_identity.items()
        },
    }
    return {
        "basis.npz": _serialize_gamma_handoff(handoff),
        "heff.npy": _npy_bytes(heff),
        "kpoints.npy": _npy_bytes(points),
        "wavefunctions.npz": _npz_bytes(wavefunction_payload),
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_canonical_payloads(
    output_directory: Path,
    payloads: Mapping[str, bytes],
) -> None:
    """Replace each payload durably; the selection marker remains the commit point."""

    output_directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(payloads):
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"invalid automatic Gamma payload name {name!r}")
        target = output_directory / name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=output_directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(bytes(payloads[name]))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(output_directory)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_gamma_automatic_runtime(
    cfg_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> GammaAutomaticRuntimePreparation:
    """Materialize real Gamma inputs and freeze the final preselection identity."""

    # Imports are local so the command module can dispatch here without a
    # module-import cycle.
    from .cli import _apply_project_overrides, _load_project_source_kpoints
    from .config.case import normalize_case_config
    from .symmetry import projection
    import yaml

    path = Path(cfg_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = normalize_case_config(yaml.safe_load(handle), config_path=path)
    material = dict(cfg.get("material", {}))
    project_cfg = _apply_project_overrides(
        dict(cfg.get("project", {})),
        None if overrides is None else dict(overrides),
    )
    if str(material.get("spin", "all")).strip().lower() != "all":
        raise ValueError("automatic Gamma v1 requires material.spin='all'")
    symm = cfg.get("symm", {})
    if isinstance(symm, Mapping) and str(
        symm.get("spin", material.get("spin", "all"))
    ).strip().lower() != "all":
        raise ValueError("automatic Gamma v1 requires symm.spin='all'")
    if not isinstance(symm, Mapping):
        raise ValueError("automatic Gamma v1 requires a symm mapping")
    if "tolerance" not in symm:
        raise ValueError("automatic Gamma v1 requires explicit symm.tolerance")
    tolerance_raw = symm["tolerance"]
    if isinstance(tolerance_raw, (bool, np.bool_)):
        raise ValueError("automatic Gamma symm.tolerance must be finite and positive")
    tolerance = float(tolerance_raw)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("automatic Gamma symm.tolerance must be finite and positive")
    selection_config = _load_strict_gamma_auto_config(
        project_cfg.get("selection", {})
    )

    run_cfg = projection._load_projection_run_config(
        str(path), developer_outputs=False
    )
    run_cfg = replace(run_cfg, project_cfg=project_cfg)
    if projection._valley_family(run_cfg.valley) != "Gamma":
        raise ValueError("automatic Gamma v1 requires symm.valley in the Gamma family")
    hamk_file = projection._resolve(
        run_cfg.material.get("hamk_file"), run_cfg.cfg_dir
    )
    if hamk_file is None:
        raise ValueError("material.hamk_file is required")
    sampled_kpoints: np.ndarray | None = None

    def resolve_packed_k_route(
        entry: dict[str, Any],
        operation: str,
        nk: int,
    ) -> tuple[tuple[int, int], ...]:
        nonlocal sampled_kpoints
        route_fields = {
            "k_pairs",
            "pairs",
            "target_indices",
            "source_indices",
            "source_k_rule",
            "k_rule",
            "target_k_rule",
        }
        declared_pairs = (
            tuple(projection._pairs_from_entry(entry, nk))
            if route_fields.intersection(entry)
            else None
        )
        if sampled_kpoints is None:
            sampled_kpoints = _load_project_source_kpoints(
                cfg,
                material,
                cfg_dir=run_cfg.cfg_dir,
                hamk_file=str(hamk_file),
                nk=nk,
                project_indices=tuple(range(nk)),
            )
        pairs = _infer_sampled_k_pairs(
            operation=operation,
            kpoints=sampled_kpoints,
            k_map=entry.get("k_map"),
        )
        if declared_pairs is not None and sorted(declared_pairs) != sorted(pairs):
            raise ValueError(
                f"operation {operation!r} packed k-route disagrees with actual "
                "sampled k-set intersection"
            )
        entry["k_pairs"] = [list(pair) for pair in pairs]
        entry["k_pair_source"] = "actual_sampled_k_set_intersection"
        entry["_k_pairs_provenance"] = {
            "schema": "kp.gamma-sampled-k-route.v1",
            "authored_in_manifest": False,
            "candidate_source": "actual_sampled_k_set_intersection",
            "kpoints_hash": hash_array(sampled_kpoints),
            "match_tolerance": GAMMA_SAMPLED_K_MATCH_TOLERANCE,
            "gray_tolerance": GAMMA_SAMPLED_K_GRAY_TOLERANCE,
        }
        return pairs

    ctx = projection._build_projection_run_context(
        run_cfg,
        create_output_dir=False,
        validate_full_space_covariance=False,
        require_nlow_state_list=False,
        packed_k_route_resolver=resolve_packed_k_route,
    )
    if ctx.mode.strip().casefold() != "gamma":
        raise ValueError("automatic Gamma v1 requires project.mode='Gamma'")
    if ctx.config.spin != "all" or ctx.config.spin_sector_sewing is not None:
        raise ValueError("automatic Gamma v1 requires one full spinful-all source space")
    if len(ctx.num_layer_list) != 2 or len(ctx.num_orb_per_layer_list) != 2:
        raise ValueError("automatic Gamma v1 requires exactly two source groups")

    source_basis_hash = _selected_manifest_basis_hash(ctx)
    layout = GammaRowLayout.build(
        qsets=(ctx.q1, ctx.q2),
        num_layer_list=ctx.num_layer_list,
        num_orb_per_layer_list=ctx.num_orb_per_layer_list,
        spin_convention="all",
        source_basis_hash=source_basis_hash,
    )
    source_hamiltonians = np.asarray(ctx.hamk3d, dtype=np.complex128)
    nk = int(source_hamiltonians.shape[0])
    k_indices = tuple(range(nk))
    kpoints = sampled_kpoints
    if kpoints is None:
        kpoints = _load_project_source_kpoints(
            cfg,
            material,
            cfg_dir=run_cfg.cfg_dir,
            hamk_file=str(hamk_file),
            nk=nk,
            project_indices=k_indices,
        )

    operations: list[GammaRawOperationSpec] = []
    presentation_records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for request in ctx.operation_requests:
        output_name = str(request["output"])
        payload = ctx.operation_payloads[output_name]
        entry = payload["entry"]
        antiunitary = bool(payload["antiunitary"])
        canonical_name = projection._canonical_internal_operation_name(
            output_name
        )
        if canonical_name in seen_names:
            raise ValueError(
                f"automatic Gamma operation name {canonical_name!r} is duplicated"
            )
        seen_names.add(canonical_name)
        source_action = projection._operation_action_metadata(
            entry,
            canonical_name,
            antiunitary,
            strict=True,
        )
        model_action = projection._model_action_metadata(
            source_action,
            valley=run_cfg.valley,
            operation=canonical_name,
            rotation_deg=ctx.q_rotation_deg,
        )
        sector_map = _manifest_sector_map(
            source_action["sector_map"], projection
        )
        full_action = payload["action"].matrix
        q_permutations = infer_gamma_raw_action_q_permutations(
            full_action=full_action,
            layout=layout,
            sector_map=sector_map,
            thresholds=selection_config.routing_thresholds,
        )
        geometry_q_permutations = _geometry_q_route(
            projection=projection,
            model_action=model_action,
            q_model1=ctx.q_model1,
            q_model2=ctx.q_model2,
            num_layer_list=ctx.num_layer_list,
            sector_map=sector_map,
            q_count=ctx.q_count,
            tolerance=tolerance,
        )
        if q_permutations != geometry_q_permutations:
            raise ValueError(
                f"operation {canonical_name!r} raw-H Q route disagrees with manifest q_map"
            )
        pairs = tuple(
            (int(target), int(source))
            for target, source in entry["_pairs"]
        )
        operations.append(
            GammaRawOperationSpec(
                name=canonical_name,
                full_action=full_action,
                antiunitary=antiunitary,
                q_permutations=q_permutations,
                sector_map=sector_map,
                pairs=pairs,
            )
        )
        group_relations = entry.get("group_relations")
        if group_relations is None:
            group_relations = [
                projection._operation_power_relation(
                    canonical_name,
                    spin_convention="spinful",
                )
            ]
        presentation_records.append(
            {
                "name": canonical_name,
                "operation": canonical_name,
                "antiunitary": antiunitary,
                "declared_model_action": model_action,
                "group_relations": group_relations,
            }
        )
    presentation = compile_continuum_magnetic_presentation(
        presentation_records
    )
    inputs = GammaAutomaticProducerInputs(
        source_hamiltonians=source_hamiltonians,
        k_indices=k_indices,
        kpoints=kpoints,
        qsets=(ctx.q1, ctx.q2),
        num_layer_list=tuple(ctx.num_layer_list),  # type: ignore[arg-type]
        num_orb_per_layer_list=tuple(
            tuple(group) for group in ctx.num_orb_per_layer_list
        ),  # type: ignore[arg-type]
        tapw_source_basis_hash=source_basis_hash,
        operations=tuple(operations),
        presentation=presentation,
    )
    raw_output = project_cfg.get("out_dir")
    if raw_output in (None, ""):
        raise ValueError("automatic Gamma v1 requires project.out_dir")
    output_directory = Path(str(raw_output)).expanduser()
    if not output_directory.is_absolute():
        output_directory = (path.parent / output_directory).resolve()
    return GammaAutomaticRuntimePreparation(
        selection_preparation=prepare_gamma_automatic_selection(
            inputs,
            selection_config,
        ),
        output_directory=output_directory,
    )


def finalize_gamma_automatic_runtime(
    preparation: GammaAutomaticRuntimePreparation,
    *,
    session: PendingCaseSelection,
) -> ResolvedProjectionSelection:
    """Evaluate candidates, commit canonical files, then publish the final marker."""

    if not isinstance(preparation, GammaAutomaticRuntimePreparation):
        raise TypeError("preparation must be GammaAutomaticRuntimePreparation")
    result = evaluate_gamma_automatic_selection(preparation.selection_preparation)
    if result.selection_input != preparation.selection_preparation.selection_input:
        raise ValueError("automatic Gamma evaluation changed the prepared identity")
    payloads = _gamma_project_payloads(result.handoff, result.handoff.kpoints)
    _install_canonical_payloads(preparation.output_directory, payloads)
    return resolve_case_selection(
        CaseSelectionInputs.auto(
            selection_input=result.selection_input,
            candidates=result.candidates,
            thresholds=preparation.selection_preparation.config.selection_thresholds,
            projection_handoff=result.handoff,
            payloads=payloads,
        ),
        session=session,
    )


__all__ = [
    "GammaAutomaticRuntimePreparation",
    "finalize_gamma_automatic_runtime",
    "prepare_gamma_automatic_runtime",
]
