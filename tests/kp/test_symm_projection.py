from __future__ import annotations

import json
import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kp.cli as cli
import kp.symmetry.projection as projection_mod
from kp.basis.selection import GaugeAnchorReport, GaugeCandidateSymmetryMetrics
from kp.identity import hash_array
from kp.model.symmetry import load_symmetry_source
from kp.orbitals import expand_orbital_order_by_sector
from kp.symmetry.projection import (
    _basis_action_for_candidate,
    _build_action_representation,
    _c2_action_audit_for_gamma,
    _model_action_metadata,
    _model_frame_action_metadata,
    _operation_entry,
    _pairs_from_entry,
    _projectors_for_k,
    _resolve_projected_model_action,
    _resolve_symmetry_project_identity,
    _select_operation_matrix_kind,
    _sector_orbital_counts,
    _infer_symmetry_operations_from_manifest,
    _load_project_artifact_identity,
    _validate_symmetry_project_identity,
    _source_manifest_operation_name,
    _validate_operation_label,
    _derive_projection_basis_frame,
    _basis_frame_raw_source_matrices,
    _select_fixed_target_source_matrices,
)
from kp.symmetry.joint_exactification import (
    BlockRouteAction,
    MagneticGenerator,
    MagneticPresentation,
    MagneticRelation,
    JointExactificationConfig,
    joint_exactify_block_actions,
    materialize_block_route_action,
)
from kp.symmetry.q_canonicalization import CanonicalQResult


def test_spin_route_inference_keeps_same_spin_action_internal() -> None:
    matrix = np.eye(4, dtype=np.complex128)

    route, diagnostics = projection_mod._infer_spin_route_from_action_matrix(
        matrix,
        source_spin="up",
    )

    assert route is None
    assert diagnostics["selected_target_spin"] == "up"
    assert diagnostics["normalized_support"]["up<-up"] == pytest.approx(1.0)
    assert diagnostics["normalized_support"]["down<-up"] == pytest.approx(0.0)


def test_spin_route_endpoints_keep_full_spin_space_internal() -> None:
    assert projection_mod._spin_route_endpoints("all", None) == ("all", "all")


def test_spin_route_inference_detects_up_to_down_sewing() -> None:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[:2, 2:] = np.eye(2)
    matrix[2:, :2] = np.eye(2)

    route, diagnostics = projection_mod._infer_spin_route_from_action_matrix(
        matrix,
        source_spin="up",
    )

    assert route == "up_to_down"
    assert diagnostics["selected_target_spin"] == "down"
    assert diagnostics["normalized_support"]["down<-up"] == pytest.approx(1.0)


def test_spin_route_inference_detects_down_to_up_sewing() -> None:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[:2, 2:] = np.eye(2)
    matrix[2:, :2] = np.eye(2)

    route, diagnostics = projection_mod._infer_spin_route_from_action_matrix(
        matrix,
        source_spin="down",
    )

    assert route == "down_to_up"
    assert diagnostics["selected_target_spin"] == "up"
    assert diagnostics["normalized_support"]["up<-down"] == pytest.approx(1.0)


def test_spin_route_inference_rejects_ambiguous_target_support() -> None:
    matrix = np.eye(4, dtype=np.complex128)
    matrix[2:, :2] = np.eye(2)

    with pytest.raises(ValueError, match="ambiguous spin target"):
        projection_mod._infer_spin_route_from_action_matrix(
            matrix,
            source_spin="up",
        )


def test_spin_route_combination_rejects_mixed_operation_routes() -> None:
    assert projection_mod._combine_spin_routes(
        {"TR": "up_to_down", "C2": "up_to_down"}
    ) == "up_to_down"

    with pytest.raises(ValueError, match="incompatible spin routes"):
        projection_mod._combine_spin_routes(
            {"C3z": None, "C2": "up_to_down"}
        )


def test_projection_run_infers_cross_spin_sewing_from_raw_actions(tmp_path: Path) -> None:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[:2, 2:] = np.eye(2)
    matrix[2:, :2] = np.eye(2)
    np.save(tmp_path / "TR.npy", matrix)
    np.save(tmp_path / "C2.npy", matrix)

    route, diagnostics = projection_mod._resolve_spin_sector_sewing(
        spin="up",
        rep_root=tmp_path,
        operation_entries={
            "TR": {"raw_h_operator_file": "TR.npy"},
            "C2": {"raw_h_operator_file": "C2.npy"},
        },
        explicit_route=None,
    )

    assert route == "up_to_down"
    assert diagnostics["source"] == "raw_h_action_support"
    assert diagnostics["operations"]["TR"]["selected_target_spin"] == "down"
    assert diagnostics["operations"]["C2"]["route"] == "up_to_down"


def test_projection_run_rejects_explicit_spin_route_that_conflicts_with_raw_action(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "C3z.npy", np.eye(4, dtype=np.complex128))

    with pytest.raises(ValueError, match="conflicts with raw-H action support"):
        projection_mod._resolve_spin_sector_sewing(
            spin="up",
            rep_root=tmp_path,
            operation_entries={"C3z": {"raw_h_operator_file": "C3z.npy"}},
            explicit_route="up_to_down",
        )


def test_spin_sliced_representation_supports_down_to_up_route(tmp_path: Path) -> None:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[:2, 2:] = np.eye(2)
    matrix[2:, :2] = 2.0 * np.eye(2)
    np.save(tmp_path / "TR.npy", matrix)

    representation = projection_mod._load_spin_sliced_representation(
        path=tmp_path / "TR.npy",
        spin="down",
        full_dim=2,
        spin_sector_sewing="down_to_up",
    )

    np.testing.assert_allclose(representation.matrix, np.eye(2))
    assert representation.from_full_spinful is True
    assert representation.spin_leakage is None


def test_source_target_hamiltonians_follow_down_to_up_route() -> None:
    hamk = np.asarray(
        [np.diag([1.0, 2.0, 10.0, 20.0])],
        dtype=np.complex128,
    )

    source, target = projection_mod._source_target_hamk_by_k(
        hamk3d=hamk,
        required_k=[0],
        spin="down",
        spin_sector_sewing="down_to_up",
    )

    np.testing.assert_allclose(source[0], np.diag([10.0, 20.0]))
    np.testing.assert_allclose(target[0], np.diag([1.0, 2.0]))


def test_projection_context_infers_spin_route_before_slicing(tmp_path: Path) -> None:
    raw_action = np.zeros((4, 4), dtype=np.complex128)
    raw_action[:2, 2:] = np.eye(2)
    raw_action[2:, :2] = np.eye(2)
    np.save(tmp_path / "TR.npy", raw_action)
    hamk = np.asarray(
        [np.diag([1.0, 2.0, 10.0, 20.0])],
        dtype=np.complex128,
    )
    q = np.zeros((1, 2), dtype=float)
    run_cfg = projection_mod._ProjectionRunConfig(
        cfg_path=str(tmp_path / "config.yaml"),
        cfg_dir=str(tmp_path),
        material={},
        plot_cfg={"hamk_index": 0},
        project_cfg={},
        symm_cfg={"output_dir": str(tmp_path / "symmetry")},
        save_projection_diagnostics=False,
        valley="M1",
        spin="up",
        spin_sector_sewing=None,
        q_rotation_raw=0.0,
        tolerance=1.0e-2,
        canonical_layout=True,
    )
    entry = {
        "raw_h_operator_file": "TR.npy",
        "antiunitary": True,
        "_pairs": [(0, 0)],
    }
    observed: dict[str, object] = {}

    def fake_payloads(**kwargs):
        observed["route"] = kwargs["run_cfg"].spin_sector_sewing
        return {}

    with (
        patch.object(
            projection_mod,
            "_load_manifest_and_operation_requests",
            return_value=(tmp_path, {}, [{"source": "TR", "output": "TR"}]),
        ),
        patch.object(
            projection_mod,
            "_load_projection_arrays_and_layout",
            return_value=(hamk, q, q.copy(), "m1", "first_order", None, [[0], [0]], [1, 1], 1, [[1], [1]]),
        ),
        patch.object(
            projection_mod,
            "_resolve_operation_entries",
            return_value=({"TR": entry}, {(0, 0)}),
        ),
        patch.object(projection_mod, "_build_operation_payloads", side_effect=fake_payloads),
    ):
        ctx = projection_mod._build_projection_run_context(
            run_cfg,
            create_output_dir=False,
            validate_full_space_covariance=False,
        )

    assert ctx.config.spin_sector_sewing == "up_to_down"
    assert observed["route"] == "up_to_down"
    assert ctx.spin_route_inference["source"] == "raw_h_action_support"
    np.testing.assert_allclose(ctx.hamk_source_by_k[0], np.diag([1.0, 2.0]))
    np.testing.assert_allclose(ctx.hamk_target_by_k[0], np.diag([10.0, 20.0]))


def test_cross_spin_sewing_does_not_reinterpret_raw_blocks_as_internal_actions() -> None:
    raw = {"TR": np.eye(2, dtype=np.complex128)}
    internal = {"TR": -np.eye(2, dtype=np.complex128)}

    assert _basis_frame_raw_source_matrices(
        raw,
        spin_sector_sewing="up_to_down",
    ) is None
    assert _basis_frame_raw_source_matrices(
        raw,
        spin_sector_sewing=None,
    ) is raw
    selected, role = _select_fixed_target_source_matrices(
        raw,
        exactified_internal_matrices=internal,
        spin_sector_sewing="up_to_down",
    )
    assert selected is internal
    assert role == "joint_exactified_cross_sector_internal_action"
    selected, role = _select_fixed_target_source_matrices(
        raw,
        exactified_internal_matrices=None,
        spin_sector_sewing=None,
    )
    assert selected is raw
    assert role == "raw_projected_internal_action"


def test_projection_basis_frame_uses_certified_closest_u1_fiber_gauge() -> None:
    permutations = {
        "C3z": (1, 2, 0, 4, 5, 3),
        "C2T": (3, 5, 4, 0, 2, 1),
    }
    root = complex(0.5, np.sqrt(3.0) / 2.0)
    phases = np.asarray([0.8, -1.1, 0.2, 1.4, -0.5, 0.3]) * 1.0e-7
    gauge = np.exp(1.0j * phases)
    actions = {}
    for name, antiunitary, base in (
        ("C3z", False, root),
        ("C2T", True, complex(1.0, 0.0)),
    ):
        blocks = []
        for source, target in enumerate(permutations[name]):
            source_gauge = np.conjugate(gauge[source]) if antiunitary else gauge[source]
            blocks.append(
                np.asarray([[np.conjugate(gauge[target]) * base * source_gauge]])
            )
        actions[name] = BlockRouteAction(
            name,
            antiunitary,
            permutations[name],
            (1,) * 6,
            tuple(blocks),
        )
    presentation = MagneticPresentation(
        generators=(
            MagneticGenerator("C3z", False),
            MagneticGenerator("C2T", True),
        ),
        relations=(
            MagneticRelation("C3z^3", ("C3z",) * 3, (), -1.0),
            MagneticRelation("C2T^2", ("C2T",) * 2, (), 1.0),
            MagneticRelation(
                "C2T_C3z_dihedral",
                ("C2T", "C3z", "C2T"),
                ("C3z", "C3z"),
                -1.0,
            ),
        ),
        central_phases=(1.0, -1.0),
        source="test",
    )
    joint = joint_exactify_block_actions(
        actions,
        presentation,
        config=JointExactificationConfig(),
    )
    matrices = {
        name: materialize_block_route_action(action)
        for name, action in joint.actions.items()
    }
    reports = {
        "__joint_exactification__": {
            "metadata": dict(joint.artifact_metadata),
            "arrays": dict(joint.artifact_arrays),
            "report": dict(joint.report),
        }
    }

    frame, canonical = _derive_projection_basis_frame(
        matrices,
        reports,
        operations=[
            {"name": "C3z", "antiunitary": False},
            {"name": "C2T", "antiunitary": True},
        ],
        sectors=[{"name": "only", "n_orb": 1, "n_q": 6}],
        tolerance=1.0e-10,
    )

    assert frame.status == "applied"
    assert canonical is not None
    assert frame.components["closest_cyclotomic_u1_gauge"]["root_order"] == 6
    assert not np.array_equal(frame.full_unitary, np.eye(6))
    nonzero = canonical["C3z"][canonical["C3z"] != 0.0]
    np.testing.assert_array_equal(nonzero, np.full(6, root))


def test_project_owned_target_is_used_for_antiunitary_generator_without_name_bypass(
    tmp_path: Path,
) -> None:
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    actions = {
        "C2T": BlockRouteAction(
            "C2T",
            True,
            (0, 1),
            (2, 2),
            (swap, swap),
        )
    }
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("C2T", True),),
        relations=(
            MagneticRelation("C2T^2", ("C2T", "C2T"), (), 1.0),
        ),
        central_phases=(1.0, -1.0),
        source="antiunitary-fixed-target-test",
    )
    joint = joint_exactify_block_actions(
        actions,
        presentation,
        config=JointExactificationConfig(),
    )
    raw = {
        name: materialize_block_route_action(action)
        for name, action in joint.actions.items()
    }
    reports = {
        "__joint_exactification__": {
            "metadata": dict(joint.artifact_metadata),
            "arrays": dict(joint.artifact_arrays),
            "report": dict(joint.report),
        }
    }
    frame, _canonical = _derive_projection_basis_frame(
        raw,
        reports,
        operations=[{"name": "C2T", "antiunitary": True}],
        sectors=[
            {"name": "L1", "n_orb": 2, "n_q": 1},
            {"name": "L2", "n_orb": 2, "n_q": 1},
        ],
        tolerance=1.0e-10,
        raw_source_matrices=raw,
    )
    assert (
        frame.components["joint_structure_transaction"]["decision"]
        == "fixed_target_raw_alignment_accepted"
    )
    operation = {
        "name": "C2T",
        "operation": "C2T",
        "antiunitary": True,
        "matrix_file": "C2T_low_raw.npy",
        "source_matrix_role": "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "continuum_internal_rep",
        "model_action": {
            "antiunitary": True,
            "k_map": {"type": "identity"},
            "q_map": {"type": "identity"},
            "sector_map": "identity",
        },
        "model_basis_action": {
            "complete": True,
            "items": [
                {
                    "source_sector": sector,
                    "source_q_index": 0,
                    "target_sector": sector,
                    "target_q_index": 0,
                    "q_residual": 0.0,
                }
                for sector in ("L1", "L2")
            ],
        },
        "pairs": [],
        "group_relations": [
            projection_mod._operation_power_relation(
                "C2T",
                spin_convention="spinful",
            )
        ],
    }
    summary = {
        "project_basis": {"symmetry_adapted_frame": frame.artifact()},
        "operations": [operation],
        "valley": "K1",
        "spin": "spinful",
        "tolerance": 1.0e-8,
        "full_dim": 4,
        "low_dim": 4,
        "artifact_identity": {},
    }
    ctx = SimpleNamespace(
        output_dir=tmp_path,
        config=SimpleNamespace(
            valley="K1",
            symm_cfg={"exactification": {}},
            tolerance=1.0e-8,
            canonical_layout=False,
        ),
        q_model1=np.zeros((1, 2), dtype=float),
        q_model2=np.zeros((1, 2), dtype=float),
    )

    result = projection_mod._exactify_and_write_projection_summary(
        ctx,
        summary=summary,
        raw_low_matrices=raw,
        n_orb=(2, 2),
    )

    exactification = result["kp_symm_exactification"]
    assert exactification["fixed_target_exactification"]["status"] == "certified"
    assert exactification["production_joint_solver"] == {
        "kind": "persisted_project_owned_fixed_target",
        "free_solver_status": "not_run",
        "production_used_free_solver": False,
        "additional_basis_gauge_applied": False,
    }
    assert "__fixed_target_bypass__" not in result


def test_projection_basis_frame_composes_standard_c2_cycle_gauge() -> None:
    root = complex(0.5, np.sqrt(3.0) / 2.0)
    actions = {
        "TR": BlockRouteAction(
            "TR",
            True,
            (1, 0),
            (1, 1),
            (np.asarray([[1.0]]), np.asarray([[-1.0]])),
        ),
        "C3z": BlockRouteAction(
            "C3z",
            False,
            (0, 1),
            (1, 1),
            (np.asarray([[root]]), np.asarray([[root.conjugate()]])),
        ),
        "C2": BlockRouteAction(
            "C2",
            False,
            (1, 0),
            (1, 1),
            (
                np.asarray([[complex(-0.5, np.sqrt(3.0) / 2.0)]]),
                np.asarray([[root]]),
            ),
        ),
    }
    presentation = MagneticPresentation(
        generators=(
            MagneticGenerator("TR", True),
            MagneticGenerator("C3z", False),
            MagneticGenerator("C2", False),
        ),
        relations=(
            MagneticRelation("TR^2", ("TR", "TR"), (), -1.0),
            MagneticRelation("C3z^3", ("C3z",) * 3, (), -1.0),
            MagneticRelation("C2^2", ("C2", "C2"), (), -1.0),
            MagneticRelation(
                "TR_C3z_commute",
                ("TR", "C3z"),
                ("C3z", "TR"),
                1.0,
            ),
            MagneticRelation(
                "TR_C2_commute",
                ("TR", "C2"),
                ("C2", "TR"),
                1.0,
            ),
            MagneticRelation(
                "C2_C3z_dihedral",
                ("C2", "C3z", "C2"),
                ("C3z", "C3z"),
                1.0,
            ),
        ),
        central_phases=(1.0, -1.0),
        source="test",
    )
    joint = joint_exactify_block_actions(
        actions,
        presentation,
        config=JointExactificationConfig(),
    )
    matrices = {
        name: materialize_block_route_action(action)
        for name, action in joint.actions.items()
    }
    reports = {
        "__joint_exactification__": {
            "metadata": dict(joint.artifact_metadata),
            "arrays": dict(joint.artifact_arrays),
            "report": dict(joint.report),
        }
    }

    frame, canonical = _derive_projection_basis_frame(
        matrices,
        reports,
        operations=[
            {"name": "TR", "antiunitary": True},
            {"name": "C3z", "antiunitary": False},
            {"name": "C2", "antiunitary": False},
        ],
        sectors=[
            {"name": "L1", "n_orb": 1, "n_q": 1},
            {"name": "L2", "n_orb": 1, "n_q": 1},
        ],
        tolerance=1.0e-10,
    )

    assert canonical is not None
    standard = frame.components["standard_generator_fiber_gauge"]
    assert standard["cycle_targets"]["C2"] == ["+i"]
    np.testing.assert_allclose(
        frame.full_unitary,
        np.diag(np.exp(1.0j * np.asarray([-np.pi / 12.0, np.pi / 12.0]))),
        atol=2.0e-15,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        canonical["C3z"],
        materialize_block_route_action(actions["C3z"]),
    )
    np.testing.assert_array_equal(
        canonical["TR"],
        materialize_block_route_action(actions["TR"]),
    )
    nonzero_c2 = canonical["C2"][canonical["C2"] != 0.0]
    np.testing.assert_array_equal(
        nonzero_c2,
        np.full(2, complex(0.0, 1.0)),
    )


def test_projection_basis_frame_round_trips_ud_negative_i_swap() -> None:
    identity = np.eye(2, dtype=np.complex128)
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    angle = 0.13
    rotation = np.asarray(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    gauges = (identity, rotation)
    canonical_block = -1.0j * swap
    blocks = tuple(
        gauges[1 - source].conjugate().T @ canonical_block @ gauges[source]
        for source in range(2)
    )
    actions = {
        "C2": BlockRouteAction(
            "C2",
            False,
            (1, 0),
            (2, 2),
            blocks,
        )
    }
    presentation = MagneticPresentation(
        generators=(MagneticGenerator("C2", False),),
        relations=(MagneticRelation("C2^2", ("C2", "C2"), (), -1.0),),
        central_phases=(1.0, -1.0),
        source="test",
    )
    joint = joint_exactify_block_actions(
        actions,
        presentation,
        config=JointExactificationConfig(),
    )
    matrices = {
        name: materialize_block_route_action(action)
        for name, action in joint.actions.items()
    }
    reports = {
        "__joint_exactification__": {
            "metadata": dict(joint.artifact_metadata),
            "arrays": dict(joint.artifact_arrays),
            "report": dict(joint.report),
        }
    }

    frame, exact = _derive_projection_basis_frame(
        matrices,
        reports,
        operations=[{"name": "C2", "antiunitary": False}],
        sectors=[{"name": "only", "n_orb": 2, "n_q": 2}],
        tolerance=1.0e-10,
    )

    assert exact is not None
    component = frame.components["standard_generator_fiber_gauge"]
    assert component["fiber_mode"] == "free_orbit_Ud"
    assert "algebraic_route_encoding" in component
    nonzero = exact["C2"][exact["C2"] != 0.0]
    np.testing.assert_array_equal(nonzero, np.full(4, complex(0.0, -1.0)))
    np.testing.assert_allclose(
        frame.full_unitary.conjugate().T @ matrices["C2"] @ frame.full_unitary,
        exact["C2"],
        atol=5.0e-14,
        rtol=0.0,
    )


def test_projection_basis_frame_preserves_q_uniform_c3_routes() -> None:
    root = complex(0.5, np.sqrt(3.0) / 2.0)
    c3_diagonal = np.diag([root.conjugate(), -1.0, -1.0, root]).astype(
        np.complex128
    )
    c2_diagonal = 1.0j * np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    tr_diagonal = np.asarray(
        [
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    angle = 0.37
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, np.cos(angle), -np.sin(angle)],
            [0.0, 0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.complex128,
    )
    dimensions = (4,) * 6
    actions = {
        "TR": BlockRouteAction(
            "TR",
            True,
            tuple(range(6)),
            dimensions,
            tuple(rotation @ tr_diagonal @ rotation.T for _ in range(6)),
        ),
        "C3z": BlockRouteAction(
            "C3z",
            False,
            (1, 2, 0, 4, 5, 3),
            dimensions,
            tuple(
                rotation @ c3_diagonal @ rotation.conjugate().T
                for _ in range(6)
            ),
        ),
        "C2": BlockRouteAction(
            "C2",
            False,
            (3, 5, 4, 0, 2, 1),
            dimensions,
            tuple(
                rotation @ c2_diagonal @ rotation.conjugate().T
                for _ in range(6)
            ),
        ),
    }
    presentation = MagneticPresentation(
        generators=(
            MagneticGenerator("TR", True),
            MagneticGenerator("C3z", False),
            MagneticGenerator("C2", False),
        ),
        relations=(
            MagneticRelation("TR^2", ("TR",) * 2, (), -1.0),
            MagneticRelation("C3z^3", ("C3z",) * 3, (), -1.0),
            MagneticRelation("C2^2", ("C2",) * 2, (), -1.0),
            MagneticRelation(
                "TR_C3z_commute",
                ("TR", "C3z"),
                ("C3z", "TR"),
                1.0,
            ),
            MagneticRelation(
                "TR_C2_commute",
                ("TR", "C2"),
                ("C2", "TR"),
                1.0,
            ),
            MagneticRelation(
                "C2_C3z_dihedral",
                ("C2", "C3z", "C2"),
                ("C3z", "C3z"),
                1.0,
            ),
        ),
        central_phases=(1.0, -1.0),
        source="test",
    )
    joint = joint_exactify_block_actions(
        actions,
        presentation,
        config=JointExactificationConfig(),
    )
    matrices = {
        name: materialize_block_route_action(action)
        for name, action in joint.actions.items()
    }
    reports = {
        "__joint_exactification__": {
            "metadata": dict(joint.artifact_metadata),
            "arrays": dict(joint.artifact_arrays),
            "report": dict(joint.report),
        }
    }

    frame, exact = _derive_projection_basis_frame(
        matrices,
        reports,
        operations=[
            {"name": "TR", "antiunitary": True},
            {"name": "C3z", "antiunitary": False},
            {"name": "C2", "antiunitary": False},
        ],
        sectors=[{"name": "only", "n_orb": 4, "n_q": 6}],
        tolerance=1.0e-10,
    )

    assert exact is not None
    component = frame.components["standard_generator_fiber_gauge"]
    assert component["fiber_mode"] == "q_uniform_Ud"
    assert component["uniform_generator"]["name"] == "C3z"
    c3_blocks = [
        exact["C3z"][4 * target : 4 * (target + 1), 4 * source : 4 * (source + 1)]
        for source, target in enumerate((1, 2, 0, 4, 5, 3))
    ]
    for block in c3_blocks[1:]:
        np.testing.assert_array_equal(block, c3_blocks[0])
    np.testing.assert_array_equal(c3_blocks[0], np.diag(np.diag(c3_blocks[0])))
    for name, antiunitary in (("TR", True), ("C3z", False), ("C2", False)):
        right = frame.full_unitary.conjugate() if antiunitary else frame.full_unitary
        np.testing.assert_allclose(
            frame.full_unitary.conjugate().T @ matrices[name] @ right,
            exact[name],
            atol=5.0e-14,
            rtol=0.0,
        )


def test_project_artifact_identity_rejects_basis_wavefunction_mismatch(tmp_path) -> None:
    project_dir = tmp_path / "projection"
    project_dir.mkdir()
    identity = {
        "identity_schema": "moirekp.artifact-identity.v1",
        "input_hash": "input-a",
        "config_hash": "config-a",
        "basis_hash": "basis-a",
        "package_version": "0.1.0",
        "schema_version": 2,
        "k_indices_hash": "k-indices-a",
        "heff_hash": "heff-a",
    }
    np.savez(project_dir / "basis.npz", **{key: np.asarray(value) for key, value in identity.items()})
    np.savez(
        project_dir / "wavefunctions.npz",
        **{
            **{key: np.asarray(value) for key, value in identity.items()},
            "basis_hash": np.asarray("basis-b"),
            "wavefunctions": np.eye(2, dtype=np.complex128)[None, :, :],
            "k_indices": np.asarray([0]),
        },
    )
    np.save(project_dir / "heff.npy", np.eye(2, dtype=np.complex128)[None, :, :])

    with pytest.raises(ValueError, match="KP projection artifacts.*basis_hash"):
        _load_project_artifact_identity(project_dir, verify_heff=False)


def test_canonical_symmetry_package_exports_certified_factorized_action(
    tmp_path: Path,
) -> None:
    exactified = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    np.save(tmp_path / "exactified_C2.npy", exactified)
    items = [
        {
            "source_sector": "L1",
            "source_q_index": 0,
            "target_sector": "L2",
            "target_q_index": 0,
            "q_residual": 0.0,
        },
        {
            "source_sector": "L2",
            "source_q_index": 0,
            "target_sector": "L1",
            "target_q_index": 0,
            "q_residual": 0.0,
        },
    ]
    summary = {
        "valley": "Gamma",
        "spin": "spinful",
        "tolerance": 1.0e-8,
        "full_dim": 2,
        "low_dim": 2,
        "artifact_identity": {},
        "kp_symm_exactification": {
            "status": "exactified",
            "matrix_source": "kp_symm_exactified_action",
            "n_orb": [1, 1],
        },
        "operations": [
            {
                "name": "C2",
                "operation": "C2",
                "antiunitary": False,
                "pairs": [],
                "matrix_file": "exactified_C2.npy",
                "matrix_kind": "continuum_internal_rep_exact",
                "matrix_source": "kp_symm_exactified_action",
                "internal_resolved_action": {
                    "antiunitary": False,
                    "k_map": {"type": "identity"},
                    "q_map": {"type": "identity"},
                    "sector_map": "layer_exchange",
                },
                "model_basis_action": {
                    "complete": True,
                    "items": items,
                },
            }
        ],
    }
    q_geometry = CanonicalQResult(
        raw_q={"L1": np.zeros((1, 2)), "L2": np.zeros((1, 2))},
        canonical_q={"L1": np.zeros((1, 2)), "L2": np.zeros((1, 2))},
        artifact={
            "sector_order": ("L1", "L2"),
            "canonical_closure_max": 0.0,
            "status": "certified",
        },
    )

    projection_mod._write_canonical_symmetry_outputs(
        tmp_path,
        summary,
        q_geometry=q_geometry,
    )

    with np.load(tmp_path / "representations.npz", allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata_json__"].item()))
        factorized_metadata = metadata["kp_symm_exactification"][
            "factorized_response_action"
        ]
        factorized_arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key.startswith("__factorized_response_action_")
        }
    from kp.symmetry.factorized_action import (
        load_factorized_actions,
        load_factorized_actions_from_npz,
    )

    restored = load_factorized_actions(factorized_metadata, factorized_arrays)
    restored_from_file = load_factorized_actions_from_npz(
        tmp_path / "representations.npz"
    )
    assert set(restored) == {"C2"}
    assert set(restored_from_file) == {"C2"}
    assert restored_from_file["C2"].artifact_hash == restored["C2"].artifact_hash
    assert restored["C2"].sector_permutation == (1, 0)
    assert restored["C2"].q_permutation == (1, 0)


def test_canonical_symmetry_package_exports_certified_joint_artifact(
    tmp_path: Path,
) -> None:
    from kp.symmetry.joint_exactification import (
        BlockRouteAction,
        JointExactificationConfig,
        compile_continuum_magnetic_presentation,
        joint_exactify_block_actions,
        load_joint_exactification_artifact,
        materialize_block_route_action,
    )

    presentation = compile_continuum_magnetic_presentation(
        [
            {
                "name": "TR",
                "operation": "TR",
                "antiunitary": True,
                "k_map": {"type": "negation"},
                "q_map": {"type": "negation"},
                "sector_map": "identity",
                "group_relations": [
                    {"type": "power", "operation": "TR", "power": 2, "phase": 1}
                ],
            },
            {
                "name": "C2",
                "operation": "C2",
                "antiunitary": False,
                "k_map": {"type": "reflection", "axis_deg": 0.0},
                "q_map": {"type": "reflection", "axis_deg": 0.0},
                "sector_map": "layer_exchange",
                "group_relations": [
                    {"type": "power", "operation": "C2", "power": 2, "phase": 1}
                ],
            },
        ]
    )
    actions = {
        "TR": BlockRouteAction(
            "TR",
            True,
            (0, 1),
            (1, 1),
            (np.ones((1, 1), dtype=np.complex128),) * 2,
        ),
        "C2": BlockRouteAction(
            "C2",
            False,
            (1, 0),
            (1, 1),
            (-np.ones((1, 1), dtype=np.complex128),) * 2,
        ),
    }
    result = joint_exactify_block_actions(
        actions,
        presentation,
        config=JointExactificationConfig(),
    )
    for name, action in result.actions.items():
        np.save(tmp_path / f"exactified_{name}.npy", materialize_block_route_action(action))
    summary = {
        "valley": "M",
        "spin": "up",
        "tolerance": 1.0e-8,
        "full_dim": 2,
        "low_dim": 2,
        "artifact_identity": {},
        "kp_symm_exactification": {
            "status": "exactified",
            "matrix_source": "kp_symm_exactified_action",
            "n_orb": [1, 1],
            "joint_block_representation": dict(result.artifact_metadata),
        },
        "operations": [
            {
                "name": name,
                "operation": name,
                "antiunitary": bool(action.antiunitary),
                "model_action": {
                    "antiunitary": bool(action.antiunitary),
                    "k_map": (
                        {"type": "negation"}
                        if name == "TR"
                        else {"type": "reflection", "axis_deg": 0.0}
                    ),
                    "q_map": (
                        {"type": "negation"}
                        if name == "TR"
                        else {"type": "reflection", "axis_deg": 0.0}
                    ),
                    "sector_map": "identity" if name == "TR" else "layer_exchange",
                },
                "pairs": [],
                "matrix_file": f"exactified_{name}.npy",
                "matrix_kind": "continuum_internal_rep_exact",
                "matrix_source": "kp_symm_exactified_action",
            }
            for name, action in result.actions.items()
        ],
    }
    q_geometry = CanonicalQResult(
        raw_q={"L1": np.zeros((1, 2)), "L2": np.zeros((1, 2))},
        canonical_q={"L1": np.zeros((1, 2)), "L2": np.zeros((1, 2))},
        artifact={
            "sector_order": ("L1", "L2"),
            "canonical_closure_max": 0.0,
            "status": "certified",
        },
    )

    projection_mod._write_canonical_symmetry_outputs(
        tmp_path,
        summary,
        q_geometry=q_geometry,
        joint_artifact_arrays=result.artifact_arrays,
    )

    with np.load(tmp_path / "representations.npz", allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata_json__"].item()))
        joint_metadata = metadata["kp_symm_exactification"][
            "joint_block_representation"
        ]
        joint_arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key.startswith("__joint_block_representation_")
        }
    restored = load_joint_exactification_artifact(joint_metadata, joint_arrays)
    assert restored.artifact_metadata["artifact_hash"] == result.artifact_metadata[
        "artifact_hash"
    ]
    assert set(restored.actions) == {"TR", "C2"}
    factorized = metadata["kp_symm_exactification"]["factorized_response_action"]
    assert factorized["status"] == "certified"
    assert {
        record["name"]: tuple(record["q_permutation"])
        for record in factorized["actions"]
    } == {"TR": (0, 1), "C2": (1, 0)}


def test_project_artifact_identity_includes_and_verifies_kpoints_hash(tmp_path) -> None:
    project_dir = tmp_path / "projection"
    project_dir.mkdir()
    heff = np.eye(2, dtype=np.complex128)[None, :, :]
    kpoints = np.array([[0.1, 0.2]], dtype=float)
    k_indices = np.asarray([0], dtype=np.int64)
    np.save(project_dir / "heff.npy", heff)
    np.save(project_dir / "kpoints.npy", kpoints)
    identity = {
        "identity_schema": "moirekp.artifact-identity.v1",
        "input_hash": "input-a",
        "config_hash": "config-a",
        "basis_hash": "basis-a",
        "package_version": "0.1.0",
        "schema_version": 2,
        "k_indices_hash": hash_array(k_indices),
        "heff_hash": hash_array(heff),
        "kpoints_hash": hash_array(kpoints),
    }
    scalar_identity = {key: np.asarray(value) for key, value in identity.items()}
    np.savez(project_dir / "basis.npz", **scalar_identity)
    np.savez(
        project_dir / "wavefunctions.npz",
        wavefunctions=np.eye(2, dtype=np.complex128)[None, :, :],
        k_indices=k_indices,
        **scalar_identity,
    )

    resolved = _load_project_artifact_identity(project_dir)

    assert resolved["kpoints_hash"] == hash_array(kpoints)

    np.save(project_dir / "kpoints.npy", kpoints + 0.25)
    with pytest.raises(ValueError, match="kpoints_hash mismatch"):
        _load_project_artifact_identity(project_dir)


def test_symmetry_project_identity_rejects_recomputed_basis_mismatch(tmp_path) -> None:
    hamk_path = tmp_path / "hamk.npy"
    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0]], dtype=float)
    hamk = np.eye(4, dtype=np.complex128)[None, :, :]
    np.save(hamk_path, hamk)
    expected = projection_mod.build_projection_basis_identity(
        qset1=q1,
        qset2=q2,
        spin="up",
        mode="k1",
        energy_scale=1.0,
        nlow_state_list=[[0], [0]],
        resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
        gauge_mode="manual",
        num_layer_list=[1, 1],
        num_orb_per_layer_list=[1, 1],
        orbital_block_dim=1,
        model_dim=2,
        k_indices=[0],
    )
    actual = {**expected, "basis_hash": "stale-basis", "heff_hash": "heff-a"}

    with pytest.raises(ValueError, match="kp symm.*basis_hash"):
        _validate_symmetry_project_identity(expected, actual)


def test_symmetry_resolves_identity_from_projection_artifacts(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "projection"
    project_dir.mkdir()
    hamk_path = tmp_path / "hamk.npy"
    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0]], dtype=float)
    hamk = np.eye(4, dtype=np.complex128)[None, :, :]
    heff = np.eye(2, dtype=np.complex128)[None, :, :]
    np.save(hamk_path, hamk)
    np.save(project_dir / "heff.npy", heff)
    identity = projection_mod.build_projection_basis_identity(
        qset1=q1,
        qset2=q2,
        spin="up",
        mode="k1",
        energy_scale=1.0,
        nlow_state_list=[[0], [0]],
        resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
        gauge_mode="manual",
        num_layer_list=[1, 1],
        num_orb_per_layer_list=[1, 1],
        orbital_block_dim=1,
        model_dim=2,
        k_indices=[0],
    )
    artifact_identity = {
        **identity,
        "heff_hash": hash_array(heff),
        "source_hamiltonian_hash": "legacy-large-file-hash",
    }
    scalar_identity = {key: np.asarray(value) for key, value in artifact_identity.items()}
    np.savez(project_dir / "basis.npz", **scalar_identity)
    np.savez(
        project_dir / "wavefunctions.npz",
        wavefunctions=np.eye(2, dtype=np.complex128)[None, :, :],
        k_indices=np.asarray([0]),
        **scalar_identity,
    )
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            cfg_dir=str(tmp_path),
            project_cfg={"out_dir": str(project_dir)},
            material={"hamk_file": str(hamk_path), "energy_unit": "eV"},
            spin="up",
        ),
        q1=q1,
        q2=q2,
        mode="k1",
        nlow_state_list=[[0], [0]],
        num_layer_list=[1, 1],
        num_orb_per_layer_list=[1, 1],
        orb0=1,
        hamk_source_by_k={0: hamk[0]},
    )
    gauge_report = SimpleNamespace(gauge_mode="manual")
    monkeypatch.setattr(
        projection_mod,
        "hash_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("kp symm rehashed the full source Hamiltonian")
        ),
        raising=False,
    )

    resolved = _resolve_symmetry_project_identity(
        ctx,
        gauge_report=gauge_report,
        resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
        low_dim=2,
    )

    assert resolved == {
        field: artifact_identity[field]
        for field in projection_mod.PROJECTION_ARTIFACT_IDENTITY_FIELDS
    }
    assert "source_hamiltonian_hash" not in resolved


def test_symmetry_identity_rejects_project_k_indices_config_drift(tmp_path) -> None:
    project_dir = tmp_path / "projection"
    project_dir.mkdir()
    hamk_path = tmp_path / "hamk.npy"
    q1 = np.array([[0.0, 0.0]], dtype=float)
    q2 = np.array([[0.0, 0.0]], dtype=float)
    hamk = np.stack([np.eye(4), 2.0 * np.eye(4)]).astype(np.complex128)
    heff = np.eye(2, dtype=np.complex128)[None, :, :]
    np.save(hamk_path, hamk)
    np.save(project_dir / "heff.npy", heff)
    identity = projection_mod.build_projection_basis_identity(
        qset1=q1,
        qset2=q2,
        spin="up",
        mode="k1",
        energy_scale=1.0,
        nlow_state_list=[[0], [0]],
        resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
        gauge_mode="manual",
        num_layer_list=[1, 1],
        num_orb_per_layer_list=[1, 1],
        orbital_block_dim=1,
        model_dim=2,
        k_indices=[0],
    )
    artifact_identity = {**identity, "heff_hash": hash_array(heff)}
    scalar_identity = {key: np.asarray(value) for key, value in artifact_identity.items()}
    np.savez(project_dir / "basis.npz", **scalar_identity)
    np.savez(
        project_dir / "wavefunctions.npz",
        wavefunctions=np.eye(2, dtype=np.complex128)[None, :, :],
        k_indices=np.asarray([0]),
        **scalar_identity,
    )
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            cfg_dir=str(tmp_path),
            project_cfg={"out_dir": str(project_dir), "k_indices": [1]},
            material={"hamk_file": str(hamk_path), "energy_unit": "eV"},
            spin="up",
        ),
        q1=q1,
        q2=q2,
        mode="k1",
        nlow_state_list=[[0], [0]],
        num_layer_list=[1, 1],
        num_orb_per_layer_list=[1, 1],
        orb0=1,
        hamk_source_by_k={0: hamk[0]},
    )
    gauge_report = SimpleNamespace(gauge_mode="manual")

    with pytest.raises(ValueError, match="kp symm.*k_indices_hash"):
        _resolve_symmetry_project_identity(
            ctx,
            gauge_report=gauge_report,
            resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
            low_dim=2,
        )


def _load_canonical_symmetry_payload(out_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(out_dir / "representations.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files if name != "__metadata_json__"}
        metadata = json.loads(str(payload["__metadata_json__"].item()))
    return arrays, metadata


def _write_projection_identity_artifacts(
    project_dir: Path,
    *,
    hamk_file: Path,
    hamk: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    mode: str,
    nlow_state_list: list,
    resolved_norb_fix_list: list,
    gauge_mode: str = "manual_norb_fix_list",
    spin: str = "up",
    num_layer_list: list[int] | None = None,
    num_orb_per_layer_list: list | None = None,
    orbital_block_dim: int = 2,
    model_dim: int = 2,
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    heff = np.zeros((1, model_dim, model_dim), dtype=np.complex128)
    np.save(project_dir / "heff.npy", heff)
    identity = projection_mod.build_projection_basis_identity(
        qset1=q1,
        qset2=q2,
        spin=spin,
        mode=mode.lower(),
        energy_scale=1.0,
        nlow_state_list=nlow_state_list,
        resolved_norb_fix_list=resolved_norb_fix_list,
        gauge_mode=gauge_mode,
        num_layer_list=num_layer_list or [1, 1],
        num_orb_per_layer_list=num_orb_per_layer_list or [[2], [2]],
        orbital_block_dim=orbital_block_dim,
        model_dim=model_dim,
        k_indices=[0],
    )
    artifact_identity = {**identity, "heff_hash": hash_array(heff)}
    scalar_identity = {key: np.asarray(value) for key, value in artifact_identity.items()}
    np.savez(project_dir / "basis.npz", **scalar_identity)
    np.savez(
        project_dir / "wavefunctions.npz",
        wavefunctions=np.eye(model_dim, dtype=np.complex128)[None, :, :],
        k_indices=np.asarray([0]),
        **scalar_identity,
    )


def test_orbital_order_accepts_per_layer_patterns() -> None:
    labels = expand_orbital_order_by_sector(
        {
            "L1": "Bi-s1,Te-p1,I-s1",
            "L2": "I-s1,Te-p1,Bi-s1",
        },
        ["L1", "L2"],
    )

    assert labels is not None
    assert labels["L1"] == ["Bi_s_r1", "Te_px_r1", "Te_py_r1", "Te_pz_r1", "I_s_r1"]
    assert labels["L2"] == ["I_s_r1", "Te_px_r1", "Te_py_r1", "Te_pz_r1", "Bi_s_r1"]


def test_projected_basis_action_uses_orbital_map_for_layer_exchange() -> None:
    action = {
        "k_map": {"type": "identity"},
        "q_map": {"type": "identity"},
        "sector_map": "layer_exchange",
        "orbital_map": {"L1": {1: 2, 2: 1}, "L2": {1: 2, 2: 1}},
    }

    basis_action = _basis_action_for_candidate(
        action=action,
        q_model1=np.array([[0.0, 0.0]]),
        q_model2=np.array([[0.0, 0.0]]),
        nlow_state_list=[[0, 1], [0, 1]],
        low_dim=4,
        tol=1.0e-8,
    )

    assert basis_action["complete"] is True
    assert basis_action["perm"].tolist() == [3, 2, 1, 0]


class SymmetryProjectionCliTests(unittest.TestCase):
    def test_kp_symm_help_lists_developer_outputs(self) -> None:
        stream = io.StringIO()
        with self.assertRaises(SystemExit) as exc, redirect_stdout(stream):
            cli.main(["symm", "--help"])

        self.assertEqual(exc.exception.code, 0)
        self.assertIn("--developer-outputs", stream.getvalue())

    def test_symmetry_validated_auto_gauge_chooses_candidate_by_residual(self) -> None:
        candidates = [
            SimpleNamespace(candidate_id="old_auto", resolved_norb_fix_list=[[[0]]]),
            SimpleNamespace(candidate_id="fixed_auto", resolved_norb_fix_list=[[[1]]]),
        ]

        selected, decision = projection_mod._select_validated_auto_gauge_candidate(
            candidates,
            [
                GaugeCandidateSymmetryMetrics(
                    candidate_id="old_auto",
                    exactification_distance_by_op={"T": 1.414},
                    active_term_count=10,
                ),
                GaugeCandidateSymmetryMetrics(
                    candidate_id="fixed_auto",
                    exactification_distance_by_op={"T": 1.0e-12, "C3z": 2.0e-12},
                    active_term_count=12,
                ),
            ],
            max_exactification_distance=1.0e-3,
        )

        self.assertEqual(selected.candidate_id, "fixed_auto")
        self.assertEqual(selected.resolved_norb_fix_list, [[[1]]])
        self.assertEqual(decision.rankings[0]["candidate_id"], "fixed_auto")
        self.assertEqual(decision.rankings[-1]["candidate_id"], "old_auto")
        self.assertEqual(decision.rankings[-1]["status"], "rejected")

    def test_projectors_for_k_honors_configured_downfold_method(self) -> None:
        ham = np.diag([0.0, 1.0, 10.0, 20.0]).astype(np.complex128)
        u_low = np.eye(4, 1, dtype=np.complex128)
        u_high = np.eye(4, 4, dtype=np.complex128)[:, 1:]
        calls = {}

        def fake_get_h_block(*_args, **_kwargs):
            return None, [np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)], None, None

        def fake_assemble_gamma_projectors(*_args, **kwargs):
            calls["include_high"] = kwargs.get("include_high")
            return u_low, u_high

        def fake_downfold_from_projectors(_ham, _u_low, _u_high, options):
            calls["method"] = options.method
            calls["e_ref"] = options.e_ref
            calls["u_high_is_none"] = _u_high is None
            return SimpleNamespace(heff=np.array([[2.0]], dtype=np.complex128))

        with (
            patch.object(projection_mod, "get_H_block", side_effect=fake_get_h_block),
            patch.object(projection_mod, "_assemble_gamma_projectors_from_block_eigenvectors", side_effect=fake_assemble_gamma_projectors),
            patch.object(projection_mod, "downfold_from_projectors", side_effect=fake_downfold_from_projectors),
        ):
            state = projection_mod._projectors_for_k(
                ham,
                np.array([[0.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0]], dtype=float),
                orb0=1,
                spin="up",
                mode="gamma",
                nlow_state_list=[[0], [0]],
                norb_fix_list=[[0], [0]],
                method="fixed_schur",
                e_ref=0.5,
                project_cfg={},
            )

        self.assertEqual(calls["include_high"], True)
        self.assertEqual(calls["method"], "fixed_schur")
        self.assertEqual(calls["e_ref"], 0.5)
        self.assertFalse(calls["u_high_is_none"])
        np.testing.assert_allclose(state.heff, [[2.0]])

    def test_auto_gauge_candidate_states_reuse_reference_downfold(self) -> None:
        reference_u = np.eye(4, 2, dtype=np.complex128)
        transform = np.asarray(
            [[0.0, 1.0], [1.0j, 0.0]],
            dtype=np.complex128,
        )
        candidate_u = reference_u @ transform
        reference_heff = np.asarray(
            [[1.0, 0.2j], [-0.2j, 3.0]],
            dtype=np.complex128,
        )
        calls: list[tuple[str, str]] = []

        def fake_projectors_for_k(*_args, method, norb_fix_list, **_kwargs):
            anchor_id = str(norb_fix_list[0])
            calls.append((anchor_id, method))
            if anchor_id == "reference":
                return projection_mod.ProjectionState(
                    hamk=np.eye(4, dtype=np.complex128),
                    heff=reference_heff,
                    u_low=reference_u,
                )
            if method != "first_order":
                raise AssertionError("candidate repeated the expensive downfold")
            return projection_mod.ProjectionState(
                hamk=np.eye(4, dtype=np.complex128),
                heff=np.zeros((2, 2), dtype=np.complex128),
                u_low=candidate_u,
            )

        ctx = SimpleNamespace(
            config=SimpleNamespace(
                spin_sector_sewing=None,
                spin="up",
                project_cfg={},
            ),
            required_k=[0],
            hamk_source_by_k={0: np.eye(4, dtype=np.complex128)},
            hamk_target_by_k={0: np.eye(4, dtype=np.complex128)},
            q1=np.zeros((1, 2), dtype=float),
            q2=np.zeros((1, 2), dtype=float),
            orb0=2,
            num_layer_list=[1, 1],
            num_orb_per_layer_list=[[2], [2]],
            mode="gamma",
            nlow_state_list=[[0], [1]],
            method="linearized_lowdin",
            e_ref=0.5,
        )
        diagnostics: list[dict] = []
        with patch.object(
            projection_mod,
            "_projectors_for_k",
            side_effect=fake_projectors_for_k,
        ):
            reference = projection_mod._states_for_resolved_anchors(
                ctx,
                ["reference"],
            )
            candidate = projection_mod._states_for_resolved_anchors(
                ctx,
                ["candidate"],
                reference_states=reference,
                reuse_diagnostics=diagnostics,
            )

        candidate_state = candidate[0][0]
        np.testing.assert_allclose(
            candidate_state.heff,
            transform.conj().T @ reference_heff @ transform,
            atol=1.0e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(candidate_state.u_low, candidate_u)
        self.assertEqual(calls, [("reference", "linearized_lowdin"), ("candidate", "first_order")])
        self.assertEqual(diagnostics[0]["status"], "reused")

    def test_auto_gauge_candidate_basis_does_not_run_configured_downfold(self) -> None:
        calls: list[tuple[str, bool, bool]] = []
        eigensystem_cache = {}

        def fake_projectors_for_k(
            *_args,
            method,
            eigensystem_cache,
            compute_heff=True,
            **_kwargs,
        ):
            calls.append(
                (
                    str(method),
                    bool(compute_heff),
                    eigensystem_cache is ctx.block_eigensystem_cache,
                )
            )
            return projection_mod.ProjectionState(
                hamk=np.eye(4, dtype=np.complex128),
                heff=np.zeros((2, 2), dtype=np.complex128),
                u_low=np.eye(4, 2, dtype=np.complex128),
            )

        ctx = SimpleNamespace(
            config=SimpleNamespace(
                spin_sector_sewing=None,
                spin="up",
                project_cfg={},
            ),
            required_k=[0],
            hamk_source_by_k={0: np.eye(4, dtype=np.complex128)},
            hamk_target_by_k={0: np.eye(4, dtype=np.complex128)},
            q1=np.zeros((1, 2), dtype=float),
            q2=np.zeros((1, 2), dtype=float),
            orb0=2,
            num_layer_list=[1, 1],
            num_orb_per_layer_list=[[2], [2]],
            mode="gamma",
            nlow_state_list=[[0], [1]],
            method="linearized_lowdin",
            e_ref=0.5,
            block_eigensystem_cache=eigensystem_cache,
        )
        with patch.object(
            projection_mod,
            "_projectors_for_k",
            side_effect=fake_projectors_for_k,
        ):
            states = projection_mod._basis_states_for_resolved_anchors(
                ctx,
                ["candidate"],
            )

        self.assertEqual(calls, [("first_order", False, True)])
        np.testing.assert_allclose(states[0][0].u_low, np.eye(4, 2))

    def test_candidate_projection_can_skip_heff_covariance_metric(self) -> None:
        state = projection_mod.ProjectionState(
            hamk=np.eye(2, dtype=np.complex128),
            heff=None,
            u_low=np.eye(2, dtype=np.complex128),
        )
        raw, _polar, rows = projection_mod._project_operation(
            operation="E",
            antiunitary=False,
            d_full=np.eye(2, dtype=np.complex128),
            states={0: state},
            pairs=[(0, 0)],
            tolerance=1.0e-12,
            compute_heff_covariance=False,
        )

        self.assertTrue(np.array_equal(raw[0], np.eye(2)))
        self.assertIsNone(rows[0]["raw"]["heff_covariance_residual"])

    def test_auto_gauge_falls_back_when_production_basis_frame_changes(self) -> None:
        def bundle(u_low):
            state = projection_mod.ProjectionState(
                hamk=np.eye(2, dtype=np.complex128),
                heff=np.zeros((1, 1), dtype=np.complex128),
                u_low=np.asarray(u_low, dtype=np.complex128),
            )
            by_k = {0: state}
            return by_k, by_k, by_k, state

        basis_states = bundle([[1.0], [0.0]])
        configured_states = bundle([[0.0], [1.0]])
        report = GaugeAnchorReport(
            gauge_mode="auto_scdm",
            resolved_norb_fix_list=[[[0]]],
            selections=[],
            metric={},
            state_selection_quality={},
            gauge_anchor_quality={},
            symmetry_closure_quality={},
        )
        candidate = SimpleNamespace(
            candidate_id="candidate",
            resolved_norb_fix_list=[[[0]]],
            report=report,
        )
        ctx = SimpleNamespace(
            config=SimpleNamespace(symm_cfg={}, tolerance=1.0e-2),
            full_dim=2,
            method="linearized_lowdin",
            selected_gauge_states="must_be_cleared",
        )
        calls = []

        def fake_metrics(_ctx, _candidate, *, candidate_states=None, state_cache=None):
            calls.append("configured" if candidate_states is not None else "basis")
            if state_cache is not None:
                state_cache["candidate"] = basis_states
            return GaugeCandidateSymmetryMetrics(
                candidate_id="candidate",
                exactification_distance_by_op={"TR": 0.0},
                metadata={
                    "status": "evaluated",
                    "symmetry_adapted_frame": {"status": "applied"},
                },
            )

        decision = SimpleNamespace(rankings=[])
        with (
            patch.object(projection_mod, "_candidate_symmetry_metrics", side_effect=fake_metrics),
            patch.object(
                projection_mod,
                "_configured_basis_states_for_resolved_anchors",
                return_value=configured_states,
            ),
            patch.object(
                projection_mod,
                "_select_validated_auto_gauge_candidate",
                return_value=(candidate, decision),
            ),
        ):
            selected, selected_report = projection_mod._select_projection_gauge(ctx, [candidate])

        self.assertIs(selected, candidate)
        self.assertEqual(calls, ["basis", "configured"])
        self.assertIsNone(ctx.selected_gauge_states)
        certification = selected_report.symmetry_closure_quality["metrics"][0]["metadata"][
            "production_basis_certification"
        ]
        self.assertEqual(certification["status"], "configured_basis_fallback")

    def test_projectors_for_k_uses_full_row_order_for_multilayer_k_mode(self) -> None:
        ham = np.diag(np.arange(6, dtype=float)).astype(np.complex128)
        captured = {}

        def fake_downfold_from_projectors(_ham, u_low, u_high, options):
            captured["u_low"] = np.asarray(u_low)
            captured["u_high"] = None if u_high is None else np.asarray(u_high)
            captured["method"] = options.method
            return SimpleNamespace(heff=np.eye(u_low.shape[1], dtype=np.complex128))

        with patch.object(projection_mod, "downfold_from_projectors", side_effect=fake_downfold_from_projectors):
            projection_mod._projectors_for_k(
                ham,
                np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
                orb0=1,
                num_layer_list=[1, 2],
                num_orb_per_layer_list=[[1], [1, 1]],
                spin="up",
                mode="K1",
                nlow_state_list=[[0], [0], [0]],
                norb_fix_list=[[0], [0], [0]],
                method="fixed_schur",
                e_ref=0.0,
                project_cfg={},
            )

        nonzero_rows = [int(np.flatnonzero(np.abs(captured["u_low"][:, col]) > 1e-12)[0]) for col in range(6)]
        self.assertEqual(nonzero_rows, [0, 1, 2, 4, 3, 5])
        self.assertEqual(captured["method"], "fixed_schur")

    def test_sector_orbital_counts_aggregate_physical_layers_by_source_group(self) -> None:
        q1 = np.zeros((12, 2), dtype=float)
        q2 = np.zeros((12, 2), dtype=float)

        self.assertEqual(
            _sector_orbital_counts(q1, q2, [[22], [22], []], low_dim=24, num_layer_list=[1, 2]),
            (1, 1),
        )
        self.assertEqual(
            _sector_orbital_counts(q1, q2, [[], [], [22]], low_dim=12, num_layer_list=[1, 2]),
            (0, 1),
        )
        self.assertEqual(
            _sector_orbital_counts(
                q1,
                q2,
                [[], [134, 135], [136, 137]],
                low_dim=48,
                num_layer_list=[1, 2],
            ),
            (0, 4),
        )
        self.assertEqual(
            _sector_orbital_counts(
                q1,
                q2,
                [[134, 135], [136, 137], []],
                low_dim=48,
                num_layer_list=[1, 2],
            ),
            (2, 2),
        )

    def test_gamma_projectors_resolve_physical_layers_to_source_group_bands(self) -> None:
        ham = np.eye(6, dtype=np.complex128)
        captured = {}

        def fake_get_h_block(*_args, **_kwargs):
            vecs = np.empty(1, dtype=object)
            vecs[0] = np.eye(3, dtype=np.complex128)
            return None, vecs, None, None

        def fake_assemble_gamma_projectors(_vecs, _idx_list, nlow_state_list, **_kwargs):
            captured["nlow_state_list"] = nlow_state_list
            return np.eye(6, 4, dtype=np.complex128), np.eye(6, 2, k=4, dtype=np.complex128)

        def fake_downfold_from_projectors(_ham, u_low, _u_high, _options):
            return SimpleNamespace(heff=np.eye(u_low.shape[1], dtype=np.complex128))

        with (
            patch.object(projection_mod, "get_H_block", side_effect=fake_get_h_block),
            patch.object(projection_mod, "_assemble_gamma_projectors_from_block_eigenvectors", side_effect=fake_assemble_gamma_projectors),
            patch.object(projection_mod, "downfold_from_projectors", side_effect=fake_downfold_from_projectors),
        ):
            projection_mod._projectors_for_k(
                ham,
                np.array([[0.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0]], dtype=float),
                orb0=1,
                num_layer_list=[1, 2],
                num_orb_per_layer_list=[[1], [1, 1]],
                spin="up",
                mode="gamma",
                nlow_state_list=[[], [0, 1], [2, 3]],
                norb_fix_list=[[], [[[0, 1.0]], [[1, 1.0]]], [[[2, 1.0]], [[0, 1.0]]]],
                method="fixed_schur",
                e_ref=0.0,
                project_cfg={},
            )

        self.assertEqual(captured["nlow_state_list"], [[], [0, 1, 2, 3]])

    def test_gamma_projectors_reject_legacy_source_group_rows_with_num_layer_list(self) -> None:
        ham = np.eye(6, dtype=np.complex128)

        with self.assertRaisesRegex(ValueError, "physical-layer rows"):
            projection_mod._projectors_for_k(
                ham,
                np.array([[0.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0]], dtype=float),
                orb0=1,
                num_layer_list=[1, 2],
                num_orb_per_layer_list=[[1], [1, 1]],
                spin="up",
                mode="gamma",
                nlow_state_list=[[0], [1, 2]],
                norb_fix_list=[[[[0, 1.0]]], [[[1, 1.0]], [[2, 1.0]]]],
                method="fixed_schur",
                e_ref=0.0,
                project_cfg={},
            )

    def test_projectors_reject_mismatched_norb_fix_layer_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "norb_fix_list.*same number of rows"):
            projection_mod._projectors_for_k(
                np.eye(6, dtype=np.complex128),
                np.array([[0.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0]], dtype=float),
                orb0=1,
                num_layer_list=[1, 2],
                num_orb_per_layer_list=[[1], [1, 1]],
                spin="up",
                mode="gamma",
                nlow_state_list=[[], [0], [1]],
                norb_fix_list=[[], [[[0, 1.0]]]],
                method="fixed_schur",
                e_ref=0.0,
                project_cfg={},
            )

    def test_projectors_reject_mismatched_norb_fix_references_per_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer 1 has 2 bands.*1 references"):
            projection_mod._projectors_for_k(
                np.eye(6, dtype=np.complex128),
                np.array([[0.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0]], dtype=float),
                orb0=1,
                num_layer_list=[1, 2],
                num_orb_per_layer_list=[[1], [1, 1]],
                spin="up",
                mode="gamma",
                nlow_state_list=[[], [0, 1], []],
                norb_fix_list=[[], [[[0, 1.0]]], []],
                method="fixed_schur",
                e_ref=0.0,
                project_cfg={},
            )

    def test_action_resolution_rejects_single_nlow_row_instead_of_inferring_from_low_dim(self) -> None:
        q = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        action = {
            "antiunitary": False,
            "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
            "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
            "sector_map": "identity",
        }
        dim = 8  # 2 sectors * 2 q points * 2 orbitals per sector.
        raw_action = np.roll(np.eye(dim, dtype=np.complex128), shift=1, axis=0)
        representation = np.zeros((dim, dim), dtype=np.complex128)
        representation[4:, :4] = np.eye(4, dtype=np.complex128)
        representation[:4, 4:] = np.eye(4, dtype=np.complex128)

        with self.assertRaisesRegex(ValueError, "nlow_state_list must have two qset rows"):
            _resolve_projected_model_action(
                D_low=raw_action,
                support_matrices=[("raw_action", raw_action), ("representation", representation)],
                model_action=action,
                q_model1=q,
                q_model2=q.copy(),
                nlow_state_list=[[10, 11, 12, 13]],
                tol=1.0e-8,
                discover_action_candidates=True,
                accept_support_resolved_action=True,
            )

    def test_projected_action_inference_resolves_layer_exchange_with_provenance(self) -> None:
        q = np.array([[0.0, 0.0]], dtype=float)
        declared = {
            "antiunitary": True,
            "k_map": {"type": "negation", "in_model_frame": True},
            "q_map": {"type": "negation", "in_model_frame": True},
            "sector_map": "identity",
        }
        projected_action = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        resolved, basis_action = _resolve_projected_model_action(
            D_low=projected_action,
            model_action=declared,
            q_model1=q,
            q_model2=q.copy(),
            nlow_state_list=[[54], [55]],
            num_layer_list=[1, 1],
            tol=1.0e-8,
            discover_action_candidates=True,
            accept_support_resolved_action=True,
        )

        assert resolved["sector_map"] == "layer_exchange"
        provenance = basis_action["support_resolution"]["provenance"]
        assert provenance["accepted_by"] == "kp_projected_basis_inference"
        assert provenance["accepted_by_user"] is False
        assert provenance["declared_model_action"]["sector_map"] == "identity"
        assert provenance["selected_action_candidate"]["sector_map"] == "layer_exchange"

    def test_projected_action_inference_preserves_matching_identity_action(self) -> None:
        q = np.array([[0.0, 0.0]], dtype=float)
        declared = {
            "antiunitary": True,
            "k_map": {"type": "negation", "in_model_frame": True},
            "q_map": {"type": "negation", "in_model_frame": True},
            "sector_map": "identity",
        }

        resolved, basis_action = _resolve_projected_model_action(
            D_low=np.eye(2, dtype=np.complex128),
            model_action=declared,
            q_model1=q,
            q_model2=q.copy(),
            nlow_state_list=[[40], [42]],
            num_layer_list=[1, 1],
            tol=1.0e-8,
            discover_action_candidates=True,
            accept_support_resolved_action=True,
        )

        assert resolved["sector_map"] == "identity"
        assert basis_action["support_resolution"]["action_mismatch"] is False
        assert "provenance" not in basis_action["support_resolution"]

    def test_projectors_for_k_uses_configured_downfold_method_and_e_ref(self) -> None:
        q = np.array([[0.0, 0.0]], dtype=float)
        ham = np.diag([0.0, 10.0, 0.0, 10.0]).astype(np.complex128)
        ham[0, 3] = ham[3, 0] = 0.5
        ham[2, 1] = ham[1, 2] = 0.25

        state = _projectors_for_k(
            ham,
            q,
            q.copy(),
            orb0=2,
            spin="up",
            mode="K1",
            nlow_state_list=[[0], [0]],
            norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
            method="fixed_schur",
            e_ref=1.0,
            project_cfg={},
        )

        expected = np.diag([-0.25 / 9.0, -0.0625 / 9.0]).astype(np.complex128)
        np.testing.assert_allclose(state.heff, expected, atol=1.0e-12)

    def test_rep_support_cleaner_stays_action_and_reports_diagnostic_only(self) -> None:
        model_basis_action = {
            "support_resolution": {
                "support_matrix_source": "representation",
                "declared_support_residuals": [
                    {"matrix": "raw_action", "block_off_support_rel": 1.0},
                    {"matrix": "representation", "block_off_support_rel": 2.0e-7},
                ],
                "declared_model_action": {
                    "antiunitary": False,
                    "k_map": {"type": "reflection", "axis_deg": 0.0},
                    "q_map": {"type": "reflection", "axis_deg": 0.0},
                    "sector_map": "layer_exchange",
                },
            }
        }

        matrix_kind, selection = _select_operation_matrix_kind(
            model_basis_action,
            representation_pair_rows=[{"quality_warnings": []}],
        )

        self.assertEqual(matrix_kind, "action")
        self.assertEqual(selection["kind"], "action")
        self.assertEqual(selection["matrix_source"], "raw_h_action_projection")
        self.assertEqual(
            selection["representation_projection_diagnostic"]["status"],
            "raw_action_exactification_problem",
        )
        self.assertTrue(
            selection["representation_projection_diagnostic"]["representation_support_cleaner_than_raw_action"]
        )
        self.assertNotEqual(matrix_kind, "representation")

    def _run_minimal_two_layer_projection(
        self,
        tmp: Path,
        *,
        valley: str,
        operation: str,
        d_up: np.ndarray,
        manifest_entry: dict,
        q_rotation_deg: float | None = 0.0,
        include_default_k_pairs: bool = True,
        include_representation_file: bool = True,
        include_energy_unit: bool = True,
        include_operations: bool = True,
        auto_gauge: bool = False,
    ) -> Path:
        q1_file = tmp / f"{operation}_q1.npy"
        q2_file = tmp / f"{operation}_q2.npy"
        hamk_file = tmp / f"{operation}_hamk.npy"
        symm_dir = tmp / f"{operation}_symmetry_analysis"
        rep_dir = symm_dir / "representations" / valley
        out_dir = tmp / "outputs" / valley / "q06" / "symmetry"
        cfg_path = tmp / f"{operation}_symm.yaml"

        np.save(q1_file, np.array([[0.0, 0.0]], dtype=float))
        np.save(q2_file, np.array([[0.0, 0.0]], dtype=float))

        h_up = np.diag([1.0, 5.0, 1.0, 5.0]).astype(np.complex128)
        hamk = np.zeros((1, 8, 8), dtype=np.complex128)
        hamk[0, :4, :4] = h_up
        hamk[0, 4:, 4:] = h_up
        np.save(hamk_file, hamk)

        d_full = np.zeros((8, 8), dtype=np.complex128)
        d_full[:4, :4] = d_up
        d_full[4:, 4:] = d_up
        rep_dir.mkdir(parents=True)
        if include_representation_file:
            np.savez(rep_dir / f"{operation}.npz", matrix=d_full)
        np.savez(rep_dir / f"{operation}_rawH.npz", matrix=d_full)

        entry = {
            "antiunitary": False,
            "raw_h_operator_file": f"{valley}/{operation}_rawH.npz",
            **manifest_entry,
        }
        if include_representation_file:
            entry["filename"] = f"{valley}/{operation}.npz"
        if include_default_k_pairs and "k_pairs" not in entry:
            entry["k_pairs"] = [[0, 0]]
        nlow_state_list = [[0], [1]] if valley.lower() == "gamma" else [[0], [0]]
        norb_fix_list = [[[[0, 1.0]]], [[[1 if valley.lower() == "gamma" else 0, 1.0]]]]
        (symm_dir / "representations" / "manifest.json").write_text(
            json.dumps({"operations": {valley: {operation: entry}}}),
            encoding="utf-8",
        )

        plot_section = {"hamk_index": 0}
        if q_rotation_deg is not None:
            plot_section["q_rotation_deg"] = q_rotation_deg
        cfg = {
            "case": {"profile": valley, "q_shell": "q06", "output_root": "outputs"},
            "material": {
                "hamk_file": str(hamk_file),
                "energy_unit": "eV",
                "qset1_file": str(q1_file),
                "qset2_file": str(q2_file),
                "spin": "up",
                "num_layers": 2,
                "num_orb_per_layer": [2, 2],
            },
            "plot": plot_section,
            "project": {
                "enable": True,
                "mode": valley,
                "downfold_method": "first_order",
                "nlow_state_list": nlow_state_list,
            },
            "symm": {
                "enable": True,
                "valley": valley,
                "spin": "up",
                "tapw_symmetry_dir": str(symm_dir),
                "tolerance": 1.0e-8,
            },
        }
        if auto_gauge:
            cfg["project"]["gauge"] = "auto"
        else:
            cfg["project"]["norb_fix_list"] = norb_fix_list
        if include_operations:
            cfg["symm"]["operations"] = [operation]
        if not include_energy_unit:
            cfg["material"].pop("energy_unit", None)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        project_dir = tmp / "outputs" / valley / "q06" / "projection"
        project_dir.mkdir(parents=True, exist_ok=True)
        heff = np.zeros((1, 2, 2), dtype=np.complex128)
        np.save(project_dir / "heff.npy", heff)
        project_identity = projection_mod.build_projection_basis_identity(
            qset1=np.load(q1_file),
            qset2=np.load(q2_file),
            spin="up",
            mode=valley.lower(),
            energy_scale=1.0,
            nlow_state_list=nlow_state_list,
            resolved_norb_fix_list=norb_fix_list,
            gauge_mode="auto_scdm" if auto_gauge else "manual_norb_fix_list",
            num_layer_list=[1, 1],
            num_orb_per_layer_list=[[2], [2]],
            orbital_block_dim=2,
            model_dim=2,
            k_indices=[0],
        )
        artifact_identity = {
            **project_identity,
            "heff_hash": hash_array(heff),
        }
        scalar_identity = {key: np.asarray(value) for key, value in artifact_identity.items()}
        np.savez(project_dir / "basis.npz", **scalar_identity)
        np.savez(
            project_dir / "wavefunctions.npz",
            wavefunctions=np.eye(2, dtype=np.complex128)[None, :, :],
            k_indices=np.asarray([0]),
            **scalar_identity,
        )

        cli.main(["symm", "--config", str(cfg_path)])
        return out_dir

    def test_symm_defaults_missing_energy_unit_to_ev(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out_dir = self._run_minimal_two_layer_projection(
                tmp,
                valley="K1",
                operation="C3",
                d_up=np.eye(4, dtype=np.complex128),
                manifest_entry={
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                },
                include_energy_unit=False,
            )

            self.assertTrue((out_dir / "representations.npz").exists())
            self.assertTrue((out_dir / "residuals.csv").exists())
            self.assertTrue((out_dir / "summary.md").exists())
            self.assertFalse((out_dir / "manifest.json").exists())

    def test_symm_infers_operations_from_tapw_manifest_when_omitted(self) -> None:
        manifest = {
            "matrices": [
                {"source_valley": "K1", "target_valley": "K1", "operation": "E", "supported": True, "raw_h_operator_file": "K1/E_rawH.npz"},
                {"source_valley": "K1", "target_valley": "K1", "operation": "C3z", "supported": True, "raw_h_operator_file": "K1/C3z_rawH.npz"},
                {"source_valley": "K1", "target_valley": "K1", "operation": "C3z^2", "supported": True, "raw_h_operator_file": "K1/C3z2_rawH.npz"},
                {"source_valley": "K1", "target_valley": "K2", "operation": "TR", "supported": True, "raw_h_operator_file": "K1/TR_rawH.npz"},
                {"source_valley": "Gamma", "target_valley": "Gamma", "operation": "TR", "supported": True, "raw_h_operator_file": "Gamma/TR_rawH.npz"},
                {"source_valley": "Gamma", "target_valley": "Gamma", "operation": "C3z", "supported": True, "raw_h_operator_file": "Gamma/C3z_rawH.npz"},
            ]
        }

        self.assertEqual(_infer_symmetry_operations_from_manifest(manifest, "K1"), ["C3z"])
        self.assertEqual(_infer_symmetry_operations_from_manifest(manifest, "Gamma"), ["TR", "C3z"])

    def test_symm_loads_packed_tapw_representations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            symm_dir = tmp / "symmetry"
            symm_dir.mkdir()
            metadata = {
                "matrices": [
                    {
                        "operation": "C3z",
                        "key": "C3z",
                        "source_valley": "Gamma",
                        "target_valley": "Gamma",
                        "supported": True,
                        "antiunitary": False,
                        "k_pairs": [[0, 0]],
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                    }
                ]
            }
            np.savez_compressed(
                symm_dir / "representations.npz",
                C3z_data=np.array([1.0 + 0.0j, -1.0 + 0.0j]),
                C3z_indices=np.array([0, 1], dtype=np.int64),
                C3z_indptr=np.array([0, 1, 2], dtype=np.int64),
                C3z_shape=np.array([2, 2], dtype=np.int64),
                metadata_json=json.dumps(metadata),
            )

            run_cfg = SimpleNamespace(
                symm_cfg={"tapw_symmetry_dir": str(symm_dir)},
                cfg_dir=str(tmp),
                valley="Gamma",
            )
            rep_root, manifest, requests = projection_mod._load_manifest_and_operation_requests(run_cfg)
            self.assertEqual(rep_root, symm_dir)
            self.assertEqual([request["source"] for request in requests], ["C3z"])

            entry = _operation_entry(manifest, "Gamma", "C3z")
            self.assertEqual(entry["raw_h_operator_file"], "representations.npz:C3z")
            matrix = projection_mod._load_matrix(rep_root / entry["raw_h_operator_file"])
            np.testing.assert_allclose(matrix.toarray(), np.diag([1.0, -1.0]))

    def test_symm_rejects_packed_tapw_representation_without_k_route(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            packed_path = Path(td) / "representations.npz"
            metadata = {
                "matrices": [
                    {
                        "operation": "C3z",
                        "key": "C3z",
                        "source_valley": "Gamma",
                        "target_valley": "Gamma",
                        "supported": True,
                    }
                ]
            }
            np.savez_compressed(
                packed_path,
                C3z_data=np.array([1.0 + 0.0j]),
                C3z_indices=np.array([0], dtype=np.int64),
                C3z_indptr=np.array([0, 1], dtype=np.int64),
                C3z_shape=np.array([1, 1], dtype=np.int64),
                metadata_json=json.dumps(metadata),
            )

            _root, manifest = projection_mod._load_packed_tapw_symmetry_manifest(
                packed_path
            )
            with self.assertRaisesRegex(
                ValueError, "k_pairs/source_indices.*k rule"
            ):
                projection_mod._resolve_operation_entries(
                    manifest=manifest,
                    valley="Gamma",
                    operation_requests=[
                        {"source": "C3z", "output": "C3z", "requested": "C3z"}
                    ],
                    nk=2,
                    default_k_index=0,
                )

    def test_symm_omitted_operations_runs_and_reports_inference(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stream = io.StringIO()
            with redirect_stdout(stream):
                out_dir = self._run_minimal_two_layer_projection(
                    tmp,
                    valley="K1",
                    operation="C3",
                    d_up=np.eye(4, dtype=np.complex128),
                    manifest_entry={
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                    },
                    include_operations=False,
                )

            _arrays, summary = _load_canonical_symmetry_payload(out_dir)
            self.assertIn("inferred symmetry operations: C3z", stream.getvalue())
            self.assertEqual([row["operation"] for row in summary["operations"]], ["C3z"])
            self.assertEqual(summary["kp_symm_exactification"]["config"]["reject_if_off_support_rel_gt"], 5.0e-3)

    def test_symm_auto_gauge_candidate_exactification_uses_valley_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            observed_thresholds: list[float] = []

            def fake_exactify_loaded_symmetry_source(**kwargs):
                exact_cfg = kwargs["raw_config"]["exactification"]
                observed_thresholds.append(float(exact_cfg["reject_if_off_support_rel_gt"]))
                matrices = kwargs["matrices"]
                reports = {
                    str(name): {
                        "report": {
                            "status": "exactified",
                            "distance_mod_global_phase": 0.0,
                            "phase_std_deg": 0.0,
                        },
                        "support_diagnostics": {"off_support_rel": 1.0e-3},
                    }
                    for name in matrices
                }
                return dict(matrices), reports

            with patch.object(projection_mod, "exactify_loaded_symmetry_source", side_effect=fake_exactify_loaded_symmetry_source):
                self._run_minimal_two_layer_projection(
                    tmp,
                    valley="K1",
                    operation="C3",
                    d_up=np.eye(4, dtype=np.complex128),
                    manifest_entry={
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                    },
                    include_operations=False,
                    auto_gauge=True,
                )

            self.assertTrue(observed_thresholds)
            self.assertTrue(all(value == 5.0e-3 for value in observed_thresholds))

    def test_symmetry_gauge_resolver_does_not_require_or_write_project_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._run_minimal_two_layer_projection(
                tmp,
                valley="K1",
                operation="C3",
                d_up=np.eye(4, dtype=np.complex128),
                manifest_entry={
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                },
                auto_gauge=True,
            )
            case_root = tmp / "outputs" / "K1" / "q06"
            shutil.rmtree(case_root)

            report = projection_mod.resolve_symmetry_validated_project_gauge(
                str(tmp / "C3_symm.yaml")
            )

            self.assertEqual(report.gauge_mode, "auto_scdm")
            self.assertEqual(report.symmetry_closure_quality["status"], "validated")
            self.assertFalse(case_root.exists())

    def test_symmetry_gauge_resolver_skips_full_space_covariance_recheck(self) -> None:
        run_cfg = SimpleNamespace(valley="Gamma")
        context = object()
        route_resolver = object()
        report = GaugeAnchorReport(
            gauge_mode="auto_scdm",
            resolved_norb_fix_list=[[[0]]],
            selections=[],
            metric={},
            state_selection_quality={},
            gauge_anchor_quality={},
            symmetry_closure_quality={"status": "validated"},
        )

        def fake_build_context(
            observed_run_cfg,
            *,
            create_output_dir,
            validate_full_space_covariance,
            packed_k_route_resolver,
        ):
            self.assertIs(observed_run_cfg, run_cfg)
            self.assertFalse(create_output_dir)
            self.assertFalse(validate_full_space_covariance)
            self.assertIs(packed_k_route_resolver, route_resolver)
            return context

        with (
            patch.object(projection_mod, "_load_projection_run_config", return_value=run_cfg),
            patch.object(
                projection_mod,
                "_build_projection_run_context",
                side_effect=fake_build_context,
            ),
            patch.object(
                projection_mod,
                "_actual_sampled_k_route_resolver",
                return_value=route_resolver,
            ),
            patch.object(
                projection_mod,
                "_resolve_validated_projection_gauge",
                return_value=(object(), report),
            ),
        ):
            actual = projection_mod.resolve_symmetry_validated_project_gauge("unused.yaml")

        self.assertIs(actual, report)

    def test_formal_symmetry_reuses_project_resolved_gauge(self) -> None:
        report = GaugeAnchorReport(
            gauge_mode="auto_scdm",
            resolved_norb_fix_list=[[[0]], [[1]]],
            selections=[],
            metric={},
            state_selection_quality={},
            gauge_anchor_quality={},
            symmetry_closure_quality={
                "status": "validated",
                "selected_candidate_id": "project-selected",
            },
        )

        with patch.object(
            projection_mod,
            "_resolve_validated_projection_gauge",
            side_effect=AssertionError("formal symmetry reran Auto-gauge"),
        ):
            candidate, actual = projection_mod._resolved_gauge_for_symmetry(
                object(),
                gauge_report=report,
            )

        self.assertIs(actual, report)
        self.assertEqual(candidate.candidate_id, "project-selected")
        self.assertEqual(candidate.resolved_norb_fix_list, report.resolved_norb_fix_list)

    def test_certified_frames_accept_roundoff_level_matrix_differences(self) -> None:
        project_frame = np.eye(4, dtype=np.complex128)
        recomputed_frame = project_frame.copy()
        recomputed_frame[0, 0] += 4.0e-16

        residual, bound = projection_mod._certified_frame_equivalence(
            project_frame,
            recomputed_frame,
        )

        self.assertGreater(residual, 0.0)
        self.assertLessEqual(residual, bound)

    def test_summary_writer_reports_skipped_full_space_covariance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary_path = Path(td) / "summary.md"
            projection_mod._write_summary_md(
                summary_path,
                {
                    "valley": "Gamma",
                    "spin": "all",
                    "tolerance": 1.0e-8,
                    "full_dim": 8,
                    "low_dim": 4,
                    "operations": [
                        {
                            "operation": "C3z",
                            "antiunitary": False,
                            "pairs": [
                                {
                                    "target_k_index": 0,
                                    "source_k_index": 0,
                                    "full_space_covariance_residual": None,
                                    "raw": {
                                        "heff_covariance_residual": None,
                                        "subspace_leakage": 0.0,
                                    },
                                }
                            ],
                        }
                    ],
                },
            )

            self.assertIn("full cov=skipped", summary_path.read_text(encoding="utf-8"))
            self.assertIn("raw cov=skipped", summary_path.read_text(encoding="utf-8"))

    def test_loads_matching_project_prepared_symmetry_package(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            symm_dir = tmp / "symmetry"
            project_dir = tmp / "projection"
            symm_dir.mkdir()
            project_dir.mkdir()
            identity = {
                field: f"value-{field}"
                for field in projection_mod.PROJECTION_ARTIFACT_IDENTITY_FIELDS
            }
            metadata = {
                "artifact_identity": identity,
                "package_preparation": {
                    "owner": "kp_project",
                    "status": "prepared",
                },
                "operations": [],
            }
            np.savez_compressed(
                symm_dir / "representations.npz",
                __metadata_json__=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            run_cfg = SimpleNamespace(
                cfg_dir=str(tmp),
                project_cfg={"out_dir": str(project_dir)},
                symm_cfg={"output_dir": str(symm_dir)},
            )

            with (
                patch.object(
                    projection_mod,
                    "_load_projection_run_config",
                    return_value=run_cfg,
                ),
                patch.object(
                    projection_mod,
                    "_load_project_artifact_identity",
                    return_value=identity,
                ),
            ):
                actual = projection_mod.load_project_prepared_symmetry_package(
                    "unused.yaml"
                )

        self.assertEqual(actual, metadata)

    def test_symm_accepts_release_manifest_with_rawh_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out_dir = self._run_minimal_two_layer_projection(
                tmp,
                valley="K1",
                operation="C3",
                d_up=np.eye(4, dtype=np.complex128),
                manifest_entry={
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                },
                include_representation_file=False,
            )

            arrays, summary = _load_canonical_symmetry_payload(out_dir)
            row = summary["operations"][0]
            self.assertIn("C3z", arrays)
            self.assertEqual(row["matrix_file"], "representations.npz")
            self.assertEqual(row["matrix_array_key"], "C3z")
            self.assertEqual(row["matrix_source"], "kp_symm_exactified_action")
            self.assertEqual(row["matrix_scope"], "point_independent_continuum_action")
            self.assertEqual(row["valid_k_domain"], "all_model_k")
            self.assertTrue(row["reference_pairs_are_not_domain_restrictions"])
            self.assertEqual(row["exactification_reference_pairs"], row["pairs"])
            self.assertEqual(row["projection_diagnostic_pairs"], row["pairs"])
            self.assertNotIn("representation_file", row)
            self.assertFalse((out_dir / "diagnostics" / "C3_low_representation_raw.npy").exists())
            polynomial_coordinate = summary["kp_symm_exactification"]["polynomial_coordinate"]
            self.assertEqual(
                polynomial_coordinate["coordinate_convention"],
                "right_handed_model_cartesian_reciprocal_v1",
            )
            self.assertEqual(polynomial_coordinate["origin"], [0.0, 0.0])
            self.assertEqual(
                polynomial_coordinate["origin_role"],
                "exactified_valley_expansion_origin_in_model_cartesian",
            )
            self.assertEqual(polynomial_coordinate["valley"], "K1")

    def test_symm_auto_frame_uses_reflection_axis_when_rotation_is_not_configured(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            d = np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            )
            out_dir = self._run_minimal_two_layer_projection(
                tmp,
                valley="K1",
                operation="C2T",
                d_up=d,
                manifest_entry={
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 150.0},
                    "q_map": {"type": "reflection", "axis_deg": 150.0},
                    "sector_map": "layer_exchange",
                },
                q_rotation_deg=None,
            )

            _arrays, summary = _load_canonical_symmetry_payload(out_dir)
            self.assertAlmostEqual(summary["frame"]["k_transform"]["rotation_deg"], 210.0)
            self.assertEqual(summary["frame"]["inference"]["source"], "reflection_axis")
            op = summary["operations"][0]
            self.assertAlmostEqual(op["source_action"]["k_map"]["axis_deg"], 150.0)
            self.assertAlmostEqual(op["model_action"]["k_map"]["axis_deg"], 360.0)
            self.assertEqual(op["model_action"]["sector_map"], "layer_exchange")

    def _reader_operation_row(
        self,
        *,
        name: str,
        operation: str,
        matrix_file: str,
        antiunitary: bool,
        k_map: dict,
        q_map: dict,
        sector_map: str,
    ) -> dict:
        return {
            "name": name,
            "operation": operation,
            "matrix_file": matrix_file,
            "antiunitary": antiunitary,
            "k_map": dict(k_map),
            "q_map": dict(q_map),
            "sector_map": sector_map,
            "spin_map": "from_kp_symm_output",
            "valley_map": "identity",
            "matrix_kind": "action",
            "source_matrix_role": "raw_h_sewing_action",
            "source_gauge": "raw_saved_TAPW",
            "target_role": "continuum_internal_rep",
            "gauge_correction": {"kind": "none"},
            "antiunitary_convention": "U_K" if antiunitary else "none",
        }

    def test_symm_rejects_nonstandard_operation_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported symm operation"):
            _validate_operation_label("C4")

    def test_symm_reads_standard_tr_manifest_entry(self) -> None:
        manifest = {"operations": {"Gamma": {"TR": {"filename": "Gamma/TR.npz", "antiunitary": True}}}}

        entry = _operation_entry(manifest, "Gamma", "TR")

        self.assertEqual(entry["filename"], "Gamma/TR.npz")

    def test_tr_request_uses_tr_as_source_manifest_name(self) -> None:
        self.assertEqual(_source_manifest_operation_name("TR"), "TR")
        self.assertEqual(_source_manifest_operation_name("T"), "TR")

    def test_source_manifest_requires_unique_raw_h_action_operator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rep_root = Path(tmpdir)
            np.savez(rep_root / "C3.npz", matrix=np.eye(2, dtype=np.complex128))

            with self.assertRaisesRegex(ValueError, "must provide raw_h_operator_file"):
                _build_action_representation(
                    operation="C3",
                    entry={"filename": "C3.npz"},
                    rep_root=rep_root,
                    filename="C3.npz",
                    antiunitary=False,
                    spin="up",
                    full_dim=2,
                    tolerance=1.0e-8,
                )

    def test_manifest_k_pairs_missing_does_not_fallback_to_default_k_index(self) -> None:
        entry = {
            "filename": "K1/C3.npz",
            "raw_h_operator_file": "K1/C3_rawH.npz",
            "k_map": {"type": "rotation", "angle_deg": 120.0},
            "q_map": {"type": "rotation", "angle_deg": 120.0},
            "sector_map": "identity",
        }

        with self.assertRaisesRegex(ValueError, "k_pairs/source_indices.*k rule"):
            _pairs_from_entry(entry, nk=3, default_k_index=1)

    def test_default_k_index_fallback_requires_explicit_diagnostic_provenance(self) -> None:
        entry = {
            "filename": "K1/C3.npz",
            "raw_h_operator_file": "K1/C3_rawH.npz",
            "k_map": {"type": "rotation", "angle_deg": 120.0},
            "q_map": {"type": "rotation", "angle_deg": 120.0},
            "sector_map": "identity",
        }

        pairs = _pairs_from_entry(entry, nk=3, default_k_index=1, allow_default_k_index=True)

        self.assertEqual(pairs, [(1, 1)])
        self.assertTrue(entry["k_pairs_inferred_from_default_k_index"])
        self.assertEqual(entry["candidate_source"], "diagnostic_default_k_index")
        self.assertFalse(entry["_k_pairs_provenance"]["authored_in_manifest"])

    def test_symm_manifest_missing_k_pairs_rejected_even_with_plot_hamk_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "k_pairs/source_indices.*k rule"):
                self._run_minimal_two_layer_projection(
                    tmp,
                    valley="K1",
                    operation="C3",
                    d_up=np.eye(4, dtype=np.complex128),
                    manifest_entry={
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                    },
                    include_default_k_pairs=False,
                )

    def test_model_frame_action_rotates_source_reflection_axis(self) -> None:
        source = {
            "antiunitary": True,
            "k_map": {"type": "reflection", "axis_deg": 180.0},
            "q_map": {"type": "reflection", "axis_deg": 180.0},
            "sector_map": "layer_exchange",
            "spin_map": "from_kp_symm_output",
            "valley_map": "identity",
        }

        model = _model_frame_action_metadata(source, rotation_deg=30.0)

        self.assertEqual(source["k_map"]["axis_deg"], 180.0)
        self.assertEqual(model["k_map"]["axis_deg"], 210.0)
        self.assertEqual(model["q_map"]["axis_deg"], 210.0)
        self.assertTrue(model["k_map"]["in_model_frame"])
        self.assertTrue(model["q_map"]["in_model_frame"])

    def test_k_single_valley_c2t_model_action_uses_frame_conjugation(self) -> None:
        source = {
            "antiunitary": True,
            "k_map": {"type": "reflection", "axis_deg": 60.0},
            "q_map": {"type": "reflection", "axis_deg": 60.0},
            "sector_map": "identity",
            "spin_map": "from_kp_symm_output",
            "valley_map": "identity",
        }

        model = _model_action_metadata(source, valley="K1", operation="C2T", rotation_deg=210.0)

        self.assertEqual(source["sector_map"], "identity")
        self.assertEqual(model["sector_map"], "identity")
        self.assertEqual(model["k_map"], {"type": "reflection", "axis_deg": 270.0, "in_model_frame": True})
        self.assertEqual(model["q_map"], {"type": "reflection", "axis_deg": 270.0, "in_model_frame": True})
        self.assertEqual(model["action_source"], "derived_by_frame_conjugation")

    def test_symm_projects_spin_up_antiunitary_representation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            q1_file = tmp / "q1.npy"
            q2_file = tmp / "q2.npy"
            hamk_file = tmp / "hamk.npy"
            symm_dir = tmp / "symmetry_analysis_test"
            rep_dir = symm_dir / "representations" / "K1"
            out_dir = tmp / "outputs" / "K1" / "q06" / "symmetry"
            cfg_path = tmp / "mote2_4_K.yaml"

            np.save(q1_file, np.array([[0.0, 0.0]], dtype=float))
            np.save(q2_file, np.array([[0.0, 0.0]], dtype=float))

            h_up = np.diag([1.0, 5.0, 1.0, 5.0]).astype(np.complex128)
            h_down = np.diag([2.0, 7.0, 2.0, 7.0]).astype(np.complex128)
            hamk = np.zeros((1, 8, 8), dtype=np.complex128)
            hamk[0, :4, :4] = h_up
            hamk[0, 4:, 4:] = h_down
            np.save(hamk_file, hamk)

            d_up = np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            )
            d_full = np.zeros((8, 8), dtype=np.complex128)
            d_full[:4, :4] = d_up
            d_full[4:, 4:] = d_up
            rep_dir.mkdir(parents=True)
            np.savez(rep_dir / "C2T.npz", matrix=d_full)
            np.savez(rep_dir / "C2T_rawH.npz", matrix=d_full)
            np.savez(rep_dir / "C3.npz", matrix=np.eye(8, dtype=np.complex128))
            np.savez(rep_dir / "C3_rawH.npz", matrix=np.eye(8, dtype=np.complex128))
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "operations": {
                            "K1": {
                                "C3": {
                                    "antiunitary": False,
                                    "filename": "K1/C3.npz",
                                    "raw_h_operator_file": "K1/C3_rawH.npz",
                                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                                    "sector_map": "identity",
                                    "k_pairs": [[0, 0]],
                                },
                                "C2T": {
                                    "antiunitary": True,
                                    "filename": "K1/C2T.npz",
                                    "raw_h_operator_file": "K1/C2T_rawH.npz",
                                    "k_map": {"type": "reflection", "axis_deg": 180.0},
                                    "q_map": {"type": "reflection", "axis_deg": 180.0},
                                    "sector_map": "layer_exchange",
                                    "k_pairs": [[0, 0]],
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            cfg = {
                "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
                "material": {
                    "hamk_file": str(hamk_file),
                    "energy_unit": "eV",
                    "qset1_file": str(q1_file),
                    "qset2_file": str(q2_file),
                    "spin": "up",
                    "num_layers": 2,
                    "num_orb_per_layer": [2, 2],
                },
                "plot": {"hamk_index": 0, "q_rotation_deg": 30.0},
                "project": {
                    "enable": True,
                    "mode": "K1",
                    "downfold_method": "first_order",
                    "nlow_state_list": [[0], [0]],
                    "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
                },
                "symm": {
                    "enable": True,
                    "valley": "K1",
                    "spin": "up",
                    "tapw_symmetry_dir": str(symm_dir),
                    "operations": ["C3", "C2T"],
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            _write_projection_identity_artifacts(
                tmp / "outputs" / "K1" / "q06" / "projection",
                hamk_file=hamk_file,
                hamk=hamk,
                q1=np.load(q1_file),
                q2=np.load(q2_file),
                mode="K1",
                nlow_state_list=[[0], [0]],
                resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
            )

            cli.main(["symm", "--config", str(cfg_path)])

            arrays, summary = _load_canonical_symmetry_payload(out_dir)
            raw = arrays["C2T"]
            c3_raw = arrays["C3z"]
            self.assertEqual(raw.shape, (2, 2))
            np.testing.assert_allclose(raw, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)
            np.testing.assert_allclose(c3_raw / c3_raw[0, 0], np.eye(2), atol=1e-12)
            self.assertFalse((out_dir / "C2T_low_polar.npy").exists())
            self.assertFalse((out_dir / "C2T_low_representation_raw.npy").exists())

            self.assertEqual(summary["frame"]["q_transform"]["formula"], "q_model = R(rotation_deg) @ (layer_mean - q_source)")
            self.assertEqual(summary["frame"]["q_transform"]["rotation_deg"], 30.0)
            self.assertEqual(summary["q_model"]["files"]["layer1"], "q_model_layer1.npy")
            self.assertEqual(summary["q_model"]["files"]["layer2"], "q_model_layer2.npy")
            by_name = {op["operation"]: op for op in summary["operations"]}
            self.assertFalse(by_name["C3"]["antiunitary"])
            self.assertTrue(by_name["C2T"]["antiunitary"])
            self.assertEqual(by_name["C2T"]["source_action"]["k_map"]["axis_deg"], 180.0)
            self.assertEqual(by_name["C2T"]["model_action"]["k_map"]["axis_deg"], 210.0)
            self.assertEqual(by_name["C2T"]["model_action"]["sector_map"], "layer_exchange")
            stale_support_flag = "allow_" + "support_discovery"
            self.assertNotIn(stale_support_flag, by_name["C2T"])
            self.assertTrue(by_name["C2T"]["k_map"]["in_model_frame"])
            self.assertLess(by_name["C3"]["pairs"][0]["raw"]["heff_covariance_residual"], 1.0e-12)
            self.assertLess(by_name["C2T"]["pairs"][0]["raw"]["heff_covariance_residual"], 1.0e-12)
            self.assertLess(by_name["C2T"]["pairs"][0]["raw"]["subspace_leakage"], 1.0e-12)

    def test_symm_records_exactified_model_basis_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            q1_file = tmp / "q1.npy"
            q2_file = tmp / "q2.npy"
            hamk_file = tmp / "hamk.npy"
            symm_dir = tmp / "symmetry_analysis_test"
            rep_dir = symm_dir / "representations" / "K1"
            out_dir = tmp / "outputs" / "K1" / "q06" / "symmetry"
            cfg_path = tmp / "k1.yaml"

            np.save(q1_file, np.array([[0.0, 0.0]], dtype=float))
            np.save(q2_file, np.array([[0.0, 0.0]], dtype=float))

            h_up = np.diag([1.0, 5.0, 1.0, 5.0]).astype(np.complex128)
            h_down = h_up.copy()
            hamk = np.zeros((1, 8, 8), dtype=np.complex128)
            hamk[0, :4, :4] = h_up
            hamk[0, 4:, 4:] = h_down
            np.save(hamk_file, hamk)

            d_swap_layers = np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            )
            d_full = np.zeros((8, 8), dtype=np.complex128)
            d_full[:4, :4] = d_swap_layers
            d_full[4:, 4:] = d_swap_layers
            rep_dir.mkdir(parents=True)
            np.savez(rep_dir / "C2.npz", matrix=d_full)
            np.savez(rep_dir / "C2_rawH.npz", matrix=d_full)
            np.savez(rep_dir / "C3.npz", matrix=np.eye(8, dtype=np.complex128))
            np.savez(rep_dir / "C3_rawH.npz", matrix=np.eye(8, dtype=np.complex128))
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "operations": {
                            "K1": {
                                "C3": {
                                    "antiunitary": False,
                                    "filename": "K1/C3.npz",
                                    "raw_h_operator_file": "K1/C3_rawH.npz",
                                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                                    "sector_map": "identity",
                                    "k_pairs": [[0, 0]],
                                },
                                "C2": {
                                    "antiunitary": False,
                                    "filename": "K1/C2.npz",
                                    "raw_h_operator_file": "K1/C2_rawH.npz",
                                    "k_map": {"type": "reflection", "axis_deg": 0.0},
                                    "q_map": {"type": "reflection", "axis_deg": 0.0},
                                    "sector_map": "layer_exchange",
                                    "k_pairs": [[0, 0]],
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            cfg = {
                "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
                "material": {
                    "hamk_file": str(hamk_file),
                    "energy_unit": "eV",
                    "qset1_file": str(q1_file),
                    "qset2_file": str(q2_file),
                    "spin": "up",
                    "num_layers": 2,
                    "num_orb_per_layer": [2, 2],
                },
                "plot": {"hamk_index": 0},
                "project": {
                    "enable": True,
                    "mode": "K1",
                    "downfold_method": "first_order",
                    "nlow_state_list": [[0], [0]],
                    "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
                },
                "symm": {
                    "enable": True,
                    "valley": "K1",
                    "spin": "up",
                    "tapw_symmetry_dir": str(symm_dir),
                    "operations": ["C3", "C2"],
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            _write_projection_identity_artifacts(
                tmp / "outputs" / "K1" / "q06" / "projection",
                hamk_file=hamk_file,
                hamk=hamk,
                q1=np.load(q1_file),
                q2=np.load(q2_file),
                mode="K1",
                nlow_state_list=[[0], [0]],
                resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
            )

            cli.main(["symm", "--config", str(cfg_path)])

            _arrays, summary = _load_canonical_symmetry_payload(out_dir)
            rows = {operation["operation"]: operation for operation in summary["operations"]}
            c3 = rows["C3"]
            self.assertEqual(c3["declared_model_action"]["sector_map"], "identity")
            self.assertEqual(c3["model_action"]["sector_map"], "identity")
            self.assertEqual(c3["sector_map"], "identity")
            self.assertTrue(c3["model_basis_action"]["complete"])
            self.assertEqual(c3["model_basis_action"]["sector_map"], "identity")
            self.assertFalse(c3["model_basis_action"]["support_resolution"]["action_mismatch"])

            row = rows["C2"]
            self.assertEqual(row["source_action"]["sector_map"], "layer_exchange")
            self.assertEqual(row["declared_model_action"]["sector_map"], "layer_exchange")
            self.assertEqual(row["model_action"]["sector_map"], "layer_exchange")
            self.assertEqual(row["sector_map"], "layer_exchange")
            self.assertTrue(row["model_basis_action"]["complete"])
            self.assertEqual(row["model_basis_action"]["sector_map"], "layer_exchange")
            self.assertFalse(row["model_basis_action"]["support_resolution"]["action_mismatch"])
            self.assertEqual(row["model_basis_action"]["support_resolution"]["block_off_support_rel"], 0.0)
            self.assertEqual(row["matrix_kind"], "continuum_internal_rep_exact")
            self.assertEqual(row["matrix_source"], "kp_symm_exactified_action")
            self.assertEqual(row["matrix_file"], "representations.npz")
            self.assertEqual(row["matrix_array_key"], "C2")
            self.assertEqual(row["source_matrix_projection_report"]["report"]["status"], "exactified")
            self.assertEqual(row["internal_resolved_action"]["sector_map"], "layer_exchange")
            self.assertEqual(
                row["model_basis_action"]["support_resolution"]["declared_model_action"]["sector_map"],
                "layer_exchange",
            )
            self.assertEqual(
                row["model_basis_action"]["support_resolution"]["selected_model_action"]["sector_map"],
                "layer_exchange",
            )
            self.assertEqual(
                [(item["source_sector"], item["target_sector"]) for item in row["model_basis_action"]["items"]],
                [("L1", "L2"), ("L2", "L1")],
            )
            self.assertEqual(
                [(item["source_q_index"], item["target_q_index"]) for item in row["model_basis_action"]["items"]],
                [(0, 0), (0, 0)],
            )

            loaded = load_symmetry_source(
                {"type": "kp_symm_output", "path": str(out_dir), "operations": [{"name": "C2", "operation": "C2"}]},
                base=tmp,
                expected_dim=2,
            )
            loaded_row = loaded.metadata["operations"][0]
            self.assertEqual(loaded_row["model_action"]["sector_map"], "layer_exchange")
            self.assertEqual(loaded_row["model_basis_action"]["items"], row["model_basis_action"]["items"])
            np.testing.assert_allclose(
                loaded.generator.get_operator("C2"),
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
                atol=1.0e-12,
            )

    def test_symm_keeps_identity_action_when_matrix_support_is_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_dir = self._run_minimal_two_layer_projection(
                tmp,
                valley="K1",
                operation="C2",
                d_up=np.eye(4, dtype=np.complex128),
                manifest_entry={
                    "k_map": {"type": "reflection", "axis_deg": 0.0},
                    "q_map": {"type": "reflection", "axis_deg": 0.0},
                    "sector_map": "identity",
                },
            )

            _arrays, summary = _load_canonical_symmetry_payload(out_dir)
            row = summary["operations"][0]
            self.assertEqual(row["declared_model_action"]["sector_map"], "identity")
            self.assertEqual(row["model_action"]["sector_map"], "identity")
            self.assertEqual(row["model_basis_action"]["sector_map"], "identity")
            self.assertFalse(row["model_basis_action"]["support_resolution"]["action_mismatch"])
            self.assertEqual(
                [(item["source_sector"], item["target_sector"]) for item in row["model_basis_action"]["items"]],
                [("L1", "L1"), ("L2", "L2")],
            )

    def test_c2_action_audit_for_gamma_report_exists(self) -> None:
        declared_model_action = {
            "antiunitary": False,
            "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
            "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
            "sector_map": "identity",
        }
        audit = _c2_action_audit_for_gamma(
            mode="gamma",
            operation="C2",
            matrix_kind="action",
            matrix_selection={"representation_projection_diagnostic": {"status": "not_selected"}},
            model_basis_action={
                "support_resolution": {
                    "declared_support_residual": 0.0,
                    "declared_support_residuals": [
                        {"matrix": "raw_action", "block_off_support_rel": 0.0},
                        {"matrix": "representation", "block_off_support_rel": 1.0e-7},
                    ],
                }
            },
            raw_matrix=np.eye(2, dtype=np.complex128),
            representation_matrix=np.eye(2, dtype=np.complex128),
            pair_rows=[{"full_space_covariance_residual": 0.0}],
            representation_pair_rows=[],
            declared_model_action=declared_model_action,
            combined_raw_h_residual=0.0,
        )

        assert audit is not None
        assert audit["matrix_kind"] == "action"
        assert audit["matrix_source"] == "raw_h_action_projection"
        assert audit["D_low_action_support_residual"] == 0.0
        assert audit["D_low_rep_support_residual"] == 1.0e-7
        assert audit["D_low_action_vs_rep_norm"] == 0.0
        assert audit["declared_model_action"]["sector_map"] == "identity"
        assert audit["sector_map"] == "identity"
        assert audit["q_map"]["type"] == "reflection"

    def test_symm_can_export_up_to_down_spin_sewing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            q1_file = tmp / "q1.npy"
            q2_file = tmp / "q2.npy"
            hamk_file = tmp / "hamk.npy"
            symm_dir = tmp / "symmetry_analysis_test"
            rep_dir = symm_dir / "representations" / "M1"
            out_dir = tmp / "outputs" / "M1" / "q06" / "symmetry"
            cfg_path = tmp / "mgi2_M1.yaml"

            np.save(q1_file, np.array([[0.0, 0.0]], dtype=float))
            np.save(q2_file, np.array([[0.0, 0.0]], dtype=float))

            h_up = np.diag([1.0, 5.0, 1.0, 5.0]).astype(np.complex128)
            h_down = np.diag([1.0, 5.0, 1.0, 5.0]).astype(np.complex128)
            hamk = np.zeros((1, 8, 8), dtype=np.complex128)
            hamk[0, :4, :4] = h_up
            hamk[0, 4:, 4:] = h_down
            np.save(hamk_file, hamk)

            d_du = np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            )
            d_full = np.zeros((8, 8), dtype=np.complex128)
            d_full[4:, :4] = d_du
            rep_dir.mkdir(parents=True)
            np.savez(rep_dir / "TR.npz", matrix=d_full)
            np.savez(rep_dir / "TR_rawH.npz", matrix=d_full)
            np.savez(rep_dir / "TR_PG.npz", matrix=np.eye(8, dtype=np.complex128))
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "matrices": [
                            {
                                "valley_label": "M1",
                                "operation": "TR",
                                "antiunitary": True,
                                "file": "M1/TR.npz",
                                "pg_file": "M1/TR_PG.npz",
                                "raw_h_operator_file": "M1/TR_rawH.npz",
                                "k_map": {"type": "negation"},
                                "q_map": {"type": "negation"},
                                "sector_map": "layer_exchange",
                                "k_pairs": [[0, 0]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cfg = {
                "case": {"profile": "M1", "q_shell": "q06", "output_root": "outputs"},
                "material": {
                    "hamk_file": str(hamk_file),
                    "energy_unit": "eV",
                    "qset1_file": str(q1_file),
                    "qset2_file": str(q2_file),
                    "spin": "up",
                    "num_layers": 2,
                    "num_orb_per_layer": [2, 2],
                },
                "plot": {"hamk_index": 0},
                "project": {
                    "enable": True,
                    "mode": "M1",
                    "downfold_method": "first_order",
                    "nlow_state_list": [[0], [0]],
                    "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
                },
                "symm": {
                    "enable": True,
                    "valley": "M1",
                    "spin": "up",
                    "spin_sector_sewing": "up_to_down",
                    "tapw_symmetry_dir": str(symm_dir),
                    "operations": ["TR"],
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            _write_projection_identity_artifacts(
                tmp / "outputs" / "M1" / "q06" / "projection",
                hamk_file=hamk_file,
                hamk=hamk,
                q1=np.load(q1_file),
                q2=np.load(q2_file),
                mode="M1",
                nlow_state_list=[[0], [0]],
                resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
            )

            cli.main(["symm", "--config", str(cfg_path)])

            arrays, summary = _load_canonical_symmetry_payload(out_dir)
            raw = arrays["TR"]
            np.testing.assert_allclose(raw, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1.0e-12)
            self.assertFalse((out_dir / "TR_low_polar.npy").exists())
            self.assertFalse((out_dir / "TR_low_representation_raw.npy").exists())
            row = summary["operations"][0]
            self.assertEqual(row["spin_sector_sewing"], "up_to_down")
            self.assertEqual(row["source_spin"], "up")
            self.assertEqual(row["target_spin"], "down")
            self.assertEqual(row["matrix_kind"], "continuum_internal_rep_exact")
            self.assertEqual(row["matrix_source"], "kp_symm_exactified_action")
            self.assertEqual(row["matrix_file"], "representations.npz")
            self.assertEqual(row["matrix_array_key"], "TR")
            self.assertEqual(row["model_action"]["sector_map"], "layer_exchange")
            self.assertEqual(row["source_matrix_projection_report"]["report"]["status"], "exactified")
            self.assertLess(row["pairs"][0]["raw"]["heff_covariance_residual"], 1.0e-12)

    def test_symm_developer_outputs_save_polar_and_representation_files_under_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            d = np.eye(4, dtype=np.complex128)
            out_dir = self._run_minimal_two_layer_projection(
                tmp,
                valley="K1",
                operation="C3",
                d_up=d,
                manifest_entry={
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                },
                q_rotation_deg=0.0,
            )
            cfg_path = tmp / "C3_symm.yaml"
            symm_dir = tmp / "C3_symmetry_analysis"
            rep_dir = symm_dir / "representations"
            diag_rep_dir = rep_dir / "diagnostics" / "K1"
            diag_rep_dir.mkdir(parents=True)
            (rep_dir / "K1" / "C3.npz").rename(diag_rep_dir / "C3.npz")
            manifest_path = rep_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest["operations"]["K1"]["C3"]
            entry.pop("filename")
            entry["developer_outputs"] = {"file": "diagnostics/K1/C3.npz"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            cfg["symm"]["developer_outputs"] = True
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cli.main(["symm", "--config", str(cfg_path)])

            self.assertFalse((out_dir / "C3_low_polar.npy").exists())
            self.assertFalse((out_dir / "C3_low_representation_raw.npy").exists())
            self.assertFalse((out_dir / "diagnostics").exists())
            _arrays, summary = _load_canonical_symmetry_payload(out_dir)
            row = summary["operations"][0]
            self.assertNotIn("developer_outputs", row)

            cfg["symm"]["developer_outputs"] = False
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            cli.main(["symm", "--config", str(cfg_path)])
            self.assertFalse((out_dir / "diagnostics" / "C3_low_polar.npy").exists())
            self.assertFalse((out_dir / "diagnostics" / "C3_low_representation_raw.npy").exists())

    def test_symm_rejects_legacy_projection_matrices_diagnostics_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            d = np.eye(4, dtype=np.complex128)
            self._run_minimal_two_layer_projection(
                tmp,
                valley="K1",
                operation="C3",
                d_up=d,
                manifest_entry={
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                },
                q_rotation_deg=0.0,
            )
            cfg_path = tmp / "C3_symm.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            cfg["symm"]["diagnostics"] = {"projection_matrices": True}
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "symm.diagnostics.projection_matrices is no longer supported"):
                cli.main(["symm", "--config", str(cfg_path)])

    def test_symm_fails_when_full_space_representation_does_not_covary_hamk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            q1_file = tmp / "q1.npy"
            q2_file = tmp / "q2.npy"
            hamk_file = tmp / "hamk.npy"
            symm_dir = tmp / "symmetry_analysis_test"
            rep_dir = symm_dir / "representations" / "K1"
            cfg_path = tmp / "mote2_4_K.yaml"

            np.save(q1_file, np.array([[0.0, 0.0]], dtype=float))
            np.save(q2_file, np.array([[0.0, 0.0]], dtype=float))
            hamk = np.zeros((1, 8, 8), dtype=np.complex128)
            hamk[0, :4, :4] = np.diag([1.0, 5.0, 2.0, 6.0])
            hamk[0, 4:, 4:] = np.diag([1.0, 5.0, 2.0, 6.0])
            np.save(hamk_file, hamk)

            d_swap = np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            )
            d_full = np.zeros((8, 8), dtype=np.complex128)
            d_full[:4, :4] = d_swap
            d_full[4:, 4:] = d_swap
            rep_dir.mkdir(parents=True)
            np.savez(rep_dir / "C3.npz", matrix=d_full)
            np.savez(rep_dir / "C3_rawH.npz", matrix=d_full)
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "operations": {
                            "K1": {
                                "C3": {
                                    "antiunitary": False,
                                    "filename": "K1/C3.npz",
                                    "raw_h_operator_file": "K1/C3_rawH.npz",
                                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                                    "sector_map": "identity",
                                    "k_pairs": [[0, 0]],
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            cfg = {
                "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
                "material": {
                    "hamk_file": str(hamk_file),
                    "energy_unit": "eV",
                    "qset1_file": str(q1_file),
                    "qset2_file": str(q2_file),
                    "spin": "up",
                    "num_layers": 2,
                    "num_orb_per_layer": [2, 2],
                },
                "plot": {"hamk_index": 0},
                "project": {
                    "enable": True,
                    "mode": "K1",
                    "downfold_method": "first_order",
                    "nlow_state_list": [[0], [0]],
                    "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
                },
                "symm": {
                    "enable": True,
                    "valley": "K1",
                    "spin": "up",
                    "tapw_symmetry_dir": str(symm_dir),
                    "operations": ["C3"],
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "full-space covariance residual"):
                cli.main(["symm", "--config", str(cfg_path)])

    def test_symm_uses_pg_file_for_unitary_raw_h_covariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            q1_file = tmp / "q1.npy"
            q2_file = tmp / "q2.npy"
            hamk_file = tmp / "hamk.npy"
            symm_dir = tmp / "symmetry_analysis_test"
            rep_dir = symm_dir / "representations" / "K1"
            out_dir = tmp / "outputs" / "K1" / "q06" / "symmetry"
            cfg_path = tmp / "mote2_4_K.yaml"

            np.save(q1_file, np.array([[0.0, 0.0]], dtype=float))
            np.save(q2_file, np.array([[0.0, 0.0]], dtype=float))
            hamk = np.zeros((1, 8, 8), dtype=np.complex128)
            hamk[0, :4, :4] = np.diag([1.0, 5.0, 2.0, 6.0])
            hamk[0, 4:, 4:] = np.diag([1.0, 5.0, 2.0, 6.0])
            np.save(hamk_file, hamk)

            d0_up = np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            )
            pg_up = d0_up.copy()
            rawh_up = d0_up @ pg_up
            d0_full = np.zeros((8, 8), dtype=np.complex128)
            pg_full = np.zeros((8, 8), dtype=np.complex128)
            rawh_full = np.zeros((8, 8), dtype=np.complex128)
            for matrix, block in ((d0_full, d0_up), (pg_full, pg_up), (rawh_full, rawh_up)):
                matrix[:4, :4] = block
                matrix[4:, 4:] = block
            rep_dir.mkdir(parents=True)
            np.savez(rep_dir / "C3.npz", matrix=d0_full)
            np.savez(rep_dir / "C3_PG.npz", matrix=pg_full)
            np.savez(rep_dir / "C3_rawH.npz", matrix=rawh_full)
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "matrices": [
                            {
                                "valley_label": "K1",
                                "operation": "C3",
                                "antiunitary": False,
                                "file": "K1/C3.npz",
                                "pg_file": "K1/C3_PG.npz",
                                "raw_h_operator_file": "K1/C3_rawH.npz",
                                "k_map": {"type": "rotation", "angle_deg": 120.0},
                                "q_map": {"type": "rotation", "angle_deg": 120.0},
                                "sector_map": "identity",
                                "k_pairs": [[0, 0]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cfg = {
                "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
                "material": {
                    "hamk_file": str(hamk_file),
                    "energy_unit": "eV",
                    "qset1_file": str(q1_file),
                    "qset2_file": str(q2_file),
                    "spin": "up",
                    "num_layers": 2,
                    "num_orb_per_layer": [2, 2],
                },
                "plot": {"hamk_index": 0},
                "project": {
                    "enable": True,
                    "mode": "K1",
                    "downfold_method": "first_order",
                    "nlow_state_list": [[0], [0]],
                    "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]],
                },
                "symm": {
                    "enable": True,
                    "valley": "K1",
                    "spin": "up",
                    "tapw_symmetry_dir": str(symm_dir),
                    "operations": ["C3"],
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            _write_projection_identity_artifacts(
                tmp / "outputs" / "K1" / "q06" / "projection",
                hamk_file=hamk_file,
                hamk=hamk,
                q1=np.load(q1_file),
                q2=np.load(q2_file),
                mode="K1",
                nlow_state_list=[[0], [0]],
                resolved_norb_fix_list=[[[[0, 1.0]]], [[[0, 1.0]]]],
            )

            cli.main(["symm", "--config", str(cfg_path)])

            arrays, summary = _load_canonical_symmetry_payload(out_dir)
            raw = arrays["C3z"]
            np.testing.assert_allclose(raw / raw[0, 0], np.eye(2), atol=1.0e-12)
            pair = summary["operations"][0]["pairs"][0]
            self.assertLess(pair["full_space_covariance_residual"], 1.0e-12)
            self.assertLess(pair["raw"]["heff_covariance_residual"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
