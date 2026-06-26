from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kp.cli as cli
import kp.symmetry.projection as projection_mod
from kp.basis.selection import GaugeCandidateSymmetryMetrics
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
    _select_operation_matrix_kind,
    _sector_orbital_counts,
    _infer_symmetry_operations_from_manifest,
    _source_manifest_operation_name,
    _validate_operation_label,
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
        out_dir = tmp / f"{operation}_symm_project"
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
                "output_dir": str(out_dir),
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

            self.assertTrue((out_dir / "manifest.json").exists())

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

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
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

            row = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["operations"][0]
            self.assertEqual(row["matrix_file"], "exactified_C3z.npy")
            self.assertEqual(row["matrix_source"], "kp_symm_exactified_action")
            self.assertNotIn("representation_file", row)
            self.assertFalse((out_dir / "diagnostics" / "C3_low_representation_raw.npy").exists())

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

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
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
            out_dir = tmp / "symm_project"
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
                    "output_dir": str(out_dir),
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cli.main(["symm", "--config", str(cfg_path)])

            raw = np.load(out_dir / "C2T_low_raw.npy")
            c3_raw = np.load(out_dir / "C3_low_raw.npy")
            self.assertEqual(raw.shape, (2, 2))
            np.testing.assert_allclose(raw, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)
            np.testing.assert_allclose(c3_raw, np.eye(2), atol=1e-12)
            self.assertFalse((out_dir / "C2T_low_polar.npy").exists())
            self.assertFalse((out_dir / "C2T_low_representation_raw.npy").exists())

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame"], summary["frame"])
            self.assertEqual(summary["frame"]["q_transform"]["formula"], "q_model = R(rotation_deg) @ (layer_mean - q_source)")
            self.assertEqual(summary["frame"]["q_transform"]["rotation_deg"], 30.0)
            self.assertEqual(summary["q_model"]["files"]["layer1"], "q_model_layer1.npy")
            self.assertEqual(summary["q_model"]["files"]["layer2"], "q_model_layer2.npy")
            np.testing.assert_allclose(np.load(out_dir / "q_model_layer1.npy"), [[0.0, 0.0]], atol=1.0e-12)
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
            out_dir = tmp / "symm_project"
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
                    "output_dir": str(out_dir),
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cli.main(["symm", "--config", str(cfg_path)])

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest, summary)
            rows = {operation["operation"]: operation for operation in summary["operations"]}
            manifest_rows = {operation["operation"]: operation for operation in manifest["operations"]}
            c3 = rows["C3"]
            self.assertEqual(c3["declared_model_action"]["sector_map"], "identity")
            self.assertEqual(c3["model_action"]["sector_map"], "identity")
            self.assertEqual(c3["sector_map"], "identity")
            self.assertTrue(c3["model_basis_action"]["complete"])
            self.assertEqual(c3["model_basis_action"]["sector_map"], "identity")
            self.assertFalse(c3["model_basis_action"]["support_resolution"]["action_mismatch"])
            self.assertEqual(manifest_rows["C3"]["model_basis_action"], c3["model_basis_action"])

            row = rows["C2"]
            self.assertEqual(row["source_action"]["sector_map"], "layer_exchange")
            self.assertEqual(row["declared_model_action"]["sector_map"], "layer_exchange")
            self.assertEqual(row["model_action"]["sector_map"], "layer_exchange")
            self.assertEqual(row["sector_map"], "layer_exchange")
            self.assertTrue(row["model_basis_action"]["complete"])
            self.assertEqual(row["model_basis_action"]["sector_map"], "layer_exchange")
            self.assertEqual(manifest_rows["C2"]["model_basis_action"], row["model_basis_action"])
            self.assertFalse(row["model_basis_action"]["support_resolution"]["action_mismatch"])
            self.assertEqual(row["model_basis_action"]["support_resolution"]["block_off_support_rel"], 0.0)
            self.assertEqual(row["matrix_kind"], "continuum_internal_rep_exact")
            self.assertEqual(row["matrix_source"], "kp_symm_exactified_action")
            self.assertEqual(row["matrix_file"], "exactified_C2.npy")
            self.assertEqual(row["raw_matrix_file"], "C2_low_raw.npy")
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

            row = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["operations"][0]
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
            out_dir = tmp / "symm_project"
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
                    "output_dir": str(out_dir),
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cli.main(["symm", "--config", str(cfg_path)])

            raw = np.load(out_dir / "TR_low_raw.npy")
            np.testing.assert_allclose(raw, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1.0e-12)
            self.assertFalse((out_dir / "TR_low_polar.npy").exists())
            self.assertFalse((out_dir / "TR_low_representation_raw.npy").exists())
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            row = summary["operations"][0]
            self.assertEqual(row["spin_sector_sewing"], "up_to_down")
            self.assertEqual(row["source_spin"], "up")
            self.assertEqual(row["target_spin"], "down")
            self.assertEqual(row["matrix_kind"], "continuum_internal_rep_exact")
            self.assertEqual(row["matrix_source"], "kp_symm_exactified_action")
            self.assertEqual(row["matrix_file"], "exactified_TR.npy")
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
            self.assertTrue((out_dir / "diagnostics" / "C3_low_polar.npy").exists())
            self.assertTrue((out_dir / "diagnostics" / "C3_low_representation_raw.npy").exists())
            self.assertTrue((out_dir / "diagnostics" / "C3_low_representation_polar.npy").exists())
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            row = summary["operations"][0]
            self.assertEqual(row["developer_outputs"]["representation_matrix_file"], "diagnostics/C3_low_representation_raw.npy")
            self.assertEqual(row["developer_outputs"]["polar_matrix_file"], "diagnostics/C3_low_polar.npy")
            self.assertIn("polar", row["pairs"][0])

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
                    "output_dir": str(tmp / "symm_project"),
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
            out_dir = tmp / "symm_project"
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
                    "output_dir": str(out_dir),
                    "tolerance": 1.0e-8,
                },
            }
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cli.main(["symm", "--config", str(cfg_path)])

            raw = np.load(out_dir / "C3_low_raw.npy")
            np.testing.assert_allclose(raw, np.eye(2), atol=1.0e-12)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            pair = summary["operations"][0]["pairs"][0]
            self.assertLess(pair["full_space_covariance_residual"], 1.0e-12)
            self.assertLess(pair["raw"]["heff_covariance_residual"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
