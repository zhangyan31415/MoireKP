from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kp.cli as cli
from kp.model.symmetry import load_symmetry_source
from kp.symmetry.projection import (
    _build_action_representation,
    _gamma_c2_action_audit,
    _model_action_metadata,
    _model_frame_action_metadata,
    _operation_entry,
    _pairs_from_entry,
    _resolve_projected_model_action,
    _select_operation_matrix_kind,
    _validate_operation_label,
)


class SymmetryProjectionCliTests(unittest.TestCase):
    def test_kp_symm_help_lists_developer_outputs(self) -> None:
        stream = io.StringIO()
        with self.assertRaises(SystemExit) as exc, redirect_stdout(stream):
            cli.main(["symm", "--help"])

        self.assertEqual(exc.exception.code, 0)
        self.assertIn("--developer-outputs", stream.getvalue())

    def test_action_resolution_infers_sector_orbitals_from_low_dim_for_single_nlow_list(self) -> None:
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

        resolved, basis_action = _resolve_projected_model_action(
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

        self.assertEqual(resolved["sector_map"], "layer_exchange")
        self.assertTrue(basis_action["complete"])
        self.assertTrue(basis_action["support_resolution"]["action_mismatch"])
        self.assertEqual(basis_action["support_resolution"]["support_matrix_source"], "representation")

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
                "norb_fix_list": norb_fix_list,
            },
            "symm": {
                "enable": True,
                "valley": valley,
                "spin": "up",
                "tapw_symmetry_dir": str(symm_dir),
                "operations": [operation],
                "output_dir": str(out_dir),
                "tolerance": 1.0e-8,
            },
        }
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        cli.main(["symm", "--config", str(cfg_path)])
        return out_dir

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

    def test_gamma_c2_action_audit_report_exists(self) -> None:
        declared_model_action = {
            "antiunitary": False,
            "k_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
            "q_map": {"type": "reflection", "axis_deg": 0.0, "in_model_frame": True},
            "sector_map": "identity",
        }
        audit = _gamma_c2_action_audit(
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
