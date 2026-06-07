from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kp.cli as cli
from kp.symmetry.project import (
    _model_action_metadata,
    _model_frame_action_metadata,
    _operation_entry,
    _validate_operation_label,
)


class SymmetryProjectionCliTests(unittest.TestCase):
    def test_symm_rejects_nonstandard_operation_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported symm operation"):
            _validate_operation_label("C4")

    def test_symm_reads_tapw_t_manifest_entry_for_standard_tr_request(self) -> None:
        manifest = {"operations": {"Gamma": {"T": {"filename": "Gamma/T.npz", "antiunitary": True}}}}

        entry = _operation_entry(manifest, "Gamma", "TR")

        self.assertEqual(entry["filename"], "Gamma/T.npz")

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

    def test_k_single_valley_c2t_model_action_is_internal_support(self) -> None:
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
        self.assertEqual(model["sector_map"], "layer_exchange")
        self.assertEqual(model["k_map"], {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True})
        self.assertEqual(model["q_map"], {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True})

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
            np.savez(rep_dir / "C3.npz", matrix=np.eye(8, dtype=np.complex128))
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "operations": {
                            "K1": {
                                "C3": {
                                    "antiunitary": False,
                                    "filename": "K1/C3.npz",
                                    "k_pairs": [[0, 0]],
                                },
                                "C2T": {
                                    "antiunitary": True,
                                    "filename": "K1/C2T.npz",
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
            polar = np.load(out_dir / "C2T_low_polar.npy")
            c3_raw = np.load(out_dir / "C3_low_raw.npy")
            self.assertEqual(raw.shape, (2, 2))
            np.testing.assert_allclose(raw, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)
            np.testing.assert_allclose(polar, raw, atol=1e-12)
            np.testing.assert_allclose(c3_raw, np.eye(2), atol=1e-12)

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
            self.assertEqual(by_name["C2T"]["model_action"]["k_map"]["axis_deg"], 180.0)
            self.assertEqual(by_name["C2T"]["model_action"]["sector_map"], "layer_exchange")
            self.assertTrue(by_name["C2T"]["k_map"]["in_model_frame"])
            self.assertLess(by_name["C3"]["pairs"][0]["raw"]["heff_covariance_residual"], 1.0e-12)
            self.assertLess(by_name["C2T"]["pairs"][0]["raw"]["heff_covariance_residual"], 1.0e-12)
            self.assertLess(by_name["C2T"]["pairs"][0]["raw"]["subspace_leakage"], 1.0e-12)

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
            np.savez(rep_dir / "T_rawH.npz", matrix=d_full)
            np.savez(rep_dir / "TR.npz", matrix=d_full)
            np.savez(rep_dir / "T_PG.npz", matrix=np.eye(8, dtype=np.complex128))
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "matrices": [
                            {
                                "valley_label": "M1",
                                "operation": "TR",
                                "antiunitary": True,
                                "file": "M1/TR.npz",
                                "pg_file": "M1/T_PG.npz",
                                "raw_h_operator_file": "M1/T_rawH.npz",
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
            polar = np.load(out_dir / "TR_low_polar.npy")
            np.testing.assert_allclose(raw, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1.0e-12)
            np.testing.assert_allclose(polar, raw, atol=1.0e-12)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            row = summary["operations"][0]
            self.assertEqual(row["spin_sector_sewing"], "up_to_down")
            self.assertEqual(row["source_spin"], "up")
            self.assertEqual(row["target_spin"], "down")
            self.assertLess(row["pairs"][0]["raw"]["heff_covariance_residual"], 1.0e-12)

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
            (symm_dir / "representations" / "manifest.json").write_text(
                json.dumps(
                    {
                        "operations": {
                            "K1": {
                                "C3": {
                                    "antiunitary": False,
                                    "filename": "K1/C3.npz",
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
