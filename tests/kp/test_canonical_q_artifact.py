from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kp.model import pipeline as pipeline_module
from kp.symmetry import projection as projection_module
from kp.symmetry.q_canonicalization import canonicalize_q_geometry, q_geometry_hash


def _write_canonical_package(
    directory: Path,
    *,
    raw_q1: np.ndarray,
    raw_q2: np.ndarray,
    canonical_q1: np.ndarray,
    canonical_q2: np.ndarray,
) -> None:
    sector_order = ("L1", "L2")
    metadata = {
        "q_model": {
            "role": "raw_model_q",
            "production_role": "symmetry_canonical_q",
            "canonicalization": {
                "status": "certified",
                "sector_order": list(sector_order),
                "raw_q_hash": q_geometry_hash(
                    {"L1": raw_q1, "L2": raw_q2}, sector_order
                ),
                "canonical_q_hash": q_geometry_hash(
                    {"L1": canonical_q1, "L2": canonical_q2}, sector_order
                ),
            },
        }
    }
    np.savez(
        directory / "representations.npz",
        C2=np.eye(4, dtype=np.complex128),
        __q_model_raw_layer1__=raw_q1,
        __q_model_raw_layer2__=raw_q2,
        __q_model_canonical_layer1__=canonical_q1,
        __q_model_canonical_layer2__=canonical_q2,
        __metadata_json__=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_canonical_writer_packs_raw_and_production_q_arrays(tmp_path: Path) -> None:
    output_dir = tmp_path / "symmetry"
    output_dir.mkdir()
    np.save(output_dir / "exactified_C2.npy", np.eye(2, dtype=np.complex128))
    raw_q = np.array([[1.0, 0.0], [-1.0 + 1.0e-12, 0.0]], dtype=float)
    q_geometry = canonicalize_q_geometry(
        {"L1": raw_q, "L2": raw_q.copy()},
        [
            {
                "name": "C2",
                "q_map": {"type": "negation"},
                "model_basis_action": {
                    "complete": True,
                    "items": [
                        {
                            "source_sector": sector,
                            "target_sector": sector,
                            "source_q_index": source,
                            "target_q_index": target,
                        }
                        for sector in ("L1", "L2")
                        for source, target in ((0, 1), (1, 0))
                    ],
                },
            }
        ],
    )
    summary = {
        "valley": "Gamma",
        "spin": "up",
        "tolerance": 1.0e-8,
        "full_dim": 2,
        "low_dim": 2,
        "operations": [
            {
                "name": "C2",
                "operation": "C2",
                "antiunitary": False,
                "matrix_file": "exactified_C2.npy",
                "status": "exactified",
                "exactification_status": "exactified",
                "pairs": [],
            }
        ],
        "q_model": {
            "role": "raw_model_q",
            "production_role": "symmetry_canonical_q",
            "canonicalization": dict(q_geometry.artifact),
        },
    }

    projection_module._write_canonical_symmetry_outputs(
        output_dir,
        summary,
        q_geometry=q_geometry,
    )

    with np.load(output_dir / "representations.npz", allow_pickle=False) as package:
        np.testing.assert_array_equal(package["__q_model_raw_layer1__"], raw_q)
        np.testing.assert_array_equal(package["__q_model_raw_layer2__"], raw_q)
        np.testing.assert_allclose(
            package["__q_model_canonical_layer1__"][0],
            -package["__q_model_canonical_layer1__"][1],
            atol=1.0e-15,
        )
        np.testing.assert_array_equal(
            package["__q_model_canonical_layer1__"],
            package["__q_model_canonical_layer2__"],
        )
        metadata = json.loads(str(package["__metadata_json__"].item()))
    assert metadata["q_model"]["production_role"] == "symmetry_canonical_q"
    assert metadata["q_model"]["canonicalization"]["status"] == "certified"


def test_complete_v2_prefers_canonical_q_arrays_from_symmetry_package(tmp_path: Path) -> None:
    symmetry_dir = tmp_path / "symmetry"
    symmetry_dir.mkdir()
    raw_q1 = np.array([[1.0, 0.0], [-1.0 + 1.0e-12, 0.0]], dtype=float)
    raw_q2 = raw_q1.copy()
    canonical_q1 = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=float)
    canonical_q2 = canonical_q1.copy()
    _write_canonical_package(
        symmetry_dir,
        raw_q1=raw_q1,
        raw_q2=raw_q2,
        canonical_q1=canonical_q1,
        canonical_q2=canonical_q2,
    )
    config = SimpleNamespace(
        symmetry_source_config={"type": "kp_symm_output", "path": str(symmetry_dir)},
        response_semantics="complete_linear_v2",
        path=tmp_path / "model.yaml",
        qset1_file=tmp_path / "missing-q1.npy",
        qset2_file=tmp_path / "missing-q2.npy",
        rotation_deg=0.0,
    )

    q1, q2 = pipeline_module._load_model_q_sets(config)

    np.testing.assert_array_equal(q1, canonical_q1)
    np.testing.assert_array_equal(q2, canonical_q2)


def test_complete_v2_rejects_symmetry_package_without_canonical_q(tmp_path: Path) -> None:
    symmetry_dir = tmp_path / "symmetry"
    symmetry_dir.mkdir()
    np.savez(
        symmetry_dir / "representations.npz",
        C2=np.eye(2, dtype=np.complex128),
        __metadata_json__=np.asarray(json.dumps({"q_model": {}}, sort_keys=True)),
    )
    q1_path = tmp_path / "q1.npy"
    q2_path = tmp_path / "q2.npy"
    np.save(q1_path, np.zeros((1, 2), dtype=float))
    np.save(q2_path, np.zeros((1, 2), dtype=float))
    config = SimpleNamespace(
        symmetry_source_config={"type": "kp_symm_output", "path": str(symmetry_dir)},
        response_semantics="complete_linear_v2",
        path=tmp_path / "model.yaml",
        qset1_file=q1_path,
        qset2_file=q2_path,
        rotation_deg=0.0,
    )

    with pytest.raises(ValueError, match="rerun `kp symm`"):
        pipeline_module._load_model_q_sets(config)
