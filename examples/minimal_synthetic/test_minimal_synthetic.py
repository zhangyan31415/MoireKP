from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for src_dir in (REPO_ROOT / "kp", REPO_ROOT / "tapw"):
    sys.path.insert(0, str(src_dir))

from kp.analysis import compute_orbital_weights, select_orbit_set
from kp.io import load_orbital_order
from kp.kmesh import KPathGenerator


ROOT = Path(__file__).resolve().parent


def _read_csv_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return np.asarray(
            [[float(row["band0"]), float(row["band1"])] for row in reader],
            dtype=float,
        )


def test_minimal_synthetic_files_match_expected_outputs(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert config["runtime_class"] == "clean-clone"

    orbital_order = load_orbital_order(str(ROOT / config["inputs"]["orbital_order"]))
    eigvecs = _read_csv_matrix(ROOT / config["inputs"]["eigvec_coefficients"])
    expected_selection = json.loads(
        (ROOT / config["expected_outputs"]["orbital_selection"]).read_text(
            encoding="utf-8"
        )
    )

    weights = compute_orbital_weights(
        eigvecs,
        np.zeros((0, 2), dtype=float),
        orbital_order,
        config["parameters"]["band_indices"],
        score=config["parameters"]["score"],
    )
    selection = select_orbit_set(weights, {"top_k": config["parameters"]["top_k"]})

    assert selection["indices"] == expected_selection["indices"]
    np.testing.assert_allclose(selection["weights"], expected_selection["weights"])

    generated_kpath = tmp_path / "KPATH.out"
    KPathGenerator().read_and_generate_kpath(
        str(ROOT / config["inputs"]["kpath"]),
        str(generated_kpath),
    )
    expected_kpath = np.loadtxt(ROOT / config["expected_outputs"]["kpath"])
    np.testing.assert_allclose(np.loadtxt(generated_kpath), expected_kpath)
