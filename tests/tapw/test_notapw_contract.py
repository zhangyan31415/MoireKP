from pathlib import Path

import pytest
import yaml

from tapw.config import Config
from tapw.workflows import band as band_workflow


def _write_notapw_config(tmp_path: Path, *, ge=True, eig_vec_cal=False) -> Path:
    h_file = tmp_path / "H.dat"
    s_file = tmp_path / "S.dat"
    input_file = tmp_path / "openmx.dat"
    kpath_in = tmp_path / "KPATH.in"
    for file_path in (h_file, s_file, input_file, kpath_in):
        file_path.write_text("placeholder\n", encoding="utf-8")

    payload = {
        "twist": {"twist_index_m": 3},
        "paths": {
            "H_file": str(h_file),
            "S_file": str(s_file),
            "input_file": str(input_file),
            "output_dir": str(tmp_path / "out"),
            "kpath_in": str(kpath_in),
            "kpath_out": str(tmp_path / "KPATH.out"),
        },
        "compute": {
            "mode": "band",
            "TAPW": False,
            "ge": ge,
            "eig_vec_cal": eig_vec_cal,
            "eigensolver": "slepc",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_notapw_requires_generalized_eigenproblem(tmp_path):
    config_path = _write_notapw_config(tmp_path, ge=False)

    with pytest.raises(ValueError, match="Top-level compute is not supported"):
        Config.from_yaml(str(config_path))


def test_notapw_rejects_wavefunction_output(tmp_path):
    config_path = _write_notapw_config(tmp_path, eig_vec_cal=True)

    with pytest.raises(ValueError, match="Top-level compute is not supported"):
        Config.from_yaml(str(config_path))


def test_notapw_legacy_compute_config_is_not_release_facing(tmp_path):
    with pytest.raises(ValueError, match="Top-level compute is not supported"):
        Config.from_yaml(str(_write_notapw_config(tmp_path)))


def test_notapw_band_output_names_do_not_use_valley_label():
    helper = getattr(band_workflow, "band_output_filename", None)
    assert helper is not None

    assert helper("VBM", valley_flag="K1", suffix="", tapw=True) == "band_VBM_K1_valley.txt"
    assert helper("VBM", valley_flag="K1", suffix="", tapw=False) == "band_VBM.txt"
    assert helper("CBM", valley_flag="M1", suffix="_2d", tapw=False) == "band_CBM_2d.txt"
