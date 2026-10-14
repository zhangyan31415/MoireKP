from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

import kp.cli as cli
import kp.basis.selection as selection
from test_cli_project_memory import _canonical_case_cfg


def _configuration(tmp_path: Path, workers: int = 1) -> Path:
    config = {
        "material": {"hamk_file": "hamk.npy", "qset1_file": "q1.npy",
            "qset2_file": "q2.npy", "spin": "up", "energy_unit": "eV",
            "num_layers": 2, "num_orb_per_layer": [1], "efermi": 0.0},
        "plot": {"mode": "K1", "hamk_index": 0, "target": "all", "out": "plot.png"},
        "project": {"mode": "K1", "workers": workers, "downfold_method": "first_order",
            "out_dir": "project", "nlow_state_list": [[0], [0]],
            "norb_fix_list": [[[[0, 1.0]]], [[[0, 1.0]]]], "e_ref_values": [1.0]},
    }
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(_canonical_case_cfg(config, tmp_path=tmp_path, nk=2)))
    return path


def _project_result(*args, **kwargs):
    return (np.eye(2, dtype=complex), np.array([0.0, 1.0]), np.eye(2, dtype=complex),
        SimpleNamespace(hermiticity_residual=0.0, pole_distance_min_mev=None))


def _broken_parallel(monkeypatch):
    joblib = ModuleType("joblib")

    class BrokenParallel:
        def __init__(self, **kwargs):
            pass

        def __call__(self, tasks):
            raise AttributeError("parallel worker implementation failed")

    joblib.Parallel = BrokenParallel
    joblib.delayed = lambda func: lambda *args, **kwargs: (func, args, kwargs)
    monkeypatch.setitem(sys.modules, "joblib", joblib)


def test_qrcp_failure_does_not_select_different_anchors(monkeypatch):
    def fail_qr(*args, **kwargs):
        raise AttributeError("QRCP implementation failed")

    monkeypatch.setattr(selection.scipy.linalg, "qr", fail_qr)
    with pytest.raises(AttributeError, match="QRCP implementation failed"):
        selection.select_anchor_rows_qrcp(np.eye(2), n_anchors=1)


def test_invalid_config_is_not_classified_as_an_old_config(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("material: [\n")
    with pytest.raises(yaml.YAMLError):
        cli._config_path_uses_canonical_case(path)


@pytest.mark.parametrize("command", [cli._cmd_project_from_config_impl, cli.cmd_sweep_from_config])
def test_parallel_calculation_failure_is_not_rerun_serially(monkeypatch, tmp_path, command):
    path = _configuration(tmp_path, workers=2)
    _broken_parallel(monkeypatch)
    serial_calls = []

    def record_serial(*args, **kwargs):
        serial_calls.append(True)
        return _project_result()

    monkeypatch.setattr(cli, "project_heff_full", record_serial)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *args, **kwargs: None)
    with pytest.raises(AttributeError, match="parallel worker implementation failed"):
        command(str(path))
    assert not serial_calls


def test_bad_numeric_stack_is_not_coerced_to_object_arrays(monkeypatch, tmp_path):
    path = _configuration(tmp_path)
    monkeypatch.setattr(cli, "project_heff_full", _project_result)

    def fail_stack(*args, **kwargs):
        raise AttributeError("numeric stack implementation failed")

    monkeypatch.setattr(cli.np, "stack", fail_stack)
    with pytest.raises(AttributeError, match="numeric stack implementation failed"):
        cli._cmd_project_from_config_impl(str(path))


def test_inspection_analysis_failure_is_not_reported_as_success(monkeypatch, tmp_path):
    path = _configuration(tmp_path)
    monkeypatch.setattr(cli, "save_spectrum_txt", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "plot_eigs_scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_write_inspect_sidecars", lambda *args, **kwargs: None)

    def fail_analysis(*args, **kwargs):
        raise AttributeError("inspection analysis implementation failed")

    monkeypatch.setattr(cli.np, "vstack", fail_analysis)
    with pytest.raises(AttributeError, match="inspection analysis implementation failed"):
        cli.cmd_plot_from_config(str(path))


def test_invalid_authored_kpath_is_not_replaced_by_an_index_axis(tmp_path):
    config = {"kpath": {"tmat": np.eye(3).tolist(), "labels": "G-M",
        "coordinates": {"G": [0, 0], "M": [0.5, 0]}, "points_per_segment": 2}}
    with pytest.raises(ValueError, match="kpath.labels"):
        cli._kpath_axis_from_config(config, cfg_dir=str(tmp_path),
            project_indices=[0, 1], row_count=2)
