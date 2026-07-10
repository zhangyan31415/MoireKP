from pathlib import Path

import pytest
import yaml

from tapw.config import Config


def _base_paths(tmp_path: Path, *, include_s: bool = True) -> dict[str, str]:
    h_file = tmp_path / "H.dat"
    s_file = tmp_path / "S.dat"
    input_file = tmp_path / "openmx.dat"
    kpath_in = tmp_path / "KPATH.in"
    h_file.write_text("H\n", encoding="utf-8")
    s_file.write_text("S\n", encoding="utf-8")
    input_file.write_text("input\n", encoding="utf-8")
    kpath_in.write_text("1\n0 0 0 G\n0 0 0 G\n", encoding="utf-8")
    paths = {
        "H_file": str(h_file),
        "input_file": str(input_file),
        "kpath_in": str(kpath_in),
    }
    if include_s:
        paths["S_file"] = str(s_file)
    return paths


def _release_payload(tmp_path: Path, *, workflow: str = "bands", include_s: bool = True) -> dict:
    payload = {
        "case": {"output_root": str(tmp_path / "outputs")},
        "twist": {"twist_index_m": 6, "twist_layer": [1, 1]},
        "paths": _base_paths(tmp_path, include_s=include_s),
        "bands": {"enable": False, "valley": "K1", "q_shell": 6},
        "symmetry": {"enable": False, "valley": "K1", "q_shell": 6},
        "topology": {
            "enable": False,
            "valley": "Gamma",
            "q_shell": 3,
            "mesh": {"n_b1": 5, "n_b2": 5, "range_b1": [-0.5, 0.5], "range_b2": [-0.5, 0.5]},
            "bands": {"vbm2": {"sector": "valence", "indices": [-1, -2]}},
            "berry_curvature": [{"bands": "vbm2"}],
            "quantum_geometry": [{"bands": "vbm2"}],
            "wcc": [{"bands": "vbm2", "loop": "b2"}],
        },
    }
    payload[workflow]["enable"] = True
    return payload


def _write_payload(tmp_path: Path, payload: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_top_level_compute_is_rejected_in_release_config(tmp_path):
    payload = _release_payload(tmp_path)
    payload["compute"] = {"mode": "band", "n_g": 6}
    config_path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="Top-level compute"):
        Config.from_yaml(str(config_path))


def test_release_band_config_loads_default_symmetry_section(tmp_path):
    config_path = _write_payload(tmp_path, _release_payload(tmp_path, workflow="bands"))

    config = Config.from_yaml(str(config_path))

    assert config.compute.mode == "band"
    assert config.compute.n_g == 6
    assert config.compute.valleys == [1]
    assert config.symmetry_analysis.enable is False
    assert config.symmetry_analysis.valleys == [1]
    assert config.symmetry_analysis.tolerance == 1.0e-2
    assert config.symmetry_analysis.validation == "full"
    assert config.symmetry_analysis.debug is False
    assert config.symmetry_analysis.developer_outputs is False
    assert config.compute.use_sparse_dot_mkl is True
    assert config.output_layout.style == "canonical_v1"


def test_release_symmetry_config_defaults_to_strict_validation(tmp_path):
    config_path = _write_payload(tmp_path, _release_payload(tmp_path, workflow="symmetry"))

    config = Config.from_yaml(str(config_path))

    assert config.compute.mode == "symmetry"
    assert config.compute.use_sparse_dot_mkl is True
    assert config.symmetry_analysis.validation == "full"
    assert config.symmetry_analysis.debug is False
    assert config.symmetry_analysis.developer_outputs is False


def test_release_symmetry_config_loads_symmetry_section(tmp_path):
    payload = _release_payload(tmp_path, workflow="symmetry")
    payload["symmetry"].update({"tolerance": 5.0e-3, "validation": "full", "debug": True})
    config_path = _write_payload(tmp_path, payload)

    config = Config.from_yaml(str(config_path))

    assert config.compute.mode == "symmetry"
    assert config.compute.n_g == 6
    assert config.compute.valleys == [1]
    assert config.symmetry_analysis.enable is True
    assert config.symmetry_analysis.valleys == [1]
    assert config.symmetry_analysis.tolerance == 5.0e-3
    assert config.symmetry_analysis.validation == "full"
    assert config.symmetry_analysis.debug is True


@pytest.mark.parametrize("validation", ["gamma", "export_only", "raw-h-only", "none"])
def test_release_symmetry_validation_rejects_non_strict_modes(tmp_path, validation):
    payload = _release_payload(tmp_path, workflow="symmetry")
    payload["symmetry"]["validation"] = validation
    config_path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="symmetry.validation"):
        Config.from_yaml(str(config_path))


def test_release_symmetry_validation_rejects_unknown_mode(tmp_path):
    payload = _release_payload(tmp_path, workflow="symmetry")
    payload["symmetry"]["validation"] = "quickish"
    config_path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="symmetry.validation"):
        Config.from_yaml(str(config_path))


def test_symmetry_section_valley_is_independent_from_band_valley(tmp_path):
    payload = _release_payload(tmp_path, workflow="bands")
    payload["symmetry"]["enable"] = True
    payload["symmetry"]["valley"] = "Gamma"
    payload["symmetry"]["q_shell"] = 4
    config_path = _write_payload(tmp_path, payload)

    config = Config.from_yaml(str(config_path))

    assert config.compute.mode == "band"
    assert config.compute.valleys == [1]
    assert config.symmetry_analysis.enable is True
    assert config.symmetry_analysis.valleys == [5]


def test_missing_s_file_defaults_to_orthogonal_band_basis(tmp_path):
    config_path = _write_payload(tmp_path, _release_payload(tmp_path, workflow="bands", include_s=False))

    config = Config.from_yaml(str(config_path))

    assert config.paths.S_file is None
    assert config.compute.orthogonal_basis is True


def test_s_file_selects_nonorthogonal_band_basis(tmp_path):
    config_path = _write_payload(tmp_path, _release_payload(tmp_path, workflow="bands", include_s=True))

    config = Config.from_yaml(str(config_path))

    assert config.paths.S_file is not None
    assert config.compute.orthogonal_basis is False


def test_symmetry_mode_does_not_require_s_file(tmp_path):
    config_path = _write_payload(tmp_path, _release_payload(tmp_path, workflow="symmetry", include_s=False))

    config = Config.from_yaml(str(config_path))

    assert config.paths.S_file is None
    assert config.compute.mode == "symmetry"


def test_topology_config_sets_chern_mode_and_mesh(tmp_path):
    config_path = _write_payload(tmp_path, _release_payload(tmp_path, workflow="topology"))

    config = Config.from_yaml(str(config_path))

    assert config.compute.mode == "chern"
    assert config.compute.n_g == 3
    assert config.compute.get_chern_grid_shape() == (5, 5)
    assert config.topology["bands"]["vbm2"]["indices"] == [-1, -2]


def test_release_tapw_example_yamls_do_not_use_removed_symmetrization_keys():
    root = Path(__file__).resolve().parents[2]
    yaml_paths = list((root / "examples" / "tapw").glob("**/*.yaml"))
    yaml_paths.append(root / "tapw" / "tapw" / "templates" / "config.yaml")

    offenders = []
    for path in yaml_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        compute = payload.get("compute", {})
        if "C3_H" in compute or "M_valley_D3_H" in compute:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
