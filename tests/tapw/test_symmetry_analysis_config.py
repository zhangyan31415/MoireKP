from pathlib import Path

import pytest
import yaml

from tapw.config import Config


def _write_config(tmp_path: Path, *, mode: str = "band", compute_valleys=None, symmetry_analysis=None) -> Path:
    compute_valleys = [5] if compute_valleys is None else list(compute_valleys)

    h_file = tmp_path / "H.dat"
    s_file = tmp_path / "S.dat"
    input_file = tmp_path / "openmx.dat"
    kpath_in = tmp_path / "KPATH.in"

    h_file.write_text("H\n", encoding="utf-8")
    s_file.write_text("S\n", encoding="utf-8")
    input_file.write_text("input\n", encoding="utf-8")
    kpath_in.write_text("1\n0 0 0 G\n0 0 0 G\n", encoding="utf-8")

    payload = {
        "twist": {
            "twist_index_m": 6,
        },
        "paths": {
            "H_file": str(h_file),
            "S_file": str(s_file),
            "input_file": str(input_file),
            "output_dir": str(tmp_path / "out"),
            "kpath_in": str(kpath_in),
            "kpath_out": str(tmp_path / "KPATH.out"),
        },
        "compute": {
            "mode": mode,
            "valleys": compute_valleys,
            "n_g": 6,
        },
    }
    if symmetry_analysis is not None:
        payload["symmetry_analysis"] = symmetry_analysis

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_c3_h_is_rejected_in_release_config(tmp_path):
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["compute"]["C3_H"] = True
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="C3_H.*symmetrize_hamiltonian"):
        Config.from_yaml(str(config_path))


def test_symmetrize_hamiltonian_accepts_bool_and_operation_list(tmp_path):
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["compute"]["symmetrize_hamiltonian"] = True
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = Config.from_yaml(str(config_path))
    assert config.compute.symmetrize_hamiltonian is True

    payload["compute"]["symmetrize_hamiltonian"] = ["C3z", "C2T"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = Config.from_yaml(str(config_path))
    assert config.compute.symmetrize_hamiltonian == ["C3z", "C2T"]


def test_symmetrize_hamiltonian_rejects_noncanonical_operations(tmp_path):
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["compute"]["symmetrize_hamiltonian"] = ["C3"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical.*C3z"):
        Config.from_yaml(str(config_path))


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


def test_old_config_without_symmetry_analysis_loads_default_section(tmp_path):
    config_path = _write_config(tmp_path)

    config = Config.from_yaml(str(config_path))

    assert config.compute.mode == "band"
    assert config.symmetry_analysis.enable is False
    assert config.symmetry_analysis.valleys is None
    assert config.symmetry_analysis.tolerance == 1.0e-2
    assert config.symmetry_analysis.output_dir == "symmetry_analysis"
    assert config.symmetry_analysis.debug is False


def test_new_config_with_symmetry_mode_loads_symmetry_section(tmp_path):
    config_path = _write_config(
        tmp_path,
        mode="symmetry",
        compute_valleys=[31],
        symmetry_analysis={
            "enable": True,
            "valleys": [5],
            "tolerance": 5.0e-3,
            "output_dir": "symm_out",
            "debug": True,
        },
    )

    config = Config.from_yaml(str(config_path))

    assert config.compute.mode == "symmetry"
    assert config.compute.valleys == [31]
    assert config.symmetry_analysis.enable is True
    assert config.symmetry_analysis.valleys == [5]
    assert config.symmetry_analysis.tolerance == 5.0e-3
    assert config.symmetry_analysis.output_dir == "symm_out"
    assert config.symmetry_analysis.debug is True


def test_symmetry_analysis_valleys_are_independent_from_compute_valleys(tmp_path):
    config_path = _write_config(
        tmp_path,
        mode="band",
        compute_valleys=[31, 32, 33],
        symmetry_analysis={
            "enable": True,
            "valleys": [5],
        },
    )

    config = Config.from_yaml(str(config_path))

    assert config.compute.valleys == [31, 32, 33]
    assert config.symmetry_analysis.valleys == [5]


def _write_config_without_s_file(tmp_path: Path, *, mode: str, orthogonal_basis: bool = False) -> Path:
    h_file = tmp_path / "H.dat"
    input_file = tmp_path / "openmx.dat"
    kpath_in = tmp_path / "KPATH.in"
    for file_path in (h_file, input_file, kpath_in):
        file_path.write_text("placeholder\n", encoding="utf-8")

    payload = {
        "twist": {"twist_index_m": 6},
        "paths": {
            "H_file": str(h_file),
            "input_file": str(input_file),
            "output_dir": str(tmp_path / "out"),
            "kpath_in": str(kpath_in),
            "kpath_out": str(tmp_path / "KPATH.out"),
        },
        "compute": {
            "mode": mode,
            "n_g": 6,
            "orthogonal_basis": orthogonal_basis,
        },
    }

    config_path = tmp_path / f"config_{mode}_{orthogonal_basis}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.parametrize("mode", ["band", "chern"])
def test_nonorthogonal_band_and_chern_require_s_file(tmp_path, mode):
    config_path = _write_config_without_s_file(tmp_path, mode=mode, orthogonal_basis=False)

    with pytest.raises(ValueError, match=r"S_file.*required.*non-orthogonal"):
        Config.from_yaml(str(config_path))


@pytest.mark.parametrize("mode", ["band", "chern"])
def test_orthogonal_band_and_chern_do_not_require_s_file(tmp_path, mode):
    config_path = _write_config_without_s_file(tmp_path, mode=mode, orthogonal_basis=True)

    config = Config.from_yaml(str(config_path))

    assert config.paths.S_file is None
    assert config.compute.orthogonal_basis is True


def test_symmetry_mode_does_not_require_s_file(tmp_path):
    config_path = _write_config_without_s_file(tmp_path, mode="symmetry", orthogonal_basis=False)

    config = Config.from_yaml(str(config_path))

    assert config.paths.S_file is None
    assert config.compute.mode == "symmetry"


def test_chern_mode_rejects_generalized_eigenvectors(tmp_path):
    config_path = _write_config(tmp_path, mode="chern")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["compute"]["ge"] = True
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Chern.*ge=true"):
        Config.from_yaml(str(config_path))
