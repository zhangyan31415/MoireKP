import builtins
from pathlib import Path
from typing import Optional

import pytest
import yaml

from tapw.config import Config, PathConfig


def _write_relative_path_config(
    config_dir: Path,
    *,
    paths: Optional[dict] = None,
    compute: Optional[dict] = None,
) -> Path:
    for name in ("H.dat", "S.dat", "openmx.dat", "KPATH.in"):
        (config_dir / name).write_text("placeholder\n", encoding="utf-8")

    payload = {
        "twist": {"twist_index_m": 3},
        "paths": {
            "H_file": "H.dat",
            "S_file": "S.dat",
            "input_file": "openmx.dat",
            "output_dir": "results",
            "kpath_in": "KPATH.in",
            "kpath_out": "KPATH.out",
        },
        "compute": {
            "mode": "band",
            "n_g": 6,
        },
    }
    if paths:
        payload["paths"].update(paths)
    if compute:
        payload["compute"].update(compute)

    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_config_relative_paths_resolve_from_config_file_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    config_path = _write_relative_path_config(config_dir)
    monkeypatch.chdir(other_cwd)

    config = Config.from_yaml(str(config_path))

    assert Path(config.paths.H_file) == config_dir / "H.dat"
    assert Path(config.paths.S_file) == config_dir / "S.dat"
    assert Path(config.paths.input_file) == config_dir / "openmx.dat"
    assert Path(config.paths.kpath_in) == config_dir / "KPATH.in"
    assert Path(config.paths.kpath_out) == config_dir / "KPATH.out"
    assert Path(config.paths.output_dir) == config_dir / "results"


def test_config_resolves_canonical_output_layout_from_config_file_directory(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_relative_path_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["output_layout"] = {
        "style": "Canonical_V1",
        "root": "../outputs",
        "q_shell": "q06",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = Config.from_yaml(str(config_path))

    assert config.output_layout is not None
    assert config.output_layout.style == "canonical_v1"
    assert Path(config.output_layout.root) == config_dir.parent / "outputs"
    assert config.output_layout.q_shell == "q06"
    assert config.output_layout.profile is None


def test_canonical_config_may_omit_kpath_out_and_q_shell(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_relative_path_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload["paths"]["kpath_out"]
    payload["output_layout"] = {
        "style": "canonical_v1",
        "root": "../outputs",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = Config.from_yaml(str(config_path))

    assert config.paths.kpath_out is None
    assert config.output_layout is not None
    assert config.output_layout.q_shell is None


def test_release_sections_normalize_to_runtime_config(tmp_path):
    config_dir = tmp_path / "case" / "tapw" / "configs"
    config_dir.mkdir(parents=True)
    for relative in (
        "../../openmx/soc/H_symm.npz",
        "../../openmx/soc/S_symm.npz",
        "../../openmx/soc/openmx.dat_rigid",
        "../KPATH.in",
    ):
        path = config_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    config_path = config_dir / "K1_q06.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "case": {
                    "name": "K1_q06",
                    "output_root": "../outputs",
                },
                "twist": {
                    "bravais": "hex",
                    "twist_index_m": 8,
                    "twist_layer": [1, 2],
                    "spin": True,
                },
                "paths": {
                    "H_file": "../../openmx/soc/H_symm.npz",
                    "S_file": "../../openmx/soc/S_symm.npz",
                    "input_file": "../../openmx/soc/openmx.dat_rigid",
                    "kpath_in": "../KPATH.in",
                },
                "bands": {
                    "enable": True,
                    "valley": "K1",
                    "q_shell": 6,
                    "efermi": -4.1,
                    "num_processes": 61,
                    "num_bands": 100,
                    "save_hamiltonian": True,
                    "save_wavefunctions": False,
                },
                "symmetry": {
                    "enable": True,
                    "valley": "K1",
                    "q_shell": 6,
                    "tolerance": 2.0e-2,
                    "spglib_symprec": 5.0e-2,
                },
                "topology": {
                    "enable": False,
                    "valley": "K1",
                    "q_shell": 6,
                    "mesh": {
                        "n_b1": 31,
                        "n_b2": 41,
                        "range_b1": [0.0, 0.5],
                        "range_b2": [-0.5, 0.5],
                    },
                    "bandsets": {
                        "vbm2": {"sector": "valence", "indices": [-1, -2]},
                        "cbm2": {"sector": "conduction", "indices": [0, 1]},
                    },
                    "observables": {
                        "berry_curvature": ["vbm2"],
                        "quantum_geometry": ["vbm2"],
                        "wcc": [{"bands": "vbm2", "loop": "b2"}],
                    },
                },
                "field": {
                    "zero_potential_layers": [1],
                    "electric_field_eVpA": 0.1,
                    "inner_symmetric": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(str(config_path))

    assert config.twist.num_layers == 3
    assert Path(config.paths.output_dir) == (config_dir / "../outputs").resolve()
    assert config.output_layout is not None
    assert config.output_layout.style == "canonical_v1"
    assert Path(config.output_layout.root) == (config_dir / "../outputs").resolve()
    assert config.output_layout.q_shell == "q06"
    assert config.compute.mode == "band"
    assert config.compute.valleys == [1]
    assert config.compute.n_g == 6
    assert config.compute.efermi == -4.1
    assert config.compute.num_processes == 61
    assert config.compute.num_bands_cal == 100
    assert config.compute.hamk_save is True
    assert config.compute.eig_vec_cal is False
    assert config.compute.orthogonal_basis is False
    assert config.compute.zero_potential_layers == [1]
    assert config.compute.Electric_field_in_eVpA == 0.1
    assert config.compute.Inner_symmetrical_Electric_Field is True
    assert config.symmetry_analysis.enable is True
    assert config.symmetry_analysis.valleys == [1]
    assert config.symmetry_analysis.tolerance == 2.0e-2
    assert config.topology["bands"]["vbm2"]["indices"] == [-1, -2]
    assert config.topology["berry_curvature"] == ["vbm2"]
    assert config.topology["quantum_geometry"] == ["vbm2"]
    assert config.topology["wcc"] == [{"bands": "vbm2", "loop": "b2"}]

    config.apply_workflow_section("chern")
    assert config.compute.mode == "chern"
    assert config.compute.valleys == [1]
    assert config.compute.n_g == 6
    assert config.compute.chern_band_indices == [-1, -2]
    assert config.compute.band_type == "VBM"
    assert config.compute.get_chern_grid_shape() == (31, 41)
    assert config.output_layout.q_shell == "q06"


def test_release_section_without_s_file_defaults_to_orthogonal_basis(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    for name in ("H.dat", "openmx.dat", "KPATH.in"):
        (config_dir / name).write_text("placeholder\n", encoding="utf-8")
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "case": {"name": "Gamma_q04", "output_root": "outputs"},
                "twist": {"twist_index_m": 1, "twist_layer": [1, 1]},
                "paths": {
                    "H_file": "H.dat",
                    "input_file": "openmx.dat",
                    "kpath_in": "KPATH.in",
                },
                "bands": {
                    "enable": True,
                    "valley": "Gamma",
                    "q_shell": 4,
                    "efermi": 0.0,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(str(config_path))

    assert config.paths.S_file is None
    assert config.compute.orthogonal_basis is True
    assert config.compute.valleys == [5]
    assert config.compute.n_g == 4


def test_path_config_has_no_filesystem_or_stack_inspection_side_effects(tmp_path, monkeypatch):
    output_dir = tmp_path / "not_created"
    original_import = builtins.__import__

    def fail_on_inspect(name, *args, **kwargs):
        if name == "inspect":
            raise AssertionError("PathConfig must not import inspect")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_on_inspect)

    PathConfig(
        H_file=str(tmp_path / "H.dat"),
        S_file=None,
        input_file=str(tmp_path / "openmx.dat"),
        output_dir=str(output_dir),
        kpath_in=str(tmp_path / "KPATH.in"),
        kpath_out=str(tmp_path / "KPATH.out"),
    )

    assert not output_dir.exists()


def test_tapw_config_requires_explicit_ng_without_twist_angle_inference(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_relative_path_config(config_dir, compute={"TAPW": True})
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload["compute"]["n_g"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"TAPW.*compute\.n_g"):
        Config.from_yaml(str(config_path))


@pytest.mark.parametrize(
    "config_path",
    [
        "examples/tapw/mote2_9.43/configs/mote2_direct.yaml",
        "examples/tapw/mote2_9.43/configs/mote2_k_tapw.yaml",
        "examples/tapw/mgi2_9.43/configs/mgi2_direct.yaml",
        "examples/tapw/mgi2_9.43/configs/mgi2_m_gamma_tapw.yaml",
    ],
)
def test_release_tapw_example_paths_are_config_relative(config_path):
    root = Path(__file__).resolve().parents[2]
    config = Config.from_yaml(str(root / config_path))
    case_root = (root / config_path).parent.parent

    assert Path(config.paths.H_file).parent == case_root / "openmx" / "soc"
    assert Path(config.paths.S_file).parent == case_root / "openmx" / "soc"
    assert Path(config.paths.input_file).parent == case_root / "openmx" / "soc"
    assert Path(config.paths.kpath_in) == case_root / "KPATH.in"
    assert Path(config.paths.output_dir).parent == case_root / "runs"
