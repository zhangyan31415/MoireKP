import builtins
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml

from tapw.config import Config, PathConfig, SystemInputConfig
from tapw.io.kpath import KPathGenerator


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


def _write_release_path_config(config_dir: Path) -> Path:
    for name in ("H.dat", "S.dat", "openmx.dat"):
        (config_dir / name).write_text("placeholder\n", encoding="utf-8")

    payload = {
        "case": {"name": "K1_q06", "output_root": "results"},
        "twist": {"twist_index_m": 3, "twist_layer": [1, 1]},
        "paths": {
            "H_file": "H.dat",
            "S_file": "S.dat",
            "input_file": "openmx.dat",
        },
        "bands": {
            "valley": "K1",
            "q_shell": 6,
            "kpath": {
                "labels": ["G", "M", "K", "G"],
                "points_per_segment": 4,
                "coordinates": {
                    "G": [0.0, 0.0],
                    "M": [0.5, 0.0],
                    "K": [0.3333333333, 0.3333333333],
                },
            },
        },
    }
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _write_canonical_system_config(config_dir: Path) -> Path:
    for name in ("POSCAR", "H.npz", "S.npz"):
        (config_dir / name).write_text("placeholder\n", encoding="utf-8")
    payload = {
        "system": {
            "output": "results",
            "structure": "POSCAR",
            "hamiltonian": "H.npz",
            "overlap": "S.npz",
            "orbitals": {"Mo": "s3p2d1", "Te": "s3p2d2"},
            "twist_index": 6,
            "layers": [1, 2],
            "spin": True,
        },
        "bands": {
            "valley": "Gamma",
            "q_shell": 4,
            "kpath": {
                "labels": ["G", "M"],
                "points_per_segment": 2,
                "coordinates": {"G": [0.0, 0.0], "M": [0.5, 0.0]},
            },
        },
    }
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_canonical_system_normalizes_to_runtime_sections_relative_to_yaml(tmp_path, monkeypatch):
    config_dir = tmp_path / "case" / "tapw" / "configs"
    config_dir.mkdir(parents=True)
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    config_path = _write_canonical_system_config(config_dir)
    monkeypatch.chdir(other_cwd)

    config = Config.from_yaml(str(config_path))

    assert config.system_input.source_kind == "canonical_structure"
    assert Path(config.paths.input_file) == config_dir / "POSCAR"
    assert Path(config.paths.H_file) == config_dir / "H.npz"
    assert Path(config.paths.S_file) == config_dir / "S.npz"
    assert Path(config.paths.output_dir) == config_dir / "results"
    assert config.twist.twist_index_m == 6
    assert config.twist.twist_layer == [1, 2]
    assert config.twist.num_layers == 3
    assert config.twist.spin is True
    assert config.system_input.orbital_mapping == {"Mo": "s3p2d1", "Te": "s3p2d2"}


def test_canonical_system_input_is_frozen_deeply_immutable_and_typed(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config = Config.from_yaml(str(_write_canonical_system_config(config_dir)))

    assert isinstance(config.system_input, SystemInputConfig)
    assert config.system_input.source_kind == "canonical_structure"
    assert isinstance(config.system_input.structure, Path)
    assert isinstance(config.system_input.hamiltonian, Path)
    assert isinstance(config.system_input.overlap, Path)
    assert config.system_input.layers == (1, 2)
    assert config.system_input.orbitals == (("Mo", "s3p2d1"), ("Te", "s3p2d2"))
    assert config.system_input.explicit_bravais is None
    with pytest.raises(FrozenInstanceError):
        config.system_input.spin = False
    with pytest.raises(TypeError):
        config.system_input.orbitals[0] = ("Mo", "s1")


def test_canonical_system_overlap_is_optional_for_orthogonal_or_raw_h_workflows(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_canonical_system_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload["system"]["overlap"]
    del payload["bands"]
    payload["symmetry"] = {"valley": "Gamma", "q_shell": 4}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = Config.from_yaml(str(config_path))

    assert config.system_input.overlap is None
    assert config.paths.S_file is None
    assert config.compute.orthogonal_basis is True


def test_legacy_and_canonical_configs_share_one_typed_system_input(tmp_path):
    legacy_dir = tmp_path / "legacy"
    canonical_dir = tmp_path / "canonical"
    legacy_dir.mkdir()
    canonical_dir.mkdir()
    legacy = Config.from_yaml(str(_write_release_path_config(legacy_dir)))
    canonical = Config.from_yaml(str(_write_canonical_system_config(canonical_dir)))

    assert isinstance(legacy.system_input, SystemInputConfig)
    assert isinstance(canonical.system_input, SystemInputConfig)
    assert legacy.system_input.source_kind == "legacy_openmx"
    assert legacy.system_input.orbitals is None
    assert legacy.system_input.explicit_bravais is None
    assert canonical.system_input.source_kind == "canonical_structure"


@pytest.mark.parametrize("unknown_key", ["node", "python", "validation_dir"])
def test_canonical_system_rejects_unknown_fields(tmp_path, unknown_key):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_canonical_system_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["system"][unknown_key] = "private-value"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"system.*unknown.*{unknown_key}"):
        Config.from_yaml(str(config_path))


@pytest.mark.parametrize("legacy_section", ["case", "twist", "paths"])
def test_canonical_system_rejects_legacy_input_sections(tmp_path, legacy_section):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_canonical_system_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload[legacy_section] = {}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"system.*{legacy_section}"):
        Config.from_yaml(str(config_path))


def test_canonical_system_rejects_unknown_top_level_sections(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_canonical_system_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["campaign"] = {"enable": True}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown top-level.*campaign"):
        Config.from_yaml(str(config_path))


def test_canonical_system_save_yaml_round_trip_remains_release_facing(tmp_path):
    source_dir = tmp_path / "source"
    saved_dir = tmp_path / "saved"
    source_dir.mkdir()
    saved_dir.mkdir()
    config = Config.from_yaml(str(_write_canonical_system_config(source_dir)))
    saved_path = saved_dir / "config.yaml"

    config.save_yaml(str(saved_path))
    payload = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    reloaded = Config.from_yaml(str(saved_path))

    assert "system" in payload
    assert not ({"case", "twist", "paths", "compute", "output_layout"} & set(payload))
    assert not Path(payload["system"]["structure"]).is_absolute()
    assert reloaded.system_input.source_kind == "canonical_structure"
    assert reloaded.system_input.structure == config.system_input.structure
    assert reloaded.system_input.hamiltonian == config.system_input.hamiltonian
    assert reloaded.system_input.overlap == config.system_input.overlap
    assert reloaded.system_input.orbitals == config.system_input.orbitals
    assert reloaded.bands == config.bands


def test_save_yaml_preserves_cluster_top_level_symmetry_analysis_and_optional_overlap(tmp_path):
    source_dir = tmp_path / "source"
    saved_dir = tmp_path / "relocated" / "configs"
    source_dir.mkdir()
    saved_dir.mkdir(parents=True)
    config_path = _write_canonical_system_config(source_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["system"].pop("overlap")
    payload["cluster"] = {
        "layer_eps": 0.75,
        "layer_min_samples": 2,
        "sublayer_eps": 0.25,
        "sublayer_min_samples": 3,
        "atom_eps": 0.15,
        "atom_min_samples": 4,
        "period": 6.0,
        "k_max": 5,
    }
    payload["symmetry_analysis"] = {
        "enable": True,
        "valleys": [5],
        "tolerance": 0.004,
        "spglib_symprec": 0.008,
        "validation": "full",
        "output_dir": "certificates",
        "debug": True,
        "developer_outputs": True,
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = Config.from_yaml(str(config_path))
    saved_path = saved_dir / "config.yaml"

    config.save_yaml(str(saved_path))
    saved_payload = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    reloaded = Config.from_yaml(str(saved_path))

    assert "overlap" not in saved_payload["system"]
    assert not Path(saved_payload["system"]["structure"]).is_absolute()
    assert reloaded.system_input.overlap is None
    assert reloaded.system_input.structure == config.system_input.structure
    assert reloaded.cluster == config.cluster
    assert reloaded.symmetry_analysis == config.symmetry_analysis


def test_legacy_save_yaml_preserves_unset_explicit_bravais(tmp_path):
    source_dir = tmp_path / "source"
    saved_dir = tmp_path / "saved"
    source_dir.mkdir()
    saved_dir.mkdir()
    config = Config.from_yaml(str(_write_release_path_config(source_dir)))
    saved_path = saved_dir / "config.yaml"

    config.save_yaml(str(saved_path))
    payload = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    reloaded = Config.from_yaml(str(saved_path))

    assert "bravais" not in payload["twist"]
    assert reloaded.system_input.explicit_bravais is None


def test_legacy_release_save_yaml_round_trip_does_not_emit_removed_sections(tmp_path):
    source_dir = tmp_path / "source"
    saved_dir = tmp_path / "saved"
    source_dir.mkdir()
    saved_dir.mkdir()
    config = Config.from_yaml(str(_write_release_path_config(source_dir)))
    saved_path = saved_dir / "config.yaml"

    config.save_yaml(str(saved_path))
    payload = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
    reloaded = Config.from_yaml(str(saved_path))

    assert "compute" not in payload
    assert "output_layout" not in payload
    assert reloaded.twist.twist_index_m == config.twist.twist_index_m
    assert reloaded.paths.H_file == config.paths.H_file
    assert reloaded.paths.S_file == config.paths.S_file
    assert reloaded.paths.input_file == config.paths.input_file


def test_packaged_tapw_template_uses_canonical_system_contract():
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "tapw/tapw/templates/config.yaml").read_text(encoding="utf-8"))

    assert set(payload["system"]) == {
        "output",
        "structure",
        "hamiltonian",
        "overlap",
        "orbitals",
        "twist_index",
        "layers",
        "spin",
    }
    assert not ({"case", "twist", "paths"} & set(payload))
    assert payload["symmetry"]["num_bands"] == 100
    assert not (
        {"num_bands", "valence_count", "conduction_count", "output_dir", "cache"}
        & set(payload["symmetry"]["representation"])
    )
    Config.from_yaml(str(root / "tapw/tapw/templates/config.yaml"))
    from tapw.symm_rep import resolve_config_request

    with pytest.raises(FileNotFoundError, match="representations.npz"):
        resolve_config_request(root / "tapw/tapw/templates/config.yaml")


def test_top_level_compute_config_is_rejected_for_release_only_configs(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_relative_path_config(config_dir)

    with pytest.raises(ValueError, match="compute.*not supported"):
        Config.from_yaml(str(config_path))


def test_output_layout_config_is_rejected_for_release_only_configs(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_release_path_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["output_layout"] = {"style": "canonical_v1", "root": "../outputs"}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="output_layout.*not supported"):
        Config.from_yaml(str(config_path))


@pytest.mark.parametrize("forbidden_key", ["band_type", "eigensolver", "orthogonal_basis"])
def test_release_sections_reject_removed_user_parameters(tmp_path, forbidden_key):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_release_path_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["bands"][forbidden_key] = "scipy" if forbidden_key == "eigensolver" else "VBM"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=forbidden_key):
        Config.from_yaml(str(config_path))


def test_config_relative_paths_resolve_from_config_file_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    config_path = _write_release_path_config(config_dir)
    monkeypatch.chdir(other_cwd)

    config = Config.from_yaml(str(config_path))

    assert Path(config.paths.H_file) == config_dir / "H.dat"
    assert Path(config.paths.S_file) == config_dir / "S.dat"
    assert Path(config.paths.input_file) == config_dir / "openmx.dat"
    assert config.paths.kpath_in is None
    assert config.paths.kpath_out is None
    assert Path(config.paths.output_dir) == config_dir / "results"
    assert config.kpath["labels"] == ["G", "M", "K", "G"]
    assert config.kpath["points_per_segment"] == 4


def test_config_resolves_internal_canonical_output_layout_from_case_root(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_release_path_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["case"]["output_root"] = "../outputs"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = Config.from_yaml(str(config_path))

    assert config.output_layout is not None
    assert config.output_layout.style == "canonical_v1"
    assert Path(config.output_layout.root) == config_dir.parent / "outputs"
    assert config.output_layout.q_shell == "q06"
    assert config.output_layout.profile is None


def test_release_config_may_omit_kpath_out(tmp_path):
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    config_path = _write_release_path_config(config_dir)

    config = Config.from_yaml(str(config_path))

    assert config.paths.kpath_out is None
    assert config.output_layout is not None
    assert config.output_layout.q_shell == "q06"


def test_inline_kpath_generates_kpoints_without_kpath_input_file(tmp_path):
    generator = KPathGenerator(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    out = tmp_path / "outputs" / "K1" / "q06" / "KPATH.out"

    generator.generate_from_config(
        {
            "labels": ["G", "M", "K", "G"],
            "points_per_segment": 2,
            "coordinates": {
                "G": [0.0, 0.0],
                "M": [0.5, 0.0],
                "K": [1.0 / 3.0, 1.0 / 3.0],
            },
        },
        out,
    )

    assert out.is_file()
    assert len(generator.kpoints) == 7
    assert generator.segment_points == 2
    assert generator.labels_ticks == [r"$\Gamma$", "M", "K", r"$\Gamma$"]
    assert len(generator.x_ticks) == 4
    assert np.loadtxt(out).shape == (7, 4)


def test_release_sections_normalize_to_runtime_config(tmp_path):
    config_dir = tmp_path / "case" / "tapw" / "configs"
    config_dir.mkdir(parents=True)
    for relative in (
        "../../openmx/soc/H_symm.npz",
        "../../openmx/soc/S_symm.npz",
        "../../openmx/soc/openmx.dat_rigid",
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
                },
                "bands": {
                    "valley": "K1",
                    "q_shell": 6,
                    "efermi": -4.1,
                    "num_processes": 61,
                    "num_bands": 100,
                    "save_hamiltonian": True,
                    "save_wavefunctions": False,
                    "use_sparse_dot_mkl": True,
                    "kpath": {
                        "labels": ["G", "M", "K", "G"],
                        "points_per_segment": 40,
                        "coordinates": {
                            "G": [0.0, 0.0],
                            "M": [0.5, 0.0],
                            "K": [0.3333333333, 0.3333333333],
                        },
                    },
                },
                "symmetry": {
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
                    "bands": {
                        "vbm2": {"sector": "valence", "indices": [-1, -2]},
                        "cbm2": {"sector": "conduction", "indices": [0, 1]},
                    },
                    "berry_curvature": [{"bands": "vbm2"}],
                    "quantum_geometry": [{"bands": "vbm2"}],
                    "wcc": [{"bands": "vbm2", "loop": "b2"}],
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
    assert config.compute.use_sparse_dot_mkl is True
    assert config.compute.orthogonal_basis is False
    assert config.compute.zero_potential_layers == [1]
    assert config.compute.Electric_field_in_eVpA == 0.1
    assert config.compute.Inner_symmetrical_Electric_Field is True
    assert config.symmetry_analysis.enable is False
    assert config.symmetry_analysis.valleys == [1]
    assert config.symmetry_analysis.tolerance == 2.0e-2
    assert config.topology["bands"]["vbm2"]["indices"] == [-1, -2]
    assert config.topology["berry_curvature"] == [{"bands": "vbm2"}]
    assert config.topology["quantum_geometry"] == [{"bands": "vbm2"}]
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
    for name in ("H.dat", "openmx.dat"):
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
                },
                "bands": {
                    "valley": "Gamma",
                    "q_shell": 4,
                    "efermi": 0.0,
                    "kpath": {
                        "labels": ["G", "M"],
                        "points_per_segment": 2,
                        "coordinates": {"G": [0.0, 0.0], "M": [0.5, 0.0]},
                    },
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
    config_path = _write_release_path_config(config_dir)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload["bands"]["q_shell"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"bands.*q_shell"):
        Config.from_yaml(str(config_path))


@pytest.mark.parametrize(
    "config_path",
    [
        "examples/mgi2_3.89/tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml",
        "examples/mgi2_3.89/tapw/configs/mgi2_3.89_M1_spinful_q07.yaml",
        "examples/mote2_3.89/tapw/configs/mote2_3.89_K1_spinful_q06.yaml",
        "examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml",
        "examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml",
        "examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml",
        "examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml",
    ],
)
def test_release_tapw_example_paths_are_config_relative(config_path):
    root = Path(__file__).resolve().parents[2]
    source = root / config_path
    raw_system = yaml.safe_load(source.read_text(encoding="utf-8"))["system"]
    config = Config.from_yaml(str(source))

    for key in ("hamiltonian", "overlap", "structure", "output"):
        assert not Path(raw_system[key]).is_absolute()
    assert Path(config.paths.H_file) == (source.parent / raw_system["hamiltonian"]).resolve()
    assert Path(config.paths.S_file) == (source.parent / raw_system["overlap"]).resolve()
    assert Path(config.paths.input_file) == (source.parent / raw_system["structure"]).resolve()
    assert config.paths.kpath_in is None
    assert config.kpath["labels"] == ["G", "M", "K", "G"]
    assert Path(config.paths.output_dir) == (source.parent / raw_system["output"]).resolve()


@pytest.mark.parametrize(
    "config_path",
    [
        "examples/mgi2_3.89/tapw/configs/mgi2_3.89_Gamma_spinful_q04.yaml",
        "examples/mgi2_3.89/tapw/configs/mgi2_3.89_M1_spinful_q07.yaml",
        "examples/mote2_3.89/tapw/configs/mote2_3.89_K1_spinful_q06.yaml",
        "examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml",
        "examples/mote2_aab_5.09/tapw/configs/mote2_aab_5.09_K1_spinful_q04.yaml",
        "examples/ptse2_7.34/tapw/configs/ptse2_7.34_Gamma_spinful_q04.yaml",
        "examples/zrs2_3.15/tapw/configs/zrs2_3.15_Gamma_spinful_q04.yaml",
    ],
)
def test_tracked_canonical_tapw_configs_use_typed_system_input(config_path):
    root = Path(__file__).resolve().parents[2]

    config = Config.from_yaml(str(root / config_path))

    assert config.system_input.source_kind == "canonical_structure"
    assert config.system_input.structure == Path(config.paths.input_file)
    assert config.system_input.hamiltonian == Path(config.paths.H_file)
    assert config.system_input.overlap == Path(config.paths.S_file)
