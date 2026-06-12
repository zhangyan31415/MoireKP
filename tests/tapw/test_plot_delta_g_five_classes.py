import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_delta_g_five_classes.py"
    spec = importlib.util.spec_from_file_location("plot_delta_g_five_classes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_records_uses_stack_category_and_exact_qshell(tmp_path):
    module = _load_module()
    config_root = tmp_path / "configs"
    example_root = tmp_path / "examples"
    config_root.mkdir()

    config = {
        "material": "10.MoS2",
        "stackings": {
            "AA": {
                "category": "4.openmx_qk__mlff__cell_fix",
                "angles": ["10_3.15"],
                "topo": {
                    "4.openmx_qk__mlff__cell_fix": {
                        "3.15": {
                            "VBM@GAMMA": {
                                "[-1,-2]": {
                                    "WCC": {"qshell": 7, "direction": "ky", "spin": "soc"},
                                    "Z2": 1,
                                }
                            }
                        }
                    }
                },
            }
        },
    }
    (config_root / "10.MoS2.json").write_text(json.dumps(config), encoding="utf-8")

    valid = (
        example_root
        / "10.MoS2"
        / "AA"
        / "4.openmx_qk__mlff__cell_fix"
        / "10_3.15"
        / "tapw"
        / "Q_shell_7"
        / "topo"
        / "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt"
    )
    valid.parent.mkdir(parents=True)
    valid.write_text("delta_g 2.5\n", encoding="utf-8")

    wrong_category = (
        example_root
        / "10.MoS2"
        / "AA"
        / "999.wrong_category"
        / "10_3.15"
        / "tapw"
        / "Q_shell_7"
        / "topo"
        / "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt"
    )
    wrong_category.parent.mkdir(parents=True)
    wrong_category.write_text("delta_g 9.9\n", encoding="utf-8")

    wrong_qshell = (
        example_root
        / "10.MoS2"
        / "AA"
        / "4.openmx_qk__mlff__cell_fix"
        / "10_3.15"
        / "tapw"
        / "Q_shell_8"
        / "topo"
        / "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt"
    )
    wrong_qshell.parent.mkdir(parents=True)
    wrong_qshell.write_text("delta_g 8.8\n", encoding="utf-8")

    records = module.collect_records(config_root, example_root)

    assert len(records) == 1
    record = records[0]
    assert record.material == "MoS2"
    assert record.raw_material == "MoS2"
    assert record.class_name == "AA Gamma VBM"
    assert record.qshell == "7"
    assert record.stacking_key == "AA"
    assert abs(record.delta_g - 2.5) < 1e-12
    assert record.path == valid


def test_collect_records_preserves_symmetry_qshell_suffix(tmp_path):
    module = _load_module()
    config_root = tmp_path / "configs"
    example_root = tmp_path / "examples"
    config_root.mkdir()

    config = {
        "material": "46.ZnI2",
        "stackings": {
            "AA": {
                "category": "6.openmx_qk__gsfev2__cell_fix",
                "angles": ["13_2.45"],
                "topo": {
                    "6.openmx_qk__gsfev2__cell_fix": {
                        "2.45": {
                            "VBM@GAMMA": {
                                "[-1,-2]": {
                                    "WCC": {"qshell": "8_symm", "direction": "ky", "spin": "soc"},
                                    "Z2": None,
                                }
                            }
                        }
                    }
                },
            }
        },
    }
    (config_root / "46.ZnI2.json").write_text(json.dumps(config), encoding="utf-8")

    valid = (
        example_root
        / "46.ZnI2"
        / "AA"
        / "6.openmx_qk__gsfev2__cell_fix"
        / "13_2.45"
        / "tapw"
        / "Q_shell_8_symm"
        / "topo"
        / "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt"
    )
    valid.parent.mkdir(parents=True)
    valid.write_text("delta_g 1.25\n", encoding="utf-8")

    wrong = (
        example_root
        / "46.ZnI2"
        / "AA"
        / "6.openmx_qk__gsfev2__cell_fix"
        / "13_2.45"
        / "tapw"
        / "Q_shell_8"
        / "topo"
        / "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt"
    )
    wrong.parent.mkdir(parents=True)
    wrong.write_text("delta_g 9.99\n", encoding="utf-8")

    records = module.collect_records(config_root, example_root)

    assert len(records) == 1
    record = records[0]
    assert record.qshell == "8_symm"
    assert abs(record.delta_g - 1.25) < 1e-12
    assert record.path == valid


def test_collect_records_includes_suffix_stackings_with_display_label(tmp_path):
    module = _load_module()
    config_root = tmp_path / "configs"
    example_root = tmp_path / "examples"
    config_root.mkdir()

    config = {
        "material": "42.BiTeCl_GSFE",
        "stackings": {
            "ClCl_AA": {
                "category": "5.openmx_qk__mlff__cell_fix",
                "angles": ["4_7.34"],
                "topo": {
                    "5.openmx_qk__mlff__cell_fix": {
                        "7.34": {
                            "CBM@GAMMA": {
                                "[0,1]": {"WCC": {"qshell": 4, "direction": "ky", "spin": "soc"}}
                            }
                        }
                    }
                },
            }
        },
    }
    (config_root / "42.BiTeCl_GSFE.json").write_text(json.dumps(config), encoding="utf-8")

    valid = (
        example_root
        / "42.BiTeCl_GSFE"
        / "ClCl_AA"
        / "5.openmx_qk__mlff__cell_fix"
        / "4_7.34"
        / "tapw"
        / "Q_shell_4"
        / "topo"
        / "qgt_bands_0_1_CBM_Gamma_31_trace_condition.txt"
    )
    valid.parent.mkdir(parents=True)
    valid.write_text("delta_g 0.75\n", encoding="utf-8")

    records = module.collect_records(config_root, example_root)

    assert len(records) == 1
    record = records[0]
    assert record.class_name == "AA Gamma CBM"
    assert record.stacking == "AA"
    assert record.stacking_key == "ClCl_AA"
    assert record.material == "BiTeCl ClCl"
    assert record.raw_material == "BiTeCl_GSFE"
    assert abs(record.delta_g - 0.75) < 1e-12


def test_collect_records_skips_files_without_delta_g(tmp_path):
    module = _load_module()
    config_root = tmp_path / "configs"
    example_root = tmp_path / "examples"
    config_root.mkdir()

    config = {
        "material": "1.MoTe2",
        "stackings": {
            "AA": {
                "category": "9.openmx_qk__mlff__cell_relaxed",
                "angles": ["14_2.28"],
                "topo": {
                    "9.openmx_qk__mlff__cell_relaxed": {
                        "2.28": {
                            "VBM@K": {
                                "[-1]": {"WCC": {"qshell": 8, "direction": "ky", "spin": "soc"}}
                            }
                        }
                    }
                },
            }
        },
    }
    (config_root / "1.MoTe2.json").write_text(json.dumps(config), encoding="utf-8")

    trace = (
        example_root
        / "1.MoTe2"
        / "AA"
        / "9.openmx_qk__mlff__cell_relaxed"
        / "14_2.28"
        / "tapw"
        / "Q_shell_8"
        / "topo"
        / "qgt_band_-1_VBM_K1_31_trace_condition.txt"
    )
    trace.parent.mkdir(parents=True)
    trace.write_text("delta_tr 0.3\n", encoding="utf-8")

    records = module.collect_records(config_root, example_root)
    assert records == []
