import importlib.util
import sys
from pathlib import Path

from matplotlib.colors import to_hex


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_delta_tr_subspace_trends.py"
    spec = importlib.util.spec_from_file_location("plot_delta_tr_subspace_trends", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_angle_and_delta_tr_subspace_from_trace_condition_files(tmp_path):
    module = _load_module()
    root = tmp_path / "22.HfS2" / "AA" / "4.openmx_qk__mlff__cell_fix" / "14_2.28" / "tapw" / "Q_shell_8" / "topo"
    root.mkdir(parents=True)
    trace = root / "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt"
    trace.write_text(
        "\n".join(
            [
                "# Quantum-geometry trace-condition diagnostic",
                "mode subspace",
                "band_indices -1,-2",
                "",
                "delta_tr_subspace 12.34567890",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = module.read_trace_condition(trace)
    assert record.material == "22.HfS2"
    assert record.stacking == "AA"
    assert record.angle_label == "14_2.28"
    assert record.angle_deg == 2.28
    assert record.valley == "Gamma"
    assert abs(record.delta_tr_value - 12.34567890) < 1e-12


def test_collect_records_filters_to_requested_valley(tmp_path):
    module = _load_module()
    base = tmp_path / "13.ZrS2" / "AB" / "5.openmx_qk__mlff__cell_fix" / "3_9.43" / "tapw" / "Q_shell_4" / "topo"
    base.mkdir(parents=True)
    (base / "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt").write_text("delta_tr_subspace 3.0\n", encoding="utf-8")
    (base / "qgt_bands_-1_-2_VBM_K1_31_trace_condition.txt").write_text("delta_tr_subspace 1.0\n", encoding="utf-8")

    records = module.collect_records(
        tmp_path,
        ["13.ZrS2"],
        pattern="qgt_bands_-1_-2_VBM_*_trace_condition.txt",
        valley="Gamma",
    )

    assert len(records) == 1
    assert records[0].valley == "Gamma"
    assert abs(records[0].delta_tr_value - 3.0) < 1e-12


def test_resolve_group_config_distinguishes_gamma_and_k_defaults():
    module = _load_module()

    gamma = module.resolve_group_config("gamma")
    k_group = module.resolve_group_config("k")

    assert gamma["valley"] == "Gamma"
    assert gamma["pattern"] == "qgt_bands_-1_-2_VBM_Gamma_31_trace_condition.txt"
    assert "22.HfS2" in gamma["materials"]
    assert k_group["valley"] == "K1"
    assert k_group["pattern"] == "qgt_band_-1_VBM_K1_31_trace_condition.txt"
    assert "1.MoTe2" in k_group["materials"]


def test_material_style_map_avoids_repeated_colors_for_many_materials():
    module = _load_module()
    materials = [f"{index}.Mat" for index in range(8)]

    style_map = module.build_material_style_map(materials)

    assert list(style_map) == materials
    colors = [to_hex(style_map[material]["color"]) for material in materials]
    assert len(set(colors)) == len(materials)
    assert len({style_map[material]["marker"] for material in materials}) > 1
    assert len({style_map[material]["linestyle"] for material in materials}) > 1
