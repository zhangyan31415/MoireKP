import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_delta_g_five_classes_clean.py"
    spec = importlib.util.spec_from_file_location("plot_delta_g_five_classes_clean", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_filter_excluded_materials_removes_numbered_material():
    module = _load_module()
    records = [
        SimpleNamespace(material="Ga2Te2", raw_material="Ga2Te2"),
        SimpleNamespace(material="MoTe2", raw_material="MoTe2"),
    ]

    filtered = module.filter_excluded_materials(records)

    assert [record.material for record in filtered] == ["MoTe2"]


def test_filter_excluded_materials_uses_band_specific_material_lists():
    module = _load_module()
    records = [
        SimpleNamespace(
            class_name="AA Gamma VBM",
            material="BiTeCl TeCl",
            raw_material="BiTeCl_GSFE",
            stacking_key="TeCl_AA",
        ),
        SimpleNamespace(
            class_name="AA Gamma CBM",
            material="BiTeCl TeTe",
            raw_material="BiTeCl_GSFE",
            stacking_key="TeTe_AA",
        ),
        SimpleNamespace(
            class_name="AB Gamma CBM",
            material="BiTeI II",
            raw_material="BiTeI",
            stacking_key="II_AB",
        ),
        SimpleNamespace(
            class_name="AB Gamma VBM",
            material="ZnCl2",
            raw_material="ZnCl2_P3",
            stacking_key="AB",
        ),
        SimpleNamespace(
            class_name="AB Gamma VBM",
            material="MoS2",
            raw_material="MoS2",
            stacking_key="AB",
        ),
    ]

    filtered = module.filter_excluded_materials(records)

    assert [record.material for record in filtered] == ["BiTeCl TeTe", "MoS2"]


def test_two_row_layout_groups_aa_then_ab():
    module = _load_module()

    assert module.HEATMAP_TWO_ROW_LAYOUT == [
        ("AA", ("AA K VBM", "AA Gamma VBM", "AA Gamma CBM")),
        ("AB", ("AB Gamma VBM", "AB Gamma CBM")),
    ]


def test_two_row_heatmaps_use_tighter_spacing_and_larger_fonts():
    module = _load_module()

    assert not module.SHOW_FIGURE_TITLE
    assert module.HEATMAP_PANEL_GAP_IN == module.HEATMAP_ROW_GAP_IN
    assert module.HEATMAP_PANEL_GAP_IN == 0.27
    assert module.HEATMAP_ROW_GAP_IN == 0.27
    assert module.HEATMAP_FONT_SIZES["title"] >= 12.0
    assert module.HEATMAP_FONT_SIZES["xtick"] >= 8.0
    assert module.HEATMAP_FONT_SIZES["ytick"] >= 9.0
    assert module.HEATMAP_FONT_SIZES["ylabel"] >= 10.5
    assert module.HEATMAP_FONT_SIZES["row_label"] >= 13.0
    assert module.HEATMAP_FONT_SIZES["cbar_label"] >= 11.0
    assert module.HEATMAP_FONT_SIZES["cbar_tick"] >= 9.0
    assert module.HEATMAP_FONT_SIZES["suptitle"] >= 13.0
    assert module.HEATMAP_CBAR_GAP_IN == 0.22
    assert module.HEATMAP_CBAR_WIDTH_IN == 0.14
    assert module.HEATMAP_ROW_LABEL_OFFSET_IN == 0.62


def test_gridspec_space_fraction_preserves_absolute_gap():
    module = _load_module()
    total_inches = 10.0
    slots = 4
    gap_inches = module.HEATMAP_PANEL_GAP_IN

    fraction = module.gridspec_space_fraction(total_inches, slots, gap_inches)
    average_axis_inches = (total_inches - gap_inches * (slots - 1)) / slots

    assert abs(fraction * average_axis_inches - gap_inches) < 1e-12


def test_material_labels_use_chemical_subscripts():
    module = _load_module()

    assert module.format_material_tick_label("MoS2") == r"$\mathrm{MoS}_{2}$"
    assert module.format_material_tick_label("In2Se2") == r"$\mathrm{In}_{2}\mathrm{Se}_{2}$"
    assert module.format_material_tick_label("BiTeCl TeCl") == "BiTeCl TeCl"


def test_compact_panel_titles_omit_stacking_and_counts():
    module = _load_module()

    assert module.heatmap_panel_title("AA K VBM", 12, compact=True, show_count=False) == "K VB"
    assert "$K$" not in module.heatmap_panel_title("AA K VBM", 12, compact=True, show_count=False)
    assert module.heatmap_panel_title("AB Gamma VBM", 12, compact=True, show_count=False) == r"$\Gamma$ VB"
    assert module.heatmap_panel_title("AA Gamma CBM", 12, compact=True, show_count=False) == r"$\Gamma$ CB"
    assert module.heatmap_panel_title("AA K VBM", 12, compact=False, show_count=True) == "AA K VBM  (12)"


def test_aligned_top_row_hides_material_labels_when_bottom_panel_matches():
    module = _load_module()

    assert module.show_aligned_xticklabels(row_index=0, ab_class_name=None)
    assert not module.show_aligned_xticklabels(row_index=0, ab_class_name="AB Gamma VBM")
    assert not module.show_aligned_xticklabels(row_index=0, ab_class_name="AB Gamma CBM")
    assert module.show_aligned_xticklabels(row_index=1, ab_class_name="AB Gamma VBM")


def test_aligned_bottom_row_hides_panel_titles():
    module = _load_module()

    assert module.show_aligned_panel_title(row_index=0)
    assert not module.show_aligned_panel_title(row_index=1)


def test_heatmap_panel_hides_y_ticks_when_ylabel_is_hidden():
    module = _load_module()
    fig, ax = plt.subplots()
    try:
        module.draw_heatmap_panel(
            ax,
            {"name": "AA K VBM", "stacking": "AA"},
            ["MoS2"],
            [7.34],
            np.array([[1.0]]),
            [],
            Normalize(vmin=0.0, vmax=2.0),
            plt.get_cmap("viridis"),
            {},
            show_ylabel=False,
        )

        assert list(ax.get_yticks()) == []
        assert [label.get_text() for label in ax.get_yticklabels()] == []
    finally:
        plt.close(fig)


def test_plot_heatmap_overview_writes_one_row_and_two_row_outputs(tmp_path, monkeypatch):
    module = _load_module()
    config_root = tmp_path / "configs"
    config_root.mkdir()
    monkeypatch.setattr(module, "CONFIG_ROOT", config_root)

    class_specs = [
        {"name": "AA K VBM", "stacking": "AA"},
        {"name": "AA Gamma VBM", "stacking": "AA"},
        {"name": "AB Gamma VBM", "stacking": "AB"},
        {"name": "AA Gamma CBM", "stacking": "AA"},
        {"name": "AB Gamma CBM", "stacking": "AB"},
    ]
    base_module = SimpleNamespace(CLASS_SPECS=class_specs)
    records = []
    for spec_index, spec in enumerate(class_specs, start=1):
        material = f"Mat{spec_index}"
        for angle_index, angle in enumerate((7.34, 6.01), start=1):
            records.append(
                SimpleNamespace(
                    class_name=spec["name"],
                    material=material,
                    raw_material=material,
                    stacking=spec["stacking"],
                    angle_deg=angle,
                    delta_g=float(spec_index + angle_index),
                )
            )

    outputs = module.plot_heatmap_overview(base_module, records, tmp_path)

    assert {path.name for path in outputs} == {
        "delta_g_heatmap_overview_aligned_two_rows.png",
        "delta_g_heatmap_overview_aligned_two_rows.pdf",
        "delta_g_heatmap_overview_two_rows.png",
        "delta_g_heatmap_overview_two_rows.pdf",
        "delta_g_heatmap_overview_one_row.png",
        "delta_g_heatmap_overview_one_row.pdf",
    }


def test_build_heatmap_matrix_preserves_shared_material_columns():
    module = _load_module()
    records = [
        SimpleNamespace(material="MatA", angle_deg=7.34, delta_g=1.0),
        SimpleNamespace(material="MatC", angle_deg=7.34, delta_g=3.0),
    ]

    materials, angles, matrix = module.build_heatmap_matrix(records, materials=["MatA", "MatB", "MatC"])

    assert materials == ["MatA", "MatB", "MatC"]
    assert angles == [7.34]
    assert matrix.shape == (3, 1)
    assert matrix[0, 0] == 1.0
    assert np.isnan(matrix[1, 0])
    assert matrix[2, 0] == 3.0


def test_aligned_two_row_heatmap_skips_missing_ab_k_vb_panel(tmp_path, monkeypatch):
    module = _load_module()
    config_root = tmp_path / "configs"
    config_root.mkdir()
    monkeypatch.setattr(module, "CONFIG_ROOT", config_root)

    def fail_if_empty_panel_is_drawn(*args, **kwargs):
        raise AssertionError("missing AB K-VB should be left blank, not drawn as an empty subplot")

    monkeypatch.setattr(module, "draw_empty_heatmap_panel", fail_if_empty_panel_is_drawn)

    class_specs = [
        {"name": "AA K VBM", "stacking": "AA"},
        {"name": "AA Gamma VBM", "stacking": "AA"},
        {"name": "AB Gamma VBM", "stacking": "AB"},
        {"name": "AA Gamma CBM", "stacking": "AA"},
        {"name": "AB Gamma CBM", "stacking": "AB"},
    ]
    base_module = SimpleNamespace(CLASS_SPECS=class_specs)
    records = [
        SimpleNamespace(
            class_name=spec["name"],
            material=f"Mat{spec_index}",
            raw_material=f"Mat{spec_index}",
            stacking=spec["stacking"],
            angle_deg=7.34,
            delta_g=float(spec_index),
        )
        for spec_index, spec in enumerate(class_specs, start=1)
    ]

    outputs = module.plot_aligned_two_row_heatmap(base_module, records, tmp_path)

    assert {path.name for path in outputs} == {
        "delta_g_heatmap_overview_aligned_two_rows.png",
        "delta_g_heatmap_overview_aligned_two_rows.pdf",
    }


def test_load_config_topology_symbol_map_uses_c_and_z2(tmp_path):
    module = _load_module()
    config_root = tmp_path / "configs"
    config_root.mkdir()
    config = {
        "material": "11.WS2",
        "stackings": {
            "AA": {
                "category": "4.openmx_qk__mlff__cell_fix",
                "topo": {
                    "4.openmx_qk__mlff__cell_fix": {
                        "7.34": {
                            "VBM@K": {"[-1]": {"C": 1}},
                            "VBM@GAMMA": {"[-1,-2]": {"Z2": 1}},
                            "CBM@GAMMA": {"[0,1]": {"Z2": 0}},
                        },
                        "6.01": {
                            "VBM@GAMMA": {"[-1,-2]": {"Z2": None}},
                        },
                    }
                },
            }
        },
    }
    (config_root / "11.WS2.json").write_text(json.dumps(config), encoding="utf-8")

    symbol_map = module.load_config_topology_symbol_map(config_root)

    assert symbol_map[("K-VB", "WS2", "AA", 7.34)] == "circle"
    assert symbol_map[("G-VB", "WS2", "AA", 7.34)] == "square"
    assert symbol_map[("G-CB", "WS2", "AA", 7.34)] == "cross"
    assert ("G-VB", "WS2", "AA", 6.01) not in symbol_map
