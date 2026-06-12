import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "triangular_chernpost_batch.py"
    spec = importlib.util.spec_from_file_location("triangular_chernpost_batch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_tasks_uses_config_category_and_groups_topo_dirs(tmp_path):
    config_root = tmp_path / "configs"
    example_root = tmp_path / "examples"
    config_root.mkdir()

    config = {
        "material": "1.MoTe2",
        "stackings": {
            "AA": {
                "angles": ["3_9.43"],
                "category": "9.openmx_qk__mlff__cell_relaxed",
            },
            "AB": {
                "angles": ["4_7.34"],
                "category": "4.openmx_qk__mlff__cell_fix",
            },
            "IGNORED": {
                "angles": ["5_6.01"],
                "category": "5.openmx_st__mlff__cell_fix",
            },
        },
    }
    (config_root / "1.MoTe2.json").write_text(json.dumps(config), encoding="utf-8")

    aa_angle_root = (
        example_root
        / "1.MoTe2"
        / "AA"
        / "9.openmx_qk__mlff__cell_relaxed"
        / "3_9.43"
        / "tapw"
    )
    aa_angle_root.mkdir(parents=True)
    (aa_angle_root / "config_chern.yaml").write_text("compute: {}\n", encoding="utf-8")
    (aa_angle_root / "Q_shell_8" / "topo").mkdir(parents=True)
    (aa_angle_root / "Q_shell_9" / "topo").mkdir(parents=True)

    ab_angle_root = (
        example_root
        / "1.MoTe2"
        / "AB"
        / "4.openmx_qk__mlff__cell_fix"
        / "4_7.34"
        / "tapw"
    )
    ab_angle_root.mkdir(parents=True)
    (ab_angle_root / "config_chern.yaml").write_text("compute: {}\n", encoding="utf-8")
    (ab_angle_root / "Q_shell_4" / "topo").mkdir(parents=True)

    module = _load_module()
    tasks = module.discover_tasks(example_root=example_root, config_root=config_root)

    assert [(task.material, task.stacking, task.angle) for task in tasks] == [
        ("1.MoTe2", "AA", "3_9.43"),
        ("1.MoTe2", "AB", "4_7.34"),
    ]
    assert [path.name for path in tasks[0].topo_dirs] == ["topo", "topo"]
    assert len(tasks[0].topo_dirs) == 2
    assert "openmx_qk__mlff__cell_relaxed" in tasks[0].category
    assert "openmx_qk__mlff__cell_fix" in tasks[1].category


def test_discover_tasks_accepts_non_mlff_category_when_config_points_to_it(tmp_path):
    config_root = tmp_path / "configs"
    example_root = tmp_path / "examples"
    config_root.mkdir()

    config = {
        "material": "6.SnSe2",
        "stackings": {
            "AA": {
                "angles": ["3_9.43"],
                "category": "11.openmx_qk__gsfev2__cell_fix",
            }
        },
    }
    (config_root / "6.SnSe2.json").write_text(json.dumps(config), encoding="utf-8")

    tapw_dir = (
        example_root
        / "6.SnSe2"
        / "AA"
        / "11.openmx_qk__gsfev2__cell_fix"
        / "3_9.43"
        / "tapw"
    )
    tapw_dir.mkdir(parents=True)
    (tapw_dir / "config_chern.yaml").write_text("compute: {}\n", encoding="utf-8")
    (tapw_dir / "Q_shell_5_symm" / "topo").mkdir(parents=True)

    module = _load_module()
    tasks = module.discover_tasks(example_root=example_root, config_root=config_root)

    assert len(tasks) == 1
    assert tasks[0].material == "6.SnSe2"
    assert tasks[0].stacking == "AA"
    assert tasks[0].category == "11.openmx_qk__gsfev2__cell_fix"


def test_assign_partitions_applies_full_partition_list_and_render_omits_mem(tmp_path):
    config_root = tmp_path / "configs"
    example_root = tmp_path / "examples"
    config_root.mkdir()

    config = {
        "material": "10.MoS2",
        "stackings": {
            "AA": {
                "angles": ["3_9.43", "4_7.34", "5_6.01"],
                "category": "4.openmx_qk__mlff__cell_fix",
            }
        },
    }
    (config_root / "10.MoS2.json").write_text(json.dumps(config), encoding="utf-8")

    for angle in ["3_9.43", "4_7.34", "5_6.01"]:
        tapw_dir = example_root / "10.MoS2" / "AA" / "4.openmx_qk__mlff__cell_fix" / angle / "tapw"
        tapw_dir.mkdir(parents=True)
        (tapw_dir / "config_chern.yaml").write_text("compute: {}\n", encoding="utf-8")
        (tapw_dir / "Q_shell_4" / "topo").mkdir(parents=True)

    module = _load_module()
    tasks = module.discover_tasks(example_root=example_root, config_root=config_root)
    assigned = module.assign_partitions(tasks, ["vaspilot", "regular256", "regular128", "regular", "bigmem", "h20", "h200"])

    assert [task.partition for task in assigned] == [
        "vaspilot,regular256,regular128,regular,bigmem,h20,h200",
        "vaspilot,regular256,regular128,regular,bigmem,h20,h200",
        "vaspilot,regular256,regular128,regular,bigmem,h20,h200",
    ]
    sbatch_text = module.render_sbatch(
        assigned[0],
        output_root=tmp_path / "out",
        cpus=5,
        time_limit="08:00:00",
        chernpost_bin="/tmp/fake/tapw-chernpost",
    )
    assert "#SBATCH -p vaspilot,regular256,regular128,regular,bigmem,h20,h200" in sbatch_text
    assert "--mem" not in sbatch_text


def test_commands_use_cbm_zero_one_and_vbm_minus_one_minus_two():
    module = _load_module()
    assert module.COMMANDS == (
        ("CBM bands", ["--config", "../../config_chern.yaml", "-b", "0", "1", "-v", "5", "-bt", "CBM"]),
        ("VBM bands", ["--config", "../../config_chern.yaml", "-b", "-1", "-2", "-v", "5", "-bt", "VBM"]),
        ("CBM wcc", ["--config", "../../config_chern.yaml", "-wb", "0", "1", "-v", "5", "-bt", "CBM"]),
        ("VBM wcc", ["--config", "../../config_chern.yaml", "-wb", "-1", "-2", "-v", "5", "-bt", "VBM"]),
    )


def test_default_cpus_is_ten():
    module = _load_module()
    assert module.DEFAULT_CPUS == 10


def test_plan_topo_commands_skips_missing_vbm_and_keeps_cbm(tmp_path):
    module = _load_module()
    topo_dir = tmp_path / "topo"
    topo_dir.mkdir()
    npy = __import__("numpy")
    npy.save(topo_dir / "vec_CBM_Gamma_valley_2d_31.npy", npy.zeros((4, 8, 2), dtype=npy.complex128))

    commands, skipped = module.plan_topo_commands(topo_dir)

    assert [label for label, _ in commands] == ["CBM bands", "CBM wcc"]
    assert skipped == [
        "Skip VBM bands: missing vec_VBM_Gamma_valley*.npy",
        "Skip VBM wcc: missing vec_VBM_Gamma_valley*.npy",
    ]


def test_plan_topo_commands_skips_cbm_when_only_one_band(tmp_path):
    module = _load_module()
    topo_dir = tmp_path / "topo"
    topo_dir.mkdir()
    npy = __import__("numpy")
    npy.save(topo_dir / "vec_CBM_Gamma_valley_2d_31.npy", npy.zeros((4, 8, 1), dtype=npy.complex128))
    npy.save(topo_dir / "vec_VBM_Gamma_valley_2d_31.npy", npy.zeros((4, 8, 2), dtype=npy.complex128))

    commands, skipped = module.plan_topo_commands(topo_dir)

    assert [label for label, _ in commands] == ["VBM bands", "VBM wcc"]
    assert skipped == [
        "Skip CBM bands: only 1 Gamma bands available",
        "Skip CBM wcc: only 1 Gamma bands available",
    ]
