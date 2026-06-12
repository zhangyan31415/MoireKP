import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "triangular_topo_explicit_submit.py"
    spec = importlib.util.spec_from_file_location("triangular_topo_explicit_submit", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_angle_folder_for_topo_key_matches_fractional_angle_tokens():
    module = _load_module()
    angles = ["3_9.43", "10_3.15", "14_2.28"]
    assert module.angle_folder_for_topo_key("9.43", angles) == "3_9.43"
    assert module.angle_folder_for_topo_key("3.15", angles) == "10_3.15"
    assert module.angle_folder_for_topo_key("2.28", angles) == "14_2.28"


def test_expand_topo_specs_emits_both_band_and_wcc_commands(tmp_path):
    module = _load_module()
    config = {
        "material": "1.MoTe2",
        "stackings": {
            "AA": {
                "angles": ["14_2.28"],
                "category": "9.openmx_qk__mlff__cell_relaxed",
                "qshell_map": {
                    "14_2.28": {
                        "VBM": {"K1": 8, "K2": 8, "Gamma": 8},
                        "CBM": {"K1": 20},
                    }
                },
                "topo": {
                    "9.openmx_qk__mlff__cell_relaxed": {
                        "2.28": {
                            "VBM@K": {
                                "[-1]": {
                                    "WCC": {"direction": "ky", "qshell": 8, "spin": "soc"}
                                }
                            },
                            "VBM@GAMMA": {
                                "[-1,-2]": {
                                    "WCC": {"direction": "ky", "qshell": 8, "spin": "soc"}
                                }
                            },
                        }
                    }
                },
            }
        },
    }
    specs = module.expand_topo_specs_for_stacking("1.MoTe2", "AA", config["stackings"]["AA"])

    summary = [(spec.angle_folder, spec.edge, spec.valley_cli, tuple(spec.bandset), spec.qshell, spec.command_kind) for spec in specs]
    assert ("14_2.28", "VBM", 1, (-1,), "8", "band") in summary
    assert ("14_2.28", "VBM", 1, (-1,), "8", "wcc") in summary
    assert ("14_2.28", "VBM", 5, (-1, -2), "8", "band") in summary
    assert ("14_2.28", "VBM", 5, (-1, -2), "8", "wcc") in summary


def test_render_sbatch_passes_explicit_args_after_double_dash():
    module = _load_module()
    spec = module.TopoSpec(
        material="1.MoTe2",
        stacking="AA",
        category="9.openmx_qk__mlff__cell_relaxed",
        angle_folder="14_2.28",
        angle_key="2.28",
        edge="VBM",
        valley_key="K",
        valley_cli=1,
        bandset=(-1,),
        qshell="8",
        command_kind="band",
    )
    task = module.Task(
        material="1.MoTe2",
        stacking="AA",
        category="9.openmx_qk__mlff__cell_relaxed",
        angle_folder="14_2.28",
        tapw_dir=Path("/tmp/fake/tapw"),
        specs=(spec,),
    )
    helper = Path("/tmp/fake/run_topo_explicit.py")
    text = module.render_sbatch(task, helper)
    assert ' -- -b -1 -v 1 -bt VBM' in text
