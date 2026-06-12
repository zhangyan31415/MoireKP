import importlib.util
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_z2_trends.py"
    spec = importlib.util.spec_from_file_location("plot_z2_trends", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_z2_records_reads_only_explicit_z2_entries(tmp_path):
    module = _load_module()
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "10.MoS2.json"
    cfg.write_text(
        """
{
  "material": "10.MoS2",
  "stackings": {
    "AA": {
      "category": "4.openmx_qk__mlff__cell_fix",
      "topo": {
        "4.openmx_qk__mlff__cell_fix": {
          "3.15": {
            "VBM@GAMMA": {
              "[-1,-2]": { "Z2": 1 },
              "[-1]": { "WCC": { "qshell": 7, "spin": "soc", "direction": "ky" } }
            }
          }
        }
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    records = module.collect_z2_records(cfg_dir, ["10.MoS2"])
    assert len(records) == 1
    record = records[0]
    assert record.material == "10.MoS2"
    assert record.stacking == "AA"
    assert record.angle_deg == 3.15
    assert record.edge == "VBM"
    assert record.valley == "GAMMA"
    assert record.bandset == "[-1,-2]"
    assert record.z2 == 1


def test_parse_angle_deg_handles_folder_and_topo_notation():
    module = _load_module()
    assert module.parse_angle_deg("10_3.15") == 3.15
    assert module.parse_angle_deg("2.28") == 2.28
