import importlib.util
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_delta_tr_trends.py"
    spec = importlib.util.spec_from_file_location("plot_delta_tr_trends", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_angle_and_delta_tr_from_trace_condition_files(tmp_path):
    module = _load_module()
    root = tmp_path / "1.MoTe2" / "AA" / "9.openmx_qk__mlff__cell_relaxed" / "14_2.28" / "tapw" / "Q_shell_8" / "topo"
    root.mkdir(parents=True)
    trace = root / "qgt_band_-1_VBM_K1_31_trace_condition.txt"
    trace.write_text(
        "\n".join(
            [
                "# Quantum-geometry trace-condition diagnostic",
                "mode single_band",
                "band_indices -1",
                "",
                "delta_tr 0.12345678",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = module.read_trace_condition(trace)
    assert record.material == "1.MoTe2"
    assert record.stacking == "AA"
    assert record.angle_label == "14_2.28"
    assert record.angle_deg == 2.28
    assert record.valley == "K1"
    assert abs(record.delta_tr - 0.12345678) < 1e-12


def test_collect_records_prefers_requested_valley(tmp_path):
    module = _load_module()
    base = tmp_path / "10.MoS2" / "AA" / "4.openmx_qk__mlff__cell_fix" / "3_9.43" / "tapw" / "Q_shell_4" / "topo"
    base.mkdir(parents=True)
    (base / "qgt_band_-1_VBM_Gamma_31_trace_condition.txt").write_text("delta_tr 0.5\n", encoding="utf-8")
    (base / "qgt_band_-1_VBM_K1_31_trace_condition.txt").write_text("delta_tr 0.2\n", encoding="utf-8")

    records = module.collect_records(tmp_path, ["10.MoS2"], preferred_valley="K1")

    assert len(records) == 1
    assert records[0].valley == "K1"
    assert abs(records[0].delta_tr - 0.2) < 1e-12
