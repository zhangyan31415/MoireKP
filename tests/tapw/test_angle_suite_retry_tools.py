import csv
import importlib.util
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_ROOT / "examples" / "manual_ab_notapw_mote2_3_9.43" / "angle_suite" / "jobs"
PREPARE_RETRY_PATH = JOBS_DIR / "prepare_vaspilot_retry_safe.py"
SUMMARIZE_RESULTS_PATH = JOBS_DIR / "summarize_results.py"

MATRIX_FIELDS = [
    "platform",
    "partition",
    "login_host",
    "total_cores",
    "angle",
    "theta_deg",
    "twist_index_m",
    "efermi",
    "bands",
    "threads",
    "num_processes",
    "input_dir",
    "run_dir",
    "config_path",
    "run_log",
    "output_dir",
]

SUMMARY_FIELDS = MATRIX_FIELDS + [
    "status",
    "running_time_sec",
    "elapsed_wall_sec",
    "max_rss_kb",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_base_row(tmp_path: Path, *, angle: str, bands: str, threads: str, num_processes: str, status: str, tag: str) -> dict[str, str]:
    input_dir = tmp_path / "inputs" / angle
    input_dir.mkdir(parents=True, exist_ok=True)

    source_run_dir = tmp_path / "source_runs" / tag
    source_run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = source_run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = source_run_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(
            {
                "paths": {
                    "output_dir": str(output_dir),
                    "input_file": str(input_dir / "openmx.dat_rigid"),
                },
                "compute": {
                    "num_processes": int(num_processes),
                    "blas_threads": int(threads),
                    "parallel_impl": "joblib",
                    "parallel_backend": "loky",
                    "eigensolver": "slepc",
                    "eig_vec_cal": False,
                    "num_bands_cal": int(bands),
                },
            },
            f,
            sort_keys=False,
        )

    run_log = source_run_dir / "run.log"
    if status == "done":
        run_log.write_text("Running time: 12.5\nElapsed (wall clock) time: 0:12.5\nExit status: 0\n")
    else:
        run_log.write_text("Maximum resident set size (kbytes): 5000000\nExit status: 1\n")

    return {
        "platform": "vaspilot",
        "partition": "vaspilot",
        "login_host": "login001",
        "total_cores": "64",
        "angle": angle,
        "theta_deg": angle.split("_", 1)[1],
        "twist_index_m": angle.split("_", 1)[0],
        "efermi": "-4.100000",
        "bands": bands,
        "threads": threads,
        "num_processes": num_processes,
        "input_dir": str(input_dir),
        "run_dir": str(source_run_dir),
        "config_path": str(config_path),
        "run_log": str(run_log),
        "output_dir": str(output_dir),
        "status": status,
        "running_time_sec": "12.5" if status == "done" else "",
        "elapsed_wall_sec": "12.5" if status == "done" else "",
        "max_rss_kb": "5000000",
    }


def test_prepare_vaspilot_retry_safe_builds_retry_bundle_with_angle_caps(tmp_path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    rows = [
        _make_base_row(tmp_path, angle="4_7.34", bands="80", threads="2", num_processes="32", status="done", tag="a_done_32"),
        _make_base_row(tmp_path, angle="4_7.34", bands="80", threads="4", num_processes="16", status="done", tag="a_done_16"),
        _make_base_row(tmp_path, angle="4_7.34", bands="80", threads="1", num_processes="64", status="failed", tag="a_fail_64"),
        _make_base_row(tmp_path, angle="7_4.41", bands="40", threads="4", num_processes="16", status="done", tag="b_done_16"),
        _make_base_row(tmp_path, angle="7_4.41", bands="80", threads="4", num_processes="16", status="failed", tag="b_fail_16"),
        _make_base_row(tmp_path, angle="7_4.41", bands="80", threads="1", num_processes="64", status="failed", tag="b_fail_64"),
        _make_base_row(tmp_path, angle="7_4.41", bands="20", threads="2", num_processes="32", status="failed", tag="b_fail_32"),
    ]
    _write_csv(jobs_dir / "summary_live.csv", SUMMARY_FIELDS, rows)

    module = _load_module("angle_suite_prepare_retry", PREPARE_RETRY_PATH)
    bundle = module.prepare_retry_bundle(jobs_dir, bundle_name="safe_v1", fallback_num_processes=8)

    matrix_path = Path(bundle["matrix_path"])
    assert matrix_path.exists()

    retry_rows = list(csv.DictReader(matrix_path.open()))
    assert len(retry_rows) == 4
    by_tag = {Path(row["source_run_dir"]).name: row for row in retry_rows}

    assert by_tag["a_fail_64"]["num_processes"] == "32"
    assert by_tag["b_fail_64"]["num_processes"] == "8"
    assert by_tag["b_fail_32"]["num_processes"] == "8"
    assert by_tag["b_fail_16"]["num_processes"] == "8"
    assert all(row["platform"] == "vaspilot_retry_safe" for row in retry_rows)
    assert all("/jobs/retries/safe_v1/" in row["run_dir"] for row in retry_rows)

    retry_cfg = Path(by_tag["a_fail_64"]["config_path"])
    cfg = yaml.safe_load(retry_cfg.read_text())
    assert cfg["compute"]["num_processes"] == 32
    assert Path(cfg["paths"]["output_dir"]) == Path(by_tag["a_fail_64"]["output_dir"])

    meta = yaml.safe_load(Path(bundle["meta_path"]).read_text())
    assert meta["safe_num_processes_by_angle"]["4_7.34"] == 32
    assert meta["safe_num_processes_by_angle"]["7_4.41"] == 8

    array_script = Path(bundle["array_script_path"]).read_text()
    assert "run_matrix_task.sh vaspilot_retry_safe" in array_script


def test_summarize_results_skips_retry_matrices_by_default(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    base_vp = _make_base_row(tmp_path, angle="4_7.34", bands="80", threads="1", num_processes="64", status="done", tag="vp")
    base_bm = dict(base_vp)
    base_bm["platform"] = "bigmem"
    base_bm["partition"] = "bigmem"
    base_bm["login_host"] = "login002"
    base_bm["total_cores"] = "128"
    base_bm["run_dir"] = str(tmp_path / "source_runs" / "bm")
    Path(base_bm["run_dir"]).mkdir(parents=True, exist_ok=True)
    base_bm["config_path"] = str(Path(base_bm["run_dir"]) / "config.yaml")
    Path(base_bm["config_path"]).write_text(Path(base_vp["config_path"]).read_text())
    base_bm["run_log"] = str(Path(base_bm["run_dir"]) / "run.log")
    Path(base_bm["run_log"]).write_text("Running time: 10.0\nElapsed (wall clock) time: 0:10.0\nExit status: 0\n")
    base_bm["output_dir"] = str(Path(base_bm["run_dir"]) / "output")
    Path(base_bm["output_dir"]).mkdir(exist_ok=True)

    retry_row = dict(base_vp)
    retry_row["platform"] = "vaspilot_retry_safe"
    retry_row["run_dir"] = str(tmp_path / "source_runs" / "retry")
    Path(retry_row["run_dir"]).mkdir(parents=True, exist_ok=True)
    retry_row["config_path"] = str(Path(retry_row["run_dir"]) / "config.yaml")
    Path(retry_row["config_path"]).write_text(Path(base_vp["config_path"]).read_text())
    retry_row["run_log"] = str(Path(retry_row["run_dir"]) / "run.log")
    Path(retry_row["run_log"]).write_text("Running time: 9.0\nElapsed (wall clock) time: 0:09.0\nExit status: 0\n")
    retry_row["output_dir"] = str(Path(retry_row["run_dir"]) / "output")
    Path(retry_row["output_dir"]).mkdir(exist_ok=True)

    matrix_rows = [{key: row[key] for key in MATRIX_FIELDS} for row in (base_vp,)]
    _write_csv(jobs_dir / "matrix_vaspilot.csv", MATRIX_FIELDS, matrix_rows)
    _write_csv(jobs_dir / "matrix_bigmem.csv", MATRIX_FIELDS, [{key: base_bm[key] for key in MATRIX_FIELDS}])
    _write_csv(jobs_dir / "matrix_vaspilot_retry_safe.csv", MATRIX_FIELDS, [{key: retry_row[key] for key in MATRIX_FIELDS}])

    module = _load_module("angle_suite_summarize_results", SUMMARIZE_RESULTS_PATH)

    monkeypatch.setattr(sys, "argv", ["summarize_results.py", "--jobs-dir", str(jobs_dir)])
    module.main()
    rows = list(csv.DictReader((jobs_dir / "summary_live.csv").open()))
    assert len(rows) == 2
    assert {row["platform"] for row in rows} == {"vaspilot", "bigmem"}

    monkeypatch.setattr(sys, "argv", ["summarize_results.py", "--jobs-dir", str(jobs_dir), "--include-retry"])
    module.main()
    rows = list(csv.DictReader((jobs_dir / "summary_live.csv").open()))
    assert len(rows) == 3
    assert {row["platform"] for row in rows} == {"vaspilot", "bigmem", "vaspilot_retry_safe"}
