#!/usr/bin/env python3
"""Run several physical-export benchmarks sequentially and collect results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-old-evaluation", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("NAME", "CONFIG", "OLD_EXPORT"),
        required=True,
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name, config, old_export in args.case:
        case_output = output / name
        started = time.perf_counter()
        command = [
            sys.executable,
            str(Path(__file__).resolve().with_name("benchmark_physical_export.py")),
            config,
            str(case_output),
            "--old-export",
            old_export,
            "--comparison-tolerance",
            "1e-9",
            "--repeat",
            "1",
        ]
        if args.skip_old_evaluation:
            command.append("--skip-old-evaluation")
        print(f"SUITE_CASE_START {name}", flush=True)
        completed = subprocess.run(command, check=False)
        benchmark_path = case_output / "benchmark.json"
        record: dict[str, object] = {
            "name": name,
            "config": str(Path(config).resolve()),
            "old_export": str(Path(old_export).resolve()),
            "wall_seconds": float(time.perf_counter() - started),
            "returncode": int(completed.returncode),
        }
        if benchmark_path.is_file():
            record["benchmark"] = json.loads(
                benchmark_path.read_text(encoding="utf-8")
            )
        records.append(record)
        (output / "suite.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"SUITE_CASE_DONE {name} returncode={completed.returncode} "
            f"wall_seconds={record['wall_seconds']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
