#!/usr/bin/env python3
"""Run the six published KP recipes against their real cached TAPW inputs.

This verification script writes only beneath --run-root. It keeps each published
recipe's physical settings and runs the installed package, without source imports.
Cached TAPW arrays are inputs: this is a full KP replay, not regeneration of DFT
or TAPW H(k). Existing published exports are comparison oracles only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


CASES = {
    "mote2_k": ("mote2_3.89", "kp_K.yaml", "K1", "q06", "examples/mote2_3.89/kp/outputs_selected_20261001/K1/q06"),
    "aab_ka": ("mote2_aab_5.09", "kp_KA.yaml", "K1", "q04", "examples/mote2_aab_5.09/kp/outputs_selected_20260929/k1_a/K1/q04"),
    "aab_kb": ("mote2_aab_5.09", "kp_KB.yaml", "K1", "q04", "examples/mote2_aab_5.09/kp/outputs_selected_20260929/k1_b/K1/q04"),
    "aab_gamma": ("mote2_aab_5.09", "kp_Gamma.yaml", "Gamma", "q04", "examples/mote2_aab_5.09/kp/outputs_selected_20260929/gamma_spinful/Gamma/q04"),
    "zrs2_gamma": ("zrs2_3.89", "kp_Gamma.yaml", "Gamma", "q04", "examples/zrs2_3.89/kp/outputs_selected_20261006/gamma_spinful/Gamma/q04"),
    "zrs2_m": ("zrs2_3.89", "kp_M.yaml", "M1", "q07", "paper/moirekp/code_paper/review/evidence/selected_models_release_20260928/zrs2_m/M1/q07"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def describe(path, *, hash_content=True):
    path = Path(path).resolve(strict=True)
    result = {"path": str(path), "bytes": path.stat().st_size,
              "mtime_ns": path.stat().st_mtime_ns}
    if hash_content:
        result["sha256"] = sha(path)
    return result


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(actual, expected):
    import numpy as np
    if actual.shape != expected.shape:
        return {"status": "different_shape", "actual_shape": list(actual.shape),
                "baseline_shape": list(expected.shape)}
    delta = np.asarray(actual) - np.asarray(expected)
    return {"status": "compared", "max_abs_ev": float(np.max(np.abs(delta))),
            "rms_ev": float(np.sqrt(np.mean(np.abs(delta) ** 2)))}


def input_identity(parent, valley, shell):
    import numpy as np
    channel = parent / valley / shell
    files = {}
    for name in ("band/hamiltonian_k.npy", "band/kpoints.npy",
                 "band/g_vectors_group1.npy", "band/g_vectors_group2.npy",
                 "symmetry/representations.npz"):
        files[name] = describe(channel / name, hash_content=name != "band/hamiltonian_k.npy")
    with np.load(channel / "symmetry/representations.npz", allow_pickle=False) as packet:
        metadata = json.loads(str(np.asarray(packet["metadata_json"]).item()))
    files["source_packet_metadata"] = metadata
    files["raw_h_hash_policy"] = "Source-packet identity metadata and file stat retained; giant H(k) is not reread solely for hashing."
    h = np.load(channel / "band/hamiltonian_k.npy", mmap_mode="r")
    files["hamiltonian_shape"] = list(h.shape)
    if h.shape[0] != 61:
        raise ValueError(f"Expected the published 61-point input, got {h.shape}")
    del h
    return files


def validate(output, baseline, name):
    import numpy as np
    required = ("projection/heff.npy", "projection/basis.npz",
                "projection/selection_artifact.json", "symmetry/representations.npz",
                "model/model_data.npz", "model/eigvals.npy", "model/evaluate.py",
                "model/model.toml", "model/terms.csv", "model/symmetry.toml",
                "model/physical_model.py", "model/q_points.csv")
    artifacts = {f: describe(output / f) for f in required}
    heff = np.load(output / "projection/heff.npy", mmap_mode="r")
    saved = np.load(output / "model/eigvals.npy")
    if len(heff) != 61 or len(saved) != 61:
        raise ValueError("Full published-path replay must produce 61 Heff and model rows")
    module = imported(output / "model/evaluate.py", "published_replay_" + name)
    evaluator = module.load_model(output / "model")
    points = np.asarray(evaluator.data["reference_kpoints"], dtype=float)
    if points.shape != (61, 2):
        raise ValueError(f"Expected 61 Cartesian model reference momenta, got {points.shape}")
    evaluated = evaluator.bands(points)
    self_check = metrics(evaluated, saved)
    if self_check["max_abs_ev"] > 1e-8:
        raise ValueError(f"Standalone evaluator differs from saved model bands: {self_check}")
    hermiticity_max = max(float(np.max(np.abs((h := evaluator.hamiltonian(k)) - h.conj().T))) for k in points)
    if hermiticity_max > 1e-8:
        raise ValueError(f"Exported Hamiltonian is not Hermitian: {hermiticity_max}")
    physical = imported(output / "model/physical_model.py", "published_physical_" + name).load_model(output / "model")
    matrix_errors = []
    for k in points:
        matrix_errors.append(metrics(physical.hamiltonian(k), evaluator.hamiltonian(k)))
    physical_max = max(row["max_abs_ev"] for row in matrix_errors)
    if physical_max > 1e-8:
        raise ValueError(f"Independent physical text export differs from frozen evaluator: {physical_max}")
    selection = json.loads((output / "projection/selection_artifact.json").read_text())
    result = {"artifacts": artifacts, "dimension": int(evaluator.dim), "kpoints": 61,
              "heff_shape": list(heff.shape), "standalone_vs_saved_bands": self_check,
              "export_hermiticity_max_ev": hermiticity_max,
              "physical_text_vs_frozen_matrix": {"points": 61, "max_abs_ev": physical_max,
                                                 "per_point": matrix_errors},
              "selection_status": selection.get("certification_status"),
              "selection_metrics": selection.get("metrics"),
              "baseline": {"path": str(baseline), "comparison_scope": "Historical published oracle; spectral comparisons are gauge invariant. Matrix comparisons are diagnostics and require the same basis for interpretation."}}
    if result["selection_status"] != "CERTIFIED":
        raise ValueError(f"Selection is not certified: {result['selection_status']}")
    old_heff = np.load(baseline / "projection/heff.npy", mmap_mode="r")
    result["baseline"]["heff_matrix"] = metrics(heff, old_heff)
    result["baseline"]["heff_spectrum"] = metrics(np.linalg.eigvalsh(heff), np.linalg.eigvalsh(old_heff))
    old_evaluator = imported(baseline / "model/evaluate.py", "published_oracle_" + name).load_model(baseline / "model")
    old_points = np.asarray(old_evaluator.data["reference_kpoints"], dtype=float)
    if old_points.shape != points.shape or not np.allclose(old_points, points, rtol=0, atol=1e-12):
        raise ValueError("Baseline and replay use different reference momenta; comparison is invalid")
    result["baseline"]["model_spectrum"] = metrics(evaluated, old_evaluator.bands(points))
    result["baseline"]["model_data"] = describe(baseline / "model/model_data.npz")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Published checkout holding recipes and real cached inputs")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", required=True, help="Absolute Python executable in the clean wheel environment")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    python = Path(args.python).absolute()
    if python != Path(sys.executable).absolute():
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    if args.workers < 1 or args.blas_threads < 1:
        parser.error("Worker and BLAS thread counts must be positive")
    import yaml
    root, run = args.root.resolve(strict=True), args.run_root.resolve()
    if not run.is_relative_to(Path("/data/work/tmp/zy/codex/moirekp")):
        raise ValueError("--run-root must be task-owned shared scratch beneath /data/work/tmp/zy/codex/moirekp")
    run.mkdir(parents=True, exist_ok=True)
    os.chdir(run)
    os.environ.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.blas_threads)
    import kp
    import tapw
    installed = {"kp": str(Path(kp.__file__).resolve()), "tapw": str(Path(tapw.__file__).resolve())}
    for path in installed.values():
        if not Path(path).is_relative_to(Path(sys.prefix).resolve()) or Path(path).is_relative_to(root):
            raise ValueError(f"Validation loaded development source instead of wheel: {path}")
    report = {"status": "preparing", "hostname": socket.gethostname(), "cwd": str(run),
              "environment_prefix": sys.prefix, "python": sys.executable, "installed_modules": installed,
              "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "command": [sys.executable, *sys.argv], "process_layout": {"sequential_cases": True, "project_workers": args.workers, "blas_threads_per_worker": args.blas_threads},
              "scope": "Published physical recipes; full 61-point KP project -> symm -> model from real cached TAPW inputs. DFT/TAPW matrices are not regenerated.",
              "cases": {}}
    summary = run / "published_examples.json"
    write(summary, report)
    current = None
    try:
        for name in args.cases:
            material, recipe, valley, shell, golden = CASES[name]
            original = root / "examples" / material / "kp" / recipe
            published = yaml.safe_load(original.read_text())
            config = yaml.safe_load(original.read_text())
            parent = root / "examples" / material / "tapw" / "outputs"
            case_root = run / "published_examples" / name
            config_path = case_root / "config.yaml"
            output_root = case_root / "outputs"
            if output_root.exists() and not args.prepare_only:
                raise FileExistsError(f"Refusing to replace a previous run: {output_root}")
            case_root.mkdir(parents=True, exist_ok=True)
            config["system"]["output"] = str(output_root)
            config["system"]["tapw_output"] = str(parent)
            config["project"]["workers"] = args.workers
            config["project"]["blas_threads"] = args.blas_threads
            unchanged = yaml.safe_load(yaml.safe_dump(config))
            for section, key in (("system", "output"), ("system", "tapw_output"), ("project", "workers"), ("project", "blas_threads")):
                if key in published[section]:
                    unchanged[section][key] = published[section][key]
                else:
                    unchanged[section].pop(key)
            if unchanged != published:
                raise ValueError("Verification script modified a physical recipe setting")
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            current = {"status": "prepared", "original_config": describe(original), "run_config": describe(config_path),
                       "inputs": input_identity(parent, valley, shell), "output": str(output_root / valley / shell), "stages": []}
            report["cases"][name] = current
            write(summary, report)
            if args.prepare_only:
                continue
            for stage in ("project", "symm", "model"):
                command = [sys.executable, "-u", "-m", "kp.cli", stage, "-c", str(config_path)]
                log = case_root / f"{stage}.log"
                entry = {"stage": stage, "command": command, "log": str(log), "status": "running"}
                current["stages"].append(entry)
                current["status"] = stage
                write(summary, report)
                print(f"{name}: {stage}; log={log}", flush=True)
                started = time.monotonic()
                with log.open("w") as stream:
                    completed = subprocess.run(command, cwd=run, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
                entry.update(returncode=completed.returncode, elapsed_s=time.monotonic() - started,
                             status="passed" if completed.returncode == 0 else "failed")
                write(summary, report)
                if completed.returncode:
                    raise subprocess.CalledProcessError(completed.returncode, command)
            current["status"] = "verifying"
            write(summary, report)
            current["verification"] = validate(output_root / valley / shell, root / golden, name)
            current["status"] = "passed"
            write(case_root / "verification.json", current)
            write(summary, report)
            print(f"{name}: passed", flush=True)
        report["status"] = "prepared" if args.prepare_only else "passed"
        write(summary, report)
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        if current is not None:
            current["status"] = "failed"
        write(summary, report)
        raise
    print(summary, flush=True)


if __name__ == "__main__":
    main()
