#!/usr/bin/env python3
"""Build and benchmark the readable physical export for one configured model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np

from kp.model.core import build_model, compute_coefficients
from kp.model.physical_export import build_physical_export_files
from kp.model.physical_runtime import PhysicalModel
from kp.model.pipeline import build_moire_config_from_file
from kp.model.response_basis import CompiledResponseRuntime


def _load_old_export(path: Path):
    spec = importlib.util.spec_from_file_location("old_export_evaluator", path / "evaluate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import old evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_model(path)


def _timings(function, points: np.ndarray, *, repeat: int) -> list[float]:
    function(points[:2])
    values = []
    for _ in range(int(repeat)):
        started = time.perf_counter()
        function(points)
        values.append(time.perf_counter() - started)
    return values


def _summary(values: list[float], point_count: int) -> dict[str, float]:
    median = float(statistics.median(values))
    return {
        "median_seconds": median,
        "minimum_seconds": float(min(values)),
        "maximum_seconds": float(max(values)),
        "microseconds_per_hamiltonian": 1.0e6 * median / int(point_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-export", type=Path, required=True)
    parser.add_argument("--expected-dimension", type=int)
    parser.add_argument("--expected-channels", type=int)
    parser.add_argument("--comparison-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-old-evaluation", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"loading configured model: {args.config}", flush=True)
    moire_config, model_config = build_moire_config_from_file(args.config.resolve())
    moire_config.output_dir = output / "response_cache"
    print("building continuum terms", flush=True)
    model = build_model(moire_config)
    print("compiling and fitting complete response basis", flush=True)
    compile_started = time.perf_counter()
    model, diagnostics = compute_coefficients(
        moire_config,
        model,
        progress_callback=lambda message, **_kwargs: print(message, flush=True),
    )
    compile_and_fit_seconds = time.perf_counter() - compile_started
    basis = model._compiled_response_basis
    fitted = model._fitted_response_model
    runtime = CompiledResponseRuntime(basis=basis, fitted=fitted)
    if (
        args.expected_dimension is not None
        and runtime.basis.dim != int(args.expected_dimension)
    ):
        raise ValueError(
            "compiled model dimension does not match benchmark contract: "
            f"actual={runtime.basis.dim}, expected={args.expected_dimension}"
        )
    if (
        args.expected_channels is not None
        and len(basis.channels) != int(args.expected_channels)
    ):
        raise ValueError(
            "compiled response channel count does not match benchmark contract: "
            f"actual={len(basis.channels)}, expected={args.expected_channels}"
        )
    arrays = runtime.export_arrays()
    np.savez_compressed(output / "operator_data.npz", **arrays)

    basis_metadata = basis.physical_export_metadata()
    sectors = [dict(row) for row in moire_config.sectors]
    physical_files = build_physical_export_files(
        model_name=model_config.path.stem,
        basis_metadata=basis_metadata,
        fitted_coefficients=np.asarray(arrays["fitted_coefficients"], dtype=float),
        qsets={"qset1": moire_config.Q_set1, "qset2": moire_config.Q_set2},
        reciprocal_basis=np.asarray((moire_config.bM1, moire_config.bM2)),
        sectors=sectors,
        symmetry_operations=model_config.symmetry_source_metadata.get(
            "operations", ()
        ),
        energy_unit=str(
            model_config.source_raw.get("material", {}).get("energy_unit", "eV")
        ),
        momentum_unit="1/angstrom",
        tolerance=1.0e-8,
    )
    for filename, payload in physical_files.items():
        (output / filename).write_text(payload, encoding="utf-8")
    physical = PhysicalModel.load(output)

    grid = np.linspace(-0.5, 0.5, 21)
    fractional = np.asarray([(u, v) for u in grid for v in grid], dtype=float)
    reciprocal = np.asarray((moire_config.bM1, moire_config.bM2), dtype=float)
    points = fractional @ reciprocal

    frozen_values = runtime.hamiltonians(points)
    physical_values = physical.hamiltonians(points)
    differences = physical_values - frozen_values
    maximum_element_error = float(np.max(np.abs(differences), initial=0.0))
    maximum_matrix_error = float(
        np.max(np.linalg.norm(differences, axis=(1, 2)), initial=0.0)
    )
    maximum_frozen_antihermitian = float(
        np.max(
            np.linalg.norm(
                frozen_values - frozen_values.conj().transpose(0, 2, 1),
                axis=(1, 2),
            ),
            initial=0.0,
        )
    )
    maximum_physical_antihermitian = float(
        np.max(
            np.linalg.norm(
                physical_values - physical_values.conj().transpose(0, 2, 1),
                axis=(1, 2),
            ),
            initial=0.0,
        )
    )

    timings = {
        "new_frozen_runtime": _summary(
            _timings(runtime.hamiltonians, points, repeat=args.repeat), len(points)
        ),
        "new_physical_routes": _summary(
            _timings(physical.hamiltonians, points, repeat=args.repeat), len(points)
        ),
    }
    if maximum_element_error > float(args.comparison_tolerance):
        raise ValueError(
            "physical export does not reproduce the frozen runtime: "
            f"error={maximum_element_error:.6e}, "
            f"tolerance={float(args.comparison_tolerance):.6e}"
        )
    old_comparison: dict[str, float | int | bool] = {
        "old_export_evaluation_skipped": bool(args.skip_old_evaluation)
    }
    if not args.skip_old_evaluation:
        old = _load_old_export(args.old_export.resolve())
        old_hamiltonians = lambda values: np.asarray(  # noqa: E731
            [old.hamiltonian(point) for point in np.asarray(values)]
        )
        old_values = old_hamiltonians(points)
        if old_values.shape != physical_values.shape:
            raise ValueError(
                "old export is not the same model dimension: "
                f"old={old_values.shape}, new={physical_values.shape}"
            )
        old_difference = physical_values - old_values
        maximum_old_element_error = float(
            np.max(np.abs(old_difference), initial=0.0)
        )
        old_comparison.update(
            {
                "old_export_dimension": int(old_values.shape[1]),
                "maximum_element_error_physical_vs_old_export": (
                    maximum_old_element_error
                ),
                "maximum_frobenius_error_physical_vs_old_export": float(
                    np.max(
                        np.linalg.norm(old_difference, axis=(1, 2)), initial=0.0
                    ),
                ),
                "old_export_equivalent_within_tolerance": bool(
                    maximum_old_element_error
                    <= float(args.comparison_tolerance)
                ),
            }
        )
        timings["old_export_frozen_runtime"] = _summary(
            _timings(old_hamiltonians, points, repeat=args.repeat), len(points)
        )

    result = {
        "config": str(args.config.resolve()),
        "mesh": [21, 21],
        "repeat_count": int(args.repeat),
        "comparison_tolerance": float(args.comparison_tolerance),
        "hamiltonian_count": int(len(points)),
        "dimension": int(runtime.basis.dim),
        "physical_term_count": int(len(physical.terms)),
        "physical_route_count": int(physical.route_count),
        "response_channel_count": int(len(basis.channels)),
        "cold_compile_and_fit_seconds": float(compile_and_fit_seconds),
        "maximum_element_error_physical_vs_frozen": maximum_element_error,
        "maximum_frobenius_error_physical_vs_frozen": maximum_matrix_error,
        "maximum_frozen_antihermitian_norm": maximum_frozen_antihermitian,
        "maximum_physical_antihermitian_norm": maximum_physical_antihermitian,
        **old_comparison,
        "timings": timings,
        "fit": {
            "basis_hash": basis.basis_hash,
            "fit_hash": fitted.fit_hash,
            "diagnostics": {
                key: value
                for key, value in diagnostics.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            },
        },
    }
    if "old_export_frozen_runtime" in timings:
        old_seconds = timings["old_export_frozen_runtime"]["median_seconds"]
        new_seconds = timings["new_physical_routes"]["median_seconds"]
        result["speed_ratio_new_physical_over_old_export"] = (
            new_seconds / old_seconds
        )
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
