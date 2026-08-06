#!/usr/bin/env python3
"""Run an isolated response-basis compiler benchmark and emit one JSON record."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import re
import resource
import socket
import subprocess
import time
from typing import Any, Sequence

import numpy as np

from kp.model.core import build_model
from kp.model.pipeline import build_moire_config_from_file
from kp.model.response_basis import (
    COMPILER_VERSION,
    clear_response_basis_cache,
    compile_model_response_basis,
)


SCHEMA_VERSION = "response-compile-benchmark-v1"
_CACHE_SUBDIRECTORY = ".compiled_response_basis_cache"
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_SECONDS_RE = re.compile(r"\bin ([0-9]+(?:\.[0-9]+)?) s(?:\b|$)")
_CACHE_PROGRESS_MARKERS = {
    "lookup_hit": "response basis persistent cache lookup hit",
    "cache_miss": "response basis persistent cache miss",
    "load_done": "response basis persistent cache load done",
    "cold_compile_done": "response basis cold compile done",
    "compile_store_done": "response basis persistent cache cold compile+store total",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one isolated response-basis compilation and emit JSON."
    )
    parser.add_argument("--config", type=Path, required=True, help="Configured-model YAML")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Dedicated moire.output_dir used only for this benchmark cache",
    )
    parser.add_argument("--cache-mode", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--profile-rss",
        action="store_true",
        help="Report process peak RSS after timing (disabled by default)",
    )
    return parser


def _cache_artifacts(cache_dir: Path) -> list[Path]:
    artifact_dir = cache_dir / _CACHE_SUBDIRECTORY
    if not artifact_dir.is_dir():
        return []
    return sorted(
        path
        for path in artifact_dir.glob("*.npz")
        if path.is_file() and _FINGERPRINT_RE.fullmatch(path.stem)
    )


def _validate_cache_mode(cache_dir: Path, cache_mode: str) -> list[Path]:
    if cache_mode == "cold":
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise ValueError(f"cold cache directory must be empty: {cache_dir}")
        return []
    artifacts = _cache_artifacts(cache_dir)
    if not artifacts:
        raise ValueError(
            "warm cache artifact is missing: expected "
            f"{cache_dir / _CACHE_SUBDIRECTORY}/*.npz"
        )
    return artifacts


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _event_seconds(messages: list[str], phrase: str) -> float | None:
    for message in reversed(messages):
        if phrase not in message:
            continue
        match = _SECONDS_RE.search(message)
        if match is not None:
            return float(match.group(1))
    return None


def _cache_progress_events(messages: list[str]) -> frozenset[str]:
    return frozenset(
        event
        for event, marker in _CACHE_PROGRESS_MARKERS.items()
        if any(marker in message for message in messages)
    )


def _validate_cache_progress(cache_mode: str, events: frozenset[str]) -> None:
    if cache_mode == "cold":
        if events & {"lookup_hit", "load_done"}:
            raise RuntimeError("cold cache unexpectedly hit a persistent artifact")
        if not {"cache_miss", "cold_compile_done"}.issubset(events):
            raise RuntimeError(
                "cold cache run did not report both a persistent miss and cold compile"
            )
        return
    if events & {"cache_miss", "cold_compile_done"}:
        raise RuntimeError("warm cache artifact did not match benchmark input")
    if not {"lookup_hit", "load_done"}.issubset(events):
        raise RuntimeError("warm cache artifact did not match benchmark input")


def _artifact_fingerprint(artifacts: list[Path], basis: Any) -> str:
    basis_hash = getattr(basis, "basis_hash", None)
    matches: list[Path] = []
    for path in artifacts:
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(
                    str(np.asarray(archive["cache_metadata_json"]).item())
                )
        except Exception:
            continue
        if metadata.get("compiler_input_key") != path.stem:
            continue
        if metadata.get("basis_hash") == basis_hash:
            matches.append(path)
    if len(matches) == 1:
        return matches[0].stem
    raise RuntimeError(
        "could not identify a unique response compiler cache artifact for this input"
    )


def _rank_metadata(basis: Any) -> dict[str, int | None]:
    channels = tuple(getattr(basis, "channels", ()))
    candidate = getattr(basis, "candidate_artifact", {})
    candidate = candidate if isinstance(candidate, Mapping) else {}
    structural = candidate.get("structural_preselection", {})
    structural = structural if isinstance(structural, Mapping) else {}
    return {
        "channel_count": len(channels),
        "retained_basis_channel_count": len(channels),
        "candidate_channel_count": candidate.get("candidate_channel_count"),
        "full_closure_rank": structural.get("full_closure_rank"),
        "selected_closure_rank": structural.get("selected_closure_rank"),
    }


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run_benchmark(
    *,
    config_path: Path,
    cache_dir: Path,
    cache_mode: str,
    profile_rss: bool = False,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    _validate_cache_mode(cache_dir, cache_mode)

    started = time.perf_counter()
    loaded_config = build_moire_config_from_file(config_path)
    moire = loaded_config[0] if isinstance(loaded_config, tuple) else loaded_config
    config_build_seconds = time.perf_counter() - started
    moire.output_dir = cache_dir

    started = time.perf_counter()
    model = build_model(moire)
    model_build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    clear_response_basis_cache()
    cache_clear_seconds = time.perf_counter() - started

    progress_messages: list[str] = []

    def record_progress(message: str, **_metadata: Any) -> None:
        progress_messages.append(str(message))

    started = time.perf_counter()
    basis = compile_model_response_basis(
        model,
        moire,
        reduce=True,
        progress_callback=record_progress,
    )
    compile_seconds = time.perf_counter() - started
    cache_progress_events = _cache_progress_events(progress_messages)
    _validate_cache_progress(cache_mode, cache_progress_events)

    started = time.perf_counter()
    artifacts = _cache_artifacts(cache_dir)
    if not artifacts:
        raise RuntimeError("response compiler did not produce or preserve a cache artifact")
    input_fingerprint = _artifact_fingerprint(artifacts, basis)
    cache_artifact_inspection_seconds = time.perf_counter() - started

    cold_compile_seconds = _event_seconds(
        progress_messages, _CACHE_PROGRESS_MARKERS["cold_compile_done"]
    )
    cache_compile_store_seconds = _event_seconds(
        progress_messages, _CACHE_PROGRESS_MARKERS["compile_store_done"]
    )
    cache_store_seconds = None
    if cold_compile_seconds is not None and cache_compile_store_seconds is not None:
        cache_store_seconds = max(
            0.0, cache_compile_store_seconds - cold_compile_seconds
        )
    cache_lookup_seconds = _event_seconds(
        progress_messages, _CACHE_PROGRESS_MARKERS["lookup_hit"]
    )
    if cache_lookup_seconds is None:
        cache_lookup_seconds = _event_seconds(
            progress_messages, _CACHE_PROGRESS_MARKERS["cache_miss"]
        )

    record = {
        "schema_version": SCHEMA_VERSION,
        "cache_mode": cache_mode,
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(),
        "compiler_version": COMPILER_VERSION,
        "input_fingerprint": input_fingerprint,
        "config_build_seconds": config_build_seconds,
        "model_build_seconds": model_build_seconds,
        "cache_clear_seconds": cache_clear_seconds,
        "compile_seconds": compile_seconds,
        "compile_seconds_definition": (
            "wall time of compile_model_response_basis only; the current compiler API "
            "includes its persistent cache lookup/load/store"
        ),
        "cold_compile_seconds": cold_compile_seconds,
        "cache_lookup_seconds": cache_lookup_seconds,
        "cache_load_seconds": _event_seconds(
            progress_messages, _CACHE_PROGRESS_MARKERS["load_done"]
        ),
        "cache_compile_store_seconds": cache_compile_store_seconds,
        "cache_store_seconds": cache_store_seconds,
        "cache_store_seconds_definition": (
            "cache_compile_store_seconds minus cold_compile_seconds; includes lock and "
            "cache bookkeeping overhead because the compiler API does not expose store alone"
        ),
        "cache_artifact_inspection_seconds": cache_artifact_inspection_seconds,
        **_rank_metadata(basis),
    }
    if profile_rss:
        record.update(
            {
                "peak_rss_bytes": _peak_rss_bytes(),
                "peak_rss_scope": "process high-water mark through compilation",
            }
        )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        record = run_benchmark(
            config_path=args.config,
            cache_dir=args.cache_dir,
            cache_mode=args.cache_mode,
            profile_rss=args.profile_rss,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
