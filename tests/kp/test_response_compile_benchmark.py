from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "devtools"
    / "benchmarks"
    / "kp"
    / "bench_response_compile.py"
)


def _load_runner():
    assert RUNNER_PATH.is_file(), f"response compiler benchmark runner is missing: {RUNNER_PATH}"
    spec = importlib.util.spec_from_file_location("bench_response_compile", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cache_artifact(
    runner: object,
    cache_dir: Path,
    *,
    path_fingerprint: str,
    input_fingerprint: str,
    basis_hash: str,
) -> Path:
    artifact_dir = cache_dir / ".compiled_response_basis_cache"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{path_fingerprint}.npz"
    runner.np.savez(
        path,
        cache_metadata_json=runner.np.asarray(
            json.dumps(
                {
                    "compiler_input_key": input_fingerprint,
                    "basis_hash": basis_hash,
                }
            )
        ),
    )
    return path


def test_cli_emits_versioned_isolated_benchmark_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    cache_dir = tmp_path / "cold-cache"
    config_path = tmp_path / "case.yaml"
    config_path.write_text("case: fake\n", encoding="utf-8")
    events: list[object] = []

    def fake_load_config(path: Path):
        events.append(("load", path))
        return SimpleNamespace(output_dir="original")

    def fake_build_model(config: object):
        events.append(("build", getattr(config, "output_dir")))
        return object()

    def fake_clear_cache() -> None:
        events.append("clear")

    def fake_compile(model: object, config: object, *, reduce: bool, progress_callback):
        events.append(("compile", getattr(config, "output_dir"), reduce))
        progress_callback("response basis persistent cache miss in 0.01 s")
        progress_callback("response basis cold compile done in 0.20 s")
        basis_hash = "cold-basis-hash"
        _write_cache_artifact(
            runner,
            Path(getattr(config, "output_dir")),
            path_fingerprint="a" * 64,
            input_fingerprint="a" * 64,
            basis_hash=basis_hash,
        )
        progress_callback(
            "response basis persistent cache cold compile+store total in 0.25 s"
        )
        return SimpleNamespace(
            basis_hash=basis_hash,
            channels=("c0", "c1", "c2"),
            candidate_artifact=MappingProxyType(
                {
                    "candidate_channel_count": 5,
                    "structural_preselection": MappingProxyType(
                        {
                            "full_closure_rank": 7,
                            "selected_closure_rank": 5,
                        }
                    ),
                }
            ),
        )

    monkeypatch.setattr(runner, "build_moire_config_from_file", fake_load_config)
    monkeypatch.setattr(runner, "build_model", fake_build_model)
    monkeypatch.setattr(runner, "clear_response_basis_cache", fake_clear_cache)
    monkeypatch.setattr(runner, "compile_model_response_basis", fake_compile)
    monkeypatch.setattr(runner, "COMPILER_VERSION", "compiler-test-v1")
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "bench-host")
    monkeypatch.setattr(runner, "_git_commit", lambda: "0123456789abcdef")

    assert runner.main(
        [
            "--config",
            str(config_path),
            "--cache-dir",
            str(cache_dir),
            "--cache-mode",
            "cold",
        ]
    ) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["schema_version"] == "response-compile-benchmark-v1"
    assert record["cache_mode"] == "cold"
    assert record["compile_seconds"] >= 0.0
    assert record["git_commit"] == "0123456789abcdef"
    assert record["input_fingerprint"] == "a" * 64
    assert record["compiler_version"] == "compiler-test-v1"
    assert record["hostname"] == "bench-host"
    assert record["cache_lookup_seconds"] == pytest.approx(0.01)
    assert record["cache_compile_store_seconds"] == pytest.approx(0.25)
    assert record["cache_store_seconds"] == pytest.approx(0.05)
    assert record["channel_count"] == 3
    assert record["retained_basis_channel_count"] == 3
    assert "response_rank" not in record
    assert record["candidate_channel_count"] == 5
    assert record["full_closure_rank"] == 7
    assert record["selected_closure_rank"] == 5
    assert "peak_rss_bytes" not in record
    assert events == [
        ("load", config_path.resolve()),
        ("build", cache_dir.resolve()),
        "clear",
        ("compile", cache_dir.resolve(), True),
    ]


def test_cold_mode_rejects_nonempty_cache_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    cache_dir = tmp_path / "occupied"
    cache_dir.mkdir()
    (cache_dir / "keep.txt").write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        runner.main(
            [
                "--config",
                str(tmp_path / "case.yaml"),
                "--cache-dir",
                str(cache_dir),
                "--cache-mode",
                "cold",
            ]
        )

    assert "cold cache directory must be empty" in capsys.readouterr().err
    assert (cache_dir / "keep.txt").read_text(encoding="utf-8") == "do not delete\n"


def test_cold_mode_rejects_persistent_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    cache_dir = tmp_path / "unexpected-hit-cache"
    config_path = tmp_path / "case.yaml"
    config_path.write_text("case: fake\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "build_moire_config_from_file",
        lambda _path: SimpleNamespace(output_dir="original"),
    )
    monkeypatch.setattr(runner, "build_model", lambda _config: object())
    monkeypatch.setattr(runner, "clear_response_basis_cache", lambda: None)

    def fake_compile(_model: object, config: object, *, progress_callback, **_kwargs: object):
        progress_callback("response basis persistent cache lookup hit in 0.01 s")
        progress_callback("response basis persistent cache load done in 0.02 s | channels=1")
        basis_hash = "unexpected-hit-basis"
        _write_cache_artifact(
            runner,
            Path(getattr(config, "output_dir")),
            path_fingerprint="e" * 64,
            input_fingerprint="e" * 64,
            basis_hash=basis_hash,
        )
        return SimpleNamespace(
            basis_hash=basis_hash,
            channels=("c0",),
            candidate_artifact={},
        )

    monkeypatch.setattr(runner, "compile_model_response_basis", fake_compile)

    with pytest.raises(SystemExit, match="2"):
        runner.main(
            [
                "--config",
                str(config_path),
                "--cache-dir",
                str(cache_dir),
                "--cache-mode",
                "cold",
            ]
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cold cache unexpectedly hit a persistent artifact" in captured.err


def test_peak_rss_is_reported_only_when_explicitly_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    cache_dir = tmp_path / "profile-cache"
    config_path = tmp_path / "case.yaml"
    config_path.write_text("case: fake\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "build_moire_config_from_file",
        lambda _path: SimpleNamespace(output_dir="original"),
    )
    monkeypatch.setattr(runner, "build_model", lambda _config: object())
    monkeypatch.setattr(runner, "clear_response_basis_cache", lambda: None)

    def fake_compile(_model: object, config: object, *, progress_callback, **_kwargs: object):
        progress_callback("response basis persistent cache miss in 0.01 s")
        progress_callback("response basis cold compile done in 0.20 s")
        basis_hash = "profile-basis-hash"
        _write_cache_artifact(
            runner,
            Path(getattr(config, "output_dir")),
            path_fingerprint="b" * 64,
            input_fingerprint="b" * 64,
            basis_hash=basis_hash,
        )
        progress_callback(
            "response basis persistent cache cold compile+store total in 0.25 s"
        )
        return SimpleNamespace(
            basis_hash=basis_hash,
            channels=(),
            candidate_artifact={},
        )

    monkeypatch.setattr(runner, "compile_model_response_basis", fake_compile)
    monkeypatch.setattr(runner, "_peak_rss_bytes", lambda: 123_456, raising=False)

    assert runner.main(
        [
            "--config",
            str(config_path),
            "--cache-dir",
            str(cache_dir),
            "--cache-mode",
            "cold",
            "--profile-rss",
        ]
    ) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["peak_rss_bytes"] == 123_456
    assert record["peak_rss_scope"] == "process high-water mark through compilation"


def test_warm_mode_rejects_missing_cache_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    cache_dir = tmp_path / "empty"
    cache_dir.mkdir()

    with pytest.raises(SystemExit, match="2"):
        runner.main(
            [
                "--config",
                str(tmp_path / "case.yaml"),
                "--cache-dir",
                str(cache_dir),
                "--cache-mode",
                "warm",
            ]
        )

    assert "warm cache artifact is missing" in capsys.readouterr().err


def test_warm_mode_loads_matching_artifact_from_tuple_config_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    cache_dir = tmp_path / "warm-cache"
    basis_hash = "warm-basis-hash"
    _write_cache_artifact(
        runner,
        cache_dir,
        path_fingerprint="e" * 64,
        input_fingerprint="f" * 64,
        basis_hash=basis_hash,
    )
    _write_cache_artifact(
        runner,
        cache_dir,
        path_fingerprint="f" * 64,
        input_fingerprint="f" * 64,
        basis_hash=basis_hash,
    )
    config_path = tmp_path / "case.yaml"
    config_path.write_text("case: fake\n", encoding="utf-8")
    moire = SimpleNamespace(output_dir="original")
    configured = SimpleNamespace(name="configured")
    events: list[object] = []

    def fake_load_config(path: Path):
        events.append(("load", path))
        return moire, configured

    def fake_build_model(config: object):
        events.append(("build", config, getattr(config, "output_dir")))
        return object()

    def fake_compile(_model: object, config: object, *, progress_callback, **_kwargs: object):
        events.append(("compile", config))
        progress_callback("response basis persistent cache lookup hit in 0.01 s")
        progress_callback("response basis persistent cache load done in 0.02 s | channels=2")
        return SimpleNamespace(
            basis_hash=basis_hash,
            channels=("c0", "c1"),
            candidate_artifact={"candidate_channel_count": 3},
        )

    monkeypatch.setattr(runner, "build_moire_config_from_file", fake_load_config)
    monkeypatch.setattr(runner, "build_model", fake_build_model)
    monkeypatch.setattr(runner, "clear_response_basis_cache", lambda: None)
    monkeypatch.setattr(runner, "compile_model_response_basis", fake_compile)

    assert runner.main(
        [
            "--config",
            str(config_path),
            "--cache-dir",
            str(cache_dir),
            "--cache-mode",
            "warm",
        ]
    ) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["cache_mode"] == "warm"
    assert record["input_fingerprint"] == "f" * 64
    assert record["cache_lookup_seconds"] == pytest.approx(0.01)
    assert record["cache_load_seconds"] == pytest.approx(0.02)
    assert record["cold_compile_seconds"] is None
    assert record["retained_basis_channel_count"] == 2
    assert record["candidate_channel_count"] == 3
    assert events == [
        ("load", config_path.resolve()),
        ("build", moire, cache_dir.resolve()),
        ("compile", moire),
    ]


def test_warm_mode_rejects_unrelated_artifact_when_compiler_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load_runner()
    cache_dir = tmp_path / "wrong-warm-cache"
    _write_cache_artifact(
        runner,
        cache_dir,
        path_fingerprint="c" * 64,
        input_fingerprint="c" * 64,
        basis_hash="unrelated-basis-hash",
    )
    config_path = tmp_path / "case.yaml"
    config_path.write_text("case: fake\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "build_moire_config_from_file",
        lambda _path: SimpleNamespace(output_dir="original"),
    )
    monkeypatch.setattr(runner, "build_model", lambda _config: object())
    monkeypatch.setattr(runner, "clear_response_basis_cache", lambda: None)

    def fake_compile(_model: object, config: object, *, progress_callback, **_kwargs: object):
        progress_callback("response basis persistent cache miss in 0.01 s")
        progress_callback("response basis cold compile done in 0.20 s")
        basis_hash = "new-basis-hash"
        _write_cache_artifact(
            runner,
            Path(getattr(config, "output_dir")),
            path_fingerprint="d" * 64,
            input_fingerprint="d" * 64,
            basis_hash=basis_hash,
        )
        return SimpleNamespace(
            basis_hash=basis_hash,
            channels=("c0",),
            candidate_artifact={},
        )

    monkeypatch.setattr(runner, "compile_model_response_basis", fake_compile)

    with pytest.raises(SystemExit, match="2"):
        runner.main(
            [
                "--config",
                str(config_path),
                "--cache-dir",
                str(cache_dir),
                "--cache-mode",
                "warm",
            ]
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "warm cache artifact did not match benchmark input" in captured.err
