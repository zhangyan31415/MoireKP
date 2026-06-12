from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DATA_MANIFEST = EXAMPLES_ROOT / "data-manifest.yaml"
README = EXAMPLES_ROOT / "README.md"
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/kp_dependency_contract"

PATH_FIELD_SUFFIXES = ("_file", "_dir", "_path", "_source", "source_config")
OUTPUT_FIELD_HINTS = {"out", "data_out", "out_dir", "output_dir", "dir"}
INPUT_FIELD_HINTS = {
    "band_file",
    "hamk_file",
    "heff_eig_file",
    "heff_file",
    "kpath.file",
    "qset1_file",
    "qset2_file",
    "source_config",
    "symmetry_source",
    "tapw_symmetry_dir",
}
README_MARKERS = ("external-data", "precomputed")


@dataclass(frozen=True)
class Dependency:
    config: Path
    field: str
    raw_path: str

    @property
    def resolved(self) -> Path:
        return (self.config.parent / self.raw_path).resolve()

    @property
    def repo_relative(self) -> str:
        try:
            return self.resolved.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return self.resolved.as_posix()

    @property
    def config_relative(self) -> str:
        return self.config.relative_to(REPO_ROOT).as_posix()


def _active_kp_configs() -> list[Path]:
    return sorted(EXAMPLES_ROOT.glob("*_3.89/kp/configs/**/*.yaml"))


def _is_path_like(value: str) -> bool:
    return "/" in value or "\\" in value


def _iter_path_dependencies(path: Path, value: object, field: str = "") -> Iterator[Dependency]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_field = f"{field}.{key}" if field else str(key)
            yield from _iter_path_dependencies(path, child, child_field)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_field = f"{field}[{index}]"
            yield from _iter_path_dependencies(path, child, child_field)
        return
    if not isinstance(value, str) or not _is_path_like(value):
        return

    leaf = field.split(".")[-1]
    if leaf in OUTPUT_FIELD_HINTS:
        return
    if field in INPUT_FIELD_HINTS or leaf.endswith(PATH_FIELD_SUFFIXES):
        yield Dependency(config=path, field=field, raw_path=value)


def _load_manifest_paths() -> set[str]:
    if not DATA_MANIFEST.exists():
        return set()

    raw = yaml.safe_load(DATA_MANIFEST.read_text(encoding="utf-8")) or {}
    paths: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"path", "paths", "pattern", "patterns"}:
                    collect(child)
                elif isinstance(child, (dict, list)):
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str) and _is_path_like(value):
            paths.add(value.strip())

    collect(raw)
    return paths


def _is_manifested(dependency: Dependency, manifest_paths: set[str]) -> bool:
    repo_relative = dependency.repo_relative
    raw_normalized = dependency.raw_path.replace("\\", "/")
    candidates = {
        repo_relative,
        raw_normalized,
        f"examples/{raw_normalized}",
        f"examples/{repo_relative}",
    }
    normalized_manifest_paths = {path.rstrip("/") for path in manifest_paths}
    return any(
        candidate == manifest_path or candidate.startswith(f"{manifest_path}/")
        for candidate in candidates
        for manifest_path in normalized_manifest_paths
    )


def test_active_kp_config_input_dependencies_exist_or_are_manifested() -> None:
    missing: list[str] = []
    manifest_paths = _load_manifest_paths()

    for config in _active_kp_configs():
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        for dependency in _iter_path_dependencies(config, raw):
            if dependency.resolved.exists() or _is_manifested(dependency, manifest_paths):
                continue
            missing.append(
                f"{dependency.config_relative}: {dependency.field} -> "
                f"{dependency.repo_relative}"
            )

    assert not missing, (
        "Input dependencies must exist in the release checkout or be declared in "
        f"{DATA_MANIFEST.relative_to(REPO_ROOT).as_posix()}:\n"
        + "\n".join(missing)
    )


def test_dependency_audit_fixture_allows_existing_files_and_manifested_external_data() -> None:
    config = FIXTURE_ROOT / "configs/source/tiny_source.yaml"
    manifest = yaml.safe_load((FIXTURE_ROOT / "data-manifest.yaml").read_text(encoding="utf-8"))
    manifest_paths = set(manifest["external_data"])
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))

    missing = [
        dependency
        for dependency in _iter_path_dependencies(config, raw)
        if not dependency.resolved.exists() and not _is_manifested(dependency, manifest_paths)
    ]

    assert missing == []


def _iter_readme_kp_command_blocks(text: str) -> Iterator[tuple[str, str]]:
    lines = text.splitlines()
    in_fence = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence or not stripped.startswith("kp "):
            continue
        context = "\n".join(lines[max(0, index - 2) : min(len(lines), index + 3)])
        yield stripped, context


def _iter_readme_active_kp_items(text: str) -> Iterator[tuple[str, str]]:
    for match in re.finditer(r"^\s*-\s+`(kp [^`]+)`(?P<context>[^\n]*)", text, re.MULTILINE):
        command = match.group(1)
        context = match.group(0)
        yield command, context


def test_examples_readme_active_kp_commands_include_symm_or_external_data_marker() -> None:
    text = README.read_text(encoding="utf-8")
    active_text = text.split("## Historical Artifacts", maxsplit=1)[0]
    commands = [
        *_iter_readme_kp_command_blocks(active_text),
        *_iter_readme_active_kp_items(active_text),
    ]

    assert commands, "examples/README.md should document the active KP DAG commands"

    undocumented = [
        command
        for command, context in commands
        if "kp symm" not in command and not any(marker in context for marker in README_MARKERS)
    ]
    assert not undocumented, (
        "Active KP README commands must include `kp symm` or be marked "
        "`external-data`/`precomputed`:\n"
        + "\n".join(undocumented)
    )
