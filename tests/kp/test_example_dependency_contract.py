from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DATA_MANIFEST = EXAMPLES_ROOT / "data-manifest.yaml"
README = EXAMPLES_ROOT / "README.md"
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/kp_dependency_contract"
README_PATHS = [
    README,
    EXAMPLES_ROOT / "mote2_3.89" / "README.md",
    EXAMPLES_ROOT / "mgi2_3.89" / "README.md",
    EXAMPLES_ROOT / "tapw" / "mote2_9.43" / "README.md",
    EXAMPLES_ROOT / "tapw" / "mgi2_9.43" / "README.md",
]

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
README_MARKERS = ("external-data", "precomputed", "clean-clone")
SOURCE_ARCHIVE_EXCLUDED_ROOTS = {
    ".codex",
    ".git",
    ".learnings",
    ".pytest_cache",
    "build",
    "context",
    "dist",
    "dev_archive",
    "review_bundles",
    "review_outputs",
    "review_packages",
    "runs",
    "results",
    "validation_runs",
}
SOURCE_ARCHIVE_EXCLUDED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    "outputs",
    "runs",
}
SOURCE_ARCHIVE_EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".dat",
    ".err",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".pyc",
    ".pyo",
    ".so",
    ".synctex.gz",
    ".xyz",
}
SOURCE_ARCHIVE_EXCLUDED_NAMES = {".DS_Store", "openmx.out"}


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
        path = Path(os.path.normpath(os.fspath(self.config.parent / self.raw_path)))
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

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

    external_data = raw.get("external_data", {})
    collect(external_data)
    for dataset in raw.get("datasets", []):
        for required_file in dataset.get("required_files", []):
            if isinstance(required_file, dict) and "external" in required_file:
                collect(required_file.get("path"))
                collect(required_file.get("paths"))
                collect(required_file.get("patterns"))
    return paths


def _is_release_source_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in SOURCE_ARCHIVE_EXCLUDED_ROOTS:
        return False
    if any(part.endswith(".egg-info") for part in parts):
        return False
    if any(part in SOURCE_ARCHIVE_EXCLUDED_PARTS for part in parts):
        return False
    name = path.name
    if name in SOURCE_ARCHIVE_EXCLUDED_NAMES:
        return False
    if any(name.endswith(suffix) for suffix in SOURCE_ARCHIVE_EXCLUDED_SUFFIXES):
        return False
    if path.suffix in {".npy", ".npz"} and parts[:2] != ("tests", "fixtures"):
        return False
    if parts[0] in {"paper", "manuscript", "cpc_submission"}:
        return False
    return True


def _source_archive_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not _is_release_source_file(path, root):
            continue
        files.add(path.relative_to(root).as_posix())
    return files


def _git_tracked_files(root: Path = REPO_ROOT) -> set[str]:
    if not (root / ".git").exists():
        return _source_archive_files(root)
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            check=True,
            cwd=root,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _source_archive_files(root)
    return {path for path in result.stdout.splitlines() if (root / path).exists()}


def test_git_tracked_files_falls_back_to_source_archive_without_git(tmp_path: Path) -> None:
    archive_root = tmp_path / "moirekp-archive"
    (archive_root / "examples" / "mote2_3.89").mkdir(parents=True)
    (archive_root / "examples" / "mote2_3.89" / "outputs").mkdir(parents=True)
    (archive_root / "validation_runs").mkdir()
    (archive_root / "review_bundles").mkdir()
    (archive_root / "__pycache__").mkdir()
    (archive_root / "README.md").write_text("release archive\n", encoding="utf-8")
    (archive_root / "examples" / "mote2_3.89" / "README.md").write_text(
        "example\n",
        encoding="utf-8",
    )
    (archive_root / "examples" / "mote2_3.89" / "outputs" / "generated.txt").write_text(
        "generated\n",
        encoding="utf-8",
    )
    (archive_root / "validation_runs" / "large.txt").write_text(
        "generated\n",
        encoding="utf-8",
    )
    (archive_root / "review_bundles" / "review.tar.gz").write_text(
        "internal\n",
        encoding="utf-8",
    )
    (archive_root / "__pycache__" / "module.pyc").write_bytes(b"cache")

    assert _git_tracked_files(archive_root) == {
        "README.md",
        "examples/mote2_3.89/README.md",
    }


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
    tracked_files = _git_tracked_files()

    for config in _active_kp_configs():
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        for dependency in _iter_path_dependencies(config, raw):
            if (
                dependency.repo_relative in tracked_files
                and dependency.resolved.exists()
            ) or _is_manifested(dependency, manifest_paths):
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


def _is_example_runtime_command(command: str) -> bool:
    if command.startswith("tapw run "):
        return True
    if command.startswith("kp --help"):
        return True
    return command.startswith("kp ") and (
        " --config " in command or " -c " in command
    )


def _iter_readme_command_blocks(text: str) -> Iterator[tuple[str, str]]:
    lines = text.splitlines()
    in_fence = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence or not _is_example_runtime_command(stripped):
            continue
        context = "\n".join(lines[max(0, index - 2) : min(len(lines), index + 3)])
        yield stripped, context


def _iter_readme_active_command_items(text: str) -> Iterator[tuple[str, str]]:
    for match in re.finditer(
        r"^\s*-\s+`((?:kp|tapw run) [^`]+)`(?P<context>[^\n]*)",
        text,
        re.MULTILINE,
    ):
        command = match.group(1)
        if not _is_example_runtime_command(command):
            continue
        context = match.group(0)
        yield command, context


def test_examples_readmes_active_commands_include_runtime_marker() -> None:
    commands: list[tuple[Path, str, str]] = []
    for readme in README_PATHS:
        text = readme.read_text(encoding="utf-8")
        active_text = text.split("## Historical Artifacts", maxsplit=1)[0]
        commands.extend(
            (readme, command, context)
            for command, context in [
                *_iter_readme_command_blocks(active_text),
                *_iter_readme_active_command_items(active_text),
            ]
        )

    assert commands, "example README files should document active commands"

    undocumented = [
        f"{readme.relative_to(REPO_ROOT).as_posix()}: {command}"
        for readme, command, context in commands
        if not any(marker in context for marker in README_MARKERS)
    ]
    assert not undocumented, (
        "Active example README commands must be marked with runtime class "
        "`external-data`, `precomputed`, or `clean-clone`:\n"
        + "\n".join(undocumented)
    )
