from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

README_PATHS = [
    ROOT / "README.md",
    ROOT / "tapw" / "README.md",
    ROOT / "examples" / "README.md",
]

FORBIDDEN_README_TERMS = [
    "coming soon",
    "CURRENT_STATUS",
    "HANDOFF",
    "/data/work",
    "/data/home",
    "gpuh",
    "mambaforge",
    "tapw_mkl",
]

FORBIDDEN_TRACKED_PATTERNS = re.compile(
    r"(^dist/|"
    r"^manuscript/|"
    r"^cpc_submission/|"
    r"egg-info|"
    r"__pycache__|"
    r"\.pyc$|"
    r"^runs/|"
    r"^results/|"
    r"examples/.*/runs/|"
    r"examples/.*/outputs/|"
    r"H\.npz$|"
    r"S\.npz$|"
    r"openmx\.out$)"
)

REQUIRED_GITIGNORE_RULES = [
    "/runs/",
    "/results/",
    "/validation_runs/",
    "/build/",
    "/dist/",
    "/*.egg-info/",
    "/review_bundles/",
    "/dev_archive/",
    "/paper/",
    "/manuscript/",
    "/cpc_submission/",
]
REQUIRED_SOURCE_MANIFEST_LINES = {
    "recursive-include examples *.md *.yaml *.yml *.json *.txt *.csv *.py *.in *.out",
    "recursive-include examples/tapw/*/openmx/soc band_info.json openmx.dat_rigid",
    "recursive-include tests *.py *.yaml *.yml *.json *.txt",
    "recursive-include tests/fixtures *.txt *.yaml *.yml *.json *.npy *.npz",
    "include .gitignore",
    "include RELEASE_VALIDATION.md",
    "include pytest.ini",
    "include kp/README.md",
    "include tapw/README.md",
    "recursive-include scripts *.sh",
    "prune moirekp.egg-info",
    "prune review_bundles",
    "prune validation_runs",
    "prune dev_archive",
    "prune devtools",
    "prune docs/internal",
    "prune docs/plans",
    "prune paper",
    "prune manuscript",
    "prune cpc_submission",
    "prune examples/mgi2_3.89/kp/outputs",
    "prune examples/mgi2_3.89/kp/runs",
    "prune examples/mgi2_3.89/tapw",
    "prune examples/mote2_3.89/kp/outputs",
    "prune examples/mote2_3.89/kp/runs",
    "prune examples/mote2_3.89/tapw",
    "prune examples/tapw/mgi2_9.43/runs",
    "prune examples/tapw/mote2_9.43/runs",
    "global-exclude *.npy",
    "global-exclude *.npz",
    "global-exclude openmx.out",
    "recursive-include tests/fixtures *.npy *.npz",
}
REQUIRED_CLI_SCRIPTS = {"kp", "tapw"}
CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
PENDING_EXTERNAL_MARKERS = {"pending_external", "pending_public_archive"}
RUNTIME_CLASSES = {"clean-clone", "external-data", "precomputed-external"}
UNRESOLVED_METADATA_VALUES = {None, "TBD"}
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
RELEASE_LOCAL_WORKDIR_ROOTS = {
    "context",
    "dev_archive",
    "review_bundles",
    "review_outputs",
    "review_packages",
    "validation_runs",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _pyproject() -> dict:
    return tomllib.loads(_read(ROOT / "pyproject.toml"))


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_manifest_path_values(value: object) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"path", "paths", "pattern", "patterns"}:
                paths.extend(_collect_manifest_path_values(child))
            elif isinstance(child, (dict, list)):
                paths.extend(_collect_manifest_path_values(child))
        return paths
    if isinstance(value, list):
        for child in value:
            paths.extend(_collect_manifest_path_values(child))
        return paths
    if isinstance(value, str) and ("/" in value or "\\" in value):
        paths.append(value.strip())
    return paths


def _source_archive_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not _is_release_source_file(path, root):
            continue
        files.add(path.relative_to(root).as_posix())
    return files


def _tracked_files(root: Path = ROOT) -> set[str]:
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


def _final_release_mode() -> bool:
    return os.environ.get("MOIREKP_RELEASE_FINAL") == "1"


def _release_manifest_final_blockers(manifest: dict) -> list[str]:
    if not _final_release_mode():
        return []

    blockers: list[str] = []
    if manifest.get("status") != "complete":
        blockers.append("status")
    if manifest.get("release_blockers"):
        blockers.append("release_blockers")

    metadata = manifest.get("metadata", {})
    for field in ("license", "doi", "data_url"):
        if metadata.get(field) in UNRESOLVED_METADATA_VALUES:
            blockers.append(f"metadata.{field}")

    external_data = manifest.get("external_data", {})
    if external_data.get("marker") in PENDING_EXTERNAL_MARKERS:
        blockers.append("external_data.marker")

    for dataset in manifest.get("datasets", []):
        dataset_id = dataset.get("id", "<unknown>")
        for field in ("license", "doi", "data_url"):
            if dataset.get(field) in UNRESOLVED_METADATA_VALUES:
                blockers.append(f"datasets.{dataset_id}.{field}")
        for required_file in dataset.get("required_files", []):
            external = required_file.get("external")
            if isinstance(external, dict) and external.get("marker") in PENDING_EXTERNAL_MARKERS:
                blockers.append(
                    f"datasets.{dataset_id}.required_files."
                    f"{required_file.get('path', '<unknown>')}.external"
                )

    return blockers


def test_tracked_files_falls_back_to_source_archive_without_git(tmp_path: Path) -> None:
    archive_root = tmp_path / "moirekp-archive"
    (archive_root / "examples").mkdir(parents=True)
    (archive_root / "examples" / "demo" / "outputs").mkdir(parents=True)
    (archive_root / "validation_runs").mkdir()
    (archive_root / "review_bundles").mkdir()
    (archive_root / "moirekp.egg-info").mkdir()
    (archive_root / "__pycache__").mkdir()
    (archive_root / "examples" / "tapw" / "openmx" / "soc").mkdir(parents=True)
    (archive_root / "README.md").write_text("release archive\n", encoding="utf-8")
    (archive_root / "examples" / "config.yaml").write_text(
        "key: value\n",
        encoding="utf-8",
    )
    (archive_root / "examples" / "demo" / "outputs" / "generated.txt").write_text(
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
    (archive_root / "moirekp.egg-info" / "SOURCES.txt").write_text(
        "build metadata\n",
        encoding="utf-8",
    )
    (archive_root / "__pycache__" / "module.pyc").write_bytes(b"cache")
    (archive_root / "examples" / "tapw" / "openmx" / "soc" / "openmx.out").write_text(
        "generated\n",
        encoding="utf-8",
    )

    assert _tracked_files(archive_root) == {
        "README.md",
        "examples/config.yaml",
    }


def test_pyproject_declares_release_python_and_cli_contract() -> None:
    pyproject = _pyproject()

    assert pyproject["project"]["requires-python"] == ">=3.11"
    scripts = pyproject["project"]["scripts"]
    assert {name: scripts[name] for name in REQUIRED_CLI_SCRIPTS} == {
        "kp": "kp.cli:main",
        "tapw": "tapw.cli:main",
    }
    assert pyproject["tool"]["setuptools"]["include-package-data"] is True
    assert "templates/*.yaml" in pyproject["tool"]["setuptools"]["package-data"]["tapw"]
    assert "templates/*.in" in pyproject["tool"]["setuptools"]["package-data"]["tapw"]


def test_source_manifest_declares_release_archive_boundary() -> None:
    manifest_path = ROOT / "MANIFEST.in"
    assert manifest_path.exists(), "MANIFEST.in is required for source archive boundary"

    manifest_lines = set(_read(manifest_path).splitlines())
    missing = REQUIRED_SOURCE_MANIFEST_LINES - manifest_lines

    assert missing == set()


def test_final_release_gate_script_requires_final_mode() -> None:
    script = ROOT / "scripts" / "release_gate.sh"
    assert script.exists()
    assert os.access(script, os.X_OK)
    text = _read(script)
    assert "MOIREKP_RELEASE_FINAL=1" in text
    assert "-p no:cacheprovider" in text
    assert 'not slow and not external_data' in text
    assert "tests/test_release_contract.py" in text
    assert "tests/kp/test_example_dependency_contract.py" in text


def test_default_pytest_discovery_includes_release_contract_and_package_tests() -> None:
    pytest_ini = ROOT / "pytest.ini"
    assert pytest_ini.exists(), "pytest.ini is required for default release test discovery"

    lines = _read(pytest_ini).splitlines()
    testpaths: list[str] = []
    in_testpaths = False
    for line in lines:
        if line.startswith("testpaths ="):
            in_testpaths = True
            remainder = line.partition("=")[2].strip()
            if remainder:
                testpaths.append(remainder)
            continue
        if in_testpaths:
            if line and not line.startswith((" ", "\t")):
                break
            stripped = line.strip()
            if stripped:
                testpaths.append(stripped)

    assert "tests" in testpaths or "tests/test_release_contract.py" in testpaths
    assert "tests/kp" in testpaths


def test_release_validation_matches_cli_exit_contract() -> None:
    validation = _read(ROOT / "RELEASE_VALIDATION.md")
    cli = _read(ROOT / "tapw" / "tapw" / "cli.py")
    hard_exit_call = "os." + "_exit"

    assert hard_exit_call not in cli
    assert hard_exit_call not in validation
    assert "unimplemented sentinels" in validation
    assert "unit placeholder overrides" in validation
    assert "Python CLI boundary" in validation


def test_kp_model_core_has_no_import_time_demo_entrypoint() -> None:
    core = _read(ROOT / "kp" / "kp" / "model" / "core.py")

    forbidden = [
        "Minimal usage (pseudo-code)",
        "def example_config(",
        "def self_test(",
        'if __name__ == "__main__"',
        "toy generator",
        "toy/legacy",
    ]
    for term in forbidden:
        assert term not in core


def test_lijh_chern_module_is_only_a_compatibility_wrapper() -> None:
    text = _read(ROOT / "tapw" / "tapw" / "chern_post_lijh.py")

    assert "from .chern_post import" in text
    assert "matplotlib" not in text
    assert "argparse" not in text
    assert "if __name__" not in text


def test_release_final_mode_rejects_unresolved_manifest_metadata(monkeypatch) -> None:
    monkeypatch.setenv("MOIREKP_RELEASE_FINAL", "1")

    assert _release_manifest_final_blockers(
        {
            "status": "pending",
            "release_blockers": ["license", "doi", "data_url"],
            "metadata": {"license": None, "doi": "TBD", "data_url": None},
            "external_data": {"marker": "pending_external"},
            "datasets": [
                {
                    "id": "dataset",
                    "license": None,
                    "doi": "TBD",
                    "data_url": None,
                    "required_files": [
                        {
                            "path": "examples/external.dat",
                            "external": {"marker": "pending_external"},
                        }
                    ],
                }
            ],
        }
    ) == [
        "status",
        "release_blockers",
        "metadata.license",
        "metadata.doi",
        "metadata.data_url",
        "external_data.marker",
        "datasets.dataset.license",
        "datasets.dataset.doi",
        "datasets.dataset.data_url",
        "datasets.dataset.required_files.examples/external.dat.external",
    ]


def test_release_prerelease_mode_allows_unresolved_manifest_metadata(monkeypatch) -> None:
    monkeypatch.delenv("MOIREKP_RELEASE_FINAL", raising=False)

    assert _release_manifest_final_blockers(
        {
            "metadata": {"license": None, "doi": "TBD", "data_url": None},
            "datasets": [
                {"id": "dataset", "license": None, "doi": "TBD", "data_url": None}
            ],
        }
    ) == []


def test_release_contract_blocks_tracked_generated_artifacts() -> None:
    tracked_files = _tracked_files()

    forbidden_files = [
        path for path in tracked_files if FORBIDDEN_TRACKED_PATTERNS.search(path)
    ]

    assert forbidden_files == []


def test_release_gitignore_declares_root_generated_directories() -> None:
    gitignore = set(_read(ROOT / ".gitignore").splitlines())

    missing_rules = [
        rule for rule in REQUIRED_GITIGNORE_RULES if rule not in gitignore
    ]

    assert missing_rules == []


def test_release_readmes_are_public_facing() -> None:
    for path in README_PATHS:
        text = _read(path)
        lower_text = text.lower()
        for term in FORBIDDEN_README_TERMS:
            assert term.lower() not in lower_text, f"{path.relative_to(ROOT)} contains {term!r}"


def test_readmes_do_not_claim_unconfirmed_mit_license() -> None:
    license_files = list(ROOT.glob("LICENSE*")) + list(ROOT.glob("COPYING*"))
    project_metadata = _read(ROOT / "pyproject.toml").lower()
    license_declared = bool(license_files) or "license =" in project_metadata

    if not license_declared:
        for path in README_PATHS:
            assert "mit license" not in _read(path).lower(), (
                f"{path.relative_to(ROOT)} claims MIT without release license metadata"
            )


def test_environment_yml_release_contract() -> None:
    env_path = ROOT / "environment.yml"
    assert env_path.exists(), "root environment.yml is required for release"

    env = yaml.safe_load(_read(env_path))
    assert env["name"] == "moirekp"

    dependencies = env["dependencies"]
    dependency_text = "\n".join(str(dep) for dep in dependencies)
    assert "python=3.11" in dependency_text
    assert "mpi4py" in dependency_text
    assert "petsc4py" in dependency_text
    assert "slepc4py" in dependency_text

    pip_sections = [
        dep["pip"]
        for dep in dependencies
        if isinstance(dep, dict) and "pip" in dep
    ]
    assert pip_sections, "environment.yml must include a pip section"
    assert any("-e ." in pip_dep for section in pip_sections for pip_dep in section)


def test_data_manifest_tracks_release_dataset_provenance() -> None:
    manifest_path = ROOT / "examples" / "data-manifest.yaml"
    assert manifest_path.exists(), "examples/data-manifest.yaml is required for release"

    manifest = yaml.safe_load(_read(manifest_path))
    final_blockers = _release_manifest_final_blockers(manifest)
    assert final_blockers == []

    manifest_paths = _collect_manifest_path_values(manifest)
    local_workdir_paths = [
        path for path in manifest_paths if Path(path).parts[:1] and Path(path).parts[0] in RELEASE_LOCAL_WORKDIR_ROOTS
    ]
    assert local_workdir_paths == []

    assert manifest["schema_version"] == 2
    if _final_release_mode():
        assert manifest["status"] == "complete"
        assert manifest["release_blockers"] == []
    else:
        assert manifest["status"] in {"pending", "provisional", "complete"}
        assert manifest["release_blockers"] == ["license", "doi", "data_url"]

    metadata = manifest["metadata"]
    if not _final_release_mode():
        assert metadata["license"] in UNRESOLVED_METADATA_VALUES
        assert metadata["doi"] in UNRESOLVED_METADATA_VALUES
        assert metadata["data_url"] in UNRESOLVED_METADATA_VALUES

    external_data = manifest["external_data"]
    if _final_release_mode():
        assert external_data["marker"] not in PENDING_EXTERNAL_MARKERS
    else:
        assert external_data["marker"] in PENDING_EXTERNAL_MARKERS
    assert external_data["description"]
    assert external_data["paths"], "external_data must identify pending external paths"

    clean_clone_smoke = manifest["clean_clone_smoke"]
    assert clean_clone_smoke["runtime_class"] == "clean-clone"
    assert clean_clone_smoke["commands"], "clean-clone smoke must list safe commands"
    for command in clean_clone_smoke["commands"]:
        assert command["command"]
        assert command["runtime_class"] == "clean-clone"
        assert command["expected_outputs"]
        assert "tapw run" not in command["command"]
        assert not re.search(r"\bkp\s+(plot|project|symm|model)\b", command["command"])

    datasets = manifest["datasets"]
    assert isinstance(datasets, list)
    assert datasets, "data manifest must list release datasets"

    tracked_files = _tracked_files()
    for dataset in datasets:
        assert dataset["id"]
        assert dataset["path"]
        assert dataset["description"]
        assert dataset["runtime_class"] in RUNTIME_CLASSES
        if _final_release_mode():
            assert dataset["license"] not in UNRESOLVED_METADATA_VALUES
            assert dataset["doi"] not in UNRESOLVED_METADATA_VALUES
            assert dataset["data_url"] not in UNRESOLVED_METADATA_VALUES
        else:
            assert dataset["license"] in UNRESOLVED_METADATA_VALUES
            assert dataset["doi"] in UNRESOLVED_METADATA_VALUES
            assert dataset["data_url"] in UNRESOLVED_METADATA_VALUES
        assert dataset["required_files"], f"{dataset['id']} must list required files"
        assert dataset["commands"], f"{dataset['id']} must list reproducibility commands"

        for required_file in dataset["required_files"]:
            relative_path = required_file["path"]
            assert not Path(relative_path).is_absolute()
            has_checksum = "sha256" in required_file or "size_bytes" in required_file
            has_external_marker = "external" in required_file
            assert has_checksum != has_external_marker, (
                f"{dataset['id']}:{relative_path} must have either sha256/size "
                "or a pending external marker"
            )
            if has_checksum:
                path = ROOT / relative_path
                assert relative_path in tracked_files
                assert path.exists()
                assert CHECKSUM_RE.fullmatch(required_file["sha256"])
                assert required_file["size_bytes"] == path.stat().st_size
                assert required_file["sha256"] == _sha256(path)
            else:
                external = required_file["external"]
                assert external["marker"] in PENDING_EXTERNAL_MARKERS
                assert external["reason"]

        for command in dataset["commands"]:
            assert command["command"]
            assert command["runtime_class"] in RUNTIME_CLASSES
            assert command["expected_outputs"]
            for output in command["expected_outputs"]:
                assert output["path"]
                assert output["status"] in {
                    "generated",
                    "precomputed-external",
                    "pending-external",
                }


def test_release_blockers_record_unresolved_metadata() -> None:
    blockers_path = ROOT / "RELEASE_BLOCKERS.md"
    assert blockers_path.exists(), "RELEASE_BLOCKERS.md is required for release"

    text = _read(blockers_path).lower()
    assert "license" in text
    assert "doi" in text
    assert "data url" in text or "data_url" in text
    assert "tbd" in text or "unresolved" in text or "to be determined" in text
