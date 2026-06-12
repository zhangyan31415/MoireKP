from __future__ import annotations

import re
import subprocess
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
    "/build/",
    "/dist/",
    "/*.egg-info/",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_contract_blocks_tracked_generated_artifacts() -> None:
    tracked_files = subprocess.run(
        ["git", "ls-files"],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    ).stdout.splitlines()

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
    assert manifest["schema_version"] == 1
    assert manifest["status"] in {"pending", "provisional", "complete"}
    assert manifest["release_blockers"] == ["license", "doi", "data_url"]

    datasets = manifest["datasets"]
    assert isinstance(datasets, list)
    assert datasets, "data manifest must list release datasets"

    for dataset in datasets:
        assert dataset["id"]
        assert dataset["path"]
        assert dataset["license"] in {None, "TBD"}
        assert dataset["doi"] in {None, "TBD"}
        assert dataset["data_url"] in {None, "TBD"}


def test_release_blockers_record_unresolved_metadata() -> None:
    blockers_path = ROOT / "RELEASE_BLOCKERS.md"
    assert blockers_path.exists(), "RELEASE_BLOCKERS.md is required for release"

    text = _read(blockers_path).lower()
    assert "license" in text
    assert "doi" in text
    assert "data url" in text or "data_url" in text
    assert "tbd" in text or "unresolved" in text or "to be determined" in text
