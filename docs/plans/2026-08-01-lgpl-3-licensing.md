# LGPL-3.0-or-later Release Licensing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Declare the MoireKP software release under LGPL-3.0-or-later while keeping external example datasets outside that software-license claim.

**Architecture:** The repository root will carry the authoritative GPL/LGPL texts and a concise copyright notice, while PEP 639 metadata will make the license machine-readable in source and wheel distributions. Release documentation and tests will distinguish the resolved software license from unresolved dataset licenses, DOI, URLs, and checksums.

**Tech Stack:** Python 3.11, pytest, setuptools/PEP 639, TOML, YAML, Markdown, GNU LGPL-3.0/GPL-3.0 legal texts.

---

### Task 1: Specify the authoritative software-license contract

**Files:**
- Modify: `tests/test_release_contract.py:61-75`
- Modify: `tests/test_release_contract.py:310-340`
- Modify: `tests/test_release_contract.py:491-500`

**Step 1: Write the failing metadata and license-file tests**

Add `COPYING`, `COPYING.LESSER`, and `COPYRIGHT` to `REQUIRED_SOURCE_MANIFEST_LINES`. Add a focused test that loads `pyproject.toml` and asserts:

```python
assert project["license"] == "LGPL-3.0-or-later"
assert project["license-files"] == ["COPYING", "COPYING.LESSER", "COPYRIGHT"]
assert "setuptools>=77" in pyproject["build-system"]["requires"]
```

The same test must assert that all three files exist; `COPYRIGHT` contains the exact author line and SPDX identifier; `COPYING` identifies GNU GPL version 3; and `COPYING.LESSER` identifies GNU LGPL version 3. Replace the old conditional MIT-only check with an unconditional assertion that repository README files do not claim MIT.

**Step 2: Run the focused test and verify failure**

Run:

```bash
python -m pytest tests/test_release_contract.py -q -p no:cacheprovider -k "license or source_manifest"
```

Expected: FAIL because license metadata and root license files do not yet exist.

**Step 3: Commit the failing contract**

```bash
git add tests/test_release_contract.py
git commit -m "test: require LGPL release metadata"
```

### Task 2: Add the authoritative LGPL texts and PEP 639 metadata

**Files:**
- Create: `COPYING`
- Create: `COPYING.LESSER`
- Create: `COPYRIGHT`
- Modify: `pyproject.toml:1-20`
- Modify: `MANIFEST.in:1-12`

**Step 1: Source the unmodified GNU legal texts**

Copy the complete official GNU GPL version 3 text into `COPYING` and the complete official GNU LGPL version 3 text into `COPYING.LESSER`. Prefer `/usr/share/common-licenses/GPL-3` and `/usr/share/common-licenses/LGPL-3` when present; otherwise download the plain-text licenses from `gnu.org` and verify their titles and version lines. Do not paraphrase either legal text.

**Step 2: Add the repository copyright notice**

Create `COPYRIGHT` with:

```text
MoireKP

Copyright (C) 2026 Yan Zhang, Jiabin Yu, and Quansheng Wu

SPDX-License-Identifier: LGPL-3.0-or-later

MoireKP is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

See COPYING.LESSER and COPYING for the complete license terms.
```

**Step 3: Add PEP 639 project metadata**

Update `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=77", "wheel"]

[project]
license = "LGPL-3.0-or-later"
license-files = ["COPYING", "COPYING.LESSER", "COPYRIGHT"]
```

Do not add a deprecated license classifier or TOML license table.

**Step 4: Include the legal files in the source manifest**

Add exact lines to `MANIFEST.in`:

```text
include COPYING
include COPYING.LESSER
include COPYRIGHT
```

**Step 5: Run the focused contract tests**

Run:

```bash
python -m pytest tests/test_release_contract.py -q -p no:cacheprovider -k "license or source_manifest"
```

Expected: PASS.

**Step 6: Commit the implementation**

```bash
git add COPYING COPYING.LESSER COPYRIGHT pyproject.toml MANIFEST.in
git commit -m "build: license MoireKP under LGPL-3.0-or-later"
```

### Task 3: Separate software licensing from dataset provenance

**Files:**
- Modify: `tests/test_release_contract.py:410-460`
- Modify: `tests/test_release_contract.py:526-590`
- Modify: `tests/test_release_contract.py:627-635`
- Modify: `pyproject.toml:47-50`
- Modify: `examples/data-manifest.yaml:1-16`
- Modify: `README.md:320-350`
- Modify: `README.zh.md:313-343`
- Modify: `RELEASE_BLOCKERS.md:1-20`
- Modify: `RELEASE_VALIDATION.md:35-45`

**Step 1: Write failing boundary tests**

Change prerelease fixtures and exact blocker expectations from `license` to `dataset_license`. Add assertions that:

```python
assert manifest["release_blockers"] == ["dataset_license", "doi", "data_url"]
assert manifest["metadata"]["license"] is None
assert all(dataset["license"] is None for dataset in manifest["datasets"])
```

Add README/blocker assertions requiring `LGPL-3.0-or-later`, a statement that external datasets are not automatically covered, a resolved software-license row, and a separate unresolved dataset-license row.

**Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_release_contract.py -q -p no:cacheprovider -k "manifest or readme or blocker or license"
```

Expected: FAIL because current release metadata still conflates software and dataset licensing.

**Step 3: Rename only the release blocker**

In `[tool.moirekp.release]` and `examples/data-manifest.yaml`, use:

```text
dataset_license, doi, data_url
```

Keep `metadata.license` and every dataset `license` value as `null`. Clarify in the manifest notes that the repository software license does not determine the licensing of externally sourced input/generated datasets.

**Step 4: Update the English and Chinese release documentation**

Add concise `License` / `许可证` sections stating that MoireKP software is LGPL-3.0-or-later and links to `COPYRIGHT`, `COPYING.LESSER`, and `COPYING`. State explicitly that external OpenMX/TAPW/KP datasets and large example artifacts require their own provenance and license metadata.

Update release-status wording so only dataset licenses, DOI, public data URLs, and checksums remain unresolved.

**Step 5: Update blocker and validation documents**

Mark the software license resolved in `RELEASE_BLOCKERS.md`, add a distinct `Dataset licenses` TBD row, and remove `final license` from `RELEASE_VALIDATION.md` in favor of unresolved dataset-license metadata.

**Step 6: Run the focused tests**

Run:

```bash
python -m pytest tests/test_release_contract.py -q -p no:cacheprovider -k "manifest or readme or blocker or license"
```

Expected: PASS.

**Step 7: Commit the boundary changes**

```bash
git add tests/test_release_contract.py pyproject.toml examples/data-manifest.yaml README.md README.zh.md RELEASE_BLOCKERS.md RELEASE_VALIDATION.md
git commit -m "docs: distinguish software and dataset licenses"
```

### Task 4: Verify source and wheel artifacts

**Files:**
- Modify if needed: `tests/test_release_contract.py`
- Output only: `validation_runs/lgpl_package_check_<timestamp>/`

**Step 1: Build clean distribution artifacts**

Run from the repository root with the active `moirekp` Python:

```bash
python -m build --sdist --wheel --outdir validation_runs/lgpl_package_check_<timestamp>
```

Expected: one `.tar.gz` and one `.whl`, with no files added to git status because `validation_runs/` is ignored.

**Step 2: Inspect package metadata and legal files**

Using `tar -tf`, `unzip -l`, and `unzip -p`, verify:

- sdist contains root-level `COPYING`, `COPYING.LESSER`, and `COPYRIGHT`;
- wheel contains all three files under `.dist-info/licenses/`;
- wheel `METADATA` contains `License-Expression: LGPL-3.0-or-later`;
- wheel `METADATA` lists all three `License-File` values.

If packaging behavior is not already protected by the release contract, add the smallest deterministic test for the missing invariant and repeat Tasks 1-2.

**Step 3: Run release-contract tests**

Run:

```bash
python -m pytest tests/test_release_contract.py tests/kp/test_example_dependency_contract.py -q -p no:cacheprovider
```

Expected: PASS.

**Step 4: Commit only durable test changes, if any**

```bash
git add tests/test_release_contract.py
git commit -m "test: verify packaged LGPL metadata"
```

Skip this commit when no test change is needed.

### Task 5: Run the release verification suite and publish the branch

**Files:**
- Verify only; do not add generated artifacts.

**Step 1: Run the default release test suite**

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Expected: PASS.

**Step 2: Audit the diff and privacy boundary**

Run:

```bash
git diff --check origin/codex/cpc-release-clean...HEAD
git status --short
git ls-files paper
rg -n "/data/|bigmem[0-9]+|validation_runs/|review_(bundles|outputs|packages)" \
  README.md README.zh.md RELEASE_BLOCKERS.md RELEASE_VALIDATION.md pyproject.toml \
  MANIFEST.in COPYRIGHT examples/data-manifest.yaml tests/test_release_contract.py
```

Expected: no whitespace errors, a clean worktree, no tracked `paper/` files, and no release-facing local paths or node names.

**Step 3: Push the reviewed feature branch**

```bash
git push -u origin codex/lgpl-license
```

Expected: the private remote receives only the reviewed LGPL branch; no paper content or validation output is uploaded.
