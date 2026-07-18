# Automatic Projection Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ten directly runnable automatic-project examples and print the selected states' layer, spin, and orbital composition from `kp project`.

**Architecture:** Reuse the existing automatic-selection report as the single data source, add a pure terminal formatter around that report, and curate portable YAML configs copied from the validated runs with repository-relative data paths. A manifest-driven runner checks each resolved selection without embedding large TAPW inputs.

**Tech Stack:** Python 3.11, NumPy, PyYAML, pytest, MoireKP CLI, SSH bigmem validation.

---

### Task 1: Terminal composition summary

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `kp/kp/cli.py`
- Test: `tests/kp/test_auto_low_energy_selection.py`

1. Add a failing test for a report containing two states with unsorted orbital, layer, and spin weights.
2. Assert the formatter prints status, resolved `nlow_state_list`, every state, dominant physical layer, spin weights, and orbitals sorted by descending weight.
3. Run the test and verify it fails because the formatter is absent.
4. Implement a pure `render_selection_console(report)` function and call it from `kp project` after writing the final report.
5. Run the focused tests and verify they pass.

### Task 2: Curated autoproject configs

**Files:**
- Create: `examples/autoproject/README.md`
- Create: `examples/autoproject/expected_selections.yaml`
- Create: `examples/autoproject/configs/*.yaml`
- Test: `tests/kp/test_autoproject_examples.py`

1. Add a failing contract test that requires ten named configs, automatic selection, no explicit `nlow_state_list`, portable paths, and exact expected state lists.
2. Run it and verify the missing directory failure.
3. Copy the validated configuration semantics from `validation_runs/auto_low_energy_20260718/configs/`, replacing output and data paths with repository-relative paths.
4. Document the one-command usage, generated output locations, external-data prerequisite, PASS/WARN meaning, and known MgI2 M spinless limitation.
5. Run the contract test and YAML-loading tests.

### Task 3: Manifest-driven batch runner

**Files:**
- Create: `examples/autoproject/run_all.py`
- Test: `tests/kp/test_autoproject_examples.py`

1. Add a failing test for command construction and report comparison.
2. Implement `--cases`, `--output-root`, and `--check-only`; execute `kp project` without shell interpolation and compare `low_energy_selection.json` against the expected manifest.
3. Ensure subprocess exit status and WARN cases are reported accurately.
4. Run focused tests.

### Task 4: Three-node external-data validation

**Files:**
- Generate only below: `validation_runs/autoproject_20260718/`

1. Assign MoTe2 cases to bigmem001, MgI2/ZrS2 cases to bigmem002, and A-AB/PtSe2 cases to bigmem003.
2. Verify the remote `python`, `kp`, repository revision/worktree, and MKL environment before running.
3. Run all ten `kp project` commands and retain true exit statuses and logs below the validation directory.
4. Compare every final report with `expected_selections.yaml` and inspect symmetry status.
5. Confirm terminal logs contain orbital, layer, and spin summaries.

### Task 5: Regression and release checks

**Files:**
- Test: `tests/kp/test_auto_low_energy_selection.py`
- Test: `tests/kp/test_autoproject_examples.py`

1. Run focused automatic-selection and CLI tests.
2. Run `python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"` and distinguish unrelated dirty-worktree failures.
3. Run `git diff --check` and confirm no generated outputs are tracked.
4. Report all ten selections, status, orbital composition summary, node assignment, and reproducible commands.

