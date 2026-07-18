# KP High/Low Automatic Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make one `kp model` run automatically select and export accurate `high` and compact edge-preserving `low` continuum models.

**Architecture:** Extend the pure model-selection layer to return two profile decisions from a shared candidate pool, then connect a production candidate evaluator to the configured pipeline. Each candidate rebuilds its vocabulary and is validated with fixed target-only edge weights and blocked folds. The winning configurations are rerun into separate profile directories and exported independently, with `high` retained as the compatibility result.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyYAML, matplotlib, pytest, existing configured KP pipeline and standalone exporter.

---

### Task 1: Dual-profile selection policy

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `tests/kp/test_model_selection.py`

**Steps:**

1. Write failing tests where high selects minimum weighted RMS and low selects fewer parameters on a `2SE` primary-edge plateau despite worse expanded-window error.
2. Run the focused tests and confirm the new profile API is absent.
3. Add candidate primary/expanded metric fields, `ProfileSelectionDecision`, and `select_high_low_profiles`.
4. Keep the overlap `0.95` target and `0.90` safety-floor status semantics for both profiles.
5. Rerun all model-selection tests and commit only the clean selection module and its tests.

### Task 2: Production candidate specification and reports

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_model_selection.py`
- Modify: `tests/kp/test_configured_model.py`

**Steps:**

1. Write failing tests for deterministic candidate names, family vocabularies, independent-real-parameter counts, profile output paths, and JSON-safe reports.
2. Add immutable production candidate/result records and report serializers.
3. Parse automatic selection as enabled by default; preserve an explicit `enabled: false` fixed-fit path.
4. Treat configured maximum orders as ceilings and generate bounded family ladders with valley defaults when absent.
5. Run config/report tests without external data.

### Task 3: Configured candidate evaluator

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_configured_model.py`

**Steps:**

1. Write a failing small-fixture test that records candidate vocabulary recompilation and coefficient refits across `Kinect -> intra -> inter`.
2. Factor one side-effect-contained candidate evaluator from `_run_model_pipeline`.
3. Build fixed target-only edge weights once, generate contiguous blocked folds, and aggregate fold means/standard errors.
4. Use centered response loss for the kinetic-only stage and complete weighted band/subspace metrics for complete candidates.
5. Cache only exact vocabulary/fold hashes; never slice a higher-order compiled basis by nominal order.
6. Run focused configured-model tests.

### Task 4: Closed-group pruning and correction integration

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_configured_model.py`

**Steps:**

1. Write failing tests mapping compiled response metadata into disjoint symmetry/Hermitian/adjoint-closed `ParameterGroup` records.
2. Connect group ablation to the production evaluator so every removal refits remaining coefficients.
3. Count independent nonzero real response channels after pruning.
4. Run the local `+/-1` family-order sweep and add every evaluated correction to the common candidate table.
5. Verify that low never removes a partial adjoint or real/imaginary group.

### Task 5: Two complete profile runs and exports

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/model/export.py` only if the existing exporter cannot target the two profile directories
- Modify: `tests/kp/test_configured_model.py`
- Modify: relevant CLI/export tests under `tests/kp/`

**Steps:**

1. Write failing tests requiring `high/` and `low/` runnable standalone packages from one configured command.
2. Rerun the two selected vocabularies with refinement enabled into isolated profile directories.
3. Return `profile_results = {"high": ..., "low": ...}` and use high for legacy top-level fields.
4. Export each result independently and print profile orders, parameters, errors, overlap, and status in the CLI summary.
5. Ensure a single selected vocabulary is evaluated once when high and low coincide.

### Task 6: User-facing comparison artifacts

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_configured_model.py`

**Steps:**

1. Write failing tests for the root JSON, Markdown, CSV, and high/low comparison plot.
2. Report primary-edge metrics separately from expanded/all-band diagnostics.
3. Plot Heff, high, and low requested edge bands and a parameter-count versus validation-error candidate frontier.
4. Record `PASS`, `WARN_BEST_AVAILABLE`, or `FAIL` without hiding unmet thresholds.

### Task 7: Local end-to-end fixture

**Files:**
- Add only ignored artifacts under `validation_runs/kp_high_low_auto_<timestamp>/`

**Steps:**

1. Run the smallest available configured fixture with the MKL `moirekp` environment.
2. Execute both exported `evaluate.py` programs and compare their archived bands to pipeline results.
3. Confirm all report paths are relative/release-safe and no validation files are staged.
4. Fix every observed runtime error and repeat until the one-command fixture passes.

### Task 8: Bigmem MoTe2 K and MgI2 M validation

**Files:**
- Add only ignored artifacts under `validation_runs/kp_high_low_auto_<timestamp>/`
- Do not modify the source example configurations during trials.

**Steps:**

1. Probe bigmem's repository path, MKL environment, available RAM/CPU, and input artifacts through direct SSH.
2. Copy derived trial configs into `validation_runs/` and run MoTe2 K and MgI2 M with automatic selection.
3. Continue diagnosing and fixing candidate failures until both cases produce high/low exports or a mechanically demonstrated data-quality safety-floor failure.
4. Retrieve compact JSON/CSV/PDF artifacts, not large source arrays or node logs.
5. Verify low uses fewer independent real parameters while retaining the requested edge-band quality.

### Task 9: Release documentation and verification

**Files:**
- Modify: `kp/README.md`
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: example README/config files only after the final interface is stable

**Steps:**

1. Document default automatic behavior, manual opt-out, high/low semantics, status meanings, and output layout.
2. Run all focused model-selection/configured-model/CLI/export tests.
3. Run `python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"`.
4. Run Python compilation and `git diff --check` for every modified release-facing file.
5. Confirm no absolute path, node name, validation output, cache, prompt, or review artifact is staged.
