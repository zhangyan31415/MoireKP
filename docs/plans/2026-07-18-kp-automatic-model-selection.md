# KP Automatic Model Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically choose family-specific k-polynomial orders and a minimal symmetry-closed term set for weighted low-energy KP model fitting.

**Architecture:** Add target-independent selection primitives and blocked validation first, then integrate an opt-in family-order candidate scan around the existing configured model pipeline.  Candidates rebuild their complete response vocabulary, use fixed target-only weights, and are ranked by a one-standard-error quality plateau followed by independent real parameter count.  Term ablation and a joint order correction sweep operate only after a complete family model exists.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyYAML, pytest, existing `kp.model.pipeline` and `complete_linear_v2` response-basis APIs.

---

### Task 1: Pure selection records and one-standard-error rule

**Files:**
- Create: `kp/kp/model/model_selection.py`
- Create: `tests/kp/test_model_selection.py`

**Steps:**

1. Write failing tests for deterministic candidate ordering, independent-real-parameter complexity, `PASS` versus `WARN_BEST_AVAILABLE`, and the one-standard-error choice of the simplest near-best model.
2. Run `python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection.py` and confirm the module is absent.
3. Implement immutable candidate/fold metric records and a lexicographic selector: finite/certified candidates, quality plateau, minimum parameter count, minimum family-order tuple, minimum maximum error.
4. Rerun the focused tests and require all to pass.

### Task 2: Blocked k-path folds and fixed target weights

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `tests/kp/test_model_selection.py`

**Steps:**

1. Add failing tests showing adjacent path points form contiguous validation blocks, duplicate junction points remain in one fold, every point is held out once, and fixed band weights have mean one and degeneracy-safe boundaries.
2. Implement deterministic blocked folds from path geometry and a weighted band metric helper shared by every candidate.
3. Add subspace principal-angle overlap metrics and validate exact-degenerate test fixtures.
4. Run the focused tests.

### Task 3: Family-order vocabulary configuration

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_model_selection.py`
- Modify: `tests/kp/test_configured_model.py`

**Steps:**

1. Write failing tests for parsing opt-in `fit.model_selection`, generating `Kinect`, `intra`, and `inter` order ladders, and updating both `max_order` and template-local `max_order` without mutating the source config.
2. Implement canonical family aliases and immutable candidate config cloning.
3. Require each `complete_linear_v2` candidate to rebuild or load a cache keyed by its exact vocabulary.
4. Run the two focused test files.

### Task 4: Centered kinetic validation and identifiability audit

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `kp/kp/model/response_basis.py`
- Modify: `tests/kp/test_model_selection.py`

**Steps:**

1. Add synthetic failing tests where an omitted constant onsite term must not force a higher kinetic order and where overlapping kinetic/intralayer response spaces are deferred to a joint stage.
2. Implement centered response/target evaluation at a deterministic reference k point.
3. Implement response-subspace principal correlations and the defer-to-joint decision.
4. Verify that the selector recovers the known kinetic order in the synthetic fixture.

### Task 5: Staged candidate runner

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_model_selection.py`
- Modify: `tests/kp/test_configured_model.py`

**Steps:**

1. Add failing small-model tests for `Kinect -> intra -> inter`, verifying that earlier family coefficients are jointly refitted rather than frozen.
2. Implement cached candidate execution and fold aggregation.
3. Use centered target-spectral response loss before the complete family set and weighted band/subspace metrics after interlayer terms are present.
4. Write `model_selection.json` plus a compact summary and preserve explicit manual-order behavior when selection is disabled.
5. Run focused tests.

### Task 6: Symmetry-closed ablation and correction sweep

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `tests/kp/test_model_selection.py`

**Steps:**

1. Add failing tests showing real/imag and adjoint-related channels are removed as one group, correlated groups are judged after refitting, and a forward-stage order can be lowered by the correction sweep.
2. Implement normalized-amplitude screening, leave-one-group-out refit importance, stability counts, and bounded joint correction sweeps.
3. Require the final reported parameter count to equal the independent nonzero real response dimension.
4. Run focused tests.

### Task 7: Bilayer MoTe2 K and MgI2 M external-data trials

**Files:**
- Create ignored outputs under: `validation_runs/kp_auto_model_selection_<timestamp>/`
- Do not modify example source configs during the trial.

**Steps:**

1. Resolve usable projection/model inputs under `examples/mote2_3.89/` and `examples/mgi2_3.89/`; copy configs into the validation directory and make paths explicit there.
2. Run manual configured baselines and the automatic selector on bigmem through direct SSH or Slurm as appropriate to runtime.
3. Record selected family orders, authored/compiled/nonzero parameter counts, weighted RMS/max errors, overlap, guards, timings, and memory.
4. Plot candidate quality versus parameter count and per-family order plateaus.
5. Classify each material as `PASS`, `WARN_BEST_AVAILABLE`, or `FAIL` without inventing missing metrics.

### Task 8: Release verification

**Files:**
- Modify user-facing docs only after the interface and validation are stable.

**Steps:**

1. Run `python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection.py tests/kp/test_configured_model.py`.
2. Run `python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"`.
3. Run `git diff --check` and compile modified Python files.
4. Confirm no validation output, cache, egg-info, review material, machine path, or node name is staged.
5. Report remaining performance or quality limitations before proposing merge or PR integration.
