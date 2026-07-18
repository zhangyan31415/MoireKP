# KP Four-Profile Automatic Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend automatic KP model selection to export linear/nonlinear high and low profiles, then validate all four profiles for MgI2 Gamma 19Q and 31Q.

**Architecture:** Reuse the existing staged linear vocabulary search as the common candidate frontier. Refine selected frontier candidates with the nonlinear band solver, score linear and nonlinear candidates with shared target-only validation data, and apply separate high-accuracy and compact-primary selection policies.

**Tech Stack:** Python 3.11, NumPy/SciPy, pytest, existing KP response compiler and nonlinear Gauss-Newton solver, standalone NumPy model exporter.

---

### Task 1: Define four-profile selection records

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Test: `tests/kp/test_model_selection.py`

**Step 1: Write the failing test**

Add candidates tagged by solver family and assert that selection returns
`linear.high`, `linear.low`, `nonlinear.high`, and `nonlinear.low`. Assert that
nonlinear-low chooses the fewest independent parameters on the allowed primary-error
plateau even when its expanded-window error is worse.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection.py -k four_profile`

Expected: FAIL because no four-profile result type or selector exists.

**Step 3: Implement the minimal selector**

Add a solver-family field to candidate identity without changing existing candidate
defaults. Compose the existing high/low policy independently over linear and nonlinear
candidate subsets. Add harmonic/group/order complexity tie breakers for low while
keeping expanded/all-band metrics diagnostic-only.

**Step 4: Run the focused tests**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection.py`

Expected: PASS.

### Task 2: Generate nonlinear frontier candidates

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write the failing pipeline test**

Build a small configured fixture with nonlinear refinement enabled. Assert that several
linear frontier vocabularies are refined independently and that each nonlinear candidate
preserves its harmonic/order/group provenance.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py -k nonlinear_frontier`

Expected: FAIL because only the selected full candidate is refined.

**Step 3: Implement frontier refinement**

Select a bounded set containing linear-high, linear-low, and distinct Pareto frontier
points. Materialize each vocabulary, run nonlinear refinement with identical primary
weights and fit indices, then evaluate every candidate on the full validation path.
Reject candidates with more than 3000 independent nonlinear variables.

**Step 4: Run focused configured-model tests**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py -k 'nonlinear or model_selection'`

Expected: PASS.

### Task 3: Export and report four runnable models

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Modify: `kp/kp/model/export.py`
- Test: `tests/kp/test_configured_model.py`
- Test: `tests/kp/test_standalone_export.py`

**Step 1: Write failing output-layout tests**

Assert all four profile directories contain runnable standalone exports and that the
root JSON/Markdown report records solver family, orders, harmonic support, active terms,
independent parameters, primary/expanded errors, overlap, and warning status.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py tests/kp/test_standalone_export.py -k 'profile or selection'`

Expected: FAIL on the missing nonlinear profile directories and report fields.

**Step 3: Implement output materialization**

Write `linear/{high,low}` and `nonlinear/{high,low}` atomically. Preserve the selected
vocabulary and nonlinear coefficients in each standalone export. Keep the compatibility
result bound to nonlinear-high when present.

**Step 4: Run output tests**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py tests/kp/test_standalone_export.py -k 'profile or selection'`

Expected: PASS.

### Task 4: Validate MgI2 Gamma 19Q and 31Q

**Files:**
- Create: `validation_runs/mgi2_four_profiles_20260718/`

**Step 1: Freeze shared validation settings**

Use top-10 as the primary target, target-only weights, the full 61-point validation
path, cluster-completed overlap diagnostics, and the same nonlinear solver settings for
both Q sets. Search only candidates below 3000 nonlinear variables.

**Step 2: Run both configured pipelines on bigmem**

Run the 19Q and 31Q cases using the `moirekp` MKL environment. Preserve real shell exit
status and place logs and generated artifacts only below the validation directory.

**Step 3: Verify every standalone model**

For all eight outputs, run `evaluate.py`, reproduce saved bands, and calculate primary
RMS/max error, projector overlap, worst principal-direction overlap, symmetry residual,
orders, harmonic counts, active groups, and independent parameters.

**Step 4: Compare the complexity frontier**

Confirm each low profile is no more complex than its high counterpart and report any
quality gate as `PASS`, `WARN_BEST_AVAILABLE`, or `FAIL`. Plot Heff with all four profiles
for both Q sets.

### Task 5: Run release verification

**Files:**
- Test: `tests/kp/`

**Step 1: Run focused KP tests**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection.py tests/kp/test_configured_model.py tests/kp/test_standalone_export.py`

Expected: PASS.

**Step 2: Run default release tests**

Run: `python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"`

Expected: PASS.

**Step 3: Audit the worktree**

Run: `git status --short` and confirm no validation artifact, node log, cache, or external
data has entered the release-facing change set.
