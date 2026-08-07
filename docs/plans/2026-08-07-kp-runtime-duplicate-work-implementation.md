# KP Runtime Duplicate-Work Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove proven duplicate eigensystem and cache-key work, then measure finite-degree owner block sizes without changing model export.

**Architecture:** Reuse values already computed by the harmonic evaluator, make the persistent response-basis cache key-based, and expose a pure read-only summary of existing exact owner actions. Production compiler routing remains unchanged.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, pytest.

---

### Task 1: Reuse the harmonic candidate eigensystem

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_configured_model.py`

1. Add a failing test that supplies the target eigensystem, forbids `np.linalg.eigvalsh`, and evaluates one harmonic candidate.
2. Run the focused test and confirm it fails inside `_band_refinement_metrics`.
3. Add an optional precomputed-eigenvalues argument to `_band_refinement_metrics` and pass `candidate_eig` to both metric calls.
4. Run the focused harmonic tests.

### Task 2: Compute the response-basis cache key once

**Files:**
- Modify: `kp/kp/model/response_basis_cache.py`
- Modify: `kp/kp/model/response_basis.py`
- Test: `tests/kp/test_response_basis_cache.py`
- Test: `tests/kp/test_complete_response_basis.py`

1. Add a failing test for a key-based persistent-cache call that must not canonicalize the compiler input again.
2. Run it and confirm the current payload-based API fails.
3. Change persistent-cache operations to accept a validated SHA-256 key and update the model compiler to call `target_independent_basis_key` once.
4. Update existing cache tests to compute their key once at the call boundary.
5. Add and run a model-level call-count test for cold and disk-hit paths.

### Task 3: Add the finite-degree action-block diagnostic

**Files:**
- Modify: `kp/kp/model/response_basis_local_fixed.py`
- Modify: `kp/kp/model/response_basis_graded.py`
- Test: `tests/kp/test_response_basis_local_fixed.py`

1. Add a failing test with known action components and one degree-lowering edge.
2. Run it and confirm the diagnostic is absent.
3. Implement a pure summary over certified action blocks: closure owner count, component dimensions, `b_max`, generator/H nonzero counts, and degree-lowering edge counts.
4. Attach the summary to existing local compiler artifacts without changing backend selection.
5. Run focused local/graded response tests.

### Task 4: Verify runtime behavior

**Files:**
- No production-file changes.

1. Run the focused harmonic, cache, local-fixed, and graded test files on `bigmem001` and `bigmem003`.
2. Run all nine non-PtSe2 `kp model` examples, retaining their existing output/export configuration.
3. Compare complete command wall time and phase timing against the recorded baseline.
