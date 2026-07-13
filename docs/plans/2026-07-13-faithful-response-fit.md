# Faithful Response Fit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Demonstrate on bigmem003 that a symmetry-preserving `PP + PQ` linear response fit can meet the ZrS2 Gamma q04 top-four energy and top-two overlap targets without nonlinear refinement.

**Architecture:** First implement a validation-only prototype that consumes existing Heff and term responses and writes only under `validation_runs/`.  Compare it against the original order-six fit and order-eight/order-ten full-matrix fits.  Integrate into `kp model` only after the held-out metrics demonstrate a clear improvement.

**Tech Stack:** Python 3.11, NumPy/SciPy least squares, existing MoireKP model builder and term-response helpers, pytest.

---

### Task 1: Validation-only response-fit prototype

**Files:**
- Create: `validation_runs/faithful_response_fit_zrs2_q04_<timestamp>/prototype.py`
- Create: `validation_runs/faithful_response_fit_zrs2_q04_<timestamp>/run.json`

1. Verify bigmem003 uses `/data/home/zy/mambaforge/envs/moirekp/bin/python` and the release checkout.
2. Build order-eight and order-ten interlayer candidate models without nonlinear refinement.
3. Fit their initial coefficients on indices `[0, 10, 20, 30, 40, 50, 60]`.
4. Select nonzero active real/imag term-response components.
5. Construct fixed-reference `PP` and inverse-gap-weighted `PQ` response rows for top four versus the next eight lower bands.
6. Add response-normalized trust rows around the initial coefficients.
7. Sweep a small fixed grid of `PQ` and trust weights.
8. Record train, holdout, and full-path top-four errors, top-two overlap, boundary gap, and matrix residuals.

Expected outcome: at least one candidate materially improves overlap without violating the band and gap guards.

### Task 2: Review the prototype result

**Files:**
- Create: `validation_runs/faithful_response_fit_zrs2_q04_<timestamp>/summary.md`

1. Rank candidates using held-out metrics only.
2. Reject candidates with top-four max error above 5 meV or a closed boundary gap.
3. Compare the selected candidate with original order-six, order-eight, order-ten, and MgI2 controls.
4. Decide whether production integration is justified or whether the candidate vocabulary must be enriched first.

### Task 3: Production tests, conditional on prototype success

**Files:**
- Modify: `tests/kp/test_configured_model.py`
- Modify: `kp/kp/model/pipeline.py`

1. Write a failing unit test showing that a pure `PP` fit leaves an off-block leakage response while `PP + PQ` removes it.
2. Run the targeted test and confirm the expected failure.
3. Add a configuration parser for the cross-subspace linear loss, disabled by default.
4. Extend `linear_low_subspace` to append fixed-reference `PQ` rows with inverse-gap weighting.
5. Add reporting for row counts, weights, rank, condition number, and train/holdout metrics.
6. Run targeted configured-model tests.
7. Run the default non-slow, non-external-data release test suite.

### Task 4: Response-aware model reduction, conditional on a faithful model

**Files:**
- Modify: `tests/kp/test_configured_model.py`
- Modify: `kp/kp/model/pipeline.py`

1. Write failing tests for symmetry-closed real/imag term grouping.
2. Write failing tests for exact Gram-block delete-and-refit costs.
3. Implement backward group ablation with held-out guards.
4. Export separate faithful and interpretable model packages and a Pareto report.
5. Verify topology diagnostics never silently consume the interpretable model.

