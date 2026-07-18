# MgI2 Gamma Common Fit Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Identify one sub-3000-channel fitting method that improves the historical MgI2 Gamma 19Q and 31Q top-10, top-12, and full-spectrum band errors.

**Architecture:** Reuse the frozen historical projection and exactified symmetry artifacts so every run changes only continuum fitting. Run three independent reduced-basis candidates on the three bigmem nodes, collect identically aligned metrics, then iterate only on the failed objective component while preserving the shared method contract.

**Tech Stack:** MoireKP `kp model`, YAML configurations, NumPy metric checks, MKL conda environment, bigmem001/002/003, validation-only artifacts.

---

### Task 1: Freeze inputs and metric contract

**Files:**
- Create: `validation_runs/mgi2_common_fit_search_20260718/manifest.md`
- Create: `validation_runs/mgi2_common_fit_search_20260718/baselines.json`

1. Record the exact 19Q and 31Q Heff, symmetry, Q-set, and k-path sources.
2. Record the historical top-10, top-12, and full RMS/Max values.
3. Verify both frozen legacy models reproduce their recorded baseline metrics.
4. Record global `align=top` and original-Heff comparison as the sole acceptance metric.

### Task 2: Prepare balanced linear candidate

**Files:**
- Create: `validation_runs/mgi2_common_fit_search_20260718/linear_balanced/{19q,31q}.yaml`

1. Copy the frozen reduced 10/6/10, harmonics-4/4 model configuration.
2. Set representative fit indices to `[0, 20, 40, 60]`.
3. Configure full supported-matrix unit weight and top-12 two-sided weight 10.
4. Run 19Q and 31Q sequentially on bigmem001 and preserve real exit codes.
5. Reject immediately if compiled candidate channels exceed 3000.

### Task 3: Prepare guarded nonlinear candidate

**Files:**
- Create: `validation_runs/mgi2_common_fit_search_20260718/nonlinear_guarded/{19q,31q}.yaml`

1. Start from each frozen historical unweighted reduced fit.
2. Enable analytic-Jacobian Gauss-Newton refinement.
3. Select kinetic, onsite, and intralayer real/imaginary components, seven path points, top-12 band loss, matrix weight 0.1, coefficient weight 0.01, and at most eight function evaluations.
4. Enable explicit top-10, top-12, and full-spectrum acceptance windows.
5. Run 19Q and 31Q sequentially on bigmem002.

### Task 4: Prepare hybrid candidate

**Files:**
- Create: `validation_runs/mgi2_common_fit_search_20260718/hybrid/{19q,31q}.yaml`

1. Use the balanced linear objective as the initial coefficient fit.
2. Apply the guarded nonlinear refinement settings from Task 3.
3. Run 19Q and 31Q sequentially on bigmem003.
4. Record whether the guard accepts, scales, or reverts the nonlinear step.

### Task 5: Verify and compare candidates

**Files:**
- Create: `validation_runs/mgi2_common_fit_search_20260718/metrics.json`
- Create: `validation_runs/mgi2_common_fit_search_20260718/report.md`

1. Confirm all six model runs have exit code zero.
2. Compute top-10, top-12, and full RMS/Max against original Heff with global top alignment over all 61 path points.
3. Count generated terms, candidate channels, retained channels, refinement variables, and final nonzero coefficients.
4. Render and inspect each full-Heff band comparison.
5. Accept only a single method that passes the strict two-data-set gate.

### Task 6: Evidence-guided bounded iteration

**Files:**
- Create additional configs below the selected candidate directory only.

1. If linear fitting fails mainly at top-10/top-12, scan two-sided weights 3, 10, and 30 while retaining unit full-matrix floor.
2. If nonlinear fitting is reverted, inspect the exact failed guard window and reduce variable families or line-search step size.
3. If full-spectrum error dominates, increase matrix weight or fit-point density before changing the model space.
4. Re-run both 19Q and 31Q for every surviving method-level change.
5. Stop when one shared method passes or evidence establishes a Pareto frontier with no strict dominator in the bounded search.
