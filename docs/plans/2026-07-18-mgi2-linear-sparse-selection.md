# MgI2 Gamma Linear Sparse Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Measure q05 and q06 fixed-Schur `3/3` baselines and select the smallest accurate models using only coefficient-linear fitting.

**Architecture:** Create isolated validation configurations that share the same projection, harmonic, and evaluation contract. Run full-vocabulary baselines first, then a deterministic order and closed-term-group pruning ladder with frozen-target linear weighting, and compare every candidate to its shell-specific baseline.

**Tech Stack:** MoireKP CLI, YAML, NumPy, pytest, bigmem002 MKL environment.

---

### Task 1: Create comparable q05 and q06 validation configurations

**Files:**
- Create: `validation_runs/kp_auto_all_materials_20260718/mgi2_q05_fixed_schur_3x3_linear.yaml`
- Create: `validation_runs/kp_auto_all_materials_20260718/mgi2_q06_fixed_schur_3x3_linear.yaml`

1. Copy the verified fixed-Schur q05 construction and set harmonics to `3/3`.
2. Create q06 with the matching q06 input and symmetry paths.
3. Keep maximum orders `10/6/10`, response semantics explicit, and target weighting disabled for baseline runs.
4. Parse both configurations with the MoireKP CLI before long execution.

### Task 2: Generate and verify q05/q06 fixed-Schur projection references

**Files:**
- Create under: `validation_runs/kp_auto_all_materials_20260718/mgi2_fixed_schur_3x3_linear/`

1. Reuse the verified q05 projection only when its identity matches all projection inputs.
2. Run `kp project` and `kp symm` for q06 on bigmem002 in the MKL environment.
3. Verify projection method, matrix dimension, symmetry residuals, and absence of residual processes.

### Task 3: Run the unpruned `3/3` linear baselines

1. Run `kp model` with `model_selection: false` and linear band refinement disabled.
2. Record top-10 RMS/max, all-band RMS/max, overlap, and active family term counts.
3. Treat these results as shell-local guards for all sparse candidates.

### Task 4: Run the coefficient-linear sparse path

1. Enable deterministic family-order candidates under the `10/6/10` ceiling.
2. Evaluate closed-term-group keep fractions including dense points near the accuracy knee.
3. Use a frozen target low-energy projector for the weighted linear solve; do not enable iterative nonlinear spectrum refinement.
4. Export high and low candidates and retain all candidate logs in `validation_runs/`.

### Task 5: Verify and report the Pareto selection

1. Independently recompute errors and subspace overlaps from saved arrays.
2. Verify the high model has mean overlap at least `0.95` and lies on the baseline-relative accuracy plateau.
3. Report q05/q06 baseline and selected high/low Kinect, onsite, intra, inter, total terms, and independent real parameters.
4. Run targeted model-selection tests if source changes become necessary; otherwise keep the experiment free of release-source edits.
