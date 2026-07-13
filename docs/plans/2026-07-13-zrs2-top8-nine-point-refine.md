# ZrS2 Top-Eight At-Most-Ten-Point Refine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce and validate on bigmem002 a symmetry-preserving top-eight refinement that uses at most ten path points and passes the agreed full-path energy and projector-overlap gates.

**Architecture:** Keep the existing symmetry-exactified RBF correction fixed and refine only exact polynomial term coefficients.  Assemble fixed-reference top-eight `PP` and neighboring-eight `PQ` response rows at ten k points, solve a small grid of normalized regularized linear systems, and select only on the 51 held-out path rows.  All implementation and outputs remain under `validation_runs/`.

**Tech Stack:** Python 3.11, NumPy/SciPy linear algebra, Matplotlib, existing MoireKP direct term-response helpers, pytest.

---

### Task 1: Add validation-only metric and response-system tests

**Files:**
- Create: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/test_top8_response_refine.py`
- Create: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_response_refine.py`

**Step 1: Write failing tests**

Test that the metric helper reports top-two, top-four, and top-eight projector
overlaps and raw top-eight energy errors.  Test a synthetic block Hamiltonian
where a `PP` response removes an internal top-space error and a `PQ` response
removes leakage.  Test that more than ten refine indices raises `ValueError`.

**Step 2: Run the tests and confirm failure**

Run:

```bash
PYTHONPATH=kp:tapw /data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q \
  validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/test_top8_response_refine.py
```

Expected: collection or import failure because the implementation helpers do
not exist.

**Step 3: Implement minimal helpers**

Implement:

- `top_window_metrics(model, target, indices=None)`;
- `assemble_pp_pq_system(candidate, target, responses, indices, pq_weight)`;
- `normalized_regularized_solve(design, rhs, ridge)`;
- `validate_refine_indices(indices, n_kpoints, maximum=10)`.

Use target eigenvectors to define top-eight `P` and the next eight lower states
as `Q`.  Return column norms, numerical rank, condition number, and coefficient
deltas from the solver.

**Step 4: Re-run the tests**

Expected: all validation-only unit tests pass.

### Task 2: Build the nine-point response basis and linear candidate grid

**Files:**
- Modify: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_response_refine.py`
- Create: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_response_refine.json`
- Create: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_response_refine_best_hamiltonians.npy`

**Step 1: Rebuild and verify the polynomial baseline**

Rebuild the order-eight model fitted at indices `[0, 40]`.  Assert its full-path
Hamiltonian matches `fixed_schur_order8_linear_hamiltonians.npy` to numerical
precision before constructing any response.

**Step 2: Construct direct symmetry-exact term responses**

Use refine indices `[0, 10, 12, 20, 21, 30, 40, 46, 47, 50]`.  Select active real
and imaginary responses from `Kinect`, `Onsite`, `intra`, and `inter` terms.
Drop only numerically zero columns and record term/component metadata.

**Step 3: Solve a bounded grid**

Solve combinations of:

```text
PQ weight: 0.25, 0.5, 1.0, 2.0
ridge:     1e-4, 1e-3, 1e-2, 1e-1
```

Apply coefficient steps with line-search factors `[1.0, 0.5, 0.25, 0.1]`.
Recompute full-path corrections directly from the exact term responses.

**Step 4: Select on held-out rows**

Reject candidates that violate any full-path overlap or gap gate.  Rank the
remaining candidates by held-out top-eight RMS, then held-out maximum error.
If none passes, retain the Pareto-best candidate and record every failed gate.

**Step 5: Run on bigmem002**

Run:

```bash
ssh bigmem002 'cd /data/work/zy/software/1.tapw_code/moirekp-release && \
  PYTHONPATH=kp:tapw /data/home/zy/mambaforge/envs/moirekp/bin/python \
  validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_response_refine.py'
```

Expected: JSON and best-Hamiltonian artifacts identify `host: bigmem002`, use
nine refine rows, and contain train, holdout, and full-path metrics.

### Task 3: Conditional damped nonlinear polish

**Files:**
- Modify: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_response_refine.py`
- Modify: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_response_refine.json`

**Step 1: Check the linear gates**

Skip nonlinear work if the best linear candidate already passes every gate.

**Step 2: Polish only if needed**

Restrict optimization to well-conditioned singular directions of the linear
PP+PQ design.  Perform at most six damped Gauss-Newton iterations using the
same ten refine points.  Include top-eight energy residuals and top-eight
principal-angle residuals, and reject line-search steps that worsen the
held-out ranking score.

**Step 3: Re-run validation-only tests and the bigmem002 experiment**

Expected: the polished result is retained only if its held-out metrics improve
and no acceptance gate regresses.

### Task 4: Plot and verify the selected result

**Files:**
- Create: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/plot_top8_refine.py`
- Create: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_refine_comparison.png`
- Create: `validation_runs/fixed_schur_hypothesis_zrs2_q04_20260713_181500/top8_refine_comparison.pdf`

**Step 1: Plot full-path bands, errors, and overlaps**

Compare fixed-Schur Heff, the unrefined exactified candidate, and the selected
refined candidate.  Show the top eight bands, all top-eight energy errors, and
top-two/top-four/top-eight projector overlaps.  Mark the ten refine indices.

**Step 2: Inspect the rendered PNG**

Verify labels, line visibility, and that no error or overlap excursion is hidden
by axis limits.

**Step 3: Run final assertions on bigmem002**

Assert host identity, refine-point count, saved array shapes, Hermiticity, all
five numerical gates, and nonempty PNG/PDF outputs.  If a gate fails, report the
measured limitation rather than claiming completion.
