# KP Standard Generator Fiber Gauge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a certified common U(1) fiber gauge that preserves primary and antiunitary generators while putting unitary `C2` cycles into their nearest uniform standard form.

**Architecture:** Keep closest-cyclotomic exactification unchanged, then derive a second scalar common gauge from exact route cycles. Compose both gauges in the projection frame, materialize exact target matrices, and reuse joint artifact recertification after the shared basis transformation.

**Tech Stack:** Python 3.11, NumPy complex linear algebra, pytest, existing KP joint exactification and projection-frame artifacts.

---

### Task 1: Specify standard scalar cycle behavior

**Files:**
- Modify: `tests/kp/test_joint_exactification.py`
- Modify: `kp/kp/symmetry/joint_exactification.py`

**Step 1: Write the failing tests**

Add scalar fixtures for PtSe2-like `C3z/TR/C2`, MgI2-M-like `TR/C2`, an
already-standard representation, a presentation without `C2`, and a
non-scalar representation.

**Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_joint_exactification.py -k standard_generator
```

Expected: collection or assertion failure because the standard-generator API
does not exist.

**Step 3: Implement the minimal scalar solver**

Add a result artifact and a function that:

1. identifies unitary `C2`;
2. computes cycle holonomies and the nearest uniform roots;
3. builds preservation equations for every other generator;
4. solves the minimum-norm common semilinear fiber gauge;
5. materializes deterministic exact roots and recertifies all relations;
6. returns `not_applicable` for missing `C2` or non-scalar fibers.

**Step 4: Run tests to verify GREEN**

Run the focused command above and expect all selected tests to pass.

### Task 2: Compose the production frame

**Files:**
- Modify: `tests/kp/test_symm_projection.py`
- Modify: `tests/kp/test_symmetry_adapted_gauge.py`
- Modify: `kp/kp/symmetry/projection.py`

**Step 1: Write a failing frame-composition test**

Assert that the stored full unitary equals the closest-cyclotomic gauge times
the standard-generator gauge and that both reports are present.

**Step 2: Run the focused test to verify RED**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_symm_projection.py \
  tests/kp/test_symmetry_adapted_gauge.py \
  -k 'standard_generator or closest_cyclotomic'
```

**Step 3: Implement frame composition**

Apply the second gauge to the closest exact actions, compose the diagonal
fiber gauges, store both reports, and provide the final canonical matrices to
the existing project/symm synchronization path.

**Step 4: Run the focused tests to verify GREEN**

Re-run the command above and expect all selected tests to pass.

### Task 3: Regression verification

**Files:**
- Test: `tests/kp/test_joint_exactification.py`
- Test: `tests/kp/test_symm_projection.py`
- Test: `tests/kp/test_project_auto_gauge_cli.py`

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_joint_exactification.py \
  tests/kp/test_symm_projection.py \
  tests/kp/test_project_auto_gauge_cli.py
```

Expected: zero failures.

### Task 4: Material validation on bigmem002

Rerun `kp project` and `kp symm` for PtSe2 Gamma Q4 and MgI2 M1. Audit every
nonzero representation entry. Check MgI2 Gamma, MoTe2 K1, and ZrS2 Gamma as
unchanged-control cases. Store concise reports only under `validation_runs/`.

### Task 5: Release verification

Run on `bigmem002`:

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Expected: zero failures, with no validation outputs added to release-facing
paths.
