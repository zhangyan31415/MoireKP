# Q-Uniform Generator Gauge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve and diagonalize a common Q-independent finite-generator route block with one shared internal frame, preventing general `U(d)` exactification from turning `C3z` into Q-dependent `+/-I` blocks.

**Architecture:** Detect a certified uniform unitary generator before independent orbit transport.  Build one deterministic cyclotomic eigenframe and apply it to every compatible fiber.  Treat the resulting Q-uniform diagonal action as a hard invariant; apply only symmetry-preserving residual gauges, otherwise keep the uniform exact frame and report that optional `C2` presentation standardization was not applicable.

**Tech Stack:** Python 3.11, NumPy/SciPy linear algebra, existing KP block-route exactification and projection-frame artifacts, pytest.

---

### Task 1: Reproduce the ZrS2 Q-dependent C3 regression

**Files:**
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write the failing test**

Construct a four-dimensional dense finite-order `C3z` route block with exact
spectrum `exp(-i*pi/3), -1, -1, exp(+i*pi/3)`.  Repeat it on every fiber of a
small `C3z/C2/TR` action.  Call `derive_standard_generator_fiber_gauge` and
assert that every final `C3z` route block is identical and diagonal.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_joint_exactification.py \
  -k q_uniform_c3_routes
```

Expected: FAIL because the existing independent orbit frame produces more than
one `C3z` route block.

### Task 2: Build a deterministic common finite-generator frame

**Files:**
- Modify: `kp/kp/symmetry/joint_exactification.py`
- Test: `tests/kp/test_joint_exactification.py`

**Step 1: Add uniform-route detection**

Add a helper that returns a common route block only when every route has the
same dimension and differs from the reference by no more than a
dimension-derived roundoff certification bound.

**Step 2: Add deterministic cyclotomic eigenspaces**

Assign each eigenvalue to the nearest declared cyclotomic root.  Build the
spectral projector for each root and use projected coordinate vectors to
obtain a deterministic basis for repeated eigenspaces.  Order columns by root
exponent and deterministic coordinate pivot.

**Step 3: Apply one frame to all fibers**

Construct `fiber_gauge=(U0,...,U0)`, transform every action semilinearly, and
verify that all `C3z` routes equal the same diagonal algebraic block.

**Step 4: Run the focused test**

Expected: the Q-uniform regression passes and the common route residual is
certified.

### Task 3: Restrict residual standardization

**Files:**
- Modify: `kp/kp/symmetry/joint_exactification.py`
- Test: `tests/kp/test_joint_exactification.py`

**Step 1: Write a preservation regression**

Assert that any residual gauge leaves the shared diagonal `C3z` block
bitwise unchanged on every Q while preserving TR and all presentation
relations.

**Step 2: Verify RED**

Run the focused standard-generator tests and confirm that the current general
orbit path violates the hard invariant.

**Step 3: Implement the restricted path**

For a certified uniform generator, do not enter the unrestricted independent
orbit canonicalization.  Attempt `C2` standardization only when its proposed
gauge reproduces the shared diagonal generator.  Otherwise retain the common
frame, exactify its monomial support/cyclotomic entries, and report
`C2_standardization_status=not_applicable_preserve_uniform_generator`.

**Step 4: Verify GREEN**

Run all standard-generator tests and require zero relation-certification
failure.

### Task 4: Compose the production projection frame

**Files:**
- Modify: `kp/kp/symmetry/projection.py` only if metadata composition requires it
- Modify: `tests/kp/test_symm_projection.py`

**Step 1: Write a U(4) production round-trip test**

Pass the joint artifact through `_derive_projection_basis_frame`.  Verify the
same full unitary transforms Heff-basis operations to the Q-uniform exact
targets and that the artifact records the common generator spectrum and frame
hash.

**Step 2: Run the test and implement only required metadata propagation**

Do not add a second implicit frame.  Reuse the common `fiber_gauge` returned by
the exactification stage.

### Task 5: Focused and release verification

**Files:**
- Test: `tests/kp/test_joint_exactification.py`
- Test: `tests/kp/test_symm_projection.py`

**Step 1: Run focused tests**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_joint_exactification.py \
  tests/kp/test_symm_projection.py \
  tests/kp/test_symmetry_adapted_gauge.py \
  tests/kp/test_exactify_representation.py \
  tests/kp/test_operation_standardization.py \
  tests/kp/test_project_auto_gauge_cli.py
```

**Step 2: Run release tests on bigmem001**

```bash
python -m pytest -q -p no:cacheprovider \
  -m "not slow and not external_data"
```

Keep the known unrelated dirty PtSe2 configuration assertion separate in the
report; do not modify that example merely to make the assertion pass.

### Task 6: Material validation

**Files:**
- External config: `/data/work/zy/software/1.tapw_code/moirekp/kp/configs/zrs2_G/AB_model/3.15/kp/configs/jby/zrs2_3.15_Gamma_q04.yaml`

**Step 1: Run ZrS2 on bigmem001**

Run `kp project` followed by `kp symm` with the current release source on
`PYTHONPATH`.  Audit every `C3z` route block, relation residual, common-frame
certificate, and output hashes.

**Step 2: Run ZrS2 on bigmem002**

Repeat sequentially to avoid shared-output races.  Require bitwise-identical
`heff.npy`, `basis.npz`, `wavefunctions.npz`, and `representations.npz`.

**Step 3: Recheck other material paths**

Run the PtSe2 Gamma, MgI2 Gamma/M, and MoTe2 K targeted symmetry paths needed
to demonstrate that scalar and no-uniform-generator behavior is unchanged.

### Task 7: Review and handoff

**Files:**
- Review: `kp/kp/symmetry/joint_exactification.py`
- Review: `kp/kp/symmetry/projection.py`
- Review: targeted tests and design documentation

**Step 1: Run `git diff --check` and inspect only task-related changes**

Preserve unrelated dirty files and validation outputs.

**Step 2: Request independent code review**

Require all Critical and Important findings to be resolved before declaring
the material validation final.
