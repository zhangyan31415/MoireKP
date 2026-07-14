# Canonical Q Exactification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce target-independent, symmetry-closed model Q coordinates in `kp symm` and make `complete_linear_v2` consume them from the exactified symmetry package.

**Architecture:** Add a small geometry-only canonicalizer that projects raw Q coordinates onto constraints derived from explicit Q maps and certified permutations. Package raw/canonical Q atomically with exactified matrices, then teach the model loader to require that package for v2 while preserving legacy fallback.

**Tech Stack:** Python 3.11, NumPy/SciPy linear algebra, NPZ/JSON artifacts, pytest.

---

### Task 1: Generic constrained Q canonicalizer

**Files:**
- Create: `kp/kp/symmetry/q_canonicalization.py`
- Create: `tests/kp/test_q_canonicalization.py`

**Step 1: Write failing unit tests**

Cover a C3 orbit perturbed at `1e-12`, a reflection, negation, sector exchange,
identity-only input, inconsistent permutations, and excessive correction.
Assertions must verify unchanged ordering/count, reduced closure residual, and
fail-closed behavior.

**Step 2: Verify RED**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_q_canonicalization.py
```

Expected: collection/import failure because the canonicalizer does not exist.

**Step 3: Implement the minimal canonicalizer**

Add immutable result/diagnostic records, explicit action parsing, construction of
the real constraint matrix, SVD-based nearest-point projection, correction
policy, and post-projection closure certification. Do not infer geometry from
operation names.

**Step 4: Verify GREEN**

Run the Task 1 test file and expect all tests to pass.

### Task 2: Package canonical Q with exactified symmetry

**Files:**
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `tests/kp/test_symm_projection.py`
- Modify: `tests/kp/test_canonical_output_layout.py`

**Step 1: Write failing artifact tests**

Require `representations.npz` to contain reserved raw/canonical Q arrays and
metadata with before/after residuals and hashes. Verify reserved keys are not
reported as symmetry operations and projection raw-Q arrays retain their values.

**Step 2: Verify RED**

Run only the new test nodes and confirm failure due to missing arrays/metadata.

**Step 3: Integrate canonicalization after action exactification**

Use exactified `internal_resolved_action` and `model_basis_action` metadata.
Canonicalize after permutations are fixed and before writing the canonical
package. Pass Q arrays directly to the package writer so the canonical output
cleanup cannot delete them.

**Step 4: Verify GREEN**

Run the targeted projection/output-layout tests and expect all to pass.

### Task 3: Make model Q loading artifact-owned

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_configured_model.py`
- Modify: `tests/kp/test_complete_response_basis.py`

**Step 1: Write failing loader tests**

Verify that v2 loads canonical arrays from `representations.npz`, legacy can use
the old file-based fallback, and v2 rejects an old package without canonical Q.
Changing canonical Q must change the response basis hash while changing Heff or
fit indices must not.

**Step 2: Verify RED**

Run the new loader/hash test nodes and confirm the old loader either falls back
or cannot find the reserved arrays.

**Step 3: Implement artifact-owned loading**

Read reserved arrays before manifest file references, validate shapes/order/Q
hashes, and require them for `complete_linear_v2`. Include canonicalization
metadata in model identity and frozen output.

**Step 4: Verify GREEN**

Run the targeted configured-model and response-basis tests.

### Task 4: Real-data and regression validation

**Files:**
- Write local-only reports under: `validation_runs/reports/`
- Do not commit generated outputs.

**Step 1: Run ZrS2 q04 on bigmem003**

Rerun `kp symm`, inspect raw/canonical C3 Q closure, then run the response-basis
compile/model command with the existing k-grid. Confirm forbidden-channel norms
are below their certified floor and no huge noise coefficients appear.

**Step 2: Run independent examples in parallel**

- bigmem001: MgI2 Gamma.
- bigmem002: MgI2 M.
- bigmem003: MoTe2 K after the ZrS2 targeted check.

Use a three-minute timeout per command and do not shorten k-grids.

**Step 3: Run release tests**

Run targeted symmetry/model/export tests, then:

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Record exact pass/fail counts and distinguish unrelated pre-existing dirty-tree
failures.

### Task 5: Commit the isolated implementation

Stage only the canonical-Q source, targeted tests, and approved plan documents.
Do not stage validation outputs or unrelated dirty files. Commit with a detailed
message describing the raw/canonical artifact contract, fail-closed v2 loader,
and validation coverage.

