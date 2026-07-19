# Public Nonlinear Vectorized Jacobian Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace repeated sparse Python traversal in public nonlinear fitting with a bounded-memory union-support cache and vectorized Hamiltonian/Jacobian contractions while preserving the exact loss and YAML interface.

**Architecture:** Prepare one immutable evaluator from the fitted complete-response basis, solver-channel indices, linear coefficients, and nonlinear band k-points.  When its union-support design is below 768 MiB, cache the design and use BLAS-backed contractions for every optimizer evaluation; otherwise delegate to the existing sparse implementation.  Keep SciPy least-squares unchanged for the first benchmark so numerical equivalence and backend speed are isolated.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse/dense linear algebra, pytest, existing `CompiledResponseBasis` complete-response runtime, gpuh204 external-data benchmark.

---

### Task 1: Exact vectorized evaluator contract

**Files:**
- Modify: `tests/kp/test_complete_response_basis.py`
- Modify: `kp/kp/model/pipeline.py:10882-10984`

**Step 1: Write the failing dense-oracle test**

Add a test that prepares a toy complete-response basis, selects a nontrivial
channel subset, and compares two coefficient vectors through the proposed
evaluator.  Assert that:

```python
np.testing.assert_allclose(cached_h, basis.hamiltonians(points, full_coefficients))
np.testing.assert_allclose(cached_jacobian, existing_sparse_jacobian)
```

Run the comparison for both `top` and `bottom` windows and include channels
whose matrices have different sparse support.

**Step 2: Run the test to verify RED**

Run:

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py::test_public_band_response_cache_matches_sparse_oracle
```

Expected: FAIL because `_prepare_public_band_response_evaluator` does not
exist.

**Step 3: Implement the minimal evaluator**

In `pipeline.py`, add a private immutable evaluator with:

```python
@dataclass(frozen=True)
class _PublicBandResponseEvaluator:
    basis: Any
    coefficients0: np.ndarray
    channel_indices: np.ndarray
    y0: np.ndarray
    kpoints: np.ndarray
    base_hamiltonians: np.ndarray
    support: np.ndarray
    design_by_k: np.ndarray | None
    backend: str
    cache_bytes: int

    def hamiltonians(self, y: np.ndarray) -> np.ndarray: ...
    def band_jacobian(self, eigenvectors: np.ndarray, selected_masks: np.ndarray) -> np.ndarray: ...
```

Discover union support by scanning the selected channels' sparse coefficient
matrices.  For the fast backend, call the existing
`basis._response_design_on_union_support`, reshape it to
`(Nk, Nsupport, Nchannel)`, and verify its returned support matches the
precomputed support.

Construct Hamiltonian displacements with:

```python
delta_support = np.einsum("ksj,j->ks", design_by_k, y - y0, optimize=True)
```

and scatter them into a copy of the exact linear Hamiltonians.  For each
k-point, construct

```python
projected_rows = (
    np.conjugate(selected_vectors[support_row])
    * selected_vectors[support_col]
).T @ design_by_k[k_index]
```

to obtain every selected-band derivative for every solver channel in one
matrix multiplication.

**Step 4: Run the test to verify GREEN**

Run the Task 1 test.  Expected: PASS with `rtol=1e-11`, `atol=1e-12`.

### Task 2: Cache reuse and memory fallback

**Files:**
- Modify: `tests/kp/test_complete_response_basis.py`
- Modify: `kp/kp/model/pipeline.py`

**Step 1: Write the failing reuse test**

Wrap `_response_design_on_union_support` with a call counter, prepare one
evaluator, and request Hamiltonians/Jacobians for two coefficient vectors.
Assert the compiler was called exactly once.

**Step 2: Write the failing memory-policy test**

Prepare the evaluator with a deliberately tiny internal limit.  Assert:

```python
assert evaluator.backend == "sparse_low_memory"
assert evaluator.design_by_k is None
```

and verify the fallback Hamiltonian/Jacobian still match the oracle.

**Step 3: Run both tests to verify RED**

Expected: FAIL until the limit and fallback are implemented.

**Step 4: Implement the 768 MiB automatic policy**

Use an internal constant:

```python
_PUBLIC_NONLINEAR_CACHE_LIMIT_BYTES = 768 * 1024**2
```

Estimate `Nk * Nsupport * Nchannel * sizeof(complex128)` before allocating the
design.  Select `union_support_vectorized` below the limit and
`sparse_low_memory` above it.  The fallback calls the pre-existing
`basis.hamiltonians` and `_compiled_response_band_jacobian` methods.

**Step 5: Run Task 1 and Task 2 tests**

Expected: all PASS.

### Task 3: Integrate one evaluator per nonlinear run

**Files:**
- Modify: `tests/kp/test_complete_response_basis.py`
- Modify: `kp/kp/model/pipeline.py:10987-11160`

**Step 1: Extend the nonlinear initialization test**

Patch the evaluator preparation function and assert it is called once before
SciPy starts.  Call the objective and Jacobian repeatedly inside the fake
optimizer and assert they reuse that same evaluator.

**Step 2: Run the test to verify RED**

Expected: FAIL because `_refine_public_nonlinear_complete_response` still calls
the standalone sparse residual helper on every new coefficient vector.

**Step 3: Integrate the evaluator**

Prepare target eigenvalues and degeneracy-safe selected masks once.  Prepare
the response evaluator once.  In the objective/Jacobian cache, use its
Hamiltonian and Jacobian methods; retain `_public_band_residual_and_jacobian`
as the low-level oracle and fallback-compatible API.

Add result metadata:

```python
"band_response_backend": evaluator.backend,
"band_response_support_entries": int(evaluator.support.size),
"band_response_cache_bytes": int(evaluator.cache_bytes),
"band_response_cache_seconds": float(cache_seconds),
```

Print one concise terminal line before optimization.

**Step 4: Run public nonlinear tests**

Run:

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py -k 'public_nonlinear or public_band_response'
```

Expected: all selected tests PASS.

### Task 4: Targeted regression suite

**Files:**
- Test: `tests/kp/test_complete_response_basis.py`
- Test: `tests/kp/test_configured_model.py`
- Test: `tests/kp/test_standalone_export.py`
- Test: `tests/kp/test_reporting.py`

**Step 1: Run complete-response and configured-model tests**

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py tests/kp/test_configured_model.py
```

Expected: PASS except no pre-existing unrelated failures in these two files.

**Step 2: Run export/reporting tests**

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_standalone_export.py tests/kp/test_reporting.py
```

Expected: PASS.

### Task 5: PtSe2 gpuh204 equivalence and performance benchmark

**Files:**
- Read: `examples/autoproject/ptse2_7.34/configs/gamma_spinful_q04.yaml`
- Output only: `validation_runs/public_nonlinear_vectorized_20260720/`

**Step 1: Preserve the current baseline metadata**

Copy only a compact JSON summary into `validation_runs`, recording the current
30-evaluation loss, top-6/top-20/all-38 RMS/Max, and observed runtime.  Do not
place logs in release-facing directories.

**Step 2: Run the unchanged configuration on gpuh204**

```bash
ssh gpuh204 'cd /data/work/zy/software/1.tapw_code/moirekp-release/examples/autoproject/ptse2_7.34/configs && \
  /usr/bin/time -p /data/home/zy/mambaforge/envs/moirekp/bin/kp model -c gamma_spinful_q04.yaml'
```

Expected terminal evidence:

```text
band response backend  union_support_vectorized
nonlinear loss evaluation 30/30
```

**Step 3: Compare numerical results**

Assert initial/final loss and band errors agree with the baseline within a
tolerance justified by the optimizer path (`rtol <= 1e-8` for the loss and
sub-micro-eV differences for eigenvalues).

**Step 4: Apply the performance gate**

Require at least 3x speedup of the nonlinear stage.  Target 17--30 seconds.  If
the gate is not met, stop and report the vectorized backend timing before
changing optimizer behavior; then make a separate approved plan for automatic
Gauss--Newton.

### Task 6: Final release-facing verification

**Files:**
- Verify only: repository default tests

**Step 1: Run diff and syntax checks**

```bash
git diff --check -- kp/kp/model/pipeline.py tests/kp/test_complete_response_basis.py
/data/home/zy/mambaforge/envs/moirekp/bin/python -m py_compile kp/kp/model/pipeline.py
```

Expected: exit 0.

**Step 2: Run the default release suite**

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  -m "not slow and not external_data"
```

Report the exact pass/fail counts and classify only failures supported by full
tracebacks.

**Step 3: Do not stage overlapping dirty source files automatically**

Because `pipeline.py` and its tests had pre-existing overlapping changes, leave
the implementation diff for explicit user review unless a clean patch boundary
can be proven.  Never stage the whole dirty files merely to create a commit.

