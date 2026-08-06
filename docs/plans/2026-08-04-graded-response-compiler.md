# Graded Response Compiler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the generate-all global rank bottleneck with an exact structural-orbit and graded-momentum compiler while preserving the complete case-derived response span.

**Architecture:** Extract the sector/orbital/harmonic support action once, represent degree-\(d\) momentum polynomials in the \(d+1\)-dimensional homogeneous basis, and solve symmetry/adjoint fixed spaces per structural-orbit/degree block.  Apply exact lower-triangular Q-center translation and materialize only retained channels; keep the existing symbolic compiler as an oracle and certified fallback.

**Tech Stack:** Python, NumPy, SciPy sparse linear algebra, pytest, existing factorized/joint-route symmetry artifacts.

---

### Task 1: Lock the mathematical contract with failing tests

**Files:**
- Modify: `tests/kp/test_response_basis_symmetry_first.py`
- Modify: `tests/kp/test_complete_response_basis.py`

**Step 1: Add the cubic irrep regression**

Construct a toy C3/C2 group with a C2-odd internal matrix and assert that the
graded compiler retains no matching degree-0/1/2 invariant but retains
\(\operatorname{Im}(w^3)A\) at degree three.

**Step 2: Add a non-closed seed-label regression**

Use the existing Q-dependent-phase fixture and assert equality of factorized
and joint-route span projectors.  Do not assert a backend name.

**Step 3: Add complete-envelope equivalence**

For a two-sector/two-orbital toy, compare graded and existing symbolic compilers:

```python
old = compile_symbolic_atom_candidate_group(...)
new = compile_graded_candidate_group(...)
np.testing.assert_allclose(projector(new), projector(old), atol=1e-10)
assert new.rank == old.rank
```

**Step 4: Run tests and verify failure**

Run:

```bash
env PYTHONPATH=$PWD/kp:$PWD/tapw python -m pytest -q \
  tests/kp/test_response_basis_symmetry_first.py \
  tests/kp/test_complete_response_basis.py -k 'graded or cubic or non_closed'
```

Expected: FAIL because the graded API does not exist and the current model-level
test still requires joint-route fallback.

**Step 5: Commit only the tests**

```bash
git add tests/kp/test_response_basis_symmetry_first.py tests/kp/test_complete_response_basis.py
git commit -m "test(kp): specify graded response compilation"
```

### Task 2: Add structural and graded representation primitives

**Files:**
- Create: `kp/kp/model/response_basis_graded.py`
- Test: `tests/kp/test_response_basis_symmetry_first.py`

**Step 1: Define immutable structural records**

Add dataclasses for structural support keys, closed structural orbits, degree
blocks, and certification artifacts.  Canonical keys include sector pair,
harmonic/Q-pair support, orbital pair, and adjoint provenance.  Family is
retained as provenance and a source of the configured order bound, but is not
part of physical structural identity.

**Step 2: Implement homogeneous momentum actions**

For the certified two-dimensional linear action \(K_g\), construct
\(M_g^{(d)}=\operatorname{Sym}^d(K_g)\) in
\((w^d,w^{d-1}\bar w,\ldots,\bar w^d)\).  Verify composition against every
finite-group word and propagate the existing action error bounds.

**Step 3: Implement exact triangular center translation**

Build sparse binomial maps between Q-centered and global monomials and certify
invertibility of the diagonal blocks.

**Step 4: Run focused primitive tests**

Run:

```bash
env PYTHONPATH=$PWD/kp:$PWD/tapw python -m pytest -q \
  tests/kp/test_response_basis_symmetry_first.py -k 'homogeneous or translation or cubic'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/model/response_basis_graded.py tests/kp/test_response_basis_symmetry_first.py
git commit -m "feat(kp): add graded response representations"
```

### Task 3: Compile structural support orbits

**Files:**
- Modify: `kp/kp/model/response_basis_graded.py`
- Test: `tests/kp/test_response_basis_symmetry_first.py`

**Step 1: Build structural fibers from raw seeds**

Collapse raw polynomial seeds that share family, sector/orbital endpoints, and
actual harmonic/Q-pair support.  Keep the coefficient fiber required for
Q-dependent phases instead of assuming a seed-label permutation.

**Step 2: Close fibers under actual actions and adjoint**

Use certified factorized actions when available and certified joint-route
actions otherwise.  Compute connected structural orbits from nonzero action
blocks and certify full coverage and disjointness.

**Step 3: Add a production-shape count test**

For an MgI2-like fixture, derive the top-level structural-support count from
its actual single-sign harmonic records and assert it against an independent
physical-key oracle.  Do not encode a fixed material count, orbital template,
or harmonic-sign template.

**Step 4: Run tests**

Expected: structural-orbit tests and existing symmetry-first tests pass.

**Step 5: Commit**

```bash
git add kp/kp/model/response_basis_graded.py tests/kp/test_response_basis_symmetry_first.py
git commit -m "feat(kp): compile structural response orbits"
```

### Task 4: Solve graded fixed spaces and materialize retained channels

**Files:**
- Modify: `kp/kp/model/response_basis_graded.py`
- Modify: `kp/kp/model/response_basis_symmetry_first.py`
- Test: `tests/kp/test_response_basis_symmetry_first.py`

**Step 1: Form per-orbit/per-degree actions**

Construct realified unitary and antiunitary actions
\(U_{g,o}^{(d)}=M_g^{(d)}\otimes A_{g,o}\).

**Step 2: Solve each fixed space independently**

Reuse the certified null/rank policy on small blocks.  Record per-degree ranks,
condition bounds, and selected deterministic coordinates.

**Step 3: Translate and materialize only retained channels**

Apply the exact triangular center map, convert retained blocks to existing
`ResponseChannel` objects, and run the existing physical covariance and
Hermiticity checks.

**Step 4: Compare with the oracle**

Run all new toy projector tests and the existing symbolic/oracle suite.

**Step 5: Commit**

```bash
git add kp/kp/model/response_basis_graded.py \
  kp/kp/model/response_basis_symmetry_first.py \
  tests/kp/test_response_basis_symmetry_first.py
git commit -m "feat(kp): solve graded response fixed spaces"
```

### Task 5: Integrate backend selection and remove the wrong fallback gate

**Files:**
- Modify: `kp/kp/model/response_basis.py`
- Modify: `tests/kp/test_complete_response_basis.py`

**Step 1: Route complete case-derived groups to the graded compiler**

Prefer certified factorized actions.  Use joint-route actions only when
factorized actions are absent or their action certificate fails.  Seed-label
non-closure must not select a backend.

**Step 2: Delete `_factorized_seed_vocabulary_is_closed`**

Remove the uncommitted helper and its call.  Replace the backend-name test with
the projector-equivalence test from Task 1.

**Step 3: Keep explicit oracle fallback**

Fallback only on a typed graded-certification exception and report the exact
failed certificate.  Do not catch arbitrary exceptions.

**Step 4: Run response-basis integration tests**

Run:

```bash
env PYTHONPATH=$PWD/kp:$PWD/tapw python -m pytest -q \
  tests/kp/test_response_basis_symmetry_first.py \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_cache.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/model/response_basis.py tests/kp/test_complete_response_basis.py
git commit -m "fix(kp): select exact graded response backend"
```

### Task 6: Cache identity, progress reporting, and warm reuse

**Files:**
- Modify: `kp/kp/model/response_basis_cache.py`
- Modify: `kp/kp/model/response_basis.py`
- Test: `tests/kp/test_response_basis_cache.py`

**Step 1: Add structural/degree certificates to the cache key**

Include structural envelope, orbit ordering, action, degree representation,
translation, and configured order hashes.

**Step 2: Report actionable timings**

Log structural support/orbit counts, per-degree ranks, materialization time, and
fallback reason.

**Step 3: Test cache invalidation and reuse**

Changing a physical order or action must miss; changing target data must reuse
the target-independent response basis.

**Step 4: Run cache tests and commit**

```bash
git add kp/kp/model/response_basis_cache.py kp/kp/model/response_basis.py \
  tests/kp/test_response_basis_cache.py
git commit -m "perf(kp): cache graded response bases"
```

### Task 7: Production A/B verification and runtime gate

**Files:**
- Create: `validation_runs/graded_response_20260804/README.md`
- Do not modify release example inputs during A/B.

**Step 1: Run MgI2 Gamma q04 cold on bigmem001**

Compare graded and oracle compilers with isolated empty cache directories.
Record phase timings, ranks, hashes, span projector error, residuals, and bands.

**Step 2: Require the material gate**

Acceptance:

- cold `kp model` at most 120 seconds;
- span projector difference below `1e-10`;
- identical certified rank;
- Hamiltonian and band differences within existing numerical tolerances;
- no configured order or harmonic change.

**Step 3: Run targeted materials in parallel**

Use bigmem001 and bigmem003 for ZrS2 Gamma, AAB Gamma, PtSe2 Gamma, and MgI2 M
spinful.  Record cold and warm times separately.

**Step 4: Run targeted unit/release tests**

Run only the response/compiler and named material regressions; do not run an
unrelated full release suite.

**Step 5: Commit the validation record**

```bash
git add validation_runs/graded_response_20260804/README.md
git commit -m "test(kp): validate graded response compiler"
```
