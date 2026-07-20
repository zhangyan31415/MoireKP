# Finite-p Symmetry-First Response Compiler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compile the finite-p `complete_linear_v2` invariant response span before physical Reynolds materialization and reduce the PtSe2 Gamma q04 cold compile to at most 18.6 seconds on bigmem001.

**Architecture:** Decompose authored finite-p responses into a closed sparse symbolic atom space, apply certified factorized Q/sector/orbital/momentum actions and adjoint there, Reynolds-project and rank-reduce authored directions symbolically, and materialize only retained physical responses. Keep the existing full Reynolds compiler as an oracle and fail-closed fallback.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse linear algebra, pytest, existing MoireKP exactified/factorized symmetry artifacts.

---

### Task 1: Raw finite-p generator action

**Files:**
- Modify: `kp/kp/model/response_basis_factorized.py`
- Test: `tests/kp/test_response_basis_factorized.py`

**Step 1: Write failing unitary and antiunitary finite-p tests**

Add tests that construct a closed finite-p raw seed vocabulary and require a new
`compile_factorized_raw_seed_action` function to return a sparse real action on
`seed:real`/`seed:imag` directions. Check the real 2x2 coefficient block:

```python
unitary = np.array([[z.real, -z.imag], [z.imag, z.real]])
antiunitary = np.array([[z.real, z.imag], [z.imag, -z.real]])
```

Require non-diagonal orbital blocks to map one input seed to the correct sparse
linear combination of output orbital-pair seeds.

**Step 2: Run the tests and verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_factorized.py -k raw_seed_action
```

Expected: FAIL because `compile_factorized_raw_seed_action` does not exist.

**Step 3: Implement descriptor-space routing**

Reuse `_OrderedSeed`, `_routed_q_support_and_phase`, and
`_relative_monomial_image`. Index every raw seed by support component, routed
sector/orbital pair, monomial powers, and Q-pair support. For each input seed,
route the descriptor with the certified factorized action and accumulate the
real/imag coefficient block in a CSC matrix. Fail if any nonzero image leaves
the authored vocabulary.

**Step 4: Verify generator action tests**

Run the command from Step 2. Expected: PASS.

**Step 5: Commit the isolated change**

```bash
git add kp/kp/model/response_basis_factorized.py \
  tests/kp/test_response_basis_factorized.py
git commit -m "feat(kp): compile raw finite-p symmetry actions"
```

### Task 2: Raw polynomial vocabulary and exact adjoint image

**Files:**
- Create: `kp/kp/model/response_basis_symmetry_first.py`
- Create: `tests/kp/test_response_basis_symmetry_first.py`

**Step 1: Write failing raw-vocabulary tests**

Require helpers that convert complex raw polynomial seeds into a sparse global
real vocabulary `V` with columns ordered as `seed:real`, `seed:imag`. Verify that
the imaginary column equals multiplication of the complex response by `i` in
the global `[Re; Im]` representation.

**Step 2: Write failing finite-p adjoint tests**

Construct a pair of `p`/`-p` seeds where changing the Q center causes a binomial
lower-order contribution. Require the adjoint image columns to equal direct
polynomial conjugate-transpose and require applying the adjoint twice to recover
the original physical vocabulary within the propagated bound.

**Step 3: Run tests and verify failure**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_symmetry_first.py
```

Expected: FAIL because the symmetry-first module does not exist.

**Step 4: Implement sparse raw vocabulary and adjoint images**

Vectorize each seed's monomial CSR matrices once. Construct the real and imaginary
directions without Hermitian projection. For adjoint images, swap monomial powers
and conjugate-transpose each polynomial matrix; the already expanded global
polynomial coefficients automatically include finite-center lower-order mixing.
Do not assume a same-degree one-to-one partner.

**Step 5: Verify tests and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_symmetry_first.py
git add kp/kp/model/response_basis_symmetry_first.py \
  tests/kp/test_response_basis_symmetry_first.py
git commit -m "feat(kp): build finite-p raw and adjoint vocabularies"
```

### Task 3: Non-closed authored vocabulary and symbolic atom projection

**Files:**
- Modify: `kp/kp/model/response_basis_symmetry_first.py`
- Modify: `tests/kp/test_response_basis_symmetry_first.py`
- Reference: `kp/kp/model/response_basis_fixed_compiler.py`

**Step 1: Write a failing non-closure dense-oracle span test**

Construct a finite-p toy model whose authored harmonic representatives are not
closed under C2 or T. Confirm that raw seed-label action compilation rejects it,
then require the symbolic compiler to match the existing complete Reynolds
candidate span. Compare ranks and physical projectors within the propagated bound.

**Step 2: Run and verify failure**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_symmetry_first.py -k fixed_space
```

**Step 3: Implement symbolic atom closure and actions**

Index scalar real atoms by monomial, matrix row, matrix column, and real/imaginary
component. Seed the atom vocabulary from authored responses and close it under
the certified generators and adjoint. Compile exact sparse real actions on this
closed atom space without requiring transformed harmonic representatives to be
authored term-registry entries.

**Step 4: Project and select a deterministic symbolic frame**

Embed authored real/imaginary seed directions into atom coordinates, apply the
finite-group Reynolds average and adjoint projection symbolically, and use RRQR
on the projected atom coordinates. Materialize only the selected independent
columns and retain logical seed/component provenance. Re-certify physical rank.

**Step 5: Verify oracle equality and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_symmetry_first.py
git add kp/kp/model/response_basis_symmetry_first.py \
  tests/kp/test_response_basis_symmetry_first.py
git commit -m "feat(kp): solve finite-p symmetry-first fixed space"
```

### Task 4: Production routing and fail-closed fallback

**Files:**
- Modify: `kp/kp/model/response_basis.py`
- Modify: `kp/kp/model/response_basis_symmetry_first.py`
- Modify: `tests/kp/test_complete_response_basis.py`

**Step 1: Write failing production-routing tests**

Require `complete_linear_v2` finite-p groups with certified factorized actions to
report `symmetry_first_finite_p_v1`, materialize only retained channels, and avoid
calling `compile_candidate_responses` for finite-p seeds. Add closure and adjoint
certification failures that require explicit fallback to
`finite_p_full_candidate_reynolds_v1`.

**Step 2: Run and verify failure**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py -k 'finite_p and symmetry_first'
```

**Step 3: Route finite-p groups through the optimized compiler**

Integrate the new compiler in `_compile_model_response_basis_uncached`. Combine
its retained channels and proofs with existing p=0 fixed groups. Preserve the
dense path as fallback/oracle, and include fallback reasons in progress output
and candidate artifacts.

**Step 4: Bump the compiler/cache version**

Change the compiler version so old persistent artifacts cannot be loaded as the
new basis. Keep the public YAML unchanged.

**Step 5: Run targeted tests and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_symmetry_first.py \
  tests/kp/test_response_basis_cache.py
git add kp/kp/model/response_basis.py \
  kp/kp/model/response_basis_symmetry_first.py \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_symmetry_first.py
git commit -m "perf(kp): route finite-p responses through fixed space"
```

### Task 5: Compact artifact and single-pass hashing

**Files:**
- Modify: `kp/kp/model/response_basis.py`
- Modify: `kp/kp/model/response_basis_cache.py`
- Modify: `tests/kp/test_response_basis_cache.py`

**Step 1: Add a failing hash-call and round-trip test**

Instrument a small compiled basis and require construction/cache export to compute
the canonical basis hash once. Require compact fixed-space arrays to round-trip
through NPZ without embedding multi-megabyte coordinate arrays repeatedly in JSON.

**Step 2: Run and verify failure**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_cache.py -k 'hash or compact'
```

**Step 3: Implement compact array-backed artifact hashing**

Hash canonical metadata and canonical sparse array digests once. Store large
fixed-coordinate mappings as NPZ arrays referenced by metadata. Preserve hash
validation on load.

**Step 4: Verify and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_cache.py
git add kp/kp/model/response_basis.py \
  kp/kp/model/response_basis_cache.py \
  tests/kp/test_response_basis_cache.py
git commit -m "perf(kp): compact response-basis artifacts"
```

### Task 6: PtSe2 physical equivalence

**Files:**
- Add local outputs only under: `validation_runs/ptse2_symmetry_first_20260720/`
- Do not modify public example YAML for validation.

**Step 1: Run dense oracle and optimized cold compiles on bigmem001**

Use the existing PtSe2 Gamma q04 configuration with persistent cache disabled.
Save both frozen bases and timing logs under the validation directory.

**Step 2: Compare invariant spans**

At random two-dimensional momenta, compare optimized and dense response spans by
principal angles and projector residual. Require equal rank 456 and agreement
within the propagated certification bound.

**Step 3: Compare fitted physical outputs**

Run the same linear initialization and nonlinear band loss. Compare Hamiltonian
residuals, bands, overlaps, covariance, and standalone evaluation. Reject any
accuracy regression outside numerical solver tolerance.

**Step 4: Run the cold performance gate**

Run three isolated cache-disabled compiles on bigmem001. Require median total
compile time `<= 18.6 s`. Record wall times and peak memory in the local report.

### Task 7: Release verification

**Files:**
- Modify documentation only if runtime output or artifact names changed.

**Step 1: Run targeted KP model tests**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_symmetry_first.py \
  tests/kp/test_response_basis_fixed_compiler.py \
  tests/kp/test_response_basis_cache.py \
  tests/kp/test_configured_model.py
```

Expected: all pass.

**Step 2: Run the default release suite**

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Expected: all tests affected by this change pass; any unrelated pre-existing
failures are listed with exact evidence rather than described as green.

**Step 3: Review the final diff and commit only scoped files**

```bash
git diff --check
git status --short
```

Do not stage validation outputs, existing dirty files, caches, paper files, or
unrelated example changes.
