# CPC Response-Space Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a clean CPC integration branch containing automatic model selection, the complete response-basis compiler, and case-derived Gamma/K/M response spaces.

**Architecture:** Start from the fetched CPC release tip and build a linear sequence of explicit, reviewable commits. Replay only patch-missing committed work, then migrate the response compiler from the validated source tree in dependency order; never merge the temporary core snapshot or a dirty development branch wholesale.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse matrices, SymPy, pytest, Git worktrees.

---

### Task 1: Repair the clean CPC output-retention baseline

**Files:**
- Modify: `kp/kp/cli.py:179`
- Test: `tests/kp/test_configured_model.py:1261`

**Step 1: Verify the existing regression is red**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_configured_model.py::test_canonical_model_cleanup_preserves_harmonic_recommendation_plot
```

Expected: FAIL because `hamiltonian_element_comparison.png` is deleted.

**Step 2: Restore the original canonical-output contract**

Add these exact names to `_cleanup_canonical_model_output`'s `keep` set:

```python
"hamiltonian_element_comparison.png",
"hamiltonian_element_comparison.pdf",
```

**Step 3: Verify the focused test is green**

Run the command from Step 1.

Expected: PASS.

**Step 4: Verify the available configured-model baseline**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py
```

Expected: `247 passed, 1 skipped`.

**Step 5: Commit**

```bash
git add kp/kp/cli.py
git commit -m "fix(kp): preserve Hamiltonian comparison diagnostics"
```

### Task 2: Replay CPC-missing automatic model selection

**Files:**
- Create: `kp/kp/model/model_selection.py`
- Create: `tests/kp/test_model_selection.py`
- Create/modify: the committed design documents in the replayed range

**Step 1: Confirm patch scope**

Run:

```bash
git log --reverse --format='%h %s' \
  7d23394..471db81
git cherry origin/codex/cpc-release-clean codex/kp-auto-model-selection
```

Expected: the range `601deaa..471db81` is missing; the later example commit is
patch-equivalent and must not be replayed.

**Step 2: Replay the missing linear range**

Cherry-pick, in order, the commits from `601deaa` through `471db81`. Resolve
conflicts by preserving the CPC release version for already-integrated workflow
and example content; preserve the source commit for `model_selection.py`, its
tests, and response-compiler design documents.

**Step 3: Verify automatic model selection**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection.py
```

Expected: all tests pass.

**Step 4: Verify no example payload was duplicated**

Run:

```bash
git diff --name-status origin/codex/cpc-release-clean...HEAD -- examples
git diff --check
```

Expected: no large example-data replay and no whitespace errors.

### Task 3: Add standalone response-basis compiler modules

**Files:**
- Create: `kp/kp/model/response_basis.py`
- Create: `kp/kp/model/response_basis_adjoint.py`
- Create: `kp/kp/model/response_basis_cache.py`
- Create: `kp/kp/model/response_basis_factorized.py`
- Create: `kp/kp/model/response_basis_fixed_compiler.py`
- Create: `kp/kp/model/response_basis_fixed_subspace.py`
- Create: `kp/kp/model/response_basis_oracle.py`
- Create: `kp/kp/model/response_basis_symmetry_first.py`
- Create: `tests/kp/test_response_basis_adjoint.py`
- Create: `tests/kp/test_response_basis_cache.py`
- Create: `tests/kp/test_response_basis_factorized.py`
- Create: `tests/kp/test_response_basis_fixed_compiler.py`
- Create: `tests/kp/test_response_basis_fixed_subspace.py`
- Create: `tests/kp/test_response_basis_symmetry_first.py`

**Step 1: Migrate tests first**

Copy only the six response-basis module tests from the validated source tree.

**Step 2: Verify red import failures**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_response_basis_adjoint.py \
  tests/kp/test_response_basis_cache.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_fixed_compiler.py \
  tests/kp/test_response_basis_fixed_subspace.py \
  tests/kp/test_response_basis_symmetry_first.py
```

Expected: collection fails because the response-basis modules are absent.

**Step 3: Migrate the standalone compiler modules**

Copy only the eight listed production modules from the validated source tree.
Do not copy pipeline, projection, CLI, TAPW, documentation, or validation files.

**Step 4: Verify standalone compiler tests**

Run the command from Step 2.

Expected: tests that do not require the factorized symmetry artifact pass;
remaining failures must identify the exact Task 4 dependency rather than an
unrelated missing file.

**Step 5: Commit the independently passing subset**

Stage explicit module and test paths only. If Task 4 dependencies prevent a
coherent green commit, leave Task 3 uncommitted and complete Task 4 in the same
review batch; do not weaken tests.

### Task 4: Add exact symmetry-action dependencies

**Files:**
- Create: `kp/kp/symmetry/canonical_target.py`
- Create: `kp/kp/symmetry/factorized_action.py`
- Create: `kp/kp/symmetry/structure_certificate.py`
- Create: `kp/kp/symmetry/structure_recovery.py`
- Modify: `kp/kp/basis/symmetry_gauge.py`
- Modify: `kp/kp/blocks/blocks.py`
- Modify: `kp/kp/identity.py`
- Modify: `kp/kp/model/symmetry.py`
- Modify: `kp/kp/symmetry/exactify_representation.py`
- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `kp/kp/symmetry/q_canonicalization.py`
- Create: corresponding focused tests under `tests/kp/`

**Step 1: Migrate focused tests before production files**

Include canonical-target, factorized-action, structure-certificate,
structure-recovery, joint-exactification, and projection artifact tests.
The high-dimensional joint exactification regression also covers the required
symmetry-gauge stabilizer matching behavior.
Projection identity and active-sector Q-canonicalization regressions cover the
additional handoff metadata used by the exact target artifact.
The projection-context eigensystem cache regression covers the required
`projection.py` to `blocks.py` call contract without migrating unrelated
automatic low-energy-selection behavior.

**Step 2: Verify expected missing-artifact failures**

Run each new focused file separately and confirm failures come from absent
production artifacts.

**Step 3: Migrate the four new modules and minimal tracked-file deltas**

Use the validated source tree as the source of truth. Review every tracked-file
diff against CPC and exclude unrelated reporting, CLI, TAPW, paper, and example
changes.

**Step 4: Run the symmetry and standalone compiler test set**

Expected: all focused symmetry and Task 3 response-basis tests pass.

**Step 5: Commit explicit paths**

Commit symmetry artifact support separately from the response compiler when the
dependency graph permits; otherwise use one documented atomic commit.

### Task 5: Connect complete responses to configured models and export

**Files:**
- Modify: `kp/kp/model/core.py`
- Modify: `kp/kp/model/export.py`
- Modify: `kp/kp/model/model_selection.py`
- Modify: `kp/kp/model/pipeline.py`
- Modify: `kp/kp/config/case.py`
- Modify: `kp/kp/identity.py`
- Modify: `kp/kp/reporting.py`
- Modify: `kp/kp/cli.py` (model-command handoff and cleanup allowlist only)
- Test: `tests/kp/test_complete_response_basis.py`
- Test: `tests/kp/test_configured_model.py`
- Test: `tests/kp/test_model_selection.py`
- Test: `tests/kp/test_reporting.py`
- Test: `tests/kp/test_standalone_export.py`

**Step 1: Migrate complete-response integration tests first**

Copy the complete-response test and the relevant configured/export test deltas.

**Step 2: Verify red production-integration failures**

Run the three focused files. Confirm failures identify absent
`complete_linear_v2` integration paths.

**Step 3: Migrate minimal production deltas**

Integrate compilation, fitting, refit, cache identity, and standalone export.
Preserve CPC release CLI and output contracts unless a test demonstrates a
required response integration. Add only the reporter primitives and
model-command result handoff needed by configured model selection; keep the
existing Hamiltonian-comparison diagnostics in the cleanup allowlist.

**Step 4: Run focused configured/response/export tests**

Expected: all focused tests pass.

**Step 5: Commit**

Stage the exact production and test files only and commit the complete-response
pipeline integration.

### Task 6: Integrate case-derived Gamma/K/M candidate generation

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Modify: `kp/kp/model/response_basis_factorized.py`
- Test: `tests/kp/test_configured_model.py`
- Test: `tests/kp/test_complete_response_basis.py`
- Test: `tests/kp/test_response_basis_factorized.py`

**Step 1: Apply tests before the final production delta**

Include default deferral, actual harmonic support, raw Q-pair support,
build/refit, dense L2 orbital-gauge covariance, and dense factorized-action
regressions.

**Step 2: Verify the tests fail on the hard-coded complete profiles**

Run only the new test selections and confirm failures describe the old Gamma/K/M
profile behavior or the structural-zero dense-action rejection.

**Step 3: Apply the validated production delta**

Add the generic case-derived generator, split diagnostic/model support-count
conventions, dispatch build/refit for Gamma/K/M, empty the active complete
profile registry, and retain the legacy registry only for
`legacy_frozen_v1`.

Drop factorized coefficients entering a certified structural-zero channel after
Hermitian projection.

**Step 4: Run the complete targeted set**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_configured_model.py \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_symmetry_first.py \
  tests/kp/test_response_basis_cache.py \
  tests/kp/test_response_basis_fixed_compiler.py
```

Expected: all tests pass; the test-only toy-generator warning is acceptable.

**Step 5: Commit**

Commit the Gamma/K/M case-derived behavior and regressions separately from the
response-basis prerequisite commits.

### Task 7: Final CPC verification and handoff

**Files:**
- Modify only release-facing documentation required by the integrated behavior
- Write local validation summaries only under ignored `validation_runs/`

**Step 1: Audit the branch**

Run `git status`, `git diff --check`, commit-range review, absolute-path scans,
and generated-file scans. Confirm no paper, validation, cache, or review bundle
is staged.

**Step 2: Run syntax and targeted tests**

Run `py_compile`, all response/configured-model focused tests, and automatic
model-selection tests.

**Step 3: Re-run actual examples**

Use only user-approved compute nodes. Validate representative Gamma, K, and M
spinless/spinful cases, inspect band plots, and verify standalone evaluator
agreement.

**Step 4: Run the release gate when authorized**

Run:

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Do not claim CPC release readiness until this gate is authorized and passes.

**Step 5: Present integration options**

Use `superpowers:finishing-a-development-branch` to offer merge, push/PR, keep,
or discard choices. Never update the shared CPC worktree implicitly.
