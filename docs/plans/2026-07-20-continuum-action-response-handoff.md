# Continuum-Action Response Handoff Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the complete-response finite group consume the certified projected continuum Q/sector action while preserving TAPW source-action provenance.

**Architecture:** Add one strict adapter in `response_basis.py` that converts a complete `model_basis_action.items` record into the global Q and sector permutations used by `FiniteGroupGenerator`. Cross-check any loaded factorized action against that projected permutation before cold compilation, while retaining the existing source-metadata path when projected metadata is absent.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse matrices, pytest, existing `kp symm` packed factorized-action format.

---

### Task 1: Capture the projected-action regression

**Files:**
- Modify: `tests/kp/test_complete_response_basis.py:2080-2180`

**Step 1: Write the failing PtSe2-shaped unit test**

Add a test whose source operation has `sector_map: identity`, whose complete
`model_basis_action.items` exchanges `L1` and `L2`, and whose exactified TR
matrix is off-diagonal.  Call `finite_group_from_model_actions` and require:

```python
assert group.elements[1].q_permutation == (1, 0)
assert group.elements[1].sector_permutation == (1, 0)
```

Do not put `internal_resolved_action` in the fixture; that would reproduce the
old explicit-override path rather than the automatic projected-basis path.

**Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py::test_model_action_adapter_uses_complete_projected_basis_action
```

Expected: FAIL because the current adapter returns source permutations `(0, 1)`.

**Step 3: Add malformed-metadata tests**

Add parameterized cases for missing source items, duplicate targets, unknown
sectors, and a recorded `sector_map` inconsistent with the item mapping.
Require precise `ValueError` messages rather than heuristic completion.

**Step 4: Run the malformed tests and verify RED**

Expected: at least one case passes through or fails for the wrong reason because
no projected-action adapter exists yet.

### Task 2: Implement strict projected permutation resolution

**Files:**
- Modify: `kp/kp/model/response_basis.py:5070-5240`
- Test: `tests/kp/test_complete_response_basis.py`

**Step 1: Add a private projected-action adapter**

Implement a helper with the conceptual interface:

```python
def _projected_basis_permutations(
    operation,
    sectors,
    Q_set1,
    Q_set2,
) -> tuple[tuple[int, ...], tuple[int, int]] | None:
    ...
```

Return `None` when no complete `model_basis_action` is present.  Otherwise:

- require `items` to be a sequence of mappings;
- resolve sector names through `_sector_slot_records`;
- require every source `(sector, q_index)` exactly once;
- require every target `(sector, q_index)` exactly once;
- construct the global Q permutation in qset1-then-qset2 order;
- derive one sector permutation consistent across all Q rows;
- verify the derived sector permutation equals the recorded
  `model_basis_action.sector_map`.

**Step 2: Use the projected mapping in `finite_group_from_model_actions`**

Keep the physical `k_map` and its canonical reciprocal-lattice action.  When
the helper returns a mapping, use its Q and sector permutations.  Otherwise use
the existing `_sector_permutation_from_metadata` and
`_q_permutation_from_metadata` path unchanged.

**Step 3: Run the focused tests and verify GREEN**

Run the new tests plus the existing explicit `internal_resolved_action` test.
Expected: PASS.

### Task 3: Cross-check the factorized continuum action before compile

**Files:**
- Modify: `kp/kp/model/response_basis.py:5130-5240, 6120-6180`
- Test: `tests/kp/test_complete_response_basis.py`
- Test: `tests/kp/test_response_basis_factorized.py`

**Step 1: Write a failing mismatch test**

Create a certified factorized TR action whose Q or sector permutation disagrees
with `model_basis_action.items`.  Require the compiler to raise a targeted
error before entering candidate compilation.

**Step 2: Verify RED**

Expected: current code reaches the later fallback diagnostic rather than
rejecting the inconsistent projected artifacts at the handoff boundary.

**Step 3: Pass factorized actions into finite-group construction**

In `compile_model_response_basis`, collect the operation's factorized actions
before constructing each group.  Cross-check:

- operation name;
- antiunitary parity;
- Cartesian `k_forward` against the physical `k_map`;
- Q permutation;
- sector permutation.

Keep the existing factorized-matrix/exactified-matrix certification in the
factorized compiler.  Do not duplicate or weaken its numerical bound.

**Step 4: Verify GREEN**

Run the focused complete-response and factorized-action tests.

### Task 4: Verify existing KP behavior

**Files:**
- No source changes expected.

**Step 1: Run targeted regression groups**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_configured_model.py \
  tests/kp/test_operation_standardization.py \
  tests/kp/test_standalone_export.py
```

Expected: all selected tests pass.

**Step 2: Run the default release suite**

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Record every unrelated dirty-worktree or missing-external-data failure exactly.

### Task 5: Run a clean-cache PtSe2 production check

**Files:**
- Read: `examples/autoproject/ptse2_7.34/configs/gamma_spinful_q04.yaml`
- Generate ignored outputs under: `validation_runs/ptse2_continuum_action_handoff_20260720/`

**Step 1: Preserve the current certified baseline**

Record hashes and metrics from the existing standalone model and the old
`18a7cc...npz` compiled response cache.  Do not delete or overwrite the user's
`model_` archive.

**Step 2: Run from an isolated empty output/cache directory**

Use a temporary validation copy of the YAML whose only change is the output
root.  Run `kp model` on an available compute node.  Capture the real shell
status and full log.

**Step 3: Check compiler diagnostics**

Require all of the following:

```text
factorized generator TR is accepted
finite-p factorized sparse group actions certified
no Q permutation mismatch
no finite-p factorized sparse Reynolds fallback
persistent cache cold compile+store completes
```

**Step 4: Run the same command again**

Require a persistent-cache hit and a materially shorter pre-fit startup.

**Step 5: Compare physics and numerical outputs**

Compare fitted coefficients where the channel vocabulary is identical, plus
top-6/top-10/top-20/all-band errors, wavefunction overlap, and exactified
symmetry residuals.  Explain any legitimate basis-vocabulary change rather than
silently comparing incompatible coefficient arrays.

### Task 6: Document and hand off

**Files:**
- Update only if user-facing behavior needs clarification: `docs/complete_response_basis.md`

**Step 1: Summarize the ownership rule**

Document that TAPW source actions remain provenance while `kp model` consumes
the `kp symm` exactified continuum action.

**Step 2: Review the dirty worktree**

List only files changed for this fix.  Do not stage unrelated source, examples,
validation outputs, caches, or paper files.

**Step 3: Report verification evidence**

Provide cold/warm compile timings, fallback status, numerical comparison, and
the exact test totals.  Do not claim the full suite is green if unrelated
failures remain.
