# Shell-Aware KP Profile Windows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Select compact and broad linear/nonlinear KP profiles with automatically resolved 2x-central and two-Q-shell band windows, while preventing broad profiles from degrading compact edge bands.

**Architecture:** Add one deterministic window resolver that derives `N_low` and `N_high` from `n_orb` and layer Q sets. Extend candidate scores with metrics for both windows, select low by constrained complexity, then filter high candidates through low-window dominance before optimizing broad-window fidelity. Keep fallback candidates runnable but mark failed dominance as `WARN`.

**Tech Stack:** Python 3.11, NumPy, pytest, existing KP configured-model pipeline and model-selection dataclasses.

---

### Task 1: Resolve low/high physical band windows

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_model_selection_primary_window.py`

**Step 1: Write failing tests**

Add tests asserting:

- `[2,2]` with Q shells `1+6+...` resolves to low/high `8/28`;
- `[1,1]` resolves to `4/14`;
- Q=0 is included as the first shell;
- shifted Q sets use the two smallest nonzero norms;
- unequal layer Q sets are counted independently;
- counts are clamped to the Hamiltonian dimension.

**Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection_primary_window.py
```

Expected: new resolver tests fail because shell-aware profile windows do not exist.

**Step 3: Implement the resolver**

Add a small immutable result record containing low/high counts and per-layer
shell counts.  Group Q norms with the existing numerical shell tolerance,
retain the first two complete shells, and compute:

```python
low = min(total_bands, 2 * sum(n_orb))
high = min(total_bands, sum(norb * nq for norb, nq in zip(n_orb, first_two_shell_counts)))
```

**Step 4: Run tests and verify pass**

Run the Task 1 test file and expect all tests to pass.

### Task 2: Score every candidate on both windows

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_model_selection.py`
- Test: `tests/kp/test_model_selection_primary_window.py`

**Step 1: Write failing tests**

Assert that candidate records contain separate low/high RMS, max error, mean
overlap, and minimum principal singular value.  Verify top/bottom slices and
band-index crossings are handled through subspace metrics rather than individual
band identity.

**Step 2: Run targeted tests and verify failure**

Run both model-selection test files.

**Step 3: Implement dual-window metrics**

Extend `CandidateScore` with a compact window-metrics record.  Compute both
records from the same eigensystems in `_complete_candidate_score`, avoiding a
second diagonalization.  Serialize both records to JSON/CSV/Markdown.

**Step 4: Run targeted tests and verify pass**

Run both model-selection test files.

### Task 3: Select low by fidelity-constrained complexity

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Test: `tests/kp/test_model_selection.py`

**Step 1: Write failing tests**

Create candidates where the absolute best RMS uses many parameters but a much
smaller candidate lies within the configured fidelity tolerance.  Assert low
chooses the smaller candidate.  Add ordering tests for parameter count, closed
groups, harmonic support, derivative order, then error.  Add no-feasible-candidate
warning behavior.

**Step 2: Run test and verify failure**

Run `tests/kp/test_model_selection.py`.

**Step 3: Implement constrained complexity selection**

Build the fidelity-eligible low set first, then apply the documented
lexicographic complexity key.  Preserve warning fallback when targets are unmet.

**Step 4: Run test and verify pass**

Run `tests/kp/test_model_selection.py`.

### Task 4: Enforce high-over-low dominance

**Files:**
- Modify: `kp/kp/model/model_selection.py`
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_model_selection.py`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing tests**

Assert that high candidates with better broad RMS but worse low-window RMS,
maximum error, or overlap are rejected.  Assert a dominating candidate is
selected even with more parameters.  When no candidate dominates, assert the
best broad candidate is exported with `WARN` and explicit failed guards.

**Step 2: Run tests and verify failure**

Run the targeted selector and configured-pipeline tests.

**Step 3: Implement dominance filtering**

Select low first.  Filter high candidates using low-window RMS/max/overlap
tolerances, then rank eligible high candidates on broad-window fidelity,
overlap, and complexity.  Carry warning status through four-profile reports and
standalone output metadata.

**Step 4: Run tests and verify pass**

Run the targeted selector and configured-pipeline tests.

### Task 5: Apply two-sided linear spectral weighting

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_complete_response_basis.py`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing tests**

Construct a toy response where an unweighted matrix fit sacrifices the edge
subspace.  Assert low uses the resolved low window on both matrix indices and
retains a configurable cross-window floor.  Assert high includes the same edge
weight plus the broad window.

**Step 2: Run tests and verify failure**

Run the two targeted test files.

**Step 3: Implement the weighted linear objective**

Reuse the target-Heff eigenbasis and complete-response matrix.  Apply separable
left/right spectral weights while keeping the coefficient solve linear.  Record
window counts and weight floors in diagnostics.

**Step 4: Run tests and verify pass**

Run the two targeted test files.

### Task 6: Verify MgI2 19Q and 31Q on bigmem

**Files:**
- Create only ignored artifacts under: `validation_runs/mgi2_shell_windows_20260718/`

**Step 1: Generate validation configs**

Use automatic low/high counts, the existing order/harmonic frontier, double-sided
linear weights, nonlinear refinement, and `max_variables: 3000`.

**Step 2: Run both pipelines**

Run 19Q and 31Q on separate available bigmem nodes.  Require all four standalone
exports per Q set.

**Step 3: Audit results**

For every profile report low-window and high-window RMS/max, mean/minimum
principal overlap, parameters, groups, harmonics, orders, symmetry residuals,
and dominance status.  Render and inspect all eight band plots.

**Step 4: Run regression tests**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_model_selection.py tests/kp/test_model_selection_primary_window.py tests/kp/test_configured_model.py tests/kp/test_complete_response_basis.py
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Report unrelated dirty-worktree failures separately; do not modify unrelated
release/example files.
