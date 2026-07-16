# KP Project Identity and Downfold Performance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the full Hamiltonian hash from `kp project` and reduce exact downfold and auto-gauge runtime without changing numerical results.

**Architecture:** Preserve cheap structural artifact identities while declaring a new projection-basis schema that does not certify Hamiltonian content. Characterize the actual MgI2 shifted high-space matrix before selecting either a Hermitian direct solver or a preconditioned iterative solver, then reuse a certified reference projection across auto-gauge basis candidates.

**Tech Stack:** Python 3.11, NumPy, SciPy/LAPACK, threadpoolctl, pytest, joblib, cProfile.

---

### Task 1: Remove Hamiltonian content hashing

**Files:**
- Modify: `tests/kp/test_kp_artifact_identity.py`
- Modify: `kp/kp/identity.py`
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/symmetry/projection.py`

**Step 1: Write the failing test**

Add a test that replaces `kp.identity.hash_file` with a function that raises, calls
`build_projection_basis_identity`, and asserts construction still succeeds. Assert that
changing only Hamiltonian file bytes does not change identity, while changing a Q set does.

**Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=kp:tapw /data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest \
  -q -p no:cacheprovider tests/kp/test_kp_artifact_identity.py
```

Expected: failure from the raising `hash_file` replacement.

**Step 3: Implement the minimal contract change**

Remove `hamk_file` and `hamk_fallback` from the projection identity builder and both call
sites. Build `input_hash` from Q-set structural hashes only, and bump the projection basis
schema version.

**Step 4: Verify GREEN and targeted consumers**

Run the identity tests, symmetry projection identity tests, and CLI project memory tests.

### Task 2: Characterize the MgI2 high-space matrix

**Files:**
- Create: `validation_runs/mgi2_project_perf_bigmem001_20260717/probe_hhh_properties.py`
- Create: `validation_runs/mgi2_project_perf_bigmem001_20260717/hhh_properties.json`

**Step 1: Build the exact one-k projector groups**

Reuse production block/projector assembly and extract `H_PP`, `H_PH`, and `H_HH` without
changing production code.

**Step 2: Record matrix properties**

Measure Hermiticity, inertia, spectral distance, block norm ratios, and representative block
ranks.

**Step 3: Benchmark solver hypotheses one at a time**

Compare current LU, Hermitian direct factorization, and block-preconditioned iterative solves.
Record factor/solve time, residual, Heff difference, and peak memory where available.

### Task 3: Implement an exact downfold optimization

**Files:**
- Modify: `tests/kp/test_downfold.py`
- Modify: `tests/kp/test_block_projector_downfold.py`
- Modify: `kp/kp/blocks/downfold.py`

**Step 1: Write a failing equivalence/residual test**

Cover Hermitian indefinite shifted matrices with multiple complex right-hand sides and a
near-pole but nonsingular case.

**Step 2: Verify RED**

Run the specific new test and confirm it fails because the selected optimized path does not
exist.

**Step 3: Implement only the winning measured solver**

Keep current LU as fallback. Reject iterative results whose residual exceeds the configured
strict threshold.

**Step 4: Verify GREEN and performance gate**

Require `1e-10 eV` numerical agreement and at least 15% kernel speedup on MgI2 before making
the new path default.

### Task 4: Reuse projection work across auto-gauge candidates

**Files:**
- Modify: `tests/kp/test_project_auto_gauge_cli.py`
- Modify: `tests/kp/test_symm_projection.py`
- Modify: `kp/kp/symmetry/projection.py`
- Possibly modify: `kp/kp/blocks/blocks.py`

**Step 1: Write a failing call-count and equivalence test**

Assert multiple gauge candidates do not repeat the expensive physical downfold for the same
Hamiltonian/k point, while preserving selection metrics and candidate ranking.

**Step 2: Verify RED**

Confirm the existing implementation calls `_projectors_for_k` once per candidate.

**Step 3: Implement certified basis-frame reuse**

Cache the physical reference state and transform candidate `U_low` and Heff through the
computed low-dimensional unitary. Fall back to full evaluation when the transform residual
exceeds tolerance.

**Step 4: Verify GREEN and bigmem timing**

Require the same resolved anchors, spectra, exactification reports, and a material reduction
from the measured ~15 s auto-gauge overhead.

### Task 5: Full verification

**Files:**
- Update: `validation_runs/mgi2_project_perf_bigmem001_20260717/summary.md`

**Step 1: Run targeted tests**

Run identity, downfold, block projector, auto gauge, symmetry projection, and CLI project tests.

**Step 2: Run release tests**

```bash
PYTHONPATH=kp:tapw /data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest \
  -q -p no:cacheprovider -m "not slow and not external_data"
```

**Step 3: Run isolated bigmem regression**

Compare one-k profiles and the 61-k/8-worker workload with the established baselines.

**Step 4: Record exact evidence**

Document commands, exit statuses, wall times, numerical differences, and any optimization
that was rejected.
