# Gamma Dual-Frame Model Gauge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve automatic Gamma routing certification while restoring the existing deterministic SCDM model gauge for authoritative Heff, symmetry actions, and continuum fitting.

**Architecture:** The producer materializes two certified bases for the same selected joint subspace.  Compact routed frames retain selection/rank evidence; compact SCDM model frames drive all production matrices.  A strict v3 handoff persists both plus their unitary bridge and fails closed on any identity or covariance mismatch.

**Tech Stack:** Python 3.11, NumPy, SciPy, pytest, existing Gamma routing, auto-SCDM, downfolding, candidate-certificate, and response-compiler modules.

---

### Task 1: Establish the focused baseline and RED model-frame tests

**Files:**

- Modify: `tests/kp/test_auto_gauge_selection.py`
- Modify: `tests/kp/test_gamma_auto_producer.py`

**Step 1: Record the baseline**

Run:

```bash
REPO=/data/work/zy/software/1.tapw_code/moirekp-release
PY=/data/home/zy/mambaforge/envs/moirekp/bin/python
PYTHONPATH="$REPO/kp:$REPO/tapw" "$PY" -m pytest -q -p no:cacheprovider \
  tests/kp/test_gamma_auto_producer.py \
  tests/kp/test_gamma_auto_runtime.py \
  tests/kp/test_gamma_projection_handoff.py \
  tests/kp/test_auto_gauge_selection.py
```

Expected before changes: all existing tests pass.

**Step 2: Write a failing pure SCDM-frame test**

Add a four-state Gamma eigensystem whose model anchors require a full `U(4)`
rotation with nonzero cross-routing-group blocks.  Specify the desired API:

```python
anchor_spec = build_gamma_model_anchor_spec(...)
model = build_gamma_model_frames(..., anchor_spec=anchor_spec)
```

Assert deterministic column order, orthonormality, fixed-anchor alignment, and
non-block-diagonal mixing.

**Step 3: Run the new test and verify RED**

Expected: import or attribute failure because the model-frame API does not yet
exist.

**Step 4: Write a failing producer behavior test**

Add `test_gamma_auto_producer_hands_off_model_frame_not_routing_frame` using a
k-dependent mixing toy.  Assert distinct frames, equal projectors, unitary
bridge, bridge reconstruction, and model-frame authoritative Heff.

**Step 5: Run the producer test and verify RED**

Expected: failure because the current producer only creates routed frames.

**Step 6: Commit only the RED tests**

```bash
git add tests/kp/test_auto_gauge_selection.py tests/kp/test_gamma_auto_producer.py
git commit -m "test(kp): specify Gamma dual-frame model gauge"
```

### Task 2: Reuse the existing Gamma auto-SCDM model-frame construction

**Files:**

- Modify: `kp/kp/blocks/blocks.py`
- Modify: `kp/kp/blocks/__init__.py`
- Modify: `tests/kp/test_auto_gauge_selection.py`

**Step 1: Add immutable model-frame records**

Implement strict records for the resolved SCDM anchor contract and one k-row of
local model frames.  The contract must bind joint bands, group ranks, resolved
references, model column order, quality thresholds, and a canonical hash.

**Step 2: Extract a pure anchor builder from the existing Gamma branch**

Reuse, without duplicating physics rules:

- `select_anchor_rows_qrcp`;
- `_complete_gamma_spinful_reference_terms`;
- `_order_gamma_references_for_model_basis`;
- existing reference-quality gates.

Keep the legacy `resolve_project_gauge_anchors` behavior unchanged by making it
call the same pure core.

**Step 3: Implement model-frame materialization**

For every Q, align all joint columns at once with `align_eigenstates`.  Return a
compact `(Nq, same_q_dim, rank)` tensor.  Validate finite values,
orthonormality, joint-band coverage, and fixed dimensions.

**Step 4: Run the pure test and verify GREEN**

Run only the new test, then all of `test_auto_gauge_selection.py`.

**Step 5: Refactor while green**

Remove any duplicate reference parsing or SCDM quality calculation introduced
by the extraction.

**Step 6: Commit**

```bash
git add kp/kp/blocks/blocks.py kp/kp/blocks/__init__.py tests/kp/test_auto_gauge_selection.py
git commit -m "feat(kp): build deterministic Gamma model frames"
```

### Task 3: Upgrade the routed handoff to strict dual-frame v3

**Files:**

- Modify: `kp/kp/projection_handoff.py`
- Modify: `tests/kp/test_gamma_projection_handoff.py`
- Modify: `tests/kp/test_selection_orchestration.py`

**Step 1: Write failing v3 roundtrip and tamper tests**

Cover compact model frames, compact local bridges, model anchor contract, all
per-k/aggregate hashes, and authoritative model Heff.  Tamper each component
independently and require `HANDOFF_IDENTITY` rejection.

**Step 2: Verify RED**

Expected: current v2 exact-key schema rejects the new fields and has no explicit
model accessor.

**Step 3: Implement v3 fields and identity**

Bump `GAMMA_ROUTED_BASIS_HANDOFF_VERSION` to v3.  Extend exact archive keys,
dtype contracts, bound basis hash, save/load, and in-memory validation.

**Step 4: Implement bridge validation and explicit accessors**

Add:

```python
assemble_routing_for_k(k_index, include_high=...)
assemble_model_for_k(k_index)
routing_to_model_for_k(k_index)
model_state_for_k(k_index)
```

Remove `assemble_for_k`.  Validate local and assembled orthonormality,
unitarity, reconstruction, projector equality, and Heff covariance.

**Step 5: Bind the candidate certificate to model states**

Keep routing certificate verification on routed frames.  Build and verify
candidate state records from `model_state_for_k`.

**Step 6: Run focused handoff/orchestration tests and verify GREEN**

**Step 7: Commit**

```bash
git add kp/kp/projection_handoff.py tests/kp/test_gamma_projection_handoff.py tests/kp/test_selection_orchestration.py
git commit -m "feat(kp): certify Gamma routing and model frames"
```

### Task 4: Produce model-frame Heff and exactified actions

**Files:**

- Modify: `kp/kp/gamma_auto_producer.py`
- Modify: `tests/kp/test_gamma_auto_producer.py`

**Step 1: Resolve one SCDM anchor contract per candidate**

Use the candidate joint bands, routed ranks, actual reference k/Q, and existing
auto-gauge defaults.  Do not add a material branch or a new physical truncation.

**Step 2: Build model frames from already computed eigensystems**

For every k/Q, materialize the compact model frame and local bridge.  Fail the
candidate if its model/routing projectors differ or the bridge loses rank.

**Step 3: Downfold in the model frame**

Assemble model low frames and the certified common high complement.  Check
orthogonality/completeness, then call the existing downfolding core.  Certify
bridge Heff covariance against the routed-coordinate result.

**Step 4: Recompute candidate actions from model states**

Feed model `u_low` and model `Heff` to exactification, candidate symmetry
certification, metrics, base basis identity, and the v3 handoff factory.

**Step 5: Run producer tests and verify GREEN**

Include the original producer suite plus the new k-dependent-mixing regression.

**Step 6: Commit**

```bash
git add kp/kp/gamma_auto_producer.py tests/kp/test_gamma_auto_producer.py
git commit -m "fix(kp): hand off automatic Gamma in the SCDM model gauge"
```

### Task 5: Switch runtime and symmetry consumers to model states

**Files:**

- Modify: `kp/kp/gamma_auto_runtime.py`
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `tests/kp/test_gamma_auto_runtime.py`
- Modify: `tests/kp/test_gamma_projection_handoff.py`

**Step 1: Write failing spin and symmetry consumer tests**

Construct routing/model frames that give distinguishable projected spin and
symmetry states.  Require runtime and kp symmetry to use the model frame.

**Step 2: Verify RED**

Expected: current consumers call the routed `assemble_for_k` API.

**Step 3: Update runtime payloads**

Project spin with `assemble_model_for_k`; derive wavefunctions from authoritative
model Heff; retain all current transaction and atomic-write behavior.

**Step 4: Update kp symmetry reconstruction and reporting**

Use `model_state_for_k` for production states.  Report routing, model, bridge,
and anchor-contract hashes.  Keep route/rank metadata from routing fields.

**Step 5: Run runtime/handoff/symmetry tests and verify GREEN**

**Step 6: Commit**

```bash
git add kp/kp/gamma_auto_runtime.py kp/kp/symmetry/projection.py \
  tests/kp/test_gamma_auto_runtime.py tests/kp/test_gamma_projection_handoff.py
git commit -m "fix(kp): consume certified Gamma model frames"
```

### Task 6: Guard fit policy and run local verification

**Files:**

- Modify: `tests/kp/test_example_config_contract.py` or the existing canonical
  example contract file selected during implementation

**Step 1: Add the MgI2 fit-policy contract**

Assert linear fitting, indices `[0, 40]`, ten bands, zero one/two-sided weights,
and no nonlinear/refinement options for the automatic q04 acceptance config.

**Step 2: Run the complete focused suite**

```bash
REPO=/data/work/zy/software/1.tapw_code/moirekp-release
PY=/data/home/zy/mambaforge/envs/moirekp/bin/python
PYTHONPATH="$REPO/kp:$REPO/tapw" "$PY" -m pytest -q -p no:cacheprovider \
  tests/kp/test_auto_gauge_selection.py \
  tests/kp/test_gamma_projector_closure.py \
  tests/kp/test_gamma_projection_handoff.py \
  tests/kp/test_gamma_auto_producer.py \
  tests/kp/test_gamma_auto_runtime.py \
  tests/kp/test_selection_orchestration.py \
  tests/kp/test_complete_response_basis.py
```

Expected: all pass with no collection warnings.

**Step 3: Run the release suite**

```bash
PYTHONPATH="$REPO/kp:$REPO/tapw" "$PY" -m pytest -q -p no:cacheprovider \
  -m "not slow and not external_data"
```

Classify any pre-existing release failures separately; do not modify unrelated
TAPW tests as part of this fix.

**Step 4: Commit the policy test if it changed tracked files**

### Task 7: Big-memory MgI2 physical acceptance

**Files:**

- Create only ignored validation artifacts under distinct
  `validation_runs/gamma_dual_frame_mgi2_20260802/{bigmem001,bigmem003}` paths.

**Step 1: Freeze provenance**

Record commit SHA, config hash, Python/module provenance, fit policy, and output
paths before launching either run.

**Step 2: Run bigmem001**

Run complete `kp project`, `kp symm`, and the original two-point linear
`kp model`.  Do not add fit points or nonlinear refinement.

**Step 3: Evaluate hard gates**

Require RMS `<= 2.31 meV`, max `<= 4.10 meV`, selected bands/ranks/dimension,
spectral equality, Hermiticity, bridge, projector, and artifact identity gates
from the design.

**Step 4: Run the same SHA/config on bigmem003**

Use a separate output directory and compare physical arrays and reported ranks.

**Step 5: Inspect both band plots**

Visually check the full path and top-10 window in addition to numeric gates.

### Task 8: Independent review and final audit

**Files:** all changed files from Tasks 1–6.

**Step 1: Run specification review**

Confirm that routing remains the selection certificate, model frames own all
production matrices, no fit-policy changes occurred, and no material special
case exists.

**Step 2: Run code-quality review**

Inspect frame ordering, antiunitary conventions, hash coverage, archive dtype
strictness, memory scaling, and error messages.

**Step 3: Resolve findings and rerun affected tests**

**Step 4: Inspect Git scope**

Ensure only KP implementation/tests/docs are staged.  Preserve the existing
unrelated TAPW working-tree changes and all historical validation artifacts.

**Step 5: Report**

Provide commits, tests, material metrics, plot paths, and any remaining Gamma
material risks before push.
