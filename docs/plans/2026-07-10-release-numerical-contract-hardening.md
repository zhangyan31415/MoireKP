# Release Numerical Contract Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make TAPW/KP production artifacts numerically self-identifying, reject inconsistent symmetry/basis data, and correct standalone and TAPW topology boundary handling.

**Architecture:** Existing NPZ outputs carry versioned identity metadata; no new JSON report is added. TAPW builds reciprocal-boundary operators from the projected atomic Bloch gauge, while KP propagates one basis identity from projection through symmetry and model export. All production consumers hard fail on missing, stale, or inconsistent identity and validation metadata.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse matrices, pytest, YAML/JSON metadata embedded in NPZ.

---

## Working-tree discipline

- Work on `codex/cpc-release-clean`, as requested.
- Never use `git add .` or stage an entire dirty file without reviewing its pre-existing diff.
- Do not stage `paper/`, `examples/**/outputs`, `examples/**/runs`, archives, validation data, or experimental interlayer campaign files.
- Before every commit run `git diff --check` and inspect `git diff --cached --stat` plus `git diff --cached`.
- Keep each task in a separate commit.

### Task 1: Correct standalone topology coordinates and band subspaces

**Files:**
- Modify: `kp/kp/model/export.py`
- Test: `tests/kp/test_standalone_export.py`

**Step 1: Write failing tests**

Add tests that export and execute a standalone model with:

- dimension 54 and a two-band topology set;
- non-identity oblique `model_reciprocal_basis`;
- a recording model that proves `k = kappa1*bM1 + kappa2*bM2`;
- WCC sewing shifts equal to the selected reciprocal vector.

Assert the selected subspace has shape `(54, 2)` and its projector is `(54, 54)`.

**Step 2: Verify red**

```bash
PYTHONPATH=$PWD/tapw:$PWD/kp PYTHONDONTWRITEBYTECODE=1 \
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
tests/kp/test_standalone_export.py -k "topology and (subspace or reciprocal or wcc)"
```

Expected: failures showing `(2,54)` selection and unit-coordinate shifts.

**Step 3: Implement**

- Replace scalar-plus-advanced indexing with `eigvecs[i, j][:, band_indices]` everywhere.
- Build Cartesian mesh points with the saved reciprocal basis.
- Keep derivatives in fractional coordinates, then transform BC/QGT tensors to Cartesian coordinates.
- Use `bM1`/`bM2` for WCC boundary shifts.
- Validate reciprocal basis shape, rank, and finiteness.

**Step 4: Verify green and commit**

```bash
git add -- kp/kp/model/export.py tests/kp/test_standalone_export.py
git commit -m "fix(kp): correct standalone topology coordinates"
```

### Task 2: Add versioned artifact identity primitives

**Files:**
- Create: `tapw/tapw/identity.py`
- Create: `kp/kp/identity.py`
- Test: `tests/tapw/test_artifact_identity.py`
- Test: `tests/kp/test_artifact_identity.py`

**Step 1: Write failing tests**

Cover deterministic canonical JSON hashing, streamed file hashing, contiguous
array hashing independent of view layout, and strict metadata comparison.

**Step 2: Implement minimal identity API**

Provide:

```python
hash_file(path) -> str
hash_array(array) -> str
hash_mapping(mapping) -> str
require_identity_fields(metadata, fields, context) -> dict
require_matching_identity(left, right, fields, context) -> None
```

Use SHA-256, explicit schema strings, stable key ordering, dtype and shape in
array hashes, and chunked file reads.

**Step 3: Verify and commit**

```bash
git add -- tapw/tapw/identity.py kp/kp/identity.py \
tests/tapw/test_artifact_identity.py tests/kp/test_artifact_identity.py
git commit -m "feat: add strict artifact identity helpers"
```

### Task 3: Bind TAPW symmetry caches and raw-H packs to their inputs

**Files:**
- Modify: `tapw/tapw/symm_rep.py`
- Modify: `tapw/tapw/workflows/symm_rep.py`
- Modify: `tapw/tapw/workflows/symmetry.py`
- Test: `tests/tapw/test_tapw_symm_rep.py`
- Test: `tests/tapw/test_symmetry_analysis_runner_outputs.py`

**Step 1: Write failing tests**

Verify cache rejection after changing each of H, S, structure, q-shell, spin,
point list, band count, or physics configuration. Verify packed raw-H metadata
contains `identity_schema`, `input_hash`, `config_hash`, `basis_hash`, and
package/schema version. Verify a changed hash is rejected even when dimensions
match.

**Step 2: Implement identity production and validation**

- Extend the existing TAPW `basis_hash`; do not replace it.
- Compute input/config identity once per configured run.
- Require exact cache request compatibility, including requested band count.
- Validate the raw-H pack identity before symm-rep projection.
- Resolve the current `write_cache=True/False` test drift in favor of validated
  cache writes and invalidation.

**Step 3: Verify and commit**

```bash
git add -- tapw/tapw/symm_rep.py tapw/tapw/workflows/symm_rep.py \
tapw/tapw/workflows/symmetry.py tests/tapw/test_tapw_symm_rep.py \
tests/tapw/test_symmetry_analysis_runner_outputs.py
git commit -m "fix(tapw): bind symmetry artifacts to source identity"
```

### Task 4: Enforce the TAPW production raw-H validation gate

**Files:**
- Modify: `tapw/tapw/config/schema.py`
- Modify: `tapw/tapw/workflows/symmetry.py`
- Test: `tests/tapw/test_symmetry_analysis_config.py`
- Test: `tests/tapw/test_symmetry_analysis_runner_outputs.py`

**Step 1: Write failing tests**

- An operation with failed H covariance is absent from production NPZ.
- An operation with missing residual is absent from production NPZ.
- For nonorthogonal H/S, passing H but failing S is absent.
- Identity remains exported and explicitly passed.
- Historical validation aliases are rejected by the release schema.

**Step 2: Implement strict production filtering**

- Make release symmetry validation strict by default.
- Validate both H and S when S participates in the generalized problem.
- Pack only `status=passed`, supported, internal operations.
- Keep failed rows in `residuals.csv`/`summary.md` only.
- Make symm-rep and KP readers reject non-passed matrix metadata.

**Step 3: Verify and commit**

```bash
git add -- tapw/tapw/config/schema.py tapw/tapw/workflows/symmetry.py \
tests/tapw/test_symmetry_analysis_config.py \
tests/tapw/test_symmetry_analysis_runner_outputs.py
git commit -m "fix(tapw): gate production symmetry on covariance"
```

### Task 5: Build general projected atomic Bloch boundary operators

**Files:**
- Create: `tapw/tapw/symmetry/periodic_gauge.py`
- Modify: `tapw/tapw/workflows/symmetry.py`
- Modify: `tapw/tapw/workflows/band.py`
- Modify: `tapw/tapw/chern_post.py`
- Test: `tests/tapw/test_chern_post_geometry.py`
- Test: `tests/tapw/test_symmetry_analysis_runner_outputs.py`
- Test: `tests/tapw/test_tapw_chern_workflow.py`

**Step 1: Extract existing periodic-gauge construction with unchanged tests**

Move atomic phase and `G P_b G^dagger` construction into the new module so
source symmetry and topology call the same implementation.

**Step 2: Write failing topology tests**

Construct synthetic projectors with unequal group dimensions representing
1+1, 1+2, and 1+3 layouts. Assert both b1 and b2 operators have the full TAPW
dimension, preserve the expected projected action, and do not infer a common
per-G internal dimension.

**Step 3: Implement topology artifact and WCC use**

- Build b1/b2 projected atomic Bloch operators from the active calculator.
- Store both sparse operators and identity metadata in one
  `boundary_sewing.npz` inside the topology grid.
- Load only identity-matching operators in canonical WCC.
- Remove canonical fallback to naked overlap or partial G relabel.
- Check boundary occupied-subspace singular values and fail below threshold.
- Keep the old G-relabel helper diagnostic-only and unreachable from canonical
  config tasks.

**Step 4: Verify and commit**

```bash
git add -- tapw/tapw/symmetry/periodic_gauge.py \
tapw/tapw/workflows/symmetry.py tapw/tapw/workflows/band.py \
tapw/tapw/chern_post.py tests/tapw/test_chern_post_geometry.py \
tests/tapw/test_symmetry_analysis_runner_outputs.py \
tests/tapw/test_tapw_chern_workflow.py
git commit -m "fix(tapw): use projected Bloch boundary sewing"
```

### Task 6: Propagate one KP basis identity through project, symm, and model

**Files:**
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `kp/kp/model/pipeline.py`
- Modify: `kp/kp/model/export.py`
- Test: `tests/kp/test_cli_project_memory.py`
- Test: `tests/kp/test_symm_projection.py`
- Test: `tests/kp/test_configured_model.py`
- Test: `tests/kp/test_standalone_export.py`

**Step 1: Write failing identity tests**

Assert project outputs share one hash, symm refuses a different projection
basis, model refuses projection/symmetry mismatch, and `model_data.npz`
contains the accepted hash.

**Step 2: Implement KP basis identity**

Hash source TAPW identity, Q sets, mode/spin, nlow states, resolved gauge
anchors, row layout, and dimensions. Store metadata inside existing NPZ files.
Production mode rejects deferred symmetry validation and missing hashes.

**Step 3: Verify and commit**

```bash
git add -- kp/kp/cli.py kp/kp/symmetry/projection.py \
kp/kp/model/pipeline.py kp/kp/model/export.py \
tests/kp/test_cli_project_memory.py tests/kp/test_symm_projection.py \
tests/kp/test_configured_model.py tests/kp/test_standalone_export.py
git commit -m "fix(kp): enforce one production basis identity"
```

### Task 7: Preserve and remap projection k-point provenance

**Files:**
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_cli_project_memory.py`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing tests**

Project source indices `[5, 10]`; verify model source index 10 selects Heff row
1, and source index 0 fails because it was not projected. Reject duplicate
projection indices and ragged/object Heff output.

**Step 2: Implement explicit source-index mapping**

- Persist `k_indices` in `basis.npz` as well as wavefunctions.
- Build a source-index-to-Heff-row map during model load.
- Interpret fit/band indices as source k-point indices.
- Validate fixed numerical Heff shape before writing and loading.

**Step 3: Verify and commit**

```bash
git add -- kp/kp/cli.py kp/kp/model/pipeline.py \
tests/kp/test_cli_project_memory.py tests/kp/test_configured_model.py
git commit -m "fix(kp): preserve projected k-point provenance"
```

### Task 8: Tighten exactified-matrix production provenance

**Files:**
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `kp/kp/model/pipeline.py`
- Modify: `kp/kp/model/symmetry.py`
- Test: `tests/kp/test_exactify_representation.py`
- Test: `tests/kp/test_configured_model.py`
- Test: `tests/kp/test_symm_projection.py`

**Step 1: Write failing tests**

Independently remove or corrupt schema, owner, status, matrix kind, matrix
source, basis hash, and exactification report. Every variant must fail model
loading. Verify diagnostic/raw matrices cannot satisfy the gate.

**Step 2: Implement strict AND validation**

Replace permissive OR logic with a single validator requiring all production
fields and matching basis identity. Remove synthesized exactified provenance
for metadata-free packs.

**Step 3: Verify and commit**

```bash
git add -- kp/kp/symmetry/projection.py kp/kp/model/pipeline.py \
kp/kp/model/symmetry.py tests/kp/test_exactify_representation.py \
tests/kp/test_configured_model.py tests/kp/test_symm_projection.py
git commit -m "fix(kp): require strict exactification provenance"
```

### Task 9: Final integration and real validation

**Files:**
- Modify tests/docs only if failures expose an actual contract change.

**Step 1: Run focused suites**

```bash
PYTHONPATH=$PWD/tapw:$PWD/kp PYTHONDONTWRITEBYTECODE=1 \
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
tests/kp/test_standalone_export.py tests/kp/test_cli_project_memory.py \
tests/kp/test_symm_projection.py tests/kp/test_exactify_representation.py \
tests/kp/test_configured_model.py tests/tapw/test_artifact_identity.py \
tests/tapw/test_chern_post_geometry.py tests/tapw/test_tapw_chern_workflow.py \
tests/tapw/test_tapw_symm_rep.py tests/tapw/test_symmetry_analysis_runner_outputs.py
```

**Step 2: Run complete fast tests explicitly**

```bash
PYTHONPATH=$PWD/tapw:$PWD/kp PYTHONDONTWRITEBYTECODE=1 \
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
tests -m "not slow and not external_data"
```

Any remaining tests for removed private scripts must be classified and fixed;
they may not be hidden by `pytest.ini`.

**Step 3: Run real-data validation**

On the designated compute nodes, rerun one MoTe2 K case, one MgI2 Gamma/M
case, and one unequal-source-group multilayer case. Record only local
validation artifacts. Check symmetry covariance, identity propagation, BC/QGT
invariants, and WCC boundary singular values.

**Step 4: Final audit commit**

Commit only any necessary release-facing tests/docs with an explicit message.
