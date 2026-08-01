# CPC Full Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver lossless response storage, certified automatic low-energy selection, complete TAPW `system.structure` support, and release-ready example contracts through isolated feature worktrees and a single validated CPC integration hub.

**Architecture:** Preserve `codex/cpc-release-clean@847e32c` as the immutable source baseline. Implement independent components in separate worktrees with non-overlapping file ownership, then review and cherry-pick narrow commits into `codex/cpc-full-integration`. Run the full local release gate before validating the exact hub SHA on `bigmem001` and `bigmem003`.

**Tech Stack:** Python 3.11, NumPy, SciPy, sparse matrices, ASE, PyYAML, pytest, Git worktrees, MKL-backed `moirekp` conda environment.

---

## Execution Rules

- Keep `/data/work/zy/software/1.tapw_code/moirekp-release` on
  `codex/cpc-release-clean`; do not edit it.
- Treat `.worktrees/kp-auto-model-selection-preserved` as read-only evidence.
- Do not merge or rebase `codex/kp-auto-model-selection`.
- Do not copy complete preserved source files over CPC files.
- Use `apply_patch` for edits.
- Run tests with `/data/home/zy/mambaforge/envs/moirekp/bin/python` and an
  explicit worktree-local `PYTHONPATH`.
- Do not set global BLAS/OpenMP thread environment variables.
- Commit only source, tests, small documentation, and approved configs.
- Keep all external runs in `validation_runs/`.

## Parallelization Map

Tasks 2, 3-5, 6, and 7 may begin in parallel in independent worktrees.
Task 8 depends on Tasks 6 and 7. Tasks 9 and 10 depend on Tasks 6-8. Task 11
begins only after public interfaces stabilize. Task 12 is the single integration
checkpoint. Tasks 13-14 are final validation and release integration.

### Task 1: Create Isolated Feature Worktrees

**Files:** None.

**Step 1: Verify the base and ignore rule**

Run from `/data/work/zy/software/1.tapw_code/moirekp-release`:

```bash
git status --short --branch
git rev-parse HEAD
git check-ignore -v .worktrees
```

Expected: clean `codex/cpc-release-clean`, SHA `847e32c...`, and an ignore rule
for `/.worktrees/`.

**Step 2: Create feature branches from the exact base**

```bash
git worktree add .worktrees/cpc-storage-compression \
  -b codex/cpc-storage-compression 847e32cca35c62d17f84e86916b81e60bb0a743e
git worktree add .worktrees/cpc-auto-selection-core \
  -b codex/cpc-auto-selection-core 847e32cca35c62d17f84e86916b81e60bb0a743e
git worktree add .worktrees/cpc-candidate-symmetry \
  -b codex/cpc-candidate-symmetry 847e32cca35c62d17f84e86916b81e60bb0a743e
git worktree add .worktrees/cpc-tapw-system-structure \
  -b codex/cpc-tapw-system-structure 847e32cca35c62d17f84e86916b81e60bb0a743e
git worktree add .worktrees/cpc-example-contracts \
  -b codex/cpc-example-contracts 847e32cca35c62d17f84e86916b81e60bb0a743e
```

Expected: every new worktree is clean and points at the approved base.

**Step 3: Record ownership**

Before dispatching, state in each agent prompt the exact writable files and the
files it must not touch. No commit is required for this task.

### Task 2: Add Lossless Model-Data Compression

**Worktree:** `.worktrees/cpc-storage-compression`

**Files:**
- Modify: `kp/kp/model/export.py:1701`
- Create: `tests/kp/test_model_data_storage.py`

**Step 1: Write the failing storage test**

```python
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from kp.model.export import _write_npz


def test_model_data_writer_uses_lossless_zip_deflate(tmp_path):
    path = tmp_path / "model_data.npz"
    arrays = {
        "zeros": np.zeros(4096, dtype=np.float64),
        "indices": np.arange(1024, dtype=np.int64),
    }
    _write_npz(path, arrays)
    with ZipFile(path) as archive:
        assert archive.infolist()
        assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())
    with np.load(path, allow_pickle=False) as restored:
        assert set(restored.files) == set(arrays)
        for key, expected in arrays.items():
            np.testing.assert_array_equal(restored[key], expected)
```

Do not copy the preserved sparse-cleanup tests.

**Step 2: Verify the test fails**

Run:

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/kp/test_model_data_storage.py::test_model_data_writer_uses_lossless_zip_deflate
```

Expected: FAIL because `_write_npz` uses uncompressed ZIP entries.

**Step 3: Implement the minimal change**

Change only `_write_npz`:

```python
def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
```

Do not change `COMPILER_VERSION` or response coefficients.

**Step 4: Run focused export tests**

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/kp/test_model_data_storage.py \
  tests/kp/test_standalone_export.py \
  tests/kp/test_kp_artifact_identity.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/model/export.py tests/kp/test_model_data_storage.py
git commit -m "perf(kp): compress standalone model data losslessly"
```

### Task 3: Define Strict TAPW `system:` Normalization

**Worktree:** `.worktrees/cpc-tapw-system-structure`

**Files:**
- Modify: `tapw/tapw/config/schema.py`
- Modify: `tests/tapw/test_config_paths.py`
- Create: `tests/tapw/test_structure_input.py`

**Step 1: Add failing canonical-config tests**

Test the exact tracked schema used by release configs:

```yaml
system:
  output: ../outputs
  structure: ../../openmx/soc/POSCAR_rigid
  hamiltonian: ../../openmx/soc/H_symm.npz
  overlap: ../../openmx/soc/S_symm.npz
  orbitals: {Mg: s2p2, I: s3p2d2}
  twist_index: 8
  layers: [1, 1]
  spin: true
```

Tests must assert:

- paths resolve relative to the YAML file;
- normalized compute fields are populated;
- unknown `system` fields fail;
- legacy and canonical structure inputs cannot be mixed;
- tracked MgI2, MoTe2, AAB, ZrS2, PtSe2, and BiTeI configs parse.

**Step 2: Run tests and confirm failure**

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/tapw/test_config_paths.py tests/tapw/test_structure_input.py
```

Expected: the new canonical tests fail because current normalization is
incomplete.

**Step 3: Add typed normalized system fields**

Implement one normalized representation instead of passing unvalidated dicts.
The representation must bind:

```python
@dataclass(frozen=True)
class SystemInputConfig:
    output: Path
    structure: Path
    hamiltonian: Path
    overlap: Path | None
    orbitals: Mapping[str, str]
    twist_index: int
    layers: tuple[int, ...]
    spin: bool
```

Validate required fields, reject unknown fields, and preserve legacy loading by
converting it to the same internal path.

**Step 4: Re-run config tests**

Expected: PASS for canonical and legacy cases; invalid mixed/unknown cases fail
with specific messages.

**Step 5: Commit schema normalization**

```bash
git add tapw/tapw/config/schema.py tests/tapw/test_config_paths.py tests/tapw/test_structure_input.py
git commit -m "feat(tapw): normalize canonical system inputs"
```

### Task 4: Resolve Structure, Orbitals, Bravais, And Basis Dimension

**Worktree:** `.worktrees/cpc-tapw-system-structure`

**Files:**
- Modify: `tapw/tapw/io/structure.py`
- Modify: `tapw/tapw/io/hr.py`
- Modify: `tests/tapw/test_structure_input.py`
- Modify: `tests/tapw/test_gen_h_new.py`

**Step 1: Add failing structure-resolution tests**

Cover:

- POSCAR and CIF preserve site order;
- compact orbital counts cover all and only present species;
- spin doubles the expected basis dimension;
- H/S dimension mismatch fails before diagonalization;
- hexagonal and square metrics infer correctly;
- ambiguous metrics fail closed;
- sequentially resolving two Bravais families does not share global state.

Use small temporary structures and matrices; do not use external data.

**Step 2: Verify tests fail**

Run the two targeted test files and record the exact failures.

**Step 3: Implement `ResolvedStructureInput`**

```python
@dataclass(frozen=True)
class ResolvedStructureInput:
    structure_path: Path
    structure_sha256: str
    symbols: tuple[str, ...]
    lattice: np.ndarray
    scaled_positions: np.ndarray
    orbital_counts: tuple[int, ...]
    spin: bool
    basis_dimension: int
    bravais: str
    provenance: Mapping[str, object]
```

Load through ASE without sorting or standardizing sites. Infer Bravais through a
pure function with explicit tolerances. Do not write `TAPW_BRAVAIS`.

**Step 4: Add early H/S validation**

Validate square H and S dimensions against `basis_dimension`, including spin,
before constructing a workflow calculator.

**Step 5: Re-run targeted tests**

Expected: PASS.

**Step 6: Commit**

```bash
git add tapw/tapw/io/structure.py tapw/tapw/io/hr.py \
  tests/tapw/test_structure_input.py tests/tapw/test_gen_h_new.py
git commit -m "feat(tapw): resolve structure and basis identity"
```

### Task 5: Integrate `system.structure` Across TAPW Workflows

**Worktree:** `.worktrees/cpc-tapw-system-structure`

**Files:**
- Modify: `tapw/tapw/cli.py`
- Modify: `tapw/tapw/workflows/band.py`
- Modify: `tapw/tapw/workflows/symmetry.py`
- Modify: `tapw/tapw/symm_rep.py`
- Modify: `tapw/tapw/chern_post.py`
- Modify: `tapw/tapw/orbital_analysis_tool.py`
- Modify: `tapw/tapw/templates/config.yaml`
- Modify: `tests/tapw/test_cli_surface.py`
- Modify: `tests/tapw/test_symmetry_analysis_main.py`
- Modify: `tests/tapw/test_tapw_symm_rep.py`
- Modify: `tests/tapw/test_chern_post_geometry.py`

**Step 1: Add failing workflow identity tests**

Build a small resolved input and assert `run`, `symm`, `symm-rep`, `topo`, and
orbital analysis receive the same source identity and site/basis order.

**Step 2: Add a failing save/load round-trip test**

Load a tracked canonical config, save it, reload it, and compare normalized
system and workflow fields. Ensure the saved YAML does not contain rejected
legacy-only sections.

**Step 3: Verify failures**

Run the listed targeted tests.

**Step 4: Route all workflows through `ResolvedStructureInput`**

Remove workflow-local structure inference. Preserve TAPW raw-H symmetry output
and legacy config compatibility. Do not introduce interlayer scaling or local
campaign options.

**Step 5: Update the package template minimally**

Use the canonical `system:` schema and portable relative placeholders. Do not
include machine paths or node names.

**Step 6: Run TAPW targeted tests**

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/tapw/test_config_paths.py \
  tests/tapw/test_structure_input.py \
  tests/tapw/test_cli_surface.py \
  tests/tapw/test_symmetry_analysis_main.py \
  tests/tapw/test_tapw_symm_rep.py \
  tests/tapw/test_chern_post_geometry.py
```

Expected: PASS.

**Step 7: Commit**

```bash
git add tapw tests/tapw
git commit -m "feat(tapw): use canonical structures in public workflows"
```

Review the staged list before committing; remove any unrelated TAPW diff.

### Task 6: Implement Fixed Target Windows And Principal-Angle Metrics

**Worktree:** `.worktrees/cpc-auto-selection-core`

**Files:**
- Create: `kp/kp/projection_selection.py`
- Create: `tests/kp/test_projection_selection.py`

**Step 1: Add failing target-window tests**

Required cases:

```python
def test_candidate_smaller_than_fixed_window_is_rejected(): ...
def test_single_band_error_keeps_absolute_edge_shift(): ...
def test_every_candidate_uses_identical_target_band_ids(): ...
def test_window_boundary_cannot_split_degenerate_multiplet(): ...
```

The one-band test must give a nonzero 20 meV error for a 20 meV shift.

**Step 2: Verify failures**

Run `tests/kp/test_projection_selection.py` and expect import/test failures.

**Step 3: Implement target-window types and errors**

```python
@dataclass(frozen=True)
class TargetWindowSpec:
    edge: Literal["valence", "conduction"]
    band_count: int
    validation_k_indices: tuple[int, ...]
    energy_reference_ev: float
    degeneracy_tolerance_ev: float


class CandidateRejected(RuntimeError):
    pass
```

Implement fixed band-id resolution once per run and compare all candidates to
those ids using one energy reference.

**Step 4: Add failing principal-angle tests**

Cover orthogonal subspaces despite perfect anchor conditioning, unitary gauge
rotations, unequal ambient dimension, and one bad validation k point.

**Step 5: Implement principal-angle metrics**

Use singular values of `target_vectors.conj().T @ model_vectors`. Return minimum
squared overlap, mean overlap, and target capture separately.

**Step 6: Run tests and commit**

```bash
git add kp/kp/projection_selection.py tests/kp/test_projection_selection.py
git commit -m "feat(kp): add fixed-window projection metrics"
```

### Task 7: Implement Typed Candidate Symmetry Certification

**Worktree:** `.worktrees/cpc-candidate-symmetry`

**Files:**
- Modify: `kp/kp/symmetry/projection.py`
- Create: `kp/kp/symmetry/candidate_certificate.py`
- Create: `tests/kp/test_candidate_symmetry_certificate.py`
- Modify: `tests/kp/test_symm_projection.py`

**Step 1: Add failing per-operation certificate tests**

Cover:

- a selected subspace whose raw-H image leaks out;
- a projectable raw action far from the exactified action;
- antiunitary TR and Kramers closure;
- missing operation/pair coverage;
- NaN metrics;
- a small failing candidate and larger passing candidate.

**Step 2: Define typed records**

```python
@dataclass(frozen=True)
class OperationPairCertificate:
    operation: str
    source_k_index: int
    target_k_index: int
    raw_h_leakage: float
    exactification_distance: float
    intertwining_residual: float
    heff_covariance_residual: float


@dataclass(frozen=True)
class CandidateSymmetryCertificate:
    pairs: tuple[OperationPairCertificate, ...]
    operation_coverage: Mapping[str, int]
    relation_residuals: Mapping[str, float]
    antiunitary_square_residuals: Mapping[str, float]
    certificate_hash: str
```

**Step 3: Extract a pure in-memory evaluator**

Reuse CPC `_project_operation` and joint exactification. Do not write files and
do not inspect previously generated JSON. Validate finite metrics and complete
coverage before constructing the certificate.

**Step 4: Run targeted symmetry tests**

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/kp/test_candidate_symmetry_certificate.py \
  tests/kp/test_symm_projection.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/symmetry/projection.py kp/kp/symmetry/candidate_certificate.py \
  tests/kp/test_candidate_symmetry_certificate.py tests/kp/test_symm_projection.py
git commit -m "feat(kp): certify candidate source symmetries"
```

### Task 8: Add Gauge-Invariant Gamma Projector Closure

**Worktree:** Start from the integrated Tasks 6 and 7 in a fresh branch or update
`codex/cpc-auto-selection-core` only after the symmetry commit is reviewed.

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `kp/kp/blocks/blocks.py`
- Modify: `kp/kp/projection_selection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`
- Modify: `tests/kp/test_blocks_get_h_block.py`

**Step 1: Add failing Gamma closure tests**

Construct a synthetic spinful `num_layer_list=[1, 2]` Gamma case with the real
projector assembler. Assert:

- TR closes a seed to its Kramers partner;
- `U_low.conj().T @ U_low` is identity;
- projected eigenvalues retain expected degeneracy;
- random unitary rotation inside the degenerate seed does not change selection;
- L2 orbital swap transforms the projector covariantly;
- layer composition may change but ownership does not.

**Step 2: Verify failures**

Run the two targeted files and record the failing invariants.

**Step 3: Expose one canonical row-layout object**

Extend the existing CPC row-order helper to return same-Q local row, full row,
source group, physical layer, spin, and orbital. Keep the current projector
ordering logic unchanged.

**Step 4: Implement projector-level closure**

Cluster by energy/degeneracy, close the joint projector under raw-H actions, and
only then resolve source-group routing. Use layer weights only for composition
reporting. Ambiguous routing raises `CandidateRejected`.

**Step 5: Run targeted and existing Gamma tests**

Expected: PASS, including existing row-order/Kramers tests.

**Step 6: Commit**

```bash
git add kp/kp/low_energy_selection.py kp/kp/blocks/blocks.py \
  kp/kp/projection_selection.py tests/kp/test_auto_low_energy_selection.py \
  tests/kp/test_blocks_get_h_block.py
git commit -m "feat(kp): close Gamma candidates under source symmetry"
```

### Task 9: Implement Selection Artifacts And Fail-Closed Decisions

**Worktree:** Integration branch after Tasks 6-8 are cherry-picked.

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `kp/kp/projection_selection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`
- Create: `tests/kp/test_selection_artifact.py`

**Step 1: Add failing status and transaction tests**

Test:

- automatic selection returns failure when no candidate passes;
- explicit selection begins `PENDING/UNVERIFIED` with null metrics;
- certification upgrades it to PASS;
- a crash before atomic replace cannot leave PASS;
- artifact hashes change with Q, target window, source action, or layout;
- `inspect` and `project` inputs produce the same artifact hash.

**Step 2: Define status and artifact types**

```python
class CertificationStatus(str, Enum):
    PENDING = "PENDING"
    CERTIFIED = "CERTIFIED"
    FAILED = "FAILED"
    UNVERIFIED_OVERRIDE = "UNVERIFIED_OVERRIDE"


@dataclass(frozen=True)
class SelectionArtifact:
    target_window: TargetWindowSpec
    q_identity: str
    row_layout_hash: str
    source_identity: str
    source_action_hash: str
    candidate_id: str
    metrics: CandidateMetrics | None
    symmetry_certificate_hash: str | None
    certification_status: CertificationStatus
    artifact_hash: str
```

**Step 3: Implement fail-closed selection**

Only candidates passing every finite hard metric enter lexicographic selection.
Use dimension first, then band error, overlap deficit, and symmetry residual.
Do not choose a WARN fallback for automatic release output.

**Step 4: Implement strict report serialization and atomic replace**

Serialize missing values as JSON null plus a reason. Write to a sibling
temporary file, fsync/close as appropriate, then use `Path.replace()`.

**Step 5: Run tests and commit**

```bash
git add kp/kp/low_energy_selection.py kp/kp/projection_selection.py \
  tests/kp/test_auto_low_energy_selection.py tests/kp/test_selection_artifact.py
git commit -m "feat(kp): freeze certified low-energy selections"
```

### Task 10: Integrate Selection Into `kp inspect` And `kp project`

**Worktree:** `.worktrees/cpc-full-integration`

**Files:**
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/config/case.py` only if typed selection config is required
- Modify: `tests/kp/test_project_auto_gauge_cli.py`
- Modify: `tests/kp/test_canonical_output_layout.py`
- Modify: `tests/kp/test_kp_artifact_identity.py`

**Step 1: Add failing CLI orchestration tests**

Mock only external I/O, not the projector/symmetry evaluator. Assert:

- `inspect` and `project` call the same selector;
- selected artifact identity is preserved;
- missing symmetry input fails before model output;
- explicit unverified projection is allowed but release status is BLOCKED;
- no stale PASS survives a later failure.

**Step 2: Verify failures**

Run the three targeted files.

**Step 3: Replace dirty-style CLI physics with orchestration**

Load canonical inputs, call the shared selector, freeze the artifact, run
production projection with the same resolved candidate, validate identity, and
write the final report. Do not port the preserved 1700-line CLI block.

**Step 4: Run KP targeted tests**

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/kp/test_projection_selection.py \
  tests/kp/test_candidate_symmetry_certificate.py \
  tests/kp/test_auto_low_energy_selection.py \
  tests/kp/test_selection_artifact.py \
  tests/kp/test_project_auto_gauge_cli.py \
  tests/kp/test_canonical_output_layout.py \
  tests/kp/test_kp_artifact_identity.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/cli.py kp/kp/config/case.py tests/kp
git commit -m "feat(kp): orchestrate certified automatic projection"
```

Review staging and exclude unrelated test files.

### Task 11: Curate Release Examples And Contracts

**Worktree:** `.worktrees/cpc-example-contracts`, rebased or recreated from the
reviewed integration hub after public interfaces stabilize.

**Files:**
- Modify: `examples/README.md`
- Modify: selected `examples/**/README.md`
- Modify: selected small `examples/**/*.yaml`
- Modify: `examples/data-manifest.yaml`
- Modify: `README.md` and `README.zh.md` only for validated public behavior
- Modify: `tests/kp/test_example_dependency_contract.py`
- Modify: `tests/test_release_contract.py`

**Step 1: Add failing contract tests**

Assert:

- all tracked canonical YAML files parse;
- documented output paths match implementation;
- external files have explicit external markers;
- release-facing files contain no local absolute paths or node names;
- forbidden operation labels are absent;
- license text remains present.

**Step 2: Verify failures against the final interface**

Run release-contract tests.

**Step 3: Apply minimal documentation/config changes**

Document certified automatic selection, explicit unverified status, and
`system.structure`. Do not copy preserved README or manifest wholesale.

**Step 4: Recompute small-file metadata**

For every changed tracked config, update exact size and SHA-256. Leave license,
DOI, or URL unresolved rather than inventing values.

**Step 5: Run contract tests and commit**

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_release_contract.py tests/kp/test_example_dependency_contract.py
git add README.md README.zh.md examples tests/test_release_contract.py \
  tests/kp/test_example_dependency_contract.py
git commit -m "docs(release): align certified workflow contracts"
```

Inspect the staged list for generated artifacts before committing.

### Task 12: Review And Integrate Narrow Commits

**Worktree:** `.worktrees/cpc-full-integration`

**Files:** All reviewed feature diffs only.

**Step 1: Review each feature commit**

For every candidate commit:

```bash
git show --stat --oneline <sha>
git show --check <sha>
```

Compare protected CPC implementations before cherry-picking. Reject broad file
replacement or unrelated deletions.

**Step 2: Cherry-pick in dependency order**

Storage and TAPW system may arrive independently. Candidate symmetry must
precede Gamma closure and CLI orchestration. Example contracts arrive last.

**Step 3: Run targeted tests after every cherry-pick**

Use the originating feature's exact test command. Stop at the first regression
and send the failure back to the owning agent.

**Step 4: Run the full local release gate**

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Expected: zero failures.

**Step 5: Verify environment and release hygiene**

Run `which python`, `which tapw`, `which kp`, CLI help, `threadpoolctl`, path and
node scans, forbidden-label scans, `git diff --check`, and `git status`.

**Step 6: Record the exact hub SHA**

Do not amend or add commits after external validation begins. Any later change
invalidates the external results and requires rerunning affected cases.

### Task 13: Validate The Exact Hub SHA On `bigmem001` And `bigmem003`

**Files:** Runtime-only files under `validation_runs/`; none are committed.

**Step 1: Preflight both nodes**

Verify hostname is exactly `bigmem001` or `bigmem003`, environment executables
come from `moirekp`, MKL is loaded, repo SHA matches the frozen hub SHA, and all
required external inputs have recorded size/checksum.

Explicitly reject `bigmem002`.

**Step 2: Run `bigmem001` cases**

- MgI2 Gamma spinless/spinful;
- ZrS2 Gamma spinful;
- response cold/warm cache;
- L2 swap and dense-gauge covariance;
- verified 2.13 and 2.45 inputs if present.

**Step 3: Run `bigmem003` cases**

- AAB MoTe2 Gamma spinful;
- PtSe2 Gamma spinful;
- BiTeI Gamma spinful;
- multilayer Gamma row order and Kramers checks;
- K/M non-regression;
- verified 2.28 and 2.65 inputs if present.

**Step 4: Run one common MgI2 Gamma case on both nodes**

Compare artifact hashes, ranks, residuals, and band arrays within declared
numerical tolerances.

**Step 5: Produce a validation summary**

For each case record command, exit code, source checksums, artifact identity,
symmetry coverage/residuals, Hermiticity, projector rank, response rank/span,
gauge covariance, expected degeneracy, Hamiltonian residual, band comparison,
cache reproducibility, and PASS/BLOCKED status.

Missing 2.45/2.65 data is `BLOCKED_EXTERNAL_DATA`, not a substituted run.

### Task 14: Final Release Review And Fast-Forward

**Worktree:** CPC root, only after all previous tasks pass.

**Files:** No new edits.

**Step 1: Verify the root remains clean and unchanged**

```bash
git status --short --branch
git rev-parse HEAD
```

Expected: clean `codex/cpc-release-clean` at the approved pre-integration SHA.

**Step 2: Audit the complete hub range**

```bash
git diff --stat codex/cpc-release-clean..codex/cpc-full-integration
git diff --check codex/cpc-release-clean..codex/cpc-full-integration
git log --oneline codex/cpc-release-clean..codex/cpc-full-integration
```

Confirm no validation output, machine path, node name, old operation label,
paper edit, or binary archive enters the release.

**Step 3: Fast-forward only**

```bash
git merge --ff-only codex/cpc-full-integration
```

If this fails, stop and re-audit topology. Do not create an unreviewed merge.

**Step 4: Re-run the final local gate from the CPC root**

Expected: zero failures and the same validated code SHA/content.

**Step 5: Request explicit push confirmation**

Do not push automatically. Report commits, tests, external validation, and any
blocked external cases to the user first.
