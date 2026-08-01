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
- Create: `kp/kp/blocks/gamma_layout.py`
- Create: `tests/kp/test_gamma_projector_closure.py`
- Modify: `kp/kp/blocks/__init__.py`
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `kp/kp/blocks/blocks.py`
- Modify: `kp/kp/projection_selection.py`
- Modify: `kp/kp/identity.py`
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`
- Modify: `tests/kp/test_blocks_get_h_block.py`
- Modify: `tests/kp/test_symm_projection.py`
- Modify: `tests/kp/test_auto_gauge_layer_layout.py`

**Step 1: Freeze the supported v1 domain**

Automatic Gamma v1 supports exactly two source groups, equal ordered Q counts,
the current release convention of equal orbital count per physical layer, and
`spinful_all` only. Single-spin and cross-spin sewing inputs remain
`ExplicitLegacyBasisSpec` in v1 and reject automatic routing with
`UNSUPPORTED_SPIN_ROUTE`. Reject all other layouts before indexing or
diagonalization. Do not claim support for unequal Q or orbital widths while
production still uses one `q_count` and one `orb_per_layer0`.

The canonical layout hash must bind ordered Q arrays, TAPW source basis
identity, source-group/layer/orbital dimensions, full-row order, and spin-slice
convention. Repeated hand-written Gamma row formulas must become wrappers around
this one layout.

Use the following exact v1 coordinates. Let
`num_layer_list=(L0,L1)`, `Nq=len(Q0)=len(Q1)>0`, every physical layer have
`o>0` orbitals, `O_g=L_g*o`, `Ns=2`, and `s in {0,1}` label the persisted
spin-up/spin-down rows. Define:

```text
N_sigma = Nq * (O0 + O1)
B_g = Nq * sum(O_h for h < g)
C_g_l = sum(O_h for h < g) + l * o
r_local(s,g,l,a) = s * (O0 + O1) + C_g_l + a
r_full(s,q,g,l,a) = s * N_sigma + B_g + q * O_g + l * o + a
```

All addresses must cover `0..full_dim-1` exactly once. The two Q sets use the
same joint `q_index`; every raw action must prove that its two source-group Q
maps yield the same target joint Q, otherwise reject with
`GAMMA_Q_ROUTE_MISMATCH`.

The canonical layout payload is versioned as `kp.gamma-row-layout.v1` and
contains at least:

```text
tapw_source_basis_hash
ordered_qset_hashes[2]
num_layer_list
uniform_orbital_count
source_group_count = 2
spin_scope and semantic spin_labels
full_dim and local_dim
full_rows_by_q_hash
basis_order = spin->group->q->layer->orbital
```

Every raw-H operation must carry the same TAPW basis hash. The source
Hamiltonian content hash is separate from the layout hash but mandatory in the
routed handoff and later selection identity.

**Step 2: Add failing Gamma closure tests**

Construct a synthetic spinful `num_layer_list=[1, 2]` Gamma case with the real
projector assembler. Assert:

- rank-four TR closure has one Kramers pair inside each source group, matching
  production `TR sector_map=identity`;
- `U_low.conj().T @ U_low` is identity;
- projected eigenvalues retain expected degeneracy;
- random unitary rotation inside the degenerate seed does not change selection;
- L2 orbital swap transforms the projector covariantly;
- an independent rotation across physical layers inside source group 1 changes
  layer composition but not source-group routing;
- an intentionally wrong legacy row permutation produces a nonzero TR residual;
- unequal Q/orbital layouts, inconsistent q routes, and whole-frame Procrustes
  mixing across routed groups fail closed.

Production seeds must contain complete energy clusters. A rank-one Kramers seed
may test only the low-level closure primitive, not a production candidate.

The wrong-row permutation test must specifically produce a FAILED candidate
symmetry certificate with raw-H leakage or Heff covariance above its gate. Add
field-by-field handoff tamper tests for Q order, source basis hash, frame bytes,
group offsets, certificate hashes, and missing required k coverage. Preserve a
positive legacy explicit-load test. The whole-frame Procrustes negative case
must demonstrate mixed group projectors while the groupwise algorithm passes.

**Step 3: Verify failures**

Run the targeted files and record the failing invariants.

**Step 4: Expose one strict canonical row-layout object**

Create immutable `GammaRowLayout` / `GammaRowAddress` records returning same-Q
local row, full row, q index, source group, physical layer, spin, and orbital.
Validate a bijection over the full row space. Reuse the layout in
`get_H_block`, `project_heff_full`, the Gamma projector assemblers, and
`symmetry.projection`; do not leave an automatic path that silently collapses
invalid metadata to one row segment.

**Step 5: Implement projector-cluster closure**

Cluster by energy using separate certified-same and certified-different bounds;
the numerical gray zone rejects the candidate. Seeds and additions are whole
cluster projectors. Before applying a raw-H local action, certify isometry,
unique q/sector routing, and off-route leakage. A target cluster is added only
when it uniquely captures essentially the complete normalized image; nonzero
overlap alone is not sufficient. After convergence, check normalized closure
residual for every required generator/q route.

Define one frozen `GammaRoutingThresholds` with at least:

```text
energy_same_ev, energy_different_ev
capture_zero_fraction, capture_loss_max
local_action_isometry, off_route_leakage, closure_residual
route_zero_gap, route_covariance, projector_residual
anchor_sigma_min, max_rank, max_iterations
```

All scalar thresholds are finite. Require
`0 <= energy_same_ev < energy_different_ev`,
`0 <= capture_zero_fraction < 1-capture_loss_max <= 1`,
`0 <= route_zero_gap <= 1`, and `0 < anchor_sigma_min <= 1`.
The residual tolerances `local_action_isometry`, `off_route_leakage`,
`closure_residual`, `route_covariance`, and `projector_residual` are all
nonnegative. Rank and iteration limits are positive integers, and the
materialized `max_rank` must not exceed the validated `local_dim`. Thresholds
must come from the normalized automatic-selection config; v1 does not invent
missing thresholds. Thresholds and their schema are identity-bound.

For a local action of dimension `d`, certify
`||D_local^dagger D_local-I||_F/sqrt(d) <= local_action_isometry`. Its off-route
residual is the Frobenius norm of all rows outside the declared target same-Q
block, divided by `sqrt(d)`, and must not exceed `off_route_leakage`.

For adjacent sorted energies, a gap `<=energy_same_ev` joins one cluster, a gap
`>=energy_different_ev` separates clusters, and an intermediate gap rejects.
For a rank-`r` source cluster and image projector, define
`eta_C = Re Tr(P_C P_image) / r`. Exactly one target cluster must satisfy
`1-eta_C <= capture_loss_max`; every other target cluster must satisfy
`eta_C <= capture_zero_fraction`, and the sum of all other target-cluster
captures must satisfy `sum(other eta_C) <= capture_loss_max`. Target clusters
must cover the complete validated local eigenspace; if a backend represents
that partition indirectly, it must instead certify
`abs(sum_C eta_C - 1) <= capture_loss_max`. An `eta_C` outside `[0,1]` by more
than the applicable numerical residual tolerance rejects; it is never clipped.
Any intermediate capture, multiple full captures, incomplete cluster coverage,
or excessive total off-cluster capture rejects as
`AMBIGUOUS_CLUSTER_CAPTURE`. The final closure residual is
`||(I-P_target)P_image||_F/sqrt(r)`.

**Step 6: Route each production k/q projector by source group**

For an orthonormal joint frame `B`, construct
`C = B.conj().T @ (R0 - R1) @ B`. Its positive and negative spectral
projectors define the two source-group subspaces. Record
`route_gap = min(abs(eig(C)))`; reject a zero/gray-zone gap, but do not require
`abs(eig(C))` to be near one because real inter-group hybridization is allowed.
Require both routed groups to be nonempty for automatic release output, stable
ranks at every production k/q, orthogonality/completeness, and raw-H route
covariance.

For routed rank `r_g`, raw-H route covariance is
`||D Pi_g^(*) D^dagger-Pi_sector_map(g)||_F/sqrt(r_g)` and must not exceed
`route_covariance`. Hermiticity, idempotence, cross-group orthogonality, and
`Pi_0+Pi_1=P` residuals use the same rank-normalized Frobenius convention and
must not exceed `projector_residual`.

Recompute this routing at every production k/q. Align frames only inside each
routed source group using separate Procrustes/polar steps. Never align the
whole joint frame and then split columns, because that can mix the positive and
negative routing subspaces. Freeze joint band indices, but reject any production
k/q with a cluster-boundary split or groupwise anchor rank loss.

Use one fixed canonical `k_ref`, but a separate reference frame for every
`(q,g)`. After sign routing, deterministically obtain and persist
`F_ref[q,g]` using a projector/QRCP/polar gauge. At production `(k,q,g)`, choose
an orthonormal routed frame `X_g`, form
`M=F_ref[q,g].conj().T @ X_g = L diag(sigma) R.conj().T`, require
`min(sigma) >= anchor_sigma_min`, and set
`F(k,q,g)=X_g @ R @ L.conj().T`. Recheck
`F_g F_g^dagger=Pi_g` and cross-group orthogonality. Automatic Gamma requires
constant positive ranks `(r0,r1)` over every production `(k,q)`; a generator
declared to exchange groups additionally requires `r0==r1`. Projector columns
use the fixed order
`c(g,alpha,q)=Nq*sum(r_h for h<g)+alpha*Nq+q`.

**Step 7: Add a typed routed handoff and real assembler**

Represent projection inputs as a typed union with a stable discriminator:

- `ExplicitLegacyBasisSpec`: existing physical-layer `nlow_state_list` plus optional
  anchors, diagnostic/unverified until separately certified;
- `GammaRoutedBasisSpec`: joint band set, canonical layout identity, per-k/q routed
  local frames and group slices, routing/closure certificate hashes.

The two variants are mutually exclusive. Add an assembler that consumes the
routed local frames directly and returns the complete low projector (and high
complement when requested); do not encode routed linear combinations as fake
physical-layer band lists. Persist the same typed handoff in projection
artifacts, bind it into the projection identity, and make `kp symm` consume that
persisted handoff rather than re-deriving the projector from
`nlow_state_list/norb_fix_list`. Keep this routed frame distinct from the later
continuum `SymmetryAdaptedBasisFrame`.

The routed handoff has an explicit new schema and contains at least:

```text
projection_basis_kind = gamma_routed
layout_payload and layout_hash
joint_band_indices
group_ranks and group_offsets
ordered k_indices
frames: complex numeric tensor (Nk,Nq,local_dim,r0+r1), never object/pickle
frame_hash and reference_frame_hash
heff_k_indices
authoritative_heff: complex numeric tensor (Nk,model_dim,model_dim)
heff_hash
routing thresholds and closure/routing certificate hashes
source-H, ordered-Q, raw-action package identities
candidate symmetry certificate_hash and input_identity_hash
```

`basis.npz` stores the discriminator as a scalar and uses no object arrays for
the routed variant. Its loader recomputes all content, layout, and frame hashes
while retaining a versioned legacy loader. `kp symm` may consume only required
k pairs present in the persisted handoff; a missing required k rejects with
`HANDOFF_K_COVERAGE`, with no re-diagonalization or anchor fallback. The low
frames used by symmetry are the persisted frames byte-for-byte. A high
complement may be rebuilt from the same frozen joint band set.

The persisted `authoritative_heff` is the sole projected Hamiltonian accepted
by `kp symm` for this routed handoff. Its ordered k mapping and byte/content hash
are mandatory. `kp symm` loads it directly; it must not recompute Heff from the
source Hamiltonian. If a storage backend requires a separate Heff payload, the
handoff binds that payload by exact content hash and ordered k mapping, and the
loader verifies byte/hash equality before use. A mismatch rejects with
`HANDOFF_IDENTITY`.

Automatic Gamma also has a hard Task 7 dependency. For every required k, wrap
the exact persisted routed `U_low` and the same authoritative persisted Heff as
`CandidateProjectionState` and call
`certify_candidate_symmetries` with the same raw-H actions, required pairs,
exactified actions, and magnetic presentation. A routed automatic candidate is
eligible only when the certificate is `CERTIFIED` and its input identity binds
the persisted frames, authoritative Heff hash/k mapping, and actions. Both
certificate hashes enter the handoff. No unconstrained Heff recomputation may
be used to create or verify the candidate certificate.
Until Task 9 adds the final selection transaction, no incomplete routed
artifact may claim PASS.

Use the following stable `CandidateRejectionReason` enum values exactly; tests,
artifacts, and CLI reporting consume these strings:

```text
UNSUPPORTED_GAMMA_LAYOUT
UNSUPPORTED_SPIN_ROUTE
INVALID_GAMMA_ROW_LAYOUT
GAMMA_Q_ROUTE_MISMATCH
LOCAL_ACTION_ISOMETRY
RAW_ACTION_ROUTE_LEAKAGE
AMBIGUOUS_ENERGY_CLUSTER
AMBIGUOUS_CLUSTER_CAPTURE
SYMMETRY_CLOSURE_FAILURE
AMBIGUOUS_SOURCE_GROUP_ROUTING
EMPTY_SOURCE_GROUP
SOURCE_GROUP_RANK_CHANGE
SOURCE_GROUP_ROUTE_COVARIANCE
PROJECTOR_FRAME_RANK
HANDOFF_K_COVERAGE
HANDOFF_IDENTITY
CANDIDATE_SYMMETRY_FAILED
```

Unsupported group/Q/orbital layouts use `UNSUPPORTED_GAMMA_LAYOUT`; invalid
canonical row identity/bijection uses `INVALID_GAMMA_ROW_LAYOUT`. Closure
iteration or max-rank failure uses `SYMMETRY_CLOSURE_FAILURE`. Do not create
backend-specific aliases for the same failure.

Automatic Gamma must remain disabled until layout, per-k/q routing,
groupwise alignment, assembler, persisted identity, and `kp symm` consumption
are all present and tested.

**Step 8: Run targeted and existing Gamma tests**

Expected: PASS, including existing row-order/Kramers tests.

```bash
PYTHONPATH="$PWD/kp:$PWD/tapw" /data/home/zy/mambaforge/envs/moirekp/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/kp/test_gamma_projector_closure.py \
  tests/kp/test_auto_low_energy_selection.py \
  tests/kp/test_blocks_get_h_block.py \
  tests/kp/test_symm_projection.py \
  tests/kp/test_auto_gauge_layer_layout.py \
  tests/kp/test_kp_artifact_identity.py
```

**Step 9: Commit in two reviewable stages**

```bash
git commit -m "feat(kp): certify routed Gamma projectors"
git commit -m "feat(kp): persist routed Gamma projection handoffs"
```

The first commit may add the strict mathematical core without enabling auto
Gamma. The second commit must close production assembly, identity, and the
`kp symm` handoff before enabling it.

### Task 9: Implement Selection Artifacts And Fail-Closed Decisions

**Worktree:** Integration branch after Tasks 6-8 are cherry-picked.

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `kp/kp/projection_selection.py`
- Modify: `kp/kp/identity.py`
- Create: `kp/kp/selection_artifact.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`
- Create: `tests/kp/test_selection_artifact.py`
- Modify: `tests/kp/test_kp_artifact_identity.py`

**Step 1: Add failing status and transaction tests**

Test:

- automatic selection returns failure when no candidate passes;
- duplicate candidate IDs and any nonfinite metric fail closed;
- explicit selection begins `PENDING` and resolves to `UNVERIFIED_OVERRIDE`
  with null metrics plus a typed reason;
- only complete automatic certification upgrades `PENDING` to `CERTIFIED`;
- a crash before the final atomic marker cannot leave a current certified
  generation;
- artifact hashes change with Q, target window, source action, or layout;
- frame, authoritative-Heff, k-map, metrics, or certificate tampering rejects;
- `inspect` and `project` inputs produce the same
  `selection_identity_hash`; transaction-specific artifact hashes need not be
  equal.

**Step 2: Define status and artifact types**

```python
class CertificationStatus(str, Enum):
    PENDING = "PENDING"
    CERTIFIED = "CERTIFIED"
    FAILED = "FAILED"
    UNVERIFIED_OVERRIDE = "UNVERIFIED_OVERRIDE"


class SelectionFailureCode(str, Enum):
    NO_CANDIDATES = "NO_CANDIDATES"
    DUPLICATE_CANDIDATE_ID = "DUPLICATE_CANDIDATE_ID"
    STRUCTURAL_REJECTION = "STRUCTURAL_REJECTION"
    NONFINITE_METRIC = "NONFINITE_METRIC"
    HARD_METRIC_FAILED = "HARD_METRIC_FAILED"
    CANDIDATE_CERTIFICATE_FAILED = "CANDIDATE_CERTIFICATE_FAILED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    PERSISTENCE_FAILURE = "PERSISTENCE_FAILURE"
    TRANSACTION_SUPERSEDED = "TRANSACTION_SUPERSEDED"
    EXPLICIT_UNVERIFIED = "EXPLICIT_UNVERIFIED"


@dataclass(frozen=True)
class ResolvedCandidateIdentity:
    selection_mode: str
    candidate_id: str
    candidate_dimension: int
    projection_basis_kind: str
    basis_handoff_hash: str
    authoritative_heff_hash: str
    heff_k_indices_hash: str
    resolved_candidate_hash: str


@dataclass(frozen=True)
class SelectionMetricEvidence:
    candidate_id: str
    frozen_target_window_hash: str
    validation_k_indices_hash: str
    basis_handoff_hash: str
    metrics: CandidateMetrics
    metrics_hash: str


@dataclass(frozen=True)
class CertificationEvidence:
    metric_evidence: SelectionMetricEvidence
    symmetry_certificate_hash: str
    symmetry_input_identity_hash: str
    evidence_hash: str


@dataclass(frozen=True)
class SelectionIdentity:
    selection_input_identity_hash: str
    selection_policy_hash: str
    resolved_candidate: ResolvedCandidateIdentity
    certification_evidence: CertificationEvidence | None
    selection_identity_hash: str


@dataclass(frozen=True)
class SelectionArtifact:
    schema_version: str
    transaction_id: str
    selection_input_identity_hash: str
    identity: SelectionIdentity | None
    metrics: CandidateMetrics | None
    certification_status: CertificationStatus
    status_reason: SelectionFailureCode | None
    failure_codes: tuple[SelectionFailureCode, ...]
    diagnostic: str | None
    artifact_hash: str
```

`FrozenTargetWindow`, not only the unresolved `TargetWindowSpec`, supplies the
target hash: ordered band IDs, energies, validation k mapping, edge, reference,
and degeneracy threshold are all bound. The action package hash includes raw
and exactified actions, antiunitary flags, required k pairs, presentation, and
their source identity. The Gamma `basis_handoff_hash` covers its full Task 8
payload, including routed frames, joint bands, layout, group ranks/offsets,
thresholds, authoritative Heff/k mapping, and closure/candidate certificates.

`selection_policy_hash` binds all hard thresholds, candidate-envelope config,
candidate generator/schema version, metric schema, and the versioned
lexicographic ordering rule. `selection_input_identity_hash` is computed from
selection mode, complete frozen target, ordered Q, source-H, raw/exact action
package, row layout, and `selection_policy_hash`, before a candidate is chosen.

`selection_identity_hash` is deterministic and command/output independent.
`artifact_hash` additionally binds schema, transaction, status, status reason,
failure codes, diagnostic text, and the optional final identity. Each hash is
computed from the canonical payload with its own hash field omitted; hashes are
never self-referential. A FAILED/no-candidate artifact has `identity=None`; it
must not invent a candidate ID. An `UNVERIFIED_OVERRIDE` keeps a complete
`ResolvedCandidateIdentity` (including its actual basis/Heff/k map) but has
`certification_evidence=None`. `PASS` is only a display word for `CERTIFIED`,
never a separate persisted status.

`SelectionMetricEvidence.metrics_hash` covers candidate ID, complete frozen
target hash, validation-k hash, basis/handoff hash, and every typed finite
metric; compute it with its own hash field omitted. `CertificationEvidence`
then binds that metric evidence to both candidate-certificate hashes. This
replaces reliance on the older freely constructible scalar-only
`CandidateMetrics` as an identity boundary.

The artifact's top-level `metrics`, when present, is a serialization view of
`CertificationEvidence.metric_evidence.metrics`, not a second source of truth;
strict loading requires exact equality. `CERTIFIED` requires non-null identity,
evidence, and metrics with empty failure codes. Every other status remains
release-blocking.

**Step 3: Implement fail-closed selection**

Only candidates passing every finite hard metric enter lexicographic selection.
Use dimension first, then band error, overlap deficit, and symmetry residual.
Candidate IDs are unique and dimensions positive. Do not choose a WARN fallback
for automatic release output. Automatic state transitions are only
`PENDING -> CERTIFIED|FAILED`. Explicit state transitions are
`PENDING -> UNVERIFIED_OVERRIDE`; explicit output can become `CERTIFIED` only by
running the same complete automatic hard gates and writing a new certified
transaction. Every state other than `CERTIFIED` is release-blocking.

To create `CERTIFIED`, require the candidate certificate itself to be
`CERTIFIED`; its candidate ID and input identity must bind the exact handoff
frames, same authoritative Heff/k mapping, and action package. Metrics must bind
the same frozen target, validation k mapping, and candidate. Recompute and
cross-check every hash after persistence.

**Step 4: Implement strict report serialization and atomic replace**

Serialize missing values as JSON null plus a `SelectionFailureCode` and separate
human diagnostic. Use a generation transaction while holding an exclusive
output-directory transaction lock from before the first marker write through
the final marker write:

1. atomically publish a new transaction's `PENDING` marker first, invalidating
   any older certified generation;
2. write projection, handoff, authoritative Heff, and certificate payloads to
   temporary files, flush/fsync, then replace their generation targets;
3. reload them with strict loaders and recompute all hashes;
4. atomically publish `CERTIFIED` last and fsync the parent directory;
5. on error, leave at most `PENDING`, or atomically replace it with `FAILED`.

Consumers accept payloads only when their generation/transaction and hashes
match the current `CERTIFIED` marker. A single atomic JSON write without this
commit-last protocol is insufficient because projection output spans multiple
files. Before final publication, verify that the current marker still contains
the lock holder's transaction ID. The lock is released by process exit, so a
crash cannot permanently block later work. Add an A/B interleaving test proving
that an older transaction cannot overwrite a newer current generation.

**Step 5: Run tests and commit**

Split this into three reviewable commits:

```bash
git commit -m "fix(kp): make projection selection fail closed"
git commit -m "feat(kp): add immutable selection artifacts"
git commit -m "feat(kp): bind certified selections to projection handoffs"
```

The third commit begins only after both Task 8 commits are integrated. Keep CLI
orchestration out of these pure selector/artifact commits.

### Task 10: Integrate Selection Into `kp inspect` And `kp project`

**Worktree:** `.worktrees/cpc-full-integration`

**Files:**
- Create: `kp/kp/selection_orchestration.py`
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/config/case.py` only if typed selection config is required
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_project_auto_gauge_cli.py`
- Modify: `tests/kp/test_canonical_output_layout.py`
- Modify: `tests/kp/test_kp_artifact_identity.py`
- Modify: `tests/kp/test_symm_projection.py` for integration assertions only
- Modify: `tests/kp/test_configured_model.py` for the pre-output model gate

**Step 1: Add failing CLI orchestration tests**

Mock only external I/O, not the projector/symmetry evaluator. Assert:

- `inspect` and `project` call the same selector;
- selected `selection_identity_hash` is command-independent and preserved;
- missing symmetry input fails before model output;
- explicit unverified projection is allowed but release status is BLOCKED;
- `--active-indices` becomes an explicit `UNVERIFIED_OVERRIDE`, never a hidden
  automatic selection;
- no stale certified generation survives a later failure;
- `kp symm` never reconstructs a routed projector/Heff when the certified
  handoff is missing or invalid;
- `kp model` rejects every non-`CERTIFIED` generation before creating output.

**Step 2: Verify failures**

Run the targeted files below.

**Step 3: Replace dirty-style CLI physics with orchestration**

Add a CLI-independent shared entry such as
`resolve_case_selection(canonical_inputs) -> ResolvedProjectionSelection`.
Load canonical inputs, call it from both commands, freeze the identity, run
production projection with the same resolved candidate, validate identity, and
write the final report. Command name, output directory, and plot choices do not
enter `selection_identity_hash`. `plot.nlow_state_list` must not remain a second
automatic physics input. Do not port the preserved 1700-line CLI block.

For `projection_basis_kind=gamma_routed`, `kp symm` uses an exclusive strict
path: load the routed frame tensor with `allow_pickle=False`, verify ordered k
coverage/layout/group/frame hashes, assemble `U_low` from those exact persisted
frames, and load the authoritative persisted Heff with its exact k mapping and
hash. It must not call legacy anchor/state resolution or re-downfold on failure.
Legacy explicit inputs remain diagnostic by default. They can create a
certified symmetry package only after completing the exact same hard metrics,
candidate certificate, identity checks, and certified transaction; merely
supplying explicit indices never certifies them.

Before creating or cleaning a model output directory, `kp model` requires the
projection marker to be `CERTIFIED` and verifies the same
`selection_identity_hash` across projection, symmetry package, and every
operation. It also cross-checks basis/layout/frame, authoritative Heff/k map,
and candidate certificate identities. `PENDING`, `FAILED`, and
`UNVERIFIED_OVERRIDE` all reject; there is no release `--allow-unverified`
bypass.

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
  tests/kp/test_kp_artifact_identity.py \
  tests/kp/test_symm_projection.py \
  tests/kp/test_configured_model.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/selection_orchestration.py kp/kp/cli.py kp/kp/config/case.py \
  tests/kp/test_project_auto_gauge_cli.py \
  tests/kp/test_canonical_output_layout.py \
  tests/kp/test_kp_artifact_identity.py tests/kp/test_symm_projection.py
git commit -m "feat(kp): share selection across inspect and project"
git add kp/kp/model/pipeline.py tests/kp/test_configured_model.py \
  tests/kp/test_kp_artifact_identity.py
git commit -m "fix(kp): gate model publication on certified selection"
```

The routed `kp symm` consumer is implemented and reviewed in Task 8's second
commit, not duplicated here. Task 10 reruns its integration tests against the
shared selection transaction.

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
