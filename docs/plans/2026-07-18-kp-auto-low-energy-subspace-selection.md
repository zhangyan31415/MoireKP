# KP Automatic Low-Energy Subspace Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically select and report the smallest symmetry-closed low-energy projector at k=0 and minimum |Q| while keeping its block-band indices fixed for every Q and k.

**Architecture:** Add a pure selection module that resolves reference blocks, generates energy-ordered seeds, closes them under supplied TAPW source actions, and ranks projected candidates.  Keep projection and CLI integration thin: existing block diagonalization supplies reference spectra/vectors, existing production projection supplies quality metrics, and both `kp inspect` and `kp project` consume the same immutable resolved selection.

**Tech Stack:** Python 3.11, NumPy, SciPy linear algebra, PyYAML, pytest, existing KP block/projection/symmetry code.

---

### Task 1: Resolve k=0 and minimum-|Q| references

**Files:**
- Create: `kp/kp/low_energy_selection.py`
- Create: `tests/kp/test_auto_low_energy_selection.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_reference_k_uses_exact_expansion_origin_not_array_zero(): ...
def test_reference_k_rejects_path_without_expansion_origin(): ...
def test_minimum_q_is_resolved_per_source_group_with_stable_tie_break(): ...
```

The k-point test uses an array whose zero vector is at row 2.  The Q test uses
two unequal Q arrays and requires the first canonical row when two vectors
have equal norm.

**Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_auto_low_energy_selection.py
```

Expected: collection fails because `kp.low_energy_selection` does not exist.

**Step 3: Write the minimal implementation**

Create:

```python
@dataclass(frozen=True)
class ReferencePoint:
    k_index: int
    k_coordinate: tuple[float, ...]
    q_indices: tuple[int, ...]
    q_vectors: tuple[tuple[float, ...], ...]

def resolve_reference_point(
    kpoints: np.ndarray,
    qsets: Sequence[np.ndarray],
    *,
    origin_tolerance: float = 1.0e-10,
) -> ReferencePoint:
    ...
```

Require an actual origin row within tolerance.  Resolve each Q set separately
with stable `np.argmin` behavior.

**Step 4: Run tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit**

```bash
git add kp/kp/low_energy_selection.py tests/kp/test_auto_low_energy_selection.py
git commit -m "feat(kp): resolve low-energy reference blocks"
```

### Task 2: Generate one-dimensional and degenerate energy seeds

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`

**Step 1: Write the failing tests**

Add:

```python
def test_spinless_nondegenerate_reference_state_produces_dimension_one_seed(): ...
def test_near_degenerate_states_form_one_energy_cluster(): ...
def test_valence_and_conduction_edges_sort_from_the_requested_side(): ...
def test_non_gamma_pool_keeps_physical_layer_labels_and_allows_inactive_layers(): ...
def test_gamma_pool_keeps_joint_block_identity_without_fake_layer_ownership(): ...
```

**Step 2: Run the new tests and verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_auto_low_energy_selection.py -k "seed or cluster or pool"
```

Expected: FAIL because the candidate generator is absent.

**Step 3: Implement energy clustering and cumulative seeds**

Add immutable `ReferenceState`, `EnergyCluster`, and `LowEnergySeed` records.
Generate clusters only from the requested side of `efermi`.  Cluster adjacent
energies within `degeneracy_tolerance`; do not impose an even dimension.
Non-Gamma state keys include physical layer.  Gamma joint-block state keys do
not claim unique layer ownership.

**Step 4: Run the focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit**

```bash
git add kp/kp/low_energy_selection.py tests/kp/test_auto_low_energy_selection.py
git commit -m "feat(kp): generate low-energy state seeds"
```

### Task 3: Close reference seeds under source symmetry

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`

**Step 1: Write the failing tests**

Add synthetic orthonormal reference blocks covering:

```python
def test_identity_closed_spinless_seed_remains_dimension_one(): ...
def test_spinful_tr_seed_adds_its_kramers_partner(): ...
def test_layer_exchange_adds_partner_from_target_layer(): ...
def test_ptse2_like_joint_gamma_seed_54_closes_with_55(): ...
def test_impossible_symmetry_image_is_a_structural_failure(): ...
```

**Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_auto_low_energy_selection.py -k "closed or partner or symmetry_image or ptse2"
```

Expected: FAIL because symmetry closure is absent.

**Step 3: Implement projector-based closure**

Represent a reference source action as a route between reference blocks with
a unitary matrix and an `antiunitary` flag.  Apply `D @ U.conj()` for an
antiunitary action and `D @ U` otherwise.  Score target eigenstates by squared
overlap, add the smallest set needed to reduce normalized off-subspace leakage
below tolerance, and iterate to a fixed point.  Record every added partner and
operation in the closure report.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit**

```bash
git add kp/kp/low_energy_selection.py tests/kp/test_auto_low_energy_selection.py
git commit -m "feat(kp): close low-energy seeds under symmetry"
```

### Task 4: Rank projection-quality candidates and implement WARN fallback

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`

**Step 1: Write the failing tests**

Add:

```python
def test_smallest_candidate_passing_every_threshold_is_selected(): ...
def test_same_dimension_uses_band_error_overlap_and_symmetry_tie_breaks(): ...
def test_no_passing_candidate_selects_minimax_normalized_violation_with_warning(): ...
def test_structural_failure_is_never_selected_as_warning_fallback(): ...
```

Metrics are `band_rms_mev`, `band_max_mev`, `subspace_overlap`,
`symmetry_residual`, and `symmetry_leakage`.

**Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_auto_low_energy_selection.py -k "threshold or minimax or structural_failure"
```

Expected: FAIL because ranking is absent.

**Step 3: Implement lexicographic ranking**

Add `SelectionThresholds`, `CandidateMetrics`, and `SelectionDecision`.
Passing candidates sort by dimension, RMS error, overlap deficit, symmetry
residual, and stable candidate id.  Fallback candidates sort by maximum
normalized violation, total normalized violation, RMS error, dimension, and
stable id.  Return `status="WARN"` and explicit violated thresholds for the
fallback.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit**

```bash
git add kp/kp/low_energy_selection.py tests/kp/test_auto_low_energy_selection.py
git commit -m "feat(kp): rank automatic projection candidates"
```

### Task 5: Report fixed bands and reference composition

**Files:**
- Modify: `kp/kp/low_energy_selection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`

**Step 1: Write the failing tests**

Add:

```python
def test_composition_reports_orbital_layer_and_spin_weights(): ...
def test_gamma_delocalized_state_reports_weights_not_unique_layer(): ...
def test_report_states_fixed_indices_apply_to_every_q_and_k(): ...
def test_report_lists_symmetry_partner_relations_and_rejected_candidates(): ...
```

**Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_auto_low_energy_selection.py -k "composition or report"
```

Expected: FAIL because reporting is absent.

**Step 3: Implement JSON and Markdown payload generation**

Accept canonical basis labels for each reference-block row.  Sum
`abs(vector[row])**2` by orbital, physical layer, and spin.  Normalize each
weight family, preserve joint Gamma weights, and write deterministic JSON and
Markdown renderers.  The payload contains no per-Q or per-k selected-band map.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit**

```bash
git add kp/kp/low_energy_selection.py tests/kp/test_auto_low_energy_selection.py
git commit -m "feat(kp): report automatic low-energy selection"
```

### Task 6: Integrate automatic selection with canonical config and block construction

**Files:**
- Modify: `kp/kp/config/case.py`
- Modify: `kp/kp/blocks/blocks.py`
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `tests/kp/test_auto_low_energy_selection.py`
- Modify: `tests/kp/test_canonical_output_layout.py`
- Modify: `tests/kp/test_project_auto_gauge_cli.py`

**Step 1: Write failing integration tests**

Cover:

```python
def test_selection_auto_allows_omitted_nlow_state_list(): ...
def test_explicit_nlow_state_list_bypasses_automatic_selection(): ...
def test_project_uses_one_frozen_selection_for_every_hamiltonian_k_row(): ...
def test_inspect_and_project_resolve_identical_selection(): ...
def test_selection_artifacts_are_written_under_projection_output(): ...
```

Use small synthetic K/M per-layer and Gamma joint-block Hamiltonians.  Assert
that the selected band lists passed to every projection row are identical.

**Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_auto_low_energy_selection.py \
  tests/kp/test_canonical_output_layout.py \
  tests/kp/test_project_auto_gauge_cli.py \
  -k "selection or frozen"
```

Expected: omitted `nlow_state_list` is rejected and no selection artifact is
written.

**Step 3: Integrate the pure selector**

Normalize scalar `project.selection: auto` to a mapping with `mode: auto`.
Keep explicit `nlow_state_list` authoritative.  Build reference blocks with
the existing Gamma/non-Gamma `get_H_block` rules, load source action routes
from the TAPW symmetry manifest, evaluate candidates with the existing
production projection and comparison helpers, freeze the resolved physical-
layer or Gamma joint band lists, and pass those lists unchanged through every
projected k row.  Write `low_energy_selection.json` and
`low_energy_selection.md` in the projection directory.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit**

```bash
git add kp/kp/config/case.py kp/kp/blocks/blocks.py kp/kp/cli.py \
  kp/kp/symmetry/projection.py tests/kp/test_auto_low_energy_selection.py \
  tests/kp/test_canonical_output_layout.py tests/kp/test_project_auto_gauge_cli.py
git commit -m "feat(kp): select projection subspace automatically"
```

### Task 7: Verify curated material contracts

**Files:**
- Modify: `tests/kp/test_auto_low_energy_selection.py`
- Modify: `examples/mote2_3.89/README.md`
- Modify: `examples/mgi2_3.89/README.md`
- Modify: `examples/zrs2_3.15/README.md`
- Modify: `examples/mote2_aab_5.09/README.md`
- Modify: `examples/ptse2_7.34/README.md`
- Modify: `docs/project_auto_gauge.md`

**Step 1: Add external-data regression tests**

Mark the tests `external_data`.  Run automatic selection against the curated
cases and compare with these current contracts:

```text
MoTe2 K spinless       [[22], [22]]
MoTe2 K spinful        [[44, 45], [44, 45]]
MgI2 M spinless        [[11], [11]]
MgI2 M spinful         [[22, 23], [22, 23]]
MgI2 Gamma             [[40, 41], [42, 43]]
ZrS2 Gamma             [[40, 41, 42, 43], [44, 45, 46, 47]]
A-AB MoTe2 K-A         [[22], [22], []]
A-AB MoTe2 K-B         [[], [], [22]]
A-AB MoTe2 Gamma       [[], [136, 137], [134, 135]]
PtSe2 Gamma            [[54], [55]]
```

For each case also require the selected candidate's band error, overlap, and
symmetry metrics to be finite.  A changed selection is not automatically a
failure if it is smaller, symmetry closed, and improves all quality metrics;
such a result requires an explicit fixture update with its report.

**Step 2: Run one small available material case**

Run the corresponding `kp inspect`/selector command in the configured
`moirekp` environment.  Store any one-off generated results only under
`validation_runs/auto_low_energy_selection_<timestamp>/`.

**Step 3: Run all available external-data regressions**

```bash
python -m pytest -q -p no:cacheprovider -m external_data \
  tests/kp/test_auto_low_energy_selection.py
```

Expected: available curated cases PASS; missing external arrays SKIP with an
explicit missing-data reason.

**Step 4: Update ordinary-use documentation**

Document automatic reference resolution, frozen band identity, PASS/WARN
selection, composition reporting, and explicit `nlow_state_list` as an expert
override.  Do not place generated candidate reports in `examples/`.

**Step 5: Commit**

```bash
git add tests/kp/test_auto_low_energy_selection.py \
  examples/mote2_3.89/README.md examples/mgi2_3.89/README.md \
  examples/zrs2_3.15/README.md examples/mote2_aab_5.09/README.md \
  examples/ptse2_7.34/README.md docs/project_auto_gauge.md
git commit -m "test(kp): validate automatic low-energy selection"
```

### Task 8: Release verification

**Files:**
- Verify only; do not add generated outputs.

**Step 1: Run targeted KP tests**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_auto_low_energy_selection.py \
  tests/kp/test_blocks_get_h_block.py \
  tests/kp/test_canonical_output_layout.py \
  tests/kp/test_cli_project_memory.py \
  tests/kp/test_project_auto_gauge_cli.py \
  tests/kp/test_symm_projection.py
```

Expected: PASS.

**Step 2: Run the release suite**

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Expected: PASS.

**Step 3: Inspect the worktree**

Confirm no `validation_runs/`, caches, egg-info, external arrays, or unrelated
dirty files are staged.

**Step 4: Commit only remaining scoped documentation or test adjustments**

Use an explicit path list.  Do not stage the whole worktree.

