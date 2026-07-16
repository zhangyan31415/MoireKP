# KP Inferred Configuration Defaults Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let ordinary KP cases omit model orbital counts, harmonic counts, project gauge, and downfold method while retaining deterministic low-cost selection, expert overrides, and fail-closed validation.

**Architecture:** Normalize projection defaults once in the canonical case configuration, derive model orbital counts from the projection artifact, and run the existing Heff harmonic ablation before model-basis compilation when harmonics are absent. Keep automatic choices in memory and normal model artifacts, and report harmonic selection on the terminal without adding harmonic-specific files.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyYAML, pytest, existing KP projection/model pipeline.

---

### Task 1: Canonical projection defaults

**Files:**
- Modify: `kp/kp/config/case.py:104-161`
- Modify: `kp/kp/cli.py:842-855,1021-1027,1622-1731`
- Modify: `kp/kp/symmetry/projection.py:542-549`
- Test: `tests/kp/test_canonical_output_layout.py`
- Test: `tests/kp/test_project_auto_gauge_cli.py`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing tests**

Add tests showing that a canonical case with a `project` section but without gauge anchors or a downfold method resolves to:

```python
assert cfg["project"]["gauge"] == "auto"
assert cfg["project"]["downfold_method"] == "linearized_lowdin"
```

Also assert that manual `norb_fix_list`, explicit `gauge`, and explicit downfold methods are preserved.

**Step 2: Run tests to verify failure**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_canonical_output_layout.py tests/kp/test_project_auto_gauge_cli.py tests/kp/test_configured_model.py -k 'default or gauge or downfold'
```

Expected: the new assertions fail because gauge is currently required and the helper fallback is `fixed_schur`.

**Step 3: Implement the defaults**

In `normalize_case_config`, when a project section exists:

```python
if "gauge" not in project and "norb_fix_list" not in project:
    project["gauge"] = "auto"
if "downfold_method" not in project and "method" not in project:
    project["downfold_method"] = "linearized_lowdin"
```

Update duplicated helper fallbacks in `cli.py` and `symmetry/projection.py`. Do not infer `e_ref`; retain the existing error when a nontrivial downfold lacks it.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit Task 1 files**

```bash
git add kp/kp/config/case.py kp/kp/cli.py kp/kp/symmetry/projection.py tests/kp/test_canonical_output_layout.py tests/kp/test_project_auto_gauge_cli.py tests/kp/test_configured_model.py
git commit -m "feat(kp): default project gauge and downfolding"
```

### Task 2: Infer model orbital counts from projection

**Files:**
- Modify: `kp/kp/model/pipeline.py:221-305,483-494,1364-1497`
- Test: `tests/kp/test_layerwise_config_contract.py`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing tests**

Create projection fixtures containing `heff.npy` and `basis.npz:nlow_state_list`. Omit `model.n_orb` and test a bilayer `(4,4)` and a 1+2 selection `[1,1,0]`. Assert provenance is `projection_basis`. Add a mismatch test for an explicit count and a legacy test where explicit counts remain accepted when projection metadata is absent.

**Step 2: Run tests to verify failure**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_layerwise_config_contract.py tests/kp/test_configured_model.py -k 'n_orb or orbital_count'
```

Expected: omitted counts use the ambiguous Heff-dimension fallback and provenance/mismatch checks fail.

**Step 3: Implement artifact-backed inference**

Read `heff_file.parent / "basis.npz"` with `allow_pickle=True`, extract `nlow_state_list`, validate physical-layer row shape, convert rows to selected-band counts, and resolve them through `_resolve_layerwise_counts`. Fall back to the normalized source case's `project.nlow_state_list`. When explicit `model.n_orb` exists alongside projection metadata, resolve both forms and require equality. Use explicit counts only as a legacy source when projection metadata is absent; remove the ambiguous array-dimension fallback from the release path.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit Task 2 files**

```bash
git add kp/kp/model/pipeline.py tests/kp/test_layerwise_config_contract.py tests/kp/test_configured_model.py
git commit -m "feat(kp): infer model orbitals from projection"
```

### Task 3: Select accepted low-cost harmonics when omitted

**Files:**
- Modify: `kp/kp/model/pipeline.py:1404-1426,1597-1670,8196-8315,8586-8702`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing tests**

Omit `model.harmonics` and assert that the resolved counts are the smallest candidate satisfying `plot_rms_mev <= 1.0` and `subspace_mean_overlap > 0.9`. Assert this does not require `fit.mode: auto_low_energy` or an explicit harmonic-selection enable flag. Add tests that explicit harmonics bypass the scan and that no accepted candidate raises.

**Step 2: Run tests to verify failure**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py -k 'harmonic and (low_cost or omitted or explicit)'
```

Expected: omitted harmonics currently receive a fixed default only in auto-low-energy mode, and the diagnostic low-cost helper falls back to unaccepted candidates.

**Step 3: Implement strict pre-fit selection**

Track whether harmonics were explicitly supplied. When absent, run `_run_harmonic_ablation_selection` after orbital resolution and choose `_low_cost_harmonic_recommendation_candidate(report)`. Change that helper to return `None` when no candidate meets the low-cost rule. Store the selected counts as canonical `intra`/`inter` count specs and retain the report in `ConfiguredModel.harmonics_diagnostics`. Explicit harmonics remain unchanged and bypass selection.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit Task 3 files**

```bash
git add kp/kp/model/pipeline.py tests/kp/test_configured_model.py
git commit -m "feat(kp): select low-cost harmonics by default"
```

### Task 4: Print inferred choices without harmonic-specific files

**Files:**
- Modify: `kp/kp/model/pipeline.py:8359-8702,11570-11600`
- Modify: `kp/kp/cli.py:2355-2380`
- Test: `tests/kp/test_configured_model.py:2841-2895`
- Test: `tests/kp/test_canonical_output_layout.py`

**Step 1: Write failing output tests**

Capture `kp model` output and require resolved `n_orb`, selected low-cost counts, plot RMS/max, subspace overlap, and the high-accuracy reference. Capture `kp project` output and require the resolved gauge, downfold method, and `e_ref`. Assert an automatic harmonic run does not create `harmonic_recommendation_bands.png` or a new harmonic YAML/JSON. Do not remove unrelated existing auto-low-energy reports.

**Step 2: Run tests to verify failure**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py tests/kp/test_canonical_output_layout.py -k 'harmonic_recommendation or inferred or projection_setup'
```

Expected: current recommendation writes a plot and labels configured harmonics as unchanged.

**Step 3: Implement terminal-only reporting**

Reuse existing formatting helpers, print the low-cost candidate as selected and the high-accuracy candidate as reference, and do not call `_save_harmonic_recommendation_band_plot` for automatic selection. Print orbital-count provenance from `orbital_count_metadata`. Make project defaults visible through the existing setup report without writing new artifacts.

**Step 4: Run focused tests and expect PASS**

Run the command from Step 2.

**Step 5: Commit Task 4 files**

```bash
git add kp/kp/model/pipeline.py kp/kp/cli.py tests/kp/test_configured_model.py tests/kp/test_canonical_output_layout.py
git commit -m "feat(kp): report inferred model settings"
```

### Task 5: Documentation and release verification

**Files:**
- Modify: `kp/README.md`
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `docs/project_auto_gauge.md`
- Modify: `docs/project_auto_low_energy.md`

**Step 1: Update user-facing examples**

Remove redundant fields from minimal ordinary-use snippets. Document expert overrides and the strict low-cost acceptance rule. Keep `project.e_ref` visible.

**Step 2: Run targeted regression tests**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_canonical_output_layout.py tests/kp/test_project_auto_gauge_cli.py tests/kp/test_layerwise_config_contract.py tests/kp/test_configured_model.py tests/kp/test_symm_projection.py
```

Expected: PASS.

**Step 3: Run the release suite**

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Expected: PASS.

**Step 4: Inspect status**

Confirm no validation outputs, caches, private paths, or unrelated dirty files are staged.

**Step 5: Commit documentation only**

```bash
git add kp/README.md README.md README.zh.md docs/project_auto_gauge.md docs/project_auto_low_energy.md
git commit -m "docs: simplify inferred KP configuration"
```
