# Configurable Continuum Model Implementation Plan

> **Internal implementation note:** use the executing-plans workflow to implement this plan task-by-task.

**Goal:** Turn the notebook-only term generation, coefficient fitting, and band comparison flow into a reusable YAML-driven pipeline.

**Architecture:** Add a thin configuration/pipeline layer over the existing `kp.model.core` numerical implementation. Keep the new code in separate files so notebook logic can be retired without rewriting the working numerical core.

**Tech Stack:** Python, NumPy, SciPy, PyYAML, pytest.

---

### Task 1: Configuration Loader

**Files:**
- Create: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_configured_model.py`

**Steps:**
1. Write failing tests for resolving a model YAML relative to its own path.
2. Verify the tests fail because the module does not exist.
3. Implement dataclasses and a loader that reads source KP config paths, Q sets, k-path arrays, heff arrays, harmonics, symmetry maps, and fit indices.
4. Re-run the focused tests.

### Task 2: Pipeline API

**Files:**
- Modify: `kp/kp/model/pipeline.py`
- Test: `tests/kp/test_configured_model.py`

**Steps:**
1. Write failing tests for producing a `MoireConfig`, block-diagonal fitting heff, and band comparison metrics.
2. Implement `build_moire_config_from_file`, `run_configured_model`, and `compare_bands`.
3. Re-run the focused tests.

### Task 3: MgI2 Gamma Example

**Files:**
- Create: `examples/mgi2_3.89/kp/configs/model/mgi2_3.89_Gamma.yaml`

**Steps:**
1. Encode the current notebook Gamma settings: formal project output, 2+2 orbitals, Gamma harmonics, `C3z/TR/C2` symmetry map, fit indices `[0, 40]`.
2. Run the configured pipeline against the formal `(61,124,124)` heff.
3. Save model eigenvalues and comparison metrics under `outputs/model/mgi2_3.89_Gamma_formal`.

### Task 4: Verification And Commit

**Files:**
- All new files above.

**Steps:**
1. Run focused pytest.
2. Run `python -m compileall` on new code.
3. Run the MgI2 Gamma model comparison.
4. Commit and push the new files.
