# Autoproject Full-Pipeline Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Organize ten material-local autoproject configurations and validate every case through strict `kp project -> kp model` execution on bigmem001.

**Architecture:** Each case is a complete public YAML below `examples/autoproject/<material>/configs/`, with no hidden profile or generated override. The runner executes project and model sequentially per case, validates both artifact sets, and never uses an automatic production response-compiler fallback. Cases run sequentially on bigmem001 and keep generated data below ignored material-local `runs/` directories.

**Tech Stack:** Python 3.11, PyYAML, NumPy/SciPy, pytest, MoireKP `kp` CLI, SSH to bigmem001.

---

### Task 1: Make the canonical response compiler strict

**Files:**
- Modify: `tests/kp/test_complete_response_basis.py`
- Modify: `kp/kp/model/response_basis.py:6328-6480`

**Step 1: Write strict production tests**

Change the production p=0 test so an injected `GeneratorSpanClosureError` must propagate instead of calling `_compile_candidate_group_with_adjoint_fallback`. Add a finite-p production test that injects `RuntimeError("synthetic symbolic bug")` and asserts propagation. Add a factorized-action test that injects `FactorizedTermActionError` and asserts termination.

**Step 2: Run RED**

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py \
  -k 'production and (fallback or symbolic or factorized)'
```

Expected: FAIL because current production catches the exceptions and enters the CSR oracle.

**Step 3: Remove automatic production fallbacks**

In `_compile_model_response_basis_uncached`, let factorized group action errors, p=0 fixed-space closure errors, and symbolic-atom compiler errors propagate with stage and operation context. Keep `_compile_candidate_group_with_adjoint_fallback` only as a direct test/development correctness oracle.

**Step 4: Run GREEN**

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_symmetry_first.py
```

Expected: PASS and no production test expects fallback.

**Step 5: Defer the code commit**

Record the verified files for the later clean dependency-closure commit. The current `response_basis.py` is itself untracked and cannot form a safe isolated commit before its prerequisite stack is assembled.

### Task 2: Specify the material-local configuration contract in tests

**Files:**
- Modify: `tests/kp/test_autoproject_examples.py`
- Modify: `examples/autoproject/expected_selections.yaml`

**Step 1: Update expected paths**

Require all ten material-local config paths and require the legacy shared `examples/autoproject/configs/` directory to be absent.

**Step 2: Assert explicit model parameters**

Every case must declare `target_bands`, `harmonics`, `max_order`, `fit.method`, `fit.kpoints`, `fit.bands`, `one_sided_weight`, and `two_sided_weight`. Nonlinear PtSe2 also declares `band_kpoints`, `band_loss_weight`, and its current settings unchanged.

Require zero one-/two-sided weights for all linear cases except ZrS2, whose two-sided weight is 300. Require PtSe2 to retain its reviewed nonlinear 1/1/1 weights.

**Step 3: Run RED**

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_autoproject_examples.py
```

Expected: FAIL because eight configs still live in the shared directory and their model sections are empty.

### Task 3: Create the ten complete YAML files

**Files:**
- Create: `examples/autoproject/mote2_3.89/configs/k1_spinless_q06.yaml`
- Create: `examples/autoproject/mote2_3.89/configs/k1_spinful_q06.yaml`
- Modify: `examples/autoproject/mgi2_3.89/configs/gamma_spinful_q04.yaml`
- Create: `examples/autoproject/mgi2_3.89/configs/m1_spinless_q07.yaml`
- Create: `examples/autoproject/mgi2_3.89/configs/m1_spinful_q07.yaml`
- Create: `examples/autoproject/zrs2_3.15/configs/gamma_spinful_q04.yaml`
- Create: `examples/autoproject/mote2_aab_5.09/configs/gamma_spinful_q04.yaml`
- Create: `examples/autoproject/mote2_aab_5.09/configs/k1_a_q04.yaml`
- Create: `examples/autoproject/mote2_aab_5.09/configs/k1_b_q04.yaml`
- Preserve: `examples/autoproject/ptse2_7.34/configs/gamma_spinful_q04.yaml`
- Delete: `examples/autoproject/configs/*.yaml`

**Step 1: Preserve projection physics**

Move system/project settings without changing valley, spin, Q shell, Fermi energy, downfold method, reference energy, selection thresholds, or expected low-energy states. Update only relative paths required by the deeper directory.

**Step 2: Copy validated model supports**

Use existing material release configs for harmonics, polynomial orders, fit rows, comparison slices, and plots. Convert the public fit section to `fit.method` without legacy objective/refinement/profile keys.

**Step 3: Encode weights**

- MoTe2, MgI2, A-AB: linear, one-sided 0, two-sided 0.
- ZrS2: linear, one-sided 0, two-sided 300.
- PtSe2: preserve the existing nonlinear model and 1/1/1 weights exactly.

**Step 4: Run config tests**

Run `tests/kp/test_autoproject_examples.py`; expected PASS.

### Task 4: Extend the collection runner to project and model

**Files:**
- Modify: `tests/kp/test_autoproject_examples.py`
- Modify: `examples/autoproject/run_all.py`

**Step 1: Add failing runner tests**

Require stage-specific `kp project` and `kp model` commands. Add a fake completed model tree and verify model checks reject missing, empty, stale, fallback-marked, or wrong-fit-method artifacts.

**Step 2: Run RED**

Run `tests/kp/test_autoproject_examples.py`; expect missing runner API failures.

**Step 3: Implement ordered stages**

Add `--stage {all,project,model}` with default `all`, plus `--check-inputs`. For `all`, execute and validate project before starting model. Parse model provenance for fit method, response compiler, weights, and absence of fallback.

**Step 4: Run GREEN**

Run `tests/kp/test_autoproject_examples.py`; expected PASS.

### Task 5: Update user documentation

**Files:**
- Modify: `examples/autoproject/README.md`

Document the material-local tree, ten cases, explicit weighting policy, unchanged PtSe2 exception, strict compiler behavior, output artifacts, and runner stage options. Do not place node names or local validation paths in release-facing documentation.

Run `git diff --check -- examples/autoproject tests/kp/test_autoproject_examples.py`; expected no errors.

### Task 6: Run targeted local verification

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  tests/kp/test_autoproject_examples.py \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_symmetry_first.py \
  tests/kp/test_configured_model.py
```

Expected: PASS. Also run `python -m py_compile` and scoped `git diff --check`.

### Task 7: Run all ten cases sequentially on bigmem001

**Files:**
- Create locally only: `validation_runs/autoproject_full_pipeline_20260720/`

**Step 1: Verify environment and inputs**

```bash
ssh bigmem001 'cd /data/work/zy/software/1.tapw_code/moirekp-release && \
  /data/home/zy/mambaforge/envs/moirekp/bin/kp --help >/dev/null && \
  /data/home/zy/mambaforge/envs/moirekp/bin/python \
  examples/autoproject/run_all.py --check-inputs'
```

Expected: all ten source band and symmetry packages exist.

**Step 2: Execute sequentially**

Run one background shell on bigmem001 that iterates manifest order. Capture separate project and model logs in `validation_runs/autoproject_full_pipeline_20260720/` and stop on the first nonzero status.

**Step 3: Validate collection**

```bash
ssh bigmem001 'cd /data/work/zy/software/1.tapw_code/moirekp-release && \
  /data/home/zy/mambaforge/envs/moirekp/bin/python \
  examples/autoproject/run_all.py --check-only'
```

Expected: all ten project and model stages pass artifact checks.

### Task 8: Summarize scientific validation

For every case report selection, layer/spin/orbital summary, project status, symmetry residual, model method/dimension/terms/fit rows/bands/weights, band RMS/max and overlap where exported, key PDFs, elapsed times, and confirmation that no fallback occurred.

Run the default release suite:

```bash
/data/home/zy/mambaforge/envs/moirekp/bin/python -m pytest -q -p no:cacheprovider \
  -m 'not slow and not external_data'
```

Report exact failures without changing unrelated dirty files.

### Task 9: Assemble a clean commit boundary

Do not stage the entire dirty worktree. Separate symmetry/compiler prerequisites, response-basis implementation/tests, and autoproject configs/runner/tests/docs. Build a temporary clean worktree from the staged index and rerun targeted tests before any code commit.
