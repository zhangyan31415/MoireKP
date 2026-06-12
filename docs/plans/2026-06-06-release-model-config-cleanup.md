# Release Model Config Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove release-facing release-only compatibility scaffolding from the model configuration layer and unify time-reversal operation naming.

**Architecture:** `configured.py` should load one explicit release model schema instead of compiling short-form examples through case-specific branches. Symmetry actions remain explicit metadata (`k_map`, `q_map`, `sector_map`), while the time-reversal family uses `TR` consistently (`TR` for physical time reversal, `TR_eff` for effective M-valley time reversal).

**Tech Stack:** Python, YAML, NumPy, pytest, existing `kp.cli model` regression examples.

---

### Task 1: Normalize Time-Reversal Names

**Files:**
- Modify: `kp/kp/model/config_schema.py`
- Modify: `kp/kp/model/configured.py`
- Modify: `kp/kp/model/symmetry.py`
- Modify: `kp/kp/model/exactify_representation.py`
- Modify: `kp/kp/model/core.py`
- Modify: `kp/kp/symmetry/generator.py`
- Modify: `kp/tests/*.py`
- Modify: `kp/examples/*/3.89/kp/configs/model/**/*.yaml`

**Steps:**
1. Replace canonical physical `TR` with `TR`.
2. Keep M-valley effective operation as `TR_eff`.
3. Remove alternate time-reversal names and fallback branches.
4. Update exactification group relation keys from `TR^2` to `TR^2`.
5. Update tests to assert there is only `TR` in release model paths.

### Task 2: Remove Public Config Compiler

**Files:**
- Modify: `kp/kp/model/configured.py`
- Modify: example YAML configs.

**Steps:**
1. Delete `SHORTFORM_MODEL_SCHEMA`, `_is_shortform_model_config`, and `_shortform_*_internal_config`.
2. Convert former short-form YAML examples into full explicit release YAMLs.
3. Keep only generic `load_model_config` path.
4. Remove generated template names from compatibility presets.

### Task 3: Remove Production/Legacy Release Semantics

**Files:**
- Modify: `kp/kp/model/config_schema.py`
- Modify: `kp/kp/model/configured.py`
- Modify: `kp/kp/model/symmetry.py`
- Modify: `kp/kp/model/exactify_representation.py`
- Modify: tests/configs.

**Steps:**
1. Remove release-only compatibility flags from release code/configs.
2. Use explicit matrix semantics instead: `matrix_kind` plus presence of `exactification`.
3. Update run summary to report source integrity without release labels.

### Task 4: Verification

**Commands:**
- `PYTHONPATH=kp python -m pytest -q kp/tests/test_operation_standardization.py kp/tests/test_configured_model.py kp/tests/test_symm_projection.py kp/tests/test_exactify_representation.py`
- `PYTHONPATH=kp python -m kp.cli model -c <K1/Gamma/M1 configs>`
- `rg -n "short-form|release_use|release_level|compatibility|effective_step|alternate-time-reversal-names" kp/kp/model kp/examples/*/3.89/kp/configs/model kp/tests/test_operation_standardization.py kp/tests/test_configured_model.py kp/tests/test_exactify_representation.py kp/tests/test_symm_projection.py`

Expected: pytest passes, K1/Gamma/M1 numerical regressions stay at the established aligned RMS/Max values, and release paths do not contain the removed terms.
