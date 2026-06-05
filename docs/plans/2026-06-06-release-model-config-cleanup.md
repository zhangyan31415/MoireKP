# Release Model Config Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove release-facing `public`/`production_use`/legacy semantics from the model configuration layer and unify time-reversal operation naming.

**Architecture:** `configured.py` should load one explicit release model schema instead of compiling public examples through case-specific branches. Symmetry actions remain explicit metadata (`k_map`, `q_map`, `sector_map`), while the time-reversal family uses `T` consistently (`T` for physical time reversal, `T_eff` for effective M-valley time reversal).

**Tech Stack:** Python, YAML, NumPy, pytest, existing `kp.cli model` regression examples.

---

### Task 1: Normalize Time-Reversal Names

**Files:**
- Modify: `kp/kp/model/config_schema.py`
- Modify: `kp/kp/model/configured.py`
- Modify: `kp/kp/model/symmetry.py`
- Modify: `kp/kp/model/exactify_representation.py`
- Modify: `kp/kp/src/moire_refactored.py`
- Modify: `kp/kp/symmetry/generator.py`
- Modify: `kp/tests/*.py`
- Modify: `kp/examples/*/3.89/kp/configs/model/**/*.yaml`

**Steps:**
1. Replace canonical physical `TR` with `T`.
2. Keep M-valley effective operation as `T_eff`.
3. Remove `TR` aliases and fallback branches.
4. Update exactification group relation keys from `TR^2` to `T^2`.
5. Update tests to assert there is no `TR` in release model paths.

### Task 2: Remove Public Config Compiler

**Files:**
- Modify: `kp/kp/model/configured.py`
- Modify: example YAML configs.

**Steps:**
1. Delete `PUBLIC_MODEL_SCHEMA`, `_is_public_model_config`, and `_public_*_internal_config`.
2. Convert former public YAML examples into full explicit release YAMLs.
3. Keep only generic `load_model_config` path.
4. Remove generated template names containing `public`.

### Task 3: Remove Production/Legacy Release Semantics

**Files:**
- Modify: `kp/kp/model/config_schema.py`
- Modify: `kp/kp/model/configured.py`
- Modify: `kp/kp/model/symmetry.py`
- Modify: `kp/kp/model/exactify_representation.py`
- Modify: tests/configs.

**Steps:**
1. Remove `production_use`, `production_level`, `effective_interim`, and `legacy_compatibility` from release code/configs.
2. Use explicit matrix semantics instead: `matrix_kind` plus presence of `exactification`.
3. Update run summary to report source integrity without release labels.

### Task 4: Verification

**Commands:**
- `PYTHONPATH=kp python -m pytest -q kp/tests/test_operation_standardization.py kp/tests/test_configured_model.py kp/tests/test_symm_projection.py kp/tests/test_exactify_representation.py`
- `PYTHONPATH=kp python -m kp.cli model -c <K1/Gamma/M1 configs>`
- `rg -n "public|production_use|production_level|legacy_compatibility|effective_interim|\\bTR\\b" kp/kp/model kp/examples/*/3.89/kp/configs/model kp/tests/test_operation_standardization.py kp/tests/test_configured_model.py kp/tests/test_exactify_representation.py kp/tests/test_symm_projection.py`

Expected: pytest passes, K1/Gamma/M1 numerical regressions stay at the established aligned RMS/Max values, and release paths do not contain the removed terms.
