# Gamma Representative-Point Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Select a public automatic Gamma branch at one representative fibre, then build the production handoff once.

**Architecture:** Split lightweight reference-fibre candidate selection from production materialization. Public automatic policies use the lightweight selector; legacy explicit candidate policies preserve their current full certification path.

**Tech Stack:** Python, NumPy, pytest, MoireKP Gamma common-anchor projection.

---

### Task 1: Lock the materialization boundary

**Files:**
- Modify: `tests/kp/test_gamma_bare_auto_selection.py`
- Modify: `kp/kp/gamma_auto_producer.py`

1. Add a test with multiple complete reference clusters that spies on `_evaluate_candidate` and requires one call for a public automatic policy.
2. Run that test and confirm it fails because every candidate is currently materialized.
3. Add a reference-fibre selection helper that returns one frozen seed before production materialization.
4. Run the test and confirm it passes.

### Task 2: Preserve local rejection semantics

**Files:**
- Modify: `tests/kp/test_gamma_common_anchor_producer.py`
- Modify: `kp/kp/gamma_auto_producer.py`

1. Add a failing test that a typed reference-anchor/rank rejection advances to the next complete cluster without an extra production materialization.
2. Implement the minimal typed local selection loop.
3. Run the focused Gamma producer tests.

### Task 3: Verify the physical regression

**Files:**
- Verify: `examples/mote2_aab_5.09/kp/configs/mote2_aab_5.09_Gamma_spinful_q04.yaml`

1. Run the representative selection diagnostic and require the four-band AAB branch.
2. Run one AAB Gamma `kp project` on bigmem001.
3. Confirm a single production materialization, dimension 76, acceptable projection residuals, and runtime close to the historical one-minute scale.
