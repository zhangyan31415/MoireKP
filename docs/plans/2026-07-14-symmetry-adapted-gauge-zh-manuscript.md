# Symmetry-Adapted Gauge Chinese Manuscript Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a code-faithful description of the symmetry-adapted low-energy gauge to the Chinese MoireKP manuscript without changing its material results or analytic model conventions.

**Architecture:** Add one mathematical subsection to the low-energy projection theory and one implementation paragraph to the output/data-flow chapter. Use the same basis transformation convention as `kp/kp/basis/symmetry_gauge.py`, and keep diagnostic low-energy sewing matrices separate from production exactified continuum actions.

**Tech Stack:** LaTeX, CJK, BibTeX, existing MoireKP notation.

---

### Task 1: Add the symmetry-adapted low-energy gauge definition

**Files:**
- Modify: `paper/moirekp/zh/main.tex` near the end of “从 TAPW-DFT 到低能连续基”

**Step 1:** Introduce the internal unitary freedom (U_L\to U_LW) and explain why a dominant-component phase convention is insufficient for a degenerate multiband subspace.

**Step 2:** State the metadata-driven frame construction: select an actually available sector-preserving finite-order unitary action, order its eigenspaces deterministically by eigenphase, pair conjugate channels with an antiunitary action when available, and use a secondary finite-order unitary only to anchor remaining phases.

**Step 3:** Add the transformations for Hamiltonians, observables, eigenvector coordinates, unitary representations, and antiunitary sewing parts.

**Step 4:** State the conservative identity fallback and clarify that `not_applicable` does not mean the physical symmetry is absent.

### Task 2: Document artifact and pipeline consistency

**Files:**
- Modify: `paper/moirekp/zh/main.tex` in “输出文件与后处理流程”

**Step 1:** Explain that `kp project` applies one versioned frame to Heff, wavefunctions, and spin operators, while `kp symm` transforms the exactified continuum actions with the same frame.

**Step 2:** Explain that the serialized frame artifact and hash participate in projection identity and prevent silent gauge mixing in subsequent model construction.

**Step 3:** Keep sampled low-energy sewing matrices labeled as diagnostics and exactified actions labeled as production inputs.

### Task 3: Compile and inspect the manuscript

**Files:**
- Verify: `paper/moirekp/zh/main.tex`
- Write generated files only under: `validation_runs/zh_symmetry_gauge_manuscript/`

**Step 1:** Run the existing LaTeX/BibTeX build from `paper/moirekp/zh/` with `validation_runs/zh_symmetry_gauge_manuscript/` as the output directory.

**Step 2:** Check the log for undefined control sequences, missing references, and fatal errors.

**Step 3:** Confirm that `paper/moirekp/zh/main.tex` is the only manuscript source modified and that no generated PDF, AUX, BBL, BLG, or LOG file is staged.
