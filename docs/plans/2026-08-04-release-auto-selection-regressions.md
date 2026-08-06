# Release automatic-selection regressions implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore fast, physically valid automatic projection for the release Gamma and partial-sector K examples without weakening full-sector structural certification.

**Architecture:** Gamma chooses a low-energy fibre at one representative `(k, Q)` and uses shared 3/3 meV adequacy thresholds before materializing the full Q model once. Non-Gamma partial-sector candidates retain structural, anchor, identity, and symmetry gates, while full-system band RMS/max are recorded only as diagnostics because the target contains branches outside the modeled sector.

**Tech stack:** Python, NumPy/SciPy, pytest, MoireKP CLI, remote big-memory validation on `bigmem001` and `bigmem003`.

---

### Task 1: Align Gamma representative-fibre thresholds

**Files:**
- Modify: `tests/kp/test_gamma_bare_auto_selection.py`
- Modify: `kp/kp/gamma_auto_producer.py`

1. Change the focused test to require 3 meV RMS and 3 meV maximum error, which jointly preserve the audited MgI2 and AAB Gamma fibres.
2. Run the focused test and confirm it fails against the current 1/3 meV constants.
3. Change only the Gamma defaults to 3/3 meV.
4. Run the focused Gamma producer and bare-auto tests and confirm they pass.

### Task 2: Support diagnostic-only band metrics

**Files:**
- Modify: `tests/kp/test_auto_low_energy_selection.py`
- Modify: `kp/kp/low_energy_selection.py`

1. Add tests showing that `band_rms_mev` and `band_max_mev` can remain in candidate evidence without rejecting a structurally valid candidate.
2. Add a narrowly validated `diagnostic_metric_names` selector argument; only the two band-error metrics are eligible.
3. Preserve raw metrics and report the diagnostic policy in `SelectionDecision` and the generated report.
4. Run the low-energy-selection tests.

### Task 3: Carry the policy through selection orchestration

**Files:**
- Modify: `tests/kp/test_selection_orchestration.py`
- Modify: `kp/kp/selection_orchestration.py`

1. Add a failing orchestration test for diagnostic band metrics.
2. Add the policy to `CaseSelectionInputs` factory methods and pass it to the selector.
3. Run the orchestration tests.

### Task 4: Detect partial-sector non-Gamma envelopes

**Files:**
- Modify: `tests/kp/test_project_auto_gauge_cli.py`
- Modify: `kp/kp/cli.py`

1. Add a K-B-like test in which every candidate leaves one physical layer inactive: large full-system band errors must not reject the smallest otherwise valid candidate.
2. Keep a full-sector regression proving that the same band errors remain hard failures.
3. Detect layer positions inactive across the candidate envelope and apply diagnostic-only band metrics only in that case.
4. Include the stable partial-sector policy marker in selection identity/provenance.
5. Run the focused CLI tests.

### Task 5: Re-run affected release materials

**Cases and hosts:**
- `bigmem001`: AAB K-B; AAB Gamma.
- `bigmem003`: MgI2 Gamma; ZrS2 Gamma.

For each case:
1. Run `kp project`, `kp symm`, and `kp model` once, using the release YAML.
2. Record selected low-energy states/model dimension, automatic harmonic orders, project/symm/model wall times, RMS/max band error, and output directory.
3. Confirm MgI2 Gamma selects `[40, 41, 42, 43]` (76-dimensional Q model), not the erroneous 228-dimensional fibre.
4. Confirm AAB K-B selects the audited B-sector branch without relaxing structural/symmetry checks.

### Task 6: Complete release verification and figures

**Files:**
- Verify: all ten release YAMLs and generated output trees.
- Output: `validation_runs/release_examples_auto_20260804/`

1. Run focused tests for all modified selector paths.
2. Run the relevant release-contract/config tests.
3. Render every model-vs-Heff band PDF to PNG and inspect the Gamma-M-K-Gamma path, energy bounds, clipping, and visible fit quality.
4. Produce one table covering all ten materials/cases, including configuration, selected basis, harmonic/kinetic orders, timings, numerical error, and exact output directory.
5. Report any remaining data-packaging issue separately from code/physics failures; do not overwrite legacy PtSe2 inputs silently.
