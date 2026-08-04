# Release Examples And Automatic Harmonics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Curate the ten release KP examples and make omitted harmonic counts resolve from complete, deduplicated physical Q-set support before a single final model fit.

**Architecture:** Keep explicit polynomial orders and fit methods in each case. When `model.harmonics` is omitted, derive finite candidate masks from the actual Q rows and active sectors, evaluate those masks with the existing Heff ablation metrics, and materialize only the selected continuum model. Preserve explicit harmonic configuration as an expert override.

**Tech Stack:** Python, NumPy, PyYAML, pytest, MoireKP CLI, Slurm/SSH big-memory validation nodes.

---

### Task 1: Specify Dynamic Physical Harmonic Candidates

**Files:**
- Modify: `tests/kp/test_configured_model.py`
- Modify: `kp/kp/model/pipeline.py`

**Step 1: Write failing unit tests**

Add tests requiring the dynamic candidate helper to:

- derive maxima from shell maps rather than `max_shell`;
- include asymmetric terminal supports such as `(6, 2)` and `(6, 5)` when present;
- collapse count pairs that create identical masks;
- collapse an inactive inter or intra family to one support choice;
- return candidates ordered by support complexity and count pair.

**Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/kp/test_configured_model.py -k "physical_harmonic_candidates" -q
```

Expected: FAIL because the dynamic helper is not implemented.

**Step 3: Implement the minimal helper**

In `kp/kp/model/pipeline.py`, add a helper that uses actual model rows and
`_harmonic_ablation_shell_maps`, enumerates the finite intra/inter count range,
builds masks with production count semantics, hashes packed masks, removes
duplicates, and returns the lowest-complexity representative for every unique
mask.

**Step 4: Run focused tests**

Run the command from Step 2. Expected: PASS.

### Task 2: Make Omitted Harmonics Use Dynamic Candidates

**Files:**
- Modify: `tests/kp/test_configured_model.py`
- Modify: `kp/kp/model/pipeline.py`

**Step 1: Write failing integration tests**

Require `load_model_config` with omitted `model.harmonics` and no harmonic
search parameters to use dynamic physical candidates. Require explicit
`model.harmonics` to bypass selection unchanged. Require metadata to report
candidate generation, input and unique counts, selected counts, and elapsed
time.

**Step 2: Verify failure**

```bash
python -m pytest tests/kp/test_configured_model.py -k "automatic_harmonic_selection or omitted_harmonics" -q
```

**Step 3: Connect the dynamic candidate set**

Replace the implicit default `max_shell=5`/ladder path for omitted harmonics
with dynamic physical masks. Keep explicitly configured legacy search options
working for compatibility, but do not emit them in release configs. Reuse
precomputed shell maps and selected masks; do not compile or fit a model for
each candidate.

**Step 4: Verify focused tests**

Run the command from Step 2. Expected: PASS.

### Task 3: Curate The Release Example Inventory

**Files:**
- Delete: tracked BiTeI example files
- Delete: `examples/mote2_3.89/kp/configs/mote2_3.89_K1_q06.yaml`
- Delete: `examples/mote2_3.89/kp/configs/mote2_3.89_K1_up_q06.yaml`
- Delete: `examples/mgi2_3.89/kp/configs/mgi2_3.89_Gamma_q05.yaml`
- Delete: `examples/mgi2_3.89/kp/configs/mgi2_3.89_M1_q07.yaml`
- Move: `examples/kp_canonical_smoke` tracked fixture files to `tests/fixtures/kp_canonical_smoke`
- Delete: material-specific tracked files under `examples/tapw/mote2_9.43` and `examples/tapw/mgi2_9.43`
- Preserve: ignored local outputs, validation runs, and `examples/toZY`

**Step 1: Add a failing exact-inventory contract test**

Require exactly ten tracked KP material configs and seven tracked material
TAPW configs, with the approved paths.

**Step 2: Remove or move only tracked release files**

Use patch-based deletions/moves. Do not delete ignored scientific data.

**Step 3: Run inventory tests**

```bash
python -m pytest tests/kp/test_example_dependency_contract.py tests/test_release_contract.py -q
```

### Task 4: Convert All Ten KP Cases To The Public Contract

**Files:**
- Modify: the ten retained `examples/**/kp/configs/*.yaml` files
- Modify: `tests/kp/test_example_dependency_contract.py`

**Step 1: Write failing config-contract tests**

For every released KP config require:

- `project.selection: auto`;
- no active `project.nlow_state_list`;
- a parseable commented audited `nlow_state_list`;
- omitted `model.harmonics`;
- explicit `model.max_order`;
- nonlinear fit only for PtSe2;
- linear fit for the other nine cases;
- A-AB MoTe2 Gamma order 8/6/8.

**Step 2: Update the ten YAML files**

Preserve each audited low-state list as comments directly below automatic
selection. Remove explicit harmonics. Preserve reviewed orders except the
validated A-AB Gamma 8/6/8 correction.

**Step 3: Run the config contracts**

```bash
python -m pytest tests/kp/test_example_dependency_contract.py tests/kp/test_case_config_system_style.py -q
```

### Task 5: Update Documentation And Data Manifest

**Files:**
- Modify: `examples/README.md`
- Modify: five retained material `README.md` files
- Modify: `examples/data-manifest.yaml`
- Modify: tests that enumerate example commands and paths

**Step 1: Update user documentation**

Document automatic projection, the commented fixed-selection override,
automatic harmonic inference, explicit polynomial orders, fit methods, and
external-data requirements.

**Step 2: Rebuild manifest entries**

List all five material families, ten KP commands, and seven TAPW inputs.
Preserve unresolved license/DOI/data-URL blockers without fabricating values.

**Step 3: Run release contracts**

```bash
python -m pytest tests/test_release_contract.py tests/kp/test_example_dependency_contract.py tests/tapw/test_config_paths.py -q
```

### Task 6: Run Focused Code Verification

**Files:**
- Verify only

**Step 1: Run model/config/selection suites**

```bash
python -m pytest \
  tests/kp/test_configured_model.py \
  tests/kp/test_auto_low_energy_selection.py \
  tests/kp/test_gamma_bare_auto_selection.py \
  tests/kp/test_project_auto_gauge_cli.py \
  tests/kp/test_example_dependency_contract.py \
  tests/test_release_contract.py -q
```

Expected: all pass, with only documented existing warnings.

### Task 7: Run Two Representative Material Cases

**Files:**
- Create under: `validation_runs/release_examples_auto_20260804/`

**Step 1: Freeze provenance**

Record branch, commit, dirty diff hash, source config SHA256, host, Python
environment, and command lines. Copy normalized configs into the validation
directory without changing the tracked examples.

**Step 2: Run MoTe2 K spinful on bigmem001**

Run `kp project`, `kp symm`, and `kp model`; record wall time and return code
for each stage.

**Step 3: Run MgI2 M spinless on bigmem003**

Run the same pipeline and record the same provenance.

**Step 4: Summarize outputs**

Write one JSON/Markdown summary containing selected low bands, commented-list
agreement, physical/unique harmonic candidate counts, selected harmonics,
fit method/order, RMS/maximum band errors, paths, and timings.

### Task 8: Expand Only After Representative Cases Pass

Run the same recorded protocol for the remaining eight release cases. Do not
claim release readiness until the exact current tree has passed all ten cases
and the release metadata blockers have real values.
