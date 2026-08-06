# CdI2 q04 Parameters on q05 Q Basis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the fitted q04 physical continuum terms on the q05 Q basis and compute the top7–8 outer-hex Wilson loop without using q05 TAPW data to determine parameters.

**Architecture:** A standalone validation module extracts q04 fitted values from `fitted_coefficients` plus `basis_metadata_json`, canonicalizes each physical term by tag, momentum transfer, monomial, layers, orbitals, and component, then assigns those values to the corresponding q05 candidate terms produced by the normal model builder. A custom ordered term collection preserves authored terms that share a legacy dictionary key. The existing continuum term evaluator assembles the 248-dimensional Hamiltonian, after which the validated C3 and reciprocal-boundary sewing routines compute the Wilson phases.

**Tech Stack:** Python, NumPy, SciPy, existing `kp.model` continuum builder/evaluator, pytest, Matplotlib.

---

### Task 1: Extract semantic q04 parameters

**Files:**
- Create: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/semantic_lift.py`
- Test: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/test_semantic_lift.py`

**Step 1: Write the failing test**

Test that `extract_fitted_physical_parameters()` returns 1460 unique component keys, that every stored value equals the corresponding q04 fitted coefficient, and that the key contains rounded `p`, `Mz`, `Mz_star`, layer pair, orbital pair, tag, and component.

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest -q validation_runs/cdi2_ab_q04_params_on_q05_20260806/test_semantic_lift.py
```

Expected: FAIL because `semantic_lift.py` does not exist.

**Step 3: Implement the extractor**

Read only `basis_metadata_json.npy` and `fitted_coefficients.npy` from the q04 NPZ. Canonicalize physical keys as JSON-safe tuples with momentum rounded to `1e-9`. Reject duplicates with conflicting values and missing metadata.

**Step 4: Run the focused test**

Expected: PASS.

**Step 5: Commit**

```bash
git add validation_runs/cdi2_ab_q04_params_on_q05_20260806
git commit -m "test: define semantic q-shell parameter lift"
```

### Task 2: Assign q04 parameters to generated q05 terms

**Files:**
- Modify: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/semantic_lift.py`
- Modify: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/test_semantic_lift.py`
- Create: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/config_q05_lift.yaml`

**Step 1: Write the failing integration test**

Build the q05 model with q04 maximum orders `{kinetic: 10, intralayer: 6, interlayer: 8}`. Assert that every nonzero q04 physical parameter maps to exactly one authored q05 term signature, with no q05 fit performed.

**Step 2: Verify the expected failure**

Expected: FAIL because assignment/build helpers are missing.

**Step 3: Implement minimal assignment**

Use `build_moire_config_from_file()` and `build_model()`. Iterate `candidate_terms` rather than the legacy term dictionary, match by semantic key, assign real/imaginary components, keep only terms receiving a q04 value, and expose them through an ordered unique internal key so duplicate authored terms are not overwritten.

The q05 config points to q05 Q sets and symmetry artifacts but its fit section is never invoked.

**Step 4: Run test and inspect transfer report**

Expected: PASS and a report listing source coefficient count, mapped count, zero/nonzero counts, and unmatched signatures. Any unmatched nonzero q04 signature is a hard failure.

**Step 5: Commit**

```bash
git add validation_runs/cdi2_ab_q04_params_on_q05_20260806
git commit -m "feat: lift q04 continuum terms onto q05 basis"
```

### Task 3: Validate the q04 reconstruction and common Q block

**Files:**
- Modify: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/semantic_lift.py`
- Modify: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/test_semantic_lift.py`

**Step 1: Write failing numerical tests**

At representative fractional k points, rebuild the q04 Hamiltonian through the same semantic term path and compare it with the exported q04 compiled evaluator. Then reorder the q05 basis to identify q04 Q vectors and compare all term-supported entries in the common block.

**Step 2: Verify failures before helpers exist**

Expected: FAIL with missing reconstruction/common-block functions.

**Step 3: Implement Hamiltonian and Q-index mapping helpers**

Use the existing continuum term assembly path and the model reciprocal basis for coordinate conversion. Require Hermiticity residual below tolerance, unique Q matches, and numerical agreement below a recorded threshold.

**Step 4: Run tests**

Expected: PASS with printed maximum absolute and relative residuals.

**Step 5: Commit**

```bash
git add validation_runs/cdi2_ab_q04_params_on_q05_20260806
git commit -m "test: certify q04 to q05 semantic lift"
```

### Task 4: Compute the lifted q05-basis hex Wilson loop

**Files:**
- Create: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/run_hex_wilson.py`
- Create: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/test_hex_wilson.py`

**Step 1: Write failing sewing tests**

Test the eight radii `0.49825..0.5`, 41-point paths, generic closure `U_start† C3† U_end`, and exact-boundary K/K′ reciprocal sewing on the q05 Q layout.

**Step 2: Verify expected failure**

Expected: FAIL because the Wilson driver does not exist.

**Step 3: Implement the driver**

Diagonalize only the highest nine states of the 248-dimensional lifted model. Compute top7–8 `W3` and `Wh`, raw and symmetry-pinned endpoint values, internal-link singular values, C3 sewing singular values, and K/K′ sewing diagnostics.

**Step 4: Run tests and the calculation**

```bash
python -m pytest -q validation_runs/cdi2_ab_q04_params_on_q05_20260806/test_hex_wilson.py
python validation_runs/cdi2_ab_q04_params_on_q05_20260806/run_hex_wilson.py
```

Expected: PASS; generate NPZ, JSON, PNG, and PDF.

**Step 5: Commit**

```bash
git add validation_runs/cdi2_ab_q04_params_on_q05_20260806
git commit -m "feat: calculate q04-parameter q05-basis hex Wilson loop"
```

### Task 5: Final comparison and verification

**Files:**
- Modify: `validation_runs/cdi2_ab_q04_params_on_q05_20260806/run_hex_wilson.py`

**Step 1: Add the comparison plot**

Overlay original q04-basis model, lifted q05-basis model, and TAPW q05. Mark both raw and reciprocal-sewn endpoints.

**Step 2: Run full verification**

```bash
python -m pytest -q validation_runs/cdi2_ab_q04_params_on_q05_20260806
python validation_runs/cdi2_ab_q04_params_on_q05_20260806/run_hex_wilson.py
```

Check all artifacts are nonempty, source hashes are recorded, no q05 TAPW energy/wavefunction path appears in the parameter provenance, and sewing diagnostics exceed the configured thresholds.

**Step 3: Commit**

```bash
git add validation_runs/cdi2_ab_q04_params_on_q05_20260806
git commit -m "test: verify q04 parameters on q05 basis"
```
