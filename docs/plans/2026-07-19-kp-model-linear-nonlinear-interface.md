# KP Model Linear/Nonlinear Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a small public `model.fit.method: linear|nonlinear` contract with explicit k rows, band count, one-sided/two-sided Hamiltonian weights, and a nonlinear-only eigenvalue-loss weight.

**Architecture:** Canonicalize the public contract in `kp.model.pipeline` and translate it to the symmetry-complete response-basis fit. The linear solver uses the exact dimension-normalized Hamiltonian, one-sided, and two-sided quadratic loss. The nonlinear solver starts from that exact linear solution and adds an analytic eigenvalue residual while retaining the same quadratic Hamiltonian loss. Existing advanced configurations remain readable, but cannot be mixed with the public method contract.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyYAML, pytest.

**Workspace note:** This implementation depends on the current uncommitted complete-response-basis work. A clean worktree from `HEAD` would omit that dependency, so work remains in the shared tree and every commit must stage only files listed by its task.

---

### Task 1: Parse and validate the public fit contract

**Files:**
- Modify: `kp/kp/model/pipeline.py:162-215`
- Modify: `kp/kp/model/pipeline.py:1400-2160`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing parser tests**

Add tests using the existing configured-model fixture for:

```python
raw["model"]["fit"] = {
    "method": "linear",
    "kpoints": [0, 20, 40],
    "bands": 10,
    "one_sided_weight": 1.0,
    "two_sided_weight": 1.0,
}
```

and:

```python
raw["model"]["fit"] = {
    "method": "nonlinear",
    "kpoints": [0, 20, 40],
    "band_kpoints": "all",
    "bands": 10,
    "one_sided_weight": 1.0,
    "two_sided_weight": 1.0,
    "band_loss_weight": 1.0,
    "max_steps": 30,
}
```

Assert the parsed `ConfiguredModel` records a canonical method contract,
linear fit indices, and nonlinear band indices.

**Step 2: Write failing rejection tests**

Cover missing/invalid `method`, empty or duplicate/out-of-range k rows,
non-positive bands, negative/non-finite weights, nonlinear fields under
`linear`, missing nonlinear-only fields, and mixing with `mode`, `indices`,
`objective`, `refine_bands`, `weighting`, `model_selection`, or `profiles`.

**Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py \
  -k 'public_fit_method'
```

Expected: failures because the public method contract is not parsed.

**Step 4: Implement canonicalization**

Add a private immutable/canonical mapping with:

```python
{
    "method": "linear" | "nonlinear",
    "hamiltonian_kpoints": [...],
    "band_kpoints": None | [...],
    "requested_bands": int,
    "one_sided_weight": float,
    "two_sided_weight": float,
    "band_loss_weight": None | float,
    "max_steps": None | int,
    "normalization": "dimension_mean_square_v1",
}
```

Store it on `ConfiguredModel`. Require explicit harmonics and max orders for
the new contract. Resolve `band_kpoints: all` to every projected Heff row.
Set the existing coefficient-fit indices from `hamiltonian_kpoints`, force
`complete_linear_v2`, and disable automatic profile/model selection.

**Step 5: Run tests and verify GREEN**

Run the command from Step 3 and the existing target-spectral parser tests.

**Step 6: Commit**

```bash
git add kp/kp/model/pipeline.py tests/kp/test_configured_model.py
git commit -m "feat(kp): parse public linear nonlinear fit methods"
```

### Task 2: Compile the normalized linear loss

**Files:**
- Modify: `kp/kp/model/response_basis.py:120-390`
- Modify: `kp/kp/model/response_basis.py:3020-3180`
- Modify: `kp/kp/model/core.py:4230-4280`
- Modify: `kp/kp/model/pipeline.py:400-550`
- Test: `tests/kp/test_complete_response_basis.py`

**Step 1: Write an independent dense oracle test**

For a small Hermitian target and response basis, construct the degeneracy-safe
projector and independently solve rows corresponding to:

```python
delta_h / d
sqrt(w1) * delta_h @ projector / sqrt(d * n)
sqrt(w2) * projector @ delta_h @ projector / n
```

Assert production coefficients and all three residual contributions match the
dense least-squares oracle.

**Step 2: Run the oracle and verify RED**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_complete_response_basis.py \
  -k 'normalized_public_linear_loss'
```

Expected: failure because no normalized public loss exists.

**Step 3: Add the loss specification**

Extend the target-dependent response-basis objective with a versioned
`normalized_low_energy_linear_v1` specification holding the band edge,
requested band count, degeneracy tolerance, and two public weights. Do not
express it through the legacy floor/alpha normalization.

**Step 4: Compile sparse-union rows**

Reuse the target eigenvectors and selected mask. Append the normalized global,
one-sided, and two-sided rows without materializing a global dense response
tensor. Record requested/resolved band counts and normalization denominators in
the fit artifact and hash.

**Step 5: Translate public config to the new objective**

Build the response objective directly from `ConfiguredModel.fit_method_config`.
Keep the existing `target_spectral_linear` path unchanged for legacy configs.

**Step 6: Run tests and verify GREEN**

Run the oracle, target-spectral tests, and configured-model parser tests.

**Step 7: Commit**

```bash
git add kp/kp/model/response_basis.py kp/kp/model/core.py \
  kp/kp/model/pipeline.py tests/kp/test_complete_response_basis.py
git commit -m "feat(kp): add normalized low-energy linear loss"
```

### Task 3: Retain the linear loss during nonlinear refinement

**Files:**
- Modify: `kp/kp/model/pipeline.py:7880-8210`
- Modify: `kp/kp/model/pipeline.py:10618-11930`
- Test: `tests/kp/test_configured_model.py`

**Step 1: Write failing residual/Jacobian tests**

Create a small basis tensor and target eigensystem. Independently calculate:

```python
R_H + w1 * R_1s + w2 * R_2s + wb * R_band
```

Compare the analytic Jacobian with central finite differences. Include both
top and bottom windows and a degenerate boundary.

**Step 2: Write a failing initialization test**

Monkeypatch the nonlinear optimizer and assert its initial coefficient vector
equals the exact coefficients returned by the normalized linear fit.

**Step 3: Run tests and verify RED**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py \
  -k 'public_nonlinear_loss or public_nonlinear_starts_from_linear'
```

**Step 4: Implement the quadratic Hamiltonian residual**

Build the full/one-sided/two-sided quadratic Gram matrix from the retained
response tensor and fixed target projector. Compress it to coefficient space
as an exact square-root residual centered on the linear minimizer. This keeps
the same linear loss in every nonlinear step without a large explicit matrix
Jacobian.

**Step 5: Add the eigenvalue residual and Jacobian**

Use the target-resolved band window on `band_kpoints` and append:

```python
sqrt(band_loss_weight / n) * (model_eigvals - target_eigvals)
```

Use Hellmann-Feynman derivatives for the analytic Jacobian. Set the iteration
cap from `max_steps`; optimizer, damping, tolerances, and compression remain
internal.

**Step 6: Do not silently roll back**

Disable the legacy acceptance rollback for the public method. Preserve the
requested nonlinear coefficients and return linear-initial plus nonlinear-final
metrics for reporting.

**Step 7: Run tests and verify GREEN**

Run the tests from Step 3 plus existing analytic Jacobian, reduced matrix loss,
and Gauss-Newton tests.

**Step 8: Commit**

```bash
git add kp/kp/model/pipeline.py tests/kp/test_configured_model.py
git commit -m "feat(kp): add public nonlinear band loss"
```

### Task 4: CLI reporting and standalone metadata

**Files:**
- Modify: `kp/kp/model/pipeline.py:6400-7350`
- Modify: `kp/kp/model/pipeline.py:11970-12120`
- Modify: `kp/kp/model/export.py`
- Modify: `kp/kp/cli.py`
- Test: `tests/kp/test_reporting.py`
- Test: `tests/kp/test_standalone_export.py`

**Step 1: Write failing reporting tests**

Assert `kp model` prints one method summary containing method, Hamiltonian k
rows, band k rows, requested/resolved bands, three applicable weights, and
linear-initial/final metrics. Assert linear output does not mention a band-loss
weight.

**Step 2: Write failing artifact tests**

Assert the standalone `model_data.npz` and `MODEL.md` record the resolved
public method contract and nonlinear provenance without exporting a second
linear model.

**Step 3: Run tests and verify RED**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_reporting.py tests/kp/test_standalone_export.py \
  -k 'public_fit_method'
```

**Step 4: Implement reports and metadata**

Add a compact `Fit method` section and a warning when selected-band improvement
coincides with material full-band or overlap degradation. Keep diagnostics
machine-readable and never emit `low`/`high` profile directories for this
contract.

**Step 5: Run tests and verify GREEN**

Run the tests from Step 3 and existing CLI plot/report tests.

**Step 6: Commit**

```bash
git add kp/kp/model/pipeline.py kp/kp/model/export.py kp/kp/cli.py \
  tests/kp/test_reporting.py tests/kp/test_standalone_export.py
git commit -m "feat(kp): report public model fit methods"
```

### Task 5: Public documentation and example migration

**Files:**
- Modify: `kp/README.md`
- Modify: `docs/project_auto_low_energy.md`
- Modify: `examples/autoproject/ptse2_7.34/configs/gamma_spinful_q04.yaml`
- Test: `tests/kp/test_autoproject_examples.py`

**Step 1: Write a failing example-contract test**

Assert the PtSe2 model block uses only the approved public keys and no
`objective`, `refine_bands`, solver, optimizer, Jacobian, compression, variable
tags, components, sigma, or model-selection profile keys.

**Step 2: Run it and verify RED**

```bash
python -m pytest -q -p no:cacheprovider tests/kp/test_autoproject_examples.py \
  -k 'ptse2_public_fit_method'
```

**Step 3: Update documentation and the example**

Document both complete YAML blocks and the four residual formulas. Migrate
PtSe2 to one nonlinear model with explicit basis orders, Hamiltonian k rows,
band k rows, band count, weights, and step cap.

**Step 4: Run tests and verify GREEN**

Run the example-contract and parser tests.

**Step 5: Commit**

```bash
git add kp/README.md docs/project_auto_low_energy.md \
  examples/autoproject/ptse2_7.34/configs/gamma_spinful_q04.yaml \
  tests/kp/test_autoproject_examples.py
git commit -m "docs(kp): publish linear nonlinear model fit interface"
```

### Task 6: Verification and external PtSe2 reproduction

**Files:**
- Create ignored outputs under: `validation_runs/ptse2_public_fit_method_20260719/`

**Step 1: Run targeted tests**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_configured_model.py \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_reporting.py \
  tests/kp/test_standalone_export.py \
  tests/kp/test_autoproject_examples.py
```

Expected: all selected tests pass.

**Step 2: Run the release suite**

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

Record the exact pass/fail count and distinguish pre-existing dirty-worktree
failures from feature regressions.

**Step 3: Run PtSe2 linear and nonlinear configurations**

Run on the user-selected compute nodes. Record energy RMS/max, raw matrix
RMS/max, same-index overlap, pair-subspace overlap, variable count, resolved
band count, and runtime. Do not place logs or generated arrays in release
directories.

**Step 4: Visually inspect PDFs**

Render the latest selected-window and all-band PDFs to PNG and inspect curve
agreement and overlap coloring.

**Step 5: Audit the diff**

```bash
git status --short
git diff --check
git diff --stat HEAD~5..HEAD
```

Confirm no validation output, node log, prompt, review package, cache, or
unrelated dirty file is staged.
