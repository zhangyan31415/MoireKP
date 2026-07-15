# TAPW and KP CLI Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give `tapw run`, `tapw symm`, `tapw symm-rep`, `kp inspect`, `kp project`, `kp symm`, and `kp model` one restrained, TTY-aware scientific CLI presentation.

**Architecture:** Extend TAPW's logger-backed `TapwReporter` and add an independent `KpReporter` with the same small formatting contract. Workflows keep ownership of calculations and artifacts; reporters only render command identity, stages, aligned fields, checks, paths, and completion timing. Existing plain-text facts and error propagation remain intact.

**Tech Stack:** Python 3.11, standard-library ANSI formatting, `argparse`, `logging`, `pytest`, `capsys`, `monkeypatch`.

---

### Task 1: Define the KP reporter contract

**Files:**
- Create: `kp/kp/reporting.py`
- Create: `tests/kp/test_reporting.py`

**Step 1: Write the failing plain-output tests**

Add focused tests that construct `KpReporter("kp project")`, emit a title,
aligned fields, a stage, a path, a warning, and completion, and assert output
such as:

```python
assert "[kp project] Project effective Hamiltonian" in out
assert "  config     case.yaml" in out
assert "[kp project] [2/4] Build projectors" in out
assert "[kp project] OK  Completed in 1.25 s" in out
assert "\033[" not in out
```

Add separate tests for TTY color and `NO_COLOR`.

**Step 2: Run the tests to verify RED**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_reporting.py`

Expected: FAIL because `kp.reporting` does not exist.

**Step 3: Implement the minimal reporter**

Implement `KpReporter` with:

```python
class KpReporter:
    def __init__(self, command: str, *, stream=None, color: bool | None = None): ...
    def title(self, text: str) -> None: ...
    def section(self, text: str) -> None: ...
    def stage(self, current: int, total: int, text: str) -> None: ...
    def fields(self, rows: Iterable[tuple[str, object]]) -> None: ...
    def check(self, label: str, *, passed: bool, detail: str = "") -> None: ...
    def path(self, label: str, value: str | Path) -> None: ...
    def warning(self, text: str) -> None: ...
    def complete(self, *, elapsed: float | None = None) -> None: ...
```

Use stdout TTY detection by default, honor `NO_COLOR`, and keep ANSI outside the
semantic text.

**Step 4: Run the tests to verify GREEN**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_reporting.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/reporting.py tests/kp/test_reporting.py
git commit -m "feat(kp): add scientific CLI reporter"
```

### Task 2: Extend the TAPW reporter contract

**Files:**
- Modify: `tapw/tapw/reporting.py`
- Modify: `tests/tapw/test_run_reporting.py`

**Step 1: Write failing compatibility and style tests**

Add tests for command prefixes, aligned fields, named/numbered stages, status,
completion timing, TTY color, and `NO_COLOR`. Retain coverage for logger-backed
output, arrays, details, and multiline values.

**Step 2: Run the tests to verify RED**

Run: `python -m pytest -q -p no:cacheprovider tests/tapw/test_run_reporting.py`

Expected: FAIL because the current reporter has no command identity or the new
helpers.

**Step 3: Implement the minimal extension**

Add a `command` constructor argument and the same formatting helpers as
`KpReporter`, while preserving `line`, `section`, `stage`, `step`, `kv`, `array`,
`detail`, and logger routing used by TAPW internals. A logger-backed reporter must
emit plain text even when the console handler targets a TTY, so log files never
receive ANSI sequences.

**Step 4: Run the tests to verify GREEN**

Run: `python -m pytest -q -p no:cacheprovider tests/tapw/test_run_reporting.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add tapw/tapw/reporting.py tests/tapw/test_run_reporting.py
git commit -m "feat(tapw): extend CLI reporter presentation"
```

### Task 3: Integrate `tapw run` and `tapw symm`

**Files:**
- Modify: `tapw/tapw/cli.py`
- Modify: `tests/tapw/test_run_reporting.py`
- Modify: `tests/tapw/test_cli_surface.py`

**Step 1: Write failing command-identity tests**

Use lightweight monkeypatched configurations/workflows to assert that band mode
starts with `[tapw run]`, symmetry mode starts with `[tapw symm]`, configuration
fields are aligned, and successful completion includes elapsed time. Assert that
the existing shutdown path and real return code remain unchanged.

**Step 2: Run the targeted tests to verify RED**

Run: `python -m pytest -q -p no:cacheprovider tests/tapw/test_run_reporting.py tests/tapw/test_cli_surface.py`

Expected: FAIL on missing command-specific presentation.

**Step 3: Route the workflows through the reporter**

Choose `tapw run` for band mode and `tapw symm` for symmetry mode, emit a concise
run title, group configuration fields, and replace the bare final logger message
with reporter completion. Do not catch calculation exceptions or change
`finish_calculation_process`.

**Step 4: Run the targeted tests to verify GREEN**

Run: `python -m pytest -q -p no:cacheprovider tests/tapw/test_run_reporting.py tests/tapw/test_cli_surface.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add tapw/tapw/cli.py tests/tapw/test_run_reporting.py tests/tapw/test_cli_surface.py
git commit -m "feat(tapw): format run and symmetry output"
```

### Task 4: Integrate `tapw symm-rep`

**Files:**
- Modify: `tapw/tapw/symm_rep.py`
- Modify: `tests/tapw/test_tapw_symm_rep.py`

**Step 1: Write the failing output test**

Update the existing CLI smoke fixture to require `[tapw symm-rep]` title, stage,
results, aligned `summary` and `time breakdown` fields, and one completion line.

**Step 2: Run the smoke test to verify RED**

Run: `python -m pytest -q -p no:cacheprovider tests/tapw/test_tapw_symm_rep.py::test_symm_rep_cli_smoke_writes_files`

Expected: FAIL on the new hierarchy.

**Step 3: Replace private formatting helpers**

Instantiate `TapwReporter("tapw symm-rep")` in `main`, route progress, output
summary, and timing through it, and preserve every generated path and timing
value. Keep legacy-input support unchanged.

**Step 4: Run the TAPW symm-rep tests to verify GREEN**

Run: `python -m pytest -q -p no:cacheprovider tests/tapw/test_tapw_symm_rep.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add tapw/tapw/symm_rep.py tests/tapw/test_tapw_symm_rep.py
git commit -m "feat(tapw): format symmetry representation output"
```

### Task 5: Integrate `kp inspect` and `kp project`

**Files:**
- Modify: `kp/kp/cli.py`
- Modify: `tests/kp/test_canonical_output_layout.py`
- Modify: `tests/kp/test_cli_project_memory.py`

**Step 1: Write failing command-summary tests**

Extend lightweight command tests to assert command title, input/setup section,
domain summary, results paths, diagnostics status, and completion. Preserve
existing semantic assertions such as selected bands and downfolding warnings.

**Step 2: Run the targeted tests to verify RED**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_canonical_output_layout.py tests/kp/test_cli_project_memory.py`

Expected: FAIL on missing reporter hierarchy.

**Step 3: Format `kp inspect`**

Instantiate `KpReporter("kp inspect")` at the workflow boundary. Group config,
source files, Q/orbital layout, Fermi source, selections, plot/data paths, and
completion. Keep the orbital coefficient table contents and warning text.

**Step 4: Run the inspect tests**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_canonical_output_layout.py`

Expected: PASS.

**Step 5: Format `kp project`**

Instantiate `KpReporter("kp project")`; group selection, basis/gauge/downfolding,
diagnostics, and result artifacts. Route near-pole attention through warning
formatting without hiding it. Preserve memory-mapping and worker behavior.

**Step 6: Run the project tests**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_cli_project_memory.py tests/kp/test_canonical_output_layout.py`

Expected: PASS.

**Step 7: Commit**

```bash
git add kp/kp/cli.py tests/kp/test_canonical_output_layout.py tests/kp/test_cli_project_memory.py
git commit -m "feat(kp): format inspect and project output"
```

### Task 6: Integrate `kp symm`

**Files:**
- Modify: `kp/kp/symmetry/projection.py`
- Modify: `kp/kp/cli.py`
- Modify: `tests/kp/test_symm_projection.py`

**Step 1: Write a failing workflow-boundary test**

Use the existing small symmetry fixture to require `[kp symm]`, source operation
families, projection/exactification stage labels, production artifact paths, and
completion. Assert no forbidden directional operation labels are introduced.

**Step 2: Run the targeted test to verify RED**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_symm_projection.py`

Expected: FAIL on missing structured output.

**Step 3: Add reporter integration**

Create the reporter at the CLI boundary and pass it into
`run_symmetry_projection_from_config`. Replace the two direct operation prints
and add result/check summaries using already-computed metadata. Clearly label
developer diagnostics separately from production exactified matrices.

**Step 4: Run the tests to verify GREEN**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_symm_projection.py tests/kp/test_operation_standardization.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/cli.py kp/kp/symmetry/projection.py tests/kp/test_symm_projection.py
git commit -m "feat(kp): format symmetry projection output"
```

### Task 7: Integrate `kp model`

**Files:**
- Modify: `kp/kp/cli.py`
- Modify: `kp/kp/model/pipeline.py`
- Modify: `tests/kp/test_configured_model.py`
- Modify: `tests/kp/test_standalone_export.py`

**Step 1: Write failing progress and summary tests**

Require one `[kp model]` title, consistent stage/status vocabulary for existing
fit progress, aligned final model fields, comparison metrics with their explicit
reference, artifact paths, export path, and completion. Keep the existing TTY
color progress test and add `NO_COLOR` coverage.

**Step 2: Run the targeted tests to verify RED**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py tests/kp/test_standalone_export.py`

Expected: FAIL only on the new presentation assertions.

**Step 3: Route model output through `KpReporter`**

Replace `_kp_model_print` with a thin reporter-backed compatibility function,
use one reporter for the final summary, and update pipeline progress emitters to
share prefix/color semantics. Do not change fitting algorithms, term selection,
metrics, or export cleanup.

**Step 4: Run the targeted tests to verify GREEN**

Run: `python -m pytest -q -p no:cacheprovider tests/kp/test_configured_model.py tests/kp/test_standalone_export.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add kp/kp/cli.py kp/kp/model/pipeline.py tests/kp/test_configured_model.py tests/kp/test_standalone_export.py
git commit -m "feat(kp): format continuum model output"
```

### Task 8: Verify release behavior and a MoTe2 K example

**Files:**
- Create locally only: `validation_runs/cli_output_mote2_k_<timestamp>/`
- Do not add validation outputs to git.

**Step 1: Check the runtime environment**

Run `which python`, `python -V`, `which tapw`, `which kp`, both help commands, and
`python -m threadpoolctl -i numpy scipy`. Confirm the commands come from the
`moirekp` environment, BLAS is MKL, and both `libiomp` and `libomp` are not loaded.

**Step 2: Run focused CLI tests**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/tapw/test_run_reporting.py \
  tests/tapw/test_cli_surface.py \
  tests/tapw/test_tapw_symm_rep.py \
  tests/kp/test_reporting.py \
  tests/kp/test_canonical_output_layout.py \
  tests/kp/test_cli_project_memory.py \
  tests/kp/test_symm_projection.py \
  tests/kp/test_configured_model.py \
  tests/kp/test_standalone_export.py
```

Expected: PASS.

**Step 3: Run the release-facing test suite**

Run: `python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"`

Expected: PASS.

**Step 4: Exercise the MoTe2 K pipeline where inputs exist**

Use the canonical MoTe2 K configuration under `examples/`, redirect the
transcript and any copied configuration to the timestamped validation directory,
and keep command-generated artifacts out of tracked example roots. Exercise the
available subset of TAPW/KP stages without fabricating absent external data.

**Step 5: Inspect release diff and ignored outputs**

Run `git diff --check`, inspect the complete targeted diff, and confirm no
`validation_runs`, cache, egg-info, large example output, prompt, absolute local
path, or forbidden operation label is staged.

**Step 6: Commit any final test-only cleanup**

Stage only files from this plan and commit with a narrowly scoped message. Do not
stage unrelated pre-existing modifications.
