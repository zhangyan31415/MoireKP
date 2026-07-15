# TAPW and KP CLI Output Design

## Goal

Give the release-facing TAPW and KP workflows a consistent, restrained
scientific-command-line presentation without changing their calculations,
configuration contracts, generated artifacts, or shell exit status.

The commands in scope are:

- `tapw run`
- `tapw symm`
- `tapw symm-rep`
- `kp inspect`
- `kp project`
- `kp symm`
- `kp model`

## Visual language

Every user-facing line belongs to the command that emitted it. Command prefixes
use the exact release-facing spelling, for example `[tapw run]` and
`[kp project]`. Output follows the same hierarchy across workflows:

1. command title and essential run identity;
2. numbered or named stages for long work;
3. aligned key/value summaries within a stage;
4. a compact results block containing generated artifact paths and the most
   useful numerical checks;
5. one explicit completion status and elapsed time.

The default style avoids wide boxes, animated spinners, terminal cursor control,
and decorative Unicode. This keeps cluster logs readable and makes copied output
useful in issue reports.

Interactive terminals use a small ANSI palette: cyan for command identity,
bold text for headings, green for successful checks, yellow for warnings and
metrics that need attention, and magenta for paths. ANSI styling is disabled
when stdout is not a TTY or `NO_COLOR` is present. The text and whitespace remain
meaningful without color.

Example:

```text
[kp project] Project effective Hamiltonian
  config       examples/.../K1_A_q04.yaml
  material     MoTe2
  basis        36 active bands; 18 Q points
  k-points     61 / 61

[kp project] [2/4] Build projectors
  gauge        symmetry-adapted
  downfolding  fixed_schur
  status       OK

[kp project] Results
  Hamiltonian  .../heff.npy
  bands        .../heff_eig.npy
  runtime      18.42 s

[kp project] OK  Completed
```

## Reporter architecture

TAPW extends its existing `TapwReporter`. The reporter gains command identity,
TTY-aware styling, aligned fields, stage/status helpers, result paths, and timing
output. It continues to write through the configured logger so the TAPW run log
and console receive the same plain message content. `tapw run` and `tapw symm`
instantiate it with different command identities; `tapw symm-rep` replaces its
private formatting helpers with the shared reporter.

KP gets its own small reporter module with the same presentation contract. KP
must not import TAPW reporting code: the two public packages remain independently
usable. `kp inspect` and `kp project` replace direct command-level `print` calls;
`kp symm` receives a reporter at its workflow boundary; and `kp model` reuses one
reporter for progress and its final summary. Existing model-pipeline progress
callbacks remain available and are routed through this reporter where practical.

The reporters format information only. They do not own domain data, create
artifacts, catch workflow exceptions, or reinterpret physics results.

## Command summaries

`tapw run` reports configuration, structure/basis preparation, Hamiltonian
assembly, diagonalization/band work, artifact paths, and total runtime.

`tapw symm` reports source configuration, discovered operation families and
action metadata, raw-H validation status, representation artifacts, and runtime.
It does not introduce deprecated directional operation labels.

`tapw symm-rep` reports input identity, high-symmetry-point progress, cache use,
character/summary outputs, timing breakdown, and completion.

`kp inspect` reports source inputs, inferred orbital/Q layout, Fermi-level source,
selected bands, plot/data outputs, and completion. Existing orbital-component
tables remain tabular and retain their numerical content.

`kp project` reports input and selection summary, basis/gauge/downfolding stages,
important diagnostics, artifact paths, and completion. Warnings near Schur poles
remain visible and are not downgraded to verbose-only output.

`kp symm` reports source operation families, basis identity, projection and
exactification stages, validation residuals, production artifacts, and
completion. Diagnostic matrices are clearly labelled and never presented as
production matrices.

`kp model` reports model identity, response/term preparation, fitting stages,
symmetry checks, comparison metrics with explicit references, export paths, and
completion. Low-level fitting progress remains available but adopts the same
prefix and status vocabulary.

## Compatibility and errors

Existing command arguments and configuration keys do not change. Existing
human-readable facts remain present, although their grouping and spacing may
change. Tests and callers must not depend on ANSI sequences. Error handling stays
at the current workflow boundary so failures retain their real exception and
nonzero shell status. The reporter may render a warning or failed check only when
the workflow already has the corresponding status; it never converts an error to
success.

No new runtime dependency is added.

## Testing and validation

Reporter unit tests cover aligned fields, command prefixes, TTY color, `NO_COLOR`,
non-TTY output, statuses, paths, and elapsed time. Command-level tests capture
plain stdout for all seven commands and assert the shared hierarchy plus their
command-specific result fields. Tests assert stable semantic substrings rather
than entire colorized transcripts.

Implementation follows red-green-refactor cycles. Targeted CLI tests run after
each integration, followed by the release-facing suite:

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

A MoTe2 K-valley example is run from the `moirekp` MKL environment when its local
external inputs are available. One-time outputs and captured transcripts go under
`validation_runs/`; no generated example data enters the release surface.
