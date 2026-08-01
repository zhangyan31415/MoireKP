# CPC Full Integration Design

**Status:** Approved

**Date:** 2026-08-01

**Base:** `codex/cpc-release-clean@847e32cca35c62d17f84e86916b81e60bb0a743e`

**Integration branch:** `codex/cpc-full-integration`

## Goal

Complete the remaining release-facing response storage, automatic low-energy
selection, TAPW `system.structure`, and example-contract work without merging
the historical `codex/kp-auto-model-selection` branch or overwriting newer CPC
implementations.

The final result must preserve physical response-space gauge covariance, make
automatic selection fail closed, support the already-tracked TAPW `system:`
configs, pass the full local release gate, and pass targeted external-data
validation on `bigmem001` and `bigmem003` only.

## Decisions

1. Work is developed in independent feature worktrees and integrated through a
   single hub. The CPC root worktree stays on `codex/cpc-release-clean` until the
   final fast-forward.
2. `kp-auto-model-selection-preserved` is a read-only reference. It must not be
   merged, rebased, edited, or used as an entire-file source.
3. Automatic selection is fail closed. An explicit `nlow_state_list` may run as
   a diagnostic, but remains `PENDING` or `UNVERIFIED` until certified.
4. Validation uses physical equivalence: symmetry, Hermiticity, projector rank,
   response span, gauge covariance, Hamiltonian residual, degeneracy, and band
   spectrum within declared tolerances. Byte-identical output is not required.
5. The experimental `interlayer_scale` and its local campaign are excluded.
6. External computation is restricted to `bigmem001` and `bigmem003`.
   `bigmem002` must be rejected explicitly.

## Repository And Branch Architecture

All feature branches start at the exact approved base SHA:

```text
codex/cpc-release-clean @ 847e32c
        |
        +-- codex/cpc-full-integration
        +-- codex/cpc-storage-compression
        +-- codex/cpc-auto-selection-core
        +-- codex/cpc-candidate-symmetry
        +-- codex/cpc-tapw-system-structure
        +-- codex/cpc-example-contracts
```

Each feature agent owns a separate worktree and non-overlapping files. The
integration agent reviews and cherry-picks narrow commits. Conflicts return to
the feature agent; the integration agent does not resolve them with broad
`ours` or `theirs` choices.

After local and external validation, the root CPC branch advances only with:

```bash
git merge --ff-only codex/cpc-full-integration
```

Push remains a separate, explicit final action.

## Response Basis And Storage

The current CPC response implementation is the only source baseline. Preserve:

- case-derived candidate generation;
- finite-momentum symbolic projection;
- exactified and factorized symmetry actions;
- Hermitian and adjoint closure;
- fixed-space rank certification and independent-column scaling;
- Gamma orbital-swap and K/M dense-gauge covariance tests;
- persistent cache identity.

Do not copy preserved versions of `response_basis.py`,
`response_basis_factorized.py`, `response_basis_fixed_compiler.py`,
`pipeline.py`, `core.py`, or their full test files.

The v47 entrywise sparse cleanup is rejected. Entrywise pruning can split an
adjoint pair, changes under dense orbital gauge transformations, and modifies
retained coefficients after dependency certification. The response compiler
therefore remains at its current physical semantics and version.

The only storage change in this release is lossless container compression:

```python
np.savez(path, **arrays)
```

becomes:

```python
np.savez_compressed(path, **arrays)
```

This does not change response coefficients, span, rank, compiler version, or
artifact identity. Any future exact compact representation receives a separate
`storage_format_version`. A lossy representation is outside this design.

## Automatic Low-Energy Selection

Automatic selection is implemented as typed, testable core logic rather than a
large CLI-local procedure.

### Data Flow

```text
canonical case/Q/layout + full Hamiltonian + raw-H source actions
        |
        v
fixed TargetWindowSpec and cached target eigenpairs
        |
        v
energy-cluster seeds at canonical reference k/Q
        |
        v
raw-H projector closure, including antiunitary partners
        |
        v
post-closure continuum-sector assignment
        |
        v
candidate projection and downfolding certificate
        |
        v
fixed-band error + principal-angle overlap
        |
        v
exactified symmetry and covariance certificate
        |
        v
select the smallest candidate that passes every hard gate
        |
        v
freeze SelectionArtifact and atomically write the final report
```

### Target Window

`TargetWindowSpec` binds the edge, fixed band count, validation k points,
energy reference, and degeneracy policy. Every candidate is evaluated against
the same target bands. A candidate smaller than the requested window is
rejected. Per-k and per-spectrum independent edge alignment is forbidden.

If the requested boundary cuts a protected or tolerance-certified degenerate
multiplet, selection fails closed unless the configured policy explicitly
expands the window.

### Gamma Closure

Gamma starts from a joint spectral cluster/projector. Individual eigenvector
layer weights do not define ownership before closure because that choice changes
under unitary rotation inside a degenerate eigenspace.

Raw-H source actions close the seed projector under unitary and antiunitary
generators. Only after closure is the joint subspace routed to continuum source
groups. Physical-layer and orbital weights are diagnostics. Ambiguous routing
rejects the candidate.

K and M use the same certificate path but may start with already-routed source
blocks.

### Metrics

Band RMS/max error uses the fixed target window and one common energy reference.

Principal-angle overlap is computed between the full target eigenspace and the
full-space reconstruction of the model target eigenspace. The minimum squared
singular value over all validation k points is the hard gate. Gauge-anchor
`sigma_min` remains a separate conditioning diagnostic.

For each operation and mapped k pair, the symmetry evaluator records:

- raw-H off-subspace leakage;
- distance from the projected raw action to the exactified action;
- exactified intertwining residual;
- effective-Hamiltonian covariance residual;
- operation and pair coverage;
- unitary, group-relation, and antiunitary-square certificates.

Missing operations, missing pairs, `None`, NaN, or infinite metrics reject the
candidate. Summary maxima are derived from typed per-operation records, not by
recursively scanning JSON output.

### Artifacts And Status

The frozen `SelectionArtifact` binds target-window configuration, canonical Q
and row-layout hashes, source identity, source-action package, resolved gauge,
candidate definition, projection certificate, symmetry certificate, metrics,
and selector version.

Report status is separated into:

```text
selection_status: SELECTED | FAILED
certification_status: PENDING | CERTIFIED | FAILED | UNVERIFIED_OVERRIDE
release_status: PASS | BLOCKED
```

Uncomputed values serialize as `null` with a reason. A top-level PASS is written
only after complete certification, using a temporary file and atomic replace.

### Failure Policy

- Invalid source manifest, Q layout, target window, or operation coverage is a
  global error and terminates the command.
- Candidate-local rank loss, near pole, insufficient dimension, failed closure,
  or failed quality threshold rejects that candidate and continues.
- If no automatic candidate passes, the command exits nonzero and produces no
  certified model.
- Explicit selection may continue for diagnostics but is not release-valid
  until it obtains the same symmetry and identity certificates.
- Artifact identity mismatch is a transaction-integrity error.

## TAPW `system.structure`

The repository already tracks release YAML files using `system:`, while the
current loader does not fully support them. This is a release blocker.

Loading is divided into:

```text
raw YAML
  -> strict schema normalization
  -> ResolvedStructureInput
  -> run / symm / symm-rep / topo / orbital analysis
```

The normalized system layer validates fields, resolves paths relative to the
YAML file, keeps new and legacy schemas mutually exclusive, and converts both
supported inputs to one internal representation.

`ResolvedStructureInput` binds structure content hash, format, original site
order, species, lattice, coordinates, periodicity, orbital mapping, spin,
expected basis dimension, resolved 2D Bravais family, and provenance.

ASE loading must preserve site order and must not silently rotate, standardize,
or species-sort the structure. Compact orbital specifications must cover every
present species and no absent species. H and S dimensions are checked against
the resolved basis before any eigensolver is entered.

Bravais inference uses explicit tolerances and returns a value through normal
data flow. Ambiguous or unsupported metrics fail closed. Process-global
`TAPW_BRAVAIS` state is not introduced.

All TAPW public workflows consume the same resolved structure and source
identity. Save/load round-trip must produce an equivalent internal config;
saved configuration must not contain fields rejected by its own loader.

## Examples, Documentation, And Data

Only final small YAML, README sections, and manifest records may enter the
release diff. Existing LGPL text is preserved. The following remain outside
Git:

- `validation_runs/`;
- large OpenMX/TAPW/KP arrays and symmetry transport files;
- ZIP/TAR/H5 archives;
- logs, plots, failed runs, temporary inputs, and duplicate source trees;
- local node scripts and absolute paths.

External datasets require manifest provenance, size, checksum, license, DOI,
and public data URL policy. Missing external input is reported as
`BLOCKED_EXTERNAL_DATA`; it is never replaced with a similar case.

## Verification

### Feature Gates

Each feature branch runs targeted tests, `git diff --check`, and a release-hygiene
scan before producing a narrow commit.

### Integration Gate

The hub verifies the Python 3.11 `moirekp` MKL environment and runs:

```bash
python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"
```

The gate requires zero failures, correct CLI provenance, cold/warm cache
identity, no fallback provenance, and a clean worktree.

### External Validation

Only `bigmem001` and `bigmem003` may run external-data jobs, using the exact same
hub SHA. All outputs go to `validation_runs/`.

`bigmem001` covers MgI2 and ZrS2 Gamma cases, response cache and gauge-covariance
checks, and verified 2.13/2.45 inputs when available.

`bigmem003` covers AAB MoTe2, PtSe2, and BiTeI Gamma cases, multilayer Gamma row
order and Kramers checks, K/M non-regression, and verified 2.28/2.65 inputs when
available.

Both nodes run one common small MgI2 Gamma case for cross-node reproducibility.
Preflight resolves and hashes every external input. Unavailable 2.45 or 2.65
data blocks only that case and is reported explicitly.

Each case records command status, artifact identity, symmetry coverage and
residuals, Hermiticity, projector rank, response span, gauge covariance,
Hamiltonian residual, expected degeneracy, band comparison, and cold/warm cache
reproducibility.

## Completion Condition

The work is complete only when all narrow commits are reviewed in the hub, the
full local release gate is green, the required available external cases pass on
the allowed nodes, release hygiene is clean, and the final hub can fast-forward
the CPC release branch.
