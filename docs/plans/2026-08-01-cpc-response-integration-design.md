# CPC Response-Space Integration Design

## Goal

Integrate automatic model selection, the complete response-basis compiler, and
Gamma/K/M case-derived response-space generation into the current CPC release
line without merging temporary research snapshots or unrelated working-tree
changes.

## Integration baseline

The integration branch starts from the latest fetched
`origin/codex/cpc-release-clean`. The older local CPC branch remains unchanged
as a historical checkpoint. All work is performed in an isolated worktree.

The clean baseline has one known contract mismatch: canonical model cleanup
deletes the Hamiltonian-element comparison PNG/PDF even though the matching
test and original feature contract require both files to be retained. This is
repaired first in an independent commit so later test results have a clean
baseline.

## Source branches and snapshots

The source material has three distinct roles:

1. `codex/kp-auto-model-selection` contains committed automatic model-selection
   policy and its design history. Patch-equivalent release and example commits
   already present in CPC are not replayed.
2. The temporary response-basis snapshot is used only as a source reference.
   It is not merged or cherry-picked as a whole because it combines many
   unrelated files in one non-release commit.
3. `codex/case-derived-response-space` contains the validated final working
   state and the Gamma/K/M case-derived design commits. Its dirty working tree
   is treated as a source tree, never as a merge target.

## Commit architecture

The final CPC history is linear and reviewable:

1. Repair the pre-existing canonical-output cleanup contract.
2. Replay only CPC-missing automatic model-selection commits.
3. Add response-basis data structures, adjoint closure, cache, oracle, and
   factorized/fixed-space compiler modules with their focused tests.
4. Add the exactified symmetry-action and structure-certificate dependencies
   required by the response compiler.
5. Connect the compiler to configured model fitting, export, and persistent
   cache paths.
6. Add Gamma/K/M case-derived candidate generation and remove active complete
   hard-coded valley profiles while retaining `legacy_frozen_v1` compatibility.
7. Add dense orbital-gauge covariance, Q-pair support-convention, refit, and
   standalone-export regressions.
8. Integrate only release-facing documentation and example configuration
   changes that are not already patch-equivalent in CPC.

Each commit must stage an explicit file list. Validation artifacts, local
reports, large generated data, and temporary snapshots are never staged.

## Response-space behavior

For `complete_linear_v2`, Gamma, K, and M all defer candidate construction until
the real case has resolved sectors, orbital dimensions, Q sets, harmonic maps,
and order cutoffs. The generator enumerates the complete physical envelope:

- same-sector zero-harmonic kinetic and onsite candidates;
- same-sector candidates for every supported intralayer harmonic;
- both cross-sector directions for every supported interlayer harmonic;
- every ordered orbital pair allowed by the resolved sector dimensions.

The existing exact group projection, Hermitian-adjoint closure, certified-zero
filtering, and target-independent rank reduction then produce the compact
basis. Legacy frozen templates remain available only when explicitly selected
through `legacy_frozen_v1`.

Support counting has two explicit conventions. Automatic harmonic diagnostics
may include configured sector offsets. Model materialization follows the core
operator equation `Q_from - p = Q_to` and must not add those offsets a second
time.

## Safety boundaries

- The shared release checkout is never switched, staged, or committed.
- The temporary response-basis snapshot is never merged wholesale.
- Paper sources and local validation directories are outside this integration.
- Active symmetry operation names remain family-only; no deprecated directional
  aliases are introduced.
- KP consumes exactified continuum actions; diagnostic matrices never enter the
  production model silently.

## Verification

Verification proceeds in increasing scope:

1. Run the focused test that demonstrates each migration step fails before the
   corresponding production code is present.
2. Run focused module tests after each commit.
3. Run the complete response/configured-model targeted set after compiler and
   case-derived integration.
4. Re-run the approved Gamma, K, and M example cases and inspect bands and
   Hamiltonian residuals.
5. Run the release-facing non-slow, non-external-data suite only after the user
   authorizes that final release gate.

The branch is ready to merge only when the integration worktree is clean and
the required verification commands pass on the integrated CPC history.
