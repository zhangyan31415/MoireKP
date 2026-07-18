# KP High/Low Automatic Model Design

## Goal

Make one `kp model -c <case.yaml>` invocation choose polynomial orders and
important terms automatically, then export two complete standalone models:

- `high`: the most accurate validated low-energy model;
- `low`: the smallest model that preserves the requested top or bottom band
  edge, while allowing larger errors away from that edge.

The command must not require users to hand-author `Kinect`, intralayer, or
interlayer polynomial orders. Explicit maximum orders remain search ceilings.
Setting `fit.model_selection.enabled: false` is the escape hatch for a fixed
manual fit.

## Shared candidate search

Both profiles are selected from one auditable candidate pool. The pool is
built in the following order:

1. select the kinetic order from the centered response
   `H(k) - H(k_ref)`, without intralayer or interlayer terms;
2. add intralayer candidates and refit every active coefficient jointly;
3. add interlayer candidates and refit every active coefficient jointly;
4. rank whole symmetry- and Hermiticity-closed continuum terms by their
   projected contribution to the requested edge subspace, scan a compact
   keep-fraction ladder, and refit every retained vocabulary;
5. run a local joint order sweep around the staged result.

Previous stages freeze family orders, not coefficients. Every vocabulary is
compiled independently so response-basis rank decisions cannot leak between
different orders.

## Fixed target and validation data

Candidate comparisons use target-only weights computed once from the current
Heff harmonic support. Residual-dependent weights may refine a selected model
but do not rank candidates. Degenerate band-window boundaries are completed
as a group.

Ordered k paths use contiguous blocked folds. Duplicate path junctions stay
in one fold. Candidate metrics report the fold mean and standard error for:

- primary edge-weighted band RMS and maximum error;
- expanded-window and all-band errors;
- principal-angle subspace overlap;
- low-subspace matrix residual;
- independent real parameter count and active closed-group count.

## High profile

The high profile first filters candidates by numerical certification, global
guards, and mean primary-subspace overlap. The default target is `0.95`, with
a `0.90` safety floor. Among candidates reaching the target, high selects the
minimum primary edge-weighted validation RMS. Expanded-window error, maximum
error, and parameter count are deterministic tie breakers.

If no candidate reaches `0.95`, high returns the best candidate above `0.90`
as `WARN_BEST_AVAILABLE` and records every unmet target. It does not silently
relax the safety floor or claim success.

## Low profile

The low profile deliberately ignores expanded-window, all-band, and raw
matrix accuracy as selection gates. Those values remain visible diagnostics.
Its hard physics gates are:

- primary edge-weighted band quality;
- primary low-energy subspace overlap;
- numerical and production-symmetry certification.

Low uses an adaptive plateau relative to the best primary edge-weighted
candidate, then minimizes the number of independent real parameters. The
allowed increase is the maximum of two standard errors, a relative RMS
tolerance that scales with material difficulty, and a small absolute tolerance.
The defaults are `2SE`, `1.0 * best RMS`, and `0.10 meV`; each is configurable.
This prevents a statistically tiny SE from disabling the compact profile on a
difficult material while the overlap gate still protects the requested bands.

## Output layout

The configured output directory contains:

```text
<output>/
  high/                    complete high standalone model and diagnostics
  low/                     complete low standalone model and diagnostics
  auto_model_selection.json
  auto_model_selection.md
  candidate_metrics.csv
  model_complexity_frontier.pdf
```

Both profile directories are independently runnable standalone exports. The
root report records candidate hashes, fold definitions, fixed-weight hashes,
selection reasons, unmet targets, selected orders, active groups, independent
real parameter counts, and links to both exports.

For compatibility, the Python return value exposes `profile_results.high` and
`profile_results.low`, and treats `high` as the primary result where older
callers expect one model.

## One-command behavior and failures

Automatic selection is enabled by default for `kp model`. Existing
`model.max_order` values are ceilings, not forced orders. Internal valley
defaults supply ceilings when the user omits them.

A failed candidate is recorded and skipped. The run fails only when every
complete candidate fails, the best complete candidate violates the overlap
safety floor, production symmetry certification fails, or neither profile can
be exported. Partial output is labelled incomplete and is never presented as
a successful model.

## Verification

The implementation is accepted only when:

1. synthetic tests recover known high and low models and demonstrate that low
   may have worse non-edge error with fewer parameters;
2. `kp model` produces both runnable standalone exports in a small fixture;
3. MoTe2 K and MgI2 M external-data cases complete on bigmem and report real
   candidate metrics;
4. generated plots compare high, low, and Heff on the requested edge bands;
5. focused tests and the default non-slow, non-external release suite pass.
