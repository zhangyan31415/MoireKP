# Finite-p Symmetry-First Response Compiler Design

## Goal

Reduce the PtSe2 Gamma q04 `complete_linear_v2` cold response-basis compile from
55.8 seconds to at most 18.6 seconds on bigmem001 without changing the invariant
response span or degrading fitted Hamiltonians, bands, overlaps, or covariance.
The optimized basis may use different channels and therefore a different basis
hash.

## Measured baseline

The uncached PtSe2 Gamma q04 compiler currently reports:

- input fingerprint: 2.02 s for 942 authored terms;
- candidate compilation: 39.58 s;
- rank reduction: 14.18 s;
- total compile call: 55.81 s;
- retained basis: 456 channels.

The finite-p vocabulary contains 720 complex seeds and 1440 real coefficient
directions. The current compiler physically Reynolds-projects all 1440 directions
before removing 263 certified zero directions and roughly 742 additional linear
dependencies. Only 435 finite-p channels survive. Function-level profiling shows
that repeated CSR construction, canonicalization, masking, addition, and hashing
dominate the runtime; the target Hamiltonian fit is not the bottleneck.

## Architecture

Compile exact generator and adjoint actions in the authored real coefficient
vocabulary before materializing symmetry-projected Hamiltonian responses.

Let `V` contain the raw real/imaginary polynomial-response columns for all authored
seeds. For every exactified generator `g`, construct a sparse real action `A_g`
such that

```text
g(V) = V A_g.
```

Construct a sparse real adjoint involution `J` such that

```text
V(c)^dagger = V(J c).
```

The allowed coefficient space is the common fixed space of the actual generators
and the adjoint:

```text
A_g c = c  for every generator g
J c = c.
```

The fixed-space coordinates form a sparse/block-structured matrix `B`. Only
`V @ B` is materialized as the production response basis. This avoids a complete
full-group Reynolds projection for every finite-p real seed direction.

## Action construction

Each authored seed is indexed by its full descriptor: support component, source
and target sector, source and target orbital, explicit harmonic/Q-pair support,
polynomial powers, and real/imaginary component.

Generator actions are derived from certified exactified metadata, not operation
name geometry:

- sector and Q permutations route the seed support;
- Q-dependent phases provide the support phase ratio;
- small orbital blocks map one orbital pair to a sparse linear combination of
  orbital pairs;
- the explicit Cartesian momentum action maps `w^r wbar^s` to a small polynomial
  combination;
- antiunitary parity supplies real/imaginary conjugation signs.

The finite-p adjoint swaps source and target support, sends `p` to `-p`, exchanges
orbital indices, conjugates the coefficient, and performs the exact binomial
translation required by the changed Q center. It is a sparse real-linear
involution and is not approximated as a pairwise same-degree mapping.

## Fixed-space solution

Build the union support graph of all `A_g` and `J`. Its connected components are
certified invariant and solved independently. Small component null spaces give
the common fixed coordinates. The physical Gram metric of `V` then normalizes
the retained fixed directions blockwise.

The result stores both the compact independent channels and the mapping from
independent amplitudes back to authored real/imaginary term coefficients. Fitting,
nonlinear refinement, evaluation, reporting, and standalone export consume the
same compiled basis.

## Certification and fallback

The optimized path must fail closed unless all of the following hold:

- every routed descriptor is present in the authored vocabulary;
- factorized actions match the active exactified continuum actions within their
  propagated bounds;
- generator matrices satisfy the certified group relations;
- `J^2 = I` and adjoint/group compatibility hold within propagated bounds;
- the fixed basis is invariant under every generator and `J`;
- materialized channels satisfy physical covariance and Hermiticity checks.

The existing per-seed dense/full-matrix Reynolds implementation remains the
correctness oracle and production fallback. Fallback is explicit in progress and
artifact diagnostics.

## Alternatives rejected

- Merely parallelizing or batching the existing per-seed Reynolds compiler is
  expected to improve runtime only to roughly 25--35 seconds and retains the
  generate-then-delete architecture.
- Hard-coded C3/C2/TR selection rules may be faster but would weaken the general
  exactified-action contract and are not acceptable for the public compiler.

## Validation

Unit and oracle tests must cover identity, unitary, and antiunitary toy groups;
non-diagonal orbital blocks; finite-p harmonic reversal; and adjoint lower-order
polynomial mixing.

For PtSe2 Gamma q04:

- compare optimized and dense-oracle invariant subspaces by principal angles;
- compare Hamiltonian spans at random two-dimensional momenta;
- compare fitted Hamiltonians, bands, overlap diagnostics, and covariance;
- require 456 retained channels, allowing a different basis hash;
- run three cache-disabled cold compiles on bigmem001 and require median total
  compile time no greater than 18.6 seconds.

The full release test suite and targeted response-basis tests must pass before
the optimized compiler becomes the default complete-response path.
