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

## Real-data correction: authored seeds are not closed

The initial coefficient-space design assumed that the authored real/imaginary
seed vocabulary `V` was closed under every generator, so that `g(V) = V A_g`.
That assumption is false for the production PtSe2 Gamma q04 vocabulary. For
example, C3, C2, and T route authored finite-p harmonic representatives to other
members of their symmetry orbits that were intentionally not authored as
independent term-registry entries. The current Reynolds compiler generates these
missing images during projection. Expanding the authored term registry would
change public harmonic semantics and is therefore not acceptable.

## Revised architecture

Compile symmetry in a closed symbolic atom space rather than in the authored
seed-label space. A complex polynomial response is decomposed into sparse atoms

```text
(monomial r,s; matrix row; matrix column; real/imaginary component).
```

Start from atoms present in the authored seeds, then take exact closure under the
certified generator actions and adjoint. Generator application only permutes or
mixes these scalar atoms through Q/sector routing, orbital blocks, phases, and the
small polynomial momentum action. It does not construct or canonicalize a full
38x38 CSR response for every group word.

Each authored real/imaginary seed direction is embedded into this closed atom
space. Reynolds projection and Hermitian projection are accumulated symbolically,
and a deterministic rank-revealing reduction selects an independent projected
frame. Only those retained symbolic channels are converted to physical sparse
polynomial responses. This preserves the existing authored term registry while
avoiding the generate-all-then-delete physical CSR path.

## Action construction

Each symbolic atom is indexed by its monomial, global matrix row and column, and
real/imaginary component. Authored seed provenance is stored separately and maps
logical fit coefficients into atom coordinates.

Generator actions are derived from certified exactified metadata, not operation
name geometry:

- sector and Q permutations route every nonzero matrix atom;
- Q-dependent phases provide the support phase ratio;
- small orbital blocks map one orbital pair to a sparse linear combination of
  orbital pairs;
- the explicit Cartesian momentum action maps `w^r wbar^s` to a small polynomial
  combination of atom keys;
- antiunitary parity supplies real/imaginary conjugation signs.

The finite-p adjoint swaps the atom row and column, swaps monomial powers, and
conjugates the real/imaginary component. Because authored seeds have already been
expanded in the global polynomial coordinate system, lower-order center-shift
terms are explicit atoms rather than an assumed same-degree seed partner.

## Fixed-space solution

Build the exact finite group from the certified generators, project authored
directions in atom coordinates, and perform deterministic sparse rank reduction.
The result stores the compact independent physical channels together with their
authored real/imaginary provenance. Fitting, nonlinear refinement, evaluation,
reporting, and standalone export consume the same compiled basis as before.

## Certification and fallback

The optimized path must fail closed unless all of the following hold:

- symbolic atom closure terminates and every routed atom remains inside the
  declared polynomial degree and matrix dimension;
- factorized actions match the active exactified continuum actions within their
  propagated bounds;
- symbolic generator actions satisfy the certified group relations;
- symbolic adjoint squares to identity and is compatible with every generator;
- retained materialized channels are invariant under every generator and adjoint;
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
