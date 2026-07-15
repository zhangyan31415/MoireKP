# KP Joint U(d) Block Exactification Design

## Status

Approved on 2026-07-15. This design is limited to KP continuum
representations. It does not change TAPW source symmetry generation.

## Problem

`kp symm` currently exactifies every operation independently. An output can
therefore satisfy each declared power relation while failing a mixed group
relation. The MgI2 M-valley spin-up output is the concrete reproducer:

```text
TR^2 residual                         0
C2^2 residual                         2.06e-16
TR C2* - C2 TR residual               1.7450083966e-5
best-global-phase mixed residual      1.7450083966e-5
```

The existing absolute storage cleanup is a separate correctness risk. It
sets every exactified matrix entry below `1e-4` to zero. It removed no entries
in the MgI2 reproducer and is therefore not the cause of that residual, but it
can destroy a legitimate small entry inside a general U(d) block.

## Requirements

1. Exactify the complete generator set, including mixed unitary/antiunitary
   relations, rather than exactifying operations independently.
2. Support route-dependent general U(d) blocks. Do not assume that all Q
   routes share one orbital block up to a scalar phase.
3. Preserve the current independently exactified branches. The joint layer
   should remove only the small closure defect and fail if a large correction
   is required.
4. Treat projective phases as explicit central group data. Do not accept an
   arbitrary best global phase as group closure.
5. Remain covariant under the existing symmetry-adapted basis gauge.
6. Create exact zeros only from structural block support. Never mutate a
   within-block value because its magnitude is below an engineering threshold.
7. Store enough provenance and diagnostics to reproduce and certify the
   exactification after loading.
8. Keep local testing to small matrix algebra. Run complete external-data
   material pipelines through SSH on a bigmem node.

## Semilinear Block Representation

Let the continuum basis be a direct sum of fibers,

```text
H = direct_sum_x V_x.
```

A generator `g` has an exact fiber permutation `p_g`, antiunitary parity
`a_g`, and a route block

```text
B_g(x): V_x -> V_{p_g(x)}.
```

Define `kappa_0(M) = M` and `kappa_1(M) = conjugate(M)`. Composition is

```text
B_gh(x) = B_g(p_h(x)) kappa_a_g(B_h(x)),
a_gh = a_g xor a_h.
```

Every relation is evaluated with this rule. In particular:

```text
MgI2 M:       T T* = I, C2^2 = I, T C2* = C2 T
MoTe2 K:      C3^3 = -I, (C2T)(C2T)* = I,
              (C2T) C3* = C3^-1 (C2T)
spinful Gamma: C3^3 = C2^2 = T T* = -I,
               T C3* = C3 T, T C2* = C2 T,
               C2 C3 = C3^-1 C2
```

The presentation compiler must record these relations and their central
targets explicitly.

## Gauge Covariance

For a fiber gauge `W_x in U(d_x)`, route blocks transform as

```text
B'_g(x) = W_{p_g(x)}^dagger B_g(x) kappa_a_g(W_x).
```

For every word `w`, induction gives

```text
B'_w(x) = W_{p_w(x)}^dagger B_w(x) kappa_a_w(W_x).
```

Consequently, the Frobenius norm of a relation defect is invariant under a
common gauge. A gauge cannot repair the MgI2 mixed defect, and a consistent
gauge cannot create one except at the floating-point multiplication floor.

The production order remains:

```text
independent support/power branch selection
-> joint closure exactification
-> derive the symmetry-adapted gauge
-> apply the same gauge to symmetry matrices, Heff, and wavefunctions
-> certify all relations again
-> store without numerical threshold cleanup
```

No operation may be re-exactified independently after the common gauge.

## Exact-by-Construction Orbit--Stabilizer Parameterization

The primary production algorithm operates on the action groupoid rather than
on dense matrices.

For each orbit, choose a root `x0` and one canonical transporter `t_x` with
`t_x x0 = x`. Let `H = Stab(x0)`. For `y = g x`, define the Schreier element

```text
h(g, x) = t_y^-1 g t_x in H.
```

Write a semilinear block as `[B, a]`. Let `S_x` be the semilinear transporter
block and let `rho` be an exact U(d) corepresentation of the stabilizer. Every
route block is reconstructed by

```text
B_g(x) = S_y rho(h(g, x)) S_x^-1
```

using semilinear composition.

This parameterization is complete: every exact block-monomial representation
defines such `S_x` and `rho`, and every exact `rho` with unitary transporters
reconstructs an exact representation. Mixed group relations are therefore
satisfied by construction rather than by a penalty term.

The closest-representation objective is

```text
min 1/2 sum_(g,x) w_(g,x) ||B_g^stage1(x) - B_g(S,rho;x)||_F^2.
```

The default weights are equal and are recorded. Any non-default weights must
be explicit configuration, not inferred silently from operation names.

### Free orbits

If the stabilizer contains only fixed central elements, the problem is a
semilinear U(d) synchronization problem over the transporters. Every iterate
materializes a group-closed representation.

The current MgI2 M and MoTe2 K datasets consist entirely of free permutation
orbits:

```text
MgI2 M:  11 orbits of size 4
MoTe2 K:  9 orbits of size 6
```

### Nontrivial stabilizers

Only the stabilizer corepresentation needs constrained U(d) exactification.
Use anti-Hermitian updates

```text
R'_j = exp(X_j) R_j, X_j^dagger = -X_j,
```

an analytic real Jacobian for all stabilizer relators, rank-revealing SVD,
and trust-region or line-search control. Record the Jacobian rank, expected
gauge nullity, smallest nonzero singular value, condition number, correction
norm, and relation-log branch margin.

ZrS2 Gamma has three free size-12 orbits and one size-2 orbit whose
permutation stabilizer has order 6. Only the latter needs the small U(4)
stabilizer solver.

### Direct route-space oracle

A direct constrained solver over every generator route is useful as a small
matrix test oracle, but it is not the production parameterization. It contains
large relation redundancy and fiber-gauge nullspaces. The oracle must agree
with the orbit--stabilizer result on correction norm and final relations for
small fixtures.

## Projective Central Extension

The group compiler must retain the central extension explicitly. A spinful
central element such as a 2-pi rotation is not interchangeable with identity:

```text
C3^3 = Ebar,
C2^2 = Ebar,
T^2 = Ebar,
rho(Ebar) = -I.
```

The current finite-group compiler identifies elements by their discrete action
and accepts alternate words modulo an arbitrary phase. That behavior remains
diagnostic-only. Production closure uses a declared presentation and central
target. If a mixed central branch is not uniquely determined from the allowed
finite central kernel with a safe margin, exactification fails and requests
explicit metadata.

V1 supports the scalar central targets needed by the current M, K, and Gamma
examples. The artifact schema leaves room for a declared central block action
for future nonsymmorphic translations; V1 must fail rather than guess such an
action.

Before solving, validate representation compatibility. For example, a fixed
fiber antiunitary satisfying `A A* = -I_d` requires even `d`.

## Stage-1 and Joint-Layer Boundary

The existing support resolution, polar normalization, power-root selection,
and optional algebraic template recognition remain the stage-1 branch
selector. The new joint layer consumes those matrices, not the original raw
TAPW projections. This keeps the requested correction local and prevents the
joint solver from selecting a different physical/root branch.

The existing block exactifier averages aligned route blocks into one common
block. That remains an optional factorized fast path only. The joint layer's
canonical input and output use one U(d) block per source fiber route.

## Artifact Model

Add a versioned joint representation artifact containing:

- basis ordering and fiber dimensions;
- generator names and antiunitary parity;
- exact fiber permutations;
- route U(d) blocks or orbit--stabilizer transport/stabilizer arrays;
- the explicit finite presentation and central targets;
- canonical words and Schreier transporters;
- stage-1 and joint matrix hashes;
- per-operation and joint correction metrics;
- pre-gauge and post-gauge certification;
- solver rank/conditioning/branch diagnostics;
- an artifact hash over metadata and arrays.

Dense matrices are derived compatibility views. Non-generator group elements
are generated from one canonical word and never independently exactified or
stored as competing primary matrices.

The existing factorized Q-phase/common-orbital-block artifact is certified
after joint exactification. It is an optional acceleration. Failure to certify
factorization must not invalidate a valid general route-block representation.

## Storage Policy

Remove the absolute-entry cleanup and its threshold-based tests. Construct a
dense compatibility matrix from an all-zero array and place the route blocks
on the exact support. This produces structural zeros exactly while preserving
every value inside a U(d) block.

Any `numerical_zero` label is diagnostic and must be based on a propagated
floating-point error bound. It must not mutate the matrix.

## Certification and Acceptance

The joint layer fails closed unless all of the following hold:

1. Every generator permutation is exact and all presentation relations close
   on the discrete action.
2. Fiber dimensions match along every route.
3. Stage-1 off-support pollution and unitarity are within their existing
   acceptance limits.
4. The central branch is unique with a recorded margin.
5. The solver converges with the expected rank/nullity and a safe relation-log
   branch margin.
6. The maximum and RMS correction are below explicit joint-correction limits.
7. Every power and mixed relation is below a dimension/word-length-aware
   floating-point certification bound.
8. Post-gauge certification also passes.
9. The projected-Hamiltonian covariance is not degraded beyond the propagated
   correction bound.
10. Reloading the stored artifact reproduces the same hashes and
    certifications.

The canonical artifact is structurally exact because its group table and
route reconstruction are unique. Materialized complex128 relation residuals
are expected at the computed floating-point multiplication floor, not at an
arbitrary fixed tolerance.

## Measured Regression Targets

### MgI2 M spin-up, d = 1

The route-phase problem reduces to a real constrained least-squares projection.
The measured nearest equal-weight correction is:

```text
TR RMS correction              4.3625209915e-6
C2 RMS correction              4.3625209915e-6
maximum route phase correction 4.4519406526e-6
final TR^2 residual             5.80e-17
final C2^2 residual             2.01e-16
final mixed residual            2.87e-16
```

### Idempotence cases

```text
MoTe2 K spin-up d=1:   current mixed residual 0
MoTe2 K spinful d=2:   current mixed residual 0
MgI2 Gamma d=2:        current mixed residual <= 4.7e-16
ZrS2 Gamma d=4:        current mixed residual <= 3.2e-16
MoTe2 AAB Gamma d=4:   current T/C3 residual about 9.0e-14
```

Already closed cases must not receive a material correction. The older MoTe2
AAB Gamma output should be reduced to its computed dense-arithmetic bound or
represented structurally through the canonical group artifact.

## Test Strategy

Local small-matrix tests cover:

- semilinear composition and inverse;
- gauge covariance for unitary and antiunitary routes;
- exact group presentation compilation and central elements;
- free-orbit U(1), U(2), and U(4) synchronization;
- nontrivial stabilizer compatibility and exactification;
- MgI2 44-dimensional regression;
- MoTe2 K d=1 and d=2 idempotence;
- ZrS2 Gamma d=4 gauge and stabilizer regression;
- structural-zero storage and preservation of legitimate sub-`1e-4` entries;
- pack/load/hash/re-certification;
- agreement with the direct small-matrix oracle.

Run targeted and default non-external pytest locally. Run full material
pipelines, external-data tests, and large model regressions by SSH on a bigmem
node, with outputs under `validation_runs/` only.

