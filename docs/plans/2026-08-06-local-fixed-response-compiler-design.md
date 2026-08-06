# Local Generator-Fixed Response Compiler Design

**Date:** 2026-08-06
**Branch:** `codex/cpc-release-clean`
**Status:** Approved

## Goal

Reduce the certified cold response-compilation time for the ZrS2 Gamma q04
spinful release case from approximately 75.3 seconds to at most 20 seconds on
both `bigmem001` and `bigmem003`, without changing the physical fixed response
space or any release-case model result.

The optimization targets avoidable mathematical work. It does not attempt to
obtain the required speedup by tuning COO/CSR constants.

## Current pipeline and measured cost

The current graded compiler follows this data flow:

```text
authored responses
  -> physical-seed deduplication
  -> structural overlap components
  -> Reynolds projection of every real/imaginary seed
  -> global degree-filtered sparse rank selection
  -> selected-channel materialization
```

For ZrS2 group 0, the measured reduction is

\[
1952\ \text{authored}
\longrightarrow1760\ \text{physical seeds}
\longrightarrow3520\ \text{real projected columns}
\longrightarrow147\ \text{retained columns}.
\]

Thus only

\[
\frac{147}{3520}=4.18\%
\]

of the projected columns survive. The projected intermediates contain about
59,689,340 nonzeros. The largest degree-eight batch has shape

\[
2{,}079{,}360\times576
\]

and contains 18,585,357 nonzeros.

The measured group-0 costs are approximately

\[
T_{\rm structural}=8.41\ \mathrm{s},\qquad
T_{\rm Reynolds}=35.34\ \mathrm{s},
\]

\[
T_{\rm global\ rank}=16.73\ \mathrm{s},\qquad
T_{\rm materialize}=1.97\ \mathrm{s}.
\]

The primary redundant cost is therefore

\[
T_{\rm Reynolds}+T_{\rm global\ rank}=52.07\ \mathrm{s}.
\]

## Constraints discovered from the production artifacts

The certified symmetry artifacts already expose factorized and block-route
actions and their hashes, unitarity bounds, route-leakage bounds, and group
relation certificates. The fast compiler must consume those certificates; it
must not rediscover route structure by thresholding a dense matrix.

The release cases are not uniformly scalar-monomial:

- MgI2, AAB, and PtSe2 reduce to permutation-times-phase fibers.
- ZrS2 has monomial C2 and time-reversal routes, but its C3 orbital action
  contains exact dense two-by-two route blocks.

Consequently, a scalar gain graph cannot be the only production algorithm.
The first implementation uses a generic local generator-nullspace solver.
A scalar/block gain-graph backend remains an optional later optimization.

The existing `StructuralOrbit` is an overlap-connected component of cyclic
images, not a discrete permutation orbit. Selecting one seed per current
orbit is therefore not correct.

## Proposed data structures

The graded compiler will build internal local problems after structural
planning and before the current Reynolds loop.

```text
LocalVocabularyComponent
  component_index
  nominal_degrees
  logical_owner_indices
  provenance_by_owner
  ambient_columns V_c              # sparse raw columns with full tails
  gram G_c = V_c^T V_c
  generator_actions[name]
  generator_error_bounds[name]
  antiunitary_parities[name]
  hermitian_action
  exact_support_rows
```

```text
LocalFixedCompilation
  component_index
  fixed_vocabulary_coordinates
  logical_projection_coordinates
  fixed_rank
  selected_global_owner_indices
  singular_values
  gray_zone_certificate
  generator_residuals
  hermitian_residual
  propagated_error_bounds
  fallback_metadata
```

Logical response IDs, term indices, harmonic provenance, adjoint provenance,
and original owner order remain attached to the local coordinates. The public
`CandidateResponseSet` interface does not change.

## Component construction

Components must include every dependency relevant to the physical fixed
space. Starting from the current structural proposals, union-find merges
columns along

\[
E=E_{\rm generator}\cup E_{J_H}\cup E_{\rm ambient\ support},
\]

where generator edges are nonzero local action blocks, Hermitian edges connect
adjoint partners, and ambient-support edges connect raw columns whose complete
global coefficient vectors share exact rows.

This protects the known lower-degree-tail case. For an affine origin shift,

\[
D_gV_d\subseteq V_d\oplus V_{d-1}\oplus\cdots\oplus V_0,
\]

so top-degree structural components cannot in general be solved independently.
The generator action is formed in centered authored coordinates, while the
complete global columns are retained for Gram, dependency, certification, and
materialization operations.

After local fixed spaces are found, the existing global degree-filtered tail
certificate is applied to the much smaller set of fixed outputs. This is a
second guard against cross-component lower-degree dependencies.

## Local generator action

Let

\[
V_c=[v_1,\ldots,v_{n_c}],\qquad G_c=V_c^\mathsf TV_c.
\]

At polynomial degree \(d\), the local complex action has the factorization

\[
A_{g,c,d}=P_{g,d}\otimes S_{g,c},
\]

where \(P_{g,d}\) is the homogeneous momentum action and \(S_{g,c}\) is the
certified local orbital, sector, and Q-fiber action. More general certified
block actions use the same local interface.

For \(z=x+iy\), a unitary action \(z\mapsto Az\) is realified as

\[
\mathcal R_U(A)=
\begin{bmatrix}
\Re A&-\Im A\\
\Im A& \Re A
\end{bmatrix},
\]

and an antiunitary action \(z\mapsto A\bar z\) as

\[
\mathcal R_A(A)=
\begin{bmatrix}
\Re A& \Im A\\
\Im A&-\Re A
\end{bmatrix}.
\]

Hermitian adjunction is another anti-linear local action,

\[
J_H:(r,s,i,j)\mapsto(s,r,j,i),
\]

including harmonic-adjoint provenance.

Dependent vocabulary columns are removed locally with an error-certified,
deterministic rank decision before the fixed solver is called. If
\(G_c=L_cL_c^\mathsf T\), orthonormal local coordinates are

\[
y=L_c^\mathsf Tz,
\]

and a generator becomes

\[
\widehat A_{g,c}=L_c^\mathsf TA_{g,c}L_c^{-\mathsf T}.
\]

The fixed space is then

\[
K_c=
\begin{bmatrix}
\widehat{\mathcal R}_{g_1,c}-I\\
\vdots\\
\widehat{\mathcal R}_{g_q,c}-I\\
\widehat{\mathcal R}_{H,c}-I
\end{bmatrix},
\qquad
F_c=\ker K_c.
\]

Only group generators are required; the compiler no longer applies every
finite-group element to every seed.

## Owner selection and materialization

The nullspace basis is not exposed as a new anonymous public basis. Its
coordinates are projected back onto the original logical owners. A
deterministic small pivoted factorization selects original owner IDs, with tie
cases treated as certification ambiguity rather than resolved by incidental
LAPACK ordering.

Only the fixed outputs that survive the small global tail certificate are
lifted to ambient response columns and materialized. The expected main-path
column count is therefore proportional to the physical fixed rank, not to all
3520 projected real seeds.

## Certification

Correctness is defined by the physical fixed subspace. For an orthonormal
physical basis \(W\), let

\[
P=WW^\mathsf T.
\]

The fast and Reynolds paths must satisfy

\[
r_{\rm fast}=r_{\rm oracle},
\]

\[
\|P_{\rm fast}-P_{\rm oracle}\|_2
=\sin\theta_{\max}
\le\max(10^{-10},10b_{\rm cert}).
\]

The certification threshold is derived from propagated numerical error:

\[
\gamma_n=\frac{n\epsilon}{1-n\epsilon},\qquad
\delta_K=\sqrt{\sum_s e_s^2},
\]

\[
\tau_K=\delta_K+\gamma_n\|K\|_2+10^{-12}.
\]

Singular values are classified by

\[
\sigma_i\le\tau_K\Rightarrow\text{fixed},
\]

\[
\sigma_i\ge100\tau_K\Rightarrow\text{nonfixed}.
\]

Any value satisfying

\[
\tau_K<\sigma_i<100\tau_K
\]

is ambiguous and makes the fast path unavailable.

The artifact records fixed rank, boundary singular values, thresholds,
principal angles to an enabled oracle, generator and full-group residuals,
antiunitary parity, Hermiticity residuals, Q-center translation residuals,
adjoint provenance bounds, deterministic ordering metadata, and all propagated
error terms.

The cache identity includes the compiler algorithm and convention versions,
absolute and relative tolerances, gray-zone factor, scaling policy, component
policy, materialization policy, generator parity, Q center, and harmonic
provenance.

## Fallback and failure behavior

The current graded Reynolds implementation remains available as a typed
fallback and as the test oracle. A dedicated
`LocalGeneratorNullspaceUnavailable` is raised for expected unsupported or
uncertifiable cases, including:

- missing or inconsistent certified factorized/block actions;
- local action or Hermitian image escaping the constructed vocabulary;
- component leakage above its certified bound;
- vocabulary-rank or fixed-nullity gray zones;
- component size above the initial dense-solver cutoff;
- failed group, antiunitary, Hermiticity, adjoint, or Q-center residuals;
- non-deterministic owner pivots without a certified gap.

Only this typed condition invokes Reynolds. Unexpected programming errors
propagate. A fallback result that is itself ambiguous fails closed and is not
cached.

The initial dense local-nullspace cutoff is 512 real dimensions. Production
component sizes are recorded before this value is finalized. Components above
the cutoff use Reynolds until a block gain-graph backend is implemented.

## Optional block gain-graph backend

If the generic local solver does not meet the performance goal, the same local
problem/result interfaces support a block gain graph. Its vertices are local
fiber-pair/monomial spaces and its semilinear edge transports are constructed
from certified route blocks. A spanning tree eliminates vertex variables, and
only cycle constraints are solved at a root.

For component \(c\), the reduced system is

\[
C_c\xi=0,
\]

where each chord contributes the difference between its direct transport and
the spanning-tree transport. Scalar release cases have local edge dimension
one; ZrS2 has block size at most two and local matrix dimension at most four.

This backend is deferred until measured component sizes or timings demonstrate
that it is necessary.

## Test and benchmark protocol

All tests and benchmarks run only on `bigmem001` and `bigmem003`. Both hosts use
the same Python environment, eight pinned CPU cores, and fixed BLAS thread
counts. Login-node work is limited to source inspection and editing.

A cold run uses a new Python process and a new empty persistent-cache directory.
A warm run uses a new Python process and reuses the preceding cold directory;
it does not measure an in-process dictionary hit.

For each host \(h\), run three cold samples and define

\[
C_h=\operatorname{median}(C_{h,1},C_{h,2},C_{h,3}).
\]

The primary performance gate is

\[
\boxed{\max(C_{001},C_{003})\le20\ \mathrm{s}}.
\]

This corresponds to a minimum speedup of

\[
\frac{75.3}{20}=3.765.
\]

The warm-cache gate is

\[
\max(W_{001},W_{003})\le5\ \mathrm{s}.
\]

The median absolute deviation on each host and the cross-host median difference
must each be at most ten percent. Memory may not exceed the larger of 110% of
the protocol-matched baseline and the baseline plus 256 MiB.

Correctness testing proceeds from focused unit and oracle tests to complete
response-basis tests, the full non-external test suite, the release gate, and
all ten release pipelines. Every release pipeline must return zero for
`project -> symm -> model`, retain `CERTIFIED` selection state, and preserve the
physical response projector, Hamiltonian, fitted bands, RMS error, and maximum
error within their certified tolerances.

Redundant channel count is not treated as the physical invariant. Stable
logical IDs and canonical owner order are preserved whenever their pivot gaps
are certifiable; any intentional basis-frame change must still pass projector
and physical-output equivalence.

## Expected result

The first-stage budget is

\[
T_{\rm cold}\approx9.5\text{--}17\ \mathrm{s},
\]

provided the exact joint components stay below the local dense-nullspace
cutoff. The first production benchmark records component dimensions; a large
component or a missed 20-second target triggers evaluation of the certified
block gain-graph backend rather than sparse-format micro-optimization.
