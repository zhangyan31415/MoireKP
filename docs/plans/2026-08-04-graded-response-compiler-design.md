# Graded case-derived response compiler design

## Goal

Keep the case-derived, profile-free physical candidate envelope while avoiding
the current generate-all-then-rank workflow.  A low-order probe identifies the
actual sector/orbital/harmonic action, but the configured kinetic, intra, and
inter maximum orders remain authoritative and are reproduced exactly.

For the current MgI2 Gamma q04 single-sign envelope the compiler should derive
40 structural supports rather than materializing 2,032 complex polynomial
seeds and 4,064 real directions.  This count is a case diagnostic, never a
hard-coded material constant.
It must preserve the invariant response span, Hermitian/antiunitary semantics,
Hamiltonian residual, bands, and orbital-gauge covariance.

## Non-goals

- Do not restore the legacy Gamma 2x2 profile.
- Do not infer or reduce a user-specified maximum momentum order.
- Do not assume that behavior through degree two can be copied blindly to all
  higher degrees.
- Do not expose a new public tuning parameter.
- Do not use a shipped cache to hide cold-compile cost.

## Structural envelope

A structural support contains only its physical identity:

- source and target sectors;
- an actual Q-pair/harmonic support;
- ordered source and target orbital slots;
- adjoint provenance.

`kinetic`, `onsite`, `intra`, and `inter` remain provenance and determine which
configured order bound authored a degree; they are not part of structural
identity.  Consequently exact physical duplicates such as

\[
\text{onsite}=\text{intra}(p=0,d=0),\qquad
\text{kinetic}(d)=\text{intra}(p=0,d)
\]

are represented once while retaining both provenance records and their
original family-specific truncation rules.

It does not contain a momentum monomial.  The current MgI2 Gamma q04 case
derives:

\[
N_K^{\rm struct}=2\cdot4=8,\qquad
N_{\rm intra}^{\rm struct}=2\cdot2\cdot4=16,\qquad
N_{\rm inter}^{\rm struct}=2\cdot2\cdot4=16.
\]

Thus only 40 dynamically discovered structural supports enter action analysis
before symmetry-orbit merging.

The actual exactified factorized or joint-route action determines how these
supports and their Q-pair amplitude fibers mix.  No operation-name geometry or
fixed orbital pair is authored.

## Graded momentum lift

For homogeneous total degree \(d\), use

\[
\mathcal P_d=\operatorname{span}
\{w^d,w^{d-1}\bar w,\ldots,\bar w^d\},\qquad
\dim\mathcal P_d=d+1.
\]

Construct the exact momentum representation \(M_g^{(d)}\) from the certified
Cartesian action.  For each closed structural orbit \(o\), construct its small
internal/support action \(A_{g,o}\).  The degree-\(d\) response action is

\[
U_{g,o}^{(d)}=M_g^{(d)}\otimes A_{g,o},
\]

with the existing realification for antiunitary operations and the existing
Hermitian-adjoint action.  Solve the invariant space separately for every
\((o,d)\) block, then take their direct sum.

Degree two is only a cheap structural/action probe.  Every configured degree is
constructed analytically.  This is required because, for example,

\[
\operatorname{Im}(w^3)=\frac{w^3-\bar w^3}{2i}
\]

is C3-even and C2-odd and first appears at degree three.  It can pair with a
C2-odd internal response to form an allowed invariant that no degree-two scan
can observe.

Q-centered polynomial seeds are related to the global homogeneous coordinate
by an exact block-triangular binomial translation.  Apply that translation only
after the graded fixed spaces are known.  It may mix degree \(d\) into lower
degrees, but it must not force one global rank problem.

## Materialization and rank certification

The compiler proceeds as follows:

1. Build the complete structural envelope from the actual sectors, Q-pair
   support, harmonics, orbital dimensions, and configured orders.
2. Close structural supports under the actual symmetry actions and adjoint.
3. Form small structural orbits and certify that their direct sum covers the
   authored envelope.
4. Build \(M_g^{(d)}\) for every required degree.
5. Solve and certify each small \((o,d)\) invariant block.
6. Apply exact Q-center translation and materialize only retained response
   channels.
7. Run the existing target-independent final dependency check as a cheap
   certificate, not as the primary reducer.

The current full symbolic-atom compiler remains a test oracle.  Production may
fall back to it only when factorization or structural-orbit certification is
genuinely unavailable.  Authored seed-label non-closure is not a fallback
condition because symbolic/structural closure is the purpose of this compiler.

## Cache and reporting

The cache identity includes:

- structural-envelope hash;
- ordered structural-orbit certificate;
- exact action hashes;
- degree representation hashes;
- configured maximum orders;
- coordinate and Q-center translation hashes.

Progress output reports structural support count, orbit count, per-degree fixed
rank, materialized channel count, and timings.  A fallback states its precise
failed certificate.

## Validation

Unit and oracle tests must cover:

1. the C3-even/C2-odd cubic channel, proving that the low-order probe does not
   truncate higher-degree irreducible content;
2. factorized and joint-route actions spanning the same certified projector;
3. Q-dependent phases and a seed vocabulary not closed by labels;
4. antiunitary and Hermitian closure;
5. equality with the current dense/symbolic oracle on random toy cases;
6. the L2 orbital-swap gauge-covariance regression.

Production checks use cold caches and compare:

- response rank and span projector;
- fitted Hamiltonian residual;
- bands and overlap diagnostics;
- symmetry covariance and Hermiticity;
- cold and warm runtime.

The first material target is MgI2 Gamma q04.  Then run ZrS2 Gamma, AAB Gamma,
PtSe2 Gamma, and MgI2 M spinful.  The target for MgI2 Gamma q04 is a cold
`kp model` time no greater than 120 seconds without changing its configured
orders or harmonic selection.

## Integration order

1. Add oracle tests and the cubic counterexample.
2. Remove the uncommitted seed-label-closure backend switch once projector
   equivalence is covered.
3. Add structural-envelope/action objects.
4. Add graded momentum lift and per-block fixed-space solving.
5. Integrate exact triangular translation and retained-channel materialization.
6. Compare against the oracle and profile MgI2 Gamma cold.
7. Run the targeted material matrix and retain the old compiler only as a
   certified fallback/oracle.
