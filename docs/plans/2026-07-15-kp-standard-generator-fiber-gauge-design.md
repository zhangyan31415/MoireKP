# KP Standard Generator Fiber Gauge Design

## Status

Approved on 2026-07-15 after reviewing the PtSe2 Gamma, MgI2 Gamma/M,
MoTe2 K/Gamma, and ZrS2 Gamma exact representations. This is a KP
continuum-basis convention. TAPW raw-H symmetry output is unchanged.

## Goal

After closest-cyclotomic U(1) exactification, use the remaining exact common
fiber gauge to put every supported unitary `C2` permutation cycle into the
nearest uniform standard form. Preserve all other generators exactly. In a
spinful two-cycle this turns route phases such as
`exp(i*pi/3), exp(i*2*pi/3)` into `+i,+i`, while a cycle near `-i` becomes
`-i,-i`.

## Separation of responsibilities

The existing closest-cyclotomic stage remains responsible for removing small
numerical phase drift and selecting exact roots closest to the projected
representation. The new standard-generator stage starts from that certified
exact representation. It changes no route support, no group relation, and no
physical observable.

For non-scalar `U(d)` fibers, the existing orbit--stabilizer joint
exactification remains authoritative. The standard-generator stage reports
`not_applicable` instead of applying entrywise or determinant-based rounding.

## Mathematical rule

For a scalar unitary route from source fiber `x` to target fiber `p(x)`,

```text
b'(x) = exp(-i theta[p(x)]) b(x) exp(i theta[x]).
```

For every permutation cycle `c=(x_0,...,x_{ell-1})` of `C2`, its holonomy

```text
h_c = product_r b(x_r)
```

is gauge invariant. A uniform cycle phase must obey `rho_c**ell = h_c`.
Among those roots, choose the one minimizing the total chordal distance from
the current exact route phases. For the expected two-cycles this selects the
nearest of `+i,-i` in spinful representations and the nearest of `+1,-1` in
spinless representations.

All non-`C2` generators are hard preservation constraints. Their scalar
route values must be unchanged after the new gauge, including the
semilinear transformation law for antiunitary operations. The common phase
system therefore fixes the balanced, minimum-norm solution. For PtSe2 this
gives

```text
W = diag(exp(-i*pi/12), exp(+i*pi/12))
```

on every paired sector, preserving `C3z` and `TR` while producing
`C2=i*sigma_x`.

## Automatic selection and safety

The stage is activated only when all fibers are scalar and the presentation
contains a unitary generator with family label `C2`. Fixed one-cycles are
already gauge invariant and are left unchanged. Every candidate must be
connected to the input representation by one certified common semilinear
fiber gauge. The final joint relations are recertified.

If there is no `C2`, as in current single-valley MoTe2 K models, the stage is
`not_applicable`. If an exact scalar representation has an incompatible
preservation system, production fails loudly rather than silently changing a
different generator.

## Artifact consistency

The standard-generator gauge is composed after the closest-cyclotomic gauge.
The composed full basis unitary is applied to Heff, projected wavefunctions,
spin operators, and every symmetry operation. The projection frame records
both stages and its hash changes with the composed gauge. `kp symm` rebuilds
and recertifies the joint artifact after applying the same frame.

## Expected material behavior

- PtSe2 Gamma: `C2` route phases become uniformly `+i`.
- MgI2 Gamma: already uniformly `+i`; identity correction.
- MgI2 M spinful: `exp(i*4*pi/3), exp(i*5*pi/3)` become uniformly `-i`.
- MgI2 M spinless: already uniformly `-1`; identity correction.
- MoTe2 K/Gamma without unitary `C2`: unchanged.
- ZrS2 Gamma: already uniformly `-i`; identity correction.

## Validation

Unit tests cover the PtSe2 and MgI2 M two-cycle branches, identity behavior,
absence of `C2`, preservation of unitary and antiunitary generators, and
non-scalar `U(d)` safety. Projection-frame tests prove that the two gauges are
composed and persisted. Material pipelines are then rerun on `bigmem002`,
followed by the release test suite.
