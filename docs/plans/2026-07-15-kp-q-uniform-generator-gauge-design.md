# Q-Uniform Generator Gauge Design

## Problem

The general `U(d)` standard-generator stage currently canonicalizes each
symmetry fiber orbit independently.  This preserves the magnetic group
relations, but it can destroy a stronger structure already present in the
exactified input.  ZrS2 Gamma-q04 is the concrete failure: before the final
gauge, every `C3z: Q -> C3z Q` route uses the same internal four-dimensional
matrix.  The orbit gauge changes those routes into a Q-dependent mixture of
`+I4`, `-I4`, and two special fixed-Q blocks.  The result is equivalent as a
representation, but it is not the closest physically legible orbital gauge.

## Required invariant

When a finite unitary generator has one common internal route block over all
fibers of a sector-compatible family, its final routes must remain Q-uniform.
For `C3z`, choose one common internal unitary `U0` and apply it to every fiber:

```text
D_C3'(Q) = U0^dagger D_C3(Q) U0.
```

`U0` is the deterministic closest eigenframe of the common finite-order
unitary.  Eigenvalues are snapped to the declared cyclotomic roots and ordered
by their algebraic root exponent.  Repeated eigenspaces are completed by
projected coordinate vectors, so the frame does not depend on arbitrary LAPACK
eigenvectors.

The common frame is a hard structural choice.  Any later fiber gauge must lie
in the semilinear stabilizer of the Q-uniform diagonal `C3z` action.  It may
change phases and rotate exactly degenerate eigenspaces, but it must not turn a
common route into Q-dependent route matrices.

## Standardizing C2

After the common `C3z` frame is installed, `C2` is standardized only within the
remaining allowed gauge:

1. Split the internal space into exact `C3z` eigenspaces.
2. Use scalar phase balancing on one-dimensional eigenspaces.
3. Use the existing nearest algebraic monomial solver only inside repeated
   eigenspaces, with the same common frame transported over Q.
4. Preserve `C3z` and all declared antiunitary relations as hard constraints.
5. Among all feasible gauges, minimize the full Frobenius distance to the
   identity; resolve exact ties algebraically.

If the requested `C2` standard block is incompatible with the hard invariant,
the stage keeps the Q-uniform exact representation and reports that `C2`
standardization was not applicable.  It must never silently fall back to an
orbit-dependent `C3z` presentation.

## Alternatives considered

- Preserve the original dense common `C3z` block exactly.  This is closest to
  the source basis, but does not provide the requested `Jz`-like diagonal
  presentation.
- Retain independent orbit gauges with a soft penalty for changing `C3z`.
  This cannot guarantee Q-uniformity and reproduces the current failure mode.
- Use the Q-uniform hard-invariant design above.  This is selected because it
  gives a deterministic diagonal generator without hiding orbital action in Q
  phases.

## Artifact and certification

The projection-frame artifact records:

- the detected uniform generator and pre-frame route hash;
- the common internal unitary and its deterministic eigenframe certificate;
- the final shared diagonal cyclotomic spectrum;
- maximum route-to-common-block residual;
- the restricted `C2` standardization status;
- the usual common-gauge and magnetic-relation certifications.

All materialized nonzero entries retain monomial support and reduced
cyclotomic exponent/order records.  No magnitude threshold is used to delete
entries.

## Tests

The unit regression uses a ZrS2-like `U(4)` action whose `C3z` route block is
the same dense two-by-two-paired matrix on every Q.  It must fail under the old
orbit gauge and pass only when:

- every final `C3z` route block is bitwise identical;
- that common block is diagonal with spectrum
  `exp(-i*pi/3), -1, -1, exp(+i*pi/3)`;
- `C2`, TR, and all group relations remain certified;
- the production full-unitary frame reproduces every final operation.

Material validation reruns ZrS2 Gamma-q04 on bigmem001 and bigmem002 and
requires identical output hashes.  PtSe2, MgI2, and MoTe2 targeted regressions
ensure scalar and non-C3 paths do not change.
