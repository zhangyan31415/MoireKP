# Release Numerical Contract Hardening Design

## Scope

This change hardens the release-facing numerical contracts shared by TAPW,
KP projection/symmetry/model workflows, and the exported standalone model.
It addresses:

- standalone Berry-curvature, quantum-geometry, and WCC coordinates;
- validated TAPW raw-H production symmetry matrices;
- cache and artifact identity;
- finite-basis TAPW boundary sewing for arbitrary source-group dimensions;
- KP basis/gauge consistency across project, symm, and model;
- projection k-point provenance;
- strict exactified-matrix provenance.

Experimental interlayer scaling/campaign code, examples, paper files, release
metadata, and broad complexity refactors are outside this change.

## Architecture

### Artifact identity

Machine-readable identity stays inside existing NPZ artifacts. No additional
JSON report is introduced.

TAPW artifacts carry:

- `identity_schema`;
- `input_hash` for H/S and structure inputs;
- `config_hash` for physics-relevant configuration;
- `basis_hash` for valley, spin, Q/G sets, internal row order, and dimension;
- package/schema version.

KP projection defines one `basis_hash` from source identity, Q sets,
low-energy selections, resolved gauge anchors, spin/mode, and row layout. The
same value is propagated through `basis.npz`, `wavefunctions.npz`, exactified
`representations.npz`, and `model_data.npz`. Consumers reject missing or
different production identities.

### TAPW raw-H production gate

Production `representations.npz` contains only internal operations that pass
the configured H covariance tolerance. For a nonorthogonal source problem,
the operation must also pass overlap covariance and overlap validity checks.
Unvalidated or failed operations may appear in human diagnostics but cannot be
loaded as production matrices.

The release path uses one strict validation behavior. Historical validation
aliases and silent `export_only` production are not accepted.

### Finite-basis TAPW boundary sewing

The boundary identification is built from the actual projected atomic Bloch
gauge,

\[
B_{\mathbf b}^{\mathrm{TAPW}}
= G\,\operatorname{diag}(e^{-i\mathbf b\cdot\mathbf r_a})\,G^\dagger,
\]

with the same spin and source-group row order used by the Hamiltonian. This
does not infer one common internal dimension from the total G-vector count and
therefore supports 1+1, 1+2, and 1+3 layouts.

The implementation reuses the periodic-gauge construction already used by
TAPW source symmetry. The topology workflow stores both reciprocal-boundary
operators and their identity metadata in one sparse NPZ artifact. WCC refuses
operators with incompatible dimensions, identity, covariance, or occupied-
subspace singular-value diagnostics.

The old partial G-block relabel remains diagnostic-only during migration and
is never selected silently by the canonical topology workflow.

### KP workflow integrity

`kp project` writes source `k_indices` and the KP `basis_hash`. `kp symm` must
reuse that basis identity. `kp model` checks projection and exactified symmetry
identities before loading matrices.

Model fit/band indices are source k-point indices. They are mapped through the
saved projection `k_indices`; a requested source point that was not projected
is an error.

An exactified operation is production-ready only when all provenance fields
agree: schema, owner, status, matrix kind, matrix source, basis hash, and
exactification report. Missing legacy metadata is rejected.

### Standalone topology

Standalone topology constructs Cartesian model momenta as

\[
\mathbf k=\kappa_1\mathbf b_{M1}+\kappa_2\mathbf b_{M2}.
\]

Band subspaces use column selection that preserves `(dim, n_band)` shape.
Finite-difference BC/QGT fields are computed in fractional coordinates and
transformed with the saved reciprocal basis before output. WCC boundary shifts
use the actual reciprocal vectors, never unit coordinate vectors.

## Failure behavior

The release workflow fails before publishing an artifact when:

- a production raw-H operation is unvalidated or exceeds tolerance;
- H/S, configuration, or basis identity differs from the cache;
- TAPW boundary operators do not match the saved wavefunction basis;
- a WCC link is rank-deficient below the configured numerical threshold;
- KP projection, symmetry, and model basis hashes differ;
- a requested model k point is absent from projection output;
- exactification provenance is incomplete or inconsistent;
- standalone topology lacks reciprocal-basis or basis metadata.

No production path falls back to naked boundary overlaps, inferred legacy
paths, object-dtype Heff arrays, or diagnostic symmetry matrices.

## Verification

Focused synthetic tests cover:

- a 54-dimensional model with a two-band projector;
- non-identity and oblique reciprocal bases;
- constant-projector zero BC/QGT and a known two-band Chern model;
- WCC shifts along both reciprocal directions;
- projected atomic Bloch sewing for equal and unequal source-group dimensions;
- spinful 1+1, 1+2, and 1+3 row layouts;
- failed H/S covariance exclusion from production NPZ;
- cache invalidation after changing H, S, structure, q-shell, spin, or config;
- project/symm/model basis-hash mismatch rejection;
- nonconsecutive projection k-index remapping;
- strict exactified-metadata rejection.

After focused tests, run the complete non-slow/non-external suite and real
MoTe2, MgI2, and multilayer smoke cases on the designated compute nodes.
