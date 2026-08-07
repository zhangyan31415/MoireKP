# KP Runtime Duplicate-Work Design

## Scope

Reduce complete `kp model` wall time without changing standalone export or fitted-model semantics.

The change has three independent parts:

1. Harmonic ablation computes one full candidate eigensystem and reuses its eigenvalues for every band window.
2. Complete-response caching computes the validated target-independent SHA-256 key once and passes that key through memory and persistent-cache operations.
3. Finite-degree response compilation gains a read-only closure/action-block diagnostic. It reports the real logical closure size, block dimensions, generator sparsity, and degree-lowering edges; it does not select a new production backend.

## Chosen design

Use surgical changes to existing functions. Do not add a prepared-identity framework, export schema, fallback backend, or generalized diagnostics subsystem.

For harmonic selection, `_band_refinement_metrics` accepts already computed eigenvalues. `_evaluate_harmonic_ablation_candidate` passes the eigenvalues returned by its existing `eigh`, preserving complete-spectrum `all_band_*` metrics exactly.

For caching, the persistent-cache API consumes the precomputed key

\[
K=\operatorname{SHA256}(\operatorname{canonical\_json}(\text{input record})).
\]

Payload validation and canonical serialization therefore happen once per model-basis lookup, while cache-file certification continues to compare the stored key and basis hash.

For finite degree, owners are vertices in the support graph of all generator actions and the Hermitian involution. If the connected components are \(C_\alpha\), the diagnostic reports

\[
L_{\rm closure}=\sum_\alpha |C_\alpha|,
\qquad
b_{\max}=\max_\alpha |C_\alpha|.
\]

Degree-lowering edges count nonzero action entries that map a source owner of degree \(d\) to a target owner of lower degree. Missing closure is reported, not repaired by an ambient fallback.

## Verification

- A focused test forbids `eigvalsh` during candidate evaluation when an eigensystem is already available.
- Cache tests count target-independent key construction and require one call on cold and persistent-hit paths.
- Diagnostic tests use a small exact action with known component sizes and lower-degree edges.
- Existing response-basis, cache, harmonic-selection, and compiler tests remain unchanged numerically.
- Full non-PtSe2 examples run on `bigmem001` and `bigmem003` after focused tests pass.
