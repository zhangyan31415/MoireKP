# Gamma Dual-Frame Model Gauge Design

## Goal

Keep automatic Gamma candidate selection and source-group routing certificates,
while restoring the deterministic SCDM model gauge that produced the reviewed
MgI2 continuum bands.  The routing frame is evidence about the selected
subspace; the model frame is the basis in which `Heff`, symmetry actions, spin,
response functions, and fitted Hamiltonians are represented.

The implementation is material independent.  MgI2 Gamma is the regression
case, not a special branch.

## Confirmed regression

The old auto-SCDM and downfolding core is unchanged between `847e32c` and the
current CPC head.  The behavior changed in two later steps:

- `a691c3c` constructed `authoritative_heff` directly from routed frames;
- `f5d0eed` routed automatic Gamma CLI requests through that producer.

The old and new frames span the same selected joint eigenspace but use different
coordinates.  For each sampled momentum and Q block,

```text
R = E V_route
M = E V_scdm
C = R^dagger M
```

where `R` is the routed frame, `M` is the model frame, and `C` is the
routing-to-model bridge.  MgI2 needs the full `U(4)` SCDM alignment.  The routed
frame instead enforces two independent `U(2)` source-group blocks.  Both are
valid bases, but only the former preserves the reviewed low-order model
representation.

## Non-goals

- Do not change candidate seeds, target windows, physical truncation orders,
  harmonic counts, or permitted term families.
- Do not add material names or MgI2-specific orbital indices to production
  code.
- Do not add fit points, nonlinear fitting, band refinement, or denser sampling.
- Do not require byte-identical NPZ files, coefficient vectors, or response
  ranks from the older implementation.
- Do not relax the fail-closed selection, identity, or symmetry certificates.

## Frame responsibilities

### Routing frame

The existing routed frame remains authoritative for:

- joint candidate closure;
- source-group ranks and offsets;
- routing gaps;
- source-group projectors;
- routed covariance and routing certificate hashes;
- automatic candidate identity and selection evidence.

No model-fit assumption is made about the routed frame.

### Model frame

The SCDM model frame is authoritative for:

- downfolded `authoritative_heff`;
- projected spin and wavefunctions;
- raw projected, exactified, and factorized symmetry actions;
- candidate symmetry certification;
- response-basis compilation;
- continuum Hamiltonian fitting and export.

Its columns use the existing `sector_orbital_q_v1` order.  Sector ownership is
a deterministic model-column semantic derived from group ranks and anchor
ordering; it is not a claim that every model column equals a routed source-group
projector.

## SCDM model-frame construction

For a candidate with ordered joint bands and routed ranks `(r0, r1)`:

1. Split the ordered joint bands into model owners of sizes `r0` and `r1`.
2. At the configured reference k and Q, reuse the existing Gamma QRCP/SCDM,
   spinful partner completion, and deterministic model-frame reference sorting.
3. Persist the resolved reference terms and their quality report in a canonical
   numeric/JSON contract.
4. At every production k and Q, align the complete joint eigenspace to those
   fixed reference vectors with the existing full-rank Procrustes operation.
5. Reject rank loss, nonfinite values, nonorthonormal frames, or a model frame
   whose joint projector differs from the routed joint projector.

The first correctness implementation may re-diagonalize only the reference
same-Q block to resolve anchors.  Production k/Q eigensystems already computed
by the automatic producer must be reused to materialize model frames.

## Bridge certificate

For every k and Q,

```text
C_r2m = R^dagger M
```

must satisfy

```text
C_r2m^dagger C_r2m = I
R C_r2m = M
R R^dagger = M M^dagger.
```

The tolerance comes from the existing routing projector tolerance.  The bridge
direction is fixed and included in its identity.  Model frames and bridges are
stored as compact local tensors:

```text
model_frames:       (Nk, Nq, same_q_dim, local_rank)
routing_to_model:   (Nk, Nq, local_rank, local_rank)
```

They are assembled into the sparse full-Q model ordering only on demand.  This
avoids persisting a large mostly-zero `(Nk, full_dim, model_dim)` tensor.

## Handoff v3

Keep `projection_basis_kind = "gamma_routed"` but bump the strict handoff schema
to v3.  Existing v2 automatic artifacts fail closed and must be regenerated.
Explicit legacy artifacts are unchanged.

The v3 handoff retains routed fields and adds:

- compact model-frame tensors and per-k/aggregate hashes;
- compact routing-to-model bridge tensors and hashes;
- a canonical SCDM anchor/model-column contract and hash;
- a bridge certificate hash;
- explicit routing/model accessors.

The ambiguous `assemble_for_k()` API is removed.  Consumers use:

```python
assemble_routing_for_k(k_index, include_high=...)
assemble_model_for_k(k_index)
routing_to_model_for_k(k_index)
model_state_for_k(k_index)
```

`model_state_for_k` returns a frame and the matching authoritative Heff
together, preventing accidental cross-frame pairing.

## Downfolding

The model low frame and routed low frame have the same projector.  For Schur or
Löwdin downfolding, the routed high frame spans the same orthogonal complement
and may be reused after explicit orthogonality/completeness checks.  Production
`Heff` is recomputed with the model low frame and the certified high complement;
it is not merely relabeled routed data.

The implementation also certifies the expected covariance

```text
H_model = C_full^dagger H_route C_full
```

within numerical tolerance.

## Certificate semantics

The routing certificate continues to bind routed projectors.  The candidate
symmetry certificate changes to bind model-frame `u_low` and model-frame
`authoritative_heff`.  Exactified/factorized actions are recomputed from the
model states rather than copied from the routed gauge.

The complete handoff identity binds:

- routed frame and routing evidence;
- model frame and SCDM anchor contract;
- bridge and bridge certificate;
- model-frame Heff;
- k/Q/layout/source Hamiltonian identities;
- candidate symmetry certificate.

Any tampering with one component invalidates load and selection publication.

## Consumer changes

- `gamma_auto_producer`: build both frames, downfold in the model frame, and
  certify model states.
- `gamma_auto_runtime`: project spin with the model frame and write model Heff.
- `symmetry/projection`: reconstruct persisted states with `model_state_for_k`;
  report both routing and model identities.
- `selection_artifact`: keep current selection semantics while binding the v3
  complete handoff hash.
- `model/pipeline`: continue consuming `heff.npy` and exactified actions; group
  ranks remain the source of `n_orb`, guarded by the model-column contract.

## Verification

### Synthetic gates

- A k-dependent mixing toy must produce distinct routing and model frames with
  identical joint projectors.
- Bridge unitarity, reconstruction, projector equality, and Heff covariance
  residuals must be at most `1e-10`.
- Handoff roundtrip must preserve both frames and reject tampering with routed
  frames, model frames, bridge, anchor contract, or Heff.
- Runtime spin and kp symmetry tests must distinguish the two frames and prove
  that production consumers use the model frame.

### MgI2 physical acceptance

Use the automatic q04 probe with the reviewed two-point linear fit:

```text
fit indices:          [0, 40]
fit method:           linear
fit bands:            top 10
one/two-sided weight: 0
nonlinear/refinement: disabled
```

Acceptance limits:

- selected joint bands `(40, 41, 42, 43)`;
- routed group ranks `(2, 2)` and model dimension 76;
- Top-10 top-aligned RMS at most `2.31 meV`;
- Top-10 top-aligned maximum at most `4.10 meV`;
- projection spectra agree with the routed projection within `1e-9 eV`;
- Hermiticity and frame/bridge residuals at most `1e-10`.

Run the full project/model acceptance independently on `bigmem001` and
`bigmem003`, using separate output directories.  Never use `bigmem002`.
