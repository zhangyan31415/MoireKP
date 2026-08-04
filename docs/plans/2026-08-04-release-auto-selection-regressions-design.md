# Release Auto-Selection Regression Design

**Goal:** Recover the audited compact Gamma and partial-sector K/M projectors
without restoring hand-authored band lists or weakening structural and symmetry
certification.

## Evidence

The current representative-point Gamma policy uses local band thresholds of
`1/3 meV` (RMS/max).  For MgI2 Gamma the complete cluster curve is:

| local rank | joint bands | RMS/max (meV) | anchor coverage |
|---:|---|---:|---:|
| 2 | 42-43 | 27.474 / 51.022 | 0.323 |
| 4 | 40-43 | 1.975 / 2.861 | 0.462 |
| 8 | 36-43 | 0.412 / 0.596 | 1.35e-13 |
| 12 | 32-43 | 0.158 / 0.226 | 0.257 |

The audited four-dimensional fibre is rejected only by the unreasonably tight
1 meV RMS gate.  The next nominally accurate candidate is anchor-singular, so
the selector reaches rank 12 and expands the production model from 76 to 228.

For AAB K-B, every candidate is measured against the full-system target even
though the intended model keeps only the B source sector.  The audited
one-dimensional fibre gives a good continuum fit to its authoritative Heff,
but the unrelated full-system branches leave a nearly dimension-independent
preselection error near `14.46/41.93 meV`.  That error cannot certify or reject
a partial-sector model.

## Selection policy

### Gamma

- Keep the actual `(k_Gamma, Q0)` complete-cluster envelope.
- Keep anchor coverage as a hard gate.
- Use the existing generic automatic-selection defaults, `10/30 meV`
  (RMS/max), for representative-fibre band adequacy.
- Select the smallest complete cluster passing those gates, then materialize
  the full production k/Q handoff exactly once.
- Do not add per-material threshold parameters to release YAML files.

This selects MgI2 rank 4 while continuing to reject rank 2.  It also preserves
the compact PtSe2 rank-2 result.

### Partial-sector K/M

- Detect a partial-sector envelope from physical-layer rows that remain empty
  in every generated candidate.
- For partial-sector envelopes only, retain full-system band RMS/max as
  diagnostic metrics, not hard gates.
- Select the smallest structurally valid candidate with adequate anchor
  coverage; band metrics remain available for reporting and tie-breaking.
- Keep band gates unchanged when every physical sector is active.
- Keep post-materialization symmetry certification hard in both cases.

The policy identity records that partial-sector band metrics are diagnostic so
the same inputs cannot be silently certified under different semantics.

## Performance

Candidate scoring remains representative-point only.  No candidate may build
all production k/Q frames.  Production materialization occurs once after the
compact candidate is frozen.  This prevents an incorrect rank increase from
amplifying response-space compilation cost.

## Regression contract

- MgI2 Gamma selects joint bands 40-43 and model dimension 76.
- PtSe2 Gamma remains joint bands 54-55 and model dimension 38.
- AAB K-B automatic selection resolves the one-dimensional B-sector fibre.
- AAB K-A, bilayer MoTe2 K, and MgI2 M keep their audited automatic choices.
- Selection artifacts retain raw band metrics and record whether those metrics
  were hard or diagnostic.
- All structural, anchor, handoff identity, and symmetry gates remain active.
