# Gamma Representative-Point Selection Design

**Goal:** Restore fast automatic Gamma projection by selecting the low-energy branch at one representative `(k_Gamma, Q0)` fibre and materializing the production basis exactly once.

## Data flow

1. Resolve the expansion-origin k row and the unique minimum-norm Q row.
2. Form complete edge-energy clusters only in that joint Gamma fibre.
3. Evaluate cluster growth and common-anchor rank locally; stop at the first locally adequate branch.
4. Freeze its joint band indices and reference-orbital pattern.
5. Build all production k/Q frames, Heff rows, handoff data, and symmetry evidence once for the frozen branch.
6. Run band and symmetry validation on the resulting authoritative handoff; validation does not reopen branch selection.

## Non-goals

- No per-candidate production k/Q materialization.
- No strict source-group routing gate.
- No changes to model-order search, response-basis compilation, K/M selection, or physical truncation settings.

## Regression contract

- Public automatic Gamma selection invokes the production candidate materializer once.
- A typed local anchor/rank rejection advances to the next complete cluster without materializing the rejected branch.
- Explicit/legacy producer policies retain their certified behavior.
- The AAB Gamma representative fibre resolves the four-band branch before the single production build.
