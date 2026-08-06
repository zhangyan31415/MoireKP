# CdI2 q04 Parameters on q05 Q Basis — Design

## Objective

Evaluate the fitted CdI2 AB 7.34° q04 continuum model on the q05 Q basis without fitting to any q05 TAPW data. The experiment isolates the effect of enlarging the Q-shell from the effect of changing model parameters.

## Physical definition

The transferable model is the semantic continuum term table

\[
(p,M_z,M_z^*,l,l',a,a')\mapsto(r_{\rm real},r_{\rm imag}).
\]

The q04 response-basis arrays are only a compiled representation of these terms. They are not the transfer key. For every q05 basis pair \(Q,Q'\), the lifted model uses the q04 term whose momentum transfer satisfies the term support rule, with the same monomial orders, layer pair, orbital pair, and fitted real/imaginary values.

## Selected approach

1. Extract the q04 fitted coefficient values together with each channel's semantic `term_key` and real/imaginary component.
2. Reconstruct the physical q04 term table using a canonical key containing rounded `p`, `Mz`, `Mz_star`, layers, and orbitals.
3. Build each term's polynomial matrix directly on the q05 Q layout. Do not use the independently fitted q05 coefficient vector.
4. Sum the term matrices to obtain the 248-dimensional q05 Hamiltonian.
5. Evaluate top7–8 on the same eight outer radii used by the dense TAPW q05 calculation.
6. Apply the validated q05 C3 and reciprocal-boundary sewing conventions at the exact outer boundary.

This preserves the physical q04 parameters while allowing the larger q05 basis to create all additional matrix elements implied by the same momentum-transfer rules.

## Rejected approaches

- **Response-channel index mapping:** q04 and q05 response bases have different hashes and dimensions. Their array indices are compiler coordinates, not physical parameter identities.
- **Zero-padding the 152-dimensional Hamiltonian:** this would decouple the additional q05 Q states and would not represent a continuum-model Q-shell enlargement.
- **Refitting q05 coefficients:** this would mix parameter changes with Q-shell changes and defeat the purpose of the experiment.

## Data flow

The q04 `model_data.npz` supplies fitted coefficients and semantic channel metadata. The q05 TAPW artifacts supply only the q05 Q vectors and symmetry/sewing representations. No q05 energies or wavefunctions enter the lifted Hamiltonian.

The output records the source hashes, all transferred term keys and values, Q-layout information, common-block checks, Wilson phases, internal-overlap diagnostics, and sewing singular values.

## Validation

- Every nonzero q04 fitted coefficient must map to exactly one semantic physical key and component.
- Rebuilding the q04 Hamiltonian from the extracted term table must reproduce the exported q04 evaluator at representative k points within numerical tolerance.
- Restricting the q05 lift to Q pairs supported inside the q04 set must reproduce the same term-level matrix elements after basis reordering.
- The lift must not read q05 TAPW band energies or wavefunctions.
- C3 covariance and K/K′ reciprocal sewing must pass numerical residual and singular-value checks.
- The final plot must distinguish TAPW q05, lifted q04-parameter/q05-basis model, and the original q04-basis model.

## Failure handling

Abort on duplicate semantic keys with conflicting values, missing Q-layout metadata, ambiguous Q-vector matching, a failed common-block check, non-Hermitian Hamiltonians beyond tolerance, or poor sewing rank. Do not silently set unmatched physical terms or refit coefficients.
