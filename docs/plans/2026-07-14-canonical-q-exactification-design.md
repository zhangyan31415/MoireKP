# Canonical Q Exactification Design

## Goal

Keep TAPW Hamiltonians, raw G/Q arrays, and basis ordering unchanged while making
the Q coordinates consumed by exact KP symmetry and `complete_linear_v2`
mathematically closed under the exactified finite-group action.

## Boundary

- TAPW remains the owner of the raw Hamiltonian basis and raw
  `g_vectors_group*.npy` files.
- KP projection continues to save the raw model-frame Q arrays associated with
  the projected Heff basis.
- `kp symm` owns canonicalization because it is the first stage that has both the
  explicit geometric action and the certified Q/sector permutations.
- `kp model` consumes canonical Q and exactified matrices from the same symmetry
  package. It must not reconstruct production Q independently for
  `complete_linear_v2`.

## Algorithm

Let `q_raw` contain all sector/Q coordinates in the existing basis order. For
every exactified generator, use its explicit Cartesian Q map and certified
sector/Q permutation to construct linear constraints

\[
q_{\pi_g(i)} = A_g q_i + t_{g,i}.
\]

Compute the nearest constrained coordinates

\[
q_{\rm canonical}=\arg\min_q\|q-q_{\rm raw}\|_2^2
\quad\text{subject to}\quad Cq=d.
\]

The implementation uses a real SVD/null-space projection, records the numerical
rank and backward error, and validates every generator after projection. It does
not branch on operation names. If no nontrivial geometric action is available,
canonical Q equals raw Q. If the constraints are inconsistent or the required
correction is larger than the versioned cleanup policy, the operation fails
closed.

## Artifact contract

The canonical `representations.npz` package stores the exactified matrices plus
reserved arrays for raw and canonical Q:

- `__q_model_raw_layer1__`
- `__q_model_raw_layer2__`
- `__q_model_canonical_layer1__`
- `__q_model_canonical_layer2__`

Metadata records the canonicalizer version, constraint rank, raw and canonical
closure residuals, RMS/max correction, tolerance policy, ordering, and raw and
canonical hashes. Reserved keys are not operation names.

Projection-side `q_model_layer*.npy` files retain their existing raw-Q meaning.
No TAPW or Heff arrays are rewritten.

For `complete_linear_v2`, a symmetry package without canonical Q is rejected
with an instruction to rerun `kp symm`. Legacy response semantics retain their
existing fallback.

## Error semantics

The post-canonical closure residual and constrained-solver backward error enter
response algebra certification. The raw-to-canonical correction is reported as
a source-geometry approximation, not mislabeled as floating-point roundoff.
This prevents analytically forbidden responses from being classified as robust
directions merely because raw Q coordinates were rounded.

## Validation

- Perturbed C3, reflection, negation, sector-exchange, and identity-only toys.
- Fail-closed tests for inconsistent permutations and excessive corrections.
- Symmetry-package round trip and hash sensitivity tests.
- ZrS2 Gamma q04: reduce the current raw C3 closure residual near
  `6.68e-12 1/Angstrom` to the certified numerical floor without reordering Q.
- MgI2 Gamma/M and MoTe2 K regression runs.
- Direct and frozen Hamiltonians must agree when both consume canonical Q.

