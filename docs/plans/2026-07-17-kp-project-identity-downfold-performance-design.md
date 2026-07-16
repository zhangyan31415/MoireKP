# KP Project Identity and Downfold Performance Design

## Scope

Remove the full-file Hamiltonian content verification from `kp project`, then optimize
the remaining projection cost without changing the effective Hamiltonian within numerical
precision. The reference workload is spinful MgI2 Gamma q05 with 31 Q points, a
`(61, 5704, 5704)` Hamiltonian stack, and a 124-dimensional continuum space.

## Source identity

KP will no longer read or hash `hamiltonian_k.npy` while building projection metadata.
The projection identity will continue to bind the inexpensive structural inputs that are
needed by downstream KP stages: Q sets, projection configuration, basis layout, selected
k indices, Heff, and projected k points. The projection-basis schema version will change
because `input_hash` no longer certifies Hamiltonian content.

This deliberately separates two concerns:

- no source-Hamiltonian content verification;
- continued internal agreement between projection, symmetry, and model artifacts.

No fake digest or path-only digest will be written.

## Dense downfold investigation

The high-space matrix is Hermitian and is represented in a basis assembled from local
Q-block eigenvectors. Before changing the solver, the MgI2 probe will measure:

- Hermiticity residual and shifted-matrix inertia;
- distance from `E_ref` to the high-space spectrum and spectral condition estimate;
- diagonal-block versus off-diagonal-block norms;
- numerical ranks of representative inter-block couplings;
- exact-solver timings and residuals;
- preconditioned iterative-solver convergence for all low-space right-hand sides.

Candidate implementations are evaluated in this order:

1. Hermitian-indefinite dense factorization, retaining an exact direct solve.
2. Preconditioned Hermitian iteration using the local-Q block structure.
3. Approximate truncation or compression, diagnostics only.

Production adoption requires max Heff and eigenvalue differences at or below `1e-10 eV`,
no degraded Hermiticity, and at least a 15% speedup on the isolated numerical kernel.

## Auto-gauge reuse

Auto-gauge candidates span the same configured low-energy subspace. Candidate-specific
anchors change the basis frame, not the physical projected subspace. The optimization will
first establish a reusable reference projection state, then derive candidate states through
low-dimensional unitary transformations. It must preserve candidate rankings, selected
anchors, exactification diagnostics, and the final projected spectrum.

If an exact reuse transformation cannot be certified for a case, that case retains the
existing full candidate evaluation.

## Validation

- Unit tests cover the no-Hamiltonian-hash contract and exact downfold equivalence.
- Targeted KP tests cover projection identity, block projector downfold, auto gauge, and
  symmetry projection.
- The non-slow release test suite is run before completion.
- Bigmem validation records wall time, numerical differences, solver residuals, and matrix
  property reports under `validation_runs/` only.
