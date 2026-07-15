# KP Joint U(d) Block Exactification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a KP-owned, gauge-covariant joint exactifier that converts the independently exactified continuum generators into one structurally exact magnetic block-monomial representation with general route-dependent U(d) blocks.

**Architecture:** Preserve the existing support, polar, power-root, and algebraic branch selection as stage 1. Compile an explicit magnetic presentation, decompose its exact fiber action into orbits and stabilizers, synchronize free-orbit transporters, exactify only the small nontrivial stabilizer corepresentations, and reconstruct all route blocks from one canonical group table. Apply the existing symmetry-adapted gauge only after joint closure, re-certify after the gauge, and store structural zeros without magnitude thresholding.

**Tech Stack:** Python 3.11, NumPy, SciPy (`scipy.linalg.expm`, rank-revealing SVD/lstsq), pytest, existing KP symmetry/action metadata, NPZ/JSON artifacts.

---

## Working Rules

- Follow @test-driven-development for every implementation task.
- Use @systematic-debugging for every unexpected failure.
- Use @verification-before-completion before claiming the feature complete.
- Preserve all unrelated dirty files. Stage only the paths named by each task.
- Run small matrix tests locally.
- Run the complete non-slow suite and all real material pipelines with
  `ssh bigmem003`; do not run those workloads locally.
- Keep all real-run output under
  `validation_runs/joint_ud_exactification_20260715/` and do not commit it.

### Task 1: Add the semilinear algebra and immutable route-action model

**Files:**

- Create: `kp/kp/symmetry/joint_exactification.py`
- Create: `tests/kp/test_joint_exactification.py`

**Step 1: Write the failing semilinear composition tests**

Add tests equivalent to:

```python
import numpy as np

from kp.symmetry.joint_exactification import (
    SemilinearBlock,
    compose_semilinear,
    inverse_semilinear,
)


def test_semilinear_composition_conjugates_only_the_inner_block() -> None:
    outer = SemilinearBlock(np.array([[1.0j]]), antiunitary=True)
    inner = SemilinearBlock(np.array([[np.exp(0.3j)]]), antiunitary=False)
    product = compose_semilinear(outer, inner)
    np.testing.assert_allclose(product.matrix, [[1.0j * np.exp(-0.3j)]])
    assert product.antiunitary


def test_semilinear_inverse_is_a_two_sided_inverse() -> None:
    unitary = np.array([[0.0, 1.0j], [1.0, 0.0]], dtype=np.complex128)
    value = SemilinearBlock(unitary, antiunitary=True)
    inverse = inverse_semilinear(value)
    identity = np.eye(2, dtype=np.complex128)
    np.testing.assert_allclose(compose_semilinear(value, inverse).matrix, identity)
    np.testing.assert_allclose(compose_semilinear(inverse, value).matrix, identity)
```

Also add validation tests for non-square, non-finite, and nonunitary blocks.

**Step 2: Run the focused test and confirm failure**

Run:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_joint_exactification.py -k semilinear
```

Expected: collection fails because `kp.symmetry.joint_exactification` does not
exist.

**Step 3: Implement the minimal semilinear core**

Implement immutable dataclasses and functions with these signatures:

```python
@dataclass(frozen=True)
class SemilinearBlock:
    matrix: np.ndarray
    antiunitary: bool


def semilinear_kappa(matrix: np.ndarray, antiunitary: bool) -> np.ndarray:
    return matrix.conj() if antiunitary else matrix


def compose_semilinear(
    outer: SemilinearBlock,
    inner: SemilinearBlock,
) -> SemilinearBlock:
    return SemilinearBlock(
        outer.matrix @ semilinear_kappa(inner.matrix, outer.antiunitary),
        outer.antiunitary ^ inner.antiunitary,
    )


def inverse_semilinear(value: SemilinearBlock) -> SemilinearBlock:
    inverse = value.matrix.conj().T
    if value.antiunitary:
        inverse = inverse.conj()
    return SemilinearBlock(inverse, value.antiunitary)
```

Copy input arrays, make them C-contiguous/read-only, and reject invalid
unitarity using a caller-supplied certification bound rather than a global
engineering tolerance.

Add:

```python
@dataclass(frozen=True)
class BlockRouteAction:
    name: str
    antiunitary: bool
    fiber_permutation: tuple[int, ...]
    fiber_dimensions: tuple[int, ...]
    route_blocks: tuple[np.ndarray, ...]
```

Validate permutation bijectivity and every route shape
`(d_target, d_source)`.

**Step 4: Run the focused tests**

Expected: all semilinear and route-model tests pass.

**Step 5: Commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "feat(kp): add semilinear block action core"
```

### Task 2: Add dense/route conversion with structural support

**Files:**

- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write failing round-trip tests**

Create a two-fiber U(2) action with a legitimate `5e-5` internal entry. Test
that extraction and materialization preserve that entry bit-for-bit and that
all off-route entries are exactly zero:

```python
def test_route_round_trip_preserves_small_internal_entries() -> None:
    eps = 5.0e-5
    block = np.array(
        [[np.sqrt(1.0 - eps**2), eps], [-eps, np.sqrt(1.0 - eps**2)]],
        dtype=np.complex128,
    )
    action = BlockRouteAction(
        name="g",
        antiunitary=False,
        fiber_permutation=(1, 0),
        fiber_dimensions=(2, 2),
        route_blocks=(block, block.conj().T),
    )
    dense = materialize_block_route_action(action)
    restored = extract_block_route_action(
        dense,
        name="g",
        antiunitary=False,
        fiber_indices=((0, 1), (2, 3)),
        fiber_permutation=(1, 0),
    )
    assert restored.route_blocks[0][0, 1] == eps
    np.testing.assert_array_equal(materialize_block_route_action(restored), dense)
```

Add failures for off-route pollution above a supplied bound and incompatible
fiber dimensions.

**Step 2: Confirm failure**

Run the two new tests. Expected: missing conversion functions.

**Step 3: Implement conversion functions**

Add:

```python
def extract_block_route_action(
    matrix: np.ndarray,
    *,
    name: str,
    antiunitary: bool,
    fiber_indices: Sequence[Sequence[int]],
    fiber_permutation: Sequence[int],
    off_route_bound: float,
) -> BlockRouteAction: ...


def materialize_block_route_action(action: BlockRouteAction) -> np.ndarray: ...
```

Materialization must start from `np.zeros` and assign complete route blocks.
Do not call `eliminate_zeros`, round, or threshold block entries.

**Step 4: Run the focused tests and commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "feat(kp): preserve general U(d) route blocks"
```

### Task 3: Compile explicit magnetic presentations and central targets

**Files:**

- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write failing presentation tests**

Add immutable models:

```python
@dataclass(frozen=True)
class MagneticGenerator:
    name: str
    antiunitary: bool


@dataclass(frozen=True)
class MagneticRelation:
    name: str
    lhs: tuple[str, ...]
    rhs: tuple[str, ...]
    central_phase: complex


@dataclass(frozen=True)
class MagneticPresentation:
    generators: tuple[MagneticGenerator, ...]
    relations: tuple[MagneticRelation, ...]
    central_phases: tuple[complex, ...]
    source: str
```

Test exact presentations for:

- MgI2 M: `TR^2=+1`, `C2^2=+1`, `TR C2=C2 TR`;
- MoTe2 K: `C3z^3=-1`, `C2T^2=+1`, dihedral relation;
- spinful Gamma: `TR^2=C3z^3=C2^2=-1`, both commutators and
  the C2/C3 dihedral relation;
- Gamma with only TR/C3z;
- unsupported or ambiguous operation sets fail closed.

Test that `-I` and `+I` remain distinct central targets even when their
discrete actions are identical.

**Step 2: Confirm the tests fail**

Expected: presentation types/compiler missing.

**Step 3: Implement the compiler**

Add:

```python
def compile_continuum_magnetic_presentation(
    operations: Sequence[Mapping[str, Any]],
) -> MagneticPresentation: ...
```

Read power phases from the existing manifest `group_relations`. Compile mixed
relations only for recognized explicit action combinations. Check
antiunitary parity and explicit k/Q actions. Never select a mixed phase by an
unrestricted best-phase fit. V1 allowed central phases are the finite scalar
kernel declared by the presentation, normally `(+1, -1)`.

**Step 4: Add discrete-relation validation**

Implement word permutation/parity evaluation and reject any presentation whose
relations do not close exactly on `fiber_permutation`.

**Step 5: Run tests and commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "feat(kp): compile magnetic group presentations"
```

### Task 4: Build quotient-group, orbit, transporter, and Schreier data

**Files:**

- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write failing groupoid tests**

Use exact permutation fixtures for V4 and D3. Assert deterministic BFS words,
orbit roots, orbit sizes, transporter words, and Schreier stabilizer words.
Include the measured structures:

```text
V4 on 44 fibers: 11 free orbits of size 4
D3 on 54 fibers: 9 free orbits of size 6
D3 x T on 38 fibers: three size-12 orbits and one size-2 orbit
```

Keep the large permutations as test constants or a compact deterministic
generator; do not read `validation_runs/` from release tests.

**Step 2: Confirm failure**

Expected: orbit compiler missing.

**Step 3: Implement deterministic groupoid compilation**

Add models and functions:

```python
@dataclass(frozen=True)
class QuotientGroupElement:
    canonical_word: tuple[str, ...]
    antiunitary: bool
    fiber_permutation: tuple[int, ...]


@dataclass(frozen=True)
class ActionOrbit:
    root: int
    fibers: tuple[int, ...]
    transporter_words: tuple[tuple[str, ...], ...]
    stabilizer_words: tuple[tuple[str, ...], ...]


def compile_action_orbits(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
) -> tuple[ActionOrbit, ...]: ...
```

Use operation order from the presentation, lexicographic tie-breaking, and
explicit group-size/word-length limits. Do not use floating matrices as group
element keys.

**Step 4: Run tests and commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "feat(kp): decompose symmetry actions into groupoid orbits"
```

### Task 5: Add a direct U(1) relation-projection oracle

**Files:**

- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write the MgI2 phase regression**

Construct 11 four-fiber free orbits with the measured order of phase defects.
Assert the equal-weight constrained projection gives:

```text
TR RMS correction              4.3625209915e-6
C2 RMS correction              4.3625209915e-6
maximum phase correction       4.4519406526e-6
final mixed residual           below 5e-15
```

Also assert exact input is idempotent.

**Step 2: Confirm failure**

Expected: U(1) oracle missing.

**Step 3: Implement phase constraint assembly**

For each relation and source fiber, evaluate the word phase as a signed linear
form. Antiunitary operations to the left contribute a conjugation sign. Fix
the `2*pi` branch from the stage-1 residual and reject if it is not safely
inside the principal branch.

Add:

```python
def project_u1_relations(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, BlockRouteAction], Mapping[str, Any]]: ...
```

Solve the minimum-norm equality projection with rank-revealing SVD. Report
matrix shape, effective rank, nullity, smallest nonzero singular value,
condition number, RMS/max correction, and branch margin.

**Step 4: Run tests and commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "test(kp): add U(1) joint exactification oracle"
```

### Task 6: Implement exact-by-construction free-orbit U(d) synchronization

**Files:**

- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write failing U(2)/U(4) tests**

Generate exact free-orbit actions from random deterministic fiber frames,
including antiunitary edges and central `-I`. Perturb every route block by
`exp(X)` with `||X|| about 1e-6`. Assert:

- reconstructed blocks remain unitary;
- all presentation relations are below their computed floating bound;
- correction is no larger than the direct route-space oracle by more than the
  test optimization tolerance;
- exact inputs are idempotent;
- applying a random fiber gauge before exactification gives the gauge transform
  of the original result.

**Step 2: Confirm failure**

Expected: synchronizer missing.

**Step 3: Implement central transition selection**

For every canonical transporter transition, compute the stage-1 holonomy and
select the unique nearest phase from `presentation.central_phases`. Require a
configured separation margin and validate the complete transition table for
associativity. Do not retain the measured phase as an unconstrained scalar.

**Step 4: Implement transporter synchronization**

Parameterize each non-root frame as `S'_x = exp(Y_x) S_x` with
`Y_x^dagger=-Y_x`. Reconstruct every edge exactly from the transporter frames
and central transition table. Use a chordal residual for the objective and an
analytic real Jacobian. Solve each orbit independently with rank-revealing
least squares and a monotone line search.

Required API:

```python
def synchronize_free_orbit(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    orbit: ActionOrbit,
    *,
    config: JointExactificationConfig,
) -> tuple[dict[str, tuple[np.ndarray, ...]], FreeOrbitReport]: ...
```

Stop at a dimension/operation-count-derived floating bound. Fail if correction,
conditioning, central margin, or iteration count violates configuration.

**Step 5: Run tests and compare with the U(1) oracle**

For `d=1`, the production result must match Task 5's correction and relation
residual within `1e-12` relative correction error.

**Step 6: Commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "feat(kp): synchronize free U(d) symmetry orbits"
```

### Task 7: Exactify nontrivial stabilizer corepresentations

**Files:**

- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write failing stabilizer tests**

Use the existing spinful Gamma four-state C3/TR/C2 matrices as a U(4)
stabilizer fixture. Add small anti-Hermitian perturbations and assert recovery
of all power, commutator, and dihedral relations. Add:

- `T T* = -I` with odd `d` fails before iteration;
- a deliberately wrong central branch fails;
- a relation-log eigenphase near `pi` fails the branch-margin gate;
- exact ZrS2-like input is idempotent.

**Step 2: Confirm failure**

Expected: stabilizer solver missing.

**Step 3: Implement real coordinates for u(d)**

Create deterministic diagonal and real/imaginary off-diagonal anti-Hermitian
basis matrices. Implement `pack_skew_hermitian` and
`unpack_skew_hermitian` with exact inverse tests.

**Step 4: Implement word residuals and analytic Jacobians**

For a relation, compute the unitary mismatch and its principal anti-Hermitian
log. For a word `F_m ... F_1`, implement the left-trivialized variation

```text
delta(B_w) B_w^-1 =
sum_j Ad_(F_m ... F_(j+1)) kappa_(left_parity_j)(X_g_j).
```

Represent the Jacobian as a real matrix. Remove redundant constraint rows with
SVD and retain the expected gauge nullity.

**Step 5: Implement constrained Riemannian SQP/Newton projection**

Minimize correction to stage-1 stabilizer blocks subject to zero relation
logs. Use `scipy.linalg.expm` updates, a trust radius, and a line search that
must reduce the maximum relation residual. Do not polar-project after an
accepted exponential update.

**Step 6: Run tests and commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "feat(kp): exactify magnetic U(d) stabilizers"
```

### Task 8: Add the joint orchestrator, certification, and artifact schema

**Files:**

- Modify: `kp/kp/symmetry/joint_exactification.py`
- Modify: `tests/kp/test_joint_exactification.py`

**Step 1: Write failing end-to-end small-matrix tests**

Exercise free and nontrivial-stabilizer orbits through one public function.
Assert deterministic artifact JSON, array hashes, correction metrics, central
branch provenance, pre/post residual tables, and load-time re-certification.

**Step 2: Confirm failure**

Expected: orchestrator/artifact API missing.

**Step 3: Implement configuration and report types**

Add versioned models including:

```python
@dataclass(frozen=True)
class JointExactificationConfig:
    enabled: bool = True
    max_rms_correction: float = 1.0e-3
    max_route_correction: float = 5.0e-3
    central_branch_margin: float = 1.0e-3
    max_iterations: int = 20
    condition_limit: float = 1.0e12


@dataclass(frozen=True)
class JointExactificationResult:
    actions: Mapping[str, BlockRouteAction]
    presentation: MagneticPresentation
    report: Mapping[str, Any]
    artifact_metadata: Mapping[str, Any]
    artifact_arrays: Mapping[str, np.ndarray]
```

Numeric defaults must be reviewed against the existing stage-1 tolerance and
the measured MgI2 correction. Record all effective values in the artifact.

**Step 4: Implement the public orchestrator**

```python
def joint_exactify_block_actions(
    actions: Mapping[str, BlockRouteAction],
    presentation: MagneticPresentation,
    *,
    config: JointExactificationConfig,
) -> JointExactificationResult: ...
```

Compile orbits, exactify stabilizers, synchronize transports, materialize
generator routes, and independently certify every relation on every source
fiber. Compute a floating bound from matrix dimension, word length, and machine
epsilon. Fail if any residual exceeds that bound multiplied by one documented
safety factor.

**Step 5: Implement pack/load/hash verification**

Store route/orbit arrays under reserved NPZ keys. Hash canonical JSON metadata
plus canonical C-order complex arrays. Loading must reject a changed byte,
changed presentation, missing array, or failed re-certification.

**Step 6: Run tests and commit**

```bash
git add kp/kp/symmetry/joint_exactification.py \
  tests/kp/test_joint_exactification.py
git commit -m "feat(kp): certify joint magnetic representations"
```

### Task 9: Integrate joint closure after stage-1 KP exactification

**Files:**

- Modify: `kp/kp/symmetry/exactify_representation.py:1366`
- Modify: `tests/kp/test_exactify_representation.py`
- Modify: `kp/kp/symmetry/joint_exactification.py`

**Step 1: Write failing integration tests**

Add a loaded-source MgI2-style TR/C2 fixture in which both power relations are
exact but the mixed relation is `about 1e-5`. Assert
`exactify_loaded_symmetry_source` returns jointly closed matrices, preserves
the selected support/power branches, and reports the measured small correction.

Add idempotence fixtures for:

- MoTe2 K C3z/C2T with `d=1`;
- MoTe2 K C3z/C2T with `d=2`;
- spinful Gamma TR/C3z/C2 with `d=4` and an applied-looking internal frame.

Add an unsupported ambiguous presentation test that fails closed only when
joint exactification is required by configuration.

**Step 2: Confirm failure**

Run only the new test names. Expected: matrices retain the mixed defect or lack
the joint report.

**Step 3: Preserve route-layout information during stage 1**

While stage 1 resolves label/block support, retain deterministic fiber index
groups and fiber permutations for every operation. Do not infer route support
again from thresholded dense matrices.

**Step 4: Call the joint orchestrator once after all operations**

At the end of `exactify_loaded_symmetry_source`, compile the presentation,
convert stage-1 matrices to route actions, run one joint exactification, replace
the dense generator matrices with the jointly materialized matrices, and attach
the same root joint report/artifact reference to every operation report.

Remove the C3-only special joint validation as the authoritative path. Keep its
algebraic template recognizer as a stage-1 branch selector, and cover the former
TR/C3/C2 checks through the generic presentation certifier.

**Step 5: Run focused tests and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_joint_exactification.py \
  tests/kp/test_exactify_representation.py

git add kp/kp/symmetry/joint_exactification.py \
  kp/kp/symmetry/exactify_representation.py \
  tests/kp/test_joint_exactification.py \
  tests/kp/test_exactify_representation.py
git commit -m "feat(kp): jointly exactify continuum symmetry generators"
```

### Task 10: Preserve joint closure through gauge and remove threshold cleanup

**Files:**

- Modify: `kp/kp/symmetry/projection.py:127-156`
- Modify: `kp/kp/symmetry/projection.py:3212-3335`
- Modify: `kp/kp/basis/symmetry_gauge.py:483`
- Replace: `tests/kp/test_symmetry_matrix_storage_cleanup.py`
- Modify: `tests/kp/test_symmetry_adapted_gauge.py`
- Modify: `tests/kp/test_symm_projection.py`

**Step 1: Replace threshold expectations with preservation tests**

Rewrite the storage test so a legitimate `5e-5` block entry survives save/load
exactly. Assert support zeros remain zero because materialization created them
as zero. Remove every expectation for `absolute_entry_threshold_v1`.

**Step 2: Write a failing post-gauge joint certification test**

Use a nontrivial U(4) frame and TR/C3z/C2 generators. Assert all relations pass
before and after `transform_basis_operation`, and assert the projection summary
contains both pre-gauge and post-gauge certification records with the same
presentation hash.

**Step 3: Confirm failures**

Expected: the small storage entry is zeroed and post-gauge certification is
absent.

**Step 4: Remove magnitude cleanup**

Delete `EXACTIFIED_MATRIX_STORAGE_ZERO_THRESHOLD` and
`_cleanup_exactified_matrix_for_storage`. Make `_save_exactified_matrix` save a
copy without numeric mutation, or inline the plain `np.save` call. Replace
`matrix_storage_cleanup` metadata with:

```json
{
  "policy": "structural_route_support_v1",
  "numeric_entries_modified": 0
}
```

**Step 5: Certify after the common gauge**

After applying the one shared `SymmetryAdaptedBasisFrame`, transform the route
artifact consistently or re-extract route blocks on the exact known support,
then call the generic relation certifier. Do not run the optimizer again.

**Step 6: Run focused tests and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_symmetry_matrix_storage_cleanup.py \
  tests/kp/test_symmetry_adapted_gauge.py \
  tests/kp/test_symm_projection.py

git add kp/kp/symmetry/projection.py \
  kp/kp/basis/symmetry_gauge.py \
  tests/kp/test_symmetry_matrix_storage_cleanup.py \
  tests/kp/test_symmetry_adapted_gauge.py \
  tests/kp/test_symm_projection.py
git commit -m "fix(kp): preserve joint symmetry closure through storage"
```

### Task 11: Pack the joint artifact and keep factorization optional

**Files:**

- Modify: `kp/kp/symmetry/projection.py:3103`
- Modify: `kp/kp/symmetry/factorized_action.py`
- Modify: `tests/kp/test_factorized_symmetry_action.py`
- Modify: `tests/kp/test_canonical_output_layout.py`

**Step 1: Write failing package tests**

Assert `representations.npz` contains the versioned joint artifact metadata and
arrays, reloads and re-certifies it, and still exposes dense matrices for
compatibility. Corrupt one route block and assert load failure.

Add a valid route-dependent U(2) action that cannot factor as
`q_phase * common_orbital_block`. Assert:

- joint artifact status is `certified`;
- factorized artifact status is `unavailable`;
- canonical symmetry output still succeeds.

**Step 2: Confirm failure**

Expected: no joint package and factorization is currently treated as the only
compact action form.

**Step 3: Pack joint arrays before optional factorization**

Reserve deterministic keys beginning with
`__joint_block_representation_`. Include the presentation/artifact hash in
`kp_symm_exactification` metadata. Run factorization certification after the
joint package is frozen; factorization failure records a reason but does not
invalidate the general representation.

**Step 4: Run focused tests and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_factorized_symmetry_action.py \
  tests/kp/test_canonical_output_layout.py

git add kp/kp/symmetry/projection.py \
  kp/kp/symmetry/factorized_action.py \
  tests/kp/test_factorized_symmetry_action.py \
  tests/kp/test_canonical_output_layout.py
git commit -m "feat(kp): package certified joint block representations"
```

### Task 12: Make response-group compilation consume explicit central closure

**Files:**

- Modify: `kp/kp/model/response_basis.py:616`
- Modify: `kp/kp/model/response_basis_factorized.py`
- Modify: `tests/kp/test_complete_response_basis.py:1656-1725`
- Modify: `tests/kp/test_response_basis_factorized.py`

**Step 1: Replace arbitrary-phase closure tests**

Change the existing test that accepts any projective global phase. New tests
must distinguish:

- a declared central `-I`, which is valid and retained;
- an undeclared arbitrary phase, which fails;
- two canonical words for the same explicit central group element, which agree
  within the floating certification bound;
- a general route-block action, whose composition uses the source-fiber route
  rather than one common `internal_u`.

**Step 2: Confirm failure**

Expected: current `build_finite_group` collapses the central phase or accepts
the undeclared phase.

**Step 3: Consume the frozen joint group table**

Use the joint artifact's canonical group elements, central labels, and words as
the production group source. Keep the old discrete-key/best-phase BFS as an
explicit diagnostic helper only. Do not independently regenerate non-generator
matrices from noisy alternate paths.

**Step 4: Run focused tests and commit**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py

git add kp/kp/model/response_basis.py \
  kp/kp/model/response_basis_factorized.py \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py
git commit -m "fix(kp): compile response groups from explicit central actions"
```

### Task 13: Run all local small-matrix regression tests

**Files:**

- Modify only if a real regression is found; do not weaken assertions.

**Step 1: Run the complete focused set locally**

```bash
python -m pytest -q -p no:cacheprovider \
  tests/kp/test_joint_exactification.py \
  tests/kp/test_exactify_representation.py \
  tests/kp/test_symmetry_matrix_storage_cleanup.py \
  tests/kp/test_symmetry_adapted_gauge.py \
  tests/kp/test_factorized_symmetry_action.py \
  tests/kp/test_complete_response_basis.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_symm_projection.py \
  tests/kp/test_canonical_output_layout.py
```

Expected: all pass without external data or large model computation.

**Step 2: Inspect the diff and dirty overlap**

```bash
git status --short
git diff --check
git diff --stat
```

Verify no validation outputs, caches, existing unrelated docs, TAPW files, or
paper files are staged.

**Step 3: Commit any test-only cleanup if needed**

Use a narrowly scoped commit. Do not amend unrelated user commits.

### Task 14: Run release and material validation on bigmem003

**Files:**

- Create locally/shared but do not commit:
  `validation_runs/joint_ud_exactification_20260715/`

**Step 1: Verify the remote environment**

Run:

```bash
ssh bigmem003 'source /data/home/zy/mambaforge/etc/profile.d/conda.sh && \
  conda activate moirekp && \
  cd /data/work/zy/software/1.tapw_code/moirekp-release && \
  hostname && which python && python -V && which kp && kp --help >/dev/null'
```

Expected: host `bigmem003`, Python 3.11 from the `moirekp` environment, and
`kp` from the same environment/editable checkout.

**Step 2: Run the required release suite remotely**

```bash
ssh bigmem003 'source /data/home/zy/mambaforge/etc/profile.d/conda.sh && \
  conda activate moirekp && \
  cd /data/work/zy/software/1.tapw_code/moirekp-release && \
  python -m pytest -q -p no:cacheprovider -m "not slow and not external_data"' \
  | tee validation_runs/joint_ud_exactification_20260715/full_non_external_bigmem003.log
```

Expected: exit status 0. Record the exact pass count and elapsed time.

**Step 3: Prepare validation-only config copies**

Create config copies under
`validation_runs/joint_ud_exactification_20260715/configs/` for:

- MgI2 M spin-up from
  `validation_runs/multimaterial_kinetic_weighting_20260715/configs/mgi2_m_spinup.yaml`;
- MoTe2 K spin-up and spinful from the adjacent configs;
- ZrS2 Gamma q04 from
  `/data/work/zy/software/1.tapw_code/moirekp/kp/configs/zrs2_G/AB_model/3.15/kp/configs/zrs2_3.15_Gamma_q04.yaml`.

Change only `symm.output_dir` and, if required, the validation projection
output path. Point every output into the new validation directory. Do not
modify the source configs.

**Step 4: Run MgI2, MoTe2, and ZrS2 `kp symm` remotely**

Use one command per case so failures are isolated:

```bash
ssh bigmem003 'source /data/home/zy/mambaforge/etc/profile.d/conda.sh && \
  conda activate moirekp && \
  cd /data/work/zy/software/1.tapw_code/moirekp-release && \
  kp symm -c validation_runs/joint_ud_exactification_20260715/configs/mgi2_m_spinup.yaml'
```

Repeat for MoTe2 K spin-up, MoTe2 K spinful, and ZrS2 Gamma q04.

**Step 5: Audit the produced representations**

Run a small matrix-only audit that reports:

- all generator unitarity residuals;
- every declared power and mixed relation residual;
- stage-1 to joint RMS/max correction;
- pre-gauge and post-gauge certification bounds;
- joint artifact and presentation hashes;
- factorization status;
- storage policy and `numeric_entries_modified=0`.

Required acceptance:

```text
MgI2 M mixed residual       <= certified floating bound, target about 3e-16
MgI2 per-op RMS correction  about 4.3625e-6 with equal weights
MoTe2 K corrections         floating-floor/idempotent
ZrS2 Gamma corrections      floating-floor/idempotent
all post-gauge relations    <= their recorded certification bounds
```

**Step 6: Write a validation report**

Create but do not commit:

`validation_runs/joint_ud_exactification_20260715/REPORT.md`

Include host, commit, environment, commands, exact residuals, correction
metrics, and links to generated NPZ/JSON artifacts. Do not copy the report into
`examples/` or release-facing docs.

### Task 15: Final verification and handoff

**Files:**

- Modify: `docs/plans/2026-07-15-kp-joint-ud-block-exactification.md` only if
  implementation facts require correcting this plan.

**Step 1: Run final evidence checks**

Use @verification-before-completion. Re-read the remote test log, material
report, and `git diff --check`. Confirm every required command exited 0.

**Step 2: Review repository hygiene**

```bash
git status --short
git log --oneline --decorate -15
```

Confirm validation output, caches, egg-info, review bundles, and unrelated
dirty files are not staged.

**Step 3: Report the result**

Summarize:

- the structurally exact joint representation architecture;
- MgI2 before/after residual and correction;
- MoTe2/ZrS2 idempotence and gauge evidence;
- removal of magnitude-based matrix mutation;
- local focused tests and bigmem003 release/material validation;
- every commit created by this plan;
- any remaining explicit limitation, especially unsupported nonsymmorphic
  central block actions.

