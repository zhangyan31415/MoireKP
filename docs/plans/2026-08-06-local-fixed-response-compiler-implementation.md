# Local Generator-Fixed Response Compiler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace per-seed full-group Reynolds projection in the graded response compiler with a certified local generator-fixed solver that compiles the ZrS2 Gamma q04 cold response basis in at most 20 seconds on both bigmem nodes.

**Architecture:** Build exact joint local vocabularies from authored raw columns, certified generator routes, Hermitian partners, and complete lower-degree support. Solve the common generator/Hermitian fixed space in whitened local real coordinates, retain original logical-owner provenance, run the existing global filtered-tail certificate only on the small fixed output, and keep the current Reynolds implementation as a typed fallback and independent oracle.

**Tech Stack:** Python 3.11, NumPy, SciPy dense/sparse linear algebra, pytest, existing MoireKP factorized-action and graded-response infrastructure.

---

## Execution rules

- Work on `codex/cpc-release-clean`; the user explicitly requested the CPC branch, so do not create or switch to another worktree.
- Edit and inspect on the login host, but run every test and benchmark only through `ssh bigmem001` or `ssh bigmem003`.
- Use `/data/home/zy/mambaforge/envs/moirekp/bin/python` and set `PYTHONPATH=kp:tapw`.
- For comparable benchmarks, pin to cores 0-7 and set all BLAS/OpenMP thread counts to eight with dynamic threading disabled.
- Do not modify or stage unrelated untracked files under `docs/plans`, `examples/bitei`, `examples/toZY`, `tmp`, or `tests/test_ai_assets.py`.
- Follow TDD: add one focused failing test, run it remotely and observe the intended failure, add the minimal implementation, rerun, then commit only the owned files.
- Keep the current Reynolds path callable until all oracle and release gates pass.

Define the common remote prefix in shell commands as:

```bash
REPO=/data/work/zy/software/1.tapw_code/moirekp-release
PY=/data/home/zy/mambaforge/envs/moirekp/bin/python
ENV='PYTHONPATH=kp:tapw PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 BLIS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE'
```

## Task 1: Add an isolated response-compiler benchmark runner

**Files:**
- Create: `devtools/benchmarks/kp/bench_response_compile.py`
- Create: `tests/kp/test_response_compile_benchmark.py`

**Step 1: Write the failing CLI-contract test**

Test a temporary fake compiler callback or monkeypatched public compile entry so that the runner records configuration fingerprint, cache mode, compile-only wall time, cache-store time, hostname, git commit, compiler version, result rank/channel metadata, and peak RSS in a JSON object. Require cold mode to reject a nonempty cache directory and warm mode to reject a missing cache artifact.

```python
def test_benchmark_record_contains_reproducible_compile_fields(tmp_path, monkeypatch):
    record = run_benchmark(...)
    assert record["schema_version"] == "response-compile-benchmark-v1"
    assert record["cache_mode"] == "cold"
    assert record["compile_seconds"] >= 0.0
    assert record["git_commit"]
    assert record["input_fingerprint"]
    assert record["compiler_version"]
```

**Step 2: Run the test on bigmem001 and verify the intended failure**

```bash
ssh bigmem001 "cd $REPO && env $ENV taskset -c 0-7 $PY -m pytest -p no:cacheprovider -q tests/kp/test_response_compile_benchmark.py"
```

Expected: FAIL because the benchmark module does not exist.

**Step 3: Implement the minimal runner**

Use the normal configuration/model builders, redirect only `moire.output_dir` to the requested cache directory, clear the in-memory response cache before timing, and bracket only `compile_model_response_basis(...)` for the main timing:

```python
started = time.perf_counter()
basis = compile_model_response_basis(...)
compile_seconds = time.perf_counter() - started
```

Do not delete cache directories in the runner. A caller creates each cold directory explicitly. RSS profiling is optional per invocation so it does not perturb official timing samples.

**Step 4: Rerun the focused test on bigmem001**

Expected: PASS.

**Step 5: Record the pre-change protocol baseline**

On both nodes run three fresh cold directories and three corresponding warm processes in interleaved order. Save JSON outside the repository under `/tmp/moirekp-response-baseline-20260806/`. Do not run node pairs concurrently with other ZrS2 pipelines.

**Step 6: Commit**

```bash
git add devtools/benchmarks/kp/bench_response_compile.py tests/kp/test_response_compile_benchmark.py
git commit -m "test(kp): add isolated response compiler benchmark"
```

## Task 2: Introduce local fixed-problem types and typed unavailability

**Files:**
- Create: `kp/kp/model/response_basis_local_fixed.py`
- Create: `tests/kp/test_response_basis_local_fixed.py`

**Step 1: Write failing type-validation tests**

Cover deterministic normalization and immutable metadata for:

```python
LocalVocabularyComponent(
    component_index=0,
    nominal_degrees=(0,),
    logical_owner_indices=(0, 1),
    ambient_columns=V,
    generator_actions={"C3": A},
    antiunitary_parities={"C3": False},
    hermitian_action=J,
    absolute_error_bounds={"C3": 0.0, "H": 0.0},
)
```

Reject inconsistent shapes, non-finite values, missing parity entries, duplicate owners, and noncanonical component order. Verify `LocalGeneratorNullspaceUnavailable` carries a stable `reason` and immutable certificate.

**Step 2: Run on bigmem001 and verify failure**

Expected: FAIL on missing module/types.

**Step 3: Implement the dataclasses and validation only**

Do not add a compiler yet. Store sparse matrices as CSC and real local action matrices as `float64`. Preserve the original logical owner order separately from canonical component ordering.

**Step 4: Rerun and commit**

```bash
git add kp/kp/model/response_basis_local_fixed.py tests/kp/test_response_basis_local_fixed.py
git commit -m "feat(kp): define local fixed response problems"
```

## Task 3: Build exact joint components with complete lower-degree tails

**Files:**
- Modify: `kp/kp/model/response_basis_local_fixed.py`
- Modify: `tests/kp/test_response_basis_local_fixed.py`
- Reference: `tests/kp/test_response_basis_graded.py:512`

**Step 1: Write failing component tests**

Create synthetic raw columns containing:

1. two disjoint generator/Hermitian components that must remain split;
2. two different top-degree structural orbits with a shared degree-zero tail that must merge;
3. a Hermitian off-diagonal owner pair that must merge;
4. an action image leaving the vocabulary, which must raise typed unavailability;
5. shuffled seed/generator input that must produce the same ordered components.

The shared-tail assertion is:

```python
components = build_exact_joint_components(...)
assert tuple(c.logical_owner_indices for c in components) == ((0, 1, 2, 3),)
```

**Step 2: Run remotely and observe failure**

Expected: FAIL because `build_exact_joint_components` is undefined.

**Step 3: Implement union-find component construction**

Merge support indices along

\[
E=E_{\rm generator}\cup E_{J_H}\cup E_{\rm ambient\ support}.
\]

Use exact sparse structural overlap for candidate edges, followed by a numerical leakage certificate. Never infer independence solely from nominal degree or `StructuralOrbit` membership. Return stable components ordered by the minimum canonical logical owner ID.

**Step 4: Add the production regression equivalent of the existing shared-tail test**

Ensure the new component builder merges the two top-orbit heads before any local nullspace solve.

**Step 5: Rerun and commit**

```bash
git add kp/kp/model/response_basis_local_fixed.py tests/kp/test_response_basis_local_fixed.py
git commit -m "feat(kp): build exact joint response components"
```

## Task 4: Compile and certify local generator/Hermitian actions

**Files:**
- Modify: `kp/kp/model/response_basis_local_fixed.py`
- Modify: `tests/kp/test_response_basis_local_fixed.py`
- Reference: `kp/kp/model/response_basis_fixed_compiler.py`
- Reference: `kp/kp/model/response_basis_fixed_subspace.py`

**Step 1: Write failing action tests**

Cover:

- unitary realification;
- antiunitary realification;
- nonorthogonal vocabulary whitening;
- ZrS2-shaped two-by-two C3 route plus monomial C2/TR routes;
- Hermitian diagonal and off-diagonal blocks;
- local image closure residual above the propagated bound;
- a singular vocabulary value in the gray band.

Use the exact conventions:

\[
\mathcal R_U(A)=\begin{bmatrix}\Re A&-\Im A\\\Im A&\Re A\end{bmatrix},
\qquad
\mathcal R_A(A)=\begin{bmatrix}\Re A&\Im A\\\Im A&-\Re A\end{bmatrix}.
\]

**Step 2: Run remotely and observe failure**

Expected: FAIL on missing action compiler.

**Step 3: Implement local independent-vocabulary filtering**

Normalize raw vocabulary columns independently, compute a small Gram/SVD, and classify rank from propagated errors. Do not pass a dependent vocabulary to `response_basis_fixed_compiler`.

**Step 4: Implement local action restriction and whitening**

For each certified image solve its local coordinates and verify

\[
\|D_gV-VA_g\|\le e_g.
\]

With \(G=LL^\mathsf T\), build

\[
\widehat A_g=L^\mathsf TA_gL^{-\mathsf T}
\]

and certify orthogonality within the propagated action bound.

**Step 5: Reuse the existing fixed-subspace solver**

Call `solve_generator_fixed_subspace` with the realified generator actions and Hermitian action. Convert vocabulary-rank, closure, representation, fixed-nullity, and oversize-component conditions to `LocalGeneratorNullspaceUnavailable`; do not catch arbitrary `RuntimeError`.

**Step 6: Rerun on both bigmem nodes and commit**

```bash
git add kp/kp/model/response_basis_local_fixed.py tests/kp/test_response_basis_local_fixed.py
git commit -m "feat(kp): solve certified local generator fixed spaces"
```

## Task 5: Preserve logical owners and certify small global tail selection

**Files:**
- Modify: `kp/kp/model/response_basis_local_fixed.py`
- Modify: `kp/kp/model/response_basis_graded.py`
- Modify: `tests/kp/test_response_basis_local_fixed.py`
- Modify: `tests/kp/test_response_basis_graded.py`

**Step 1: Write failing owner/provenance tests**

Require:

- output owner IDs come from original `seed_id:real/imag` IDs;
- deterministic owner selection under shuffled vocabulary input;
- pivot ambiguity without a certified gap causes typed fallback;
- error bounds are propagated through local transfer coefficients;
- a shared lower-degree tail across two locally proposed components is removed by the global certificate;
- raw-adjoint and harmonic provenance survive unchanged.

**Step 2: Run remotely and observe failure**

Expected: FAIL because local fixed outputs are not mapped to graded owners.

**Step 3: Implement logical projection coordinates**

Map the fixed projector back to authored logical columns, run deterministic small pivot selection, and retain both the fixed-space coordinates and the original owner index. Selection must operate on the small fixed coordinate matrix, not on ambient projected columns.

**Step 4: Adapt the existing filtered selector**

Extract or wrap `_select_filtered_independent_column_blocks` so it can certify the fixed outputs and their complete lower-degree tails. Preserve the current degree proof and residual-bound artifact schema where possible.

**Step 5: Materialize only selected fixed outputs**

Lift selected coordinates through raw `ambient_columns`; do not construct omitted Reynolds columns. Run final Hermiticity, generator, adjoint, and Q-center residual checks before returning.

**Step 6: Rerun focused tests and commit**

```bash
git add kp/kp/model/response_basis_local_fixed.py kp/kp/model/response_basis_graded.py tests/kp/test_response_basis_local_fixed.py tests/kp/test_response_basis_graded.py
git commit -m "feat(kp): select local fixed response owners"
```

## Task 6: Integrate the local path into the graded compiler

**Files:**
- Modify: `kp/kp/model/response_basis_graded.py:1493`
- Modify: `kp/kp/model/response_basis.py:34`
- Modify: `tests/kp/test_complete_response_basis.py`
- Modify: `tests/kp/test_response_basis_cache.py`

**Step 1: Write failing routing/fallback tests**

Require:

1. p0 and finite-p complete policies enter the local graded path;
2. `optimize_zero_harmonic` remains unchanged;
3. typed local unavailability invokes the legacy Reynolds path;
4. unexpected `RuntimeError` propagates;
5. fast/oracle mismatch is not cached;
6. changing algorithm/tolerance/gray-factor/component policy invalidates persistent cache;
7. v53 cache artifacts cannot bypass the new compiler version.

**Step 2: Run remotely and observe failure**

Expected: FAIL because the graded compiler has no local route and the compiler version is still v53.

**Step 3: Extract the legacy implementation**

Move the current projection/selection body behind a private Reynolds helper without changing behavior. Keep its output and timing artifact stable so it remains a trustworthy fallback/oracle.

**Step 4: Add the new entry route**

After `build_structural_compilation_plan` and before `_symbolic_reynolds_batch`, attempt the local compiler. Catch only `LocalGeneratorNullspaceUnavailable`, record its reason/certificate, and call the private Reynolds helper.

**Step 5: Update artifacts and cache identity**

Bump `COMPILER_VERSION` to a new explicit local-fixed version. Add algorithm version, numerical policy, component cutoff, materialization policy, generator parity, Q center, and harmonic-adjoint provenance to the input record.

**Step 6: Rerun focused routing/cache tests on both nodes and commit**

```bash
git add kp/kp/model/response_basis_graded.py kp/kp/model/response_basis.py tests/kp/test_complete_response_basis.py tests/kp/test_response_basis_cache.py
git commit -m "feat(kp): route graded compilation through local fixed spaces"
```

## Task 7: Add independent fast-versus-Reynolds production oracles

**Files:**
- Create: `tests/kp/test_response_basis_local_fixed_production.py`
- Modify: `tests/kp/test_complete_response_basis.py`

**Step 1: Add marked production oracle tests**

For MgI2 Gamma and ZrS2 Gamma q04, compile once with the local backend and once with forced legacy Reynolds. Orthonormalize physical outputs and assert

\[
r_{\rm fast}=r_{\rm oracle},\qquad
\|P_{\rm fast}-P_{\rm oracle}\|_2
\le\max(10^{-10},10b_{\rm cert}).
\]

Also verify every generator and full group word, antiunitary mixed words, final Hermiticity, finite-Q shifted-origin translation, raw-adjoint provenance, and final model RMS/Max values.

**Step 2: Verify the test fails if one local action or parity bit is deliberately perturbed**

Expected: projector/residual certification failure, followed by typed fallback; no fast cache entry.

**Step 3: Run the true oracle test on bigmem003**

This is allowed to be slow. Save the JSON/artifact comparison under `/tmp`, not in the repository.

**Step 4: Run deterministic-order tests on both nodes**

Run with 1, 8, and 32 BLAS threads and shuffled seed/generator input. Physical projector and certified rank must match. Canonical owner order/hash must match whenever all pivot gaps are certified.

**Step 5: Commit**

```bash
git add tests/kp/test_response_basis_local_fixed_production.py tests/kp/test_complete_response_basis.py
git commit -m "test(kp): certify local fixed compiler against Reynolds"
```

## Task 8: Measure, profile, and decide whether block gain graph is needed

**Files:**
- Modify if required: `kp/kp/model/response_basis_local_fixed.py`
- Modify if required: `tests/kp/test_response_basis_local_fixed.py`
- Modify: `devtools/benchmarks/kp/bench_response_compile.py`

**Step 1: Record component diagnostics**

Emit per component:

```text
raw/logical dimension
independent real dimension
fixed rank
generator nnz
Gram/action/nullspace/materialization seconds
fallback reason
peak local matrix bytes
```

**Step 2: Run three cold and three warm samples per node**

Use independent cold cache directories and new Python processes in order

\[
C_1,W_1,C_2,W_2,C_3,W_3.
\]

Compute

\[
C_h=\operatorname{median}(C_{h,*}),\qquad
W_h=\operatorname{median}(W_{h,*}).
\]

**Step 3: Apply the decision gate**

If

\[
\max(C_{001},C_{003})\le20\ \mathrm{s}
\]

and no component falls back for size, do not implement gain graph.

If the target is missed and local dense solve is the measured bottleneck, stop and write a separate approved plan for the certified block gain-graph backend. Do not mix a second algorithm into this implementation without evidence.

If Reynolds fallback or ambient materialization is the bottleneck, fix that certified integration defect instead of adding gain graph.

**Step 4: Commit any benchmark/diagnostic-only changes**

```bash
git add devtools/benchmarks/kp/bench_response_compile.py kp/kp/model/response_basis_local_fixed.py tests/kp/test_response_basis_local_fixed.py
git commit -m "perf(kp): expose local fixed compiler diagnostics"
```

## Task 9: Run the complete verification matrix

**Files:**
- No production changes unless a failing test identifies a defect.

**Step 1: Run the focused response suite on both nodes**

```bash
$PY -m pytest -p no:cacheprovider -q \
  tests/kp/test_response_basis_fixed_subspace.py \
  tests/kp/test_response_basis_fixed_compiler.py \
  tests/kp/test_response_basis_local_fixed.py \
  tests/kp/test_response_basis_local_fixed_production.py \
  tests/kp/test_response_basis_structural.py \
  tests/kp/test_response_basis_graded.py \
  tests/kp/test_response_basis_symmetry_first.py \
  tests/kp/test_response_basis_factorized.py \
  tests/kp/test_response_basis_adjoint.py \
  tests/kp/test_response_basis_cache.py \
  tests/kp/test_complete_response_basis.py
```

Expected: zero failures on both nodes.

**Step 2: Run the complete non-external suite on both nodes**

Use separate `--basetemp` paths and `-p no:cacheprovider`:

```bash
$PY -m pytest -p no:cacheprovider --basetemp=/tmp/moirekp-pytest-$HOSTNAME -m 'not slow and not external_data' -q
```

Expected: zero failures on both nodes.

**Step 3: Run `scripts/release_gate.sh` on both nodes**

Expected: zero failures.

**Step 4: Run all ten release pipelines**

Use `validation_runs/release_full_20260805/run_pipeline.py`, split the cases between bigmem001 and bigmem003, and require every case to report zero for project, symm, and model plus `CERTIFIED` selection state.

For every case require physical output equivalence:

\[
\operatorname{allclose}(E_{\rm new},E_{\rm base};
\mathrm{atol}=10^{-8}\ \mathrm{eV},\mathrm{rtol}=10^{-9}),
\]

\[
|\mathrm{RMS}_{\rm new}-\mathrm{RMS}_{\rm base}|
\le10^{-5}\ \mathrm{meV},
\]

\[
|\mathrm{Max}_{\rm new}-\mathrm{Max}_{\rm base}|
\le10^{-5}\ \mathrm{meV}.
\]

Do not overlap the ZrS2 microbenchmark with the ZrS2 full pipeline because the latter refreshes canonical outputs.

**Step 5: Re-run the official performance samples after correctness tests**

Acceptance requires

\[
\max(C_{001},C_{003})\le20\ \mathrm{s},
\qquad
\max(W_{001},W_{003})\le5\ \mathrm{s},
\]

per-host median absolute deviation at most ten percent, cross-host median difference at most ten percent, and no single cold sample above 1.25 times its host median.

Memory must satisfy

\[
M_{\rm new}\le\max(1.10M_{\rm old},M_{\rm old}+256\ \mathrm{MiB}).
\]

**Step 6: Review all changes and commit any final test-only correction**

Run `git diff --check`, inspect `git status --short`, and stage only files belonging to this optimization.

## Task 10: Final independent review

**Files:**
- Review all commits since `67cf0cd`.

**Step 1: Request a correctness review**

Use a fresh review agent to inspect action conventions, antiunitary parity, lower-degree component merging, error-bound propagation, cache identity, and typed fallback behavior.

**Step 2: Request a performance/data-structure review**

Use another fresh review agent to verify that the production fast path no longer constructs Reynolds columns for omitted owners and that timings exclude cache store/configuration work.

**Step 3: Address findings with focused tests**

Any fix starts with a reproducing test on bigmem001 or bigmem003 and receives its own commit.

**Step 4: Run the affected focused suite again**

Do not claim completion from code inspection alone.
