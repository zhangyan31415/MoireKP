# Fixed physical orbitals and Sz: ZrS2 prototype

This opt-in experiment constructs a constant molecular-orbital × physical-spin
parent space, independently of the existing spectral/SCDM projection. It does
not replace a production model, change the old exports, or delete SOC terms.

## What is fixed

The spatial functions are real S1+S2 / S1−S2 combinations of S px, py and
optionally pz coordinates, with explicit layer labels. Both spin copies use
identical spatial coefficients. These coefficients are repeated at every k and
Q. The 8-state candidate uses the single frozen radial combination
`0.8342506012922942*r1 + 0.5513854679291479*r2`; it is selected once from the
central-Q Gamma reference density summed over spin, layer, and px/py. Other
candidates retain both radial functions independently. px/py can be changed to
p+/p− by a constant spatial unitary without changing physical Sz.

| Internal states per Q | Spatial content | Dimension at 19Q |
|---:|---|---:|
| 8 | S1+S2 px/py, one frozen radial combination, both layers/spins | 152 |
| 16 | S1+S2 px/py, both radial functions, both layers/spins | 304 |
| 32 | Both S parities, px/py, both radial functions, both layers/spins | 608 |
| 48 | Both S parities, px/py/pz, both radial functions, both layers/spins | 912 |

The parent embedding F obeys F†F=I and sz F=F S0 exactly to numerical precision,
with S0 diagonal +1/−1 in the declared physical-spin order. The output column
order is Q → spin → spatial function; it is different from the old July export.
Every output carries its own basis definition and projected symmetry matrices.

"Fixed" here means fixed coefficients in **orthonormal TAPW coordinates**.
The microscopic Bloch basis and source Lowdin transformation remain k dependent.
These functions must not be described as validated localized Wannier orbitals,
and coefficient overlaps alone are not a complete microscopic form-factor or
quantum-geometry implementation.

## Three Hamiltonians and two different spin operators

Let G be an orthonormal complement, HXY=X† Horth Y and
Y=(Eref−HGG)^−1 HGF. The experiment compares:

1. Direct parent projection: HFF, with exactly fixed physical Sz=S0 within
   that truncated parent theory.
2. Fixed Schur: HFF+HFG Y. Its output coefficients are in the parent coordinates.
   The reported wavefunction overlaps for this method use **bare F lifts** and
   omit eliminated-space amplitudes; they are not full reconstructed physical
   wavefunction metrics.
3. Normalized reconstruction: M=I+Y†Y, Z=(F+GY) M^−1/2,
   Hrec=Z† Horth Z. Algebraically this equals the energy-linearized Lowdin
   expression M^−1/2 [HSchur+Eref(M−I)] M^−1/2.

The reconstructed physical spin is **Z† sz Z**, not S0. The program saves both
and reports their difference. S0 is strictly fixed in the parent model; it is
not automatically the exact microscopic observable after high-state dressing.
Even a very accurate band spectrum does not justify silently replacing Z†szZ
with S0. No self-consistent energy-dependent elimination is claimed.

All variants retain the source SOC couplings. First-order spin-flip matrix norms
are recorded, so fixed Sz cannot be mistaken for an imposed U(1) symmetry.

## Implementation and checks

- `kp/kp/experimental/fixed_spin.py`: explicit molecular basis and reductions.
- `scripts/zrs2_fixed_spin_trial.py`: 3.15°/19Q source-specific audit runner.
- `tests/kp/test_fixed_spin.py`: orbital support, spin closure, retained SOC,
  opposite-spin dressing, same-spin elimination, complete-space limit, and
  invalid-frame/pole handling.

Run the focused tests from the repository root:

```bash
PYTHONPATH=kp OPENBLAS_NUM_THREADS=1 /data/home/zy/mambaforge/bin/python -m pytest tests/kp/test_fixed_spin.py -q
```

Run only on an authorized compute node, choosing a **new absolute output path**:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
/data/home/zy/mambaforge/bin/python scripts/zrs2_fixed_spin_trial.py \
  --indices 0,40 --candidates 8,16,32,48 --threads 8 \
  --output /absolute/new/result/directory
```

The runner refuses an existing result directory. This prototype deliberately
requires the July 3.15°/19Q joint-Gamma source layout. Its source paths are
explicit constants rather than a general production configuration API.

Validation uses the existing native TAPW top-six eigenvectors, checks their
eigenpair residuals freshly, and assigns reduced bands one-to-one by physical
overlap. No energy alignment is applied. Singular values of the six-band
overlap measure subspace fidelity independently of rotations within degenerate
pairs. Individual assignment weights need not be invariant under such rotations.
Reported energy errors are sampled overlap-assigned errors, not global band
bounds. A larger parent space need not monotonically improve a fixed target-band
assignment or fixed-reference-energy approximation.

## Algorithm and manuscript impact

This is an additional basis-selection branch, not a replacement of TAPW or the
symmetry-constrained continuum construction:

- Keep the current P(k) / projected-reference construction for existing models.
  For this branch replace that construction by a specified fixed isometry F.
- Add the explicit spin-closure condition sz F=F S0, separate from Hamiltonian
  spin conservation. Retain the source symmetry closure
  Dg F*^eta=F rho_g, including its Q-channel routing.
- The Schur and energy-linearized formulas remain valid with the new low/high
  blocks. Observable reconstruction must be documented alongside them.
- Allowed-term generation and matrix-coefficient fitting retain their formulas,
  but new dimensions/representations require newly generated terms and a new fit.
- Physical topology, Wilson loops and interaction form factors still require the
  microscopic embedding and reciprocal sewing. They have not been validated by
  this experiment, and existing HF/ED inputs are not updated.

Numerical results and the limits of the attempted orbital spaces are recorded
in `docs/fixed-sz-physical-orbitals-results.md`.
