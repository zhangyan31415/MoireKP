# KP Project Auto Gauge Anchors

`kp project` builds a finite-basis projector after the user chooses a low-energy
block subspace. The low-energy choice and the gauge anchors are separate
concepts.

## Recommended Config

For ordinary use, write `nlow_state_list` and request automatic gauge anchors:

```yaml
project:
  nlow_state_list:
    - [54]
    - [55]
  gauge: auto
```

The expanded form is:

```yaml
project:
  gauge:
    method: auto_scdm
    anchor_scope: per_sector
    candidate_pool: all
    projected_anchors: false
    min_sigma: 1.0e-6
    max_condition: 1.0e6
```

`norb_fix_list: auto` is accepted as a legacy alias for `gauge: auto`.

## Meaning Of The Fields

`nlow_state_list` contains block eigenstate indices. It does not count orbitals.
For multilayer inputs it must use physical-layer rows, with `[]` for inactive
physical layers.

`norb_fix_list` is an expert-only manual gauge-anchor list. It defines trial
projectors used to fix phases and rotations inside the selected low-energy
subspace. It is not itself the physical low-energy model.

Auto gauge does not choose the physical model. It only chooses stable finite
basis rows after `nlow_state_list` has already selected the low subspace.

## Algorithm

For an orthonormal finite basis, auto gauge forms the low-subspace row scores

```text
P = U U^dagger
leverage_i = P_ii
```

For a future overlap-metric path, the API can compute row scores after a Lowdin
transform,

```text
U_tilde = S^(1/2) U
leverage_i = (U_tilde U_tilde^dagger)_ii
```

The selector uses pivoted QR on the candidate rows of `U_tilde^T` or `U^T`.
The selected rows must make the trial-overlap matrix full rank. The report
records singular values, `sigma_min`, and condition number. The existing
Procrustes alignment then fixes the low-subspace gauge:

```text
O = Phi_ref^dagger U_low
O = X Sigma Y^dagger
V = Y X^dagger
U_aligned = U_low V
```

When the same source config also contains a usable `symm` section
(`symm.tapw_symmetry_dir` and `symm.operations`), auto gauge generates a small
deterministic candidate set, projects each candidate through the TAPW raw-H
symmetry actions, exactifies in memory, and scores the candidates by:

- exactification residual;
- phase-branch stability;
- off-support leakage;
- active-term count when available;
- deterministic model-frame/layout priority only when the numerical metrics
  cannot distinguish candidates.

If every candidate fails, or if indistinguishable candidates remain after all
metrics and priorities, the command fails instead of silently writing a model.
`kp project` and `kp symm` use this same resolver, so `projection/heff.npy` and
the symmetry representations are produced in the same resolved gauge.

## Outputs

`kp project` writes these canonical files in `projection/`:

- `heff.npy`
- `eigvals.txt`
- `wavefunctions.npz`
- `basis.npz`
- `basis.md`
- `scatter.pdf`

The JSON report separates:

- `state_selection_quality`: whether the configured `nlow_state_list` was
  assessed. Phase 1 records this as not evaluated by auto gauge.
- `gauge_anchor_quality`: rank, singular values, and conditioning of the
  selected anchors.
- `symmetry_closure_quality`: symmetry validation status. Missing symmetry
  information is reported as `null` with `not_available`, never as zero. When
  symmetry validation is available this section records the selected candidate,
  rejected candidates, exactification residuals, phase metrics, and support
  leakage.

## Failure Recovery

If auto gauge fails, first inspect `basis_selection.md` and then:

- check that `nlow_state_list` selects a symmetry-closed low subspace;
- include all low-energy partners needed for the model;
- check `material.num_layer_list`, `num_orb_per_layer`, spin, and Q-set layout;
- use manual `norb_fix_list` only as an expert override.

Manual `norb_fix_list` and `gauge: auto` cannot be combined in the same config.
