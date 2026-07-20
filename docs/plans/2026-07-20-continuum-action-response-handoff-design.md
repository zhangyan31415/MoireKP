# Continuum-Action Response Handoff Design

## Problem

`kp symm` preserves two different symmetry descriptions:

- the TAPW source action, which records the physical action before projection;
- the exactified continuum action, which records how the same operation acts in
  the selected low-energy model basis.

These descriptions normally have the same discrete sector support.  They need
not agree after projection.  In the PtSe2 Gamma q04 example, time reversal is
sector preserving in the TAPW source description, while the selected bands 54
and 55 form a two-slot Kramers basis in which the exactified continuum action
exchanges the slots.

The response-basis compiler currently constructs its finite group from source
`sector_map` and `q_map` metadata, then separately loads the factorized action
from the exactified continuum matrix.  For PtSe2 this produces two different Q
permutations for the same TR generator.  The fast factorized compiler correctly
rejects the mismatch and falls back to dense Reynolds compilation.  The
fallback preserves correctness but makes every clean-cache run unnecessarily
slow.

## Constraints

- Do not add material, valley, band-index, or PtSe2-specific branches.
- Do not change the public YAML interface.
- Preserve the TAPW source action unchanged as physical provenance.
- Continue to use only `kp symm` exactified continuum matrices in production.
- Fail closed when projected action metadata is incomplete or inconsistent.
- Preserve the dense fallback when no certified projected/factorized action is
  available.

## Chosen Design

The response compiler will resolve each finite-group generator in this order:

1. Read the physical momentum action from the operation metadata.  Projection
   does not redefine the physical Cartesian k action.
2. If a complete `model_basis_action.items` mapping is present, derive the
   discrete Q and continuum-sector permutations from those items.  Require
   every active source Q slot exactly once, a bijective target mapping, valid
   sector names and indices, and consistency with the recorded
   `model_basis_action.sector_map`.
3. If a certified factorized action is present, require its antiunitary parity,
   Cartesian k action, Q permutation, and sector permutation to agree with the
   projected mapping.  The existing matrix certification continues to compare
   the factorized matrix with the active exactified continuum matrix.
4. If no complete projected mapping is available, retain the current
   source-metadata construction and dense fallback behavior.

This changes only the model-side representation of an already exactified
operation.  It does not rewrite the source action, infer a material profile, or
ask the user to approve an internal basis permutation.

## Data Flow

```text
TAPW source action
    -> provenance and source covariance validation

kp project selected subspace
    -> kp symm exactified continuum matrix
    -> complete model_basis_action.items
    -> certified factorized action
    -> kp model finite group and response basis
```

The exactified continuum matrix remains the production authority.  The
projected discrete action and factorized representation must describe that same
matrix and basis ordering.

## Failure Handling

- Incomplete projected Q items: reject that projected mapping rather than fill
  missing entries heuristically.
- Non-bijective projected mapping: raise a precise action-metadata error.
- Factorized/projected permutation mismatch: fail closed before expensive cold
  compilation.
- Factorized/exactified matrix mismatch: retain the existing certification
  failure/fallback behavior.
- No projected or factorized metadata: preserve legacy source-action behavior.

## Expected Compatibility

For ordinary cases where source and projected actions agree, the derived
permutations are identical and fitted models do not change.  Cases like PtSe2,
where the selected low-energy basis changes block support, use the certified
continuum permutation and no longer enter the TR factorized fallback.

## Verification

1. A unit test reproduces source-identity/projected-exchange TR and proves that
   the finite group uses the projected Q permutation.
2. Tests reject incomplete, non-bijective, or factorized-inconsistent projected
   mappings.
3. Existing response-basis, symmetry, configured-model, and standalone-export
   tests remain green.
4. PtSe2 Gamma q04 is run from an empty response cache.  Its log must contain
   certified factorized TR compilation and must not contain `Q permutation
   mismatch` or `finite-p factorized sparse Reynolds fallback`.
5. The fresh fitted coefficients, band errors, overlap, and symmetry residuals
   are compared with the previous certified output.
