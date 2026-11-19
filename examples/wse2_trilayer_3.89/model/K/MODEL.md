# Model Description

## Scope

Standalone NumPy evaluator for `count43_spin_up_projected`.

- Material: Alternating trilayer WSe2 at 3.89 deg
- Valley: `K1`
- Spin convention: `spin_up_projected`
- Dimension: `81`
- Energy unit: `eV`
- Response semantics: `complete_linear`

Files: `evaluate.py` runs the model; `model_data.npz` stores readable physical authority files, a default integer-Q realization, frozen validation arrays, and exactified symmetry matrices.

## Fit Method

- Method: `linear`
- Hamiltonian k-point rows: `[10, 40, 50]`
- Requested bands: `8`
- One-sided / two-sided weights: `2.25` / `22.5`

## Basis And Q Sets

Basis order is qset-slot major, then orbital index, then q index:

```text
index = offset(qset) + orbital_zero_based * Nq(qset) + q_index
```

| Q-set | Offset | Q count | Orbitals | Block dim |
| --- | ---: | ---: | ---: | ---: |
| `qset1` | 0 | 27 | 2 | 54 |
| `qset2` | 54 | 27 | 1 | 27 |

Q sets are `qset1` and `qset2` in `model_data.npz`; friendly aliases
`qset_layer1` and `qset_layer2` may also be present.

## Coordinate Convention

`evaluate.py` expects k-points with shape `(Nk, 2)`.
Coordinate type: `fractional_model_basis`.
The high-symmetry points in `evaluate.py` use `fractional_model_basis` coordinates.

$$
k_{\rm phys} = k_1 b_{M1} + k_2 b_{M2}.
$$

- `bM1 = [0.14972584053, 0]`
- `bM2 = [0.0748629202651, 0.129666381502]`
- Default k path: `['G', 'M', 'K', 'G']`
- Points per segment: `20`

## Hamiltonian And Runtime Recipe

The authoritative expandable model is defined by
`model.toml` and `terms.csv`. For a row-sector integer label $n_i$, each term
routes to

$$
n_j=n_i-g_\mu,
$$

and contributes

$$
H_{ij}(k) \mathrel{+}= \beta_\mu
[(k_x-Q_{i,x})+i(k_y-Q_{i,y})]^{M_z}
[(k_x-Q_{i,x})-i(k_y-Q_{i,y})]^{M_{z^*}}.
$$

`q_points.csv` is the validated default finite realization; it is not the
authority for the abstract model. `physical_model.py` can enlarge the basis
with `set_q_shell(shell)`. The frozen `model_data.npz` response arrays remain a
reference cache used for exact export validation.

Output bands are `raw_model_eigenvalues`.
Validation alignment is `none` and is applied
to output bands: `False`.

## Terms

Active term counts: `Kinect` 27, `Onsite` 4, `intra` 225, `inter` 90; total `346`.
Kinetic (`"Kinect"` tag) terms, `Onsite`, `intra`, and `inter`
terms share the same runtime recipe. The terms in `terms.csv` are obtained by a deterministic row-centered re-expansion of the already compiled Hamiltonian polynomial; no fit, projection, or frozen-matrix inverse is performed during export.

## Symmetry

Production operations: `C3z`. The standalone evaluator does not
apply these operations dynamically; their effects are already folded into
the authoritative `terms.csv` route table.

| Operation | Antiunitary | Matrix | k action | Q action | Sector map |
| --- | --- | --- | --- | --- | --- |
| `C3z` | no | `exactified_C3z` | `k' = [[-0.5, 0.866025403784]; [-0.866025403784, -0.5]] k` | `Q' = [[-0.5, 0.866025403784]; [-0.866025403784, -0.5]] Q` | `"identity"` |

Compact matrix-element form:

$$
[\rho_g]_{(\alpha,a,q),(\beta,b,q')} =
A^g_{\alpha a,\beta b}e^{i\theta^g_{\alpha a,\beta b}}
\delta_{q,\pi^g_{\alpha a\leftarrow\beta b}(q')}.
$$

For this exported model, the actual nonzero \(\rho_g\) rules are:

The table above gives the geometric action. The exact continuum representation `rho_g` is the matrix stored under the listed `exactified_*` key. Some spinful or projected gauges produce dense representations; the array is authoritative. The rules below give the nonzero block structure and phase angles. The full q-index permutation is intentionally not expanded in this user-facing document; the `exactified_*` array is the authoritative matrix.

Use the basis label \(i=(\alpha,a,q)\), where \(\alpha\) is the qset slot, \(a\) is the orbital index, and \(q\) is the Q-index inside that qset. The flat basis index is

$$
i(\alpha,a,q)=\mathrm{offset}(\alpha)+aN_Q(\alpha)+q.
$$

Each nonzero block rule has the form

$$
[\rho_g]_{(\alpha,a,q),(\beta,b,q')} =
A^g_{\alpha a,\beta b}e^{i\theta^g_{\alpha a,\beta b}}\,
\delta_{q,\pi^g_{\alpha a\leftarrow\beta b}(q')}.
$$

All omitted matrix elements are zero.

`sector_map` is descriptive metadata. The equations below, or the `exactified_*` array, define the actual representation including phase angle, orbital exchange, spin exchange, and the projection gauge.

### `C3z`
Matrix key: `exactified_C3z`; shape `(81, 81)`; nonzero entries `81`.
Phase-permutation form: `yes`.

- Nonzero block: `(qset1, orbital 0) -> (qset1, orbital 0)`; matrix-element magnitude `1`; phase angle `pi/3`; count `27`.

  $$
  [\rho_{\mathrm{C3z}}]_{(\mathrm{qset1},0,q),(\mathrm{qset1},0,q')} =
  e^{i(\pi/3)}\,\delta_{q,\pi(q')}.
  $$
- Nonzero block: `(qset1, orbital 1) -> (qset1, orbital 1)`; matrix-element magnitude `1`; phase angle `pi/3`; count `27`.

  $$
  [\rho_{\mathrm{C3z}}]_{(\mathrm{qset1},1,q),(\mathrm{qset1},1,q')} =
  e^{i(\pi/3)}\,\delta_{q,\pi(q')}.
  $$
- Nonzero block: `(qset2, orbital 0) -> (qset2, orbital 0)`; matrix-element magnitude `1`; phase angle `pi/3`; count `27`.

  $$
  [\rho_{\mathrm{C3z}}]_{(\mathrm{qset2},0,q),(\mathrm{qset2},0,q')} =
  e^{i(\pi/3)}\,\delta_{q,\pi(q')}.
  $$

## Validation

- Reference kind: `current_heff_support_mask`
- Reference arrays: `{'model_eigvals': 'model_data.npz:reference_eigvals', 'primary_current_heff_support_eigvals': 'model_data.npz:reference_current_heff_support_eigvals'}`
- Band window: `[73, 81]`
- Compared bands: `8`
- K points: `61`
- Alignment: `none`
- RMS error: `1.0989782363386136` meV
- Max error: `3.0989941478516414` meV
- Full TAPW validation: `False`

Missing fields are recorded as `null` in this report; no validation numbers are
invented during export.

## Limitations

This package uses the current two-qset KP core. Custom k-points must use the
recorded coordinate convention. Exactified matrices are included for diagnostics;
physical runtime evaluation uses integer Q routes from `model.toml` and `terms.csv`. The default
`MODEL.md` includes compact nonzero block rules needed to reconstruct
the exported symmetry matrices. Use `debug_files=True` only for separated
developer-facing metadata files.
