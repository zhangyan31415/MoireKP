# User-facing linear and nonlinear `kp model` interface

## Goal

Replace the current user-visible mixture of `fit.objective`, `refine_bands`,
solver, Jacobian, compression, variable-tag, sigma, and acceptance settings
with one explicit model fit method:

```yaml
model:
  fit:
    method: linear  # or nonlinear
```

One invocation writes one requested model. There are no implicit `low` or
`high` profiles. A nonlinear fit always starts from the corresponding linear
solution.

## Low-energy window

Let

\[
\Delta H_\theta(k)=H_\theta(k)-H_{\mathrm{eff}}(k),
\]

where the full Hamiltonian dimension is \(d\). `model.fit.bands` requests
\(n\) bands at `model.target_bands`, either `top` or `bottom`. The target
Hamiltonian eigenvectors define

\[
P_k=\sum_{j\in\mathcal W_k}
|u^{\mathrm{eff}}_{jk}\rangle
\langle u^{\mathrm{eff}}_{jk}|.
\]

The requested window is completed across a degeneracy at its boundary. The
CLI must report both the requested and resolved band counts.

## Normalized residuals

The public weights multiply dimension-normalized mean-square residuals:

\[
R_H(k)=\frac{\|\Delta H_\theta(k)\|_F^2}{d^2},
\]

\[
R_{\mathrm{1s}}(k)=
\frac{\|\Delta H_\theta(k)P_k\|_F^2}{dn},
\]

\[
R_{\mathrm{2s}}(k)=
\frac{\|P_k\Delta H_\theta(k)P_k\|_F^2}{n^2}.
\]

The base Hamiltonian coefficient is fixed to one and is not configurable.

## Linear method

For the required Hamiltonian fit rows \(K_H\),

\[
L_{\mathrm{linear}}(\theta)=
\left\langle
R_H+w_{\mathrm{1s}}R_{\mathrm{1s}}+
w_{\mathrm{2s}}R_{\mathrm{2s}}
\right\rangle_{k\in K_H}.
\]

The user-facing mapping is:

- \(K_H\): `fit.kpoints`
- \(n\): `fit.bands`
- \(w_{\mathrm{1s}}\): `fit.one_sided_weight`
- \(w_{\mathrm{2s}}\): `fit.two_sided_weight`

Numerical ridge regularization remains an internal implementation detail and
is recorded in diagnostics.

```yaml
model:
  target_bands: top
  harmonics: {intralayer: 4, interlayer: 3}
  max_order: {kinetic: 14, intralayer: 8, interlayer: 9}
  fit:
    method: linear
    kpoints: [0, 20, 40]
    bands: 10
    one_sided_weight: 1.0
    two_sided_weight: 1.0
```

## Nonlinear method

The nonlinear method first computes

\[
\theta_0=\operatorname*{argmin}_\theta L_{\mathrm{linear}}(\theta).
\]

For the explicitly configured band-loss rows \(K_E\), define

\[
R_{\mathrm{band}}(k)=\frac{1}{n}
\sum_{j\in\mathcal W_k}
\left[\varepsilon_j^\theta(k)-
\varepsilon_j^{\mathrm{eff}}(k)\right]^2.
\]

Starting from \(\theta_0\), minimize

\[
L_{\mathrm{nonlinear}}(\theta)=
\left\langle
R_H+w_{\mathrm{1s}}R_{\mathrm{1s}}+
w_{\mathrm{2s}}R_{\mathrm{2s}}
\right\rangle_{k\in K_H}
+w_{\mathrm{band}}
\left\langle R_{\mathrm{band}}\right\rangle_{k\in K_E}.
\]

The nonlinear-only mapping is:

- \(K_E\): `fit.band_kpoints`
- \(w_{\mathrm{band}}\): `fit.band_loss_weight`
- iteration cap: `fit.max_steps`, default 30

```yaml
model:
  target_bands: top
  harmonics: {intralayer: 4, interlayer: 3}
  max_order: {kinetic: 14, intralayer: 8, interlayer: 9}
  fit:
    method: nonlinear
    kpoints: [0, 20, 40]
    band_kpoints: all
    bands: 10
    one_sided_weight: 1.0
    two_sided_weight: 1.0
    band_loss_weight: 1.0
    max_steps: 30
```

## Validation rules

- `method` must be `linear` or `nonlinear`.
- `kpoints` is required, non-empty, unique, and in range.
- `bands` is positive and does not exceed the Hamiltonian dimension before
  degeneracy completion.
- `one_sided_weight` and `two_sided_weight` are required, finite, and
  non-negative.
- `band_kpoints` and `band_loss_weight` are required for `nonlinear` and
  rejected for `linear`.
- `band_kpoints` accepts `all` or a non-empty unique index list.
- `band_loss_weight` is finite and positive.
- `max_steps` is nonlinear-only and positive when present.

## Internal resolution

The public contract resolves to the existing symmetry-complete response basis
and solver machinery. Users do not configure response semantics, target
reference, spectral floor, normalization mode, solver name, optimizer,
Jacobian, matrix compression, variable tags, real/imag components, sigma, or
acceptance line search.

The resolved configuration is written to model diagnostics. Legacy advanced
configurations remain readable during a deprecation period, but mixing the new
`fit.method` contract with legacy `fit.objective` or `fit.refine_bands` is an
error rather than a precedence rule.

## Output behavior

One run exports one model. Nonlinear initialization is diagnostic state, not a
second exported model. The CLI reports:

- requested and resolved band windows;
- Hamiltonian and band k-point rows;
- all three user weights;
- linear-initial and final nonlinear band, matrix, and overlap metrics;
- a warning when nonlinear refinement improves the selected bands while
  materially degrading full-band or overlap diagnostics.

The requested nonlinear result is not silently replaced by the linear initial
model.

## Testing

Parser tests cover both valid YAML forms, method-specific rejection, band and
k-point bounds, and legacy/new conflicts. Solver tests use a small Hermitian
oracle for all three normalized matrix residuals and verify that nonlinear
starts from the exact linear coefficients before adding the eigenvalue loss.
External PtSe2 validation compares energy, matrix, and overlap metrics without
placing generated artifacts in the release surface.
