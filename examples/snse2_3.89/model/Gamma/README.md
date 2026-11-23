# SnSe2 Gamma continuum model

`polynomial.npz` and `coefficients.npy` store the archived finite-Q model. Use `evaluate.load_model().bands(kpoints)` to evaluate the model at Cartesian momenta in inverse angstrom. The eigenvalues are in eV.

For the 61-point comparison, use
`np.load("reference_bands.npz")["k_cartesian_inv_angstrom"]` as the argument to
`bands`. The original construction spectrum in `eigvals.npy` has 31 rows.
