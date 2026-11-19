# CaI2 AB Gamma valence at 3.89024 degrees

The twenty real coefficients were optimized for this angle using its own training
data. The six-state basis is shared between angles. `MODEL.json` records the
training indices. `COEFFICIENTS.npz` stores the contracted polynomial and Fourier
matrices, and `MATERIAL_CONTROLS.npz` stores the twenty physical coefficients.
The fixed orbitals and operator templates are included alongside the export.

Use `evaluate.load_model().bands(kpoints)` for all electron energies in eV
relative to the model valence edge at Gamma. Input momenta are Cartesian,
in inverse angstrom. TAPW references in `reference_bands.npz` use their own
Gamma valence edge. RMS and maximum errors use the 45 held-out path points.
The columns of `pair_overlap_squared` in `overlap_samples.npz` run from the
highest to the lowest valence pair at the indicated sampled momenta.
