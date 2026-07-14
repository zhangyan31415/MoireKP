# Current-support primary comparison design

## Goal

When `model.harmonics` restricts the continuum support, report how faithfully the
fitted model represents the correspondingly support-masked projected Hamiltonian.
Do not mix the error caused by the selected model vocabulary with the separate
error caused by truncating the original projected Hamiltonian.

## References

For the selected band-path rows, define

\[
H_{\mathrm{support}}(k)
=P_{\mathrm{support}}\odot H_{\mathrm{eff}}(k),
\]

followed by explicit Hermitian projection. The support mask is the same mask
already constructed from the resolved `model.harmonics` for the harmonic
recommendation and Hamiltonian-element diagnostics.

The two comparison references have distinct roles:

- Primary reference: `current_heff_support_mask`, using
  \(H_{\mathrm{support}}\).
- Secondary reference: `original_heff`, using the unmasked projected
  \(H_{\mathrm{eff}}\), retained only as an explicit truncation/total-error
  diagnostic.

## Primary outputs

The following outputs use eigensystems of \(H_{\mathrm{support}}\):

- `band_comparison.pdf`;
- per-band wavefunction-overlap coloring;
- unaligned RMS and maximum band error;
- aligned plot-window RMS, maximum error, and alignment shift;
- all-band RMS and maximum error;
- the corresponding `comparison`, `plot_comparison`, and
  `all_band_plot_comparison` result records.

The plot uses the current support target as `Reference`; it must not draw that
same target a second time as a separate orange curve. Overlap clustering and
degeneracy handling use the model eigensystem and the support-target eigensystem.

## Secondary outputs

The original projected Heff remains available without affecting primary metrics:

- `band_comparison_vs_full_heff.pdf`;
- `comparison_vs_full_heff`;
- `plot_comparison_vs_full_heff`;
- `all_band_plot_comparison_vs_full_heff`.

CLI labels must state the reference explicitly. Primary metric labels end in
`vs current Heff support mask`; secondary metrics, when printed, end in
`vs original Heff`. The secondary output prevents a good support-fit score from
hiding a large harmonic-truncation error.

## Failure behavior

If the resolved support mask cannot be constructed, a run that requests Heff
comparison fails closed. It must not silently fall back to the original Heff as
the primary reference, because that changes the meaning of the public metrics.

## Result metadata

Comparison records and `run_summary.json` carry an explicit reference identifier:

```text
current_heff_support_mask
original_heff
```

This makes downstream reports independent of filenames or legend text.

## Tests

Regression tests cover:

1. primary energy metrics use support-target eigenvalues;
2. overlap weights use support-target eigenvectors;
3. primary plot receives one support reference, not duplicate full/support
   curves;
4. secondary full-Heff comparison remains available and is explicitly labelled;
5. missing support construction fails closed;
6. CLI and `run_summary.json` expose the reference identities.

