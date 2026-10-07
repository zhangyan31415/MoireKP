# Separate ABACUS CSR converter

`analysis_abacus.py` is vendored from the GPL-3.0-only
`abacus-wanniertools-hs-kit/tools/analysis_abacus.py` utility. Its complete license
is in `LICENSE.GPL-3.0`. This file remains separately licensed; it is not relicensed
under the TAPW package's license. The importer executes it as a separate Python
process and exchanges matrix files and JSON metadata.

Source checkout used for this snapshot:
`/data/work/zy/tmp/abacus_hr_analysis/abacus-wanniertools-hs-kit`.
The converter supports legacy `data-HR/SR-sparse_SPIN*.csr` and current
`hrs*_nao.csr` / `srs1_nao.csr` matrix families.

Local modification, 2026-09-23: H and S, and the two collinear H channels,
may contain different translation supports. Conversion streams their independent
blocks, treating missing blocks as zero. It continues rejecting duplicate R
blocks within one matrix and inconsistent dimensions or ionic steps. Output
translation count records the union. This corrects a failure found with actual
Si output (55 H translations versus 43 S translations).

Local modification, 2026-09-23: sparse value serialization uses 17 significant
digits instead of 7 fixed decimal places. This preserves small overlap entries
and input precision through the intermediate files. The original formatting
introduced a 2.18e-5 eV error in Si Gamma eigenvalues despite zero cutoffs.
