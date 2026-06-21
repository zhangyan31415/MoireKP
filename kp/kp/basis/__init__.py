"""Finite-basis selection utilities for KP projection."""

from .selection import (
    AnchorCandidate,
    AutoGaugeConfig,
    AutoGaugeSelection,
    GaugeAnchorReport,
    build_reference_projectors_from_rows,
    compute_leverage_scores,
    polar_align_low_subspace,
    select_anchor_rows_qrcp,
    write_basis_selection_report,
)

__all__ = [
    "AnchorCandidate",
    "AutoGaugeConfig",
    "AutoGaugeSelection",
    "GaugeAnchorReport",
    "build_reference_projectors_from_rows",
    "compute_leverage_scores",
    "polar_align_low_subspace",
    "select_anchor_rows_qrcp",
    "write_basis_selection_report",
]
