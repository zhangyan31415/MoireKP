"""Finite-basis selection utilities for KP projection."""

from .selection import (
    AnchorCandidate,
    AutoGaugeConfig,
    AutoGaugeSelection,
    GaugeCandidateSymmetryDecision,
    GaugeCandidateSymmetryMetrics,
    GaugeAnchorReport,
    build_reference_projectors_from_rows,
    compute_leverage_scores,
    polar_align_low_subspace,
    select_anchor_rows_qrcp,
    select_gauge_candidate_by_symmetry,
    write_basis_selection_report,
)

__all__ = [
    "AnchorCandidate",
    "AutoGaugeConfig",
    "AutoGaugeSelection",
    "GaugeCandidateSymmetryDecision",
    "GaugeCandidateSymmetryMetrics",
    "GaugeAnchorReport",
    "build_reference_projectors_from_rows",
    "compute_leverage_scores",
    "polar_align_low_subspace",
    "select_anchor_rows_qrcp",
    "select_gauge_candidate_by_symmetry",
    "write_basis_selection_report",
]
