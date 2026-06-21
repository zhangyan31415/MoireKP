"""Blocks: low-energy subspace and diagonalization utilities."""

from .blocks import (
    align_eigenstates,
    get_H_block,
    project_heff_full,
    resolve_project_gauge_anchors,
    set_projector_blas_threads,
)

__all__ = [
    "get_H_block",
    "align_eigenstates",
    # "get_h_dft_low",
    "project_heff_full",
    "resolve_project_gauge_anchors",
    "set_projector_blas_threads",
]
