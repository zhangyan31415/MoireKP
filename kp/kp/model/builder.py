from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .continuum import ContinuumModel, ContinuumTermKey


@dataclass
class ContinuumModelBuilder:
    """Builder for continuum models: basis, symmetrization, orthogonalization, fitting (skeleton)."""

    symmetry_gen: Any
    symmetry_map: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # optional global cache for symmetrization
    _SYMMETRIZE_GLOBAL_CACHE: Dict[Any, Any] | None = None

    def build_terms(self) -> None:
        """Construct basis terms according to symmetry_map/config (placeholder)."""
        raise NotImplementedError

    def symmetrize_Y_basis(self, Y_basis, k: np.ndarray, sym_ops: List[Dict[str, Any]]):
        """Return symmetrized basis for k (placeholder)."""
        raise NotImplementedError

    def orthogonalize_hermitian_matrices(self, matlist: List[np.ndarray], tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """Orthogonalize a list of Hermitian matrices (placeholder)."""
        raise NotImplementedError

    def compute_coefficients_by_tag(self, heff: np.ndarray, k_points: List[np.ndarray], tol: float = 1e-8) -> Dict[str, Dict[Any, np.ndarray]]:
        """Fit coefficients for each tag using least squares (placeholder)."""
        raise NotImplementedError

