from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class SymmetryGenerator:
    """Minimal public symmetry-generator facade.

    This helper intentionally does not synthesize physical symmetry matrices.
    Production code should consume explicit `kp symm` artifacts or use the
    continuum-model generator in `kp.model.core`, where basis conventions are
    available.
    """

    Q_set1: np.ndarray
    Q_set2: np.ndarray | None = None
    nlow_state: List[int] | None = None
    cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def _dimension(self) -> int:
        nlow = self.nlow_state or [1, 1]
        q1 = len(np.asarray(self.Q_set1))
        q2 = len(np.asarray(self.Q_set2)) if self.Q_set2 is not None else 0
        if len(nlow) == 1:
            return q1 * int(nlow[0])
        return q1 * int(nlow[0]) + q2 * int(nlow[1])

    def get_C3z_operator(self, params: int) -> np.ndarray:
        """C3z requires basis-specific metadata and is not inferred here."""
        raise ValueError("C3z requires explicit symmetry matrices or kp.model.core.SymmetryGenerator")

    def get_C2T_operator(self) -> np.ndarray:
        """C2T requires basis-specific metadata and is not inferred here."""
        raise ValueError("C2T requires explicit symmetry matrices or kp.model.core.SymmetryGenerator")

    def get_time_reversal_matrix(self) -> np.ndarray:
        """Time reversal requires basis-specific metadata and is not inferred here."""
        raise ValueError("TR requires explicit symmetry matrices or kp.model.core.SymmetryGenerator")

    def get_operator(self, name: str, params: Any | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (k_transform, D) for a named symmetry operation."""
        op = str(name)
        if op in {"I", "identity"}:
            return np.eye(2), np.eye(self._dimension(), dtype=np.complex128)
        if op == "C3z":
            return np.eye(2), self.get_C3z_operator(1 if params is None else int(params))
        if op in {"TR", "time_reversal"}:
            return -np.eye(2), self.get_time_reversal_matrix()
        if op in {"C2", "C2z"}:
            raise ValueError("C2 requires explicit symmetry matrices or kp.model.core.SymmetryGenerator")
        if op == "C2T":
            return np.eye(2), self.get_C2T_operator()
        raise ValueError(f"Unknown symmetry operation: {name}")
