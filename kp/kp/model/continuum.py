from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class ContinuumTermKey:
    """Key describing a continuum term (skeleton)."""

    layer_pair: tuple[int, int]
    delta_q: tuple[int, int]
    tag: str = "intra"


@dataclass
class ContinuumTerm:
    """Continuum term with basis and symmetry annotations (skeleton)."""

    key: ContinuumTermKey
    Y_basis: Callable[[np.ndarray], np.ndarray]
    r_value_real: float = 0.0
    r_value_imag: float = 0.0
    tag: str = "intra"
    symmetry_ops: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ContinuumModel:
    """Continuum model as a collection of terms (skeleton)."""

    terms: Dict[ContinuumTermKey, ContinuumTerm] = field(default_factory=dict)

    def add_term(
        self,
        key: ContinuumTermKey,
        Y_basis: Callable[[np.ndarray], np.ndarray],
        *,
        tag: str = "intra",
        symmetry_ops: List[Dict[str, Any]] | None = None,
    ) -> None:
        raise NotImplementedError

    def assemble_hamiltonian(self, k: np.ndarray, symmetry_gen: Any, use_cache: bool = True) -> np.ndarray:
        raise NotImplementedError

