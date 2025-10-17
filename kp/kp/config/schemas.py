from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MaterialConfig:
    """Material-specific inputs and structural choices (skeleton)."""

    hamk_file: str
    qset1_file: Optional[str] = None
    qset2_file: Optional[str] = None
    orbital_order_file: Optional[str] = None
    valley: str = "K1"
    spin: str = "all"
    q_shells: int | None = None
    truncate_shells: int | None = None
    num_layers: int | None = None
    num_orb_per_layer: List[int] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Run-time configuration for selection, symmetry, and fitting (skeleton)."""

    energy_window: tuple[float, float] = (-0.1, 0.1)
    fermi_ref: str = "auto"
    selection_score: str = "max"
    top_k: int = 6
    symmetry_ops: Dict[str, List[Dict]] = field(default_factory=dict)
    orthogonalize_tol: float = 1e-8
    fit_tol: float = 1e-8
    workers: int = 4

