"""Logging helpers for the continuum model."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger("kp.model.core")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for command-line workflows."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def summarize_symmetry_ops(symm: Sequence[Mapping[str, Any]] | Sequence[Any]) -> str:
    names: list[str] = []
    for op in symm:
        if isinstance(op, Mapping):
            name = str(op.get("name", op.get("operation", "?")))
            matrix_kind = op.get("matrix_kind")
            source = op.get("source")
            suffix = []
            if source:
                suffix.append(str(source))
            if matrix_kind:
                suffix.append(str(matrix_kind))
            if suffix:
                name = f"{name}({','.join(suffix)})"
            names.append(name)
        else:
            names.append(str(op))
    return "[" + ", ".join(names) + "]"


def summarize_coefficients(values: Sequence[Any] | np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.complex128).ravel()
    if arr.size == 0:
        return "count=0"
    abs_arr = np.abs(arr)
    return (
        f"count={arr.size}, nonzero={int(np.count_nonzero(abs_arr > 0.0))}, "
        f"max_abs={float(np.max(abs_arr)):.6g}, median_abs={float(np.median(abs_arr)):.6g}"
    )
