from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class TAPWPaths:
    """Container for TAPW output file paths.

    Fields
    - hamk_file: path to k-space Hamiltonian (.npy)
    - qset1_file/qset2_file: Q reciprocal-lattice sets for two layers
    - orbital_order_file: CSV/JSON mapping orbital indices to semantic labels
    """

    hamk_file: str
    qset1_file: Optional[str] = None
    qset2_file: Optional[str] = None
    orbital_order_file: Optional[str] = None


def load_hamk(path: str, mmap_mode: str | None = "r") -> np.ndarray:
    """Load k-space Hamiltonian array from .npy.

    Returns
    - hamk: np.ndarray, expected shape (n_k, n, n) or (n, n) for a single k
    Note: uses memory mapping by default to avoid loading huge arrays at once.
    """

    return np.load(path, mmap_mode=mmap_mode)


def load_Q_sets(path1: str, path2: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load Q sets for layer-1/2 from .npy files.

    Returns
    - Q_set1, Q_set2 (optional)
    """

    q1 = np.load(path1)
    q2 = np.load(path2) if path2 is not None else None
    return q1, q2


def load_orbital_order(path: str) -> List[Dict[str, Any]]:
    """Load orbital ordering metadata (CSV/JSON).

    Expected columns/keys include: index, name, layer, atom, l, m, spin(optional).
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("orbitals", data.get("orbital_order", data))
        if not isinstance(data, list):
            raise ValueError("orbital order JSON must contain a list of orbital records")
        return [_coerce_record(dict(row)) for row in data]
    if suffix in {".csv", ".txt"}:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"orbital order CSV has no header: {path}")
            return [_coerce_record(dict(row)) for row in reader]
    raise ValueError(f"Unsupported orbital order file extension {source.suffix!r}; use CSV or JSON")


def _coerce_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): _coerce_value(value) for key, value in record.items()}


def _coerce_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return ""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
