from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class KPathGenerator:
    """High-symmetry k-path generator interface.

    Mirrors the notebook's KPathGenerator responsibilities:
    - read labels + points
    - convert between direct/cart coordinates
    - interpolate segments and export k-path
    """

    labels: List[str] = field(default_factory=list)
    high_symmetry_points: np.ndarray | None = None
    kpoints: List[np.ndarray] = field(default_factory=list)

    @staticmethod
    def calculate_reciprocal_vectors(Amat: np.ndarray) -> np.ndarray:
        """Compute reciprocal lattice vectors from a real-space lattice."""
        lattice = np.asarray(Amat, dtype=float)
        if lattice.ndim != 2 or lattice.shape[0] != lattice.shape[1]:
            raise ValueError(f"Amat must be a square matrix, got shape {lattice.shape}")
        return 2.0 * np.pi * np.linalg.inv(lattice).T

    @staticmethod
    def direct_cart_real(Amat: np.ndarray, pos_direct: np.ndarray) -> np.ndarray:
        """Convert direct coordinates to cartesian coordinates using a real-space lattice."""
        lattice = np.asarray(Amat, dtype=float)
        direct = np.asarray(pos_direct, dtype=float)
        return direct @ lattice

    def read_and_generate_kpath(self, file_path: str, output_file_path: str) -> None:
        """Read a k-path specification file and export the interpolated path."""
        labels: list[str] = []
        points: list[np.ndarray] = []
        counts: list[int] = []
        with Path(file_path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.split("#", maxsplit=1)[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 3:
                    raise ValueError(f"k-path line must contain label and two coordinates: {raw_line.rstrip()!r}")
                labels.append(parts[0])
                points.append(np.array([float(parts[1]), float(parts[2])], dtype=float))
                counts.append(int(parts[3]) if len(parts) > 3 else 20)
        if len(points) < 2:
            raise ValueError("k-path specification requires at least two points")
        self.labels = labels
        self.high_symmetry_points = np.asarray(points, dtype=float)
        generated: list[np.ndarray] = []
        for start, stop, count in zip(points[:-1], points[1:], counts[:-1]):
            if count <= 0:
                raise ValueError(f"k-path segment counts must be positive, got {count}")
            segment = np.linspace(start, stop, count + 1)
            if generated:
                segment = segment[1:]
            generated.extend(np.asarray(row, dtype=float) for row in segment)
        self.kpoints = generated
        output = Path(output_file_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, np.asarray(generated, dtype=float), fmt="%.12g")
