"""NumPy-only runtime for ``model.toml`` plus ``terms.csv`` exports.

This file is intentionally self-contained so the exporter can copy it into a
standalone model directory without importing :mod:`kp`.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tomllib

import numpy as np


def _hexagonal_labels(shell: int) -> list[tuple[int, int]]:
    radius = int(shell)
    if radius < 0:
        raise ValueError("Q shell must be nonnegative")
    return [
        (n1, n2)
        for n1 in range(-radius, radius + 1)
        for n2 in range(-radius, radius + 1)
        if max(abs(n1), abs(n2), abs(n1 + n2)) <= radius
    ]


class PhysicalModel:
    """Route-compiled Hamiltonian defined by readable physical term files."""

    def __init__(self, root: Path, config: dict, terms: list[dict], q_rows: list[dict]):
        self.root = Path(root)
        self.config = config
        self.terms = terms
        self._basis = np.asarray(
            (config["q_lattice"]["b1"], config["q_lattice"]["b2"]),
            dtype=float,
        )
        self._sectors = tuple(config["sectors"])
        self._sector_by_id = {str(row["id"]): row for row in self._sectors}
        if len(self._sector_by_id) != len(self._sectors):
            raise ValueError("sector ids must be unique")
        self._lattice_tolerance = float(
            config.get("validation", {}).get("lattice_tolerance", 1.0e-9)
        )
        if not np.isfinite(self._lattice_tolerance) or self._lattice_tolerance <= 0.0:
            raise ValueError("validation.lattice_tolerance must be positive and finite")
        self._set_q_rows(q_rows)

    @classmethod
    def load(cls, root: str | Path = ".") -> "PhysicalModel":
        directory = Path(root)
        with (directory / "model.toml").open("rb") as handle:
            config = tomllib.load(handle)
        if config.get("schema_version") != "moirekp-physical-v1":
            raise ValueError("unsupported physical model schema")
        runtime = config["runtime"]
        with (directory / runtime["terms_file"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            terms = [dict(row) for row in csv.DictReader(handle)]
        with (directory / runtime["default_q_points_file"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            q_rows = [dict(row) for row in csv.DictReader(handle)]
        return cls(directory, config, terms, q_rows)

    @property
    def dimension(self) -> int:
        return int(self._dimension)

    @property
    def route_count(self) -> int:
        return int(self._route_rows.size)

    def _set_q_rows(self, q_rows: list[dict]) -> None:
        labels_by_sector: dict[str, list[tuple[int, int]]] = {
            str(sector["id"]): [] for sector in self._sectors
        }
        for row in q_rows:
            sector_id = str(row["sector"])
            if sector_id not in labels_by_sector:
                raise ValueError(f"q_points.csv uses unknown sector {sector_id!r}")
            label = (int(row["n1"]), int(row["n2"]))
            offset = np.asarray(self._sector_by_id[sector_id]["q_offset"], dtype=float)
            reconstructed = offset + np.asarray(label, dtype=float) @ self._basis
            saved = np.asarray((float(row["q_x"]), float(row["q_y"])))
            if np.linalg.norm(reconstructed - saved) > self._lattice_tolerance:
                raise ValueError("q_points.csv is inconsistent with model.toml")
            labels_by_sector[sector_id].append(label)
        self._labels_by_sector = {
            sector_id: tuple(labels) for sector_id, labels in labels_by_sector.items()
        }
        self._compile_routes()

    def set_q_shell(self, shell: int) -> "PhysicalModel":
        labels = _hexagonal_labels(int(shell))
        self._labels_by_sector = {
            str(sector["id"]): tuple(labels) for sector in self._sectors
        }
        self._compile_routes()
        return self

    def _compile_routes(self) -> None:
        label_index = {
            sector_id: {label: index for index, label in enumerate(labels)}
            for sector_id, labels in self._labels_by_sector.items()
        }
        block_offsets: dict[str, int] = {}
        offset = 0
        for sector in self._sectors:
            sector_id = str(sector["id"])
            block_offsets[sector_id] = offset
            offset += len(self._labels_by_sector[sector_id]) * int(
                sector["number_of_orbitals"]
            )
        self._dimension = int(offset)

        route_rows: list[int] = []
        route_columns: list[int] = []
        route_q: list[np.ndarray] = []
        route_mz: list[int] = []
        route_mz_star: list[int] = []
        route_beta: list[complex] = []
        for term in self.terms:
            row_sector = str(term["sector_row"])
            column_sector = str(term["sector_col"])
            if row_sector not in self._sector_by_id or column_sector not in self._sector_by_id:
                raise ValueError("terms.csv uses an unknown sector")
            row_orbital = int(term["orbital_row"]) - 1
            column_orbital = int(term["orbital_col"]) - 1
            row_labels = self._labels_by_sector[row_sector]
            column_lookup = label_index[column_sector]
            dg = (int(term["dg1"]), int(term["dg2"]))
            beta = complex(float(term["beta_real"]), float(term["beta_imag"]))
            mz = int(term["Mz"])
            mz_star = int(term["Mz_star"])
            row_offset = np.asarray(
                self._sector_by_id[row_sector]["q_offset"], dtype=float
            )
            column_offset = np.asarray(
                self._sector_by_id[column_sector]["q_offset"], dtype=float
            )
            reconstructed_p = row_offset - column_offset + np.asarray(dg) @ self._basis
            saved_p = np.asarray((float(term["p_x"]), float(term["p_y"])))
            if np.linalg.norm(reconstructed_p - saved_p) > self._lattice_tolerance:
                raise ValueError("terms.csv transfer is inconsistent with model.toml")
            row_count = len(row_labels)
            column_count = len(self._labels_by_sector[column_sector])
            for row_q_index, label in enumerate(row_labels):
                column_label = (label[0] - dg[0], label[1] - dg[1])
                column_q_index = column_lookup.get(column_label)
                if column_q_index is None:
                    continue
                route_rows.append(
                    block_offsets[row_sector] + row_orbital * row_count + row_q_index
                )
                route_columns.append(
                    block_offsets[column_sector]
                    + column_orbital * column_count
                    + column_q_index
                )
                route_q.append(row_offset + np.asarray(label) @ self._basis)
                route_mz.append(mz)
                route_mz_star.append(mz_star)
                route_beta.append(beta)
        self._route_rows = np.asarray(route_rows, dtype=np.int64)
        self._route_columns = np.asarray(route_columns, dtype=np.int64)
        self._route_q = (
            np.asarray(route_q, dtype=float).reshape((-1, 2))
            if route_q
            else np.zeros((0, 2), dtype=float)
        )
        self._route_mz = np.asarray(route_mz, dtype=np.int64)
        self._route_mz_star = np.asarray(route_mz_star, dtype=np.int64)
        self._route_beta = np.asarray(route_beta, dtype=np.complex128)

    def hamiltonian(self, kpoint) -> np.ndarray:
        k = np.asarray(kpoint, dtype=float)
        if k.shape != (2,):
            raise ValueError("kpoint must be a 2-vector")
        z = (k[0] - self._route_q[:, 0]) + 1.0j * (
            k[1] - self._route_q[:, 1]
        )
        values = (
            self._route_beta
            * np.power(z, self._route_mz)
            * np.power(np.conjugate(z), self._route_mz_star)
        )
        matrix = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        np.add.at(matrix, (self._route_rows, self._route_columns), values)
        return matrix

    def hamiltonians(self, kpoints) -> np.ndarray:
        points = np.asarray(kpoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("kpoints must have shape (N,2)")
        return np.asarray([self.hamiltonian(point) for point in points])


def load_model(root: str | Path = ".") -> PhysicalModel:
    return PhysicalModel.load(root)
