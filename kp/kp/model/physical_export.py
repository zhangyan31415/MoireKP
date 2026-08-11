"""Lossless helpers for the human-readable physical model export."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class PhysicalExportError(ValueError):
    """Raised when compiled data cannot define an expandable physical model."""


@dataclass(frozen=True)
class AffineIntegerAction:
    """Integer action ``n' = matrix @ n + shift`` for one sector route."""

    matrix: np.ndarray
    shift: np.ndarray


_TERM_KEY_FIELDS = (
    "Mz",
    "Mz_star",
    "layer_from",
    "layer_to",
    "orbital_from",
    "orbital_to",
    "p",
)


def _normalized_term_key(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalExportError("raw owner coordinate has no term_key descriptor")
    missing = [field for field in _TERM_KEY_FIELDS if field not in value]
    if missing:
        raise PhysicalExportError(
            "raw owner term_key is missing fields: " + ", ".join(missing)
        )
    try:
        p = tuple(float(component) for component in value["p"])
        integers = {
            field: int(value[field]) for field in _TERM_KEY_FIELDS if field != "p"
        }
    except (TypeError, ValueError) as exc:
        raise PhysicalExportError("raw owner term_key contains invalid values") from exc
    if len(p) != 2 or not np.all(np.isfinite(p)):
        raise PhysicalExportError("raw owner term_key p must be a finite 2-vector")
    if integers["Mz"] < 0 or integers["Mz_star"] < 0:
        raise PhysicalExportError("raw owner monomial powers must be nonnegative")
    return {**integers, "p": [p[0], p[1]]}


def _term_identity(term_key: Mapping[str, Any]) -> tuple[Any, ...]:
    p = [
        0.0 if float(value) == 0.0 else float(value) for value in term_key["p"]
    ]
    return (
        int(term_key["layer_from"]),
        int(term_key["layer_to"]),
        int(term_key["orbital_from"]),
        int(term_key["orbital_to"]),
        int(term_key["Mz"]),
        int(term_key["Mz_star"]),
        float(p[0]).hex(),
        float(p[1]).hex(),
    )


def contract_fitted_channels(
    basis_metadata: Mapping[str, Any],
    fitted_coefficients: Sequence[float] | np.ndarray,
    *,
    zero_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Contract fitted invariant channels into complex raw-term coefficients.

    The coordinate map is emitted by the response compiler.  This routine does
    not inspect or invert the frozen sparse Hamiltonian basis.
    """

    channels = basis_metadata.get("channels")
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
        raise PhysicalExportError("basis metadata has no channel sequence")
    coefficients = np.asarray(fitted_coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.shape[0] != len(channels):
        raise PhysicalExportError(
            "fitted coefficient count does not match response channel count"
        )
    if not np.all(np.isfinite(coefficients)):
        raise PhysicalExportError("fitted coefficients must be finite")

    accumulated: dict[tuple[Any, ...], complex] = {}
    descriptors: dict[tuple[Any, ...], dict[str, Any]] = {}
    families: dict[tuple[Any, ...], set[str]] = {}
    for channel_index, (channel, theta) in enumerate(zip(channels, coefficients)):
        if not isinstance(channel, Mapping):
            raise PhysicalExportError(f"channel {channel_index} metadata is invalid")
        metadata = channel.get("metadata")
        if not isinstance(metadata, Mapping):
            raise PhysicalExportError(f"channel {channel_index} has no metadata")
        coordinates = metadata.get(
            "raw_owner_coordinates",
            metadata.get("local_fixed_owner_coordinates"),
        )
        if not isinstance(coordinates, Sequence) or isinstance(
            coordinates, (str, bytes)
        ):
            raise PhysicalExportError(
                f"channel {channel_index} has no raw-owner coordinate map"
            )
        for source in coordinates:
            if not isinstance(source, Mapping):
                raise PhysicalExportError(
                    f"channel {channel_index} contains an invalid raw-owner coordinate"
                )
            term_key = _normalized_term_key(source.get("term_key"))
            identity = _term_identity(term_key)
            try:
                weight = float(source["coefficient"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PhysicalExportError(
                    f"channel {channel_index} has an invalid raw-owner coefficient"
                ) from exc
            source_channel_id = str(source.get("channel_id", ""))
            if source_channel_id.endswith(":real"):
                complex_weight = weight
            elif source_channel_id.endswith(":imag"):
                complex_weight = 1.0j * weight
            else:
                raise PhysicalExportError(
                    f"channel {channel_index} raw owner lacks real/imag identity"
                )
            accumulated[identity] = accumulated.get(identity, 0.0j) + (
                float(theta) * complex_weight
            )
            descriptors.setdefault(identity, term_key)
            families.setdefault(identity, set()).add(str(source.get("tag", "")))

    terms: list[dict[str, Any]] = []
    for identity in sorted(accumulated):
        beta = complex(accumulated[identity])
        if abs(beta) <= float(zero_tolerance):
            continue
        family_values = sorted(value for value in families[identity] if value)
        terms.append(
            {
                "term_key": descriptors[identity],
                "family": "+".join(family_values),
                "beta": beta,
            }
        )
    return terms


def physical_terms_from_polynomial_coefficients(
    *,
    polynomial_coefficients: Mapping[tuple[int, int], Any],
    coordinate_origin: Sequence[float],
    coordinate_scale: float,
    qsets: Mapping[str, Any],
    n_orb_by_qset: Mapping[str, int],
    reciprocal_basis: Any,
    sectors: Sequence[Mapping[str, Any]],
    tolerance: float = 1.0e-9,
) -> list[dict[str, Any]]:
    """Collect a fitted polynomial Hamiltonian into row-centred route terms.

    This is a triangular change of polynomial origin, not a fit or matrix
    inverse.  Repeated finite-Q routes certify that one coefficient defines an
    expandable integer-routed term.
    """

    origin = np.asarray(coordinate_origin, dtype=np.float64)
    scale = float(coordinate_scale)
    basis = np.asarray(reciprocal_basis, dtype=np.float64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise PhysicalExportError("polynomial coordinate origin must be a finite 2-vector")
    if not np.isfinite(scale) or scale <= 0.0:
        raise PhysicalExportError("polynomial coordinate scale must be positive and finite")
    _reciprocal_matrix(basis)

    sector_by_qset: dict[str, tuple[int, Mapping[str, Any], np.ndarray]] = {}
    for sector in sectors:
        qset_name = str(sector.get("qset", ""))
        if qset_name not in qsets:
            raise PhysicalExportError(f"sector uses unknown qset {qset_name!r}")
        if qset_name in sector_by_qset:
            raise PhysicalExportError("physical sectors must map one-to-one onto qsets")
        suffix = qset_name.removeprefix("qset")
        if not suffix.isdigit():
            raise PhysicalExportError(f"physical qset name is not indexed: {qset_name!r}")
        offset = np.asarray(sector.get("q_offset"), dtype=np.float64)
        q_values = np.asarray(qsets[qset_name], dtype=np.float64)
        if bool(sector.get("_q_offset_inferred", False)):
            if q_values.ndim != 2 or q_values.shape[1:] != (2,) or not len(q_values):
                raise PhysicalExportError(
                    f"sector {qset_name!r} cannot infer an offset from an empty qset"
                )
            offset = np.asarray(q_values[0], dtype=np.float64)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise PhysicalExportError(f"sector {qset_name!r} lacks a finite q_offset")
        sector_by_qset[qset_name] = (int(suffix), sector, offset)

    row_layer: list[int] = []
    row_orbital: list[int] = []
    row_q_index: list[int] = []
    row_q: list[np.ndarray] = []
    labels_by_layer: dict[int, np.ndarray] = {}
    offsets_by_layer: dict[int, np.ndarray] = {}
    q_count_by_layer: dict[int, int] = {}
    block_offset_by_layer: dict[int, int] = {}
    running_offset = 0
    for qset_name in ("qset1", "qset2"):
        q_values = np.asarray(qsets.get(qset_name, np.empty((0, 2))), dtype=np.float64)
        n_orb = int(n_orb_by_qset.get(qset_name, 0))
        if not len(q_values) or n_orb <= 0:
            continue
        if qset_name not in sector_by_qset:
            raise PhysicalExportError(f"active qset {qset_name!r} has no physical sector")
        layer, _sector, offset = sector_by_qset[qset_name]
        labels = integerize_q_points(q_values, basis, offset, tolerance=tolerance)
        labels_by_layer[layer] = labels
        offsets_by_layer[layer] = offset
        q_count_by_layer[layer] = int(len(q_values))
        block_offset_by_layer[layer] = int(running_offset)
        for orbital in range(n_orb):
            for q_index, q_value in enumerate(q_values):
                row_layer.append(layer)
                row_orbital.append(orbital + 1)
                row_q_index.append(int(q_index))
                row_q.append(np.asarray(q_value, dtype=np.float64))
        running_offset += int(n_orb * len(q_values))

    q_array = np.asarray(row_q, dtype=np.float64).reshape((-1, 2))
    if not len(q_array):
        return []
    route_coefficients: dict[tuple[int, ...], dict[tuple[int, int], complex]] = {}
    maximum_input = 0.0
    for monomial, raw_matrix in polynomial_coefficients.items():
        r, s = (int(monomial[0]), int(monomial[1]))
        if r < 0 or s < 0:
            raise PhysicalExportError("compiled polynomial powers must be nonnegative")
        if hasattr(raw_matrix, "tocoo"):
            matrix = raw_matrix.tocoo()
            rows = np.asarray(matrix.row, dtype=np.int64)
            columns = np.asarray(matrix.col, dtype=np.int64)
            values = np.asarray(matrix.data, dtype=np.complex128)
        else:
            dense = np.asarray(raw_matrix, dtype=np.complex128)
            if dense.shape != (len(q_array), len(q_array)):
                raise PhysicalExportError("compiled polynomial matrix has the wrong dimension")
            rows, columns = np.nonzero(dense)
            values = dense[rows, columns]
        if getattr(raw_matrix, "shape", None) != (len(q_array), len(q_array)):
            raise PhysicalExportError("compiled polynomial matrix has the wrong dimension")
        maximum_input = max(
            maximum_input,
            float(np.max(np.abs(values), initial=0.0)),
        )
        denominator = scale ** (r + s)
        for row, column, value in zip(rows, columns, values):
            row_index = int(row)
            column_index = int(column)
            layer_from = row_layer[row_index]
            layer_to = row_layer[column_index]
            row_label = labels_by_layer[layer_from][row_q_index[row_index]]
            column_label = labels_by_layer[layer_to][row_q_index[column_index]]
            dg = np.asarray(row_label - column_label, dtype=np.int64)
            shift = complex(*(q_array[row_index] - origin))
            for mz in range(r + 1):
                left = comb(r, mz) * shift ** (r - mz)
                for mz_star in range(s + 1):
                    translated = (
                        complex(value)
                        * left
                        * comb(s, mz_star)
                        * np.conjugate(shift) ** (s - mz_star)
                        / denominator
                    )
                    if translated == 0.0j:
                        continue
                    descriptor = (
                        layer_from,
                        layer_to,
                        row_orbital[row_index],
                        row_orbital[column_index],
                        int(mz),
                        int(mz_star),
                        int(dg[0]),
                        int(dg[1]),
                    )
                    route = (row_index, column_index)
                    by_route = route_coefficients.setdefault(descriptor, {})
                    by_route[route] = by_route.get(route, 0.0j) + translated

    cleanup = float(
        4096.0 * np.finfo(np.float64).eps * max(1.0, maximum_input)
    )
    terms: list[dict[str, Any]] = []
    for descriptor in sorted(route_coefficients):
        (
            layer_from,
            layer_to,
            orbital_from,
            orbital_to,
            mz,
            mz_star,
            dg1,
            dg2,
        ) = descriptor
        by_route = route_coefficients[descriptor]
        row_labels = labels_by_layer[layer_from]
        column_lookup = {
            tuple(int(item) for item in label): index
            for index, label in enumerate(labels_by_layer[layer_to])
        }
        row_count = q_count_by_layer[layer_from]
        column_count = q_count_by_layer[layer_to]
        expected: list[complex] = []
        for row_q_position, row_label in enumerate(row_labels):
            column_label = (
                int(row_label[0]) - dg1,
                int(row_label[1]) - dg2,
            )
            column_q_position = column_lookup.get(column_label)
            if column_q_position is None:
                continue
            row_index = (
                block_offset_by_layer[layer_from]
                + (orbital_from - 1) * row_count
                + row_q_position
            )
            column_index = (
                block_offset_by_layer[layer_to]
                + (orbital_to - 1) * column_count
                + column_q_position
            )
            expected.append(by_route.get((row_index, column_index), 0.0j))
        if not expected:
            raise PhysicalExportError("compiled physical descriptor has no finite-Q route")
        beta = complex(sum(expected) / len(expected))
        spread = max(abs(value - beta) for value in expected)
        consistency_bound = max(
            cleanup,
            float(tolerance) * max(1.0, abs(beta)),
        )
        if spread > consistency_bound:
            raise PhysicalExportError(
                "row-centred route coefficients disagree across the finite Q basis: "
                f"spread={spread:.6e}, bound={consistency_bound:.6e}"
            )
        if abs(beta) <= cleanup:
            continue
        p = (
            offsets_by_layer[layer_from]
            - offsets_by_layer[layer_to]
            + np.asarray((dg1, dg2), dtype=np.float64) @ basis
        )
        terms.append(
            {
                "term_key": {
                    "Mz": mz,
                    "Mz_star": mz_star,
                    "layer_from": layer_from,
                    "layer_to": layer_to,
                    "orbital_from": orbital_from,
                    "orbital_to": orbital_to,
                    "p": [float(p[0]), float(p[1])],
                },
                "family": "compiled_response",
                "beta": beta,
            }
        )
    return terms


def _reciprocal_matrix(reciprocal_basis: Any) -> np.ndarray:
    basis = np.asarray(reciprocal_basis, dtype=np.float64)
    if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
        raise PhysicalExportError("reciprocal basis must contain two finite 2-vectors")
    matrix = basis.T
    if abs(float(np.linalg.det(matrix))) <= np.finfo(float).eps:
        raise PhysicalExportError("reciprocal basis is singular")
    return matrix


def _validated_integer_coordinates(
    cartesian: np.ndarray,
    matrix: np.ndarray,
    *,
    tolerance: float,
    label: str,
) -> np.ndarray:
    fractional = np.linalg.solve(matrix, np.asarray(cartesian, dtype=np.float64).T).T
    integers = np.rint(fractional).astype(np.int64)
    reconstructed = integers @ matrix.T
    residuals = np.linalg.norm(
        np.asarray(cartesian, dtype=np.float64) - reconstructed, axis=-1
    )
    maximum = float(np.max(residuals, initial=0.0))
    if maximum > float(tolerance):
        raise PhysicalExportError(
            f"{label} does not lie on the declared affine reciprocal lattice; "
            f"maximum residual={maximum:.6e}, tolerance={float(tolerance):.6e}"
        )
    return integers


def integerize_q_points(
    q_points: Any,
    reciprocal_basis: Any,
    offset: Any,
    *,
    tolerance: float,
) -> np.ndarray:
    """Return integer labels for ``Q = offset + n1*b1 + n2*b2``."""

    values = np.asarray(q_points, dtype=np.float64)
    origin = np.asarray(offset, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or origin.shape != (2,):
        raise PhysicalExportError("Q points must have shape (N,2) and offset shape (2,)")
    return _validated_integer_coordinates(
        values - origin,
        _reciprocal_matrix(reciprocal_basis),
        tolerance=tolerance,
        label="Q point",
    )


def integerize_transfer(
    transfer: Any,
    reciprocal_basis: Any,
    row_offset: Any,
    column_offset: Any,
    *,
    tolerance: float,
) -> np.ndarray:
    """Return ``g`` from ``p = delta_row - delta_col + B @ g``."""

    p = np.asarray(transfer, dtype=np.float64)
    delta_row = np.asarray(row_offset, dtype=np.float64)
    delta_column = np.asarray(column_offset, dtype=np.float64)
    if p.shape != (2,) or delta_row.shape != (2,) or delta_column.shape != (2,):
        raise PhysicalExportError("transfer and sector offsets must be 2-vectors")
    return _validated_integer_coordinates(
        p - (delta_row - delta_column),
        _reciprocal_matrix(reciprocal_basis),
        tolerance=tolerance,
        label="term transfer",
    )


def affine_integer_action(
    cartesian_action: Any,
    reciprocal_basis: Any,
    source_offset: Any,
    target_offset: Any,
    *,
    tolerance: float = 1.0e-9,
) -> AffineIntegerAction:
    """Convert a Cartesian point action into its sector-aware affine form."""

    action = np.asarray(cartesian_action, dtype=np.float64)
    source = np.asarray(source_offset, dtype=np.float64)
    target = np.asarray(target_offset, dtype=np.float64)
    if action.shape != (2, 2) or source.shape != (2,) or target.shape != (2,):
        raise PhysicalExportError("Cartesian action must be 2x2 and offsets 2-vectors")
    matrix = _reciprocal_matrix(reciprocal_basis)
    linear_fractional = np.linalg.solve(matrix, action @ matrix)
    integer_matrix = np.rint(linear_fractional).astype(np.int64)
    linear_residual = float(np.linalg.norm(action @ matrix - matrix @ integer_matrix))
    if linear_residual > float(tolerance):
        raise PhysicalExportError(
            "Cartesian symmetry does not preserve the reciprocal lattice; "
            f"residual={linear_residual:.6e}"
        )
    shift = _validated_integer_coordinates(
        action @ source - target,
        matrix,
        tolerance=tolerance,
        label="sector symmetry offset",
    )
    return AffineIntegerAction(matrix=integer_matrix, shift=shift)


def _toml_string(value: str) -> str:
    import json

    return json.dumps(str(value), ensure_ascii=False)


def _toml_vector(value: Any) -> str:
    return "[" + ", ".join(repr(float(item)) for item in value) + "]"


def build_physical_export_files(
    *,
    model_name: str,
    basis_metadata: Mapping[str, Any],
    fitted_coefficients: Sequence[float] | np.ndarray,
    qsets: Mapping[str, Any],
    reciprocal_basis: Any,
    sectors: Sequence[Mapping[str, Any]],
    energy_unit: str,
    momentum_unit: str,
    symmetry_operations: Sequence[Mapping[str, Any]] = (),
    tolerance: float = 1.0e-9,
    physical_terms: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Build the readable authority files for an expandable physical model."""

    basis = np.asarray(reciprocal_basis, dtype=float)
    _reciprocal_matrix(basis)
    if not sectors:
        raise PhysicalExportError("physical export requires at least one sector")
    normalized_sectors: list[dict[str, Any]] = []
    sector_by_layer_slot: dict[int, dict[str, Any]] = {}
    seen_qsets: set[str] = set()
    for index, sector in enumerate(sectors, start=1):
        sector_id = str(sector.get("name") or f"sector{index}")
        qset_name = str(sector.get("qset", f"qset{index}"))
        if qset_name not in qsets or qset_name in seen_qsets:
            raise PhysicalExportError("sectors must map one-to-one onto exported qsets")
        seen_qsets.add(qset_name)
        offset = np.asarray(sector.get("q_offset"), dtype=float)
        if bool(sector.get("_q_offset_inferred", False)):
            q_values = np.asarray(qsets[qset_name], dtype=float)
            if q_values.ndim != 2 or q_values.shape[1] != 2 or not len(q_values):
                raise PhysicalExportError(
                    f"sector {sector_id!r} cannot fix its inferred offset gauge "
                    "from an empty or invalid qset"
                )
            offset = np.asarray(q_values[0], dtype=float)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise PhysicalExportError(f"sector {sector_id!r} lacks an explicit q_offset")
        n_orb = int(sector.get("n_orb", 0))
        if n_orb <= 0:
            raise PhysicalExportError(f"sector {sector_id!r} must contain orbitals")
        normalized = {
            "id": sector_id,
            "qset": qset_name,
            "q_offset": offset,
            "number_of_orbitals": n_orb,
        }
        normalized_sectors.append(normalized)
        if qset_name.startswith("qset") and qset_name[4:].isdigit():
            sector_by_layer_slot[int(qset_name[4:])] = normalized

    contracted = (
        [dict(term) for term in physical_terms]
        if physical_terms is not None
        else contract_fitted_channels(basis_metadata, fitted_coefficients)
    )
    csv_terms: list[dict[str, Any]] = []
    for term_index, term in enumerate(contracted, start=1):
        key = term["term_key"]
        layer_from = int(key["layer_from"])
        layer_to = int(key["layer_to"])
        if layer_from not in sector_by_layer_slot or layer_to not in sector_by_layer_slot:
            raise PhysicalExportError(
                "term layer index has no active exported sector: "
                f"row={layer_from}, column={layer_to}"
            )
        row_sector = sector_by_layer_slot[layer_from]
        column_sector = sector_by_layer_slot[layer_to]
        g = integerize_transfer(
            key["p"],
            basis,
            row_sector["q_offset"],
            column_sector["q_offset"],
            tolerance=tolerance,
        )
        beta = complex(term["beta"])
        csv_terms.append(
            {
                "term_id": f"T{term_index:06d}",
                "family": term["family"],
                "sector_row": row_sector["id"],
                "orbital_row": int(key["orbital_from"]),
                "sector_col": column_sector["id"],
                "orbital_col": int(key["orbital_to"]),
                "dg1": int(g[0]),
                "dg2": int(g[1]),
                "p_x": float(key["p"][0]),
                "p_y": float(key["p"][1]),
                "Mz": int(key["Mz"]),
                "Mz_star": int(key["Mz_star"]),
                "beta_real": float(beta.real),
                "beta_imag": float(beta.imag),
            }
        )

    q_rows: list[dict[str, Any]] = []
    for sector in normalized_sectors:
        q_values = np.asarray(qsets[sector["qset"]], dtype=float)
        labels = integerize_q_points(
            q_values,
            basis,
            sector["q_offset"],
            tolerance=tolerance,
        )
        for label, q_value in zip(labels, q_values):
            shell = max(
                abs(int(label[0])),
                abs(int(label[1])),
                abs(int(label[0] + label[1])),
            )
            q_rows.append(
                {
                    "sector": sector["id"],
                    "n1": int(label[0]),
                    "n2": int(label[1]),
                    "q_x": float(q_value[0]),
                    "q_y": float(q_value[1]),
                    "shell": shell,
                }
            )

    maximum_order = max(
        (int(row["Mz"]) + int(row["Mz_star"]) for row in csv_terms),
        default=0,
    )
    lines = [
        'schema_version = "moirekp-physical-v1"',
        f"model_name = {_toml_string(model_name)}",
        f"energy_unit = {_toml_string(energy_unit)}",
        f"momentum_unit = {_toml_string(momentum_unit)}",
        "",
        "[polynomial]",
        'complex_coordinate = "z_Q=(kx-Qx)+i*(ky-Qy)"',
        f"maximum_order = {maximum_order}",
        "",
        "[q_lattice]",
        f"b1 = {_toml_vector(basis[0])}",
        f"b2 = {_toml_vector(basis[1])}",
        'offset_gauge = "explicit_sector_offsets_v1"',
    ]
    for sector in normalized_sectors:
        lines.extend(
            (
                "",
                "[[sectors]]",
                f"id = {_toml_string(sector['id'])}",
                f"qset = {_toml_string(sector['qset'])}",
                f"q_offset = {_toml_vector(sector['q_offset'])}",
                f"number_of_orbitals = {sector['number_of_orbitals']}",
            )
        )
    lines.extend(
        (
            "",
            "[validation]",
            f"lattice_tolerance = {repr(float(tolerance))}",
            "",
            "[runtime]",
            'canonical_evaluator = "equation_17_integer_routes"',
            'terms_file = "terms.csv"',
            'default_q_points_file = "q_points.csv"',
            "",
        )
    )

    def csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    terms_fields = [
        "term_id",
        "family",
        "sector_row",
        "orbital_row",
        "sector_col",
        "orbital_col",
        "dg1",
        "dg2",
        "p_x",
        "p_y",
        "Mz",
        "Mz_star",
        "beta_real",
        "beta_imag",
    ]
    q_fields = ["sector", "n1", "n2", "q_x", "q_y", "shell"]
    runtime_source = Path(__file__).with_name("physical_runtime.py").read_text(
        encoding="utf-8"
    )
    files = {
        "model.toml": "\n".join(lines),
        "terms.csv": csv_text(csv_terms, terms_fields),
        "q_points.csv": csv_text(q_rows, q_fields),
        "physical_model.py": runtime_source,
    }
    symmetry_text = _render_symmetry_toml(
        symmetry_operations,
        reciprocal_basis=basis,
        sectors=normalized_sectors,
        tolerance=tolerance,
    )
    if symmetry_text:
        files["symmetry.toml"] = symmetry_text
    return files


def _declared_q_map(operation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    direct = operation.get("q_map")
    if isinstance(direct, Mapping):
        return direct
    declared = operation.get("declared_model_action")
    if isinstance(declared, Mapping) and isinstance(declared.get("q_map"), Mapping):
        return declared["q_map"]
    return None


def _declared_sector_map(operation: Mapping[str, Any]) -> Any:
    if operation.get("sector_map") is not None:
        return operation["sector_map"]
    declared = operation.get("declared_model_action")
    if isinstance(declared, Mapping):
        return declared.get("sector_map", "identity")
    return "identity"


def _declared_antiunitary(operation: Mapping[str, Any]) -> bool:
    if "antiunitary" in operation:
        return bool(operation["antiunitary"])
    for key in ("internal_resolved_action", "declared_model_action", "model_action"):
        action = operation.get(key)
        if isinstance(action, Mapping) and "antiunitary" in action:
            return bool(action["antiunitary"])
    return str(
        operation.get(
            "canonical_operation",
            operation.get("name", operation.get("operation", "")),
        )
    ) in {"C2T", "TR", "TR_eff", "C2TR_eff"}


def _inverse_declared_q_action(
    q_map: Mapping[str, Any],
) -> np.ndarray:
    kind = str(q_map.get("type", "")).strip().lower()
    if kind == "rotation":
        angle = -np.deg2rad(float(q_map["angle_deg"]))
        action = np.asarray(
            (
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            )
        )
    elif kind == "reflection":
        axis = np.deg2rad(float(q_map["axis_deg"]))
        cosine = np.cos(2.0 * axis)
        sine = np.sin(2.0 * axis)
        action = np.asarray(((cosine, sine), (sine, -cosine)))
    elif (matrix_value := q_map.get("matrix", q_map.get("linear_matrix"))) is not None:
        matrix = np.asarray(matrix_value, dtype=float)
        if matrix.shape != (2, 2):
            raise PhysicalExportError("symmetry q_map matrix must have shape (2,2)")
        action = np.linalg.inv(matrix)
    else:
        return np.empty((0, 0), dtype=float)
    return action


def _render_symmetry_toml(
    operations: Sequence[Mapping[str, Any]],
    *,
    reciprocal_basis: np.ndarray,
    sectors: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> str:
    records: list[dict[str, Any]] = []
    sector_by_id = {str(sector["id"]): sector for sector in sectors}
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        q_map = _declared_q_map(operation)
        if q_map is None:
            continue
        antiunitary = _declared_antiunitary(operation)
        action = _inverse_declared_q_action(q_map)
        if action.shape != (2, 2):
            continue
        raw_sector_map = _declared_sector_map(operation)
        if isinstance(raw_sector_map, Mapping):
            sector_map = {
                source: str(raw_sector_map.get(source, source))
                for source in sector_by_id
            }
        elif str(raw_sector_map).strip().lower() == "identity":
            sector_map = {source: source for source in sector_by_id}
        elif str(raw_sector_map).strip().lower() in {
            "swap",
            "exchange",
            "layer_exchange",
        } and len(sectors) == 2:
            sector_map = {
                str(sectors[0]["id"]): str(sectors[1]["id"]),
                str(sectors[1]["id"]): str(sectors[0]["id"]),
            }
        else:
            continue
        affine_by_source = {
            source: affine_integer_action(
                action,
                reciprocal_basis,
                sector_by_id[source]["q_offset"],
                sector_by_id[target]["q_offset"],
                tolerance=tolerance,
            )
            for source, target in sector_map.items()
        }
        matrices = {tuple(result.matrix.ravel()) for result in affine_by_source.values()}
        if len(matrices) != 1:
            raise PhysicalExportError("symmetry integer matrix depends on source sector")
        records.append(
            {
                "name": str(operation.get("name", operation.get("operation", ""))),
                "antiunitary": antiunitary,
                "action": action,
                "matrix": next(iter(affine_by_source.values())).matrix,
                "sector_map": sector_map,
                "shifts": {
                    source: result.shift for source, result in affine_by_source.items()
                },
            }
        )
    if not records:
        return ""

    def integer_matrix(value: np.ndarray) -> str:
        return "[" + ", ".join(
            "[" + ", ".join(str(int(item)) for item in row) + "]"
            for row in np.asarray(value)
        ) + "]"

    def float_matrix(value: np.ndarray) -> str:
        return "[" + ", ".join(_toml_vector(row) for row in np.asarray(value)) + "]"

    lines = [
        'schema_version = "moirekp-sector-symmetry-v1"',
        'q_action_convention = "inverse_declared_q_map_on_Q_labels; antiunitarity_is_coefficient_conjugation"',
    ]
    for record in records:
        lines.extend(
            (
                "",
                "[[operations]]",
                f"name = {_toml_string(record['name'])}",
                f"antiunitary = {str(record['antiunitary']).lower()}",
                f"cartesian_q_action = {float_matrix(record['action'])}",
                f"integer_matrix = {integer_matrix(record['matrix'])}",
                "",
                "[operations.sector_map]",
            )
        )
        lines.extend(
            f"{_toml_string(source)} = {_toml_string(target)}"
            for source, target in record["sector_map"].items()
        )
        lines.extend(("", "[operations.integer_shift_by_source_sector]"))
        lines.extend(
            f"{_toml_string(source)} = [{int(shift[0])}, {int(shift[1])}]"
            for source, shift in record["shifts"].items()
        )
    return "\n".join(lines) + "\n"
