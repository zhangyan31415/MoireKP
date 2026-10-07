from __future__ import annotations

from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from kp.model.physical_export import (
    PhysicalExportError,
    affine_integer_action,
    build_physical_export_files,
    contract_fitted_channels,
    integerize_q_points,
    integerize_transfer,
    physical_terms_from_polynomial_coefficients,
)
from kp.model.physical_runtime import PhysicalModel
from kp.model.export import _physical_files_from_complete_runtime


def _term_key(*, p=(0.0, 0.0), orbital_from=1, orbital_to=1):
    return {
        "Mz": 1,
        "Mz_star": 0,
        "layer_from": 1,
        "layer_to": 2,
        "orbital_from": orbital_from,
        "orbital_to": orbital_to,
        "p": list(p),
    }


def test_contract_fitted_channels_recovers_complex_raw_coefficients() -> None:
    first = _term_key()
    second = _term_key(p=(1.0, 0.0), orbital_to=2)
    metadata = {
        "channels": [
            {
                "component": "real",
                "metadata": {
                    "local_fixed_owner_coordinates": [
                        {
                            "channel_id": "term:0:real",
                            "coefficient": 0.5,
                            "tag": "inter",
                            "term_key": first,
                        },
                        {
                            "channel_id": "term:1:imag",
                            "coefficient": -0.25,
                            "tag": "inter",
                            "term_key": second,
                        },
                    ]
                },
            },
            {
                "component": "imag",
                "metadata": {
                    "local_fixed_owner_coordinates": [
                        {
                            "channel_id": "term:0:imag",
                            "coefficient": 2.0,
                            "tag": "inter",
                            "term_key": first,
                        }
                    ]
                },
            },
        ]
    }

    terms = contract_fitted_channels(metadata, np.asarray([4.0, -3.0]))

    assert len(terms) == 2
    assert terms[0]["term_key"] == first
    assert terms[0]["beta"] == pytest.approx(2.0 - 6.0j)
    assert terms[1]["term_key"] == second
    assert terms[1]["beta"] == pytest.approx(-1.0j)


def test_contract_fitted_channels_rejects_missing_raw_descriptor() -> None:
    metadata = {
        "channels": [
            {
                "component": "real",
                "metadata": {
                    "local_fixed_owner_coordinates": [
                        {
                            "channel_id": "term:0:real",
                            "coefficient": 1.0,
                        }
                    ]
                },
            }
        ]
    }
    with pytest.raises(PhysicalExportError, match="term_key"):
        contract_fitted_channels(metadata, np.asarray([1.0]))


def test_real_mote2_q06_integer_labels_and_c3_affine_shifts() -> None:
    basis = np.asarray(
        ((0.1382071217, 0.0), (0.0691035608, 0.1196908784)),
        dtype=float,
    )
    delta1 = np.asarray((0.0691035608, -0.0398969595))
    delta2 = np.asarray((0.0691035608, 0.0398969595))
    qset1 = np.asarray(
        (
            delta1,
            (0.0, 0.0797939189),
            (-0.0691035608, -0.0398969595),
        )
    )
    labels1 = integerize_q_points(qset1, basis, delta1, tolerance=2.0e-10)
    np.testing.assert_array_equal(labels1, ((0, 0), (-1, 1), (-1, 0)))

    c3 = np.asarray(
        (
            (-0.5, np.sqrt(3.0) / 2.0),
            (-np.sqrt(3.0) / 2.0, -0.5),
        )
    )
    integer_matrix = np.asarray(((0, 1), (-1, -1)))
    shift1 = affine_integer_action(c3, basis, delta1, delta1)
    shift2 = affine_integer_action(c3, basis, delta2, delta2)
    np.testing.assert_array_equal(shift1.matrix, integer_matrix)
    np.testing.assert_array_equal(shift1.shift, (-1, 0))
    np.testing.assert_array_equal(shift2.matrix, integer_matrix)
    np.testing.assert_array_equal(shift2.shift, (0, -1))

    files = build_physical_export_files(
        model_name="mote2-k-q06",
        basis_metadata={"channels": []},
        fitted_coefficients=np.asarray([], dtype=float),
        qsets={"qset1": qset1, "qset2": delta2[None, :]},
        reciprocal_basis=basis,
        sectors=(
            {
                "name": "L1",
                "qset": "qset1",
                "q_offset": delta1 - basis[0],
                "_q_offset_inferred": True,
                "n_orb": 1,
            },
            {
                "name": "L2",
                "qset": "qset2",
                "q_offset": delta2,
                "_q_offset_inferred": True,
                "n_orb": 1,
            },
        ),
        energy_unit="eV",
        momentum_unit="1/angstrom",
        symmetry_operations=(
            {
                "name": "C3z",
                "antiunitary": False,
                "q_map": {"type": "rotation", "angle_deg": 120.0},
                "sector_map": "identity",
            },
        ),
        tolerance=2.0e-10,
    )
    symmetry = tomllib.loads(files["symmetry.toml"])["operations"][0]
    assert symmetry["integer_matrix"] == [[0, 1], [-1, -1]]
    assert symmetry["integer_shift_by_source_sector"] == {
        "L1": [-1, 0],
        "L2": [0, -1],
    }


def test_physical_export_keeps_reflection_c2t_with_layer_exchange() -> None:
    basis = np.asarray(
        ((0.1382071217, 0.0), (0.0691035608, 0.1196908784)),
        dtype=float,
    )
    delta1 = np.asarray((0.0691035608, -0.0398969595))
    delta2 = np.asarray((0.0691035608, 0.0398969595))
    files = build_physical_export_files(
        model_name="c2t-layer-exchange",
        basis_metadata={"channels": []},
        fitted_coefficients=np.asarray([], dtype=float),
        qsets={"qset1": delta1[None, :], "qset2": delta2[None, :]},
        reciprocal_basis=basis,
        sectors=(
            {"name": "L1", "qset": "qset1", "q_offset": delta1, "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": delta2, "n_orb": 1},
        ),
        energy_unit="eV",
        momentum_unit="1/angstrom",
        symmetry_operations=(
            {
                "name": "C2T",
                "declared_model_action": {
                    "antiunitary": True,
                    "q_map": {
                        "type": "reflection",
                        "axis_deg": 0.0,
                        "reflection_axis_convention": "mirror_axis_deg",
                    },
                    "sector_map": "layer_exchange",
                },
            },
        ),
        tolerance=2.0e-10,
    )

    operation = tomllib.loads(files["symmetry.toml"])["operations"][0]
    assert operation["name"] == "C2T"
    assert operation["antiunitary"] is True
    assert operation["integer_matrix"] == [[1, 1], [0, -1]]
    assert operation["sector_map"] == {"L1": "L2", "L2": "L1"}
    assert operation["integer_shift_by_source_sector"] == {
        "L1": [0, 0],
        "L2": [0, 0],
    }


def test_transfer_integerization_subtracts_sector_offsets() -> None:
    basis = np.asarray(((1.0, 0.0), (0.0, 1.0)))
    delta_row = np.asarray((0.3, -0.2))
    delta_col = np.asarray((0.1, 0.4))
    transfer = delta_row - delta_col + np.asarray((2.0, -1.0))

    result = integerize_transfer(
        transfer,
        basis,
        delta_row,
        delta_col,
        tolerance=1.0e-12,
    )

    np.testing.assert_array_equal(result, (2, -1))


def test_polynomial_coefficients_export_as_row_centered_physical_terms() -> None:
    delta1 = np.asarray((0.3, -0.2))
    delta2 = np.asarray((0.1, 0.4))
    qset1 = np.asarray((delta1, delta1 + (1.0, 0.0)))
    qset2 = np.asarray((delta2, delta2 + (1.0, 0.0)))
    beta = 2.0 + 3.0j
    constant = np.zeros((4, 4), dtype=np.complex128)
    linear = np.zeros((4, 4), dtype=np.complex128)
    for row, column, q_row in ((0, 2, qset1[0]), (1, 3, qset1[1])):
        constant[row, column] = -beta * complex(q_row[0], q_row[1])
        linear[row, column] = 2.0 * beta

    terms = physical_terms_from_polynomial_coefficients(
        polynomial_coefficients={(0, 0): constant, (1, 0): linear},
        coordinate_origin=np.zeros(2),
        coordinate_scale=2.0,
        qsets={"qset1": qset1, "qset2": qset2},
        n_orb_by_qset={"qset1": 1, "qset2": 1},
        reciprocal_basis=np.eye(2),
        sectors=(
            {"name": "L1", "qset": "qset1", "q_offset": delta1, "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": delta2, "n_orb": 1},
        ),
        tolerance=1.0e-12,
    )

    assert len(terms) == 1
    assert terms[0]["beta"] == pytest.approx(beta)
    assert terms[0]["term_key"] == {
        "Mz": 1,
        "Mz_star": 0,
        "layer_from": 1,
        "layer_to": 2,
        "orbital_from": 1,
        "orbital_to": 1,
        "p": pytest.approx(delta1 - delta2),
    }


def test_physical_files_drive_route_evaluator(tmp_path) -> None:
    delta1 = np.asarray((0.3, -0.2))
    delta2 = np.asarray((0.1, 0.4))
    first = _term_key(p=tuple(delta1 - delta2))
    metadata = {
        "channels": [
            {
                "component": "real",
                "metadata": {
                    "raw_owner_coordinates": [
                        {
                            "channel_id": "term:0:real",
                            "coefficient": 1.0,
                            "tag": "inter",
                            "term_key": first,
                        }
                    ]
                },
            }
        ]
    }
    files = build_physical_export_files(
        model_name="two-sector-test",
        basis_metadata=metadata,
        fitted_coefficients=np.asarray([2.0]),
        qsets={"qset1": delta1[None, :], "qset2": delta2[None, :]},
        reciprocal_basis=np.eye(2),
        sectors=(
            {"name": "L1", "qset": "qset1", "q_offset": delta1, "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": delta2, "n_orb": 2},
        ),
        energy_unit="eV",
        momentum_unit="1/angstrom",
        tolerance=1.0e-12,
    )
    for name, payload in files.items():
        (tmp_path / name).write_text(payload, encoding="utf-8")

    model = PhysicalModel.load(tmp_path)
    kpoint = np.asarray((0.7, 0.1))
    hamiltonian = model.hamiltonian(kpoint)

    assert model.dimension == 3
    assert hamiltonian[0, 1] == pytest.approx(
        2.0 * complex(*(kpoint - delta1))
    )
    assert np.count_nonzero(hamiltonian) == 1


def test_runtime_uses_the_exported_lattice_tolerance(tmp_path) -> None:
    delta1 = np.asarray((0.3, -0.2))
    delta2 = np.asarray((0.1, 0.4))
    accepted_residual = np.asarray((5.0e-9, 0.0))
    files = build_physical_export_files(
        model_name="tolerance-test",
        basis_metadata={"channels": []},
        fitted_coefficients=np.asarray([], dtype=float),
        qsets={
            "qset1": (delta1 + accepted_residual)[None, :],
            "qset2": delta2[None, :],
        },
        reciprocal_basis=np.eye(2),
        sectors=(
            {"name": "L1", "qset": "qset1", "q_offset": delta1, "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": delta2, "n_orb": 1},
        ),
        energy_unit="eV",
        momentum_unit="1/angstrom",
        tolerance=1.0e-8,
    )
    for name, payload in files.items():
        (tmp_path / name).write_text(payload, encoding="utf-8")

    model = PhysicalModel.load(tmp_path)

    assert model.dimension == 2


def test_standalone_export_includes_physical_files_from_compiled_polynomial() -> None:
    delta1 = np.asarray((0.3, -0.2))
    delta2 = np.asarray((0.1, 0.4))
    constant = np.zeros((3, 3), dtype=np.complex128)
    linear = np.zeros((3, 3), dtype=np.complex128)
    constant[0, 1] = -2.0 * complex(delta1[0], delta1[1])
    linear[0, 1] = 2.0
    runtime = SimpleNamespace(
        basis=SimpleNamespace(
            coordinate=SimpleNamespace(origin=(0.0, 0.0), scale=1.0)
        ),
        polynomial_coefficients={(0, 0): constant, (1, 0): linear},
    )
    files = _physical_files_from_complete_runtime(
        model_name="certified-test",
        response_semantics="complete_linear_v2",
        frozen_response_data={},
        qsets={"qset1": delta1[None, :], "qset2": delta2[None, :]},
        reciprocal_basis=np.eye(2),
        sectors=(
            {"name": "L1", "qset": "qset1", "q_offset": delta1, "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": delta2, "n_orb": 2},
        ),
        symmetry_operations=(),
        energy_unit="eV",
        compiled_runtime=runtime,
        n_orb_by_qset={"qset1": 1, "qset2": 2},
    )

    assert {"model.toml", "terms.csv", "q_points.csv", "physical_model.py"} <= set(
        files
    )
