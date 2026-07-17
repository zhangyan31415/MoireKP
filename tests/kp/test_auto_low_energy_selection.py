from __future__ import annotations

import importlib

import numpy as np
import pytest


def _selection_module():
    try:
        return importlib.import_module("kp.low_energy_selection")
    except ModuleNotFoundError:
        pytest.fail("kp.low_energy_selection is not implemented")


def test_reference_k_uses_exact_expansion_origin_not_array_zero() -> None:
    selection = _selection_module()
    kpoints = np.asarray(
        [
            [0.25, 0.0],
            [0.10, -0.20],
            [0.0, 0.0],
            [-0.15, 0.05],
        ],
        dtype=float,
    )
    qsets = [np.asarray([[0.2, 0.0], [0.0, 0.0]], dtype=float)]

    reference = selection.resolve_reference_point(kpoints, qsets)

    assert reference.k_index == 2
    assert reference.k_coordinate == (0.0, 0.0)


def test_reference_k_rejects_path_without_expansion_origin() -> None:
    selection = _selection_module()
    kpoints = np.asarray([[0.1, 0.0], [-0.1, 0.0]], dtype=float)
    qsets = [np.asarray([[0.0, 0.0]], dtype=float)]

    with pytest.raises(ValueError, match="expansion origin"):
        selection.resolve_reference_point(kpoints, qsets, origin_tolerance=1.0e-8)


def test_minimum_q_is_resolved_per_source_group_with_stable_tie_break() -> None:
    selection = _selection_module()
    kpoints = np.asarray([[0.0, 0.0]], dtype=float)
    qsets = [
        np.asarray([[0.3, 0.0], [0.1, 0.0], [-0.1, 0.0]], dtype=float),
        np.asarray([[0.0, 0.4], [0.0, 0.2]], dtype=float),
    ]

    reference = selection.resolve_reference_point(kpoints, qsets)

    assert reference.q_indices == (1, 1)
    assert reference.q_vectors == ((0.1, 0.0), (0.0, 0.2))
