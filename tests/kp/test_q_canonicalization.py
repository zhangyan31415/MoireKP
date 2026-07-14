from __future__ import annotations

import numpy as np
import pytest

from kp.symmetry.q_canonicalization import (
    QCanonicalizationError,
    canonicalize_q_geometry,
)


def _items(source_sector: str, target_sector: str, permutation: list[int]) -> list[dict[str, object]]:
    return [
        {
            "source_sector": source_sector,
            "source_q_index": source,
            "target_sector": target_sector,
            "target_q_index": target,
        }
        for source, target in enumerate(permutation)
    ]


def _operation(
    name: str,
    q_map: dict[str, object],
    items: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "name": name,
        "internal_resolved_action": {"q_map": q_map},
        "model_basis_action": {"complete": True, "items": items},
    }


def test_canonicalizes_rounded_c3_orbit_without_reordering() -> None:
    theta = 2.0 * np.pi / 3.0
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    q0 = np.array([0.1, -0.02], dtype=float)
    raw = np.stack([q0, rotation @ q0, rotation @ rotation @ q0])
    raw[1, 1] += 6.7e-12
    operation = _operation(
        "threefold",
        {"type": "rotation", "angle_deg": 120.0},
        _items("L1", "L1", [1, 2, 0]),
    )

    result = canonicalize_q_geometry({"L1": raw}, [operation])

    canonical = result.canonical_q["L1"]
    assert canonical.shape == raw.shape
    assert result.raw_q["L1"].tobytes() == raw.tobytes()
    assert result.artifact["raw_closure_max"] > 6.0e-12
    assert result.artifact["canonical_closure_max"] < 1.0e-14
    assert result.artifact["max_correction"] < 1.0e-11
    assert result.artifact["ordering_changed"] is False
    np.testing.assert_allclose(canonical[1], rotation @ canonical[0], atol=1.0e-14, rtol=0.0)


def test_handles_reflection_negation_and_sector_exchange_from_explicit_maps() -> None:
    q = np.array([[0.12, 0.0], [-0.12, 0.0]], dtype=float)
    q2 = q.copy()
    q2[:, 1] += np.array([3.0e-13, -3.0e-13])
    reflection = _operation(
        "unnamed_reflection",
        {"type": "reflection", "axis_deg": 90.0},
        _items("L1", "L1", [1, 0]) + _items("L2", "L2", [1, 0]),
    )
    exchange = _operation(
        "unnamed_exchange",
        {"type": "identity"},
        _items("L1", "L2", [0, 1]) + _items("L2", "L1", [0, 1]),
    )
    negation = _operation(
        "unnamed_negation",
        {"type": "negation"},
        _items("L1", "L1", [1, 0]) + _items("L2", "L2", [1, 0]),
    )

    result = canonicalize_q_geometry(
        {"L1": q, "L2": q2},
        [reflection, exchange, negation],
    )

    np.testing.assert_allclose(result.canonical_q["L1"], result.canonical_q["L2"], atol=1.0e-14)
    assert result.artifact["canonical_closure_max"] < 1.0e-14
    assert {row["name"] for row in result.artifact["operations"]} == {
        "unnamed_reflection",
        "unnamed_exchange",
        "unnamed_negation",
    }


def test_identity_only_geometry_is_preserved_exactly() -> None:
    raw = np.array([[0.0, 0.0], [0.1, -0.2]], dtype=float)

    result = canonicalize_q_geometry({"only": raw}, [])

    assert result.artifact["status"] == "not_needed"
    assert result.canonical_q["only"].tobytes() == raw.tobytes()
    assert result.artifact["max_correction"] == 0.0


def test_rejects_conflicting_discrete_permutation() -> None:
    raw = np.array([[0.1, 0.0], [-0.1, 0.0]], dtype=float)
    items = _items("L1", "L1", [1, 0])
    items.append(
        {
            "source_sector": "L1",
            "source_q_index": 0,
            "target_sector": "L1",
            "target_q_index": 0,
        }
    )

    with pytest.raises(QCanonicalizationError, match="conflicting target"):
        canonicalize_q_geometry(
            {"L1": raw},
            [_operation("bad", {"type": "negation"}, items)],
        )


def test_rejects_correction_larger_than_cleanup_policy() -> None:
    raw = np.array([[0.1, 0.0], [-0.08, 0.0]], dtype=float)
    negation = _operation(
        "negation",
        {"type": "negation"},
        _items("L1", "L1", [1, 0]),
    )

    with pytest.raises(QCanonicalizationError, match="exceeds cleanup tolerance"):
        canonicalize_q_geometry(
            {"L1": raw},
            [negation],
            relative_cleanup_tolerance=1.0e-8,
        )


def test_redundant_constraints_do_not_inflate_the_solver_error_floor() -> None:
    raw = np.array([[0.1, 0.0]], dtype=float)
    operations = [
        _operation(
            f"identity_{index}",
            {
                "type": "identity",
                "translation": [1.0e-12 if index == 0 else 0.0, 0.0],
            },
            _items("L1", "L1", [0]),
        )
        for index in range(256)
    ]

    with pytest.raises(QCanonicalizationError, match="numerically inconsistent"):
        canonicalize_q_geometry({"L1": raw}, operations)
