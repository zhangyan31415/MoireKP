import numpy as np
from types import SimpleNamespace

import tapw.workflows.symmetry as symmetry_analysis


def test_each_spatial_operation_yields_independent_unitary_and_antiunitary_candidates():
    helper = getattr(symmetry_analysis, "expand_spatial_operation_candidates", None)
    assert helper is not None

    operations = [
        {
            "name": "E",
            "rotation_cart": np.eye(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
        {
            "name": "C2",
            "rotation_cart": np.diag([-1.0, -1.0, 1.0]),
            "translation_cart": np.zeros(3, dtype=float),
        },
    ]

    candidates = helper(operations)
    names = [(candidate["name"], candidate["antiunitary"]) for candidate in candidates]

    assert ("E", False) in names
    assert ("T", True) in names
    assert ("C2", False) in names
    assert ("C2T", True) in names


def test_valley_closure_checks_unitary_and_antiunitary_independently():
    helper = getattr(symmetry_analysis, "check_valley_closure", None)
    assert helper is not None

    k_center = np.array([0.2, 0.1], dtype=float)
    reciprocal_basis = np.eye(2, dtype=float)
    c2 = -np.eye(2, dtype=float)

    unitary = helper(k_center, c2, reciprocal_basis, antiunitary=False)
    antiunitary = helper(k_center, c2, reciprocal_basis, antiunitary=True)

    assert unitary["closed"] is False
    assert unitary["reason"] == "valley_not_closed"
    assert antiunitary["closed"] is True
    assert antiunitary["reason"] == ""


def test_required_not_supported_reason_codes_are_declared():
    reason_codes = getattr(symmetry_analysis, "NOT_SUPPORTED_REASONS", None)
    assert reason_codes is not None

    required = {
        "valley_not_closed",
        "q_mapping_missing",
        "atom_mapping_missing",
        "orbital_rotation_unsupported",
        "spin_lift_unsupported",
        "antiunitary_not_order2",
        "orbit_not_closed",
        "S_not_positive_definite",
        "reference_operation_not_structure_symmetry",
        "spglib_operation_not_valley_closed",
    }

    assert required.issubset(set(reason_codes))


def test_minimal_symmetry_candidates_include_only_gamma_c2y_without_duplicates():
    helper = getattr(symmetry_analysis, "_minimal_symmetry_candidates_for_valley", None)
    assert helper is not None

    valley_ctx = SimpleNamespace(
        valley=5,
        valley_label="Gamma",
    )
    operations = [
        {
            "index": 10,
            "rotation_cart": np.array([[-0.5, 0.8660254, 0.0], [0.8660254, 0.5, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
        {
            "index": 11,
            "rotation_cart": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
        {
            "index": 12,
            "rotation_cart": np.array([[-0.5, -0.8660254, 0.0], [-0.8660254, 0.5, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
        {
            "index": 13,
            "rotation_cart": np.array([[-0.5, 0.8660254, 0.0], [0.8660254, 0.5, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
    ]

    candidates = helper(valley_ctx, "hex", spatial_operations=operations)
    names = [candidate["name"] for candidate in candidates]

    assert "E" in names
    assert "T" in names
    assert "C3z" in names
    assert "C3z^2" in names
    assert "C2y" in names
    assert names.count("C2y") == 1
    assert all(not name.startswith("C2(") for name in names)


def test_minimal_symmetry_candidates_do_not_add_c2_for_non_gamma_valley():
    helper = getattr(symmetry_analysis, "_minimal_symmetry_candidates_for_valley", None)
    assert helper is not None

    valley_ctx = SimpleNamespace(
        valley=1,
        valley_label="K1",
    )
    operations = [
        {
            "index": 10,
            "rotation_cart": np.array([[-0.5, 0.8660254, 0.0], [0.8660254, 0.5, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
    ]

    candidates = helper(valley_ctx, "hex", spatial_operations=operations, structure=object())
    names = [candidate["name"] for candidate in candidates]

    assert "C2y" not in names


def test_minimal_symmetry_candidates_include_generic_layer_exchange_c2_for_gamma(monkeypatch):
    helper = getattr(symmetry_analysis, "_minimal_symmetry_candidates_for_valley", None)
    assert helper is not None

    valley_ctx = SimpleNamespace(
        valley=5,
        valley_label="Gamma",
    )
    skew_c2 = {
        "index": 23,
        "rotation_cart": np.array(
            [
                [0.9918032787, 0.1277742402, 0.0],
                [0.1277742398, -0.9918032787, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        "rotation_frac": np.eye(3, dtype=float),
        "translation_frac": np.zeros(3, dtype=float),
        "translation_cart": np.zeros(3, dtype=float),
    }
    operations = [
        skew_c2,
        {
            "index": 24,
            "rotation_cart": np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=float,
            ),
            "rotation_frac": np.eye(3, dtype=float),
            "translation_frac": np.zeros(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
    ]

    def fake_atom_mapping(structure, operation, tol=5.0e-6):
        if int(operation["index"]) == 23:
            return {"group_target_map": {0: 1, 1: 0}}
        return {"group_target_map": {0: 0, 1: 1}}

    monkeypatch.setattr(symmetry_analysis, "_build_atom_mapping", fake_atom_mapping)

    candidates = helper(valley_ctx, "hex", spatial_operations=operations, structure=object())
    c2_candidates = [candidate for candidate in candidates if candidate["name"] == "C2y"]

    assert len(c2_candidates) == 1
    assert int(c2_candidates[0]["index"]) == 23


def test_minimal_symmetry_candidates_include_named_k_c2yt_for_k_valley():
    helper = getattr(symmetry_analysis, "_minimal_symmetry_candidates_for_valley", None)
    assert helper is not None

    valley_ctx = SimpleNamespace(
        valley=1,
        valley_label="K1",
    )
    candidates = helper(valley_ctx, "hex", spatial_operations=[])
    names = [candidate["name"] for candidate in candidates]

    assert "K_C2yT" in names


def test_minimal_symmetry_candidates_include_c3_for_supported_k_valley():
    helper = getattr(symmetry_analysis, "_minimal_symmetry_candidates_for_valley", None)
    assert helper is not None

    valley_ctx = SimpleNamespace(
        valley=1,
        valley_label="K1",
    )
    candidates = helper(valley_ctx, "hex", spatial_operations=[])
    names = [candidate["name"] for candidate in candidates]

    assert "C3z" in names
    assert "C3z^2" in names


def test_minimal_symmetry_candidates_include_named_m_c2_eta_for_m_valley():
    helper = getattr(symmetry_analysis, "_minimal_symmetry_candidates_for_valley", None)
    assert helper is not None

    valley_ctx = SimpleNamespace(
        valley=31,
        valley_label="M1",
    )
    candidates = helper(valley_ctx, "hex", spatial_operations=[])
    names = [candidate["name"] for candidate in candidates]

    assert "M_C2_eta" in names


def test_minimal_symmetry_candidates_include_time_reversal_for_m_valley():
    helper = getattr(symmetry_analysis, "_minimal_symmetry_candidates_for_valley", None)
    assert helper is not None

    valley_ctx = SimpleNamespace(
        valley=31,
        valley_label="M1",
    )
    candidates = helper(valley_ctx, "hex", spatial_operations=[])
    named = {(candidate["name"], bool(candidate.get("antiunitary", False))) for candidate in candidates}

    assert ("T", True) in named


def test_find_layer_exchange_c2_spatial_operation_prefers_gamma_like_layer_exchange(monkeypatch):
    helper = getattr(symmetry_analysis, "find_layer_exchange_c2_spatial_operation", None)
    assert helper is not None

    operations = [
        {
            "index": 1,
            "rotation_cart": np.diag([-1.0, 1.0, -1.0]),
            "rotation_frac": np.eye(3, dtype=float),
            "translation_frac": np.zeros(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
        {
            "index": 2,
            "rotation_cart": np.diag([1.0, -1.0, -1.0]),
            "rotation_frac": np.eye(3, dtype=float),
            "translation_frac": np.zeros(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
    ]

    def fake_atom_mapping(structure, operation, tol=5.0e-6):
        if int(operation["index"]) == 2:
            return {"group_target_map": {0: 1, 1: 0}}
        return {"group_target_map": {0: 0, 1: 1}}

    monkeypatch.setattr(symmetry_analysis, "_build_atom_mapping", fake_atom_mapping)

    chosen = helper(structure=object(), spatial_operations=operations)

    assert chosen is not None
    assert int(chosen["index"]) == 2


def test_list_layer_exchange_c2_spatial_operations_returns_all_layer_exchange_candidates(monkeypatch):
    helper = getattr(symmetry_analysis, "list_layer_exchange_c2_spatial_operations", None)
    assert helper is not None

    operations = [
        {
            "index": 1,
            "rotation_cart": np.array(
                [[-0.5, np.sqrt(3.0) / 2.0, 0.0], [np.sqrt(3.0) / 2.0, 0.5, 0.0], [0.0, 0.0, -1.0]],
                dtype=float,
            ),
            "rotation_frac": np.eye(3, dtype=float),
            "translation_frac": np.zeros(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
        {
            "index": 2,
            "rotation_cart": np.diag([1.0, -1.0, -1.0]),
            "rotation_frac": np.eye(3, dtype=float),
            "translation_frac": np.zeros(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
        {
            "index": 3,
            "rotation_cart": np.diag([-1.0, 1.0, -1.0]),
            "rotation_frac": np.eye(3, dtype=float),
            "translation_frac": np.zeros(3, dtype=float),
            "translation_cart": np.zeros(3, dtype=float),
        },
    ]

    def fake_atom_mapping(structure, operation, tol=5.0e-6):
        if int(operation["index"]) in {1, 2}:
            return {"group_target_map": {0: 1, 1: 0}}
        return {"group_target_map": {0: 0, 1: 1}}

    monkeypatch.setattr(symmetry_analysis, "_build_atom_mapping", fake_atom_mapping)

    found = helper(structure=object(), spatial_operations=operations)

    assert [int(operation["index"]) for operation in found] == [1, 2]
