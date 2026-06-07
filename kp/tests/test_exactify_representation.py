from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from kp.model.configured import build_moire_config_from_file
from kp.model.exactify_representation import (
    BasisLabel,
    OperationAction,
    analyze_monomial_support,
    build_label_action,
    compare_operation_convention,
    exactify_1d_monomial_phases,
    exactify_loaded_symmetry_source,
    nearest_root_of_unity,
    root_of_unity,
)


def _source_meta(*, antiunitary: bool = False) -> dict[str, object]:
    return {
        "source_matrix_role": "raw_h_sewing_action",
        "source_gauge": "raw_saved_TAPW",
        "target_role": "continuum_internal_rep",
        "gauge_correction": {"kind": "none"},
        "antiunitary_convention": "U_K" if antiunitary else "none",
        "spin_map": "from_kp_symm_output",
        "valley_map": "identity",
    }


def _hex_bm() -> tuple[np.ndarray, np.ndarray]:
    b1 = np.array([1.0, 0.0], dtype=float)
    b2 = np.array([0.5, np.sqrt(3.0) / 2.0], dtype=float)
    return b1, b2


def _k_basis_labels() -> list[BasisLabel]:
    b1, b2 = _hex_bm()
    qshell = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
    ]
    labels: list[BasisLabel] = []
    idx = 0
    for sector in ("L1", "L2"):
        for n1, n2 in qshell:
            labels.append(
                BasisLabel(
                    index=idx,
                    sector=sector,
                    q_integer=(n1, n2),
                    q_vector=n1 * b1 + n2 * b2,
                    orbital=1,
                    internal_label=None,
                )
            )
            idx += 1
    return labels


def _k_basis_labels_two_orbital() -> list[BasisLabel]:
    b1, b2 = _hex_bm()
    qshell = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1),
    ]
    labels: list[BasisLabel] = []
    idx = 0
    for sector in ("L1", "L2"):
        for orbital in (1, 2):
            for n1, n2 in qshell:
                labels.append(
                    BasisLabel(
                        index=idx,
                        sector=sector,
                        q_integer=(n1, n2),
                        q_vector=n1 * b1 + n2 * b2,
                        orbital=orbital,
                        internal_label=None,
                    )
                )
                idx += 1
    return labels


def _c3_action() -> OperationAction:
    theta = 2.0 * np.pi / 3.0
    return OperationAction(
        name="C3z",
        canonical_name="C3z",
        antiunitary=False,
        k_map={"type": "rotation", "angle_deg": 120},
        R=np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float),
        sector_map={"L1": "L1", "L2": "L2"},
        q_map={"type": "rotation"},
        central_phase=complex(-1.0, 0.0),
        group_relations=[{"type": "power", "power": 3, "phase": -1.0, "name": "C3z^3"}],
        source="test",
    )


def _make_exact_c3_matrix(labels: list[BasisLabel]) -> tuple[np.ndarray, np.ndarray]:
    b1, b2 = _hex_bm()
    action = _c3_action()
    label_action = build_label_action(labels, action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    mat = np.zeros((len(labels), len(labels)), dtype=complex)
    phase = np.exp(1j * np.pi / 3.0)
    for i, j in enumerate(label_action.perm):
        mat[j, i] = phase
    return mat, label_action.perm


def _make_exact_c2t_exchange_matrix(labels: list[BasisLabel]) -> np.ndarray:
    by_key = {(label.sector, label.q_integer, label.orbital): label.index for label in labels}
    mat = np.zeros((len(labels), len(labels)), dtype=complex)
    for label in labels:
        target_sector = "L2" if label.sector == "L1" else "L1"
        target = by_key[(target_sector, label.q_integer, label.orbital)]
        mat[target, label.index] = 1.0 + 0.0j
    return mat


def _make_block_mixed_c3_case() -> tuple[
    list[BasisLabel],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    labels = _k_basis_labels_two_orbital()
    q1 = np.array(
        [label.q_vector for label in labels if label.sector == "L1" and label.orbital == 1],
        dtype=float,
    )
    q2 = np.array(
        [label.q_vector for label in labels if label.sector == "L2" and label.orbital == 1],
        dtype=float,
    )
    b1, b2 = _hex_bm()
    action = _c3_action()
    label_action = build_label_action(labels, action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)

    eps = 3.5e-5
    rot = np.array([[np.cos(eps), -np.sin(eps)], [np.sin(eps), np.cos(eps)]], dtype=complex)
    roots = np.diag([-1.0 + 0.0j, np.exp(-1j * np.pi / 3.0)])
    block = rot @ roots @ rot.conj().T
    cleanup_residual = float(np.linalg.norm(block - np.diag(np.diag(block))) / np.linalg.norm(block))
    raw = np.zeros((len(labels), len(labels)), dtype=complex)
    expected = np.zeros_like(raw)
    groups: dict[tuple[str, tuple[int, int]], list[BasisLabel]] = {}
    for label in labels:
        groups.setdefault((label.sector, label.q_integer), []).append(label)
    for group in groups.values():
        group.sort(key=lambda item: item.orbital)
        cols = np.asarray([label.index for label in group], dtype=int)
        rows = np.asarray([label_action.perm[col] for col in cols], dtype=int)
        raw[np.ix_(rows, cols)] = block
        expected[rows[0], cols[0]] = -1.0 + 0.0j
        expected[rows[1], cols[1]] = np.exp(-1j * np.pi / 3.0)
    return labels, q1, q2, b1, b2, raw, expected, cleanup_residual


def test_nearest_root_of_unity() -> None:
    roots = [root_of_unity(6, p) for p in range(6)]
    root, residual = nearest_root_of_unity(np.exp(1j * (np.pi / 3.0 + 1.0e-3)), roots)
    assert root == pytest.approx(np.exp(1j * np.pi / 3.0))
    assert abs(residual) < 1.0e-2


def test_exactify_c3_noisy_root_phase() -> None:
    rng = np.random.default_rng(0)
    labels = _k_basis_labels()
    exact, perm = _make_exact_c3_matrix(labels)
    noisy = exact.copy()
    support = [(j, i) for i, j in enumerate(perm)]
    for j, i in support:
        noisy[j, i] *= (1.0 + 5.0e-4 * rng.normal()) * np.exp(1j * 1.0e-3 * rng.normal())
    noisy += (1.0e-7 * rng.normal(size=noisy.shape) + 1.0e-7j * rng.normal(size=noisy.shape))

    report = analyze_monomial_support(noisy, perm)
    exactified, exact_report = exactify_1d_monomial_phases(
        noisy,
        perm,
        labels=labels,
        phase_classes="global",
        allowed_roots=[np.exp(1j * (np.pi + 2.0 * np.pi * m) / 3.0) for m in range(3)],
        operation_name="C3z",
        power=3,
        central_phase=-1.0,
    )

    assert report.off_support_rel < 1.0e-5
    np.testing.assert_allclose(exactified, exact, atol=1.0e-12)
    assert exact_report.status == "exactified"
    assert exact_report.group_residuals["power"] < 1.0e-14


def test_exactify_rejects_bad_amplitude() -> None:
    labels = _k_basis_labels()
    exact, perm = _make_exact_c3_matrix(labels)
    bad = exact * 0.63

    with pytest.raises(ValueError, match="amplitude"):
        exactify_1d_monomial_phases(
            bad,
            perm,
            labels=labels,
            phase_classes="global",
            allowed_roots=[np.exp(1j * (np.pi + 2.0 * np.pi * m) / 3.0) for m in range(3)],
            operation_name="C3z",
            power=3,
            central_phase=-1.0,
        )


def test_exactify_rejects_bad_support() -> None:
    labels = _k_basis_labels()
    exact, perm = _make_exact_c3_matrix(labels)
    bad = exact.copy()
    bad[0, 1] = 0.2
    report = analyze_monomial_support(bad, perm)
    assert report.off_support_rel > 1.0e-2


def test_compare_operation_convention_rejects_same_matrix_different_kmap() -> None:
    mat = np.eye(2, dtype=complex)
    op1 = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "identity"},
        R=np.eye(2, dtype=float),
        sector_map={},
        q_map={"type": "identity"},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="a",
    )
    op2 = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 0.0},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map={},
        q_map={"type": "reflection", "axis_deg": 0.0},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="b",
    )
    comparison = compare_operation_convention(op1, mat, op2, mat)
    assert comparison["matrix_close_up_to_global_phase"] is True
    assert comparison["interchangeable_for_term_symmetrization"] is False
    assert comparison["k_map_equal"] is False


def test_exactify_respects_explicit_operation_action_candidates() -> None:
    labels = _k_basis_labels()
    exact, _perm = _make_exact_c3_matrix(labels)
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C3z",
                    "antiunitary": False,
                    "k_map": {"type": "identity"},
                    "sector_map": "identity",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"C3z": exact},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
        ],
        n_orb=(1, 1),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C3z": {
                        "support_mode": "monomial",
                        "power": 3,
                        "central_phase": -1.0,
                        "allowed_roots": [np.exp(1j * (np.pi + 2.0 * np.pi * m) / 3.0) for m in range(3)],
                        "phase_classes": "global",
                        "action_candidates": [
                            {
                                "k_map": {"type": "rotation", "angle_deg": 120.0},
                                "q_map": {"type": "rotation", "angle_deg": 120.0},
                                "sector_map": "identity",
                                "antiunitary": False,
                            }
                        ],
                    }
                }
            }
        },
    )

    np.testing.assert_allclose(exactified["C3z"], exact, atol=1.0e-12)
    assert reports["C3z"]["resolved_action"]["k_map"]["type"] == "rotation"
    assert reports["C3z"]["resolved_action"]["q_map"]["type"] == "rotation"
    assert reports["C3z"]["report"]["status"] == "exactified"


def test_exactify_infers_missing_sector_q_offsets() -> None:
    b1, b2 = _hex_bm()
    q1 = np.array([0.5 * b2], dtype=float)
    q2 = np.array([0.5 * b1 + 0.5 * b2], dtype=float)

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "TR_eff",
                    "antiunitary": True,
                    "k_map": {"type": "identity"},
                    "sector_map": "identity",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"TR_eff": np.eye(2, dtype=complex)},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "bottom", "qset": "qset1", "n_orb": 1},
            {"name": "top", "qset": "qset2", "n_orb": 1},
        ],
        n_orb=(1, 1),
        bM1=b1,
        bM2=b2,
        raw_config={"exactification": {"support_mode": "monomial"}},
    )

    np.testing.assert_allclose(exactified["TR_eff"], np.eye(2, dtype=complex))
    assert reports["TR_eff"]["inferred_q_offsets"] == {"bottom": True, "top": True}
    np.testing.assert_allclose(reports["TR_eff"]["q_offsets"]["bottom"], (-0.5 * b2).tolist())
    np.testing.assert_allclose(reports["TR_eff"]["q_offsets"]["top"], (-0.5 * b1 - 0.5 * b2).tolist())


def test_exactify_preserves_explicit_in_model_frame_reflection_axis() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    exact_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "reflection", "axis_deg": 180.0},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    label_action = build_label_action(labels, exact_action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    exact = np.zeros((len(labels), len(labels)), dtype=complex)
    for i, j in enumerate(label_action.perm):
        exact[j, i] = 1.0 + 0.0j

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "C2T",
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 0.0},
                    "sector_map": "identity",
                    "q_map": {"type": "reflection", "axis_deg": 0.0},
                }
            ]
        },
        matrices={"C2T": exact},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
        ],
        n_orb=(1, 1),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C2T": {
                        "support_mode": "block",
                        "power": 2,
                        "central_phase": 1.0,
                        "action_candidates": [
                            {
                                "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                                "q_map": {"type": "reflection", "axis_deg": 180.0},
                                "sector_map": "layer_exchange",
                                "antiunitary": True,
                            }
                        ],
                    }
                }
            }
        },
        rotation_deg=210.0,
    )

    np.testing.assert_allclose(exactified["C2T"], exact, atol=1.0e-12)
    assert reports["C2T"]["resolved_action"]["k_map"]["axis_deg"] == pytest.approx(180.0)
    assert reports["C2T"]["resolved_action"]["sector_map"] == "layer_exchange"


def test_block_monomial_cleanup_removes_tiny_internal_mixing() -> None:
    labels, q1, q2, b1, b2, raw, expected, expected_cleanup_residual = _make_block_mixed_c3_case()

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C3z",
                    "antiunitary": False,
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                }
            ]
        },
        matrices={"C3z": raw},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 2},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 2},
        ],
        n_orb=(2, 2),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C3z": {
                        "support_mode": "block_monomial",
                        "power": 3,
                        "central_phase": -1.0,
                        "action_candidates": [
                            {
                                "k_map": {"type": "rotation", "angle_deg": 120.0},
                                "q_map": {"type": "rotation", "angle_deg": 120.0},
                                "sector_map": "identity",
                                "antiunitary": False,
                            }
                        ],
                    }
                }
            }
        },
    )

    np.testing.assert_allclose(exactified["C3z"], expected, atol=1.0e-12)
    assert reports["C3z"]["preferred_mode"] == "block_monomial"
    assert reports["C3z"]["report"]["off_support_rel"] == pytest.approx(0.0)
    assert reports["C3z"]["report"]["joint_group_residuals"]["cleanup_input_off_support_rel"] == pytest.approx(
        expected_cleanup_residual
    )
    assert "block_monomial_cleanup" in reports["C3z"]["report"]["notes"]


def test_c3z_block_support_is_not_auto_cleaned_without_explicit_mode() -> None:
    labels, q1, q2, b1, b2, raw, expected, _cleanup_residual = _make_block_mixed_c3_case()

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C3z",
                    "antiunitary": False,
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                }
            ]
        },
        matrices={"C3z": raw},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 2},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 2},
        ],
        n_orb=(2, 2),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C3z": {
                        "power": 3,
                        "central_phase": -1.0,
                        "action_candidates": [
                            {
                                "k_map": {"type": "rotation", "angle_deg": 120.0},
                                "q_map": {"type": "rotation", "angle_deg": 120.0},
                                "sector_map": "identity",
                                "antiunitary": False,
                            }
                        ],
                    }
                }
            }
        },
    )

    assert reports["C3z"]["preferred_mode"] == "block"
    assert "block_monomial_cleanup" not in reports["C3z"]["report"]["notes"]
    assert np.linalg.norm(exactified["C3z"] - expected) / np.linalg.norm(expected) > 1.0e-7


def test_block_monomial_cleanup_allows_internal_orbital_exchange() -> None:
    labels = _k_basis_labels_two_orbital()
    q1 = np.array(
        [label.q_vector for label in labels if label.sector == "L1" and label.orbital == 1],
        dtype=float,
    )
    q2 = np.array(
        [label.q_vector for label in labels if label.sector == "L2" and label.orbital == 1],
        dtype=float,
    )
    b1, b2 = _hex_bm()
    action = OperationAction(
        name="C2",
        canonical_name="C2",
        antiunitary=False,
        k_map={"type": "identity"},
        R=np.eye(2, dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "identity"},
        central_phase=-1.0 + 0.0j,
        group_relations=[{"type": "power", "power": 2, "phase": -1.0, "name": "C2^2"}],
        source="test",
    )
    label_action = build_label_action(labels, action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)

    block = np.array([[0.0, 1.0j], [1.0j, 0.0]], dtype=complex)
    raw = np.zeros((len(labels), len(labels)), dtype=complex)
    expected = np.zeros_like(raw)
    groups: dict[tuple[str, tuple[int, int]], list[BasisLabel]] = {}
    for label in labels:
        groups.setdefault((label.sector, label.q_integer), []).append(label)
    for group in groups.values():
        group.sort(key=lambda item: item.orbital)
        cols = np.asarray([label.index for label in group], dtype=int)
        rows = np.asarray([label_action.perm[col] for col in cols], dtype=int)
        raw[np.ix_(rows, cols)] = block
        expected[rows[0], cols[1]] = 1.0j
        expected[rows[1], cols[0]] = 1.0j

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C2",
                    "antiunitary": False,
                    "k_map": {"type": "identity"},
                    "sector_map": "layer_exchange",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"C2": raw},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 2},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 2},
        ],
        n_orb=(2, 2),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C2": {
                        "support_mode": "block_monomial",
                        "power": 2,
                        "central_phase": -1.0,
                        "allowed_roots": [[0.0, 1.0], [0.0, -1.0]],
                        "phase_classes": "global",
                        "monomial_cleanup_tol": 1.0e-8,
                        "action_candidates": [
                            {
                                "k_map": {"type": "identity"},
                                "q_map": {"type": "identity"},
                                "sector_map": "layer_exchange",
                                "antiunitary": False,
                            }
                        ],
                    }
                }
            }
        },
    )

    np.testing.assert_allclose(exactified["C2"], expected, atol=1.0e-12)
    assert reports["C2"]["preferred_mode"] == "block_monomial"
    assert reports["C2"]["report"]["off_support_rel"] == pytest.approx(0.0)
    assert reports["C2"]["report"]["group_residuals"]["power"] < 1.0e-14


def test_c2_block_monomial_cleanup_infers_exchange_phase_roots() -> None:
    labels = _k_basis_labels_two_orbital()
    q1 = np.array(
        [label.q_vector for label in labels if label.sector == "L1" and label.orbital == 1],
        dtype=float,
    )
    q2 = np.array(
        [label.q_vector for label in labels if label.sector == "L2" and label.orbital == 1],
        dtype=float,
    )
    b1, b2 = _hex_bm()
    action = OperationAction(
        name="C2",
        canonical_name="C2",
        antiunitary=False,
        k_map={"type": "identity"},
        R=np.eye(2, dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "identity"},
        central_phase=1.0 + 0.0j,
        group_relations=[{"type": "power", "power": 2, "phase": 1.0, "name": "C2^2"}],
        source="test",
    )
    label_action = build_label_action(labels, action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)

    block = np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=complex)
    raw = np.zeros((len(labels), len(labels)), dtype=complex)
    groups: dict[tuple[str, tuple[int, int]], list[BasisLabel]] = {}
    for label in labels:
        groups.setdefault((label.sector, label.q_integer), []).append(label)
    for group in groups.values():
        group.sort(key=lambda item: item.orbital)
        cols = np.asarray([label.index for label in group], dtype=int)
        rows = np.asarray([label_action.perm[col] for col in cols], dtype=int)
        raw[np.ix_(rows, cols)] = block

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C2",
                    "antiunitary": False,
                    "k_map": {"type": "identity"},
                    "sector_map": "layer_exchange",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"C2": raw},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 2},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 2},
        ],
        n_orb=(2, 2),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C2": {
                        "power": 2,
                        "central_phase": 1.0,
                        "action_candidates": [
                            {
                                "k_map": {"type": "identity"},
                                "q_map": {"type": "identity"},
                                "sector_map": "layer_exchange",
                                "antiunitary": False,
                            }
                        ],
                    }
                }
            }
        },
    )

    np.testing.assert_allclose(exactified["C2"], raw, atol=1.0e-12)
    assert reports["C2"]["preferred_mode"] == "block_monomial"
    assert reports["C2"]["report"]["group_residuals"]["power"] < 1.0e-14


def test_exactify_layer_exchange_supports_bottom_top_sector_labels() -> None:
    b1, b2 = _hex_bm()
    q_bottom = np.array([b2], dtype=float)
    q_top = np.array([b1 - b2], dtype=float)
    exact = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C2_eff",
                    "antiunitary": False,
                    "k_map": {"type": "identity"},
                    "sector_map": "identity",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"C2_eff": exact},
        Q_set1=q_bottom,
        Q_set2=q_top,
        sectors=[
            {"name": "bottom", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
            {"name": "top", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
        ],
        n_orb=(1, 1),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C2_eff": {
                        "support_mode": "monomial",
                        "power": 2,
                        "central_phase": 1.0,
                        "allowed_roots": [1.0, -1.0],
                        "phase_classes": "global",
                        "action_candidates": [
                            {
                                "k_map": {"type": "reflection", "axis_deg": 0.0},
                                "q_map": {"type": "reflection", "axis_deg": 0.0},
                                "sector_map": "layer_exchange",
                                "antiunitary": False,
                            }
                        ],
                    }
                }
            }
        },
    )

    np.testing.assert_allclose(exactified["C2_eff"], exact, atol=1.0e-12)
    assert reports["C2_eff"]["label_action"]["perm_complete"] is True
    assert reports["C2_eff"]["label_action"]["missing_count"] == 0
    assert reports["C2_eff"]["resolved_action"]["sector_map"] == "layer_exchange"


def test_exactify_strict_forbids_inferred_q_offset() -> None:
    labels = _k_basis_labels()
    exact, _perm = _make_exact_c3_matrix(labels)
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float) + np.array([0.125, 0.0])
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float) + np.array([0.125, 0.0])
    b1, b2 = _hex_bm()

    with pytest.raises(ValueError, match="forbids inferred q_offset"):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "source_matrix_role": "raw_h_sewing_action",
                        "source_gauge": "raw_saved_TAPW",
                        "target_role": "continuum_internal_rep",
                        "antiunitary_convention": "none",
                                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120},
                        "sector_map": "identity",
                        "q_map": {"type": "rotation", "angle_deg": 120},
                    }
                ]
            },
            matrices={"C3z": exact},
            Q_set1=q1,
            Q_set2=q2,
            sectors=[
                {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
                {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
            ],
            n_orb=(1, 1),
            bM1=b1,
            bM2=b2,
            raw_config={"exactification": {"strict": True, "forbid_inferred_q_offset": True}},
        )


def test_exactify_strict_requires_action_candidates() -> None:
    labels = _k_basis_labels()
    exact, _perm = _make_exact_c3_matrix(labels)
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()

    with pytest.raises(ValueError, match="requires explicit action_candidates"):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "source_matrix_role": "raw_h_sewing_action",
                        "source_gauge": "raw_saved_TAPW",
                        "target_role": "continuum_internal_rep",
                        "antiunitary_convention": "none",
                                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120},
                        "sector_map": "identity",
                        "q_map": {"type": "rotation", "angle_deg": 120},
                    }
                ]
            },
            matrices={"C3z": exact},
            Q_set1=q1,
            Q_set2=q2,
            sectors=[
                {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
                {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
            ],
            n_orb=(1, 1),
            bM1=b1,
            bM2=b2,
            raw_config={"exactification": {"strict": True, "require_explicit_action_candidates": True}},
        )


def test_exactify_strict_requires_group_relations() -> None:
    labels = _k_basis_labels()
    exact, _perm = _make_exact_c3_matrix(labels)
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()

    with pytest.raises(ValueError, match="requires explicit group relation"):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "source_matrix_role": "raw_h_sewing_action",
                        "source_gauge": "raw_saved_TAPW",
                        "target_role": "continuum_internal_rep",
                        "antiunitary_convention": "none",
                                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120},
                        "sector_map": "identity",
                        "q_map": {"type": "rotation", "angle_deg": 120},
                    }
                ]
            },
            matrices={"C3z": exact},
            Q_set1=q1,
            Q_set2=q2,
            sectors=[
                {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
                {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
            ],
            n_orb=(1, 1),
            bM1=b1,
            bM2=b2,
            raw_config={
                "exactification": {
                    "strict": True,
                    "require_explicit_action_candidates": True,
                    "operations": {
                        "C3z": {
                            "action_candidates": [
                                {
                                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                                    "sector_map": "identity",
                                    "antiunitary": False,
                                }
                            ]
                        }
                    },
                }
            },
        )


def test_exactify_report_contains_source_semantics_and_inference_flag() -> None:
    labels = _k_basis_labels()
    exact, _perm = _make_exact_c3_matrix(labels)
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()

    _exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    "name": "C3z",
                    "matrix_file": "C3_low_raw.npy",
                    "source_matrix_role": "raw_h_sewing_action",
                    "source_gauge": "raw_saved_TAPW",
                    "target_role": "continuum_internal_rep",
                    "antiunitary_convention": "none",
                                "antiunitary": False,
                    "k_map": {"type": "identity"},
                    "sector_map": "identity",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"C3z": exact},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
        ],
        n_orb=(1, 1),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C3z": {
                        "support_mode": "monomial",
                        "power": 3,
                        "central_phase": -1.0,
                        "allowed_roots": [np.exp(1j * (np.pi + 2.0 * np.pi * m) / 3.0) for m in range(3)],
                        "phase_classes": "global",
                        "action_candidates": [
                            {
                                "k_map": {"type": "rotation", "angle_deg": 120.0},
                                "q_map": {"type": "rotation", "angle_deg": 120.0},
                                "sector_map": "identity",
                                "antiunitary": False,
                            }
                        ],
                    }
                }
            }
        },
    )

    report = reports["C3z"]
    assert report["input_matrix_file"] == "C3_low_raw.npy"
    assert report["source_matrix_role"] == "raw_h_sewing_action"
    assert report["source_gauge"] == "raw_saved_TAPW"
    assert report["target_role"] == "continuum_internal_rep"
    assert report["inference_used"] is False
    assert "off_support_rel" in report["report"]
    assert "group_residuals" in report["report"]
    assert "distance_mod_global_phase" in report["report"]


def test_no_raw_action_in_production_symmetrization(tmp_path: Path) -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    heff = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=complex)
    np.save(tmp_path / "q1.npy", q)
    np.save(tmp_path / "q2.npy", q)
    np.save(tmp_path / "kpoints.npy", q)
    out_dir = tmp_path / "project"
    out_dir.mkdir()
    np.save(out_dir / "heff_list.npy", heff)
    np.save(out_dir / "heff_eig.npy", np.linalg.eigvalsh(heff))
    (tmp_path / "source.yaml").write_text(
        yaml.safe_dump(
            {
                "material": {"qset1_file": "q1.npy", "qset2_file": "q2.npy"},
                "plot": {},
                "project": {"out_dir": "project"},
            }
        ),
        encoding="utf-8",
    )
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    np.save(symm_dir / "C3_low_raw.npy", np.eye(2, dtype=complex))
    (symm_dir / "manifest.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120},
                        "sector_map": "identity",
                        "spin_map": "identity",
                        "valley_map": "identity",
                        "residual": 0.0,
                        "leakage": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "source_config": "source.yaml",
                "coordinate_frame": {"rotation_deg": 0},
                "valley_model": {
                    "lattice": "hexagonal",
                    "system": "bilayer",
                    "valley_type": "Gamma",
                    "mode": "single_valley",
                    "active_valleys": ["Gamma"],
                    "spin_convention": "spinless_effective",
                    "allowed_internal_symmetries": ["C3z"],
                    "external_sewing_symmetries": [],
                },
                "kpoints_file": "kpoints.npy",
                "symmetry_source": {
                    "type": "kp_symm_output",
                    "path": "symm",
                    "use": "raw",
                    "matrix_kind": "action",
                    "operations": ["C3z"],
                },
                "model": {
                    "n_orb": [1, 1],
                    "nlow_state": [1, 1],
                        "bM": {"bM1": [1.0, 0.0], "bM2": [0.5, float(np.sqrt(3.0) / 2.0)]},
                    "harmonics": {"intra": {1: "zero"}, "inter": {1: "zero"}},
                    "max_order": {"Kinect": 0, "intra": 0, "inter": 0},
                    "symmetry_map": {"Kinect": [{"name": "C3z"}], "intra": [], "inter": []},
                },
                "fit": {"indices": [0]},
                "bands": {"indices": [0], "compare_to_heff": False},
                "output": {"dir": "model_out"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactification"):
        build_moire_config_from_file(cfg_path)
