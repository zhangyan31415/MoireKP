from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from kp.model.pipeline import build_moire_config_from_file
from kp.identity import hash_array
from kp.symmetry.exactify_representation import (
    BasisLabel,
    OperationAction,
    analyze_monomial_support,
    antiunitary_square_residual,
    build_label_action,
    compare_operation_convention,
    exactify_1d_monomial_phases,
    exactify_block_monomial_representation,
    exactify_loaded_symmetry_source,
    nearest_root_of_unity,
    root_of_unity,
)
from kp.symmetry.geometry import infer_q_offset_from_qset


def _source_meta(*, antiunitary: bool = False, source_matrix_role: str = "raw_h_sewing_action") -> dict[str, object]:
    return {
        "source_matrix_role": source_matrix_role,
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


def test_infer_q_offset_handles_branch_cut_half_coset() -> None:
    b1, b2 = _hex_bm()
    offset = 0.5 * b2
    qset = np.array(
        [
            offset + b1,
            offset - b1,
            offset + 2.0 * b1 - b2,
            offset - b1 + b2,
        ],
        dtype=float,
    )

    inferred = infer_q_offset_from_qset(qset, b1, b2)
    coeffs = np.linalg.solve(np.column_stack([b1, b2]), (qset - inferred).T).T

    np.testing.assert_allclose(coeffs, np.rint(coeffs), atol=1.0e-10)


def test_label_action_uses_orbital_map_for_layer_exchange() -> None:
    b1, b2 = _hex_bm()
    q = np.zeros(2)
    labels = [
        BasisLabel(index=0, sector="L1", q_integer=(0, 0), q_vector=q, orbital=1, internal_label="Bi_s"),
        BasisLabel(index=1, sector="L1", q_integer=(0, 0), q_vector=q, orbital=2, internal_label="I_s"),
        BasisLabel(index=2, sector="L2", q_integer=(0, 0), q_vector=q, orbital=1, internal_label="I_s"),
        BasisLabel(index=3, sector="L2", q_integer=(0, 0), q_vector=q, orbital=2, internal_label="Bi_s"),
    ]
    action = OperationAction(
        name="C2",
        canonical_name="C2",
        antiunitary=False,
        k_map={"type": "identity"},
        R=np.eye(2),
        sector_map="layer_exchange",
        q_map={"type": "identity"},
        orbital_map={"L1": {1: 2, 2: 1}, "L2": {1: 2, 2: 1}},
        central_phase=complex(1.0, 0.0),
        group_relations=[],
        source="test",
    )

    label_action = build_label_action(labels, action, b1, b2, {"L1": q, "L2": q}, tol=1.0e-8)

    assert label_action.missing == []
    assert label_action.perm.tolist() == [3, 2, 1, 0]


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


def _make_spinful_tr_exchange_matrix(
    labels: list[BasisLabel],
    action: OperationAction,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    label_action = build_label_action(labels, action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    mat = np.zeros((len(labels), len(labels)), dtype=complex)
    for label in labels:
        phase = -1.0 if label.sector == "L1" else 1.0
        mat[label_action.perm[label.index], label.index] = phase
    return mat


def _make_matrix_for_action(
    labels: list[BasisLabel],
    action: OperationAction,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    label_action = build_label_action(labels, action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    mat = np.zeros((len(labels), len(labels)), dtype=complex)
    for source, target in enumerate(label_action.perm):
        mat[target, source] = 1.0 + 0.0j
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


def _gamma_spinful_c3_four_state_template() -> np.ndarray:
    sqrt3 = np.sqrt(3.0)
    return np.array(
        [
            [-0.25 - 0.25j * sqrt3, -0.25 * sqrt3 - 0.75j, 0.0, 0.0],
            [0.25 * sqrt3 + 0.75j, -0.25 - 0.25j * sqrt3, 0.0, 0.0],
            [0.0, 0.0, -0.25 + 0.25j * sqrt3, -0.25 * sqrt3 + 0.75j],
            [0.0, 0.0, 0.25 * sqrt3 - 0.75j, -0.25 + 0.25j * sqrt3],
        ],
        dtype=np.complex128,
    )


def test_block_exactification_canonicalizes_gamma_spinful_c3_four_state() -> None:
    expected = _gamma_spinful_c3_four_state_template()
    phase_gauge = np.diag(np.exp(1j * np.array([8.0e-8, -8.0e-8, 5.0e-8, -5.0e-8])))
    mixing = np.eye(4, dtype=np.complex128)
    angle = 3.0e-8
    mixing[np.ix_([0, 2], [0, 2])] = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.complex128,
    )
    gauge = mixing @ phase_gauge
    raw = gauge @ expected @ gauge.conj().T

    exact, report = exactify_block_monomial_representation(
        raw,
        [(np.arange(4, dtype=int), np.arange(4, dtype=int))],
        operation_name="C3z",
        power=3,
        central_phase=-1.0 + 0.0j,
        antiunitary=False,
        algebraic_template="gamma_spinful_c3_4",
        algebraic_match_tol=1.0e-5,
    )

    np.testing.assert_array_equal(exact, expected)
    np.testing.assert_array_equal(exact[:2, 2:], np.zeros((2, 2), dtype=np.complex128))
    np.testing.assert_array_equal(exact[2:, :2], np.zeros((2, 2), dtype=np.complex128))
    assert report.algebraic_canonicalization["status"] == "canonicalized"
    assert report.algebraic_canonicalization["template"] == "gamma_spinful_c3_4"
    assert report.algebraic_canonicalization["match_distance"] < 1.0e-5


def test_block_exactification_auto_does_not_snap_unrecognized_four_state_block() -> None:
    rng = np.random.default_rng(17)
    q, _r = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    roots = np.diag(
        [
            -1.0 + 0.0j,
            -1.0 + 0.0j,
            np.exp(1j * np.pi / 3.0),
            np.exp(-1j * np.pi / 3.0),
        ]
    )
    raw = q @ roots @ q.conj().T

    exact, report = exactify_block_monomial_representation(
        raw,
        [(np.arange(4, dtype=int), np.arange(4, dtype=int))],
        operation_name="C3z",
        power=3,
        central_phase=-1.0 + 0.0j,
        antiunitary=False,
        algebraic_template="auto",
        algebraic_match_tol=1.0e-5,
    )

    assert report.algebraic_canonicalization["status"] == "not_recognized"
    assert report.algebraic_canonicalization["template"] == "gamma_spinful_c3_4"
    assert report.algebraic_canonicalization["match_distance"] > 1.0e-5
    assert not np.array_equal(exact, _gamma_spinful_c3_four_state_template())


def test_block_exactification_uses_algebraic_global_c3_root() -> None:
    template = _gamma_spinful_c3_four_state_template()
    omega = -0.5 + 0.5j * np.sqrt(3.0)
    raw = omega * template

    exact, report = exactify_block_monomial_representation(
        raw,
        [(np.arange(4, dtype=int), np.arange(4, dtype=int))],
        operation_name="C3z",
        power=3,
        central_phase=-1.0 + 0.0j,
        antiunitary=False,
        algebraic_template="gamma_spinful_c3_4",
        algebraic_match_tol=1.0e-5,
    )

    np.testing.assert_array_equal(exact, omega * template)
    assert report.algebraic_canonicalization["selected_branch"]["global_root_power"] == 1


def test_loaded_gamma_c3_algebraic_exactification_validates_tr_and_c2_relations() -> None:
    q1 = np.zeros((1, 2), dtype=float)
    q2 = np.zeros((1, 2), dtype=float)
    b1, b2 = _hex_bm()
    internal_c3 = _gamma_spinful_c3_four_state_template()
    phase_gauge = np.diag(np.exp(1j * np.array([8.0e-8, -8.0e-8, 5.0e-8, -5.0e-8])))
    raw_internal_c3 = phase_gauge @ internal_c3 @ phase_gauge.conj().T
    raw_c3 = np.block(
        [
            [raw_internal_c3, np.zeros((4, 4), dtype=np.complex128)],
            [np.zeros((4, 4), dtype=np.complex128), raw_internal_c3],
        ]
    )
    tr_internal = np.zeros((4, 4), dtype=np.complex128)
    tr_internal[0, 2] = -1.0
    tr_internal[1, 3] = -1.0
    tr_internal[2, 0] = 1.0
    tr_internal[3, 1] = 1.0
    tr = np.block(
        [
            [tr_internal, np.zeros((4, 4), dtype=np.complex128)],
            [np.zeros((4, 4), dtype=np.complex128), tr_internal],
        ]
    )
    c2_internal = np.zeros((4, 4), dtype=np.complex128)
    c2_internal[0, 2] = -1j
    c2_internal[1, 3] = 1j
    c2_internal[2, 0] = -1j
    c2_internal[3, 1] = 1j
    c2 = np.block(
        [
            [np.zeros((4, 4), dtype=np.complex128), c2_internal],
            [c2_internal, np.zeros((4, 4), dtype=np.complex128)],
        ]
    )
    orbital_exchange = {"*": {0: 2, 1: 3, 2: 0, 3: 1}}

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                    {
                        **_source_meta(antiunitary=True),
                        "name": "TR",
                        "operation": "TR",
                        "antiunitary": True,
                        "k_map": {"type": "negation"},
                        "q_map": {"type": "negation"},
                        "sector_map": "identity",
                        "orbital_map": orbital_exchange,
                        "group_relations": [
                            {"type": "power", "operation": "TR", "power": 2, "phase": -1.0}
                        ],
                    },
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "operation": "C3z",
                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                        "group_relations": [
                            {"type": "power", "operation": "C3z", "power": 3, "phase": -1.0}
                        ],
                    },
                    {
                        **_source_meta(),
                        "name": "C2",
                        "operation": "C2",
                        "antiunitary": False,
                        "k_map": {"type": "reflection", "axis_deg": 0.0},
                        "q_map": {"type": "reflection", "axis_deg": 0.0},
                        "sector_map": "layer_exchange",
                        "orbital_map": orbital_exchange,
                        "group_relations": [
                            {"type": "power", "operation": "C2", "power": 2, "phase": -1.0}
                        ],
                    },
            ]
        },
        matrices={"TR": tr, "C3z": raw_c3, "C2": c2},
        Q_set1=q1,
        Q_set2=q2,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 4},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 4},
        ],
        n_orb=(4, 4),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "TR": {"support_mode": "monomial", "power": 2, "central_phase": -1.0},
                    "C3z": {
                        "support_mode": "block",
                        "power": 3,
                        "central_phase": -1.0,
                        "algebraic_template": "auto",
                    },
                    "C2": {"support_mode": "monomial", "power": 2, "central_phase": -1.0},
                }
            }
        },
    )

    expected_c3 = np.block(
        [
            [internal_c3, np.zeros((4, 4), dtype=np.complex128)],
            [np.zeros((4, 4), dtype=np.complex128), internal_c3],
        ]
    )
    np.testing.assert_array_equal(exactified["C3z"], expected_c3)
    np.testing.assert_allclose(tr @ expected_c3.conj() @ tr.conj().T, expected_c3, atol=1.0e-14)
    np.testing.assert_allclose(c2 @ expected_c3 @ c2.conj().T, np.linalg.inv(expected_c3), atol=1.0e-14)
    joint = reports["C3z"]["report"]["joint_group_residuals"]
    assert reports["C3z"]["joint_exactification"]["status"] == "certified"
    assert joint["TR_C3z_commute"] < 1.0e-14
    assert joint["C2_C3z_dihedral"] < 1.0e-14


@pytest.mark.parametrize("block_dimension", [1, 2])
def test_loaded_k_valley_generators_are_jointly_idempotent(
    block_dimension: int,
) -> None:
    q = np.zeros((1, 2), dtype=float)
    b1, b2 = _hex_bm()
    root = np.exp(1j * np.pi / 3.0)
    if block_dimension == 1:
        c3_block = np.asarray([[root]], dtype=np.complex128)
    else:
        angle = 0.37
        gauge = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.complex128,
        )
        c3_block = gauge.T @ np.diag([root, np.conjugate(root)]) @ gauge
    c2t_block = np.eye(block_dimension, dtype=np.complex128)
    c3 = np.block(
        [
            [c3_block, np.zeros_like(c3_block)],
            [np.zeros_like(c3_block), c3_block],
        ]
    )
    c2t = np.block(
        [
            [c2t_block, np.zeros_like(c2t_block)],
            [np.zeros_like(c2t_block), c2t_block],
        ]
    )

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C3z",
                    "operation": "C3z",
                    "antiunitary": False,
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
                    "group_relations": [
                        {"type": "power", "operation": "C3z", "power": 3, "phase": -1.0}
                    ],
                },
                {
                    **_source_meta(antiunitary=True),
                    "name": "C2T",
                    "operation": "C2T",
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 0.0},
                    "q_map": {"type": "reflection", "axis_deg": 0.0},
                    "sector_map": "identity",
                    "group_relations": [
                        {"type": "power", "operation": "C2T", "power": 2, "phase": 1.0}
                    ],
                },
            ]
        },
        matrices={"C3z": c3, "C2T": c2t},
        Q_set1=q,
        Q_set2=q,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": block_dimension},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": block_dimension},
        ],
        n_orb=(block_dimension, block_dimension),
        bM1=b1,
        bM2=b2,
        raw_config={
            "exactification": {
                "operations": {
                    "C3z": {"support_mode": "block", "algebraic_template": "none"},
                    "C2T": {"support_mode": "block"},
                }
            }
        },
    )

    np.testing.assert_allclose(exactified["C3z"], c3, atol=2.0e-15)
    np.testing.assert_allclose(exactified["C2T"], c2t, atol=2.0e-15)
    joint = reports["__joint_exactification__"]["report"]
    assert joint["status"] == "certified"
    assert joint["post_relation_residual_max"] <= joint["relation_certification_bound"]
    assert joint["route_correction_max"] < 2.0e-15


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


def test_exactify_c3_global_phase_preserves_orbital_dependent_roots() -> None:
    labels = _k_basis_labels_two_orbital()
    b1, b2 = _hex_bm()
    action = _c3_action()
    label_action = build_label_action(labels, action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    raw = np.zeros((len(labels), len(labels)), dtype=complex)
    for source, target in enumerate(label_action.perm):
        phase = np.exp(1j * np.pi / 3.0) if labels[source].orbital == 1 else np.exp(-1j * np.pi / 3.0)
        raw[target, source] = phase

    exactified, exact_report = exactify_1d_monomial_phases(
        raw,
        label_action.perm,
        labels=labels,
        phase_classes="global",
        allowed_roots=[np.exp(1j * (np.pi + 2.0 * np.pi * m) / 3.0) for m in range(3)],
        operation_name="C3z",
        power=3,
        central_phase=-1.0,
    )

    np.testing.assert_allclose(exactified, raw, atol=1.0e-12)
    assert exact_report.group_residuals["power"] < 1.0e-14
    assert "split_phase_class" in exact_report.notes


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


def test_exactify_spinful_tr_exchange_preserves_antiunitary_square() -> None:
    labels = _k_basis_labels()
    b1, b2 = _hex_bm()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    action = OperationAction(
        name="TR",
        canonical_name="TR",
        antiunitary=True,
        k_map={"type": "negation", "in_model_frame": True},
        R=-np.eye(2, dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "negation", "in_model_frame": True},
        central_phase=-1.0 + 0.0j,
        group_relations=[{"type": "power", "power": 2, "phase": -1.0, "name": "TR^2"}],
        source="test",
    )
    raw = _make_spinful_tr_exchange_matrix(labels, action, b1, b2)

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "TR",
                    "antiunitary": True,
                    "k_map": {"type": "negation", "in_model_frame": True},
                    "q_map": {"type": "negation", "in_model_frame": True},
                    "sector_map": "layer_exchange",
                    "group_relations": [{"type": "power", "power": 2, "phase": -1.0, "name": "TR^2"}],
                }
            ]
        },
        matrices={"TR": raw},
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
                    "TR": {
                        "action_candidates": [
                            {
                                "k_map": action.k_map,
                                "q_map": action.q_map,
                                "sector_map": action.sector_map,
                                "antiunitary": True,
                            }
                        ],
                    }
                }
            }
        },
    )

    assert antiunitary_square_residual(exactified["TR"], -1.0) < 1.0e-14
    assert reports["TR"]["report"]["group_residuals"]["power"] < 1.0e-14


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
                    "name": "TR",
                    "operation_alias": "TR_eff",
                    "canonical_physical_operation": "TR",
                    "representation_level": "effective_single_spin",
                    "effective_name": "TR_eff",
                    "approximation": {"kind": "spin_SU2_effective_block"},
                    "antiunitary": True,
                    "k_map": {"type": "identity"},
                    "sector_map": "identity",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"TR": np.eye(2, dtype=complex)},
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

    np.testing.assert_allclose(exactified["TR"], np.eye(2, dtype=complex))
    assert reports["TR"]["inferred_q_offsets"] == {"bottom": True, "top": True}
    assert reports["TR"]["operation_alias"] == "TR_eff"
    assert reports["TR"]["representation_level"] == "effective_single_spin"
    np.testing.assert_allclose(reports["TR"]["q_offsets"]["bottom"], (-0.5 * b2).tolist())
    np.testing.assert_allclose(reports["TR"]["q_offsets"]["top"], (-0.5 * b1 - 0.5 * b2).tolist())


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
        q_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
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
                                    "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
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


def test_exactify_default_does_not_discover_exchange_or_shifted_reflection_candidates() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    manifest_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
        R=np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float),
        sector_map="identity",
        q_map={"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    label_action = build_label_action(labels, manifest_action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    exact = np.zeros((len(labels), len(labels)), dtype=complex)
    for source, target in enumerate(label_action.perm):
        exact[target, source] = 1.0 + 0.0j

    _exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "C2T",
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "sector_map": "identity",
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
        raw_config={"exactification": {"operations": {"C2T": {"support_mode": "monomial", "power": 2}}}},
    )

    resolution = reports["C2T"]["support_resolution"]
    assert resolution["action_mismatch"] is False
    assert len(resolution["candidate_support_residuals"]) == 1
    only_action = resolution["candidate_support_residuals"][0]["action"]
    assert only_action["sector_map"] == "identity"
    assert only_action["k_map"]["axis_deg"] == pytest.approx(90.0)


def test_exactify_ignores_discovery_switch_and_rejects_wrong_manifest_action() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    support_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    label_action = build_label_action(labels, support_action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    exact = np.zeros((len(labels), len(labels)), dtype=complex)
    for source, target in enumerate(label_action.perm):
        exact[target, source] = 1.0 + 0.0j

    with pytest.raises(ValueError, match="set accept_support_resolved_action=true"):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        **_source_meta(antiunitary=True),
                        "name": "C2T",
                        "antiunitary": True,
                        "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                        "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                        "sector_map": "identity",
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
                    "discover_action_candidates": True,
                    "operations": {"C2T": {"support_mode": "monomial", "power": 2}},
                }
            },
        )


def test_exactify_keeps_manifest_action_when_kp_symm_model_action_matches_support() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    manifest_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    label_action = build_label_action(labels, manifest_action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    exact = np.zeros((len(labels), len(labels)), dtype=complex)
    for source, target in enumerate(label_action.perm):
        exact[target, source] = 1.0 + 0.0j

    _exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "C2T",
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "sector_map": "layer_exchange",
                    "model_basis_action": {
                        "complete": True,
                        "sector_map": "layer_exchange",
                        "items": [
                            {
                                "source_sector": "L1",
                                "source_q_index": 0,
                                "target_sector": "L2",
                                "target_q_index": 0,
                                "q_residual": 0.0,
                            }
                        ],
                    },
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
                "discover_action_candidates": True,
                "accept_support_resolved_action": True,
                "operations": {"C2T": {"support_mode": "monomial", "power": 2}},
            }
        },
    )

    resolution = reports["C2T"]["support_resolution"]
    assert resolution["action_mismatch"] is False
    assert resolution["candidate_source"] == "manifest_model_action"
    assert reports["C2T"]["resolved_action"]["sector_map"] == "layer_exchange"


def test_exactify_ignores_operation_record_support_discovery_flag() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    support_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    label_action = build_label_action(labels, support_action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    exact = np.zeros((len(labels), len(labels)), dtype=complex)
    for source, target in enumerate(label_action.perm):
        exact[target, source] = 1.0 + 0.0j

    with pytest.raises(ValueError, match="off-support pollution"):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        **_source_meta(antiunitary=True),
                        "name": "C2T",
                        "antiunitary": True,
                        "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                        "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                        "sector_map": "identity",
                        "allow_support_discovery": True,
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
            raw_config={"exactification": {"operations": {"C2T": {"support_mode": "monomial", "power": 2}}}},
        )


def test_accept_support_resolved_action_writes_provenance() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    support_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    label_action = build_label_action(labels, support_action, b1, b2, {"L1": np.zeros(2), "L2": np.zeros(2)}, tol=1.0e-8)
    exact = np.zeros((len(labels), len(labels)), dtype=complex)
    for source, target in enumerate(label_action.perm):
        exact[target, source] = 1.0 + 0.0j

    _exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "C2T",
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                    "sector_map": "identity",
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
                "discover_action_candidates": True,
                "accept_support_resolved_action": True,
                "operations": {
                    "C2T": {
                        "support_mode": "monomial",
                        "power": 2,
                        "action_candidates": [
                            {
                                "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                                "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                                "sector_map": "identity",
                            },
                            {
                                "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                                "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                                "sector_map": "layer_exchange",
                            },
                        ],
                    }
                },
            }
        },
    )

    report = reports["C2T"]
    assert report["support_resolution"]["action_mismatch"] is True
    assert report["resolved_action"]["sector_map"] == "layer_exchange"
    assert report["resolved_action"]["provenance"]["accepted_by_user"] is True
    assert report["support_resolution"]["provenance"]["source"] == "support_exactification"


def test_accept_support_resolved_action_reports_support_discovery_source() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    support_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    exact = _make_matrix_for_action(labels, support_action, b1, b2)

    _exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "C2T",
                    "antiunitary": True,
                    "k_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "q_map": {"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
                    "sector_map": "identity",
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
                "discover_action_candidates": True,
                "accept_support_resolved_action": True,
                "operations": {"C2T": {"support_mode": "monomial", "power": 2}},
            }
        },
    )

    resolution = reports["C2T"]["support_resolution"]
    assert resolution["action_mismatch"] is True
    assert resolution["candidate_source"] == "support_discovery"


def test_exactify_strict_rejects_support_discovery_mismatch_even_when_accept_enabled() -> None:
    labels = _k_basis_labels()
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    support_action = OperationAction(
        name="C2T",
        canonical_name="C2T",
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        R=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
        sector_map="layer_exchange",
        q_map={"type": "reflection", "axis_deg": 180.0, "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="test",
    )
    exact = _make_matrix_for_action(labels, support_action, b1, b2)

    with pytest.raises(ValueError, match="differs from manifest model_action in strict mode"):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        **_source_meta(antiunitary=True),
                        "name": "C2T",
                        "antiunitary": True,
                        "k_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                        "q_map": {"type": "reflection", "axis_deg": 90.0, "in_model_frame": True},
                        "sector_map": "identity",
                        "model_basis_action": {
                            "complete": True,
                            "sector_map": "identity",
                            "items": [
                                {
                                    "source_sector": "L1",
                                    "source_q_index": 0,
                                    "target_sector": "L1",
                                    "target_q_index": 0,
                                    "q_residual": 0.0,
                                }
                            ],
                        },
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
                    "strict": True,
                    "require_explicit_action_candidates": False,
                    "discover_action_candidates": True,
                    "accept_support_resolved_action": True,
                    "operations": {"C2T": {"support_mode": "monomial", "power": 2}},
                }
            },
        )


def test_raw_c3z_sewing_action_auto_removes_tiny_internal_mixing() -> None:
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


def test_raw_c3z_auto_prefers_cleaner_block_support_even_when_monomial_passes_threshold() -> None:
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
                "reject_if_off_support_rel_gt": 5.0e-3,
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
                },
            }
        },
    )

    np.testing.assert_allclose(exactified["C3z"], expected, atol=1.0e-12)
    assert reports["C3z"]["preferred_mode"] == "block_monomial"
    assert reports["C3z"]["support_diagnostics"]["off_support_rel"] < expected_cleanup_residual
    assert reports["C3z"]["report"]["joint_group_residuals"]["cleanup_input_off_support_rel"] == pytest.approx(
        expected_cleanup_residual
    )


def test_c3z_bare_representation_block_support_is_not_auto_cleaned() -> None:
    labels, q1, q2, b1, b2, raw, expected, _cleanup_residual = _make_block_mixed_c3_case()

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(source_matrix_role="bare_D0_internal_rep"),
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


def test_spinful_c2_monomial_exactification_preserves_two_cycle_pair_phase() -> None:
    labels = [
        BasisLabel(0, "L1", (0, 0), np.array([0.0, 0.0]), 1, None),
        BasisLabel(1, "L2", (0, 0), np.array([0.0, 0.0]), 1, None),
    ]
    forward = np.exp(2.0j * np.pi / 3.0)
    backward = np.exp(1.0j * np.pi / 3.0)
    raw = np.array(
        [
            [0.0, backward],
            [forward, 0.0],
        ],
        dtype=complex,
    )

    exact, report = exactify_1d_monomial_phases(
        raw,
        [1, 0],
        labels=labels,
        phase_classes="matrix_element",
        allowed_roots=[],
        operation_name="C2",
        power=2,
        central_phase=-1.0 + 0.0j,
        antiunitary=False,
    )

    np.testing.assert_allclose(exact, raw, atol=1.0e-12)
    assert report.group_residuals["power"] < 1.0e-14


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
                    "name": "C2",
                    "operation_alias": "C2_eff",
                    "canonical_physical_operation": "C2",
                    "representation_level": "effective_single_spin",
                    "effective_name": "C2_eff",
                    "approximation": {"kind": "spin_SU2_effective_block"},
                    "antiunitary": False,
                    "k_map": {"type": "identity"},
                    "sector_map": "identity",
                    "q_map": {"type": "identity"},
                }
            ]
        },
        matrices={"C2": exact},
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
                    "C2": {
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

    np.testing.assert_allclose(exactified["C2"], exact, atol=1.0e-12)
    assert reports["C2"]["label_action"]["perm_complete"] is True
    assert reports["C2"]["label_action"]["missing_count"] == 0
    assert reports["C2"]["resolved_action"]["sector_map"] == "layer_exchange"
    assert reports["C2"]["operation_alias"] == "C2_eff"
    assert reports["C2"]["representation_level"] == "effective_single_spin"


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


def test_manifest_group_relation_overrides_legacy_central_phase_config() -> None:
    labels = _k_basis_labels()
    exact, _perm = _make_exact_c3_matrix(labels)
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()

    _out, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C3z",
                    "matrix_file": "C3_low_raw.npy",
                    "antiunitary": False,
                    "k_map": {"type": "rotation", "angle_deg": 120},
                    "sector_map": "identity",
                    "q_map": {"type": "rotation", "angle_deg": 120},
                    "group_relations": [{"type": "power", "power": 3, "phase": -1.0, "name": "C3z^3"}],
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
                "central_phase": {"C3z^3": 1.0},
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

    assert reports["C3z"]["group_relation_source"] == "manifest"
    assert reports["C3z"]["group_relation"]["power"] == 3
    assert reports["C3z"]["group_relation"]["phase"] == [-1.0, 0.0]
    assert reports["C3z"]["report"]["group_residuals"]["power"] == pytest.approx(0.0)


def test_exactify_no_complete_support_candidate_raises_by_default() -> None:
    b1, b2 = _hex_bm()
    q1 = np.array([b1], dtype=float)
    q2 = np.array([b1], dtype=float)

    with pytest.raises(ValueError, match="No candidate produced complete geometry support"):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "matrix_file": "C3_low_raw.npy",
                        "antiunitary": False,
                        "k_map": {"type": "rotation", "angle_deg": 120.0},
                        "q_map": {"type": "rotation", "angle_deg": 120.0},
                        "sector_map": "identity",
                    }
                ]
            },
            matrices={"C3z": np.eye(2, dtype=complex)},
            Q_set1=q1,
            Q_set2=q2,
            sectors=[
                {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
                {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
            ],
            n_orb=(1, 1),
            bM1=b1,
            bM2=b2,
            raw_config={"exactification": {"support_mode": "monomial"}},
        )


def test_exactify_output_dir_removes_stale_reports(tmp_path: Path) -> None:
    labels = _k_basis_labels()
    exact, _perm = _make_exact_c3_matrix(labels)
    q1 = np.array([label.q_vector for label in labels if label.sector == "L1"], dtype=float)
    q2 = np.array([label.q_vector for label in labels if label.sector == "L2"], dtype=float)
    b1, b2 = _hex_bm()
    output_dir = tmp_path / "projection"
    output_dir.mkdir()
    stale_report = output_dir / "c2_eff_exactification_report.json"
    stale_report.write_text("{}", encoding="utf-8")

    exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(),
                    "name": "C3z",
                    "matrix_file": "C3_low_raw.npy",
                    "antiunitary": False,
                    "k_map": {"type": "rotation", "angle_deg": 120.0},
                    "q_map": {"type": "rotation", "angle_deg": 120.0},
                    "sector_map": "identity",
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
        raw_config={"exactification": {"support_mode": "monomial"}},
        output_dir=output_dir,
    )

    assert not stale_report.exists()
    assert (output_dir / "c3z_exactification_report.json").exists()


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


def test_loaded_m_valley_generators_use_joint_exactification(tmp_path: Path) -> None:
    q = np.zeros((1, 2), dtype=float)
    b1, b2 = _hex_bm()
    phase_error = 8.5e-6
    layer_exchange_phase = np.exp(1j * (np.pi - phase_error))
    tr = np.eye(2, dtype=np.complex128)
    c2 = np.array(
        [
            [0.0, np.conjugate(layer_exchange_phase)],
            [layer_exchange_phase, 0.0],
        ],
        dtype=np.complex128,
    )

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "TR",
                    "operation": "TR",
                    "antiunitary": True,
                    "k_map": {"type": "negation", "in_model_frame": True},
                    "q_map": {"type": "negation", "in_model_frame": True},
                    "sector_map": "identity",
                    "group_relations": [
                        {
                            "type": "power",
                            "operation": "TR",
                            "power": 2,
                            "phase": 1.0,
                        }
                    ],
                },
                {
                    **_source_meta(),
                    "name": "C2",
                    "operation": "C2",
                    "antiunitary": False,
                    "k_map": {
                        "type": "reflection",
                        "axis_deg": 0.0,
                        "in_model_frame": True,
                    },
                    "q_map": {
                        "type": "reflection",
                        "axis_deg": 0.0,
                        "in_model_frame": True,
                    },
                    "sector_map": "layer_exchange",
                    "group_relations": [
                        {
                            "type": "power",
                            "operation": "C2",
                            "power": 2,
                            "phase": 1.0,
                        }
                    ],
                },
            ]
        },
        matrices={"TR": tr, "C2": c2},
        Q_set1=q,
        Q_set2=q,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
        ],
        n_orb=(1, 1),
        bM1=b1,
        bM2=b2,
        raw_config={"exactification": {}},
        output_dir=tmp_path,
    )

    stage1_residual = float(
        np.linalg.norm(tr @ c2.conj() - c2 @ tr) / np.sqrt(c2.shape[0])
    )
    final_residual = float(
        np.linalg.norm(
            exactified["TR"] @ exactified["C2"].conj()
            - exactified["C2"] @ exactified["TR"]
        )
        / np.sqrt(c2.shape[0])
    )
    assert stage1_residual > 1.0e-6
    assert final_residual < 5.0e-15
    joint = reports["__joint_exactification__"]
    assert joint["metadata"]["version"] == "joint_block_representation_v1"
    assert joint["metadata"]["artifact_hash"]
    assert reports["TR"]["joint_exactification"]["status"] == "certified"
    assert reports["C2"]["joint_exactification"]["status"] == "certified"
    assert reports["C2"]["joint_exactification"]["route_correction_rms"] > 1.0e-6
    np.testing.assert_array_equal(
        np.load(tmp_path / "exactified_C2.npy"),
        exactified["C2"],
    )
    persisted_report = json.loads(
        (tmp_path / "c2_exactification_report.json").read_text(encoding="utf-8")
    )
    assert persisted_report["joint_exactification"]["status"] == "certified"
    assert (
        persisted_report["joint_exactification"]["artifact_hash"]
        == joint["metadata"]["artifact_hash"]
    )


def test_joint_exactification_preserves_raw_spinful_tr_c2_common_phase_frame() -> None:
    q = np.zeros((1, 2), dtype=float)
    b1, b2 = _hex_bm()
    tr_block = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
    c2_forward = np.diag(
        [np.exp(1.0j * np.pi / 3.0), np.exp(2.0j * np.pi / 3.0)]
    ).astype(np.complex128)
    c2_backward = -c2_forward.conjugate().T
    tr = np.block(
        [
            [tr_block, np.zeros((2, 2), dtype=np.complex128)],
            [np.zeros((2, 2), dtype=np.complex128), -tr_block],
        ]
    )
    c2 = np.block(
        [
            [np.zeros((2, 2), dtype=np.complex128), c2_backward],
            [c2_forward, np.zeros((2, 2), dtype=np.complex128)],
        ]
    )

    exactified, reports = exactify_loaded_symmetry_source(
        loaded_metadata={
            "operations": [
                {
                    **_source_meta(antiunitary=True),
                    "name": "TR",
                    "operation": "TR",
                    "antiunitary": True,
                    "k_map": {"type": "negation", "in_model_frame": True},
                    "q_map": {"type": "negation", "in_model_frame": True},
                    "sector_map": "identity",
                    "orbital_map": {"*": {0: 1, 1: 0}},
                    "group_relations": [
                        {"type": "power", "operation": "TR", "power": 2, "phase": -1.0}
                    ],
                },
                {
                    **_source_meta(),
                    "name": "C2",
                    "operation": "C2",
                    "antiunitary": False,
                    "k_map": {
                        "type": "reflection",
                        "axis_deg": 0.0,
                        "in_model_frame": True,
                    },
                    "q_map": {
                        "type": "reflection",
                        "axis_deg": 0.0,
                        "in_model_frame": True,
                    },
                    "sector_map": "layer_exchange",
                    "group_relations": [
                        {"type": "power", "operation": "C2", "power": 2, "phase": -1.0}
                    ],
                },
            ]
        },
        matrices={"TR": tr, "C2": c2},
        Q_set1=q,
        Q_set2=q,
        sectors=[
            {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 2},
            {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 2},
        ],
        n_orb=(2, 2),
        bM1=b1,
        bM2=b2,
        raw_config={"exactification": {}},
    )

    np.testing.assert_allclose(exactified["TR"], tr, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(exactified["C2"], c2, atol=1.0e-14, rtol=0.0)
    for name in ("TR", "C2"):
        assert np.isfinite(
            reports[name]["report"]["distance_mod_global_phase"]
        )
    joint = reports["__joint_exactification__"]
    assert joint["phase_preserving_source"]["status"] == "certified"
    assert joint["report"]["post_relation_residual_max"] < 1.0e-14
    assert joint["report"]["route_correction_max"] < 1.0e-14


def test_required_joint_exactification_rejects_unsupported_presentation() -> None:
    q = np.zeros((1, 2), dtype=float)
    b1, b2 = _hex_bm()
    with pytest.raises(
        ValueError,
        match="joint exactification is required.*unsupported",
    ):
        exactify_loaded_symmetry_source(
            loaded_metadata={
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C2",
                        "operation": "C2",
                        "antiunitary": False,
                        "k_map": {"type": "reflection", "axis_deg": 0.0},
                        "q_map": {"type": "reflection", "axis_deg": 0.0},
                        "sector_map": "identity",
                        "group_relations": [
                            {"type": "power", "operation": "C2", "power": 2, "phase": 1.0}
                        ],
                    }
                ]
            },
            matrices={"C2": np.eye(2, dtype=np.complex128)},
            Q_set1=q,
            Q_set2=q,
            sectors=[
                {"name": "L1", "qset": "qset1", "q_offset": [0.0, 0.0], "n_orb": 1},
                {"name": "L2", "qset": "qset2", "q_offset": [0.0, 0.0], "n_orb": 1},
            ],
            n_orb=(1, 1),
            bM1=b1,
            bM2=b2,
            raw_config={
                "exactification": {
                    "joint_exactification": {"required": True}
                }
            },
        )


def test_model_rejects_raw_kp_symm_action_without_source_exactification(tmp_path: Path) -> None:
    q = np.array([[0.0, 0.0]], dtype=float)
    heff = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=complex)
    np.save(tmp_path / "q1.npy", q)
    np.save(tmp_path / "q2.npy", q)
    np.save(tmp_path / "kpoints.npy", q)
    out_dir = tmp_path / "outputs" / "K1" / "q06" / "projection"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "heff.npy", heff)
    identity = {
        "identity_schema": "moirekp.artifact-identity.v1",
        "input_hash": "input-fixture",
        "config_hash": "config-fixture",
        "basis_hash": "basis-fixture",
        "package_version": "0.1.0",
        "schema_version": 2,
        "k_indices_hash": hash_array(np.asarray([0], dtype=np.int64)),
        "heff_hash": hash_array(heff),
    }
    scalar_identity = {key: np.asarray(value) for key, value in identity.items()}
    np.savez(out_dir / "basis.npz", **scalar_identity)
    np.savez(
        out_dir / "wavefunctions.npz",
        wavefunctions=np.eye(2, dtype=np.complex128)[None, :, :],
        k_indices=np.asarray([0], dtype=int),
        **scalar_identity,
    )
    symm_dir = tmp_path / "symm"
    symm_dir.mkdir()
    np.save(symm_dir / "C3_low_raw.npy", np.eye(2, dtype=complex))
    raw_operation = {
        **_source_meta(),
        "name": "C3z",
        "operation": "C3",
        "matrix_file": "representations.npz",
        "matrix_array_key": "C3z",
        "matrix_kind": "action",
        "matrix_source": "raw_h_action_projection",
        "status": "projected",
        "exactification_owner": "kp_symm",
        "basis_hash": identity["basis_hash"],
        "k_indices_hash": identity["k_indices_hash"],
        "heff_hash": identity["heff_hash"],
        "antiunitary": False,
        "k_map": {"type": "rotation", "angle_deg": 120},
        "q_map": {"type": "rotation", "angle_deg": 120},
        "sector_map": "identity",
        "spin_map": "identity",
        "valley_map": "identity",
        "residual": 0.0,
        "leakage": 0.0,
    }
    np.savez_compressed(
        symm_dir / "representations.npz",
        C3z=np.eye(2, dtype=np.complex128),
        __metadata_json__=np.asarray(
            json.dumps(
                {
                    "artifact_identity": identity,
                    "exactification_owner": "kp_symm",
                    "frame": {"q_transform": {"rotation_deg": 0.0}},
                    "operations": [raw_operation],
                },
                sort_keys=True,
            )
        ),
    )
    (symm_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frame": {"q_transform": {"rotation_deg": 0.0}},
                "requires_model_exactification": True,
                "operations": [
                    {
                        **_source_meta(),
                        "name": "C3z",
                        "operation": "C3",
                        "matrix_file": "C3_low_raw.npy",
                        "matrix_kind": "action",
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
                "case": {"profile": "K1", "q_shell": "q06", "output_root": "outputs"},
                "material": {"qset1_file": "q1.npy", "qset2_file": "q2.npy"},
                "plot": {},
                "project": {},
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
                    "operations": [
                        {
                            **_source_meta(),
                            "name": "C3z",
                            "operation": "C3",
                            "matrix_kind": "action",
                            "antiunitary": False,
                            "k_map": {"type": "rotation", "angle_deg": 120.0},
                            "q_map": {"type": "rotation", "angle_deg": 120.0},
                            "sector_map": "identity",
                        }
                    ],
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

    with pytest.raises(ValueError, match="lacks complete exactification provenance"):
        build_moire_config_from_file(cfg_path)

    assert not (tmp_path / "model_out" / "symmetry_source_matrix_projection").exists()
    assert not (tmp_path / "model_out" / "symmetry_exactification").exists()
