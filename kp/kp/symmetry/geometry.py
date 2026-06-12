from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def rotate_vector(vector: Sequence[float], angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    return matrix @ np.asarray(vector, dtype=float)


def infer_q_offset_from_qset(qset: np.ndarray, bM1: Sequence[float], bM2: Sequence[float]) -> np.ndarray:
    arr = np.asarray(qset, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return np.zeros(2, dtype=float)
    basis = np.column_stack([np.asarray(bM1, dtype=float), np.asarray(bM2, dtype=float)])
    coeffs = np.linalg.solve(basis, arr.T).T
    frac = ((coeffs - np.floor(coeffs)) + 0.5) % 1.0 - 0.5
    candidates = []
    for row in frac:
        shifted = coeffs - row
        residual = float(np.max(np.linalg.norm((shifted - np.rint(shifted)) @ basis.T, axis=1)))
        candidates.append((residual, float(np.linalg.norm(row)), row))
    best = min(candidates, key=lambda item: (item[0], item[1]))[2]
    return basis @ best


def sector_qset(sector: Mapping[str, Any], Q_set1: np.ndarray, Q_set2: np.ndarray) -> np.ndarray:
    qset_name = str(sector.get("qset", ""))
    if qset_name == "qset1":
        return Q_set1
    if qset_name == "qset2":
        return Q_set2
    raise ValueError(f"Unsupported sector qset {qset_name!r}; expected qset1/qset2")


def sectors_with_q_offsets(
    sectors: Sequence[Mapping[str, Any]],
    *,
    Q_set1: np.ndarray,
    Q_set2: np.ndarray,
    bM1: np.ndarray,
    bM2: np.ndarray,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sector in sectors:
        row = dict(sector)
        if "q_offset" not in row:
            row["q_offset"] = infer_q_offset_from_qset(sector_qset(row, Q_set1, Q_set2), bM1, bM2).tolist()
            row["_q_offset_inferred"] = True
        else:
            row["_q_offset_inferred"] = False
        out.append(row)
    return out


def _candidate_score_for_bM(vector: np.ndarray) -> tuple[float, float, float]:
    theta = float(np.mod(np.arctan2(vector[1], vector[0]), 2.0 * np.pi))
    circular_angle = min(theta, 2.0 * np.pi - theta)
    if circular_angle < 1.0e-8:
        circular_angle = 0.0
    return (circular_angle, theta, float(np.linalg.norm(vector)))


def canonical_bM_pair_from_candidates(
    candidates: list[np.ndarray],
    *,
    angle_deg: float,
    tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = [np.asarray(vector, dtype=float) for vector in candidates if np.linalg.norm(vector) > 1.0e-12]
    if not vectors:
        raise ValueError("Cannot infer bM vectors from an empty candidate set")

    min_norm = min(float(np.linalg.norm(vector)) for vector in vectors)
    tol_abs = float(tol if tol is not None else max(1.0e-10, min_norm * 1.0e-6))
    first_shell = [vector for vector in vectors if abs(np.linalg.norm(vector) - min_norm) <= tol_abs]
    bM1 = np.array(min(first_shell, key=_candidate_score_for_bM), dtype=float)

    target_angle = np.deg2rad(float(angle_deg))
    b1_norm = float(np.linalg.norm(bM1))
    b2_candidates = []
    for vector in first_shell:
        cross = float(bM1[0] * vector[1] - bM1[1] * vector[0])
        if cross <= tol_abs * b1_norm:
            continue
        dot = float(np.dot(bM1, vector) / (b1_norm * np.linalg.norm(vector)))
        angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        b2_candidates.append((abs(angle - target_angle), _candidate_score_for_bM(vector), vector))

    if b2_candidates:
        bM2 = np.array(min(b2_candidates, key=lambda item: (item[0], item[1]))[2], dtype=float)
    else:
        bM2 = rotate_vector(bM1, float(angle_deg))
    return bM1, bM2


def bM_candidates_from_q_distances(Q_set1: np.ndarray, Q_set2: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for Q in (np.asarray(Q_set1, dtype=float), np.asarray(Q_set2, dtype=float)):
        for i in range(len(Q)):
            for j in range(len(Q)):
                if i != j:
                    candidates.append(Q[i] - Q[j])
    return candidates
