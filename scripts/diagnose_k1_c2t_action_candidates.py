#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from kp.model.exactify_representation import (
    OperationAction,
    analyze_block_support,
    analyze_monomial_support,
    build_basis_labels,
    build_block_groups,
    build_label_action,
    infer_q_offset_from_qset,
)


def _reflection_matrix(axis_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(axis_deg))
    axis = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    return 2.0 * np.outer(axis, axis) - np.eye(2)


def _infer_hex_bm(q1: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([q1, q2])
    distances = sorted(
        float(np.linalg.norm(points[i] - points[j]))
        for i in range(len(points))
        for j in range(i + 1, len(points))
        if np.linalg.norm(points[i] - points[j]) > 1.0e-10
    )
    if not distances:
        raise ValueError("Cannot infer bM vectors from a single Q point")
    scale = distances[0]
    return np.array([scale, 0.0], dtype=float), np.array([0.5 * scale, np.sqrt(3.0) * 0.5 * scale], dtype=float)


def _candidate(name: str, axis_deg: float, sector_map: str) -> OperationAction:
    return OperationAction(
        name=name,
        canonical_name=name,
        antiunitary=True,
        k_map={"type": "reflection", "axis_deg": float(axis_deg), "in_model_frame": True},
        R=_reflection_matrix(axis_deg),
        sector_map=sector_map,
        q_map={"type": "reflection", "axis_deg": float(axis_deg), "in_model_frame": True},
        central_phase=1.0 + 0.0j,
        group_relations=[],
        source="diagnostic",
    )


def _action_payload(op: OperationAction) -> dict[str, Any]:
    return {
        "k_map": op.k_map,
        "q_map": op.q_map,
        "sector_map": op.sector_map,
        "antiunitary": op.antiunitary,
    }


def diagnose(artifact_dir: Path) -> dict[str, Any]:
    required = {
        "matrix": artifact_dir / "C2T_low_raw.npy",
        "q1": artifact_dir / "q_model_layer1.npy",
        "q2": artifact_dir / "q_model_layer2.npy",
        "manifest": artifact_dir / "manifest.json",
    }
    missing = {key: str(path) for key, path in required.items() if not path.exists()}
    if missing:
        return {"status": "skipped", "reason": "missing input files", "missing": missing}

    matrix = np.asarray(np.load(required["matrix"]), dtype=complex)
    q1 = np.asarray(np.load(required["q1"]), dtype=float)
    q2 = np.asarray(np.load(required["q2"]), dtype=float)
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    operations = {row.get("operation"): row for row in manifest.get("operations", []) if isinstance(row, dict)}
    c2t_row = operations.get("C2T", {})
    try:
        bM1, bM2 = _infer_hex_bm(q1, q2)
        sectors = [
            {"name": "L1", "qset": "qset1", "q_offset": infer_q_offset_from_qset(q1, bM1, bM2), "n_orb": 1},
            {"name": "L2", "qset": "qset2", "q_offset": infer_q_offset_from_qset(q2, bM1, bM2), "n_orb": 1},
        ]
        labels = build_basis_labels(
            Q_set1=q1,
            Q_set2=q2,
            sectors=sectors,
            n_orb1=1,
            n_orb2=1,
            bM1=bM1,
            bM2=bM2,
            tol=1.0e-6,
            q_offsets={sector["name"]: np.asarray(sector["q_offset"], dtype=float) for sector in sectors},
        )
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "could not infer integer Q lattice from artifact q_model files",
            "error": str(exc),
            "artifact_dir": str(artifact_dir),
            "source_action": c2t_row.get("source_action"),
            "manifest_model_action": c2t_row.get("model_action"),
        }
    candidates = [
        _candidate("C2T", 90.0, "identity"),
        _candidate("C2T", 90.0, "layer_exchange"),
        _candidate("C2T", 180.0, "identity"),
        _candidate("C2T", 180.0, "layer_exchange"),
    ]
    rows: list[dict[str, Any]] = []
    for op in candidates:
        row: dict[str, Any] = {"action": _action_payload(op)}
        try:
            label_action = build_label_action(labels, op, bM1, bM2, {"L1": sectors[0]["q_offset"], "L2": sectors[1]["q_offset"]}, 1.0e-6)
            row["perm_complete"] = bool(label_action.diagnostics["perm_complete"])
            row["missing_count"] = int(label_action.diagnostics["missing_count"])
            if not label_action.missing:
                mono = analyze_monomial_support(matrix, label_action.perm)
                row["off_support_rel"] = float(mono.off_support_rel)
                row["amplitude_deviation"] = float(max(abs(mono.amplitude_min - 1.0), abs(mono.amplitude_max - 1.0)))
                try:
                    blocks = build_block_groups(labels, op, bM1, bM2, {"L1": sectors[0]["q_offset"], "L2": sectors[1]["q_offset"]}, 1.0e-6)
                    block = analyze_block_support(matrix, blocks)
                    row["block_off_support_rel"] = float(block.off_support_rel)
                    row["block_amplitude_deviation"] = float(max(abs(block.amplitude_min - 1.0), abs(block.amplitude_max - 1.0)))
                except Exception as exc:
                    row["block_error"] = str(exc)
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    selectable = [row for row in rows if row.get("perm_complete")]
    selected = min(selectable, key=lambda row: (row.get("off_support_rel", float("inf")), row.get("amplitude_deviation", float("inf"))), default=None)
    return {
        "status": "ok",
        "artifact_dir": str(artifact_dir),
        "source_action": c2t_row.get("source_action"),
        "manifest_model_action": c2t_row.get("model_action"),
        "candidates": rows,
        "selected_by_support": None if selected is None else selected["action"],
        "action_mismatch": selected is not None and selected["action"] != c2t_row.get("model_action"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default="kp/examples/mote2/3.89/kp/outputs/symm/mote2_3.89_K1",
        help="K1 kp symm output directory containing C2T_low_raw.npy and q_model_layer*.npy",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    result = diagnose(Path(args.artifact_dir))
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
