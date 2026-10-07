from __future__ import annotations

import json

import numpy as np

from kp.basis.selection import write_basis_selection_report
from kp.blocks import resolve_project_gauge_anchors


def test_auto_gauge_report_keeps_missing_symmetry_metrics_null(tmp_path) -> None:
    _resolved, report = resolve_project_gauge_anchors(
        np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128),
        q_count=1,
        orb_per_layer0=2,
        num_layer_list=[1, 1],
        spin="up",
        Qlayer_list=[[np.zeros((1, 2))], [np.zeros((1, 2))]],
        num_orb_per_layer_list=[[2], [2]],
        nlow_state_list=[[0], [0]],
        norb_fix_list="auto",
        gauge_config=None,
        mode="K1",
    )

    write_basis_selection_report(tmp_path, report)

    payload = json.loads((tmp_path / "basis_selection.json").read_text(encoding="utf-8"))
    assert payload["symmetry_closure_quality"]["status"] == "not_available"
    assert payload["symmetry_closure_quality"]["subspace_leakage"] is None
    assert payload["symmetry_closure_quality"]["subspace_leakage"] != 0
