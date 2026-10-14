from __future__ import annotations

import numpy as np

from kp.blocks import resolve_project_gauge_anchors


def test_explicit_norb_fix_list_stays_manual_and_unmodified() -> None:
    manual = [[[[1, 1.0]]], [[[0, -1.0]]]]
    resolved, report = resolve_project_gauge_anchors(
        np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128),
        q_count=1,
        orb_per_layer0=2,
        num_layer_list=[1, 1],
        spin="up",
        Qlayer_list=[[np.zeros((1, 2))], [np.zeros((1, 2))]],
        num_orb_per_layer_list=[[2], [2]],
        nlow_state_list=[[0], [0]],
        norb_fix_list=manual,
        gauge_config=None,
        mode="K1",
    )

    assert resolved is manual
    assert report.gauge_mode == "manual_norb_fix_list"
    assert report.resolved_norb_fix_list == manual
