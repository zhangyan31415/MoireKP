from __future__ import annotations

import numpy as np

from kp.blocks import resolve_project_gauge_anchors


def test_gamma_auto_gauge_preserves_physical_layer_rows_for_1_plus_2_layout() -> None:
    resolved, report = resolve_project_gauge_anchors(
        np.diag([0.0, 1.0, 2.0]).astype(np.complex128),
        q_count=1,
        orb_per_layer0=1,
        num_layer_list=[1, 2],
        spin="up",
        Qlayer_list=[
            [np.zeros((1, 2))],
            [np.zeros((1, 2)), np.zeros((1, 2))],
        ],
        num_orb_per_layer_list=[[1], [1, 1]],
        nlow_state_list=[[], [0], [1]],
        norb_fix_list=None,
        gauge_config="auto",
        mode="gamma",
    )

    assert resolved == [[], [[[0, 1.0]]], [[[1, 1.0]]]]
    assert report.gauge_mode == "auto_scdm"
    assert report.selections[0]["scope"] == "gamma_same_q"
    assert report.selections[0]["bands"] == [0, 1]
