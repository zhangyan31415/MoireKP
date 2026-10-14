from __future__ import annotations

import numpy as np

import kp.cli as cli


def test_non_gamma_ref_q_reports_third_physical_layer() -> None:
    assert cli._plot_report_layer_for_ref_q(
        50,
        num_layer_list=[1, 2],
        q_lengths=[18, 18],
        index_order=None,
    ) == 2


def test_non_gamma_ref_q_reports_layer_after_q_norm_sorting() -> None:
    order = np.concatenate(
        [
            np.arange(18),
            18 + np.arange(17, -1, -1),
            36 + np.arange(17, -1, -1),
        ]
    )

    assert cli._plot_report_layer_for_ref_q(
        50,
        num_layer_list=[1, 2],
        q_lengths=[18, 18],
        index_order=order,
    ) == 2
