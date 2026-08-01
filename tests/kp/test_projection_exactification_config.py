from __future__ import annotations

from kp.symmetry.projection import _kp_symm_exactification_config


def test_kp_symm_exactification_config_preserves_joint_safety_budget() -> None:
    config = _kp_symm_exactification_config(
        {
            "joint_exactification": {
                "max_rms_correction": 1.0e-2,
                "max_route_correction": 2.0e-2,
            }
        }
    )

    assert config["joint_exactification"] == {
        "max_rms_correction": 1.0e-2,
        "max_route_correction": 2.0e-2,
    }
