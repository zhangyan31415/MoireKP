import pytest


ROUND7_ROWS = [
    {
        "case": "mote2_K1_center_C2T",
        "operation": "C2T",
        "residual_H_raw": 5.659200888954053e-06,
        "g_perm_max_delta": 1.496202286144631e-11,
        "square_residual": 2.0151456826347057e-11,
    },
    {
        "case": "mote2_K1_q1_C2T",
        "operation": "C2T",
        "residual_H_raw": 5.663810036822519e-06,
        "g_perm_max_delta": 1.496202286144631e-11,
        "square_residual": 2.0151456826347057e-11,
    },
    {
        "case": "snse2_M1_q1_T",
        "operation": "M_T",
        "residual_H_raw": 3.145887499597861e-06,
        "g_perm_max_delta": 5.551115123125783e-17,
        "square_residual": 4.5324665183683945e-17,
    },
    {
        "case": "snse2_M1_q2_T",
        "operation": "M_T",
        "residual_H_raw": 3.1453501088520705e-06,
        "g_perm_max_delta": 5.551115123125783e-17,
        "square_residual": 4.5324665183683945e-17,
    },
    {
        "case": "snse2_M1_q1_C2",
        "operation": "C2",
        "residual_H_raw": 0.0016780778139397546,
        "g_perm_max_delta": 1.2095680956576634e-10,
        "square_residual": 7.871721543270746e-06,
    },
    {
        "case": "snse2_M1_q2_C2",
        "operation": "C2",
        "residual_H_raw": 0.001678201236630244,
        "g_perm_max_delta": 1.2095680956576634e-10,
        "square_residual": 7.871721543270746e-06,
    },
    {
        "case": "snse2_M1_q3_C2",
        "operation": "C2",
        "residual_H_raw": 0.0016789814603395475,
        "g_perm_max_delta": 1.2095680956576634e-10,
        "square_residual": 7.871721543270746e-06,
    },
]


@pytest.mark.parametrize("row", ROUND7_ROWS, ids=[row["case"] for row in ROUND7_ROWS])
def test_round7_accepted_residuals_stay_below_thresholds(row):
    if row["operation"] == "C2T":
        assert row["residual_H_raw"] <= 1.0e-4
    elif row["operation"] == "M_T":
        assert row["residual_H_raw"] <= 1.0e-4
    elif row["operation"] == "C2":
        assert row["residual_H_raw"] <= 1.0e-2
    else:
        raise AssertionError(f"Unexpected operation {row['operation']}")

    assert row["g_perm_max_delta"] <= 1.0e-8
    assert row["square_residual"] <= 1.0e-5
