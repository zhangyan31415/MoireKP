from __future__ import annotations

import math

import pytest

from kp.model.model_selection import (
    CandidateScore,
    FamilyOrders,
    select_simplest_near_best,
)


def _candidate(
    name: str,
    *,
    loss: float,
    loss_se: float,
    parameters: int,
    orders: tuple[int, int, int] = (2, 0, 0),
    overlap: float = 0.98,
    maximum: float = 3.0,
    certified: bool = True,
    guards_passed: bool = True,
) -> CandidateScore:
    return CandidateScore(
        name=name,
        orders=FamilyOrders(*orders),
        independent_real_parameters=parameters,
        weighted_rms_mev=loss,
        weighted_rms_se_mev=loss_se,
        weighted_max_mev=maximum,
        mean_subspace_overlap=overlap,
        certified=certified,
        guards_passed=guards_passed,
    )


def test_one_se_selects_simplest_candidate_on_near_best_plateau() -> None:
    best = _candidate(
        "best",
        loss=0.80,
        loss_se=0.20,
        parameters=11,
        orders=(4, 2, 1),
    )
    simpler = _candidate(
        "simpler",
        loss=0.97,
        loss_se=0.05,
        parameters=5,
        orders=(2, 1, 1),
    )
    outside = _candidate(
        "outside",
        loss=1.01,
        loss_se=0.01,
        parameters=2,
        orders=(1, 0, 0),
    )

    decision = select_simplest_near_best([best, simpler, outside])

    assert decision.status == "PASS"
    assert decision.selected.name == "simpler"
    assert decision.best_weighted_rms_mev == pytest.approx(0.80)
    assert decision.one_se_threshold_mev == pytest.approx(1.00)
    assert decision.plateau_names == ("best", "simpler")


def test_warn_best_available_does_not_hide_unmet_overlap_target() -> None:
    accurate = _candidate(
        "accurate",
        loss=1.0,
        loss_se=0.2,
        parameters=8,
        overlap=0.943,
    )
    simple = _candidate(
        "simple",
        loss=1.1,
        loss_se=0.1,
        parameters=4,
        overlap=0.940,
    )

    decision = select_simplest_near_best(
        [accurate, simple],
        overlap_target=0.95,
        overlap_safety_floor=0.90,
    )

    assert decision.status == "WARN_BEST_AVAILABLE"
    assert decision.selected.name == "simple"
    assert decision.unmet_targets == ("mean_subspace_overlap>=0.95",)


def test_candidates_below_overlap_safety_floor_fail_closed() -> None:
    candidates = [
        _candidate("bad-a", loss=1.0, loss_se=0.1, parameters=3, overlap=0.88),
        _candidate("bad-b", loss=0.8, loss_se=0.1, parameters=8, overlap=0.89),
    ]

    decision = select_simplest_near_best(
        candidates,
        overlap_target=0.95,
        overlap_safety_floor=0.90,
    )

    assert decision.status == "FAIL"
    assert decision.selected is None
    assert decision.unmet_targets == ("mean_subspace_overlap>=0.90",)


def test_invalid_candidates_are_rejected_before_plateau_selection() -> None:
    valid = _candidate("valid", loss=1.2, loss_se=0.1, parameters=5)
    uncertified = _candidate(
        "uncertified",
        loss=0.1,
        loss_se=0.1,
        parameters=1,
        certified=False,
    )
    guard_failure = _candidate(
        "guard-failure",
        loss=0.2,
        loss_se=0.1,
        parameters=2,
        guards_passed=False,
    )
    nonfinite = _candidate(
        "nonfinite",
        loss=math.nan,
        loss_se=0.1,
        parameters=2,
    )

    decision = select_simplest_near_best(
        [uncertified, guard_failure, nonfinite, valid]
    )

    assert decision.selected.name == "valid"
    assert decision.rejected_reasons == {
        "uncertified": "not_certified",
        "guard-failure": "guard_failure",
        "nonfinite": "non_finite_metric",
    }


def test_candidate_validation_rejects_negative_parameter_count() -> None:
    with pytest.raises(ValueError, match="independent_real_parameters"):
        _candidate("invalid", loss=1.0, loss_se=0.1, parameters=-1)


def test_ties_use_family_order_then_maximum_error_then_name() -> None:
    candidates = [
        _candidate("z", loss=1.0, loss_se=0.1, parameters=4, orders=(3, 1, 0), maximum=2.0),
        _candidate("b", loss=1.0, loss_se=0.1, parameters=4, orders=(2, 1, 0), maximum=2.0),
        _candidate("a", loss=1.0, loss_se=0.1, parameters=4, orders=(2, 1, 0), maximum=2.0),
    ]

    decision = select_simplest_near_best(candidates)

    assert decision.selected.name == "a"
