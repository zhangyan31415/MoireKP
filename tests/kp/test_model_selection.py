from __future__ import annotations

import math

import numpy as np
import pytest

from kp.model.model_selection import (
    CandidateScore,
    FamilyOrders,
    clone_family_vocabulary,
    blocked_kpath_folds,
    build_fixed_band_weights,
    parse_model_selection_config,
    select_simplest_near_best,
    subspace_overlap_metrics,
    weighted_band_error,
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


def test_blocked_kpath_folds_hold_out_every_point_once_in_contiguous_blocks() -> None:
    kpoints = np.column_stack((np.linspace(0.0, 1.0, 12), np.zeros(12)))

    folds = blocked_kpath_folds(kpoints, n_folds=3)

    assert len(folds) == 3
    held_out = np.concatenate([fold.validation_indices for fold in folds])
    np.testing.assert_array_equal(np.sort(held_out), np.arange(12))
    for fold in folds:
        assert np.all(np.diff(fold.validation_indices) == 1)
        assert not np.intersect1d(fold.train_indices, fold.validation_indices).size


def test_blocked_kpath_folds_keep_duplicate_junction_rows_together() -> None:
    kpoints = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
        ]
    )

    folds = blocked_kpath_folds(kpoints, n_folds=3)

    junction_folds = [
        fold.index
        for fold in folds
        if 2 in fold.validation_indices or 3 in fold.validation_indices
    ]
    assert junction_folds == [junction_folds[0]]
    owner = folds[junction_folds[0]]
    assert 2 in owner.validation_indices
    assert 3 in owner.validation_indices


def test_fixed_top_band_weights_complete_degenerate_boundary_and_normalize_mean() -> None:
    target = np.array(
        [
            [0.0, 1.0, 2.0, 2.00005],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )

    weights = build_fixed_band_weights(
        target,
        band_edge="top",
        n_bands=1,
        floor=0.10,
        degeneracy_tolerance_mev=0.10,
    )

    assert weights.shape == target.shape
    assert np.mean(weights) == pytest.approx(1.0)
    assert weights[0, -2] == pytest.approx(weights[0, -1])
    assert weights[0, -2] > weights[0, 0]
    assert weights[1, -1] > weights[1, -2]


def test_weighted_band_error_uses_fixed_weights_and_reports_unweighted_maximum() -> None:
    target = np.zeros((2, 3))
    model = np.array([[0.0, 0.001, 0.002], [0.0, 0.001, 0.004]])
    weights = np.array([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])

    metrics = weighted_band_error(model, target, weights)

    expected_rms_ev = np.sqrt(np.sum(weights * model**2) / np.sum(weights))
    assert metrics.weighted_rms_mev == pytest.approx(expected_rms_ev * 1000.0)
    assert metrics.maximum_abs_error_mev == pytest.approx(4.0)


def test_subspace_overlap_is_invariant_to_rotation_inside_degenerate_subspace() -> None:
    target = np.zeros((1, 3, 2), dtype=complex)
    target[0, 0, 0] = 1.0
    target[0, 1, 1] = 1.0
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    model = target @ rotation

    metrics = subspace_overlap_metrics(model, target)

    assert metrics.mean_overlap == pytest.approx(1.0)
    assert metrics.minimum_overlap == pytest.approx(1.0)
    assert metrics.maximum_leakage == pytest.approx(0.0, abs=1.0e-14)
    assert metrics.minimum_singular_value == pytest.approx(1.0)


def test_subspace_overlap_reports_leakage_outside_target_subspace() -> None:
    target = np.zeros((1, 3, 2), dtype=complex)
    target[0, 0, 0] = 1.0
    target[0, 1, 1] = 1.0
    model = np.zeros_like(target)
    model[0, 0, 0] = 1.0
    model[0, 2, 1] = 1.0

    metrics = subspace_overlap_metrics(model, target)

    assert metrics.mean_overlap == pytest.approx(0.5)
    assert metrics.maximum_leakage == pytest.approx(0.5)
    assert metrics.minimum_singular_value == pytest.approx(0.0)


def test_model_selection_config_defaults_to_family_specific_order_ladders() -> None:
    config = parse_model_selection_config(
        True,
        maximum_orders=FamilyOrders(kinetic=4, intra=2, inter=1),
    )

    assert config.enabled is True
    assert config.order_candidates == {
        "kinetic": (1, 2, 3, 4),
        "intra": (0, 1, 2),
        "inter": (0, 1),
    }
    assert config.overlap_target == pytest.approx(0.95)
    assert config.overlap_safety_floor == pytest.approx(0.90)


def test_model_selection_config_normalizes_explicit_candidates_and_limits() -> None:
    config = parse_model_selection_config(
        {
            "enabled": True,
            "folds": 4,
            "orders": {
                "Kinect": [4, 2, 2, 1],
                "intralayer": {"min": 0, "max": 2},
                "inter": [0, 1],
            },
            "quality": {"overlap_target": 0.96, "overlap_safety_floor": 0.91},
        },
        maximum_orders=FamilyOrders(kinetic=4, intra=3, inter=2),
    )

    assert config.n_folds == 4
    assert config.order_candidates == {
        "kinetic": (1, 2, 4),
        "intra": (0, 1, 2),
        "inter": (0, 1),
    }
    assert config.overlap_target == pytest.approx(0.96)
    assert config.overlap_safety_floor == pytest.approx(0.91)


def test_model_selection_config_rejects_order_above_declared_ceiling() -> None:
    with pytest.raises(ValueError, match="kinetic.*ceiling"):
        parse_model_selection_config(
            {"orders": {"kinetic": [2, 6]}},
            maximum_orders=FamilyOrders(kinetic=4, intra=2, inter=2),
        )


def test_clone_kinetic_vocabulary_drops_intra_and_inter_without_mutating_source() -> None:
    max_order = {
        "Kinect": 6,
        "intra": 4,
        "inter": 4,
        "moire_intra_nonzero": 4,
        "tunneling_zero": 4,
    }
    templates = [
        {"name": "kinetic", "source": "diagonal_kp", "max_order": 6},
        {"name": "onsite", "source": "onsite", "max_order": 0},
        {"name": "moire", "source": "moire_potential", "max_order": 4},
        {"name": "tunnel", "source": "tunneling", "max_order": 4},
    ]

    vocabulary = clone_family_vocabulary(
        max_order=max_order,
        term_templates=templates,
        orders=FamilyOrders(kinetic=2, intra=1, inter=1),
        active_families=("kinetic",),
    )

    assert [row["source"] for row in vocabulary.term_templates] == [
        "diagonal_kp",
        "onsite",
    ]
    assert vocabulary.term_templates[0]["max_order"] == 2
    assert vocabulary.term_templates[1]["max_order"] == 0
    assert vocabulary.max_order["Kinect"] == 2
    assert vocabulary.max_order["intra"] == 1
    assert vocabulary.max_order["inter"] == 1
    assert vocabulary.max_order["moire_intra_nonzero"] == 1
    assert vocabulary.max_order["tunneling_zero"] == 1
    assert templates[0]["max_order"] == 6
    assert max_order["Kinect"] == 6


def test_clone_joint_vocabulary_updates_template_local_orders_by_family() -> None:
    templates = [
        {"name": "kinetic", "tag": "Kinect", "source": "diagonal_kp", "max_order": 8},
        {"name": "moire", "tag": "intra", "source": "moire_potential", "max_order": 5},
        {"name": "tunnel", "tag": "inter", "source": "tunneling", "max_order": 5},
    ]

    vocabulary = clone_family_vocabulary(
        max_order={"Kinect": 8, "intra": 5, "inter": 5},
        term_templates=templates,
        orders=FamilyOrders(kinetic=3, intra=2, inter=1),
        active_families=("Kinect", "intralayer", "interlayer"),
    )

    assert [row["max_order"] for row in vocabulary.term_templates] == [3, 2, 1]
    assert vocabulary.active_families == ("kinetic", "intra", "inter")
