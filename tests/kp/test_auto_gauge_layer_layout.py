from __future__ import annotations

import numpy as np

from kp.blocks import resolve_project_gauge_anchors
from kp.blocks.blocks import _gamma_same_q_row_segments, _gamma_source_group_translated_references


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


def test_gamma_source_group_translation_preserves_spin_and_local_row_offset() -> None:
    segments = _gamma_same_q_row_segments(
        block_dim=30,
        num_layer_list=[1, 2],
        num_orb_per_layer_list=[[5], [5, 5]],
        spin="all",
    )
    resolved = [
        [],
        [[[8, 1.0]], [[23, 1.0]]],
        [[[2, 1.0]], [[17, 1.0]]],
    ]
    owners = [(1, 0), (1, 1), (2, 0), (2, 1)]

    references, details = _gamma_source_group_translated_references(
        resolved,
        owners,
        num_layer_list=[1, 2],
        segments=segments,
    )

    assert references == [
        [(8, 1.0 + 0.0j)],
        [(23, 1.0 + 0.0j)],
        [(13, 1.0 + 0.0j)],
        [(28, 1.0 + 0.0j)],
    ]
    assert details == [
        {
            "source_group": 1,
            "template_layer": 1,
            "target_layer": 2,
            "band_position": 0,
            "source_terms": [[8, 1.0]],
            "translated_terms": [[13, 1.0]],
        },
        {
            "source_group": 1,
            "template_layer": 1,
            "target_layer": 2,
            "band_position": 1,
            "source_terms": [[23, 1.0]],
            "translated_terms": [[28, 1.0]],
        },
    ]


def test_source_group_translation_is_optional_without_a_shared_group():
    assert _gamma_source_group_translated_references(
        [[[[0, 1.0]]], [[[1, 1.0]]]], [(0, 0), (1, 0)],
        num_layer_list=[1, 1], segments=[(0, 1), (1, 2), (2, 3), (3, 4)]) is None


def test_source_group_translation_is_optional_for_unequal_anchor_counts():
    assert _gamma_source_group_translated_references(
        [[], [[[1, 1.0]], [[4, 1.0]]], [[[2, 1.0]]]], [(1, 0), (1, 1), (2, 0)],
        num_layer_list=[1, 2], segments=[(i, i+1) for i in range(6)]) is None


def test_source_group_translation_validates_owners_before_optional_skip():
    import pytest

    with pytest.raises(ValueError, match="owners"):
        _gamma_source_group_translated_references(
            [[[[0, 1.0]]], [[[1, 1.0]]]], [(2, 0)], num_layer_list=[1, 1],
            segments=[(0, 1), (1, 2), (2, 3), (3, 4)])


def test_source_group_translation_validates_segments_before_optional_skip():
    import pytest

    with pytest.raises(ValueError, match="segments"):
        _gamma_source_group_translated_references(
            [[[[0, 1.0]]], [[[1, 1.0]]]], [(0, 0), (1, 0)],
            num_layer_list=[1, 1], segments=[(0, 1), (1, 2)])


def test_gamma_candidate_enumeration_does_not_swallow_translation_metadata_errors(monkeypatch):
    import pytest
    import kp.blocks.blocks as blocks

    def bad_metadata(*args, **kwargs):
        raise ValueError("invalid shared translation metadata")

    monkeypatch.setattr(blocks, "_gamma_source_group_translated_references", bad_metadata)
    with pytest.raises(ValueError, match="invalid shared translation metadata"):
        blocks.resolve_project_gauge_anchor_candidates(
            np.diag(np.arange(4)).astype(complex), q_count=1, orb_per_layer0=1,
            num_layer_list=[1, 1], spin="all", Qlayer_list=[[np.zeros((1, 2))]]*2,
            num_orb_per_layer_list=[[1], [1]], nlow_state_list=[[0, 1], [2, 3]],
            norb_fix_list=None, gauge_config="auto", mode="gamma")
