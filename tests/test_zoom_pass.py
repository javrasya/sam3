# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Tests for the pure Zoom Pass decisions.

Plain values only: no GPU, no network, no model, no tensors. Every assertion is
on something a consumer can observe -- the Zoom Window each Object gets, whether
an Object was planned at all, the outcome and score recorded for a mask, the
verdict the pass reports -- never on how it was reached.
"""

import pytest

from sam3.zoom_anchor import (
    ObjectZoomState,
    ZoomAnchorConfig,
    ZoomAnchorSource,
    ZoomWindow,
)
from sam3.zoom_pass import (
    DEFAULT_ZOOM_PASS_CONFIG,
    MaskOutcome,
    MaskOutcomeRecord,
    ObjectMaskGeometry,
    ZoomPassConfig,
    ZoomPassReport,
    ZoomPassStatus,
    is_zoom_beneficial,
    plan_zoom_pass_frame,
)

# A 1080p frame throughout, so the floor (1/6 of the shorter side) is 180px.
WIDTH, HEIGHT = 1920, 1080
FLOOR = 180


def square_at(center_x, center_y, side):
    """A mask bounding box of ``side`` pixels centred where asked."""
    half = side // 2
    return (
        center_x - half,
        center_y - half,
        center_x - half + side,
        center_y - half + side,
    )


def geometry_at(center_x, center_y, side):
    return ObjectMaskGeometry(
        bbox=square_at(center_x, center_y, side), area=side * side
    )


def plan(frame_masks, states=None, config=DEFAULT_ZOOM_PASS_CONFIG):
    return plan_zoom_pass_frame(frame_masks, states or {}, WIDTH, HEIGHT, config)


def improved(frame_index, object_id, score=0.8):
    return MaskOutcomeRecord(
        frame_index=frame_index,
        object_id=object_id,
        outcome=MaskOutcome.IMPROVED,
        anchor_source=ZoomAnchorSource.MASK_DERIVED,
        score=score,
    )


def unchanged(frame_index, object_id, outcome):
    return MaskOutcomeRecord(
        frame_index=frame_index,
        object_id=object_id,
        outcome=outcome,
        anchor_source=ZoomAnchorSource.MASK_DERIVED,
    )


# --------------------------------------------------------------------------
# Per-Object windows -- what replaced the shared union window
# --------------------------------------------------------------------------


def test_each_object_gets_a_window_centred_on_itself():
    plans = plan(
        {
            1: geometry_at(200, 540, 100),
            2: geometry_at(1700, 540, 100),
        }
    )

    assert plans[1].window.center == pytest.approx((200, 540), abs=1)
    assert plans[2].window.center == pytest.approx((1700, 540), abs=1)


def test_two_objects_at_opposite_edges_get_windows_that_contain_them():
    plans = plan(
        {
            1: geometry_at(200, 540, 100),
            2: geometry_at(1700, 540, 100),
        }
    )

    for object_id, center_x in ((1, 200), (2, 1700)):
        window = plans[object_id].window
        assert window.x1 <= center_x - 50 and window.x2 >= center_x + 50


def test_a_distant_object_does_not_change_another_objects_window():
    alone = plan({1: geometry_at(200, 540, 100)})
    crowded = plan(
        {
            1: geometry_at(200, 540, 100),
            2: geometry_at(1700, 60, 400),
            3: geometry_at(960, 1000, 300),
        }
    )

    assert crowded[1].window == alone[1].window


def test_window_size_follows_each_objects_own_size_not_the_largest_one():
    plans = plan(
        {
            1: geometry_at(400, 540, 100),
            2: geometry_at(1400, 540, 600),
        }
    )

    assert plans[1].window.width < plans[2].window.width


def test_every_object_with_a_mask_is_planned():
    frame_masks = {
        1: geometry_at(60, 60, 80),
        2: geometry_at(1860, 1020, 80),
        3: geometry_at(960, 540, 80),
    }

    plans = plan(frame_masks)

    assert set(plans) == set(frame_masks)


def test_every_planned_window_is_square():
    plans = plan(
        {
            1: geometry_at(200, 540, 100),
            2: geometry_at(1700, 200, 400),
        }
    )

    assert all(p.window.is_square for p in plans.values())


def test_an_object_too_big_to_zoom_is_planned_on_the_whole_frame_and_abandoned():
    """The resolver refuses to truncate it; the pass then has nothing to gain."""
    plans = plan({1: geometry_at(960, 540, 900)})

    assert plans[1].window.as_tuple() == (0, 0, WIDTH, HEIGHT)
    assert plans[1].anchor_source is ZoomAnchorSource.FULL_FRAME
    assert plans[1].zoom_is_beneficial is False


def test_every_planned_window_lies_inside_the_frame():
    plans = plan(
        {
            1: geometry_at(20, 20, 200),
            2: geometry_at(1900, 1060, 200),
        }
    )

    for p in plans.values():
        assert 0 <= p.window.x1 and p.window.x2 <= WIDTH
        assert 0 <= p.window.y1 and p.window.y2 <= HEIGHT


def test_an_objects_window_is_zoom_locked_to_its_mask():
    near = plan({1: geometry_at(960, 540, 400)})[1].window
    far = plan({1: geometry_at(960, 540, 200)})[1].window

    assert near.width == pytest.approx(2 * far.width, rel=0.02)


def test_anchor_source_is_recorded_for_every_object():
    plans = plan({1: geometry_at(200, 540, 100), 2: geometry_at(1700, 540, 100)})

    assert [p.anchor_source for p in plans.values()] == [
        ZoomAnchorSource.MASK_DERIVED,
        ZoomAnchorSource.MASK_DERIVED,
    ]


# --------------------------------------------------------------------------
# Carried state -- the Zoom Lock guards still pace across frames
# --------------------------------------------------------------------------


def test_the_rate_limit_paces_growth_between_processed_frames():
    first = plan({1: geometry_at(960, 540, 200)})
    grown = plan({1: geometry_at(960, 540, 900)}, {1: first[1].next_state})

    assert grown[1].window.width < 2 * first[1].window.width


def test_a_collapsed_mask_does_not_shrink_the_window():
    frame_one = plan({1: geometry_at(960, 540, 400)})
    frame_two = plan({1: geometry_at(960, 540, 390)}, {1: frame_one[1].next_state})
    collapsed = plan({1: geometry_at(960, 540, 40)}, {1: frame_two[1].next_state})

    assert collapsed[1].window.width >= frame_two[1].window.width * 0.99


def test_the_window_never_falls_below_the_floor():
    plans = plan({1: geometry_at(960, 540, 4)})

    assert plans[1].window.width >= FLOOR


def test_an_object_absent_from_a_frame_is_simply_not_planned():
    plans = plan({1: geometry_at(200, 540, 100)}, {2: ObjectZoomState()})

    assert set(plans) == {1}


def test_next_state_remembers_this_frames_geometry():
    plans = plan({1: geometry_at(960, 540, 100)})

    state = plans[1].next_state
    assert state.previous_bbox == square_at(960, 540, 100)
    assert state.previous_mask_area == 100 * 100
    assert state.previous_window_size == plans[1].window.width


def test_states_of_other_objects_never_leak_into_a_plan():
    busy = ObjectZoomState(
        previous_bbox=square_at(1700, 200, 900),
        previous_mask_area=900 * 900,
        previous_window_size=1000,
    )

    isolated = plan({1: geometry_at(200, 540, 100)})
    with_neighbour = plan(
        {1: geometry_at(200, 540, 100), 2: geometry_at(1700, 200, 900)},
        {2: busy},
    )

    assert with_neighbour[1].window == isolated[1].window


# --------------------------------------------------------------------------
# Per-Object abandonment -- what replaced the global 90% exit
# --------------------------------------------------------------------------


def test_an_object_filling_the_frame_is_abandoned_for_that_frame():
    plans = plan({1: geometry_at(960, 540, 1080)})

    assert plans[1].zoom_is_beneficial is False


def test_a_small_object_is_still_worth_zooming_into():
    plans = plan({1: geometry_at(960, 540, 100)})

    assert plans[1].zoom_is_beneficial is True


def test_abandonment_is_per_object_not_shared():
    plans = plan(
        {
            1: geometry_at(960, 540, 1080),
            2: geometry_at(300, 300, 90),
        }
    )

    assert plans[1].zoom_is_beneficial is False
    assert plans[2].zoom_is_beneficial is True


def test_is_zoom_beneficial_follows_the_configured_fraction():
    window = ZoomWindow(0, 0, 1000, 1000)
    strict = ZoomPassConfig(anchor=ZoomAnchorConfig(max_window_frame_fraction=0.4))
    lax = ZoomPassConfig(anchor=ZoomAnchorConfig(max_window_frame_fraction=0.99))

    assert is_zoom_beneficial(window, WIDTH, HEIGHT, strict) is False
    assert is_zoom_beneficial(window, WIDTH, HEIGHT, lax) is True


def test_a_full_frame_window_is_never_beneficial():
    assert is_zoom_beneficial(ZoomWindow(0, 0, WIDTH, HEIGHT), WIDTH, HEIGHT) is False


def test_is_zoom_beneficial_rejects_an_empty_frame():
    with pytest.raises(ValueError):
        is_zoom_beneficial(ZoomWindow(0, 0, 10, 10), 0, 100)


# --------------------------------------------------------------------------
# Honest scoring -- an untouched mask cannot be presented as a refined one
# --------------------------------------------------------------------------


def test_an_improved_mask_carries_the_score_it_scored():
    refinement = improved(7, 1, score=0.62)

    assert refinement.score == pytest.approx(0.62)
    assert refinement.emits_mask is True


def test_an_undetected_mask_carries_no_score_at_all():
    refinement = unchanged(7, 1, MaskOutcome.NOT_DETECTED)

    assert refinement.score is None
    assert refinement.emits_mask is False


def test_an_abandoned_object_carries_no_score_at_all():
    refinement = unchanged(7, 1, MaskOutcome.NO_ZOOM_BENEFIT)

    assert refinement.score is None
    assert refinement.emits_mask is False


def test_an_untouched_mask_cannot_be_scored_maximally_confident():
    with pytest.raises(ValueError):
        MaskOutcomeRecord(
            frame_index=7,
            object_id=1,
            outcome=MaskOutcome.NOT_DETECTED,
            anchor_source=ZoomAnchorSource.MASK_DERIVED,
            score=1.0,
        )


def test_an_improved_mask_must_say_what_it_scored():
    with pytest.raises(ValueError):
        MaskOutcomeRecord(
            frame_index=7,
            object_id=1,
            outcome=MaskOutcome.IMPROVED,
            anchor_source=ZoomAnchorSource.MASK_DERIVED,
        )


def test_a_score_outside_the_scale_is_refused():
    with pytest.raises(ValueError):
        improved(7, 1, score=1.4)


# --------------------------------------------------------------------------
# The verdict -- "could not improve anything" is not success
# --------------------------------------------------------------------------


def test_a_pass_that_improved_one_mask_reports_improvement():
    report = ZoomPassReport()
    report.record(improved(0, 1))
    report.record(unchanged(0, 2, MaskOutcome.NOT_DETECTED))

    assert report.status is ZoomPassStatus.IMPROVED


def test_a_pass_that_detected_nothing_anywhere_reports_no_improvement():
    report = ZoomPassReport()
    for frame_index in range(5):
        report.record(unchanged(frame_index, 1, MaskOutcome.NOT_DETECTED))

    assert report.status is ZoomPassStatus.NO_IMPROVEMENT


def test_a_pass_that_abandoned_every_object_reports_no_improvement():
    report = ZoomPassReport()
    for frame_index in range(5):
        report.record(unchanged(frame_index, 1, MaskOutcome.NO_ZOOM_BENEFIT))
        report.record(unchanged(frame_index, 2, MaskOutcome.NO_ZOOM_BENEFIT))

    assert report.status is ZoomPassStatus.NO_IMPROVEMENT


def test_a_pass_that_did_nothing_at_all_is_not_a_success():
    assert ZoomPassReport().status is ZoomPassStatus.NO_IMPROVEMENT


def test_the_verdict_counts_what_happened():
    report = ZoomPassReport()
    report.record(improved(0, 1))
    report.record(improved(1, 1))
    report.record(unchanged(0, 2, MaskOutcome.NOT_DETECTED))
    report.record(unchanged(1, 2, MaskOutcome.NO_ZOOM_BENEFIT))

    assert report.improved_count == 2
    assert report.not_detected_count == 1
    assert report.no_zoom_benefit_count == 1
    assert report.total_count == 4
    assert report.frames_improved == (0, 1)


def test_the_verdict_reports_per_object_outcomes():
    report = ZoomPassReport()
    report.record(improved(0, 1))
    report.record(improved(1, 1))
    report.record(unchanged(0, 2, MaskOutcome.NO_ZOOM_BENEFIT))
    report.record(unchanged(1, 2, MaskOutcome.NO_ZOOM_BENEFIT))

    by_object = report.outcomes_by_object()
    assert by_object[1] == {MaskOutcome.IMPROVED: 2}
    assert by_object[2] == {MaskOutcome.NO_ZOOM_BENEFIT: 2}
    assert report.object_ids == (1, 2)


def test_the_verdict_reports_which_zoom_anchors_were_used():
    report = ZoomPassReport()
    report.record(improved(0, 1))
    report.record(
        MaskOutcomeRecord(
            frame_index=1,
            object_id=1,
            outcome=MaskOutcome.NOT_DETECTED,
            anchor_source=ZoomAnchorSource.FULL_FRAME,
        )
    )

    assert report.anchor_source_counts == {
        ZoomAnchorSource.MASK_DERIVED: 1,
        ZoomAnchorSource.FULL_FRAME: 1,
    }


def test_the_no_improvement_summary_says_re_running_will_not_help():
    report = ZoomPassReport()
    report.record(unchanged(0, 1, MaskOutcome.NOT_DETECTED))

    summary = report.summary()
    assert "improved none" in summary
    assert "Re-running" in summary


def test_the_summary_distinguishes_improvement_from_none():
    improving = ZoomPassReport()
    improving.record(improved(0, 1))
    idle = ZoomPassReport()
    idle.record(unchanged(0, 1, MaskOutcome.NOT_DETECTED))

    assert improving.summary() != idle.summary()


def test_the_summary_of_an_empty_pass_says_there_was_nothing_to_do():
    assert "no propagated mask" in ZoomPassReport().summary()


def test_the_report_refuses_anything_that_is_not_an_outcome_record():
    with pytest.raises(TypeError):
        ZoomPassReport().record({"object_id": 1})


# --------------------------------------------------------------------------
# Configuration -- out of range fails loudly
# --------------------------------------------------------------------------


def test_an_out_of_range_abandonment_fraction_is_refused():
    with pytest.raises(ValueError):
        ZoomAnchorConfig(max_window_frame_fraction=0)
    with pytest.raises(ValueError):
        ZoomAnchorConfig(max_window_frame_fraction=1.5)


def test_the_pass_hands_its_anchor_settings_to_the_resolver():
    tight = ZoomPassConfig(anchor=ZoomAnchorConfig(crop_padding=0.0))
    loose = ZoomPassConfig(anchor=ZoomAnchorConfig(crop_padding=1.0))

    tight_window = plan({1: geometry_at(960, 540, 400)}, config=tight)[1].window
    loose_window = plan({1: geometry_at(960, 540, 400)}, config=loose)[1].window

    assert tight_window.width < loose_window.width


def test_a_degenerate_mask_geometry_is_refused():
    with pytest.raises(ValueError):
        ObjectMaskGeometry(bbox=(10, 10, 10, 20), area=5)


def test_an_empty_mask_has_no_geometry():
    with pytest.raises(ValueError):
        ObjectMaskGeometry(bbox=(10, 10, 20, 20), area=0)
