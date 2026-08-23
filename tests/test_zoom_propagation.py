# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Tests for the per-Object bookkeeping of one Chained Detection pass.

These assert on what a pass produces -- which Objects are still being propagated,
which Zoom Window each of them gets, and what the per-frame Zoom Anchor record
says -- never on how the propagation loop reached it. No torch, no GPU, no model:
the point of :mod:`sam3.zoom_propagation` is that all of this is decidable from
plain values.
"""

import pytest

from sam3.zoom_anchor import (
    ObjectZoomState,
    ZoomAnchor,
    ZoomAnchorConfig,
    ZoomAnchorSource,
    ZoomWindow,
)
from sam3.zoom_propagation import (
    FrameRecord,
    MaskObservation,
    PassLedger,
    partition_re_grounding,
)

FRAME_W = 1000
FRAME_H = 1000
CONFIG = ZoomAnchorConfig(re_grounding_interval=5)


def _seed(bbox=(400, 400, 700, 700), area=90000):
    return ObjectZoomState(previous_bbox=bbox, previous_mask_area=area)


def _ledger(seed_states=None, seed_frame=10, limit=3, config=CONFIG):
    if seed_states is None:
        seed_states = {1: _seed()}
    return PassLedger(
        seed_states=seed_states,
        seed_frame=seed_frame,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        config=config,
        absence_streak_limit=limit,
    )


def _observe(bbox, area=None):
    x1, y1, x2, y2 = bbox
    return MaskObservation(bbox=bbox, area=area or (x2 - x1) * (y2 - y1))


def _step(ledger, frame_index, outcomes, re_grounding=None):
    """Resolve, report the given outcomes, return (anchors, record)."""
    anchors = ledger.resolve(re_grounding)
    record = ledger.record_frame(frame_index, anchors, outcomes, re_grounding)
    return anchors, record


# ---------------------------------------------------------------------------
# Zoom Anchor record
# ---------------------------------------------------------------------------


def test_record_names_the_anchor_that_produced_each_window():
    ledger = _ledger()
    _, record = _step(ledger, 11, {1: _observe((410, 410, 710, 710))})
    assert record.anchor_sources == {1: ZoomAnchorSource.MASK_DERIVED.value}


def test_a_fresh_re_grounding_is_recorded_as_re_grounded():
    ledger = _ledger()
    _, record = _step(
        ledger,
        11,
        {1: _observe((410, 410, 710, 710))},
        re_grounding={1: (200, 200, 500, 500)},
    )
    assert record.anchor_sources == {1: ZoomAnchorSource.RE_GROUNDED.value}


def test_full_frame_degradation_is_visible_in_the_record():
    ledger = _ledger(seed_states={1: ObjectZoomState()})
    anchors, record = _step(ledger, 11, {1: None})
    assert record.anchor_sources == {1: ZoomAnchorSource.FULL_FRAME.value}
    assert anchors[1].window.as_tuple() == (0, 0, FRAME_W, FRAME_H)


def test_an_absent_object_falls_back_to_its_last_re_grounding_and_says_so():
    ledger = _ledger()
    # Frame 11 re-grounds and finds nothing; the box is remembered as stale.
    _step(ledger, 11, {1: None}, re_grounding={1: (200, 200, 500, 500)})
    # Frame 12 has no mask to follow, so the stale box is all that is left.
    _, record = _step(ledger, 12, {1: None})
    assert record.anchor_sources == {1: ZoomAnchorSource.STALE.value}


def test_record_summary_counts_every_source():
    record = FrameRecord(
        frame_index=7,
        anchor_sources={1: "re-grounded", 2: "mask-derived", 3: "mask-derived"},
        absent_obj_ids=(4,),
        stopped_obj_ids=(4,),
        newly_stopped_obj_ids=(),
    )
    summary = record.summary()
    assert "re-grounded=1" in summary
    assert "mask-derived=2" in summary
    assert "stale=0" in summary
    assert "full-frame=0" in summary
    assert "absent=1" in summary
    assert "stopped=1" in summary


def test_pass_summary_accumulates_over_the_pass():
    ledger = _ledger()
    _step(ledger, 11, {1: _observe((400, 400, 700, 700))}, {1: (400, 400, 700, 700)})
    _step(ledger, 12, {1: _observe((400, 400, 700, 700))})
    summary = ledger.pass_summary()
    assert "re-grounded=1" in summary
    assert "mask-derived=1" in summary


# ---------------------------------------------------------------------------
# Windows follow their own Object between ticks
# ---------------------------------------------------------------------------


def test_window_follows_this_objects_own_previous_mask_between_ticks():
    ledger = _ledger()
    first, _ = _step(ledger, 11, {1: _observe((500, 500, 800, 800))})
    second, _ = _step(ledger, 12, {1: _observe((500, 500, 800, 800))})

    # Frame 11 was framed on the seed mask, frame 12 on what frame 11 produced.
    assert first[1].window.center == (550.0, 550.0)
    assert second[1].window.center == (650.0, 650.0)


def test_one_objects_re_grounding_does_not_move_another_objects_window():
    ledger = _ledger(seed_states={1: _seed(), 2: _seed((100, 100, 200, 200), 10000)})
    alone, _ = _step(ledger, 11, {1: _observe((400, 400, 700, 700)), 2: None})

    other = _ledger(seed_states={1: _seed(), 2: _seed((100, 100, 200, 200), 10000)})
    together, _ = _step(
        other,
        11,
        {1: _observe((400, 400, 700, 700)), 2: None},
        re_grounding={2: (900, 900, 950, 950)},
    )

    assert together[1].window == alone[1].window
    assert together[2].window != alone[2].window


def test_every_window_is_square_unless_nothing_is_known():
    ledger = _ledger(seed_states={1: _seed(), 2: ObjectZoomState()})
    anchors, _ = _step(ledger, 11, {1: _observe((400, 400, 700, 700)), 2: None})
    assert anchors[1].window.is_square
    assert anchors[2].window.as_tuple() == (0, 0, FRAME_W, FRAME_H)


# ---------------------------------------------------------------------------
# Absence and the streak stop
# ---------------------------------------------------------------------------


def test_absent_object_is_listed_absent_without_being_stopped():
    ledger = _ledger(limit=3)
    _, record = _step(ledger, 11, {1: None})
    assert record.absent_obj_ids == (1,)
    assert record.stopped_obj_ids == ()
    assert record.newly_stopped_obj_ids == ()
    assert ledger.active_object_ids == (1,)


def test_object_stops_after_the_configured_absence_streak():
    ledger = _ledger(limit=3)
    for frame_index in (11, 12):
        _, record = _step(ledger, frame_index, {1: None})
        assert record.newly_stopped_obj_ids == ()

    _, record = _step(ledger, 13, {1: None})
    assert record.newly_stopped_obj_ids == (1,)
    assert ledger.active_object_ids == ()
    assert ledger.stopped_object_ids == (1,)


def test_a_stopped_object_is_no_longer_processed_but_still_reported_absent():
    ledger = _ledger(limit=1)
    _step(ledger, 11, {1: None})
    assert ledger.active_object_ids == ()

    anchors, record = _step(ledger, 12, {})
    assert anchors == {}
    assert record.anchor_sources == {}
    assert record.absent_obj_ids == (1,)
    assert record.stopped_obj_ids == (1,)


def test_a_present_frame_resets_the_streak():
    ledger = _ledger(limit=3)
    _step(ledger, 11, {1: None})
    _step(ledger, 12, {1: None})
    _step(ledger, 13, {1: _observe((400, 400, 700, 700))})
    for frame_index in (14, 15):
        _, record = _step(ledger, frame_index, {1: None})
        assert record.newly_stopped_obj_ids == ()
    assert ledger.active_object_ids == (1,)


def test_one_object_stopping_leaves_the_others_propagating():
    ledger = _ledger(seed_states={1: _seed(), 2: _seed()}, limit=2)
    for frame_index in (11, 12):
        _step(ledger, frame_index, {1: None, 2: _observe((400, 400, 700, 700))})
    assert ledger.stopped_object_ids == (1,)
    assert ledger.active_object_ids == (2,)


def test_absence_streak_limit_must_be_positive():
    with pytest.raises(ValueError):
        _ledger(limit=0)


# ---------------------------------------------------------------------------
# Tick cadence, delegated to the resolver
# ---------------------------------------------------------------------------


def test_ticks_land_every_n_frames_from_the_first_frame_of_the_pass():
    ledger = _ledger(seed_frame=10, config=ZoomAnchorConfig(re_grounding_interval=5))
    ticks = [f for f in range(11, 26) if ledger.should_re_ground(f)]
    assert ticks == [11, 16, 21]


def test_ticks_run_the_same_way_backwards_from_the_seed_frame():
    ledger = _ledger(seed_frame=10, config=ZoomAnchorConfig(re_grounding_interval=5))
    ticks = [f for f in range(9, -1, -1) if ledger.should_re_ground(f)]
    assert ticks == [9, 4]


def test_nothing_is_re_grounded_once_every_object_has_stopped():
    ledger = _ledger(seed_frame=10, limit=1)
    assert ledger.should_re_ground(11)
    _step(ledger, 11, {1: None})
    assert not ledger.should_re_ground(16)


# ---------------------------------------------------------------------------
# Pass restart
# ---------------------------------------------------------------------------


def test_a_second_pass_starts_again_from_the_seed_masks():
    seed_states = {1: _seed()}
    forward = _ledger(seed_states=seed_states, limit=1)
    _step(forward, 11, {1: None})
    assert forward.active_object_ids == ()

    backward = _ledger(seed_states=seed_states, limit=1)
    anchors, record = _step(backward, 9, {1: _observe((400, 400, 700, 700))})
    assert backward.active_object_ids == (1,)
    assert record.anchor_sources == {1: ZoomAnchorSource.MASK_DERIVED.value}
    assert anchors[1].window.center == (550.0, 550.0)


# ---------------------------------------------------------------------------
# Overlay boxes handed to Re-grounding
# ---------------------------------------------------------------------------


def test_previous_bboxes_are_reported_in_the_renderers_inclusive_convention():
    ledger = _ledger(seed_states={1: _seed((400, 400, 700, 700))})
    assert ledger.previous_bboxes() == {1: (400, 400, 699, 699)}


def test_previous_bboxes_omit_objects_with_no_mask():
    ledger = _ledger(seed_states={1: _seed(), 2: ObjectZoomState()})
    assert set(ledger.previous_bboxes()) == {1}


# ---------------------------------------------------------------------------
# Re-grounding responses
# ---------------------------------------------------------------------------


def test_usable_re_grounding_boxes_are_accepted():
    accepted, rejected = partition_re_grounding({1: (10, 20, 30, 40)})
    assert accepted == {1: (10, 20, 30, 40)}
    assert rejected == {}


def test_an_unmentioned_object_is_not_a_rejection():
    accepted, rejected = partition_re_grounding({1: None})
    assert accepted == {}
    assert rejected == {}


def test_a_degenerate_box_is_rejected_by_name_rather_than_repaired():
    accepted, rejected = partition_re_grounding({1: (30, 20, 30, 40), 2: (0, 0, 5, 5)})
    assert accepted == {2: (0, 0, 5, 5)}
    assert rejected == {1: (30, 20, 30, 40)}


def test_a_malformed_box_is_rejected():
    accepted, rejected = partition_re_grounding({1: (1, 2, 3), 2: "everywhere"})
    assert accepted == {}
    assert set(rejected) == {1, 2}


def test_no_response_partitions_to_nothing():
    assert partition_re_grounding(None) == ({}, {})


def test_a_rejected_box_leaves_the_object_on_its_own_mask():
    ledger = _ledger()
    accepted, _ = partition_re_grounding({1: (500, 500, 500, 600)})
    _, record = _step(ledger, 11, {1: _observe((400, 400, 700, 700))}, accepted)
    assert record.anchor_sources == {1: ZoomAnchorSource.MASK_DERIVED.value}


# ---------------------------------------------------------------------------
# The ledger refuses to guess
# ---------------------------------------------------------------------------


def test_an_unreported_active_object_is_an_error_not_an_assumed_absence():
    ledger = _ledger(seed_states={1: _seed(), 2: _seed()})
    anchors = ledger.resolve()
    with pytest.raises(ValueError):
        ledger.record_frame(11, anchors, {1: None})


def test_reporting_an_object_that_is_not_being_propagated_is_an_error():
    ledger = _ledger()
    anchors = ledger.resolve()
    with pytest.raises(ValueError):
        ledger.record_frame(11, anchors, {1: None, 99: None})


def test_a_missing_zoom_window_is_an_error():
    ledger = _ledger()
    with pytest.raises(ValueError):
        ledger.record_frame(11, {}, {1: None})


def test_a_pass_needs_at_least_one_object():
    with pytest.raises(ValueError):
        _ledger(seed_states={})


def test_an_empty_observation_is_rejected_rather_than_recorded_as_a_mask():
    with pytest.raises(ValueError):
        MaskObservation(bbox=(10, 10, 10, 20), area=5)
    with pytest.raises(ValueError):
        MaskObservation(bbox=(10, 10, 20, 20), area=0)


def test_anchors_may_carry_objects_the_ledger_no_longer_processes():
    ledger = _ledger(limit=1)
    _step(ledger, 11, {1: None})
    stale_anchor = {1: ZoomAnchor(ZoomWindow(0, 0, 10, 10), ZoomAnchorSource.STALE)}
    record = ledger.record_frame(12, stale_anchor, {})
    assert record.anchor_sources == {}
