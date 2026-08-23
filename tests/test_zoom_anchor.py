# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Tests for the pure Zoom Anchor resolver.

Plain values only: no GPU, no network, no model, no tensors. Every assertion is
on something a consumer can observe -- the Zoom Window produced, the order frames
are processed in, the Zoom Anchor source reported -- never on how it was reached.
"""

import pytest

from sam3.zoom_anchor import (
    BACKWARD,
    BOTH,
    FORWARD,
    DEFAULT_ZOOM_ANCHOR_CONFIG,
    ObjectZoomState,
    ZoomAnchorConfig,
    ZoomAnchorSource,
    ZoomWindow,
    advance_object_state,
    plan_processing_order,
    plan_processing_passes,
    resolve_zoom_anchor,
    resolve_zoom_anchors,
    should_re_ground,
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


def resolve(state, fresh=None, config=DEFAULT_ZOOM_ANCHOR_CONFIG):
    return resolve_zoom_anchor(state, fresh, WIDTH, HEIGHT, config)


# --------------------------------------------------------------------------
# Processing order -- both sides of the seed frame
# --------------------------------------------------------------------------


def test_order_from_a_mid_video_seed_covers_frames_on_both_sides():
    order = plan_processing_order(seed_frame=10, frame_count=20, direction=BOTH)

    assert [f for f in order if f > 10], "no frames after the seed frame"
    assert [f for f in order if f < 10], "no frames before the seed frame"
    assert sorted(order) == [f for f in range(20) if f != 10]


def test_order_visits_every_frame_once_and_never_the_seed_frame():
    order = plan_processing_order(seed_frame=3, frame_count=9, direction=BOTH)

    assert len(order) == len(set(order))
    assert 3 not in order


def test_order_walks_outward_from_the_seed_frame_in_each_direction():
    order = plan_processing_order(seed_frame=10, frame_count=20, direction=BOTH)

    assert order == (11, 12, 13, 14, 15, 16, 17, 18, 19, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)


def test_single_direction_orders_stay_single_direction():
    assert plan_processing_order(4, 8, FORWARD) == (5, 6, 7)
    assert plan_processing_order(4, 8, BACKWARD) == (3, 2, 1, 0)


def test_both_is_two_passes_so_callers_can_restart_the_chain_at_the_seam():
    passes = plan_processing_passes(seed_frame=2, frame_count=5, direction=BOTH)

    assert [(p.direction, p.frames) for p in passes] == [
        (FORWARD, (3, 4)),
        (BACKWARD, (1, 0)),
    ]


def test_a_seed_at_the_first_frame_has_no_backward_pass():
    passes = plan_processing_passes(seed_frame=0, frame_count=4, direction=BOTH)

    assert [p.direction for p in passes] == [FORWARD]
    assert plan_processing_order(0, 4, BOTH) == (1, 2, 3)


def test_an_unknown_direction_is_refused_rather_than_treated_as_forward():
    with pytest.raises(ValueError):
        plan_processing_order(5, 10, "sideways")


def test_a_seed_outside_the_video_is_refused():
    with pytest.raises(ValueError):
        plan_processing_order(10, 10, BOTH)


# --------------------------------------------------------------------------
# Tick cadence -- interval N means every N frames
# --------------------------------------------------------------------------


def test_re_grounding_happens_every_n_frames_not_every_n_plus_one():
    config = ZoomAnchorConfig(re_grounding_interval=5)
    forward = plan_processing_order(seed_frame=10, frame_count=40, direction=FORWARD)

    ticks = [f for f in forward if should_re_ground(f, 10, config)]

    assert ticks == [11, 16, 21, 26, 31, 36]
    assert {b - a for a, b in zip(ticks, ticks[1:])} == {5}


def test_the_cadence_holds_on_the_backward_pass_too():
    config = ZoomAnchorConfig(re_grounding_interval=5)
    backward = plan_processing_order(seed_frame=20, frame_count=40, direction=BACKWARD)

    ticks = [f for f in backward if should_re_ground(f, 20, config)]

    assert ticks == [19, 14, 9, 4]
    assert {a - b for a, b in zip(ticks, ticks[1:])} == {5}


@pytest.mark.parametrize("interval", [1, 2, 3, 7])
def test_every_interval_produces_gaps_of_exactly_that_many_frames(interval):
    config = ZoomAnchorConfig(re_grounding_interval=interval)
    frames = plan_processing_order(seed_frame=0, frame_count=60, direction=FORWARD)

    ticks = [f for f in frames if should_re_ground(f, 0, config)]

    assert ticks[0] == 1, "the first frame of a pass is always re-grounded"
    assert {b - a for a, b in zip(ticks, ticks[1:])} == {interval}


def test_the_seed_frame_is_never_a_tick_because_it_is_never_processed():
    with pytest.raises(ValueError):
        should_re_ground(10, 10, DEFAULT_ZOOM_ANCHOR_CONFIG)


# --------------------------------------------------------------------------
# Anchor precedence
# --------------------------------------------------------------------------


def test_a_fresh_re_grounding_result_wins_over_everything():
    state = ObjectZoomState(
        previous_bbox=square_at(200, 200, 100),
        stale_re_grounding=square_at(1700, 900, 100),
    )

    anchor = resolve(state, fresh=square_at(900, 500, 100))

    assert anchor.source is ZoomAnchorSource.RE_GROUNDED
    assert anchor.window.center == (900.0, 500.0)


def test_a_stale_re_grounding_result_loses_to_a_fresh_mask_bounding_box():
    state = ObjectZoomState(
        previous_bbox=square_at(300, 300, 100),
        stale_re_grounding=square_at(1600, 800, 100),
    )

    anchor = resolve(state, fresh=None)

    assert anchor.source is ZoomAnchorSource.MASK_DERIVED
    assert anchor.window.center == (300.0, 300.0)


def test_a_stale_result_is_used_only_when_this_object_has_no_mask():
    state = ObjectZoomState(
        previous_bbox=None, stale_re_grounding=square_at(1600, 800, 100)
    )

    anchor = resolve(state, fresh=None)

    assert anchor.source is ZoomAnchorSource.STALE
    assert anchor.window.center == (1600.0, 800.0)


def test_with_nothing_known_the_window_is_the_whole_frame():
    anchor = resolve(ObjectZoomState())

    assert anchor.source is ZoomAnchorSource.FULL_FRAME
    assert anchor.window.as_tuple() == (0, 0, WIDTH, HEIGHT)


def test_an_object_with_no_previous_mask_still_gets_a_usable_window():
    """A never-seeded Object must not blow up, and must not get an empty window."""
    anchors = resolve_zoom_anchors(
        {1: ObjectZoomState(), 2: ObjectZoomState()}, None, WIDTH, HEIGHT
    )

    for anchor in anchors.values():
        assert anchor.source is ZoomAnchorSource.FULL_FRAME
        assert anchor.window.width > 0 and anchor.window.height > 0


def test_the_reported_source_is_one_of_the_four_named_anchors():
    cases = {
        ZoomAnchorSource.RE_GROUNDED: (
            ObjectZoomState(previous_bbox=square_at(100, 100, 80)),
            square_at(500, 500, 80),
        ),
        ZoomAnchorSource.MASK_DERIVED: (
            ObjectZoomState(previous_bbox=square_at(100, 100, 80)),
            None,
        ),
        ZoomAnchorSource.STALE: (
            ObjectZoomState(stale_re_grounding=square_at(100, 100, 80)),
            None,
        ),
        ZoomAnchorSource.FULL_FRAME: (ObjectZoomState(), None),
    }

    for expected, (state, fresh) in cases.items():
        assert resolve(state, fresh).source is expected


# --------------------------------------------------------------------------
# Per-Object independence
# --------------------------------------------------------------------------


def test_one_objects_window_is_unaffected_by_where_the_others_are():
    person = ObjectZoomState(previous_bbox=square_at(960, 540, 120))
    car_near = ObjectZoomState(previous_bbox=square_at(1000, 560, 120))
    car_far = ObjectZoomState(previous_bbox=square_at(60, 60, 120))

    alone = resolve_zoom_anchors({1: person}, None, WIDTH, HEIGHT)
    with_near_car = resolve_zoom_anchors({1: person, 2: car_near}, None, WIDTH, HEIGHT)
    with_far_car = resolve_zoom_anchors({1: person, 2: car_far}, None, WIDTH, HEIGHT)

    assert with_near_car[1] == alone[1]
    assert with_far_car[1] == alone[1]


def test_a_huge_object_does_not_loosen_the_framing_of_a_small_one():
    small = ObjectZoomState(previous_bbox=square_at(300, 300, 100))
    huge = ObjectZoomState(previous_bbox=(0, 0, 1900, 1000))

    anchors = resolve_zoom_anchors({1: small, 2: huge}, None, WIDTH, HEIGHT)

    assert anchors[1] == resolve(small)
    assert anchors[2].window.width > anchors[1].window.width


def test_a_re_grounding_result_for_one_object_does_not_move_another():
    states = {
        1: ObjectZoomState(previous_bbox=square_at(300, 300, 100)),
        2: ObjectZoomState(previous_bbox=square_at(1500, 700, 100)),
    }

    anchors = resolve_zoom_anchors(states, {2: square_at(900, 200, 100)}, WIDTH, HEIGHT)

    assert anchors[1].source is ZoomAnchorSource.MASK_DERIVED
    assert anchors[1].window.center == (300.0, 300.0)
    assert anchors[2].source is ZoomAnchorSource.RE_GROUNDED


# --------------------------------------------------------------------------
# Squareness and clamping to frame bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bbox",
    [
        square_at(960, 540, 300),
        square_at(10, 10, 40),
        square_at(1900, 1060, 40),
        (0, 0, 1900, 1000),
        (100, 100, 900, 200),
        (100, 100, 200, 900),
    ],
)
def test_every_bbox_derived_window_is_square_and_inside_the_frame(bbox):
    window = resolve(ObjectZoomState(previous_bbox=bbox)).window

    assert window.is_square
    assert 0 <= window.x1 and window.x2 <= WIDTH
    assert 0 <= window.y1 and window.y2 <= HEIGHT


def test_at_a_frame_edge_the_window_slides_inward_rather_than_losing_its_shape():
    """A square that will not fit centred on the Object is translated, not shrunk."""
    corner = resolve(ObjectZoomState(previous_bbox=square_at(10, 10, 40))).window
    middle = resolve(ObjectZoomState(previous_bbox=square_at(960, 540, 40))).window

    assert corner.is_square
    assert corner.width == middle.width
    assert corner.as_tuple() == (0, 0, FLOOR, FLOOR)


def test_a_window_wider_than_the_frame_is_capped_to_the_largest_square_that_fits():
    window = resolve(ObjectZoomState(previous_bbox=(0, 0, 1900, 1000))).window

    assert window.is_square
    assert window.width == min(WIDTH, HEIGHT)
    assert window.y1 == 0 and window.y2 == HEIGHT


def test_the_full_frame_fallback_is_the_whole_frame_and_says_so():
    """The one non-square window: nothing is known, so nothing may be cropped off."""
    window = resolve(ObjectZoomState()).window

    assert not window.is_square
    assert window.as_tuple() == (0, 0, WIDTH, HEIGHT)


# --------------------------------------------------------------------------
# Zoom Lock
# --------------------------------------------------------------------------


def test_zoom_lock_keeps_apparent_size_constant_as_an_object_approaches():
    """Window size tracks mask size, so the Object fills the same share of SAM3's input."""
    ratios = []
    for side in (200, 300, 400, 500):
        window = resolve(
            ObjectZoomState(previous_bbox=square_at(960, 540, side))
        ).window
        ratios.append(window.width / side)

    assert len(set(ratios)) == 1
    assert ratios[0] == pytest.approx(2.0)


def test_a_fixed_size_window_is_not_what_zoom_lock_produces():
    near = resolve(ObjectZoomState(previous_bbox=square_at(960, 540, 500))).window
    far = resolve(ObjectZoomState(previous_bbox=square_at(960, 540, 250))).window

    assert near.width == 2 * far.width


def test_window_size_follows_the_longest_side_of_a_lopsided_mask():
    tall = resolve(ObjectZoomState(previous_bbox=(900, 300, 1000, 700))).window
    wide = resolve(ObjectZoomState(previous_bbox=(400, 500, 800, 600))).window

    assert tall.width == wide.width == 400 * 2


# --------------------------------------------------------------------------
# The three guards, each shown preventing collapse on its own
# --------------------------------------------------------------------------

# Floor and freeze switched off, so only the rate limit can hold the size up.
RATE_LIMIT_ONLY = ZoomAnchorConfig(min_size_fraction=0.01, min_size_px=1)
# Rate limit and freeze switched off, so only the floor can.
FLOOR_ONLY = ZoomAnchorConfig(max_size_change=10.0)
# Rate limit and floor switched off, so only the area-collapse freeze can.
FREEZE_ONLY = ZoomAnchorConfig(
    max_size_change=10.0, min_size_fraction=0.01, min_size_px=1
)


def test_the_rate_limit_alone_stops_one_bad_mask_from_collapsing_the_window():
    collapsed = ObjectZoomState(
        previous_bbox=square_at(960, 540, 10), previous_window_size=400
    )

    window = resolve(collapsed, config=RATE_LIMIT_ONLY).window

    assert window.width == 340  # 400 paced down by at most 15%, not 20
    assert (
        resolve(
            ObjectZoomState(previous_bbox=square_at(960, 540, 10)),
            config=RATE_LIMIT_ONLY,
        ).window.width
        == 20
    )


def test_the_rate_limit_alone_also_paces_growth():
    window = resolve(
        ObjectZoomState(
            previous_bbox=square_at(960, 540, 1000), previous_window_size=400
        ),
        config=RATE_LIMIT_ONLY,
    ).window

    assert window.width == 460


def test_the_floor_alone_stops_a_fragmenting_object_from_starving_its_window():
    fragmented = ObjectZoomState(
        previous_bbox=square_at(960, 540, 10), previous_window_size=400
    )

    window = resolve(fragmented, config=FLOOR_ONLY).window

    assert window.width == FLOOR


def test_the_floor_beats_the_rate_limit_when_they_disagree():
    """A window may never be paced downward past the floor."""
    state = ObjectZoomState(
        previous_bbox=square_at(960, 540, 10), previous_window_size=190
    )

    window = resolve(state, config=DEFAULT_ZOOM_ANCHOR_CONFIG).window

    assert window.width == FLOOR  # not 190 * 0.85 == 162


def test_the_area_collapse_freeze_alone_holds_the_size_through_a_bad_frame():
    collapsed = ObjectZoomState(
        previous_bbox=square_at(960, 540, 10),
        previous_mask_area=1_000,
        earlier_mask_area=10_000,
        previous_window_size=400,
    )
    steady = ObjectZoomState(
        previous_bbox=square_at(960, 540, 10),
        previous_mask_area=10_000,
        earlier_mask_area=10_000,
        previous_window_size=400,
    )

    assert resolve(collapsed, config=FREEZE_ONLY).window.width == 400
    assert resolve(steady, config=FREEZE_ONLY).window.width == 20


def test_a_shrinking_object_is_still_followed_when_its_area_did_not_collapse():
    """The freeze must not turn Zoom Lock off for an Object that genuinely recedes."""
    receding = ObjectZoomState(
        previous_bbox=square_at(960, 540, 350),
        previous_mask_area=90_000,
        earlier_mask_area=100_000,
        previous_window_size=800,
    )

    assert resolve(receding, config=FREEZE_ONLY).window.width == 700


def test_the_window_survives_a_long_run_of_empty_masks_without_shrinking_away():
    """The guards compose: repeated bad frames still cannot starve the window."""
    state = ObjectZoomState(
        previous_bbox=square_at(960, 540, 300), previous_window_size=600
    )
    sizes = []
    for _ in range(30):
        window = resolve(state).window
        sizes.append(window.width)
        state = advance_object_state(state, square_at(960, 540, 4), 16, window)

    assert min(sizes) >= FLOOR
    assert all(abs(b - a) <= a * 0.15 + 1 for a, b in zip(sizes, sizes[1:]))


# --------------------------------------------------------------------------
# Carrying state between frames
# --------------------------------------------------------------------------


def test_a_successful_re_grounding_becomes_the_stale_anchor_for_later_frames():
    state = ObjectZoomState(previous_bbox=square_at(300, 300, 100))
    fresh = square_at(1200, 600, 100)

    window = resolve(state, fresh=fresh).window
    state = advance_object_state(state, None, 0, window, fresh_re_grounding=fresh)

    anchor = resolve(state)
    assert anchor.source is ZoomAnchorSource.STALE
    assert anchor.window.center == (1200.0, 600.0)


def test_an_absent_object_does_not_keep_pointing_at_a_mask_that_is_gone():
    state = ObjectZoomState(previous_bbox=square_at(300, 300, 100))

    state = advance_object_state(state, None, 0, resolve(state).window)

    assert resolve(state).source is ZoomAnchorSource.FULL_FRAME


def test_the_full_frame_fallback_does_not_reset_the_rate_limit_to_the_whole_frame():
    state = ObjectZoomState(previous_window_size=400)

    full_frame = resolve(state).window
    state = advance_object_state(state, square_at(960, 540, 10), 100, full_frame)

    assert resolve(state, config=RATE_LIMIT_ONLY).window.width == 340


def test_degenerate_boxes_are_refused_rather_than_quietly_repaired():
    with pytest.raises(ValueError):
        ObjectZoomState(previous_bbox=(100, 100, 100, 200))
    with pytest.raises(ValueError):
        resolve(ObjectZoomState(), fresh=(500, 500, 400, 600))
    with pytest.raises(ValueError):
        ZoomWindow(10, 10, 10, 20)
