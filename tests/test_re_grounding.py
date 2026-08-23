# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Tests for the Re-grounding request/response contract.

Plain values only: no network, no provider, no image library. Every assertion is
on something a consumer can observe -- the box a given answer yields, the outcome
reported for one Object, the text the model is sent -- never on how it was reached.
"""

import pytest

from sam3.re_grounding import (
    DEFAULT_RE_GROUNDING_CONCURRENCY,
    DEFAULT_RE_GROUNDING_TIMEOUT_S,
    ObjectReGrounding,
    ReGroundingError,
    ReGroundingOutcome,
    ReGroundingResponseError,
    ReGroundingResult,
    build_re_grounding_prompt,
    parse_re_grounding_response,
)
from sam3.zoom_anchor import ObjectZoomState, ZoomAnchorSource, resolve_zoom_anchors

WIDTH, HEIGHT = 1000, 500


def parse(text, width=WIDTH, height=HEIGHT):
    return parse_re_grounding_response(text, width, height)


# --------------------------------------------------------------------------
# Reading one Object's answer
# --------------------------------------------------------------------------


def test_a_located_object_becomes_a_pixel_box():
    assert parse('{"found": true, "box": [0.1, 0.2, 0.3, 0.6]}') == (100, 100, 300, 300)


def test_the_box_is_raw_and_unpadded():
    # Padding is the resolver's, and a box padded twice defeats the Zoom Lock.
    box = parse('{"found": true, "box": [0.4, 0.4, 0.6, 0.6]}')
    assert box == (400, 200, 600, 300)


def test_an_answer_wrapped_in_a_code_fence_is_read():
    assert parse('```json\n{"found": true, "box": [0, 0, 1, 1]}\n```') == (
        0,
        0,
        WIDTH,
        HEIGHT,
    )


def test_an_answer_with_prose_around_it_is_read():
    text = (
        'Sure! Here it is:\n{"found": true, "box": [0, 0, 0.5, 0.5]}\nHope that helps.'
    )
    assert parse(text) == (0, 0, 500, 250)


def test_a_box_reaching_outside_the_frame_is_clamped():
    assert parse('{"found": true, "box": [-0.5, -0.5, 2.0, 2.0]}') == (
        0,
        0,
        WIDTH,
        HEIGHT,
    )


def test_an_absent_object_is_an_answer_not_a_failure():
    assert parse('{"found": false}') is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I could not tell.",
        "{not json at all",
        '{"found": true}',
        '{"found": true, "box": [0.1, 0.2]}',
        '{"found": true, "box": "somewhere"}',
        '{"found": false, "box": [0.1, 0.2, 0.3, 0.4]}',
        '{"found": true, "box": [0.5, 0.5, 0.5, 0.5]}',
        '{"found": true, "box": [0.9, 0.1, 0.2, 0.9]}',
        '{"found": true, "box": [2.0, 2.0, 3.0, 3.0]}',
    ],
)
def test_an_unusable_answer_raises_rather_than_reporting_an_absent_object(text):
    # Reporting these as "not visible" is the silent degradation being removed:
    # an answer nobody can read is not evidence that the Object left the frame.
    with pytest.raises(ReGroundingResponseError):
        parse(text)


# --------------------------------------------------------------------------
# Telling a failed provider from an absent Object
# --------------------------------------------------------------------------


def result_of(*entries):
    return ReGroundingResult({entry.object_id: entry for entry in entries})


def test_a_failure_and_an_absence_are_distinguishable():
    result = result_of(
        ObjectReGrounding.failed(1, "AuthenticationError: 401"),
        ObjectReGrounding.not_found(2),
    )

    assert result.failures == {1: "AuthenticationError: 401"}
    assert result.absent == (2,)
    assert result[1] is None and result[2] is None


def test_the_reason_a_request_failed_reaches_the_caller():
    result = result_of(ObjectReGrounding.failed(3, "APITimeoutError: 20s"))

    assert "20s" in result.failures[3]
    with pytest.raises(ReGroundingError, match="Object 3"):
        result.raise_for_failures()


def test_a_healthy_frame_raises_nothing():
    result = result_of(
        ObjectReGrounding.located(1, (0, 0, 10, 10)), ObjectReGrounding.not_found(2)
    )

    result.raise_for_failures()
    assert result.failures == {}
    assert not result.every_request_failed


def test_every_request_failing_is_recognisable():
    assert result_of(
        ObjectReGrounding.failed(1, "401"), ObjectReGrounding.failed(2, "401")
    ).every_request_failed


def test_an_object_that_was_never_asked_about_does_not_look_like_a_failure():
    result = result_of(
        ObjectReGrounding.located(1, (0, 0, 10, 10)),
        ObjectReGrounding.not_asked(2, "no Object Hint and no mask"),
    )

    assert result.failures == {}
    assert not result.every_request_failed
    assert result.per_object[2].outcome is ReGroundingOutcome.NOT_ASKED


def test_only_located_objects_carry_a_box():
    with pytest.raises(ValueError):
        ObjectReGrounding(1, ReGroundingOutcome.NOT_FOUND, bbox=(0, 0, 10, 10))
    with pytest.raises(ValueError):
        ObjectReGrounding.located(1, (10, 10, 10, 20))
    with pytest.raises(ValueError):
        ObjectReGrounding(1, ReGroundingOutcome.FAILED)


# --------------------------------------------------------------------------
# The seam with the resolver
# --------------------------------------------------------------------------


def test_a_result_is_the_resolver_input_as_it_stands():
    result = result_of(
        ObjectReGrounding.located(1, (400, 200, 600, 300)),
        ObjectReGrounding.failed(2, "401"),
    )

    anchors = resolve_zoom_anchors(
        {1: ObjectZoomState(), 2: ObjectZoomState()}, result, WIDTH, HEIGHT
    )

    assert anchors[1].source is ZoomAnchorSource.RE_GROUNDED
    # Nothing known about Object 2, and the failure is not smuggled in as a box.
    assert anchors[2].source is ZoomAnchorSource.FULL_FRAME


def test_the_located_box_positions_the_window_where_the_model_looked():
    result = result_of(ObjectReGrounding.located(1, (400, 200, 600, 300)))

    window = resolve_zoom_anchors({1: ObjectZoomState()}, result, WIDTH, HEIGHT)[
        1
    ].window

    assert window.center == (500.0, 250.0)
    assert window.is_square


# --------------------------------------------------------------------------
# What one Object's request says
# --------------------------------------------------------------------------


def test_the_object_hint_reaches_the_model():
    prompt = build_re_grounding_prompt(3, "the referee", has_previous_mask=True)

    assert "the referee" in prompt
    assert "obj_3" in prompt


def test_an_object_without_a_hint_is_still_asked_about():
    prompt = build_re_grounding_prompt(3, None, has_previous_mask=True)

    assert "obj_3" in prompt


def test_a_request_says_when_the_object_has_no_overlay_to_look_at():
    with_mask = build_re_grounding_prompt(3, "the referee", has_previous_mask=True)
    without_mask = build_re_grounding_prompt(3, "the referee", has_previous_mask=False)

    assert with_mask != without_mask
    assert "no overlay" in without_mask


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------


def test_the_request_timeout_is_inside_discerns_first_frame_guard():
    # api/backend_client.py gives up on a stream with no first frame after 60s,
    # and the first frame after the seed frame is always a Re-grounding tick.
    assert 0 < DEFAULT_RE_GROUNDING_TIMEOUT_S < 60.0


def test_concurrency_is_bounded():
    assert 1 < DEFAULT_RE_GROUNDING_CONCURRENCY <= 8
