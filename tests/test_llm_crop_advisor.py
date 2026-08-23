# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Tests for Re-grounding as it is actually issued: one request per Object.

The provider is replaced by a recording stand-in, so these assert on what a
consumer can observe -- how many requests were made, what each one carried, which
outcome came back, and whether the temporary images survived the call. Nothing
here needs a GPU, a network or a model.

They need numpy, PIL and openai (the last only because the transport module
imports it), so they skip rather than fail when the bare-pytest command is used.
See NOTES/how-to-test.md for the command that runs them.
"""

import os
import threading
import time

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("openai")

import numpy as np  # noqa: E402

from sam3.agent.llm_crop_advisor import LLMCropAdvisor  # noqa: E402
from sam3.re_grounding import ReGroundingOutcome  # noqa: E402
from sam3.zoom_anchor import (  # noqa: E402
    ObjectZoomState,
    ZoomAnchorSource,
    resolve_zoom_anchors,
)

WIDTH, HEIGHT = 320, 240


def frame():
    return np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)


def mask_at(x1, y1, x2, y2):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def image_paths(messages):
    return [
        part["image"]
        for message in messages
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image"
    ]


def prompt_text(messages):
    return "\n".join(
        part["text"]
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part.get("type") == "text"
    )


class Provider:
    """A stand-in transport that records every request it is handed."""

    def __init__(self, answer='{"found": true, "box": [0.25, 0.25, 0.5, 0.5]}'):
        self.answer = answer
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, *, messages, server_url, model, api_key, max_tokens, timeout):
        call = {
            "messages": messages,
            "prompt": prompt_text(messages),
            "images": image_paths(messages),
            "images_present": [os.path.exists(p) for p in image_paths(messages)],
            "timeout": timeout,
            "max_tokens": max_tokens,
            "api_key": api_key,
        }
        with self.lock:
            self.calls.append(call)
        return self.answer(call) if callable(self.answer) else self.answer

    def call_for(self, object_id):
        (call,) = [c for c in self.calls if f"obj_{object_id}." in c["prompt"]]
        return call


def advisor(provider, **kwargs):
    return LLMCropAdvisor(
        server_url="http://provider.invalid/v1",
        model="a-vision-model",
        api_key="k",
        send_request=provider,
        **kwargs,
    )


def re_ground(provider, object_ids=(1, 2, 3), hints=None, masks=None, **kwargs):
    if masks is None:
        masks = {
            object_id: mask_at(10 * object_id, 10, 10 * object_id + 20, 40)
            for object_id in object_ids
        }
    return advisor(provider, **kwargs).re_ground_objects(
        prev_frame=frame(),
        curr_frame=frame(),
        object_ids=list(object_ids),
        prev_masks=masks,
        object_hints=hints,
    )


# --------------------------------------------------------------------------
# One request per Object
# --------------------------------------------------------------------------


def test_each_object_gets_its_own_request():
    provider = Provider()

    result = re_ground(provider, object_ids=(1, 2, 3))

    assert len(provider.calls) == 3
    assert sorted(result.located) == [1, 2, 3]


def test_a_request_names_only_its_own_object():
    provider = Provider()

    re_ground(provider, object_ids=(1, 2, 3))

    for object_id in (1, 2, 3):
        prompt = provider.call_for(object_id)["prompt"]
        others = {f"obj_{other}" for other in (1, 2, 3) if other != object_id}
        assert not [name for name in others if name in prompt]


def test_each_object_hint_reaches_its_own_request_and_no_other():
    provider = Provider()

    re_ground(
        provider,
        object_ids=(1, 2),
        hints={1: "the referee", 2: "the red car"},
    )

    assert "the referee" in provider.call_for(1)["prompt"]
    assert "the red car" not in provider.call_for(1)["prompt"]
    assert "the red car" in provider.call_for(2)["prompt"]


def test_an_object_without_a_hint_is_still_re_grounded():
    provider = Provider()

    result = re_ground(provider, object_ids=(1, 2), hints={2: "the red car"})

    assert len(provider.calls) == 2
    assert result.per_object[1].outcome is ReGroundingOutcome.LOCATED


def test_an_object_with_neither_a_hint_nor_a_mask_is_not_asked_about():
    provider = Provider()

    result = re_ground(provider, object_ids=(1, 2), masks={2: mask_at(10, 10, 30, 30)})

    assert len(provider.calls) == 1
    assert result.per_object[1].outcome is ReGroundingOutcome.NOT_ASKED
    assert result.failures == {}


def test_an_object_with_a_hint_but_no_mask_is_re_grounded_from_the_hint():
    provider = Provider()

    result = re_ground(provider, object_ids=(1,), masks={}, hints={1: "the referee"})

    assert len(provider.calls) == 1
    assert "the referee" in provider.calls[0]["prompt"]
    assert result.per_object[1].outcome is ReGroundingOutcome.LOCATED


def test_a_request_carries_two_images_that_exist_while_it_is_in_flight():
    provider = Provider()

    re_ground(provider, object_ids=(1,))

    assert len(provider.calls[0]["images"]) == 2
    assert provider.calls[0]["images_present"] == [True, True]


# --------------------------------------------------------------------------
# The answer, and what it becomes
# --------------------------------------------------------------------------


def test_a_located_object_yields_its_raw_box_ready_for_the_resolver():
    provider = Provider('{"found": true, "box": [0.25, 0.25, 0.5, 0.5]}')

    result = re_ground(provider, object_ids=(1,))

    # Exactly the normalized box in pixels -- no padding applied here.
    assert result[1] == (80, 60, 160, 120)
    anchors = resolve_zoom_anchors({1: ObjectZoomState()}, result, WIDTH, HEIGHT)
    assert anchors[1].source is ZoomAnchorSource.RE_GROUNDED


def test_an_absent_object_is_reported_as_absent_not_as_a_failure():
    provider = Provider('{"found": false}')

    result = re_ground(provider, object_ids=(1,))

    assert result.absent == (1,)
    assert result.failures == {}


def test_a_provider_failure_is_reported_as_a_failure_with_its_own_words():
    def explode(call):
        raise RuntimeError("401 invalid api key")

    provider = Provider(explode)

    result = re_ground(provider, object_ids=(1,))

    assert result.per_object[1].outcome is ReGroundingOutcome.FAILED
    assert "401 invalid api key" in result.failures[1]
    assert result[1] is None
    assert result.every_request_failed


def test_one_objects_failure_leaves_the_others_re_grounded():
    def one_bad_apple(call):
        if "obj_2." in call["prompt"]:
            raise RuntimeError("provider hiccup")
        return '{"found": true, "box": [0.25, 0.25, 0.5, 0.5]}'

    provider = Provider(one_bad_apple)

    result = re_ground(provider, object_ids=(1, 2, 3))

    assert sorted(result.located) == [1, 3]
    assert list(result.failures) == [2]
    assert not result.every_request_failed


def test_an_unreadable_answer_is_a_failure_not_an_absent_object():
    provider = Provider("I think it is near the middle somewhere.")

    result = re_ground(provider, object_ids=(1,))

    assert result.per_object[1].outcome is ReGroundingOutcome.FAILED
    assert result.absent == ()


# --------------------------------------------------------------------------
# Time and concurrency
# --------------------------------------------------------------------------


def test_every_request_carries_an_explicit_timeout_below_the_first_frame_guard():
    provider = Provider()

    re_ground(provider, object_ids=(1, 2))

    assert {call["timeout"] for call in provider.calls} == {
        advisor(provider).request_timeout
    }
    assert all(0 < call["timeout"] < 60.0 for call in provider.calls)


def test_a_hung_request_fails_instead_of_stalling_the_frame(monkeypatch):
    import sam3.agent.llm_crop_advisor as module

    monkeypatch.setattr(module, "_BATCH_GRACE_S", 0.05)

    def hang(call):
        time.sleep(1.0)
        return '{"found": false}'

    started = time.monotonic()
    result = re_ground(Provider(hang), object_ids=(1,), request_timeout=0.05)
    elapsed = time.monotonic() - started

    assert result.per_object[1].outcome is ReGroundingOutcome.FAILED
    assert elapsed < 1.0, "the frame waited for the hung request"


def test_requests_for_different_objects_are_in_flight_together():
    barrier = threading.Barrier(3, timeout=5)

    def wait_for_the_others(call):
        barrier.wait()
        return '{"found": true, "box": [0.25, 0.25, 0.5, 0.5]}'

    result = re_ground(
        Provider(wait_for_the_others), object_ids=(1, 2, 3), max_concurrent_requests=3
    )

    # Only reachable if all three requests were open at the same time.
    assert sorted(result.located) == [1, 2, 3]


def test_concurrency_stays_within_the_configured_bound():
    barrier = threading.Barrier(2, timeout=0.3)

    def wait_for_a_partner(call):
        barrier.wait()
        return '{"found": true, "box": [0.25, 0.25, 0.5, 0.5]}'

    result = re_ground(
        Provider(wait_for_a_partner), object_ids=(1, 2), max_concurrent_requests=1
    )

    # With one worker the second request cannot join the first, so neither pairs.
    assert result.located == {}


# --------------------------------------------------------------------------
# Temporary images
# --------------------------------------------------------------------------


def test_the_temporary_images_are_deleted_after_a_successful_request():
    provider = Provider()

    re_ground(provider, object_ids=(1, 2, 3))

    written = [path for call in provider.calls for path in call["images"]]
    assert len(written) == 6
    assert [path for path in written if os.path.exists(path)] == []


def test_the_temporary_images_are_deleted_after_a_failed_request():
    def explode(call):
        raise RuntimeError("provider is down")

    provider = Provider(explode)

    re_ground(provider, object_ids=(1, 2))

    written = [path for call in provider.calls for path in call["images"]]
    assert len(written) == 4
    assert [path for path in written if os.path.exists(path)] == []


def test_the_temporary_images_are_deleted_after_an_unreadable_answer():
    provider = Provider("no json here")

    re_ground(provider, object_ids=(1,))

    assert [path for path in provider.calls[0]["images"] if os.path.exists(path)] == []
