# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Frames handed to a Re-grounding worker must not be the caller's own objects.

The bug these cover: Discern opens session frames lazily and shares one object
with every consumer, so the Re-grounding worker and the propagation loop were
decoding the same PIL image at once and PIL asserted its file pointer away.
The invariant is ownership, not timing -- a race test would pass or fail by luck,
where this fails deterministically the moment a frame crosses a thread boundary
without being copied first.
"""

import pytest

from sam3.re_grounding import own_frame


class FakeLazyImage:
    """Stands in for a PIL image opened but not yet decoded."""

    def __init__(self):
        self.decoded = False

    def convert(self, mode):
        self.decoded = True
        copy = FakeLazyImage()
        copy.decoded = True
        copy.mode = mode
        return copy


class FakeArray:
    def copy(self):
        return FakeArray()


class FakeTensor:
    def clone(self):
        return FakeTensor()


class TestOwnFrame:
    def test_a_pil_style_frame_is_never_handed_on_as_the_caller_s_object(self):
        original = FakeLazyImage()
        assert own_frame(original) is not original

    def test_a_pil_style_frame_is_decoded_before_it_travels(self):
        original = FakeLazyImage()
        owned = own_frame(original)
        # Decoding happened on this thread, so no worker has to do it.
        assert owned.decoded

    def test_an_array_frame_is_copied(self):
        original = FakeArray()
        assert own_frame(original) is not original

    def test_a_tensor_frame_is_cloned(self):
        original = FakeTensor()
        assert own_frame(original) is not original

    def test_a_missing_frame_stays_missing(self):
        # The caller raises on an unreadable frame; owning one must not turn
        # None into something that looks readable.
        assert own_frame(None) is None
