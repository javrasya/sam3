# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""The Re-grounding request/response contract, as plain values.

DISCERN FORK LOCAL ADDITION -- this whole module is local to javrasya/sam3 and has
no upstream counterpart. Every public callable below carries the same marker, per
Discern ADR 0002 ("Recording How Our SAM3 Fork Diverges From Upstream").

Re-grounding (Discern ``CONTEXT.md``) is asking a vision model to locate one Object
afresh from its Object Hint, discarding what the mask chain believed. It is now
**one request per Object**: each request carries only that Object's overlay, box
and Object Hint, so a session with five Objects is not competing for one small
response budget. ``sam3.agent.llm_crop_advisor.LLMCropAdvisor.re_ground_objects``
issues them; this module holds everything about that exchange that is decidable
without a network, a GPU or an image library, so it can be tested with plain
values -- and so the *contract* the propagation loop and the Zoom Pass consume
does not drag PIL and openai in behind it.

The two faces of :class:`ReGroundingResult`
-------------------------------------------
It **is** a ``Mapping[int, Optional[BBox]]``, which is exactly the
``re_grounding`` argument of :func:`sam3.zoom_anchor.resolve_zoom_anchors`, so a
caller hands the result straight to the resolver. Reading it that way answers
"where is this Object, or nothing".

It also carries :attr:`per_object`, which answers the question the mapping face
deliberately cannot: *why* nothing. A provider that 401s and a model that looked
and saw nothing both yield no box, and collapsing the two is what made a run whose
every request failed indistinguishable from a successful one.

Boxes are pixel ``(x1, y1, x2, y2)``, x2/y2 **exclusive**, matching
``sam3.zoom_anchor``. They are the Object's raw located box: padding is the
resolver's business, not this module's, because the resolver owns all geometry.
"""

import collections.abc
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, Mapping, Optional, Tuple

__all__ = [
    "BBox",
    "DEFAULT_RE_GROUNDING_TIMEOUT_S",
    "DEFAULT_RE_GROUNDING_CONCURRENCY",
    "DEFAULT_RE_GROUNDING_MAX_TOKENS",
    "RE_GROUNDING_SYSTEM_PROMPT",
    "ReGroundingError",
    "ReGroundingResponseError",
    "ReGroundingOutcome",
    "ObjectReGrounding",
    "ReGroundingResult",
    "build_re_grounding_prompt",
    "parse_re_grounding_response",
]

# (x1, y1, x2, y2) in pixels, x2/y2 exclusive -- sam3.zoom_anchor's convention.
BBox = Tuple[int, int, int, int]

# Wall-clock budget for one frame's Re-grounding, in seconds.
#
# Discern's API gives up on a propagation stream that has not produced its first
# frame within 60s (``api/backend_client.py``, ``first_frame_timeout: float =
# 60.0``, never overridden by a caller). The first frame after the seed frame is
# always a Re-grounding tick, so that request sits inside those 60 seconds. 20s
# leaves 40s for model setup, the seed-frame work and the frame's own inference,
# and -- because requests for different Objects are issued concurrently and the
# budget bounds the whole batch -- it stays 20s whether there is one Object or
# five. Without it the openai SDK's own default applies: 600s per attempt with up
# to two retries, i.e. a hung call can block the stream for half an hour, long
# after the caller has stopped listening.
DEFAULT_RE_GROUNDING_TIMEOUT_S = 20.0

# How many of a frame's per-Object requests may be in flight at once. Bounded
# because providers rate-limit (Gemini's free tier allows 15 requests/minute, the
# same reason Discern's AGENT_CONCURRENCY defaults to 2) and because an unbounded
# fan-out on a many-Object session would trade one crowded request for a burst of
# throttled ones.
DEFAULT_RE_GROUNDING_CONCURRENCY = 4

# Per-request response budget. One Object's answer is a few dozen tokens, but this
# reaches the provider as ``max_completion_tokens``, which on Gemini also bounds
# *thinking* tokens -- a thinking model can spend the whole budget before emitting
# any content, which arrives here as an empty message. The old shared-across-all-
# Objects budget of 512 is a prime suspect for "the feature never ran".
DEFAULT_RE_GROUNDING_MAX_TOKENS = 1024

RE_GROUNDING_SYSTEM_PROMPT = """\
You locate one specific object in a video frame.

You receive two images:
1. The previous frame, in which that object -- and only that object -- is \
highlighted with a coloured mask overlay and a box.
2. The current frame, unannotated.

Find the same object in image 2 and report where it is.

Respond with ONLY valid JSON, in one of these two forms:
{"found": true, "box": [x1, y1, x2, y2]}
{"found": false}

Use the second form when the object is not visible in image 2 -- occluded, out of \
frame, or genuinely absent. Do not guess a location for an object you cannot see.

x1, y1, x2, y2 are normalized to [0, 1]: x1, y1 is the top-left corner and x2, y2 \
the bottom-right. Make the box tight around the object itself and add no margin; \
margin is added later and adding it twice defeats the purpose."""


# DISCERN FORK LOCAL ADDITION
class ReGroundingError(RuntimeError):
    """Re-grounding did not work, as opposed to working and finding nothing."""


# DISCERN FORK LOCAL ADDITION
class ReGroundingResponseError(ReGroundingError):
    """The provider answered, but not with something this contract can read."""


# DISCERN FORK LOCAL ADDITION
class ReGroundingOutcome(str, Enum):
    """What became of one Object's Re-grounding request on one frame.

    ``FAILED`` is the operator's cue to check the provider settings; ``NOT_FOUND``
    is the model doing its job and reporting an absent Object. Telling them apart
    is the whole reason this enum exists.
    """

    LOCATED = "located"
    NOT_FOUND = "not-found"
    FAILED = "failed"
    NOT_ASKED = "not-asked"


def _validate_bbox(bbox: BBox) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate Re-grounding box: {bbox}")


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ObjectReGrounding:
    """One Object's Re-grounding outcome on one frame.

    Args:
        object_id: The Object this concerns.
        outcome: See :class:`ReGroundingOutcome`.
        bbox: The Object's raw located box, set only when ``LOCATED``. Unpadded --
            :mod:`sam3.zoom_anchor` applies ``crop_padding``.
        reason: Why there is no box, in words an operator can act on. Set on
            ``FAILED`` and ``NOT_ASKED``.
    """

    object_id: int
    outcome: ReGroundingOutcome
    bbox: Optional[BBox] = None
    reason: Optional[str] = None

    def __post_init__(self):
        if self.outcome is ReGroundingOutcome.LOCATED:
            if self.bbox is None:
                raise ValueError("a located Object must carry a box")
            _validate_bbox(self.bbox)
            if self.reason is not None:
                raise ValueError("a located Object has nothing to explain")
        else:
            if self.bbox is not None:
                raise ValueError(f"{self.outcome.value} must not carry a box")
        if self.outcome in (ReGroundingOutcome.FAILED, ReGroundingOutcome.NOT_ASKED):
            if not self.reason:
                raise ValueError(f"{self.outcome.value} must say why")

    # DISCERN FORK LOCAL ADDITION
    @classmethod
    def located(cls, object_id: int, bbox: BBox) -> "ObjectReGrounding":
        return cls(object_id, ReGroundingOutcome.LOCATED, bbox=bbox)

    # DISCERN FORK LOCAL ADDITION
    @classmethod
    def not_found(cls, object_id: int) -> "ObjectReGrounding":
        return cls(object_id, ReGroundingOutcome.NOT_FOUND)

    # DISCERN FORK LOCAL ADDITION
    @classmethod
    def failed(cls, object_id: int, reason: str) -> "ObjectReGrounding":
        return cls(object_id, ReGroundingOutcome.FAILED, reason=reason)

    # DISCERN FORK LOCAL ADDITION
    @classmethod
    def not_asked(cls, object_id: int, reason: str) -> "ObjectReGrounding":
        """No request was issued, so neither the provider nor the model failed."""
        return cls(object_id, ReGroundingOutcome.NOT_ASKED, reason=reason)


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ReGroundingResult(collections.abc.Mapping):
    """One frame's Re-grounding, per Object.

    Reads as ``Mapping[int, Optional[BBox]]`` -- the resolver's ``re_grounding``
    argument -- while :attr:`per_object` keeps the reason each missing box is
    missing. See this module's docstring for why both faces exist.
    """

    per_object: Mapping[int, ObjectReGrounding]

    def __post_init__(self):
        for object_id, entry in self.per_object.items():
            if entry.object_id != object_id:
                raise ValueError(
                    f"entry for Object {object_id} carries object_id {entry.object_id}"
                )

    def __getitem__(self, object_id: int) -> Optional[BBox]:
        return self.per_object[object_id].bbox

    def __iter__(self) -> Iterator[int]:
        return iter(self.per_object)

    def __len__(self) -> int:
        return len(self.per_object)

    # DISCERN FORK LOCAL ADDITION
    @property
    def boxes(self) -> Dict[int, Optional[BBox]]:
        """The mapping face as a plain dict, for callers that want a copy."""
        return {object_id: entry.bbox for object_id, entry in self.per_object.items()}

    # DISCERN FORK LOCAL ADDITION
    @property
    def located(self) -> Dict[int, BBox]:
        """Only the Objects the vision model actually placed."""
        return {
            object_id: entry.bbox
            for object_id, entry in self.per_object.items()
            if entry.outcome is ReGroundingOutcome.LOCATED
        }

    # DISCERN FORK LOCAL ADDITION
    @property
    def failures(self) -> Dict[int, str]:
        """Objects whose request failed, and why. Empty on a healthy frame."""
        return {
            object_id: entry.reason
            for object_id, entry in self.per_object.items()
            if entry.outcome is ReGroundingOutcome.FAILED
        }

    # DISCERN FORK LOCAL ADDITION
    @property
    def absent(self) -> Tuple[int, ...]:
        """Objects the vision model looked for and reported as not visible."""
        return tuple(
            object_id
            for object_id, entry in self.per_object.items()
            if entry.outcome is ReGroundingOutcome.NOT_FOUND
        )

    # DISCERN FORK LOCAL ADDITION
    @property
    def every_request_failed(self) -> bool:
        """True when requests were issued and every one of them failed.

        The signature of a misconfigured provider: nothing was learned, and the
        reason is the provider rather than the scene.
        """
        issued = [
            entry
            for entry in self.per_object.values()
            if entry.outcome is not ReGroundingOutcome.NOT_ASKED
        ]
        return bool(issued) and all(
            entry.outcome is ReGroundingOutcome.FAILED for entry in issued
        )

    # DISCERN FORK LOCAL ADDITION
    def raise_for_failures(self) -> None:
        """Turn any failure into an exception, for callers that want to stop."""
        failures = self.failures
        if failures:
            detail = "; ".join(
                f"Object {object_id}: {reason}"
                for object_id, reason in sorted(failures.items())
            )
            raise ReGroundingError(f"Re-grounding failed -- {detail}")


# DISCERN FORK LOCAL ADDITION
def build_re_grounding_prompt(
    object_id: int,
    object_hint: Optional[str],
    has_previous_mask: bool,
) -> str:
    """The user text of one Object's request.

    ``object_hint`` is that Object's Object Hint. Discern resolves the documented
    fallback -- the Object's name when no Hint was written -- before the request
    leaves its API, so ``None`` here means the Object has neither, and all the
    model has to go on is the overlay.
    """
    lines = [f"Track exactly one object: obj_{object_id}."]
    if object_hint:
        lines.append(f'The person tracking it calls it: "{object_hint}".')
    if has_previous_mask:
        lines.append(
            "Image 1 is the previous frame, with obj_{0} highlighted by a coloured "
            "mask overlay and a box. No other object is highlighted.".format(object_id)
        )
    else:
        # An Object with no mask on the previous frame is exactly the case
        # Re-grounding exists for: the chain lost it, and only the Object Hint can
        # bring it back. Say so rather than sending an unannotated image that
        # looks like a rendering bug.
        lines.append(
            "Image 1 is the previous frame. obj_{0} has no overlay there because "
            "it was not segmented on that frame.".format(object_id)
        )
    lines.append("Image 2 is the current frame.")
    lines.append(
        "Report where obj_{0} is in image 2, as JSON and nothing else.".format(
            object_id
        )
    )
    return "\n".join(lines)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start = body.find("{")
    end = body.rfind("}") + 1
    if start == -1 or end == 0:
        raise ReGroundingResponseError(
            f"no JSON object in the provider's answer: {text[:200]!r}"
        )
    return body[start:end]


# DISCERN FORK LOCAL ADDITION
def parse_re_grounding_response(
    response_text: str,
    frame_width: int,
    frame_height: int,
) -> Optional[BBox]:
    """Read one Object's answer into a pixel box, or ``None`` for "not visible".

    Returns the Object's raw located box in pixels, x2/y2 exclusive, clamped to the
    frame -- or ``None`` when the model reported the Object as absent, which is an
    answer rather than a failure.

    Anything else raises :class:`ReGroundingResponseError`. An unreadable answer is
    not evidence that the Object is gone, and reporting it as such is the silent
    degradation this whole change is about.
    """
    if frame_width < 1 or frame_height < 1:
        raise ValueError(f"frame must be non-empty, got {frame_width}x{frame_height}")
    if not response_text or not response_text.strip():
        raise ReGroundingResponseError("the provider returned an empty answer")

    try:
        data = json.loads(_extract_json(response_text))
    except json.JSONDecodeError as exc:
        raise ReGroundingResponseError(
            f"unparseable JSON in the provider's answer ({exc}): "
            f"{response_text[:200]!r}"
        ) from None
    if not isinstance(data, dict):
        raise ReGroundingResponseError(
            f"expected a JSON object, got {type(data).__name__}"
        )

    box = data.get("box")
    if box is None:
        if data.get("found") is False:
            return None
        raise ReGroundingResponseError(
            f"answer carries neither a box nor found=false: {response_text[:200]!r}"
        )
    if data.get("found") is False:
        raise ReGroundingResponseError(
            f"answer says found=false and still carries a box: "
            f"{response_text[:200]!r}"
        )

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ReGroundingResponseError(f"box is not four numbers: {box!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        raise ReGroundingResponseError(f"box is not four numbers: {box!r}") from None

    pixels = (
        max(0, min(frame_width, int(round(x1 * frame_width)))),
        max(0, min(frame_height, int(round(y1 * frame_height)))),
        max(0, min(frame_width, int(round(x2 * frame_width)))),
        max(0, min(frame_height, int(round(y2 * frame_height)))),
    )
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        # Inverted, empty, or entirely outside the frame. The model produced
        # something, but nothing that names a place, so it is a failed answer and
        # not an absent Object.
        raise ReGroundingResponseError(
            f"box {box!r} is empty or outside the frame at "
            f"{frame_width}x{frame_height}"
        )
    return pixels


# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
def own_frame(frame):
    """A private, fully-decoded copy of a frame, safe to hand to a worker thread.

    Discern opens session frames lazily (``Image.open`` with no ``load``) and
    hands the same objects to every consumer. PIL's ``load`` is a mutation: it
    consumes the file pointer and clears the tile list. So a Re-grounding worker
    decoding one of those images while the propagation loop is also decoding it
    crashes on ``assert self.fp is not None`` -- not because either side did
    anything wrong, but because neither owned the image.

    Decoding here, on the thread that reads the frame, and passing only the copy
    onwards means no two threads ever touch one image. Shapes are recognised by
    behaviour rather than by type so this module stays free of PIL, numpy and
    torch, and can be tested with none of them installed.
    """
    if frame is None:
        return None
    convert = getattr(frame, "convert", None)  # PIL: decodes, then copies
    if callable(convert):
        return convert("RGB")
    clone = getattr(frame, "clone", None)  # torch
    if callable(clone):
        return clone()
    copy = getattr(frame, "copy", None)  # numpy
    if callable(copy):
        return copy()
    return frame
