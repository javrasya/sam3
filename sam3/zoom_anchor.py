# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Pure Zoom Anchor resolution for per-Object cropped propagation.

DISCERN FORK LOCAL ADDITION -- this whole module is local to javrasya/sam3 and has
no upstream counterpart. Every public callable below carries the same marker, per
Discern ADR 0002 ("Recording How Our SAM3 Fork Diverges From Upstream").

This module owns *every* decision about Zoom Window geometry and about the order
frames are processed in. Both consumers use it:

* ``Sam3UnifiedProcessor.propagate_with_llm_crop`` (LLM-guided Chained Detection), and
* Discern's Zoom Pass (``Sam3Backend.refine_all_frames``).

Sharing the decisions is the point: a fix here reaches both, instead of being
reimplemented once per consumer.

It is deliberately pure -- no tensors, no I/O, no GPU, and nothing heavy imported.
It must stay importable without ``torch`` so its tests run anywhere. Note that
``sam3/__init__.py`` *does* import the model, so tests import this module without
executing that package ``__init__`` (see ``tests/conftest.py``).

Vocabulary is Discern's glossary (see Discern ``CONTEXT.md``):

Zoom Window
    The square region of a frame handed to SAM3 so one Object fills more of the
    model's input.
Zoom Anchor
    Whatever decides where that window sits: the Object's own mask from the
    neighbouring frame, or a vision model asked to relocate it.
Re-grounding
    Asking a vision model to locate an Object afresh from its Object Hint.
Zoom Lock
    Sizing the window in proportion to the Object's mask, so the Object's
    apparent size stays constant as it approaches or recedes.

Conventions
-----------
* Boxes and windows are pixel coordinates ``(x1, y1, x2, y2)``, x2/y2 **exclusive**
  (the convention Discern's Zoom Pass already uses). Callers holding the inclusive
  convention of ``sam3.agent.helpers.frame_renderer.mask_to_bbox`` must add 1.
* Nothing here is silently forgiving: invalid input raises ``ValueError`` rather
  than being coerced, per Discern's no-fallback rule.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

__all__ = [
    "BBox",
    "FORWARD",
    "BACKWARD",
    "BOTH",
    "ZoomAnchorSource",
    "ZoomWindow",
    "ZoomAnchorConfig",
    "DEFAULT_ZOOM_ANCHOR_CONFIG",
    "ObjectZoomState",
    "ZoomAnchor",
    "ProcessingPass",
    "plan_processing_passes",
    "plan_processing_order",
    "should_re_ground",
    "re_grounding_is_fresh",
    "resolve_zoom_anchor",
    "resolve_zoom_anchors",
    "advance_object_state",
]

# (x1, y1, x2, y2) in pixels, x2/y2 exclusive.
BBox = Tuple[int, int, int, int]

FORWARD = "forward"
BACKWARD = "backward"
BOTH = "both"


# DISCERN FORK LOCAL ADDITION
class ZoomAnchorSource(str, Enum):
    """Which Zoom Anchor actually produced a window, for the per-frame record.

    A run that quietly degraded to ``FULL_FRAME`` on every frame is otherwise
    indistinguishable from a fully re-grounded one; recording this is the point.

    ``FULL_FRAME`` means nothing was cropped, for either of the reasons
    :func:`resolve_zoom_anchor` documents: nothing is known about where the Object
    is, or it is too big for zooming to buy anything.
    """

    RE_GROUNDED = "re-grounded"
    MASK_DERIVED = "mask-derived"
    STALE = "stale"
    FULL_FRAME = "full-frame"


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ZoomWindow:
    """A region of a frame, x2/y2 exclusive.

    Every window is square except the whole frame -- see
    :func:`resolve_zoom_anchor` for the two cases that produce that one.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate Zoom Window: {self.as_tuple()}")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def is_square(self) -> bool:
        return self.width == self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_tuple(self) -> BBox:
        return (self.x1, self.y1, self.x2, self.y2)


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ZoomAnchorConfig:
    """Everything the guards need. Defaults match ADR 0001's stated numbers.

    Args:
        crop_padding: Fraction of the Object's longest side added on *each* side,
            so a window is ``(1 + 2 * crop_padding)`` times that side.
        max_size_change: Largest fraction by which window size may change between
            consecutive frames (the rate limit).
        min_size_fraction: Floor on window size, as a fraction of the frame's
            shorter side.
        min_size_px: Absolute floor in pixels, below which SAM3's input is useless.
        area_collapse_ratio: If an Object's mask area drops below this fraction of
            the area one frame earlier, window size is frozen instead of followed
            downward.
        max_window_frame_fraction: An Object whose Zoom Window would reach this
            fraction of the frame's shorter side is not zoomed at all -- the whole
            frame is used. Cropping to a square that large buys no resolution, and
            for an Object wider than the frame is tall it would cut the Object off
            at the window edges on every frame. See :func:`resolve_zoom_anchor`.
        re_grounding_interval: Tick cadence -- one Re-grounding request every N
            frames. ``N`` means N, not N+1.
        max_re_grounding_age: How many frames after the frame it describes a
            Re-grounding answer may still be applied as *fresh*. A vision model
            slower than the frame rate answers about a frame the pass has long
            left behind; past this age the answer is only that Object's stale box,
            which the precedence ranks below the Object's own current mask.
    """

    crop_padding: float = 0.5
    max_size_change: float = 0.15
    min_size_fraction: float = 1.0 / 6.0
    min_size_px: int = 32
    area_collapse_ratio: float = 0.5
    max_window_frame_fraction: float = 0.9
    re_grounding_interval: int = 5
    max_re_grounding_age: int = 1

    def __post_init__(self):
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be >= 0")
        if not 0 < self.max_size_change:
            raise ValueError("max_size_change must be > 0")
        if not 0 < self.min_size_fraction <= 1:
            raise ValueError("min_size_fraction must be in (0, 1]")
        if self.min_size_px < 1:
            raise ValueError("min_size_px must be >= 1")
        if not 0 < self.area_collapse_ratio <= 1:
            raise ValueError("area_collapse_ratio must be in (0, 1]")
        if not 0 < self.max_window_frame_fraction <= 1:
            raise ValueError("max_window_frame_fraction must be in (0, 1]")
        if self.re_grounding_interval < 1:
            raise ValueError("re_grounding_interval must be >= 1")
        if self.max_re_grounding_age < 0:
            raise ValueError("max_re_grounding_age must be >= 0")


DEFAULT_ZOOM_ANCHOR_CONFIG = ZoomAnchorConfig()


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ObjectZoomState:
    """What the resolver remembers about one Object between frames.

    One instance per Object; no field of one Object's state is ever read while
    resolving another's, which is what makes windows per-Object independent.

    Args:
        previous_bbox: Bounding box of this Object's mask on the previously
            processed frame, or ``None`` when the Object has no mask at all
            (never seeded, or absent on that frame).
        previous_mask_area: Mask area in pixels on that frame.
        earlier_mask_area: Mask area one frame before that, used only to notice a
            collapse between two consecutive frames.
        previous_window_size: Side of the square window used on the previous
            frame, which the rate limit paces against.
        stale_re_grounding: The most recent successful Re-grounding box for this
            Object, however old. Loses to a fresh mask bounding box by design.
    """

    previous_bbox: Optional[BBox] = None
    previous_mask_area: Optional[int] = None
    earlier_mask_area: Optional[int] = None
    previous_window_size: Optional[int] = None
    stale_re_grounding: Optional[BBox] = None

    def __post_init__(self):
        if self.previous_bbox is not None:
            _validate_bbox(self.previous_bbox, "previous_bbox")
        if self.stale_re_grounding is not None:
            _validate_bbox(self.stale_re_grounding, "stale_re_grounding")
        if self.previous_window_size is not None and self.previous_window_size < 1:
            raise ValueError("previous_window_size must be >= 1")


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ZoomAnchor:
    """The resolver's answer for one Object on one frame."""

    window: ZoomWindow
    source: ZoomAnchorSource


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ProcessingPass:
    """One run of frames away from the seed frame, in one direction."""

    direction: str
    frames: Tuple[int, ...]


def _validate_bbox(bbox: BBox, name: str) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate {name}: {bbox}")


def _validate_frame_size(frame_width: int, frame_height: int) -> None:
    if frame_width < 1 or frame_height < 1:
        raise ValueError(f"frame must be non-empty, got {frame_width}x{frame_height}")


# DISCERN FORK LOCAL ADDITION
def plan_processing_passes(
    seed_frame: int,
    frame_count: int,
    direction: str = BOTH,
) -> Tuple[ProcessingPass, ...]:
    """Split the video into the passes to run, away from the seed frame.

    The seed frame itself is never in a pass: it already carries the Seed Masks,
    and re-detecting it would overwrite what the person drew.

    ``"both"`` yields the forward pass first, then the backward one. Callers must
    restart each Object's chain from its Seed Mask at the pass boundary -- that is
    exactly what the two passes are for.

    An unknown direction raises rather than being treated as forward; treating
    "anything but backward" as forward is the bug that lost every frame before the
    seed frame.
    """
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if not 0 <= seed_frame < frame_count:
        raise ValueError(f"seed_frame {seed_frame} outside 0..{frame_count - 1}")

    forward = ProcessingPass(FORWARD, tuple(range(seed_frame + 1, frame_count)))
    backward = ProcessingPass(BACKWARD, tuple(range(seed_frame - 1, -1, -1)))

    if direction == FORWARD:
        passes = (forward,)
    elif direction == BACKWARD:
        passes = (backward,)
    elif direction == BOTH:
        passes = (forward, backward)
    else:
        raise ValueError(
            f"unknown direction {direction!r}, expected "
            f"{FORWARD!r}, {BACKWARD!r} or {BOTH!r}"
        )
    return tuple(p for p in passes if p.frames)


# DISCERN FORK LOCAL ADDITION
def plan_processing_order(
    seed_frame: int,
    frame_count: int,
    direction: str = BOTH,
) -> Tuple[int, ...]:
    """The flattened frame order, for callers that do not care about the seam."""
    order: Tuple[int, ...] = ()
    for pass_ in plan_processing_passes(seed_frame, frame_count, direction):
        order += pass_.frames
    return order


# DISCERN FORK LOCAL ADDITION
def should_re_ground(
    frame_index: int,
    seed_frame: int,
    config: ZoomAnchorConfig = DEFAULT_ZOOM_ANCHOR_CONFIG,
) -> bool:
    """Whether this frame is a Re-grounding tick.

    Interval N means a request every N frames: the first frame of each pass, then
    every Nth frame after it. Because the answer is a function of the frame index
    alone, a failed request cannot shift the cadence and no counter can drift by
    one -- it simply waits for the next tick.
    """
    step = abs(frame_index - seed_frame) - 1
    if step < 0:
        raise ValueError("the seed frame is not processed, so it has no tick")
    return step % config.re_grounding_interval == 0


# DISCERN FORK LOCAL ADDITION
def re_grounding_is_fresh(
    described_frame: int,
    frame_index: int,
    config: ZoomAnchorConfig = DEFAULT_ZOOM_ANCHOR_CONFIG,
) -> bool:
    """Whether a Re-grounding answer still describes where the Object is now.

    A request asks where an Object is on the frame it was made for; the answer is
    about *that* frame, whenever it happens to arrive. Requests are not waited on,
    so a provider slower than the frame rate answers about a frame the pass left
    behind several frames ago.

    Only an answer no older than ``max_re_grounding_age`` frames may be used as
    ``fresh_re_grounding``, because a fresh Re-grounding outranks the Object's own
    mask bounding box. An older one is worth exactly what the precedence says a
    stale box is worth -- pass it to :func:`advance_object_state` instead, where it
    becomes the Object's ``stale_re_grounding`` and loses to any current mask.

    Distance is measured in frames, unsigned: a backward pass counts down.
    """
    return abs(frame_index - described_frame) <= config.max_re_grounding_age


def _place_square(
    center_x: float,
    center_y: float,
    size: int,
    frame_width: int,
    frame_height: int,
) -> ZoomWindow:
    """Centre a square of ``size`` and slide it back inside the frame.

    Clamping translates the window rather than shrinking it, so the window stays
    square at a frame edge; the caller keeps the size it asked for and the Object
    simply sits off-centre. A square too large to fit never reaches here:
    :func:`resolve_zoom_anchor` uses the whole frame instead of cropping to it.
    """
    x1 = int(round(center_x - size / 2.0))
    y1 = int(round(center_y - size / 2.0))
    x1 = max(0, min(x1, frame_width - size))
    y1 = max(0, min(y1, frame_height - size))
    return ZoomWindow(x1, y1, x1 + size, y1 + size)


def _locked_size(
    anchor_bbox: BBox,
    state: ObjectZoomState,
    frame_width: int,
    frame_height: int,
    config: ZoomAnchorConfig,
) -> int:
    """Zoom Lock plus its three guards, in the order they must be applied.

    The size returned is what the Object asks for, not what fits: deciding that a
    window this large is not worth cropping to belongs to
    :func:`resolve_zoom_anchor`, which has the whole frame to fall back on.
    """
    x1, y1, x2, y2 = anchor_bbox

    # Zoom Lock: size proportional to the Object, so apparent size after SAM3's
    # square resize stays constant as the Object approaches or recedes.
    wanted = int(round(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * config.crop_padding)))

    # Guard 1 -- area-collapse freeze. A mask that lost most of its area between
    # the two previous frames is not evidence that the Object shrank, so hold the
    # size instead of following it down (and then starving the next frame).
    if (
        state.previous_window_size is not None
        and state.earlier_mask_area is not None
        and state.previous_mask_area is not None
        and state.earlier_mask_area > 0
        and state.previous_mask_area
        < state.earlier_mask_area * config.area_collapse_ratio
    ):
        wanted = state.previous_window_size

    # Guard 2 -- rate limit. At most a bounded fraction of change per frame.
    size = wanted
    previous = state.previous_window_size
    if previous is not None:
        upper = max(previous + 1, int(round(previous * (1.0 + config.max_size_change))))
        lower = min(previous - 1, int(round(previous * (1.0 - config.max_size_change))))
        size = max(min(size, upper), max(1, lower))

    # Guard 3 -- floor. Applied after the rate limit so it always wins: a window
    # can never be paced downward past the floor.
    floor = max(
        config.min_size_px,
        int(round(min(frame_width, frame_height) * config.min_size_fraction)),
    )
    return max(1, max(size, floor))


def _select_anchor(
    state: ObjectZoomState,
    fresh_re_grounding: Optional[BBox],
) -> Tuple[Optional[BBox], ZoomAnchorSource]:
    """The precedence, fixed and explicit.

    Fresh Re-grounding for *this* Object, else this Object's own previous mask
    bounding box, else the last (stale) Re-grounding result for this Object, else
    the full frame. The stale result losing to a fresh mask is the whole point:
    the other order pins a moving Object to where the vision model saw it up to a
    tick ago.
    """
    if fresh_re_grounding is not None:
        _validate_bbox(fresh_re_grounding, "re-grounding box")
        return fresh_re_grounding, ZoomAnchorSource.RE_GROUNDED
    if state.previous_bbox is not None:
        return state.previous_bbox, ZoomAnchorSource.MASK_DERIVED
    if state.stale_re_grounding is not None:
        return state.stale_re_grounding, ZoomAnchorSource.STALE
    return None, ZoomAnchorSource.FULL_FRAME


# DISCERN FORK LOCAL ADDITION
def resolve_zoom_anchor(
    state: ObjectZoomState,
    fresh_re_grounding: Optional[BBox],
    frame_width: int,
    frame_height: int,
    config: ZoomAnchorConfig = DEFAULT_ZOOM_ANCHOR_CONFIG,
) -> ZoomAnchor:
    """Resolve one Object's Zoom Window on one frame.

    ``fresh_re_grounding`` is that Object's box from a Re-grounding request that
    was made for this frame and is no older than ``max_re_grounding_age`` frames
    (see :func:`re_grounding_is_fresh`); ``None`` covers every other case (no tick,
    a failed request, an answer that arrived too late, or a response that did not
    mention this Object).

    Two things produce the whole frame rather than a square, and both are recorded
    as ``FULL_FRAME`` because in both nothing was cropped:

    * Nothing is known about where the Object is. A square could not contain a
      non-square frame anyway, and there is no defensible side to crop off.
    * The Object is too big to zoom -- the window it asks for reaches
      ``max_window_frame_fraction`` of the frame's shorter side. Cropping to the
      largest square that fits would gain no resolution and would truncate an
      Object wider than the frame is tall at the window edges, on every frame,
      permanently: the next window re-anchors on the truncated mask and lands in
      the same place. Running the frame whole is strictly better, and saying so in
      the record is what makes the lost zoom visible.
    """
    _validate_frame_size(frame_width, frame_height)
    full_frame = ZoomWindow(0, 0, frame_width, frame_height)
    anchor_bbox, source = _select_anchor(state, fresh_re_grounding)
    if anchor_bbox is None:
        return ZoomAnchor(full_frame, source)

    size = _locked_size(anchor_bbox, state, frame_width, frame_height, config)
    if size >= min(frame_width, frame_height) * config.max_window_frame_fraction:
        return ZoomAnchor(full_frame, ZoomAnchorSource.FULL_FRAME)

    x1, y1, x2, y2 = anchor_bbox
    window = _place_square(
        (x1 + x2) / 2.0, (y1 + y2) / 2.0, size, frame_width, frame_height
    )
    return ZoomAnchor(window, source)


# DISCERN FORK LOCAL ADDITION
def resolve_zoom_anchors(
    object_states: Mapping[int, ObjectZoomState],
    re_grounding: Optional[Mapping[int, Optional[BBox]]],
    frame_width: int,
    frame_height: int,
    config: ZoomAnchorConfig = DEFAULT_ZOOM_ANCHOR_CONFIG,
) -> Dict[int, ZoomAnchor]:
    """Resolve one Zoom Window per Object for one frame.

    ``re_grounding`` is this frame's Re-grounding result keyed by Object id, or
    ``None`` when no request was made. Objects missing from it, or mapped to
    ``None``, simply fall through the precedence.

    Each Object is resolved from its own state alone, so one Object's position can
    never influence another's framing.
    """
    return {
        object_id: resolve_zoom_anchor(
            state,
            re_grounding.get(object_id) if re_grounding else None,
            frame_width,
            frame_height,
            config,
        )
        for object_id, state in object_states.items()
    }


# DISCERN FORK LOCAL ADDITION
def advance_object_state(
    state: ObjectZoomState,
    mask_bbox: Optional[BBox],
    mask_area: Optional[int],
    anchor: ZoomAnchor,
    fresh_re_grounding: Optional[BBox] = None,
) -> ObjectZoomState:
    """Carry one Object's state to the next frame.

    Pass what actually happened, including the :class:`ZoomAnchor` the frame was
    resolved with. ``mask_bbox=None`` means the Object produced no mask on this
    frame, and the next frame will therefore fall through to the stale
    Re-grounding box or the full frame -- absent means absent, and the resolver
    will not quietly keep pointing at a mask that is no longer there. A caller
    that would rather hold the anchor for one dropped frame should keep the
    current state instead of calling this.

    The window size is remembered only when a window was actually cropped, so
    neither full-frame case resets the rate limit to the whole frame. It is the
    anchor's *source* that says so, not the window's shape: on a square-resolution
    video the full frame is itself a square, and reading squareness there stored
    the whole frame as the size to pace against, holding the window near
    full-frame for a dozen frames after the Object was reacquired.
    """
    if mask_bbox is not None:
        _validate_bbox(mask_bbox, "mask_bbox")
    cropped = anchor.source is not ZoomAnchorSource.FULL_FRAME
    return replace(
        state,
        previous_bbox=mask_bbox,
        previous_mask_area=mask_area,
        earlier_mask_area=state.previous_mask_area,
        previous_window_size=(
            anchor.window.width if cropped else state.previous_window_size
        ),
        stale_re_grounding=(
            fresh_re_grounding
            if fresh_re_grounding is not None
            else state.stale_re_grounding
        ),
    )
