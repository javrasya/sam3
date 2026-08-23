# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Per-Object bookkeeping for one pass of LLM-guided Chained Detection.

DISCERN FORK LOCAL ADDITION -- this whole module is local to javrasya/sam3 and has
no upstream counterpart. Every public callable below carries the same marker, per
Discern ADR 0002 ("Recording How Our SAM3 Fork Diverges From Upstream").

``Sam3UnifiedProcessor.propagate_with_llm_crop`` walks one Processing Pass away
from the seed frame, and on every frame it has to know four things: which Objects
are still worth processing, where each of their Zoom Windows goes, which Zoom
Anchor produced each window, and which Objects have been absent long enough to
give up on. None of that involves a tensor, a GPU or a network call, so none of it
lives in the propagation loop -- it lives here, where it can be tested with plain
values.

Geometry itself is not decided here: every window comes from
:mod:`sam3.zoom_anchor`, which owns that decision for both this path and Discern's
Zoom Pass. This module only carries state from frame to frame and keeps the record.

Vocabulary is Discern's glossary (see Discern ``CONTEXT.md``):

Chained Detection
    Detecting each frame afresh inside a Zoom Window, guided only by the previous
    frame's mask. There is no tracker on this path and therefore no memory beyond
    one frame -- which is exactly why an Object that goes absent cannot come back
    on its own, and why the absence streak below has to end the chain.
Zoom Anchor
    Whatever decides where a Zoom Window sits. Recorded per frame so a run that
    quietly degraded to the full frame is distinguishable from one that worked.

Conventions
-----------
* Bounding boxes are pixel ``(x1, y1, x2, y2)`` with x2/y2 **exclusive**, matching
  :mod:`sam3.zoom_anchor`.
* Nothing here is silently forgiving: a caller that reports an outcome for an
  Object that was not processed, or omits one that was, gets a ``ValueError``.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from sam3.zoom_anchor import (
    DEFAULT_ZOOM_ANCHOR_CONFIG,
    BBox,
    ObjectZoomState,
    ZoomAnchor,
    ZoomAnchorConfig,
    ZoomAnchorSource,
    advance_object_state,
    re_grounding_is_fresh,
    resolve_zoom_anchors,
    should_re_ground,
)

__all__ = [
    "DEFAULT_ABSENCE_STREAK_LIMIT",
    "MaskObservation",
    "FrameRecord",
    "PassLedger",
    "partition_re_grounding",
]

# How many consecutive frames an Object may be absent before it stops being
# propagated for the remainder of the pass.
DEFAULT_ABSENCE_STREAK_LIMIT = 5


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class MaskObservation:
    """What one Object's mask actually was on one frame.

    Args:
        bbox: Bounding box of the produced mask, x2/y2 exclusive.
        area: Mask area in pixels, which the Zoom Lock's collapse guard reads.
    """

    bbox: BBox
    area: int

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate observation bbox: {self.bbox}")
        if self.area < 1:
            raise ValueError("an observed mask has at least one pixel")


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class FrameRecord:
    """What happened to every Object on one frame.

    This is the per-frame Zoom Anchor record the spec asks for. It travels back to
    the caller alongside the masks, so "every window fell back to the full frame"
    and "every window was re-grounded" are no longer the same observation.

    Args:
        frame_index: The frame these outcomes belong to.
        anchor_sources: Object id -> Zoom Anchor source value, for the Objects that
            were actually processed on this frame.
        absent_obj_ids: Every Object carrying no mask on this frame, whether it was
            processed and found nothing or had already been stopped.
        stopped_obj_ids: Objects that were no longer being propagated when this
            frame ran.
        newly_stopped_obj_ids: Objects whose absence streak ended on this frame.
    """

    frame_index: int
    anchor_sources: Dict[int, str]
    absent_obj_ids: Tuple[int, ...]
    stopped_obj_ids: Tuple[int, ...]
    newly_stopped_obj_ids: Tuple[int, ...]

    # DISCERN FORK LOCAL ADDITION
    def summary(self) -> str:
        """One human-readable line, stable enough to log only when it changes."""
        counts = {source.value: 0 for source in ZoomAnchorSource}
        for value in self.anchor_sources.values():
            counts[value] += 1
        parts = [f"{name}={count}" for name, count in counts.items()]
        parts.append(f"absent={len(self.absent_obj_ids)}")
        parts.append(f"stopped={len(self.stopped_obj_ids)}")
        return " ".join(parts)


# DISCERN FORK LOCAL ADDITION
def partition_re_grounding(
    boxes: Optional[Mapping[int, Optional[BBox]]],
) -> Tuple[Dict[int, BBox], Dict[int, object]]:
    """Split a Re-grounding response into boxes we can use and boxes we cannot.

    A vision model can answer with a degenerate or malformed box. Handing that to
    the resolver would abort the whole propagation over one bad reply, and quietly
    repairing it would be exactly the silent fallback this feature is being cured
    of. So it is rejected, by name, and the caller logs what was thrown away; that
    Object then falls through the documented Zoom Anchor precedence like any other
    Object the response did not mention.

    ``None`` values are not rejections -- they are the normal way of saying "this
    response has nothing for that Object".

    Returns:
        ``(accepted, rejected)``, the second keyed by Object id with the offending
        value, for reporting.
    """
    accepted: Dict[int, BBox] = {}
    rejected: Dict[int, object] = {}
    if not boxes:
        return accepted, rejected

    for obj_id, box in boxes.items():
        if box is None:
            continue
        try:
            x1, y1, x2, y2 = (int(v) for v in box)
        except (TypeError, ValueError):
            rejected[obj_id] = box
            continue
        if x2 <= x1 or y2 <= y1:
            rejected[obj_id] = box
            continue
        accepted[obj_id] = (x1, y1, x2, y2)
    return accepted, rejected


# DISCERN FORK LOCAL ADDITION
class PassLedger:
    """Per-Object state for one Processing Pass, and the record it produces.

    One ledger per pass, not per video: ``plan_processing_passes`` splits a "both"
    run into a forward and a backward pass, and each of them starts again from the
    Seed Masks. Constructing a second ledger for the second pass is what makes that
    restart happen.
    """

    def __init__(
        self,
        seed_states: Mapping[int, ObjectZoomState],
        seed_frame: int,
        frame_width: int,
        frame_height: int,
        config: ZoomAnchorConfig = DEFAULT_ZOOM_ANCHOR_CONFIG,
        absence_streak_limit: int = DEFAULT_ABSENCE_STREAK_LIMIT,
    ):
        if not seed_states:
            raise ValueError("a pass needs at least one Object")
        if absence_streak_limit < 1:
            raise ValueError("absence_streak_limit must be >= 1")
        if frame_width < 1 or frame_height < 1:
            raise ValueError(
                f"frame must be non-empty, got {frame_width}x{frame_height}"
            )

        self._states: Dict[int, ObjectZoomState] = dict(seed_states)
        self._object_ids: Tuple[int, ...] = tuple(sorted(seed_states))
        self._seed_frame = seed_frame
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._config = config
        self._absence_streak_limit = absence_streak_limit

        self._active = set(self._object_ids)
        self._streaks: Dict[int, int] = {obj_id: 0 for obj_id in self._object_ids}
        self._source_counts: Dict[str, int] = {
            source.value: 0 for source in ZoomAnchorSource
        }
        self._absent_observations = 0
        self._tick_frames = set()
        self._skipped_tick_frames = set()
        self._aged_re_groundings = 0

    # DISCERN FORK LOCAL ADDITION
    @property
    def object_ids(self) -> Tuple[int, ...]:
        """Every Object in the pass, in a stable order, stopped ones included."""
        return self._object_ids

    # DISCERN FORK LOCAL ADDITION
    @property
    def active_object_ids(self) -> Tuple[int, ...]:
        """The Objects still being propagated, in a stable order."""
        return tuple(obj_id for obj_id in self._object_ids if obj_id in self._active)

    # DISCERN FORK LOCAL ADDITION
    @property
    def stopped_object_ids(self) -> Tuple[int, ...]:
        """The Objects whose absence streak ended their chain."""
        return tuple(
            obj_id for obj_id in self._object_ids if obj_id not in self._active
        )

    # DISCERN FORK LOCAL ADDITION
    def state_of(self, obj_id: int) -> ObjectZoomState:
        """This Object's carried state, for callers that need to inspect it."""
        return self._states[obj_id]

    # DISCERN FORK LOCAL ADDITION
    def previous_bboxes(self) -> Dict[int, BBox]:
        """Active Objects' previous mask boxes, for the Re-grounding overlay.

        Boxes are returned in the *inclusive* convention that
        ``sam3.agent.helpers.frame_renderer`` draws with, which is not the
        exclusive one used everywhere else here.
        """
        boxes = {}
        for obj_id in self.active_object_ids:
            bbox = self._states[obj_id].previous_bbox
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                boxes[obj_id] = (x1, y1, x2 - 1, y2 - 1)
        return boxes

    # DISCERN FORK LOCAL ADDITION
    def should_re_ground(self, frame_index: int) -> bool:
        """Whether this frame is a Re-grounding tick with anything left to ask about.

        The cadence itself is the resolver's, a function of the frame index alone,
        so a request that failed or was skipped cannot shift it by one.
        """
        if not self._active:
            return False
        due = should_re_ground(frame_index, self._seed_frame, self._config)
        if due:
            # Keyed by frame so asking twice about one frame counts it once: the
            # denominator of the effective cadence has to be frames, not calls.
            self._tick_frames.add(frame_index)
        return due

    # DISCERN FORK LOCAL ADDITION
    def note_tick_skipped(self, frame_index: int) -> int:
        """Record a tick that issued nothing because the last request is in flight.

        The cadence a person chose is only the cadence they got while the provider
        answers faster than ``interval`` frames take. When it does not, requests
        go out every *latency* frames instead, and the only honest thing to do is
        count it: :meth:`effective_re_grounding_interval` turns the count into the
        number they actually got, and the pass summary carries it out of the loop.

        Returns the number of frames in this pass whose tick issued nothing.
        """
        self._skipped_tick_frames.add(frame_index)
        return len(self._skipped_tick_frames)

    # DISCERN FORK LOCAL ADDITION
    def effective_re_grounding_interval(self) -> Optional[float]:
        """Frames per Re-grounding request actually issued, or None before any tick.

        ``interval * ticks_due / ticks_issued``: with nothing skipped it is the
        configured interval, and with two ticks in three skipped it is three times
        it, which is the number worth telling the annotator.
        """
        due = len(self._tick_frames)
        if due == 0:
            return None
        issued = due - len(self._skipped_tick_frames)
        if issued < 1:
            return None
        return self._config.re_grounding_interval * due / issued

    # DISCERN FORK LOCAL ADDITION
    def classify_re_grounding(
        self,
        boxes: Mapping[int, BBox],
        described_frame: int,
        frame_index: int,
    ) -> Tuple[Dict[int, BBox], Dict[int, BBox]]:
        """Split an arrived Re-grounding answer by how old it is.

        Returns ``(fresh, aged)``. ``fresh`` may anchor this frame, where it
        outranks the Object's own mask. ``aged`` may not: it describes where the
        Object was on a frame the pass has already left behind, and preferring it
        to the mask produced on the previous frame is exactly the lag this feature
        is being cured of. Hand ``aged`` to :meth:`record_frame` anyway -- it
        becomes each Object's stale box, which the precedence ranks *below* a
        current mask and above nothing at all.
        """
        if not boxes:
            return {}, {}
        if re_grounding_is_fresh(described_frame, frame_index, self._config):
            return dict(boxes), {}
        self._aged_re_groundings += len(boxes)
        return {}, dict(boxes)

    # DISCERN FORK LOCAL ADDITION
    def resolve(
        self,
        re_grounding: Optional[Mapping[int, Optional[BBox]]] = None,
    ) -> Dict[int, ZoomAnchor]:
        """One Zoom Window per still-active Object, from the resolver."""
        return resolve_zoom_anchors(
            {obj_id: self._states[obj_id] for obj_id in self.active_object_ids},
            re_grounding,
            self._frame_width,
            self._frame_height,
            self._config,
        )

    # DISCERN FORK LOCAL ADDITION
    def record_frame(
        self,
        frame_index: int,
        anchors: Mapping[int, ZoomAnchor],
        observations: Mapping[int, Optional[MaskObservation]],
        re_grounding: Optional[Mapping[int, Optional[BBox]]] = None,
    ) -> FrameRecord:
        """Carry every active Object to the next frame and record what happened.

        ``observations`` must have one entry per active Object: a
        :class:`MaskObservation` when SAM3 produced a mask, or ``None`` when it
        genuinely did not. ``None`` means absent, and absent is carried forward as
        absent -- the Object's anchor is not held at its last known position, and
        its mask is not reused as the next frame's guidance.
        """
        active = self.active_object_ids
        stopped_before = self.stopped_object_ids

        missing = [obj_id for obj_id in active if obj_id not in observations]
        if missing:
            raise ValueError(f"no outcome reported for active Objects {missing}")
        extra = [obj_id for obj_id in observations if obj_id not in self._active]
        if extra:
            raise ValueError(
                f"outcome reported for Objects not being propagated: {extra}"
            )
        unanchored = [obj_id for obj_id in active if obj_id not in anchors]
        if unanchored:
            raise ValueError(f"no Zoom Window resolved for active Objects {unanchored}")

        anchor_sources: Dict[int, str] = {}
        absent = list(stopped_before)
        newly_stopped = []

        for obj_id in active:
            anchor = anchors[obj_id]
            anchor_sources[obj_id] = anchor.source.value
            self._source_counts[anchor.source.value] += 1

            fresh = re_grounding.get(obj_id) if re_grounding else None
            observation = observations[obj_id]

            if observation is None:
                self._absent_observations += 1
                absent.append(obj_id)
                self._streaks[obj_id] += 1
                self._states[obj_id] = advance_object_state(
                    self._states[obj_id], None, None, anchor, fresh
                )
                if self._streaks[obj_id] >= self._absence_streak_limit:
                    self._active.discard(obj_id)
                    newly_stopped.append(obj_id)
            else:
                self._streaks[obj_id] = 0
                self._states[obj_id] = advance_object_state(
                    self._states[obj_id],
                    observation.bbox,
                    observation.area,
                    anchor,
                    fresh,
                )

        return FrameRecord(
            frame_index=frame_index,
            anchor_sources=anchor_sources,
            absent_obj_ids=tuple(sorted(absent)),
            stopped_obj_ids=stopped_before,
            newly_stopped_obj_ids=tuple(sorted(newly_stopped)),
        )

    # DISCERN FORK LOCAL ADDITION
    def pass_summary(self) -> str:
        """One line for the end of the pass: where every window actually came from.

        It also carries what the Re-grounding cadence really was. A tick that
        issues nothing, and an answer that arrives too late to anchor with, are
        both invisible frame by frame and both mean the annotator's chosen
        interval is not the interval they got.
        """
        parts = [f"{name}={count}" for name, count in self._source_counts.items()]
        parts.append(f"absent={self._absent_observations}")
        parts.append(f"stopped={list(self.stopped_object_ids)}")
        parts.append(f"ticks_skipped={len(self._skipped_tick_frames)}")
        parts.append(f"re_groundings_too_late={self._aged_re_groundings}")
        effective = self.effective_re_grounding_interval()
        if effective is not None:
            parts.append(f"effective_interval={effective:.1f}")
        return " ".join(parts)
