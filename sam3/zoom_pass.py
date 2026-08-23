# DISCERN FORK LOCAL ADDITION -- not part of upstream SAM3 (see Discern ADR 0002).
"""Pure Zoom Pass decisions: per-Object windows, per-Object abandonment, honest scoring.

DISCERN FORK LOCAL ADDITION -- this whole module is local to javrasya/sam3 and has
no upstream counterpart. Every public callable below carries the same marker, per
Discern ADR 0002 ("Recording How Our SAM3 Fork Diverges From Upstream").

The Zoom Pass itself (Discern's ``Sam3Backend.refine_all_frames``) is tensors, PIL
crops and a GPU. Everything it *decides* lives here, in the same style as
:mod:`sam3.zoom_anchor` and on top of it: no tensors, no I/O, no GPU, nothing heavy
imported, so the decisions are testable on any machine.

Window geometry is not re-derived here. It is delegated to
:func:`sam3.zoom_anchor.resolve_zoom_anchor`, which is the same resolver the
LLM-guided propagation loop uses -- so a fix to the Zoom Anchor logic reaches both
consumers instead of being reimplemented once per consumer.

What this module adds on top of the resolver is the three things that are the Zoom
Pass's own:

Per-Object abandonment
    A Zoom Window that already spans nearly the whole frame buys no resolution,
    so that Object is not re-segmented on that frame. This replaces the old rule
    that abandoned the *entire* pass -- every Object, every frame -- as soon as one
    shared window grew large.
Honest outcomes
    Every (frame, Object) the pass looked at gets a :class:`MaskOutcomeRecord`. Only
    ``IMPROVED`` carries a score, and only ``IMPROVED`` may be emitted as a mask:
    a mask the pass never improved is left exactly as propagation produced it,
    rather than being re-encoded, re-emitted and scored ``1.0``.
Its own verdict
    :class:`ZoomPassReport` accumulates those outcomes and answers the question
    the caller has to relay to a person: did this pass improve anything at all?

Vocabulary is Discern's glossary (see Discern ``CONTEXT.md``). Nothing here is
named *refinement*: the glossary reserves that word for the workflow state in
which a person hand-corrects frames, and this pass is automated end to end.
Discern's caller is still spelled ``Sam3Backend.refine_all_frames`` -- that is the
legacy name this vocabulary replaces, and the mapping is recorded here so it stays
findable rather than being carried into new type names.

Nothing in either repository calls this module yet: Discern's Zoom Pass still runs
the older shared-union-window code that ``refine_all_frames`` was written around,
and moving it onto these decisions is a follow-up of its own (see the Zoom Pass
backend hand-off note for spec javrasya/discern#91). Until then this module is
exercised only by ``tests/test_zoom_pass.py``.

Anchoring, and how it differs from propagation
----------------------------------------------
Propagation resolves frame ``n``'s window from the mask it produced on frame
``n-1``, because frame ``n``'s mask does not exist yet. The Zoom Pass is not
chained: it already holds a propagated mask for *every* frame, so each Object's
window on frame ``n`` is anchored on that Object's own mask on frame ``n`` --
which is the mask it is about to improve. The resolver reads that box from
``ObjectZoomState.previous_bbox``; :func:`plan_zoom_pass_frame` puts the current
frame's box there. The Zoom Lock guards (rate limit, area-collapse freeze) still
pace against the previously *processed* frame, so the window sequence stays stable
across the pass instead of jittering with each frame's mask noise.

Re-grounding never happens in the Zoom Pass -- there is no vision model in this
path -- so the anchor source recorded here is ``MASK_DERIVED`` for every Object
that has a mask. It is recorded anyway, per frame per Object, because the record
is what makes a degraded run visible rather than indistinguishable from a good one.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

from .zoom_anchor import (
    BBox,
    DEFAULT_ZOOM_ANCHOR_CONFIG,
    ObjectZoomState,
    ZoomAnchorConfig,
    ZoomAnchorSource,
    ZoomWindow,
    advance_object_state,
    resolve_zoom_anchor,
)

__all__ = [
    "ZoomPassConfig",
    "DEFAULT_ZOOM_PASS_CONFIG",
    "MaskOutcome",
    "ZoomPassStatus",
    "ObjectMaskGeometry",
    "ObjectZoomPlan",
    "MaskOutcomeRecord",
    "ZoomPassReport",
    "is_zoom_beneficial",
    "plan_zoom_pass_frame",
]


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ZoomPassConfig:
    """Zoom Pass settings: the resolver's, plus the abandonment threshold.

    Args:
        anchor: Geometry settings, handed straight to the Zoom Anchor resolver.
            ``anchor.max_window_frame_fraction`` is the abandonment threshold: an
            Object whose Zoom Window would reach that fraction of the frame's
            shorter side is abandoned *for that frame*, because cropping to it and
            resizing back gains no resolution. This is the old global 90% rule,
            now scoped to one Object and measured against a side rather than an
            area -- see :func:`is_zoom_beneficial`. It lives on the resolver's
            config because the resolver applies the same rule on the propagation
            path, and one rule spelled in two places is how the two drift apart.
    """

    anchor: ZoomAnchorConfig = DEFAULT_ZOOM_ANCHOR_CONFIG


DEFAULT_ZOOM_PASS_CONFIG = ZoomPassConfig()


# DISCERN FORK LOCAL ADDITION
class MaskOutcome(str, Enum):
    """What the Zoom Pass actually did to one Object's mask on one frame.

    Only ``IMPROVED`` means a new mask exists. The other two mean the propagated
    mask stands unchanged -- which is a truthful thing to report and a dishonest
    thing to re-emit as though the pass had produced it.
    """

    IMPROVED = "improved"
    NOT_DETECTED = "not-detected"
    NO_ZOOM_BENEFIT = "no-zoom-benefit"


# DISCERN FORK LOCAL ADDITION
class ZoomPassStatus(str, Enum):
    """The pass's verdict. The value is the status string the stream carries.

    ``NO_IMPROVEMENT`` is the outcome a person needs told: the pass ran, it
    finished, and not one mask changed. Reporting it as plain success is what
    makes people re-run a Zoom Pass hoping for a different result.
    """

    IMPROVED = "improved"
    NO_IMPROVEMENT = "no_improvement"


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ObjectMaskGeometry:
    """The only two things the decisions need from an Object's mask on a frame.

    Args:
        bbox: Bounding box of the propagated mask, x2/y2 exclusive.
        area: Number of set pixels in that mask.
    """

    bbox: BBox
    area: int

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate mask bbox: {self.bbox}")
        if self.area < 1:
            raise ValueError("mask area must be >= 1; an absent Object has no geometry")


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class ObjectZoomPlan:
    """What the Zoom Pass should do with one Object on one frame.

    Args:
        object_id: The Object this plan is for.
        window: Its own Zoom Window, from the resolver.
        anchor_source: Which Zoom Anchor produced that window, for the per-frame
            per-Object record.
        zoom_is_beneficial: False when the window is so close to the whole frame
            that re-segmenting inside it cannot add detail. The caller must then
            skip the backbone for this Object and record
            :attr:`MaskOutcome.NO_ZOOM_BENEFIT`.
        next_state: The Object's carried state for the next processed frame.
    """

    object_id: int
    window: ZoomWindow
    anchor_source: ZoomAnchorSource
    zoom_is_beneficial: bool
    next_state: ObjectZoomState


# DISCERN FORK LOCAL ADDITION
@dataclass(frozen=True)
class MaskOutcomeRecord:
    """One (frame, Object) result, scored honestly.

    ``score`` is the detection score of the mask the pass produced, and exists
    only when it produced one. A mask it did not touch has no score of its own --
    ``None`` says "not attempted", which is the distinction the old ``1.0``
    destroyed by making an untouched mask outrank every genuinely refined one.
    """

    frame_index: int
    object_id: int
    outcome: MaskOutcome
    anchor_source: ZoomAnchorSource
    score: Optional[float] = None

    def __post_init__(self):
        if self.outcome is MaskOutcome.IMPROVED:
            if self.score is None:
                raise ValueError("an improved mask must carry the score it scored")
            if not 0.0 <= self.score <= 1.0:
                raise ValueError(f"score {self.score} outside 0..1")
        elif self.score is not None:
            raise ValueError(
                f"outcome {self.outcome.value} produced no mask, so it has no score"
            )

    @property
    def emits_mask(self) -> bool:
        """Whether the caller may put a mask on the wire for this result.

        False means: leave the propagated mask alone. Not "send it again".
        """
        return self.outcome is MaskOutcome.IMPROVED


# DISCERN FORK LOCAL ADDITION
class ZoomPassReport:
    """Accumulates :class:`MaskOutcomeRecord` records and delivers the pass's verdict.

    Mutable on purpose: the Zoom Pass is a generator streaming frame by frame, and
    the verdict is only known once it has finished.
    """

    def __init__(self):
        self._outcomes: list = []

    def record(self, outcome: MaskOutcomeRecord) -> None:
        if not isinstance(outcome, MaskOutcomeRecord):
            raise TypeError(
                f"expected MaskOutcomeRecord, got {type(outcome).__name__}"
            )
        self._outcomes.append(outcome)

    @property
    def outcomes(self) -> Tuple[MaskOutcomeRecord, ...]:
        return tuple(self._outcomes)

    def count(self, outcome: MaskOutcome) -> int:
        return sum(1 for r in self._outcomes if r.outcome is outcome)

    @property
    def improved_count(self) -> int:
        return self.count(MaskOutcome.IMPROVED)

    @property
    def not_detected_count(self) -> int:
        return self.count(MaskOutcome.NOT_DETECTED)

    @property
    def no_zoom_benefit_count(self) -> int:
        return self.count(MaskOutcome.NO_ZOOM_BENEFIT)

    @property
    def total_count(self) -> int:
        return len(self._outcomes)

    @property
    def object_ids(self) -> Tuple[int, ...]:
        return tuple(sorted({r.object_id for r in self._outcomes}))

    @property
    def frames_improved(self) -> Tuple[int, ...]:
        return tuple(sorted({r.frame_index for r in self._outcomes if r.emits_mask}))

    @property
    def anchor_source_counts(self) -> Dict[ZoomAnchorSource, int]:
        """How many (frame, Object) decisions each Zoom Anchor produced.

        A run that is entirely ``FULL_FRAME`` here is a degraded run, and saying
        so is the whole reason the source is recorded.
        """
        counts: Dict[ZoomAnchorSource, int] = {}
        for r in self._outcomes:
            counts[r.anchor_source] = counts.get(r.anchor_source, 0) + 1
        return counts

    def outcomes_by_object(self) -> Dict[int, Dict[MaskOutcome, int]]:
        """Per-Object outcome counts, so one abandoned Object is visible.

        Abandonment is per Object now, so "the pass succeeded" and "Object 3 was
        never improved on any frame" are both true at once and both worth seeing.
        """
        by_object: Dict[int, Dict[MaskOutcome, int]] = {}
        for r in self._outcomes:
            counts = by_object.setdefault(r.object_id, {})
            counts[r.outcome] = counts.get(r.outcome, 0) + 1
        return by_object

    @property
    def status(self) -> ZoomPassStatus:
        """``IMPROVED`` if the pass improved at least one mask, else ``NO_IMPROVEMENT``.

        A pass that recorded nothing at all improved nothing, so it is
        ``NO_IMPROVEMENT`` too -- never a bare success.
        """
        return (
            ZoomPassStatus.IMPROVED
            if self.improved_count > 0
            else ZoomPassStatus.NO_IMPROVEMENT
        )

    def summary(self) -> str:
        """One sentence for a person, naming what happened and why.

        This is the text the caller shows when the pass could not improve
        anything; it is a contract with the reader, not a log line.
        """
        objects = _plural(len(self.object_ids), "Object")
        if self.status is ZoomPassStatus.IMPROVED:
            text = (
                f"Zoom Pass improved {self.improved_count} of "
                f"{_plural(self.total_count, 'mask')} across "
                f"{_plural(len(self.frames_improved), 'frame')} ({objects})"
            )
            unchanged = self._unchanged_clause()
            return f"{text}; {unchanged}." if unchanged else f"{text}."
        if self.total_count == 0:
            return "Zoom Pass improved nothing: it found no propagated mask to work on."
        unchanged = self._unchanged_clause()
        return (
            f"Zoom Pass improved none of {_plural(self.total_count, 'mask')} "
            f"({objects}): {unchanged}. Re-running it will do the same."
        )

    def _unchanged_clause(self) -> str:
        parts = []
        if self.not_detected_count:
            parts.append(
                f"{self.not_detected_count} left unchanged because SAM3 detected "
                f"nothing in the Object's Zoom Window"
            )
        if self.no_zoom_benefit_count:
            parts.append(
                f"{self.no_zoom_benefit_count} left unchanged because the Object's "
                f"Zoom Window already spans the frame, so zooming adds no detail"
            )
        return ", ".join(parts)


# DISCERN FORK LOCAL ADDITION
def is_zoom_beneficial(
    window: ZoomWindow,
    frame_width: int,
    frame_height: int,
    config: ZoomPassConfig = DEFAULT_ZOOM_PASS_CONFIG,
) -> bool:
    """Whether cropping to ``window`` buys this Object any resolution.

    Per Object, per frame. The old rule asked the same question of one window
    shared by everybody and, on a No, abandoned the whole pass -- so a single
    sprawling Object silently cost every other Object its improved mask.

    The comparison is the window's longest side against the frame's shortest
    side, not the old area-against-area: Zoom Windows are square, so on a 16:9
    frame a window can never exceed 56% of the frame *area* however large the
    Object is -- an area test would never fire, which is a guard that quietly does
    not exist.

    The resolver applies the same threshold, so a window it hands back over the
    line is already the whole frame; this stays a predicate on the window rather
    than a reading of the anchor source, so the Zoom Pass's own decision is
    checkable against any window a caller holds.
    """
    if frame_width < 1 or frame_height < 1:
        raise ValueError(f"frame must be non-empty, got {frame_width}x{frame_height}")
    window_side = max(window.width, window.height)
    frame_side = min(frame_width, frame_height)
    return window_side < frame_side * config.anchor.max_window_frame_fraction


def _plural(count: int, noun: str) -> str:
    """``3 masks`` / ``1 mask`` -- the summary is read by people, not parsers."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _anchor_state(
    state: ObjectZoomState, geometry: ObjectMaskGeometry
) -> ObjectZoomState:
    """Move this frame's own mask into the slot the resolver anchors on.

    Field for field this is what ``advance_object_state`` does, minus the window
    size it cannot know yet -- which is why the plan calls ``advance_object_state``
    for ``next_state`` rather than deriving that here as well.
    """
    return replace(
        state,
        previous_bbox=geometry.bbox,
        previous_mask_area=geometry.area,
        earlier_mask_area=state.previous_mask_area,
    )


# DISCERN FORK LOCAL ADDITION
def plan_zoom_pass_frame(
    frame_masks: Mapping[int, ObjectMaskGeometry],
    object_states: Mapping[int, ObjectZoomState],
    frame_width: int,
    frame_height: int,
    config: ZoomPassConfig = DEFAULT_ZOOM_PASS_CONFIG,
) -> Dict[int, ObjectZoomPlan]:
    """One Zoom Window per Object for one frame of the Zoom Pass.

    Args:
        frame_masks: Every Object that has a propagated mask on this frame, and
            that mask's geometry. Objects absent from this mapping have nothing
            to refine here; their carried state is simply left untouched by the
            caller.
        object_states: Carried state per Object, from the previously processed
            frame. Missing entries start fresh.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        config: Zoom Pass settings.

    Returns:
        One :class:`ObjectZoomPlan` per Object in ``frame_masks`` -- every one of
        them. No Object is dropped for being far from another Object, which is the
        behaviour the shared union window used to produce.

    Each Object is resolved from its own geometry and its own state alone, so two
    Objects at opposite edges of the frame each get a window centred on
    themselves, instead of one window centred between them containing neither.
    """
    plans: Dict[int, ObjectZoomPlan] = {}
    for object_id, geometry in frame_masks.items():
        state = object_states.get(object_id, ObjectZoomState())
        anchor = resolve_zoom_anchor(
            _anchor_state(state, geometry),
            None,  # the Zoom Pass has no vision model, so never a fresh Re-grounding
            frame_width,
            frame_height,
            config.anchor,
        )
        plans[object_id] = ObjectZoomPlan(
            object_id=object_id,
            window=anchor.window,
            anchor_source=anchor.source,
            zoom_is_beneficial=is_zoom_beneficial(
                anchor.window, frame_width, frame_height, config
            ),
            next_state=advance_object_state(
                state, geometry.bbox, geometry.area, anchor
            ),
        )
    return plans
