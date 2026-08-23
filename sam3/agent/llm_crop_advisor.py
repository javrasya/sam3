# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
LLM-guided crop zone prediction for video tracking.

Uses a vision LLM to predict per-frame crop zones during propagation,
so the SAM3 tracker always sees small objects at high effective resolution.

DISCERN FORK LOCAL ADDITION -- this whole module is local to javrasya/sam3 and has
no upstream counterpart (see Discern ADR 0002). In Discern's vocabulary what it
does is Re-grounding: asking a vision model to locate an Object afresh from its
Object Hint. The request/response contract lives in :mod:`sam3.re_grounding`;
window geometry lives in :mod:`sam3.zoom_anchor`; this module is the part that
needs images, a network and threads.
"""

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

import numpy as np
from PIL import Image

from sam3.agent.client_llm import send_vision_request
from sam3.agent.helpers.frame_renderer import mask_to_bbox, render_frame_with_masks
from sam3.re_grounding import (
    DEFAULT_RE_GROUNDING_CONCURRENCY,
    DEFAULT_RE_GROUNDING_MAX_TOKENS,
    DEFAULT_RE_GROUNDING_TIMEOUT_S,
    RE_GROUNDING_SYSTEM_PROMPT,
    ObjectReGrounding,
    ReGroundingResult,
    build_re_grounding_prompt,
    parse_re_grounding_response,
)

logger = logging.getLogger(__name__)

# Slack added to the per-request timeout before the batch as a whole is abandoned,
# covering rendering, base64 encoding and handing results back between threads.
_BATCH_GRACE_S = 2.0


# DISCERN FORK LOCAL ADDITION
def _to_pil(frame):
    """Convert a frame in any of the three shapes callers hold into a PIL image."""
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame.astype(np.uint8))
    if hasattr(frame, "cpu"):  # torch.Tensor
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0:
            arr = arr * 255
        return Image.fromarray(arr.astype(np.uint8))
    raise TypeError(f"cannot render a frame of type {type(frame)}")


# DISCERN FORK LOCAL ADDITION
def _frame_size(frame):
    """(width, height) of a frame, raising rather than guessing.

    The previous version returned ``None`` for an unknown type, which reached the
    caller as "the vision model found nothing" -- a programming error wearing a
    tracking result's clothes.
    """
    if isinstance(frame, np.ndarray):
        return frame.shape[1], frame.shape[0]
    if isinstance(frame, Image.Image):
        return frame.size
    if hasattr(frame, "shape"):  # torch.Tensor
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            return int(frame.shape[2]), int(frame.shape[1])
        return int(frame.shape[1]), int(frame.shape[0])
    raise TypeError(f"cannot measure a frame of type {type(frame)}")


class LLMCropAdvisor:
    """
    Handles LLM communication for predicting crop zones during video tracking.

    Uses a vision LLM to analyze previous frame masks and current frame appearance
    to predict where objects will be, enabling high-resolution cropped tracking.

    One Re-grounding request per Object, not one request for all of them: a shared
    request makes every Object compete for one response budget, and one malformed
    entry silently unguides an Object nobody was told about.
    """

    def __init__(
        self,
        server_url,
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_key=None,
        crop_padding=0.5,
        max_tokens=DEFAULT_RE_GROUNDING_MAX_TOKENS,
        request_timeout=DEFAULT_RE_GROUNDING_TIMEOUT_S,
        max_concurrent_requests=DEFAULT_RE_GROUNDING_CONCURRENCY,
        send_request=send_vision_request,
    ):
        """
        Args:
            server_url: OpenAI-compatible API endpoint URL
            model: Model name/ID for the API
            api_key: Optional API key
            crop_padding: Retained for callers that still construct the advisor
                with it. Nothing here pads anything: Re-grounding returns raw
                located boxes and :mod:`sam3.zoom_anchor` owns the padding.
            max_tokens: Max tokens for one Object's response
            request_timeout: Seconds one Object's request may take, and the budget
                for a whole frame's batch of them
            max_concurrent_requests: How many of a frame's requests are in flight
                at once
            send_request: The transport, injectable so the request contract can be
                tested without a provider
        """
        self.server_url = server_url
        self.model = model
        self.api_key = api_key
        self.crop_padding = crop_padding
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.send_request = send_request

    # DISCERN FORK LOCAL ADDITION
    def re_ground_objects(
        self,
        prev_frame,
        curr_frame,
        object_ids,
        prev_masks=None,
        prev_bboxes=None,
        object_hints=None,
    ):
        """Re-ground every Object on one frame: one request each, concurrently.

        Each request carries only its own Object -- an overlay of that Object's
        mask and box on the previous frame, the current frame, and that Object's
        Object Hint -- so adding a fourth Object cannot degrade the guidance for
        the first three.

        Args:
            prev_frame: Previous frame (PIL Image, numpy array, or torch Tensor)
            curr_frame: Current frame, same shapes accepted
            object_ids: The Objects to re-ground
            prev_masks: dict {obj_id: mask_array} from the previous frame. An
                Object may be missing or its mask empty; Re-grounding from the
                Object Hint alone is exactly how a lost Object comes back.
            prev_bboxes: dict {obj_id: (x1, y1, x2, y2)} in pixels, for the
                overlay. Derived from the mask when absent.
            object_hints: dict {obj_id: str} of Object Hints, or one str for all.
                Reaches the model per Object. Discern calls this
                ``object_descriptions`` on the wire and resolves the documented
                fallback to the Object's name before sending, so an Object missing
                here has neither a Hint nor a name.

        Returns:
            :class:`~sam3.re_grounding.ReGroundingResult` -- a
            ``Mapping[int, Optional[BBox]]`` ready to hand to
            :func:`sam3.zoom_anchor.resolve_zoom_anchors`, which also reports why
            any missing box is missing.

        Raises:
            ValueError: No Objects were asked for.
            TypeError: A frame is of a shape this module cannot read.
        """
        ordered_ids = list(dict.fromkeys(object_ids))
        if not ordered_ids:
            raise ValueError("re_ground_objects needs at least one Object")

        frame_width, frame_height = _frame_size(curr_frame)
        prev_pil = _to_pil(prev_frame)
        curr_pil = _to_pil(curr_frame)
        # Decode both once, here: the worker threads only ever read them, and PIL
        # would otherwise decode lazily inside several threads at the same time.
        prev_pil.load()
        curr_pil.load()
        hints = self._hints_by_object(object_hints, ordered_ids)

        per_object = {}
        pending = []
        workers = max(1, min(len(ordered_ids), self.max_concurrent_requests))
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="re-grounding"
        )
        try:
            for object_id in ordered_ids:
                bbox = self._overlay_bbox(object_id, prev_masks, prev_bboxes)
                hint = hints.get(object_id)
                if bbox is None and not hint:
                    # Nothing to point at and nothing to describe: a request would
                    # carry no way to tell this Object from anything else in the
                    # frame. Not asking is honest; calling it a provider failure
                    # would not be.
                    per_object[object_id] = ObjectReGrounding.not_asked(
                        object_id,
                        "no Object Hint and no mask on the previous frame",
                    )
                    continue
                mask = None if bbox is None else prev_masks.get(object_id)
                pending.append(
                    (
                        object_id,
                        executor.submit(
                            self._re_ground_one,
                            object_id,
                            prev_pil,
                            curr_pil,
                            mask,
                            bbox,
                            hint,
                            frame_width,
                            frame_height,
                        ),
                    )
                )

            # One wall-clock budget for the whole frame, so the time a
            # Re-grounding tick can cost does not grow with the Object count --
            # which is what keeps it inside Discern's 60s first-frame guard.
            deadline = time.monotonic() + self.request_timeout + _BATCH_GRACE_S
            for object_id, future in pending:
                try:
                    per_object[object_id] = future.result(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except FuturesTimeoutError:
                    per_object[object_id] = ObjectReGrounding.failed(
                        object_id,
                        "no answer within the frame's Re-grounding budget of "
                        f"{self.request_timeout + _BATCH_GRACE_S:.0f}s",
                    )
        finally:
            # Never wait: a straggler deletes its own temporary images and its
            # answer is already too late to be used.
            executor.shutdown(wait=False, cancel_futures=True)

        result = ReGroundingResult(per_object)
        self._log_outcomes(result)
        return result

    # DISCERN FORK LOCAL ADDITION
    def _re_ground_one(
        self,
        object_id,
        prev_pil,
        curr_pil,
        mask,
        bbox,
        hint,
        frame_width,
        frame_height,
    ):
        """One Object's request, start to finish, in its own thread."""
        temp_paths = []
        try:
            masks = {} if mask is None else {object_id: mask}
            bboxes = {} if bbox is None else {object_id: bbox}
            overlay = render_frame_with_masks(prev_pil, masks, bboxes)

            temp_paths.append(self._save_temp_image(overlay))
            temp_paths.append(self._save_temp_image(curr_pil))

            response_text = self.send_request(
                messages=self._build_messages(
                    object_id, hint, bbox is not None, *temp_paths
                ),
                server_url=self.server_url,
                model=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                timeout=self.request_timeout,
            )
            box = parse_re_grounding_response(response_text, frame_width, frame_height)
        except Exception as exc:
            # Not swallowed: it becomes this Object's visible FAILED outcome,
            # carrying the provider's own words, and is logged as an error.
            return ObjectReGrounding.failed(object_id, f"{type(exc).__name__}: {exc}")
        finally:
            for path in temp_paths:
                self._delete_temp_image(path)

        if box is None:
            return ObjectReGrounding.not_found(object_id)
        return ObjectReGrounding.located(object_id, box)

    # DISCERN FORK LOCAL ADDITION
    def _build_messages(self, object_id, hint, has_previous_mask, prev_path, curr_path):
        """One Object's OpenAI-compatible message pair."""
        return [
            {"role": "system", "content": RE_GROUNDING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_re_grounding_prompt(
                            object_id, hint, has_previous_mask
                        ),
                    },
                    {"type": "text", "text": "\nImage 1 (previous frame):\n"},
                    {"type": "image", "image": prev_path},
                    {"type": "text", "text": "\nImage 2 (current frame):\n"},
                    {"type": "image", "image": curr_path},
                ],
            },
        ]

    # DISCERN FORK LOCAL ADDITION
    @staticmethod
    def _hints_by_object(object_hints, object_ids):
        """Object Hints as {obj_id: str}, accepting one shared string as well."""
        if not object_hints:
            return {}
        if isinstance(object_hints, str):
            return {object_id: object_hints for object_id in object_ids}
        return dict(object_hints)

    # DISCERN FORK LOCAL ADDITION
    @staticmethod
    def _overlay_bbox(object_id, prev_masks, prev_bboxes):
        """The box to draw for this Object, or None if it has no mask to draw.

        An empty mask is the same as no mask here: there is nothing to highlight.
        """
        if not prev_masks or object_id not in prev_masks:
            return None
        if prev_bboxes and prev_bboxes.get(object_id) is not None:
            return prev_bboxes[object_id]
        return mask_to_bbox(prev_masks[object_id])

    # DISCERN FORK LOCAL ADDITION
    @staticmethod
    def _log_outcomes(result):
        """Make the frame's Re-grounding visible in normal operation."""
        for object_id, reason in sorted(result.failures.items()):
            logger.error("Re-grounding failed for obj_%s: %s", object_id, reason)
        for object_id, entry in sorted(result.per_object.items()):
            if entry.reason and object_id not in result.failures:
                logger.warning(
                    "Re-grounding skipped for obj_%s: %s", object_id, entry.reason
                )
        logger.info(
            "Re-grounding: %d located, %d absent, %d failed of %d Object(s)",
            len(result.located),
            len(result.absent),
            len(result.failures),
            len(result.per_object),
        )

    # DISCERN FORK LOCAL ADDITION
    @staticmethod
    def _delete_temp_image(path):
        """Delete one temporary request image, on every path including failure."""
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("could not delete temporary image %s: %s", path, exc)

    @staticmethod
    def _save_temp_image(pil_image):
        """Save PIL image to a temporary file and return the path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        if isinstance(pil_image, Image.Image):
            pil_image.save(tmp, format="JPEG", quality=85)
        else:
            Image.fromarray(np.array(pil_image)).save(tmp, format="JPEG", quality=85)
        tmp.close()
        return tmp.name


def interpolate_crop_zones(prev_zones, next_zones, alpha):
    """
    Linearly interpolate between two sets of crop zones.

    Args:
        prev_zones: dict {obj_id: (x1, y1, x2, y2)} from last LLM call
        next_zones: dict {obj_id: (x1, y1, x2, y2)} from next LLM call (or None)
        alpha: interpolation factor in [0, 1] (0 = prev, 1 = next)

    Returns:
        dict {obj_id: (x1, y1, x2, y2)} interpolated crop zones
    """
    if next_zones is None or alpha <= 0:
        return prev_zones
    if alpha >= 1:
        return next_zones

    result = {}
    for obj_id in prev_zones:
        if obj_id in next_zones:
            p = prev_zones[obj_id]
            n = next_zones[obj_id]
            result[obj_id] = tuple(
                int(p[i] * (1 - alpha) + n[i] * alpha) for i in range(4)
            )
        else:
            result[obj_id] = prev_zones[obj_id]

    # Include any objects only in next_zones
    for obj_id in next_zones:
        if obj_id not in result:
            result[obj_id] = next_zones[obj_id]

    return result
