# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
LLM-guided crop zone prediction for video tracking.

Uses a vision LLM to predict per-frame crop zones during propagation,
so the SAM3 tracker always sees small objects at high effective resolution.
"""

import io
import json
import logging
import tempfile

import numpy as np
from PIL import Image

from sam3.agent.client_llm import send_generate_request
from sam3.agent.helpers.frame_renderer import mask_to_bbox, render_frame_with_masks

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a visual object tracking assistant. Your task is to predict bounding crop \
zones for objects being tracked in a video sequence.

You will receive two images:
1. **Previous frame** with colored mask overlays and bounding boxes showing where \
each tracked object was in the previous frame.
2. **Current frame** (no annotations) where you need to predict where each object \
has moved.

For each tracked object, predict a crop zone in the current frame that:
- Centers on where you expect the object to be
- Is large enough to contain the object with some margin
- Accounts for motion between frames

Respond with ONLY valid JSON in this exact format:
{"objects": [{"id": <obj_id>, "crop": [x1, y1, x2, y2]}]}

Where x1, y1, x2, y2 are normalized coordinates in [0, 1] range:
- x1, y1: top-left corner
- x2, y2: bottom-right corner

IMPORTANT: Return coordinates for ALL tracked objects. Be generous with crop size \
to avoid cutting off the object. If unsure about object location, use a larger crop zone."""


class LLMCropAdvisor:
    """
    Handles LLM communication for predicting crop zones during video tracking.

    Uses a vision LLM to analyze previous frame masks and current frame appearance
    to predict where objects will be, enabling high-resolution cropped tracking.
    """

    def __init__(
        self,
        server_url,
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_key=None,
        crop_padding=0.5,
        max_tokens=512,
    ):
        """
        Args:
            server_url: OpenAI-compatible API endpoint URL
            model: Model name/ID for the API
            api_key: Optional API key
            crop_padding: Default padding factor around predicted crops
            max_tokens: Max tokens for LLM response
        """
        self.server_url = server_url
        self.model = model
        self.api_key = api_key
        self.crop_padding = crop_padding
        self.max_tokens = max_tokens

    def predict_crop_zones(
        self,
        prev_frame,
        prev_masks,
        prev_bboxes,
        curr_frame,
        object_descriptions=None,
    ):
        """
        Predict crop zones for all tracked objects using one LLM call.

        Args:
            prev_frame: Previous frame (PIL Image, numpy array, or torch Tensor)
            prev_masks: dict {obj_id: mask_array} from previous frame
            prev_bboxes: dict {obj_id: (x1, y1, x2, y2)} in pixel coords
            curr_frame: Current frame (PIL Image, numpy array, or torch Tensor)
            object_descriptions: optional dict {obj_id: str} or single str

        Returns:
            dict {obj_id: (x1, y1, x2, y2)} crop zones in pixel coordinates,
            or None if LLM call fails entirely
        """
        # Get frame dimensions
        if isinstance(curr_frame, np.ndarray):
            orig_h, orig_w = curr_frame.shape[:2]
        elif isinstance(curr_frame, Image.Image):
            orig_w, orig_h = curr_frame.size
        elif hasattr(curr_frame, "shape"):  # torch.Tensor
            if curr_frame.ndim == 3 and curr_frame.shape[0] in (1, 3):
                orig_h, orig_w = curr_frame.shape[1], curr_frame.shape[2]
            else:
                orig_h, orig_w = curr_frame.shape[:2]
        else:
            logger.warning(f"Unknown frame type {type(curr_frame)}, cannot get dimensions")
            return None

        # Render previous frame with mask overlays
        prev_rendered = self._render_frame_with_masks(prev_frame, prev_masks, prev_bboxes)

        # Build messages
        messages = self._build_messages(
            prev_rendered, curr_frame, object_descriptions, list(prev_masks.keys())
        )

        # Call LLM
        try:
            response_text = send_generate_request(
                messages=messages,
                server_url=self.server_url,
                model=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            return None

        if response_text is None:
            logger.warning("LLM returned None response")
            return None

        # Parse response
        crop_zones = self._parse_response(response_text, orig_h, orig_w)
        if crop_zones is None:
            logger.warning(f"Failed to parse LLM response: {response_text[:200]}")
            return None

        # Apply padding to predicted crops
        padded_zones = {}
        for obj_id, (x1, y1, x2, y2) in crop_zones.items():
            padded_zones[obj_id] = self._apply_padding(
                x1, y1, x2, y2, orig_h, orig_w, self.crop_padding
            )

        return padded_zones

    def _render_frame_with_masks(self, frame, masks, bboxes):
        """Render frame with colored mask overlays and bbox annotations."""
        return render_frame_with_masks(frame, masks, bboxes)

    def _build_messages(self, prev_rendered, curr_frame, object_descriptions, obj_ids):
        """Build OpenAI-compatible message list with images."""
        # Save images to temp files for the API
        prev_path = self._save_temp_image(prev_rendered)

        # Convert curr_frame to PIL if needed
        if isinstance(curr_frame, np.ndarray):
            curr_pil = Image.fromarray(curr_frame.astype(np.uint8))
        elif hasattr(curr_frame, "cpu"):
            arr = curr_frame.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            curr_pil = Image.fromarray(arr.astype(np.uint8))
        elif isinstance(curr_frame, Image.Image):
            curr_pil = curr_frame
        else:
            curr_pil = curr_frame

        curr_path = self._save_temp_image(curr_pil)

        # Build description text
        desc_text = ""
        if object_descriptions:
            if isinstance(object_descriptions, str):
                desc_text = f"\nObject descriptions: {object_descriptions}"
            elif isinstance(object_descriptions, dict):
                descs = [f"obj_{k}: {v}" for k, v in object_descriptions.items()]
                desc_text = "\nObject descriptions: " + "; ".join(descs)

        user_text = (
            f"I'm tracking {len(obj_ids)} object(s) with IDs: {obj_ids}.{desc_text}\n\n"
            "Image 1 (previous frame with mask overlays and bounding boxes):\n"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": prev_path},
                    {"type": "text", "text": "\nImage 2 (current frame, predict crop zones for this frame):\n"},
                    {"type": "image", "image": curr_path},
                    {
                        "type": "text",
                        "text": "\nPredict the crop zones as JSON.",
                    },
                ],
            },
        ]
        return messages

    def _parse_response(self, response_text, orig_h, orig_w):
        """Parse LLM JSON response into pixel crop zones."""
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        json_str = text[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        objects = data.get("objects", [])
        if not objects:
            return None

        crop_zones = {}
        for obj in objects:
            obj_id = obj.get("id")
            crop = obj.get("crop")
            if obj_id is None or crop is None or len(crop) != 4:
                continue

            # Convert normalized [0,1] coords to pixel coords
            x1 = int(crop[0] * orig_w)
            y1 = int(crop[1] * orig_h)
            x2 = int(crop[2] * orig_w)
            y2 = int(crop[3] * orig_h)

            # Clamp to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # Ensure valid box (min size)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop_zones[obj_id] = (x1, y1, x2, y2)

        return crop_zones if crop_zones else None

    def fallback_crop(self, prev_bbox, orig_h, orig_w, padding=None):
        """
        Fallback: expand previous bbox with padding when LLM fails.

        Args:
            prev_bbox: (x1, y1, x2, y2) in pixel coords from previous frame
            orig_h: Original image height
            orig_w: Original image width
            padding: Padding factor (defaults to self.crop_padding)

        Returns:
            (x1, y1, x2, y2) padded crop zone in pixel coords
        """
        if padding is None:
            padding = self.crop_padding

        x1, y1, x2, y2 = prev_bbox
        return self._apply_padding(x1, y1, x2, y2, orig_h, orig_w, padding)

    @staticmethod
    def _apply_padding(x1, y1, x2, y2, orig_h, orig_w, padding):
        """Apply padding to a bounding box and clamp to image bounds."""
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(orig_w, x2 + pad_w)
        y2 = min(orig_h, y2 + pad_h)

        return (x1, y1, x2, y2)

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
