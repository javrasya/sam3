# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Frame rendering utilities for LLM-guided crop propagation.

Renders frames with mask overlays and bounding boxes for LLM visual input.
"""

import numpy as np
from PIL import Image, ImageDraw


def render_frame_with_masks(
    frame,
    masks,
    bboxes=None,
    obj_ids=None,
    mask_alpha=0.4,
    colors=None,
):
    """
    Render a frame with colored semi-transparent mask overlays and bounding boxes.

    Args:
        frame: PIL.Image, numpy array (H, W, 3) RGB, or torch.Tensor
        masks: dict {obj_id: mask} where mask is numpy [H, W] bool/float or torch.Tensor
        bboxes: optional dict {obj_id: [x1, y1, x2, y2]} in pixel coords
        obj_ids: optional list of obj_ids to render (defaults to all keys in masks)
        mask_alpha: transparency for mask overlay (0=transparent, 1=opaque)
        colors: optional dict {obj_id: (R, G, B)} with 0-255 values

    Returns:
        PIL.Image with overlays rendered
    """
    # Convert frame to PIL
    if isinstance(frame, np.ndarray):
        pil_frame = Image.fromarray(frame.astype(np.uint8))
    elif hasattr(frame, "cpu"):  # torch.Tensor
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        pil_frame = Image.fromarray(arr.astype(np.uint8))
    elif isinstance(frame, Image.Image):
        pil_frame = frame.copy()
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    pil_frame = pil_frame.convert("RGB")

    if obj_ids is None:
        obj_ids = sorted(masks.keys())

    # Default color palette
    _DEFAULT_COLORS = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 128, 0),  # orange
        (128, 0, 255),  # purple
    ]

    for i, obj_id in enumerate(obj_ids):
        if obj_id not in masks:
            continue

        mask = masks[obj_id]
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = mask.squeeze()
        mask_bool = mask > 0.5

        if not mask_bool.any():
            continue

        # Get color
        if colors and obj_id in colors:
            color = colors[obj_id]
        else:
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

        # Create colored overlay
        overlay = np.array(pil_frame, dtype=np.float32)
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - mask_alpha)
            + np.array(color, dtype=np.float32) * mask_alpha
        )
        pil_frame = Image.fromarray(overlay.astype(np.uint8))

    # Draw bboxes and labels
    draw = ImageDraw.Draw(pil_frame)
    for i, obj_id in enumerate(obj_ids):
        if colors and obj_id in colors:
            color = colors[obj_id]
        else:
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]

        if bboxes and obj_id in bboxes:
            box = bboxes[obj_id]
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, y1 + 2), f"obj_{obj_id}", fill=color)

    return pil_frame


def mask_to_bbox(mask):
    """
    Compute pixel bounding box from a binary mask.

    Args:
        mask: numpy array [H, W] or torch.Tensor [H, W]

    Returns:
        (x1, y1, x2, y2) in pixel coords, or None if mask is empty
    """
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = mask.squeeze()
    coords = np.nonzero(mask > 0.5)
    if len(coords[0]) == 0:
        return None
    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()
    return (int(x1), int(y1), int(x2), int(y2))
