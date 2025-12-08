# Sam3UnifiedPredictor Usage Guide

## Overview

`Sam3UnifiedPredictor` provides a unified interface for both high-precision DETR detection and temporal video tracking in a single, memory-efficient model.

## Installation & Import

```python
from sam3 import build_sam3_unified_predictor

# Build the unified predictor
predictor = build_sam3_unified_predictor(
    checkpoint_path=None,           # None = load from HuggingFace
    device="cuda",                  # "cuda" or "cpu"
    detection_confidence_threshold=0.5,  # Confidence threshold for detection
)
```

## Use Case 1: High-Precision Single Frame Detection

Use `detect_frame()` for DETR-based high-precision segmentation on single images:

```python
from PIL import Image

# Load image
image = Image.open("photo.jpg")

# Detect with text prompt
result = predictor.detect_frame(
    image=image,
    text="person wearing red shirt",
    confidence_threshold=0.5
)

# Access results
masks = result["masks"]          # Binary masks [N, H, W]
boxes = result["boxes"]          # Bounding boxes [N, 4] in xyxy
scores = result["scores"]        # Confidence scores [N]
mask_logits = result["masks_logits"]  # Mask logits [N, 1, H, W]
```

### Detection with Geometric Prompts

```python
# With box prompts (normalized [0,1] coordinates: [cx, cy, w, h])
result = predictor.detect_frame(
    image=image,
    text="dog",
    boxes=[[0.5, 0.5, 0.3, 0.4]],  # Center at (0.5, 0.5), size 0.3x0.4
    box_labels=[True],              # True = positive prompt
)

# With point prompts (normalized [0,1] coordinates: [x, y])
result = predictor.detect_frame(
    image=image,
    text="cat",
    points=[[0.3, 0.4], [0.7, 0.6]],
    point_labels=[1, 1],  # 1 = foreground, 0 = background
)

# Geometry only (no text)
result = predictor.detect_frame(
    image=image,
    boxes=[[0.5, 0.5, 0.2, 0.3]],
    box_labels=[True],
)
```

### Batch Detection

Process multiple frames efficiently:

```python
images = [Image.open(f"frame_{i}.jpg") for i in range(10)]

results = predictor.detect_frame_batch(
    images=images,
    text="bicycle",
    confidence_threshold=0.6
)

for i, result in enumerate(results):
    print(f"Frame {i}: {len(result['masks'])} objects detected")
```

## Use Case 2: Video Object Tracking

Use inherited video tracking methods for temporal consistency:

```python
# Start a video session
session_id = predictor.start_session(
    video_path="video.mp4",
    # or provide frames directly:
    # frames=list_of_frames,
)

# Add text prompt
predictor.add_text_prompt(
    session_id=session_id,
    text="person in blue jacket",
    frame_idx=0,  # Initialize on first frame
)

# Or add geometric prompts
predictor.add_point_prompt(
    session_id=session_id,
    frame_idx=0,
    points=[[0.5, 0.5]],
    labels=[1],
    object_id=1,
)

# Propagate through video
for result in predictor.propagate_in_video(
    session_id=session_id,
    propagation_direction="forward",  # "forward", "backward", or "both"
):
    frame_idx = result["frame_index"]
    outputs = result["outputs"]

    masks = outputs["out_binary_masks"]  # [N, H, W]
    boxes = outputs["out_boxes_xywh"]    # [N, 4] normalized xywh
    scores = outputs.get("out_tracker_scores", [])

    print(f"Frame {frame_idx}: {len(masks)} tracked objects")
```

## Use Case 3: Hybrid Workflow (Propagate + Refine)

Combine tracker speed with detector precision:

```python
# Start session and add prompts
session_id = predictor.start_session("video.mp4")
predictor.add_text_prompt(session_id, "car", frame_idx=0)

# Propagate with tracker, then refine low-confidence frames
for result in predictor.propagate_and_refine(
    session_id=session_id,
    refinement_threshold=0.7,  # Refine frames where tracker score < 0.7
    refine_every_n=10,         # Also refine every 10th frame
    propagation_direction="both",
):
    frame_idx = result["frame_index"]
    outputs = result["outputs"]

    is_refined = outputs.get("refined", False)
    print(f"Frame {frame_idx}: {'REFINED' if is_refined else 'tracked'}")
```

### Manual Frame Refinement

Refine specific frames after propagation:

```python
# Refine a specific frame with high precision
refined_result = predictor.refine_frame(
    session_id=session_id,
    frame_idx=42,
    text_prompt="dog",  # Override session prompt if needed
    use_propagated_boxes=True,  # Use propagated masks as guidance
)

masks = refined_result["masks"]
boxes = refined_result["boxes"]
scores = refined_result["scores"]
```

## Advanced Options

### Custom Confidence Threshold per Call

```python
# Override default threshold
result = predictor.detect_frame(
    image=image,
    text="person",
    confidence_threshold=0.8,  # Higher threshold = fewer, more confident results
)
```

### Multi-GPU Support

```python
# Use specific GPUs
predictor = build_sam3_unified_predictor(
    device="cuda",
    gpus_to_use=[0, 1, 2],  # Use GPUs 0, 1, and 2
)
```

### Text Embedding Cache

Text embeddings are automatically cached for performance:

```python
# First call: embeds "person"
result1 = predictor.detect_frame(image1, text="person")

# Second call: uses cached embedding
result2 = predictor.detect_frame(image2, text="person")

# Clear cache if needed
predictor.clear_text_cache()
```

## Coordinate Systems

- **Boxes**: `[cx, cy, w, h]` normalized to [0, 1]
  - `cx, cy`: center coordinates
  - `w, h`: width and height

- **Points**: `[x, y]` normalized to [0, 1]
  - `x`: horizontal position (0=left, 1=right)
  - `y`: vertical position (0=top, 1=bottom)

- **Output boxes from detect_frame()**: `[x1, y1, x2, y2]` in pixel coordinates
  - `x1, y1`: top-left corner
  - `x2, y2`: bottom-right corner

## Memory Efficiency

The unified predictor is more memory-efficient than using separate models:

| Approach | Memory Usage | Use Case |
|----------|--------------|----------|
| `Sam3Processor` + `Sam3VideoPredictor` | ~2x | Separate models |
| `Sam3UnifiedPredictor` | ~1x | Shared backbone |

## Performance Tips

1. **Batch processing**: Use `detect_frame_batch()` for multiple frames
2. **Confidence threshold**: Higher threshold = faster (fewer objects to process)
3. **Text cache**: Reuse the same text prompts across frames
4. **Hybrid workflow**: Use tracker for most frames, refine only low-confidence ones
5. **Multi-GPU**: Enable for large videos or high-resolution frames

## API Reference Summary

### Detection Methods
- `detect_frame(image, text=None, boxes=None, points=None, confidence_threshold=None)` → Dict
- `detect_frame_batch(images, text=None, confidence_threshold=None)` → List[Dict]

### Video Tracking Methods (inherited)
- `start_session(video_path=None, frames=None)` → str
- `add_text_prompt(session_id, text, frame_idx)` → None
- `add_point_prompt(session_id, frame_idx, points, labels, object_id)` → None
- `add_box_prompt(session_id, frame_idx, boxes, object_id)` → None
- `propagate_in_video(session_id, propagation_direction="both")` → Iterator

### Hybrid Methods
- `propagate_and_refine(session_id, refinement_threshold=0.7, refine_every_n=None)` → Iterator
- `refine_frame(session_id, frame_idx, text_prompt=None)` → Dict

### Utility Methods
- `clear_text_cache()` → None
