# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Sam3UnifiedProcessor: A unified interface for both high-precision detection
and temporal propagation using a single model.

Design Philosophy:
- Stateless: All state is passed explicitly via state dict
- Composable: Functions can be chained, state flows through
- Consistent: Mirrors Sam3Processor API pattern

This class exposes both:
- Detection path: DETR-based high-precision segmentation
- Propagation path: Tracker-based temporal propagation
- Hybrid path: Propagate then refine with mask guidance

Memory efficient: The backbone is loaded once and shared between both paths.
"""

import time
from typing import Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2
from tqdm.auto import tqdm

from sam3.logger import get_logger
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import FindStage, interpolate
from sam3.model.geometry_encoders import Prompt

logger = get_logger(__name__)


class Sam3UnifiedProcessor:
    """
    Unified predictor providing both high-precision detection and
    temporal propagation from a single model.

    This follows a STATELESS design pattern (like Sam3Processor):
    - All state is passed explicitly as a dict
    - Functions take state in, return state out
    - No hidden session management
    - User controls state lifetime

    Example usage:
    ```python
    predictor = Sam3UnifiedProcessor(model)

    # ═══ Detection Path ═══
    state = predictor.set_image(image)
    state = predictor.set_text_prompt("person", state)
    state = predictor.add_point_prompt([0.5, 0.5], label=1, state=state)
    masks = state["masks"]

    # ═══ Video Path ═══
    state = predictor.set_video(video_path)
    state = predictor.add_prompt_on_frame(0, text="dog", state=state)
    state = predictor.propagate(state)
    state = predictor.refine_all_frames(state)
    ```
    """

    def __init__(
        self,
        model,
        detection_confidence_threshold: float = 0.5,
        resolution: int = 1008,
    ):
        """
        Initialize the unified predictor.

        Args:
            model: The underlying Sam3VideoInferenceWithInstanceInteractivity model
            detection_confidence_threshold: Default confidence threshold for detection
            resolution: Model input resolution (default 1008)
        """
        self.model = model
        self.detection_confidence_threshold = detection_confidence_threshold
        self.resolution = resolution

        # Image transform
        self._transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Text embedding cache (stateless - just for performance)
        self._text_cache: Dict[str, Dict] = {}
        self._text_cache_size = 100

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ═══════════════════════════════════════════════════════════════════════
    # DETECTION PATH - Single Frame, High Precision
    # Follows Sam3Processor pattern: set_image → set_text → add_prompt → get masks
    # ═══════════════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def set_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, np.ndarray],
        state: Optional[Dict] = None,
    ) -> Dict:
        """
        Set the image for detection. First step in detection pipeline.

        Args:
            image: Input image (PIL, tensor, or numpy array)
            state: Optional existing state to update

        Returns:
            State dict with backbone features computed
        """
        if state is None:
            state = {}

        # Get original dimensions
        if isinstance(image, PIL.Image.Image):
            orig_w, orig_h = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            orig_h, orig_w = image.shape[-2:]
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Transform image
        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self._transform(image_tensor).unsqueeze(0)

        # Compute backbone features
        state["original_height"] = orig_h
        state["original_width"] = orig_w
        state["image_tensor"] = image_tensor
        state["backbone_out"] = self.model.detector.backbone.forward_image(image_tensor)

        # Clear any previous prompts/results
        state.pop("geometric_prompt", None)
        state.pop("masks", None)
        state.pop("boxes", None)
        state.pop("scores", None)

        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict) -> Dict:
        """
        Set text prompt and run inference.

        Args:
            prompt: Text description of object to segment
            state: State from set_image()

        Returns:
            Updated state with masks, boxes, scores
        """
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before set_text_prompt")

        text_out = self._get_text_embeddings(prompt)
        state["backbone_out"].update(text_out)
        state["text_prompt"] = prompt

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model.detector._get_dummy_prompt()

        return self._forward_detection(state)

    @torch.inference_mode()
    def add_box_prompt(
        self,
        box: List[float],
        label: bool,
        state: Dict,
    ) -> Dict:
        """
        Add a box prompt and run inference.

        Args:
            box: [center_x, center_y, width, height] normalized in [0, 1]
            label: True for positive, False for negative
            state: State from set_image()

        Returns:
            Updated state with masks, boxes, scores
        """
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before add_box_prompt")

        # Ensure we have text features (use "visual" if no text prompt)
        if "language_features" not in state["backbone_out"]:
            text_out = self._get_text_embeddings("visual")
            state["backbone_out"].update(text_out)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model.detector._get_dummy_prompt()

        # Add box
        box_t = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        label_t = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(box_t, label_t)

        return self._forward_detection(state)

    @torch.inference_mode()
    def add_point_prompt(
        self,
        point: List[float],
        label: int,
        state: Dict,
    ) -> Dict:
        """
        Add a point prompt and run inference.

        Args:
            point: [x, y] normalized in [0, 1]
            label: 1 for foreground, 0 for background
            state: State from set_image()

        Returns:
            Updated state with masks, boxes, scores
        """
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before add_point_prompt")

        # Ensure we have text features
        if "language_features" not in state["backbone_out"]:
            text_out = self._get_text_embeddings("visual")
            state["backbone_out"].update(text_out)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model.detector._get_dummy_prompt()

        # Add point
        pt_t = torch.tensor(point, device=self.device, dtype=torch.float32).view(1, 1, 2)
        label_t = torch.tensor([label], device=self.device, dtype=torch.long).view(1, 1)
        state["geometric_prompt"].append_points(pt_t, label_t)

        return self._forward_detection(state)

    @torch.inference_mode()
    def add_mask_prompt(
        self,
        mask: Union[torch.Tensor, np.ndarray],
        state: Dict,
    ) -> Dict:
        """
        Add a mask as guidance and run inference.

        The mask conditions the detector, providing strong spatial guidance.
        Useful for refining a rough mask or using a user-drawn polygon.

        Args:
            mask: Binary mask [H, W] or [1, H, W]
            state: State from set_image()

        Returns:
            Updated state with refined masks
        """
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before add_mask_prompt")

        # Ensure we have text features
        if "language_features" not in state["backbone_out"]:
            text_out = self._get_text_embeddings("visual")
            state["backbone_out"].update(text_out)

        # Prepare mask
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.to(device=self.device, dtype=torch.float32)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)

        # Resize to model resolution
        mask = F.interpolate(mask, size=(self.resolution, self.resolution),
                            mode="bilinear", align_corners=False)

        # Store mask guidance in state
        state["mask_guidance"] = mask

        # Also derive box from mask
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model.detector._get_dummy_prompt()

        mask_squeezed = mask.squeeze()
        if mask_squeezed.numel() > 0 and mask_squeezed.any():
            boxes = self._mask_to_boxes_cxcywh(mask_squeezed)
        else:
            boxes = []
        for box in boxes:
            box_t = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
            label_t = torch.tensor([True], device=self.device, dtype=torch.bool).view(1, 1)
            state["geometric_prompt"].append_boxes(box_t, label_t)

        return self._forward_detection_with_mask(state)

    @torch.inference_mode()
    def set_all_prompts(
        self,
        state: Dict,
        text: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[bool]] = None,
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict:
        """
        Set all prompts at once and run inference only once.

        More efficient than calling individual prompt methods when you have
        multiple prompts, as it avoids repeated forward passes.

        Args:
            state: State from set_image()
            text: Text prompt string
            points: List of [x, y] points normalized [0, 1]
            point_labels: Labels for points (1=foreground, 0=background)
            boxes: List of [cx, cy, w, h] boxes normalized [0, 1]
            box_labels: Labels for boxes (True=positive, False=negative)
            mask: Optional mask for guidance

        Returns:
            Updated state with inference results
        """
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before set_all_prompts")

        # Reset existing prompts
        state = self.reset_prompts(state)

        has_points = points is not None and len(points) > 0
        has_boxes = boxes is not None and len(boxes) > 0
        has_mask = mask is not None
        has_geometric = has_points or has_boxes or has_mask

        # Set text (or "visual" if only geometric prompts)
        if text is not None:
            text_out = self._get_text_embeddings(text)
            state["backbone_out"].update(text_out)
            state["text_prompt"] = text
        elif has_geometric:
            text_out = self._get_text_embeddings("visual")
            state["backbone_out"].update(text_out)

        # Initialize geometric prompt
        state["geometric_prompt"] = self.model.detector._get_dummy_prompt()

        # Add all points
        if has_points:
            if point_labels is None:
                point_labels = [1] * len(points)
            for pt, lbl in zip(points, point_labels):
                pt_t = torch.tensor(pt, device=self.device, dtype=torch.float32).view(1, 1, 2)
                lbl_t = torch.tensor([lbl], device=self.device, dtype=torch.long).view(1, 1)
                state["geometric_prompt"].append_points(pt_t, lbl_t)

        # Add all boxes
        if has_boxes:
            if box_labels is None:
                box_labels = [True] * len(boxes)
            for box, lbl in zip(boxes, box_labels):
                box_t = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
                lbl_t = torch.tensor([lbl], device=self.device, dtype=torch.bool).view(1, 1)
                state["geometric_prompt"].append_boxes(box_t, lbl_t)

        # Add mask guidance
        if has_mask:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(device=self.device, dtype=torch.float32)

            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)

            mask = F.interpolate(mask, size=(self.resolution, self.resolution),
                                mode="bilinear", align_corners=False)
            state["mask_guidance"] = mask

            # Also derive box from mask
            mask_squeezed = mask.squeeze()
            if mask_squeezed.numel() > 0 and mask_squeezed.any():
                mask_boxes = self._mask_to_boxes_cxcywh(mask_squeezed)
            else:
                mask_boxes = []
            for box in mask_boxes:
                box_t = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
                lbl_t = torch.tensor([True], device=self.device, dtype=torch.bool).view(1, 1)
                state["geometric_prompt"].append_boxes(box_t, lbl_t)

        # Run inference once
        if has_mask:
            return self._forward_detection_with_mask(state)
        elif text is not None or has_geometric:
            return self._forward_detection(state)

        return state

    def reset_prompts(self, state: Dict) -> Dict:
        """
        Reset all prompts while keeping image features.

        Args:
            state: Current state

        Returns:
            State with prompts cleared
        """
        if "backbone_out" in state:
            # Remove text features
            for key in ["language_features", "language_mask", "language_embeds"]:
                state["backbone_out"].pop(key, None)

        # Remove prompts and results
        for key in ["geometric_prompt", "mask_guidance", "masks", "masks_logits",
                    "boxes", "scores", "text_prompt"]:
            state.pop(key, None)

        return state

    def set_confidence_threshold(self, threshold: float, state: Dict) -> Dict:
        """Update confidence threshold and re-filter results."""
        self.detection_confidence_threshold = threshold
        if "masks" in state:
            return self._forward_detection(state)
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # VIDEO PATH - Multi-frame propagation
    # ═══════════════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def set_video(
        self,
        video_source: Union[str, List],
        state: Optional[Dict] = None,
    ) -> Dict:
        """
        Set video for tracking/propagation.

        Args:
            video_source: Path to video file or list of frame tensors/images
            state: Optional existing state

        Returns:
            State with video loaded
        """
        if state is None:
            state = {}

        # Store video reference
        state["video_source"] = video_source
        state["frame_prompts"] = {}  # frame_idx -> prompt info
        state["frame_masks"] = {}    # frame_idx -> masks from detection
        state["propagated_masks"] = {}  # frame_idx -> masks from propagation
        state["propagation_done"] = False

        # Get video info
        if isinstance(video_source, str):
            import cv2
            cap = cv2.VideoCapture(video_source)
            state["num_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            state["orig_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            state["orig_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        elif isinstance(video_source, list):
            state["num_frames"] = len(video_source)
            # Get dims from first frame (expects PIL.Image, tensor, or ndarray - same as set_image)
            first = video_source[0]
            if isinstance(first, PIL.Image.Image):
                state["orig_width"], state["orig_height"] = first.size
            else:
                # tensor or ndarray
                state["orig_height"], state["orig_width"] = first.shape[-2:]

        return state

    def set_seed_masks(
        self,
        masks_by_frame: Dict[int, Dict[int, Union[torch.Tensor, np.ndarray]]],
        state: Dict,
        text_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Directly inject pre-computed masks as seeds for propagation.

        Use this when you already have high-quality masks from an interactive
        session and want to propagate + refine without re-running detection.

        Args:
            masks_by_frame: Dict mapping frame_idx -> {obj_id -> mask}
                Example: {
                    0: {1: mask_obj1_f0, 2: mask_obj2_f0},
                    50: {1: mask_obj1_f50},
                }
            state: State from set_video()
            text_prompt: Optional text prompt for later refinement

        Returns:
            State with seed masks ready for propagation

        Example:
            >>> state = processor.set_video(video_path)
            >>> state = processor.set_seed_masks({
            ...     0: {1: mask1, 2: mask2},
            ...     50: {1: mask1_corrected},
            ... }, state, text_prompt="person")
            >>> state = processor.propagate(state)
            >>> state = processor.refine_all_frames(state, refine_every_n=1)
        """
        if "video_source" not in state:
            raise ValueError("Must call set_video before set_seed_masks")

        # Store text prompt for refinement
        if text_prompt:
            state["text_prompt"] = text_prompt

        # Initialize seed_masks dict if not present
        if "seed_masks" not in state:
            state["seed_masks"] = {}

        # Store masks in seed_masks for direct tracker propagation
        for frame_idx, obj_masks in masks_by_frame.items():
            state["seed_masks"][frame_idx] = {}
            for obj_id, mask in obj_masks.items():
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                mask = mask.to(dtype=torch.float32)

                # Ensure 2D
                if mask.dim() == 3:
                    mask = mask.squeeze(0)

                state["seed_masks"][frame_idx][obj_id] = mask

        # Mark propagation as needing to run
        state["propagation_done"] = False

        return state

    @torch.inference_mode()
    def add_prompt_on_frame(
        self,
        frame_idx: int,
        state: Dict,
        text: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[bool]] = None,
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict:
        """
        Add prompts on a specific frame and run detection.

        This performs high-precision DETR detection on the frame,
        storing results for later propagation.

        Args:
            frame_idx: Frame index
            state: State from set_video()
            text: Text prompt
            points: List of [x, y] points normalized [0,1]
            point_labels: Labels for points (1=fg, 0=bg)
            boxes: List of [cx, cy, w, h] boxes normalized [0,1]
            box_labels: Labels for boxes (True=pos, False=neg)
            mask: Optional mask guidance

        Returns:
            Updated state with detection results for this frame
        """
        if "video_source" not in state:
            raise ValueError("Must call set_video before add_prompt_on_frame")

        # Get the frame
        frame = self._get_frame(state, frame_idx)

        # Run detection pipeline
        det_state = self.set_image(frame)

        if text:
            det_state = self.set_text_prompt(text, det_state)
            state["text_prompt"] = text  # Store for propagation

        if mask is not None:
            det_state = self.add_mask_prompt(mask, det_state)

        if points:
            if point_labels is None:
                point_labels = [1] * len(points)
            for pt, lbl in zip(points, point_labels):
                det_state = self.add_point_prompt(pt, lbl, det_state)

        if boxes:
            if box_labels is None:
                box_labels = [True] * len(boxes)
            for box, lbl in zip(boxes, box_labels):
                det_state = self.add_box_prompt(box, lbl, det_state)

        # Store results
        state["frame_prompts"][frame_idx] = {
            "text": text,
            "points": points,
            "point_labels": point_labels,
            "boxes": boxes,
            "box_labels": box_labels,
            "mask": mask,
        }

        if "masks" in det_state:
            state["frame_masks"][frame_idx] = {
                "masks": det_state["masks"].cpu(),
                "boxes": det_state["boxes"].cpu(),
                "scores": det_state["scores"].cpu(),
            }

        # Mark propagation as needing re-run
        state["propagation_done"] = False

        return state

    @torch.inference_mode()
    def propagate(
        self,
        state: Dict,
        direction: str = "both",
        stream: bool = False,
    ):
        """
        Propagate masks through video using tracker.

        Uses masks from add_prompt_on_frame() or set_seed_masks() as conditioning frames.

        Args:
            state: State from add_prompt_on_frame() or set_seed_masks()
            direction: "both", "forward", or "backward"
            stream: If True, yields (frame_idx, masks_dict, state) for each frame.
                   If False (default), collects all results and returns state.

        Returns/Yields:
            If stream=False: State with propagated_masks populated
            If stream=True: Generator yielding (frame_idx, masks_dict, state) tuples
        """
        if "video_source" not in state:
            raise ValueError("Must call set_video before propagate")

        if not state.get("frame_masks") and not state.get("seed_masks"):
            raise ValueError("Must call add_prompt_on_frame or set_seed_masks before propagate")

        # Initialize tracker state
        video_source = state["video_source"]
        inference_state = self.model.init_state(video_source)

        # Helper to ensure mask is 2D numpy array
        def _prepare_mask_for_tracker(mask):
            """Ensure mask is 2D (H, W) numpy array for tracker."""
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            # Squeeze any singleton dimensions
            mask = np.squeeze(mask)

            # If still not 2D, try common patterns
            if mask.ndim == 3:
                if mask.shape[0] == 1:  # [1, H, W]
                    mask = mask[0]
                elif mask.shape[-1] == 1:  # [H, W, 1]
                    mask = mask[..., 0]
                else:
                    raise ValueError(f"Cannot convert mask with shape {mask.shape} to 2D")

            if mask.ndim != 2:
                raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

            return mask.astype(bool)

        # Add conditioning frames from detection
        if state.get("frame_masks"):
            for frame_idx, data in state["frame_masks"].items():
                masks = data["masks"]
                for obj_id, mask in enumerate(masks):
                    mask_2d = _prepare_mask_for_tracker(mask)
                    logger.debug(f"Adding detection mask for obj_id={obj_id} on frame {frame_idx}, shape={mask_2d.shape}")
                    result = self.model.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=mask_2d,
                    )
                    logger.debug(f"add_new_mask result: frame_idx={result[0]}, has_output={result[1] is not None}")

        # Add conditioning frames from seed masks
        # We use the tracker directly for mask-only prompts since the full VG pipeline
        # is designed for text+detection workflow
        if state.get("seed_masks"):
            # Pre-compute backbone features for seed mask frames
            # The tracker needs cached features to run inference
            seed_frame_indices = sorted(state["seed_masks"].keys())
            logger.debug(f"Pre-computing backbone features for seed frames: {seed_frame_indices}")
            for i, frame_idx in enumerate(seed_frame_indices):
                if frame_idx not in inference_state.get("feature_cache", {}):
                    # Run backbone to cache features for this frame
                    logger.debug(f"Computing backbone features for frame {frame_idx} ({i+1}/{len(seed_frame_indices)})...")
                    t0 = time.time()
                    self.model._prepare_backbone_feats(inference_state, frame_idx, reverse=False)
                    elapsed = time.time() - t0
                    logger.debug(f"Cached backbone features for frame {frame_idx} in {elapsed:.2f}s")

            # Create a single tracker state for all objects if not already present
            if len(inference_state.get("tracker_inference_states", [])) == 0:
                # Initialize tracker state using model's helper to share feature_cache
                tracker_state = self.model._init_new_tracker_state(inference_state)
                inference_state["tracker_inference_states"] = [tracker_state]
            else:
                tracker_state = inference_state["tracker_inference_states"][0]

            for frame_idx, obj_masks in state["seed_masks"].items():
                for obj_id, mask in obj_masks.items():
                    mask_2d = _prepare_mask_for_tracker(mask)
                    logger.debug(f"Adding seed mask for obj_id={obj_id} on frame {frame_idx}, shape={mask_2d.shape}")

                    # Convert to tensor for tracker
                    mask_tensor = torch.from_numpy(mask_2d).to(self.device)

                    # Ensure backbone features are cached for this frame before calling add_new_mask
                    # (the pre-compute loop should have done this, but we double-check here as a safety measure)
                    feature_cache = inference_state.get("feature_cache", {})
                    if frame_idx not in feature_cache or "tracker_backbone_out" not in feature_cache.get(frame_idx, (None, {}))[1]:
                        logger.info(f"Computing backbone features for frame {frame_idx} (not cached, computing before add_new_mask)...")
                        self.model._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

                    # Call tracker's add_new_mask directly
                    logger.debug(f"Calling tracker.add_new_mask for obj_id={obj_id} frame={frame_idx}...")
                    result = self.model.tracker.add_new_mask(
                        inference_state=tracker_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=mask_tensor,
                    )
                    logger.debug(f"tracker.add_new_mask completed: frame_idx={result[0]}, obj_ids={result[1]}")

        # Determine start frame
        cond_frames = set(state.get("frame_masks", {}).keys()) | set(state.get("seed_masks", {}).keys())
        start_frame = min(cond_frames)

        # Verify masks were added
        num_tracker_states = len(inference_state.get("tracker_inference_states", []))
        if num_tracker_states == 0:
            raise RuntimeError(
                "No tracker states were created. Masks may not have been added correctly. "
                f"seed_masks frames: {list(state.get('seed_masks', {}).keys())}, "
                f"frame_masks frames: {list(state.get('frame_masks', {}).keys())}"
            )

        # Debug: Check action history
        action_history = inference_state.get("action_history", [])
        logger.debug(f"Action history: {[(a.get('type'), a.get('obj_ids'), a.get('frame_idx')) for a in action_history]}")

        # Debug: Check tracker metadata
        tracker_metadata = inference_state.get("tracker_metadata", {})
        logger.debug(f"Tracker metadata obj_ids: {tracker_metadata.get('obj_ids_all_gpu', [])}")

        # Debug: Check tracker state internals
        for i, ts in enumerate(inference_state.get("tracker_inference_states", [])):
            cond_frames = list(ts.get("output_dict", {}).get("cond_frame_outputs", {}).keys())
            non_cond_frames = list(ts.get("output_dict", {}).get("non_cond_frame_outputs", {}).keys())
            obj_ids = ts.get("obj_ids", [])
            temp_cond = {k: list(v.get("cond_frame_outputs", {}).keys())
                        for k, v in ts.get("temp_output_dict_per_obj", {}).items()}
            logger.debug(
                f"Tracker state {i}: obj_ids={obj_ids}, "
                f"cond_frames={cond_frames}, non_cond_frames={non_cond_frames}, "
                f"temp_cond_per_obj={temp_cond}"
            )
            if len(cond_frames) == 0 and len(non_cond_frames) == 0:
                logger.warning(f"Tracker state {i} has NO conditioning frames! Mask add may have failed.")

        logger.debug(f"Propagating with {num_tracker_states} tracker state(s), starting from frame {start_frame}")

        # Propagate
        state["propagated_masks"] = {}
        state["_inference_state"] = inference_state

        def _propagate_generator():
            # Check if we should use direct tracker propagation (for seed_masks path)
            # or the full model propagation (for frame_masks from VG detection)
            use_direct_tracker = bool(state.get("seed_masks")) and not state.get("frame_masks")

            # Use inference_mode context manager to ensure proper context even when
            # generator is iterated externally (stream=True)
            with torch.inference_mode():
                if use_direct_tracker:
                    # Direct tracker propagation for mask-only workflow
                    # We need to compute backbone features frame-by-frame since the tracker
                    # doesn't have its own backbone and relies on cached features
                    tracker_state = inference_state["tracker_inference_states"][0]

                    # Call preflight to consolidate temp outputs to cond_frame_outputs
                    logger.debug(f"Calling propagate_in_video_preflight...")
                    self.model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
                    logger.debug(f"Preflight done, cond_frames: {list(tracker_state.get('output_dict', {}).get('cond_frame_outputs', {}).keys())}")

                    # Get processing order (same logic as tracker's propagate_in_video)
                    num_frames = state.get("num_frames", inference_state["num_frames"])
                    if direction == "backward":
                        processing_order = list(range(start_frame - 1, -1, -1))
                    elif direction == "both":
                        # Forward from start_frame, then backward from start_frame-1
                        forward = list(range(start_frame, num_frames))
                        backward = list(range(start_frame - 1, -1, -1))
                        processing_order = forward + backward
                    else:  # forward
                        processing_order = list(range(start_frame, num_frames))

                    video_height = state.get("orig_height", tracker_state.get("video_height"))
                    video_width = state.get("orig_width", tracker_state.get("video_width"))

                    # Get consolidated frame indices (frames with mask inputs)
                    consolidated_cond = tracker_state.get("consolidated_frame_inds", {}).get("cond_frame_outputs", set())
                    consolidated_non_cond = tracker_state.get("consolidated_frame_inds", {}).get("non_cond_frame_outputs", set())
                    output_dict = tracker_state.get("output_dict", {})

                    # Track the previous frame to determine direction
                    prev_frame_idx = None

                    logger.debug(f"Starting frame processing loop with {len(processing_order)} frames: first={processing_order[0] if processing_order else 'N/A'}, last={processing_order[-1] if processing_order else 'N/A'}")
                    for frame_idx in tqdm(processing_order, desc="propagate in video"):
                        # Determine if this is a reverse step
                        if prev_frame_idx is not None:
                            is_reverse = (frame_idx < prev_frame_idx)
                        else:
                            is_reverse = (direction == "backward")
                        prev_frame_idx = frame_idx

                        # Pre-compute backbone features for this frame if not cached
                        feature_cache = inference_state.get("feature_cache", {})
                        if frame_idx not in feature_cache or "tracker_backbone_out" not in feature_cache.get(frame_idx, (None, {}))[1]:
                            logger.debug(f"Computing backbone features for frame {frame_idx}...")
                            self.model._prepare_backbone_feats(inference_state, frame_idx, reverse=is_reverse)
                            logger.debug(f"Computed backbone features for frame {frame_idx}")

                        # Check if this frame is already in consolidated outputs (conditioning frame)
                        if frame_idx in consolidated_cond:
                            storage_key = "cond_frame_outputs"
                            current_out = output_dict[storage_key][frame_idx]
                            pred_masks = current_out.get("pred_masks")
                            obj_scores = current_out.get("object_score_logits")
                        elif frame_idx in consolidated_non_cond:
                            storage_key = "non_cond_frame_outputs"
                            current_out = output_dict[storage_key][frame_idx]
                            pred_masks = current_out.get("pred_masks")
                            obj_scores = current_out.get("object_score_logits")
                        else:
                            # Run tracker inference on this frame
                            storage_key = "non_cond_frame_outputs"
                            batch_size = self.model.tracker._get_obj_num(tracker_state)
                            logger.debug(f"Running tracker inference on frame {frame_idx} (batch_size={batch_size})...")
                            current_out, pred_masks = self.model.tracker._run_single_frame_inference(
                                inference_state=tracker_state,
                                output_dict=output_dict,
                                frame_idx=frame_idx,
                                batch_size=batch_size,
                                is_init_cond_frame=False,
                                point_inputs=None,
                                mask_inputs=None,
                                reverse=is_reverse,
                                run_mem_encoder=True,
                            )
                            obj_scores = current_out.get("object_score_logits")
                            output_dict[storage_key][frame_idx] = current_out
                            # Create per-object output slices
                            self.model.tracker._add_output_per_object(
                                tracker_state, frame_idx, current_out, storage_key
                            )

                        # Track that we've processed this frame
                        tracker_state["frames_already_tracked"][frame_idx] = {"reverse": is_reverse}

                        # Get video resolution masks
                        out_obj_ids = tracker_state["obj_ids"]
                        if pred_masks is not None:
                            low_res_masks, video_res_masks = self.model.tracker._get_orig_video_res_output(
                                tracker_state, pred_masks
                            )
                            masks = (video_res_masks > 0).cpu()
                        else:
                            # For conditioning frames, use pred_masks_video_res directly
                            video_res_masks = current_out.get("pred_masks_video_res")
                            if video_res_masks is not None:
                                masks = (video_res_masks > 0).squeeze(1).cpu()
                            else:
                                masks = torch.zeros(len(out_obj_ids), video_height, video_width, dtype=torch.bool)

                        scores = obj_scores.cpu() if obj_scores is not None else torch.ones(len(out_obj_ids))

                        masks_dict = {
                            "obj_ids": list(out_obj_ids),
                            "masks": masks,
                            "scores": scores,
                        }
                        state["propagated_masks"][frame_idx] = masks_dict
                        yield frame_idx, masks_dict, state
                else:
                    # Full model propagation (VG + tracker)
                    # Ensure preflight is called
                    for ts in inference_state.get("tracker_inference_states", []):
                        self.model.tracker.propagate_in_video_preflight(ts, run_mem_encoder=True)

                    for out_frame_idx, out in self.model.propagate_in_video(
                        inference_state,
                        start_frame_idx=start_frame,
                        reverse=(direction == "backward"),
                    ):
                        if out is None:
                            continue  # Skip non-rank-0 outputs in multi-GPU

                        # out is a dict with: out_obj_ids, out_probs, out_boxes_xywh, out_binary_masks
                        masks_dict = {
                            "obj_ids": list(out["out_obj_ids"]),
                            "masks": torch.from_numpy(out["out_binary_masks"]),
                            "scores": torch.from_numpy(out["out_probs"]),
                            "boxes": torch.from_numpy(out["out_boxes_xywh"]),
                        }
                        state["propagated_masks"][out_frame_idx] = masks_dict
                        yield out_frame_idx, masks_dict, state

                state["propagation_done"] = True

        if stream:
            return _propagate_generator()
        else:
            # Consume generator and return final state
            for _ in _propagate_generator():
                pass
            return state

    @torch.inference_mode()
    def refine_frame(
        self,
        frame_idx: int,
        state: Dict,
        use_mask_guidance: bool = True,
        crop_padding: float = 1.0,
    ) -> Dict:
        """
        Refine a single frame using high-precision detector.

        Uses propagated mask or interactive mask as guidance. Crops the frame
        around the guidance mask region for improved accuracy and speed.

        Args:
            frame_idx: Frame to refine
            state: State from propagate()
            use_mask_guidance: Use mask conditioning (recommended)
            crop_padding: Padding factor around guidance mask bbox (1.0 = 100% padding)

        Returns:
            Updated state with refined mask for this frame
        """
        # Determine guidance mask
        # Priority: seed > interactive > propagated
        guidance_mask = None

        if frame_idx in state.get("seed_masks", {}):
            # Convert seed_masks format {obj_id: mask} to combined mask
            obj_masks = state["seed_masks"][frame_idx]
            masks_list = [obj_masks[oid] for oid in obj_masks.keys()]
            guidance_mask = torch.stack(masks_list)
            guidance_source = "seed"
        elif frame_idx in state.get("frame_masks", {}):
            guidance_mask = state["frame_masks"][frame_idx]["masks"]
            guidance_source = "interactive"
        elif frame_idx in state.get("propagated_masks", {}):
            guidance_mask = state["propagated_masks"][frame_idx]["masks"]
            guidance_source = "propagated"

        if guidance_mask is None:
            logger.warning(f"No guidance mask for frame {frame_idx}, skipping refinement")
            return state

        # Get frame
        frame = self._get_frame(state, frame_idx)

        # Get original frame dimensions (C, H, W)
        if isinstance(frame, torch.Tensor):
            orig_h, orig_w = frame.shape[-2:]
        elif isinstance(frame, np.ndarray):
            if frame.ndim == 3:
                orig_h, orig_w = frame.shape[:2] if frame.shape[2] <= 4 else frame.shape[1:3]
            else:
                orig_h, orig_w = frame.shape
        else:
            # PIL Image
            orig_w, orig_h = frame.size

        # Compute crop region from guidance mask
        guidance_for_crop = guidance_mask.to(self.device)
        # Resize guidance to original frame size if needed
        if guidance_for_crop.shape[-2:] != (orig_h, orig_w):
            guidance_for_crop = F.interpolate(
                guidance_for_crop.unsqueeze(0).float() if guidance_for_crop.dim() == 2
                else guidance_for_crop.unsqueeze(0).float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        crop_region = self._get_crop_region(
            guidance_for_crop, orig_h, orig_w, padding=crop_padding
        )
        x1, y1, x2, y2 = crop_region
        crop_h, crop_w = y2 - y1, x2 - x1

        # Skip cropping if crop is too small or nearly full image
        min_crop_size = 64
        use_crop = (
            crop_w >= min_crop_size
            and crop_h >= min_crop_size
            and (crop_w < orig_w * 0.95 or crop_h < orig_h * 0.95)
        )

        if use_crop:
            # Crop the frame
            if isinstance(frame, torch.Tensor):
                cropped_frame = frame[..., y1:y2, x1:x2]
            elif isinstance(frame, np.ndarray):
                cropped_frame = frame[y1:y2, x1:x2]
            else:
                # PIL Image
                cropped_frame = frame.crop((x1, y1, x2, y2))

            # Crop the guidance mask
            if guidance_for_crop.dim() == 2:
                cropped_guidance = guidance_for_crop[y1:y2, x1:x2]
            else:
                cropped_guidance = guidance_for_crop[:, y1:y2, x1:x2]

            # Run detection on cropped frame
            det_state = self.set_image(cropped_frame)

            if state.get("text_prompt"):
                det_state = self.set_text_prompt(state["text_prompt"], det_state)

            if use_mask_guidance:
                # Combine multi-object masks if needed
                if cropped_guidance.dim() == 3 and cropped_guidance.shape[0] > 1:
                    combined_mask = cropped_guidance.any(dim=0).float()
                else:
                    combined_mask = cropped_guidance.squeeze().float()

                det_state = self.add_mask_prompt(combined_mask, det_state)
            else:
                # Fallback to box guidance
                if cropped_guidance.numel() > 0 and cropped_guidance.any():
                    boxes = self._mask_to_boxes_cxcywh(cropped_guidance)
                else:
                    boxes = []
                for box in boxes:
                    det_state = self.add_box_prompt(box, True, det_state)

            # Transform results back to original coordinates
            if "masks" in det_state:
                # Paste masks back to full image size
                det_masks_full = self._paste_mask_back(
                    det_state["masks"], crop_region, (orig_h, orig_w)
                )
                det_boxes_full = self._offset_boxes(
                    det_state["boxes"], crop_region, (orig_h, orig_w)
                )
                det_state["masks"] = det_masks_full
                det_state["boxes"] = det_boxes_full

            # Use full-size guidance for IoU validation
            guidance_for_iou = guidance_for_crop
            if guidance_for_iou.dim() == 3 and guidance_for_iou.shape[0] > 1:
                guidance_for_iou = guidance_for_iou.any(dim=0).float()
            else:
                guidance_for_iou = guidance_for_iou.squeeze().float()
        else:
            # No cropping - use full frame
            det_state = self.set_image(frame)

            if state.get("text_prompt"):
                det_state = self.set_text_prompt(state["text_prompt"], det_state)

            if use_mask_guidance:
                # Combine multi-object masks if needed
                if guidance_mask.dim() == 3 and guidance_mask.shape[0] > 1:
                    combined_mask = guidance_mask.any(dim=0).float()
                else:
                    combined_mask = guidance_mask.squeeze().float()

                det_state = self.add_mask_prompt(combined_mask, det_state)
            else:
                # Fallback to box guidance
                if guidance_mask.numel() > 0 and guidance_mask.any():
                    boxes = self._mask_to_boxes_cxcywh(guidance_mask)
                else:
                    boxes = []
                for box in boxes:
                    det_state = self.add_box_prompt(box, True, det_state)

            guidance_for_iou = combined_mask if use_mask_guidance else guidance_mask.squeeze()

        # Update propagated masks with refined result (with IoU validation)
        if "masks" in det_state:
            det_masks = det_state["masks"]

            # Resize guidance to match detection size if needed
            if guidance_for_iou.shape[-2:] != det_masks.shape[-2:]:
                guidance_for_iou = F.interpolate(
                    guidance_for_iou.unsqueeze(0).unsqueeze(0).float(),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            # Find best matching detection by IoU
            best_iou = 0.0
            best_idx = -1
            for i in range(det_masks.shape[0]):
                iou = self._compute_mask_iou(det_masks[i], guidance_for_iou)
                iou_val = iou.item() if iou.dim() == 0 else iou.max().item()
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i

            # IoU threshold of 0.5 - only keep detection if it matches guidance
            if best_iou >= 0.5 and best_idx >= 0:
                if "refined_masks" not in state:
                    state["refined_masks"] = {}

                state["refined_masks"][frame_idx] = {
                    "masks": det_masks[best_idx : best_idx + 1].cpu(),
                    "boxes": det_state["boxes"][best_idx : best_idx + 1].cpu(),
                    "scores": det_state["scores"][best_idx : best_idx + 1].cpu(),
                    "guidance_source": guidance_source,
                    "match_iou": best_iou,
                    "used_crop": use_crop,
                }
            else:
                # IoU too low - silently keep propagated mask (don't add to refined_masks)
                logger.debug(
                    f"Frame {frame_idx}: best IoU={best_iou:.2f} < 0.5, keeping {guidance_source} mask"
                )

        return state

    @torch.inference_mode()
    def refine_all_frames(
        self,
        state: Dict,
        refine_every_n: Optional[int] = 1,
        use_mask_guidance: bool = True,
        stream: bool = False,
        include_seed_frames: bool = True,
        crop_padding: float = 1.0,
    ):
        """
        Refine all frames using DETR with mask guidance.

        Args:
            state: State from propagate() or with seed_masks loaded
            refine_every_n: Refine every Nth frame (default=1, all frames)
            use_mask_guidance: Use mask conditioning (default=True)
            stream: If True, yields (frame_idx, refined_masks_dict, state) per frame.
                   If False (default), collects all and returns state.
            include_seed_frames: If True (default), also refine frames with seed_masks.
                   Set to False to skip frames that have seed_masks (original behavior).
            crop_padding: Padding factor around guidance mask bbox (1.0 = 100% padding).
                   Cropping improves accuracy by focusing on the region of interest.

        Returns/Yields:
            If stream=False: State with refined_masks populated
            If stream=True: Generator yielding (frame_idx, refined_dict, state) tuples
        """
        # Check if there are any masks to refine (from propagation OR seed_masks)
        has_propagated = bool(state.get("propagated_masks"))
        has_seed_masks = bool(state.get("seed_masks"))

        if not has_propagated and not has_seed_masks:
            raise ValueError(
                "No masks available for refinement. Either call propagate() first "
                "or provide pre-computed masks via set_seed_masks()."
            )

        # Determine frames to refine
        frames_to_refine = []

        # Get all frame indices from all available mask sources
        all_frames = set()
        all_frames.update(state.get("propagated_masks", {}).keys())
        if include_seed_frames:
            all_frames.update(state.get("seed_masks", {}).keys())
        all_frames = sorted(all_frames)

        for frame_idx in all_frames:
            # Skip frames that already have interactive masks (user-provided, high quality)
            if frame_idx in state.get("frame_masks", {}):
                continue

            # Optionally skip seed_mask frames (backward compatibility)
            if not include_seed_frames and frame_idx in state.get("seed_masks", {}):
                continue

            # Apply refine_every_n filter
            if refine_every_n is None or frame_idx % refine_every_n == 0:
                frames_to_refine.append(frame_idx)

        logger.info(f"Refining {len(frames_to_refine)} frames")

        def _refine_generator():
            for frame_idx in frames_to_refine:
                state_updated = self.refine_frame(
                    frame_idx, state, use_mask_guidance, crop_padding
                )

                refined_dict = state_updated.get("refined_masks", {}).get(frame_idx, {})
                yield frame_idx, refined_dict, state_updated

        if stream:
            return _refine_generator()
        else:
            # Consume generator and return final state
            for frame_idx, refined_dict, state in _refine_generator():
                pass
            return state

    def get_masks(self, state: Dict, frame_idx: Optional[int] = None) -> Dict:
        """
        Get final masks from state.

        Priority: refined > seed > interactive > propagated

        Args:
            state: Current state
            frame_idx: Specific frame, or None for all

        Returns:
            Dict with masks for requested frame(s)
        """
        if frame_idx is not None:
            # Single frame
            if frame_idx in state.get("refined_masks", {}):
                return state["refined_masks"][frame_idx]
            elif frame_idx in state.get("seed_masks", {}):
                # Convert seed_masks format {obj_id: mask} to standard format
                obj_masks = state["seed_masks"][frame_idx]
                obj_ids = list(obj_masks.keys())
                masks = torch.stack([obj_masks[oid] for oid in obj_ids])
                return {
                    "masks": masks,
                    "obj_ids": obj_ids,
                    "scores": torch.ones(len(obj_ids)),
                    "source": "seed",
                }
            elif frame_idx in state.get("frame_masks", {}):
                return state["frame_masks"][frame_idx]
            elif frame_idx in state.get("propagated_masks", {}):
                return state["propagated_masks"][frame_idx]
            else:
                return {"masks": None}

        # All frames
        result = {}
        all_frames = set()
        all_frames.update(state.get("seed_masks", {}).keys())
        all_frames.update(state.get("frame_masks", {}).keys())
        all_frames.update(state.get("propagated_masks", {}).keys())
        all_frames.update(state.get("refined_masks", {}).keys())

        for idx in sorted(all_frames):
            result[idx] = self.get_masks(state, idx)

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def _get_text_embeddings(self, prompt: str) -> Dict:
        """Get text embeddings with caching."""
        if prompt in self._text_cache:
            return self._text_cache[prompt]

        text_out = self.model.detector.backbone.forward_text([prompt], device=self.device)

        if len(self._text_cache) >= self._text_cache_size:
            oldest = next(iter(self._text_cache))
            del self._text_cache[oldest]

        self._text_cache[prompt] = text_out
        return text_out

    def _forward_detection(self, state: Dict) -> Dict:
        """Run detection forward pass."""
        find_input = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        outputs = self.model.detector.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_input,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        return self._postprocess_detection(outputs, state)

    def _forward_detection_with_mask(self, state: Dict) -> Dict:
        """Run detection with mask conditioning."""
        mask = state["mask_guidance"]

        # Check if mask_encoder is available
        mask_encoder = getattr(
            getattr(self.model.detector, "geometry_encoder", None),
            "mask_encoder",
            None
        )

        if mask_encoder is not None:
            # Use direct mask encoding if available
            mask_encoded = mask_encoder.mask_downsampler(mask)
            mask_encoded = mask_encoded.flatten(-2).permute(2, 0, 1)

            find_input = FindStage(
                img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            # Run with mask conditioning
            prompt, prompt_mask, backbone_out = self.model.detector._encode_prompt(
                state["backbone_out"], find_input, state["geometric_prompt"],
                prev_mask_pred=mask_encoded
            )

            backbone_out, encoder_out, _ = self.model.detector._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )

            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {"encoder_out": encoder_out, "backbone_out": backbone_out},
            }

            out, hs = self.model.detector._run_decoder(
                pos_embed=encoder_out["pos_embed"],
                memory=encoder_out["encoder_hidden_states"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=encoder_out["prompt_after_enc"],
                prompt_mask=encoder_out["prompt_mask"],
                encoder_out=encoder_out,
            )

            if self.model.detector.segmentation_head is not None:
                out["pred_masks"] = self.model.detector.segmentation_head(
                    hs=hs, encoder_out=encoder_out, out=out
                )

            state["backbone_out"] = backbone_out
            return self._postprocess_detection(out, state)
        else:
            # Fallback: mask_encoder not available, use box-based guidance
            # The boxes have already been derived from mask in add_mask_prompt
            # Just run standard detection with the geometric prompt (boxes)
            logger.debug("mask_encoder not available, using box-based guidance for refinement")
            return self._forward_detection(state)

    def _postprocess_detection(self, outputs: Dict, state: Dict) -> Dict:
        """Post-process detection outputs."""
        orig_h = state["original_height"]
        orig_w = state["original_width"]

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs.get("pred_masks")
        out_probs = out_logits.sigmoid()

        if "presence_logit_dec" in outputs:
            presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = (out_probs * presence).squeeze(-1)
        else:
            out_probs = out_probs.squeeze(-1)

        # Filter by confidence
        keep = out_probs > self.detection_confidence_threshold
        out_probs = out_probs[keep]
        out_bbox = out_bbox[keep]

        if out_masks is not None:
            out_masks = out_masks[keep]

            # Resize masks
            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()

            state["masks_logits"] = out_masks
            state["masks"] = (out_masks.squeeze(1) > 0.5)
        else:
            state["masks"] = None
            state["masks_logits"] = None

        # Convert boxes
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                            device=self.device, dtype=torch.float32)
        state["boxes"] = boxes * scale
        state["scores"] = out_probs

        return state

    def _get_frame(self, state: Dict, frame_idx: int):
        """Get a frame from video source."""
        video_source = state["video_source"]

        if isinstance(video_source, str):
            import cv2
            cap = cv2.VideoCapture(video_source)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        elif isinstance(video_source, list):
            if 0 <= frame_idx < len(video_source):
                return video_source[frame_idx]
        return None

    def _mask_to_boxes_cxcywh(
        self,
        mask: torch.Tensor,
    ) -> List[List[float]]:
        """Convert mask to [cx, cy, w, h] normalized boxes."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = mask.to(self.device)

        if not mask.any():
            return []

        boxes_xyxy = masks_to_boxes(mask)
        h, w = mask.shape[-2:]

        result = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box.tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            result.append([cx, cy, bw, bh])

        return result

    def _compute_mask_iou(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU between masks.

        Args:
            mask1: First mask tensor [H, W] or [N, H, W]
            mask2: Second mask tensor [H, W] or [N, H, W]

        Returns:
            IoU value(s) as tensor
        """
        # Ensure same device
        mask1 = mask1.to(self.device)
        mask2 = mask2.to(self.device)

        # Binarize masks
        mask1_binary = (mask1 > 0.5).float()
        mask2_binary = (mask2 > 0.5).float()

        # Compute intersection and union
        intersection = (mask1_binary * mask2_binary).sum(dim=(-2, -1))
        union = ((mask1_binary + mask2_binary) > 0).float().sum(dim=(-2, -1))

        return intersection / (union + 1e-6)

    def _get_crop_region(
        self,
        mask: torch.Tensor,
        img_h: int,
        img_w: int,
        padding: float = 1.0,
    ) -> tuple:
        """
        Get crop region from mask with padding.

        Args:
            mask: Binary mask [H, W] or [N, H, W]
            img_h: Original image height
            img_w: Original image width
            padding: Padding factor (1.0 = 100% padding on each side)

        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates, clamped to image bounds
        """
        if mask.dim() == 3:
            mask = mask.any(dim=0)

        # Find non-zero pixels
        nonzero = torch.nonzero(mask > 0.5)
        if len(nonzero) == 0:
            return (0, 0, img_w, img_h)

        y_coords = nonzero[:, 0]
        x_coords = nonzero[:, 1]

        y1, y2 = y_coords.min().item(), y_coords.max().item()
        x1, x2 = x_coords.min().item(), x_coords.max().item()

        # Add padding
        box_h = y2 - y1
        box_w = x2 - x1
        pad_h = int(box_h * padding)
        pad_w = int(box_w * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(img_w, x2 + pad_w)
        y2 = min(img_h, y2 + pad_h)

        return (x1, y1, x2, y2)

    def _paste_mask_back(
        self,
        cropped_mask: torch.Tensor,
        crop_region: tuple,
        original_size: tuple,
    ) -> torch.Tensor:
        """
        Paste cropped mask back to original image size.

        Args:
            cropped_mask: Mask from cropped region [N, H, W] or [H, W]
            crop_region: (x1, y1, x2, y2) of the crop
            original_size: (H, W) of original image

        Returns:
            Full-size mask tensor
        """
        x1, y1, x2, y2 = crop_region
        orig_h, orig_w = original_size

        # Handle batch dimension
        had_batch = cropped_mask.dim() == 3
        if not had_batch:
            cropped_mask = cropped_mask.unsqueeze(0)

        n_masks = cropped_mask.shape[0]
        crop_h, crop_w = y2 - y1, x2 - x1

        # Resize cropped mask to crop region size if needed
        if cropped_mask.shape[-2:] != (crop_h, crop_w):
            cropped_mask = F.interpolate(
                cropped_mask.unsqueeze(1).float(),
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Create full-size mask and paste
        full_mask = torch.zeros(
            (n_masks, orig_h, orig_w),
            dtype=cropped_mask.dtype,
            device=cropped_mask.device,
        )
        full_mask[:, y1:y2, x1:x2] = cropped_mask

        if not had_batch:
            full_mask = full_mask.squeeze(0)

        return full_mask

    def _offset_boxes(
        self,
        boxes: torch.Tensor,
        crop_region: tuple,
        original_size: tuple,
    ) -> torch.Tensor:
        """
        Convert boxes from cropped coordinates to original image coordinates.

        Args:
            boxes: Boxes in cxcywh normalized format [N, 4]
            crop_region: (x1, y1, x2, y2) of the crop in pixels
            original_size: (H, W) of original image

        Returns:
            Boxes in cxcywh normalized format relative to original image
        """
        x1, y1, x2, y2 = crop_region
        orig_h, orig_w = original_size
        crop_h, crop_w = y2 - y1, x2 - x1

        # Convert from normalized crop coords to pixel coords in crop
        boxes_pixel = boxes.clone()
        boxes_pixel[:, 0] = boxes[:, 0] * crop_w  # cx
        boxes_pixel[:, 1] = boxes[:, 1] * crop_h  # cy
        boxes_pixel[:, 2] = boxes[:, 2] * crop_w  # w
        boxes_pixel[:, 3] = boxes[:, 3] * crop_h  # h

        # Offset to original image pixel coords
        boxes_pixel[:, 0] += x1  # cx
        boxes_pixel[:, 1] += y1  # cy

        # Convert back to normalized coords relative to original image
        boxes_orig = boxes_pixel.clone()
        boxes_orig[:, 0] = boxes_pixel[:, 0] / orig_w
        boxes_orig[:, 1] = boxes_pixel[:, 1] / orig_h
        boxes_orig[:, 2] = boxes_pixel[:, 2] / orig_w
        boxes_orig[:, 3] = boxes_pixel[:, 3] / orig_h

        return boxes_orig

    def clear_text_cache(self):
        """Clear text embedding cache."""
        self._text_cache.clear()


def build_sam3_unified_predictor(
    checkpoint_path: Optional[str] = None,
    load_from_HF: bool = True,
    bpe_path: Optional[str] = None,
    device: str = "cuda",
    detection_confidence_threshold: float = 0.5,
    **model_kwargs,
) -> Sam3UnifiedProcessor:
    """
    Build Sam3UnifiedProcessor.

    Args:
        checkpoint_path: Path to checkpoint
        load_from_HF: Load from HuggingFace if no path
        bpe_path: Path to BPE tokenizer
        device: Device to use
        detection_confidence_threshold: Default confidence threshold
        **model_kwargs: Additional args for model building

    Returns:
        Sam3UnifiedProcessor instance
    """
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        bpe_path=bpe_path,
        device=device,
        **model_kwargs,
    )

    return Sam3UnifiedProcessor(
        model=model,
        detection_confidence_threshold=detection_confidence_threshold,
    )
