# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Dict, List

import numpy as np
import PIL
import torch

from sam3.model import box_ops

from sam3.model.data_misc import FindStage, interpolate
from sam3.model.model_misc import get_default_device
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(
        self,
        model,
        resolution=1008,
        device=None,
        confidence_threshold=0.5,
        text_cache_size: int = 100,
    ):
        self.model = model
        self.resolution = resolution
        if device is None:
            device = get_default_device()
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        # Text embedding cache for faster repeated prompts
        self._text_cache: Dict[str, Dict] = {}
        self._text_cache_size = text_cache_size

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _get_text_embeddings(self, prompt: str) -> Dict:
        """Get text embeddings, using cache if available."""
        if prompt in self._text_cache:
            return self._text_cache[prompt]

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)

        # Evict oldest entry if cache is full
        if len(self._text_cache) >= self._text_cache_size:
            oldest_key = next(iter(self._text_cache))
            del self._text_cache[oldest_key]

        self._text_cache[prompt] = text_outputs
        return text_outputs

    def clear_text_cache(self):
        """Clear the text embedding cache."""
        self._text_cache.clear()

    @torch.inference_mode()
    def warmup(
        self,
        prompts: List[str] = None,
        warmup_boxes: bool = True,
        warmup_points: bool = True,
    ):
        """Warm up the model by running dummy inference.

        This triggers CUDA kernel compilation and optionally pre-caches text embeddings.

        Args:
            prompts: Optional list of text prompts to pre-cache embeddings for
            warmup_boxes: Whether to warm up box prompt inference path
            warmup_points: Whether to warm up point prompt inference path
        """
        # Create dummy image tensor at the expected resolution
        dummy_image = torch.zeros(
            3, self.resolution, self.resolution, device=self.device
        )

        # Run backbone forward
        state = self.set_image(dummy_image)

        # Run text encoder and grounding with dummy prompt
        state = self.set_text_prompt("warmup", state)

        # Warm up box prompt path
        if warmup_boxes:
            self.reset_all_prompts(state)
            state = self.add_geometric_prompt(
                box=[0.5, 0.5, 0.2, 0.2], label=True, state=state
            )

        # Warm up point prompt path
        if warmup_points:
            self.reset_all_prompts(state)
            state = self.add_point_prompt(point=[0.5, 0.5], label=1, state=state)

        # Warm up combined prompts path (set_all_prompts)
        if warmup_boxes or warmup_points:
            self.reset_all_prompts(state)
            state = self.set_all_prompts(
                state,
                text="warmup",
                points=[[0.5, 0.5]] if warmup_points else None,
                point_labels=[1] if warmup_points else None,
                boxes=[[0.5, 0.5, 0.2, 0.2]] if warmup_boxes else None,
                box_labels=[True] if warmup_boxes else None,
            )

        # Pre-cache additional text prompts if provided
        if prompts:
            for prompt in prompts:
                self._get_text_embeddings(prompt)

        # Synchronize to ensure all CUDA kernels are compiled
        torch.cuda.synchronize()

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(
            images[0], PIL.Image.Image
        ), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self._get_text_embeddings(prompt)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self._get_text_embeddings("visual")
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_point_prompt(self, point: List, label: int, state: Dict):
        """Adds a point prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The point is assumed to be in [x, y] format and normalized in [0, 1] range.
        The label is 1 for a positive point (foreground), 0 for a negative point (background).
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self._get_text_embeddings("visual")
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a sequence and batch dimension (sequence first, batch second)
        points = torch.tensor(point, device=self.device, dtype=torch.float32).view(1, 1, 2)
        labels = torch.tensor([label], device=self.device, dtype=torch.long).view(1, 1)
        state["geometric_prompt"].append_points(points, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_all_prompts(
        self,
        state: Dict,
        text: str = None,
        points: List[List] = None,
        point_labels: List[int] = None,
        boxes: List[List] = None,
        box_labels: List[bool] = None,
    ):
        """Sets all prompts at once and runs inference only once.

        This is more efficient than calling individual prompt methods when
        rebuilding prompts after removal, as it avoids repeated _forward_grounding calls.

        Args:
            state: The inference state from set_image()
            text: Text prompt string, or None
            points: List of [x, y] points normalized in [0, 1], or None
            point_labels: List of labels (1=foreground, 0=background) for each point
            boxes: List of [center_x, center_y, width, height] normalized in [0, 1], or None
            box_labels: List of labels (True=positive, False=negative) for each box

        Returns:
            Updated state with inference results
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_all_prompts")

        # Reset existing prompts first
        self.reset_all_prompts(state)

        # Check if we have any geometric prompts
        has_points = points is not None and len(points) > 0
        has_boxes = boxes is not None and len(boxes) > 0
        has_geometric = has_points or has_boxes

        # Set text prompt (or dummy "visual" if only geometric prompts)
        if text is not None:
            text_outputs = self._get_text_embeddings(text)
            state["backbone_out"].update(text_outputs)
        elif has_geometric:
            dummy_text_outputs = self._get_text_embeddings("visual")
            state["backbone_out"].update(dummy_text_outputs)
        else:
            # No prompts at all
            return state

        # Initialize geometric prompt
        state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Add all points
        if has_points:
            if point_labels is None or len(point_labels) != len(points):
                raise ValueError("point_labels must match length of points")
            for point, label in zip(points, point_labels):
                pt = torch.tensor(point, device=self.device, dtype=torch.float32).view(
                    1, 1, 2
                )
                lbl = torch.tensor([label], device=self.device, dtype=torch.long).view(
                    1, 1
                )
                state["geometric_prompt"].append_points(pt, lbl)

        # Add all boxes
        if has_boxes:
            if box_labels is None or len(box_labels) != len(boxes):
                raise ValueError("box_labels must match length of boxes")
            for box, label in zip(boxes, box_labels):
                bx = torch.tensor(box, device=self.device, dtype=torch.float32).view(
                    1, 1, 4
                )
                lbl = torch.tensor([label], device=self.device, dtype=torch.bool).view(
                    1, 1
                )
                state["geometric_prompt"].append_boxes(bx, lbl)

        return self._forward_grounding(state)

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
