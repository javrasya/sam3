# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3 (Segment Anything Model 3) is Meta's unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts (points, boxes, masks). Key capabilities include open-vocabulary concept segmentation with 270K+ unique concepts.

## Build and Development Commands

```bash
# Install for development
pip install -e ".[dev,train]"

# Install for running notebooks
pip install -e ".[notebooks]"

# Format code
ufmt format .

# Run tests
pytest tests/
```

## Architecture Overview

### Core Model Components

The model has 848M parameters with a **detector** and **tracker** sharing a vision encoder:

1. **Vision Backbone** (`sam3/model/vitdet.py`, `sam3/model/necks.py`):
   - ViT-based encoder (1024 embed dim, 32 layers)
   - `Sam3DualViTDetNeck` creates feature pyramids at multiple scales

2. **Text Encoder** (`sam3/model/text_encoder_ve.py`):
   - `VETextEncoder` with BPE tokenizer
   - Processes text prompts into language features

3. **Visual-Language Backbone** (`sam3/model/vl_combiner.py`):
   - `SAM3VLBackbone` combines visual and text encoders

4. **Detector** (`sam3/model/sam3_image.py`):
   - DETR-based architecture with transformer encoder/decoder
   - `Sam3Image` handles grounding with text/geometric prompts
   - Uses presence token for discriminating similar prompts

5. **Tracker** (`sam3/model/sam3_tracker_base.py`, `sam3/model/sam3_tracking_predictor.py`):
   - Inherits SAM 2 transformer encoder-decoder architecture
   - `Sam3TrackerPredictor` for video segmentation
   - Memory-based tracking with mask encoder

### Main Entry Points

- **Image inference**: `build_sam3_image_model()` in `sam3/model_builder.py`
- **Video inference**: `build_sam3_video_predictor()` in `sam3/model_builder.py`
- **Image processor**: `Sam3Processor` in `sam3/model/sam3_image_processor.py`
- **Video predictor**: `Sam3VideoPredictor` in `sam3/model/sam3_video_predictor.py`

### Inference State Pattern

Both image and video APIs use a **state dictionary** pattern:
- `set_image()` initializes state with backbone features
- `set_text_prompt()` / `add_geometric_prompt()` / `add_point_prompt()` update state and run inference
- Results (masks, boxes, scores) are stored in state and returned as tensors with N detections

### Key Data Structures

- `Prompt` (`sam3/model/geometry_encoders.py`): Encapsulates geometric prompts (boxes, points)
- `FindStage` (`sam3/model/data_misc.py`): Input structure for grounding inference
- `SAM3Output` (`sam3/model/model_misc.py`): Output structure from model forward pass

## Agent Module

`sam3/agent/` contains the SAM 3 Agent for complex text prompt segmentation using an MLLM to break down prompts.

## Evaluation

- Evaluation scripts in `scripts/eval/`
- SA-Co benchmarks: Gold (`scripts/eval/gold/`), Silver (`scripts/eval/silver/`), VEval (`scripts/eval/veval/`)
- Metrics: cgF1 (`sam3/eval/cgf1_eval.py`), COCO eval (`sam3/eval/coco_eval.py`), HOTA (`sam3/eval/hota_eval_toolkit/`)

## HuggingFace Integration

Checkpoints hosted at `facebook/sam3`. Authentication required:
```bash
huggingface-cli login
```
