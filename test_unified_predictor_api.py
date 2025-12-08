#!/usr/bin/env python3
"""
Simple API validation script for Sam3UnifiedPredictor.

This script validates the API structure without requiring a GPU or model weights.
"""

import inspect
import sys
from pathlib import Path

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from sam3 import build_sam3_unified_predictor, Sam3UnifiedPredictor
    print("✓ Successfully imported build_sam3_unified_predictor and Sam3UnifiedPredictor")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Check build function signature
print("\n" + "="*70)
print("build_sam3_unified_predictor signature:")
print("="*70)
sig = inspect.signature(build_sam3_unified_predictor)
print(f"Parameters: {list(sig.parameters.keys())}")
print(f"Return type: {sig.return_annotation}")

expected_params = [
    "checkpoint_path",
    "load_from_HF",
    "bpe_path",
    "device",
    "detection_confidence_threshold",
    "gpus_to_use",
    "model_kwargs",
]

for param in expected_params:
    if param in sig.parameters or (param == "model_kwargs" and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())):
        print(f"  ✓ {param}")
    else:
        print(f"  ✗ Missing parameter: {param}")

# Check Sam3UnifiedPredictor methods
print("\n" + "="*70)
print("Sam3UnifiedPredictor methods:")
print("="*70)

expected_methods = [
    "detect_frame",
    "detect_frame_batch",
    "propagate_and_refine",
    "refine_frame",
    "start_session",
    "propagate_in_video",
]

for method_name in expected_methods:
    if hasattr(Sam3UnifiedPredictor, method_name):
        method = getattr(Sam3UnifiedPredictor, method_name)
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        print(f"  ✓ {method_name}({', '.join(params[1:3])}{'...' if len(params) > 3 else ''})")
    else:
        print(f"  ✗ Missing method: {method_name}")

# Check detect_frame signature
print("\n" + "="*70)
print("detect_frame signature:")
print("="*70)
detect_sig = inspect.signature(Sam3UnifiedPredictor.detect_frame)
print(f"Parameters: {list(detect_sig.parameters.keys())}")

detect_params = [
    "self",
    "image",
    "text",
    "boxes",
    "box_labels",
    "points",
    "point_labels",
    "confidence_threshold",
]

for param in detect_params:
    if param in detect_sig.parameters:
        print(f"  ✓ {param}")
    else:
        print(f"  ✗ Missing parameter: {param}")

print("\n" + "="*70)
print("API validation complete!")
print("="*70)
