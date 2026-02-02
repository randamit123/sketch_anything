#!/usr/bin/env python3
"""Test VLM inference on a real LIBERO frame using Qwen2.5-VL-3B on Mac MPS.

Usage:
    python -m sketch_anything.tools.test_vlm_local

This script:
    1. Loads an extracted LIBERO demo frame (PNG)
    2. Creates a mock object registry (since we don't have MuJoCo)
    3. Sends the image + prompt to Qwen2.5-VL-3B-Instruct
    4. Validates the returned primitives
    5. Renders the sketch overlay onto the frame
    6. Saves the annotated image

No GPU or LIBERO environment required -- runs on Mac MPS (Apple Silicon).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from sketch_anything.rendering.config import RenderConfig
from sketch_anything.rendering.renderer import render_primitives
from sketch_anything.schemas.primitives import SketchPrimitives
from sketch_anything.validation.validator import validate_primitives
from sketch_anything.vlm.config import VLMConfig
from sketch_anything.vlm.generator import VLMPrimitiveGenerator
from sketch_anything.vlm.prompt import format_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("test_vlm_local")


def get_stove_registry() -> dict:
    """Create a mock object registry for the 'turn on the stove' task.

    Since we don't have MuJoCo running, we approximate bounding boxes
    from visual inspection of the extracted frame.
    """
    return {
        "gripper": {
            "id": "gripper",
            "label": "robot gripper",
            "bbox": [0.40, 0.15, 0.60, 0.35],
            "center": [0.50, 0.25],
        },
        "stove": {
            "id": "stove",
            "label": "stove",
            "bbox": [0.25, 0.55, 0.75, 0.85],
            "center": [0.50, 0.70],
        },
        "stove_knob": {
            "id": "stove_knob",
            "label": "stove knob",
            "bbox": [0.35, 0.70, 0.45, 0.80],
            "center": [0.40, 0.75],
        },
    }


def main():
    # --- Locate extracted frame ---
    output_dir = project_root / "outputs" / "turn_on_the_stove"
    frame_path = output_dir / "demo_0_agentview_frame0.png"

    if not frame_path.exists():
        logger.error(f"Frame not found: {frame_path}")
        logger.error("Run extract_demos.py first to extract LIBERO demo frames.")
        sys.exit(1)

    logger.info(f"Loading frame: {frame_path}")
    image_bgr = cv2.imread(str(frame_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    logger.info(f"Image shape: {image_rgb.shape}")

    # --- Object registry ---
    registry = get_stove_registry()
    task_instruction = "turn on the stove"
    logger.info(f"Task: {task_instruction}")
    logger.info(f"Objects: {list(registry.keys())}")

    # --- Print the prompt (for debugging) ---
    prompt = format_prompt(registry, task_instruction)
    logger.info(f"Prompt length: {len(prompt)} chars")

    # --- Configure VLM for Mac ---
    vlm_config = VLMConfig(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        max_tokens=2048,
        temperature=0.1,
        max_retries=3,
        use_constrained_decoding=False,  # No Outlines on Mac
        device="mps",
        use_mock=False,
    )

    logger.info(f"Model: {vlm_config.model_name}")
    logger.info(f"Device: {vlm_config.device}")
    logger.info("Loading model (this will download ~6GB on first run)...")

    # --- Generate primitives ---
    generator = VLMPrimitiveGenerator(vlm_config)

    t0 = time.time()
    try:
        primitives = generator.generate(
            image=image_rgb,
            object_registry=registry,
            task_instruction=task_instruction,
        )
        elapsed = time.time() - t0
        logger.info(f"Generation succeeded in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"Generation failed after {elapsed:.1f}s: {e}")
        sys.exit(1)

    # --- Show results ---
    logger.info(f"Generated {len(primitives.primitives)} primitives:")
    for i, p in enumerate(primitives.primitives):
        pdict = p.model_dump()
        logger.info(f"  [{i}] type={pdict['type']}, step={pdict['step']}")

    # --- Validate ---
    validation = validate_primitives(primitives, registry)
    logger.info(f"Validation: valid={validation.is_valid}")
    if validation.errors:
        for err in validation.errors:
            logger.error(f"  Error: {err}")
    if validation.warnings:
        for warn in validation.warnings:
            logger.warning(f"  Warning: {warn}")

    # --- Render overlay ---
    render_config = RenderConfig()
    annotated = render_primitives(
        image=image_rgb.copy(),
        primitives=primitives,
        object_registry=registry,
        config=render_config,
    )

    # --- Save outputs ---
    out_path = output_dir / "demo_0_agentview_vlm_annotated.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved annotated image: {out_path}")

    # Save primitives JSON
    json_path = output_dir / "vlm_primitives.json"
    with open(json_path, "w") as f:
        json.dump(primitives.model_dump(), f, indent=2)
    logger.info(f"Saved primitives: {json_path}")

    # Save raw prompt for reference
    prompt_path = output_dir / "vlm_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(prompt)
    logger.info(f"Saved prompt: {prompt_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
