"""VLM Primitive Generator.

Generates structured sketch primitives from scene images using Qwen2.5-VL.
Supports three backends:
    1. Outlines constrained decoding (guaranteed valid JSON)
    2. vLLM guided generation (faster inference)
    3. Transformers fallback with JSON extraction + retry
    4. Mock backend for development/testing without GPU
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

import numpy as np

from sketch_anything.schemas.primitives import SketchPrimitives
from sketch_anything.validation.validator import validate_primitives
from sketch_anything.vlm.config import VLMConfig
from sketch_anything.vlm.prompt import format_prompt

logger = logging.getLogger(__name__)


class VLMPrimitiveGenerator:
    """Generates sketch primitives from scene images using a VLM."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._initialized = False
        self._generator = None  # Outlines constrained generator
        self._fallback = None   # Transformers-based fallback

    def generate(
        self,
        image: np.ndarray,
        object_registry: Dict[str, dict],
        task_instruction: str,
    ) -> SketchPrimitives:
        """Generate validated sketch primitives.

        Args:
            image: RGB image array, shape (H, W, 3), dtype uint8.
            object_registry: Single-view object registry.
            task_instruction: Natural language task description.

        Returns:
            Validated SketchPrimitives.

        Raises:
            ValueError: If all retries fail.
        """
        if self.config.use_mock:
            return _generate_mock_primitives(object_registry, task_instruction)

        self._lazy_init()

        prompt = format_prompt(object_registry, task_instruction)
        last_error: Optional[str] = None

        for attempt in range(self.config.max_retries):
            try:
                if self._generator is not None:
                    result = self._generate_constrained(image, prompt)
                else:
                    result = self._generate_fallback(image, prompt, attempt, last_error)

                validation = validate_primitives(result, object_registry)
                if validation.is_valid:
                    if validation.warnings:
                        for w in validation.warnings:
                            logger.warning(f"Primitive warning: {w}")
                    return result

                last_error = "\n".join(validation.errors)
                logger.warning(f"Attempt {attempt + 1} validation failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

        raise ValueError(
            f"Failed to generate valid primitives after {self.config.max_retries} "
            f"attempts. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        """Initialize model on first use."""
        if self._initialized:
            return
        self._initialized = True

        if self.config.use_constrained_decoding:
            try:
                self._init_constrained()
                return
            except Exception as e:
                logger.warning(
                    f"Constrained decoding unavailable ({e}), "
                    f"falling back to standard generation"
                )

        self._init_fallback()

    def _init_constrained(self) -> None:
        """Initialize Outlines constrained generator."""
        from outlines import generate, models

        logger.info("Initializing Outlines constrained generator")
        model = models.transformers_vision(
            self.config.model_name,
            device=self.config.device,
            model_kwargs={"torch_dtype": "auto"},
        )
        self._generator = generate.json(model, SketchPrimitives)
        logger.info("Constrained generator ready")

    def _init_fallback(self) -> None:
        """Initialize standard Transformers generator."""
        import torch
        from transformers import AutoProcessor

        # Try the specific Qwen2.5-VL class first, then Qwen2-VL, then the
        # generic AutoModelForVision2Seq which works with local model dirs
        # (the saved config.json specifies the correct architecture).
        ModelClass = None
        model_lower = self.config.model_name.lower()

        if "qwen2.5" in model_lower or "qwen2_5" in model_lower:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
            except ImportError:
                logger.info("Qwen2_5_VLForConditionalGeneration not available in this transformers version")

        if ModelClass is None:
            try:
                from transformers import Qwen2VLForConditionalGeneration as ModelClass
            except ImportError:
                logger.info("Qwen2VLForConditionalGeneration not available either")

        if ModelClass is None:
            try:
                from transformers import AutoModelForVision2Seq
                ModelClass = AutoModelForVision2Seq
                logger.info("Falling back to AutoModelForVision2Seq")
            except ImportError:
                from transformers import AutoModel
                ModelClass = AutoModel
                logger.info("Falling back to AutoModel")

        logger.info(f"Initializing Transformers fallback generator ({self.config.model_name})")
        logger.info(f"  Model class: {ModelClass.__name__}")

        # Determine dtype and device
        if self.config.device == "mps":
            # MPS works best with float16; device_map="auto" doesn't support MPS
            dtype = torch.float16
            self._fallback_model = ModelClass.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to("mps")
        else:
            dtype = "auto"
            self._fallback_model = ModelClass.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

        self._fallback_processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        logger.info("Fallback generator ready")

    # ------------------------------------------------------------------
    # Generation backends
    # ------------------------------------------------------------------

    def _generate_constrained(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> SketchPrimitives:
        """Generate using Outlines constrained decoding."""
        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            result = self._generator(prompt, [f.name])
        return result

    def _generate_fallback(
        self,
        image: np.ndarray,
        prompt: str,
        attempt: int,
        previous_error: Optional[str],
    ) -> SketchPrimitives:
        """Generate using standard Transformers with JSON extraction."""
        import base64
        import torch
        import cv2

        full_prompt = prompt
        if attempt > 0 and previous_error:
            full_prompt += (
                f"\n\nYour previous output contained errors:\n{previous_error}\n\n"
                f"Please correct these errors and provide a valid JSON output."
            )

        # Encode image to base64
        _, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{image_b64}"},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        text = self._fallback_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._fallback_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._fallback_model.device)

        with torch.no_grad():
            output_ids = self._fallback_model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self._fallback_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        result_dict = _extract_json(output_text)
        return SketchPrimitives(**result_dict)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract JSON from potentially messy VLM output."""
    # Try direct parse
    try:
        raw = json.loads(text)
        return _normalize_primitives(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in text
    matches = re.findall(r"\{[\s\S]*\}", text)
    for match in matches:
        try:
            raw = json.loads(match)
            return _normalize_primitives(raw)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract valid JSON from output: {text[:200]}...")


def _normalize_primitives(data: dict) -> dict:
    """Normalize VLM output to match our Pydantic schema.

    VLMs sometimes output positions in nested format like:
        {"object_relative": {"object_id": "x", "anchor": "center"}}
    or:
        {"absolute": {"coords": [0.5, 0.5]}}

    Our schema expects flat discriminated unions:
        {"type": "object_relative", "object_id": "x", "anchor": "center"}
        {"type": "absolute", "coords": [0.5, 0.5]}

    This function handles both formats and also fixes missing 'step' fields.
    """
    if "primitives" not in data:
        return data

    normalized = []
    for prim in data["primitives"]:
        prim = _normalize_primitive(prim)
        normalized.append(prim)

    data["primitives"] = normalized
    return data


def _normalize_primitive(prim: dict) -> dict:
    """Normalize a single primitive dict."""
    # Normalize position fields based on primitive type
    ptype = prim.get("type", "")

    if ptype == "arrow":
        if "start" in prim:
            prim["start"] = _normalize_position(prim["start"])
        if "end" in prim:
            prim["end"] = _normalize_position(prim["end"])
        if "waypoints" in prim:
            prim["waypoints"] = [_normalize_position(w) for w in prim["waypoints"]]
    elif ptype == "circle":
        if "center" in prim:
            prim["center"] = _normalize_position(prim["center"])
    elif ptype == "gripper":
        if "position" in prim:
            prim["position"] = _normalize_position(prim["position"])

    return prim


def _normalize_position(pos: dict) -> dict:
    """Normalize a position dict to the flat discriminated union format.

    Handles these VLM output patterns:
        1. Already correct: {"type": "absolute", "coords": [...]}
        2. Nested: {"absolute": {"coords": [...]}}
        3. Nested: {"object_relative": {"object_id": ..., "anchor": ...}}
        4. Nested with "absolute": {"absolute": [x, y]}
    """
    if not isinstance(pos, dict):
        return pos

    # Already in correct format
    if "type" in pos and pos["type"] in ("absolute", "object_relative"):
        return pos

    # Nested format: {"object_relative": {...}}
    if "object_relative" in pos:
        inner = pos["object_relative"]
        if isinstance(inner, dict):
            return {"type": "object_relative", **inner}

    # Nested format: {"absolute": {...}} or {"absolute": [x, y]}
    if "absolute" in pos:
        inner = pos["absolute"]
        if isinstance(inner, dict):
            return {"type": "absolute", **inner}
        if isinstance(inner, (list, tuple)):
            return {"type": "absolute", "coords": list(inner)}

    return pos


# ---------------------------------------------------------------------------
# Mock generator
# ---------------------------------------------------------------------------

def _generate_mock_primitives(
    object_registry: Dict[str, dict],
    task_instruction: str,
) -> SketchPrimitives:
    """Generate mock primitives for testing without a VLM.

    Produces a standard pick-and-place sequence using the first two
    non-gripper objects in the registry:
        Step 1: approach arrow + grasp circle
        Step 2: gripper close
        Step 3: transport arrow
        Step 4: release circle + gripper open
    """
    # Identify source and target objects
    non_gripper = [
        (oid, data) for oid, data in object_registry.items()
        if oid != "gripper"
    ]

    if len(non_gripper) == 0:
        return SketchPrimitives(primitives=[])

    source_id = non_gripper[0][0]

    if len(non_gripper) >= 2:
        target_id = non_gripper[1][0]
    else:
        target_id = source_id

    gripper_id = "gripper" if "gripper" in object_registry else None

    primitives_data: list[dict] = []

    # Step 1: approach arrow + grasp circle
    primitives_data.append({
        "type": "circle",
        "center": {"type": "object_relative", "object_id": source_id, "anchor": "center"},
        "radius": 0.04,
        "purpose": "grasp_point",
        "step": 1,
    })

    if gripper_id:
        primitives_data.append({
            "type": "arrow",
            "start": {"type": "object_relative", "object_id": gripper_id, "anchor": "center"},
            "end": {"type": "object_relative", "object_id": source_id, "anchor": "top", "offset": [0.0, -0.03]},
            "waypoints": [],
            "step": 1,
        })

    # Step 2: gripper close
    primitives_data.append({
        "type": "gripper",
        "position": {"type": "object_relative", "object_id": source_id, "anchor": "center"},
        "action": "close",
        "step": 2,
    })

    # Step 3: transport arrow
    if target_id != source_id:
        src_center = object_registry[source_id]["center"]
        tgt_center = object_registry[target_id]["center"]
        mid_x = (src_center[0] + tgt_center[0]) / 2
        waypoint_y = min(src_center[1], tgt_center[1]) - 0.1

        primitives_data.append({
            "type": "arrow",
            "start": {"type": "object_relative", "object_id": source_id, "anchor": "center"},
            "end": {"type": "object_relative", "object_id": target_id, "anchor": "center", "offset": [0.0, -0.05]},
            "waypoints": [{"type": "absolute", "coords": [max(0.0, min(1.0, mid_x)), max(0.0, min(1.0, waypoint_y))]}],
            "step": 3,
        })

    # Step 4: release circle + gripper open
    primitives_data.append({
        "type": "circle",
        "center": {"type": "object_relative", "object_id": target_id, "anchor": "center"},
        "radius": 0.05,
        "purpose": "release_point",
        "step": 4,
    })
    primitives_data.append({
        "type": "gripper",
        "position": {"type": "object_relative", "object_id": target_id, "anchor": "center", "offset": [0.0, -0.05]},
        "action": "open",
        "step": 4,
    })

    return SketchPrimitives(**{"primitives": primitives_data})
