"""End-to-end pipeline orchestrator.

Coordinates object registry building, VLM generation, validation, and
rendering across multiple camera views.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sketch_anything.config import Config
from sketch_anything.libero_utils.env import get_camera_image
from sketch_anything.registry.builder import build_object_registry
from sketch_anything.rendering.config import RenderConfig
from sketch_anything.rendering.renderer import render_primitives
from sketch_anything.schemas.primitives import SketchPrimitives
from sketch_anything.validation.validator import validate_primitives
from sketch_anything.vlm.config import VLMConfig
from sketch_anything.vlm.generator import VLMPrimitiveGenerator

logger = logging.getLogger(__name__)


@dataclass
class AnnotatedView:
    """Result for a single camera view."""

    camera_name: str
    original_image: np.ndarray
    annotated_image: np.ndarray
    primitives: SketchPrimitives
    object_registry: dict


class SketchPipeline:
    """Orchestrates end-to-end sketch generation for LIBERO tasks."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self._vlm_generator: Optional[VLMPrimitiveGenerator] = None

    @property
    def vlm_generator(self) -> VLMPrimitiveGenerator:
        """Lazy-initialize VLM generator."""
        if self._vlm_generator is None:
            self._vlm_generator = VLMPrimitiveGenerator(self.config.vlm)
        return self._vlm_generator

    def generate(
        self,
        env,
        task_instruction: str,
        camera_names: Optional[List[str]] = None,
    ) -> Dict[str, AnnotatedView]:
        """Generate sketch annotations for all camera views.

        Args:
            env: LIBERO ControlEnv / OffScreenRenderEnv instance.
            task_instruction: Natural language task description.
            camera_names: Camera identifiers. Defaults to config.

        Returns:
            Dict mapping camera_name -> AnnotatedView.
        """
        if camera_names is None:
            camera_names = self.config.libero.camera_names

        image_w = self.config.libero.image_width
        image_h = self.config.libero.image_height

        # Build object registries for all views
        logger.info(f"Building object registries for task: {task_instruction}")
        registries = build_object_registry(
            env, task_instruction, camera_names, image_w, image_h
        )

        results: Dict[str, AnnotatedView] = {}

        for camera_name in camera_names:
            logger.info(f"Processing view: {camera_name}")

            # Capture image
            image = get_camera_image(env, camera_name)

            # Infer actual image dimensions from captured image
            actual_h, actual_w = image.shape[:2]
            if actual_w != image_w or actual_h != image_h:
                logger.warning(
                    f"Image size ({actual_w}x{actual_h}) differs from config "
                    f"({image_w}x{image_h}). Using actual image dimensions."
                )

            # Get view-specific registry
            registry = registries.get(camera_name, {})
            if not registry:
                logger.warning(f"Empty registry for {camera_name}, skipping VLM")
                continue

            # Generate primitives
            primitives = self.vlm_generator.generate(
                image=image,
                object_registry=registry,
                task_instruction=task_instruction,
            )

            # Validate (should pass since generator validates internally)
            validation = validate_primitives(primitives, registry)
            if not validation.is_valid:
                logger.error(f"Post-generation validation failed: {validation.errors}")
            for w in validation.warnings:
                logger.warning(f"Validation warning: {w}")

            # Render
            annotated = render_primitives(
                image=image.copy(),
                primitives=primitives,
                object_registry=registry,
                config=self.config.rendering,
            )

            results[camera_name] = AnnotatedView(
                camera_name=camera_name,
                original_image=image,
                annotated_image=annotated,
                primitives=primitives,
                object_registry=registry,
            )

        return results


def check_view_consistency(
    annotations: Dict[str, AnnotatedView],
) -> List[str]:
    """Check that primitives across views are logically consistent.

    Args:
        annotations: Dict of camera_name -> AnnotatedView.

    Returns:
        List of warning messages for any inconsistencies found.
    """
    warnings: List[str] = []

    if len(annotations) < 2:
        return warnings

    view_names = list(annotations.keys())
    reference = annotations[view_names[0]]

    for view_name in view_names[1:]:
        other = annotations[view_name]

        # Check same number of primitives
        ref_count = len(reference.primitives.primitives)
        other_count = len(other.primitives.primitives)
        if ref_count != other_count:
            warnings.append(
                f"Primitive count mismatch: {view_names[0]} has {ref_count}, "
                f"{view_name} has {other_count}"
            )

        # Check same step sequence
        ref_steps = sorted({p.step for p in reference.primitives.primitives})
        other_steps = sorted({p.step for p in other.primitives.primitives})
        if ref_steps != other_steps:
            warnings.append(
                f"Step sequence mismatch: {view_names[0]} has steps {ref_steps}, "
                f"{view_name} has steps {other_steps}"
            )

        # Check same gripper actions in order
        from sketch_anything.schemas.primitives import GripperPrimitive

        ref_gripper = [
            (p.step, p.action)
            for p in reference.primitives.primitives
            if isinstance(p, GripperPrimitive)
        ]
        other_gripper = [
            (p.step, p.action)
            for p in other.primitives.primitives
            if isinstance(p, GripperPrimitive)
        ]
        if ref_gripper != other_gripper:
            warnings.append("Gripper action mismatch between views")

    return warnings
