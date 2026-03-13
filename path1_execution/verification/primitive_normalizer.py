"""Stage 6a: Normalize and validate PlannedSketch primitives.

Converts primitives_per_camera to canonical SketchPrimitives objects,
validates them, and logs errors/warnings.
"""

from __future__ import annotations

import logging
from typing import Dict

from sketch_anything.validation.validator import validate_primitives, ValidationResult

from path1_execution.config import PlannedSketch

logger = logging.getLogger(__name__)


def normalize_primitives(
    planned_sketch: PlannedSketch,
    object_registry: dict,
) -> Dict[str, ValidationResult]:
    """Validate primitives for each camera view.

    Since trajectory projector always emits AbsolutePosition primitives, we pass
    an empty object registry to the validator (no object-relative IDs to resolve).

    Args:
        planned_sketch: The PlannedSketch from the projector.
        object_registry: Per-camera object registry dict. May be empty.

    Returns:
        Dict mapping camera_name → ValidationResult.
    """
    results: Dict[str, ValidationResult] = {}

    for camera_name, sketch_primitives in planned_sketch.primitives_per_camera.items():
        # Use empty registry because we use only AbsolutePosition primitives
        validation_result = validate_primitives(sketch_primitives, object_registry={})

        if not validation_result.is_valid:
            for err in validation_result.errors:
                logger.error(
                    "Validation error [%s]: %s", camera_name, err
                )
        for warn in validation_result.warnings:
            logger.warning(
                "Validation warning [%s]: %s", camera_name, warn
            )

        results[camera_name] = validation_result

    return results
