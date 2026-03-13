"""Stage 4: Project 3D trajectory waypoints to 2D sketch primitives per camera view."""

from __future__ import annotations

import logging
import os
from typing import List

import cv2
import numpy as np

from sketch_anything.libero_utils.camera import project_points
from sketch_anything.schemas.primitives import (
    AbsolutePosition,
    ArrowPrimitive,
    CirclePrimitive,
    SketchPrimitives,
)
from sketch_anything.rendering.renderer import render_primitives

from path1_execution.config import Path1Config, PlannedSketch, SceneContext, Trajectory

logger = logging.getLogger(__name__)


def _make_abs_pos(x: float, y: float) -> AbsolutePosition:
    """Create an AbsolutePosition with coords clamped to [0, 1]."""
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    return AbsolutePosition(type="absolute", coords=(x, y))


def project_trajectory(
    trajectory: Trajectory,
    scene_context: SceneContext,
    config: Path1Config,
) -> PlannedSketch:
    """Project a 3D trajectory onto 2D camera views as sketch primitives.

    For each camera with matrices available in scene_context:
    - Projects each EEF position to normalized 2D coords
    - Builds an ArrowPrimitive covering all actions
    - Emits CirclePrimitive for each grasp index
    - Renders onto the camera image (or a blank image if unavailable)
    - Saves debug rendered images to config.output_dir/debug/

    Args:
        trajectory: The planned trajectory.
        scene_context: Scene context with camera matrices and images.
        config: Pipeline configuration.

    Returns:
        PlannedSketch with primitives and rendered images per camera.
    """
    primitives_per_camera: dict = {}
    rendered_images: dict = {}

    # Ensure debug output directory exists
    debug_dir = os.path.join(config.output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    for camera_name, (K, R, t) in scene_context.camera_matrices.items():
        # Project all EEF positions to 2D
        positions_3d = np.array([action[:3] for action in trajectory.actions], dtype=np.float64)
        if len(positions_3d) == 0:
            logger.warning("Trajectory has no actions; skipping camera %s", camera_name)
            continue

        positions_2d = project_points(
            positions_3d, K, R, t, config.image_width, config.image_height
        )  # (N, 2), already clamped to [0, 1]

        primitives: list = []

        # Build single ArrowPrimitive for entire trajectory motion
        if len(positions_2d) >= 2:
            start_pos = _make_abs_pos(positions_2d[0, 0], positions_2d[0, 1])
            end_pos = _make_abs_pos(positions_2d[-1, 0], positions_2d[-1, 1])
            waypoints: List[AbsolutePosition] = [
                _make_abs_pos(positions_2d[i, 0], positions_2d[i, 1])
                for i in range(1, len(positions_2d) - 1)
            ]
            arrow = ArrowPrimitive(
                type="arrow",
                start=start_pos,
                end=end_pos,
                waypoints=waypoints,
                step=1,
            )
            primitives.append(arrow)
        elif len(positions_2d) == 1:
            # Single-point degenerate case: start == end, no waypoints
            pos = _make_abs_pos(positions_2d[0, 0], positions_2d[0, 1])
            arrow = ArrowPrimitive(
                type="arrow",
                start=pos,
                end=pos,
                waypoints=[],
                step=1,
            )
            primitives.append(arrow)

        # Emit CirclePrimitive for each grasp index
        for grasp_idx in trajectory.grasp_indices:
            if 0 <= grasp_idx < len(positions_2d):
                px, py = positions_2d[grasp_idx]
                circle = CirclePrimitive(
                    type="circle",
                    center=_make_abs_pos(px, py),
                    radius=0.04,
                    purpose="grasp_point",
                    step=2,
                )
                primitives.append(circle)
            else:
                logger.warning(
                    "grasp_index %d out of range for trajectory length %d",
                    grasp_idx,
                    len(positions_2d),
                )

        sketch_primitives = SketchPrimitives(primitives=primitives)
        primitives_per_camera[camera_name] = sketch_primitives

        # Get or create base image
        if camera_name in scene_context.camera_images:
            base_image = scene_context.camera_images[camera_name].copy()
        else:
            logger.warning(
                "No image found for camera %s; using blank canvas", camera_name
            )
            base_image = np.zeros(
                (config.image_height, config.image_width, 3), dtype=np.uint8
            )

        # Render primitives onto image (use empty object registry — all AbsolutePosition)
        try:
            rendered = render_primitives(base_image, sketch_primitives, {})
        except Exception as exc:
            logger.error(
                "render_primitives failed for camera %s: %s", camera_name, exc
            )
            rendered = base_image

        rendered_images[camera_name] = rendered

        # Save debug image
        debug_path = os.path.join(debug_dir, f"projected_{camera_name}.png")
        try:
            # render_primitives returns RGB; convert to BGR for cv2.imwrite
            cv2.imwrite(debug_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            logger.info("Saved debug projection image: %s", debug_path)
        except Exception as exc:
            logger.warning("Could not save debug image %s: %s", debug_path, exc)

    return PlannedSketch(
        primitives_per_camera=primitives_per_camera,
        rendered_images=rendered_images,
    )
