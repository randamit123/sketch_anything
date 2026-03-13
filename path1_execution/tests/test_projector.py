"""Tests for projection/trajectory_projector.py.

Uses synthetic camera matrices and trajectories — no real LIBERO env required.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from path1_execution.config import (
    Path1Config,
    SceneContext,
    Trajectory,
)
from path1_execution.projection.trajectory_projector import project_trajectory
from sketch_anything.schemas.primitives import ArrowPrimitive, CirclePrimitive


def _make_synthetic_scene(tmp_path) -> tuple:
    """Build a synthetic SceneContext with a simple top-down camera."""
    # Camera looking straight down from 2m above origin (approximate pinhole)
    # K: focal length = 256, principal point at center of 256x256 image
    K = np.array([[256.0, 0.0, 128.0],
                  [0.0, 256.0, 128.0],
                  [0.0,  0.0,   1.0]], dtype=np.float64)
    # Identity rotation (camera axes aligned with world axes)
    R = np.eye(3, dtype=np.float64)
    # Camera 2m above origin along Z
    t = np.array([[0.0], [0.0], [2.0]], dtype=np.float64)

    camera_images = {"test_cam": np.zeros((256, 256, 3), dtype=np.uint8)}
    camera_matrices = {"test_cam": (K, R, t)}

    scene_context = SceneContext(
        point_cloud=np.zeros((8, 3), dtype=np.float64),
        object_registry={},
        camera_images=camera_images,
        camera_matrices=camera_matrices,
        task_instruction="test task",
        initial_eef_pose=np.zeros(7),
    )
    config = Path1Config(
        camera_names=["test_cam"],
        output_dir=str(tmp_path),
        image_width=256,
        image_height=256,
    )
    return scene_context, config


def test_project_trajectory_synthetic(tmp_path):
    """All projected 2D points should be in [0, 1]. Rendered image must exist."""
    scene_context, config = _make_synthetic_scene(tmp_path)

    actions = [
        np.array([0.0, 0.0, 0.8, 0, 0, 0, 1.0]),
        np.array([0.1, 0.0, 0.8, 0, 0, 0, 1.0]),
        np.array([0.2, 0.0, 0.8, 0, 0, 0, -1.0]),
        np.array([0.3, 0.0, 0.8, 0, 0, 0, -1.0]),
        np.array([0.4, 0.0, 0.8, 0, 0, 0, 1.0]),
    ]
    trajectory = Trajectory(
        actions=actions,
        sub_goals=["test"],
        grasp_indices=[2],
        metadata={},
    )

    sketch = project_trajectory(trajectory, scene_context, config)

    # Primitives and rendered image must exist for the test camera
    assert "test_cam" in sketch.primitives_per_camera, "Missing test_cam in primitives"
    assert "test_cam" in sketch.rendered_images, "Missing test_cam in rendered_images"

    # Rendered image must have 3 channels
    img = sketch.rendered_images["test_cam"]
    assert img.ndim == 3 and img.shape[2] == 3, f"Unexpected image shape: {img.shape}"

    # At least one primitive must be generated
    prims = sketch.primitives_per_camera["test_cam"].primitives
    assert len(prims) > 0, "No primitives generated"

    # There should be exactly one ArrowPrimitive
    arrows = [p for p in prims if isinstance(p, ArrowPrimitive)]
    assert len(arrows) == 1, f"Expected 1 arrow, got {len(arrows)}"

    # There should be exactly one CirclePrimitive (grasp at index 2)
    circles = [p for p in prims if isinstance(p, CirclePrimitive)]
    assert len(circles) == 1, f"Expected 1 circle, got {len(circles)}"
    assert circles[0].purpose == "grasp_point"

    # All coords in arrow start/end/waypoints must be in [0, 1]
    arrow = arrows[0]
    for pos in [arrow.start, arrow.end] + list(arrow.waypoints):
        x, y = pos.coords
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, (
            f"Projected coord out of bounds: ({x}, {y})"
        )

    # Circle center must also be in bounds
    cx, cy = circles[0].center.coords
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, (
        f"Circle center out of bounds: ({cx}, {cy})"
    )

    # Debug image must have been saved
    debug_path = os.path.join(str(tmp_path), "debug", "projected_test_cam.png")
    assert os.path.exists(debug_path), f"Debug image not saved at {debug_path}"


def test_project_trajectory_no_images(tmp_path):
    """Projector should create a blank fallback image when camera_images is empty."""
    K = np.array([[256.0, 0.0, 128.0],
                  [0.0, 256.0, 128.0],
                  [0.0,  0.0,   1.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([[0.0], [0.0], [2.0]], dtype=np.float64)

    scene_context = SceneContext(
        point_cloud=np.zeros((8, 3)),
        object_registry={},
        camera_images={},  # intentionally empty
        camera_matrices={"blank_cam": (K, R, t)},
        task_instruction="test",
        initial_eef_pose=np.zeros(7),
    )
    config = Path1Config(
        camera_names=["blank_cam"],
        output_dir=str(tmp_path),
        image_width=256,
        image_height=256,
    )

    actions = [
        np.array([0.0, 0.0, 0.5, 0, 0, 0, 1.0]),
        np.array([0.1, 0.0, 0.5, 0, 0, 0, 1.0]),
    ]
    trajectory = Trajectory(
        actions=actions,
        sub_goals=["test"],
        grasp_indices=[],
        metadata={},
    )

    sketch = project_trajectory(trajectory, scene_context, config)

    assert "blank_cam" in sketch.rendered_images
    img = sketch.rendered_images["blank_cam"]
    assert img.ndim == 3 and img.shape[2] == 3


def test_project_trajectory_multiple_grasps(tmp_path):
    """Multiple grasp indices should produce multiple CirclePrimitives."""
    scene_context, config = _make_synthetic_scene(tmp_path)

    actions = [np.array([float(i) * 0.05, 0.0, 0.5, 0, 0, 0, 1.0]) for i in range(6)]
    trajectory = Trajectory(
        actions=actions,
        sub_goals=["test"],
        grasp_indices=[1, 4],
        metadata={},
    )

    sketch = project_trajectory(trajectory, scene_context, config)
    prims = sketch.primitives_per_camera["test_cam"].primitives
    circles = [p for p in prims if isinstance(p, CirclePrimitive)]
    assert len(circles) == 2, f"Expected 2 circles for 2 grasps, got {len(circles)}"
