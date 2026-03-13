"""Tests for SceneAgent (Stage 1 + Stage 2).

Gate: Test loads a real LIBERO HDF5, calls SceneAgent.run(), and confirms
SceneContext.object_registry contains at least one entry with a valid
non-zero centroid_3d.
"""

import numpy as np
import pytest

from path1_execution.config import Path1Config
from path1_execution.agents.scene_agent import SceneAgent
from sketch_anything.tools.run_pipeline import load_libero_env

HDF5_PATH = (
    "/home/amitrand/Desktop/sketch_anything/LIBERO/libero/datasets/"
    "libero_goal/put_the_bowl_on_the_plate_demo.hdf5"
)


def test_scene_agent_with_real_hdf5():
    config = Path1Config(
        camera_names=["agentview"],
        image_width=256,
        image_height=256,
    )

    env, task_instruction, _ = load_libero_env(
        HDF5_PATH, camera_names=config.camera_names
    )

    agent = SceneAgent()
    scene = agent.run(env, task_instruction, config)

    # Basic structure checks
    assert scene.object_registry is not None, "object_registry must not be None"
    assert len(scene.object_registry) >= 1, (
        f"Expected at least 1 task-relevant object, got {len(scene.object_registry)}. "
        f"Task: '{task_instruction}'"
    )

    # Each ObjectInfo must have a non-zero centroid_3d
    for label, obj_info in scene.object_registry.items():
        assert obj_info.centroid_3d is not None, f"centroid_3d is None for '{label}'"
        assert np.linalg.norm(obj_info.centroid_3d) > 0, (
            f"centroid_3d is zero vector for '{label}'"
        )
        assert obj_info.bbox_3d_corners.shape == (8, 3), (
            f"bbox_3d_corners shape is {obj_info.bbox_3d_corners.shape} for '{label}'"
        )

    # Camera images
    assert scene.camera_images is not None, "camera_images must not be None"
    assert "agentview" in scene.camera_images, "agentview image missing"
    img = scene.camera_images["agentview"]
    assert img.ndim == 3 and img.shape[2] == 3, (
        f"agentview image has unexpected shape {img.shape}"
    )

    # Point cloud
    assert scene.point_cloud is not None, "point_cloud must not be None"
    assert scene.point_cloud.shape[1] == 3, (
        f"point_cloud must have shape (N, 3), got {scene.point_cloud.shape}"
    )
    assert len(scene.point_cloud) > 0, "point_cloud must be non-empty"

    # Camera matrices
    assert "agentview" in scene.camera_matrices, "agentview camera matrices missing"
    K, R, t = scene.camera_matrices["agentview"]
    assert K.shape == (3, 3), f"K shape is {K.shape}"
    assert R.shape == (3, 3), f"R shape is {R.shape}"

    # Initial EEF pose
    assert scene.initial_eef_pose.shape == (7,), (
        f"initial_eef_pose shape is {scene.initial_eef_pose.shape}"
    )

    print(f"\nTask: {task_instruction}")
    print(f"Objects found: {list(scene.object_registry.keys())}")
    print(f"Point cloud: {len(scene.point_cloud)} points")
    for label, obj_info in scene.object_registry.items():
        print(f"  {label}: centroid={obj_info.centroid_3d.round(3).tolist()}")
