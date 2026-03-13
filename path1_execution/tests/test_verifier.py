"""Tests for verification/verifier.py.

Uses a mock LIBERO env — no real LIBERO installation required.
"""

from __future__ import annotations

import numpy as np
import pytest

from path1_execution.config import (
    ObjectInfo,
    Path1Config,
    PlannedSketch,
    SceneContext,
    Trajectory,
)
from path1_execution.verification.verifier import verify_trajectory


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------

class MockEnv:
    """Minimal mock env for verifier tests.

    Returns final_eef_pos on every step. Signals success (done + reward > 0)
    only when self.success is True and step count reaches the trajectory length.
    """

    def __init__(self, final_eef_pos, success: bool = False):
        self.final_eef_pos = np.array(final_eef_pos, dtype=np.float64)
        self.success = success
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return {
            "robot0_eef_pos": np.zeros(3, dtype=np.float64),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }

    def step(self, action):
        self._step_count += 1
        obs = {
            "robot0_eef_pos": self.final_eef_pos.copy(),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        done = self._step_count >= 3
        reward = 1.0 if (done and self.success) else 0.0
        return obs, reward, done, {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_object_info(label: str, centroid: np.ndarray) -> ObjectInfo:
    return ObjectInfo(
        label=label,
        centroid_3d=np.array(centroid, dtype=np.float64),
        bbox_3d_corners=np.zeros((8, 3)),
        orientation=np.eye(3),
        bbox_2d_per_camera={},
    )


def _make_scene(task: str, goal_pos: np.ndarray) -> SceneContext:
    source = _make_object_info("block", np.array([0.2, 0.0, 0.8]))
    goal = _make_object_info("basket", goal_pos)
    return SceneContext(
        point_cloud=np.zeros((8, 3)),
        object_registry={"block": source, "basket": goal},
        camera_images={},
        camera_matrices={},
        task_instruction=task,
        initial_eef_pose=np.zeros(7),
    )


def _make_trajectory(goal_label: str = "basket") -> Trajectory:
    return Trajectory(
        actions=[np.array([0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0])] * 3,
        sub_goals=["pick", "place"],
        grasp_indices=[1],
        metadata={"goal_label": goal_label},
    )


def _make_empty_sketch() -> PlannedSketch:
    return PlannedSketch(primitives_per_camera={}, rendered_images={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_verifier_success():
    """Trajectory ending within 0.03m of goal with sim reward → verified=True."""
    goal_pos = np.array([0.5, 0.0, 0.8])
    # EEF ends 0.02m from goal — within 5cm threshold
    final_pos = goal_pos + np.array([0.02, 0.0, 0.0])

    env = MockEnv(final_eef_pos=final_pos, success=True)
    scene = _make_scene("pick up the block and place it in the basket", goal_pos)
    trajectory = _make_trajectory(goal_label="basket")
    config = Path1Config(goal_threshold_m=0.05)
    sketch = _make_empty_sketch()

    result = verify_trajectory(trajectory, scene, env, config, sketch)

    assert result.verified is True
    assert result.sim_success is True
    assert result.geometric_success is True
    assert result.feedback is None
    assert result.distance_to_goal < 0.05


def test_verifier_failure_distance():
    """Trajectory ending 0.20m from goal → geometric_success=False, verified=False."""
    goal_pos = np.array([0.5, 0.0, 0.8])
    final_pos = np.array([0.3, 0.0, 0.8])  # 0.20m away in X

    env = MockEnv(final_eef_pos=final_pos, success=False)
    scene = _make_scene("pick up the block and place it in the basket", goal_pos)
    trajectory = _make_trajectory(goal_label="basket")
    config = Path1Config(goal_threshold_m=0.05)
    sketch = _make_empty_sketch()

    result = verify_trajectory(trajectory, scene, env, config, sketch)

    assert result.verified is False
    assert result.geometric_success is False
    assert result.distance_to_goal >= 0.20 - 1e-6
    assert result.feedback is not None
    assert len(result.feedback) > 0


def test_verifier_failure_no_sim_success():
    """Near goal but sim didn't signal success → verified=False."""
    goal_pos = np.array([0.5, 0.0, 0.8])
    final_pos = goal_pos + np.array([0.01, 0.0, 0.0])  # 0.01m away — geometric passes

    env = MockEnv(final_eef_pos=final_pos, success=False)  # no reward
    scene = _make_scene("pick up the block and place it in the basket", goal_pos)
    trajectory = _make_trajectory(goal_label="basket")
    config = Path1Config(goal_threshold_m=0.05)
    sketch = _make_empty_sketch()

    result = verify_trajectory(trajectory, scene, env, config, sketch)

    assert result.verified is False
    assert result.sim_success is False
    assert result.geometric_success is True  # close enough geometrically
    assert result.feedback is not None


def test_verifier_feedback_content_mock():
    """Mock feedback string must be non-empty and contain expected keywords."""
    goal_pos = np.array([0.5, 0.0, 0.8])
    env = MockEnv(final_eef_pos=np.zeros(3), success=False)
    scene = _make_scene("pick up the block", goal_pos)
    trajectory = _make_trajectory(goal_label="basket")
    config = Path1Config(goal_threshold_m=0.05, use_mock_planner=True)
    sketch = _make_empty_sketch()

    result = verify_trajectory(trajectory, scene, env, config, sketch)

    assert result.feedback is not None
    # Should mention trajectory and/or coordinates
    assert "Trajectory" in result.feedback or "trajectory" in result.feedback


def test_verifier_no_registry_fallback():
    """Verifier works when object registry is empty (uses zero-vector fallback)."""
    env = MockEnv(final_eef_pos=np.zeros(3), success=False)
    scene = SceneContext(
        point_cloud=np.zeros((1, 3)),
        object_registry={},  # empty
        camera_images={},
        camera_matrices={},
        task_instruction="test",
        initial_eef_pose=np.zeros(7),
    )
    trajectory = Trajectory(
        actions=[np.array([0.0, 0.0, 0.0, 0, 0, 0, 1.0])] * 3,
        sub_goals=[],
        grasp_indices=[],
        metadata={},
    )
    config = Path1Config(goal_threshold_m=0.05)
    sketch = _make_empty_sketch()

    # Should not raise
    result = verify_trajectory(trajectory, scene, env, config, sketch)
    assert isinstance(result.verified, bool)
    assert result.distance_to_goal >= 0.0


def test_verifier_goal_label_lookup():
    """goal_label in metadata takes priority over alphabetical fallback."""
    goal_pos = np.array([0.5, 0.0, 0.8])
    other_pos = np.array([0.9, 0.9, 0.9])  # far away

    env = MockEnv(final_eef_pos=goal_pos + np.array([0.01, 0.0, 0.0]), success=True)
    scene = SceneContext(
        point_cloud=np.zeros((8, 3)),
        object_registry={
            "zzz_far_object": _make_object_info("zzz_far_object", other_pos),
            "basket": _make_object_info("basket", goal_pos),
        },
        camera_images={},
        camera_matrices={},
        task_instruction="place in basket",
        initial_eef_pose=np.zeros(7),
    )
    # metadata explicitly sets basket as goal
    trajectory = Trajectory(
        actions=[np.array([0.5, 0.0, 0.8, 0, 0, 0, 1.0])] * 3,
        sub_goals=[],
        grasp_indices=[],
        metadata={"goal_label": "basket"},
    )
    config = Path1Config(goal_threshold_m=0.05)
    sketch = _make_empty_sketch()

    result = verify_trajectory(trajectory, scene, env, config, sketch)

    # Distance should be to basket, not zzz_far_object
    assert result.distance_to_goal < 0.05
    assert result.geometric_success is True
