"""Tests for the mock planner and PlannerAgent."""

import numpy as np
import pytest

from path1_execution.config import (
    ObjectInfo,
    Path1Config,
    SceneContext,
    Trajectory,
)
from path1_execution.planning.mock_planner import generate_mock_trajectory
from path1_execution.agents.planner_agent import PlannerAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene(source_centroid=(0.5, 0.0, 0.8), goal_centroid=(0.3, 0.2, 0.8)):
    source = ObjectInfo(
        label="red_block",
        centroid_3d=np.array(source_centroid, dtype=float),
        bbox_3d_corners=np.zeros((8, 3)),
        orientation=np.eye(3),
        bbox_2d_per_camera={},
    )
    goal = ObjectInfo(
        label="basket",
        centroid_3d=np.array(goal_centroid, dtype=float),
        bbox_3d_corners=np.zeros((8, 3)),
        orientation=np.eye(3),
        bbox_2d_per_camera={},
    )
    scene = SceneContext(
        point_cloud=np.zeros((8, 3)),
        object_registry={"red_block": source, "basket": goal},
        camera_images={},
        camera_matrices={},
        task_instruction="pick up the red block and place it in the basket",
        initial_eef_pose=np.array([0.5, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]),
    )
    return scene


# ---------------------------------------------------------------------------
# Gate tests (from CLAUDE.md step 3 specification)
# ---------------------------------------------------------------------------

class TestMockPlannerBasic:
    """Verify the mock planner produces a structurally valid trajectory."""

    def test_at_least_three_actions(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        assert len(traj.actions) >= 3, f"Expected >= 3 actions, got {len(traj.actions)}"

    def test_all_actions_shape_7(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        for i, action in enumerate(traj.actions):
            assert action.shape == (7,), (
                f"Action {i} has shape {action.shape}, expected (7,)"
            )

    def test_grasp_indices_nonempty(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        assert len(traj.grasp_indices) > 0, "grasp_indices should be non-empty"

    def test_sub_goals_nonempty(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        assert len(traj.sub_goals) > 0, "sub_goals should be non-empty"

    def test_zero_rotation_components(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        for i, action in enumerate(traj.actions):
            assert action[3] == 0.0, f"Action {i}: roll should be 0.0, got {action[3]}"
            assert action[4] == 0.0, f"Action {i}: pitch should be 0.0, got {action[4]}"
            assert action[5] == 0.0, f"Action {i}: yaw should be 0.0, got {action[5]}"

    def test_gripper_values_binary(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        for i, action in enumerate(traj.actions):
            assert action[6] in (1.0, -1.0), (
                f"Action {i}: gripper should be 1.0 or -1.0, got {action[6]}"
            )

    def test_returns_trajectory_type(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        assert isinstance(traj, Trajectory)

    def test_grasp_index_in_range(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        for idx in traj.grasp_indices:
            assert 0 <= idx < len(traj.actions), (
                f"grasp_index {idx} out of range [0, {len(traj.actions)})"
            )

    def test_metadata_has_planner_type(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        assert traj.metadata.get("planner_type") == "mock"

    def test_retry_num_stored(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config(), retry_num=3)
        assert traj.metadata.get("retry_num") == 3


class TestMockPlannerEdgeCases:
    """Edge cases and boundary conditions for the mock planner."""

    def test_single_object_no_crash(self):
        """Only one object in registry — planner should use a synthetic goal."""
        source = ObjectInfo(
            label="cube",
            centroid_3d=np.array([0.4, 0.1, 0.75]),
            bbox_3d_corners=np.zeros((8, 3)),
            orientation=np.eye(3),
            bbox_2d_per_camera={},
        )
        scene = SceneContext(
            point_cloud=np.zeros((8, 3)),
            object_registry={"cube": source},
            camera_images={},
            camera_matrices={},
            task_instruction="push the cube",
            initial_eef_pose=np.array([0.5, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0]),
        )
        traj = generate_mock_trajectory(scene, Path1Config())
        assert len(traj.actions) >= 3
        assert len(traj.grasp_indices) > 0

    def test_respects_max_action_steps(self):
        scene = _make_scene()
        config = Path1Config(max_action_steps=10)
        traj = generate_mock_trajectory(scene, config)
        assert len(traj.actions) <= 10, (
            f"Expected <= 10 actions, got {len(traj.actions)}"
        )

    def test_actions_are_numpy_arrays(self):
        scene = _make_scene()
        traj = generate_mock_trajectory(scene, Path1Config())
        for i, action in enumerate(traj.actions):
            assert isinstance(action, np.ndarray), (
                f"Action {i} is not a numpy array: {type(action)}"
            )

    def test_deterministic_output(self):
        """Same input always yields same trajectory."""
        scene = _make_scene()
        config = Path1Config()
        traj1 = generate_mock_trajectory(scene, config)
        traj2 = generate_mock_trajectory(scene, config)
        assert len(traj1.actions) == len(traj2.actions)
        for a1, a2 in zip(traj1.actions, traj2.actions):
            np.testing.assert_array_equal(a1, a2)


class TestPlannerAgent:
    """PlannerAgent delegation tests."""

    def test_mock_mode_returns_trajectory(self):
        scene = _make_scene()
        agent = PlannerAgent()
        config = Path1Config(use_mock_planner=True)
        traj = agent.run(scene, config)
        assert isinstance(traj, Trajectory)

    def test_mock_mode_structure(self):
        scene = _make_scene()
        agent = PlannerAgent()
        config = Path1Config(use_mock_planner=True)
        traj = agent.run(scene, config)
        assert len(traj.actions) >= 3
        for action in traj.actions:
            assert action.shape == (7,)
        assert len(traj.grasp_indices) > 0

    def test_llm_mode_raises_without_api_key(self):
        """LLM mode should raise EnvironmentError when no API key is available."""
        from unittest.mock import patch
        scene = _make_scene()
        agent = PlannerAgent()
        config = Path1Config(use_mock_planner=False)
        # Patch _load_api_key to simulate a missing key regardless of env/.env
        with patch(
            "path1_execution.planning.llm_planner._load_api_key",
            side_effect=EnvironmentError("OPENAI_API_KEY not set"),
        ):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                agent.run(scene, config)

    def test_feedback_ignored_in_mock_mode(self):
        """Passing feedback string should not cause errors in mock mode."""
        scene = _make_scene()
        agent = PlannerAgent()
        config = Path1Config(use_mock_planner=True)
        traj = agent.run(
            scene, config,
            feedback="Trajectory did not satisfy verification.",
            retry_num=2,
        )
        assert isinstance(traj, Trajectory)
        assert traj.metadata["retry_num"] == 2

    def test_retry_num_propagated(self):
        scene = _make_scene()
        agent = PlannerAgent()
        config = Path1Config(use_mock_planner=True)
        traj = agent.run(scene, config, retry_num=4)
        assert traj.metadata["retry_num"] == 4
