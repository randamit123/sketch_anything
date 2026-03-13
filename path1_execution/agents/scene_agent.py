"""SceneAgent — Sub-agent 1.

Orchestrates Stage 1 (3D reconstruction) and Stage 2 (object extraction)
to produce a SceneContext for the planner.
"""

from __future__ import annotations

import logging

import numpy as np

from path1_execution.config import Path1Config, SceneContext
from path1_execution.scene.reconstruction import reconstruct_scene
from path1_execution.scene.object_extraction import extract_objects

logger = logging.getLogger(__name__)


class SceneAgent:
    """Builds a SceneContext from a LIBERO environment."""

    def run(self, env, task_instruction: str, config: Path1Config, initial_state=None) -> SceneContext:
        """Execute scene reconstruction and object extraction.

        Resets the environment to get a clean initial state, captures
        camera images and matrices, builds the point cloud, and extracts
        task-relevant objects with 3D info.

        Args:
            env: LIBERO OffScreenRenderEnv instance.
            task_instruction: Natural language task description.
            config: Path1Config instance.
            initial_state: Demo initial state to restore after env.reset().

        Returns:
            SceneContext populated with all scene data.
        """
        # Reset to get a fresh initial state and retrieve the initial EEF pose.
        # IMPORTANT: env.reset() may create a new MjSim object, so we must
        # capture env.sim AFTER the reset call, not before.
        obs = env.reset()
        if initial_state is not None:
            from sketch_anything.tools.run_pipeline import set_env_state
            set_env_state(env, initial_state)
            obs, _, _, _ = env.step(np.zeros(7))
        sim = env.sim

        # Extract initial EEF pose: [pos(3) + quat(4)] = shape (7,)
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))   # shape (3,)
        eef_quat = obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))  # shape (4,)
        initial_eef_pose = np.concatenate([eef_pos, eef_quat])  # shape (7,)

        logger.info(f"Initial EEF pose: {initial_eef_pose.round(4).tolist()}")

        # Stage 1: Reconstruct scene (images, camera matrices, point cloud)
        scene_data = reconstruct_scene(env, sim, config)

        # Stage 2: Extract task-relevant objects
        object_registry = extract_objects(env, sim, task_instruction, config)

        return SceneContext(
            point_cloud=scene_data["point_cloud"],
            object_registry=object_registry,
            camera_images=scene_data["camera_images"],
            camera_matrices=scene_data["camera_matrices"],
            task_instruction=task_instruction,
            initial_eef_pose=initial_eef_pose,
        )
