"""Deterministic mock planner for Path 1.

Generates a fixed three-phase pick-and-place trajectory without any API calls.
Used for end-to-end pipeline testing before the LLM planner is active.
"""

import logging

import numpy as np
from typing import List

from path1_execution.config import Path1Config, SceneContext, Trajectory

logger = logging.getLogger(__name__)


def _interpolate(start: np.ndarray, end: np.ndarray, n_steps: int) -> List[np.ndarray]:
    """Linearly interpolate between start and end positions over n_steps steps.

    Returns n_steps points, NOT including start, including end.
    """
    points = []
    for i in range(1, n_steps + 1):
        t = i / n_steps
        points.append(start + t * (end - start))
    return points


def generate_mock_trajectory(
    scene_context: SceneContext,
    config: Path1Config,
    retry_num: int = 0,
) -> Trajectory:
    """Generate a deterministic three-phase pick-and-place trajectory.

    Phases:
      1. Pre-grasp approach: move EEF from initial pose to 15 cm above source object.
      2. Grasp: descend to source centroid and close gripper.
      3. Transport:
         a. Lift back to 15 cm above source.
         b. Move horizontally to 10 cm above goal.
         c. Descend to goal centroid and open gripper.

    Args:
        scene_context: Scene information including object registry and initial EEF pose.
        config: Pipeline configuration (max_action_steps, etc.).
        retry_num: Current retry attempt number (stored in metadata).

    Returns:
        A Trajectory with all actions as (7,) arrays [x, y, z, 0, 0, 0, gripper].
    """
    registry = scene_context.object_registry

    # Sort for determinism
    sorted_labels = sorted(registry.keys())

    if not sorted_labels:
        # No task-relevant objects found — use a fallback position in front of the robot
        logger.warning(
            "Object registry is empty; using fallback centroid for mock planner."
        )
        initial_pos = scene_context.initial_eef_pose[:3].copy()
        source_centroid = initial_pos + np.array([0.0, 0.0, -0.20])  # 20 cm below EEF
    else:
        source_obj = registry[sorted_labels[0]]
        source_centroid = source_obj.centroid_3d.copy()

    if len(sorted_labels) >= 2:
        goal_obj = registry[sorted_labels[1]]
        goal_centroid = goal_obj.centroid_3d.copy()
    else:
        # No second object — place 30 cm to the right of source
        goal_centroid = source_centroid + np.array([0.3, 0.0, 0.0])

    initial_pos = scene_context.initial_eef_pose[:3].copy()

    # ------------------------------------------------------------------ #
    # Phase 1 — Pre-grasp approach (10 steps, gripper open)
    # ------------------------------------------------------------------ #
    approach_target = source_centroid + np.array([0.0, 0.0, 0.15])
    phase1_positions = _interpolate(initial_pos, approach_target, n_steps=10)
    phase1_actions = [
        np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, 1.0], dtype=float)
        for p in phase1_positions
    ]

    # ------------------------------------------------------------------ #
    # Phase 2 — Grasp (5 steps, gripper closes on last step)
    # ------------------------------------------------------------------ #
    grasp_positions = _interpolate(approach_target, source_centroid, n_steps=5)
    phase2_actions = []
    for i, p in enumerate(grasp_positions):
        gripper = -1.0  # close throughout the descent
        phase2_actions.append(
            np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, gripper], dtype=float)
        )

    # ------------------------------------------------------------------ #
    # Phase 3a — Lift (10 steps, gripper closed)
    # ------------------------------------------------------------------ #
    lift_target = source_centroid + np.array([0.0, 0.0, 0.15])
    phase3a_positions = _interpolate(source_centroid, lift_target, n_steps=10)
    phase3a_actions = [
        np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, -1.0], dtype=float)
        for p in phase3a_positions
    ]

    # ------------------------------------------------------------------ #
    # Phase 3b — Move to goal (10 steps, gripper closed)
    # ------------------------------------------------------------------ #
    goal_approach = goal_centroid + np.array([0.0, 0.0, 0.10])
    phase3b_positions = _interpolate(lift_target, goal_approach, n_steps=10)
    phase3b_actions = [
        np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, -1.0], dtype=float)
        for p in phase3b_positions
    ]

    # ------------------------------------------------------------------ #
    # Phase 3c — Descend and release (10 steps, gripper opens)
    # ------------------------------------------------------------------ #
    phase3c_positions = _interpolate(goal_approach, goal_centroid, n_steps=10)
    phase3c_actions = []
    for i, p in enumerate(phase3c_positions):
        gripper = 1.0  # open throughout descent to goal
        phase3c_actions.append(
            np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, gripper], dtype=float)
        )

    # ------------------------------------------------------------------ #
    # Assemble and cap
    # ------------------------------------------------------------------ #
    all_actions = (
        phase1_actions
        + phase2_actions
        + phase3a_actions
        + phase3b_actions
        + phase3c_actions
    )

    # Identify grasp index (last step of phase 2) and release index
    grasp_index = len(phase1_actions) + len(phase2_actions) - 1
    release_index = len(all_actions) - 1

    # Cap at max_action_steps
    if len(all_actions) > config.max_action_steps:
        all_actions = all_actions[: config.max_action_steps]
        # Clamp indices to valid range
        grasp_index = min(grasp_index, len(all_actions) - 1)
        release_index = min(release_index, len(all_actions) - 1)

    return Trajectory(
        actions=all_actions,
        sub_goals=["approach", "grasp", "transport"],
        grasp_indices=[grasp_index],
        metadata={
            "planner_type": "mock",
            "retry_num": retry_num,
            "source_label": sorted_labels[0] if sorted_labels else "fallback",
            "goal_label": sorted_labels[1] if len(sorted_labels) >= 2 else None,
            "release_index": release_index,
        },
    )
