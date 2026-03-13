"""Stage 5: Verify a planned trajectory against geometric and simulator success signals."""

from __future__ import annotations

import logging

import numpy as np

from path1_execution.config import (
    Path1Config,
    PlannedSketch,
    SceneContext,
    Trajectory,
    VerifyResult,
)
from path1_execution.execution.robot_executor import run_trial
from path1_execution.verification.llm_feedback import get_feedback

logger = logging.getLogger(__name__)


def _infer_goal_label(task_instruction: str, labels: list) -> str | None:
    """Parse 'put X in/on Y' to extract the goal label Y.

    Checks for common placement prepositions and returns the first label that
    appears immediately after them in the task string.
    """
    task_lower = task_instruction.lower()
    for prep in [
        " in the ", " on the ", " on top of ", " onto ", " into ", " inside ",
        " to the front of ", " to the back of ", " to the side of ",
        " toward the ", " to the ",
    ]:
        idx = task_lower.find(prep)
        if idx < 0:
            continue
        suffix = task_lower[idx + len(prep):]
        for label in labels:
            if suffix.startswith(label.lower()):
                logger.debug("Inferred goal label '%s' from task instruction", label)
                return label
    return None


def _get_goal_centroid(
    trajectory: Trajectory,
    scene_context: SceneContext,
) -> np.ndarray:
    """Determine goal position for the geometric distance check.

    Returns the TOP SURFACE of the goal object (centroid_x, centroid_y,
    bbox_top_z) rather than the body centroid.  For placement tasks the EEF
    ends up above the goal surface, so using the centroid would make the
    threshold too tight and cause false-negatives.

    Priority:
    1. trajectory.metadata["goal_label"] if set and in registry
    2. Task-instruction parsing: "put X in/on Y" → Y is goal
    3. Last object alphabetically in registry (fallback heuristic)
    4. Zero vector if registry is empty
    """
    registry = scene_context.object_registry

    def _top_surface(obj_info) -> np.ndarray:
        c = obj_info.centroid_3d
        corners = obj_info.bbox_3d_corners
        top_z = float(corners[:, 2].max())
        return np.array([c[0], c[1], top_z], dtype=np.float64)

    # Priority 1: explicit goal_label in metadata (set by LLM or mock planner)
    goal_label = trajectory.metadata.get("goal_label")
    if goal_label and goal_label in registry:
        logger.debug("Using goal_label from trajectory metadata: %s", goal_label)
        return _top_surface(registry[goal_label])

    # Priority 2: parse task instruction for "put X in/on Y"
    labels = list(registry.keys())
    inferred = _infer_goal_label(scene_context.task_instruction, labels)
    if inferred and inferred in registry:
        return _top_surface(registry[inferred])

    # Priority 3: last object alphabetically (heuristic: goal tends to sort last
    #             for tasks like "put bowl on plate" — but NOT for "cream cheese in bowl")
    if registry:
        sorted_labels = sorted(registry.keys())
        return _top_surface(registry[sorted_labels[-1]])

    # Priority 4: fallback
    logger.warning("No objects in registry; using zero vector as goal centroid")
    return np.zeros(3)


def _get_pushed_object_pos(env, scene_context: SceneContext) -> np.ndarray | None:
    """After a push trial, read the pushed object's current 3D position from sim.

    Identifies the source (pushed) object as the first registry entry whose label
    appears before any preposition in the task instruction.
    """
    from sketch_anything.libero_utils.env import get_object_bbox_3d

    task_lower = scene_context.task_instruction.lower()
    # "push the PLATE to the front of the stove" → source = "plate"
    # Find source: it's the object label that appears after "push" and before any prep
    for prep in [" to ", " toward ", " into ", " onto "]:
        idx = task_lower.find(prep)
        if idx > 0:
            source_part = task_lower[len("push "):idx]
            break
    else:
        source_part = task_lower[len("push "):]

    sim = env.sim
    for label, obj_info in scene_context.object_registry.items():
        if label.lower() in source_part:
            # Found the pushed object — look up its body in the registry
            # Try to find the mujoco body name
            try:
                # obj_info doesn't store body name, so search sim for matching centroid
                for body_name in sim.model.body_names:
                    try:
                        corners = get_object_bbox_3d(sim, body_name)
                        centroid = corners.mean(axis=0)
                        # Match by initial centroid proximity
                        if np.linalg.norm(centroid[:2] - obj_info.centroid_3d[:2]) < 0.15:
                            if label.lower().replace(" ", "_") in body_name.lower() or \
                               any(kw in body_name.lower() for kw in label.lower().split()):
                                logger.debug("Pushed object '%s' matched to body '%s' at %s",
                                             label, body_name, centroid.round(4).tolist())
                                return centroid
                    except Exception:
                        continue
            except Exception as e:
                logger.warning("Could not find pushed object body for '%s': %s", label, e)
    return None


def verify_trajectory(
    trajectory: Trajectory,
    scene_context: SceneContext,
    env,
    config: Path1Config,
    planned_sketch: PlannedSketch,
    initial_state=None,
) -> VerifyResult:
    """Run a trial rollout and evaluate success.

    Two independent success signals must both pass:
    - sim_success: LIBERO env reports done=True with reward > 0
    - geometric_success: final EEF position within config.goal_threshold_m of goal centroid

    If verification fails, corrective feedback is generated via llm_feedback.get_feedback.

    Args:
        trajectory: Planned trajectory.
        scene_context: Scene context including object registry.
        env: LIBERO environment.
        config: Pipeline configuration.
        planned_sketch: 2D projection of the trajectory.

    Returns:
        VerifyResult with success flags, distance, feedback (if failed), and sketch.
    """
    # Execute trial rollout
    final_obs, sim_success, total_reward = run_trial(trajectory, env, config, initial_state=initial_state)

    # Extract final EEF pose
    final_eef_pos = np.asarray(
        final_obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64
    )
    final_eef_quat = np.asarray(
        final_obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0])),
        dtype=np.float64,
    )
    final_eef_pose = np.concatenate([final_eef_pos, final_eef_quat])  # (7,)

    # Geometric distance check
    goal_centroid = _get_goal_centroid(trajectory, scene_context)

    # For PUSH tasks, measure the pushed object's final position instead of EEF.
    # The EEF ends at the push endpoint, which may overshoot the object's position.
    task_lower = scene_context.task_instruction.lower()
    is_push_task = task_lower.startswith("push ")
    if is_push_task:
        pushed_pos = _get_pushed_object_pos(env, scene_context)
        if pushed_pos is not None:
            distance = float(np.linalg.norm(pushed_pos[:2] - goal_centroid[:2]))
            logger.info("Push task: measuring pushed object pos %s vs goal %s (2D dist=%.4fm)",
                        pushed_pos.round(4).tolist(), goal_centroid.round(4).tolist(), distance)
        else:
            distance = float(np.linalg.norm(final_eef_pos - goal_centroid))
    else:
        distance = float(np.linalg.norm(final_eef_pos - goal_centroid))
    geometric_success = distance < config.goal_threshold_m

    # sim_success is the authoritative LIBERO ground truth.
    # geometric_success is a secondary signal — useful when sim fires but EEF ends
    # above the surface (ON tasks) or when sim doesn't fire but robot is clearly done.
    # Use sim_success OR geometric_success to handle both cases:
    #   - "IN" container tasks: sim fires (reward=1), EEF is below rim → geo fails → use sim
    #   - placement tasks with imprecise release: EEF near goal → geo catches it
    # This can cause false positives when the EEF happens to be near the goal object
    # without completing the task; those are filtered by sim_success=False in practice
    # because incomplete tasks leave the EEF far from the goal.
    verified = sim_success or geometric_success

    logger.info(
        "Verification: sim_success=%s, geometric_success=%s (dist=%.4fm), verified=%s",
        sim_success,
        geometric_success,
        distance,
        verified,
    )

    # Generate feedback if not verified
    feedback = None
    if not verified:
        # Build a lightweight struct for feedback context (avoids circular import)
        class _PartialResult:
            pass

        partial = _PartialResult()
        partial.distance_to_goal = distance
        partial.sim_success = sim_success
        partial.geometric_success = geometric_success
        feedback = get_feedback(config, verify_result=partial)

    return VerifyResult(
        verified=verified,
        sim_success=sim_success,
        geometric_success=geometric_success,
        final_eef_pose=final_eef_pose,
        distance_to_goal=distance,
        feedback=feedback,
        planned_sketch=planned_sketch,
    )
