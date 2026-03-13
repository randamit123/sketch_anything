"""Orchestrator: coordinates sub-agents and runs the retry loop."""
import json
import logging
import os
from typing import Optional

from path1_execution.config import OrchestratorResult, Path1Config, VerifyResult, Trajectory
from path1_execution.agents.scene_agent import SceneAgent
from path1_execution.agents.planner_agent import PlannerAgent
from path1_execution.agents.verifier_agent import VerifierAgent
from path1_execution.execution.robot_executor import execute_trajectory, save_rollout_hdf5

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates scene extraction, planning, verification, and execution with retry loop."""

    def __init__(self):
        self.scene_agent = SceneAgent()
        self.planner_agent = PlannerAgent()
        self.verifier_agent = VerifierAgent()

    def run(self, hdf5_path: str, task_instruction: str, config: Path1Config, demo_index: int = 0) -> OrchestratorResult:
        """Run the full pipeline with retry loop.

        Args:
            hdf5_path: Path to LIBERO demo HDF5 file.
            task_instruction: Natural language task description.
            config: Pipeline configuration.
            demo_index: Which demo episode to load initial state from.

        Returns:
            OrchestratorResult with success flag, attempt count, and final results.
        """
        from sketch_anything.tools.run_pipeline import load_libero_env, load_demo_actions, set_env_state

        env, task_from_hdf5, _ = load_libero_env(hdf5_path, camera_names=config.camera_names)

        # Use provided task_instruction or fall back to what's in the HDF5
        if task_instruction is None:
            task_instruction = task_from_hdf5

        # Load demo initial state and set env to the correct starting configuration
        initial_state, _ = load_demo_actions(hdf5_path, demo_index=demo_index)
        set_env_state(env, initial_state)
        logger.info("Loaded initial state from demo %d", demo_index)

        scene_context = self.scene_agent.run(env, task_instruction, config, initial_state=initial_state)

        os.makedirs(config.output_dir, exist_ok=True)

        feedback: Optional[str] = None
        trajectory: Optional[Trajectory] = None
        verify_result: Optional[VerifyResult] = None

        for attempt in range(config.max_retries):
            logger.info("Attempt %d/%d", attempt + 1, config.max_retries)

            trajectory = self.planner_agent.run(
                scene_context, config, feedback=feedback, retry_num=attempt
            )
            verify_result = self.verifier_agent.run(
                trajectory, scene_context, env, config, initial_state=initial_state
            )

            self._log_attempt(attempt, verify_result, config.output_dir)

            if verify_result.verified:
                logger.info("Trajectory verified on attempt %d!", attempt + 1)
                obs_list, acts, rews, dn = execute_trajectory(
                    trajectory, env, config, initial_state=initial_state
                )
                hdf5_out = save_rollout_hdf5(obs_list, acts, rews, dn, config.output_dir)
                logger.info("Saved demo to %s", hdf5_out)
                return OrchestratorResult(
                    success=True,
                    attempts=attempt + 1,
                    final_trajectory=trajectory,
                    final_verify_result=verify_result,
                )

            feedback = verify_result.feedback
            logger.info("Attempt %d failed. Feedback: %s", attempt + 1, feedback)

        logger.warning("All %d attempts exhausted without success.", config.max_retries)
        return OrchestratorResult(
            success=False,
            attempts=config.max_retries,
            final_trajectory=trajectory,
            final_verify_result=verify_result,
        )

    def _log_attempt(self, attempt: int, verify_result: VerifyResult, output_dir: str):
        """Append one JSON line per attempt to attempts.jsonl."""
        log_path = os.path.join(output_dir, "attempts.jsonl")
        final_eef = verify_result.final_eef_pose[:3].tolist() if verify_result.final_eef_pose is not None else None
        entry = {
            "attempt": attempt + 1,
            "sim_success": bool(verify_result.sim_success),
            "geometric_success": bool(verify_result.geometric_success),
            "distance_to_goal": float(verify_result.distance_to_goal),
            "final_eef_pos": [round(v, 4) for v in final_eef] if final_eef else None,
            "feedback": verify_result.feedback,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
