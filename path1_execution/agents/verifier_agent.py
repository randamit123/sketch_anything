"""VerifierAgent: orchestrates 2D projection and trajectory verification."""

from __future__ import annotations

import logging

from path1_execution.config import (
    Path1Config,
    SceneContext,
    Trajectory,
    VerifyResult,
)
from path1_execution.projection.trajectory_projector import project_trajectory
from path1_execution.verification.verifier import verify_trajectory

logger = logging.getLogger(__name__)


class VerifierAgent:
    """Sub-agent that projects trajectory to 2D and verifies it via sim rollout."""

    def run(
        self,
        trajectory: Trajectory,
        scene_context: SceneContext,
        env,
        config: Path1Config,
        initial_state=None,
    ) -> VerifyResult:
        """Project trajectory to 2D sketches, then verify via sim.

        Args:
            trajectory: Planned trajectory from PlannerAgent.
            scene_context: Scene context with camera matrices and images.
            env: LIBERO environment for trial execution.
            config: Pipeline configuration.
            initial_state: Demo initial state to restore after env.reset().

        Returns:
            VerifyResult with verified flag, metrics, feedback (if failed), and sketch.
        """
        logger.info("VerifierAgent: projecting trajectory to 2D...")
        planned_sketch = project_trajectory(trajectory, scene_context, config)

        logger.info("VerifierAgent: verifying trajectory via sim rollout...")
        result = verify_trajectory(trajectory, scene_context, env, config, planned_sketch, initial_state=initial_state)

        return result
