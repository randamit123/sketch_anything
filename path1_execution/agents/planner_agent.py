"""PlannerAgent: selects and invokes the appropriate trajectory planner."""

from path1_execution.config import Path1Config, SceneContext, Trajectory
from path1_execution.planning.mock_planner import generate_mock_trajectory


class PlannerAgent:
    """Selects the mock or LLM planner based on config and returns a Trajectory."""

    def run(
        self,
        scene_context: SceneContext,
        config: Path1Config,
        feedback: str = None,
        retry_num: int = 0,
    ) -> Trajectory:
        """Generate a trajectory for the given scene context.

        Args:
            scene_context: Grounded 3D scene representation.
            config: Pipeline configuration.
            feedback: Optional corrective feedback from a failed verification attempt.
                      Passed to the LLM planner when active; ignored by the mock planner.
            retry_num: Current retry attempt number (stored in trajectory metadata).

        Returns:
            A Trajectory object ready for projection and verification.
        """
        if config.use_mock_planner:
            return generate_mock_trajectory(scene_context, config, retry_num=retry_num)
        else:
            from path1_execution.planning.llm_planner import generate_llm_trajectory

            return generate_llm_trajectory(
                scene_context,
                config,
                feedback=feedback,
                retry_num=retry_num,
            )
