"""Stage 6b: Generate corrective feedback for the planner on verification failure.

In mock mode: returns a static template string (no API call).
In o3 mode: builds a specific feedback string from the VerifyResult fields.
"""

from __future__ import annotations

from typing import Optional

from path1_execution.config import Path1Config

_MOCK_FEEDBACK = (
    "Trajectory did not satisfy verification. Re-examine target object "
    "position and adjust approach height and goal coordinates."
)


def get_feedback(config: Path1Config, verify_result=None) -> str:
    """Return corrective feedback string for injection into the planner prompt.

    Args:
        config: Pipeline configuration. ``use_mock_planner`` controls which
                branch runs.
        verify_result: Optional VerifyResult (or duck-typed object with
                       ``distance_to_goal``, ``sim_success``, and
                       ``geometric_success`` attributes). Used in o3 mode to
                       produce specific failure analysis.

    Returns:
        A feedback string describing the failure and suggested corrections.
    """
    if config.use_mock_planner:
        return _MOCK_FEEDBACK

    # o3 mode: build specific feedback from verify_result if available
    if verify_result is not None:
        dist: Optional[float] = getattr(verify_result, "distance_to_goal", None)
        sim_ok: bool = getattr(verify_result, "sim_success", False)
        geo_ok: bool = getattr(verify_result, "geometric_success", False)

        parts: list[str] = []
        if not sim_ok:
            parts.append(
                "The simulator did not report task completion (reward=0). "
                "Check that the gripper actually contacted and grasped the object."
            )
        if not geo_ok and dist is not None:
            parts.append(
                f"End-effector ended {dist:.3f} m from the goal centroid "
                f"(threshold: {config.goal_threshold_m} m). "
                "Adjust the final target position."
            )

        if parts:
            return (
                "Trajectory failed verification: "
                + " ".join(parts)
                + " Re-examine object centroids from the registry and revise "
                "approach height and goal coordinates."
            )

    return (
        "Trajectory failed verification. "
        "Adjust approach height, gripper timing, and goal coordinates."
    )
