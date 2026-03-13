"""LLM-based trajectory planner using OpenAI o3.

Uses the OpenAI Python SDK to call the o3 model and parse the returned
Python function into a list of (7,) action arrays.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import numpy as np

from path1_execution.config import Path1Config, SceneContext, Trajectory
from path1_execution.planning.prompts import (
    FEEDBACK_SECTION_TEMPLATE,
    SYSTEM_PROMPT,
    TRAJECTORY_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _load_api_key() -> str:
    """Load OpenAI API key from env var or path1_execution/.env file."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Set it in the environment or in path1_execution/.env"
        )
    return key


def generate_llm_trajectory(
    scene_context: SceneContext,
    config: Path1Config,
    feedback: Optional[str] = None,
    retry_num: int = 0,
) -> Trajectory:
    """Generate a trajectory using the OpenAI o3 model.

    Args:
        scene_context: Grounded 3D scene representation.
        config: Pipeline configuration (uses config.llm_model and config.max_action_steps).
        feedback: Optional corrective feedback from a prior failed verification attempt.
        retry_num: Current retry attempt number, stored in trajectory metadata.

    Returns:
        A Trajectory with actions, sub_goals, grasp_indices, and metadata.
    """
    import openai

    client = openai.OpenAI(api_key=_load_api_key())

    # Serialize object registry for the prompt
    registry_json: dict = {}
    for label, obj_info in scene_context.object_registry.items():
        centroid = obj_info.centroid_3d.tolist()
        corners = obj_info.bbox_3d_corners
        extents = (corners.max(axis=0) - corners.min(axis=0)).tolist()
        registry_json[label] = {
            "centroid_3d": [round(v, 4) for v in centroid],
            "bbox_extents": [round(v, 4) for v in extents],
        }

    eef_pose = scene_context.initial_eef_pose.tolist()

    feedback_section = (
        FEEDBACK_SECTION_TEMPLATE.format(feedback=feedback) if feedback else ""
    )

    prompt = TRAJECTORY_PROMPT_TEMPLATE.format(
        task_instruction=scene_context.task_instruction,
        object_registry=json.dumps(registry_json, indent=2),
        eef_pose=eef_pose,
        max_steps=config.max_action_steps,
        feedback_section=feedback_section,
    )

    logger.info(
        "Calling %s for trajectory planning (attempt %d)...",
        config.llm_model,
        retry_num + 1,
    )

    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=8192,
    )

    raw = response.choices[0].message.content or ""
    logger.info("o3 response received (%d chars)", len(raw))
    logger.debug("o3 raw response:\n%s", raw)

    actions = _parse_trajectory(raw, config)

    # Detect grasp indices: gripper transitions from open (>0) to closed (<0)
    grasp_indices: list[int] = []
    for i in range(1, len(actions)):
        if actions[i - 1][6] > 0 and actions[i][6] < 0:
            grasp_indices.append(i)
    if not grasp_indices and actions:
        # Fallback: mark midpoint as grasp
        grasp_indices = [len(actions) // 2]

    sub_goals = _infer_sub_goals(actions)

    return Trajectory(
        actions=actions,
        sub_goals=sub_goals,
        grasp_indices=grasp_indices,
        metadata={
            "planner_type": "o3",
            "model": config.llm_model,
            "retry_num": retry_num,
            "raw_response_len": len(raw),
        },
    )


def _parse_trajectory(raw: str, config: Path1Config) -> list:
    """Parse o3 output into a list of (7,) action arrays.

    Three strategies, tried in order:
    1. Execute ``def get_trajectory():`` Python function from the response.
    2. Parse a raw JSON array of 7-element lists.
    3. Extract any sequence of 7-number groups from the text.

    Returns an empty list only if all strategies fail.
    """
    # Strategy 1: find and exec get_trajectory function
    if "def get_trajectory" in raw:
        try:
            code = raw
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            # Provide a namespace that supports numpy imports inside the generated code
            import builtins
            namespace: dict = {"np": np, "numpy": np, "__builtins__": builtins}
            exec(code, namespace)  # noqa: S102
            result = namespace["get_trajectory"]()
            actions = [np.array(a, dtype=float) for a in result]
            actions = [
                a[:7] if len(a) >= 7 else np.pad(a, (0, 7 - len(a)))
                for a in actions
            ]
            actions = actions[: config.max_action_steps]
            logger.info("Parsed trajectory via exec: %d actions", len(actions))
            return actions
        except Exception as exc:
            logger.warning("exec strategy failed: %s", exc)

    # Strategy 2: JSON array of 7-element lists
    json_match = re.search(r"\[[\s\S]*?\]", raw)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if (
                isinstance(data, list)
                and data
                and isinstance(data[0], list)
                and len(data[0]) == 7
            ):
                actions = [
                    np.array(a, dtype=float) for a in data[: config.max_action_steps]
                ]
                logger.info("Parsed trajectory via JSON: %d actions", len(actions))
                return actions
        except Exception:
            pass

    # Strategy 3: any sequence of 7-number groups
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if len(numbers) >= 7:
        try:
            vals = [float(n) for n in numbers]
            actions = []
            for i in range(0, min(len(vals) - 6, config.max_action_steps * 7), 7):
                a = np.array(vals[i : i + 7], dtype=float)
                actions.append(a)
            if actions:
                logger.info(
                    "Parsed trajectory via number extraction: %d actions", len(actions)
                )
                return actions
        except Exception:
            pass

    logger.error(
        "Could not parse trajectory from o3 response. Returning empty trajectory."
    )
    return []


def _infer_sub_goals(actions: list) -> list:
    """Infer sub-goal labels from gripper state transitions in the action list."""
    if not actions:
        return ["unknown"]
    goals = ["approach"]
    for i in range(1, len(actions)):
        if actions[i - 1][6] > 0 and actions[i][6] < 0:
            goals.append("grasp")
        elif actions[i - 1][6] < 0 and actions[i][6] > 0:
            goals.append("release")
    if len(goals) == 1:
        goals.append("transport")
    return goals
