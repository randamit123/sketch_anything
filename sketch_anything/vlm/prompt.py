"""VLM prompt template for sketch primitive generation.

The template instructs Qwen2.5-VL to output structured JSON conforming to
the SketchPrimitives schema. Double braces ``{{`` / ``}}`` are used to
escape literal JSON braces in the f-string / .format() calls.
"""

from __future__ import annotations

from typing import Dict

PROMPT_TEMPLATE = """\
You are a robot motion sketch annotator. Given an image of a tabletop scene \
with detected objects and a manipulation task instruction, you must specify \
drawing primitives that visualize the intended end-effector motion path.

Your output will be rendered as a sketch overlay on the scene image to verify \
that the robot's planned trajectory matches the task intent.

## Detected Objects

The following objects have been detected in the scene. Use these object IDs \
when specifying object-relative positions.

{object_registry_formatted}

## Primitive Vocabulary

You may ONLY use these three primitive types:

### ARROW
Represents directed end-effector motion from one point to another.
- type: "arrow" (required)
- start: Position - motion origin point (required)
- end: Position - motion destination point (required)
- waypoints: List of Position - intermediate points for curved paths (optional, default [])
- step: Integer >= 1 - temporal ordering (required)

### CIRCLE
Marks a point of interest in the scene.
- type: "circle" (required)
- center: Position - center of the circle (required)
- radius: Float in [0.02, 0.06] - circle size as proportion of image (required)
- purpose: One of "grasp_point", "release_point", "contact", "rotation_pivot", "target_location" (required)
- step: Integer >= 1 - temporal ordering (required)

### GRIPPER
Indicates a gripper state change at a specific location.
- type: "gripper" (required)
- position: Position - location of state change (required)
- action: "open" or "close" (required)
- step: Integer >= 1 - temporal ordering (required)

## Position Specification

### Absolute Position (for free-space waypoints only)
{{"type": "absolute", "coords": [x, y]}}
- Coordinates normalized to [0.0, 1.0], origin top-left

### Object-Relative Position (PREFERRED for all object interactions)
{{"type": "object_relative", "object_id": "...", "anchor": "...", "offset": [dx, dy]}}
- object_id: Must match an ID from Detected Objects
- anchor: One of "center", "top", "bottom", "left", "right", "top_left", \
"top_right", "bottom_left", "bottom_right"
- offset: Optional displacement, default [0.0, 0.0]

IMPORTANT: Use object-relative positions whenever the location involves a \
detected object.

## Output Format

Respond with ONLY a valid JSON object:
{{"primitives": [...]}}

Do not include any text before or after the JSON.

## Temporal Ordering Guidelines

- Step 1: Initial approach and marking the grasp point
- Step 2: Gripper close (grasping)
- Step 3: Transport motion (lifting and moving)
- Step 4: Gripper open (releasing) and marking the release point

Multiple primitives may share the same step number if they occur simultaneously.

## Task

Instruction: {task_instruction}

Analyze the scene and the task instruction. Specify the primitives needed to \
visualize the complete end-effector motion path, including:
1. Where the gripper must go (approach motion)
2. What it must grasp (grasp point and gripper close)
3. Where it must transport the object (transport motion)
4. Where it must release (release point and gripper open)\
"""


def format_object_registry(registry: Dict[str, dict]) -> str:
    """Format an object registry for insertion into the prompt.

    Args:
        registry: Mapping of object_id -> {id, label, bbox, center}.

    Returns:
        Formatted string with one entry per object.
    """
    lines = []
    for obj_id, obj_data in registry.items():
        bbox = obj_data["bbox"]
        center = obj_data["center"]
        lines.append(
            f'- id: "{obj_id}"\n'
            f'  label: "{obj_data["label"]}"\n'
            f"  bbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]\n"
            f"  center: [{center[0]:.3f}, {center[1]:.3f}]"
        )
    return "\n\n".join(lines)


def format_prompt(
    object_registry: Dict[str, dict],
    task_instruction: str,
) -> str:
    """Build the complete VLM prompt.

    Args:
        object_registry: Single-view object registry.
        task_instruction: Natural language task description.

    Returns:
        Formatted prompt string ready for VLM input.
    """
    registry_formatted = format_object_registry(object_registry)
    return PROMPT_TEMPLATE.format(
        object_registry_formatted=registry_formatted,
        task_instruction=task_instruction,
    )
