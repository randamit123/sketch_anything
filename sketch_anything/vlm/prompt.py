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

The following objects have been detected in the scene with their bounding boxes. \
You MUST use these exact object IDs in your object_relative positions.

{object_registry_formatted}

## Primitive Types

### ARROW - directed motion between two DIFFERENT locations
- type: "arrow"
- start: Position (where motion begins)
- end: Position (where motion ends — MUST be different from start)
- waypoints: list of Position (optional, default [])
- step: integer >= 1

### CIRCLE - marks a key location
- type: "circle"
- center: Position
- radius: float in [0.02, 0.06]
- purpose: one of "grasp_point", "release_point", "contact", "rotation_pivot", "target_location"
- step: integer >= 1

### GRIPPER - gripper state change
- type: "gripper"
- position: Position
- action: "open" or "close"
- step: integer >= 1

## Position Types

You MUST use "object_relative" positions when referencing any detected object. \
Only use "absolute" for free-space waypoints not near any object.

Object-relative position (REQUIRED for any location on/near an object):
{{"type": "object_relative", "object_id": "<id from above>", "anchor": "<anchor>", "offset": [dx, dy]}}

Anchors: "center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"

Absolute position (ONLY for free-space waypoints):
{{"type": "absolute", "coords": [x, y]}}
Coordinates in [0.0, 1.0], origin at top-left.

## CRITICAL RULES

1. Arrow start and end MUST be at DIFFERENT locations. An arrow from an object to itself is INVALID.
2. The approach arrow MUST start at the "gripper" object and end at the target object.
3. You MUST use "object_relative" positions (with object_id from the Detected Objects list) for any position on or near a detected object.
4. Do NOT use absolute coordinates for positions that could reference a detected object.

## Example

For task "pick up the red_block and place it on the plate" with objects gripper, red_block, plate:

{{"primitives": [
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "gripper", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "red_block", "anchor": "top", "offset": [0.0, -0.02]}}, "waypoints": [], "step": 1}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "red_block", "anchor": "center"}}, "radius": 0.04, "purpose": "grasp_point", "step": 1}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "red_block", "anchor": "center"}}, "action": "close", "step": 2}},
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "red_block", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "plate", "anchor": "center", "offset": [0.0, -0.05]}}, "waypoints": [{{"type": "absolute", "coords": [0.45, 0.20]}}], "step": 3}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "plate", "anchor": "center"}}, "radius": 0.05, "purpose": "release_point", "step": 4}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "plate", "anchor": "center", "offset": [0.0, -0.05]}}, "action": "open", "step": 4}}
]}}

## Motion Patterns

Choose the appropriate pattern based on the task:

**Pick and place** (pick, move, place): approach arrow from gripper to source object, \
grasp, transport arrow from source to destination, release.

**Turn/rotate** (turn on, turn off, twist): approach arrow from gripper to the knob/dial, \
contact/grasp at the knob, rotation arrow showing the turning motion direction.

**Push** (push, slide): approach arrow from gripper to the object, contact, \
push arrow showing the push direction.

**Open/close** (open drawer, close door): approach arrow from gripper to the handle, \
grasp at handle, pull/push arrow showing the opening/closing motion.

## Output Format

Respond with ONLY valid JSON. No text before or after.

## Task

Instruction: {task_instruction}

Now analyze the image and specify primitives to visualize this task. \
Remember: arrows must connect DIFFERENT locations, and you MUST use \
object_relative positions with the exact object IDs listed above.\
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
