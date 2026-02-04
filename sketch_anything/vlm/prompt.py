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
- end: Position (where motion ends — MUST be a different location from start)
- waypoints: list of Position (intermediate points for curved paths)
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
{{"type": "object_relative", "object_id": "<id>", "anchor": "<anchor>", "offset": [dx, dy]}}

Anchors: "center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"

Absolute position (ONLY for free-space waypoints):
{{"type": "absolute", "coords": [x, y]}}
Coordinates in [0.0, 1.0], origin at top-left.

## CRITICAL RULES

1. Arrow start and end MUST reference DIFFERENT objects or different anchors. An arrow that starts and ends at the same position is INVALID.
2. The first approach arrow MUST start at the "gripper" object and end at the target object. If NO "gripper" object is listed in Detected Objects (e.g. for an eye-in-hand camera view), start the approach arrow from an absolute position at the bottom-center of the image: {{"type": "absolute", "coords": [0.5, 1.0]}}.
3. You MUST use "object_relative" positions (with object_id from the Detected Objects list) for any position on or near a detected object. Do NOT use absolute coordinates for these.
4. Use non-zero offsets to position precisely. For approach arrows, end at anchor "top" with offset [0.0, -0.02] to land above the target. For release, use offset [0.0, -0.05] above the destination.
5. WAYPOINT PLACEMENT: For transport/move arrows between two objects, add one absolute waypoint to create a smooth lift arc. Compute the waypoint as follows:
   - waypoint_x = midpoint between the start object center_x and end object center_x
   - waypoint_y = min(start object center_y, end object center_y) minus 0.15 (i.e. well ABOVE both objects on screen, since y=0 is the top of the image)
   - Example: if source center is [0.44, 0.43] and destination center is [0.19, 0.71], then waypoint = [(0.44+0.19)/2, min(0.43,0.71)-0.15] = [0.32, 0.28]
   - The waypoint must be ABOVE (lower y) than both the start and end objects to show a lift arc.
6. Primitives that happen simultaneously MUST share the same step number. The approach arrow and the grasp/contact circle share the same step. The release circle and gripper open share the same step.

## Motion Patterns with Examples

Choose the pattern that matches the task instruction. Each example shows the complete JSON output.

### Pattern 1: Pick and Place
Keywords: pick, place, put, move, put on, put in

Task "pick up the red_block and place it on the plate" with objects gripper (center [0.45, 0.21]), red_block (center [0.32, 0.46]), plate (center [0.59, 0.42]):

Waypoint calculation for transport arrow: midpoint_x = (0.32+0.59)/2 = 0.46, waypoint_y = min(0.46, 0.42) - 0.15 = 0.27. So waypoint = [0.46, 0.27].

{{"primitives": [
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "gripper", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "red_block", "anchor": "top", "offset": [0.0, -0.02]}}, "waypoints": [], "step": 1}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "red_block", "anchor": "center"}}, "radius": 0.04, "purpose": "grasp_point", "step": 1}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "red_block", "anchor": "center"}}, "action": "close", "step": 2}},
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "red_block", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "plate", "anchor": "center", "offset": [0.0, -0.05]}}, "waypoints": [{{"type": "absolute", "coords": [0.46, 0.27]}}], "step": 3}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "plate", "anchor": "center"}}, "radius": 0.05, "purpose": "release_point", "step": 4}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "plate", "anchor": "center", "offset": [0.0, -0.05]}}, "action": "open", "step": 4}}
]}}

### Pattern 2: Turn / Rotate
Keywords: turn on, turn off, twist, rotate, switch

Task "turn on the stove" with objects gripper, stove, stove_knob:

{{"primitives": [
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "gripper", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "stove_knob", "anchor": "top", "offset": [0.0, -0.02]}}, "waypoints": [], "step": 1}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "stove_knob", "anchor": "center"}}, "radius": 0.03, "purpose": "contact", "step": 1}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "stove_knob", "anchor": "center"}}, "action": "close", "step": 2}},
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "stove_knob", "anchor": "right"}}, "end": {{"type": "object_relative", "object_id": "stove_knob", "anchor": "left"}}, "waypoints": [{{"type": "object_relative", "object_id": "stove_knob", "anchor": "top", "offset": [0.0, -0.04]}}], "step": 3}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "stove_knob", "anchor": "center"}}, "radius": 0.03, "purpose": "rotation_pivot", "step": 3}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "stove_knob", "anchor": "center"}}, "action": "open", "step": 4}}
]}}

### Pattern 3: Push / Slide
Keywords: push, slide

Task "push the plate to the front_of_stove" with objects gripper, plate, front_of_stove:

{{"primitives": [
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "gripper", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "plate", "anchor": "top", "offset": [0.0, -0.02]}}, "waypoints": [], "step": 1}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "plate", "anchor": "center"}}, "radius": 0.04, "purpose": "contact", "step": 1}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "plate", "anchor": "center"}}, "action": "close", "step": 2}},
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "plate", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "front_of_stove", "anchor": "center", "offset": [0.0, -0.03]}}, "waypoints": [], "step": 3}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "front_of_stove", "anchor": "center"}}, "radius": 0.05, "purpose": "target_location", "step": 3}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "front_of_stove", "anchor": "center"}}, "action": "open", "step": 4}}
]}}

### Pattern 4: Open / Close
Keywords: open, close, pull

Task "open the top_drawer" with objects gripper, top_drawer:

{{"primitives": [
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "gripper", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "top_drawer", "anchor": "center", "offset": [0.0, -0.02]}}, "waypoints": [], "step": 1}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "top_drawer", "anchor": "center"}}, "radius": 0.04, "purpose": "grasp_point", "step": 1}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "top_drawer", "anchor": "center"}}, "action": "close", "step": 2}},
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "top_drawer", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "top_drawer", "anchor": "center", "offset": [0.0, -0.15]}}, "waypoints": [], "step": 3}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "top_drawer", "anchor": "center", "offset": [0.0, -0.15]}}, "action": "open", "step": 4}}
]}}

## Output Format

Respond with ONLY valid JSON matching the format above. No text before or after the JSON.

## Task

Instruction: {task_instruction}

Analyze the image carefully. Identify which pattern best matches the task, then specify the complete set of primitives. \
Use the exact object IDs from Detected Objects above. Use non-zero offsets as shown in the examples. \
For transport arrows, compute a waypoint ABOVE both objects: waypoint_x = midpoint of source and destination center_x, \
waypoint_y = min(source center_y, destination center_y) - 0.15. Use the object centers from Detected Objects to calculate this.\
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
