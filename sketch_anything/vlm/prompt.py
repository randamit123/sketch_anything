"""VLM prompt template for sketch primitive generation.

The template instructs Qwen2.5-VL to output structured JSON conforming to
the SketchPrimitives schema. Double braces ``{{`` / ``}}`` are used to
escape literal JSON braces in the f-string / .format() calls.
"""

from __future__ import annotations

from typing import Dict, Optional

PROMPT_TEMPLATE = """\
You are a robot motion sketch annotator. Given an image of a tabletop scene \
with detected objects and a manipulation task instruction, you must specify \
drawing primitives that visualize the intended end-effector motion path.

Your output will be rendered as a sketch overlay on the scene image to verify \
that the robot's planned trajectory matches the task intent.

## Detected Objects

The following objects have been detected in the scene with their bounding boxes. \
You MUST use these exact object IDs in your object_relative positions. \
Using any other string as object_id is INVALID.

{object_registry_formatted}

**VALID object_id values (copy exactly, no variations):** {valid_object_ids}

{privileged_context}
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

Object-relative position (REQUIRED for any location on/near an object):
{{"type": "object_relative", "object_id": "<id>", "anchor": "<anchor>", "offset": [dx, dy]}}

Anchors: "center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"

Absolute position (ONLY for waypoints[] entries — never for start, end, or gripper position):
{{"type": "absolute", "coords": [x, y]}}
Coordinates in [0.0, 1.0], origin at top-left.

**POSITION RULE: Every start, end, and gripper/circle position MUST be "object_relative". \
Absolute coordinates are ONLY permitted inside waypoints[] arrays. \
Using absolute for start or end is INVALID.**

## CRITICAL RULES

1. Arrow start and end MUST reference DIFFERENT objects or different anchors. An arrow that starts and ends at the same position is INVALID.
2. The first approach arrow (and EVERY other arrow that represents the robot's end-effector moving toward an object) MUST start at {{"type": "object_relative", "object_id": "gripper", "anchor": "center"}}. The "gripper" object is always present in every camera view. NEVER use an absolute position for the start of an approach arrow.
3. You MUST use "object_relative" positions (with object_id from the Detected Objects list) for any position on or near a detected object. Do NOT use absolute coordinates for these.
4. Use non-zero offsets to position precisely. For approach arrows, end at anchor "top" with offset [0.0, -0.02] to land above the target. For release, use offset [0.0, -0.05] above the destination.
5. WAYPOINT PLACEMENT: For transport/move arrows between two objects, add one absolute waypoint to create a smooth lift arc. You MUST compute and show the arithmetic:
   - waypoint_x = (start_center_x + end_center_x) / 2
   - waypoint_y = min(start_center_y, end_center_y) - 0.15  [clamp to 0.02 if result < 0]
   - CONSTRAINT: waypoint_y MUST be at least 0.10 less than the end object's center_y. If not, decrease waypoint_y until this holds.
   - Example: source center [0.67, 0.57], dest center [0.50, 0.31] → waypoint_x=(0.67+0.50)/2=0.585, waypoint_y=min(0.57,0.31)-0.15=0.31-0.15=0.16. Constraint check: 0.16 < 0.31-0.10=0.21 ✓. Waypoint = [0.585, 0.16].
   - The waypoint must be ABOVE (lower y) than both the start and end objects to show a clear lift arc.
6. Primitives that happen simultaneously MUST share the same step number. The approach arrow and the grasp/contact circle share the same step. The release circle and gripper open share the same step.
7. CONTAINER/SLOT TARGETS: When placing an object INTO a container, rack, basket, or slot (e.g. wine rack, bowl, bin), use anchor "top" with a small positive y-offset for the transport end and release position: {{"type": "object_relative", "object_id": "<container>", "anchor": "top", "offset": [0.0, 0.08]}}. This places the arrowhead and release circle inside the container opening, not at the geometric center of the container body.

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
  {{"type": "arrow", "start": {{"type": "object_relative", "object_id": "red_block", "anchor": "center"}}, "end": {{"type": "object_relative", "object_id": "plate", "anchor": "top", "offset": [0.0, 0.08]}}, "waypoints": [{{"type": "absolute", "coords": [0.46, 0.27]}}], "step": 3}},
  {{"type": "circle", "center": {{"type": "object_relative", "object_id": "plate", "anchor": "top", "offset": [0.0, 0.08]}}, "radius": 0.05, "purpose": "release_point", "step": 4}},
  {{"type": "gripper", "position": {{"type": "object_relative", "object_id": "plate", "anchor": "top", "offset": [0.0, 0.08]}}, "action": "open", "step": 4}}
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

Respond with ONLY raw JSON — no markdown, no ```json blocks, no explanation. \
Start your response with {{ and end with }}. \
**REMINDER: only use these object_id values: {valid_object_ids}**

## Task

Instruction: {task_instruction}

Analyze the image carefully. Identify which pattern best matches the task, then specify the complete set of primitives. \
Use the exact object IDs from Detected Objects above. Use non-zero offsets as shown in the examples. \
For transport arrows, compute a waypoint ABOVE both objects: waypoint_x = midpoint of source and destination center_x, \
waypoint_y = min(source center_y, destination center_y) - 0.15. Use the object centers from Detected Objects to calculate this.\
"""


def compute_privileged_context(
    object_registry: Dict[str, dict],
    task_instruction: str,
) -> str:
    """Generate a privileged task context block for injection into the prompt.

    Uses ground-truth object positions from the registry to:
    1. Annotate each object with its task role (SOURCE, DESTINATION, TARGET, etc.).
    2. Pre-compute the transport waypoint so the VLM copies exact values.

    Returns an empty string if roles cannot be determined (unknown verb pattern).

    Args:
        object_registry: Single-view object registry with {id, label, bbox, center}.
        task_instruction: Natural language task description.

    Returns:
        Multi-line string block, or "" if no privileged context can be generated.
    """
    task_lower = task_instruction.lower().strip()
    non_gripper = {k: v for k, v in object_registry.items() if k != "gripper"}
    gripper = object_registry.get("gripper")

    if not non_gripper:
        return ""

    obj_ids = list(non_gripper.keys())

    # --- Determine verb category ---
    is_open_put = (
        ("open" in task_lower or "put" in task_lower)
        and "put" in task_lower
        and "inside" in task_lower
    )
    is_push = any(v in task_lower for v in ["push ", "slide "])
    is_transport = (
        not is_open_put
        and not is_push
        and any(v in task_lower for v in ["put ", "place ", "pick ", "move "])
    )
    is_open = (
        not is_open_put
        and any(v in task_lower for v in ["open ", "close ", "pull "])
    )
    is_rotate = any(v in task_lower for v in ["turn on", "turn off", "twist", "rotate"])

    # --- Assign roles ---
    roles: Dict[str, str] = {}

    if is_open_put and len(obj_ids) >= 2:
        # "open X and put Y inside" — extraction order: [container, source_object]
        roles[obj_ids[0]] = "CONTAINER (= DESTINATION)"
        for oid in obj_ids[1:]:
            roles[oid] = "SOURCE"
    elif (is_transport or is_push) and len(obj_ids) >= 2:
        roles[obj_ids[0]] = "SOURCE"
        roles[obj_ids[1]] = "DESTINATION"
    elif (is_transport or is_push) and len(obj_ids) == 1:
        roles[obj_ids[0]] = "TARGET"
    elif is_open:
        for oid in obj_ids:
            roles[oid] = "TARGET"
    elif is_rotate:
        for oid in obj_ids:
            if any(k in oid for k in ("knob", "button", "switch")):
                roles[oid] = "PIVOT"
            else:
                roles[oid] = "TARGET"
    else:
        # Unknown pattern — don't emit privileged context to avoid misleading the VLM
        return ""

    # --- Build context string ---
    lines = ["## Privileged Task Context (ground-truth positions)\n"]
    lines.append("Task roles:")
    if gripper:
        gcx, gcy = gripper["center"]
        lines.append(
            f"- GRIPPER (start of ALL approach arrows): gripper"
            f" — center [{gcx:.3f}, {gcy:.3f}]"
        )
    for oid, role in roles.items():
        cx, cy = non_gripper[oid]["center"]
        lines.append(f"- {role}: {oid} — center [{cx:.3f}, {cy:.3f}]")

    # --- Pre-compute transport waypoint ---
    source_id = next((oid for oid, r in roles.items() if r == "SOURCE"), None)
    dest_id = next((oid for oid, r in roles.items() if "DESTINATION" in r), None)

    if source_id and dest_id and is_transport:
        src_cx, src_cy = non_gripper[source_id]["center"]
        dst_cx, dst_cy = non_gripper[dest_id]["center"]

        wp_x = (src_cx + dst_cx) / 2
        wp_y_raw = min(src_cy, dst_cy) - 0.15
        wp_y = max(0.02, wp_y_raw)
        if wp_y >= dst_cy - 0.10:
            wp_y = max(0.02, dst_cy - 0.12)

        clamped = wp_y != wp_y_raw
        constraint_ok = wp_y < dst_cy - 0.10

        lines.append(
            "\nPre-computed transport waypoint (USE THESE EXACT VALUES in waypoints[]):"
        )
        lines.append(
            f"  waypoint_x = ({src_cx:.3f} + {dst_cx:.3f}) / 2 = {wp_x:.3f}"
        )
        wp_y_display = f"min({src_cy:.3f}, {dst_cy:.3f}) - 0.15 = {wp_y_raw:.3f}"
        if clamped:
            wp_y_display += f" → clamped to {wp_y:.3f}"
        lines.append(f"  waypoint_y = {wp_y_display}")
        constraint_label = "✓" if constraint_ok else f"✗ → adjusted to {wp_y:.3f}"
        lines.append(
            f"  Constraint check: {wp_y:.3f} < {dst_cy:.3f} - 0.10"
            f" = {dst_cy - 0.10:.3f} {constraint_label}"
        )
        lines.append(f"  → WAYPOINT = [{wp_x:.3f}, {wp_y:.3f}]")

    return "\n".join(lines) + "\n"


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
    valid_ids = ", ".join(f'"{k}"' for k in object_registry.keys())
    privileged_context = compute_privileged_context(object_registry, task_instruction)
    return PROMPT_TEMPLATE.format(
        object_registry_formatted=registry_formatted,
        valid_object_ids=valid_ids,
        privileged_context=privileged_context,
        task_instruction=task_instruction,
    )
