"""Prompt constants for the LLM planner (o3).

DESIGN PRINCIPLE — GENERALIZATION:
These prompts must work across ALL LIBERO tasks, not just bowl-on-plate.
Tasks include: pick-and-place (various objects), opening drawers/cabinets,
pushing objects, stacking, and more.

LIBERO-SPECIFIC LEARNING (validated against multiple demo trajectories):
  - LIBERO uses CRADLE grasp universally — even for narrow objects (cream cheese
    width=0.040m).  Attempting a PINCH descent (open fingers, descend to centroid,
    close) fails because fingers slip off curved or smooth surfaces.
  - The CRADLE depth formula:  cradle_z = max(centroid_z - extents_z, 0.910)
    This clips to 0.910 to avoid table collisions for tall objects.  Verified:
      bowl (centroid=0.990, extents_z=0.064) → cradle_z = max(0.926, 0.910) = 0.926
      cream cheese (centroid=0.935, extents_z=0.081) → cradle_z = max(0.854, 0.910) = 0.910
"""

SYSTEM_PROMPT = """\
You are a robot manipulation trajectory planner for the LIBERO simulation environment.

CONTROLLER: OSC (Operational Space Control).
  - Each action unit moves the EEF ~0.01m per step. Actions are clipped to [-1, 1].
  - You output absolute world-frame target positions. The executor converts to deltas automatically.
  - Gripper: 1.0 = OPEN (fingers spread apart), -1.0 = CLOSED (fingers together)
  - Roll/pitch/yaw = 0.0 for all actions (no wrist rotation needed)

COORDINATE SYSTEM: World frame (meters).
  Workspace: x [-0.5, 0.5], y [-0.5, 0.5], z [0.90, 1.30]
  Higher z = higher above the table. Objects rest at z ≈ 0.92–1.10.
  Initial EEF z is typically ≈ 1.17. Table surface z ≈ 0.87. Do NOT plan z < 0.90.

GRIPPER PHYSICS — UNIVERSAL CRADLE GRASP:
  LIBERO requires CRADLE grasp for ALL objects (not PINCH).
  PINCH (open-descend-close) is unreliable; fingers slip off smooth/curved surfaces.

  CRADLE GRASP sequence — use for every pick:
    Step A: Approach above source with gripper CLOSED (-1.0).
            EEF → [src_x, src_y, src_z + 0.10]
    Step B: Descend with gripper CLOSED to cradle depth:
            cradle_z = max(src_centroid_z - src_extents_z, 0.910)
            EEF → [src_x, src_y, cradle_z]
    Step C: OPEN gripper (1.0) — 5 dwell steps at cradle_z to spread fingers under object.
    Step D: Lift AND transport DIAGONALLY with gripper OPEN (avoids mid-scene obstacles).
            lift_z = max(src_z + 0.15, dst_top_z + 0.05)   ← MUST clear the goal's top surface!
            dst_top_z = dst_centroid_z + dst_extents_z/2
            Go directly from [src_x, src_y, cradle_z] to [dst_x, dst_y, lift_z] in ONE arc.
            Do NOT lift straight up first — a diagonal avoids obstacles between source and goal.
    Step E: Descend to approach height.
            EEF → [dst_x, dst_y, place_z + 0.05]
    Step F: Descend to placement with gripper OPEN.
            EEF → [dst_x, dst_y, place_z]   ← see PLACEMENT DEPTH below
    Step G: CLOSE gripper (-1.0) — 3 steps to press object onto surface.
    Step H: OPEN gripper (1.0) — 5 final dwell steps to release. End trajectory here.

PLACEMENT DEPTH — compute place_z based on task type:
  "put X ON Y" (plate, stove surface, cabinet top, any solid surface):
      dst_top_z = dst_centroid_z + dst_extents_z / 2   ← actual top surface of destination
      place_z = dst_top_z
      Using dst_centroid_z + 0.02 is WRONG for tall objects (cabinet, box) where centroid
      is far below the actual placement surface.
  "put X IN Y" (bowl, basket, drawer — object must go INSIDE the container):
      place_z = dst_centroid_z - 0.03   ← 3cm BELOW centroid to reach container bottom
      Placing at dst_centroid_z+0.02 (near rim) scores as NOT inside — go deeper!
  "put X on rack / wine rack" (slot-based tall structure):
      place_z = dst_top_z - 0.10   where dst_top_z = dst_centroid_z + dst_extents_z/2
      The rack slot is near the top (about 0.10m below the uppermost edge).
      lift_z MUST exceed dst_top_z + 0.05 so the object clears the rack frame.

INTERACTION POINTS:
  - Source objects: use centroid_3d for grasp target (always)
  - Drawer/cabinet handles: target the handle body centroid for pull/push
  - Stacking surface: dst = top surface z → use (centroid_z + extents_z/2) as dst[2]
  - Cabinet top: cabinet centroid in registry is the cabinet_top body; use it directly as dst
  bbox_extents in the registry helps identify surfaces relative to centroid.

SPECIAL TASK TYPES:

  PUSH tasks ("push X to/toward Y"):
    Do NOT use CRADLE grasp. Keep gripper CLOSED (-1.0) throughout.
    Compute these values FIRST:
      push_dir   = [dst_x - src_x, dst_y - src_y]  (unnormalized)
      push_norm  = sqrt(push_dir[0]**2 + push_dir[1]**2)
      push_unit  = [push_dir[0]/push_norm, push_dir[1]/push_norm]
      approach_margin = max(src_extents_x, src_extents_y)/2 + 0.06
      approach_x = src_x - push_unit[0] * approach_margin  ← BEHIND the object, OUTSIDE its bbox
      approach_y = src_y - push_unit[1] * approach_margin
      contact_z  = src_centroid_z + src_extents_z / 2      ← top surface, not centroid (avoids clipping into object)
    Motion:
      Step 1: Descend above approach point: EEF → [approach_x, approach_y, contact_z + 0.10] CLOSED
      Step 2: Lower to contact height:      EEF → [approach_x, approach_y, contact_z] CLOSED
      Step 3: Push diagonally to goal:       EEF → [dst_x, dst_y, contact_z] CLOSED
    The EEF stays at contact_z for the entire push. Use enough steps for the full push distance.

  INTERACT/TURN tasks ("turn on X", "turn off X", "press X"):
    No transport needed. The target is a button or knob.
    Motion:
      Step 1: Approach above target: EEF → [target_x, target_y, target_z + 0.10] CLOSED
      Step 2: Descend to target: EEF → [target_x, target_y, target_z] CLOSED
      Step 3: Interact: OPEN gripper (1.0) — 5 dwell steps at target_z
    That is all. Do not plan transport to another location.

STEP BUDGET: steps needed = distance_meters / 0.01, minimum 10 steps per phase.
  Simple pick-and-place (~1.0m total path): ~100 steps.
  Complex insertion (rack, drawer, cabinet): 300–400 steps — use as many as needed.
  Do NOT truncate or abbreviate to hit a step limit. Use the full step count."""

TRAJECTORY_PROMPT_TEMPLATE = """\
Task: {task_instruction}

Object registry (world frame, meters):
{object_registry}

Current end-effector pose [x, y, z, roll, pitch, yaw, gripper]:
{eef_pose}

Generate a complete trajectory to accomplish the task above.
Use as many steps as needed (maximum {max_steps}). Do not truncate.
{feedback_section}
Before writing code, briefly identify:
  1. Source object label and centroid_3d / bbox_extents_z
  2. Goal object label and centroid_3d / bbox_extents_z; compute dst_top_z = dst_centroid_z + dst_extents_z/2
  3. Cradle depth: cradle_z = max(src_centroid_z - src_extents_z, 0.910)
  4. Lift clearance: lift_z = max(src_centroid_z + 0.15, dst_top_z + 0.05)
  5. Placement depth: place_z (ON surface / IN container / rack slot)

Then write the Python function implementing the CRADLE sequence (Steps A–H from the system prompt).

Output ONLY a Python function (preceded by the brief identification above). No other explanation.

```python
import numpy as np

def get_trajectory():
    \"\"\"Returns list of [x, y, z, roll, pitch, yaw, gripper] actions.\"\"\"
    actions = []

    def phase(start, end, n_steps, gripper_val):
        pts = np.linspace(start, end, n_steps)
        return [[float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0, gripper_val] for p in pts]

    def n(a, b):
        return max(10, int(np.linalg.norm(np.array(a) - np.array(b)) / 0.01) + 2)

    eef = {eef_pose}[:3]

    # --- Fill in src, dst, extents from the registry ---
    # src = [x, y, z]           # centroid_3d of source object
    # src_extents_z = ...        # bbox_extents[2] of source object
    # dst = [x, y, z]           # centroid_3d of goal location
    # dst_extents_z = ...        # bbox_extents[2] of goal object
    # dst_top_z = dst[2] + dst_extents_z / 2.0
    # cradle_z = max(src[2] - src_extents_z, 0.910)
    #
    # Compute place_z from task type:
    #   "on surface/plate/stove/cabinet" → place_z = dst_top_z = dst[2] + dst_extents_z/2
    #   "in container/bowl/basket"       → place_z = dst[2] - 0.03
    #   "on rack (tall slot structure)"  → place_z = dst_top_z - 0.10
    #
    # Compute lift_z (MUST clear goal's top):
    #   lift_z = max(src[2] + 0.15, dst_top_z + 0.05)

    # --- UNIVERSAL CRADLE GRASP (Steps A–H) ---
    # Step A: approach above source (gripper CLOSED)
    # actions += phase(eef, [src[0],src[1],src[2]+0.10], n(eef,[src[0],src[1],src[2]+0.10]), -1.0)
    # Step B: descend to cradle depth (gripper CLOSED)
    # actions += phase([src[0],src[1],src[2]+0.10],[src[0],src[1],cradle_z], n([src[0],src[1],src[2]+0.10],[src[0],src[1],cradle_z]), -1.0)
    # Step C: open at cradle depth (5 dwell steps)
    # actions += [[src[0],src[1],cradle_z, 0.,0.,0., 1.0] for _ in range(5)]
    # Step D: DIAGONAL lift+transport in ONE arc (avoids mid-scene obstacles)
    # actions += phase([src[0],src[1],cradle_z],[dst[0],dst[1],lift_z], n([src[0],src[1],cradle_z],[dst[0],dst[1],lift_z]), 1.0)
    # Step E: descend from lift_z to place_z + 0.05 (approach surface/slot)
    # actions += phase([dst[0],dst[1],lift_z],[dst[0],dst[1],place_z+0.05], n([dst[0],dst[1],lift_z],[dst[0],dst[1],place_z+0.05]), 1.0)
    # Step F: descend to place_z (gripper OPEN)
    # actions += phase([dst[0],dst[1],place_z+0.05],[dst[0],dst[1],place_z], n([dst[0],dst[1],place_z+0.05],[dst[0],dst[1],place_z]), 1.0)
    # Step G: close to press object onto surface (3 steps)
    # actions += [[dst[0],dst[1],place_z, 0.,0.,0.,-1.0] for _ in range(3)]
    # Step H: open to release (5 dwell steps)
    # actions += [[dst[0],dst[1],place_z, 0.,0.,0., 1.0] for _ in range(5)]

    return actions
```"""

FEEDBACK_SECTION_TEMPLATE = """\
CORRECTION REQUIRED:
{feedback}

Revise your trajectory to address the above failure. Adjust approach height, \
grasp position, target coordinates, or gripper timing as needed.

"""
