# CLAUDE.md — Path 1: 3D + Execution (`path1_execution/`)

Read this file completely before writing any code. Every decision, constraint, and design
rationale you need is embedded here. There are no external documents to consult.

---

## STOP — Critical Constraints

**DO NOT merge `path1_execution/` with `sketch_anything/`.** Ever. Not now, not at the
end of this task, not when tests pass. The merge is a separate future phase with its own
specification. If you find yourself touching anything inside `sketch_anything/`, stop.

**DO NOT modify anything inside `sketch_anything/`.** That directory is read-only. Import
from it freely; never edit it.

**DO NOT implement the LLM planner (o3) until all mock-based tests pass end-to-end.**
The mock planner is the default. It must work completely before any API calls are
introduced.

**DO NOT skip steps in the implementation order.** Each step has a verification gate.
Complete and confirm it before proceeding.

**The LLM planner prompt (`planning/prompts.py`) MUST be task-agnostic and fully generalizable.**
It must work across ALL LIBERO tasks: pick-and-place with varied objects, opening drawers,
pushing objects, stacking, etc. Do NOT hardcode object-specific strategies (e.g. "scoop
for bowls") or fixed phase sequences into the prompt. The system prompt describes controller
physics and coordinate conventions only. The LLM (o3) must reason about the task and choose
the appropriate motion strategy from the task instruction and object registry it is given.
The only fixed scaffold in the template is the Python function output format.

---

## Visual Debugging

**Use native visual capabilities to inspect debug images before and after code changes.**

When the pipeline runs it saves `outputs/path1_<run>/debug/projected_{camera}.png` for
each camera view.  These images show the planned trajectory overlaid on the scene.

To debug object targeting issues (wrong body selected, bbox in wrong place):
1. Read the debug PNG with the Read tool — it renders inline as an image.
2. Visually confirm that trajectory arrows and circle markers land on the correct
   objects (e.g., stove BURNER surface, bowl centroid, drawer HANDLE).
3. Compare the visual position against the centroid logged by `extract_objects`.

If the circle/arrow misses the physical object in the image, the body selected by
`build_object_registry` is wrong.  Add an override to `PLACEMENT_BODY_OVERRIDES` in
`scene/object_extraction.py` (path1-local, does NOT touch `sketch_anything/`).

**Known override in place:**
- `flat_stove_1_button / flat_stove_1_main / flat_stove_1_base` → `flat_stove_1_burner`
  (static mapping in `sketch_anything/` resolves "stove" to the knob/base body;
  the burner is the correct placement surface)

---

## Prior Work & References

**"Language Models as Zero-Shot Trajectory Generators" (L2T)**
Paper: https://arxiv.org/abs/2310.11604

This is the primary prior art for Path 1. Key validated findings that directly
inform our design:

- Code output (Python function) beats raw numerical lists 6×: 60% vs 10% success
  on identical tasks with GPT-4. This validates our `def get_trajectory()` approach
  in `llm_planner.py` and the sandboxed exec() call.

- A single task-agnostic prompt works across 30+ diverse real-world tasks without
  in-context examples or task-specific tuning — exactly our design.

- Chain-of-thought decomposition before code generation: the LLM should identify
  sub-goals and interaction points BEFORE writing the trajectory function.

- Object dimensions in the prompt enable collision awareness and grasp strategy
  selection. bbox_extents must be in the registry passed to the LLM.

- Specific interaction points matter more than centroids: rim vs centroid for bowls,
  handle vs body for drawers, top surface for stacking targets.

- Explicit gripper reasoning (describing WHEN and WHY to open/close) outperforms
  hardcoded binary states in the generated code.

---

## What This Is

`path1_execution/` is an isolated Python package implementing Path 1 of a robot
manipulation pipeline. It operates inside the LIBERO simulation environment (MuJoCo).

Given a natural language task description and multi-view RGB images from a LIBERO episode,
it must:

1. Build a grounded 3D scene representation from MuJoCo ground-truth data.
2. Generate a sequence of 6-DoF end-effector poses (a trajectory) that accomplishes
   the task.
3. Project that trajectory back to 2D sketch overlays on each camera view.
4. Verify the trajectory using the LIBERO simulator's success signal and a geometric
   distance check.
5. If verification fails: generate corrective feedback, re-plan, and retry automatically.
6. If verification passes: execute the trajectory, save the rollout as an HDF5 demo.

The pipeline runs in a loop until it either succeeds or exhausts its retry budget. It
requires no human input after launch.

### Pipeline Flow

```
Inputs
  ├── task_instruction: str          (e.g. "pick up the red block and place it in the basket")
  └── hdf5_path: str                 (path to LIBERO demo file)
          │
          ▼
┌─────────────────────────────┐
│  SceneAgent                 │  ← runs once per episode
│  Stage 1: 3D Reconstruction │
│  Stage 2: Object Extraction │
│  → SceneContext             │
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  PlannerAgent               │  ← runs once per retry attempt
│  Stage 3: Trajectory Plan   │
│  → Trajectory               │
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  VerifierAgent              │  ← runs once per retry attempt
│  Stage 4: 2D Projection     │
│  Stage 5: Verification      │
│  Stage 6a: Normalization    │  (only on failure)
│  Stage 6b: LLM Feedback     │  (only on failure, LLM mode only)
│  → VerifyResult             │
└─────────────────────────────┘
          │
          ├── verified=True  → Execute → Save HDF5 → Done
          └── verified=False → feedback → PlannerAgent (retry)
                                          (up to max_retries times)
```

---

## Repository Layout

Create every file listed here. None of them exist yet.

```
path1_execution/
├── CLAUDE.md                        ← this file
├── __init__.py
├── config.py                        ← ALL dataclasses live here, nowhere else
├── pipeline.py                      ← instantiates orchestrator, called by CLI
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py              ← coordinates sub-agents + retry loop
│   ├── scene_agent.py               ← sub-agent 1
│   ├── planner_agent.py             ← sub-agent 2
│   └── verifier_agent.py            ← sub-agent 3
│
├── scene/
│   ├── __init__.py
│   ├── reconstruction.py            ← stage 1
│   └── object_extraction.py         ← stage 2
│
├── planning/
│   ├── __init__.py
│   ├── mock_planner.py              ← deterministic, no API calls, implement first
│   ├── llm_planner.py               ← o3-based, implement last
│   └── prompts.py                   ← all prompt strings as module-level constants
│
├── projection/
│   ├── __init__.py
│   └── trajectory_projector.py      ← stage 4
│
├── verification/
│   ├── __init__.py
│   ├── verifier.py                  ← stage 5
│   ├── primitive_normalizer.py      ← stage 6a
│   └── llm_feedback.py              ← stage 6b
│
├── execution/
│   ├── __init__.py
│   └── robot_executor.py
│
├── tools/
│   ├── __init__.py
│   └── run_path1.py                 ← CLI entry point
│
└── tests/
    ├── __init__.py
    ├── test_scene.py
    ├── test_planner.py
    ├── test_projector.py
    └── test_verifier.py
```

---

## Imports from `sketch_anything/`

Import these directly. Never copy, reimplement, or modify them.

| What you need | Where to import it from |
|---|---|
| Camera intrinsics + extrinsics (K, R, t) | `sketch_anything.libero_utils.camera.get_camera_matrices` |
| Project 3D points → normalized 2D | `sketch_anything.libero_utils.camera.project_points` |
| 2D bounding box from 3D corners | `sketch_anything.libero_utils.camera.compute_2d_bbox` |
| Capture RGB image from camera | `sketch_anything.libero_utils.env.get_camera_image` |
| 3D bounding box from MuJoCo body | `sketch_anything.libero_utils.env.get_object_bbox_3d` |
| Load LIBERO env from HDF5 | `sketch_anything.tools.run_pipeline.load_libero_env` |
| Load demo actions from HDF5 | `sketch_anything.tools.run_pipeline.load_demo_actions` |
| Build object registry from MuJoCo | `sketch_anything.registry.builder.build_object_registry` |
| Sketch primitive types | `sketch_anything.schemas.primitives` |
| Render primitives onto image | `sketch_anything.rendering.renderer.render_primitives` |
| Validate primitives | `sketch_anything.validation.validator.validate_primitives` |

---

## Sketch Primitive Schema

The existing codebase in `sketch_anything/schemas/primitives.py` defines three primitive
types. You must use these exact types when projecting trajectory waypoints back to 2D.
Do not define your own primitive types.

**ArrowPrimitive** — directed motion from one location to another. Used to represent
end-effector movement. Has a `start` position, `end` position, and a list of `waypoints`
for curved paths. Each waypoint is an intermediate position along the arc.

**CirclePrimitive** — marks a point of interest. Used to represent grasp events, release
points, and contact affordances. Has a `center`, `radius`, and `purpose` field. Valid
purposes are: `"grasp_point"`, `"release_point"`, `"contact"`, `"rotation_pivot"`,
`"target_location"`.

**GripperPrimitive** — marks a gripper state change (open or close) at a position.

All positions use either `AbsolutePosition` (normalized x,y in [0,1]) or
`ObjectRelativePosition` (relative to a detected object's bounding box anchor). See
`sketch_anything/schemas/primitives.py` for the complete Pydantic definitions.

**Mapping trajectory actions to primitives:**
- Motion waypoints → `ArrowPrimitive` with the sequence of projected 2D EEF positions
  as `waypoints`.
- Grasp events (indices in `trajectory.grasp_indices`) → `CirclePrimitive` with
  `purpose="grasp_point"` at the projected 2D position.
- Release events → `CirclePrimitive` with `purpose="release_point"`.
- Gripper open/close → `GripperPrimitive`.

---

## All Dataclasses (`config.py`)

Every dataclass used anywhere in this package must be defined in `config.py`. No other
file should define dataclasses. Import from `config.py` everywhere else.

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Path1Config:
    max_retries: int = 5
    max_action_steps: int = 150      # hard cap on trajectory length
    goal_threshold_m: float = 0.05   # 5 cm geometric success threshold
    use_mock_planner: bool = True     # True = mock, False = o3. Default is True.
    llm_model: str = "o3"            # only used when use_mock_planner=False
    render_scale: int = 2
    camera_names: list = field(
        default_factory=lambda: ["agentview", "frontview", "robot0_eye_in_hand"]
    )
    output_dir: str = "./outputs/path1"
    image_width: int = 256
    image_height: int = 256

@dataclass
class ObjectInfo:
    label: str
    centroid_3d: np.ndarray          # shape (3,), world frame
    bbox_3d_corners: np.ndarray      # shape (8, 3), world frame
    orientation: np.ndarray          # shape (3, 3), rotation matrix
    bbox_2d_per_camera: dict         # camera_name → [x_min, y_min, x_max, y_max] in [0,1]

@dataclass
class SceneContext:
    point_cloud: np.ndarray          # shape (N, 3), world frame
    object_registry: dict            # label → ObjectInfo
    camera_images: dict              # camera_name → np.ndarray (H, W, 3) uint8
    camera_matrices: dict            # camera_name → tuple(K, R, t)
    task_instruction: str
    initial_eef_pose: np.ndarray     # shape (7,) from env observation

@dataclass
class Trajectory:
    actions: list                    # list of np.ndarray, each shape (7,)
                                     # [x, y, z, roll, pitch, yaw, gripper]
    sub_goals: list                  # list of str, one label per phase
    grasp_indices: list              # list of int — which action indices are grasps
    metadata: dict                   # planner_type, model, retry_num, etc.

@dataclass
class PlannedSketch:
    primitives_per_camera: dict      # camera_name → SketchPrimitives
    rendered_images: dict            # camera_name → np.ndarray (H, W, 3)

@dataclass
class VerifyResult:
    verified: bool
    sim_success: bool
    geometric_success: bool
    final_eef_pose: np.ndarray       # shape (7,)
    distance_to_goal: float          # metres
    feedback: Optional[str]          # None if verified=True
    planned_sketch: PlannedSketch

@dataclass
class OrchestratorResult:
    success: bool
    attempts: int
    final_trajectory: Optional[Trajectory] = None
    final_verify_result: Optional[VerifyResult] = None
```

---

## Stage Specifications

### Stage 1 — 3D Reconstruction (`scene/reconstruction.py`)

For each camera in `config.camera_names`:
- Call `get_camera_image(env, camera_name)` → store in `SceneContext.camera_images`.
- Call `get_camera_matrices(sim, camera_name, config.image_width, config.image_height)`
  → store `(K, R, t)` tuple in `SceneContext.camera_matrices`.

For the point cloud: iterate over all MuJoCo body names tracked in the object registry.
For each body, call `get_object_bbox_3d(sim, body_name)` to get 8 corner points in world
frame. Concatenate all corners into a single `(N, 3)` array as the point cloud. This is
the sim-appropriate approximation — no depth sensor or stereo reconstruction needed for
isolated testing.

### Stage 2 — Object Extraction (`scene/object_extraction.py`)

Call `build_object_registry(env, sim, camera_names, image_size)` from
`sketch_anything.registry.builder`. This returns a per-camera registry with labels,
bounding boxes, and projected 2D centers.

Enrich each entry into an `ObjectInfo`:
- `centroid_3d` = mean of the 8 corners from `get_object_bbox_3d`.
- `bbox_3d_corners` = the raw (8, 3) array.
- `orientation` = from `sim.data.body_xmat` for the body.
- `bbox_2d_per_camera` = already computed by `build_object_registry`.

Apply a task-relevance filter: retain only objects whose `label` appears as a
case-insensitive substring in `task_instruction`. This keeps the object registry small
and focused, which reduces prompt length for the LLM planner.

### Stage 3 — Trajectory Planning (`planning/`)

**Mock planner (`planning/mock_planner.py`) — implement and use first:**

Deterministic. Takes `SceneContext`. Inspects `object_registry` for the first
task-relevant object (source) and optionally a second (goal). Generates a fixed
three-phase trajectory:

- Phase 1 — Pre-grasp approach: EEF moves to `source.centroid_3d + [0, 0, 0.15]`
  (15 cm above the object). Gripper open.
- Phase 2 — Grasp: EEF moves down to `source.centroid_3d`. Gripper closes.
  Record this action index in `grasp_indices`.
- Phase 3 — Transport: EEF lifts back to `source.centroid_3d + [0, 0, 0.15]`, then
  moves to `goal.centroid_3d + [0, 0, 0.10]`, then descends to `goal.centroid_3d`.
  Gripper opens. Record this release index.

Each phase is discretized into equal-step linear interpolations. Total steps must not
exceed `config.max_action_steps`. Each action is a `(7,)` array: `[x, y, z, 0, 0, 0,
gripper_state]` where `gripper_state = 1.0` for open, `-1.0` for closed.

Returns a valid `Trajectory`. No imports from `llm_planner.py`.

**LLM planner (`planning/llm_planner.py`) — implement only after mock tests pass:**

Uses o3 via the Anthropic API (same endpoint pattern as `sketch_anything/vlm/generator.py`
— match the API call structure exactly). Model: `config.llm_model = "o3"`.

Build the prompt from constants in `planning/prompts.py`. The prompt must contain, in
this order:

1. Task instruction.
2. Serialized object registry as JSON: `{label: {centroid_3d: [...], bbox_extents: [...]}}`.
3. Current EEF pose as a list.
4. Available action primitives with type signatures:
   `move_to(x, y, z, roll, pitch, yaw)`, `open_gripper()`, `close_gripper()`.
5. If `feedback` is not None: include it under a clearly labelled "Correction required"
   section.
6. Chain-of-thought instruction: "First decompose the task into sub-goals. Then for each
   sub-goal, generate the action sequence."
7. Output format instruction: "Write a Python function `def get_trajectory():` that
   returns a list of 7-element lists. Each element is `[x, y, z, roll, pitch, yaw,
   gripper]` where gripper is 1.0 (open) or -1.0 (closed). Do not include any other
   text. Only the function."

Execute the returned Python function in a sandboxed `exec()` call to produce the action
list. Fall back to parsing raw numeric output if code execution fails. Cap at
`config.max_action_steps`.

The code-output approach is preferred over raw numeric output because it handles complex
curved trajectory shapes (arcs, spiral motions, rotations) more reliably than asking the
model to enumerate poses directly.

### Stage 4 — 2D Trajectory Projection (`projection/trajectory_projector.py`)

For each action `aᵢ = (x, y, z, roll, pitch, yaw, gripper)` in `trajectory.actions`:
- Extract the 3D EEF position `[x, y, z]`.
- For each camera, call `project_points(np.array([[x, y, z]]), K, R, t, image_width,
  image_height)` → returns normalized `(1, 2)` array.

Group consecutive non-grasp actions into `ArrowPrimitive` objects, using the sequence of
projected 2D points as `waypoints`. At each index in `trajectory.grasp_indices`, emit a
`CirclePrimitive` with `purpose="grasp_point"` and `radius=0.04`.

Render the resulting `SketchPrimitives` onto a copy of the camera image using
`render_primitives`. Store both the primitives and rendered images in `PlannedSketch`.

### Stage 5 — Verification (`verification/verifier.py`)

Execute the trajectory via `robot_executor.py` in a trial rollout. Read the LIBERO
`done` flag and reward from `env.step()` responses. If the environment reports task
completion (`done=True` and `reward > 0`), set `sim_success = True`.

Independently, compute `np.linalg.norm(final_eef_pos - goal_centroid_3d)`. If this is
less than `config.goal_threshold_m` (0.05 m), set `geometric_success = True`.

`verified = sim_success AND geometric_success`. Both must pass.

If `verified = False`, proceed to stages 6a and 6b.

### Stage 6a — Primitive Normalization (`verification/primitive_normalizer.py`)

Convert `PlannedSketch.primitives_per_camera` to canonical `SketchPrimitives` objects
using the existing schema. Validate using `validate_primitives`. Log any validation
errors. This normalized form is what gets passed to stage 6b.

### Stage 6b — LLM Feedback (`verification/llm_feedback.py`)

If `config.use_mock_planner = True`: skip the API call. Return this template string:
`"Trajectory did not satisfy verification. Re-examine target object position and adjust
approach height and goal coordinates."` This is sufficient for the mock-only testing
phase.

If `config.use_mock_planner = False`: call o3 with the rendered planned sketch image for
the `agentview` camera alongside the task instruction. Ask it to identify the specific
failure mode from this list: wrong grasp position, wrong goal position, wrong path shape,
trajectory too short, collision. Return a one-paragraph corrective instruction suitable
for direct injection into the planner prompt.

---

## Orchestrator and Retry Loop (`agents/orchestrator.py`)

```python
def run(self, hdf5_path: str, task_instruction: str, config: Path1Config) -> OrchestratorResult:
    env, _, _ = load_libero_env(hdf5_path)
    scene_context = self.scene_agent.run(env, task_instruction, config)
    feedback = None

    for attempt in range(config.max_retries):
        trajectory = self.planner_agent.run(scene_context, config, feedback=feedback)
        verify_result = self.verifier_agent.run(trajectory, scene_context, env, config)

        if verify_result.verified:
            self.executor.execute(trajectory, env, config)
            self._save_demo(trajectory, verify_result, config.output_dir)
            return OrchestratorResult(
                success=True,
                attempts=attempt + 1,
                final_trajectory=trajectory,
                final_verify_result=verify_result,
            )

        feedback = verify_result.feedback
        self._log_attempt(attempt, verify_result)

    return OrchestratorResult(
        success=False,
        attempts=config.max_retries,
        final_trajectory=trajectory,
        final_verify_result=verify_result,
    )
```

`_log_attempt` must write one JSON object per line to `{config.output_dir}/attempts.jsonl`,
recording: attempt number, `sim_success`, `geometric_success`, `distance_to_goal`, and
the feedback string.

---

## Robot Executor (`execution/robot_executor.py`)

The trajectory contains absolute 6-DoF poses. LIBERO `env.step()` expects delta actions.
At each step, compute `delta = target_pose[:6] - current_eef_pose[:6]`. The gripper
dimension is passed directly (no delta). Call `env.step(np.append(delta, gripper_state))`
for each action. Capture observations, rewards, and the `done` flag at each step.

On success, save the rollout to `{config.output_dir}/demo_{timestamp}.hdf5`. The HDF5
structure must match the input demo format so it is loadable by existing LIBERO utilities
and usable for downstream behavioral cloning.

---

## CLI (`tools/run_path1.py`)

```
python -m path1_execution.tools.run_path1 \
  --hdf5 /path/to/demo.hdf5 \
  --output ./outputs/path1 \
  --demo-index 0 \
  [--mock | --no-mock] \
  --max-retries 5 \
  --gpu 0
```

`--mock` is the default. It sets `config.use_mock_planner = True`. `--no-mock` sets it
to `False` and activates the o3 planner. Mirror the argument structure of
`sketch_anything/tools/run_pipeline.py` exactly, including `--gpu` for
`CUDA_VISIBLE_DEVICES`.

---

## Implementation Order and Verification Gates

Do not proceed to the next step until the current step's gate passes.

**Step 1 — `config.py`**
Gate: `python -c "from path1_execution.config import *; print('ok')"` exits cleanly.

**Step 2 — SceneAgent**
Implement `scene/reconstruction.py`, `scene/object_extraction.py`,
`agents/scene_agent.py`. Run `tests/test_scene.py`.
Gate: Test loads a real LIBERO HDF5, calls `SceneAgent.run()`, and confirms
`SceneContext.object_registry` contains at least one entry with a valid non-zero
`centroid_3d`.

**Step 3 — Mock Planner + PlannerAgent**
Implement `planning/mock_planner.py`, `planning/prompts.py`, `agents/planner_agent.py`.
Run `tests/test_planner.py`.
Gate: Given a mock `SceneContext` with a known object at `[0.5, 0.0, 0.8]`, the returned
`Trajectory` has at least 3 actions, all actions are `(7,)` arrays, and
`grasp_indices` is non-empty.

**Step 4 — Trajectory Projector**
Implement `projection/trajectory_projector.py`. Run `tests/test_projector.py`.
Gate: All projected 2D points fall within `[0.0, 1.0]` for all cameras. Save one
rendered image per camera to `outputs/path1/debug/` for visual inspection. Confirm
arrows and circles are visible.

**Step 5 — Verifier**
Implement `verification/verifier.py`, `verification/primitive_normalizer.py`,
`agents/verifier_agent.py`. Run `tests/test_verifier.py`.
Gate: A trajectory that ends within 0.03 m of the goal returns `verified=True`. A
trajectory that ends 0.20 m from the goal returns `verified=False` with a non-None
`feedback` string.

**Step 6 — Orchestrator + Retry Loop**
Implement `agents/orchestrator.py`, `pipeline.py`. Run full end-to-end test with mock
planner on three distinct LIBERO tasks.
Gate: All three tasks produce `OrchestratorResult(success=True)` within `max_retries`.
`attempts.jsonl` exists and is valid JSON-lines.

**Step 7 — Robot Executor**
Implement `execution/robot_executor.py`. Run a verified trajectory through the executor.
Gate: LIBERO env reaches `done=True`. Output HDF5 is loadable with `h5py.File(path, "r")`
without errors.

**Step 8 — CLI**
Implement `tools/run_path1.py`.
Gate: `python -m path1_execution.tools.run_path1 --hdf5 <path> --mock` runs and exits
with code 0, producing a valid output HDF5.

**Step 9 — LLM Planner (o3)**
Implement `planning/llm_planner.py`, `verification/llm_feedback.py`. Set
`use_mock_planner=False`. Run one LIBERO task end-to-end.
Gate: Task completes with `success=True`. Confirm API was called (check logs). Confirm
code-output parsing worked (action list produced from executed Python function).

---

## Definition of Done

`path1_execution/` is complete when all gates above pass and the following checklist is
confirmed:

- [ ] `test_scene.py` passes on three distinct LIBERO tasks.
- [ ] `test_planner.py` passes for mock output (o3 output tested in step 9).
- [ ] `test_projector.py` passes; rendered debug images are visually correct.
- [ ] `test_verifier.py` correctly classifies success and failure cases.
- [ ] Full end-to-end run with mock planner succeeds on three LIBERO tasks.
- [ ] Full end-to-end run with o3 planner succeeds on at least one LIBERO task.
- [ ] All output HDF5 files are loadable and structurally valid.
- [ ] `attempts.jsonl` logs are present and valid for every run.

**The `path1_execution/` phase ends here. No merge with `sketch_anything/` should be
initiated until this checklist is reviewed and signed off externally.**

---

## Design Decisions and Rationale

These explanations are embedded here because Claude Code has no access to external papers
or project documentation. Do not look for references elsewhere.

**Why mock planner first?** It allows the full pipeline — scene extraction, projection,
verification, retry loop, execution, HDF5 saving — to be tested without any API cost.
Every bug in orchestration, data flow, and verification logic can be found and fixed
before the o3 planner introduces API latency and cost.

**Why o3 for planning, and why code output?** L2T (arxiv 2310.11604) demonstrates a 6×
improvement from code output over raw numeric lists: 60% vs 10% task success on identical
tasks with GPT-4. Complex trajectory shapes (arcs, curved lift motions, rotations) are
difficult to express as raw numeric lists; a Python function can call numpy to generate
these programmatically. o3's stronger reasoning capability handles spatial decomposition
and gripper strategy selection more reliably than weaker models. The `def get_trajectory()`
convention and sandboxed exec() call in `llm_planner.py` implement exactly the approach
validated in L2T, applied here in the LIBERO sim context.

**Why sim success signal + geometric threshold for verification?** Both checks are free
(zero API calls). The LIBERO sim success signal can fire in edge cases; the geometric
threshold is an independent sanity check on the final EEF position. Requiring both
ensures robustness without LLM costs during mock testing. LLM-assisted feedback in
stage 6b is only added when the o3 planner is active, since at that point API costs are
already being incurred anyway.

**Panda gripper max opening is 0.08m (hard physical constraint):**
From MuJoCo joint limits: gripper0_finger_joint1: [0.0, 0.04] and
gripper0_finger_joint2: [-0.04, 0.0] — total span = 0.08m. Objects whose
bbox_extents[0] > 0.08m cannot be pinch-grasped from the sides. For those objects
the LLM must use a cradle approach: approach CLOSED, descend below centroid, OPEN
to spread fingers underneath, lift OPEN. This decision rule is encoded in
SYSTEM_PROMPT via the GRIPPER PHYSICS section so the LLM applies it autonomously
based on bbox_extents from the registry. It generalizes to all task types.

**Why 150 action steps maximum?** At ~0.01m/step, a pick-and-place task needs ~100 steps
for 0.8–1.0m total EEF travel. L2T targets 80–100 steps for complex tasks. Cap at
150 to handle longer transport distances while preventing runaway generation.

**Why 5 retries?** Conservative relative to prior work on sub-goal retry loops (which
used 30 retries per sub-goal). Five is appropriate for full-task retries and keeps total
API cost bounded during testing.

**Why ArrowPrimitive + CirclePrimitive for trajectory visualization?** This matches the
primitive schema already used in `sketch_anything/` (Path 2). The two-primitive
vocabulary is the established representation for robot visual instructions: arrows encode
directed movement from a start to an end position with intermediate waypoints; circles
encode spatial affordances such as grasp points, release points, and rotation pivots.
Using the same schema now means the Path 1 and Path 2 outputs will be directly comparable
when the future merge phase begins — but that is not a concern for this phase.

**Why `path1_execution/` as a separate package?** Isolation ensures that every stage of
Path 1 can be independently tested without any dependency on whether Path 2 is working.
It also makes the eventual merge explicit and reviewable rather than something that
accumulates through incremental edits to shared files.