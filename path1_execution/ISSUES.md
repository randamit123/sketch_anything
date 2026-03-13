# Path 1 — Known Issues and Review Notes

Last updated: 2026-03-12

This document records all identified issues, design gaps, and deferred concerns
discovered during implementation and verification. Organized by severity.

---

## CRITICAL (blocks real task success)

### 1. Object registry is empty for most LIBERO tasks

**File:** `scene/object_extraction.py`

**What happens:** `build_object_registry` is called with `use_llm=False` to avoid
needing a GPU. Without the LLM resolver, the static mapping in
`sketch_anything/registry/builder.py:LIBERO_OBJECT_MAPPING` must cover the task's
objects exactly. For many tasks (e.g. "open the middle drawer of the cabinet"), the
static mapping doesn't resolve "middle drawer" or "cabinet" to a MuJoCo body name, so
the registry comes back empty.

**Effect:** Mock planner's `sorted_labels` list is empty → falls back to the EEF position
as source centroid → trajectory misses by ~1.1m → `verified=False` on every attempt.

**Fix needed:**
- Either enable the LLM resolver (`use_llm=True`) when a GPU is available
- Or extend `LIBERO_OBJECT_MAPPING` in `sketch_anything/registry/builder.py` to cover
  all 10 libero_goal task objects (requires modifying sketch_anything — coordinate
  with Path 2 team)
- Or add a Path 1–local static mapping fallback in `object_extraction.py` that extends
  coverage without touching `sketch_anything/`

---

### 2. Demo initial state is never loaded from HDF5

**File:** `agents/orchestrator.py`, `agents/scene_agent.py`

**What happens:** `load_libero_env` creates the environment but doesn't set the demo's
initial state. `env.reset()` returns the environment's *default* initial state, not the
specific episode's initial state from the HDF5 file. The robot starts in a different
configuration than the demo's initial pose.

**CLAUDE.md spec:** `load_demo_actions(hdf5_path, demo_index)` returns
`(initial_state, actions)` and is listed in the imports table explicitly. It is never
called in the current implementation.

**Fix needed:** After `load_libero_env`, call:
```python
initial_state, _ = load_demo_actions(hdf5_path, demo_index)
env.set_init_state(initial_state)
obs = env.reset()
```
(or equivalent MuJoCo state reset — check `run_pipeline.py` for how it handles this)

This affects: `agents/orchestrator.py` (Orchestrator.run), `agents/scene_agent.py`
(SceneAgent.run). The `demo_index` parameter needs to be threaded through from the CLI.

---

### 3. Mock planner generates identical trajectory on every retry

**File:** `planning/mock_planner.py`

**What happens:** The mock planner is fully deterministic. `feedback` is passed in but
ignored — `generate_mock_trajectory` doesn't inspect it. All retry attempts produce the
same trajectory → same `distance_to_goal` → same failure → wasted retries.

**This is by design for mock mode** (per CLAUDE.md: feedback loop is tested to confirm
the loop runs, not that the mock improves). However, for meaningful end-to-end testing,
at least one retry should produce a different trajectory. A simple fix: add a small
random perturbation scaled by `retry_num` to the goal position on retry > 0.

**Not a bug for Stage 6 (mock-only) — becomes important when LLM planner is added.**

---

## SIGNIFICANT (correctness gap)

### 4. `pipeline.py` calls `load_libero_env` twice

**File:** `pipeline.py`, `agents/orchestrator.py`

**What happens:** When `task_instruction=None` (the CLI default), `pipeline.py:run_pipeline`
calls `load_libero_env` to extract the task string, then immediately discards the env.
`Orchestrator.run` then calls `load_libero_env` again, creating a second env.

**Effect:** Two full LIBERO environment initializations per run. ~3–5 seconds of wasted
startup time. No correctness impact.

**Fix:** Pass the env object through from `pipeline.py` to `Orchestrator.run`, or
refactor `Orchestrator.run` to accept an optional env parameter.

---

### 5. `run_trial` resets env state — consumes the verified trajectory's initial conditions

**File:** `execution/robot_executor.py`

**What happens:** `run_trial` starts with `env.reset()`, which resets to the default
initial state. If `execute_trajectory` (called after verification) also resets with
`env.reset()`, the two executions are independent — each starts from scratch. This is
consistent but means the "verified" run and the "final execution" run may see slightly
different physics (if env has any stochasticity), though LIBERO is deterministic.

**Not a bug for deterministic sim** — but worth noting if stochastic physics is added.

---

### 6. `VerifierAgent` projection uses camera matrices from pre-reset SceneContext

**File:** `agents/verifier_agent.py`, `projection/trajectory_projector.py`

**What happens:** `scene_context.camera_matrices` were captured right after the first
`env.reset()` in `SceneAgent.run`. When `run_trial` calls `env.reset()` again, a new
`MjSim` is created. Camera matrices are stable across resets (they depend only on the
camera's XML definition, not simulation state), so in practice this is fine. But it is
a hidden assumption — if camera extrinsics could change (e.g., head-mounted cameras),
this would silently produce wrong projections.

**No immediate fix needed** — LIBERO cameras are fixed. Document the assumption.

---

### 7. `run_trial` loop index `i` used after loop when trajectory is empty

**File:** `execution/robot_executor.py` (lines 49–54)

**What happens:** The `logger.info` at the end uses `num_steps` which is set to `i + 1`
inside the loop. If `trajectory.actions` is empty, the loop never executes and `num_steps`
remains `0`, which is correct. However, the original code (before the fix) used `i + 1`
directly after the loop — if actions was empty this would have been a `NameError`. The
fix introduced `num_steps = 0` as a pre-assignment, which is correct.

**Status: Fixed** during Sub-Agent 4 implementation.

---

## DEFERRED (by design)

### 8. LLM planner (`planning/llm_planner.py`) not implemented

**Status:** Intentionally deferred per CLAUDE.md Step 9. All mock-based gates must pass
before o3 is introduced. `PlannerAgent.run` raises `NotImplementedError` when
`use_mock_planner=False`.

---

### 9. LLM feedback (`verification/llm_feedback.py`) returns template string only

**Status:** Intentionally deferred. Returns:
`"Trajectory did not satisfy verification. Re-examine target object position and adjust approach height and goal coordinates."`
The o3 vision-based feedback path raises `NotImplementedError`. This is correct for
mock-only testing phase.

---

### 10. HDF5 output format is not fully LIBERO-compatible

**File:** `execution/robot_executor.py:save_rollout_hdf5`

**What happens:** The saved HDF5 has the structure `data/demo_0/{actions, rewards, dones, obs/*}`.
LIBERO's standard HDF5 format includes additional top-level metadata: `data.attrs["env_name"]`,
`data.attrs["env_version"]`, `data.attrs["problem_info"]`, and per-demo `env_init_state`.
Without these, the file may fail to load with `robosuite.utils.dataset_utils` or LIBERO's
playback utilities, even though `h5py.File(path, "r")` succeeds.

**Fix needed before Step 8 gate:** Add env metadata to the HDF5. Capture `env_name` and
`problem_info` from the source HDF5 file at orchestrator init time and write to output.

---

### 11. `object_extraction.py` uses `label` as dict key — not `object_id`

**File:** `scene/object_extraction.py`

**What happens:** The returned dict maps `label → ObjectInfo` (e.g. `"bowl" → ...`).
If two objects have the same label (unlikely in LIBERO but possible), the second one
silently overwrites the first.

**Low risk** in current scope. Would need deduplication if tasks have
multiple instances of the same object type.

---

### 12. `mock_planner.py` sorts objects alphabetically to determine source/goal

**File:** `planning/mock_planner.py`

**What happens:** Source = alphabetically first label, goal = alphabetically second.
For "pick up alphabet_soup and place in basket": `a < b` → source=alphabet_soup, goal=basket. ✓
For "put the bowl on the plate": `b < p` → source=bowl, goal=plate. ✓
For "push the plate to the stove": `p=plate < s=stove` → source=plate, goal=stove. ✓

Alphabetical order happens to be correct for currently tested tasks but will fail for
tasks where the source label sorts after the goal label (e.g. "pick up wine and put in
rack": `w > r` → would pick up rack and place at wine).

**Fix for LLM planner phase:** The LLM planner will parse the task instruction to determine
source/goal explicitly. Mock planner alphabetical sort is acceptable for now.

---

## ENVIRONMENT / SETUP NOTES

### 13. `sketch_anything/` pre-existing modifications

The files listed below were modified *before* `path1_execution/` was created (confirmed
by initial `git status`). Path 1 implementation made no changes to these files:
```
sketch_anything/libero_utils/camera.py
sketch_anything/registry/builder.py
sketch_anything/rendering/renderer.py
sketch_anything/rendering/resolver.py
sketch_anything/tests/test_pipeline.py
sketch_anything/tests/test_renderer.py
sketch_anything/tests/test_validation.py
sketch_anything/tools/run_pipeline.py
sketch_anything/validation/validator.py
sketch_anything/vlm/config.py
sketch_anything/vlm/generator.py
sketch_anything/vlm/prompt.py
```

### 14. `conda run -n sketch` required for all runs

The `libero` conda env has Transformers 4.46.3 (too old). All pipeline commands must use
`sketch` env. The `sketch` env has LIBERO accessible via `.pth` file.

---

## DEFINITION OF DONE STATUS

Last updated: 2026-03-12

| Gate | Status |
|---|---|
| Step 1: `config.py` imports cleanly | ✅ PASS |
| Step 2: `test_scene.py` passes | ✅ PASS (1/1) |
| Step 3: `test_planner.py` passes | ✅ PASS (19/19) |
| Step 4: `test_projector.py` passes | ✅ PASS (3/3) |
| Step 5: `test_verifier.py` passes | ✅ PASS (6/6) |
| Step 6: Orchestrator + retry loop confirmed | ✅ PASS (loop runs, jsonl valid) |
| Step 7: Robot executor HDF5 loadable | ⚠️ PARTIAL — `h5py.File(path,"r")` works, but full LIBERO metadata missing (Issue #10) |
| Step 8: CLI exits code 0 on success | ⚠️ BLOCKED — mock planner generates pick-and-place but LIBERO tasks need task-specific motions (drawer pull, etc.) |
| Step 9: LLM planner (o3) | ⚠️ IMPLEMENTED, BLOCKED ON BILLING — `planning/llm_planner.py` is complete; API call reaches OpenAI but returns 429 insufficient_quota (see Issue #15) |

---

## CORRECTIONS TO EARLIER ANALYSIS

### Issue #1 was misdiagnosed

The object registry is NOT empty. `build_object_registry` with `use_llm=False` correctly
resolves objects via static mapping for all tested LIBERO tasks. The 1.1354m distance
failure has a different root cause: the mock planner generates a pick-and-place
trajectory against task-specific targets (e.g. treating the cabinet as a graspable
object), and the robot physically cannot execute those motions in simulation. The 1.1354m
is the distance from wherever the robot ends up to the goal centroid, not an indicator
of wrong object coordinates.

**Issue #1 is closed.** Static mapping works correctly.

---

## NEW ISSUES

### 15. OpenAI API key has insufficient quota (billing issue)

**Status:** The LLM planner implementation is complete and the API call is correctly
formed. OpenAI returns HTTP 429 `insufficient_quota`, meaning the account associated
with the key needs billing/credits added at platform.openai.com.

**Fix:** Add billing credits to the OpenAI account. No code changes needed.

### 16. Issue #2 is now fixed: demo initial state loaded from HDF5

`load_demo_actions` and `set_env_state` are now called in `Orchestrator.run` before
`SceneAgent.run`. `demo_index` is threaded through from CLI → pipeline.py →
Orchestrator.run. **Issue #2 is closed.**
