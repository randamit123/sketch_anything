#!/usr/bin/env python3
"""Run the full Sketch Anything pipeline on a LIBERO task.

Loads a LIBERO environment from a demo HDF5 file, builds a real object
registry from MuJoCo state, generates sketch primitives (via VLM or mock),
renders annotated scene images, and saves a video of the demo playback.

Usage (VLM mode -- requires GPU):
    python -m sketch_anything.tools.run_pipeline \
        --hdf5 /path/to/demo.hdf5 \
        --output outputs/run \
        --demo-index 0

Usage (mock mode -- no GPU needed):
    python -m sketch_anything.tools.run_pipeline \
        --hdf5 /path/to/demo.hdf5 \
        --output outputs/run \
        --demo-index 0 \
        --mock

Outputs:
    {output}/
        agentview_original.png
        agentview_annotated.png
        robot0_eye_in_hand_original.png
        robot0_eye_in_hand_annotated.png
        demo_video_agentview.mp4
        demo_video_eye_in_hand.mp4
        primitives_agentview.json
        primitives_eye_in_hand.json
        object_registry.json
        metadata.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("run_pipeline")


def _resolve_bddl_path(bddl_basename: str) -> str:
    """Resolve a BDDL filename to its full path on this machine.

    Searches LIBERO's configured bddl_files directory for the file.
    Falls back to a recursive glob if the LIBERO config is unavailable.

    Args:
        bddl_basename: Just the filename, e.g. "turn_on_the_stove.bddl".

    Returns:
        Absolute path to the BDDL file, or empty string if not found.
    """
    # Try LIBERO's path config first
    try:
        from libero.libero import get_libero_path
        bddl_root = get_libero_path("bddl_files")
        matches = glob.glob(
            os.path.join(bddl_root, "**", bddl_basename), recursive=True
        )
        if matches:
            logger.info(f"Resolved BDDL via LIBERO config: {matches[0]}")
            return matches[0]
    except Exception as e:
        logger.warning(f"LIBERO path config unavailable: {e}")

    # Fallback: search relative to this file's project root
    project_root = Path(__file__).resolve().parent.parent.parent
    libero_dir = project_root / "LIBERO"
    if libero_dir.exists():
        matches = list(libero_dir.rglob(bddl_basename))
        if matches:
            result = str(matches[0])
            logger.info(f"Resolved BDDL via project search: {result}")
            return result

    return ""


def load_libero_env(hdf5_path: str):
    """Load a LIBERO environment from an HDF5 demo file.

    Reads env_args and bddl_file_name from the HDF5 attributes and
    initialises an OffScreenRenderEnv.

    The HDF5 env_args has a nested structure::

        {"type": 1, "env_name": "...", "bddl_file": "...",
         "env_kwargs": {"bddl_file_name": "...", ...}}

    OffScreenRenderEnv expects the flat env_kwargs dict, not the
    outer wrapper.

    Args:
        hdf5_path: Path to a LIBERO HDF5 demo file.

    Returns:
        (env, task_instruction, env_args_raw) tuple.
    """
    from libero.libero.envs import OffScreenRenderEnv

    f = h5py.File(hdf5_path, "r")
    env_args_raw = json.loads(f["data"].attrs["env_args"])
    problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
    task_instruction = problem_info.get("language_instruction", "unknown")

    # Also grab the direct bddl_file_name attr if available
    bddl_attr = f["data"].attrs.get("bddl_file_name", None)
    if isinstance(bddl_attr, bytes):
        bddl_attr = bddl_attr.decode("utf-8")

    f.close()

    # The HDF5 env_args is nested: extract the inner env_kwargs
    # which is what OffScreenRenderEnv actually needs.
    if "env_kwargs" in env_args_raw:
        env_kwargs = dict(env_args_raw["env_kwargs"])
    else:
        # Fallback: assume env_args is already flat
        env_kwargs = dict(env_args_raw)

    # Make sure bddl_file_name is present
    if "bddl_file_name" not in env_kwargs:
        # Try the outer bddl_file key, then the HDF5 attr
        bddl_file = env_args_raw.get("bddl_file") or bddl_attr
        if bddl_file:
            env_kwargs["bddl_file_name"] = bddl_file
        else:
            raise ValueError(
                "Could not find bddl_file_name in HDF5 env_args. "
                f"Keys available: {list(env_args_raw.keys())}"
            )

    # The bddl_file_name stored in the HDF5 is often a relative path from
    # the original machine (e.g. "chiliocosm/bddl_files/libero_goal/X.bddl").
    # If it doesn't exist on this machine, resolve it via LIBERO's path config.
    bddl_path = env_kwargs["bddl_file_name"]
    if not os.path.exists(bddl_path):
        bddl_basename = os.path.basename(bddl_path)
        logger.info(
            f"BDDL path not found: {bddl_path}. "
            f"Resolving '{bddl_basename}' via LIBERO config..."
        )
        resolved = _resolve_bddl_path(bddl_basename)
        if resolved:
            env_kwargs["bddl_file_name"] = resolved
        else:
            raise FileNotFoundError(
                f"Cannot find BDDL file '{bddl_basename}'. "
                f"Stored path was: {bddl_path}"
            )

    # ControlEnv.__init__ builds controller_configs internally from the
    # "controller" parameter.  The HDF5 env_kwargs often also contains a
    # "controller_configs" key which would collide.  Strip it (and any
    # other keys that are not accepted as named parameters) to avoid
    # "got multiple values for keyword argument" errors.
    _VALID_ENV_KEYS = {
        "bddl_file_name",
        "robots",
        "controller",
        "gripper_types",
        "initialization_noise",
        "use_camera_obs",
        "has_renderer",
        "has_offscreen_renderer",
        "render_camera",
        "render_collision_mesh",
        "render_visual_mesh",
        "render_gpu_device_id",
        "control_freq",
        "horizon",
        "ignore_done",
        "hard_reset",
        "camera_names",
        "camera_heights",
        "camera_widths",
        "camera_depths",
        "camera_segmentations",
        "renderer",
        "renderer_config",
    }
    env_kwargs = {k: v for k, v in env_kwargs.items() if k in _VALID_ENV_KEYS}

    # Override camera sizes to 256x256 for consistency
    env_kwargs["camera_heights"] = 256
    env_kwargs["camera_widths"] = 256

    logger.info(f"Task: {task_instruction}")
    logger.info(f"BDDL: {env_kwargs['bddl_file_name']}")
    logger.info("Creating LIBERO environment...")

    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(0)
    env.reset()

    return env, task_instruction, env_args_raw


def load_demo_actions(hdf5_path: str, demo_index: int = 0):
    """Load the action sequence and initial state from an HDF5 demo.

    Args:
        hdf5_path: Path to the HDF5 demo file.
        demo_index: Which demo to load (default 0).

    Returns:
        (initial_state, actions) tuple where initial_state is the MuJoCo
        state vector and actions is an (T, action_dim) array.
    """
    f = h5py.File(hdf5_path, "r")
    demo_key = f"demo_{demo_index}"

    if demo_key not in f["data"]:
        available = sorted(f["data"].keys())
        f.close()
        raise ValueError(
            f"Demo '{demo_key}' not found. Available: {available}"
        )

    states = f[f"data/{demo_key}/states"][()]
    actions = f[f"data/{demo_key}/actions"][()]
    initial_state = states[0]

    logger.info(f"Loaded {demo_key}: {actions.shape[0]} steps")
    f.close()

    return initial_state, actions


def set_env_state(env, initial_state: np.ndarray):
    """Reset the LIBERO environment to a specific MuJoCo state.

    Uses LIBERO's own ``set_init_state`` when available.  Falls back to
    manual qpos/qvel injection with padding if the state vector length
    differs from the current model (can happen across MuJoCo versions).

    Args:
        env: LIBERO OffScreenRenderEnv.
        initial_state: Full MuJoCo state vector from the demo.
    """
    # Try LIBERO's built-in method first
    try:
        env.set_init_state(initial_state)
        logger.info("Set env state via set_init_state")
        return
    except (ValueError, RuntimeError) as e:
        logger.warning(f"set_init_state failed ({e}), trying manual restore")

    # Manual fallback: handle dimension mismatches by padding/truncating
    sim = env.env.sim
    nq = sim.model.nq
    nv = sim.model.nv
    expected = nq + nv

    state = initial_state
    if len(state) < expected:
        # Pad with zeros (extra joints in current model get zero velocity)
        padded = np.zeros(expected)
        # Copy qpos (take min of both sizes)
        nq_src = min(nq, len(state))
        padded[:nq_src] = state[:nq_src]
        # Copy qvel from after qpos in source
        nv_src_start = nq_src
        nv_available = len(state) - nv_src_start
        nv_copy = min(nv, nv_available)
        if nv_copy > 0:
            padded[nq:nq + nv_copy] = state[nv_src_start:nv_src_start + nv_copy]
        state = padded
        logger.warning(
            f"State vector padded: {len(initial_state)} -> {expected} "
            f"(nq={nq}, nv={nv})"
        )
    elif len(state) > expected:
        state = state[:expected]
        logger.warning(
            f"State vector truncated: {len(initial_state)} -> {expected}"
        )

    try:
        sim.set_state_from_flattened(state)
        sim.forward()
        logger.info(f"Set env state manually: qpos={nq}, qvel={nv}")
    except Exception as e:
        logger.warning(
            f"Manual state restore failed ({e}). "
            f"Using default reset state instead."
        )


def capture_image(env, camera_name: str) -> np.ndarray:
    """Capture an RGB image from a LIBERO environment camera.

    Args:
        env: LIBERO OffScreenRenderEnv.
        camera_name: Camera identifier.

    Returns:
        RGB uint8 image array (H, W, 3).
    """
    obs = env.env._get_observations()
    key = f"{camera_name}_image"

    if key not in obs:
        available = [k for k in obs if k.endswith("_image")]
        raise KeyError(
            f"Camera key '{key}' not found. Available: {available}"
        )

    image = obs[key]

    # LIBERO images use bottom-left origin; flip to top-left
    if image.ndim == 3:
        image = np.flip(image, axis=0).copy()

    return image


def record_demo_video(
    env,
    actions: np.ndarray,
    camera_name: str,
    output_path: str,
    fps: int = 20,
) -> str:
    """Replay demo actions and record a video from a camera.

    Args:
        env: LIBERO OffScreenRenderEnv (already at initial state).
        actions: (T, action_dim) array of demo actions.
        camera_name: Camera to record from.
        output_path: Path for the output MP4 file.
        fps: Video frame rate.

    Returns:
        Path to the saved video file.
    """
    frames = []

    # Capture initial frame
    frames.append(capture_image(env, camera_name))

    # Step through actions
    for t in range(len(actions)):
        obs, reward, done, info = env.step(actions[t])

        # Capture frame from raw observations
        frame = capture_image(env, camera_name)
        frames.append(frame)

        if done:
            break

    # Write video
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    logger.info(f"Saved video ({len(frames)} frames): {output_path}")
    return output_path


def run_pipeline(
    hdf5_path: str,
    output_dir: str,
    demo_index: int = 0,
    use_mock: bool = False,
    camera_names=None,
    fps: int = 20,
    dump_bodies: bool = False,
    use_llm: bool = True,
    model_path: str = None,
    llm_model_path: str = None,
):
    """Run the full sketch annotation pipeline on a LIBERO demo.

    Steps:
        1. Load LIBERO env from HDF5
        2. Restore initial demo state
        3. Build object registry from MuJoCo ground truth
        4. Capture scene images from each camera
        5. Generate sketch primitives (VLM or mock)
        6. Validate and render annotated images
        7. Replay demo and save video
        8. Save all outputs

    Args:
        hdf5_path: Path to the LIBERO HDF5 demo file.
        output_dir: Directory for all outputs.
        demo_index: Which demo to use (default 0).
        use_mock: Use mock primitives instead of VLM (no GPU needed).
        camera_names: Camera list. Defaults to agentview + eye_in_hand.
        fps: Video frame rate.
        dump_bodies: If True, log all MuJoCo body and site names then exit.
        use_llm: If True, use LLM-based object resolution (default True).
        model_path: Local path to VLM model weights (overrides default HF name).
        llm_model_path: Local path to LLM resolver model weights.
    """
    from sketch_anything.config import Config
    from sketch_anything.registry.builder import build_object_registry
    from sketch_anything.rendering.config import RenderConfig
    from sketch_anything.rendering.renderer import build_legend_data, render_primitives
    from sketch_anything.validation.validator import validate_primitives
    from sketch_anything.vlm.config import VLMConfig
    from sketch_anything.vlm.generator import VLMPrimitiveGenerator

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if camera_names is None:
        camera_names = ["agentview", "robot0_eye_in_hand"]

    # ---- 1. Load environment ----
    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info("=" * 60)
    env, task_instruction, env_args_raw = load_libero_env(hdf5_path)
    initial_state, actions = load_demo_actions(hdf5_path, demo_index)

    # ---- 2. Restore initial state ----
    set_env_state(env, initial_state)

    # ---- Optional: dump all MuJoCo bodies/sites for debugging ----
    if dump_bodies:
        sim = env.sim
        logger.info("-" * 60)
        logger.info("DUMP: All MuJoCo bodies in this environment")
        logger.info("-" * 60)
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name:
                pos = sim.data.body_xpos[i]
                logger.info(f"  body[{i:3d}] '{name}'  pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        logger.info("-" * 60)
        logger.info("DUMP: All MuJoCo sites in this environment")
        logger.info("-" * 60)
        for i in range(sim.model.nsite):
            name = sim.model.site_id2name(i)
            if name:
                parent_body_id = sim.model.site_bodyid[i]
                parent_name = sim.model.body_id2name(parent_body_id)
                logger.info(f"  site[{i:3d}] '{name}'  parent_body='{parent_name}'")
        logger.info("-" * 60)
        logger.info("DUMP: objects_dict and fixtures_dict")
        logger.info("-" * 60)
        inner = env.env if hasattr(env, "env") else env
        obj_dict = getattr(inner, "objects_dict", None)
        fix_dict = getattr(inner, "fixtures_dict", None)
        if obj_dict:
            for k, v in obj_dict.items():
                root = getattr(v, "root_body", "?")
                logger.info(f"  objects_dict['{k}'] root_body='{root}'")
        else:
            logger.info("  objects_dict: None")
        if fix_dict:
            for k, v in fix_dict.items():
                root = getattr(v, "root_body", "?")
                logger.info(f"  fixtures_dict['{k}'] root_body='{root}'")
        else:
            logger.info("  fixtures_dict: None")
        logger.info("-" * 60)

        # Save body dump to file
        dump_path = Path(output_dir) / "mujoco_bodies.txt"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as df:
            for i in range(sim.model.nbody):
                name = sim.model.body_id2name(i)
                if name:
                    df.write(f"body {name}\n")
            for i in range(sim.model.nsite):
                name = sim.model.site_id2name(i)
                if name:
                    df.write(f"site {name}\n")
        logger.info(f"Saved body dump to: {dump_path}")
        logger.info("Exiting (--dump-bodies mode).")
        env.close()
        return

    # ---- 3. Build object registry from MuJoCo ----
    logger.info("-" * 60)
    logger.info("STAGE: Object Registry (source: MuJoCo ground-truth 3D poses)")
    logger.info("-" * 60)
    if llm_model_path:
        logger.info(f"Using local LLM resolver model path: {llm_model_path}")
    registries = build_object_registry(
        env, task_instruction, camera_names, 256, 256,
        use_llm=use_llm, llm_model_path=llm_model_path,
    )

    # Detailed logging for each camera registry
    for cam, reg in registries.items():
        logger.info(f"  [{cam}] {len(reg)} objects detected:")
        for obj_id, obj_data in reg.items():
            bbox = obj_data.get("bbox", [])
            center = obj_data.get("center", [])
            bbox_str = "[{:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(*bbox) if len(bbox) == 4 else str(bbox)
            center_str = "[{:.3f}, {:.3f}]".format(*center) if len(center) == 2 else str(center)
            logger.info(
                f"    - {obj_id} (label='{obj_data.get('label', obj_id)}') "
                f"bbox={bbox_str} center={center_str}"
            )
    if not any(registries.values()):
        logger.warning("  No objects detected in any view. Check name mapping.")

    # Save registry
    registry_path = out / "object_registry.json"
    registry_serializable = {}
    for cam, reg in registries.items():
        registry_serializable[cam] = {}
        for obj_id, obj_data in reg.items():
            entry = dict(obj_data)
            for k, v in entry.items():
                if isinstance(v, np.ndarray):
                    entry[k] = v.tolist()
            registry_serializable[cam][obj_id] = entry
    with open(registry_path, "w") as f:
        json.dump(registry_serializable, f, indent=2)
    logger.info(f"Saved registry: {registry_path}")

    # ---- 4. Capture images + generate/render per camera ----
    logger.info("-" * 60)
    logger.info("STAGE: Primitive Generation")
    logger.info("-" * 60)
    if use_mock:
        logger.info("MODE: --mock flag set. Using MOCK primitive generator.")
        logger.info("  -> VLM is NOT being queried. Primitives are hard-coded templates.")
        logger.info("  -> To use real VLM inference, remove the --mock flag.")
    else:
        logger.info("MODE: VLM inference (Qwen2.5-VL)")
        logger.info("  -> The VLM WILL be queried for each camera view.")
        logger.info("  -> Model: Qwen/Qwen2.5-VL-7B-Instruct")

    vlm_model = model_path if model_path else "Qwen/Qwen2.5-VL-7B-Instruct"
    if model_path:
        logger.info(f"Using local VLM model path: {model_path}")

    vlm_config = VLMConfig(
        model_name=vlm_model,
        max_tokens=2048,
        temperature=0.1,
        max_retries=3,
        use_constrained_decoding=False,
        device="cuda",
        use_mock=use_mock,
    )

    generator = VLMPrimitiveGenerator(vlm_config)
    render_config = RenderConfig()  # legend disabled by default now

    all_primitives = {}

    for cam_name in camera_names:
        logger.info(f"Processing camera: {cam_name}")
        registry = registries.get(cam_name, {})

        if not registry:
            logger.warning(f"  Empty registry for {cam_name}, skipping")
            continue

        # Capture scene image at initial state
        image = capture_image(env, cam_name)

        # Save original
        orig_path = out / f"{cam_name}_original.png"
        cv2.imwrite(str(orig_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"  Saved original: {orig_path.name}")

        # Generate primitives
        logger.info(f"  Generating primitives ({'MOCK' if use_mock else 'VLM'})...")
        t0 = time.time()
        try:
            primitives = generator.generate(
                image=image,
                object_registry=registry,
                task_instruction=task_instruction,
            )
            elapsed = time.time() - t0
            logger.info(
                f"  Generated {len(primitives.primitives)} primitives in {elapsed:.1f}s "
                f"({'mock template' if use_mock else 'VLM inference'})"
            )
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  Generation failed after {elapsed:.1f}s: {e}")
            continue

        # Log each primitive for visibility
        for i, prim in enumerate(primitives.primitives):
            prim_dict = prim.model_dump()
            ptype = prim_dict.get("type", "?")
            pstep = prim_dict.get("step", "?")
            if ptype == "circle":
                logger.info(
                    f"    [{i}] step={pstep} circle purpose={prim_dict.get('purpose')} "
                    f"radius={prim_dict.get('radius')}"
                )
            elif ptype == "arrow":
                logger.info(f"    [{i}] step={pstep} arrow")
            elif ptype == "gripper":
                logger.info(
                    f"    [{i}] step={pstep} gripper action={prim_dict.get('action')}"
                )

        # Validate
        validation = validate_primitives(primitives, registry)
        logger.info(f"  Validation: valid={validation.is_valid}")
        for err in validation.errors:
            logger.error(f"    Error: {err}")
        for warn in validation.warnings:
            logger.warning(f"    Warning: {warn}")

        # Render (legend is NOT drawn on image)
        logger.info("-" * 60)
        logger.info("STAGE: Rendering")
        logger.info("-" * 60)
        annotated = render_primitives(
            image=image.copy(),
            primitives=primitives,
            object_registry=registry,
            config=render_config,
        )

        # Save annotated image
        ann_path = out / f"{cam_name}_annotated.png"
        cv2.imwrite(str(ann_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        logger.info(f"  Saved annotated image (no legend overlay): {ann_path.name}")

        # Save primitives JSON
        prim_data = primitives.model_dump()
        prim_path = out / f"primitives_{cam_name}.json"
        with open(prim_path, "w") as f:
            json.dump(prim_data, f, indent=2)
        logger.info(f"  Saved primitives: {prim_path.name}")

        # Save legend as separate JSON
        legend_data = build_legend_data(primitives)
        legend_path = out / f"legend_{cam_name}.json"
        with open(legend_path, "w") as f:
            json.dump(legend_data, f, indent=2)
        logger.info(f"  Saved legend: {legend_path.name}")

        all_primitives[cam_name] = prim_data

    # ---- 5. Replay demo and save video ----
    logger.info("-" * 60)
    logger.info("STAGE: Demo Replay + Video Recording")
    logger.info("-" * 60)

    # Re-restore initial state before replay
    set_env_state(env, initial_state)

    for cam_name in camera_names:
        video_path = str(out / f"demo_video_{cam_name}.mp4")

        # We need to restore state before each camera recording
        # because stepping through actions changes the state
        set_env_state(env, initial_state)
        record_demo_video(env, actions, cam_name, video_path, fps=fps)

    # ---- 6. Save metadata ----
    metadata = {
        "hdf5_file": str(hdf5_path),
        "demo_index": demo_index,
        "task_instruction": task_instruction,
        "camera_names": camera_names,
        "num_demo_steps": int(actions.shape[0]),
        "use_mock": use_mock,
        "vlm_model": vlm_config.model_name if not use_mock else "mock",
        "generation_mode": "mock (hard-coded template)" if use_mock else "VLM inference",
        "registry_source": "MuJoCo ground-truth 3D poses projected to 2D",
        "primitives_per_view": {
            cam: len(p.get("primitives", []))
            for cam, p in all_primitives.items()
        },
    }

    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")

    # ---- Cleanup ----
    env.close()
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"All outputs saved to: {out}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Sketch Anything pipeline on a LIBERO demo"
    )
    parser.add_argument(
        "--hdf5", required=True,
        help="Path to LIBERO HDF5 demo file",
    )
    parser.add_argument(
        "--output", default="outputs/pipeline_run",
        help="Output directory (default: outputs/pipeline_run)",
    )
    parser.add_argument(
        "--demo-index", type=int, default=0,
        help="Demo index to use (default: 0)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock primitives instead of VLM (no GPU needed)",
    )
    parser.add_argument(
        "--cameras", nargs="*", default=None,
        help="Camera names (default: agentview robot0_eye_in_hand)",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Video frame rate (default: 20)",
    )
    parser.add_argument(
        "--dump-bodies", action="store_true",
        help="Log all MuJoCo body/site names and exit (for debugging object mapping)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM-based object resolution (use static mapping only)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Local path to VLM model weights (e.g. ~/models/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--llm-model-path", default=None,
        help="Local path to LLM resolver model weights (e.g. ~/models/Qwen2.5-1.5B-Instruct)",
    )
    args = parser.parse_args()

    run_pipeline(
        hdf5_path=args.hdf5,
        output_dir=args.output,
        demo_index=args.demo_index,
        use_mock=args.mock,
        camera_names=args.cameras,
        fps=args.fps,
        dump_bodies=args.dump_bodies,
        use_llm=not args.no_llm,
        model_path=args.model_path,
        llm_model_path=args.llm_model_path,
    )


if __name__ == "__main__":
    main()
