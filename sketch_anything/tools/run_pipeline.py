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
import json
import logging
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

    # Override camera sizes to 256x256 for consistency
    env_kwargs["camera_heights"] = 256
    env_kwargs["camera_widths"] = 256

    logger.info(f"Task: {task_instruction}")
    logger.info(f"BDDL: {env_kwargs.get('bddl_file_name', 'unknown')}")
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

    Args:
        env: LIBERO OffScreenRenderEnv.
        initial_state: Full MuJoCo state vector from the demo.
    """
    # robosuite environments store the sim on env.env.sim
    sim = env.env.sim

    # Split state into qpos and qvel
    nq = sim.model.nq
    nv = sim.model.nv

    if len(initial_state) >= nq + nv:
        qpos = initial_state[:nq]
        qvel = initial_state[nq:nq + nv]
        sim.set_state_from_flattened(np.concatenate([qpos, qvel]))
        sim.forward()
        logger.info(f"Set env state: qpos={nq}, qvel={nv}")
    else:
        logger.warning(
            f"State vector length {len(initial_state)} < expected "
            f"{nq + nv}. Skipping state restoration."
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
    """
    from sketch_anything.config import Config
    from sketch_anything.registry.builder import build_object_registry
    from sketch_anything.rendering.config import RenderConfig
    from sketch_anything.rendering.renderer import render_primitives
    from sketch_anything.validation.validator import validate_primitives
    from sketch_anything.vlm.config import VLMConfig
    from sketch_anything.vlm.generator import VLMPrimitiveGenerator

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if camera_names is None:
        camera_names = ["agentview", "robot0_eye_in_hand"]

    # ---- 1. Load environment ----
    env, task_instruction, env_args_raw = load_libero_env(hdf5_path)
    initial_state, actions = load_demo_actions(hdf5_path, demo_index)

    # ---- 2. Restore initial state ----
    set_env_state(env, initial_state)

    # ---- 3. Build object registry from MuJoCo ----
    logger.info("Building object registry from MuJoCo state...")
    registries = build_object_registry(
        env, task_instruction, camera_names, 256, 256
    )

    # Log what was found
    for cam, reg in registries.items():
        obj_names = list(reg.keys())
        logger.info(f"  {cam}: {len(obj_names)} objects -> {obj_names}")

    # Save registry
    registry_path = out / "object_registry.json"
    # Convert for JSON serialization (numpy arrays -> lists)
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
    vlm_config = VLMConfig(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens=2048,
        temperature=0.1,
        max_retries=3,
        use_constrained_decoding=False,
        device="cuda",
        use_mock=use_mock,
    )

    generator = VLMPrimitiveGenerator(vlm_config)
    render_config = RenderConfig()

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
        logger.info(f"  Generating primitives ({'mock' if use_mock else 'VLM'})...")
        t0 = time.time()
        try:
            primitives = generator.generate(
                image=image,
                object_registry=registry,
                task_instruction=task_instruction,
            )
            elapsed = time.time() - t0
            logger.info(f"  Generated {len(primitives.primitives)} primitives in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  Generation failed after {elapsed:.1f}s: {e}")
            continue

        # Validate
        validation = validate_primitives(primitives, registry)
        logger.info(f"  Validation: valid={validation.is_valid}")
        for err in validation.errors:
            logger.error(f"    Error: {err}")
        for warn in validation.warnings:
            logger.warning(f"    Warning: {warn}")

        # Render
        annotated = render_primitives(
            image=image.copy(),
            primitives=primitives,
            object_registry=registry,
            config=render_config,
        )

        # Save annotated image
        ann_path = out / f"{cam_name}_annotated.png"
        cv2.imwrite(str(ann_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        logger.info(f"  Saved annotated: {ann_path.name}")

        # Save primitives JSON
        prim_data = primitives.model_dump()
        prim_path = out / f"primitives_{cam_name}.json"
        with open(prim_path, "w") as f:
            json.dump(prim_data, f, indent=2)
        logger.info(f"  Saved primitives: {prim_path.name}")

        all_primitives[cam_name] = prim_data

    # ---- 5. Replay demo and save video ----
    logger.info("Replaying demo for video recording...")

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
    logger.info("Pipeline complete.")
    logger.info(f"All outputs saved to: {out}")


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
    args = parser.parse_args()

    run_pipeline(
        hdf5_path=args.hdf5,
        output_dir=args.output,
        demo_index=args.demo_index,
        use_mock=args.mock,
        camera_names=args.cameras,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
