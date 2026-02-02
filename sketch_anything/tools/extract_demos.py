#!/usr/bin/env python3
"""Extract first frames and demo videos from LIBERO HDF5 demo files.

Usage:
    python -m sketch_anything.tools.extract_demos \
        --hdf5 /path/to/task_demo.hdf5 \
        --output outputs/demos \
        --demos 0 1 2 \
        --fps 20

Produces for each demo:
    {output}/demo_{i}_agentview_frame0.png
    {output}/demo_{i}_eye_in_hand_frame0.png
    {output}/demo_{i}_agentview.mp4
    {output}/demo_{i}_eye_in_hand.mp4
    {output}/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def extract_demo(
    hdf5_path: str,
    output_dir: str,
    demo_indices=None,
    fps: int = 20,
    save_video: bool = True,
) -> dict:
    """Extract first frames and optional videos from an HDF5 demo file.

    Args:
        hdf5_path: Path to the LIBERO HDF5 demo file.
        output_dir: Directory to save outputs.
        demo_indices: Which demo indices to extract. None = first 3.
        fps: Frame rate for output videos.
        save_video: Whether to also save MP4 videos of the full demos.

    Returns:
        Dict with metadata about the extraction.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    f = h5py.File(hdf5_path, "r")

    # Extract metadata
    problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
    task_instruction = problem_info.get("language_instruction", "unknown")
    image_convention = f["data"].attrs.get("macros_image_convention", "opengl")
    flip = image_convention == "opengl"

    all_demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

    if demo_indices is None:
        demo_indices = list(range(min(3, len(all_demos))))

    camera_keys = ["agentview_rgb", "eye_in_hand_rgb"]
    camera_short = {"agentview_rgb": "agentview", "eye_in_hand_rgb": "eye_in_hand"}

    metadata = {
        "hdf5_file": str(hdf5_path),
        "task_instruction": task_instruction,
        "image_convention": image_convention,
        "num_demos_total": len(all_demos),
        "extracted_demos": [],
    }

    for idx in demo_indices:
        demo_key = f"demo_{idx}"
        if demo_key not in f["data"]:
            print(f"Warning: {demo_key} not found, skipping")
            continue

        demo = f[f"data/{demo_key}"]
        n_steps = demo["actions"].shape[0]
        print(f"Extracting {demo_key}: {n_steps} steps")

        demo_info = {"demo_index": idx, "num_steps": n_steps, "frames": {}, "videos": {}}

        for cam_key in camera_keys:
            obs_key = f"obs/{cam_key}"
            if obs_key not in demo:
                print(f"  Warning: {obs_key} not found")
                continue

            images = demo[obs_key][()]  # (T, H, W, 3)
            cam_name = camera_short[cam_key]

            # Flip if OpenGL convention (origin bottom-left -> top-left)
            if flip:
                images = images[:, ::-1, :, :]

            # Save first frame
            frame0 = images[0]
            frame_path = out / f"{demo_key}_{cam_name}_frame0.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
            demo_info["frames"][cam_name] = str(frame_path)
            print(f"  Saved: {frame_path.name}")

            # Save video
            if save_video:
                video_path = out / f"{demo_key}_{cam_name}.mp4"
                h, w = images.shape[1], images.shape[2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
                for frame in images:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                demo_info["videos"][cam_name] = str(video_path)
                print(f"  Saved: {video_path.name}")

        metadata["extracted_demos"].append(demo_info)

    f.close()

    # Save metadata
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as mf:
        json.dump(metadata, mf, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"Task: {task_instruction}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract LIBERO demo frames and videos")
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 demo file")
    parser.add_argument("--output", default="outputs/demos", help="Output directory")
    parser.add_argument("--demos", nargs="*", type=int, default=None, help="Demo indices (default: 0 1 2)")
    parser.add_argument("--fps", type=int, default=20, help="Video frame rate")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    args = parser.parse_args()

    extract_demo(
        hdf5_path=args.hdf5,
        output_dir=args.output,
        demo_indices=args.demos,
        fps=args.fps,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
