"""Stage 1 — 3D Reconstruction.

Captures camera images and matrices for all configured cameras, and builds
a point cloud from all MuJoCo body bounding boxes.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from sketch_anything.libero_utils.camera import get_camera_matrices
from sketch_anything.libero_utils.env import get_object_bbox_3d
from path1_execution.config import Path1Config

logger = logging.getLogger(__name__)


def reconstruct_scene(env, sim, config: Path1Config) -> dict:
    """Capture camera images, matrices, and build a point cloud.

    For each camera in config.camera_names:
      - Captures the RGB image via env observations.
      - Retrieves camera intrinsic/extrinsic matrices from sim.

    For the point cloud: iterates over all MuJoCo bodies, calls
    get_object_bbox_3d for each, and concatenates all corners.

    Args:
        env: LIBERO OffScreenRenderEnv instance.
        sim: MuJoCo simulation object (env.sim).
        config: Path1Config with camera_names, image_width, image_height.

    Returns:
        dict with keys:
            camera_images: camera_name -> np.ndarray (H, W, 3) uint8
            camera_matrices: camera_name -> (K, R, t)
            point_cloud: np.ndarray (N, 3) world-frame corners
    """
    from sketch_anything.libero_utils.env import get_camera_image

    camera_images: Dict[str, np.ndarray] = {}
    camera_matrices: Dict[str, Tuple] = {}

    for cam in config.camera_names:
        try:
            camera_images[cam] = get_camera_image(env, cam)
        except KeyError as e:
            logger.warning(f"Could not capture image for camera '{cam}': {e}")
            continue

        K, R, t = get_camera_matrices(sim, cam, config.image_width, config.image_height)
        camera_matrices[cam] = (K, R, t)

    # Build point cloud from all MuJoCo body bounding boxes
    all_corners = []
    for i in range(sim.model.nbody):
        body_name = sim.model.body_id2name(i)
        if not body_name:
            continue
        try:
            corners = get_object_bbox_3d(sim, body_name)  # (8, 3)
            all_corners.append(corners)
        except Exception as e:
            logger.debug(f"Skipping body '{body_name}' for point cloud: {e}")

    if all_corners:
        point_cloud = np.concatenate(all_corners, axis=0)  # (N, 3)
    else:
        point_cloud = np.zeros((0, 3), dtype=np.float64)

    return {
        "camera_images": camera_images,
        "camera_matrices": camera_matrices,
        "point_cloud": point_cloud,
    }
