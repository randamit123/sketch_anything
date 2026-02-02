"""Camera projection utilities for LIBERO / MuJoCo environments.

Handles 3D-to-2D projection using MuJoCo camera parameters with the
MuJoCo-to-OpenCV convention conversion.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def get_camera_matrices(
    sim,
    camera_name: str,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get camera intrinsic and extrinsic matrices from MuJoCo sim.

    Args:
        sim: MuJoCo simulation object (env.sim).
        camera_name: Name of the camera in the MuJoCo model.
        image_width: Width of rendered images in pixels.
        image_height: Height of rendered images in pixels.

    Returns:
        K: 3x3 intrinsic matrix.
        R: 3x3 rotation matrix (world to camera, OpenCV convention).
        t: 3x1 translation vector.
    """
    camera_id = sim.model.camera_name2id(camera_name)

    # Extrinsics from MuJoCo
    cam_pos = sim.data.cam_xpos[camera_id].copy()
    cam_mat = sim.data.cam_xmat[camera_id].reshape(3, 3).copy()

    # MuJoCo camera convention: -Z forward, Y up
    # OpenCV convention: Z forward, -Y up
    R_mujoco_to_cv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float64)

    R = R_mujoco_to_cv @ cam_mat.T
    t = (-R @ cam_pos).reshape(3, 1)

    # Intrinsics
    fovy = sim.model.cam_fovy[camera_id]
    fy = image_height / (2.0 * np.tan(np.radians(fovy / 2.0)))
    fx = fy  # Square pixels
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)

    return K, R, t


def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Project 3D points to normalized 2D coordinates.

    Args:
        points_3d: (N, 3) array of 3D world points.
        K: 3x3 intrinsic matrix.
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        (N, 2) array of normalized [x, y] coordinates in [0, 1].
    """
    # Transform to camera frame
    points_cam = (R @ points_3d.T + t).T  # (N, 3)

    points_2d = np.zeros((len(points_3d), 2))

    # Only project points in front of camera
    valid = points_cam[:, 2] > 0.01
    if valid.any():
        pts_valid = points_cam[valid]
        projected = (K @ pts_valid.T).T  # (M, 3)
        points_2d[valid, 0] = projected[:, 0] / projected[:, 2]
        points_2d[valid, 1] = projected[:, 1] / projected[:, 2]

    # Normalize to [0, 1]
    points_2d[:, 0] /= image_width
    points_2d[:, 1] /= image_height

    return np.clip(points_2d, 0.0, 1.0)


def compute_2d_bbox(
    corners_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_width: int,
    image_height: int,
) -> List[float]:
    """Compute 2D bounding box from 3D corners.

    Args:
        corners_3d: (8, 3) array of 3D bounding box corners.
        K, R, t: Camera matrices.
        image_width, image_height: Image dimensions.

    Returns:
        [x_min, y_min, x_max, y_max] in normalized [0, 1] coordinates.
    """
    corners_2d = project_points(corners_3d, K, R, t, image_width, image_height)

    x_min = float(corners_2d[:, 0].min())
    y_min = float(corners_2d[:, 1].min())
    x_max = float(corners_2d[:, 0].max())
    y_max = float(corners_2d[:, 1].max())

    return [x_min, y_min, x_max, y_max]
