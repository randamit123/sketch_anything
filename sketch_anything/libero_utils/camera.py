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
    clamp: bool = True,
) -> np.ndarray:
    """Project 3D points to normalized 2D coordinates.

    Args:
        points_3d: (N, 3) array of 3D world points.
        K: 3x3 intrinsic matrix.
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        clamp: If True (default), clamp output to [0, 1]. If False, return
            raw normalized coordinates (may be outside [0, 1] for off-screen
            points). Behind-camera points are always returned as (0, 0).

    Returns:
        (N, 2) array of normalized [x, y] coordinates.
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

    if clamp:
        return np.clip(points_2d, 0.0, 1.0)
    return points_2d


def compute_2d_bbox(
    corners_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[List[float], List[float]]:
    """Compute 2D bounding box and visible centroid from 3D corners.

    The bbox uses clamped projections (existing behaviour). The visible_center
    is the mean of only those projected corners that land within [0, 1] in both
    axes — giving a better estimate of where the object actually appears in the
    frame for large or partially off-screen objects.

    Args:
        corners_3d: (8, 3) array of 3D bounding box corners.
        K, R, t: Camera matrices.
        image_width, image_height: Image dimensions.

    Returns:
        bbox: [x_min, y_min, x_max, y_max] in normalized [0, 1] coordinates.
        visible_center: [cx, cy] centroid of in-frame projected corners, or
            the clamped bbox midpoint if no corners project into the frame.
    """
    # Raw (unclamped) projections to find which corners are truly in-frame
    raw = project_points(corners_3d, K, R, t, image_width, image_height, clamp=False)

    # Filter behind-camera points (they project to (0, 0) which would skew centroid)
    points_cam = (R @ corners_3d.T + t).T
    in_front = points_cam[:, 2] > 0.01
    in_frame = (
        in_front
        & (raw[:, 0] >= 0.0) & (raw[:, 0] <= 1.0)
        & (raw[:, 1] >= 0.0) & (raw[:, 1] <= 1.0)
    )

    if in_frame.any():
        visible_center: List[float] = raw[in_frame].mean(axis=0).tolist()
    else:
        visible_center = None  # will be set to bbox midpoint below

    # Clamped bbox (existing behaviour)
    clamped = np.clip(raw, 0.0, 1.0)
    x_min = float(clamped[:, 0].min())
    y_min = float(clamped[:, 1].min())
    x_max = float(clamped[:, 0].max())
    y_max = float(clamped[:, 1].max())
    bbox = [x_min, y_min, x_max, y_max]

    if visible_center is None:
        visible_center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]

    return bbox, visible_center
