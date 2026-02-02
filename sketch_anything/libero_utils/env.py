"""LIBERO environment helper utilities.

Provides functions for image capture and 3D bounding box extraction from
MuJoCo simulation state.
"""

from __future__ import annotations

import logging
from typing import Set

import numpy as np

logger = logging.getLogger(__name__)


def get_camera_image(env, camera_name: str) -> np.ndarray:
    """Capture an RGB image from a LIBERO environment camera.

    Uses env._get_observations() pattern from the LIBERO render scripts.
    Images come from the observation dict as ``{camera_name}_image``.

    Args:
        env: LIBERO ControlEnv / OffScreenRenderEnv instance.
        camera_name: Camera identifier, e.g. ``"agentview"``.

    Returns:
        RGB image array, shape (H, W, 3), dtype uint8.
    """
    obs = env.env._get_observations()
    key = f"{camera_name}_image"

    if key not in obs:
        available = [k for k in obs if k.endswith("_image")]
        raise KeyError(
            f"Camera image key '{key}' not found. "
            f"Available image keys: {available}"
        )

    image = obs[key]

    # LIBERO/robosuite images may be vertically flipped (origin bottom-left)
    # Flip to standard top-left origin if needed
    if image.ndim == 3:
        image = np.flip(image, axis=0).copy()

    return image


def get_object_bbox_3d(sim, body_name: str) -> np.ndarray:
    """Get approximate 3D bounding box corners for an object.

    Iterates over all geoms attached to the body and computes an
    axis-aligned bounding box in world frame.

    Args:
        sim: MuJoCo simulation object.
        body_name: MuJoCo body name.

    Returns:
        (8, 3) array of corner positions in world coordinates.
    """
    body_id = sim.model.body_name2id(body_name)
    body_pos = sim.data.body_xpos[body_id].copy()
    body_mat = sim.data.body_xmat[body_id].reshape(3, 3).copy()

    # Collect geoms attached to this body
    geom_ids = [
        i for i in range(sim.model.ngeom)
        if sim.model.geom_bodyid[i] == body_id
    ]

    if not geom_ids:
        # Fallback: small default box
        half_size = np.array([0.02, 0.02, 0.02])
        center_offset = np.zeros(3)
    else:
        min_corner = np.full(3, np.inf)
        max_corner = np.full(3, -np.inf)

        for geom_id in geom_ids:
            geom_type = sim.model.geom_type[geom_id]
            geom_size = sim.model.geom_size[geom_id]
            geom_pos = sim.model.geom_pos[geom_id]

            # Approximate half-extents based on geom type
            # MuJoCo geom types: 0=plane, 2=sphere, 3=capsule, 5=cylinder, 6=box
            if geom_type == 6:  # Box
                half = geom_size[:3].copy()
            elif geom_type == 5:  # Cylinder
                half = np.array([geom_size[0], geom_size[0], geom_size[1]])
            elif geom_type == 2:  # Sphere
                half = np.array([geom_size[0]] * 3)
            elif geom_type == 3:  # Capsule
                half = np.array([geom_size[0], geom_size[0], geom_size[0] + geom_size[1]])
            else:
                half = np.array([0.02, 0.02, 0.02])

            local_min = geom_pos - half
            local_max = geom_pos + half
            min_corner = np.minimum(min_corner, local_min)
            max_corner = np.maximum(max_corner, local_max)

        half_size = (max_corner - min_corner) / 2.0
        center_offset = (max_corner + min_corner) / 2.0

    # Compute world-frame center accounting for body orientation
    world_center = body_pos + body_mat @ center_offset

    # Generate 8 corners in local frame, then transform to world
    signs = np.array([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ], dtype=np.float64)

    corners_local = signs * half_size
    corners_world = (body_mat @ corners_local.T).T + world_center

    return corners_world


def list_available_bodies(sim) -> Set[str]:
    """List all body names in the MuJoCo model.

    Args:
        sim: MuJoCo simulation object.

    Returns:
        Set of body name strings.
    """
    bodies: Set[str] = set()
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if name:
            bodies.add(name)
    return bodies
