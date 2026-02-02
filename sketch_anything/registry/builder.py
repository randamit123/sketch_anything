"""Object Registry Builder for LIBERO environments.

Constructs a per-view registry of detected objects with bounding boxes by
extracting 3D poses from MuJoCo and projecting to 2D via camera matrices.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from sketch_anything.libero_utils.camera import (
    compute_2d_bbox,
    get_camera_matrices,
)
from sketch_anything.libero_utils.env import get_object_bbox_3d, list_available_bodies
from sketch_anything.registry.extractor import extract_object_names

logger = logging.getLogger(__name__)

# Type alias for a single view's object registry.
ViewObjectRegistry = Dict[str, dict]

# ---------------------------------------------------------------------------
# Static name mapping for common LIBERO objects.
# Maps natural language name -> list of possible MuJoCo body names to try.
# ---------------------------------------------------------------------------
LIBERO_OBJECT_MAPPING: Dict[str, List[str]] = {
    # Containers
    "bowl": ["bowl", "akita_black_bowl"],
    "blue bowl": ["akita_black_bowl_1", "bowl_blue"],
    "plate": ["plate", "plate_1"],
    "basket": ["basket"],
    "tray": ["tray"],
    # Blocks
    "red block": ["red_block", "cube_red"],
    "blue block": ["blue_block", "cube_blue"],
    "green block": ["green_block", "cube_green"],
    "cube": ["cube", "wooden_cube"],
    # Kitchen items
    "mug": ["mug", "coffee_mug"],
    "cup": ["cup"],
    "butter": ["butter"],
    "cream cheese": ["cream_cheese"],
    "milk": ["milk"],
    "orange juice": ["orange_juice"],
    # Furniture
    "drawer": ["drawer", "top_drawer", "middle_drawer", "bottom_drawer"],
    "top drawer": ["top_drawer"],
    "middle drawer": ["middle_drawer"],
    "bottom drawer": ["bottom_drawer"],
    "cabinet": ["cabinet_door", "cabinet"],
    "door": ["door", "microwave_door"],
    # Appliances
    "microwave": ["microwave"],
    "stove": ["stove", "flat_stove"],
}

# Body names to try for the robot gripper.
GRIPPER_BODY_NAMES = ["robot0_grip_site", "gripper0_grip_site"]


def resolve_object_names(
    task_instruction: str,
    available_bodies: Set[str],
    objects_dict: Optional[dict] = None,
    fixtures_dict: Optional[dict] = None,
) -> Dict[str, str]:
    """Map natural language object names to MuJoCo body names.

    Tries the static mapping first, then falls back to dynamic matching
    against the environment's objects/fixtures dictionaries.

    Args:
        task_instruction: Task description string.
        available_bodies: Set of all MuJoCo body names in the environment.
        objects_dict: env.env.objects_dict (optional, for dynamic fallback).
        fixtures_dict: env.env.fixtures_dict (optional, for dynamic fallback).

    Returns:
        Dict mapping natural_name -> MuJoCo body_name.
    """
    extracted_names = extract_object_names(task_instruction)
    resolved: Dict[str, str] = {}

    for name in extracted_names:
        body = _resolve_single(name, available_bodies, objects_dict, fixtures_dict)
        if body is not None:
            resolved[name] = body
        else:
            logger.warning(f"Could not resolve object '{name}' to a MuJoCo body")

    # Always include gripper
    for gripper_name in GRIPPER_BODY_NAMES:
        if gripper_name in available_bodies:
            resolved["gripper"] = gripper_name
            break
    else:
        # Try site-based lookup as last resort
        logger.warning("Could not find gripper body in environment")

    return resolved


def _resolve_single(
    name: str,
    available_bodies: Set[str],
    objects_dict: Optional[dict],
    fixtures_dict: Optional[dict],
) -> Optional[str]:
    """Resolve a single object name to a MuJoCo body name."""
    name_lower = name.lower().strip()

    # 1. Try static mapping
    if name_lower in LIBERO_OBJECT_MAPPING:
        for candidate in LIBERO_OBJECT_MAPPING[name_lower]:
            if candidate in available_bodies:
                return candidate
            # Partial match
            for body in available_bodies:
                if candidate in body or body in candidate:
                    return body

    # 2. Try direct match against available bodies
    name_underscore = name_lower.replace(" ", "_")
    if name_underscore in available_bodies:
        return name_underscore

    # 3. Try substring match against available bodies
    for body in available_bodies:
        body_lower = body.lower()
        if name_underscore in body_lower or body_lower in name_underscore:
            return body

    # 4. Dynamic fallback: search objects_dict and fixtures_dict
    for d in [objects_dict, fixtures_dict]:
        if d is None:
            continue
        for obj_key, obj_val in d.items():
            key_lower = obj_key.lower()
            if name_lower in key_lower or key_lower in name_lower:
                # Use root_body if available (robosuite object)
                if hasattr(obj_val, "root_body"):
                    return obj_val.root_body
                return obj_key

    return None


def build_object_registry(
    env,
    task_instruction: str,
    camera_names: List[str],
    image_width: int = 256,
    image_height: int = 256,
) -> Dict[str, ViewObjectRegistry]:
    """Build object registries with bounding boxes for each camera view.

    Args:
        env: LIBERO ControlEnv / OffScreenRenderEnv instance.
        task_instruction: Natural language task description.
        camera_names: List of camera names, e.g. ["agentview", "robot0_eye_in_hand"].
        image_width: Width of rendered images.
        image_height: Height of rendered images.

    Returns:
        Dict mapping camera_name -> ViewObjectRegistry.
        Each ViewObjectRegistry maps object_id -> {id, label, bbox, center}.
    """
    sim = env.sim

    available_bodies = list_available_bodies(sim)

    # Get object/fixture dicts if available (LIBERO ControlEnv wraps env.env)
    objects_dict = getattr(env.env, "objects_dict", None) if hasattr(env, "env") else None
    fixtures_dict = getattr(env.env, "fixtures_dict", None) if hasattr(env, "env") else None

    object_mapping = resolve_object_names(
        task_instruction, available_bodies, objects_dict, fixtures_dict
    )
    logger.info(f"Resolved objects: {object_mapping}")

    registries: Dict[str, ViewObjectRegistry] = {}

    for camera_name in camera_names:
        K, R, t = get_camera_matrices(sim, camera_name, image_width, image_height)
        view_registry: ViewObjectRegistry = {}

        for natural_name, body_name in object_mapping.items():
            try:
                corners_3d = get_object_bbox_3d(sim, body_name)
                bbox = compute_2d_bbox(corners_3d, K, R, t, image_width, image_height)
                center = [
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0,
                ]
                obj_id = natural_name.replace(" ", "_")

                view_registry[obj_id] = {
                    "id": obj_id,
                    "label": natural_name,
                    "bbox": bbox,
                    "center": center,
                }
            except Exception as e:
                logger.warning(f"Failed to process '{natural_name}' ({body_name}): {e}")

        registries[camera_name] = view_registry

    return registries
